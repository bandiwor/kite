//! AST -> register bytecode.
//!
//! Every function owns a window of *registers* `R[0..nregs)`.  The analyzer already
//! gave each parameter and local variable a fixed register (`R[0..nslots)`); the
//! compiler hands out the registers above them as temporaries in stack order
//! (`alloc` / `release`).  Instructions name their operands directly
//! (`Add(dst, a, b)`), so locals are used in place and there is no push/pop traffic.
//!
//! Everything is an expression.  [`Compiler::expr`] evaluates a node into a
//! destination ([`Dest::Reg`] or [`Dest::Discard`]), [`Compiler::expr_reg`] returns a
//! register that holds the value (a local's own register when the node is a plain
//! local, otherwise a fresh temporary).
//!
//! Evaluation order is left to right, as if operands were pushed in turn.  Because a
//! local is used in place, `x + { x = 5; 1 }` would see the new `x`; whenever the right
//! operand is not "simple" (see [`Compiler::is_simple`]) the left local is copied first.
//! Likewise, `x = <expr>` evaluates straight into `x`'s register only for expressions
//! whose last instruction is what writes the result ([`Compiler::writes_last`]);
//! constructors, `&&`/`||`, `if`, blocks ... go through a temporary.

use crate::analyzer::{Analysis, FuncInfo, Res};
use crate::ast::*;
use crate::hash::FxHashMap;
use crate::heap::Heap;
use crate::intern::{Interner, Sym};
use crate::object::*;
use crate::op::{Op, I24, MAX_REG, U24};
use crate::peephole;
use crate::source::{Diag, SourceFile};
use crate::text::unescape;
use crate::value::Value;
use std::rc::Rc;

pub struct CompileOptions {
    pub optimize: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions { optimize: true }
    }
}

/// Where the value of an expression goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dest {
    /// The value is not needed (side effects only).
    Discard,
    Reg(u8),
}

struct LoopCtx {
    /// Receives the value of `break value` (and `nil` when the loop ends normally).
    dest: Dest,
    breaks: Vec<usize>,
    continues: Vec<usize>,
    /// Lowest captured register inside the loop; closed on `break`/`continue`.
    close: Option<u16>,
}

struct FnBuilder {
    code: Vec<Op>,
    offsets: Vec<u32>,
    consts: Vec<Value>,
    const_map: FxHashMap<u64, u32>,
    protos: Vec<Gc<Proto>>,
    /// First temporary register (= number of parameter/local registers).
    nslots: usize,
    /// Next free register.
    free: usize,
    /// One past the highest register used.
    max_reg: usize,
    loops: Vec<LoopCtx>,
    is_ctor: bool,
    /// The register budget was exceeded (reported once).
    overflowed: bool,
    /// Per slot: captured by a closure, so any call might change it.
    captured: Vec<bool>,
}

impl FnBuilder {
    fn new(is_ctor: bool, nslots: usize, captured: &[u16]) -> FnBuilder {
        let mut flags = vec![false; nslots];
        for &c in captured {
            flags[c as usize] = true;
        }
        FnBuilder {
            captured: flags,
            code: Vec::new(),
            offsets: Vec::new(),
            consts: Vec::new(),
            const_map: FxHashMap::default(),
            protos: Vec::new(),
            nslots,
            free: nslots,
            max_reg: nslots,
            loops: Vec::new(),
            is_ctor,
            overflowed: false,
        }
    }
}

struct Compiler<'a> {
    scratch: peephole::Scratch,
    /// Interned runtime string per identifier, so field names are hashed once.
    sym_strs: Vec<Option<Gc<Str>>>,
    ast: &'a Ast,
    an: &'a Analysis,
    interner: &'a Interner,
    heap: &'a mut Heap,
    source: &'a Rc<SourceFile>,
    opts: &'a CompileOptions,
    fns: Vec<FnBuilder>,
    errors: Vec<Diag>,
    /// Source offset attributed to emitted instructions.
    cur: u32,
}

pub fn compile(
    ast: &Ast,
    an: &Analysis,
    interner: &Interner,
    heap: &mut Heap,
    source: &Rc<SourceFile>,
    opts: &CompileOptions,
) -> Result<Gc<Proto>, Vec<Diag>> {
    let mut c = Compiler {
        scratch: peephole::Scratch::default(),
        sym_strs: vec![None; interner.len()],
        ast,
        an,
        interner,
        heap,
        source,
        opts,
        fns: Vec::new(),
        errors: Vec::new(),
        cur: 0,
    };
    let root = ast.root.expect("parsed program");
    c.fns.push(FnBuilder::new(false, an.main.nslots as usize, &an.main.captured));
    if an.module {
        c.module_root(root);
    } else {
        c.body(root, false);
    }
    let fb = c.fns.pop().unwrap();
    let proto = c.finish(fb, "<main>".into(), 0, &an.main);
    if c.errors.is_empty() {
        Ok(proto)
    } else {
        Err(c.errors)
    }
}

fn is_cmp(op: BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
}

/// `R[a] = R[b] op R[c]`
fn arith_op(op: BinOp, a: u8, b: u8, c: u8) -> Op {
    match op {
        BinOp::Add => Op::Add(a, b, c),
        BinOp::Sub => Op::Sub(a, b, c),
        BinOp::Mul => Op::Mul(a, b, c),
        BinOp::Div => Op::Div(a, b, c),
        BinOp::Rem => Op::Rem(a, b, c),
        BinOp::Pow => Op::Pow(a, b, c),
        BinOp::BitAnd => Op::BitAnd(a, b, c),
        BinOp::BitOr => Op::BitOr(a, b, c),
        BinOp::BitXor => Op::BitXor(a, b, c),
        BinOp::Shl => Op::Shl(a, b, c),
        BinOp::Shr => Op::Shr(a, b, c),
        BinOp::Eq => Op::Eq(a, b, c),
        BinOp::Ne => Op::Ne(a, b, c),
        BinOp::Lt => Op::Lt(a, b, c),
        BinOp::Le => Op::Le(a, b, c),
        BinOp::Gt => Op::Gt(a, b, c),
        BinOp::Ge => Op::Ge(a, b, c),
        BinOp::And | BinOp::Or => unreachable!("short-circuit ops are control flow"),
    }
}

/// `R[a] = R[b] op K[k]` for the operators that have a constant form.
fn arith_op_k(op: BinOp, a: u8, b: u8, k: u8) -> Option<Op> {
    Some(match op {
        BinOp::Add => Op::AddK(a, b, k),
        BinOp::Sub => Op::SubK(a, b, k),
        BinOp::Mul => Op::MulK(a, b, k),
        BinOp::Div => Op::DivK(a, b, k),
        BinOp::Rem => Op::RemK(a, b, k),
        _ => return None,
    })
}

/// Branch taken when `a op b` is true (`jump_if`) or false.
fn jcmp(op: BinOp, jump_if: bool, a: u8, b: u8) -> Op {
    match (op, jump_if) {
        (BinOp::Eq, true) | (BinOp::Ne, false) => Op::JEq(a, b),
        (BinOp::Ne, true) | (BinOp::Eq, false) => Op::JNe(a, b),
        (BinOp::Lt, true) => Op::JLt(a, b),
        (BinOp::Lt, false) => Op::JNotLt(a, b),
        (BinOp::Le, true) => Op::JLe(a, b),
        (BinOp::Le, false) => Op::JNotLe(a, b),
        (BinOp::Gt, true) => Op::JGt(a, b),
        (BinOp::Gt, false) => Op::JNotGt(a, b),
        (BinOp::Ge, true) => Op::JGe(a, b),
        (BinOp::Ge, false) => Op::JNotGe(a, b),
        _ => unreachable!("not a comparison"),
    }
}

fn jcmp_k(op: BinOp, jump_if: bool, a: u8, k: u16) -> Op {
    match (op, jump_if) {
        (BinOp::Eq, true) | (BinOp::Ne, false) => Op::JEqK(a, k),
        (BinOp::Ne, true) | (BinOp::Eq, false) => Op::JNeK(a, k),
        (BinOp::Lt, true) => Op::JLtK(a, k),
        (BinOp::Lt, false) => Op::JNotLtK(a, k),
        (BinOp::Le, true) => Op::JLeK(a, k),
        (BinOp::Le, false) => Op::JNotLeK(a, k),
        (BinOp::Gt, true) => Op::JGtK(a, k),
        (BinOp::Gt, false) => Op::JNotGtK(a, k),
        (BinOp::Ge, true) => Op::JGeK(a, k),
        (BinOp::Ge, false) => Op::JNotGeK(a, k),
        _ => unreachable!("not a comparison"),
    }
}

impl<'a> Compiler<'a> {
    // ---- emission helpers ------------------------------------------------------

    fn fb(&mut self) -> &mut FnBuilder {
        self.fns.last_mut().unwrap()
    }

    fn error(&mut self, offset: u32, msg: impl Into<String>) {
        if self.errors.len() < 20 {
            self.errors.push(Diag::new(msg, offset, 1));
        }
    }

    fn at(&mut self, id: NodeId) {
        self.cur = self.ast.span(id);
    }

    fn emit(&mut self, op: Op) -> usize {
        let cur = self.cur;
        let fb = self.fb();
        fb.code.push(op);
        fb.offsets.push(cur);
        fb.code.len() - 1
    }

    fn here(&mut self) -> usize {
        self.fb().code.len()
    }

    /// Emit a double instruction whose `Ext` will hold a branch offset (patched later
    /// with `patch_to`).  Returns the op's index.
    fn emit_wide(&mut self, op: Op) -> usize {
        debug_assert!(op.is_branch_wide());
        let at = self.emit(op);
        let cur = self.cur;
        let fb = self.fb();
        fb.code.push(Op::Ext(I24::ZERO));
        fb.offsets.push(cur);
        at
    }

    /// Emit a double instruction whose `Ext` carries a 24-bit payload.
    fn emit_ext(&mut self, op: Op, payload: u32) {
        debug_assert!(op.is_wide() && !op.is_branch_wide());
        let ext = I24::new(((payload << 8) as i32) >> 8).expect("24-bit payload");
        self.emit(op);
        let cur = self.cur;
        let fb = self.fb();
        fb.code.push(Op::Ext(ext));
        fb.offsets.push(cur);
    }

    /// Point the branch at `at` to the instruction index `target`.
    fn patch_to(&mut self, at: usize, target: usize) {
        let wide = self.fb().code[at].is_branch_wide();
        let rel = target as i64 - (at as i64 + 1 + wide as i64);
        let off = i32::try_from(rel).ok().and_then(I24::new);
        let Some(off) = off else {
            self.error(self.cur, "function is too large (a jump spans more than 8M instructions)");
            return;
        };
        let fb = self.fb();
        if wide {
            fb.code[at + 1] = Op::Ext(off);
        } else {
            let op = fb.code[at];
            fb.code[at] = op.with_jump_offset(off.get()).expect("jump patch");
        }
    }

    fn patch_here(&mut self, at: usize) {
        let h = self.here();
        self.patch_to(at, h);
    }

    fn patch_all_here(&mut self, list: Vec<usize>) {
        for j in list {
            self.patch_here(j);
        }
    }

    fn emit_jump(&mut self) -> usize {
        self.emit(Op::Jump(I24::ZERO))
    }

    fn emit_jump_to(&mut self, target: usize) {
        let at = self.emit_jump();
        self.patch_to(at, target);
    }

    // ---- registers ---------------------------------------------------------------

    /// A fresh temporary register.
    fn alloc(&mut self) -> u8 {
        let fb = self.fb();
        let r = fb.free;
        if r >= MAX_REG {
            let first = !fb.overflowed;
            fb.overflowed = true;
            if first {
                self.error(
                    self.cur,
                    format!("function needs more than {} registers; split it or simplify the expression", MAX_REG),
                );
            }
            return 0;
        }
        fb.free = r + 1;
        if fb.free > fb.max_reg {
            fb.max_reg = fb.free;
        }
        r as u8
    }

    fn mark(&mut self) -> usize {
        self.fb().free
    }

    fn release(&mut self, mark: usize) {
        self.fb().free = mark;
    }

    fn is_top_temp(&mut self, r: u8) -> bool {
        let fb = self.fb();
        r as usize >= fb.nslots && r as usize + 1 == fb.free
    }

    // ---- constants ---------------------------------------------------------------

    fn add_const(&mut self, v: Value) -> u32 {
        let fb = self.fb();
        if let Some(&i) = fb.const_map.get(&v.bits()) {
            return i;
        }
        let i = fb.consts.len() as u32;
        fb.consts.push(v);
        fb.const_map.insert(v.bits(), i);
        if i > U24::MAX {
            self.error(self.cur, "too many constants in one function (limit is 16M)");
            return 0;
        }
        i
    }

    fn num_const(&mut self, n: f64) -> u32 {
        self.add_const(Value::num(n))
    }

    fn str_const(&mut self, s: &str) -> u32 {
        let g = self.heap.intern(s);
        self.add_const(Value::from(g))
    }

    fn sym_const(&mut self, s: Sym) -> u32 {
        let g = match self.sym_strs[s.idx()] {
            Some(g) => g,
            None => {
                let g = self.heap.intern(self.interner.get(s));
                self.sym_strs[s.idx()] = Some(g);
                g
            }
        };
        self.add_const(Value::from(g))
    }

    fn k16(&mut self, k: u32, what: &str) -> u16 {
        match u16::try_from(k) {
            Ok(v) => v,
            Err(_) => {
                self.error(self.cur, format!("too many constants in this function to compile {}", what));
                0
            }
        }
    }

    fn add_proto(&mut self, p: Gc<Proto>) -> u16 {
        let fb = self.fb();
        fb.protos.push(p);
        let i = fb.protos.len() - 1;
        match u16::try_from(i) {
            Ok(v) => v,
            Err(_) => {
                self.error(self.cur, "too many nested functions in one function (limit is 65535)");
                0
            }
        }
    }

    /// `R[dst] = K[k]`
    fn load_const(&mut self, dst: u8, k: u32) {
        match u16::try_from(k) {
            Ok(k) => {
                self.emit(Op::LoadK(dst, k));
            }
            Err(_) => self.emit_ext(Op::LoadKW(dst), k),
        }
    }

    fn load_num(&mut self, dst: u8, n: f64) {
        // small integers need no constant (but keep -0.0 exact)
        if n.fract() == 0.0 && (-32768.0..=32767.0).contains(&n) && !(n == 0.0 && n.is_sign_negative()) {
            self.emit(Op::LoadI(dst, n as i16));
        } else {
            let k = self.num_const(n);
            self.load_const(dst, k);
        }
    }

    fn load_nil(&mut self, dst: u8) {
        self.emit(Op::LoadNil(dst, 0));
    }

    // ---- constant folding --------------------------------------------------------

    fn const_num(&self, id: NodeId, depth: u32) -> Option<f64> {
        if depth > 8 {
            return None;
        }
        match self.ast.node(id) {
            Node::Num(n) => Some(n),
            Node::Unary { op: UnOp::Neg, expr } => self.const_num(expr, depth + 1).map(|n| -n),
            Node::Binary { op, lhs, rhs } => {
                let a = self.const_num(lhs, depth + 1)?;
                let b = self.const_num(rhs, depth + 1)?;
                Some(match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Rem => a % b,
                    BinOp::Pow => a.powf(b),
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// A literal that can be a constant-table operand of a comparison.
    fn literal_const(&mut self, id: NodeId) -> Option<u32> {
        if let Some(n) = self.const_num(id, 0) {
            return Some(self.num_const(n));
        }
        match self.ast.node(id) {
            Node::Str { start, len } => {
                let raw = &self.source.text[start as usize..(start + len) as usize];
                let s = unescape(raw).unwrap_or_default();
                Some(self.str_const(&s))
            }
            Node::Nil => Some(self.add_const(Value::NIL)),
            Node::True => Some(self.add_const(Value::TRUE)),
            Node::False => Some(self.add_const(Value::FALSE)),
            _ => None,
        }
    }

    // ---- evaluation-order helpers ---------------------------------------------------

    /// Is `id` free of anything that could assign to the local being read around it?
    /// Blocks and assignments might; calls might only if a closure captured the local
    /// (`calls` says that evaluating calls is harmless).  Hooks reached through fields,
    /// indexes and operators are deliberately not counted.
    fn is_simple(&self, id: NodeId, depth: u32, calls: bool) -> bool {
        if depth > 6 {
            return false;
        }
        let d = depth + 1;
        match self.ast.node(id) {
            Node::Nil | Node::True | Node::False | Node::Num(_) | Node::Str { .. } | Node::Key(_) | Node::Ident(_) => {
                true
            }
            Node::Unary { expr, .. } => self.is_simple(expr, d, calls),
            Node::Binary { lhs, rhs, .. } => self.is_simple(lhs, d, calls) && self.is_simple(rhs, d, calls),
            Node::Field { obj, .. } => self.is_simple(obj, d, calls),
            Node::Index { obj, index } => self.is_simple(obj, d, calls) && self.is_simple(index, d, calls),
            Node::Call { callee, args } if calls => {
                self.is_simple(callee, d, calls) && self.ast.list(args).iter().all(|&x| self.is_simple(NodeId(x), d, calls))
            }
            Node::MethodCall { args, .. } if calls => {
                self.ast.list(args).iter().all(|&x| self.is_simple(NodeId(x), d, calls))
            }
            _ => false,
        }
    }

    /// Can `r` (a local's register) be left in place while `rest` is evaluated?
    fn stays_put(&self, r: u8, rest: impl Iterator<Item = NodeId>) -> bool {
        let calls = !self.fns.last().unwrap().captured.get(r as usize).copied().unwrap_or(true);
        rest.into_iter().all(|n| n.is_none() || self.is_simple(n, 0, calls))
    }

    /// Does the node compute its value in one final instruction that writes the destination?
    /// (Only then may it be evaluated straight into a variable that it also reads.)
    fn writes_last(&self, id: NodeId) -> bool {
        match self.ast.node(id) {
            Node::Nil
            | Node::True
            | Node::False
            | Node::Num(_)
            | Node::Str { .. }
            | Node::Key(_)
            | Node::Ident(_)
            | Node::Unary { .. }
            | Node::Field { .. }
            | Node::Index { .. }
            | Node::Call { .. }
            | Node::MethodCall { .. }
            | Node::Func(_) => true,
            Node::Binary { op, .. } => !matches!(op, BinOp::And | BinOp::Or),
            _ => false,
        }
    }

    /// The register of `id` if it is a plain local variable.
    fn local_reg(&self, id: NodeId) -> Option<u8> {
        if let Node::Ident(_) = self.ast.node(id) {
            if let Res::Local(s) = self.an.res[id.idx()] {
                return Some(s as u8);
            }
        }
        None
    }

    /// Register for the left operand `lhs` of something whose other operands `rest` are
    /// evaluated afterwards: a local is used in place unless `rest` could change it.
    fn operand(&mut self, lhs: NodeId, rest: &[NodeId]) -> u8 {
        if let Some(r) = self.local_reg(lhs) {
            if self.stays_put(r, rest.iter().copied()) {
                return r;
            }
            let t = self.alloc();
            self.emit(Op::Move(t, r));
            return t;
        }
        self.expr_reg(lhs)
    }

    // ---- functions ---------------------------------------------------------------

    fn function(&mut self, fid: FuncId, is_ctor: bool) -> Gc<Proto> {
        let def = self.ast.func(fid);
        let info = &self.an.funcs[fid.0 as usize];
        let arity = def.params.len;
        if arity as usize >= MAX_REG - 1 {
            let off = self.ast.span(def.body);
            self.error(off, format!("too many parameters (max {})", MAX_REG - 2));
        }
        self.fns.push(FnBuilder::new(is_ctor, info.nslots as usize, &info.captured));
        self.body(def.body, is_ctor);
        let fb = self.fns.pop().unwrap();
        let name: Rc<str> = if def.name.is_none() { Rc::from("<anon>") } else { Rc::from(self.interner.get(def.name)) };
        self.finish(fb, name, arity.min(255) as u8, info)
    }

    /// The body of a function (or of a script): evaluates it and returns the value of
    /// its last expression.  Constructors return `self`.
    fn body(&mut self, body: NodeId, is_ctor: bool) {
        if is_ctor {
            let m = self.mark();
            self.expr(body, Dest::Discard);
            self.release(m);
            self.cur = self.ast.span(body);
            self.emit(Op::Return(0));
            return;
        }
        self.tail_return(body);
    }

    /// Evaluate `id` and return its value.  `if`s and blocks in tail position return from
    /// each branch directly instead of joining; a block needs no `CloseUpvals` since
    /// returning closes everything.
    fn tail_return(&mut self, id: NodeId) {
        match self.ast.node(id) {
            Node::Block { stmts, tail } => {
                for s in self.ast.nodes_of(stmts) {
                    self.stmt(s);
                }
                if tail.is_some() {
                    self.tail_return(tail);
                } else {
                    self.cur = self.ast.span(id);
                    self.emit(Op::ReturnNil);
                }
            }
            Node::If { cond, then, els } => {
                let jf = self.cond_jump(cond, false);
                self.tail_return(then);
                self.patch_all_here(jf);
                if els.is_some() {
                    self.tail_return(els);
                } else {
                    self.at(id);
                    self.emit(Op::ReturnNil);
                }
            }
            _ => {
                let m = self.mark();
                let r = self.expr_reg(id);
                self.at(id);
                self.emit(Op::Return(r));
                self.release(m);
            }
        }
    }

    fn finish(&mut self, fb: FnBuilder, name: Rc<str>, arity: u8, info: &FuncInfo) -> Gc<Proto> {
        let FnBuilder { mut code, offsets, consts, protos, max_reg, .. } = fb;
        // `return f(x)` reuses the frame: `Call(a, n); Return(a)` => `TailCall(a, n); Return(a)`
        for i in 1..code.len() {
            if let (Op::Call(a, n), Op::Return(r)) = (code[i - 1], code[i]) {
                if a == r {
                    code[i - 1] = Op::TailCall(a, n);
                }
            }
        }
        let (code, offsets) = if self.opts.optimize {
            match peephole::optimize_with(&mut self.scratch, code.clone(), offsets.clone()) {
                Ok(r) => r,
                Err(msg) => {
                    self.error(self.cur, msg);
                    (code, offsets)
                }
            }
        } else {
            (code, offsets)
        };
        debug_assert_eq!(code.len(), offsets.len());
        debug_assert!(crate::op::verify(&code).is_ok(), "{:?} in `{}`", crate::op::verify(&code), name);
        let nslots = info.nslots as usize;
        let proto = Proto {
            hdr: Header::new(Kind::Proto),
            name,
            arity,
            nslots: info.nslots,
            nregs: max_reg.max(nslots).max(arity as usize) as u16,
            upvals: info.upvals.clone(),
            code,
            consts,
            protos,
            offsets,
            source: self.source.clone(),
        };
        self.heap.alloc(proto)
    }

    // ---- statements ----------------------------------------------------------------

    /// A module evaluates its statements and then returns the table of its exports.
    fn module_root(&mut self, root: NodeId) {
        let Node::Block { stmts, tail } = self.ast.node(root) else { unreachable!() };
        for s in self.ast.nodes_of(stmts) {
            self.stmt(s);
        }
        if tail.is_some() {
            let m = self.mark();
            self.expr(tail, Dest::Discard);
            self.release(m);
        }
        self.cur = 0;
        let t = self.alloc();
        self.emit(Op::NewTable(t, self.an.exports.len().min(u16::MAX as usize) as u16));
        for &(name, res) in &self.an.exports {
            let m = self.mark();
            let v = self.var_reg(res, name);
            let k = self.sym_const(name);
            self.init_field(t, k, v);
            self.release(m);
        }
        self.emit(Op::Return(t));
    }

    /// Bind the value of register `src` to the declared variable.
    fn store_decl(&mut self, res: Res, name: Sym, src: u8) {
        match res {
            Res::Local(s) => {
                if s as u8 != src {
                    self.emit(Op::Move(s as u8, src));
                }
            }
            Res::Global => self.def_global(src, name),
            _ => unreachable!("declaration resolved to {:?}", res),
        }
    }

    fn def_global(&mut self, src: u8, name: Sym) {
        match u16::try_from(name.0) {
            Ok(s) => {
                self.emit(Op::DefGlobal(src, s));
            }
            Err(_) => self.emit_ext(Op::DefGlobalW(src), name.0),
        }
    }

    /// Register that receives a declaration's value: the variable itself for locals.
    fn decl_dest(&mut self, res: Res) -> (u8, bool) {
        match res {
            Res::Local(s) => (s as u8, true),
            _ => (self.alloc(), false),
        }
    }

    fn stmt(&mut self, id: NodeId) {
        let m = self.mark();
        self.at(id);
        match self.ast.node(id) {
            Node::Nil => {} // empty statement
            Node::Var { name, init } | Node::Const { name, init } => {
                let res = self.an.res[id.idx()];
                let (d, direct) = self.decl_dest(res);
                if init.is_some() {
                    self.expr(init, Dest::Reg(d));
                } else {
                    self.load_nil(d);
                }
                self.at(id);
                if !direct {
                    self.store_decl(res, name, d);
                }
            }
            Node::Struct { name, fields } => {
                let sname = self.heap.intern(self.interner.get(name));
                let fs: Vec<Gc<Str>> =
                    self.ast.syms_of(fields).map(|f| self.heap.intern(self.interner.get(f))).collect();
                let ty = self.heap.alloc(StructType {
                    hdr: Header::new(Kind::StructType),
                    name: sname,
                    variants: Vec::new(),
                    fields: fs,
                    methods: Vec::new(),
                    hooks: [Value::NIL; HOOK_COUNT],
                });
                self.declare_type(id, name, ty);
            }
            Node::Union { name, variants } => {
                let uname = self.heap.intern(self.interner.get(name));
                let vs: Vec<Variant> = self
                    .ast
                    .list(variants)
                    .iter()
                    .map(|&v| {
                        let d = &self.ast.variants[v as usize];
                        Variant {
                            name: self.heap.intern(self.interner.get(d.name)),
                            fields: self
                                .ast
                                .syms_of(d.fields)
                                .map(|f| self.heap.intern(self.interner.get(f)))
                                .collect(),
                            value: Value::NIL,
                        }
                    })
                    .collect();
                let ty = self.heap.alloc(StructType {
                    hdr: Header::new(Kind::StructType),
                    name: uname,
                    fields: Vec::new(),
                    variants: vs,
                    methods: Vec::new(),
                    hooks: [Value::NIL; HOOK_COUNT],
                });
                self.declare_type(id, name, ty);
            }
            Node::Export(inner) => self.stmt(inner),
            Node::Import(iid) => {
                let d = self.ast.import_def(iid);
                let raw = &self.source.text[d.path_start as usize..(d.path_start + d.path_len) as usize];
                let path = unescape(raw).unwrap_or_default();
                let k = self.str_const(&path);
                let k = self.k16(k, "an import");
                let module = self.alloc();
                self.emit(Op::Import(module, k));
                if d.module_name.is_none() {
                    // import { a, b } from "path"
                    for (j, name) in self.ast.syms_of(d.names).enumerate() {
                        let res = self.an.import_res[iid.0 as usize][j];
                        let (dst, direct) = self.decl_dest(res);
                        let nk = self.sym_const(name);
                        self.emit_ext(Op::GetExport(dst, module), nk);
                        if !direct {
                            self.store_decl(res, name, dst);
                        }
                    }
                } else {
                    let res = self.an.import_res[iid.0 as usize][0];
                    self.store_decl(res, d.module_name, module);
                }
            }
            Node::Impl { target, methods } => {
                let ty = self.expr_reg(target);
                for m in self.ast.nodes_of(methods) {
                    let Node::Func(fid) = self.ast.node(m) else { unreachable!() };
                    let name = self.ast.func(fid).name;
                    let is_ctor = self.interner.get(name) == "new";
                    let proto = self.function(fid, is_ctor);
                    let p = self.add_proto(proto);
                    self.at(m);
                    let f = self.alloc();
                    self.emit(Op::Closure(f, p));
                    let nk = self.sym_const(name);
                    self.emit_ext(Op::DefMethod(ty, f), nk);
                    self.release(f as usize);
                }
            }
            _ => self.expr(id, Dest::Discard),
        }
        self.release(m);
    }

    /// `struct`/`union` declaration: instantiate the template type into the declared name.
    fn declare_type(&mut self, id: NodeId, name: Sym, ty: Gc<StructType>) {
        let k = self.add_const(Value::from(ty));
        let k = self.k16(k, "a type declaration");
        let res = self.an.res[id.idx()];
        let (d, direct) = self.decl_dest(res);
        self.emit(Op::Struct(d, k));
        if !direct {
            self.store_decl(res, name, d);
        }
    }

    // ---- conditions ----------------------------------------------------------------

    /// Emit code that jumps (the returned patch list) when the truthiness of `id` equals
    /// `jump_if`, and falls through otherwise.  `&&`, `||`, `!` and comparisons compile to
    /// pure control flow.
    fn cond_jump(&mut self, id: NodeId, jump_if: bool) -> Vec<usize> {
        match self.ast.node(id) {
            Node::Unary { op: UnOp::Not, expr } => self.cond_jump(expr, !jump_if),
            Node::Binary { op: BinOp::And, lhs, rhs } => {
                if jump_if {
                    let jf = self.cond_jump(lhs, false);
                    let jt = self.cond_jump(rhs, true);
                    self.patch_all_here(jf);
                    jt
                } else {
                    let mut j = self.cond_jump(lhs, false);
                    j.extend(self.cond_jump(rhs, false));
                    j
                }
            }
            Node::Binary { op: BinOp::Or, lhs, rhs } => {
                if jump_if {
                    let mut j = self.cond_jump(lhs, true);
                    j.extend(self.cond_jump(rhs, true));
                    j
                } else {
                    let jt = self.cond_jump(lhs, true);
                    let jf = self.cond_jump(rhs, false);
                    self.patch_all_here(jt);
                    jf
                }
            }
            Node::Binary { op, lhs, rhs } if is_cmp(op) => {
                let m = self.mark();
                let l = self.operand(lhs, &[rhs]);
                let lit = self.literal_const(rhs).and_then(|k| u16::try_from(k).ok());
                self.at(id);
                let j = match lit {
                    Some(k) => self.emit_wide(jcmp_k(op, jump_if, l, k)),
                    None => {
                        let r = self.expr_reg(rhs);
                        self.at(id);
                        self.emit_wide(jcmp(op, jump_if, l, r))
                    }
                };
                self.release(m);
                vec![j]
            }
            _ => {
                let m = self.mark();
                let r = self.expr_reg(id);
                self.at(id);
                let j = self.emit_wide(if jump_if { Op::JTrue(r) } else { Op::JFalse(r) });
                self.release(m);
                vec![j]
            }
        }
    }

    // ---- expressions ---------------------------------------------------------------

    /// A register that holds the value of `id`: the local's own register for a plain
    /// local, otherwise a fresh temporary (which the caller releases via `release`).
    fn expr_reg(&mut self, id: NodeId) -> u8 {
        if let Some(r) = self.local_reg(id) {
            return r;
        }
        let r = self.alloc();
        self.expr(id, Dest::Reg(r));
        r
    }

    /// The register that will receive a value: `dest`, or a scratch temporary when the
    /// value is only wanted for its side effects.
    fn target(&mut self, dest: Dest) -> u8 {
        match dest {
            Dest::Reg(d) => d,
            Dest::Discard => self.alloc(),
        }
    }

    /// `R[dst] = value of the variable`
    fn get_var(&mut self, res: Res, sym: Sym, dst: u8) {
        match res {
            Res::Local(s) => {
                if s as u8 != dst {
                    self.emit(Op::Move(dst, s as u8));
                }
            }
            Res::Upvalue(i) => {
                let i = self.upval_index(i);
                self.emit(Op::GetUpval(dst, i));
            }
            Res::Global => match u16::try_from(sym.0) {
                Ok(s) => {
                    self.emit(Op::GetGlobal(dst, s));
                }
                Err(_) => self.emit_ext(Op::GetGlobalW(dst), sym.0),
            },
            Res::None => unreachable!("unresolved identifier"),
        }
    }

    /// Register holding the variable (its own register for a local).
    fn var_reg(&mut self, res: Res, sym: Sym) -> u8 {
        if let Res::Local(s) = res {
            return s as u8;
        }
        let t = self.alloc();
        self.get_var(res, sym, t);
        t
    }

    fn upval_index(&mut self, i: u16) -> u8 {
        match u8::try_from(i) {
            Ok(v) => v,
            Err(_) => {
                self.error(self.cur, "too many captured variables in one function (limit is 255)");
                0
            }
        }
    }

    /// Store the value of register `src` into an existing variable.
    fn set_var(&mut self, res: Res, sym: Sym, src: u8) {
        match res {
            Res::Local(s) => {
                if s as u8 != src {
                    self.emit(Op::Move(s as u8, src));
                }
            }
            Res::Upvalue(i) => {
                let i = self.upval_index(i);
                self.emit(Op::SetUpval(src, i));
            }
            Res::Global => match u16::try_from(sym.0) {
                Ok(s) => {
                    self.emit(Op::SetGlobal(src, s));
                }
                Err(_) => self.emit_ext(Op::SetGlobalW(src), sym.0),
            },
            Res::None => unreachable!("unresolved identifier"),
        }
    }

    fn init_field(&mut self, table: u8, k: u32, v: u8) {
        match u8::try_from(k) {
            Ok(k) => {
                self.emit(Op::InitField(table, k, v));
            }
            Err(_) => self.emit_ext(Op::InitFieldW(table, v), k),
        }
    }

    fn get_field_op(&mut self, dst: u8, obj: u8, k: u32) {
        match u8::try_from(k) {
            Ok(k) => {
                self.emit(Op::GetField(dst, obj, k));
            }
            Err(_) => self.emit_ext(Op::GetFieldW(dst, obj), k),
        }
    }

    fn set_field_op(&mut self, obj: u8, k: u32, v: u8) {
        match u8::try_from(k) {
            Ok(k) => {
                self.emit(Op::SetField(obj, k, v));
            }
            Err(_) => self.emit_ext(Op::SetFieldW(obj, v), k),
        }
    }

    /// `R[dst] = value of expr`, or evaluate it for its effects (`Dest::Discard`).
    fn expr(&mut self, id: NodeId, dest: Dest) {
        self.at(id);
        let node = self.ast.node(id);
        match node {
            Node::Nil => {
                if let Dest::Reg(d) = dest {
                    self.load_nil(d);
                }
            }
            Node::True => {
                if let Dest::Reg(d) = dest {
                    self.emit(Op::LoadTrue(d));
                }
            }
            Node::False => {
                if let Dest::Reg(d) = dest {
                    self.emit(Op::LoadFalse(d));
                }
            }
            Node::Num(n) => {
                if let Dest::Reg(d) = dest {
                    self.load_num(d, n);
                }
            }
            Node::Str { start, len } => {
                if let Dest::Reg(d) = dest {
                    let raw = &self.source.text[start as usize..(start + len) as usize];
                    let s = unescape(raw).unwrap_or_default();
                    let k = self.str_const(&s);
                    self.load_const(d, k);
                }
            }
            Node::Key(sym) => {
                if let Dest::Reg(d) = dest {
                    let k = self.sym_const(sym);
                    self.load_const(d, k);
                }
            }
            Node::Ident(sym) => {
                if let Dest::Reg(d) = dest {
                    let res = self.an.res[id.idx()];
                    self.get_var(res, sym, d);
                }
            }
            Node::Unary { op, expr } => {
                if op == UnOp::Neg {
                    if let Some(n) = self.const_num(id, 0) {
                        if let Dest::Reg(d) = dest {
                            self.load_num(d, n);
                        }
                        return;
                    }
                }
                let m = self.mark();
                let t = self.target(dest);
                let r = self.expr_reg(expr);
                self.at(id);
                self.emit(match op {
                    UnOp::Neg => Op::Neg(t, r),
                    UnOp::Not => Op::Not(t, r),
                    UnOp::BitNot => Op::BitNot(t, r),
                });
                self.release(m);
            }
            Node::Binary { op: op @ (BinOp::And | BinOp::Or), lhs, rhs } => {
                let m = self.mark();
                let t = self.target(dest);
                self.expr(lhs, Dest::Reg(t));
                self.at(id);
                let j = self.emit_wide(if op == BinOp::And { Op::JFalse(t) } else { Op::JTrue(t) });
                self.expr(rhs, Dest::Reg(t));
                self.patch_here(j);
                self.release(m);
            }
            Node::Binary { op, lhs, rhs } => {
                if let Some(n) = self.const_num(id, 0) {
                    if let Dest::Reg(d) = dest {
                        self.load_num(d, n);
                    }
                    return;
                }
                let m = self.mark();
                let t = self.target(dest);
                let l = self.operand(lhs, &[rhs]);
                self.arith(id, op, t, l, rhs);
                self.release(m);
            }
            Node::Assign { op, target, value } => self.assign(id, op, target, value, dest),
            Node::Call { callee, args } => {
                let m = self.mark();
                let a = self.call_slot(dest);
                self.expr(callee, Dest::Reg(a));
                let n = args.len;
                for arg in self.ast.nodes_of(args) {
                    let r = self.alloc();
                    self.expr(arg, Dest::Reg(r));
                }
                self.finish_call(id, a, n as usize, dest);
                self.release(m);
            }
            Node::MethodCall { name, args } => {
                let m = self.mark();
                let a = self.call_slot(dest);
                let list = self.ast.list(args);
                let recv = NodeId(list[0]);
                let n = list.len();
                let obj_slot = self.alloc(); // a + 1: receives the receiver
                debug_assert_eq!(obj_slot as usize, a as usize + 1);
                let rest = &list[1..];
                let b = if let Some(r) = self.local_reg(recv) {
                    // use the local in place unless the arguments could change it
                    if self.stays_put(r, rest.iter().map(|&x| NodeId(x))) {
                        r
                    } else {
                        self.emit(Op::Move(obj_slot, r));
                        obj_slot
                    }
                } else {
                    self.expr(recv, Dest::Reg(obj_slot));
                    obj_slot
                };
                self.at(id);
                let k = self.sym_const(name);
                match u8::try_from(k) {
                    Ok(k) => {
                        self.emit(Op::GetMethod(a, b, k));
                    }
                    Err(_) => self.emit_ext(Op::GetMethodW(a, b), k),
                }
                for &arg in rest {
                    let r = self.alloc();
                    self.expr(NodeId(arg), Dest::Reg(r));
                }
                self.finish_call(id, a, n, dest);
                self.release(m);
            }
            Node::Field { obj, name } => {
                let m = self.mark();
                let t = self.target(dest);
                let o = self.expr_reg(obj);
                self.at(id);
                let k = self.sym_const(name);
                self.get_field_op(t, o, k);
                self.release(m);
            }
            Node::Index { obj, index } => {
                let m = self.mark();
                let t = self.target(dest);
                let o = self.operand(obj, &[index]);
                let i = self.expr_reg(index);
                self.at(id);
                self.emit(Op::GetIndex(t, o, i));
                self.release(m);
            }
            Node::Array { items } => {
                let m = self.mark();
                let t = self.target(dest);
                // build in a scratch register when the destination is read by the items
                // (callers guarantee that `dest` is private, see `writes_last`)
                self.emit(Op::NewArray(t, items.len.min(u16::MAX as u32) as u16));
                let list = self.ast.list(items);
                // items are evaluated into consecutive temporaries and appended in batches
                for chunk in list.chunks(32) {
                    let first = self.mark();
                    for &item in chunk {
                        let r = self.alloc();
                        self.expr(NodeId(item), Dest::Reg(r));
                    }
                    self.at(id);
                    self.emit(Op::AppendN(t, first as u8, chunk.len() as u8));
                    self.release(first);
                }
                self.release(m);
            }
            Node::Table { entries } => {
                let m = self.mark();
                let t = self.target(dest);
                self.emit(Op::NewTable(t, (entries.len / 2).min(u16::MAX as u32) as u16));
                let list = self.ast.list(entries);
                for pair in list.chunks(2) {
                    let (k, v) = (NodeId(pair[0]), NodeId(pair[1]));
                    let pm = self.mark();
                    if let Node::Key(sym) = self.ast.node(k) {
                        let r = self.expr_reg(v);
                        self.at(k);
                        let c = self.sym_const(sym);
                        self.init_field(t, c, r);
                    } else {
                        let kr = self.expr_reg(k);
                        let r = self.expr_reg(v);
                        self.at(k);
                        self.emit(Op::InitIndex(t, kr, r));
                    }
                    self.release(pm);
                }
                self.release(m);
            }
            Node::Func(fid) => {
                let m = self.mark();
                let t = self.target(dest);
                let proto = self.function(fid, false);
                let p = self.add_proto(proto);
                self.at(id);
                self.emit(Op::Closure(t, p));
                self.release(m);
            }
            Node::If { cond, then, els } => {
                let jf = self.cond_jump(cond, false);
                self.expr(then, dest);
                let needs_else = els.is_some() || matches!(dest, Dest::Reg(_));
                let jend = if needs_else { Some(self.emit_jump()) } else { None };
                self.patch_all_here(jf);
                if els.is_some() {
                    self.expr(els, dest);
                } else if let Dest::Reg(d) = dest {
                    self.at(id);
                    self.load_nil(d);
                }
                if let Some(j) = jend {
                    self.patch_here(j);
                }
            }
            Node::Block { stmts, tail } => {
                for s in self.ast.nodes_of(stmts) {
                    self.stmt(s);
                }
                if tail.is_some() {
                    self.expr(tail, dest);
                } else if let Dest::Reg(d) = dest {
                    self.load_nil(d);
                }
                if let Res::Local(m) = self.an.res[id.idx()] {
                    self.at(id);
                    self.emit(Op::CloseUpvals(m as u8));
                }
            }
            Node::Loop { body } => self.loop_expr(id, body, dest),
            Node::While { cond, body } => self.while_expr(id, cond, body, dest),
            Node::For(fid) => self.for_expr(id, fid, dest),
            Node::Break(v) => self.jump_out(id, v, true),
            Node::Continue => self.jump_out(id, NodeId::NONE, false),
            Node::Return(v) => {
                let m = self.mark();
                let is_ctor = self.fb().is_ctor;
                if is_ctor {
                    if v.is_some() {
                        self.expr(v, Dest::Discard);
                    }
                    self.at(id);
                    self.emit(Op::Return(0));
                } else if v.is_some() {
                    self.tail_return(v);
                } else {
                    self.emit(Op::ReturnNil);
                }
                self.release(m);
            }
            Node::Try(e) => {
                let m = self.mark();
                let t = self.target(dest);
                self.expr(e, Dest::Reg(t));
                self.at(id);
                self.emit(Op::Try(t));
                self.release(m);
            }
            Node::Match(mid) => self.match_expr(id, mid, dest),
            Node::Union { .. }
            | Node::Var { .. }
            | Node::Const { .. }
            | Node::Struct { .. }
            | Node::Impl { .. }
            | Node::Import(_)
            | Node::Export(_) => {
                unreachable!("declarations are statements")
            }
        }
    }

    /// `R[t] = R[l] op rhs`, using the constant form when `rhs` is a number literal.
    fn arith(&mut self, id: NodeId, op: BinOp, t: u8, l: u8, rhs: NodeId) {
        if arith_op_k(op, 0, 0, 0).is_some() {
            if let Some(n) = self.const_num(rhs, 0) {
                let k = self.num_const(n);
                if let Ok(k8) = u8::try_from(k) {
                    self.at(id);
                    self.emit(arith_op_k(op, t, l, k8).unwrap());
                    return;
                }
            }
        }
        let r = self.expr_reg(rhs);
        self.at(id);
        self.emit(arith_op(op, t, l, r));
    }

    /// Register for the callee of a call: the destination itself when it is the topmost
    /// temporary (so the result needs no move), otherwise a fresh temporary.
    fn call_slot(&mut self, dest: Dest) -> u8 {
        if let Dest::Reg(d) = dest {
            if self.is_top_temp(d) {
                return d;
            }
        }
        self.alloc()
    }

    /// Emit the call whose callee and `n` arguments are in `a, a+1, …`; deliver the result.
    fn finish_call(&mut self, id: NodeId, a: u8, n: usize, dest: Dest) {
        self.at(id);
        if n > 255 {
            self.error(self.cur, "too many arguments (max 255)");
        }
        self.emit(Op::Call(a, n.min(255) as u8));
        if let Dest::Reg(d) = dest {
            if d != a {
                self.emit(Op::Move(d, a));
            }
        }
    }

    fn assign(&mut self, id: NodeId, op: Option<BinOp>, target: NodeId, value: NodeId, dest: Dest) {
        let m = self.mark();
        match self.ast.node(target) {
            Node::Ident(sym) => {
                let res = self.an.res[target.idx()];
                // the register that ends up holding the new value
                let result;
                match (res, op) {
                    (Res::Local(s), None) => {
                        let s = s as u8;
                        if self.writes_last(value) {
                            self.expr(value, Dest::Reg(s));
                        } else {
                            let t = self.alloc();
                            self.expr(value, Dest::Reg(t));
                            self.at(id);
                            self.emit(Op::Move(s, t));
                        }
                        result = s;
                    }
                    (Res::Local(s), Some(op)) => {
                        let s = s as u8;
                        let l = self.operand(target, &[value]);
                        self.arith(id, op, s, l, value);
                        result = s;
                    }
                    (_, None) => {
                        let r = self.expr_reg(value);
                        self.at(id);
                        self.set_var(res, sym, r);
                        result = r;
                    }
                    (_, Some(op)) => {
                        let cur = self.alloc();
                        self.get_var(res, sym, cur);
                        self.arith(id, op, cur, cur, value);
                        self.at(id);
                        self.set_var(res, sym, cur);
                        result = cur;
                    }
                }
                if let Dest::Reg(d) = dest {
                    if d != result {
                        self.emit(Op::Move(d, result));
                    }
                }
            }
            Node::Field { obj, name } => {
                let k = self.sym_const(name);
                let o = self.operand(obj, &[value]);
                let v = if let Some(op) = op {
                    let cur = self.alloc();
                    self.at(id);
                    self.get_field_op(cur, o, k);
                    self.arith(id, op, cur, cur, value);
                    cur
                } else {
                    self.expr_reg(value)
                };
                self.at(id);
                self.set_field_op(o, k, v);
                if let Dest::Reg(d) = dest {
                    if d != v {
                        self.emit(Op::Move(d, v));
                    }
                }
            }
            Node::Index { obj, index } => {
                let o = self.operand(obj, &[index, value]);
                let i = self.operand(index, &[value]);
                let v = if let Some(op) = op {
                    let cur = self.alloc();
                    self.at(id);
                    self.emit(Op::GetIndex(cur, o, i));
                    self.arith(id, op, cur, cur, value);
                    cur
                } else {
                    self.expr_reg(value)
                };
                self.at(id);
                self.emit(Op::SetIndex(o, i, v));
                if let Dest::Reg(d) = dest {
                    if d != v {
                        self.emit(Op::Move(d, v));
                    }
                }
            }
            _ => unreachable!("parser validated assignment targets"),
        }
        self.release(m);
    }

    // ---- match -----------------------------------------------------------------

    /// The scrutinee is stored in a hidden local; each arm tests its pattern against
    /// locals (failures jump to the next arm), binds variables straight into their
    /// registers, checks the guard and evaluates the body.
    fn match_expr(&mut self, id: NodeId, mid: MatchId, dest: Dest) {
        let def = self.ast.match_def(mid);
        let slot = self.an.match_slot[mid.0 as usize] as u8;
        self.expr(def.scrutinee, Dest::Reg(slot));
        let mut to_end = Vec::new();
        for k in self.ast.arms_of(mid) {
            let (pat, guard, body) = {
                let a = &self.ast.arms[k];
                (a.pat, a.guard, a.body)
            };
            let close = self.an.arm_close[k];
            let mut fails = Vec::new();
            self.pat_test(pat, &mut fails);
            if guard.is_some() {
                fails.extend(self.cond_jump(guard, false));
            }
            self.expr(body, dest);
            if let Some(m) = close {
                self.at(id);
                self.emit(Op::CloseUpvals(m as u8));
            }
            self.at(id);
            to_end.push(self.emit_jump());
            self.patch_all_here(fails);
            if let Some(m) = close {
                // variables of a failed arm may have been captured by a guard
                self.emit(Op::CloseUpvals(m as u8));
            }
        }
        self.at(id);
        self.emit(Op::MatchFail(slot));
        self.patch_all_here(to_end);
    }

    /// Emit the test of one pattern against its source register; every failing check
    /// jumps (patch list `fails`), a success falls through with bindings assigned.
    fn pat_test(&mut self, pid: PatId, fails: &mut Vec<usize>) {
        let info = self.an.pats[pid.0 as usize];
        let at_pat = self.ast.pat_spans[pid.0 as usize];
        self.cur = at_pat;
        let src = info.src as u8;
        match self.ast.pat(pid) {
            Pat::Wild => {}
            Pat::Bind(_) => {
                if info.bind != info.src {
                    self.emit(Op::Move(info.bind as u8, src));
                }
            }
            Pat::Num(n) => {
                let k = self.num_const(n);
                self.pat_literal(src, k, fails);
            }
            Pat::Str { start, len } => {
                let raw = &self.source.text[start as usize..(start + len) as usize];
                let s = unescape(raw).unwrap_or_default();
                let k = self.str_const(&s);
                self.pat_literal(src, k, fails);
            }
            Pat::Bool(b) => {
                let k = self.add_const(Value::bool(b));
                self.pat_literal(src, k, fails);
            }
            Pat::Nil => {
                let k = self.add_const(Value::NIL);
                self.pat_literal(src, k, fails);
            }
            Pat::Variant { qual, name, args, parens } => {
                if args.len >= 255 {
                    self.error(at_pat, "too many fields in a pattern (max 254)");
                }
                let arity = if parens { args.len.min(254) as u8 } else { 255 };
                let nk = self.sym_const(name);
                let nk = self.k16(nk, "a match");
                let m = self.mark();
                let t = self.alloc();
                let ty = if qual.is_some() {
                    let q = self.alloc();
                    self.expr(qual, Dest::Reg(q));
                    self.cur = at_pat;
                    q
                } else {
                    255
                };
                self.emit_ext(Op::IsVariant(t, src, ty), ((arity as u32) << 16) | nk as u32);
                self.cur = at_pat;
                fails.push(self.emit_wide(Op::JFalse(t)));
                self.release(m);
                for (j, a) in self.ast.pats_of(args).enumerate() {
                    if matches!(self.ast.pat(a), Pat::Wild) {
                        continue;
                    }
                    self.cur = at_pat;
                    self.emit(Op::Unpack(self.an.pats[a.0 as usize].src as u8, src, j as u8));
                    self.pat_test(a, fails);
                }
            }
            Pat::Or(alts) => {
                let n = alts.len as usize;
                let mut ok = Vec::new();
                for (i, alt) in self.ast.pats_of(alts).enumerate() {
                    if i + 1 == n {
                        // the last alternative's failures are the whole pattern's failures
                        self.pat_test(alt, fails);
                    } else {
                        let mut alt_fails = Vec::new();
                        self.pat_test(alt, &mut alt_fails);
                        ok.push(self.emit_jump());
                        self.patch_all_here(alt_fails);
                    }
                }
                self.patch_all_here(ok);
            }
        }
    }

    /// `src == K[k]`, else fail.
    fn pat_literal(&mut self, src: u8, k: u32, fails: &mut Vec<usize>) {
        match u16::try_from(k) {
            Ok(k) => fails.push(self.emit_wide(Op::JNeK(src, k))),
            Err(_) => {
                let m = self.mark();
                let t = self.alloc();
                self.load_const(t, k);
                fails.push(self.emit_wide(Op::JNe(src, t)));
                self.release(m);
            }
        }
    }

    // ---- loops ---------------------------------------------------------------------

    fn push_loop(&mut self, id: NodeId, dest: Dest) {
        let close = match self.an.res[id.idx()] {
            Res::Local(m) => Some(m),
            _ => None,
        };
        self.fb().loops.push(LoopCtx { dest, breaks: Vec::new(), continues: Vec::new(), close });
    }

    fn emit_close(&mut self, id: NodeId) {
        if let Res::Local(m) = self.an.res[id.idx()] {
            self.at(id);
            self.emit(Op::CloseUpvals(m as u8));
        }
    }

    /// Patch the pending `continue`s to `cont` and the `break`s to here.
    fn finish_loop(&mut self, cont: usize) {
        let ctx = self.fb().loops.pop().unwrap();
        for c in ctx.continues {
            self.patch_to(c, cont);
        }
        self.patch_all_here(ctx.breaks);
    }

    fn loop_expr(&mut self, id: NodeId, body: NodeId, dest: Dest) {
        self.push_loop(id, dest);
        let start = self.here();
        let m = self.mark();
        self.expr(body, Dest::Discard);
        self.release(m);
        self.emit_close(id);
        self.at(id);
        self.emit_jump_to(start);
        // control only arrives here through `break`, which has stored its value
        self.finish_loop(start);
    }

    /// `while c { body }` is compiled with the test at the bottom:
    /// `jump test; body: …; test: if c jump body`.
    fn while_expr(&mut self, id: NodeId, cond: NodeId, body: NodeId, dest: Dest) {
        self.push_loop(id, dest);
        self.at(id);
        let to_test = self.emit_jump();
        let start = self.here();
        let m = self.mark();
        self.expr(body, Dest::Discard);
        self.release(m);
        self.emit_close(id);
        let test = self.here();
        self.patch_to(to_test, test);
        let back = self.cond_jump(cond, true);
        for j in back {
            self.patch_to(j, start);
        }
        // normal exit: the loop evaluates to nil
        if let Dest::Reg(d) = dest {
            self.at(id);
            self.load_nil(d);
        }
        self.finish_loop(test);
    }

    fn for_expr(&mut self, id: NodeId, fid: ForId, dest: Dest) {
        let def = self.ast.for_def(fid);
        let base = self.an.for_base[fid.0 as usize] as u8;
        let (iter, end, body, nvars) = (def.iter, def.end, def.body, def.nvars);
        let exit;
        let body_start;
        let m = self.mark();
        if end.is_some() {
            // for i in a..b
            self.expr(iter, Dest::Reg(base));
            self.expr(end, Dest::Reg(base + 1));
            self.at(id);
            exit = self.emit_wide(Op::ForPrep(base));
            body_start = self.here();
        } else {
            // for [k,] v in collection
            self.expr(iter, Dest::Reg(base));
            self.at(id);
            self.load_num(base + 1, 0.0);
            body_start = self.here();
            exit = self.emit_wide(if nvars == 1 { Op::IterNext1(base) } else { Op::IterNext2(base) });
        }
        self.release(m);
        self.push_loop(id, dest);
        let bm = self.mark();
        self.expr(body, Dest::Discard);
        self.release(bm);
        self.emit_close(id);
        let cont = self.here();
        self.at(id);
        if end.is_some() {
            let at = self.emit_wide(Op::ForLoop(base));
            // loop back to the first instruction of the body
            self.patch_to(at, body_start);
        } else {
            self.emit_jump_to(body_start);
        }
        self.patch_here(exit);
        if let Dest::Reg(d) = dest {
            self.load_nil(d);
        }
        let ct = if end.is_some() { cont } else { body_start };
        self.finish_loop(ct);
    }

    /// `break [value]` / `continue`.
    fn jump_out(&mut self, id: NodeId, value: NodeId, is_break: bool) {
        let (dest, close) = {
            let l = self.fb().loops.last().expect("analyzer checked loop context");
            (l.dest, l.close)
        };
        if is_break {
            let m = self.mark();
            if value.is_some() {
                self.expr(value, dest);
            } else if let Dest::Reg(d) = dest {
                self.at(id);
                self.load_nil(d);
            }
            self.release(m);
        }
        if let Some(m) = close {
            self.at(id);
            self.emit(Op::CloseUpvals(m as u8));
        }
        self.at(id);
        let j = self.emit_jump();
        let l = self.fb().loops.last_mut().unwrap();
        if is_break {
            l.breaks.push(j);
        } else {
            l.continues.push(j);
        }
    }
}
