//! AST -> bytecode.
//!
//! The VM is stack based, but variables never live on the operand stack:
//! the analyzer gave each of them a fixed frame slot, so expressions can nest
//! arbitrarily (`f(1, { var t = 2; t })`) without the compiler having to
//! track where locals are.  The compiler only tracks the *depth* of the
//! operand stack, which it needs for `break`/`continue` (to drop temporaries)
//! and to size frames.
//!
//! Everything is an expression: `expr()` always leaves exactly one value on
//! the stack; `stmt()` leaves none.

use crate::analyzer::{Analysis, FuncInfo, Res};
use crate::ast::*;
use crate::hash::FxHashMap;
use crate::heap::Heap;
use crate::intern::{Interner, Sym};
use crate::object::*;
use crate::op::{Op, I24, U24};
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

struct LoopCtx {
    /// Operand stack depth at loop entry (before the loop's result value).
    depth: i32,
    breaks: Vec<usize>,
    continues: Vec<usize>,
    /// Lowest captured slot inside the loop; closed on `break`/`continue`.
    close: Option<u16>,
}

struct FnBuilder {
    code: Vec<Op>,
    offsets: Vec<u32>,
    consts: Vec<Value>,
    const_map: FxHashMap<u64, u32>,
    depth: i32,
    max_depth: i32,
    loops: Vec<LoopCtx>,
    is_ctor: bool,
}

impl FnBuilder {
    fn new(is_ctor: bool) -> FnBuilder {
        FnBuilder {
            code: Vec::new(),
            offsets: Vec::new(),
            consts: Vec::new(),
            const_map: FxHashMap::default(),
            depth: 0,
            max_depth: 0,
            loops: Vec::new(),
            is_ctor,
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
    c.fns.push(FnBuilder::new(false));
    if an.module {
        c.module_root(root);
    } else {
        c.expr(root);
        c.emit(Op::Return);
    }
    let fb = c.fns.pop().unwrap();
    let proto = c.finish(fb, "<main>".into(), 0, &an.main);
    if c.errors.is_empty() {
        Ok(proto)
    } else {
        Err(c.errors)
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
        fb.depth += op.stack_effect();
        if fb.depth > fb.max_depth {
            fb.max_depth = fb.depth;
        }
        debug_assert!(fb.depth >= 0, "operand stack underflow after {:?}", op);
        fb.code.len() - 1
    }

    fn here(&mut self) -> usize {
        self.fb().code.len()
    }

    fn emit_jump(&mut self, op: Op) -> usize {
        self.emit(op)
    }

    /// Emit a double instruction: the op plus the `Ext` word that will hold its
    /// branch offset (patched later with `patch_to`).  Returns the op's index.
    fn emit_wide(&mut self, op: Op) -> usize {
        debug_assert!(op.is_wide());
        let at = self.emit(op);
        let cur = self.cur;
        let fb = self.fb();
        fb.code.push(Op::Ext(I24::ZERO));
        fb.offsets.push(cur);
        at
    }

    /// Point the branch at `at` to the instruction index `target`.
    fn patch_to(&mut self, at: usize, target: usize) {
        let wide = self.fb().code[at].is_wide();
        let rel = target as i64 - (at as i64 + 1 + wide as i64);
        let off = I24::new(rel as i32).filter(|_| i32::try_from(rel).is_ok());
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

    fn u24(&mut self, v: u32) -> U24 {
        match U24::new(v) {
            Some(x) => x,
            None => {
                self.error(self.cur, "too many constants or globals (limit is 16M)");
                U24::ZERO
            }
        }
    }

    fn patch_here(&mut self, at: usize) {
        let h = self.here();
        self.patch_to(at, h);
    }

    fn emit_loop_back(&mut self, target: usize) {
        let at = self.emit(Op::Jump(I24::ZERO));
        self.patch_to(at, target);
    }

    fn set_depth(&mut self, d: i32) {
        self.fb().depth = d;
    }

    fn add_const(&mut self, v: Value) -> U24 {
        let fb = self.fb();
        if let Some(&i) = fb.const_map.get(&v.bits()) {
            return U24::new(i).unwrap();
        }
        let i = fb.consts.len() as u32;
        fb.consts.push(v);
        fb.const_map.insert(v.bits(), i);
        self.u24(i)
    }

    fn num_const(&mut self, n: f64) -> U24 {
        self.add_const(Value::num(n))
    }

    fn str_const(&mut self, s: &str) -> U24 {
        let g = self.heap.intern(s);
        self.add_const(Value::from(g))
    }

    fn sym_const(&mut self, s: Sym) -> U24 {
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

    fn pop_n(&mut self, n: i32) {
        match n {
            0 => {}
            1 => {
                self.emit(Op::Pop);
            }
            _ => {
                self.emit(Op::PopN(n as u16));
            }
        }
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

    // ---- functions ---------------------------------------------------------------

    fn function(&mut self, fid: FuncId, is_ctor: bool) -> Gc<Proto> {
        let def = self.ast.func(fid);
        let info = &self.an.funcs[fid.0 as usize];
        let arity = def.params.len;
        if arity > 255 {
            let off = self.ast.span(def.body);
            self.error(off, "too many parameters (max 255)");
        }
        self.fns.push(FnBuilder::new(is_ctor));
        self.expr(def.body);
        if is_ctor {
            // constructors always evaluate to the new instance
            self.emit(Op::Pop);
            self.emit(Op::GetLocal(0));
        }
        self.emit(Op::Return);
        let fb = self.fns.pop().unwrap();
        let name: Rc<str> = if def.name.is_none() { Rc::from("<anon>") } else { Rc::from(self.interner.get(def.name)) };
        self.finish(fb, name, arity as u8, info)
    }

    fn finish(&mut self, fb: FnBuilder, name: Rc<str>, arity: u8, info: &FuncInfo) -> Gc<Proto> {
        let FnBuilder { code, offsets, consts, max_depth, .. } = fb;
        let (code, offsets) = if self.opts.optimize {
            match peephole::optimize_with(&mut self.scratch, code.clone(), offsets.clone(), &consts) {
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
        let proto = Proto {
            hdr: Header::new(Kind::Proto),
            name,
            arity,
            nslots: info.nslots,
            max_stack: (max_depth + 1) as u16,
            upvals: info.upvals.clone(),
            code,
            consts,
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
            self.expr(tail);
            self.emit(Op::Pop);
        }
        self.cur = 0;
        self.emit(Op::NewTable(self.an.exports.len().min(u16::MAX as usize) as u16));
        for &(name, res) in &self.an.exports {
            self.get_var(res, name);
            let k = self.sym_const(name);
            self.emit(Op::InitField(k));
        }
        self.emit(Op::Return);
    }

    fn store_decl(&mut self, res: Res, name: Sym) {
        match res {
            Res::Local(s) => {
                self.emit(Op::StoreLocal(s));
            }
            Res::Global => {
                let g = self.u24(name.0);
                self.emit(Op::DefGlobal(g));
            }
            _ => unreachable!("declaration resolved to {:?}", res),
        }
    }

    fn stmt(&mut self, id: NodeId) {
        self.at(id);
        match self.ast.node(id) {
            Node::Nil => {} // empty statement
            Node::Var { name, init } | Node::Const { name, init } => {
                if init.is_some() {
                    self.expr(init);
                } else {
                    self.emit(Op::Nil);
                }
                self.at(id);
                let res = self.an.res[id.idx()];
                self.store_decl(res, name);
            }
            Node::Struct { name, fields } => {
                let sname = self.heap.intern(self.interner.get(name));
                let fs: Vec<Gc<Str>> =
                    self.ast.syms_of(fields).map(|f| self.heap.intern(self.interner.get(f))).collect();
                let ty = self.heap.alloc(StructType {
                    hdr: Header::new(Kind::StructType),
                    name: sname,
                    fields: fs,
                    methods: Vec::new(),
                    hooks: [Value::NIL; HOOK_COUNT],
                });
                let k = self.add_const(Value::from(ty));
                self.emit(Op::Struct(k));
                let res = self.an.res[id.idx()];
                self.store_decl(res, name);
            }
            Node::Export(inner) => self.stmt(inner),
            Node::Import(iid) => {
                let d = self.ast.import_def(iid);
                let raw = &self.source.text[d.path_start as usize..(d.path_start + d.path_len) as usize];
                let path = unescape(raw).unwrap_or_default();
                let k = self.str_const(&path);
                self.emit(Op::Import(k));
                if d.module_name.is_none() {
                    // import { a, b } from "path"
                    for (j, name) in self.ast.syms_of(d.names).enumerate() {
                        self.emit(Op::Dup);
                        let nk = self.sym_const(name);
                        self.emit(Op::GetExport(nk));
                        let res = self.an.import_res[iid.0 as usize][j];
                        self.store_decl(res, name);
                    }
                    self.emit(Op::Pop);
                } else {
                    let res = self.an.import_res[iid.0 as usize][0];
                    self.store_decl(res, d.module_name);
                }
            }
            Node::Impl { target, methods } => {
                self.expr(target);
                for m in self.ast.nodes_of(methods) {
                    let Node::Func(fid) = self.ast.node(m) else { unreachable!() };
                    let name = self.ast.func(fid).name;
                    let is_ctor = self.interner.get(name) == "new";
                    let proto = self.function(fid, is_ctor);
                    let pk = self.add_const(Value::from(proto));
                    self.at(m);
                    self.emit(Op::Closure(pk));
                    let nk = self.sym_const(name);
                    self.emit(Op::DefMethod(nk));
                }
                self.at(id);
                self.emit(Op::Pop);
            }
            _ => {
                self.expr(id);
                self.emit(Op::Pop);
            }
        }
    }

    // ---- conditions ----------------------------------------------------------------

    /// Emit code that jumps (returned patch list) when `id` is falsy and falls
    /// through when truthy.  `&&`, `||` and `!` compile to pure control flow.
    fn cond_false(&mut self, id: NodeId) -> Vec<usize> {
        match self.ast.node(id) {
            Node::Binary { op: BinOp::And, lhs, rhs } => {
                let mut j = self.cond_false(lhs);
                j.extend(self.cond_false(rhs));
                j
            }
            Node::Binary { op: BinOp::Or, lhs, rhs } => {
                let jt = self.cond_true(lhs);
                let jf = self.cond_false(rhs);
                for j in jt {
                    self.patch_here(j);
                }
                jf
            }
            Node::Unary { op: UnOp::Not, expr } => self.cond_true(expr),
            _ => {
                self.expr(id);
                self.at(id);
                vec![self.emit_jump(Op::JumpIfFalse(I24::ZERO))]
            }
        }
    }

    /// Jumps when truthy.
    fn cond_true(&mut self, id: NodeId) -> Vec<usize> {
        match self.ast.node(id) {
            Node::Binary { op: BinOp::Or, lhs, rhs } => {
                let mut j = self.cond_true(lhs);
                j.extend(self.cond_true(rhs));
                j
            }
            Node::Binary { op: BinOp::And, lhs, rhs } => {
                let jf = self.cond_false(lhs);
                let jt = self.cond_true(rhs);
                for j in jf {
                    self.patch_here(j);
                }
                jt
            }
            Node::Unary { op: UnOp::Not, expr } => self.cond_false(expr),
            _ => {
                self.expr(id);
                self.at(id);
                vec![self.emit_jump(Op::JumpIfTrue(I24::ZERO))]
            }
        }
    }

    // ---- expressions ---------------------------------------------------------------

    fn get_var(&mut self, res: Res, sym: Sym) {
        match res {
            Res::Local(s) => self.emit(Op::GetLocal(s)),
            Res::Upvalue(i) => self.emit(Op::GetUpval(i)),
            Res::Global => {
                let g = self.u24(sym.0);
                self.emit(Op::GetGlobal(g))
            }
            Res::None => unreachable!("unresolved identifier"),
        };
    }

    fn set_var(&mut self, res: Res, sym: Sym) {
        match res {
            Res::Local(s) => self.emit(Op::SetLocal(s)),
            Res::Upvalue(i) => self.emit(Op::SetUpval(i)),
            Res::Global => {
                let g = self.u24(sym.0);
                self.emit(Op::SetGlobal(g))
            }
            Res::None => unreachable!("unresolved identifier"),
        };
    }

    fn binop(&mut self, op: BinOp) {
        let o = match op {
            BinOp::Add => Op::Add,
            BinOp::Sub => Op::Sub,
            BinOp::Mul => Op::Mul,
            BinOp::Div => Op::Div,
            BinOp::Rem => Op::Rem,
            BinOp::Pow => Op::Pow,
            BinOp::BitAnd => Op::BitAnd,
            BinOp::BitOr => Op::BitOr,
            BinOp::BitXor => Op::BitXor,
            BinOp::Shl => Op::Shl,
            BinOp::Shr => Op::Shr,
            BinOp::Eq => Op::Eq,
            BinOp::Ne => Op::Ne,
            BinOp::Lt => Op::Lt,
            BinOp::Le => Op::Le,
            BinOp::Gt => Op::Gt,
            BinOp::Ge => Op::Ge,
            BinOp::And | BinOp::Or => unreachable!("short-circuit ops are control flow"),
        };
        self.emit(o);
    }

    fn expr(&mut self, id: NodeId) {
        self.at(id);
        let node = self.ast.node(id);
        match node {
            Node::Nil => {
                self.emit(Op::Nil);
            }
            Node::True => {
                self.emit(Op::True);
            }
            Node::False => {
                self.emit(Op::False);
            }
            Node::Num(n) => {
                let k = self.num_const(n);
                self.emit(Op::Const(k));
            }
            Node::Str { start, len } => {
                let raw = &self.source.text[start as usize..(start + len) as usize];
                let s = unescape(raw).unwrap_or_default();
                let k = self.str_const(&s);
                self.emit(Op::Const(k));
            }
            Node::Key(sym) => {
                let k = self.sym_const(sym);
                self.emit(Op::Const(k));
            }
            Node::Ident(sym) => {
                let res = self.an.res[id.idx()];
                self.get_var(res, sym);
            }
            Node::Unary { op, expr } => {
                if op == UnOp::Neg {
                    if let Some(n) = self.const_num(id, 0) {
                        let k = self.num_const(n);
                        self.emit(Op::Const(k));
                        return;
                    }
                }
                self.expr(expr);
                self.at(id);
                self.emit(match op {
                    UnOp::Neg => Op::Neg,
                    UnOp::Not => Op::Not,
                    UnOp::BitNot => Op::BitNot,
                });
            }
            Node::Binary { op: BinOp::And, lhs, rhs } => {
                self.expr(lhs);
                self.at(id);
                let j = self.emit_jump(Op::And(I24::ZERO));
                self.expr(rhs);
                self.patch_here(j);
            }
            Node::Binary { op: BinOp::Or, lhs, rhs } => {
                self.expr(lhs);
                self.at(id);
                let j = self.emit_jump(Op::Or(I24::ZERO));
                self.expr(rhs);
                self.patch_here(j);
            }
            Node::Binary { op, lhs, rhs } => {
                if let Some(n) = self.const_num(id, 0) {
                    let k = self.num_const(n);
                    self.emit(Op::Const(k));
                    return;
                }
                self.expr(lhs);
                self.expr(rhs);
                self.at(id);
                self.binop(op);
            }
            Node::Assign { op, target, value } => self.assign(id, op, target, value),
            Node::Call { callee, args } => {
                self.expr(callee);
                let n = args.len;
                for a in self.ast.nodes_of(args) {
                    self.expr(a);
                }
                self.at(id);
                if n > 255 {
                    self.error(self.cur, "too many arguments (max 255)");
                }
                self.emit(Op::Call(n.min(255) as u8));
            }
            Node::MethodCall { name, args } => {
                let list = self.ast.list(args);
                self.expr(NodeId(list[0]));
                self.at(id);
                let k = self.sym_const(name);
                self.emit(Op::GetMethod(k));
                for &a in &list[1..] {
                    self.expr(NodeId(a));
                }
                self.at(id);
                if list.len() > 255 {
                    self.error(self.cur, "too many arguments (max 255)");
                }
                self.emit(Op::Call(list.len().min(255) as u8));
            }
            Node::Field { obj, name } => {
                self.expr(obj);
                self.at(id);
                let k = self.sym_const(name);
                self.emit(Op::GetField(k));
            }
            Node::Index { obj, index } => {
                self.expr(obj);
                self.expr(index);
                self.at(id);
                self.emit(Op::GetIndex);
            }
            Node::Array { items } => {
                self.emit(Op::NewTable(items.len.min(u16::MAX as u32) as u16));
                for (i, a) in self.ast.nodes_of(items).enumerate() {
                    if i < u16::MAX as usize {
                        self.expr(a);
                        self.at(id);
                        self.emit(Op::Append(i as u16));
                    } else {
                        let k = self.num_const(i as f64);
                        self.emit(Op::Const(k));
                        self.expr(a);
                        self.at(id);
                        self.emit(Op::InitIndex);
                    }
                }
            }
            Node::Table { entries } => {
                self.emit(Op::NewTable(0));
                let list = self.ast.list(entries);
                for pair in list.chunks(2) {
                    let (k, v) = (NodeId(pair[0]), NodeId(pair[1]));
                    if let Node::Key(sym) = self.ast.node(k) {
                        self.expr(v);
                        self.at(k);
                        let c = self.sym_const(sym);
                        self.emit(Op::InitField(c));
                    } else {
                        self.expr(k);
                        self.expr(v);
                        self.at(k);
                        self.emit(Op::InitIndex);
                    }
                }
            }
            Node::Func(fid) => {
                let proto = self.function(fid, false);
                let k = self.add_const(Value::from(proto));
                self.at(id);
                self.emit(Op::Closure(k));
            }
            Node::If { cond, then, els } => {
                let d = self.fb().depth;
                let jf = self.cond_false(cond);
                self.expr(then);
                let jend = self.emit_jump(Op::Jump(I24::ZERO));
                for j in jf {
                    self.patch_here(j);
                }
                self.set_depth(d);
                if els.is_some() {
                    self.expr(els);
                } else {
                    self.at(id);
                    self.emit(Op::Nil);
                }
                self.patch_here(jend);
            }
            Node::Block { stmts, tail } => {
                for s in self.ast.nodes_of(stmts) {
                    self.stmt(s);
                }
                if tail.is_some() {
                    self.expr(tail);
                } else {
                    self.emit(Op::Nil);
                }
                if let Res::Local(m) = self.an.res[id.idx()] {
                    self.at(id);
                    self.emit(Op::CloseUpvals(m));
                }
            }
            Node::Loop { body } => self.loop_expr(id, body),
            Node::While { cond, body } => self.while_expr(id, cond, body),
            Node::For(fid) => self.for_expr(id, fid),
            Node::Break(v) => self.jump_out(id, v, true),
            Node::Continue => self.jump_out(id, NodeId::NONE, false),
            Node::Return(v) => {
                let is_ctor = self.fb().is_ctor;
                if is_ctor {
                    if v.is_some() {
                        self.expr(v);
                        self.emit(Op::Pop);
                    }
                    self.at(id);
                    self.emit(Op::GetLocal(0));
                } else if v.is_some() {
                    self.expr(v);
                } else {
                    self.emit(Op::Nil);
                }
                self.at(id);
                self.emit(Op::Return);
                // `return` is an expression too: keep the stack model balanced
                // (this `nil` is dead code and removed by the peephole).
                self.emit(Op::Nil);
            }
            Node::Var { .. }
            | Node::Const { .. }
            | Node::Struct { .. }
            | Node::Impl { .. }
            | Node::Import(_)
            | Node::Export(_) => {
                unreachable!("declarations are statements")
            }
        }
    }

    fn assign(&mut self, id: NodeId, op: Option<BinOp>, target: NodeId, value: NodeId) {
        match self.ast.node(target) {
            Node::Ident(sym) => {
                let res = self.an.res[target.idx()];
                if let Some(op) = op {
                    self.get_var(res, sym);
                    self.expr(value);
                    self.at(id);
                    self.binop(op);
                } else {
                    self.expr(value);
                }
                self.at(id);
                self.set_var(res, sym);
            }
            Node::Field { obj, name } => {
                let k = self.sym_const(name);
                self.expr(obj);
                if let Some(op) = op {
                    self.at(id);
                    self.emit(Op::Dup);
                    self.emit(Op::GetField(k));
                    self.expr(value);
                    self.at(id);
                    self.binop(op);
                } else {
                    self.expr(value);
                }
                self.at(id);
                self.emit(Op::SetField(k));
            }
            Node::Index { obj, index } => {
                self.expr(obj);
                self.expr(index);
                if let Some(op) = op {
                    self.at(id);
                    self.emit(Op::Dup2);
                    self.emit(Op::GetIndex);
                    self.expr(value);
                    self.at(id);
                    self.binop(op);
                } else {
                    self.expr(value);
                }
                self.at(id);
                self.emit(Op::SetIndex);
            }
            _ => unreachable!("parser validated assignment targets"),
        }
    }

    // ---- loops ---------------------------------------------------------------------

    fn push_loop(&mut self, id: NodeId) {
        let close = match self.an.res[id.idx()] {
            Res::Local(m) => Some(m),
            _ => None,
        };
        let depth = self.fb().depth;
        self.fb().loops.push(LoopCtx { depth, breaks: Vec::new(), continues: Vec::new(), close });
    }

    fn emit_close(&mut self, id: NodeId) {
        if let Res::Local(m) = self.an.res[id.idx()] {
            self.at(id);
            self.emit(Op::CloseUpvals(m));
        }
    }

    fn loop_expr(&mut self, id: NodeId, body: NodeId) {
        let d = self.fb().depth;
        self.push_loop(id);
        let start = self.here();
        self.expr(body);
        self.emit(Op::Pop);
        self.emit_close(id);
        self.emit_loop_back(start);
        let ctx = self.fb().loops.pop().unwrap();
        for c in ctx.continues {
            self.patch_to(c, start);
        }
        // control only arrives here through `break`, with the value on the stack
        self.set_depth(d);
        let h = self.here();
        for b in ctx.breaks {
            self.patch_to(b, h);
        }
        self.set_depth(d + 1);
    }

    fn while_expr(&mut self, id: NodeId, cond: NodeId, body: NodeId) {
        let d = self.fb().depth;
        self.push_loop(id);
        let start = self.here();
        let exits = self.cond_false(cond);
        self.expr(body);
        self.emit(Op::Pop);
        self.emit_close(id);
        self.emit_loop_back(start);
        for j in exits {
            self.patch_here(j);
        }
        self.set_depth(d);
        self.at(id);
        self.emit(Op::Nil);
        let ctx = self.fb().loops.pop().unwrap();
        for c in ctx.continues {
            self.patch_to(c, start);
        }
        let h = self.here();
        for b in ctx.breaks {
            self.patch_to(b, h);
        }
    }

    fn for_expr(&mut self, id: NodeId, fid: ForId) {
        let def = self.ast.for_def(fid);
        let base = self.an.for_base[fid.0 as usize];
        let d = self.fb().depth;
        let (iter, end, body, nvars) = (def.iter, def.end, def.body, def.nvars);
        let exit;
        let loop_target;
        let cont_target_is_start;
        let body_start;
        if end.is_some() {
            // for i in a..b
            self.expr(iter);
            self.emit(Op::StoreLocal(base));
            self.expr(end);
            self.emit(Op::StoreLocal(base + 1));
            self.at(id);
            exit = self.emit_wide(Op::ForPrep(base));
            body_start = self.here();
            loop_target = body_start;
            cont_target_is_start = false;
        } else {
            // for [k,] v in table
            self.expr(iter);
            self.emit(Op::StoreLocal(base));
            let k = self.num_const(0.0);
            self.emit(Op::Const(k));
            self.emit(Op::StoreLocal(base + 1));
            self.at(id);
            body_start = self.here();
            exit = self.emit_wide(if nvars == 1 { Op::IterNext1(base) } else { Op::IterNext2(base) });
            loop_target = body_start;
            cont_target_is_start = true;
        }
        self.push_loop(id);
        self.expr(body);
        self.emit(Op::Pop);
        self.emit_close(id);
        let cont_label = self.here();
        self.at(id);
        if end.is_some() {
            let at = self.emit_wide(Op::ForLoop(base));
            // loop back to the first instruction of the body
            self.patch_to(at, body_start);
        } else {
            self.emit_loop_back(loop_target);
        }
        self.patch_here(exit);
        self.set_depth(d);
        self.emit(Op::Nil);
        let ctx = self.fb().loops.pop().unwrap();
        let ct = if cont_target_is_start { loop_target } else { cont_label };
        for c in ctx.continues {
            self.patch_to(c, ct);
        }
        let h = self.here();
        for b in ctx.breaks {
            self.patch_to(b, h);
        }
    }

    /// `break [value]` / `continue`.
    fn jump_out(&mut self, id: NodeId, value: NodeId, is_break: bool) {
        let cur_depth = self.fb().depth;
        let (ld, close) = {
            let l = self.fb().loops.last().expect("analyzer checked loop context");
            (l.depth, l.close)
        };
        // drop temporaries that belong to enclosing expressions inside the loop
        self.at(id);
        self.pop_n(cur_depth - ld);
        if is_break {
            if value.is_some() {
                self.expr(value);
            } else {
                self.emit(Op::Nil);
            }
        }
        if let Some(m) = close {
            self.at(id);
            self.emit(Op::CloseUpvals(m));
        }
        self.at(id);
        let j = self.emit_jump(Op::Jump(I24::ZERO));
        let l = self.fb().loops.last_mut().unwrap();
        if is_break {
            l.breaks.push(j);
        } else {
            l.continues.push(j);
        }
        // the expression "evaluates" to a (dead) value
        self.set_depth(cur_depth);
        self.emit(Op::Nil);
    }
}
