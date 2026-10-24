//! Semantic analysis: one pass over the AST that
//!  * resolves every identifier to a local slot, an upvalue or a global,
//!  * assigns frame slots (scope-based reuse, so the compiler never has to
//!    guess where a variable lives even though the expression stack grows
//!    on top of the locals),
//!  * finds which variables are captured by closures (so scopes know when
//!    upvalues must be closed),
//!  * reports const-assignment, duplicate declarations and stray `break`.
//!
//! Variables declared directly in the top-level scope of a chunk are
//! *globals* (dense `Sym` indices into the VM's global table).  That keeps
//! forward references between top-level functions working and lets the REPL
//! keep state between lines.

use crate::ast::*;
use crate::hash::{FxHashMap, FxHashSet};
use crate::intern::{Interner, Sym};
use crate::source::Diag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Res {
    None,
    Local(u16),
    Upvalue(u16),
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpvalDesc {
    /// Capture a local of the directly enclosing function (else one of its upvalues).
    pub is_local: bool,
    pub index: u16,
}

#[derive(Clone, Debug, Default)]
pub struct FuncInfo {
    /// Number of frame slots used by parameters and locals.
    pub nslots: u16,
    pub upvals: Vec<UpvalDesc>,
}

pub struct Analysis {
    /// Resolution per node id (identifiers, declarations; for scope nodes the
    /// lowest captured slot that must be closed when the scope ends).
    pub res: Vec<Res>,
    pub funcs: Vec<FuncInfo>,
    /// First hidden slot of each `for` loop.
    pub for_base: Vec<u16>,
    pub main: FuncInfo,
}

/// Global facts that outlive a single chunk (REPL sessions).
#[derive(Default)]
pub struct GlobalInfo {
    pub consts: FxHashSet<Sym>,
}

struct Local {
    name: Sym,
    slot: u16,
    is_const: bool,
    scope: usize,
}

struct ScopeInfo {
    first_local: usize,
    base_slot: u32,
    captured_min: Option<u16>,
}

struct FuncState {
    locals: Vec<Local>,
    scopes: Vec<ScopeInfo>,
    next_slot: u32,
    max_slot: u32,
    upvals: Vec<UpvalDesc>,
    loop_depth: u32,
    is_main: bool,
}

impl FuncState {
    fn new(is_main: bool) -> FuncState {
        FuncState {
            locals: Vec::new(),
            scopes: Vec::new(),
            next_slot: 0,
            max_slot: 0,
            upvals: Vec::new(),
            loop_depth: 0,
            is_main,
        }
    }
}

struct Analyzer<'a> {
    ast: &'a Ast,
    interner: &'a Interner,
    globals: &'a mut GlobalInfo,
    res: Vec<Res>,
    funcs: Vec<FuncInfo>,
    for_base: Vec<u16>,
    fns: Vec<FuncState>,
    errors: Vec<Diag>,
    repl: bool,
}

pub fn analyze(ast: &Ast, interner: &Interner, globals: &mut GlobalInfo, repl: bool) -> Result<Analysis, Vec<Diag>> {
    let mut a = Analyzer {
        ast,
        interner,
        globals,
        res: vec![Res::None; ast.nodes.len()],
        funcs: vec![FuncInfo::default(); ast.funcs.len()],
        for_base: vec![0; ast.fors.len()],
        fns: Vec::new(),
        errors: Vec::new(),
        repl,
    };
    let root = ast.root.expect("parsed program");
    a.fns.push(FuncState::new(true));
    a.hoist_globals(root);
    a.root_block(root);
    let main = a.fns.pop().unwrap();
    if !a.errors.is_empty() {
        return Err(a.errors);
    }
    Ok(Analysis {
        res: a.res,
        funcs: a.funcs,
        for_base: a.for_base,
        main: FuncInfo { nslots: main.max_slot as u16, upvals: Vec::new() },
    })
}

impl<'a> Analyzer<'a> {
    fn error(&mut self, id: NodeId, msg: impl Into<String>) {
        let (len, start) = match self.ast.node(id) {
            Node::Ident(s) => (self.interner.get(s).len() as u32, self.ast.span(id)),
            _ => (1, self.ast.span(id)),
        };
        if self.errors.len() < 20 {
            self.errors.push(Diag::new(msg, start, len));
        }
    }

    fn name(&self, s: Sym) -> &str {
        self.interner.get(s)
    }

    fn fs(&mut self) -> &mut FuncState {
        self.fns.last_mut().unwrap()
    }

    // ---- globals ---------------------------------------------------------------

    /// Pre-declare top-level names so that functions may refer to constants
    /// that are defined later in the file and so that assignments to them are
    /// checked.
    fn hoist_globals(&mut self, root: NodeId) {
        let Node::Block { stmts, .. } = self.ast.node(root) else { return };
        let mut seen: FxHashMap<Sym, ()> = FxHashMap::default();
        for id in self.ast.nodes_of(stmts) {
            let (name, is_const) = match self.ast.node(id) {
                Node::Var { name, .. } => (name, false),
                Node::Const { name, .. } => (name, true),
                Node::Struct { name, .. } => (name, true),
                _ => continue,
            };
            if seen.insert(name, ()).is_some() && !self.repl {
                let msg = format!("'{}' is already declared in this scope", self.name(name));
                self.error(id, msg);
            }
            if is_const {
                self.globals.consts.insert(name);
            } else {
                self.globals.consts.remove(&name);
            }
        }
    }

    // ---- scopes & slots ----------------------------------------------------------

    fn push_scope(&mut self) {
        let fs = self.fs();
        let s = ScopeInfo { first_local: fs.locals.len(), base_slot: fs.next_slot, captured_min: None };
        fs.scopes.push(s);
    }

    /// Pops the innermost scope; returns the lowest captured slot, if any.
    fn pop_scope(&mut self) -> Option<u16> {
        let fs = self.fs();
        let s = fs.scopes.pop().unwrap();
        fs.locals.truncate(s.first_local);
        fs.next_slot = s.base_slot;
        if let (Some(m), Some(parent)) = (s.captured_min, fs.scopes.last_mut()) {
            parent.captured_min = Some(parent.captured_min.map_or(m, |p| p.min(m)));
        }
        s.captured_min
    }

    fn alloc_slot(&mut self, at: NodeId) -> u16 {
        let fs = self.fs();
        let slot = fs.next_slot;
        fs.next_slot += 1;
        if fs.next_slot > fs.max_slot {
            fs.max_slot = fs.next_slot;
        }
        if slot > u16::MAX as u32 - 8 {
            self.error(at, "too many local variables in one function");
            return 0;
        }
        slot as u16
    }

    fn at_top_level(&mut self) -> bool {
        let fs = self.fs();
        fs.is_main && fs.scopes.len() == 1
    }

    /// Declares a variable; returns where it lives.
    fn declare(&mut self, name: Sym, is_const: bool, at: NodeId) -> Res {
        if self.at_top_level() {
            return Res::Global;
        }
        let dup = {
            let fs = self.fns.last().unwrap();
            let first = fs.scopes.last().map_or(0, |s| s.first_local);
            fs.locals[first..].iter().any(|l| l.name == name)
        };
        if dup {
            let msg = format!("'{}' is already declared in this scope", self.name(name));
            self.error(at, msg);
        }
        let slot = self.alloc_slot(at);
        let fs = self.fs();
        let scope = fs.scopes.len() - 1;
        fs.locals.push(Local { name, slot, is_const, scope });
        Res::Local(slot)
    }

    // ---- name resolution ---------------------------------------------------------

    fn find_local(&self, fi: usize, name: Sym) -> Option<usize> {
        self.fns[fi].locals.iter().rposition(|l| l.name == name)
    }

    fn add_upvalue(&mut self, fi: usize, desc: UpvalDesc) -> u16 {
        let ups = &mut self.fns[fi].upvals;
        if let Some(i) = ups.iter().position(|u| *u == desc) {
            return i as u16;
        }
        ups.push(desc);
        (ups.len() - 1) as u16
    }

    fn find_upvalue(&mut self, fi: usize, name: Sym) -> Option<(u16, bool)> {
        if fi == 0 {
            return None;
        }
        let parent = fi - 1;
        if let Some(li) = self.find_local(parent, name) {
            let (slot, is_const, scope) = {
                let l = &self.fns[parent].locals[li];
                (l.slot, l.is_const, l.scope)
            };
            let sc = &mut self.fns[parent].scopes[scope];
            sc.captured_min = Some(sc.captured_min.map_or(slot, |m| m.min(slot)));
            return Some((self.add_upvalue(fi, UpvalDesc { is_local: true, index: slot }), is_const));
        }
        if let Some((idx, is_const)) = self.find_upvalue(parent, name) {
            return Some((self.add_upvalue(fi, UpvalDesc { is_local: false, index: idx }), is_const));
        }
        None
    }

    /// Resolve a name; the bool says whether the binding is a constant.
    fn lookup(&mut self, name: Sym) -> (Res, bool) {
        let fi = self.fns.len() - 1;
        if let Some(li) = self.find_local(fi, name) {
            let l = &self.fns[fi].locals[li];
            return (Res::Local(l.slot), l.is_const);
        }
        if let Some((idx, c)) = self.find_upvalue(fi, name) {
            return (Res::Upvalue(idx), c);
        }
        (Res::Global, self.globals.consts.contains(&name))
    }

    // ---- traversal ---------------------------------------------------------------

    fn root_block(&mut self, id: NodeId) {
        let Node::Block { stmts, tail } = self.ast.node(id) else { unreachable!() };
        self.push_scope(); // scope 0 of main: its declarations are globals
        for s in self.ast.nodes_of(stmts) {
            self.expr(s);
        }
        if tail.is_some() {
            self.expr(tail);
        }
        self.pop_scope();
    }

    fn block(&mut self, id: NodeId) {
        let Node::Block { stmts, tail } = self.ast.node(id) else { unreachable!() };
        self.push_scope();
        for s in self.ast.nodes_of(stmts) {
            self.expr(s);
        }
        if tail.is_some() {
            self.expr(tail);
        }
        if let Some(m) = self.pop_scope() {
            self.res[id.idx()] = Res::Local(m);
        }
    }

    fn function(&mut self, fid: FuncId) {
        let def = self.ast.func(fid);
        let mut fs = FuncState::new(false);
        fs.scopes.push(ScopeInfo { first_local: 0, base_slot: 0, captured_min: None });
        self.fns.push(fs);
        let body = def.body;
        for (i, p) in self.ast.syms_of(def.params).enumerate() {
            let dup = self.fns.last().unwrap().locals.iter().any(|l| l.name == p);
            if dup {
                let msg = format!("duplicate parameter '{}'", self.name(p));
                self.error(body, msg);
            }
            let slot = self.alloc_slot(body);
            debug_assert_eq!(slot as usize, i);
            let fs = self.fs();
            fs.locals.push(Local { name: p, slot, is_const: false, scope: 0 });
        }
        self.expr(body);
        let fs = self.fns.pop().unwrap();
        self.funcs[fid.0 as usize] = FuncInfo { nslots: fs.max_slot as u16, upvals: fs.upvals };
    }

    fn begin_loop(&mut self) {
        self.push_scope();
        self.fs().loop_depth += 1;
    }

    fn end_loop(&mut self, id: NodeId) {
        self.fs().loop_depth -= 1;
        if let Some(m) = self.pop_scope() {
            self.res[id.idx()] = Res::Local(m);
        }
    }

    fn expr(&mut self, id: NodeId) {
        match self.ast.node(id) {
            Node::Nil | Node::True | Node::False | Node::Num(_) | Node::Str { .. } | Node::Key(_) => {}
            Node::Ident(sym) => {
                let (r, _) = self.lookup(sym);
                self.res[id.idx()] = r;
            }
            Node::Unary { expr, .. } => self.expr(expr),
            Node::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Node::Assign { target, value, .. } => {
                match self.ast.node(target) {
                    Node::Ident(sym) => {
                        let (r, is_const) = self.lookup(sym);
                        self.res[target.idx()] = r;
                        if is_const {
                            let msg = format!("cannot assign to constant '{}'", self.name(sym));
                            self.error(target, msg);
                        }
                    }
                    _ => self.expr(target),
                }
                self.expr(value);
            }
            Node::Call { callee, args } => {
                self.expr(callee);
                for a in self.ast.nodes_of(args) {
                    self.expr(a);
                }
            }
            Node::MethodCall { args, .. } => {
                for a in self.ast.nodes_of(args) {
                    self.expr(a);
                }
            }
            Node::Field { obj, .. } => self.expr(obj),
            Node::Index { obj, index } => {
                self.expr(obj);
                self.expr(index);
            }
            Node::Array { items } => {
                for a in self.ast.nodes_of(items) {
                    self.expr(a);
                }
            }
            Node::Table { entries } => {
                for a in self.ast.nodes_of(entries) {
                    self.expr(a);
                }
            }
            Node::Func(f) => self.function(f),
            Node::If { cond, then, els } => {
                self.expr(cond);
                self.expr(then);
                if els.is_some() {
                    self.expr(els);
                }
            }
            Node::Block { .. } => self.block(id),
            Node::Loop { body } => {
                self.begin_loop();
                self.expr(body);
                self.end_loop(id);
            }
            Node::While { cond, body } => {
                self.begin_loop();
                self.expr(cond);
                self.expr(body);
                self.end_loop(id);
            }
            Node::For(fid) => {
                let def = self.ast.for_def(fid);
                // the iterable / bounds are evaluated before the loop scope exists
                self.expr(def.iter);
                if def.end.is_some() {
                    self.expr(def.end);
                }
                self.begin_loop();
                let base = self.alloc_slot(id); // hidden: cursor / table
                let _ = self.alloc_slot(id); // hidden: end / index
                self.for_base[fid.0 as usize] = base;
                for i in 0..def.nvars as usize {
                    // loop variables are fresh per iteration (closures capture each one)
                    let name = def.vars[i];
                    let slot = self.alloc_slot(id);
                    let fs = self.fs();
                    let scope = fs.scopes.len() - 1;
                    fs.locals.push(Local { name, slot, is_const: false, scope });
                }
                self.expr(def.body);
                self.end_loop(id);
            }
            Node::Break(v) => {
                if self.fs().loop_depth == 0 {
                    self.error(id, "`break` outside of a loop");
                }
                if v.is_some() {
                    self.expr(v);
                }
            }
            Node::Continue => {
                if self.fs().loop_depth == 0 {
                    self.error(id, "`continue` outside of a loop");
                }
            }
            Node::Return(v) => {
                if v.is_some() {
                    self.expr(v);
                }
            }
            Node::Var { name, init } | Node::Const { name, init } => {
                let is_const = matches!(self.ast.node(id), Node::Const { .. });
                let is_fn = init.is_some() && matches!(self.ast.node(init), Node::Func(_));
                if is_fn {
                    // function literals may refer to themselves (recursion)
                    let r = self.declare(name, is_const, id);
                    self.res[id.idx()] = r;
                    self.expr(init);
                } else {
                    if init.is_some() {
                        self.expr(init);
                    }
                    let r = self.declare(name, is_const, id);
                    self.res[id.idx()] = r;
                }
            }
            Node::Struct { name, .. } => {
                let r = self.declare(name, true, id);
                self.res[id.idx()] = r;
            }
            Node::Impl { target, methods } => {
                self.expr(target);
                for m in self.ast.nodes_of(methods) {
                    self.expr(m);
                }
            }
        }
    }
}
