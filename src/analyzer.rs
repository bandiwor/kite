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
    /// Slots that some closure captures (so a call may change them behind the code's back).
    pub captured: Vec<u16>,
}

/// Frame slots of one pattern node.
#[derive(Clone, Copy, Default)]
pub struct PatInfo {
    /// Slot holding the value this pattern is tested against.
    pub src: u16,
    /// For `Bind` patterns: the slot of the bound variable (equal to `src` for fields
    /// of a variant pattern, so no copy is needed).
    pub bind: u16,
}

pub struct Analysis {
    /// Per pattern node (`PatId`).
    pub pats: Vec<PatInfo>,
    /// Hidden slot holding the scrutinee of each `match` (`MatchId`).
    pub match_slot: Vec<u16>,
    /// Per match arm: lowest captured slot to close when the arm is left.
    pub arm_close: Vec<Option<u16>>,
    /// This chunk is a module: top-level names are private locals and the chunk
    /// evaluates to its exports table.
    pub module: bool,
    /// Exported names and where they live (module chunks only).
    pub exports: Vec<(Sym, Res)>,
    /// Where each import binds its names: one entry per `import` (a single
    /// binding for `import "p"`, one per name for `import { a, b } from "p"`).
    pub import_res: Vec<Vec<Res>>,
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
    captured: Vec<u16>,
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
            captured: Vec::new(),
            loop_depth: 0,
            is_main,
        }
    }
}

struct TopDecl {
    node: NodeId,
    name: Sym,
    is_const: bool,
    exported: bool,
    import: Option<(ImportId, usize)>,
}

struct Analyzer<'a> {
    pats: Vec<PatInfo>,
    match_slot: Vec<u16>,
    arm_close: Vec<Option<u16>>,
    module_scope_pushed: bool,
    ast: &'a Ast,
    interner: &'a Interner,
    globals: &'a mut GlobalInfo,
    res: Vec<Res>,
    funcs: Vec<FuncInfo>,
    for_base: Vec<u16>,
    fns: Vec<FuncState>,
    errors: Vec<Diag>,
    repl: bool,
    module: bool,
    exports: Vec<(Sym, Res)>,
    import_res: Vec<Vec<Res>>,
}

/// `module`: analyse the chunk as a module (private top-level names, exports).
pub fn analyze(
    ast: &Ast,
    interner: &Interner,
    globals: &mut GlobalInfo,
    repl: bool,
    module: bool,
) -> Result<Analysis, Vec<Diag>> {
    let mut a = Analyzer {
        pats: vec![PatInfo::default(); ast.pats.len()],
        match_slot: vec![0; ast.matches.len()],
        arm_close: vec![None; ast.arms.len()],
        module_scope_pushed: false,
        module,
        exports: Vec::new(),
        import_res: ast
            .imports
            .iter()
            .map(|d| vec![Res::None; if d.module_name.is_none() { d.names.len as usize } else { 1 }])
            .collect(),
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
    if module {
        a.hoist_module(root);
    } else {
        a.hoist_globals(root);
    }
    a.root_block(root);
    let main = a.fns.pop().unwrap();
    if !a.errors.is_empty() {
        return Err(a.errors);
    }
    Ok(Analysis {
        pats: a.pats,
        match_slot: a.match_slot,
        arm_close: a.arm_close,
        module,
        exports: a.exports,
        import_res: a.import_res,
        res: a.res,
        funcs: a.funcs,
        for_base: a.for_base,
        main: FuncInfo { nslots: main.max_slot as u16, upvals: Vec::new(), captured: main.captured },
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

    /// Every name a top-level statement declares:
    /// `(declaring node, name, is_const, exported, which import binding)`.
    fn top_level_decls(&self, root: NodeId) -> Vec<TopDecl> {
        let Node::Block { stmts, .. } = self.ast.node(root) else { return Vec::new() };
        let mut out = Vec::new();
        for id in self.ast.nodes_of(stmts) {
            let (inner, exported) = match self.ast.node(id) {
                Node::Export(inner) => (inner, true),
                _ => (id, false),
            };
            match self.ast.node(inner) {
                Node::Var { name, .. } => {
                    out.push(TopDecl { node: inner, name, is_const: false, exported, import: None })
                }
                Node::Const { name, .. } | Node::Struct { name, .. } | Node::Union { name, .. } => {
                    out.push(TopDecl { node: inner, name, is_const: true, exported, import: None })
                }
                Node::Import(iid) => {
                    let d = self.ast.import_def(iid);
                    if d.module_name.is_none() {
                        for (k, name) in self.ast.syms_of(d.names).enumerate() {
                            out.push(TopDecl { node: inner, name, is_const: true, exported, import: Some((iid, k)) });
                        }
                    } else {
                        out.push(TopDecl {
                            node: inner,
                            name: d.module_name,
                            is_const: true,
                            exported,
                            import: Some((iid, 0)),
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn duplicate_check(&mut self, decls: &[TopDecl]) {
        let mut seen: FxHashMap<Sym, ()> = FxHashMap::default();
        for d in decls {
            if seen.insert(d.name, ()).is_some() && !self.repl {
                let msg = format!("'{}' is already declared in this scope", self.name(d.name));
                self.error(d.node, msg);
            }
        }
    }

    /// Script chunks: pre-declare top-level names as globals so that functions may
    /// refer to constants that are defined later in the file and so that
    /// assignments to them are checked.
    fn hoist_globals(&mut self, root: NodeId) {
        let decls = self.top_level_decls(root);
        self.duplicate_check(&decls);
        for d in &decls {
            if d.is_const {
                self.globals.consts.insert(d.name);
            } else {
                self.globals.consts.remove(&d.name);
            }
            if let Some((iid, k)) = d.import {
                self.import_res[iid.0 as usize][k] = Res::Global;
            }
        }
    }

    /// Module chunks: every top-level name becomes a private local slot of the module
    /// function, allocated up front so that functions defined earlier can refer to
    /// ones defined later (they capture the slot as an upvalue).
    fn hoist_module(&mut self, root: NodeId) {
        let decls = self.top_level_decls(root);
        self.duplicate_check(&decls);
        self.push_scope(); // the module's root scope; `root_block` reuses it
        self.module_scope_pushed = true;
        for d in &decls {
            let slot = self.alloc_slot(d.node);
            let fs = self.fs();
            let scope = fs.scopes.len() - 1;
            fs.locals.push(Local { name: d.name, slot, is_const: d.is_const, scope });
            match d.import {
                Some((iid, k)) => self.import_res[iid.0 as usize][k] = Res::Local(slot),
                None => self.res[d.node.idx()] = Res::Local(slot),
            }
            if d.exported {
                self.exports.push((d.name, Res::Local(slot)));
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
        if slot as usize >= crate::op::MAX_REG - 1 {
            self.error(at, "too many local variables in one function (a function can use 250 registers)");
            return 0;
        }
        slot as u16
    }

    /// Directly in the top-level scope of the chunk (not inside a block or function)?
    fn in_root_scope(&mut self) -> bool {
        let fs = self.fs();
        fs.is_main && fs.scopes.len() == 1
    }

    /// Does a declaration here create a global?  (Script chunks only.)
    fn at_top_level(&mut self) -> bool {
        !self.module && self.in_root_scope()
    }

    /// Declares a variable; returns where it lives.
    fn declare(&mut self, name: Sym, is_const: bool, at: NodeId) -> Res {
        if self.at_top_level() {
            return Res::Global;
        }
        let slot = self.alloc_slot(at);
        self.bind_local(name, is_const, at, slot)
    }

    /// Make `name` visible as the local in register `slot` (reserved by `alloc_slot`).
    fn bind_local(&mut self, name: Sym, is_const: bool, at: NodeId, slot: u16) -> Res {
        let dup = {
            let fs = self.fns.last().unwrap();
            let first = fs.scopes.last().map_or(0, |s| s.first_local);
            fs.locals[first..].iter().any(|l| l.name == name)
        };
        if dup {
            let msg = format!("'{}' is already declared in this scope", self.name(name));
            self.error(at, msg);
        }
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
            self.fns[parent].captured.push(slot);
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
        if !self.module_scope_pushed {
            self.push_scope(); // scope 0 of main: its declarations are globals
        }
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
        self.funcs[fid.0 as usize] = FuncInfo { nslots: fs.max_slot as u16, upvals: fs.upvals, captured: fs.captured };
    }

    /// Bind `name` for a pattern: declare it, or (in later alternatives of an
    /// or-pattern) reuse the slot of the first alternative.
    fn bind_name(
        &mut self,
        name: Sym,
        at: NodeId,
        binds: &mut Vec<(Sym, u16)>,
        declaring: bool,
        used: &mut Vec<Sym>,
    ) -> u16 {
        used.push(name);
        if let Some(&(_, slot)) = binds.iter().find(|b| b.0 == name) {
            if declaring {
                let msg = format!("'{}' is bound more than once in this pattern", self.name(name));
                self.error(at, msg);
            }
            return slot;
        }
        if !declaring {
            let msg = format!("'{}' is not bound in every alternative of the or-pattern", self.name(name));
            self.error(at, msg);
            return 0;
        }
        let slot = match self.declare(name, false, at) {
            Res::Local(s) => s,
            _ => 0,
        };
        binds.push((name, slot));
        slot
    }

    /// Resolve a pattern: allocate the slots it needs and declare its bindings.
    /// `declaring` is false in the second and later alternatives of an or-pattern.
    fn pattern(&mut self, pid: PatId, at: NodeId, binds: &mut Vec<(Sym, u16)>, declaring: bool, used: &mut Vec<Sym>) {
        match self.ast.pat(pid) {
            Pat::Wild | Pat::Num(_) | Pat::Str { .. } | Pat::Bool(_) | Pat::Nil => {}
            Pat::Bind(name) => {
                let slot = self.bind_name(name, at, binds, declaring, used);
                self.pats[pid.0 as usize].bind = slot;
            }
            Pat::Variant { qual, args, .. } => {
                if qual.is_some() {
                    self.expr(qual);
                }
                for a in self.ast.pats_of(args) {
                    match self.ast.pat(a) {
                        Pat::Wild => {}
                        Pat::Bind(name) => {
                            // the variable's slot receives the field directly
                            let slot = self.bind_name(name, at, binds, declaring, used);
                            self.pats[a.0 as usize] = PatInfo { src: slot, bind: slot };
                        }
                        _ => {
                            let tmp = self.alloc_slot(at);
                            self.pats[a.0 as usize].src = tmp;
                            self.pattern(a, at, binds, declaring, used);
                        }
                    }
                }
            }
            Pat::Or(alts) => {
                let src = self.pats[pid.0 as usize].src;
                let mut first: Vec<Sym> = Vec::new();
                for (k, alt) in self.ast.pats_of(alts).enumerate() {
                    self.pats[alt.0 as usize].src = src;
                    let mut names = Vec::new();
                    self.pattern(alt, at, binds, declaring && k == 0, &mut names);
                    names.sort();
                    names.dedup();
                    if k == 0 {
                        first = names;
                    } else if names != first {
                        self.error(at, "all alternatives of an or-pattern must bind the same names");
                    }
                }
                used.extend(first);
            }
        }
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
            Node::Try(e) => {
                if self.module && self.fns.len() == 1 {
                    self.error(id, "`?` is not allowed at the top level of a module (it returns from the function)");
                }
                self.expr(e);
            }
            Node::Return(v) => {
                if self.module && self.fns.len() == 1 {
                    self.error(id, "`return` is not allowed at the top level of a module");
                }
                if v.is_some() {
                    self.expr(v);
                }
            }
            Node::Var { name, init } | Node::Const { name, init } => {
                let is_const = matches!(self.ast.node(id), Node::Const { .. });
                let is_fn = init.is_some() && matches!(self.ast.node(init), Node::Func(_));
                if self.res[id.idx()] != Res::None {
                    // a module's top-level name, declared up front by `hoist_module`
                    if init.is_some() {
                        self.expr(init);
                    }
                } else if is_fn {
                    // function literals may refer to themselves (recursion)
                    let r = self.declare(name, is_const, id);
                    self.res[id.idx()] = r;
                    self.expr(init);
                } else {
                    // The register is reserved *before* the initialiser, which lets the compiler
                    // evaluate it straight into the variable.  The name only becomes visible
                    // afterwards (`var x = x + 1` still refers to the outer `x`).
                    let reserved = if self.at_top_level() { None } else { Some(self.alloc_slot(id)) };
                    if init.is_some() {
                        self.expr(init);
                    }
                    let r = match reserved {
                        Some(slot) => self.bind_local(name, is_const, id, slot),
                        None => Res::Global,
                    };
                    self.res[id.idx()] = r;
                }
            }
            Node::Struct { name, .. } | Node::Union { name, .. } => {
                if self.res[id.idx()] == Res::None {
                    let r = self.declare(name, true, id);
                    self.res[id.idx()] = r;
                }
            }
            Node::Match(mid) => {
                let scrutinee = self.ast.match_def(mid).scrutinee;
                self.expr(scrutinee);
                self.push_scope();
                let s = self.alloc_slot(id); // the scrutinee lives here while the arms test it
                self.match_slot[mid.0 as usize] = s;
                for k in self.ast.arms_of(mid) {
                    let (pat, guard, body) = {
                        let arm = &self.ast.arms[k];
                        (arm.pat, arm.guard, arm.body)
                    };
                    self.push_scope();
                    let mut binds = Vec::new();
                    let mut used = Vec::new();
                    self.pats[pat.0 as usize].src = s;
                    self.pattern(pat, id, &mut binds, true, &mut used);
                    if guard.is_some() {
                        self.expr(guard);
                    }
                    self.expr(body);
                    self.arm_close[k] = self.pop_scope();
                }
                self.pop_scope();
            }
            Node::Import(_) => {
                if !self.in_root_scope() {
                    self.error(id, "`import` is only allowed at the top level of a file");
                }
            }
            Node::Export(inner) => {
                if !self.in_root_scope() {
                    self.error(id, "`export` is only allowed at the top level of a file");
                }
                self.expr(inner);
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
