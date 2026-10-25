//! Thin AST.
//!
//! All nodes live in one `Vec<Node>` arena and refer to each other by `NodeId`
//! (a `u32`).  Variable-length children (arguments, statements, table
//! entries ...) are stored in a shared side array and addressed by a
//! `ListRef`.  A node is 16 bytes; positions are kept in a parallel `Vec<u32>`
//! so the hot data stays small.  Identifiers are interned `Sym`s; string
//! literals are just ranges into the source text.

use crate::intern::{Interner, Sym};
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NONE: NodeId = NodeId(u32::MAX);
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
    #[inline]
    pub fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FuncId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ForId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImportId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PatId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatchId(pub u32);

/// A slice of the shared side list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListRef {
    pub start: u32,
    pub len: u32,
}

impl ListRef {
    pub const EMPTY: ListRef = ListRef { start: 0, len: 0 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug)]
pub enum Node {
    Nil,
    True,
    False,
    Num(f64),
    /// Raw literal text between the quotes (escapes still unprocessed).
    Str {
        start: u32,
        len: u32,
    },
    Ident(Sym),
    /// Bare key inside a table literal (`{ name: 1 }`).
    Key(Sym),
    Unary {
        op: UnOp,
        expr: NodeId,
    },
    Binary {
        op: BinOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// `target = value` or, when `op` is set, `target op= value`.
    Assign {
        op: Option<BinOp>,
        target: NodeId,
        value: NodeId,
    },
    /// `args[0..]` are the arguments.
    Call {
        callee: NodeId,
        args: ListRef,
    },
    /// `recv:name(args)`; `args[0]` is the receiver.
    MethodCall {
        name: Sym,
        args: ListRef,
    },
    Field {
        obj: NodeId,
        name: Sym,
    },
    Index {
        obj: NodeId,
        index: NodeId,
    },
    /// Array literal; `items` are node ids.
    Array {
        items: ListRef,
    },
    /// Table literal; `entries` is `[key, value, key, value, ...]`.
    Table {
        entries: ListRef,
    },
    Func(FuncId),
    If {
        cond: NodeId,
        then: NodeId,
        els: NodeId,
    },
    Block {
        stmts: ListRef,
        tail: NodeId,
    },
    Loop {
        body: NodeId,
    },
    While {
        cond: NodeId,
        body: NodeId,
    },
    For(ForId),
    Break(NodeId),
    Continue,
    Return(NodeId),
    /// `expr?`: unwrap an `Ok`/`Some`, or return the `Err`/`None` from the function.
    Try(NodeId),
    Var {
        name: Sym,
        init: NodeId,
    },
    Const {
        name: Sym,
        init: NodeId,
    },
    /// `struct Name { fields }`; `fields` are raw `Sym` values.
    Struct {
        name: Sym,
        fields: ListRef,
    },
    /// `impl Target { methods }`; `methods` are `Func` node ids.
    Impl {
        target: NodeId,
        methods: ListRef,
    },
    /// `union Name { Variant(field, ...), Unit, ... }`; `variants` are indices into `Ast::variants`.
    Union {
        name: Sym,
        variants: ListRef,
    },
    /// `match scrutinee { pattern [if guard] => body, ... }`
    Match(MatchId),
    /// `import "path" [as name]` / `import { a, b } from "path"`.
    Import(ImportId),
    /// `export <var | const | fn | struct declaration>`.
    Export(NodeId),
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

pub struct FuncDef {
    pub name: Sym,
    /// Raw `Sym` values.
    pub params: ListRef,
    pub body: NodeId,
}

/// One variant of a `union` declaration.
pub struct VariantDef {
    pub name: Sym,
    /// Payload field names (raw `Sym` values); their count is the variant's arity.
    pub fields: ListRef,
}

/// Patterns live in their own arena: they are not expressions.
#[derive(Clone, Copy, Debug)]
pub enum Pat {
    /// `_`
    Wild,
    /// A lower-case identifier: binds the matched value.
    Bind(Sym),
    Num(f64),
    /// Raw literal text between the quotes.
    Str {
        start: u32,
        len: u32,
    },
    Bool(bool),
    Nil,
    /// `Name`, `Name(p1, p2)`, `Type.Name(..)`.  Matches a union variant (or struct) by name;
    /// `qual` (when set) is an expression that must evaluate to the very type.
    /// Without parentheses any number of fields is accepted.
    Variant {
        qual: NodeId,
        name: Sym,
        args: ListRef,
        parens: bool,
    },
    /// `p1 | p2 | ...` (alternatives are pattern ids)
    Or(ListRef),
}

pub struct Arm {
    pub pat: PatId,
    /// `NodeId::NONE` without a guard.
    pub guard: NodeId,
    pub body: NodeId,
}

pub struct MatchDef {
    pub scrutinee: NodeId,
    pub arms_start: u32,
    pub arms_len: u32,
}

pub struct ImportDef {
    /// Source range of the path literal (without quotes, escapes unprocessed).
    pub path_start: u32,
    pub path_len: u32,
    /// Binding for the whole module (`Sym::NONE` for a named import list).
    pub module_name: Sym,
    /// Raw `Sym` values of `import { a, b }`.
    pub names: ListRef,
}

pub struct ForDef {
    pub vars: [Sym; 2],
    pub nvars: u8,
    /// Iterable, or range start when `end` is set.
    pub iter: NodeId,
    pub end: NodeId,
    pub body: NodeId,
}

#[derive(Default)]
pub struct Ast {
    pub nodes: Vec<Node>,
    pub spans: Vec<u32>,
    pub lists: Vec<u32>,
    pub funcs: Vec<FuncDef>,
    pub fors: Vec<ForDef>,
    pub imports: Vec<ImportDef>,
    pub variants: Vec<VariantDef>,
    pub pats: Vec<Pat>,
    pub pat_spans: Vec<u32>,
    pub arms: Vec<Arm>,
    pub matches: Vec<MatchDef>,
    pub root: Option<NodeId>,
}

impl Ast {
    pub fn new() -> Ast {
        Ast::default()
    }

    #[inline]
    pub fn push(&mut self, node: Node, pos: u32) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.spans.push(pos);
        id
    }

    #[inline]
    pub fn node(&self, id: NodeId) -> Node {
        self.nodes[id.idx()]
    }

    #[inline]
    pub fn span(&self, id: NodeId) -> u32 {
        self.spans[id.idx()]
    }

    pub fn add_list(&mut self, items: &[u32]) -> ListRef {
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(items);
        ListRef { start, len: items.len() as u32 }
    }

    #[inline]
    pub fn list(&self, r: ListRef) -> &[u32] {
        &self.lists[r.start as usize..(r.start + r.len) as usize]
    }

    pub fn nodes_of(&self, r: ListRef) -> impl Iterator<Item = NodeId> + '_ {
        self.list(r).iter().map(|&i| NodeId(i))
    }

    pub fn syms_of(&self, r: ListRef) -> impl Iterator<Item = Sym> + '_ {
        self.list(r).iter().map(|&i| Sym(i))
    }

    pub fn add_func(&mut self, f: FuncDef) -> FuncId {
        self.funcs.push(f);
        FuncId(self.funcs.len() as u32 - 1)
    }

    pub fn add_for(&mut self, f: ForDef) -> ForId {
        self.fors.push(f);
        ForId(self.fors.len() as u32 - 1)
    }

    pub fn add_pat(&mut self, p: Pat, pos: u32) -> PatId {
        self.pats.push(p);
        self.pat_spans.push(pos);
        PatId(self.pats.len() as u32 - 1)
    }

    pub fn pat(&self, id: PatId) -> Pat {
        self.pats[id.0 as usize]
    }

    pub fn pats_of(&self, r: ListRef) -> impl Iterator<Item = PatId> + '_ {
        self.list(r).iter().map(|&i| PatId(i))
    }

    pub fn match_def(&self, id: MatchId) -> &MatchDef {
        &self.matches[id.0 as usize]
    }

    pub fn arms_of(&self, id: MatchId) -> std::ops::Range<usize> {
        let m = self.match_def(id);
        m.arms_start as usize..(m.arms_start + m.arms_len) as usize
    }

    pub fn add_import(&mut self, i: ImportDef) -> ImportId {
        self.imports.push(i);
        ImportId(self.imports.len() as u32 - 1)
    }

    pub fn import_def(&self, id: ImportId) -> &ImportDef {
        &self.imports[id.0 as usize]
    }

    pub fn func(&self, id: FuncId) -> &FuncDef {
        &self.funcs[id.0 as usize]
    }

    pub fn for_def(&self, id: ForId) -> &ForDef {
        &self.fors[id.0 as usize]
    }

    /// S-expression dump used by `--dump-ast` and in tests.
    pub fn dump(&self, interner: &Interner, src: &str) -> String {
        let mut out = String::new();
        if let Some(root) = self.root {
            self.dump_node(root, interner, src, 0, &mut out);
        }
        out
    }

    fn dump_pat(&self, id: PatId, i: &Interner, src: &str) -> String {
        match self.pat(id) {
            Pat::Wild => "_".into(),
            Pat::Bind(s) => i.get(s).to_string(),
            Pat::Num(n) => format!("{}", n),
            Pat::Str { start, len } => format!("{:?}", &src[start as usize..(start + len) as usize]),
            Pat::Bool(b) => b.to_string(),
            Pat::Nil => "nil".into(),
            Pat::Variant { qual, name, args, parens } => {
                let q = if qual.is_some() { "T." } else { "" };
                let a: Vec<String> = self.pats_of(args).map(|p| self.dump_pat(p, i, src)).collect();
                if parens {
                    format!("{}{}({})", q, i.get(name), a.join(", "))
                } else {
                    format!("{}{}", q, i.get(name))
                }
            }
            Pat::Or(alts) => self.pats_of(alts).map(|p| self.dump_pat(p, i, src)).collect::<Vec<_>>().join(" | "),
        }
    }

    fn dump_node(&self, id: NodeId, i: &Interner, src: &str, ind: usize, out: &mut String) {
        let pad = "  ".repeat(ind);
        if id.is_none() {
            let _ = writeln!(out, "{}<none>", pad);
            return;
        }
        let n = self.node(id);
        macro_rules! line {
            ($($a:tt)*) => {{ let _ = write!(out, "{}", pad); let _ = writeln!(out, $($a)*); }};
        }
        match n {
            Node::Nil => line!("nil"),
            Node::True => line!("true"),
            Node::False => line!("false"),
            Node::Num(v) => line!("{}", v),
            Node::Str { start, len } => line!("{:?}", &src[start as usize..(start + len) as usize]),
            Node::Ident(s) => line!("{}", i.get(s)),
            Node::Key(s) => line!("key {}", i.get(s)),
            Node::Unary { op, expr } => {
                line!("(unary {:?}", op);
                self.dump_node(expr, i, src, ind + 1, out);
                line!(")");
            }
            Node::Binary { op, lhs, rhs } => {
                line!("(binary {:?}", op);
                self.dump_node(lhs, i, src, ind + 1, out);
                self.dump_node(rhs, i, src, ind + 1, out);
                line!(")");
            }
            Node::Assign { op, target, value } => {
                line!("(assign {:?}", op);
                self.dump_node(target, i, src, ind + 1, out);
                self.dump_node(value, i, src, ind + 1, out);
                line!(")");
            }
            Node::Call { callee, args } => {
                line!("(call");
                self.dump_node(callee, i, src, ind + 1, out);
                for a in self.nodes_of(args) {
                    self.dump_node(a, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::MethodCall { name, args } => {
                line!("(method {}", i.get(name));
                for a in self.nodes_of(args) {
                    self.dump_node(a, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Field { obj, name } => {
                line!("(field {}", i.get(name));
                self.dump_node(obj, i, src, ind + 1, out);
                line!(")");
            }
            Node::Index { obj, index } => {
                line!("(index");
                self.dump_node(obj, i, src, ind + 1, out);
                self.dump_node(index, i, src, ind + 1, out);
                line!(")");
            }
            Node::Array { items } => {
                line!("(array");
                for a in self.nodes_of(items) {
                    self.dump_node(a, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Table { entries } => {
                line!("(table");
                for a in self.nodes_of(entries) {
                    self.dump_node(a, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Func(f) => {
                let d = self.func(f);
                let params: Vec<&str> = self.syms_of(d.params).map(|s| i.get(s)).collect();
                let name = if d.name.is_none() { "<anon>" } else { i.get(d.name) };
                line!("(fn {} ({})", name, params.join(" "));
                self.dump_node(d.body, i, src, ind + 1, out);
                line!(")");
            }
            Node::If { cond, then, els } => {
                line!("(if");
                self.dump_node(cond, i, src, ind + 1, out);
                self.dump_node(then, i, src, ind + 1, out);
                if els.is_some() {
                    self.dump_node(els, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Block { stmts, tail } => {
                line!("(block");
                for a in self.nodes_of(stmts) {
                    self.dump_node(a, i, src, ind + 1, out);
                }
                if tail.is_some() {
                    line!("  tail:");
                    self.dump_node(tail, i, src, ind + 2, out);
                }
                line!(")");
            }
            Node::Loop { body } => {
                line!("(loop");
                self.dump_node(body, i, src, ind + 1, out);
                line!(")");
            }
            Node::While { cond, body } => {
                line!("(while");
                self.dump_node(cond, i, src, ind + 1, out);
                self.dump_node(body, i, src, ind + 1, out);
                line!(")");
            }
            Node::For(f) => {
                let d = self.for_def(f);
                let vars: Vec<&str> = d.vars[..d.nvars as usize].iter().map(|&s| i.get(s)).collect();
                line!("(for {} {}", vars.join(","), if d.end.is_some() { "range" } else { "in" });
                self.dump_node(d.iter, i, src, ind + 1, out);
                if d.end.is_some() {
                    self.dump_node(d.end, i, src, ind + 1, out);
                }
                self.dump_node(d.body, i, src, ind + 1, out);
                line!(")");
            }
            Node::Break(v) => {
                line!("(break");
                if v.is_some() {
                    self.dump_node(v, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Continue => line!("(continue)"),
            Node::Try(e) => {
                line!("(try");
                self.dump_node(e, i, src, ind + 1, out);
                line!(")");
            }
            Node::Return(v) => {
                line!("(return");
                if v.is_some() {
                    self.dump_node(v, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Var { name, init } => {
                line!("(var {}", i.get(name));
                if init.is_some() {
                    self.dump_node(init, i, src, ind + 1, out);
                }
                line!(")");
            }
            Node::Const { name, init } => {
                line!("(const {}", i.get(name));
                self.dump_node(init, i, src, ind + 1, out);
                line!(")");
            }
            Node::Struct { name, fields } => {
                let fs: Vec<&str> = self.syms_of(fields).map(|s| i.get(s)).collect();
                line!("(struct {} [{}])", i.get(name), fs.join(" "));
            }
            Node::Union { name, variants } => {
                let vs: Vec<String> = self
                    .list(variants)
                    .iter()
                    .map(|&v| {
                        let d = &self.variants[v as usize];
                        let fs: Vec<&str> = self.syms_of(d.fields).map(|s| i.get(s)).collect();
                        format!("{}({})", i.get(d.name), fs.join(" "))
                    })
                    .collect();
                line!("(union {} [{}])", i.get(name), vs.join(" "));
            }
            Node::Match(mid) => {
                line!("(match");
                self.dump_node(self.match_def(mid).scrutinee, i, src, ind + 1, out);
                for k in self.arms_of(mid) {
                    let arm = &self.arms[k];
                    line!("  (arm {}", self.dump_pat(arm.pat, i, src));
                    if arm.guard.is_some() {
                        line!("    guard:");
                        self.dump_node(arm.guard, i, src, ind + 3, out);
                    }
                    self.dump_node(arm.body, i, src, ind + 2, out);
                    line!("  )");
                }
                line!(")");
            }
            Node::Import(id) => {
                let d = self.import_def(id);
                let path = &src[d.path_start as usize..(d.path_start + d.path_len) as usize];
                if d.module_name.is_none() {
                    let names: Vec<&str> = self.syms_of(d.names).map(|s| i.get(s)).collect();
                    line!("(import {:?} {{{}}})", path, names.join(" "));
                } else {
                    line!("(import {:?} as {})", path, i.get(d.module_name));
                }
            }
            Node::Export(inner) => {
                line!("(export");
                self.dump_node(inner, i, src, ind + 1, out);
                line!(")");
            }
            Node::Impl { target, methods } => {
                line!("(impl");
                self.dump_node(target, i, src, ind + 1, out);
                for m in self.nodes_of(methods) {
                    self.dump_node(m, i, src, ind + 1, out);
                }
                line!(")");
            }
        }
    }
}
