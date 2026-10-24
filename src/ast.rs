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
    Str { start: u32, len: u32 },
    Ident(Sym),
    /// Bare key inside a table literal (`{ name: 1 }`).
    Key(Sym),
    Unary { op: UnOp, expr: NodeId },
    Binary { op: BinOp, lhs: NodeId, rhs: NodeId },
    /// `target = value` or, when `op` is set, `target op= value`.
    Assign { op: Option<BinOp>, target: NodeId, value: NodeId },
    /// `args[0..]` are the arguments.
    Call { callee: NodeId, args: ListRef },
    /// `recv:name(args)`; `args[0]` is the receiver.
    MethodCall { name: Sym, args: ListRef },
    Field { obj: NodeId, name: Sym },
    Index { obj: NodeId, index: NodeId },
    /// Array literal; `items` are node ids.
    Array { items: ListRef },
    /// Table literal; `entries` is `[key, value, key, value, ...]`.
    Table { entries: ListRef },
    Func(FuncId),
    If { cond: NodeId, then: NodeId, els: NodeId },
    Block { stmts: ListRef, tail: NodeId },
    Loop { body: NodeId },
    While { cond: NodeId, body: NodeId },
    For(ForId),
    Break(NodeId),
    Continue,
    Return(NodeId),
    Var { name: Sym, init: NodeId },
    Const { name: Sym, init: NodeId },
    /// `struct Name { fields }`; `fields` are raw `Sym` values.
    Struct { name: Sym, fields: ListRef },
    /// `impl Target { methods }`; `methods` are `Func` node ids.
    Impl { target: NodeId, methods: ListRef },
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

pub struct FuncDef {
    pub name: Sym,
    /// Raw `Sym` values.
    pub params: ListRef,
    pub body: NodeId,
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
