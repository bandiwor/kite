//! Pratt parser producing the thin arena AST.
//!
//! Tricks used to keep it fast:
//!  * the lexer already ran to completion, the parser indexes a `&[Token]`
//!    (no lookahead buffering, `Eof` sentinel at the end);
//!  * binding powers come from one `match` (compiled to a jump table);
//!  * child lists are built on a shared scratch stack and copied once into
//!    the AST's side array (no per-node `Vec`);
//!  * identifiers are interned immediately, literals stay as source ranges.

use crate::ast::*;
use crate::intern::{Interner, Sym};
use crate::source::Diag;
use crate::text::{parse_number, unescape};
use crate::token::{Token, TokenKind, TokenKind as T};

type PResult<T> = Result<T, ()>;

const MAX_DEPTH: u32 = 200;
const MAX_ERRORS: usize = 10;

pub struct Parser<'a> {
    src: &'a str,
    toks: &'a [Token],
    pos: usize,
    interner: &'a mut Interner,
    ast: Ast,
    scratch: Vec<u32>,
    errors: Vec<Diag>,
    depth: u32,
}

/// Parse a whole program.  The root is a `Block` whose tail (if any) is the
/// value of the program.
pub fn parse(src: &str, toks: &[Token], interner: &mut Interner) -> Result<Ast, Vec<Diag>> {
    let mut p = Parser {
        src,
        toks,
        pos: 0,
        interner,
        ast: Ast::new(),
        scratch: Vec::new(),
        errors: Vec::new(),
        depth: 0,
    };
    p.ast.nodes.reserve(toks.len() / 2);
    p.ast.spans.reserve(toks.len() / 2);
    let root = p.parse_program();
    if p.errors.is_empty() {
        p.ast.root = Some(root);
        Ok(p.ast)
    } else {
        Err(p.errors)
    }
}

// ---- binding powers --------------------------------------------------------

const BP_ASSIGN: u8 = 10;
const BP_UNARY: u8 = 120;
const BP_POSTFIX: u8 = 140;

#[inline]
fn infix_bp(k: TokenKind) -> Option<(u8, u8)> {
    Some(match k {
        T::Eq
        | T::PlusEq
        | T::MinusEq
        | T::StarEq
        | T::SlashEq
        | T::PercentEq
        | T::StarStarEq
        | T::AmpEq
        | T::PipeEq
        | T::CaretEq
        | T::ShlEq
        | T::ShrEq => (BP_ASSIGN, BP_ASSIGN),
        T::PipePipe => (20, 21),
        T::AmpAmp => (30, 31),
        T::EqEq | T::BangEq => (40, 41),
        T::Lt | T::Le | T::Gt | T::Ge => (50, 51),
        T::Pipe => (60, 61),
        T::Caret => (70, 71),
        T::Amp => (80, 81),
        T::Shl | T::Shr => (90, 91),
        T::Plus | T::Minus => (100, 101),
        T::Star | T::Slash | T::Percent => (110, 111),
        T::StarStar => (130, 130),
        T::LParen | T::LBracket | T::Dot | T::Colon => (BP_POSTFIX, BP_POSTFIX),
        _ => return None,
    })
}

fn binop(k: TokenKind) -> Option<BinOp> {
    Some(match k {
        T::Plus | T::PlusEq => BinOp::Add,
        T::Minus | T::MinusEq => BinOp::Sub,
        T::Star | T::StarEq => BinOp::Mul,
        T::Slash | T::SlashEq => BinOp::Div,
        T::Percent | T::PercentEq => BinOp::Rem,
        T::StarStar | T::StarStarEq => BinOp::Pow,
        T::Amp | T::AmpEq => BinOp::BitAnd,
        T::Pipe | T::PipeEq => BinOp::BitOr,
        T::Caret | T::CaretEq => BinOp::BitXor,
        T::Shl | T::ShlEq => BinOp::Shl,
        T::Shr | T::ShrEq => BinOp::Shr,
        T::EqEq => BinOp::Eq,
        T::BangEq => BinOp::Ne,
        T::Lt => BinOp::Lt,
        T::Le => BinOp::Le,
        T::Gt => BinOp::Gt,
        T::Ge => BinOp::Ge,
        T::AmpAmp => BinOp::And,
        T::PipePipe => BinOp::Or,
        _ => return None,
    })
}

fn is_assign(k: TokenKind) -> bool {
    matches!(
        k,
        T::Eq
            | T::PlusEq
            | T::MinusEq
            | T::StarEq
            | T::SlashEq
            | T::PercentEq
            | T::StarStarEq
            | T::AmpEq
            | T::PipeEq
            | T::CaretEq
            | T::ShlEq
            | T::ShrEq
    )
}

/// Can a token begin an expression?  (Used for `return`/`break` operands.)
fn starts_expr(k: TokenKind) -> bool {
    matches!(
        k,
        T::Ident
            | T::Number
            | T::String
            | T::Nil
            | T::True
            | T::False
            | T::LParen
            | T::LBracket
            | T::LBrace
            | T::Pipe
            | T::PipePipe
            | T::Fn
            | T::If
            | T::While
            | T::Loop
            | T::For
            | T::Do
            | T::Minus
            | T::Bang
            | T::Tilde
            | T::Return
            | T::Break
            | T::Continue
    )
}

impl<'a> Parser<'a> {
    // ---- token helpers -------------------------------------------------------

    #[inline(always)]
    fn tok(&self) -> Token {
        self.toks[self.pos]
    }

    #[inline(always)]
    fn kind(&self) -> TokenKind {
        self.toks[self.pos].kind()
    }

    #[inline(always)]
    fn kind_at(&self, off: usize) -> TokenKind {
        let i = (self.pos + off).min(self.toks.len() - 1);
        self.toks[i].kind()
    }

    #[inline(always)]
    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos];
        if t.kind() != T::Eof {
            self.pos += 1;
        }
        t
    }

    #[inline(always)]
    fn eat(&mut self, k: TokenKind) -> bool {
        if self.kind() == k {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn text(&self, t: Token) -> &'a str {
        &self.src[t.start as usize..t.end()]
    }

    fn pos_here(&self) -> u32 {
        self.tok().start
    }

    fn error_at(&mut self, t: Token, msg: String) {
        if self.errors.len() < MAX_ERRORS {
            // avoid piling up several errors at the same position
            if self.errors.last().map_or(true, |d| d.start != t.start) {
                self.errors.push(Diag::new(msg, t.start, t.len() as u32));
            }
        }
    }

    fn err<R>(&mut self, msg: impl Into<String>) -> PResult<R> {
        let t = self.tok();
        self.error_at(t, msg.into());
        Err(())
    }

    fn describe_current(&self) -> String {
        let t = self.tok();
        match t.kind() {
            T::Ident | T::Number => format!("`{}`", self.text(t)),
            T::Error => {
                let s = self.text(t);
                if s.starts_with('"') || s.starts_with('\'') {
                    "unterminated string".to_string()
                } else if s.starts_with("/*") {
                    "unterminated comment".to_string()
                } else {
                    format!("invalid character `{}`", s)
                }
            }
            k => k.describe().to_string(),
        }
    }

    fn expect(&mut self, k: TokenKind, what: &str) -> PResult<Token> {
        if self.kind() == k {
            Ok(self.advance())
        } else {
            let found = self.describe_current();
            self.err(format!("expected {} {}, found {}", k.describe(), what, found))
        }
    }

    fn expect_ident(&mut self, what: &str) -> PResult<Sym> {
        if self.kind() == T::Ident {
            let t = self.advance();
            Ok(self.interner.intern(self.text(t)))
        } else {
            let found = self.describe_current();
            self.err(format!("expected identifier {}, found {}", what, found))
        }
    }

    fn enter(&mut self) -> PResult<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err("expression nested too deeply");
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    // ---- error recovery ------------------------------------------------------

    fn synchronize(&mut self) {
        let start = self.pos;
        loop {
            match self.kind() {
                T::Eof => return,
                T::Semi => {
                    self.pos += 1;
                    return;
                }
                T::RBrace => return,
                T::Var | T::Const | T::Fn | T::Struct | T::Impl | T::If | T::While | T::Loop | T::For
                | T::Return | T::Break | T::Continue
                    if self.pos > start =>
                {
                    return
                }
                _ => self.pos += 1,
            }
        }
    }

    // ---- program & blocks ----------------------------------------------------

    fn parse_program(&mut self) -> NodeId {
        let pos = 0;
        let mark = self.scratch.len();
        let mut tail = NodeId::NONE;
        while self.kind() != T::Eof {
            if self.kind() == T::RBrace {
                let _ = self.err::<()>("unexpected `}`");
                self.pos += 1;
                continue;
            }
            let (depth, scratch_len) = (self.depth, self.scratch.len());
            match self.parse_stmt(true) {
                Ok(StmtResult::Stmt(n)) => self.scratch.push(n.0),
                Ok(StmtResult::Tail(n)) => {
                    tail = n;
                    break;
                }
                Err(()) => {
                    self.depth = depth;
                    self.scratch.truncate(scratch_len);
                    if self.errors.len() >= MAX_ERRORS {
                        break;
                    }
                    self.synchronize();
                }
            }
        }
        let stmts = self.finish_list(mark);
        self.ast.push(Node::Block { stmts, tail }, pos)
    }

    fn finish_list(&mut self, mark: usize) -> ListRef {
        let r = self.ast.add_list(&self.scratch[mark..]);
        self.scratch.truncate(mark);
        r
    }

    fn parse_block(&mut self) -> PResult<NodeId> {
        let open = self.expect(T::LBrace, "to start a block")?;
        self.enter()?;
        let mark = self.scratch.len();
        let mut tail = NodeId::NONE;
        let mut failed = false;
        loop {
            match self.kind() {
                T::RBrace => break,
                T::Eof => {
                    let _ = self.err::<()>("unclosed block, expected `}`");
                    failed = true;
                    break;
                }
                _ => {}
            }
            let (depth, scratch_len) = (self.depth, self.scratch.len());
            match self.parse_stmt(false) {
                Ok(StmtResult::Stmt(n)) => self.scratch.push(n.0),
                Ok(StmtResult::Tail(n)) => {
                    tail = n;
                    break;
                }
                Err(()) => {
                    self.depth = depth;
                    self.scratch.truncate(scratch_len);
                    if self.errors.len() >= MAX_ERRORS {
                        failed = true;
                        break;
                    }
                    self.synchronize();
                    if self.kind() == T::Eof {
                        failed = true;
                        break;
                    }
                }
            }
        }
        let stmts = self.finish_list(mark);
        self.leave();
        if failed || !self.eat(T::RBrace) {
            if !failed {
                let found = self.describe_current();
                return self.err(format!("expected `}}` to close block, found {}", found));
            }
            return Err(());
        }
        Ok(self.ast.push(Node::Block { stmts, tail }, open.start))
    }

    // ---- statements ----------------------------------------------------------

    fn parse_stmt(&mut self, top: bool) -> PResult<StmtResult> {
        let start = self.pos_here();
        match self.kind() {
            T::Semi => {
                self.pos += 1;
                // an empty statement: represent as a harmless `nil` expression statement
                Ok(StmtResult::Stmt(self.ast.push(Node::Nil, start)))
            }
            T::Var => {
                self.pos += 1;
                let name = self.expect_ident("after `var`")?;
                let init = if self.eat(T::Eq) { self.parse_expr()? } else { NodeId::NONE };
                self.expect_semi("after variable declaration")?;
                Ok(StmtResult::Stmt(self.ast.push(Node::Var { name, init }, start)))
            }
            T::Const => {
                self.pos += 1;
                let name = self.expect_ident("after `const`")?;
                if !self.eat(T::Eq) {
                    let found = self.describe_current();
                    return self.err(format!("expected `=` in const declaration (constants need a value), found {}", found));
                }
                let init = self.parse_expr()?;
                self.name_function(init, name);
                self.expect_semi("after constant declaration")?;
                Ok(StmtResult::Stmt(self.ast.push(Node::Const { name, init }, start)))
            }
            T::Fn if self.kind_at(1) == T::Ident => {
                self.pos += 1;
                let name = self.expect_ident("after `fn`")?;
                let f = self.parse_fn_rest(name, start)?;
                Ok(StmtResult::Stmt(self.ast.push(Node::Const { name, init: f }, start)))
            }
            T::Struct => {
                self.pos += 1;
                let name = self.expect_ident("after `struct`")?;
                self.expect(T::LBrace, "after struct name")?;
                let mark = self.scratch.len();
                while self.kind() != T::RBrace {
                    if self.kind() == T::Eof {
                        self.scratch.truncate(mark);
                        return self.err("unclosed struct declaration");
                    }
                    let f = self.expect_ident("as struct field")?;
                    self.scratch.push(f.0);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                let fields = self.finish_list(mark);
                self.expect(T::RBrace, "to close struct")?;
                Ok(StmtResult::Stmt(self.ast.push(Node::Struct { name, fields }, start)))
            }
            T::Impl => {
                self.pos += 1;
                let tpos = self.pos_here();
                let tname = self.expect_ident("after `impl`")?;
                let target = self.ast.push(Node::Ident(tname), tpos);
                self.expect(T::LBrace, "after impl target")?;
                let mark = self.scratch.len();
                while self.kind() != T::RBrace {
                    if self.kind() != T::Fn {
                        self.scratch.truncate(mark);
                        let found = self.describe_current();
                        return self.err(format!("expected `fn` inside impl block, found {}", found));
                    }
                    let fpos = self.pos_here();
                    self.pos += 1;
                    let mname = self.expect_ident("as method name")?;
                    let f = self.parse_fn_rest(mname, fpos)?;
                    self.scratch.push(f.0);
                }
                let methods = self.finish_list(mark);
                self.expect(T::RBrace, "to close impl block")?;
                Ok(StmtResult::Stmt(self.ast.push(Node::Impl { target, methods }, start)))
            }
            T::If | T::While | T::Loop | T::For | T::LBrace | T::Do => {
                // A block-like expression in statement position ends the statement
                // by itself (like Rust).  `{` here is a block, never a table literal.
                let e = if self.kind() == T::LBrace { self.parse_block()? } else { self.parse_prefix()? };
                if self.eat(T::Semi) {
                    return Ok(StmtResult::Stmt(e));
                }
                if self.kind() == T::RBrace || (top && self.kind() == T::Eof) {
                    return Ok(StmtResult::Tail(e));
                }
                Ok(StmtResult::Stmt(e))
            }
            _ => {
                let e = self.parse_expr()?;
                if self.eat(T::Semi) {
                    return Ok(StmtResult::Stmt(e));
                }
                if self.kind() == T::RBrace || (top && self.kind() == T::Eof) {
                    return Ok(StmtResult::Tail(e));
                }
                if self.prev_is_rbrace() {
                    return Ok(StmtResult::Stmt(e));
                }
                let found = self.describe_current();
                self.err(format!("expected `;` after expression, found {}", found))
            }
        }
    }

    /// A statement ending in `}` (block, `loop`, table literal, lambda body …)
    /// needs no `;`: nothing after it could continue the expression anyway.
    fn prev_is_rbrace(&self) -> bool {
        self.pos > 0 && self.toks[self.pos - 1].kind() == T::RBrace
    }

    fn expect_semi(&mut self, what: &str) -> PResult<()> {
        if self.eat(T::Semi) {
            return Ok(());
        }
        if self.prev_is_rbrace() {
            return Ok(());
        }
        // be lenient before a closing brace / EOF
        if matches!(self.kind(), T::RBrace | T::Eof) {
            return Ok(());
        }
        let found = self.describe_current();
        self.err(format!("expected `;` {}, found {}", what, found))
    }

    /// `const f = |..| ...` gives the lambda the name `f` (for stack traces).
    fn name_function(&mut self, init: NodeId, name: Sym) {
        if let Node::Func(f) = self.ast.node(init) {
            let def = &mut self.ast.funcs[f.0 as usize];
            if def.name.is_none() {
                def.name = name;
            }
        }
    }

    // ---- functions -----------------------------------------------------------

    /// Parses `(params) { body }` after the name has been consumed.
    fn parse_fn_rest(&mut self, name: Sym, pos: u32) -> PResult<NodeId> {
        self.expect(T::LParen, "to start the parameter list")?;
        let mark = self.scratch.len();
        while self.kind() != T::RParen {
            let p = match self.expect_ident("as parameter name") {
                Ok(p) => p,
                Err(()) => {
                    self.scratch.truncate(mark);
                    return Err(());
                }
            };
            self.scratch.push(p.0);
            if !self.eat(T::Comma) {
                break;
            }
        }
        let params = self.finish_list(mark);
        self.expect(T::RParen, "to close the parameter list")?;
        if self.kind() != T::LBrace {
            let found = self.describe_current();
            return self.err(format!("expected `{{` to start function body, found {}", found));
        }
        let body = self.parse_block()?;
        let id = self.ast.add_func(FuncDef { name, params, body });
        Ok(self.ast.push(Node::Func(id), pos))
    }

    fn parse_lambda(&mut self) -> PResult<NodeId> {
        let pos = self.pos_here();
        let mark = self.scratch.len();
        if self.eat(T::PipePipe) {
            // `||` : empty parameter list
        } else {
            self.expect(T::Pipe, "to start a lambda")?;
            while self.kind() != T::Pipe {
                let p = match self.expect_ident("as lambda parameter") {
                    Ok(p) => p,
                    Err(()) => {
                        self.scratch.truncate(mark);
                        return Err(());
                    }
                };
                self.scratch.push(p.0);
                if !self.eat(T::Comma) {
                    break;
                }
            }
            if let Err(()) = self.expect(T::Pipe, "to close the lambda parameters") {
                self.scratch.truncate(mark);
                return Err(());
            }
        }
        let params = self.finish_list(mark);
        let body = if self.kind() == T::LBrace && !self.looks_like_table() {
            self.parse_block()?
        } else if self.eat(T::Arrow) {
            if self.kind() == T::LBrace && !self.looks_like_table() {
                self.parse_block()?
            } else {
                self.parse_expr()?
            }
        } else if self.kind() == T::LBrace {
            self.parse_expr()?
        } else {
            let found = self.describe_current();
            return self.err(format!("expected `=>` or `{{` after lambda parameters, found {}", found));
        };
        let id = self.ast.add_func(FuncDef { name: Sym::NONE, params, body });
        Ok(self.ast.push(Node::Func(id), pos))
    }

    // ---- expressions ---------------------------------------------------------

    /// At a `{` in a position where both a block and a table literal are
    /// possible (lambda bodies): `{}`, `{ key: …`, `{ "k": …`, `{ [e]: …` and
    /// `{ name, …` are tables, everything else is a block.
    fn looks_like_table(&self) -> bool {
        match self.kind_at(1) {
            T::RBrace => true,
            T::Ident => matches!(self.kind_at(2), T::Colon | T::Comma),
            T::String | T::Number => self.kind_at(2) == T::Colon,
            T::LBracket => {
                // find the matching `]` and look for `:` after it
                let mut depth = 0;
                let mut i = 1;
                loop {
                    match self.kind_at(i) {
                        T::LBracket => depth += 1,
                        T::RBracket => {
                            depth -= 1;
                            if depth == 0 {
                                return self.kind_at(i + 1) == T::Colon;
                            }
                        }
                        T::Eof => return false,
                        _ => {}
                    }
                    i += 1;
                }
            }
            _ => false,
        }
    }

    pub fn parse_expr(&mut self) -> PResult<NodeId> {
        self.parse_bp(0)
    }

    fn parse_bp(&mut self, min_bp: u8) -> PResult<NodeId> {
        self.enter()?;
        let mut lhs = self.parse_prefix()?;
        loop {
            let tok = self.tok();
            let k = tok.kind();
            let Some((lbp, rbp)) = infix_bp(k) else { break };
            if lbp < min_bp {
                break;
            }
            // nodes are positioned at their operator token so that runtime
            // errors point at the right line even in multi-line chains
            let pos = tok.start;
            match k {
                T::LParen => {
                    self.pos += 1;
                    let args = self.parse_args(T::RParen)?;
                    lhs = self.ast.push(Node::Call { callee: lhs, args }, pos);
                }
                T::LBracket => {
                    self.pos += 1;
                    let index = self.parse_expr()?;
                    self.expect(T::RBracket, "to close the index")?;
                    lhs = self.ast.push(Node::Index { obj: lhs, index }, pos);
                }
                T::Dot => {
                    self.pos += 1;
                    let name = self.expect_ident("after `.`")?;
                    lhs = self.ast.push(Node::Field { obj: lhs, name }, pos);
                }
                T::Colon => {
                    // method call sugar `recv:name(args)`
                    if self.kind_at(1) != T::Ident || self.kind_at(2) != T::LParen {
                        break;
                    }
                    self.pos += 1;
                    let name = self.expect_ident("after `:`")?;
                    self.pos += 1; // `(`
                    let mark = self.scratch.len();
                    self.scratch.push(lhs.0);
                    let args = self.parse_list_into(mark, T::RParen)?;
                    lhs = self.ast.push(Node::MethodCall { name, args }, pos);
                }
                _ if is_assign(k) => {
                    if !matches!(self.ast.node(lhs), Node::Ident(_) | Node::Field { .. } | Node::Index { .. }) {
                        self.error_at(tok, "invalid assignment target".to_string());
                        return Err(());
                    }
                    self.pos += 1;
                    let value = self.parse_bp(rbp)?;
                    let op = if k == T::Eq { None } else { binop(k) };
                    if let Node::Ident(s) = self.ast.node(lhs) {
                        self.name_function(value, s);
                    }
                    lhs = self.ast.push(Node::Assign { op, target: lhs, value }, pos);
                }
                _ => {
                    self.pos += 1;
                    let rhs = self.parse_bp(rbp)?;
                    let op = binop(k).expect("binary operator");
                    lhs = self.ast.push(Node::Binary { op, lhs, rhs }, pos);
                }
            }
        }
        self.leave();
        Ok(lhs)
    }

    /// Comma separated expressions up to `close` (consumed).  Result is a list ref.
    fn parse_args(&mut self, close: TokenKind) -> PResult<ListRef> {
        let mark = self.scratch.len();
        self.parse_list_into(mark, close)
    }

    /// Like `parse_args`, but the list starts at scratch position `mark`, so the
    /// caller may have pushed leading items already.
    fn parse_list_into(&mut self, mark: usize, close: TokenKind) -> PResult<ListRef> {
        while self.kind() != close {
            match self.parse_expr() {
                Ok(e) => self.scratch.push(e.0),
                Err(()) => {
                    self.scratch.truncate(mark);
                    return Err(());
                }
            }
            if !self.eat(T::Comma) {
                break;
            }
        }
        if let Err(()) = self.expect(close, "to close the list") {
            self.scratch.truncate(mark);
            return Err(());
        }
        Ok(self.finish_list(mark))
    }

    fn parse_prefix(&mut self) -> PResult<NodeId> {
        let t = self.tok();
        let pos = t.start;
        match t.kind() {
            T::Number => {
                self.pos += 1;
                match parse_number(self.text(t)) {
                    Some(v) => Ok(self.ast.push(Node::Num(v), pos)),
                    None => {
                        self.error_at(t, format!("invalid number literal `{}`", self.text(t)));
                        Err(())
                    }
                }
            }
            T::String => {
                self.pos += 1;
                let raw = &self.src[t.start as usize + 1..t.end() - 1];
                if raw.contains('\\') {
                    if let Err(m) = unescape(raw) {
                        self.error_at(t, m);
                        return Err(());
                    }
                }
                Ok(self.ast.push(Node::Str { start: t.start + 1, len: t.len() as u32 - 2 }, pos))
            }
            T::Nil => {
                self.pos += 1;
                Ok(self.ast.push(Node::Nil, pos))
            }
            T::True => {
                self.pos += 1;
                Ok(self.ast.push(Node::True, pos))
            }
            T::False => {
                self.pos += 1;
                Ok(self.ast.push(Node::False, pos))
            }
            T::Ident => {
                self.pos += 1;
                let s = self.interner.intern(self.text(t));
                Ok(self.ast.push(Node::Ident(s), pos))
            }
            T::LParen => {
                self.pos += 1;
                let e = self.parse_expr()?;
                self.expect(T::RParen, "to close the group")?;
                Ok(e)
            }
            T::LBracket => {
                self.pos += 1;
                let items = self.parse_args(T::RBracket)?;
                Ok(self.ast.push(Node::Array { items }, pos))
            }
            T::LBrace => self.parse_table_literal(),
            T::Pipe | T::PipePipe => self.parse_lambda(),
            T::Fn => {
                self.pos += 1;
                if self.kind() == T::Ident {
                    return self.err("named function declarations are statements; use `fn (..) {..}` for a function expression");
                }
                self.parse_fn_rest(Sym::NONE, pos)
            }
            T::Minus | T::Bang | T::Tilde => {
                self.pos += 1;
                let op = match t.kind() {
                    T::Minus => UnOp::Neg,
                    T::Bang => UnOp::Not,
                    _ => UnOp::BitNot,
                };
                let expr = self.parse_bp(BP_UNARY)?;
                Ok(self.ast.push(Node::Unary { op, expr }, pos))
            }
            T::If => self.parse_if(),
            T::While => {
                self.pos += 1;
                let cond = self.parse_expr()?;
                let body = self.parse_block()?;
                Ok(self.ast.push(Node::While { cond, body }, pos))
            }
            T::Loop => {
                self.pos += 1;
                let body = self.parse_block()?;
                Ok(self.ast.push(Node::Loop { body }, pos))
            }
            T::Do => {
                self.pos += 1;
                self.parse_block()
            }
            T::For => self.parse_for(),
            T::Return => {
                self.pos += 1;
                let v = if starts_expr(self.kind()) { self.parse_expr()? } else { NodeId::NONE };
                Ok(self.ast.push(Node::Return(v), pos))
            }
            T::Break => {
                self.pos += 1;
                let v = if starts_expr(self.kind()) { self.parse_expr()? } else { NodeId::NONE };
                Ok(self.ast.push(Node::Break(v), pos))
            }
            T::Continue => {
                self.pos += 1;
                Ok(self.ast.push(Node::Continue, pos))
            }
            T::Var | T::Const | T::Struct | T::Impl => {
                let found = self.describe_current();
                self.err(format!("{} is a declaration and cannot be used as an expression", found))
            }
            _ => {
                let found = self.describe_current();
                self.err(format!("expected expression, found {}", found))
            }
        }
    }

    fn parse_if(&mut self) -> PResult<NodeId> {
        let pos = self.pos_here();
        self.pos += 1; // `if`
        let cond = self.parse_expr()?;
        let then = self.parse_block()?;
        let els = if self.eat(T::Else) {
            if self.kind() == T::If {
                self.parse_if()?
            } else {
                self.parse_block()?
            }
        } else {
            NodeId::NONE
        };
        Ok(self.ast.push(Node::If { cond, then, els }, pos))
    }

    fn parse_for(&mut self) -> PResult<NodeId> {
        let pos = self.pos_here();
        self.pos += 1; // `for`
        let v1 = self.expect_ident("as loop variable")?;
        let (vars, nvars) = if self.eat(T::Comma) {
            let v2 = self.expect_ident("as second loop variable")?;
            ([v1, v2], 2u8)
        } else {
            ([v1, Sym::NONE], 1u8)
        };
        self.expect(T::In, "in `for` loop")?;
        // iterable / range start: parse above the range operator level
        let iter = self.parse_bp(20)?;
        let end = if self.eat(T::DotDot) {
            if nvars != 1 {
                return self.err("range loops take exactly one variable");
            }
            self.parse_bp(20)?
        } else {
            NodeId::NONE
        };
        let body = self.parse_block()?;
        let id = self.ast.add_for(ForDef { vars, nvars, iter, end, body });
        Ok(self.ast.push(Node::For(id), pos))
    }

    fn parse_table_literal(&mut self) -> PResult<NodeId> {
        let pos = self.pos_here();
        self.pos += 1; // `{`
        let mark = self.scratch.len();
        let r = self.parse_table_entries(mark);
        if r.is_err() {
            self.scratch.truncate(mark);
            return Err(());
        }
        let entries = self.finish_list(mark);
        Ok(self.ast.push(Node::Table { entries }, pos))
    }

    fn parse_table_entries(&mut self, _mark: usize) -> PResult<()> {
        while self.kind() != T::RBrace {
            let kt = self.tok();
            let key = match kt.kind() {
                T::Ident => {
                    let next = self.kind_at(1);
                    self.pos += 1;
                    let s = self.interner.intern(self.text(kt));
                    let kn = self.ast.push(Node::Key(s), kt.start);
                    if matches!(next, T::Comma | T::RBrace) {
                        // shorthand `{ x }` == `{ x: x }`
                        let v = self.ast.push(Node::Ident(s), kt.start);
                        self.scratch.push(kn.0);
                        self.scratch.push(v.0);
                        if !self.eat(T::Comma) {
                            break;
                        }
                        continue;
                    }
                    kn
                }
                T::String => self.parse_prefix()?,
                T::Number => self.parse_prefix()?,
                T::LBracket => {
                    self.pos += 1;
                    let e = self.parse_expr()?;
                    self.expect(T::RBracket, "after computed key")?;
                    e
                }
                _ => {
                    let found = self.describe_current();
                    return self.err(format!("expected table key, found {}", found));
                }
            };
            self.expect(T::Colon, "after table key")?;
            let value = self.parse_expr()?;
            self.scratch.push(key.0);
            self.scratch.push(value.0);
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::RBrace, "to close the table literal")?;
        Ok(())
    }
}

enum StmtResult {
    Stmt(NodeId),
    Tail(NodeId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn dump(src: &str) -> String {
        let toks = lex(src);
        let mut i = Interner::new();
        match parse(src, &toks, &mut i) {
            Ok(ast) => ast.dump(&i, src),
            Err(e) => format!("ERR: {}", e[0].msg),
        }
    }

    #[test]
    fn precedence() {
        let d = dump("1 + 2 * 3;");
        let add = d.find("Add").unwrap();
        let mul = d.find("Mul").unwrap();
        assert!(add < mul);
        let d = dump("2 ** 3 ** 2;");
        // right associative: (2 ** (3 ** 2))
        let first = d.find("Pow").unwrap();
        let second = d[first + 1..].find("Pow").unwrap() + first + 1;
        assert!(d[second..].contains("3") && !d[first..second].contains("2\n  2"));
    }

    #[test]
    fn errors() {
        assert!(dump("1 +;").starts_with("ERR"));
        assert!(dump("var = 1;").starts_with("ERR"));
        assert!(dump("1 = 2;").starts_with("ERR: invalid assignment"));
    }

    #[test]
    fn statements() {
        assert!(!dump("fn f(a, b) { return a + b; } const g = |x| => x; const l = loop { break 1; };").starts_with("ERR"));
        assert!(!dump("for i in 0..10 { } for k, v in t { } while a < 2 { continue; }").starts_with("ERR"));
        assert!(!dump("struct P { x, y } impl P { fn new(self, a) { } }").starts_with("ERR"));
    }
}
