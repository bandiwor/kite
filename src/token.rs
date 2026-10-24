//! 8-byte tokens.  The lexer never interprets anything: a token is just
//! `(kind, start, len)`; numbers, escapes and identifiers are decoded lazily
//! by whoever needs them (the parser).

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TokenKind {
    Eof = 0,
    /// Anything the lexer could not classify (bad char, unterminated string/comment).
    Error,
    Ident,
    Number,
    String,
    // keywords
    Var,
    Const,
    Fn,
    Return,
    If,
    Else,
    While,
    Loop,
    For,
    In,
    Break,
    Continue,
    Struct,
    Impl,
    Do,
    Nil,
    True,
    False,
    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Dot,
    DotDot,
    Semi,
    Colon,
    Arrow,
    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Bang,
    Shl,
    Shr,
    AmpAmp,
    PipePipe,
    Eq,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    StarStarEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
}

impl TokenKind {
    /// Human readable spelling used in diagnostics.
    pub fn describe(self) -> &'static str {
        use TokenKind::*;
        match self {
            Eof => "end of input",
            Error => "invalid token",
            Ident => "identifier",
            Number => "number",
            String => "string",
            Var => "`var`",
            Const => "`const`",
            Fn => "`fn`",
            Return => "`return`",
            If => "`if`",
            Else => "`else`",
            While => "`while`",
            Loop => "`loop`",
            For => "`for`",
            In => "`in`",
            Break => "`break`",
            Continue => "`continue`",
            Struct => "`struct`",
            Impl => "`impl`",
            Do => "`do`",
            Nil => "`nil`",
            True => "`true`",
            False => "`false`",
            LParen => "`(`",
            RParen => "`)`",
            LBrace => "`{`",
            RBrace => "`}`",
            LBracket => "`[`",
            RBracket => "`]`",
            Comma => "`,`",
            Dot => "`.`",
            DotDot => "`..`",
            Semi => "`;`",
            Colon => "`:`",
            Arrow => "`=>`",
            Plus => "`+`",
            Minus => "`-`",
            Star => "`*`",
            Slash => "`/`",
            Percent => "`%`",
            StarStar => "`**`",
            Amp => "`&`",
            Pipe => "`|`",
            Caret => "`^`",
            Tilde => "`~`",
            Bang => "`!`",
            Shl => "`<<`",
            Shr => "`>>`",
            AmpAmp => "`&&`",
            PipePipe => "`||`",
            Eq => "`=`",
            EqEq => "`==`",
            BangEq => "`!=`",
            Lt => "`<`",
            Le => "`<=`",
            Gt => "`>`",
            Ge => "`>=`",
            PlusEq => "`+=`",
            MinusEq => "`-=`",
            StarEq => "`*=`",
            SlashEq => "`/=`",
            PercentEq => "`%=`",
            StarStarEq => "`**=`",
            AmpEq => "`&=`",
            PipeEq => "`|=`",
            CaretEq => "`^=`",
            ShlEq => "`<<=`",
            ShrEq => "`>>=`",
        }
    }
}

/// `start:u32 | kind:u8 | len:24 bits` packed into exactly 8 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Token {
    pub start: u32,
    kind: TokenKind,
    len_lo: u8,
    len_hi: u16,
}

pub const MAX_TOKEN_LEN: usize = 0xFF_FFFF;

impl Token {
    #[inline(always)]
    pub fn new(kind: TokenKind, start: usize, len: usize) -> Token {
        debug_assert!(len <= MAX_TOKEN_LEN);
        Token { start: start as u32, kind, len_lo: len as u8, len_hi: (len >> 8) as u16 }
    }
    #[inline(always)]
    pub fn kind(self) -> TokenKind {
        self.kind
    }
    #[inline(always)]
    pub fn len(self) -> usize {
        self.len_lo as usize | (self.len_hi as usize) << 8
    }
    #[inline(always)]
    pub fn end(self) -> usize {
        self.start as usize + self.len()
    }
}

const _: () = assert!(std::mem::size_of::<Token>() == 8);
