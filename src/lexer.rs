//! Single-pass byte lexer.  It produces a `Vec<Token>` terminated by `Eof`
//! and does no interpretation of the lexemes (no number parsing, no escape
//! processing, no interning).

use crate::token::{Token, TokenKind, MAX_TOKEN_LEN};
use TokenKind::*;

#[inline(always)]
fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c >= 0x80
}

#[inline(always)]
fn is_ident_cont(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

fn keyword(s: &[u8]) -> TokenKind {
    match s {
        b"var" => Var,
        b"const" => Const,
        b"fn" => Fn,
        b"return" => Return,
        b"if" => If,
        b"else" => Else,
        b"while" => While,
        b"loop" => Loop,
        b"for" => For,
        b"in" => In,
        b"break" => Break,
        b"continue" => Continue,
        b"struct" => Struct,
        b"impl" => Impl,
        b"do" => Do,
        b"nil" => Nil,
        b"true" => True,
        b"false" => False,
        _ => Ident,
    }
}

/// Tokenise `src`.  Sources larger than 4 GiB are not supported (offsets are u32).
pub fn lex(src: &str) -> Vec<Token> {
    let b = src.as_bytes();
    let n = b.len();
    assert!(n < u32::MAX as usize, "source too large");
    let mut toks: Vec<Token> = Vec::with_capacity(n / 3 + 4);
    let mut i = 0usize;

    macro_rules! at {
        ($k:expr) => {
            if $k < n {
                b[$k]
            } else {
                0
            }
        };
    }

    loop {
        // ---- skip whitespace and comments
        loop {
            match at!(i) {
                b' ' | b'\t' | b'\n' | b'\r' if i < n => i += 1,
                b'/' if at!(i + 1) == b'/' => {
                    i += 2;
                    while i < n && b[i] != b'\n' {
                        i += 1;
                    }
                }
                b'/' if at!(i + 1) == b'*' => {
                    let start = i;
                    let mut depth = 1;
                    i += 2;
                    while i < n && depth > 0 {
                        if b[i] == b'/' && at!(i + 1) == b'*' {
                            depth += 1;
                            i += 2;
                        } else if b[i] == b'*' && at!(i + 1) == b'/' {
                            depth -= 1;
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    if depth > 0 {
                        toks.push(Token::new(Error, start, (i - start).min(MAX_TOKEN_LEN)));
                    }
                }
                _ => break,
            }
        }
        if i >= n {
            toks.push(Token::new(Eof, n, 0));
            return toks;
        }

        let start = i;
        let c = b[i];
        i += 1;
        let kind = match c {
            b'(' => LParen,
            b')' => RParen,
            b'{' => LBrace,
            b'}' => RBrace,
            b'[' => LBracket,
            b']' => RBracket,
            b',' => Comma,
            b';' => Semi,
            b':' => Colon,
            b'~' => Tilde,
            b'.' => {
                if at!(i) == b'.' {
                    i += 1;
                    DotDot
                } else {
                    Dot
                }
            }
            b'+' => {
                if at!(i) == b'=' {
                    i += 1;
                    PlusEq
                } else {
                    Plus
                }
            }
            b'-' => {
                if at!(i) == b'=' {
                    i += 1;
                    MinusEq
                } else {
                    Minus
                }
            }
            b'*' => {
                if at!(i) == b'*' {
                    i += 1;
                    if at!(i) == b'=' {
                        i += 1;
                        StarStarEq
                    } else {
                        StarStar
                    }
                } else if at!(i) == b'=' {
                    i += 1;
                    StarEq
                } else {
                    Star
                }
            }
            b'/' => {
                if at!(i) == b'=' {
                    i += 1;
                    SlashEq
                } else {
                    Slash
                }
            }
            b'%' => {
                if at!(i) == b'=' {
                    i += 1;
                    PercentEq
                } else {
                    Percent
                }
            }
            b'^' => {
                if at!(i) == b'=' {
                    i += 1;
                    CaretEq
                } else {
                    Caret
                }
            }
            b'&' => match at!(i) {
                b'&' => {
                    i += 1;
                    AmpAmp
                }
                b'=' => {
                    i += 1;
                    AmpEq
                }
                _ => Amp,
            },
            b'|' => match at!(i) {
                b'|' => {
                    i += 1;
                    PipePipe
                }
                b'=' => {
                    i += 1;
                    PipeEq
                }
                _ => Pipe,
            },
            b'!' => {
                if at!(i) == b'=' {
                    i += 1;
                    BangEq
                } else {
                    Bang
                }
            }
            b'=' => match at!(i) {
                b'=' => {
                    i += 1;
                    EqEq
                }
                b'>' => {
                    i += 1;
                    Arrow
                }
                _ => Eq,
            },
            b'<' => match at!(i) {
                b'=' => {
                    i += 1;
                    Le
                }
                b'<' => {
                    i += 1;
                    if at!(i) == b'=' {
                        i += 1;
                        ShlEq
                    } else {
                        Shl
                    }
                }
                _ => Lt,
            },
            b'>' => match at!(i) {
                b'=' => {
                    i += 1;
                    Ge
                }
                b'>' => {
                    i += 1;
                    if at!(i) == b'=' {
                        i += 1;
                        ShrEq
                    } else {
                        Shr
                    }
                }
                _ => Gt,
            },
            b'"' | b'\'' => {
                let quote = c;
                let mut closed = false;
                while i < n {
                    let ch = b[i];
                    i += 1;
                    if ch == b'\\' {
                        i += 1; // skip escaped byte
                    } else if ch == quote {
                        closed = true;
                        break;
                    }
                }
                i = i.min(n);
                if closed {
                    String
                } else {
                    Error
                }
            }
            b'0'..=b'9' => {
                let radix_prefix = c == b'0' && matches!(at!(i), b'x' | b'X' | b'b' | b'B' | b'o' | b'O');
                if radix_prefix {
                    i += 1;
                    while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                        i += 1;
                    }
                } else {
                    while i < n && (b[i].is_ascii_digit() || b[i] == b'_') {
                        i += 1;
                    }
                    // fraction: only if a digit follows the dot (so `1..5` stays a range)
                    if at!(i) == b'.' && at!(i + 1).is_ascii_digit() {
                        i += 1;
                        while i < n && (b[i].is_ascii_digit() || b[i] == b'_') {
                            i += 1;
                        }
                    }
                    // exponent
                    if matches!(at!(i), b'e' | b'E') {
                        let mut j = i + 1;
                        if matches!(at!(j), b'+' | b'-') {
                            j += 1;
                        }
                        if at!(j).is_ascii_digit() {
                            i = j;
                            while i < n && b[i].is_ascii_digit() {
                                i += 1;
                            }
                        }
                    }
                }
                Number
            }
            _ if is_ident_start(c) => {
                while i < n && is_ident_cont(b[i]) {
                    i += 1;
                }
                keyword(&b[start..i])
            }
            _ => Error,
        };
        let len = i - start;
        if len > MAX_TOKEN_LEN {
            toks.push(Token::new(Error, start, MAX_TOKEN_LEN));
        } else {
            toks.push(Token::new(kind, start, len));
        }
    }
}

/// True when the source ends inside an open bracket / string / comment, which
/// is what the REPL uses to decide whether to ask for another line.
pub fn is_incomplete(src: &str) -> bool {
    let toks = lex(src);
    let mut depth: i32 = 0;
    for t in &toks {
        match t.kind() {
            LParen | LBrace | LBracket => depth += 1,
            RParen | RBrace | RBracket => depth -= 1,
            Error => {
                let text = &src[t.start as usize..t.end()];
                if text.starts_with('"') || text.starts_with('\'') || text.starts_with("/*") {
                    return true;
                }
            }
            _ => {}
        }
    }
    depth > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(s: &str) -> Vec<TokenKind> {
        lex(s).iter().map(|t| t.kind()).collect()
    }

    #[test]
    fn basic() {
        assert_eq!(kinds("var x = 10;"), vec![Var, Ident, Eq, Number, Semi, Eof]);
        assert_eq!(kinds("a += b ** 2"), vec![Ident, PlusEq, Ident, StarStar, Number, Eof]);
        assert_eq!(kinds("|a| => a"), vec![Pipe, Ident, Pipe, Arrow, Ident, Eof]);
        assert_eq!(kinds("1..5"), vec![Number, DotDot, Number, Eof]);
        assert_eq!(kinds("1.5e-3 0xFF 1_000"), vec![Number, Number, Number, Eof]);
    }

    #[test]
    fn comments_and_strings() {
        assert_eq!(kinds("a // c\n b /* x /* y */ z */ c"), vec![Ident, Ident, Ident, Eof]);
        assert_eq!(kinds(r#""a\"b" 'c'"#), vec![String, String, Eof]);
        assert_eq!(kinds("\"abc"), vec![Error, Eof]);
    }

    #[test]
    fn incomplete() {
        assert!(is_incomplete("fn f() {"));
        assert!(is_incomplete("\"abc"));
        assert!(!is_incomplete("fn f() {}"));
    }
}
