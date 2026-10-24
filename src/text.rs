//! Lexeme decoding helpers (kept out of the lexer on purpose) and number
//! formatting.

/// Parse the text of a `Number` token.
pub fn parse_number(text: &str) -> Option<f64> {
    let t: String;
    let s = if text.contains('_') {
        t = text.replace('_', "");
        t.as_str()
    } else {
        text
    };
    let bytes = s.as_bytes();
    if bytes.len() > 2 && bytes[0] == b'0' {
        let radix = match bytes[1] {
            b'x' | b'X' => 16,
            b'b' | b'B' => 2,
            b'o' | b'O' => 8,
            _ => 0,
        };
        if radix != 0 {
            return u64::from_str_radix(&s[2..], radix).ok().map(|v| v as f64);
        }
    }
    if !bytes.first().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    s.parse::<f64>().ok()
}

/// Process escape sequences of a string literal body.
pub fn unescape(raw: &str) -> Result<std::borrow::Cow<'_, str>, String> {
    if !raw.contains('\\') {
        return Ok(std::borrow::Cow::Borrowed(raw));
    }
    let mut out = String::with_capacity(raw.len());
    let mut it = raw.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('x') => {
                let h: String = it.by_ref().take(2).collect();
                let v = u8::from_str_radix(&h, 16).map_err(|_| "invalid \\x escape".to_string())?;
                if v > 0x7f {
                    return Err("\\x escape must be <= 0x7f".to_string());
                }
                out.push(v as char);
            }
            Some('u') => {
                if it.next() != Some('{') {
                    return Err("expected `{` after \\u".to_string());
                }
                let mut h = String::new();
                loop {
                    match it.next() {
                        Some('}') => break,
                        Some(c) => h.push(c),
                        None => return Err("unterminated \\u{...} escape".to_string()),
                    }
                }
                let v = u32::from_str_radix(&h, 16).map_err(|_| "invalid \\u escape".to_string())?;
                out.push(char::from_u32(v).ok_or_else(|| "invalid unicode scalar".to_string())?);
            }
            Some(c) => return Err(format!("unknown escape `\\{}`", c)),
            None => return Err("dangling backslash".to_string()),
        }
    }
    Ok(out.into())
}

/// Format a number the way `print` shows it: integral values without `.0`.
pub fn fmt_num(n: f64, out: &mut String) {
    use std::fmt::Write;
    if n.is_nan() {
        out.push_str("nan");
    } else if n.is_infinite() {
        out.push_str(if n > 0.0 { "inf" } else { "-inf" });
    } else if n == n.trunc() && n.abs() < 1e15 {
        // negative zero prints as plain `0` (`-4 % 2` would otherwise show `-0`)
        let _ = write!(out, "{}", n as i64);
    } else if n.abs() >= 1e16 || (n != 0.0 && n.abs() < 1e-5) {
        let _ = write!(out, "{:e}", n);
    } else {
        let _ = write!(out, "{}", n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(parse_number("12"), Some(12.0));
        assert_eq!(parse_number("1_000"), Some(1000.0));
        assert_eq!(parse_number("0xff"), Some(255.0));
        assert_eq!(parse_number("0b101"), Some(5.0));
        assert_eq!(parse_number("1.5e3"), Some(1500.0));
        assert_eq!(parse_number("0x"), None);
    }

    #[test]
    fn escapes() {
        assert_eq!(unescape("a\\nb").unwrap(), "a\nb");
        assert_eq!(unescape("\\u{1F600}").unwrap(), "😀");
        assert!(unescape("\\q").is_err());
    }

    #[test]
    fn formatting() {
        let f = |n| {
            let mut s = String::new();
            fmt_num(n, &mut s);
            s
        };
        assert_eq!(f(3.0), "3");
        assert_eq!(f(2.75), "2.75");
        assert_eq!(f(-0.5), "-0.5");
        assert_eq!(f(1e21), "1e21");
        assert_eq!(f(f64::NAN), "nan");
    }
}
