//! Source files, diagnostics and the user-facing error type.

use std::cell::OnceCell;
use std::fmt;
use std::rc::Rc;

pub struct SourceFile {
    pub name: String,
    pub text: String,
    line_starts: OnceCell<Vec<u32>>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Rc<SourceFile> {
        Rc::new(SourceFile { name: name.into(), text: text.into(), line_starts: OnceCell::new() })
    }

    fn starts(&self) -> &Vec<u32> {
        self.line_starts.get_or_init(|| {
            let mut v = vec![0u32];
            for (i, b) in self.text.bytes().enumerate() {
                if b == b'\n' {
                    v.push(i as u32 + 1);
                }
            }
            v
        })
    }

    /// 1-based (line, column) of a byte offset. Columns count characters.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let starts = self.starts();
        let line = match starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let ls = starts[line] as usize;
        let end = (offset as usize).min(self.text.len());
        let col = self.text.get(ls..end).map(|s| s.chars().count()).unwrap_or(end - ls);
        (line as u32 + 1, col as u32 + 1)
    }

    pub fn line(&self, offset: u32) -> u32 {
        self.line_col(offset).0
    }

    pub fn line_text(&self, line: u32) -> &str {
        let starts = self.starts();
        let i = (line as usize).saturating_sub(1).min(starts.len() - 1);
        let s = starts[i] as usize;
        let e = starts.get(i + 1).map(|&e| e as usize).unwrap_or(self.text.len());
        self.text[s..e].trim_end_matches(['\n', '\r'])
    }
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub msg: String,
    pub start: u32,
    pub len: u32,
}

impl Diag {
    pub fn new(msg: impl Into<String>, start: u32, len: u32) -> Diag {
        Diag { msg: msg.into(), start, len }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
    Compile,
    Runtime,
}

/// Error returned by the whole pipeline.  `Display` renders rustc-like
/// messages with a source snippet and (for runtime errors) a stack trace.
pub struct KiteError {
    pub kind: ErrorKind,
    pub diags: Vec<Diag>,
    pub source: Rc<SourceFile>,
    pub trace: Vec<String>,
}

impl KiteError {
    pub fn message(&self) -> String {
        self.diags.iter().map(|d| d.msg.clone()).collect::<Vec<_>>().join("\n")
    }
}

impl fmt::Debug for KiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for KiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self.kind {
            ErrorKind::Syntax => "syntax error",
            ErrorKind::Compile => "error",
            ErrorKind::Runtime => "runtime error",
        };
        for (i, d) in self.diags.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            let (line, col) = self.source.line_col(d.start);
            writeln!(f, "{}: {}", label, d.msg)?;
            writeln!(f, "  --> {}:{}:{}", self.source.name, line, col)?;
            let full = self.source.line_text(line);
            let gutter = line.to_string();
            let pad = " ".repeat(gutter.len());
            // very long lines: show a window around the error
            let chars: Vec<char> = full.chars().collect();
            let (from, to) = if chars.len() > 100 {
                let c = col as usize - 1;
                let from = c.saturating_sub(40);
                (from, (from + 100).min(chars.len()))
            } else {
                (0, chars.len())
            };
            let shown: String = chars[from..to].iter().collect();
            let lead = if from > 0 { "…" } else { "" };
            let trail = if to < chars.len() { "…" } else { "" };
            writeln!(f, " {} |", pad)?;
            writeln!(f, " {} | {}{}{}", gutter, lead, shown, trail)?;
            let prefix: String =
                chars[from..(col as usize - 1).max(from)].iter().map(|&c| if c == '\t' { '\t' } else { ' ' }).collect();
            let lead_pad = if from > 0 { " " } else { "" };
            let width = {
                let line_start = d.start as usize;
                let take = (d.len as usize).max(1);
                let slice_end = (line_start + take).min(self.source.text.len());
                let seg = self.source.text.get(line_start..slice_end).unwrap_or("");
                let seg = seg.split('\n').next().unwrap_or("");
                seg.chars().count().clamp(1, 60)
            };
            write!(f, " {} | {}{}{}", pad, lead_pad, prefix, "^".repeat(width))?;
        }
        for t in &self.trace {
            write!(f, "\n{}", t)?;
        }
        Ok(())
    }
}

impl std::error::Error for KiteError {}
