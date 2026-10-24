//! Kite: a small, fast, dynamically typed, expression-oriented language.
//!
//! Pipeline: `lexer` (8-byte tokens) → `parser` (Pratt, thin arena AST) →
//! `analyzer` (scopes, slots, captures) → `compiler` (stack bytecode) →
//! `peephole` → `vm` (NaN-boxed values, closures, mark & sweep GC).

pub mod analyzer;
pub mod ast;
pub mod builtins;
pub mod compiler;
pub mod disasm;
pub mod hash;
pub mod heap;
pub mod intern;
pub mod lexer;
pub mod object;
pub mod op;
pub mod parser;
pub mod peephole;
pub mod repl;
pub mod source;
pub mod table;
pub mod text;
pub mod token;
pub mod value;
pub mod vm;

pub use source::KiteError;
pub use value::Value;
pub use vm::{Options, Vm};
