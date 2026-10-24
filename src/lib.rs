//! Kite: a small, fast, dynamically typed, expression-oriented language.
//!
//! Pipeline: [`lexer`] (8-byte tokens) → [`parser`] (Pratt, thin arena AST) →
//! [`analyzer`] (scopes, slots, captures) → [`compiler`] (stack bytecode) →
//! [`peephole`] → [`vm`] (NaN-boxed values, closures, mark & sweep GC).
//!
//! # Embedding
//!
//! ```
//! use kite::{Value, Vm};
//!
//! let mut vm = Vm::new();
//! // expose a Rust function to scripts
//! vm.define_native("twice", 1, |_vm, args| Ok(Value::num(args[0].as_num() * 2.0)));
//! let v = vm.eval("example", "fn f(x) { twice(x) + 1 } f(20)").unwrap();
//! assert_eq!(v.as_num(), 41.0);
//!
//! // errors carry a rendered source snippet
//! let err = vm.eval("example", "1 +").err().unwrap();
//! assert!(err.to_string().contains("expected expression"));
//! ```
//!
//! Natives that call back into the VM (`Vm::call_value`) must keep freshly
//! allocated objects reachable with `Vm::root`; see `ARCHITECTURE.md`.

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
#[cfg(feature = "op-profile")]
pub mod profile;
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
