//! Interactive read-eval-print loop.

use crate::lexer;
use crate::vm::Vm;
use std::io::{self, BufRead, Write};

const HELP: &str = "\
commands:
  :help            show this help
  :quit, :q        leave the REPL (Ctrl-D works too)
  :bytecode        toggle bytecode dumps
  :ast             toggle AST dumps
  :opt             toggle the peephole optimiser
  :gc              run the garbage collector
  :mem             show heap statistics
  :globals         list user-defined globals
Incomplete input (open brackets or strings) continues on the next line.
The value of the last expression is printed.";

pub fn run(vm: &mut Vm) {
    vm.opts.repl = true;
    println!("Kite {} — :help for commands, Ctrl-D to exit", env!("CARGO_PKG_VERSION"));
    let stdin = io::stdin();
    let mut buf = String::new();
    loop {
        let prompt = if buf.is_empty() { ">>> " } else { "... " };
        print!("{}", prompt);
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                println!();
                break;
            }
            Ok(_) => {}
        }
        if buf.is_empty() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Some(cmd) = t.strip_prefix(':') {
                if !command(vm, cmd.trim()) {
                    break;
                }
                continue;
            }
        }
        let blank = line.trim().is_empty();
        buf.push_str(&line);
        // a blank line abandons a continuation and shows the error
        if !blank && (lexer::is_incomplete(&buf) || ends_unexpectedly(vm, &buf)) {
            continue;
        }
        let src = std::mem::take(&mut buf);
        match vm.eval("<repl>", &src) {
            Ok(v) => {
                vm.flush();
                if !v.is_nil() {
                    match vm.repr(v) {
                        Ok(s) => println!("=> {}", s),
                        Err(e) => eprintln!("(error while printing value: {})", e.msg),
                    }
                }
            }
            Err(e) => {
                vm.flush();
                eprintln!("{}", e);
            }
        }
    }
    vm.shutdown();
}

/// `x = 1 +` : the parser ran into the end of input, so more is expected.
fn ends_unexpectedly(vm: &mut Vm, src: &str) -> bool {
    let toks = lexer::lex(src);
    match crate::parser::parse(src, &toks, &mut vm.interner) {
        Ok(_) => false,
        Err(diags) => diags.iter().any(|d| d.msg.contains("end of input") || d.msg.contains("unclosed")),
    }
}

fn command(vm: &mut Vm, cmd: &str) -> bool {
    match cmd {
        "q" | "quit" | "exit" => return false,
        "help" | "h" | "?" => println!("{}", HELP),
        "bytecode" | "bc" => {
            vm.opts.dump_bytecode = !vm.opts.dump_bytecode;
            println!("bytecode dumps {}", if vm.opts.dump_bytecode { "on" } else { "off" });
        }
        "ast" => {
            vm.opts.dump_ast = !vm.opts.dump_ast;
            println!("AST dumps {}", if vm.opts.dump_ast { "on" } else { "off" });
        }
        "opt" => {
            vm.opts.no_optimize = !vm.opts.no_optimize;
            println!("optimiser {}", if vm.opts.no_optimize { "off" } else { "on" });
        }
        "gc" => {
            let before = vm.heap.bytes;
            vm.collect_garbage();
            println!("collected {} bytes, {} live", before.saturating_sub(vm.heap.bytes), vm.heap.bytes);
        }
        "mem" => {
            println!(
                "heap: {} bytes in {} objects, next collection at {} bytes, {} collections so far",
                vm.heap.bytes,
                vm.heap.object_count(),
                vm.heap.next_gc,
                vm.heap.collections
            );
        }
        "globals" => {
            let mut names: Vec<String> = Vec::new();
            for i in 0..vm.interner.len() {
                let name = vm.interner.get(crate::intern::Sym(i as u32)).to_string();
                if let Some(v) = vm.get_global(&name) {
                    if v.is_kind(crate::object::Kind::Native) || name == "math" || name == "string" {
                        continue;
                    }
                    let shown = vm.repr(v).unwrap_or_else(|_| "?".into());
                    names.push(format!("{} = {}", name, shown));
                }
            }
            if names.is_empty() {
                println!("(no user globals)");
            }
            for n in names {
                println!("{}", n);
            }
        }
        other => println!("unknown command `:{}` — try :help", other),
    }
    true
}
