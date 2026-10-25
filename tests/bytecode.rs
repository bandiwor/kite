//! Encoding tests: 4-byte instructions, double instructions, operand limits.

use kite::op::{self, Op};
use kite::value::Value;
use kite::vm::Vm;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

struct Sink(Rc<RefCell<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn run(src: &str, optimize: bool) -> (String, Result<Value, String>) {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut vm = Vm::new();
    vm.out = Box::new(Sink(buf.clone()));
    vm.opts.no_optimize = !optimize;
    let r = vm.eval("bytecode.kite", src).map_err(|e| e.to_string());
    let out = String::from_utf8(buf.borrow().clone()).unwrap();
    (out, r)
}

/// Run on a big stack: some generated programs are large.
fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(256 << 20).spawn(f).unwrap().join().unwrap()
}

/// All protos reachable from the main one.
fn all_protos(p: kite::object::Gc<kite::object::Proto>, out: &mut Vec<kite::object::Gc<kite::object::Proto>>) {
    out.push(p);
    for c in &p.consts {
        if let Some(q) = c.as_gc::<kite::object::Proto>() {
            all_protos(q, out);
        }
    }
}

#[test]
fn instructions_are_four_bytes() {
    assert_eq!(std::mem::size_of::<Op>(), 4);
}

#[test]
fn compiled_code_is_structurally_valid() {
    big_stack(|| {
        let src = r#"
            fn f(n) {
                var s = 0;
                for i in 0..n { if i % 3 == 0 && i != 9 { s += i; } else { s -= 1; } }
                var t = { a: 1 };
                for k, v in t { s += v; }
                while s > 100 { s = s - 50; if s == 77 { break; } }
                return s;
            }
            print(f(100));
            union U { A(x), B(x, y), C }
            fn m(v) {
                return match v { A(1 | 2) => 1, A(x) if x > 5 => x, B(_, 0) | C => 0, B(x, y) => x + y, 3 => 7, _ => -1 };
            }
            print(m(U.A(1)), m(U.C), m(3));
        "#;
        for optimize in [true, false] {
            let mut vm = Vm::new();
            vm.opts.no_optimize = !optimize;
            let (proto, _) = vm.compile("t.kite", src).unwrap();
            let mut protos = Vec::new();
            all_protos(proto, &mut protos);
            assert!(protos.len() >= 2);
            let mut wide = 0;
            for p in &protos {
                op::verify(&p.code).unwrap_or_else(|e| panic!("{}: {}", p.name, e));
                assert_eq!(p.code.len(), p.offsets.len(), "one source offset per word");
                wide += p.code.iter().filter(|o| o.is_wide()).count();
            }
            // loops always use double instructions
            assert!(wide >= 3, "expected double instructions, found {}", wide);
        }
    });
}

#[test]
fn fused_ops_use_double_instructions_only_when_optimizing() {
    let src = "fn f(a, b) { var n = 0; while n < 10 { n += 1; if a < b { n += 1; } } n } f(1, 2);";
    let count = |optimize: bool| {
        let mut vm = Vm::new();
        vm.opts.no_optimize = !optimize;
        let (proto, _) = vm.compile("t.kite", src).unwrap();
        let mut protos = Vec::new();
        all_protos(proto, &mut protos);
        protos
            .iter()
            .flat_map(|p| p.code.to_vec())
            .filter(|o| matches!(o, Op::JNotLtLL(..) | Op::JNotLtLK(..)))
            .count()
    };
    assert_eq!(count(false), 0);
    assert!(count(true) >= 2);
}

#[test]
fn jumps_longer_than_16_bits() {
    big_stack(|| {
        // a loop body of ~100k instructions: the back jump and the exit jump exceed i16
        let mut src = String::from("fn f() { var i = 0; var s = 0; while i < 3 { i += 1;\n");
        for _ in 0..30_000 {
            src.push_str("s = s + i * 2;\n");
        }
        src.push_str("} s }\nprint(f());\n");
        // each pass adds 30000 * 2 * i  => 60000 * (1 + 2 + 3)
        for optimize in [true, false] {
            let (out, r) = run(&src, optimize);
            r.unwrap();
            assert_eq!(out.trim(), "360000", "optimize={}", optimize);
        }
    });
}

#[test]
fn slots_above_255_and_constants_above_65535() {
    big_stack(|| {
        // 300 locals: fused forms need slots < 256, the rest must fall back to plain ops
        let mut src = String::from("fn f() {\n");
        for i in 0..300 {
            src.push_str(&format!("var v{} = {};\n", i, i));
        }
        src.push_str("var s = v299 + v298;\n s += v297;\n s = s + v1 * v2;\n");
        src.push_str("var i = 0; while v299 > i { i += 1; if v298 < v299 { s += 1; } }\n");
        src.push_str("for k in 0..3 { s += k; } for q in [5, 6] { s += q; }\n");
        src.push_str("s }\nprint(f());\n");
        // 299+298+297 + 1*2 + 299 (loop) + 3 + 11
        let expect = 299 + 298 + 297 + 2 + 299 + 3 + 11;
        for optimize in [true, false] {
            let (out, r) = run(&src, optimize);
            r.unwrap();
            assert_eq!(out.trim(), expect.to_string(), "optimize={}", optimize);
        }

        // 70000 distinct constants, then comparisons against late constants
        let mut src = String::from("fn g() { var s = 0;\n");
        for i in 0..70_000 {
            src.push_str(&format!("s += {}.5;\n", i));
        }
        src.push_str(
            "var x = 70000; if x < 70001.25 { s += 1; } while x < 70002.25 { x += 1; } s + x }\nprint(g());\n",
        );
        let expect: f64 = (0..70_000).map(|i| i as f64 + 0.5).sum::<f64>() + 1.0 + 70003.0;
        for optimize in [true, false] {
            let (out, r) = run(&src, optimize);
            r.unwrap();
            assert_eq!(out.trim(), format!("{}", expect), "optimize={}", optimize);
        }
    });
}
