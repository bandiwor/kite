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
    for q in &p.protos {
        all_protos(*q, out);
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
            fn t(r) { const v = r?; for i in 0..v { if i == 3 { Err(i)?; } } return Ok(v); }
            print(t(Ok(2)), t(Ok(9)), t(Err(0)));
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

fn count_ops(src: &str, optimize: bool, pred: impl Fn(&Op) -> bool) -> usize {
    let mut vm = Vm::new();
    vm.opts.no_optimize = !optimize;
    let (proto, _) = vm.compile("t.kite", src).unwrap();
    let mut protos = Vec::new();
    all_protos(proto, &mut protos);
    protos.iter().flat_map(|p| p.code.to_vec()).filter(|o| pred(o)).count()
}

#[test]
fn comparisons_in_conditions_fuse_with_the_jump() {
    let src = "fn f(a, b) { var n = 0; while n < 10 { n += 1; if a < b { n += 1; } } n } f(1, 2);";
    for optimize in [true, false] {
        // `while n < 10` and `if a < b` never materialise a boolean
        assert_eq!(count_ops(src, optimize, |o| matches!(o, Op::Lt(..) | Op::Gt(..) | Op::Le(..) | Op::Ge(..))), 0);
        assert!(count_ops(src, optimize, |o| matches!(o, Op::JLtK(..))) >= 1);
        assert!(count_ops(src, optimize, |o| matches!(o, Op::JNotLt(..))) >= 1);
    }
}

#[test]
fn peephole_shrinks_code_without_changing_the_result() {
    let src = "fn f(n) { var s = 0; for i in 0..n { if i % 2 == 0 { continue; } s += i; } s } print(f(10));";
    let len = |optimize: bool| count_ops(src, optimize, |_| true);
    assert!(len(true) <= len(false));
    for optimize in [true, false] {
        let (out, r) = run(src, optimize);
        r.unwrap();
        assert_eq!(out.trim(), "25");
    }
}

#[test]
fn arithmetic_on_locals_is_one_instruction() {
    // `s = s + i * 2` needs no temporaries to shuttle values around: Mul into a register, Add into s
    let src = "fn f(i) { var s = 0; s = s + i * 2; s } f(3);";
    assert_eq!(count_ops(src, true, |o| matches!(o, Op::Move(..))), 0);
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
fn constants_and_names_above_the_short_operand_range() {
    big_stack(|| {
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

        // 600 field names, 600 globals and 600 methods: the u8 operand overflows into Ext words
        let mut src = String::from("struct S { a }\nimpl S {\n");
        for i in 0..600 {
            src.push_str(&format!("fn m{}(self) {{ self.a + {} }}\n", i, i));
        }
        src.push_str("}\n");
        for i in 0..600 {
            src.push_str(&format!("var g{} = {};\n", i, i));
        }
        src.push_str("fn h() { const o = S(7); var t = { };\n");
        for i in 0..600 {
            src.push_str(&format!("t.f{} = g{}; g{} = g{} + 1;\n", i, i, i, i));
        }
        src.push_str("t.f599 + t.f0 + g599 + o:m599() + o:m0() }\nprint(h());\n");
        // 599 + 0 + 600 + (7 + 599) + 7
        for optimize in [true, false] {
            let (out, r) = run(&src, optimize);
            r.unwrap();
            assert_eq!(out.trim(), "1812", "optimize={}", optimize);
        }
    });
}

#[test]
fn register_limit_is_reported_not_miscompiled() {
    // more than 250 live locals cannot fit in the register window
    let mut src = String::from("fn f() {\n");
    for i in 0..300 {
        src.push_str(&format!("var v{} = {};\n", i, i));
    }
    src.push_str("v299 }\nprint(f());\n");
    let (_, r) = run(&src, true);
    let e = r.unwrap_err();
    assert!(e.contains("250 registers"), "{}", e);

    // 250 - 2 locals still work (and so do many temporaries on top of them)
    let mut src = String::from("fn f() {\n");
    for i in 0..200 {
        src.push_str(&format!("var v{} = {};\n", i, i));
    }
    src.push_str("(v199 + (v198 + (v197 + (v196 + (v195 + v194))))) + [v1, v2, v3]:len() }\nprint(f());\n");
    let (out, r) = run(&src, true);
    r.unwrap();
    assert_eq!(out.trim(), "1182");

    // an expression that nests deeper than the free registers
    let mut src = String::from("fn f(x) { return ");
    for _ in 0..300 {
        src.push_str("1 + (");
    }
    src.push_str("x");
    for _ in 0..300 {
        src.push(')');
    }
    src.push_str("; }\nprint(f(1));\n");
    let (_, r) = run(&src, true);
    assert!(r.is_err());
}

#[test]
fn evaluation_order_with_register_targets() {
    // an operand that is a local must be read before a later operand modifies it
    let src = r#"
        fn f() {
            var x = 1;
            const a = x + { x = 5; 1 };
            var y = 1;
            y = [y, { y = 9; 2 }];
            var z = 3;
            z = z && z - 1;
            var w = 2;
            w = w * 10 + w;
            return [a, x, y[0], y[1], z, w];
        }
        print(f());
    "#;
    for optimize in [true, false] {
        let (out, r) = run(src, optimize);
        r.unwrap();
        assert_eq!(out.trim(), "[2, 5, 1, 2, 2, 22]", "optimize={}", optimize);
    }
}

#[test]
fn calls_may_change_captured_locals_only() {
    // `x` is captured, so a call can change it: the left operand is read first.
    // `n` is not captured: nothing a call does can change it, so it is read in place.
    let src = r#"
        fn f() {
            var x = 1;
            const g = || => { x = 10; 5 };
            const r = x + g();
            var y = 1;
            const h = || => y;
            y += h() + { y = 7; 1 };
            var n = 1;
            n += (|| => 2)();
            [r, x, y, n]
        }
        print(f());
    "#;
    for optimize in [true, false] {
        let (out, r) = run(src, optimize);
        r.unwrap();
        assert_eq!(out.trim(), "[6, 10, 3, 3]", "optimize={}", optimize);
    }
    let uncaptured = "fn f(c) { var s = 0; s += c(); s } f(|| => 1);";
    assert_eq!(count_ops(uncaptured, true, |o| matches!(o, Op::Move(..))), 1);
}
