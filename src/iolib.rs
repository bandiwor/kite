//! The `io` library: standard input and output.
//!
//! Output goes through the VM's buffered stdout (the same sink as `print`), so ordering with
//! `print` is preserved; `eprint`/`ewrite` flush it first and then write to stderr.  Reading
//! `read_line` is an `Option` (`None` at end of input; invalid UTF-8 is replaced); the
//! whole-input readers return a `Result` whose error `kind` is `"utf8"` or `"io"`.

use crate::builtins::{apush, arg_str};
use crate::value::Value;
use crate::vm::{RtResult, Vm};
use std::io::{self, BufRead, Read, Write};

type R = RtResult<Value>;

pub fn install(vm: &mut Vm) {
    crate::builtins::std_module(
        vm,
        "io",
        &[
            ("write", -1, |vm, a| out(vm, a, "", false)),
            ("writeln", -1, |vm, a| out(vm, a, "", true)),
            ("print", -1, |vm, a| out(vm, a, " ", true)),
            ("eprint", -1, |vm, a| err_out(vm, a, " ", true)),
            ("ewrite", -1, |vm, a| err_out(vm, a, "", false)),
            ("flush", 0, flush),
            ("read_line", 0, |vm, _| Ok(read_line(vm))),
            ("read_all", 0, read_all),
            ("lines", 0, lines),
            ("prompt", 1, prompt),
        ],
    );
}

/// The text of `a` joined by `sep`, in the same form `print` shows it.
fn render(vm: &mut Vm, a: &[Value], sep: &str, newline: bool) -> RtResult<String> {
    let mut s = String::new();
    for (i, &v) in a.iter().enumerate() {
        if i > 0 {
            s.push_str(sep);
        }
        vm.write_value(v, &mut s, false, 0)?;
    }
    if newline {
        s.push('\n');
    }
    Ok(s)
}

fn out(vm: &mut Vm, a: &[Value], sep: &str, newline: bool) -> R {
    let s = render(vm, a, sep, newline)?;
    let _ = vm.out.write_all(s.as_bytes());
    Ok(Value::NIL)
}

fn err_out(vm: &mut Vm, a: &[Value], sep: &str, newline: bool) -> R {
    let s = render(vm, a, sep, newline)?;
    vm.flush();
    let _ = io::stderr().write_all(s.as_bytes());
    Ok(Value::NIL)
}

/// `io.flush()` -> `Ok(nil)`.
fn flush(vm: &mut Vm, _: &[Value]) -> R {
    let r = vm.out.flush();
    Ok(match r {
        Ok(()) => vm.make_ok(Value::NIL),
        Err(e) => failure(vm, &e),
    })
}

fn failure(vm: &mut Vm, e: &io::Error) -> Value {
    let kind = if e.kind() == io::ErrorKind::InvalidData { "utf8" } else { "io" };
    let msg = if kind == "utf8" { "input is not valid UTF-8 text".to_string() } else { e.to_string() };
    let err = vm.make_error(&msg, kind);
    vm.make_err(err)
}

/// `Some(line)` without its line ending, `None` at end of input (or on a read error).
fn read_line(vm: &mut Vm) -> Value {
    vm.flush();
    let mut buf = Vec::new();
    match io::stdin().lock().read_until(b'\n', &mut buf) {
        Ok(0) | Err(_) => vm.make_none(),
        Ok(_) => {
            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }
            let s = vm.new_string(String::from_utf8_lossy(&buf));
            vm.make_some(s)
        }
    }
}

fn read_text() -> io::Result<String> {
    let mut buf = Vec::new();
    io::stdin().lock().read_to_end(&mut buf)?;
    String::from_utf8(buf).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "not utf-8"))
}

/// `io.read_all()` -> `Ok(everything left on stdin)`.
fn read_all(vm: &mut Vm, _: &[Value]) -> R {
    vm.flush();
    Ok(match read_text() {
        Ok(t) => {
            let s = vm.new_string(t);
            vm.make_ok(s)
        }
        Err(e) => failure(vm, &e),
    })
}

/// `io.lines()` -> `Ok([line, …])` for the rest of stdin.
fn lines(vm: &mut Vm, _: &[Value]) -> R {
    vm.flush();
    let text = match read_text() {
        Ok(t) => t,
        Err(e) => return Ok(failure(vm, &e)),
    };
    let mut t = vm.new_array(Vec::new());
    for line in text.lines() {
        let s = vm.new_string(line);
        apush(vm, &mut t, s);
    }
    Ok(vm.make_ok(Value::from(t)))
}

/// `io.prompt(text)`: show `text` (no newline), then `read_line()`.
fn prompt(vm: &mut Vm, a: &[Value]) -> R {
    let text = if a[0].is_str() { arg_str(vm, a, 0, "io.prompt")?.as_str().to_string() } else { vm.display(a[0])? };
    let _ = vm.out.write_all(text.as_bytes());
    Ok(read_line(vm))
}
