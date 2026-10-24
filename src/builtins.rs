//! Native standard library.
//!
//! Natives are plain `fn(&mut Vm, &[Value]) -> RtResult<Value>`.  Allocation
//! never triggers a collection, so natives can build objects freely; a
//! native that calls back into the VM (`call_value`) must keep fresh objects
//! reachable with `vm.root(..)` / `vm.unroot(..)`.

use crate::object::*;
use crate::table::Table;
use crate::value::Value;
use crate::vm::{char_count, rt_err, RtResult, Vm};
use std::io::{BufRead, Write};

type R = RtResult<Value>;

pub fn install(vm: &mut Vm) {
    // ---- core
    vm.define_native("print", -1, print);
    vm.define_native("write", -1, write);
    vm.define_native("type", 1, type_of);
    vm.define_native("str", 1, to_str);
    vm.define_native("num", 1, to_num);
    vm.define_native("len", 1, len);
    vm.define_native("push", -1, push);
    vm.define_native("pop", 1, pop);
    vm.define_native("insert", 3, insert);
    vm.define_native("remove", 2, remove);
    vm.define_native("keys", 1, keys);
    vm.define_native("values", 1, values);
    vm.define_native("has", 2, has);
    vm.define_native("assert", -1, assert);
    vm.define_native("error", 1, error);
    vm.define_native("pcall", -1, pcall);
    vm.define_native("clock", 0, clock);
    vm.define_native("time", 0, time);
    vm.define_native("exit", -1, exit);
    vm.define_native("input", -1, input);
    vm.define_native("gc", 0, gc);
    vm.define_native("mem", 0, mem);
    vm.define_native("destroy", 1, destroy);
    // ---- functional helpers
    vm.define_native("sort", -1, sort);
    vm.define_native("map", 2, map);
    vm.define_native("filter", 2, filter);
    vm.define_native("reduce", -1, reduce);
    vm.define_native("each", 2, each);

    let math = lib(
        vm,
        "math",
        &[
            ("floor", 1, |_, a| num1(a, "math.floor", f64::floor)),
            ("ceil", 1, |_, a| num1(a, "math.ceil", f64::ceil)),
            ("round", 1, |_, a| num1(a, "math.round", f64::round)),
            ("trunc", 1, |_, a| num1(a, "math.trunc", f64::trunc)),
            ("abs", 1, |_, a| num1(a, "math.abs", f64::abs)),
            ("sqrt", 1, |_, a| num1(a, "math.sqrt", f64::sqrt)),
            ("sin", 1, |_, a| num1(a, "math.sin", f64::sin)),
            ("cos", 1, |_, a| num1(a, "math.cos", f64::cos)),
            ("tan", 1, |_, a| num1(a, "math.tan", f64::tan)),
            ("asin", 1, |_, a| num1(a, "math.asin", f64::asin)),
            ("acos", 1, |_, a| num1(a, "math.acos", f64::acos)),
            ("atan", 1, |_, a| num1(a, "math.atan", f64::atan)),
            ("exp", 1, |_, a| num1(a, "math.exp", f64::exp)),
            ("ln", 1, |_, a| num1(a, "math.ln", f64::ln)),
            ("log2", 1, |_, a| num1(a, "math.log2", f64::log2)),
            ("log10", 1, |_, a| num1(a, "math.log10", f64::log10)),
            ("sign", 1, |_, a| num1(a, "math.sign", |x| if x == 0.0 || x.is_nan() { x } else { x.signum() })),
            ("atan2", 2, |vm, a| Ok(Value::num(arg_num(vm, a, 0, "math.atan2")?.atan2(arg_num(vm, a, 1, "math.atan2")?)))),
            ("pow", 2, |vm, a| Ok(Value::num(arg_num(vm, a, 0, "math.pow")?.powf(arg_num(vm, a, 1, "math.pow")?)))),
            ("min", -1, |vm, a| minmax(vm, a, "math.min", true)),
            ("max", -1, |vm, a| minmax(vm, a, "math.max", false)),
            ("random", -1, random),
            ("seed", 1, |vm, a| {
                vm.rng_state = (arg_num(vm, a, 0, "math.seed")? as u64) ^ 0x9E37_79B9_7F4A_7C15;
                Ok(Value::NIL)
            }),
        ],
    );
    set_num(vm, math, "pi", std::f64::consts::PI);
    set_num(vm, math, "e", std::f64::consts::E);
    set_num(vm, math, "huge", f64::INFINITY);

    let string = lib(
        vm,
        "string",
        &[
            ("len", 1, len),
            ("upper", 1, |vm, a| str1(vm, a, "string.upper", |s| s.to_uppercase())),
            ("lower", 1, |vm, a| str1(vm, a, "string.lower", |s| s.to_lowercase())),
            ("trim", 1, |vm, a| str1(vm, a, "string.trim", |s| s.trim().to_string())),
            ("reverse", 1, |vm, a| str1(vm, a, "string.reverse", |s| s.chars().rev().collect())),
            ("sub", -1, str_sub),
            ("find", -1, str_find),
            ("contains", 2, |vm, a| {
                let (s, n) = (arg_str(vm, a, 0, "string.contains")?, arg_str(vm, a, 1, "string.contains")?);
                Ok(Value::bool(s.as_str().contains(n.as_str())))
            }),
            ("starts_with", 2, |vm, a| {
                let (s, n) = (arg_str(vm, a, 0, "string.starts_with")?, arg_str(vm, a, 1, "string.starts_with")?);
                Ok(Value::bool(s.as_str().starts_with(n.as_str())))
            }),
            ("ends_with", 2, |vm, a| {
                let (s, n) = (arg_str(vm, a, 0, "string.ends_with")?, arg_str(vm, a, 1, "string.ends_with")?);
                Ok(Value::bool(s.as_str().ends_with(n.as_str())))
            }),
            ("split", 2, str_split),
            ("join", -1, str_join),
            ("replace", 3, |vm, a| {
                let (s, f, t) = (
                    arg_str(vm, a, 0, "string.replace")?,
                    arg_str(vm, a, 1, "string.replace")?,
                    arg_str(vm, a, 2, "string.replace")?,
                );
                let r = s.as_str().replace(f.as_str(), t.as_str());
                Ok(vm.new_string(r))
            }),
            ("rep", 2, |vm, a| {
                let s = arg_str(vm, a, 0, "string.rep")?;
                let n = arg_num(vm, a, 1, "string.rep")?;
                if n < 0.0 || n * s.as_str().len() as f64 > 1e9 {
                    return Err(rt_err("string.rep: invalid repeat count"));
                }
                let r = s.as_str().repeat(n as usize);
                Ok(vm.new_string(r))
            }),
            ("chars", 1, |vm, a| {
                let s = arg_str(vm, a, 0, "string.chars")?;
                let mut t = vm.new_table();
                for c in s.as_str().chars() {
                    let mut buf = [0u8; 4];
                    let cs = vm.new_string(&*c.encode_utf8(&mut buf));
                    tpush(vm, &mut t, cs);
                }
                Ok(Value::from(t))
            }),
            ("byte", -1, |vm, a| {
                let s = arg_str(vm, a, 0, "string.byte")?;
                let i = if a.len() > 1 { arg_num(vm, a, 1, "string.byte")? as usize } else { 0 };
                Ok(s.as_str().as_bytes().get(i).map_or(Value::NIL, |&b| Value::num(b as f64)))
            }),
            ("char", 1, |vm, a| {
                let n = arg_num(vm, a, 0, "string.char")?;
                match char::from_u32(n as u32) {
                    Some(c) => {
                        let mut buf = [0u8; 4];
                        Ok(vm.new_string(&*c.encode_utf8(&mut buf)))
                    }
                    None => Err(rt_err("string.char: invalid code point")),
                }
            }),
            ("format", -1, format),
        ],
    );
    vm.string_lib = Value::from(string);
}

// ---- registration helpers -----------------------------------------------------------------

fn lib(vm: &mut Vm, name: &str, funcs: &[(&'static str, i8, NativeFn)]) -> Gc<Table> {
    let mut t = vm.new_table();
    for &(fname, arity, f) in funcs {
        let n = vm.new_native(fname, arity, f);
        let key = vm.heap.intern(fname);
        let _ = t.set(Value::from(key), n);
    }
    vm.define_global(name, Value::from(t));
    t
}

fn set_num(vm: &mut Vm, mut t: Gc<Table>, name: &str, v: f64) {
    let key = vm.heap.intern(name);
    let _ = t.set(Value::from(key), Value::num(v));
}

// ---- argument helpers ---------------------------------------------------------------------

fn arg_num(vm: &Vm, a: &[Value], i: usize, f: &str) -> RtResult<f64> {
    match a.get(i) {
        Some(v) if v.is_num() => Ok(v.as_num()),
        Some(v) => Err(rt_err(format!("{}: argument {} must be a number, got {}", f, i + 1, vm.type_of(*v)))),
        None => Err(rt_err(format!("{}: missing argument {}", f, i + 1))),
    }
}

fn arg_str(vm: &Vm, a: &[Value], i: usize, f: &str) -> RtResult<Gc<Str>> {
    match a.get(i) {
        Some(v) => v
            .as_str()
            .ok_or_else(|| rt_err(format!("{}: argument {} must be a string, got {}", f, i + 1, vm.type_of(*v)))),
        None => Err(rt_err(format!("{}: missing argument {}", f, i + 1))),
    }
}

fn arg_table(vm: &Vm, a: &[Value], i: usize, f: &str) -> RtResult<Gc<Table>> {
    match a.get(i) {
        Some(v) => v
            .as_gc::<Table>()
            .ok_or_else(|| rt_err(format!("{}: argument {} must be a table, got {}", f, i + 1, vm.type_of(*v)))),
        None => Err(rt_err(format!("{}: missing argument {}", f, i + 1))),
    }
}

fn num1(a: &[Value], f: &str, op: impl Fn(f64) -> f64) -> R {
    match a.first() {
        Some(v) if v.is_num() => Ok(Value::num(op(v.as_num()))),
        _ => Err(rt_err(format!("{}: argument 1 must be a number", f))),
    }
}

fn str1(vm: &mut Vm, a: &[Value], f: &str, op: impl Fn(&str) -> String) -> R {
    let s = arg_str(vm, a, 0, f)?;
    let r = op(s.as_str());
    Ok(vm.new_string(r))
}

/// Append to a table's array part, keeping the GC's byte accounting honest.
fn tpush(vm: &mut Vm, t: &mut Gc<Table>, v: Value) {
    let before = t.mem_size();
    t.push(v);
    vm.heap.bytes += t.mem_size().saturating_sub(before);
}

fn tset(vm: &mut Vm, t: &mut Gc<Table>, k: Value, v: Value) {
    let before = t.mem_size();
    let _ = t.set(k, v);
    vm.heap.bytes += t.mem_size().saturating_sub(before);
}

// ---- core ---------------------------------------------------------------------------------

fn print(vm: &mut Vm, a: &[Value]) -> R {
    let mut s = String::new();
    for (i, &v) in a.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        vm.write_value(v, &mut s, false, 0)?;
    }
    s.push('\n');
    let _ = vm.out.write_all(s.as_bytes());
    Ok(Value::NIL)
}

fn write(vm: &mut Vm, a: &[Value]) -> R {
    let mut s = String::new();
    for &v in a {
        vm.write_value(v, &mut s, false, 0)?;
    }
    let _ = vm.out.write_all(s.as_bytes());
    Ok(Value::NIL)
}

fn type_of(vm: &mut Vm, a: &[Value]) -> R {
    let name = vm.type_of(a[0]);
    Ok(vm.new_string(name))
}

fn to_str(vm: &mut Vm, a: &[Value]) -> R {
    if a[0].is_str() {
        return Ok(a[0]);
    }
    let s = vm.display(a[0])?;
    Ok(vm.new_string(s))
}

fn to_num(vm: &mut Vm, a: &[Value]) -> R {
    let v = a[0];
    if v.is_num() {
        return Ok(v);
    }
    if let Some(s) = v.as_str() {
        let t = s.as_str().trim();
        let parsed = crate::text::parse_number(t)
            .or_else(|| t.strip_prefix('-').and_then(crate::text::parse_number).map(|n| -n))
            .or_else(|| t.parse::<f64>().ok().filter(|_| t.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '.' || c == '+')));
        return Ok(parsed.map_or(Value::NIL, Value::num));
    }
    if v.is_bool() {
        return Ok(Value::num(if v == Value::TRUE { 1.0 } else { 0.0 }));
    }
    let _ = vm;
    Ok(Value::NIL)
}

fn len(vm: &mut Vm, a: &[Value]) -> R {
    let v = a[0];
    match v.obj_kind() {
        Some(Kind::Str) => Ok(Value::num(char_count(unsafe { v.as_gc_unchecked::<Str>() }.as_str()) as f64)),
        Some(Kind::Table) => Ok(Value::num(unsafe { v.as_gc_unchecked::<Table>() }.len() as f64)),
        Some(Kind::Instance) => {
            let inst = unsafe { v.as_gc_unchecked::<Instance>() };
            let hook = inst.ty.hook(Hook::Len);
            if hook.is_nil() {
                Err(rt_err(format!("struct '{}' has no `len` method", inst.ty.name.as_str())))
            } else {
                vm.call_value(hook, &[v])
            }
        }
        _ => Err(rt_err(format!("len: cannot take the length of {}", vm.type_of(v)))),
    }
}

fn push(vm: &mut Vm, a: &[Value]) -> R {
    let mut t = arg_table(vm, a, 0, "push")?;
    if a.len() < 2 {
        return Err(rt_err("push: expected a value to push"));
    }
    for &v in &a[1..] {
        tpush(vm, &mut t, v);
    }
    Ok(Value::num(t.arr.len() as f64))
}

fn pop(vm: &mut Vm, a: &[Value]) -> R {
    let mut t = arg_table(vm, a, 0, "pop")?;
    Ok(t.pop())
}

fn insert(vm: &mut Vm, a: &[Value]) -> R {
    let mut t = arg_table(vm, a, 0, "insert")?;
    let i = arg_num(vm, a, 1, "insert")?;
    if i < 0.0 || i.fract() != 0.0 || i as usize > t.arr.len() {
        return Err(rt_err("insert: index out of range"));
    }
    if a[2].is_nil() {
        return Ok(Value::NIL);
    }
    let before = t.mem_size();
    t.arr.insert(i as usize, a[2]);
    vm.heap.bytes += t.mem_size().saturating_sub(before);
    Ok(a[2])
}

/// `remove(t, i)` shifts later array items down; for other keys it deletes the key.
fn remove(vm: &mut Vm, a: &[Value]) -> R {
    let mut t = arg_table(vm, a, 0, "remove")?;
    let k = a[1];
    if k.is_num() {
        let n = k.as_num();
        let i = n as usize;
        if i as f64 == n && i < t.arr.len() {
            return Ok(t.arr.remove(i));
        }
    }
    Ok(t.remove(k))
}

fn keys(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "keys")?;
    let mut out = vm.new_table();
    let mut pos = 0;
    while let Some((np, k, _)) = t.next(pos) {
        pos = np;
        tpush(vm, &mut out, k);
    }
    Ok(Value::from(out))
}

fn values(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "values")?;
    let mut out = vm.new_table();
    let mut pos = 0;
    while let Some((np, _, v)) = t.next(pos) {
        pos = np;
        tpush(vm, &mut out, v);
    }
    Ok(Value::from(out))
}

fn has(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "has")?;
    Ok(Value::bool(t.contains(a[1])))
}

fn assert(vm: &mut Vm, a: &[Value]) -> R {
    if a.is_empty() {
        return Err(rt_err("assert: missing condition"));
    }
    if a[0].is_falsy() {
        let msg = match a.get(1) {
            Some(&m) => vm.display(m)?,
            None => "assertion failed".to_string(),
        };
        return Err(rt_err(msg));
    }
    Ok(a[0])
}

fn error(vm: &mut Vm, a: &[Value]) -> R {
    let msg = vm.display(a[0])?;
    Err(rt_err(msg))
}

/// `pcall(f, args…)` -> `[true, result]` or `[false, message]`.
fn pcall(vm: &mut Vm, a: &[Value]) -> R {
    if a.is_empty() {
        return Err(rt_err("pcall: missing function"));
    }
    let r = vm.call_protected(a[0], &a[1..]);
    let mut out = vm.new_table();
    match r {
        Ok(v) => {
            tpush(vm, &mut out, Value::TRUE);
            tpush(vm, &mut out, v);
        }
        Err(e) => {
            let m = vm.new_string(e.msg);
            tpush(vm, &mut out, Value::FALSE);
            tpush(vm, &mut out, m);
        }
    }
    Ok(Value::from(out))
}

fn clock(_: &mut Vm, _: &[Value]) -> R {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    Ok(Value::num(start.elapsed().as_secs_f64()))
}

fn time(_: &mut Vm, _: &[Value]) -> R {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    Ok(Value::num(d))
}

fn exit(vm: &mut Vm, a: &[Value]) -> R {
    let code = if a.is_empty() { 0 } else { arg_num(vm, a, 0, "exit")? as i32 };
    vm.shutdown();
    vm.flush();
    std::process::exit(code);
}

fn input(vm: &mut Vm, a: &[Value]) -> R {
    if let Some(&p) = a.first() {
        let s = vm.display(p)?;
        let _ = vm.out.write_all(s.as_bytes());
    }
    vm.flush();
    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => Ok(Value::NIL),
        Ok(_) => {
            while line.ends_with('\n') || line.ends_with('\r') {
                line.pop();
            }
            Ok(vm.new_string(line))
        }
    }
}

fn gc(vm: &mut Vm, _: &[Value]) -> R {
    let before = vm.heap.bytes;
    vm.collect_garbage();
    Ok(Value::num(before.saturating_sub(vm.heap.bytes) as f64))
}

fn mem(vm: &mut Vm, _: &[Value]) -> R {
    Ok(Value::num(vm.heap.bytes as f64))
}

fn destroy(vm: &mut Vm, a: &[Value]) -> R {
    vm.destroy_now(a[0])?;
    Ok(Value::NIL)
}

// ---- math ---------------------------------------------------------------------------------

fn minmax(vm: &mut Vm, a: &[Value], f: &str, min: bool) -> R {
    if a.is_empty() {
        return Err(rt_err(format!("{}: expected at least one argument", f)));
    }
    let mut best = arg_num(vm, a, 0, f)?;
    for i in 1..a.len() {
        let x = arg_num(vm, a, i, f)?;
        if (min && x < best) || (!min && x > best) {
            best = x;
        }
    }
    Ok(Value::num(best))
}

fn next_random(vm: &mut Vm) -> f64 {
    // xorshift64*
    let mut x = vm.rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    vm.rng_state = x;
    let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    (r >> 11) as f64 / (1u64 << 53) as f64
}

/// `random()` in [0,1), `random(n)` integer in [0,n), `random(a,b)` integer in [a,b].
fn random(vm: &mut Vm, a: &[Value]) -> R {
    let r = next_random(vm);
    match a.len() {
        0 => Ok(Value::num(r)),
        1 => {
            let n = arg_num(vm, a, 0, "math.random")?;
            if n < 1.0 {
                return Err(rt_err("math.random: upper bound must be >= 1"));
            }
            Ok(Value::num((r * n).floor()))
        }
        _ => {
            let (lo, hi) = (arg_num(vm, a, 0, "math.random")?, arg_num(vm, a, 1, "math.random")?);
            if hi < lo {
                return Err(rt_err("math.random: empty range"));
            }
            Ok(Value::num(lo + (r * (hi - lo + 1.0)).floor()))
        }
    }
}

// ---- strings ------------------------------------------------------------------------------

/// Map a possibly negative char index into `0..=len`.
fn clamp_index(i: f64, len: usize) -> usize {
    let len = len as f64;
    let i = if i < 0.0 { i + len } else { i };
    i.clamp(0.0, len) as usize
}

fn str_sub(vm: &mut Vm, a: &[Value]) -> R {
    let s = arg_str(vm, a, 0, "string.sub")?;
    let text = s.as_str();
    let n = char_count(text);
    let start = clamp_index(arg_num(vm, a, 1, "string.sub")?.trunc(), n);
    let end = if a.len() > 2 && !a[2].is_nil() { clamp_index(arg_num(vm, a, 2, "string.sub")?.trunc(), n) } else { n };
    if start >= end {
        return Ok(vm.new_string(""));
    }
    let r: String = if text.is_ascii() { text[start..end].to_string() } else { text.chars().skip(start).take(end - start).collect() };
    Ok(vm.new_string(r))
}

fn str_find(vm: &mut Vm, a: &[Value]) -> R {
    let s = arg_str(vm, a, 0, "string.find")?;
    let needle = arg_str(vm, a, 1, "string.find")?;
    let text = s.as_str();
    let start = if a.len() > 2 { clamp_index(arg_num(vm, a, 2, "string.find")?.trunc(), char_count(text)) } else { 0 };
    let byte_start = if text.is_ascii() { start } else { text.char_indices().nth(start).map_or(text.len(), |(b, _)| b) };
    match text[byte_start..].find(needle.as_str()) {
        Some(off) => {
            let b = byte_start + off;
            let ci = if text.is_ascii() { b } else { text[..b].chars().count() };
            Ok(Value::num(ci as f64))
        }
        None => Ok(Value::NIL),
    }
}

fn str_split(vm: &mut Vm, a: &[Value]) -> R {
    let s = arg_str(vm, a, 0, "string.split")?;
    let sep = arg_str(vm, a, 1, "string.split")?;
    let mut out = vm.new_table();
    if sep.as_str().is_empty() {
        for c in s.as_str().chars() {
            let mut buf = [0u8; 4];
            let cs = vm.new_string(&*c.encode_utf8(&mut buf));
            tpush(vm, &mut out, cs);
        }
    } else {
        let parts: Vec<String> = s.as_str().split(sep.as_str()).map(|p| p.to_string()).collect();
        for p in parts {
            let ps = vm.new_string(p);
            tpush(vm, &mut out, ps);
        }
    }
    Ok(Value::from(out))
}

fn str_join(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "string.join")?;
    let sep = if a.len() > 1 { arg_str(vm, a, 1, "string.join")?.as_str().to_string() } else { String::new() };
    let mut s = String::new();
    let mut pos = 0;
    let mut first = true;
    while let Some((np, _, v)) = t.next(pos) {
        pos = np;
        if !first {
            s.push_str(&sep);
        }
        first = false;
        vm.write_value(v, &mut s, false, 0)?;
    }
    Ok(vm.new_string(s))
}

/// `format("x = {}, y = {:.2}, name = {:>8}", …)`
fn format(vm: &mut Vm, a: &[Value]) -> R {
    let f = arg_str(vm, a, 0, "string.format")?;
    let mut out = String::new();
    let mut next = 1;
    let mut chars = f.as_str().chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut spec = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(c) => spec.push(c),
                        None => return Err(rt_err("string.format: unclosed `{`")),
                    }
                }
                let Some(&arg) = a.get(next) else {
                    return Err(rt_err("string.format: not enough arguments"));
                };
                next += 1;
                let spec = spec.strip_prefix(':').unwrap_or(&spec);
                let mut text = String::new();
                let (mut align_left, mut width, mut prec) = (false, 0usize, None::<usize>);
                let mut rest = spec;
                if let Some(r) = rest.strip_prefix('<') {
                    align_left = true;
                    rest = r;
                } else if let Some(r) = rest.strip_prefix('>') {
                    rest = r;
                }
                let (w, p) = match rest.split_once('.') {
                    Some((w, p)) => (w, Some(p)),
                    None => (rest, None),
                };
                if !w.is_empty() {
                    width = w.parse().map_err(|_| rt_err("string.format: bad width"))?;
                }
                if let Some(p) = p {
                    prec = Some(p.parse().map_err(|_| rt_err("string.format: bad precision"))?);
                }
                match (prec, arg.is_num()) {
                    (Some(p), true) => text = format!("{:.*}", p, arg.as_num()),
                    _ => vm.write_value(arg, &mut text, false, 0)?,
                }
                let n = text.chars().count();
                if n < width {
                    let pad = " ".repeat(width - n);
                    if align_left {
                        text.push_str(&pad);
                    } else {
                        text.insert_str(0, &pad);
                    }
                }
                out.push_str(&text);
            }
            c => out.push(c),
        }
    }
    Ok(vm.new_string(out))
}

// ---- higher order helpers --------------------------------------------------------------------

/// Call a callback with `(value)` or, if it takes two parameters, `(value, key)`.
fn call_cb(vm: &mut Vm, f: Value, v: Value, k: Value) -> R {
    if let Some(c) = f.as_gc::<Closure>() {
        if c.proto.arity >= 2 {
            return vm.call_value(f, &[v, k]);
        }
    }
    vm.call_value(f, &[v])
}

fn less(vm: &mut Vm, cmp: Value, a: Value, b: Value) -> RtResult<bool> {
    if !cmp.is_nil() {
        return Ok(vm.call_value(cmp, &[a, b])?.is_truthy());
    }
    if a.is_num() && b.is_num() {
        return Ok(a.as_num() < b.as_num());
    }
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        return Ok(x.as_str() < y.as_str());
    }
    Err(rt_err(format!("sort: cannot compare {} and {}", vm.type_of(a), vm.type_of(b))))
}

fn merge_sort(vm: &mut Vm, v: &mut Vec<Value>, cmp: Value) -> RtResult<()> {
    let n = v.len();
    if n < 2 {
        return Ok(());
    }
    let mut buf = v.clone();
    let mut width = 1;
    while width < n {
        let mut i = 0;
        while i < n {
            let mid = (i + width).min(n);
            let hi = (i + 2 * width).min(n);
            let (mut l, mut r, mut k) = (i, mid, i);
            while l < mid && r < hi {
                // stable: take from the right only when strictly less
                if less(vm, cmp, v[r], v[l])? {
                    buf[k] = v[r];
                    r += 1;
                } else {
                    buf[k] = v[l];
                    l += 1;
                }
                k += 1;
            }
            while l < mid {
                buf[k] = v[l];
                l += 1;
                k += 1;
            }
            while r < hi {
                buf[k] = v[r];
                r += 1;
                k += 1;
            }
            i += 2 * width;
        }
        std::mem::swap(v, &mut buf);
        width *= 2;
    }
    Ok(())
}

/// `sort(t)` / `sort(t, less)`: in-place, stable, array part only.
fn sort(vm: &mut Vm, a: &[Value]) -> R {
    let mut t = arg_table(vm, a, 0, "sort")?;
    let cmp = a.get(1).copied().unwrap_or(Value::NIL);
    let mut items: Vec<Value> = t.arr.clone();
    merge_sort(vm, &mut items, cmp)?;
    // the callback may have changed the table; only write back if it did not
    if t.arr.len() == items.len() {
        t.arr.copy_from_slice(&items);
    }
    Ok(a[0])
}

fn map(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "map")?;
    let f = a[1];
    let mut out = vm.new_table();
    vm.root(Value::from(out));
    let mut pos = 0;
    let r = (|| -> RtResult<()> {
        while let Some((np, k, v)) = t.next(pos) {
            pos = np;
            let r = call_cb(vm, f, v, k)?;
            tset(vm, &mut out, k, r);
        }
        Ok(())
    })();
    vm.unroot(1);
    r?;
    Ok(Value::from(out))
}

fn filter(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "filter")?;
    let f = a[1];
    let mut out = vm.new_table();
    vm.root(Value::from(out));
    let mut pos = 0;
    let r = (|| -> RtResult<()> {
        loop {
            let in_array = pos < t.arr.len();
            let Some((np, k, v)) = t.next(pos) else { break };
            pos = np;
            if call_cb(vm, f, v, k)?.is_truthy() {
                if in_array {
                    tpush(vm, &mut out, v);
                } else {
                    tset(vm, &mut out, k, v);
                }
            }
        }
        Ok(())
    })();
    vm.unroot(1);
    r?;
    Ok(Value::from(out))
}

fn reduce(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "reduce")?;
    let f = a.get(1).copied().ok_or_else(|| rt_err("reduce: missing function"))?;
    let mut pos = 0;
    let mut acc = if a.len() > 2 {
        a[2]
    } else {
        match t.next(0) {
            Some((np, _, v)) => {
                pos = np;
                v
            }
            None => return Ok(Value::NIL),
        }
    };
    while let Some((np, _, v)) = t.next(pos) {
        pos = np;
        vm.root(acc);
        let r = vm.call_value(f, &[acc, v]);
        vm.unroot(1);
        acc = r?;
    }
    Ok(acc)
}

fn each(vm: &mut Vm, a: &[Value]) -> R {
    let t = arg_table(vm, a, 0, "each")?;
    let f = a[1];
    let mut pos = 0;
    while let Some((np, k, v)) = t.next(pos) {
        pos = np;
        call_cb(vm, f, v, k)?;
    }
    Ok(Value::NIL)
}
