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
    vm.define_native("assert", -1, assert);
    vm.define_native("panic", 1, panic);
    vm.define_native("catch_panic", -1, catch_panic);
    vm.define_native("option", 1, option);
    vm.define_native("clock", 0, clock);
    vm.define_native("time", 0, time);
    vm.define_native("exit", -1, exit);
    vm.define_native("input", -1, input);
    vm.define_native("gc", 0, gc);
    vm.define_native("mem", 0, mem);
    vm.define_native("destroy", 1, destroy);

    crate::fslib::install(vm);

    let array = lib(
        vm,
        "array",
        &[
            ("new", -1, array_new),
            ("push", -1, push),
            ("pop", 1, pop),
            ("insert", 3, insert),
            ("remove", 2, remove),
            ("len", 1, len),
            ("get", 2, array_get),
            ("first", 1, array_first),
            ("last", 1, array_last),
            ("slice", -1, array_slice),
            ("concat", -1, array_concat),
            ("extend", 2, array_extend),
            ("reverse", 1, array_reverse),
            ("copy", 1, array_copy),
            ("clear", 1, array_clear),
            ("index_of", 2, array_index_of),
            ("contains", 2, array_contains),
            ("join", -1, str_join),
            ("sort", -1, sort),
            ("map", 2, map),
            ("filter", 2, filter),
            ("reduce", -1, reduce),
            ("each", 2, each),
            ("keys", 1, keys),
            ("values", 1, values),
            ("has", 2, has),
        ],
    );
    vm.array_lib = Value::from(array);

    // methods of tables: `t:keys()`; a field of the same name wins (see `Vm::get_method`)
    let table = lib(
        vm,
        "table",
        &[
            ("len", 1, len),
            ("keys", 1, keys),
            ("values", 1, values),
            ("has", 2, has),
            ("remove", 2, remove),
            ("map", 2, map),
            ("filter", 2, filter),
            ("reduce", -1, reduce),
            ("each", 2, each),
        ],
    );
    vm.table_lib = Value::from(table);

    let math = std_module(
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
            ("atan2", 2, |vm, a| {
                Ok(Value::num(arg_num(vm, a, 0, "math.atan2")?.atan2(arg_num(vm, a, 1, "math.atan2")?)))
            }),
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
                let mut t = vm.new_array(Vec::new());
                for c in s.as_str().chars() {
                    let mut buf = [0u8; 4];
                    let cs = vm.new_string(&*c.encode_utf8(&mut buf));
                    apush(vm, &mut t, cs);
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
            ("parse_num", 1, parse_num),
        ],
    );
    vm.string_lib = Value::from(string);
}

// ---- registration helpers -----------------------------------------------------------------

fn native_table(vm: &mut Vm, funcs: &[(&'static str, i8, NativeFn)]) -> Gc<Table> {
    let mut t = vm.new_table();
    for &(fname, arity, f) in funcs {
        let n = vm.new_native(fname, arity, f);
        let key = vm.heap.intern(fname);
        let _ = t.set(Value::from(key), n);
    }
    t
}

/// A table of natives bound to a global name: the method table of a type (`string`, `array`, `table`).
pub(crate) fn lib(vm: &mut Vm, name: &str, funcs: &[(&'static str, i8, NativeFn)]) -> Gc<Table> {
    let t = native_table(vm, funcs);
    vm.define_global(name, Value::from(t));
    t
}

/// A built-in module: nothing is global, `import "std/<name>"` brings it in.
pub(crate) fn std_module(vm: &mut Vm, name: &str, funcs: &[(&'static str, i8, NativeFn)]) -> Gc<Table> {
    let t = native_table(vm, funcs);
    vm.register_module(&format!("std/{}", name), Value::from(t));
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

pub(crate) fn arg_str(vm: &Vm, a: &[Value], i: usize, f: &str) -> RtResult<Gc<Str>> {
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

/// Append to an array, keeping the GC's byte accounting honest.
pub(crate) fn apush(vm: &mut Vm, a: &mut Gc<Array>, v: Value) {
    vm.array_push(a, v);
}

fn arg_array(vm: &Vm, a: &[Value], i: usize, f: &str) -> RtResult<Gc<Array>> {
    match a.get(i) {
        Some(v) => v
            .as_gc::<Array>()
            .ok_or_else(|| rt_err(format!("{}: argument {} must be an array, got {}", f, i + 1, vm.type_of(*v)))),
        None => Err(rt_err(format!("{}: missing argument {}", f, i + 1))),
    }
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
            .or_else(|| {
                t.parse::<f64>()
                    .ok()
                    .filter(|_| t.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '.' || c == '+'))
            });
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
        Some(Kind::Array) => Ok(Value::num(unsafe { v.as_gc_unchecked::<Array>() }.items.len() as f64)),
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
    let mut arr = arg_array(vm, a, 0, "push")?;
    if a.len() < 2 {
        return Err(rt_err("push: expected a value to push"));
    }
    for &v in &a[1..] {
        apush(vm, &mut arr, v);
    }
    Ok(Value::num(arr.items.len() as f64))
}

/// `pop(a)` -> `Some(last)` or `None` for an empty array.
fn pop(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "pop")?;
    Ok(match arr.items.pop() {
        Some(v) => vm.make_some(v),
        None => vm.make_none(),
    })
}

/// `insert(a, i, v)`: `i` may be the length (then it appends).
fn insert(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "insert")?;
    let i = arg_num(vm, a, 1, "insert")?;
    let v = *a.get(2).ok_or_else(|| rt_err("insert: missing argument 3"))?;
    if i < 0.0 || i.fract() != 0.0 || i as usize > arr.items.len() {
        return Err(rt_err(format!("insert: index {} is out of bounds (length {})", i, arr.items.len())));
    }
    let before = arr.mem_size();
    arr.items.insert(i as usize, v);
    vm.heap.bytes += arr.mem_size().saturating_sub(before);
    Ok(v)
}

/// `remove(a, i)` removes and returns the item, shifting later ones down; `remove(t, key)` deletes a table key.
fn remove(vm: &mut Vm, a: &[Value]) -> R {
    if let Some(mut arr) = a.first().and_then(|v| v.as_gc::<Array>()) {
        let i = arg_num(vm, a, 1, "remove")?;
        if i < 0.0 || i.fract() != 0.0 || i as usize >= arr.items.len() {
            return Err(rt_err(format!("remove: index {} is out of bounds (length {})", i, arr.items.len())));
        }
        return Ok(arr.items.remove(i as usize));
    }
    let mut t = arg_table(vm, a, 0, "remove")?;
    if a.len() < 2 {
        return Err(rt_err("remove: missing argument 2"));
    }
    Ok(t.remove(a[1]))
}

/// Indices of an array, keys of a table.
fn keys(vm: &mut Vm, a: &[Value]) -> R {
    if let Some(arr) = a.first().and_then(|v| v.as_gc::<Array>()) {
        let items = (0..arr.items.len()).map(|i| Value::num(i as f64)).collect();
        return Ok(Value::from(vm.new_array(items)));
    }
    let t = arg_table(vm, a, 0, "keys")?;
    let mut out = vm.new_array(Vec::new());
    let mut pos = 0;
    while let Some((np, k, _)) = t.next(pos) {
        pos = np;
        apush(vm, &mut out, k);
    }
    Ok(Value::from(out))
}

/// A new array with the items of an array / the values of a table.
fn values(vm: &mut Vm, a: &[Value]) -> R {
    if let Some(arr) = a.first().and_then(|v| v.as_gc::<Array>()) {
        let items = arr.items.clone();
        return Ok(Value::from(vm.new_array(items)));
    }
    let t = arg_table(vm, a, 0, "values")?;
    let mut out = vm.new_array(Vec::new());
    let mut pos = 0;
    while let Some((np, _, v)) = t.next(pos) {
        pos = np;
        apush(vm, &mut out, v);
    }
    Ok(Value::from(out))
}

/// `has(t, key)`; for an array `has(a, i)` asks whether `i` is a valid index.
fn has(vm: &mut Vm, a: &[Value]) -> R {
    if let Some(arr) = a.first().and_then(|v| v.as_gc::<Array>()) {
        let k = *a.get(1).ok_or_else(|| rt_err("has: missing argument 2"))?;
        return Ok(Value::bool(
            k.is_num() && k.as_num().fract() == 0.0 && k.as_num() >= 0.0 && k.as_num() < arr.items.len() as f64,
        ));
    }
    let t = arg_table(vm, a, 0, "has")?;
    Ok(Value::bool(t.contains(a[1])))
}

// ---- arrays (also reachable as `arr:method()`) ---------------------------------------------

/// `array.new(n, fill?)`: `n` copies of `fill` (default `nil`).
fn array_new(vm: &mut Vm, a: &[Value]) -> R {
    let n = arg_num(vm, a, 0, "array.new")?;
    if n < 0.0 || n.fract() != 0.0 || n > 1e8 {
        return Err(rt_err("array.new: invalid length"));
    }
    let fill = a.get(1).copied().unwrap_or(Value::NIL);
    Ok(Value::from(vm.new_array(vec![fill; n as usize])))
}

/// `get(a, i)` -> `Some(item)` or `None` (the checked alternative to `a[i]`).
fn array_get(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "get")?;
    let i = arg_num(vm, a, 1, "get")?;
    Ok(if i >= 0.0 && i.fract() == 0.0 && (i as usize) < arr.items.len() {
        vm.make_some(arr.items[i as usize])
    } else {
        vm.make_none()
    })
}

fn array_first(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "first")?;
    Ok(match arr.items.first() {
        Some(&v) => vm.make_some(v),
        None => vm.make_none(),
    })
}

fn array_last(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "last")?;
    Ok(match arr.items.last() {
        Some(&v) => vm.make_some(v),
        None => vm.make_none(),
    })
}

/// `slice(a, from, to?)`: a new array; negative positions count from the end, out-of-range ones are clamped.
fn array_slice(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "slice")?;
    let n = arr.items.len();
    let from = clamp_index(arg_num(vm, a, 1, "slice")?.trunc(), n);
    let to = if a.len() > 2 && !a[2].is_nil() { clamp_index(arg_num(vm, a, 2, "slice")?.trunc(), n) } else { n };
    let items = if from < to { arr.items[from..to].to_vec() } else { Vec::new() };
    Ok(Value::from(vm.new_array(items)))
}

/// `concat(a, b, …)`: a new array.
fn array_concat(vm: &mut Vm, a: &[Value]) -> R {
    let mut items = Vec::new();
    for i in 0..a.len() {
        items.extend_from_slice(&arg_array(vm, a, i, "concat")?.items);
    }
    Ok(Value::from(vm.new_array(items)))
}

/// `extend(a, b)`: append the items of `b` to `a`.
fn array_extend(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "extend")?;
    let other = arg_array(vm, a, 1, "extend")?.items.clone();
    let before = arr.mem_size();
    arr.items.extend(other);
    vm.heap.bytes += arr.mem_size().saturating_sub(before);
    Ok(a[0])
}

/// In place; returns the array.
fn array_reverse(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "reverse")?;
    arr.items.reverse();
    Ok(a[0])
}

fn array_copy(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "copy")?;
    let items = arr.items.clone();
    Ok(Value::from(vm.new_array(items)))
}

fn array_clear(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "clear")?;
    arr.items.clear();
    Ok(a[0])
}

/// `index_of(a, v)` -> `Some(i)` of the first item equal to `v`, or `None`.
fn array_index_of(vm: &mut Vm, a: &[Value]) -> R {
    let arr = arg_array(vm, a, 0, "index_of")?;
    let needle = *a.get(1).ok_or_else(|| rt_err("index_of: missing argument 2"))?;
    let mut i = 0;
    while i < arr.items.len() {
        let item = arr.items[i];
        if vm.values_equal(item, needle)? {
            return Ok(vm.make_some(Value::num(i as f64)));
        }
        i += 1;
    }
    Ok(vm.make_none())
}

fn array_contains(vm: &mut Vm, a: &[Value]) -> R {
    let found = array_index_of(vm, a)?;
    Ok(Value::bool(found != vm.make_none()))
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

/// `panic(message)`: a bug or an impossible situation.  Unwinds the whole program (see `catch_panic`).
fn panic(vm: &mut Vm, a: &[Value]) -> R {
    let msg = vm.display(a[0])?;
    Err(rt_err(msg))
}

/// `catch_panic(f, args…)` -> `Ok(result)` or `Err(Error(message, "panic"))`.
/// A last resort for program boundaries (tests, REPL, plugins); expected failures are `Result`s.
fn catch_panic(vm: &mut Vm, a: &[Value]) -> R {
    if a.is_empty() {
        return Err(rt_err("catch_panic: missing function"));
    }
    Ok(match vm.call_protected(a[0], &a[1..]) {
        Ok(v) => vm.make_ok(v),
        Err(e) => {
            let err = vm.make_error(&e.msg, "panic");
            vm.make_err(err)
        }
    })
}

/// `parse_num(s)` -> `Ok(number)` or `Err(Error)`.
fn parse_num(vm: &mut Vm, a: &[Value]) -> R {
    let Some(s) = a[0].as_str() else {
        let shown = vm.repr(a[0])?;
        let err = vm.make_error(&format!("parse_num expects a string, got {}", shown), "type");
        return Ok(vm.make_err(err));
    };
    let text = s.as_str().to_string();
    let parsed = to_num(vm, &[a[0]])?;
    if parsed.is_num() {
        Ok(vm.make_ok(parsed))
    } else {
        let err = vm.make_error(&format!("invalid number {:?}", text), "parse");
        Ok(vm.make_err(err))
    }
}

/// `option(x)` -> `None` for `nil`, otherwise `Some(x)`.
fn option(vm: &mut Vm, a: &[Value]) -> R {
    Ok(if a[0].is_nil() { vm.make_none() } else { vm.make_some(a[0]) })
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
    vm.collect_garbage_precise();
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
    let r: String = if text.is_ascii() {
        text[start..end].to_string()
    } else {
        text.chars().skip(start).take(end - start).collect()
    };
    Ok(vm.new_string(r))
}

fn str_find(vm: &mut Vm, a: &[Value]) -> R {
    let s = arg_str(vm, a, 0, "string.find")?;
    let needle = arg_str(vm, a, 1, "string.find")?;
    let text = s.as_str();
    let start = if a.len() > 2 { clamp_index(arg_num(vm, a, 2, "string.find")?.trunc(), char_count(text)) } else { 0 };
    let byte_start =
        if text.is_ascii() { start } else { text.char_indices().nth(start).map_or(text.len(), |(b, _)| b) };
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
    let mut out = vm.new_array(Vec::new());
    if sep.as_str().is_empty() {
        for c in s.as_str().chars() {
            let mut buf = [0u8; 4];
            let cs = vm.new_string(&*c.encode_utf8(&mut buf));
            apush(vm, &mut out, cs);
        }
    } else {
        let parts: Vec<String> = s.as_str().split(sep.as_str()).map(|p| p.to_string()).collect();
        for p in parts {
            let ps = vm.new_string(p);
            apush(vm, &mut out, ps);
        }
    }
    Ok(Value::from(out))
}

fn str_join(vm: &mut Vm, a: &[Value]) -> R {
    let items: Vec<Value> = match a.first() {
        Some(v) if v.as_gc::<Array>().is_some() => v.as_gc::<Array>().unwrap().items.clone(),
        _ => {
            let t = arg_table(vm, a, 0, "join")?;
            let mut out = Vec::new();
            let mut pos = 0;
            while let Some((np, _, v)) = t.next(pos) {
                pos = np;
                out.push(v);
            }
            out
        }
    };
    let sep = if a.len() > 1 { arg_str(vm, a, 1, "join")?.as_str().to_string() } else { String::new() };
    let mut s = String::new();
    for (i, v) in items.into_iter().enumerate() {
        if i > 0 {
            s.push_str(&sep);
        }
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

/// `sort(a)` / `sort(a, less)`: in place, stable.
fn sort(vm: &mut Vm, a: &[Value]) -> R {
    let mut arr = arg_array(vm, a, 0, "sort")?;
    let cmp = a.get(1).copied().unwrap_or(Value::NIL);
    let mut items: Vec<Value> = arr.items.clone();
    vm.root(Value::from(arr));
    let r = merge_sort(vm, &mut items, cmp);
    vm.unroot(1);
    r?;
    // the callback may have changed the array; only write back if it did not
    if arr.items.len() == items.len() {
        arr.items.copy_from_slice(&items);
    }
    Ok(a[0])
}

/// `map(a, f)` -> array; `map(t, f)` -> table with the same keys.  `f(value)` or `f(value, key)`.
fn map(vm: &mut Vm, a: &[Value]) -> R {
    let f = *a.get(1).ok_or_else(|| rt_err("map: missing function"))?;
    if let Some(arr) = a[0].as_gc::<Array>() {
        let mut out = vm.new_array(Vec::new());
        vm.root(Value::from(out));
        let r = (|| -> RtResult<()> {
            let mut i = 0;
            while i < arr.items.len() {
                let v = call_cb(vm, f, arr.items[i], Value::num(i as f64))?;
                apush(vm, &mut out, v);
                i += 1;
            }
            Ok(())
        })();
        vm.unroot(1);
        r?;
        return Ok(Value::from(out));
    }
    let t = arg_table(vm, a, 0, "map")?;
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
    let f = *a.get(1).ok_or_else(|| rt_err("filter: missing function"))?;
    if let Some(arr) = a[0].as_gc::<Array>() {
        let mut out = vm.new_array(Vec::new());
        vm.root(Value::from(out));
        let r = (|| -> RtResult<()> {
            let mut i = 0;
            while i < arr.items.len() {
                let v = arr.items[i];
                if call_cb(vm, f, v, Value::num(i as f64))?.is_truthy() {
                    apush(vm, &mut out, v);
                }
                i += 1;
            }
            Ok(())
        })();
        vm.unroot(1);
        r?;
        return Ok(Value::from(out));
    }
    let t = arg_table(vm, a, 0, "filter")?;
    let mut out = vm.new_table();
    vm.root(Value::from(out));
    let mut pos = 0;
    let r = (|| -> RtResult<()> {
        while let Some((np, k, v)) = t.next(pos) {
            pos = np;
            if call_cb(vm, f, v, k)?.is_truthy() {
                tset(vm, &mut out, k, v);
            }
        }
        Ok(())
    })();
    vm.unroot(1);
    r?;
    Ok(Value::from(out))
}

/// `reduce(c, f, init?)`; without `init` the first item starts the fold (an empty collection gives `nil`).
fn reduce(vm: &mut Vm, a: &[Value]) -> R {
    let f = a.get(1).copied().ok_or_else(|| rt_err("reduce: missing function"))?;
    if let Some(arr) = a[0].as_gc::<Array>() {
        let mut i = 0;
        let mut acc = if a.len() > 2 {
            a[2]
        } else {
            match arr.items.first() {
                Some(&v) => {
                    i = 1;
                    v
                }
                None => return Ok(Value::NIL),
            }
        };
        while i < arr.items.len() {
            let v = arr.items[i];
            vm.root(acc);
            let r = vm.call_value(f, &[acc, v]);
            vm.unroot(1);
            acc = r?;
            i += 1;
        }
        return Ok(acc);
    }
    let t = arg_table(vm, a, 0, "reduce")?;
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
    let f = *a.get(1).ok_or_else(|| rt_err("each: missing function"))?;
    if let Some(arr) = a[0].as_gc::<Array>() {
        let mut i = 0;
        while i < arr.items.len() {
            call_cb(vm, f, arr.items[i], Value::num(i as f64))?;
            i += 1;
        }
        return Ok(Value::NIL);
    }
    let t = arg_table(vm, a, 0, "each")?;
    let mut pos = 0;
    while let Some((np, k, v)) = t.next(pos) {
        pos = np;
        call_cb(vm, f, v, k)?;
    }
    Ok(Value::NIL)
}
