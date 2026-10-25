//! The `fs` library: whole-file operations and path helpers.
//!
//! Everything that can fail for reasons outside the program (missing file, permissions, a full
//! disk) returns a `Result`: `Ok(value)` or `Err(Error)` whose `kind` is one of `"not_found"`,
//! `"permission"`, `"exists"`, `"utf8"` or `"io"`.  Passing something that is not a string as a
//! path is a bug and panics.

use crate::builtins::{arg_str, lib, tpush};
use crate::value::Value;
use crate::vm::{rt_err, RtResult, Vm};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

type R = RtResult<Value>;

pub fn install(vm: &mut Vm) {
    lib(
        vm,
        "fs",
        &[
            ("read", 1, read),
            ("write", 2, write),
            ("append", 2, append),
            ("lines", 1, lines),
            ("exists", 1, |vm, a| Ok(Value::bool(Path::new(arg_str(vm, a, 0, "fs.exists")?.as_str()).exists()))),
            ("is_file", 1, |vm, a| Ok(Value::bool(Path::new(arg_str(vm, a, 0, "fs.is_file")?.as_str()).is_file()))),
            ("is_dir", 1, |vm, a| Ok(Value::bool(Path::new(arg_str(vm, a, 0, "fs.is_dir")?.as_str()).is_dir()))),
            ("size", 1, size),
            ("list", 1, list),
            ("mkdir", 1, mkdir),
            ("remove", 1, remove),
            ("remove_all", 1, remove_all),
            ("rename", 2, rename),
            ("copy", 2, copy),
            ("cwd", 0, cwd),
            ("join", -1, join),
            ("basename", 1, |vm, a| path_part(vm, a, "fs.basename", |p| p.file_name())),
            ("stem", 1, |vm, a| path_part(vm, a, "fs.stem", |p| p.file_stem())),
            ("ext", 1, |vm, a| path_part(vm, a, "fs.ext", |p| p.extension())),
            ("dirname", 1, dirname),
        ],
    );
}

// ---- errors ---------------------------------------------------------------------------------

fn failure(vm: &mut Vm, path: &str, e: &io::Error) -> Value {
    let (kind, why) = match e.kind() {
        io::ErrorKind::NotFound => ("not_found", "no such file or directory".to_string()),
        io::ErrorKind::PermissionDenied => ("permission", "permission denied".to_string()),
        io::ErrorKind::AlreadyExists => ("exists", "already exists".to_string()),
        io::ErrorKind::InvalidData => ("utf8", "file is not valid UTF-8 text".to_string()),
        _ => ("io", e.to_string()),
    };
    let err = vm.make_error(&format!("{}: {}", path, why), kind);
    vm.make_err(err)
}

/// `Ok(nil)` or the failure, for operations without a result.
fn unit(vm: &mut Vm, path: &str, r: io::Result<()>) -> R {
    Ok(match r {
        Ok(()) => vm.make_ok(Value::NIL),
        Err(e) => failure(vm, path, &e),
    })
}

// ---- files ----------------------------------------------------------------------------------

/// `fs.read(path)` -> `Ok(text)`.
fn read(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.read")?.as_str().to_string();
    Ok(match std::fs::read_to_string(&path) {
        Ok(text) => {
            let s = vm.new_string(text);
            vm.make_ok(s)
        }
        Err(e) => failure(vm, &path, &e),
    })
}

/// `fs.write(path, text)` creates or truncates the file.
fn write(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.write")?.as_str().to_string();
    let text = arg_str(vm, a, 1, "fs.write")?;
    let r = std::fs::write(&path, text.as_str());
    unit(vm, &path, r)
}

/// `fs.append(path, text)` creates the file if needed.
fn append(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.append")?.as_str().to_string();
    let text = arg_str(vm, a, 1, "fs.append")?;
    let r = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(text.as_str().as_bytes()));
    unit(vm, &path, r)
}

/// `fs.lines(path)` -> `Ok([line, …])`, without the line terminators.
fn lines(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.lines")?.as_str().to_string();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return Ok(failure(vm, &path, &e)),
    };
    let mut t = vm.new_table();
    for line in text.lines() {
        let s = vm.new_string(line);
        tpush(vm, &mut t, s);
    }
    Ok(vm.make_ok(Value::from(t)))
}

/// `fs.size(path)` -> `Ok(bytes)`.
fn size(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.size")?.as_str().to_string();
    Ok(match std::fs::metadata(&path) {
        Ok(m) => vm.make_ok(Value::num(m.len() as f64)),
        Err(e) => failure(vm, &path, &e),
    })
}

fn copy(vm: &mut Vm, a: &[Value]) -> R {
    let from = arg_str(vm, a, 0, "fs.copy")?.as_str().to_string();
    let to = arg_str(vm, a, 1, "fs.copy")?.as_str().to_string();
    let r = std::fs::copy(&from, &to).map(|_| ());
    unit(vm, &from, r)
}

fn rename(vm: &mut Vm, a: &[Value]) -> R {
    let from = arg_str(vm, a, 0, "fs.rename")?.as_str().to_string();
    let to = arg_str(vm, a, 1, "fs.rename")?.as_str().to_string();
    let r = std::fs::rename(&from, &to);
    unit(vm, &from, r)
}

// ---- directories ----------------------------------------------------------------------------

/// `fs.list(dir)` -> `Ok([name, …])`, sorted, names only.
fn list(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.list")?.as_str().to_string();
    let entries = match std::fs::read_dir(&path) {
        Ok(rd) => rd,
        Err(e) => return Ok(failure(vm, &path, &e)),
    };
    let mut names = Vec::new();
    for entry in entries {
        match entry {
            Ok(en) => names.push(en.file_name().to_string_lossy().into_owned()),
            Err(e) => return Ok(failure(vm, &path, &e)),
        }
    }
    names.sort();
    let mut t = vm.new_table();
    for n in names {
        let s = vm.new_string(n);
        tpush(vm, &mut t, s);
    }
    Ok(vm.make_ok(Value::from(t)))
}

/// `fs.mkdir(path)` creates missing parents too; an existing directory is fine.
fn mkdir(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.mkdir")?.as_str().to_string();
    let r = std::fs::create_dir_all(&path);
    unit(vm, &path, r)
}

/// `fs.remove(path)` removes a file or an empty directory.
fn remove(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.remove")?.as_str().to_string();
    let r = if Path::new(&path).is_dir() { std::fs::remove_dir(&path) } else { std::fs::remove_file(&path) };
    unit(vm, &path, r)
}

/// `fs.remove_all(path)` removes a file or a whole directory tree.
fn remove_all(vm: &mut Vm, a: &[Value]) -> R {
    let path = arg_str(vm, a, 0, "fs.remove_all")?.as_str().to_string();
    let r = if Path::new(&path).is_dir() { std::fs::remove_dir_all(&path) } else { std::fs::remove_file(&path) };
    unit(vm, &path, r)
}

fn cwd(vm: &mut Vm, _: &[Value]) -> R {
    Ok(match std::env::current_dir() {
        Ok(p) => {
            let s = vm.new_string(p.to_string_lossy().into_owned());
            vm.make_ok(s)
        }
        Err(e) => failure(vm, ".", &e),
    })
}

// ---- paths (pure string work, never touches the disk) -------------------------------------

/// `fs.join(a, b, …)`: an absolute later part replaces what came before, like `Path::join`.
fn join(vm: &mut Vm, a: &[Value]) -> R {
    if a.is_empty() {
        return Err(rt_err("fs.join: missing argument 1"));
    }
    let mut p = PathBuf::new();
    for i in 0..a.len() {
        p.push(arg_str(vm, a, i, "fs.join")?.as_str());
    }
    Ok(vm.new_string(p.to_string_lossy().into_owned()))
}

/// `nil` when the part does not exist (`fs.ext("Makefile")`).
fn path_part(vm: &mut Vm, a: &[Value], f: &str, part: impl Fn(&Path) -> Option<&std::ffi::OsStr>) -> R {
    let s = arg_str(vm, a, 0, f)?;
    Ok(match part(Path::new(s.as_str())) {
        Some(x) => vm.new_string(x.to_string_lossy().into_owned()),
        None => Value::NIL,
    })
}

/// `fs.dirname("a/b.txt")` is `"a"`, `fs.dirname("b.txt")` is `""`.
fn dirname(vm: &mut Vm, a: &[Value]) -> R {
    let s = arg_str(vm, a, 0, "fs.dirname")?;
    let d = Path::new(s.as_str()).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(vm.new_string(d))
}
