//! Runs every `tests/cases/*.kite` program and checks its output.
//!
//! Expectations live in comments inside the program:
//!   `// expect: text`   next line of stdout
//!   `// error: text`    the program must fail with a message containing `text`
//!   `// config: opt-only`  skip the run without the optimiser (e.g. relies on tail calls)
//! Each program runs three times: normally, without the peephole optimiser,
//! and with a garbage collection at every safe point.

use kite::Vm;
use std::cell::RefCell;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Clone, Default)]
struct SharedBuf(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Outcome {
    stdout: String,
    error: Option<String>,
}

fn run(src: &str, no_opt: bool, stress: bool) -> Outcome {
    let buf = SharedBuf::default();
    let mut vm = Vm::new();
    vm.out = Box::new(buf.clone());
    vm.opts.no_optimize = no_opt;
    vm.heap.stress = stress;
    let r = vm.eval("test.kite", src);
    vm.shutdown();
    let stdout = String::from_utf8(buf.0.borrow().clone()).unwrap();
    Outcome { stdout, error: r.err().map(|e| e.to_string()) }
}

fn expectations(src: &str) -> (Vec<String>, Option<String>) {
    let mut lines = Vec::new();
    let mut err = None;
    for l in src.lines() {
        if let Some(i) = l.find("// expect:") {
            lines.push(l[i + "// expect:".len()..].strip_prefix(' ').unwrap_or(&l[i + "// expect:".len()..]).to_string());
        } else if let Some(i) = l.find("// error:") {
            err = Some(l[i + "// error:".len()..].trim().to_string());
        }
    }
    (lines, err)
}

fn check(path: &PathBuf, no_opt: bool, stress: bool) -> Result<(), String> {
    let src = std::fs::read_to_string(path).unwrap();
    let (expected, expected_err) = expectations(&src);
    let out = run(&src, no_opt, stress);
    let label = format!("{} [{}{}]", path.file_name().unwrap().to_string_lossy(), if no_opt { "no-opt " } else { "" }, if stress { "gc-stress" } else { "" });
    match (&expected_err, &out.error) {
        (Some(want), Some(got)) => {
            if !got.contains(want.as_str()) {
                return Err(format!("{}: expected error containing {:?}, got:\n{}", label, want, got));
            }
        }
        (Some(want), None) => return Err(format!("{}: expected error {:?} but the program succeeded", label, want)),
        (None, Some(got)) => return Err(format!("{}: unexpected error:\n{}\n--- stdout so far:\n{}", label, got, out.stdout)),
        (None, None) => {}
    }
    let actual: Vec<&str> = out.stdout.lines().collect();
    let want: Vec<&str> = expected.iter().map(|s| s.as_str()).collect();
    if actual != want {
        let mut msg = format!("{}: output mismatch\n", label);
        for i in 0..actual.len().max(want.len()) {
            let a = actual.get(i).copied().unwrap_or("<missing>");
            let w = want.get(i).copied().unwrap_or("<missing>");
            if a != w {
                msg.push_str(&format!("  line {:>3}: expected {:?}, got {:?}\n", i + 1, w, a));
            }
        }
        return Err(msg);
    }
    Ok(())
}

#[test]
fn all_cases() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cases");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "kite"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no test cases found");
    let handle = std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            let mut failures = Vec::new();
            for f in &files {
                let opt_only = std::fs::read_to_string(f).unwrap().contains("// config: opt-only");
                for (no_opt, stress) in [(false, false), (true, false), (false, true)] {
                    if no_opt && opt_only {
                        continue;
                    }
                    if let Err(m) = check(f, no_opt, stress) {
                        failures.push(m);
                    }
                }
            }
            failures
        })
        .unwrap();
    let failures = handle.join().unwrap();
    if !failures.is_empty() {
        panic!("{} failure(s):\n\n{}", failures.len(), failures.join("\n"));
    }
}
