//! Every program in `examples/` (except the benchmarks) must run to completion.

use kite::Vm;
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

#[test]
fn examples_run() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let handle = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let mut ran = 0;
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_none_or(|e| e != "kite") {
                    continue;
                }
                let src = std::fs::read_to_string(&path).unwrap();
                let buf = Rc::new(RefCell::new(Vec::new()));
                let mut vm = Vm::new();
                vm.out = Box::new(Sink(buf.clone()));
                if let Err(e) = vm.eval(&path.to_string_lossy(), &src) {
                    panic!("{} failed:\n{}", path.display(), e);
                }
                vm.shutdown();
                assert!(!buf.borrow().is_empty(), "{} printed nothing", path.display());
                ran += 1;
            }
            assert!(ran >= 5, "expected several examples, ran {}", ran);
        })
        .unwrap();
    handle.join().unwrap();
}
