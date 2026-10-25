//! Cases of the `fs` library that Kite code alone cannot produce.

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

fn run(src: &str) -> String {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut vm = Vm::new();
    vm.out = Box::new(Sink(buf.clone()));
    if let Err(e) = vm.eval("fs_test.kite", src) {
        panic!("{}", e);
    }
    vm.flush();
    let out = String::from_utf8(buf.borrow().clone()).unwrap();
    out
}

#[test]
fn invalid_utf8_is_an_error_with_a_kind() {
    let dir = std::env::temp_dir().join(format!("kite_fs_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("bad.bin");
    std::fs::write(&file, [0x66, 0xff, 0xfe, 0x6f]).unwrap();
    let path = file.to_string_lossy().replace('\\', "/");
    let out =
        run(&format!("import \"std/fs\"; const r = fs.read({path:?}); print(r:is_err(), r:unwrap_err().kind); print(fs.size({path:?}));"));
    assert_eq!(out, "true utf8\nOk(4)\n");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn absolute_paths_and_unicode_names() {
    let dir = std::env::temp_dir().join(format!("kite_fs_test_u_{}", std::process::id()));
    let d = dir.to_string_lossy().replace('\\', "/");
    let out = run(&format!(
        "import \"std/fs\"; const d = {d:?}; fs.remove_all(d); fs.mkdir(fs.join(d, \"ünï\")); \
         fs.write(fs.join(d, \"ünï\", \"файл.txt\"), \"привет\"); \
         print(fs.list(fs.join(d, \"ünï\"))); print(fs.read(fs.join(d, \"ünï\", \"файл.txt\"))); fs.remove_all(d); print(fs.exists(d));"
    ));
    assert_eq!(out, "Ok([\"файл.txt\"])\nOk(\"привет\")\nfalse\n");
}
