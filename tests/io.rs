//! The `io` library's readers, driven through the real binary with a piped stdin.

use std::io::Write;
use std::process::{Command, Stdio};

fn run(src: &str, stdin: &str) -> (String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_kite"))
        .args(["-e", src])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn read_line_returns_options_until_end_of_input() {
    let src = r#"
        import "std/io";
        var n = 0;
        while true {
            match io.read_line() {
                Some(l) => { n += 1; print(n, l, l:len()); },
                None => break,
            }
        }
        print("done", n);
    "#;
    let (out, _) = run(src, "one\ntwo\r\n\nlast");
    assert_eq!(out, "1 one 3\n2 two 3\n3  0\n4 last 4\ndone 4\n");
    let (out, _) = run(src, "");
    assert_eq!(out, "done 0\n");
}

#[test]
fn whole_input_readers() {
    let (out, _) = run(r#"import "std/io"; print(io.read_all());"#, "a\nb\n");
    assert_eq!(out, "Ok(\"a\\nb\\n\")\n");
    let (out, _) = run(r#"import "std/io"; print(io.lines());"#, "a\n\nb");
    assert_eq!(out, "Ok([\"a\", \"\", \"b\"])\n");
    // read_line first, then the rest
    let (out, _) = run(r#"import "std/io"; print(io.read_line():unwrap(), io.lines():unwrap());"#, "first\nsecond\n");
    assert_eq!(out, "first [\"second\"]\n");
}

#[test]
fn invalid_utf8_is_an_error_for_whole_input_and_lossy_for_lines() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_kite"))
        .args(["-e", r#"import "std/io"; const r = io.read_all(); print(r:is_err(), r:unwrap_err().kind);"#])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&[0x66, 0xff, 0x6f]).unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "true utf8\n");

    let mut child = Command::new(env!("CARGO_BIN_EXE_kite"))
        .args(["-e", r#"import "std/io"; print(io.read_line():unwrap():len());"#])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&[0x66, 0xff, 0x6f, b'\n']).unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3\n");
}

#[test]
fn prompt_prints_before_reading_and_stderr_is_separate() {
    let (out, err) = run(
        r#"import "std/io"; const name = io.prompt("name? "); io.print("hi", name); io.eprint("warn", 1);"#,
        "Kite\n",
    );
    assert_eq!(out, "name? hi Some(\"Kite\")\n");
    assert_eq!(err, "warn 1\n");
}
