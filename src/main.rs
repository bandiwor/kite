use kite::{repl, Options, Vm};
use std::io::Read;
use std::process::ExitCode;

const USAGE: &str = "\
Kite — a small, fast, dynamically typed language

usage: kite [options] [script.kite]
       kite                      start the interactive REPL
       kite script.kite          run a file
       kite -e 'print(1 + 2)'    run a snippet
       kite -                    read the program from stdin

options:
  -e, --eval <code>    evaluate code instead of reading a file
      --dump-ast       print the AST (stderr)
      --dump-bytecode  print the bytecode (stderr)
      --no-opt         disable the peephole optimiser
      --gc-stress      collect garbage at every opportunity (debugging)
      --time           print compile/run timings (stderr)
  -h, --help           show this help
  -v, --version        show the version";

fn main() -> ExitCode {
    // Deeply nested overloads / callbacks recurse on the Rust stack, so run on a big one.
    let child = std::thread::Builder::new().stack_size(256 << 20).spawn(real_main).expect("spawn main thread");
    match child.join() {
        Ok(code) => ExitCode::from(code),
        Err(_) => ExitCode::from(101),
    }
}

fn real_main() -> u8 {
    let mut opts = Options::default();
    let mut gc_stress = false;
    let mut timing = false;
    let mut file: Option<String> = None;
    let mut code: Option<String> = None;
    let mut script_args: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{}", USAGE);
                return 0;
            }
            "-v" | "--version" => {
                println!("kite {}", env!("CARGO_PKG_VERSION"));
                return 0;
            }
            "-e" | "--eval" => match args.next() {
                Some(c) => code = Some(c),
                None => {
                    eprintln!("error: {} needs an argument", a);
                    return 2;
                }
            },
            "--dump-ast" => opts.dump_ast = true,
            "--dump-bytecode" => opts.dump_bytecode = true,
            "--no-opt" => opts.no_optimize = true,
            "--gc-stress" => gc_stress = true,
            "--time" => timing = true,
            "-" => file = Some("-".into()),
            s if s.starts_with('-') => {
                eprintln!("error: unknown option `{}`\n\n{}", s, USAGE);
                return 2;
            }
            _ => {
                if file.is_some() {
                    // everything after the script path belongs to the script
                    script_args.push(a);
                    script_args.extend(args.by_ref());
                    break;
                }
                file = Some(a);
            }
        }
    }

    let mut vm = Vm::new();
    // `args`: command-line arguments after the script path
    let mut args_table = vm.new_table();
    for a in &script_args {
        let v = vm.new_string(a.as_str());
        args_table.push(v);
    }
    vm.define_global("args", kite::Value::from(args_table));
    opts.timing = timing;
    vm.opts = opts;
    vm.heap.stress = gc_stress || std::env::var_os("KITE_GC_STRESS").is_some();

    if code.is_none() && file.is_none() {
        repl::run(&mut vm);
        return 0;
    }

    let (name, src) = if let Some(c) = code {
        ("<eval>".to_string(), c)
    } else {
        let path = file.unwrap();
        let mut src = String::new();
        let r = if path == "-" {
            std::io::stdin().read_to_string(&mut src).map(|_| ())
        } else {
            std::fs::read_to_string(&path).map(|s| src = s)
        };
        if let Err(e) = r {
            eprintln!("error: cannot read `{}`: {}", path, e);
            return 2;
        }
        (path, src)
    };

    let t0 = std::time::Instant::now();
    let result = vm.eval(&name, &src);
    let elapsed = t0.elapsed();
    let mut status = 0;
    match result {
        Ok(_) => {}
        Err(e) => {
            vm.flush();
            eprintln!("{}", e);
            status = 1;
        }
    }
    vm.shutdown();
    #[cfg(feature = "op-profile")]
    kite::profile::report();
    if timing {
        eprintln!(
            "[time] {:.3} ms total, {} collections, {} bytes live",
            elapsed.as_secs_f64() * 1000.0,
            vm.heap.collections,
            vm.heap.bytes
        );
    }
    status
}
