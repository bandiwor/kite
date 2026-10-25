//! The virtual machine.
//!
//! * One value stack (fixed allocation, so raw pointers into it stay valid and
//!   open upvalues can point straight at slots); each call frame owns
//!   `[locals…][operand temporaries…]` starting at `frame.base`.
//! * `run` is a single `match` over 8-byte `Op`s with the hot registers
//!   (`ip`, `sp`, `base`, `consts`) cached in locals.
//! * Calls to closures never recurse on the Rust stack; only native
//!   callbacks and operator overloads re-enter `run` (bounded by `MAX_NEST`).
//! * Garbage collection only happens at *safe points* (after allocating
//!   instructions), where every live value is reachable from the stack,
//!   frames, globals or open upvalues.

use crate::analyzer::GlobalInfo;
use crate::compiler::{self, CompileOptions};
use crate::hash::FxHashMap;
use crate::heap::Heap;
use crate::intern::{Interner, Sym};
use crate::lexer;
use crate::object::*;
use crate::op::Op;
use crate::parser;
use crate::source::{Diag, ErrorKind, KiteError, SourceFile};
use crate::table::{KeyError, Table};
use crate::text::fmt_num;
use crate::value::Value;
use crate::{analyzer, disasm};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::io::Write;
use std::rc::Rc;

pub const STACK_SIZE: usize = 1 << 22; // values (32 MiB of address space, committed lazily)
pub const MAX_FRAMES: usize = 400_000;
const MAX_NEST: u32 = 120;

// ---- errors ----------------------------------------------------------------------

pub struct TraceEntry {
    pub func: Rc<str>,
    pub source: Rc<SourceFile>,
    pub offset: u32,
}

pub struct RtError {
    pub msg: String,
    pub trace: Vec<TraceEntry>,
}

pub type RtResult<T> = Result<T, Box<RtError>>;

pub fn rt_err(msg: impl Into<String>) -> Box<RtError> {
    Box::new(RtError { msg: msg.into(), trace: Vec::new() })
}

#[macro_export]
macro_rules! rt_bail {
    ($($a:tt)*) => { return Err($crate::vm::rt_err(format!($($a)*))) };
}

// ---- configuration -----------------------------------------------------------------

#[derive(Clone, Default)]
pub struct Options {
    pub no_optimize: bool,
    pub dump_ast: bool,
    pub dump_bytecode: bool,
    /// REPL mode: top-level names may be redeclared.
    pub repl: bool,
    /// Print a per-phase timing line to stderr after compiling.
    pub timing: bool,
}

enum ModuleState {
    /// Currently executing (importing it again is a cycle).
    Loading,
    Done(Value),
}

pub struct Frame {
    pub closure: Gc<Closure>,
    pub ip: *const Op,
    /// Register 0 of this frame (its first parameter).
    pub base: *mut Value,
    /// The register of the *calling* frame that receives this call's result.
    pub ret: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl Arith {
    fn symbol(self) -> &'static str {
        match self {
            Arith::Add => "+",
            Arith::Sub => "-",
            Arith::Mul => "*",
            Arith::Div => "/",
            Arith::Rem => "%",
            Arith::Pow => "**",
            Arith::BitAnd => "&",
            Arith::BitOr => "|",
            Arith::BitXor => "^",
            Arith::Shl => "<<",
            Arith::Shr => ">>",
        }
    }
    fn hook(self) -> Option<Hook> {
        Some(match self {
            Arith::Add => Hook::Add,
            Arith::Sub => Hook::Sub,
            Arith::Mul => Hook::Mul,
            Arith::Div => Hook::Div,
            Arith::Rem => Hook::Rem,
            Arith::Pow => Hook::Pow,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

// ---- the machine ---------------------------------------------------------------------

pub struct Vm {
    pub interner: Interner,
    pub heap: Heap,
    pub globals: Vec<Value>,
    pub global_info: GlobalInfo,
    pub opts: Options,
    pub out: Box<dyn Write>,
    /// The `string` library table, used for `"text":method()` calls.
    pub string_lib: Value,
    /// The `array` library table, used for `arr:method()` calls.
    pub array_lib: Value,
    /// The `table` library table: `t:keys()` and friends (after the table's own fields).
    pub table_lib: Value,
    /// The prelude's `Option`, `Result` and `Error` types (`NIL` while the prelude loads).
    pub option_ty: Value,
    pub result_ty: Value,
    pub error_ty: Value,
    pub rng_state: u64,
    /// Loaded modules by canonical path.
    modules: FxHashMap<String, ModuleState>,

    stack: *mut Value,
    stack_end: *mut Value,
    sp: *mut Value,
    frames: Vec<Frame>,
    open_upvals: Vec<Gc<Upvalue>>,
    pending_final: Vec<Value>,
    finalizing: bool,
    nest: u32,
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe { dealloc(self.stack as *mut u8, Layout::array::<Value>(STACK_SIZE).unwrap()) };
    }
}

impl Default for Vm {
    fn default() -> Self {
        Vm::new()
    }
}

impl Vm {
    pub fn new() -> Vm {
        let layout = Layout::array::<Value>(STACK_SIZE).unwrap();
        // Zeroed pages are only committed when touched.  A zero slot reads as the
        // number 0.0, which is a valid value, and the collector only scans below `sp`.
        let stack = unsafe { alloc_zeroed(layout) as *mut Value };
        assert!(!stack.is_null(), "cannot allocate the VM stack");
        let mut vm = Vm {
            interner: Interner::new(),
            heap: Heap::new(),
            globals: Vec::new(),
            global_info: GlobalInfo::default(),
            opts: Options::default(),
            out: Box::new(std::io::BufWriter::new(std::io::stdout())),
            string_lib: Value::NIL,
            array_lib: Value::NIL,
            table_lib: Value::NIL,
            option_ty: Value::NIL,
            result_ty: Value::NIL,
            error_ty: Value::NIL,
            rng_state: 0x9E37_79B9_7F4A_7C15,
            modules: FxHashMap::default(),
            stack,
            stack_end: unsafe { stack.add(STACK_SIZE) },
            sp: stack,
            frames: Vec::with_capacity(MAX_FRAMES),
            open_upvals: Vec::new(),
            pending_final: Vec::new(),
            finalizing: false,
            nest: 0,
        };
        crate::builtins::install(&mut vm);
        crate::prelude::install(&mut vm);
        vm
    }

    // ---- embedding helpers ----------------------------------------------------------

    /// Make `v` the module that `import "key"` returns (used for the `std/…` modules).
    pub fn register_module(&mut self, key: &str, v: Value) {
        self.modules.insert(key.to_string(), ModuleState::Done(v));
    }

    pub fn define_global(&mut self, name: &str, v: Value) {
        let sym = self.interner.intern(name);
        if self.globals.len() <= sym.idx() {
            self.globals.resize(self.interner.len().max(sym.idx() + 1), Value::UNDEF);
        }
        self.globals[sym.idx()] = v;
    }

    pub fn get_global(&self, name: &str) -> Option<Value> {
        let sym = self.interner.lookup(name)?;
        self.globals.get(sym.idx()).copied().filter(|&v| v != Value::UNDEF)
    }

    pub fn define_native(&mut self, name: &'static str, arity: i8, func: NativeFn) {
        let v = self.new_native(name, arity, func);
        self.define_global(name, v);
    }

    pub fn new_native(&mut self, name: &'static str, arity: i8, func: NativeFn) -> Value {
        Value::from(self.heap.alloc(Native { hdr: Header::new(Kind::Native), name, arity, func }))
    }

    pub fn new_string(&mut self, s: impl Into<Box<str>>) -> Value {
        Value::from(self.heap.new_str(s))
    }

    pub fn new_table(&mut self) -> Gc<Table> {
        self.heap.new_table(0)
    }

    pub fn new_array(&mut self, items: Vec<Value>) -> Gc<Array> {
        self.heap.new_array(items)
    }

    /// Push onto an array, keeping the GC's byte accounting honest.
    #[inline]
    pub fn array_push(&mut self, arr: &mut Gc<Array>, v: Value) {
        let before = arr.items.capacity();
        arr.items.push(v);
        let after = arr.items.capacity();
        if after != before {
            self.heap.bytes += (after - before) * std::mem::size_of::<Value>();
        }
    }

    /// Temporarily keep `v` alive across calls back into the VM (natives
    /// that hold fresh objects while invoking callbacks must root them).
    pub fn root(&mut self, v: Value) {
        unsafe {
            if self.sp >= self.stack_end.sub(8) {
                return;
            }
            *self.sp = v;
            self.sp = self.sp.add(1);
        }
    }

    pub fn unroot(&mut self, n: usize) {
        unsafe { self.sp = self.sp.sub(n) };
    }

    pub fn flush(&mut self) {
        let _ = self.out.flush();
    }

    // ---- pipeline -------------------------------------------------------------------

    /// Compile and run `src`; returns the value of the final expression.
    pub fn eval(&mut self, name: &str, src: &str) -> Result<Value, KiteError> {
        let (proto, source) = self.compile(name, src)?;
        self.run_main(proto).map_err(|e| self.to_kite_error(*e, &source))
    }

    /// Run the front end and the compiler without executing anything.
    pub fn compile(&mut self, name: &str, src: &str) -> Result<(Gc<Proto>, Rc<SourceFile>), KiteError> {
        self.compile_chunk(name, src, false)
    }

    /// Like `compile`; `module` chunks keep their top-level names private and evaluate
    /// to the table of their exports.
    pub fn compile_chunk(
        &mut self,
        name: &str,
        src: &str,
        module: bool,
    ) -> Result<(Gc<Proto>, Rc<SourceFile>), KiteError> {
        let source = SourceFile::new(name, src);
        let err = |kind: ErrorKind, diags: Vec<Diag>, source: &Rc<SourceFile>| KiteError {
            kind,
            diags,
            source: source.clone(),
            trace: Vec::new(),
        };
        let t0 = std::time::Instant::now();
        let toks = lexer::lex(&source.text);
        let t_lex = t0.elapsed();
        let ast =
            parser::parse(&source.text, &toks, &mut self.interner).map_err(|d| err(ErrorKind::Syntax, d, &source))?;
        let t_parse = t0.elapsed();
        if self.opts.dump_ast {
            eprintln!("{}", ast.dump(&self.interner, &source.text));
        }
        let an = analyzer::analyze(&ast, &self.interner, &mut self.global_info, self.opts.repl, module)
            .map_err(|d| err(ErrorKind::Compile, d, &source))?;
        let t_an = t0.elapsed();
        if self.globals.len() < self.interner.len() {
            self.globals.resize(self.interner.len(), Value::UNDEF);
        }
        let copts = CompileOptions { optimize: !self.opts.no_optimize };
        let proto = compiler::compile(&ast, &an, &self.interner, &mut self.heap, &source, &copts)
            .map_err(|d| err(ErrorKind::Compile, d, &source))?;
        if self.opts.timing {
            let t_all = t0.elapsed();
            let us = |d: std::time::Duration| d.as_secs_f64() * 1e6;
            eprintln!(
                "[compile] {} bytes, {} tokens, {} ast nodes | lex {:.0}us parse {:.0}us analyze {:.0}us compile+peephole {:.0}us | total {:.0}us ({:.1} MB/s)",
                source.text.len(),
                toks.len(),
                ast.nodes.len(),
                us(t_lex),
                us(t_parse - t_lex),
                us(t_an - t_parse),
                us(t_all - t_an),
                us(t_all),
                source.text.len() as f64 / t_all.as_secs_f64() / 1e6
            );
        }
        if self.opts.dump_bytecode {
            eprintln!("{}", disasm::disassemble(proto, &self.interner));
        }
        Ok((proto, source))
    }

    fn to_kite_error(&self, mut e: RtError, fallback: &Rc<SourceFile>) -> KiteError {
        // frames of the prelude are an implementation detail: report the caller's line instead
        if e.trace.iter().any(|t| t.source.name != "<prelude>") {
            e.trace.retain(|t| t.source.name != "<prelude>");
        }
        let (source, offset) = match e.trace.first() {
            Some(t) => (t.source.clone(), t.offset),
            None => (fallback.clone(), 0),
        };
        let mut trace = Vec::new();
        trace.push("stack trace:".to_string());
        let total = e.trace.len();
        for (i, t) in e.trace.iter().enumerate() {
            if i >= 12 && total > 14 && i < total - 2 {
                if i == 12 {
                    trace.push(format!("  ... {} more frames ...", total - 14));
                }
                continue;
            }
            let line = t.source.line(t.offset);
            trace.push(format!("  at {} ({}:{})", t.func, t.source.name, line));
        }
        if e.trace.is_empty() {
            trace.clear();
        }
        KiteError { kind: ErrorKind::Runtime, diags: vec![Diag::new(e.msg.clone(), offset, 1)], source, trace }
    }

    fn run_main(&mut self, proto: Gc<Proto>) -> RtResult<Value> {
        let closure = self.heap.alloc(Closure { hdr: Header::new(Kind::Closure), proto, upvals: Box::new([]) });
        self.reset();
        unsafe {
            let slot = self.sp;
            *slot = Value::from(closure);
            let stop = self.frames.len();
            self.push_frame(closure, slot.add(1), 0, 0)?;
            let r = self.run(stop);
            if r.is_err() {
                self.unwind(0, self.stack);
            }
            r
        }
    }

    /// Reset the machine between top-level runs.
    fn reset(&mut self) {
        self.sp = self.stack;
        self.frames.clear();
        self.open_upvals.clear();
        self.nest = 0;
        self.finalizing = false;
    }

    /// Drop frames above `nframes`, close their upvalues and restore `sp`.
    fn unwind(&mut self, nframes: usize, sp: *mut Value) {
        self.close_upvalues(sp);
        self.frames.truncate(nframes);
        self.sp = sp;
    }

    // ---- upvalues -------------------------------------------------------------------

    fn capture_upvalue(&mut self, slot: *mut Value) -> Gc<Upvalue> {
        let mut i = self.open_upvals.len();
        while i > 0 {
            let uv = self.open_upvals[i - 1];
            if uv.location == slot {
                return uv;
            }
            if uv.location < slot {
                break;
            }
            i -= 1;
        }
        let uv = self.heap.alloc(Upvalue { hdr: Header::new(Kind::Upvalue), location: slot, closed: Value::NIL });
        self.open_upvals.insert(i, uv);
        uv
    }

    #[inline]
    fn close_upvalues(&mut self, from: *mut Value) {
        while let Some(&last) = self.open_upvals.last() {
            if last.location < from {
                break;
            }
            let mut uv = last;
            uv.closed = unsafe { *uv.location };
            uv.location = &mut uv.closed;
            self.open_upvals.pop();
        }
    }

    // ---- calls ----------------------------------------------------------------------

    /// Push a frame for the closure `cl` whose `argc` arguments are at `base[0..argc)`.
    /// `ret` is the register of the *caller's* frame that receives the result.
    /// Sets `self.sp` to the top of the new frame.
    #[inline(always)]
    unsafe fn push_frame(&mut self, cl: Gc<Closure>, base: *mut Value, argc: usize, ret: u32) -> RtResult<()> {
        let proto = cl.proto;
        if proto.arity as usize != argc {
            return Err(rt_err(format!(
                "function '{}' expects {} argument{} but got {}",
                proto.name,
                proto.arity,
                if proto.arity == 1 { "" } else { "s" },
                argc
            )));
        }
        let top = base.add(proto.nregs as usize);
        // a few slots of slack: operator hooks stage their operands just above the frame
        if self.frames.len() >= MAX_FRAMES || top.add(8) > self.stack_end {
            return Err(rt_err("stack overflow"));
        }
        // registers behind the arguments start as nil: the collector scans the whole frame
        let mut p = base.add(argc);
        while p < top {
            *p = Value::NIL;
            p = p.add(1);
        }
        self.sp = top;
        self.frames.push(Frame { closure: cl, ip: proto.code.as_ptr(), base, ret });
        Ok(())
    }

    /// Call the value at `slot`, with its `argc` arguments in the registers right behind it.
    /// Returns `true` if a new bytecode frame was pushed (the caller must run it), `false`
    /// if the call completed and its result was stored at `slot`.  A pushed frame returns
    /// into register `ret` of the frame below.
    ///
    /// `self.sp` must be above the arguments (natives that call back push there).
    unsafe fn call_at(&mut self, slot: *mut Value, argc: usize, ret: u32) -> RtResult<bool> {
        let callee = *slot;
        match callee.obj_kind() {
            Some(Kind::Closure) => {
                self.push_frame(callee.as_gc_unchecked::<Closure>(), slot.add(1), argc, ret)?;
                Ok(true)
            }
            Some(Kind::Native) => {
                let nat = callee.as_gc_unchecked::<Native>();
                if nat.arity >= 0 && nat.arity as usize != argc {
                    return Err(rt_err(format!(
                        "function '{}' expects {} argument{} but got {}",
                        nat.name,
                        nat.arity,
                        if nat.arity == 1 { "" } else { "s" },
                        argc
                    )));
                }
                let args = std::slice::from_raw_parts(slot.add(1), argc);
                *slot = (nat.func)(self, args)?;
                Ok(false)
            }
            Some(Kind::StructType) => {
                let ty = callee.as_gc_unchecked::<StructType>();
                let nfields = ty.fields.len();
                if ty.is_union() {
                    return Err(rt_err(format!(
                        "union '{}' cannot be called; construct a variant, e.g. {}.{}(…)",
                        ty.name.as_str(),
                        ty.name.as_str(),
                        ty.variants[0].name.as_str()
                    )));
                }
                let inst = self.heap.alloc(Instance {
                    hdr: Header::new(Kind::Instance),
                    ty,
                    fields: vec![Value::NIL; nfields].into_boxed_slice(),
                    tag: 0,
                    finalized: false,
                });
                let ctor = ty.hook(Hook::New);
                if !ctor.is_nil() {
                    // [Type, a1..an] -> [new, inst, a1..an]
                    if slot.add(argc + 10) >= self.stack_end {
                        return Err(rt_err("stack overflow"));
                    }
                    std::ptr::copy(slot.add(1), slot.add(2), argc);
                    *slot.add(1) = Value::from(inst);
                    *slot = ctor;
                    if self.sp < slot.add(argc + 2) {
                        self.sp = slot.add(argc + 2);
                    }
                    return self.call_at(slot, argc + 1, ret);
                }
                if argc > nfields {
                    return Err(rt_err(format!(
                        "struct '{}' has {} field{} but {} value{} given",
                        ty.name.as_str(),
                        nfields,
                        if nfields == 1 { "" } else { "s" },
                        argc,
                        if argc == 1 { " was" } else { "s were" }
                    )));
                }
                let mut inst = inst;
                for i in 0..argc {
                    inst.fields[i] = *slot.add(1 + i);
                }
                *slot = Value::from(inst);
                Ok(false)
            }
            Some(Kind::Variant) => {
                let vc = callee.as_gc_unchecked::<VariantCtor>();
                let ty = vc.ty;
                let var = &ty.variants[vc.tag as usize];
                if var.fields.len() != argc {
                    return Err(rt_err(format!(
                        "variant '{}.{}' takes {} value{} but {} {} given",
                        ty.name.as_str(),
                        var.name.as_str(),
                        var.fields.len(),
                        if var.fields.len() == 1 { "" } else { "s" },
                        argc,
                        if argc == 1 { "was" } else { "were" }
                    )));
                }
                let fields: Box<[Value]> = std::slice::from_raw_parts(slot.add(1), argc).into();
                let inst = self.heap.alloc(Instance {
                    hdr: Header::new(Kind::Instance),
                    ty,
                    fields,
                    tag: vc.tag,
                    finalized: false,
                });
                *slot = Value::from(inst);
                Ok(false)
            }
            Some(Kind::Instance) => {
                let inst = callee.as_gc_unchecked::<Instance>();
                let hook = inst.ty.hook(Hook::Call);
                if hook.is_nil() {
                    return Err(rt_err(format!(
                        "struct '{}' instances are not callable (no `call` method)",
                        inst.ty.name.as_str()
                    )));
                }
                if slot.add(argc + 10) >= self.stack_end {
                    return Err(rt_err("stack overflow"));
                }
                // [obj, a1..an] -> [call, obj, a1..an]
                std::ptr::copy(slot.add(1), slot.add(2), argc);
                *slot.add(1) = callee;
                *slot = hook;
                if self.sp < slot.add(argc + 2) {
                    self.sp = slot.add(argc + 2);
                }
                self.call_at(slot, argc + 1, ret)
            }
            _ => Err(rt_err(format!("cannot call a value of type {}", callee.type_name()))),
        }
    }

    /// Call any callable from native code and wait for the result.
    pub fn call_value(&mut self, f: Value, args: &[Value]) -> RtResult<Value> {
        if self.nest >= MAX_NEST {
            return Err(rt_err("stack overflow (callbacks nested too deeply)"));
        }
        unsafe {
            if self.sp.add(args.len() + 16) >= self.stack_end {
                return Err(rt_err("stack overflow"));
            }
            let slot = self.sp;
            *slot = f;
            for (i, &a) in args.iter().enumerate() {
                *slot.add(1 + i) = a;
            }
            self.sp = slot.add(args.len() + 1);
            let stop = self.frames.len();
            self.nest += 1;
            let r = match self.call_at(slot, args.len(), 0) {
                // the frame returns with `sp` back at `slot`
                Ok(true) => self.run(stop),
                Ok(false) => {
                    let v = *slot;
                    self.sp = slot;
                    Ok(v)
                }
                Err(e) => Err(e),
            };
            self.nest -= 1;
            r
        }
    }

    /// Like `call_value`, but on error the VM state is restored so execution
    /// can continue (used by `catch_panic` and finalizers).
    pub fn call_protected(&mut self, f: Value, args: &[Value]) -> RtResult<Value> {
        let (nframes, sp, nest) = (self.frames.len(), self.sp, self.nest);
        let r = self.call_value(f, args);
        if r.is_err() {
            self.unwind(nframes, sp);
            self.nest = nest;
        }
        r
    }

    // ---- Option / Result / Error -----------------------------------------------------------

    /// For `?`: `Some(Some(payload))` for `Ok`/`Some`, `Some(None)` for `Err`/`None`
    /// (the value is returned as it is), `None` if `v` is neither an Option nor a Result.
    fn split_try(&self, v: Value) -> Option<Option<Value>> {
        let inst = v.as_gc::<Instance>()?;
        let ty = Value::from(inst.ty);
        // Option { None, Some } and Result { Ok, Err }: the payload variant is tag 1 for
        // Option and tag 0 for Result
        let success_tag = if ty == self.option_ty {
            1
        } else if ty == self.result_ty {
            0
        } else {
            return None;
        };
        Some((inst.tag == success_tag).then(|| inst.fields[0]))
    }

    fn payload_variant(&mut self, ty: Value, tag: u16, payload: Value) -> Value {
        let ty = ty.as_gc::<StructType>().expect("prelude types are installed");
        let inst = self.heap.alloc(Instance {
            hdr: Header::new(Kind::Instance),
            ty,
            fields: Box::new([payload]),
            tag,
            finalized: false,
        });
        Value::from(inst)
    }

    pub fn make_some(&mut self, v: Value) -> Value {
        self.payload_variant(self.option_ty, 1, v)
    }

    pub fn make_none(&self) -> Value {
        self.option_ty.as_gc::<StructType>().expect("prelude types are installed").variants[0].value
    }

    pub fn make_ok(&mut self, v: Value) -> Value {
        self.payload_variant(self.result_ty, 0, v)
    }

    pub fn make_err(&mut self, v: Value) -> Value {
        self.payload_variant(self.result_ty, 1, v)
    }

    /// `Error { message, kind }`.
    pub fn make_error(&mut self, message: &str, kind: &str) -> Value {
        let ty = self.error_ty.as_gc::<StructType>().expect("prelude types are installed");
        let fields = vec![self.new_string(message), self.new_string(kind)].into_boxed_slice();
        let inst = self.heap.alloc(Instance { hdr: Header::new(Kind::Instance), ty, fields, tag: 0, finalized: false });
        Value::from(inst)
    }

    // ---- unions & patterns ---------------------------------------------------------------

    /// Fresh runtime type from a compile-time template.  Unions get their variant
    /// values here: a shared instance for unit variants, a constructor otherwise.
    fn instantiate_type(&mut self, tmpl: Gc<StructType>) -> Gc<StructType> {
        let variants = tmpl
            .variants
            .iter()
            .map(|v| Variant { name: v.name, fields: v.fields.clone(), value: Value::NIL })
            .collect();
        let mut ty = self.heap.alloc(StructType {
            hdr: Header::new(Kind::StructType),
            name: tmpl.name,
            fields: tmpl.fields.clone(),
            variants,
            methods: Vec::new(),
            hooks: [Value::NIL; HOOK_COUNT],
        });
        for tag in 0..ty.variants.len() {
            let value = if ty.variants[tag].fields.is_empty() {
                let inst = self.heap.alloc(Instance {
                    hdr: Header::new(Kind::Instance),
                    ty,
                    fields: Box::new([]),
                    tag: tag as u16,
                    finalized: false,
                });
                Value::from(inst)
            } else {
                Value::from(self.heap.alloc(VariantCtor { hdr: Header::new(Kind::Variant), ty, tag: tag as u16 }))
            };
            ty.variants[tag].value = value;
        }
        ty
    }

    /// Does `v` belong to the variant (or struct) `name`?  `arity` 255 accepts any
    /// number of fields; otherwise it must match.  With `ty`, the type must be that one.
    fn variant_matches(&mut self, v: Value, name: Gc<Str>, arity: u8, ty: Option<Value>) -> RtResult<bool> {
        let qual = match ty {
            Some(t) => match t.as_gc::<StructType>() {
                Some(t) => Some(t),
                None => {
                    return Err(rt_err(format!(
                        "the qualifier of a pattern must be a struct or union, got {}",
                        self.type_of(t)
                    )))
                }
            },
            None => None,
        };
        let Some(inst) = v.as_gc::<Instance>() else { return Ok(false) };
        if qual.is_some_and(|t| t != inst.ty) {
            return Ok(false);
        }
        let (matches, n) = if inst.ty.is_union() {
            let var = &inst.ty.variants[inst.tag as usize];
            (str_eq(var.name, name), var.fields.len())
        } else {
            (str_eq(inst.ty.name, name), inst.fields.len())
        };
        if !matches {
            return Ok(false);
        }
        if arity != 255 && arity as usize != n {
            return Err(rt_err(format!(
                "pattern '{}' has {} field{} but the value has {}",
                name.as_str(),
                arity,
                if arity == 1 { "" } else { "s" },
                n
            )));
        }
        Ok(true)
    }

    // ---- modules -------------------------------------------------------------------

    /// Directory that relative imports are resolved against: the directory of the
    /// file containing the running `import` (the working directory for the REPL / `-e`).
    fn import_base(&self) -> std::path::PathBuf {
        let name = self.frames.last().map(|f| f.closure.proto.source.name.clone()).unwrap_or_default();
        if name.starts_with('<') {
            return std::path::PathBuf::from(".");
        }
        match std::path::Path::new(&name).parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => std::path::PathBuf::from("."),
        }
    }

    /// `import "spec"`: find the file (`spec`, `spec.kite`, `spec/mod.kite`), run it once and
    /// return its exports table.
    fn import_module(&mut self, spec: Value) -> RtResult<Value> {
        let spec = spec.as_str().map(|s| s.as_str().to_string()).unwrap_or_default();
        if spec.starts_with("std/") {
            return match self.modules.get(&spec) {
                Some(ModuleState::Done(v)) => Ok(*v),
                _ => {
                    let mut known: Vec<&str> =
                        self.modules.keys().filter(|k| k.starts_with("std/")).map(|k| k.as_str()).collect();
                    known.sort();
                    Err(rt_err(format!("unknown standard module '{}' (available: {})", spec, known.join(", "))))
                }
            };
        }
        let base = self.import_base();
        let path = {
            let p = std::path::Path::new(&spec);
            let joined = if p.is_absolute() { p.to_path_buf() } else { base.join(p) };
            let candidates = [
                joined.clone(),
                std::path::PathBuf::from(format!("{}.kite", joined.display())),
                joined.join("mod.kite"),
            ];
            candidates.into_iter().find(|c| c.is_file()).ok_or_else(|| {
                rt_err(format!("cannot find module '{}' (searched relative to '{}')", spec, base.display()))
            })?
        };
        let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone()).to_string_lossy().into_owned();
        match self.modules.get(&key) {
            Some(ModuleState::Done(v)) => return Ok(*v),
            Some(ModuleState::Loading) => return Err(rt_err(format!("circular import of module '{}'", spec))),
            None => {}
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| rt_err(format!("cannot read module '{}': {}", path.display(), e)))?;
        self.modules.insert(key.clone(), ModuleState::Loading);
        let result = self.run_module(&path.to_string_lossy(), &text);
        match result {
            Ok(v) => {
                self.modules.insert(key, ModuleState::Done(v));
                Ok(v)
            }
            Err(e) => {
                self.modules.remove(&key);
                Err(e)
            }
        }
    }

    fn run_module(&mut self, name: &str, text: &str) -> RtResult<Value> {
        let (proto, _) =
            self.compile_chunk(name, text, true).map_err(|e| rt_err(format!("while importing '{}':\n{}", name, e)))?;
        let closure = self.heap.alloc(Closure { hdr: Header::new(Kind::Closure), proto, upvals: Box::new([]) });
        self.call_value(Value::from(closure), &[])
    }

    // ---- hooks & semantics used by slow paths ------------------------------------------

    fn instance_hook(&self, v: Value, h: Hook) -> Option<Value> {
        let inst = v.as_gc::<Instance>()?;
        let f = inst.ty.hook(h);
        if f.is_nil() {
            None
        } else {
            Some(f)
        }
    }

    fn binary_hook(&self, h: Hook, a: Value, b: Value) -> Option<Value> {
        self.instance_hook(a, h).or_else(|| self.instance_hook(b, h))
    }

    /// An operator hook implemented by a bytecode function of two parameters.  The interpreter
    /// loop runs those as ordinary calls (no nested `run`), which makes overloaded operators
    /// about as cheap as function calls.
    fn arith_hook_closure(&self, op: Arith, a: Value, b: Value) -> Option<Gc<Closure>> {
        if a.is_num() && b.is_num() || (op == Arith::Add && (a.is_str() || b.is_str())) {
            return None;
        }
        let f = self.binary_hook(op.hook()?, a, b)?;
        let cl = f.as_gc::<Closure>()?;
        (cl.proto.arity == 2).then_some(cl)
    }

    fn arith_slow(&mut self, op: Arith, a: Value, b: Value) -> RtResult<Value> {
        if op == Arith::Add && (a.is_str() || b.is_str()) {
            // exact capacity for string operands (numbers etc. need a few bytes): one allocation, one copy
            let size = |v: Value| v.as_str().map_or(24, |s| s.as_str().len());
            let mut s = String::with_capacity(size(a) + size(b));
            self.write_value(a, &mut s, false, 0)?;
            self.write_value(b, &mut s, false, 0)?;
            return Ok(self.new_string(s));
        }
        if a.is_num() && b.is_num() {
            let (x, y) = (a.as_num(), b.as_num());
            return Ok(Value::num(arith_num(op, x, y)));
        }
        if let Some(h) = op.hook() {
            if let Some(f) = self.binary_hook(h, a, b) {
                return self.call_value(f, &[a, b]);
            }
        }
        Err(rt_err(format!("invalid operands for '{}': {} and {}", op.symbol(), self.type_of(a), self.type_of(b))))
    }

    /// Type name for diagnostics (struct name for instances).
    pub fn type_of(&self, v: Value) -> String {
        match v.as_gc::<Instance>() {
            Some(i) => i.ty.name.as_str().to_string(),
            None => v.type_name().to_string(),
        }
    }

    fn neg_slow(&mut self, a: Value) -> RtResult<Value> {
        if let Some(f) = self.instance_hook(a, Hook::Neg) {
            return self.call_value(f, &[a]);
        }
        Err(rt_err(format!("cannot negate {}", self.type_of(a))))
    }

    pub fn values_equal(&mut self, a: Value, b: Value) -> RtResult<bool> {
        if a.is_num() && b.is_num() {
            return Ok(a.as_num() == b.as_num());
        }
        if a.bits() == b.bits() {
            return Ok(true);
        }
        if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
            return Ok(str_eq(x, y));
        }
        if let Some(f) = self.binary_hook(Hook::Eq, a, b) {
            return Ok(self.call_value(f, &[a, b])?.is_truthy());
        }
        Ok(false)
    }

    fn compare_slow(&mut self, kind: Cmp, a: Value, b: Value) -> RtResult<bool> {
        match kind {
            Cmp::Eq => return self.values_equal(a, b),
            Cmp::Ne => return self.values_equal(a, b).map(|r| !r),
            _ => {}
        }
        if a.is_num() && b.is_num() {
            let (x, y) = (a.as_num(), b.as_num());
            return Ok(match kind {
                Cmp::Lt => x < y,
                Cmp::Le => x <= y,
                Cmp::Gt => x > y,
                _ => x >= y,
            });
        }
        if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
            let o = x.as_str().cmp(y.as_str());
            use std::cmp::Ordering::*;
            return Ok(match kind {
                Cmp::Lt => o == Less,
                Cmp::Le => o != Greater,
                Cmp::Gt => o == Greater,
                _ => o != Less,
            });
        }
        // `a > b` is `b < a`, `a >= b` is `b <= a`
        let (hook, x, y) = match kind {
            Cmp::Lt => (Hook::Lt, a, b),
            Cmp::Gt => (Hook::Lt, b, a),
            Cmp::Le => (Hook::Le, a, b),
            _ => (Hook::Le, b, a),
        };
        if let Some(f) = self.binary_hook(hook, x, y) {
            return Ok(self.call_value(f, &[x, y])?.is_truthy());
        }
        // `le` falls back to `!lt(b, a)`
        if hook == Hook::Le {
            if let Some(f) = self.binary_hook(Hook::Lt, y, x) {
                return Ok(!self.call_value(f, &[y, x])?.is_truthy());
            }
        }
        Err(rt_err(format!("cannot compare {} and {}", self.type_of(a), self.type_of(b))))
    }

    fn get_field_slow(&mut self, obj: Value, key: Gc<Str>) -> RtResult<Value> {
        match obj.obj_kind() {
            Some(Kind::Instance) => {
                let inst = unsafe { obj.as_gc_unchecked::<Instance>() };
                if let Some(i) = inst.ty.field_index(key) {
                    return Ok(inst.fields[i]);
                }
                if inst.ty.is_union() {
                    let var = &inst.ty.variants[inst.tag as usize];
                    if let Some(i) = var.fields.iter().position(|&f| str_eq(f, key)) {
                        return Ok(inst.fields[i]);
                    }
                }
                if let Some(f) = self.instance_hook(obj, Hook::Get) {
                    return self.call_value(f, &[obj, Value::from(key)]);
                }
                if let Some(m) = inst.ty.method(key) {
                    return Ok(m);
                }
                Err(rt_err(format!("struct '{}' has no field '{}'", inst.ty.name.as_str(), key.as_str())))
            }
            Some(Kind::Table) => Ok(unsafe { obj.as_gc_unchecked::<Table>() }.get_str(key)),
            Some(Kind::StructType) => {
                let ty = unsafe { obj.as_gc_unchecked::<StructType>() };
                if let Some(i) = ty.variant_index(key) {
                    return Ok(ty.variants[i].value);
                }
                ty.method(key)
                    .ok_or_else(|| rt_err(format!("struct '{}' has no function '{}'", ty.name.as_str(), key.as_str())))
            }
            _ => Err(rt_err(format!("cannot read field '{}' of {}", key.as_str(), self.type_of(obj)))),
        }
    }

    fn set_field_slow(&mut self, obj: Value, key: Gc<Str>, v: Value) -> RtResult<()> {
        match obj.obj_kind() {
            Some(Kind::Instance) => {
                let mut inst = unsafe { obj.as_gc_unchecked::<Instance>() };
                if let Some(i) = inst.ty.field_index(key) {
                    inst.fields[i] = v;
                    return Ok(());
                }
                if inst.ty.is_union() {
                    let pos = inst.ty.variants[inst.tag as usize].fields.iter().position(|&f| str_eq(f, key));
                    if let Some(i) = pos {
                        inst.fields[i] = v;
                        return Ok(());
                    }
                }
                if let Some(f) = self.instance_hook(obj, Hook::Set) {
                    self.call_value(f, &[obj, Value::from(key), v])?;
                    return Ok(());
                }
                Err(rt_err(format!("struct '{}' has no field '{}'", inst.ty.name.as_str(), key.as_str())))
            }
            Some(Kind::Table) => {
                let mut t = unsafe { obj.as_gc_unchecked::<Table>() };
                self.table_set(&mut t, Value::from(key), v)
            }
            _ => Err(rt_err(format!("cannot set field '{}' of {}", key.as_str(), self.type_of(obj)))),
        }
    }

    #[inline]
    fn table_set(&mut self, t: &mut Gc<Table>, k: Value, v: Value) -> RtResult<()> {
        let before = t.mem_size();
        let r = t.set(k, v);
        let after = t.mem_size();
        self.heap.bytes += after.saturating_sub(before);
        r.map_err(|e| {
            rt_err(match e {
                KeyError::Nil => "table key cannot be nil",
                KeyError::NaN => "table key cannot be NaN",
            })
        })
    }

    /// Bounds-checked position for `arr[key]`; every failure is a panic.
    fn array_position(&self, arr: &Array, key: Value) -> RtResult<usize> {
        if !key.is_num() {
            return Err(rt_err(format!("array index must be a number, got {}", self.type_of(key))));
        }
        let n = key.as_num();
        if n.fract() != 0.0 || n < 0.0 {
            return Err(rt_err(format!("array index must be a non-negative integer, got {}", fmt_num_string(n))));
        }
        if n >= arr.items.len() as f64 {
            return Err(rt_err(format!(
                "array index {} is out of bounds (length {})",
                fmt_num_string(n),
                arr.items.len()
            )));
        }
        Ok(n as usize)
    }

    fn get_index_slow(&mut self, obj: Value, key: Value) -> RtResult<Value> {
        match obj.obj_kind() {
            Some(Kind::Array) => {
                let a = unsafe { obj.as_gc_unchecked::<Array>() };
                let i = self.array_position(&a, key)?;
                Ok(a.items[i])
            }
            Some(Kind::Table) => Ok(unsafe { obj.as_gc_unchecked::<Table>() }.get(key)),
            Some(Kind::Str) => {
                let s = unsafe { obj.as_gc_unchecked::<Str>() };
                if !key.is_num() {
                    return Err(rt_err(format!("string index must be a number, got {}", self.type_of(key))));
                }
                let n = key.as_num();
                let text = s.as_str();
                let idx = if n < 0.0 { n + char_count(text) as f64 } else { n };
                if idx < 0.0 || idx.fract() != 0.0 {
                    return Ok(Value::NIL);
                }
                match nth_char(text, idx as usize) {
                    Some(c) => {
                        let mut buf = [0u8; 4];
                        Ok(self.new_string(&*c.encode_utf8(&mut buf)))
                    }
                    None => Ok(Value::NIL),
                }
            }
            Some(Kind::Instance) => {
                if let Some(f) = self.instance_hook(obj, Hook::Get) {
                    return self.call_value(f, &[obj, key]);
                }
                if let Some(k) = key.as_str() {
                    return self.get_field_slow(obj, k);
                }
                let inst = unsafe { obj.as_gc_unchecked::<Instance>() };
                Err(rt_err(format!(
                    "struct '{}' cannot be indexed by {} (no `get` method)",
                    inst.ty.name.as_str(),
                    self.type_of(key)
                )))
            }
            _ => Err(rt_err(format!("cannot index a value of type {}", self.type_of(obj)))),
        }
    }

    fn set_index_slow(&mut self, obj: Value, key: Value, v: Value) -> RtResult<()> {
        match obj.obj_kind() {
            Some(Kind::Array) => {
                let mut a = unsafe { obj.as_gc_unchecked::<Array>() };
                let i = self.array_position(&a, key)?;
                a.items[i] = v;
                Ok(())
            }
            Some(Kind::Table) => {
                let mut t = unsafe { obj.as_gc_unchecked::<Table>() };
                self.table_set(&mut t, key, v)
            }
            Some(Kind::Instance) => {
                if let Some(f) = self.instance_hook(obj, Hook::Set) {
                    self.call_value(f, &[obj, key, v])?;
                    return Ok(());
                }
                if let Some(k) = key.as_str() {
                    return self.set_field_slow(obj, k, v);
                }
                let inst = unsafe { obj.as_gc_unchecked::<Instance>() };
                Err(rt_err(format!(
                    "struct '{}' cannot be indexed by {} (no `set` method)",
                    inst.ty.name.as_str(),
                    self.type_of(key)
                )))
            }
            _ => Err(rt_err(format!("cannot assign into a value of type {}", self.type_of(obj)))),
        }
    }

    /// Resolve `obj:name(...)`.
    fn get_method(&mut self, obj: Value, name: Gc<Str>) -> RtResult<Value> {
        match obj.obj_kind() {
            Some(Kind::Instance) => {
                let inst = unsafe { obj.as_gc_unchecked::<Instance>() };
                if let Some(i) = inst.ty.field_index(name) {
                    return Ok(inst.fields[i]);
                }
                inst.ty.method(name).ok_or_else(|| {
                    rt_err(format!("struct '{}' has no method '{}'", inst.ty.name.as_str(), name.as_str()))
                })
            }
            Some(Kind::Table) => {
                let v = unsafe { obj.as_gc_unchecked::<Table>() }.get_str(name);
                if !v.is_nil() {
                    return Ok(v);
                }
                if let Some(lib) = self.table_lib.as_gc::<Table>() {
                    let f = lib.get_str(name);
                    if !f.is_nil() {
                        return Ok(f);
                    }
                }
                Err(rt_err(format!("table has no function '{}'", name.as_str())))
            }
            Some(Kind::Array) => {
                if let Some(lib) = self.array_lib.as_gc::<Table>() {
                    let f = lib.get_str(name);
                    if !f.is_nil() {
                        return Ok(f);
                    }
                }
                Err(rt_err(format!("array has no method '{}'", name.as_str())))
            }
            Some(Kind::Str) => {
                if let Some(lib) = self.string_lib.as_gc::<Table>() {
                    let f = lib.get_str(name);
                    if !f.is_nil() {
                        return Ok(f);
                    }
                }
                Err(rt_err(format!("string has no method '{}'", name.as_str())))
            }
            _ => Err(rt_err(format!("cannot call method '{}' on {}", name.as_str(), self.type_of(obj)))),
        }
    }

    /// Advance table / string iteration.
    fn iter_next(&mut self, v: Value, pos: usize) -> RtResult<Option<(usize, Value, Value)>> {
        match v.obj_kind() {
            Some(Kind::Table) => Ok(unsafe { v.as_gc_unchecked::<Table>() }.next(pos)),
            Some(Kind::Array) => {
                let a = unsafe { v.as_gc_unchecked::<Array>() };
                Ok(a.items.get(pos).map(|&x| (pos + 1, Value::num(pos as f64), x)))
            }
            Some(Kind::Str) => {
                let s = unsafe { v.as_gc_unchecked::<Str>() };
                let text = s.as_str();
                if pos >= text.len() {
                    return Ok(None);
                }
                let c = text[pos..].chars().next().unwrap();
                let mut buf = [0u8; 4];
                let cs = self.new_string(&*c.encode_utf8(&mut buf));
                Ok(Some((pos + c.len_utf8(), Value::num(pos as f64), cs)))
            }
            _ => Err(rt_err(format!("cannot iterate over {}", self.type_of(v)))),
        }
    }

    // ---- formatting -------------------------------------------------------------------

    /// `print`-style conversion (strings raw).
    pub fn display(&mut self, v: Value) -> RtResult<String> {
        let mut s = String::new();
        self.write_value(v, &mut s, false, 0)?;
        Ok(s)
    }

    /// Debug-style conversion (strings quoted); used by the REPL and inside containers.
    pub fn repr(&mut self, v: Value) -> RtResult<String> {
        let mut s = String::new();
        self.write_value(v, &mut s, true, 0)?;
        Ok(s)
    }

    pub fn write_value(&mut self, v: Value, out: &mut String, quote: bool, depth: usize) -> RtResult<()> {
        use std::fmt::Write;
        if v.is_num() {
            fmt_num(v.as_num(), out);
            return Ok(());
        }
        if v.is_nil() {
            out.push_str("nil");
            return Ok(());
        }
        if v.is_bool() {
            out.push_str(if v == Value::TRUE { "true" } else { "false" });
            return Ok(());
        }
        match v.obj_kind() {
            Some(Kind::Str) => {
                let s = unsafe { v.as_gc_unchecked::<Str>() };
                if quote {
                    out.push('"');
                    for c in s.as_str().chars() {
                        match c {
                            '"' => out.push_str("\\\""),
                            '\\' => out.push_str("\\\\"),
                            '\n' => out.push_str("\\n"),
                            '\t' => out.push_str("\\t"),
                            '\r' => out.push_str("\\r"),
                            c => out.push(c),
                        }
                    }
                    out.push('"');
                } else {
                    out.push_str(s.as_str());
                }
            }
            Some(Kind::Array) => {
                if depth > 6 {
                    out.push_str("[...]");
                    return Ok(());
                }
                let a = unsafe { v.as_gc_unchecked::<Array>() };
                out.push('[');
                // snapshot: printing may call `str` hooks that mutate the array
                let items = a.items.clone();
                for (i, item) in items.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.write_value(item, out, true, depth + 1)?;
                }
                out.push(']');
            }
            Some(Kind::Table) => {
                if depth > 6 {
                    out.push_str("{...}");
                    return Ok(());
                }
                let t = unsafe { v.as_gc_unchecked::<Table>() };
                out.push('{');
                let mut pos = 0;
                let mut first = true;
                // snapshot entries: printing may call `str` hooks that mutate the table
                let mut items = Vec::new();
                while let Some((np, k, val)) = t.next(pos) {
                    pos = np;
                    items.push((k, val));
                }
                for (k, val) in items {
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    match k.as_str() {
                        Some(ks) if is_ident(ks.as_str()) => out.push_str(ks.as_str()),
                        _ => self.write_value(k, out, true, depth + 1)?,
                    }
                    out.push_str(": ");
                    self.write_value(val, out, true, depth + 1)?;
                }
                out.push('}');
            }
            Some(Kind::Closure) => {
                let c = unsafe { v.as_gc_unchecked::<Closure>() };
                let _ = write!(out, "<fn {}>", c.proto.name);
            }
            Some(Kind::Native) => {
                let n = unsafe { v.as_gc_unchecked::<Native>() };
                let _ = write!(out, "<native {}>", n.name);
            }
            Some(Kind::Proto) => out.push_str("<proto>"),
            Some(Kind::Upvalue) => out.push_str("<upvalue>"),
            Some(Kind::StructType) => {
                let t = unsafe { v.as_gc_unchecked::<StructType>() };
                let _ = write!(out, "<{} {}>", if t.is_union() { "union" } else { "struct" }, t.name.as_str());
            }
            Some(Kind::Variant) => {
                let c = unsafe { v.as_gc_unchecked::<VariantCtor>() };
                let _ = write!(out, "<variant {}.{}>", c.ty.name.as_str(), c.ty.variants[c.tag as usize].name.as_str());
            }
            Some(Kind::Instance) => {
                let inst = unsafe { v.as_gc_unchecked::<Instance>() };
                let hook = inst.ty.hook(Hook::Str);
                if !hook.is_nil() {
                    let r = self.call_value(hook, &[v])?;
                    match r.as_str() {
                        Some(s) => out.push_str(s.as_str()),
                        None => {
                            return Err(rt_err(format!(
                                "`str` of struct '{}' must return a string",
                                inst.ty.name.as_str()
                            )))
                        }
                    }
                    return Ok(());
                }
                if inst.ty.is_union() {
                    // `Circle(2)` / `Point`
                    out.push_str(inst.ty.variants[inst.tag as usize].name.as_str());
                    if !inst.fields.is_empty() {
                        if depth > 6 {
                            out.push_str("(...)");
                            return Ok(());
                        }
                        out.push('(');
                        let vals: Vec<Value> = inst.fields.to_vec();
                        for (i, val) in vals.into_iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            self.write_value(val, out, true, depth + 1)?;
                        }
                        out.push(')');
                    }
                    return Ok(());
                }
                if depth > 6 {
                    let _ = write!(out, "{}{{...}}", inst.ty.name.as_str());
                    return Ok(());
                }
                let _ = write!(out, "{}{{", inst.ty.name.as_str());
                let fields: Vec<(Gc<Str>, Value)> =
                    inst.ty.fields.iter().copied().zip(inst.fields.iter().copied()).collect();
                for (i, (name, val)) in fields.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(name.as_str());
                    out.push_str(": ");
                    self.write_value(val, out, true, depth + 1)?;
                }
                out.push('}');
            }
            None => out.push_str("<undefined>"),
        }
        Ok(())
    }

    // ---- garbage collection -----------------------------------------------------------

    /// Mark & sweep.  Must only be called at a safe point (`self.sp` synced).
    pub fn collect_garbage(&mut self) {
        self.collect(false)
    }

    /// Like `collect_garbage`, but trims the dead temporaries of the topmost frame; only valid
    /// when that frame is stopped in a call (the explicit `gc()` native).
    pub fn collect_garbage_precise(&mut self) {
        self.collect(true)
    }

    fn collect(&mut self, precise: bool) {
        unsafe {
            let (live_end, frame_top) = self.live_stack(precise);
            let heap = &mut self.heap;
            let mut p = self.stack;
            while p < live_end {
                heap.mark_value(*p);
                p = p.add(1);
            }
            // the dead registers are not marked, so they must not be read as live later on
            let mut p = live_end;
            while p < frame_top {
                *p = Value::NIL;
                p = p.add(1);
            }
            // Registers of the suspended frames that lie above the stack top (the callee's
            // frame is smaller) hold dead temporaries.  They are not marked, so they must be
            // cleared: when the frame resumes they would otherwise be scanned as dangling.
            let mut high = self.sp;
            for f in &self.frames {
                high = high.max(f.base.add(f.closure.proto.nregs as usize));
            }
            let mut p = self.sp;
            while p < high {
                *p = Value::NIL;
                p = p.add(1);
            }
            // values staged above the topmost frame (arguments of a native called back into
            // the VM, values rooted by natives)
            let mut p = frame_top;
            while p < self.sp {
                heap.mark_value(*p);
                p = p.add(1);
            }
            for f in &self.frames {
                heap.mark_gc(f.closure);
            }
            for &g in &self.globals {
                heap.mark_value(g);
            }
            for &u in &self.open_upvals {
                heap.mark_gc(u);
            }
            for &v in &self.pending_final {
                heap.mark_value(v);
            }
            heap.mark_value(self.string_lib);
            heap.mark_value(self.array_lib);
            heap.mark_value(self.table_lib);
            heap.mark_value(self.option_ty);
            heap.mark_value(self.result_ty);
            heap.mark_value(self.error_ty);
            for m in self.modules.values() {
                if let ModuleState::Done(v) = m {
                    heap.mark_value(*v);
                }
            }
            heap.trace();

            // Unreachable instances with a `destroy` method are resurrected once
            // so their finalizer can run; they are freed by a later collection.
            let mut doomed: Vec<*mut Header> = Vec::new();
            for o in heap.objects() {
                if (*o).kind == Kind::Instance && !(*o).marked {
                    let i = &*(o as *mut Instance);
                    if !i.finalized && !i.ty.hook(Hook::Destroy).is_nil() {
                        doomed.push(o);
                    }
                }
            }
            for &o in &doomed {
                (*(o as *mut Instance)).finalized = true;
                heap.mark_obj(o);
                self.pending_final.push(Value::from_header(o));
            }
            heap.trace();
            heap.sweep();
        }
        self.run_finalizers();
    }

    /// `(end of the live registers, top of the topmost frame)`.  Only `precise` collections
    /// (see `collect_garbage_precise`) trust the instruction the frame stopped at.
    ///
    /// The topmost frame is stopped right after a known instruction, which tells how many of
    /// its registers can hold live values: the locals, plus the temporaries that the
    /// instruction names (temporaries are handed out in stack order, so nothing live lies above
    /// the registers an instruction uses).  The dead temporaries above hold stale values that
    /// must not keep garbage alive - `work(); gc()` has to finalize what `work` left behind.
    /// Frames below the top are scanned completely.
    fn live_stack(&self, precise: bool) -> (*mut Value, *mut Value) {
        let Some(f) = self.frames.last() else { return (self.sp, self.sp) };
        unsafe {
            let proto = f.closure.proto;
            let top = f.base.add(proto.nregs as usize);
            let next = f.ip.offset_from(proto.code.as_ptr()) as usize;
            let mut at = next.saturating_sub(1).min(proto.code.len().saturating_sub(1));
            if at > 0 && matches!(proto.code[at], Op::Ext(_)) {
                at -= 1;
            }
            let live = match live_regs(proto.code[at]) {
                Some(n) if precise => n.max(proto.nslots as usize),
                _ => proto.nregs as usize,
            };
            (f.base.add(live.min(proto.nregs as usize)), top)
        }
    }

    fn run_finalizers(&mut self) {
        if self.finalizing {
            return;
        }
        self.finalizing = true;
        while let Some(obj) = self.pending_final.pop() {
            let Some(inst) = obj.as_gc::<Instance>() else { continue };
            let hook = inst.ty.hook(Hook::Destroy);
            if hook.is_nil() {
                continue;
            }
            if let Err(e) = self.call_protected(hook, &[obj]) {
                let _ = self.out.flush();
                eprintln!("warning: error in `destroy` of '{}': {}", inst.ty.name.as_str(), e.msg);
            }
        }
        self.finalizing = false;
    }

    /// Run `destroy` for every instance that is still alive (program exit).
    pub fn shutdown(&mut self) {
        let mut live: Vec<*mut Header> = Vec::new();
        for o in self.heap.objects() {
            unsafe {
                if (*o).kind == Kind::Instance {
                    let i = &*(o as *mut Instance);
                    if !i.finalized && !i.ty.hook(Hook::Destroy).is_nil() {
                        live.push(o);
                    }
                }
            }
        }
        // objects() is newest-first; destroy in creation order
        for &o in live.iter().rev() {
            unsafe { (*(o as *mut Instance)).finalized = true };
            self.pending_final.insert(0, Value::from_header(o));
        }
        self.run_finalizers();
        self.flush();
    }

    /// Explicitly run an instance's `destroy` now (and never again).
    pub fn destroy_now(&mut self, v: Value) -> RtResult<()> {
        let Some(mut inst) = v.as_gc::<Instance>() else {
            return Err(rt_err(format!("destroy expects a struct instance, got {}", self.type_of(v))));
        };
        if inst.finalized {
            return Ok(());
        }
        inst.finalized = true;
        let hook = inst.ty.hook(Hook::Destroy);
        if !hook.is_nil() {
            self.call_value(hook, &[v])?;
        }
        Ok(())
    }

    // ---- traces -------------------------------------------------------------------------

    fn build_trace(&self) -> Vec<TraceEntry> {
        let mut out = Vec::new();
        for f in self.frames.iter().rev() {
            let proto = f.closure.proto;
            let idx = unsafe { f.ip.offset_from(proto.code.as_ptr()) } as usize;
            let idx = idx.saturating_sub(1).min(proto.offsets.len().saturating_sub(1));
            let offset = proto.offsets.get(idx).copied().unwrap_or(0);
            out.push(TraceEntry { func: proto.name.clone(), source: proto.source.clone(), offset });
        }
        out
    }

    // ---- the interpreter loop -------------------------------------------------------------

    /// Run until the frame depth drops back to `stop`.
    pub fn run(&mut self, stop: usize) -> RtResult<Value> {
        match self.run_inner(stop) {
            Err(mut e) => {
                if e.trace.is_empty() {
                    e.trace = self.build_trace();
                }
                Err(e)
            }
            ok => ok,
        }
    }

    #[allow(unused_assignments)]
    fn run_inner(&mut self, stop: usize) -> RtResult<Value> {
        unsafe {
            let mut frame: *mut Frame = self.frames.as_mut_ptr().add(self.frames.len() - 1);
            let mut ip: *const Op = (*frame).ip;
            let mut base: *mut Value = (*frame).base;

            // copy the handle out of the frame first: auto-deref must not borrow through the raw pointer
            macro_rules! closure {
                () => {{
                    let c: Gc<Closure> = (*frame).closure;
                    c
                }};
            }
            let mut consts: *const Value = closure!().proto.consts.as_ptr();

            // register `i` of the current frame (usable as a place)
            macro_rules! reg {
                ($i:expr) => {
                    *base.add($i as usize)
                };
            }
            macro_rules! konst {
                ($i:expr) => {
                    *consts.add($i as usize)
                };
            }
            // `self.sp` is always the top of the current frame, so only `ip` has to be saved
            macro_rules! sync {
                () => {{
                    (*frame).ip = ip;
                }};
            }
            // switch to the frame on top (after a call or a return)
            macro_rules! reload_nosp {
                () => {{
                    frame = self.frames.as_mut_ptr().add(self.frames.len() - 1);
                    ip = (*frame).ip;
                    base = (*frame).base;
                    consts = closure!().proto.consts.as_ptr();
                }};
            }
            macro_rules! reload {
                () => {{
                    frame = self.frames.as_mut_ptr().add(self.frames.len() - 1);
                    ip = (*frame).ip;
                    base = (*frame).base;
                    let proto = closure!().proto;
                    consts = proto.consts.as_ptr();
                    self.sp = base.add(proto.nregs as usize);
                }};
            }
            // run something fallible; on error the state is already synced for the trace
            macro_rules! protect {
                ($e:expr) => {{
                    sync!();
                    match $e {
                        Ok(v) => v,
                        Err(e) => return Err(e),
                    }
                }};
            }
            macro_rules! throw {
                ($($a:tt)*) => {{
                    sync!();
                    return Err(rt_err(format!($($a)*)));
                }};
            }
            macro_rules! gc_check {
                () => {
                    if self.heap.should_collect() {
                        sync!();
                        self.collect_garbage();
                    }
                };
            }
            macro_rules! jump {
                ($o:expr) => {
                    ip = ip.offset($o as isize)
                };
            }
            // the offset / payload of a double instruction lives in the word after it
            macro_rules! ext {
                () => {{
                    let w = *ip;
                    ip = ip.add(1);
                    w.simm24()
                }};
            }
            macro_rules! payload {
                () => {{
                    let w = *ip;
                    ip = ip.add(1);
                    w.payload()
                }};
            }
            // `R[a] = x <op> y`: numbers inline, operator hooks written in Kite run as calls
            // inside this loop (their result lands in `R[a]` when they return), the rest is slow
            macro_rules! arith {
                ($kind:expr, $a:expr, $x:expr, $y:expr, $fast:expr) => {{
                    let (xv, yv): (Value, Value) = ($x, $y);
                    if xv.is_num() && yv.is_num() {
                        reg!($a) = Value::num(($fast)(xv.as_num(), yv.as_num()));
                    } else if let Some(cl) = self.arith_hook_closure($kind, xv, yv) {
                        // stage [hook, x, y] just above the frame
                        let top = self.sp;
                        *top = Value::from(cl);
                        *top.add(1) = xv;
                        *top.add(2) = yv;
                        sync!();
                        self.push_frame(cl, top.add(1), 2, $a as u32)?;
                        reload_nosp!();
                    } else {
                        sync!();
                        reg!($a) = self.arith_slow($kind, xv, yv)?;
                        gc_check!();
                    }
                }};
            }
            macro_rules! bitop {
                ($kind:expr, $a:expr, $x:expr, $y:expr, $fast:expr) => {{
                    let (xv, yv): (Value, Value) = ($x, $y);
                    if xv.is_num() && yv.is_num() {
                        reg!($a) = Value::num(($fast)(xv.as_num() as i64, yv.as_num() as i64) as f64);
                    } else {
                        sync!();
                        reg!($a) = self.arith_slow($kind, xv, yv)?;
                    }
                }};
            }
            // `R[a] = x <cmp> y`
            macro_rules! compare {
                ($kind:expr, $a:expr, $x:expr, $y:expr, $fast:expr) => {{
                    let (xv, yv): (Value, Value) = ($x, $y);
                    let r = if xv.is_num() && yv.is_num() {
                        ($fast)(xv.as_num(), yv.as_num())
                    } else {
                        sync!();
                        self.compare_slow($kind, xv, yv)?
                    };
                    reg!($a) = Value::bool(r);
                }};
            }
            // jump if `x <cmp> y` equals `$want`
            macro_rules! jcmp {
                ($kind:expr, $x:expr, $y:expr, $want:expr, $fast:expr) => {{
                    let o = ext!();
                    let (xv, yv): (Value, Value) = ($x, $y);
                    let t = if xv.is_num() && yv.is_num() {
                        ($fast)(xv.as_num(), yv.as_num())
                    } else {
                        sync!();
                        self.compare_slow($kind, xv, yv)?
                    };
                    if t == $want {
                        jump!(o);
                    }
                }};
            }
            // `==` / `!=`: immediates compare by their bits, only objects need the slow path
            macro_rules! jeq {
                ($x:expr, $y:expr, $want:expr) => {{
                    let o = ext!();
                    let (xv, yv): (Value, Value) = ($x, $y);
                    let t = if xv.is_num() && yv.is_num() {
                        xv.as_num() == yv.as_num()
                    } else if !xv.is_obj() && !yv.is_obj() {
                        xv.bits() == yv.bits()
                    } else {
                        sync!();
                        self.values_equal(xv, yv)?
                    };
                    if t == $want {
                        jump!(o);
                    }
                }};
            }
            // read `obj.name` where `name` is the interned string constant `k`
            macro_rules! get_field {
                ($obj:expr, $key:expr) => {{
                    let obj: Value = $obj;
                    let key = ($key).as_gc_unchecked::<Str>();
                    match obj.obj_kind() {
                        Some(Kind::Table) => obj.as_gc_unchecked::<Table>().get_str(key),
                        Some(Kind::Instance) => {
                            let inst = obj.as_gc_unchecked::<Instance>();
                            match inst.ty.field_index(key) {
                                Some(i) => *inst.fields.get_unchecked(i),
                                None => protect!(self.get_field_slow(obj, key)),
                            }
                        }
                        _ => protect!(self.get_field_slow(obj, key)),
                    }
                }};
            }
            macro_rules! set_field {
                ($obj:expr, $key:expr, $v:expr) => {{
                    let (obj, v): (Value, Value) = ($obj, $v);
                    let key = ($key).as_gc_unchecked::<Str>();
                    let mut done = false;
                    if let Some(mut inst) = obj.as_gc::<Instance>() {
                        if let Some(i) = inst.ty.field_index(key) {
                            *inst.fields.get_unchecked_mut(i) = v;
                            done = true;
                        }
                    }
                    if !done {
                        protect!(self.set_field_slow(obj, key, v));
                    }
                    gc_check!();
                }};
            }
            macro_rules! init_field {
                ($t:expr, $key:expr, $v:expr) => {{
                    let mut t = ($t).as_gc_unchecked::<Table>();
                    let before = t.mem_size();
                    let _ = t.set($key, $v);
                    self.heap.bytes += t.mem_size().saturating_sub(before);
                }};
            }
            // call the value in register `a` with `n` arguments behind it; the result replaces it
            macro_rules! call_reg {
                ($a:expr, $n:expr) => {{
                    let a = $a as usize;
                    let n = $n as usize;
                    let slot = base.add(a);
                    let callee = *slot;
                    sync!();
                    if callee.is_kind(Kind::Closure) {
                        self.push_frame(callee.as_gc_unchecked::<Closure>(), slot.add(1), n, a as u32)?;
                        reload_nosp!();
                    } else if callee.is_kind(Kind::Native)
                        && (callee.as_gc_unchecked::<Native>().arity < 0
                            || callee.as_gc_unchecked::<Native>().arity as usize == n)
                    {
                        // a native with a matching arity: call it straight from the loop
                        let nat = callee.as_gc_unchecked::<Native>();
                        let args = std::slice::from_raw_parts(slot.add(1), n);
                        *slot = (nat.func)(self, args)?;
                        gc_check!();
                    } else {
                        match self.call_at(slot, n, a as u32) {
                            Ok(true) => reload_nosp!(),
                            Ok(false) => gc_check!(),
                            Err(e) => return Err(e),
                        }
                    }
                }};
            }
            macro_rules! do_return {
                ($result:expr) => {{
                    let result: Value = $result;
                    if !self.open_upvals.is_empty() {
                        self.close_upvalues(base);
                    }
                    let ret = (*frame).ret;
                    self.frames.pop();
                    if self.frames.len() == stop {
                        // back to the register the caller of `run` staged the callee in
                        self.sp = base.sub(1);
                        return Ok(result);
                    }
                    reload!();
                    reg!(ret) = result;
                }};
            }

            loop {
                let op = *ip;
                ip = ip.add(1);
                #[cfg(feature = "op-profile")]
                crate::profile::record(&op);
                match op {
                    // ---- moves and constants
                    Op::Move(a, b) => reg!(a) = reg!(b),
                    Op::LoadK(a, k) => reg!(a) = konst!(k),
                    Op::LoadKW(a) => {
                        let k = payload!();
                        reg!(a) = konst!(k);
                    }
                    Op::LoadI(a, i) => reg!(a) = Value::num(i as f64),
                    Op::LoadNil(a, n) => {
                        for i in 0..=n as usize {
                            reg!(a as usize + i) = Value::NIL;
                        }
                    }
                    Op::LoadTrue(a) => reg!(a) = Value::TRUE,
                    Op::LoadFalse(a) => reg!(a) = Value::FALSE,

                    // ---- upvalues and globals
                    Op::GetUpval(a, i) => {
                        let uv = *closure!().upvals.get_unchecked(i as usize);
                        reg!(a) = *uv.location;
                    }
                    Op::SetUpval(a, i) => {
                        let uv = *closure!().upvals.get_unchecked(i as usize);
                        let loc = uv.location;
                        *loc = reg!(a);
                    }
                    Op::GetGlobal(a, s) => {
                        let v = *self.globals.get_unchecked(s as usize);
                        if v == Value::UNDEF {
                            throw!("undefined variable '{}'", self.interner.get(Sym(s as u32)));
                        }
                        reg!(a) = v;
                    }
                    Op::GetGlobalW(a) => {
                        let s = payload!();
                        let v = *self.globals.get_unchecked(s as usize);
                        if v == Value::UNDEF {
                            throw!("undefined variable '{}'", self.interner.get(Sym(s)));
                        }
                        reg!(a) = v;
                    }
                    Op::SetGlobal(a, s) => {
                        let slot = self.globals.get_unchecked_mut(s as usize);
                        if *slot == Value::UNDEF {
                            throw!("assignment to undefined variable '{}'", self.interner.get(Sym(s as u32)));
                        }
                        *slot = reg!(a);
                    }
                    Op::SetGlobalW(a) => {
                        let s = payload!();
                        let slot = self.globals.get_unchecked_mut(s as usize);
                        if *slot == Value::UNDEF {
                            throw!("assignment to undefined variable '{}'", self.interner.get(Sym(s)));
                        }
                        *slot = reg!(a);
                    }
                    Op::DefGlobal(a, s) => *self.globals.get_unchecked_mut(s as usize) = reg!(a),
                    Op::DefGlobalW(a) => {
                        let s = payload!();
                        *self.globals.get_unchecked_mut(s as usize) = reg!(a);
                    }
                    Op::CloseUpvals(a) => {
                        if !self.open_upvals.is_empty() {
                            self.close_upvalues(base.add(a as usize));
                        }
                    }

                    // ---- arithmetic
                    Op::Add(a, b, c) => arith!(Arith::Add, a, reg!(b), reg!(c), |x: f64, y: f64| x + y),
                    Op::Sub(a, b, c) => arith!(Arith::Sub, a, reg!(b), reg!(c), |x: f64, y: f64| x - y),
                    Op::Mul(a, b, c) => arith!(Arith::Mul, a, reg!(b), reg!(c), |x: f64, y: f64| x * y),
                    Op::Div(a, b, c) => arith!(Arith::Div, a, reg!(b), reg!(c), |x: f64, y: f64| x / y),
                    Op::Rem(a, b, c) => arith!(Arith::Rem, a, reg!(b), reg!(c), rem_f64),
                    Op::Pow(a, b, c) => arith!(Arith::Pow, a, reg!(b), reg!(c), |x: f64, y: f64| x.powf(y)),
                    Op::AddK(a, b, k) => arith!(Arith::Add, a, reg!(b), konst!(k), |x: f64, y: f64| x + y),
                    Op::SubK(a, b, k) => arith!(Arith::Sub, a, reg!(b), konst!(k), |x: f64, y: f64| x - y),
                    Op::MulK(a, b, k) => arith!(Arith::Mul, a, reg!(b), konst!(k), |x: f64, y: f64| x * y),
                    Op::DivK(a, b, k) => arith!(Arith::Div, a, reg!(b), konst!(k), |x: f64, y: f64| x / y),
                    Op::RemK(a, b, k) => arith!(Arith::Rem, a, reg!(b), konst!(k), rem_f64),
                    Op::BitAnd(a, b, c) => bitop!(Arith::BitAnd, a, reg!(b), reg!(c), |x: i64, y: i64| x & y),
                    Op::BitOr(a, b, c) => bitop!(Arith::BitOr, a, reg!(b), reg!(c), |x: i64, y: i64| x | y),
                    Op::BitXor(a, b, c) => bitop!(Arith::BitXor, a, reg!(b), reg!(c), |x: i64, y: i64| x ^ y),
                    Op::Shl(a, b, c) => {
                        bitop!(Arith::Shl, a, reg!(b), reg!(c), |x: i64, y: i64| x.wrapping_shl((y & 63) as u32))
                    }
                    Op::Shr(a, b, c) => {
                        bitop!(Arith::Shr, a, reg!(b), reg!(c), |x: i64, y: i64| x.wrapping_shr((y & 63) as u32))
                    }
                    Op::Neg(a, b) => {
                        let x = reg!(b);
                        if x.is_num() {
                            reg!(a) = Value::num(-x.as_num());
                        } else {
                            reg!(a) = protect!(self.neg_slow(x));
                        }
                    }
                    Op::Not(a, b) => reg!(a) = Value::bool(reg!(b).is_falsy()),
                    Op::BitNot(a, b) => {
                        let x = reg!(b);
                        if x.is_num() {
                            reg!(a) = Value::num(!(x.as_num() as i64) as f64);
                        } else {
                            throw!("invalid operand for '~': {}", self.type_of(x));
                        }
                    }

                    // ---- comparisons producing a bool
                    Op::Eq(a, b, c) => {
                        let (x, y) = (reg!(b), reg!(c));
                        let r = if x.is_num() && y.is_num() {
                            x.as_num() == y.as_num()
                        } else if !x.is_obj() && !y.is_obj() {
                            x.bits() == y.bits()
                        } else {
                            protect!(self.values_equal(x, y))
                        };
                        reg!(a) = Value::bool(r);
                    }
                    Op::Ne(a, b, c) => {
                        let (x, y) = (reg!(b), reg!(c));
                        let r = if x.is_num() && y.is_num() {
                            x.as_num() != y.as_num()
                        } else if !x.is_obj() && !y.is_obj() {
                            x.bits() != y.bits()
                        } else {
                            !protect!(self.values_equal(x, y))
                        };
                        reg!(a) = Value::bool(r);
                    }
                    Op::Lt(a, b, c) => compare!(Cmp::Lt, a, reg!(b), reg!(c), |x: f64, y: f64| x < y),
                    Op::Le(a, b, c) => compare!(Cmp::Le, a, reg!(b), reg!(c), |x: f64, y: f64| x <= y),
                    Op::Gt(a, b, c) => compare!(Cmp::Gt, a, reg!(b), reg!(c), |x: f64, y: f64| x > y),
                    Op::Ge(a, b, c) => compare!(Cmp::Ge, a, reg!(b), reg!(c), |x: f64, y: f64| x >= y),

                    // ---- jumps
                    Op::Jump(_) => jump!(op.simm24()),
                    Op::JTrue(a) => {
                        let o = ext!();
                        if reg!(a).is_truthy() {
                            jump!(o);
                        }
                    }
                    Op::JFalse(a) => {
                        let o = ext!();
                        if reg!(a).is_falsy() {
                            jump!(o);
                        }
                    }
                    Op::JEq(a, b) => jeq!(reg!(a), reg!(b), true),
                    Op::JNe(a, b) => jeq!(reg!(a), reg!(b), false),
                    Op::JLt(a, b) => jcmp!(Cmp::Lt, reg!(a), reg!(b), true, |x: f64, y: f64| x < y),
                    Op::JNotLt(a, b) => jcmp!(Cmp::Lt, reg!(a), reg!(b), false, |x: f64, y: f64| x < y),
                    Op::JLe(a, b) => jcmp!(Cmp::Le, reg!(a), reg!(b), true, |x: f64, y: f64| x <= y),
                    Op::JNotLe(a, b) => jcmp!(Cmp::Le, reg!(a), reg!(b), false, |x: f64, y: f64| x <= y),
                    Op::JGt(a, b) => jcmp!(Cmp::Gt, reg!(a), reg!(b), true, |x: f64, y: f64| x > y),
                    Op::JNotGt(a, b) => jcmp!(Cmp::Gt, reg!(a), reg!(b), false, |x: f64, y: f64| x > y),
                    Op::JGe(a, b) => jcmp!(Cmp::Ge, reg!(a), reg!(b), true, |x: f64, y: f64| x >= y),
                    Op::JNotGe(a, b) => jcmp!(Cmp::Ge, reg!(a), reg!(b), false, |x: f64, y: f64| x >= y),
                    Op::JEqK(a, k) => jeq!(reg!(a), konst!(k), true),
                    Op::JNeK(a, k) => jeq!(reg!(a), konst!(k), false),
                    Op::JLtK(a, k) => jcmp!(Cmp::Lt, reg!(a), konst!(k), true, |x: f64, y: f64| x < y),
                    Op::JNotLtK(a, k) => jcmp!(Cmp::Lt, reg!(a), konst!(k), false, |x: f64, y: f64| x < y),
                    Op::JLeK(a, k) => jcmp!(Cmp::Le, reg!(a), konst!(k), true, |x: f64, y: f64| x <= y),
                    Op::JNotLeK(a, k) => jcmp!(Cmp::Le, reg!(a), konst!(k), false, |x: f64, y: f64| x <= y),
                    Op::JGtK(a, k) => jcmp!(Cmp::Gt, reg!(a), konst!(k), true, |x: f64, y: f64| x > y),
                    Op::JNotGtK(a, k) => jcmp!(Cmp::Gt, reg!(a), konst!(k), false, |x: f64, y: f64| x > y),
                    Op::JGeK(a, k) => jcmp!(Cmp::Ge, reg!(a), konst!(k), true, |x: f64, y: f64| x >= y),
                    Op::JNotGeK(a, k) => jcmp!(Cmp::Ge, reg!(a), konst!(k), false, |x: f64, y: f64| x >= y),

                    // ---- calls
                    Op::Call(a, n) => call_reg!(a, n),
                    Op::TailCall(a, n) => {
                        let n = n as usize;
                        let slot = base.add(a as usize);
                        match (*slot).as_gc::<Closure>() {
                            Some(cl) if cl.proto.arity as usize == n => {
                                if !self.open_upvals.is_empty() {
                                    self.close_upvalues(base);
                                }
                                // the callee replaces this frame: its arguments move down to our base
                                let ret = (*frame).ret;
                                std::ptr::copy(slot.add(1), base, n);
                                self.frames.pop();
                                self.push_frame(cl, base, n, ret)?;
                                reload_nosp!();
                            }
                            // not a plain closure: an ordinary call, then the `Return` that follows
                            _ => call_reg!(a, n),
                        }
                    }
                    Op::Return(a) => {
                        let r = reg!(a);
                        do_return!(r);
                    }
                    Op::ReturnNil => do_return!(Value::NIL),
                    Op::Closure(a, p) => {
                        let proto = *closure!().proto.protos.get_unchecked(p as usize);
                        let mut ups: Vec<Gc<Upvalue>> = Vec::with_capacity(proto.upvals.len());
                        for d in proto.upvals.iter() {
                            if d.is_local {
                                ups.push(self.capture_upvalue(base.add(d.index as usize)));
                            } else {
                                ups.push(*closure!().upvals.get_unchecked(d.index as usize));
                            }
                        }
                        let cl = self.heap.alloc(Closure {
                            hdr: Header::new(Kind::Closure),
                            proto,
                            upvals: ups.into_boxed_slice(),
                        });
                        reg!(a) = Value::from(cl);
                        gc_check!();
                    }

                    // ---- tables and arrays
                    Op::NewTable(a, n) => {
                        let t = self.heap.new_table(n as usize);
                        reg!(a) = Value::from(t);
                        gc_check!();
                    }
                    Op::NewArray(a, n) => {
                        let arr = self.heap.new_array(Vec::with_capacity(n as usize));
                        reg!(a) = Value::from(arr);
                        gc_check!();
                    }
                    Op::AppendN(a, b, c) => {
                        let mut arr = reg!(a).as_gc_unchecked::<Array>();
                        for i in 0..c as usize {
                            self.array_push(&mut arr, reg!(b as usize + i));
                        }
                    }
                    Op::InitField(a, k, c) => init_field!(reg!(a), konst!(k), reg!(c)),
                    Op::InitFieldW(a, b) => {
                        let k = payload!();
                        init_field!(reg!(a), konst!(k), reg!(b));
                    }
                    Op::InitIndex(a, b, c) => {
                        let mut t = reg!(a).as_gc_unchecked::<Table>();
                        let before = t.mem_size();
                        if let Err(e) = t.set(reg!(b), reg!(c)) {
                            throw!(
                                "{}",
                                if e == KeyError::Nil { "table key cannot be nil" } else { "table key cannot be NaN" }
                            );
                        }
                        self.heap.bytes += t.mem_size().saturating_sub(before);
                    }
                    Op::GetField(a, b, k) => {
                        let v = get_field!(reg!(b), konst!(k));
                        reg!(a) = v;
                    }
                    Op::GetFieldW(a, b) => {
                        let k = payload!();
                        let v = get_field!(reg!(b), konst!(k));
                        reg!(a) = v;
                    }
                    Op::SetField(a, k, c) => set_field!(reg!(a), konst!(k), reg!(c)),
                    Op::SetFieldW(a, c) => {
                        let k = payload!();
                        set_field!(reg!(a), konst!(k), reg!(c));
                    }
                    Op::GetIndex(a, b, c) => {
                        let (obj, key) = (reg!(b), reg!(c));
                        let r = if let Some(arr) = obj.as_gc::<Array>() {
                            match array_slot(&arr, key) {
                                Some(i) => *arr.items.get_unchecked(i),
                                None => protect!(self.get_index_slow(obj, key)),
                            }
                        } else if let Some(t) = obj.as_gc::<Table>() {
                            t.get(key)
                        } else {
                            protect!(self.get_index_slow(obj, key))
                        };
                        reg!(a) = r;
                        gc_check!();
                    }
                    Op::SetIndex(a, b, c) => {
                        let (obj, key, v) = (reg!(a), reg!(b), reg!(c));
                        // fast path: in-bounds array write
                        let mut done = false;
                        if let Some(mut arr) = obj.as_gc::<Array>() {
                            if let Some(i) = array_slot(&arr, key) {
                                *arr.items.get_unchecked_mut(i) = v;
                                done = true;
                            }
                        } else if let Some(mut t) = obj.as_gc::<Table>() {
                            if key.is_num() && !v.is_nil() {
                                let n = key.as_num();
                                let i = n as usize;
                                if i as f64 == n && i < t.arr.len() {
                                    *t.arr.get_unchecked_mut(i) = v;
                                    done = true;
                                }
                            }
                        }
                        if !done {
                            protect!(self.set_index_slow(obj, key, v));
                        }
                        gc_check!();
                    }
                    Op::GetMethod(a, b, k) => {
                        let obj = reg!(b);
                        let name = konst!(k).as_gc_unchecked::<Str>();
                        let f = protect!(self.get_method(obj, name));
                        reg!(a as usize + 1) = obj;
                        reg!(a) = f;
                    }
                    Op::GetMethodW(a, b) => {
                        let k = payload!();
                        let obj = reg!(b);
                        let name = konst!(k).as_gc_unchecked::<Str>();
                        let f = protect!(self.get_method(obj, name));
                        reg!(a as usize + 1) = obj;
                        reg!(a) = f;
                    }

                    // ---- structs
                    Op::Struct(a, k) => {
                        let tmpl = konst!(k).as_gc_unchecked::<StructType>();
                        let ty = self.instantiate_type(tmpl);
                        reg!(a) = Value::from(ty);
                        gc_check!();
                    }
                    Op::DefMethod(a, b) => {
                        let k = payload!();
                        let target = reg!(a);
                        match target.as_gc::<StructType>() {
                            Some(mut ty) => {
                                let name = konst!(k).as_gc_unchecked::<Str>();
                                ty.set_method(name, reg!(b));
                            }
                            None => throw!("`impl` target must be a struct, got {}", self.type_of(target)),
                        }
                    }

                    // ---- pattern matching
                    Op::IsVariant(a, b, c) => {
                        let e = payload!();
                        let name = konst!(e & 0xFFFF).as_gc_unchecked::<Str>();
                        let arity = (e >> 16) as u8;
                        let ty = if c == 255 { None } else { Some(reg!(c)) };
                        let r = protect!(self.variant_matches(reg!(b), name, arity, ty));
                        reg!(a) = Value::bool(r);
                    }
                    Op::Unpack(a, b, i) => {
                        let v = reg!(b);
                        match v.as_gc::<Instance>() {
                            Some(inst) if (i as usize) < inst.fields.len() => reg!(a) = inst.fields[i as usize],
                            _ => throw!("cannot unpack field {} of {}", i, self.type_of(v)),
                        }
                    }
                    Op::MatchFail(a) => {
                        let v = reg!(a);
                        let shown = protect!(self.repr(v));
                        throw!("no match arm matched the value {}", shown);
                    }

                    Op::Try(a) => {
                        let v = reg!(a);
                        match self.split_try(v) {
                            Some(Some(payload)) => reg!(a) = payload,
                            Some(None) => do_return!(v),
                            None => throw!("`?` expects an Option or a Result, got {}", self.type_of(v)),
                        }
                    }

                    // ---- modules
                    Op::Import(a, k) => {
                        let spec = konst!(k);
                        let m = protect!(self.import_module(spec));
                        reg!(a) = m;
                    }
                    Op::GetExport(a, b) => {
                        let k = payload!();
                        let module = reg!(b);
                        let key = konst!(k).as_gc_unchecked::<Str>();
                        let v = match module.as_gc::<Table>() {
                            Some(t) => t.get_str(key),
                            None => Value::NIL,
                        };
                        if v.is_nil() {
                            throw!("module has no export '{}'", key.as_str());
                        }
                        reg!(a) = v;
                    }

                    // ---- loops
                    Op::ForPrep(a) => {
                        let o = ext!();
                        let cur = reg!(a);
                        let end = reg!(a as usize + 1);
                        if !cur.is_num() || !end.is_num() {
                            throw!("range bounds must be numbers, got {} and {}", self.type_of(cur), self.type_of(end));
                        }
                        if cur.as_num() < end.as_num() {
                            reg!(a as usize + 2) = cur;
                        } else {
                            jump!(o);
                        }
                    }
                    Op::ForLoop(a) => {
                        let o = ext!();
                        let next = reg!(a).as_num() + 1.0;
                        if next < reg!(a as usize + 1).as_num() {
                            let v = Value::num(next);
                            reg!(a) = v;
                            reg!(a as usize + 2) = v;
                            jump!(o);
                        }
                    }
                    Op::IterNext1(a) | Op::IterNext2(a) => {
                        let o = ext!();
                        let two = matches!(op, Op::IterNext2(..));
                        let coll = reg!(a);
                        let pos = reg!(a as usize + 1).as_num() as usize;
                        let next = if let Some(arr) = coll.as_gc::<Array>() {
                            arr.items.get(pos).map(|&v| (pos + 1, Value::num(pos as f64), v))
                        } else if let Some(t) = coll.as_gc::<Table>() {
                            t.next(pos)
                        } else {
                            protect!(self.iter_next(coll, pos))
                        };
                        match next {
                            Some((np, key, val)) => {
                                reg!(a as usize + 1) = Value::num(np as f64);
                                if two {
                                    reg!(a as usize + 2) = key;
                                    reg!(a as usize + 3) = val;
                                } else {
                                    reg!(a as usize + 2) = val;
                                }
                            }
                            None => jump!(o),
                        }
                    }
                    // the second word of a double instruction is consumed by its owner
                    Op::Ext(_) => std::hint::unreachable_unchecked(),
                }
            }
        }
    }
}

// ---- free helpers -----------------------------------------------------------------------

/// How many registers (counted from the frame base) can hold live values while a frame is
/// stopped right after `op`; `None` = all of them.
///
/// Only a call tells: its arguments are the topmost temporaries (they are handed out in stack
/// order), so everything above them is dead.  Any other instruction may be writing a local
/// while temporaries above it are still waiting for their operation.
fn live_regs(op: Op) -> Option<usize> {
    match op {
        Op::Call(a, n) | Op::TailCall(a, n) => Some(a as usize + n as usize + 1),
        _ => None,
    }
}

/// In-bounds position of `key` in `arr`, or `None` (the caller then reports the precise error).
#[inline(always)]
fn array_slot(arr: &Array, key: Value) -> Option<usize> {
    if key.is_num() {
        let n = key.as_num();
        // the range check also rejects NaN, so the unchecked (single instruction) conversion is sound
        if n >= 0.0 && n < (arr.items.len() as i64) as f64 {
            let i = unsafe { n.to_int_unchecked::<usize>() };
            if i as f64 == n {
                return Some(i);
            }
        }
    }
    None
}

fn fmt_num_string(n: f64) -> String {
    let mut s = String::new();
    fmt_num(n, &mut s);
    s
}

fn arith_num(op: Arith, a: f64, b: f64) -> f64 {
    match op {
        Arith::Add => a + b,
        Arith::Sub => a - b,
        Arith::Mul => a * b,
        Arith::Div => a / b,
        Arith::Rem => rem_f64(a, b),
        Arith::Pow => a.powf(b),
        Arith::BitAnd => ((a as i64) & (b as i64)) as f64,
        Arith::BitOr => ((a as i64) | (b as i64)) as f64,
        Arith::BitXor => ((a as i64) ^ (b as i64)) as f64,
        Arith::Shl => (a as i64).wrapping_shl((b as i64 & 63) as u32) as f64,
        Arith::Shr => (a as i64).wrapping_shr((b as i64 & 63) as u32) as f64,
    }
}

/// `a % b` with C `fmod` semantics.  `fmod` is slow, so integral operands
/// (the overwhelmingly common case) take an integer path: first a 32-bit unsigned
/// one (cheap division, covers loop counters and array indices), then a 64-bit signed one.
#[inline(always)]
pub fn rem_f64(a: f64, b: f64) -> f64 {
    // `as u32` saturates and maps NaN/negatives to 0, so the round-trip check rejects them;
    // `au != 0` keeps the zero (and `-0.0`) cases on the slow path where signs are handled
    let (au, bu) = (a as u32, b as u32);
    if au != 0 && au as f64 == a && bu as f64 == b && bu != 0 {
        return (au % bu) as f64;
    }
    rem_f64_slow(a, b)
}

#[inline(never)]
fn rem_f64_slow(a: f64, b: f64) -> f64 {
    if a.abs() < 9.0e15 && b.abs() < 9.0e15 {
        let (ai, bi) = (a as i64, b as i64);
        if ai as f64 == a && bi as f64 == b && bi != 0 {
            let r = (ai % bi) as f64;
            // fmod keeps the sign of the dividend even for a zero result
            return if r == 0.0 && a.is_sign_negative() { -0.0 } else { r };
        }
    }
    a % b
}

pub fn is_ident(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if c.is_alphabetic() || c == '_' => it.all(|c| c.is_alphanumeric() || c == '_'),
        _ => false,
    }
}

pub fn char_count(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

pub fn nth_char(s: &str, n: usize) -> Option<char> {
    if s.is_ascii() {
        s.as_bytes().get(n).map(|&b| b as char)
    } else {
        s.chars().nth(n)
    }
}
