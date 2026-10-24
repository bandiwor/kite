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
use std::alloc::{alloc, dealloc, Layout};
use std::io::Write;
use std::rc::Rc;

pub const STACK_SIZE: usize = 1 << 20;
pub const MAX_FRAMES: usize = 1 << 16;
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

pub struct Frame {
    pub closure: Gc<Closure>,
    pub ip: *const Op,
    pub base: *mut Value,
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
    pub rng_state: u64,

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
        let stack = unsafe { alloc(layout) as *mut Value };
        assert!(!stack.is_null(), "cannot allocate the VM stack");
        for i in 0..STACK_SIZE {
            unsafe { stack.add(i).write(Value::NIL) };
        }
        let mut vm = Vm {
            interner: Interner::new(),
            heap: Heap::new(),
            globals: Vec::new(),
            global_info: GlobalInfo::default(),
            opts: Options::default(),
            out: Box::new(std::io::BufWriter::new(std::io::stdout())),
            string_lib: Value::NIL,
            rng_state: 0x9E37_79B9_7F4A_7C15,
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
        vm
    }

    // ---- embedding helpers ----------------------------------------------------------

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
        let ast = parser::parse(&source.text, &toks, &mut self.interner).map_err(|d| err(ErrorKind::Syntax, d, &source))?;
        let t_parse = t0.elapsed();
        if self.opts.dump_ast {
            eprintln!("{}", ast.dump(&self.interner, &source.text));
        }
        let an = analyzer::analyze(&ast, &self.interner, &mut self.global_info, self.opts.repl)
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
        self.run_main(proto).map_err(|e| self.to_kite_error(*e, &source))
    }

    fn to_kite_error(&self, e: RtError, fallback: &Rc<SourceFile>) -> KiteError {
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
            *self.sp = Value::from(closure);
            self.sp = self.sp.add(1);
            let stop = self.frames.len();
            self.push_frame(closure, 0)?;
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

    #[inline(always)]
    unsafe fn push_frame(&mut self, cl: Gc<Closure>, argc: usize) -> RtResult<()> {
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
        let base = self.sp.sub(argc);
        if self.frames.len() >= MAX_FRAMES || base.add(proto.frame_size()) > self.stack_end {
            return Err(rt_err("stack overflow"));
        }
        let locals_end = base.add(proto.nslots as usize);
        let mut p = self.sp;
        while p < locals_end {
            *p = Value::NIL;
            p = p.add(1);
        }
        self.sp = locals_end;
        self.frames.push(Frame { closure: cl, ip: proto.code.as_ptr(), base });
        Ok(())
    }

    /// Call the value sitting below `argc` arguments at the top of the stack.
    /// Returns `true` if a new bytecode frame was pushed (the caller must run
    /// it), `false` if the call completed and its result replaced the callee.
    unsafe fn call_value_at(&mut self, argc: usize) -> RtResult<bool> {
        let callee = *self.sp.sub(argc + 1);
        match callee.obj_kind() {
            Some(Kind::Closure) => {
                self.push_frame(callee.as_gc_unchecked::<Closure>(), argc)?;
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
                let args = std::slice::from_raw_parts(self.sp.sub(argc), argc);
                let r = (nat.func)(self, args)?;
                self.sp = self.sp.sub(argc + 1);
                *self.sp = r;
                self.sp = self.sp.add(1);
                Ok(false)
            }
            Some(Kind::StructType) => {
                let ty = callee.as_gc_unchecked::<StructType>();
                let nfields = ty.fields.len();
                let inst = self.heap.alloc(Instance {
                    hdr: Header::new(Kind::Instance),
                    ty,
                    fields: vec![Value::NIL; nfields].into_boxed_slice(),
                    finalized: false,
                });
                let ctor = ty.hook(Hook::New);
                let first = self.sp.sub(argc);
                if !ctor.is_nil() {
                    // [Type, a1..an] -> [new, inst, a1..an]
                    if self.sp.add(2) >= self.stack_end {
                        return Err(rt_err("stack overflow"));
                    }
                    std::ptr::copy(first, first.add(1), argc);
                    *first = Value::from(inst);
                    *first.sub(1) = ctor;
                    self.sp = self.sp.add(1);
                    return self.call_value_at(argc + 1);
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
                    inst.fields[i] = *first.add(i);
                }
                self.sp = self.sp.sub(argc + 1);
                *self.sp = Value::from(inst);
                self.sp = self.sp.add(1);
                Ok(false)
            }
            Some(Kind::Instance) => {
                let inst = callee.as_gc_unchecked::<Instance>();
                let hook = inst.ty.hook(Hook::Call);
                if hook.is_nil() {
                    return Err(rt_err(format!("struct '{}' instances are not callable (no `call` method)", inst.ty.name.as_str())));
                }
                if self.sp.add(2) >= self.stack_end {
                    return Err(rt_err("stack overflow"));
                }
                // [obj, a1..an] -> [call, obj, a1..an]
                let first = self.sp.sub(argc);
                std::ptr::copy(first.sub(1), first, argc + 1);
                *first.sub(1) = hook;
                self.sp = self.sp.add(1);
                self.call_value_at(argc + 1)
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
            if self.sp.add(args.len() + 4) >= self.stack_end {
                return Err(rt_err("stack overflow"));
            }
            *self.sp = f;
            self.sp = self.sp.add(1);
            for &a in args {
                *self.sp = a;
                self.sp = self.sp.add(1);
            }
            let stop = self.frames.len();
            self.nest += 1;
            let r = match self.call_value_at(args.len()) {
                Ok(true) => self.run(stop),
                Ok(false) => {
                    self.sp = self.sp.sub(1);
                    Ok(*self.sp)
                }
                Err(e) => Err(e),
            };
            self.nest -= 1;
            r
        }
    }

    /// Like `call_value`, but on error the VM state is restored so execution
    /// can continue (used by `pcall` and finalizers).
    pub fn call_protected(&mut self, f: Value, args: &[Value]) -> RtResult<Value> {
        let (nframes, sp, nest) = (self.frames.len(), self.sp, self.nest);
        let r = self.call_value(f, args);
        if r.is_err() {
            self.unwind(nframes, sp);
            self.nest = nest;
        }
        r
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

    fn arith_slow(&mut self, op: Arith, a: Value, b: Value) -> RtResult<Value> {
        if op == Arith::Add && (a.is_str() || b.is_str()) {
            let mut s = String::new();
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
        Err(rt_err(format!(
            "invalid operands for '{}': {} and {}",
            op.symbol(),
            self.type_of(a),
            self.type_of(b)
        )))
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

    fn get_index_slow(&mut self, obj: Value, key: Value) -> RtResult<Value> {
        match obj.obj_kind() {
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
                Err(rt_err(format!("struct '{}' cannot be indexed by {} (no `get` method)", inst.ty.name.as_str(), self.type_of(key))))
            }
            _ => Err(rt_err(format!("cannot index a value of type {}", self.type_of(obj)))),
        }
    }

    fn set_index_slow(&mut self, obj: Value, key: Value, v: Value) -> RtResult<()> {
        match obj.obj_kind() {
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
                Err(rt_err(format!("struct '{}' cannot be indexed by {} (no `set` method)", inst.ty.name.as_str(), self.type_of(key))))
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
                if v.is_nil() {
                    return Err(rt_err(format!("table has no function '{}'", name.as_str())));
                }
                Ok(v)
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
            Some(Kind::Table) => {
                if depth > 6 {
                    out.push_str("{...}");
                    return Ok(());
                }
                let t = unsafe { v.as_gc_unchecked::<Table>() };
                let only_array = t.len() == t.arr.len();
                out.push(if only_array { '[' } else { '{' });
                let mut pos = 0;
                let mut first = true;
                // snapshot entries: printing may call `str` hooks that mutate the table
                let mut items = Vec::new();
                while let Some((np, k, val)) = t.next(pos) {
                    pos = np;
                    items.push((k, val));
                }
                for (i, (k, val)) in items.into_iter().enumerate() {
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    if !(only_array && k.is_num() && k.as_num() == i as f64) {
                        match k.as_str() {
                            Some(ks) if is_ident(ks.as_str()) => out.push_str(ks.as_str()),
                            _ => self.write_value(k, out, true, depth + 1)?,
                        }
                        out.push_str(": ");
                    }
                    self.write_value(val, out, true, depth + 1)?;
                }
                out.push(if only_array { ']' } else { '}' });
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
                let _ = write!(out, "<struct {}>", t.name.as_str());
            }
            Some(Kind::Instance) => {
                let inst = unsafe { v.as_gc_unchecked::<Instance>() };
                let hook = inst.ty.hook(Hook::Str);
                if !hook.is_nil() {
                    let r = self.call_value(hook, &[v])?;
                    match r.as_str() {
                        Some(s) => out.push_str(s.as_str()),
                        None => return Err(rt_err(format!("`str` of struct '{}' must return a string", inst.ty.name.as_str()))),
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
        unsafe {
            let heap = &mut self.heap;
            let mut p = self.stack;
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
            let mut sp: *mut Value = self.sp;

            // copy the handle out of the frame first: auto-deref must not borrow through the raw pointer
            macro_rules! closure {
                () => {{
                    let c: Gc<Closure> = (*frame).closure;
                    c
                }};
            }
            let mut consts: *const Value = closure!().proto.consts.as_ptr();

            macro_rules! push {
                ($v:expr) => {{
                    let v = $v;
                    *sp = v;
                    sp = sp.add(1);
                }};
            }
            macro_rules! pop {
                () => {{
                    sp = sp.sub(1);
                    *sp
                }};
            }
            macro_rules! top {
                () => {
                    *sp.sub(1)
                };
            }
            macro_rules! peek {
                ($n:expr) => {
                    *sp.sub(1 + $n)
                };
            }
            macro_rules! sync {
                () => {{
                    (*frame).ip = ip;
                    self.sp = sp;
                }};
            }
            macro_rules! reload {
                () => {{
                    frame = self.frames.as_mut_ptr().add(self.frames.len() - 1);
                    ip = (*frame).ip;
                    base = (*frame).base;
                    consts = closure!().proto.consts.as_ptr();
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
            macro_rules! arith {
                ($kind:expr, $a:ident, $b:ident, $fast:expr) => {{
                    let $b = peek!(0);
                    let $a = peek!(1);
                    if $a.is_num() && $b.is_num() {
                        let ($a, $b) = ($a.as_num(), $b.as_num());
                        sp = sp.sub(1);
                        *sp.sub(1) = Value::num($fast);
                    } else {
                        let r = protect!(self.arith_slow($kind, $a, $b));
                        sp = sp.sub(1);
                        *sp.sub(1) = r;
                        gc_check!();
                    }
                }};
            }
            macro_rules! arith_k {
                ($kind:expr, $k:expr, $a:ident, $b:ident, $fast:expr) => {{
                    let $a = top!();
                    let kv = *consts.add($k as usize);
                    if $a.is_num() {
                        let ($a, $b) = ($a.as_num(), kv.as_num());
                        *sp.sub(1) = Value::num($fast);
                    } else {
                        let r = protect!(self.arith_slow($kind, $a, kv));
                        *sp.sub(1) = r;
                        gc_check!();
                    }
                }};
            }
            macro_rules! bitop {
                ($kind:expr, $a:ident, $b:ident, $fast:expr) => {{
                    let $b = peek!(0);
                    let $a = peek!(1);
                    if $a.is_num() && $b.is_num() {
                        let ($a, $b) = ($a.as_num() as i64, $b.as_num() as i64);
                        sp = sp.sub(1);
                        *sp.sub(1) = Value::num(($fast) as f64);
                    } else {
                        let r = protect!(self.arith_slow($kind, $a, $b));
                        sp = sp.sub(1);
                        *sp.sub(1) = r;
                    }
                }};
            }
            macro_rules! compare {
                ($kind:expr, $a:ident, $b:ident, $fast:expr) => {{
                    let $b = peek!(0);
                    let $a = peek!(1);
                    let r = if $a.is_num() && $b.is_num() {
                        let ($a, $b) = ($a.as_num(), $b.as_num());
                        $fast
                    } else {
                        protect!(self.compare_slow($kind, $a, $b))
                    };
                    sp = sp.sub(1);
                    *sp.sub(1) = Value::bool(r);
                }};
            }
            // value comparison used by the fused compare+jump ops
            macro_rules! cmp_val {
                ($kind:expr, $va:expr, $vb:expr, $fast:expr) => {{
                    let (a, b) = ($va, $vb);
                    if a.is_num() && b.is_num() {
                        let f = $fast;
                        f(a.as_num(), b.as_num())
                    } else {
                        protect!(self.compare_slow($kind, a, b))
                    }
                }};
            }
            macro_rules! jnot_stack {
                ($kind:expr, $o:expr, $fast:expr) => {{
                    let b = peek!(0);
                    let a = peek!(1);
                    let t = cmp_val!($kind, a, b, $fast);
                    sp = sp.sub(2);
                    if !t {
                        jump!($o);
                    }
                }};
            }
            macro_rules! jnot_ll {
                ($kind:expr, $a:expr, $b:expr, $o:expr, $fast:expr) => {{
                    let t = cmp_val!($kind, *base.add($a as usize), *base.add($b as usize), $fast);
                    if !t {
                        jump!($o);
                    }
                }};
            }
            macro_rules! jnot_lk {
                ($kind:expr, $a:expr, $k:expr, $o:expr, $fast:expr) => {{
                    let t = cmp_val!($kind, *base.add($a as usize), *consts.add($k as usize), $fast);
                    if !t {
                        jump!($o);
                    }
                }};
            }
            // read `obj.name` where `name` is the interned string constant `k`
            macro_rules! get_field {
                ($obj:expr, $k:expr) => {{
                    let obj: Value = $obj;
                    let key = (*consts.add($k as usize)).as_gc_unchecked::<Str>();
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
            macro_rules! enter_closure {
                ($cl:expr, $argc:expr) => {{
                    self.push_frame($cl, $argc)?;
                    sp = self.sp;
                    reload!();
                }};
            }
            macro_rules! call_generic {
                ($argc:expr) => {{
                    let argc: usize = $argc;
                    sync!();
                    let callee = *sp.sub(argc + 1);
                    if callee.is_kind(Kind::Closure) {
                        enter_closure!(callee.as_gc_unchecked::<Closure>(), argc);
                    } else {
                        match self.call_value_at(argc) {
                            Ok(true) => {
                                sp = self.sp;
                                reload!();
                            }
                            Ok(false) => {
                                sp = self.sp;
                                gc_check!();
                            }
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
                    self.frames.pop();
                    sp = base.sub(1);
                    if self.frames.len() == stop {
                        self.sp = sp;
                        return Ok(result);
                    }
                    push!(result);
                    reload!();
                }};
            }

            loop {
                let op = *ip;
                ip = ip.add(1);
                #[cfg(feature = "op-profile")]
                crate::profile::record(&op);
                match op {
                    // ---- constants & stack
                    Op::Nil => push!(Value::NIL),
                    Op::True => push!(Value::TRUE),
                    Op::False => push!(Value::FALSE),
                    Op::Const(k) => push!(*consts.add(k as usize)),
                    Op::Pop => sp = sp.sub(1),
                    Op::PopN(n) => sp = sp.sub(n as usize),
                    Op::Dup => {
                        let v = top!();
                        push!(v);
                    }
                    Op::Dup2 => {
                        let (a, b) = (peek!(1), peek!(0));
                        push!(a);
                        push!(b);
                    }

                    // ---- variables
                    Op::GetLocal(i) => push!(*base.add(i as usize)),
                    Op::SetLocal(i) => *base.add(i as usize) = top!(),
                    Op::StoreLocal(i) => {
                        sp = sp.sub(1);
                        *base.add(i as usize) = *sp;
                    }
                    Op::GetLocal2(a, b) => {
                        push!(*base.add(a as usize));
                        push!(*base.add(b as usize));
                    }
                    Op::AddLL(a, b) => {
                        let (x, y) = (*base.add(a as usize), *base.add(b as usize));
                        if x.is_num() && y.is_num() {
                            push!(Value::num(x.as_num() + y.as_num()));
                        } else {
                            let r = protect!(self.arith_slow(Arith::Add, x, y));
                            push!(r);
                            gc_check!();
                        }
                    }
                    Op::SubLL(a, b) => {
                        let (x, y) = (*base.add(a as usize), *base.add(b as usize));
                        if x.is_num() && y.is_num() {
                            push!(Value::num(x.as_num() - y.as_num()));
                        } else {
                            let r = protect!(self.arith_slow(Arith::Sub, x, y));
                            push!(r);
                        }
                    }
                    Op::MulLL(a, b) => {
                        let (x, y) = (*base.add(a as usize), *base.add(b as usize));
                        if x.is_num() && y.is_num() {
                            push!(Value::num(x.as_num() * y.as_num()));
                        } else {
                            let r = protect!(self.arith_slow(Arith::Mul, x, y));
                            push!(r);
                        }
                    }
                    Op::GetUpval(i) => {
                        let uv = *closure!().upvals.get_unchecked(i as usize);
                        push!(*uv.location);
                    }
                    Op::SetUpval(i) => {
                        let uv = *closure!().upvals.get_unchecked(i as usize);
                        let loc = uv.location;
                        *loc = top!();
                    }
                    Op::StoreUpval(i) => {
                        let uv = *closure!().upvals.get_unchecked(i as usize);
                        let loc = uv.location;
                        *loc = pop!();
                    }
                    Op::GetGlobal(s) => {
                        let v = *self.globals.get_unchecked(s as usize);
                        if v == Value::UNDEF {
                            throw!("undefined variable '{}'", self.interner.get(Sym(s)));
                        }
                        push!(v);
                    }
                    Op::SetGlobal(s) => {
                        let slot = self.globals.get_unchecked_mut(s as usize);
                        if *slot == Value::UNDEF {
                            throw!("assignment to undefined variable '{}'", self.interner.get(Sym(s)));
                        }
                        *slot = top!();
                    }
                    Op::StoreGlobal(s) => {
                        let v = pop!();
                        let slot = self.globals.get_unchecked_mut(s as usize);
                        if *slot == Value::UNDEF {
                            throw!("assignment to undefined variable '{}'", self.interner.get(Sym(s)));
                        }
                        *slot = v;
                    }
                    Op::DefGlobal(s) => {
                        let v = pop!();
                        *self.globals.get_unchecked_mut(s as usize) = v;
                    }
                    Op::CloseUpvals(n) => {
                        if !self.open_upvals.is_empty() {
                            self.close_upvalues(base.add(n as usize));
                        }
                    }

                    // ---- arithmetic
                    Op::Add => {
                        let b = peek!(0);
                        let a = peek!(1);
                        if a.is_num() && b.is_num() {
                            sp = sp.sub(1);
                            *sp.sub(1) = Value::num(a.as_num() + b.as_num());
                        } else {
                            let r = protect!(self.arith_slow(Arith::Add, a, b));
                            sp = sp.sub(1);
                            *sp.sub(1) = r;
                            gc_check!();
                        }
                    }
                    Op::Sub => arith!(Arith::Sub, a, b, a - b),
                    Op::Mul => arith!(Arith::Mul, a, b, a * b),
                    Op::Div => arith!(Arith::Div, a, b, a / b),
                    Op::Rem => arith!(Arith::Rem, a, b, rem_f64(a, b)),
                    Op::Pow => arith!(Arith::Pow, a, b, a.powf(b)),
                    Op::AddK(k) => arith_k!(Arith::Add, k, a, b, a + b),
                    Op::SubK(k) => arith_k!(Arith::Sub, k, a, b, a - b),
                    Op::MulK(k) => arith_k!(Arith::Mul, k, a, b, a * b),
                    Op::DivK(k) => arith_k!(Arith::Div, k, a, b, a / b),
                    Op::RemK(k) => arith_k!(Arith::Rem, k, a, b, rem_f64(a, b)),
                    Op::AddLocalK(s, k) => {
                        let slot = base.add(s as usize);
                        let a = *slot;
                        let kv = *consts.add(k as usize);
                        if a.is_num() {
                            *slot = Value::num(a.as_num() + kv.as_num());
                        } else {
                            let r = protect!(self.arith_slow(Arith::Add, a, kv));
                            *slot = r;
                            gc_check!();
                        }
                    }
                    Op::Neg => {
                        let a = top!();
                        if a.is_num() {
                            *sp.sub(1) = Value::num(-a.as_num());
                        } else {
                            let r = protect!(self.neg_slow(a));
                            *sp.sub(1) = r;
                        }
                    }
                    Op::Not => {
                        let a = top!();
                        *sp.sub(1) = Value::bool(a.is_falsy());
                    }
                    Op::BitNot => {
                        let a = top!();
                        if a.is_num() {
                            *sp.sub(1) = Value::num(!(a.as_num() as i64) as f64);
                        } else {
                            throw!("invalid operand for '~': {}", self.type_of(a));
                        }
                    }
                    Op::BitAnd => bitop!(Arith::BitAnd, a, b, a & b),
                    Op::BitOr => bitop!(Arith::BitOr, a, b, a | b),
                    Op::BitXor => bitop!(Arith::BitXor, a, b, a ^ b),
                    Op::Shl => bitop!(Arith::Shl, a, b, a.wrapping_shl((b & 63) as u32)),
                    Op::Shr => bitop!(Arith::Shr, a, b, a.wrapping_shr((b & 63) as u32)),

                    // ---- comparison
                    Op::Eq => {
                        let b = peek!(0);
                        let a = peek!(1);
                        let r = if a.is_num() && b.is_num() {
                            a.as_num() == b.as_num()
                        } else if !a.is_obj() && !b.is_obj() {
                            a.bits() == b.bits()
                        } else {
                            protect!(self.values_equal(a, b))
                        };
                        sp = sp.sub(1);
                        *sp.sub(1) = Value::bool(r);
                    }
                    Op::Ne => {
                        let b = peek!(0);
                        let a = peek!(1);
                        let r = if a.is_num() && b.is_num() {
                            a.as_num() != b.as_num()
                        } else if !a.is_obj() && !b.is_obj() {
                            a.bits() != b.bits()
                        } else {
                            !protect!(self.values_equal(a, b))
                        };
                        sp = sp.sub(1);
                        *sp.sub(1) = Value::bool(r);
                    }
                    Op::Lt => compare!(Cmp::Lt, a, b, a < b),
                    Op::Le => compare!(Cmp::Le, a, b, a <= b),
                    Op::Gt => compare!(Cmp::Gt, a, b, a > b),
                    Op::Ge => compare!(Cmp::Ge, a, b, a >= b),

                    // ---- jumps
                    Op::Jump(o) => jump!(o),
                    Op::JumpIfFalse(o) => {
                        sp = sp.sub(1);
                        if (*sp).is_falsy() {
                            jump!(o);
                        }
                    }
                    Op::JumpIfTrue(o) => {
                        sp = sp.sub(1);
                        if (*sp).is_truthy() {
                            jump!(o);
                        }
                    }
                    Op::And(o) => {
                        if top!().is_falsy() {
                            jump!(o);
                        } else {
                            sp = sp.sub(1);
                        }
                    }
                    Op::Or(o) => {
                        if top!().is_truthy() {
                            jump!(o);
                        } else {
                            sp = sp.sub(1);
                        }
                    }
                    Op::JNotEq(o) => jnot_stack!(Cmp::Eq, o, |a, b| a == b),
                    Op::JNotNe(o) => jnot_stack!(Cmp::Ne, o, |a, b| a != b),
                    Op::JNotLt(o) => jnot_stack!(Cmp::Lt, o, |a, b| a < b),
                    Op::JNotLe(o) => jnot_stack!(Cmp::Le, o, |a, b| a <= b),
                    Op::JNotGt(o) => jnot_stack!(Cmp::Gt, o, |a, b| a > b),
                    Op::JNotGe(o) => jnot_stack!(Cmp::Ge, o, |a, b| a >= b),
                    Op::JNotEqLL(a, b, o) => jnot_ll!(Cmp::Eq, a, b, o, |x, y| x == y),
                    Op::JNotNeLL(a, b, o) => jnot_ll!(Cmp::Ne, a, b, o, |x, y| x != y),
                    Op::JNotLtLL(a, b, o) => jnot_ll!(Cmp::Lt, a, b, o, |x, y| x < y),
                    Op::JNotLeLL(a, b, o) => jnot_ll!(Cmp::Le, a, b, o, |x, y| x <= y),
                    Op::JNotGtLL(a, b, o) => jnot_ll!(Cmp::Gt, a, b, o, |x, y| x > y),
                    Op::JNotGeLL(a, b, o) => jnot_ll!(Cmp::Ge, a, b, o, |x, y| x >= y),
                    Op::JNotEqLK(a, k, o) => jnot_lk!(Cmp::Eq, a, k, o, |x, y| x == y),
                    Op::JNotNeLK(a, k, o) => jnot_lk!(Cmp::Ne, a, k, o, |x, y| x != y),
                    Op::JNotLtLK(a, k, o) => jnot_lk!(Cmp::Lt, a, k, o, |x, y| x < y),
                    Op::JNotLeLK(a, k, o) => jnot_lk!(Cmp::Le, a, k, o, |x, y| x <= y),
                    Op::JNotGtLK(a, k, o) => jnot_lk!(Cmp::Gt, a, k, o, |x, y| x > y),
                    Op::JNotGeLK(a, k, o) => jnot_lk!(Cmp::Ge, a, k, o, |x, y| x >= y),

                    // ---- calls
                    Op::Call(argc) => call_generic!(argc as usize),
                    Op::TailCall(argc) => {
                        let argc = argc as usize;
                        let callee = *sp.sub(argc + 1);
                        match callee.as_gc::<Closure>() {
                            Some(cl) if cl.proto.arity as usize == argc => {
                                if !self.open_upvals.is_empty() {
                                    self.close_upvalues(base);
                                }
                                let dst = base.sub(1);
                                std::ptr::copy(sp.sub(argc + 1), dst, argc + 1);
                                sp = dst.add(argc + 1);
                                self.sp = sp;
                                self.frames.pop();
                                enter_closure!(cl, argc);
                            }
                            _ => call_generic!(argc),
                        }
                    }
                    Op::Return => {
                        let r = pop!();
                        do_return!(r);
                    }
                    Op::ReturnNil => do_return!(Value::NIL),
                    Op::ReturnLocal(s) => {
                        let r = *base.add(s as usize);
                        do_return!(r);
                    }
                    Op::Closure(k) => {
                        let proto = (*consts.add(k as usize)).as_gc_unchecked::<Proto>();
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
                        push!(Value::from(cl));
                        gc_check!();
                    }

                    // ---- tables
                    Op::NewTable(n) => {
                        let t = self.heap.new_table(n as usize);
                        push!(Value::from(t));
                        gc_check!();
                    }
                    Op::Append(i) => {
                        let v = pop!();
                        let mut t = top!().as_gc_unchecked::<Table>();
                        let before = t.mem_size();
                        if t.arr.len() == i as usize {
                            t.push(v);
                        } else {
                            let _ = t.set(Value::num(i as f64), v);
                        }
                        self.heap.bytes += t.mem_size().saturating_sub(before);
                    }
                    Op::InitField(k) => {
                        let v = pop!();
                        let mut t = top!().as_gc_unchecked::<Table>();
                        let key = *consts.add(k as usize);
                        let before = t.mem_size();
                        let _ = t.set(key, v);
                        self.heap.bytes += t.mem_size().saturating_sub(before);
                    }
                    Op::InitIndex => {
                        let v = pop!();
                        let k = pop!();
                        let mut t = top!().as_gc_unchecked::<Table>();
                        let before = t.mem_size();
                        if let Err(e) = t.set(k, v) {
                            throw!("{}", if e == KeyError::Nil { "table key cannot be nil" } else { "table key cannot be NaN" });
                        }
                        self.heap.bytes += t.mem_size().saturating_sub(before);
                    }
                    Op::GetField(k) => {
                        let v = get_field!(top!(), k);
                        *sp.sub(1) = v;
                    }
                    Op::GetLocalField(s, k) => {
                        let v = get_field!(*base.add(s as usize), k);
                        push!(v);
                    }
                    Op::SetField(k) | Op::StoreField(k) => {
                        let store = matches!(op, Op::StoreField(_));
                        let v = peek!(0);
                        let obj = peek!(1);
                        let key = (*consts.add(k as usize)).as_gc_unchecked::<Str>();
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
                        if store {
                            sp = sp.sub(2);
                        } else {
                            sp = sp.sub(1);
                            *sp.sub(1) = v;
                        }
                        gc_check!();
                    }
                    Op::GetIndex => {
                        let key = peek!(0);
                        let obj = peek!(1);
                        let r = if let Some(t) = obj.as_gc::<Table>() {
                            t.get(key)
                        } else {
                            protect!(self.get_index_slow(obj, key))
                        };
                        sp = sp.sub(1);
                        *sp.sub(1) = r;
                        gc_check!();
                    }
                    Op::SetIndex | Op::StoreIndex => {
                        let store = matches!(op, Op::StoreIndex);
                        let v = peek!(0);
                        let key = peek!(1);
                        let obj = peek!(2);
                        // fast path: in-bounds array write
                        let mut done = false;
                        if let Some(mut t) = obj.as_gc::<Table>() {
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
                        if store {
                            sp = sp.sub(3);
                        } else {
                            sp = sp.sub(2);
                            *sp.sub(1) = v;
                        }
                        gc_check!();
                    }
                    Op::GetMethod(k) => {
                        let obj = top!();
                        let name = (*consts.add(k as usize)).as_gc_unchecked::<Str>();
                        let f = protect!(self.get_method(obj, name));
                        *sp.sub(1) = f;
                        push!(obj);
                    }

                    // ---- structs
                    Op::Struct(k) => {
                        let tmpl = (*consts.add(k as usize)).as_gc_unchecked::<StructType>();
                        let ty = self.heap.alloc(StructType {
                            hdr: Header::new(Kind::StructType),
                            name: tmpl.name,
                            fields: tmpl.fields.clone(),
                            methods: Vec::new(),
                            hooks: [Value::NIL; HOOK_COUNT],
                        });
                        push!(Value::from(ty));
                        gc_check!();
                    }
                    Op::DefMethod(k) => {
                        let target = peek!(1);
                        match target.as_gc::<StructType>() {
                            Some(mut ty) => {
                                let f = pop!();
                                let name = (*consts.add(k as usize)).as_gc_unchecked::<Str>();
                                ty.set_method(name, f);
                            }
                            None => throw!("`impl` target must be a struct, got {}", self.type_of(target)),
                        }
                    }

                    // ---- loops
                    Op::ForPrep(s, o) => {
                        let cur = *base.add(s as usize);
                        let end = *base.add(s as usize + 1);
                        if !cur.is_num() || !end.is_num() {
                            throw!("range bounds must be numbers, got {} and {}", self.type_of(cur), self.type_of(end));
                        }
                        if cur.as_num() < end.as_num() {
                            *base.add(s as usize + 2) = cur;
                        } else {
                            jump!(o);
                        }
                    }
                    Op::ForLoop(s, o) => {
                        let next = (*base.add(s as usize)).as_num() + 1.0;
                        if next < (*base.add(s as usize + 1)).as_num() {
                            let v = Value::num(next);
                            *base.add(s as usize) = v;
                            *base.add(s as usize + 2) = v;
                            jump!(o);
                        }
                    }
                    Op::IterNext1(s, o) | Op::IterNext2(s, o) => {
                        let two = matches!(op, Op::IterNext2(..));
                        let coll = *base.add(s as usize);
                        let pos = (*base.add(s as usize + 1)).as_num() as usize;
                        let next = match coll.as_gc::<Table>() {
                            Some(t) => t.next(pos),
                            None => protect!(self.iter_next(coll, pos)),
                        };
                        match next {
                            Some((np, key, val)) => {
                                *base.add(s as usize + 1) = Value::num(np as f64);
                                if two {
                                    *base.add(s as usize + 2) = key;
                                    *base.add(s as usize + 3) = val;
                                } else {
                                    *base.add(s as usize + 2) = val;
                                }
                            }
                            None => jump!(o),
                        }
                    }
                }
            }
        }
    }
}

// ---- free helpers -----------------------------------------------------------------------

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
/// (the overwhelmingly common case) take an integer path.
#[inline(always)]
pub fn rem_f64(a: f64, b: f64) -> f64 {
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
