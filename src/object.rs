//! Heap object layouts.  Every object is `#[repr(C)]` and starts with a
//! `Header`, which lets the collector treat any object pointer as a
//! `*mut Header` and dispatch on `kind`.

use crate::analyzer::UpvalDesc;
use crate::op::Op;
use crate::source::SourceFile;
use crate::table::Table;
use crate::value::Value;
use crate::vm::{RtResult, Vm};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::rc::Rc;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Str,
    Table,
    Proto,
    Closure,
    Upvalue,
    Native,
    StructType,
    Instance,
    /// Callable that builds instances of one union variant.
    Variant,
}

#[repr(C)]
pub struct Header {
    pub kind: Kind,
    pub marked: bool,
    pub next: *mut Header,
}

impl Header {
    pub fn new(kind: Kind) -> Header {
        Header { kind, marked: false, next: std::ptr::null_mut() }
    }
}

pub trait GcObj {
    const KIND: Kind;
    /// Approximate number of bytes this object keeps alive (drives GC pacing).
    fn heap_size(&self) -> usize;
}

/// Copyable handle to a garbage-collected object.  The collector is the only
/// thing that frees objects, so a `Gc` is valid as long as it is reachable
/// from a root.  Single-threaded by construction (`!Send`).
pub struct Gc<T>(NonNull<T>);

impl<T> Clone for Gc<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Gc<T> {}
impl<T> PartialEq for Gc<T> {
    #[inline(always)]
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<T> Eq for Gc<T> {}

impl<T> Gc<T> {
    /// # Safety
    /// `p` must point to a live heap object of type `T`.
    #[inline(always)]
    pub unsafe fn from_raw(p: *mut T) -> Gc<T> {
        Gc(NonNull::new_unchecked(p))
    }
    #[inline(always)]
    pub fn as_ptr(self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T> Deref for Gc<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        unsafe { &*self.0.as_ptr() }
    }
}
impl<T> DerefMut for Gc<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0.as_ptr() }
    }
}

impl<T> std::fmt::Debug for Gc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gc({:p})", self.0.as_ptr())
    }
}

// ---- strings -----------------------------------------------------------------

#[repr(C)]
pub struct Str {
    pub hdr: Header,
    pub hash: u32,
    /// Interned strings are unique per content, so equality is pointer equality.
    pub interned: bool,
    pub data: Box<str>,
}

impl Str {
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        &self.data
    }
}

impl GcObj for Str {
    const KIND: Kind = Kind::Str;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Str>() + self.data.len()
    }
}

/// String equality (pointer fast path, then content).
#[inline(always)]
pub fn str_eq(a: Gc<Str>, b: Gc<Str>) -> bool {
    a == b || (!(a.interned && b.interned) && a.hash == b.hash && a.data == b.data)
}

// ---- functions ---------------------------------------------------------------

#[repr(C)]
pub struct Proto {
    pub hdr: Header,
    pub name: Rc<str>,
    pub arity: u8,
    /// Params + locals.
    pub nslots: u16,
    /// Maximum operand stack depth above the locals.
    pub max_stack: u16,
    pub upvals: Vec<UpvalDesc>,
    pub code: Vec<Op>,
    pub consts: Vec<Value>,
    /// Source offset per instruction (turned into line numbers lazily).
    pub offsets: Vec<u32>,
    pub source: Rc<SourceFile>,
}

impl Proto {
    #[inline(always)]
    pub fn frame_size(&self) -> usize {
        self.nslots as usize + self.max_stack as usize
    }
}

impl GcObj for Proto {
    const KIND: Kind = Kind::Proto;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Proto>() + self.code.len() * 8 + self.consts.len() * 8 + self.offsets.len() * 4
    }
}

#[repr(C)]
pub struct Closure {
    pub hdr: Header,
    pub proto: Gc<Proto>,
    pub upvals: Box<[Gc<Upvalue>]>,
}

impl GcObj for Closure {
    const KIND: Kind = Kind::Closure;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Closure>() + self.upvals.len() * 8
    }
}

/// A captured variable.  While open, `location` points into the VM stack;
/// closing copies the value into `closed` and re-points `location` there.
#[repr(C)]
pub struct Upvalue {
    pub hdr: Header,
    pub location: *mut Value,
    pub closed: Value,
}

impl GcObj for Upvalue {
    const KIND: Kind = Kind::Upvalue;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Upvalue>()
    }
}

pub type NativeFn = fn(&mut Vm, &[Value]) -> RtResult<Value>;

#[repr(C)]
pub struct Native {
    pub hdr: Header,
    pub name: &'static str,
    /// Exact argument count, or -1 for variadic.
    pub arity: i8,
    pub func: NativeFn,
}

impl GcObj for Native {
    const KIND: Kind = Kind::Native;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Native>()
    }
}

// ---- structs -----------------------------------------------------------------

/// Special methods looked up by the VM (operator overloads and life-cycle).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hook {
    New,
    Destroy,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Neg,
    Eq,
    Lt,
    Le,
    Call,
    Str,
    Len,
    Get,
    Set,
}

pub const HOOK_COUNT: usize = 17;

impl Hook {
    pub fn from_name(s: &str) -> Option<Hook> {
        Some(match s {
            "new" => Hook::New,
            "destroy" => Hook::Destroy,
            "add" => Hook::Add,
            "sub" => Hook::Sub,
            "mul" => Hook::Mul,
            "div" => Hook::Div,
            "rem" => Hook::Rem,
            "pow" => Hook::Pow,
            "neg" => Hook::Neg,
            "eq" => Hook::Eq,
            "lt" => Hook::Lt,
            "le" => Hook::Le,
            "call" => Hook::Call,
            "str" => Hook::Str,
            "len" => Hook::Len,
            "get" => Hook::Get,
            "set" => Hook::Set,
            _ => return None,
        })
    }
}

/// One variant of a `union` type.
pub struct Variant {
    pub name: Gc<Str>,
    /// Payload field names; their count is the arity.
    pub fields: Vec<Gc<Str>>,
    /// What `Type.Variant` evaluates to: the shared instance for a unit variant,
    /// a `VariantCtor` otherwise (`NIL` in the compile-time template).
    pub value: Value,
}

/// `struct` and `union` types share this object: a union has `variants` and no `fields`.
#[repr(C)]
pub struct StructType {
    pub hdr: Header,
    pub name: Gc<Str>,
    pub fields: Vec<Gc<Str>>,
    pub variants: Vec<Variant>,
    /// Functions defined in `impl` blocks, in definition order.
    pub methods: Vec<(Gc<Str>, Value)>,
    /// Quick access to the special methods (`Value::NIL` when absent).
    pub hooks: [Value; HOOK_COUNT],
}

impl StructType {
    pub fn is_union(&self) -> bool {
        !self.variants.is_empty()
    }

    pub fn variant_index(&self, name: Gc<Str>) -> Option<usize> {
        self.variants.iter().position(|v| str_eq(v.name, name))
    }

    pub fn field_index(&self, name: Gc<Str>) -> Option<usize> {
        if let Some(i) = self.fields.iter().position(|&f| f == name) {
            return Some(i);
        }
        if name.interned {
            return None;
        }
        self.fields.iter().position(|&f| str_eq(f, name))
    }

    pub fn method(&self, name: Gc<Str>) -> Option<Value> {
        self.methods.iter().find(|(n, _)| str_eq(*n, name)).map(|&(_, v)| v)
    }

    pub fn set_method(&mut self, name: Gc<Str>, f: Value) {
        if let Some(h) = Hook::from_name(name.as_str()) {
            self.hooks[h as usize] = f;
        }
        if let Some(slot) = self.methods.iter_mut().find(|(n, _)| str_eq(*n, name)) {
            slot.1 = f;
        } else {
            self.methods.push((name, f));
        }
    }

    #[inline(always)]
    pub fn hook(&self, h: Hook) -> Value {
        self.hooks[h as usize]
    }
}

impl GcObj for StructType {
    const KIND: Kind = Kind::StructType;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<StructType>()
            + self.fields.len() * 8
            + self.methods.len() * 24
            + self.variants.iter().map(|v| std::mem::size_of::<Variant>() + v.fields.len() * 8).sum::<usize>()
    }
}

#[repr(C)]
pub struct VariantCtor {
    pub hdr: Header,
    pub ty: Gc<StructType>,
    pub tag: u16,
}

impl GcObj for VariantCtor {
    const KIND: Kind = Kind::Variant;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<VariantCtor>()
    }
}

#[repr(C)]
pub struct Instance {
    pub hdr: Header,
    pub ty: Gc<StructType>,
    pub fields: Box<[Value]>,
    /// Which variant, for union instances (0 for structs).
    pub tag: u16,
    /// `destroy` already ran (or is scheduled).
    pub finalized: bool,
}

impl GcObj for Instance {
    const KIND: Kind = Kind::Instance;
    fn heap_size(&self) -> usize {
        std::mem::size_of::<Instance>() + self.fields.len() * 8
    }
}

impl GcObj for Table {
    const KIND: Kind = Kind::Table;
    fn heap_size(&self) -> usize {
        self.mem_size()
    }
}
