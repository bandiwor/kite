//! NaN-boxed values.
//!
//! Every value is 8 bytes.  Anything that is not a quiet NaN with bit 50 set is
//! a plain `f64`.  Otherwise the sign bit distinguishes heap objects (pointer
//! in the low 48 bits) from the immediate singletons (`nil`, booleans and two
//! VM-internal markers) which are told apart by their low tag bits.
//! Hardware-generated NaNs (`0x7FF8…`/`0xFFF8…`) never have bit 50 set, so
//! arithmetic can never forge an object.

use crate::object::{Gc, GcObj, Header, Kind, Str};

const SIGN: u64 = 1 << 63;
const QNAN: u64 = 0x7FFC_0000_0000_0000;
const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

const TAG_NIL: u64 = 1;
const TAG_FALSE: u64 = 2;
const TAG_TRUE: u64 = 3;
const TAG_UNDEF: u64 = 4;
const TAG_TOMB: u64 = 5;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Value(u64);

const _: () = assert!(std::mem::size_of::<Value>() == 8);

impl Value {
    pub const NIL: Value = Value(QNAN | TAG_NIL);
    pub const FALSE: Value = Value(QNAN | TAG_FALSE);
    pub const TRUE: Value = Value(QNAN | TAG_TRUE);
    /// Marks a global that was never defined.
    pub const UNDEF: Value = Value(QNAN | TAG_UNDEF);
    /// Deleted hash-table entry.
    pub const TOMB: Value = Value(QNAN | TAG_TOMB);

    #[inline(always)]
    pub fn num(n: f64) -> Value {
        Value(n.to_bits())
    }
    #[inline(always)]
    pub fn bool(b: bool) -> Value {
        if b {
            Value::TRUE
        } else {
            Value::FALSE
        }
    }
    #[inline(always)]
    pub fn bits(self) -> u64 {
        self.0
    }
    #[inline(always)]
    pub fn is_num(self) -> bool {
        (self.0 & QNAN) != QNAN
    }
    /// Caller must have checked `is_num`.
    #[inline(always)]
    pub fn as_num(self) -> f64 {
        f64::from_bits(self.0)
    }
    #[inline(always)]
    pub fn is_nil(self) -> bool {
        self.0 == Value::NIL.0
    }
    #[inline(always)]
    pub fn is_bool(self) -> bool {
        (self.0 | 1) == Value::TRUE.0
    }
    #[inline(always)]
    pub fn is_falsy(self) -> bool {
        self.0 == Value::NIL.0 || self.0 == Value::FALSE.0
    }
    #[inline(always)]
    pub fn is_truthy(self) -> bool {
        !self.is_falsy()
    }
    #[inline(always)]
    pub fn is_obj(self) -> bool {
        (self.0 & (QNAN | SIGN)) == (QNAN | SIGN)
    }
    #[inline(always)]
    pub fn from_header(p: *mut Header) -> Value {
        debug_assert!((p as u64) & !PTR_MASK == 0, "pointer does not fit in 48 bits");
        Value(SIGN | QNAN | p as u64)
    }
    /// Caller must have checked `is_obj`.
    #[inline(always)]
    pub fn as_header(self) -> *mut Header {
        (self.0 & PTR_MASK) as *mut Header
    }
    #[inline(always)]
    pub fn obj_kind(self) -> Option<Kind> {
        if self.is_obj() {
            Some(unsafe { (*self.as_header()).kind })
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn is_kind(self, k: Kind) -> bool {
        self.is_obj() && unsafe { (*self.as_header()).kind == k }
    }
    #[inline(always)]
    pub fn is_str(self) -> bool {
        self.is_kind(Kind::Str)
    }
    #[inline(always)]
    pub fn as_gc<T: GcObj>(self) -> Option<Gc<T>> {
        if self.is_kind(T::KIND) {
            Some(unsafe { Gc::from_raw(self.as_header() as *mut T) })
        } else {
            None
        }
    }
    /// # Safety
    /// The value must hold an object of type `T`.
    #[inline(always)]
    pub unsafe fn as_gc_unchecked<T: GcObj>(self) -> Gc<T> {
        Gc::from_raw(self.as_header() as *mut T)
    }
    pub fn as_str(self) -> Option<Gc<Str>> {
        self.as_gc::<Str>()
    }

    pub fn type_name(self) -> &'static str {
        if self.is_num() {
            return "number";
        }
        if self.is_nil() {
            return "nil";
        }
        if self.is_bool() {
            return "bool";
        }
        match self.obj_kind() {
            Some(Kind::Str) => "string",
            Some(Kind::Table) => "table",
            Some(Kind::Closure) | Some(Kind::Native) | Some(Kind::Proto) | Some(Kind::Variant) => "function",
            Some(Kind::StructType) => {
                if unsafe { self.as_gc_unchecked::<crate::object::StructType>() }.is_union() {
                    "union"
                } else {
                    "struct"
                }
            }
            Some(Kind::Instance) => "instance",
            Some(Kind::Upvalue) => "upvalue",
            None => "undefined",
        }
    }
}

impl<T: GcObj> From<Gc<T>> for Value {
    #[inline(always)]
    fn from(g: Gc<T>) -> Value {
        Value::from_header(g.as_ptr() as *mut Header)
    }
}

impl From<f64> for Value {
    #[inline(always)]
    fn from(n: f64) -> Value {
        Value::num(n)
    }
}

impl From<bool> for Value {
    #[inline(always)]
    fn from(b: bool) -> Value {
        Value::bool(b)
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_num() {
            write!(f, "{}", self.as_num())
        } else if self.is_nil() {
            write!(f, "nil")
        } else if self.is_bool() {
            write!(f, "{}", *self == Value::TRUE)
        } else if self.is_obj() {
            write!(f, "<{} {:p}>", self.type_name(), self.as_header())
        } else {
            write!(f, "<marker {:#x}>", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediates() {
        assert!(Value::num(1.5).is_num());
        assert!(Value::num(f64::NAN).is_num());
        assert!(Value::num(-f64::NAN).is_num());
        assert!(Value::num(f64::INFINITY).is_num());
        assert!(!Value::NIL.is_num());
        assert!(!Value::TRUE.is_num() && Value::TRUE.is_bool());
        assert!(Value::FALSE.is_falsy() && Value::NIL.is_falsy());
        assert!(Value::num(0.0).is_truthy());
        assert!(!Value::NIL.is_obj() && !Value::UNDEF.is_obj() && !Value::num(1.0).is_obj());
        let nan = Value::num(0.0 / f64::from_bits(0));
        assert!(nan.is_num() && !nan.is_obj());
    }
}
