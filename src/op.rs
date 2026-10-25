//! Bytecode.
//!
//! An instruction is a Rust `enum` that is exactly **4 bytes**: a tag byte plus up
//! to three bytes of operands (`u8`, `u16`, `(u8, u8)`, `(u8, u16)`, or the 24-bit
//! `U24`/`I24`).  The VM dispatches with a plain `match` (jump table), the
//! compiler/peephole can pattern-match on slices of ops, and the disassembler comes
//! for free via `Debug`.
//!
//! Instructions that need more than three operand bytes are **double**: the
//! first word holds the opcode and its operands, the following word is an
//! `Op::Ext(I24)` carrying the branch offset (see [`Op::is_wide`]).  These are the
//! fused compare-and-branch ops and the loop ops.
//!
//! Branch offsets are counted in words and are relative to the first word *after*
//! the instruction (after the `Ext` word for double instructions).
//!
//! Adding an opcode = add a variant, its `stack_effect`, and a VM arm.

use std::fmt;

/// Unsigned 24-bit operand (constant indices, global slots, ...).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, packed)]
pub struct U24 {
    lo: u16,
    hi: u8,
}

impl U24 {
    pub const MAX: u32 = 0xFF_FFFF;
    pub const ZERO: U24 = U24 { lo: 0, hi: 0 };

    #[inline(always)]
    pub fn new(v: u32) -> Option<U24> {
        if v <= Self::MAX {
            Some(U24 { lo: v as u16, hi: (v >> 16) as u8 })
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn get(self) -> u32 {
        self.lo as u32 | (self.hi as u32) << 16
    }
}

impl fmt::Debug for U24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

/// Signed 24-bit operand (branch offsets).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, packed)]
pub struct I24 {
    lo: u16,
    hi: i8,
}

impl I24 {
    pub const MIN: i32 = -(1 << 23);
    pub const MAX: i32 = (1 << 23) - 1;
    pub const ZERO: I24 = I24 { lo: 0, hi: 0 };

    #[inline(always)]
    pub fn new(v: i32) -> Option<I24> {
        if (Self::MIN..=Self::MAX).contains(&v) {
            Some(I24 { lo: v as u16, hi: (v >> 16) as i8 })
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn get(self) -> i32 {
        (self.hi as i32) << 16 | self.lo as i32
    }
}

impl fmt::Debug for I24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Op {
    // ---- constants & stack ----
    Nil,
    True,
    False,
    Const(U24),
    Pop,
    PopN(u16),
    /// Duplicate the top value.
    Dup,
    /// Duplicate the top two values.
    Dup2,

    // ---- variables ----
    GetLocal(u16),
    /// Assign, keep the value on the stack (assignment is an expression).
    SetLocal(u16),
    /// Assign and pop (produced by the compiler for declarations and by the peephole).
    StoreLocal(u16),
    /// Push two locals (slots below 256).
    GetLocal2(u8, u8),
    /// Push `local[a] <op> local[b]` (slots below 256).
    AddLL(u8, u8),
    SubLL(u8, u8),
    MulLL(u8, u8),
    GetUpval(u16),
    SetUpval(u16),
    StoreUpval(u16),
    GetGlobal(U24),
    SetGlobal(U24),
    StoreGlobal(U24),
    /// Define (or redefine) a global; pops.
    DefGlobal(U24),
    /// Close open upvalues for slots >= n of the current frame.
    CloseUpvals(u16),

    // ---- arithmetic ----
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Neg,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    /// `top = top <op> consts[k]` where the constant is a number.
    AddK(U24),
    SubK(U24),
    MulK(U24),
    DivK(U24),
    RemK(U24),
    /// `local[s] += consts[k]` (statement form, pushes nothing); slot < 256, const < 65536.
    AddLocalK(u8, u16),

    // ---- comparison ----
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // ---- jumps ----
    Jump(I24),
    /// Pop; jump if falsy.
    JumpIfFalse(I24),
    /// Pop; jump if truthy.
    JumpIfTrue(I24),
    /// `a && b`: if top is falsy jump (keeping it), else pop and fall through.
    And(I24),
    /// `a || b`: if top is truthy jump (keeping it), else pop and fall through.
    Or(I24),

    // ---- fused compare + "jump if the comparison is false" (peephole) ----
    JNotEq(I24),
    JNotNe(I24),
    JNotLt(I24),
    JNotLe(I24),
    JNotGt(I24),
    JNotGe(I24),
    // double: local OP local, offset in the following `Ext`
    JNotEqLL(u8, u8),
    JNotNeLL(u8, u8),
    JNotLtLL(u8, u8),
    JNotLeLL(u8, u8),
    JNotGtLL(u8, u8),
    JNotGeLL(u8, u8),
    // double: local OP constant (local < 256, const < 65536)
    JNotEqLK(u8, u16),
    JNotNeLK(u8, u16),
    JNotLtLK(u8, u16),
    JNotLeLK(u8, u16),
    JNotGtLK(u8, u16),
    JNotGeLK(u8, u16),

    // ---- calls ----
    /// Call the value below the `n` arguments.
    Call(u8),
    /// Like `Call`, but reuses the current frame when the callee is a closure.
    /// Always followed by a `Return` (used when the callee is not a closure).
    TailCall(u8),
    Return,
    ReturnNil,
    /// `GetLocal; Return`
    ReturnLocal(u16),
    /// Create a closure from the prototype in `consts[k]`.
    Closure(U24),

    // ---- tables, fields, structs ----
    /// New empty table (`n` = expected number of entries).
    NewTable(u16),
    /// New empty array with room for `n` items.
    NewArray(u16),
    /// `[arr, v] -> [arr]`: array literal element (any value, `nil` included).
    Append,
    /// `[t, v] -> [t]`  (table literal `name: v`)
    InitField(U24),
    /// `[t, k, v] -> [t]`  (table literal `[k]: v`)
    InitIndex,
    GetField(U24),
    /// `GetLocal; GetField`: push `local[s].name` (slot < 256, const < 65536).
    GetLocalField(u8, u16),
    /// `[obj, v] -> [v]`
    SetField(U24),
    /// `[obj, v] -> []`
    StoreField(U24),
    GetIndex,
    /// `[obj, k, v] -> [v]`
    SetIndex,
    /// `[obj, k, v] -> []`
    StoreIndex,
    /// `[obj] -> [function, obj]`: resolve `obj:name(...)`.
    GetMethod(U24),
    /// Instantiate the struct type template in `consts[k]`.
    Struct(U24),
    /// `[type, closure] -> [type]`: register an `impl` function under `consts[k]`.
    DefMethod(U24),

    // ---- pattern matching ----
    /// `[v] -> [bool]`: is `v` an instance of the variant (or struct) named `consts[k]`?
    /// Operands: `arity` (number of sub-patterns, or 255 when the pattern has no parentheses), name constant.
    IsVariant(u8, u16),
    /// `[v, type] -> [bool]`: like `IsVariant`, and `v` must also belong to `type`.
    IsVariantQ(u8, u16),
    /// `[inst] -> [field]`: payload field `i` of a union variant / struct instance.
    Unpack(u8),
    /// No arm matched the value in local `slot`: raise an error (pushes, for the stack model).
    MatchFail(u16),

    /// `[v] -> [payload]` for `Ok(payload)` / `Some(payload)`; for `Err(..)` / `None` the current
    /// function returns `v` instead.  Anything else is an error.
    Try,

    // ---- modules ----
    /// Load (or fetch from the cache) the module whose path is `consts[k]`; pushes its exports table.
    Import(U24),
    /// `[module] -> [value]`: read an export, failing if the module does not have it.
    GetExport(U24),

    // ---- loops (double) ----
    /// Range loop setup: locals `s`,`s+1` hold start/end; if start >= end jump.
    /// Otherwise the loop variable (`s+2`) is set to start.
    ForPrep(u16),
    /// Advance the range cursor; if still in range set the variable and jump back.
    ForLoop(u16),
    /// Table iteration, one variable (value).  Locals `s`,`s+1` = table, cursor.
    IterNext1(u16),
    /// Table iteration, two variables (key, value).
    IterNext2(u16),

    /// Second word of a double instruction: its branch offset.  Never executed.
    Ext(I24),
}

const _: () = assert!(std::mem::size_of::<Op>() == 4);

impl Op {
    /// Net effect on the operand stack when execution falls through to the next op.
    pub fn stack_effect(&self) -> i32 {
        use Op::*;
        match *self {
            Nil | True | False | Const(_) | Dup | GetLocal(_) | GetUpval(_) | GetGlobal(_) | Closure(_)
            | NewTable(_) | NewArray(_) | Struct(_) | GetMethod(_) | AddLL(..) | SubLL(..) | MulLL(..)
            | GetLocalField(..) | Import(_) | MatchFail(_) => 1,
            Dup2 | GetLocal2(..) => 2,
            Pop | StoreLocal(_) | StoreUpval(_) | StoreGlobal(_) | DefGlobal(_) | Append | InitField(_)
            | SetField(_) | GetIndex | DefMethod(_) | Return => -1,
            PopN(n) => -(n as i32),
            SetLocal(_) | SetUpval(_) | SetGlobal(_) | CloseUpvals(_) | Neg | Not | BitNot | AddK(_) | SubK(_)
            | MulK(_) | DivK(_) | RemK(_) | AddLocalK(..) | Jump(_) | GetField(_) | ForPrep(_) | ForLoop(_)
            | IterNext1(_) | IterNext2(_) | ReturnNil | ReturnLocal(_) | GetExport(_) | IsVariant(..) | Unpack(_)
            | Try | JNotEqLL(..) | JNotNeLL(..) | JNotLtLL(..) | JNotLeLL(..) | JNotGtLL(..) | JNotGeLL(..)
            | JNotEqLK(..) | JNotNeLK(..) | JNotLtLK(..) | JNotLeLK(..) | JNotGtLK(..) | JNotGeLK(..) | Ext(_) => 0,
            Add | Sub | Mul | Div | Rem | Pow | BitAnd | BitOr | BitXor | Shl | Shr | Eq | Ne | Lt | Le | Gt | Ge
            | JumpIfFalse(_) | JumpIfTrue(_) | And(_) | Or(_) => -1,
            JNotEq(_) | JNotNe(_) | JNotLt(_) | JNotLe(_) | JNotGt(_) | JNotGe(_) | StoreField(_) | InitIndex
            | SetIndex => -2,
            IsVariantQ(..) => -1,
            StoreIndex => -3,
            Call(n) | TailCall(n) => -(n as i32),
        }
    }

    /// Double instruction: the word after it is an `Ext` with the branch offset.
    #[inline]
    pub fn is_wide(&self) -> bool {
        use Op::*;
        matches!(
            self,
            JNotEqLL(..)
                | JNotNeLL(..)
                | JNotLtLL(..)
                | JNotLeLL(..)
                | JNotGtLL(..)
                | JNotGeLL(..)
                | JNotEqLK(..)
                | JNotNeLK(..)
                | JNotLtLK(..)
                | JNotLeLK(..)
                | JNotGtLK(..)
                | JNotGeLK(..)
                | ForPrep(_)
                | ForLoop(_)
                | IterNext1(_)
                | IterNext2(_)
        )
    }

    /// Number of words the instruction occupies.
    #[inline]
    pub fn width(&self) -> usize {
        1 + self.is_wide() as usize
    }

    /// Offset of a single-word branch (`Jump`, `JumpIfFalse`, ...).  Double
    /// instructions keep their offset in the `Ext` word instead.
    pub fn jump_offset(&self) -> Option<i32> {
        use Op::*;
        Some(match *self {
            Jump(o) | JumpIfFalse(o) | JumpIfTrue(o) | And(o) | Or(o) | JNotEq(o) | JNotNe(o) | JNotLt(o)
            | JNotLe(o) | JNotGt(o) | JNotGe(o) => o.get(),
            _ => return None,
        })
    }

    /// Returns the single-word branch with a new offset, or `None` if it does not fit
    /// (or the op is not a single-word branch).
    pub fn with_jump_offset(self, off: i32) -> Option<Op> {
        use Op::*;
        let o = I24::new(off)?;
        Some(match self {
            Jump(_) => Jump(o),
            JumpIfFalse(_) => JumpIfFalse(o),
            JumpIfTrue(_) => JumpIfTrue(o),
            And(_) => And(o),
            Or(_) => Or(o),
            JNotEq(_) => JNotEq(o),
            JNotNe(_) => JNotNe(o),
            JNotLt(_) => JNotLt(o),
            JNotLe(_) => JNotLe(o),
            JNotGt(_) => JNotGt(o),
            JNotGe(_) => JNotGe(o),
            _ => return None,
        })
    }

    /// Does control transfer somewhere (single-word or double instruction)?
    #[inline]
    pub fn is_branch(&self) -> bool {
        self.is_wide() || self.jump_offset().is_some()
    }

    /// Value of an `Ext` word.
    #[inline(always)]
    pub fn ext(self) -> i32 {
        match self {
            Op::Ext(o) => o.get(),
            _ => {
                debug_assert!(false, "expected Ext, found {:?}", self);
                // SAFETY: callers only read the word after a double instruction
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    /// The unsigned 24-bit operand of a `U24` variant, read straight from the
    /// word (`#[repr(u8)]` puts the tag in byte 0 and the three operand bytes
    /// right behind it).  Only valid for variants whose whole payload is a `U24`.
    #[inline(always)]
    pub fn imm24(self) -> u32 {
        debug_assert!(self.has_24bit_payload());
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `Op` is 4 bytes and these variants have no padding bytes
            unsafe { std::mem::transmute::<Op, u32>(self) >> 8 }
        }
        #[cfg(not(target_endian = "little"))]
        {
            self.imm24_slow()
        }
    }

    /// The signed 24-bit operand of an `I24` variant (`Jump`, `Ext`, ...); see [`Op::imm24`].
    #[inline(always)]
    pub fn simm24(self) -> i32 {
        debug_assert!(self.has_24bit_payload());
        #[cfg(target_endian = "little")]
        {
            // SAFETY: as for `imm24`
            unsafe { std::mem::transmute::<Op, i32>(self) >> 8 }
        }
        #[cfg(not(target_endian = "little"))]
        {
            self.imm24_slow() as i32
        }
    }

    fn has_24bit_payload(&self) -> bool {
        use Op::*;
        matches!(
            self,
            Const(_)
                | GetGlobal(_)
                | SetGlobal(_)
                | StoreGlobal(_)
                | DefGlobal(_)
                | AddK(_)
                | SubK(_)
                | MulK(_)
                | DivK(_)
                | RemK(_)
                | Closure(_)
                | InitField(_)
                | GetField(_)
                | SetField(_)
                | StoreField(_)
                | GetMethod(_)
                | Struct(_)
                | DefMethod(_)
                | Import(_)
                | GetExport(_)
                | Jump(_)
                | JumpIfFalse(_)
                | JumpIfTrue(_)
                | And(_)
                | Or(_)
                | JNotEq(_)
                | JNotNe(_)
                | JNotLt(_)
                | JNotLe(_)
                | JNotGt(_)
                | JNotGe(_)
                | Ext(_)
        )
    }

    #[cfg(not(target_endian = "little"))]
    fn imm24_slow(self) -> u32 {
        use Op::*;
        match self {
            Const(k) | GetGlobal(k) | SetGlobal(k) | StoreGlobal(k) | DefGlobal(k) | AddK(k) | SubK(k) | MulK(k)
            | DivK(k) | RemK(k) | Closure(k) | InitField(k) | GetField(k) | SetField(k) | StoreField(k)
            | GetMethod(k) | Struct(k) | DefMethod(k) | Import(k) | GetExport(k) => k.get(),
            Jump(o) | JumpIfFalse(o) | JumpIfTrue(o) | And(o) | Or(o) | JNotEq(o) | JNotNe(o) | JNotLt(o)
            | JNotLe(o) | JNotGt(o) | JNotGe(o) | Ext(o) => o.get() as u32,
            _ => unreachable!(),
        }
    }

    /// Ops after which control never falls through to the next instruction.
    pub fn is_terminator(&self) -> bool {
        matches!(self, Op::Jump(_) | Op::Return | Op::ReturnNil | Op::ReturnLocal(_))
    }
}

/// Branch target (word index) of the instruction at `i`, if it branches.
pub fn branch_target(code: &[Op], i: usize) -> Option<usize> {
    let op = code[i];
    if op.is_wide() {
        Some((i as i64 + 2 + code[i + 1].ext() as i64) as usize)
    } else {
        op.jump_offset().map(|o| (i as i64 + 1 + o as i64) as usize)
    }
}

/// Structural check of encoded code: every double instruction is followed by its
/// `Ext`, no `Ext` stands alone, and every branch lands on the start of an
/// instruction inside the function.
pub fn verify(code: &[Op]) -> Result<(), String> {
    let mut starts = vec![false; code.len() + 1];
    let mut i = 0;
    while i < code.len() {
        starts[i] = true;
        let op = code[i];
        if matches!(op, Op::Ext(_)) {
            return Err(format!("stray Ext at {}", i));
        }
        if op.is_wide() {
            match code.get(i + 1) {
                Some(Op::Ext(_)) => {}
                other => return Err(format!("{:?} at {} is not followed by Ext but by {:?}", op, i, other)),
            }
        }
        i += op.width();
    }
    starts[code.len()] = true;
    let mut i = 0;
    while i < code.len() {
        if let Some(t) = branch_target(code, i) {
            if t >= starts.len() || !starts[t] {
                return Err(format!("{:?} at {} branches to {} which is not an instruction start", code[i], i, t));
            }
        }
        i += code[i].width();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_is_four_bytes() {
        assert_eq!(std::mem::size_of::<Op>(), 4);
        assert_eq!(std::mem::size_of::<U24>(), 3);
        assert_eq!(std::mem::size_of::<I24>(), 3);
    }

    #[test]
    fn u24_roundtrip() {
        for v in [0, 1, 255, 256, 65535, 65536, 0x12_3456, U24::MAX] {
            assert_eq!(U24::new(v).unwrap().get(), v);
        }
        assert!(U24::new(U24::MAX + 1).is_none());
    }

    #[test]
    fn i24_roundtrip() {
        for v in [0, 1, -1, 127, -128, 32767, -32768, 65536, -65537, I24::MIN, I24::MAX] {
            assert_eq!(I24::new(v).unwrap().get(), v, "value {}", v);
        }
        assert!(I24::new(I24::MAX + 1).is_none());
        assert!(I24::new(I24::MIN - 1).is_none());
    }

    #[test]
    fn raw_operand_access_matches_the_typed_one() {
        for v in [0u32, 1, 255, 256, 65535, 65536, 0x12_3456, U24::MAX] {
            let op = Op::Const(U24::new(v).unwrap());
            assert_eq!(op.imm24(), v);
            assert_eq!(Op::GetGlobal(U24::new(v).unwrap()).imm24(), v);
        }
        for v in [0, 1, -1, 127, -128, 32767, -32768, 65536, -65537, I24::MIN, I24::MAX] {
            let o = I24::new(v).unwrap();
            assert_eq!(Op::Jump(o).simm24(), v, "Jump {}", v);
            assert_eq!(Op::Ext(o).simm24(), v);
            assert_eq!(Op::JNotLt(o).simm24(), v);
            assert_eq!(Op::Ext(o).ext(), v);
        }
    }

    #[test]
    fn jump_roundtrip() {
        let op = Op::JNotLt(I24::new(5).unwrap());
        assert_eq!(op.jump_offset(), Some(5));
        assert_eq!(op.with_jump_offset(-7).unwrap().jump_offset(), Some(-7));
        assert_eq!(op.with_jump_offset(I24::MAX + 1), None);
        assert_eq!(Op::Nil.with_jump_offset(1), None);
    }

    #[test]
    fn wide_ops_branch_through_ext() {
        let code = [Op::JNotLtLL(1, 2), Op::Ext(I24::new(3).unwrap()), Op::Nil, Op::Nil, Op::Nil, Op::Nil];
        assert!(code[0].is_wide() && code[0].is_branch());
        assert_eq!(code[0].width(), 2);
        assert_eq!(branch_target(&code, 0), Some(5));
        assert_eq!(Op::Jump(I24::new(-2).unwrap()).width(), 1);
    }
}
