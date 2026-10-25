//! Bytecode of the register machine.
//!
//! An instruction is a Rust `enum` that is exactly **4 bytes**: a tag byte plus up
//! to three bytes of operands.  The VM dispatches with a plain `match` (jump table),
//! the compiler can pattern-match on slices of ops, and the disassembler comes for
//! free via `Debug`.
//!
//! Operands are *registers*: slots of the current frame (`R[0..nregs)`, at most 250).
//! The first registers are the function's parameters and local variables, the rest
//! are temporaries handed out by the compiler in stack order.  In the descriptions
//! below `A`, `B`, `C` are register operands (`u8`), `K[x]` is entry `x` of the
//! function's constant table, `Bx` a 16-bit operand.
//!
//! Instructions that need more than three operand bytes are **double**: the first
//! word holds the opcode and its register operands, the following word is an
//! `Op::Ext(I24)`.  For *branching* doubles ([`Op::is_branch`]) the `Ext` is the branch
//! offset (counted in words from the first word after the `Ext`); for the others
//! it carries an extra 24-bit payload (a constant index, a global symbol, ...).
//!
//! Adding an opcode = add a variant, its VM arm, and (if it branches or has an `Ext`)
//! list it in `is_wide`/`is_branch`.

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

/// Highest register number the compiler hands out.
pub const MAX_REG: usize = 250;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Op {
    // ---- moves and constants ----
    /// `R[A] = R[B]`
    Move(u8, u8),
    /// `R[A] = K[Bx]`
    LoadK(u8, u16),
    /// `R[A] = K[ext]` (double; the constant index does not fit 16 bits)
    LoadKW(u8),
    /// `R[A] = imm` (a small integer)
    LoadI(u8, i16),
    /// `R[A ..= A+B] = nil`
    LoadNil(u8, u8),
    LoadTrue(u8),
    LoadFalse(u8),

    // ---- upvalues, globals ----
    /// `R[A] = Upval[B]`
    GetUpval(u8, u8),
    /// `Upval[B] = R[A]`
    SetUpval(u8, u8),
    /// `R[A] = globals[Bx]` (`Bx` is the symbol index); errors when undefined
    GetGlobal(u8, u16),
    /// `globals[Bx] = R[A]`; errors when undefined
    SetGlobal(u8, u16),
    /// define (or redefine) `globals[Bx] = R[A]`
    DefGlobal(u8, u16),
    /// double variants with a 24-bit symbol index in the `Ext`
    GetGlobalW(u8),
    SetGlobalW(u8),
    DefGlobalW(u8),
    /// Close the open upvalues that point at registers `>= A`.
    CloseUpvals(u8),

    // ---- arithmetic: R[A] = R[B] op R[C] ----
    Add(u8, u8, u8),
    Sub(u8, u8, u8),
    Mul(u8, u8, u8),
    Div(u8, u8, u8),
    Rem(u8, u8, u8),
    Pow(u8, u8, u8),
    BitAnd(u8, u8, u8),
    BitOr(u8, u8, u8),
    BitXor(u8, u8, u8),
    Shl(u8, u8, u8),
    Shr(u8, u8, u8),
    /// `R[A] = R[B] op K[C]` where the constant is a number
    AddK(u8, u8, u8),
    SubK(u8, u8, u8),
    MulK(u8, u8, u8),
    DivK(u8, u8, u8),
    RemK(u8, u8, u8),
    /// `R[A] = op R[B]`
    Neg(u8, u8),
    Not(u8, u8),
    BitNot(u8, u8),

    // ---- comparisons producing a bool: R[A] = R[B] cmp R[C] ----
    Eq(u8, u8, u8),
    Ne(u8, u8, u8),
    Lt(u8, u8, u8),
    Le(u8, u8, u8),
    Gt(u8, u8, u8),
    Ge(u8, u8, u8),

    // ---- jumps ----
    Jump(I24),
    /// Double: jump (offset in the `Ext`) if `R[A]` is truthy / falsy.
    JTrue(u8),
    JFalse(u8),
    /// Double: jump if `R[A] cmp R[B]` holds (`JNot…`: if it does not hold).
    /// `==`/`!=` are exact complements, so only `JEq`/`JNe` exist for them; for the
    /// orderings `!(a < b)` differs from `a >= b` when a NaN is involved, so both exist.
    JEq(u8, u8),
    JNe(u8, u8),
    JLt(u8, u8),
    JNotLt(u8, u8),
    JLe(u8, u8),
    JNotLe(u8, u8),
    JGt(u8, u8),
    JNotGt(u8, u8),
    JGe(u8, u8),
    JNotGe(u8, u8),
    /// The same against a constant: `R[A] cmp K[Bx]`.
    JEqK(u8, u16),
    JNeK(u8, u16),
    JLtK(u8, u16),
    JNotLtK(u8, u16),
    JLeK(u8, u16),
    JNotLeK(u8, u16),
    JGtK(u8, u16),
    JNotGtK(u8, u16),
    JGeK(u8, u16),
    JNotGeK(u8, u16),

    // ---- calls ----
    /// `R[A] = R[A](R[A+1], …, R[A+B])`
    Call(u8, u8),
    /// Like `Call`, but reuses the current frame when the callee is a closure.
    /// Always followed by `Return(A)` (which runs when the callee is not a plain closure).
    TailCall(u8, u8),
    Return(u8),
    ReturnNil,
    /// `R[A] = closure(protos[Bx])`
    Closure(u8, u16),

    // ---- tables, arrays, fields ----
    /// `R[A] = {}` (`Bx` = expected number of entries)
    NewTable(u8, u16),
    /// `R[A] = []` (`Bx` = expected number of items)
    NewArray(u8, u16),
    /// Append `R[B ..< B+C]` to the array `R[A]` (array literals; any value, `nil` included).
    AppendN(u8, u8, u8),
    /// Table literal `{ name: v }`: `R[A].K[B] = R[C]`
    InitField(u8, u8, u8),
    /// Double: `R[A].K[ext] = R[B]`
    InitFieldW(u8, u8),
    /// Table literal `{ [k]: v }`: `R[A][R[B]] = R[C]`
    InitIndex(u8, u8, u8),
    /// `R[A] = R[B].K[C]`
    GetField(u8, u8, u8),
    /// Double: `R[A] = R[B].K[ext]`
    GetFieldW(u8, u8),
    /// `R[A].K[B] = R[C]`
    SetField(u8, u8, u8),
    /// Double: `R[A].K[ext] = R[B]`
    SetFieldW(u8, u8),
    /// `R[A] = R[B][R[C]]`
    GetIndex(u8, u8, u8),
    /// `R[A][R[B]] = R[C]`
    SetIndex(u8, u8, u8),
    /// Resolve `obj:name(…)`: `R[A+1] = R[B]; R[A] = method(R[B], K[C])`.
    GetMethod(u8, u8, u8),
    /// Double: the method name is `K[ext]`.
    GetMethodW(u8, u8),

    // ---- structs and unions ----
    /// `R[A] = instantiate(K[Bx])` (a struct/union type template)
    Struct(u8, u16),
    /// Double: register the function `R[B]` as method `K[ext]` of the type `R[A]`.
    DefMethod(u8, u8),

    // ---- pattern matching ----
    /// Double: `R[A] = R[B] is the variant named K[ext & 0xFFFF]` with the arity `ext >> 16`
    /// (255 = any arity); with `C != 255` the value must also belong to the type `R[C]`.
    IsVariant(u8, u8, u8),
    /// `R[A] = field C of the instance R[B]`
    Unpack(u8, u8, u8),
    /// No arm matched the value in `R[A]`: raise an error.
    MatchFail(u8),

    /// `?`: if `R[A]` is `Ok(v)`/`Some(v)` set `R[A] = v`; if it is `Err`/`None` return it.
    Try(u8),

    // ---- modules ----
    /// `R[A] = import K[Bx]`
    Import(u8, u16),
    /// Double: `R[A] = R[B].K[ext]` as an export; errors if the module has no such export.
    GetExport(u8, u8),

    // ---- loops (double: the `Ext` is the branch offset) ----
    /// Range loop setup: `R[A]`, `R[A+1]` hold start/end; if start >= end jump,
    /// otherwise the loop variable `R[A+2]` is set to start.
    ForPrep(u8),
    /// Advance `R[A]`; while it is below `R[A+1]` set `R[A+2]` and jump back.
    ForLoop(u8),
    /// Iteration over an array / table / string: `R[A]` = collection, `R[A+1]` = cursor,
    /// `R[A+2]` = value (one variable) or key and value in `R[A+2]`, `R[A+3]`.
    IterNext1(u8),
    IterNext2(u8),

    /// Second word of a double instruction.  Never executed.
    Ext(I24),
}

const _: () = assert!(std::mem::size_of::<Op>() == 4);

impl Op {
    /// Double instruction: the word after it is an `Ext`.
    #[inline]
    pub fn is_wide(&self) -> bool {
        use Op::*;
        self.is_branch_wide()
            || matches!(
                self,
                LoadKW(_)
                    | GetGlobalW(_)
                    | SetGlobalW(_)
                    | DefGlobalW(_)
                    | InitFieldW(..)
                    | GetFieldW(..)
                    | SetFieldW(..)
                    | GetMethodW(..)
                    | DefMethod(..)
                    | IsVariant(..)
                    | GetExport(..)
            )
    }

    /// Double instruction whose `Ext` is a branch offset.
    #[inline]
    pub fn is_branch_wide(&self) -> bool {
        use Op::*;
        matches!(
            self,
            JTrue(_)
                | JFalse(_)
                | JEq(..)
                | JNe(..)
                | JLt(..)
                | JNotLt(..)
                | JLe(..)
                | JNotLe(..)
                | JGt(..)
                | JNotGt(..)
                | JGe(..)
                | JNotGe(..)
                | JEqK(..)
                | JNeK(..)
                | JLtK(..)
                | JNotLtK(..)
                | JLeK(..)
                | JNotLeK(..)
                | JGtK(..)
                | JNotGtK(..)
                | JGeK(..)
                | JNotGeK(..)
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

    /// Offset of the single-word branch (`Jump`).  Double instructions keep their
    /// offset in the `Ext` word instead.
    pub fn jump_offset(&self) -> Option<i32> {
        match *self {
            Op::Jump(o) => Some(o.get()),
            _ => None,
        }
    }

    /// Returns the single-word branch with a new offset, or `None` if it does not fit
    /// (or the op is not a single-word branch).
    pub fn with_jump_offset(self, off: i32) -> Option<Op> {
        match self {
            Op::Jump(_) => Some(Op::Jump(I24::new(off)?)),
            _ => None,
        }
    }

    /// Does control transfer somewhere (single-word or double instruction)?
    #[inline]
    pub fn is_branch(&self) -> bool {
        self.is_branch_wide() || matches!(self, Op::Jump(_))
    }

    /// Signed value of an `Ext` word.
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

    /// Unsigned 24-bit payload of an `Ext` word (constant indices and the like).
    #[inline(always)]
    pub fn ext_u(self) -> u32 {
        self.ext() as u32 & U24::MAX
    }

    /// Unsigned 24-bit payload of an `Ext` word, read straight from the word.
    #[inline(always)]
    pub fn payload(self) -> u32 {
        debug_assert!(matches!(self, Op::Ext(_)));
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `Op` is 4 bytes and `Ext` has no padding bytes
            unsafe { std::mem::transmute::<Op, u32>(self) >> 8 }
        }
        #[cfg(not(target_endian = "little"))]
        {
            self.ext_u()
        }
    }

    /// The signed 24-bit operand of `Jump` / `Ext`, read straight from the word
    /// (`#[repr(u8)]` puts the tag in byte 0 and the three operand bytes right behind it).
    #[inline(always)]
    pub fn simm24(self) -> i32 {
        debug_assert!(matches!(self, Op::Jump(_) | Op::Ext(_)));
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `Op` is 4 bytes and these variants have no padding bytes
            unsafe { std::mem::transmute::<Op, i32>(self) >> 8 }
        }
        #[cfg(not(target_endian = "little"))]
        {
            match self {
                Op::Jump(o) | Op::Ext(o) => o.get(),
                _ => unreachable!(),
            }
        }
    }

    /// Ops after which control never falls through to the next instruction.
    pub fn is_terminator(&self) -> bool {
        matches!(self, Op::Jump(_) | Op::Return(_) | Op::ReturnNil | Op::MatchFail(_))
    }
}

/// Branch target (word index) of the instruction at `i`, if it branches.
pub fn branch_target(code: &[Op], i: usize) -> Option<usize> {
    let op = code[i];
    if op.is_branch_wide() {
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
    fn operand_layouts_do_not_grow_the_word() {
        // every operand shape used by the instruction set fits next to the tag byte
        assert_eq!(std::mem::size_of::<(u8, u8, u8)>(), 3);
        let _ = [Op::LoadI(1, -5), Op::LoadK(2, 65535), Op::Add(1, 2, 3), Op::JEqK(3, 65535), Op::Closure(4, 9)];
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
        for v in [0, 1, -1, 127, -128, 32767, -32768, 65536, -65537, I24::MIN, I24::MAX] {
            let o = I24::new(v).unwrap();
            assert_eq!(Op::Jump(o).simm24(), v, "Jump {}", v);
            assert_eq!(Op::Ext(o).simm24(), v);
            assert_eq!(Op::Ext(o).ext(), v);
        }
        for v in [0u32, 1, 255, 65535, 65536, 0x12_3456, 0x80_0000, U24::MAX] {
            let e = Op::Ext(I24::new((v << 8) as i32 >> 8).unwrap());
            assert_eq!(e.ext_u(), v);
            assert_eq!(e.payload(), v);
        }
    }

    #[test]
    fn jump_roundtrip() {
        let op = Op::Jump(I24::new(5).unwrap());
        assert_eq!(op.jump_offset(), Some(5));
        assert_eq!(op.with_jump_offset(-7).unwrap().jump_offset(), Some(-7));
        assert_eq!(op.with_jump_offset(I24::MAX + 1), None);
        assert_eq!(Op::ReturnNil.with_jump_offset(1), None);
    }

    #[test]
    fn wide_ops_branch_through_ext() {
        let code = [
            Op::JLt(1, 2),
            Op::Ext(I24::new(3).unwrap()),
            Op::ReturnNil,
            Op::ReturnNil,
            Op::ReturnNil,
            Op::ReturnNil,
        ];
        assert!(code[0].is_wide() && code[0].is_branch());
        assert_eq!(code[0].width(), 2);
        assert_eq!(branch_target(&code, 0), Some(5));
        assert_eq!(Op::Jump(I24::new(-2).unwrap()).width(), 1);
        // a data double has an Ext too but does not branch
        assert!(Op::LoadKW(1).is_wide() && !Op::LoadKW(1).is_branch());
        assert!(verify(&code).is_ok());
    }

    #[test]
    fn verify_rejects_bad_code() {
        assert!(verify(&[Op::LoadKW(0)]).is_err());
        assert!(verify(&[Op::Ext(I24::ZERO)]).is_err());
        assert!(verify(&[Op::Jump(I24::new(5).unwrap()), Op::ReturnNil]).is_err());
    }
}
