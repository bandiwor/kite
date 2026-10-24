//! Bytecode.  An instruction is a Rust `enum` that fits in 8 bytes, so the VM
//! dispatches with a plain `match` (jump table), the compiler/peephole can
//! pattern-match on slices of ops, and the disassembler comes for free via
//! `Debug`.  Adding an opcode = add a variant, its `stack_effect`, and a VM arm.
//!
//! Jump offsets are relative to the *next* instruction.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    // ---- constants & stack ----
    Nil,
    True,
    False,
    Const(u32),
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
    GetLocal2(u16, u16),
    /// Push `local[a] <op> local[b]`.
    AddLL(u16, u16),
    SubLL(u16, u16),
    MulLL(u16, u16),
    GetUpval(u16),
    SetUpval(u16),
    StoreUpval(u16),
    GetGlobal(u32),
    SetGlobal(u32),
    StoreGlobal(u32),
    /// Define (or redefine) a global; pops.
    DefGlobal(u32),
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
    AddK(u32),
    SubK(u32),
    MulK(u32),
    DivK(u32),
    RemK(u32),
    /// `local[s] += consts[k]` (statement form, pushes nothing).
    AddLocalK(u16, u32),

    // ---- comparison ----
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // ---- jumps ----
    Jump(i32),
    /// Pop; jump if falsy.
    JumpIfFalse(i32),
    /// Pop; jump if truthy.
    JumpIfTrue(i32),
    /// `a && b`: if top is falsy jump (keeping it), else pop and fall through.
    And(i32),
    /// `a || b`: if top is truthy jump (keeping it), else pop and fall through.
    Or(i32),

    // ---- fused compare + "jump if the comparison is false" (peephole) ----
    JNotEq(i32),
    JNotNe(i32),
    JNotLt(i32),
    JNotLe(i32),
    JNotGt(i32),
    JNotGe(i32),
    // local OP local
    JNotEqLL(u16, u16, i16),
    JNotNeLL(u16, u16, i16),
    JNotLtLL(u16, u16, i16),
    JNotLeLL(u16, u16, i16),
    JNotGtLL(u16, u16, i16),
    JNotGeLL(u16, u16, i16),
    // local OP constant
    JNotEqLK(u16, u16, i16),
    JNotNeLK(u16, u16, i16),
    JNotLtLK(u16, u16, i16),
    JNotLeLK(u16, u16, i16),
    JNotGtLK(u16, u16, i16),
    JNotGeLK(u16, u16, i16),

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
    Closure(u32),

    // ---- tables, fields, structs ----
    /// New table with room for `n` array items.
    NewTable(u16),
    /// `[t, v] -> [t]`  (array literal element number `i`; a `nil` leaves a hole)
    Append(u16),
    /// `[t, v] -> [t]`  (table literal `name: v`)
    InitField(u32),
    /// `[t, k, v] -> [t]`  (table literal `[k]: v`)
    InitIndex,
    GetField(u32),
    /// `GetLocal; GetField`: push `local[s].name`.
    GetLocalField(u16, u32),
    /// `[obj, v] -> [v]`
    SetField(u32),
    /// `[obj, v] -> []`
    StoreField(u32),
    GetIndex,
    /// `[obj, k, v] -> [v]`
    SetIndex,
    /// `[obj, k, v] -> []`
    StoreIndex,
    /// `[obj] -> [function, obj]`: resolve `obj:name(...)`.
    GetMethod(u32),
    /// Instantiate the struct type template in `consts[k]`.
    Struct(u32),
    /// `[type, closure] -> [type]`: register an `impl` function under `consts[k]`.
    DefMethod(u32),

    // ---- loops ----
    /// Range loop setup: locals `s`,`s+1` hold start/end; if start >= end jump.
    /// Otherwise the loop variable (`s+2`) is set to start.
    ForPrep(u16, i32),
    /// Advance the range cursor; if still in range set the variable and jump back.
    ForLoop(u16, i32),
    /// Table iteration, one variable (value).  Locals `s`,`s+1` = table, cursor.
    IterNext1(u16, i32),
    /// Table iteration, two variables (key, value).
    IterNext2(u16, i32),
}

const _: () = assert!(std::mem::size_of::<Op>() == 8);

impl Op {
    /// Net effect on the operand stack when execution falls through to the next op.
    pub fn stack_effect(&self) -> i32 {
        use Op::*;
        match *self {
            Nil | True | False | Const(_) | Dup | GetLocal(_) | GetUpval(_) | GetGlobal(_) | Closure(_)
            | NewTable(_) | Struct(_) | GetMethod(_) | AddLL(..) | SubLL(..) | MulLL(..) | GetLocalField(..) => 1,
            Dup2 | GetLocal2(..) => 2,
            Pop | StoreLocal(_) | StoreUpval(_) | StoreGlobal(_) | DefGlobal(_) | Append(_) | InitField(_)
            | SetField(_) | GetIndex | DefMethod(_) | Return => -1,
            PopN(n) => -(n as i32),
            SetLocal(_) | SetUpval(_) | SetGlobal(_) | CloseUpvals(_) | Neg | Not | BitNot | AddK(_) | SubK(_)
            | MulK(_) | DivK(_) | RemK(_) | AddLocalK(..) | Jump(_) | GetField(_) | ForPrep(..) | ForLoop(..)
            | IterNext1(..) | IterNext2(..) | ReturnNil | ReturnLocal(_) | JNotEqLL(..) | JNotNeLL(..) | JNotLtLL(..)
            | JNotLeLL(..) | JNotGtLL(..) | JNotGeLL(..) | JNotEqLK(..) | JNotNeLK(..) | JNotLtLK(..)
            | JNotLeLK(..) | JNotGtLK(..) | JNotGeLK(..) => 0,
            Add | Sub | Mul | Div | Rem | Pow | BitAnd | BitOr | BitXor | Shl | Shr | Eq | Ne | Lt | Le | Gt
            | Ge | JumpIfFalse(_) | JumpIfTrue(_) | And(_) | Or(_) => -1,
            JNotEq(_) | JNotNe(_) | JNotLt(_) | JNotLe(_) | JNotGt(_) | JNotGe(_) | StoreField(_) | InitIndex
            | SetIndex => -2,
            StoreIndex => -3,
            Call(n) | TailCall(n) => -(n as i32),
        }
    }

    /// Relative jump offset, if this op branches.
    pub fn jump_offset(&self) -> Option<i32> {
        use Op::*;
        Some(match *self {
            Jump(o) | JumpIfFalse(o) | JumpIfTrue(o) | And(o) | Or(o) | JNotEq(o) | JNotNe(o) | JNotLt(o)
            | JNotLe(o) | JNotGt(o) | JNotGe(o) | ForPrep(_, o) | ForLoop(_, o) | IterNext1(_, o)
            | IterNext2(_, o) => o,
            JNotEqLL(_, _, o) | JNotNeLL(_, _, o) | JNotLtLL(_, _, o) | JNotLeLL(_, _, o) | JNotGtLL(_, _, o)
            | JNotGeLL(_, _, o) | JNotEqLK(_, _, o) | JNotNeLK(_, _, o) | JNotLtLK(_, _, o)
            | JNotLeLK(_, _, o) | JNotGtLK(_, _, o) | JNotGeLK(_, _, o) => o as i32,
            _ => return None,
        })
    }

    /// Returns the op with a new offset, or `None` if it does not fit.
    pub fn with_jump_offset(self, off: i32) -> Option<Op> {
        use Op::*;
        let small = i16::try_from(off).ok();
        Some(match self {
            Jump(_) => Jump(off),
            JumpIfFalse(_) => JumpIfFalse(off),
            JumpIfTrue(_) => JumpIfTrue(off),
            And(_) => And(off),
            Or(_) => Or(off),
            JNotEq(_) => JNotEq(off),
            JNotNe(_) => JNotNe(off),
            JNotLt(_) => JNotLt(off),
            JNotLe(_) => JNotLe(off),
            JNotGt(_) => JNotGt(off),
            JNotGe(_) => JNotGe(off),
            ForPrep(s, _) => ForPrep(s, off),
            ForLoop(s, _) => ForLoop(s, off),
            IterNext1(s, _) => IterNext1(s, off),
            IterNext2(s, _) => IterNext2(s, off),
            JNotEqLL(a, b, _) => JNotEqLL(a, b, small?),
            JNotNeLL(a, b, _) => JNotNeLL(a, b, small?),
            JNotLtLL(a, b, _) => JNotLtLL(a, b, small?),
            JNotLeLL(a, b, _) => JNotLeLL(a, b, small?),
            JNotGtLL(a, b, _) => JNotGtLL(a, b, small?),
            JNotGeLL(a, b, _) => JNotGeLL(a, b, small?),
            JNotEqLK(a, b, _) => JNotEqLK(a, b, small?),
            JNotNeLK(a, b, _) => JNotNeLK(a, b, small?),
            JNotLtLK(a, b, _) => JNotLtLK(a, b, small?),
            JNotLeLK(a, b, _) => JNotLeLK(a, b, small?),
            JNotGtLK(a, b, _) => JNotGtLK(a, b, small?),
            JNotGeLK(a, b, _) => JNotGeLK(a, b, small?),
            _ => return None,
        })
    }

    /// Ops after which control never falls through to the next instruction.
    pub fn is_terminator(&self) -> bool {
        matches!(self, Op::Jump(_) | Op::Return | Op::ReturnNil | Op::ReturnLocal(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn jump_roundtrip() {
        let op = Op::JNotLtLL(1, 2, 5);
        assert_eq!(op.jump_offset(), Some(5));
        assert_eq!(op.with_jump_offset(-7), Some(Op::JNotLtLL(1, 2, -7)));
        assert_eq!(op.with_jump_offset(100_000), None);
        assert_eq!(Op::Jump(1).with_jump_offset(100_000), Some(Op::Jump(100_000)));
    }
}
