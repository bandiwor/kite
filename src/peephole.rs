//! Peephole optimiser, run once per function after code generation.
//!
//! One left-to-right pass copies the code into an output buffer of logical
//! instructions (a double instruction with its `Ext` word is one entry) and, every
//! time an instruction is appended, tries to fuse the *tail* of that buffer into a
//! superinstruction (`Set+Pop -> Store`, compare+jump, `x += k`, ...).  Because
//! fusion looks at the output, chains compose in the same pass (`SetLocal;Pop` ->
//! `StoreLocal`, then `GetLocal;AddK;StoreLocal` -> `AddLocalK`).  The pass also
//!  * threads jumps (a jump to an unconditional jump goes to its target; a
//!    jump straight to a `Return` becomes that `Return`),
//!  * drops code that follows `Jump`/`Return` and that nothing targets,
//!  * drops `Jump(0)`.
//!
//! Fused ops need small operands (slots below 256, constants below 65536); when a
//! window does not fit it is simply left alone.  A window is only fused when none of
//! its interior instructions is a jump target.  Branches are remapped afterwards
//! through an old -> new word index table, which is also where double instructions
//! get their `Ext` word.  The scratch buffers are reused between functions.

use crate::op::{branch_target, Op, I24, U24};
use crate::value::Value;

/// An output instruction together with the bookkeeping needed to remap branches.
#[derive(Clone, Copy)]
struct Entry {
    op: Op,
    /// Source offset of the instruction (for error positions).
    off: u32,
    /// Old word index of the first instruction this entry replaces.
    first: u32,
    /// Old word index of the branch target, if `op` branches.
    target: u32,
}

#[derive(Default)]
pub struct Scratch {
    is_target: Vec<bool>,
    out: Vec<Entry>,
    map: Vec<u32>,
    pos: Vec<u32>,
}

/// Convenience wrapper with throw-away buffers (tests, one-off use).
pub fn optimize(code: Vec<Op>, offsets: Vec<u32>, consts: &[Value]) -> Result<(Vec<Op>, Vec<u32>), String> {
    optimize_with(&mut Scratch::default(), code, offsets, consts)
}

pub fn optimize_with(
    sc: &mut Scratch,
    mut code: Vec<Op>,
    mut offsets: Vec<u32>,
    consts: &[Value],
) -> Result<(Vec<Op>, Vec<u32>), String> {
    pass(sc, &mut code, &mut offsets, consts)?;
    Ok((code, offsets))
}

#[derive(Clone, Copy, PartialEq)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn as_cmp(op: Op) -> Option<Cmp> {
    Some(match op {
        Op::Eq => Cmp::Eq,
        Op::Ne => Cmp::Ne,
        Op::Lt => Cmp::Lt,
        Op::Le => Cmp::Le,
        Op::Gt => Cmp::Gt,
        Op::Ge => Cmp::Ge,
        _ => return None,
    })
}

// the fused ops get their real offsets when the output is written back
fn jnot(c: Cmp, o: I24) -> Op {
    match c {
        Cmp::Eq => Op::JNotEq(o),
        Cmp::Ne => Op::JNotNe(o),
        Cmp::Lt => Op::JNotLt(o),
        Cmp::Le => Op::JNotLe(o),
        Cmp::Gt => Op::JNotGt(o),
        Cmp::Ge => Op::JNotGe(o),
    }
}

fn jnot_ll(c: Cmp, a: u8, b: u8) -> Op {
    match c {
        Cmp::Eq => Op::JNotEqLL(a, b),
        Cmp::Ne => Op::JNotNeLL(a, b),
        Cmp::Lt => Op::JNotLtLL(a, b),
        Cmp::Le => Op::JNotLeLL(a, b),
        Cmp::Gt => Op::JNotGtLL(a, b),
        Cmp::Ge => Op::JNotGeLL(a, b),
    }
}

fn jnot_lk(c: Cmp, a: u8, k: u16) -> Op {
    match c {
        Cmp::Eq => Op::JNotEqLK(a, k),
        Cmp::Ne => Op::JNotNeLK(a, k),
        Cmp::Lt => Op::JNotLtLK(a, k),
        Cmp::Le => Op::JNotLeLK(a, k),
        Cmp::Gt => Op::JNotGtLK(a, k),
        Cmp::Ge => Op::JNotGeLK(a, k),
    }
}

fn is_num_const(consts: &[Value], k: U24) -> bool {
    consts.get(k.get() as usize).is_some_and(|v| v.is_num())
}

fn is_pure_push(op: Op) -> bool {
    matches!(op, Op::Nil | Op::True | Op::False | Op::Const(_) | Op::GetLocal(_) | Op::GetUpval(_))
}

fn small_slot(s: u16) -> Option<u8> {
    u8::try_from(s).ok()
}

fn small_const(k: U24) -> Option<u16> {
    u16::try_from(k.get()).ok()
}

/// Cheap pre-filter: only these ops can be the last op of a fusable window.
#[inline]
fn can_end_window(op: Op) -> bool {
    use Op::*;
    matches!(
        op,
        Pop | PopN(_)
            | Return
            | GetField(_)
            | GetLocal(_)
            | Add
            | Sub
            | Mul
            | Div
            | Rem
            | JumpIfFalse(_)
            | JumpIfTrue(_)
            | StoreLocal(_)
    )
}

/// What a matched window turns into.
enum Rewrite {
    /// Replace the whole window by one op.
    One(Op),
    /// Replace the whole window by two ops (the second carries the jump/position).
    Two(Op, Op),
    /// The window cancels out.
    Nothing,
}

/// Try to rewrite the tail of the output buffer.  Returns the rewrite and how many
/// trailing entries it consumed.  `ok(width)` says whether a window of that many
/// entries may be fused (no jump target inside it).
fn match_tail(entries: &[Entry], consts: &[Value], ok: &dyn Fn(usize) -> bool) -> Option<(Rewrite, usize)> {
    use Op::*;
    let n = entries.len();
    let t = |k: usize| entries[n - k].op;
    // ---- 4 wide: local OP constant, then branch
    if n >= 4 && ok(4) {
        if let [GetLocal(a), Const(k), c, JumpIfFalse(_)] = [t(4), t(3), t(2), t(1)] {
            if let (Some(c), Some(a), Some(k)) = (as_cmp(c), small_slot(a), small_const(k)) {
                return Some((Rewrite::One(jnot_lk(c, a, k)), 4));
            }
        }
    }
    // ---- 3 wide
    if n >= 3 && ok(3) {
        match [t(3), t(2), t(1)] {
            [GetLocal2(a, b), c, JumpIfFalse(_)] => {
                if let Some(c) = as_cmp(c) {
                    return Some((Rewrite::One(jnot_ll(c, a, b)), 3));
                }
            }
            [GetLocal(s), AddK(k), StoreLocal(d)] if s == d => {
                if let (Some(s), Some(k)) = (small_slot(s), small_const(k)) {
                    return Some((Rewrite::One(AddLocalK(s, k)), 3));
                }
            }
            _ => {}
        }
    }
    // ---- 2 wide
    if !ok(2) {
        return None;
    }
    let (a, b) = (t(2), t(1));
    let r = match (a, b) {
        (SetLocal(s), Pop) => StoreLocal(s),
        (SetUpval(s), Pop) => StoreUpval(s),
        (SetGlobal(s), Pop) => StoreGlobal(s),
        (SetField(s), Pop) => StoreField(s),
        (SetIndex, Pop) => StoreIndex,
        (Nil, Return) => ReturnNil,
        (GetLocal(s), Return) => ReturnLocal(s),
        (GetLocal(s), GetField(k)) => GetLocalField(small_slot(s)?, small_const(k)?),
        // `a; b; .field` : the field read wants `b` on its own
        (GetLocal2(x, y), GetField(k)) => {
            return Some((Rewrite::Two(GetLocal(x as u16), GetLocalField(y, small_const(k)?)), 2));
        }
        (GetLocal2(x, y), Add) => AddLL(x, y),
        (GetLocal2(x, y), Sub) => SubLL(x, y),
        (GetLocal2(x, y), Mul) => MulLL(x, y),
        // fused ops keep the `StoreLocal` after them (see `Op::AddLLS`)
        (AddLL(x, y), StoreLocal(d)) => return Some((Rewrite::Two(AddLLS(x, y, small_slot(d)?), StoreLocal(d)), 2)),
        (SubLL(x, y), StoreLocal(d)) => return Some((Rewrite::Two(SubLLS(x, y, small_slot(d)?), StoreLocal(d)), 2)),
        (MulLL(x, y), StoreLocal(d)) => return Some((Rewrite::Two(MulLLS(x, y, small_slot(d)?), StoreLocal(d)), 2)),
        (Add, StoreLocal(d)) => return Some((Rewrite::Two(AddS(d), StoreLocal(d)), 2)),
        (Sub, StoreLocal(d)) => return Some((Rewrite::Two(SubS(d), StoreLocal(d)), 2)),
        (Mul, StoreLocal(d)) => return Some((Rewrite::Two(MulS(d), StoreLocal(d)), 2)),
        (GetLocal(x), GetLocal(y)) => GetLocal2(small_slot(x)?, small_slot(y)?),
        (Pop, Pop) => PopN(2),
        (PopN(n), Pop) | (Pop, PopN(n)) if n < u16::MAX => PopN(n + 1),
        (Not, JumpIfFalse(o)) => JumpIfTrue(o),
        (Not, JumpIfTrue(o)) => JumpIfFalse(o),
        (Const(k), Add) if is_num_const(consts, k) => AddK(k),
        (Const(k), Sub) if is_num_const(consts, k) => SubK(k),
        (Const(k), Mul) if is_num_const(consts, k) => MulK(k),
        (Const(k), Div) if is_num_const(consts, k) => DivK(k),
        (Const(k), Rem) if is_num_const(consts, k) => RemK(k),
        (c, JumpIfFalse(o)) if as_cmp(c).is_some() => jnot(as_cmp(c).unwrap(), o),
        (p, Pop) if is_pure_push(p) => return Some((Rewrite::Nothing, 2)),
        _ => return None,
    };
    Some((Rewrite::One(r), 2))
}

/// Resolve the destination of the branch at `i` through chains of unconditional jumps.
/// Returns the final target, or `None` when the instruction was replaced by a return.
fn thread(code: &mut [Op], i: usize) -> Option<usize> {
    let n = code.len();
    let t0 = branch_target(code, i)?;
    let mut t = t0;
    let mut hops = 0;
    while t < n && hops < 8 {
        match code[t] {
            Op::Jump(o) => {
                t = (t as i64 + 1 + o.get() as i64) as usize;
                hops += 1;
            }
            _ => break,
        }
    }
    // an unconditional jump straight to a return is just that return
    if let (Op::Jump(_), Some(&r)) = (code[i], code.get(t)) {
        if matches!(r, Op::Return | Op::ReturnNil | Op::ReturnLocal(_)) {
            code[i] = r;
            return None;
        }
    }
    if t != t0 {
        let wide = code[i].is_wide();
        let rel = t as i64 - (i as i64 + 1 + wide as i64);
        match I24::new(rel as i32) {
            Some(o) if wide => code[i + 1] = Op::Ext(o),
            Some(o) => code[i] = code[i].with_jump_offset(o.get()).unwrap(),
            None => t = t0, // does not fit; keep the original destination
        }
    }
    Some(t)
}

/// The whole optimisation: threading, target marking, and one fusing sweep.
fn pass(sc: &mut Scratch, code: &mut Vec<Op>, offsets: &mut Vec<u32>, consts: &[Value]) -> Result<(), String> {
    let n = code.len();
    let Scratch { is_target, out, map, pos } = sc;
    is_target.clear();
    is_target.resize(n + 1, false);

    let mut i = 0;
    while i < n {
        // `Call; Return` becomes `TailCall; Return`.  The Return stays: it is what
        // runs when the callee is a native (or anything that is not a plain closure).
        if let Op::Call(argc) = code[i] {
            if code.get(i + 1) == Some(&Op::Return) {
                code[i] = Op::TailCall(argc);
            }
        }
        if let Some(t) = thread(code, i) {
            if t <= n {
                is_target[t] = true;
            }
        }
        i += code[i].width();
    }

    out.clear();
    map.clear();
    map.resize(n + 1, 0);

    let mut dead = false;
    let mut i = 0;
    while i < n {
        let op = code[i];
        let w = op.width();
        let end = i + w - 1;
        if is_target[i] {
            dead = false;
        }
        if dead || matches!(op, Op::Jump(o) if o.get() == 0) {
            map[i..=end].fill(out.len() as u32);
            i += w;
            continue;
        }
        map[i..=end].fill(out.len() as u32);
        let target = branch_target(code, i).map_or(0, |t| t as u32);
        out.push(Entry { op, off: offsets[i], first: i as u32, target });

        // fuse the tail as long as something matches
        let mut vanished = false;
        while out.len() >= 2 && can_end_window(out[out.len() - 1].op) {
            // a window may only be fused when no old instruction inside it (other
            // than its first) is a jump target - including removed `Jump(0)`s
            let ok = |width: usize| {
                let lo = out[out.len() - width].first as usize + 1;
                !(lo..=end).any(|o| is_target[o])
            };
            let Some((rw, width)) = match_tail(out, consts, &ok) else { break };
            let start = out.len() - width;
            let carrier = out[out.len() - 1];
            let first_old = out[start].first;
            out.truncate(start);
            match rw {
                Rewrite::One(op) => out.push(Entry { op, off: carrier.off, first: first_old, target: carrier.target }),
                Rewrite::Two(o1, o2) => {
                    out.push(Entry { op: o1, off: carrier.off, first: first_old, target: 0 });
                    out.push(Entry { op: o2, off: carrier.off, first: carrier.first, target: carrier.target });
                }
                Rewrite::Nothing => vanished = true,
            }
            // old indices that pointed into the window now point at its replacement
            // (or, if it vanished, at whatever is emitted next, which is also `start`)
            map[first_old as usize..=end].fill(start as u32);
            if let Rewrite::Two(..) = rw {
                map[carrier.first as usize] = (start + 1) as u32;
            }
            if vanished {
                break;
            }
        }
        // code after an unconditional transfer is dead - unless the window just
        // vanished, in which case the last op is an older one
        if !vanished && out.last().is_some_and(|e| e.op.is_terminator()) {
            dead = true;
        }
        i += w;
    }
    map[n] = out.len() as u32;

    // word positions of the output entries
    pos.clear();
    let mut words = 0u32;
    for e in out.iter() {
        pos.push(words);
        words += e.op.width() as u32;
    }
    let word_of = |entry: u32| -> i64 {
        if entry as usize >= pos.len() {
            words as i64
        } else {
            pos[entry as usize] as i64
        }
    };

    // remap branches and write back
    code.clear();
    offsets.clear();
    let too_large = || "function is too large (a jump spans more than 8M instructions)".to_string();
    for (j, e) in out.iter().enumerate() {
        let here = pos[j] as i64;
        if e.op.is_wide() {
            let new_target = word_of(map[(e.target as usize).min(n)]);
            let rel = new_target - (here + 2);
            let o = i32::try_from(rel).ok().and_then(I24::new).ok_or_else(too_large)?;
            code.push(e.op);
            code.push(Op::Ext(o));
            offsets.push(e.off);
            offsets.push(e.off);
        } else if e.op.jump_offset().is_some() {
            let new_target = word_of(map[(e.target as usize).min(n)]);
            let rel = new_target - (here + 1);
            let op = i32::try_from(rel).ok().and_then(|r| e.op.with_jump_offset(r)).ok_or_else(too_large)?;
            code.push(op);
            offsets.push(e.off);
        } else {
            code.push(e.op);
            offsets.push(e.off);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(v: i32) -> I24 {
        I24::new(v).unwrap()
    }
    fn k(v: u32) -> U24 {
        U24::new(v).unwrap()
    }

    fn opt(code: Vec<Op>) -> Vec<Op> {
        let n = code.len();
        optimize(code, vec![0; n], &[Value::num(1.0)]).unwrap().0
    }

    #[test]
    fn store_fusion() {
        assert_eq!(opt(vec![Op::SetLocal(1), Op::Pop, Op::Nil, Op::Return]), vec![Op::StoreLocal(1), Op::ReturnNil]);
    }

    #[test]
    fn arithmetic_fused_with_its_store_keeps_the_store() {
        // `x = a + b`: the fused op is followed by the plain store (its slow path falls into it)
        assert_eq!(
            opt(vec![Op::GetLocal2(0, 1), Op::Add, Op::StoreLocal(2), Op::Nil, Op::Return]),
            vec![Op::AddLLS(0, 1, 2), Op::StoreLocal(2), Op::ReturnNil]
        );
        assert_eq!(
            opt(vec![Op::GetLocal2(0, 1), Op::Mul, Op::StoreLocal(0), Op::Nil, Op::Return]),
            vec![Op::MulLLS(0, 1, 0), Op::StoreLocal(0), Op::ReturnNil]
        );
        // operands from the stack
        assert_eq!(
            opt(vec![Op::GetLocal(0), Op::Const(k(0)), Op::Sub, Op::StoreLocal(3), Op::Nil, Op::Return]),
            vec![Op::GetLocal(0), Op::SubK(k(0)), Op::StoreLocal(3), Op::ReturnNil]
        );
        assert_eq!(
            opt(vec![Op::GetUpval(0), Op::GetUpval(1), Op::Add, Op::StoreLocal(300), Op::Nil, Op::Return]),
            vec![Op::GetUpval(0), Op::GetUpval(1), Op::AddS(300), Op::StoreLocal(300), Op::ReturnNil]
        );
        // a destination above slot 255 cannot use the three-byte form
        assert_eq!(
            opt(vec![Op::GetLocal2(0, 1), Op::Add, Op::StoreLocal(300), Op::Nil, Op::Return]),
            vec![Op::AddLL(0, 1), Op::StoreLocal(300), Op::ReturnNil]
        );
    }

    #[test]
    fn a_jump_into_the_store_prevents_the_fusion() {
        // 0: Jump -> 3 (the StoreLocal) ; 1: GetLocal2 ; 2: Add ; 3: StoreLocal ; 4: Nil ; 5: Return
        let code = vec![Op::Jump(o(2)), Op::GetLocal2(0, 1), Op::Add, Op::StoreLocal(2), Op::Nil, Op::Return];
        let out = opt(code);
        assert!(!out.iter().any(|op| matches!(op, Op::AddLLS(..) | Op::AddS(_))), "{:?}", out);
    }

    #[test]
    fn dead_code_and_threading() {
        // 0: Jump -> 2 ; 1: dead ; 2: Jump -> 4 ; 3: dead ; 4: Return
        let code = vec![Op::Jump(o(1)), Op::Nil, Op::Jump(o(1)), Op::True, Op::Return];
        assert_eq!(opt(code), vec![Op::Return]);
    }

    #[test]
    fn compare_jump_becomes_a_double_instruction() {
        let code = vec![
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::Lt,
            Op::JumpIfFalse(o(2)),
            Op::Nil,
            Op::Pop,
            Op::Nil,
            Op::Return,
        ];
        let out = opt(code);
        assert_eq!(out[0], Op::JNotLtLL(0, 1));
        assert!(matches!(out[1], Op::Ext(_)));
        assert_eq!(out[1].ext(), 0); // falls through to the (fused) tail
    }

    #[test]
    fn loop_with_double_instruction_is_remapped() {
        // 0..3: i < K ; 4: GetLocal;Pop (vanishes) ; 6: back jump ; 7: exit
        let code = vec![
            Op::GetLocal(0),
            Op::Const(k(0)),
            Op::Lt,
            Op::JumpIfFalse(o(3)),
            Op::GetLocal(0),
            Op::Pop,
            Op::Jump(o(-7)),
            Op::Nil,
            Op::Return,
        ];
        let out = opt(code);
        assert_eq!(out, vec![Op::JNotLtLK(0, 0), Op::Ext(o(1)), Op::Jump(o(-3)), Op::ReturnNil]);
    }

    #[test]
    fn increment_chain_in_one_pass() {
        let code = vec![Op::GetLocal(2), Op::Const(k(0)), Op::Add, Op::SetLocal(2), Op::Pop, Op::Nil, Op::Return];
        assert_eq!(opt(code), vec![Op::AddLocalK(2, 0), Op::ReturnNil]);
    }

    #[test]
    fn field_read_prefers_second_local() {
        let code = vec![Op::GetLocal(0), Op::GetLocal(1), Op::GetField(k(3)), Op::Return];
        assert_eq!(opt(code), vec![Op::GetLocal(0), Op::GetLocalField(1, 3), Op::Return]);
    }

    #[test]
    fn large_operands_are_not_fused() {
        // slot 300 does not fit the 8-bit operand of the fused forms
        let code = vec![Op::GetLocal(300), Op::GetLocal(1), Op::Add, Op::Return];
        assert_eq!(opt(code.clone()), code);
        let code = vec![Op::GetLocal(300), Op::AddK(k(0)), Op::StoreLocal(300), Op::Nil, Op::Return];
        assert_eq!(opt(code), vec![Op::GetLocal(300), Op::AddK(k(0)), Op::StoreLocal(300), Op::ReturnNil]);
        // constant 70000 does not fit 16 bits
        let code = vec![Op::GetLocal(1), Op::Const(k(70_000)), Op::Lt, Op::JumpIfFalse(o(0)), Op::Nil, Op::Return];
        let out = opt(code);
        assert!(!out.iter().any(|op| op.is_wide()), "{:?}", out);
    }

    #[test]
    fn keeps_jump_targets_intact() {
        // the Pop is a jump target, so SetLocal+Pop must not fuse across it
        let code = vec![Op::True, Op::JumpIfTrue(o(1)), Op::SetLocal(0), Op::Pop, Op::Nil, Op::Return];
        let out = opt(code);
        assert!(out.contains(&Op::SetLocal(0)) && out.contains(&Op::Pop));
    }

    #[test]
    fn jump_over_a_double_instruction() {
        // a forward jump that skips a double instruction must count both words
        let code = vec![
            Op::True,
            Op::JumpIfFalse(o(4)), // -> index 6 (Nil)
            Op::GetLocal(0),
            Op::GetLocal(1),
            Op::Lt,
            Op::JumpIfFalse(o(0)),
            Op::Nil,
            Op::Return,
        ];
        let out = opt(code);
        // True, JumpIfFalse, JNotLtLL, Ext, ...
        let jf = out.iter().position(|op| matches!(op, Op::JumpIfFalse(_))).unwrap();
        let wide = out.iter().position(|op| op.is_wide()).unwrap();
        assert!(wide > jf);
        let target = jf as i32 + 1 + out[jf].jump_offset().unwrap();
        assert!(target as usize >= wide + 2, "jump must land after the double instruction");
    }
}
