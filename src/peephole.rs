//! Peephole optimiser, run once per function after code generation.
//!
//! One left-to-right pass copies the code into an output buffer and, every
//! time an instruction is appended, tries to fuse the *tail* of that buffer
//! into a superinstruction (`Set+Pop -> Store`, compare+jump, `x += k`, ...).
//! Because fusion looks at the output, chains compose in the same pass
//! (`SetLocal;Pop` -> `StoreLocal`, then `GetLocal;AddK;StoreLocal` -> `AddLocalK`).
//! The pass also
//!  * threads jumps (a jump to an unconditional jump goes to its target; a
//!    jump straight to a `Return` becomes that `Return`),
//!  * drops code that follows `Jump`/`Return` and that nothing targets,
//!  * drops `Jump(0)`.
//!
//! A window is only fused when none of its interior instructions is a jump
//! target.  Branches are remapped afterwards through an old -> new index table.
//! The scratch buffers are reused between functions.

use crate::op::Op;
use crate::value::Value;

/// An output instruction together with the bookkeeping needed to remap branches.
#[derive(Clone, Copy)]
struct Entry {
    op: Op,
    /// Source offset of the instruction (for error positions).
    off: u32,
    /// Old index of the first instruction this entry replaces.
    first: u32,
    /// Old absolute branch target, if `op` branches.
    target: u32,
}

#[derive(Default)]
pub struct Scratch {
    is_target: Vec<bool>,
    out: Vec<Entry>,
    map: Vec<u32>,
}

/// Convenience wrapper with throw-away buffers (tests, one-off use).
pub fn optimize(code: Vec<Op>, offsets: Vec<u32>, consts: &[Value]) -> (Vec<Op>, Vec<u32>) {
    optimize_with(&mut Scratch::default(), code, offsets, consts)
}

pub fn optimize_with(sc: &mut Scratch, mut code: Vec<Op>, mut offsets: Vec<u32>, consts: &[Value]) -> (Vec<Op>, Vec<u32>) {
    pass(sc, &mut code, &mut offsets, consts);
    (code, offsets)
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

fn jnot(c: Cmp, o: i32) -> Op {
    match c {
        Cmp::Eq => Op::JNotEq(o),
        Cmp::Ne => Op::JNotNe(o),
        Cmp::Lt => Op::JNotLt(o),
        Cmp::Le => Op::JNotLe(o),
        Cmp::Gt => Op::JNotGt(o),
        Cmp::Ge => Op::JNotGe(o),
    }
}

fn jnot_ll(c: Cmp, a: u16, b: u16, o: i16) -> Op {
    match c {
        Cmp::Eq => Op::JNotEqLL(a, b, o),
        Cmp::Ne => Op::JNotNeLL(a, b, o),
        Cmp::Lt => Op::JNotLtLL(a, b, o),
        Cmp::Le => Op::JNotLeLL(a, b, o),
        Cmp::Gt => Op::JNotGtLL(a, b, o),
        Cmp::Ge => Op::JNotGeLL(a, b, o),
    }
}

fn jnot_lk(c: Cmp, a: u16, k: u16, o: i16) -> Op {
    match c {
        Cmp::Eq => Op::JNotEqLK(a, k, o),
        Cmp::Ne => Op::JNotNeLK(a, k, o),
        Cmp::Lt => Op::JNotLtLK(a, k, o),
        Cmp::Le => Op::JNotLeLK(a, k, o),
        Cmp::Gt => Op::JNotGtLK(a, k, o),
        Cmp::Ge => Op::JNotGeLK(a, k, o),
    }
}

fn is_num_const(consts: &[Value], k: u32) -> bool {
    consts.get(k as usize).is_some_and(|v| v.is_num())
}

fn is_pure_push(op: Op) -> bool {
    matches!(op, Op::Nil | Op::True | Op::False | Op::Const(_) | Op::GetLocal(_) | Op::GetUpval(_))
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

/// Try to rewrite the tail `t` of the output buffer (`t.len() >= 2`).  Returns
/// the rewrite and how many trailing ops it consumed.  `ok(width)` says whether
/// a window of that many ops may be fused (no jump target inside it).
fn match_tail(entries: &[Entry], consts: &[Value], ok: &dyn Fn(usize) -> bool) -> Option<(Rewrite, usize)> {
    use Op::*;
    let n = entries.len();
    let t = |k: usize| entries[n - k].op;
    // ---- 4 wide: local OP constant, then branch
    if n >= 4 && ok(4) {
        if let [GetLocal(a), Const(k), c, JumpIfFalse(o)] = [t(4), t(3), t(2), t(1)] {
            if let (Some(c), Ok(o16), Ok(k16)) = (as_cmp(c), i16::try_from(o), u16::try_from(k)) {
                return Some((Rewrite::One(jnot_lk(c, a, k16, o16)), 4));
            }
        }
    }
    // ---- 3 wide
    if n >= 3 && ok(3) {
        match [t(3), t(2), t(1)] {
            [GetLocal2(a, b), c, JumpIfFalse(o)] => {
                if let (Some(c), Ok(o16)) = (as_cmp(c), i16::try_from(o)) {
                    return Some((Rewrite::One(jnot_ll(c, a, b, o16)), 3));
                }
            }
            [GetLocal(s), AddK(k), StoreLocal(d)] if s == d => return Some((Rewrite::One(AddLocalK(s, k)), 3)),
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
        (GetLocal(s), GetField(k)) => GetLocalField(s, k),
        // `a; b; .field` : the field read wants `b` on its own
        (GetLocal2(x, y), GetField(k)) => return Some((Rewrite::Two(GetLocal(x), GetLocalField(y, k)), 2)),
        (GetLocal2(x, y), Add) => AddLL(x, y),
        (GetLocal2(x, y), Sub) => SubLL(x, y),
        (GetLocal2(x, y), Mul) => MulLL(x, y),
        (GetLocal(x), GetLocal(y)) => GetLocal2(x, y),
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

/// Resolve a branch's final destination through chains of unconditional jumps.
fn thread(code: &mut [Op], i: usize) {
    let Some(off) = code[i].jump_offset() else { return };
    let mut t = (i as i64 + 1 + off as i64) as usize;
    let mut hops = 0;
    while t < code.len() && hops < 8 {
        match code[t] {
            Op::Jump(o) => {
                t = (t as i64 + 1 + o as i64) as usize;
                hops += 1;
            }
            _ => break,
        }
    }
    if hops > 0 {
        let rel = t as i64 - (i as i64 + 1);
        if let Some(op) = code[i].with_jump_offset(rel as i32) {
            code[i] = op;
        }
    }
    // an unconditional jump straight to a return is just that return
    if let (Op::Jump(_), Some(&r)) = (code[i], code.get(t)) {
        if matches!(r, Op::Return | Op::ReturnNil | Op::ReturnLocal(_)) {
            code[i] = r;
        }
    }
}

/// The whole optimisation: threading, target marking, and one fusing sweep.
fn pass(sc: &mut Scratch, code: &mut Vec<Op>, offsets: &mut Vec<u32>, consts: &[Value]) {
    let n = code.len();
    let Scratch { is_target, out, map } = sc;
    is_target.clear();
    is_target.resize(n + 1, false);

    for i in 0..n {
        // `Call; Return` becomes `TailCall; Return`.  The Return stays: it is what
        // runs when the callee is a native (or anything that is not a plain closure).
        if let Op::Call(argc) = code[i] {
            if code.get(i + 1) == Some(&Op::Return) {
                code[i] = Op::TailCall(argc);
            }
        }
        thread(code, i);
        if let Some(off) = code[i].jump_offset() {
            let t = (i as i64 + 1 + off as i64) as usize;
            if t <= n {
                is_target[t] = true;
            }
        }
    }

    out.clear();
    map.clear();
    map.resize(n + 1, 0);

    let mut dead = false;
    for i in 0..n {
        if is_target[i] {
            dead = false;
        }
        if dead || code[i] == Op::Jump(0) {
            map[i] = out.len() as u32;
            continue;
        }
        let op = code[i];
        map[i] = out.len() as u32;
        let target = op.jump_offset().map_or(0, |o| (i as i64 + 1 + o as i64) as u32);
        out.push(Entry { op, off: offsets[i], first: i as u32, target });

        // fuse the tail as long as something matches
        let mut vanished = false;
        while out.len() >= 2 && can_end_window(out[out.len() - 1].op) {
            // a window may only be fused when no old instruction inside it (other
            // than its first) is a jump target - including removed `Jump(0)`s
            let ok = |width: usize| {
                let lo = out[out.len() - width].first as usize + 1;
                !(lo..=i).any(|o| is_target[o])
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
            map[first_old as usize..=i].fill(start as u32);
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
    }
    map[n] = out.len() as u32;

    // remap branches and write back
    code.clear();
    offsets.clear();
    for (j, e) in out.iter().enumerate() {
        let mut op = e.op;
        if op.jump_offset().is_some() {
            let new_target = map[(e.target as usize).min(n)] as i64;
            let rel = new_target - (j as i64 + 1);
            op = op.with_jump_offset(rel as i32).expect("peephole: jump out of range");
        }
        code.push(op);
        offsets.push(e.off);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(code: Vec<Op>) -> Vec<Op> {
        let n = code.len();
        optimize(code, vec![0; n], &[Value::num(1.0)]).0
    }

    #[test]
    fn store_fusion() {
        assert_eq!(opt(vec![Op::SetLocal(1), Op::Pop, Op::Nil, Op::Return]), vec![Op::StoreLocal(1), Op::ReturnNil]);
    }

    #[test]
    fn dead_code_and_threading() {
        // 0: Jump -> 2 ; 1: dead ; 2: Jump -> 4 ; 3: dead ; 4: Return
        let code = vec![Op::Jump(1), Op::Nil, Op::Jump(1), Op::True, Op::Return];
        assert_eq!(opt(code), vec![Op::Return]);
    }

    #[test]
    fn compare_jump() {
        let code = vec![Op::GetLocal(0), Op::GetLocal(1), Op::Lt, Op::JumpIfFalse(2), Op::Nil, Op::Pop, Op::Nil, Op::Return];
        let out = opt(code);
        assert_eq!(out[0], Op::JNotLtLL(0, 1, 0));
    }

    #[test]
    fn increment_chain_in_one_pass() {
        let code = vec![Op::GetLocal(2), Op::Const(0), Op::Add, Op::SetLocal(2), Op::Pop, Op::Nil, Op::Return];
        assert_eq!(opt(code), vec![Op::AddLocalK(2, 0), Op::ReturnNil]);
    }

    #[test]
    fn field_read_prefers_second_local() {
        let code = vec![Op::GetLocal(0), Op::GetLocal(1), Op::GetField(3), Op::Return];
        assert_eq!(opt(code), vec![Op::GetLocal(0), Op::GetLocalField(1, 3), Op::Return]);
    }

    #[test]
    fn keeps_jump_targets_intact() {
        // the Pop is a jump target, so SetLocal+Pop must not fuse across it
        let code = vec![Op::True, Op::JumpIfTrue(1), Op::SetLocal(0), Op::Pop, Op::Nil, Op::Return];
        let out = opt(code);
        assert!(out.contains(&Op::SetLocal(0)) && out.contains(&Op::Pop));
    }

    #[test]
    fn jump_into_a_fused_window_start_is_remapped() {
        // loop: 0: GetLocal0 1: Const 2: Lt 3: JumpIfFalse(+3) 4: GetLocal 5: Pop 6: Jump(-7) 7: Nil 8: Return
        let code = vec![
            Op::GetLocal(0),
            Op::Const(0),
            Op::Lt,
            Op::JumpIfFalse(3),
            Op::GetLocal(0),
            Op::Pop,
            Op::Jump(-7),
            Op::Nil,
            Op::Return,
        ];
        let out = opt(code);
        // the back jump must land on the fused compare at index 0
        assert_eq!(out[0], Op::JNotLtLK(0, 0, 1));
        assert_eq!(out[1], Op::Jump(-2));
    }
}
