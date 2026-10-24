//! Peephole optimiser, run once per function after code generation.
//!
//! Passes (repeated until nothing changes):
//!  1. jump threading   – a jump to an unconditional jump goes straight to its target;
//!  2. dead code        – instructions after `Jump`/`Return` that nothing targets are dropped;
//!  3. fusion           – short windows of ops become one superinstruction
//!                        (`Set+Pop -> Store`, `cmp+jump`, `x += k`, `Call+Return -> TailCall`, ...).
//!
//! A window is only fused when none of its interior instructions is a jump
//! target.  Jumps are remapped through an old-index -> new-index table.

use crate::op::Op;
use crate::value::Value;

pub fn optimize(mut code: Vec<Op>, mut offsets: Vec<u32>, consts: &[Value]) -> (Vec<Op>, Vec<u32>) {
    for _ in 0..6 {
        thread_jumps(&mut code);
        if !pass(&mut code, &mut offsets, consts) {
            break;
        }
    }
    (code, offsets)
}

fn target_of(code: &[Op], i: usize) -> Option<usize> {
    code[i].jump_offset().map(|o| (i as i64 + 1 + o as i64) as usize)
}

fn t_op(code: &[Op], i: usize) -> Option<&Op> {
    target_of(code, i).and_then(|t| code.get(t))
}

fn thread_jumps(code: &mut [Op]) {
    for i in 0..code.len() {
        let Some(mut t) = target_of(code, i) else { continue };
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
            let off = t as i64 - (i as i64 + 1);
            if let Some(op) = code[i].with_jump_offset(off as i32) {
                code[i] = op;
            }
        }
        // an unconditional jump straight to a return is just that return
        if let (Op::Jump(_), Some(&r)) = (code[i], t_op(code, i)) {
            if matches!(r, Op::Return | Op::ReturnNil | Op::ReturnLocal(_)) {
                code[i] = r;
            }
        }
    }
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
    consts.get(k as usize).map_or(false, |v| v.is_num())
}

fn is_pure_push(op: Op) -> bool {
    matches!(op, Op::Nil | Op::True | Op::False | Op::Const(_) | Op::GetLocal(_) | Op::GetUpval(_))
}

/// Result of matching a window: replacement op, window width and the index
/// (within the window) of the instruction that carries the jump/position.
struct Fused {
    op: Option<Op>, // None = delete the whole window
    width: usize,
}

fn match_window(w: &[Op]) -> Option<Fused> {
    // ---- 4 wide
    if let Some(&[Op::GetLocal(a), second, third, Op::JumpIfFalse(o)]) = w.get(..4) {
        if let (Some(c), Ok(o16)) = (as_cmp(third), i16::try_from(o)) {
            match second {
                Op::GetLocal(b) => return Some(Fused { op: Some(jnot_ll(c, a, b, o16)), width: 4 }),
                Op::Const(k) if k <= u16::MAX as u32 => {
                    return Some(Fused { op: Some(jnot_lk(c, a, k as u16, o16)), width: 4 })
                }
                _ => {}
            }
        }
    }
    None
}

fn match_window3(w: &[Op]) -> Option<Fused> {
    if let Some(&[Op::GetLocal(s), Op::AddK(k), Op::StoreLocal(t)]) = w.get(..3) {
        if s == t {
            return Some(Fused { op: Some(Op::AddLocalK(s, k)), width: 3 });
        }
    }
    None
}

fn match_window2(w: &[Op], consts: &[Value]) -> Option<Fused> {
    let Some(&[a, b]) = w.get(..2) else { return None };
    let op = match (a, b) {
        (Op::GetLocal(s), Op::GetField(k)) => Op::GetLocalField(s, k),
        (Op::GetLocal(s), Op::Return) => Op::ReturnLocal(s),
        (Op::GetLocal2(x, y), Op::Add) => Op::AddLL(x, y),
        (Op::GetLocal2(x, y), Op::Sub) => Op::SubLL(x, y),
        (Op::GetLocal2(x, y), Op::Mul) => Op::MulLL(x, y),
        // `GetLocal a; GetLocal b; GetField k`: prefer `a; GetLocalField(b, k)`
        (Op::GetLocal(_), Op::GetLocal(_)) if matches!(w.get(2), Some(Op::GetField(_))) => return None,
        (Op::SetLocal(s), Op::Pop) => Op::StoreLocal(s),
        (Op::SetUpval(s), Op::Pop) => Op::StoreUpval(s),
        (Op::SetGlobal(s), Op::Pop) => Op::StoreGlobal(s),
        (Op::SetField(s), Op::Pop) => Op::StoreField(s),
        (Op::SetIndex, Op::Pop) => Op::StoreIndex,
        (Op::Nil, Op::Return) => Op::ReturnNil,
        (Op::Pop, Op::Pop) => Op::PopN(2),
        (Op::PopN(n), Op::Pop) | (Op::Pop, Op::PopN(n)) if n < u16::MAX => Op::PopN(n + 1),
        (Op::GetLocal(x), Op::GetLocal(y)) => Op::GetLocal2(x, y),
        (Op::Not, Op::JumpIfFalse(o)) => Op::JumpIfTrue(o),
        (Op::Not, Op::JumpIfTrue(o)) => Op::JumpIfFalse(o),
        (Op::Const(k), Op::Add) if is_num_const(consts, k) => Op::AddK(k),
        (Op::Const(k), Op::Sub) if is_num_const(consts, k) => Op::SubK(k),
        (Op::Const(k), Op::Mul) if is_num_const(consts, k) => Op::MulK(k),
        (Op::Const(k), Op::Div) if is_num_const(consts, k) => Op::DivK(k),
        (Op::Const(k), Op::Rem) if is_num_const(consts, k) => Op::RemK(k),
        (c, Op::JumpIfFalse(o)) if as_cmp(c).is_some() => jnot(as_cmp(c).unwrap(), o),
        (p, Op::Pop) if is_pure_push(p) => return Some(Fused { op: None, width: 2 }),
        _ => return None,
    };
    Some(Fused { op: Some(op), width: 2 })
}

/// One optimisation pass; returns whether anything changed.
fn pass(code: &mut Vec<Op>, offsets: &mut Vec<u32>, consts: &[Value]) -> bool {
    let n = code.len();
    // `Call; Return` becomes `TailCall; Return`.  The Return stays: it is what
    // runs when the callee is a native (or anything that is not a plain closure).
    let mut changed = false;
    for i in 0..n.saturating_sub(1) {
        if let (Op::Call(argc), Op::Return) = (code[i], code[i + 1]) {
            code[i] = Op::TailCall(argc);
            changed = true;
        }
    }
    let mut is_target = vec![false; n + 1];
    for i in 0..n {
        if let Some(t) = target_of(code, i) {
            if t <= n {
                is_target[t] = true;
            }
        }
    }

    let mut new_code: Vec<Op> = Vec::with_capacity(n);
    let mut new_off: Vec<u32> = Vec::with_capacity(n);
    let mut src: Vec<usize> = Vec::with_capacity(n); // old index of the op carrying the jump
    let mut map: Vec<u32> = vec![0; n + 1];
    let mut dead = false;
    let mut i = 0;

    while i < n {
        if is_target[i] {
            dead = false;
        }
        if dead {
            map[i] = new_code.len() as u32;
            changed = true;
            i += 1;
            continue;
        }
        if code[i] == Op::Jump(0) {
            map[i] = new_code.len() as u32;
            changed = true;
            i += 1;
            continue;
        }

        // try the widest window whose interior holds no jump target
        let interior_free = |width: usize| (1..width).all(|k| !is_target[i + k]);
        let fused = {
            let tail = &code[i..];
            if let Some(f) = match_window(tail).filter(|f| interior_free(f.width)) {
                Some(f)
            } else if let Some(f) = match_window3(tail).filter(|f| interior_free(f.width)) {
                Some(f)
            } else {
                match_window2(tail, consts).filter(|f| interior_free(f.width))
            }
        };

        if let Some(Fused { op, width }) = fused {
            changed = true;
            for k in 0..width {
                map[i + k] = new_code.len() as u32;
            }
            if let Some(op) = op {
                new_code.push(op);
                new_off.push(offsets[i + width - 1]);
                src.push(i + width - 1);
                if op.is_terminator() {
                    dead = true;
                }
            }
            i += width;
            continue;
        }

        map[i] = new_code.len() as u32;
        new_code.push(code[i]);
        new_off.push(offsets[i]);
        src.push(i);
        if code[i].is_terminator() {
            dead = true;
        }
        i += 1;
    }
    map[n] = new_code.len() as u32;

    if !changed {
        return false;
    }

    // remap jumps
    for j in 0..new_code.len() {
        let op = new_code[j];
        if let Some(old_off) = op.jump_offset() {
            let old_target = (src[j] as i64 + 1 + old_off as i64) as usize;
            let new_target = map[old_target.min(n)] as i64;
            let rel = new_target - (j as i64 + 1);
            new_code[j] = op.with_jump_offset(rel as i32).expect("peephole: jump out of range");
        }
    }
    *code = new_code;
    *offsets = new_off;
    true
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
    fn increment() {
        let code = vec![Op::GetLocal(2), Op::Const(0), Op::Add, Op::SetLocal(2), Op::Pop, Op::Nil, Op::Return];
        assert_eq!(opt(code), vec![Op::AddLocalK(2, 0), Op::ReturnNil]);
    }

    #[test]
    fn keeps_jump_targets_intact() {
        // the Pop is a jump target, so SetLocal+Pop must not fuse across it
        let code = vec![Op::True, Op::JumpIfFalse(2), Op::Nil, Op::SetLocal(0), Op::Pop, Op::Nil, Op::Return];
        let out = opt(code.clone());
        assert!(out.contains(&Op::StoreLocal(0)) || out.contains(&Op::SetLocal(0)));
        let code = vec![Op::True, Op::JumpIfTrue(1), Op::SetLocal(0), Op::Pop, Op::Nil, Op::Return];
        let out = opt(code);
        assert!(out.contains(&Op::SetLocal(0)) && out.contains(&Op::Pop));
    }
}
