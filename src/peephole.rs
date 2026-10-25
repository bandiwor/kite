//! Peephole pass, run once per function after code generation.
//!
//! The register compiler already writes results straight into their destinations, so
//! there is little left to fuse.  What remains is control-flow clean-up:
//!  * jumps are threaded (a branch to an unconditional jump goes to its target; an
//!    unconditional jump to a `Return` becomes that `Return`),
//!  * code after `Jump`/`Return`/`MatchFail` that nothing targets is dropped,
//!  * `Jump(0)` and `Move(a, a)` are dropped.
//!
//! No instruction changes its width, so removing instructions only shifts positions:
//! branches are re-encoded through an old -> new word index table.  The scratch
//! buffers are reused between functions.

use crate::op::{branch_target, Op, I24};

#[derive(Default)]
pub struct Scratch {
    is_target: Vec<bool>,
    keep: Vec<bool>,
    new_pos: Vec<u32>,
}

/// Convenience wrapper with throw-away buffers (tests, one-off use).
pub fn optimize(code: Vec<Op>, offsets: Vec<u32>) -> Result<(Vec<Op>, Vec<u32>), String> {
    optimize_with(&mut Scratch::default(), code, offsets)
}

pub fn optimize_with(
    sc: &mut Scratch,
    mut code: Vec<Op>,
    offsets: Vec<u32>,
) -> Result<(Vec<Op>, Vec<u32>), String> {
    let n = code.len();
    let Scratch { is_target, keep, new_pos } = sc;
    is_target.clear();
    is_target.resize(n + 1, false);

    // 1. thread jumps and find the branch targets
    let mut i = 0;
    while i < n {
        if let Some(t) = thread(&mut code, i) {
            if t <= n {
                is_target[t] = true;
            }
        }
        i += code[i].width();
    }

    // 2. decide what stays
    keep.clear();
    keep.resize(n, true);
    let mut dead = false;
    let mut i = 0;
    while i < n {
        let op = code[i];
        let w = op.width();
        if is_target[i] {
            dead = false;
        }
        let drop = dead || matches!(op, Op::Jump(o) if o.get() == 0) || matches!(op, Op::Move(a, b) if a == b);
        if drop {
            keep[i..i + w].fill(false);
        } else if op.is_terminator() {
            dead = true;
        }
        i += w;
    }

    // 3. new word positions
    new_pos.clear();
    new_pos.resize(n + 1, 0);
    let mut words = 0u32;
    for i in 0..n {
        new_pos[i] = words;
        words += keep[i] as u32;
    }
    new_pos[n] = words;

    // 4. write back, re-encoding the branches
    let mut out = Vec::with_capacity(words as usize);
    let mut out_offsets = Vec::with_capacity(words as usize);
    let too_large = || "function is too large (a jump spans more than 8M instructions)".to_string();
    let mut i = 0;
    while i < n {
        let op = code[i];
        let w = op.width();
        if keep[i] {
            let here = new_pos[i] as i64;
            if let Some(t) = branch_target(&code, i) {
                let rel = new_pos[t.min(n)] as i64 - (here + w as i64);
                let o = i32::try_from(rel).ok().and_then(I24::new).ok_or_else(too_large)?;
                if op.is_branch_wide() {
                    out.push(op);
                    out.push(Op::Ext(o));
                } else {
                    out.push(op.with_jump_offset(o.get()).expect("single-word branch"));
                }
            } else {
                out.push(op);
                if w == 2 {
                    out.push(code[i + 1]);
                }
            }
            out_offsets.push(offsets[i]);
            if w == 2 {
                out_offsets.push(offsets[i + 1]);
            }
        }
        i += w;
    }
    Ok((out, out_offsets))
}

/// Resolve the destination of the branch at `i` through chains of unconditional jumps.
/// Returns the final target, or `None` when the instruction was replaced by a return
/// (or is not a branch).
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
        if matches!(r, Op::Return(_) | Op::ReturnNil) {
            code[i] = r;
            return None;
        }
    }
    if t != t0 {
        let wide = code[i].is_branch_wide();
        let rel = t as i64 - (i as i64 + 1 + wide as i64);
        match i32::try_from(rel).ok().and_then(I24::new) {
            Some(o) if wide => code[i + 1] = Op::Ext(o),
            Some(o) => code[i] = code[i].with_jump_offset(o.get()).unwrap(),
            None => t = t0, // does not fit; keep the original destination
        }
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(v: i32) -> I24 {
        I24::new(v).unwrap()
    }

    fn opt(code: Vec<Op>) -> Vec<Op> {
        let n = code.len();
        optimize(code, vec![0; n]).unwrap().0
    }

    #[test]
    fn dead_code_after_jump_and_return_is_dropped() {
        // 0: Jump -> 2 ; 1: dead ; 2: Return
        let code = vec![Op::Jump(o(1)), Op::LoadTrue(0), Op::Return(1)];
        assert_eq!(opt(code), vec![Op::Return(1)]);
        // code after a return is dead unless something jumps to it
        let code = vec![Op::Return(0), Op::LoadTrue(1), Op::Return(1)];
        assert_eq!(opt(code), vec![Op::Return(0)]);
    }

    #[test]
    fn jumps_are_threaded_and_returns_inlined() {
        // 0: Jump -> 2 ; 1: Move ; 2: Jump -> 4 ; 3: Move ; 4: ReturnNil
        let code = vec![Op::Jump(o(1)), Op::Move(1, 2), Op::Jump(o(1)), Op::Move(1, 2), Op::ReturnNil];
        assert_eq!(opt(code), vec![Op::ReturnNil]);
    }

    #[test]
    fn useless_moves_and_zero_jumps_vanish() {
        let code = vec![Op::Move(3, 3), Op::Jump(o(0)), Op::LoadTrue(1), Op::Return(1)];
        assert_eq!(opt(code), vec![Op::LoadTrue(1), Op::Return(1)]);
    }

    #[test]
    fn double_instructions_keep_their_targets() {
        // 0: JLt(1,2) +2 -> 4 ; 2: LoadTrue ; 3: Move(5,5) (dropped) ; 4: Return
        let code = vec![Op::JLt(1, 2), Op::Ext(o(2)), Op::LoadTrue(0), Op::Move(5, 5), Op::Return(0)];
        let out = opt(code);
        assert_eq!(out, vec![Op::JLt(1, 2), Op::Ext(o(1)), Op::LoadTrue(0), Op::Return(0)]);
        crate::op::verify(&out).unwrap();
    }

    #[test]
    fn a_jump_into_removed_code_lands_on_the_next_instruction() {
        // 0: JTrue(0) -> 3 (a dropped Move) ; 2: LoadTrue ; 3: Move(5, 5) ; 4: Return
        let code = vec![Op::JTrue(0), Op::Ext(o(1)), Op::LoadTrue(1), Op::Move(5, 5), Op::Return(1)];
        let out = opt(code);
        assert_eq!(out, vec![Op::JTrue(0), Op::Ext(o(1)), Op::LoadTrue(1), Op::Return(1)]);
    }

    #[test]
    fn loop_with_double_instruction_is_remapped() {
        // a loop whose back edge crosses a removable instruction
        let code = vec![
            Op::LoadI(0, 0),
            Op::Move(2, 2),
            Op::JNotLtK(0, 1),
            Op::Ext(o(2)),
            Op::AddK(0, 0, 0),
            Op::Jump(o(-5)),
            Op::Return(0),
        ];
        let out = opt(code);
        // positions: 0 LoadI, 1-2 JNotLtK+Ext, 3 AddK, 4 Jump -> 1, 5 Return
        assert_eq!(out[4], Op::Jump(o(-4)));
        assert_eq!(out[2], Op::Ext(o(2)));
        crate::op::verify(&out).unwrap();
    }
}
