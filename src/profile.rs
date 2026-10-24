//! Opcode frequency profiler (cargo feature `op-profile`).
//!
//! Records how often each opcode and each consecutive pair of opcodes runs and
//! prints the top entries when the process exits.  Used to decide which
//! patterns deserve a superinstruction in the peephole optimiser.

use crate::op::Op;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Default)]
struct Profile {
    single: HashMap<&'static str, u64>,
    pairs: HashMap<(&'static str, &'static str), u64>,
    prev: Option<&'static str>,
    names: HashMap<u8, &'static str>,
}

thread_local! {
    static PROFILE: RefCell<Profile> = RefCell::new(Profile::default());
}

fn tag(op: &Op) -> u8 {
    // the discriminant is the first byte of the 8-byte instruction
    unsafe { *(op as *const Op as *const u8) }
}

pub fn record(op: &Op) {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        let t = tag(op);
        let name = *p.names.entry(t).or_insert_with(|| {
            let s = format!("{:?}", op);
            let end = s.find('(').unwrap_or(s.len());
            Box::leak(s[..end].to_string().into_boxed_str())
        });
        *p.single.entry(name).or_insert(0) += 1;
        if let Some(prev) = p.prev {
            *p.pairs.entry((prev, name)).or_insert(0) += 1;
        }
        p.prev = Some(name);
    });
}

pub fn report() {
    PROFILE.with(|p| {
        let p = p.borrow();
        let total: u64 = p.single.values().sum();
        eprintln!("--- opcode profile: {} instructions executed", total);
        let mut s: Vec<_> = p.single.iter().collect();
        s.sort_by(|a, b| b.1.cmp(a.1));
        for (n, c) in s.iter().take(15) {
            eprintln!("{:>12}  {:5.1}%  {}", c, **c as f64 * 100.0 / total as f64, n);
        }
        eprintln!("--- hottest pairs");
        let mut s: Vec<_> = p.pairs.iter().collect();
        s.sort_by(|a, b| b.1.cmp(a.1));
        for ((a, b), c) in s.iter().take(15) {
            eprintln!("{:>12}  {:5.1}%  {} -> {}", c, **c as f64 * 100.0 / total as f64, a, b);
        }
    });
}
