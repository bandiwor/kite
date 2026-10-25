//! Bytecode disassembler (`--dump-bytecode`).

use crate::intern::{Interner, Sym};
use crate::object::*;
use crate::op::{branch_target, Op};
use crate::value::Value;
use std::fmt::Write;

pub fn disassemble(proto: Gc<Proto>, interner: &Interner) -> String {
    let mut out = String::new();
    dis(proto, interner, &mut out);
    out
}

fn const_repr(v: Value) -> String {
    if v.is_num() {
        let mut s = String::new();
        crate::text::fmt_num(v.as_num(), &mut s);
        s
    } else if let Some(s) = v.as_str() {
        format!("{:?}", s.as_str())
    } else if let Some(p) = v.as_gc::<Proto>() {
        format!("<fn {}>", p.name)
    } else if let Some(t) = v.as_gc::<StructType>() {
        format!("<struct {}>", t.name.as_str())
    } else {
        format!("{:?}", v)
    }
}

fn dis(proto: Gc<Proto>, interner: &Interner, out: &mut String) {
    let _ = writeln!(
        out,
        "== {} (arity {}, slots {}, stack {}, upvalues {}) ==",
        proto.name,
        proto.arity,
        proto.nslots,
        proto.max_stack,
        proto.upvals.len()
    );
    let mut last_line = 0;
    let mut i = 0;
    while i < proto.code.len() {
        let op = proto.code[i];
        let line = proto.source.line(proto.offsets[i]);
        let line_s = if line != last_line { format!("{:>4}", line) } else { "   |".to_string() };
        last_line = line;
        let mut note = String::new();
        let k = |k: usize| const_repr(proto.consts[k]);
        match op {
            Op::Const(a) | Op::AddK(a) | Op::SubK(a) | Op::MulK(a) | Op::DivK(a) | Op::RemK(a) | Op::GetField(a)
            | Op::SetField(a) | Op::StoreField(a) | Op::InitField(a) | Op::GetMethod(a) | Op::Closure(a)
            | Op::Struct(a) | Op::DefMethod(a) => note = k(a.get() as usize),
            Op::AddLocalK(_, a) | Op::GetLocalField(_, a) => note = k(a as usize),
            Op::JNotEqLK(_, a) | Op::JNotNeLK(_, a) | Op::JNotLtLK(_, a) | Op::JNotLeLK(_, a) | Op::JNotGtLK(_, a)
            | Op::JNotGeLK(_, a) => note = k(a as usize),
            Op::GetGlobal(s) | Op::SetGlobal(s) | Op::StoreGlobal(s) | Op::DefGlobal(s) => {
                note = interner.get(Sym(s.get())).to_string();
            }
            _ => {}
        }
        if let Some(target) = branch_target(&proto.code, i) {
            let _ = write!(note, "{}-> {:04}", if note.is_empty() { "" } else { "  " }, target);
        }
        // a double instruction prints as one line: `Op(..) +off`
        let shown = if op.is_wide() { format!("{:?} +{}", op, proto.code[i + 1].ext()) } else { format!("{:?}", op) };
        if note.is_empty() {
            let _ = writeln!(out, "{:04} {} {}", i, line_s, shown);
        } else {
            let _ = writeln!(out, "{:04} {} {:<28} ; {}", i, line_s, shown, note);
        }
        i += op.width();
    }
    for c in &proto.consts {
        if let Some(p) = c.as_gc::<Proto>() {
            out.push('\n');
            dis(p, interner, out);
        }
    }
}
