//! Bytecode disassembler (`--dump-bytecode`).

use crate::intern::{Interner, Sym};
use crate::object::*;
use crate::op::Op;
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
    for (i, op) in proto.code.iter().enumerate() {
        let line = proto.source.line(proto.offsets[i]);
        let line_s = if line != last_line { format!("{:>4}", line) } else { "   |".to_string() };
        last_line = line;
        let mut note = String::new();
        match *op {
            Op::Const(k)
            | Op::AddK(k)
            | Op::SubK(k)
            | Op::MulK(k)
            | Op::DivK(k)
            | Op::RemK(k)
            | Op::GetField(k)
            | Op::SetField(k)
            | Op::StoreField(k)
            | Op::InitField(k)
            | Op::GetMethod(k)
            | Op::Closure(k)
            | Op::Struct(k)
            | Op::DefMethod(k)
            | Op::AddLocalK(_, k)
            | Op::GetLocalField(_, k) => {
                note = const_repr(proto.consts[k as usize]);
            }
            Op::JNotEqLK(_, k, _)
            | Op::JNotNeLK(_, k, _)
            | Op::JNotLtLK(_, k, _)
            | Op::JNotLeLK(_, k, _)
            | Op::JNotGtLK(_, k, _)
            | Op::JNotGeLK(_, k, _) => {
                note = const_repr(proto.consts[k as usize]);
            }
            Op::GetGlobal(s) | Op::SetGlobal(s) | Op::StoreGlobal(s) | Op::DefGlobal(s) => {
                note = interner.get(Sym(s)).to_string();
            }
            _ => {}
        }
        if let Some(off) = op.jump_offset() {
            let target = i as i64 + 1 + off as i64;
            let _ = write!(note, "{}-> {:04}", if note.is_empty() { "" } else { "  " }, target);
        }
        if note.is_empty() {
            let _ = writeln!(out, "{:04} {} {:?}", i, line_s, op);
        } else {
            let _ = writeln!(out, "{:04} {} {:<28} ; {}", i, line_s, format!("{:?}", op), note);
        }
    }
    for c in &proto.consts {
        if let Some(p) = c.as_gc::<Proto>() {
            out.push('\n');
            dis(p, interner, out);
        }
    }
}
