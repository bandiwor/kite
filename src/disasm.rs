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
    } else if let Some(t) = v.as_gc::<StructType>() {
        format!("<struct {}>", t.name.as_str())
    } else {
        format!("{:?}", v)
    }
}

fn dis(proto: Gc<Proto>, interner: &Interner, out: &mut String) {
    let _ = writeln!(
        out,
        "== {} (arity {}, locals {}, registers {}, upvalues {}) ==",
        proto.name,
        proto.arity,
        proto.nslots,
        proto.nregs,
        proto.upvals.len()
    );
    let mut last_line = 0;
    let mut i = 0;
    while i < proto.code.len() {
        let op = proto.code[i];
        let line = proto.source.line(proto.offsets[i]);
        let line_s = if line != last_line { format!("{:>4}", line) } else { "   |".to_string() };
        last_line = line;
        let k = |k: usize| proto.consts.get(k).map_or_else(|| "?".to_string(), |&v| const_repr(v));
        let ext = if op.is_wide() { Some(proto.code[i + 1]) } else { None };
        let payload = ext.map_or(0, |e| e.ext_u());
        let global = |s: u32| interner.get(Sym(s)).to_string();
        let mut note = match op {
            Op::LoadK(_, a)
            | Op::JEqK(_, a)
            | Op::JNeK(_, a)
            | Op::JLtK(_, a)
            | Op::JNotLtK(_, a)
            | Op::JLeK(_, a)
            | Op::JNotLeK(_, a)
            | Op::JGtK(_, a)
            | Op::JNotGtK(_, a)
            | Op::JGeK(_, a)
            | Op::JNotGeK(_, a)
            | Op::Struct(_, a)
            | Op::Import(_, a) => k(a as usize),
            Op::AddK(_, _, a)
            | Op::SubK(_, _, a)
            | Op::MulK(_, _, a)
            | Op::DivK(_, _, a)
            | Op::RemK(_, _, a)
            | Op::GetField(_, _, a)
            | Op::SetField(_, a, _)
            | Op::InitField(_, a, _)
            | Op::GetMethod(_, _, a) => k(a as usize),
            Op::LoadKW(_)
            | Op::GetFieldW(..)
            | Op::SetFieldW(..)
            | Op::InitFieldW(..)
            | Op::GetMethodW(..)
            | Op::DefMethod(..)
            | Op::GetExport(..) => k(payload as usize),
            Op::IsVariant(..) => format!("{} / {}", k((payload & 0xFFFF) as usize), payload >> 16),
            Op::GetGlobal(_, s) | Op::SetGlobal(_, s) | Op::DefGlobal(_, s) => global(s as u32),
            Op::GetGlobalW(_) | Op::SetGlobalW(_) | Op::DefGlobalW(_) => global(payload),
            Op::Closure(_, p) => {
                proto.protos.get(p as usize).map_or_else(|| "?".to_string(), |q| format!("<fn {}>", q.name))
            }
            _ => String::new(),
        };
        if let Some(target) = branch_target(&proto.code, i) {
            let _ = write!(note, "{}-> {:04}", if note.is_empty() { "" } else { "  " }, target);
        }
        // a double instruction prints as one line
        let shown = match ext {
            Some(e) if op.is_branch_wide() => format!("{:?} +{}", op, e.ext()),
            Some(e) => format!("{:?} #{}", op, e.ext_u()),
            None => format!("{:?}", op),
        };
        if note.is_empty() {
            let _ = writeln!(out, "{:04} {} {}", i, line_s, shown);
        } else {
            let _ = writeln!(out, "{:04} {} {:<30} ; {}", i, line_s, shown, note);
        }
        i += op.width();
    }
    for p in &proto.protos {
        out.push('\n');
        dis(*p, interner, out);
    }
}
