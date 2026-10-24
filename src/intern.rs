//! Identifier interner.  Every identifier in the program becomes a dense
//! `Sym(u32)`, which doubles as the index into the VM's global table.

use crate::hash::FxHashMap;
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Sym(pub u32);

impl Sym {
    pub const NONE: Sym = Sym(u32::MAX);
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Default)]
pub struct Interner {
    map: FxHashMap<Rc<str>, u32>,
    strings: Vec<Rc<str>>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner::default()
    }

    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.map.get(s) {
            return Sym(id);
        }
        let id = self.strings.len() as u32;
        let rc: Rc<str> = Rc::from(s);
        self.strings.push(rc.clone());
        self.map.insert(rc, id);
        Sym(id)
    }

    pub fn lookup(&self, s: &str) -> Option<Sym> {
        self.map.get(s).map(|&i| Sym(i))
    }

    #[inline]
    pub fn get(&self, sym: Sym) -> &str {
        &self.strings[sym.idx()]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interning() {
        let mut i = Interner::new();
        let a = i.intern("foo");
        let b = i.intern("bar");
        assert_ne!(a, b);
        assert_eq!(a, i.intern("foo"));
        assert_eq!(i.get(b), "bar");
        assert_eq!(i.len(), 2);
    }
}
