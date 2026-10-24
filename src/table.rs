//! The `table` type: a hash map with an array part.
//!
//! * Keys can be any value except `nil` and `NaN`.  Strings compare by
//!   content, everything else by identity/value.  `-0` and `0` are one key.
//! * Integral keys `0..n` that are inserted contiguously live in a dense
//!   `Vec` (the array part); everything else lives in an insertion-ordered
//!   open-addressing hash part (index vector + entry vector, like CPython's
//!   dict), so iteration order is deterministic.

use crate::hash::mix64;
use crate::object::{Gc, Header, Kind, Str};
use crate::value::Value;

#[derive(Debug, PartialEq, Eq)]
pub enum KeyError {
    Nil,
    NaN,
}

#[derive(Clone, Copy)]
struct Entry {
    key: Value,
    val: Value,
}

const EMPTY: u32 = u32::MAX;
/// Hash parts up to this many entries have no index: a linear scan over the
/// (insertion ordered) entries is faster than hashing and saves an allocation.
/// Records / small objects, the most common tables, stay in this mode.
const SMALL: usize = 8;

#[derive(Default)]
struct HashPart {
    entries: Vec<Entry>,
    index: Vec<u32>,
    live: usize,
}

#[inline(always)]
fn hash_key(k: Value) -> u32 {
    if k.is_str() {
        unsafe { k.as_gc_unchecked::<Str>().hash }
    } else {
        mix64(k.bits()) as u32
    }
}

#[inline(always)]
fn keys_equal(a: Value, b: Value) -> bool {
    if a.bits() == b.bits() {
        return true;
    }
    if a.is_str() && b.is_str() {
        let (a, b) = unsafe { (a.as_gc_unchecked::<Str>(), b.as_gc_unchecked::<Str>()) };
        return crate::object::str_eq(a, b);
    }
    false
}

impl HashPart {
    #[inline]
    fn find(&self, key: Value, h: u32) -> Option<usize> {
        if self.index.is_empty() {
            for (i, e) in self.entries.iter().enumerate() {
                if keys_equal(e.key, key) {
                    return Some(i);
                }
            }
            return None;
        }
        let mask = self.index.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let e = unsafe { *self.index.get_unchecked(i) };
            if e == EMPTY {
                return None;
            }
            let en = unsafe { self.entries.get_unchecked(e as usize) };
            if keys_equal(en.key, key) {
                return Some(e as usize);
            }
            i = (i + 1) & mask;
        }
    }

    #[inline]
    fn get(&self, key: Value) -> Option<Value> {
        self.find(key, hash_key(key)).map(|e| self.entries[e].val)
    }

    fn rebuild(&mut self, need: usize) {
        self.entries.retain(|e| e.key != Value::TOMB);
        let cap = (need.max(self.entries.len()) * 2).next_power_of_two().max(8);
        self.index.clear();
        self.index.resize(cap, EMPTY);
        let mask = cap - 1;
        for (n, e) in self.entries.iter().enumerate() {
            let mut i = hash_key(e.key) as usize & mask;
            while self.index[i] != EMPTY {
                i = (i + 1) & mask;
            }
            self.index[i] = n as u32;
        }
    }

    fn insert(&mut self, key: Value, val: Value) {
        let h = hash_key(key);
        if let Some(e) = self.find(key, h) {
            self.entries[e].val = val;
            return;
        }
        if self.index.is_empty() {
            if self.entries.len() < SMALL {
                self.entries.push(Entry { key, val });
                self.live += 1;
                return;
            }
            // full: drop tombstones first, maybe that makes room
            self.entries.retain(|e| e.key != Value::TOMB);
            if self.entries.len() < SMALL {
                self.entries.push(Entry { key, val });
                self.live += 1;
                return;
            }
            self.rebuild(self.live + 1);
        } else if (self.entries.len() + 1) * 4 > self.index.len() * 3 {
            self.rebuild(self.live + 1);
        }
        let mask = self.index.len() - 1;
        let mut i = h as usize & mask;
        while self.index[i] != EMPTY {
            i = (i + 1) & mask;
        }
        self.index[i] = self.entries.len() as u32;
        self.entries.push(Entry { key, val });
        self.live += 1;
    }

    fn remove(&mut self, key: Value) -> Option<Value> {
        let e = self.find(key, hash_key(key))?;
        let old = self.entries[e].val;
        self.entries[e] = Entry { key: Value::TOMB, val: Value::NIL };
        self.live -= 1;
        if self.live == 0 {
            self.entries.clear();
            self.index.iter_mut().for_each(|x| *x = EMPTY);
        }
        Some(old)
    }
}

#[repr(C)]
pub struct Table {
    pub hdr: Header,
    pub arr: Vec<Value>,
    map: HashPart,
}

#[inline(always)]
fn arr_index(key: Value) -> Option<usize> {
    if key.is_num() {
        let n = key.as_num();
        let i = n as usize;
        if i as f64 == n {
            return Some(i);
        }
    }
    None
}

#[inline(always)]
fn norm_key(key: Value) -> Result<Value, KeyError> {
    if key.is_num() {
        let n = key.as_num();
        if n.is_nan() {
            return Err(KeyError::NaN);
        }
        if n == 0.0 {
            return Ok(Value::num(0.0));
        }
    } else if key.is_nil() {
        return Err(KeyError::Nil);
    }
    Ok(key)
}

impl Table {
    pub fn new(arr_cap: usize) -> Table {
        Table { hdr: Header::new(Kind::Table), arr: Vec::with_capacity(arr_cap), map: HashPart::default() }
    }

    pub fn mem_size(&self) -> usize {
        std::mem::size_of::<Table>() + self.arr.capacity() * 8 + self.map.entries.capacity() * 16 + self.map.index.capacity() * 4
    }

    /// Number of entries (array part + hash part).
    #[inline]
    pub fn len(&self) -> usize {
        self.arr.len() + self.map.live
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lookup; absent keys (and invalid keys such as nil/NaN) yield `nil`.
    #[inline]
    pub fn get(&self, key: Value) -> Value {
        if let Some(i) = arr_index(key) {
            if let Some(&v) = self.arr.get(i) {
                return v;
            }
        }
        if self.map.live == 0 {
            return Value::NIL;
        }
        let key = if key.is_num() && key.as_num() == 0.0 { Value::num(0.0) } else { key };
        self.map.get(key).unwrap_or(Value::NIL)
    }

    #[inline]
    pub fn get_str(&self, key: Gc<Str>) -> Value {
        if self.map.live == 0 {
            return Value::NIL;
        }
        self.map.find(Value::from(key), key.hash).map_or(Value::NIL, |e| self.map.entries[e].val)
    }

    pub fn contains(&self, key: Value) -> bool {
        !self.get(key).is_nil()
    }

    /// Assigning `nil` removes the key.
    pub fn set(&mut self, key: Value, val: Value) -> Result<(), KeyError> {
        let key = norm_key(key)?;
        if let Some(i) = arr_index(key) {
            let n = self.arr.len();
            if i < n {
                if val.is_nil() {
                    self.remove_arr(i);
                } else {
                    self.arr[i] = val;
                }
                return Ok(());
            }
            if i == n {
                if !val.is_nil() {
                    self.arr.push(val);
                    self.migrate();
                }
                return Ok(());
            }
        }
        if val.is_nil() {
            self.map.remove(key);
        } else {
            self.map.insert(key, val);
        }
        Ok(())
    }

    /// Append at the end of the array part.
    pub fn push(&mut self, v: Value) {
        if v.is_nil() {
            return;
        }
        self.arr.push(v);
        self.migrate();
    }

    pub fn pop(&mut self) -> Value {
        self.arr.pop().unwrap_or(Value::NIL)
    }

    pub fn remove(&mut self, key: Value) -> Value {
        let old = self.get(key);
        if !old.is_nil() {
            let _ = self.set(key, Value::NIL);
        }
        old
    }

    pub fn clear(&mut self) {
        self.arr.clear();
        self.map = HashPart::default();
    }

    /// Pull keys `arr.len(), arr.len()+1, …` out of the hash part.
    fn migrate(&mut self) {
        while self.map.live > 0 {
            let k = Value::num(self.arr.len() as f64);
            match self.map.remove(k) {
                Some(v) => self.arr.push(v),
                None => break,
            }
        }
    }

    /// Deleting from the middle of the array part moves the tail into the hash.
    fn remove_arr(&mut self, i: usize) {
        if i + 1 == self.arr.len() {
            self.arr.pop();
            return;
        }
        let tail: Vec<Value> = self.arr.drain(i + 1..).collect();
        self.arr.pop();
        for (j, v) in tail.into_iter().enumerate() {
            self.map.insert(Value::num((i + 1 + j) as f64), v);
        }
    }

    /// Cursor based iteration: `pos` starts at 0; returns the next position.
    pub fn next(&self, pos: usize) -> Option<(usize, Value, Value)> {
        let alen = self.arr.len();
        if pos < alen {
            return Some((pos + 1, Value::num(pos as f64), self.arr[pos]));
        }
        let mut e = pos - alen;
        while e < self.map.entries.len() {
            let en = self.map.entries[e];
            e += 1;
            if en.key != Value::TOMB {
                return Some((alen + e, en.key, en.val));
            }
        }
        None
    }

    /// Visit every key and value (used by the GC).
    pub fn for_each(&self, mut f: impl FnMut(Value)) {
        for &v in &self.arr {
            f(v);
        }
        for e in &self.map.entries {
            if e.key != Value::TOMB {
                f(e.key);
                f(e.val);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(n: f64) -> Value {
        Value::num(n)
    }

    #[test]
    fn array_part() {
        let mut t = Table::new(0);
        for i in 0..10 {
            t.set(num(i as f64), num(i as f64 * 2.0)).unwrap();
        }
        assert_eq!(t.arr.len(), 10);
        assert_eq!(t.get(num(3.0)), num(6.0));
        assert_eq!(t.len(), 10);
        // out-of-order insert goes to the hash, then migrates
        t.set(num(11.0), num(1.0)).unwrap();
        assert_eq!(t.arr.len(), 10);
        t.set(num(10.0), num(5.0)).unwrap();
        assert_eq!(t.arr.len(), 12);
        assert_eq!(t.get(num(11.0)), num(1.0));
    }

    #[test]
    fn hash_part_and_delete() {
        let mut t = Table::new(0);
        for i in 0..100 {
            t.set(num(1000.0 + i as f64), num(i as f64)).unwrap();
        }
        assert_eq!(t.len(), 100);
        for i in 0..100 {
            assert_eq!(t.get(num(1000.0 + i as f64)), num(i as f64));
        }
        for i in (0..100).step_by(2) {
            t.set(num(1000.0 + i as f64), Value::NIL).unwrap();
        }
        assert_eq!(t.len(), 50);
        assert_eq!(t.get(num(1000.0)), Value::NIL);
        assert_eq!(t.get(num(1001.0)), num(1.0));
        // iteration skips tombstones
        let mut n = 0;
        let mut pos = 0;
        while let Some((p, _, _)) = t.next(pos) {
            pos = p;
            n += 1;
        }
        assert_eq!(n, 50);
    }

    #[test]
    fn small_mode_transitions() {
        let mut t = Table::new(0);
        // stay in the index-less mode, delete, reinsert, then cross the threshold
        for i in 0..6 {
            t.set(num(10.0 + i as f64), num(i as f64)).unwrap();
        }
        assert!(t.map.index.is_empty());
        t.set(num(12.0), Value::NIL).unwrap();
        t.set(num(99.0), num(99.0)).unwrap();
        assert_eq!(t.len(), 6);
        for i in 0..20 {
            t.set(num(200.0 + i as f64), num(i as f64)).unwrap();
        }
        assert!(!t.map.index.is_empty());
        assert_eq!(t.len(), 26);
        assert_eq!(t.get(num(12.0)), Value::NIL);
        assert_eq!(t.get(num(99.0)), num(99.0));
        assert_eq!(t.get(num(210.0)), num(10.0));
        // repeatedly add and remove one key while small
        let mut u = Table::new(0);
        for _ in 0..100 {
            u.set(num(5.0), num(1.0)).unwrap();
            u.set(num(5.0), Value::NIL).unwrap();
        }
        assert_eq!(u.len(), 0);
        u.set(num(7.0), num(2.0)).unwrap();
        assert_eq!(u.get(num(7.0)), num(2.0));
    }

    /// Random operation sequences checked against a trivially correct model.
    #[test]
    fn matches_a_reference_model() {
        use crate::heap::Heap;
        use std::collections::HashMap;

        #[derive(Hash, PartialEq, Eq, Clone, Debug)]
        enum MKey {
            Num(i64), // keys are small integers or halves, scaled by 2
            Str(String),
        }

        let mut heap = Heap::new();
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for round in 0..300 {
            let mut t = Table::new(0);
            let mut model: HashMap<MKey, f64> = HashMap::new();
            let domain = 3 + (next() % 60) as i64;
            for step in 0..400 {
                let r = next();
                let (key, mkey) = match r % 3 {
                    0 => {
                        let n = (next() as i64).rem_euclid(domain);
                        (Value::num(n as f64), MKey::Num(n * 2))
                    }
                    1 => {
                        let n = (next() as i64).rem_euclid(domain);
                        (Value::num(n as f64 + 0.5), MKey::Num(n * 2 + 1))
                    }
                    _ => {
                        let n = (next() as i64).rem_euclid(domain);
                        let text = format!("key{}", n);
                        // alternate between interned and fresh (non-interned) copies
                        let sv = if next() % 2 == 0 { heap.intern(&text) } else { heap.new_str(text.clone()) };
                        (Value::from(sv), MKey::Str(text))
                    }
                };
                if next() % 3 == 0 {
                    t.set(key, Value::NIL).unwrap();
                    model.remove(&mkey);
                } else {
                    let v = (step * 7 + round) as f64;
                    t.set(key, Value::num(v)).unwrap();
                    model.insert(mkey.clone(), v);
                }
                // point lookups agree
                assert_eq!(t.len(), model.len(), "len mismatch at round {} step {}", round, step);
                if step % 16 == 0 {
                    // iteration yields every live entry exactly once
                    let mut seen = 0;
                    let mut pos = 0;
                    while let Some((np, k, v)) = t.next(pos) {
                        pos = np;
                        seen += 1;
                        let mk = if k.is_num() {
                            let n = k.as_num();
                            MKey::Num((n * 2.0).round() as i64)
                        } else {
                            MKey::Str(k.as_str().unwrap().as_str().to_string())
                        };
                        assert_eq!(model.get(&mk).copied(), Some(v.as_num()), "iterated entry {:?} wrong", mk);
                    }
                    assert_eq!(seen, model.len());
                }
            }
            for (mk, want) in &model {
                let key = match mk {
                    MKey::Num(n) => Value::num(*n as f64 / 2.0),
                    MKey::Str(s) => Value::from(heap.new_str(s.clone())),
                };
                assert_eq!(t.get(key), Value::num(*want), "get {:?}", mk);
            }
        }
    }

    #[test]
    fn invalid_keys() {
        let mut t = Table::new(0);
        assert_eq!(t.set(Value::NIL, num(1.0)), Err(KeyError::Nil));
        assert_eq!(t.set(num(f64::NAN), num(1.0)), Err(KeyError::NaN));
        t.set(num(-0.0), num(7.0)).unwrap();
        assert_eq!(t.get(num(0.0)), num(7.0));
        assert_eq!(t.get(num(-0.0)), num(7.0));
    }

    #[test]
    fn middle_removal() {
        let mut t = Table::new(0);
        for i in 0..5 {
            t.push(num(i as f64));
        }
        t.set(num(2.0), Value::NIL).unwrap();
        assert_eq!(t.arr.len(), 2);
        assert_eq!(t.get(num(3.0)), num(3.0));
        assert_eq!(t.get(num(4.0)), num(4.0));
        assert_eq!(t.len(), 4);
    }
}
