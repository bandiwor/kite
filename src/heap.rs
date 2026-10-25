//! The garbage-collected heap: allocation, string interning, and the
//! mark & sweep phases.  *Which* objects are roots is decided by the VM
//! (`Vm::collect_garbage`); this module only knows how to trace and free.

use crate::hash::FxHashMap;
use crate::object::*;
use crate::table::Table;
use crate::value::Value;

pub struct Heap {
    head: *mut Header,
    /// Estimated live bytes (exact right after a collection).
    pub bytes: usize,
    pub next_gc: usize,
    gray: Vec<*mut Header>,
    strings: FxHashMap<&'static str, Gc<Str>>,
    /// Collect at every safe point (debugging aid for GC bugs).
    pub stress: bool,
    pub collections: usize,
}

const MIN_GC_THRESHOLD: usize = 1 << 20;

impl Default for Heap {
    fn default() -> Self {
        Heap::new()
    }
}

impl Heap {
    pub fn new() -> Heap {
        Heap {
            head: std::ptr::null_mut(),
            bytes: 0,
            next_gc: MIN_GC_THRESHOLD,
            gray: Vec::new(),
            strings: FxHashMap::default(),
            stress: false,
            collections: 0,
        }
    }

    #[inline(always)]
    pub fn should_collect(&self) -> bool {
        self.bytes > self.next_gc || self.stress
    }

    // ---- allocation -----------------------------------------------------------

    pub fn alloc<T: GcObj>(&mut self, obj: T) -> Gc<T> {
        self.bytes += obj.heap_size();
        let p = Box::into_raw(Box::new(obj));
        unsafe {
            let h = p as *mut Header;
            (*h).next = self.head;
            self.head = h;
            Gc::from_raw(p)
        }
    }

    /// A fresh (non-interned) string.
    pub fn new_str(&mut self, s: impl Into<Box<str>>) -> Gc<Str> {
        self.alloc(Str::new(s.into(), false))
    }

    /// The unique string with this content.  Used for constants, field names
    /// and identifiers, so that most string comparisons are pointer compares.
    pub fn intern(&mut self, s: &str) -> Gc<Str> {
        if let Some(&g) = self.strings.get(s) {
            return g;
        }
        let g = self.alloc(Str::new(s.into(), true));
        // The key borrows the string's own heap buffer, which lives (at a
        // stable address) exactly as long as the object; the entry is removed
        // in `sweep` before the object is freed.
        let key: &'static str = unsafe { &*(g.as_str() as *const str) };
        self.strings.insert(key, g);
        g
    }

    pub fn concat(&mut self, a: &str, b: &str) -> Gc<Str> {
        let mut s = String::with_capacity(a.len() + b.len());
        s.push_str(a);
        s.push_str(b);
        self.new_str(s)
    }

    pub fn new_table(&mut self, cap: usize) -> Gc<Table> {
        self.alloc(Table::new(cap))
    }

    pub fn new_array(&mut self, items: Vec<Value>) -> Gc<Array> {
        self.alloc(Array::new(items))
    }

    // ---- marking ---------------------------------------------------------------

    #[inline]
    pub fn mark_value(&mut self, v: Value) {
        if v.is_obj() {
            // SAFETY: an object value always points at a live heap object
            unsafe { self.mark_obj(v.as_header()) };
        }
    }

    /// # Safety
    /// `p` must point to a live object of this heap.
    #[inline]
    pub unsafe fn mark_obj(&mut self, p: *mut Header) {
        if !(*p).marked {
            (*p).marked = true;
            match (*p).kind {
                // leaves: nothing to trace
                Kind::Str | Kind::Native => {}
                _ => self.gray.push(p),
            }
        }
    }

    pub fn mark_gc<T>(&mut self, g: Gc<T>) {
        // SAFETY: a `Gc` handle points at a live object
        unsafe { self.mark_obj(g.as_ptr() as *mut Header) };
    }

    /// Drain the gray stack, marking everything reachable.
    pub fn trace(&mut self) {
        while let Some(p) = self.gray.pop() {
            unsafe { self.blacken(p) };
        }
    }

    unsafe fn blacken(&mut self, p: *mut Header) {
        match (*p).kind {
            Kind::Str | Kind::Native => {}
            Kind::Table => {
                let t = &*(p as *mut Table);
                // collect first: `for_each` borrows the table, `mark_value` borrows self
                let mut tmp: Vec<Value> = Vec::new();
                t.for_each(|v| {
                    if v.is_obj() {
                        tmp.push(v)
                    }
                });
                for v in tmp {
                    self.mark_value(v);
                }
            }
            Kind::Array => {
                let a = &*(p as *mut Array);
                for i in 0..a.items.len() {
                    let v = a.items[i];
                    self.mark_value(v);
                }
            }
            Kind::Proto => {
                let pr = &*(p as *mut Proto);
                for &c in &pr.consts {
                    self.mark_value(c);
                }
                for &p in &pr.protos {
                    self.mark_gc(p);
                }
            }
            Kind::Closure => {
                let c = &*(p as *mut Closure);
                self.mark_gc(c.proto);
                for &u in c.upvals.iter() {
                    self.mark_gc(u);
                }
            }
            Kind::Upvalue => {
                let u = &*(p as *mut Upvalue);
                self.mark_value(*u.location);
            }
            Kind::Variant => {
                let v = &*(p as *mut VariantCtor);
                self.mark_gc(v.ty);
            }
            Kind::StructType => {
                let s = &*(p as *mut StructType);
                self.mark_gc(s.name);
                for v in &s.variants {
                    self.mark_gc(v.name);
                    for &f in &v.fields {
                        self.mark_gc(f);
                    }
                    self.mark_value(v.value);
                }
                for &f in &s.fields {
                    self.mark_gc(f);
                }
                for &(n, v) in &s.methods {
                    self.mark_gc(n);
                    self.mark_value(v);
                }
                for &h in &s.hooks {
                    self.mark_value(h);
                }
            }
            Kind::Instance => {
                let i = &*(p as *mut Instance);
                self.mark_gc(i.ty);
                for &v in i.fields.iter() {
                    self.mark_value(v);
                }
            }
        }
    }

    // ---- sweeping --------------------------------------------------------------

    /// All objects, newest first.
    pub fn objects(&self) -> ObjIter {
        ObjIter { cur: self.head }
    }

    /// Free every unmarked object and clear the mark bits.  Returns the
    /// number of live bytes.
    pub fn sweep(&mut self) -> usize {
        let mut live = 0usize;
        unsafe {
            let mut prev: *mut *mut Header = &mut self.head;
            while !(*prev).is_null() {
                let obj = *prev;
                if (*obj).marked {
                    (*obj).marked = false;
                    live += obj_size(obj);
                    prev = &mut (*obj).next;
                } else {
                    *prev = (*obj).next;
                    if (*obj).kind == Kind::Str {
                        let s = &*(obj as *mut Str);
                        if s.interned {
                            self.strings.remove(s.as_str());
                        }
                    }
                    free_obj(obj);
                }
            }
        }
        self.bytes = live;
        self.next_gc = (live * 2).max(MIN_GC_THRESHOLD);
        self.collections += 1;
        live
    }

    pub fn object_count(&self) -> usize {
        self.objects().count()
    }
}

pub struct ObjIter {
    cur: *mut Header,
}

impl Iterator for ObjIter {
    type Item = *mut Header;
    fn next(&mut self) -> Option<*mut Header> {
        if self.cur.is_null() {
            return None;
        }
        let p = self.cur;
        self.cur = unsafe { (*p).next };
        Some(p)
    }
}

unsafe fn obj_size(p: *mut Header) -> usize {
    match (*p).kind {
        Kind::Str => (*(p as *mut Str)).heap_size(),
        Kind::Table => (*(p as *mut Table)).heap_size(),
        Kind::Array => (*(p as *mut Array)).heap_size(),
        Kind::Proto => (*(p as *mut Proto)).heap_size(),
        Kind::Closure => (*(p as *mut Closure)).heap_size(),
        Kind::Upvalue => (*(p as *mut Upvalue)).heap_size(),
        Kind::Native => (*(p as *mut Native)).heap_size(),
        Kind::StructType => (*(p as *mut StructType)).heap_size(),
        Kind::Instance => (*(p as *mut Instance)).heap_size(),
        Kind::Variant => (*(p as *mut VariantCtor)).heap_size(),
    }
}

unsafe fn free_obj(p: *mut Header) {
    match (*p).kind {
        Kind::Str => drop(Box::from_raw(p as *mut Str)),
        Kind::Table => drop(Box::from_raw(p as *mut Table)),
        Kind::Array => drop(Box::from_raw(p as *mut Array)),
        Kind::Proto => drop(Box::from_raw(p as *mut Proto)),
        Kind::Closure => drop(Box::from_raw(p as *mut Closure)),
        Kind::Upvalue => drop(Box::from_raw(p as *mut Upvalue)),
        Kind::Native => drop(Box::from_raw(p as *mut Native)),
        Kind::StructType => drop(Box::from_raw(p as *mut StructType)),
        Kind::Instance => drop(Box::from_raw(p as *mut Instance)),
        Kind::Variant => drop(Box::from_raw(p as *mut VariantCtor)),
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        unsafe {
            let mut p = self.head;
            while !p.is_null() {
                let next = (*p).next;
                free_obj(p);
                p = next;
            }
        }
        self.head = std::ptr::null_mut();
    }
}
