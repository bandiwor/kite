//! A small-object allocator for the `kite` binary.
//!
//! Profiles of allocation-heavy programs (many short-lived tables, arrays, instances, strings)
//! spend 30-45% of their time in `malloc`/`free`.  This allocator serves every request of up to
//! `MAX_SMALL` bytes from per-thread size-class free lists (16-byte granularity) carved out of
//! large chunks, so both `alloc` and `dealloc` are a handful of instructions and need no locking.
//! Bigger requests go to the system allocator unchanged.
//!
//! Properties worth knowing:
//! * Small blocks are never returned to the operating system; they are reused for requests of
//!   the same size class.  That suits a VM whose garbage has the same shapes as its live data.
//! * A block freed by a different thread than the one that allocated it simply joins the freeing
//!   thread's list.  This is sound because chunks are never unmapped.
//! * The `system-alloc` Cargo feature switches it off (use it with valgrind or sanitizers, which
//!   cannot see inside the pool).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::ptr::null_mut;

/// Requests up to this size (and alignment up to 16) are pooled.
pub const MAX_SMALL: usize = 512;
const GRAIN_SHIFT: usize = 4;
const CLASSES: usize = MAX_SMALL >> GRAIN_SHIFT;
/// Size of the chunks the pool carves blocks from.
const CHUNK: usize = 256 * 1024;

struct Pool {
    /// Head of the free list of each size class (index = size / 16 - 1).
    free: [*mut u8; CLASSES],
    cur: *mut u8,
    end: *mut u8,
}

thread_local! {
    // `const` initialisation and no destructor: access never allocates or fails.
    static POOL: UnsafeCell<Pool> = const { UnsafeCell::new(Pool { free: [null_mut(); CLASSES], cur: null_mut(), end: null_mut() }) };
}

pub struct PoolAlloc;

#[inline(always)]
fn is_small(size: usize, align: usize) -> bool {
    size <= MAX_SMALL && align <= 16
}

/// Size class (0-based) of a request of `size` bytes, `size >= 1`.
#[inline(always)]
fn class_of(size: usize) -> usize {
    (size - 1) >> GRAIN_SHIFT
}

impl Pool {
    #[inline(always)]
    unsafe fn alloc(&mut self, class: usize) -> *mut u8 {
        let head = *self.free.get_unchecked(class);
        if !head.is_null() {
            *self.free.get_unchecked_mut(class) = *(head as *mut *mut u8);
            return head;
        }
        let bytes = (class + 1) << GRAIN_SHIFT;
        if (self.end as usize) - (self.cur as usize) < bytes {
            return self.refill_and_alloc(bytes);
        }
        let p = self.cur;
        self.cur = p.add(bytes);
        p
    }

    #[cold]
    #[inline(never)]
    unsafe fn refill_and_alloc(&mut self, bytes: usize) -> *mut u8 {
        let chunk = System.alloc(Layout::from_size_align_unchecked(CHUNK, 16));
        if chunk.is_null() {
            return null_mut();
        }
        // the unused tail of the old chunk is abandoned (less than MAX_SMALL bytes)
        self.cur = chunk.add(bytes);
        self.end = chunk.add(CHUNK);
        chunk
    }

    #[inline(always)]
    unsafe fn dealloc(&mut self, p: *mut u8, class: usize) {
        *(p as *mut *mut u8) = *self.free.get_unchecked(class);
        *self.free.get_unchecked_mut(class) = p;
    }
}

unsafe impl GlobalAlloc for PoolAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_small(layout.size(), layout.align()) {
            let class = class_of(layout.size().max(1));
            POOL.with(|p| (*p.get()).alloc(class))
        } else {
            System.alloc(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if is_small(layout.size(), layout.align()) {
            let class = class_of(layout.size().max(1));
            POOL.with(|p| (*p.get()).dealloc(ptr, class))
        } else {
            System.dealloc(ptr, layout)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_small(layout.size(), layout.align()) {
            let p = self.alloc(layout);
            if !p.is_null() {
                std::ptr::write_bytes(p, 0, layout.size());
            }
            p
        } else {
            System.alloc_zeroed(layout)
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (old_small, new_small) = (is_small(layout.size(), layout.align()), is_small(new_size, layout.align()));
        if !old_small && !new_small {
            return System.realloc(ptr, layout, new_size);
        }
        if old_small && new_small && class_of(layout.size().max(1)) == class_of(new_size.max(1)) {
            return ptr;
        }
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new = self.alloc(new_layout);
        if !new.is_null() {
            std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn blocks_are_aligned_distinct_and_reused() {
        let a = PoolAlloc;
        unsafe {
            let mut ptrs = Vec::new();
            for size in [1, 8, 16, 17, 31, 32, 100, 255, 512] {
                for _ in 0..3 {
                    let p = a.alloc(layout(size, 8));
                    assert!(!p.is_null());
                    assert_eq!(p as usize % 16, 0, "pooled blocks are 16-aligned");
                    std::ptr::write_bytes(p, 0xAB, size);
                    ptrs.push((p, size));
                }
            }
            // no overlap: every block still holds its own bytes
            for &(p, size) in &ptrs {
                assert!((0..size).all(|i| *p.add(i) == 0xAB));
            }
            let mut sorted: Vec<usize> = ptrs.iter().map(|&(p, _)| p as usize).collect();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), ptrs.len());
            // a freed block is handed out again for the same class
            let (p, size) = ptrs[0];
            a.dealloc(p, layout(size, 8));
            let q = a.alloc(layout(size, 8));
            assert_eq!(p, q);
            for &(p, size) in &ptrs[1..] {
                a.dealloc(p, layout(size, 8));
            }
            a.dealloc(q, layout(size, 8));
        }
    }

    #[test]
    fn large_and_over_aligned_requests_use_the_system_allocator() {
        let a = PoolAlloc;
        unsafe {
            for l in [layout(MAX_SMALL + 1, 8), layout(4096, 64), layout(64, 64), layout(1 << 20, 16)] {
                let p = a.alloc(l);
                assert!(!p.is_null());
                assert_eq!(p as usize % l.align(), 0);
                std::ptr::write_bytes(p, 1, l.size());
                a.dealloc(p, l);
            }
            let z = a.alloc_zeroed(layout(1 << 20, 16));
            assert!((0..1 << 20).step_by(4099).all(|i| *z.add(i) == 0));
            a.dealloc(z, layout(1 << 20, 16));
            let z = a.alloc_zeroed(layout(100, 8));
            assert!((0..100).all(|i| *z.add(i) == 0));
            a.dealloc(z, layout(100, 8));
        }
    }

    #[test]
    fn realloc_preserves_contents_across_classes() {
        let a = PoolAlloc;
        unsafe {
            let mut size = 8;
            let mut p = a.alloc(layout(size, 8));
            for i in 0..size {
                *p.add(i) = i as u8;
            }
            // grow through small classes into the system allocator, then shrink back
            for new in [12, 16, 40, 400, 512, 513, 5000, 100, 8] {
                let np = a.realloc(p, layout(size, 8), new);
                assert!(!np.is_null());
                for i in 0..size.min(new).min(8) {
                    assert_eq!(*np.add(i), i as u8, "byte {} after growing to {}", i, new);
                }
                p = np;
                size = new;
            }
            a.dealloc(p, layout(size, 8));
            // same class: the pointer does not move
            let p = a.alloc(layout(20, 8));
            assert_eq!(a.realloc(p, layout(20, 8), 30), p);
            a.dealloc(p, layout(30, 8));
        }
    }

    #[test]
    fn many_chunks_and_cross_thread_frees() {
        let a = PoolAlloc;
        unsafe {
            // more than a few chunks worth of 512-byte blocks
            let blocks: Vec<usize> = (0..2000).map(|_| a.alloc(layout(512, 16)) as usize).collect();
            assert!(blocks.iter().all(|&p| p != 0));
            let handle = std::thread::spawn(move || {
                let a = PoolAlloc;
                for p in blocks {
                    a.dealloc(p as *mut u8, layout(512, 16));
                }
                // blocks freed here are reused here
                let p = a.alloc(layout(512, 16));
                assert!(!p.is_null());
                a.dealloc(p, layout(512, 16));
            });
            handle.join().unwrap();
        }
    }
}
