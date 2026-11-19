//! AVX2 + FMA + F16C helpers. Every hot kernel is a `#[target_feature]` function; these inline into it.
#![allow(clippy::missing_safety_doc)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::arch::x86_64::*;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub type F8 = __m256;

/// The kernels need AVX2, FMA and F16C (x86-64-v3, Haswell or later).
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
}

pub fn has_vnni() -> bool {
    std::env::var("HS_ISA").map_or_else(|_| is_x86_feature_detected!("avxvnni"), |v| v == "vnni")
}

/// Zeroed, 64-byte aligned, fixed length.
pub struct Buf<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}
unsafe impl<T: Copy + Send> Send for Buf<T> {}
unsafe impl<T: Copy + Sync> Sync for Buf<T> {}

impl<T: Copy> Buf<T> {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Buf { ptr: NonNull::dangling(), len };
        }
        let layout = Layout::from_size_align(len * size_of::<T>(), 64).unwrap();
        let p = unsafe { alloc_zeroed(layout) } as *mut T;
        Buf { ptr: NonNull::new(p).expect("out of memory"), len }
    }
}

impl<T: Copy> Drop for Buf<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, Layout::from_size_align(self.len * size_of::<T>(), 64).unwrap())
            }
        }
    }
}

impl<T: Copy> Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}
impl<T: Copy> DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Default for Buf<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Grow-only storage reused across calls.
#[derive(Default)]
pub struct Scratch(Buf<f32>);
impl Scratch {
    pub fn get(&mut self, n: usize) -> *mut f32 {
        if n > self.0.len() {
            self.0 = Buf::new(n);
        }
        self.0.as_mut_ptr()
    }
}

pub fn f16(h: u16) -> f32 {
    let (e, m) = ((h >> 10) & 31, (h & 1023) as u32);
    let v = match e {
        0 => m as f32 * f32::from_bits(103 << 23),
        31 => {
            if m != 0 {
                f32::NAN
            } else {
                f32::INFINITY
            }
        }
        _ => f32::from_bits((e as u32 + 112) << 23 | m << 13),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

#[inline(always)]
pub unsafe fn ld(p: *const f32) -> F8 {
    unsafe { _mm256_loadu_ps(p) }
}
#[inline(always)]
pub unsafe fn st(p: *mut f32, v: F8) {
    unsafe { _mm256_storeu_ps(p, v) }
}
/// Eight f16 values widened to f32.
#[inline(always)]
pub unsafe fn ldh(p: *const u16) -> F8 {
    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)) }
}
#[inline(always)]
pub unsafe fn set(x: f32) -> F8 {
    unsafe { _mm256_set1_ps(x) }
}
#[inline(always)]
pub unsafe fn zero() -> F8 {
    unsafe { _mm256_setzero_ps() }
}
#[inline(always)]
pub unsafe fn add(a: F8, b: F8) -> F8 {
    unsafe { _mm256_add_ps(a, b) }
}
#[inline(always)]
pub unsafe fn sub(a: F8, b: F8) -> F8 {
    unsafe { _mm256_sub_ps(a, b) }
}
#[inline(always)]
pub unsafe fn mul(a: F8, b: F8) -> F8 {
    unsafe { _mm256_mul_ps(a, b) }
}
#[inline(always)]
pub unsafe fn div(a: F8, b: F8) -> F8 {
    unsafe { _mm256_div_ps(a, b) }
}
#[inline(always)]
pub unsafe fn max(a: F8, b: F8) -> F8 {
    unsafe { _mm256_max_ps(a, b) }
}
#[inline(always)]
pub unsafe fn min(a: F8, b: F8) -> F8 {
    unsafe { _mm256_min_ps(a, b) }
}
/// a * b + c in one rounding.
#[inline(always)]
pub unsafe fn fma(a: F8, b: F8, c: F8) -> F8 {
    unsafe { _mm256_fmadd_ps(a, b, c) }
}
/// c - a * b in one rounding.
#[inline(always)]
pub unsafe fn fnma(a: F8, b: F8, c: F8) -> F8 {
    unsafe { _mm256_fnmadd_ps(a, b, c) }
}

/// ((t0 + t4) + (t2 + t6)) + ((t1 + t5) + (t3 + t7)).
#[inline(always)]
pub unsafe fn sum8(a: F8) -> f32 {
    unsafe {
        let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let mut t = [0f32; 4];
        _mm_storeu_ps(t.as_mut_ptr(), s);
        (t[0] + t[2]) + (t[1] + t[3])
    }
}

#[inline(always)]
pub unsafe fn peak8(a: F8) -> f32 {
    unsafe {
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), a);
        t.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Denormal results and inputs become zero while this lives.
pub struct FlushDenormals(u32);
impl FlushDenormals {
    #[allow(deprecated)]
    pub fn new() -> Self {
        unsafe {
            let saved = _mm_getcsr();
            _mm_setcsr(saved | 0x8040);
            FlushDenormals(saved)
        }
    }
}
impl Drop for FlushDenormals {
    #[allow(deprecated)]
    fn drop(&mut self) {
        unsafe { _mm_setcsr(self.0) }
    }
}
