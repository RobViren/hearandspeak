//! The int8 linear layers of the encoder, where two thirds of the time goes. Dense f16 weights are quantized to int8
//! per output row and 128-column group; activations per frame and 128-column group, with a +128 offset so one
//! `vpdpbusd` multiplies unsigned codes by signed weights.

use crate::pool::{P, Pool};
use crate::simd::{Buf, f16};
use std::arch::x86_64::*;

pub const GROUP: usize = 128;
const ROWS: usize = 8;
const CHUNK: usize = 48;

/// Rows go in blocks of 8 so one 32-byte vector holds 4 consecutive input columns for each of 8 output rows.
/// w: [G][out/8][32][8][4]. init: [G][out/8][8] = -128 * sum of the group's weights, cancelling the +128 on the
/// activations. scale: [G][out/8][8].
#[derive(Default)]
pub struct Int8W {
    pub out: usize,
    pub inp: usize,
    pub g: usize,
    pub w: Buf<i8>,
    pub init: Buf<i32>,
    pub scale: Buf<f32>,
}

impl Int8W {
    pub fn alloc(out: usize, inp: usize) -> Int8W {
        let g = inp / GROUP;
        Int8W { out, inp, g, w: Buf::new(out * inp), init: Buf::new(g * out), scale: Buf::new(g * out) }
    }

    /// weight [out][in] in f16.
    pub fn quantize(weight: &[u16], out: usize, inp: usize) -> Int8W {
        let mut p = Int8W::alloc(out, inp);
        let rb = out / ROWS;
        let mut w = vec![0f32; inp];
        let mut values = vec![0i8; inp];
        for i in 0..out {
            for (k, v) in w.iter_mut().enumerate() {
                *v = f16(weight[i * inp + k]);
            }
            for g in 0..p.g {
                let grp = &w[g * GROUP..(g + 1) * GROUP];
                let mx = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                for k in 0..GROUP {
                    values[g * GROUP + k] = (grp[k] / scale).round_ties_even() as i8;
                }
                let o = g * rb + i / ROWS;
                let mut sum = 0i32;
                for s in 0..32 {
                    let c = &values[g * GROUP + s * 4..g * GROUP + s * 4 + 4];
                    let at = o * 1024 + s * 32 + (i % ROWS) * 4;
                    for (d, &v) in p.w[at..at + 4].iter_mut().zip(c) {
                        *d = v;
                    }
                    sum += c.iter().map(|&v| v as i32).sum::<i32>();
                }
                p.init[o * 8 + i % ROWS] = -128 * sum;
                p.scale[o * 8 + i % ROWS] = scale;
            }
        }
        p
    }
}

/// Activations quantized per (frame, 128-column group): q = round(x / sx) + 128.
#[derive(Default)]
pub struct QAct {
    pub t: usize,
    pub c: usize,
    pub q: Buf<u8>,
    pub sx: Buf<f32>,
}

impl QAct {
    pub fn reserve(&mut self, t: usize, c: usize) {
        self.t = t;
        self.c = c;
        if t * c > self.q.len() {
            self.q = Buf::new(t * c);
            self.sx = Buf::new(t * c / GROUP);
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn quantize_rows(x: *const f32, c: usize, q: *mut u8, sx: *mut f32, t0: usize, t1: usize) {
    unsafe {
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        let fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let off = _mm256_set1_epi8(0x80u8 as i8);
        let g_n = c / GROUP;
        for t in t0..t1 {
            for g in 0..g_n {
                let p = x.add(t * c + g * GROUP);
                let mut m = _mm256_setzero_ps();
                for i in (0..GROUP).step_by(8) {
                    m = _mm256_max_ps(m, _mm256_and_ps(absmask, _mm256_loadu_ps(p.add(i))));
                }
                let mut m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
                m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
                let mx = _mm_cvtss_f32(_mm_max_ss(m4, _mm_shuffle_ps::<1>(m4, m4)));
                let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                *sx.add(t * g_n + g) = s;
                let inv = _mm256_set1_ps(1.0 / s);
                let cv = |i: usize| _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(i)), inv));
                for i in (0..GROUP).step_by(32) {
                    let b = _mm256_packs_epi16(
                        _mm256_packs_epi32(cv(i), cv(i + 8)),
                        _mm256_packs_epi32(cv(i + 16), cv(i + 24)),
                    );
                    _mm256_storeu_si256(
                        q.add(t * c + g * GROUP + i) as *mut __m256i,
                        _mm256_xor_si256(_mm256_permutevar8x32_epi32(b, fix), off),
                    );
                }
            }
        }
    }
}

pub fn quantize(pool: &Pool, x: *const f32, a: &mut QAct) {
    let (q, sx, c, t) = (P(a.q.as_mut_ptr()), P(a.sx.as_mut_ptr()), a.c, a.t);
    let x = P(x as *mut f32);
    pool.for_each(t.div_ceil(16), 1, |i| unsafe {
        quantize_rows(x.ptr(), c, q.ptr(), sx.ptr(), i * 16, t.min(i * 16 + 16))
    });
}

struct TileArgs {
    w: *const i8,
    init: *const i32,
    scale: *const f32,
    x: *const u8,
    xstride: usize,
    sx: *const f32,
    sxstride: usize,
    y: *mut f32,
    ystride: usize,
    t: usize,
    first: bool,
}

#[inline(always)]
unsafe fn out_rows<const R: usize>(a: &[__m256i; R], scale: *const f32, sx: *const f32, y: *mut f32, first: bool) {
    unsafe {
        let sxb = _mm256_broadcast_ss(&*sx);
        for r in 0..R {
            let v = _mm256_mul_ps(_mm256_cvtepi32_ps(a[r]), _mm256_loadu_ps(scale.add(r * 8)));
            let yr = y.add(r * 8);
            _mm256_storeu_ps(
                yr,
                if first { _mm256_mul_ps(v, sxb) } else { _mm256_fmadd_ps(v, sxb, _mm256_loadu_ps(yr)) },
            );
        }
    }
}

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn tile_vnni<const R: usize>(a: &TileArgs) {
    unsafe {
        let (mut x, mut sx, mut y) = (a.x, a.sx, a.y);
        for _ in 0..a.t {
            let mut acc = [_mm256_setzero_si256(); R];
            for (r, v) in acc.iter_mut().enumerate() {
                *v = _mm256_loadu_si256(a.init.add(r * 8) as *const __m256i);
            }
            for s in 0..32 {
                let xb = _mm256_set1_epi32((x.add(4 * s) as *const i32).read_unaligned());
                for (r, v) in acc.iter_mut().enumerate() {
                    *v = _mm256_dpbusd_avx_epi32(
                        *v,
                        xb,
                        _mm256_load_si256(a.w.add(r * 1024 + s * 32) as *const __m256i),
                    );
                }
            }
            out_rows(&acc, a.scale, sx, y, a.first);
            x = x.add(a.xstride);
            sx = sx.add(a.sxstride);
            y = y.add(a.ystride);
        }
    }
}

// Dense int8 weights overflow the 16-bit pair sums of maddubs against unsigned codes. Signed codes (q - 128,
// within +-127) times |w| stay under 32767, so each step widens to int32 and the +128 offset needs no init term.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2<const R: usize>(a: &TileArgs) {
    unsafe {
        let (mut x, mut sx, mut y) = (a.x, a.sx, a.y);
        let ones = _mm256_set1_epi16(1);
        let off = _mm256_set1_epi8(0x80u8 as i8);
        for _ in 0..a.t {
            let mut acc = [_mm256_setzero_si256(); R];
            for s in 0..32 {
                let xb = _mm256_set1_epi32((x.add(4 * s) as *const i32).read_unaligned());
                let xs = _mm256_xor_si256(xb, off);
                for (r, v) in acc.iter_mut().enumerate() {
                    let wv = _mm256_load_si256(a.w.add(r * 1024 + s * 32) as *const __m256i);
                    let pr = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xs, wv));
                    *v = _mm256_add_epi32(*v, _mm256_madd_epi16(pr, ones));
                }
            }
            out_rows(&acc, a.scale, sx, y, a.first);
            x = x.add(a.xstride);
            sx = sx.add(a.sxstride);
            y = y.add(a.ystride);
        }
    }
}

/// Rows [rb, rb + R) of 8-row blocks, frames [t0, t1). R is 12, 4 or 1.
unsafe fn gemm_tile(w: &Int8W, a: &QAct, y: *mut f32, rb: usize, r: usize, t0: usize, t1: usize, vnni: bool) {
    unsafe {
        let rbn = w.out / ROWS;
        for g in 0..w.g {
            let o = g * rbn + rb;
            if g + 1 < w.g {
                let next = w.w.as_ptr().add((o + rbn) * 1024);
                for i in (0..r * 1024).step_by(64) {
                    _mm_prefetch::<_MM_HINT_T0>(next.add(i));
                }
            }
            let args = TileArgs {
                w: w.w.as_ptr().add(o * 1024),
                init: w.init.as_ptr().add(o * 8),
                scale: w.scale.as_ptr().add(o * 8),
                x: a.q.as_ptr().add(t0 * a.c + g * GROUP),
                xstride: a.c,
                sx: a.sx.as_ptr().add(t0 * w.g + g),
                sxstride: w.g,
                y: y.add(t0 * w.out + rb * ROWS),
                ystride: w.out,
                t: t1 - t0,
                first: g == 0,
            };
            match (vnni, r) {
                (true, 12) => tile_vnni::<12>(&args),
                (true, 4) => tile_vnni::<4>(&args),
                (true, _) => tile_vnni::<1>(&args),
                (false, 12) => tile_avx2::<12>(&args),
                (false, 4) => tile_avx2::<4>(&args),
                (false, _) => tile_avx2::<1>(&args),
            }
        }
    }
}

/// y [T][out] = dequantized a * w. Work items are (row tile, frame chunk), tile-major, so the cores sharing a tile
/// pull its weights from L3 once.
pub fn gemm(pool: &Pool, w: &Int8W, a: &QAct, y: *mut f32, vnni: bool) {
    let rbn = w.out / ROWS;
    let (n12, n4, n1) = (rbn / 12, rbn % 12 / 4, rbn % 4);
    let chunks = a.t.div_ceil(CHUNK);
    let y = P(y);
    pool.for_each((n12 + n4 + n1) * chunks, 1, |j| {
        let tile = j / chunks;
        let t0 = j % chunks * CHUNK;
        let t1 = a.t.min(t0 + CHUNK);
        let (r, rb) = if tile < n12 {
            (12, tile * 12)
        } else if tile < n12 + n4 {
            (4, n12 * 12 + (tile - n12) * 4)
        } else {
            (1, n12 * 12 + n4 * 4 + tile - n12 - n4)
        };
        unsafe { gemm_tile(w, a, y.ptr(), rb, r, t0, t1, vnni) };
    });
}
