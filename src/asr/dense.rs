//! Float GEMM on packed panels, f16 GEMV for the decoder, and the row ops of the encoder.

use crate::pool::{P, Pool};
use crate::simd::*;
use std::arch::x86_64::*;

/// Right operand of c = a * B, B [K][N], stored as ceil(N / 16) panels of [K][16] (zero padded) so the kernel
/// streams one panel while 6 rows x 16 columns of c sit in registers. A linear layer is B = weight^T plus bias.
#[derive(Default)]
pub struct Packed {
    pub k: usize,
    pub n: usize,
    pub p: Buf<f32>,
    pub bias: Option<Buf<f32>>,
}

impl Packed {
    pub fn panels(&self) -> usize {
        self.n.div_ceil(16)
    }
    fn resize(&mut self, k: usize, n: usize) {
        self.k = k;
        self.n = n;
        let need = n.div_ceil(16) * k * 16;
        if need > self.p.len() {
            self.p = Buf::new(need);
        }
    }
    /// B[k][n] = src[k * sk + n * sn]. A linear weight [N][K] is (sk 1, sn K); a matrix [K][N] is (sk N, sn 1).
    pub unsafe fn pack(
        &mut self,
        k: usize,
        n: usize,
        src: *const f32,
        sk: usize,
        sn: usize,
        bias: Option<&[f32]>,
    ) {
        self.resize(k, n);
        let mut o = 0;
        for n0 in (0..n).step_by(16) {
            for kk in 0..k {
                for j in 0..16 {
                    self.p[o + j] = if n0 + j < n {
                        unsafe { *src.add(kk * sk + (n0 + j) * sn) }
                    } else {
                        0.0
                    };
                }
                o += 16;
            }
        }
        if let Some(b) = bias {
            let mut buf = Buf::new(self.panels() * 16);
            buf[..n].copy_from_slice(&b[..n]);
            self.bias = Some(buf);
        }
    }
    pub fn linear(w: &[f32], bias: &[f32]) -> Packed {
        let mut p = Packed::default();
        let k = w.len() / bias.len();
        unsafe { p.pack(k, bias.len(), w.as_ptr(), 1, k, Some(bias)) };
        p
    }
}

/// R rows of a against one panel. U independent accumulator sets over k keep a single row from being bound by the
/// FMA latency chain.
#[inline(always)]
unsafe fn kernel<const R: usize, const U: usize>(
    p: *const f32,
    k: usize,
    a: *const f32,
    astride: usize,
    bias: *const f32,
    out: *mut f32,
) {
    unsafe {
        let mut acc = [[[zero(); 2]; R]; U];
        if !bias.is_null() {
            for r in 0..R {
                acc[0][r] = [ld(bias), ld(bias.add(8))];
            }
        }
        let mut kk = 0;
        while kk + U <= k {
            for u in 0..U {
                let w0 = ld(p.add((kk + u) * 16));
                let w1 = ld(p.add((kk + u) * 16 + 8));
                for r in 0..R {
                    let x = set(*a.add(r * astride + kk + u));
                    acc[u][r][0] = fma(x, w0, acc[u][r][0]);
                    acc[u][r][1] = fma(x, w1, acc[u][r][1]);
                }
            }
            kk += U;
        }
        while kk < k {
            let w0 = ld(p.add(kk * 16));
            let w1 = ld(p.add(kk * 16 + 8));
            for r in 0..R {
                let x = set(*a.add(r * astride + kk));
                acc[0][r][0] = fma(x, w0, acc[0][r][0]);
                acc[0][r][1] = fma(x, w1, acc[0][r][1]);
            }
            kk += 1;
        }
        for r in 0..R {
            for h in 0..2 {
                let mut s = acc[0][r][h];
                for u in acc.iter().skip(1) {
                    s = add(s, u[r][h]);
                }
                st(out.add(r * 16 + h * 8), s);
            }
        }
    }
}

/// Single thread: c[m][16 * p0 .. 16 * p1) for m < rows, clipped to N columns.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemm(
    b: &Packed,
    a: *const f32,
    astride: usize,
    rows: usize,
    c: *mut f32,
    cstride: usize,
    p0: usize,
    p1: usize,
) {
    unsafe {
        let mut out = [0f32; 96];
        let o = out.as_mut_ptr();
        const BLOCK: usize = 48;
        let bias_base = b.bias.as_ref().map_or(std::ptr::null(), |x| x.as_ptr());
        for mb in (0..rows).step_by(BLOCK) {
            for p in p0..p1 {
                let w = b.p.as_ptr().add(p * b.k * 16);
                let bias = if bias_base.is_null() {
                    bias_base
                } else {
                    bias_base.add(p * 16)
                };
                let n = 16.min(b.n - p * 16);
                let me = rows.min(mb + BLOCK);
                let mut m = mb;
                while m + 6 <= me {
                    kernel::<6, 1>(w, b.k, a.add(m * astride), astride, bias, o);
                    for r in 0..6 {
                        std::ptr::copy_nonoverlapping(
                            o.add(r * 16),
                            c.add((m + r) * cstride + p * 16),
                            n,
                        );
                    }
                    m += 6;
                }
                while m < me {
                    kernel::<1, 4>(w, b.k, a.add(m * astride), astride, bias, o);
                    std::ptr::copy_nonoverlapping(o, c.add(m * cstride + p * 16), n);
                    m += 1;
                }
            }
        }
    }
}

/// c [M][N] = a [M][K] * B + bias, threaded over rows, or over panels when M is small.
pub fn linear(pool: &Pool, b: &Packed, a: *const f32, rows: usize, c: *mut f32) {
    let (np, n) = (b.panels(), pool.n);
    let (a, c) = (P(a as *mut f32), P(c));
    if rows >= 6 * n {
        let cut = |i: usize| if i == n { rows } else { rows * i / n / 6 * 6 };
        pool.run(|i| unsafe {
            let m0 = cut(i);
            gemm(
                b,
                a.at(m0 * b.k),
                b.k,
                cut(i + 1) - m0,
                c.at(m0 * b.n),
                b.n,
                0,
                np,
            );
        });
    } else {
        pool.run(|i| unsafe {
            gemm(
                b,
                a.ptr(),
                b.k,
                rows,
                c.ptr(),
                b.n,
                np * i / n,
                np * (i + 1) / n,
            )
        });
    }
}

/// A linear layer applied to one row at a time, kept as the f16 the weights ship in. A single row product is bound
/// by the bytes it streams, not by its multiplies.
#[derive(Default)]
pub struct Half {
    pub k: usize,
    pub n: usize,
    pub p: Buf<u16>,
    pub bias: Buf<f32>,
}

impl Half {
    pub fn panels(&self) -> usize {
        self.n.div_ceil(16)
    }
    /// weight [N][K]
    pub fn pack(k: usize, n: usize, weight: &[u16], bias: &[f32]) -> Half {
        let panels = n.div_ceil(16);
        let mut p = Buf::new(panels * k * 16);
        let mut b = Buf::new(panels * 16);
        let mut o = 0;
        for n0 in (0..n).step_by(16) {
            for kk in 0..k {
                for j in 0..16 {
                    p[o + j] = if n0 + j < n {
                        weight[(n0 + j) * k + kk]
                    } else {
                        0
                    };
                }
                o += 16;
            }
        }
        b[..n].copy_from_slice(&bias[..n]);
        Half { k, n, p, bias: b }
    }
}

/// The input columns a row product visits, split by k % 4: one accumulator set each. Zero inputs can be left out
/// without changing any sum.
pub struct Picks {
    n: [usize; 4],
    k: [[u16; 320]; 4],
}

impl Picks {
    pub fn new(x: &[f32], skip_zeros: bool) -> Picks {
        let mut p = Picks {
            n: [0; 4],
            k: [[0; 320]; 4],
        };
        for (i, &v) in x.iter().enumerate() {
            if !skip_zeros || v != 0.0 {
                p.k[i % 4][p.n[i % 4]] = i as u16;
                p.n[i % 4] += 1;
            }
        }
        p
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv(b: &Half, x: *const f32, s: &Picks, c: *mut f32, p0: usize, p1: usize) {
    unsafe {
        let common = *s.n.iter().min().unwrap();
        let most = *s.n.iter().max().unwrap();
        for p in p0..p1 {
            let w = b.p.as_ptr().add(p * b.k * 16);
            let mut acc = [[zero(); 2]; 4];
            acc[0] = [
                ld(b.bias.as_ptr().add(p * 16)),
                ld(b.bias.as_ptr().add(p * 16 + 8)),
            ];
            macro_rules! step {
                ($u:expr, $i:expr) => {{
                    let k = s.k[$u][$i] as usize;
                    let v = set(*x.add(k));
                    acc[$u][0] = fma(v, ldh(w.add(k * 16)), acc[$u][0]);
                    acc[$u][1] = fma(v, ldh(w.add(k * 16 + 8)), acc[$u][1]);
                }};
            }
            for i in 0..common {
                step!(0, i);
                step!(1, i);
                step!(2, i);
                step!(3, i);
            }
            for i in common..most {
                for u in 0..4 {
                    if i < s.n[u] {
                        step!(u, i);
                    }
                }
            }
            for h in 0..2 {
                st(
                    c.add(p * 16 + h * 8),
                    add(add(add(acc[0][h], acc[1][h]), acc[2][h]), acc[3][h]),
                );
            }
        }
    }
}

/// c [16 * panels] = x [K] * B + bias, threaded over panels with a fixed split so each core keeps its share of the
/// weights in its own L2 from one step to the next.
pub fn linear_half(pool: &Pool, b: &Half, x: *const f32, picks: &Picks, c: *mut f32) {
    let (np, n) = (b.panels(), pool.n);
    let (x, c) = (P(x as *mut f32), P(c));
    pool.run(|i| unsafe { gemv(b, x.ptr(), picks, c.ptr(), np * i / n, np * (i + 1) / n) });
}

#[inline(always)]
pub unsafe fn exp8(x: F8) -> F8 {
    unsafe {
        let x = min(max(x, set(-87.33654)), set(88.0));
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(mul(
            x,
            set(1.442_695_040_888_963_4),
        ));
        let mut r = fnma(n, set(0.693359375), x);
        r = fnma(n, set(-2.12194440e-4), r);
        let mut p = set(1.9875691500e-4);
        p = fma(p, r, set(1.3981999507e-3));
        p = fma(p, r, set(8.3334519073e-3));
        p = fma(p, r, set(4.1665795894e-2));
        p = fma(p, r, set(1.6666665459e-1));
        p = fma(p, r, set(5.0000001201e-1));
        p = add(fma(p, mul(r, r), r), set(1.0));
        let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvttps_epi32(n),
            _mm256_set1_epi32(127),
        ));
        mul(p, _mm256_castsi256_ps(e))
    }
}

#[inline(always)]
pub unsafe fn sigmoid8(x: F8) -> F8 {
    unsafe { div(set(1.0), add(set(1.0), exp8(_mm256_xor_ps(x, set(-0.0))))) }
}

/// First index of the largest value.
#[target_feature(enable = "avx2,fma,f16c")]
pub fn argmax(x: &[f32]) -> usize {
    let mut peak = f32::NEG_INFINITY;
    for &v in x {
        peak = peak.max(v);
    }
    x.iter().position(|&v| v == peak).unwrap_or(0)
}

// n is a multiple of 8 in all of these.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn silu(x: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let v = ld(x.add(i));
            st(x.add(i), mul(v, sigmoid8(v)));
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn sigmoid(x: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            st(x.add(i), sigmoid8(ld(x.add(i))));
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn glu(value: *const f32, gate: *const f32, y: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            st(y.add(i), mul(ld(value.add(i)), sigmoid8(ld(gate.add(i)))));
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn softmax(x: *mut f32, n: usize) {
    unsafe {
        let mut m = ld(x);
        for i in (8..n).step_by(8) {
            m = max(m, ld(x.add(i)));
        }
        let peak = set(peak8(m));
        let mut sum = zero();
        for i in (0..n).step_by(8) {
            let e = exp8(sub(ld(x.add(i)), peak));
            sum = add(sum, e);
            st(x.add(i), e);
        }
        let inv = set(1.0 / sum8(sum));
        for i in (0..n).step_by(8) {
            st(x.add(i), mul(ld(x.add(i)), inv));
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn layer_norm(x: *const f32, w: *const f32, b: *const f32, y: *mut f32, n: usize) {
    unsafe {
        let mut s = zero();
        for i in (0..n).step_by(8) {
            s = add(s, ld(x.add(i)));
        }
        let mean = set(sum8(s) / n as f32);
        let mut ss = zero();
        for i in (0..n).step_by(8) {
            let d = sub(ld(x.add(i)), mean);
            ss = fma(d, d, ss);
        }
        let inv = set(1.0 / (sum8(ss) / n as f32 + 1e-5).sqrt());
        for i in (0..n).step_by(8) {
            let d = mul(sub(ld(x.add(i)), mean), inv);
            st(y.add(i), fma(d, ld(w.add(i)), ld(b.add(i))));
        }
    }
}

/// y[c] = bias[c] + the sum over taps j, in order, of x(j, c) * w[j][c] (a product then a sum, not fused).
#[inline(always)]
pub unsafe fn taps<X: Fn(usize, usize) -> F8>(
    bias: *const f32,
    w: &[*const f32],
    x: X,
    y: *mut f32,
    ch: usize,
    relu: bool,
) {
    unsafe {
        for c in (0..ch).step_by(8) {
            let mut acc = ld(bias.add(c));
            for (j, wj) in w.iter().enumerate() {
                acc = add(acc, mul(x(j, c), ld(wj.add(c))));
            }
            st(y.add(c), if relu { max(zero(), acc) } else { acc });
        }
    }
}
