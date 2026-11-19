//! Kokoro's kernels: direct conv on time tiles (4 output rows x 24 samples in registers), 1D Winograd for the
//! vocoder resblocks with snake fused in, register-blocked GEMM for BERT, and the Highway math the reference uses.

use crate::pool::{P, Pool};
use crate::simd::*;
use std::arch::x86_64::*;

pub const KM: usize = 4;
pub const HALO: usize = 64;
const NV: usize = 3;
pub const STEP: usize = NV * 8;
const TILE: usize = 240;

pub fn round_up(a: usize, m: usize) -> usize {
    a.div_ceil(m) * m
}

/// Highway's ReduceSum on 8 lanes: ((t0 + t4) + (t3 + t7)) + ((t1 + t5) + (t2 + t6)).
#[inline(always)]
pub unsafe fn hsum(v: F8) -> f32 {
    unsafe {
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), v);
        let s = [t[0] + t[4], t[1] + t[5], t[2] + t[6], t[3] + t[7]];
        (s[0] + s[3]) + (s[1] + s[2])
    }
}

#[inline(always)]
unsafe fn estrin4(x: F8, c0: f32, c1: f32, c2: f32, c3: f32) -> F8 {
    unsafe {
        let x2 = mul(x, x);
        fma(x2, fma(set(c3), x, set(c2)), fma(set(c1), x, set(c0)))
    }
}

#[inline(always)]
unsafe fn exp_poly(x: F8) -> F8 {
    unsafe {
        let x2 = mul(x, x);
        let x4 = mul(x2, x2);
        let e = fma(
            x4,
            fma(set(0.000198527617612853646278381), x, set(0.00139304355252534151077271)),
            fma(
                x2,
                fma(set(0.00833336077630519866943359), x, set(0.0416664853692054748535156)),
                fma(set(0.166666671633720397949219), x, set(0.5)),
            ),
        );
        fma(e, x2, x)
    }
}

#[inline(always)]
unsafe fn exp_q(x: F8) -> (__m256i, F8) {
    unsafe {
        let half = _mm256_or_ps(set(0.5), _mm256_and_ps(x, set(-0.0)));
        let q = _mm256_cvttps_epi32(fma(x, set(1.442695040888963407359924681), half));
        let qf = _mm256_cvtepi32_ps(q);
        let r = fma(qf, set(-0.693145751953125), x);
        (q, fma(qf, set(-1.428606765330187045e-6), r))
    }
}

#[inline(always)]
unsafe fn load_exp_short_range(x: F8, e: __m256i) -> F8 {
    unsafe {
        let pow2 =
            |v: __m256i| _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(v, _mm256_set1_epi32(0x7f))));
        let y = _mm256_srai_epi32::<1>(e);
        mul(mul(x, pow2(y)), pow2(_mm256_sub_epi32(e, y)))
    }
}

#[inline(always)]
pub unsafe fn hexp(x: F8) -> F8 {
    unsafe {
        let (q, r) = exp_q(x);
        let y = load_exp_short_range(add(exp_poly(r), set(1.0)), q);
        _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(x, set(-104.0)), y)
    }
}

#[inline(always)]
unsafe fn hexpm1(x: F8) -> F8 {
    unsafe {
        let (q, r) = exp_q(x);
        let y = exp_poly(r);
        let abs = _mm256_andnot_ps(set(-0.0), x);
        let big = sub(load_exp_short_range(add(y, set(1.0)), q), set(1.0));
        let z = _mm256_blendv_ps(big, y, _mm256_cmp_ps::<_CMP_LT_OQ>(abs, set(0.346573590279972654708616)));
        _mm256_blendv_ps(z, set(-1.0), _mm256_cmp_ps::<_CMP_LT_OQ>(x, set(-104.0)))
    }
}

#[inline(always)]
pub unsafe fn htanh(x: F8) -> F8 {
    unsafe {
        let sign = _mm256_and_ps(x, set(-0.0));
        let abs = _mm256_xor_ps(x, sign);
        let y = hexpm1(mul(abs, set(2.0)));
        let z =
            _mm256_blendv_ps(div(y, add(y, set(2.0))), set(1.0), _mm256_cmp_ps::<_CMP_GT_OQ>(abs, set(18.714973875)));
        _mm256_xor_ps(z, sign)
    }
}

#[inline(always)]
pub unsafe fn hsin(x: F8) -> F8 {
    unsafe {
        let abs = _mm256_andnot_ps(set(-0.0), x);
        let sign_x = _mm256_xor_ps(abs, x);
        let q = _mm256_cvttps_epi32(fma(abs, set(0.31830988618379067153), set(0.5)));
        let qf = _mm256_cvtepi32_ps(q);
        let mut r = fma(qf, set(-3.140625), abs);
        r = fma(qf, set(-0.0009670257568359375), r);
        r = fma(qf, set(-6.2771141529083251953e-7), r);
        r = fma(qf, set(-1.2154201256553420762e-10), r);
        let qsign = _mm256_castsi256_ps(_mm256_slli_epi32::<31>(_mm256_and_si256(q, _mm256_set1_epi32(1))));
        let r = _mm256_xor_ps(r, _mm256_xor_ps(qsign, sign_x));
        let y = mul(r, r);
        fma(
            estrin4(
                y,
                -1.66666597127914428710938e-1,
                8.33307858556509017944336e-3,
                -1.981069071916863322258e-4,
                2.6083159809786593541503e-6,
            ),
            mul(y, r),
            r,
        )
    }
}

/// Channel-major [C][T], time contiguous, HALO zeros around every row. Rows are padded to KM and time to STEP;
/// everything past T stays zero.
pub struct Act {
    pub c: usize,
    pub t: usize,
    pub stride: usize,
    pub buf: Buf<f32>,
}

impl Act {
    pub fn new(c: usize, t: usize) -> Act {
        let stride = round_up(round_up(t, STEP) + 2 * HALO, 16);
        Act { c, t, stride, buf: Buf::new(round_up(c, KM) * stride) }
    }
    pub fn row(&self, c: usize) -> *const f32 {
        unsafe { self.buf.as_ptr().add(c * self.stride + HALO) }
    }
    pub fn row_mut(&mut self, c: usize) -> *mut f32 {
        unsafe { self.buf.as_mut_ptr().add(c * self.stride + HALO) }
    }
    pub fn r(&self, c: usize) -> &[f32] {
        &self.buf[c * self.stride + HALO..c * self.stride + HALO + self.t]
    }
    pub fn rm(&mut self, c: usize) -> &mut [f32] {
        let (s, t) = (self.stride, self.t);
        &mut self.buf[c * s + HALO..c * s + HALO + t]
    }
    /// Row 0 sample 0 for writes from pool workers; rows are `stride` apart.
    pub fn base(&mut self) -> P<f32> {
        P(self.row_mut(0))
    }
    pub fn copy(&self) -> Act {
        let mut y = Act::new(self.c, self.t);
        y.buf.copy_from_slice(&self.buf);
        y
    }
}

#[derive(Default)]
pub struct Conv {
    pub w: Buf<f32>, // [Cout/KM][Cin][K][KM]
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub dil: usize,
    pub wm: usize, // Winograd F(wm, K) outputs per tile when enabled
    pub at: Vec<f32>,
    pub bt: Vec<f32>,
    pub u: Vec<Conv>,
}

pub fn pack_conv(
    cout: usize,
    cin: usize,
    k: usize,
    dil: usize,
    b: Option<&[f32]>,
    w: impl Fn(usize, usize, usize) -> f32,
) -> Conv {
    let cp = round_up(cout, KM);
    let mut c = Conv { w: Buf::new(cp * cin * k), b: vec![0.0; cp], cin, cout, k, dil, ..Default::default() };
    for co in 0..cp {
        for ci in 0..cin {
            for kk in 0..k {
                c.w[((co / KM * cin + ci) * k + kk) * KM + co % KM] = if co < cout { w(co, ci, kk) } else { 0.0 };
            }
        }
    }
    if let Some(b) = b {
        c.b[..cout].copy_from_slice(&b[..cout]);
    }
    c
}

/// x: row ci at x + ci * xs, sample t at [t] (caller guarantees halo). w: [Cout/KM][Cin][K][KM].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn conv_range(
    x: *const f32,
    xs: usize,
    w: *const f32,
    bias: *const f32,
    y: *mut f32,
    ys: usize,
    cin: usize,
    cout: usize,
    k: usize,
    dil: usize,
    t0: usize,
    t1: usize,
    tile: usize,
) {
    unsafe {
        let pad = dil * (k - 1) / 2;
        let mut tt = t0;
        while tt < t1 {
            let te = (tt + tile).min(t1);
            for cb in 0..cout / KM {
                let wb = w.add(cb * cin * k * KM);
                let mut t = tt;
                while t < te {
                    let mut a = [[zero(); NV]; KM];
                    for (r, ar) in a.iter_mut().enumerate() {
                        *ar = [set(*bias.add(cb * KM + r)); NV];
                    }
                    let mut wp = wb;
                    for ci in 0..cin {
                        let xr = x.add(ci * xs + t).sub(pad);
                        for kk in 0..k {
                            let xp = xr.add(kk * dil);
                            let xv = [ld(xp), ld(xp.add(8)), ld(xp.add(16))];
                            for (r, ar) in a.iter_mut().enumerate() {
                                let wv = set(*wp.add(r));
                                for v in 0..NV {
                                    ar[v] = fma(wv, xv[v], ar[v]);
                                }
                            }
                            wp = wp.add(KM);
                        }
                    }
                    for (r, ar) in a.iter().enumerate() {
                        let yr = y.add((cb * KM + r) * ys + t);
                        for (v, av) in ar.iter().enumerate() {
                            st(yr.add(v * 8), *av);
                        }
                    }
                    t += STEP;
                }
            }
            tt += tile;
        }
    }
}

pub fn conv_raw(pool: &Pool, c: &Conv, x: *const f32, xs: usize, y: *mut f32, ys: usize, t: usize) {
    let chunks = round_up(t, STEP) / STEP;
    let n = pool.n;
    let (x, y) = (P(x as *mut f32), P(y));
    pool.run(|i| unsafe {
        let (a, e) = (chunks * i / n * STEP, chunks * (i + 1) / n * STEP);
        conv_range(
            x.ptr(),
            xs,
            c.w.as_ptr(),
            c.b.as_ptr(),
            y.ptr(),
            ys,
            c.cin,
            round_up(c.cout, KM),
            c.k,
            c.dil,
            a,
            e,
            TILE,
        );
    });
}

pub fn apply(pool: &Pool, c: &Conv, x: &Act, y: &mut Act) {
    conv_raw(pool, c, x.row(0), x.stride, y.row_mut(0), y.stride, x.t);
    let e = round_up(x.t, STEP);
    for co in 0..round_up(c.cout, KM) {
        let r = y.row_mut(co);
        for t in y.t..e {
            unsafe { *r.add(t) = 0.0 };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub fn enable_winograd(c: &mut Conv, m: usize) {
    const PTS: [f64; 13] = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 1.0 / 3.0, -1.0 / 3.0, 4.0, -4.0];
    let r = c.k;
    let a = m + r - 1;
    let pw = |x: f64, e: usize| if e == 0 { 1.0 } else { x.powf(e as f64) };
    let row = |j: usize, i: usize, len: usize| if j == a - 1 { (i == len - 1) as u8 as f64 } else { pw(PTS[j], i) };
    let mut v = vec![0f64; a * a];
    let mut inv = vec![0f64; a * a];
    for j in 0..a {
        for i in 0..a {
            v[j * a + i] = row(j, i, a);
            inv[j * a + i] = (i == j) as u8 as f64;
        }
    }
    for col in 0..a {
        let mut piv = col;
        for j in col + 1..a {
            if v[j * a + col].abs() > v[piv * a + col].abs() {
                piv = j;
            }
        }
        for i in 0..a {
            v.swap(col * a + i, piv * a + i);
            inv.swap(col * a + i, piv * a + i);
        }
        let f = 1.0 / v[col * a + col];
        for i in 0..a {
            v[col * a + i] *= f;
            inv[col * a + i] *= f;
        }
        for j in 0..a {
            if j != col && v[j * a + col] != 0.0 {
                let g = v[j * a + col];
                for i in 0..a {
                    v[j * a + i] = (-g).mul_add(v[col * a + i], v[j * a + i]);
                    inv[j * a + i] = (-g).mul_add(inv[col * a + i], inv[j * a + i]);
                }
            }
        }
    }
    c.wm = m;
    c.at = (0..m * a).map(|i| row(i % a, i / a, m) as f32).collect();
    c.bt = (0..a * a).map(|i| inv[(i % a) * a + i / a] as f32).collect();
    let cp = round_up(c.cout, KM);
    let g: Vec<f64> = (0..a * r).map(|i| row(i / r, i % r, r)).collect();
    c.u = (0..a)
        .map(|x| {
            let mut u = Conv {
                w: Buf::new(cp * c.cin),
                b: vec![0.0; cp],
                cin: c.cin,
                cout: c.cout,
                k: 1,
                dil: 1,
                ..Default::default()
            };
            for blk in 0..cp / KM * c.cin {
                for lane in 0..KM {
                    let mut acc = 0f64;
                    for k in 0..r {
                        acc = g[x * r + k].mul_add(c.w[(blk * r + k) * KM + lane] as f64, acc);
                    }
                    u.w[blk * KM + lane] = acc as f32;
                }
            }
            u
        })
        .collect();
}

/// s row ci holds times [lo, hi): snake(a * x + b), zero outside [0, T) because torch zero-pads after the activation.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn snake_tile(
    x: *const f32,
    xs: usize,
    a: &[f32],
    b: &[f32],
    alpha: &[f32],
    s: *mut f32,
    ss: usize,
    cin: usize,
    lo: isize,
    hi: isize,
    tlen: isize,
) {
    unsafe {
        for ci in 0..cin {
            let xr = x.add(ci * xs);
            let sr = s.add(ci * ss);
            let (va, vb, al, ial) = (set(a[ci]), set(b[ci]), set(alpha[ci]), set(1.0 / alpha[ci]));
            let mut t = lo;
            while t < hi {
                let u = fma(va, ld(xr.offset(t)), vb);
                let sn = hsin(mul(al, u));
                st(sr.offset(t - lo), fma(ial, mul(sn, sn), u));
                t += 8;
            }
            for t in lo..hi.min(0) {
                *sr.offset(t - lo) = 0.0;
            }
            for t in tlen.max(lo)..hi {
                *sr.offset(t - lo) = 0.0;
            }
        }
    }
}

/// y[c][t] = (residual ? y[c][t] : 0) + o[c][t - t0] over [t0, t1), zero past T; per-row sum and sum of squares.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn conv_epilogue(
    o: *const f32,
    os: usize,
    y: *mut f32,
    ys: usize,
    cout: usize,
    t0: usize,
    t1: usize,
    tlen: usize,
    residual: bool,
    sum: &mut [f64],
    sq: &mut [f64],
) {
    unsafe {
        for c in 0..cout {
            let orow = o.add(c * os).sub(t0);
            let yr = y.add(c * ys);
            let (mut vs, mut vq) = (zero(), zero());
            let mut t = t0;
            while t < t1 {
                let mut v = ld(orow.add(t));
                if residual {
                    v = add(v, ld(yr.add(t)));
                }
                if t + 8 > tlen {
                    let n = tlen.saturating_sub(t) as i32;
                    let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
                    v = _mm256_and_ps(v, _mm256_castsi256_ps(mask));
                }
                st(yr.add(t), v);
                vs = add(vs, v);
                vq = fma(v, v, vq);
                t += 8;
            }
            sum[c] += hsum(vs) as f64;
            sq[c] += hsum(vq) as f64;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wino_input<const A: usize>(
    s: *const f32,
    ss: usize,
    cin: usize,
    bt: &[f32],
    m: usize,
    d: usize,
    n: usize,
    z: *mut f32,
    v: *mut f32,
    vs: usize,
    cs: usize,
) {
    unsafe {
        let dd = d * m;
        let zlen = n + (d + d * A) / dd + 2;
        for ci in 0..cin {
            let row = s.add(ci * ss);
            for q in 0..dd {
                for j in 0..zlen {
                    *z.add(q * zlen + j) = *row.add(q + dd * j);
                }
            }
            for p in 0..d {
                let mut zp = [std::ptr::null::<f32>(); A];
                for (u, zu) in zp.iter_mut().enumerate() {
                    *zu = z.add(((p + d * u) % dd) * zlen + (p + d * u) / dd);
                }
                let vo = v.add(ci * cs + p * n);
                for i in (0..n).step_by(8) {
                    let mut acc = [zero(); A];
                    for (u, zu) in zp.iter().enumerate() {
                        let xv = ld(zu.add(i));
                        for (x, ax) in acc.iter_mut().enumerate() {
                            *ax = fma(set(bt[x * A + u]), xv, *ax);
                        }
                    }
                    for (x, ax) in acc.iter().enumerate() {
                        st(vo.add(x * vs + i), *ax);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn wino_output<const A: usize>(
    mm: *const f32,
    vs: usize,
    cs: usize,
    cout: usize,
    at: &[f32],
    m: usize,
    d: usize,
    n: usize,
    bias: &[f32],
    o: *mut f32,
    os: usize,
) {
    unsafe {
        let dd = d * m;
        let mut tmp = [0f32; 8 * 16];
        for co in 0..cout {
            for p in 0..d {
                for i in (0..n).step_by(8) {
                    let mi = mm.add(co * cs + p * n + i);
                    let mut mv = [zero(); A];
                    for (x, v) in mv.iter_mut().enumerate() {
                        *v = ld(mi.add(x * vs));
                    }
                    for k in 0..m {
                        let mut r = set(bias[co]);
                        for (x, v) in mv.iter().enumerate() {
                            r = fma(set(at[k * A + x]), *v, r);
                        }
                        st(tmp.as_mut_ptr().add(k * 8), r);
                    }
                    let orow = o.add(co * os + p + dd * i);
                    for l in 0..8 {
                        for k in 0..m {
                            *orow.add(dd * l + d * k) = tmp[k * 8 + l];
                        }
                    }
                }
            }
        }
    }
}

macro_rules! wino_switch {
    ($f:ident, $a:expr, $($arg:expr),*) => {
        match $a {
            6 => $f::<6>($($arg),*),
            8 => $f::<8>($($arg),*),
            10 => $f::<10>($($arg),*),
            12 => $f::<12>($($arg),*),
            13 => $f::<13>($($arg),*),
            a => panic!("winograd alpha {a}"),
        }
    };
}

/// y = conv(snake(a * x + b)) (+ y if residual), with alpha per input channel; returns the moments of the new y.
#[allow(clippy::too_many_arguments)]
pub fn conv_snake(
    pool: &Pool,
    c: &Conv,
    x: &Act,
    a: &[f32],
    b: &[f32],
    alpha: &[f32],
    y: &mut Act,
    residual: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (m, aa, d) = (c.wm, c.u.len(), c.dil);
    let dd = d * m;
    let pad = d * (c.k - 1) / 2;
    let cout = round_up(c.cout, KM);
    let mut n = round_up((48 + d - 1) / d, 8);
    while d * n % 24 != 0 {
        n += 8;
    }
    let cols = d * n;
    let l = dd * n;
    let zlen = n + (d + d * aa) / dd + 2;
    let ss = round_up((l + 2 * pad + 16).max(dd * (zlen + 1)), 16);
    let tp = round_up(x.t, STEP);
    let tiles = tp.div_ceil(l);
    let (vs, ms, os) = (c.cin * cols, cout * cols, l + 16);
    let zero_b = vec![0f32; cout];
    let parts: Vec<std::sync::Mutex<Vec<f64>>> =
        (0..pool.n).map(|_| std::sync::Mutex::new(vec![0.0; 2 * cout])).collect();
    let (xb, xs, xt) = (P(x.row(0) as *mut f32), x.stride, x.t);
    let (yb, ys) = (y.base(), y.stride);
    pool.run(|i| unsafe {
        let mut s = Buf::<f32>::new(c.cin * ss);
        let mut z = Buf::<f32>::new(dd * zlen + 16);
        let mut v = Buf::<f32>::new(aa * vs);
        let mut mm = Buf::<f32>::new(aa * ms);
        let mut o = Buf::<f32>::new(cout * os);
        let mut part = parts[i].lock().unwrap();
        let (sum, sq) = part.split_at_mut(cout);
        for j in tiles * i / pool.n..tiles * (i + 1) / pool.n {
            let tt = j * l;
            let te = (tt + l).min(tp);
            let (lo, hi) = (tt as isize - pad as isize, (te + pad) as isize);
            snake_tile(xb.ptr(), xs, a, b, alpha, s.as_mut_ptr(), ss, c.cin, lo, hi, xt as isize);
            for ci in 0..c.cin {
                s[ci * ss + (hi - lo) as usize..(ci + 1) * ss].fill(0.0);
            }
            wino_switch!(
                wino_input,
                aa,
                s.as_ptr(),
                ss,
                c.cin,
                &c.bt,
                m,
                d,
                n,
                z.as_mut_ptr(),
                v.as_mut_ptr(),
                vs,
                cols
            );
            for xi in 0..aa {
                conv_range(
                    v.as_ptr().add(xi * vs),
                    cols,
                    c.u[xi].w.as_ptr(),
                    zero_b.as_ptr(),
                    mm.as_mut_ptr().add(xi * ms),
                    cols,
                    c.cin,
                    cout,
                    1,
                    1,
                    0,
                    cols,
                    cols,
                );
            }
            wino_switch!(wino_output, aa, mm.as_ptr(), ms, cols, c.cout, &c.at, m, d, n, &c.b, o.as_mut_ptr(), os);
            conv_epilogue(o.as_ptr(), os, yb.ptr(), ys, c.cout, tt, te, xt, residual, sum, sq);
        }
    });
    let mut sum = vec![0f64; c.cout];
    let mut sq = vec![0f64; c.cout];
    for p in &parts {
        let p = p.lock().unwrap();
        for co in 0..c.cout {
            sum[co] += p[co];
            sq[co] += p[cout + co];
        }
    }
    (sum, sq)
}

/// y[t][o] = b[o] + w[o] . x[t] for o in [o0, o1); w is torch Linear layout [out][in].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn linear_range(
    x: *const f32,
    n: usize,
    inp: usize,
    w: *const f32,
    b: *const f32,
    y: *mut f32,
    out: usize,
    o0: usize,
    o1: usize,
) {
    unsafe {
        for o in o0..o1 {
            let wr = w.add(o * inp);
            for t in 0..n {
                let xr = x.add(t * inp);
                let (mut a0, mut a1) = (zero(), zero());
                let mut i = 0;
                while i + 16 <= inp {
                    a0 = fma(ld(wr.add(i)), ld(xr.add(i)), a0);
                    a1 = fma(ld(wr.add(i + 8)), ld(xr.add(i + 8)), a1);
                    i += 16;
                }
                let mut s = hsum(add(a0, a1));
                while i < inp {
                    s = (*wr.add(i)).mul_add(*xr.add(i), s);
                    i += 1;
                }
                *y.add(t * out + o) = s + if b.is_null() { 0.0 } else { *b.add(o) };
            }
        }
    }
}

/// A torch Linear used in place: w [out][in] pointing into the loaded weights, bias optional.
#[derive(Clone, Copy)]
pub struct Lin {
    pub w: *const f32,
    pub b: *const f32,
    pub inp: usize,
    pub out: usize,
}
unsafe impl Send for Lin {}
unsafe impl Sync for Lin {}

impl Default for Lin {
    fn default() -> Self {
        Lin { w: std::ptr::null(), b: std::ptr::null(), inp: 0, out: 0 }
    }
}

impl Lin {
    pub fn run(&self, pool: Option<&Pool>, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0f32; n * self.out];
        self.run_into(pool, x.as_ptr(), n, y.as_mut_ptr());
        y
    }
    pub fn run_into(&self, pool: Option<&Pool>, x: *const f32, n: usize, y: *mut f32) {
        let (inp, out, w, b) = (self.inp, self.out, P(self.w as *mut f32), P(self.b as *mut f32));
        match pool {
            None => unsafe { linear_range(x, n, inp, self.w, self.b, y, out, 0, out) },
            Some(pool) => {
                let (x, y) = (P(x as *mut f32), P(y));
                pool.run(|i| unsafe {
                    linear_range(
                        x.ptr(),
                        n,
                        inp,
                        w.ptr(),
                        b.ptr(),
                        y.ptr(),
                        out,
                        out * i / pool.n,
                        out * (i + 1) / pool.n,
                    )
                });
            }
        }
    }
}

/// Linear weights packed at load into column panels [out/16][in][16] for a register-blocked GEMM.
pub struct Gemm {
    pub w: Buf<f32>,
    pub b: Buf<f32>,
    pub inp: usize,
    pub out: usize,
}

const MR: usize = 6;
// Depth per pass: a panel slice stays in L1 across all row tiles, and it bounds each sequential FMA chain.
const KC: usize = 128;

impl Gemm {
    /// w is torch Linear layout [out][in].
    pub fn pack(out: usize, inp: usize, w: &[f32], b: Option<&[f32]>) -> Gemm {
        let cols = round_up(out, 16);
        let mut g = Gemm { w: Buf::new(cols * inp), b: Buf::new(cols), inp, out };
        for o in 0..cols {
            g.b[o] = match b {
                Some(b) if o < out => b[o],
                _ => 0.0,
            };
            for i in 0..inp {
                g.w[(o / 16 * inp + i) * 16 + o % 16] = if o < out { w[o * inp + i] } else { 0.0 };
            }
        }
        g
    }

    /// Slice `part` of `parts` of y[n][out] = x W + b (+ res) (gelu applied before the residual).
    pub fn part(&self, x: *const f32, n: usize, y: *mut f32, part: usize, parts: usize, res: *const f32, gelu: bool) {
        let panels = self.out.div_ceil(16);
        unsafe {
            gemm_range(
                x,
                n,
                self.inp,
                self.w.as_ptr(),
                self.b.as_ptr(),
                y,
                self.out,
                panels * part / parts,
                panels * (part + 1) / parts,
                res,
                gelu,
            )
        };
    }

    pub fn run(&self, pool: &Pool, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.inp;
        let mut y = vec![0f32; n * self.out];
        let (xp, yp) = (P(x.as_ptr() as *mut f32), P(y.as_mut_ptr()));
        pool.run(|i| self.part(xp.ptr(), n, yp.ptr(), i, pool.n, std::ptr::null(), false));
        y
    }
}

#[inline(always)]
unsafe fn gelu(z: F8) -> F8 {
    unsafe {
        let u = mul(set(0.7978845608), fma(mul(set(0.044715), z), mul(z, z), z));
        mul(mul(set(0.5), z), add(set(1.0), htanh(u)))
    }
}

#[inline(always)]
unsafe fn load_n(p: *const f32, n: usize) -> F8 {
    unsafe {
        let mut t = [0f32; 8];
        std::ptr::copy_nonoverlapping(p, t.as_mut_ptr(), n);
        ld(t.as_ptr())
    }
}

#[inline(always)]
unsafe fn store_n(v: F8, p: *mut f32, n: usize) {
    unsafe {
        let mut t = [0f32; 8];
        st(t.as_mut_ptr(), v);
        std::ptr::copy_nonoverlapping(t.as_ptr(), p, n);
    }
}

/// c[r][0..cols) = (first ? bias : c[r]) + a[r][0..kc) * b[kc][16]. last applies the epilogue.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn tile<const R: usize>(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    kc: usize,
    c: *mut f32,
    ldc: usize,
    cols: usize,
    bias: *const f32,
    first: bool,
    last: bool,
    res: *const f32,
    gelu_on: bool,
) {
    unsafe {
        let full = cols == 16;
        let (c0, c1) = (cols.min(8), cols.saturating_sub(8).min(8));
        let mut acc = [[zero(); 2]; R];
        let mut bp = b;
        for k in 0..kc {
            let (b0, b1) = (ld(bp), ld(bp.add(8)));
            for (r, ar) in acc.iter_mut().enumerate() {
                let av = set(*a.add(r * lda + k));
                ar[0] = fma(av, b0, ar[0]);
                ar[1] = fma(av, b1, ar[1]);
            }
            bp = bp.add(ldb);
        }
        for (r, ar) in acc.iter().enumerate() {
            let (mut v0, mut v1) = (ar[0], ar[1]);
            let cr = c.add(r * ldc);
            if first && !bias.is_null() {
                v0 = add(v0, ld(bias));
                v1 = add(v1, ld(bias.add(8)));
            } else if !first && full {
                v0 = add(v0, ld(cr));
                v1 = add(v1, ld(cr.add(8)));
            } else if !first {
                v0 = add(v0, load_n(cr, c0));
                v1 = add(v1, load_n(cr.add(8), c1));
            }
            if last && gelu_on {
                v0 = gelu(v0);
                v1 = gelu(v1);
            }
            if last && !res.is_null() {
                v0 = add(v0, load_n(res.add(r * ldc), c0));
                v1 = add(v1, load_n(res.add(r * ldc + 8), c1));
            }
            if full {
                st(cr, v0);
                st(cr.add(8), v1);
            } else {
                store_n(v0, cr, c0);
                store_n(v1, cr.add(8), c1);
            }
        }
    }
}

/// Row tiles of a times one 16-wide column panel of b, for all n rows. Prefetches the next kc x 16 slice of
/// packed weights spread over the tiles, since the first tile of each slice otherwise stalls on L3.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn panel(
    a: *const f32,
    lda: usize,
    n: usize,
    b: *const f32,
    ldb: usize,
    kc: usize,
    c: *mut f32,
    ldc: usize,
    cols: usize,
    bias: *const f32,
    first: bool,
    last: bool,
    res: *const f32,
    gelu_on: bool,
    next: *const f32,
) {
    unsafe {
        let tiles = n.div_ceil(MR);
        let lines = kc * ldb / 16;
        let per = lines.div_ceil(tiles.max(1));
        let mut i = 0;
        while i < n {
            if !next.is_null() {
                let l0 = i / MR * per;
                for l in l0..lines.min(l0 + per) {
                    _mm_prefetch::<_MM_HINT_T0>(next.add(l * 16) as *const i8);
                }
            }
            let (ai, ci) = (a.add(i * lda), c.add(i * ldc));
            let ri = if res.is_null() { res } else { res.add(i * ldc) };
            match MR.min(n - i) {
                6 => tile::<6>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
                5 => tile::<5>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
                4 => tile::<4>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
                3 => tile::<3>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
                2 => tile::<2>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
                _ => tile::<1>(ai, lda, b, ldb, kc, ci, ldc, cols, bias, first, last, ri, gelu_on),
            }
            i += MR;
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemm_range(
    x: *const f32,
    n: usize,
    inp: usize,
    w: *const f32,
    bias: *const f32,
    y: *mut f32,
    out: usize,
    p0: usize,
    p1: usize,
    res: *const f32,
    gelu_on: bool,
) {
    unsafe {
        for p in p0..p1 {
            let col = p * 16;
            let wp = w.add(p * inp * 16);
            let mut k0 = 0;
            while k0 < inp {
                let kc = KC.min(inp - k0);
                let r = if res.is_null() { res } else { res.add(col) };
                panel(
                    x.add(k0),
                    inp,
                    n,
                    wp.add(k0 * 16),
                    16,
                    kc,
                    y.add(col),
                    out,
                    16.min(out - col),
                    bias.add(col),
                    k0 == 0,
                    k0 + kc == inp,
                    r,
                    gelu_on,
                    wp.add((k0 + kc) * 16),
                );
                k0 += kc;
            }
        }
    }
}

/// qkv rows are [q | k | v], each d = heads * hd wide; heads [h0, h1).
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn attention_range(qkv: *const f32, n: usize, d: usize, heads: usize, ctx: *mut f32, h0: usize, h1: usize) {
    unsafe {
        let hd = d / heads;
        let np = round_up(n, 16);
        let ld3 = 3 * d;
        let scale = set(1.0 / (hd as f32).sqrt());
        let mut kt = Buf::<f32>::new(hd * np);
        let mut s = Buf::<f32>::new(n * np);
        let null = std::ptr::null();
        for h in h0..h1 {
            let (q, k, v) = (qkv.add(h * hd), qkv.add(d + h * hd), qkv.add(2 * d + h * hd));
            for j in 0..n {
                for c in 0..hd {
                    kt[c * np + j] = *k.add(j * ld3 + c);
                }
            }
            for j in (0..np).step_by(16) {
                panel(
                    q,
                    ld3,
                    n,
                    kt.as_ptr().add(j),
                    np,
                    hd,
                    s.as_mut_ptr().add(j),
                    np,
                    16,
                    null,
                    true,
                    true,
                    null,
                    false,
                    null,
                );
            }
            for i in 0..n {
                let p = s.as_mut_ptr().add(i * np);
                for j in n..np {
                    *p.add(j) = f32::NEG_INFINITY;
                }
                let mut mx = set(f32::NEG_INFINITY);
                for j in (0..np).step_by(8) {
                    mx = max(mx, ld(p.add(j)));
                }
                let mx = set(peak8(mx));
                for j in (0..np).step_by(8) {
                    st(p.add(j), hexp(mul(sub(ld(p.add(j)), mx), scale)));
                }
                for j in n..np {
                    *p.add(j) = 0.0;
                }
                let mut sum = zero();
                for j in (0..np).step_by(8) {
                    sum = add(sum, ld(p.add(j)));
                }
                let inv = set(1.0 / hsum(sum));
                for j in (0..np).step_by(8) {
                    st(p.add(j), mul(ld(p.add(j)), inv));
                }
            }
            for c in (0..hd).step_by(16) {
                panel(
                    s.as_ptr(),
                    np,
                    n,
                    v.add(c),
                    ld3,
                    n,
                    ctx.add(h * hd + c),
                    d,
                    16,
                    null,
                    true,
                    true,
                    null,
                    false,
                    null,
                );
            }
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn layer_norm_rows(x: *mut f32, c: usize, g: &[f32], b: &[f32], eps: f32, r0: usize, r1: usize) {
    unsafe {
        for t in r0..r1 {
            let r = x.add(t * c);
            let mut s = zero();
            for i in (0..c).step_by(8) {
                s = add(s, ld(r.add(i)));
            }
            let m = set(hsum(s) / c as f32);
            let mut q = zero();
            for i in (0..c).step_by(8) {
                let z = sub(ld(r.add(i)), m);
                q = fma(z, z, q);
            }
            let inv = set(1.0 / (hsum(q) / c as f32 + eps).sqrt());
            for i in (0..c).step_by(8) {
                let z = mul(sub(ld(r.add(i)), m), inv);
                st(r.add(i), fma(z, ld(g.as_ptr().add(i)), ld(b.as_ptr().add(i))));
            }
        }
    }
}
