//! Log-mel features: preemphasis, 512-point FFT in double, 128 mel bands, per-mel normalization.

use crate::pool::{P, Pool};
use std::arch::x86_64::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

pub const SAMPLE_RATE: usize = 16000;
pub const HOP: usize = 160;
pub const MELS: usize = 128;
const WIN: usize = 400;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;

struct Tables {
    window: [f32; FFT],
    bank: Vec<f32>,
    first: [usize; MELS],
    last: [usize; MELS],
    tw_re: [f64; FFT],
    tw_im: [f64; FFT],
    reversed: [usize; FFT],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            window: [0.0; FFT],
            bank: vec![0.0; MELS * BINS],
            first: [BINS; MELS],
            last: [0; MELS],
            tw_re: [0.0; FFT],
            tw_im: [0.0; FFT],
            reversed: [0; FFT],
        };
        for i in 0..WIN {
            t.window[(FFT - WIN) / 2 + i] = (0.5 - 0.5 * (2.0 * PI * i as f64 / (WIN - 1) as f64).cos()) as f32;
        }
        let to_hz = |mel: f64| {
            if mel >= 15.0 { 1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp() } else { 200.0 * mel / 3.0 }
        };
        let top = 15.0 + (8000.0f64 / 1000.0).ln() * (27.0 / 6.4f64.ln());
        let step = top / (MELS + 1) as f64;
        let edges: Vec<f64> = (0..MELS + 2).map(|i| to_hz(if i == MELS + 1 { top } else { i as f64 * step })).collect();
        for m in 0..MELS {
            for b in 0..BINS {
                let hz = b as f64 * 8000.0 / (BINS - 1) as f64;
                let rise = (hz - edges[m]) / (edges[m + 1] - edges[m]);
                let fall = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
                let lo = if fall < rise { fall } else { rise };
                let v = if 0.0 < lo { lo } else { 0.0 } * (2.0 / (edges[m + 2] - edges[m]));
                t.bank[m * BINS + b] = v as f32;
                if v > 0.0 {
                    t.first[m] = t.first[m].min(b);
                    t.last[m] = b + 1;
                }
            }
        }
        let mut half = 1;
        while half < FFT {
            for i in 0..half {
                let angle = -2.0 * PI * (i * (FFT / 2 / half)) as f64 / FFT as f64;
                t.tw_re[half - 1 + i] = angle.cos();
                t.tw_im[half - 1 + i] = angle.sin();
            }
            half *= 2;
        }
        for (i, r) in t.reversed.iter_mut().enumerate() {
            *r = (0..9).fold(0, |acc, b| acc | ((i >> b) & 1) << (8 - b));
        }
        t
    })
}

#[target_feature(enable = "avx2,fma")]
fn frame(t: &Tables, x: &[f32], out: &mut [f32]) {
    let mut re = [0f64; FFT];
    let mut im = [0f64; FFT];
    for i in 0..FFT {
        re[t.reversed[i]] = (x[i] * t.window[i]) as f64;
    }
    let (r, m) = (re.as_mut_ptr(), im.as_mut_ptr());
    let mut half = 1;
    while half < FFT {
        let (wr, wi) = (t.tw_re[half - 1..].as_ptr(), t.tw_im[half - 1..].as_ptr());
        for at in (0..FFT).step_by(2 * half) {
            unsafe {
                // Vectorizing a 1 or 2 trip loop gives masked loads and stores, several times slower than scalar.
                let butterfly = |i: usize| {
                    let (a, b) = (at + i, at + half + i);
                    let tr = *r.add(b) * *wr.add(i) - *m.add(b) * *wi.add(i);
                    let ti = *r.add(b) * *wi.add(i) + *m.add(b) * *wr.add(i);
                    *r.add(b) = *r.add(a) - tr;
                    *m.add(b) = *m.add(a) - ti;
                    *r.add(a) += tr;
                    *m.add(a) += ti;
                };
                if half < 4 {
                    butterfly(0);
                    if half == 2 {
                        butterfly(1);
                    }
                    continue;
                }
                for i in (0..half).step_by(4) {
                    let (a, b) = (at + i, at + half + i);
                    let (xr, xi) = (_mm256_loadu_pd(r.add(b)), _mm256_loadu_pd(m.add(b)));
                    let (cr, ci) = (_mm256_loadu_pd(wr.add(i)), _mm256_loadu_pd(wi.add(i)));
                    let tr = _mm256_sub_pd(_mm256_mul_pd(xr, cr), _mm256_mul_pd(xi, ci));
                    let ti = _mm256_add_pd(_mm256_mul_pd(xr, ci), _mm256_mul_pd(xi, cr));
                    let (yr, yi) = (_mm256_loadu_pd(r.add(a)), _mm256_loadu_pd(m.add(a)));
                    _mm256_storeu_pd(r.add(a), _mm256_add_pd(yr, tr));
                    _mm256_storeu_pd(m.add(a), _mm256_add_pd(yi, ti));
                    _mm256_storeu_pd(r.add(b), _mm256_sub_pd(yr, tr));
                    _mm256_storeu_pd(m.add(b), _mm256_sub_pd(yi, ti));
                }
            }
        }
        half *= 2;
    }
    let mut power = [0f32; BINS];
    for b in 0..BINS {
        let (r, i) = (re[b] as f32, im[b] as f32);
        power[b] = r * r + i * i;
    }
    for m in 0..MELS {
        let mut s = 0f32;
        for b in t.first[m]..t.last[m] {
            s += power[b] * t.bank[m * BINS + b];
        }
        out[m] = (s + f32::from_bits(103 << 23)).ln();
    }
}

/// [n / 160 + 1][128], normalized over the first valid = n / 160 frames. The trailing frame is zeroed.
pub fn log_mel(pool: &Pool, pcm: &[f32]) -> (Vec<f32>, usize, usize) {
    let t = tables();
    let n = pcm.len();
    let valid = n / HOP;
    let frames = valid + 1;
    let mut x = vec![0f32; n + FFT];
    let mut mel = vec![0f32; frames * MELS];
    {
        let px = P::new(&mut x);
        const GRAIN: usize = 1 << 14;
        pool.for_each(n.div_ceil(GRAIN), 1, |g| {
            for i in g * GRAIN..n.min(g * GRAIN + GRAIN) {
                unsafe { *px.at(FFT / 2 + i) = if i > 0 { pcm[i] - 0.97 * pcm[i - 1] } else { pcm[0] } };
            }
        });
    }
    let pm = P::new(&mut mel);
    pool.for_each(frames, 32, |f| unsafe { frame(t, &x[f * HOP..f * HOP + FFT], pm.slice(f * MELS, MELS)) });
    // Per mel statistics accumulate over frames in order; 16 mels per item keeps the walk on one cache line per frame.
    pool.for_each(MELS / 16, 1, |g| unsafe { normalize(pm.at(g * 16), valid) });
    (mel, frames, valid)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn normalize(col: *mut f32, valid: usize) {
    unsafe {
        let (mut sum, mut sq, mut mean) = ([0f64; 16], [0f64; 16], [0f64; 16]);
        let (mut fmean, mut spread) = ([0f32; 16], [0f32; 16]);
        for t in 0..valid {
            let row = std::slice::from_raw_parts(col.add(t * MELS), 16);
            for j in 0..16 {
                sum[j] += row[j] as f64;
            }
        }
        for j in 0..16 {
            mean[j] = sum[j] / valid as f64;
        }
        for t in 0..valid {
            let row = std::slice::from_raw_parts(col.add(t * MELS), 16);
            for j in 0..16 {
                let d = row[j] as f64 - mean[j];
                sq[j] += d * d;
            }
        }
        for j in 0..16 {
            fmean[j] = mean[j] as f32;
            spread[j] = (sq[j] / (valid as f64 - 1.0)).sqrt() as f32 + 1e-5;
        }
        for t in 0..valid {
            let row = std::slice::from_raw_parts_mut(col.add(t * MELS), 16);
            for j in 0..16 {
                row[j] = (row[j] - fmean[j]) / spread[j];
            }
        }
        std::ptr::write_bytes(col.add(valid * MELS), 0, 16);
    }
}
