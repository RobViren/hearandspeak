//! Streaming polyphase resampler: Kaiser windowed sinc, passband flat to 91% of the lower Nyquist, stopband from
//! the lower Nyquist at 120 dB, the response soxr's HQ setting aims for.

use std::arch::x86_64::*;

const PASS: f64 = 0.91;
const ATTEN: f64 = 120.0;
const MAX_TAPS: usize = 1 << 16;
const MAX_TABLE: usize = 1 << 24;

/// A zero rate, or upsampling by more than MAX_TABLE / taps (about 95000x), produces no output. A ratio whose exact
/// table would pass MAX_TABLE floats (coprime rates such as 1264013 to 16000) uses the closest ratio that fits, which
/// can make the output length off by about 1e-5 of itself, more for extreme upsampling. A ratio past MAX_TAPS gets a
/// shorter filter.
pub struct Resampler {
    l: usize,
    m: usize,
    taps: usize,
    table: Vec<f32>,
    buf: Vec<f32>,
    // Absolute input index of buf[0], and the next output's position numerator: output n sits at input n * m / l.
    origin: usize,
    next: u64,
    inputs: u64,
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// to / from in lowest terms, else its last continued fraction convergent with numerator at most `lmax`, else 0 / 1.
fn ratio(to: u64, from: u64, lmax: u64) -> (u64, u64) {
    let g = gcd(to, from);
    if to / g <= lmax {
        return (to / g, from / g);
    }
    let (mut a, mut b) = (to, from);
    let (mut p0, mut q0, mut p1, mut q1) = (0, 1, 1, 0);
    let mut best = (0, 1);
    while b != 0 {
        let k = a / b;
        let (p, q) = (k * p1 + p0, k * q1 + q0);
        if p > lmax {
            break;
        }
        best = (p, q);
        (p0, q0, p1, q1) = (p1, q1, p, q);
        (a, b) = (b, a % b);
    }
    best
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > sum * 1e-17 {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        k += 1.0;
    }
    sum
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Resampler {
        if from == 0 || to == 0 {
            return Resampler {
                l: 0,
                m: 1,
                taps: 16,
                table: Vec::new(),
                buf: Vec::new(),
                origin: 0,
                next: 0,
                inputs: 0,
            };
        }
        let low = from.min(to) as f64 / 2.0;
        let width = (1.0 - PASS) * low / from as f64;
        let beta = 0.1102 * (ATTEN - 8.7);
        let taps = (((ATTEN - 7.95) / (14.36 * width)).ceil() as usize + 1).next_multiple_of(16).min(MAX_TAPS);
        let (l, m) = ratio(to as u64, from as u64, (MAX_TABLE / taps) as u64);
        let (l, m) = (l as usize, m as usize);
        let fc = (1.0 + PASS) / 2.0 * low / from as f64;
        let half = taps as f64 / 2.0;
        let mut table = vec![0f32; l * taps];
        for p in 0..l {
            let frac = p as f64 / l as f64;
            let row = &mut table[p * taps..(p + 1) * taps];
            let mut coef: Vec<f64> = (0..taps)
                .map(|i| {
                    let tau = frac + half - 1.0 - i as f64;
                    let x = tau / half;
                    if x.abs() >= 1.0 {
                        return 0.0;
                    }
                    let arg = std::f64::consts::PI * 2.0 * fc * tau;
                    let sinc = if tau == 0.0 { 1.0 } else { arg.sin() / arg };
                    2.0 * fc * sinc * bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
                })
                .collect();
            let sum: f64 = coef.iter().sum();
            coef.iter_mut().for_each(|c| *c /= sum);
            for (d, c) in row.iter_mut().zip(coef) {
                *d = c as f32;
            }
        }
        Resampler { l, m, taps, table, buf: vec![0.0; taps / 2 - 1], origin: 0, next: 0, inputs: 0 }
    }

    fn produce(&mut self, out: &mut Vec<f32>, limit: u64) {
        let (l, m) = (self.l as u64, self.m as u64);
        if l == 0 {
            self.buf.clear();
            return;
        }
        loop {
            let pos = self.next * m;
            let (base, phase) = ((pos / l) as usize, (pos % l) as usize);
            if self.next >= limit || base + self.taps > self.origin + self.buf.len() {
                break;
            }
            let x = &self.buf[base - self.origin..base - self.origin + self.taps];
            let h = &self.table[phase * self.taps..(phase + 1) * self.taps];
            out.push(unsafe { dot(x, h) });
            self.next += 1;
        }
        let first = ((self.next * m / l) as usize).min(self.origin + self.buf.len());
        if first > self.origin {
            self.buf.drain(..first - self.origin);
            self.origin = first;
        }
    }

    /// Resamples a chunk, returning what can be produced so far.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        self.inputs += input.len() as u64;
        self.produce(out, u64::MAX);
    }

    /// The tail: ceil(inputs * to / from) outputs in all.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        self.buf.extend(std::iter::repeat_n(0.0, self.taps));
        let total = (self.inputs * self.l as u64).div_ceil(self.m as u64);
        self.produce(out, total);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot(x: &[f32], h: &[f32]) -> f32 {
    unsafe {
        let mut a = [_mm256_setzero_ps(); 2];
        for i in (0..x.len()).step_by(16) {
            for (j, acc) in a.iter_mut().enumerate() {
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * j)),
                    _mm256_loadu_ps(h.as_ptr().add(i + 8 * j)),
                    *acc,
                );
            }
        }
        crate::simd::sum8(_mm256_add_ps(a[0], a[1]))
    }
}

/// Whole-buffer convenience.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }
    let mut r = Resampler::new(from, to);
    let mut out = Vec::new();
    r.push(input, &mut out);
    r.finish(&mut out);
    out
}
