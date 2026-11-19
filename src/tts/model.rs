//! Kokoro-82M: ALBERT, duration and prosody predictors, text encoder, and the iSTFTNet decoder.
//! The reference build lets clang fuse `a * b + c` in scalar code (-ffp-contract=on with FMA); every such
//! expression here is an explicit `mul_add` in the same place so the audio rounds the same way.

use super::ops::*;
use crate::pool::{P, Pool};
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering::*};

const STYLE: usize = 128;
const HARMONICS: usize = 9;
const NFFT: usize = 20;
const HOP: usize = 5;
const BINS: usize = NFFT / 2 + 1;
const UP_F0: usize = 300;
pub const SAMPLE_RATE: usize = 24000;

/// kokoro.bin (flat f32) with its kokoro.tsv index: name, offset in floats, shape.
pub struct Weights {
    data: crate::simd::Buf<f32>,
    index: HashMap<String, (usize, Vec<usize>)>,
}

impl Weights {
    pub fn load(bin: &Path, tsv: &Path) -> io::Result<Weights> {
        let mut f = std::fs::File::open(bin)?;
        let n = f.metadata()?.len() as usize / 4;
        let mut data = crate::simd::Buf::<f32>::new(n);
        f.read_exact(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) })?;
        Ok(Weights { data, index: parse_index(&std::fs::read_to_string(tsv)?) })
    }
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
    pub fn shape(&self, name: &str) -> &[usize] {
        &self.index[name].1
    }
    pub fn get(&self, name: &str) -> &[f32] {
        let (off, shape) = self.index.get(name).unwrap_or_else(|| panic!("missing weight {name}"));
        &self.data[*off..off + shape.iter().product::<usize>()]
    }
    fn ptr(&self, name: &str) -> *const f32 {
        if self.has(name) { self.get(name).as_ptr() } else { std::ptr::null() }
    }
}

fn parse_index(s: &str) -> HashMap<String, (usize, Vec<usize>)> {
    s.lines()
        .filter_map(|l| {
            let mut p = l.split('\t');
            let name = p.next()?.to_string();
            let off = p.next()?.parse().ok()?;
            let shape = p.next()?.split(',').filter_map(|d| d.parse().ok()).collect();
            Some((name, (off, shape)))
        })
        .collect()
}

struct Lstm {
    ih: Gemm, // both directions, [forward 4H | reverse 4H]
    hh: [Lin; 2],
    h: usize,
}

#[derive(Default)]
struct ResBlk {
    norm1: Lin,
    norm2: Lin,
    conv1: Conv,
    conv2: Conv,
    conv1x1: Option<Conv>,
    pool_w: Option<(*const f32, *const f32)>,
    cin: usize,
}
unsafe impl Send for ResBlk {}
unsafe impl Sync for ResBlk {}

struct ResBlock1 {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    adain1: Vec<Lin>,
    adain2: Vec<Lin>,
    alpha1: Vec<*const f32>,
    alpha2: Vec<*const f32>,
}
unsafe impl Send for ResBlock1 {}
unsafe impl Sync for ResBlock1 {}

/// stride s, kernel 2s, as s stride-1 two-tap convs, one per output phase.
struct ConvT {
    phase: Vec<Conv>,
    s: usize,
    p: usize,
    cout: usize,
}

struct Strided {
    w: *const f32,
    b: *const f32,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}
unsafe impl Send for Strided {}
unsafe impl Sync for Strided {}

pub struct Model {
    w: Weights,
    emb_in: Gemm,
    qkv: Gemm,
    dense: Gemm,
    ffn: Gemm,
    ffn_out: Gemm,
    bert_encoder: Gemm,
    duration_proj: Gemm,
    dur_lstm: Vec<Lstm>,
    dur_norm: Vec<Lin>,
    lstm: Lstm,
    shared: Lstm,
    f0: Vec<ResBlk>,
    n: Vec<ResBlk>,
    f0_proj: Conv,
    n_proj: Conv,
    text_cnn: Vec<Conv>,
    text_lstm: Lstm,
    f0_conv: Strided,
    n_conv: Strided,
    noise_conv0: Strided,
    encode: ResBlk,
    decode: Vec<ResBlk>,
    asr_res: Conv,
    noise_conv1: Conv,
    conv_post: Conv,
    ups: Vec<ConvT>,
    resblocks: Vec<ResBlock1>,
    noise_res: Vec<ResBlock1>,
}

fn lin(w: &Weights, p: &str) -> Lin {
    let s = w.shape(&format!("{p}.weight"));
    Lin { w: w.ptr(&format!("{p}.weight")), b: w.ptr(&format!("{p}.bias")), inp: s[1], out: s[0] }
}

fn gemm(w: &Weights, p: &str) -> Gemm {
    let s = w.shape(&format!("{p}.weight"));
    let b = format!("{p}.bias");
    Gemm::pack(s[0], s[1], w.get(&format!("{p}.weight")), w.has(&b).then(|| w.get(&b)))
}

/// Stacks same-shaped [out][in] linears along out.
fn stack_gemm(w: &Weights, ws: &[String], bs: &[String]) -> Gemm {
    let s = w.shape(&ws[0]).to_vec();
    let wt: Vec<f32> = ws.iter().flat_map(|n| w.get(n).iter().copied()).collect();
    let b: Vec<f32> = bs.iter().flat_map(|n| w.get(n).iter().copied()).collect();
    Gemm::pack(s[0] * ws.len(), s[1], &wt, Some(&b))
}

fn lstm(w: &Weights, p: &str) -> Lstm {
    let ih = stack_gemm(
        w,
        &[format!("{p}.weight_ih_l0"), format!("{p}.weight_ih_l0_reverse")],
        &[format!("{p}.bias_ih_l0"), format!("{p}.bias_ih_l0_reverse")],
    );
    let h = ih.out / 8;
    let hh = |sfx: &str| Lin {
        w: w.ptr(&format!("{p}.weight_hh{sfx}")),
        b: w.ptr(&format!("{p}.bias_hh{sfx}")),
        inp: h,
        out: 4 * h,
    };
    Lstm { hh: [hh("_l0"), hh("_l0_reverse")], ih, h }
}

fn conv(w: &Weights, p: &str, dil: usize) -> Conv {
    let s = w.shape(&format!("{p}.weight")).to_vec();
    let wp = w.get(&format!("{p}.weight"));
    let (cin, k) = (s[1], s[2]);
    let b = format!("{p}.bias");
    pack_conv(s[0], cin, k, dil, w.has(&b).then(|| w.get(&b)), |co, ci, kk| wp[(co * cin + ci) * k + kk])
}

fn conv_t(w: &Weights, p: &str, s: usize) -> ConvT {
    let sh = w.shape(&format!("{p}.weight")).to_vec();
    let wp = w.get(&format!("{p}.weight"));
    let (cin, cout, k) = (sh[0], sh[1], sh[2]);
    let b = w.get(&format!("{p}.bias"));
    let phase = (0..s)
        .map(|r| pack_conv(cout, cin, 2, 1, Some(b), |co, ci, kk| wp[(ci * cout + co) * k + r + s * (1 - kk)]))
        .collect();
    ConvT { phase, s, p: (k - s) / 2, cout }
}

fn strided(w: &Weights, p: &str, stride: usize, pad: usize) -> Strided {
    let s = w.shape(&format!("{p}.weight"));
    Strided {
        w: w.ptr(&format!("{p}.weight")),
        b: w.ptr(&format!("{p}.bias")),
        cin: s[1],
        cout: s[0],
        k: s[2],
        stride,
        pad,
    }
}

fn res_blk(w: &Weights, p: &str) -> ResBlk {
    let s = w.shape(&format!("{p}.conv1.weight"));
    ResBlk {
        norm1: lin(w, &format!("{p}.norm1.fc")),
        norm2: lin(w, &format!("{p}.norm2.fc")),
        conv1: conv(w, &format!("{p}.conv1"), 1),
        conv2: conv(w, &format!("{p}.conv2"), 1),
        conv1x1: w.has(&format!("{p}.conv1x1.weight")).then(|| conv(w, &format!("{p}.conv1x1"), 1)),
        pool_w: w
            .has(&format!("{p}.pool.weight"))
            .then(|| (w.ptr(&format!("{p}.pool.weight")), w.ptr(&format!("{p}.pool.bias")))),
        cin: s[1],
    }
}

fn res_block1(w: &Weights, p: &str) -> ResBlock1 {
    const DIL: [usize; 3] = [1, 3, 5];
    let mut b =
        ResBlock1 { convs1: vec![], convs2: vec![], adain1: vec![], adain2: vec![], alpha1: vec![], alpha2: vec![] };
    for j in 0..3 {
        b.convs1.push(conv(w, &format!("{p}.convs1.{j}"), DIL[j]));
        b.convs2.push(conv(w, &format!("{p}.convs2.{j}"), 1));
        b.adain1.push(lin(w, &format!("{p}.adain1.{j}.fc")));
        b.adain2.push(lin(w, &format!("{p}.adain2.{j}.fc")));
        b.alpha1.push(w.ptr(&format!("{p}.alpha1.{j}")));
        b.alpha2.push(w.ptr(&format!("{p}.alpha2.{j}")));
    }
    let k = b.convs1[0].k;
    let m = if k == 3 || k == 7 { 4 } else { 3 };
    for c in b.convs1.iter_mut().chain(b.convs2.iter_mut()) {
        unsafe { enable_winograd(c, m) };
    }
    b
}

impl Model {
    pub fn new(w: Weights) -> Model {
        let (bp, l, pr, g) = ("bert.", "bert.layer.", "predictor.", "decoder.generator.");
        let names = |a: &[&str], sfx: &str| a.iter().map(|n| format!("{l}attention.{n}.{sfx}")).collect::<Vec<_>>();
        let qkv =
            stack_gemm(&w, &names(&["query", "key", "value"], "weight"), &names(&["query", "key", "value"], "bias"));
        let resblocks: Vec<ResBlock1> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..6)
                .map(|i| {
                    s.spawn({
                        let w = &w;
                        move || res_block1(w, &format!("{g}resblocks.{i}"))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        Model {
            emb_in: gemm(&w, &format!("{bp}embedding_hidden_mapping_in")),
            qkv,
            dense: gemm(&w, &format!("{l}attention.dense")),
            ffn: gemm(&w, &format!("{l}ffn")),
            ffn_out: gemm(&w, &format!("{l}ffn_output")),
            bert_encoder: gemm(&w, "bert_encoder"),
            duration_proj: gemm(&w, &format!("{pr}duration_proj")),
            dur_lstm: (0..3).map(|i| lstm(&w, &format!("{pr}text_encoder.lstms.{}", 2 * i))).collect(),
            dur_norm: (0..3).map(|i| lin(&w, &format!("{pr}text_encoder.lstms.{}.fc", 2 * i + 1))).collect(),
            lstm: lstm(&w, &format!("{pr}lstm")),
            shared: lstm(&w, &format!("{pr}shared")),
            f0: (0..3).map(|i| res_blk(&w, &format!("{pr}F0.{i}"))).collect(),
            n: (0..3).map(|i| res_blk(&w, &format!("{pr}N.{i}"))).collect(),
            f0_proj: conv(&w, &format!("{pr}F0_proj"), 1),
            n_proj: conv(&w, &format!("{pr}N_proj"), 1),
            text_cnn: (0..3).map(|i| conv(&w, &format!("text_encoder.cnn.{i}.0"), 1)).collect(),
            text_lstm: lstm(&w, "text_encoder.lstm"),
            f0_conv: strided(&w, "decoder.F0_conv", 2, 1),
            n_conv: strided(&w, "decoder.N_conv", 2, 1),
            noise_conv0: strided(&w, &format!("{g}noise_convs.0"), 6, 3),
            encode: res_blk(&w, "decoder.encode"),
            decode: (0..4).map(|i| res_blk(&w, &format!("decoder.decode.{i}"))).collect(),
            asr_res: conv(&w, "decoder.asr_res.0", 1),
            noise_conv1: conv(&w, &format!("{g}noise_convs.1"), 1),
            conv_post: conv(&w, &format!("{g}conv_post"), 1),
            ups: vec![conv_t(&w, &format!("{g}ups.0"), 10), conv_t(&w, &format!("{g}ups.1"), 6)],
            resblocks,
            noise_res: (0..2).map(|i| res_block1(&w, &format!("{g}noise_res.{i}"))).collect(),
            w,
        }
    }

    fn g(&self, name: &str) -> &[f32] {
        self.w.get(name)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Spins, then yields: the BERT layers are many short phases.
struct Barrier {
    waiting: AtomicU32,
    generation: AtomicU32,
    n: u32,
}

impl Barrier {
    fn wait(&self) {
        let g = self.generation.load(Acquire);
        if self.waiting.fetch_add(1, AcqRel) == self.n - 1 {
            self.waiting.store(0, Relaxed);
            self.generation.store(g + 1, Release);
        } else {
            let mut spins = 0;
            while self.generation.load(Acquire) == g {
                if spins < 4096 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                spins += 1;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
fn layer_norm(x: &mut [f32], g: Option<&[f32]>, b: Option<&[f32]>, eps: f32) {
    let c = x.len();
    let mut mean = 0f64;
    for &v in x.iter() {
        mean += v as f64;
    }
    mean /= c as f64;
    let mut var = 0f64;
    for &v in x.iter() {
        let d = v as f64 - mean;
        var = d.mul_add(d, var);
    }
    let m = mean as f32;
    let inv = (1.0 / (var / c as f64 + eps as f64).sqrt()) as f32;
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((*v - m) * inv).mul_add(g.map_or(1.0, |g| g[i]), b.map_or(0.0, |b| b[i]));
    }
}

fn run_lstm(pool: &Pool, m: &Lstm, x: &[f32]) -> Vec<f32> {
    let h = m.h;
    let n = x.len() / m.ih.inp;
    let mut y = vec![0f32; n * 2 * h];
    let g = m.ih.run(pool, x);
    let yp = P::new(&mut y);
    pool.for_split(2, |d| unsafe { lstm_dir(m, &g, n, d, yp) });
    y
}

#[target_feature(enable = "avx2,fma")]
unsafe fn lstm_dir(m: &Lstm, g: &[f32], n: usize, d: usize, y: P<f32>) {
    let h = m.h;
    let mut hs = vec![0f32; h];
    let mut c = vec![0f32; h];
    let mut r = vec![0f32; 4 * h];
    for s in 0..n {
        let t = if d == 1 { n - 1 - s } else { s };
        m.hh[d].run_into(None, hs.as_ptr(), 1, r.as_mut_ptr());
        let gt = &g[t * 8 * h + d * 4 * h..];
        for j in 0..h {
            let i = sigmoid(gt[j] + r[j]);
            let f = sigmoid(gt[h + j] + r[h + j]);
            let gg = (gt[2 * h + j] + r[2 * h + j]).tanh();
            let o = sigmoid(gt[3 * h + j] + r[3 * h + j]);
            c[j] = f.mul_add(c[j], i * gg);
            hs[j] = o * c[j].tanh();
            unsafe { *y.at(t * 2 * h + d * h + j) = hs[j] };
        }
    }
}

fn map(pool: &Pool, x: &mut Act, f: impl Fn(usize, f32) -> f32 + Sync) {
    let (b, s, t) = (x.base(), x.stride, x.t);
    pool.for_split(x.c, |c| {
        let r = unsafe { b.slice(c * s, t) };
        for v in r.iter_mut() {
            *v = f(c, *v);
        }
    });
}

fn leaky(pool: &Pool, x: &mut Act, slope: f32) {
    map(pool, x, |_, v| if v < 0.0 { v * slope } else { v });
}

fn adain(pool: &Pool, fc: &Lin, style: &[f32], x: &Act, y: &mut Act) {
    let gb = fc.run(None, style, 1);
    let c_n = fc.out / 2;
    let (yb, ys) = (y.base(), y.stride);
    pool.for_split(c_n, |c| unsafe { adain_row(x.r(c), &gb, c, c_n, yb.slice(c * ys, x.t)) });
}

#[target_feature(enable = "avx2,fma")]
fn adain_row(r: &[f32], gb: &[f32], c: usize, c_n: usize, o: &mut [f32]) {
    let t = r.len();
    let mut mean = 0f64;
    for &v in r {
        mean += v as f64;
    }
    mean /= t as f64;
    let mut var = 0f64;
    for &v in r {
        let d = v as f64 - mean;
        var = d.mul_add(d, var);
    }
    let m = mean as f32;
    let inv = (1.0 / (var / t as f64 + 1e-5).sqrt()) as f32;
    let (g, b) = (1.0 + gb[c], gb[c_n + c]);
    for (ov, &v) in o.iter_mut().zip(r) {
        *ov = g.mul_add((v - m) * inv, b);
    }
}

fn run_strided(pool: &Pool, s: &Strided, x: &Act) -> Act {
    let mut y = Act::new(s.cout, (x.t + 2 * s.pad - s.k) / s.stride + 1);
    let (yb, ys, yt) = (y.base(), y.stride, y.t);
    pool.for_split(s.cout, |co| unsafe { strided_row(s, x, co, yb.slice(co * ys, yt)) });
    y
}

#[target_feature(enable = "avx2,fma")]
unsafe fn strided_row(s: &Strided, x: &Act, co: usize, yr: &mut [f32]) {
    unsafe {
        yr.fill(*s.b.add(co));
        for ci in 0..s.cin {
            for k in 0..s.k {
                let wv = *s.w.add((co * s.cin + ci) * s.k + k);
                let xr = x.row(ci).offset(k as isize - s.pad as isize);
                for (t, y) in yr.iter_mut().enumerate() {
                    *y = wv.mul_add(*xr.add(t * s.stride), *y);
                }
            }
        }
    }
}

fn run_conv_t(pool: &Pool, c: &ConvT, x: &Act) -> Act {
    let mut z = Act::new(c.cout, x.t + 1);
    let mut y = Act::new(c.cout, x.t * c.s);
    for r in 0..c.s {
        conv_raw(pool, &c.phase[r], unsafe { x.row(0).sub(1) }, x.stride, z.row_mut(0), z.stride, z.t);
        let (yb, ys, yt) = (y.base(), y.stride, y.t as isize);
        let zr = &z;
        pool.for_split(c.cout, |co| {
            let zrow = zr.r(co);
            for (q, &v) in zrow.iter().enumerate() {
                let o = (q * c.s + r) as isize - c.p as isize;
                if o >= 0 && o < yt {
                    unsafe { *yb.at(co * ys + o as usize) = v };
                }
            }
        });
    }
    y
}

fn run_res_blk(pool: &Pool, b: &ResBlk, style: &[f32], x: &Act) -> Act {
    let mut a = Act::new(b.cin, x.t);
    adain(pool, &b.norm1, style, x, &mut a);
    leaky(pool, &mut a, 0.2);
    let t = if b.pool_w.is_some() { 2 * x.t } else { x.t };
    if let Some((pw, pb)) = b.pool_w {
        let mut u = Act::new(b.cin, t);
        let (ub, us) = (u.base(), u.stride);
        let (pw, pb) = (P(pw as *mut f32), P(pb as *mut f32));
        let ar = &a;
        pool.for_split(b.cin, |c| unsafe { pool_row(pw.at(3 * c), *pb.at(c), ar.row(c), x.t, ub.at(c * us)) });
        a = u;
    }
    let cout = b.conv1.cout;
    let mut h = Act::new(cout, t);
    let mut r = Act::new(cout, t);
    apply(pool, &b.conv1, &a, &mut h);
    adain(pool, &b.norm2, style, &h, &mut r);
    leaky(pool, &mut r, 0.2);
    apply(pool, &b.conv2, &r, &mut h);
    let up = b.pool_w.map(|_| {
        let mut up = Act::new(b.cin, t);
        let (ub, us) = (up.base(), up.stride);
        pool.for_split(b.cin, |c| {
            let xr = x.r(c);
            for tt in 0..t {
                unsafe { *ub.at(c * us + tt) = xr[tt / 2] };
            }
        });
        up
    });
    let sx = up.as_ref().unwrap_or(x);
    if let Some(c1) = &b.conv1x1 {
        apply(pool, c1, sx, &mut r);
    }
    let sc = if b.conv1x1.is_some() { &r } else { sx };
    let k = 0.5f64.sqrt() as f32;
    let (hb, hs) = (h.base(), h.stride);
    pool.for_split(cout, |c| {
        let hr = unsafe { hb.slice(c * hs, t) };
        for (v, s) in hr.iter_mut().zip(sc.r(c)) {
            *v = (*v + s) * k;
        }
    });
    h
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pool_row(w: *const f32, pb: f32, r: *const f32, t: usize, u: *mut f32) {
    unsafe {
        let w = std::slice::from_raw_parts(w, 3);
        for m in 0..t {
            *u.add(2 * m) = w[1].mul_add(*r.add(m), pb);
            *u.add(2 * m + 1) = w[2].mul_add(*r.add(m), w[0].mul_add(*r.add(m + 1), pb));
        }
    }
}

fn row_moments(pool: &Pool, x: &Act) -> (Vec<f64>, Vec<f64>) {
    let mut m = vec![(0f64, 0f64); x.c];
    let mp = P::new(&mut m);
    pool.for_split(x.c, |c| unsafe { *mp.at(c) = moments(x.r(c)) });
    m.into_iter().unzip()
}

#[target_feature(enable = "avx2,fma")]
fn moments(r: &[f32]) -> (f64, f64) {
    let (mut s, mut q) = (0f64, 0f64);
    for &v in r {
        s += v as f64;
        q = (v as f64).mul_add(v as f64, q);
    }
    (s, q)
}

/// InstanceNorm then (1 + gamma), beta, folded to y = a * x + b per channel.
#[target_feature(enable = "avx2,fma")]
fn adain_affine(fc: &Lin, style: &[f32], m: &(Vec<f64>, Vec<f64>), t: usize) -> (Vec<f32>, Vec<f32>) {
    let gb = fc.run(None, style, 1);
    let c_n = fc.out / 2;
    let (mut a, mut b) = (vec![0f32; c_n], vec![0f32; c_n]);
    for c in 0..c_n {
        let mean = m.0[c] / t as f64;
        let inv = 1.0 / ((-mean).mul_add(mean, m.1[c] / t as f64).max(0.0) + 1e-5).sqrt();
        let g = 1.0 + gb[c] as f64;
        a[c] = (g * inv) as f32;
        b[c] = (-(g * mean)).mul_add(inv, gb[c_n + c] as f64) as f32;
    }
    (a, b)
}

fn run_res_block1(pool: &Pool, b: &ResBlock1, style: &[f32], x: &mut Act) {
    let (c, t) = (x.c, x.t);
    let mut u = Act::new(c, t);
    let mut mx = row_moments(pool, x);
    let alpha = |p: *const f32| unsafe { std::slice::from_raw_parts(p, c) };
    for j in 0..3 {
        let (sa, sb) = unsafe { adain_affine(&b.adain1[j], style, &mx, t) };
        let mu = conv_snake(pool, &b.convs1[j], x, &sa, &sb, alpha(b.alpha1[j]), &mut u, false);
        let (sa, sb) = unsafe { adain_affine(&b.adain2[j], style, &mu, t) };
        mx = conv_snake(pool, &b.convs2[j], &u, &sa, &sb, alpha(b.alpha2[j]), x, true);
    }
}

/// F.interpolate(mode="linear") as torch's AVX2 build contracts it.
#[target_feature(enable = "avx2,fma")]
fn interp(x: &[f32], nout: usize, scale: f32) -> Vec<f32> {
    let nin = x.len();
    (0..nout)
        .map(|i| {
            let src = scale.mul_add(i as f32 + 0.5, -0.5).max(0.0);
            let i0 = (src as usize).min(nin - 1);
            let i1 = i0 + (i0 < nin - 1) as usize;
            let l = (src - i0 as f32).clamp(0.0, 1.0);
            x[i0].mul_add(1.0 - l, x[i1] * l)
        })
        .collect()
}

struct Twiddle {
    c: [f64; NFFT],
    s: [f64; NFFT],
}

fn twiddle() -> &'static Twiddle {
    static T: std::sync::OnceLock<Twiddle> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Twiddle { c: [0.0; NFFT], s: [0.0; NFFT] };
        for j in 0..NFFT {
            let a = 2.0 * std::f64::consts::PI * j as f64 / NFFT as f64;
            t.c[j] = a.cos();
            t.s[j] = a.sin();
            if j % (NFFT / 4) == 0 {
                t.c[j] = t.c[j].round();
                t.s[j] = t.s[j].round();
            }
        }
        t
    })
}

fn hann() -> &'static [f32; NFFT] {
    static W: std::sync::OnceLock<[f32; NFFT]> = std::sync::OnceLock::new();
    W.get_or_init(|| {
        std::array::from_fn(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / NFFT as f64).cos()) as f32)
    })
}

/// Initial harmonic phases and the noise of the source, drawn as libstdc++ draws them from mt19937.
pub struct Noise {
    pub initial_phase: Vec<f32>,
    pub randn: Vec<f32>,
    pub seed: u32,
}

#[target_feature(enable = "avx2,fma")]
fn draw_noise(seed: u32, len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = super::rng::Mt19937::new(seed);
    let phase: Vec<f32> = (0..HARMONICS).map(|_| rng.uniform()).collect();
    let mut normal = super::rng::Normal::default();
    let randn = (0..len).map(|_| normal.sample(&mut rng)).collect();
    (phase, randn)
}

fn source(pool: &Pool, m: &Model, f0: &[f32], nz: &mut Noise) -> Vec<f32> {
    let f = f0.len();
    let l = f * UP_F0;
    if nz.initial_phase.is_empty() {
        (nz.initial_phase, nz.randn) = unsafe { draw_noise(nz.seed, l * HARMONICS) };
    }
    let mut sines: Vec<Vec<f32>> = vec![Vec::new(); HARMONICS];
    let sp = P::new(&mut sines);
    let phase0 = &nz.initial_phase;
    pool.for_split(HARMONICS, |h| unsafe { *sp.at(h) = harmonic(f0, h, phase0[h], l) });
    unsafe {
        mix_source(
            f0,
            &sines,
            &nz.randn,
            m.g("decoder.generator.m_source.l_linear.weight"),
            m.g("decoder.generator.m_source.l_linear.bias")[0],
        )
    }
}

#[target_feature(enable = "avx2,fma")]
fn harmonic(f0: &[f32], h: usize, phase0: f32, l: usize) -> Vec<f32> {
    let mut rad: Vec<f32> = (0..l)
        .map(|t| {
            let a = f0[t / UP_F0] * (h + 1) as f32 / SAMPLE_RATE as f32;
            a - a.floor()
        })
        .collect();
    if h > 0 {
        rad[0] += phase0;
    }
    let mut ph = interp(&rad, f0.len(), UP_F0 as f32);
    let mut acc = 0f64;
    for v in ph.iter_mut() {
        acc += *v as f64;
        *v = acc as f32 * 2.0 * std::f32::consts::PI * UP_F0 as f32;
    }
    let mut s = interp(&ph, l, (1.0 / UP_F0 as f64) as f32);
    for v in s.iter_mut() {
        *v = v.sin() * 0.1;
    }
    s
}

#[target_feature(enable = "avx2,fma")]
fn mix_source(f0: &[f32], sines: &[Vec<f32>], randn: &[f32], lw: &[f32], lb: f32) -> Vec<f32> {
    let l = sines[0].len();
    (0..l)
        .map(|t| {
            let uv = if f0[t / UP_F0] > 10.0 { 1.0f32 } else { 0.0 };
            let amp = uv.mul_add(0.003, (1.0 - uv) * 0.1 / 3.0);
            let mut s = lb;
            for h in 0..HARMONICS {
                let inner = sines[h][t].mul_add(uv, amp * randn[t * HARMONICS + h]);
                s = lw[h].mul_add(inner, s);
            }
            s.tanh()
        })
        .collect()
}

fn stft(pool: &Pool, x: &[f32]) -> Act {
    let l = x.len();
    let f = l / HOP + 1;
    let mut y = Act::new(2 * BINS, f);
    let (yb, ys) = (y.base(), y.stride);
    pool.for_split(f, |fi| unsafe { stft_frame(x, fi, yb, ys) });
    y
}

#[target_feature(enable = "avx2,fma")]
unsafe fn stft_frame(x: &[f32], f: usize, y: P<f32>, ys: usize) {
    let (w, tw) = (hann(), twiddle());
    let l = x.len() as isize;
    let mut fr = [0f32; NFFT];
    for n in 0..NFFT {
        let mut j = (f * HOP + n) as isize - (NFFT / 2) as isize;
        j = if j < 0 {
            -j
        } else if j >= l {
            2 * (l - 1) - j
        } else {
            j
        };
        fr[n] = w[n] * x[j as usize];
    }
    for k in 0..BINS {
        let (mut re, mut im) = (0f64, 0f64);
        for n in 0..NFFT {
            re = (fr[n] as f64).mul_add(tw.c[k * n % NFFT], re);
            im = (-(fr[n] as f64)).mul_add(tw.s[k * n % NFFT], im);
        }
        unsafe {
            *y.at(k * ys + f) = re.mul_add(re, im * im).sqrt() as f32;
            *y.at((BINS + k) * ys + f) = im.atan2(re) as f32;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
fn istft(x: &Act) -> Vec<f32> {
    let f = x.t;
    let n = HOP * (f - 1) + NFFT;
    let (w, tw) = (hann(), twiddle());
    let mut ola = vec![0f32; n];
    let mut env = vec![0f32; n];
    let rows: Vec<&[f32]> = (0..2 * BINS).map(|k| x.r(k)).collect();
    for fi in 0..f {
        let mut re = [0f64; BINS];
        let mut im = [0f64; BINS];
        for k in 0..BINS {
            let mag = rows[k][fi].exp();
            let ph = rows[BINS + k][fi].sin();
            re[k] = (mag * ph.cos()) as f64;
            im[k] = (mag * ph.sin()) as f64;
        }
        for j in 0..NFFT {
            let mut s = re[0] + if j % 2 == 1 { -re[BINS - 1] } else { re[BINS - 1] };
            for k in 1..BINS - 1 {
                let inner = re[k].mul_add(tw.c[k * j % NFFT], -(im[k] * tw.s[k * j % NFFT]));
                s = 2.0f64.mul_add(inner, s);
            }
            let o = fi * HOP + j;
            ola[o] = ((s / NFFT as f64) as f32).mul_add(w[j], ola[o]);
            env[o] = w[j].mul_add(w[j], env[o]);
        }
    }
    (0..HOP * (f - 1)).map(|j| ola[j + NFFT / 2] / env[j + NFFT / 2]).collect()
}

fn cat(parts: &[&Act]) -> Act {
    let c: usize = parts.iter().map(|p| p.c).sum();
    let mut y = Act::new(c, parts[0].t);
    let mut ci = 0;
    for p in parts {
        for i in 0..p.c {
            y.rm(ci).copy_from_slice(p.r(i));
            ci += 1;
        }
    }
    y
}

fn curve(v: &[f32]) -> Act {
    let mut a = Act::new(1, v.len());
    a.rm(0).copy_from_slice(v);
    a
}

fn bert(pool: &Pool, m: &Model, ids: &[usize]) -> Vec<f32> {
    let n = ids.len();
    let e = 128;
    let d = m.dense.out;
    let (word, pos, tok) = (
        m.g("bert.embeddings.word_embeddings.weight"),
        m.g("bert.embeddings.position_embeddings.weight"),
        m.g("bert.embeddings.token_type_embeddings.weight"),
    );
    let (lg, lb) = (m.g("bert.embeddings.LayerNorm.weight"), m.g("bert.embeddings.LayerNorm.bias"));
    let mut emb = vec![0f32; n * e];
    for t in 0..n {
        let r = &mut emb[t * e..(t + 1) * e];
        for c in 0..e {
            r[c] = word[ids[t] * e + c] + tok[c] + pos[t * e + c];
        }
        unsafe { layer_norm(r, Some(lg), Some(lb), 1e-12) };
    }
    let (ag, ab) = (m.g("bert.layer.attention.LayerNorm.weight"), m.g("bert.layer.attention.LayerNorm.bias"));
    let (fg, fb) = (m.g("bert.layer.full_layer_layer_norm.weight"), m.g("bert.layer.full_layer_layer_norm.bias"));
    let mut x = vec![0f32; n * d];
    let mut qkv = vec![0f32; n * 3 * d];
    let mut ctx = vec![0f32; n * d];
    let mut a = vec![0f32; n * d];
    let mut f = vec![0f32; n * m.ffn.out];
    let (xp, qp, cp, ap, fp) = (P::new(&mut x), P::new(&mut qkv), P::new(&mut ctx), P::new(&mut a), P::new(&mut f));
    let ep = P(emb.as_mut_ptr());
    let bar = Barrier { waiting: AtomicU32::new(0), generation: AtomicU32::new(0), n: pool.n as u32 };
    pool.run(|i| unsafe {
        let parts = pool.n;
        let null = std::ptr::null();
        m.emb_in.part(ep.ptr(), n, xp.ptr(), i, parts, null, false);
        bar.wait();
        for _ in 0..12 {
            m.qkv.part(xp.ptr(), n, qp.ptr(), i, parts, null, false);
            bar.wait();
            attention_range(qp.ptr(), n, d, 12, cp.ptr(), 12 * i / parts, 12 * (i + 1) / parts);
            bar.wait();
            m.dense.part(cp.ptr(), n, ap.ptr(), i, parts, xp.ptr(), false);
            bar.wait();
            layer_norm_rows(ap.ptr(), d, ag, ab, 1e-12, n * i / parts, n * (i + 1) / parts);
            bar.wait();
            m.ffn.part(ap.ptr(), n, fp.ptr(), i, parts, null, true);
            bar.wait();
            m.ffn_out.part(fp.ptr(), n, xp.ptr(), i, parts, ap.ptr(), false);
            bar.wait();
            layer_norm_rows(xp.ptr(), d, fg, fb, 1e-12, n * i / parts, n * (i + 1) / parts);
            bar.wait();
        }
    });
    x
}

fn with_style(x: &[f32], c: usize, style: &[f32]) -> Vec<f32> {
    x.chunks_exact(c).flat_map(|r| r.iter().chain(style.iter()).copied()).collect()
}

fn duration_encoder(pool: &Pool, m: &Model, x: &[f32], prosody: &[f32]) -> Vec<f32> {
    let mut d = with_style(x, 512, prosody);
    for l in 0..3 {
        let mut h = run_lstm(pool, &m.dur_lstm[l], &d);
        let gb = m.dur_norm[l].run(None, prosody, 1);
        let c = m.dur_norm[l].out / 2;
        for r in h.chunks_exact_mut(c) {
            unsafe { norm_style(r, &gb, c) };
        }
        d = with_style(&h, c, prosody);
    }
    d
}

#[target_feature(enable = "avx2,fma")]
fn norm_style(r: &mut [f32], gb: &[f32], c: usize) {
    layer_norm(r, None, None, 1e-5);
    for i in 0..c {
        r[i] = (1.0 + gb[i]).mul_add(r[i], gb[c + i]);
    }
}

fn text_encoder(pool: &Pool, m: &Model, ids: &[usize]) -> Act {
    let (n, c) = (ids.len(), 512);
    let emb = m.g("text_encoder.embedding.weight");
    let mut x = Act::new(c, n);
    let mut y = Act::new(c, n);
    for ci in 0..c {
        let r = x.rm(ci);
        for t in 0..n {
            r[t] = emb[ids[t] * c + ci];
        }
    }
    for l in 0..3 {
        apply(pool, &m.text_cnn[l], &x, &mut y);
        let (g, b) = (m.g(&format!("text_encoder.cnn.{l}.1.gamma")), m.g(&format!("text_encoder.cnn.{l}.1.beta")));
        let mut col = vec![0f32; c];
        for t in 0..n {
            for ci in 0..c {
                col[ci] = y.r(ci)[t];
            }
            unsafe { layer_norm(&mut col, Some(g), Some(b), 1e-5) };
            for ci in 0..c {
                x.rm(ci)[t] = if col[ci] < 0.0 { 0.2 * col[ci] } else { col[ci] };
            }
        }
    }
    let mut xt = vec![0f32; n * c];
    for ci in 0..c {
        for t in 0..n {
            xt[t * c + ci] = x.r(ci)[t];
        }
    }
    let h = run_lstm(pool, &m.text_lstm, &xt);
    for ci in 0..c {
        let r = x.rm(ci);
        for t in 0..n {
            r[t] = h[t * c + ci];
        }
    }
    x
}

/// One chunk of at most 510 phoneme ids (plus the two pads) to 24 kHz audio.
pub fn synthesize(m: &Model, pool: &Pool, ids: &[usize], style: &[f32], speed: f32, noise: &mut Noise) -> Vec<f32> {
    let (timbre, prosody) = (&style[..STYLE], &style[STYLE..]);
    let n = ids.len();
    let b = bert(pool, m, ids);
    let bert_enc = m.bert_encoder.run(pool, &b);
    let d = duration_encoder(pool, m, &bert_enc, prosody);
    let logits = m.duration_proj.run(pool, &run_lstm(pool, &m.lstm, &d));
    let dp = m.duration_proj.out;
    let frames: Vec<usize> = (0..n)
        .map(|t| {
            let mut s = 0f32;
            for j in 0..dp {
                s += sigmoid(logits[t * dp + j]);
            }
            (s / speed).round_ties_even().max(1.0) as usize
        })
        .collect();
    let t_total: usize = frames.iter().sum();
    let src: Vec<usize> = frames.iter().enumerate().flat_map(|(t, &k)| std::iter::repeat_n(t, k)).collect();
    let mut en = vec![0f32; t_total * 640];
    for (j, &s) in src.iter().enumerate() {
        en[j * 640..(j + 1) * 640].copy_from_slice(&d[s * 640..(s + 1) * 640]);
    }
    let sh = run_lstm(pool, &m.shared, &en);
    let mut x = Act::new(512, t_total);
    for c in 0..512 {
        let r = x.rm(c);
        for j in 0..t_total {
            r[j] = sh[j * 512 + c];
        }
    }
    let mut curves: [Vec<f32>; 2] = Default::default();
    for (k, cv) in curves.iter_mut().enumerate() {
        let blocks = if k == 1 { &m.n } else { &m.f0 };
        let mut y = run_res_blk(pool, &blocks[0], prosody, &x);
        y = run_res_blk(pool, &blocks[1], prosody, &y);
        y = run_res_blk(pool, &blocks[2], prosody, &y);
        let mut p = Act::new(1, y.t);
        apply(pool, if k == 1 { &m.n_proj } else { &m.f0_proj }, &y, &mut p);
        *cv = p.r(0).to_vec();
    }
    let [f0, nc] = curves;

    let te = text_encoder(pool, m, ids);
    let mut asr = Act::new(512, t_total);
    for c in 0..512 {
        let (tr, ar) = (te.r(c).to_vec(), asr.rm(c));
        for j in 0..t_total {
            ar[j] = tr[src[j]];
        }
    }
    let f0c = run_strided(pool, &m.f0_conv, &curve(&f0));
    let ncc = run_strided(pool, &m.n_conv, &curve(&nc));
    let mut x = run_res_blk(pool, &m.encode, timbre, &cat(&[&asr, &f0c, &ncc]));
    let mut ar = Act::new(64, t_total);
    apply(pool, &m.asr_res, &asr, &mut ar);
    for i in 0..4 {
        x = run_res_blk(pool, &m.decode[i], timbre, &cat(&[&x, &ar, &f0c, &ncc]));
    }

    let src_wave = source(pool, m, &f0, noise);
    let har = stft(pool, &src_wave);
    for i in 0..2 {
        leaky(pool, &mut x, 0.1);
        x = run_conv_t(pool, &m.ups[i], &x);
        if i == 1 {
            let mut p = Act::new(x.c, x.t + 1);
            for c in 0..x.c {
                let xr = x.r(c);
                let pr = p.rm(c);
                pr[0] = xr[1];
                pr[1..].copy_from_slice(xr);
            }
            x = p;
        }
        let mut ns = if i == 0 { run_strided(pool, &m.noise_conv0, &har) } else { Act::new(x.c, x.t) };
        if i == 1 {
            apply(pool, &m.noise_conv1, &har, &mut ns);
        }
        run_res_block1(pool, &m.noise_res[i], timbre, &mut ns);
        let (xb, xs, xt) = (x.base(), x.stride, x.t);
        let nsr = &ns;
        pool.for_split(x.c, |c| {
            let r = unsafe { xb.slice(c * xs, xt) };
            for (v, n) in r.iter_mut().zip(nsr.r(c)) {
                *v += n;
            }
        });
        let mut sums = [x.copy(), x.copy(), x.copy()];
        for (j, s) in sums.iter_mut().enumerate() {
            run_res_block1(pool, &m.resblocks[3 * i + j], timbre, s);
        }
        let (xb, xs, xt) = (x.base(), x.stride, x.t);
        let sr = &sums;
        pool.for_split(x.c, |c| {
            let r = unsafe { xb.slice(c * xs, xt) };
            let (a, b, d) = (sr[0].r(c), sr[1].r(c), sr[2].r(c));
            for t in 0..xt {
                r[t] = (a[t] + b[t] + d[t]) / 3.0;
            }
        });
    }
    leaky(pool, &mut x, 0.01);
    let mut post = Act::new(2 * BINS, x.t);
    apply(pool, &m.conv_post, &x, &mut post);
    unsafe { istft(&post) }
}
