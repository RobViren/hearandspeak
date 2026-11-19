//! Parakeet TDT 0.6B v3 (ultra weights): subsampler, 24 conformer layers, VAD head, and weight loading.

use super::dense::{self, Half, Packed, linear, taps};
use super::frontend::{self, MELS};
use super::int8::{self, Int8W, QAct, quantize_rows};
use super::{Stage, Times};
use crate::pool::{P, Pool};
use crate::simd::{FlushDenormals, Scratch, ld, set};
use crate::st::{Entry, SafeTensors, bad};
use std::cell::RefCell;
use std::io;
use std::path::Path;
use std::sync::Arc;

pub const LAYERS: usize = 24;
pub const HIDDEN: usize = 1024;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 9;
const SUB_CH: usize = 256;
pub const ENCODED: usize = 640;
pub const VOCAB: usize = 8193;
pub const DURATIONS: usize = 5;
const ATT_BLOCK: usize = 48;
const SEGMENT_FRAMES: usize = 376;

#[rustfmt::skip]
const LINEARS: [&str; 11] = [
    "feed_forward1.linear1", "feed_forward1.linear2", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.relative_k_proj", "self_attn.o_proj", "conv.pointwise_conv1", "conv.pointwise_conv2",
    "feed_forward2.linear1", "feed_forward2.linear2",
];
const FF1_IN: usize = 0;
const FF1_OUT: usize = 1;
const Q: usize = 2;
const K: usize = 3;
const V: usize = 4;
const REL: usize = 5;
const O: usize = 6;
const PW1: usize = 7;
const PW2: usize = 8;
const FF2_IN: usize = 9;
const FF2_OUT: usize = 10;

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    lin: Vec<Int8W>,
    norm_ff1: Norm,
    norm_att: Norm,
    norm_conv: Norm,
    norm_ff2: Norm,
    norm_out: Norm,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    dw: Vec<f32>, // [9][1024], BatchNorm folded in
    dw_bias: Vec<f32>,
}

pub struct Model {
    pub pool: Arc<Pool>,
    pub vnni: bool,
    layers: Vec<Layer>,
    conv0: Vec<f32>,
    conv0_bias: Vec<f32>,
    sub_dw: [Vec<f32>; 2],
    sub_dw_bias: [Vec<f32>; 2],
    sub_pw: [Packed; 2],
    sub_linear: Packed,
    enc_proj: Packed,
    vad_proj: Packed,
    vad_ctx: Packed,
    vad_out: Vec<f32>,
    vad_out_bias: f32,
    pub cell: [Half; 2],
    pub dec_proj: Half,
    pub joint: Half,
    pub embedding: Vec<f32>,
    act: QAct,
    scratch: [Scratch; 7],
    kp: [Packed; HEADS],
    vp: [Packed; HEADS],
    // relative_k_proj of the position table depends only on the offset, never on the audio. Cached per layer and
    // head for rel_t frames; offset d is column rel_t - 1 - d.
    rel: Vec<[Packed; HEADS]>,
    rel_t: usize,
    pub times: Times,
}
unsafe impl Send for Model {}

fn lin_name(layer: usize, i: usize) -> String {
    format!("encoder.layers.{layer}.{}", LINEARS[i])
}

/// The encoder linears as int8, either prepacked (`.q8`, `.init`, `.scale`) or quantized here from f16 `.weight`.
fn load_linear(st: &SafeTensors, name: &str) -> io::Result<Int8W> {
    if st.has(&format!("{name}.q8")) {
        let s = st.shape(&format!("{name}.q8"))?;
        let mut w = Int8W::alloc(s[0], s[1]);
        unsafe {
            st.read_into(
                &format!("{name}.q8"),
                std::slice::from_raw_parts_mut(w.w.as_mut_ptr() as *mut u8, w.w.len()),
            )?;
            st.read_into(
                &format!("{name}.init"),
                std::slice::from_raw_parts_mut(w.init.as_mut_ptr() as *mut u8, w.init.len() * 4),
            )?;
            st.read_into(
                &format!("{name}.scale"),
                std::slice::from_raw_parts_mut(w.scale.as_mut_ptr() as *mut u8, w.scale.len() * 4),
            )?;
        }
        return Ok(w);
    }
    let s = st.shape(&format!("{name}.weight"))?.to_vec();
    Ok(Int8W::quantize(&st.f16(&format!("{name}.weight"))?, s[0], s[1]))
}

impl Model {
    pub fn load(path: &Path, pool: Arc<Pool>) -> io::Result<Model> {
        let st = SafeTensors::open(path)?;
        let mut lin: Vec<Option<io::Result<Int8W>>> = (0..LAYERS * 11).map(|_| None).collect();
        {
            let out = P::new(&mut lin);
            pool.for_each(LAYERS * 11, 1, |i| unsafe {
                *out.at(i) = Some(load_linear(&st, &lin_name(i / 11, i % 11)))
            });
        }
        let mut lin = lin.into_iter().map(|x| x.unwrap());
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("encoder.layers.{l}.");
            let norm = |n: &str| -> io::Result<Norm> {
                Ok(Norm { w: st.f32(&format!("{p}{n}.weight"))?, b: st.f32(&format!("{p}{n}.bias"))? })
            };
            let w = st.f32(&format!("{p}conv.depthwise_conv.weight"))?;
            let gamma = st.f32(&format!("{p}conv.norm.weight"))?;
            let beta = st.f32(&format!("{p}conv.norm.bias"))?;
            let mean = st.f32(&format!("{p}conv.norm.running_mean"))?;
            let var = st.f32(&format!("{p}conv.norm.running_var"))?;
            let mut dw = vec![0f32; CONV_KERNEL * HIDDEN];
            let mut dw_bias = vec![0f32; HIDDEN];
            for c in 0..HIDDEN {
                let scale = gamma[c] / (var[c] + 1e-5).sqrt();
                dw_bias[c] = beta[c] - mean[c] * scale;
                for j in 0..CONV_KERNEL {
                    dw[j * HIDDEN + c] = w[c * CONV_KERNEL + j] * scale;
                }
            }
            layers.push(Layer {
                lin: (0..11).map(|_| lin.next().unwrap()).collect::<io::Result<_>>()?,
                norm_ff1: norm("norm_feed_forward1")?,
                norm_att: norm("norm_self_att")?,
                norm_conv: norm("norm_conv")?,
                norm_ff2: norm("norm_feed_forward2")?,
                norm_out: norm("norm_out")?,
                bias_u: st.f32(&format!("{p}self_attn.bias_u"))?,
                bias_v: st.f32(&format!("{p}self_attn.bias_v"))?,
                dw,
                dw_bias,
            });
        }
        let lin_f = |w: &str, b: &str| -> io::Result<Packed> { Ok(Packed::linear(&st.f32(w)?, &st.f32(b)?)) };
        let taps_of = |name: &str| -> io::Result<Vec<f32>> {
            let w = st.f32(name)?;
            let mut t = vec![0f32; w.len()];
            for c in 0..SUB_CH {
                for k in 0..9 {
                    t[k * SUB_CH + c] = w[c * 9 + k];
                }
            }
            Ok(t)
        };
        let sub = "encoder.subsampling.";
        let sub_linear = {
            // The reference flattens [channel][mel]; activations here are [mel][channel].
            let w = st.f32(&format!("{sub}linear.weight"))?;
            let mut t = vec![0f32; w.len()];
            for o in 0..HIDDEN {
                for c in 0..SUB_CH {
                    for m in 0..16 {
                        t[o * 4096 + m * SUB_CH + c] = w[o * 4096 + c * 16 + m];
                    }
                }
            }
            Packed::linear(&t, &st.f32(&format!("{sub}linear.bias"))?)
        };
        let vad_ctx = {
            let w = st.f32("vad_head.ctx.weight")?;
            let mut t = vec![0f32; w.len()];
            for o in 0..128 {
                for c in 0..128 {
                    for j in 0..5 {
                        t[o * 640 + j * 128 + c] = w[(o * 128 + c) * 5 + j];
                    }
                }
            }
            Packed::linear(&t, &st.f32("vad_head.ctx.bias")?)
        };
        let mut cell: [Half; 2] = Default::default();
        for (i, c) in cell.iter_mut().enumerate() {
            let p = "decoder.lstm.";
            let ih = st.f16(&format!("{p}weight_ih_l{i}"))?;
            let hh = st.f16(&format!("{p}weight_hh_l{i}"))?;
            let mut bias = st.f32(&format!("{p}bias_ih_l{i}"))?;
            let bias_hh = st.f32(&format!("{p}bias_hh_l{i}"))?;
            let mut w = vec![0u16; bias.len() * 2 * ENCODED];
            for o in 0..bias.len() {
                w[o * 2 * ENCODED..o * 2 * ENCODED + ENCODED].copy_from_slice(&ih[o * ENCODED..(o + 1) * ENCODED]);
                w[o * 2 * ENCODED + ENCODED..(o + 1) * 2 * ENCODED]
                    .copy_from_slice(&hh[o * ENCODED..(o + 1) * ENCODED]);
                bias[o] += bias_hh[o];
            }
            *c = Half::pack(2 * ENCODED, bias.len(), &w, &bias);
        }
        let vnni = crate::simd::has_vnni();
        Ok(Model {
            vnni,
            layers,
            conv0: taps_of(&format!("{sub}layers.0.weight"))?,
            conv0_bias: st.f32(&format!("{sub}layers.0.bias"))?,
            sub_dw: [taps_of(&format!("{sub}layers.2.weight"))?, taps_of(&format!("{sub}layers.5.weight"))?],
            sub_dw_bias: [st.f32(&format!("{sub}layers.2.bias"))?, st.f32(&format!("{sub}layers.5.bias"))?],
            sub_pw: [
                lin_f(&format!("{sub}layers.3.weight"), &format!("{sub}layers.3.bias"))?,
                lin_f(&format!("{sub}layers.6.weight"), &format!("{sub}layers.6.bias"))?,
            ],
            sub_linear,
            enc_proj: lin_f("encoder_projector.weight", "encoder_projector.bias")?,
            vad_proj: lin_f("vad_head.proj.weight", "vad_head.proj.bias")?,
            vad_ctx,
            vad_out: st.f32("vad_head.out.weight")?,
            vad_out_bias: st.f32("vad_head.out.bias")?[0],
            cell,
            dec_proj: Half::pack(
                ENCODED,
                ENCODED,
                &st.f16("decoder.decoder_projector.weight")?,
                &st.f32("decoder.decoder_projector.bias")?,
            ),
            joint: Half::pack(ENCODED, VOCAB + DURATIONS, &st.f16("joint.head.weight")?, &st.f32("joint.head.bias")?),
            embedding: st.f32("decoder.embedding.weight")?,
            act: QAct::default(),
            scratch: Default::default(),
            kp: Default::default(),
            vp: Default::default(),
            rel: (0..LAYERS).map(|_| Default::default()).collect(),
            rel_t: 0,
            times: Times::default(),
            pool,
        })
    }

    fn reserve(&mut self, t: usize, c: usize) {
        self.act.reserve(t, c);
    }

    fn prepare(&mut self, x: *const f32, t: usize, c: usize) {
        self.reserve(t, c);
        let _c = self.times.clock(Stage::Quantize);
        int8::quantize(&self.pool, x, &mut self.act);
    }

    fn apply(&self, layer: usize, which: usize, y: *mut f32) {
        let _c = self.times.clock(Stage::Int8Gemm);
        int8::gemm(&self.pool, &self.layers[layer].lin[which], &self.act, y, self.vnni);
    }

    /// Rows past the valid length are zeroed after each stride 2 conv only (reference quirk), so the pointwise bias
    /// refills them and the next depthwise conv reads that row at its right edge. Each chunk of output rows computes
    /// its own small pyramid, keeping the 256 channel maps in L2.
    pub fn subsample(&mut self, mel: &[f32], t0: usize, valid: usize) -> (Vec<f32>, usize, usize) {
        let _c = self.times.clock(Stage::Subsampler);
        let c = SUB_CH;
        let (t1, v1) = (t0.div_ceil(2), valid.div_ceil(2));
        let (t2, v2) = (t1.div_ceil(2), v1.div_ceil(2));
        let (t3, v3) = (t2.div_ceil(2), v2.div_ceil(2));
        let flat = P(self.scratch[6].get(t3 * 16 * c));
        const CHUNK: usize = 8;
        thread_local! {
            static BUFS: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        let m = &*self;
        self.pool.for_each(t3.div_ceil(CHUNK), 1, |chunk| {
            BUFS.with_borrow_mut(|(ring, s2, tmp)| unsafe {
                let a = chunk * CHUNK;
                let n3 = (t3 - a).min(CHUNK);
                let n2 = 2 * n3 + 1;
                let lo2 = 2 * a as isize - 1;
                let lo1 = 2 * lo2 - 1;
                let (w1, w2, w3) = (64 * c, 32 * c, 16 * c);
                ring.resize(3 * w1, 0.0);
                tmp.resize(n2 * w2, 0.0);
                s2.resize(n2 * w2, 0.0);
                let ring_p = P::new(ring);
                let conv0_row = |r: usize| {
                    let out = ring_p.at(r % 3 * w1);
                    let row = lo1 + r as isize;
                    if row < 0 || row >= t1.min(v1) as isize {
                        std::ptr::write_bytes(out, 0, w1);
                        return;
                    }
                    let row = row as usize;
                    for w in 0..64 {
                        let mut x = [0f32; 9];
                        let mut t: [*const f32; 9] = [std::ptr::null(); 9];
                        let mut n = 0;
                        for ky in 0..3 {
                            let y = (2 * row + ky) as isize - 1;
                            if y < 0 || y >= t0 as isize {
                                continue;
                            }
                            for kx in if w > 0 { 0 } else { 1 }..3 {
                                x[n] = mel[y as usize * MELS + 2 * w + kx - 1];
                                t[n] = m.conv0.as_ptr().add((ky * 3 + kx) * c);
                                n += 1;
                            }
                        }
                        conv_taps(m.conv0_bias.as_ptr(), &t[..n], |j, _| set(x[j]), out.add(w * c), c, true);
                    }
                };
                for i in 0..n2 {
                    for r in if i > 0 { 2 * i + 1 } else { 0 }..=2 * i + 2 {
                        conv0_row(r);
                    }
                    let rows =
                        [ring_p.at(2 * i % 3 * w1), ring_p.at((2 * i + 1) % 3 * w1), ring_p.at((2 * i + 2) % 3 * w1)];
                    let live = lo2 + i as isize >= 0 && lo2 + (i as isize) < v2 as isize;
                    depthwise_row(&rows, 32, live, &m.sub_dw[0], &m.sub_dw_bias[0], tmp.as_mut_ptr().add(i * w2));
                }
                pointwise(&m.sub_pw[0], tmp.as_ptr(), 32, lo2, n2, t2, s2.as_mut_ptr());
                for i in 0..n3 {
                    let rows = [
                        s2.as_mut_ptr().add(2 * i * w2),
                        s2.as_mut_ptr().add((2 * i + 1) * w2),
                        s2.as_mut_ptr().add((2 * i + 2) * w2),
                    ];
                    depthwise_row(&rows, 16, a + i < v3, &m.sub_dw[1], &m.sub_dw_bias[1], tmp.as_mut_ptr().add(i * w3));
                }
                pointwise(&m.sub_pw[1], tmp.as_ptr(), 16, a as isize, n3, t3, flat.at(a * w3));
            });
        });
        let mut out = vec![0f32; t3 * HIDDEN];
        linear(&self.pool, &self.sub_linear, flat.ptr(), t3, out.as_mut_ptr());
        (out, t3, v3)
    }

    pub fn vad(&mut self, hidden: &[f32], t: usize) -> Vec<f32> {
        let _c = self.times.clock(Stage::Vad);
        let mut a = vec![0f32; t * 128];
        let mut window = vec![0f32; t * 640];
        let mut b = vec![0f32; t * 128];
        // Activations here reach thousands, where the sigmoid lands in denormals and every vector takes a microcode
        // assist. Flushed to zero they give the exact sigmoid's result.
        let silu = |v: &mut [f32]| {
            let _f = FlushDenormals::new();
            unsafe { dense::silu(v.as_mut_ptr(), v.len()) };
        };
        linear(&self.pool, &self.vad_proj, hidden.as_ptr(), t, a.as_mut_ptr());
        silu(&mut a);
        for i in 0..t {
            for j in 0..5 {
                let s = i as isize + j as isize - 2;
                if s >= 0 && (s as usize) < t {
                    window[(i * 5 + j) * 128..(i * 5 + j + 1) * 128]
                        .copy_from_slice(&a[s as usize * 128..(s as usize + 1) * 128]);
                }
            }
        }
        linear(&self.pool, &self.vad_ctx, window.as_ptr(), t, b.as_mut_ptr());
        silu(&mut b);
        (0..t)
            .map(|i| {
                let mut s = self.vad_out_bias;
                for c in 0..128 {
                    s += b[i * 128 + c] * self.vad_out[c];
                }
                1.0 / (1.0 + (-s).exp())
            })
            .collect()
    }

    /// Sized to the first segment, then once more to the 30 s cap, so the table is projected at most twice.
    fn positions(&mut self, t: usize) {
        if t <= self.rel_t {
            return;
        }
        self.rel_t = if self.rel_t > 0 { t.max(SEGMENT_FRAMES) } else { t };
        let rt = self.rel_t;
        let r = 2 * rt - 1;
        let h = HIDDEN;
        let mut positions = vec![0f32; r * h];
        let mut proj = vec![0f32; r * h];
        let inverse: Vec<f32> = (0..h / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / h as f32)).collect();
        {
            let pp = P::new(&mut positions);
            self.pool.for_each(r, 0, |j| {
                for i in 0..h / 2 {
                    let phase = (rt as f32 - 1.0 - j as f32) * inverse[i];
                    unsafe {
                        *pp.at(j * h + 2 * i) = phase.sin();
                        *pp.at(j * h + 2 * i + 1) = phase.cos();
                    }
                }
            });
        }
        let mut a = QAct::default();
        a.reserve(r, h);
        int8::quantize(&self.pool, positions.as_ptr(), &mut a);
        for layer in 0..LAYERS {
            int8::gemm(&self.pool, &self.layers[layer].lin[REL], &a, proj.as_mut_ptr(), self.vnni);
            let rel = P(self.rel[layer].as_mut_ptr());
            let pr = P(proj.as_mut_ptr());
            self.pool.for_each(HEADS, 0, |hd| unsafe {
                (*rel.at(hd)).pack(HEAD_DIM, r, pr.at(hd * HEAD_DIM), 1, h, None);
            });
        }
    }

    /// Query i reads its T position scores at columns rel_t - 1 - i onward, so a block of query rows needs only the
    /// union window of T + rows - 1 columns. One work item is (head, row block).
    fn attention_mix(&mut self, layer: usize, t: usize, q: *const f32, k: *const f32, v: *const f32, out: *mut f32) {
        let _c = self.times.clock(Stage::Attention);
        let d = HEAD_DIM;
        let tp = (t + 15) & !15;
        let wp = tp + ATT_BLOCK + 32;
        let blocks = t.div_ceil(ATT_BLOCK);
        let (kp, vp) = (P(self.kp.as_mut_ptr()), P(self.vp.as_mut_ptr()));
        let (k, v) = (P(k as *mut f32), P(v as *mut f32));
        self.pool.for_each(2 * HEADS, 1, |i| unsafe {
            let h = i / 2;
            if i % 2 == 0 {
                (*kp.at(h)).pack(d, t, k.at(h * d), 1, HIDDEN, None);
            } else {
                (*vp.at(h)).pack(t, d, v.at(h * d), HIDDEN, 1, None);
            }
        });
        let scale = 1.0 / (d as f32).sqrt();
        let lay = &self.layers[layer];
        let rp = &self.rel[layer];
        let rel_t = self.rel_t;
        let (kp, vp) = (&self.kp, &self.vp);
        let (q, out) = (P(q as *mut f32), P(out));
        thread_local! {
            static BUFS: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
        }
        self.pool.for_each(HEADS * blocks, 1, |item| {
            BUFS.with_borrow_mut(|(query, content, position)| unsafe {
                query.resize(ATT_BLOCK * d, 0.0);
                content.resize(ATT_BLOCK * tp, 0.0);
                position.resize(ATT_BLOCK * wp, 0.0);
                let h = item / blocks;
                let i0 = item % blocks * ATT_BLOCK;
                let m = ATT_BLOCK.min(t - i0);
                let biased = |query: &mut [f32], bias: &[f32]| {
                    for i in 0..m {
                        for dd in 0..d {
                            query[i * d + dd] = *q.at((i0 + i) * HIDDEN + h * d + dd) + bias[h * d + dd];
                        }
                    }
                };
                // Padding and scores far below the row peak become denormal probabilities, each a microcode assist.
                let _f = FlushDenormals::new();
                biased(query, &lay.bias_u);
                dense::gemm(&kp[h], query.as_ptr(), d, m, content.as_mut_ptr(), tp, 0, kp[h].panels());
                biased(query, &lay.bias_v);
                let first = (rel_t - 1 - i0) as isize;
                let p0 = (first - m as isize + 1) / 16;
                let p1 = (first + t as isize - 1) / 16 + 1;
                let pos = position.as_mut_ptr();
                for p in p0..p1 {
                    // Rows whose window [first - i, first - i + T) meets this panel, widened to whole 6 row groups.
                    let r0 = (first - 16 * p - 15).max(0) / 6 * 6;
                    let r1 = (m as isize).min((first + t as isize - 16 * p + 5) / 6 * 6);
                    if r1 > r0 {
                        let c = pos.offset(r0 * wp as isize - p0 * 16);
                        dense::gemm(
                            &rp[h],
                            query.as_ptr().add(r0 as usize * d),
                            d,
                            (r1 - r0) as usize,
                            c,
                            wp,
                            p as usize,
                            p as usize + 1,
                        );
                    }
                }
                for i in 0..m {
                    let s = content.as_mut_ptr().add(i * tp);
                    let pp = pos.offset((i * wp) as isize + (rel_t - 1 - (i0 + i)) as isize - p0 * 16);
                    for j in 0..t {
                        *s.add(j) = (*s.add(j) + *pp.add(j)) * scale;
                    }
                    for j in t..tp {
                        *s.add(j) = f32::NEG_INFINITY;
                    }
                    dense::softmax(s, tp);
                }
                dense::gemm(&vp[h], content.as_ptr(), tp, m, out.at(i0 * HIDDEN + h * d), HIDDEN, 0, d / 16);
            });
        });
    }

    pub fn encode(&mut self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let (mel, frames, valid) = {
            let _c = self.times.clock(Stage::Frontend);
            frontend::log_mel(&self.pool, pcm)
        };
        let (mut x, _, t) = self.subsample(&mel, frames, valid);
        x.truncate(t * HIDDEN);
        let h = HIDDEN;
        let wide = P(self.scratch[0].get(t * 4 * h));
        let n = P(self.scratch[1].get(t * h));
        let y = P(self.scratch[2].get(t * h));
        let q = P(self.scratch[3].get(t * h));
        let k = P(self.scratch[4].get(t * h));
        let v = P(self.scratch[5].get(t * h));
        self.positions(t);
        let xp = P::new(&mut x);

        for layer in 0..LAYERS {
            self.feed_forward(layer, t, 0.0, FF1_IN, FF1_OUT, xp, y, n, wide, true);
            self.norm(layer, t, 0.5, 1, xp, y, n, true);
            self.apply(layer, Q, q.ptr());
            self.apply(layer, K, k.ptr());
            self.apply(layer, V, v.ptr());
            self.attention_mix(layer, t, q.ptr(), k.ptr(), v.ptr(), n.ptr());
            self.prepare(n.ptr(), t, h);
            self.apply(layer, O, y.ptr());

            self.norm(layer, t, 1.0, 2, xp, y, n, true);
            self.apply(layer, PW1, wide.ptr());
            {
                let _c = self.times.clock(Stage::Norms);
                self.pool.for_each(t, 0, |i| unsafe {
                    dense::glu(wide.at(i * 2 * h), wide.at(i * 2 * h + h), q.at(i * h), h)
                });
            }
            {
                let _c = self.times.clock(Stage::Depthwise);
                self.reserve(t, h);
                let lay = &self.layers[layer];
                let (qa, sx) = (P(self.act.q.as_mut_ptr()), P(self.act.sx.as_mut_ptr()));
                self.pool.for_each(t, 0, |i| unsafe {
                    let j0 = 4usize.saturating_sub(i);
                    let ntaps = CONV_KERNEL.min(t + 4 - i) - j0;
                    let mut w: [*const f32; CONV_KERNEL] = [std::ptr::null(); CONV_KERNEL];
                    for (j, wj) in w.iter_mut().take(ntaps).enumerate() {
                        *wj = lay.dw.as_ptr().add((j0 + j) * h);
                    }
                    let inp = q.at((i + j0 - 4) * h);
                    let o = n.at(i * h);
                    conv_taps(lay.dw_bias.as_ptr(), &w[..ntaps], |j, c| ld(inp.add(j * h + c)), o, h, false);
                    dense::silu(o, h);
                    quantize_rows(n.ptr(), h, qa.ptr(), sx.ptr(), i, i + 1);
                });
            }
            self.apply(layer, PW2, y.ptr());

            self.feed_forward(layer, t, 1.0, FF2_IN, FF2_OUT, xp, y, n, wide, false);
            self.norm(layer, t, 0.5, 4, xp, y, xp, false);
        }
        let mut encoded = vec![0f32; t * ENCODED];
        linear(&self.pool, &self.enc_proj, x.as_ptr(), t, encoded.as_mut_ptr());
        (encoded, t)
    }

    /// x += scale * y per row, then LayerNorm into out, writing int8 codes of out while the row is in L1.
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &mut self,
        layer: usize,
        t: usize,
        scale: f32,
        which: usize,
        x: P<f32>,
        y: P<f32>,
        out: P<f32>,
        quantize: bool,
    ) {
        let _c = self.times.clock(Stage::Norms);
        if quantize {
            self.reserve(t, HIDDEN);
        }
        let lay = &self.layers[layer];
        let ln = [&lay.norm_ff1, &lay.norm_att, &lay.norm_conv, &lay.norm_ff2, &lay.norm_out][which];
        let (qa, sx) = (P(self.act.q.as_mut_ptr()), P(self.act.sx.as_mut_ptr()));
        self.pool.for_each(t, 0, |i| unsafe {
            let row = x.at(i * HIDDEN);
            if scale != 0.0 {
                dense::residual(row, y.at(i * HIDDEN), scale, HIDDEN);
            }
            dense::layer_norm(row, ln.w.as_ptr(), ln.b.as_ptr(), out.at(i * HIDDEN), HIDDEN);
            if quantize {
                quantize_rows(out.ptr(), HIDDEN, qa.ptr(), sx.ptr(), i, i + 1);
            }
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn feed_forward(
        &mut self,
        layer: usize,
        t: usize,
        scale: f32,
        lin_in: usize,
        lin_out: usize,
        x: P<f32>,
        y: P<f32>,
        n: P<f32>,
        wide: P<f32>,
        first: bool,
    ) {
        self.norm(layer, t, scale, if first { 0 } else { 3 }, x, y, n, true);
        self.apply(layer, lin_in, wide.ptr());
        {
            let _c = self.times.clock(Stage::Norms);
            self.reserve(t, 4 * HIDDEN);
            let (qa, sx) = (P(self.act.q.as_mut_ptr()), P(self.act.sx.as_mut_ptr()));
            let h4 = 4 * HIDDEN;
            self.pool.for_each(t, 0, |i| unsafe {
                dense::silu(wide.at(i * h4), h4);
                quantize_rows(wide.ptr(), h4, qa.ptr(), sx.ptr(), i, i + 1);
            });
        }
        self.apply(layer, lin_out, y.ptr());
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn conv_taps<X: Fn(usize, usize) -> crate::simd::F8>(
    bias: *const f32,
    w: &[*const f32],
    x: X,
    y: *mut f32,
    ch: usize,
    relu: bool,
) {
    unsafe { taps(bias, w, x, y, ch, relu) }
}

/// One output row [W][channel] of a stride 2 depthwise 3x3 from its three input rows [2W][channel].
unsafe fn depthwise_row(rows: &[*mut f32; 3], wd: usize, live: bool, dw: &[f32], dw_bias: &[f32], out: *mut f32) {
    unsafe {
        let c = SUB_CH;
        if !live {
            std::ptr::write_bytes(out, 0, wd * c);
            return;
        }
        for w in 0..wd {
            let mut x: [*const f32; 9] = [std::ptr::null(); 9];
            let mut t: [*const f32; 9] = [std::ptr::null(); 9];
            let mut n = 0;
            for (ky, row) in rows.iter().enumerate() {
                for kx in if w > 0 { 0 } else { 1 }..3 {
                    x[n] = row.add((2 * w + kx - 1) * c);
                    t[n] = dw.as_ptr().add((ky * 3 + kx) * c);
                    n += 1;
                }
            }
            conv_taps(dw_bias.as_ptr(), &t[..n], |j, ch| ld(x[j].add(ch)), out.add(w * c), c, false);
        }
    }
}

/// Pointwise conv and ReLU over n rows [W][channel] starting at global row lo. Rows outside [0, total) are padding.
unsafe fn pointwise(pw: &Packed, inp: *const f32, wd: usize, lo: isize, n: usize, total: usize, out: *mut f32) {
    unsafe {
        let c = SUB_CH;
        dense::gemm(pw, inp, c, n * wd, out, c, 0, pw.panels());
        for i in 0..n {
            let o = std::slice::from_raw_parts_mut(out.add(i * wd * c), wd * c);
            let row = lo + i as isize;
            let padding = row < 0 || row >= total as isize;
            for v in o.iter_mut() {
                *v = if padding { 0.0 } else { v.max(0.0) };
            }
        }
    }
}

/// Writes the model with its encoder linears prepacked as int8: half the size of the f16 file, no load-time
/// quantization, identical results.
#[doc(hidden)]
pub fn convert(src: &Path, dst: &Path) -> io::Result<()> {
    let st = SafeTensors::open(src)?;
    let mut names: Vec<&String> = st.t.keys().collect();
    names.sort();
    let mut packed = Vec::new();
    for l in 0..LAYERS {
        for i in 0..11 {
            let name = lin_name(l, i);
            let w = load_linear(&st, &name)?;
            packed.push((name, w));
        }
    }
    let lin_weights: std::collections::HashSet<String> = packed.iter().map(|(n, _)| format!("{n}.weight")).collect();
    let mut raw = Vec::new();
    for n in names {
        if !lin_weights.contains(n) && !n.ends_with("num_batches_tracked") {
            raw.push((n.clone(), st.get(n)?.dtype.clone(), st.shape(n)?.to_vec(), st.bytes(n)?));
        }
    }
    let as_bytes = |p: *const u8, n: usize| unsafe { std::slice::from_raw_parts(p, n) };
    let mut entries = Vec::new();
    for (n, w) in &packed {
        entries.push(Entry {
            name: format!("{n}.q8"),
            dtype: "I8",
            shape: vec![w.out, w.inp],
            data: as_bytes(w.w.as_ptr() as *const u8, w.w.len()),
        });
        entries.push(Entry {
            name: format!("{n}.init"),
            dtype: "I32",
            shape: vec![w.init.len()],
            data: as_bytes(w.init.as_ptr() as *const u8, w.init.len() * 4),
        });
        entries.push(Entry {
            name: format!("{n}.scale"),
            dtype: "F32",
            shape: vec![w.scale.len()],
            data: as_bytes(w.scale.as_ptr() as *const u8, w.scale.len() * 4),
        });
    }
    for (n, d, s, b) in &raw {
        let dtype = match d.as_str() {
            "F16" => "F16",
            "F32" => "F32",
            other => return Err(bad(format!("{n}: unexpected dtype {other}"))),
        };
        entries.push(Entry { name: n.clone(), dtype, shape: s.clone(), data: b });
    }
    crate::st::write(dst, &entries)
}
