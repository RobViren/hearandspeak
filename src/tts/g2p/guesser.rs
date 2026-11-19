//! Unknown-word guesser: char-to-phone pre-LN transformer, f16 weights, greedy. Vowel tokens carry their stress;
//! `guess` moves it back to the syllable onset by maximal onset, giving the lexicon's dialect.

use crate::simd::{F8, add, fma, ldh, max, set, zero};
use std::collections::{BTreeSet, HashMap};
use std::io;

struct Dense {
    w: Vec<u16>, // f16, [out/16][in][16]
    b: Vec<f32>,
    inp: usize,
    out: usize,
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    n1: Norm,
    n2: Option<Norm>,
    n3: Norm,
    qkv: Dense,
    o: Dense,
    cross: Option<(Dense, Dense, Dense)>, // q, kv, o
    f1: Dense,
    f2: Dense,
}

struct Phone {
    kind: u8,
    stress: u8,
    ipa: String,
}

pub struct Guesser {
    d: usize,
    heads: usize,
    ff: usize,
    max_src: usize,
    max_tgt: usize,
    tag: i32,
    char_id: [i16; 256],
    tgt: Vec<Phone>,
    onsets: BTreeSet<Vec<usize>>,
    src_emb: Vec<f32>,
    src_pos: Vec<f32>,
    tgt_emb: Vec<f32>,
    tgt_pos: Vec<f32>,
    enc: Vec<Layer>,
    dec: Vec<Layer>,
    enc_norm: Norm,
    dec_norm: Norm,
    out: Dense,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let s = self.b.get(self.p..self.p + n).ok_or_else(|| crate::st::bad("guesser.bin truncated"))?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> io::Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn str(&mut self, n: usize) -> io::Result<String> {
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

fn pack(out: usize, inp: usize, w: &[u16], b: &[f32]) -> Dense {
    let cols = out.div_ceil(16) * 16;
    let mut p = vec![0u16; cols * inp];
    let mut bias = vec![0f32; cols];
    bias[..out].copy_from_slice(&b[..out]);
    for o in 0..out {
        for i in 0..inp {
            p[(o / 16 * inp + i) * 16 + o % 16] = w[o * inp + i];
        }
    }
    Dense { w: p, b: bias, inp, out }
}

#[inline(always)]
unsafe fn store(v: F8, bias: *const f32, res: *const f32, relu: bool, y: *mut f32, n: usize) {
    unsafe {
        let mut v = add(v, crate::simd::ld(bias));
        let mut t = [0f32; 8];
        if !res.is_null() {
            std::ptr::copy_nonoverlapping(res, t.as_mut_ptr(), n);
            v = add(v, crate::simd::ld(t.as_ptr()));
        }
        if relu {
            v = max(v, zero());
        }
        crate::simd::st(t.as_mut_ptr(), v);
        std::ptr::copy_nonoverlapping(t.as_ptr(), y, n);
    }
}

/// R rows of x against one 16-wide panel. One row splits k into four phases to keep eight FMA chains.
#[inline(always)]
unsafe fn tile<const R: usize>(
    w: *const u16,
    bias: *const f32,
    x: *const f32,
    inp: usize,
    out: usize,
    col: usize,
    y: *mut f32,
    res: *const f32,
    relu: bool,
) {
    unsafe {
        let p = if R == 1 { 4 } else { 1 };
        let mut a = [[zero(); 2]; 4];
        let mut k = 0;
        let mut wp = w;
        while k + p <= inp {
            for j in 0..R * p {
                let (r, kk) = if R == 1 { (0, j) } else { (j, 0) };
                let xv = set(*x.add(r * inp + k + kk));
                a[j][0] = fma(xv, ldh(wp.add(kk * 16)), a[j][0]);
                a[j][1] = fma(xv, ldh(wp.add(kk * 16 + 8)), a[j][1]);
            }
            k += p;
            wp = wp.add(p * 16);
        }
        while k < inp {
            let xv = set(*x.add(k));
            a[0][0] = fma(xv, ldh(wp), a[0][0]);
            a[0][1] = fma(xv, ldh(wp.add(8)), a[0][1]);
            k += 1;
            wp = wp.add(16);
        }
        if R == 1 {
            for h in 0..2 {
                a[0][h] = add(add(a[0][h], a[1][h]), add(a[2][h], a[3][h]));
            }
        }
        for r in 0..R {
            for h in 0..2 {
                let c = col + h * 8;
                let n = out.saturating_sub(c).min(8);
                let rp = if res.is_null() { res } else { res.add(r * out + c) };
                store(a[r][h], bias.add(c), rp, relu, y.add(r * out + c), n);
            }
        }
    }
}

/// y[n][out] = x[n][in] W + b (+ res) (relu).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dense16(g: &Dense, x: *const f32, n: usize, y: *mut f32, res: *const f32, relu: bool) {
    unsafe {
        let mut w = g.w.as_ptr();
        for col in (0..g.out).step_by(16) {
            let mut i = 0;
            while i < n {
                let (o, oy) = (i * g.inp, i * g.out);
                let ri = if res.is_null() { res } else { res.add(oy) };
                let (b, xi, yi) = (g.b.as_ptr(), x.add(o), y.add(oy));
                match 4.min(n - i) {
                    4 => tile::<4>(w, b, xi, g.inp, g.out, col, yi, ri, relu),
                    3 => tile::<3>(w, b, xi, g.inp, g.out, col, yi, ri, relu),
                    2 => tile::<2>(w, b, xi, g.inp, g.out, col, yi, ri, relu),
                    _ => tile::<1>(w, b, xi, g.inp, g.out, col, yi, ri, relu),
                }
                i += 4.min(n - i);
            }
            w = w.add(g.inp * 16);
        }
    }
}

fn lin(g: &Dense, x: &[f32], n: usize, y: &mut [f32], res: Option<&[f32]>, relu: bool) {
    unsafe { dense16(g, x.as_ptr(), n, y.as_mut_ptr(), res.map_or(std::ptr::null(), |r| r.as_ptr()), relu) }
}

#[target_feature(enable = "avx2,fma")]
fn layer_norm(n: &Norm, x: &[f32], y: &mut [f32], d: usize) {
    let mut m = 0f32;
    for &v in &x[..d] {
        m += v;
    }
    m /= d as f32;
    let mut v = 0f32;
    for &xv in &x[..d] {
        v = (xv - m).mul_add(xv - m, v);
    }
    let inv = 1.0 / (v / d as f32 + 1e-5).sqrt();
    for i in 0..d {
        y[i] = ((x[i] - m) * inv).mul_add(n.g[i], n.b[i]);
    }
}

/// One query row against n keys/values (row strides ldk, ldv), all heads.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn attend(
    q: &[f32],
    k: &[f32],
    ldk: usize,
    v: &[f32],
    ldv: usize,
    n: usize,
    d: usize,
    heads: usize,
    ctx: &mut [f32],
    s: &mut [f32],
) {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..heads {
        let qh = &q[h * hd..(h + 1) * hd];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..n {
            let kj = &k[j * ldk + h * hd..];
            let mut a = 0f32;
            for c in 0..hd {
                a = qh[c].mul_add(kj[c], a);
            }
            s[j] = a * scale;
            mx = mx.max(s[j]);
        }
        let mut sum = 0f32;
        for sj in s[..n].iter_mut() {
            *sj = (*sj - mx).exp();
            sum += *sj;
        }
        let o = &mut ctx[h * hd..(h + 1) * hd];
        o.fill(0.0);
        for j in 0..n {
            let (pj, vj) = (s[j] / sum, &v[j * ldv + h * hd..]);
            for c in 0..hd {
                o[c] = pj.mul_add(vj[c], o[c]);
            }
        }
    }
}

impl Guesser {
    pub fn load(data: &[u8]) -> io::Result<Guesser> {
        let mut r = Reader { b: data, p: 0 };
        if r.take(8)? != b"KKG2P001" {
            return Err(crate::st::bad("guesser.bin: bad magic"));
        }
        let h: Vec<i32> = (0..8).map(|_| r.i32()).collect::<io::Result<_>>()?;
        let (d, heads, ne, nd, ff) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3] as usize, h[4] as usize);
        let mut char_id = [-1i16; 256];
        for i in 0..r.i32()? {
            let n = r.u8()? as usize;
            let c = r.take(n)?;
            if c.len() == 1 {
                char_id[c[0] as usize] = i as i16;
            }
        }
        let mut tgt = Vec::new();
        for _ in 0..r.i32()? {
            let (kind, stress, n) = (r.u8()?, r.u8()?, r.u8()? as usize);
            tgt.push(Phone { kind, stress, ipa: r.str(n)? });
        }
        let mut onsets = BTreeSet::new();
        for _ in 0..r.i32()? {
            let n = r.u8()? as usize;
            onsets.insert((0..n).map(|_| r.u8().map(|v| v as usize)).collect::<io::Result<Vec<_>>>()?);
        }
        let mut t: HashMap<String, (Vec<u16>, usize)> = HashMap::new();
        for _ in 0..r.i32()? {
            let n = r.u8()? as usize;
            let name = r.str(n)?;
            let nd = r.u8()? as usize;
            let mut count = 1;
            for _ in 0..nd {
                count *= r.i32()? as usize;
            }
            let v = r.take(count * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            t.insert(name, (v, count));
        }
        let get = |n: &str| t.get(n).map(|x| &x.0).ok_or_else(|| crate::st::bad(format!("guesser: missing {n}")));
        let f32s = |n: &str| -> io::Result<Vec<f32>> { Ok(get(n)?.iter().map(|&h| crate::simd::f16(h)).collect()) };
        let lin_of = |p: &str| -> io::Result<Dense> {
            let (w, b) = (get(&format!("{p}.weight"))?, f32s(&format!("{p}.bias"))?);
            Ok(pack(b.len(), w.len() / b.len(), w, &b))
        };
        let cat = |ps: &[String]| -> io::Result<Dense> {
            let (mut w, mut b) = (Vec::new(), Vec::new());
            for p in ps {
                w.extend_from_slice(get(&format!("{p}.weight"))?);
                b.extend(f32s(&format!("{p}.bias"))?);
            }
            Ok(pack(b.len(), d, &w, &b))
        };
        let norm = |p: &str| -> io::Result<Norm> {
            Ok(Norm { g: f32s(&format!("{p}.weight"))?, b: f32s(&format!("{p}.bias"))? })
        };
        let scale = (d as f32).sqrt();
        let mut src_emb = f32s("src_emb.weight")?;
        let mut tgt_emb = f32s("tgt_emb.weight")?;
        src_emb.iter_mut().for_each(|x| *x *= scale);
        tgt_emb.iter_mut().for_each(|x| *x *= scale);
        let (mut enc, mut dec) = (Vec::new(), Vec::new());
        for i in 0..ne + nd {
            let is_dec = i >= ne;
            let p = if is_dec { format!("dec.{}", i - ne) } else { format!("enc.{i}") };
            let s = |x: &str| format!("{p}.{x}");
            let layer = Layer {
                n1: norm(&s("n1"))?,
                n3: norm(&s("n3"))?,
                qkv: cat(&[s("sa.q"), s("sa.k"), s("sa.v")])?,
                o: lin_of(&s("sa.o"))?,
                f1: lin_of(&s("f1"))?,
                f2: lin_of(&s("f2"))?,
                n2: if is_dec { Some(norm(&s("n2"))?) } else { None },
                cross: if is_dec {
                    Some((lin_of(&s("ca.q"))?, cat(&[s("ca.k"), s("ca.v")])?, lin_of(&s("ca.o"))?))
                } else {
                    None
                },
            };
            if is_dec { dec.push(layer) } else { enc.push(layer) }
        }
        Ok(Guesser {
            d,
            heads,
            ff,
            max_src: h[5] as usize,
            max_tgt: h[6] as usize,
            tag: h[7],
            char_id,
            tgt,
            onsets,
            src_pos: f32s("src_pos.weight")?,
            tgt_pos: f32s("tgt_pos.weight")?,
            src_emb,
            tgt_emb,
            enc_norm: norm("enc_norm")?,
            dec_norm: norm("dec_norm")?,
            out: lin_of("out")?,
            enc,
            dec,
        })
    }

    /// Lexicon-dialect IPA for a word, empty if there is no answer.
    pub fn guess(&self, word: &str) -> String {
        let mut src: Vec<usize> = Vec::new();
        if self.tag >= 0 {
            src.push(self.tag as usize);
        }
        let b = word.as_bytes();
        let mut i = 0;
        while i < b.len() && src.len() < self.max_src {
            let mut c = b[i];
            if c == 0xE2 && i + 2 < b.len() && b[i + 1] == 0x80 && b[i + 2] == 0x99 {
                c = b'\'';
                i += 2;
            }
            c = c.to_ascii_lowercase();
            if self.char_id[c as usize] >= 0 {
                src.push(self.char_id[c as usize] as usize);
            }
            i += 1;
        }
        let (s_n, d, nd, mt) = (src.len(), self.d, self.dec.len(), self.max_tgt);
        if s_n == (self.tag >= 0) as usize {
            return String::new();
        }
        let m = s_n.max(mt + 1);
        let (mut x, mut y, mut h) = (vec![0f32; m * d], vec![0f32; m * d], vec![0f32; m * d]);
        let (mut qkv, mut ctx, mut f) = (vec![0f32; 3 * m * d], vec![0f32; m * d], vec![0f32; m * self.ff]);
        let mut cross = vec![0f32; nd * 2 * s_n * d];
        let mut cache = vec![0f32; nd * 2 * (mt + 1) * d];
        let mut s = vec![0f32; m];
        for i in 0..s_n {
            for c in 0..d {
                x[i * d + c] = self.src_emb[src[i] * d + c] + self.src_pos[i * d + c];
            }
        }
        unsafe {
            for l in &self.enc {
                for i in 0..s_n {
                    layer_norm(&l.n1, &x[i * d..], &mut h[i * d..], d);
                }
                lin(&l.qkv, &h, s_n, &mut qkv, None, false);
                for i in 0..s_n {
                    attend(
                        &qkv[i * 3 * d..],
                        &qkv[d..],
                        3 * d,
                        &qkv[2 * d..],
                        3 * d,
                        s_n,
                        d,
                        self.heads,
                        &mut ctx[i * d..],
                        &mut s,
                    );
                }
                lin(&l.o, &ctx, s_n, &mut y, Some(&x), false);
                for i in 0..s_n {
                    layer_norm(&l.n3, &y[i * d..], &mut h[i * d..], d);
                }
                lin(&l.f1, &h, s_n, &mut f, None, true);
                lin(&l.f2, &f, s_n, &mut x, Some(&y), false);
            }
            for i in 0..s_n {
                layer_norm(&self.enc_norm, &x[i * d..], &mut h[i * d..], d);
            }
            for (l, layer) in self.dec.iter().enumerate() {
                lin(&layer.cross.as_ref().unwrap().1, &h, s_n, &mut cross[l * 2 * s_n * d..], None, false);
            }
            let mut toks = Vec::new();
            let mut tok = 1;
            for t in 0..mt {
                for c in 0..d {
                    x[c] = self.tgt_emb[tok * d + c] + self.tgt_pos[t * d + c];
                }
                for (l, layer) in self.dec.iter().enumerate() {
                    let (cq, ckv, co) = layer.cross.as_ref().unwrap();
                    let _ = ckv;
                    let kv = &mut cache[l * 2 * (mt + 1) * d..(l + 1) * 2 * (mt + 1) * d];
                    layer_norm(&layer.n1, &x, &mut h, d);
                    lin(&layer.qkv, &h, 1, &mut qkv, None, false);
                    kv[t * 2 * d..(t + 1) * 2 * d].copy_from_slice(&qkv[d..3 * d]);
                    attend(&qkv, kv, 2 * d, &kv[d..], 2 * d, t + 1, d, self.heads, &mut ctx, &mut s);
                    lin(&layer.o, &ctx, 1, &mut y, Some(&x), false);
                    layer_norm(layer.n2.as_ref().unwrap(), &y, &mut h, d);
                    lin(cq, &h, 1, &mut qkv, None, false);
                    let kc = &cross[l * 2 * s_n * d..(l + 1) * 2 * s_n * d];
                    attend(&qkv, kc, 2 * d, &kc[d..], 2 * d, s_n, d, self.heads, &mut ctx, &mut s);
                    lin(co, &ctx, 1, &mut x, Some(&y), false);
                    layer_norm(&layer.n3, &x, &mut h, d);
                    lin(&layer.f1, &h, 1, &mut f, None, true);
                    lin(&layer.f2, &f, 1, &mut y, Some(&x), false);
                    std::mem::swap(&mut x, &mut y);
                }
                layer_norm(&self.dec_norm, &x, &mut h, d);
                lin(&self.out, &h, 1, &mut y, None, false);
                let logits = &y[..self.out.out];
                tok = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
                if self.tgt[tok].kind == 0 {
                    break;
                }
                toks.push(tok);
            }
            // Phone ids, with usize::MAX - 1 primary and usize::MAX - 2 secondary stress marks.
            let mut ph: Vec<usize> = Vec::new();
            let mark = |st: u8| usize::MAX - st as usize;
            for k in toks {
                let st = self.tgt[k].stress;
                if st != 0 {
                    let is_mark = |p: usize| p >= usize::MAX - 2;
                    let mut j = ph.len();
                    while j > 0 && !is_mark(ph[j - 1]) && self.tgt[ph[j - 1]].kind == 1 {
                        j -= 1;
                    }
                    let mut on = j;
                    if j > 0 {
                        on = ph.len();
                        for a in j..ph.len() {
                            if self.onsets.contains(&ph[a..]) {
                                on = a;
                                break;
                            }
                        }
                    }
                    ph.insert(on, mark(st));
                }
                ph.push(k);
            }
            ph.iter()
                .map(|&p| {
                    if p == usize::MAX - 1 {
                        "ˈ"
                    } else if p == usize::MAX - 2 {
                        "ˌ"
                    } else {
                        self.tgt[p].ipa.as_str()
                    }
                })
                .collect()
        }
    }
}
