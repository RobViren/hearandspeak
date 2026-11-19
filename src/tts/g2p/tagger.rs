//! Averaged perceptron POS tagger (spaCy en_core_web_sm tag_ set) over hashed features.

use std::io;

const MAGIC: u32 = 0x47544B4B;
const SEP: u8 = 0x1f;

pub struct Tagger {
    words: Vec<u32>, // slots [cap][2] then entries
    cap: usize,
    tags: Vec<String>,
}

struct Hash(u64);
impl Hash {
    fn new() -> Hash {
        Hash(0xCBF29CE484222325)
    }
    fn b(mut self, x: u8) -> Hash {
        self.0 = (self.0 ^ x as u64).wrapping_mul(0x100000001B3);
        self
    }
    fn s(mut self, s: &[u8]) -> Hash {
        for &c in s {
            self = self.b(c);
        }
        self
    }
}

fn f1(id: u8, v: &[u8]) -> u64 {
    Hash::new().b(id).b(SEP).s(v).0
}
fn f2(id: u8, a: &[u8], b: &[u8]) -> u64 {
    Hash::new().b(id).b(SEP).s(a).b(SEP).s(b).0
}

fn suffix(s: &[u8], n: usize) -> &[u8] {
    &s[s.len().saturating_sub(n)..]
}
fn prefix(s: &[u8], n: usize) -> &[u8] {
    &s[..s.len().min(n)]
}

fn shape(w: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut last, mut run) = (-1i32, 0);
    for &b in w {
        let c = match b {
            b'A'..=b'Z' => b'X',
            b'a'..=b'z' => b'x',
            b'0'..=b'9' => b'd',
            128.. => b'u',
            _ => b,
        } as i32;
        run = if c == last { run + 1 } else { 1 };
        last = c;
        if run <= 4 {
            out.push(c as u8);
        }
    }
    out
}

fn lower(w: &[u8]) -> Vec<u8> {
    w.iter().map(|&c| if c.is_ascii_uppercase() { c + 32 } else { c }).collect()
}

impl Tagger {
    pub fn load(data: &[u8]) -> io::Result<Tagger> {
        let bad = || crate::st::bad("tagger.bin");
        let u = |i: usize| data.get(i * 4..i * 4 + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let (magic, version, ntags, cap, nent) = (
            u(0).ok_or_else(bad)?,
            u(1).ok_or_else(bad)?,
            u(2).ok_or_else(bad)? as usize,
            u(3).ok_or_else(bad)? as usize,
            u(4).ok_or_else(bad)? as usize,
        );
        if magic != MAGIC
            || version != 1
            || ntags == 0
            || ntags > 256
            || !cap.is_power_of_two()
            || data.len() < 20 + 8 * ntags + 8 * cap + 4 * nent
        {
            return Err(bad());
        }
        let tags = (0..ntags)
            .map(|i| {
                let t = &data[20 + 8 * i..28 + 8 * i];
                String::from_utf8_lossy(&t[..t.iter().position(|&c| c == 0).unwrap_or(8)]).into_owned()
            })
            .collect();
        let base = 20 + 8 * ntags;
        let words = data[base..base + 8 * cap + 4 * nent]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Tagger { words, cap, tags })
    }

    fn add(&self, h: u64, scores: &mut [i32]) {
        let mask = self.cap as u64 - 1;
        let fp = ((h >> 32) as u32).max(1);
        let mut s = h & mask;
        loop {
            let key = self.words[2 * s as usize];
            if key == 0 {
                return;
            }
            if key == fp {
                let mut e = 2 * self.cap + self.words[2 * s as usize + 1] as usize;
                loop {
                    let v = self.words[e];
                    scores[(v & 0xff) as usize] += (v >> 16) as u16 as i16 as i32;
                    if v & 0x100 != 0 {
                        return;
                    }
                    e += 1;
                }
            }
            s = (s + 1) & mask;
        }
    }

    pub fn tag(&self, tokens: &[String]) -> Vec<String> {
        let n = tokens.len();
        let mut lw: Vec<Vec<u8>> = vec![Vec::new(); n + 4];
        let mut sh: Vec<Vec<u8>> = vec![Vec::new(); n + 4];
        lw[0] = b"-S2-".to_vec();
        lw[1] = b"-S-".to_vec();
        lw[n + 2] = b"-E-".to_vec();
        lw[n + 3] = b"-E2-".to_vec();
        for i in [0, 1, n + 2, n + 3] {
            sh[i] = shape(&lw[i]);
        }
        for (i, t) in tokens.iter().enumerate() {
            lw[i + 2] = lower(t.as_bytes());
            sh[i + 2] = shape(t.as_bytes());
        }
        let mut out = Vec::with_capacity(n);
        let (mut p1, mut p2) = (b"-S-".to_vec(), b"-S2-".to_vec());
        let mut scores = vec![0i32; self.tags.len()];
        for i in 0..n {
            let k = i + 2;
            let (w, p, nx) = (&lw[k], &lw[k - 1], &lw[k + 1]);
            scores.fill(0);
            let hs = [
                f1(0, b""),
                f1(1, w),
                f1(2, suffix(w, 2)),
                f1(3, suffix(w, 3)),
                f1(4, suffix(w, 4)),
                f1(5, prefix(w, 1)),
                f1(6, prefix(w, 2)),
                f1(7, &sh[k]),
                f1(8, p),
                f1(9, suffix(p, 3)),
                f1(10, &sh[k - 1]),
                f1(11, &lw[k - 2]),
                f1(12, nx),
                f1(13, suffix(nx, 3)),
                f1(14, &sh[k + 1]),
                f1(15, &lw[k + 2]),
                f2(16, p, w),
                f2(17, w, nx),
                f1(30, &p1),
                f1(31, &p2),
                f2(32, &p1, &p2),
                f2(33, &p1, w),
            ];
            for h in hs {
                self.add(h, &mut scores);
            }
            let mut best = 0;
            for c in 1..scores.len() {
                if scores[c] > scores[best] {
                    best = c;
                }
            }
            out.push(self.tags[best].clone());
            p2 = std::mem::replace(&mut p1, self.tags[best].as_bytes().to_vec());
        }
        out
    }
}
