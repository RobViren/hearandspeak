//! Tokenizer, TDT greedy decode, and VAD segmentation so long audio is cut into segments of at most 30 s.

use super::Stage;
use super::dense::{Picks, argmax, linear_half, sigmoid};
use super::encoder::{DURATIONS, ENCODED, Model, VOCAB};
use super::frontend::{self, SAMPLE_RATE};
use crate::json::Json;
use crate::st::bad;
use std::io;
use std::path::Path;

const BLANK: usize = 8192;
const PAD: usize = 2;

pub struct Tokenizer {
    vocab: Vec<String>,
    special: Vec<bool>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> io::Result<Tokenizer> {
        let text = std::fs::read_to_string(path)?;
        let j = Json::parse(&text).ok_or_else(|| bad("tokenizer.json"))?;
        let mut vocab = vec![String::new(); VOCAB];
        let mut special = vec![false; VOCAB];
        for t in j.get("added_tokens").map_or(&[][..], |a| a.arr()) {
            let id = t.get("id").map_or(0.0, |x| x.num()) as usize;
            if id < VOCAB {
                special[id] = t.get("special").is_some_and(|s| s.bool());
            }
        }
        for (tok, id) in j.get("model").and_then(|m| m.get("vocab")).map_or(&[][..], |v| v.obj()) {
            let id = id.num() as usize;
            if id < VOCAB {
                vocab[id] = tok.clone();
            }
        }
        Ok(Tokenizer { vocab, special })
    }

    /// steps are (frame, token, duration) triples.
    fn detokenize(&self, steps: &[usize]) -> String {
        let mut text = String::new();
        for s in steps.chunks_exact(3) {
            let t = s[1];
            if t != BLANK && t != PAD && !self.special[t] {
                text.push_str(&self.vocab[t]);
            }
        }
        text.replace('\u{2581}', " ").trim_matches([' ', '\n', '\t', '\r']).to_string()
    }
}

impl Model {
    fn predict(&self, token: usize, h: &mut [f32], c: &mut [f32], predicted: &mut [f32]) {
        let d = ENCODED;
        let mut inp = [0f32; 2 * ENCODED];
        let mut gates = [0f32; 4 * ENCODED];
        static ALL: std::sync::OnceLock<(Picks, Picks)> = std::sync::OnceLock::new();
        let (all, one) =
            ALL.get_or_init(|| (Picks::new(&[0.0; 2 * ENCODED], false), Picks::new(&[0.0; ENCODED], false)));
        for i in 0..2 {
            if i == 0 {
                inp[..d].copy_from_slice(&self.embedding[token * d..(token + 1) * d]);
            } else {
                inp[..d].copy_from_slice(&h[..d]);
            }
            inp[d..].copy_from_slice(&h[i * d..(i + 1) * d]);
            linear_half(&self.pool, &self.cell[i], inp.as_ptr(), all, gates.as_mut_ptr());
            unsafe {
                sigmoid(gates.as_mut_ptr(), 2 * d);
                sigmoid(gates.as_mut_ptr().add(3 * d), d);
            }
            for j in 0..d {
                c[i * d + j] = gates[d + j] * c[i * d + j] + gates[j] * gates[2 * d + j].tanh();
                h[i * d + j] = gates[3 * d + j] * c[i * d + j].tanh();
            }
        }
        linear_half(&self.pool, &self.dec_proj, h[d..].as_ptr(), one, predicted.as_mut_ptr());
    }

    /// (frame, token, duration) per joint evaluation, blanks included. The step budget is 10 per frame for the
    /// segment with no per frame symbol cap: a non blank token with duration 0 stays on its frame.
    pub fn greedy(&mut self, encoded: &[f32], t: usize) -> Vec<usize> {
        let _c = self.times.clock(Stage::Decode);
        let d = ENCODED;
        let mut h = vec![0f32; 2 * d];
        let mut c = vec![0f32; 2 * d];
        let mut predicted = [0f32; ENCODED];
        let mut inp = [0f32; ENCODED];
        let mut logits = vec![0f32; (VOCAB + DURATIONS).div_ceil(16) * 16];
        let mut steps = Vec::new();
        self.predict(BLANK, &mut h, &mut c, &mut predicted);
        let (mut frame, mut tokens) = (0, 0);
        while frame < t && steps.len() < 3 * 10 * t && tokens < 4096 {
            for i in 0..d {
                inp[i] = (encoded[frame * d + i] + predicted[i]).max(0.0);
            }
            linear_half(&self.pool, &self.joint, inp.as_ptr(), &Picks::new(&inp, true), logits.as_mut_ptr());
            let token = unsafe { argmax(&logits[..VOCAB]) };
            let dl = &logits[VOCAB..VOCAB + DURATIONS];
            let mut duration = (0..DURATIONS).fold(0, |b, i| if dl[i] > dl[b] { i } else { b });
            if token == BLANK && duration == 0 {
                duration = 1;
            }
            steps.extend_from_slice(&[frame, token, duration]);
            frame += duration;
            if token != BLANK {
                self.predict(token, &mut h, &mut c, &mut predicted);
                tokens += 1;
            }
        }
        steps
    }

    /// The VAD head sees every subsampler row, including the ones past the valid length (reference quirk).
    pub(crate) fn block_speech(&mut self, pcm: &[f32]) -> Vec<(f64, f64)> {
        let (mel, frames, mel_valid) = frontend::log_mel(&self.pool, pcm);
        let (hidden, rows, valid) = self.subsample(&mel, frames, mel_valid);
        let p = self.vad(&hidden, rows);
        let seconds = pcm.len() as f64 / SAMPLE_RATE as f64;
        let mut regions: Vec<(f64, f64)> = Vec::new();
        let mut start: Option<usize> = None;
        for i in 0..=valid {
            let speaking = i < valid && p[i] >= VAD_THRESHOLD;
            if speaking && start.is_none() {
                start = Some(i);
            }
            let Some(s) = start else { continue };
            if speaking {
                continue;
            }
            let (a, b) = (s as f64 * FRAME_SECONDS, i as f64 * FRAME_SECONDS);
            match regions.last_mut() {
                Some(last) if a - last.1 < VAD_MIN_GAP => last.1 = b,
                _ => regions.push((a, b)),
            }
            start = None;
        }
        regions.into_iter().filter(|(a, b)| b - a >= VAD_MIN_SPEECH).map(|(a, b)| (a, b.min(seconds))).collect()
    }

    /// Sample ranges to transcribe for a whole recording, as parakeetogo cuts them: up to 30 s is one segment and
    /// never reaches the VAD head; longer audio is scanned in 120 s VAD blocks and cut at pauses into segments of at
    /// most 30 s, dropping those without speech.
    pub fn segments(&mut self, pcm: &[f32]) -> Vec<(usize, usize)> {
        let cap = (SEGMENT_SECONDS * SAMPLE_RATE as f64) as usize;
        let total = pcm.len();
        if total <= cap {
            return vec![(0, total)];
        }
        let sr = SAMPLE_RATE as f64;
        let mut out = Vec::new();
        let mut regions: Vec<(f64, f64)> = Vec::new();
        let (mut start, mut read) = (0, 0);
        while read < total {
            let remaining = total - read;
            let block =
                if remaining <= BLOCK_SAMPLES { remaining } else { BLOCK_SAMPLES.min(remaining - MIN_TAIL_SAMPLES) };
            let offset = (read as f64 - start as f64) / sr;
            for (a, b) in self.block_speech(&pcm[read..read + block]) {
                if b + offset > 0.0 {
                    regions.push(((a + offset).max(0.0), b + offset));
                }
            }
            read += block;
            while read - start > cap {
                let cut =
                    (next_cut(&pauses(regions.clone(), (read - start) as f64 / sr)) * sr).round_ties_even() as usize;
                let seconds = cut as f64 / sr;
                if has_speech(&regions, seconds) {
                    out.push((start, start + cut));
                }
                start += cut;
                regions = regions
                    .iter()
                    .filter(|r| r.1 > seconds)
                    .map(|r| ((r.0 - seconds).max(0.0), r.1 - seconds))
                    .collect();
            }
        }
        if read - start >= MIN_SAMPLES && has_speech(&regions, (read - start) as f64 / sr) {
            out.push((start, read));
        }
        out
    }

    pub fn transcribe_segment(&mut self, pcm: &[f32], tok: &Tokenizer) -> String {
        let (encoded, t) = self.encode(pcm);
        let steps = self.greedy(&encoded, t);
        tok.detokenize(&steps)
    }
}

const FRAME_SECONDS: f64 = 0.08;
const VAD_THRESHOLD: f32 = 0.5;
const VAD_MIN_SPEECH: f64 = 0.1;
const VAD_MIN_GAP: f64 = 0.1;
const SEGMENT_SECONDS: f64 = 30.0;
const MIN_SEGMENT_SECONDS: f64 = 1.0;
const MIN_PAUSE: f64 = 0.2;
const PAUSE_EPS: f64 = 1e-6;
const BLOCK_SAMPLES: usize = 120 * SAMPLE_RATE;
const MIN_TAIL_SAMPLES: usize = SAMPLE_RATE / 2;
pub const MIN_SAMPLES: usize = 320;

pub(crate) fn pauses(mut regions: Vec<(f64, f64)>, seconds: f64) -> Vec<(f64, f64)> {
    regions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = Vec::new();
    let mut previous: f64 = 0.0;
    for (a, b) in regions {
        if a - previous >= MIN_PAUSE - PAUSE_EPS {
            out.push((previous, a));
        }
        previous = previous.max(b);
    }
    if seconds - previous >= MIN_PAUSE - PAUSE_EPS {
        out.push((previous, seconds));
    }
    out
}

pub(crate) fn next_cut(pauses: &[(f64, f64)]) -> f64 {
    let mut pick = pauses.iter().rev().find(|p| p.0 >= MIN_SEGMENT_SECONDS && p.1 <= SEGMENT_SECONDS);
    if pick.is_none() {
        pick = pauses.iter().rev().find(|p| {
            let mid = (p.0 + p.1) / 2.0;
            (MIN_SEGMENT_SECONDS..=SEGMENT_SECONDS).contains(&mid)
        });
    }
    pick.map_or(SEGMENT_SECONDS, |p| (p.0 + p.1) / 2.0)
}

fn has_speech(regions: &[(f64, f64)], seconds: f64) -> bool {
    regions.iter().any(|r| r.1 > 0.0 && r.0 < seconds)
}
