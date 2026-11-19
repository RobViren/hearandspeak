//! Speech to text with Parakeet TDT 0.6B v3 (moondream's parakeet-ultra weights), int8 encoder on AVX2/AVX-VNNI.

mod decode;
mod dense;
mod encoder;
mod frontend;
mod int8;

use crate::pool::Pool;
use crate::resample::Resampler;
use crate::weights::Weights;
use decode::{Cursor, MIN_SAMPLES, Tokenizer};
use encoder::Model;
pub use encoder::convert;
use frontend::SAMPLE_RATE;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const MODEL_FILE: &str = "model.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

pub struct Asr {
    model: Model,
    tok: Tokenizer,
}

impl Asr {
    /// Loads from `weights`, downloading missing files first (with the `download` feature).
    pub fn new(weights: &Weights, pool: Arc<Pool>) -> io::Result<Asr> {
        Asr::from_files(
            &weights.file(MODEL_FILE)?,
            &weights.file(TOKENIZER_FILE)?,
            pool,
        )
    }

    /// `model` is moondream's parakeet-ultra model.safetensors (f16, quantized to int8 at load) or the int8 file
    /// `convert` writes.
    pub fn from_files(model: &Path, tokenizer: &Path, pool: Arc<Pool>) -> io::Result<Asr> {
        if !crate::simd::supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "needs a CPU with AVX2, FMA and F16C",
            ));
        }
        Ok(Asr {
            tok: Tokenizer::load(tokenizer)?,
            model: Model::load(model, pool)?,
        })
    }

    /// Whole recording at any sample rate, mono in [-1, 1].
    pub fn transcribe(&mut self, pcm: &[f32], rate: u32) -> String {
        let mut s = self.stream(rate);
        let mut parts = s.push(pcm);
        parts.extend(s.finish());
        parts.join(" ")
    }

    /// Incremental transcription of audio pushed in chunks of any size.
    pub fn stream(&mut self, rate: u32) -> Stream<'_> {
        Stream {
            asr: self,
            resampler: (rate != SAMPLE_RATE as u32)
                .then(|| Resampler::new(rate, SAMPLE_RATE as u32)),
            pcm: Vec::new(),
            cursor: Cursor::default(),
            origin: 0,
            seen: 0,
            cache: HashMap::new(),
        }
    }

    /// Per stage time since load: stage, total ms, calls.
    pub fn timings(&self) -> String {
        self.model.times.report()
    }
}

/// Segments close at pauses once the VAD can place them, on exactly the cuts a whole-recording transcription makes:
/// a recording up to 30 s is one segment, longer ones are cut into segments of at most 30 s, decided a 120 s block
/// at a time. `push` returns the text of segments that closed; `partial` transcribes the audio still open (cached per
/// segment, so calling it regularly keeps `finish` down to the last segment).
pub struct Stream<'a> {
    asr: &'a mut Asr,
    resampler: Option<Resampler>,
    pcm: Vec<f32>,
    cursor: Cursor,
    origin: usize,
    seen: usize,
    cache: HashMap<(usize, usize), String>,
}

impl Stream<'_> {
    pub fn push(&mut self, chunk: &[f32]) -> Vec<String> {
        match &mut self.resampler {
            Some(r) => r.push(chunk, &mut self.pcm),
            None => self.pcm.extend_from_slice(chunk),
        }
        if self.pcm.len() < self.cursor.need {
            return Vec::new();
        }
        self.close(false)
    }

    pub fn finish(mut self) -> Vec<String> {
        if let Some(r) = &mut self.resampler {
            r.finish(&mut self.pcm);
        }
        if self.origin + self.pcm.len() < MIN_SAMPLES {
            return Vec::new();
        }
        self.close(true)
    }

    /// Text of the audio not yet closed into segments, as it would read if the recording ended now.
    pub fn partial(&mut self) -> String {
        let open = &self.pcm[self.cursor.open..];
        if open.len() < MIN_SAMPLES {
            return String::new();
        }
        let base = self.origin + self.cursor.open;
        let mut c = Cursor::default();
        let segs = self.asr.model.segments(open, &mut c, true);
        let mut parts = Vec::new();
        for (a, b) in segs {
            let key = (base + a, base + b);
            if !self.cache.contains_key(&key) {
                let text = self
                    .asr
                    .model
                    .transcribe_segment(&open[a..b], &self.asr.tok);
                self.cache.insert(key, text);
            }
            parts.push(self.cache[&key].clone());
        }
        parts.retain(|p| !p.is_empty());
        parts.join(" ")
    }

    fn close(&mut self, last: bool) -> Vec<String> {
        self.cursor.moved |= self.origin > 0;
        let segs = self.asr.model.segments(&self.pcm, &mut self.cursor, last);
        let mut out = Vec::new();
        for (a, b) in segs {
            let key = (self.origin + a, self.origin + b);
            let text = match self.cache.remove(&key) {
                Some(t) => t,
                None => self
                    .asr
                    .model
                    .transcribe_segment(&self.pcm[a..b], &self.asr.tok),
            };
            self.seen = key.1;
            if !text.is_empty() {
                out.push(text);
            }
        }
        let seen = self.seen;
        self.cache.retain(|k, _| k.0 >= seen);
        let keep = self.cursor.keep;
        self.pcm.drain(..keep);
        self.origin += keep;
        self.cursor.open -= keep;
        self.cursor.need -= keep;
        self.cursor.keep = 0;
        out
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Frontend,
    Subsampler,
    Quantize,
    Int8Gemm,
    Attention,
    Depthwise,
    Norms,
    Decode,
    Vad,
}
const STAGES: [&str; 9] = [
    "frontend",
    "subsampler",
    "quantize",
    "int8_gemm",
    "attention",
    "depthwise",
    "norms_act",
    "decoder_joint",
    "vad",
];

#[derive(Default)]
pub(crate) struct Times {
    nanos: [AtomicU64; 9],
    calls: [AtomicU64; 9],
}

pub(crate) struct Clock(*const Times, Stage, Instant);
impl Drop for Clock {
    fn drop(&mut self) {
        let t = unsafe { &*self.0 };
        t.nanos[self.1 as usize].fetch_add(self.2.elapsed().as_nanos() as u64, Relaxed);
        t.calls[self.1 as usize].fetch_add(1, Relaxed);
    }
}

impl Times {
    pub fn clock(&self, s: Stage) -> Clock {
        Clock(self, s, Instant::now())
    }
    fn report(&self) -> String {
        (0..9)
            .map(|i| {
                format!(
                    "{}\t{:.1}\t{}\n",
                    STAGES[i],
                    self.nanos[i].load(Relaxed) as f64 / 1e6,
                    self.calls[i].load(Relaxed)
                )
            })
            .collect()
    }
}
