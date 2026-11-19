//! Speech to text with Parakeet TDT 0.6B v3 (moondream's parakeet-ultra weights), int8 encoder on AVX2/AVX-VNNI.

mod decode;
mod dense;
mod encoder;
mod frontend;
mod int8;

use crate::pool::Pool;
use crate::resample::Resampler;
use crate::weights::Weights;
use decode::{MIN_SAMPLES, Tokenizer};
use encoder::Model;
pub use encoder::convert;
use frontend::SAMPLE_RATE;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const MODEL_FILE: &str = "parakeet-ultra/model.safetensors";
pub const TOKENIZER_FILE: &str = "parakeet-ultra/tokenizer.json";

pub struct Asr {
    model: Model,
    tok: Tokenizer,
}

impl Asr {
    /// Loads from `weights`, downloading missing files first (with the `download` feature).
    pub fn new(weights: &Weights, pool: Arc<Pool>) -> io::Result<Asr> {
        Asr::from_files(&weights.file(MODEL_FILE)?, &weights.file(TOKENIZER_FILE)?, pool)
    }

    /// `model` is moondream's parakeet-ultra model.safetensors (f16, quantized to int8 at load) or the int8 file
    /// `convert` writes.
    pub fn from_files(model: &Path, tokenizer: &Path, pool: Arc<Pool>) -> io::Result<Asr> {
        if !crate::simd::supported() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "needs a CPU with AVX2, FMA and F16C"));
        }
        Ok(Asr { tok: Tokenizer::load(tokenizer)?, model: Model::load(model, pool)? })
    }

    /// Whole recording at any sample rate, mono in [-1, 1]. Segments exactly as parakeetogo does: a recording up to
    /// 30 s is one segment, longer ones are cut at pauses into segments of at most 30 s.
    pub fn transcribe(&mut self, pcm: &[f32], rate: u32) -> String {
        let pcm = crate::resample::resample(pcm, rate, SAMPLE_RATE as u32);
        if pcm.len() < MIN_SAMPLES {
            return String::new();
        }
        let segs = self.model.segments(&pcm);
        let parts: Vec<String> =
            segs.into_iter().map(|(a, b)| self.model.transcribe_segment(&pcm[a..b], &self.tok)).collect();
        parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ")
    }

    /// Incremental transcription of audio pushed in chunks of any size, with default options.
    pub fn stream(&mut self, rate: u32) -> Stream<'_> {
        self.stream_with(rate, StreamOpts::default())
    }

    pub fn stream_with(&mut self, rate: u32, opts: StreamOpts) -> Stream<'_> {
        Stream {
            asr: self,
            resampler: (rate != SAMPLE_RATE as u32).then(|| Resampler::new(rate, SAMPLE_RATE as u32)),
            opts,
            pcm: Vec::new(),
            origin: 0,
            checked: 0,
            finals: 0,
            partial: None,
        }
    }

    /// Per stage time since load: stage, total ms, calls.
    pub fn timings(&self) -> String {
        self.model.times.report()
    }
}

/// When a streamed segment closes.
#[derive(Clone, Copy, Debug)]
pub struct StreamOpts {
    /// Trailing silence after speech, in seconds, that closes the segment.
    pub pause: f32,
    /// Longest segment in seconds; past this the segment is cut at its best pause (or hard cut).
    pub max_seconds: f32,
    /// How much new audio, in seconds, between VAD checks of the open segment.
    pub check_every: f32,
}

impl Default for StreamOpts {
    fn default() -> Self {
        StreamOpts { pause: 0.6, max_seconds: 30.0, check_every: 0.2 }
    }
}

/// A finished segment: its text and where it sits in the stream, in seconds.
#[derive(Clone, Debug)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Holds only the audio since the last final segment. Every `check_every` seconds of new audio the VAD head scans
/// it: once speech is followed by `pause` seconds of silence the segment is transcribed and returned as final, and
/// its audio dropped. Silence alone is dropped without transcribing.
pub struct Stream<'a> {
    asr: &'a mut Asr,
    resampler: Option<Resampler>,
    opts: StreamOpts,
    pcm: Vec<f32>,
    origin: usize,
    checked: usize,
    finals: usize,
    partial: Option<(usize, String)>,
}

enum Cut {
    Final(usize),
    Drop(usize),
}

impl Stream<'_> {
    pub fn push(&mut self, chunk: &[f32]) -> Vec<Segment> {
        match &mut self.resampler {
            Some(r) => r.push(chunk, &mut self.pcm),
            None => self.pcm.extend_from_slice(chunk),
        }
        let mut out = Vec::new();
        let step = (self.opts.check_every * SAMPLE_RATE as f32) as usize;
        if self.pcm.len() < self.checked + step.max(1) {
            return out;
        }
        self.checked = self.pcm.len();
        while let Some(cut) = self.cut() {
            match cut {
                Cut::Final(n) => out.extend(self.close(n)),
                Cut::Drop(n) => self.drop_front(n),
            }
        }
        out
    }

    pub fn finish(mut self) -> Vec<Segment> {
        if let Some(r) = &mut self.resampler {
            r.finish(&mut self.pcm);
        }
        if self.pcm.len() < MIN_SAMPLES {
            return Vec::new();
        }
        let speech = !self.asr.model.block_speech(&self.pcm).is_empty();
        if !speech && self.finals > 0 {
            return Vec::new();
        }
        let n = self.pcm.len();
        self.close(n).into_iter().collect()
    }

    /// Text of the open segment as it would read if it closed now. Costs one transcription of at most
    /// `max_seconds` of audio, and nothing when no audio arrived since the last call.
    pub fn partial(&mut self) -> String {
        if self.pcm.len() < MIN_SAMPLES {
            return String::new();
        }
        if let Some((n, t)) = &self.partial
            && *n == self.pcm.len()
        {
            return t.clone();
        }
        let t = self.asr.model.transcribe_segment(&self.pcm, &self.asr.tok);
        self.partial = Some((self.pcm.len(), t.clone()));
        t
    }

    fn cut(&mut self) -> Option<Cut> {
        let sr = SAMPLE_RATE as f64;
        let n = self.pcm.len();
        if n < MIN_SAMPLES {
            return None;
        }
        let secs = n as f64 / sr;
        let (pause, max) = (self.opts.pause as f64, self.opts.max_seconds as f64);
        let regions = self.asr.model.block_speech(&self.pcm);
        let Some(&(_, end)) = regions.last() else {
            // Silence only: keep a pause's worth in case speech is starting.
            let keep = (pause * sr) as usize;
            return (n > 2 * keep + MIN_SAMPLES).then(|| Cut::Drop(n - keep));
        };
        if secs - end >= pause {
            return Some(Cut::Final((((end + pause / 2.0) * sr) as usize).min(n)));
        }
        if secs > max {
            let at = decode::next_cut(&decode::pauses(regions, secs)).min(max);
            return Some(Cut::Final(((at * sr).round() as usize).clamp(MIN_SAMPLES, n)));
        }
        None
    }

    fn close(&mut self, n: usize) -> Option<Segment> {
        let text = self.asr.model.transcribe_segment(&self.pcm[..n], &self.asr.tok);
        let sr = SAMPLE_RATE as f64;
        let seg = Segment { text, start: self.origin as f64 / sr, end: (self.origin + n) as f64 / sr };
        self.drop_front(n);
        self.finals += 1;
        (!seg.text.is_empty()).then_some(seg)
    }

    fn drop_front(&mut self, n: usize) {
        self.pcm.drain(..n);
        self.origin += n;
        self.checked = self.checked.saturating_sub(n);
        self.partial = None;
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
const STAGES: [&str; 9] =
    ["frontend", "subsampler", "quantize", "int8_gemm", "attention", "depthwise", "norms_act", "decoder_joint", "vad"];

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

#[doc(hidden)]
pub fn bench_frontend(pool: &Pool, pcm: &[f32], reps: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(frontend::log_mel(pool, pcm));
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1e3
}
