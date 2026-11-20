//! Text to speech with Kokoro-82M and kokoallovero's English G2P (American English).

mod g2p;
mod model;
mod ops;
mod rng;

use crate::pool::Pool;
use crate::weights::Weights;
use g2p::G2p;
pub use model::SAMPLE_RATE;
use model::{Model, Noise};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Files a `Weights` directory must hold for TTS, laid out as in the HF repo.
pub const FILES: [&str; 6] = [
    "kokoro/kokoro.bin",
    "kokoro/kokoro.tsv",
    "kokoro/vocab.tsv",
    "hearandspeak/g2p/lexicon.bin",
    "kokoro/g2p/tagger.bin",
    "kokoro/g2p/guesser.bin",
];

const BUILTIN: [(&str, &[u8]); 3] = [
    ("af_heart", include_bytes!("../../voices/af_heart.bin")),
    ("am_michael", include_bytes!("../../voices/am_michael.bin")),
    ("bf_emma", include_bytes!("../../voices/bf_emma.bin")),
];

/// Silence between chunks of one text, as kokoallovero's CLI puts it.
const GAP_SECONDS: f32 = 0.15;

/// A Kokoro voice pack: 510 style vectors of 256 floats, one per phoneme count (hexgrad's voices/*.pt as raw f32).
#[derive(Clone)]
pub struct Voice(Vec<f32>);

impl Voice {
    /// af_heart, am_michael and bf_emma are built in.
    pub fn builtin(name: &str) -> Option<Voice> {
        BUILTIN.iter().find(|(n, _)| *n == name).and_then(|(_, b)| Voice::from_bytes(b).ok())
    }
    pub fn builtin_names() -> impl Iterator<Item = &'static str> {
        BUILTIN.iter().map(|(n, _)| *n)
    }
    /// Raw little-endian f32 [510][1][256] (522240 bytes), all finite.
    pub fn from_bytes(b: &[u8]) -> io::Result<Voice> {
        if b.len() != 510 * 256 * 4 {
            return Err(crate::st::bad(format!("voice pack is {} bytes, expected {}", b.len(), 510 * 256 * 4)));
        }
        let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if !v.iter().all(|x| x.is_finite()) {
            return Err(crate::st::bad("voice pack holds non-finite values"));
        }
        Ok(Voice(v))
    }
    pub fn from_file(path: &Path) -> io::Result<Voice> {
        Voice::from_bytes(&std::fs::read(path)?)
    }
    fn style(&self, phonemes: usize) -> &[f32] {
        let i = phonemes.clamp(1, 510) - 1;
        &self.0[i * 256..(i + 1) * 256]
    }
}

pub struct Tts {
    model: Model,
    g2p: G2p,
    vocab: HashMap<char, usize>,
    pool: Arc<Pool>,
}

impl Tts {
    /// Loads from `weights`, downloading missing files first (with the `download` feature).
    pub fn new(weights: &Weights, pool: Arc<Pool>) -> io::Result<Tts> {
        let f = FILES.map(|n| weights.file(n));
        let [bin, tsv, vocab, lex, tag, guess] = f;
        Tts::from_files(&bin?, &tsv?, &vocab?, &lex?, &tag?, &guess?, pool)
    }

    pub fn from_files(
        bin: &Path,
        tsv: &Path,
        vocab: &Path,
        lexicon: &Path,
        tagger: &Path,
        guesser: &Path,
        pool: Arc<Pool>,
    ) -> io::Result<Tts> {
        if !crate::simd::supported() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "needs a CPU with AVX2, FMA and F16C"));
        }
        let vocab = std::fs::read_to_string(vocab)?
            .lines()
            .filter_map(|l| {
                let mut p = l.split_whitespace();
                Some((char::from_u32(p.next()?.parse().ok()?)?, p.next()?.parse().ok()?))
            })
            .collect();
        let g2p = G2p::load(std::fs::read(lexicon)?, &std::fs::read(tagger)?, &std::fs::read(guesser)?)?;
        Ok(Tts { model: Model::new(model::Weights::load(bin, tsv)?), g2p, vocab, pool })
    }

    /// Kokoro phoneme chunks for English text, each at most 510 symbols.
    pub fn phonemize(&self, text: &str) -> Vec<String> {
        self.g2p.phonemize(text)
    }

    /// The text's normalized form (numbers, abbreviations, paths and symbols spelled out) before tokenizing.
    pub fn normalize(&self, text: &str) -> String {
        g2p::normalize(&self.g2p, text)
    }

    /// One chunk of Kokoro phonemes to 24 kHz mono audio. Symbols outside Kokoro's vocabulary are skipped and only
    /// the first 510 known ones are spoken; none gives empty audio. Speed is clamped to [0.25, 4], and a
    /// non-positive or NaN speed means 1.
    pub fn synthesize(&self, phonemes: &str, voice: &Voice, speed: f32) -> Vec<f32> {
        let n = phonemes.chars().count();
        let mut ids = vec![0];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()).take(510));
        if ids.len() == 1 {
            return Vec::new();
        }
        ids.push(0);
        let speed = if speed > 0.0 { speed.clamp(0.25, 4.0) } else { 1.0 };
        let mut noise = Noise { initial_phase: Vec::new(), randn: Vec::new(), seed: 0 };
        model::synthesize(&self.model, &self.pool, &ids, voice.style(n), speed, &mut noise)
    }

    /// Streams text as audio: chunks are split at sentence boundaries (the first sentence alone, so it plays
    /// sooner) and each is rendered only when the iterator reaches it. Chunks after the first start with 0.15 s
    /// of silence.
    pub fn speak<'a>(&'a self, text: &str, voice: &'a Voice, speed: f32) -> impl Iterator<Item = Vec<f32>> + 'a {
        self.phonemize(text).into_iter().enumerate().map(move |(i, ps)| {
            let mut audio = if i > 0 { vec![0.0; (GAP_SECONDS * SAMPLE_RATE as f32) as usize] } else { Vec::new() };
            audio.extend(self.synthesize(&ps, voice, speed));
            audio
        })
    }
}
