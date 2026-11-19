//! Text to speech with Kokoro-82M and kokoallovero's English G2P.

mod model;
mod ops;
mod rng;

use crate::pool::Pool;
use crate::weights::Weights;
pub use model::SAMPLE_RATE;
use model::{Model, Noise};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// A voice pack: 510 style vectors of 256, one per phoneme count.
#[derive(Clone)]
pub struct Voice(Vec<f32>);

impl Voice {
    pub fn from_bytes(b: &[u8]) -> io::Result<Voice> {
        if b.len() != 510 * 256 * 4 {
            return Err(crate::st::bad(format!("voice pack is {} bytes, expected {}", b.len(), 510 * 256 * 4)));
        }
        Ok(Voice(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
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
    vocab: HashMap<char, usize>,
    pool: Arc<Pool>,
}

impl Tts {
    pub fn from_files(bin: &Path, tsv: &Path, vocab: &Path, pool: Arc<Pool>) -> io::Result<Tts> {
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
        Ok(Tts { model: Model::new(model::Weights::load(bin, tsv)?), vocab, pool })
    }

    pub fn new(weights: &Weights, pool: Arc<Pool>) -> io::Result<Tts> {
        Tts::from_files(&weights.file("kokoro.bin")?, &weights.file("kokoro.tsv")?, &weights.file("vocab.tsv")?, pool)
    }

    /// One chunk of Kokoro phonemes (at most 510) to 24 kHz audio.
    pub fn synthesize(&self, phonemes: &str, voice: &Voice, speed: f32) -> Vec<f32> {
        let n = phonemes.chars().count();
        if n == 0 {
            return Vec::new();
        }
        let mut ids = vec![0];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()));
        ids.push(0);
        let mut noise = Noise { initial_phase: Vec::new(), randn: Vec::new(), seed: 0 };
        model::synthesize(&self.model, &self.pool, &ids, voice.style(n), speed, &mut noise)
    }
}
