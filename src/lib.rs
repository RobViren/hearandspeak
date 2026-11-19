//! CPU speech recognition (Parakeet TDT) and text to speech (Kokoro) for x86-64 with AVX2, FMA and F16C.
//! One `Pool` of pinned workers serves both; calls through it are serialized.

#[cfg(feature = "asr")]
pub mod asr;
mod json;
pub mod pool;
pub mod resample;
mod simd;
mod st;
#[cfg(feature = "tts")]
pub mod tts;
pub mod weights;

pub use pool::Pool;
pub use weights::Weights;
