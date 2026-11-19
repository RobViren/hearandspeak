//! Synthesizes text (-i) or a phoneme string (-p) to a wav. Timing per chunk on stderr.
//! cargo run --release --example tts -- [-t threads] [-w weights_dir] [-v voice.bin] [-r reps] (-i text | -p phonemes) -o out.wav

#[path = "util/wav.rs"]
mod wav;

use hearandspeak::tts::{SAMPLE_RATE, Tts, Voice};
use hearandspeak::{Pool, Weights};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let pool = Pool::new(arg("-t").map_or(0, |t| t.parse().unwrap()));
    let mut w = Weights::tts();
    if let Some(d) = arg("-w") {
        w.dir = d.into();
    }
    let t0 = Instant::now();
    let tts = Tts::new(&w, pool.clone()).unwrap();
    eprintln!("load\t{:.2}\tthreads {}", t0.elapsed().as_secs_f64(), pool.n);
    let voice = Voice::from_file(arg("-v").expect("-v voice.bin").as_ref()).unwrap();
    let reps: usize = arg("-r").map_or(1, |r| r.parse().unwrap());
    let ps = arg("-p").expect("-p phonemes");
    let mut audio = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        audio = tts.synthesize(&ps, &voice, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let secs = audio.len() as f64 / SAMPLE_RATE as f64;
        eprintln!("synth\t{ms:.1}\tphonemes {}\taudio {secs:.2}\trtf {:.3}", ps.chars().count(), ms / 1e3 / secs);
    }
    if let Some(o) = arg("-o") {
        wav::write(&o, &audio, SAMPLE_RATE as u32);
    }
}
