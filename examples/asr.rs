//! Transcribes wav files, one line each. Timing per file and per stage on stderr.
//! cargo run --release --example asr -- [-t threads] [-m model.safetensors -k tokenizer.json] [--chunk ms] files...

#[path = "util/wav.rs"]
mod wav;

use hearandspeak::{Pool, Weights, asr::Asr};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut threads, mut model, mut tok, mut chunk_ms, mut files) = (0, None, None, 0usize, Vec::new());
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-t" => (threads, i) = (args[i + 1].parse().unwrap(), i + 1),
            "-m" => (model, i) = (Some(args[i + 1].clone()), i + 1),
            "-k" => (tok, i) = (Some(args[i + 1].clone()), i + 1),
            "--chunk" => (chunk_ms, i) = (args[i + 1].parse().unwrap(), i + 1),
            f => files.push(f.to_string()),
        }
        i += 1;
    }
    let pool = Pool::new(threads);
    let t0 = Instant::now();
    let mut asr = match (model, tok) {
        (Some(m), Some(k)) => Asr::from_files(m.as_ref(), k.as_ref(), pool.clone()),
        _ => Asr::new(&Weights::asr(), pool.clone()),
    }
    .unwrap();
    eprintln!("load\t{:.2}\tthreads {}", t0.elapsed().as_secs_f64(), pool.n);
    let (mut audio, mut busy) = (0.0, 0.0);
    for f in &files {
        let (pcm, rate) = wav::read(f);
        let t = Instant::now();
        let text = if chunk_ms == 0 {
            asr.transcribe(&pcm, rate)
        } else {
            let mut s = asr.stream(rate);
            let mut parts = Vec::new();
            for c in pcm.chunks(rate as usize * chunk_ms / 1000) {
                parts.extend(s.push(c));
            }
            parts.extend(s.finish());
            parts.join(" ")
        };
        let secs = pcm.len() as f64 / rate as f64;
        let took = t.elapsed().as_secs_f64();
        audio += secs;
        busy += took;
        eprintln!("file\t{f}\t{secs:.2}\t{took:.3}");
        println!("{text}");
    }
    eprint!("{}", asr.timings());
    eprintln!("total\t{busy:.2}\taudio {audio:.1}\trtf {:.4}", busy / audio.max(1e-9));
}
