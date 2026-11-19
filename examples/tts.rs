//! Speaks text (-i) or a phoneme string (-p) into a wav, or with --g2p prints phoneme chunks for stdin lines
//! (tab separated, like kkg2p). Timing per chunk on stderr.
//! cargo run --release --example tts -- [-t threads] [-w weights_dir] [-v voice|file.bin] [-s speed] [-r reps]
//!     (-i text | -p phonemes | --g2p) [-o out.wav]

#[path = "util/wav.rs"]
mod wav;

use hearandspeak::tts::{SAMPLE_RATE, Tts, Voice};
use hearandspeak::{Pool, Weights};
use std::io::BufRead;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let pool = Pool::new(arg("-t").map_or(0, |t| t.parse().unwrap()));
    let t0 = Instant::now();
    // -w is a kokoallovero weights dir; -l swaps in another lexicon.bin.
    let tts = match arg("-w") {
        Some(d) => {
            let p = |n: &str| std::path::PathBuf::from(&d).join(n);
            let lex = arg("-l").map_or_else(|| p("g2p/lexicon.bin"), Into::into);
            let (bin, tsv, voc, tag, gs) =
                (p("kokoro.bin"), p("kokoro.tsv"), p("vocab.tsv"), p("g2p/tagger.bin"), p("g2p/guesser.bin"));
            Tts::from_files(&bin, &tsv, &voc, &lex, &tag, &gs, pool.clone())
        }
        None => Tts::new(&Weights::tts(), pool.clone()),
    }
    .unwrap();
    eprintln!("load\t{:.2}\tthreads {}", t0.elapsed().as_secs_f64(), pool.n);
    if args.iter().any(|a| a == "--g2p") {
        for line in std::io::stdin().lock().lines() {
            let t = Instant::now();
            println!("{}", tts.phonemize(&line.unwrap()).join("\t"));
            eprintln!("g2p\t{:.2}", t.elapsed().as_secs_f64() * 1e3);
        }
        return;
    }
    let v = arg("-v").unwrap_or_else(|| "af_heart".into());
    let voice = Voice::builtin(&v).unwrap_or_else(|| Voice::from_file(v.as_ref()).unwrap());
    let speed: f32 = arg("-s").map_or(1.0, |s| s.parse().unwrap());
    let reps: usize = arg("-r").map_or(1, |r| r.parse().unwrap());
    let mut audio = Vec::new();
    for _ in 0..reps {
        audio.clear();
        let t = Instant::now();
        if let Some(ps) = arg("-p") {
            audio = tts.synthesize(&ps, &voice, speed);
        } else {
            let text = arg("-i").expect("-i text, -p phonemes or --g2p");
            let mut last = Instant::now();
            for (i, chunk) in tts.speak(&text, &voice, speed).enumerate() {
                let secs = chunk.len() as f64 / SAMPLE_RATE as f64;
                let ms = last.elapsed().as_secs_f64() * 1e3;
                eprintln!("chunk\t{i}\t{ms:.1}\taudio {secs:.2}\tsince start {:.1}", t.elapsed().as_secs_f64() * 1e3);
                audio.extend(chunk);
                last = Instant::now();
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let secs = audio.len() as f64 / SAMPLE_RATE as f64;
        eprintln!("synth\t{ms:.1}\taudio {secs:.2}\trtf {:.3}", ms / 1e3 / secs);
    }
    if let Some(o) = arg("-o") {
        wav::write(&o, &audio, SAMPLE_RATE as u32);
    }
}
