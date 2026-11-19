//! Transcribes wav files, one line each. Timing per file and per stage on stderr.
//! cargo run --release --example asr -- [-t threads] [-m model.safetensors -k tokenizer.json] [--chunk ms] files...

#[path = "util/wav.rs"]
mod wav;

use hearandspeak::{Pool, Weights, asr::Asr};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut threads, mut model, mut tok, mut chunk_ms, mut partial_s, mut s_pause, mut files) =
        (0, None, None, 0usize, 0usize, 0.6f64, Vec::new());
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-t" => (threads, i) = (args[i + 1].parse().unwrap(), i + 1),
            "-m" => (model, i) = (Some(args[i + 1].clone()), i + 1),
            "-k" => (tok, i) = (Some(args[i + 1].clone()), i + 1),
            "--chunk" => (chunk_ms, i) = (args[i + 1].parse().unwrap(), i + 1),
            "--pause" => (s_pause, i) = (args[i + 1].parse().unwrap(), i + 1),
            "--partial" => (partial_s, i) = (args[i + 1].parse().unwrap(), i + 1),
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
            let opts = hearandspeak::asr::StreamOpts { pause: s_pause as f32, ..Default::default() };
            let mut s = asr.stream_with(rate, opts);
            let mut parts = Vec::new();
            let per = rate as usize * chunk_ms / 1000;
            let pause = s_pause;
            for (k, c) in pcm.chunks(per).enumerate() {
                let tp = Instant::now();
                let finals = s.push(c);
                let wall = tp.elapsed().as_secs_f64();
                let at = ((k * per + c.len()) as f64) / rate as f64;
                for f in finals {
                    // Pause cuts land pause / 2 after the end of speech.
                    eprintln!(
                        "final\t{:.2}\t{:.2}\tlatency {:.3}\t{}",
                        f.start,
                        f.end,
                        at - (f.end - pause / 2.0) + wall,
                        f.text
                    );
                    parts.push(f.text);
                }
                if partial_s > 0 && (k + 1) * chunk_ms % (partial_s * 1000) == 0 {
                    let tp = Instant::now();
                    let text = s.partial();
                    eprintln!("partial\t{at:.1}\t{:.3}\t{}", tp.elapsed().as_secs_f64(), text.len());
                }
            }
            let tf = Instant::now();
            parts.extend(s.finish().into_iter().map(|f| f.text));
            eprintln!("finish\t{:.3}", tf.elapsed().as_secs_f64());
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
