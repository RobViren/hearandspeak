//! Real models on hostile inputs: nothing may panic or hang, and output stays finite.
//! Needs the weights, so ignored by default: cargo test --release --test gross -- --ignored

use hearandspeak::asr::{Asr, StreamOpts};
use hearandspeak::tts::{Tts, Voice};
use hearandspeak::{Pool, Weights};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn pool_survives_panics() {
    let pool = Pool::new(4);
    for bad in [0, 3] {
        let r = catch_unwind(AssertUnwindSafe(|| pool.run(|i| assert!(i != bad))));
        assert!(r.is_err());
    }
    let hits = std::sync::atomic::AtomicUsize::new(0);
    pool.run(|_| {
        hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(hits.into_inner(), pool.n);
}

#[test]
#[ignore]
fn asr_gross() {
    let mut asr = Asr::new(&Weights::asr(), Pool::new(0)).unwrap();
    let noise: Vec<f32> =
        (0..48000u32).map(|i| if (i.wrapping_mul(2654435761) >> 16) & 1 == 0 { 1.0 } else { -1.0 }).collect();
    let inputs: Vec<(&str, Vec<f32>)> = vec![
        ("empty", vec![]),
        ("one", vec![0.5]),
        ("zeros", vec![0.0; 32000]),
        ("nan", vec![f32::NAN; 32000]),
        ("inf", vec![f32::INFINITY; 32000]),
        ("huge", vec![f32::MAX; 32000]),
        ("mixed", (0..32000).map(|i| [f32::NAN, 1e30, -f32::INFINITY, 0.3][i % 4]).collect()),
        ("silence", vec![0.0; 16000 * 70]),
        ("clipped", noise),
    ];
    for (name, pcm) in &inputs {
        for rate in [16000, 0, 1, 8000, 44100, 1_000_000, u32::MAX] {
            let text = asr.transcribe(pcm, rate);
            eprintln!("transcribe\t{name}\t{rate}\t{text:?}");
            let opts = [
                StreamOpts::default(),
                StreamOpts { pause: f32::NAN, max_seconds: f32::INFINITY, check_every: -1.0 },
                StreamOpts { pause: 0.0, max_seconds: 0.0, check_every: 0.0 },
                StreamOpts { pause: f32::MAX, max_seconds: 1e-9, check_every: f32::MAX },
            ];
            for o in opts {
                let mut s = asr.stream_with(rate, o);
                let mut segs = Vec::new();
                for c in pcm.chunks(3333) {
                    segs.extend(s.push(c));
                }
                s.partial();
                segs.extend(s.finish());
                assert!(segs.iter().all(|g| g.start.is_finite() && g.end >= g.start), "{name} {rate} {o:?}");
            }
        }
    }
}

#[test]
#[ignore]
fn tts_gross() {
    let tts = Tts::new(&Weights::tts(), Pool::new(0)).unwrap();
    let voice = Voice::builtin("af_heart").unwrap();
    assert!(Voice::from_bytes(&[0; 10]).is_err());
    assert!(Voice::from_bytes(&[0xff; 510 * 256 * 4]).is_err());
    let giant = "a".repeat(3000);
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let texts = [
        "",
        "   \n\t ",
        "?!...;;--",
        giant.as_str(),
        long.as_str(),
        "\u{0}\u{1b}[31m\u{202e}",
        "\u{1F600}\u{1F468}\u{200D}\u{1F469}",
    ];
    for t in texts {
        for speed in [1.0, 0.0, -1.0, f32::NAN, f32::INFINITY, 1e9] {
            let audio: Vec<f32> = tts.speak(t, &voice, speed).flatten().collect();
            eprintln!("speak\t{}\t{speed}\t{}", t.len(), audio.len());
            assert!(audio.iter().all(|v| v.is_finite()));
        }
    }
    let ps = "ðə kwˈɪk bɹˈWn fˈɑks ".repeat(60);
    for p in ["", "XYZ123", "\u{0}", ps.as_str(), "ə"] {
        let audio = tts.synthesize(p, &voice, 1.0);
        eprintln!("synthesize\t{}\t{}", p.chars().count(), audio.len());
        assert!(audio.iter().all(|v| v.is_finite()));
    }
}

#[cfg(feature = "asr")]
#[test]
fn check_rate_rejects_low_rates() {
    use hearandspeak::asr::{Asr, MIN_RATE};
    assert!(Asr::check_rate(0).is_err());
    assert!(Asr::check_rate(MIN_RATE - 1).is_err());
    assert!(Asr::check_rate(MIN_RATE).is_ok());
    assert!(Asr::check_rate(48000).is_ok());
}
