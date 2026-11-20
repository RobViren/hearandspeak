#![no_main]

use hearandspeak::asr::{Asr, StreamOpts};
use hearandspeak::{Pool, Weights};
use libfuzzer_sys::fuzz_target;
use std::sync::{LazyLock, Mutex};

static ASR: LazyLock<Mutex<Asr>> = LazyLock::new(|| Mutex::new(Asr::new(&Weights::asr(), Pool::new(4)).unwrap()));

const RATES: [u32; 12] = [16000, 0, 1, 7, 8000, 11025, 22050, 44100, 48000, 96001, 1_000_000, u32::MAX];

/// A block of `len` samples of one kind, so a few bytes describe seconds of hostile audio.
fn block(kind: u8, len: u16, seed: &mut u32, out: &mut Vec<f32>) {
    let mut noise = || {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 8) as f32 / (1 << 23) as f32 - 1.0
    };
    out.extend((0..len as usize).map(|i| match kind % 10 {
        0 => 0.0,
        1 => (i as f32 * (len % 997 + 1) as f32 * 0.01).sin(),
        2 => noise(),
        3 => f32::NAN,
        4 => [f32::INFINITY, f32::NEG_INFINITY][i % 2],
        5 => 1e30 * noise(),
        6 => {
            if noise() > 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        7 => 1e-40,
        8 => f32::MAX,
        _ => noise() * 32768.0,
    }));
}

fuzz_target!(|x: (u8, f32, f32, f32, u8, Vec<(u8, u16)>)| {
    let (rate_pick, pause, max_seconds, check_every, mode, blocks) = x;
    let rate = RATES[rate_pick as usize % RATES.len()];
    let (mut seed, mut pcm, mut ends) = (1, Vec::new(), Vec::new());
    for (kind, len) in blocks {
        if pcm.len() > 16000 * 40 {
            break;
        }
        block(kind, len, &mut seed, &mut pcm);
        ends.push(pcm.len());
    }
    let mut asr = ASR.lock().unwrap_or_else(|e| e.into_inner());
    if mode & 1 == 1 {
        asr.transcribe(&pcm, rate);
        return;
    }
    let mut s = asr.stream_with(rate, StreamOpts { pause, max_seconds, check_every });
    let mut at = 0;
    for end in ends {
        s.push(&pcm[at..end]);
        at = end;
        if mode & 2 == 2 {
            s.partial();
        }
    }
    if mode & 4 == 4 {
        let p = s.park();
        s = asr.resume(p);
    }
    s.finish();
});
