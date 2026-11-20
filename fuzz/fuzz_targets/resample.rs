#![no_main]

use hearandspeak::resample::{Resampler, resample};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|x: (u32, u32, Vec<u16>, Vec<f32>)| {
    let (from, to, cuts, pcm) = x;
    let whole = resample(&pcm, from, to);
    let mut r = Resampler::new(from, to);
    let mut out = Vec::new();
    let mut rest = &pcm[..];
    for c in cuts {
        let (a, b) = rest.split_at((c as usize).min(rest.len()));
        r.push(a, &mut out);
        rest = b;
    }
    r.push(rest, &mut out);
    r.finish(&mut out);
    if from == to {
        return;
    }
    assert_eq!(out.len(), whole.len());
    if from > 0 && to > 0 && (to as u64) <= 16000 * from as u64 {
        let want = (pcm.len() as u64 * to as u64).div_ceil(from as u64) as f64;
        assert!((out.len() as f64 - want).abs() <= 1.0 + want * 1e-4, "{} outputs, want {want}", out.len());
    }
});
