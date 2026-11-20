#![no_main]

use hearandspeak::tts::{Tts, Voice};
use hearandspeak::{Pool, Weights};
use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

static TTS: LazyLock<Tts> = LazyLock::new(|| Tts::new(&Weights::tts(), Pool::new(4)).unwrap());

fuzz_target!(|x: (&str, f32, bool)| {
    let (text, speed, as_phonemes) = x;
    let voice = Voice::builtin("af_heart").unwrap();
    let audio: Vec<f32> = if as_phonemes {
        TTS.synthesize(text, &voice, speed)
    } else {
        TTS.speak(text, &voice, speed).flatten().collect()
    };
    assert!(audio.iter().all(|v| v.is_finite()), "non-finite audio");
});
