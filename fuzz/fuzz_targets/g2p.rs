#![no_main]

use hearandspeak::tts::Tts;
use hearandspeak::{Pool, Weights};
use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

static TTS: LazyLock<Tts> = LazyLock::new(|| Tts::new(&Weights::tts(), Pool::new(1)).unwrap());

fuzz_target!(|text: &str| {
    TTS.normalize(text);
    for c in TTS.phonemize(text) {
        let n = c.chars().count();
        assert!(n > 0 && n <= 510, "chunk of {n} phonemes");
    }
});
