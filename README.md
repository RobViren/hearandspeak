# hearandspeak

Speech recognition and text to speech on the CPU, in pure Rust. No ONNX, no Python, no GPU.

- **Hear**: streaming speech to text with Parakeet TDT 0.6B v3 (int8 encoder). Feed it audio chunks as they arrive and it hands back finished segments when the speaker pauses.
- **Speak**: text to speech with Kokoro-82M plus an English G2P (lexicon, POS tagger, and a fallback guesser for unknown words). Streams audio sentence by sentence.

Requires x86-64 with AVX2, FMA and F16C. Uses AVX-VNNI for the int8 matmuls when the CPU has it.

## How it works

Both models are reimplemented directly: the Conformer encoder and TDT decoder for ASR, and Kokoro's text encoder, duration predictor and iSTFTNet decoder for TTS. Hot kernels are hand written SIMD (`#[target_feature]` functions) running on a small pool of threads pinned to the fast cores. One `Pool` serves both engines.

Weights are not bundled. On first use they download from [rodbiren/kokoro_parakeet_aio_models](https://huggingface.co/rodbiren/kokoro_parakeet_aio_models) into `$XDG_DATA_HOME/hearandspeak` (or `~/.local/share/hearandspeak`). Turn off the `download` feature to skip the HTTP client and load files yourself with `from_files`. Three voices (`af_heart`, `am_michael`, `bf_emma`) are built into the crate.

## Use

```toml
hearandspeak = "0.1"
```

```rust
use hearandspeak::{Pool, Weights, asr::Asr, tts::{Tts, Voice}};

let pool = Pool::new(0); // 0 = pick the fast cores

// Speak
let tts = Tts::new(&Weights::tts(), pool.clone())?;
let voice = Voice::builtin("af_heart").unwrap();
for chunk in tts.speak("Hello there. This is a test.", &voice, 1.0) {
    // chunk: Vec<f32> mono PCM at 24 kHz
}

// Hear, all at once
let mut asr = Asr::new(&Weights::asr(), pool.clone())?;
let text = asr.transcribe(&pcm, 16000);

// Hear, streaming
let mut stream = asr.stream(48000);
for segment in stream.push(&chunk) {
    println!("{}", segment.text);
}
```

A stream can be parked and resumed so one engine serves several conversations or turns.

## Features

| feature    | default | what                                   |
|------------|---------|----------------------------------------|
| `asr`      | yes     | speech to text                         |
| `tts`      | yes     | text to speech                         |
| `download` | yes     | fetch weights from Hugging Face on use |

## Examples

```sh
cargo run --release --example tts -- -i "Text to say" -o out.wav
cargo run --release --example asr -- input.wav            # whole file
cargo run --release --example asr -- --chunk 100 input.wav # streamed in 100 ms chunks
```

## License

Code is Apache-2.0. Model weights carry their own licenses: Kokoro-82M is Apache-2.0, Parakeet TDT 0.6B v3 is CC-BY-4.0 (NVIDIA).
