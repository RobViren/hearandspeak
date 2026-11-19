//! Writes parakeet-ultra with its encoder linears prepacked as int8 (half the size, no load-time quantization,
//! identical transcripts). cargo run --release --example convert_parakeet -- model.safetensors out.safetensors

fn main() {
    let a: Vec<String> = std::env::args().collect();
    hearandspeak::asr::convert(a[1].as_ref(), a[2].as_ref()).unwrap();
}
