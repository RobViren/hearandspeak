#[path = "util/wav.rs"]
mod wav;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (pcm, _) = wav::read(&a[1]);
    let pool = hearandspeak::Pool::new(1);
    println!("frontend_ms {:.3}", hearandspeak::asr::bench_frontend(&pool, &pcm, 500));
}
