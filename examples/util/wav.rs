//! Minimal wav io for the examples: PCM16 or float32, mono or mixed down.

#[allow(dead_code)]
pub fn read(path: &str) -> (Vec<f32>, u32) {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let u16_at = |i: usize| u16::from_le_bytes([d[i], d[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
    let (mut fmt, mut channels, mut rate, mut bits) = (0, 1, 0, 0);
    let mut at = 12;
    while at + 8 <= d.len() {
        let size = (u32_at(at + 4) as usize).min(d.len() - at - 8);
        let body = at + 8;
        match &d[at..at + 4] {
            b"fmt " => {
                fmt = u16_at(body);
                channels = u16_at(body + 2) as usize;
                rate = u32_at(body + 4);
                bits = u16_at(body + 14);
                if fmt == 0xfffe {
                    fmt = u16_at(body + 24);
                }
            }
            b"data" => {
                let raw = &d[body..body + size];
                let s: Vec<f32> = match (fmt, bits) {
                    (1, 16) => raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
                    (3, 32) => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                    _ => panic!("{path}: unsupported wav format {fmt}/{bits}"),
                };
                let mono = s.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
                return (mono, rate);
            }
            _ => {}
        }
        at = body + size + (size & 1);
    }
    panic!("{path}: no data chunk")
}

#[allow(dead_code)]
pub fn write(path: &str, pcm: &[f32], rate: u32) {
    let mut o = Vec::with_capacity(44 + pcm.len() * 2);
    let data = (pcm.len() * 2) as u32;
    o.extend_from_slice(b"RIFF");
    o.extend_from_slice(&(36 + data).to_le_bytes());
    o.extend_from_slice(b"WAVEfmt ");
    for v in [16u32.to_le_bytes(), [1, 0, 1, 0]] {
        o.extend_from_slice(&v);
    }
    o.extend_from_slice(&rate.to_le_bytes());
    o.extend_from_slice(&(rate * 2).to_le_bytes());
    o.extend_from_slice(&[2, 0, 16, 0]);
    o.extend_from_slice(b"data");
    o.extend_from_slice(&data.to_le_bytes());
    for &x in pcm {
        o.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, o).unwrap();
}
