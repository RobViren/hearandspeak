//! std::mt19937 and the libstdc++ (GCC 16) float uniform and normal distributions, so the vocoder source gets the
//! same noise as the reference.

pub struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }

    pub fn next(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 == 1 { 0x9908_b0df } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// generate_canonical<float, 24>: one draw, redrawn if it rounds up to 1.
    pub fn uniform(&mut self) -> f32 {
        loop {
            let r = self.next() as f32 / 4294967296.0;
            if r < 1.0 {
                return r;
            }
        }
    }
}

/// Marsaglia polar method, keeping the second value of each pair.
#[derive(Default)]
pub struct Normal {
    saved: Option<f32>,
}

impl Normal {
    pub fn sample(&mut self, rng: &mut Mt19937) -> f32 {
        if let Some(s) = self.saved.take() {
            return s;
        }
        loop {
            let x = ((2.0f32 * rng.uniform()) as f64 - 1.0) as f32;
            let y = ((2.0f32 * rng.uniform()) as f64 - 1.0) as f32;
            let r2 = x.mul_add(x, y * y);
            if r2 as f64 > 1.0 || r2 == 0.0 {
                continue;
            }
            let mult = (-2.0 * r2.ln() / r2).sqrt();
            self.saved = Some(x * mult);
            return y * mult;
        }
    }
}
