//! Where model files live and where missing ones come from.

use std::io;
use std::path::PathBuf;

/// moondream/parakeet-ultra at a pinned revision: model.safetensors (f16, 1.2 GB) and tokenizer.json.
pub const ASR_URL: &str =
    "https://huggingface.co/moondream/parakeet-ultra/resolve/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b";
/// kokoallovero's assets at a pinned revision: kokoro.bin/.tsv (f32, weight norm folded), vocab.tsv, g2p/*.bin.
pub const TTS_URL: &str = "https://huggingface.co/rodbiren/kokoro_parakeet_aio_models/resolve/766eb98b496915a437f38b1cec8e9458879cb2f3/kokoro";

/// A directory of model files plus the base URL missing files are fetched from (`url/name`).
#[derive(Clone, Debug)]
pub struct Weights {
    pub dir: PathBuf,
    pub url: String,
}

/// $XDG_DATA_HOME/hearandspeak, else ~/.local/share/hearandspeak.
pub fn data_dir() -> PathBuf {
    match std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        Some(d) => PathBuf::from(d).join("hearandspeak"),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share/hearandspeak"),
    }
}

impl Weights {
    pub fn asr() -> Weights {
        Weights { dir: data_dir().join("parakeet-ultra"), url: ASR_URL.into() }
    }

    pub fn tts() -> Weights {
        Weights { dir: data_dir().join("kokoro"), url: TTS_URL.into() }
    }

    /// Path of `name` under dir, downloaded first if it is missing.
    pub fn file(&self, name: &str) -> io::Result<PathBuf> {
        let path = self.dir.join(name);
        if !path.exists() {
            fetch(&format!("{}/{name}", self.url.trim_end_matches('/')), &path)?;
        }
        Ok(path)
    }
}

#[cfg(feature = "download")]
fn fetch(url: &str, path: &std::path::Path) -> io::Result<()> {
    let part = path.with_extension("part");
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    eprintln!("hearandspeak: fetching {url}");
    let resp = ureq::get(url).call().map_err(|e| io::Error::other(format!("{url}: {e}")))?;
    let mut body = resp.into_body();
    let mut reader = body.as_reader();
    let mut f = std::fs::File::create(&part)?;
    io::copy(&mut reader, &mut f)?;
    f.sync_all()?;
    std::fs::rename(&part, path)
}

#[cfg(not(feature = "download"))]
fn fetch(url: &str, path: &std::path::Path) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::NotFound, format!("{} missing; fetch it from {url}", path.display())))
}
