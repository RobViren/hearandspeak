//! Where model files live and where missing ones come from.

use std::io;
use std::path::PathBuf;

/// Rob's model repo. parakeet-ultra/ holds parakeet-ultra with its encoder linears prepacked as int8 (written by
/// examples/convert_parakeet.rs) and tokenizer.json; kokoro/ holds kokoallovero's assets as they are; hearandspeak/
/// holds files this crate bakes itself (the lexicon from ipa_everywhere). tools/publish_weights.sh uploads them.
pub const REPO_URL: &str = "https://huggingface.co/rodbiren/kokoro_parakeet_aio_models/resolve/main";

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
        Weights { dir: data_dir(), url: REPO_URL.into() }
    }

    pub fn tts() -> Weights {
        Weights { dir: data_dir(), url: REPO_URL.into() }
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
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    let part = path.with_extension(format!("part.{}.{nanos}", std::process::id()));
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
