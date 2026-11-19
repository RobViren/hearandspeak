//! safetensors, read tensor by tensor with positioned reads so the whole file is never resident.

use crate::json::{Json, quote};
use crate::simd::f16;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

pub struct SafeTensors {
    file: File,
    base: u64,
    pub t: HashMap<String, Tensor>,
}

pub fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl SafeTensors {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        let mut n = [0u8; 8];
        file.read_exact_at(&mut n, 0)?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        file.read_exact_at(&mut h, 8)?;
        let header = Json::parse(std::str::from_utf8(&h).map_err(|_| bad("safetensors header"))?)
            .ok_or_else(|| bad("safetensors header"))?;
        let mut t = HashMap::new();
        for (name, v) in header.obj() {
            if name == "__metadata__" {
                continue;
            }
            let off = v.get("data_offsets").map(|o| o.arr()).unwrap_or(&[]);
            t.insert(
                name.clone(),
                Tensor {
                    dtype: v.get("dtype").map_or("", |d| d.str()).to_string(),
                    shape: v
                        .get("shape")
                        .map_or(&[][..], |s| s.arr())
                        .iter()
                        .map(|x| x.num() as usize)
                        .collect(),
                    start: off.first().map_or(0, |x| x.num() as u64),
                    end: off.get(1).map_or(0, |x| x.num() as u64),
                },
            );
        }
        Ok(SafeTensors {
            file,
            base: 8 + n,
            t,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.t.contains_key(name)
    }

    pub fn get(&self, name: &str) -> io::Result<&Tensor> {
        self.t
            .get(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))
    }

    pub fn shape(&self, name: &str) -> io::Result<&[usize]> {
        Ok(&self.get(name)?.shape)
    }

    /// Raw bytes into dst, which must be exactly the tensor's size.
    pub fn read_into(&self, name: &str, dst: &mut [u8]) -> io::Result<()> {
        let t = self.get(name)?;
        if (t.end - t.start) as usize != dst.len() {
            return Err(bad(format!(
                "{name}: {} bytes, expected {}",
                t.end - t.start,
                dst.len()
            )));
        }
        self.file.read_exact_at(dst, self.base + t.start)
    }

    pub fn bytes(&self, name: &str) -> io::Result<Vec<u8>> {
        let t = self.get(name)?;
        let mut v = vec![0u8; (t.end - t.start) as usize];
        self.read_into(name, &mut v)?;
        Ok(v)
    }

    pub fn f16(&self, name: &str) -> io::Result<Vec<u16>> {
        if self.get(name)?.dtype != "F16" {
            return Err(bad(format!("{name} is not F16")));
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// F16 or F32 widened to f32.
    pub fn f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let b = self.bytes(name)?;
        match self.get(name)?.dtype.as_str() {
            "F32" => Ok(b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "F16" => Ok(b
                .chunks_exact(2)
                .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            d => Err(bad(format!("{name}: dtype {d}"))),
        }
    }
}

pub struct Entry<'a> {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

pub fn write(path: &Path, entries: &[Entry]) -> io::Result<()> {
    let mut header = String::from("{");
    let mut off = 0;
    for (i, e) in entries.iter().enumerate() {
        let shape: Vec<String> = e.shape.iter().map(|d| d.to_string()).collect();
        header += &format!(
            "{}{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            if i > 0 { "," } else { "" },
            quote(&e.name),
            e.dtype,
            shape.join(","),
            off,
            off + e.data.len()
        );
        off += e.data.len();
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut f = io::BufWriter::new(File::create(path)?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for e in entries {
        f.write_all(e.data)?;
    }
    f.flush()
}
