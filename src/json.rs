//! Just enough JSON for safetensors headers and tokenizer.json.

pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(s: &str) -> Option<Json> {
        let mut p = Parser { b: s.as_bytes(), i: 0 };
        p.value()
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn obj(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(v) => v,
            _ => &[],
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }
    pub fn num(&self) -> f64 {
        if let Json::Num(n) = self { *n } else { 0.0 }
    }
    pub fn str(&self) -> &str {
        if let Json::Str(s) = self { s } else { "" }
    }
    pub fn bool(&self) -> bool {
        matches!(self, Json::Bool(true))
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Option<Json> {
        self.ws();
        let c = *self.b.get(self.i)?;
        Some(match c {
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b.get(self.i)? == &b'}' {
                        self.i += 1;
                        break;
                    }
                    let k = self.string()?;
                    self.ws();
                    self.i += 1;
                    v.push((k, self.value()?));
                    self.ws();
                    if self.b.get(self.i)? == &b',' {
                        self.i += 1;
                    }
                }
                Json::Obj(v)
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b.get(self.i)? == &b']' {
                        self.i += 1;
                        break;
                    }
                    v.push(self.value()?);
                    self.ws();
                    if self.b.get(self.i)? == &b',' {
                        self.i += 1;
                    }
                }
                Json::Arr(v)
            }
            b'"' => Json::Str(self.string()?),
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    self.i += 1;
                }
                Json::Num(std::str::from_utf8(&self.b[s..self.i]).ok()?.parse().ok()?)
            }
        })
    }
    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }
    fn string(&mut self) -> Option<String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.b.get(self.i) == Some(&b'\\') {
                                self.i += 2;
                                u = 0x10000 + ((u - 0xd800) << 10) + (self.hex4()? - 0xdc00);
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        c => c as char,
                    };
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).ok()
    }
}

pub fn quote(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
