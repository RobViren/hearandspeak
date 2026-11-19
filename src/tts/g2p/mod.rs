//! kokoallovero's English G2P (a port of ag2p's Slime speaking path): normalize, spaCy-style tokenize, tag, look up
//! the lexicon with part-of-speech variants and suffix rules, guess unknown words, map to Kokoro phonemes, chunk.
//! Regexes keep PCRE2's semantics (Unicode \w \b \d, lookaround, atomic groups) through fancy-regex.

mod guesser;
mod tagger;

use fancy_regex::{Captures, Regex};
use guesser::Guesser;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::LazyLock;
use tagger::Tagger;

struct Re(Regex);

impl Re {
    fn new(p: &str) -> Re {
        Re(Regex::new(p).unwrap_or_else(|e| panic!("regex {e}: {}", &p[..p.len().min(80)])))
    }
    fn full(p: &str) -> Re {
        Re::new(&format!(r"\A(?:{p})\z"))
    }
    fn find<'t>(&self, s: &'t str, from: usize) -> Option<Captures<'t>> {
        self.0.captures_from_pos(s, from).ok().flatten()
    }
    fn search(&self, s: &str) -> bool {
        self.find(s, 0).is_some()
    }
}

fn b(m: &Captures) -> usize {
    m.get(0).unwrap().start()
}
fn e(m: &Captures) -> usize {
    m.get(0).unwrap().end()
}
fn g<'t>(m: &Captures<'t>, i: usize) -> &'t str {
    m.get(i).map_or("", |x| x.as_str())
}
fn set(m: &Captures, i: usize) -> bool {
    m.get(i).is_some()
}

fn char_len(s: &str, at: usize) -> usize {
    s[at..].chars().next().map_or(1, |c| c.len_utf8())
}

/// re.sub with a function; an empty match copies one character and moves on.
fn sub(r: &Re, s: &str, mut f: impl FnMut(&Captures) -> String) -> String {
    let mut o = String::new();
    let mut last = 0;
    while last <= s.len() {
        let Some(m) = r.find(s, last) else { break };
        o.push_str(&s[last..b(&m)]);
        o += &f(&m);
        last = e(&m);
        if e(&m) == b(&m) {
            if last >= s.len() {
                break;
            }
            let k = char_len(s, last);
            o.push_str(&s[last..last + k]);
            last += k;
        }
    }
    if last < s.len() {
        o.push_str(&s[last..]);
    }
    o
}

/// re.sub with a template where \1..\9 are groups.
fn sub_t(r: &Re, s: &str, t: &str) -> String {
    sub(r, s, |m| {
        let mut o = String::new();
        let tb = t.as_bytes();
        let mut i = 0;
        while i < tb.len() {
            if tb[i] == b'\\' && i + 1 < tb.len() && (b'1'..=b'9').contains(&tb[i + 1]) {
                o += g(m, (tb[i + 1] - b'0') as usize);
                i += 2;
            } else {
                let k = char_len(t, i);
                o.push_str(&t[i..i + k]);
                i += k;
            }
        }
        o
    })
}

fn find_iter(r: &Re, s: &str, mut f: impl FnMut(&Captures)) {
    let mut p = 0;
    while p <= s.len() {
        let Some(m) = r.find(s, p) else { break };
        f(&m);
        p = if e(&m) > b(&m) { e(&m) } else { e(&m) + if e(&m) < s.len() { char_len(s, e(&m)) } else { 1 } };
    }
}

fn is_space(c: char) -> bool {
    let c = c as u32;
    (9..=13).contains(&c)
        || (0x1C..=0x20).contains(&c)
        || c == 0x85
        || c == 0xA0
        || c == 0x1680
        || (0x2000..=0x200A).contains(&c)
        || c == 0x2028
        || c == 0x2029
        || c == 0x202F
        || c == 0x205F
        || c == 0x3000
}

fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

fn strip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_matches(|c| chars.contains(c))
}

fn lower(s: &str) -> String {
    s.chars()
        .map(|c| match c as u32 {
            0x41..=0x5A => (c as u8 + 32) as char,
            0xC0..=0xDE if c as u32 != 0xD7 => char::from_u32(c as u32 + 32).unwrap(),
            0x152 => '\u{153}',
            _ => c,
        })
        .collect()
}

fn upper(s: &str) -> String {
    s.chars().map(|c| c.to_ascii_uppercase()).collect()
}

fn is_upper(s: &str) -> bool {
    !s.bytes().any(|c| c.is_ascii_lowercase()) && s.bytes().any(|c| c.is_ascii_uppercase())
}

fn drop_n(w: &str, n: usize) -> &str {
    &w[..w.len() - n]
}

fn last_char(s: &str) -> Option<char> {
    s.chars().next_back()
}

#[rustfmt::skip]
const ONES: [&str; 20] = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// Python ints are unbounded and every scale is a power of ten, so divmod is a split of the digit string.
fn int_words(s: &str) -> String {
    let s = s.trim_start_matches('0');
    if s.is_empty() {
        return "zero".into();
    }
    if s.len() <= 2 {
        let n: usize = s.parse().unwrap();
        if n < 20 {
            return ONES[n].into();
        }
        return format!("{}{}", TENS[n / 10], if n % 10 > 0 { format!(" {}", ONES[n % 10]) } else { String::new() });
    }
    let rest = |r: &str| {
        let r = r.trim_start_matches('0');
        if r.is_empty() { String::new() } else { format!(" {}", int_words(r)) }
    };
    if s.len() == 3 {
        return format!("{} hundred{}", ONES[(s.as_bytes()[0] - b'0') as usize], rest(&s[1..]));
    }
    for (k, name) in [(12, "trillion"), (9, "billion"), (6, "million"), (3, "thousand")] {
        if s.len() > k {
            return format!("{} {name}{}", int_words(&s[..s.len() - k]), rest(&s[s.len() - k..]));
        }
    }
    String::new()
}

fn int_words_n(n: i64) -> String {
    int_words(&n.to_string())
}

fn digit_words(s: &str) -> Vec<String> {
    s.bytes()
        .filter(u8::is_ascii_digit)
        .map(|c| if c == b'0' { "zero".into() } else { ONES[(c - b'0') as usize].into() })
        .collect()
}

fn ordinal_words(s: &str) -> String {
    let w = int_words(s);
    let tail = w.rsplit(' ').next().unwrap_or(&w).to_string();
    #[rustfmt::skip]
    let ord = [
        ("one", "first"), ("two", "second"), ("three", "third"), ("five", "fifth"), ("eight", "eighth"),
        ("nine", "ninth"), ("twelve", "twelfth"),
    ];
    let rep =
        ord.iter().find(|(a, _)| *a == tail).map(|(_, b)| b.to_string()).unwrap_or_else(|| {
            if let Some(t) = tail.strip_suffix('y') { format!("{t}ieth") } else { format!("{tail}th") }
        });
    format!("{}{rep}", &w[..w.len() - tail.len()])
}

fn year_words(n: i64) -> String {
    let (hi, lo) = (n / 100, n % 100);
    if lo == 0 {
        return format!("{} hundred", int_words_n(hi));
    }
    if lo < 10 {
        return format!("{} oh {}", int_words_n(hi), ONES[lo as usize]);
    }
    format!("{} {}", int_words_n(hi), int_words_n(lo))
}

type Opt = Option<String>;

const SIBILANT: [&str; 6] = ["tʃ", "dʒ", "s", "z", "ʃ", "ʒ"];

fn ends_any(s: &str, xs: &[&str]) -> bool {
    xs.iter().any(|x| s.ends_with(x))
}

fn plural_suffix(b: &str) -> String {
    if ends_any(b, &SIBILANT) {
        return format!("{b}ɪz");
    }
    format!("{b}{}", if ends_any(b, &["p", "t", "k", "f", "θ"]) { "s" } else { "z" })
}

fn join_compound(ipas: &[String], head: usize) -> String {
    ipas.iter().enumerate().map(|(k, i)| if k == head { i.clone() } else { i.replace('ˈ', "ˌ") }).collect()
}

/// A rule lookup: the lexicon reading of a word under the current tag, if the rules may use it.
type Lookup<'a> = dyn Fn(&str) -> Opt + 'a;

fn parts<'w>(word: &'w str, l: &Lookup, count: usize) -> Option<(Vec<&'w str>, Vec<String>)> {
    if !word.contains('-') {
        return None;
    }
    let p: Vec<&str> = word.split('-').collect();
    if p.iter().any(|x| x.is_empty()) || (count > 0 && p.len() != count) {
        return None;
    }
    let ipas = p.iter().map(|x| l(x)).collect::<Option<Vec<_>>>()?;
    Some((p, ipas))
}

fn compound_numeral(w: &str, l: &Lookup) -> Opt {
    let (p, i) = parts(w, l, 0)?;
    let ok = |x: &str| {
        ["and", "hundred", "thousand", "million", "billion"].contains(&x)
            || ONES[1..].contains(&x)
            || TENS[2..].contains(&x)
    };
    p.iter().all(|x| ok(x)).then(|| join_compound(&i, i.len() - 1))
}

fn participial_compound(w: &str, l: &Lookup) -> Opt {
    let (p, i) = parts(w, l, 2)?;
    (p[1].ends_with("ing") || p[1].ends_with("ed")).then(|| join_compound(&i, 1))
}

fn prefixed_compound(w: &str, l: &Lookup) -> Opt {
    const PRE: [&str; 26] = [
        "anti", "non", "self", "post", "pre", "co", "auto", "bio", "neuro", "multi", "micro", "sub", "over", "under",
        "intra", "hypo", "hyper", "ultra", "pseudo", "immuno", "cardio", "radio", "electro", "photo", "thermo", "semi",
    ];
    let (p, i) = parts(w, l, 2)?;
    PRE.contains(&p[0]).then(|| join_compound(&i, 1))
}

fn nchars(s: &str) -> usize {
    s.chars().count()
}

fn suffix_inflection(w: &str, l: &Lookup) -> Opt {
    let base = if w.ends_with("'s") {
        l(drop_n(w, 2))
    } else if w.ends_with("es") && nchars(w) > 2 {
        let drop_es = if nchars(w) >= 5 { l(drop_n(w, 2)) } else { None };
        match drop_es {
            Some(d) if !d.is_empty() && ends_any(&d, &SIBILANT) => Some(d),
            _ => {
                let b = l(drop_n(w, 1));
                if b.is_none() && drop_n(w, 2).ends_with('i') { l(&format!("{}y", drop_n(w, 3))) } else { b }
            }
        }
    } else if w.ends_with('s') && nchars(w) > 1 {
        l(drop_n(w, 1))
    } else {
        return None;
    };
    base.filter(|b| !b.is_empty()).map(|b| plural_suffix(&b))
}

fn base(w: &str, suffix: &str, l: &Lookup) -> Opt {
    let stem = w.strip_suffix(suffix)?;
    let mut c = vec![stem.to_string()];
    if "eaiou".contains(suffix.chars().next().unwrap()) {
        c.push(format!("{stem}e"));
    }
    if stem.ends_with('i') && nchars(stem) > 1 {
        c.push(format!("{}y", drop_n(stem, 1)));
    }
    let sb = stem.as_bytes();
    let n = sb.len();
    if nchars(stem) > 2 && sb[n - 1] == sb[n - 2] && b"bdgklmnprtvz".contains(&sb[n - 1]) {
        c.push(drop_n(stem, 1).to_string());
    }
    c.iter().find_map(|s| l(s))
}

fn suffix_rule(w: &str, l: &Lookup, suffix: &str, add: impl Fn(&str) -> String) -> Opt {
    base(w, suffix, l).filter(|b| !b.is_empty()).map(|b| add(&b))
}

fn past_ed(w: &str, l: &Lookup) -> Opt {
    let b = base(w, "ed", l)?;
    if ends_any(&b, &["t", "d"]) {
        return Some(format!("{b}ɪd"));
    }
    Some(format!("{b}{}", if ends_any(&b, &["p", "t", "k", "f", "θ", "s", "ʃ", "tʃ"]) { "t" } else { "d" }))
}

fn noun_ness(w: &str, l: &Lookup) -> Opt {
    if ["business", "businesses", "wilderness", "wildernesses"].contains(&w) {
        return None;
    }
    suffix_rule(w, l, "ness", |b| format!("{b}nəs"))
}

fn negative_un(w: &str, l: &Lookup) -> Opt {
    if w.starts_with("un") && nchars(w) >= 4 && !["unit", "units", "union", "unions"].contains(&w) {
        return l(&w[2..]).filter(|b| !b.is_empty()).map(|b| format!("ʌn{b}"));
    }
    None
}

fn british_our(w: &str, l: &Lookup) -> Opt {
    if w != "our" && w.ends_with("our") && nchars(w) > 3 && !["tour", "tours", "amour", "amours"].contains(&w) {
        return l(&format!("{}or", drop_n(w, 3)));
    }
    None
}

fn apply_rules(w: &str, l: &Lookup) -> Opt {
    compound_numeral(w, l)
        .or_else(|| participial_compound(w, l))
        .or_else(|| prefixed_compound(w, l))
        .or_else(|| suffix_inflection(w, l))
        .or_else(|| suffix_rule(w, l, "ly", |b| if b.ends_with('l') { format!("{b}i") } else { format!("{b}li") }))
        .or_else(|| suffix_rule(w, l, "er", |b| format!("{b}ɚ")))
        .or_else(|| suffix_rule(w, l, "est", |b| format!("{b}ɪst")))
        .or_else(|| suffix_rule(w, l, "ing", |b| format!("{b}ɪŋ")))
        .or_else(|| past_ed(w, l))
        .or_else(|| noun_ness(w, l))
        .or_else(|| suffix_rule(w, l, "less", |b| format!("{b}ləs")))
        .or_else(|| negative_un(w, l))
        .or_else(|| british_our(w, l))
}

/// lexicon.bin: sorted string sections over one owned buffer.
struct Strs {
    spans: Vec<(u32, u32)>,
}

pub struct Lexicon {
    data: Vec<u8>,
    keys: Strs,
    vals: Strs,
    skeys: Strs,
    svals: Strs,
    vocab: HashSet<char>,
    nfc_pair: HashMap<(char, char), char>,
    nfc_single: HashMap<char, char>,
    prefix: Re,
    suffix: Re,
    infix: Re,
    url: Re,
    // Tokenizer._special_matcher: special cases containing affixes, as their affix-split token texts.
    matcher: HashMap<String, Vec<(Vec<String>, String)>>,
}

impl Lexicon {
    fn s(&self, span: (u32, u32)) -> &str {
        std::str::from_utf8(&self.data[span.0 as usize..(span.0 + span.1) as usize]).unwrap_or("")
    }
    fn find(&self, keys: &Strs, vals: &Strs, k: &str) -> Option<&str> {
        let i = keys.spans.partition_point(|&sp| self.s(sp).as_bytes() < k.as_bytes());
        (i < keys.spans.len() && self.s(keys.spans[i]) == k).then(|| self.s(vals.spans[i]))
    }
    fn special(&self, k: &str) -> Option<&str> {
        self.find(&self.skeys, &self.svals, k)
    }
    fn row(&self, w: &str) -> Option<Row<'_>> {
        let v = self.find(&self.keys, &self.vals, w)?;
        let f: Vec<&str> = v[2..].split('\x1f').collect();
        Some(Row {
            status: v.as_bytes()[0],
            import: v.as_bytes()[1] == b'1',
            ipa: f[1],
            def: f[2],
            opts: f[3],
            conds: f[4],
        })
    }
}

struct Row<'a> {
    status: u8,
    import: bool,
    ipa: &'a str,
    def: &'a str,
    opts: &'a str,
    conds: &'a str,
}

impl Row<'_> {
    fn cond(&self, c: &str) -> Opt {
        if self.conds.is_empty() || c.is_empty() {
            return None;
        }
        for kv in self.conds.split('\x1e') {
            match kv.find('\x1d') {
                Some(p) if &kv[..p] == c && p + 1 < kv.len() => return Some(kv[p + 1..].to_string()),
                None if kv == c => return Some(kv.to_string()),
                _ => {}
            }
        }
        None
    }
}

const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻᵊ";
const PUNCT: &str = ",.!?;:…";

fn restress(ps: &str) -> String {
    let mut o = String::new();
    let mut pending: Option<char> = None;
    for c in ps.chars() {
        if c == 'ˈ' || c == 'ˌ' {
            pending = if c == 'ˈ' || pending != Some('ˈ') { Some(c) } else { pending };
        } else if let (true, Some(p)) = (VOWELS.contains(c), pending) {
            o.push(p);
            o.push(c);
            pending = None;
        } else {
            o.push(c);
        }
    }
    o
}

struct Unmappable;

fn to_k(l: &Lexicon, ipa: &str, unstressed: bool) -> Result<String, Unmappable> {
    #[rustfmt::skip]
    const SEQ: [(&str, &str); 18] = [
        ("ɝ", "ɜɹ"), ("ɚ", "əɹ"), ("eɪ", "A"), ("aɪ", "I"), ("oʊ", "O"), ("aʊ", "W"), ("ɔɪ", "Y"), ("dʒ", "ʤ"),
        ("tʃ", "ʧ"), ("g", "ɡ"), ("r", "ɹ"), ("ɾ", "T"), ("ʔ", "t"), ("ː", ""), ("e", "A"), ("o", "O"), ("a", "æ"),
        ("iɹ", "ɪɹ"),
    ];
    let mut s = ipa.replace(' ', "");
    for (a, b) in SEQ {
        s = s.replace(a, b);
    }
    s = if unstressed { s.replace('ˈ', "").replace('ˌ', "") } else { restress(&s) };
    if s.chars().any(|c| c != ' ' && !l.vocab.contains(&c)) {
        return Err(Unmappable);
    }
    Ok(s)
}

#[rustfmt::skip]
const EXT: &[(&str, &str)] = &[
    ("py", "pie"), ("md", "MD"), ("txt", "text"), ("jsonl", "JSON L"), ("sh", "SH"), ("db", "DB"), ("tsv", "TSV"),
    ("csv", "CSV"), ("js", "JS"), ("ts", "TS"), ("rs", "RS"), ("wav", "wave"), ("pt", "PT"), ("onnx", "onyx"),
    ("yml", "YAML"), ("yaml", "YAML"), ("toml", "tommel"), ("cfg", "config"), ("so", "SO"), ("m4a", "M4A"),
    ("log", "log"), ("jsx", "JSX"), ("tsx", "TSX"), ("mjs", "MJS"), ("cjs", "CJS"), ("json", "JSON"), ("html", "HTML"),
    ("htm", "HTML"), ("css", "CSS"), ("scss", "SCSS"), ("svg", "SVG"), ("png", "PNG"), ("jpg", "jay peg"),
    ("jpeg", "jay peg"), ("gif", "gif"), ("webp", "web P"), ("pdf", "PDF"), ("mp3", "MP3"), ("mp4", "MP4"),
    ("webm", "web M"), ("opus", "opus"), ("flac", "flack"), ("ogg", "ogg"), ("kt", "KT"), ("kts", "KTS"),
    ("java", "java"), ("go", "go"), ("c", "C"), ("h", "H"), ("cpp", "CPP"), ("hpp", "HPP"), ("cu", "CU"), ("rb", "RB"),
    ("lua", "lua"), ("sql", "sequel"), ("xml", "XML"), ("ini", "INI"), ("env", "env"), ("lock", "lock"), ("zsh", "ZSH"),
    ("bash", "bash"), ("ps1", "PS1"), ("bat", "bat"), ("exe", "exe"), ("dll", "DLL"), ("apk", "APK"), ("zip", "zip"),
    ("gz", "GZ"), ("tar", "tar"), ("gguf", "GGUF"), ("bin", "bin"), ("safetensors", "safe tensors"),
    ("ckpt", "checkpoint"), ("npy", "NPY"), ("npz", "NPZ"), ("ipynb", "notebook"), ("gradle", "gradle"),
    ("vue", "view"), ("svelte", "svelte"), ("swift", "swift"),
];

fn ext(e: &str) -> Option<&'static str> {
    if e.is_empty() {
        return None;
    }
    EXT.iter().find(|(k, _)| *k == e).map(|(_, v)| *v)
}

fn path_words(m: &str) -> String {
    let lead = if m.starts_with("~/") {
        "home slash "
    } else if m.starts_with('/') {
        "slash "
    } else {
        ""
    };
    let mut s = m.trim_start_matches(['~', '/']).to_string();
    while s.ends_with('/') {
        s.pop();
    }
    static EXT_RE: LazyLock<Re> = LazyLock::new(|| Re::new(r"\.([A-Za-z0-9]{1,11})\b"));
    static SLASH: LazyLock<Re> = LazyLock::new(|| Re::new(r"\s*/\s*"));
    let s = sub(&EXT_RE, &s, |e| format!(" dot {}", ext(&lower(g(e, 1))).unwrap_or(g(e, 1))));
    format!(" {lead}{} ", sub_t(&SLASH, &s, " slash ").replace('_', " "))
}

fn markdown(t: &str) -> String {
    static R: LazyLock<[Re; 7]> = LazyLock::new(|| {
        [
            Re::new("(?s)```.*?(```|$)"),
            Re::new(r"!?\[([^\]]+)\]\([^)]*\)"),
            Re::new(r"(?m)^\s{0,3}#{1,6}\s+"),
            Re::new(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+"),
            Re::new(r"(?m)^\s*\|?[\s:|-]+\|[\s:|-]*$"),
            Re::new(r"(\*\*|__)(.+?)\1"),
            Re::new(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])"),
        ]
    });
    let s = sub_t(&R[0], t, " ");
    let s = sub_t(&R[1], &s, "\\1");
    let s = sub_t(&R[2], &s, "");
    let s = sub_t(&R[3], &s, "");
    let s = sub_t(&R[4], &s, "");
    let s = sub_t(&R[5], &s, "\\2");
    let s = sub_t(&R[6], &s, "\\1");
    s.replace('`', "").replace("**", "")
}

/// Divergence from ag2p: 2000-2099 read as years only after a year cue; leading zeros read digit by digit.
fn year_cue(before: &str) -> bool {
    static CUE: LazyLock<Re> = LazyLock::new(|| {
        let months = "January|February|March|April|May|June|July|August|September|October|November|December";
        Re::new(&format!(
            r"(?:\b(?i:in|since|by|until|till|from|during|year|circa|before|after|through|around|early|late|mid)|\b(?i:summer|winter|spring|fall|autumn|class|end|start|beginning|middle)\s+of|\b(?:{months})\b(?:\s+\w+)?,?|\b(?:1[1-9]|20)\d\d\s+(?:to|and|or|through))\s+\z"
        ))
    });
    let mut start = before.len().saturating_sub(80);
    while !before.is_char_boundary(start) {
        start -= 1;
    }
    CUE.search(&before[start..])
}

fn number(m: &Captures, subject: &str) -> String {
    static COMMA: LazyLock<Re> = LazyLock::new(|| Re::new(r"\d,\d"));
    let s = g(m, 0).replace(',', "");
    if let Some(dot) = s.find('.') {
        return format!("{} point {}", int_words(&s[..dot]), digit_words(&s[dot + 1..]).join(" "));
    }
    if s.len() > 1 && s.starts_with('0') && !g(m, 0).contains(',') {
        return digit_words(&s)
            .into_iter()
            .map(|x| if x == "zero" { "oh".to_string() } else { x })
            .collect::<Vec<_>>()
            .join(" ");
    }
    let mut d = s.as_str();
    while d.len() > 1 && d.starts_with('0') {
        d = &d[1..];
    }
    let n: i64 = if d.len() <= 4 { d.parse().unwrap_or(0) } else { 0 };
    if (1100..=1999).contains(&n) && s.len() == 4 && !COMMA.search(g(m, 0)) {
        return year_words(n);
    }
    if (2000..=2099).contains(&n) && s.len() == 4 && year_cue(&subject[..b(m)]) {
        return if n < 2010 { int_words_n(n) } else { year_words(n) };
    }
    int_words(&s)
}

fn decade(m: &Captures) -> String {
    let d = g(m, 1);
    let w = if d.len() == 2 {
        int_words(d)
    } else if d < "2000" {
        year_words(d.parse().unwrap())
    } else if d < "2010" {
        int_words(d)
    } else {
        year_words(d.parse().unwrap())
    };
    match w.strip_suffix('y') {
        Some(t) => format!("{t}ies"),
        None => format!("{w}s"),
    }
}

/// A match ending in "." keeps it when the sentence ends there.
fn keep_dot(m: &Captures, subject: &str) -> bool {
    static NEXT: LazyLock<Re> = LazyLock::new(|| Re::new(r"\s*\z|\s+[A-Z]"));
    g(m, 0).ends_with('.') && NEXT.find(subject, e(m)).is_some_and(|n| b(&n) == e(m))
}

fn roman(s: &str) -> i64 {
    let v: Vec<i64> = s
        .bytes()
        .filter_map(|c| match c | 32 {
            b'i' => Some(1),
            b'v' => Some(5),
            b'x' => Some(10),
            b'l' => Some(50),
            b'c' => Some(100),
            b'd' => Some(500),
            b'm' => Some(1000),
            _ => None,
        })
        .collect();
    (0..v.len()).map(|i| if i + 1 < v.len() && v[i] < v[i + 1] { -v[i] } else { v[i] }).sum()
}

fn numbers(t: String) -> String {
    static R: LazyLock<Vec<Re>> = LazyLock::new(|| {
        let roman = "M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})";
        vec![
            Re::new(&format!(
                r"\b((?i:chapter|book|part|volume|section|act|scene|canto))\s+(?:(?=[MDCLXVI]+\b)({roman})|(?=[ivx]+\b)(x{{0,3}}(?:ix|iv|v?i{{0,3}})))(?<=[MDCLXVIivx])\b(?<!\bI\b(?=\s+[a-z]))"
            )),
            Re::new(r"\$\s*(\d[\d,]*)(?:\.(\d\d))?\b"),
            Re::new(r"\bv?(\d+(?:\.\d+){2,})\b"),
            Re::new(r"\bv(\d+(?:\.\d+)?)\b"),
            Re::new(r"(?<![\w.,'])'?([1-9]0|1[1-9]\d0|20\d0)'?s\b"),
            Re::new(r"(?<=\d)\s*([apAP])\.?[mM]\b\.?"),
            Re::new(r"\b(\d{1,2}):(\d\d)(?!\d)"),
            Re::new(r"~\s*(?=\d)"),
            Re::new(r"(?<![\w.])(\d+)\s*[-—–]\s*(\d+)(?![\w.-])"),
            Re::new(concat!(
                r"(?<![A-Za-z\d.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:(k|K|M|B|x|×)|\s?",
                r"(KB|kB|MB|GB|TB|PB|ms|ns|µs|μs|GHz|MHz|kHz|Hz|Gbps|Mbps|kbps|fps|km|kg|cm|mm))\b"
            )),
            Re::new(r"\b(\d+)(st|nd|rd|th)\b"),
            Re::new(r"(\d)\+"),
            Re::new(r"(?<![A-Za-z\d])(?>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Za-z\d])"),
        ]
    });
    let t = sub(&R[0], &t, |m| format!("{} {}", g(m, 1), roman(if !g(m, 2).is_empty() { g(m, 2) } else { g(m, 3) })));
    let t = sub(&R[1], &t, |m| {
        format!("{} dollars{}", g(m, 1), if set(m, 2) { format!(" and {} cents", g(m, 2)) } else { String::new() })
    });
    let t = sub(&R[2], &t, |m| g(m, 1).split('.').map(int_words).collect::<Vec<_>>().join(" point "));
    let t = sub_t(&R[3], &t, "version \\1");
    let t = sub(&R[4], &t, decade);
    let t = {
        let s = t.clone();
        sub(&R[5], &t, |m| format!(" {} M{}", upper(g(m, 1)), if keep_dot(m, &s) { "." } else { "" }))
    };
    let t = sub(&R[6], &t, |m| {
        let mm = g(m, 2);
        let tail = if mm == "00" {
            String::new()
        } else if mm.starts_with('0') {
            format!("oh {}", int_words(mm))
        } else {
            int_words(mm)
        };
        format!("{} {tail}", int_words(g(m, 1)))
    });
    let t = sub_t(&R[7], &t, "about ");
    let t = sub_t(&R[8], &t, "\\1 to \\2");
    let t = sub(&R[9], &t, |m| {
        #[rustfmt::skip]
        const U: [(&str, &str); 28] = [
            ("k", "thousand"), ("K", "thousand"), ("M", "million"), ("B", "billion"), ("x", "times"), ("×", "times"),
            ("KB", "kilobytes"), ("kB", "kilobytes"), ("MB", "megabytes"), ("GB", "gigabytes"), ("TB", "terabytes"),
            ("PB", "petabytes"), ("ms", "milliseconds"), ("ns", "nanoseconds"), ("µs", "microseconds"),
            ("μs", "microseconds"), ("GHz", "gigahertz"), ("MHz", "megahertz"), ("kHz", "kilohertz"), ("Hz", "hertz"),
            ("Gbps", "gigabits per second"), ("Mbps", "megabits per second"), ("kbps", "kilobits per second"),
            ("fps", "frames per second"), ("km", "kilometers"), ("kg", "kilograms"), ("cm", "centimeters"),
            ("mm", "millimeters"),
        ];
        let unit = if set(m, 2) { g(m, 2) } else { g(m, 3) };
        match U.iter().find(|(k, _)| *k == unit) {
            Some((_, v)) => {
                let mut name = v.to_string();
                let end = name.find(" per").unwrap_or(name.len());
                if g(m, 1) == "1" && !set(m, 2) && name.as_bytes()[end - 1] == b's' {
                    name.remove(end - 1);
                }
                format!("{} {name}", g(m, 1))
            }
            None => g(m, 0).to_string(),
        }
    });
    let t = sub(&R[10], &t, |m| ordinal_words(g(m, 1)));
    let t = sub_t(&R[11], &t, "\\1 plus");
    let s = t.clone();
    sub(&R[12], &t, |m| number(m, &s))
}

fn nfc(l: &Lexicon, s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    let mut o: Vec<char> = Vec::new();
    for c in s.chars() {
        let c = l.nfc_single.get(&c).copied().unwrap_or(c);
        if let Some(&p) = o.last().and_then(|&last| l.nfc_pair.get(&(last, c))) {
            *o.last_mut().unwrap() = p;
            continue;
        }
        o.push(c);
    }
    o.into_iter().collect()
}

fn normalize_impl(l: &Lexicon, text: &str) -> String {
    let t = nfc(l, text).replace('“', "\"").replace('”', "\"").replace('‘', "'").replace('’', "'");
    let mut t = markdown(&t);
    #[rustfmt::skip]
    const ABBREV: [(&str, &str); 8] = [
        ("e.g.", "for example"), ("e.g", "for example"), ("i.e.", "that is"), ("i.e", "that is"), ("etc.", "et cetera"),
        ("vs.", "versus"), ("vs", "versus"), ("approx.", "approximately"),
    ];
    static R: LazyLock<Vec<Re>> = LazyLock::new(|| {
        let mut v: Vec<Re> =
            ABBREV.iter().map(|(k, _)| Re::new(&format!(r"(?<!\w){}(?!\w)", k.replace('.', r"\.")))).collect();
        v.extend([
            Re::new(r"\b(Dr|Mr|Mrs|Ms|Prof|Mt)\.(?=\s+[A-Z])"),
            Re::new(r"\bSt\."),
            Re::new(r"[a-z0-9,]\s+[A-Z][a-z]+\s+\z"),
            Re::new(r"\s+[A-Z]"),
            Re::new(r"\b(Jr|Sr)\."),
            Re::new(r"(?<!\w)etc\.(?=\s*\z|\s+[A-Z])"),
            Re::new(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s+\d)"),
            Re::new(r"(?:https?://|www\.)[^\s)>\]]+?(?=[.,;:!?)]*(?:\s|$))"),
            Re::new(r"^(https?://)?(www\.)?"),
            Re::new(r"(?<![\w/])(?:~/|/|\w[\w.-]*/)[\w./~-]*\w/?|\b\w[\w-]*\.[A-Za-z][A-Za-z0-9]{0,4}\b(?=[\s,;:!?)]|\.(?:\s|$)|$)"),
            Re::new(r"\b([A-Za-z]{2,})\+\+"),
            Re::new(r"\bC\+\+"),
            Re::new(r"\b(?:[A-Z]\.){2,}(?![A-Za-z])"),
            Re::new(r"\b([A-Z]{2,})-(\d+)\b"),
            Re::new(r"(?<=[A-Za-z])\.(?=[A-Za-z])"),
            Re::new(r"(?<=\w)/(?=\w)"),
            Re::new(r"(?<=\w)_(?=\w)"),
            Re::new(r"(?<=\w)=(?=\w)"),
            Re::new(r"-{2,}|[—–]|\s-\s"),
            Re::new(r"\s*\.{3}|…"),
            Re::new(r"[ \t]+"),
            Re::new(r".*"),
        ]);
        v
    });
    let r = |i: usize| &R[8 + i];
    t = sub(r(0), &t, |m| {
        #[rustfmt::skip]
        let w = [
            ("Dr", "Doctor"), ("Mr", "Mister"), ("Mrs", "Missus"), ("Ms", "Miz"), ("Prof", "Professor"),
            ("Mt", "Mount"),
        ];
        w.iter().find(|(k, _)| *k == g(m, 1)).map_or_else(|| g(m, 0).to_string(), |(_, v)| v.to_string())
    });
    let s = t.clone();
    t = sub(r(1), &t, |m| {
        let before = &s[..b(m)];
        let mut st = before.len().saturating_sub(40);
        while !before.is_char_boundary(st) {
            st -= 1;
        }
        if r(2).search(&before[st..]) {
            return format!("Street{}", if keep_dot(m, &s) { "." } else { "" });
        }
        if r(3).find(&s, e(m)).is_some_and(|n| b(&n) == e(m)) { "Saint".into() } else { g(m, 0).to_string() }
    });
    let s = t.clone();
    t = sub(r(4), &t, |m| {
        format!("{}{}", if g(m, 1) == "Jr" { "Junior" } else { "Senior" }, if keep_dot(m, &s) { "." } else { "" })
    });
    t = sub_t(r(5), &t, "et cetera.");
    t = sub(r(6), &t, |m| {
        #[rustfmt::skip]
        let w = [
            ("Jan", "January"), ("Feb", "February"), ("Mar", "March"), ("Apr", "April"), ("Jun", "June"),
            ("Jul", "July"), ("Aug", "August"), ("Sep", "September"), ("Sept", "September"), ("Oct", "October"),
            ("Nov", "November"), ("Dec", "December"),
        ];
        w.iter().find(|(k, _)| *k == g(m, 1)).map_or_else(|| g(m, 0).to_string(), |(_, v)| v.to_string())
    });
    for (i, (_, v)) in ABBREV.iter().enumerate() {
        t = sub_t(&R[i], &t, v);
    }
    t = sub(r(7), &t, |m| {
        let mut s = sub_t(r(8), g(m, 0), "");
        while s.ends_with('/') {
            s.pop();
        }
        let line = r(21).find(&s, 0).map(|d| g(&d, 0).to_string()).unwrap_or_default();
        path_words(&line)
    });
    t = sub(r(9), &t, |m| {
        let s = g(m, 0);
        let e = lower(match s.rfind('.') {
            Some(dp) => &s[dp + 1..],
            None => s,
        });
        if s.contains('/') || ext(&e).is_some() || s.contains('_') { path_words(s) } else { s.to_string() }
    });
    t = sub_t(r(10), &t, "\\1 plus plus");
    t = sub_t(r(11), &t, "C plus plus");
    t = sub(r(12), &t, |m| g(m, 0).replace('.', ""));
    t = sub_t(r(13), &t, "\\1 \\2");
    t = numbers(t);
    t = sub_t(r(14), &t, " dot ");
    t = sub_t(r(15), &t, " slash ");
    t = sub_t(r(16), &t, " ");
    t = sub_t(r(17), &t, " equals ");
    #[rustfmt::skip]
    const SYMBOLS: [(&str, &str); 17] = [
        ("→", " to "), ("←", " from "), ("↔", " and "), ("≈", " about "), ("≤", " at most "), ("≥", " at least "),
        ("&", " and "), ("=", " equals "), ("×", " times "), ("@", " at "), ("#", " number "), ("•", ", "), ("|", ", "),
        ("+", " plus "), ("%", " percent "), ("°", " degrees "), ("§", " section "),
    ];
    for (k, v) in SYMBOLS {
        t = t.replace(k, v);
    }
    t = sub_t(r(18), &t, ", ");
    t = sub_t(r(19), &t, "...");
    strip(&sub_t(r(20), &t, " ")).to_string()
}

type Spans = Vec<(usize, usize)>;

/// spaCy Tokenizer._tokenize_affixes for one whitespace-free span [b, e) of full.
fn tok_span(l: &Lexicon, full: &str, mut b: usize, mut e: usize, out: &mut Spans, specials: bool) {
    let special = |s: usize, t: usize| if specials { l.special(&full[s..t]) } else { None };
    let emit_special = |s: usize, pieces: &str, out: &mut Spans| {
        let mut s = s;
        for p in pieces.split('\x1f') {
            out.push((s, s + p.len()));
            s += p.len();
        }
    };
    if let Some(sp) = special(b, e) {
        return emit_special(b, sp, out);
    }
    let (mut pre, mut suf): (Spans, Spans) = (Vec::new(), Vec::new());
    let mut last = usize::MAX;
    while e > b && e - b != last {
        if special(b, e).is_some() {
            break;
        }
        last = e - b;
        let pre_len = l.prefix.find(&full[b..e], 0).map_or(0, |m| e_(&m) - b_(&m));
        if pre_len > 0 && b + pre_len < e && special(b + pre_len, e).is_some() {
            pre.push((b, b + pre_len));
            b += pre_len;
            break;
        }
        let suf_len = l.suffix.find(&full[b + pre_len..e], 0).map_or(0, |m| e_(&m) - b_(&m));
        if suf_len > 0 && e - suf_len > b && special(b, e - suf_len).is_some() {
            suf.push((e - suf_len, e));
            e -= suf_len;
            break;
        }
        if pre_len > 0 && suf_len > 0 && pre_len + suf_len <= e - b {
            pre.push((b, b + pre_len));
            suf.push((e - suf_len, e));
            b += pre_len;
            e -= suf_len;
        } else if pre_len > 0 {
            pre.push((b, b + pre_len));
            b += pre_len;
        } else if suf_len > 0 {
            suf.push((e - suf_len, e));
            e -= suf_len;
        }
    }
    out.extend(pre);
    if e > b {
        let s = &full[b..e];
        if let Some(sp) = special(b, e) {
            emit_special(b, sp, out);
        } else if l.url.search(s) {
            out.push((b, e));
        } else {
            let (mut start, mut any) = (0, false);
            find_iter(&l.infix, s, |x| {
                any = true;
                if b_(x) == 0 {
                    return;
                }
                if b_(x) != start {
                    out.push((b + start, b + b_(x)));
                }
                if b_(x) != e_(x) {
                    out.push((b + b_(x), b + e_(x)));
                }
                start = e_(x);
            });
            if !any {
                out.push((b, e));
            } else if start < s.len() {
                out.push((b + start, e));
            }
        }
    }
    out.extend(suf.into_iter().rev());
}

fn b_(m: &Captures) -> usize {
    b(m)
}
fn e_(m: &Captures) -> usize {
    e(m)
}

fn tokenize_spans(l: &Lexicon, text: &str, out: &mut Spans, specials: bool) {
    let mut start = 0;
    let mut in_ws = true;
    let mut it = text.char_indices().peekable();
    loop {
        let (i, c) = match it.next() {
            Some(x) => x,
            None => (text.len(), ' '),
        };
        let ws = i == text.len() || is_space(c);
        if ws && !in_ws {
            tok_span(l, text, start, i, out, specials);
        }
        if !ws && in_ws {
            start = i;
        }
        in_ws = ws;
        if i == text.len() {
            break;
        }
    }
}

/// spaCy en_core_web_sm tokenizer: [begin, end) byte spans of the non-space tokens.
fn tokenize_impl(l: &Lexicon, text: &str) -> Spans {
    let mut out = Vec::new();
    tokenize_spans(l, text, &mut out, true);
    struct Span<'a> {
        s: usize,
        n: usize,
        key: &'a str,
    }
    let tok = |i: usize, out: &Spans| &text[out[i].0..out[i].1];
    let mut spans: Vec<Span> = Vec::new();
    for i in 0..out.len() {
        let Some(cands) = l.matcher.get(tok(i, &out)) else { continue };
        for (toks, key) in cands {
            let ok = i + toks.len() <= out.len() && (1..toks.len()).all(|k| tok(i + k, &out) == toks[k]);
            if ok {
                spans.push(Span { s: i, n: toks.len(), key });
            }
        }
    }
    if spans.is_empty() {
        return out;
    }
    spans.sort_by(|a, b| b.n.cmp(&a.n).then(a.s.cmp(&b.s)));
    let mut taken = vec![false; out.len()];
    let mut keep: Vec<Option<usize>> = vec![None; out.len()];
    for (si, sp) in spans.iter().enumerate() {
        if taken[sp.s..sp.s + sp.n].iter().any(|&t| t) {
            continue;
        }
        taken[sp.s..sp.s + sp.n].fill(true);
        keep[sp.s] = Some(si);
    }
    let mut res = Vec::new();
    let mut i = 0;
    while i < out.len() {
        let b0 = out[i].0;
        if let Some(si) = keep[i]
            && &text[b0..out[i + spans[si].n - 1].1] == spans[si].key
        {
            let mut s = b0;
            for p in l.special(spans[si].key).unwrap_or("").split('\x1f') {
                res.push((s, s + p.len()));
                s += p.len();
            }
            i += spans[si].n;
        } else {
            res.push(out[i]);
            i += 1;
        }
    }
    res
}

/// Latin-1 and Œœ letters to ASCII, for the accent-stripped lexicon retry.
fn fold(s: &str) -> String {
    const LATIN1: [&str; 64] = [
        "A", "A", "A", "A", "A", "A", "AE", "C", "E", "E", "E", "E", "I", "I", "I", "I", "D", "N", "O", "O", "O", "O",
        "O", "", "O", "U", "U", "U", "U", "Y", "TH", "ss", "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e",
        "i", "i", "i", "i", "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "y",
    ];
    let mut o = String::new();
    for c in s.chars() {
        let u = c as u32;
        if (0xC0..=0xFF).contains(&u) && !LATIN1[(u - 0xC0) as usize].is_empty() {
            o += LATIN1[(u - 0xC0) as usize];
        } else if u == 0x152 || u == 0x153 {
            o += if u == 0x152 { "OE" } else { "oe" };
        } else {
            o.push(c);
        }
    }
    o
}

const IPA_CHARS: &str = "əɚɝɪʊʌɛæɑɔɹʃʒθðŋˈˌɾʔ";
#[rustfmt::skip]
const LETTERS: [&str; 26] = [
    "eɪ", "bi", "si", "di", "i", "ɛf", "dʒi", "eɪtʃ", "aɪ", "dʒeɪ", "keɪ", "ɛl", "ɛm", "ɛn", "oʊ", "pi", "kju", "ɑɹ",
    "ɛs", "ti", "ju", "vi", "dʌbəlju", "ɛks", "waɪ", "zi",
];

fn tag_class(tag: &str) -> &'static str {
    match tag {
        "NN" | "NNS" | "NNP" | "NNPS" => "noun",
        "VB" | "VBP" | "VBZ" | "VBG" => "verb",
        "VBD" | "VBN" => "past",
        "JJ" | "JJR" | "JJS" => "adj",
        "RB" | "RBR" | "RBS" => "adv",
        "IN" => "prep",
        _ => "",
    }
}

fn tag_fallback(klass: &str) -> &'static str {
    match klass {
        "past" => "verb",
        "adv" => "adj",
        _ => "",
    }
}

struct Part {
    text: String,
    ipa: Opt,
    ps: String,
    src: String,
}

struct Hit {
    ipa: String,
    src: String,
    unstressed: bool,
}

struct Tok {
    tag: String,
    ps: String,
    source: String,
}

pub struct G2p {
    lex: Lexicon,
    tagger: Tagger,
    guesser: Guesser,
}

type R<T> = Result<T, Unmappable>;

impl G2p {
    pub fn load(lexicon: Vec<u8>, tagger: &[u8], guesser: &[u8]) -> io::Result<G2p> {
        Ok(G2p { lex: load_lexicon(lexicon)?, tagger: Tagger::load(tagger)?, guesser: Guesser::load(guesser)? })
    }

    fn k(&self, ipa: &str, unstressed: bool) -> R<String> {
        to_k(&self.lex, ipa, unstressed)
    }

    fn rule_lookup<'a>(&'a self, tag: &str) -> impl Fn(&str) -> Opt + 'a {
        let klass = tag_class(tag);
        move |w: &str| {
            let r = self.lex.row(w)?;
            if r.status == b'r' {
                return Some(r.ipa.to_string());
            }
            if r.status == b'c' && !klass.is_empty() {
                return r.cond(klass).or_else(|| r.cond(tag_fallback(klass)));
            }
            None
        }
    }

    fn lex(&self, key: &str, tag: &str) -> Option<Hit> {
        let r = self.lex.row(key)?;
        #[rustfmt::skip]
        const STRONG: [(&str, &str); 7] = [
            ("us", "ʌs"), ("them", "ðɛm"), ("her", "hɝ"), ("your", "jɔɹ"), ("am", "æm"), ("as", "æz"), ("than", "ðæn"),
        ];
        let hit = |ipa: String, src: String| Some(Hit { ipa, src, unstressed: false });
        if r.status == b'r' {
            if let Some((_, v)) = STRONG.iter().find(|(k, _)| *k == key) {
                return Some(Hit { ipa: v.to_string(), src: "lex-strong".into(), unstressed: true });
            }
            return hit(r.ipa.to_string(), if r.import { "lex-import" } else { "lex" }.into());
        }
        if r.status == b'w' {
            let v = r.cond("strong").unwrap_or_else(|| r.opts.split('\x1e').next().unwrap_or("").to_string());
            return Some(Hit { ipa: v, src: "lex-strong".into(), unstressed: true });
        }
        let ts = if tag.is_empty() { "None" } else { tag };
        if r.status == b'c' {
            let klass = tag_class(tag);
            if !klass.is_empty()
                && let Some(v) = r.cond(klass)
            {
                return hit(v, format!("variant+{ts}"));
            }
            let fb = if ["NN", "NNS", "NNP", "NNPS", "CD", "FW"].contains(&tag) { "noun" } else { tag_fallback(klass) };
            if !fb.is_empty()
                && let Some(v) = r.cond(fb)
            {
                return hit(v, format!("variant-{fb}+{ts}"));
            }
            return hit(r.def.to_string(), "ambiguous-default".into());
        }
        if let Some(d) = apply_rules(key, &self.rule_lookup(tag))
            && !d.is_empty()
        {
            let dd = d.replace('ˌ', "");
            if r.opts.split('\x1e').any(|o| o.replace('ˌ', "") == dd) {
                return hit(d, format!("rule+{ts}"));
            }
        }
        hit(r.def.to_string(), "ambiguous-default".into())
    }

    fn lex_key(&self, w: &str) -> String {
        static A: LazyLock<Re> = LazyLock::new(|| Re::full("(?=.*[A-Z])[A-Z0-9]{2,}s?"));
        static B: LazyLock<Re> = LazyLock::new(|| Re::full("[B-HJ-Z]"));
        if A.search(w) || B.search(w) { w.to_string() } else { lower(w) }
    }

    fn spell(&self, word: &str) -> R<Part> {
        let plural = word.ends_with('s') && !is_upper(word);
        let letters = if plural { word.trim_end_matches('s') } else { word };
        let up: Vec<u8> =
            (if letters.is_empty() { word } else { letters }).bytes().filter(u8::is_ascii_uppercase).collect();
        let mut ipa = String::new();
        for (i, &c) in up.iter().enumerate() {
            ipa += if i + 1 < up.len() { "ˌ" } else { "ˈ" };
            ipa += LETTERS[(c - b'A') as usize];
        }
        if plural && !ipa.is_empty() {
            ipa = plural_suffix(&ipa);
        }
        Ok(Part { text: word.into(), ps: self.k(&ipa, false)?, ipa: Some(ipa), src: "norm-spell".into() })
    }

    fn digit_ipa(&self, s: &str, between: bool, out: &mut Vec<String>) {
        let words: Vec<String> = if between {
            vec!["to".into()]
        } else if nchars(s) < 5 {
            int_words(s).split(' ').map(String::from).collect()
        } else {
            digit_words(s)
        };
        for w in words {
            if let Some(h) = self.lex(&w, "") {
                out.push(if w == "to" { h.ipa.replace('ˈ', "") } else { h.ipa });
            }
        }
    }

    fn alnum(&self, word: &str) -> R<Part> {
        static RX: LazyLock<Re> = LazyLock::new(|| Re::new(r"[A-Za-z]+|\d+"));
        let up = upper(word);
        let mut out = Vec::new();
        let mut err = None;
        find_iter(&RX, &up, |m| {
            let s = g(m, 0);
            if s.as_bytes()[0].is_ascii_digit() {
                self.digit_ipa(s, b(m) > 0 && e(m) < word.len() && s == "2", &mut out);
            } else {
                match self.spell(s) {
                    Ok(p) => {
                        let i = p.ipa.unwrap_or_default();
                        if !i.is_empty() {
                            out.push(i);
                        }
                    }
                    Err(e) => err = Some(e),
                }
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        let ps = out.iter().map(|p| self.k(p, false)).collect::<R<Vec<_>>>()?;
        Ok(Part { text: word.into(), ipa: Some(out.join(" ")), ps: ps.join(" "), src: "norm-spell".into() })
    }

    fn model(&self, word: &str) -> R<(String, String)> {
        let ipa = self.guesser.guess(word);
        if !ipa.is_empty()
            && let Ok(ps) = self.k(&ipa, false)
        {
            return Ok((ipa, ps));
        }
        static ALNUM: LazyLock<Re> = LazyLock::new(|| Re::new("[A-Za-z0-9]"));
        let f = fold(word);
        let p = if ALNUM.search(&f) { self.alnum(&f)? } else { self.spell("X")? };
        Ok((p.ipa.unwrap_or_default(), p.ps))
    }

    fn split_parts(&self, word: &str, tag: &str) -> R<Vec<Part>> {
        let parts: Vec<&str> = word.split('-').filter(|p| !p.is_empty()).collect();
        let mut o = Vec::new();
        for (i, p) in parts.iter().enumerate() {
            o.extend(self.resolve(p, if i == parts.len() - 1 { tag } else { "NN" })?);
        }
        Ok(o)
    }

    fn resolve(&self, word: &str, tag: &str) -> R<Vec<Part>> {
        if word.chars().any(|c| IPA_CHARS.contains(c)) {
            let f: String = word.chars().filter(|&c| self.lex.vocab.contains(&c) || c == 'ˈ' || c == 'ˌ').collect();
            return Ok(vec![Part {
                text: word.into(),
                ipa: Some(word.into()),
                ps: self.k(&f, false)?,
                src: "norm".into(),
            }]);
        }
        let key = self.lex_key(word);
        let lw = lower(word);
        let mut hit = self.lex(&key, tag);
        static KEEP: LazyLock<Re> = LazyLock::new(|| Re::full("[A-Z]{2,3}s?|[A-Z]+[0-9][A-Za-z0-9]*|[^AEIOUY]+s?"));
        if hit.is_none() && key != lw && !KEEP.search(word) {
            hit = self.lex(&lw, tag);
        }
        if let Some(h) = hit {
            return Ok(vec![Part {
                text: word.into(),
                ps: self.k(&h.ipa, h.unstressed)?,
                ipa: Some(h.ipa),
                src: h.src,
            }]);
        }
        static ACR_S: LazyLock<Re> = LazyLock::new(|| Re::full("[A-Z]{2,}s"));
        let poss = lw.ends_with("'s") && nchars(word) > 2;
        if poss || ACR_S.search(word) {
            let mut parts = self.resolve(&word[..word.len() - if poss { 2 } else { 1 }], tag)?;
            let Some(last) = parts.pop() else { return Ok(parts) };
            match last.ipa {
                Some(ipa) => {
                    let ipa = plural_suffix(&ipa);
                    parts.push(Part { text: word.into(), ps: self.k(&ipa, false)?, ipa: Some(ipa), src: last.src });
                }
                None => {
                    let c = last_char(&last.ps);
                    let sfx = if c.is_none_or(|c| "szʃʒʧʤ".contains(c)) { "ɪz" } else { "z" };
                    parts.push(Part { text: word.into(), ipa: None, ps: format!("{}{sfx}", last.ps), src: last.src });
                }
            }
            return Ok(parts);
        }
        static CAMEL: LazyLock<Re> = LazyLock::new(|| Re::new("[a-z][A-Z]|[A-Z]{2}[a-z]"));
        static CAMEL_PARTS: LazyLock<Re> =
            LazyLock::new(|| Re::new(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+(?:'[a-z]+)?|[A-Z]+s?(?![a-z])|\d+"));
        if CAMEL.search(word) && !word.contains('-') {
            let mut parts = Vec::new();
            find_iter(&CAMEL_PARTS, word, |m| parts.push(g(m, 0).to_string()));
            if parts.len() > 1 {
                let mut o = Vec::new();
                for p in parts {
                    o.extend(self.resolve(&p, tag)?);
                }
                return Ok(o);
            }
        }
        static ALNUM: LazyLock<Re> =
            LazyLock::new(|| Re::full("[A-Z]+[0-9]+[A-Za-z0-9]*|[a-z]{1,3}[0-9]+[a-z0-9]{0,3}"));
        static ACR: LazyLock<Re> = LazyLock::new(|| Re::full("[A-Z]{2,}s?"));
        static VOWEL: LazyLock<Re> = LazyLock::new(|| Re::new("[AEIOU]"));
        if ALNUM.search(word) {
            return Ok(vec![self.alnum(word)?]);
        }
        if ACR.search(word) {
            let base = word.trim_end_matches('s');
            if nchars(base) <= 3 || !VOWEL.search(word) {
                return Ok(vec![self.spell(word)?]);
            }
        }
        let hyphen = strip_chars(word, "-").contains('-');
        if hyphen
            && word
                .split('-')
                .all(|p| p.is_empty() || self.lex.row(&self.lex_key(p)).is_some() || self.lex.row(&lower(p)).is_some())
        {
            return self.split_parts(word, tag);
        }
        if let Some(ipa) = apply_rules(&key, &self.rule_lookup(tag)) {
            return Ok(vec![Part { text: word.into(), ps: self.k(&ipa, false)?, ipa: Some(ipa), src: "rule".into() }]);
        }
        if hyphen {
            #[rustfmt::skip]
            const PREFIXES: [(&str, &str); 7] = [
                ("re", "ˌɹi"), ("un", "ˌʌn"), ("de", "ˌdi"), ("pre", "ˌpɹi"), ("co", "ˌkoʊ"), ("non", "ˌnɑn"),
                ("sub", "ˌsʌb"),
            ];
            let d = word.find('-').unwrap();
            let (head, rest) = (&word[..d], &word[d + 1..]);
            if let Some((_, v)) = PREFIXES.iter().find(|(k, _)| *k == lower(head))
                && !rest.is_empty()
            {
                let mut o = vec![Part {
                    text: head.into(),
                    ipa: Some(v.to_string()),
                    ps: self.k(v, false)?,
                    src: "rule".into(),
                }];
                o.extend(self.resolve(rest, tag)?);
                return Ok(o);
            }
            return self.split_parts(word, tag);
        }
        static DIGIT: LazyLock<Re> = LazyLock::new(|| Re::new("[0-9]"));
        static RUNS: LazyLock<Re> = LazyLock::new(|| Re::new("[0-9]+|[^0-9]+"));
        if DIGIT.search(word) {
            let mut o = Vec::new();
            let mut err = None;
            find_iter(&RUNS, word, |m| {
                let s = g(m, 0);
                if s.as_bytes()[0].is_ascii_digit() {
                    let mut ipas = Vec::new();
                    self.digit_ipa(s, false, &mut ipas);
                    match ipas.iter().map(|i| self.k(i, false)).collect::<R<Vec<_>>>() {
                        Ok(ps) if !ipas.is_empty() => o.push(Part {
                            text: s.into(),
                            ipa: Some(ipas.join(" ")),
                            ps: ps.join(" "),
                            src: "norm-num".into(),
                        }),
                        Ok(_) => {}
                        Err(e) => err = Some(e),
                    }
                } else {
                    let r = strip_chars(s, "-'");
                    if !r.is_empty() {
                        match self.resolve(r, tag) {
                            Ok(p) => o.extend(p),
                            Err(e) => err = Some(e),
                        }
                    }
                }
            });
            return match err {
                Some(e) => Err(e),
                None => Ok(o),
            };
        }
        let f = fold(word);
        if f != word {
            let (fk, fl) = (self.lex_key(&f), lower(&f));
            let mut h = self.lex(&fk, tag);
            if h.is_none() && fk != fl {
                h = self.lex(&fl, tag);
            }
            if let Some(h) = h {
                return Ok(vec![Part {
                    text: word.into(),
                    ps: self.k(&h.ipa, h.unstressed)?,
                    ipa: Some(h.ipa),
                    src: format!("{}+fold", h.src),
                }]);
            }
            if let Some(ipa) = apply_rules(&fl, &self.rule_lookup(tag)) {
                return Ok(vec![Part {
                    text: word.into(),
                    ps: self.k(&ipa, false)?,
                    ipa: Some(ipa),
                    src: "rule+fold".into(),
                }]);
            }
        }
        let (ipa, ps) = self.model(word)?;
        Ok(vec![Part { text: word.into(), ipa: Some(ipa), ps, src: "model-fallback".into() }])
    }

    fn tokens(&self, line: &str) -> R<Vec<Tok>> {
        let text = normalize_impl(&self.lex, line);
        static TOKEN: LazyLock<Re> = LazyLock::new(|| {
            Re::new(
                r#"[A-Za-z]+[0-9]+[A-Za-z0-9]*|[A-Za-zÀ-ÖØ-öø-ÿŒœÆæəɚɝɪʊʌɛæɑɔɹʃʒθðŋˈˌɾʔ'-]+|[0-9]+|\.\.\.|[,.!?;:()"]"#,
            )
        });
        let spans = tokenize_impl(&self.lex, &text);
        let words: Vec<String> = spans.iter().map(|&(b, e)| text[b..e].to_string()).collect();
        let tags = self.tagger.tag(&words);
        let tagged: Vec<(usize, usize, &str)> =
            spans.iter().zip(&tags).map(|(&(b, e), t)| (b, e, t.as_str())).collect();
        let mut out = Vec::new();
        let mut j = 0;
        let mut err = None;
        find_iter(&TOKEN, &text, |m| {
            if err.is_some() {
                return;
            }
            let (s, en, w) = (b(m), e(m), g(m, 0));
            if w.chars().all(|c| "'-,.!?;:()\"".contains(c)) {
                let p = match w {
                    "(" | ")" => ",",
                    "\"" => "",
                    "..." => "…",
                    _ => w,
                };
                if !p.is_empty() {
                    out.push(Tok { tag: String::new(), ps: p.into(), source: "punct".into() });
                }
                return;
            }
            let w2 = strip_chars(w, "-'");
            if w2.is_empty() {
                return;
            }
            while j < tagged.len() && tagged[j].1 <= s {
                j += 1;
            }
            let tag = if j < tagged.len() && tagged[j].0 < en { tagged[j].2 } else { "" };
            let parts = match self.resolve(w2, tag) {
                Ok(p) => p,
                Err(_) => match self.model(w2) {
                    Ok((ipa, ps)) => vec![Part { text: w2.into(), ipa: Some(ipa), ps, src: "model-fallback".into() }],
                    Err(e) => {
                        err = Some(e);
                        return;
                    }
                },
            };
            const FUNCTION_TAGS: [&str; 12] =
                ["CC", "DT", "IN", "PRP", "PRP$", "TO", "MD", "WDT", "WP", "WP$", "EX", "PDT"];
            let single = parts.len() == 1;
            for p in parts {
                let mut ps = p.ps;
                if FUNCTION_TAGS.contains(&tag) && single && ps.chars().filter(|&c| VOWELS.contains(c)).count() == 1 {
                    ps = ps.replace('ˈ', "").replace('ˌ', "");
                }
                let _ = (&p.text, &p.ipa);
                out.push(Tok { tag: tag.into(), ps, source: p.src });
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    fn line(&self, line: &str) -> R<String> {
        let mut ps = String::new();
        for t in self.tokens(line)? {
            let _ = &t.tag;
            if t.source == "punct" {
                let c = last_char(&ps);
                if !ps.is_empty() && !c.is_some_and(|c| PUNCT.contains(c)) {
                    ps += &t.ps;
                } else if !ps.is_empty() && c == Some(',') && t.ps != "," {
                    ps.pop();
                    ps += &t.ps;
                }
            } else if !t.ps.is_empty() {
                if !ps.is_empty() {
                    ps.push(' ');
                }
                ps += &t.ps;
            }
        }
        Ok(strip_chars(&ps, " ,;:").to_string())
    }

    /// Phonemes per non-empty input line, unchunked.
    pub fn phonemize_lines(&self, text: &str) -> Vec<String> {
        strip(text)
            .split('\n')
            .filter(|l| !strip(l).is_empty())
            .filter_map(|l| self.line(l).ok())
            .filter(|p| !p.is_empty())
            .collect()
    }

    /// Kokoro phoneme chunks of at most 510 symbols, split at sentence and clause boundaries.
    pub fn phonemize(&self, text: &str) -> Vec<String> {
        chunk(&self.phonemize_lines(text))
    }
}

fn load_lexicon(data: Vec<u8>) -> io::Result<Lexicon> {
    let bad = |m: &str| crate::st::bad(format!("lexicon.bin: {m}"));
    if data.len() < 12 || &data[..8] != b"KKG2P\x01\0\0" {
        return Err(bad("bad magic"));
    }
    let mut p = 8;
    let u32_at = |p: &mut usize| -> io::Result<u32> {
        let b = data.get(*p..*p + 4).ok_or_else(|| bad("truncated"))?;
        *p += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut sec: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
    for _ in 0..u32_at(&mut p)? {
        let n = u32_at(&mut p)?;
        let name = String::from_utf8_lossy(&data[p..p + n as usize]).into_owned();
        p += n as usize;
        let count = u32_at(&mut p)?;
        let mut v = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let n = u32_at(&mut p)?;
            if p + n as usize > data.len() {
                return Err(bad("truncated"));
            }
            v.push((p as u32, n));
            p += n as usize;
        }
        sec.insert(name, v);
    }
    let mut take = |k: &str| Strs { spans: sec.remove(k).unwrap_or_default() };
    let (keys, vals, skeys, svals) = (take("lex.k"), take("lex.v"), take("spacy.k"), take("spacy.v"));
    let (vocab_s, pair_s, single_s, pat_s) = (take("vocab"), take("nfc.pair"), take("nfc.single"), take("spacy.pat"));
    let text = |sp: (u32, u32)| String::from_utf8_lossy(&data[sp.0 as usize..(sp.0 + sp.1) as usize]).into_owned();
    let vocab = vocab_s.spans.iter().filter_map(|&s| text(s).chars().next()).collect();
    let nfc_pair = pair_s
        .spans
        .iter()
        .filter_map(|&s| {
            let c: Vec<char> = text(s).chars().collect();
            (c.len() >= 3).then(|| ((c[0], c[1]), c[2]))
        })
        .collect();
    let nfc_single = single_s
        .spans
        .iter()
        .filter_map(|&s| {
            let c: Vec<char> = text(s).chars().collect();
            (c.len() >= 2).then(|| (c[0], c[1]))
        })
        .collect();
    if pat_s.spans.len() != 4 {
        return Err(bad("missing spaCy patterns"));
    }
    let pats: Vec<Re> = pat_s.spans.iter().map(|&s| Re::new(&text(s))).collect();
    let [prefix, suffix, infix, url]: [Re; 4] = pats.try_into().ok().unwrap();
    let mut l = Lexicon {
        data,
        keys,
        vals,
        skeys,
        svals,
        vocab,
        nfc_pair,
        nfc_single,
        prefix,
        suffix,
        infix,
        url,
        matcher: HashMap::new(),
    };
    let mut matcher: HashMap<String, Vec<(Vec<String>, String)>> = HashMap::new();
    for &sk in &l.skeys.spans {
        let k = l.s(sk).to_string();
        let mut affix = k.contains(' ') || l.prefix.search(&k) || l.suffix.search(&k);
        find_iter(&l.infix, &k, |_| affix = true);
        if !affix {
            continue;
        }
        let mut spans = Vec::new();
        tokenize_spans(&l, &k, &mut spans, false);
        let toks: Vec<String> = spans.iter().map(|&(b, e)| k[b..e].to_string()).collect();
        if let Some(first) = toks.first().cloned() {
            matcher.entry(first).or_default().push((toks, k));
        }
    }
    l.matcher = matcher;
    Ok(l)
}

/// Chunks never exceed 510 Kokoro phonemes and are never empty. The first chunk is the first sentence alone (cut
/// at its first clause mark past 60 if longer than 150); later sentences pack greedily up to 400. A sentence over
/// 510 splits at its last clause mark past 127, else its last space, else a hard cut. Newlines end sentences.
pub fn chunk(lines: &[String]) -> Vec<String> {
    const MAX: usize = 510;
    const TARGET: usize = 400;
    const FIRST_MAX: usize = 150;
    const FIRST_MIN: usize = 60;
    type U = Vec<char>;
    let trim = |s: &[char]| -> U {
        let b = s.iter().position(|&c| c != ' ');
        match b {
            None => Vec::new(),
            Some(b) => s[b..=s.iter().rposition(|&c| c != ' ').unwrap()].to_vec(),
        }
    };
    let speakable = |s: &U| s.iter().any(|&c| c != ' ' && !PUNCT.contains(c));
    let pieces = |mut s: U, out: &mut Vec<U>| {
        while s.len() > MAX {
            let mut cut = 0;
            let mut i = MAX;
            while i > MAX / 4 && cut == 0 {
                i -= 1;
                if ",;:…".contains(s[i]) {
                    cut = i + 1;
                }
            }
            let mut i = MAX + 1;
            while i > 1 && cut == 0 {
                i -= 1;
                if s[i] == ' ' {
                    cut = i;
                }
            }
            if cut == 0 {
                cut = MAX;
            }
            let h = trim(&s[..cut]);
            if speakable(&h) {
                out.push(h);
            }
            s = trim(&s[cut..]);
        }
        if speakable(&s) {
            out.push(s);
        }
    };
    let mut sents: Vec<U> = Vec::new();
    for line in lines {
        let mut s = trim(&line.chars().collect::<U>());
        if s.is_empty() {
            continue;
        }
        if !PUNCT.contains(*s.last().unwrap()) {
            s.push('.');
        }
        let mut b0 = 0;
        for i in 0..s.len() {
            if i + 1 == s.len() || (".!?".contains(s[i]) && s[i + 1] == ' ') {
                let t = trim(&s[b0..=i]);
                if !t.is_empty() {
                    sents.push(t);
                }
                b0 = i + 1;
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur: U = Vec::new();
    for (k, mut s) in sents.into_iter().enumerate() {
        let mut atoms: Vec<U> = Vec::new();
        let cut_first = |s: &mut U, from: usize, marks: &str, atoms: &mut Vec<U>| {
            let mut i = from;
            while i + 1 < s.len() && i < MAX {
                if marks.contains(s[i]) {
                    let h = trim(&s[..=i]);
                    if speakable(&h) {
                        atoms.push(h);
                    }
                    *s = trim(&s[i + 1..]);
                    return true;
                }
                i += 1;
            }
            false
        };
        if k == 0 && s.len() > FIRST_MAX && !cut_first(&mut s, FIRST_MIN.max(1) - 1, ",;:…", &mut atoms) {
            cut_first(&mut s, (FIRST_MIN + FIRST_MAX) / 2, " ", &mut atoms);
        }
        pieces(s, &mut atoms);
        for a in atoms {
            if !cur.is_empty() && cur.len() + 1 + a.len() > TARGET {
                out.push(cur.iter().collect());
                cur.clear();
            }
            if cur.is_empty() {
                cur = a;
            } else {
                cur.push(' ');
                cur.extend(a);
            }
            if out.is_empty() {
                out.push(cur.iter().collect());
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur.iter().collect());
    }
    out
}

pub fn normalize(g: &G2p, text: &str) -> String {
    normalize_impl(&g.lex, text)
}
