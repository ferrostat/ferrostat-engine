//! `pdf2txt srcfile [, saving(destfile) replace nomsg]`: extract the text of
//! a PDF file.
//!
//! The reader handles classic cross-reference tables and PDF 1.5 object
//! streams, Flate/ASCIIHex/ASCII85 filters, the page tree (inherited
//! resources), simple fonts in WinAnsi encoding and fonts with a
//! `/ToUnicode` map. Lines and word spaces are rebuilt from the text
//! positions.

use std::collections::BTreeMap;
use std::io::Read;

use ferrostat_cmd::eval::rc_err;
use ferrostat_core::vfs;

use super::{has, opt_arg, with_ext, Opts};
use crate::macros::unquote;

pub fn command(src: &str, opts: &Opts) -> Result<String, String> {
    let src = if src.to_lowercase().ends_with(".pdf") || vfs::exists(src) { src.to_string() } else { format!("{src}.pdf") };
    let data = vfs::read(&src).map_err(|_| rc_err(601, format!("file {src} not found")))?;
    let text = extract(&data).map_err(|e| rc_err(610, format!("{src}: {e}")))?;
    let target = match opt_arg(opts, "saving", 3) {
        Some(s) if !s.trim().is_empty() => {
            let s = unquote(s.split(',').next().unwrap_or("").trim());
            if s.rsplit(['/', '\\']).next().unwrap_or(&s).contains('.') { s } else { format!("{s}.txt") }
        }
        _ => with_ext(&src, "txt"),
    };
    let replace = has(opts, "replace", 3) || opt_arg(opts, "saving", 3).is_some_and(|s| s.contains("replace"));
    if vfs::exists(&target) && !replace {
        return Err(rc_err(602, format!("file {target} already exists")));
    }
    vfs::write(&target, text.as_bytes()).map_err(|e| rc_err(603, format!("could not write {target}: {e}")))?;
    if has(opts, "nomsg", 5) {
        return Ok(String::new());
    }
    Ok(format!("successfully converted \"{}\"", crate::putdocx::full_path(&target)))
}

// ---------------------------------------------------------------------------
// PDF objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Num(f64),
    Str(Vec<u8>),
    Name(String),
    Arr(Vec<Obj>),
    Dict(Vec<(String, Obj)>),
    Ref(u32),
    /// Content-stream operator (or an unknown keyword).
    Op(String),
}

impl Obj {
    fn get(&self, k: &str) -> Option<&Obj> {
        match self {
            Obj::Dict(d) => d.iter().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        if let Obj::Num(x) = self { Some(*x) } else { None }
    }
    fn name(&self) -> Option<&str> {
        if let Obj::Name(x) = self { Some(x) } else { None }
    }
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n' | 0 | 12)
}

fn is_delim(c: u8) -> bool {
    matches!(c, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

/// A tokenizer/parser over PDF syntax.
pub struct Lexer<'a> {
    pub b: &'a [u8],
    pub p: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Lexer { b, p: 0 }
    }

    fn skip_ws(&mut self) {
        while self.p < self.b.len() {
            let c = self.b[self.p];
            if is_ws(c) {
                self.p += 1;
            } else if c == b'%' {
                while self.p < self.b.len() && self.b[self.p] != b'\n' && self.b[self.p] != b'\r' {
                    self.p += 1;
                }
            } else {
                break;
            }
        }
    }

    fn literal(&mut self) -> Vec<u8> {
        // after '('
        let mut out = Vec::new();
        let mut depth = 1;
        while self.p < self.b.len() {
            let c = self.b[self.p];
            self.p += 1;
            match c {
                b'\\' => {
                    let Some(&e) = self.b.get(self.p) else { break };
                    self.p += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'\r' => {
                            if self.b.get(self.p) == Some(&b'\n') {
                                self.p += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut v = (e - b'0') as u32;
                            for _ in 0..2 {
                                match self.b.get(self.p) {
                                    Some(&d) if (b'0'..=b'7').contains(&d) => {
                                        v = v * 8 + (d - b'0') as u32;
                                        self.p += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        e => out.push(e),
                    }
                }
                b'(' => {
                    depth += 1;
                    out.push(c);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(c);
                }
                c => out.push(c),
            }
        }
        out
    }

    fn hex(&mut self) -> Vec<u8> {
        // after '<'
        let mut digits = Vec::new();
        while self.p < self.b.len() && self.b[self.p] != b'>' {
            let c = self.b[self.p];
            if c.is_ascii_hexdigit() {
                digits.push(c);
            }
            self.p += 1;
        }
        self.p += 1;
        if digits.len() % 2 == 1 {
            digits.push(b'0');
        }
        digits.chunks(2).map(|d| u8::from_str_radix(std::str::from_utf8(d).unwrap_or("00"), 16).unwrap_or(0)).collect()
    }

    fn word(&mut self) -> String {
        let s = self.p;
        while self.p < self.b.len() && !is_ws(self.b[self.p]) && !is_delim(self.b[self.p]) {
            self.p += 1;
        }
        String::from_utf8_lossy(&self.b[s..self.p]).to_string()
    }

    /// Next object; `None` at the end or at a closing delimiter.
    pub fn next(&mut self) -> Option<Obj> {
        self.skip_ws();
        let c = *self.b.get(self.p)?;
        match c {
            b'(' => {
                self.p += 1;
                Some(Obj::Str(self.literal()))
            }
            b'<' if self.b.get(self.p + 1) == Some(&b'<') => {
                self.p += 2;
                let mut d = Vec::new();
                loop {
                    self.skip_ws();
                    if self.p >= self.b.len() {
                        break;
                    }
                    if self.b[self.p..].starts_with(b">>") {
                        self.p += 2;
                        break;
                    }
                    match self.next() {
                        Some(Obj::Name(k)) => {
                            let v = self.next().unwrap_or(Obj::Null);
                            d.push((k, v));
                        }
                        Some(_) => {}
                        None => {
                            self.p += 1;
                        }
                    }
                }
                Some(Obj::Dict(d))
            }
            b'<' => {
                self.p += 1;
                Some(Obj::Str(self.hex()))
            }
            b'[' => {
                self.p += 1;
                let mut a = Vec::new();
                loop {
                    self.skip_ws();
                    match self.b.get(self.p) {
                        None => break,
                        Some(b']') => {
                            self.p += 1;
                            break;
                        }
                        _ => {}
                    }
                    match self.next() {
                        Some(Obj::Op(o)) if o == "R" => {
                            // `n g R` inside an array
                            if a.len() >= 2 {
                                a.pop();
                                if let Some(Obj::Num(n)) = a.pop() {
                                    a.push(Obj::Ref(n as u32));
                                }
                            }
                        }
                        Some(o) => a.push(o),
                        None => self.p += 1,
                    }
                }
                Some(Obj::Arr(a))
            }
            b'/' => {
                self.p += 1;
                let w = self.word();
                // #xx escapes in names
                let mut n = String::new();
                let wb = w.as_bytes();
                let mut i = 0;
                while i < wb.len() {
                    if wb[i] == b'#' && i + 2 < wb.len() {
                        if let Ok(v) = u8::from_str_radix(&w[i + 1..i + 3], 16) {
                            n.push(v as char);
                            i += 3;
                            continue;
                        }
                    }
                    n.push(wb[i] as char);
                    i += 1;
                }
                Some(Obj::Name(n))
            }
            b']' | b'>' | b')' | b'{' | b'}' => {
                self.p += 1;
                Some(Obj::Op(String::new()))
            }
            _ => {
                let w = self.word();
                if w.is_empty() {
                    self.p += 1;
                    return Some(Obj::Op(String::new()));
                }
                if let Ok(x) = w.parse::<f64>() {
                    // `n g R` reference?
                    let save = self.p;
                    if w.bytes().all(|c| c.is_ascii_digit()) {
                        self.skip_ws();
                        let s2 = self.p;
                        let g = self.word();
                        if !g.is_empty() && g.bytes().all(|c| c.is_ascii_digit()) {
                            self.skip_ws();
                            if self.b.get(self.p) == Some(&b'R') && self.b.get(self.p + 1).is_none_or(|c| is_ws(*c) || is_delim(*c)) {
                                self.p += 1;
                                return Some(Obj::Ref(x as u32));
                            }
                        }
                        let _ = s2;
                        self.p = save;
                    }
                    return Some(Obj::Num(x));
                }
                if w.starts_with('.') || w.starts_with('-') || w.starts_with('+') {
                    return Some(Obj::Num(w.trim_start_matches('+').parse().unwrap_or(0.0)));
                }
                Some(match w.as_str() {
                    "true" => Obj::Bool(true),
                    "false" => Obj::Bool(false),
                    "null" => Obj::Null,
                    _ => Obj::Op(w),
                })
            }
        }
    }
}

/// An indirect object: its value and the raw (still encoded) stream.
#[derive(Debug, Clone)]
struct Indirect {
    obj: Obj,
    stream: Option<Vec<u8>>,
}

struct Doc {
    objs: BTreeMap<u32, Indirect>,
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

impl Doc {
    fn parse(data: &[u8]) -> Result<Doc, String> {
        if find(data, b"%PDF", 0).is_none_or(|p| p > 1024) {
            return Err("not a PDF file".into());
        }
        let mut objs: BTreeMap<u32, Indirect> = BTreeMap::new();
        // scan for `n g obj`
        let mut i = 0;
        while let Some(k) = find(data, b"obj", i) {
            i = k + 3;
            // must be followed by a delimiter and preceded by `n g `
            if data.get(k + 3).is_some_and(|c| !is_ws(*c) && !is_delim(*c)) {
                continue;
            }
            let mut j = k;
            let back_num = |j: &mut usize| -> Option<u32> {
                while *j > 0 && is_ws(data[*j - 1]) {
                    *j -= 1;
                }
                let e = *j;
                while *j > 0 && data[*j - 1].is_ascii_digit() {
                    *j -= 1;
                }
                if *j == e {
                    return None;
                }
                std::str::from_utf8(&data[*j..e]).ok()?.parse().ok()
            };
            if back_num(&mut j).is_none() {
                continue;
            }
            let Some(num) = back_num(&mut j) else { continue };
            if j > 0 && !is_ws(data[j - 1]) && !is_delim(data[j - 1]) {
                continue;
            }
            let mut lx = Lexer::new(data);
            lx.p = k + 3;
            let obj = lx.next().unwrap_or(Obj::Null);
            lx.skip_ws();
            let mut stream = None;
            if data[lx.p..].starts_with(b"stream") {
                let mut s = lx.p + 6;
                if data.get(s) == Some(&b'\r') {
                    s += 1;
                }
                if data.get(s) == Some(&b'\n') {
                    s += 1;
                }
                let declared = obj.get("Length").and_then(Obj::num).map(|x| x as usize);
                let end = match declared {
                    Some(n) if s + n <= data.len() && find(&data[s + n..(s + n + 20).min(data.len())], b"endstream", 0).is_some() => s + n,
                    _ => {
                        let mut e = find(data, b"endstream", s).unwrap_or(data.len());
                        // drop the end-of-line before `endstream`
                        if e > s && data[e - 1] == b'\n' {
                            e -= 1;
                        }
                        if e > s && data[e - 1] == b'\r' {
                            e -= 1;
                        }
                        e
                    }
                };
                stream = Some(data[s..end].to_vec());
                i = end;
            } else {
                i = lx.p.max(i);
            }
            // later definitions (incremental updates) replace earlier ones
            objs.insert(num, Indirect { obj, stream });
        }
        let mut doc = Doc { objs };
        doc.expand_object_streams();
        Ok(doc)
    }

    /// Unpack the objects stored in `/Type /ObjStm` streams.
    fn expand_object_streams(&mut self) {
        let streams: Vec<(Obj, Vec<u8>)> = self
            .objs
            .values()
            .filter(|o| o.obj.get("Type").and_then(Obj::name) == Some("ObjStm"))
            .filter_map(|o| o.stream.clone().map(|s| (o.obj.clone(), s)))
            .collect();
        for (dict, raw) in streams {
            let Some(data) = self.decode(&dict, &raw) else { continue };
            let n = dict.get("N").and_then(Obj::num).unwrap_or(0.0) as usize;
            let first = dict.get("First").and_then(Obj::num).unwrap_or(0.0) as usize;
            let mut lx = Lexer::new(&data);
            let mut pairs = Vec::new();
            for _ in 0..n {
                let (Some(Obj::Num(a)), Some(Obj::Num(b))) = (lx.next(), lx.next()) else { break };
                pairs.push((a as u32, b as usize));
            }
            for (num, off) in pairs {
                if self.objs.contains_key(&num) {
                    continue;
                }
                let mut lx = Lexer::new(&data);
                lx.p = first + off;
                if lx.p >= data.len() {
                    continue;
                }
                if let Some(o) = lx.next() {
                    self.objs.insert(num, Indirect { obj: o, stream: None });
                }
            }
        }
    }

    fn resolve<'b>(&'b self, o: &'b Obj) -> &'b Obj {
        let mut cur = o;
        for _ in 0..16 {
            match cur {
                Obj::Ref(n) => match self.objs.get(n) {
                    Some(i) => cur = &i.obj,
                    None => return &Obj::Null,
                },
                _ => return cur,
            }
        }
        cur
    }

    fn decode(&self, dict: &Obj, raw: &[u8]) -> Option<Vec<u8>> {
        let filters: Vec<String> = match dict.get("Filter").map(|f| self.resolve(f)) {
            Some(Obj::Name(n)) => vec![n.clone()],
            Some(Obj::Arr(a)) => a.iter().filter_map(|x| self.resolve(x).name().map(String::from)).collect(),
            _ => vec![],
        };
        let parms: Vec<Option<Obj>> = match dict.get("DecodeParms").map(|f| self.resolve(f)) {
            Some(Obj::Arr(a)) => a.iter().map(|x| Some(self.resolve(x).clone())).collect(),
            Some(d @ Obj::Dict(_)) => vec![Some(d.clone())],
            _ => vec![],
        };
        let mut data = raw.to_vec();
        for (k, f) in filters.iter().enumerate() {
            data = match f.as_str() {
                "FlateDecode" | "Fl" => {
                    let d = inflate(&data)?;
                    match parms.get(k).cloned().flatten() {
                        Some(p) => unpredict(d, &p),
                        None => d,
                    }
                }
                "ASCIIHexDecode" | "AHx" => {
                    let mut lx = Lexer::new(&data);
                    lx.hex()
                }
                "ASCII85Decode" | "A85" => ascii85(&data),
                _ => return None,
            };
        }
        Some(data)
    }

    fn stream_of(&self, o: &Obj) -> Option<Vec<u8>> {
        let Obj::Ref(n) = o else { return None };
        let ind = self.objs.get(n)?;
        let raw = ind.stream.as_ref()?;
        let mut dict = ind.obj.clone();
        // an indirect /Length was resolved when the stream was located
        if let Obj::Dict(d) = &mut dict {
            for (_, v) in d.iter_mut() {
                if let Obj::Ref(_) = v {
                    *v = self.resolve(v).clone();
                }
            }
        }
        self.decode(&dict, raw)
    }

    /// Pages in document order, with their inherited resources.
    fn pages(&self) -> Vec<(Obj, Obj)> {
        let mut out = Vec::new();
        let root = self.objs.values().find(|o| o.obj.get("Type").and_then(Obj::name) == Some("Catalog")).map(|o| o.obj.clone());
        if let Some(pages) = root.as_ref().and_then(|r| r.get("Pages")) {
            self.walk(pages, &Obj::Null, &mut out, 0);
        }
        if out.is_empty() {
            for o in self.objs.values() {
                if o.obj.get("Type").and_then(Obj::name) == Some("Page") {
                    let res = o.obj.get("Resources").map(|r| self.resolve(r).clone()).unwrap_or(Obj::Null);
                    out.push((o.obj.clone(), res));
                }
            }
        }
        out
    }

    fn walk(&self, node: &Obj, res: &Obj, out: &mut Vec<(Obj, Obj)>, depth: usize) {
        if depth > 64 {
            return;
        }
        let n = self.resolve(node);
        let res = n.get("Resources").map(|r| self.resolve(r).clone()).unwrap_or_else(|| res.clone());
        match n.get("Type").and_then(Obj::name) {
            Some("Pages") => {
                if let Some(Obj::Arr(kids)) = n.get("Kids").map(|k| self.resolve(k)) {
                    for k in kids {
                        self.walk(k, &res, out, depth + 1);
                    }
                }
            }
            _ => {
                if n.get("Contents").is_some() || n.get("Type").and_then(Obj::name) == Some("Page") {
                    out.push((n.clone(), res));
                }
            }
        }
    }
}

fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut z = flate2::read::ZlibDecoder::new(data);
    if z.read_to_end(&mut out).is_ok() {
        return Some(out);
    }
    // truncated streams: keep what was decoded, else try raw deflate
    if !out.is_empty() {
        return Some(out);
    }
    let mut out = Vec::new();
    let mut d = flate2::read::DeflateDecoder::new(data);
    match d.read_to_end(&mut out) {
        Ok(_) => Some(out),
        Err(_) if !out.is_empty() => Some(out),
        Err(_) => None,
    }
}

/// Undo PNG predictors (used by cross-reference and object streams).
fn unpredict(data: Vec<u8>, parms: &Obj) -> Vec<u8> {
    let pred = parms.get("Predictor").and_then(Obj::num).unwrap_or(1.0) as u32;
    if pred < 10 {
        return data;
    }
    let cols = parms.get("Columns").and_then(Obj::num).unwrap_or(1.0) as usize;
    let colors = parms.get("Colors").and_then(Obj::num).unwrap_or(1.0) as usize;
    let bpc = parms.get("BitsPerComponent").and_then(Obj::num).unwrap_or(8.0) as usize;
    let bpp = (colors * bpc).div_ceil(8).max(1);
    let row = (cols * colors * bpc).div_ceil(8);
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        if chunk.len() < row + 1 {
            break;
        }
        let ft = chunk[0];
        let mut cur = chunk[1..].to_vec();
        for i in 0..row {
            let a = if i >= bpp { cur[i - bpp] as i32 } else { 0 };
            let b = prev[i] as i32;
            let c = if i >= bpp { prev[i - bpp] as i32 } else { 0 };
            let add = match ft {
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
                }
                _ => 0,
            };
            cur[i] = (cur[i] as i32 + add) as u8;
        }
        out.extend_from_slice(&cur);
        prev = cur;
    }
    out
}

fn ascii85(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut group = Vec::with_capacity(5);
    let body = match find(data, b"~>", 0) {
        Some(e) => &data[..e],
        None => data,
    };
    let body = body.strip_prefix(b"<~").unwrap_or(body);
    for &c in body {
        if is_ws(c) {
            continue;
        }
        if c == b'z' && group.is_empty() {
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        if !(b'!'..=b'u').contains(&c) {
            continue;
        }
        group.push(c - b'!');
        if group.len() == 5 {
            let v = group.iter().fold(0u64, |a, &d| a * 85 + d as u64) as u32;
            out.extend_from_slice(&v.to_be_bytes());
            group.clear();
        }
    }
    if !group.is_empty() {
        let n = group.len();
        while group.len() < 5 {
            group.push(84);
        }
        let v = group.iter().fold(0u64, |a, &d| a * 85 + d as u64) as u32;
        out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
    }
    out
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

/// Characters 128-159 of WinAnsiEncoding (the rest is Latin-1).
const WIN_ANSI_HIGH: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}', '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
];

#[derive(Debug, Clone, Default)]
struct Font {
    /// code -> text from `/ToUnicode`
    map: BTreeMap<u32, String>,
    /// bytes per character code
    width: usize,
}

impl Font {
    fn decode(&self, s: &[u8]) -> String {
        if self.map.is_empty() && self.width == 1 {
            return s.iter().map(|&b| if (128..160).contains(&b) { WIN_ANSI_HIGH[(b - 128) as usize] } else { b as char }).collect();
        }
        let w = self.width.max(1);
        let mut out = String::new();
        for ch in s.chunks(w) {
            let code = ch.iter().fold(0u32, |a, &b| a << 8 | b as u32);
            match self.map.get(&code) {
                Some(t) => out.push_str(t),
                None if w == 1 => out.push(if (128..160).contains(&code) { WIN_ANSI_HIGH[(code - 128) as usize] } else { char::from_u32(code).unwrap_or('?') }),
                None => out.push(char::from_u32(code).filter(|c| !c.is_control()).unwrap_or('\u{fffd}')),
            }
        }
        out
    }
}

fn utf16be(b: &[u8]) -> String {
    let u: Vec<u16> = b.chunks(2).map(|c| (c[0] as u16) << 8 | *c.get(1).unwrap_or(&0) as u16).collect();
    String::from_utf16_lossy(&u)
}

fn be(b: &[u8]) -> u32 {
    b.iter().fold(0u32, |a, &x| a << 8 | x as u32)
}

/// Parse a `/ToUnicode` CMap.
fn parse_cmap(data: &[u8], font: &mut Font) {
    let mut lx = Lexer::new(data);
    let mut stack: Vec<Obj> = Vec::new();
    let mut mode = "";
    while let Some(o) = lx.next() {
        match &o {
            Obj::Op(op) => match op.as_str() {
                "begincodespacerange" => {
                    mode = "cs";
                    stack.clear();
                }
                "beginbfchar" => {
                    mode = "char";
                    stack.clear();
                }
                "beginbfrange" => {
                    mode = "range";
                    stack.clear();
                }
                "endcodespacerange" => {
                    if let Some(Obj::Str(s)) = stack.first() {
                        font.width = s.len().max(1);
                    }
                    stack.clear();
                    mode = "";
                }
                "endbfchar" => {
                    for pair in stack.chunks(2) {
                        if let [Obj::Str(c), Obj::Str(t)] = pair {
                            font.map.insert(be(c), utf16be(t));
                        }
                    }
                    stack.clear();
                    mode = "";
                }
                "endbfrange" => {
                    for tri in stack.chunks(3) {
                        match tri {
                            [Obj::Str(a), Obj::Str(b), Obj::Str(t)] => {
                                let (lo, hi) = (be(a), be(b));
                                let base = utf16be(t);
                                let mut chars: Vec<char> = base.chars().collect();
                                for code in lo..=hi.min(lo + 65535) {
                                    font.map.insert(code, chars.iter().collect());
                                    if let Some(l) = chars.last_mut() {
                                        *l = char::from_u32(*l as u32 + 1).unwrap_or(*l);
                                    }
                                }
                            }
                            [Obj::Str(a), Obj::Str(_), Obj::Arr(list)] => {
                                let lo = be(a);
                                for (k, t) in list.iter().enumerate() {
                                    if let Obj::Str(t) = t {
                                        font.map.insert(lo + k as u32, utf16be(t));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    stack.clear();
                    mode = "";
                }
                _ => {}
            },
            _ if !mode.is_empty() => stack.push(o),
            _ => {}
        }
    }
}

fn load_fonts(doc: &Doc, res: &Obj) -> BTreeMap<String, Font> {
    let mut fonts = BTreeMap::new();
    let Some(Obj::Dict(fd)) = res.get("Font").map(|f| doc.resolve(f)) else { return fonts };
    for (name, f) in fd {
        let fobj = doc.resolve(f);
        let mut font = Font { map: BTreeMap::new(), width: 1 };
        if fobj.get("Subtype").and_then(Obj::name) == Some("Type0") {
            font.width = 2;
        }
        if let Some(tu) = fobj.get("ToUnicode") {
            if let Some(data) = doc.stream_of(tu) {
                let w = font.width;
                parse_cmap(&data, &mut font);
                if font.width == 0 {
                    font.width = w;
                }
            }
        }
        fonts.insert(name.clone(), font);
    }
    fonts
}

// ---------------------------------------------------------------------------
// Content streams
// ---------------------------------------------------------------------------

/// A piece of text placed on the page.
#[derive(Debug, Clone)]
struct Frag {
    x: f64,
    y: f64,
    /// estimated end of the text on the x axis
    x_end: f64,
    size: f64,
    text: String,
}

fn char_em(c: char) -> f64 {
    match c {
        ' ' | '.' | ',' | ':' | ';' | '!' | '\'' | '|' | 'i' | 'j' | 'l' | 'I' => 0.278,
        'f' | 't' | 'r' | '(' | ')' | '[' | ']' | '-' | '/' => 0.333,
        'm' | 'M' | 'W' | '%' | '@' => 0.833,
        'w' => 0.722,
        'A'..='Z' => 0.667,
        _ => 0.556,
    }
}

type M = [f64; 6];

fn mul(a: &M, b: &M) -> M {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

const ID: M = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn page_text(doc: &Doc, page: &Obj, res: &Obj) -> String {
    let mut content = Vec::new();
    match page.get("Contents") {
        Some(Obj::Arr(a)) => {
            for c in a {
                if let Some(d) = doc.stream_of(c) {
                    content.extend_from_slice(&d);
                    content.push(b'\n');
                }
            }
        }
        Some(r @ Obj::Ref(_)) => match doc.resolve(r) {
            Obj::Arr(a) => {
                for c in a {
                    if let Some(d) = doc.stream_of(c) {
                        content.extend_from_slice(&d);
                        content.push(b'\n');
                    }
                }
            }
            _ => {
                if let Some(d) = doc.stream_of(r) {
                    content = d;
                }
            }
        },
        _ => {}
    }
    let fonts = load_fonts(doc, res);
    let mut frags = Vec::new();
    run_content(doc, &content, res, &fonts, ID, &mut frags, 0);
    layout(&frags)
}

fn run_content(doc: &Doc, content: &[u8], res: &Obj, fonts: &BTreeMap<String, Font>, ctm0: M, frags: &mut Vec<Frag>, depth: usize) {
    let mut lx = Lexer::new(content);
    let mut ops: Vec<Obj> = Vec::new();
    let mut ctm = ctm0;
    let mut stack: Vec<M> = Vec::new();
    let (mut tm, mut tlm) = (ID, ID);
    let (mut size, mut leading, mut tc, mut tw, mut th) = (12.0f64, 0.0f64, 0.0f64, 0.0f64, 1.0f64);
    let default_font = Font { map: BTreeMap::new(), width: 1 };
    let mut font = default_font.clone();
    let num = |ops: &[Obj], k: usize| -> f64 { ops.get(k).and_then(Obj::num).unwrap_or(0.0) };
    while let Some(o) = lx.next() {
        let Obj::Op(op) = o else {
            ops.push(o);
            continue;
        };
        match op.as_str() {
            "q" => stack.push(ctm),
            "Q" => ctm = stack.pop().unwrap_or(ctm0),
            "cm" if ops.len() >= 6 => {
                let m = [num(&ops, ops.len() - 6), num(&ops, ops.len() - 5), num(&ops, ops.len() - 4), num(&ops, ops.len() - 3), num(&ops, ops.len() - 2), num(&ops, ops.len() - 1)];
                ctm = mul(&m, &ctm);
            }
            "BT" => {
                tm = ID;
                tlm = ID;
            }
            "Tf" if ops.len() >= 2 => {
                size = num(&ops, ops.len() - 1);
                if let Some(n) = ops[ops.len() - 2].name() {
                    font = fonts.get(n).cloned().unwrap_or_else(|| default_font.clone());
                }
            }
            "TL" => leading = num(&ops, ops.len().saturating_sub(1)),
            "Tc" => tc = num(&ops, ops.len().saturating_sub(1)),
            "Tw" => tw = num(&ops, ops.len().saturating_sub(1)),
            "Tz" => th = num(&ops, ops.len().saturating_sub(1)) / 100.0,
            "Td" | "TD" if ops.len() >= 2 => {
                let (tx, ty) = (num(&ops, ops.len() - 2), num(&ops, ops.len() - 1));
                if op == "TD" {
                    leading = -ty;
                }
                tlm = mul(&[1.0, 0.0, 0.0, 1.0, tx, ty], &tlm);
                tm = tlm;
            }
            "Tm" if ops.len() >= 6 => {
                let k = ops.len() - 6;
                tlm = [num(&ops, k), num(&ops, k + 1), num(&ops, k + 2), num(&ops, k + 3), num(&ops, k + 4), num(&ops, k + 5)];
                tm = tlm;
            }
            "T*" => {
                tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -leading], &tlm);
                tm = tlm;
            }
            "Tj" | "'" | "\"" | "TJ" => {
                if op == "'" || op == "\"" {
                    if op == "\"" && ops.len() >= 3 {
                        tw = num(&ops, ops.len() - 3);
                        tc = num(&ops, ops.len() - 2);
                    }
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -leading], &tlm);
                    tm = tlm;
                }
                let items: Vec<Obj> = match ops.last() {
                    Some(Obj::Arr(a)) if op == "TJ" => a.clone(),
                    Some(s @ Obj::Str(_)) => vec![s.clone()],
                    _ => vec![],
                };
                let mut text = String::new();
                let trm = mul(&tm, &ctm);
                let scale = (trm[2] * trm[2] + trm[3] * trm[3]).sqrt().max(1e-9);
                let eff = size * scale;
                let (x0, y0) = (trm[4], trm[5]);
                let mut adv = 0.0; // in unscaled text space units
                for it in items {
                    match it {
                        Obj::Str(s) => {
                            let t = font.decode(&s);
                            for c in t.chars() {
                                adv += (char_em(c) * size + tc + if c == ' ' { tw } else { 0.0 }) * th;
                            }
                            text.push_str(&t);
                        }
                        Obj::Num(k) => {
                            adv -= k / 1000.0 * size * th;
                            // a large negative kern is a word gap
                            if k < -180.0 && !text.ends_with(' ') && !text.is_empty() {
                                text.push(' ');
                            }
                        }
                        _ => {}
                    }
                }
                let hscale = (trm[0] * trm[0] + trm[1] * trm[1]).sqrt();
                if !text.is_empty() {
                    frags.push(Frag { x: x0, y: y0, x_end: x0 + adv * hscale / size.max(1e-9) * size, size: eff, text });
                }
                tm = mul(&[1.0, 0.0, 0.0, 1.0, adv, 0.0], &tm);
            }
            "Do" if depth < 8 => {
                // text inside form XObjects
                if let Some(n) = ops.last().and_then(Obj::name) {
                    if let Some(Obj::Dict(xd)) = res.get("XObject").map(|x| doc.resolve(x)) {
                        if let Some((_, r)) = xd.iter().find(|(k, _)| k == n) {
                            let xo = doc.resolve(r);
                            if xo.get("Subtype").and_then(Obj::name) == Some("Form") {
                                if let Some(data) = doc.stream_of(r) {
                                    let xres = xo.get("Resources").map(|x| doc.resolve(x).clone()).unwrap_or_else(|| res.clone());
                                    let xf = load_fonts(doc, &xres);
                                    let m = match xo.get("Matrix") {
                                        Some(Obj::Arr(a)) if a.len() == 6 => [0, 1, 2, 3, 4, 5].map(|i| a[i].num().unwrap_or(0.0)),
                                        _ => ID,
                                    };
                                    run_content(doc, &data, &xres, &xf, mul(&m, &ctm), frags, depth + 1);
                                }
                            }
                        }
                    }
                }
            }
            "BI" => {
                // skip inline image data
                if let Some(e) = find(content, b"EI", lx.p) {
                    lx.p = e + 2;
                }
            }
            _ => {}
        }
        ops.clear();
    }
}

/// Join fragments into lines: a new line when the baseline moves, a space
/// when there is a gap between pieces of text.
fn layout(frags: &[Frag]) -> String {
    let mut out = String::new();
    let mut prev: Option<&Frag> = None;
    for f in frags {
        if let Some(p) = prev {
            let tol = p.size.min(f.size).max(1.0) * 0.4;
            if (f.y - p.y).abs() > tol || f.x + tol < p.x - p.size * 2.0 {
                out.push('\n');
                // a blank line for paragraph-sized gaps
                if p.y - f.y > p.size.max(f.size) * 1.9 {
                    out.push('\n');
                }
            } else if f.x - p.x_end > f.size * 0.12 && !out.ends_with(' ') && !f.text.starts_with(' ') {
                out.push(' ');
            }
        }
        out.push_str(&f.text);
        prev = Some(f);
    }
    out
}

/// Text of every page; pages are separated by form feeds.
pub fn extract(data: &[u8]) -> Result<String, String> {
    let doc = Doc::parse(data)?;
    if doc.objs.values().any(|o| o.obj.get("Encrypt").is_some()) || doc.objs.values().any(|o| o.obj.get("Filter").and_then(Obj::name) == Some("Standard")) {
        return Err("encrypted PDF files are not supported".into());
    }
    let pages = doc.pages();
    if pages.is_empty() {
        return Err("no pages found".into());
    }
    let texts: Vec<String> = pages.iter().map(|(p, r)| page_text(&doc, p, r)).collect();
    let mut s = texts.join("\n\u{c}");
    s = s.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    Ok(s.trim_end().to_string() + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_basics() {
        let mut lx = Lexer::new(b"<< /A 1 0 R /B [1 2 0 R (x\\(y\\)) <414> ] /C /N#20m >>");
        let o = lx.next().unwrap();
        assert_eq!(o.get("A"), Some(&Obj::Ref(1)));
        assert_eq!(o.get("B"), Some(&Obj::Arr(vec![Obj::Num(1.0), Obj::Ref(2), Obj::Str(b"x(y)".to_vec()), Obj::Str(vec![0x41, 0x40])])));
        assert_eq!(o.get("C"), Some(&Obj::Name("N m".into())));
    }

    #[test]
    fn a85() {
        assert_eq!(ascii85(b"<~9jqo^~>"), b"Man ");
    }

    #[test]
    fn simple_pdf() {
        let content = b"BT /F1 12 Tf 72 700 Td (Hello) Tj 40 0 Td (world) Tj ET BT /F1 12 Tf 72 680 Td [(Next)-250(line)] TJ ET";
        let mut pdf = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n".to_vec();
        pdf.extend_from_slice(format!("4 0 obj << /Length {} >>\nstream\n", content.len()).as_bytes());
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream\nendobj\n5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n");
        let t = extract(&pdf).unwrap();
        assert_eq!(t, "Hello world\nNext line\n");
    }
}
