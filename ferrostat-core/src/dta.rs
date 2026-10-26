//! Reading and writing .dta files.
//!
//! Implemented from the publicly documented file format:
//! - releases 117, 118, 119 (tagged XML-like layout), read and write;
//! - releases 113, 114, 115 (binary layout), read and write (115).
//!
//! Sections of the tagged formats are located through the `<map>` offsets,
//! never by searching the payload for tag bytes.

use std::io::{self, Write};
use std::path::Path;

use crate::dataset::DataSet;
use crate::missing;
use crate::value_label::ValueLabel;
use crate::var::{Value, VarType, Variable};

/// Root tags of the XML-style .dta formats (117-119); they are part of the
/// file format and must be written verbatim.
pub const FORMAT_OPEN: &[u8] = b"<stata_dta><header><release>";
pub const FORMAT_CLOSE: &[u8] = b"</stata_dta>";
const FORMAT_ROOT_LEN: usize = 11;

/// Error type for DTA I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum DtaError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid DTA file: {0}")]
    InvalidFormat(String),
    #[error("Unsupported DTA release: {0}")]
    UnsupportedRelease(u16),
    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

type Result<T> = std::result::Result<T, DtaError>;

fn bad(msg: impl Into<String>) -> DtaError {
    DtaError::InvalidFormat(msg.into())
}

// ---------------------------------------------------------------------------
// Byte cursor
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    big: bool,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, big: false }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(bad(format!("unexpected end of file at byte {}", self.pos)));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn expect(&mut self, tag: &[u8]) -> Result<()> {
        let got = self.take(tag.len())?;
        if got != tag {
            return Err(bad(format!(
                "expected {} at byte {}",
                String::from_utf8_lossy(tag),
                self.pos - tag.len()
            )));
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn uint(&mut self, n: usize) -> Result<u64> {
        let b = self.take(n)?;
        let mut v: u64 = 0;
        if self.big {
            for &x in b {
                v = (v << 8) | x as u64;
            }
        } else {
            for &x in b.iter().rev() {
                v = (v << 8) | x as u64;
            }
        }
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(self.uint(2)? as u16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(self.uint(4)? as u32)
    }
    fn u64(&mut self) -> Result<u64> {
        self.uint(8)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Fixed-width, NUL-padded string field.
    fn fixed_str(&mut self, n: usize, utf8: bool) -> Result<String> {
        Ok(decode_str(self.take(n)?, utf8))
    }
}

fn decode_str(b: &[u8], utf8: bool) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    let b = &b[..end];
    if utf8 {
        String::from_utf8_lossy(b).into_owned()
    } else {
        b.iter().map(|&c| c as char).collect()
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Layout constants that differ by release.
#[derive(Clone, Copy)]
struct Layout {
    name_w: usize,
    fmt_w: usize,
    lblname_w: usize,
    varlbl_w: usize,
    datalbl_w: usize,
    utf8: bool,
}

impl Layout {
    fn for_release(r: u16) -> Result<Self> {
        let (name_w, fmt_w, lblname_w, varlbl_w, utf8) = match r {
            113 => (33, 12, 33, 81, false),
            114 | 115 => (33, 49, 33, 81, false),
            117 => (33, 49, 33, 81, false),
            118 | 119 => (129, 57, 129, 321, true),
            _ => return Err(DtaError::UnsupportedRelease(r)),
        };
        Ok(Self { name_w, fmt_w, lblname_w, varlbl_w, datalbl_w: 81, utf8 })
    }
}

pub struct DtaReader;

impl DtaReader {
    /// Read a .dta file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<DataSet> {
        let data = crate::vfs::read(&path.as_ref().to_string_lossy())?;
        Self::from_bytes(data)
    }

    /// Parse a .dta from memory.
    pub fn from_bytes(data: Vec<u8>) -> Result<DataSet> {
        if data.starts_with(&FORMAT_OPEN[..FORMAT_ROOT_LEN]) {
            read_tagged(&data)
        } else if !data.is_empty() && matches!(data[0], 113 | 114 | 115) {
            read_binary(&data)
        } else if data.is_empty() {
            Err(bad("empty file"))
        } else {
            Err(bad("not a .dta file"))
        }
    }
}

fn read_tagged(data: &[u8]) -> Result<DataSet> {
    let mut c = Cursor::new(data);
    c.expect(FORMAT_OPEN)?;
    let rel: u16 = std::str::from_utf8(c.take(3)?)?.parse().map_err(|_| bad("bad release"))?;
    if !matches!(rel, 117 | 118 | 119) {
        return Err(DtaError::UnsupportedRelease(rel));
    }
    let lay = Layout::for_release(rel)?;
    c.expect(b"</release><byteorder>")?;
    c.big = match c.take(3)? {
        b"MSF" => true,
        b"LSF" => false,
        _ => return Err(bad("bad byteorder")),
    };
    c.expect(b"</byteorder><K>")?;
    let k = if rel == 119 { c.u32()? as usize } else { c.u16()? as usize };
    c.expect(b"</K><N>")?;
    let n = if rel == 117 { c.u32()? as usize } else { c.u64()? as usize };
    c.expect(b"</N><label>")?;
    let ll = if rel == 117 { c.u8()? as usize } else { c.u16()? as usize };
    let datalabel = decode_str(c.take(ll)?, lay.utf8);
    c.expect(b"</label><timestamp>")?;
    let tl = c.u8()? as usize;
    let timestamp = decode_str(c.take(tl)?, lay.utf8);
    c.expect(b"</timestamp></header><map>")?;
    let mut map = [0u64; 14];
    for m in map.iter_mut() {
        *m = c.u64()?;
    }
    c.expect(b"</map>")?;

    let mut types = Vec::with_capacity(k);
    let sec = |c: &mut Cursor, off: u64, tag: &[u8]| -> Result<()> {
        // some writers leave map entries at 0: fall back to scanning forward
        if off == 0 {
            if let Some(i) = c.buf[c.pos..].windows(tag.len()).position(|w| w == tag) {
                c.pos += i;
            }
        } else {
            c.pos = off as usize;
        }
        c.expect(tag)
    };

    sec(&mut c, map[2], b"<variable_types>")?;
    for _ in 0..k {
        let code = c.u16()?;
        types.push(VarType::from_type_code(code).ok_or_else(|| bad(format!("unknown type code {code}")))?);
    }
    sec(&mut c, map[3], b"<varnames>")?;
    let names: Vec<String> = (0..k).map(|_| c.fixed_str(lay.name_w, lay.utf8)).collect::<Result<_>>()?;
    sec(&mut c, map[4], b"<sortlist>")?;
    let mut sortlist = Vec::new();
    for _ in 0..=k {
        let v = if rel == 119 { c.u32()? as usize } else { c.u16()? as usize };
        if v == 0 {
            break;
        }
        sortlist.push(v);
    }
    sec(&mut c, map[5], b"<formats>")?;
    let formats: Vec<String> = (0..k).map(|_| c.fixed_str(lay.fmt_w, lay.utf8)).collect::<Result<_>>()?;
    sec(&mut c, map[6], b"<value_label_names>")?;
    let lblnames: Vec<String> = (0..k).map(|_| c.fixed_str(lay.lblname_w, lay.utf8)).collect::<Result<_>>()?;
    sec(&mut c, map[7], b"<variable_labels>")?;
    let varlabels: Vec<String> = (0..k).map(|_| c.fixed_str(lay.varlbl_w, lay.utf8)).collect::<Result<_>>()?;

    let mut ds = DataSet::new();
    ds.datalabel = datalabel;
    ds.time_stamp = timestamp;
    for i in 0..k {
        let mut v = Variable::new(names[i].clone(), types[i]);
        v.format = formats[i].clone();
        v.value_label_name = lblnames[i].clone();
        v.label = varlabels[i].clone();
        v.data.reserve(n);
        ds.add_variable(v).map_err(bad)?;
    }
    ds.sort_by = sortlist.iter().filter_map(|&i| names.get(i - 1).cloned()).collect();

    // characteristics
    sec(&mut c, map[8], b"<characteristics>")?;
    loop {
        if c.buf[c.pos..].starts_with(b"</characteristics>") {
            break;
        }
        c.expect(b"<ch>")?;
        let len = c.u32()? as usize;
        let body = c.take(len)?;
        c.expect(b"</ch>")?;
        if body.len() >= 2 * lay.name_w {
            let var = decode_str(&body[..lay.name_w], lay.utf8);
            let name = decode_str(&body[lay.name_w..2 * lay.name_w], lay.utf8);
            let val = decode_str(&body[2 * lay.name_w..], lay.utf8);
            ds.characteristics.entry(var).or_default().insert(name, val);
        }
    }

    // data
    sec(&mut c, map[9], b"<data>")?;
    let mut strl_refs: Vec<(usize, usize, u64, u64)> = Vec::new(); // (var, obs, v, o)
    for obs in 0..n {
        for (vi, ty) in types.iter().enumerate() {
            let val = match ty {
                VarType::StrL => {
                    let raw = c.u64()?;
                    let (v, o) = match rel {
                        117 => {
                            // v and o are two 4-byte fields
                            let lo = raw & 0xFFFF_FFFF;
                            let hi = raw >> 32;
                            if c.big { (hi, lo) } else { (lo, hi) }
                        }
                        118 => split_vo(raw, 2, c.big),
                        _ => split_vo(raw, 3, c.big),
                    };
                    if v != 0 || o != 0 {
                        strl_refs.push((vi, obs, v, o));
                    }
                    Value::Str(String::new())
                }
                _ => read_cell(&mut c, *ty, lay.utf8)?,
            };
            ds.vars[vi].data.push(val);
        }
    }
    ds.n_obs = n;

    // strLs
    sec(&mut c, map[10], b"<strls>")?;
    let mut gso: std::collections::HashMap<(u64, u64), String> = Default::default();
    while c.buf[c.pos..].starts_with(b"GSO") {
        c.expect(b"GSO")?;
        let v = c.u32()? as u64;
        let o = if rel == 117 { c.u32()? as u64 } else { c.u64()? };
        let t = c.u8()?;
        let len = c.u32()? as usize;
        let body = c.take(len)?;
        let s = if t == 130 { decode_str(body, true) } else { String::from_utf8_lossy(body).into_owned() };
        gso.insert((v, o), s);
    }
    for (vi, obs, v, o) in strl_refs {
        if let Some(s) = gso.get(&(v, o)) {
            ds.vars[vi].data[obs] = Value::Str(s.clone());
        }
    }

    // value labels
    sec(&mut c, map[11], b"<value_labels>")?;
    while c.buf[c.pos..].starts_with(b"<lbl>") {
        c.expect(b"<lbl>")?;
        let len = c.u32()? as usize;
        let name = c.fixed_str(lay.lblname_w, lay.utf8)?;
        c.take(3)?;
        let table = c.take(len)?;
        c.expect(b"</lbl>")?;
        let mut tc = Cursor { buf: table, pos: 0, big: c.big };
        ds.value_labels.insert(name.clone(), read_label_table(&mut tc, &name, lay.utf8)?);
    }
    Ok(ds)
}

fn split_vo(raw: u64, vbytes: u32, big: bool) -> (u64, u64) {
    // In LSF files v occupies the low-order bytes; in MSF files the
    // high-order ones.
    let vbits = vbytes * 8;
    if big {
        (raw >> (64 - vbits), raw & ((1u64 << (64 - vbits)) - 1))
    } else {
        (raw & ((1u64 << vbits) - 1), raw >> vbits)
    }
}

fn read_label_table(c: &mut Cursor, name: &str, utf8: bool) -> Result<ValueLabel> {
    let n = c.u32()? as usize;
    let txtlen = c.u32()? as usize;
    let offs: Vec<usize> = (0..n).map(|_| c.u32().map(|v| v as usize)).collect::<Result<_>>()?;
    let vals: Vec<i32> = (0..n).map(|_| c.i32()).collect::<Result<_>>()?;
    let txt = c.take(txtlen)?;
    let mut vl = ValueLabel::new(name);
    for (o, v) in offs.iter().zip(vals) {
        if *o < txt.len() {
            vl.add(int_code_to_f64(v), decode_str(&txt[*o..], utf8));
        }
    }
    Ok(vl)
}

/// Value-label values are stored as long; map long missing codes.
fn int_code_to_f64(v: i32) -> f64 {
    if missing::is_missing_long(v) { missing::missing(missing::code_long(v)) } else { v as f64 }
}

fn read_cell(c: &mut Cursor, ty: VarType, utf8: bool) -> Result<Value> {
    Ok(match ty {
        VarType::Byte => {
            let v = c.u8()? as i8;
            if missing::is_missing_byte(v) { Value::Double(missing::missing(missing::code_byte(v))) } else { Value::Byte(v) }
        }
        VarType::Int => {
            let v = c.i16()?;
            if missing::is_missing_int(v) { Value::Double(missing::missing(missing::code_int(v))) } else { Value::Int(v) }
        }
        VarType::Long => {
            let v = c.i32()?;
            if missing::is_missing_long(v) { Value::Double(missing::missing(missing::code_long(v))) } else { Value::Long(v) }
        }
        VarType::Float => {
            let v = c.f32()?;
            if missing::is_missing_float(v) { Value::Double(missing::missing(missing::code_float(v))) } else { Value::Float(v) }
        }
        VarType::Double => {
            let v = c.f64()?;
            Value::Double(if v.is_nan() { missing::SYSMISS } else { v })
        }
        VarType::Str(w) => Value::Str(c.fixed_str(w, utf8)?),
        VarType::StrL => unreachable!("strL handled by caller"),
    })
}

fn old_type(code: u8) -> Result<VarType> {
    Ok(match code {
        251 => VarType::Byte,
        252 => VarType::Int,
        253 => VarType::Long,
        254 => VarType::Float,
        255 => VarType::Double,
        1..=244 => VarType::Str(code as usize),
        _ => return Err(bad(format!("unknown type code {code}"))),
    })
}

fn read_binary(data: &[u8]) -> Result<DataSet> {
    let mut c = Cursor::new(data);
    let rel = c.u8()? as u16;
    let lay = Layout::for_release(rel)?;
    c.big = match c.u8()? {
        1 => true,
        2 => false,
        _ => return Err(bad("bad byteorder")),
    };
    c.take(2)?; // filetype, unused
    let k = c.u16()? as usize;
    let n = c.u32()? as usize;
    let datalabel = c.fixed_str(lay.datalbl_w, false)?;
    let timestamp = c.fixed_str(18, false)?;
    let types: Vec<VarType> = (0..k).map(|_| c.u8().and_then(old_type)).collect::<Result<_>>()?;
    let names: Vec<String> = (0..k).map(|_| c.fixed_str(lay.name_w, false)).collect::<Result<_>>()?;
    let mut sortlist = Vec::new();
    for _ in 0..=k {
        let v = c.u16()? as usize;
        if v != 0 && sortlist.len() < k {
            sortlist.push(v);
        }
    }
    let formats: Vec<String> = (0..k).map(|_| c.fixed_str(lay.fmt_w, false)).collect::<Result<_>>()?;
    let lblnames: Vec<String> = (0..k).map(|_| c.fixed_str(lay.lblname_w, false)).collect::<Result<_>>()?;
    let varlabels: Vec<String> = (0..k).map(|_| c.fixed_str(lay.varlbl_w, false)).collect::<Result<_>>()?;
    // expansion fields
    loop {
        let t = c.u8()?;
        let len = c.u32()? as usize;
        if t == 0 && len == 0 {
            break;
        }
        let body = c.take(len)?;
        if body.len() >= 66 {
            let var = decode_str(&body[..33], false);
            let name = decode_str(&body[33..66], false);
            let _ = (var, name);
        }
    }
    let mut ds = DataSet::new();
    ds.datalabel = datalabel;
    ds.time_stamp = timestamp;
    for i in 0..k {
        let mut v = Variable::new(names[i].clone(), types[i]);
        v.format = formats[i].clone();
        v.value_label_name = lblnames[i].clone();
        v.label = varlabels[i].clone();
        ds.add_variable(v).map_err(bad)?;
    }
    ds.sort_by = sortlist.iter().filter_map(|&i| names.get(i - 1).cloned()).collect();
    for _ in 0..n {
        for (vi, ty) in types.iter().enumerate() {
            let v = read_cell(&mut c, *ty, false)?;
            ds.vars[vi].data.push(v);
        }
    }
    ds.n_obs = n;
    while c.pos + 4 <= data.len() {
        let len = c.u32()? as usize;
        let name = c.fixed_str(33, false)?;
        c.take(3)?;
        let table = c.take(len)?;
        let mut tc = Cursor { buf: table, pos: 0, big: c.big };
        ds.value_labels.insert(name.clone(), read_label_table(&mut tc, &name, false)?);
    }
    Ok(ds)
}

// ---------------------------------------------------------------------------
// Writer (always little-endian)
// ---------------------------------------------------------------------------

pub struct DtaWriter;

struct Out {
    buf: Vec<u8>,
}

impl Out {
    fn put(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.put(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.put(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.put(&v.to_le_bytes());
    }
    fn fixed(&mut self, s: &str, w: usize, utf8: bool) {
        let mut b = encode_str(s, utf8);
        b.truncate(w.saturating_sub(1));
        b.resize(w, 0);
        self.put(&b);
    }
}

fn encode_str(s: &str, utf8: bool) -> Vec<u8> {
    if utf8 {
        s.as_bytes().to_vec()
    } else {
        s.chars().map(|c| if (c as u32) < 256 { c as u8 } else { b'?' }).collect()
    }
}

fn timestamp_now() -> String {
    chrono::Local::now().format("%d %b %Y %H:%M").to_string()
}

impl DtaWriter {
    /// Write in the current default format (118, or 119 for > 32,767 variables).
    pub fn write<W: Write>(writer: &mut W, ds: &DataSet) -> Result<()> {
        let rel = if ds.n_vars() > 32767 { 119 } else { 118 };
        Self::write_release(writer, ds, rel)
    }

    /// Write a specific release: 115, 117, 118 or 119.
    pub fn write_release<W: Write>(writer: &mut W, ds: &DataSet, release: u16) -> Result<()> {
        let bytes = match release {
            117..=119 => write_tagged(ds, release)?,
            113..=115 => write_binary(ds, release)?,
            r => return Err(DtaError::UnsupportedRelease(r)),
        };
        writer.write_all(&bytes)?;
        Ok(())
    }
}

/// Encode a numeric cell for its storage type (out-of-range values become missing).
fn put_num(o: &mut Out, ty: VarType, v: f64) {
    let code = missing::missing_code(v);
    match ty {
        VarType::Byte => o.u8(match code {
            Some(c) => (101 + c as i16) as u8,
            None if v.fract() == 0.0 && (-127.0..=100.0).contains(&v) => v as i8 as u8,
            None => 101,
        }),
        VarType::Int => o.u16(match code {
            Some(c) => (32741 + c as i32) as u16,
            None if v.fract() == 0.0 && (-32767.0..=32740.0).contains(&v) => v as i16 as u16,
            None => 32741,
        }),
        VarType::Long => o.u32(match code {
            Some(c) => (2147483621i64 + c as i64) as u32,
            None if v.fract() == 0.0 && (-2147483647.0..=2147483620.0).contains(&v) => v as i32 as u32,
            None => 2147483621,
        }),
        VarType::Float => o.u32(match code {
            Some(c) => 0x7F00_0000 + c as u32 * 0x800,
            None => (v as f32).to_bits(),
        }),
        VarType::Double => o.u64(missing::normalize(v).to_bits()),
        _ => unreachable!(),
    }
}

fn write_label_table(o: &mut Out, vl: &ValueLabel, utf8: bool) -> Vec<u8> {
    let _ = o;
    let mut txt = Vec::new();
    let mut offs = Vec::new();
    let mut vals = Vec::new();
    for (v, l) in vl.iter() {
        offs.push(txt.len() as u32);
        vals.push(match missing::missing_code(v) {
            Some(c) => 2147483621 + c as i32,
            None => v as i32,
        });
        txt.extend_from_slice(&encode_str(l, utf8));
        txt.push(0);
    }
    let mut t = Out { buf: Vec::new() };
    t.u32(offs.len() as u32);
    t.u32(txt.len() as u32);
    for x in offs {
        t.u32(x);
    }
    for v in vals {
        t.u32(v as u32);
    }
    t.put(&txt);
    t.buf
}

fn sorted_labels(ds: &DataSet) -> Vec<&ValueLabel> {
    let mut v: Vec<&ValueLabel> = ds.value_labels.values().collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

fn write_tagged(ds: &DataSet, rel: u16) -> Result<Vec<u8>> {
    let lay = Layout::for_release(rel)?;
    let k = ds.n_vars();
    let n = ds.n_obs;
    if rel != 119 && k > 32767 {
        return Err(bad("too many variables for this release"));
    }
    let mut o = Out { buf: Vec::new() };
    o.put(FORMAT_OPEN);
    o.put(rel.to_string().as_bytes());
    o.put(b"</release><byteorder>LSF</byteorder><K>");
    if rel == 119 { o.u32(k as u32) } else { o.u16(k as u16) }
    o.put(b"</K><N>");
    if rel == 117 { o.u32(n as u32) } else { o.u64(n as u64) }
    o.put(b"</N><label>");
    let mut lbl = encode_str(&ds.datalabel, lay.utf8);
    lbl.truncate(if rel == 117 { 80 } else { 320 });
    if rel == 117 { o.u8(lbl.len() as u8) } else { o.u16(lbl.len() as u16) }
    o.put(&lbl);
    o.put(b"</label><timestamp>");
    let ts = if ds.time_stamp.is_empty() { timestamp_now() } else { ds.time_stamp.clone() };
    let ts = encode_str(&ts, lay.utf8);
    o.u8(ts.len() as u8);
    o.put(&ts);
    o.put(b"</timestamp></header>");

    let mut map = [0u64; 14];
    map[1] = o.buf.len() as u64;
    o.put(b"<map>");
    let map_at = o.buf.len();
    for _ in 0..14 {
        o.u64(0);
    }
    o.put(b"</map>");

    map[2] = o.buf.len() as u64;
    o.put(b"<variable_types>");
    for v in &ds.vars {
        o.u16(v.var_type.type_code());
    }
    o.put(b"</variable_types>");

    map[3] = o.buf.len() as u64;
    o.put(b"<varnames>");
    for v in &ds.vars {
        o.fixed(&v.name, lay.name_w, lay.utf8);
    }
    o.put(b"</varnames>");

    map[4] = o.buf.len() as u64;
    o.put(b"<sortlist>");
    let mut sl: Vec<usize> = ds.sort_by.iter().filter_map(|nm| ds.var_index.get(nm).map(|i| i + 1)).collect();
    sl.resize(k + 1, 0);
    for s in sl {
        if rel == 119 { o.u32(s as u32) } else { o.u16(s as u16) }
    }
    o.put(b"</sortlist>");

    map[5] = o.buf.len() as u64;
    o.put(b"<formats>");
    for v in &ds.vars {
        o.fixed(&v.format, lay.fmt_w, lay.utf8);
    }
    o.put(b"</formats>");

    map[6] = o.buf.len() as u64;
    o.put(b"<value_label_names>");
    for v in &ds.vars {
        o.fixed(&v.value_label_name, lay.lblname_w, lay.utf8);
    }
    o.put(b"</value_label_names>");

    map[7] = o.buf.len() as u64;
    o.put(b"<variable_labels>");
    for v in &ds.vars {
        o.fixed(&v.label, lay.varlbl_w, lay.utf8);
    }
    o.put(b"</variable_labels>");

    map[8] = o.buf.len() as u64;
    o.put(b"<characteristics>");
    for (var, chars) in &ds.characteristics {
        for (name, val) in chars {
            let mut body = Out { buf: Vec::new() };
            body.fixed(var, lay.name_w, lay.utf8);
            body.fixed(name, lay.name_w, lay.utf8);
            body.put(&encode_str(val, lay.utf8));
            body.u8(0);
            o.put(b"<ch>");
            o.u32(body.buf.len() as u32);
            o.put(&body.buf);
            o.put(b"</ch>");
        }
    }
    o.put(b"</characteristics>");

    map[9] = o.buf.len() as u64;
    o.put(b"<data>");
    let mut gso: Vec<(u32, u64, String)> = Vec::new();
    for obs in 0..n {
        for (vi, v) in ds.vars.iter().enumerate() {
            let cell = &v.data[obs];
            match v.var_type {
                VarType::Str(w) => {
                    let mut b = encode_str(cell_str(ds, cell), lay.utf8);
                    b.truncate(w);
                    b.resize(w, 0);
                    o.put(&b);
                }
                VarType::StrL => {
                    let s = cell_str(ds, cell);
                    let (vv, oo) = ((vi + 1) as u64, (obs + 1) as u64);
                    if s.is_empty() {
                        o.u64(0);
                    } else {
                        let raw = match rel {
                            117 => vv | (oo << 32),
                            118 => vv | (oo << 16),
                            _ => vv | (oo << 24),
                        };
                        o.u64(raw);
                        gso.push((vv as u32, oo, s.to_string()));
                    }
                }
                ty => put_num(&mut o, ty, cell.num()),
            }
        }
    }
    o.put(b"</data>");

    map[10] = o.buf.len() as u64;
    o.put(b"<strls>");
    for (v, oo, s) in gso {
        o.put(b"GSO");
        o.u32(v);
        if rel == 117 { o.u32(oo as u32) } else { o.u64(oo) }
        o.u8(130);
        let b = encode_str(&s, lay.utf8);
        o.u32(b.len() as u32 + 1);
        o.put(&b);
        o.u8(0);
    }
    o.put(b"</strls>");

    map[11] = o.buf.len() as u64;
    o.put(b"<value_labels>");
    for vl in sorted_labels(ds) {
        let table = write_label_table(&mut Out { buf: Vec::new() }, vl, lay.utf8);
        o.put(b"<lbl>");
        o.u32(table.len() as u32);
        o.fixed(&vl.name, lay.lblname_w, lay.utf8);
        o.put(&[0, 0, 0]);
        o.put(&table);
        o.put(b"</lbl>");
    }
    o.put(b"</value_labels>");

    map[12] = o.buf.len() as u64;
    o.put(FORMAT_CLOSE);
    map[13] = o.buf.len() as u64;
    for (i, m) in map.iter().enumerate() {
        o.buf[map_at + i * 8..map_at + i * 8 + 8].copy_from_slice(&m.to_le_bytes());
    }
    Ok(o.buf)
}

fn cell_str<'a>(ds: &'a DataSet, v: &'a Value) -> &'a str {
    match v {
        Value::Str(s) => s,
        Value::StrL(a, b) => ds.resolve_strl(*a, *b).unwrap_or(""),
        _ => "",
    }
}

fn write_binary(ds: &DataSet, rel: u16) -> Result<Vec<u8>> {
    let lay = Layout::for_release(rel)?;
    let k = ds.n_vars();
    let mut o = Out { buf: Vec::new() };
    o.u8(rel as u8);
    o.u8(2); // LOHI
    o.u8(1);
    o.u8(0);
    o.u16(k as u16);
    o.u32(ds.n_obs as u32);
    o.fixed(&ds.datalabel, 81, false);
    let ts = if ds.time_stamp.is_empty() { timestamp_now() } else { ds.time_stamp.clone() };
    o.fixed(&ts, 18, false);
    let mut widths = Vec::new();
    for v in &ds.vars {
        let (code, ty) = match v.var_type {
            VarType::Byte => (251, VarType::Byte),
            VarType::Int => (252, VarType::Int),
            VarType::Long => (253, VarType::Long),
            VarType::Float => (254, VarType::Float),
            VarType::Double => (255, VarType::Double),
            VarType::Str(w) => (w.min(244) as u8, VarType::Str(w.min(244))),
            VarType::StrL => (244, VarType::Str(244)),
        };
        o.u8(code);
        widths.push(ty);
    }
    for v in &ds.vars {
        o.fixed(&v.name, lay.name_w, false);
    }
    let mut sl: Vec<usize> = ds.sort_by.iter().filter_map(|nm| ds.var_index.get(nm).map(|i| i + 1)).collect();
    sl.resize(k + 1, 0);
    for s in sl {
        o.u16(s as u16);
    }
    for v in &ds.vars {
        o.fixed(&v.format, lay.fmt_w, false);
    }
    for v in &ds.vars {
        o.fixed(&v.value_label_name, lay.lblname_w, false);
    }
    for v in &ds.vars {
        o.fixed(&v.label, lay.varlbl_w, false);
    }
    o.u8(0);
    o.u32(0);
    for obs in 0..ds.n_obs {
        for (vi, v) in ds.vars.iter().enumerate() {
            match widths[vi] {
                VarType::Str(w) => {
                    let mut b = encode_str(cell_str(ds, &v.data[obs]), false);
                    b.truncate(w);
                    b.resize(w, 0);
                    o.put(&b);
                }
                ty => put_num(&mut o, ty, v.data[obs].num()),
            }
        }
    }
    for vl in sorted_labels(ds) {
        let table = write_label_table(&mut Out { buf: Vec::new() }, vl, false);
        o.u32(table.len() as u32);
        o.fixed(&vl.name, 33, false);
        o.put(&[0, 0, 0]);
        o.put(&table);
    }
    Ok(o.buf)
}
