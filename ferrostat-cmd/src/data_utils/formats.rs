//! Foreign data formats, all pure Rust: SPSS system files (`import spss`),
//! SAS7BDAT files (`import sas`), SAS transport v5/v8 (`import sasxport5`,
//! `import sasxport8`, `export sasxport5`, `export sasxport8`) and dBase
//! tables (`import dbase`).

use std::collections::{BTreeMap, HashSet};

use ferrostat_core::missing::{self, is_missing, SYSMISS};
use ferrostat_core::value_label::FloatKey;
use ferrostat_core::{vfs, DataSet, Value, ValueLabel, VarType, Variable};
use ferrostat_parser::Command;

use super::files::decode_single;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;

/// One decoded column before it becomes a variable.
#[derive(Default, Clone)]
struct Col {
    name: String,
    label: String,
    format: String,
    numeric: bool,
    nums: Vec<f64>,
    strs: Vec<String>,
    labels: Vec<(f64, String)>,
}

fn clean_name(raw: &str, case: &str) -> String {
    let t = raw.trim();
    let mut s: String = t.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if s.is_empty() || s.chars().next().unwrap().is_ascii_digit() {
        s.insert(0, '_');
    }
    s.truncate(32);
    match case {
        "lower" => s.to_lowercase(),
        "upper" => s.to_uppercase(),
        _ => s,
    }
}

fn text(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => decode_single(bytes, true),
    }
}

fn trim_text(bytes: &[u8]) -> String {
    let end = bytes.iter().rposition(|&b| b != b' ' && b != 0).map_or(0, |p| p + 1);
    text(&bytes[..end])
}

fn build(cols: Vec<Col>, nobs: usize, case: &str) -> Result<DataSet, String> {
    let mut ds = DataSet::new();
    ds.n_obs = nobs;
    let mut used: HashSet<String> = HashSet::new();
    for c in cols {
        let mut name = clean_name(&c.name, case);
        let mut k = 1;
        while used.contains(&name) {
            let suffix = format!("_{k}");
            name.truncate(32 - suffix.len());
            name.push_str(&suffix);
            k += 1;
        }
        used.insert(name.clone());
        let mut v;
        if c.numeric {
            let integer = c.nums.iter().all(|x| is_missing(*x) || x.fract() == 0.0);
            let force_double = c.format.starts_with("%tc") || !integer;
            let ty = if force_double { VarType::Double } else { util::best_numeric_type(&c.nums) };
            v = Variable::new(&name, ty);
            v.data = c.nums.iter().map(|&x| util::typed_value(ty, x)).collect();
            v.format = if c.format.is_empty() { "%9.0g".into() } else { c.format.clone() };
            if !c.labels.is_empty() {
                let mut vl = ValueLabel::new(name.clone());
                for (k, l) in &c.labels {
                    vl.mapping.insert(FloatKey(*k), l.clone());
                }
                ds.value_labels.insert(name.clone(), vl);
                v.value_label_name = name.clone();
            }
        } else {
            let w = c.strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let ty = util::str_type_for(w);
            v = Variable::new(&name, ty);
            v.data = c.strs.iter().map(|s| Value::Str(s.clone())).collect();
            v.format = format!("%{}s", w.min(2045).max(1));
        }
        v.label = c.label.clone();
        ds.add_variable(v)?;
    }
    ds.n_obs = nobs;
    Ok(ds)
}

// ---------------------------------------------------------------- entry points

pub fn import_cmd(kind: &str, rest: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let k = match kind {
        "spss" | "sas" | "sas7bdat" | "sasxport5" | "sasxport8" | "sasxport" | "dbase" | "dbf" => kind,
        _ => return None,
    };
    Some((|| {
        let raw = cmd.using.clone().unwrap_or_else(|| util::unquote(rest.trim()));
        if raw.is_empty() {
            return Err(rc_err(100, "using required"));
        }
        if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
            return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
        }
        let ext = match k {
            "spss" => "sav",
            "sas" | "sas7bdat" => "sas7bdat",
            "dbase" | "dbf" => "dbf",
            _ => "xpt",
        };
        let path = util::unquote(&raw);
        let path = if std::path::Path::new(&path).extension().is_none() { format!("{path}.{ext}") } else { path };
        let data = || vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")));
        let case = cmd.opt("case", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "preserve".into());
        let new = match k {
            "spss" => read_sav(&data()?, &case)?,
            "sas" | "sas7bdat" => read_sas7bdat(&data()?, &case)?,
            "dbase" | "dbf" => read_dbase(&path, &case)?,
            _ => read_xport(&data()?, &case)?,
        };
        let msg = format!("({} vars, {} obs)", new.n_vars(), util::commas(new.n_obs));
        *ds = new;
        st.data_changed = false;
        st.tsset = None;
        st.filename.clear();
        Ok(msg)
    })())
}

pub fn export_cmd(kind: &str, rest: &str, ds: &DataSet, st: &Store, cmd: &Command) -> Option<Result<String, String>> {
    let v8 = match kind {
        "sasxport5" | "sasxport" => false,
        "sasxport8" => true,
        _ => return None,
    };
    Some((|| {
        let (vars, path) = match &cmd.using {
            Some(p) => (if rest.trim().is_empty() { ds.varlist() } else { crate::varlist::expand(ds, rest)? }, p.clone()),
            None => (ds.varlist(), util::unquote(rest.trim())),
        };
        let path = util::unquote(&path);
        let path = if std::path::Path::new(&path).extension().is_none() { format!("{path}.xpt") } else { path };
        if vfs::exists(&path) && !cmd.has_opt("replace", 3) {
            return Err(rc_err(602, format!("file {path} already exists")));
        }
        let mut sub = cmd.clone();
        sub.main.clear();
        let mask = util::touse(ds, st, &sub)?;
        let bytes = write_xport(ds, &vars, &mask, &path, v8)?;
        vfs::write(&path, &bytes).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
        Ok(format!("file {path} saved"))
    })())
}

// ---------------------------------------------------------------- dBase

fn read_dbase(path: &str, case: &str) -> Result<DataSet, String> {
    let (fields, numeric, rows) = crate::spatial::read_dbf(path)?;
    let case = if case == "preserve" { "lower" } else { case };
    let mut cols = Vec::new();
    for (k, name) in fields.iter().enumerate() {
        let cells: Vec<&String> = rows.iter().map(|r| &r[k]).collect();
        let mut c = Col { name: name.clone(), ..Default::default() };
        if numeric[k] && cells.iter().all(|s| s.is_empty() || s.parse::<f64>().is_ok()) {
            c.numeric = true;
            c.nums = cells.iter().map(|s| s.parse().unwrap_or(SYSMISS)).collect();
        } else {
            c.strs = cells.into_iter().cloned().collect();
        }
        cols.push(c);
    }
    build(cols, rows.len(), case)
}

// ---------------------------------------------------------------- SPSS

struct Bytes<'a> {
    b: &'a [u8],
    le: bool,
}

impl<'a> Bytes<'a> {
    fn i32(&self, p: usize) -> Result<i32, String> {
        let s = self.b.get(p..p + 4).ok_or_else(|| rc_err(610, "file is truncated"))?;
        let a = [s[0], s[1], s[2], s[3]];
        Ok(if self.le { i32::from_le_bytes(a) } else { i32::from_be_bytes(a) })
    }
    fn f64(&self, p: usize) -> Result<f64, String> {
        let s = self.b.get(p..p + 8).ok_or_else(|| rc_err(610, "file is truncated"))?;
        Ok(self.f64_of(s))
    }
    fn f64_of(&self, s: &[u8]) -> f64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&s[..8]);
        if self.le {
            f64::from_le_bytes(a)
        } else {
            f64::from_be_bytes(a)
        }
    }
    fn slice(&self, p: usize, n: usize) -> Result<&'a [u8], String> {
        self.b.get(p..p + n).ok_or_else(|| rc_err(610, "file is truncated"))
    }
}

struct SavVar {
    typ: i32,
    name: String,
    label: String,
    print: i32,
    missing: Vec<f64>,
    range: Option<(f64, f64)>,
}

#[derive(Clone, Copy)]
enum Item {
    Num(f64),
    Raw([u8; 8]),
}

const SPSS_EPOCH_DAYS: f64 = 137775.0;

fn read_sav(data: &[u8], case: &str) -> Result<DataSet, String> {
    if data.len() < 176 || !(data.starts_with(b"$FL2") || data.starts_with(b"$FL3")) {
        return Err(rc_err(610, "file is not an SPSS system file (.sav)"));
    }
    let layout_le = i32::from_le_bytes(data[64..68].try_into().unwrap());
    let le = (2..=3).contains(&layout_le);
    let r = Bytes { b: data, le };
    let case_size = r.i32(68)?;
    let compression = r.i32(72)?;
    let ncases = r.i32(80)?;
    let bias = r.f64(84)?;
    if compression == 2 {
        return Err(rc_err(610, "zlib-compressed (.zsav) files are not supported"));
    }
    let file_label = trim_text(r.slice(109, 64).unwrap_or(&[]));
    let mut p = 176usize;
    let mut vars: Vec<SavVar> = Vec::new();
    // dictionary slot -> index into `vars` of the variable record
    let mut labelsets: Vec<(Vec<(Vec<u8>, String)>, Vec<usize>)> = Vec::new();
    let mut long_names: BTreeMap<String, String> = BTreeMap::new();
    let mut very_long: BTreeMap<String, usize> = BTreeMap::new();
    let mut encoding = String::new();
    loop {
        let rec = r.i32(p)?;
        p += 4;
        match rec {
            2 => {
                let typ = r.i32(p)?;
                let has_label = r.i32(p + 4)?;
                let n_miss = r.i32(p + 8)?;
                let print = r.i32(p + 12)?;
                let name = trim_text(r.slice(p + 20, 8)?);
                p += 28;
                let mut label = String::new();
                if has_label != 0 {
                    let len = r.i32(p)? as usize;
                    label = trim_text(r.slice(p + 4, len)?);
                    p += 4 + len.div_ceil(4) * 4;
                }
                let mut missing = Vec::new();
                let mut range = None;
                if n_miss != 0 {
                    let k = n_miss.unsigned_abs() as usize;
                    let vals: Vec<f64> = (0..k).map(|i| r.f64(p + 8 * i)).collect::<Result<_, _>>()?;
                    p += 8 * k;
                    if n_miss < 0 {
                        range = Some((vals[0], vals[1]));
                        missing = vals[2..].to_vec();
                    } else {
                        missing = vals;
                    }
                }
                vars.push(SavVar { typ, name, label, print, missing, range });
            }
            3 => {
                let n = r.i32(p)? as usize;
                p += 4;
                let mut pairs = Vec::new();
                for _ in 0..n {
                    let val = r.slice(p, 8)?.to_vec();
                    let len = r.slice(p + 8, 1)?[0] as usize;
                    let lab = trim_text(r.slice(p + 9, len)?);
                    p += 8 + ((1 + len).div_ceil(8)) * 8;
                    pairs.push((val, lab));
                }
                // the variable-index record (type 4) follows
                let t = r.i32(p)?;
                if t != 4 {
                    return Err(rc_err(610, "damaged value-label record"));
                }
                let k = r.i32(p + 4)? as usize;
                let idx: Vec<usize> = (0..k).map(|i| r.i32(p + 8 + 4 * i).map(|x| x as usize)).collect::<Result<_, _>>()?;
                p += 8 + 4 * k;
                labelsets.push((pairs, idx));
            }
            6 => {
                let n = r.i32(p)? as usize;
                p += 4 + 80 * n;
            }
            7 => {
                let sub = r.i32(p)?;
                let size = r.i32(p + 4)? as usize;
                let count = r.i32(p + 8)? as usize;
                p += 12;
                let body = r.slice(p, size * count)?;
                p += size * count;
                match sub {
                    13 => {
                        for kv in String::from_utf8_lossy(body).split('\t') {
                            if let Some((k, v)) = kv.split_once('=') {
                                long_names.insert(k.trim().to_string(), v.trim().to_string());
                            }
                        }
                    }
                    14 => {
                        for kv in String::from_utf8_lossy(body).split('\t') {
                            if let Some((k, v)) = kv.split_once('=') {
                                if let Ok(n) = v.trim_matches('\0').trim().parse::<usize>() {
                                    very_long.insert(k.trim().to_string(), n);
                                }
                            }
                        }
                    }
                    20 => encoding = String::from_utf8_lossy(body).to_string(),
                    _ => {}
                }
            }
            999 => {
                p += 4;
                break;
            }
            other => return Err(rc_err(610, format!("unexpected record type {other} in SPSS file"))),
        }
    }
    let _ = encoding;
    // slots
    let nslots = if case_size > 0 { case_size as usize } else { vars.len() };
    // main variables: records that are not continuation
    let mains: Vec<usize> = (0..vars.len()).filter(|&i| vars[i].typ != -1).collect();
    // decode data into items
    let mut items: Vec<Item> = Vec::new();
    let body = &data[p.min(data.len())..];
    if compression == 0 {
        for c in body.chunks_exact(8) {
            let mut a = [0u8; 8];
            a.copy_from_slice(c);
            items.push(Item::Raw(a));
        }
    } else {
        let mut q = 0usize;
        'outer: while q + 8 <= body.len() {
            let ctrl = &body[q..q + 8];
            q += 8;
            for &c in ctrl {
                match c {
                    0 => {}
                    252 => break 'outer,
                    253 => {
                        if q + 8 > body.len() {
                            break 'outer;
                        }
                        let mut a = [0u8; 8];
                        a.copy_from_slice(&body[q..q + 8]);
                        q += 8;
                        items.push(Item::Raw(a));
                    }
                    254 => items.push(Item::Raw([b' '; 8])),
                    255 => items.push(Item::Num(f64::MIN)),
                    k => items.push(Item::Num(k as f64 - bias)),
                }
            }
        }
    }
    let ncases_found = items.len() / nslots.max(1);
    let n = if ncases >= 0 { (ncases as usize).min(ncases_found) } else { ncases_found };
    // slot index of each main variable
    let mut slot_of = Vec::new();
    let mut s = 0usize;
    for (i, v) in vars.iter().enumerate() {
        if v.typ != -1 {
            slot_of.push((i, s));
        }
        s += 1;
    }
    let mut cols: Vec<Col> = Vec::new();
    let mut col_of_var: BTreeMap<usize, usize> = BTreeMap::new();
    for &(vi, slot) in &slot_of {
        let v = &vars[vi];
        let name = long_names.get(&v.name).cloned().unwrap_or_else(|| v.name.clone());
        let mut c = Col { name, label: v.label.clone(), numeric: v.typ == 0, ..Default::default() };
        let fmt_type = (v.print >> 16) & 0xff;
        let w = (v.print >> 8) & 0xff;
        let d = v.print & 0xff;
        let is_date = c.numeric && matches!(fmt_type, 20 | 23 | 24 | 38 | 39);
        let is_dt = c.numeric && fmt_type == 22;
        c.format = if is_date {
            "%td".into()
        } else if is_dt {
            "%tc".into()
        } else if c.numeric {
            if d > 0 { format!("%{}.{}f", w.max(1), d) } else { "%9.0g".into() }
        } else {
            format!("%{}s", v.typ.max(1))
        };
        let nsl = if v.typ > 0 { (v.typ as usize).div_ceil(8) } else { 1 };
        for case_i in 0..n {
            let base = case_i * nslots + slot;
            if c.numeric {
                let x = match items[base] {
                    Item::Num(x) => x,
                    Item::Raw(b) => r.f64_of(&b),
                };
                let mut val = if x <= f64::MIN || x.is_nan() { SYSMISS } else { x };
                if !is_missing(val) {
                    if let Some(k) = v.missing.iter().position(|m| *m == x) {
                        val = missing::missing((k as u8 + 1).min(26));
                    } else if let Some((lo, hi)) = v.range {
                        if x >= lo && x <= hi {
                            val = missing::missing(1);
                        }
                    }
                }
                if !is_missing(val) {
                    if is_date {
                        val = (val / 86400.0).floor() - SPSS_EPOCH_DAYS;
                    } else if is_dt {
                        val = (val - SPSS_EPOCH_DAYS * 86400.0) * 1000.0;
                    }
                }
                c.nums.push(val);
            } else {
                let mut bytes = Vec::with_capacity(nsl * 8);
                for k in 0..nsl {
                    match items.get(base + k) {
                        Some(Item::Raw(b)) => bytes.extend_from_slice(b),
                        Some(Item::Num(_)) => bytes.extend_from_slice(&[b' '; 8]),
                        None => {}
                    }
                }
                bytes.truncate(v.typ as usize);
                c.strs.push(trim_text(&bytes));
            }
        }
        col_of_var.insert(vi, cols.len());
        cols.push(c);
    }
    let _ = (&mains, &very_long);
    // value labels (record 3/4 indexes are 1-based dictionary slots)
    for (pairs, idx) in labelsets {
        for ix in idx {
            let Some(&ci) = ix.checked_sub(1).and_then(|k| col_of_var.get(&k)) else { continue };
            if !cols[ci].numeric {
                continue;
            }
            cols[ci].labels = pairs.iter().map(|(v, l)| (r.f64_of(v), l.clone())).collect();
        }
    }
    let mut ds = build(cols, n, case)?;
    ds.datalabel = file_label;
    Ok(ds)
}

// ---------------------------------------------------------------- SAS7BDAT

fn sas_date_format(name: &str) -> Option<&'static str> {
    let base: String = name.trim().to_uppercase().trim_end_matches(|c: char| c.is_ascii_digit() || c == '.').to_string();
    const DATES: [&str; 12] = ["DATE", "DDMMYY", "MMDDYY", "YYMMDD", "EURDFDE", "WORDDATE", "WORDDATX", "WEEKDATE", "MONYY", "JULIAN", "NENGO", "YYMON"];
    const DTS: [&str; 4] = ["DATETIME", "DATEAMPM", "E8601DT", "MDYAMPM"];
    if DATES.contains(&base.as_str()) {
        Some("%td")
    } else if DTS.contains(&base.as_str()) {
        Some("%tc")
    } else if base == "TIME" {
        Some("%tcHH:MM:SS")
    } else {
        None
    }
}

struct SasCol {
    name: (usize, usize, usize),
    format: (usize, usize, usize),
    label: (usize, usize, usize),
    offset: usize,
    width: usize,
    numeric: bool,
}

fn rd_uint(b: &[u8], p: usize, n: usize, le: bool) -> Result<u64, String> {
    let s = b.get(p..p + n).ok_or_else(|| rc_err(610, "file is truncated"))?;
    let mut v = 0u64;
    if le {
        for &x in s.iter().rev() {
            v = (v << 8) | x as u64;
        }
    } else {
        for &x in s {
            v = (v << 8) | x as u64;
        }
    }
    Ok(v)
}

fn sas_rle(input: &[u8], out_len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(out_len);
    let mut i = 0;
    let bad = || rc_err(610, "damaged compressed row in SAS file");
    while i < input.len() {
        let control = input[i];
        i += 1;
        let cmd = control >> 4;
        let len = (control & 0x0F) as usize;
        let mut copy = 0usize;
        let mut ins = 0usize;
        let mut byte = 0u8;
        let next = |i: &mut usize| -> Result<usize, String> {
            let v = *input.get(*i).ok_or_else(bad)? as usize;
            *i += 1;
            Ok(v)
        };
        match cmd {
            0 => copy = next(&mut i)? + 64 + len * 256,
            1 => copy = next(&mut i)? + 64 + len * 256 + 4096,
            2 => copy = len + 96,
            4 => {
                ins = next(&mut i)? + 18 + len * 256;
                byte = next(&mut i)? as u8;
            }
            5 => {
                ins = next(&mut i)? + 17 + len * 256;
                byte = b'@';
            }
            6 => {
                ins = next(&mut i)? + 17 + len * 256;
                byte = b' ';
            }
            7 => {
                ins = next(&mut i)? + 17 + len * 256;
                byte = 0;
            }
            8 => copy = len + 1,
            9 => copy = len + 17,
            10 => copy = len + 33,
            11 => copy = len + 49,
            12 => {
                byte = next(&mut i)? as u8;
                ins = len + 3;
            }
            13 => {
                byte = b'@';
                ins = len + 2;
            }
            14 => {
                byte = b' ';
                ins = len + 2;
            }
            15 => {
                byte = 0;
                ins = len + 2;
            }
            _ => {}
        }
        if copy > 0 {
            out.extend_from_slice(input.get(i..i + copy).ok_or_else(bad)?);
            i += copy;
        } else if ins > 0 {
            out.extend(std::iter::repeat(byte).take(ins));
        }
        if out.len() > out_len {
            return Err(bad());
        }
    }
    if out.len() != out_len {
        return Err(bad());
    }
    Ok(out)
}

fn sas_rdc(input: &[u8], out_len: usize) -> Result<Vec<u8>, String> {
    let bad = || rc_err(610, "damaged compressed row in SAS file");
    let mut out: Vec<u8> = Vec::with_capacity(out_len);
    let mut i = 0;
    while i + 2 <= input.len() {
        let prefix = ((input[i] as u32) << 8) + input[i + 1] as u32;
        i += 2;
        for bit in 0..16 {
            if prefix & (1 << (15 - bit)) == 0 {
                if i + 1 > input.len() {
                    break;
                }
                out.push(input[i]);
                i += 1;
                continue;
            }
            if i + 2 > input.len() {
                return Err(bad());
            }
            let marker = input[i];
            let nextb = input[i + 1] as usize;
            i += 2;
            let (mut ins, mut cpy, mut ib, mut back) = (0usize, 0usize, 0u8, 0usize);
            if marker <= 0x0F {
                ins = 3 + marker as usize;
                ib = nextb as u8;
            } else if marker >> 4 == 1 {
                ins = 19 + (marker & 0x0F) as usize + nextb * 16;
                ib = *input.get(i).ok_or_else(bad)?;
                i += 1;
            } else if marker >> 4 == 2 {
                cpy = 16 + *input.get(i).ok_or_else(bad)? as usize;
                i += 1;
                back = 3 + (marker & 0x0F) as usize + nextb * 16;
            } else {
                cpy = (marker >> 4) as usize;
                back = 3 + (marker & 0x0F) as usize + nextb * 16;
            }
            if ins > 0 {
                out.extend(std::iter::repeat(ib).take(ins));
            } else if cpy > 0 {
                if out.len() < back || cpy > back {
                    return Err(bad());
                }
                let start = out.len() - back;
                for k in 0..cpy {
                    let b = out[start + k];
                    out.push(b);
                }
            }
            if out.len() > out_len {
                return Err(bad());
            }
        }
    }
    if out.len() != out_len {
        return Err(bad());
    }
    Ok(out)
}

fn read_sas7bdat(d: &[u8], case: &str) -> Result<DataSet, String> {
    const MAGIC: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xc2, 0xea, 0x81, 0x60];
    if d.len() < 1024 || d[..16] != MAGIC {
        return Err(rc_err(610, "file is not a SAS7BDAT data set"));
    }
    let pad1 = if d[35] == 0x33 { 4 } else { 0 };
    let u64f = d[32] == 0x33;
    let le = d[37] == 1;
    let encoding_code = d[70];
    let header_size = rd_uint(d, 196 + pad1, 4, le)? as usize;
    let page_size = rd_uint(d, 200 + pad1, 4, le)? as usize;
    let page_count = rd_uint(d, 204 + pad1, if u64f { 8 } else { 4 }, le)? as usize;
    if header_size < 1024 || page_size < 1024 || header_size > (1 << 24) || page_size > (1 << 24) {
        return Err(rc_err(610, "damaged SAS file header"));
    }
    let (phs, shps, sigl, wl) = if u64f { (40usize, 24usize, 8usize, 8usize) } else { (24, 12, 4, 4) };
    let _ = encoding_code;
    let decode = |b: &[u8]| -> String {
        let end = b.iter().rposition(|&x| x != b' ' && x != 0).map_or(0, |p| p + 1);
        text(&b[..end])
    };
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut names: Vec<(usize, usize, usize)> = Vec::new();
    let mut attrs: Vec<(usize, usize, bool)> = Vec::new();
    let mut fmts: Vec<((usize, usize, usize), (usize, usize, usize))> = Vec::new();
    let mut row_length = 0usize;
    let mut row_count = 0usize;
    let mut mix_rows = 0usize;
    let mut rdc = false;
    let mut compressed_file = false;
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut label = String::new();
    let tref = |b: &[u8], p: usize| -> (usize, usize, usize) {
        let g = |q: usize| -> usize { if q + 2 <= b.len() { rd_uint(b, q, 2, le).unwrap_or(0) as usize } else { 0 } };
        (g(p), g(p + 2), g(p + 4))
    };
    let resolve = |blobs: &Vec<Vec<u8>>, r: (usize, usize, usize)| -> String {
        if r.2 == 0 {
            return String::new();
        }
        match blobs.get(r.0) {
            Some(b) if r.1 + r.2 <= b.len() => decode(&b[r.1..r.1 + r.2]),
            _ => String::new(),
        }
    };
    // pass 1: collect the column-text blobs (later subheaders refer to them)
    for pg in 0..page_count {
        let start = header_size + pg * page_size;
        if start + page_size > d.len() {
            break;
        }
        let page = &d[start..start + page_size];
        let page_type = rd_uint(page, phs - 8, 2, le)? as usize;
        let nsub = rd_uint(page, phs - 4, 2, le)? as usize;
        if page_type & 0x9000 != 0 || page_type & 0x0F00 == 0x0100 || phs + nsub * shps > page_size {
            continue;
        }
        for k in 0..nsub {
            let sp = phs + k * shps;
            let off = rd_uint(page, sp, wl, le)? as usize;
            let len = rd_uint(page, sp + wl, wl, le)? as usize;
            if len < sigl + 2 || off + len > page_size || page[sp + 2 * wl] != 0 {
                continue;
            }
            let mut sig = rd_uint(page, off, 4, le)? as u32;
            if !le && sig == 0xFFFF_FFFF && u64f {
                sig = rd_uint(page, off + 4, 4, le)? as u32;
            }
            if sig == 0xFFFFFFFD {
                blobs.push(page[off + sigl..off + len].to_vec());
            }
        }
    }
    for pg in 0..page_count {
        let start = header_size + pg * page_size;
        if start + page_size > d.len() {
            break;
        }
        let page = &d[start..start + page_size];
        let page_type = rd_uint(page, phs - 8, 2, le)? as usize;
        let block_count = rd_uint(page, phs - 6, 2, le)? as usize;
        let nsub = rd_uint(page, phs - 4, 2, le)? as usize;
        let mut data_at: Option<usize> = None;
        let mut data_rows = 0usize;
        if page_type & 0x0F00 == 0x0100 {
            data_at = Some(phs);
            data_rows = block_count;
        } else if page_type & 0x9000 == 0 {
            if phs + nsub * shps > page_size {
                return Err(rc_err(610, "damaged SAS page"));
            }
            for k in 0..nsub {
                let sp = phs + k * shps;
                let off = rd_uint(page, sp, wl, le)? as usize;
                let len = rd_uint(page, sp + wl, wl, le)? as usize;
                let comp = page[sp + 2 * wl];
                let is_comp_data = page[sp + 2 * wl + 1];
                if len == 0 || comp == 1 {
                    continue;
                }
                if off + len > page_size {
                    return Err(rc_err(610, "damaged SAS subheader"));
                }
                let sh = &page[off..off + len];
                if comp == 4 {
                    let row = if rdc { sas_rdc(sh, row_length)? } else { sas_rle(sh, row_length)? };
                    rows.push(row);
                    continue;
                }
                let mut sig = rd_uint(sh, 0, 4, le)? as u32;
                if !le && sig == 0xFFFF_FFFF && u64f {
                    sig = rd_uint(sh, 4, 4, le)? as u32;
                }
                let recognized = matches!(sig, 0xF7F7F7F7 | 0xF6F6F6F6 | 0xFFFFFC00 | 0xFFFFFBFE) || sig & 0xFFFFFFF8 == 0xFFFFFFF8;
                if is_comp_data != 0 && !recognized {
                    if len != row_length {
                        return Err(rc_err(610, "row width mismatch in SAS file"));
                    }
                    rows.push(sh.to_vec());
                    continue;
                }
                match sig {
                    0xF7F7F7F7 => {
                        if len < if u64f { 250 } else { 190 } {
                            return Err(rc_err(610, "damaged SAS row-size subheader"));
                        }
                        if u64f {
                            row_length = rd_uint(sh, 40, 8, le)? as usize;
                            row_count = rd_uint(sh, 48, 8, le)? as usize;
                            mix_rows = rd_uint(sh, 120, 8, le)? as usize;
                        } else {
                            row_length = rd_uint(sh, 20, 4, le)? as usize;
                            row_count = rd_uint(sh, 24, 4, le)? as usize;
                            mix_rows = rd_uint(sh, 60, 4, le)? as usize;
                        }
                        pending_label_and_comp(sh, len, le, &mut label, &mut rdc, &mut compressed_file, &blobs, &decode);
                    }
                    0xF6F6F6F6 => {}
                    0xFFFFFFFD => {}
                    0xFFFFFFFF => {
                        let cmax = (len - if u64f { 28 } else { 20 }) / 8;
                        for i in 0..cmax {
                            names.push(tref(sh, sigl + 8 + 8 * i));
                        }
                    }
                    0xFFFFFFFC => {
                        let stride = wl + 8;
                        let cmax = (len - if u64f { 28 } else { 20 }) / stride.max(1);
                        for i in 0..cmax {
                            let q = sigl + 8 + i * stride;
                            let off = rd_uint(sh, q, wl, le)? as usize;
                            let width = rd_uint(sh, q + wl, 4, le)? as usize;
                            let ty = sh.get(q + wl + 6).copied().unwrap_or(0);
                            attrs.push((off, width, ty == 1));
                        }
                    }
                    0xFFFFFBFE => {
                        if len >= if u64f { 58 } else { 46 } {
                            let fo = if u64f { 46 } else { 34 };
                            let lo = if u64f { 52 } else { 40 };
                            fmts.push((tref(sh, fo), tref(sh, lo)));
                        }
                    }
                    _ => {}
                }
            }
            if page_type & 0x0F00 == 0x0200 {
                let mut at = phs + nsub * shps;
                if at % 8 == 4 {
                    at += 4;
                }
                data_at = Some(at);
                data_rows = mix_rows.min(row_count.saturating_sub(rows.len()));
            }
        }
        if let Some(at) = data_at {
            if row_length == 0 {
                continue;
            }
            for k in 0..data_rows {
                let s = at + k * row_length;
                if s + row_length > page.len() {
                    return Err(rc_err(610, "row width mismatch in SAS file"));
                }
                if rows.len() >= row_count {
                    break;
                }
                rows.push(page[s..s + row_length].to_vec());
            }
        }
    }
    let _ = compressed_file;
    let ncol = attrs.len();
    if names.len() < ncol {
        return Err(rc_err(610, "damaged SAS file: column names missing"));
    }
    let mut cols_meta: Vec<SasCol> = Vec::new();
    for i in 0..ncol {
        let (f, l) = fmts.get(i).copied().unwrap_or_default();
        cols_meta.push(SasCol { name: names[i], format: f, label: l, offset: attrs[i].0, width: attrs[i].1, numeric: attrs[i].2 });
    }
    let nrows = rows.len().min(if row_count > 0 { row_count } else { rows.len() });
    let mut cols = Vec::new();
    for m in &cols_meta {
        let mut c = Col { name: resolve(&blobs, m.name), label: resolve(&blobs, m.label), numeric: m.numeric, ..Default::default() };
        let fmtname = resolve(&blobs, m.format);
        let sasfmt = sas_date_format(&fmtname);
        let (is_td, is_tc, is_time) = (sasfmt == Some("%td"), sasfmt == Some("%tc"), sasfmt == Some("%tcHH:MM:SS"));
        if m.numeric {
            c.format = sasfmt.unwrap_or("%9.0g").to_string();
            if m.width < 3 || m.width > 8 {
                return Err(rc_err(610, "invalid numeric column width in SAS file"));
            }
        }
        for row in rows.iter().take(nrows) {
            if m.offset + m.width > row.len() {
                return Err(rc_err(610, "row width mismatch in SAS file"));
            }
            let cell = &row[m.offset..m.offset + m.width];
            if m.numeric {
                let mut v = 0u64;
                if le {
                    for &x in cell.iter().rev() {
                        v = (v << 8) | x as u64;
                    }
                } else {
                    for &x in cell {
                        v = (v << 8) | x as u64;
                    }
                }
                v <<= (8 - m.width) * 8;
                let x = f64::from_bits(v);
                let mut val = if x.is_nan() {
                    // the tag byte of a SAS missing value: 0x5F = ._, 0x2E = ., 0x41.. = .a-.z? (stored inverted)
                    let tag = !((v >> 40) & 0xFF) as u8;
                    match tag {
                        b'A'..=b'Z' => missing::missing(tag - b'A' + 1),
                        _ => SYSMISS,
                    }
                } else {
                    x
                };
                if !is_missing(val) {
                    if is_tc {
                        val *= 1000.0;
                    } else if is_time {
                        val *= 1000.0;
                    } else if is_td {
                        val = val.floor();
                    }
                }
                c.nums.push(val);
            } else {
                c.strs.push(decode(cell));
            }
        }
        cols.push(c);
    }
    let mut ds = build(cols, nrows, case)?;
    ds.datalabel = label;
    Ok(ds)
}

#[allow(clippy::too_many_arguments)]
fn pending_label_and_comp(sh: &[u8], len: usize, le: bool, label: &mut String, rdc: &mut bool, compressed: &mut bool, blobs: &[Vec<u8>], decode: &dyn Fn(&[u8]) -> String) {
    // text references near the end of the row-size subheader (file label, compression)
    let g = |q: usize| -> usize { rd_uint(sh, q, 2, le).unwrap_or(0) as usize };
    let reference = |from_end: usize| -> Option<(usize, usize, usize)> {
        if len < from_end + 6 {
            return None;
        }
        let p = len - from_end;
        Some((g(p), g(p + 2), g(p + 4)))
    };
    if let Some((i, o, n)) = reference(118) {
        if n > 0 {
            if let Some(b) = blobs.get(i) {
                if o + n <= b.len() {
                    let t = &b[o..o + n];
                    *rdc = t.starts_with(b"SASYZCR2");
                    *compressed = t.starts_with(b"SASYZCR");
                }
            }
        }
    }
    if let Some((i, o, n)) = reference(130) {
        if n > 0 {
            if let Some(b) = blobs.get(i) {
                if o + n <= b.len() {
                    *label = decode(&b[o..o + n]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- XPORT

fn ibm_to_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return SYSMISS;
    }
    let mut a = [0u8; 8];
    a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
    if (a[0] == b'.' || a[0] == b'_' || a[0].is_ascii_uppercase()) && a[1..].iter().all(|&x| x == 0) {
        return if a[0].is_ascii_uppercase() { missing::missing(a[0] - b'A' + 1) } else { SYSMISS };
    }
    let sign = if a[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (a[0] & 0x7f) as i32 - 64;
    let mut mant = 0u64;
    for &x in &a[1..8] {
        mant = (mant << 8) | x as u64;
    }
    if mant == 0 {
        return 0.0;
    }
    sign * (mant as f64) / 2f64.powi(56) * 16f64.powi(exp)
}

fn f64_to_ibm(x: f64) -> [u8; 8] {
    let mut out = [0u8; 8];
    if is_missing(x) {
        let code = missing::missing_code(x).unwrap_or(0);
        out[0] = if code == 0 { b'.' } else { b'A' + code - 1 };
        return out;
    }
    if x == 0.0 {
        return out;
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let mut m = x.abs();
    let mut e = 0i32;
    while m >= 1.0 {
        m /= 16.0;
        e += 1;
    }
    while m < 1.0 / 16.0 {
        m *= 16.0;
        e -= 1;
    }
    let mut frac = (m * 2f64.powi(56)).round() as u64;
    if frac >= 1u64 << 56 {
        frac >>= 4;
        e += 1;
    }
    out[0] = sign | ((e + 64) as u8 & 0x7f);
    for k in 0..7 {
        out[1 + k] = (frac >> (8 * (6 - k))) as u8;
    }
    out
}

fn be16(b: &[u8], p: usize) -> usize {
    ((b[p] as usize) << 8) | b[p + 1] as usize
}

fn read_xport(d: &[u8], case: &str) -> Result<DataSet, String> {
    if d.len() < 80 * 5 || !d.starts_with(b"HEADER RECORD*******LIB") {
        return Err(rc_err(610, "file is not a SAS transport file"));
    }
    let rec = |i: usize| -> &[u8] { d.get(i * 80..(i + 1) * 80).unwrap_or(&[]) };
    let starts = |r: &[u8], s: &str| r.starts_with(s.as_bytes());
    let mut i = 3;
    let mut nvars = 0usize;
    let mut namestr_len = 140usize;
    let mut label = String::new();
    while i * 80 < d.len() {
        let r = rec(i);
        if starts(r, "HEADER RECORD*******MEMBER") || starts(r, "HEADER RECORD*******MEMBV8") {
            let n: String = String::from_utf8_lossy(&r[68..78]).to_string();
            if let Ok(v) = n.trim().parse::<usize>() {
                if v >= 136 {
                    namestr_len = v;
                }
            }
        } else if starts(r, "HEADER RECORD*******NAMESTR") || starts(r, "HEADER RECORD*******NAMSTV8") {
            nvars = String::from_utf8_lossy(&r[54..58]).trim().parse().map_err(|_| rc_err(610, "damaged transport header"))?;
            i += 1;
            break;
        } else if starts(r, "HEADER RECORD*******DSCRPTR") || starts(r, "HEADER RECORD*******DSCPTV8") {
            // two descriptor records follow; the second carries the data-set label
            let r2 = rec(i + 2);
            if r2.len() == 80 {
                label = trim_text(&r2[32..72]);
            }
        }
        i += 1;
    }
    let ns_bytes = nvars * namestr_len;
    let ns_start = i * 80;
    let block = d.get(ns_start..ns_start + ns_bytes).ok_or_else(|| rc_err(610, "damaged transport file"))?;
    let mut vars = Vec::new();
    for k in 0..nvars {
        let n = &block[k * namestr_len..(k + 1) * namestr_len];
        let ntype = be16(n, 0);
        let nlng = be16(n, 4);
        let mut name = trim_text(&n[8..16]);
        let lab = trim_text(&n[16..56]);
        let fmt = trim_text(&n[56..64]);
        let nfl = be16(n, 64);
        let npos = ((n[84] as usize) << 24) | ((n[85] as usize) << 16) | ((n[86] as usize) << 8) | n[87] as usize;
        if namestr_len >= 140 {
            let ln = trim_text(&n[88..120]);
            if !ln.is_empty() {
                name = ln;
            }
        }
        let _ = nfl;
        vars.push((ntype, nlng, name, lab, fmt, npos));
    }
    let mut j = i + (ns_bytes + 79) / 80;
    // optional long-name/label block (v8/v9)
    let rj = rec(j);
    if starts(rj, "HEADER RECORD*******LABELV") {
        let v9 = starts(rj, "HEADER RECORD*******LABELV9");
        let count: usize = String::from_utf8_lossy(&rj[48..78]).trim().trim_start_matches('0').parse().unwrap_or(0);
        j += 1;
        let mut p = j * 80;
        for _ in 0..count {
            let hdr = if v9 { 10 } else { 6 };
            if p + hdr > d.len() {
                break;
            }
            let num = be16(d, p);
            let nl = be16(d, p + 2);
            let ll = be16(d, p + 4);
            let (fl, il) = if v9 { (be16(d, p + 6), be16(d, p + 8)) } else { (0, 0) };
            p += hdr;
            let nm = trim_text(d.get(p..p + nl).unwrap_or(&[]));
            p += nl;
            let lb = trim_text(d.get(p..p + ll).unwrap_or(&[]));
            p += ll;
            let fm = trim_text(d.get(p..p + fl).unwrap_or(&[]));
            p += fl + il;
            if num >= 1 && num <= vars.len() {
                let v = &mut vars[num - 1];
                if !nm.is_empty() {
                    v.2 = nm;
                }
                if !lb.is_empty() {
                    v.3 = lb;
                }
                if !fm.is_empty() {
                    v.4 = fm;
                }
            }
        }
        j = p.div_ceil(80);
    }
    // OBS header
    while j * 80 < d.len() && !starts(rec(j), "HEADER RECORD*******OBS") {
        j += 1;
    }
    j += 1;
    let obs_len: usize = vars.iter().map(|v| v.1).sum();
    let data = &d[(j * 80).min(d.len())..];
    let mut nrows = if obs_len == 0 { 0 } else { data.len() / obs_len };
    // trailing blank padding can look like a row
    while nrows > 0 && data[(nrows - 1) * obs_len..nrows * obs_len].iter().all(|&b| b == b' ') {
        nrows -= 1;
    }
    let mut cols = Vec::new();
    for (ntype, nlng, name, lab, fmt, npos) in &vars {
        let mut c = Col { name: name.clone(), label: lab.clone(), numeric: *ntype == 1, ..Default::default() };
        let sf = sas_date_format(fmt);
        if c.numeric {
            c.format = sf.unwrap_or("%9.0g").to_string();
        }
        for r in 0..nrows {
            let cell = &data[r * obs_len + npos..r * obs_len + npos + nlng];
            if c.numeric {
                let mut v = ibm_to_f64(cell);
                if !is_missing(v) && matches!(sf, Some("%tc") | Some("%tcHH:MM:SS")) {
                    v *= 1000.0;
                }
                c.nums.push(v);
            } else {
                c.strs.push(trim_text(cell));
            }
        }
        cols.push(c);
    }
    let mut ds = build(cols, nrows, case)?;
    ds.datalabel = label;
    Ok(ds)
}

fn pad(s: &str, n: usize) -> Vec<u8> {
    let mut v: Vec<u8> = s.bytes().take(n).collect();
    v.resize(n, b' ');
    v
}

fn write_xport(ds: &DataSet, vars: &[String], mask: &[bool], path: &str, v8: bool) -> Result<Vec<u8>, String> {
    let mut vs = Vec::new();
    for n in vars {
        let v = ds.get_var(n).ok_or_else(|| rc_err(111, format!("variable {n} not found")))?;
        if !v8 && v.name.len() > 8 {
            return Err(rc_err(198, format!("variable {} has more than 8 characters; use export sasxport8", v.name)));
        }
        vs.push(v);
    }
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let stamp = "01JAN60:00:00:00";
    let mut out: Vec<u8> = Vec::new();
    let rec = |s: &str, out: &mut Vec<u8>| out.extend(pad(s, 80));
    let tag = |k: &str| -> String { format!("HEADER RECORD*******{k} HEADER RECORD!!!!!!!") };
    rec(&format!("{}000000000000000000000000000000", tag(if v8 { "LIBV8  " } else { "LIBRARY" })), &mut out);
    rec(&format!("{:<8}{:<8}{:<8}{:<8}{:<8}{:<24}{}", "SAS", "SAS", "SASLIB", "9.4", "FERROST", "", stamp), &mut out);
    rec(stamp, &mut out);
    rec(&format!("{}00000000000000000160{}", tag(if v8 { "MEMBV8 " } else { "MEMBER " }), "0000000140"), &mut out);
    rec(&format!("{}000000000000000000000000000000", tag(if v8 { "DSCPTV8" } else { "DSCRPTR" })), &mut out);
    let base = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("DATA").to_uppercase();
    let dsname: String = base.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    let nlen = if v8 { 32 } else { 8 };
    let mut r1 = Vec::new();
    r1.extend(pad("SAS", 8));
    r1.extend(pad(&dsname, nlen));
    r1.extend(pad("SASDATA", 8));
    r1.extend(pad("9.4", 8));
    r1.extend(pad("FERROST", 8));
    r1.extend(pad("", if v8 { 0 } else { 24 }));
    r1.extend(pad(stamp, 16));
    let mut r1p = r1.clone();
    r1p.resize(80, b' ');
    out.extend(r1p);
    let mut r2 = pad(stamp, 16);
    r2.extend(pad("", 16));
    r2.extend(pad(&ds.datalabel, 40));
    r2.extend(pad("", 8));
    out.extend(r2);
    rec(&format!("{}000000{:04}00000000000000000000", tag(if v8 { "NAMSTV8" } else { "NAMESTR" }), vs.len()), &mut out);
    // namestrs
    let mut ns = Vec::new();
    let mut pos = 0usize;
    for (k, v) in vs.iter().enumerate() {
        let numeric = v.var_type.is_numeric();
        let w = str_width(ds, v, &rows);
        let mut n = vec![0u8; 140];
        n[0..2].copy_from_slice(&(if numeric { 1u16 } else { 2u16 }).to_be_bytes());
        n[4..6].copy_from_slice(&(w as u16).to_be_bytes());
        n[6..8].copy_from_slice(&((k + 1) as u16).to_be_bytes());
        n[8..16].copy_from_slice(&pad(&v.name, 8));
        n[16..56].copy_from_slice(&pad(&v.label, 40));
        let (fname, fl) = if v.format.starts_with("%td") { ("DATE", 9u16) } else if v.format.starts_with("%tc") { ("DATETIME", 19) } else { ("", 0) };
        n[56..64].copy_from_slice(&pad(fname, 8));
        n[64..66].copy_from_slice(&fl.to_be_bytes());
        n[68..70].copy_from_slice(&(if numeric { 1u16 } else { 0u16 }).to_be_bytes());
        n[72..80].copy_from_slice(&pad("", 8));
        n[84..88].copy_from_slice(&(pos as u32).to_be_bytes());
        if v8 {
            n[88..120].copy_from_slice(&pad(&v.name, 32));
            n[120..122].copy_from_slice(&(v.label.len().min(256) as u16).to_be_bytes());
        }
        pos += w;
        ns.extend(n);
    }
    ns.resize(ns.len().div_ceil(80) * 80, b' ');
    out.extend(ns);
    if v8 {
        let mut lb = Vec::new();
        for (k, v) in vs.iter().enumerate() {
            lb.extend(((k + 1) as u16).to_be_bytes());
            lb.extend((v.name.len() as u16).to_be_bytes());
            lb.extend((v.label.len() as u16).to_be_bytes());
            lb.extend(v.name.bytes());
            lb.extend(v.label.bytes());
        }
        rec(&format!("{}{:<30}", tag("LABELV8"), vs.len()), &mut out);
        lb.resize(lb.len().div_ceil(80) * 80, b' ');
        out.extend(lb);
    }
    rec(&format!("{}000000000000000000000000000000", tag(if v8 { "OBSV8  " } else { "OBS    " })), &mut out);
    let mut data = Vec::new();
    for &i in &rows {
        for v in vs.iter() {
            if v.var_type.is_numeric() {
                let mut x = v.num(i);
                if !is_missing(x) && v.format.starts_with("%tc") {
                    x /= 1000.0;
                }
                data.extend(f64_to_ibm(x));
            } else {
                data.extend(pad(&str_of(ds, v, i), str_width(ds, v, &rows)));
            }
        }
    }
    data.resize(data.len().div_ceil(80) * 80, b' ');
    out.extend(data);
    Ok(out)
}

fn str_of(ds: &DataSet, v: &Variable, i: usize) -> String {
    match &v.data[i] {
        Value::Str(s) => s.clone(),
        Value::StrL(a, b) => ds.resolve_strl(*a, *b).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

fn str_width(ds: &DataSet, v: &Variable, rows: &[usize]) -> usize {
    match v.var_type {
        t if t.is_numeric() => 8,
        VarType::Str(n) => n,
        _ => rows.iter().map(|&i| str_of(ds, v, i).len()).max().unwrap_or(1).max(1),
    }
}
