//! `import fred`: economic time series in the format served by the FRED
//! web service (JSON `observations` responses or two-column CSV downloads),
//! and `export dbase`.
//!
//! This build has no network layer, so the series are read from local files
//! named `<series>.json` or `<series>.csv`, looked up in the directory given
//! by `cachedir()` (or the global setting `set fredcache`), or in the current
//! directory. Without such files the command fails with a clear message.

use std::collections::BTreeMap;

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::{vfs, DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::dates;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;

/// (date as %td, value) pairs of one series.
pub fn parse_series(text: &str) -> Result<Vec<(f64, f64)>, String> {
    let t = text.trim_start();
    let mut out = Vec::new();
    if t.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(t).map_err(|e| format!("invalid JSON: {e}"))?;
        let obs = v.get("observations").and_then(|o| o.as_array()).ok_or("JSON has no \"observations\" array")?;
        for o in obs {
            let d = o.get("date").and_then(|d| d.as_str()).ok_or("observation without date")?;
            let x = match o.get("value") {
                Some(serde_json::Value::String(s)) => s.trim().parse::<f64>().unwrap_or(SYSMISS),
                Some(serde_json::Value::Number(n)) => n.as_f64().unwrap_or(SYSMISS),
                _ => SYSMISS,
            };
            out.push((ymd(d)?, x));
        }
    } else {
        for (k, line) in t.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (d, x) = line.split_once(',').ok_or_else(|| format!("line {}: expected date,value", k + 1))?;
            if k == 0 && d.parse::<f64>().is_err() && !d.trim().starts_with(|c: char| c.is_ascii_digit()) {
                continue; // header
            }
            out.push((ymd(d)?, x.trim().parse::<f64>().unwrap_or(SYSMISS)));
        }
    }
    Ok(out)
}

fn ymd(s: &str) -> Result<f64, String> {
    let d = dates::date(s.trim(), "YMD", None);
    if d >= SYSMISS {
        return Err(format!("invalid date {s}"));
    }
    Ok(d)
}

/// `import fred series1 [series2 ...] [, clear daterange(start end) cachedir(dir) aggregate(...)]`
pub fn import(rest: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let series: Vec<String> = rest.split_whitespace().map(|s| s.to_string()).collect();
    if series.is_empty() {
        return Err(rc_err(198, "series codes required"));
    }
    if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
    }
    let dir = cmd.opt("cachedir", 5).map(|s| util::unquote(s.trim())).or_else(|| st.settings.get("fredcache").cloned());
    let range = match cmd.opt("daterange", 5) {
        Some(r) => {
            let w: Vec<&str> = r.split_whitespace().collect();
            let p = |s: Option<&&str>| -> Result<f64, String> {
                match s {
                    None | Some(&".") => Ok(SYSMISS),
                    Some(x) => ymd(x).map_err(|e| rc_err(198, e)),
                }
            };
            (p(w.first())?, p(w.get(1))?)
        }
        None => (SYSMISS, SYSMISS),
    };
    let mut table: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (k, s) in series.iter().enumerate() {
        let candidates: Vec<String> = ["json", "csv"]
            .iter()
            .flat_map(|ext| {
                let f = format!("{s}.{ext}");
                match &dir {
                    Some(d) => vec![format!("{}/{f}", d.trim_end_matches('/'))],
                    None => vec![f],
                }
            })
            .collect();
        let path = candidates.iter().find(|p| vfs::exists(p)).ok_or_else(|| {
            rc_err(
                631,
                format!("series {s}: no network access in this build and no local copy found ({}); download the series and point cachedir() to its folder", candidates.join(", ")),
            )
        })?;
        let text = vfs::read_to_string(path).map_err(|e| rc_err(603, format!("{path}: {e}")))?;
        let obs = parse_series(&text).map_err(|e| rc_err(610, format!("{path}: {e}")))?;
        for (d, x) in obs {
            if (range.0 < SYSMISS && d < range.0) || (range.1 < SYSMISS && d > range.1) {
                continue;
            }
            let row = table.entry(d as i64).or_insert_with(|| vec![SYSMISS; series.len()]);
            row[k] = x;
        }
    }
    let mut new = DataSet::new();
    let mut ds_s = Variable::new("datestr", VarType::Str(10));
    ds_s.format = "%10s".into();
    let mut dd = Variable::new("daten", VarType::Long);
    dd.format = "%td".into();
    dd.label = "Observation date".into();
    for &d in table.keys() {
        ds_s.push(Value::Str(dates::format_date(d as f64, "%tdCCYY-NN-DD")));
        dd.push(Value::Long(d as i32));
    }
    new.add_variable(ds_s).map_err(|e| rc_err(110, e))?;
    new.add_variable(dd).map_err(|e| rc_err(110, e))?;
    for (k, s) in series.iter().enumerate() {
        let mut v = Variable::new(s.as_str(), VarType::Double);
        v.format = "%10.0g".into();
        v.label = s.clone();
        for row in table.values() {
            v.push(if row[k] >= SYSMISS { Value::Missing } else { Value::Double(row[k]) });
        }
        new.add_variable(v).map_err(|e| rc_err(110, e))?;
    }
    new.n_obs = table.len();
    *ds = new;
    st.data_changed = false;
    st.tsset = None;
    st.filename.clear();
    Ok(format!("({} series, {} observations)", series.len(), table.len()))
}

// ---------------------------------------------------------------- dBase output

/// `export dbase [varlist] using filename [if] [in] [, replace datafmt]`
pub fn export_dbase(rest: &str, ds: &DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let (vars, path) = match &cmd.using {
        Some(p) => (if rest.trim().is_empty() { ds.varlist() } else { crate::varlist::expand(ds, rest)? }, p.clone()),
        None => (ds.varlist(), util::unquote(rest.trim())),
    };
    let path = util::unquote(&path);
    let path = if std::path::Path::new(&path).extension().is_none() { format!("{path}.dbf") } else { path };
    if vfs::exists(&path) && !cmd.has_opt("replace", 3) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut sub = cmd.clone();
    sub.main.clear();
    let mask = util::touse(ds, st, &sub)?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let datafmt = cmd.has_opt("datafmt", 4);
    // per field: (name, type, width, decimals, cell texts)
    let mut fields: Vec<(String, u8, usize, usize, Vec<String>)> = Vec::new();
    for name in &vars {
        if name.len() > 10 {
            return Err(rc_err(198, format!("variable name {name} is longer than 10 characters, the dBase limit")));
        }
        let v = ds.get_var(name).unwrap();
        if v.var_type.is_string() {
            let cells: Vec<String> = rows.iter().map(|&i| util::cell_text(ds, v, i, false)).collect();
            let w = cells.iter().map(|c| c.len()).max().unwrap_or(1).clamp(1, 254);
            fields.push((name.clone(), b'C', w, 0, cells));
        } else {
            let vals: Vec<f64> = rows.iter().map(|&i| v.num(i)).collect();
            let integer = vals.iter().all(|x| *x >= SYSMISS || x.fract() == 0.0);
            let dec = if integer { 0 } else { 8 };
            let cells: Vec<String> = vals
                .iter()
                .map(|&x| {
                    if x >= SYSMISS {
                        String::new()
                    } else if datafmt {
                        crate::format::fmt(x, &v.format).trim().to_string()
                    } else if integer {
                        format!("{x:.0}")
                    } else {
                        format!("{x:.dec$}")
                    }
                })
                .collect();
            let w = cells.iter().map(|c| c.len()).max().unwrap_or(1).clamp(1, 20);
            if cells.iter().any(|c| c.len() > 20) {
                return Err(rc_err(198, format!("variable {name} has values too wide for a dBase numeric field")));
            }
            fields.push((name.clone(), b'N', w.max(if integer { 1 } else { dec + 2 }), dec, cells));
        }
    }
    let rec_len: usize = 1 + fields.iter().map(|f| f.2).sum::<usize>();
    let header_len = 32 + 32 * fields.len() + 1;
    let mut buf = Vec::with_capacity(header_len + rec_len * rows.len() + 1);
    let today = chrono::Local::now();
    use chrono::Datelike;
    buf.push(0x03);
    buf.push((today.year() - 1900).clamp(0, 255) as u8);
    buf.push(today.month() as u8);
    buf.push(today.day() as u8);
    buf.extend((rows.len() as u32).to_le_bytes());
    buf.extend((header_len as u16).to_le_bytes());
    buf.extend((rec_len as u16).to_le_bytes());
    buf.extend([0u8; 20]);
    for (name, ty, w, dec, _) in &fields {
        let mut n = [0u8; 11];
        n[..name.len()].copy_from_slice(name.as_bytes());
        buf.extend(n);
        buf.push(*ty);
        buf.extend([0u8; 4]);
        buf.push(*w as u8);
        buf.push(*dec as u8);
        buf.extend([0u8; 14]);
    }
    buf.push(0x0D);
    for r in 0..rows.len() {
        buf.push(b' ');
        for (_, ty, w, _, cells) in &fields {
            let c = &cells[r];
            let mut bytes: Vec<u8> = c.bytes().take(*w).collect();
            if *ty == b'N' {
                let mut padded = vec![b' '; w - bytes.len()];
                padded.append(&mut bytes);
                buf.extend(padded);
            } else {
                bytes.resize(*w, b' ');
                buf.extend(bytes);
            }
        }
    }
    buf.push(0x1A);
    vfs::write(&path, &buf).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
    Ok(format!("file {path} saved"))
}
