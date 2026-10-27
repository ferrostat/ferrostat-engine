//! JSON records: `import json` / `export json`.
//!
//! ```text
//! import json using file [, clear lines columns(namelist) rowrange(a:b)]
//! export json [varlist] using file [if] [in] [, replace lines pretty]
//! ```
//!
//! Input is either an array of flat objects (`[{"a":1,"b":"x"}, ...]`) or
//! JSON Lines (one object per line, `.jsonl`); the layout is detected from
//! the first character unless `lines` is given. Columns appear in the order
//! their keys are first seen. Numbers and booleans give numeric variables
//! (integers get the narrowest type, other numbers double), `null` and absent
//! keys give missing, strings give str#/strL. A column whose strings are all
//! ISO dates (`2024-03-01`) becomes a `%td` date, all ISO date-times
//! (`2024-03-01T12:30:00`, an offset or `Z` is converted to UTC) a `%tc`
//! date-time. Nested arrays/objects and columns mixing strings and numbers
//! are kept as text.
//!
//! Export writes numbers as JSON numbers (value labels are not applied),
//! `%td` as "YYYY-MM-DD", `%tc` as "YYYY-MM-DDTHH:MM:SS.sss", missing values
//! (including `.a`-`.z`) as null, strings as strings.

use chrono::{DateTime, NaiveDate, NaiveDateTime, Timelike};
use ferrostat_core::missing::{is_missing, missing_code, SYSMISS};
use ferrostat_core::{vfs, DataSet, Value, VarType, Variable};
use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::Value as J;

use crate::columnar::{str_cell, var_name, ReadOptions};
use crate::eval::rc_err;
use crate::util;

/// A JSON object with its keys in file order.
struct Obj(Vec<(String, J)>);

impl<'de> Deserialize<'de> for Obj {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Obj;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Obj, A::Error> {
                let mut v = Vec::new();
                while let Some((k, x)) = m.next_entry::<String, J>()? {
                    v.push((k, x));
                }
                Ok(Obj(v))
            }
        }
        d.deserialize_map(V)
    }
}

fn parse_records(text: &str, lines: bool) -> Result<Vec<Obj>, String> {
    let t = text.trim_start_matches('\u{feff}').trim_start();
    if !lines && t.starts_with('[') {
        return serde_json::from_str::<Vec<Obj>>(t).map_err(|e| rc_err(610, format!("invalid JSON: {e} (expected an array of objects)")));
    }
    let mut out = Vec::new();
    for (k, line) in t.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        out.push(serde_json::from_str::<Obj>(l).map_err(|e| rc_err(610, format!("invalid JSON on line {}: {e}", k + 1)))?);
    }
    Ok(out)
}

fn parse_date(s: &str) -> Option<f64> {
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    let epoch = NaiveDate::from_ymd_opt(1960, 1, 1).unwrap();
    Some((d - epoch).num_days() as f64)
}

fn parse_datetime(s: &str) -> Option<f64> {
    let epoch = NaiveDate::from_ymd_opt(1960, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let naive = DateTime::parse_from_rfc3339(s)
        .map(|d| d.naive_utc())
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok())
        .or_else(|| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f").ok())
        .or_else(|| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M").ok())?;
    Some((naive - epoch).num_milliseconds() as f64)
}

pub fn read_json(path: &str, opts: &ReadOptions, lines: bool) -> Result<(DataSet, Vec<String>), String> {
    let bytes = vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let text = String::from_utf8_lossy(&bytes);
    let lines = lines || path.to_lowercase().ends_with(".jsonl") || path.to_lowercase().ends_with(".ndjson");
    let mut recs = parse_records(&text, lines)?;
    // row range
    let start = opts.offset.min(recs.len());
    let end = opts.limit.map_or(recs.len(), |l| (start + l).min(recs.len()));
    recs.truncate(end);
    recs.drain(..start);
    // columns in order of first appearance
    let mut names: Vec<String> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for r in &recs {
        for (k, _) in &r.0 {
            if !index.contains_key(k) {
                index.insert(k.clone(), names.len());
                names.push(k.clone());
            }
        }
    }
    let sel = opts.select(&names)?;
    let n = recs.len();
    let mut cols: Vec<Vec<Option<J>>> = vec![vec![None; n]; names.len()];
    for (i, r) in recs.into_iter().enumerate() {
        for (k, v) in r.0 {
            let j = index[&k];
            cols[j][i] = if v.is_null() { None } else { Some(v) };
        }
    }
    let mut ds = DataSet::new();
    let mut used = std::collections::HashSet::new();
    for &j in &sel {
        let cells = std::mem::take(&mut cols[j]);
        let name = var_name(&names[j], j, &mut used);
        let present: Vec<&J> = cells.iter().flatten().collect();
        let numeric = present.iter().all(|v| v.is_number() || v.is_boolean());
        let all_str = !present.is_empty() && present.iter().all(|v| v.is_string());
        let mut v;
        let as_num = |x: &J| match x {
            J::Number(n) => n.as_f64().unwrap_or(SYSMISS),
            J::Bool(b) => *b as i32 as f64,
            _ => SYSMISS,
        };
        if numeric {
            let vals: Vec<f64> = cells.iter().map(|c| c.as_ref().map_or(SYSMISS, as_num)).collect();
            let fractional = present.iter().any(|x| matches!(x, J::Number(n) if !(n.is_i64() || n.is_u64())));
            let ty = if fractional { VarType::Double } else { util::best_numeric_type(&vals) };
            v = Variable::new(&name, ty);
            v.format = match ty {
                VarType::Double => "%10.0g".into(),
                VarType::Long => "%12.0g".into(),
                _ => "%8.0g".into(),
            };
            v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
        } else if all_str && present.iter().all(|x| parse_date(x.as_str().unwrap()).is_some()) {
            v = Variable::new(&name, VarType::Long);
            v.format = "%td".into();
            v.data = cells.iter().map(|c| util::typed_value(VarType::Long, c.as_ref().and_then(|x| parse_date(x.as_str()?)).unwrap_or(SYSMISS))).collect();
        } else if all_str && present.iter().all(|x| parse_datetime(x.as_str().unwrap()).is_some()) {
            v = Variable::new(&name, VarType::Double);
            v.format = "%tc".into();
            v.data = cells.iter().map(|c| Value::Double(c.as_ref().and_then(|x| parse_datetime(x.as_str()?)).unwrap_or(SYSMISS))).collect();
        } else {
            let strs: Vec<String> = cells
                .iter()
                .map(|c| match c {
                    None => String::new(),
                    Some(J::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                })
                .collect();
            let w = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            v = Variable::new(&name, util::str_type_for(w));
            v.format = match v.var_type {
                VarType::Str(w) => format!("%{w}s"),
                _ => "%9s".into(),
            };
            v.data = strs.into_iter().map(Value::Str).collect();
        }
        if name != names[j] {
            v.label = names[j].chars().take(80).collect();
        }
        ds.add_variable(v)?;
    }
    ds.n_obs = if ds.vars.is_empty() { 0 } else { n };
    Ok((ds, Vec::new()))
}

fn td_text(x: f64) -> String {
    let d = NaiveDate::from_ymd_opt(1960, 1, 1).unwrap() + chrono::Duration::days(x.floor() as i64);
    d.format("%Y-%m-%d").to_string()
}

fn tc_text(x: f64) -> String {
    let t = NaiveDate::from_ymd_opt(1960, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::milliseconds(x.round() as i64);
    let base = t.format("%Y-%m-%dT%H:%M:%S").to_string();
    let ms = t.nanosecond() / 1_000_000;
    format!("{base}.{ms:03}")
}

/// JSON text of one numeric cell (None for missing).
fn num_text(var: &Variable, x: f64) -> String {
    let f = var.format.as_str();
    if f.starts_with("%td") || f.starts_with("%d") {
        return serde_json::to_string(&td_text(x)).unwrap();
    }
    if f.starts_with("%tc") || f.starts_with("%tC") {
        return serde_json::to_string(&tc_text(x)).unwrap();
    }
    if x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0 {
        return format!("{}", x as i64);
    }
    if var.var_type == VarType::Float {
        serde_json::to_string(&(x as f32)).unwrap()
    } else {
        serde_json::to_string(&x).unwrap()
    }
}

pub fn write_json(ds: &DataSet, vars: &[String], rows: &[usize], lines: bool, pretty: bool) -> Result<(Vec<u8>, Vec<String>), String> {
    let mut notes = Vec::new();
    let vlist: Vec<&Variable> = vars.iter().map(|v| ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))).collect::<Result<_, _>>()?;
    let keys: Vec<String> = vars.iter().map(|v| serde_json::to_string(v).unwrap()).collect();
    let mut ext = vec![0usize; vlist.len()];
    let mut out = String::new();
    if !lines {
        out.push_str("[\n");
    }
    for (r, &i) in rows.iter().enumerate() {
        let mut fields = Vec::with_capacity(vlist.len());
        for (j, var) in vlist.iter().enumerate() {
            let val = if var.var_type.is_string() {
                serde_json::to_string(str_cell(ds, &var.data[i])).unwrap()
            } else {
                let x = var.data[i].num();
                if is_missing(x) {
                    if missing_code(x).unwrap_or(0) > 0 {
                        ext[j] += 1;
                    }
                    "null".to_string()
                } else {
                    num_text(var, x)
                }
            };
            fields.push(if pretty { format!("{}: {val}", keys[j]) } else { format!("{}:{val}", keys[j]) });
        }
        let rec = if pretty && !lines {
            format!("  {{\n    {}\n  }}", fields.join(",\n    "))
        } else if pretty {
            format!("{{{}}}", fields.join(", "))
        } else {
            format!("{{{}}}", fields.join(","))
        };
        out.push_str(&rec);
        if !lines && r + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    if !lines {
        out.push_str("]\n");
    }
    for (j, &e) in ext.iter().enumerate() {
        if e > 0 {
            notes.push(format!("note: {}: {e} extended missing value(s) (.a-.z) written as null", vars[j]));
        }
    }
    Ok((out.into_bytes(), notes))
}
