//! Comparison and integrity commands: `compare`, `checksum`,
//! `datasignature`, `assertnested`.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::{vfs, DataSet};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("compare", Box::new(Compare)),
        ("checksum", Box::new(Checksum)),
        ("datasignature", Box::new(Datasig)),
        ("assertnested", Box::new(AssertNested)),
    ]
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

// ---------------------------------------------------------------- compare

struct Compare;

fn num(x: f64) -> String {
    crate::format::fmt_g(x, 12).trim().to_string()
}

impl CommandHandler for Compare {
    fn name(&self) -> &'static str {
        "compare"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(102, "compare requires exactly two variables"));
        }
        let (a, b) = (&vars[0], &vars[1]);
        let va = ds.get_var(a).unwrap();
        let vb = ds.get_var(b).unwrap();
        if va.var_type.is_string() != vb.var_type.is_string() {
            return Err(rc_err(109, "type mismatch: one variable is string, the other numeric"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let rows = util::rows(&mask);
        let hdr = vec![
            format!("{:>40}---------- difference ----------", ""),
            format!("{:>31}{:>10}{:>14}{:>14}{:>14}", "", "Count", "Minimum", "Average", "Maximum"),
            "-".repeat(79),
        ];
        let mut out = hdr;
        let line = |label: String, n: usize, stats: Option<(f64, f64, f64)>| -> String {
            match stats {
                Some((mn, av, mx)) if n > 0 => format!("{:<31}{:>10}{:>14}{:>14}{:>14}", label, n, num(mn), num(av), num(mx)),
                _ => format!("{:<31}{:>10}", label, n),
            }
        };
        let mut joint = 0usize;
        let (mut m1, mut m2, mut mb) = (0usize, 0usize, 0usize);
        if va.var_type.is_string() {
            let (mut eq, mut ne) = (0, 0);
            for &i in &rows {
                let x = va.data[i].as_string().unwrap_or("");
                let y = vb.data[i].as_string().unwrap_or("");
                if x == y {
                    eq += 1;
                } else {
                    ne += 1;
                }
            }
            out.push(line(format!("{a}={b}"), eq, None));
            out.push(line(format!("{a}!={b}"), ne, None));
            out.push(format!("{:>41}", "----------"));
            out.push(line("Total".into(), eq + ne, None));
            st.begin_r();
            st.r.set("N", rows.len() as f64);
            return Ok(out.join("\n"));
        }
        let mut lt: Vec<f64> = Vec::new();
        let mut eq = 0usize;
        let mut gt: Vec<f64> = Vec::new();
        for &i in &rows {
            let (x, y) = (va.num(i), vb.num(i));
            match (is_missing(x), is_missing(y)) {
                (true, true) => mb += 1,
                (true, false) => m1 += 1,
                (false, true) => m2 += 1,
                _ => {
                    joint += 1;
                    if x < y {
                        lt.push(x - y);
                    } else if x > y {
                        gt.push(x - y);
                    } else {
                        eq += 1;
                    }
                }
            }
        }
        let stats = |v: &Vec<f64>| -> Option<(f64, f64, f64)> {
            if v.is_empty() {
                None
            } else {
                Some((v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)))
            }
        };
        out.push(line(format!("{a}<{b}"), lt.len(), stats(&lt)));
        out.push(line(format!("{a}={b}"), eq, if eq > 0 { Some((0.0, 0.0, 0.0)) } else { None }));
        out.push(line(format!("{a}>{b}"), gt.len(), stats(&gt)));
        out.push(format!("{:>41}", "----------"));
        out.push(line("Jointly defined".into(), joint, None));
        out.push(line(format!("{a} missing only"), m1, None));
        out.push(line(format!("{b} missing only"), m2, None));
        out.push(line("Jointly missing".into(), mb, None));
        out.push(format!("{:>41}", "----------"));
        out.push(line("Total".into(), rows.len(), None));
        let r = st.begin_r();
        r.set("N", rows.len() as f64);
        r.set("n_lt", lt.len() as f64);
        r.set("n_eq", eq as f64);
        r.set("n_gt", gt.len() as f64);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------- checksum

struct Checksum;

impl CommandHandler for Checksum {
    fn name(&self) -> &'static str {
        "checksum"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = util::unquote(cmd.main.trim());
        if path.is_empty() {
            return Err(rc_err(100, "filename required"));
        }
        let path = if !vfs::exists(&path) && vfs::exists(&format!("{path}.dta")) { format!("{path}.dta") } else { path };
        let data = vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
        let crc = crate::xlsx::crc32(&data);
        let r = st.begin_r();
        r.set("checksum", crc as f64);
        r.set("filelen", data.len() as f64);
        r.set("version", 1.0);
        let msg = format!("Checksum for {path} = {crc}, size = {}", data.len());
        if cmd.has_opt("save", 4) || cmd.has_opt("saving", 6) {
            let target = cmd.opt("saving", 6).map(|s| util::unquote(s)).unwrap_or_else(|| format!("{path}.sum"));
            if vfs::exists(&target) && !cmd.has_opt("replace", 3) {
                return Err(rc_err(602, format!("file {target} already exists")));
            }
            vfs::write(&target, format!("{crc} {}\n", data.len()).as_bytes()).map_err(|e| rc_err(603, e.to_string()))?;
        }
        Ok(msg)
    }
}

// ---------------------------------------------------------------- datasignature

fn signature(ds: &DataSet, vars: &[String]) -> String {
    let (mut h1, mut h2, mut h3) = (0xcbf29ce484222325u64, 0xcbf29ce484222325u64, 0x84222325cbf29ce4u64);
    for v in vars {
        let var = ds.get_var(v).unwrap();
        fnv(&mut h1, v.as_bytes());
        fnv(&mut h1, var.var_type.type_str().as_bytes());
        for i in 0..ds.n_obs {
            match var.data.get(i) {
                Some(x) if x.is_string() => {
                    let s = x.as_string().unwrap_or("");
                    fnv(&mut h2, s.as_bytes());
                    fnv(&mut h2, &[0]);
                    fnv(&mut h3, &[s.len() as u8]);
                }
                Some(x) => {
                    let n = x.num();
                    let bits = if is_missing(n) { u64::MAX } else { n.to_bits() };
                    fnv(&mut h2, &bits.to_le_bytes());
                    fnv(&mut h3, &bits.to_be_bytes());
                }
                None => {}
            }
        }
    }
    let w: usize = vars.iter().filter_map(|v| ds.get_var(v)).map(|v| v.var_type.storage_width()).sum();
    format!("{}:{}({}):{}:{}", ds.n_obs, vars.len(), (h1 % 100_000) as u32 + w as u32 * 0, h2 % 4_294_967_296, h3 % 4_294_967_296)
}

struct Datasig;

const SIG_KEY: &str = "datasignature_si";

impl CommandHandler for Datasig {
    fn name(&self) -> &'static str {
        "datasignature"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let (sub, rest) = match sub {
            "set" | "confirm" | "report" | "clear" => (sub, rest.trim()),
            _ => ("", main),
        };
        let vars = if rest.is_empty() { ds.varlist() } else { varlist::expand(ds, rest)? };
        let cur = signature(ds, &vars);
        let stored = ds.characteristics.get("_dta").and_then(|c| c.get(SIG_KEY)).cloned();
        st.begin_r();
        st.r.set_str("datasignature", cur.clone());
        match sub {
            "" => Ok(format!("  {cur}")),
            "set" => {
                if stored.is_some() && !cmd.has_opt("reset", 5) {
                    return Err(rc_err(459, "data signature already set; use the reset option to replace it"));
                }
                ds.characteristics.entry("_dta".into()).or_default().insert(SIG_KEY.into(), cur.clone());
                st.data_changed = true;
                Ok(format!("  {cur}\n  (data signature set)"))
            }
            "clear" => {
                if let Some(c) = ds.characteristics.get_mut("_dta") {
                    c.remove(SIG_KEY);
                }
                Ok(String::new())
            }
            "confirm" => match stored {
                None => {
                    if cmd.has_opt("strict", 3) {
                        Err(rc_err(9, "data signature not set"))
                    } else {
                        Ok(String::new())
                    }
                }
                Some(s) if s == cur => Ok(String::new()),
                Some(_) => Err(rc_err(9, "data have changed since signature was set")),
            },
            _ => {
                let mut out = vec!["Data signature summary".to_string(), String::new(), format!("      Current:  {cur}")];
                match stored {
                    Some(s) => {
                        out.push(format!("      Stored:   {s}"));
                        out.push(format!("      Status:   {}", if s == cur { "signature unchanged" } else { "DATA HAVE CHANGED" }));
                    }
                    None => out.push("      Stored:   (none)".to_string()),
                }
                Ok(out.join("\n"))
            }
        }
    }
}

// ---------------------------------------------------------------- assertnested

struct AssertNested;

impl CommandHandler for AssertNested {
    fn name(&self) -> &'static str {
        "assertnested"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let key = |v: &str, i: usize| -> String {
            let var = ds.get_var(v).unwrap();
            match var.data[i].as_string() {
                Some(s) => format!("s{s}"),
                None => format!("n{}", var.num(i)),
            }
        };
        for w in vars.windows(2) {
            let mut map: BTreeMap<String, String> = BTreeMap::new();
            for i in util::rows(&mask) {
                let (k, p) = (key(&w[0], i), key(&w[1], i));
                match map.get(&k) {
                    Some(q) if *q != p => {
                        return Err(rc_err(459, format!("{} is not nested within {}", w[0], w[1])));
                    }
                    _ => {
                        map.insert(k, p);
                    }
                }
            }
        }
        st.begin_r().set("nested", 1.0);
        Ok(String::new())
    }
}
