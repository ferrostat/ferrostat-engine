//! Summary-statistics data: `ssd init`, `ssd set`, `ssd addgroup`,
//! `ssd describe`, `ssd list`, `ssd status`, `ssd unaddgroup`.
//!
//! The statistics live in the dataset characteristics (`_dta[_ssd_*]`), so
//! `save`/`use` keep them; `sem` fits models directly from them.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::model::Stats;
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;

fn ch<'a>(ds: &'a DataSet, k: &str) -> Option<&'a String> {
    ds.characteristics.get("_dta").and_then(|m| m.get(k))
}
fn set_ch(ds: &mut DataSet, k: &str, v: String) {
    ds.characteristics.entry("_dta".into()).or_default().insert(k.into(), v);
}
fn nums(s: &str) -> Result<Vec<f64>, String> {
    s.split(|c: char| c.is_whitespace() || c == '\\' || c == ',')
        .filter(|w| !w.is_empty())
        .map(|w| w.parse::<f64>().map_err(|_| rc_err(198, format!("{w} invalid number"))))
        .collect()
}
fn vec_str(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ")
}

pub fn is_ssd(ds: &DataSet) -> bool {
    ch(ds, "_ssd_vars").is_some()
}

pub fn group_var(_st: &Store) -> Option<String> {
    GV.with(|g| g.borrow().clone())
}

thread_local! {
    static GV: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn vars(ds: &DataSet) -> Vec<String> {
    ch(ds, "_ssd_vars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}
fn ngroups(ds: &DataSet) -> usize {
    ch(ds, "_ssd_groups").map(|s| s.split_whitespace().count()).unwrap_or(1).max(1)
}

/// Full (ML) statistics for `want` from summary-statistics data, per group.
pub fn stats(ds: &DataSet, want: &[String]) -> Result<Option<Vec<(String, Stats)>>, String> {
    if !is_ssd(ds) {
        return Ok(None);
    }
    let all = vars(ds);
    let p0 = all.len();
    let idx: Vec<usize> = want.iter().map(|w| all.iter().position(|a| a == w).ok_or_else(|| rc_err(111, format!("variable {w} not in summary statistics")))).collect::<Result<_, _>>()?;
    let labels: Vec<String> = ch(ds, "_ssd_groups").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_else(|| vec!["1".into()]);
    GV.with(|g| *g.borrow_mut() = ch(ds, "_ssd_gvar").cloned());
    let mut out = Vec::new();
    for (gi, lab) in labels.iter().enumerate() {
        let k = |s: &str| format!("_ssd_{s}_{}", gi + 1);
        let n: f64 = ch(ds, &k("obs")).and_then(|s| s.trim().parse().ok()).ok_or_else(|| rc_err(459, "ssd: number of observations not set"))?;
        let mean = ch(ds, &k("mean")).map(|s| nums(s)).transpose()?.unwrap_or(vec![0.0; p0]);
        let cov: Vec<f64> = if let Some(c) = ch(ds, &k("cov")) {
            nums(c)?
        } else if let Some(r) = ch(ds, &k("cor")) {
            let r = nums(r)?;
            let sd = match (ch(ds, &k("sd")), ch(ds, &k("var"))) {
                (Some(s), _) => nums(s)?,
                (None, Some(v)) => nums(v)?.iter().map(|x| x.sqrt()).collect(),
                _ => return Err(rc_err(459, "ssd: standard deviations not set")),
            };
            (0..p0 * p0).map(|t| r[t] * sd[t / p0] * sd[t % p0]).collect()
        } else {
            return Err(rc_err(459, "ssd: covariances or correlations not set"));
        };
        let p = idx.len();
        let mut c = vec![0.0; p * p];
        for a in 0..p {
            for b in 0..p {
                c[a * p + b] = cov[idx[a] * p0 + idx[b]] * (n - 1.0) / n;
            }
        }
        out.push((lab.clone(), Stats { n, mean: idx.iter().map(|&i| mean[i]).collect(), cov: c }));
    }
    Ok(Some(out))
}

/// Lower triangle (rows separated by `\`) or full matrix to a full p x p matrix.
fn full_matrix(v: &[f64], p: usize) -> Result<Vec<f64>, String> {
    let mut m = vec![0.0; p * p];
    if v.len() == p * p {
        return Ok(v.to_vec());
    }
    if v.len() == p * (p + 1) / 2 {
        let mut t = 0;
        for i in 0..p {
            for j in 0..=i {
                m[i * p + j] = v[t];
                m[j * p + i] = v[t];
                t += 1;
            }
        }
        return Ok(m);
    }
    if v.len() == p * (p - 1) / 2 {
        let mut t = 0;
        for i in 0..p {
            m[i * p + i] = 1.0;
            for j in 0..i {
                m[i * p + j] = v[t];
                m[j * p + i] = v[t];
                t += 1;
            }
        }
        return Ok(m);
    }
    Err(rc_err(198, format!("expected {} or {} values", p * (p + 1) / 2, p * p)))
}

pub struct Ssd;

impl CommandHandler for Ssd {
    fn name(&self) -> &'static str {
        "ssd"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let rest = raw.split_once(char::is_whitespace).map_or("", |x| x.1).trim();
        let (sub, tail) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let tail = tail.trim();
        match sub {
            "init" => {
                let names: Vec<String> = tail.split_whitespace().map(String::from).collect();
                if names.is_empty() {
                    return Err(rc_err(198, "varlist required"));
                }
                let mut nd = DataSet::new();
                for n in &names {
                    crate::varlist::check_name(n)?;
                    nd.add_variable(crate::util::new_double(n, vec![])).map_err(|e| rc_err(198, e))?;
                }
                *ds = nd;
                set_ch(ds, "_ssd_vars", names.join(" "));
                Ok(format!("Summary statistics data initialized.  Next use, in any order,\n    ssd set observations (required)\n    ssd set means (optional)\n    ssd set variances or ssd set sd (required unless covariances set)\n    ssd set covariances or ssd set correlations (required)"))
            }
            "addgroup" => {
                need(ds)?;
                let g = tail.split_whitespace().next().ok_or_else(|| rc_err(198, "group variable required"))?;
                if ch(ds, "_ssd_gvar").is_none() {
                    set_ch(ds, "_ssd_gvar", g.to_string());
                    set_ch(ds, "_ssd_groups", "1 2".into());
                } else {
                    let n = ngroups(ds) + 1;
                    let l = (1..=n).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
                    set_ch(ds, "_ssd_groups", l);
                }
                Ok(format!("(group {} added; use ssd set #{} ... to set its statistics)", ngroups(ds), ngroups(ds)))
            }
            "unaddgroup" => {
                need(ds)?;
                let n = ngroups(ds);
                if n <= 1 {
                    return Err(rc_err(198, "no groups to remove"));
                }
                if n == 2 {
                    if let Some(m) = ds.characteristics.get_mut("_dta") {
                        m.remove("_ssd_gvar");
                        m.remove("_ssd_groups");
                    }
                } else {
                    set_ch(ds, "_ssd_groups", (1..n).map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
                }
                Ok(String::new())
            }
            "set" => {
                need(ds)?;
                let (g, t) = match tail.strip_prefix('#') {
                    Some(r) => {
                        let (a, b) = r.split_once(char::is_whitespace).unwrap_or((r, ""));
                        (a.parse::<usize>().map_err(|_| rc_err(198, "group number invalid"))?, b.trim())
                    }
                    None => (1, tail),
                };
                if g > ngroups(ds) {
                    return Err(rc_err(198, format!("group {g} not defined")));
                }
                let (what, vals) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
                let p = vars(ds).len();
                let v = nums(vals)?;
                let key = |s: &str| format!("_ssd_{s}_{g}");
                let w = what.to_lowercase();
                let is = |full: &str, min: usize| w.len() >= min && full.starts_with(&w);
                if is("observations", 3) {
                    set_ch(ds, &key("obs"), fmt_g(*v.first().ok_or_else(|| rc_err(198, "number required"))?, 12));
                } else if is("means", 4) {
                    check_len(&v, p)?;
                    set_ch(ds, &key("mean"), vec_str(&v));
                } else if w == "sd" {
                    check_len(&v, p)?;
                    set_ch(ds, &key("sd"), vec_str(&v));
                } else if is("variances", 3) {
                    check_len(&v, p)?;
                    set_ch(ds, &key("var"), vec_str(&v));
                } else if is("covariances", 3) {
                    set_ch(ds, &key("cov"), vec_str(&full_matrix(&v, p)?));
                } else if is("correlations", 3) {
                    set_ch(ds, &key("cor"), vec_str(&full_matrix(&v, p)?));
                } else {
                    return Err(rc_err(198, format!("ssd set {what} invalid")));
                }
                Ok(String::new())
            }
            "describe" | "status" | "" => {
                need(ds)?;
                let v = vars(ds);
                let mut out = vec![format!("  Summary statistics data"), format!("    Variables: {}", v.len()), format!("    Groups:    {}", ngroups(ds))];
                for g in 1..=ngroups(ds) {
                    let has = |s: &str| ch(ds, &format!("_ssd_{s}_{g}")).is_some();
                    out.push(format!(
                        "    group {g}: obs {}  means {}  variances {}  covariances {}",
                        ch(ds, &format!("_ssd_obs_{g}")).cloned().unwrap_or_else(|| "(not set)".into()),
                        if has("mean") { "set" } else { "unset" },
                        if has("sd") || has("var") || has("cov") { "set" } else { "unset" },
                        if has("cov") || has("cor") { "set" } else { "unset" }
                    ));
                }
                out.push(format!("    {}", v.join(" ")));
                Ok(out.join("\n"))
            }
            "list" => {
                need(ds)?;
                let v = vars(ds);
                let s = stats(ds, &v)?.unwrap_or_default();
                let mut out = Vec::new();
                for (lab, st) in s {
                    let n = st.n;
                    let p = v.len();
                    out.push(format!("Group {lab}:  Observations = {n}"));
                    out.push(format!("  Means: {}", st.mean.iter().map(|x| fmt_g(*x, 9)).collect::<Vec<_>>().join(" ")));
                    out.push("  Covariances:".into());
                    for i in 0..p {
                        let row: Vec<String> = (0..=i).map(|j| fmt_g(st.cov[i * p + j] * n / (n - 1.0), 9)).collect();
                        out.push(format!("  {:>12} {}", v[i], row.join(" ")));
                    }
                }
                Ok(out.join("\n"))
            }
            o => Err(rc_err(198, format!("ssd {o} not supported"))),
        }
    }
}

fn need(ds: &DataSet) -> Result<(), String> {
    if is_ssd(ds) { Ok(()) } else { Err(rc_err(459, "data not summary statistics data; see ssd init")) }
}
fn check_len(v: &[f64], p: usize) -> Result<(), String> {
    if v.len() == p { Ok(()) } else { Err(rc_err(198, format!("{p} values required"))) }
}
