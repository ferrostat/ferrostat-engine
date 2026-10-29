//! Time-series data utilities: `tsfill`, `tsappend`, `tsreport`, `tsrevar`
//! and `psdensity` (parametric spectral density after arima/arfima/ucm).

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::{Results, Store};
use crate::tsops;
use crate::util;

fn ts_vars(st: &Store) -> Result<(Option<String>, String), String> {
    match &st.tsset {
        Some((p, t)) if !t.is_empty() => Ok((p.clone(), t.clone())),
        _ => Err(rc_err(111, "time variable not set, use tsset varname ...")),
    }
}

/// Append observations with the given (panel, time) values, then sort.
fn append_obs(ds: &mut DataSet, pvar: &Option<String>, tvar: &str, new: &[(Option<f64>, f64)]) -> Result<(), String> {
    let start = ds.n_obs;
    for _ in new {
        ds.add_observation();
    }
    let tty = ds.get_var(tvar).map(|v| v.var_type).ok_or_else(|| rc_err(111, format!("variable {tvar} not found")))?;
    let pty = pvar.as_ref().and_then(|p| ds.get_var(p)).map(|v| v.var_type);
    for (k, (p, t)) in new.iter().enumerate() {
        let i = start + k;
        if !util::fits(tty, *t) {
            let v = ds.get_var_mut(tvar).unwrap();
            v.var_type = ferrostat_core::VarType::Double;
            for x in v.data.iter_mut() {
                *x = util::typed_value(ferrostat_core::VarType::Double, x.num());
            }
        }
        let ty = ds.get_var(tvar).unwrap().var_type;
        ds.get_var_mut(tvar).unwrap().data[i] = util::typed_value(ty, *t);
        if let (Some(pn), Some(pv), Some(pty)) = (pvar, p, pty) {
            ds.get_var_mut(pn).unwrap().data[i] = util::typed_value(pty, *pv);
        }
    }
    let mut keys = Vec::new();
    if let Some(p) = pvar {
        keys.push(p.clone());
    }
    keys.push(tvar.to_string());
    ds.sort(&keys, true).map_err(|e| rc_err(198, e))?;
    Ok(())
}

fn panels(ds: &DataSet, pvar: &Option<String>, tvar: &str) -> Result<BTreeMap<i64, (Option<f64>, Vec<f64>)>, String> {
    let tc = util::col(ds, tvar)?;
    let pc = match pvar {
        Some(p) => Some(util::col(ds, p)?),
        None => None,
    };
    let mut m: BTreeMap<i64, (Option<f64>, Vec<f64>)> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if is_missing(tc[i]) {
            continue;
        }
        let pv = pc.as_ref().map(|c| c[i]);
        if pv.is_some_and(is_missing) {
            continue;
        }
        let key = pv.map_or(0, |v| (v * 1e6).round() as i64);
        m.entry(key).or_insert((pv, vec![])).1.push(tc[i]);
    }
    for v in m.values_mut() {
        v.1.sort_by(f64::total_cmp);
        v.1.dedup();
    }
    Ok(m)
}

pub struct Tsfill;

impl CommandHandler for Tsfill {
    fn name(&self) -> &'static str {
        "tsfill"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (pvar, tvar) = ts_vars(st)?;
        let full = cmd.has_opt("full", 4);
        let pm = panels(ds, &pvar, &tvar)?;
        let gmin = pm.values().filter_map(|v| v.1.first().copied()).fold(f64::INFINITY, f64::min);
        let gmax = pm.values().filter_map(|v| v.1.last().copied()).fold(f64::NEG_INFINITY, f64::max);
        let mut new = Vec::new();
        for (pv, times) in pm.values() {
            if times.is_empty() {
                continue;
            }
            let (a, b) = if full && pvar.is_some() { (gmin, gmax) } else { (times[0], *times.last().unwrap()) };
            let have: std::collections::HashSet<i64> = times.iter().map(|t| t.round() as i64).collect();
            let mut t = a;
            while t <= b + 1e-9 {
                if !have.contains(&(t.round() as i64)) {
                    new.push((*pv, t));
                }
                t += 1.0;
            }
        }
        if !new.is_empty() {
            append_obs(ds, &pvar, &tvar, &new)?;
            st.data_changed = true;
        }
        Ok(String::new())
    }
}

pub struct Tsappend;

impl CommandHandler for Tsappend {
    fn name(&self) -> &'static str {
        "tsappend"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (pvar, tvar) = ts_vars(st)?;
        let add = util::opt_num(cmd, "add", 3)?;
        let last = cmd.opt("last", 4);
        if add.is_some() == last.is_some() {
            return Err(rc_err(198, "exactly one of add() or last() must be specified"));
        }
        let last_val = match last {
            Some(s) => {
                let s = s.trim();
                match s.parse::<f64>() {
                    Ok(v) => Some(v),
                    Err(_) => {
                        let fmt = cmd.opt("tsfmt", 4).map(|f| f.trim().to_string()).unwrap_or_else(|| ds.get_var(&tvar).map(|v| v.format.clone()).unwrap_or_default());
                        let f = fmt.trim_start_matches('%').trim_start_matches('t').chars().next().unwrap_or('d');
                        let fun = match f {
                            'q' => "tq",
                            'm' => "tm",
                            'w' => "tw",
                            'h' => "th",
                            'y' => "ty",
                            _ => "td",
                        };
                        let ex = ferrostat_parser::parser::parse_expression(&format!("{fun}({s})")).map_err(|e| rc_err(198, e.message))?;
                        Some(crate::eval::Env::with_store(ds, st).eval(&ex, 0)?.num()?)
                    }
                }
            }
            None => None,
        };
        let pm = panels(ds, &pvar, &tvar)?;
        let mut new = Vec::new();
        for (pv, times) in pm.values() {
            let Some(&tl) = times.last() else { continue };
            let end = match (add, last_val) {
                (Some(a), _) => tl + a.max(0.0).round(),
                (_, Some(l)) => l,
                _ => tl,
            };
            let mut t = tl + 1.0;
            while t <= end + 1e-9 {
                new.push((*pv, t));
                t += 1.0;
            }
        }
        if !new.is_empty() {
            append_obs(ds, &pvar, &tvar, &new)?;
            st.data_changed = true;
        }
        Ok(String::new())
    }
}

pub struct Tsreport;

impl CommandHandler for Tsreport {
    fn name(&self) -> &'static str {
        "tsreport"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (pvar, tvar) = ts_vars(st)?;
        let mask = util::touse(ds, st, cmd)?;
        let vars: Vec<String> = if cmd.main.trim().is_empty() { vec![] } else { tsops::expand_text(cmd.main.trim()).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()? };
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| tsops::values(ds, st, v)).collect::<Result<_, _>>()?;
        let tc = util::col(ds, &tvar)?;
        let pc = match &pvar {
            Some(p) => Some(util::col(ds, p)?),
            None => None,
        };
        let mut groups: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for i in 0..ds.n_obs {
            if !mask[i] || is_missing(tc[i]) || cols.iter().any(|c| is_missing(c[i])) {
                continue;
            }
            let key = pc.as_ref().map_or(0, |c| (c[i] * 1e6).round() as i64);
            groups.entry(key).or_default().push(tc[i]);
        }
        let mut n_gaps = 0usize;
        let mut gap_list = Vec::new();
        let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for (k, times) in groups.iter_mut() {
            times.sort_by(f64::total_cmp);
            for w in times.windows(2) {
                if w[1] - w[0] > 1.0 + 1e-9 {
                    n_gaps += 1;
                    gap_list.push((*k, w[0] + 1.0, w[1] - 1.0));
                }
            }
            tmin = tmin.min(times[0]);
            tmax = tmax.max(*times.last().unwrap());
        }
        let lab = |v: f64| tsops::time_label(ds, &tvar, v);
        let mut out = Vec::new();
        out.push(String::new());
        if let Some(p) = &pvar {
            out.push(format!("Panel variable: {p}"));
        }
        if tmin.is_finite() {
            out.push(format!(" Time variable: {tvar}, {} to {}", lab(tmin), lab(tmax)));
        } else {
            out.push(format!(" Time variable: {tvar}"));
        }
        out.push(format!("         Delta: 1 {}", tsops::periodicity(ds, &tvar).map_or("unit", |p| match p as i64 {
            4 => "quarter",
            12 => "month",
            1 => "year",
            2 => "half-year",
            52 => "week",
            _ => "day",
        })));
        out.push(String::new());
        out.push(format!("Number of gaps in sample: {n_gaps}"));
        if (cmd.has_opt("report", 3) || cmd.has_opt("list", 1)) && !gap_list.is_empty() {
            out.push(String::new());
            out.push("Gap report".into());
            for (_, a, b) in &gap_list {
                out.push(format!("  {} - {}  ({} period{})", lab(*a), lab(*b), fmt_g(b - a + 1.0, 9), if b - a > 0.5 { "s" } else { "" }));
            }
        }
        let mut r = Results::new();
        r.set("N_gaps", n_gaps as f64);
        r.set("N_obs", groups.values().map(|v| v.len()).sum::<usize>() as f64);
        if tmin.is_finite() {
            r.set("start", tmin);
            r.set("end", tmax);
            r.set_str("tsfmt", ds.get_var(&tvar).map(|v| v.format.clone()).unwrap_or_default());
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Tsrevar;

impl CommandHandler for Tsrevar {
    fn name(&self) -> &'static str {
        "tsrevar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let toks: Vec<String> = tsops::expand_text(cmd.main.trim()).split_whitespace().map(String::from).collect();
        let list_only = cmd.has_opt("list", 4);
        let substitute = cmd.has_opt("substitute", 3);
        let mut out = Vec::new();
        let mut counter = 0usize;
        for t in toks {
            let r = tsops::resolve(ds, &t)?;
            match r.split_once('.') {
                Some((_, base)) if !(base.is_empty()) => {
                    if list_only {
                        if !out.contains(&base.to_string()) {
                            out.push(base.to_string());
                        }
                        continue;
                    }
                    let vals = tsops::values(ds, st, &r)?;
                    let name = loop {
                        let n = format!("__{:06}", counter);
                        counter += 1;
                        if !ds.var_index.contains_key(&n) {
                            break n;
                        }
                    };
                    let mut v = util::new_double(&name, vals);
                    v.var_type = ferrostat_core::VarType::Float;
                    ds.add_variable(v).map_err(|e| rc_err(198, e))?;
                    out.push(name);
                }
                _ => out.push(r),
            }
        }
        let _ = substitute;
        let mut res = Results::new();
        res.set_str("varlist", out.join(" "));
        st.r = res;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// psdensity
// ---------------------------------------------------------------------------

/// Power spectrum of an ARMA(+fractional d) process at frequency w:
/// sigma2/(2 pi) |1 + sum m e^{-iwl}|^2 / |1 - sum a e^{-iwl}|^2 |1 - e^{-iw}|^{-2d}.
pub fn arma_spectrum(a: &[f64], m: &[f64], sigma2: f64, d: f64, w: f64) -> f64 {
    let poly = |c: &[f64], sign: f64| -> f64 {
        let (mut re, mut im) = (1.0, 0.0);
        for (l, v) in c.iter().enumerate() {
            let ang = w * (l + 1) as f64;
            re += sign * v * ang.cos();
            im -= sign * v * ang.sin();
        }
        re * re + im * im
    };
    let mut s = sigma2 / (2.0 * std::f64::consts::PI) * poly(m, 1.0) / poly(a, -1.0);
    if d != 0.0 {
        s *= (2.0 * (w / 2.0).sin()).abs().powf(-2.0 * d);
    }
    s
}

pub struct Psdensity;

impl CommandHandler for Psdensity {
    fn name(&self) -> &'static str {
        "psdensity"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let names: Vec<&str> = cmd.main.split_whitespace().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).collect();
        if names.len() != 2 {
            return Err(rc_err(198, "two new variable names required"));
        }
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let get = |name: &str| b.colnames.iter().position(|c| c == name).map(|j| b.data[j]);
        let collect = |eq: &str, w: &str| -> Vec<f64> {
            let mut v: Vec<(usize, f64)> = b
                .colnames
                .iter()
                .enumerate()
                .filter_map(|(j, c)| {
                    let (e, n) = c.split_once(':')?;
                    if e != eq {
                        return None;
                    }
                    let (op, var) = n.split_once('.')?;
                    if var != w {
                        return None;
                    }
                    let l: usize = op.trim_start_matches('L').parse().unwrap_or(1);
                    Some((l, b.data[j]))
                })
                .collect();
            v.sort_by_key(|x| x.0);
            let maxl = v.last().map_or(0, |x| x.0);
            let mut out = vec![0.0; maxl];
            for (l, c) in v {
                out[l - 1] = c;
            }
            out
        };
        let (a, m, s2, d) = match ecmd.as_str() {
            "arima" => {
                let sig = st.e.get("sigma").unwrap_or(1.0);
                (collect("ARMA", "ar"), collect("ARMA", "ma"), sig * sig, 0.0)
            }
            "arfima" => (collect("ARFIMA", "ar"), collect("ARFIMA", "ma"), get("/:sigma2").unwrap_or(1.0), if cmd.has_opt("smemory", 2) { 0.0 } else { get("ARFIMA:d").unwrap_or(0.0) }),
            "ucm" => {
                // stochastic cycle: rho, lambda, var(cycle)
                let rho = get("cycle1:damping").ok_or_else(|| rc_err(321, "no cycle in the model"))?;
                let lam = get("cycle1:frequency").unwrap_or(0.0);
                let var = get("/:var(cycle1)").unwrap_or(1.0);
                // spectrum of a stochastic cycle (Harvey 1989)
                let pi = std::f64::consts::PI;
                let n = ds.n_obs.max(2);
                let (lo, hi) = range(cmd, n)?;
                let mut fr = vec![SYSMISS; ds.n_obs];
                let mut sp = vec![SYSMISS; ds.n_obs];
                let cyc_var = var / (1.0 - rho * rho);
                for i in 0..ds.n_obs {
                    let w = lo + (hi - lo) * i as f64 / (ds.n_obs.max(2) - 1) as f64;
                    let num = 1.0 + rho * rho - 2.0 * rho * lam.cos() * w.cos();
                    let den = 1.0 + 4.0 * rho * rho * lam.cos().powi(2) + rho.powi(4) - 4.0 * (rho + rho.powi(3)) * lam.cos() * w.cos() + 2.0 * rho * rho * (2.0 * w).cos();
                    let pw = var / (2.0 * pi) * num / den;
                    fr[i] = w;
                    sp[i] = if cmd.has_opt("pspectrum", 2) { pw } else { pw / cyc_var };
                }
                tsops::add_var(ds, names[0], sp)?;
                tsops::add_var(ds, names[1], fr)?;
                return Ok(String::new());
            }
            _ => return Err(rc_err(321, "psdensity is allowed only after arima, arfima and ucm")),
        };
        if !st.e.macros.get("seasons").map_or(true, |s| s.trim().is_empty()) {
            return Err(rc_err(198, "multiplicative seasonal models are not supported"));
        }
        let n = ds.n_obs.max(2);
        let (lo, hi) = range(cmd, n)?;
        // variance of the process for normalization: integral of the power spectrum
        let grid = 4096;
        let mut gamma0 = 0.0;
        for i in 0..grid {
            let w = std::f64::consts::PI * (i as f64 + 0.5) / grid as f64;
            gamma0 += 2.0 * arma_spectrum(&a, &m, s2, d, w) * std::f64::consts::PI / grid as f64;
        }
        let mut fr = vec![SYSMISS; ds.n_obs];
        let mut sp = vec![SYSMISS; ds.n_obs];
        for i in 0..ds.n_obs {
            let w = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            let p = arma_spectrum(&a, &m, s2, d, w);
            fr[i] = w;
            sp[i] = if cmd.has_opt("pspectrum", 2) { p } else { p / gamma0 };
            if d > 0.0 && w == 0.0 {
                sp[i] = SYSMISS;
            }
        }
        tsops::add_var(ds, names[0], sp)?;
        tsops::add_var(ds, names[1], fr)?;
        Ok(String::new())
    }
}

fn range(cmd: &Command, _n: usize) -> Result<(f64, f64), String> {
    let pi = std::f64::consts::PI;
    match cmd.opt("range", 1) {
        Some(r) => {
            let v: Vec<f64> = r.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).map(|w| w.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "invalid range()"))?;
            if v.len() != 2 || v[0] < 0.0 || v[1] > pi + 1e-12 || v[0] >= v[1] {
                return Err(rc_err(198, "range() must contain two increasing numbers in [0, pi]"));
            }
            Ok((v[0], v[1]))
        }
        None => Ok((0.0, pi)),
    }
}
