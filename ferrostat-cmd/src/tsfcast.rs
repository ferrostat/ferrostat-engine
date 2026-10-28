//! Dynamic forecasts after `var`, `svar` and `vec`: `fcast compute` and
//! `fcast graph` (also the older `varfcast compute|graph`).
//!
//! Forecast variables carry characteristics (`fcast_endog`, `fcast_lb`,
//! `fcast_ub`, `fcast_se`) naming the forecast series and its bounds, which
//! `fcast graph` uses.

use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsla::{delta_var, jacobian, mul, tr};
use crate::tsvar::{self, a_mats_from, eqname, exog_coefs, series, ts, Ts};
use crate::util;

/// Levels representation y_t = Σ A_l y_{t-l} + c0 + c1·trend + Bx x_t.
struct Lev {
    endog: Vec<String>,
    a: Vec<Matrix>,
    c0: Vec<f64>,
    c1: Vec<f64>,
    trend0: f64,
    exog: Vec<String>,
    bx: Matrix,
    /// innovation covariance for the MSE and its scale factor
    sigma: Matrix,
    scale: f64,
    /// VAR parameters for the estimation-uncertainty term
    var: Option<(tsvar::Spec, Vec<f64>, Matrix, usize)>,
    tmax: f64,
    is_vec: bool,
}

fn lev_from_e(e: &Results) -> Result<Lev, String> {
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    match cmd.as_str() {
        "var" | "svar" => {
            let spec = tsvar::spec_from_e(e)?;
            let svar = cmd == "svar";
            let b = e.matrices.get(if svar { "b_var" } else { "b" }).cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
            let v = e.matrices.get(if svar { "V_var" } else { "V" }).cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
            let m = spec.regnames().len();
            let k = spec.endog.len();
            let c0: Vec<f64> = (0..k).map(|i| if spec.cons { b.data[i * m + m - 1] } else { 0.0 }).collect();
            Ok(Lev {
                endog: spec.endog.clone(),
                a: a_mats_from(&spec, &b.data, m),
                c0,
                c1: vec![0.0; k],
                trend0: 0.0,
                exog: spec.exog.clone(),
                bx: exog_coefs(&spec, &b.data, m),
                sigma: e.matrices.get("Sigma").cloned().ok_or_else(|| rc_err(301, "e(Sigma) not found"))?,
                scale: 1.0,
                var: Some((spec, b.data.clone(), v, m)),
                tmax: e.get("tmax").unwrap_or(SYSMISS),
                is_vec: false,
            })
        }
        "vec" => {
            let rep = crate::tsvec::rep_from_e(e)?;
            let k = rep.endog.len();
            Ok(Lev {
                endog: rep.endog.clone(),
                a: rep.a.clone(),
                c0: rep.c0.clone(),
                c1: rep.c1.clone(),
                trend0: rep.tmin,
                exog: Vec::new(),
                bx: Matrix::zeros(k, 0),
                sigma: rep.omega.clone(),
                scale: rep.t / (rep.t - rep.d),
                var: None,
                tmax: e.get("tmax").unwrap_or(SYSMISS),
                is_vec: true,
            })
        }
        _ => Err(rc_err(301, "fcast compute requires var, svar or vec estimation results")),
    }
}

/// Forecasts for times t0..t0+h-1 given actual values before t0.
#[allow(clippy::too_many_arguments)]
fn path(a: &[Matrix], c0: &[f64], c1: &[f64], trend0: f64, bx: &Matrix, xs: &[Vec<f64>], hist: &dyn Fn(i64) -> Option<Vec<f64>>, t0: i64, h: usize) -> Result<Vec<Vec<f64>>, String> {
    let k = c0.len();
    let mut fc: Vec<Vec<f64>> = Vec::with_capacity(h);
    for s in 0..h {
        let tau = t0 + s as i64;
        let mut y: Vec<f64> = (0..k).map(|i| c0[i] + c1[i] * (tau as f64 - trend0 + 1.0)).collect();
        for i in 0..k {
            for (j, xv) in xs.get(s).map(|v| v.as_slice()).unwrap_or(&[]).iter().enumerate() {
                y[i] += bx[(i, j)] * xv;
            }
        }
        for (l, al) in a.iter().enumerate() {
            let lag = l as i64 + 1;
            let prev = if tau - lag >= t0 { fc[(tau - lag - t0) as usize].clone() } else { hist(tau - lag).ok_or_else(|| rc_err(459, "missing values of the dependent variables before the forecast period"))? };
            for i in 0..k {
                for j in 0..k {
                    y[i] += al[(i, j)] * prev[j];
                }
            }
        }
        fc.push(y);
    }
    Ok(fc)
}

fn phis(a: &[Matrix], h: usize) -> Vec<Matrix> {
    let k = a.first().map_or(0, |m| m.rows);
    let mut phi = vec![Matrix::identity(k)];
    for i in 1..h {
        let mut m = Matrix::zeros(k, k);
        for j in 1..=i.min(a.len()) {
            m = crate::tsla::add(&m, &mul(&phi[i - j], &a[j - 1]));
        }
        phi.push(m);
    }
    phi
}

fn set_time_value(ds: &mut DataSet, tvar: &str, row: usize, t: f64) {
    if let Some(v) = ds.get_var_mut(tvar) {
        let ty = v.var_type;
        v.data[row] = util::typed_value(ty, t);
    }
}

pub fn compute(ds: &mut DataSet, st: &mut Store, cmd: &Command, prefix: &str, suffix_style: bool) -> Result<String, String> {
    let e = tsvar::est_results(st, cmd)?;
    let lev = lev_from_e(&e)?;
    let k = lev.endog.len();
    let h = util::opt_num(cmd, "step", 3)?.unwrap_or(1.0);
    if h < 1.0 {
        return Err(rc_err(198, "step() must be positive"));
    }
    let h = h as usize;
    let t0 = match util::opt_num(cmd, "dynamic", 3)? {
        Some(t) => t.round() as i64,
        None => (lev.tmax + 1.0).round() as i64,
    };
    let level = util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
    let names: Vec<String> = lev.endog.iter().map(|n| if suffix_style { format!("{}{prefix}", eqname(n)) } else { format!("{prefix}{}", eqname(n)) }).collect();
    let nose = cmd.has_opt("nose", 4);
    let replace = cmd.has_opt("replace", 3);
    let mut all_names = names.clone();
    if !nose {
        for n in &names {
            for s in ["_LB", "_UB", "_SE"] {
                all_names.push(format!("{n}{s}"));
            }
        }
    }
    for n in &all_names {
        crate::varlist::check_name(n)?;
        if ds.var_index.contains_key(n.as_str()) {
            if replace {
                ds.drop_variable(n)?;
            } else {
                return Err(rc_err(110, format!("variable {n} already defined")));
            }
        }
    }
    // rows for the forecast period
    let mut tsd = ts(ds, st)?;
    let mut appended = false;
    for s in 0..h {
        let tau = (t0 + s as i64) as f64;
        if tsd.row_at(tau).is_none() {
            ds.add_observation();
            let row = ds.n_obs - 1;
            set_time_value(ds, &tsd.tvar.clone(), row, tau);
            appended = true;
        }
    }
    if appended {
        tsd = ts(ds, st)?;
    }
    let ycols: Vec<Vec<f64>> = lev.endog.iter().map(|n| series(ds, st, n)).collect::<Result<_, _>>()?;
    let xcols: Vec<Vec<f64>> = lev.exog.iter().map(|n| series(ds, st, n)).collect::<Result<_, _>>()?;
    let hist = |tsd: &Ts, ycols: &[Vec<f64>], t: i64| -> Option<Vec<f64>> {
        let r = tsd.row_at(t as f64)?;
        let v: Vec<f64> = ycols.iter().map(|c| c[r]).collect();
        if v.iter().any(|x| is_missing(*x)) { None } else { Some(v) }
    };
    let xs: Vec<Vec<f64>> = (0..h)
        .map(|s| {
            let r = tsd.row_at((t0 + s as i64) as f64).unwrap();
            let v: Vec<f64> = xcols.iter().map(|c| c[r]).collect();
            if v.iter().any(|x| is_missing(*x)) { Err(rc_err(459, "exogenous variables have missing values in the forecast period")) } else { Ok(v) }
        })
        .collect::<Result<_, _>>()?;
    let fc = path(&lev.a, &lev.c0, &lev.c1, lev.trend0, &lev.bx, &xs, &|t| hist(&tsd, &ycols, t), t0, h)?;
    // forecast-error variances
    let mut var = vec![vec![0.0; k]; h];
    if !nose {
        let phi = phis(&lev.a, h);
        let mut acc = Matrix::zeros(k, k);
        for s in 0..h {
            acc = crate::tsla::add(&acc, &mul(&mul(&phi[s], &lev.sigma), &tr(&phi[s])));
            for i in 0..k {
                var[s][i] = acc[(i, i)] * lev.scale;
            }
        }
        // estimation uncertainty (VAR without exogenous variables)
        if let (Some((spec, b, v, m)), true) = (&lev.var, lev.exog.is_empty()) {
            let f = tsvar::fit_from_e(ds, st, &e)?;
            let mut add = vec![vec![0.0; k]; h];
            let mut n = 0.0;
            for &row in &f.data.rows {
                let origin = tsd.time[row].round() as i64;
                let fun = |bb: &[f64]| -> Vec<f64> {
                    let a = a_mats_from(spec, bb, *m);
                    let c0: Vec<f64> = (0..k).map(|i| if spec.cons { bb[i * m + m - 1] } else { 0.0 }).collect();
                    match path(&a, &c0, &vec![0.0; k], 0.0, &Matrix::zeros(k, 0), &[], &|t| hist(&tsd, &ycols, t), origin + 1, h) {
                        Ok(p) => p.into_iter().flatten().collect(),
                        Err(_) => vec![SYSMISS; k * h],
                    }
                };
                let j = jacobian(&fun, b);
                if j.data.iter().any(|x| is_missing(*x) || !x.is_finite()) {
                    continue;
                }
                let d = delta_var(&j, v);
                for s in 0..h {
                    for i in 0..k {
                        add[s][i] += d[s * k + i];
                    }
                }
                n += 1.0;
            }
            if n > 0.0 {
                for s in 0..h {
                    for i in 0..k {
                        var[s][i] += add[s][i] / n;
                    }
                }
            }
        }
    }
    let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let rows: Vec<usize> = (0..h).map(|s| tsd.row_at((t0 + s as i64) as f64).unwrap()).collect();
    for (i, n) in names.iter().enumerate() {
        let mut cols = vec![vec![SYSMISS; ds.n_obs]; 4];
        for (s, &r) in rows.iter().enumerate() {
            let se = var[s][i].sqrt();
            cols[0][r] = fc[s][i];
            cols[1][r] = fc[s][i] - z * se;
            cols[2][r] = fc[s][i] + z * se;
            cols[3][r] = se;
        }
        let mut v = util::new_double(n, cols[0].clone());
        v.label = format!("{} forecast", lev.endog[i]);
        ds.add_variable(v)?;
        let ch = ds.characteristics.entry(n.clone()).or_default();
        ch.insert("fcast_endog".into(), lev.endog[i].clone());
        if !nose {
            for (j, s) in ["_LB", "_UB", "_SE"].iter().enumerate() {
                let mut v = util::new_double(&format!("{n}{s}"), cols[j + 1].clone());
                v.label = format!("{}% {} bound of {n}", fmt_level(level), if j == 0 { "lower" } else if j == 1 { "upper" } else { "standard error" });
                if j == 2 {
                    v.label = format!("standard error of {n}");
                }
                ds.add_variable(v)?;
            }
            let ch = ds.characteristics.entry(n.clone()).or_default();
            ch.insert("fcast_lb".into(), format!("{n}_LB"));
            ch.insert("fcast_ub".into(), format!("{n}_UB"));
            ch.insert("fcast_se".into(), format!("{n}_SE"));
        }
        ds.characteristics.entry(n.clone()).or_default().insert("fcast_level".into(), fmt_level(level));
    }
    st.data_changed = true;
    let _: HashMap<i64, usize> = HashMap::new();
    let _ = lev.is_vec;
    Ok(String::new())
}

fn fmt_level(l: f64) -> String {
    crate::format::fmt_g(l, 4)
}

/// Resolve a `fcast graph` item: a forecast variable, or a dependent
/// variable whose forecasts were computed.
fn forecast_var(ds: &DataSet, tok: &str) -> Result<String, String> {
    if let Ok(n) = crate::varlist::resolve_name(ds, tok) {
        if ds.characteristics.get(&n).is_some_and(|c| c.contains_key("fcast_endog")) {
            return Ok(n);
        }
    }
    let want = tsvar::canon(ds, tok).ok().and_then(|v| v.first().cloned()).unwrap_or_else(|| tok.to_string());
    ds.vars
        .iter()
        .rev()
        .find(|v| ds.characteristics.get(&v.name).and_then(|c| c.get("fcast_endog")).is_some_and(|e| e.eq_ignore_ascii_case(&want)))
        .map(|v| v.name.clone())
        .ok_or_else(|| rc_err(111, format!("{tok} is not a forecast variable created by fcast compute")))
}

pub fn graph(ds: &mut DataSet, st: &mut Store, cmd: &Command, list: &str) -> Result<String, String> {
    let tsd = ts(ds, st)?;
    let toks: Vec<String> = list.split_whitespace().map(String::from).collect();
    if toks.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let observed = cmd.has_opt("observed", 3);
    let noci = cmd.has_opt("noci", 4);
    let pal = crate::graph::palette("");
    let mut panels = Vec::new();
    for t in &toks {
        let n = forecast_var(ds, t)?;
        let ch = ds.characteristics.get(&n).cloned().unwrap_or_default();
        let endog = ch.get("fcast_endog").cloned().unwrap_or_default();
        let fcol = util::col(ds, &n)?;
        let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&r| !is_missing(fcol[r]) && !is_missing(tsd.time[r])).collect();
        rows.sort_by(|&a, &b| tsd.time[a].total_cmp(&tsd.time[b]));
        let mut g = Graph::default();
        let c0 = pal[0];
        let c1 = pal[1 % pal.len()];
        if !noci {
            if let (Some(lb), Some(ub)) = (ch.get("fcast_lb"), ch.get("fcast_ub")) {
                if let (Ok(l), Ok(u)) = (util::col(ds, lb), util::col(ds, ub)) {
                    let level = ch.get("fcast_level").cloned().unwrap_or_else(|| "95".into());
                    g.layers.push(Layer { kind: LayerKind::Range(RangeStyle::Area), ranges: rows.iter().map(|&r| (tsd.time[r], l[r], u[r])).collect(), color: Some((c0.0, c0.1, c0.2)), alpha: 0.3, label: format!("{level}% CI"), ..Default::default() });
                }
            }
        }
        g.layers.push(Layer { kind: LayerKind::Line, points: rows.iter().map(|&r| (tsd.time[r], fcol[r])).collect(), color: Some((c0.0, c0.1, c0.2)), label: "Forecast".into(), ..Default::default() });
        if observed {
            let y = series(ds, st, &endog)?;
            g.layers.push(Layer { kind: LayerKind::Line, points: rows.iter().filter(|&&r| !is_missing(y[r])).map(|&r| (tsd.time[r], y[r])).collect(), color: Some((c1.0, c1.1, c1.2)), label: "Observed".into(), ..Default::default() });
        }
        g.title = format!("Forecast for {endog}");
        g.xformat = tsd.tfmt.clone();
        g.xtitle = tsd.tvar.clone();
        g.legend = true;
        panels.push(g);
    }
    let mut g = if panels.len() == 1 {
        panels.pop().unwrap()
    } else {
        let mut root = Graph::default();
        root.cols = (panels.len() as f64).sqrt().ceil() as usize;
        root.children = panels;
        root
    };
    crate::graph::finish(&mut g, cmd, st)
}

pub struct Fcast;

impl CommandHandler for Fcast {
    fn name(&self) -> &'static str {
        "fcast"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = match main.split_once(char::is_whitespace) {
            Some((a, b)) => (a.to_lowercase(), b.trim().to_string()),
            None => (main.to_lowercase(), String::new()),
        };
        let old = cmd.verb == "varfcast";
        match sub.as_str() {
            "compute" | "c" | "co" | "com" | "comp" | "compu" | "comput" => {
                if old {
                    let p = cmd.opt("prefix", 3).map(|s| s.trim().to_string());
                    match p {
                        Some(p) => compute(ds, st, cmd, &p, false),
                        None => compute(ds, st, cmd, "_f", true),
                    }
                } else {
                    let prefix = rest.trim();
                    if prefix.is_empty() {
                        return Err(rc_err(198, "prefix required"));
                    }
                    compute(ds, st, cmd, prefix, false)
                }
            }
            "graph" | "gr" | "gra" | "grap" => graph(ds, st, cmd, &rest),
            "clear" if old => Ok(String::new()),
            other => Err(rc_err(199, format!("{} {other} is unrecognized", cmd.verb))),
        }
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("fcast", Box::new(Fcast)), ("varfcast", Box::new(Fcast))]
}
