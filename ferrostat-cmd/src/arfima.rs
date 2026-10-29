//! `arfima`: autoregressive fractionally integrated moving-average models
//! by exact maximum likelihood (autocovariances of the ARFIMA process and
//! the Durbin-Levinson recursion); `predict` (xb, residuals, rstandard)
//! after arfima; and `arimasoc` / `arfimasoc` lag-order selection tables.
//!
//!   phi(L) (1 - L)^d (y_t - x_t b) = theta(L) e_t,   Var(e) = sigma2

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::{Results, Store};
use crate::ts_ml::{self, parse_lags, Post, LN2PI};
use crate::tsops;
use crate::util;

/// Autocovariances (unit innovation variance) of ARFIMA with AR a, MA m and d, lags 0..n-1.
pub fn arfima_acov(a: &[f64], m: &[f64], d: f64, n: usize) -> Option<Vec<f64>> {
    if !(d > -0.5 && d < 0.5) {
        return None;
    }
    // psi weights of the ARMA part
    let jmax = 4000usize;
    let mut psi = vec![0.0; jmax];
    psi[0] = 1.0;
    for j in 1..jmax {
        let mut v = if j <= m.len() { m[j - 1] } else { 0.0 };
        for (i, ai) in a.iter().enumerate() {
            if j > i {
                v += ai * psi[j - i - 1];
            }
        }
        psi[j] = v;
        if !v.is_finite() || v.abs() > 1e8 {
            return None;
        }
    }
    let tail: f64 = psi[jmax - 50..].iter().map(|v| v.abs()).sum();
    if tail > 1e-6 {
        return None; // not stationary
    }
    // effective length of psi
    let mut len_psi = jmax;
    while len_psi > a.len().max(m.len()) + 2 && psi[len_psi - 1].abs() < 1e-13 {
        len_psi -= 1;
    }
    let psi = &psi[..len_psi];
    // r_j = sum_i psi_i psi_{i+j}
    let r: Vec<f64> = (0..len_psi).map(|j| (0..len_psi - j).map(|i| psi[i] * psi[i + j]).sum()).collect();
    let jr = r.len() - 1;
    if d == 0.0 {
        let mut g = vec![0.0; n];
        for h in 0..n.min(r.len()) {
            g[h] = r[h];
        }
        return Some(g);
    }
    // fractional-noise autocovariances up to n + jr
    let len = n + jr + 1;
    let mut gd = vec![0.0; len];
    gd[0] = (dist::lngamma(1.0 - 2.0 * d) - 2.0 * dist::lngamma(1.0 - d)).exp();
    for h in 1..len {
        gd[h] = gd[h - 1] * (h as f64 - 1.0 + d) / (h as f64 - d);
    }
    let mut g = vec![0.0; n];
    for h in 0..n {
        let mut s = r[0] * gd[h];
        for j in 1..=jr {
            s += r[j] * (gd[h + j] + gd[(h as i64 - j as i64).unsigned_abs() as usize]);
        }
        g[h] = s;
    }
    Some(g)
}

/// Durbin-Levinson: one-step prediction coefficients applied to u;
/// returns (prediction, variance factor) per observation.
pub fn durbin_levinson(g: &[f64], u: &[f64]) -> Option<Vec<(f64, f64)>> {
    let n = u.len();
    let mut out = Vec::with_capacity(n);
    let mut phi: Vec<f64> = Vec::new();
    let mut v = g[0];
    if !(v > 0.0) {
        return None;
    }
    out.push((0.0, v));
    for t in 1..n {
        // update phi_{t,1..t}
        let num = g[t] - (0..phi.len()).map(|j| phi[j] * g[t - 1 - j]).sum::<f64>();
        let k = num / v;
        let mut np = vec![0.0; t];
        for j in 0..t - 1 {
            np[j] = phi[j] - k * phi[t - 2 - j];
        }
        np[t - 1] = k;
        phi = np;
        v *= 1.0 - k * k;
        if !(v > 0.0) {
            return None;
        }
        let pred: f64 = (0..t).map(|j| phi[j] * u[t - 1 - j]).sum();
        out.push((pred, v));
    }
    Some(out)
}

struct Lay {
    kx: usize,
    ar: Vec<usize>,
    ma: Vec<usize>,
    fixd: Option<f64>,
}

impl Lay {
    fn npar(&self) -> usize {
        self.kx + self.ar.len() + self.ma.len() + usize::from(self.fixd.is_none()) + 1
    }
    fn polys(&self, th: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        let mut a = vec![0.0; self.ar.iter().max().copied().unwrap_or(0)];
        let mut m = vec![0.0; self.ma.iter().max().copied().unwrap_or(0)];
        let mut i = self.kx;
        for &l in &self.ar {
            a[l - 1] = th[i];
            i += 1;
        }
        for &l in &self.ma {
            m[l - 1] = th[i];
            i += 1;
        }
        let d = self.fixd.unwrap_or_else(|| th[i]);
        (a, m, d)
    }
}

fn ll_obs(lay: &Lay, th: &[f64], y: &[f64], x: &[Vec<f64>]) -> Option<Vec<f64>> {
    let s2 = th[lay.npar() - 1];
    if !(s2 > 0.0) {
        return None;
    }
    let (a, m, d) = lay.polys(th);
    let n = y.len();
    let g = arfima_acov(&a, &m, d, n)?;
    let u: Vec<f64> = (0..n).map(|t| y[t] - ts_ml::dot(&x[t], &th[..lay.kx])).collect();
    let dl = durbin_levinson(&g, &u)?;
    Some((0..n).map(|t| {
        let var = s2 * dl[t].1;
        -0.5 * (LN2PI + var.ln() + (u[t] - dl[t].0).powi(2) / var)
    }).collect())
}

pub struct Arfima;

impl CommandHandler for Arfima {
    fn name(&self) -> &'static str {
        "arfima"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "arfima");
        }
        let samp = ts_ml::ts_sample(ds, st, cmd, &cmd.main, cmd.has_opt("noconstant", 4), vec![])?;
        if samp.n_gaps > 0 {
            return Err(rc_err(498, "gaps not allowed"));
        }
        let ar = cmd.opt("ar", 2).map_or(Ok(vec![]), parse_lags)?;
        let ma = cmd.opt("ma", 2).map_or(Ok(vec![]), parse_lags)?;
        let fixd = if cmd.has_opt("smemory", 2) { Some(0.0) } else { None };
        let lay = Lay { kx: samp.kx(), ar: ar.clone(), ma: ma.clone(), fixd };
        let n = samp.n();
        if n <= lay.npar() + 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let llf = |th: &[f64]| ll_obs(&lay, th, &samp.y, &samp.x);
        let beta = ts_ml::lsq(&samp.x, &samp.y);
        let resid: Vec<f64> = (0..n).map(|t| samp.y[t] - samp.xb(t, &beta)).collect();
        let v0 = resid.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let mk = |c: f64, dd: f64| -> Vec<f64> {
            let mut th = beta.clone();
            th.extend(ar.iter().map(|_| c / ar.len() as f64));
            th.extend(ma.iter().map(|_| 0.0));
            if fixd.is_none() {
                th.push(dd);
            }
            th.push(v0 * 0.7);
            th
        };
        let mut th0 = mk(0.1, 0.1);
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (c, dd) in [(0.0, 0.2), (0.3, 0.1), (0.5, 0.0), (0.1, 0.35), (-0.2, 0.3), (0.0, -0.2)] {
            let cand = mk(c, dd);
            let l = ts_ml::ll_sum(&llf, &cand);
            if l > best {
                best = l;
                th0 = cand;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, lay.npar())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let opt = ts_ml::optimize(&llf, th0, maxiter, ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs), "log likelihood", true)?;
        let th = opt.th.clone();
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "oim")?;
        let eq = samp.depvar.rsplit('.').next().unwrap_or(&samp.depvar).to_string();
        let lag = |l: usize, w: &str| if l == 1 { format!("L.{w}") } else { format!("L{l}.{w}") };
        let mut names: Vec<String> = samp.xnames.iter().map(|n| format!("{eq}:{n}")).collect();
        names.extend(ar.iter().map(|&l| format!("ARFIMA:{}", lag(l, "ar"))));
        names.extend(ma.iter().map(|&l| format!("ARFIMA:{}", lag(l, "ma"))));
        if fixd.is_none() {
            names.push("ARFIMA:d".into());
        }
        names.push("/sigma2".into());
        let k = th.len();
        let test_idx: Vec<usize> = (0..k - 1).filter(|&j| !(j < lay.kx && samp.xnames[j] == "_cons")).collect();
        let (chi2, df) = ts_ml::wald(&th, &v, &test_idx);
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let header = ts_ml::std_header(&sline, n, chi2, df, opt.ll, "Log likelihood");
        let notes = vec!["Note: The test of the variance against zero is one sided, and the two-sided".to_string(), "      confidence interval is truncated at zero.".to_string()];
        let text = ts_ml::post(
            st,
            cmd,
            Post { title: "ARFIMA regression", cmd: "arfima", depvar: &samp.depvar, names, b: th.clone(), v, ll: opt.ll, n, header, vcetype, vce: vce_name, log: opt.log, notes, converged: opt.converged, iters: opt.iters, df: None },
            Some(&samp.mask),
        )?;
        let e = &mut st.e;
        e.set("k1", lay.kx as f64);
        e.set("df_m", df);
        e.set("chi2", chi2);
        e.set("p", if df > 0.0 { dist::chi2tail(df, chi2) } else { SYSMISS });
        e.set("tmin", samp.tmin());
        e.set("tmax", samp.tmax());
        e.set("s2", th[k - 1]);
        e.set("ar_max", ar.iter().max().copied().unwrap_or(0) as f64);
        e.set("ma_max", ma.iter().max().copied().unwrap_or(0) as f64);
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("chi2type", "Wald");
        e.set_str("method", "mle");
        e.set_str("ar", ar.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
        e.set_str("ma", ma.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
        e.set_str("covariates", samp.xnames.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
        e.set_str("predict", "arfima_p");
        e.set_str("_fixd", fixd.map(|v| v.to_string()).unwrap_or_default());
        Ok(text)
    }
}

/// `predict` after arfima: xb (one-step predictions, default), residuals, rstandard.
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let kx = e.get("k1").unwrap_or(0.0) as usize;
    let lags = |k: &str| e.macros.get(k).map(|s| s.split_whitespace().filter_map(|v| v.parse().ok()).collect()).unwrap_or_default();
    let lay = Lay { kx, ar: lags("ar"), ma: lags("ma"), fixd: e.macros.get("_fixd").and_then(|s| s.parse().ok()) };
    let th = b.data.clone();
    let xnames: Vec<String> = b.colnames[..kx].iter().map(|c| c.split_once(':').map_or(c.clone(), |x| x.1.to_string())).collect();
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let newvar = ts_ml::predict_name(cmd)?;
    let yv = tsops::values(ds, st, &dep)?;
    let xv: Vec<Vec<f64>> = xnames.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?;
    let mask = util::touse(ds, st, cmd)?;
    let tmin = e.get("tmin").unwrap_or(f64::NEG_INFINITY);
    let tmax = e.get("tmax").unwrap_or(f64::INFINITY);
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(tc[i]) && tc[i] >= tmin && tc[i] <= tmax && !is_missing(yv[i]) && xv.iter().all(|c| !is_missing(c[i]))).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    let (a, m, d) = lay.polys(&th);
    let g = arfima_acov(&a, &m, d, rows.len()).ok_or_else(|| rc_err(498, "model not stationary"))?;
    let xb: Vec<f64> = rows.iter().map(|&i| xv.iter().zip(&th).map(|(c, b)| c[i] * b).sum()).collect();
    let u: Vec<f64> = rows.iter().zip(&xb).map(|(&i, x)| yv[i] - x).collect();
    let dl = durbin_levinson(&g, &u).ok_or_else(|| rc_err(498, "prediction failed"))?;
    let s2 = th[th.len() - 1];
    let stat = if cmd.has_opt("rstandard", 2) { "rstandard" } else if cmd.has_opt("residuals", 1) { "residuals" } else { "xb" };
    let mut out = vec![SYSMISS; ds.n_obs];
    for (t, &i) in rows.iter().enumerate() {
        out[i] = match stat {
            "residuals" => u[t] - dl[t].0,
            "rstandard" => (u[t] - dl[t].0) / (s2 * dl[t].1).sqrt(),
            _ => xb[t] + dl[t].0,
        };
    }
    ts_ml::add_prediction(ds, &newvar, out, &mask, if stat == "xb" && !cmd.has_opt("xb", 2) { "(option xb assumed)" } else { "" })
}

// ---------------------------------------------------------------------------
// arimasoc / arfimasoc
// ---------------------------------------------------------------------------

pub struct Soc {
    pub fractional: bool,
}

impl CommandHandler for Soc {
    fn name(&self) -> &'static str {
        if self.fractional { "arfimasoc" } else { "arimasoc" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let maxar = util::opt_num(cmd, "maxar", 4)?.unwrap_or(2.0) as usize;
        let maxma = util::opt_num(cmd, "maxma", 4)?.unwrap_or(2.0) as usize;
        let verb = if self.fractional { "arfima" } else { "arima" };
        let mut extra = String::new();
        for (n, v) in &cmd.opts {
            let nl = n.to_lowercase();
            if nl == "noconstant" || nl == "nocons" || nl == "diffuse" || nl == "condition" {
                extra.push_str(&format!(" {nl}"));
            } else if nl == "arima" {
                if let Some(v) = v {
                    extra.push_str(&format!(" arima({v})"));
                }
            }
        }
        let iff = cmd.if_raw.as_ref().map(|s| format!(" if {s}")).unwrap_or_default();
        let saved_e = st.e.clone();
        let saved_r = st.r.clone();
        let mut rows = Vec::new();
        let mut nobs = 0.0;
        let handler: Box<dyn CommandHandler> = if self.fractional { Box::new(Arfima) } else { Box::new(crate::arima::Arima) };
        for p in 0..=maxar {
            for q in 0..=maxma {
                let mut opts = Vec::new();
                if p > 0 {
                    opts.push(format!("ar(1/{p})"));
                }
                if q > 0 {
                    opts.push(format!("ma(1/{q})"));
                }
                let line = format!("{verb} {}{iff}, {}{extra}", cmd.main.trim(), opts.join(" "));
                let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
                match handler.execute(ds, st, &c) {
                    Ok(_) => {
                        let ll = st.e.get("ll").unwrap_or(SYSMISS);
                        let k = st.e.get("rank").or(st.e.get("k")).unwrap_or(0.0);
                        nobs = st.e.get("N").unwrap_or(0.0);
                        rows.push((p, q, ll, k));
                    }
                    Err(_) => rows.push((p, q, SYSMISS, SYSMISS)),
                }
            }
        }
        st.e = saved_e;
        st.r = saved_r;
        let mut out = vec![String::new(), "Lag-order selection criteria".to_string(), String::new(), format!("{:<48}Number of obs = {}", "", fmt_g(nobs, 9))];
        out.push("  +---------------------------------------------------------------+".into());
        out.push("  |  Model  |        LL     df          AIC          BIC          HQIC |".into());
        out.push("  |---------+-----------------------------------------------------|".into());
        let ic = |ll: f64, k: f64| -> (f64, f64, f64) { (-2.0 * ll + 2.0 * k, -2.0 * ll + k * nobs.ln(), -2.0 * ll + 2.0 * k * nobs.ln().ln()) };
        let best = |f: &dyn Fn(f64, f64) -> f64| rows.iter().enumerate().filter(|r| r.1 .2 < SYSMISS).min_by(|a, b| f(a.1 .2, a.1 .3).total_cmp(&f(b.1 .2, b.1 .3))).map(|r| r.0);
        let ba = best(&|l, k| ic(l, k).0);
        let bb = best(&|l, k| ic(l, k).1);
        let bh = best(&|l, k| ic(l, k).2);
        let mut m = crate::matrix::Matrix::zeros(rows.len(), 7);
        for (i, &(p, q, ll, k)) in rows.iter().enumerate() {
            let (a, b, h) = if ll < SYSMISS { ic(ll, k) } else { (SYSMISS, SYSMISS, SYSMISS) };
            let star = |x: Option<usize>| if x == Some(i) { "*" } else { " " };
            out.push(format!("  | ({p},{q})   | {:>10} {:>6} {:>11}{} {:>11}{} {:>11}{} |", fmt_g(ll, 9), fmt_g(k, 6), fmt_g(a, 9), star(ba), fmt_g(b, 9), star(bb), fmt_g(h, 9), star(bh)));
            m.data[i * 7..i * 7 + 7].copy_from_slice(&[p as f64, q as f64, ll, k, a, b, h]);
        }
        out.push("  +---------------------------------------------------------------+".into());
        out.push("  * optimal model".into());
        let mut r = Results::new();
        r.set("N", nobs);
        r.set_mat("table", m.with_names((0..rows.len()).map(|i| format!("r{}", i + 1)).collect(), ["p", "q", "LL", "df", "AIC", "BIC", "HQIC"].iter().map(|s| s.to_string()).collect()));
        st.r = r;
        Ok(out.join("\n"))
    }
}
