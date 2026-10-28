//! Unit-root and white-noise tests: `dfuller` (augmented Dickey–Fuller),
//! `pperron` (Phillips & Perron 1988, *Biometrika* 75: 335-346, in the form
//! of Hamilton, *Time Series Analysis*, 1994, sec. 17.6), `dfgls` (Elliott,
//! Rothenberg & Stock 1996, *Econometrica* 64: 813-836), `corrgram` and
//! `wntestq` (Ljung & Box 1978, *Biometrika* 65: 297-303).
//!
//! Critical values are interpolated from Fuller, *Introduction to
//! Statistical Time Series*, 1976, table 8.5.2; approximate p-values use
//! the response surfaces of MacKinnon (1994), *JBES* 12: 167-176.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsops;
use crate::tsplot::{autocorrelations, partial_autocorrelations};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Det {
    None,
    Const,
    Trend,
}

/// Fuller (1976) table 8.5.2: rows n = 25, 50, 100, 250, 500, infinity;
/// columns 1%, 5%, 10%.
const TAU: [[[f64; 3]; 6]; 3] = [
    [[-2.66, -1.95, -1.60], [-2.62, -1.95, -1.61], [-2.60, -1.95, -1.61], [-2.58, -1.95, -1.62], [-2.58, -1.95, -1.62], [-2.58, -1.95, -1.62]],
    [[-3.75, -3.00, -2.63], [-3.58, -2.93, -2.60], [-3.51, -2.89, -2.58], [-3.46, -2.88, -2.57], [-3.44, -2.87, -2.57], [-3.43, -2.86, -2.57]],
    [[-4.38, -3.60, -3.24], [-4.15, -3.50, -3.18], [-4.04, -3.45, -3.15], [-3.99, -3.43, -3.13], [-3.98, -3.42, -3.13], [-3.96, -3.41, -3.12]],
];
const RHO: [[[f64; 3]; 6]; 3] = [
    [[-11.9, -7.3, -5.3], [-12.9, -7.7, -5.5], [-13.3, -7.9, -5.6], [-13.6, -8.0, -5.7], [-13.7, -8.0, -5.7], [-13.8, -8.1, -5.7]],
    [[-17.2, -12.5, -10.2], [-18.9, -13.3, -10.7], [-19.8, -13.7, -11.0], [-20.3, -14.0, -11.2], [-20.5, -14.0, -11.2], [-20.7, -14.1, -11.3]],
    [[-22.5, -17.9, -15.6], [-25.7, -19.8, -16.8], [-27.4, -20.7, -17.5], [-28.4, -21.3, -18.0], [-28.9, -21.5, -18.1], [-29.5, -21.8, -18.3]],
];

/// Critical values interpolated linearly in the sample size.
pub fn critical_values(det: Det, n: f64, rho: bool) -> [f64; 3] {
    let t = if rho { &RHO } else { &TAU };
    let tab = &t[det as usize];
    let ns = [25.0, 50.0, 100.0, 250.0, 500.0];
    if n <= 25.0 {
        return tab[0];
    }
    if n > 500.0 {
        return tab[5];
    }
    let i = ns.iter().position(|&x| n <= x).unwrap();
    let w = (n - ns[i - 1]) / (ns[i] - ns[i - 1]);
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = tab[i - 1][c] + w * (tab[i][c] - tab[i - 1][c]);
    }
    out
}

/// MacKinnon (1994) approximate p-value of the Dickey–Fuller t statistic.
pub fn mackinnon_p(z: f64, det: Det) -> f64 {
    let (tmax, tmin, tstar, small, large): (f64, f64, f64, [f64; 3], [f64; 4]) = match det {
        Det::None => (f64::INFINITY, -19.04, -1.04, [0.6344, 1.2378, 3.2496], [0.4797, 9.3557, -0.6999, 3.3066]),
        Det::Const => (2.74, -18.83, -1.61, [2.1659, 1.4412, 3.8269], [1.7339, 9.3202, -1.2745, -1.0368]),
        Det::Trend => (0.7, -16.18, -2.89, [3.2512, 1.6047, 4.9588], [2.5261, 6.1654, -3.7956, -6.0285]),
    };
    if z > tmax {
        return 1.0;
    }
    if z < tmin {
        return 0.0;
    }
    let x = if z <= tstar {
        small[0] + small[1] * z + small[2] * 1e-2 * z * z
    } else {
        large[0] + large[1] * 1e-1 * z + large[2] * 1e-1 * z * z + large[3] * 1e-2 * z * z * z
    };
    dist::normal(x)
}

/// OLS with conventional standard errors: (b, V, residuals).
fn ols(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Matrix, Vec<f64>) {
    let k = x[0].len();
    let n = y.len();
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for (r, yv) in x.iter().zip(y) {
        for a in 0..k {
            xy[a] += r[a] * yv;
            for c in 0..k {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    let (inv, _) = m.sym_inverse(1e-14);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xy[c]).sum()).collect();
    let e: Vec<f64> = x.iter().zip(y).map(|(r, yv)| yv - r.iter().zip(&b).map(|(p, q)| p * q).sum::<f64>()).collect();
    let s2 = e.iter().map(|v| v * v).sum::<f64>() / (n - k) as f64;
    (b, inv.scale(s2), e)
}

/// Operator prefix and variable of `OP.var`.
fn split_var(ds: &DataSet, tok: &str) -> Result<(String, String), String> {
    let r = tsops::resolve(ds, tok)?;
    Ok(match r.rsplit_once('.') {
        Some((o, v)) => (o.to_string(), v.to_string()),
        None => (String::new(), r),
    })
}

/// Column `op1` applied to `op0.var`, e.g. ("L", "D", "x") -> values of LD.x.
fn col(ds: &DataSet, st: &Store, op: &str, base_op: &str, var: &str) -> Result<Vec<f64>, String> {
    let o = format!("{op}{base_op}");
    tsops::values(ds, st, &if o.is_empty() { var.to_string() } else { format!("{o}.{var}") })
}

fn name(op: &str, base_op: &str, var: &str) -> String {
    let o = format!("{op}{base_op}");
    if o.is_empty() { var.to_string() } else { format!("{o}.{var}") }
}

struct UnitRootReg {
    names: Vec<String>,
    b: Vec<f64>,
    v: Matrix,
    resid: Vec<f64>,
    n: usize,
    depname: String,
    sample: Vec<bool>,
}

/// Regression of `dep_op.var` on `regs` (op strings), optional trend and constant.
fn unit_root_reg(ds: &DataSet, st: &Store, cmd: &Command, var_tok: &str, dep_op: &str, regs: &[String], det: Det) -> Result<UnitRootReg, String> {
    let tvar = tsops::time_var(st)?;
    if st.tsset.as_ref().is_some_and(|(p, _)| p.is_some()) {
        return Err(rc_err(459, format!("{} may not be used with panel data", cmd.verb)));
    }
    let (bop, var) = split_var(ds, var_tok)?;
    let y = col(ds, st, dep_op, &bop, &var)?;
    let cols: Vec<Vec<f64>> = regs.iter().map(|o| col(ds, st, o, &bop, &var)).collect::<Result<_, _>>()?;
    let tc = util::col(ds, &tvar)?;
    let mask = util::touse(ds, st, cmd)?;
    let mut refs: Vec<&Vec<f64>> = cols.iter().collect();
    refs.push(&y);
    let (rows, times) = tsops::ordered_rows(ds, st, &mask, &refs)?;
    let mut names: Vec<String> = regs.iter().map(|o| name(o, &bop, &var)).collect();
    if det == Det::Trend {
        names.push("_trend".into());
    }
    if det != Det::None {
        names.push("_cons".into());
    }
    let k = names.len();
    if rows.len() <= k + 1 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let t0 = times[0];
    let x: Vec<Vec<f64>> = rows
        .iter()
        .zip(&times)
        .map(|(&i, &t)| {
            let mut r: Vec<f64> = cols.iter().map(|c| c[i]).collect();
            if det == Det::Trend {
                r.push(t - t0 + 1.0);
            }
            if det != Det::None {
                r.push(1.0);
            }
            r
        })
        .collect();
    let _ = tc;
    let yy: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let (b, v, resid) = ols(&x, &yy);
    let mut sample = vec![false; ds.n_obs];
    for &i in &rows {
        sample[i] = true;
    }
    Ok(UnitRootReg { names, b, v, resid, n: rows.len(), depname: name(dep_op, &bop, &var), sample })
}

fn det_of(cmd: &Command) -> Result<Det, String> {
    let nc = cmd.has_opt("noconstant", 3);
    let tr = cmd.has_opt("trend", 3);
    if nc && tr {
        return Err(rc_err(198, "noconstant and trend may not be specified together"));
    }
    Ok(if nc { Det::None } else if tr { Det::Trend } else { Det::Const })
}

fn h0_line(det: Det, drift: bool) -> &'static str {
    match (det, drift) {
        (Det::Trend, _) => "H0: Random walk with or without drift",
        (_, true) => "H0: Random walk with drift",
        _ => "H0: Random walk without drift, d = 0",
    }
}

fn stat_table(rows: &[(&str, f64, [f64; 3])], dist_label: &str) -> Vec<String> {
    let mut out = vec![
        format!("{:>39}{}", "", dist_label),
        "                   Test      -------- critical value ---------".to_string(),
        "              statistic           1%           5%          10%".to_string(),
        "-".repeat(62),
    ];
    for (lab, z, cv) in rows {
        let w = 22 - lab.chars().count();
        out.push(format!(" {lab}{:>w$.3}{:>13.3}{:>13.3}{:>13.3}", z, cv[0], cv[1], cv[2]));
    }
    out.push("-".repeat(62));
    out
}

fn var_lines(var: &str, n: usize, lab2: &str, v2: usize) -> Vec<String> {
    let w = n.to_string().len().max(v2.to_string().len());
    let l1 = "Number of obs";
    let lw = l1.chars().count().max(lab2.chars().count());
    vec![format!("{:<43}{:<lw$} = {:>w$}", format!("Variable: {var}"), l1, n), format!("{:<43}{:<lw$} = {:>w$}", "", lab2, v2)]
}

fn reg_table(r: &UnitRootReg, level: f64) -> Vec<String> {
    let k = r.names.len();
    let zeros = vec![false; k];
    est::coef_table(&est::TableSpec {
        depvar: &r.depname,
        names: &r.names,
        b: &r.b,
        v: &r.v,
        omitted: &zeros,
        is_base: &zeros,
        df: Some((r.n - k) as f64),
        level,
        eform: None,
        coef_title: "Coefficient",
    })
}

// ---------------------------------------------------------------------------
// dfuller
// ---------------------------------------------------------------------------

pub struct Dfuller;

impl CommandHandler for Dfuller {
    fn name(&self) -> &'static str {
        "dfuller"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let toks: Vec<&str> = cmd.main.split_whitespace().collect();
        if toks.len() != 1 {
            return Err(rc_err(if toks.is_empty() { 100 } else { 103 }, if toks.is_empty() { "varname required" } else { "too many variables specified" }));
        }
        let det = det_of(cmd)?;
        let drift = cmd.has_opt("drift", 3);
        if drift && det != Det::Const {
            return Err(rc_err(198, "drift may not be specified with noconstant or trend"));
        }
        let lags = util::opt_num(cmd, "lags", 1)?.unwrap_or(0.0).max(0.0) as usize;
        let mut regs = vec!["L".to_string()];
        regs.extend((1..=lags).map(|l| if l == 1 { "LD".to_string() } else { format!("L{l}D") }));
        let r = unit_root_reg(ds, st, cmd, toks[0], "D", &regs, det)?;
        let k = r.names.len();
        let zt = r.b[0] / r.v[(0, 0)].sqrt();
        let n = r.n as f64;
        let df = (r.n - k) as f64;
        let (cv, p) = if drift {
            ([-dist::invttail(df, 0.01), -dist::invttail(df, 0.05), -dist::invttail(df, 0.10)], 1.0 - dist::ttail(df, zt))
        } else {
            (critical_values(det, n, false), mackinnon_p(zt, det))
        };
        let (_, var) = split_var(ds, toks[0])?;
        let mut out = vec![String::new(), format!("{} test for unit root", if lags > 0 { "Augmented Dickey–Fuller" } else { "Dickey–Fuller" }), String::new()];
        out.extend(var_lines(&tsops::resolve(ds, toks[0])?, r.n, "Number of lags", lags));
        let _ = var;
        out.push(String::new());
        out.push(h0_line(det, drift).into());
        out.push(String::new());
        out.extend(stat_table(&[("Z(t)", zt, cv)], if drift { " Student's t" } else { "Dickey–Fuller" }));
        if drift {
            out.push(format!("p-value for Z(t) = {:.4}.", p));
        } else {
            out.push(format!("MacKinnon approximate p-value for Z(t) = {:.4}.", p));
        }
        if cmd.has_opt("regress", 3) {
            out.push(String::new());
            out.push("Regression table".into());
            out.extend(reg_table(&r, util::level(cmd, st)?));
        }
        let rr = st.begin_r();
        rr.set("N", n);
        rr.set("lags", lags as f64);
        rr.set("Zt", zt);
        rr.set("p", p);
        rr.set("cv_1", cv[0]);
        rr.set("cv_5", cv[1]);
        rr.set("cv_10", cv[2]);
        let _ = r.sample;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// pperron
// ---------------------------------------------------------------------------

pub struct Pperron;

impl CommandHandler for Pperron {
    fn name(&self) -> &'static str {
        "pperron"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let toks: Vec<&str> = cmd.main.split_whitespace().collect();
        if toks.len() != 1 {
            return Err(rc_err(if toks.is_empty() { 100 } else { 103 }, "one variable required"));
        }
        let det = det_of(cmd)?;
        let r = unit_root_reg(ds, st, cmd, toks[0], "", &["L".to_string()], det)?;
        let t = r.n as f64;
        let k = r.names.len();
        let lags = match util::opt_num(cmd, "lags", 1)? {
            Some(l) => l.max(0.0) as usize,
            None => (4.0 * (t / 100.0).powf(2.0 / 9.0)).floor() as usize,
        };
        let e = &r.resid;
        let gamma = |j: usize| -> f64 { (j..e.len()).map(|i| e[i] * e[i - j]).sum::<f64>() / t };
        let g0 = gamma(0);
        let lam2 = g0 + 2.0 * (1..=lags).map(|j| (1.0 - j as f64 / (lags as f64 + 1.0)) * gamma(j)).sum::<f64>();
        let s2 = e.iter().map(|v| v * v).sum::<f64>() / (t - k as f64);
        let rho = r.b[0];
        let se = r.v[(0, 0)].sqrt();
        let zrho = t * (rho - 1.0) - 0.5 * (t * t * se * se / s2) * (lam2 - g0);
        let zt = (g0 / lam2).sqrt() * (rho - 1.0) / se - 0.5 * (lam2 - g0) / lam2.sqrt() * (t * se / s2.sqrt());
        let p = mackinnon_p(zt, det);
        let cvr = critical_values(det, t, true);
        let cvt = critical_values(det, t, false);
        let mut out = vec![String::new(), "Phillips–Perron test for unit root".to_string(), String::new()];
        out.extend(var_lines(&tsops::resolve(ds, toks[0])?, r.n, "Newey–West lags", lags));
        out.push(String::new());
        out.push(h0_line(det, false).into());
        out.push(String::new());
        out.extend(stat_table(&[("Z(rho)", zrho, cvr), ("Z(t)", zt, cvt)], "Dickey–Fuller"));
        out.push(format!("MacKinnon approximate p-value for Z(t) = {:.4}.", p));
        if cmd.has_opt("regress", 3) {
            out.push(String::new());
            out.push("Regression table".into());
            out.extend(reg_table(&r, util::level(cmd, st)?));
        }
        let rr = st.begin_r();
        rr.set("N", t);
        rr.set("lags", lags as f64);
        rr.set("Zt", zt);
        rr.set("Zrho", zrho);
        rr.set("pval", p);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// dfgls
// ---------------------------------------------------------------------------

/// Critical values of the DF-GLS tau: Elliott, Rothenberg & Stock (1996,
/// table 1) for the trend case (interpolated in 1/T), Fuller's
/// no-constant table otherwise.
fn dfgls_cv(trend: bool, t: f64, _k: f64) -> [f64; 3] {
    if !trend {
        return critical_values(Det::None, t, false);
    }
    let tab: [(f64, [f64; 3]); 4] = [(50.0, [-3.77, -3.19, -2.89]), (100.0, [-3.58, -3.03, -2.74]), (200.0, [-3.46, -2.93, -2.64]), (f64::INFINITY, [-3.48, -2.89, -2.57])];
    if t <= 50.0 {
        return tab[0].1;
    }
    let i = tab.iter().position(|r| t <= r.0).unwrap();
    let (x0, x1, x) = (1.0 / tab[i - 1].0, 1.0 / tab[i].0, 1.0 / t);
    let w = (x - x0) / (x1 - x0);
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = tab[i - 1].1[c] + w * (tab[i].1[c] - tab[i - 1].1[c]);
    }
    out
}

pub struct Dfgls;

impl CommandHandler for Dfgls {
    fn name(&self) -> &'static str {
        "dfgls"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let toks: Vec<&str> = cmd.main.split_whitespace().collect();
        if toks.len() != 1 {
            return Err(rc_err(if toks.is_empty() { 100 } else { 103 }, "one variable required"));
        }
        let trend = !cmd.has_opt("notrend", 5);
        let tvar = tsops::time_var(st)?;
        let (bop, var) = split_var(ds, toks[0])?;
        let y = col(ds, st, "", &bop, &var)?;
        let mask = util::touse(ds, st, cmd)?;
        let (rows, times) = tsops::ordered_rows(ds, st, &mask, &[&y])?;
        let _ = tvar;
        if tsops::gaps(&times) > 0 {
            return Err(rc_err(498, "sample may not contain gaps"));
        }
        let tn = rows.len();
        let t = tn as f64;
        let maxlag = match util::opt_num(cmd, "maxlag", 3)? {
            Some(l) => l as usize,
            None => (12.0 * ((t + 1.0) / 100.0).powf(0.25)).floor() as usize,
        };
        if tn < maxlag + 10 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // GLS detrending with cbar = -13.5 (trend) or -7 (constant)
        let cbar = if trend { -13.5 } else { -7.0 };
        let a = 1.0 + cbar / t;
        let yv: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let zrow = |i: usize| -> Vec<f64> { if trend { vec![1.0, (i + 1) as f64] } else { vec![1.0] } };
        let mut qx = Vec::with_capacity(tn);
        let mut qy = Vec::with_capacity(tn);
        for i in 0..tn {
            if i == 0 {
                qx.push(zrow(0));
                qy.push(yv[0]);
            } else {
                let (z1, z0) = (zrow(i), zrow(i - 1));
                qx.push(z1.iter().zip(&z0).map(|(p, q)| p - a * q).collect());
                qy.push(yv[i] - a * yv[i - 1]);
            }
        }
        let (bg, _, _) = ols(&qx, &qy);
        let yd: Vec<f64> = (0..tn).map(|i| yv[i] - zrow(i).iter().zip(&bg).map(|(p, q)| p * q).sum::<f64>()).collect();
        // ADF regressions without deterministic terms on the detrended series
        let mut res = Vec::new();
        for k in (0..=maxlag).rev() {
            let start = maxlag + 1;
            let x: Vec<Vec<f64>> = (start..tn).map(|i| {
                let mut r = vec![yd[i - 1]];
                r.extend((1..=k).map(|j| yd[i - j] - yd[i - j - 1]));
                r
            }).collect();
            let yy: Vec<f64> = (start..tn).map(|i| yd[i] - yd[i - 1]).collect();
            let (b, v, e) = ols(&x, &yy);
            let tau = b[0] / v[(0, 0)].sqrt();
            let nn = yy.len() as f64;
            let rss = e.iter().map(|v| v * v).sum::<f64>();
            let sig2 = rss / nn;
            let sic = sig2.ln() + (k + 1) as f64 * nn.ln() / nn;
            // modified AIC of Ng & Perron (2001)
            let sumy: f64 = (start..tn).map(|i| yd[i - 1] * yd[i - 1]).sum();
            let tauk = b[0] * b[0] * sumy / sig2;
            let maic = sig2.ln() + 2.0 * (tauk + k as f64) / nn;
            // significance of the last lag (sequential t test)
            let tlast = if k > 0 { b[k] / v[(k, k)].sqrt() } else { 0.0 };
            res.push((k, tau, dfgls_cv(trend, nn, k as f64), rss / (nn - x[0].len() as f64), sic, maic, tlast, nn));
        }
        let opt_sc = res.iter().min_by(|a, b| a.4.total_cmp(&b.4)).unwrap();
        let opt_ma = res.iter().min_by(|a, b| a.5.total_cmp(&b.5)).unwrap();
        let opt_ng = res.iter().find(|r| r.0 == 0 || r.6.abs() > 1.645).unwrap();
        let g = |v: f64| crate::format::fmt_g(v, 8);
        let mut out = vec![String::new(), format!("{:<57}Number of obs = {:>5}", format!("DF-GLS for {}", tsops::resolve(ds, toks[0])?), res[0].7 as usize)];
        out.push(if cmd.opt("maxlag", 3).is_some() { format!("Maxlag = {maxlag} specified") } else { format!("Maxlag = {maxlag} chosen by Schwert criterion") });
        out.push(String::new());
        out.push(format!("{:>15}{}", "", if trend { "DF-GLS tau      1% Critical       5% Critical      10% Critical" } else { "DF-GLS mu       1% Critical       5% Critical      10% Critical" }));
        out.push("  [lags]     Test Statistic        Value             Value             Value".into());
        out.push("-".repeat(78));
        for r in &res {
            out.push(format!("{:>6}{:>17.3}{:>18.3}{:>18.3}{:>18.3}", r.0, r.1, r.2[0], r.2[1], r.2[2]));
        }
        out.push("-".repeat(78));
        out.push(format!("Opt Lag (Ng–Perron seq t) = {:>2} with RMSE {:>9}", opt_ng.0, g(opt_ng.3.sqrt())));
        out.push(format!("Min SC   = {:>9} at lag {:>2} with RMSE {:>9}", g(opt_sc.4), opt_sc.0, g(opt_sc.3.sqrt())));
        out.push(format!("Min MAIC = {:>9} at lag {:>2} with RMSE {:>9}", g(opt_ma.5), opt_ma.0, g(opt_ma.3.sqrt())));
        let mut m = Matrix::zeros(res.len(), 5);
        for (i, r) in res.iter().enumerate() {
            m[(i, 0)] = r.0 as f64;
            m[(i, 1)] = r.1;
            m[(i, 2)] = r.2[0];
            m[(i, 3)] = r.2[1];
            m[(i, 4)] = r.2[2];
        }
        let rr = st.begin_r();
        rr.set("N", res[0].7);
        rr.set("maxlag", maxlag as f64);
        rr.set("sclag", opt_sc.0 as f64);
        rr.set("maiclag", opt_ma.0 as f64);
        rr.set("optlag", opt_ng.0 as f64);
        rr.set_mat("results", m.with_names(vec![], vec!["k".into(), "DFGLS".into(), "cv1".into(), "cv5".into(), "cv10".into()]));
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// corrgram, wntestq
// ---------------------------------------------------------------------------

fn series(ds: &DataSet, st: &Store, cmd: &Command) -> Result<(String, Vec<f64>), String> {
    let toks: Vec<&str> = cmd.main.split_whitespace().collect();
    if toks.len() != 1 {
        return Err(rc_err(if toks.is_empty() { 100 } else { 103 }, "one variable required"));
    }
    let name = tsops::resolve(ds, toks[0])?;
    let x = tsops::values(ds, st, &name)?;
    let mask = util::touse(ds, st, cmd)?;
    let (rows, _) = tsops::ordered_rows(ds, st, &mask, &[&x])?;
    if rows.len() < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    Ok((name, rows.iter().map(|&i| x[i]).collect()))
}

fn ljung_box(r: &[f64], n: f64) -> Vec<f64> {
    let mut q = 0.0;
    r.iter().enumerate().map(|(j, rj)| {
        q += rj * rj / (n - (j + 1) as f64);
        n * (n + 2.0) * q
    }).collect()
}

fn default_lags(n: usize) -> usize {
    (n / 2).saturating_sub(2).clamp(1, 40)
}

fn bar(v: f64, left_neg: bool) -> String {
    // 19 cells, zero at position 9; one cell per 1/8
    let k = ((v.abs() * 8.0) as usize).min(8);
    let mut cells = [' '; 19];
    cells[9] = '|';
    if v.is_finite() {
        if v < 0.0 && left_neg {
            for c in 0..k {
                cells[8 - c] = '-';
            }
        } else if v > 0.0 {
            for c in 0..k {
                cells[10 + c] = '-';
            }
        }
    }
    cells.iter().collect()
}

pub struct Corrgram;

impl CommandHandler for Corrgram {
    fn name(&self) -> &'static str {
        "corrgram"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (_, x) = series(ds, st, cmd)?;
        let n = x.len();
        let lags = util::opt_num(cmd, "lags", 1)?.map_or(default_lags(n), |l| l as usize).clamp(1, n - 1);
        let ac = autocorrelations(&x, lags);
        let pac = partial_autocorrelations(&x, lags);
        let q = ljung_box(&ac, n as f64);
        let plot = !cmd.has_opt("noplot", 3);
        let mut out = vec![String::new()];
        if plot {
            out.push(format!("{:>42}-1       0       1 -1       0       1", ""));
            out.push(" LAG       AC       PAC      Q     Prob>Q  [Autocorrelation]  [Partial autocor]".into());
            out.push("-".repeat(79));
        } else {
            out.push(" LAG       AC       PAC      Q     Prob>Q".into());
            out.push("-".repeat(41));
        }
        let rr = st.begin_r();
        rr.set("lags", lags as f64);
        let mut mac = Matrix::zeros(1, lags);
        let mut mpac = Matrix::zeros(1, lags);
        let mut mq = Matrix::zeros(1, lags);
        for j in 0..lags {
            let p = dist::chi2tail((j + 1) as f64, q[j]);
            let pv = pac[j];
            let pac_s = if pv.is_finite() { format!("{:>9.4}", pv) } else { format!("{:>9}", ".") };
            let mut line = format!("{:<6}{:>9.4}{}{:>9.2}{:>8.4}", j + 1, ac[j], pac_s, q[j], p);
            if plot {
                line.push_str(&format!(" {}{}", bar(ac[j], true), bar(if pv.is_finite() { pv } else { 0.0 }, true)));
                line = line.trim_end().to_string();
            }
            out.push(line);
            rr.set(&format!("ac{}", j + 1), ac[j]);
            rr.set(&format!("pac{}", j + 1), if pv.is_finite() { pv } else { ferrostat_core::missing::SYSMISS });
            rr.set(&format!("q{}", j + 1), q[j]);
            mac[(0, j)] = ac[j];
            mpac[(0, j)] = if pv.is_finite() { pv } else { ferrostat_core::missing::SYSMISS };
            mq[(0, j)] = q[j];
        }
        rr.set_mat("AC", mac);
        rr.set_mat("PAC", mpac);
        rr.set_mat("Q", mq);
        Ok(out.join("\n"))
    }
}

pub struct Wntestq;

impl CommandHandler for Wntestq {
    fn name(&self) -> &'static str {
        "wntestq"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (_, x) = series(ds, st, cmd)?;
        let n = x.len();
        let lags = util::opt_num(cmd, "lags", 1)?.map_or(default_lags(n), |l| l as usize).clamp(1, n - 1);
        let ac = autocorrelations(&x, lags);
        let q = *ljung_box(&ac, n as f64).last().unwrap();
        let p = dist::chi2tail(lags as f64, q);
        let rr = st.begin_r();
        rr.set("stat", q);
        rr.set("df", lags as f64);
        rr.set("p", p);
        let _ = is_missing(q);
        let mut r2 = Results::new();
        let _ = &mut r2;
        Ok(format!(
            "\nPortmanteau test for white noise\n---------------------------------------\n Portmanteau (Q) statistic = {:>10.4}\n {:<26}= {:>10.4}",
            q,
            format!("Prob > chi2({lags})"),
            p
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuller_interpolation_and_mackinnon() {
        // 140 observations with trend: -4.027, -3.445, -3.145
        let cv = critical_values(Det::Trend, 140.0, false);
        assert!((cv[0] + 4.0267).abs() < 1e-3 && (cv[1] + 3.4447).abs() < 1e-3 && (cv[2] + 3.1447).abs() < 1e-3);
        // asymptotic 5% critical value with a constant has p close to .05
        assert!((mackinnon_p(-2.86, Det::Const) - 0.05).abs() < 0.003);
        assert!((mackinnon_p(-3.41, Det::Trend) - 0.05).abs() < 0.003);
        assert!((mackinnon_p(-1.95, Det::None) - 0.05).abs() < 0.003);
    }
}
