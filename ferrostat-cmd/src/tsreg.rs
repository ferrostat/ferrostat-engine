//! Time-series regressions: `newey` (OLS with Newey–West HAC standard
//! errors; Newey & West 1987, *Econometrica* 55: 703-708) and `prais`
//! (Prais–Winsten and Cochrane–Orcutt feasible GLS for AR(1) errors; see
//! Judge et al., *The Theory and Practice of Econometrics*, 1985, ch. 8, and
//! Davidson & MacKinnon, *Estimation and Inference in Econometrics*, 1993).

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::models::replay;
use crate::store::{Results, Store};
use crate::tsops;
use crate::util;

pub(crate) fn split_dep(ds: &DataSet, main: &str) -> Result<(String, String), String> {
    let main = main.trim();
    let (dep, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
    if dep.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    Ok((tsops::resolve(ds, dep)?, rest.trim().to_string()))
}

/// Put the variance-type caption above the "std. err." column of a
/// coefficient table (lowercasing the column title as the reference does).
pub(crate) fn with_vcetype(mut lines: Vec<String>, vcetype: &str) -> Vec<String> {
    if lines.len() < 2 || vcetype.is_empty() {
        return lines;
    }
    let lw = lines[0].chars().count().saturating_sub(66);
    let start = (lw as i64 + 16 + (9 - vcetype.chars().count() as i64).div_euclid(2)).max(lw as i64 + 2) as usize;
    let cap = format!("{:lw$} |{}{}", "", " ".repeat(start - lw - 2), vcetype);
    lines[1] = lines[1].replacen("Std. err.", "std. err.", 1);
    lines.insert(1, cap);
    lines
}

/// Header line: left text in 48 columns, `label = value` on the right.
pub(crate) fn hline(left: &str, label: &str, value: &str) -> String {
    if label.is_empty() {
        return left.to_string();
    }
    format!("{:<48}{:<18}= {:>10}", left, label, value)
}

pub(crate) fn f2(v: f64) -> String {
    if !v.is_finite() || v >= SYSMISS { ".".into() } else { format!("{:.2}", v) }
}
pub(crate) fn f4(v: f64) -> String {
    if !v.is_finite() || v >= SYSMISS { ".".into() } else { format!("{:.4}", v) }
}

/// Positions of the design rows in time order and their times.
fn time_order(ds: &DataSet, st: &Store, d: &Design) -> Result<(Vec<usize>, Vec<f64>), String> {
    let t = tsops::time_var(st)?;
    let tc = util::col(ds, &t)?;
    let mut pos: Vec<usize> = (0..d.n()).collect();
    pos.sort_by(|&a, &b| tc[d.rows[a]].total_cmp(&tc[d.rows[b]]));
    let times = pos.iter().map(|&p| tc[d.rows[p]]).collect();
    Ok((pos, times))
}

/// Active-column sub-matrix helpers.
fn xtx(rows: &[Vec<f64>]) -> Matrix {
    let k = rows.first().map_or(0, |r| r.len());
    let mut m = Matrix::zeros(k, k);
    for r in rows {
        for a in 0..k {
            for b in 0..k {
                m[(a, b)] += r[a] * r[b];
            }
        }
    }
    m
}

fn mat_vec(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

// ---------------------------------------------------------------------------
// newey
// ---------------------------------------------------------------------------

pub struct Newey;

impl CommandHandler for Newey {
    fn name(&self) -> &'static str {
        "newey"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "newey");
        }
        let lag = util::opt_num(cmd, "lag", 1)?.ok_or_else(|| rc_err(198, "option lag() required"))?;
        if lag < 0.0 || lag.fract() != 0.0 {
            return Err(rc_err(198, "lag() must be a nonnegative integer"));
        }
        let lag = lag as usize;
        let (dep, indep) = split_dep(ds, &cmd.main)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "a", extra: vec![] })?;
        let fit = est::ols(&d, &Vce::Ols, None)?;
        let act = d.active();
        let k = act.len();
        let n = d.n_eff();
        // time position of every observation (lags are in units of the time variable)
        let (pos, times) = if lag > 0 || st.tsset.is_some() { time_order(ds, st, &d)? } else { ((0..d.n()).collect(), (0..d.n()).map(|i| i as f64).collect()) };
        if lag > 0 && tsops::gaps(&times) > 0 && !cmd.has_opt("force", 5) {
            return Err(rc_err(498, format!("{} is not regularly spaced -- use the force option to override", tsops::time_var(st)?)));
        }
        let scores: Vec<Vec<f64>> = pos.iter().map(|&t| act.iter().map(|&j| d.cols[j][t] * d.w[t] * fit.resid[t]).collect()).collect();
        let mut omega = Matrix::zeros(k, k);
        for s in &scores {
            for a in 0..k {
                for b in 0..k {
                    omega[(a, b)] += s[a] * s[b];
                }
            }
        }
        if lag > 0 {
            let force = cmd.has_opt("force", 5);
            let index: std::collections::HashMap<u64, usize> = times.iter().enumerate().map(|(i, t)| (t.to_bits(), i)).collect();
            for l in 1..=lag {
                let wl = 1.0 - l as f64 / (lag as f64 + 1.0);
                for i in 0..scores.len() {
                    // with gaps and force, observations are paired by position
                    let j = if force { i.checked_sub(l) } else { index.get(&(times[i] - l as f64).to_bits()).copied() };
                    let Some(j) = j else { continue };
                    for a in 0..k {
                        for b in 0..k {
                            omega[(a, b)] += wl * (scores[i][a] * scores[j][b] + scores[j][a] * scores[i][b]);
                        }
                    }
                }
            }
        }
        let mut bread = Matrix::zeros(k, k);
        for (a, &ja) in act.iter().enumerate() {
            for (b, &jb) in act.iter().enumerate() {
                bread[(a, b)] = fit.xtx_inv[(ja, jb)];
            }
        }
        let vact = bread.mul(&omega)?.mul(&bread)?.scale(n / (n - k as f64));
        let v = est::expand_v(&vact, &act, d.k());
        let df_r = n - k as f64;
        let (w, q) = est::wald_all(&d, &fit.b, &v);
        let f = if q > 0.0 { w / q } else { SYSMISS };
        let pf = if q > 0.0 { dist::ftail(q, df_r, f) } else { SYSMISS };
        let level = util::level(cmd, st)?;
        let mut out = d.notes.clone();
        out.push(String::new());
        out.push(hline("Regression with Newey–West standard errors", "Number of obs", &fmt_g(n, 10)));
        out.push(hline(&format!("Maximum lag = {lag}"), &format!("F({:>3},{:>10})", q, df_r), &f2(f)));
        out.push(hline("", "Prob > F", &f4(pf)));
        out.push(String::new());
        let table = est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &d.names,
            b: &fit.b,
            v: &v,
            omitted: &d.omitted,
            is_base: &d.is_base,
            df: Some(df_r),
            level,
            eform: None,
            coef_title: "Coefficient",
        });
        out.extend(with_vcetype(table, "Newey–West"));
        let text = out.join("\n");
        est::post(st, "newey", &d, &fit.b, &v, &|e: &mut Results| {
            e.set("df_m", q);
            e.set("df_r", df_r);
            e.set("F", f);
            e.set("lag", lag as f64);
            e.set_str("title", "Regression with Newey–West standard errors");
            e.set_str("vcetype", "Newey–West");
            e.set_str("predict", "newey_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// prais
// ---------------------------------------------------------------------------

/// OLS of y on the columns of x (rows), no implicit constant:
/// (b, (X'X)^-1, residuals).
fn ols_rows(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Matrix, Vec<f64>) {
    let (inv, _) = xtx(x).sym_inverse(1e-14);
    let k = inv.rows;
    let mut xty = vec![0.0; k];
    for (r, yv) in x.iter().zip(y) {
        for a in 0..k {
            xty[a] += r[a] * yv;
        }
    }
    let b = mat_vec(&inv, &xty);
    let e = x.iter().zip(y).map(|(r, yv)| yv - r.iter().zip(&b).map(|(a, c)| a * c).sum::<f64>()).collect();
    (b, inv, e)
}

fn dw(e: &[f64], first: &[bool]) -> f64 {
    let num: f64 = (1..e.len()).filter(|&t| !first[t]).map(|t| (e[t] - e[t - 1]).powi(2)).sum();
    num / e.iter().map(|v| v * v).sum::<f64>()
}

#[derive(Clone, Copy, PartialEq)]
enum RhoType {
    Regress,
    Freg,
    Tscorr,
    Dw,
    Theil,
    Nagar,
}

fn estimate_rho(e: &[f64], first: &[bool], kind: RhoType, k: usize) -> f64 {
    let n = e.len() as f64;
    let pairs: Vec<(f64, f64)> = (1..e.len()).filter(|&t| !first[t]).map(|t| (e[t], e[t - 1])).collect();
    let cross: f64 = pairs.iter().map(|(a, b)| a * b).sum();
    match kind {
        RhoType::Regress => cross / pairs.iter().map(|(_, b)| b * b).sum::<f64>(),
        RhoType::Freg => cross / pairs.iter().map(|(a, _)| a * a).sum::<f64>(),
        RhoType::Tscorr => cross / e.iter().map(|v| v * v).sum::<f64>(),
        RhoType::Dw => 1.0 - dw(e, first) / 2.0,
        RhoType::Theil => cross / e.iter().map(|v| v * v).sum::<f64>() * (n - k as f64) / n,
        RhoType::Nagar => {
            let r = 1.0 - dw(e, first) / 2.0;
            (r * n * n + (k * k) as f64) / (n * n - (k * k) as f64)
        }
    }
}

pub struct Prais;

struct PraisFit {
    b: Vec<f64>,
    inv: Matrix,
    rss: f64,
    ystar: Vec<f64>,
    estar: Vec<f64>,
    estar_first: Vec<bool>,
    resid: Vec<f64>,
}

impl CommandHandler for Prais {
    fn name(&self) -> &'static str {
        "prais"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "prais");
        }
        let corc = cmd.has_opt("corc", 4);
        let twostep = cmd.has_opt("twostep", 3);
        let ssesearch = cmd.has_opt("ssesearch", 3);
        let kind = match cmd.opt("rhotype", 4).map(|s| s.trim().to_lowercase()) {
            None => RhoType::Regress,
            Some(s) => match s.as_str() {
                "regress" | "reg" => RhoType::Regress,
                "freg" => RhoType::Freg,
                "tscorr" => RhoType::Tscorr,
                "dw" => RhoType::Dw,
                "theil" => RhoType::Theil,
                "nagar" => RhoType::Nagar,
                o => return Err(rc_err(198, format!("rhotype({o}) invalid"))),
            },
        };
        let tol = util::opt_num(cmd, "tolerance", 3)?.unwrap_or(1e-6);
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(100.0) as usize;
        let (dep, indep) = split_dep(ds, &cmd.main)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra: vec![] })?;
        let (pos, times) = time_order(ds, st, &d)?;
        let act = d.active();
        let k = act.len();
        let x: Vec<Vec<f64>> = pos.iter().map(|&t| act.iter().map(|&j| d.cols[j][t]).collect()).collect();
        let y: Vec<f64> = pos.iter().map(|&t| d.y[t]).collect();
        let n = y.len();
        // first observation of each run of consecutive periods
        let first: Vec<bool> = (0..n).map(|i| i == 0 || times[i] - times[i - 1] > 1.0 + 1e-9).collect();
        let n_gaps = first.iter().filter(|f| **f).count() - 1;
        if n < k + 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let (b0, _, e0) = ols_rows(&x, &y);
        let _ = b0;
        let dw0 = dw(&e0, &first);
        let fit_at = |rho: f64| -> PraisFit {
            let s = (1.0 - rho * rho).max(0.0).sqrt();
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut keep = Vec::new();
            for t in 0..n {
                if first[t] {
                    if corc {
                        continue;
                    }
                    xs.push(x[t].iter().map(|v| v * s).collect::<Vec<f64>>());
                    ys.push(y[t] * s);
                } else {
                    xs.push(x[t].iter().zip(&x[t - 1]).map(|(a, b)| a - rho * b).collect());
                    ys.push(y[t] - rho * y[t - 1]);
                }
                keep.push(t);
            }
            let (b, inv, es) = ols_rows(&xs, &ys);
            let resid: Vec<f64> = (0..n).map(|t| y[t] - x[t].iter().zip(&b).map(|(a, c)| a * c).sum::<f64>()).collect();
            let rss = es.iter().map(|v| v * v).sum();
            let estar_first = keep.iter().enumerate().map(|(i, &t)| i == 0 || first[t] || keep[i - 1] + 1 != t).collect();
            PraisFit { b, inv, rss, ystar: ys, estar: es, estar_first, resid }
        };
        let mut log = Vec::new();
        let mut rho;
        let mut iters = 0usize;
        if ssesearch {
            // golden-section search for the rho minimising the transformed SSE
            let (mut a, mut c) = (-0.999_999, 0.999_999);
            let g = (5f64.sqrt() - 1.0) / 2.0;
            let mut x1 = c - g * (c - a);
            let mut x2 = a + g * (c - a);
            let (mut f1, mut f2v) = (fit_at(x1).rss, fit_at(x2).rss);
            while c - a > tol {
                iters += 1;
                if f1 < f2v {
                    c = x2;
                    x2 = x1;
                    f2v = f1;
                    x1 = c - g * (c - a);
                    f1 = fit_at(x1).rss;
                } else {
                    a = x1;
                    x1 = x2;
                    f1 = f2v;
                    x2 = a + g * (c - a);
                    f2v = fit_at(x2).rss;
                }
                log.push(format!("Iteration {}: rho = {:.4}, criterion = {}", iters, (a + c) / 2.0, fmt_g(-(f1.min(f2v)), 10)));
            }
            rho = (a + c) / 2.0;
        } else {
            log.push("Iteration 0:  rho = 0.0000".to_string());
            rho = estimate_rho(&e0, &first, kind, k);
            iters = 1;
            log.push(format!("Iteration 1:  rho = {:.4}", rho));
            if !twostep {
                loop {
                    let f = fit_at(rho);
                    let new = estimate_rho(&f.resid, &first, kind, k).clamp(-0.999_999_9, 0.999_999_9);
                    iters += 1;
                    let pad = if iters < 10 { "  " } else { " " };
                    log.push(format!("Iteration {iters}:{pad}rho = {:.4}", new));
                    let done = (new - rho).abs() < tol;
                    rho = new;
                    if done || iters >= maxiter {
                        break;
                    }
                }
            }
        }
        let fit = fit_at(rho);
        let n_used = fit.ystar.len() as f64;
        let df_r = n_used - k as f64;
        let s2 = fit.rss / df_r;
        let mut b = vec![0.0; d.k()];
        for (a, &j) in act.iter().enumerate() {
            b[j] = fit.b[a];
        }
        let v = est::expand_v(&fit.inv.scale(s2), &act, d.k());
        let ybar = fit.ystar.iter().sum::<f64>() / n_used;
        let tss: f64 = if d.has_cons { fit.ystar.iter().map(|v| (v - ybar).powi(2)).sum() } else { fit.ystar.iter().map(|v| v * v).sum() };
        let mss = tss - fit.rss;
        let df_m = k as f64 - if d.has_cons { 1.0 } else { 0.0 };
        let f = (mss / df_m) / s2;
        let pf = dist::ftail(df_m, df_r, f);
        let r2 = mss / tss;
        let adj = 1.0 - (1.0 - r2) * (n_used - if d.has_cons { 1.0 } else { 0.0 }) / df_r;
        let rmse = s2.sqrt();
        let dwt = dw(&fit.estar, &fit.estar_first);
        let level = util::level(cmd, st)?;
        let method = if corc { "Cochrane–Orcutt" } else { "Prais–Winsten" };
        let how = if twostep { "twostep" } else if ssesearch { "SSE search" } else { "iterated" };
        let title = format!("{method} AR(1) regression with {how} estimates");
        let mut out = d.notes.clone();
        out.push(String::new());
        out.extend(log);
        out.push(String::new());
        out.push(title.clone());
        out.push(String::new());
        let ss = |v: f64| fmt_g(v, 10);
        let g9 = |v: f64| fmt_g(v, 9);
        out.push(format!("      Source |       SS           df       MS      Number of obs   = {:>9}", g9(n_used)));
        out.push(format!("-------------+----------------------------------   {:<16}= {:>9}", format!("F({}, {})", df_m, df_r), f2(f)));
        out.push(format!("       Model | {:>11} {:>9} {:>11}   Prob > F        = {:>9}", ss(mss), df_m, ss(mss / df_m), f4(pf)));
        out.push(format!("    Residual | {:>11} {:>9} {:>11}   R-squared       = {:>9}", ss(fit.rss), df_r, ss(s2), f4(r2)));
        out.push(format!("-------------+----------------------------------   Adj R-squared   = {:>9}", f4(adj)));
        out.push(format!("       Total | {:>11} {:>9} {:>11}   Root MSE        = {:>9}", ss(tss), n_used - 1.0, ss(tss / (n_used - 1.0)), crate::format::fmt(rmse, "%10.5g").trim()));
        out.push(String::new());
        let mut table = est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &d.names,
            b: &b,
            v: &v,
            omitted: &d.omitted,
            is_base: &d.is_base,
            df: Some(df_r),
            level,
            eform: None,
            coef_title: "Coefficient",
        });
        let bar = table.pop().unwrap_or_default();
        let lw = bar.chars().count().saturating_sub(66);
        table.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        table.push(format!("{:>lw$} | {:>10}", "rho", fmt_g(rho, 8)));
        table.push(bar);
        out.extend(table);
        out.push(format!("Durbin–Watson statistic (original)    = {:.6}", dw0));
        out.push(format!("Durbin–Watson statistic (transformed) = {:.6}", dwt));
        let text = out.join("\n");
        let mut d2 = d.clone();
        if corc {
            // the first observation does not enter the estimation sample
            let t0 = d.rows[pos[0]];
            d2.mask[t0] = false;
        }
        est::post(st, "prais", &d2, &b, &v, &|e: &mut Results| {
            e.set("N", n_used);
            e.set("N_gaps", n_gaps as f64);
            e.set("df_m", df_m);
            e.set("df_r", df_r);
            e.set("F", f);
            e.set("r2", r2);
            e.set("r2_a", adj);
            e.set("rmse", rmse);
            e.set("mss", mss);
            e.set("rss", fit.rss);
            e.set("rho", rho);
            e.set("dw", dwt);
            e.set("dw_0", dw0);
            e.set("tol", tol);
            e.set("max_ic", maxiter as f64);
            e.set("ic", iters as f64);
            e.set_str("method", if twostep { "twostep" } else if ssesearch { "SSE search" } else { "iterated" });
            e.set_str("tranmeth", if corc { "corc" } else { "prais" });
            e.set_str("rhotype", match kind { RhoType::Regress => "regress", RhoType::Freg => "freg", RhoType::Tscorr => "tscorr", RhoType::Dw => "dw", RhoType::Theil => "theil", RhoType::Nagar => "nagar" });
            e.set_str("title", title.clone());
            e.set_str("predict", "prais_p");
            e.set_str("vce", "ols");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
