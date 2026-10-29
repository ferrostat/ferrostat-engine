//! `threshold`: threshold regression. The coefficients of the region
//! variables (and the constant) change when the threshold variable crosses
//! unknown thresholds, which are found by a grid search minimizing the sum
//! of squared residuals (sequentially for several thresholds); the number
//! of thresholds can be chosen by an information criterion.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_ml::{self, Post};
use crate::tsops;
use crate::tsreg::hline;
use crate::util;

/// OLS on the rows: (SSR, b, (X'X)^-1).
fn ols(x: &[Vec<f64>], y: &[f64]) -> Option<(f64, Vec<f64>, Matrix)> {
    let k = x.first()?.len();
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for (r, yv) in x.iter().zip(y) {
        for a in 0..k {
            xy[a] += r[a] * yv;
            for c in 0..=a {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    for a in 0..k {
        for c in 0..a {
            m[(c, a)] = m[(a, c)];
        }
    }
    let (inv, dropped) = m.sym_inverse(1e-12);
    if !dropped.is_empty() {
        return None;
    }
    let b = ts_ml::mv(&inv, &xy);
    let ssr = x.iter().zip(y).map(|(r, yv)| (yv - ts_ml::dot(r, &b)).powi(2)).sum();
    Some((ssr, b, inv))
}

fn design(common: &[Vec<f64>], region: &[Vec<f64>], q: &[f64], th: &[f64]) -> Vec<Vec<f64>> {
    let nr = th.len() + 1;
    (0..q.len())
        .map(|t| {
            let r = th.iter().filter(|&&g| q[t] > g).count();
            let mut row = Vec::with_capacity(common[t].len() + region[t].len() * nr);
            for j in 0..nr {
                for v in &region[t] {
                    row.push(if j == r { *v } else { 0.0 });
                }
            }
            row.extend(common[t].iter().copied());
            row
        })
        .collect()
}

pub struct Threshold;

impl CommandHandler for Threshold {
    fn name(&self) -> &'static str {
        "threshold"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "threshold");
        }
        let tv = cmd.opt("threshvar", 7).ok_or_else(|| rc_err(198, "option threshvar() required"))?.trim().to_string();
        let tv = tsops::resolve(ds, &tv)?;
        let (dep, indep) = crate::tsreg::split_dep(ds, &cmd.main)?;
        let common_names: Vec<String> = tsops::expand_text(&indep).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        let mut region_names: Vec<String> = match cmd.opt("regionvars", 7) {
            Some(r) => tsops::expand_text(r).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?,
            None => vec![],
        };
        if !cmd.has_opt("noconstant", 4) {
            region_names.push("_cons".into());
        }
        if region_names.is_empty() {
            return Err(rc_err(198, "no region-specific variables"));
        }
        let yv = tsops::values(ds, st, &dep)?;
        let qv = tsops::values(ds, st, &tv)?;
        let colv = |n: &String| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) };
        let cc: Vec<Vec<f64>> = common_names.iter().map(colv).collect::<Result<_, _>>()?;
        let rc: Vec<Vec<f64>> = region_names.iter().map(colv).collect::<Result<_, _>>()?;
        let mask0 = util::touse(ds, st, cmd)?;
        let mut refs: Vec<&Vec<f64>> = vec![&yv, &qv];
        refs.extend(cc.iter());
        refs.extend(rc.iter());
        let (rows, times) = if st.tsset.is_some() {
            tsops::ordered_rows(ds, st, &mask0, &refs)?
        } else {
            let r: Vec<usize> = (0..ds.n_obs).filter(|&i| mask0[i] && refs.iter().all(|c| !is_missing(c[i]))).collect();
            let t = r.iter().map(|&i| i as f64 + 1.0).collect();
            (r, t)
        };
        let n = rows.len();
        let y: Vec<f64> = rows.iter().map(|&i| yv[i]).collect();
        let q: Vec<f64> = rows.iter().map(|&i| qv[i]).collect();
        let common: Vec<Vec<f64>> = rows.iter().map(|&i| cc.iter().map(|c| c[i]).collect()).collect();
        let region: Vec<Vec<f64>> = rows.iter().map(|&i| rc.iter().map(|c| c[i]).collect()).collect();
        let trim = util::opt_num(cmd, "trim", 4)?.unwrap_or(10.0) / 100.0;
        let mut sorted = q.clone();
        sorted.sort_by(f64::total_cmp);
        let lo = sorted[((trim * n as f64).ceil() as usize).min(n - 1)];
        let hi = sorted[((((1.0 - trim) * n as f64).floor() as usize).max(1) - 1).min(n - 1)];
        let mut cands: Vec<f64> = sorted.iter().copied().filter(|v| *v >= lo && *v < hi).collect();
        cands.dedup();
        let ic_kind = cmd.opt("ic", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "bic".into());
        let (max_th, choose) = match (util::opt_num(cmd, "optthresh", 4)?, util::opt_num(cmd, "nthresholds", 2)?) {
            (Some(o), _) => (o as usize, true),
            (None, Some(k)) => (k as usize, false),
            _ => (1, false),
        };
        let kr = region_names.len();
        let kc = common_names.len();
        let ic = |ssr: f64, nth: usize| -> (f64, f64, f64) {
            let k = (kr * (nth + 1) + kc) as f64;
            let nf = n as f64;
            let base = nf * (ssr / nf).ln();
            (base + 2.0 * k, base + k * nf.ln(), base + 2.0 * k * nf.ln().ln())
        };
        let pick_ic = |t: (f64, f64, f64)| match ic_kind.as_str() {
            "aic" => t.0,
            "hqic" => t.2,
            _ => t.1,
        };
        let mut log = Vec::new();
        let mut ths: Vec<f64> = Vec::new();
        let mut path: Vec<(f64, f64)> = Vec::new(); // (threshold, ssr) in order found
        let ssr0 = ols(&design(&common, &region, &q, &[]), &y).ok_or_else(|| rc_err(2001, "insufficient observations"))?.0;
        let mut best_n = 0usize;
        let mut best_ic = pick_ic(ic(ssr0, 0));
        let mut chosen: Vec<f64> = vec![];
        let min_region = (kr + kc + 1).max(1);
        for k in 1..=max_th {
            let ncand = cands.iter().filter(|c| !ths.contains(c)).count();
            log.push(format!("Searching for threshold: {k}"));
            log.push(format!("(running {ncand} regressions)"));
            let mut dots = String::new();
            let mut best: Option<(f64, f64)> = None;
            for (ci, &g) in cands.iter().filter(|c| !ths.contains(c)).enumerate() {
                let mut tt = ths.clone();
                tt.push(g);
                tt.sort_by(f64::total_cmp);
                // every region needs enough observations
                let mut counts = vec![0usize; tt.len() + 1];
                for &qq in &q {
                    counts[tt.iter().filter(|&&x| qq > x).count()] += 1;
                }
                dots.push('.');
                if (ci + 1) % 50 == 0 {
                    log.push(format!("{dots}{:>6}", ci + 1));
                    dots.clear();
                }
                if counts.iter().any(|&c| c < min_region) {
                    continue;
                }
                if let Some((ssr, _, _)) = ols(&design(&common, &region, &q, &tt), &y) {
                    if best.is_none_or(|b| ssr < b.1 - 1e-12) {
                        best = Some((g, ssr));
                    }
                }
            }
            if !dots.is_empty() {
                log.push(dots);
            }
            let Some((g, ssr)) = best else { break };
            ths.push(g);
            path.push((g, ssr));
            let v = pick_ic(ic(ssr, k));
            if !choose || v < best_ic {
                if choose {
                    best_ic = v;
                }
                best_n = k;
                chosen = ths.clone();
            }
        }
        if !choose {
            best_n = ths.len();
            chosen = ths.clone();
        }
        let mut thr = chosen.clone();
        thr.sort_by(f64::total_cmp);
        let x = design(&common, &region, &q, &thr);
        let (ssr, b, xtxi) = ols(&x, &y).ok_or_else(|| rc_err(2001, "insufficient observations"))?;
        let k = b.len();
        let s2 = ssr / (n - k) as f64;
        let v = xtxi.scale(s2);
        let mut names = Vec::new();
        for r in 0..=thr.len() {
            for nm in &region_names {
                names.push(format!("Region{}:{nm}", r + 1));
            }
        }
        for nm in &common_names {
            names.push(format!("{dep}:{nm}"));
        }
        let (aic, bic, hqic) = ic(ssr, best_n);
        let tvar = st.tsset.as_ref().map(|t| t.1.clone()).unwrap_or_default();
        let full = if tvar.is_empty() { String::new() } else { format!("Full sample: {} thru {}", tsops::time_label(ds, &tvar, times[0]), tsops::time_label(ds, &tvar, *times.last().unwrap())) };
        let mut header = vec![
            hline("", "Number of obs", &fmt_g(n as f64, 10)),
            hline(&full, "AIC", &format!("{:.4}", aic)),
            hline(&format!("Number of thresholds = {best_n}"), "BIC", &format!("{:.4}", bic)),
            hline(&format!("Threshold variable: {tv}"), "HQIC", &format!("{:.4}", hqic)),
            String::new(),
            "---------------------------------".into(),
            "Order |   Threshold          SSR".into(),
            "------+--------------------------".into(),
        ];
        let mut tmat = Matrix::zeros(best_n, 2);
        for (i, (g, s)) in path.iter().take(best_n).enumerate() {
            header.push(format!("{:>5} | {:>11.4} {:>12.4}", i + 1, g, s));
            tmat[(i, 0)] = *g;
            tmat[(i, 1)] = *s;
        }
        header.push("---------------------------------".into());
        let text = ts_ml::post(
            st,
            cmd,
            Post { title: "Threshold regression", cmd: "threshold", depvar: &dep, names, b, v, ll: f64::NAN, n, header, vcetype: "", vce: "ols", log, notes: vec![], converged: true, iters: 0, df: None },
            None,
        )?;
        let mut sample = vec![false; ds.n_obs];
        for &r in &rows {
            sample[r] = true;
        }
        let e = &mut st.e;
        e.sample = Some(sample);
        e.scalars.remove("ll");
        e.set("nthresholds", best_n as f64);
        e.set("ssr", ssr);
        e.set("aic", aic);
        e.set("bic", bic);
        e.set("hqic", hqic);
        e.set("k_eq", (best_n + 1 + usize::from(kc > 0)) as f64);
        e.set_str("threshvar", tv.clone());
        e.set_str("regionvars", region_names.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
        e.set_str("indepvars", common_names.join(" "));
        e.set_mat("thresholds", tmat.with_names((1..=best_n).map(|i| format!("r{i}")).collect(), vec!["threshold".into(), "SSR".into()]));
        e.set_str("predict", "threshold_p");
        e.set_str("_thr", thr.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(" "));
        Ok(text)
    }
}

/// `predict` after threshold: xb (default) or residuals.
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let thr: Vec<f64> = e.macros.get("_thr").map(|s| s.split_whitespace().filter_map(|v| v.parse().ok()).collect()).unwrap_or_default();
    let qv = tsops::values(ds, st, e.macros.get("threshvar").map(String::as_str).unwrap_or(""))?;
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let newvar = ts_ml::predict_name(cmd)?;
    let mask = util::touse(ds, st, cmd)?;
    let mut cols: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut out = vec![ferrostat_core::missing::SYSMISS; ds.n_obs];
    for (j, c) in b.colnames.iter().enumerate() {
        let (eq, nm) = c.split_once(':').unwrap_or(("", c));
        if !cols.contains_key(nm) {
            let v = if nm == "_cons" { vec![1.0; ds.n_obs] } else { tsops::values(ds, st, nm)? };
            cols.insert(nm.to_string(), v);
        }
        let region = eq.strip_prefix("Region").and_then(|r| r.parse::<usize>().ok());
        for i in 0..ds.n_obs {
            if is_missing(qv[i]) {
                continue;
            }
            let r = thr.iter().filter(|&&g| qv[i] > g).count() + 1;
            if region.is_some_and(|x| x != r) {
                continue;
            }
            let x = cols[nm][i];
            if is_missing(x) || (is_missing(out[i]) && out[i] != ferrostat_core::missing::SYSMISS) {
                out[i] = f64::NAN;
                continue;
            }
            out[i] = if out[i] == ferrostat_core::missing::SYSMISS { 0.0 } else { out[i] } + b.data[j] * x;
        }
    }
    let resid = cmd.has_opt("residuals", 1);
    let yv = tsops::values(ds, st, &dep)?;
    for i in 0..ds.n_obs {
        if out[i].is_nan() {
            out[i] = ferrostat_core::missing::SYSMISS;
        } else if resid && !is_missing(out[i]) {
            out[i] = if is_missing(yv[i]) { ferrostat_core::missing::SYSMISS } else { yv[i] - out[i] };
        }
    }
    ts_ml::add_prediction(ds, &newvar, out, &mask, if resid || cmd.has_opt("xb", 2) { "" } else { "(option xb assumed; fitted values)" })
}
