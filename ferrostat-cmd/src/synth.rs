//! `synth`: synthetic control method (Abadie & Gardeazabal 2003; Abadie,
//! Diamond & Hainmueller 2010, JASA 105:493-505; 2011, J Stat Softw
//! 42(13)), reimplemented from the papers. The donor weights W solve a
//! quadratic program on the simplex for a given diagonal predictor weight
//! matrix V; V is regression based by default or chosen by a nested
//! search minimizing the pre-treatment mean squared prediction error.
//!
//!   synth depvar predictorvars, trunit(#) trperiod(#) [counit(numlist)
//!       xperiod(numlist) mspeperiod(numlist) resultsperiod(numlist)
//!       nested allopt customV(numlist) figure]
//!
//! Predictors are `var` (averaged over xperiod), `var(#)`,
//! `var(#(#)#)` or `var(#&#&...)`.

use std::collections::{BTreeMap, BTreeSet};

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::teffects::solve;
use crate::{util, varlist};

/// Euclidean projection onto the probability simplex.
fn project_simplex(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut css = 0.0;
    let mut theta = 0.0;
    for (j, &x) in u.iter().enumerate() {
        css += x;
        let t = (css - 1.0) / (j + 1) as f64;
        if x - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

/// min w'Hw - 2 f'w over the simplex: accelerated projected gradient,
/// then an exact active-set refinement.
pub(crate) fn simplex_qp(h: &[Vec<f64>], f: &[f64]) -> Vec<f64> {
    let j = f.len();
    let lip = 2.0 * h.iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(1e-300, f64::max);
    let grad = |w: &[f64]| -> Vec<f64> { (0..j).map(|a| 2.0 * (crate::teffects::dot(&h[a], w) - f[a])).collect() };
    let mut w = vec![1.0 / j as f64; j];
    let mut yv = w.clone();
    let mut tk: f64 = 1.0;
    for _ in 0..4000 {
        let g = grad(&yv);
        let wn = project_simplex(&yv.iter().zip(&g).map(|(a, b)| a - b / lip).collect::<Vec<_>>());
        let tn: f64 = (1.0 + (1.0 + 4.0 * tk * tk).sqrt()) / 2.0;
        let diff: f64 = wn.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum();
        yv = wn.iter().zip(&w).map(|(a, b)| a + (tk - 1.0) / tn * (a - b)).collect();
        w = wn;
        tk = tn;
        if diff < 1e-12 {
            break;
        }
    }
    // active-set refinement on the support
    let mut sup: BTreeSet<usize> = (0..j).filter(|&a| w[a] > 1e-7).collect();
    for _ in 0..100 {
        let s: Vec<usize> = sup.iter().copied().collect();
        let m = s.len();
        let mut a = vec![vec![0.0; m + 1]; m + 1];
        let mut b = vec![0.0; m + 1];
        for (r, &sr) in s.iter().enumerate() {
            for (c, &sc) in s.iter().enumerate() {
                a[r][c] = 2.0 * h[sr][sc];
            }
            a[r][m] = 1.0;
            a[m][r] = 1.0;
            b[r] = 2.0 * f[sr];
        }
        b[m] = 1.0;
        let Some(sol) = solve(&a, &b) else { break };
        if let Some((pos, _)) = sol[..m].iter().enumerate().filter(|(_, v)| **v < -1e-12).min_by(|x, y| x.1.total_cmp(y.1)) {
            sup.remove(&s[pos]);
            continue;
        }
        let mut cand = vec![0.0; j];
        for (r, &sr) in s.iter().enumerate() {
            cand[sr] = sol[r].max(0.0);
        }
        // KKT: reduced gradient of excluded donors must not be below the multiplier
        let g = grad(&cand);
        let mu = -sol[m];
        let worst = (0..j).filter(|a| !sup.contains(a)).map(|a| (a, g[a] - mu)).min_by(|x, y| x.1.total_cmp(&y.1));
        let obj = |w: &[f64]| crate::teffects::dot(w, &(0..j).map(|a| crate::teffects::dot(&h[a], w)).collect::<Vec<_>>()) - 2.0 * crate::teffects::dot(f, w);
        if obj(&cand) <= obj(&w) + 1e-12 * obj(&w).abs().max(1e-300) {
            w = cand;
        }
        match worst {
            Some((a, v)) if v < -1e-9 * (1.0 + mu.abs()) => {
                sup.insert(a);
            }
            _ => break,
        }
    }
    let s: f64 = w.iter().sum();
    w.iter().map(|v| v / s).collect()
}

/// Nelder–Mead minimization (unconstrained).
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, maxeval: usize) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += step;
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    let mut evals = n + 1;
    while evals < maxeval {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() <= 1e-12 * vals[0].abs().max(1e-300) {
            break;
        }
        let cen: Vec<f64> = (0..n).map(|d| pts[..n].iter().map(|p| p[d]).sum::<f64>() / n as f64).collect();
        let at = |t: f64| -> Vec<f64> { (0..n).map(|d| cen[d] + t * (pts[n][d] - cen[d])).collect() };
        let xr = at(-1.0);
        let fr = f(&xr);
        evals += 1;
        if fr < vals[0] {
            let xe = at(-2.0);
            let fe = f(&xe);
            evals += 1;
            if fe < fr {
                pts[n] = xe;
                vals[n] = fe;
            } else {
                pts[n] = xr;
                vals[n] = fr;
            }
        } else if fr < vals[n - 1] {
            pts[n] = xr;
            vals[n] = fr;
        } else {
            let xc = if fr < vals[n] { at(-0.5) } else { at(0.5) };
            let fc = f(&xc);
            evals += 1;
            if fc < vals[n].min(fr) {
                pts[n] = xc;
                vals[n] = fc;
            } else {
                for i in 1..=n {
                    pts[i] = (0..n).map(|d| pts[0][d] + 0.5 * (pts[i][d] - pts[0][d])).collect();
                    vals[i] = f(&pts[i]);
                    evals += 1;
                }
            }
        }
    }
    let b = (0..=n).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    (pts[b].clone(), vals[b])
}

fn softmax(t: &[f64]) -> Vec<f64> {
    let mx = t.iter().copied().fold(0.0, f64::max);
    let mut e: Vec<f64> = std::iter::once(0.0).chain(t.iter().copied()).map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter_mut().for_each(|v| *v /= s);
    e
}

pub struct Synth;

impl CommandHandler for Synth {
    fn name(&self) -> &'static str {
        "synth"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (pvar, tvar) = match &st.tsset {
            Some((Some(p), t)) if !t.is_empty() => (p.clone(), t.clone()),
            _ => return Err(rc_err(459, "panel and time variables must be set with tsset panelvar timevar")),
        };
        let words = split_top_level_words(&cmd.main);
        let dep = varlist::resolve_name(ds, words.first().ok_or_else(|| rc_err(100, "varlist required"))?)?;
        let trunit = util::opt_num(cmd, "trunit", 3)?.ok_or_else(|| rc_err(198, "option trunit() required"))?;
        let trperiod = util::opt_num(cmd, "trperiod", 3)?.ok_or_else(|| rc_err(198, "option trperiod() required"))?;
        let pcol = util::col(ds, &pvar)?;
        let tcol = util::col(ds, &tvar)?;
        let mask = util::touse(ds, st, cmd)?;
        let mut cell: BTreeMap<(i64, i64), usize> = BTreeMap::new();
        let mut units: BTreeSet<i64> = BTreeSet::new();
        let mut times: BTreeSet<i64> = BTreeSet::new();
        for i in 0..ds.n_obs {
            if mask[i] && !is_missing(pcol[i]) && !is_missing(tcol[i]) {
                cell.insert((pcol[i] as i64, tcol[i] as i64), i);
                units.insert(pcol[i] as i64);
                times.insert(tcol[i] as i64);
            }
        }
        let tr = trunit as i64;
        if !units.contains(&tr) {
            return Err(rc_err(198, "treated unit not found"));
        }
        let donors: Vec<i64> = match cmd.opt("counit", 3) {
            Some(c) => crate::margins::numlist(c)?.into_iter().map(|v| v as i64).filter(|&u| u != tr).collect(),
            None => units.iter().copied().filter(|&u| u != tr).collect(),
        };
        if donors.len() < 2 {
            return Err(rc_err(2001, "at least two control units are required"));
        }
        let pre: Vec<i64> = times.iter().copied().filter(|&t| (t as f64) < trperiod).collect();
        if pre.is_empty() {
            return Err(rc_err(198, "no pre-treatment periods"));
        }
        let periods = |name: &str| -> Result<Option<Vec<i64>>, String> { cmd.opt(name, 4).map(|s| crate::margins::numlist(s).map(|v| v.into_iter().map(|x| x as i64).collect())).transpose() };
        let xper = periods("xperiod")?.unwrap_or_else(|| pre.clone());
        let mper = periods("mspeperiod")?.unwrap_or_else(|| pre.clone());
        let rper = periods("resultsperiod")?.unwrap_or_else(|| times.iter().copied().collect());
        let value = |var: &[f64], u: i64, t: i64| -> f64 { cell.get(&(u, t)).map_or(f64::NAN, |&i| if is_missing(var[i]) { f64::NAN } else { var[i] }) };
        // predictors
        let mut pnames = Vec::new();
        let mut x: Vec<Vec<f64>> = Vec::new(); // predictor x unit (treated first)
        let all_units: Vec<i64> = std::iter::once(tr).chain(donors.iter().copied()).collect();
        for w in &words[1..] {
            let (name, spec) = match w.find('(') {
                Some(i) => (&w[..i], Some(w[i + 1..].trim_end_matches(')').to_string())),
                None => (w.as_str(), None),
            };
            for v in varlist::expand(ds, name)? {
                let col = util::col(ds, &v)?;
                let per: Vec<i64> = match &spec {
                    Some(s) if s.contains('&') => s.split('&').map(|p| p.trim().parse::<f64>().map(|x| x as i64).map_err(|_| rc_err(198, format!("invalid period in {w}")))).collect::<Result<_, _>>()?,
                    Some(s) => crate::margins::numlist(s)?.into_iter().map(|x| x as i64).collect(),
                    None => xper.clone(),
                };
                let row: Vec<f64> = all_units
                    .iter()
                    .map(|&u| {
                        let vals: Vec<f64> = per.iter().map(|&t| value(&col, u, t)).filter(|v| !v.is_nan()).collect();
                        if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
                    })
                    .collect();
                if row.iter().any(|v| v.is_nan()) {
                    return Err(rc_err(416, format!("predictor {v} is missing for some unit in the requested periods")));
                }
                pnames.push(match &spec {
                    Some(s) => format!("{v}({s})"),
                    None => v.clone(),
                });
                x.push(row);
            }
        }
        if x.is_empty() {
            return Err(rc_err(198, "predictors required"));
        }
        let k = x.len();
        let j = donors.len();
        // normalize the predictors by their standard deviation across units
        let xs: Vec<Vec<f64>> = x
            .iter()
            .map(|r| {
                let sd = util::variance(r).sqrt();
                let sd = if sd > 0.0 { sd } else { 1.0 };
                r.iter().map(|v| v / sd).collect()
            })
            .collect();
        let dcol = util::col(ds, &dep)?;
        let z: Vec<Vec<f64>> = mper.iter().map(|&t| all_units.iter().map(|&u| value(&dcol, u, t)).collect()).collect();
        if z.iter().flatten().any(|v| v.is_nan()) {
            return Err(rc_err(416, format!("{dep} is missing for some unit in the MSPE periods")));
        }
        let weights_for = |v: &[f64]| -> Vec<f64> {
            let h: Vec<Vec<f64>> = (0..j).map(|a| (0..j).map(|b| (0..k).map(|m| v[m] * xs[m][a + 1] * xs[m][b + 1]).sum()).collect()).collect();
            let f: Vec<f64> = (0..j).map(|a| (0..k).map(|m| v[m] * xs[m][a + 1] * xs[m][0]).sum()).collect();
            simplex_qp(&h, &f)
        };
        let loss = |w: &[f64]| -> f64 { z.iter().map(|r| (r[0] - (0..j).map(|a| w[a] * r[a + 1]).sum::<f64>()).powi(2)).sum::<f64>() / z.len() as f64 };
        // regression-based V: squared coefficients of the outcomes on the predictors
        let reg_v = || -> Vec<f64> {
            let nu = all_units.len();
            let kk = k + 1;
            let mut xx = vec![vec![0.0; kk]; kk];
            for u in 0..nu {
                let row: Vec<f64> = (0..k).map(|m| xs[m][u]).chain(std::iter::once(1.0)).collect();
                for a in 0..kk {
                    for b in 0..kk {
                        xx[a][b] += row[a] * row[b];
                    }
                }
            }
            for (a, r) in xx.iter_mut().enumerate().take(k) {
                r[a] += 1e-8;
            }
            let mut v = vec![0.0; k];
            for r in &z {
                let xy: Vec<f64> = (0..kk).map(|a| (0..nu).map(|u| if a < k { xs[a][u] } else { 1.0 } * r[u]).sum()).collect();
                if let Some(b) = solve(&xx, &xy) {
                    for m in 0..k {
                        v[m] += b[m] * b[m];
                    }
                }
            }
            let s: f64 = v.iter().sum();
            if s > 0.0 { v.iter().map(|x| x / s).collect() } else { vec![1.0 / k as f64; k] }
        };
        let (vdiag, how) = if let Some(cv) = cmd.opt("customv", 7) {
            let v = crate::margins::numlist(cv)?;
            if v.len() != k {
                return Err(rc_err(198, "customV() needs one weight per predictor"));
            }
            let s: f64 = v.iter().sum();
            (v.iter().map(|x| x / s).collect::<Vec<_>>(), "user supplied")
        } else if cmd.has_opt("nested", 4) && k > 1 {
            let obj = |t: &[f64]| loss(&weights_for(&softmax(t)));
            let to_theta = |v: &[f64]| -> Vec<f64> { (1..k).map(|m| (v[m].max(1e-10) / v[0].max(1e-10)).ln()).collect() };
            let starts = vec![to_theta(&reg_v()), vec![0.0; k - 1]];
            let mut best: Option<(Vec<f64>, f64)> = None;
            let iters = if cmd.has_opt("allopt", 3) { 4000 } else { 1500 };
            for s0 in starts {
                let mut cur = nelder_mead(&obj, &s0, 1.0, iters);
                // restart once from the optimum
                let again = nelder_mead(&obj, &cur.0, 0.3, iters / 2);
                if again.1 < cur.1 {
                    cur = again;
                }
                if best.as_ref().is_none_or(|b| cur.1 < b.1) {
                    best = Some(cur);
                }
            }
            (softmax(&best.unwrap().0), "nested optimization")
        } else {
            (if k == 1 { vec![1.0] } else { reg_v() }, "regression based")
        };
        let w = weights_for(&vdiag);
        let rmspe = loss(&w).sqrt();
        // labels
        let ulab = |u: i64| -> String {
            ds.get_var(&pvar).and_then(|v| ds.value_labels.get(&v.value_label_name)).and_then(|vl| vl.lookup(u as f64)).map(str::to_string).unwrap_or_else(|| u.to_string())
        };
        let mut out = Vec::new();
        out.push("Synthetic Control Method for Comparative Case Studies".into());
        out.push("-".repeat(80));
        out.push(String::new());
        out.push("First Step: Data Setup".into());
        out.push("-".repeat(80));
        out.push(format!("   Treated Unit: {}", ulab(tr)));
        out.push(format!("   Control Units: {}", donors.iter().map(|&u| ulab(u)).collect::<Vec<_>>().join(", ")));
        out.push(format!("   Dependent Variable: {dep}"));
        out.push(format!("   MSPE minimized for periods: {}", mper.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")));
        out.push(format!("   Results obtained for periods: {}", rper.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")));
        out.push(format!("   Predictors: {}", pnames.join(" ")));
        out.push(format!("   Unless period is specified, predictors are averaged over: {}", xper.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")));
        out.push(String::new());
        out.push("Second Step: Run Optimization".into());
        out.push("-".repeat(80));
        out.push(format!("   Predictor weights (V): {how}"));
        out.push(String::new());
        out.push("Third Step: Obtain Results".into());
        out.push("-".repeat(80));
        out.push("Loss: Root Mean Squared Prediction Error".into());
        out.push(String::new());
        out.push("-".repeat(22));
        out.push(format!("{:>9} | {}", "RMSPE", fmt_g(rmspe, 7).trim()));
        out.push("-".repeat(22));
        out.push("Unit Weights:".into());
        out.push(String::new());
        out.push("-".repeat(30));
        out.push(format!("{:>16} | {:>11}", "Co_No", "Unit_Weight"));
        out.push(format!("{}+{}", "-".repeat(17), "-".repeat(12)));
        for (a, &u) in donors.iter().enumerate() {
            let wv = (w[a] * 1000.0).round() / 1000.0;
            out.push(format!("{:>16} | {:>11}", crate::summarize::abbrev(&ulab(u), 16), fmt_g(wv, 6).trim()));
        }
        out.push("-".repeat(30));
        out.push("Predictor Balance:".into());
        out.push("-".repeat(60));
        out.push(format!("{:>36} | {:>9} {:>10}", "", "Treated", "Synthetic"));
        out.push(format!("{}+{}", "-".repeat(37), "-".repeat(22)));
        let mut xb = Matrix::zeros(k, 2);
        for m in 0..k {
            let syn: f64 = (0..j).map(|a| w[a] * x[m][a + 1]).sum();
            xb[(m, 0)] = x[m][0];
            xb[(m, 1)] = syn;
            out.push(format!("{:>36} | {:>9} {:>10}", crate::summarize::abbrev(&pnames[m], 36), fmt_g(x[m][0], 7), fmt_g(syn, 7)));
        }
        out.push("-".repeat(60));
        // outcome paths
        let ytr: Vec<f64> = rper.iter().map(|&t| value(&dcol, tr, t)).collect();
        let ysy: Vec<f64> = rper.iter().map(|&t| donors.iter().enumerate().map(|(a, &u)| w[a] * value(&dcol, u, t)).sum()).collect();
        if cmd.has_opt("figure", 3) {
            let pts = |v: &[f64]| -> Vec<(f64, f64)> { rper.iter().zip(v).filter(|(_, y)| !y.is_nan()).map(|(&t, &y)| (t as f64, y)).collect() };
            let layers = vec![
                Layer { kind: LayerKind::Line, points: pts(&ytr), label: "treated unit".into(), color: Some((0, 0, 0)), ..Default::default() },
                Layer { kind: LayerKind::Line, points: pts(&ysy), label: "synthetic control unit".into(), pattern: "dash".into(), color: Some((0, 0, 0)), ..Default::default() },
            ];
            let ylab = ds.get_var(&dep).map(|v| if v.label.is_empty() { dep.clone() } else { v.label.clone() }).unwrap_or(dep.clone());
            let mut g = Graph { xtitle: tvar.clone(), ytitle: ylab, layers, legend: true, xlines: vec![trperiod], ..Default::default() };
            graph::finish(&mut g, cmd, st)?;
        }
        let mut wm = Matrix::zeros(j, 2);
        for a in 0..j {
            wm[(a, 0)] = donors[a] as f64;
            wm[(a, 1)] = w[a];
        }
        let wm = wm.with_names(donors.iter().map(|&u| ulab(u)).collect(), vec!["_Co_Number".into(), "_W_Weight".into()]);
        let mut vm = Matrix::zeros(k, k);
        for m in 0..k {
            vm[(m, m)] = vdiag[m];
        }
        let vm = vm.with_names(pnames.clone(), pnames.clone());
        let xb = xb.with_names(pnames.clone(), vec!["Treated".into(), "Synthetic".into()]);
        let col_mat = |v: &[f64], name: &str| -> Matrix {
            let mut m = Matrix::zeros(v.len(), 1);
            for (i, x) in v.iter().enumerate() {
                m[(i, 0)] = if x.is_nan() { ferrostat_core::missing::SYSMISS } else { *x };
            }
            m.with_names(rper.iter().map(|t| t.to_string()).collect(), vec![name.to_string()])
        };
        let mut rm = Matrix::zeros(1, 1);
        rm[(0, 0)] = rmspe;
        let e = st.begin_e();
        e.set_str("cmd", "synth");
        e.set_str("depvar", dep.clone());
        e.set("trunit", trunit);
        e.set("trperiod", trperiod);
        e.set_mat("RMSPE", rm);
        e.set_mat("W_weights", wm);
        e.set_mat("V_matrix", vm);
        e.set_mat("X_balance", xb);
        e.set_mat("Y_treated", col_mat(&ytr, &dep));
        e.set_mat("Y_synthetic", col_mat(&ysy, &dep));
        Ok(out.join("\n"))
    }
}
