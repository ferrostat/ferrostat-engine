//! `ebalance`: entropy balancing (Hainmueller 2012, Political Analysis
//! 20:25-46; Hainmueller & Xu 2013, J Stat Softw 54(7)), reimplemented from
//! the papers: control weights w_i = q_i exp(lambda' c(X_i)) of minimal
//! entropy relative to base weights q that reproduce the treated moments
//! (mean, variance, skewness) of each covariate, found by Newton's method
//! on the convex dual.
//!
//!   ebalance treat covars [if] [in] [, targets(numlist) generate(name)
//!       tolerance(#) maxiter(#) basewt(var)]
//!   ebalance covars [if] [in], manualtargets(numlist) [...]

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::psmatch2::put_var;
use crate::store::Store;
use crate::{util, varlist};

/// Weighted mean, variance (frequency-style) and skewness.
fn moments3(x: &[f64], w: &[f64]) -> (f64, f64, f64) {
    let sw: f64 = w.iter().sum();
    let m = x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let m2 = x.iter().zip(w).map(|(a, b)| b * (a - m).powi(2)).sum::<f64>() / sw;
    let m3 = x.iter().zip(w).map(|(a, b)| b * (a - m).powi(3)).sum::<f64>() / sw;
    let npos = w.iter().filter(|&&v| v > 0.0).count() as f64;
    let var = if npos > 1.0 { m2 * npos / (npos - 1.0) } else { 0.0 };
    (m, var, if m2 > 0.0 { m3 / m2.powf(1.5) } else { 0.0 })
}

/// Entropy-balancing weights: `d` holds the constraint deviations c(X_i) - target
/// (one row per unit); returns weights proportional to q exp(lambda'd).
fn solve_weights(d: &[Vec<f64>], q: &[f64], tol: f64, maxiter: usize, log: &mut Vec<String>) -> Result<Vec<f64>, String> {
    let n = d.len();
    let k = d[0].len();
    // scale the constraints for numerical stability
    let sc: Vec<f64> = (0..k).map(|a| (d.iter().map(|r| r[a] * r[a]).sum::<f64>() / n as f64).sqrt().max(1e-12)).collect();
    let z: Vec<Vec<f64>> = d.iter().map(|r| r.iter().zip(&sc).map(|(v, s)| v / s).collect()).collect();
    let mut lam = vec![0.0; k];
    let eval = |lam: &[f64]| -> (f64, Vec<f64>, Vec<f64>) {
        let e: Vec<f64> = z.iter().map(|r| crate::teffects::dot(r, lam)).collect();
        let mx = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = e.iter().zip(q).map(|(v, qi)| qi * (v - mx).exp()).collect();
        let s: f64 = w.iter().sum();
        let w: Vec<f64> = w.iter().map(|v| v / s).collect();
        let g: Vec<f64> = (0..k).map(|a| (0..n).map(|i| w[i] * z[i][a]).sum()).collect();
        (s.ln() + mx, g, w)
    };
    for it in 0..maxiter.max(1) * 10 {
        let (f, g, w) = eval(&lam);
        let maxdev = g.iter().zip(&sc).map(|(v, s)| (v * s).abs()).fold(0.0, f64::max);
        if it < maxiter {
            log.push(format!("Iteration {}: Max Difference = {}", it + 1, fmt_g(maxdev, 9).trim()));
        }
        if maxdev < tol.min(1e-8) || g.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-12 {
            let _ = f;
            return Ok(w);
        }
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            for a in 0..k {
                for b in 0..k {
                    h[a][b] += w[i] * z[i][a] * z[i][b];
                }
            }
        }
        for a in 0..k {
            for b in 0..k {
                h[a][b] -= g[a] * g[b];
            }
            h[a][a] += 1e-12;
        }
        let step = crate::teffects::solve(&h, &g).ok_or_else(|| rc_err(430, "entropy balancing: singular constraint matrix (collinear moments?)"))?;
        let mut t = 1.0;
        let mut ok = false;
        for _ in 0..50 {
            let cand: Vec<f64> = lam.iter().zip(&step).map(|(l, s)| l - t * s).collect();
            if eval(&cand).0 <= f + 1e-14 * f.abs() {
                lam = cand;
                ok = true;
                break;
            }
            t /= 2.0;
        }
        if !ok {
            break;
        }
    }
    let (_, g, w) = eval(&lam);
    let maxdev = g.iter().zip(&sc).map(|(v, s)| (v * s).abs()).fold(0.0, f64::max);
    if maxdev <= tol {
        return Ok(w);
    }
    Err(rc_err(430, format!("entropy balancing did not converge (maximum difference {}); the targets may be infeasible", fmt_g(maxdev, 6).trim())))
}

pub struct Ebalance;

impl CommandHandler for Ebalance {
    fn name(&self) -> &'static str {
        "ebalance"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut vars = varlist::expand(ds, &cmd.main)?;
        let manual = match cmd.opt("manualtargets", 3) {
            Some(m) => Some(crate::margins::numlist(m)?),
            None => None,
        };
        let tvar = if manual.is_none() {
            if vars.len() < 2 {
                return Err(rc_err(198, "a treatment variable and covariates are required"));
            }
            Some(vars.remove(0))
        } else {
            None
        };
        let gname = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| "_webal".into());
        let tol = util::opt_num(cmd, "tolerance", 3)?.unwrap_or(0.015);
        let maxiter = util::opt_num(cmd, "maxiter", 4)?.unwrap_or(20.0) as usize;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = vars.clone();
        if let Some(t) = &tvar {
            need.push(t.clone());
        }
        util::markout(ds, &mut mask, &need);
        let base = match cmd.opt("basewt", 5) {
            Some(b) => Some(util::col(ds, &varlist::resolve_name(ds, b.trim())?)?),
            None => None,
        };
        if let Some(b) = &base {
            for i in 0..ds.n_obs {
                if is_missing(b[i]) {
                    mask[i] = false;
                }
            }
        }
        let rows = util::rows(&mask);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let tcol = match &tvar {
            Some(t) => util::col(ds, t)?,
            None => vec![0.0; ds.n_obs],
        };
        let tr: Vec<usize> = rows.iter().copied().filter(|&r| tcol[r] != 0.0).collect();
        let co: Vec<usize> = rows.iter().copied().filter(|&r| tcol[r] == 0.0).collect();
        if co.is_empty() || (tvar.is_some() && tr.is_empty()) {
            return Err(rc_err(2000, "both treated and control observations are required"));
        }
        // targets: moment order per covariate
        let orders: Vec<usize> = match cmd.opt("targets", 3) {
            Some(t) => {
                let v = crate::margins::numlist(t)?;
                if v.len() == 1 {
                    vec![v[0] as usize; vars.len()]
                } else if v.len() == vars.len() {
                    v.iter().map(|x| *x as usize).collect()
                } else {
                    return Err(rc_err(198, "targets() must have one entry or one per covariate"));
                }
            }
            None => vec![1; vars.len()],
        };
        if orders.iter().any(|&o| !(1..=3).contains(&o)) {
            return Err(rc_err(198, "targets() must be 1, 2 or 3"));
        }
        // target moments (treated means of centred powers) and constraint rows
        let mut tmean = Vec::new();
        let mut cons: Vec<(usize, usize, f64)> = Vec::new(); // (var, power, target)
        for (k, c) in cols.iter().enumerate() {
            let mu = match &manual {
                Some(m) => *m.get(k).ok_or_else(|| rc_err(198, "manualtargets() needs one value per covariate"))?,
                None => tr.iter().map(|&r| c[r]).sum::<f64>() / tr.len() as f64,
            };
            tmean.push(mu);
            cons.push((k, 1, mu));
            for p in 2..=orders[k] {
                let tgt = tr.iter().map(|&r| (c[r] - mu).powi(p as i32)).sum::<f64>() / tr.len() as f64;
                cons.push((k, p, tgt));
            }
        }
        let d: Vec<Vec<f64>> = co
            .iter()
            .map(|&r| cons.iter().map(|&(k, p, tgt)| if p == 1 { cols[k][r] - tgt } else { (cols[k][r] - tmean[k]).powi(p as i32) - tgt }).collect())
            .collect();
        let q: Vec<f64> = co.iter().map(|&r| base.as_ref().map_or(1.0, |b| b[r])).collect();
        let mut log = Vec::new();
        let w = solve_weights(&d, &q, tol, maxiter, &mut log)?;
        let total = if tvar.is_some() { tr.len() as f64 } else { co.len() as f64 };
        let mut wv = vec![SYSMISS; ds.n_obs];
        for &r in &tr {
            wv[r] = 1.0;
        }
        for (k, &r) in co.iter().enumerate() {
            wv[r] = w[k] * total;
        }
        let mut out = Vec::new();
        out.push("Data Setup".into());
        if let Some(t) = &tvar {
            out.push(format!("Treatment variable:   {t}"));
        }
        let ordname = |o: usize| match o {
            1 => "1st order",
            2 => "2nd order",
            _ => "3rd order",
        };
        for o in 1..=3 {
            let vs: Vec<&str> = vars.iter().zip(&orders).filter(|(_, k)| **k >= o).map(|(v, _)| v.as_str()).collect();
            if !vs.is_empty() {
                out.push(format!("Covariate adjustment: {} ({})", vs.join(" "), ordname(o)));
            }
        }
        out.push(String::new());
        out.push("Optimizing...".into());
        out.extend(log);
        out.push("maximum difference smaller than the tolerance level; convergence achieved".into());
        out.push(String::new());
        if tvar.is_some() {
            out.push(format!("Treated units: {:>6}     total of weights: {:>9}", tr.len(), fmt_g(tr.len() as f64, 9).trim()));
        }
        out.push(format!("Control units: {:>6}     total of weights: {:>9}", co.len(), fmt_g(total, 9).trim()));
        out.push(String::new());
        let table = |out: &mut Vec<String>, title: &str, wts: &dyn Fn(usize) -> f64| {
            out.push(title.to_string());
            out.push(format!("{:>13}|{:^34}|{:^34}", "", if tvar.is_some() { "Treat" } else { "Target" }, "Control"));
            out.push(format!("{:>13}| {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}", "", "mean", "variance", "skewness", "mean", "variance", "skewness"));
            out.push(format!("{}+{}+{}", "-".repeat(13), "-".repeat(34), "-".repeat(34)));
            for (k, c) in cols.iter().enumerate() {
                let (m0, v0, s0) = moments3(&co.iter().map(|&r| c[r]).collect::<Vec<_>>(), &co.iter().map(|&r| wts(r)).collect::<Vec<_>>());
                let left = if tvar.is_some() {
                    let (m1, v1, s1) = moments3(&tr.iter().map(|&r| c[r]).collect::<Vec<_>>(), &vec![1.0; tr.len()]);
                    format!(" {:>10} {:>10} {:>10} ", fmt_g(m1, 6), fmt_g(v1, 6), fmt_g(s1, 6))
                } else {
                    format!(" {:>10} {:>10} {:>10} ", fmt_g(tmean[k], 6), "", "")
                };
                out.push(format!("{:>12} |{}| {:>10} {:>10} {:>10}", crate::summarize::abbrev(&vars[k], 12), left, fmt_g(m0, 6), fmt_g(v0, 6), fmt_g(s0, 6)));
            }
            out.push(String::new());
        };
        table(&mut out, "Before: without weighting", &|r| base.as_ref().map_or(1.0, |b| b[r]));
        table(&mut out, &format!("After:  {gname} as the weighting variable"), &|r| wv[r]);
        put_var(ds, &gname, wv)?;
        st.data_changed = true;
        let r = st.begin_r();
        r.set("N_treated", tr.len() as f64);
        r.set("N_control", co.len() as f64);
        Ok(out.join("\n"))
    }
}
