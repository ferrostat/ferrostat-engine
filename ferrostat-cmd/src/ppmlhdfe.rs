//! `ppmlhdfe`: Poisson pseudo-maximum likelihood with several
//! high-dimensional fixed effects (Correia, Guimarães & Zylkin 2020).
//!
//! Estimation is iteratively reweighted least squares: each step partials
//! the fixed effects out of the working dependent variable and the
//! regressors with weights mu (using the `hdfe` demeaning code) and solves a
//! weighted least-squares problem. Before estimating, observations whose
//! outcome is perfectly predicted are dropped: fixed-effect groups whose
//! outcome is always zero, singletons, and observations separated by a
//! linear combination of regressors and fixed effects (iterated
//! least-squares on a rectified outcome, the "ReLU" method).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::hdfe::{self, Factor};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};

pub struct Ppmlhdfe;

fn var_codes(ds: &DataSet, name: &str, rows: &[usize]) -> Result<(Vec<u32>, usize), String> {
    let var = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if var.var_type.is_string() {
        let keys: Vec<&str> = rows.iter().map(|&i| var.str_at(i)).collect();
        Ok(hdfe::codes_str(&keys))
    } else {
        let keys: Vec<f64> = rows.iter().map(|&i| var.data[i].num()).collect();
        Ok(hdfe::codes_f64(&keys))
    }
}

fn multi_codes(ds: &DataSet, names: &[String], rows: &[usize]) -> Result<(Vec<u32>, usize), String> {
    let (mut c, mut g) = var_codes(ds, &names[0], rows)?;
    for n in &names[1..] {
        let (c2, g2) = var_codes(ds, n, rows)?;
        let r = hdfe::combine(&c, g, &c2, g2);
        c = r.0;
        g = r.1;
    }
    Ok((c, g))
}

fn wdot(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    a.iter().zip(b).zip(w).map(|((x, y), wi)| x * y * wi).sum()
}

fn wmean(a: &[f64], w: &[f64]) -> f64 {
    wdot(a, &vec![1.0; a.len()], w) / w.iter().sum::<f64>()
}

/// Keep flagged entries of a vector.
fn keep<T: Clone>(v: &[T], k: &[bool]) -> Vec<T> {
    v.iter().zip(k).filter(|(_, k)| **k).map(|(x, _)| x.clone()).collect()
}

/// Weighted least squares of y on x and the fixed effects. Returns the
/// coefficients (zero for dropped columns), the residuals and the
/// partialled-out regressors.
struct Wls {
    b: Vec<f64>,
    resid: Vec<f64>,
    xt: Vec<Vec<f64>>,
    active: Vec<usize>,

}

fn wls(factors: &mut [Factor], y: &[f64], x: &[Vec<f64>], w: &[f64], tol: f64, droppable: &mut [bool]) -> Wls {
    for f in factors.iter_mut() {
        f.prepare(Some(w));
    }
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(x.len() + 1);
    cols.push(y.to_vec());
    cols.extend(x.iter().cloned());
    {
        let mut refs: Vec<&mut Vec<f64>> = cols.iter_mut().collect();
        hdfe::partial_out_all(factors, &mut refs, Some(w), tol, 10_000);
    }
    let yt = cols.remove(0);
    // drop regressors explained by the fixed effects or other regressors
    let mut active = Vec::new();
    for (j, c) in cols.iter().enumerate() {
        let ss = wdot(c, c, w);
        let ss0 = wdot(&x[j], &x[j], w);
        if !droppable[j] && ss > 1e-9 * ss0.max(1e-300) {
            active.push(j);
        } else {
            droppable[j] = true;
        }
    }
    let k = active.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for a in 0..k {
        xty[a] = wdot(&cols[active[a]], &yt, w);
        for c in 0..=a {
            let v = wdot(&cols[active[a]], &cols[active[c]], w);
            xtx[(a, c)] = v;
            xtx[(c, a)] = v;
        }
    }
    let (inv, dropped) = xtx.sym_inverse(1e-10);
    for &dj in &dropped {
        droppable[active[dj]] = true;
    }
    let mut b = vec![0.0; x.len()];
    for a in 0..k {
        b[active[a]] = (0..k).map(|c| inv[(a, c)] * xty[c]).sum();
    }
    let mut resid = yt;
    for &j in &active {
        if b[j] != 0.0 {
            for (r, xv) in resid.iter_mut().zip(&cols[j]) {
                *r -= b[j] * xv;
            }
        }
    }
    Wls { b, resid, xt: cols, active }
}

/// Drop observations in fixed-effect groups whose outcome is always zero,
/// then singletons, until nothing changes. Returns the number dropped.
fn fe_separation(factors: &[Factor], y: &[f64], keep_mask: &mut [bool], singletons: bool) -> (usize, usize) {
    let n = y.len();
    let (mut n_sep, mut n_single) = (0, 0);
    loop {
        let mut changed = false;
        for f in factors {
            let mut sum = vec![0.0; f.ng];
            let mut cnt = vec![0usize; f.ng];
            for i in 0..n {
                if keep_mask[i] {
                    sum[f.ids[i] as usize] += y[i];
                    cnt[f.ids[i] as usize] += 1;
                }
            }
            for i in 0..n {
                if !keep_mask[i] {
                    continue;
                }
                let g = f.ids[i] as usize;
                if sum[g] == 0.0 {
                    keep_mask[i] = false;
                    n_sep += 1;
                    changed = true;
                } else if singletons && cnt[g] == 1 {
                    keep_mask[i] = false;
                    n_single += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (n_sep, n_single)
}

fn subset_factors(factors: &mut [Factor], k: &[bool]) {
    for f in factors.iter_mut() {
        f.subset(k);
    }
}

/// Observations separated by regressors and fixed effects (ReLU method).
fn relu_separation(factors: &mut [Factor], y: &[f64], x: &[Vec<f64>], tol: f64) -> Vec<bool> {
    let n = y.len();
    let boundary: Vec<bool> = y.iter().map(|v| *v == 0.0).collect();
    let mut sep = vec![false; n];
    if !boundary.iter().any(|b| *b) {
        return sep;
    }
    let big = 1e6;
    let w: Vec<f64> = boundary.iter().map(|b| if *b { 1.0 } else { big }).collect();
    let mut u: Vec<f64> = boundary.iter().map(|b| *b as u8 as f64).collect();
    let delta = tol.max(1e-8);
    let mut drop = vec![false; x.len()];
    for _ in 0..200 {
        let mut d2 = drop.clone();
        let fit = wls(factors, &u, x, &w, 1e-12, &mut d2);
        let mut xbd: Vec<f64> = u.iter().zip(&fit.resid).map(|(a, r)| a - r).collect();
        for i in 0..n {
            if !boundary[i] || (xbd[i] >= -0.1 * delta && xbd[i] <= delta) {
                xbd[i] = 0.0;
            }
        }
        if (0..n).filter(|&i| boundary[i]).all(|i| xbd[i] >= 0.0) {
            for i in 0..n {
                sep[i] = boundary[i] && xbd[i] > 0.0;
            }
            return sep;
        }
        if (0..n).filter(|&i| boundary[i]).all(|i| fit.resid[i] >= -1e-12) {
            for i in 0..n {
                sep[i] = boundary[i] && xbd[i] > 0.0 && fit.resid[i] <= delta;
            }
            return sep;
        }
        for i in 0..n {
            u[i] = if boundary[i] { xbd[i].max(0.0) } else { 0.0 };
        }
        drop = vec![false; x.len()];
    }
    sep
}

fn f4(v: f64) -> String {
    if is_missing(v) || !v.is_finite() { ".".into() } else { format!("{v:.4}") }
}

impl CommandHandler for Ppmlhdfe {
    fn name(&self) -> &'static str {
        "ppmlhdfe"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "ppmlhdfe");
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let noabsorb = cmd.has_opt("noabsorb", 3);
        let absorb = cmd.opt("absorb", 1).map(|a| a.split(',').next().unwrap_or("").trim().to_string());
        if absorb.is_none() && !noabsorb {
            return Err(rc_err(198, "option absorb() required (or noabsorb)"));
        }
        let mut fe_specs: Vec<(Option<String>, Vec<String>)> = Vec::new();
        for tok in absorb.as_deref().unwrap_or("").split_whitespace() {
            let (save, body) = match tok.split_once('=') {
                Some((n, b)) => (Some(n.to_string()), b),
                None => (None, tok),
            };
            if body.contains("c.") {
                return Err(rc_err(198, format!("{tok}: heterogeneous slopes are not supported in absorb() here")));
            }
            let vars: Vec<String> = body.split('#').filter(|p| !p.is_empty()).map(|p| crate::varlist::resolve_name(ds, p.trim_start_matches("i."))).collect::<Result<_, _>>()?;
            fe_specs.push((save, vars));
        }
        // vce: robust by default
        let (cluster, vce_name) = match cmd.opt("vce", 3).map(|s| s.trim().to_string()) {
            None => match cmd.opt("cluster", 2) {
                Some(c) => (Some(c.trim().to_string()), "cluster"),
                None => (None, "robust"),
            },
            Some(v) => {
                let (first, rest) = v.split_once(char::is_whitespace).unwrap_or((&v, ""));
                match first.to_lowercase().as_str() {
                    "robust" | "r" => (None, "robust"),
                    "cluster" | "cl" | "clu" | "clus" => {
                        if rest.trim().is_empty() {
                            return Err(rc_err(198, "vce(cluster clustvar) requires a variable"));
                        }
                        (Some(rest.trim().to_string()), "cluster")
                    }
                    o => return Err(rc_err(198, format!("vce({o}) not allowed"))),
                }
            }
        };
        let cluster = cluster.map(|c| crate::varlist::resolve_name(ds, &c)).transpose()?;
        let exposure = cmd.opt("exposure", 3).map(|s| crate::varlist::resolve_name(ds, s.trim())).transpose()?;
        let offset = cmd.opt("offset", 3).map(|s| crate::varlist::resolve_name(ds, s.trim())).transpose()?;
        if exposure.is_some() && offset.is_some() {
            return Err(rc_err(198, "only one of offset() or exposure() can be specified"));
        }
        let d_name = cmd.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case("d")).map(|(_, a)| a.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("_ppmlhdfe_d").to_string());
        let tol = util::opt_num(cmd, "tolerance", 3)?.unwrap_or(1e-8);
        let maxiter = util::opt_num(cmd, "maxiterations", 4)?.unwrap_or(1000.0) as usize;
        let level = util::level(cmd, st)?;
        let mut extra: Vec<String> = fe_specs.iter().flat_map(|s| s.1.clone()).collect();
        extra.extend(cluster.iter().cloned());
        extra.extend(exposure.iter().cloned());
        extra.extend(offset.iter().cloned());
        let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: indep.trim(), noconstant: false, weights: "fpa", extra })?;
        if d.y.iter().any(|v| *v < 0.0) {
            return Err(rc_err(459, format!("{dep} must be greater than or equal to zero")));
        }
        let n_full = d.n();
        let cons = d.k() - 1;
        let off_all: Option<Vec<f64>> = match (&exposure, &offset) {
            (Some(e), _) => {
                let v = util::col(ds, e)?;
                if d.rows.iter().any(|&i| v[i] <= 0.0) {
                    return Err(rc_err(459, "exposure() must be greater than zero"));
                }
                Some(d.rows.iter().map(|&i| v[i].ln()).collect())
            }
            (None, Some(o)) => {
                let v = util::col(ds, o)?;
                Some(d.rows.iter().map(|&i| v[i]).collect())
            }
            _ => None,
        };
        let mut factors: Vec<Factor> = Vec::new();
        for (save, vars) in &fe_specs {
            let (ids, ng) = multi_codes(ds, vars, &d.rows)?;
            let mut f = Factor::new(vars.join("#"), ids, ng, true, vec![], vec![]);
            f.save_as = save.clone();
            factors.push(f);
        }
        if factors.is_empty() {
            factors.push(Factor::new("_cons".into(), vec![0; d.n()], 1, true, vec![], vec![]));
        }
        // ---- separation -----------------------------------------------------
        let n0 = d.n();
        let mut mask = vec![true; n0];
        let singletons = !cmd.has_opt("keepsingletons", 5) && !noabsorb;
        let (mut n_fe_sep, mut n_single) = fe_separation(&factors, &d.y, &mut mask, singletons);
        let mut n_relu = 0;
        let mut cur: Vec<usize> = (0..n0).collect();
        let sub_all = |mask: &[bool], factors: &mut Vec<Factor>, cur: &mut Vec<usize>| {
            subset_factors(factors, mask);
            *cur = keep(cur, mask);
        };
        sub_all(&mask, &mut factors, &mut cur);
        let xcols_idx: Vec<usize> = (0..d.k()).filter(|&j| !d.omitted[j] && j != cons).collect();
        let sep_methods = cmd.opt("separation", 3).map(|s| s.to_lowercase()).unwrap_or_else(|| "fe simplex relu".into());
        if sep_methods.contains("relu") || sep_methods.contains("simplex") {
            loop {
                let y: Vec<f64> = cur.iter().map(|&i| d.y[i]).collect();
                let x: Vec<Vec<f64>> = xcols_idx.iter().map(|&j| cur.iter().map(|&i| d.cols[j][i]).collect()).collect();
                let sep = relu_separation(&mut factors, &y, &x, 1e-5);
                let ns = sep.iter().filter(|s| **s).count();
                if ns == 0 {
                    break;
                }
                n_relu += ns;
                let k: Vec<bool> = sep.iter().map(|s| !s).collect();
                sub_all(&k, &mut factors, &mut cur);
                let y2: Vec<f64> = cur.iter().map(|&i| d.y[i]).collect();
                let mut m2 = vec![true; cur.len()];
                let (a, b) = fe_separation(&factors, &y2, &mut m2, singletons);
                n_fe_sep += a;
                n_single += b;
                sub_all(&m2, &mut factors, &mut cur);
            }
        }
        if cur.is_empty() {
            return Err(rc_err(2000, "no observations left after dropping separated observations"));
        }
        // restrict the design
        let mut kmask = vec![false; n0];
        for &i in &cur {
            kmask[i] = true;
        }
        for (t, &k) in kmask.iter().enumerate() {
            if !k {
                d.mask[d.rows[t]] = false;
            }
        }
        d.rows = keep(&d.rows, &kmask);
        d.y = keep(&d.y, &kmask);
        d.w = keep(&d.w, &kmask);
        for c in d.cols.iter_mut() {
            *c = keep(c, &kmask);
        }
        let off: Vec<f64> = off_all.as_ref().map(|o| keep(o, &kmask)).unwrap_or_else(|| vec![0.0; d.n()]);
        let n = d.n();
        let tw: Vec<f64> = if d.wkind.is_some() { d.w.clone() } else { vec![1.0; n] };
        let y = d.y.clone();
        if y.iter().all(|v| *v == 0.0) {
            return Err(rc_err(2000, "dependent variable is zero for all observations"));
        }
        let x: Vec<Vec<f64>> = xcols_idx.iter().map(|&j| d.cols[j].clone()).collect();
        // ---- IRLS -------------------------------------------------------------
        let ybar = wmean(&y, &tw);
        let mut mu: Vec<f64> = y.iter().map(|v| (v + ybar) / 2.0).collect();
        let mut eta: Vec<f64> = mu.iter().map(|m| m.ln()).collect();
        let mut dev_old = f64::NAN;
        let mut iter = 0;
        let mut log = Vec::new();
        let mut converged = false;
        let mut droppable = vec![false; x.len()];
        let mut last: Option<Wls> = None;
        let mut deviance = f64::NAN;
        let mut z = vec![0.0; n];
        while iter < maxiter {
            iter += 1;
            for i in 0..n {
                z[i] = eta[i] - off[i] + (y[i] - mu[i]) / mu[i];
            }
            let w: Vec<f64> = mu.iter().zip(&tw).map(|(m, t)| m * t).collect();
            let inner_tol = (tol / 10.0).clamp(1e-14, 1e-8);
            let fit = wls(&mut factors, &z, &x, &w, inner_tol, &mut droppable);
            for i in 0..n {
                eta[i] = z[i] - fit.resid[i] + off[i];
                mu[i] = eta[i].exp().max(1e-300);
            }
            deviance = 2.0 * (0..n).map(|i| tw[i] * (if y[i] > 0.0 { y[i] * (y[i].ln() - eta[i]) } else { 0.0 } - (y[i] - mu[i]))).sum::<f64>();
            if deviance < 0.0 {
                deviance = 0.0;
            }
            let eps = if dev_old.is_nan() { f64::NAN } else { (dev_old - deviance).abs() / dev_old.min(deviance).max(0.1) };
            log.push(format!("Iteration {iter}:   deviance = {:<11}  eps = {:<9}  iters = 1", crate::format::fmt(deviance, "%11.5e").trim(), if eps.is_nan() { ".".to_string() } else { crate::format::fmt(eps, "%9.4e").trim().to_string() }));
            last = Some(fit);
            if !eps.is_nan() && eps < tol {
                converged = true;
                break;
            }
            dev_old = deviance;
        }
        if !converged {
            return Err(rc_err(430, format!("Failed to converge in {maxiter} iterations")));
        }
        let fit = last.unwrap();
        // ---- inference ------------------------------------------------------
        let w: Vec<f64> = mu.iter().zip(&tw).map(|(m, t)| m * t).collect();
        let act = fit.active.clone();
        let k = act.len();
        let resid_raw: Vec<f64> = (0..n).map(|i| y[i] - mu[i]).collect();
        // score weights: (y - mu) times frequency/sampling weights
        let fw = d.wkind == Some(WeightKind::F);
        let score: Vec<f64> = (0..n).map(|i| resid_raw[i] * tw[i]).collect();
        let n_eff = if fw { tw.iter().sum::<f64>() } else { n as f64 };
        let clus: Option<(Vec<u32>, usize)> = match &cluster {
            Some(c) => Some(var_codes(ds, c, &d.rows)?),
            None => None,
        };
        // add back means and a constant (reported _cons)
        let xbar: Vec<f64> = act.iter().map(|&j| wmean(&x[j], &w)).collect();
        // fit.xt holds the partialled regressors (index j)
        let mut zc: Vec<Vec<f64>> = act.iter().enumerate().map(|(a, &j)| fit.xt[j].iter().map(|v| v + xbar[a]).collect()).collect();
        zc.push(vec![1.0; n]);
        let kc = k + 1;
        let mut xtx = Matrix::zeros(kc, kc);
        for a in 0..kc {
            for c in 0..=a {
                let v = wdot(&zc[a], &zc[c], &w);
                xtx[(a, c)] = v;
                xtx[(c, a)] = v;
            }
        }
        let (bread, _) = xtx.sym_inverse(1e-12);
        let mut meat = Matrix::zeros(kc, kc);
        let mut n_clust = 0usize;
        match &clus {
            None => {
                for i in 0..n {
                    let s2 = if fw { tw[i] * resid_raw[i] * resid_raw[i] } else { score[i] * score[i] };
                    for a in 0..kc {
                        for c in 0..kc {
                            meat[(a, c)] += zc[a][i] * zc[c][i] * s2;
                        }
                    }
                }
                meat = meat.scale(n_eff / (n_eff - 1.0));
            }
            Some((ids, g)) => {
                let mut sums = vec![vec![0.0; kc]; *g];
                for i in 0..n {
                    for a in 0..kc {
                        sums[ids[i] as usize][a] += zc[a][i] * score[i];
                    }
                }
                let used: Vec<&Vec<f64>> = sums.iter().filter(|s| s.iter().any(|v| *v != 0.0)).collect();
                n_clust = (0..*g).filter(|&gg| ids.contains(&(gg as u32))).count();
                for s in used {
                    for a in 0..kc {
                        for c in 0..kc {
                            meat[(a, c)] += s[a] * s[c];
                        }
                    }
                }
                let gf = n_clust as f64;
                meat = meat.scale(gf / (gf - 1.0));
            }
        }
        let vfull = bread.mul(&meat).unwrap().mul(&bread).unwrap();
        // coefficients: slopes and the constant
        let mut b = vec![0.0; d.k()];
        let mut actfull = Vec::new();
        for (a, &j) in act.iter().enumerate() {
            b[xcols_idx[j]] = fit.b[j];
            actfull.push(xcols_idx[j]);
            let _ = a;
        }
        let eta_bar = wmean(&(0..n).map(|i| eta[i] - off[i]).collect::<Vec<_>>(), &w);
        let xb_bar: f64 = act.iter().enumerate().map(|(a, &j)| xbar[a] * fit.b[j]).sum();
        b[cons] = eta_bar - xb_bar;
        actfull.push(cons);
        let mut omitted = d.omitted.clone();
        let mut names = d.names.clone();
        let mut notes = d.notes.clone();
        for (j, &dj) in droppable.iter().enumerate() {
            if dj {
                let col = xcols_idx[j];
                omitted[col] = true;
                notes.push(format!("note: {} omitted because of collinearity", names[col]));
                names[col] = match names[col].split_once('.') {
                    Some((a, rest)) if a.chars().all(|c| c.is_ascii_digit()) => format!("{a}o.{rest}"),
                    _ => format!("o.{}", names[col]),
                };
            }
        }
        let v = est::expand_v(&vfull, &actfull, d.k());
        let df_m = k as f64;
        let chi2 = {
            let idx: Vec<usize> = actfull.iter().copied().filter(|&j| j != cons).collect();
            if idx.is_empty() {
                SYSMISS
            } else {
                let mut vs = Matrix::zeros(idx.len(), idx.len());
                for (a, &i) in idx.iter().enumerate() {
                    for (c, &j) in idx.iter().enumerate() {
                        vs[(a, c)] = v[(i, j)];
                    }
                }
                let (vi, _) = vs.sym_inverse(1e-14);
                (0..idx.len()).map(|a| (0..idx.len()).map(|c| b[idx[a]] * vi[(a, c)] * b[idx[c]]).sum::<f64>()).sum()
            }
        };
        // absorbed degrees of freedom
        let mut df_a = 0usize;
        if !noabsorb {
            for (kf, f) in factors.iter().enumerate() {
                let nested = clus.as_ref().is_some_and(|(c, _)| hdfe::nested(&f.ids, f.ng, c));
                if nested {
                    continue;
                }
                df_a += match kf {
                    0 => f.ng,
                    1 => f.ng - hdfe::components(&factors[0].ids, factors[0].ng, &f.ids, f.ng),
                    _ => f.ng.saturating_sub(1),
                };
            }
        } else {
            df_a = 1;
        }
        let df_r = n_eff - df_a as f64 - df_m;
        let lgy: Vec<f64> = y.iter().map(|v| dist::lngamma(v + 1.0)).collect();
        let ll: f64 = (0..n).map(|i| tw[i] * (if y[i] > 0.0 { y[i] * eta[i] } else { 0.0 } - mu[i] - lgy[i])).sum();
        let ybar2 = wmean(&y, &tw);
        let ll_0: f64 = (0..n).map(|i| tw[i] * (y[i] * ybar2.ln() - ybar2 - lgy[i])).sum();
        let r2_p = 1.0 - ll / ll_0;
        // ---- output ---------------------------------------------------------
        let mut out: Vec<String> = Vec::new();
        if n_fe_sep + n_single > 0 {
            out.push(format!("(dropped {} observations that are either singletons or separated by a fixed effect)", n_fe_sep + n_single));
        }
        if n_relu > 0 {
            out.push(format!("(ReLU method dropped {n_relu} separated observation{})", if n_relu == 1 { "" } else { "s" }));
        }
        out.extend(notes.iter().cloned());
        out.extend(log);
        out.push(format!("Converged in {iter} iterations (tol = {})", crate::format::fmt(tol, "%4.0e").trim()));
        out.push(String::new());
        let nabs = if noabsorb { 0 } else { fe_specs.len() };
        let title2 = format!("Absorbing {} HDFE group{}", nabs, if nabs == 1 { "" } else { "s" });
        let title3 = if clus.is_some() { "Statistics robust to heteroskedasticity" } else { "Statistics robust to heteroskedasticity" };
        let line = |l: &str, lab: &str, val: String| format!("{:<50}{:<16}= {:>10}", l, lab, val);
        out.push(line("HDFE PPML regression", "No. of obs", util::commas(n_eff as usize)));
        out.push(line(&title2, "Residual df", util::commas(df_r.max(0.0) as usize)));
        out.push(line(title3, &format!("Wald chi2({})", df_m), format!("{chi2:.2}")));
        out.push(format!("{:<21}= {:>12}{:15}{:<16}= {:>10}", "Deviance", crate::format::fmt_g(deviance, 12).trim(), "", "Prob > chi2", f4(dist::chi2tail(df_m, chi2))));
        out.push(format!("{:<21}= {:>12}{:15}{:<16}= {:>10}", "Log pseudolikelihood", crate::format::fmt_g(ll, 12).trim(), "", "Pseudo R2", f4(r2_p)));
        if let (Some(c), true) = (&cluster, clus.is_some()) {
            out.push(String::new());
            out.push(format!("{:<28}= {:>10}", format!("Number of clusters ({c})"), util::commas(n_clust)));
        }
        let show_k = d.k();
        let mut table = est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &names,
            b: &b[..show_k],
            v: &v,
            omitted: &omitted,
            is_base: &d.is_base,
            df: None,
            level,
            eform: if cmd.has_opt("irr", 3) { Some("IRR") } else { None },
            coef_title: "Coefficient",
        });
        if table.len() >= 2 {
            let lw = table[1].find(" | ").unwrap_or(12);
            table[1] = table[1].replacen("Std. err.", "std. err.", 1);
            table.insert(1, format!("{:>lw$} |               Robust", ""));
        }
        out.push(String::new());
        out.extend(table);
        if !noabsorb {
            out.push(String::new());
            out.push("Absorbed degrees of freedom:".into());
            out.push(format!("{}+", "-".repeat(53)));
            out.push(" Absorbed FE | Categories  - Redundant  = Num. Coefs |".into());
            out.push(format!("{}+{}|", "-".repeat(13), "-".repeat(39)));
            for (kf, f) in factors.iter().enumerate() {
                let nested = clus.as_ref().is_some_and(|(c, _)| hdfe::nested(&f.ids, f.ng, c));
                let red = if nested {
                    f.ng
                } else {
                    match kf {
                        0 => 0,
                        1 => hdfe::components(&factors[0].ids, factors[0].ng, &f.ids, f.ng),
                        _ => 1.min(f.ng),
                    }
                };
                let mark = if nested { '*' } else if kf >= 2 { '?' } else { ' ' };
                out.push(format!("{:>12} |{:>10}{:>12}{:>12}    {}|", crate::summarize::abbrev(&f.label, 12), f.ng, red, f.ng - red, mark));
            }
            out.push(format!("{}+", "-".repeat(53)));
        }
        let text = out.join("\n");
        // ---- saved variables --------------------------------------------------
        let xb: Vec<f64> = (0..n).map(|i| act.iter().map(|&j| fit.b[j] * x[j][i]).sum::<f64>()).collect();
        let dsum: Vec<f64> = (0..n).map(|i| eta[i] - off[i] - xb[i]).collect();
        let dmean = wmean(&dsum, &w);
        if let Some(dn) = &d_name {
            if ds.var_index.contains_key(dn.as_str()) {
                if dn == "_ppmlhdfe_d" {
                    ds.drop_variable(dn)?;
                } else {
                    return Err(rc_err(110, format!("variable {dn} already defined")));
                }
            }
            let mut all = vec![SYSMISS; ds.n_obs];
            for (t, &r) in d.rows.iter().enumerate() {
                all[r] = dsum[t] - dmean;
            }
            let mut var = util::new_double(dn, all);
            var.label = "Sum of fixed effects".into();
            ds.add_variable(var)?;
            st.data_changed = true;
        }
        if factors.iter().any(|f| f.save_as.is_some()) {
            let centered: Vec<f64> = dsum.iter().map(|v| v - dmean).collect();
            let parts = hdfe::recover(&factors, &centered, Some(&w), 1e-10, 10_000);
            for (f, comp) in factors.iter().zip(&parts) {
                let Some(name) = &f.save_as else { continue };
                if ds.var_index.contains_key(name.as_str()) {
                    return Err(rc_err(110, format!("variable {name} already defined")));
                }
                let mut all = vec![SYSMISS; ds.n_obs];
                for (t, &r) in d.rows.iter().enumerate() {
                    all[r] = comp[0][t];
                }
                let mut var = util::new_double(name, all);
                var.label = format!("[FE] {}", f.label);
                ds.add_variable(var)?;
                st.data_changed = true;
            }
        }
        // ---- e() ------------------------------------------------------------
        let absvars: Vec<String> = fe_specs.iter().map(|s| s.1.join("#")).collect();
        let indepvars: Vec<String> = names.iter().filter(|nm| *nm != "_cons").cloned().collect();
        let cmdline = format!("ppmlhdfe {}", cmd.raw.trim().strip_prefix("ppmlhdfe").unwrap_or(cmd.raw.trim()).trim());
        let mut dpost = d.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted.clone();
        let wkind = d.wkind;
        est::post(st, "ppmlhdfe", &dpost, &b, &v, &|e: &mut Results| {
            e.set("N", n_eff);
            e.set("N_full", n_full as f64);
            e.set("df", df_r);
            e.set("df_r", df_r);
            e.set("df_m", df_m);
            e.set("df_a", df_a as f64);
            e.set("chi2", est::missing_or(chi2));
            e.set("p", est::missing_or(dist::chi2tail(df_m, chi2)));
            e.set("ll", ll);
            e.set("ll_0", ll_0);
            e.set("r2_p", r2_p);
            e.set("deviance", deviance);
            e.set("ic", iter as f64);
            e.set("converged", 1.0);
            e.set("num_singletons", n_single as f64);
            e.set("num_separated", (n_fe_sep + n_relu) as f64);
            e.set("N_hdfe", nabs as f64);
            e.set("rank", (k + 1) as f64);
            e.set_str("title", "HDFE PPML regression");
            e.set_str("title2", title2.clone());
            e.set_str("absvars", if noabsorb { "_cons".to_string() } else { absvars.join(" ") });
            e.set_str("indepvars", indepvars.join(" "));
            e.set_str("predict", "ppmlhdfe_p");
            e.set_str("chi2type", "Wald");
            e.set_str("cmdline", cmdline.clone());
            e.set_str("vce", vce_name);
            e.set_str("vcetype", "Robust");
            if let Some(c) = &cluster {
                e.set_str("clustvar", c.clone());
                e.set_str("clustvar1", c.clone());
                e.set("N_clust", n_clust as f64);
                e.set("N_clust1", n_clust as f64);
                e.set("N_clustervars", 1.0);
            }
            if let Some(o) = &offset {
                e.set_str("offset", o.clone());
            }
            if let Some(x) = &exposure {
                e.set_str("offset", format!("ln({x})"));
            }
            if let Some(dn) = &d_name {
                e.set_str("d", dn.clone());
            }
            if let Some(k) = wkind {
                e.set_str("wtype", match k { WeightKind::F => "fweight", WeightKind::A => "aweight", WeightKind::P => "pweight", _ => "iweight" });
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
