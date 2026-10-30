//! Inference on variables of interest with lasso-selected controls:
//!
//! * double selection (`dsregress`, `dslogit`, `dspoisson`): lassos of the
//!   outcome and of each variable of interest on the controls; the final
//!   model uses the union of the selected controls (Belloni, Chernozhukov
//!   & Hansen 2014);
//! * partialling-out (`poregress`, `pologit`, `popoisson`, `poivregress`):
//!   outcome and variables of interest are residualized on their selected
//!   controls (post-lasso fits) and the effect is estimated from the
//!   residuals; for logit and Poisson outcomes the variables of interest
//!   are residualized by weighted lassos and the effect solves the
//!   orthogonal score sum_i z_i (y_i - m(d_i'a + x_i'b)) = 0
//!   (Belloni, Chernozhukov & Wei 2016);
//! * cross-fit partialling-out (`xporegress`, ...): the same with the
//!   nuisance fits estimated on all folds but one and evaluated on the
//!   held-out fold (double machine learning, Chernozhukov et al. 2018),
//!   optionally repeated over `resample(#)` fold splits (estimates
//!   averaged, with the spread across splits added to the variance).
//!
//! `poivregress` partials the controls out of the outcome, the exogenous
//! and endogenous variables of interest and the instruments, then runs
//! two-stage least squares on the residuals. Standard errors are
//! heteroskedasticity-robust (or cluster-robust with vce(cluster)).

use std::collections::BTreeMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::core::{self, Family, Problem};
use super::fit::{self, Lasso, Sel, Settings};
use crate::commands::CommandHandler;
use crate::est::{self, Vce};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::teffects::{invert, solve};
use crate::util;
use crate::xmodels as xm;

pub const VERBS: [&str; 11] = ["dsregress", "dslogit", "dspoisson", "poregress", "pologit", "popoisson", "poivregress", "xporegress", "xpologit", "xpopoisson", "xpoivregress"];

pub struct Inf {
    pub verb: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Ds,
    Po,
    Xpo,
}

fn method(verb: &str) -> Method {
    if verb.starts_with("ds") {
        Method::Ds
    } else if verb.starts_with("xpo") {
        Method::Xpo
    } else {
        Method::Po
    }
}

fn family(verb: &str) -> Family {
    if verb.ends_with("logit") {
        Family::Logit
    } else if verb.ends_with("poisson") {
        Family::Poisson
    } else {
        Family::Linear
    }
}

/// Everything needed for one fit.
struct Inputs {
    dep: String,
    y: Vec<f64>,
    /// controls
    x: Vec<Vec<f64>>,
    xnames: Vec<String>,
    xalways: Vec<bool>,
    /// variables of interest (exogenous first, then endogenous)
    d: Vec<Vec<f64>>,
    dnames: Vec<String>,
    n_endog: usize,
    /// instruments (poivregress)
    z: Vec<Vec<f64>>,
    znames: Vec<String>,
    mask: Vec<bool>,
    cluster: Option<Vec<f64>>,
}

fn design_cols(ds: &DataSet, st: &Store, cmd: &Command, text: &str, extra: &[String], rows: &[usize]) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    if text.trim().is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let d = est::build(ds, st, cmd, est::DesignSpec { depvar: None, indep: text, noconstant: true, weights: "", extra: extra.to_vec() })?;
    if d.rows != rows {
        return Err(rc_err(459, "variables of interest and controls have different estimation samples"));
    }
    let mut names = Vec::new();
    let mut cols = Vec::new();
    for j in 0..d.k() {
        if d.is_base[j] {
            continue;
        }
        names.push(d.names[j].clone());
        cols.push(d.cols[j].clone());
    }
    Ok((names, cols))
}

fn inputs(ds: &DataSet, st: &Store, cmd: &Command, verb: &str) -> Result<Inputs, String> {
    let main = cmd.main.trim();
    let iv = verb.ends_with("ivregress");
    let (dep, exo, endo, inst) = if iv {
        let groups = xm::paren_groups(main);
        let outside = xm::outside_parens(main);
        let (dep, exo) = xm::first_word(&outside);
        let g = groups.first().ok_or_else(|| rc_err(198, "(endogenous = instruments) required"))?;
        let (a, b) = g.split_once('=').ok_or_else(|| rc_err(198, "(endogenous = instruments) required"))?;
        (dep, exo, a.trim().to_string(), b.trim().to_string())
    } else {
        let (dep, rest) = xm::first_word(main);
        (dep, rest, String::new(), String::new())
    };
    if dep.is_empty() || (exo.trim().is_empty() && endo.trim().is_empty()) {
        return Err(rc_err(100, "depvar and variables of interest required"));
    }
    let dep = xm::var_name(ds, &dep)?;
    let ctl = cmd.opt("controls", 4).ok_or_else(|| rc_err(198, "option controls() required"))?;
    let (always, others) = (xm::paren_groups(ctl).join(" "), xm::outside_parens(ctl));
    let mut all: Vec<String> = Vec::new();
    for t in [&exo, &endo, &inst, &always, &others] {
        all.extend(crate::zip::base_vars(ds, t));
    }
    let vce = est::parse_vce(cmd, Vce::Robust)?;
    if let Vce::Cluster(c) = &vce {
        all.push(c.clone());
    }
    let data = fit::build(ds, st, cmd, &dep, &always, &others, &all)?;
    let (mut dnames, mut d) = design_cols(ds, st, cmd, &exo, &all, &data.rows)?;
    let (en, ec) = design_cols(ds, st, cmd, &endo, &all, &data.rows)?;
    let n_endog = en.len();
    dnames.extend(en);
    d.extend(ec);
    let (znames, z) = design_cols(ds, st, cmd, &inst, &all, &data.rows)?;
    if iv && znames.len() < n_endog {
        return Err(rc_err(481, "equation not identified; must have at least as many instruments as endogenous variables"));
    }
    let cluster = match &vce {
        Vce::Cluster(c) => {
            let v = util::col(ds, c)?;
            Some(data.rows.iter().map(|&r| v[r]).collect())
        }
        _ => None,
    };
    let fam = family(verb);
    let y = fit::outcome(fam, &data.y, &dep)?;
    Ok(Inputs { dep, y, x: data.x, xnames: data.names, xalways: data.always, d, dnames, n_endog, z, znames, mask: data.mask, cluster })
}

/// Lasso selection context: settings, generator and stored lassos.
struct Ctx<'a> {
    s: Settings,
    st: &'a Store,
    lassos: Vec<(String, Lasso)>,
    fixed: BTreeMap<String, Lasso>,
}

impl Ctx<'_> {
    /// Lasso of `y` on the controls (plus `extra` always-included
    /// columns placed first) over rows flagged in `train`; returns the
    /// indexes of the selected controls (always-included ones included).
    #[allow(clippy::too_many_arguments)]
    fn select(&mut self, label: &str, fam: Family, inp: &Inputs, y: &[f64], w: Option<&[f64]>, extra: &[(String, Vec<f64>)], train: &[bool]) -> Result<Vec<usize>, String> {
        let ne = extra.len();
        let l = if let Some(l) = self.fixed.get(label) {
            l.clone()
        } else {
            let pick = |v: &[f64]| -> Vec<f64> { v.iter().zip(train).filter(|(_, t)| **t).map(|(a, _)| *a).collect() };
            let mut x: Vec<Vec<f64>> = extra.iter().map(|(_, c)| pick(c)).collect();
            x.extend(inp.x.iter().map(|c| pick(c)));
            let mut names: Vec<String> = extra.iter().map(|(n, _)| n.clone()).collect();
            names.extend(inp.xnames.iter().cloned());
            let mut always = vec![true; ne];
            always.extend(inp.xalways.iter().copied());
            let yy = pick(y);
            let ww = match w {
                Some(w) => pick(w),
                None => vec![1.0; yy.len()],
            };
            let p = Problem { family: fam, x, y: yy, w: ww, offset: None, loadings: always.iter().map(|a| if *a { 0.0 } else { 1.0 }).collect(), alpha: 1.0, standardize: self.s.standardize };
            let mut s = self.s.clone();
            s.sqrt = s.sqrt && fam == Family::Linear;
            let mut rng = self.st.rng.borrow_mut();
            let mut u = || rng.uniform();
            fit::run(if s.sqrt { "sqrtlasso" } else { "lasso" }, label.split('#').next().unwrap_or(label), &names, &always, &p, &s, &mut u)?
        };
        let k = l.sel.ok_or_else(|| rc_err(498, format!("no lambda selected for the lasso of {label}")))?;
        let mut sel: Vec<usize> = (ne..l.names.len()).filter(|&j| l.pts[k].b[j] != 0.0 || l.always[j]).map(|j| j - ne).collect();
        sel.sort_unstable();
        self.lassos.push((label.to_string(), l));
        Ok(sel)
    }
}

/// Weighted least squares of `y` on `cols` (+ constant) over `train`;
/// returns the coefficients (constant last).
fn wls(cols: &[&[f64]], y: &[f64], w: Option<&[f64]>, train: &[bool]) -> Vec<f64> {
    let k = cols.len() + 1;
    let mut a = vec![vec![0.0; k]; k];
    let mut b = vec![0.0; k];
    for i in 0..y.len() {
        if !train[i] {
            continue;
        }
        let wi = w.map_or(1.0, |w| w[i]);
        let row: Vec<f64> = cols.iter().map(|c| c[i]).chain(std::iter::once(1.0)).collect();
        for r in 0..k {
            b[r] += wi * row[r] * y[i];
            for c in 0..k {
                a[r][c] += wi * row[r] * row[c];
            }
        }
    }
    // tiny ridge guards against collinear selections
    let tr: f64 = (0..k).map(|r| a[r][r]).sum::<f64>() / k as f64;
    for r in 0..k.saturating_sub(1) {
        a[r][r] += 1e-10 * tr.max(1e-300);
    }
    solve(&a, &b).unwrap_or_else(|| vec![0.0; k])
}

/// Residuals of `v` after a post-lasso linear fit on the selected
/// controls, estimated over `train` and evaluated on all rows.
fn residual(inp: &Inputs, v: &[f64], sel: &[usize], w: Option<&[f64]>, train: &[bool]) -> Vec<f64> {
    let cols: Vec<&[f64]> = sel.iter().map(|&j| inp.x[j].as_slice()).collect();
    let c = wls(&cols, v, w, train);
    (0..v.len()).map(|i| v[i] - c[cols.len()] - cols.iter().zip(&c).map(|(x, b)| x[i] * b).sum::<f64>()).collect()
}

/// Robust (or cluster-robust) variance of an M-estimator: J^{-1} S J^{-1}'
/// with per-observation scores.
fn sandwich(j: &[Vec<f64>], scores: &[Vec<f64>], cluster: Option<&[f64]>, small: f64) -> Result<Matrix, String> {
    let ji = invert(j).ok_or_else(|| rc_err(459, "variables of interest are collinear with the selected controls"))?;
    let k = ji.len();
    let mut m = Matrix::zeros(k, k);
    for r in 0..k {
        for c in 0..k {
            m[(r, c)] = ji[r][c];
        }
    }
    let (v, _) = est::sandwich(&m, scores, cluster, small);
    Ok(v)
}

struct Estimate {
    b: Vec<f64>,
    v: Matrix,
}

/// OLS (no constant) of `y` on `d` (rows) with robust variance.
fn ols_robust(d: &[Vec<f64>], y: &[f64], cluster: Option<&[f64]>, extra_k: usize) -> Result<Estimate, String> {
    let n = y.len();
    let k = d.len();
    let mut a = vec![vec![0.0; k]; k];
    let mut c = vec![0.0; k];
    for i in 0..n {
        for r in 0..k {
            c[r] += d[r][i] * y[i];
            for s in 0..k {
                a[r][s] += d[r][i] * d[s][i];
            }
        }
    }
    let b = solve(&a, &c).ok_or_else(|| rc_err(459, "variables of interest are collinear with the controls"))?;
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let e = y[i] - (0..k).map(|r| d[r][i] * b[r]).sum::<f64>();
            (0..k).map(|r| d[r][i] * e).collect()
        })
        .collect();
    let ktot = (k + extra_k) as f64;
    let small = n as f64 / (n as f64 - ktot).max(1.0);
    let v = sandwich(&a, &scores, cluster, if cluster.is_some() { 1.0 } else { small })?;
    Ok(Estimate { b, v })
}

/// Two-stage least squares (no constant) with robust variance.
fn tsls_robust(r: &[Vec<f64>], q: &[Vec<f64>], y: &[f64], cluster: Option<&[f64]>) -> Result<Estimate, String> {
    let n = y.len();
    let (kr, kq) = (r.len(), q.len());
    let mut qq = vec![vec![0.0; kq]; kq];
    let mut qr = vec![vec![0.0; kr]; kq];
    for i in 0..n {
        for a in 0..kq {
            for b in 0..kq {
                qq[a][b] += q[a][i] * q[b][i];
            }
            for b in 0..kr {
                qr[a][b] += q[a][i] * r[b][i];
            }
        }
    }
    let qqi = invert(&qq).ok_or_else(|| rc_err(459, "instruments are collinear"))?;
    // fitted regressors A = Q (Q'Q)^{-1} Q'R
    let pi: Vec<Vec<f64>> = (0..kq).map(|a| (0..kr).map(|b| (0..kq).map(|c| qqi[a][c] * qr[c][b]).sum()).collect()).collect();
    let afit: Vec<Vec<f64>> = (0..kr).map(|b| (0..n).map(|i| (0..kq).map(|a| q[a][i] * pi[a][b]).sum()).collect()).collect();
    let mut ar = vec![vec![0.0; kr]; kr];
    let mut ay = vec![0.0; kr];
    for i in 0..n {
        for a in 0..kr {
            ay[a] += afit[a][i] * y[i];
            for b in 0..kr {
                ar[a][b] += afit[a][i] * r[b][i];
            }
        }
    }
    let b = solve(&ar, &ay).ok_or_else(|| rc_err(459, "equation not identified"))?;
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let e = y[i] - (0..kr).map(|a| r[a][i] * b[a]).sum::<f64>();
            (0..kr).map(|a| afit[a][i] * e).collect()
        })
        .collect();
    let small = n as f64 / (n as f64 - kr as f64).max(1.0);
    let v = sandwich(&ar, &scores, cluster, if cluster.is_some() { 1.0 } else { small })?;
    Ok(Estimate { b, v })
}

/// Solve the orthogonal score sum_i z_i (y_i - m(d_i'a + off_i)) = 0.
fn glm_score(fam: Family, d: &[Vec<f64>], z: &[Vec<f64>], y: &[f64], off: &[f64], a0: Vec<f64>, cluster: Option<&[f64]>) -> Result<Estimate, String> {
    let n = y.len();
    let k = d.len();
    let mut a = a0;
    let deriv = |eta: f64| -> f64 {
        match fam {
            Family::Logit => {
                let p = fam.mean(eta);
                p * (1.0 - p)
            }
            Family::Poisson => fam.mean(eta),
            Family::Probit => {
                let f = (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt();
                f
            }
            Family::Linear => 1.0,
        }
    };
    let jac = |a: &[f64]| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut j = vec![vec![0.0; k]; k];
        let mut g = vec![0.0; k];
        for i in 0..n {
            let eta = off[i] + (0..k).map(|r| d[r][i] * a[r]).sum::<f64>();
            let m = fam.mean(eta);
            let dm = deriv(eta);
            for r in 0..k {
                g[r] += z[r][i] * (y[i] - m);
                for c in 0..k {
                    j[r][c] += dm * z[r][i] * d[c][i];
                }
            }
        }
        (j, g)
    };
    for _ in 0..100 {
        let (j, g) = jac(&a);
        let step = solve(&j, &g).ok_or_else(|| rc_err(430, "orthogonal score: singular Jacobian"))?;
        let mx = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let damp = if mx > 2.0 { 2.0 / mx } else { 1.0 };
        for r in 0..k {
            a[r] += damp * step[r];
        }
        if mx < 1e-10 {
            break;
        }
    }
    let (j, _) = jac(&a);
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let eta = off[i] + (0..k).map(|r| d[r][i] * a[r]).sum::<f64>();
            let e = y[i] - fam.mean(eta);
            (0..k).map(|r| z[r][i] * e).collect()
        })
        .collect();
    let v = sandwich(&j, &scores, cluster, 1.0)?;
    Ok(Estimate { b: a, v })
}

/// Unpenalized GLM (with constant) of y on columns, with robust variance
/// of the first `kd` coefficients.
fn glm_robust(fam: Family, cols: &[&[f64]], y: &[f64], kd: usize, cluster: Option<&[f64]>) -> Result<Estimate, String> {
    let n = y.len();
    let rows: Vec<Vec<f64>> = (0..n).map(|i| cols.iter().map(|c| c[i]).chain(std::iter::once(1.0)).collect()).collect();
    let b = core::glm(&rows, y, &vec![1.0; n], None, fam).ok_or_else(|| rc_err(430, "final model did not converge"))?;
    let k = b.len();
    let mut j = vec![vec![0.0; k]; k];
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let eta: f64 = rows[i].iter().zip(&b).map(|(a, c)| a * c).sum();
        let m = fam.mean(eta);
        let w = match fam {
            Family::Logit => m * (1.0 - m),
            Family::Poisson => m,
            _ => 1.0,
        };
        for r in 0..k {
            for c in 0..k {
                j[r][c] += w * rows[i][r] * rows[i][c];
            }
        }
        scores.push(rows[i].iter().map(|x| x * (y[i] - m)).collect::<Vec<f64>>());
    }
    let small = if cluster.is_some() { 1.0 } else { n as f64 / (n as f64 - 1.0) };
    let v = sandwich(&j, &scores, cluster, small)?;
    let mut vd = Matrix::zeros(kd, kd);
    for r in 0..kd {
        for c in 0..kd {
            vd[(r, c)] = v[(r, c)];
        }
    }
    Ok(Estimate { b: b[..kd].to_vec(), v: vd })
}

impl Inf {
    fn estimate(&self, inp: &Inputs, ctx: &mut Ctx, train_folds: &[Vec<bool>], apply: &[Vec<bool>], tag: &dyn Fn(usize) -> String) -> Result<(Estimate, usize), String> {
        let fam = family(self.verb);
        let n = inp.y.len();
        let m = method(self.verb);
        let kd = inp.d.len();
        let iv = self.verb.ends_with("ivregress");
        let mut union: Vec<usize> = Vec::new();
        let add = |u: &mut Vec<usize>, s: &[usize]| {
            for j in s {
                if !u.contains(j) {
                    u.push(*j);
                }
            }
        };
        if m == Method::Ds {
            let all = vec![true; n];
            let dx: Vec<(String, Vec<f64>)> = if fam == Family::Linear { Vec::new() } else { inp.dnames.iter().cloned().zip(inp.d.iter().cloned()).collect() };
            let s = ctx.select(&inp.dep, fam, inp, &inp.y, None, &dx, &all)?;
            add(&mut union, &s);
            for (nm, c) in inp.dnames.iter().zip(&inp.d) {
                let s = ctx.select(nm, Family::Linear, inp, c, None, &[], &all)?;
                add(&mut union, &s);
            }
            union.sort_unstable();
            let mut cols: Vec<&[f64]> = inp.d.iter().map(|c| c.as_slice()).collect();
            cols.extend(union.iter().map(|&j| inp.x[j].as_slice()));
            let est = if fam == Family::Linear {
                // OLS with constant via partialling the constant out
                let mut rows: Vec<Vec<f64>> = cols.iter().map(|c| c.to_vec()).collect();
                rows.push(vec![1.0; n]);
                let full = ols_robust(&rows, &inp.y, inp.cluster.as_deref(), 0)?;
                let mut v = Matrix::zeros(kd, kd);
                for r in 0..kd {
                    for c in 0..kd {
                        v[(r, c)] = full.v[(r, c)];
                    }
                }
                Estimate { b: full.b[..kd].to_vec(), v }
            } else {
                glm_robust(fam, &cols, &inp.y, kd, inp.cluster.as_deref())?
            };
            return Ok((est, union.len()));
        }
        // partialling-out, possibly cross-fit: residuals on apply rows
        let mut yt = vec![0.0; n];
        let mut dt: Vec<Vec<f64>> = vec![vec![0.0; n]; kd];
        let mut zt: Vec<Vec<f64>> = vec![vec![0.0; n]; inp.z.len()];
        let mut off = vec![0.0; n];
        let mut a0 = vec![0.0; kd];
        for (f, (train, app)) in train_folds.iter().zip(apply).enumerate() {
            if fam == Family::Linear {
                let s = ctx.select(&tag_of(&inp.dep, f, tag), fam, inp, &inp.y, None, &[], train)?;
                add(&mut union, &s);
                let r = residual(inp, &inp.y, &s, None, train);
                for i in 0..n {
                    if app[i] {
                        yt[i] = r[i];
                    }
                }
                for (j, (nm, c)) in inp.dnames.iter().zip(&inp.d).enumerate() {
                    let s = ctx.select(&tag_of(nm, f, tag), fam, inp, c, None, &[], train)?;
                    add(&mut union, &s);
                    let r = residual(inp, c, &s, None, train);
                    for i in 0..n {
                        if app[i] {
                            dt[j][i] = r[i];
                        }
                    }
                }
                for (j, (nm, c)) in inp.znames.iter().zip(&inp.z).enumerate() {
                    let s = ctx.select(&tag_of(nm, f, tag), fam, inp, c, None, &[], train)?;
                    add(&mut union, &s);
                    let r = residual(inp, c, &s, None, train);
                    for i in 0..n {
                        if app[i] {
                            zt[j][i] = r[i];
                        }
                    }
                }
            } else {
                // step 1: penalized GLM of y on d (always) and controls,
                // post-selection GLM gives the control index
                let dx: Vec<(String, Vec<f64>)> = inp.dnames.iter().cloned().zip(inp.d.iter().cloned()).collect();
                let s = ctx.select(&tag_of(&inp.dep, f, tag), fam, inp, &inp.y, None, &dx, train)?;
                add(&mut union, &s);
                let idx: Vec<usize> = (0..n).filter(|&i| train[i]).collect();
                let rows: Vec<Vec<f64>> = idx.iter().map(|&i| inp.d.iter().map(|c| c[i]).chain(s.iter().map(|&j| inp.x[j][i])).chain(std::iter::once(1.0)).collect()).collect();
                let yy: Vec<f64> = idx.iter().map(|&i| inp.y[i]).collect();
                let th = core::glm(&rows, &yy, &vec![1.0; idx.len()], None, fam).ok_or_else(|| rc_err(430, "post-selection model did not converge"))?;
                let cpos = kd + s.len();
                if f == 0 {
                    a0 = th[..kd].to_vec();
                }
                let eta_x: Vec<f64> = (0..n).map(|i| th[cpos] + s.iter().enumerate().map(|(a, &j)| th[kd + a] * inp.x[j][i]).sum::<f64>()).collect();
                // weights from the fitted model
                let w: Vec<f64> = (0..n)
                    .map(|i| {
                        let eta = eta_x[i] + (0..kd).map(|r| th[r] * inp.d[r][i]).sum::<f64>();
                        let m = fam.mean(eta);
                        match fam {
                            Family::Logit => (m * (1.0 - m)).max(1e-6),
                            _ => m.max(1e-6),
                        }
                    })
                    .collect();
                for i in 0..n {
                    if app[i] {
                        off[i] = eta_x[i];
                    }
                }
                for (j, (nm, c)) in inp.dnames.iter().zip(&inp.d).enumerate() {
                    let s = ctx.select(&tag_of(nm, f, tag), Family::Linear, inp, c, Some(&w), &[], train)?;
                    add(&mut union, &s);
                    let r = residual(inp, c, &s, Some(&w), train);
                    for i in 0..n {
                        if app[i] {
                            dt[j][i] = r[i];
                        }
                    }
                }
            }
        }
        let est = if fam == Family::Linear {
            if iv {
                let nex = kd - inp.n_endog;
                let mut q: Vec<Vec<f64>> = dt[..nex].to_vec();
                q.extend(zt.iter().cloned());
                tsls_robust(&dt, &q, &yt, inp.cluster.as_deref())?
            } else {
                ols_robust(&dt, &yt, inp.cluster.as_deref(), 0)?
            }
        } else {
            glm_score(fam, &inp.d, &dt, &inp.y, &off, a0, inp.cluster.as_deref())?
        };
        Ok((est, union.len()))
    }
}

fn tag_of(name: &str, f: usize, tag: &dyn Fn(usize) -> String) -> String {
    format!("{name}{}", tag(f))
}

impl CommandHandler for Inf {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut fixed = BTreeMap::new();
        let mut cmd = cmd.clone();
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            if !cmd.has_opt("reestimate", 5) {
                crate::est::stash_table_from_e(&st.e, st.level);
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            if method(self.verb) == Method::Xpo {
                return Err(rc_err(198, "reestimate not allowed after cross-fit estimators"));
            }
            for (l, x) in fit::load_all(&st.e) {
                fixed.insert(l, x);
            }
            let line = st.e.macros.get("_cmdline").cloned().unwrap_or_default();
            cmd = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, format!("{e:?}")))?;
        }
        let verb = self.verb;
        let inp = inputs(ds, st, &cmd, verb)?;
        let n = inp.y.len();
        let mut s = fit::settings(&cmd, Sel::Plugin)?;
        if s.sel == Sel::None {
            return Err(rc_err(198, "selection(none) not allowed"));
        }
        s.sqrt = cmd.has_opt("sqrtlasso", 4);
        fit::rseed(&cmd, st)?;
        let level = util::level(&cmd, st)?;
        let m = method(verb);
        let mut ctx = Ctx { s, st, lassos: Vec::new(), fixed };
        let (est, nsel, xfolds, nres) = if m == Method::Xpo {
            let k = util::opt_num(&cmd, "xfolds", 2)?.map(|v| v as usize).unwrap_or(10);
            if k < 2 || k > n {
                return Err(rc_err(198, "xfolds() must be between 2 and the number of observations"));
            }
            let r = match cmd.opt("resample", 3) {
                Some(t) if !t.trim().is_empty() => t.trim().parse::<f64>().map_err(|_| rc_err(198, "invalid resample()"))? as usize,
                Some(_) => 10,
                None => {
                    if cmd.has_opt("resample", 3) {
                        10
                    } else {
                        1
                    }
                }
            }
            .max(1);
            let mut ests: Vec<Estimate> = Vec::new();
            let mut nsel = 0;
            for rep in 0..r {
                let folds = {
                    let mut rng = st.rng.borrow_mut();
                    let mut u = || rng.uniform();
                    core::folds(n, k, &mut u)
                };
                let train: Vec<Vec<bool>> = (1..=k).map(|f| folds.iter().map(|&g| g != f).collect()).collect();
                let apply: Vec<Vec<bool>> = (1..=k).map(|f| folds.iter().map(|&g| g == f).collect()).collect();
                let tag = move |f: usize| if r > 1 { format!("#f{}#r{}", f + 1, rep + 1) } else { format!("#f{}", f + 1) };
                let (e, ns) = self.estimate(&inp, &mut ctx, &train, &apply, &tag)?;
                nsel = nsel.max(ns);
                ests.push(e);
            }
            let kd = inp.d.len();
            let rr = ests.len() as f64;
            let b: Vec<f64> = (0..kd).map(|j| ests.iter().map(|e| e.b[j]).sum::<f64>() / rr).collect();
            let mut v = Matrix::zeros(kd, kd);
            for e in &ests {
                for a in 0..kd {
                    for c in 0..kd {
                        v[(a, c)] += (e.v[(a, c)] + (e.b[a] - b[a]) * (e.b[c] - b[c])) / rr;
                    }
                }
            }
            (Estimate { b, v }, nsel, k, r)
        } else {
            let all = vec![vec![true; n]];
            let (e, ns) = self.estimate(&inp, &mut ctx, &all, &all, &|_| String::new())?;
            (e, ns, 0, 0)
        };
        // display
        let kd = inp.d.len();
        let title = match (m, family(verb), verb.ends_with("ivregress")) {
            (Method::Ds, Family::Linear, _) => "Double-selection linear model",
            (Method::Ds, Family::Logit, _) => "Double-selection logit model",
            (Method::Ds, _, _) => "Double-selection Poisson model",
            (Method::Po, _, true) => "Partialing-out IV linear model",
            (Method::Po, Family::Linear, _) => "Partialing-out linear model",
            (Method::Po, Family::Logit, _) => "Partialing-out logit model",
            (Method::Po, _, _) => "Partialing-out Poisson model",
            (Method::Xpo, _, true) => "Cross-fit partialing-out IV linear model",
            (Method::Xpo, Family::Linear, _) => "Cross-fit partialing-out linear model",
            (Method::Xpo, Family::Logit, _) => "Cross-fit partialing-out logit model",
            (Method::Xpo, _, _) => "Cross-fit partialing-out Poisson model",
        };
        let idx: Vec<usize> = (0..kd).collect();
        let (chi2, df) = crate::heckman::wald(&est.b, &est.v, &idx);
        let hr = |a: &str, k: &str, v: String| format!("{a:<38}{k:<29}= {v:>10}");
        let mut out = vec![String::new()];
        out.push(hr(title, "Number of obs", util::commas(n)));
        out.push(hr("", "Number of controls", util::commas(inp.x.len())));
        if !inp.znames.is_empty() {
            out.push(hr("", "Number of instruments", util::commas(inp.znames.len())));
        }
        out.push(hr("", "Number of selected controls", util::commas(nsel)));
        if m == Method::Xpo {
            out.push(hr("", "Number of cross-fit folds", xfolds.to_string()));
            out.push(hr("", "Number of resamples", nres.to_string()));
        }
        out.push(hr("", &format!("Wald chi2({})", df as usize), xm::f2(chi2)));
        out.push(hr("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(String::new());
        let eform = match family(verb) {
            Family::Logit if cmd.has_opt("or", 2) => Some("Odds ratio"),
            Family::Poisson if cmd.has_opt("irr", 3) => Some("IRR"),
            _ => None,
        };
        let names = inp.dnames.clone();
        let falses = vec![false; kd];
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &inp.dep, names: &names, b: &est.b, v: &est.v, omitted: &falses, is_base: &falses, df: None, level, eform, coef_title: "Coefficient" });
        let lw = tbl.first().map_or(12, |b| b.len().saturating_sub(66));
        tbl.insert(1, format!("{:>lw$} |               Robust", ""));
        out.extend(tbl);
        out.push(String::new());
        out.push("Note: Chi-squared test is a Wald test of the coefficients of the variables".into());
        out.push("      of interest jointly equal to zero. Lassos select controls for model".into());
        out.push("      estimation. Type lassoinfo to see number of selected variables in each".into());
        out.push("      lasso.".into());
        let text = out.join("\n");
        // post
        let lassos = std::mem::take(&mut ctx.lassos);
        let sel_name = ctx.s.sel_name();
        drop(ctx);
        let e = st.begin_e();
        e.set_str("cmd", verb);
        e.set_str("title", title);
        e.set_str("depvar", inp.dep.clone());
        e.set_str("model", family(verb).name());
        e.set_str("varsofinterest", inp.dnames.join(" "));
        e.set_str("controls", inp.xnames.join(" "));
        if !inp.znames.is_empty() {
            e.set_str("inst", inp.znames.join(" "));
            e.set_str("endog", inp.dnames[kd - inp.n_endog..].join(" "));
            e.set("k_inst", inp.znames.len() as f64);
        }
        e.set("N", n as f64);
        e.set("k_varsofinterest", kd as f64);
        e.set("k_controls", inp.x.len() as f64);
        e.set("k_controls_sel", nsel as f64);
        e.set("chi2", chi2);
        e.set("df", df);
        e.set("p", xm::chi2p(df, chi2));
        if m == Method::Xpo {
            e.set("n_xfolds", xfolds as f64);
            e.set("n_resample", nres as f64);
        }
        e.set_str("vce", if inp.cluster.is_some() { "cluster" } else { "robust" });
        e.set_str("vcetype", "Robust");
        if let Vce::Cluster(c) = est::parse_vce(&cmd, Vce::Robust)? {
            e.set_str("clustvar", c);
        }
        e.set_str("lasso_selection", sel_name);
        e.set_str("_output", text.clone());
        e.set_str("_cmdline", cmd.raw.clone());
        e.set_str("properties", "b V");
        e.set_mat("b", Matrix::row_vector(&est.b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", est.v.clone().with_names(names.clone(), names));
        e.sample = Some(inp.mask.clone());
        fit::save(e, &lassos);
        Ok(text)
    }
}

impl Settings {
    pub fn sel_name(&self) -> &'static str {
        match self.sel {
            Sel::Cv => "cv",
            Sel::Adaptive => "adaptive",
            Sel::Plugin => "plugin",
            Sel::Bic => "bic",
            Sel::None => "none",
        }
    }
}
