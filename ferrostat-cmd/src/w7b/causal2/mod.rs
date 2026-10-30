//! Treatment effects with instruments, heterogeneity, mediation and
//! survival outcomes:
//!
//! * `lateffects`, `lateoverlap`, `latebalance` (local average treatment
//!   effects with a binary instrument) in `late`;
//! * `cate` (conditional average treatment effects) and `telasso`
//!   (doubly robust effects with lasso-selected models) in `cate`;
//! * `mediate` (causal mediation analysis) in `mediate`;
//! * `stteffects` (survival-time treatment effects) in `stte`.
//!
//! Standard errors come from stacked estimating equations (M-estimation)
//! with numerical Jacobians, as in `teffects`.

use ferrostat_core::DataSet;
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::w7b::lasso::core::Family;
use crate::xmodels::Handlers;

pub mod cate;
pub mod late;
pub mod mediate;
pub mod stte;

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.extend(late::handlers());
    v.extend(cate::handlers());
    v.extend(mediate::handlers());
    v.extend(stte::handlers());
    v
}

/// `predict` after this family's commands (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "cate" => Some(cate::predict(ds, st, cmd)),
        _ => None,
    }
}

/// `estat` after this family's commands (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "cate" => Some(cate::estat(ds, st, sub, cmd)),
        "mediate" => Some(mediate::estat(ds, st, sub, cmd)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Parenthesized groups of a command line, in order.
pub(crate) fn groups(text: &str) -> Vec<String> {
    split_top_level_words(text).into_iter().filter(|w| w.starts_with('(')).collect()
}

/// Outcome family named among model options (default `def`).
pub(crate) fn family_of(opts: &str, def: Family) -> Family {
    let w: Vec<&str> = opts.split(|c: char| c.is_whitespace() || c == ',').collect();
    if w.contains(&"logit") {
        Family::Logit
    } else if w.contains(&"probit") {
        Family::Probit
    } else if w.iter().any(|o| o.starts_with("pois")) {
        Family::Poisson
    } else if w.contains(&"linear") {
        Family::Linear
    } else {
        def
    }
}

pub(crate) fn fam_label(f: Family) -> &'static str {
    match f {
        Family::Linear => "linear",
        Family::Logit => "logit",
        Family::Probit => "probit",
        Family::Poisson => "Poisson",
    }
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// d log L / d eta of one observation.
pub(crate) fn score(f: Family, y: f64, eta: f64) -> f64 {
    match f {
        Family::Linear | Family::Logit | Family::Poisson => y - f.mean(eta),
        Family::Probit => {
            let p = dist::normal(eta).clamp(1e-12, 1.0 - 1e-12);
            dist::normalden(eta) * (y - p) / (p * (1.0 - p))
        }
    }
}

/// Weighted unpenalized GLM on rows `x`.
pub(crate) fn fit(x: &[Vec<f64>], y: &[f64], w: &[f64], f: Family) -> Result<Vec<f64>, String> {
    crate::w7b::lasso::core::glm(x, y, w, None, f).ok_or_else(|| rc_err(459, "model could not be fit: collinear covariates"))
}

/// Mean over observations of the moment vector at `th`.
fn mean_moments(th: &[f64], n: usize, np: usize, moments: &dyn Fn(&[f64], usize, &mut [f64])) -> Vec<f64> {
    let mut acc = vec![0.0; np];
    let mut m = vec![0.0; np];
    for i in 0..n {
        m.iter_mut().for_each(|v| *v = 0.0);
        moments(th, i, &mut m);
        for a in 0..np {
            acc[a] += m[a];
        }
    }
    acc.iter().map(|v| v / n as f64).collect()
}

fn jacobian(theta: &[f64], n: usize, np: usize, cols: std::ops::Range<usize>, moments: &dyn Fn(&[f64], usize, &mut [f64])) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols.len()]; np];
    for (ci, c) in cols.enumerate() {
        let h = 1e-6 * theta[c].abs().max(1e-3);
        let mut tp = theta.to_vec();
        tp[c] += h;
        let up = mean_moments(&tp, n, np, moments);
        tp[c] -= 2.0 * h;
        let dn = mean_moments(&tp, n, np, moments);
        for r in 0..np {
            a[r][ci] = (up[r] - dn[r]) / (2.0 * h);
        }
    }
    a
}

/// Solve the moments of the parameter block `from..np` (the others held
/// fixed) by Newton steps with a numerical Jacobian.
pub(crate) fn solve_block(theta: &mut [f64], n: usize, np: usize, from: usize, moments: &dyn Fn(&[f64], usize, &mut [f64])) -> Result<(), String> {
    for _ in 0..50 {
        let m = mean_moments(theta, n, np, moments);
        let g: Vec<f64> = m[from..].to_vec();
        if g.iter().all(|v| v.abs() < 1e-12) {
            return Ok(());
        }
        let j = jacobian(theta, n, np, from..np, moments);
        let jb: Vec<Vec<f64>> = j[from..].to_vec();
        let step = crate::teffects::solve(&jb, &g).ok_or_else(|| rc_err(459, "estimating equations are singular"))?;
        let mx = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for (a, s) in step.iter().enumerate() {
            theta[from + a] -= s;
        }
        if mx < 1e-11 * (1.0 + theta[from..].iter().map(|v| v.abs()).fold(0.0, f64::max)) {
            return Ok(());
        }
    }
    Ok(())
}

/// M-estimation sandwich A^-1 B A^-T / n.
pub(crate) fn sandwich(theta: &[f64], n: usize, np: usize, moments: &dyn Fn(&[f64], usize, &mut [f64])) -> Result<Vec<Vec<f64>>, String> {
    let a = jacobian(theta, n, np, 0..np, moments);
    let mut b = vec![vec![0.0; np]; np];
    let mut m = vec![0.0; np];
    for i in 0..n {
        m.iter_mut().for_each(|v| *v = 0.0);
        moments(theta, i, &mut m);
        for r in 0..np {
            if m[r] == 0.0 {
                continue;
            }
            for c in 0..np {
                b[r][c] += m[r] * m[c] / n as f64;
            }
        }
    }
    let ai = crate::teffects::invert(&a).ok_or_else(|| rc_err(459, "estimating equations are singular"))?;
    let tmp: Vec<Vec<f64>> = (0..np).map(|r| (0..np).map(|l| (0..np).map(|k| ai[r][k] * b[k][l]).sum()).collect()).collect();
    Ok((0..np).map(|r| (0..np).map(|c| (0..np).map(|l| tmp[r][l] * ai[c][l]).sum::<f64>() / n as f64).collect()).collect())
}

/// Linear combinations `c` of parameters with covariance `v` restricted
/// to the block starting at `from`.
pub(crate) fn combine(cm: &[Vec<f64>], theta: &[f64], v: &[Vec<f64>], from: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = cm.len();
    let b: Vec<f64> = cm.iter().map(|c| c.iter().enumerate().map(|(a, w)| w * theta[from + a]).sum()).collect();
    let vv: Vec<Vec<f64>> = (0..k)
        .map(|r| (0..k).map(|c| cm[r].iter().enumerate().map(|(a, wa)| cm[c].iter().enumerate().map(|(bb, wb)| wa * v[from + a][from + bb] * wb).sum::<f64>()).sum()).collect())
        .collect();
    (b, vv)
}

/// Post b and V (with names) and the estimation sample.
pub(crate) fn post(st: &mut Store, ds: &DataSet, rows: &[usize], cmd: &str, b: &[f64], v: &[Vec<f64>], names: &[String], extra: &dyn Fn(&mut crate::store::Results)) {
    let k = b.len();
    let bm = Matrix::row_vector(b).with_names(vec!["y1".into()], names.to_vec());
    let mut vm = Matrix::zeros(k, k);
    for r in 0..k {
        for c in 0..k {
            vm[(r, c)] = v[r][c];
        }
    }
    let vm = vm.with_names(names.to_vec(), names.to_vec());
    let mut sample = vec![false; ds.n_obs];
    for &r in rows {
        sample[r] = true;
    }
    let e = st.begin_e();
    e.set("N", rows.len() as f64);
    e.set_str("cmd", cmd);
    e.set_str("vce", "robust");
    e.set_str("vcetype", "Robust");
    e.set_str("properties", "b V");
    e.set_mat("b", bm);
    e.set_mat("V", vm);
    e.sample = Some(sample);
    extra(e);
}

/// Distinct sorted levels of a treatment-like variable (nonnegative integers).
pub(crate) fn levels(v: &[f64], name: &str) -> Result<Vec<f64>, String> {
    if v.iter().any(|x| x.fract() != 0.0 || *x < 0.0) {
        return Err(rc_err(459, format!("{name} must have nonnegative integer values")));
    }
    let s: std::collections::BTreeSet<i64> = v.iter().map(|&x| x as i64).collect();
    let out: Vec<f64> = s.into_iter().map(|x| x as f64).collect();
    if out.len() < 2 {
        return Err(rc_err(459, format!("{name} does not vary")));
    }
    Ok(out)
}

pub(crate) fn fmtv(v: f64) -> String {
    crate::format::fmt_g(v, 9).trim().to_string()
}
