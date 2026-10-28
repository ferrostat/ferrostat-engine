//! Selection, bivariate-probit, conditional-logit and linear mixed models:
//! the command list, the `predict` / `estat` hooks and shared numerics
//! (bivariate normal distribution, numerical index Hessians, table rows).

use std::sync::OnceLock;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

/// Commands implemented in this family.
pub fn handlers() -> Handlers {
    vec![
        ("heckman", Box::new(crate::heckman::Heckman { probit: false })),
        ("heckprobit", Box::new(crate::heckman::Heckman { probit: true })),
        ("biprobit", Box::new(crate::heckman::Biprobit)),
        ("clogit", Box::new(crate::clogit::Clogit)),
        ("asclogit", Box::new(crate::clogit::Asclogit)),
        ("mixed", Box::new(crate::mixed::Mixed)),
        ("xtmixed", Box::new(crate::mixed::Mixed)),
        ("meprobit", Box::new(crate::melogit::Melogit { verb: "meprobit" })),
        ("mepoisson", Box::new(crate::melogit::Melogit { verb: "mepoisson" })),
        ("meglm", Box::new(crate::melogit::Melogit { verb: "meglm" })),
    ]
}

/// `predict` after the commands of this family (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "heckman" | "heckprobit" | "biprobit" => crate::heckman::predict(ds, st, cmd).map(Some),
        "clogit" | "asclogit" => crate::clogit::predict(ds, st, cmd).map(Some),
        "mixed" => crate::mixed::predict(ds, st, cmd).map(Some),
        _ => Ok(None),
    }
}

/// `estat` after the commands of this family (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match (ecmd.as_str(), sub) {
        ("mixed", "icc") | ("mixed", "recovariance") | ("mixed", "group") | ("mixed", "sd") => Some(crate::mixed::estat(ds, st, sub, cmd)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// bivariate normal
// ---------------------------------------------------------------------------

/// Gauss-Legendre nodes and weights on [-1, 1] (20 points).
fn gl20() -> &'static (Vec<f64>, Vec<f64>) {
    static GL: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    GL.get_or_init(|| {
        let n = 20;
        let mut j = Matrix::zeros(n, n);
        for i in 1..n {
            let b = i as f64 / ((4 * i * i - 1) as f64).sqrt();
            j[(i, i - 1)] = b;
            j[(i - 1, i)] = b;
        }
        let (vals, vecs) = j.sym_eigen();
        let mut pairs: Vec<(f64, f64)> = (0..n).map(|k| (vals[k], 2.0 * vecs[(0, k)] * vecs[(0, k)])).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1).collect())
    })
}

/// P(X < h, Y < k) for standard bivariate normal variables with correlation
/// `r` (Plackett's identity integrated over the arcsine of the correlation,
/// with panels graded towards |r| = 1).
pub fn bvn(h: f64, k: f64, r: f64) -> f64 {
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        return 0.0;
    }
    if h == f64::INFINITY {
        return dist::normal(k);
    }
    if k == f64::INFINITY {
        return dist::normal(h);
    }
    if r >= 1.0 {
        return dist::normal(h.min(k));
    }
    if r <= -1.0 {
        return (dist::normal(h) - dist::normal(-k)).max(0.0);
    }
    let (x, w) = gl20();
    let t1 = r.asin();
    let hs = 0.5 * (h * h + k * k);
    let hk = h * k;
    let f = |th: f64| -> f64 {
        let s = th.sin();
        let c2 = 1.0 - s * s;
        ((s * hk - hs) / c2).exp()
    };
    let mut panels = vec![0.0];
    if r.abs() < 0.925 {
        panels.push(1.0);
    } else {
        for j in 1..=8 {
            panels.push(1.0 - 0.5f64.powi(j));
        }
        panels.push(1.0);
    }
    let mut s = 0.0;
    for p in panels.windows(2) {
        let (a, b) = (p[0] * t1, p[1] * t1);
        let (m, hw) = (0.5 * (a + b), 0.5 * (b - a));
        for i in 0..x.len() {
            s += w[i] * hw * f(m + hw * x[i]);
        }
    }
    (dist::normal(h) * dist::normal(k) + s / (2.0 * std::f64::consts::PI)).clamp(0.0, 1.0)
}

/// ln P(X < h, Y < k) and its derivatives in h, k and r.
pub fn ln_bvn(h: f64, k: f64, r: f64) -> (f64, f64, f64, f64) {
    let p = bvn(h, k, r).max(1e-300);
    let s = (1.0 - r * r).max(1e-300).sqrt();
    let dh = dist::normalden(h) * dist::normal((k - r * h) / s);
    let dk = dist::normalden(k) * dist::normal((h - r * k) / s);
    let q = (h * h - 2.0 * r * h * k + k * k) / (2.0 * s * s);
    let dr = (-q).exp() / (2.0 * std::f64::consts::PI * s);
    (p.ln(), dh / p, dk / p, dr / p)
}

/// ln Phi(z) and d/dz.
pub fn ln_phi_d(z: f64) -> (f64, f64) {
    let l = crate::xtnl::ln_phi(z);
    (l, (dist::lnnormalden(z) - l).exp())
}

/// Hessian of a per-observation index function from its analytic gradient
/// (central differences, symmetrised).
pub fn num_hess(g: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
    let m = x.len();
    let mut h = vec![vec![0.0; m]; m];
    let mut y = x.to_vec();
    for j in 0..m {
        let s = 1e-6 * (1.0 + x[j].abs());
        y[j] = x[j] + s;
        let gp = g(&y);
        y[j] = x[j] - s;
        let gm = g(&y);
        y[j] = x[j];
        for i in 0..m {
            h[i][j] = (gp[i] - gm[i]) / (2.0 * s);
        }
    }
    for i in 0..m {
        for j in 0..i {
            let v = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = v;
            h[j][i] = v;
        }
    }
    h
}

/// `maximize_robust` followed by plain Newton steps until the scaled
/// gradient vanishes (tighter convergence of the parameters).
pub fn maximize_tight(model: &dyn crate::est::Likelihood, b0: Vec<f64>, max_iter: usize) -> Result<crate::est::MlResult, String> {
    let mut res = crate::zip::maximize_robust(model, b0, max_iter)?;
    let k = model.k();
    for _ in 0..20 {
        let (ll, g, h, _) = model.eval(&res.b, false);
        let (hi, _) = h.sym_inverse(1e-14);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * g[j]).sum()).collect();
        let dec: f64 = step.iter().zip(&g).map(|(s, gg)| s * gg).sum();
        if !(dec > 1e-20) || h.cholesky().is_err() {
            break;
        }
        let nb: Vec<f64> = res.b.iter().zip(&step).map(|(b, s)| b + s).collect();
        let (nl, ..) = model.eval(&nb, false);
        if !(nl >= ll - 1e-9 * ll.abs().max(1.0)) {
            break;
        }
        res.b = nb;
        if dec < 1e-16 {
            break;
        }
    }
    let (ll, _, info, scores) = model.eval(&res.b, true);
    res.ll = ll;
    res.info = info;
    res.scores = scores;
    if let Some(last) = res.iterations.last_mut() {
        *last = ll;
    }
    Ok(res)
}

// ---------------------------------------------------------------------------
// output helpers
// ---------------------------------------------------------------------------

/// Critical value of the normal distribution for a confidence level.
pub fn zcrit(level: f64) -> f64 {
    dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0)
}

/// A coefficient-table row with test statistic and interval.
pub fn row_z(lw: usize, label: &str, b: f64, se: f64, crit: f64) -> String {
    if !(se > 0.0 && se.is_finite()) {
        return format!("{:>lw$} | {:>10}  {:>9}", crate::summarize::abbrev(label, lw), fmt_g(b, 8), ".");
    }
    let z = b / se;
    let p = 2.0 * dist::normal(-z.abs());
    format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", crate::summarize::abbrev(label, lw), fmt_g(b, 8), fmt_g(se, 8), z, p, fmt_g(b - crit * se, 8), fmt_g(b + crit * se, 8))
}

/// A row with an estimate, standard error and an interval but no test.
pub fn row_ci(lw: usize, label: &str, b: f64, se: f64, lo: f64, hi: f64) -> String {
    format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", crate::summarize::abbrev(label, lw), fmt_g(b, 8), fmt_g(se, 8), "", fmt_g(lo, 8), fmt_g(hi, 8))
}

/// A separator inside a coefficient table.
pub fn sep(lw: usize) -> String {
    format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64))
}

/// Header line: left text, then a right-hand statistic.
pub fn hrow(left: &str, key: &str, val: String) -> String {
    format!("{:<48}{:<17} = {:>10}", left, key, val)
}

pub fn f2(v: f64) -> String {
    if v.is_finite() && !ferrostat_core::missing::is_missing(v) { format!("{v:.2}") } else { ".".into() }
}

pub fn f4(v: f64) -> String {
    if v.is_finite() && !ferrostat_core::missing::is_missing(v) { format!("{v:.4}") } else { ".".into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvn_reference_values() {
        // independent, symmetric and closed-form cases
        assert!((bvn(0.3, -0.2, 0.0) - dist::normal(0.3) * dist::normal(-0.2)).abs() < 1e-14);
        assert!((bvn(0.0, 0.0, 0.5) - 1.0 / 3.0).abs() < 1e-13);
        assert!((bvn(0.0, 0.0, -0.5) - 1.0 / 6.0).abs() < 1e-13);
        // P(X<0,Y<0) = 1/4 + asin(r)/(2 pi)
        for r in [0.93, 0.97, 0.995, -0.99] {
            let e = 0.25 + f64::asin(r) / (2.0 * std::f64::consts::PI);
            assert!((bvn(0.0, 0.0, r) - e).abs() < 1e-12, "{r}");
        }
        // derivative in r equals the density (Plackett)
        let (h, k, r) = (0.4, -0.7, 0.96);
        let d = (bvn(h, k, r + 1e-6) - bvn(h, k, r - 1e-6)) / 2e-6;
        let (lp, _, _, dr) = ln_bvn(h, k, r);
        assert!((d - dr * lp.exp()).abs() < 1e-7);
    }
}
