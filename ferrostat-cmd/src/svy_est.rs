//! The `svy:` prefix: survey estimation of means, totals, ratios,
//! proportions, tables and regression models with design-based variances
//! (Taylor linearization or replicate weights), plus `estat effects` and
//! `estat sd` after it.
//!
//! Every estimator works on the rows of the design sample.  Rows outside
//! the subpopulation keep a zero weight, so they add nothing to the
//! estimates but still define the strata and sampling units, which is what
//! makes subpopulation variances correct.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options};
use ferrostat_parser::Command;

use crate::dist;
use crate::est::{self, DesignSpec, TableSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::svy::{current, DesignData, SvySet};
use crate::{util, varlist};

type Opts = Vec<(String, Option<String>)>;

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str()))
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opts.iter().find(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone())
}

fn vce_method(word: &str) -> Option<&'static str> {
    let w = word.trim().to_lowercase();
    match w.as_str() {
        x if x.len() >= 3 && "linearized".starts_with(x) => Some("linearized"),
        "brr" => Some("brr"),
        x if x.len() >= 2 && "jackknife".starts_with(x) => Some("jackknife"),
        x if x.len() >= 2 && "bootstrap".starts_with(x) => Some("bootstrap"),
        "sdr" => Some("sdr"),
        _ => None,
    }
}

/// Resolve the command after the colon to a supported estimator.
fn resolve(verb: &str) -> Option<&'static str> {
    let v = verb.to_lowercase();
    let is = |full: &'static str, min: usize| v.len() >= min && full.starts_with(v.as_str());
    [
        ("mean", 4),
        ("total", 5),
        ("ratio", 5),
        ("proportion", 4),
        ("tabulate", 2),
        ("regress", 3),
        ("logit", 4),
        ("logistic", 8),
        ("probit", 4),
        ("poisson", 7),
        ("ologit", 6),
        ("oprobit", 7),
        ("mlogit", 6),
    ]
    .into_iter()
    .find(|(f, m)| is(f, *m))
    .map(|(f, _)| f)
}

// ---------------------------------------------------------------------------
// Estimators
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Mean,
    Total,
    Ratio,
    Prop,
    Tab,
    Model,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Family {
    Regress,
    Logit,
    Probit,
    Poisson,
    Ologit,
    Oprobit,
    Mlogit,
}

/// Data of an estimator over the design rows.
enum Data {
    /// means / totals / proportions: one column per statistic and the group
    /// (over()) each statistic belongs to: (column, group) pairs
    Means { cols: Vec<Vec<f64>>, gid: Vec<usize>, stats: Vec<(usize, usize)>, total: bool },
    Ratios { num: Vec<Vec<f64>>, den: Vec<Vec<f64>> },
    Model { fam: Family, x: Vec<Vec<f64>>, y: Vec<f64>, ncat: usize, base: usize, yidx: Vec<usize>, offset: Vec<f64> },
}

struct Fit {
    b: Vec<f64>,
    /// weighted influence values per design row
    z: Vec<Vec<f64>>,
    r2: Option<f64>,
}

fn solve_inv(m: &Matrix) -> Matrix {
    m.sym_inverse(1e-14).0
}

impl Data {
    fn fit(&self, w: &[f64]) -> Result<Fit, String> {
        match self {
            Data::Means { cols, gid, stats, total } => {
                let n = w.len();
                let ng = gid.iter().copied().max().map_or(1, |g| g + 1);
                let mut wg = vec![0.0; ng];
                for t in 0..n {
                    wg[gid[t]] += w[t];
                }
                let k = stats.len();
                let mut b = vec![0.0; k];
                for (s, &(c, g)) in stats.iter().enumerate() {
                    let sum: f64 = (0..n).filter(|&t| gid[t] == g && w[t] != 0.0).map(|t| w[t] * cols[c][t]).sum();
                    b[s] = if *total { sum } else if wg[g] > 0.0 { sum / wg[g] } else { SYSMISS };
                }
                let mut z = vec![vec![0.0; k]; n];
                for t in 0..n {
                    if w[t] == 0.0 {
                        continue;
                    }
                    for (s, &(c, g)) in stats.iter().enumerate() {
                        if gid[t] != g {
                            continue;
                        }
                        z[t][s] = if *total { w[t] * cols[c][t] } else { w[t] * (cols[c][t] - b[s]) / wg[g] };
                    }
                }
                Ok(Fit { b, z, r2: None })
            }
            Data::Ratios { num, den } => {
                let n = w.len();
                let k = num.len();
                let mut b = vec![0.0; k];
                let mut xs = vec![0.0; k];
                for j in 0..k {
                    let ys: f64 = (0..n).map(|t| if w[t] != 0.0 { w[t] * num[j][t] } else { 0.0 }).sum();
                    xs[j] = (0..n).map(|t| if w[t] != 0.0 { w[t] * den[j][t] } else { 0.0 }).sum();
                    b[j] = if xs[j] != 0.0 { ys / xs[j] } else { SYSMISS };
                }
                let z = (0..n).map(|t| (0..k).map(|j| if w[t] == 0.0 { 0.0 } else { w[t] * (num[j][t] - b[j] * den[j][t]) / xs[j] }).collect()).collect();
                Ok(Fit { b, z, r2: None })
            }
            Data::Model { .. } => self.fit_model(w),
        }
    }

    fn fit_model(&self, w: &[f64]) -> Result<Fit, String> {
        let Data::Model { fam, x, y, ncat, base, yidx, offset } = self else { unreachable!() };
        let n = w.len();
        let kx = x.len();
        let xrow = |t: usize| -> Vec<f64> { x.iter().map(|c| c[t]).collect() };
        match fam {
            Family::Regress => {
                let mut xtx = Matrix::zeros(kx, kx);
                let mut xty = vec![0.0; kx];
                for t in 0..n {
                    if w[t] == 0.0 {
                        continue;
                    }
                    let r = xrow(t);
                    for a in 0..kx {
                        xty[a] += w[t] * r[a] * y[t];
                        for c in 0..kx {
                            xtx[(a, c)] += w[t] * r[a] * r[c];
                        }
                    }
                }
                let d = solve_inv(&xtx);
                let b: Vec<f64> = (0..kx).map(|a| (0..kx).map(|c| d[(a, c)] * xty[c]).sum()).collect();
                let sw: f64 = w.iter().sum();
                let ybar = (0..n).map(|t| w[t] * y[t]).sum::<f64>() / sw;
                let mut rss = 0.0;
                let mut tss = 0.0;
                let mut z = vec![vec![0.0; kx]; n];
                for t in 0..n {
                    if w[t] == 0.0 {
                        continue;
                    }
                    let r = xrow(t);
                    let e = y[t] - r.iter().zip(&b).map(|(a, c)| a * c).sum::<f64>();
                    rss += w[t] * e * e;
                    tss += w[t] * (y[t] - ybar).powi(2);
                    for a in 0..kx {
                        z[t][a] = w[t] * e * (0..kx).map(|c| d[(a, c)] * r[c]).sum::<f64>();
                    }
                }
                Ok(Fit { b, z, r2: Some(1.0 - rss / tss) })
            }
            _ => {
                let k = self.nparams();
                let b0 = self.start(w);
                let b = newton(k, b0, &|b: &[f64]| self.ll_grad_hess(b, w))?;
                let (_, _, negh) = self.ll_grad_hess(&b, w);
                let d = solve_inv(&negh);
                let s = self.scores(&b);
                let z = (0..n)
                    .map(|t| {
                        if w[t] == 0.0 {
                            return vec![0.0; k];
                        }
                        (0..k).map(|a| w[t] * (0..k).map(|c| d[(a, c)] * s[t][c]).sum::<f64>()).collect()
                    })
                    .collect();
                let _ = (ncat, base, yidx, offset);
                Ok(Fit { b, z, r2: None })
            }
        }
    }

    fn nparams(&self) -> usize {
        match self {
            Data::Model { fam, x, ncat, .. } => match fam {
                Family::Ologit | Family::Oprobit => x.len() + ncat - 1,
                Family::Mlogit => x.len() * (ncat - 1),
                _ => x.len(),
            },
            _ => 0,
        }
    }

    fn start(&self, w: &[f64]) -> Vec<f64> {
        let Data::Model { fam, x, y, ncat, base, yidx, .. } = self else { return vec![] };
        let kx = x.len();
        let sw: f64 = w.iter().sum();
        let ybar = (0..w.len()).map(|t| w[t] * y[t]).sum::<f64>() / sw;
        // index of the constant: the last column when it is all ones
        let cons = (kx > 0 && (0..w.len()).all(|t| w[t] == 0.0 || x[kx - 1][t] == 1.0)).then(|| kx - 1);
        let mut b = vec![0.0; self.nparams()];
        match fam {
            Family::Logit | Family::Probit | Family::Poisson => {
                if let Some(c) = cons {
                    let ybin = (0..w.len()).map(|t| w[t] * if y[t] != 0.0 { 1.0 } else { 0.0 }).sum::<f64>() / sw;
                    b[c] = match fam {
                        Family::Logit => dist::logit(ybin.clamp(1e-6, 1.0 - 1e-6)),
                        Family::Probit => dist::invnormal(ybin.clamp(1e-6, 1.0 - 1e-6)),
                        _ => ybar.max(1e-6).ln(),
                    };
                }
            }
            Family::Ologit | Family::Oprobit => {
                let mut cum = 0.0;
                for c in 0..ncat - 1 {
                    cum += (0..w.len()).filter(|&t| yidx[t] == c).map(|t| w[t]).sum::<f64>() / sw;
                    let p = cum.clamp(1e-6, 1.0 - 1e-6);
                    b[kx + c] = if *fam == Family::Ologit { dist::logit(p) } else { dist::invnormal(p) };
                }
            }
            Family::Mlogit => {
                if let Some(c0) = cons {
                    let f = |c: usize| (0..w.len()).filter(|&t| yidx[t] == c).map(|t| w[t]).sum::<f64>();
                    let fb = f(*base);
                    for (e, c) in (0..*ncat).filter(|c| c != base).enumerate() {
                        let fc = f(c);
                        if fc > 0.0 && fb > 0.0 {
                            b[e * kx + c0] = (fc / fb).ln();
                        }
                    }
                }
            }
            Family::Regress => {}
        }
        b
    }

    /// Per-observation scores (derivatives of the log likelihood).
    fn scores(&self, b: &[f64]) -> Vec<Vec<f64>> {
        let Data::Model { fam, x, y, ncat, base, yidx, offset } = self else { return vec![] };
        let n = y.len();
        let kx = x.len();
        let k = self.nparams();
        (0..n)
            .map(|t| {
                let xb: f64 = (0..kx).map(|j| x[j][t] * b[j]).sum::<f64>() + offset[t];
                let mut s = vec![0.0; k];
                match fam {
                    Family::Logit | Family::Probit | Family::Poisson => {
                        let (_, d1, _) = index(*fam, y[t], xb);
                        for j in 0..kx {
                            s[j] = d1 * x[j][t];
                        }
                    }
                    Family::Ologit | Family::Oprobit => {
                        let g = ordered_obs(*fam, b, kx, *ncat, yidx[t], xb);
                        for j in 0..kx {
                            s[j] = g.1 * x[j][t];
                        }
                        for c in 0..ncat - 1 {
                            s[kx + c] = g.2[c];
                        }
                    }
                    Family::Mlogit => {
                        let p = mlogit_probs(b, x, t, kx, *ncat, *base);
                        for (e, c) in (0..*ncat).filter(|c| c != base).enumerate() {
                            let r = (if yidx[t] == c { 1.0 } else { 0.0 }) - p[c];
                            for j in 0..kx {
                                s[e * kx + j] = r * x[j][t];
                            }
                        }
                    }
                    Family::Regress => {}
                }
                s
            })
            .collect()
    }

    /// (weighted log likelihood, gradient, negative Hessian)
    fn ll_grad_hess(&self, b: &[f64], w: &[f64]) -> (f64, Vec<f64>, Matrix) {
        let Data::Model { fam, x, y, ncat, base, yidx, offset } = self else { unreachable!() };
        let n = y.len();
        let kx = x.len();
        let k = self.nparams();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        match fam {
            Family::Logit | Family::Probit | Family::Poisson => {
                for t in 0..n {
                    if w[t] == 0.0 {
                        continue;
                    }
                    let xb: f64 = (0..kx).map(|j| x[j][t] * b[j]).sum::<f64>() + offset[t];
                    let (l, d1, d2) = index(*fam, y[t], xb);
                    ll += w[t] * l;
                    for a in 0..kx {
                        let xa = x[a][t];
                        if xa == 0.0 {
                            continue;
                        }
                        g[a] += w[t] * d1 * xa;
                        for c in 0..kx {
                            h[(a, c)] -= w[t] * d2 * xa * x[c][t];
                        }
                    }
                }
            }
            Family::Mlogit => {
                let eqs: Vec<usize> = (0..*ncat).filter(|c| c != base).collect();
                for t in 0..n {
                    if w[t] == 0.0 {
                        continue;
                    }
                    let p = mlogit_probs(b, x, t, kx, *ncat, *base);
                    ll += w[t] * p[yidx[t]].max(1e-300).ln();
                    for (e, &c) in eqs.iter().enumerate() {
                        let r = (if yidx[t] == c { 1.0 } else { 0.0 }) - p[c];
                        for j in 0..kx {
                            g[e * kx + j] += w[t] * r * x[j][t];
                        }
                        for (f, &c2) in eqs.iter().enumerate() {
                            let m = p[c] * ((if c == c2 { 1.0 } else { 0.0 }) - p[c2]);
                            for i in 0..kx {
                                let xi = x[i][t];
                                if xi == 0.0 {
                                    continue;
                                }
                                for j in 0..kx {
                                    h[(e * kx + i, f * kx + j)] += w[t] * m * xi * x[j][t];
                                }
                            }
                        }
                    }
                }
            }
            Family::Ologit | Family::Oprobit => {
                let grad = |bb: &[f64]| -> (f64, Vec<f64>) {
                    let mut ll = 0.0;
                    let mut g = vec![0.0; k];
                    for t in 0..n {
                        if w[t] == 0.0 {
                            continue;
                        }
                        let xb: f64 = (0..kx).map(|j| x[j][t] * bb[j]).sum::<f64>() + offset[t];
                        let (l, dxb, dc) = ordered_obs(*fam, bb, kx, *ncat, yidx[t], xb);
                        ll += w[t] * l;
                        for j in 0..kx {
                            g[j] += w[t] * dxb * x[j][t];
                        }
                        for c in 0..ncat - 1 {
                            g[kx + c] += w[t] * dc[c];
                        }
                    }
                    (ll, g)
                };
                let (l0, g0) = grad(b);
                ll = l0;
                g = g0;
                for j in 0..k {
                    let step = 1e-5 * b[j].abs().max(1.0);
                    let mut bp = b.to_vec();
                    bp[j] += step;
                    let mut bm = b.to_vec();
                    bm[j] -= step;
                    let (_, gp) = grad(&bp);
                    let (_, gm) = grad(&bm);
                    for i in 0..k {
                        h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * step);
                    }
                }
                for i in 0..k {
                    for j in 0..i {
                        let a = (h[(i, j)] + h[(j, i)]) / 2.0;
                        h[(i, j)] = a;
                        h[(j, i)] = a;
                    }
                }
                // cutpoints out of order: no valid likelihood
                if (1..ncat - 1).any(|c| b[kx + c] <= b[kx + c - 1]) {
                    ll = f64::NEG_INFINITY;
                }
            }
            Family::Regress => {}
        }
        (ll, g, h)
    }
}

/// (ll, d ll/d xb, d2 ll/d xb2) of single-index models.
fn index(fam: Family, y: f64, xb: f64) -> (f64, f64, f64) {
    match fam {
        Family::Logit => {
            let p = dist::invlogit(xb);
            let lnp = -(1.0 + (-xb).exp()).ln();
            let lnq = -(1.0 + xb.exp()).ln();
            let lnp = if lnp.is_finite() { lnp } else { xb };
            let lnq = if lnq.is_finite() { lnq } else { -xb };
            let ll = if y != 0.0 { lnp } else { lnq };
            (ll, (if y != 0.0 { 1.0 } else { 0.0 }) - p, -p * (1.0 - p))
        }
        Family::Probit => {
            let q = if y != 0.0 { 1.0 } else { -1.0 };
            let z = q * xb;
            let cdf = dist::normal(z);
            let lncdf = if cdf > 1e-300 { cdf.ln() } else { -0.5 * z * z - (-z).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() };
            let lam = if cdf > 1e-12 { dist::normalden(z) / cdf } else { -z + 1.0 / (-z) };
            (lncdf, q * lam, -lam * (lam + z))
        }
        _ => {
            let mu = xb.exp();
            (y * xb - mu - dist::lngamma(y + 1.0), y - mu, -mu)
        }
    }
}

/// Ordered model observation: (ll, d ll/d xb, d ll/d cut_c).
fn ordered_obs(fam: Family, b: &[f64], kx: usize, ncat: usize, j: usize, xb: f64) -> (f64, f64, Vec<f64>) {
    let (cdf, pdf): (fn(f64) -> f64, fn(f64) -> f64) = if fam == Family::Ologit {
        (dist::invlogit, |z| {
            let p = dist::invlogit(z);
            p * (1.0 - p)
        })
    } else {
        (dist::normal, dist::normalden)
    };
    let hi = if j < ncat - 1 { Some(b[kx + j] - xb) } else { None };
    let lo = if j > 0 { Some(b[kx + j - 1] - xb) } else { None };
    let fh = hi.map_or(1.0, cdf);
    let fl = lo.map_or(0.0, cdf);
    let p = (fh - fl).max(1e-300);
    let dh = hi.map_or(0.0, pdf);
    let dl = lo.map_or(0.0, pdf);
    let mut dc = vec![0.0; ncat - 1];
    if j < ncat - 1 {
        dc[j] = dh / p;
    }
    if j > 0 {
        dc[j - 1] = -dl / p;
    }
    (p.ln(), -(dh - dl) / p, dc)
}

fn mlogit_probs(b: &[f64], x: &[Vec<f64>], t: usize, kx: usize, ncat: usize, base: usize) -> Vec<f64> {
    let mut eta = vec![0.0; ncat];
    for (e, c) in (0..ncat).filter(|&c| c != base).enumerate() {
        eta[c] = (0..kx).map(|j| x[j][t] * b[e * kx + j]).sum();
    }
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = eta.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|v| v / s).collect()
}

/// Newton-Raphson with step halving on (ll, gradient, negative Hessian).
fn newton(k: usize, b0: Vec<f64>, f: &dyn Fn(&[f64]) -> (f64, Vec<f64>, Matrix)) -> Result<Vec<f64>, String> {
    let mut b = b0;
    let (mut ll, mut g, mut h) = f(&b);
    if !ll.is_finite() {
        return Err(rc_err(430, "initial values not feasible"));
    }
    for _ in 0..300 {
        let hinv = solve_inv(&h);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum()).collect();
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..50 {
            let nb: Vec<f64> = b.iter().zip(&step).map(|(x, s)| x + lambda * s).collect();
            let r = f(&nb);
            if r.0.is_finite() && r.0 >= ll - 1e-12 * ll.abs().max(1.0) {
                accepted = Some((nb, r));
                break;
            }
            lambda /= 2.0;
        }
        let Some((nb, r)) = accepted else { break };
        let change = (r.0 - ll).abs();
        let bchange = b.iter().zip(&nb).map(|(a, c)| (a - c).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        ll = r.0;
        g = r.1;
        h = r.2;
        let hi = solve_inv(&h);
        let gn: f64 = (0..k).map(|i| (0..k).map(|j| g[i] * hi[(i, j)] * g[j]).sum::<f64>()).sum();
        if (change < 1e-11 * ll.abs().max(1.0) && bchange < 1e-9) || gn.abs() < 1e-16 {
            break;
        }
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// Specification of an estimation
// ---------------------------------------------------------------------------

struct Spec {
    kind: Kind,
    cmd: String,
    title: String,
    names: Vec<String>,
    /// which names are base levels / omitted (fixed at zero)
    omitted: Vec<bool>,
    is_base: Vec<bool>,
    /// map from estimator parameter to position in `names`
    map: Vec<usize>,
    depvar: String,
    /// rows of the dataset usable by the estimator (model variables nonmissing)
    ok: Vec<bool>,
    data_of: Box<dyn Fn(&[usize]) -> Result<Data, String>>,
    /// display rows: (group header, label) per name
    labels: Vec<(String, String)>,
    eform: Option<&'static str>,
    legend: Vec<String>,
    /// tabulate: rows, columns and labels
    tab: Option<TabInfo>,
    extra: Vec<(String, String)>,
    /// obs counts per statistic name (e(_N))
    stat_rows: Vec<Box<dyn Fn(usize) -> bool>>,
}

struct TabInfo {
    rvar: String,
    cvar: Option<String>,
    rlabels: Vec<String>,
    clabels: Vec<String>,
}

fn level_text(ds: &DataSet, var: &str, v: f64, labels: bool) -> String {
    let var_ = ds.get_var(var).unwrap();
    if labels && !var_.value_label_name.is_empty() {
        if let Some(l) = ds.value_labels.get(&var_.value_label_name).and_then(|vl| vl.lookup(v)) {
            return l.to_string();
        }
    }
    if is_missing(v) { ".".into() } else { fmt_g(v, 9) }
}

fn levels_of(ds: &DataSet, var: &str, mask: &[bool]) -> Vec<f64> {
    let v = ds.get_var(var).unwrap();
    let mut l: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| v.num(i)).collect();
    l.sort_by(f64::total_cmp);
    l.dedup();
    l
}

fn col_vals(ds: &DataSet, var: &str) -> Vec<f64> {
    let v = ds.get_var(var).unwrap();
    (0..ds.n_obs).map(|i| v.num(i)).collect()
}

/// over(varlist): group id per dataset row and group descriptions.
fn over_groups(ds: &DataSet, over: &[String], mask: &[bool]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let vals: Vec<Vec<f64>> = (0..ds.n_obs).map(|i| over.iter().map(|v| ds.get_var(v).unwrap().num(i)).collect()).collect();
    let mut levels: Vec<Vec<f64>> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| vals[i].clone()).collect();
    levels.sort_by(|a, b| a.iter().zip(b).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup();
    let gid = (0..ds.n_obs).map(|i| levels.iter().position(|l| *l == vals[i]).unwrap_or(0)).collect();
    (gid, levels)
}

fn build_spec(ds: &DataSet, st: &Store, cmd: &Command, name: &str, touse: &[bool]) -> Result<Spec, String> {
    let n_all = ds.n_obs;
    let labels_on = !cmd.has_opt("nolabel", 5);
    let base_spec = |kind: Kind, cmdname: &str, title: &str| Spec {
        kind,
        cmd: cmdname.into(),
        title: title.into(),
        names: vec![],
        omitted: vec![],
        is_base: vec![],
        map: vec![],
        depvar: String::new(),
        ok: vec![true; n_all],
        data_of: Box::new(|_| Err(String::new())),
        labels: vec![],
        eform: None,
        legend: vec![],
        tab: None,
        extra: vec![],
        stat_rows: vec![],
    };
    match name {
        "mean" | "total" | "proportion" => {
            let vars = varlist::expand(ds, &cmd.main)?;
            if vars.is_empty() {
                return Err(rc_err(100, "varlist required"));
            }
            for v in &vars {
                if ds.get_var(v).is_some_and(|x| x.var_type.is_string()) {
                    return Err(rc_err(109, format!("{v}: string variables not allowed")));
                }
            }
            let over = match cmd.opt("over", 2) {
                Some(o) => varlist::expand(ds, o.split(',').next().unwrap_or(""))?,
                None => vec![],
            };
            let mut ok = vec![true; n_all];
            let missing = name == "proportion" && cmd.has_opt("missing", 4);
            if !missing {
                util::markout(ds, &mut ok, &vars);
            }
            util::markout(ds, &mut ok, &over);
            let both: Vec<bool> = (0..n_all).map(|i| ok[i] && touse[i]).collect();
            let (gid, levels) = if over.is_empty() { (vec![0; n_all], vec![vec![]]) } else { over_groups(ds, &over, &both) };
            let ng = levels.len();
            let glab = |g: usize| levels[g].iter().zip(&over).map(|(l, o)| level_text(ds, o, *l, labels_on)).collect::<Vec<_>>().join("#");
            let gname = |g: usize| levels[g].iter().zip(&over).map(|(l, o)| format!("{}.{o}", fmt_g(*l, 12))).collect::<Vec<_>>().join("#");
            let kind = match name {
                "mean" => Kind::Mean,
                "total" => Kind::Total,
                _ => Kind::Prop,
            };
            let mut s = base_spec(
                kind,
                name,
                match kind {
                    Kind::Mean => "Mean estimation",
                    Kind::Total => "Total estimation",
                    _ => "Proportion estimation",
                },
            );
            // statistic columns: the variables, or level indicators for proportions
            let mut cols: Vec<(String, Box<dyn Fn(f64) -> f64>, String)> = Vec::new();
            for v in &vars {
                if kind == Kind::Prop {
                    for l in levels_of(ds, v, &both) {
                        let lt = if is_missing(l) { ".".to_string() } else { fmt_g(l, 12) };
                        cols.push((v.clone(), Box::new(move |x: f64| if x == l || (is_missing(x) && is_missing(l)) { 1.0 } else { 0.0 }), format!("{lt}.{v}")));
                    }
                } else {
                    cols.push((v.clone(), Box::new(|x: f64| x), v.clone()));
                }
            }
            let mut stats = Vec::new();
            for (c, (v, _, nm)) in cols.iter().enumerate() {
                for g in 0..ng {
                    stats.push((c, g));
                    let name = if over.is_empty() {
                        nm.clone()
                    } else if kind == Kind::Prop {
                        format!("{nm}@{}", gname(g))
                    } else {
                        format!("c.{v}@{}", gname(g))
                    };
                    s.names.push(name);
                    let head = if over.is_empty() {
                        if kind == Kind::Prop { v.clone() } else { String::new() }
                    } else if kind == Kind::Prop {
                        format!("{v}@{}", over.join("#"))
                    } else {
                        format!("c.{v}@{}", over.join("#"))
                    };
                    let lev_label = if kind == Kind::Prop {
                        let lv = ds.get_var(v).unwrap();
                        let first = (0..n_all).find(|&i| both[i] && (cols[c].1)(lv.num(i)) == 1.0);
                        first.map(|i| level_text(ds, v, lv.num(i), labels_on)).unwrap_or_default()
                    } else {
                        String::new()
                    };
                    let label = match (over.is_empty(), kind == Kind::Prop) {
                        (true, true) => lev_label,
                        (true, false) => v.clone(),
                        (false, true) => format!("{lev_label} {}", glab(g)),
                        (false, false) => glab(g),
                    };
                    s.labels.push((head, label));
                    let gid2 = gid.clone();
                    s.stat_rows.push(Box::new(move |i: usize| gid2[i] == g));
                }
            }
            let k = s.names.len();
            s.omitted = vec![false; k];
            s.is_base = vec![false; k];
            s.map = (0..k).collect();
            s.depvar = vars.join(" ");
            s.ok = ok;
            let colv: Vec<Vec<f64>> = cols.iter().map(|(v, f, _)| col_vals(ds, v).into_iter().map(|x| f(x)).collect()).collect();
            let total = kind == Kind::Total;
            s.data_of = Box::new(move |rows: &[usize]| {
                Ok(Data::Means {
                    cols: colv.iter().map(|c| rows.iter().map(|&i| if is_missing(c[i]) { 0.0 } else { c[i] }).collect()).collect(),
                    gid: rows.iter().map(|&i| gid[i]).collect(),
                    stats: stats.clone(),
                    total,
                })
            });
            if !over.is_empty() {
                s.extra.push(("over".into(), over.join(" ")));
                s.extra.push(("over_namelist".into(), (0..ng).map(gname).collect::<Vec<_>>().join(" ")));
                s.extra.push(("over_labels".into(), (0..ng).map(|g| format!("\"{}\"", glab(g))).collect::<Vec<_>>().join(" ")));
            }
            s.extra.push(("varlist".into(), vars.join(" ")));
            Ok(s)
        }
        "ratio" => {
            // ratio [name:] y/x [[name:] y/x ...] or (name: y/x)
            let text = cmd.main.replace(['(', ')'], " ");
            let toks: Vec<String> = text.split_whitespace().map(String::from).collect();
            let mut items: Vec<(String, String, String)> = Vec::new();
            let mut pending: Option<String> = None;
            for t in toks {
                if let Some(nm) = t.strip_suffix(':') {
                    pending = Some(nm.to_string());
                    continue;
                }
                let (y, x) = t.split_once('/').ok_or_else(|| rc_err(198, format!("invalid ratio specification: {t}")))?;
                let y = varlist::resolve_name(ds, y)?;
                let x = varlist::resolve_name(ds, x)?;
                let nm = pending.take().unwrap_or_else(|| format!("_ratio_{}", items.len() + 1));
                items.push((nm, y, x));
            }
            if items.is_empty() {
                return Err(rc_err(198, "ratio specification required"));
            }
            let mut s = base_spec(Kind::Ratio, "ratio", "Ratio estimation");
            let mut ok = vec![true; n_all];
            for (_, y, x) in &items {
                util::markout(ds, &mut ok, &[y.clone(), x.clone()]);
            }
            s.ok = ok;
            for (nm, y, x) in &items {
                s.names.push(nm.clone());
                s.labels.push((String::new(), nm.clone()));
                s.legend.push(format!("{:>17}: {y}/{x}", nm));
                s.stat_rows.push(Box::new(|_| true));
            }
            let k = items.len();
            s.omitted = vec![false; k];
            s.is_base = vec![false; k];
            s.map = (0..k).collect();
            let nums: Vec<Vec<f64>> = items.iter().map(|(_, y, _)| col_vals(ds, y)).collect();
            let dens: Vec<Vec<f64>> = items.iter().map(|(_, _, x)| col_vals(ds, x)).collect();
            s.data_of = Box::new(move |rows: &[usize]| {
                let pick = |c: &Vec<f64>| rows.iter().map(|&i| if is_missing(c[i]) { 0.0 } else { c[i] }).collect();
                Ok(Data::Ratios { num: nums.iter().map(pick).collect(), den: dens.iter().map(pick).collect() })
            });
            s.extra.push(("namelist".into(), items.iter().map(|x| x.0.clone()).collect::<Vec<_>>().join(" ")));
            Ok(s)
        }
        "tabulate" => {
            let vars = varlist::expand(ds, &cmd.main)?;
            if vars.is_empty() || vars.len() > 2 {
                return Err(rc_err(198, "one or two variables required"));
            }
            let mut ok = vec![true; n_all];
            if !cmd.has_opt("missing", 4) {
                util::markout(ds, &mut ok, &vars);
            }
            let both: Vec<bool> = (0..n_all).map(|i| ok[i] && touse[i]).collect();
            let rl = levels_of(ds, &vars[0], &both);
            let cl = if vars.len() == 2 { levels_of(ds, &vars[1], &both) } else { vec![SYSMISS] };
            let rv = col_vals(ds, &vars[0]);
            let cv = if vars.len() == 2 { col_vals(ds, &vars[1]) } else { vec![SYSMISS; n_all] };
            let mut s = base_spec(Kind::Tab, "tabulate", if vars.len() == 2 { "Two-way tabulation" } else { "One-way tabulation" });
            s.ok = ok;
            let mut cells = Vec::new();
            for (r, &a) in rl.iter().enumerate() {
                for (c, &b) in cl.iter().enumerate() {
                    cells.push((a, b));
                    s.names.push(format!("p{}{}", r + 1, c + 1));
                    s.labels.push((String::new(), String::new()));
                    s.stat_rows.push(Box::new(|_| true));
                }
            }
            let k = cells.len();
            s.omitted = vec![false; k];
            s.is_base = vec![false; k];
            s.map = (0..k).collect();
            s.tab = Some(TabInfo {
                rvar: vars[0].clone(),
                cvar: vars.get(1).cloned(),
                rlabels: rl.iter().map(|&l| level_text(ds, &vars[0], l, labels_on)).collect(),
                clabels: if vars.len() == 2 { cl.iter().map(|&l| level_text(ds, &vars[1], l, labels_on)).collect() } else { vec![] },
            });
            let same = |x: f64, l: f64| x == l || (is_missing(x) && is_missing(l));
            let oneway = vars.len() == 1;
            s.data_of = Box::new(move |rows: &[usize]| {
                let cols = cells.iter().map(|&(a, b)| rows.iter().map(|&i| if same(rv[i], a) && (oneway || same(cv[i], b)) { 1.0 } else { 0.0 }).collect()).collect();
                Ok(Data::Means { cols, gid: vec![0; rows.len()], stats: (0..k).map(|c| (c, 0)).collect(), total: false })
            });
            s.depvar = vars.join(" ");
            Ok(s)
        }
        _ => {
            // regression models
            let fam = match name {
                "regress" => Family::Regress,
                "logit" | "logistic" => Family::Logit,
                "probit" => Family::Probit,
                "poisson" => Family::Poisson,
                "ologit" => Family::Ologit,
                "oprobit" => Family::Oprobit,
                _ => Family::Mlogit,
            };
            let ordered = matches!(fam, Family::Ologit | Family::Oprobit);
            let mut words = cmd.main.split_whitespace();
            let dep0 = words.next().ok_or_else(|| rc_err(100, "varlist required"))?;
            let dep = varlist::resolve_name(ds, dep0)?;
            let indep = cmd.main.trim()[dep0.len()..].trim().to_string();
            let mut c2 = cmd.clone();
            c2.weight = None;
            let d = est::build(ds, st, &c2, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: ordered || cmd.has_opt("noconstant", 3), weights: "", extra: vec![] })?;
            let mut ok = vec![false; n_all];
            for &i in &d.rows {
                ok[i] = true;
            }
            let title = match fam {
                Family::Regress => "Linear regression",
                Family::Logit => "Logistic regression",
                Family::Probit => "Probit regression",
                Family::Poisson => "Poisson regression",
                Family::Ologit => "Ordered logistic regression",
                Family::Oprobit => "Ordered probit regression",
                Family::Mlogit => "Multinomial logistic regression",
            };
            let mut s = base_spec(Kind::Model, name, title);
            s.depvar = dep.clone();
            s.ok = ok.clone();
            // outcome categories
            let both: Vec<bool> = (0..n_all).map(|i| ok[i] && touse[i]).collect();
            let cats = if matches!(fam, Family::Ologit | Family::Oprobit | Family::Mlogit) { levels_of(ds, &dep, &both) } else { vec![] };
            if (ordered || fam == Family::Mlogit) && cats.len() < 2 {
                return Err(rc_err(2000, "outcome does not vary"));
            }
            let base = if fam == Family::Mlogit {
                match util::opt_num(cmd, "baseoutcome", 1)? {
                    Some(v) => cats.iter().position(|&c| c == v).ok_or_else(|| rc_err(198, "baseoutcome() not found"))?,
                    None => {
                        let yv = col_vals(ds, &dep);
                        (0..cats.len()).max_by_key(|&c| ((0..n_all).filter(|&i| both[i] && yv[i] == cats[c]).count(), std::cmp::Reverse(c))).unwrap()
                    }
                }
            } else {
                0
            };
            // columns: active design columns (computed over the estimation rows later)
            let act = d.active();
            let kx = act.len();
            match fam {
                Family::Mlogit => {
                    let eq_name = |v: f64| {
                        let lab = ds.get_var(&dep).and_then(|var| ds.value_labels.get(&var.value_label_name)).and_then(|vl| vl.lookup(v)).map(str::to_string);
                        lab.filter(|l| !l.is_empty() && !l.contains(|c: char| c.is_whitespace() || c == ':' || c == '.')).unwrap_or_else(|| fmt_g(v, 12))
                    };
                    let mut map = vec![0; kx * (cats.len() - 1)];
                    for c in 0..cats.len() {
                        let eq = eq_name(cats[c]);
                        for j in 0..d.k() {
                            s.names.push(format!("{eq}:{}", d.names[j]));
                            s.labels.push((eq.clone(), d.names[j].clone()));
                            if c == base {
                                s.omitted.push(true);
                                s.is_base.push(true);
                            } else {
                                let e = (0..cats.len()).filter(|&x| x != base).position(|x| x == c).unwrap();
                                match act.iter().position(|&a| a == j) {
                                    Some(a) => {
                                        map[e * kx + a] = s.names.len() - 1;
                                        s.omitted.push(false);
                                    }
                                    None => s.omitted.push(true),
                                }
                                s.is_base.push(d.is_base[j]);
                            }
                        }
                    }
                    s.map = map;
                    s.extra.push(("baseout".into(), fmt_g(cats[base], 16)));
                    s.extra.push(("eqnames".into(), cats.iter().map(|&c| eq_name(c)).collect::<Vec<_>>().join(" ")));
                    if cmd.has_opt("rrr", 2) {
                        s.eform = Some("RRR");
                    }
                }
                _ => {
                    for j in 0..d.k() {
                        s.names.push(d.names[j].clone());
                        s.labels.push((String::new(), d.names[j].clone()));
                        s.omitted.push(d.omitted[j]);
                        s.is_base.push(d.is_base[j]);
                    }
                    s.map = act.clone();
                    if ordered {
                        for c in 0..cats.len() - 1 {
                            s.map.push(s.names.len());
                            s.names.push(format!("/:cut{}", c + 1));
                            s.labels.push((String::new(), format!("cut{}", c + 1)));
                            s.omitted.push(false);
                            s.is_base.push(false);
                        }
                    }
                    if name == "logistic" || (fam == Family::Logit && cmd.has_opt("or", 2)) {
                        s.eform = Some("Odds ratio");
                    }
                    if fam == Family::Poisson && cmd.has_opt("irr", 3) {
                        s.eform = Some("IRR");
                    }
                }
            }
            for _ in 0..s.names.len() {
                s.stat_rows.push(Box::new(|_| true));
            }
            if !cats.is_empty() {
                s.extra.push(("k_cat".into(), cats.len().to_string()));
                s.extra.push(("cat".into(), cats.iter().map(|c| fmt_g(*c, 16)).collect::<Vec<_>>().join(" ")));
            }
            // offset / exposure
            let mut off = vec![0.0; n_all];
            if let Some(o) = cmd.opt("offset", 3) {
                let c = col_vals(ds, &varlist::resolve_name(ds, o.trim())?);
                for i in 0..n_all {
                    off[i] += c[i];
                }
            }
            if let Some(o) = cmd.opt("exposure", 1) {
                let c = col_vals(ds, &varlist::resolve_name(ds, o.trim())?);
                for i in 0..n_all {
                    off[i] += c[i].ln();
                }
            }
            let yv = col_vals(ds, &dep);
            let pos: std::collections::HashMap<usize, usize> = d.rows.iter().enumerate().map(|(t, &i)| (i, t)).collect();
            let cols: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
            s.data_of = Box::new(move |rows: &[usize]| {
                let pick = |i: usize, c: &Vec<f64>| pos.get(&i).map_or(0.0, |&t| c[t]);
                let x: Vec<Vec<f64>> = cols.iter().map(|c| rows.iter().map(|&i| pick(i, c)).collect()).collect();
                let y: Vec<f64> = rows.iter().map(|&i| if pos.contains_key(&i) { yv[i] } else { 0.0 }).collect();
                let yidx: Vec<usize> = y.iter().map(|v| cats.iter().position(|c| c == v).unwrap_or(0)).collect();
                if fam == Family::Logit || fam == Family::Probit {
                    // binary outcomes: any nonzero value is a success
                }
                if fam == Family::Poisson && rows.iter().any(|&i| pos.contains_key(&i) && yv[i] < 0.0) {
                    return Err(rc_err(459, format!("{} must be greater than or equal to zero", "depvar")));
                }
                let offset = rows.iter().map(|&i| if pos.contains_key(&i) && !is_missing(off[i]) { off[i] } else { 0.0 }).collect();
                Ok(Data::Model { fam, x, y, ncat: cats.len(), base, yidx, offset })
            });
            Ok(s)
        }
    }
}

// ---------------------------------------------------------------------------
// The prefix
// ---------------------------------------------------------------------------

/// `svy [vcetype] [, options] : command`.  `head` is the text between `svy`
/// and the colon.
pub fn run(ds: &mut DataSet, st: &mut Store, head: &str, command: &str) -> Result<String, String> {
    let set = current(ds)?;
    let (lead, opts): (String, Opts) = match find_top_level(head, ',') {
        Some(i) => (head[..i].trim().to_string(), split_options(&head[i + 1..])),
        None => (head.trim().to_string(), Vec::new()),
    };
    let mut method = set.vce.clone();
    if !lead.is_empty() {
        method = vce_method(&lead).ok_or_else(|| rc_err(198, format!("{lead} invalid vcetype")))?.to_string();
    }
    if let Some(v) = arg(&opts, "vce", 3) {
        method = vce_method(v.split(',').next().unwrap_or("")).ok_or_else(|| rc_err(198, format!("vce({v}) not allowed")))?.to_string();
    }
    let level = match arg(&opts, "level", 1) {
        Some(l) => l.trim().parse::<f64>().map_err(|_| rc_err(198, "level() invalid"))?,
        None => st.level,
    };
    let cmd = parse_line(command).map_err(|e| rc_err(198, e.message))?;
    let name = resolve(&cmd.verb).ok_or_else(|| rc_err(322, format!("{} is not supported by svy", cmd.verb)))?;
    if cmd.weight.is_some() {
        return Err(rc_err(101, "weights not allowed with the svy prefix; the weights are specified using svyset"));
    }
    let level = if cmd.has_opt("level", 1) { util::level(&cmd, st)? } else { level };
    let mse = set.mse || has(&opts, "mse", 3);

    // sample: if/in of the command and the design variables
    let n_all = ds.n_obs;
    let mut touse = util::touse(ds, st, &cmd)?;
    let usable = DesignData::usable(ds, &set);
    let reps: Vec<String> = if method == "linearized" { vec![] } else { set.rep_weights(&method).to_vec() };
    let mut rep_ok = vec![true; n_all];
    util::markout(ds, &mut rep_ok, &reps);
    for i in 0..n_all {
        touse[i] = touse[i] && usable[i] && rep_ok[i];
    }
    // subpopulation
    let (sub, subpop_text) = match arg(&opts, "subpop", 3) {
        Some(spec) => (subpop_mask(ds, st, &spec)?, Some(spec)),
        None => (vec![true; n_all], None),
    };
    let spec = build_spec(ds, st, &cmd, name, &touse)?;
    // observations with missing values are left out of the design entirely,
    // also outside the subpopulation
    let design_mask: Vec<bool> = (0..n_all).map(|i| touse[i] && spec.ok[i]).collect();
    let rows = util::rows(&design_mask);
    let in_est: Vec<bool> = rows.iter().map(|&i| sub[i] && spec.ok[i]).collect();
    if !in_est.iter().any(|&x| x) {
        return Err(rc_err(2000, "no observations"));
    }
    let dd = DesignData::new(ds, &set, &rows);
    let w: Vec<f64> = (0..rows.len()).map(|t| if in_est[t] { dd.w[t] } else { 0.0 }).collect();
    let data = (spec.data_of)(&rows)?;
    let fit = data.fit(&w)?;
    let kp = fit.b.len();
    let (h, npsu) = dd.counts();

    // variance
    let mut nreps = 0usize;
    let mut misreps = 0usize;
    let mut dots = String::new();
    let (vpar, df) = if method == "linearized" {
        (dd.variance(&fit.z), set.dof.unwrap_or(dd.df()))
    } else {
        let rep_w: Vec<(Vec<f64>, f64, u64)> = replicate_weights(ds, &set, &method, &dd, &rows, &in_est)?;
        nreps = rep_w.len();
        let mut bs = Vec::new();
        let mut marks = Vec::new();
        for (wr, _, _) in &rep_w {
            match data.fit(wr) {
                Ok(f) if f.b.iter().all(|x| x.is_finite() && !is_missing(*x)) => {
                    bs.push(Some(f.b));
                    marks.push('.');
                }
                _ => {
                    bs.push(None);
                    marks.push('x');
                }
            }
        }
        misreps = bs.iter().filter(|b| b.is_none()).count();
        let title = match method.as_str() {
            "brr" => "BRR",
            "jackknife" => "Jackknife",
            "bootstrap" => "Bootstrap",
            _ => "SDR",
        };
        if !has(&opts, "nodots", 5) {
            dots = dots_line(&format!("{title} replications"), nreps, &marks);
        }
        let v = replicate_variance(&method, &set, &rep_w, &bs, &fit.b, mse, kp);
        let rdf = if let Some(d) = set.dof {
            d
        } else if reps.is_empty() {
            dd.df()
        } else if method == "jackknife" {
            let mut hs: Vec<u64> = set.jk.iter().map(|x| x.0.to_bits()).collect();
            hs.sort();
            hs.dedup();
            (nreps as f64 - hs.len() as f64).max(1.0)
        } else {
            (nreps as f64 - 1.0).max(1.0)
        };
        (Some(v), rdf)
    };
    // weighted counts of a tabulation: variance of the estimated totals
    let vtot = match (&data, spec.kind == Kind::Tab && cmd.has_opt("count", 2) && method == "linearized") {
        (Data::Means { cols, gid, stats, .. }, true) => {
            let dt = Data::Means { cols: cols.clone(), gid: gid.clone(), stats: stats.clone(), total: true };
            dt.fit(&w).ok().and_then(|f| dd.variance(&f.z))
        }
        _ => None,
    };
    let single_note = vpar.is_none();
    let vpar = vpar.unwrap_or_else(|| {
        let mut m = Matrix::zeros(kp, kp);
        for a in 0..kp {
            for b in 0..kp {
                m[(a, b)] = SYSMISS;
            }
        }
        m
    });

    // srs variances for design effects
    let mhat: f64 = w.iter().sum();
    let m = in_est.iter().filter(|&&x| x).count() as f64;
    let mut vsrs = Matrix::zeros(kp, kp);
    {
        let zbar: Vec<f64> = (0..kp).map(|a| fit.z.iter().map(|z| z[a]).sum::<f64>() / mhat).collect();
        for t in 0..rows.len() {
            if w[t] == 0.0 {
                continue;
            }
            let u: Vec<f64> = (0..kp).map(|a| fit.z[t][a] / w[t] - zbar[a]).collect();
            for a in 0..kp {
                for b in 0..kp {
                    vsrs[(a, b)] += w[t] * u[a] * u[b];
                }
            }
        }
        vsrs = vsrs.scale(mhat / (mhat - 1.0) * mhat / m);
    }
    // without sampling weights or an fpc the srs comparison is with replacement
    let fsrs = if set.wvar.is_some() && mhat > 0.0 { (1.0 - m / mhat).max(0.0) } else { 1.0 };

    // expand to the full coefficient vector
    let k = spec.names.len();
    let mut b = vec![0.0; k];
    let mut v = Matrix::zeros(k, k);
    let mut vs = Matrix::zeros(k, k);
    for a in 0..kp {
        b[spec.map[a]] = fit.b[a];
        for c in 0..kp {
            v[(spec.map[a], spec.map[c])] = vpar[(a, c)];
            vs[(spec.map[a], spec.map[c])] = vsrs[(a, c)];
        }
    }
    let vswor = vs.scale(fsrs);
    let deff: Vec<f64> = (0..k).map(|j| if vswor[(j, j)] > 0.0 && !is_missing(v[(j, j)]) { v[(j, j)] / vswor[(j, j)] } else { SYSMISS }).collect();
    let deft: Vec<f64> = (0..k).map(|j| if vs[(j, j)] > 0.0 && !is_missing(v[(j, j)]) { (v[(j, j)] / vs[(j, j)]).sqrt() } else { SYSMISS }).collect();

    // model test: adjusted Wald F
    let test_idx: Vec<usize> = if spec.kind == Kind::Model {
        (0..k).filter(|&j| !spec.omitted[j] && !spec.names[j].ends_with("_cons") && !spec.names[j].starts_with("/:")).collect()
    } else {
        vec![]
    };
    let mut ftest: Option<(f64, f64, f64)> = None;
    if !test_idx.is_empty() && !single_note {
        let q0 = test_idx.len();
        let mut sub_v = Matrix::zeros(q0, q0);
        for (a, &i) in test_idx.iter().enumerate() {
            for (c, &j) in test_idx.iter().enumerate() {
                sub_v[(a, c)] = v[(i, j)];
            }
        }
        let (inv, dropped) = sub_v.sym_inverse(1e-12);
        let q = (q0 - dropped.len()) as f64;
        let wald: f64 = test_idx.iter().enumerate().map(|(a, &i)| test_idx.iter().enumerate().map(|(c, &j)| b[i] * inv[(a, c)] * b[j]).sum::<f64>()).sum();
        let d2 = df - q + 1.0;
        if q > 0.0 && d2 > 0.0 {
            let f = (df - q + 1.0) / (df * q) * wald;
            ftest = Some((f, q, d2));
        }
    }

    // ---- output
    let vcetype = match method.as_str() {
        "linearized" => "Linearized",
        "brr" => "BRR *",
        "jackknife" => "Jackknife *",
        "bootstrap" => "Bootstrap",
        _ => "SDR *",
    };
    let nobs = rows.len();
    let npop: f64 = dd.w.iter().sum();
    let mut out = Vec::new();
    out.push(format!("(running {} on estimation sample)", spec.cmd));
    if !dots.is_empty() {
        out.push(String::new());
        out.push(dots);
    }
    out.push(String::new());
    let model = spec.kind == Kind::Model;
    if !has(&opts, "noheader", 5) {
        if spec.kind != Kind::Tab {
            out.push(format!("Survey: {}", spec.title));
            out.push(String::new());
        }
        let mut right: Vec<(String, String)> = vec![("Number of obs".into(), util::commas(nobs))];
        if subpop_text.is_some() {
            right.push(("Subpop. no. obs".into(), util::commas(m as usize)));
        }
        right.push(("Population size".into(), commas_f(npop)));
        if subpop_text.is_some() {
            right.push(("Subpop. size".into(), commas_f(mhat)));
        }
        if nreps > 0 {
            right.push(("Replications".into(), util::commas(nreps)));
        }
        right.push(("Design df".into(), fmt_g(df, 9)));
        if let Some((f, q, d2)) = ftest {
            right.push((format!("F({}, {})", fmt_g(q, 9), fmt_g(d2, 9)), format!("{:.2}", f)));
            right.push(("Prob > F".into(), format!("{:.4}", dist::ftail(q, d2, f))));
        } else if model && !test_idx.is_empty() {
            right.push((format!("F({}, {})", test_idx.len(), fmt_g(df - test_idx.len() as f64 + 1.0, 9)), ".".into()));
            right.push(("Prob > F".into(), ".".into()));
        }
        if let Some(r2) = fit.r2 {
            right.push(("R-squared".into(), format!("{:.4}", r2)));
        }
        let left: Vec<String> = if method == "linearized" || reps.is_empty() {
            let (hs, us) = (util::commas(h), util::commas(npsu));
            let wv = hs.len().max(us.len());
            vec![format!("Number of strata = {hs:>wv$}"), format!("Number of PSUs   = {us:>wv$}")]
        } else {
            vec![]
        };
        let (col, lw, vw) = if model { (49, 18, 10) } else { (34, 16, 11) };
        for (i, (l, v)) in right.iter().enumerate() {
            let lt = left.get(i).cloned().unwrap_or_default();
            out.push(format!("{:<col$}{:<lw$}= {:>vw$}", lt, l, v));
        }
        out.push(String::new());
    }
    if !spec.legend.is_empty() {
        out.extend(spec.legend.iter().cloned());
        out.push(String::new());
    }
    let table_lines: Vec<String>;
    match spec.kind {
        Kind::Model => {
            let ts = TableSpec { depvar: &spec.depvar, names: &spec.names, b: &b, v: &v, omitted: &spec.omitted, is_base: &spec.is_base, df: Some(df), level, eform: spec.eform, coef_title: "Coefficient" };
            let mut t = est::coef_table(&ts);
            if t.len() > 1 {
                let bar = t[1].find('|').unwrap_or(13);
                t[1] = t[1].replacen("Std. err.", "std. err.", 1);
                t.insert(1, format!("{}| {:>22}", " ".repeat(bar), vcetype));
            }
            if name == "mlogit" {
                if let Some(bo) = spec.extra.iter().find(|x| x.0 == "baseout") {
                    let _ = bo;
                    let eqs: Vec<String> = spec.extra.iter().find(|x| x.0 == "eqnames").map(|x| x.1.split(' ').map(String::from).collect()).unwrap_or_default();
                    let bi = spec.names.iter().position(|_| true).map(|_| ()).and_then(|_| spec.is_base.iter().zip(&spec.omitted).position(|(a, o)| *a && *o)).map(|p| spec.labels[p].0.clone());
                    if let Some(beq) = bi {
                        let base_line = format!("{:<12} |  (base outcome)", abbrev(&beq, 12));
                        let sep = format!("{}+{}", "-".repeat(13), "-".repeat(64));
                        let pos = eqs.iter().position(|e| *e == beq).unwrap_or(0);
                        let next = eqs.get(pos + 1).map(|e| format!("{:<12} |", abbrev(e, 12)));
                        match next.and_then(|l| t.iter().position(|x| *x == l)) {
                            Some(i) => {
                                t.insert(i, sep);
                                t.insert(i, base_line);
                            }
                            None => {
                                let end = t.len() - 1;
                                t.insert(end, base_line);
                                t.insert(end, sep);
                            }
                        }
                    }
                }
            }
            table_lines = t;
        }
        Kind::Tab => {
            table_lines = tab_output(&spec, &b, &v, vtot.as_ref(), &cmd, df, m, mhat, &mut Vec::new());
        }
        _ => {
            table_lines = simple_table(&spec, &b, &v, df, level, vcetype, !cmd.opt("citype", 4).is_some_and(|c| c.trim().starts_with('w')));
        }
    }
    if !has(&opts, "notable", 5) {
        out.extend(table_lines);
    }
    if single_note {
        out.push("Note: Missing standard errors because of stratum with single sampling unit.".into());
    }
    if misreps > 0 {
        out.push(format!("Note: One or more parameters could not be estimated in {misreps} replicate{};", if misreps == 1 { "" } else { "s" }));
        out.push("      standard-error estimates include only complete replications.".into());
    }

    // ---- e()
    let mut e = Results::new();
    e.set("N", nobs as f64);
    e.set("N_pop", npop);
    e.set("N_strata", h as f64);
    e.set("N_psu", npsu as f64);
    e.set("df_r", df);
    e.set("k_eq", 1.0);
    e.set("level", level);
    e.set("rank", (0..k).filter(|&j| !spec.omitted[j]).count() as f64);
    if subpop_text.is_some() {
        e.set("N_sub", m);
        e.set("N_subpop", mhat);
        e.set_str("subpop", subpop_text.clone().unwrap_or_default());
    }
    if nreps > 0 {
        e.set("N_reps", nreps as f64);
        e.set("N_misreps", misreps as f64);
    }
    if let Some((f, q, d2)) = ftest {
        e.set("F", f);
        e.set("df_m", q);
        e.set("p", dist::ftail(q, d2, f));
    }
    if let Some(r2) = fit.r2 {
        e.set("r2", r2);
    }
    e.set_str("prefix", "svy");
    e.set_str("cmd", if name == "logistic" { "logistic" } else { name });
    e.set_str("cmdname", name);
    e.set_str("command", command.trim());
    e.set_str("title", format!("Survey: {}", spec.title));
    e.set_str("depvar", spec.depvar.clone());
    e.set_str("vce", method.clone());
    e.set_str("vcetype", vcetype);
    if mse {
        e.set_str("mse", "mse");
    }
    if let (Some(t), Some(wv)) = (&set.wtype, &set.wvar) {
        e.set_str("wtype", t.clone());
        e.set_str("wvar", wv.clone());
        e.set_str("wexp", format!("= {wv}"));
    }
    for (s, stg) in set.stages.iter().enumerate() {
        e.set_str(&format!("su{}", s + 1), stg.su.clone().unwrap_or_else(|| "_n".into()));
        if let Some(x) = &stg.strata {
            e.set_str(&format!("strata{}", s + 1), x.clone());
        }
        if let Some(x) = &stg.fpc {
            e.set_str(&format!("fpc{}", s + 1), x.clone());
        }
    }
    e.set("stages", set.stages.len() as f64);
    e.set_str("singleunit", set.singleunit.clone());
    e.set_str("properties", "b V");
    for (key, val) in &spec.extra {
        if key == "k_cat" {
            e.set("k_cat", val.parse().unwrap_or(0.0));
        } else if key != "eqnames" && key != "cat" && key != "baseout" {
            e.set_str(key, val.clone());
        } else if key == "baseout" {
            e.set("baseout", val.parse().unwrap_or(0.0));
        }
    }
    let names = spec.names.clone();
    let rowv = |vals: &[f64]| Matrix::row_vector(vals).with_names(vec!["y1".into()], names.clone());
    e.set_mat("b", rowv(&b));
    e.set_mat("V", v.clone().with_names(names.clone(), names.clone()));
    e.set_mat("V_srs", vswor.with_names(names.clone(), names.clone()));
    e.set_mat("V_srswr", vs.with_names(names.clone(), names.clone()));
    e.set_mat("deff", rowv(&deff));
    e.set_mat("deft", rowv(&deft));
    // observation counts per statistic
    let counts: Vec<f64> = (0..k)
        .map(|j| {
            let f = spec.stat_rows.get(j);
            rows.iter().zip(&in_est).filter(|(i, e)| **e && f.map_or(true, |f| f(**i))).count() as f64
        })
        .collect();
    e.set_mat("_N", rowv(&counts));
    if spec.kind == Kind::Tab {
        let mut rr = Vec::new();
        tab_output(&spec, &b, &v, vtot.as_ref(), &cmd, df, m, mhat, &mut rr);
        for (kk, val) in rr {
            e.set(&kk, val);
        }
        if let Some(t) = &spec.tab {
            e.set("r", t.rlabels.len() as f64);
            e.set("c", t.clabels.len().max(1) as f64);
            e.set_str("rowvar", t.rvar.clone());
            if let Some(c) = &t.cvar {
                e.set_str("colvar", c.clone());
            }
        }
    }
    e.sample = Some(design_mask);
    st.e = e;
    Ok(out.join("\n"))
}

fn commas_f(x: f64) -> String {
    let r = (x * 1e4).round() / 1e4;
    if (r - r.round()).abs() < 1e-9 && r.abs() < 1e15 {
        let s = util::commas(r.round().abs() as usize);
        return if r < 0.0 { format!("-{s}") } else { s };
    }
    let whole = util::commas(r.trunc().abs() as usize);
    let frac = format!("{:.4}", r.fract().abs());
    let frac = frac.trim_start_matches('0').trim_end_matches('0');
    format!("{}{whole}{frac}", if r < 0.0 { "-" } else { "" })
}

fn dots_line(title: &str, reps: usize, marks: &[char]) -> String {
    let mut s = format!("{title} ({}): ", util::commas(reps));
    for (i, c) in marks.iter().enumerate() {
        if (i + 1) % 10 == 0 && *c == '.' {
            s.push_str(&util::commas(i + 1));
        } else {
            s.push(*c);
        }
    }
    s.push_str(" done");
    s
}

fn subpop_mask(ds: &DataSet, st: &Store, spec: &str) -> Result<Vec<bool>, String> {
    let n = ds.n_obs;
    let spec = spec.trim();
    let (var, cond) = match spec.find(" if ").map(|i| (spec[..i].trim(), Some(spec[i + 4..].trim()))) {
        Some(x) => x,
        None => match spec.strip_prefix("if ") {
            Some(c) => ("", Some(c.trim())),
            None => (spec, None),
        },
    };
    let mut mask = vec![true; n];
    if !var.is_empty() {
        let v = varlist::resolve_name(ds, var)?;
        let c = col_vals(ds, &v);
        for i in 0..n {
            mask[i] = !is_missing(c[i]) && c[i] != 0.0;
        }
    }
    if let Some(c) = cond {
        let e = parse_expression(c).map_err(|e| rc_err(198, e.message))?;
        let all = vec![true; n];
        let vals = util::eval_col(ds, st, &e, &all)?;
        for i in 0..n {
            mask[i] = mask[i] && !is_missing(vals[i]) && vals[i] != 0.0;
        }
    }
    Ok(mask)
}

/// Sylvester Hadamard matrix of order 2^p.
fn hadamard(order: usize) -> Vec<Vec<i8>> {
    let mut h = vec![vec![1i8]];
    while h.len() < order {
        let n = h.len();
        let mut g = vec![vec![0i8; 2 * n]; 2 * n];
        for i in 0..n {
            for j in 0..n {
                g[i][j] = h[i][j];
                g[i][j + n] = h[i][j];
                g[i + n][j] = h[i][j];
                g[i + n][j + n] = -h[i][j];
            }
        }
        h = g;
    }
    h
}

/// Replicate weights over the design rows: (weights, scale, stratum key).
fn replicate_weights(ds: &DataSet, set: &SvySet, method: &str, dd: &DesignData, rows: &[usize], in_est: &[bool]) -> Result<Vec<(Vec<f64>, f64, u64)>, String> {
    let n = rows.len();
    let given = set.rep_weights(method);
    if !given.is_empty() {
        let mut out = Vec::new();
        for (r, v) in given.iter().enumerate() {
            let c = crate::svy::key_col(ds, v);
            let wr: Vec<f64> = (0..n).map(|t| if in_est[t] { c[rows[t]] } else { 0.0 }).collect();
            let (h, mult) = if method == "jackknife" { set.jk.get(r).copied().unwrap_or((1.0, None)) } else { (1.0, None) };
            out.push((wr, mult.unwrap_or(SYSMISS), h.to_bits()));
        }
        return Ok(out);
    }
    // replicates from the design (first stage)
    let (strata, su, fpc) = &dd.stages[0];
    let mut groups: std::collections::BTreeMap<u64, std::collections::BTreeMap<u64, Vec<usize>>> = Default::default();
    for t in 0..n {
        let u = su.as_ref().map_or(t as u64, |c| c[t].to_bits());
        groups.entry(strata[t].to_bits()).or_default().entry(u).or_default().push(t);
    }
    let base: Vec<f64> = (0..n).map(|t| if in_est[t] { dd.w[t] } else { 0.0 }).collect();
    match method {
        "jackknife" => {
            let mut out = Vec::new();
            for (hk, units) in &groups {
                let nh = units.len();
                if nh < 2 {
                    continue;
                }
                let first = units.values().next().unwrap()[0];
                let f = fpc.as_ref().map_or(0.0, |c| {
                    let x = c[first];
                    if is_missing(x) { 0.0 } else if x <= 1.0 { x } else { nh as f64 / x }
                });
                for (uk, _) in units.iter() {
                    let mut wr = base.clone();
                    for (uk2, rws) in units.iter() {
                        for &t in rws {
                            wr[t] = if uk2 == uk { 0.0 } else { base[t] * nh as f64 / (nh as f64 - 1.0) };
                        }
                    }
                    out.push((wr, (1.0 - f) * (nh as f64 - 1.0) / nh as f64, *hk));
                }
            }
            Ok(out)
        }
        "brr" => {
            if groups.values().any(|u| u.len() != 2) {
                return Err(rc_err(460, "BRR requires exactly two sampling units per stratum"));
            }
            let hn = groups.len();
            let mut order = 1;
            while order < hn + 1 {
                order *= 2;
            }
            let had = hadamard(order);
            let mut out = Vec::new();
            for r in 0..order {
                let mut wr = base.clone();
                for (hidx, units) in groups.values().enumerate() {
                    let sign = had[r][hidx + 1];
                    for (u, rws) in units.values().enumerate() {
                        let chosen = (u == 0) == (sign > 0);
                        for &t in rws {
                            wr[t] = base[t] * if chosen { 2.0 - set.fay } else { set.fay };
                        }
                    }
                }
                out.push((wr, 1.0, 0));
            }
            Ok(out)
        }
        _ => Err(rc_err(111, format!("{method} replicate weights not set; see svyset"))),
    }
}

fn replicate_variance(method: &str, set: &SvySet, rep_w: &[(Vec<f64>, f64, u64)], bs: &[Option<Vec<f64>>], b: &[f64], mse: bool, k: usize) -> Matrix {
    let mut v = Matrix::zeros(k, k);
    let ok: Vec<(usize, &Vec<f64>)> = bs.iter().enumerate().filter_map(|(r, x)| x.as_ref().map(|x| (r, x))).collect();
    let r = ok.len() as f64;
    if ok.is_empty() {
        return v;
    }
    let add = |v: &mut Matrix, x: &[f64], c: &[f64], s: f64| {
        for a in 0..k {
            for bb in 0..k {
                v[(a, bb)] += s * (x[a] - c[a]) * (x[bb] - c[bb]);
            }
        }
    };
    let mean_of = |idx: &[usize]| -> Vec<f64> { (0..k).map(|a| idx.iter().map(|&i| ok[i].1[a]).sum::<f64>() / idx.len() as f64).collect() };
    match method {
        "jackknife" => {
            // by stratum
            let mut strata: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
            for (i, (rr, _)) in ok.iter().enumerate() {
                strata.entry(rep_w[*rr].2).or_default().push(i);
            }
            for idx in strata.values() {
                let c = if mse { b.to_vec() } else { mean_of(idx) };
                let nh = idx.len() as f64;
                for &i in idx {
                    let m = rep_w[ok[i].0].1;
                    let m = if is_missing(m) { (nh - 1.0) / nh } else { m };
                    add(&mut v, ok[i].1, &c, m);
                }
            }
        }
        _ => {
            let all: Vec<usize> = (0..ok.len()).collect();
            let c = if mse { b.to_vec() } else { mean_of(&all) };
            let s = match method {
                "brr" => 1.0 / (r * (1.0 - set.fay).powi(2)),
                "bootstrap" => set.bsn / if mse { r } else { r - 1.0 },
                _ => 4.0 / r,
            };
            for (_, x) in &ok {
                add(&mut v, x, &c, s);
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

fn simple_table(spec: &Spec, b: &[f64], v: &Matrix, df: f64, level: f64, vcetype: &str, logit_ci: bool) -> Vec<String> {
    let stat = match spec.kind {
        Kind::Mean => "Mean",
        Kind::Total => "Total",
        Kind::Ratio => "Ratio",
        _ => "Proportion",
    };
    let prop = spec.kind == Kind::Prop;
    let lw = spec.labels.iter().flat_map(|(h, l)| [h.chars().count(), l.chars().count() + if h.is_empty() { 0 } else { 1 }]).max().unwrap_or(12).clamp(12, 32);
    let tcrit = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
    let mut out = Vec::new();
    let bar = "-".repeat(lw + 50);
    out.push(bar.clone());
    let first = if prop && logit_ci { format!("{:>lw$} | {:>22}{:>17}", "", vcetype, "Logit") } else { format!("{:>lw$} | {:>22}", "", vcetype) };
    out.push(first);
    out.push(format!("{:>lw$} | {:>10}   std. err.     [{}% conf. interval]", "", stat, fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(48)));
    let mut last_head = String::new();
    let mut lo_hi = Vec::new();
    for j in 0..b.len() {
        let (head, label) = &spec.labels[j];
        if !head.is_empty() && *head != last_head {
            out.push(format!("{:<lw$} |", abbrev(head, lw)));
            last_head = head.clone();
        }
        let se = if is_missing(v[(j, j)]) { SYSMISS } else { v[(j, j)].max(0.0).sqrt() };
        let (lo, hi) = if is_missing(se) {
            (SYSMISS, SYSMISS)
        } else if prop && logit_ci {
            let p = b[j];
            if p > 0.0 && p < 1.0 {
                let l = (p / (1.0 - p)).ln();
                let s = se / (p * (1.0 - p));
                (dist::invlogit(l - tcrit * s), dist::invlogit(l + tcrit * s))
            } else {
                (SYSMISS, SYSMISS)
            }
        } else {
            (b[j] - tcrit * se, b[j] + tcrit * se)
        };
        lo_hi.push((lo, hi));
        let lab = if head.is_empty() || prop { abbrev(label, lw) } else { format!("{} ", abbrev(label, lw - 1)) };
        out.push(format!("{:>lw$} | {:>10} {:>10}  {:>10}  {:>10}", lab, g9(b[j]), g9(se), g9(lo), g9(hi)));
    }
    out.push(bar);
    let names = &spec.names;
    let omit = vec![false; b.len()];
    est::stash_table(est::table_matrix(&TableSpec { depvar: "", names, b, v, omitted: &omit, is_base: &omit, df: Some(df), level, eform: None, coef_title: stat }), level);
    out
}

fn g9(x: f64) -> String {
    if is_missing(x) { ".".into() } else { fmt_g(x, 8) }
}

/// Rao-Scott corrected tests and the table of a two-way (or one-way)
/// tabulation.  Fills `res` with the test results.
#[allow(clippy::too_many_arguments)]
fn tab_output(spec: &Spec, p: &[f64], v: &Matrix, vtot: Option<&Matrix>, cmd: &Command, df: f64, m: f64, mhat: f64, res: &mut Vec<(String, f64)>) -> Vec<String> {
    let t = spec.tab.as_ref().unwrap();
    let nr = t.rlabels.len();
    let nc = t.clabels.len().max(1);
    let at = |r: usize, c: usize| r * nc + c;
    let count = cmd.has_opt("count", 2);
    let row_o = cmd.has_opt("row", 3);
    let col_o = cmd.has_opt("column", 3);
    let se_o = cmd.has_opt("se", 2);
    let percent = cmd.has_opt("percent", 3);
    let scale = if count { mhat } else if percent { 100.0 } else { 1.0 };
    let num = |x: f64| -> String {
        if is_missing(x) {
            return ".".into();
        }
        if count {
            return fmt_g(x, 9);
        }
        let s = if percent { format!("{:.2}", x) } else { format!("{:.4}", x) };
        let s = if let Some(r) = s.strip_prefix("0.") { format!(".{r}") } else { s };
        if s == "1.0000" { "1".into() } else { s }
    };
    let rsum: Vec<f64> = (0..nr).map(|r| (0..nc).map(|c| p[at(r, c)]).sum()).collect();
    let csum: Vec<f64> = (0..nc).map(|c| (0..nr).map(|r| p[at(r, c)]).sum()).collect();
    let cell = |r: usize, c: usize| -> f64 {
        if row_o {
            p[at(r, c)] / rsum[r]
        } else if col_o {
            p[at(r, c)] / csum[c]
        } else {
            p[at(r, c)]
        }
    };
    let rlw = t.rlabels.iter().map(|l| l.chars().count()).chain([t.rvar.len(), 5]).max().unwrap_or(9).clamp(9, 20);
    let cw = if t.cvar.is_none() { if count { 14 } else { 11 } } else { 9usize.max(t.clabels.iter().map(|l| l.chars().count() + 1).max().unwrap_or(0)).min(12) };
    let mut out = Vec::new();
    let total_w = if t.cvar.is_some() { nc * cw + cw + 3 } else { cw + 1 };
    out.push("-".repeat(rlw + 2 + total_w));
    if let Some(cv) = &t.cvar {
        out.push(format!("{:>rlw$} | {:^w$}", "", abbrev(cv, nc * cw), w = nc * cw));
        let mut h = format!("{:>rlw$} |", abbrev(&t.rvar, rlw));
        for l in &t.clabels {
            h.push_str(&format!("{:>cw$}", abbrev(l, cw - 1)));
        }
        h.push_str(&format!("{:>w$}", "Total", w = cw + 3));
        out.push(h);
    } else {
        out.push(format!("{:>rlw$} |{:>w$}", abbrev(&t.rvar, rlw), if count { "count" } else if percent { "percentage" } else { "proportion" }, w = cw));
    }
    out.push(format!("{}+{}", "-".repeat(rlw + 1), "-".repeat(total_w)));
    for r in 0..nr {
        let mut line = format!("{:>rlw$} |", abbrev(&t.rlabels[r], rlw));
        for c in 0..nc {
            line.push_str(&format!("{:>cw$}", num(cell(r, c) * if count { mhat } else { scale })));
        }
        if t.cvar.is_some() {
            let tot = if row_o { 1.0 } else if col_o { rsum[r] } else { rsum[r] };
            line.push_str(&format!("{:>w$}", num(tot * scale), w = cw + 3));
        }
        out.push(line);
        if se_o && !row_o && !col_o {
            let mut l2 = format!("{:>rlw$} |", "");
            for c in 0..nc {
                let k = at(r, c);
                let se = match vtot {
                    Some(vt) if count => vt[(k, k)].max(0.0).sqrt(),
                    _ => v[(k, k)].max(0.0).sqrt() * scale,
                };
                l2.push_str(&format!("{:>cw$}", format!("({})", num(se))));
            }
            out.push(l2);
        }
    }
    out.push(format!("{:>rlw$} |", ""));
    let mut tl = format!("{:>rlw$} |", "Total");
    for c in 0..nc {
        let x = if row_o { csum[c] } else if col_o { 1.0 } else { csum[c] };
        tl.push_str(&format!("{:>cw$}", num(x * scale)));
    }
    if t.cvar.is_some() {
        tl.push_str(&format!("{:>w$}", num(scale), w = cw + 3));
    }
    out.push(tl);
    out.push("-".repeat(rlw + 2 + total_w));
    let key = if count {
        "Weighted count"
    } else if row_o {
        "Row proportion"
    } else if col_o {
        "Column proportion"
    } else {
        "Cell proportion"
    };
    if t.cvar.is_some() {
        out.push(format!("Key: {key}"));
    } else {
        out.push(format!("Key: {} = {key}", if count { "count" } else if percent { "percentage" } else { "proportion" }));
    }
    if t.cvar.is_none() || nr < 2 || nc < 2 {
        return out;
    }
    // Pearson and likelihood-ratio statistics with the Rao-Scott second-order correction
    let k = nr * nc;
    let pe: Vec<f64> = (0..k).map(|i| rsum[i / nc] * csum[i % nc]).collect();
    let x2: f64 = m * (0..k).map(|i| if pe[i] > 0.0 { (p[i] - pe[i]).powi(2) / pe[i] } else { 0.0 }).sum::<f64>();
    let g2: f64 = 2.0 * m * (0..k).map(|i| if p[i] > 0.0 && pe[i] > 0.0 { p[i] * (p[i] / pe[i]).ln() } else { 0.0 }).sum::<f64>();
    // main effects X (rows 1..R-1, cols 1..C-1) and interactions Z
    let nx = (nr - 1) + (nc - 1);
    let nz = (nr - 1) * (nc - 1);
    let xm = |i: usize, j: usize| -> f64 {
        let (r, c) = (i / nc, i % nc);
        if j < nr - 1 { (r == j) as i32 as f64 } else { (c == j - (nr - 1)) as i32 as f64 }
    };
    let zm = |i: usize, j: usize| -> f64 {
        let (r, c) = (i / nc, i % nc);
        let (a, b) = (j / (nc - 1), j % (nc - 1));
        (r == a && c == b) as i32 as f64
    };
    // P = D_p - p p' (multinomial covariance per observation)
    let mut pm = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            pm[(i, j)] = if i == j { p[i] } else { 0.0 } - p[i] * p[j];
        }
    }
    let xmat = {
        let mut x = Matrix::zeros(k, nx);
        for i in 0..k {
            for j in 0..nx {
                x[(i, j)] = xm(i, j);
            }
        }
        x
    };
    let zmat = {
        let mut z = Matrix::zeros(k, nz);
        for i in 0..k {
            for j in 0..nz {
                z[(i, j)] = zm(i, j);
            }
        }
        z
    };
    let xt = xmat.transpose();
    let xpx = xt.mul(&pm).unwrap().mul(&xmat).unwrap();
    let (xpx_i, _) = xpx.sym_inverse(1e-12);
    let proj = xmat.mul(&xpx_i).unwrap().mul(&xt).unwrap().mul(&pm).unwrap();
    let ztil = zmat.sub(&proj.mul(&zmat).unwrap()).unwrap();
    let zt = ztil.transpose();
    let a = zt.mul(&pm.scale(1.0 / m)).unwrap().mul(&ztil).unwrap();
    let bmat = zt.mul(v).unwrap().mul(&ztil).unwrap();
    let (ai, _) = a.sym_inverse(1e-12);
    let delta = ai.mul(&bmat).unwrap();
    let tr = delta.trace();
    let tr2 = delta.mul(&delta).unwrap().trace();
    let d1 = tr * tr / tr2;
    let d2 = d1 * df;
    let fp = x2 / tr;
    let fl = g2 / tr;
    let pp = dist::ftail(d1, d2, fp);
    let pl = dist::ftail(d1, d2, fl);
    let dfu = nz as f64;
    out.push(String::new());
    out.push("Pearson:".into());
    out.push(format!("    Uncorrected   chi2({})         = {:>9.4}", nz, x2));
    out.push(format!("    Design-based  F({}, {})  = {:>9.4}     P = {:.4}", fmt_dec2(d1), fmt_dec2(d2), fp, pp));
    if cmd.has_opt("lr", 2) {
        out.push(String::new());
        out.push("Likelihood ratio:".into());
        out.push(format!("    Uncorrected   chi2({})         = {:>9.4}", nz, g2));
        out.push(format!("    Design-based  F({}, {})  = {:>9.4}     P = {:.4}", fmt_dec2(d1), fmt_dec2(d2), fl, pl));
    }
    res.extend([
        ("cun_Pear".to_string(), x2),
        ("F_Pear".to_string(), fp),
        ("df1_Pear".to_string(), d1),
        ("df2_Pear".to_string(), d2),
        ("p_Pear".to_string(), pp),
        ("cun_LR".to_string(), g2),
        ("F_LR".to_string(), fl),
        ("df1_LR".to_string(), d1),
        ("df2_LR".to_string(), d2),
        ("p_LR".to_string(), pl),
        ("mgdeff".to_string(), tr / dfu),
    ]);
    out
}

fn fmt_dec2(x: f64) -> String {
    format!("{:.2}", x)
}

// ---------------------------------------------------------------------------
// estat after svy
// ---------------------------------------------------------------------------

/// `estat effects` / `estat sd` after `svy:`; None for other subcommands.
pub fn estat(_ds: &DataSet, st: &mut Store, sub: &str, opts: &str) -> Result<Option<String>, String> {
    let e = st.e.clone();
    if e.macros.get("prefix").map(String::as_str) != Some("svy") {
        return Ok(None);
    }
    let sub = sub.to_lowercase();
    let opts = split_options(opts);
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = b.colnames.clone();
    let k = names.len();
    let vcetype = e.macros.get("vcetype").cloned().unwrap_or_default();
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
    let stat = match cmd.as_str() {
        "mean" => "Mean",
        "total" => "Total",
        "ratio" => "Ratio",
        "proportion" => "Proportion",
        _ => "Coefficient",
    };
    let model = stat == "Coefficient";
    let label = |n: &str| -> String {
        let n = n.rsplit(':').next().unwrap_or(n);
        if model { est::display_name(n).0 } else { n.to_string() }
    };
    let lw = names.iter().map(|n| label(n).chars().count()).max().unwrap_or(12).clamp(12, 32);
    if sub.len() >= 3 && "effects".starts_with(sub.as_str()) {
        let deff = e.matrices.get("deff").cloned().unwrap_or_else(|| Matrix::zeros(1, k));
        let deft = e.matrices.get("deft").cloned().unwrap_or_else(|| Matrix::zeros(1, k));
        let mut out = Vec::new();
        let bar = "-".repeat(lw + 46);
        out.push(bar.clone());
        out.push(format!("{:>lw$} | {:>22}", "", vcetype));
        out.push(format!("{:>lw$} | {:>10}   std. err.        DEFF      DEFT", if model { abbrev(&depvar, lw) } else { String::new() }, stat));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(44)));
        let mut last_eq = String::new();
        for j in 0..k {
            if let Some((eq, _)) = names[j].split_once(':') {
                if eq != "/" && eq != last_eq {
                    out.push(format!("{:<lw$} |", abbrev(eq, lw)));
                    last_eq = eq.to_string();
                }
            }
            if v[(j, j)] == 0.0 && b.data[j] == 0.0 {
                continue;
            }
            let se = v[(j, j)].max(0.0).sqrt();
            let nm = if names[j].starts_with("/:") { format!("/{}", &names[j][2..]) } else { label(&names[j]) };
            out.push(format!("{:>lw$} | {:>10} {:>10}  {:>10}{:>10}", abbrev(&nm, lw), g9(b.data[j]), g9(se), fmt_g5(deff.data[j]), fmt_g5(deft.data[j])));
        }
        out.push(bar);
        let r = st.begin_r();
        r.set_mat("deff", deff.clone());
        r.set_mat("deft", deft.clone());
        let _ = opts;
        return Ok(Some(out.join("\n")));
    }
    if sub == "sd" {
        if !matches!(cmd.as_str(), "mean") {
            return Err(rc_err(321, "estat sd is allowed only after mean"));
        }
        let vs = e.matrices.get("V_srswr").ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let counts = e.matrices.get("_N").cloned().unwrap_or_else(|| Matrix::filled(1, k, e.get("N").unwrap_or(1.0)));
        let var_o = has(&opts, "variance", 3);
        let mut out = Vec::new();
        let bar = "-".repeat(lw + 26);
        out.push(bar.clone());
        out.push(format!("{:>lw$} | {:>10} {:>11}", "", "Mean", if var_o { "Variance" } else { "Std. dev." }));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(24)));
        let mut sds = Vec::new();
        for j in 0..k {
            let m = e.get("N_sub").map(|_| counts.data[j]).unwrap_or(counts.data[j]);
            let var = vs[(j, j)] * m;
            let x = if var_o { var } else { var.max(0.0).sqrt() };
            sds.push(x);
            out.push(format!("{:>lw$} | {:>10} {:>11}", abbrev(&label(&names[j]), lw), g9(b.data[j]), g9(x)));
        }
        out.push(bar);
        let r = st.begin_r();
        r.set_mat("mean", b.clone());
        r.set_mat(if var_o { "variance" } else { "sd" }, Matrix::row_vector(&sds).with_names(vec!["y1".into()], names.clone()));
        return Ok(Some(out.join("\n")));
    }
    Ok(None)
}

fn fmt_g5(x: f64) -> String {
    if is_missing(x) { ".".into() } else { format!("{:.5}", x) }
}


/// Fit a logit, probit, poisson, ologit, oprobit or mlogit model on
/// columns `x` (include a constant column where wanted; none for ordered
/// models) with unit weights.  `cats` are the outcome categories of the
/// ordered and multinomial models and `base` the base outcome of mlogit.
/// Returns the coefficients (cutpoints after the slopes for ordered models,
/// equation by equation for mlogit) and the inverse information matrix.
pub fn fit_ml(family: &str, x: &[Vec<f64>], y: &[f64], cats: &[f64], base: usize) -> Result<(Vec<f64>, Matrix), String> {
    let fam = match family {
        "logit" => Family::Logit,
        "probit" => Family::Probit,
        "poisson" => Family::Poisson,
        "ologit" => Family::Ologit,
        "oprobit" => Family::Oprobit,
        "mlogit" => Family::Mlogit,
        f => return Err(rc_err(198, format!("{f}: unknown model"))),
    };
    let n = y.len();
    let yidx: Vec<usize> = y.iter().map(|v| cats.iter().position(|c| c == v).unwrap_or(0)).collect();
    let data = Data::Model { fam, x: x.to_vec(), y: y.to_vec(), ncat: cats.len(), base, yidx, offset: vec![0.0; n] };
    let w = vec![1.0; n];
    let k = data.nparams();
    let b0 = data.start(&w);
    let b = newton(k, b0, &|b: &[f64]| data.ll_grad_hess(b, &w))?;
    let (_, _, negh) = data.ll_grad_hess(&b, &w);
    Ok((b, solve_inv(&negh)))
}
