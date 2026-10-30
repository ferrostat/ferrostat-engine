//! Penalized generalized linear models by cyclic coordinate descent.
//!
//! The objective over standardized covariates is
//!
//!   (1/W) sum_i w_i loss_i(b0 + x_i'b) + lambda * sum_j om_j * (alpha |b_j| + (1 - alpha)/2 b_j^2)
//!
//! where `loss` is half the squared error (linear) or minus the
//! log likelihood (logit, probit, Poisson), `W` the sum of weights and
//! `om_j` the penalty loadings (zero: unpenalized). Non-Gaussian families
//! use iteratively reweighted least squares around coordinate descent.
//! Paths are computed from the largest lambda down, each fit starting at
//! the previous solution (warm starts) (Friedman, Hastie & Tibshirani 2010).
//!
//! Also here: cross-validation folds from the session generator,
//! information criteria, unpenalized refits on the selected covariates,
//! the square-root lasso by alternating the scale and a lasso step
//! (scaled lasso), and the iterated plug-in penalty of Belloni,
//! Chernozhukov & Hansen (2012) with heteroskedasticity-robust loadings.

use crate::dist;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Family {
    Linear,
    Logit,
    Probit,
    Poisson,
}

impl Family {
    pub fn parse(s: &str) -> Option<Family> {
        match s {
            "linear" => Some(Family::Linear),
            "logit" => Some(Family::Logit),
            "probit" => Some(Family::Probit),
            "poisson" => Some(Family::Poisson),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Family::Linear => "linear",
            Family::Logit => "logit",
            Family::Probit => "probit",
            Family::Poisson => "poisson",
        }
    }
    /// Mean of the outcome at index `eta`.
    pub fn mean(self, eta: f64) -> f64 {
        match self {
            Family::Linear => eta,
            Family::Logit => 1.0 / (1.0 + (-eta).exp()),
            Family::Probit => dist::normal(eta),
            Family::Poisson => eta.min(700.0).exp(),
        }
    }
    /// Log likelihood contribution (linear: minus half the squared error).
    pub fn ll(self, y: f64, eta: f64) -> f64 {
        match self {
            Family::Linear => -0.5 * (y - eta) * (y - eta),
            Family::Logit => {
                // y*eta - log(1 + exp(eta)), stable
                let l1p = if eta > 0.0 { eta + (-eta).exp().ln_1p() } else { eta.exp().ln_1p() };
                y * eta - l1p
            }
            Family::Probit => {
                let p = dist::normal(eta).clamp(1e-300, 1.0);
                let q = dist::normal(-eta).clamp(1e-300, 1.0);
                y * p.ln() + (1.0 - y) * q.ln()
            }
            Family::Poisson => {
                let e = eta.min(700.0);
                y * e - e.exp() - ln_gamma1(y)
            }
        }
    }
    /// Deviance contribution (linear: squared error).
    pub fn dev(self, y: f64, eta: f64) -> f64 {
        match self {
            Family::Linear => (y - eta) * (y - eta),
            Family::Poisson => {
                let mu = self.mean(eta);
                2.0 * (if y > 0.0 { y * (y / mu).ln() } else { 0.0 } - (y - mu))
            }
            _ => -2.0 * self.ll(y, eta),
        }
    }
    /// IRLS working weight and working residual at `eta`.
    fn working(self, y: f64, eta: f64) -> (f64, f64) {
        match self {
            Family::Linear => (1.0, y - eta),
            Family::Logit => {
                let p = self.mean(eta);
                let w = (p * (1.0 - p)).max(1e-5);
                (w, (y - p) / w)
            }
            Family::Probit => {
                let p = dist::normal(eta).clamp(1e-10, 1.0 - 1e-10);
                let f = (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt();
                let w = (f * f / (p * (1.0 - p))).max(1e-8);
                let fd = f.max(1e-10);
                (w, (y - p) / fd)
            }
            Family::Poisson => {
                let mu = self.mean(eta).max(1e-8);
                (mu, (y - mu) / mu)
            }
        }
    }
}

fn ln_gamma1(y: f64) -> f64 {
    if y <= 1.0 {
        0.0
    } else {
        dist::lngamma(y + 1.0)
    }
}

/// A penalized problem: covariate columns over observations, outcome,
/// observation weights, penalty loadings (on the standardized scale) and
/// the elastic-net mixing parameter.
#[derive(Clone)]
pub struct Problem {
    pub family: Family,
    /// columns, each over the n observations (original scale)
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub w: Vec<f64>,
    pub offset: Option<Vec<f64>>,
    pub loadings: Vec<f64>,
    pub alpha: f64,
    /// standardize covariates (otherwise loadings are divided by the
    /// standard deviation so the penalty acts on the original scale)
    pub standardize: bool,
}

/// Standardized design: centred and scaled columns with their means and
/// standard deviations, normalized weights.
struct Std {
    xs: Vec<Vec<f64>>,
    mean: Vec<f64>,
    sd: Vec<f64>,
    w: Vec<f64>,
    om: Vec<f64>,
}

fn standardize(p: &Problem) -> Std {
    let n = p.y.len();
    let sw: f64 = p.w.iter().sum::<f64>().max(1e-300);
    let w: Vec<f64> = p.w.iter().map(|v| v * n as f64 / sw).collect();
    let mut xs = Vec::with_capacity(p.x.len());
    let mut mean = Vec::new();
    let mut sd = Vec::new();
    let mut om = Vec::new();
    for (j, c) in p.x.iter().enumerate() {
        let m = c.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / n as f64;
        let v = c.iter().zip(&w).map(|(a, b)| b * (a - m) * (a - m)).sum::<f64>() / n as f64;
        let s = if v > 1e-24 { v.sqrt() } else { 0.0 };
        let col: Vec<f64> = if s > 0.0 { c.iter().map(|a| (a - m) / s).collect() } else { vec![0.0; n] };
        xs.push(col);
        mean.push(m);
        sd.push(s);
        let l = p.loadings.get(j).copied().unwrap_or(1.0);
        om.push(if p.standardize || s == 0.0 { l } else { l / s });
    }
    Std { xs, mean, sd, w, om }
}

/// One solution on the path, on the original scale.
#[derive(Clone, Debug)]
pub struct Point {
    pub lambda: f64,
    pub b0: f64,
    pub b: Vec<f64>,
    /// mean deviance over the fitting sample
    pub dev: f64,
    /// number of nonzero coefficients
    pub df: usize,
}

/// Coordinate-descent state over the standardized design.
struct Cd<'a> {
    p: &'a Problem,
    s: &'a Std,
    b0: f64,
    b: Vec<f64>,
}

impl Cd<'_> {
    fn eta(&self) -> Vec<f64> {
        let n = self.p.y.len();
        let mut e = vec![self.b0; n];
        if let Some(o) = &self.p.offset {
            for i in 0..n {
                e[i] += o[i];
            }
        }
        for (j, bj) in self.b.iter().enumerate() {
            if *bj != 0.0 {
                for (ei, xij) in e.iter_mut().zip(&self.s.xs[j]) {
                    *ei += bj * xij;
                }
            }
        }
        e
    }

    /// Penalized weighted least squares by coordinate descent with
    /// working weights `ww` and working response residuals `r`
    /// (r = z - eta, updated in place).
    fn wls(&mut self, lambda: f64, ww: &[f64], r: &mut [f64], tol: f64, active_only: &mut Vec<bool>) {
        let n = r.len() as f64;
        let alpha = self.p.alpha;
        let pcount = self.b.len();
        let v: Vec<f64> = (0..pcount).map(|j| self.s.xs[j].iter().zip(ww).map(|(x, w)| w * x * x).sum::<f64>() / n).collect();
        let sww: f64 = ww.iter().sum::<f64>().max(1e-300);
        let sweep = |cd: &mut Self, set: &[usize], r: &mut [f64]| -> f64 {
            let mut maxd: f64 = 0.0;
            // intercept
            let d0 = r.iter().zip(ww).map(|(a, w)| a * w).sum::<f64>() / sww;
            if d0 != 0.0 {
                cd.b0 += d0;
                for ri in r.iter_mut() {
                    *ri -= d0;
                }
                maxd = maxd.max(d0 * d0);
            }
            for &j in set {
                if v[j] <= 0.0 {
                    continue;
                }
                let xj = &cd.s.xs[j];
                let g = xj.iter().zip(r.iter()).zip(ww).map(|((x, ri), w)| w * x * ri).sum::<f64>() / n;
                let old = cd.b[j];
                let z = g + v[j] * old;
                let pen = lambda * cd.s.om[j];
                let new = soft(z, pen * alpha) / (v[j] + pen * (1.0 - alpha));
                if new != old {
                    let d = new - old;
                    for (ri, x) in r.iter_mut().zip(xj) {
                        *ri -= d * x;
                    }
                    cd.b[j] = new;
                    maxd = maxd.max(v[j] * d * d);
                }
            }
            maxd
        };
        let all: Vec<usize> = (0..pcount).collect();
        for _outer in 0..200 {
            // full sweep, then iterate on the active set
            let d = sweep(self, &all, r);
            for j in 0..pcount {
                active_only[j] = self.b[j] != 0.0;
            }
            if d < tol {
                break;
            }
            let act: Vec<usize> = (0..pcount).filter(|&j| active_only[j]).collect();
            for _ in 0..10_000 {
                if sweep(self, &act, r) < tol {
                    break;
                }
            }
        }
    }

    /// Fit at one lambda from the current state.
    fn fit(&mut self, lambda: f64, tol: f64) {
        let fam = self.p.family;
        let n = self.p.y.len();
        let mut act = vec![false; self.b.len()];
        if fam == Family::Linear {
            let eta = self.eta();
            let mut r: Vec<f64> = (0..n).map(|i| self.p.y[i] - eta[i]).collect();
            let ww = self.s.w.clone();
            self.wls(lambda, &ww, &mut r, tol, &mut act);
            return;
        }
        let mut last = f64::INFINITY;
        for _ in 0..100 {
            let eta = self.eta();
            let mut ww = vec![0.0; n];
            let mut r = vec![0.0; n];
            for i in 0..n {
                let (a, z) = fam.working(self.p.y[i], eta[i]);
                ww[i] = a * self.s.w[i];
                r[i] = z;
            }
            let old_b = self.b.clone();
            let old0 = self.b0;
            self.wls(lambda, &ww, &mut r, tol, &mut act);
            let ch = (self.b0 - old0).abs().max(self.b.iter().zip(&old_b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
            let obj = self.objective(lambda);
            if ch < 1e-6 || (last - obj).abs() < 1e-10 * obj.abs().max(1.0) {
                break;
            }
            last = obj;
        }
    }

    fn objective(&self, lambda: f64) -> f64 {
        let eta = self.eta();
        let n = eta.len() as f64;
        let loss: f64 = (0..eta.len()).map(|i| -self.s.w[i] * self.p.family.ll(self.p.y[i], eta[i])).sum::<f64>() / n;
        let pen: f64 = self.b.iter().zip(&self.s.om).map(|(b, o)| o * (self.p.alpha * b.abs() + 0.5 * (1.0 - self.p.alpha) * b * b)).sum();
        loss + lambda * pen
    }

    fn point(&self, lambda: f64) -> Point {
        let eta = self.eta();
        let n = eta.len() as f64;
        let dev = (0..eta.len()).map(|i| self.s.w[i] * self.p.family.dev(self.p.y[i], eta[i])).sum::<f64>() / n;
        let mut b = vec![0.0; self.b.len()];
        let mut b0 = self.b0;
        for j in 0..b.len() {
            if self.s.sd[j] > 0.0 {
                b[j] = self.b[j] / self.s.sd[j];
                b0 -= b[j] * self.s.mean[j];
            }
        }
        Point { lambda, b0, b, dev, df: self.b.iter().filter(|v| **v != 0.0).count() }
    }
}

fn soft(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// Null fit (only unpenalized covariates) and the smallest lambda at
/// which every penalized coefficient is zero.
pub fn lambda_max(p: &Problem) -> f64 {
    let s = standardize(p);
    let mut cd = Cd { p, s: &s, b0: start_b0(p), b: vec![0.0; p.x.len()] };
    // unpenalized covariates enter at any lambda
    cd.fit(1e30, 1e-10);
    let eta = cd.eta();
    let n = p.y.len() as f64;
    let mut best: f64 = 0.0;
    for j in 0..p.x.len() {
        if s.om[j] <= 0.0 || s.sd[j] == 0.0 {
            continue;
        }
        let g: f64 = (0..p.y.len())
            .map(|i| {
                let (a, z) = p.family.working(p.y[i], eta[i]);
                s.w[i] * a * z * s.xs[j][i]
            })
            .sum::<f64>()
            / n;
        best = best.max(g.abs() / (s.om[j] * p.alpha.max(1e-3)));
    }
    best
}

fn start_b0(p: &Problem) -> f64 {
    let n = p.y.len() as f64;
    let ybar = p.y.iter().zip(&p.w).map(|(a, b)| a * b).sum::<f64>() / p.w.iter().sum::<f64>().max(1e-300);
    let _ = n;
    match p.family {
        Family::Linear => ybar,
        Family::Logit => {
            let q = ybar.clamp(1e-6, 1.0 - 1e-6);
            (q / (1.0 - q)).ln()
        }
        Family::Probit => dist::invnormal(ybar.clamp(1e-6, 1.0 - 1e-6)),
        Family::Poisson => ybar.max(1e-8).ln(),
    }
}

/// Log-spaced grid of `n` values from `lmax` down to `ratio * lmax`.
pub fn grid(lmax: f64, n: usize, ratio: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![lmax];
    }
    let (a, b) = (lmax.max(1e-300).ln(), (lmax * ratio).max(1e-300).ln());
    (0..n).map(|k| (a + (b - a) * k as f64 / (n - 1) as f64).exp()).collect()
}

/// Default ratio of the smallest to the largest lambda.
pub fn default_ratio(n: usize, p: usize) -> f64 {
    if p < n {
        1e-4
    } else {
        1e-2
    }
}

/// Path over `lambdas` (decreasing). With `stop` > 0 the path ends once the
/// relative change in deviance falls below it, or once more coefficients
/// than observations are nonzero.
pub fn path(p: &Problem, lambdas: &[f64], stop: f64) -> Vec<Point> {
    let s = standardize(p);
    let mut cd = Cd { p, s: &s, b0: start_b0(p), b: vec![0.0; p.x.len()] };
    let mut out: Vec<Point> = Vec::new();
    let n = p.y.len();
    for (k, &l) in lambdas.iter().enumerate() {
        cd.fit(l, 1e-9);
        let pt = cd.point(l);
        let df = pt.df;
        if stop > 0.0 && k >= 2 {
            let prev = out[k - 1].dev;
            if prev > 0.0 && (prev - pt.dev) / prev < stop && df > 0 {
                out.push(pt);
                break;
            }
        }
        out.push(pt);
        if df + 1 >= n {
            break;
        }
    }
    out
}

/// Fit at a single lambda (warm path from lambda_max down to it).
pub fn fit_at(p: &Problem, lambda: f64) -> Point {
    let lmax = lambda_max(p);
    if lambda >= lmax {
        return path(p, &[lambda], 0.0).pop().unwrap();
    }
    let k = 20usize;
    let lam: Vec<f64> = grid(lmax, k, lambda / lmax);
    path(p, &lam, 0.0).pop().unwrap()
}

/// Linear index of a point for observation `i` of columns `x`.
pub fn eta_of(pt: &Point, x: &[Vec<f64>], i: usize) -> f64 {
    pt.b0 + pt.b.iter().zip(x).map(|(b, c)| if *b != 0.0 { b * c[i] } else { 0.0 }).sum::<f64>()
}

/// Random fold assignment (1..=k) from uniform draws.
pub fn folds(n: usize, k: usize, uniform: &mut dyn FnMut() -> f64) -> Vec<usize> {
    let mut idx: Vec<(f64, usize)> = (0..n).map(|i| (uniform(), i)).collect();
    idx.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut f = vec![0; n];
    for (r, (_, i)) in idx.into_iter().enumerate() {
        f[i] = r % k + 1;
    }
    f
}

pub fn sub(p: &Problem, keep: &[bool]) -> Problem {
    let pick = |v: &Vec<f64>| -> Vec<f64> { v.iter().zip(keep).filter(|(_, k)| **k).map(|(a, _)| *a).collect() };
    Problem { family: p.family, x: p.x.iter().map(pick).collect(), y: pick(&p.y), w: pick(&p.w), offset: p.offset.as_ref().map(pick), loadings: p.loadings.clone(), alpha: p.alpha, standardize: p.standardize }
}

/// Cross-validation over `lambdas`: mean held-out deviance and its
/// standard error across folds, for each lambda.
pub fn cv(p: &Problem, lambdas: &[f64], fold: &[usize], k: usize) -> (Vec<f64>, Vec<f64>) {
    cv_with(p, lambdas, fold, k, &|q: &Problem, l: &[f64]| path(q, l, 0.0))
}

/// Cross-validation with a caller-supplied path function (square-root
/// lasso, fixed loadings, ...).
pub fn cv_with(p: &Problem, lambdas: &[f64], fold: &[usize], k: usize, fit: &dyn Fn(&Problem, &[f64]) -> Vec<Point>) -> (Vec<f64>, Vec<f64>) {
    let m = lambdas.len();
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); m];
    for f in 1..=k {
        let train: Vec<bool> = fold.iter().map(|&g| g != f).collect();
        let test: Vec<usize> = (0..fold.len()).filter(|&i| fold[i] == f).collect();
        if test.is_empty() {
            continue;
        }
        let pt = sub(p, &train);
        let pts = fit(&pt, lambdas);
        let sw: f64 = test.iter().map(|&i| p.w[i]).sum::<f64>().max(1e-300);
        for (l, q) in pts.iter().enumerate() {
            let d: f64 = test
                .iter()
                .map(|&i| {
                    let e = eta_of(q, &p.x, i) + p.offset.as_ref().map_or(0.0, |o| o[i]);
                    p.w[i] * p.family.dev(p.y[i], e)
                })
                .sum::<f64>()
                / sw;
            per[l].push(d);
        }
    }
    let mut mean = Vec::with_capacity(m);
    let mut se = Vec::with_capacity(m);
    for v in per {
        if v.is_empty() {
            mean.push(f64::NAN);
            se.push(f64::NAN);
            continue;
        }
        let k = v.len() as f64;
        let mu = v.iter().sum::<f64>() / k.max(1.0);
        let var = if k > 1.0 { v.iter().map(|a| (a - mu) * (a - mu)).sum::<f64>() / (k - 1.0) } else { 0.0 };
        mean.push(mu);
        se.push((var / k.max(1.0)).sqrt());
    }
    (mean, se)
}

/// Bayesian information criterion of a path point.
pub fn bic(p: &Problem, pt: &Point) -> f64 {
    let n = p.y.len() as f64;
    let ll: f64 = match p.family {
        Family::Linear => {
            let rss = pt.dev * n;
            -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI).ln() + (rss / n).max(1e-300).ln())
        }
        _ => (0..p.y.len()).map(|i| p.w[i] * p.family.ll(p.y[i], eta_of(pt, &p.x, i) + p.offset.as_ref().map_or(0.0, |o| o[i]))).sum(),
    };
    -2.0 * ll + (pt.df as f64 + 1.0) * n.ln()
}

/// Unpenalized refit on the columns `sel` (with intercept): (b0, b over
/// all columns, zero outside `sel`).
pub fn refit(p: &Problem, sel: &[usize]) -> (f64, Vec<f64>) {
    let n = p.y.len();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut r: Vec<f64> = sel.iter().map(|&j| p.x[j][i]).collect();
            r.push(1.0);
            r
        })
        .collect();
    let mut b = vec![0.0; p.x.len()];
    let coef = glm(&rows, &p.y, &p.w, p.offset.as_deref(), p.family);
    match coef {
        Some(c) => {
            for (a, &j) in sel.iter().enumerate() {
                b[j] = c[a];
            }
            (c[sel.len()], b)
        }
        None => {
            // collinear selection: fall back to a tiny ridge path point
            let mut q = p.clone();
            q.loadings = (0..p.x.len()).map(|j| if sel.contains(&j) { 1.0 } else { 1e6 }).collect();
            q.alpha = 0.0;
            let pt = path(&q, &[1e-8], 0.0).pop().unwrap();
            (pt.b0, pt.b)
        }
    }
}

/// Unpenalized GLM by Newton steps on rows `x` (None: singular).
pub fn glm(x: &[Vec<f64>], y: &[f64], w: &[f64], offset: Option<&[f64]>, fam: Family) -> Option<Vec<f64>> {
    let k = x.first().map_or(0, |r| r.len());
    let n = y.len();
    let mut b = vec![0.0; k];
    if fam != Family::Linear && k > 0 {
        let sw: f64 = w.iter().sum::<f64>().max(1e-300);
        let ybar = y.iter().zip(w).map(|(a, c)| a * c).sum::<f64>() / sw;
        b[k - 1] = match fam {
            Family::Logit => (ybar.clamp(1e-6, 1.0 - 1e-6) / (1.0 - ybar.clamp(1e-6, 1.0 - 1e-6))).ln(),
            Family::Probit => dist::invnormal(ybar.clamp(1e-6, 1.0 - 1e-6)),
            Family::Poisson => ybar.max(1e-8).ln(),
            Family::Linear => 0.0,
        };
    }
    for it in 0..100 {
        let mut g = vec![0.0; k];
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..n {
            let eta: f64 = x[i].iter().zip(&b).map(|(a, c)| a * c).sum::<f64>() + offset.map_or(0.0, |o| o[i]);
            let (a, z) = fam.working(y[i], eta);
            let ww = w[i] * a;
            for r in 0..k {
                g[r] += ww * z * x[i][r];
                for c in 0..=r {
                    h[r][c] += ww * x[i][r] * x[i][c];
                }
            }
        }
        for r in 0..k {
            for c in 0..r {
                h[c][r] = h[r][c];
            }
        }
        let step = crate::teffects::solve(&h, &g)?;
        let scale = h.iter().enumerate().map(|(r, row)| row[r].abs()).fold(0.0, f64::max).max(1e-300);
        // singular designs show up as huge steps along null directions
        if step.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mx = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let damp = if mx > 5.0 && fam != Family::Linear { 5.0 / mx } else { 1.0 };
        for r in 0..k {
            b[r] += damp * step[r];
        }
        let _ = scale;
        if fam == Family::Linear || mx < 1e-10 || it > 90 {
            break;
        }
    }
    Some(b)
}

/// Iterated plug-in lambda with heteroskedasticity-robust penalty
/// loadings: lambda = c Phi^{-1}(1 - gamma/(2p)) / sqrt(n) with c = 1.1
/// and gamma = 0.1 / ln(max(p, n)); loadings are the root mean squares of
/// standardized covariate times score residual, recomputed from the
/// post-selection fit. `homoskedastic` uses loadings sigma (one value).
/// Returns (lambda, loadings on the standardized scale, final point).
pub fn plugin(p: &Problem, homoskedastic: bool) -> (f64, Vec<f64>, Point) {
    let s = standardize(p);
    let n = p.y.len() as f64;
    let np = p.x.len().max(1) as f64;
    let gamma = 0.1 / np.max(n).ln();
    let lam = 1.1 * dist::invnormal(1.0 - gamma / (2.0 * np)) / n.sqrt();
    let base: Vec<f64> = p.loadings.clone();
    // initial residuals: outcome around its mean (unpenalized covariates kept)
    let b0 = start_b0(p);
    let mut resid: Vec<f64> = p.y.iter().map(|y| y - p.family.mean(b0 + 0.0)).collect();
    let mut q = p.clone();
    q.standardize = true;
    let mut pt = Point { lambda: lam, b0, b: vec![0.0; p.x.len()], dev: 0.0, df: 0 };
    let mut load = vec![1.0; p.x.len()];
    for _ in 0..15 {
        let new_load: Vec<f64> = (0..p.x.len())
            .map(|j| {
                if base[j] == 0.0 {
                    return 0.0;
                }
                if homoskedastic {
                    let sig2 = resid.iter().zip(&s.w).map(|(r, w)| w * r * r).sum::<f64>() / n;
                    sig2.sqrt() * base[j]
                } else {
                    let v = (0..p.y.len()).map(|i| s.w[i] * s.xs[j][i] * s.xs[j][i] * resid[i] * resid[i]).sum::<f64>() / n;
                    v.sqrt() * base[j]
                }
            })
            .collect();
        let ch = new_load.iter().zip(&load).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        load = new_load;
        q.loadings = load.iter().map(|v| if *v == 0.0 { 0.0 } else { v.max(1e-12) }).collect();
        pt = fit_at(&q, lam);
        let sel: Vec<usize> = (0..p.x.len()).filter(|&j| pt.b[j] != 0.0).collect();
        let (c0, cb) = refit(p, &sel);
        let post = Point { lambda: lam, b0: c0, b: cb, dev: 0.0, df: sel.len() };
        resid = (0..p.y.len()).map(|i| p.y[i] - p.family.mean(eta_of(&post, &p.x, i) + p.offset.as_ref().map_or(0.0, |o| o[i]))).collect();
        if ch < 1e-6 {
            break;
        }
    }
    (lam, load, pt)
}

/// Warm-started solver: successive fits start from the last solution.
pub struct Warm<'a> {
    p: &'a Problem,
    s: Std,
    b0: f64,
    b: Vec<f64>,
}

impl<'a> Warm<'a> {
    pub fn new(p: &'a Problem) -> Self {
        Warm { p, s: standardize(p), b0: start_b0(p), b: vec![0.0; p.x.len()] }
    }
    pub fn fit(&mut self, lambda: f64) -> Point {
        let mut cd = Cd { p: self.p, s: &self.s, b0: self.b0, b: std::mem::take(&mut self.b) };
        cd.fit(lambda, 1e-9);
        let pt = cd.point(lambda);
        self.b0 = cd.b0;
        self.b = cd.b;
        pt
    }
}

/// Square-root lasso continuing from a warm solver (see `sqrt_fit`).
pub fn sqrt_fit_warm(w: &mut Warm, lambda: f64, warm_sigma: f64) -> (Point, f64) {
    let p = w.p;
    let mut sigma = warm_sigma.max(1e-8);
    let mut pt = w.fit(lambda * sigma);
    for _ in 0..100 {
        let n = p.y.len() as f64;
        let sw = p.w.iter().sum::<f64>().max(1e-300);
        let rss: f64 = (0..p.y.len()).map(|i| p.w[i] * (p.y[i] - eta_of(&pt, &p.x, i)).powi(2)).sum::<f64>() / sw * n;
        let ns = (rss / n).sqrt().max(1e-10);
        let done = (ns - sigma).abs() < 1e-8 * sigma.max(1e-8);
        sigma = ns;
        if done {
            break;
        }
        pt = w.fit(lambda * sigma);
    }
    pt.lambda = lambda;
    (pt, sigma)
}

/// Square-root lasso at `lambda` (objective sqrt(mean squared error) +
/// lambda * penalty): alternate sigma = RMS residual and a lasso step at
/// lambda * sigma. Returns the point (with `lambda` as given) and sigma.
pub fn sqrt_fit(p: &Problem, lambda: f64, warm_sigma: f64) -> (Point, f64) {
    let mut sigma = warm_sigma.max(1e-8);
    let mut pt = fit_at(p, lambda * sigma);
    for _ in 0..50 {
        let n = p.y.len() as f64;
        let rss: f64 = (0..p.y.len()).map(|i| p.w[i] * (p.y[i] - eta_of(&pt, &p.x, i)).powi(2)).sum::<f64>() / p.w.iter().sum::<f64>().max(1e-300) * n;
        let ns = (rss / n).sqrt().max(1e-10);
        let done = (ns - sigma).abs() < 1e-8 * sigma.max(1e-8);
        sigma = ns;
        pt = fit_at(p, lambda * sigma);
        if done {
            break;
        }
    }
    pt.lambda = lambda;
    (pt, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut s: u64 = 7;
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        let x: Vec<Vec<f64>> = (0..p).map(|_| (0..n).map(|_| u() * 2.0 - 1.0).collect()).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[0][i] - x[1][i] + 0.3 * (u() - 0.5)).collect();
        (x, y)
    }

    #[test]
    fn tiny_lambda_matches_least_squares_and_kkt_holds() {
        let (x, y) = data(80, 4);
        let p = Problem { family: Family::Linear, x: x.clone(), y: y.clone(), w: vec![1.0; 80], offset: None, loadings: vec![1.0; 4], alpha: 1.0, standardize: true };
        let pt = fit_at(&p, 1e-9);
        let (b0, b) = refit(&p, &[0, 1, 2, 3]);
        assert!((pt.b0 - b0).abs() < 1e-5);
        for j in 0..4 {
            assert!((pt.b[j] - b[j]).abs() < 1e-5, "{j}");
        }
        let lmax = lambda_max(&p);
        let z = fit_at(&p, lmax * 1.0001);
        assert_eq!(z.df, 0);
        let nz = fit_at(&p, lmax * 0.9);
        assert!(nz.df >= 1);
    }
}
