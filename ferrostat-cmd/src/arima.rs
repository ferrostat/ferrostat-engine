//! `arima`: ARIMA, ARMAX and multiplicative seasonal ARIMA models estimated
//! by exact maximum likelihood with the Kalman filter (Harvey,
//! *Forecasting, Structural Time Series Models and the Kalman Filter*, 1989;
//! Hamilton, *Time Series Analysis*, 1994, ch. 13), or by conditional ML
//! (`condition`). Also `predict` and `estat aroots` after `arima`.
//!
//! Model: y_t = x_t b + u_t,  phi(L) PHI(L^s) u_t = theta(L) THETA(L^s) e_t,
//! with AR coefficients entering as  u_t = rho_1 u_{t-1} + ... and MA
//! coefficients as  e_t + theta_1 e_{t-1} + ...

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsops;
use crate::tsreg::{f2, f4, hline, with_vcetype};
use crate::util;

const LN2PI: f64 = 1.837_877_066_409_345_5;

/// Lags of the ARMA terms; seasonal groups are (season, lags in seasonal units).
#[derive(Clone, Debug, Default)]
pub struct ArmaSpec {
    pub ar: Vec<usize>,
    pub ma: Vec<usize>,
    pub sar: Vec<(usize, Vec<usize>)>,
    pub sma: Vec<(usize, Vec<usize>)>,
}

impl ArmaSpec {
    pub(crate) fn n_coef(&self) -> usize {
        self.ar.len() + self.ma.len() + self.sar.iter().map(|g| g.1.len()).sum::<usize>() + self.sma.iter().map(|g| g.1.len()).sum::<usize>()
    }

    fn seasons(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.sar.iter().chain(&self.sma).map(|g| g.0).collect();
        s.sort();
        s.dedup();
        s
    }

    /// Coefficient names in e(b) order: (equation, name, AR?).
    fn names(&self) -> Vec<(String, String)> {
        let lag = |l: usize, w: &str| if l == 1 { format!("L.{w}") } else { format!("L{l}.{w}") };
        let mut out = Vec::new();
        if !self.ar.is_empty() || !self.ma.is_empty() {
            out.extend(self.ar.iter().map(|&l| ("ARMA".to_string(), lag(l, "ar"))));
            out.extend(self.ma.iter().map(|&l| ("ARMA".to_string(), lag(l, "ma"))));
        }
        for s in self.seasons() {
            let eq = format!("ARMA{s}");
            for g in self.sar.iter().filter(|g| g.0 == s) {
                out.extend(g.1.iter().map(|&l| (eq.clone(), lag(l, "ar"))));
            }
            for g in self.sma.iter().filter(|g| g.0 == s) {
                out.extend(g.1.iter().map(|&l| (eq.clone(), lag(l, "ma"))));
            }
        }
        out
    }

    /// Full AR coefficients a_1..a_P (u_t = sum a_l u_{t-l} + ...) and MA
    /// coefficients m_1..m_Q from the (multiplicative) parameters.
    pub(crate) fn expand(&self, c: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut i = 0;
        let mut take = |lags: &[usize], sign: f64, step: usize| -> Vec<f64> {
            let max = lags.iter().max().copied().unwrap_or(0) * step;
            let mut p = vec![0.0; max + 1];
            p[0] = 1.0;
            for &l in lags {
                p[l * step] += sign * c[i];
                i += 1;
            }
            p
        };
        let mut ar = take(&self.ar, -1.0, 1);
        let mut ma = take(&self.ma, 1.0, 1);
        for s in self.seasons() {
            for g in self.sar.iter().filter(|g| g.0 == s) {
                let p = take(&g.1, -1.0, s);
                ar = polymul(&ar, &p);
            }
            for g in self.sma.iter().filter(|g| g.0 == s) {
                let p = take(&g.1, 1.0, s);
                ma = polymul(&ma, &p);
            }
        }
        (ar[1..].iter().map(|v| -v).collect(), ma[1..].to_vec())
    }

    fn to_text(&self) -> String {
        let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let mut parts = vec![format!("ar:{}", j(&self.ar)), format!("ma:{}", j(&self.ma))];
        parts.extend(self.sar.iter().map(|g| format!("sar{}:{}", g.0, j(&g.1))));
        parts.extend(self.sma.iter().map(|g| format!("sma{}:{}", g.0, j(&g.1))));
        parts.join("|")
    }

    pub(crate) fn from_text(s: &str) -> ArmaSpec {
        let mut sp = ArmaSpec::default();
        for part in s.split('|') {
            let Some((k, v)) = part.split_once(':') else { continue };
            let lags: Vec<usize> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            if k == "ar" {
                sp.ar = lags;
            } else if k == "ma" {
                sp.ma = lags;
            } else if let Some(s) = k.strip_prefix("sar") {
                sp.sar.push((s.parse().unwrap_or(1), lags));
            } else if let Some(s) = k.strip_prefix("sma") {
                sp.sma.push((s.parse().unwrap_or(1), lags));
            }
        }
        sp
    }
}

fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Kalman filter
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Init {
    Stationary,
    Diffuse,
    Conditional,
}

/// One-step predictions of u_t (prediction, variance factor F_t; the
/// prediction variance is sigma^2 F_t) for every grid point; observations
/// are used to update the state when present. None if the AR part is not
/// stationary (exact likelihood only).
fn filter(a: &[f64], m: &[f64], u: &[Option<f64>], init: Init) -> Option<Vec<(f64, f64)>> {
    let p = a.len();
    let q = m.len();
    if init == Init::Conditional {
        // conditional on presample values: e_t = u_t - sum a u - sum m e
        let mut e = vec![0.0; u.len()];
        let mut uu = vec![0.0; u.len()];
        let mut out = Vec::with_capacity(u.len());
        for t in 0..u.len() {
            let mut pred = 0.0;
            for l in 1..=p {
                if t >= l {
                    pred += a[l - 1] * uu[t - l];
                }
            }
            for l in 1..=q {
                if t >= l {
                    pred += m[l - 1] * e[t - l];
                }
            }
            match u[t] {
                Some(v) => {
                    uu[t] = v;
                    e[t] = v - pred;
                }
                None => {
                    uu[t] = pred;
                    e[t] = 0.0;
                }
            }
            out.push((pred, 1.0));
        }
        return Some(out);
    }
    let r = p.max(q + 1);
    let tcol: Vec<f64> = (0..r).map(|i| if i < p { a[i] } else { 0.0 }).collect();
    let rv: Vec<f64> = (0..r).map(|i| if i == 0 { 1.0 } else if i <= q { m[i - 1] } else { 0.0 }).collect();
    // T x for companion T: (T x)_i = a_i x_0 + x_{i+1}
    let tmul = |pm: &[f64]| -> Vec<f64> {
        // returns T P T' for symmetric P (r x r, row-major)
        let mut tp = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut v = tcol[i] * pm[j];
                if i + 1 < r {
                    v += pm[(i + 1) * r + j];
                }
                tp[i * r + j] = v;
            }
        }
        let mut out = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut v = tp[i * r] * tcol[j];
                if j + 1 < r {
                    v += tp[i * r + j + 1];
                }
                out[i * r + j] = v;
            }
        }
        out
    };
    let mut pm = vec![0.0; r * r];
    match init {
        Init::Diffuse => {
            for i in 0..r {
                pm[i * r + i] = 1e6;
            }
        }
        _ => {
            // stationary covariance by the doubling algorithm
            let mut am = vec![0.0; r * r];
            for i in 0..r {
                am[i * r] = tcol[i];
                if i + 1 < r {
                    am[i * r + i + 1] = 1.0;
                }
            }
            for i in 0..r {
                for j in 0..r {
                    pm[i * r + j] = rv[i] * rv[j];
                }
            }
            let mut ok = false;
            for _ in 0..60 {
                // P += A P A';  A = A A
                let mut ap = vec![0.0; r * r];
                for i in 0..r {
                    for k in 0..r {
                        let x = am[i * r + k];
                        if x != 0.0 {
                            for j in 0..r {
                                ap[i * r + j] += x * pm[k * r + j];
                            }
                        }
                    }
                }
                for i in 0..r {
                    for j in 0..r {
                        let mut s = 0.0;
                        for k in 0..r {
                            s += ap[i * r + k] * am[j * r + k];
                        }
                        pm[i * r + j] += s;
                    }
                }
                let mut a2 = vec![0.0; r * r];
                for i in 0..r {
                    for k in 0..r {
                        let x = am[i * r + k];
                        if x != 0.0 {
                            for j in 0..r {
                                a2[i * r + j] += x * am[k * r + j];
                            }
                        }
                    }
                }
                am = a2;
                let norm = am.iter().fold(0.0f64, |s, v| s.max(v.abs()));
                if !norm.is_finite() || pm.iter().any(|v| !v.is_finite() || v.abs() > 1e12) {
                    return None;
                }
                if norm < 1e-13 {
                    ok = true;
                    break;
                }
            }
            if !ok {
                return None;
            }
        }
    }
    let mut st = vec![0.0; r];
    let mut out = Vec::with_capacity(u.len());
    for ut in u {
        let pred = st[0];
        let f = pm[0];
        out.push((pred, f));
        if let Some(v) = ut {
            if f <= 0.0 || !f.is_finite() {
                return None;
            }
            let k: Vec<f64> = (0..r).map(|i| pm[i * r] / f).collect();
            let innov = v - pred;
            for i in 0..r {
                st[i] += k[i] * innov;
            }
            for i in 0..r {
                for j in 0..r {
                    pm[i * r + j] -= k[i] * k[j] * f;
                }
            }
        }
        // predict
        let mut ns = vec![0.0; r];
        for i in 0..r {
            ns[i] = tcol[i] * st[0] + if i + 1 < r { st[i + 1] } else { 0.0 };
        }
        st = ns;
        pm = tmul(&pm);
        for i in 0..r {
            for j in 0..r {
                pm[i * r + j] += rv[i] * rv[j];
            }
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Likelihood
// ---------------------------------------------------------------------------

struct Model<'a> {
    spec: &'a ArmaSpec,
    /// differenced dependent variable on the time grid (None = gap)
    w: &'a [Option<f64>],
    /// covariates on the grid (rows)
    x: &'a [Vec<f64>],
    kx: usize,
    init: Init,
    /// observations skipped at the start (conditional ML)
    skip: usize,
}

impl Model<'_> {
    fn npar(&self) -> usize {
        self.kx + self.spec.n_coef() + 1
    }

    /// Log-likelihood contribution of every used observation.
    fn llvec(&self, th: &[f64]) -> Option<Vec<f64>> {
        let sigma = th[self.npar() - 1];
        if !(sigma > 0.0) || th.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let (a, m) = self.spec.expand(&th[self.kx..self.npar() - 1]);
        let u: Vec<Option<f64>> = self.w.iter().zip(self.x).map(|(w, x)| w.map(|w| w - x.iter().zip(th).map(|(a, b)| a * b).sum::<f64>())).collect();
        let pr = filter(&a, &m, &u, self.init)?;
        let s2 = sigma * sigma;
        let mut out = Vec::new();
        for (t, (ut, (pred, f))) in u.iter().zip(pr).enumerate() {
            if let Some(v) = ut {
                if t < self.skip {
                    continue;
                }
                let var = s2 * f;
                out.push(-0.5 * (LN2PI + var.ln() + (v - pred).powi(2) / var));
            }
        }
        Some(out)
    }

    fn ll(&self, th: &[f64]) -> f64 {
        self.llvec(th).map_or(f64::NEG_INFINITY, |v| v.iter().sum())
    }

    fn step(th: &[f64], j: usize) -> f64 {
        1e-6 * th[j].abs().max(1e-2)
    }

    /// Per-observation scores by central differences.
    fn scores(&self, th: &[f64]) -> Option<Vec<Vec<f64>>> {
        let k = self.npar();
        let base = self.llvec(th)?;
        let mut s = vec![vec![0.0; k]; base.len()];
        for j in 0..k {
            let h = Self::step(th, j);
            let mut tp = th.to_vec();
            tp[j] += h;
            let mut tm = th.to_vec();
            tm[j] -= h;
            let (lp, lm, denom) = match (self.llvec(&tp), self.llvec(&tm)) {
                (Some(a), Some(b)) => (a, b, 2.0 * h),
                (Some(a), None) => (a, base.clone(), h),
                (None, Some(b)) => (base.clone(), b, h),
                _ => return None,
            };
            for t in 0..base.len() {
                s[t][j] = (lp[t] - lm[t]) / denom;
            }
        }
        Some(s)
    }

    fn gradient(&self, th: &[f64]) -> Option<Vec<f64>> {
        let s = self.scores(th)?;
        Some((0..self.npar()).map(|j| s.iter().map(|r| r[j]).sum()).collect())
    }

    /// Negative Hessian by differences of the gradient.
    fn neg_hessian(&self, th: &[f64]) -> Option<Matrix> {
        let k = self.npar();
        let mut h = Matrix::zeros(k, k);
        for j in 0..k {
            let d = 1e-4 * th[j].abs().max(1e-2);
            let mut tp = th.to_vec();
            tp[j] += d;
            let mut tm = th.to_vec();
            tm[j] -= d;
            let gp = self.gradient(&tp)?;
            let gm = self.gradient(&tm)?;
            for i in 0..k {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * d);
            }
        }
        // symmetrise
        for i in 0..k {
            for j in 0..i {
                let v = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        Some(h)
    }
}

fn opg(s: &[Vec<f64>], k: usize) -> Matrix {
    let mut m = Matrix::zeros(k, k);
    for r in s {
        for a in 0..k {
            for b in 0..k {
                m[(a, b)] += r[a] * r[b];
            }
        }
    }
    m
}

fn mv(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Optimum {
    th: Vec<f64>,
    ll: f64,
    log: Vec<String>,
    iters: usize,
    converged: bool,
}

fn iter_line(i: usize, ll: f64) -> String {
    format!("Iteration {i}:{}Log likelihood = {}", if i < 10 { "  " } else { " " }, fmt_g(ll, 9))
}

/// BHHH for 5 iterations, then BFGS for 10, alternating; then a Newton refinement.
fn optimize(model: &Model, th0: Vec<f64>, maxiter: usize) -> Result<Optimum, String> {
    let k = model.npar();
    let mut th = th0;
    let mut ll = model.ll(&th);
    if !ll.is_finite() {
        return Err(rc_err(1400, "initial values not feasible"));
    }
    let mut log = vec!["(setting optimization to BHHH)".to_string()];
    let mut hinv = Matrix::identity(k);
    let mut prev: Option<(f64, f64)> = None; // (rel ll change, rel coef change)
    let mut converged = false;
    let mut it = 0;
    loop {
        let s = model.scores(&th).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"))?;
        let g: Vec<f64> = (0..k).map(|j| s.iter().map(|r| r[j]).sum()).collect();
        let o = opg(&s, k);
        let (oinv, _) = o.sym_inverse(1e-14);
        let phase_bhhh = (it % 15) < 5;
        if it > 0 && it % 15 == 5 {
            log.push("(switching optimization to BFGS)".into());
            hinv = oinv.clone();
        } else if it > 0 && it % 15 == 0 {
            log.push("(switching optimization to BHHH)".into());
        }
        log.push(iter_line(it, ll));
        let nr = dot(&g, &mv(&oinv, &g)).abs();
        if let Some((dl, db)) = prev {
            if (dl < 1e-7 || db < 1e-6) && nr < 1e-5 {
                converged = true;
                break;
            }
            if dl == 0.0 && db == 0.0 {
                break;
            }
        }
        if it >= maxiter {
            break;
        }
        let dir = if phase_bhhh { mv(&oinv, &g) } else { mv(&hinv, &g) };
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let cand: Vec<f64> = th.iter().zip(&dir).map(|(t, d)| t + lambda * d).collect();
            let lc = model.ll(&cand);
            if lc.is_finite() && lc >= ll - 1e-12 * ll.abs() {
                accepted = Some((cand, lc));
                break;
            }
            lambda *= 0.5;
        }
        let Some((nth, nll)) = accepted else {
            // no improvement possible along the direction
            it += 1;
            log.push(iter_line(it, ll));
            converged = nr < 1e-3;
            break;
        };
        if !phase_bhhh {
            // BFGS update of the inverse of the negative Hessian
            if let Some(gn) = model.gradient(&nth) {
                let sv: Vec<f64> = nth.iter().zip(&th).map(|(a, b)| a - b).collect();
                let yv: Vec<f64> = g.iter().zip(&gn).map(|(a, b)| a - b).collect();
                let sy = dot(&sv, &yv);
                if sy > 1e-16 {
                    let hy = mv(&hinv, &yv);
                    let yhy = dot(&yv, &hy);
                    let mut nh = hinv.clone();
                    for i in 0..k {
                        for j in 0..k {
                            nh[(i, j)] += (sy + yhy) * sv[i] * sv[j] / (sy * sy) - (hy[i] * sv[j] + sv[i] * hy[j]) / sy;
                        }
                    }
                    hinv = nh;
                }
            }
        }
        let db = th.iter().zip(&nth).map(|(a, b)| (a - b).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        let dl = (nll - ll).abs() / (ll.abs() + 1.0);
        prev = Some((dl, db));
        th = nth;
        ll = nll;
        it += 1;
    }
    if maxiter == 0 {
        return Ok(Optimum { th, ll, log, iters: it, converged });
    }
    // Newton refinement with a numerical Hessian
    log.push("Refining estimates:".into());
    let mut rll = ll;
    log.push(iter_line(0, rll));
    for r in 1..=20 {
        let (Some(g), Some(h)) = (model.gradient(&th), model.neg_hessian(&th)) else { break };
        let (hi, _) = h.sym_inverse(1e-14);
        let dir = mv(&hi, &g);
        let mut lambda = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            let cand: Vec<f64> = th.iter().zip(&dir).map(|(t, d)| t + lambda * d).collect();
            let lc = model.ll(&cand);
            if lc.is_finite() && lc >= rll {
                th = cand;
                rll = lc;
                moved = true;
                break;
            }
            lambda *= 0.5;
        }
        let small = dir.iter().zip(&th).all(|(d, t)| (lambda * d).abs() <= 1e-9 * (t.abs() + 1.0));
        if moved && r <= 9 && (r == 1 || !small) {
            log.push(iter_line(r, rll));
        }
        if !moved || small || dot(&g, &dir).abs() < 1e-12 {
            break;
        }
    }
    Ok(Optimum { th, ll: rll, log, iters: it, converged })
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

fn parse_lags(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for w in s.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()) {
        if let Some((a, b)) = w.split_once('/') {
            let (a, b): (usize, usize) = (a.parse().map_err(|_| rc_err(121, "invalid numlist"))?, b.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
            out.extend(a..=b);
        } else {
            out.push(w.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
        }
    }
    if out.contains(&0) {
        return Err(rc_err(125, "lags must be positive"));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// (lags, season) of `mar(1/2, 12)`.
fn parse_seasonal(s: &str) -> Result<(Vec<usize>, usize), String> {
    let (l, sea) = s.rsplit_once(',').ok_or_else(|| rc_err(198, "seasonal lag required"))?;
    let sea: usize = sea.trim().parse().map_err(|_| rc_err(198, "invalid seasonal lag"))?;
    Ok((parse_lags(l)?, sea))
}

fn three(s: &str, n: usize) -> Result<Vec<usize>, String> {
    let v: Vec<usize> = s.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "invalid specification"))?;
    if v.len() != n {
        return Err(rc_err(198, "invalid specification"));
    }
    Ok(v)
}

fn base_var(dep: &str) -> String {
    dep.rsplit_once('.').map_or(dep.to_string(), |(_, v)| v.to_string())
}

pub struct Arima;

impl CommandHandler for Arima {
    fn name(&self) -> &'static str {
        "arima"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "arima");
        }
        let tvar = tsops::time_var(st)?;
        let mut spec = ArmaSpec::default();
        let mut dop = String::new();
        if let Some(a) = cmd.opt("arima", 5) {
            let v = three(a, 3)?;
            spec.ar = (1..=v[0]).collect();
            spec.ma = (1..=v[2]).collect();
            if v[1] > 0 {
                dop.push_str(&if v[1] == 1 { "D".to_string() } else { format!("D{}", v[1]) });
            }
        }
        if let Some(a) = cmd.opt("ar", 2) {
            spec.ar = parse_lags(a)?;
        }
        if let Some(a) = cmd.opt("ma", 2) {
            spec.ma = parse_lags(a)?;
        }
        if let Some(a) = cmd.opt("sarima", 6) {
            let v = three(a, 4)?;
            let s = v[3];
            if s < 2 {
                return Err(rc_err(198, "seasonal period must be at least 2"));
            }
            if v[0] > 0 {
                spec.sar.push((s, (1..=v[0]).collect()));
            }
            if v[2] > 0 {
                spec.sma.push((s, (1..=v[2]).collect()));
            }
            for _ in 0..v[1] {
                dop.push_str(&format!("S{s}"));
            }
        }
        for (name, raw) in &cmd.opts {
            let n = name.to_lowercase();
            if let Some(r) = raw {
                if n == "mar" {
                    let (l, s) = parse_seasonal(r)?;
                    spec.sar.push((s, l));
                } else if n == "mma" {
                    let (l, s) = parse_seasonal(r)?;
                    spec.sma.push((s, l));
                }
            }
        }
        let (dep0, indep0) = crate::tsreg::split_dep(ds, &cmd.main)?;
        // the difference operators of arima(p,d,q)/sarima() apply to every variable
        let with_op = |v: &str| -> String {
            if dop.is_empty() {
                return v.to_string();
            }
            match v.rsplit_once('.') {
                Some((o, b)) => format!("{}{}.{b}", dop, o.to_uppercase()),
                None => format!("{dop}.{v}"),
            }
        };
        let dep = with_op(&dep0);
        let indep_list: Vec<String> = tsops::expand_text(&indep0).split_whitespace().map(|v| with_op(v)).collect();
        let noconstant = cmd.has_opt("noconstant", 4);
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep_list.join(" "), noconstant, weights: "", extra: vec![] })?;
        let act = d.active();
        let kx = act.len();
        let xnames: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        // time grid
        let tc = util::col(ds, &tvar)?;
        let times: Vec<f64> = d.rows.iter().map(|&i| tc[i]).collect();
        let tmin = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let tmax = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ng = (tmax - tmin).round() as usize + 1;
        let mut w: Vec<Option<f64>> = vec![None; ng];
        let mut x: Vec<Vec<f64>> = vec![vec![0.0; kx]; ng];
        for (p, &t) in times.iter().enumerate() {
            let g = (t - tmin).round() as usize;
            w[g] = Some(d.y[p]);
            x[g] = act.iter().map(|&j| d.cols[j][p]).collect();
        }
        let n_obs = d.n();
        let n_gaps = (0..ng).filter(|&g| w[g].is_none() && g > 0 && w[g - 1].is_some()).count();
        let (a_full, _) = spec.expand(&vec![0.0; spec.n_coef()]);
        let pmax = a_full.len();
        let init = if cmd.has_opt("condition", 4) { Init::Conditional } else if cmd.has_opt("diffuse", 4) { Init::Diffuse } else { Init::Stationary };
        let skip = if init == Init::Conditional { pmax } else { 0 };
        if n_obs <= spec.n_coef() + kx + skip + 1 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let model = Model { spec: &spec, w: &w, x: &x, kx, init, skip };
        let mut th0 = start_values(&model, &d, &act);
        if let Some(f) = cmd.opt("from", 4) {
            let v: Vec<f64> = f.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).map(|w| w.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "invalid from()"))?;
            if v.len() != th0.len() {
                return Err(rc_err(503, format!("from() must contain {} values", th0.len())));
            }
            th0 = v;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(100.0) as usize;
        let opt = optimize(&model, th0, maxiter)?;
        let k = model.npar();
        let th = opt.th.clone();
        let vce_kind = cmd.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "opg".into());
        let s = model.scores(&th).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"))?;
        let o = opg(&s, k);
        let (v, vcetype, vce_name) = match vce_kind.as_str() {
            "opg" => (o.sym_inverse(1e-14).0, "OPG", "opg"),
            "oim" => (model.neg_hessian(&th).ok_or_else(|| rc_err(430, "Hessian could not be computed"))?.sym_inverse(1e-14).0, "", "oim"),
            r if "robust".starts_with(r) && r.len() >= 1 => {
                let (hi, _) = model.neg_hessian(&th).ok_or_else(|| rc_err(430, "Hessian could not be computed"))?.sym_inverse(1e-14);
                (hi.mul(&o)?.mul(&hi)?, "Semirobust", "robust")
            }
            o => return Err(rc_err(198, format!("vce({o}) not allowed"))),
        };
        // names
        let eq = base_var(&dep0);
        let mut names: Vec<String> = xnames.iter().map(|n| format!("{eq}:{n}")).collect();
        names.extend(spec.names().into_iter().map(|(e, n)| format!("{e}:{n}")));
        names.push("sigma:_cons".into());
        // model test: all but the constant and sigma
        let test_idx: Vec<usize> = (0..k - 1).filter(|&j| !(j < kx && xnames[j] == "_cons")).collect();
        let (chi2, df_m) = wald(&th, &v, &test_idx);
        let pchi = if df_m > 0.0 { dist::chi2tail(df_m, chi2) } else { SYSMISS };
        let level = util::level(cmd, st)?;
        // output
        let mut out = Vec::new();
        out.push(String::new());
        out.extend(opt.log.iter().cloned());
        out.push(String::new());
        out.push("ARIMA regression".into());
        out.push(String::new());
        let tmins = tsops::time_label(ds, &tvar, tmin);
        let tmaxs = tsops::time_label(ds, &tvar, tmax);
        let sample = format!("Sample: {tmins} thru {tmaxs}{}", if n_gaps > 0 { ", but with gaps" } else { "" });
        out.push(hline(&sample, "Number of obs", &fmt_g(n_obs as f64, 10)));
        out.push(hline("", &format!("Wald chi2({})", df_m), &f2(chi2)));
        out.push(hline(&format!("Log likelihood = {}", fmt_g(opt.ll, 8)), "Prob > chi2", &f4(pchi)));
        out.push(String::new());
        let mut disp_names = names.clone();
        *disp_names.last_mut().unwrap() = "/sigma".into();
        let zeros = vec![false; k];
        let mut table = est::coef_table(&est::TableSpec { depvar: &dep, names: &disp_names, b: &th, v: &v, omitted: &zeros, is_base: &zeros, df: None, level, eform: None, coef_title: "Coefficient" });
        // a rule above /sigma
        if let Some(i) = table.iter().position(|l| l.trim_start().starts_with("/sigma |")) {
            let lw = table[0].chars().count() - 66;
            let rule = format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64));
            if table[i - 1].trim_end().ends_with('|') && table[i - 1].trim().len() == 1 {
                table[i - 1] = rule;
            } else if !table[i - 1].contains('+') {
                table.insert(i, rule);
            }
        }
        out.extend(with_vcetype(table, vcetype));
        out.push("Note: The test of the variance against zero is one sided, and the two-sided".into());
        out.push("      confidence interval is truncated at zero.".into());
        if !opt.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let mut eqnames: Vec<String> = Vec::new();
        for n in &names {
            let e = n.split(':').next().unwrap().to_string();
            if !eqnames.contains(&e) {
                eqnames.push(e);
            }
        }
        let e = st.begin_e();
        e.set("N", n_obs as f64);
        e.set("N_gaps", n_gaps as f64);
        e.set("k", k as f64);
        e.set("k_eq", eqnames.len() as f64);
        e.set("k_eq_model", 1.0);
        e.set("k_dv", 1.0);
        e.set("k1", kx as f64);
        e.set("df_m", df_m);
        e.set("ll", opt.ll);
        e.set("sigma", th[k - 1]);
        e.set("chi2", chi2);
        e.set("p", pchi);
        e.set("tmin", tmin);
        e.set("tmax", tmax);
        e.set("ar_max", a_full.len() as f64);
        e.set("ma_max", spec.expand(&vec![0.0; spec.n_coef()]).1.len() as f64);
        e.set("rank", k as f64);
        e.set("ic", opt.iters as f64);
        e.set("rc", 0.0);
        e.set("converged", if opt.converged { 1.0 } else { 0.0 });
        e.set_str("cmd", "arima");
        e.set_str("cmdline", format!("arima {}", cmd.raw.trim_start_matches("arima").trim()));
        e.set_str("depvar", dep.clone());
        e.set_str("covariates", xnames.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
        e.set_str("eqnames", eqnames.join(" "));
        e.set_str("title", "ARIMA regression");
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("chi2type", "Wald");
        e.set_str("vce", vce_name);
        if !vcetype.is_empty() {
            e.set_str("vcetype", vcetype);
        }
        let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
        e.set_str("ar", j(&spec.ar));
        e.set_str("ma", j(&spec.ma));
        for (i, g) in spec.sar.iter().enumerate() {
            e.set_str(&format!("mar{}", i + 1), format!("{} {}", j(&g.1), g.0));
        }
        for (i, g) in spec.sma.iter().enumerate() {
            e.set_str(&format!("mma{}", i + 1), format!("{} {}", j(&g.1), g.0));
        }
        e.set_str("seasons", j(&spec.seasons()));
        e.set_str("opt", "ml");
        e.set_str("technique", "bhhh 5 bfgs 10");
        e.set_str("properties", "b V");
        e.set_str("predict", "arima_p");
        e.set_str("estat_cmd", "arima_estat");
        e.set_str("_spec", spec.to_text());
        e.set_str("_init", match init { Init::Conditional => "condition", Init::Diffuse => "diffuse", Init::Stationary => "" });
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&th).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.sample = Some(d.mask.clone());
        Ok(text)
    }
}

fn wald(b: &[f64], v: &Matrix, idx: &[usize]) -> (f64, f64) {
    let k = idx.len();
    if k == 0 {
        return (SYSMISS, 0.0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-14);
    let bs: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
    (dot(&bs, &mv(&vi, &bs)), (k - dropped.len()) as f64)
}

/// Least-squares regression of y on columns (no constant added).
fn lsq(rows: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = rows.first().map_or(0, |r| r.len());
    if k == 0 || rows.len() <= k {
        return vec![0.0; k];
    }
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for (r, yv) in rows.iter().zip(y) {
        for a in 0..k {
            xy[a] += r[a] * yv;
            for c in 0..k {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    mv(&m.sym_inverse(1e-12).0, &xy)
}

/// Starting values: OLS for the regression part; Hannan-Rissanen
/// regressions for the ARMA coefficients; shrunk until stationary.
fn start_values(model: &Model, d: &est::Design, act: &[usize]) -> Vec<f64> {
    let kx = model.kx;
    let rows: Vec<Vec<f64>> = (0..d.n()).map(|t| act.iter().map(|&j| d.cols[j][t]).collect()).collect();
    let beta = lsq(&rows, &d.y);
    let ng = model.w.len();
    let u: Vec<f64> = (0..ng).map(|g| model.w[g].map_or(0.0, |w| w - model.x[g].iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>())).collect();
    let spec = model.spec;
    let ar_lags: Vec<usize> = spec.ar.iter().cloned().chain(spec.sar.iter().flat_map(|g| g.1.iter().map(move |l| l * g.0))).collect();
    let ma_lags: Vec<usize> = spec.ma.iter().cloned().chain(spec.sma.iter().flat_map(|g| g.1.iter().map(move |l| l * g.0))).collect();
    // long autoregression for the innovations
    let mut e = u.clone();
    if !ma_lags.is_empty() {
        let mlong = (ar_lags.iter().chain(&ma_lags).max().copied().unwrap_or(1) + 3).min(ng / 4).max(1);
        let rr: Vec<Vec<f64>> = (mlong..ng).map(|t| (1..=mlong).map(|l| u[t - l]).collect()).collect();
        let yy: Vec<f64> = (mlong..ng).map(|t| u[t]).collect();
        let c = lsq(&rr, &yy);
        for t in 0..ng {
            e[t] = if t >= mlong { u[t] - (1..=mlong).map(|l| c[l - 1] * u[t - l]).sum::<f64>() } else { 0.0 };
        }
    }
    let maxl = ar_lags.iter().chain(&ma_lags).max().copied().unwrap_or(0);
    let mut coefs = vec![0.0; spec.n_coef()];
    if maxl > 0 && ng > maxl + ar_lags.len() + ma_lags.len() + 2 {
        let rr: Vec<Vec<f64>> = (maxl..ng).map(|t| ar_lags.iter().map(|&l| u[t - l]).chain(ma_lags.iter().map(|&l| e[t - l])).collect()).collect();
        let yy: Vec<f64> = (maxl..ng).map(|t| u[t]).collect();
        let c = lsq(&rr, &yy);
        // map back to parameter order: ar, ma, then per season sar, sma
        let nar = spec.ar.len();
        let nma = spec.ma.len();
        let nsar: usize = spec.sar.iter().map(|g| g.1.len()).sum();
        let mut i = 0;
        for j in 0..nar {
            coefs[i] = c[j];
            i += 1;
        }
        for j in 0..nma {
            coefs[i] = c[nar + nsar + j];
            i += 1;
        }
        // seasonal groups in season order
        let mut sar_off = nar;
        let mut sma_off = nar + nsar + nma;
        let mut sar_pos = std::collections::HashMap::new();
        for (gi, g) in spec.sar.iter().enumerate() {
            sar_pos.insert(gi, sar_off);
            sar_off += g.1.len();
        }
        let mut sma_pos = std::collections::HashMap::new();
        for (gi, g) in spec.sma.iter().enumerate() {
            sma_pos.insert(gi, sma_off);
            sma_off += g.1.len();
        }
        for s in spec.seasons() {
            for (gi, g) in spec.sar.iter().enumerate().filter(|(_, g)| g.0 == s) {
                for j in 0..g.1.len() {
                    coefs[i] = c[sar_pos[&gi] + j];
                    i += 1;
                }
            }
            for (gi, g) in spec.sma.iter().enumerate().filter(|(_, g)| g.0 == s) {
                for j in 0..g.1.len() {
                    coefs[i] = c[sma_pos[&gi] + j];
                    i += 1;
                }
            }
        }
    }
    let mut th: Vec<f64> = beta.clone();
    th.extend(coefs.iter().map(|c| c.clamp(-0.95, 0.95)));
    let resid_var = u.iter().map(|v| v * v).sum::<f64>() / ng as f64;
    th.push(resid_var.sqrt().max(1e-8));
    let _ = kx;
    // shrink the ARMA part until the likelihood can be evaluated
    for _ in 0..20 {
        if model.ll(&th).is_finite() {
            break;
        }
        for c in th.iter_mut().skip(model.kx).take(spec.n_coef()) {
            *c *= 0.5;
        }
    }
    th
}

// ---------------------------------------------------------------------------
// predict after arima
// ---------------------------------------------------------------------------

/// `predict` after `arima`: xb (default), y, residuals, yresiduals, mse,
/// with `structural` and `dynamic()`.
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let spec = ArmaSpec::from_text(e.macros.get("_spec").map(String::as_str).unwrap_or(""));
    let init = match e.macros.get("_init").map(String::as_str) {
        Some("condition") => Init::Conditional,
        Some("diffuse") => Init::Diffuse,
        _ => Init::Stationary,
    };
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let kx = e.get("k1").unwrap_or(0.0) as usize;
    let th: Vec<f64> = b.data[..b.cols].to_vec();
    let k = th.len();
    let sigma = th[k - 1];
    let xnames: Vec<String> = b.colnames[..kx].iter().map(|c| c.split_once(':').map_or(c.clone(), |x| x.1.to_string())).collect();
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let names: Vec<&str> = words.iter().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).cloned().collect();
    if names.len() != 1 {
        return Err(rc_err(if names.is_empty() { 100 } else { 103 }, "one new variable name required"));
    }
    let newvar = names[0];
    let mut stat = "xb";
    let mut explicit = false;
    for (n, _) in &cmd.opts {
        let n = n.to_lowercase();
        let s = match n.as_str() {
            "xb" => "xb",
            "y" => "y",
            "mse" => "mse",
            _ if n.len() >= 1 && "residuals".starts_with(n.as_str()) => "residuals",
            _ if n.len() >= 2 && "yresiduals".starts_with(n.as_str()) => "yresiduals",
            _ => continue,
        };
        stat = s;
        explicit = true;
    }
    let structural = cmd.has_opt("structural", 3);
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let mask = util::touse(ds, st, cmd)?;
    let wv = tsops::values(ds, st, &dep)?;
    let xv: Vec<Vec<f64>> = xnames.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?;
    let tmin = e.get("tmin").unwrap_or(0.0);
    let tmax_data = tc.iter().cloned().filter(|v| !is_missing(*v)).fold(f64::NEG_INFINITY, f64::max);
    let dyn_t = match cmd.opt("dynamic", 3) {
        Some(t) => {
            let ex = ferrostat_parser::parser::parse_expression(t).map_err(|e| rc_err(198, e.message))?;
            Some(Env::with_store(ds, st).eval(&ex, 0)?.num()?)
        }
        None => None,
    };
    if !tmax_data.is_finite() || tmax_data < tmin {
        return Err(rc_err(2000, "no observations"));
    }
    let ng = (tmax_data - tmin).round() as usize + 1;
    let mut row_of: Vec<Option<usize>> = vec![None; ng];
    for i in 0..ds.n_obs {
        if !is_missing(tc[i]) && tc[i] >= tmin {
            let g = (tc[i] - tmin).round() as usize;
            if g < ng {
                row_of[g] = Some(i);
            }
        }
    }
    let xb: Vec<Option<f64>> = row_of
        .iter()
        .map(|r| r.and_then(|i| {
            let vals: Vec<f64> = xv.iter().map(|c| c[i]).collect();
            if vals.iter().any(|v| is_missing(*v)) { None } else { Some(vals.iter().zip(&th).map(|(a, b)| a * b).sum()) }
        }))
        .collect();
    let gtime = |g: usize| tmin + g as f64;
    let u: Vec<Option<f64>> = (0..ng)
        .map(|g| {
            if dyn_t.is_some_and(|d| gtime(g) >= d) {
                return None;
            }
            match (row_of[g], xb[g]) {
                (Some(i), Some(x)) if !is_missing(wv[i]) => Some(wv[i] - x),
                _ => None,
            }
        })
        .collect();
    let (a, m) = spec.expand(&th[kx..k - 1]);
    let pr = filter(&a, &m, &u, init).ok_or_else(|| rc_err(498, "AR part not stationary"))?;
    let pw: Vec<Option<f64>> = (0..ng).map(|g| xb[g].map(|x| if structural { x } else { x + pr[g].0 })).collect();
    let mut out = vec![SYSMISS; ds.n_obs];
    let mut msg = String::new();
    match stat {
        "xb" => {
            if !explicit {
                msg = "(option xb assumed; predicted values)".into();
            }
            for g in 0..ng {
                if let (Some(i), Some(p)) = (row_of[g], pw[g]) {
                    out[i] = p;
                }
            }
        }
        "residuals" => {
            for g in 0..ng {
                if let (Some(i), Some(p)) = (row_of[g], pw[g]) {
                    if !is_missing(wv[i]) {
                        out[i] = wv[i] - p;
                    }
                }
            }
        }
        "mse" => {
            for g in 0..ng {
                if let (Some(i), Some(_)) = (row_of[g], pw[g]) {
                    out[i] = sigma * sigma * if structural { 1.0 } else { pr[g].1 };
                }
            }
        }
        _ => {
            // levels: y_t = w_t - sum_{l>=1} c_l y_{t-l}
            let (op, var) = dep.rsplit_once('.').unwrap_or(("", dep.as_str()));
            let poly = if op.is_empty() { vec![(0, 1.0)] } else { tsops::lag_poly(op).unwrap_or(vec![(0, 1.0)]) };
            let yv = util::col(ds, var)?;
            let mut lev: Vec<Option<f64>> = vec![None; ng];
            let actual = |g: i64| -> Option<f64> {
                if g >= 0 && (g as usize) < ng {
                    row_of[g as usize].map(|i| yv[i]).filter(|v| !is_missing(*v))
                } else {
                    // before the grid: look up by time
                    let t = tmin + g as f64;
                    (0..ds.n_obs).find(|&i| tc[i] == t).map(|i| yv[i]).filter(|v| !is_missing(*v))
                }
            };
            for g in 0..ng {
                let Some(p) = pw[g] else { continue };
                let mut v = p;
                let mut ok = true;
                for &(l, c) in &poly {
                    if l == 0 {
                        continue;
                    }
                    let gl = g as i64 - l;
                    let use_pred = dyn_t.is_some_and(|d| tmin + gl as f64 >= d) || actual(gl).is_none();
                    let val = if use_pred && gl >= 0 { lev[gl as usize] } else { actual(gl) };
                    match val {
                        Some(y) => v -= c * y,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    lev[g] = Some(v);
                }
            }
            for g in 0..ng {
                if let (Some(i), Some(l)) = (row_of[g], lev[g]) {
                    out[i] = if stat == "y" { l } else if is_missing(yv[i]) { SYSMISS } else { yv[i] - l };
                }
            }
        }
    }
    for i in 0..ds.n_obs {
        if !mask[i] {
            out[i] = SYSMISS;
        }
    }
    let nmiss = out.iter().filter(|v| is_missing(**v)).count();
    let mut var = util::new_double(newvar, out);
    var.var_type = ferrostat_core::VarType::Float;
    var.format = "%9.0g".into();
    crate::varlist::check_name(newvar)?;
    if ds.var_index.contains_key(newvar) {
        return Err(rc_err(110, format!("variable {newvar} already defined")));
    }
    ds.add_variable(var).map_err(|e| rc_err(198, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", nmiss, if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// estat aroots
// ---------------------------------------------------------------------------

/// Roots of the monic polynomial z^n + c_1 z^{n-1} + ... + c_n (Durand-Kerner).
pub fn poly_roots(c: &[f64]) -> Vec<(f64, f64)> {
    let n = c.len();
    if n == 0 {
        return vec![];
    }
    let eval = |z: (f64, f64)| -> (f64, f64) {
        let mut acc = (1.0, 0.0);
        for &ci in c {
            acc = (acc.0 * z.0 - acc.1 * z.1 + ci, acc.0 * z.1 + acc.1 * z.0);
        }
        acc
    };
    let scale = 1.0 + c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut roots: Vec<(f64, f64)> = (0..n).map(|k| {
        let ang = 0.4 + std::f64::consts::TAU * k as f64 / n as f64;
        (0.9 * scale.min(2.0) * ang.cos(), 0.9 * scale.min(2.0) * ang.sin())
    }).collect();
    for _ in 0..2000 {
        let mut maxd: f64 = 0.0;
        for i in 0..n {
            let zi = roots[i];
            let num = eval(zi);
            let mut den = (1.0, 0.0);
            for j in 0..n {
                if j != i {
                    let dz = (zi.0 - roots[j].0, zi.1 - roots[j].1);
                    den = (den.0 * dz.0 - den.1 * dz.1, den.0 * dz.1 + den.1 * dz.0);
                }
            }
            let dd = den.0 * den.0 + den.1 * den.1;
            if dd == 0.0 {
                continue;
            }
            let q = ((num.0 * den.0 + num.1 * den.1) / dd, (num.1 * den.0 - num.0 * den.1) / dd);
            roots[i] = (zi.0 - q.0, zi.1 - q.1);
            maxd = maxd.max(q.0.abs() + q.1.abs());
        }
        if maxd < 1e-15 {
            break;
        }
    }
    for r in roots.iter_mut() {
        if r.1.abs() < 1e-10 {
            r.1 = 0.0;
        }
    }
    roots.sort_by(|a, b| (b.0.hypot(b.1)).total_cmp(&a.0.hypot(a.1)).then(b.1.total_cmp(&a.1)));
    roots
}

fn root_table(title: &str, roots: &[(f64, f64)], ok_msg: &str, bad_msg: &str) -> Vec<String> {
    let mut out = vec![String::new(), format!("   {title}"), "  +----------------------------------------+".into(), "  |        Eigenvalue        |   Modulus   |".into(), "  |--------------------------+-------------|".into()];
    for &(re, im) in roots {
        let ev = if im == 0.0 { format!("{:>9}", fmt_g(re, 9)) } else { format!("{:>9} {} {:>8}i", fmt_g(re, 9), if im < 0.0 { "-" } else { "+" }, fmt_g(im.abs(), 8)) };
        out.push(format!("  | {:<24} |  {:>9}  |", ev, fmt_g(re.hypot(im), 6)));
    }
    out.push("  +----------------------------------------+".into());
    if roots.iter().all(|r| r.0.hypot(r.1) < 1.0) {
        out.push("   All the eigenvalues lie inside the unit circle.".into());
        out.push(format!("   {ok_msg}"));
    } else {
        out.push("   At least one eigenvalue is at least 1.0.".into());
        out.push(format!("   {bad_msg}"));
    }
    out
}

/// `estat aroots`: eigenvalues of the AR and MA companion matrices.
pub fn aroots(st: &mut Store) -> Result<String, String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let spec = ArmaSpec::from_text(st.e.macros.get("_spec").map(String::as_str).unwrap_or(""));
    let kx = st.e.get("k1").unwrap_or(0.0) as usize;
    let k = b.cols;
    if spec.n_coef() == 0 {
        return Err(rc_err(321, "no ARMA terms in the model"));
    }
    let (a, m) = spec.expand(&b.data[kx..k - 1]);
    let mut out = Vec::new();
    let mut r = Results::new();
    let store = |tag: &str, roots: &[(f64, f64)], r: &mut Results| {
        let n = roots.len();
        let mut re = Matrix::zeros(1, n);
        let mut im = Matrix::zeros(1, n);
        let mut md = Matrix::zeros(1, n);
        for (j, z) in roots.iter().enumerate() {
            re[(0, j)] = z.0;
            im[(0, j)] = z.1;
            md[(0, j)] = z.0.hypot(z.1);
        }
        r.set_mat(&format!("Re_{tag}"), re);
        r.set_mat(&format!("Im_{tag}"), im);
        r.set_mat(&format!("Modulus_{tag}"), md);
    };
    if !a.is_empty() {
        let roots = poly_roots(&a.iter().map(|v| -v).collect::<Vec<_>>());
        out.extend(root_table("Eigenvalue stability condition", &roots, "AR parameters satisfy stability condition.", "AR parameters do not satisfy stability condition."));
        store("ar", &roots, &mut r);
    }
    if !m.is_empty() {
        let roots = poly_roots(&m);
        out.extend(root_table("Eigenvalue stability condition", &roots, "MA parameters satisfy invertibility condition.", "MA parameters do not satisfy invertibility condition.").into_iter().map(|l| l.replace("Eigenvalue stability condition", "Invertibility condition")));
        store("ma", &roots, &mut r);
    }
    st.r = r;
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seasonal_expansion_and_roots() {
        let sp = ArmaSpec { ar: vec![1], ma: vec![], sar: vec![(4, vec![1])], sma: vec![] };
        let (a, _) = sp.expand(&[0.5, 0.3]);
        // (1 - .5L)(1 - .3L^4) = 1 - .5L - .3L^4 + .15L^5
        assert_eq!(a.len(), 5);
        assert!((a[0] - 0.5).abs() < 1e-12 && (a[3] - 0.3).abs() < 1e-12 && (a[4] + 0.15).abs() < 1e-12);
        // (z - .5)(z + .4)(z - .2) = z^3 - .3z^2 - .18z + .04
        let r = poly_roots(&[-0.3, -0.18, 0.04]);
        let mut re: Vec<f64> = r.iter().map(|z| z.0).collect();
        re.sort_by(f64::total_cmp);
        assert!((re[0] + 0.4).abs() < 1e-10 && (re[1] - 0.2).abs() < 1e-10 && (re[2] - 0.5).abs() < 1e-10);
        // z^2 + .25: roots +-.5i
        let r = poly_roots(&[0.0, 0.25]);
        assert!(r.iter().all(|z| z.0.abs() < 1e-10 && (z.1.abs() - 0.5).abs() < 1e-10));
    }

    #[test]
    fn ar1_exact_likelihood() {
        // AR(1) exact likelihood: first term uses variance sigma^2/(1-a^2)
        let w = [Some(1.0), Some(0.5), Some(-0.2)];
        let pr = filter(&[0.6], &[], &w, Init::Stationary).unwrap();
        assert!((pr[0].1 - 1.0 / (1.0 - 0.36)).abs() < 1e-12);
        assert!((pr[1].0 - 0.6).abs() < 1e-12 && (pr[1].1 - 1.0).abs() < 1e-12);
        assert!(filter(&[1.2], &[], &w, Init::Stationary).is_none());
    }
}
