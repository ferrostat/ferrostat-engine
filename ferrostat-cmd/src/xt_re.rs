//! Random-intercept models with a general response density, fitted by
//! mean-variance adaptive Gauss-Hermite quadrature over nested levels:
//!
//! * panel random effects: `xtologit`, `xtoprobit`, `xttobit`,
//!   `xtintreg`, `xtcloglog`, `xtstreg`;
//! * multilevel: `meologit`, `meoprobit`, `menbreg`, `mecloglog`,
//!   `metobit`, `meintreg`, `mestreg`;
//! * interval-censored parametric survival: `stintreg` (no random effects).
//!
//! Each density supplies analytic (or Richardson-differenced) first
//! derivatives per observation; the information matrix is obtained by
//! differencing the total gradient with the quadrature nodes held fixed.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::streg::Dist;
use crate::util;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    for n in ["xtologit", "xtoprobit", "xttobit", "xtintreg", "xtcloglog", "xtstreg"] {
        v.push((n, Box::new(XtRe { verb: n })));
    }
    for n in ["meologit", "meoprobit", "menbreg", "mecloglog", "metobit", "meintreg", "mestreg"] {
        v.push((n, Box::new(MeRe { verb: n })));
    }
    v.push(("stintreg", Box::new(StIntreg)));
    v
}

// ---------------------------------------------------------------------------
// densities
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dens {
    Logit,
    Probit,
    Cloglog,
    /// ordered response with `ncut` cutpoints; `true` = probit link
    Ordinal(bool, usize),
    /// negative binomial, mean dispersion (aux ln alpha)
    Nb2,
    /// Gaussian with censoring/intervals: y = lower (-inf), y2 = upper (+inf)
    Interval,
    /// survival: y = t, y2 = failure, t0 = entry
    Surv(Dist, bool),
    /// interval-censored survival: y = lower time, y2 = upper time (+inf)
    SurvInt(Dist, bool),
}

impl Dens {
    pub fn naux(&self) -> usize {
        match *self {
            Dens::Ordinal(_, k) => k,
            Dens::Nb2 | Dens::Interval => 1,
            Dens::Surv(d, _) | Dens::SurvInt(d, _) => d.anc().len(),
            _ => 0,
        }
    }
}

fn lnphi(z: f64) -> f64 {
    crate::xtnl::ln_phi(z)
}

/// Richardson-extrapolated central-difference gradient.
fn nd_grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xx = x.to_vec();
    for j in 0..x.len() {
        let h = 1e-3 * (1.0 + x[j].abs());
        let mut d = |h: f64| {
            xx[j] = x[j] + h;
            let a = f(&xx);
            xx[j] = x[j] - h;
            let b = f(&xx);
            xx[j] = x[j];
            (a - b) / (2.0 * h)
        };
        let d1 = d(h);
        let d2 = d(h / 2.0);
        g[j] = (4.0 * d2 - d1) / 3.0;
    }
    g
}

/// ln(S(a) - S(b)) for a < b from ln S values.
fn ln_diff_exp(la: f64, lb: f64) -> f64 {
    if lb == f64::NEG_INFINITY {
        return la;
    }
    la + (-(lb - la).exp()).ln_1p()
}

/// Per-observation log density, its derivative in eta and in the auxiliary
/// parameters.
fn lf(dens: Dens, y: f64, y2: f64, t0: f64, eta: f64, aux: &[f64]) -> (f64, f64, Vec<f64>) {
    match dens {
        Dens::Logit => {
            let p = dist::invlogit(eta);
            let ll = if y != 0.0 { -(-eta).exp().ln_1p() } else { -eta.exp().ln_1p() };
            let ll = if ll.is_finite() { ll } else if y != 0.0 { eta } else { -eta };
            (ll, (if y != 0.0 { 1.0 } else { 0.0 }) - p, vec![])
        }
        Dens::Probit => {
            let q = if y != 0.0 { 1.0 } else { -1.0 };
            let z = q * eta;
            let lp = lnphi(z);
            let lam = (dist::lnnormalden(z) - lp).exp();
            (lp, q * lam, vec![])
        }
        Dens::Cloglog => {
            let m = eta.exp();
            if y != 0.0 {
                // ln(1 - exp(-m))
                let ll = if m < 1e-8 { m.ln() - m / 2.0 } else { (-(-m).exp()).ln_1p() };
                let d = if m > 700.0 { 0.0 } else { m / m.exp_m1() };
                (ll, d, vec![])
            } else {
                (-m, -m, vec![])
            }
        }
        Dens::Ordinal(probit, ncut) => {
            let k = y as usize;
            let (cdf, pdf, sf): (fn(f64) -> f64, fn(f64) -> f64, fn(f64) -> f64) = if probit {
                (dist::normal, dist::normalden, |z| dist::normal(-z))
            } else {
                (dist::invlogit, |z| {
                    let p = dist::invlogit(z);
                    p * (1.0 - p)
                }, |z| dist::invlogit(-z))
            };
            let a = if k < ncut { aux[k] - eta } else { f64::INFINITY };
            let b = if k > 0 { aux[k - 1] - eta } else { f64::NEG_INFINITY };
            let p = if b > 0.0 { sf(b) - if a.is_finite() { sf(a) } else { 0.0 } } else { (if a.is_finite() { cdf(a) } else { 1.0 }) - if b.is_finite() { cdf(b) } else { 0.0 } };
            let p = p.max(1e-300);
            let fa = if a.is_finite() { pdf(a) } else { 0.0 };
            let fb = if b.is_finite() { pdf(b) } else { 0.0 };
            let mut g = vec![0.0; ncut];
            if k < ncut {
                g[k] = fa / p;
            }
            if k > 0 {
                g[k - 1] = -fb / p;
            }
            (p.ln(), -(fa - fb) / p, g)
        }
        Dens::Nb2 => {
            let al = aux[0].exp();
            let m = 1.0 / al;
            let mu = eta.exp();
            let l1 = (al * mu).ln_1p();
            let ll = dist::lngamma(y + m) - dist::lngamma(m) - dist::lngamma(y + 1.0) - (m + y) * l1 + y * (aux[0] + eta);
            let de = (y - mu) / (1.0 + al * mu);
            let da = -m * (dist::digamma(y + m) - dist::digamma(m) - l1) - (m + y) * al * mu / (1.0 + al * mu) + y;
            (ll, de, vec![da])
        }
        Dens::Interval => {
            let ls = aux[0];
            let s = ls.exp();
            let lo = y;
            let hi = y2;
            if lo == hi {
                let z = (lo - eta) / s;
                (-0.5 * z * z - ls - 0.5 * (2.0 * std::f64::consts::PI).ln(), z / s, vec![z * z - 1.0])
            } else if lo == f64::NEG_INFINITY {
                let z = (hi - eta) / s;
                let lp = lnphi(z);
                let lam = (dist::lnnormalden(z) - lp).exp();
                (lp, -lam / s, vec![-lam * z])
            } else if hi == f64::INFINITY {
                let z = (lo - eta) / s;
                let lp = lnphi(-z);
                let lam = (dist::lnnormalden(z) - lp).exp();
                (lp, lam / s, vec![lam * z])
            } else {
                let z1 = (lo - eta) / s;
                let z2 = (hi - eta) / s;
                let p = if z1 > 0.0 { dist::normal(-z1) - dist::normal(-z2) } else { dist::normal(z2) - dist::normal(z1) };
                let p = p.max(1e-300);
                let (f1, f2) = (dist::normalden(z1), dist::normalden(z2));
                (p.ln(), -(f2 - f1) / (s * p), vec![-(f2 * z2 - f1 * z1) / p])
            }
        }
        Dens::Surv(d, aft) => {
            let f = |v: &[f64]| -> f64 {
                let (lh, ch) = crate::streg::lnh_cumh(d, aft, y, v[0], &v[1..]);
                let mut l = -ch;
                if y2 != 0.0 {
                    l += lh;
                }
                if t0 > 0.0 {
                    l += crate::streg::lnh_cumh(d, aft, t0, v[0], &v[1..]).1;
                }
                l
            };
            let mut v = vec![eta];
            v.extend_from_slice(aux);
            let ll = f(&v);
            let g = nd_grad(&f, &v);
            (ll, g[0], g[1..].to_vec())
        }
        Dens::SurvInt(d, aft) => {
            let f = |v: &[f64]| -> f64 {
                let h = |t: f64| crate::streg::lnh_cumh(d, aft, t, v[0], &v[1..]);
                if y == y2 {
                    let (lh, ch) = h(y);
                    lh - ch
                } else if y2 == f64::INFINITY {
                    -h(y).1
                } else if y <= 0.0 {
                    // 1 - S(hi)
                    let ch = h(y2).1;
                    (-(-ch).exp()).ln_1p()
                } else {
                    ln_diff_exp(-h(y).1, -h(y2).1)
                }
            };
            let mut v = vec![eta];
            v.extend_from_slice(aux);
            let ll = f(&v);
            let g = nd_grad(&f, &v);
            (ll, g[0], g[1..].to_vec())
        }
    }
}

// ---------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Unit {
    pub obs: Vec<usize>,
    pub children: Vec<Unit>,
}

/// Nested units from per-level group codes (outermost first).
pub fn build_units(codes: &[Vec<String>], l: usize, obs: &[usize]) -> Vec<Unit> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for &i in obs {
        let key = codes[l][i].clone();
        if !map.contains_key(&key) {
            order.push(key.clone());
        }
        map.entry(key).or_default().push(i);
    }
    order
        .into_iter()
        .map(|key| {
            let o = map.remove(&key).unwrap();
            let children = if l + 1 < codes.len() { build_units(codes, l + 1, &o) } else { vec![] };
            Unit { obs: o, children }
        })
        .collect()
}

pub struct ReModel<'a> {
    pub dens: Dens,
    pub x: Vec<&'a [f64]>,
    pub off: &'a [f64],
    pub y: &'a [f64],
    pub y2: &'a [f64],
    pub t0: &'a [f64],
    pub nlev: usize,
    pub units: Vec<Unit>,
    a: Vec<f64>,
    lw: Vec<f64>,
    cache: RefCell<HashMap<Vec<u32>, (f64, f64)>>,
    frozen: Cell<bool>,
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl<'a> ReModel<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(dens: Dens, x: Vec<&'a [f64]>, off: &'a [f64], y: &'a [f64], y2: &'a [f64], t0: &'a [f64], nlev: usize, units: Vec<Unit>, nq: usize) -> Self {
        let (a, w) = crate::melogit::gauss_hermite(nq.max(1));
        let lw = w.iter().zip(&a).map(|(w, a)| w.ln() + a * a).collect();
        ReModel { dens, x, off, y, y2, t0, nlev, units, a, lw, cache: RefCell::new(HashMap::new()), frozen: Cell::new(false) }
    }
    fn kx(&self) -> usize {
        self.x.len()
    }
    fn naux(&self) -> usize {
        self.dens.naux()
    }
    fn np(&self) -> usize {
        self.kx() + self.naux() + self.nlev
    }

    fn obs_sum(&self, obs: &[usize], off: f64, xb: &[f64], b: &[f64], grad: bool) -> (f64, Vec<f64>) {
        let kx = self.kx();
        let m = self.naux();
        let aux = &b[kx..kx + m];
        let mut ll = 0.0;
        let mut g = if grad { vec![0.0; self.np()] } else { vec![] };
        for &i in obs {
            let (l, d1, da) = lf(self.dens, self.y[i], self.y2[i], self.t0[i], xb[i] + off, aux);
            ll += l;
            if grad {
                for j in 0..kx {
                    g[j] += d1 * self.x[j][i];
                }
                for a in 0..m {
                    g[kx + a] += da[a];
                }
            }
        }
        (ll, g)
    }

    /// Posterior mode and curvature of a lowest-level intercept.
    fn mode(&self, obs: &[usize], off: f64, xb: &[f64], b: &[f64], s2: f64, start: f64) -> (f64, f64) {
        let kx = self.kx();
        let aux = &b[kx..kx + self.naux()];
        let mut u = if start.is_finite() { start } else { 0.0 };
        let mut h2 = -1.0 / s2;
        for _ in 0..100 {
            let (mut g1, mut g2) = (-u / s2, -1.0 / s2);
            for &i in obs {
                let e = xb[i] + off + u;
                let (_, d1, _) = lf(self.dens, self.y[i], self.y2[i], self.t0[i], e, aux);
                let hh = 1e-5 * (1.0 + e.abs());
                let dp = lf(self.dens, self.y[i], self.y2[i], self.t0[i], e + hh, aux).1;
                let dm = lf(self.dens, self.y[i], self.y2[i], self.t0[i], e - hh, aux).1;
                g1 += d1;
                g2 += ((dp - dm) / (2.0 * hh)).min(0.0);
            }
            h2 = g2;
            let step = (-g1 / g2).clamp(-2.0, 2.0);
            if !step.is_finite() {
                break;
            }
            u += step;
            if step.abs() < 1e-10 * (1.0 + u.abs()) {
                break;
            }
        }
        (u, 1.0 / (-h2).sqrt())
    }

    /// Log likelihood (and gradient) of `unit` at level `l`.
    #[allow(clippy::too_many_arguments)]
    fn unit_ll(&self, l: usize, unit: &Unit, path: &mut Vec<u32>, off: f64, xb: &[f64], b: &[f64], grad: bool) -> (f64, Vec<f64>) {
        if l == self.nlev {
            return self.obs_sum(&unit.obs, off, xb, b, grad);
        }
        let pi = self.kx() + self.naux() + l;
        let th = b[pi];
        let s2 = (2.0 * th).exp();
        let lnorm = -0.5 * (2.0 * std::f64::consts::PI).ln() - th;
        let sq2 = std::f64::consts::SQRT_2;
        let nq = self.a.len();
        let deepest = l + 1 == self.nlev;
        let cached = self.cache.borrow().get(path).copied();
        let inner = |me: &Self, path: &mut Vec<u32>, k: usize, u: f64, grad: bool| -> (f64, Vec<f64>) {
            if deepest {
                me.obs_sum(&unit.obs, off + u, xb, b, grad)
            } else {
                let mut ll = 0.0;
                let mut g = if grad { vec![0.0; me.np()] } else { vec![] };
                for (ci, ch) in unit.children.iter().enumerate() {
                    path.push(k as u32);
                    path.push(ci as u32);
                    let (lc, gc) = me.unit_ll(l + 1, ch, path, off + u, xb, b, grad);
                    path.pop();
                    path.pop();
                    ll += lc;
                    if grad {
                        for j in 0..g.len() {
                            g[j] += gc[j];
                        }
                    }
                }
                (ll, g)
            }
        };
        let (mu, sd) = match cached {
            Some(c) if self.frozen.get() => c,
            _ => {
                let (mut mu, mut sd) = match cached {
                    Some(c) => c,
                    None => (0.0, s2.sqrt()),
                };
                if deepest {
                    let r = self.mode(&unit.obs, off, xb, b, s2, mu);
                    if r.0.is_finite() && r.1.is_finite() && r.1 > 0.0 {
                        mu = r.0;
                        sd = r.1;
                    }
                }
                if nq > 1 {
                    for _ in 0..60 {
                        let pts: Vec<f64> = self.a.iter().map(|a| mu + sq2 * sd * a).collect();
                        let lv: Vec<f64> = pts.iter().enumerate().map(|(k, &u)| self.lw[k] + lnorm - u * u / (2.0 * s2) + inner(self, path, k, u, false).0).collect();
                        let tot = lse(&lv);
                        if !tot.is_finite() {
                            break;
                        }
                        let p: Vec<f64> = lv.iter().map(|v| (v - tot).exp()).collect();
                        let m: f64 = pts.iter().zip(&p).map(|(u, p)| u * p).sum();
                        let v: f64 = pts.iter().zip(&p).map(|(u, p)| p * (u - m) * (u - m)).sum();
                        let ns = v.sqrt();
                        if !ns.is_finite() || ns <= 1e-12 || !m.is_finite() {
                            break;
                        }
                        let delta = (m - mu).abs() + (ns - sd).abs();
                        mu = m;
                        sd = ns;
                        if delta < 1e-10 * (1.0 + mu.abs() + sd) {
                            break;
                        }
                    }
                }
                self.cache.borrow_mut().insert(path.clone(), (mu, sd));
                (mu, sd)
            }
        };
        let lsd = (sq2 * sd).ln();
        let mut lv = Vec::with_capacity(nq);
        let mut gs = Vec::with_capacity(nq);
        for k in 0..nq {
            let u = mu + sq2 * sd * self.a[k];
            let (li, mut gi) = inner(self, path, k, u, grad);
            lv.push(self.lw[k] + lsd + lnorm - u * u / (2.0 * s2) + li);
            if grad {
                gi[pi] += -1.0 + u * u / s2;
                gs.push(gi);
            }
        }
        let tot = lse(&lv);
        if !grad {
            return (tot, vec![]);
        }
        let mut g = vec![0.0; self.np()];
        for k in 0..nq {
            let p = (lv[k] - tot).exp();
            if p == 0.0 {
                continue;
            }
            for j in 0..g.len() {
                g[j] += p * gs[k][j];
            }
        }
        (tot, g)
    }

    fn total(&self, b: &[f64], scores: bool) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let n = self.y.len();
        let xb: Vec<f64> = (0..n).map(|i| self.off[i] + self.x.iter().zip(b).map(|(c, bj)| c[i] * bj).sum::<f64>()).collect();
        let p = self.np();
        let mut ll = 0.0;
        let mut g = vec![0.0; p];
        let mut sc = Vec::new();
        for (ui, u) in self.units.iter().enumerate() {
            let mut path = vec![ui as u32];
            let (l, gg) = self.unit_ll(0, u, &mut path, 0.0, &xb, b, true);
            ll += l;
            for j in 0..p {
                g[j] += gg[j];
            }
            if scores {
                sc.push(gg);
            }
        }
        (ll, g, sc)
    }
}

impl<'a> Likelihood for ReModel<'a> {
    fn k(&self) -> usize {
        self.np()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let p = self.np();
        self.frozen.set(false);
        let (ll, g, sc) = self.total(b, need_scores);
        if !ll.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return (f64::NEG_INFINITY, vec![0.0; p], Matrix::identity(p), sc);
        }
        self.frozen.set(true);
        let mut h = Matrix::zeros(p, p);
        let mut bb = b.to_vec();
        for j in 0..p {
            let step = 1e-5 * (1.0 + b[j].abs());
            bb[j] = b[j] + step;
            let gp = self.total(&bb, false).1;
            bb[j] = b[j] - step;
            let gm = self.total(&bb, false).1;
            bb[j] = b[j];
            for i in 0..p {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * step);
            }
        }
        self.frozen.set(false);
        for i in 0..p {
            for j in 0..i {
                let m = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = m;
                h[(j, i)] = m;
            }
        }
        (ll, g, h, sc)
    }
}

// ---------------------------------------------------------------------------
// fitting driver
// ---------------------------------------------------------------------------

/// Response data prepared for a density.
pub struct RespData {
    pub y: Vec<f64>,
    pub y2: Vec<f64>,
    pub t0: Vec<f64>,
    /// ordered categories (for ordinal responses)
    pub cats: Vec<f64>,
}

pub struct FitResult {
    pub b: Vec<f64>,
    pub v: Matrix,
    pub ll: f64,
    pub ll_c: f64,
    pub converged: bool,
    pub ic: usize,
    pub log: Vec<String>,
    pub n_clust: Option<usize>,
}

/// Starting values of the pooled model: (fixed part, auxiliary part).
fn pooled_start(dens: Dens, d: &Design, act: &[usize], r: &RespData) -> Vec<f64> {
    let n = r.y.len() as f64;
    let kx = act.len();
    let mut b = vec![0.0; kx + dens.naux()];
    let cons = act.iter().position(|&j| d.names[j] == "_cons");
    match dens {
        Dens::Logit | Dens::Probit | Dens::Cloglog => {
            let m = (r.y.iter().sum::<f64>() / n).clamp(1e-4, 1.0 - 1e-4);
            if let Some(c) = cons {
                b[c] = match dens {
                    Dens::Logit => dist::logit(m),
                    Dens::Probit => dist::invnormal(m),
                    _ => (-(1.0 - m).ln()).ln(),
                };
            }
        }
        Dens::Ordinal(probit, ncut) => {
            let mut cum = 0.0;
            for k in 0..ncut {
                cum += r.y.iter().filter(|&&v| v as usize == k).count() as f64 / n;
                let c = cum.clamp(1e-4, 1.0 - 1e-4);
                b[kx + k] = if probit { dist::invnormal(c) } else { dist::logit(c) };
            }
        }
        Dens::Nb2 => {
            if let Some(c) = cons {
                b[c] = (r.y.iter().sum::<f64>() / n).max(1e-3).ln();
            }
        }
        Dens::Interval => {
            let vals: Vec<f64> = r.y.iter().zip(&r.y2).map(|(a, b)| if a.is_finite() && b.is_finite() { (a + b) / 2.0 } else if a.is_finite() { *a } else { *b }).collect();
            let m = util::mean(&vals);
            let s = util::variance(&vals).sqrt().max(1e-3);
            if let Some(c) = cons {
                b[c] = m;
            }
            b[kx] = s.ln();
        }
        Dens::Surv(dd, aft) | Dens::SurvInt(dd, aft) => {
            let (ev, tt) = match dens {
                Dens::Surv(..) => (r.y2.iter().sum::<f64>(), r.y.iter().zip(&r.t0).map(|(t, t0)| t - t0).sum::<f64>()),
                _ => (r.y2.iter().filter(|v| v.is_finite()).count() as f64, r.y.iter().zip(&r.y2).map(|(a, b)| if b.is_finite() { (a + b) / 2.0 } else { *a }).sum::<f64>()),
            };
            let rate = (ev.max(1.0) / tt.max(1e-8)).max(1e-8);
            let ph_like = matches!(dd, Dist::Exponential | Dist::Weibull | Dist::Gompertz) && !aft;
            if let Some(c) = cons {
                b[c] = if ph_like { rate.ln() } else { -rate.ln() };
            }
        }
    }
    b
}

/// Fit the pooled model and the random-intercept model.
#[allow(clippy::too_many_arguments)]
pub fn fit(dens: Dens, d: &Design, act: &[usize], off: &[f64], r: &RespData, codes: &[Vec<String>], nq: usize, cluster: Option<&[f64]>, ll_label: &str, pooled_title: &str) -> Result<FitResult, String> {
    let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
    let n = r.y.len();
    let mut log = Vec::new();
    let single: Vec<Unit> = (0..n).map(|i| Unit { obs: vec![i], children: vec![] }).collect();
    let pooled = ReModel::new(dens, cols.clone(), off, &r.y, &r.y2, &r.t0, 0, single, 1);
    let b0 = pooled_start(dens, d, act, r);
    let pres = crate::zip::maximize_robust(&pooled, b0, 300)?;
    let nlev = codes.len();
    if nlev == 0 {
        let (hinv, _) = pres.info.sym_inverse(1e-14);
        let (v, n_clust) = match cluster {
            Some(c) => {
                let (v, g) = est::sandwich(&hinv, &pres.scores, Some(c), 1.0);
                let gf = g as f64;
                (v.scale(gf / (gf - 1.0)), Some(g))
            }
            None => (hinv, None),
        };
        log.extend(est::iteration_log(&pres.iterations, ll_label));
        return Ok(FitResult { b: pres.b, v, ll: pres.ll, ll_c: SYSMISS, converged: pres.converged, ic: pres.iterations.len() - 1, log, n_clust });
    }
    log.push(String::new());
    log.push(pooled_title.to_string());
    log.push(String::new());
    log.extend(est::iteration_log(&pres.iterations, ll_label));
    let all: Vec<usize> = (0..n).collect();
    let units = build_units(codes, 0, &all);
    let model = ReModel::new(dens, cols.clone(), off, &r.y, &r.y2, &r.t0, nlev, units, nq);
    let mut best = (f64::NEG_INFINITY, vec![]);
    log.push(String::new());
    log.push("Refining starting values:".into());
    log.push(String::new());
    for (gi, lsd) in [-0.5f64, -1.5, 0.0, 0.5, -3.0].iter().enumerate() {
        let mut s = pres.b.clone();
        s.extend(std::iter::repeat(*lsd).take(nlev));
        model.cache.borrow_mut().clear();
        let ll = model.total(&s, false).0;
        if gi == 0 {
            log.push(format!("Grid node 0:   {ll_label} = {}", fmt_g(ll, 9)));
        }
        if ll > best.0 {
            best = (ll, s);
        }
    }
    if best.1.is_empty() {
        return Err(rc_err(430, "initial values not feasible"));
    }
    model.cache.borrow_mut().clear();
    let res = crate::zip::maximize_robust(&model, best.1, 200)?;
    log.push(String::new());
    log.push("Fitting full model:".into());
    log.push(String::new());
    log.extend(est::iteration_log(&res.iterations, ll_label));
    let (hinv, _) = res.info.sym_inverse(1e-14);
    let (v, n_clust) = match cluster {
        Some(c) => {
            let (v, g) = est::sandwich(&hinv, &res.scores, Some(c), 1.0);
            let gf = g as f64;
            (v.scale(gf / (gf - 1.0)), Some(g))
        }
        None => (hinv, None),
    };
    Ok(FitResult { b: res.b, v, ll: res.ll, ll_c: pres.ll, converged: res.converged, ic: res.iterations.len() - 1, log, n_clust })
}

// ---------------------------------------------------------------------------
// posting: transformed parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Tr {
    Id,
    /// exp(theta)
    Exp,
    /// exp(2 theta): variance from a log standard deviation
    Exp2,
}

/// Map internal parameters to posted ones: (name, internal index, transform).
fn post_params(names_int: &[(String, usize, Tr)], b: &[f64], v: &Matrix) -> (Vec<String>, Vec<f64>, Matrix) {
    let p = names_int.len();
    let mut bo = vec![0.0; p];
    let mut jac = vec![0.0; p];
    for (a, (_, j, t)) in names_int.iter().enumerate() {
        let x = b[*j];
        let (val, d) = match t {
            Tr::Id => (x, 1.0),
            Tr::Exp => (x.exp(), x.exp()),
            Tr::Exp2 => ((2.0 * x).exp(), 2.0 * (2.0 * x).exp()),
        };
        bo[a] = val;
        jac[a] = d;
    }
    let mut vo = Matrix::zeros(p, p);
    for (a, (_, i, _)) in names_int.iter().enumerate() {
        for (c, (_, j, _)) in names_int.iter().enumerate() {
            vo[(a, c)] = jac[a] * v[(*i, *j)] * jac[c];
        }
    }
    (names_int.iter().map(|x| x.0.clone()).collect(), bo, vo)
}

// ---------------------------------------------------------------------------
// response preparation shared by the commands
// ---------------------------------------------------------------------------

struct Prepared {
    d: Design,
    act: Vec<usize>,
    resp: RespData,
    off: Vec<f64>,
    offset: Option<String>,
    dep: String,
    notes: Vec<String>,
}

fn num_or_var(ds: &DataSet, s: &str, rows: &[usize]) -> Result<Vec<f64>, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Ok(vec![v; rows.len()]);
    }
    let name = crate::varlist::resolve_name(ds, s)?;
    let c = util::col(ds, &name)?;
    Ok(rows.iter().map(|&i| c[i]).collect())
}

/// Build the design and response for `kind` from `main` = "depvar(s) indepvars".
fn prepare(ds: &DataSet, st: &Store, cmd: &Command, dens_kind: &str, extra: Vec<String>) -> Result<Prepared, String> {
    if cmd.weight.is_some() {
        return Err(rc_err(101, "weights not allowed"));
    }
    let main = cmd.main.trim();
    let mut words = main.split_whitespace();
    let mut notes = Vec::new();
    let surv = dens_kind == "surv";
    let two = dens_kind == "intreg";
    let (dep, dep2) = if surv {
        crate::survival::require_st(ds)?;
        ("_t".to_string(), None)
    } else {
        let a = words.next().ok_or_else(|| rc_err(100, "varlist required"))?;
        let a = crate::varlist::resolve_name(ds, a)?;
        let b = if two { Some(crate::varlist::resolve_name(ds, words.next().ok_or_else(|| rc_err(102, "too few variables specified"))?)?) } else { None };
        (a, b)
    };
    let indep: String = if surv { main.to_string() } else { words.collect::<Vec<_>>().join(" ") };
    let mut ex = extra;
    let offv = match cmd.opt("offset", 3) {
        Some(o) => Some(crate::varlist::resolve_name(ds, o.trim())?),
        None => None,
    };
    let expv = match cmd.opt("exposure", 1) {
        Some(o) => Some(crate::varlist::resolve_name(ds, o.trim())?),
        None => None,
    };
    ex.extend(offv.iter().cloned());
    ex.extend(expv.iter().cloned());
    let mut cmd2 = cmd.clone();
    if surv {
        cmd2 = crate::stcox::with_st_if(cmd)?;
        ex.push("_t0".into());
        ex.push("_d".into());
    }
    let d = if two {
        // either bound may be missing: build without a depvar, drop rows missing both
        let mut d = est::build(ds, st, &cmd2, DesignSpec { depvar: None, indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra: ex })?;
        let y1 = util::col(ds, &dep)?;
        let y2 = util::col(ds, dep2.as_ref().unwrap())?;
        let keep: Vec<bool> = d.rows.iter().map(|&i| !(is_missing(y1[i]) && is_missing(y2[i]))).collect();
        d = crate::xtnl::subset_design(&d, &keep);
        d.depvar = dep.clone();
        d
    } else {
        est::build(ds, st, &cmd2, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra: ex })?
    };
    if d.n() == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    let n = d.n();
    let mut off = vec![0.0; n];
    let mut offset = None;
    if let Some(o) = &offv {
        let c = util::col(ds, o)?;
        for (t, &i) in d.rows.iter().enumerate() {
            off[t] += c[i];
        }
        offset = Some(o.clone());
    }
    if let Some(o) = &expv {
        let c = util::col(ds, o)?;
        for (t, &i) in d.rows.iter().enumerate() {
            if c[i] <= 0.0 {
                return Err(rc_err(459, format!("{o} must be greater than zero")));
            }
            off[t] += c[i].ln();
        }
        offset = Some(format!("ln({o})"));
    }
    let act = d.active();
    let mut resp = RespData { y: d.y.clone(), y2: vec![0.0; n], t0: vec![0.0; n], cats: vec![] };
    match dens_kind {
        "binary" => {
            for y in resp.y.iter_mut() {
                *y = if *y != 0.0 { 1.0 } else { 0.0 };
            }
        }
        "ordinal" => {
            let mut cats = d.y.clone();
            cats.sort_by(f64::total_cmp);
            cats.dedup();
            if cats.len() < 2 {
                return Err(rc_err(2000, "outcome does not vary"));
            }
            for y in resp.y.iter_mut() {
                *y = cats.iter().position(|c| c == y).unwrap() as f64;
            }
            resp.cats = cats;
        }
        "count" => {
            if d.y.iter().any(|v| *v < 0.0) {
                return Err(rc_err(459, format!("{dep} must be greater than or equal to zero")));
            }
        }
        "tobit" => {
            let lo = match cmd.opt("ll", 2) {
                Some(s) if !s.trim().is_empty() => Some(num_or_var(ds, s, &d.rows)?),
                Some(_) => Some(vec![d.y.iter().cloned().fold(f64::INFINITY, f64::min); n]),
                None if cmd.has_opt("ll", 2) => Some(vec![d.y.iter().cloned().fold(f64::INFINITY, f64::min); n]),
                None => None,
            };
            let hi = match cmd.opt("ul", 2) {
                Some(s) if !s.trim().is_empty() => Some(num_or_var(ds, s, &d.rows)?),
                Some(_) => Some(vec![d.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max); n]),
                None if cmd.has_opt("ul", 2) => Some(vec![d.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max); n]),
                None => None,
            };
            let (mut nl, mut nu) = (0, 0);
            for t in 0..n {
                let y = d.y[t];
                resp.y[t] = y;
                resp.y2[t] = y;
                if let Some(l) = &lo {
                    if !is_missing(l[t]) && y <= l[t] {
                        resp.y[t] = f64::NEG_INFINITY;
                        resp.y2[t] = l[t];
                        nl += 1;
                        continue;
                    }
                }
                if let Some(u) = &hi {
                    if !is_missing(u[t]) && y >= u[t] {
                        resp.y[t] = u[t];
                        resp.y2[t] = f64::INFINITY;
                        nu += 1;
                    }
                }
            }
            notes.push(format!("{}|{}|{}", n - nl - nu, nl, nu));
        }
        "intreg" => {
            let y1 = util::col(ds, &dep)?;
            let y2 = util::col(ds, dep2.as_ref().unwrap())?;
            let (mut nu, mut nl, mut nr, mut ni) = (0, 0, 0, 0);
            for (t, &i) in d.rows.iter().enumerate() {
                let a = if is_missing(y1[i]) { f64::NEG_INFINITY } else { y1[i] };
                let b = if is_missing(y2[i]) { f64::INFINITY } else { y2[i] };
                if a > b {
                    return Err(rc_err(498, format!("{} > {} in some observations", dep, dep2.as_ref().unwrap())));
                }
                resp.y[t] = a;
                resp.y2[t] = b;
                if a == b {
                    nu += 1;
                } else if a == f64::NEG_INFINITY {
                    nl += 1;
                } else if b == f64::INFINITY {
                    nr += 1;
                } else {
                    ni += 1;
                }
            }
            notes.push(format!("{nu}|{nl}|{nr}|{ni}"));
        }
        "surv" => {
            let t0 = util::col(ds, "_t0")?;
            let dd = util::col(ds, "_d")?;
            for (t, &i) in d.rows.iter().enumerate() {
                resp.t0[t] = t0[i];
                resp.y2[t] = if dd[i] != 0.0 { 1.0 } else { 0.0 };
            }
        }
        _ => {}
    }
    let mut d = d;
    if dens_kind == "ordinal" {
        // no constant in ordinal models
        if let Some(j) = d.names.iter().position(|n| n == "_cons") {
            d.omitted[j] = true;
            d.names.remove(j);
            d.cols.remove(j);
            d.omitted.remove(j);
            d.is_base.remove(j);
            d.has_cons = false;
        }
    }
    let act = if dens_kind == "ordinal" { d.active() } else { act };
    Ok(Prepared { d, act, resp, off, offset, dep: if two { format!("{dep} {}", dep2.unwrap()) } else { dep }, notes })
}

// ---------------------------------------------------------------------------
// display helpers
// ---------------------------------------------------------------------------

fn cell(v: f64) -> String {
    if v.is_finite() {
        fmt_g(v, 8)
    } else {
        ".".into()
    }
}

/// Row with z and p (ancillary parameters on their natural metric).
fn zrow(lw: usize, label: &str, b: f64, se: f64, crit: f64) -> String {
    let z = b / se;
    let p = 2.0 * (1.0 - dist::normal(z.abs()));
    format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", label, cell(b), cell(se), z, p, cell(b - crit * se), cell(b + crit * se))
}

/// Row without a test (variance-type parameters; CI from the log metric).
fn vrow(lw: usize, label: &str, b: f64, se: f64, crit: f64, logci: bool) -> String {
    let (lo, hi) = if logci && b > 0.0 {
        let sl = se / b;
        ((b.ln() - crit * sl).exp(), (b.ln() + crit * sl).exp())
    } else {
        (b - crit * se, b + crit * se)
    };
    format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", label, cell(b), cell(se), "", cell(lo), cell(hi))
}

fn rule(lw: usize) -> String {
    format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64))
}

/// Fixed-part coefficient table (without the closing rule) and label width.
fn fixed_table(e: &Results, eform: Option<&str>, level: f64) -> (Vec<String>, usize, String) {
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let dep_show = dep.split_whitespace().next().unwrap_or("").to_string();
    let kf = e.get("k_f").unwrap_or(0.0) as usize;
    let names: Vec<String> = b.colnames[..kf].iter().map(|n| n.split_once(':').map(|x| x.1.to_string()).unwrap_or_else(|| n.clone())).collect();
    let is_base: Vec<bool> = names.iter().map(|c| c.split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = names.iter().zip(&is_base).map(|(c, ib)| *ib || c.starts_with("o.")).collect();
    let mut fv = Matrix::zeros(kf, kf);
    for i in 0..kf {
        for j in 0..kf {
            fv[(i, j)] = v[(i, j)];
        }
    }
    let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep_show, names: &names, b: &b.data[..kf], v: &fv, omitted: &omitted, is_base: &is_base, df: None, level, eform, coef_title: "Coefficient" });
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    (tbl, lw, bar)
}

fn chibar_line(e: &Results, label: &str) -> Option<String> {
    let c = e.get("chi2_c")?;
    let df = e.get("df_c").unwrap_or(1.0);
    if df <= 1.0 {
        let p = e.get("p_c").unwrap_or(SYSMISS);
        Some(format!("{:<54}Prob >= chibar2 = {:.4}", format!("{label}: chibar2(01) = {}", if c >= 1e4 { format!("{:.1e}", c).replace("e", "e+") } else { format!("{:.2}", c) }), p))
    } else {
        Some(format!("{:<54}Prob > chi2 = {:.4}", format!("{label}: chi2({}) = {:.2}", fmt_g(df, 9), c), e.get("p_c").unwrap_or(SYSMISS)))
    }
}

// ---------------------------------------------------------------------------
// xt commands
// ---------------------------------------------------------------------------

pub struct XtRe {
    pub verb: &'static str,
}

fn dens_kind(verb: &str) -> &'static str {
    match verb {
        "xtologit" | "xtoprobit" | "meologit" | "meoprobit" => "ordinal",
        "xttobit" | "metobit" => "tobit",
        "xtintreg" | "meintreg" => "intreg",
        "xtcloglog" | "mecloglog" => "binary",
        "menbreg" => "count",
        _ => "surv",
    }
}

/// Survival distribution and metric from options (default exponential/PH).
fn surv_dist(cmd: &Command) -> Result<(Dist, bool), String> {
    let d = match cmd.opt("distribution", 1) {
        Some(s) => Dist::parse(s)?,
        None => return Err(rc_err(198, "option distribution() required")),
    };
    let aft = match d {
        Dist::Exponential | Dist::Weibull => cmd.has_opt("time", 2),
        Dist::Gompertz => false,
        _ => true,
    };
    Ok((d, aft))
}

fn obs_per_group(units_sizes: &[usize]) -> (f64, f64, f64) {
    let mn = units_sizes.iter().min().copied().unwrap_or(0) as f64;
    let mx = units_sizes.iter().max().copied().unwrap_or(0) as f64;
    let n: usize = units_sizes.iter().sum();
    (mn, n as f64 / units_sizes.len().max(1) as f64, mx)
}

fn group_codes(ds: &DataSet, var: &str, rows: &[usize]) -> Result<Vec<String>, String> {
    let v = ds.get_var(var).ok_or_else(|| rc_err(111, format!("variable {var} not found")))?;
    Ok(rows
        .iter()
        .map(|&i| match &v.data[i] {
            Value::Str(s) => s.clone(),
            x => fmt_g(x.num(), 17),
        })
        .collect())
}

impl CommandHandler for XtRe {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() && self.verb != "xtstreg" {
            if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(render_xt(&st.e, cmd.has_opt("or", 2) || cmd.has_opt("eform", 2) || cmd.has_opt("tr", 2), util::level(cmd, st)?));
        }
        if self.verb == "xtcloglog" && cmd.has_opt("pa", 2) {
            let line = format!("xtgee {}{}, family(binomial) link(cloglog) corr(exchangeable) {}", cmd.main.trim(), cmd.if_raw.as_ref().map(|s| format!(" if {s}")).unwrap_or_default(), cmd.opts.iter().filter(|(n, _)| n != "pa").map(|(n, a)| a.as_ref().map(|a| format!("{n}({a})")).unwrap_or(n.clone())).collect::<Vec<_>>().join(" "));
            let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
            let out = crate::xt_gee::XtGee.execute(ds, st, &c)?;
            st.e.set_str("cmd2", "xtcloglog");
            return Ok(out);
        }
        if cmd.has_opt("fe", 2) || cmd.has_opt("pa", 2) && self.verb != "xtcloglog" {
            return Err(rc_err(198, format!("{}: only random-effects models are available", self.verb)));
        }
        let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
        let kind = dens_kind(self.verb);
        let p = prepare(ds, st, cmd, kind, vec![ivar.clone()])?;
        let dens = match self.verb {
            "xtologit" => Dens::Ordinal(false, p.resp.cats.len() - 1),
            "xtoprobit" => Dens::Ordinal(true, p.resp.cats.len() - 1),
            "xttobit" | "xtintreg" => Dens::Interval,
            "xtcloglog" => Dens::Cloglog,
            _ => {
                let (d, aft) = surv_dist(cmd)?;
                Dens::Surv(d, aft)
            }
        };
        let nq = util::opt_num(cmd, "intpoints", 3)?.unwrap_or(12.0) as usize;
        if !(1..=195).contains(&nq) {
            return Err(rc_err(198, "intpoints() must be between 1 and 195"));
        }
        let codes = vec![group_codes(ds, &ivar, &p.d.rows)?];
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let all: Vec<usize> = (0..p.d.n()).collect();
        let units = build_units(&codes, 0, &all);
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Robust => Some((0..units.len()).map(|g| g as f64).collect()),
            Vce::Cluster(c) => {
                let name = crate::varlist::resolve_name(ds, c)?;
                let col = util::col(ds, &name)?;
                Some(units.iter().map(|u| col[p.d.rows[u.obs[0]]]).collect())
            }
            _ => None,
        };
        let pooled_title = match self.verb {
            "xtologit" => "Fitting comparison model:",
            _ => "Fitting comparison model:",
        };
        let ll_label = "Log likelihood";
        let fitr = fit(dens, &p.d, &p.act, &p.off, &p.resp, &codes, nq, cl.as_deref(), ll_label, pooled_title)?;
        let sizes: Vec<usize> = units.iter().map(|u| u.obs.len()).collect();
        let (mn, av, mx) = obs_per_group(&sizes);
        let kx = p.act.len();
        let m = dens.naux();
        let k = p.d.k();
        // posted parameters
        let mut post: Vec<(String, usize, Tr)> = Vec::new();
        let dep0 = p.dep.split_whitespace().next().unwrap_or("").to_string();
        let eqn = if kind == "surv" { "_t".to_string() } else { dep0.clone() };
        let mut amap = vec![usize::MAX; k];
        for (a, &j) in p.act.iter().enumerate() {
            amap[j] = a;
        }
        for (j, nm) in p.d.names.iter().enumerate() {
            post.push((format!("{eqn}:{nm}"), amap[j], Tr::Id));
        }
        let lnsu = kx + m;
        match dens {
            Dens::Ordinal(..) => {
                for c in 0..m {
                    post.push((format!("/:cut{}", c + 1), kx + c, Tr::Id));
                }
                post.push(("/:sigma2_u".into(), lnsu, Tr::Exp2));
            }
            Dens::Interval => {
                post.push(("/:sigma_u".into(), lnsu, Tr::Exp));
                post.push(("/:sigma_e".into(), kx, Tr::Exp));
            }
            Dens::Cloglog => {
                // lnsig2u = 2 ln sigma_u: posted on the log-variance metric below
                post.push(("/:lnsig2u".into(), lnsu, Tr::Id));
            }
            Dens::Surv(d, _) => {
                for (a, nm) in d.anc().iter().enumerate() {
                    post.push((format!("/:{nm}"), kx + a, Tr::Id));
                }
                post.push(("/:sigma2_u".into(), lnsu, Tr::Exp2));
            }
            _ => {}
        }
        // internal VCE extended with a zero slot for omitted columns
        let pint = fitr.b.len();
        let mut bint = fitr.b.clone();
        bint.push(0.0);
        let mut vint = Matrix::zeros(pint + 1, pint + 1);
        for i in 0..pint {
            for j in 0..pint {
                vint[(i, j)] = fitr.v[(i, j)];
            }
        }
        let post: Vec<(String, usize, Tr)> = post.into_iter().map(|(n, i, t)| (n, if i == usize::MAX { pint } else { i }, t)).collect();
        let (names, mut bo, mut vo) = post_params(&post, &bint, &vint);
        if dens == Dens::Cloglog {
            // lnsig2u = 2 * ln sigma_u
            let j = names.iter().position(|n| n == "/:lnsig2u").unwrap();
            bo[j] *= 2.0;
            for a in 0..names.len() {
                vo[(a, j)] *= 2.0;
                vo[(j, a)] *= 2.0;
            }
        }
        let mut d = p.d.clone();
        d.names = names.clone();
        d.omitted.extend(vec![false; names.len() - k]);
        d.is_base.extend(vec![false; names.len() - k]);
        let mut dk = p.d.clone();
        let mut vk = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                vk[(i, j)] = vo[(i, j)];
            }
        }
        dk.names = p.d.names.clone();
        let (chi2, df_m) = est::wald_all(&dk, &bo[..k], &vk);
        let lr = if cl.is_none() { (2.0 * (fitr.ll - fitr.ll_c)).max(0.0) } else { SYSMISS };
        let s2u = (2.0 * fitr.b[lnsu]).exp();
        let title = match self.verb {
            "xtologit" => "Random-effects ordered logistic regression",
            "xtoprobit" => "Random-effects ordered probit regression",
            "xttobit" => "Random-effects tobit regression",
            "xtintreg" => "Random-effects interval regression",
            "xtcloglog" => "Random-effects complementary log-log model",
            _ => "Random-effects parametric survival model",
        };
        let mut head = Vec::new();
        head.extend(p.d.notes.iter().cloned());
        head.extend(fitr.log.iter().cloned());
        let head = head.join("\n");
        let verb = self.verb;
        let ng = units.len();
        let level = util::level(cmd, st)?;
        let cats = p.resp.cats.clone();
        let cens = p.notes.first().cloned().unwrap_or_default();
        let robust = cl.is_some();
        let clustvar = match &vce {
            Vce::Cluster(c) => c.clone(),
            _ => ivar.clone(),
        };
        let offset = p.offset.clone();
        let sd_e = if dens == Dens::Interval { fitr.b[kx].exp() } else { SYSMISS };
        est::post(st, verb, &d, &bo, &vo, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            e.set("ll", fitr.ll);
            e.set("ll_c", fitr.ll_c);
            if !is_missing(lr) {
                e.set("chi2_c", lr);
                e.set("p_c", 0.5 * dist::chi2tail(1.0, lr));
                e.set("df_c", 1.0);
            }
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("k", pint as f64);
            e.set("k_f", k as f64);
            e.set("rank", pint as f64);
            e.set("n_quad", nq as f64);
            e.set("ic", fitr.ic as f64);
            e.set("converged", if fitr.converged { 1.0 } else { 0.0 });
            e.set("sigma_u", s2u.sqrt());
            match dens {
                Dens::Ordinal(probit, ncut) => {
                    e.set("k_cat", (ncut + 1) as f64);
                    e.set("rho", s2u / (s2u + if probit { 1.0 } else { std::f64::consts::PI.powi(2) / 3.0 }));
                    e.set_mat("cat", Matrix::row_vector(&cats));
                }
                Dens::Interval => {
                    e.set("sigma_e", sd_e);
                    e.set("rho", s2u / (s2u + sd_e * sd_e));
                }
                Dens::Cloglog => {
                    e.set("rho", s2u / (s2u + std::f64::consts::PI.powi(2) / 6.0));
                }
                _ => {}
            }
            if !cens.is_empty() {
                let c: Vec<f64> = cens.split('|').map(|x| x.parse().unwrap_or(0.0)).collect();
                if c.len() == 3 {
                    e.set("N_unc", c[0]);
                    e.set("N_lc", c[1]);
                    e.set("N_rc", c[2]);
                } else if c.len() == 4 {
                    e.set("N_unc", c[0]);
                    e.set("N_lc", c[1]);
                    e.set("N_rc", c[2]);
                    e.set("N_int", c[3]);
                }
            }
            if let Some(g) = fitr.n_clust {
                e.set("N_clust", g as f64);
            }
            e.set_str("model", "re");
            e.set_str("ivar", ivar.clone());
            e.set_str("title", title);
            e.set_str("distrib", "Gaussian");
            e.set_str("intmethod", if nq == 1 { "laplace" } else { "mvaghermite" });
            e.set_str("chi2type", "Wald");
            if robust {
                e.set_str("vce", "cluster");
                e.set_str("vcetype", "Robust");
                e.set_str("clustvar", clustvar.clone());
            } else {
                e.set_str("vce", "oim");
            }
            if let Dens::Surv(dd, aft) = dens {
                e.set_str("distribution", dd.label().to_lowercase());
                e.set_str("frm2", if aft { "time" } else { "hazard" });
            }
            if let Some(o) = &offset {
                e.set_str("offset", o.clone());
            }
            e.set_str("predict", format!("{verb}_p"));
            e.set_str("_head", head.clone());
        });
        let eform = cmd.has_opt("or", 2) || cmd.has_opt("eform", 2) || cmd.has_opt("tr", 2);
        let mut text = format!("{head}\n{}", render_xt(&st.e, eform, level));
        if !fitr.converged {
            text.push_str("\nconvergence not achieved");
        }
        Ok(text)
    }
}

fn render_xt(e: &Results, eform: bool, level: f64) -> String {
    use crate::xtnl::{header_block, hl, hv};
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let mac = |s: &str| e.macros.get(s).cloned().unwrap_or_default();
    let cmd = mac("cmd");
    let mut h = vec![hl(""), hv(&mac("title"), "Number of obs", util::commas(get("N") as usize)), hv(&format!("Group variable: {}", mac("ivar")), "Number of groups", util::commas(get("N_g") as usize)), hl("")];
    h.push(hv("Random effects u_i ~ Gaussian", "Obs per group:", String::new()));
    h.push(hv("", &format!("{:>16}", "min"), util::commas(get("g_min") as usize)));
    h.push(hv("", &format!("{:>16}", "avg"), format!("{:.1}", get("g_avg"))));
    h.push(hv("", &format!("{:>16}", "max"), util::commas(get("g_max") as usize)));
    h.push(hl(""));
    if let Some(nu) = e.get("N_unc") {
        h.pop();
        h.push(hv("", "Censoring:", String::new()));
        h.push(hv("", &format!("{:>16}", "Uncensored"), util::commas(nu as usize)));
        h.push(hv("", &format!("{:>16}", "Left-censored"), util::commas(get("N_lc") as usize)));
        h.push(hv("", &format!("{:>16}", "Right-censored"), util::commas(get("N_rc") as usize)));
        if let Some(ni) = e.get("N_int") {
            h.push(hv("", &format!("{:>16}", "Interval-cens."), util::commas(ni as usize)));
        }
        h.push(hl(""));
    }
    h.push(hv(&format!("Integration method: {}", mac("intmethod")), "Integration pts.", fmt_g(get("n_quad"), 9)));
    h.push(hl(""));
    h.push(hv("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
    h.push(hv(&format!("Log likelihood = {}", fmt_g(get("ll"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    let mut out = header_block(&h);
    if !mac("vcetype").is_empty() {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(get("N_clust") as usize), mac("clustvar"))));
    }
    out.push(String::new());
    let ef = if eform {
        Some(match cmd.as_str() {
            "xtologit" => "Odds ratio",
            "xtcloglog" => "exp(b)",
            "xtstreg" if mac("frm2") == "hazard" => "Haz. ratio",
            "xtstreg" => "Time ratio",
            _ => "exp(b)",
        })
    } else {
        None
    };
    let (mut tbl, lw, bar) = fixed_table(e, ef, level);
    let b = e.matrices.get("b").cloned().unwrap();
    let v = e.matrices.get("V").cloned().unwrap();
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let kf = get("k_f") as usize;
    let mut prev_kind = "";
    for j in kf..b.cols {
        let nm = b.colnames[j].trim_start_matches("/:").to_string();
        let bj = b.data[j];
        let se = v[(j, j)].max(0.0).sqrt();
        let kind = if nm.starts_with("cut") || nm == "ln_p" || nm == "gamma" || nm.starts_with("ln_") || nm == "kappa" { "z" } else if nm == "lnsig2u" { "plain" } else { "var" };
        if kind != prev_kind || kind == "var" && prev_kind == "" {
            tbl.push(rule(lw));
        }
        prev_kind = kind;
        let label = format!("/{nm}");
        match kind {
            "z" => tbl.push(zrow(lw, &label, bj, se, crit)),
            "plain" => tbl.push(zrow(lw, &label, bj, se, crit)),
            _ => tbl.push(vrow(lw, &label, bj, se, crit, true)),
        }
    }
    // derived rows
    let pos = |n: &str| b.colnames.iter().position(|c| c == n);
    if let Some(j) = pos("/:lnsig2u") {
        let t = b.data[j];
        let se = v[(j, j)].max(0.0).sqrt();
        let s = (t / 2.0).exp();
        tbl.push(rule(lw));
        tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", "sigma_u", cell(s), cell(0.5 * s * se), "", cell(((t - crit * se) / 2.0).exp()), cell(((t + crit * se) / 2.0).exp())));
        let c = std::f64::consts::PI.powi(2) / 6.0;
        let rho = |t: f64| t.exp() / (t.exp() + c);
        let r = rho(t);
        tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", "rho", cell(r), cell(r * (1.0 - r) * se), "", cell(rho(t - crit * se)), cell(rho(t + crit * se))));
    }
    if let (Some(ju), Some(je)) = (pos("/:sigma_u"), pos("/:sigma_e")) {
        let (su, se_) = (b.data[ju], b.data[je]);
        let r = su * su / (su * su + se_ * se_);
        // delta method for rho
        let gu = 2.0 * su * se_ * se_ / (su * su + se_ * se_).powi(2);
        let ge = -2.0 * se_ * su * su / (su * su + se_ * se_).powi(2);
        let var = gu * gu * v[(ju, ju)] + ge * ge * v[(je, je)] + 2.0 * gu * ge * v[(ju, je)];
        let s = var.max(0.0).sqrt();
        let lg = (r / (1.0 - r)).ln();
        let sl = s / (r * (1.0 - r));
        let il = |x: f64| 1.0 / (1.0 + (-x).exp());
        tbl.push(rule(lw));
        tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", "rho", cell(r), cell(s), "", cell(il(lg - crit * sl)), cell(il(lg + crit * sl))));
    }
    tbl.push(bar);
    out.extend(tbl);
    let lab = match cmd.as_str() {
        "xtologit" => "LR test vs. ologit model",
        "xtoprobit" => "LR test vs. oprobit model",
        "xttobit" | "xtintreg" => "LR test of sigma_u=0",
        "xtcloglog" => "LR test of rho=0",
        _ => "",
    };
    let dl = format!("LR test vs. {} model", capital(&mac("distribution")));
    let lab = if lab.is_empty() { dl.as_str() } else { lab };
    if let Some(l) = chibar_line(e, lab) {
        out.push(l);
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// me commands
// ---------------------------------------------------------------------------

pub struct MeRe {
    pub verb: &'static str,
}

impl CommandHandler for MeRe {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let after_verb = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        let mut parts = after_verb.split("||");
        let fixed = parts.next().unwrap_or("").trim().to_string();
        let mut global_opts: Vec<String> = Vec::new();
        let mut levels: Vec<String> = Vec::new();
        for part in parts {
            let (body, opts) = match part.split_once(',') {
                Some((a, b)) => (a.trim(), b.trim()),
                None => (part.trim(), ""),
            };
            let (gv, rest) = body.split_once(':').ok_or_else(|| rc_err(198, "invalid random-effects specification"))?;
            if !rest.trim().is_empty() {
                return Err(rc_err(198, "random slopes are not supported by this command; only random intercepts"));
            }
            let var = crate::varlist::resolve_name(ds, gv.trim())?;
            levels.push(var);
            if !opts.is_empty() {
                let lopts = ferrostat_parser::parser::parse_line(&format!("x , {opts}")).map_err(|e| rc_err(198, e.message))?;
                for (n, a) in &lopts.opts {
                    let nl = n.to_lowercase();
                    if nl.starts_with("cov") {
                        continue;
                    }
                    global_opts.push(match a {
                        Some(a) => format!("{n}({a})"),
                        None => n.clone(),
                    });
                }
            }
        }
        if levels.is_empty() && self.verb != "mestreg" {
            if fixed.is_empty() || !fixed.contains(' ') && fixed.starts_with(',') {
                if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                    return Err(rc_err(301, "last estimates not found"));
                }
                return Ok(render_me(&st.e, cmd.has_opt("or", 2) || cmd.has_opt("irr", 3) || cmd.has_opt("eform", 2), util::level(cmd, st)?));
            }
        }
        if levels.is_empty() && st.e.macros.get("cmd").map(String::as_str) == Some(self.verb) && cmd.main.trim().is_empty() {
            return Ok(render_me(&st.e, cmd.has_opt("eform", 2), util::level(cmd, st)?));
        }
        let fixed_line = if global_opts.is_empty() {
            format!("{} {fixed}", self.verb)
        } else if fixed.contains(',') {
            format!("{} {fixed} {}", self.verb, global_opts.join(" "))
        } else {
            format!("{} {fixed}, {}", self.verb, global_opts.join(" "))
        };
        let fcmd = ferrostat_parser::parser::parse_line(&fixed_line).map_err(|e| rc_err(198, e.message))?;
        let kind = dens_kind(self.verb);
        let p = prepare(ds, st, &fcmd, kind, levels.clone())?;
        let dens = match self.verb {
            "meologit" => Dens::Ordinal(false, p.resp.cats.len() - 1),
            "meoprobit" => Dens::Ordinal(true, p.resp.cats.len() - 1),
            "metobit" | "meintreg" => Dens::Interval,
            "mecloglog" => Dens::Cloglog,
            "menbreg" => Dens::Nb2,
            _ => {
                let (d, aft) = surv_dist(&fcmd)?;
                Dens::Surv(d, aft)
            }
        };
        let nq = util::opt_num(&fcmd, "intpoints", 3)?.unwrap_or(7.0) as usize;
        if !(1..=128).contains(&nq) {
            return Err(rc_err(198, "intpoints() must be between 1 and 128"));
        }
        let n = p.d.n();
        let raw_codes: Vec<Vec<String>> = levels.iter().map(|v| group_codes(ds, v, &p.d.rows)).collect::<Result<_, _>>()?;
        let mut codes = raw_codes.clone();
        for l in 1..codes.len() {
            for i in 0..n {
                codes[l][i] = format!("{}\u{1}{}", codes[l - 1][i], raw_codes[l][i]);
            }
        }
        let vce = est::parse_vce(&fcmd, Vce::Oim)?;
        let all: Vec<usize> = (0..n).collect();
        let units = if codes.is_empty() { vec![] } else { build_units(&codes, 0, &all) };
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Robust if !units.is_empty() => Some((0..units.len()).map(|g| g as f64).collect()),
            Vce::Cluster(c) if !units.is_empty() => {
                let name = crate::varlist::resolve_name(ds, c)?;
                let col = util::col(ds, &name)?;
                Some(units.iter().map(|u| col[p.d.rows[u.obs[0]]]).collect())
            }
            _ => None,
        };
        let pooled_title = "Fitting fixed-effects model:";
        let fitr = fit(dens, &p.d, &p.act, &p.off, &p.resp, &codes, nq, cl.as_deref(), "Log likelihood", pooled_title)?;
        // group counts per level
        let mut counts: Vec<Vec<usize>> = vec![Vec::new(); levels.len()];
        fn walk(us: &[Unit], depth: usize, out: &mut Vec<Vec<usize>>) {
            for u in us {
                out[depth].push(u.obs.len());
                walk(&u.children, depth + 1, out);
            }
        }
        walk(&units, 0, &mut counts);
        let kx = p.act.len();
        let m = dens.naux();
        let k = p.d.k();
        let dep0 = p.dep.split_whitespace().next().unwrap_or("").to_string();
        let eqn = if kind == "surv" { "_t".to_string() } else { dep0.clone() };
        let mut amap = vec![usize::MAX; k];
        for (a, &j) in p.act.iter().enumerate() {
            amap[j] = a;
        }
        let mut post: Vec<(String, usize, Tr)> = Vec::new();
        for (j, nm) in p.d.names.iter().enumerate() {
            post.push((format!("{eqn}:{nm}"), amap[j], Tr::Id));
        }
        match dens {
            Dens::Ordinal(..) => {
                for c in 0..m {
                    post.push((format!("/:cut{}", c + 1), kx + c, Tr::Id));
                }
            }
            Dens::Nb2 => post.push(("/:lnalpha".into(), kx, Tr::Id)),
            Dens::Surv(d, _) => {
                for (a, nm) in d.anc().iter().enumerate() {
                    post.push((format!("/:{nm}"), kx + a, Tr::Id));
                }
            }
            _ => {}
        }
        let mut path = String::new();
        for (l, lv) in levels.iter().enumerate() {
            path = if l == 0 { lv.clone() } else { format!("{path}>{lv}") };
            post.push((format!("/:var(_cons[{path}])"), kx + m + l, Tr::Exp2));
        }
        if dens == Dens::Interval {
            post.push((format!("/:var(e.{dep0})"), kx, Tr::Exp2));
        }
        let pint = fitr.b.len();
        let mut bint = fitr.b.clone();
        bint.push(0.0);
        let mut vint = Matrix::zeros(pint + 1, pint + 1);
        for i in 0..pint {
            for j in 0..pint {
                vint[(i, j)] = fitr.v[(i, j)];
            }
        }
        let post: Vec<(String, usize, Tr)> = post.into_iter().map(|(n, i, t)| (n, if i == usize::MAX { pint } else { i }, t)).collect();
        let (names, bo, vo) = post_params(&post, &bint, &vint);
        let mut d = p.d.clone();
        d.names = names.clone();
        d.omitted.extend(vec![false; names.len() - k]);
        d.is_base.extend(vec![false; names.len() - k]);
        let mut dk = p.d.clone();
        dk.names = p.d.names.clone();
        let mut vk = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                vk[(i, j)] = vo[(i, j)];
            }
        }
        let (chi2, df_m) = est::wald_all(&dk, &bo[..k], &vk);
        let nre = levels.len();
        let lr = if cl.is_none() && nre > 0 { (2.0 * (fitr.ll - fitr.ll_c)).max(0.0) } else { SYSMISS };
        let title = match self.verb {
            "meologit" => "Mixed-effects ologit regression",
            "meoprobit" => "Mixed-effects oprobit regression",
            "metobit" => "Mixed-effects tobit regression",
            "meintreg" => "Mixed-effects interval regression",
            "mecloglog" => "Mixed-effects cloglog regression",
            "menbreg" => "Mixed-effects negative binomial regression",
            _ => "Mixed-effects parametric survival model",
        };
        let mut head = Vec::new();
        head.extend(p.d.notes.iter().cloned());
        head.extend(fitr.log.iter().cloned());
        let head = head.join("\n");
        let verb = self.verb;
        let level = util::level(&fcmd, st)?;
        let cats = p.resp.cats.clone();
        let cens = p.notes.first().cloned().unwrap_or_default();
        let offset = p.offset.clone();
        let robust = cl.is_some();
        let clustvar = match &vce {
            Vce::Cluster(c) => c.clone(),
            _ => levels.first().cloned().unwrap_or_default(),
        };
        est::post(st, verb, &d, &bo, &vo, &|e: &mut Results| {
            e.set("ll", fitr.ll);
            if nre > 0 {
                e.set("ll_c", fitr.ll_c);
            }
            if !is_missing(lr) {
                e.set("chi2_c", lr);
                e.set("df_c", nre as f64);
                e.set("p_c", if nre == 1 { 0.5 * dist::chi2tail(1.0, lr) } else { dist::chi2tail(nre as f64, lr) });
            }
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("k", pint as f64);
            e.set("k_f", k as f64);
            e.set("k_r", nre as f64);
            e.set("rank", pint as f64);
            e.set("n_quad", nq as f64);
            e.set("ic", fitr.ic as f64);
            e.set("converged", if fitr.converged { 1.0 } else { 0.0 });
            let l = counts.len().max(1);
            let mut ng = Matrix::zeros(1, l);
            let mut mn = Matrix::zeros(1, l);
            let mut av = Matrix::zeros(1, l);
            let mut mx = Matrix::zeros(1, l);
            for (i, c) in counts.iter().enumerate() {
                ng[(0, i)] = c.len() as f64;
                mn[(0, i)] = *c.iter().min().unwrap_or(&0) as f64;
                av[(0, i)] = c.iter().sum::<usize>() as f64 / c.len().max(1) as f64;
                mx[(0, i)] = *c.iter().max().unwrap_or(&0) as f64;
            }
            e.set_mat("N_g", ng);
            e.set_mat("g_min", mn);
            e.set_mat("g_avg", av);
            e.set_mat("g_max", mx);
            e.set_str("ivars", levels.join(" "));
            e.set_str("title", title);
            if let Dens::Ordinal(probit, ncut) = dens {
                e.set("k_cat", (ncut + 1) as f64);
                e.set_mat("cat", Matrix::row_vector(&cats));
                e.set_str("link", if probit { "probit" } else { "logit" });
                e.set_str("family", "ordinal");
            }
            if !cens.is_empty() {
                let c: Vec<f64> = cens.split('|').map(|x| x.parse().unwrap_or(0.0)).collect();
                e.set("N_unc", c[0]);
                e.set("N_lc", c[1]);
                e.set("N_rc", c[2]);
                if c.len() == 4 {
                    e.set("N_int", c[3]);
                }
            }
            if let Some(g) = fitr.n_clust {
                e.set("N_clust", g as f64);
            }
            if let Dens::Surv(dd, aft) = dens {
                e.set_str("distribution", dd.label().to_lowercase());
                e.set_str("frm2", if aft { "time" } else { "hazard" });
            }
            if let Some(o) = &offset {
                e.set_str("offset", o.clone());
            }
            e.set_str("intmethod", if nq == 1 { "laplace" } else { "mvaghermite" });
            e.set_str("chi2type", "Wald");
            if robust {
                e.set_str("vce", "cluster");
                e.set_str("vcetype", "Robust");
                e.set_str("clustvar", clustvar.clone());
            } else {
                e.set_str("vce", "oim");
            }
            e.set_str("predict", "meglm_p");
            e.set_str("_head", head.clone());
        });
        let eform = fcmd.has_opt("or", 2) || fcmd.has_opt("irr", 3) || fcmd.has_opt("eform", 2);
        let mut text = format!("{head}\n{}", render_me(&st.e, eform, level));
        if !fitr.converged {
            text.push_str("\nconvergence not achieved");
        }
        Ok(text)
    }
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn render_me(e: &Results, eform: bool, level: f64) -> String {
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let mac = |s: &str| e.macros.get(s).cloned().unwrap_or_default();
    let ivars: Vec<String> = mac("ivars").split_whitespace().map(String::from).collect();
    let mat = |s: &str| e.matrices.get(s).map(|m| m.data.clone()).unwrap_or_default();
    let (ng, mn, av, mx) = (mat("N_g"), mat("g_min"), mat("g_avg"), mat("g_max"));
    let row = |l: &str, k: &str, v: String| format!("{:<48}{:<17} = {:>10}", l, k, v);
    let mut out = vec![String::new()];
    out.push(row(&mac("title"), "Number of obs", util::commas(get("N") as usize)));
    if let Some(nu) = e.get("N_unc") {
        out.push(row("", "Uncensored", util::commas(nu as usize)));
        out.push(row(&String::new(), "Left-censored", util::commas(get("N_lc") as usize)));
        out.push(row("", "Right-censored", util::commas(get("N_rc") as usize)));
        if let Some(ni) = e.get("N_int") {
            out.push(row("", "Interval-cens.", util::commas(ni as usize)));
        }
    }
    if mac("cmd") == "mestreg" {
        out.push(format!("{:<48}", format!("Distribution: {}", mac("distribution"))).trim_end().to_string());
    }
    if ivars.len() == 1 {
        out.push(row(&format!("Group variable: {}", ivars[0]), "Number of groups", util::commas(ng.first().copied().unwrap_or(0.0) as usize)));
        out.push(String::new());
        out.push(format!("{:<48}Obs per group:", ""));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "min", util::commas(mn[0] as usize)));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "avg", format!("{:.1}", av[0])));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "max", util::commas(mx[0] as usize)));
    } else if ivars.len() > 1 {
        out.push(String::new());
        out.push("-".repeat(61));
        out.push("                |     No. of       Observations per group".into());
        out.push(" Group variable |     groups    Minimum    Average    Maximum".into());
        out.push(format!("{}+{}", "-".repeat(16), "-".repeat(44)));
        for (l, iv) in ivars.iter().enumerate() {
            out.push(format!("{:>15} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(iv, 15), util::commas(ng[l] as usize), util::commas(mn[l] as usize), format!("{:.1}", av[l]), util::commas(mx[l] as usize)));
        }
        out.push("-".repeat(61));
    }
    out.push(String::new());
    if !ivars.is_empty() {
        out.push(row(&format!("Integration method: {}", mac("intmethod")), "Integration pts.", fmt_g(get("n_quad"), 9)));
        out.push(String::new());
    }
    out.push(row("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
    out.push(row(&format!("Log likelihood = {}", fmt_g(get("ll"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    if !mac("vcetype").is_empty() {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(get("N_clust") as usize), mac("clustvar"))));
    }
    out.push(String::new());
    let cmd = mac("cmd");
    let ef = if eform {
        Some(match cmd.as_str() {
            "meologit" => "Odds ratio",
            "menbreg" => "IRR",
            "mestreg" if mac("frm2") == "hazard" => "Haz. ratio",
            "mestreg" => "Time ratio",
            _ => "exp(b)",
        })
    } else {
        None
    };
    let (mut tbl, lw, bar) = fixed_table(e, ef, level);
    let b = e.matrices.get("b").cloned().unwrap();
    let v = e.matrices.get("V").cloned().unwrap();
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let kf = get("k_f") as usize;
    let mut j = kf;
    let mut first_anc = true;
    // ancillary parameters (cutpoints, lnalpha, survival shape)
    while j < b.cols && !b.colnames[j].starts_with("/:var(") {
        if first_anc {
            tbl.push(rule(lw));
            first_anc = false;
        }
        let nm = b.colnames[j].trim_start_matches("/:");
        let se = v[(j, j)].max(0.0).sqrt();
        if nm.starts_with("cut") {
            tbl.push(zrow(lw, &format!("/{nm}"), b.data[j], se, crit));
        } else {
            tbl.push(vrow(lw, &format!("/{nm}"), b.data[j], se, crit, false));
        }
        j += 1;
    }
    let mut last = String::new();
    while j < b.cols {
        let inner = b.colnames[j].trim_start_matches("/:var(").trim_end_matches(')');
        let se = v[(j, j)].max(0.0).sqrt();
        if let Some(ev) = inner.strip_prefix("e.") {
            tbl.push(rule(lw));
            tbl.push(vrow(lw, &format!("var(e.{})", crate::summarize::abbrev(ev, lw.saturating_sub(7))), b.data[j], se, crit, true));
            j += 1;
            continue;
        }
        let (zn, lvar) = inner.split_once('[').map(|(a, c)| (a.to_string(), c.trim_end_matches(']').to_string())).unwrap_or((inner.to_string(), String::new()));
        if lvar != last {
            tbl.push(rule(lw));
            tbl.push(format!("{:<lw$} |", crate::summarize::abbrev(&lvar, lw)));
            last = lvar;
        }
        tbl.push(vrow(lw, &format!("var({zn})"), b.data[j], se, crit, true));
        j += 1;
    }
    tbl.push(bar);
    out.extend(tbl);
    let model = match cmd.as_str() {
        "meologit" => "ologit",
        "meoprobit" => "oprobit",
        "metobit" => "tobit",
        "meintreg" => "interval",
        "mecloglog" => "cloglog",
        "menbreg" => "nbinomial",
        _ => "",
    };
    let dl = capital(&mac("distribution"));
    let model = if model.is_empty() { dl.as_str() } else { model };
    let lab = format!("LR test vs. {model} model");
    if e.get("df_c").unwrap_or(1.0) > 1.0 {
        if let Some(l) = chibar_line(e, &lab) {
            out.push(l);
            out.push(String::new());
            out.push("Note: LR test is conservative and provided only for reference.".into());
        }
    } else if let Some(l) = chibar_line(e, &lab) {
        out.push(l);
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// stintreg
// ---------------------------------------------------------------------------

/// `stintreg [indepvars], interval(t_l t_u) distribution(...)`: parametric
/// survival models for interval-censored data.
pub struct StIntreg;

impl CommandHandler for StIntreg {
    fn name(&self) -> &'static str {
        "stintreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let iv = cmd.opt("interval", 3).ok_or_else(|| rc_err(198, "option interval() required"))?;
        let ivs: Vec<String> = iv.split_whitespace().map(|v| crate::varlist::resolve_name(ds, v)).collect::<Result<_, _>>()?;
        if ivs.len() != 2 {
            return Err(rc_err(198, "interval() requires two variables"));
        }
        let (d, aft) = surv_dist(cmd)?;
        let dens = Dens::SurvInt(d, aft);
        let mut mask = util::touse(ds, st, cmd)?;
        let tl = util::col(ds, &ivs[0])?;
        let tu = util::col(ds, &ivs[1])?;
        for i in 0..ds.n_obs {
            if is_missing(tl[i]) && is_missing(tu[i]) {
                mask[i] = false;
            }
        }
        let mut st2 = st.clone();
        st2.by_mask = Some(mask.clone());
        let mut c2 = cmd.clone();
        c2.if_expr = None;
        c2.in_range = None;
        let dd = est::build(ds, &st2, &c2, DesignSpec { depvar: None, indep: cmd.main.trim(), noconstant: cmd.has_opt("noconstant", 3), weights: "", extra: vec![] })?;
        let n = dd.n();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let mut resp = RespData { y: vec![0.0; n], y2: vec![0.0; n], t0: vec![0.0; n], cats: vec![] };
        let (mut nu, mut nl, mut nr, mut ni) = (0, 0, 0, 0);
        for (t, &i) in dd.rows.iter().enumerate() {
            let a = if is_missing(tl[i]) { 0.0 } else { tl[i] };
            let b = if is_missing(tu[i]) { f64::INFINITY } else { tu[i] };
            if a > b || b <= 0.0 {
                return Err(rc_err(498, "invalid interval in some observations"));
            }
            resp.y[t] = a;
            resp.y2[t] = b;
            if a == b {
                nu += 1;
            } else if a <= 0.0 {
                nl += 1;
            } else if b == f64::INFINITY {
                nr += 1;
            } else {
                ni += 1;
            }
        }
        let act = dd.active();
        let off = vec![0.0; n];
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Robust => Some((0..n).map(|t| t as f64).collect()),
            Vce::Cluster(c) => {
                let col = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
                Some(dd.rows.iter().map(|&i| col[i]).collect())
            }
            _ => None,
        };
        // constant-only model for the LR test
        let mut d0 = dd.clone();
        for (j, nm) in d0.names.iter().enumerate() {
            if nm != "_cons" {
                d0.omitted[j] = true;
            }
        }
        let a0 = d0.active();
        let f0 = fit(dens, &d0, &a0, &off, &resp, &[], 1, None, "Log likelihood", "")?;
        let fitr = fit(dens, &dd, &act, &off, &resp, &[], 1, cl.as_deref(), "Log likelihood", "")?;
        let k = dd.k();
        let kx = act.len();
        let mut amap = vec![usize::MAX; k];
        for (a, &j) in act.iter().enumerate() {
            amap[j] = a;
        }
        let pint = fitr.b.len();
        let mut post: Vec<(String, usize, Tr)> = dd.names.iter().enumerate().map(|(j, nm)| (format!("_t:{nm}"), if amap[j] == usize::MAX { pint } else { amap[j] }, Tr::Id)).collect();
        for (a, nm) in d.anc().iter().enumerate() {
            post.push((format!("/:{nm}"), kx + a, Tr::Id));
        }
        let mut bint = fitr.b.clone();
        bint.push(0.0);
        let mut vint = Matrix::zeros(pint + 1, pint + 1);
        for i in 0..pint {
            for j in 0..pint {
                vint[(i, j)] = fitr.v[(i, j)];
            }
        }
        let (names, bo, vo) = post_params(&post, &bint, &vint);
        let mut dp = dd.clone();
        dp.names = names.clone();
        dp.omitted.extend(vec![false; names.len() - k]);
        dp.is_base.extend(vec![false; names.len() - k]);
        dp.depvar = "_t".into();
        let lr = 2.0 * (fitr.ll - f0.ll);
        let df_m = (kx - a0.len()) as f64;
        let robust = cl.is_some();
        let (chi2, chi2type) = if robust {
            let mut dk = dd.clone();
            dk.names = dd.names.clone();
            let mut vk = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    vk[(i, j)] = vo[(i, j)];
                }
            }
            (est::wald_all(&dk, &bo[..k], &vk).0, "Wald")
        } else {
            (lr, "LR")
        };
        let level = util::level(cmd, st)?;
        let head = fitr.log.join("\n");
        let dist_name = d.label();
        let metric = if aft { "AFT" } else { "PH" };
        let lw = 12;
        est::post(st, "stintreg", &dp, &bo, &vo, &|e: &mut Results| {
            e.set("ll", fitr.ll);
            e.set("ll_0", f0.ll);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("k_f", k as f64);
            e.set("N_unc", nu as f64);
            e.set("N_lc", nl as f64);
            e.set("N_rc", nr as f64);
            e.set("N_int", ni as f64);
            e.set("converged", if fitr.converged { 1.0 } else { 0.0 });
            e.set_str("distribution", dist_name.to_lowercase());
            e.set_str("frm2", if aft { "time" } else { "hazard" });
            e.set_str("chi2type", chi2type);
            e.set_str("title", format!("{dist_name} {metric} regression"));
            e.set_str("vce", if robust { "robust" } else { "oim" });
            if robust {
                e.set_str("vcetype", "Robust");
            }
            e.set_str("_head", head.clone());
        });
        let row = |l: &str, k: &str, v: String| format!("{:<48}{:<17} = {:>10}", l, k, v);
        let mut out = vec![head.clone(), String::new()];
        out.push(row(&format!("{dist_name} {metric} regression"), "Number of obs", util::commas(n)));
        out.push(row("", "Uncensored", util::commas(nu)));
        out.push(row("", "Left-censored", util::commas(nl)));
        out.push(row("", "Right-censored", util::commas(nr)));
        out.push(row("", "Interval-cens.", util::commas(ni)));
        out.push(String::new());
        out.push(row("", &format!("{chi2type} chi2({})", fmt_g(df_m, 9)), format!("{:.2}", chi2)));
        out.push(row(&format!("Log likelihood = {}", fmt_g(fitr.ll, 9)), "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))));
        out.push(String::new());
        let eform = if aft {
            if cmd.has_opt("tratio", 2) { Some("Time ratio") } else { None }
        } else if cmd.has_opt("nohr", 4) {
            None
        } else {
            Some("Haz. ratio")
        };
        let (mut tbl, lw2, bar) = fixed_table(&st.e, eform, level);
        let lw = lw.max(lw2);
        let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        if !d.anc().is_empty() {
            tbl.push(rule(lw));
            for (a, nm) in d.anc().iter().enumerate() {
                let j = k + a;
                tbl.push(zrow(lw, &format!("/{nm}"), bo[j], vo[(j, j)].max(0.0).sqrt(), crit));
            }
            tbl.push(rule(lw));
            for (a, nm) in d.anc().iter().enumerate() {
                let j = k + a;
                let (t, se) = (bo[j], vo[(j, j)].max(0.0).sqrt());
                let (lab, f): (&str, fn(f64) -> f64) = match *nm {
                    "ln_p" => ("p", |x| x.exp()),
                    "ln_sig" => ("sigma", |x| x.exp()),
                    "ln_gam" => ("gamma", |x| x.exp()),
                    _ => continue,
                };
                tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", lab, cell(f(t)), cell(f(t) * se), "", cell(f(t - crit * se)), cell(f(t + crit * se))));
                if *nm == "ln_p" {
                    tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", "1/p", cell((-t).exp()), cell((-t).exp() * se), "", cell((-t - crit * se).exp()), cell((-t + crit * se).exp())));
                }
            }
        }
        tbl.push(bar);
        out.extend(tbl);
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn densities_gradients() {
        let cases: Vec<(Dens, f64, f64, f64, Vec<f64>)> = vec![
            (Dens::Ordinal(false, 3), 1.0, 0.0, 0.0, vec![-0.5, 0.4, 1.2]),
            (Dens::Ordinal(true, 3), 3.0, 0.0, 0.0, vec![-0.5, 0.4, 1.2]),
            (Dens::Nb2, 3.0, 0.0, 0.0, vec![-0.3]),
            (Dens::Interval, 1.0, 1.0, 0.0, vec![0.2]),
            (Dens::Interval, f64::NEG_INFINITY, 0.5, 0.0, vec![0.2]),
            (Dens::Interval, 0.5, f64::INFINITY, 0.0, vec![0.2]),
            (Dens::Interval, 0.1, 0.9, 0.0, vec![0.2]),
            (Dens::Cloglog, 1.0, 0.0, 0.0, vec![]),
        ];
        for (dens, y, y2, t0, aux) in cases {
            let eta = 0.3;
            let (_, de, da) = lf(dens, y, y2, t0, eta, &aux);
            let h = 1e-6;
            let num = (lf(dens, y, y2, t0, eta + h, &aux).0 - lf(dens, y, y2, t0, eta - h, &aux).0) / (2.0 * h);
            assert!((num - de).abs() < 1e-6, "{dens:?} eta {num} {de}");
            for a in 0..aux.len() {
                let mut ap = aux.clone();
                ap[a] += h;
                let mut am = aux.clone();
                am[a] -= h;
                let num = (lf(dens, y, y2, t0, eta, &ap).0 - lf(dens, y, y2, t0, eta, &am).0) / (2.0 * h);
                assert!((num - da[a]).abs() < 1e-6, "{dens:?} aux {a} {num} {}", da[a]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// estat after the me commands
// ---------------------------------------------------------------------------

/// `estat icc | sd | group | recovariance` after meologit, menbreg, ...
pub fn estat(st: &mut Store, sub: &str) -> Option<Result<String, String>> {
    let e = st.e.clone();
    let b = e.matrices.get("b")?.clone();
    let v = e.matrices.get("V")?.clone();
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    let vars: Vec<(usize, String)> = b.colnames.iter().enumerate().filter(|(_, n)| n.starts_with("/:var(_cons[")).map(|(j, n)| (j, n.trim_start_matches("/:var(_cons[").trim_end_matches("])").to_string())).collect();
    let crit = dist::invnormal(0.975);
    match sub {
        "group" => {
            let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
            let mat = |s: &str| e.matrices.get(s).map(|m| m.data.clone()).unwrap_or_default();
            let (ng, mn, av, mx) = (mat("N_g"), mat("g_min"), mat("g_avg"), mat("g_max"));
            let mut out = vec![String::new(), "-".repeat(61), "                |     No. of       Observations per group".into(), " Group variable |     groups    Minimum    Average    Maximum".into(), format!("{}+{}", "-".repeat(16), "-".repeat(44))];
            for (l, iv) in ivars.iter().enumerate() {
                out.push(format!("{:>15} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(iv, 15), util::commas(ng[l] as usize), util::commas(mn[l] as usize), format!("{:.1}", av[l]), util::commas(mx[l] as usize)));
            }
            out.push("-".repeat(61));
            Some(Ok(out.join("\n")))
        }
        "sd" => {
            let mut out = vec![String::new(), "-".repeat(78), format!("{:>12} | {:>11}  Std. err.{:>21}[95% conf. interval]", "", "Estimate", ""), format!("{}+{}", "-".repeat(13), "-".repeat(64))];
            for (j, lv) in &vars {
                let var = b.data[*j];
                let se = v[(*j, *j)].max(0.0).sqrt();
                let sd = var.sqrt();
                let ssd = se / (2.0 * sd);
                let sl = ssd / sd;
                out.push(format!("{:<12} |", crate::summarize::abbrev(lv, 12)));
                out.push(vrow(12, "sd(_cons)", sd, ssd, crit, true).replace(&format!("{:21}", ""), &format!("{:21}", "")));
                let _ = sl;
            }
            out.push("-".repeat(78));
            Some(Ok(out.join("\n")))
        }
        "recovariance" => {
            let mut out = Vec::new();
            for (j, lv) in &vars {
                out.push(String::new());
                out.push(format!("Random-effects covariance matrix for level {lv}"));
                out.push(String::new());
                out.push(format!("{:>12} | {:>10}", "", "_cons"));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11)));
                out.push(format!("{:>12} | {:>10}", "_cons", fmt_g(b.data[*j], 8)));
            }
            Some(Ok(out.join("\n")))
        }
        "icc" => {
            let resid = match cmd.as_str() {
                "meologit" => std::f64::consts::PI.powi(2) / 3.0,
                "meoprobit" => 1.0,
                "mecloglog" => std::f64::consts::PI.powi(2) / 6.0,
                "metobit" | "meintreg" => {
                    let j = b.colnames.iter().position(|n| n.starts_with("/:var(e."));
                    match j {
                        Some(j) => b.data[j],
                        None => return Some(Err(rc_err(321, "estat icc not available"))),
                    }
                }
                _ => return Some(Err(rc_err(321, format!("estat icc not available after {cmd}")))),
            };
            let resid_j = b.colnames.iter().position(|n| n.starts_with("/:var(e."));
            let tot: f64 = vars.iter().map(|(j, _)| b.data[*j]).sum::<f64>() + resid;
            let mut out = vec![String::new(), "Intraclass correlation".to_string(), String::new(), "-".repeat(64), format!("{:>20} | {:>9}  Std. err.     [95% conf. interval]", "Level", "ICC"), format!("{}+{}", "-".repeat(21), "-".repeat(42))];
            let r = st.begin_r();
            for l in 0..vars.len() {
                // icc of levels 0..=l
                let num: f64 = vars[..=l].iter().map(|(j, _)| b.data[*j]).sum();
                let icc = num / tot;
                // gradient wrt each variance parameter
                let mut gr: Vec<(usize, f64)> = Vec::new();
                for (a, (j, _)) in vars.iter().enumerate() {
                    let d = if a <= l { (tot - num) / (tot * tot) } else { -num / (tot * tot) };
                    gr.push((*j, d));
                }
                if let Some(rj) = resid_j {
                    gr.push((rj, -num / (tot * tot)));
                }
                let var: f64 = gr.iter().map(|(i, a)| gr.iter().map(|(j, c)| a * v[(*i, *j)] * c).sum::<f64>()).sum();
                let se = var.max(0.0).sqrt();
                let lg = (icc / (1.0 - icc)).ln();
                let sl = se / (icc * (1.0 - icc));
                let il = |x: f64| 1.0 / (1.0 + (-x).exp());
                let label: Vec<String> = vars[..=l].iter().map(|x| x.1.rsplit('>').next().unwrap_or("").to_string()).rev().collect();
                out.push(format!("{:>20} | {:>9}  {:>9}    {:>9}   {:>9}", crate::summarize::abbrev(&label.join("|"), 20), fmt_g(icc, 8), fmt_g(se, 8), fmt_g(il(lg - crit * sl), 8), fmt_g(il(lg + crit * sl), 8)));
                r.set("icc", icc);
                r.set("se", se);
                r.set(&format!("icc{}", l + 2), icc);
                r.set(&format!("se{}", l + 2), se);
            }
            out.push("-".repeat(64));
            Some(Ok(out.join("\n")))
        }
        _ => None,
    }
}
