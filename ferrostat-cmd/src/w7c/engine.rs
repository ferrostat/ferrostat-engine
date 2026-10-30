//! Generalized-response latent-variable engine shared by `gsem`, `irt` and
//! `fmm`.
//!
//! Outcomes have a family and link; their linear predictors combine
//! observed covariates, continuous latent variables (normal, at the
//! observation level or at the level of a grouping variable) and, with
//! latent classes, class-specific parameters. Continuous latents are
//! integrated out by mean-variance adaptive Gauss-Hermite quadrature;
//! latent classes are summed over with multinomial-logit class
//! probabilities. Estimation: EM warm start for latent classes, then
//! quasi-Newton on the full likelihood with adaptation refreshed between
//! rounds.

use std::cell::RefCell;

use super::opt::{self, Opts, Tech};
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fam {
    Gaussian,
    Bernoulli,
    Binomial,
    Ordinal,
    Multinomial,
    Poisson,
    Nbinomial,
    Gamma,
    Weibull,
    Exponential,
    Lognormal,
    Beta,
    Pointmass,
    /// adjacent-category (partial credit) model
    Pcm,
}

impl Fam {
    pub fn word(&self) -> &'static str {
        match self {
            Fam::Gaussian => "gaussian",
            Fam::Bernoulli => "bernoulli",
            Fam::Binomial => "binomial",
            Fam::Ordinal => "ordinal",
            Fam::Multinomial => "multinomial",
            Fam::Poisson => "poisson",
            Fam::Nbinomial => "nbinomial",
            Fam::Gamma => "gamma",
            Fam::Weibull => "weibull",
            Fam::Exponential => "exponential",
            Fam::Lognormal => "lognormal",
            Fam::Beta => "beta",
            Fam::Pointmass => "pointmass",
            Fam::Pcm => "pcm",
        }
    }
    pub fn from_word(w: &str) -> Option<Fam> {
        Some(match w {
            "gaussian" | "normal" => Fam::Gaussian,
            "bernoulli" => Fam::Bernoulli,
            "binomial" => Fam::Binomial,
            "ordinal" => Fam::Ordinal,
            "multinomial" => Fam::Multinomial,
            "poisson" => Fam::Poisson,
            "nbinomial" => Fam::Nbinomial,
            "gamma" => Fam::Gamma,
            "weibull" => Fam::Weibull,
            "exponential" => Fam::Exponential,
            "lognormal" => Fam::Lognormal,
            "beta" => Fam::Beta,
            "pointmass" => Fam::Pointmass,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Link {
    Identity,
    Logit,
    Probit,
    Cloglog,
    Log,
}

impl Link {
    pub fn word(&self) -> &'static str {
        match self {
            Link::Identity => "identity",
            Link::Logit => "logit",
            Link::Probit => "probit",
            Link::Cloglog => "cloglog",
            Link::Log => "log",
        }
    }
    pub fn from_word(w: &str) -> Option<Link> {
        Some(match w {
            "identity" => Link::Identity,
            "logit" => Link::Logit,
            "probit" => Link::Probit,
            "cloglog" => Link::Cloglog,
            "log" => Link::Log,
            _ => return None,
        })
    }
    pub fn inv(&self, e: f64) -> f64 {
        match self {
            Link::Identity => e,
            Link::Logit => dist::invlogit(e),
            Link::Probit => dist::normal(e),
            Link::Cloglog => 1.0 - (-e.exp()).exp(),
            Link::Log => e.exp(),
        }
    }
    /// ln F(e) and ln (1 - F(e)) for binary links
    fn lnp(&self, e: f64) -> (f64, f64) {
        match self {
            Link::Logit => (-softplus(-e), -softplus(e)),
            Link::Probit => (dist::lnnormal(e), dist::lnnormal(-e)),
            Link::Cloglog => {
                let p = 1.0 - (-e.exp()).exp();
                (p.max(1e-300).ln(), -e.exp())
            }
            _ => {
                let p = e.clamp(1e-300, 1.0 - 1e-16);
                (p.ln(), (1.0 - p).ln())
            }
        }
    }
    fn cdf(&self, e: f64) -> f64 {
        match self {
            Link::Probit => dist::normal(e),
            Link::Cloglog => 1.0 - (-e.exp()).exp(),
            _ => dist::invlogit(e),
        }
    }
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else if x < -30.0 { x.exp() } else { x.exp().ln_1p() }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TKind {
    X(usize),
    Lat(usize),
    One,
}

#[derive(Clone, Debug)]
pub struct Term {
    pub kind: TKind,
    /// linear predictor index (multinomial category)
    pub eta: usize,
    pub slot: usize,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub name: String,
    pub ycol: usize,
    pub y2col: Option<usize>,
    pub fam: Fam,
    pub link: Link,
    pub levels: Vec<f64>,
    pub base: usize,
    pub trials: Option<(Option<usize>, f64)>,
    pub fail: Option<usize>,
    pub cens: (Option<f64>, Option<f64>),
    pub trunc: (Option<f64>, Option<f64>),
    pub offset: Option<usize>,
    /// latent class this outcome belongs to (None: all classes)
    pub class: Option<usize>,
    pub terms: Vec<Term>,
    pub neta: usize,
    /// ancillary parameter slots (cuts, variance, ln alpha, ...)
    pub aux: Vec<usize>,
    /// 3PL guessing parameter slot (logit scale)
    pub guess: Option<usize>,
    pub pm: f64,
}

impl Outcome {
    pub fn new(name: &str, ycol: usize, fam: Fam, link: Link) -> Outcome {
        Outcome {
            name: name.to_string(),
            ycol,
            y2col: None,
            fam,
            link,
            levels: vec![],
            base: 0,
            trials: None,
            fail: None,
            cens: (None, None),
            trunc: (None, None),
            offset: None,
            class: None,
            terms: vec![],
            neta: 1,
            aux: vec![],
            guess: None,
            pm: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PKind {
    Coef,
    Cut,
    Aux,
    LatVar,
    LatCov,
    LatMean,
    ClassCoef,
}

#[derive(Clone, Debug)]
pub struct PSlot {
    pub eq: String,
    pub name: String,
    pub class: Option<usize>,
    pub kind: PKind,
    pub fix: Option<f64>,
    pub p: Option<usize>,
    pub start: f64,
}

#[derive(Clone, Debug)]
pub struct Engine {
    pub outcomes: Vec<Outcome>,
    pub slots: Vec<PSlot>,
    pub k: usize,
    pub nlat: usize,
    pub lat_names: Vec<String>,
    pub lat_level: Option<String>,
    /// per latent group: covariance slots (d x d, symmetric) and mean slots
    pub phi: Vec<Vec<Option<usize>>>,
    pub lmean: Vec<Vec<Option<usize>>>,
    pub nclass: usize,
    pub class_name: String,
    /// class-membership terms: (class index >= 1, term)
    pub cterms: Vec<(usize, TKind, usize)>,
    /// data rows (all columns used), units of rows and the latent group of each unit
    pub data: Vec<Vec<f64>>,
    pub units: Vec<Vec<usize>>,
    pub ugroup: Vec<usize>,
    pub cols: Vec<String>,
    pub nq: usize,
    pub laplace: bool,
    /// adaptive quadrature state per unit: mean and Cholesky factor
    pub adapt: RefCell<Vec<(Vec<f64>, Vec<f64>)>>,
    /// expected-complete-data mode: fixed posterior class weights
    pub em_w: RefCell<Option<Vec<Vec<f64>>>>,
}

pub fn gh(n: usize) -> (Vec<f64>, Vec<f64>) {
    // nodes for N(0,1) expectations
    let (x, w) = crate::melogit::gauss_hermite(n);
    let s2 = 2f64.sqrt();
    let sp = std::f64::consts::PI.sqrt();
    (x.iter().map(|v| v * s2).collect(), w.iter().map(|v| v / sp).collect())
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl Engine {
    pub fn add_slot(&mut self, eq: &str, name: &str, class: Option<usize>, kind: PKind, fix: Option<f64>, start: f64) -> usize {
        let p = if fix.is_some() {
            None
        } else {
            self.k += 1;
            Some(self.k - 1)
        };
        self.slots.push(PSlot { eq: eq.into(), name: name.into(), class, kind, fix, p, start });
        self.slots.len() - 1
    }
    /// share the parameter of slot `b` with slot `a`
    pub fn tie(&mut self, a: usize, b: usize) {
        if let (Some(pa), Some(pb)) = (self.slots[a].p, self.slots[b].p) {
            if pa == pb {
                return;
            }
            for s in self.slots.iter_mut() {
                if s.p == Some(pb) {
                    s.p = Some(pa);
                }
            }
            // renumber
            let mut used: Vec<usize> = self.slots.iter().filter_map(|s| s.p).collect();
            used.sort();
            used.dedup();
            for s in self.slots.iter_mut() {
                if let Some(p) = s.p {
                    s.p = Some(used.iter().position(|&u| u == p).unwrap());
                }
            }
            self.k = used.len();
        }
    }
    pub fn start(&self) -> Vec<f64> {
        let mut th = vec![0.0; self.k];
        for s in &self.slots {
            if let Some(p) = s.p {
                th[p] = s.start;
            }
        }
        th
    }
    pub fn values(&self, th: &[f64]) -> Vec<f64> {
        self.slots.iter().map(|s| s.fix.unwrap_or_else(|| s.p.map_or(0.0, |p| th[p]))).collect()
    }
    fn phi_of(&self, sv: &[f64], g: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        let d = self.nlat;
        let mut m = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                if let Some(s) = self.phi[g][i * d + j] {
                    m[i * d + j] = sv[s];
                }
            }
        }
        let mu: Vec<f64> = (0..d).map(|i| self.lmean[g][i].map_or(0.0, |s| sv[s])).collect();
        Some((m, mu))
    }

    /// Linear predictors of outcome o for a data row and latent values z.
    #[inline]
    fn etas(&self, o: &Outcome, row: &[f64], z: &[f64], sv: &[f64], out: &mut [f64]) {
        for e in out.iter_mut().take(o.neta) {
            *e = 0.0;
        }
        for t in &o.terms {
            let v = sv[t.slot];
            out[t.eta] += v * match t.kind {
                TKind::X(c) => row[c],
                TKind::Lat(l) => z[l],
                TKind::One => 1.0,
            };
        }
        if let Some(c) = o.offset {
            for e in out.iter_mut().take(o.neta) {
                *e += row[c];
            }
        }
    }

    /// Log density of outcome o (0 when its value is missing).
    pub fn logf(&self, o: &Outcome, row: &[f64], eta: &[f64], sv: &[f64]) -> f64 {
        let y = row[o.ycol];
        if y.is_nan() && o.y2col.is_none() {
            return 0.0;
        }
        let e = eta[0];
        match o.fam {
            Fam::Gaussian | Fam::Lognormal => {
                let s2 = sv[o.aux[0]];
                if !(s2 > 0.0) {
                    return f64::NAN;
                }
                let s = s2.sqrt();
                let tr = |v: f64| if o.fam == Fam::Lognormal { v.ln() } else { v };
                let jac = |v: f64| if o.fam == Fam::Lognormal { -v.ln() } else { 0.0 };
                let mut l = if let Some(c2) = o.y2col {
                    let (a, b) = (y, row[c2]);
                    match (a.is_nan(), b.is_nan()) {
                        (true, true) => 0.0,
                        (true, false) => dist::lnnormal((tr(b) - e) / s),
                        (false, true) => dist::lnnormal(-(tr(a) - e) / s),
                        _ if a == b => dist::lnnormalden((tr(a) - e) / s) - 0.5 * s2.ln() + jac(a),
                        _ => (dist::normal((tr(b) - e) / s) - dist::normal((tr(a) - e) / s)).max(1e-300).ln(),
                    }
                } else {
                    match o.cens {
                        (Some(lo), _) if y <= lo => dist::lnnormal((lo - e) / s),
                        (_, Some(hi)) if y >= hi => dist::lnnormal(-(hi - e) / s),
                        _ => dist::lnnormalden((tr(y) - e) / s) - 0.5 * s2.ln() + jac(y),
                    }
                };
                if o.trunc.0.is_some() || o.trunc.1.is_some() {
                    let lo = o.trunc.0.map_or(0.0, |a| dist::normal((a - e) / s));
                    let hi = o.trunc.1.map_or(1.0, |b| dist::normal((b - e) / s));
                    l -= (hi - lo).max(1e-300).ln();
                }
                l
            }
            Fam::Bernoulli => {
                if let Some(gs) = o.guess {
                    let c = dist::invlogit(sv[gs]);
                    let p = c + (1.0 - c) * o.link.inv(e);
                    return if y != 0.0 { p.max(1e-300).ln() } else { (1.0 - p).max(1e-300).ln() };
                }
                let (a, b) = o.link.lnp(e);
                if y != 0.0 { a } else { b }
            }
            Fam::Binomial => {
                let n = match o.trials {
                    Some((Some(c), _)) => row[c],
                    Some((None, n)) => n,
                    None => 1.0,
                };
                let (a, b) = o.link.lnp(e);
                dist::lnfactorial(n) - dist::lnfactorial(y) - dist::lnfactorial(n - y) + y * a + (n - y) * b
            }
            Fam::Ordinal => {
                let k = match o.levels.iter().position(|v| *v == y) {
                    Some(k) => k,
                    None => return f64::NEG_INFINITY,
                };
                let nc = o.levels.len();
                let cut = |j: usize| sv[o.aux[j]];
                for j in 1..o.aux.len() {
                    if cut(j) <= cut(j - 1) {
                        return f64::NAN;
                    }
                }
                if k == 0 {
                    o.link.lnp(cut(0) - e).0
                } else if k == nc - 1 {
                    o.link.lnp(cut(nc - 2) - e).1
                } else {
                    (o.link.cdf(cut(k) - e) - o.link.cdf(cut(k - 1) - e)).max(1e-300).ln()
                }
            }
            Fam::Pcm => {
                let k = match o.levels.iter().position(|v| *v == y) {
                    Some(k) => k,
                    None => return f64::NEG_INFINITY,
                };
                let nc = o.levels.len();
                let mut num = vec![0.0; nc];
                let mut acc = 0.0;
                for j in 1..nc {
                    acc += e - sv[o.aux[j - 1]];
                    num[j] = acc;
                }
                num[k] - lse(&num)
            }
            Fam::Multinomial => {
                let k = match o.levels.iter().position(|v| *v == y) {
                    Some(k) => k,
                    None => return f64::NEG_INFINITY,
                };
                // eta index: categories except base in order
                let mut v = Vec::with_capacity(o.levels.len());
                let mut idx = 0;
                let mut mine = 0.0;
                for j in 0..o.levels.len() {
                    let x = if j == o.base {
                        0.0
                    } else {
                        idx += 1;
                        eta[idx - 1]
                    };
                    if j == k {
                        mine = x;
                    }
                    v.push(x);
                }
                mine - lse(&v)
            }
            Fam::Poisson => {
                let mu = e.exp();
                let mut l = y * e - mu - dist::lnfactorial(y);
                if let Some(lo) = o.trunc.0 {
                    // P(Y > lo)
                    if !(mu > 0.0) || !mu.is_finite() {
                        return f64::NAN;
                    }
                    // ln P(Y > lo) by summing the pmf
                    let mut pk = (-mu).exp();
                    let mut cdf = 0.0;
                    for k in 0..=(lo.max(0.0) as usize) {
                        if k > 0 {
                            pk *= mu / k as f64;
                        }
                        cdf += pk;
                    }
                    l -= if cdf < 0.5 { (-cdf).ln_1p() } else { (1.0 - cdf).max(1e-300).ln() };
                    if lo == 0.0 {
                        // exact form for zero truncation
                        l += (1.0 - cdf).max(1e-300).ln() - (-(-mu).exp_m1()).ln();
                    }
                }
                l
            }
            Fam::Nbinomial => {
                let a = sv[o.aux[0]].exp();
                let m = 1.0 / a;
                let mu = e.exp();
                dist::lngamma(y + m) - dist::lngamma(m) - dist::lnfactorial(y) + m * (m / (m + mu)).ln() + y * (mu / (m + mu)).ln()
            }
            Fam::Gamma => {
                // log scale s: shape 1/s, mean exp(e)
                let s = sv[o.aux[0]].exp();
                let nu = 1.0 / s;
                let mu = e.exp();
                if y <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                nu * (nu / mu).ln() - dist::lngamma(nu) + (nu - 1.0) * y.ln() - nu * y / mu
            }
            Fam::Weibull | Fam::Exponential => {
                let p = if o.fam == Fam::Weibull { sv[o.aux[0]].exp() } else { 1.0 };
                let d = o.fail.map_or(1.0, |c| if row[c] != 0.0 { 1.0 } else { 0.0 });
                if y <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                d * (p.ln() + (p - 1.0) * y.ln() + e) - e.exp() * y.powf(p)
            }
            Fam::Beta => {
                let mu = o.link.inv(e).clamp(1e-12, 1.0 - 1e-12);
                let phi = sv[o.aux[0]].exp();
                if !(y > 0.0 && y < 1.0) {
                    return f64::NEG_INFINITY;
                }
                dist::lngamma(phi) - dist::lngamma(mu * phi) - dist::lngamma((1.0 - mu) * phi) + (mu * phi - 1.0) * y.ln() + ((1.0 - mu) * phi - 1.0) * (1.0 - y).ln()
            }
            Fam::Pointmass => {
                if y == o.pm { 0.0 } else { f64::NEG_INFINITY }
            }
        }
    }

    /// Sum of log densities of the rows of unit u in class c at latent z.
    fn unit_logf(&self, u: usize, c: usize, z: &[f64], sv: &[f64], buf: &mut [f64]) -> f64 {
        let mut l = 0.0;
        for &r in &self.units[u] {
            let row = &self.data[r];
            for o in &self.outcomes {
                if let Some(oc) = o.class {
                    if oc != c {
                        continue;
                    }
                }
                self.etas(o, row, z, sv, buf);
                l += self.logf(o, row, buf, sv);
            }
        }
        l
    }

    /// Log class probabilities for a data row.
    pub fn class_lnp(&self, row: &[f64], sv: &[f64]) -> Vec<f64> {
        let k = self.nclass;
        let mut e = vec![0.0; k];
        for &(c, kind, s) in &self.cterms {
            e[c] += sv[s] * match kind {
                TKind::X(col) => row[col],
                _ => 1.0,
            };
        }
        let m = lse(&e);
        e.iter().map(|x| x - m).collect()
    }

    /// Per-unit log likelihood contributions.
    pub fn ll_units(&self, th: &[f64]) -> Option<Vec<f64>> {
        let sv = self.values(th);
        let d = self.nlat;
        let ngr = self.phi.len().max(1);
        let mut phis = Vec::new();
        for g in 0..ngr {
            if d == 0 {
                phis.push((vec![], vec![], vec![], 0.0));
                continue;
            }
            let (phi, mu) = self.phi_of(&sv, g)?;
            let l = super::la::chol(&phi, d)?;
            let inv = super::la::chol_inv(&l, d);
            let ld = super::la::chol_logdet(&l, d);
            phis.push((mu, l, inv, ld));
        }
        let mut buf = vec![0.0; 64];
        let em = self.em_w.borrow();
        let mut out = Vec::with_capacity(self.units.len());
        if d == 0 {
            for u in 0..self.units.len() {
                if self.nclass <= 1 {
                    out.push(self.unit_logf(u, 0, &[], &sv, &mut buf));
                } else {
                    let lp = self.class_lnp(&self.data[self.units[u][0]], &sv);
                    let terms: Vec<f64> = (0..self.nclass).map(|c| lp[c] + self.unit_logf(u, c, &[], &sv, &mut buf)).collect();
                    match em.as_ref() {
                        Some(w) => out.push((0..self.nclass).map(|c| if w[u][c] > 0.0 { w[u][c] * terms[c] } else { 0.0 }).sum()),
                        None => out.push(lse(&terms)),
                    }
                }
            }
            return Some(out);
        }
        // adaptive quadrature
        let (x, w) = gh(self.nq);
        let nq = self.nq;
        let npts = nq.pow(d as u32);
        let adapt = self.adapt.borrow();
        let half_ln2pi = 0.5 * super::sem::LN2PI;
        let mut idx = vec![0usize; d];
        let mut xi = vec![0.0; d];
        let mut z = vec![0.0; d];
        let mut terms = vec![0.0; npts];
        for u in 0..self.units.len() {
            let (mu, _l, inv, ld) = &phis[self.ugroup[u]];
            let (am, ac) = &adapt[u];
            let lnc: f64 = (0..d).map(|i| ac[i * d + i].ln()).sum();
            for q in 0..npts {
                let mut t = q;
                let mut lw = 0.0;
                let mut xx = 0.0;
                for i in 0..d {
                    idx[i] = t % nq;
                    t /= nq;
                    xi[i] = x[idx[i]];
                    lw += w[idx[i]].ln();
                    xx += xi[i] * xi[i];
                }
                for i in 0..d {
                    z[i] = am[i] + (0..=i).map(|j| ac[i * d + j] * xi[j]).sum::<f64>();
                }
                // ln phi_d(z; mu, Phi)
                let mut qf = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        qf += (z[i] - mu[i]) * inv[i * d + j] * (z[j] - mu[j]);
                    }
                }
                let lphi = -(d as f64) * half_ln2pi - 0.5 * ld - 0.5 * qf;
                let base = lw + 0.5 * xx + d as f64 * half_ln2pi + lnc + lphi;
                terms[q] = base + self.unit_logf(u, 0, &z, &sv, &mut buf);
            }
            out.push(lse(&terms));
        }
        Some(out)
    }

    pub fn ll(&self, th: &[f64]) -> Option<f64> {
        let v = self.ll_units(th)?;
        let s: f64 = v.iter().sum();
        if s.is_finite() { Some(s) } else { None }
    }

    /// Reset adaptation to the prior.
    pub fn init_adapt(&self, th: &[f64]) {
        let d = self.nlat;
        let sv = self.values(th);
        let mut a = self.adapt.borrow_mut();
        a.clear();
        for u in 0..self.units.len() {
            if d == 0 {
                a.push((vec![], vec![]));
                continue;
            }
            let (phi, mu) = self.phi_of(&sv, self.ugroup[u]).unwrap_or((vec![1.0; d * d], vec![0.0; d]));
            let l = super::la::chol(&phi, d).unwrap_or_else(|| {
                let mut i = vec![0.0; d * d];
                for k in 0..d {
                    i[k * d + k] = 1.0;
                }
                i
            });
            a.push((mu, l));
        }
    }

    /// Posterior means and Cholesky factors of the latents of each unit
    /// under the current adaptation; updates the adaptation.
    pub fn update_adapt(&self, th: &[f64]) {
        let d = self.nlat;
        if d == 0 {
            return;
        }
        let post = self.posterior(th);
        let mut a = self.adapt.borrow_mut();
        for (u, (m, v)) in post.into_iter().enumerate() {
            if let Some(l) = super::la::chol(&v, d) {
                if m.iter().all(|x| x.is_finite()) {
                    a[u] = (m, l);
                }
            }
        }
    }

    /// Posterior mean and covariance of the latents of each unit.
    pub fn posterior(&self, th: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let d = self.nlat;
        let sv = self.values(th);
        let (x, w) = gh(self.nq);
        let nq = self.nq;
        let npts = nq.pow(d as u32);
        let adapt = self.adapt.borrow();
        let mut buf = vec![0.0; 64];
        let half_ln2pi = 0.5 * super::sem::LN2PI;
        let mut res = Vec::with_capacity(self.units.len());
        for u in 0..self.units.len() {
            let g = self.ugroup[u];
            let Some((phi, mu)) = self.phi_of(&sv, g) else {
                res.push((vec![0.0; d], vec![1.0; d * d]));
                continue;
            };
            let Some(l) = super::la::chol(&phi, d) else {
                res.push((vec![0.0; d], vec![1.0; d * d]));
                continue;
            };
            let inv = super::la::chol_inv(&l, d);
            let ld = super::la::chol_logdet(&l, d);
            let (am, ac) = &adapt[u];
            let lnc: f64 = (0..d).map(|i| ac[i * d + i].ln()).sum();
            let mut zs = Vec::with_capacity(npts);
            let mut lt = Vec::with_capacity(npts);
            for q in 0..npts {
                let mut t = q;
                let mut lw = 0.0;
                let mut xx = 0.0;
                let mut xi = vec![0.0; d];
                for i in 0..d {
                    let id = t % nq;
                    t /= nq;
                    xi[i] = x[id];
                    lw += w[id].ln();
                    xx += xi[i] * xi[i];
                }
                let z: Vec<f64> = (0..d).map(|i| am[i] + (0..=i).map(|j| ac[i * d + j] * xi[j]).sum::<f64>()).collect();
                let mut qf = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        qf += (z[i] - mu[i]) * inv[i * d + j] * (z[j] - mu[j]);
                    }
                }
                let lphi = -(d as f64) * half_ln2pi - 0.5 * ld - 0.5 * qf;
                lt.push(lw + 0.5 * xx + lnc + lphi + self.unit_logf(u, 0, &z, &sv, &mut buf));
                zs.push(z);
            }
            let m = lse(&lt);
            let mut mean = vec![0.0; d];
            let mut cov = vec![0.0; d * d];
            for (q, z) in zs.iter().enumerate() {
                let p = (lt[q] - m).exp();
                for i in 0..d {
                    mean[i] += p * z[i];
                }
            }
            for (q, z) in zs.iter().enumerate() {
                let p = (lt[q] - m).exp();
                for i in 0..d {
                    for j in 0..d {
                        cov[i * d + j] += p * (z[i] - mean[i]) * (z[j] - mean[j]);
                    }
                }
            }
            for i in 0..d {
                cov[i * d + i] = cov[i * d + i].max(1e-8);
            }
            res.push((mean, cov));
        }
        res
    }

    /// Posterior class probabilities per unit (latent-class models).
    pub fn class_post(&self, th: &[f64]) -> Vec<Vec<f64>> {
        let sv = self.values(th);
        let mut buf = vec![0.0; 64];
        (0..self.units.len())
            .map(|u| {
                let lp = self.class_lnp(&self.data[self.units[u][0]], &sv);
                let t: Vec<f64> = (0..self.nclass).map(|c| lp[c] + self.unit_logf(u, c, &[], &sv, &mut buf)).collect();
                let m = lse(&t);
                t.iter().map(|x| (x - m).exp()).collect()
            })
            .collect()
    }

    /// Prior class probabilities per unit.
    pub fn class_prior(&self, th: &[f64]) -> Vec<Vec<f64>> {
        let sv = self.values(th);
        (0..self.units.len()).map(|u| self.class_lnp(&self.data[self.units[u][0]], &sv).iter().map(|v| v.exp()).collect()).collect()
    }

    /// Mean of outcome o for a row, class c and latent z.
    pub fn mean(&self, o: &Outcome, row: &[f64], z: &[f64], th: &[f64]) -> f64 {
        let sv = self.values(th);
        let mut buf = vec![0.0; 64];
        self.etas(o, row, z, &sv, &mut buf);
        let e = buf[0];
        match o.fam {
            Fam::Gaussian => e,
            Fam::Lognormal => (e + 0.5 * sv[o.aux[0]]).exp(),
            Fam::Bernoulli => match o.guess {
                Some(g) => {
                    let c = dist::invlogit(sv[g]);
                    c + (1.0 - c) * o.link.inv(e)
                }
                None => o.link.inv(e),
            },
            Fam::Binomial => {
                let n = match o.trials {
                    Some((Some(c), _)) => row[c],
                    Some((None, n)) => n,
                    None => 1.0,
                };
                n * o.link.inv(e)
            }
            Fam::Poisson => {
                let mu = e.exp();
                match o.trunc.0 {
                    Some(lo) => {
                        // E[Y | Y > lo]
                        let mut s = 0.0;
                        let mut p = 0.0;
                        for k in 0..=(lo as usize) {
                            let pk = dist::poissonp(mu, k as f64);
                            s += k as f64 * pk;
                            p += pk;
                        }
                        (mu - s) / (1.0 - p).max(1e-300)
                    }
                    None => mu,
                }
            }
            Fam::Nbinomial | Fam::Gamma => e.exp(),
            Fam::Weibull => {
                let p = sv[o.aux[0]].exp();
                (-e / p).exp() * dist::lngamma(1.0 + 1.0 / p).exp()
            }
            Fam::Exponential => (-e).exp(),
            Fam::Beta => o.link.inv(e),
            Fam::Pointmass => o.pm,
            Fam::Ordinal | Fam::Multinomial | Fam::Pcm => {
                let probs = self.cat_probs(o, row, z, th);
                o.levels.iter().zip(&probs).map(|(l, p)| l * p).sum()
            }
        }
    }

    /// Category probabilities for categorical outcomes.
    pub fn cat_probs(&self, o: &Outcome, row: &[f64], z: &[f64], th: &[f64]) -> Vec<f64> {
        let sv = self.values(th);
        let mut buf = vec![0.0; 64];
        self.etas(o, row, z, &sv, &mut buf);
        let mut r = row.to_vec();
        o.levels
            .iter()
            .map(|l| {
                r[o.ycol] = *l;
                self.logf(o, &r, &buf, &sv).exp()
            })
            .collect()
    }

    /// Linear prediction of outcome o (first predictor).
    pub fn eta0(&self, o: &Outcome, row: &[f64], z: &[f64], th: &[f64]) -> f64 {
        let sv = self.values(th);
        let mut buf = vec![0.0; 64];
        self.etas(o, row, z, &sv, &mut buf);
        buf[0]
    }

    /// Maximize the likelihood. Returns (theta, ll, log, converged).
    pub fn fit(&self, th0: Vec<f64>, maxiter: usize, em_iter: usize, show: bool) -> Result<(Vec<f64>, f64, Vec<String>, bool), String> {
        let mut th = th0;
        let mut log = Vec::new();
        self.init_adapt(&th);
        if self.nlat > 0 {
            // a few adaptation passes at the starting values
            for _ in 0..3 {
                self.update_adapt(&th);
            }
        }
        if self.nclass > 1 && em_iter > 0 {
            if show {
                log.push("Refining starting values:".into());
                log.push(String::new());
            }
            for it in 0..em_iter {
                let w = self.class_post(&th);
                *self.em_w.borrow_mut() = Some(w);
                let f = |t: &[f64]| self.ll(t);
                let r = opt::maximize(&f, th.clone(), &Opts { maxiter: 15, tech: Tech::Bfgs, label: "ll", tol: 1e-8 });
                *self.em_w.borrow_mut() = None;
                if let Ok(r) = r {
                    th = r.th;
                }
                let l = self.ll(&th).unwrap_or(f64::NAN);
                if show {
                    log.push(format!("Iteration {it}:{}(EM) log likelihood = {}", if it < 10 { "   " } else { "  " }, crate::format::fmt_g(l, 10)));
                }
            }
            if show {
                log.push(String::new());
                log.push("Fitting full model:".into());
                log.push(String::new());
            }
        }
        let mut converged = false;
        let mut prev = f64::NEG_INFINITY;
        let mut ll = f64::NEG_INFINITY;
        let rounds = if self.nlat > 0 { 8 } else { 1 };
        for round in 0..rounds {
            let f = |t: &[f64]| self.ll(t);
            let r = opt::maximize(&f, th.clone(), &Opts { maxiter, tech: Tech::Bfgs, label: "log likelihood", tol: 1e-9 })?;
            th = r.th;
            ll = r.ll;
            converged = r.converged;
            if round == 0 || !show {
                if show {
                    log.extend(r.log);
                }
            } else if show {
                log.push(crate::w7c::opt::iter_line(r.iters, r.ll, "log likelihood", false));
            }
            if self.nlat == 0 || (ll - prev).abs() < 1e-7 * ll.abs().max(1.0) {
                break;
            }
            prev = ll;
            self.update_adapt(&th);
            ll = self.ll(&th).unwrap_or(ll);
        }
        // Newton polish
        let f = |t: &[f64]| self.ll(t);
        let mut fr = opt::Fit { th, ll, log: vec![], iters: 0, converged };
        opt::polish(&f, &mut fr, 5);
        Ok((fr.th, fr.ll, log, converged))
    }

    /// Variance of theta: oim (Hessian), opg or robust.
    pub fn vce(&self, th: &[f64], kind: &str, cluster: Option<&[f64]>) -> Result<Matrix, String> {
        let k = th.len();
        if k == 0 {
            return Ok(Matrix::zeros(0, 0));
        }
        let f = |t: &[f64]| self.ll(t);
        let gf = |t: &[f64]| opt::gradient(&f, t);
        let h = || -> Result<Matrix, String> { opt::neg_hessian_g(&gf, th).or_else(|| opt::neg_hessian(&f, th)).ok_or_else(|| rc_err(430, "Hessian could not be computed")) };
        let scores = || -> Result<Vec<Vec<f64>>, String> {
            let n = self.units.len();
            let mut s = vec![vec![0.0; k]; n];
            for j in 0..k {
                let hh = 1e-6 * th[j].abs().max(1.0);
                let mut a = th.to_vec();
                a[j] += hh;
                let mut b = th.to_vec();
                b[j] -= hh;
                let la_ = self.ll_units(&a).ok_or_else(|| rc_err(430, "scores"))?;
                let lb = self.ll_units(&b).ok_or_else(|| rc_err(430, "scores"))?;
                for u in 0..n {
                    s[u][j] = (la_[u] - lb[u]) / (2.0 * hh);
                }
            }
            Ok(s)
        };
        let opg = |s: &[Vec<f64>]| {
            let mut m = Matrix::zeros(k, k);
            for r in s {
                for a in 0..k {
                    for b in 0..k {
                        m[(a, b)] += r[a] * r[b];
                    }
                }
            }
            m
        };
        Ok(match kind {
            "opg" => opg(&scores()?).sym_inverse(1e-12).0,
            "robust" | "cluster" => {
                let hi = h()?.sym_inverse(1e-12).0;
                let mut s = scores()?;
                if let Some(cl) = cluster {
                    let mut sums: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
                    for (u, r) in s.iter().enumerate() {
                        let key = cl[self.units[u][0]].to_bits();
                        let e = sums.entry(key).or_insert_with(|| vec![0.0; k]);
                        for a in 0..k {
                            e[a] += r[a];
                        }
                    }
                    s = sums.into_values().collect();
                }
                let g = s.len() as f64;
                hi.mul(&opg(&s).scale(g / (g - 1.0).max(1.0)))?.mul(&hi)?
            }
            _ => match h() {
                Ok(h) => h.sym_inverse(1e-12).0,
                Err(_) => opg(&scores()?).sym_inverse(1e-12).0,
            },
        })
    }
}

/// Starting intercept for an outcome from the sample mean of y.
pub fn start_cons(o: &Outcome, ys: &[f64]) -> f64 {
    let v: Vec<f64> = ys.iter().copied().filter(|y| !y.is_nan()).collect();
    let n = v.len().max(1) as f64;
    let m = v.iter().sum::<f64>() / n;
    match o.fam {
        Fam::Gaussian => m,
        Fam::Lognormal => v.iter().map(|y| y.max(1e-12).ln()).sum::<f64>() / n,
        Fam::Bernoulli | Fam::Beta => {
            let p = m.clamp(0.02, 0.98);
            match o.link {
                Link::Probit => dist::invnormal(p),
                Link::Cloglog => (-(1.0 - p).ln()).ln(),
                _ => (p / (1.0 - p)).ln(),
            }
        }
        Fam::Binomial => 0.0,
        Fam::Poisson | Fam::Nbinomial | Fam::Gamma => m.max(1e-3).ln(),
        Fam::Weibull | Fam::Exponential => -(m.max(1e-3)).ln(),
        _ => 0.0,
    }
}

pub fn var_of(ys: &[f64]) -> f64 {
    let v: Vec<f64> = ys.iter().copied().filter(|y| !y.is_nan()).collect();
    let n = v.len().max(2) as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (n - 1.0)).max(1e-6)
}
