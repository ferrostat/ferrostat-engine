//! Nonlinear panel-data models: `xtlogit`, `xtprobit`, `xtpoisson` and
//! `xtnbreg`.
//!
//! * random effects: Gaussian intercepts integrated out by mean-variance
//!   adaptive Gauss-Hermite quadrature (logit, probit, Poisson-normal), or
//!   the closed-form gamma (Poisson) and beta (negative binomial) mixtures;
//! * fixed effects: conditional likelihoods (logit, Poisson, negative
//!   binomial of Hausman, Hall and Griliches);
//! * population averaged: GEE with an exchangeable working correlation.
//!
//! Every likelihood supplies analytic per-panel gradients; the Hessian is
//! obtained by differencing them (with the quadrature nodes held fixed).

use std::cell::{Cell, RefCell};

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood, Vce};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

/// Commands of the panel nonlinear / dynamic panel family.
pub fn handlers() -> Handlers {
    let mut v: Handlers = vec![
        ("xtlogit", Box::new(XtNl { verb: "xtlogit" })),
        ("xtprobit", Box::new(XtNl { verb: "xtprobit" })),
        ("xtpoisson", Box::new(XtNl { verb: "xtpoisson" })),
        ("xtnbreg", Box::new(XtNl { verb: "xtnbreg" })),
    ];
    v.extend(crate::xtgls::handlers());
    v.extend(crate::xtdpd::handlers());
    v.extend(crate::xtdiag::handlers());
    v
}

// ---------------------------------------------------------------------------
// shared panel helpers
// ---------------------------------------------------------------------------

/// The panel variable: `i()` or the one set by `xtset`.
pub fn panel_var(ds: &DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    match cmd.opt("i", 1) {
        Some(p) => crate::varlist::resolve_name(ds, p.trim()),
        None => st.tsset.as_ref().and_then(|(p, _)| p.clone()).ok_or_else(|| rc_err(459, "panel variable not set; use xtset varname ...")),
    }
}

/// The time variable set by `xtset`.
pub fn time_var(st: &Store) -> Result<String, String> {
    st.tsset.as_ref().map(|(_, t)| t.clone()).filter(|t| !t.is_empty()).ok_or_else(|| rc_err(459, "time variable not set; use xtset panelvar timevar"))
}

/// Group index (0-based, in sorted order of the panel values) of each
/// design row, and the rows of each group.
pub fn panel_groups(ds: &DataSet, ivar: &str, rows: &[usize]) -> Result<(Vec<usize>, Vec<Vec<usize>>), String> {
    let c = util::col(ds, ivar)?;
    let mut keys: Vec<f64> = rows.iter().map(|&i| c[i]).collect();
    keys.sort_by(f64::total_cmp);
    keys.dedup();
    let gid: Vec<usize> = rows.iter().map(|&i| keys.binary_search_by(|k| k.total_cmp(&c[i])).unwrap()).collect();
    let mut groups = vec![Vec::new(); keys.len()];
    for (t, &g) in gid.iter().enumerate() {
        groups[g].push(t);
    }
    Ok((gid, groups))
}

/// Keep only the design rows flagged in `keep`.
pub fn subset_design(d: &Design, keep: &[bool]) -> Design {
    let mut n = d.clone();
    let idx: Vec<usize> = (0..d.n()).filter(|&t| keep[t]).collect();
    n.cols = d.cols.iter().map(|c| idx.iter().map(|&t| c[t]).collect()).collect();
    n.y = idx.iter().map(|&t| d.y[t]).collect();
    n.w = idx.iter().map(|&t| d.w[t]).collect();
    n.rows = idx.iter().map(|&t| d.rows[t]).collect();
    for (t, &r) in d.rows.iter().enumerate() {
        if !keep[t] {
            n.mask[r] = false;
        }
    }
    n
}

/// Omit columns without within-panel variation, then within-collinear ones.
pub fn omit_within(d: &mut Design, groups: &[Vec<usize>], notes: &mut Vec<String>) {
    let k = d.k();
    for j in 0..k {
        if d.omitted[j] || d.names[j] == "_cons" {
            continue;
        }
        let c = &d.cols[j];
        let scale = c.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let varies = groups.iter().any(|g| g.iter().any(|&t| (c[t] - c[g[0]]).abs() > 1e-12 * scale));
        if !varies {
            d.omitted[j] = true;
            if !d.is_base[j] {
                notes.push(format!("note: {} omitted because of no within-group variance.", d.names[j]));
                d.names[j] = omit_name(&d.names[j]);
            }
        }
    }
    // within-transformed collinearity
    let mut dm = d.clone();
    for j in 0..k {
        for g in groups {
            let m = g.iter().map(|&t| d.cols[j][t]).sum::<f64>() / g.len() as f64;
            for &t in g {
                dm.cols[j][t] = d.cols[j][t] - m;
            }
        }
    }
    let had_cons = d.has_cons;
    if had_cons {
        dm.omitted[k - 1] = true;
    }
    dm.has_cons = false;
    dm.notes.clear();
    est::drop_collinear(&mut dm);
    for j in 0..k {
        if dm.omitted[j] && !d.omitted[j] && !(had_cons && j == k - 1) {
            d.omitted[j] = true;
            notes.push(format!("note: {} omitted because of collinearity.", d.names[j]));
            d.names[j] = omit_name(&d.names[j]);
        }
    }
}

fn omit_name(n: &str) -> String {
    match n.split_once('.') {
        Some((a, b)) if a.chars().all(|c| c.is_ascii_digit()) => format!("{a}o.{b}"),
        _ => format!("o.{n}"),
    }
}

pub struct Prep {
    pub d: Design,
    pub dep: String,
    pub ivar: String,
    pub groups: Vec<Vec<usize>>,
    pub off: Vec<f64>,
    pub offset: Option<String>,
    pub level: f64,
}

/// Estimation sample, design and panel structure of `cmd depvar indepvars`.
pub fn prepare(ds: &DataSet, st: &Store, cmd: &Command, noconstant: bool) -> Result<Prep, String> {
    if cmd.weight.is_some() {
        return Err(rc_err(101, "weights not allowed"));
    }
    let main = cmd.main.trim();
    let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
    if dep.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let dep = crate::varlist::resolve_name(ds, dep)?;
    let ivar = panel_var(ds, st, cmd)?;
    let mut extra = vec![ivar.clone()];
    let offv = match cmd.opt("offset", 3) {
        Some(o) => Some(crate::varlist::resolve_name(ds, o.trim())?),
        None => None,
    };
    let expv = match cmd.opt("exposure", 1) {
        Some(o) => Some(crate::varlist::resolve_name(ds, o.trim())?),
        None => None,
    };
    extra.extend(offv.iter().cloned());
    extra.extend(expv.iter().cloned());
    let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant, weights: "", extra })?;
    let mut off = vec![0.0; d.n()];
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
    let (_, groups) = panel_groups(ds, &ivar, &d.rows)?;
    let level = util::level(cmd, st)?;
    Ok(Prep { d, dep, ivar, groups, off, offset, level })
}

// ---------------------------------------------------------------------------
// likelihoods
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fam {
    Logit,
    Probit,
    Poisson,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Pooled(Fam),
    Quad(Fam),
    PoisGamma,
    PoolNb1,
    NbRe,
    FeLogit,
    FePois,
    FeNb,
}

/// ln Phi(z), accurate far into the lower tail.
pub fn ln_phi(z: f64) -> f64 {
    if z > -30.0 {
        let p = dist::normal(z);
        if p > 0.0 {
            return p.ln();
        }
    }
    let z2 = z * z;
    dist::lnnormalden(z) - (-z).ln() + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2) - 15.0 / (z2 * z2 * z2)).ln()
}

/// Per-observation log density and its first two derivatives in eta.
fn lf(f: Fam, y: f64, eta: f64) -> (f64, f64, f64) {
    match f {
        Fam::Logit => {
            let p = dist::invlogit(eta);
            let ll = if y != 0.0 { -(-eta).exp().ln_1p() } else { -eta.exp().ln_1p() };
            let ll = if ll.is_finite() { ll } else if y != 0.0 { eta } else { -eta };
            (ll, (if y != 0.0 { 1.0 } else { 0.0 }) - p, -p * (1.0 - p))
        }
        Fam::Probit => {
            let q = if y != 0.0 { 1.0 } else { -1.0 };
            let z = q * eta;
            let lp = ln_phi(z);
            let lam = (dist::lnnormalden(z) - lp).exp();
            (lp, q * lam, -lam * (z + lam))
        }
        Fam::Poisson => {
            let mu = eta.exp();
            (y * eta - mu - dist::lngamma(y + 1.0), y - mu, -mu)
        }
    }
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

struct XtLik<'a> {
    x: Vec<&'a [f64]>,
    y: &'a [f64],
    off: &'a [f64],
    groups: &'a [Vec<usize>],
    kind: Kind,
    a: Vec<f64>,
    w: Vec<f64>,
    cache: RefCell<Vec<(f64, f64)>>,
    frozen: Cell<bool>,
}

impl<'a> XtLik<'a> {
    fn new(x: Vec<&'a [f64]>, y: &'a [f64], off: &'a [f64], groups: &'a [Vec<usize>], kind: Kind, nq: usize) -> Self {
        let (a, w) = crate::melogit::gauss_hermite(nq.max(1));
        XtLik { x, y, off, groups, kind, a, w, cache: RefCell::new(vec![(f64::NAN, f64::NAN); groups.len()]), frozen: Cell::new(false) }
    }
    fn naux(&self) -> usize {
        match self.kind {
            Kind::Quad(_) | Kind::PoisGamma | Kind::PoolNb1 => 1,
            Kind::NbRe => 2,
            _ => 0,
        }
    }
    fn kx(&self) -> usize {
        self.x.len()
    }
    fn eta(&self, b: &[f64], t: usize) -> f64 {
        self.off[t] + self.x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>()
    }

    /// Log likelihood and gradient of panel `gi`.
    fn group(&self, gi: usize, b: &[f64]) -> (f64, Vec<f64>) {
        let obs = &self.groups[gi];
        let kx = self.kx();
        let p = kx + self.naux();
        let mut g = vec![0.0; p];
        let eta: Vec<f64> = obs.iter().map(|&t| self.eta(b, t)).collect();
        let y = self.y;
        let lg = dist::lngamma;
        let dg = dist::digamma;
        match self.kind {
            Kind::Pooled(f) => {
                let mut ll = 0.0;
                for (&t, &e) in obs.iter().zip(&eta) {
                    let (l, d1, _) = lf(f, y[t], e);
                    ll += l;
                    for j in 0..kx {
                        g[j] += d1 * self.x[j][t];
                    }
                }
                (ll, g)
            }
            Kind::Quad(f) => self.quad_group(gi, f, b, &eta),
            Kind::PoisGamma => {
                let th = (-b[kx]).exp();
                let lam: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
                let ys: f64 = obs.iter().map(|&t| y[t]).sum();
                let ls: f64 = lam.iter().sum();
                let mut ll = lg(th + ys) - lg(th) + th * th.ln() - (th + ys) * (th + ls).ln();
                for (k, &t) in obs.iter().enumerate() {
                    ll += y[t] * eta[k] - lg(y[t] + 1.0);
                }
                let r = (th + ys) / (th + ls);
                for (k, &t) in obs.iter().enumerate() {
                    for j in 0..kx {
                        g[j] += (y[t] - r * lam[k]) * self.x[j][t];
                    }
                }
                let dth = dg(th + ys) - dg(th) + th.ln() + 1.0 - (th + ls).ln() - r;
                g[kx] = -th * dth;
                (ll, g)
            }
            Kind::PoolNb1 => {
                let dl = b[kx].exp();
                let l1d = dl.ln_1p();
                let mut ll = 0.0;
                for (k, &t) in obs.iter().enumerate() {
                    let m = eta[k].exp() / dl;
                    let yt = y[t];
                    ll += lg(yt + m) - lg(m) - lg(yt + 1.0) - m * l1d + yt * (dl / (1.0 + dl)).ln();
                    let ps = dg(yt + m) - dg(m);
                    let de = m * (ps - l1d);
                    for j in 0..kx {
                        g[j] += de * self.x[j][t];
                    }
                    g[kx] += -m * ps + m * l1d - m * dl / (1.0 + dl) + yt / (1.0 + dl);
                }
                (ll, g)
            }
            Kind::NbRe => {
                let r = b[kx].exp();
                let s = b[kx + 1].exp();
                let lam: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
                let ys: f64 = obs.iter().map(|&t| y[t]).sum();
                let ls: f64 = lam.iter().sum();
                let mut ll = lg(r + s) + lg(r + ls) + lg(s + ys) - lg(r) - lg(s) - lg(r + s + ls + ys);
                let c = dg(r + s + ls + ys);
                for (k, &t) in obs.iter().enumerate() {
                    ll += lg(lam[k] + y[t]) - lg(lam[k]) - lg(y[t] + 1.0);
                    let dl = dg(r + ls) - c + dg(lam[k] + y[t]) - dg(lam[k]);
                    for j in 0..kx {
                        g[j] += dl * lam[k] * self.x[j][t];
                    }
                }
                g[kx] = r * (dg(r + s) + dg(r + ls) - dg(r) - c);
                g[kx + 1] = s * (dg(r + s) + dg(s + ys) - dg(s) - c);
                (ll, g)
            }
            Kind::FeNb => {
                let lam: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
                let ys: f64 = obs.iter().map(|&t| y[t]).sum();
                let ls: f64 = lam.iter().sum();
                let mut ll = lg(ls) + lg(ys + 1.0) - lg(ls + ys);
                let c = dg(ls) - dg(ls + ys);
                for (k, &t) in obs.iter().enumerate() {
                    ll += lg(lam[k] + y[t]) - lg(lam[k]) - lg(y[t] + 1.0);
                    let dl = c + dg(lam[k] + y[t]) - dg(lam[k]);
                    for j in 0..kx {
                        g[j] += dl * lam[k] * self.x[j][t];
                    }
                }
                (ll, g)
            }
            Kind::FePois => {
                let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lam: Vec<f64> = eta.iter().map(|e| (e - m).exp()).collect();
                let ls: f64 = lam.iter().sum();
                let ys: f64 = obs.iter().map(|&t| y[t]).sum();
                let mut ll = lg(ys + 1.0);
                for (k, &t) in obs.iter().enumerate() {
                    let pk = lam[k] / ls;
                    ll += y[t] * pk.ln() - lg(y[t] + 1.0);
                    for j in 0..kx {
                        g[j] += (y[t] - ys * pk) * self.x[j][t];
                    }
                }
                (ll, g)
            }
            Kind::FeLogit => {
                let kk = obs.iter().filter(|&&t| y[t] != 0.0).count();
                let c = eta.iter().sum::<f64>() / eta.len() as f64;
                let mut f = vec![0.0; kk + 1];
                let mut gv = vec![vec![0.0; kx]; kk + 1];
                f[0] = 1.0;
                for (k, &t) in obs.iter().enumerate() {
                    let e = (eta[k] - c).exp();
                    for j in (1..=kk.min(k + 1)).rev() {
                        let fo = f[j - 1];
                        for a in 0..kx {
                            let add = e * (gv[j - 1][a] + fo * self.x[a][t]);
                            gv[j][a] += add;
                        }
                        f[j] += fo * e;
                    }
                }
                let mut ll = -f[kk].ln();
                for (k, &t) in obs.iter().enumerate() {
                    if y[t] != 0.0 {
                        ll += eta[k] - c;
                        for j in 0..kx {
                            g[j] += self.x[j][t];
                        }
                    }
                }
                for j in 0..kx {
                    g[j] -= gv[kk][j] / f[kk];
                }
                (ll, g)
            }
        }
    }

    /// Random-intercept panel by mean-variance adaptive quadrature.
    fn quad_group(&self, gi: usize, f: Fam, b: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
        let obs = &self.groups[gi];
        let kx = self.kx();
        let y = self.y;
        let s2 = b[kx].exp();
        let lnorm = -0.5 * (2.0 * std::f64::consts::PI * s2).ln();
        let h = |u: f64| -> f64 { obs.iter().zip(eta).map(|(&t, &e)| lf(f, y[t], e + u).0).sum::<f64>() - u * u / (2.0 * s2) + lnorm };
        let sq2 = std::f64::consts::SQRT_2;
        let nq = self.a.len();
        let lw: Vec<f64> = self.w.iter().zip(&self.a).map(|(w, a)| w.ln() + a * a).collect();
        let (mut mu, mut sd) = self.cache.borrow()[gi];
        if !self.frozen.get() || !mu.is_finite() {
            // posterior mode and curvature
            let mut u = if mu.is_finite() { mu } else { 0.0 };
            let mut h2 = -1.0 / s2;
            for _ in 0..200 {
                let (mut g1, mut g2) = (-u / s2, -1.0 / s2);
                for (&t, &e) in obs.iter().zip(eta) {
                    let (_, d1, d2) = lf(f, y[t], e + u);
                    g1 += d1;
                    g2 += d2;
                }
                h2 = g2;
                let step = (-g1 / g2).clamp(-4.0, 4.0);
                u += step;
                if step.abs() < 1e-13 * (1.0 + u.abs()) {
                    break;
                }
            }
            mu = u;
            sd = 1.0 / (-h2).sqrt();
            if nq > 2 {
                for _ in 0..100 {
                    let pts: Vec<f64> = self.a.iter().map(|a| mu + sq2 * sd * a).collect();
                    let lv: Vec<f64> = pts.iter().zip(&lw).map(|(u, l)| l + h(*u)).collect();
                    let tot = lse(&lv);
                    let pi: Vec<f64> = lv.iter().map(|v| (v - tot).exp()).collect();
                    let m: f64 = pts.iter().zip(&pi).map(|(u, p)| u * p).sum();
                    let v: f64 = pts.iter().zip(&pi).map(|(u, p)| p * (u - m) * (u - m)).sum();
                    let ns = v.sqrt();
                    if !ns.is_finite() || ns <= 0.0 || !m.is_finite() {
                        break;
                    }
                    let delta = (m - mu).abs() + (ns - sd).abs();
                    mu = m;
                    sd = ns;
                    if delta < 1e-12 * (1.0 + mu.abs() + sd) {
                        break;
                    }
                }
            }
            self.cache.borrow_mut()[gi] = (mu, sd);
        }
        let lsd = (sq2 * sd).ln();
        let pts: Vec<f64> = self.a.iter().map(|a| mu + sq2 * sd * a).collect();
        let mut lv = Vec::with_capacity(nq);
        let mut gs: Vec<Vec<f64>> = Vec::with_capacity(nq);
        for (k, &u) in pts.iter().enumerate() {
            let mut l = lw[k] + lsd - u * u / (2.0 * s2) + lnorm;
            let mut gk = vec![0.0; kx + 1];
            for (&t, &e) in obs.iter().zip(eta) {
                let (ll, d1, _) = lf(f, y[t], e + u);
                l += ll;
                for j in 0..kx {
                    gk[j] += d1 * self.x[j][t];
                }
            }
            gk[kx] = -0.5 + u * u / (2.0 * s2);
            lv.push(l);
            gs.push(gk);
        }
        let tot = lse(&lv);
        let mut g = vec![0.0; kx + 1];
        for (k, gk) in gs.iter().enumerate() {
            let p = (lv[k] - tot).exp();
            for j in 0..=kx {
                g[j] += p * gk[j];
            }
        }
        (tot, g)
    }

    fn total(&self, b: &[f64], scores: bool) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let p = self.kx() + self.naux();
        let mut ll = 0.0;
        let mut g = vec![0.0; p];
        let mut sc = Vec::new();
        for gi in 0..self.groups.len() {
            let (l, gg) = self.group(gi, b);
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

impl<'a> Likelihood for XtLik<'a> {
    fn k(&self) -> usize {
        self.kx() + self.naux()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let p = self.k();
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
// GEE (population averaged)
// ---------------------------------------------------------------------------

struct GeeFit {
    b: Vec<f64>,
    v: Matrix,
    alpha: f64,
    iters: usize,
    n_clust: Option<usize>,
}

fn link_mu(f: Fam, eta: f64) -> (f64, f64, f64) {
    // (mu, dmu/deta, V(mu))
    match f {
        Fam::Logit => {
            let p = dist::invlogit(eta);
            (p, p * (1.0 - p), p * (1.0 - p))
        }
        Fam::Probit => {
            let p = dist::normal(eta);
            (p, dist::normalden(eta), p * (1.0 - p))
        }
        Fam::Poisson => {
            let m = eta.exp();
            (m, m, m)
        }
    }
}

fn gee(x: &[&[f64]], y: &[f64], off: &[f64], groups: &[Vec<usize>], f: Fam, b0: Vec<f64>, robust: Option<&[f64]>) -> Result<GeeFit, String> {
    let k = x.len();
    let n: usize = groups.iter().map(|g| g.len()).sum();
    let mut b = b0;
    let mut alpha = 0.0;
    let mut iters = 0;
    let eta = |b: &[f64], t: usize| off[t] + x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>();
    // pieces for one panel: D' W^-1 D and D' W^-1 (y - mu)
    let pieces = |b: &[f64], alpha: f64, gi: usize| -> (Matrix, Vec<f64>) {
        let g = &groups[gi];
        let m = g.len();
        let mut a = Matrix::zeros(k, k);
        let mut u = vec![0.0; k];
        let vals: Vec<(f64, f64, f64)> = g.iter().map(|&t| link_mu(f, eta(b, t))).collect();
        // standardised quantities: d_t = dmu/sqrt(V) x_t, r_t = (y-mu)/sqrt(V)
        let r: Vec<f64> = g.iter().zip(&vals).map(|(&t, (mu, _, v))| (y[t] - mu) / v.sqrt()).collect();
        let dd: Vec<Vec<f64>> = g.iter().zip(&vals).map(|(&t, (_, dm, v))| (0..k).map(|j| dm / v.sqrt() * x[j][t]).collect()).collect();
        // R^-1 = (I - c J) / (1 - alpha), c = alpha / (1 + (m-1) alpha)
        let c = alpha / (1.0 + (m as f64 - 1.0) * alpha);
        let s = 1.0 / (1.0 - alpha);
        let sd: Vec<f64> = (0..k).map(|j| dd.iter().map(|d| d[j]).sum()).collect();
        let sr: f64 = r.iter().sum();
        for t in 0..m {
            for i in 0..k {
                u[i] += s * dd[t][i] * r[t];
                for j in 0..k {
                    a[(i, j)] += s * dd[t][i] * dd[t][j];
                }
            }
        }
        for i in 0..k {
            u[i] -= s * c * sd[i] * sr;
            for j in 0..k {
                a[(i, j)] -= s * c * sd[i] * sd[j];
            }
        }
        (a, u)
    };
    for it in 0..200 {
        iters = it + 1;
        let mut a = Matrix::zeros(k, k);
        let mut u = vec![0.0; k];
        for gi in 0..groups.len() {
            let (ag, ug) = pieces(&b, alpha, gi);
            for i in 0..k {
                u[i] += ug[i];
                for j in 0..k {
                    a[(i, j)] += ag[(i, j)];
                }
            }
        }
        let (ai, _) = a.sym_inverse(1e-14);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * u[j]).sum()).collect();
        for i in 0..k {
            b[i] += step[i];
        }
        // update the exchangeable correlation from Pearson residuals
        let (mut s2, mut sx, mut npair) = (0.0, 0.0, 0.0);
        for g in groups {
            let r: Vec<f64> = g.iter().map(|&t| {
                let (mu, _, v) = link_mu(f, eta(&b, t));
                (y[t] - mu) / v.sqrt()
            })
            .collect();
            let sr: f64 = r.iter().sum();
            let sq: f64 = r.iter().map(|v| v * v).sum();
            s2 += sq;
            sx += sr * sr - sq;
            npair += (g.len() * (g.len() - 1)) as f64;
        }
        let phi = s2 / n as f64;
        let na = if npair > 0.0 { sx / npair / phi } else { 0.0 };
        let dif = step.iter().zip(&b).map(|(s, bb)| s.abs() / (bb.abs() + 1.0)).fold(0.0, f64::max).max((na - alpha).abs());
        alpha = na;
        if dif < 1e-11 {
            break;
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(rc_err(430, "convergence not achieved"));
        }
    }
    let mut a = Matrix::zeros(k, k);
    let mut us = Vec::new();
    for gi in 0..groups.len() {
        let (ag, ug) = pieces(&b, alpha, gi);
        for i in 0..k {
            for j in 0..k {
                a[(i, j)] += ag[(i, j)];
            }
        }
        us.push(ug);
    }
    let (ai, _) = a.sym_inverse(1e-14);
    let (v, n_clust) = match robust {
        None => (ai, None),
        Some(cl) => {
            // cl indexes panels; panels sharing a value form one cluster
            let (v, g) = est::sandwich(&ai, &us, Some(cl), 1.0);
            (v, Some(g))
        }
    };
    Ok(GeeFit { b, v, alpha, iters, n_clust })
}

// ---------------------------------------------------------------------------
// the commands
// ---------------------------------------------------------------------------

pub struct XtNl {
    pub verb: &'static str,
}

fn family_of(verb: &str) -> Fam {
    match verb {
        "xtlogit" => Fam::Logit,
        "xtprobit" => Fam::Probit,
        _ => Fam::Poisson,
    }
}

/// Values of the cluster variable for each panel (must be constant within
/// panels).
fn panel_clusters(ds: &DataSet, p: &Prep, vce: &Vce) -> Result<Option<Vec<f64>>, String> {
    match vce {
        Vce::Robust => Ok(Some((0..p.groups.len()).map(|g| g as f64).collect())),
        Vce::Cluster(v) => {
            let name = crate::varlist::resolve_name(ds, v)?;
            let c = util::col(ds, &name)?;
            let mut out = Vec::new();
            for g in &p.groups {
                let v0 = c[p.d.rows[g[0]]];
                if g.iter().any(|&t| c[p.d.rows[t]] != v0) {
                    return Err(rc_err(498, "panels are not nested within clusters"));
                }
                out.push(v0);
            }
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

fn obs_per_group(groups: &[Vec<usize>]) -> (f64, f64, f64) {
    let mn = groups.iter().map(|g| g.len()).min().unwrap_or(0) as f64;
    let mx = groups.iter().map(|g| g.len()).max().unwrap_or(0) as f64;
    let n: usize = groups.iter().map(|g| g.len()).sum();
    (mn, n as f64 / groups.len().max(1) as f64, mx)
}

impl CommandHandler for XtNl {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            let e = &st.e;
            let ok = e.macros.get("cmd").map(String::as_str) == Some(self.verb) || e.macros.get("cmd2").map(String::as_str) == Some(self.verb);
            if !ok {
                return Err(rc_err(301, "last estimates not found"));
            }
            let eform = cmd.has_opt("or", 2) || cmd.has_opt("irr", 3);
            return Ok(render(&st.e, eform, util::level(cmd, st)?));
        }
        let fam = family_of(self.verb);
        let model = if cmd.has_opt("fe", 2) {
            "fe"
        } else if cmd.has_opt("pa", 2) {
            "pa"
        } else if cmd.has_opt("be", 2) {
            return Err(rc_err(198, "option be not allowed"));
        } else {
            "re"
        };
        if model == "fe" && self.verb == "xtprobit" {
            return Err(rc_err(198, "option fe not allowed; there is no sufficient statistic for the panel effects of the probit model"));
        }
        if model == "pa" && self.verb == "xtnbreg" {
            return Err(rc_err(198, "option pa not supported for xtnbreg"));
        }
        let normal = cmd.has_opt("normal", 3);
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let robust_ok = match model {
            "pa" => true,
            "re" => self.verb != "xtnbreg",
            _ => self.verb == "xtpoisson",
        };
        if matches!(vce, Vce::Robust | Vce::Cluster(_)) && !robust_ok {
            return Err(rc_err(198, "vce(robust) not allowed with this model"));
        }
        let eform = match self.verb {
            "xtlogit" if cmd.has_opt("or", 2) => Some("Odds ratio"),
            "xtpoisson" | "xtnbreg" if cmd.has_opt("irr", 3) => Some("IRR"),
            _ => None,
        };
        let nocons_fe = model == "fe" && self.verb != "xtnbreg";
        let mut p = prepare(ds, st, cmd, nocons_fe || cmd.has_opt("noconstant", 3))?;
        if fam != Fam::Poisson {
            for y in p.d.y.iter_mut() {
                *y = if *y != 0.0 { 1.0 } else { 0.0 };
            }
        } else if p.d.y.iter().any(|&y| y < 0.0) {
            return Err(rc_err(459, format!("{} must be greater than or equal to zero", p.dep)));
        }
        let mut notes: Vec<String> = Vec::new();
        // conditional models: drop uninformative panels
        if model == "fe" {
            let mut keep = vec![true; p.d.n()];
            let (mut ng_drop, mut n_drop) = (0usize, 0usize);
            for g in &p.groups {
                let s: f64 = g.iter().map(|&t| p.d.y[t]).sum();
                let bad = if fam == Fam::Poisson { s == 0.0 } else { g.len() < 2 || s == 0.0 || s == g.len() as f64 };
                if bad {
                    ng_drop += 1;
                    n_drop += g.len();
                    for &t in g {
                        keep[t] = false;
                    }
                }
            }
            if ng_drop > 0 {
                let why = if fam == Fam::Poisson { "all zero outcomes" } else { "all positive or all negative outcomes" };
                let s = if ng_drop == 1 { "" } else { "s" };
                notes.push(format!("note: {} group{s} ({} obs) dropped because of {why}.", util::commas(ng_drop), util::commas(n_drop)));
                p.d = subset_design(&p.d, &keep);
                p.off = (0..keep.len()).filter(|&t| keep[t]).map(|t| p.off[t]).collect();
                p.groups = panel_groups(ds, &p.ivar, &p.d.rows)?.1;
            }
            if p.d.n() == 0 {
                return Err(rc_err(2000, "no observations"));
            }
            if self.verb != "xtnbreg" {
                omit_within(&mut p.d, &p.groups, &mut notes);
            }
        }
        let act = p.d.active();
        let k = p.d.k();
        let kx = act.len();
        let cols: Vec<&[f64]> = act.iter().map(|&j| p.d.cols[j].as_slice()).collect();
        let ng = p.groups.len();
        let n = p.d.n();
        let cl = panel_clusters(ds, &p, &vce)?;
        let mut out: Vec<String> = Vec::new();
        out.extend(p.d.notes.iter().cloned());
        out.extend(notes.iter().cloned());
        let ll_label = "Log likelihood";
        let cons_pos = act.iter().position(|&j| p.d.names[j] == "_cons");
        let ybar = p.d.y.iter().sum::<f64>() / n as f64;
        let mut b0 = vec![0.0; kx];
        if let Some(c) = cons_pos {
            b0[c] = match fam {
                Fam::Logit => dist::logit(ybar.clamp(1e-6, 1.0 - 1e-6)),
                Fam::Probit => dist::invnormal(ybar.clamp(1e-6, 1.0 - 1e-6)),
                Fam::Poisson => {
                    let so: f64 = p.off.iter().map(|o| o.exp()).sum::<f64>() / n as f64;
                    (ybar.max(1e-6) / so).ln()
                }
            };
        }
        let pooled = |kind: Kind, start: Vec<f64>| -> Result<est::MlResult, String> {
            let m = XtLik::new(cols.clone(), &p.d.y, &p.off, &p.groups, kind, 1);
            est::maximize(&m, start, 300)
        };
        // ------------------------------------------------------------ pa
        if model == "pa" {
            let pool = pooled(Kind::Pooled(fam), b0.clone())?;
            let fit = gee(&cols, &p.d.y, &p.off, &p.groups, fam, pool.b.clone(), cl.as_deref())?;
            let mut bfull = vec![0.0; k];
            for (a, &j) in act.iter().enumerate() {
                bfull[j] = fit.b[a];
            }
            let v = est::expand_v(&fit.v, &act, k);
            let (chi2, df_m) = est::wald_all(&p.d, &bfull, &v);
            let (mn, av, mx) = obs_per_group(&p.groups);
            let mut out_it = out.clone();
            out_it.push(String::new());
            for i in 1..=fit.iters {
                out_it.push(format!("Iteration {i}: Tolerance = {}", if i == fit.iters { "converged" } else { "." }).replace(": Tolerance = .", ": Tolerance = (iterating)"));
            }
            let mut d = p.d.clone();
            d.names = d.names.iter().map(|nm| format!("{}:{nm}", p.dep)).collect();
            let fam_name = match fam {
                Fam::Poisson => "Poisson",
                _ => "Binomial",
            };
            let link_name = match fam {
                Fam::Logit => "Logit",
                Fam::Probit => "Probit",
                Fam::Poisson => "Log",
            };
            let tmax = mx as usize;
            let mut rm = Matrix::identity(tmax);
            for i in 0..tmax {
                for j in 0..tmax {
                    if i != j {
                        rm[(i, j)] = fit.alpha;
                    }
                }
            }
            let tn: Vec<String> = (1..=tmax).map(|i| format!("c{i}")).collect();
            let rn: Vec<String> = (1..=tmax).map(|i| format!("r{i}")).collect();
            let verb = self.verb;
            let robust = cl.is_some();
            est::post(st, "xtgee", &d, &bfull, &v, &|e: &mut Results| {
                e.set("N_g", ng as f64);
                e.set("g_min", mn);
                e.set("g_avg", av);
                e.set("g_max", mx);
                e.set("chi2", chi2);
                e.set("df_m", df_m);
                e.set("p", dist::chi2tail(df_m, chi2));
                e.set("phi", 1.0);
                e.set("tol", 1e-6);
                if let Some(g) = fit.n_clust {
                    e.set("N_clust", g as f64);
                }
                e.set_mat("R", rm.clone().with_names(rn.clone(), tn.clone()));
                e.set_str("cmd2", verb);
                e.set_str("model", "pa");
                e.set_str("family", fam_name);
                e.set_str("link", link_name);
                e.set_str("corr", "exchangeable");
                e.set_str("scale", "1");
                e.set_str("ivar", p.ivar.clone());
                e.set_str("title", "GEE population-averaged model");
                e.set_str("chi2type", "Wald");
                e.set_str("vce", if robust { "robust" } else { "conventional" });
                if robust {
                    e.set_str("vcetype", "Robust");
                    e.set_str("clustvar", p.ivar.clone());
                }
                e.set_str("predict", "xtgee_p");
                if let Some(o) = &p.offset {
                    e.set_str("offset", o.clone());
                }
                e.set_str("_head", out_it.join("\n"));
            });
            let text = format!("{}\n{}", out_it.join("\n"), render(&st.e, eform.is_some(), p.level));
            return Ok(text);
        }
        // ------------------------------------------------------------ ml models
        let mut aux_names: Vec<&str> = Vec::new();
        let (kind, comparison): (Kind, Option<Kind>) = match (self.verb, model) {
            (_, "fe") => (
                match self.verb {
                    "xtlogit" => Kind::FeLogit,
                    "xtpoisson" => Kind::FePois,
                    _ => Kind::FeNb,
                },
                None,
            ),
            ("xtnbreg", _) => {
                aux_names = vec!["ln_r", "ln_s"];
                (Kind::NbRe, Some(Kind::PoolNb1))
            }
            ("xtpoisson", _) if !normal => {
                aux_names = vec!["lnalpha"];
                (Kind::PoisGamma, Some(Kind::Pooled(Fam::Poisson)))
            }
            _ => {
                aux_names = vec!["lnsig2u"];
                (Kind::Quad(fam), Some(Kind::Pooled(fam)))
            }
        };
        let nq = match util::opt_num(cmd, "intpoints", 3)? {
            Some(q) => {
                if !(1.0..=195.0).contains(&q) {
                    return Err(rc_err(198, "intpoints() must be between 1 and 195"));
                }
                q as usize
            }
            None => 12,
        };
        let mut ll_c = SYSMISS;
        let start: Vec<f64>;
        // comparison model
        let mut pool_b = b0.clone();
        let pool_ll;
        {
            let pk = match comparison {
                Some(Kind::PoolNb1) => Kind::Pooled(Fam::Poisson),
                Some(c) => c,
                None => Kind::Pooled(Fam::Poisson),
            };
            let pre_needed = comparison.is_some() || kind == Kind::FeNb;
            if pre_needed {
                let r = pooled(pk, b0.clone())?;
                let title = match pk {
                    Kind::Pooled(Fam::Logit) => "Fitting comparison model:",
                    Kind::Pooled(Fam::Probit) => "Fitting comparison model:",
                    _ => "Fitting Poisson model:",
                };
                out.push(String::new());
                out.push(title.into());
                out.push(String::new());
                out.extend(est::iteration_log(&r.iterations, ll_label));
                pool_b = r.b.clone();
                pool_ll = r.ll;
            } else {
                pool_ll = SYSMISS;
            }
        }
        if comparison == Some(Kind::PoolNb1) {
            let mut s0 = pool_b.clone();
            s0.push(0.0);
            let r = pooled(Kind::PoolNb1, s0)?;
            out.push(String::new());
            out.push("Fitting negative binomial (constant dispersion) model:".into());
            out.push(String::new());
            out.extend(est::iteration_log(&r.iterations, ll_label));
            ll_c = r.ll;
            pool_b = r.b[..kx].to_vec();
        } else if comparison.is_some() {
            ll_c = pool_ll;
        }
        let lik = XtLik::new(cols.clone(), &p.d.y, &p.off, &p.groups, kind, nq);
        out.push(String::new());
        out.push("Fitting full model:".into());
        out.push(String::new());
        match kind {
            Kind::Quad(f) => {
                // grid search over rho
                let c = match f {
                    Fam::Logit => std::f64::consts::PI.powi(2) / 3.0,
                    _ => 1.0,
                };
                let mut best: (f64, Vec<f64>) = (f64::NEG_INFINITY, vec![]);
                let mut tau = 0.0f64;
                let mut prev = f64::NEG_INFINITY;
                while tau < 0.95 {
                    let mut s: Vec<f64> = if f == Fam::Poisson { pool_b.clone() } else { pool_b.iter().map(|b| b / (1.0 - tau).sqrt()).collect() };
                    let s2 = if tau == 0.0 { 1e-8 } else { tau / (1.0 - tau) * c };
                    if f == Fam::Poisson {
                        if let Some(cp) = cons_pos {
                            s[cp] -= s2 / 2.0;
                        }
                    }
                    s.push(s2.ln());
                    let ll = if tau == 0.0 { pool_ll } else { lik.eval(&s, false).0 };
                    out.push(format!("tau = {:>4.1}     {ll_label} = {}", tau, fmt_g(ll, 9)));
                    if ll > best.0 {
                        best = (ll, s);
                    }
                    if ll < prev {
                        break;
                    }
                    prev = ll;
                    tau = ((tau * 10.0).round() + 1.0) / 10.0;
                }
                if best.1.len() != kx + 1 || !best.0.is_finite() {
                    let mut s = pool_b.clone();
                    s.push(0.0);
                    best.1 = s;
                }
                out.push(String::new());
                start = best.1;
            }
            Kind::PoisGamma => {
                let mut best = (f64::NEG_INFINITY, vec![]);
                for la in [-2.0, -1.0, 0.0, 1.0] {
                    let mut s = pool_b.clone();
                    s.push(la);
                    let ll = lik.eval(&s, false).0;
                    if ll > best.0 {
                        best = (ll, s);
                    }
                }
                start = best.1;
            }
            Kind::NbRe => {
                let mut best = (f64::NEG_INFINITY, vec![]);
                for (lr, ls) in [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.5, -0.5), (3.0, 2.0)] {
                    let mut s = pool_b.clone();
                    s.push(lr);
                    s.push(ls);
                    let ll = lik.eval(&s, false).0;
                    if ll > best.0 {
                        best = (ll, s);
                    }
                }
                start = best.1;
            }
            Kind::FeNb => {
                start = pool_b.clone();
            }
            _ => {
                start = vec![0.0; kx];
            }
        }
        let res = est::maximize(&lik, start, 300)?;
        out.extend(est::iteration_log(&res.iterations, ll_label));
        let pn = res.b.len();
        let (hinv, _) = res.info.sym_inverse(1e-14);
        let (vth, n_clust) = match &cl {
            Some(c) => {
                let (v, g) = est::sandwich(&hinv, &res.scores, Some(c), 1.0);
                let gf = g as f64;
                (v.scale(gf / (gf - 1.0)), Some(g))
            }
            None => (hinv, None),
        };
        // full coefficient vector: fixed part then auxiliary parameters
        let total = k + pn - kx;
        let mut b = vec![0.0; total];
        let mut idx: Vec<usize> = act.clone();
        for a in 0..pn - kx {
            idx.push(k + a);
        }
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vth, &idx, total);
        let mut names: Vec<String> = p.d.names.iter().map(|nm| format!("{}:{nm}", p.dep)).collect();
        for an in &aux_names {
            names.push(format!("/:{an}"));
        }
        let mut d = p.d.clone();
        d.names = names;
        d.omitted.extend(vec![false; pn - kx]);
        d.is_base.extend(vec![false; pn - kx]);
        let vfix = est::expand_v(&vth, &idx, total);
        let mut dk = p.d.clone();
        dk.names = p.d.names.clone();
        let mut vk = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                vk[(i, j)] = vfix[(i, j)];
            }
        }
        let (chi2, df_m) = est::wald_all(&dk, &b[..k], &vk);
        // LR chi2 for the conditional logit (vs. all coefficients zero)
        let (chi2_out, chi2type, ll0) = if kind == Kind::FeLogit && cl.is_none() {
            let ll0 = lik.eval(&vec![0.0; kx], false).0;
            (2.0 * (res.ll - ll0), "LR", ll0)
        } else {
            (chi2, "Wald", SYSMISS)
        };
        let lr = if is_missing(ll_c) || cl.is_some() { SYSMISS } else { (2.0 * (res.ll - ll_c)).max(0.0) };
        let (mn, av, mx) = obs_per_group(&p.groups);
        let title = match (self.verb, model) {
            ("xtlogit", "fe") => "Conditional fixed-effects logistic regression",
            ("xtpoisson", "fe") => "Conditional fixed-effects Poisson regression",
            ("xtnbreg", "fe") => "Conditional FE negative binomial regression",
            ("xtlogit", _) => "Random-effects logistic regression",
            ("xtprobit", _) => "Random-effects probit regression",
            ("xtpoisson", _) => "Random-effects Poisson regression",
            _ => "Random-effects negative binomial regression",
        };
        let distrib = match kind {
            Kind::Quad(_) => "Gaussian",
            Kind::PoisGamma => "Gamma",
            Kind::NbRe => "Beta",
            _ => "",
        };
        let verb = self.verb;
        let head = out.join("\n");
        let ivar = p.ivar.clone();
        let quad = matches!(kind, Kind::Quad(_));
        let aux: Vec<f64> = res.b[kx..].to_vec();
        est::post(st, verb, &d, &b, &v, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            e.set("ll", res.ll);
            if !is_missing(ll0) {
                e.set("ll_0", ll0);
            }
            if !is_missing(ll_c) {
                e.set("ll_c", ll_c);
            }
            if !is_missing(lr) {
                e.set("chi2_c", lr);
                e.set("p_c", 0.5 * dist::chi2tail(1.0, lr));
            }
            e.set("chi2", chi2_out);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2_out));
            e.set("k", pn as f64);
            e.set("rank", pn as f64);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            if quad {
                e.set("n_quad", nq as f64);
                let s2 = aux[0].exp();
                e.set("sigma_u", s2.sqrt());
                if fam != Fam::Poisson {
                    let c = if fam == Fam::Logit { std::f64::consts::PI.powi(2) / 3.0 } else { 1.0 };
                    e.set("rho", s2 / (s2 + c));
                }
                e.set_str("intmethod", "mvaghermite");
            }
            if kind == Kind::PoisGamma {
                e.set("alpha", aux[0].exp());
            }
            if kind == Kind::NbRe {
                e.set("r", aux[0].exp());
                e.set("s", aux[1].exp());
            }
            e.set_str("model", model);
            e.set_str("ivar", ivar.clone());
            e.set_str("title", title);
            if !distrib.is_empty() {
                e.set_str("distrib", distrib);
            }
            e.set_str("chi2type", chi2type);
            match &vce {
                Vce::Robust | Vce::Cluster(_) => {
                    e.set_str("vce", "cluster");
                    e.set_str("vcetype", "Robust");
                    e.set_str("clustvar", match &vce {
                        Vce::Cluster(c) => c.clone(),
                        _ => ivar.clone(),
                    });
                }
                _ => e.set_str("vce", "oim"),
            }
            e.set_str("predict", format!("{verb}_p"));
            if let Some(o) = &p.offset {
                e.set_str("offset", o.clone());
            }
            e.set_str("_head", head.clone());
        });
        let text = format!("{head}\n{}", render(&st.e, eform.is_some(), p.level));
        if !res.converged {
            return Ok(format!("{text}\nconvergence not achieved"));
        }
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// display
// ---------------------------------------------------------------------------

/// Right-hand header column: label = value, values right-aligned together.
pub fn header_block(rows: &[(String, Option<(String, String)>)]) -> Vec<String> {
    let w = rows.iter().filter_map(|r| r.1.as_ref()).filter(|(_, v)| !v.is_empty()).map(|(_, v)| v.chars().count()).max().unwrap_or(0);
    rows.iter()
        .map(|(l, r)| match r {
            Some((lab, v)) if v.is_empty() => format!("{:<49}{}", l, lab),
            Some((lab, v)) => format!("{:<49}{:<16} = {:>w$}", l, lab, v),
            None => l.clone(),
        })
        .collect()
}

pub fn hv(l: &str, lab: &str, v: String) -> (String, Option<(String, String)>) {
    (l.to_string(), Some((lab.to_string(), v)))
}

pub fn hl(l: &str) -> (String, Option<(String, String)>) {
    (l.to_string(), None)
}

fn aux_row(lw: usize, label: &str, b: f64, se: f64, lo: f64, hi: f64) -> String {
    let c = |v: f64| fmt_g(v, 8);
    format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", label, c(b), c(se), "", c(lo), c(hi))
}

/// Header, coefficient table, ancillary parameters and LR test from e().
fn render(e: &Results, eform: bool, level: f64) -> String {
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let mac = |s: &str| e.macros.get(s).cloned().unwrap_or_default();
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let dep = mac("depvar");
    let model = mac("model");
    let cmd = if mac("cmd") == "xtgee" { mac("cmd2") } else { mac("cmd") };
    let robust = !mac("vcetype").is_empty();
    let mut h = vec![hl(""), hv(&mac("title"), "Number of obs", util::commas(get("N") as usize)), hv(&format!("Group variable: {}", mac("ivar")), "Number of groups", util::commas(get("N_g") as usize)), hl("")];
    let opg = |h: &mut Vec<_>| {
        h.push(hv("", &format!("{:>16}", "min"), util::commas(get("g_min") as usize)));
        h.push(hv("", &format!("{:>16}", "avg"), format!("{:.1}", get("g_avg"))));
        h.push(hv("", &format!("{:>16}", "max"), util::commas(get("g_max") as usize)));
    };
    if model == "pa" {
        h.pop();
        h.push(hv(&format!("Family: {}", mac("family")), "Obs per group:", String::new()));
        h.push(hv(&format!("Link:   {}", mac("link")), &format!("{:>16}", "min"), util::commas(get("g_min") as usize)));
        h.push(hv(&format!("Correlation: {}", mac("corr")), &format!("{:>16}", "avg"), format!("{:.1}", get("g_avg"))));
        h.push(hv("", &format!("{:>16}", "max"), util::commas(get("g_max") as usize)));
        h.push(hv("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
        h.push(hv(&format!("Scale parameter = {}", fmt_g(get("phi"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    } else {
        let dist_line = match mac("distrib").as_str() {
            "" => String::new(),
            d => format!("Random effects u_i ~ {d}"),
        };
        h.push(hv(&dist_line, "Obs per group:", String::new()));
        opg(&mut h);
        h.push(hl(""));
        if e.get("n_quad").is_some() {
            h.push(hv(&format!("Integration method: {}", mac("intmethod")), "Integration pts.", fmt_g(get("n_quad"), 9)));
            h.push(hl(""));
        }
        h.push(hv("", &format!("{} chi2({})", mac("chi2type"), fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
        h.push(hv(&format!("Log likelihood = {}", fmt_g(get("ll"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    }
    let mut out = header_block(&h);
    if robust {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(get("N_clust") as usize), mac("clustvar"))));
    }
    out.push(String::new());
    // coefficient table
    let names: Vec<String> = b.colnames.iter().map(|n| if n.starts_with("/:") { n.clone() } else { n.split_once(':').map(|x| x.1.to_string()).unwrap_or_else(|| n.clone()) }).collect();
    let is_base: Vec<bool> = names.iter().map(|c| c.split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = names.iter().zip(&is_base).map(|(c, ib)| *ib || c.starts_with("o.") || c.split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('o') && l.trim_end_matches(['o', 'b']).parse::<f64>().is_ok()))).collect();
    let eform_title = if eform {
        Some(match cmd.as_str() {
            "xtlogit" => "Odds ratio",
            _ => "IRR",
        })
    } else {
        None
    };
    let bv = b.data[..b.cols].to_vec();
    let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &bv, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: eform_title, coef_title: "Coefficient" });
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    if robust {
        let label = if model == "pa" { "Robust" } else { "Robust" };
        tbl.insert(1, format!("{:>lw$} |{:>21}", "", label));
    }
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let aux = |name: &str| -> Option<(f64, f64)> { b.colnames.iter().position(|c| c == &format!("/:{name}")).map(|j| (b.data[j], v[(j, j)].max(0.0).sqrt())) };
    let mut lr_line = String::new();
    if let Some((t, se)) = aux("lnsig2u") {
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let s = (t / 2.0).exp();
        tbl.push(aux_row(lw, "sigma_u", s, 0.5 * s * se, ((t - crit * se) / 2.0).exp(), ((t + crit * se) / 2.0).exp()));
        if cmd != "xtpoisson" {
            let c = if cmd == "xtlogit" { std::f64::consts::PI.powi(2) / 3.0 } else { 1.0 };
            let rho = |t: f64| t.exp() / (t.exp() + c);
            let r = rho(t);
            tbl.push(aux_row(lw, "rho", r, r * (1.0 - r) * se, rho(t - crit * se), rho(t + crit * se)));
            lr_line = "LR test of rho=0".into();
        } else {
            lr_line = "LR test of sigma_u=0".into();
        }
    }
    if let Some((t, se)) = aux("lnalpha") {
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let a = t.exp();
        tbl.push(aux_row(lw, "alpha", a, a * se, (t - crit * se).exp(), (t + crit * se).exp()));
        lr_line = "LR test of alpha=0".into();
    }
    if let (Some((r, rse)), Some((s, sse))) = (aux("ln_r"), aux("ln_s")) {
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        tbl.push(aux_row(lw, "r", r.exp(), r.exp() * rse, (r - crit * rse).exp(), (r + crit * rse).exp()));
        tbl.push(aux_row(lw, "s", s.exp(), s.exp() * sse, (s - crit * sse).exp(), (s + crit * sse).exp()));
        lr_line = "LR test vs. pooled".into();
    }
    tbl.push(bar);
    out.extend(tbl);
    if eform && model == "re" {
        let what = if cmd == "xtlogit" { "odds" } else { "rate" };
        out.push(format!("Note: _cons estimates baseline {what} (conditional on zero random effects)."));
    }
    if let (Some(c), Some(pc)) = (e.get("chi2_c"), e.get("p_c")) {
        if !lr_line.is_empty() {
            out.push(format!("{:<52}Prob >= chibar2 = {:.3}", format!("{lr_line}: chibar2(01) = {:.2}", c), pc));
        }
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// `predict` after the commands of this module; None for other estimators.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let cmd2 = st.e.macros.get("cmd2").cloned().unwrap_or_default();
    let verb = if ecmd == "xtgee" && !cmd2.is_empty() { cmd2.clone() } else { ecmd.clone() };
    if !matches!(verb.as_str(), "xtlogit" | "xtprobit" | "xtpoisson" | "xtnbreg") {
        return Ok(None);
    }
    let model = st.e.macros.get("model").cloned().unwrap_or_default();
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let known = ["xb", "stdp", "pu0", "pr", "pc1", "nu0", "iru0", "n", "mu", "rate"];
    let stat = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).find(|n| known.contains(&n.as_str()));
    let mut msg = String::new();
    let stat = match stat {
        Some(s) => s,
        None => {
            let (s, m) = if model == "pa" {
                if verb == "xtpoisson" { ("mu", "(option mu assumed; predicted mean)") } else { ("mu", "(option mu assumed; Pr(" ) }
            } else {
                ("xb", "(option xb assumed; linear prediction)")
            };
            msg = if s == "mu" && verb != "xtpoisson" { format!("{m}{}))", st.e.macros.get("depvar").cloned().unwrap_or_default()) } else { m.to_string() };
            s.to_string()
        }
    };
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let mask = util::touse(ds, st, cmd)?;
    let sample = st.e.sample.clone();
    let n = ds.n_obs;
    let env = Env::with_store(ds, st);
    let offset = st.e.macros.get("offset").cloned();
    let offv = |i: usize| -> Result<f64, String> {
        match &offset {
            Some(o) => {
                let ex = ferrostat_parser::parser::parse_expression(o).map_err(|e| rc_err(198, e.message))?;
                env.eval(&ex, i)?.num()
            }
            None => Ok(0.0),
        }
    };
    let lns2 = b.colnames.iter().position(|c| c == "/:lnsig2u").map(|j| b.data[j]);
    let fam = family_of(&verb);
    let mut vals = vec![SYSMISS; n];
    let mut xb = vec![SYSMISS; n];
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let x = crate::postest::xb_row(&env, &b, None, i)?;
        if is_missing(x) {
            continue;
        }
        let o = offv(i)?;
        if is_missing(o) {
            continue;
        }
        xb[i] = x + o;
        if stat == "iru0" {
            xb[i] = x;
        }
    }
    let cdf = |z: f64| match fam {
        Fam::Logit => dist::invlogit(z),
        Fam::Probit => dist::normal(z),
        Fam::Poisson => z.exp(),
    };
    match stat.as_str() {
        "xb" => vals = xb.clone(),
        "stdp" => {
            for i in 0..n {
                if is_missing(xb[i]) {
                    continue;
                }
                let x: Vec<f64> = b.colnames.iter().map(|c| if c.starts_with("/:") { Ok(0.0) } else { crate::postest::coef_value(&env, c, i) }).collect::<Result<_, _>>()?;
                let mut s = 0.0;
                for a in 0..x.len() {
                    for c in 0..x.len() {
                        s += x[a] * v[(a, c)] * x[c];
                    }
                }
                vals[i] = s.max(0.0).sqrt();
            }
        }
        "pu0" | "nu0" | "iru0" | "mu" | "n" | "rate" => {
            for i in 0..n {
                if !is_missing(xb[i]) {
                    vals[i] = cdf(xb[i]);
                }
            }
        }
        "pr" => {
            if model == "pa" || model == "fe" || lns2.is_none() {
                for i in 0..n {
                    if !is_missing(xb[i]) {
                        vals[i] = cdf(xb[i]);
                    }
                }
            } else {
                // marginal probability: integrate over the random effect
                let s = (lns2.unwrap() / 2.0).exp();
                let (a, w) = crate::melogit::gauss_hermite(40);
                for i in 0..n {
                    if !is_missing(xb[i]) {
                        vals[i] = a.iter().zip(&w).map(|(a, w)| w * cdf(xb[i] + std::f64::consts::SQRT_2 * s * a)).sum::<f64>() / std::f64::consts::PI.sqrt();
                    }
                }
            }
        }
        "pc1" => {
            if model != "fe" || verb != "xtlogit" {
                return Err(rc_err(198, "option pc1 not allowed"));
            }
            let ivar = st.e.macros.get("ivar").cloned().unwrap_or_default();
            let pv = util::col(ds, &ivar)?;
            let mut sums: std::collections::HashMap<u64, f64> = Default::default();
            for i in 0..n {
                if !is_missing(xb[i]) && sample.as_ref().map_or(true, |s| s.get(i).copied().unwrap_or(false)) {
                    *sums.entry(pv[i].to_bits()).or_insert(0.0) += xb[i].exp();
                }
            }
            for i in 0..n {
                if !is_missing(xb[i]) {
                    if let Some(s) = sums.get(&pv[i].to_bits()) {
                        vals[i] = xb[i].exp() / s;
                    }
                }
            }
        }
        _ => return Err(rc_err(198, format!("option {stat} not allowed"))),
    }
    let nmiss = (0..n).filter(|&i| is_missing(vals[i])).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(Some(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_grad(kind: Kind, nq: usize) {
        // small synthetic panel
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 5.0 - 1.0).collect();
        let x0 = vec![1.0; n];
        let y: Vec<f64> = (0..n).map(|i| ((i * 13) % 5) as f64 * if matches!(kind, Kind::Quad(Fam::Logit) | Kind::Quad(Fam::Probit) | Kind::Pooled(Fam::Logit) | Kind::FeLogit) { 0.0 } else { 1.0 } + if (i * 3) % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let off = vec![0.0; n];
        let groups: Vec<Vec<usize>> = (0..8).map(|g| (g * 5..g * 5 + 5).collect()).collect();
        let fe = matches!(kind, Kind::FeLogit | Kind::FePois);
        let x: Vec<&[f64]> = if fe { vec![&x1] } else { vec![&x1, &x0] };
        let m = XtLik::new(x, &y, &off, &groups, kind, nq);
        let mut b: Vec<f64> = vec![0.3; m.k()];
        if m.naux() > 0 {
            let k = m.k();
            b[k - 1] = -0.4;
        }
        let (_, g, _) = m.total(&b, false);
        for j in 0..b.len() {
            let h = 1e-6;
            let mut bp = b.clone();
            bp[j] += h;
            m.frozen.set(true);
            let lp = m.total(&bp, false).0;
            bp[j] -= 2.0 * h;
            let lm = m.total(&bp, false).0;
            m.frozen.set(false);
            let num = (lp - lm) / (2.0 * h);
            assert!((num - g[j]).abs() < 1e-5 * (1.0 + g[j].abs()), "{kind:?} param {j}: analytic {} numeric {num}", g[j]);
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        for k in [Kind::Pooled(Fam::Logit), Kind::Pooled(Fam::Poisson), Kind::Quad(Fam::Logit), Kind::Quad(Fam::Probit), Kind::Quad(Fam::Poisson), Kind::PoisGamma, Kind::PoolNb1, Kind::NbRe, Kind::FeLogit, Kind::FePois, Kind::FeNb] {
            check_grad(k, 8);
        }
    }
}
