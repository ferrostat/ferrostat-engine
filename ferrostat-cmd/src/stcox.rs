//! Cox proportional hazards regression (`stcox`) and its postestimation:
//! `predict` (hazard ratios, baseline functions, residuals), `estat phtest`
//! (Grambsch-Therneau test on scaled Schoenfeld residuals), `estat
//! concordance` (Harrell's C) and `stcurve`.
//!
//! The partial likelihood handles delayed entry and multiple records per
//! subject (a record is at risk at t when `_t0 < t <= _t`), strata, weights
//! and the four tie methods: Breslow, Efron, exact marginal (the integral
//! of Kalbfleisch & Prentice, evaluated by quadrature) and exact partial
//! (the conditional-logistic likelihood, by the recursion of Gail, Lubin &
//! Rubinstein). Robust variances use the efficient score residuals of
//! Lin & Wei (1989). Time-varying coefficients (`tvc()`) multiply the
//! covariates by a function of analysis time evaluated at each failure time.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::est::{self, DesignSpec, Likelihood, Vce};
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::survival::{gc, require_st, st_header, st_id, st_weight};
use crate::util;
use crate::varlist;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

/// The survival-analysis commands of this module family.
pub fn handlers() -> Handlers {
    vec![
        ("stcox", Box::new(Stcox)),
        ("streg", Box::new(crate::streg::Streg)),
        ("stcurve", Box::new(Stcurve)),
        ("stsum", Box::new(crate::stsum::Stsum)),
        ("stci", Box::new(crate::stsum::Stci)),
        ("stdescribe", Box::new(crate::stsum::Stdescribe)),
        ("stdes", Box::new(crate::stsum::Stdescribe)),
        ("stptime", Box::new(crate::stsum::Stptime)),
        ("stsplit", Box::new(crate::stsum::Stsplit)),
        ("stjoin", Box::new(crate::stsum::Stjoin)),
    ]
}

// ---------------------------------------------------------------------------
// partial likelihood
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ties {
    Breslow,
    Efron,
    ExactM,
    ExactP,
}

impl Ties {
    fn name(self) -> &'static str {
        match self {
            Ties::Breslow => "breslow",
            Ties::Efron => "efron",
            Ties::ExactM => "exactm",
            Ties::ExactP => "exactp",
        }
    }
    fn parse(s: &str) -> Ties {
        match s {
            "efron" => Ties::Efron,
            "exactm" => Ties::ExactM,
            "exactp" => Ties::ExactP,
            _ => Ties::Breslow,
        }
    }
}

/// Distinct failure time within a stratum.
#[derive(Debug, Clone)]
pub struct FailGroup {
    pub stratum: usize,
    pub tau: f64,
    /// value of the tvc time function at tau
    pub g: f64,
    pub fails: Vec<usize>,
}

/// Data of a Cox model; covariates are centred (the partial likelihood is
/// invariant to centring) and `xmean` restores the original scale.
#[derive(Debug, Clone)]
pub struct Cox {
    pub x: Vec<Vec<f64>>,
    pub z: Vec<Vec<f64>>,
    pub kx: usize,
    pub kz: usize,
    pub xmean: Vec<f64>,
    pub zmean: Vec<f64>,
    pub t0: Vec<f64>,
    pub t: Vec<f64>,
    pub d: Vec<bool>,
    pub w: Vec<f64>,
    pub strata: Vec<usize>,
    /// offset added to every linear predictor (log frailties)
    pub off: Vec<f64>,
    pub ties: Ties,
    pub groups: Vec<FailGroup>,
    /// failure groups of each stratum, ordered by time
    pub sgroups: Vec<Vec<usize>>,
    /// records of each stratum
    pub srecs: Vec<Vec<usize>>,
}

/// Per-failure-time quantities used by residuals and baseline functions.
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    /// risk-set sums at beta (centred scale)
    pub s0: f64,
    pub sw_d: f64,
    /// hazard weights for at-risk records not failing / failing here
    pub hz: f64,
    pub hx: Vec<f64>,
    pub hz_d: f64,
    pub hx_d: Vec<f64>,
    /// expected covariate of the failures (Schoenfeld centring)
    pub xstar: Vec<f64>,
}

impl Cox {
    #[allow(clippy::too_many_arguments)]
    pub fn new(mut x: Vec<Vec<f64>>, mut z: Vec<Vec<f64>>, gfun: &dyn Fn(f64) -> f64, t0: Vec<f64>, t: Vec<f64>, d: Vec<bool>, w: Vec<f64>, strata: Vec<usize>, ties: Ties) -> Cox {
        let n = t.len();
        let kx = x.first().map_or(0, |r| r.len());
        let kz = z.first().map_or(0, |r| r.len());
        let sw: f64 = w.iter().sum();
        let center = |m: &mut Vec<Vec<f64>>, k: usize| -> Vec<f64> {
            let mean: Vec<f64> = (0..k).map(|j| (0..n).map(|i| w[i] * m[i][j]).sum::<f64>() / sw).collect();
            for r in m.iter_mut() {
                for j in 0..k {
                    r[j] -= mean[j];
                }
            }
            mean
        };
        let xmean = center(&mut x, kx);
        let zmean = center(&mut z, kz);
        let ns = strata.iter().copied().max().map_or(0, |m| m + 1);
        let mut srecs = vec![Vec::new(); ns];
        for i in 0..n {
            srecs[strata[i]].push(i);
        }
        let mut groups = Vec::new();
        let mut sgroups = vec![Vec::new(); ns];
        for (s, recs) in srecs.iter().enumerate() {
            let mut fails: Vec<usize> = recs.iter().copied().filter(|&i| d[i]).collect();
            fails.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
            let mut k = 0;
            while k < fails.len() {
                let tau = t[fails[k]];
                let mut f = Vec::new();
                while k < fails.len() && t[fails[k]] == tau {
                    f.push(fails[k]);
                    k += 1;
                }
                sgroups[s].push(groups.len());
                groups.push(FailGroup { stratum: s, tau, g: gfun(tau), fails: f });
            }
        }
        Cox { x, z, kx, kz, xmean, zmean, t0, t, d, w, strata, off: vec![0.0; n], ties, groups, sgroups, srecs }
    }

    pub fn n(&self) -> usize {
        self.t.len()
    }

    fn k(&self) -> usize {
        self.kx + self.kz
    }

    /// Covariate vector of record i at failure time with tvc multiplier g.
    fn cov(&self, i: usize, g: f64) -> Vec<f64> {
        let mut v = self.x[i].clone();
        v.extend(self.z[i].iter().map(|z| z * g));
        v
    }

    fn eta(&self, b: &[f64], i: usize, g: f64) -> f64 {
        let mut s: f64 = self.off[i] + self.x[i].iter().zip(b).map(|(x, b)| x * b).sum::<f64>();
        if self.kz > 0 {
            s += self.z[i].iter().zip(&b[self.kx..]).map(|(z, b)| z * b * g).sum::<f64>();
        }
        s
    }

    /// Records at risk at group f.
    fn at_risk(&self, f: &FailGroup) -> impl Iterator<Item = usize> + '_ {
        let tau = f.tau;
        self.srecs[f.stratum].iter().copied().filter(move |&i| self.t0[i] < tau && self.t[i] >= tau)
    }

    /// Risk-set sums (S0, S1, S2) of every failure group at beta.
    fn risk_sums(&self, b: &[f64]) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        let k = self.k();
        let mut out: Vec<(f64, Vec<f64>, Vec<f64>)> = vec![(0.0, vec![0.0; k], vec![0.0; k * k]); self.groups.len()];
        if self.kz > 0 {
            for (gi, f) in self.groups.iter().enumerate() {
                let (mut s0, mut s1, mut s2) = (0.0, vec![0.0; k], vec![0.0; k * k]);
                for i in self.at_risk(f) {
                    let x = self.cov(i, f.g);
                    let r = self.w[i] * self.eta(b, i, f.g).exp();
                    s0 += r;
                    for a in 0..k {
                        s1[a] += r * x[a];
                        for c in 0..=a {
                            s2[a * k + c] += r * x[a] * x[c];
                        }
                    }
                }
                sym_fill(&mut s2, k);
                out[gi] = (s0, s1, s2);
            }
            return out;
        }
        // sweep down the failure times: add records with t >= tau, remove those with t0 >= tau
        for (s, recs) in self.srecs.iter().enumerate() {
            let mut by_t = recs.clone();
            by_t.sort_by(|&a, &c| self.t[c].total_cmp(&self.t[a]));
            let mut by_t0 = recs.clone();
            by_t0.sort_by(|&a, &c| self.t0[c].total_cmp(&self.t0[a]));
            let (mut pa, mut pr) = (0usize, 0usize);
            let (mut s0, mut s1, mut s2) = (0.0, vec![0.0; k], vec![0.0; k * k]);
            let add = |i: usize, sign: f64, s0: &mut f64, s1: &mut [f64], s2: &mut [f64]| {
                let x = &self.x[i];
                let r = sign * self.w[i] * self.eta(b, i, 0.0).exp();
                *s0 += r;
                for a in 0..k {
                    s1[a] += r * x[a];
                    for c in 0..=a {
                        s2[a * k + c] += r * x[a] * x[c];
                    }
                }
            };
            for &gi in self.sgroups[s].iter().rev() {
                let tau = self.groups[gi].tau;
                while pa < by_t.len() && self.t[by_t[pa]] >= tau {
                    add(by_t[pa], 1.0, &mut s0, &mut s1, &mut s2);
                    pa += 1;
                }
                while pr < by_t0.len() && self.t0[by_t0[pr]] >= tau {
                    // only records already added can leave (t0 < t always)
                    add(by_t0[pr], -1.0, &mut s0, &mut s1, &mut s2);
                    pr += 1;
                }
                let mut s2f = s2.clone();
                sym_fill(&mut s2f, k);
                out[gi] = (s0, s1.clone(), s2f);
            }
        }
        out
    }

    /// Log likelihood, gradient, information and per-group statistics for
    /// the Breslow and Efron approximations.
    pub fn approx(&self, b: &[f64]) -> (f64, Vec<f64>, Matrix, Vec<GroupStats>) {
        let k = self.k();
        let sums = self.risk_sums(b);
        let mut ll = 0.0;
        let mut grad = vec![0.0; k];
        let mut info = Matrix::zeros(k, k);
        let mut stats = Vec::with_capacity(self.groups.len());
        for (gi, f) in self.groups.iter().enumerate() {
            let (s0, s1, s2) = &sums[gi];
            let m = f.fails.len() as f64;
            let (mut d0, mut d1, mut d2) = (0.0, vec![0.0; k], vec![0.0; k * k]);
            let mut sw_d = 0.0;
            for &i in &f.fails {
                let x = self.cov(i, f.g);
                let e = self.eta(b, i, f.g);
                let r = self.w[i] * e.exp();
                sw_d += self.w[i];
                ll += self.w[i] * e;
                d0 += r;
                for a in 0..k {
                    grad[a] += self.w[i] * x[a];
                    d1[a] += r * x[a];
                    for c in 0..k {
                        d2[a * k + c] += r * x[a] * x[c];
                    }
                }
            }
            let mut gs = GroupStats { s0: *s0, sw_d, hz: 0.0, hx: vec![0.0; k], hz_d: 0.0, hx_d: vec![0.0; k], xstar: vec![0.0; k] };
            let reps = if self.ties == Ties::Efron { f.fails.len() } else { 1 };
            let wbar = if self.ties == Ties::Efron { sw_d / m } else { sw_d };
            for r in 0..reps {
                let fr = if self.ties == Ties::Efron { r as f64 / m } else { 0.0 };
                let a0 = s0 - fr * d0;
                if a0 <= 0.0 {
                    continue;
                }
                let a1: Vec<f64> = (0..k).map(|a| s1[a] - fr * d1[a]).collect();
                ll -= wbar * a0.ln();
                for a in 0..k {
                    grad[a] -= wbar * a1[a] / a0;
                    for c in 0..k {
                        info[(a, c)] += wbar * ((s2[a * k + c] - fr * d2[a * k + c]) / a0 - a1[a] * a1[c] / (a0 * a0));
                    }
                }
                gs.hz += wbar / a0;
                gs.hz_d += (1.0 - fr) * wbar / a0;
                for a in 0..k {
                    gs.hx[a] += wbar * a1[a] / (a0 * a0);
                    gs.hx_d[a] += (1.0 - fr) * wbar * a1[a] / (a0 * a0);
                    gs.xstar[a] += a1[a] / a0 / reps as f64;
                }
            }
            stats.push(gs);
        }
        (ll, grad, info, stats)
    }

    /// Log likelihood under the exact methods.
    fn exact_ll(&self, b: &[f64]) -> f64 {
        let mut ll = 0.0;
        for f in &self.groups {
            let risk: Vec<usize> = self.at_risk(f).collect();
            let etas: Vec<f64> = risk.iter().map(|&i| self.eta(b, i, f.g)).collect();
            let emax = etas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let m = f.fails.len();
            match self.ties {
                Ties::ExactP => {
                    // B[q] = sum over subsets of size q of prod r
                    let mut bq = vec![0.0; m + 1];
                    bq[0] = 1.0;
                    for (j, _) in risk.iter().enumerate() {
                        let r = (etas[j] - emax).exp();
                        for q in (1..=m.min(j + 1)).rev() {
                            bq[q] += r * bq[q - 1];
                        }
                    }
                    let num: f64 = f.fails.iter().map(|&i| self.eta(b, i, f.g) - emax).sum();
                    ll += num - bq[m].ln();
                }
                _ => {
                    // integral over (0, inf) of prod_i (1 - exp(-a_i t)) exp(-t)
                    let psi: f64 = risk.iter().zip(&etas).filter(|(i, _)| !f.fails.contains(i)).map(|(_, e)| (e - emax).exp()).sum();
                    if psi <= 0.0 {
                        continue;
                    }
                    let a: Vec<f64> = f.fails.iter().map(|&i| (self.eta(b, i, f.g) - emax).exp() / psi).collect();
                    ll += marginal_integral(&a).ln();
                }
            }
        }
        ll
    }

    /// Efficient score residuals (per record, unweighted) at beta.
    pub fn score_residuals(&self, b: &[f64], stats: &[GroupStats]) -> Vec<Vec<f64>> {
        let k = self.k();
        let mut out = vec![vec![0.0; k]; self.n()];
        for (s, recs) in self.srecs.iter().enumerate() {
            let gl = &self.sgroups[s];
            let taus: Vec<f64> = gl.iter().map(|&g| self.groups[g].tau).collect();
            for &i in recs {
                let lo = taus.partition_point(|&tau| tau <= self.t0[i]);
                let hi = taus.partition_point(|&tau| tau <= self.t[i]);
                let res = &mut out[i];
                for &gi in &gl[lo..hi] {
                    let f = &self.groups[gi];
                    let st = &stats[gi];
                    let x = self.cov(i, f.g);
                    let r = self.eta(b, i, f.g).exp();
                    let own = self.d[i] && f.tau == self.t[i];
                    let (hz, hx) = if own { (st.hz_d, &st.hx_d) } else { (st.hz, &st.hx) };
                    for a in 0..k {
                        res[a] -= r * (x[a] * hz - hx[a]);
                    }
                    if own {
                        for a in 0..k {
                            res[a] += x[a] - st.xstar[a];
                        }
                    }
                }
            }
        }
        out
    }

    /// Schoenfeld residuals of the failures (unweighted, per record).
    pub fn schoenfeld(&self, stats: &[GroupStats]) -> Vec<Option<Vec<f64>>> {
        let mut out = vec![None; self.n()];
        for (gi, f) in self.groups.iter().enumerate() {
            for &i in &f.fails {
                let x = self.cov(i, f.g);
                out[i] = Some(x.iter().zip(&stats[gi].xstar).map(|(a, b)| a - b).collect());
            }
        }
        out
    }

    /// Baseline (covariates = 0, original scale) hazard increments of each
    /// failure group: Breslow d/S0 and the Kalbfleisch-Prentice 1 - alpha.
    pub fn baseline(&self, b: &[f64]) -> Vec<(f64, f64)> {
        let shift: f64 = self.xmean.iter().zip(b).map(|(m, b)| m * b).sum();
        let mut out = Vec::with_capacity(self.groups.len());
        for f in &self.groups {
            let zshift: f64 = if self.kz > 0 { self.zmean.iter().zip(&b[self.kx..]).map(|(m, b)| m * b * f.g).sum() } else { 0.0 };
            let sh = shift + zshift;
            // r on the original scale: exp(eta_centred + shift)
            let risk: Vec<(f64, f64)> = self.at_risk(f).map(|i| (self.w[i], (self.eta(b, i, f.g) + sh).exp())).collect();
            let s0: f64 = risk.iter().map(|(w, r)| w * r).sum();
            let fr: Vec<(f64, f64)> = f.fails.iter().map(|&i| (self.w[i], (self.eta(b, i, f.g) + sh).exp())).collect();
            let dw: f64 = fr.iter().map(|p| p.0).sum();
            let breslow = dw / s0;
            // solve sum_D w r / (1 - alpha^r) = S0 for alpha in (0, 1)
            let kp = if fr.len() == 1 && fr[0].0 == 1.0 {
                let (_, r) = fr[0];
                1.0 - (1.0 - r / s0).max(0.0).powf(1.0 / r)
            } else {
                let fun = |a: f64| fr.iter().map(|(w, r)| w * r / (1.0 - a.powf(*r))).sum::<f64>() - s0;
                let (mut lo, mut hi) = (1e-300f64, 1.0 - 1e-15);
                if fun(lo) > 0.0 {
                    // all at risk fail: alpha = 0
                    lo = 0.0;
                    hi = 0.0;
                }
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if fun(mid) > 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                1.0 - 0.5 * (lo + hi)
            };
            out.push((breslow, kp));
        }
        out
    }
}

fn sym_fill(m: &mut [f64], k: usize) {
    for a in 0..k {
        for c in 0..a {
            m[c * k + a] = m[a * k + c];
        }
    }
}

/// Gauss-Legendre nodes and weights on (-1, 1).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// int_0^inf prod_i (1 - exp(-a_i t)) exp(-t) dt by composite Gauss-Legendre
/// over geometric panels.
fn marginal_integral(a: &[f64]) -> f64 {
    thread_local! {
        static GL: (Vec<f64>, Vec<f64>) = gauss_legendre(20);
    }
    if a.len() == 1 {
        return a[0] / (1.0 + a[0]);
    }
    GL.with(|(x, w)| {
        let amax = a.iter().cloned().fold(0.0, f64::max);
        let mut edges = vec![0.0];
        let mut e = 1e-4 / amax.max(1e-300);
        while e < 80.0 {
            edges.push(e);
            e *= 2.0;
        }
        edges.push(80.0);
        let mut tot = 0.0;
        for p in edges.windows(2) {
            let (lo, hi) = (p[0], p[1]);
            let (c, h) = ((lo + hi) / 2.0, (hi - lo) / 2.0);
            for (xi, wi) in x.iter().zip(w) {
                let t = c + h * xi;
                let f: f64 = a.iter().map(|ai| -(-ai * t).exp_m1()).product::<f64>() * (-t).exp();
                tot += h * wi * f;
            }
        }
        tot
    })
}

/// Likelihood wrapper for the Newton iterations.
struct CoxLik<'a> {
    cox: &'a Cox,
}

impl Likelihood for CoxLik<'_> {
    fn k(&self) -> usize {
        self.cox.k()
    }
    fn eval(&self, b: &[f64], _need: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        match self.cox.ties {
            Ties::Breslow | Ties::Efron => {
                let (ll, g, h, _) = self.cox.approx(b);
                (ll, g, h, Vec::new())
            }
            _ => {
                let k = b.len();
                let f = |bb: &[f64]| self.cox.exact_ll(bb);
                let ll = f(b);
                let hstep: Vec<f64> = b.iter().map(|x| 1e-4 * x.abs().max(1.0)).collect();
                let shift = |d: &[(usize, f64)]| {
                    let mut v = b.to_vec();
                    for &(j, s) in d {
                        v[j] += s;
                    }
                    f(&v)
                };
                let mut g = vec![0.0; k];
                let mut h = Matrix::zeros(k, k);
                for j in 0..k {
                    let (p, m) = (shift(&[(j, hstep[j])]), shift(&[(j, -hstep[j])]));
                    g[j] = (p - m) / (2.0 * hstep[j]);
                    h[(j, j)] = -(p - 2.0 * ll + m) / (hstep[j] * hstep[j]);
                    for c in 0..j {
                        let v = (shift(&[(j, hstep[j]), (c, hstep[c])]) - shift(&[(j, hstep[j]), (c, -hstep[c])]) - shift(&[(j, -hstep[j]), (c, hstep[c])]) + shift(&[(j, -hstep[j]), (c, -hstep[c])])) / (4.0 * hstep[j] * hstep[c]);
                        h[(j, c)] = -v;
                        h[(c, j)] = -v;
                    }
                }
                (ll, g, h, Vec::new())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// data preparation shared by stcox and its postestimation
// ---------------------------------------------------------------------------

/// Stratum index (0-based) of each row from the strata() variables.
fn strata_index(ds: &DataSet, vars: &[String], rows: &[usize]) -> Result<Vec<usize>, String> {
    if vars.is_empty() {
        return Ok(vec![0; rows.len()]);
    }
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let keys: Vec<Vec<u64>> = rows.iter().map(|&i| cols.iter().map(|c| c[i].to_bits()).collect()).collect();
    let mut uniq = keys.clone();
    uniq.sort_by(|a, b| {
        let fa: Vec<f64> = a.iter().map(|x| f64::from_bits(*x)).collect();
        let fb: Vec<f64> = b.iter().map(|x| f64::from_bits(*x)).collect();
        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    uniq.dedup();
    Ok(keys.iter().map(|k| uniq.iter().position(|u| u == k).unwrap()).collect())
}

/// Evaluate texp() at a set of analysis times.
fn texp_fn(texp: Option<&str>) -> Result<Box<dyn Fn(f64) -> f64>, String> {
    let Some(src) = texp else { return Ok(Box::new(|t| t)) };
    let e = ferrostat_parser::parser::parse_expression(src).map_err(|e| rc_err(198, e.message))?;
    Ok(Box::new(move |t: f64| {
        let mut ds = DataSet::new();
        let mut v = ferrostat_core::Variable::new("_t", VarType::Double);
        v.data = vec![ferrostat_core::Value::Double(t)];
        let _ = ds.add_variable(v);
        ds.n_obs = 1;
        let st = Store::new();
        let env = Env::with_store(&ds, &st);
        env.eval(&e, 0).and_then(|v| v.num()).unwrap_or(SYSMISS)
    }))
}

/// The fitted Cox model rebuilt from e() for postestimation.
pub struct CoxFit {
    pub cox: Cox,
    pub b: Vec<f64>,
    /// rows of the records
    pub rows: Vec<usize>,
    /// names of the coefficients (with equation prefixes)
    pub names: Vec<String>,
    pub v: Matrix,
}

pub fn fit_from_e(ds: &DataSet, st: &Store) -> Result<CoxFit, String> {
    require_st(ds)?;
    let e = &st.e;
    let bm = e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(bm.cols, bm.cols));
    let sample = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    let env = Env::with_store(ds, st);
    let mut x = vec![Vec::new(); rows.len()];
    let mut z = vec![Vec::new(); rows.len()];
    for (j, name) in bm.colnames.iter().enumerate() {
        let tvc = name.starts_with("tvc:");
        for (r, &i) in rows.iter().enumerate() {
            let val = if bm.data[j] == 0.0 && v[(j, j)] == 0.0 { 0.0 } else { crate::postest::coef_value(&env, name, i)? };
            if tvc { z[r].push(val) } else { x[r].push(val) }
        }
    }
    let (t, t0, d) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?);
    let w = match st_weight(ds, st)? {
        Some((_, w)) => rows.iter().map(|&i| w[i]).collect(),
        None => vec![1.0; rows.len()],
    };
    let svars: Vec<String> = e.macros.get("strata").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let strata = strata_index(ds, &svars, &rows)?;
    let g = texp_fn(e.macros.get("texp").map(String::as_str).filter(|s| !s.is_empty()))?;
    let ties = Ties::parse(e.macros.get("ties").map(String::as_str).unwrap_or("breslow"));
    let cox = Cox::new(x, z, &*g, rows.iter().map(|&i| t0[i]).collect(), rows.iter().map(|&i| t[i]).collect(), rows.iter().map(|&i| d[i] == 1.0).collect(), w, strata, ties);
    Ok(CoxFit { cox, b: bm.data.clone(), rows, names: bm.colnames.clone(), v })
}

// ---------------------------------------------------------------------------
// shared gamma frailty
// ---------------------------------------------------------------------------

/// Result of the shared-frailty fit.
pub struct FrailFit {
    pub b: Vec<f64>,
    pub v: Matrix,
    pub ll: f64,
    pub theta: f64,
    pub se_theta: f64,
    pub profile_log: Vec<f64>,
    pub converged: bool,
}

/// For a fixed theta: EM over the group frailties (Klein 1992). Returns the
/// marginal log likelihood (shifted so that theta -> 0 gives the partial
/// likelihood), the coefficients and the frailty means.
fn frailty_em(cox: &mut Cox, grp: &[usize], ng: usize, theta: f64, b0: &[f64], w0: &[f64]) -> Result<(f64, Vec<f64>, Vec<f64>), String> {
    let n = cox.n();
    let k = cox.k();
    let mut b = b0.to_vec();
    let mut wg = w0.to_vec();
    let mut dg = vec![0.0; ng];
    for i in 0..n {
        if cox.d[i] {
            dg[grp[i]] += cox.w[i];
        }
    }
    let it = 1.0 / theta;
    let mut hg = vec![0.0; ng];
    for _ in 0..2000 {
        for i in 0..n {
            cox.off[i] = wg[grp[i]].ln();
        }
        if k > 0 {
            let res = est::maximize(&CoxLik { cox }, b.clone(), 100)?;
            b = res.b;
        }
        // Breslow increments with the frailty-weighted risk sets
        let base = cox.baseline(&b);
        let shift: f64 = cox.xmean.iter().zip(&b).map(|(m, b)| m * b).sum();
        hg.iter_mut().for_each(|h| *h = 0.0);
        let cum = cumulative_by_record(cox, &base);
        for i in 0..n {
            let r = (cox.eta(&b, i, 0.0) - cox.off[i] + shift).exp();
            hg[grp[i]] += cox.w[i] * r * cum[i];
        }
        let new: Vec<f64> = (0..ng).map(|g| (it + dg[g]) / (it + hg[g])).collect();
        let change = new.iter().zip(&wg).map(|(a, c)| (a.ln() - c.ln()).abs()).fold(0.0, f64::max);
        wg = new;
        if change < 1e-13 {
            break;
        }
    }
    for i in 0..n {
        cox.off[i] = wg[grp[i]].ln();
    }
    let base = cox.baseline(&b);
    let shift: f64 = cox.xmean.iter().zip(&b).map(|(m, b)| m * b).sum();
    let cum = cumulative_by_record(cox, &base);
    hg.iter_mut().for_each(|h| *h = 0.0);
    let mut ll = 0.0;
    for (gi, f) in cox.groups.iter().enumerate() {
        let dt: f64 = f.fails.iter().map(|&i| cox.w[i]).sum();
        ll += dt * base[gi].0.ln() - dt * (dt.ln() - 1.0);
    }
    for i in 0..n {
        let xb = cox.eta(&b, i, 0.0) - cox.off[i] + shift;
        hg[grp[i]] += cox.w[i] * xb.exp() * cum[i];
        if cox.d[i] {
            ll += cox.w[i] * xb;
        }
    }
    for g in 0..ng {
        ll += dg[g] * theta.ln() + dist::lngamma(it + dg[g]) - dist::lngamma(it) - (it + dg[g]) * (theta * hg[g]).ln_1p();
    }
    Ok((ll, b, wg))
}

/// Baseline cumulative hazard over each record's interval (t0, t].
fn cumulative_by_record(cox: &Cox, base: &[(f64, f64)]) -> Vec<f64> {
    let mut out = vec![0.0; cox.n()];
    for (s, recs) in cox.srecs.iter().enumerate() {
        let gl = &cox.sgroups[s];
        let taus: Vec<f64> = gl.iter().map(|&g| cox.groups[g].tau).collect();
        let mut cum = vec![0.0; gl.len() + 1];
        for (k, &g) in gl.iter().enumerate() {
            cum[k + 1] = cum[k] + base[g].0;
        }
        for &r in recs {
            let hi = taus.partition_point(|&tau| tau <= cox.t[r]);
            let lo = taus.partition_point(|&tau| tau <= cox.t0[r]);
            out[r] = cum[hi] - cum[lo];
        }
    }
    out
}

/// Shared gamma frailty Cox model: profile likelihood over ln(theta) with
/// EM inside; standard errors from the penalized partial likelihood
/// (frailties as parameters, penalty (1/theta) sum(omega - exp(omega))).
pub fn frailty_fit(cox: &mut Cox, grp: &[usize], ng: usize) -> Result<FrailFit, String> {
    let k = cox.k();
    let mut b = vec![0.0; k];
    let mut w = vec![1.0; ng];
    let mut prof = |phi: f64, b: &mut Vec<f64>, w: &mut Vec<f64>| -> Result<f64, String> {
        let (ll, nb, nw) = frailty_em(cox, grp, ng, phi.exp(), b, w)?;
        *b = nb;
        *w = nw;
        Ok(ll)
    };
    let mut phi = (0.5f64).ln();
    let mut cur = prof(phi, &mut b, &mut w)?;
    let mut log = vec![cur];
    let h = 2e-3;
    let mut converged = false;
    let mut d2 = -1.0;
    for _ in 0..60 {
        let mut at = |x: f64| {
            let (mut bp, mut wp) = (b.clone(), w.clone());
            prof(x, &mut bp, &mut wp)
        };
        let (up, dn, up2, dn2) = (at(phi + h)?, at(phi - h)?, at(phi + h / 2.0)?, at(phi - h / 2.0)?);
        // Richardson-extrapolated central differences
        let d1 = (4.0 * (up2 - dn2) / h - (up - dn) / (2.0 * h)) / 3.0;
        d2 = (4.0 * (up2 - 2.0 * cur + dn2) / (h * h / 4.0) - (up - 2.0 * cur + dn) / (h * h)) / 3.0;
        let mut step = if d2 < 0.0 { -d1 / d2 } else { d1.signum() * 1.0 };
        step = step.clamp(-3.0, 3.0);
        if d1.abs() < 1e-9 && d2 < 0.0 {
            converged = true;
            break;
        }
        let mut accepted = false;
        for _ in 0..30 {
            let (mut bt, mut wt) = (b.clone(), w.clone());
            let cand = prof(phi + step, &mut bt, &mut wt)?;
            if cand >= cur - 1e-12 {
                phi += step;
                cur = cand;
                b = bt;
                w = wt;
                accepted = true;
                break;
            }
            step /= 2.0;
        }
        log.push(cur);
        if !accepted || phi < -15.0 {
            break;
        }
    }
    let theta = if phi < -15.0 { 0.0 } else { phi.exp() };
    let se_theta = if d2 < 0.0 { theta / (-d2).sqrt() } else { SYSMISS };
    // penalized information over (beta, omega)
    let n = cox.n();
    let xa: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut r: Vec<f64> = cox.x[i].iter().zip(&cox.xmean).map(|(x, m)| x + m).collect();
        r.extend((0..ng).map(|g| (grp[i] == g) as u8 as f64));
        r
    }).collect();
    let ba: Vec<f64> = b.iter().cloned().chain(w.iter().map(|x| x.ln())).collect();
    let aug = Cox::new(xa, vec![Vec::new(); n], &|t| t, cox.t0.clone(), cox.t.clone(), cox.d.clone(), cox.w.clone(), cox.strata.clone(), Ties::Breslow);
    let (_, _, mut info, _) = aug.approx(&ba);
    if theta > 0.0 {
        for g in 0..ng {
            info[(k + g, k + g)] += w[g] / theta;
        }
    }
    let (inv, _) = info.sym_inverse(1e-14);
    let mut v = Matrix::zeros(k, k);
    for a in 0..k {
        for c in 0..k {
            v[(a, c)] = inv[(a, c)];
        }
    }
    for i in 0..n {
        cox.off[i] = 0.0;
    }
    Ok(FrailFit { b, v, ll: cur, theta, se_theta, profile_log: log, converged })
}

// ---------------------------------------------------------------------------
// stcox
// ---------------------------------------------------------------------------

pub struct Stcox;

/// A command with `& _st==1` added to its if condition.
pub fn with_st_if(cmd: &Command) -> Result<Command, String> {
    let mut c = cmd.clone();
    let cond = match &cmd.if_raw {
        Some(r) => format!("({r}) & _st==1"),
        None => "_st==1".to_string(),
    };
    c.if_expr = Some(ferrostat_parser::parser::parse_expression(&cond).map_err(|e| rc_err(198, e.message))?);
    c.if_raw = Some(cond);
    c.weight = None;
    Ok(c)
}

/// Log-likelihood value as `%10.0g` (the sign counts against the width).
pub fn fmt_ll(v: f64) -> String {
    if v < 0.0 { fmt_g(v, 9) } else { fmt_g(v, 10) }
}

/// Iteration log; with `loose`, stop at the first iteration whose relative
/// change is below 1e-5 (the first pass before "Refining estimates").
pub fn iter_log(it: &[f64], label: &str, loose: bool) -> Vec<String> {
    let mut end = it.len();
    if loose {
        for k in 1..it.len() {
            if (it[k] - it[k - 1]).abs() <= 1e-5 * it[k - 1].abs().max(1e-300) {
                end = k + 1;
                break;
            }
        }
    }
    it[..end].iter().enumerate().map(|(i, ll)| format!("Iteration {i}:  {label} = {}", fmt_ll(*ll))).collect()
}

/// Header with the subject/failure counts and the model test.
pub fn st_model_header(title: &str, nsub: f64, nfail: f64, risk: f64, nobs: f64, chi_label: &str, chi2: f64, p: f64, ll_label: &str, ll: f64) -> Vec<String> {
    let vals = [gc(nsub, 12), gc(nfail, 12), gc(risk, 12)];
    let w = vals.iter().map(|s| s.len()).max().unwrap_or(1);
    let right = |l: &str, v: String| format!("{:<13} = {:>6}", l, v);
    vec![
        title.to_string(),
        String::new(),
        format!("{:<56}{}", format!("No. of subjects = {:>w$}", vals[0]), right("Number of obs", gc(nobs, 10))),
        format!("No. of failures = {:>w$}", vals[1]),
        format!("Time at risk    = {:>w$}", vals[2]),
        format!("{:<56}{}", "", right(chi_label, if is_missing(chi2) { ".".into() } else { format!("{chi2:.2}") })),
        format!("{:<56}{}", format!("{ll_label} = {}", fmt_ll(ll)), right("Prob > chi2", if is_missing(p) { ".".into() } else { format!("{p:.4}") })),
    ]
}

/// Subjects in the sample (distinct id values, or records).
pub fn n_subjects(ds: &DataSet, rows: &[usize], w: &[f64]) -> f64 {
    match st_id(ds).and_then(|v| util::col(ds, &v).ok()) {
        Some(c) => {
            let mut k: Vec<u64> = rows.iter().map(|&i| c[i].to_bits()).collect();
            k.sort();
            k.dedup();
            k.len() as f64
        }
        None => w.iter().sum(),
    }
}

impl CommandHandler for Stcox {
    fn name(&self) -> &'static str {
        "stcox"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() && !cmd.has_opt("estimate", 3) {
            if st.e.macros.get("cmd2").is_some_and(|c| c == "stcox") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        require_st(ds)?;
        let ties = if cmd.has_opt("efron", 3) {
            Ties::Efron
        } else if cmd.has_opt("exactm", 6) {
            Ties::ExactM
        } else if cmd.has_opt("exactp", 6) {
            Ties::ExactP
        } else {
            Ties::Breslow
        };
        let shared = cmd.opt("shared", 3).map(|v| varlist::resolve_name(ds, v.trim())).transpose()?;
        let strata: Vec<String> = match cmd.opt("strata", 3) {
            Some(s) => varlist::expand(ds, s)?,
            None => Vec::new(),
        };
        let tvc: Vec<String> = match cmd.opt("tvc", 3) {
            Some(s) => varlist::expand(ds, s)?,
            None => Vec::new(),
        };
        let texp = cmd.opt("texp", 3).map(|s| s.trim().to_string());
        let mut vce = est::parse_vce(cmd, Vce::Oim)?;
        let wt = st_weight(ds, st)?;
        if wt.as_ref().is_some_and(|(k, _)| k == "pweight") && vce == Vce::Oim {
            vce = Vce::Robust;
        }
        if vce == Vce::Robust {
            if let Some(id) = st_id(ds) {
                vce = Vce::Cluster(id);
            }
        }
        if vce != Vce::Oim && matches!(ties, Ties::ExactM | Ties::ExactP) {
            return Err(rc_err(198, "vce(robust) not allowed with exactm or exactp"));
        }
        if shared.is_some() && (ties != Ties::Breslow || vce != Vce::Oim || cmd.opt("tvc", 3).is_some()) {
            return Err(rc_err(198, "shared() allowed only with Breslow ties, the default variance and no tvc()"));
        }
        if wt.is_some() && matches!(ties, Ties::ExactM | Ties::ExactP) {
            return Err(rc_err(101, "weights not allowed with exactm or exactp"));
        }
        let c2 = with_st_if(cmd)?;
        let mut extra = strata.clone();
        extra.extend(tvc.iter().cloned());
        if let Vce::Cluster(v) = &vce {
            extra.push(varlist::resolve_name(ds, v)?);
        }
        if let Some(v) = &shared {
            extra.push(v.clone());
        }
        let mut d = est::build(ds, st, &c2, DesignSpec { depvar: None, indep: &cmd.main, noconstant: true, weights: "", extra })?;
        // drop weight-invalid rows
        if let Some((_, w)) = &wt {
            let keep: Vec<bool> = d.rows.iter().map(|&i| !is_missing(w[i]) && w[i] > 0.0).collect();
            if keep.iter().any(|k| !k) {
                d = crate::xtnl::subset_design(&d, &keep);
            }
        }
        // covariates constant within strata are collinear with the baseline
        // hazards: sweep stratum indicators first, then the covariates
        {
            let sidx = strata_index(ds, &strata, &d.rows)?;
            let ns = sidx.iter().copied().max().map_or(1, |m| m + 1);
            let kd = d.k();
            for s in (0..ns).rev() {
                d.cols.insert(0, sidx.iter().map(|&x| (x == s) as u8 as f64).collect());
                d.names.insert(0, format!("__s{s}"));
                d.omitted.insert(0, false);
                d.is_base.insert(0, true);
            }
            est::drop_collinear(&mut d);
            for _ in 0..ns {
                d.cols.remove(0);
                d.names.remove(0);
                d.omitted.remove(0);
                d.is_base.remove(0);
            }
            debug_assert_eq!(d.k(), kd);
        }
        let rows = d.rows.clone();
        let n = rows.len();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let act = d.active();
        let (tc, t0c, dc) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?);
        let w: Vec<f64> = match &wt {
            Some((_, w)) => rows.iter().map(|&i| w[i]).collect(),
            None => vec![1.0; n],
        };
        let x: Vec<Vec<f64>> = (0..n).map(|r| act.iter().map(|&j| d.cols[j][r]).collect()).collect();
        let tcols: Vec<Vec<f64>> = tvc.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let z: Vec<Vec<f64>> = rows.iter().map(|&i| tcols.iter().map(|c| c[i]).collect()).collect();
        let sidx = strata_index(ds, &strata, &rows)?;
        let g = texp_fn(texp.as_deref())?;
        let mut cox = Cox::new(x, z, &*g, rows.iter().map(|&i| t0c[i]).collect(), rows.iter().map(|&i| tc[i]).collect(), rows.iter().map(|&i| dc[i] == 1.0).collect(), w.clone(), sidx, ties);
        let ka = cox.k();
        let lik = CoxLik { cox: &cox };
        let res = est::maximize(&lik, vec![0.0; ka], 100)?;
        let ll0 = res.iterations[0];
        // shared gamma frailty: profile likelihood over theta
        let frail = match &shared {
            Some(sv) => {
                let c = util::col(ds, sv)?;
                let mut keys: Vec<u64> = rows.iter().map(|&i| c[i].to_bits()).collect();
                keys.sort();
                keys.dedup();
                let gidx: Vec<usize> = rows.iter().map(|&i| keys.binary_search(&c[i].to_bits()).unwrap()).collect();
                let f = frailty_fit(&mut cox, &gidx, keys.len())?;
                Some((f, gidx, keys.len()))
            }
            None => None,
        };
        let ll_cox = res.ll;
        let (bhat, llhat) = match &frail {
            Some((f, _, _)) => (f.b.clone(), f.ll),
            None => (res.b.clone(), res.ll),
        };
        // variance
        let (hinv, _) = res.info.sym_inverse(1e-14);
        let mut nclust = None;
        let vact = if let Some((f, _, _)) = &frail { f.v.clone() } else { match &vce {
            Vce::Oim | Vce::Ols => hinv.clone(),
            _ => {
                let (_, _, _, stats) = cox.approx(&res.b);
                let sres = cox.score_residuals(&res.b, &stats);
                let scores: Vec<Vec<f64>> = sres.iter().zip(&cox.w).map(|(s, w)| s.iter().map(|x| x * w).collect()).collect();
                let cl = match &vce {
                    Vce::Cluster(v) => {
                        let c = util::col(ds, &varlist::resolve_name(ds, v)?)?;
                        Some(rows.iter().map(|&i| c[i]).collect::<Vec<f64>>())
                    }
                    _ => None,
                };
                let (vv, g) = est::sandwich(&hinv, &scores, cl.as_deref(), 1.0);
                let gf = g as f64;
                if cl.is_some() {
                    nclust = Some(g);
                }
                vv.scale(gf / (gf - 1.0))
            }
        } };
        // full coefficient vector: main columns then tvc
        let mut names: Vec<String> = Vec::new();
        let mut omitted = Vec::new();
        let mut is_base = Vec::new();
        let eq = |n: &str, e: &str| if tvc.is_empty() { n.to_string() } else { format!("{e}:{n}") };
        for j in 0..d.k() {
            names.push(eq(&d.names[j], "main"));
            omitted.push(d.omitted[j]);
            is_base.push(d.is_base[j]);
        }
        for v in &tvc {
            names.push(format!("tvc:{v}"));
            omitted.push(false);
            is_base.push(false);
        }
        let k = names.len();
        let mut pos = act.clone();
        pos.extend((0..tvc.len()).map(|j| d.k() + j));
        let mut b = vec![0.0; k];
        for (a, &j) in pos.iter().enumerate() {
            b[j] = bhat[a];
        }
        let v = est::expand_v(&vact, &pos, k);
        let robust = vce != Vce::Oim;
        let df_m = ka as f64;
        let (chi2, chi_label) = if robust || frail.is_some() {
            let bs: Vec<f64> = bhat.clone();
            let (vi, dropped) = vact.sym_inverse(1e-14);
            let w: f64 = (0..ka).map(|a| (0..ka).map(|c| bs[a] * vi[(a, c)] * bs[c]).sum::<f64>()).sum();
            (w, format!("Wald chi2({})", ka - dropped.len()))
        } else {
            (2.0 * (res.ll - ll0), format!("LR chi2({})", ka))
        };
        let p = if ka == 0 { SYSMISS } else { dist::chi2tail(df_m, chi2) };
        let nfail: f64 = (0..n).filter(|&r| cox.d[r]).map(|r| w[r]).sum();
        let risk: f64 = (0..n).map(|r| (cox.t[r] - cox.t0[r]) * w[r]).sum();
        let nsub = n_subjects(ds, &rows, &w);
        let nobs: f64 = if wt.as_ref().is_some_and(|(k, _)| k == "fweight") { w.iter().sum() } else { n as f64 };
        let level = util::level(cmd, st)?;
        let ll_label = if robust { "Log pseudolikelihood" } else { "Log likelihood" };
        let mut out = vec![st_header(ds), String::new()];
        out.extend(d.notes.iter().cloned());
        if let Some((f, _, _)) = &frail {
            out.push("Fitting comparison Cox model:".into());
            out.push(String::new());
            out.extend(iter_log(&res.iterations, ll_label, true));
            out.push("Refining estimates:".into());
            out.push(format!("Iteration 0:  {ll_label} = {}", fmt_ll(res.ll)));
            out.push(String::new());
            out.push("Estimating frailty variance:".into());
            out.push(String::new());
            out.extend(f.profile_log.iter().enumerate().map(|(i, l)| format!("Iteration {i}:  Log profile likelihood = {}", fmt_ll(*l))));
            out.push(String::new());
            out.push("Fitting final Cox model:".into());
            out.push(String::new());
        } else {
            out.extend(iter_log(&res.iterations, ll_label, true));
            out.push("Refining estimates:".into());
            out.push(format!("Iteration 0:  {ll_label} = {}", fmt_ll(res.ll)));
            out.push(String::new());
        }
        let method = match ties {
            Ties::Breslow => "Breslow method for ties",
            Ties::Efron => "Efron method for ties",
            Ties::ExactM => "exact marginal likelihood",
            Ties::ExactP => "exact partial likelihood",
        };
        let mut title = if strata.is_empty() { format!("Cox regression with {method}") } else { format!("Stratified Cox regression with {method}") };
        if let (Some(sv), Some((_, gidx, ng))) = (&shared, &frail) {
            let mut counts = vec![0usize; *ng];
            for &g in gidx {
                counts[g] += 1;
            }
            let (mn, mx) = (counts.iter().min().copied().unwrap_or(0), counts.iter().max().copied().unwrap_or(0));
            let vals = [gc(nsub, 12), gc(nfail, 12), gc(risk, 12)];
            let wv = vals.iter().map(|s| s.len()).max().unwrap_or(1);
            let r = |l: &str, v: String| format!("{:<17}= {:>10}", l, v);
            out.push(title.clone());
            out.push(format!("{:<48}{}", "Gamma shared frailty", r("Number of obs", gc(nobs, 10))));
            out.push(format!("{:<48}{}", format!("Group variable: {sv}"), r("Number of groups", gc(*ng as f64, 10))));
            out.push(String::new());
            out.push(format!("{:<48}Obs per group:", format!("No. of subjects = {:>wv$}", vals[0])));
            out.push(format!("{:<48}{}", format!("No. of failures = {:>wv$}", vals[1]), r("             min", mn.to_string())));
            out.push(format!("{:<48}{}", format!("Time at risk    = {:>wv$}", vals[2]), r("             avg", fmt_g(n as f64 / *ng as f64, 9))));
            out.push(format!("{:<48}{}", "", r("             max", mx.to_string())));
            out.push(format!("{:<48}{}", "", r(&chi_label, format!("{chi2:.2}"))));
            out.push(format!("{:<48}{}", format!("{ll_label} = {}", fmt_ll(llhat)), r("Prob > chi2", format!("{p:.4}"))));
            title.clear();
        } else {
            out.extend(st_model_header(&title, nsub, nfail, risk, nobs, &chi_label, if ka == 0 { SYSMISS } else { chi2 }, p, ll_label, llhat));
        }
        if let Some(g) = nclust {
            let cv = match &vce {
                Vce::Cluster(c) => c.clone(),
                _ => String::new(),
            };
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), cv)));
        }
        out.push(String::new());
        let nohr = cmd.has_opt("nohr", 3);
        if k > 0 {
            out.extend(est::coef_table(&est::TableSpec {
                depvar: "_t",
                names: &names,
                b: &b,
                v: &v,
                omitted: &omitted,
                is_base: &is_base,
                df: None,
                level,
                eform: if nohr { None } else { Some("Haz. ratio") },
                coef_title: "Coefficient",
            }));
            if let Some((f, _, _)) = &frail {
                let bar = out.pop().unwrap_or_default();
                let lw = bar.len().saturating_sub(66);
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
                out.push(format!("{:>lw$} | {:>10}  {:>9}", "theta", fmt_g(f.theta, 8), fmt_g(f.se_theta, 8)));
                out.push(bar);
                let chibar = (2.0 * (f.ll - ll_cox)).max(0.0);
                let pc = if chibar <= 0.0 { 1.0 } else { 0.5 * dist::chi2tail(1.0, chibar) };
                out.push(format!("LR test of theta=0: chibar2(01) = {:.2}{:>34}", chibar, format!("Prob >= chibar2 = {pc:.3}")));
                out.push("Note: Standard errors of hazard ratios are conditional on theta.".into());
            }
            if !tvc.is_empty() {
                out.push(format!("Note: Variables in tvc equation interacted with {}.", texp.clone().unwrap_or_else(|| "_t".into())));
            }
            if !strata.is_empty() {
                out.push(format!("{:>78}", format!("Stratified by {}", strata.join(" "))));
            }
        }
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let sample = d.mask.clone();
        let mut sample_mask = vec![false; ds.n_obs];
        for &i in &rows {
            sample_mask[i] = true;
        }
        let _ = sample;
        let e = st.begin_e();
        e.set("N", nobs);
        e.set("N_sub", nsub);
        e.set("N_fail", nfail);
        e.set("risk", risk);
        e.set("ll", llhat);
        if let Some((f, _, ng)) = &frail {
            e.set("theta", f.theta);
            e.set("se_theta", f.se_theta);
            e.set("ll_c", ll_cox);
            let chibar = (2.0 * (f.ll - ll_cox)).max(0.0);
            e.set("chi2_c", chibar);
            e.set("p_c", if chibar <= 0.0 { 1.0 } else { 0.5 * dist::chi2tail(1.0, chibar) });
            e.set("N_g", *ng as f64);
            e.set_str("shared", shared.clone().unwrap_or_default());
            e.set_str("distrib", "gamma");
        }
        e.set("ll_0", ll0);
        e.set("chi2", if ka == 0 { SYSMISS } else { chi2 });
        e.set("df_m", df_m);
        e.set("p", p);
        e.set("rank", ka as f64);
        e.set("k_eq_model", if ka == 0 { 0.0 } else { 1.0 });
        e.set("converged", res.converged as u8 as f64);
        e.set("ic", (res.iterations.len() - 1) as f64);
        if let Some(g) = nclust {
            e.set("N_clust", g as f64);
        }
        e.set_str("cmd", "cox");
        e.set_str("cmd2", "stcox");
        e.set_str("title", "Cox regression");
        e.set_str("depvar", "_t");
        e.set_str("t0", "_t0");
        e.set_str("ties", ties.name());
        e.set_str("method", ties.name());
        e.set_str("predict", "stcox_p");
        e.set_str("estat_cmd", "stcox_estat");
        e.set_str("chi2type", if robust { "Wald" } else { "LR" });
        e.set_str("properties", "b V");
        let (vcetype, vcelabel) = est::vce_label(&vce);
        e.set_str("vce", vcetype);
        if !vcelabel.is_empty() {
            e.set_str("vcetype", vcelabel);
        }
        if let Vce::Cluster(c) = &vce {
            e.set_str("clustvar", c.clone());
        }
        if !strata.is_empty() {
            e.set_str("strata", strata.join(" "));
        }
        if !tvc.is_empty() {
            e.set_str("tvc", tvc.join(" "));
            e.set_str("texp", texp.clone().unwrap_or_else(|| "_t".into()));
        }
        e.set_str("_output", text.clone());
        if k > 0 {
            e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
            e.set_mat("V", v.with_names(names.clone(), names));
        } else {
            e.set_mat("b", Matrix::zeros(1, 0));
            e.set_mat("V", Matrix::zeros(0, 0));
        }
        e.sample = Some(sample_mask);
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict after stcox
// ---------------------------------------------------------------------------

/// New-variable names of `predict [type] names` (stub* expands to k names).
pub fn predict_names(ds: &DataSet, main: &str, k: usize) -> Result<Vec<String>, String> {
    let mut words: Vec<&str> = main.split_whitespace().collect();
    if words.first().is_some_and(|w| VarType::from_type_str(w).is_some()) {
        words.remove(0);
    }
    if words.is_empty() {
        return Err(rc_err(100, "new variable name required"));
    }
    let names: Vec<String> = if words.len() == 1 && words[0].ends_with('*') {
        let stub = words[0].trim_end_matches('*');
        (1..=k).map(|j| format!("{stub}{j}")).collect()
    } else {
        words.iter().map(|s| s.to_string()).collect()
    };
    for n in &names {
        varlist::check_name(n)?;
        if ds.var_index.contains_key(n) {
            return Err(rc_err(110, format!("variable {n} already defined")));
        }
    }
    Ok(names)
}

fn add_var(ds: &mut DataSet, name: &str, vals: Vec<f64>, label: &str) -> Result<usize, String> {
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    let mut v = util::new_double(name, vals);
    v.var_type = VarType::Float;
    v.label = label.to_string();
    ds.add_variable(v).map_err(|e| rc_err(110, e))?;
    Ok(nmiss)
}

fn missing_msg(n: usize) -> String {
    if n == 0 { String::new() } else { format!("({} missing value{} generated)", util::commas(n), if n == 1 { "" } else { "s" }) }
}

/// Group subjects for per-subject residuals: map row -> last record flag and
/// sums over the subject (None when records are not aggregated).
fn subject_of(ds: &DataSet, rows: &[usize], t: &[f64]) -> Option<Vec<Vec<usize>>> {
    let id = st_id(ds)?;
    let c = util::col(ds, &id).ok()?;
    let mut m: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
    for (r, &i) in rows.iter().enumerate() {
        m.entry(c[i].to_bits()).or_default().push(r);
    }
    let mut v: Vec<Vec<usize>> = m.into_values().collect();
    for s in v.iter_mut() {
        s.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    }
    Some(v)
}

/// `predict` after stcox; None for other estimators.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("cox") {
        return Ok(None);
    }
    let known = ["hr", "xb", "stdp", "basesurv", "basechazard", "basehc", "csnell", "mgale", "deviance", "schoenfeld", "scaledsch", "esr", "dfbeta", "ldisplace", "lmax", "effects"];
    let stat = cmd
        .opts
        .iter()
        .map(|(n, _)| n.to_lowercase())
        .find_map(|n| known.iter().find(|k| k.starts_with(n.as_str()) && n.len() >= 2.min(k.len())).map(|k| k.to_string()))
        .unwrap_or_default();
    let partial = cmd.has_opt("partial", 4);
    let fit = fit_from_e(ds, st)?;
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let bm = st.e.matrices.get("b").cloned().unwrap();
    let kx = fit.cox.kx;
    let k = fit.cox.k();
    let mut msg = String::new();
    let stat = if stat.is_empty() {
        msg = "(option hr assumed; relative hazard)".into();
        "hr".to_string()
    } else {
        stat
    };
    let vector = matches!(stat.as_str(), "schoenfeld" | "scaledsch" | "esr" | "dfbeta");
    let names = predict_names(ds, &cmd.main, if vector { k } else { 1 })?;
    if vector && names.len() != k {
        return Err(rc_err(103, format!("{} new variables required", k)));
    }
    if !vector && names.len() != 1 {
        return Err(rc_err(103, "too many variables specified"));
    }
    let mut row_of = vec![usize::MAX; n];
    for (r, &i) in fit.rows.iter().enumerate() {
        row_of[i] = r;
    }
    let b = &fit.b;
    let main_b: Vec<f64> = fit.names.iter().zip(b).filter(|(nm, _)| !nm.starts_with("tvc:")).map(|(_, b)| *b).collect();
    let env = Env::with_store(ds, st);
    // xb on the original scale from the main equation
    let xb_at = |i: usize| -> Result<f64, String> {
        let mut s = 0.0;
        for (j, name) in fit.names.iter().enumerate() {
            if name.starts_with("tvc:") || b[j] == 0.0 {
                continue;
            }
            let x = crate::postest::coef_value(&env, name, i)?;
            if is_missing(x) {
                return Ok(SYSMISS);
            }
            s += x * b[j];
        }
        Ok(s)
    };
    let mut out_vals: Vec<Vec<f64>> = Vec::new();
    let label: String;
    // centred-scale estimates of the fitted model
    let cb: Vec<f64> = b.clone();
    let active_b: Vec<f64> = {
        // b over the columns used by the Cox data (omitted/base columns are 0)
        let mut v = Vec::new();
        for (j, nm) in fit.names.iter().enumerate() {
            if nm.starts_with("tvc:") {
                continue;
            }
            v.push(cb[j]);
        }
        v.extend(fit.names.iter().zip(&cb).filter(|(nm, _)| nm.starts_with("tvc:")).map(|(_, b)| *b));
        v
    };
    let _ = main_b;
    match stat.as_str() {
        "xb" | "hr" | "stdp" => {
            let mut vals = vec![SYSMISS; n];
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                let xb = xb_at(i)?;
                if is_missing(xb) {
                    continue;
                }
                vals[i] = match stat.as_str() {
                    "xb" => xb,
                    "hr" => xb.exp(),
                    _ => {
                        let x: Vec<f64> = fit.names.iter().map(|nm| if nm.starts_with("tvc:") { Ok(0.0) } else { crate::postest::coef_value(&env, nm, i) }).collect::<Result<_, _>>()?;
                        let mut s = 0.0;
                        for a in 0..x.len() {
                            for c in 0..x.len() {
                                s += x[a] * fit.v[(a, c)] * x[c];
                            }
                        }
                        s.max(0.0).sqrt()
                    }
                };
            }
            label = match stat.as_str() {
                "xb" => "Linear prediction".into(),
                "hr" => "Predicted hazard ratio".into(),
                _ => "S.E. of the linear prediction".into(),
            };
            out_vals.push(vals);
        }
        _ => {
            let cox = &fit.cox;
            let (_, _, info, stats) = cox.approx(&active_b);
            let base = cox.baseline(&active_b);
            // baseline functions at each record's exit time, per stratum
            let shift: f64 = cox.xmean.iter().zip(&active_b).map(|(m, b)| m * b).sum();
            let mut h0 = vec![0.0; cox.n()];
            let mut s0 = vec![1.0; cox.n()];
            let mut hc = vec![SYSMISS; cox.n()];
            let mut hcum_range = vec![0.0; cox.n()]; // baseline cumulative hazard over (t0, t]
            for (s, recs) in cox.srecs.iter().enumerate() {
                let gl = &cox.sgroups[s];
                let mut ch = Vec::with_capacity(gl.len());
                let mut cs = Vec::with_capacity(gl.len());
                let (mut a, mut p) = (0.0, 1.0);
                for &gi in gl {
                    a += base[gi].0;
                    p *= 1.0 - base[gi].1;
                    ch.push(a);
                    cs.push(p);
                }
                let taus: Vec<f64> = gl.iter().map(|&g| cox.groups[g].tau).collect();
                for &r in recs {
                    let hi = taus.partition_point(|&tau| tau <= cox.t[r]);
                    let lo = taus.partition_point(|&tau| tau <= cox.t0[r]);
                    h0[r] = if hi == 0 { 0.0 } else { ch[hi - 1] };
                    s0[r] = if hi == 0 { 1.0 } else { cs[hi - 1] };
                    let h_lo = if lo == 0 { 0.0 } else { ch[lo - 1] };
                    hcum_range[r] = h0[r] - h_lo;
                    if cox.d[r] && hi > 0 && taus[hi - 1] == cox.t[r] {
                        hc[r] = base[gl[hi - 1]].1;
                    }
                }
            }
            // relative hazard of each record (original scale, at its exit time)
            let rel: Vec<f64> = (0..cox.n()).map(|r| (cox.eta(&active_b, r, 0.0) + shift).exp()).collect();
            let tvc_note = cox.kz > 0;
            let mg: Vec<f64> = (0..cox.n()).map(|r| cox.d[r] as u8 as f64 - rel[r] * hcum_range[r]).collect();
            let subj = if partial { None } else { subject_of(ds, &fit.rows, &cox.t) };
            let per_record = |vals: &[f64]| -> Vec<f64> {
                let mut o = vec![SYSMISS; n];
                for (r, &i) in fit.rows.iter().enumerate() {
                    if mask[i] {
                        o[i] = vals[r];
                    }
                }
                o
            };
            let per_subject = |vals: &[f64]| -> Vec<f64> {
                match &subj {
                    None => per_record(vals),
                    Some(groups) => {
                        let mut o = vec![SYSMISS; n];
                        for g in groups {
                            let last = *g.last().unwrap();
                            let i = fit.rows[last];
                            if mask[i] {
                                o[i] = g.iter().map(|&r| vals[r]).sum();
                            }
                        }
                        o
                    }
                }
            };
            let _ = tvc_note;
            match stat.as_str() {
                "basesurv" => {
                    out_vals.push(per_record(&s0));
                    label = "Baseline survivor function".into();
                }
                "basechazard" => {
                    out_vals.push(per_record(&h0));
                    label = "Cumulative baseline hazard".into();
                }
                "basehc" => {
                    out_vals.push(per_record(&hc));
                    label = "Baseline hazard contribution".into();
                }
                "mgale" => {
                    out_vals.push(per_subject(&mg));
                    label = "Martingale".into();
                }
                "csnell" => {
                    let cs: Vec<f64> = (0..cox.n()).map(|r| cox.d[r] as u8 as f64 - mg[r]).collect();
                    out_vals.push(per_subject(&cs));
                    label = "Cox–Snell residual".into();
                }
                "deviance" => {
                    let m = per_subject(&mg);
                    let dsum = per_subject(&cox.d.iter().map(|&x| x as u8 as f64).collect::<Vec<_>>());
                    let dev: Vec<f64> = (0..n)
                        .map(|i| {
                            if is_missing(m[i]) {
                                return SYSMISS;
                            }
                            let dd = dsum[i];
                            let inner = if dd > 0.0 { m[i] + dd * (dd - m[i]).ln() } else { m[i] };
                            m[i].signum() * (-2.0 * inner).max(0.0).sqrt()
                        })
                        .collect();
                    out_vals.push(dev);
                    label = "Deviance residual".into();
                }
                "schoenfeld" | "scaledsch" => {
                    let sch = cox.schoenfeld(&stats);
                    let (vinv, _) = info.sym_inverse(1e-14);
                    let nf: f64 = cox.d.iter().zip(&cox.w).filter(|(d, _)| **d).map(|(_, w)| *w).sum();
                    for j in 0..k {
                        let mut col = vec![SYSMISS; n];
                        for (r, &i) in fit.rows.iter().enumerate() {
                            if !mask[i] {
                                continue;
                            }
                            if let Some(s) = &sch[r] {
                                col[i] = if stat == "schoenfeld" {
                                    s[j]
                                } else {
                                    active_b[j] + nf * (0..k).map(|c| vinv[(j, c)] * s[c]).sum::<f64>()
                                };
                            }
                        }
                        out_vals.push(col);
                    }
                    label = if stat == "schoenfeld" { "Schoenfeld" } else { "scaled Schoenfeld" }.into();
                }
                "esr" | "dfbeta" => {
                    let sr = cox.score_residuals(&active_b, &stats);
                    let (vinv, _) = info.sym_inverse(1e-14);
                    for j in 0..k {
                        let vals: Vec<f64> = (0..cox.n())
                            .map(|r| if stat == "esr" { sr[r][j] } else { (0..k).map(|c| vinv[(j, c)] * sr[r][c]).sum::<f64>() })
                            .collect();
                        out_vals.push(per_subject(&vals));
                    }
                    label = if stat == "esr" { "Efficient score" } else { "DFBETA" }.into();
                }
                s => return Err(rc_err(198, format!("option {s} not allowed"))),
            }
        }
    }
    drop(env);
    let _ = kx;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    let covnames: Vec<String> = bm.colnames.clone();
    for (j, (name, vals)) in names.iter().zip(out_vals).enumerate() {
        let lab = if vector { format!("{label} - {}", covnames.get(j).cloned().unwrap_or_default()) } else { label.clone() };
        let nm = add_var(ds, name, vals, &lab)?;
        if !vector || j == 0 {
            let m = missing_msg(nm);
            if !m.is_empty() {
                lines.push(m);
            }
        }
    }
    Ok(Some(lines.join("\n")))
}

// ---------------------------------------------------------------------------
// estat phtest / concordance
// ---------------------------------------------------------------------------

/// `estat` subcommands after stcox; None when not applicable.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("cox") {
        return None;
    }
    match sub {
        "phtest" | "phtes" | "phte" | "pht" => Some(phtest(ds, st, cmd)),
        "concordance" | "concord" | "conc" => Some(concordance(ds, st, cmd)),
        _ => None,
    }
}

fn phtest(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let fit = fit_from_e(ds, st)?;
    let cox = &fit.cox;
    if cox.kz > 0 {
        return Err(rc_err(198, "estat phtest not allowed after stcox with tvc()"));
    }
    let b = fit.b.clone();
    let (_, _, info, stats) = cox.approx(&b);
    let (vinv, _) = info.sym_inverse(1e-14);
    let sch = cox.schoenfeld(&stats);
    let k = cox.k();
    // failures in time order
    let mut fails: Vec<usize> = (0..cox.n()).filter(|&r| cox.d[r]).collect();
    fails.sort_by(|&a, &c| cox.t[a].total_cmp(&cox.t[c]));
    let tf = cmd.opt("time", 2).map(|s| s.trim().to_lowercase());
    let kind = if cmd.has_opt("log", 3) {
        "log"
    } else if cmd.has_opt("km", 2) {
        "km"
    } else if cmd.has_opt("rank", 2) {
        "rank"
    } else {
        "identity"
    };
    let times: Vec<f64> = fails.iter().map(|&r| cox.t[r]).collect();
    let g: Vec<f64> = match (tf.as_deref(), kind) {
        (Some(v), _) => {
            let name = varlist::resolve_name(ds, v)?;
            let c = util::col(ds, &name)?;
            fails.iter().map(|&r| c[fit.rows[r]]).collect()
        }
        (None, "log") => times.iter().map(|t| t.ln()).collect(),
        (None, "rank") => {
            // average ranks of the failure times
            let m = times.len();
            let mut out = vec![0.0; m];
            let mut a = 0;
            while a < m {
                let mut e = a;
                while e + 1 < m && times[e + 1] == times[a] {
                    e += 1;
                }
                for o in out.iter_mut().take(e + 1).skip(a) {
                    *o = (a + e) as f64 / 2.0 + 1.0;
                }
                a = e + 1;
            }
            out
        }
        (None, "km") => {
            // 1 - KM just before each failure time (pooled over strata)
            let obs: Vec<crate::survival::StObs> = (0..cox.n()).map(|r| crate::survival::StObs { t0: cox.t0[r], t: cox.t[r], d: cox.d[r], row: r, w: cox.w[r] }).collect();
            let steps = crate::stsum::km(&obs);
            times
                .iter()
                .map(|&t| {
                    let k = steps.partition_point(|s| s.t < t);
                    1.0 - if k == 0 { 1.0 } else { steps[k - 1].s }
                })
                .collect()
        }
        _ => times.clone(),
    };
    let nf = fails.len() as f64;
    let gbar = g.iter().sum::<f64>() / nf;
    let gc: Vec<f64> = g.iter().map(|x| x - gbar).collect();
    let sgg: f64 = gc.iter().map(|x| x * x).sum();
    // u = sum (g - gbar) r_k
    let mut u = vec![0.0; k];
    for (q, &r) in fails.iter().enumerate() {
        let s = sch[r].as_ref().unwrap();
        for a in 0..k {
            u[a] += gc[q] * s[a];
        }
    }
    let vu: Vec<f64> = (0..k).map(|a| (0..k).map(|c| vinv[(a, c)] * u[c]).sum()).collect();
    let global = nf * u.iter().zip(&vu).map(|(a, b)| a * b).sum::<f64>() / sgg;
    let df = k as f64;
    let pg = dist::chi2tail(df, global);
    let names: Vec<String> = fit.names.clone();
    let tlabel = match (tf.as_deref(), kind) {
        (Some(v), _) => v.to_string(),
        (None, "log") => "Log(t)".to_string(),
        (None, "km") => "1 - KM".to_string(),
        (None, "rank") => "Rank(t)".to_string(),
        _ => "Analysis time".to_string(),
    };
    let detail = cmd.has_opt("detail", 1);
    let mut out = vec!["Test of proportional-hazards assumption".to_string(), String::new(), format!("Time function: {tlabel}")];
    let bar = "-".repeat(56);
    out.push(bar.clone());
    let mut r = Results::new();
    let mut phm = Matrix::zeros(k, 4);
    if detail {
        out.push(format!("{:>12} | {:>9} {:>8} {:>8} {:>12}", "", "rho", "chi2", "df", "Prob>chi2"));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(42)));
        for j in 0..k {
            if fit.v[(j, j)] == 0.0 {
                continue;
            }
            let chi = nf * vu[j] * vu[j] / (vinv[(j, j)] * sgg);
            // correlation of the scaled residual with g
            let ss: Vec<f64> = fails.iter().map(|&rr| b[j] + nf * (0..k).map(|c| vinv[(j, c)] * sch[rr].as_ref().unwrap()[c]).sum::<f64>()).collect();
            let ms = ss.iter().sum::<f64>() / nf;
            let sxy: f64 = ss.iter().zip(&gc).map(|(s, g)| (s - ms) * g).sum();
            let sxx: f64 = ss.iter().map(|s| (s - ms).powi(2)).sum();
            let rho = sxy / (sxx * sgg).sqrt();
            let p = dist::chi2tail(1.0, chi);
            out.push(format!("{:>12} | {:>9.5} {:>8.2} {:>8} {:>12.4}", crate::summarize::abbrev(&names[j], 12), rho, chi, 1, p));
            phm[(j, 0)] = rho;
            phm[(j, 1)] = chi;
            phm[(j, 2)] = 1.0;
            phm[(j, 3)] = p;
        }
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(42)));
        out.push(format!("{:>12} | {:>9} {:>8.2} {:>8} {:>12.4}", "Global test", "", global, df, pg));
        r.set_mat("phtest", phm.with_names(names.clone(), vec!["rho".into(), "chi2".into(), "df".into(), "p".into()]));
    } else {
        out.push(format!("{:>12} | {:>18} {:>8} {:>12}", "", "chi2", "df", "Prob>chi2"));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(42)));
        out.push(format!("{:>12} | {:>18.2} {:>8} {:>12.4}", "Global test", global, df, pg));
    }
    out.push(bar);
    if let Some(p) = cmd.opt("plot", 2) {
        let name = p.trim();
        let j = names.iter().position(|n| n == name).ok_or_else(|| rc_err(198, format!("{name} not found in model")))?;
        let pts: Vec<(f64, f64)> = fails.iter().zip(&g).map(|(&rr, &gv)| (gv, b[j] + nf * (0..k).map(|c| vinv[(j, c)] * sch[rr].as_ref().unwrap()[c]).sum::<f64>())).collect();
        let mut sorted = pts.clone();
        sorted.sort_by(|a, c| a.0.total_cmp(&c.0));
        // running-mean smooth (bandwidth 0.8 like lowess default)
        let m = sorted.len();
        let bw = ((0.8 * m as f64) as usize).max(1);
        let smooth: Vec<(f64, f64)> = (0..m)
            .map(|q| {
                let lo = q.saturating_sub(bw / 2);
                let hi = (q + bw / 2 + 1).min(m);
                (sorted[q].0, sorted[lo..hi].iter().map(|p| p.1).sum::<f64>() / (hi - lo) as f64)
            })
            .collect();
        let mut gph = Graph { title: "Test of PH assumption".into(), xtitle: tlabel.clone(), ytitle: format!("Scaled Schoenfeld – {name}"), ..Default::default() };
        gph.layers.push(Layer { kind: LayerKind::Scatter, points: pts, label: "Scaled Schoenfeld".into(), ..Default::default() });
        gph.layers.push(Layer { kind: LayerKind::Line, points: smooth, label: "Smoothed".into(), ..Default::default() });
        gph.legend = false;
        let extra = graph::finish(&mut gph, cmd, st)?;
        if !extra.is_empty() {
            out.push(extra);
        }
    }
    r.set("chi2", global);
    r.set("df", df);
    r.set("p", pg);
    st.r = r;
    Ok(out.join("\n"))
}

fn concordance(ds: &mut DataSet, st: &mut Store, _cmd: &Command) -> Result<String, String> {
    let fit = fit_from_e(ds, st)?;
    let cox = &fit.cox;
    let b = fit.b.clone();
    let shift: f64 = cox.xmean.iter().zip(&b).map(|(m, b)| m * b).sum();
    let hr: Vec<f64> = (0..cox.n()).map(|r| (cox.eta(&b, r, 0.0) + shift).exp()).collect();
    let n = cox.n();
    let (mut pairs, mut agree, mut tied) = (0.0f64, 0.0f64, 0.0f64);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &c| cox.t[a].total_cmp(&cox.t[c]));
    for a in 0..n {
        let i = idx[a];
        if !cox.d[i] {
            continue;
        }
        for &j in &idx[a + 1..] {
            // j has a longer time, or the same time and is censored
            if cox.t[j] == cox.t[i] && cox.d[j] {
                continue;
            }
            if cox.strata[i] != cox.strata[j] {
                continue;
            }
            let wgt = cox.w[i] * cox.w[j];
            pairs += wgt;
            if (hr[i] - hr[j]).abs() <= 1e-12 * hr[i].abs().max(hr[j].abs()) {
                tied += wgt;
            } else if hr[i] > hr[j] {
                agree += wgt;
            }
        }
    }
    let c = (agree + tied / 2.0) / pairs;
    let somers = 2.0 * c - 1.0;
    let mut out = vec!["Harrell's C concordance statistic".to_string(), String::new()];
    out.push(st_header(ds));
    out.push(String::new());
    out.push(format!("  Number of subjects (N)              = {:>9}", gc(n_subjects(ds, &fit.rows, &cox.w), 9)));
    out.push(format!("  Number of comparison pairs (P)      = {:>9}", gc(pairs, 9)));
    out.push(format!("  Number of orderings as expected (E) = {:>9}", gc(agree, 9)));
    out.push(format!("  Number of tied predictions (T)      = {:>9}", gc(tied, 9)));
    out.push(String::new());
    out.push(format!("{:>30}Harrell's C = (E + T/2) / P = {}", "", fmt_g(c, 8)));
    out.push(format!("{:>48}Somers' D = {}", "", fmt_g(somers, 8)));
    let mut r = Results::new();
    r.set("N", n as f64);
    r.set("n_P", pairs);
    r.set("n_E", agree);
    r.set("n_T", tied);
    r.set("C", c);
    r.set("D", somers);
    st.r = r;
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// stcurve
// ---------------------------------------------------------------------------

/// Parse `var = # var = #` into (name, value) pairs.
pub fn parse_at(ds: &DataSet, text: &str) -> Result<Vec<(String, f64)>, String> {
    let mut out = Vec::new();
    let t = text.replace('=', " = ");
    let toks: Vec<&str> = t.split_whitespace().collect();
    let mut k = 0;
    while k < toks.len() {
        let name = varlist::resolve_name(ds, toks[k])?;
        if toks.get(k + 1) != Some(&"=") {
            return Err(rc_err(198, "at(): varname = # expected"));
        }
        let v: f64 = toks.get(k + 2).and_then(|s| s.parse().ok()).ok_or_else(|| rc_err(198, "at(): number expected"))?;
        out.push((name, v));
        k += 3;
    }
    Ok(out)
}

/// Covariate values for a curve: sample means overridden by at().
pub fn curve_covariates(ds: &DataSet, st: &Store, names: &[String], at: &[(String, f64)]) -> Result<Vec<f64>, String> {
    let sample = st.e.sample.clone().unwrap_or_default();
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    let env = Env::with_store(ds, st);
    let mut out = Vec::new();
    for name in names {
        let bare = name.rsplit(':').next().unwrap_or(name);
        if bare == "_cons" || bare.starts_with('/') || name.starts_with("/:") {
            out.push(1.0);
            continue;
        }
        if bare.starts_with("_S") {
            // stratum indicators of streg: the base stratum
            out.push(0.0);
            continue;
        }
        // factor level: "2.drug" with at(drug=2) -> 1/0
        let fixed = at.iter().find_map(|(v, x)| {
            if bare == v {
                Some(*x)
            } else if let Some((lv, var)) = bare.split_once('.') {
                let lv = lv.trim_end_matches(['b', 'o', 'n']);
                if var == v { lv.parse::<f64>().ok().map(|l| if l == *x { 1.0 } else { 0.0 }) } else { None }
            } else {
                None
            }
        });
        if let Some(x) = fixed {
            out.push(x);
            continue;
        }
        let vals: Vec<f64> = rows.iter().map(|&i| crate::postest::coef_value(&env, name, i)).collect::<Result<_, _>>()?;
        out.push(vals.iter().sum::<f64>() / vals.len().max(1) as f64);
    }
    Ok(out)
}

pub struct Stcurve;

impl CommandHandler for Stcurve {
    fn name(&self) -> &'static str {
        "stcurve"
    }

    /// stcurve, {survival|hazard|cumhaz|failure} [at(...) at1(...) at2(...) range(# #)]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        let what = if cmd.has_opt("hazard", 3) {
            "hazard"
        } else if cmd.has_opt("cumhaz", 3) {
            "cumhaz"
        } else if cmd.has_opt("failure", 3) {
            "failure"
        } else if cmd.has_opt("survival", 3) {
            "survival"
        } else {
            return Err(rc_err(198, "must specify survival, hazard, cumhaz or failure"));
        };
        let mut ats: Vec<(String, Vec<(String, f64)>)> = Vec::new();
        if let Some(a) = cmd.opt("at", 2) {
            ats.push((a.trim().to_string(), parse_at(ds, a)?));
        }
        for k in 1..=20 {
            if let Some(a) = cmd.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case(&format!("at{k}"))).and_then(|(_, a)| a.clone()) {
                ats.push((a.trim().to_string(), parse_at(ds, &a)?));
            }
        }
        if ats.is_empty() {
            ats.push((String::new(), Vec::new()));
        }
        let names = st.e.matrices.get("b").map(|b| b.colnames.clone()).unwrap_or_default();
        let mut gph = Graph { xtitle: "Analysis time".into(), ..Default::default() };
        let mut curves: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
        match ecmd.as_str() {
            "cox" => {
                let fit = fit_from_e(ds, st)?;
                let cox = &fit.cox;
                if cox.kz > 0 {
                    return Err(rc_err(198, "stcurve not allowed after stcox with tvc()"));
                }
                let b = fit.b.clone();
                let base = cox.baseline(&b);
                gph.title = "Cox proportional hazards regression".into();
                for (label, at) in &ats {
                    let xv = curve_covariates(ds, st, &names, at)?;
                    let xb: f64 = xv.iter().zip(&b).map(|(x, b)| x * b).sum();
                    let hr = xb.exp();
                    for (s, gl) in cox.sgroups.iter().enumerate() {
                        let (mut h, mut sv) = (0.0, 1.0);
                        let mut pts = vec![(0.0, if what == "survival" { 1.0 } else { 0.0 })];
                        let mut haz = Vec::new();
                        for &gi in gl {
                            let tau = cox.groups[gi].tau;
                            let (prev_h, prev_s) = (h, sv);
                            h += base[gi].0;
                            sv *= 1.0 - base[gi].1;
                            haz.push((tau, base[gi].0 * hr));
                            match what {
                                "survival" => {
                                    pts.push((tau, prev_s.powf(hr)));
                                    pts.push((tau, sv.powf(hr)));
                                }
                                "failure" => {
                                    pts.push((tau, 1.0 - prev_s.powf(hr)));
                                    pts.push((tau, 1.0 - sv.powf(hr)));
                                }
                                _ => {
                                    pts.push((tau, prev_h * hr));
                                    pts.push((tau, h * hr));
                                }
                            }
                        }
                        let tmax = cox.srecs[s].iter().map(|&r| cox.t[r]).fold(0.0, f64::max);
                        if let Some(&(_, y)) = pts.last() {
                            pts.push((tmax, y));
                        }
                        if what == "hazard" {
                            pts = kernel_hazard(&haz, tmax, cmd)?;
                        }
                        let lab = match (cox.srecs.len() > 1, label.is_empty()) {
                            (true, true) => format!("stratum {}", s + 1),
                            (true, false) => format!("{label}, stratum {}", s + 1),
                            (false, _) => label.clone(),
                        };
                        curves.push((lab, pts));
                    }
                }
            }
            _ if st.e.macros.get("cmd2").is_some_and(|c| c == "streg") || ecmd.starts_with("ereg") || ecmd.starts_with("weib") => {
                gph.title = st.e.macros.get("title").cloned().unwrap_or_default();
                for (label, at) in &ats {
                    let pts = crate::streg::curve(ds, st, what, at, cmd)?;
                    curves.push((label.clone(), pts));
                }
            }
            _ => return Err(rc_err(301, "last estimates not found (stcurve requires stcox or streg)")),
        }
        gph.ytitle = match what {
            "survival" => "Survival",
            "failure" => "Failure",
            "cumhaz" => "Cumulative hazard",
            _ => "Smoothed hazard function",
        }
        .into();
        if gph.title.is_empty() {
            gph.title = "Survival model".into();
        }
        let multi = curves.len() > 1;
        for (label, pts) in curves {
            gph.layers.push(Layer { kind: LayerKind::Line, points: pts, label: if label.is_empty() { gph.ytitle.clone() } else { label }, ..Default::default() });
        }
        gph.legend = multi;
        let extra = graph::finish(&mut gph, cmd, st)?;
        Ok(extra)
    }
}

/// Epanechnikov-kernel smooth of hazard contributions (time, increment).
fn kernel_hazard(inc: &[(f64, f64)], tmax: f64, cmd: &Command) -> Result<Vec<(f64, f64)>, String> {
    let times: Vec<f64> = inc.iter().map(|p| p.0).collect();
    let m = times.len().max(1) as f64;
    let mean = times.iter().sum::<f64>() / m;
    let sd = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (m - 1.0).max(1.0)).sqrt();
    let bw = match util::opt_num(cmd, "width", 1)? {
        Some(w) => w,
        None => (0.9 * sd * m.powf(-0.2)).max(1e-8),
    };
    let tmin = times.first().copied().unwrap_or(0.0);
    let hi = times.last().copied().unwrap_or(tmax);
    let (lo, hi) = (tmin + bw, (hi - bw).max(tmin + bw));
    let grid: Vec<f64> = (0..=100).map(|q| lo + (hi - lo) * q as f64 / 100.0).collect();
    Ok(grid
        .into_iter()
        .map(|t| {
            let h: f64 = inc
                .iter()
                .map(|(tj, dh)| {
                    let u = (t - tj) / bw;
                    if u.abs() < 1.0 { 0.75 * (1.0 - u * u) / bw * dh } else { 0.0 }
                })
                .sum();
            (t, h)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marginal_integral_matches_closed_form() {
        // two failures: 1 - 1/(1+a) - 1/(1+b) + 1/(1+a+b)
        let (a, b) = (0.3, 1.7);
        let exact = 1.0 - 1.0 / (1.0 + a) - 1.0 / (1.0 + b) + 1.0 / (1.0 + a + b);
        assert!((marginal_integral(&[a, b]) - exact).abs() < 1e-10);
    }

    #[test]
    fn legendre_nodes_integrate_polynomials() {
        let (x, w) = gauss_legendre(10);
        let s: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(8)).sum();
        assert!((s - 2.0 / 9.0).abs() < 1e-13);
    }
}
