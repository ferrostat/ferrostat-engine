//! Parametric survival regression `streg` (exponential, Weibull, Gompertz,
//! log-logistic, lognormal and generalized gamma), in proportional-hazards
//! or accelerated-failure-time metric, with modelled ancillary parameters,
//! strata (as indicator shifts of the ancillary parameter) and gamma or
//! inverse-Gaussian frailty (unshared or shared). Delayed entry conditions
//! on survival to `_t0`.
//!
//! Each record contributes `d ln h(t) - H(t) + H(t0)`; with frailty the
//! cumulative hazards of a group are integrated over the frailty
//! distribution (Gutierrez 2002, 2: 22-44, gives the
//! closed forms). Derivatives with respect to each record's linear
//! predictors are taken by Richardson-extrapolated central differences
//! and chained to the coefficients analytically.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::est::{self, DesignSpec, Likelihood, Vce};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::stcox::{curve_covariates, n_subjects, predict_names, st_model_header, with_st_if};
use crate::store::Store;
use crate::survival::{require_st, st_header, st_id, st_weight};
use crate::util;
use crate::varlist;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dist {
    Exponential,
    Weibull,
    Gompertz,
    Loglogistic,
    Lognormal,
    Ggamma,
}

impl Dist {
    fn parse(s: &str) -> Result<Dist, String> {
        let s = s.trim().to_lowercase();
        let pick = |full: &str, min: usize| s.len() >= min && full.starts_with(s.as_str());
        Ok(if pick("exponential", 1) {
            Dist::Exponential
        } else if pick("weibull", 1) {
            Dist::Weibull
        } else if pick("gompertz", 3) {
            Dist::Gompertz
        } else if pick("loglogistic", 4) || pick("llogistic", 2) {
            Dist::Loglogistic
        } else if pick("lognormal", 4) || pick("lnormal", 2) {
            Dist::Lognormal
        } else if pick("ggamma", 2) || pick("gamma", 3) {
            Dist::Ggamma
        } else {
            return Err(rc_err(198, format!("distribution({s}) invalid")));
        })
    }
    /// Names of the ancillary parameters.
    fn anc(self) -> &'static [&'static str] {
        match self {
            Dist::Exponential => &[],
            Dist::Weibull => &["ln_p"],
            Dist::Gompertz => &["gamma"],
            Dist::Loglogistic => &["ln_gam"],
            Dist::Lognormal => &["ln_sig"],
            Dist::Ggamma => &["ln_sig", "kappa"],
        }
    }
    fn label(self) -> &'static str {
        match self {
            Dist::Exponential => "Exponential",
            Dist::Weibull => "Weibull",
            Dist::Gompertz => "Gompertz",
            Dist::Loglogistic => "Loglogistic",
            Dist::Lognormal => "Lognormal",
            Dist::Ggamma => "Generalized gamma",
        }
    }
    fn ecmd(self) -> &'static str {
        match self {
            Dist::Exponential => "ereg",
            Dist::Weibull => "weibull",
            Dist::Gompertz => "gompertz",
            Dist::Loglogistic => "llogistic",
            Dist::Lognormal => "lnormal",
            Dist::Ggamma => "gamma",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Dist::Exponential => "exponential",
            Dist::Weibull => "weibull",
            Dist::Gompertz => "gompertz",
            Dist::Loglogistic => "loglogistic",
            Dist::Lognormal => "lognormal",
            Dist::Ggamma => "ggamma",
        }
    }
    fn from_ecmd(s: &str) -> Option<Dist> {
        Some(match s {
            "ereg" => Dist::Exponential,
            "weibull" => Dist::Weibull,
            "gompertz" => Dist::Gompertz,
            "llogistic" => Dist::Loglogistic,
            "lnormal" => Dist::Lognormal,
            "gamma" => Dist::Ggamma,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Frailty {
    None,
    Gamma,
    InvGauss,
}

/// ln Phi(z) accurate in the lower tail.
fn ln_phi(z: f64) -> f64 {
    if z > -30.0 {
        dist::normal(z).ln()
    } else {
        dist::lnnormalden(z) - (-z).ln() + (1.0 - 1.0 / (z * z) + 3.0 / z.powi(4)).ln()
    }
}

fn softplus(u: f64) -> f64 {
    if u > 30.0 { u + (-u).exp().ln_1p() } else { u.exp().ln_1p() }
}

/// (ln h(t), H(t)) of one distribution at linear predictor eta and
/// ancillary values `anc`; `aft` selects the time metric.
pub fn lnh_cumh(dist: Dist, aft: bool, t: f64, eta: f64, anc: &[f64]) -> (f64, f64) {
    let lt = t.ln();
    match dist {
        Dist::Exponential => {
            let e = if aft { -eta } else { eta };
            (e, e.exp() * t)
        }
        Dist::Weibull => {
            let p = anc[0].exp();
            if aft {
                (anc[0] - p * eta + (p - 1.0) * lt, (p * (lt - eta)).exp())
            } else {
                (eta + anc[0] + (p - 1.0) * lt, (eta + p * lt).exp())
            }
        }
        Dist::Gompertz => {
            let g = anc[0];
            let ratio = if (g * t).abs() < 1e-10 { t * (1.0 + g * t / 2.0) } else { (g * t).exp_m1() / g };
            (eta + g * t, eta.exp() * ratio)
        }
        Dist::Loglogistic => {
            let gam = anc[0].exp();
            let u = (lt - eta) / gam;
            (u - lt - anc[0] - softplus(u), softplus(u))
        }
        Dist::Lognormal => {
            let s = anc[0].exp();
            let z = (lt - eta) / s;
            let lns = ln_phi(-z);
            (dist::lnnormalden(z) - anc[0] - lt - lns, -lns)
        }
        Dist::Ggamma => {
            let (lsig, kappa) = (anc[0], anc[1]);
            let s = lsig.exp();
            if kappa.abs() < 1e-7 {
                return lnh_cumh(Dist::Lognormal, true, t, eta, &[lsig]);
            }
            let gam = 1.0 / (kappa * kappa);
            let z = kappa.signum() * (lt - eta) / s;
            let u = gam * (kappa.abs() * z).exp();
            if !u.is_finite() || !gam.is_finite() || u <= 0.0 {
                return (f64::NAN, f64::NAN);
            }
            let lnf = gam * gam.ln() - lsig - lt - 0.5 * gam.ln() - dist::lngamma(gam) + z * gam.sqrt() - u;
            let sv = if kappa > 0.0 { dist::gammaptail(gam, u) } else { dist::gammap(gam, u) };
            let lns = if sv > 0.0 { sv.ln() } else { -745.0 };
            (lnf - lns, -lns)
        }
    }
}

/// Value, gradient and Hessian of a vector function of a few arguments by
/// Richardson-extrapolated central differences.
fn num_derivs<const M: usize>(f: &dyn Fn(&[f64]) -> [f64; M], x: &[f64]) -> ([f64; M], Vec<[f64; M]>, Vec<[f64; M]>) {
    let m = x.len();
    let f0 = f(x);
    let mut g = vec![[0.0; M]; m];
    let mut h = vec![[0.0; M]; m * m];
    let at = |d: &[(usize, f64)]| {
        let mut v = x.to_vec();
        for &(j, s) in d {
            v[j] += s;
        }
        f(&v)
    };
    let step: Vec<f64> = x.iter().map(|v| 2e-4 * v.abs().max(1.0)).collect();
    for j in 0..m {
        let hj = step[j];
        let (p1, m1, p2, m2) = (at(&[(j, hj)]), at(&[(j, -hj)]), at(&[(j, hj / 2.0)]), at(&[(j, -hj / 2.0)]));
        for q in 0..M {
            let d1 = (p1[q] - m1[q]) / (2.0 * hj);
            let d2 = (p2[q] - m2[q]) / hj;
            g[j][q] = (4.0 * d2 - d1) / 3.0;
            let s1 = (p1[q] - 2.0 * f0[q] + m1[q]) / (hj * hj);
            let s2 = (p2[q] - 2.0 * f0[q] + m2[q]) / (hj * hj / 4.0);
            h[j * m + j][q] = (4.0 * s2 - s1) / 3.0;
        }
        for c in 0..j {
            let hc = step[c];
            let cross = |a: f64| {
                let (pp, pm, mp, mm) = (at(&[(j, a * hj), (c, a * hc)]), at(&[(j, a * hj), (c, -a * hc)]), at(&[(j, -a * hj), (c, a * hc)]), at(&[(j, -a * hj), (c, -a * hc)]));
                let mut o = [0.0; M];
                for q in 0..M {
                    o[q] = (pp[q] - pm[q] - mp[q] + mm[q]) / (4.0 * a * a * hj * hc);
                }
                o
            };
            let (c1, c2) = (cross(1.0), cross(0.5));
            for q in 0..M {
                let v = (4.0 * c2[q] - c1[q]) / 3.0;
                h[j * m + c][q] = v;
                h[c * m + j][q] = v;
            }
        }
    }
    (f0, g, h)
}

/// Group term of the frailty likelihood: F(A, B, ln theta) with D failures,
/// A = sum H(t), B = sum H(t0).
fn frailty_f(fr: Frailty, dcount: f64, a: f64, b: f64, lnth: f64) -> f64 {
    let th = lnth.exp();
    match fr {
        Frailty::None => b - a,
        Frailty::Gamma => {
            let it = 1.0 / th;
            dcount * lnth + dist::lngamma(it + dcount) - dist::lngamma(it) - (it + dcount) * (th * a).ln_1p() + it * (th * b).ln_1p()
        }
        Frailty::InvGauss => {
            let ua = (1.0 + 2.0 * th * a).sqrt();
            let ub = (1.0 + 2.0 * th * b).sqrt();
            let base = (1.0 - ua) / th - (1.0 - ub) / th;
            if dcount < 0.5 {
                return base;
            }
            let n = dcount.round() as usize;
            // ln sum_k c_{n,k} (theta/2)^k ua^-(n+k)
            let terms: Vec<f64> = (0..n)
                .map(|k| {
                    let kf = k as f64;
                    dist::lnfactorial((n - 1 + k) as f64) - dist::lnfactorial(kf) - dist::lnfactorial((n - 1 - k) as f64) + kf * (th / 2.0).ln() - (n as f64 + kf) * ua.ln()
                })
                .collect();
            let mx = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            base + mx + terms.iter().map(|t| (t - mx).exp()).sum::<f64>().ln()
        }
    }
}

/// Design of one equation over the sample records.
#[derive(Clone)]
struct Eq {
    name: String,
    names: Vec<String>,
    cols: Vec<Vec<f64>>,
}

struct Model<'a> {
    dist: Dist,
    aft: bool,
    eqs: &'a [Eq],
    t: &'a [f64],
    t0: &'a [f64],
    d: &'a [f64],
    w: &'a [f64],
    frailty: Frailty,
    /// frailty groups (records); one per record when unshared
    groups: Vec<Vec<usize>>,
}

impl Model<'_> {
    fn nparam(&self) -> usize {
        self.eqs.iter().map(|e| e.cols.len()).sum::<usize>() + if self.frailty != Frailty::None { 1 } else { 0 }
    }
    fn args(&self, b: &[f64], i: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.eqs.len());
        let mut off = 0;
        for e in self.eqs {
            out.push(e.cols.iter().enumerate().map(|(j, c)| c[i] * b[off + j]).sum());
            off += e.cols.len();
        }
        out
    }
    /// [ln h(t), H(t), H(t0)] of record i at linear predictors a.
    fn terms(&self, i: usize, a: &[f64]) -> [f64; 3] {
        let (lh, h1) = lnh_cumh(self.dist, self.aft, self.t[i], a[0], &a[1..]);
        let h0 = if self.t0[i] > 0.0 { lnh_cumh(self.dist, self.aft, self.t0[i], a[0], &a[1..]).1 } else { 0.0 };
        [lh, h1, h0]
    }
    /// Map a derivative with respect to equation `e` of record i into
    /// parameter positions.
    fn xvec(&self, i: usize) -> Vec<(usize, usize, f64)> {
        // (param index, equation, x value)
        let mut v = Vec::new();
        let mut off = 0;
        for (e, eq) in self.eqs.iter().enumerate() {
            for (j, c) in eq.cols.iter().enumerate() {
                if c[i] != 0.0 {
                    v.push((off + j, e, c[i]));
                }
            }
            off += eq.cols.len();
        }
        v
    }
}

impl Likelihood for Model<'_> {
    fn k(&self) -> usize {
        self.nparam()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.nparam();
        let m = self.eqs.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let mut scores = if need_scores { vec![vec![0.0; k]; self.t.len()] } else { Vec::new() };
        let lnth = if self.frailty != Frailty::None { b[k - 1] } else { 0.0 };
        for grp in &self.groups {
            // per-record terms and derivatives
            let mut a_sum = 0.0;
            let mut b_sum = 0.0;
            let mut dcount = 0.0;
            let mut per = Vec::with_capacity(grp.len());
            for &i in grp {
                let a = self.args(b, i);
                let f = |x: &[f64]| self.terms(i, x);
                let (v, dg, dh) = num_derivs::<3>(&f, &a);
                a_sum += v[1];
                b_sum += v[2];
                dcount += self.d[i];
                per.push((i, v, dg, dh));
            }
            if self.frailty == Frailty::None {
                for (i, v, dg, dh) in &per {
                    let (i, w) = (*i, self.w[*i]);
                    let di = self.d[i];
                    ll += w * (di * v[0] - v[1] + v[2]);
                    let xv = self.xvec(i);
                    let ge: Vec<f64> = (0..m).map(|e| di * dg[e][0] - dg[e][1] + dg[e][2]).collect();
                    for &(p, e, x) in &xv {
                        g[p] += w * ge[e] * x;
                        if need_scores {
                            scores[i][p] = w * ge[e] * x;
                        }
                        for &(q, f2, x2) in &xv {
                            let he = di * dh[e * m + f2][0] - dh[e * m + f2][1] + dh[e * m + f2][2];
                            h[(p, q)] -= w * he * x * x2;
                        }
                    }
                }
                continue;
            }
            // frailty group: sum d ln h + F(A, B, ln theta)
            let fr = self.frailty;
            let ff = |x: &[f64]| [frailty_f(fr, dcount, x[0], x[1], x[2])];
            let (fv, fg, fh) = num_derivs::<1>(&ff, &[a_sum, b_sum, lnth]);
            ll += fv[0];
            let (fa, fb, fp) = (fg[0][0], fg[1][0], fg[2][0]);
            let mut ga = vec![0.0; k];
            let mut gb = vec![0.0; k];
            for (i, v, dg, dh) in &per {
                let i = *i;
                let di = self.d[i];
                ll += di * v[0];
                let xv = self.xvec(i);
                for &(p, e, x) in &xv {
                    let ge = di * dg[e][0] + fa * dg[e][1] + fb * dg[e][2];
                    g[p] += ge * x;
                    if need_scores {
                        scores[i][p] += ge * x;
                    }
                    ga[p] += dg[e][1] * x;
                    gb[p] += dg[e][2] * x;
                    for &(q, f2, x2) in &xv {
                        let he = di * dh[e * m + f2][0] + fa * dh[e * m + f2][1] + fb * dh[e * m + f2][2];
                        h[(p, q)] -= he * x * x2;
                    }
                }
            }
            g[k - 1] += fp;
            if need_scores {
                if let Some(&last) = grp.last() {
                    scores[last][k - 1] += fp;
                }
            }
            let (faa, fab, fbb) = (fh[0][0], fh[1][0], fh[4][0]);
            let (fap, fbp, fpp) = (fh[2][0], fh[5][0], fh[8][0]);
            for p in 0..k {
                for q in 0..k {
                    h[(p, q)] -= faa * ga[p] * ga[q] + fab * (ga[p] * gb[q] + gb[p] * ga[q]) + fbb * gb[p] * gb[q];
                }
                h[(p, k - 1)] -= fap * ga[p] + fbp * gb[p];
                h[(k - 1, p)] -= fap * ga[p] + fbp * gb[p];
            }
            h[(k - 1, k - 1)] -= fpp;
        }
        (ll, g, h, scores)
    }
}

/// Newton-Raphson with step halving; where the information matrix is not
/// positive definite (non-concave region) a ridge is added until it is.
pub fn maximize(model: &dyn Likelihood, b0: Vec<f64>, max_iter: usize) -> Result<est::MlResult, String> {
    let k = model.k();
    let mut b = b0;
    let (mut ll, mut g, mut h, _) = model.eval(&b, false);
    if !ll.is_finite() {
        return Err(rc_err(1400, "initial values not feasible"));
    }
    let mut log = vec![ll];
    let mut converged = false;
    for _ in 0..max_iter {
        let scale: f64 = (0..k).map(|i| h[(i, i)].abs()).fold(0.0, f64::max).max(1e-8);
        let mut mu = 0.0;
        let mut step = vec![0.0; k];
        for _ in 0..60 {
            let mut hm = h.clone();
            for i in 0..k {
                hm[(i, i)] += mu;
            }
            if hm.cholesky().is_ok() {
                let (hinv, _) = hm.sym_inverse(1e-14);
                step = (0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum()).collect();
                break;
            }
            mu = if mu == 0.0 { 1e-6 * scale } else { mu * 4.0 };
        }
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..50 {
            let nb: Vec<f64> = b.iter().zip(&step).map(|(x, s)| x + lambda * s).collect();
            let res = model.eval(&nb, false);
            if res.0.is_finite() && res.0 >= ll - 1e-12 * ll.abs().max(1.0) {
                accepted = Some((nb, res));
                break;
            }
            lambda /= 2.0;
        }
        let Some((nb, res)) = accepted else { break };
        let change = (res.0 - ll).abs();
        let bchange = b.iter().zip(&nb).map(|(a, c)| (a - c).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        ll = res.0;
        g = res.1;
        h = res.2;
        log.push(ll);
        let pd = h.cholesky().is_ok();
        let gnorm: f64 = {
            let (hi, _) = h.sym_inverse(1e-14);
            (0..k).map(|i| (0..k).map(|j| g[i] * hi[(i, j)] * g[j]).sum::<f64>()).sum::<f64>()
        };
        if pd && ((change < 1e-10 * ll.abs().max(1.0) && bchange < 1e-8) || gnorm.abs() < 1e-14) {
            converged = true;
            break;
        }
    }
    let (ll, _, info, scores) = model.eval(&b, true);
    Ok(est::MlResult { b, ll, info, scores, iterations: log, converged })
}

// ---------------------------------------------------------------------------
// streg command
// ---------------------------------------------------------------------------

pub struct Streg;

fn anc_row(lw: usize, label: &str, b: f64, se: f64, lo: f64, hi: f64, with_z: bool) -> String {
    let c = |v: f64| fmt_g(v, 8);
    if with_z {
        let z = b / se;
        let p = 2.0 * dist::normal(-z.abs());
        format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", label, c(b), c(se), z, p, c(lo), c(hi))
    } else {
        format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", label, c(b), c(se), "", c(lo), c(hi))
    }
}

impl CommandHandler for Streg {
    fn name(&self) -> &'static str {
        "streg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() && cmd.opt("distribution", 1).is_none() {
            if st.e.macros.get("cmd2").is_some_and(|c| c == "streg") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        require_st(ds)?;
        let dist = Dist::parse(cmd.opt("distribution", 1).unwrap_or("exponential"))?;
        let want_time = cmd.has_opt("time", 2);
        let aft = match dist {
            Dist::Exponential | Dist::Weibull => want_time,
            Dist::Gompertz => {
                if want_time {
                    return Err(rc_err(198, "option time not allowed with distribution(gompertz)"));
                }
                false
            }
            _ => true,
        };
        let frailty = match cmd.opt("frailty", 2).map(|s| s.trim().to_lowercase()) {
            None => Frailty::None,
            Some(f) if "gamma".starts_with(&f) => Frailty::Gamma,
            Some(f) if "invgaussian".starts_with(&f) && f.len() >= 3 => Frailty::InvGauss,
            Some(f) => return Err(rc_err(198, format!("frailty({f}) invalid"))),
        };
        let shared = cmd.opt("shared", 3).map(|s| varlist::resolve_name(ds, s.trim())).transpose()?;
        if shared.is_some() && frailty == Frailty::None {
            return Err(rc_err(198, "shared() requires frailty()"));
        }
        let strata: Vec<String> = match cmd.opt("strata", 3) {
            Some(s) => varlist::expand(ds, s)?,
            None => Vec::new(),
        };
        let anc_vars: Vec<String> = match cmd.opt("ancillary", 3) {
            Some(s) => varlist::expand(ds, s)?,
            None => Vec::new(),
        };
        let anc2_vars: Vec<String> = match cmd.opt("anc2", 4) {
            Some(s) => varlist::expand(ds, s)?,
            None => Vec::new(),
        };
        if dist == Dist::Exponential && (!anc_vars.is_empty() || !strata.is_empty()) {
            return Err(rc_err(198, "ancillary() and strata() not allowed with distribution(exponential)"));
        }
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
        if frailty != Frailty::None && wt.is_some() {
            return Err(rc_err(101, "weights not allowed with frailty()"));
        }
        let c2 = with_st_if(cmd)?;
        let mut extra = strata.clone();
        extra.extend(anc_vars.iter().cloned());
        extra.extend(anc2_vars.iter().cloned());
        if let Some(s) = &shared {
            extra.push(s.clone());
        }
        if let Vce::Cluster(v) = &vce {
            extra.push(varlist::resolve_name(ds, v)?);
        }
        let mut d = est::build(ds, st, &c2, DesignSpec { depvar: None, indep: &cmd.main, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra })?;
        if let Some((_, w)) = &wt {
            let keep: Vec<bool> = d.rows.iter().map(|&i| !is_missing(w[i]) && w[i] > 0.0).collect();
            if keep.iter().any(|k| !k) {
                d = crate::xtnl::subset_design(&d, &keep);
            }
        }
        let rows = d.rows.clone();
        let n = rows.len();
        let act = d.active();
        // equations
        let main = Eq { name: "_t".into(), names: act.iter().map(|&j| d.names[j].clone()).collect(), cols: act.iter().map(|&j| d.cols[j].clone()).collect() };
        let mut eqs = vec![main];
        // strata enter the ancillary equation as indicators (and the main one)
        let mut strata_cols: Vec<(String, Vec<f64>)> = Vec::new();
        if !strata.is_empty() {
            let sv = util::col(ds, &strata[0])?;
            let mut levels: Vec<f64> = rows.iter().map(|&i| sv[i]).collect();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            for &l in levels.iter().skip(1) {
                strata_cols.push((format!("_S{}_{}", strata[0], fmt_g(l, 8)), rows.iter().map(|&i| (sv[i] == l) as u8 as f64).collect()));
            }
            let e0 = &mut eqs[0];
            let cons = e0.names.iter().position(|n| n == "_cons");
            for (k, (nm, c)) in strata_cols.iter().enumerate() {
                let at = cons.map_or(e0.names.len(), |p| p + k);
                e0.names.insert(at, nm.clone());
                e0.cols.insert(at, c.clone());
            }
        }
        for (k, anc) in dist.anc().iter().enumerate() {
            let vars = if k == 0 { &anc_vars } else { &anc2_vars };
            let mut names = Vec::new();
            let mut cols = Vec::new();
            for v in vars {
                let c = util::col(ds, v)?;
                names.push(v.clone());
                cols.push(rows.iter().map(|&i| c[i]).collect());
            }
            if k == 0 {
                for (nm, c) in &strata_cols {
                    names.push(nm.clone());
                    cols.push(c.clone());
                }
            }
            names.push("_cons".into());
            cols.push(vec![1.0; n]);
            eqs.push(Eq { name: anc.to_string(), names, cols });
        }
        let (tc, t0c, dc) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?);
        let t: Vec<f64> = rows.iter().map(|&i| tc[i]).collect();
        let t0: Vec<f64> = rows.iter().map(|&i| t0c[i]).collect();
        let dv: Vec<f64> = rows.iter().map(|&i| dc[i]).collect();
        let w: Vec<f64> = match &wt {
            Some((_, w)) => rows.iter().map(|&i| w[i]).collect(),
            None => vec![1.0; n],
        };
        let groups: Vec<Vec<usize>> = match &shared {
            Some(s) => {
                let c = util::col(ds, s)?;
                let mut m: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
                for (r, &i) in rows.iter().enumerate() {
                    m.entry(c[i].to_bits()).or_default().push(r);
                }
                m.into_values().collect()
            }
            None => (0..n).map(|r| vec![r]).collect(),
        };
        // starting values from moments of ln t among failures
        let nf: f64 = dv.iter().zip(&w).map(|(d, w)| d * w).sum();
        let tr: f64 = t.iter().zip(&t0).zip(&w).map(|((a, b), w)| (a - b) * w).sum();
        let lts: Vec<f64> = t.iter().map(|x| x.ln()).collect();
        let mlt = lts.iter().sum::<f64>() / n as f64;
        let sdlt = (lts.iter().map(|x| (x - mlt).powi(2)).sum::<f64>() / n as f64).sqrt().max(0.1);
        let start = |eqs: &[Eq], frail: bool| -> Vec<f64> {
            let mut b = Vec::new();
            for (e, eq) in eqs.iter().enumerate() {
                for nm in &eq.names {
                    let v = if nm != "_cons" {
                        0.0
                    } else if e == 0 {
                        let lam = (nf / tr).max(1e-10).ln();
                        match dist {
                            Dist::Exponential | Dist::Weibull | Dist::Gompertz => if aft { -lam } else { lam },
                            _ => mlt,
                        }
                    } else {
                        match (dist, eq.name.as_str()) {
                            (Dist::Lognormal, _) | (Dist::Ggamma, "ln_sig") => sdlt.ln(),
                            (Dist::Loglogistic, _) => (sdlt * 3f64.sqrt() / std::f64::consts::PI).ln(),
                            (Dist::Ggamma, "kappa") => 1.0,
                            _ => 0.0,
                        }
                    };
                    b.push(v);
                }
            }
            if frail {
                b.push((0.5f64).ln());
            }
            b
        };
        let mut out = vec![st_header(ds), String::new()];
        out.extend(d.notes.iter().cloned());
        let robust = vce != Vce::Oim;
        let ll_label = if robust { "Log pseudolikelihood" } else { "Log likelihood" };
        // constant-only model (ancillary equations kept)
        let mut eq0 = eqs.clone();
        eq0[0].cols.retain(|_| true);
        let keep0: Vec<bool> = eq0[0].names.iter().map(|nm| nm == "_cons" || nm.starts_with("_S")).collect();
        eq0[0].cols = eq0[0].cols.iter().zip(&keep0).filter(|(_, k)| **k).map(|(c, _)| c.clone()).collect();
        eq0[0].names = eq0[0].names.iter().zip(&keep0).filter(|(_, k)| **k).map(|(c, _)| c.clone()).collect();
        let fit = |eqs: &[Eq], fr: Frailty, b0: Vec<f64>| -> Result<est::MlResult, String> {
            let model = Model { dist, aft, eqs, t: &t, t0: &t0, d: &dv, w: &w, frailty: fr, groups: if fr == Frailty::None { (0..n).map(|r| vec![r]).collect() } else { groups.clone() } };
            // a Weibull start for the generalized gamma
            maximize(&model, b0, 300)
        };
        let mut b0 = start(&eq0, false);
        let has_cons0 = eq0[0].names.iter().any(|n| n == "_cons");
        let null = if has_cons0 { Some(fit(&eq0, Frailty::None, b0.clone())?) } else { None };
        if let Some(nr) = &null {
            out.push("Fitting constant-only model:".into());
            out.extend(crate::stcox::iter_log(&nr.iterations, ll_label, false));
            out.push(String::new());
            b0 = nr.b.clone();
        }
        // full model starting from the constant-only fit
        let mut bfull = start(&eqs, false);
        if null.is_some() {
            let mut pos0 = 0;
            for (e, eq) in eqs.iter().enumerate() {
                let off: usize = eqs[..e].iter().map(|q| q.cols.len()).sum();
                for (j, nm) in eq.names.iter().enumerate() {
                    if eq0[e].names.contains(nm) {
                        let k0: usize = eq0[..e].iter().map(|q| q.cols.len()).sum::<usize>() + eq0[e].names.iter().position(|x| x == nm).unwrap();
                        bfull[off + j] = b0[k0];
                        pos0 += 1;
                    }
                }
            }
            let _ = pos0;
        }
        out.push("Fitting full model:".into());
        let mut res = fit(&eqs, Frailty::None, bfull)?;
        out.extend(crate::stcox::iter_log(&res.iterations, ll_label, false));
        let mut ll_nofrail = SYSMISS;
        if frailty != Frailty::None {
            ll_nofrail = res.ll;
            let b_nofrail = res.b.clone();
            let mut bf = res.b.clone();
            bf.push((0.5f64).ln());
            out.push(String::new());
            out.push(format!("Fitting {} model with frailty:", dist.name()));
            res = fit(&eqs, frailty, bf)?;
            // The model without frailty is the limit theta -> 0, so the
            // frailty fit can never be worse; if the search stopped below it
            // (a flat ridge when the frailty variance is near zero), restart
            // from that boundary and keep the better fit.
            if res.ll < ll_nofrail - 1e-9 {
                let mut bb = b_nofrail;
                bb.push(-12.0);
                if let Ok(r2) = fit(&eqs, frailty, bb) {
                    if r2.ll > res.ll {
                        res = r2;
                    }
                }
            }
            out.extend(crate::stcox::iter_log(&res.iterations, ll_label, false));
        }
        out.push(String::new());
        let mut ll0 = null.as_ref().map_or(SYSMISS, |r| r.ll);
        if frailty != Frailty::None && null.is_some() {
            // constant-only model with frailty, for the LR test of the covariates
            let mut b0f = b0.clone();
            b0f.push(*res.b.last().unwrap());
            if let Ok(r0) = fit(&eq0, frailty, b0f) {
                ll0 = r0.ll.max(ll0);
            }
        }
        let k = res.b.len();
        // variance
        let (hinv, _) = res.info.sym_inverse(1e-14);
        let mut nclust = None;
        let v = match &vce {
            Vce::Oim | Vce::Ols => hinv.clone(),
            _ => {
                let cl = match &vce {
                    Vce::Cluster(c) => {
                        let col = util::col(ds, &varlist::resolve_name(ds, c)?)?;
                        Some(rows.iter().map(|&i| col[i]).collect::<Vec<f64>>())
                    }
                    _ => None,
                };
                let (vv, g) = est::sandwich(&hinv, &res.scores, cl.as_deref(), 1.0);
                if cl.is_some() {
                    nclust = Some(g);
                }
                let gf = g as f64;
                vv.scale(gf / (gf - 1.0))
            }
        };
        // names
        let mut names: Vec<String> = Vec::new();
        for (e, eq) in eqs.iter().enumerate() {
            for nm in &eq.names {
                if e > 0 && eq.names.len() == 1 {
                    names.push(format!("/:{}", eq.name));
                } else {
                    names.push(format!("{}:{nm}", eq.name));
                }
            }
        }
        if frailty != Frailty::None {
            names.push("/:ln_the".into());
        }
        let kmain = eqs[0].cols.len();
        let df_m = eqs[0].names.iter().filter(|n| *n != "_cons" && !n.starts_with("_S")).count() as f64;
        let (chi2, chi_label) = if robust || null.is_none() {
            let idx: Vec<usize> = (0..kmain).filter(|&j| eqs[0].names[j] != "_cons" && !eqs[0].names[j].starts_with("_S")).collect();
            let mut vs = Matrix::zeros(idx.len(), idx.len());
            for (a, &i) in idx.iter().enumerate() {
                for (c, &j) in idx.iter().enumerate() {
                    vs[(a, c)] = v[(i, j)];
                }
            }
            let (vi, _) = vs.sym_inverse(1e-14);
            let bs: Vec<f64> = idx.iter().map(|&i| res.b[i]).collect();
            ((0..bs.len()).map(|a| (0..bs.len()).map(|c| bs[a] * vi[(a, c)] * bs[c]).sum::<f64>()).sum::<f64>(), format!("Wald chi2({})", df_m))
        } else {
            (2.0 * (res.ll - ll0), format!("LR chi2({})", df_m))
        };
        let p = if df_m > 0.0 { dist::chi2tail(df_m, chi2) } else { SYSMISS };
        let nobs: f64 = if wt.as_ref().is_some_and(|(k, _)| k == "fweight") { w.iter().sum() } else { n as f64 };
        let nsub = n_subjects(ds, &rows, &w);
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let metric = if aft { "AFT" } else { "PH" };
        let mut title = format!("{} {metric} regression", dist.label());
        if frailty != Frailty::None {
            let fl = if frailty == Frailty::Gamma { "gamma" } else { "inverse-Gaussian" };
            title = match &shared {
                Some(s) => format!("{title}\nShared {fl} frailty\nGroup variable: {s}"),
                None => format!("{title}\n{}{} frailty", fl[..1].to_uppercase(), &fl[1..]),
            };
        }
        out.extend(st_model_header(&title, nsub, nf, tr, nobs, &chi_label, if df_m > 0.0 { chi2 } else { SYSMISS }, p, ll_label, res.ll));
        if let Some(g) = nclust {
            let cv = match &vce {
                Vce::Cluster(c) => c.clone(),
                _ => String::new(),
            };
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), cv)));
        }
        out.push(String::new());
        let eform = if !aft && !cmd.has_opt("nohr", 3) {
            Some("Haz. ratio")
        } else if aft && cmd.has_opt("tr", 2) {
            Some("Time ratio")
        } else {
            None
        };
        // table: equations with covariates through coef_table, scalar ancillaries by hand
        let multi_eq = eqs.iter().skip(1).any(|e| e.names.len() > 1);
        let mut tnames = Vec::new();
        let mut tb = Vec::new();
        let mut tidx = Vec::new();
        for (e, eq) in eqs.iter().enumerate() {
            if e > 0 && eq.names.len() == 1 {
                continue;
            }
            let off: usize = eqs[..e].iter().map(|q| q.cols.len()).sum();
            for (j, nm) in eq.names.iter().enumerate() {
                tnames.push(if multi_eq { format!("{}:{nm}", eq.name) } else { nm.clone() });
                tb.push(res.b[off + j]);
                tidx.push(off + j);
            }
        }
        let mut tv = Matrix::zeros(tidx.len(), tidx.len());
        for (a, &i) in tidx.iter().enumerate() {
            for (c, &j) in tidx.iter().enumerate() {
                tv[(a, c)] = v[(i, j)];
            }
        }
        let falses = vec![false; tnames.len()];
        // eform applies to the first equation only
        let mut table = est::coef_table(&est::TableSpec { depvar: "_t", names: &tnames, b: &tb, v: &tv, omitted: &falses, is_base: &falses, df: None, level, eform, coef_title: "Coefficient" });
        if multi_eq && eform.is_some() {
            // undo the transform for ancillary equations: redraw those rows untransformed
            let plain = est::coef_table(&est::TableSpec { depvar: "_t", names: &tnames, b: &tb, v: &tv, omitted: &falses, is_base: &falses, df: None, level, eform: None, coef_title: "Coefficient" });
            let first_other = table.iter().position(|l| eqs.iter().skip(1).any(|e| l.trim_end().starts_with(&e.name) && l.trim_end().ends_with('|')));
            if let Some(pos) = first_other {
                table.truncate(pos);
                table.extend(plain[pos..].iter().cloned());
            }
        }
        let bar = table.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        let mut scalars: Vec<(String, f64, f64)> = Vec::new(); // (name, b, se)
        for (e, eq) in eqs.iter().enumerate() {
            if e > 0 && eq.names.len() == 1 {
                let off: usize = eqs[..e].iter().map(|q| q.cols.len()).sum();
                scalars.push((eq.name.clone(), res.b[off], v[(off, off)].max(0.0).sqrt()));
            }
        }
        if frailty != Frailty::None {
            scalars.push(("ln_the".into(), res.b[k - 1], v[(k - 1, k - 1)].max(0.0).sqrt()));
        }
        if !scalars.is_empty() {
            table.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            for (nm, b, se) in &scalars {
                table.push(anc_row(lw, &format!("/{nm}"), *b, *se, b - z * se, b + z * se, true));
            }
            let mut trans = Vec::new();
            let find = |nm: &str| scalars.iter().find(|s| s.0 == nm).cloned();
            if let Some((_, b, se)) = find("ln_p") {
                let p = b.exp();
                trans.push(anc_row(lw, "p", p, p * se, (b - z * se).exp(), (b + z * se).exp(), false));
                trans.push(anc_row(lw, "1/p", 1.0 / p, se / p, (-(b + z * se)).exp(), (-(b - z * se)).exp(), false));
            }
            if let Some((_, b, se)) = find("ln_sig") {
                trans.push(anc_row(lw, "sigma", b.exp(), b.exp() * se, (b - z * se).exp(), (b + z * se).exp(), false));
            }
            if let Some((_, b, se)) = find("ln_gam") {
                trans.push(anc_row(lw, "gamma", b.exp(), b.exp() * se, (b - z * se).exp(), (b + z * se).exp(), false));
            }
            if let Some((_, b, se)) = find("ln_the") {
                trans.push(anc_row(lw, "theta", b.exp(), b.exp() * se, (b - z * se).exp(), (b + z * se).exp(), false));
            }
            if !trans.is_empty() {
                table.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
                table.extend(trans);
            }
        }
        table.push(bar);
        out.extend(table);
        let mut chibar = SYSMISS;
        let mut pchibar = SYSMISS;
        if frailty != Frailty::None {
            chibar = (2.0 * (res.ll - ll_nofrail)).max(0.0);
            pchibar = if chibar <= 0.0 { 1.0 } else { 0.5 * dist::chi2tail(1.0, chibar) };
            out.push(format!("LR test of theta=0: chibar2(01) = {:.2}{:>27}", chibar, format!("Prob >= chibar2 = {:.3}", pchibar)));
        }
        if eform.is_some() {
            if eqs.len() > 1 || frailty != Frailty::None {
                out.push(format!("Note: Estimates are transformed only in the first equation to {}.", if aft { "time ratios" } else { "hazard ratios" }));
            }
            if eqs[0].names.iter().any(|n| n == "_cons") {
                out.push(format!("Note: _cons estimates baseline {}.", if aft { "time" } else { "hazard" }));
            }
        }
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let mut sample = vec![false; ds.n_obs];
        for &i in &rows {
            sample[i] = true;
        }
        let e = st.begin_e();
        e.set("N", nobs);
        e.set("N_sub", nsub);
        e.set("N_fail", nf);
        e.set("risk", tr);
        e.set("ll", res.ll);
        e.set("ll_0", ll0);
        e.set("chi2", if df_m > 0.0 { chi2 } else { SYSMISS });
        e.set("df_m", df_m);
        e.set("p", p);
        e.set("k", k as f64);
        e.set("k_eq", (eqs.len() + (frailty != Frailty::None) as usize) as f64);
        e.set("k_aux", (eqs.len() - 1 + (frailty != Frailty::None) as usize) as f64);
        e.set("rank", k as f64);
        e.set("converged", res.converged as u8 as f64);
        e.set("ic", (res.iterations.len() - 1) as f64);
        if let Some(g) = nclust {
            e.set("N_clust", g as f64);
        }
        for (nm, b, _) in &scalars {
            match nm.as_str() {
                "ln_p" => e.set("aux_p", b.exp()),
                "ln_sig" => e.set("sigma", b.exp()),
                "ln_gam" => e.set("gamma", b.exp()),
                "gamma" => e.set("gamma", *b),
                "kappa" => e.set("kappa", *b),
                "ln_the" => e.set("theta", b.exp()),
                _ => {}
            }
        }
        if frailty != Frailty::None {
            e.set("ll_c", ll_nofrail);
            e.set("chi2_c", chibar);
            e.set("p_c", pchibar);
            e.set_str("frailty", if frailty == Frailty::Gamma { "gamma" } else { "invgaussian" });
            if let Some(s) = &shared {
                e.set_str("shared", s.clone());
            }
        }
        e.set_str("cmd", dist.ecmd());
        e.set_str("cmd2", "streg");
        e.set_str("distribution", dist.name());
        e.set_str("frm2", if aft { "time" } else { "hazard" });
        e.set_str("title", title.lines().next().unwrap_or("").to_string());
        e.set_str("depvar", "_t");
        e.set_str("t0", "_t0");
        e.set_str("predict", "streg_p");
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
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&res.b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.sample = Some(sample);
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// postestimation
// ---------------------------------------------------------------------------

/// Linear predictors of every equation for observation i: main, then each
/// ancillary parameter (strata indicators are rebuilt from their names).
fn eq_values(ds: &DataSet, env: &Env, b: &Matrix, i: usize, at: Option<&[f64]>) -> Result<Vec<f64>, String> {
    let mut vals: Vec<(String, f64)> = Vec::new();
    for (j, name) in b.colnames.iter().enumerate() {
        let (eq, nm) = name.split_once(':').unwrap_or(("_t", name));
        let (eq, nm) = if eq == "/" { (nm, "_cons") } else { (eq, nm) };
        let x = if let Some(a) = at {
            a[j]
        } else if nm == "_cons" {
            1.0
        } else if let Some(rest) = nm.strip_prefix("_S") {
            // _S<var>_<level>
            let (var, lv) = rest.rsplit_once('_').unwrap_or((rest, ""));
            let c = util::col(ds, var)?;
            (c[i] == lv.parse::<f64>().unwrap_or(f64::NAN)) as u8 as f64
        } else if eq == "ln_the" {
            0.0
        } else {
            crate::postest::coef_value(env, nm, i)?
        };
        if is_missing(x) {
            return Ok(vec![SYSMISS]);
        }
        match vals.iter_mut().find(|(e, _)| e == eq) {
            Some(v) => v.1 += x * b.data[j],
            None => vals.push((eq.to_string(), x * b.data[j])),
        }
    }
    Ok(vals.into_iter().filter(|(e, _)| e != "ln_the").map(|(_, v)| v).collect())
}

fn model_of(st: &Store) -> Option<(Dist, bool)> {
    if st.e.macros.get("cmd2").map(String::as_str) != Some("streg") {
        return None;
    }
    let d = Dist::from_ecmd(st.e.macros.get("cmd").map(String::as_str).unwrap_or(""))?;
    Some((d, st.e.macros.get("frm2").is_some_and(|f| f == "time")))
}

/// Survivor function at t from (ln h, H).
fn surv(dist: Dist, aft: bool, t: f64, a: &[f64]) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-lnh_cumh(dist, aft, t, a[0], &a[1..]).1).exp()
}

fn median_time(dist: Dist, aft: bool, a: &[f64], q: f64) -> f64 {
    // solve S(t) = q by bisection on ln t
    let (mut lo, mut hi) = (-50.0f64, 50.0f64);
    if surv(dist, aft, hi.exp(), a) > q {
        return SYSMISS;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if surv(dist, aft, mid.exp(), a) > q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

fn mean_time(dist: Dist, aft: bool, a: &[f64]) -> f64 {
    // integral of S over (0, inf) on the log scale: int S(e^u) e^u du
    let (x, w) = crate::stcox::gauss_legendre(40);
    let tmax = median_time(dist, aft, a, 1e-12);
    if is_missing(tmax) {
        return SYSMISS;
    }
    let (lo, hi) = ((median_time(dist, aft, a, 1.0 - 1e-12).max(1e-300)).ln() - 1.0, tmax.ln());
    let mut tot = 0.0;
    let panels = 20;
    for p in 0..panels {
        let a0 = lo + (hi - lo) * p as f64 / panels as f64;
        let b0 = lo + (hi - lo) * (p + 1) as f64 / panels as f64;
        let (c, h) = ((a0 + b0) / 2.0, (b0 - a0) / 2.0);
        for (xi, wi) in x.iter().zip(&w) {
            let u = c + h * xi;
            tot += h * wi * surv(dist, aft, u.exp(), a) * u.exp();
        }
    }
    tot + lo.exp()
}

/// `predict` after streg; None for other estimators.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let Some((dist, aft)) = model_of(st) else { return Ok(None) };
    let known = ["median", "mean", "hazard", "hr", "xb", "stdp", "surv", "csurv", "csnell", "mgale", "deviance", "time", "lntime"];
    let opts: Vec<String> = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).collect();
    let pick = |k: &str| opts.iter().any(|o| o == k || (o.len() >= 3 && k.starts_with(o.as_str())));
    let stat = known.iter().find(|k| pick(k)).map(|s| s.to_string()).unwrap_or_default();
    let mut msg = String::new();
    let stat = match stat.as_str() {
        "" | "time" => {
            if stat.is_empty() {
                msg = "(option median time assumed; predicted median time)".into();
            }
            if pick("mean") { "mean".to_string() } else { "median".to_string() }
        }
        "median" | "mean" => stat,
        s => s.to_string(),
    };
    let names = predict_names(ds, &cmd.main, 1)?;
    if names.len() != 1 {
        return Err(rc_err(103, "too many variables specified"));
    }
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(b.cols, b.cols));
    let mask = util::touse(ds, st, cmd)?;
    let sample = st.e.sample.clone().unwrap_or_default();
    let n = ds.n_obs;
    let (tc, t0c, dc) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?);
    let lntime = pick("lntime") || (stat == "median" && opts.iter().any(|o| o == "lntime"));
    let env = Env::with_store(ds, st);
    let mut vals = vec![SYSMISS; n];
    let mut label = String::new();
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let a = eq_values(ds, &env, &b, i, None)?;
        if a.iter().any(|x| is_missing(*x)) {
            continue;
        }
        let in_st = !is_missing(tc[i]);
        vals[i] = match stat.as_str() {
            "median" => {
                label = "Predicted median _t".into();
                let m = median_time(dist, aft, &a, 0.5);
                if lntime { m.ln() } else { m }
            }
            "mean" => {
                label = "Predicted mean _t".into();
                mean_time(dist, aft, &a)
            }
            "xb" => {
                label = "Linear prediction".into();
                a[0]
            }
            "hr" => {
                if aft {
                    return Err(rc_err(198, "option hr not allowed after an AFT model"));
                }
                label = "Predicted hazard ratio".into();
                a[0].exp()
            }
            "stdp" => {
                label = "S.E. of linear prediction".into();
                let x: Vec<f64> = b.colnames.iter().map(|c| if c.starts_with("_t:") { crate::postest::coef_value(&env, c.trim_start_matches("_t:"), i) } else { Ok(0.0) }).collect::<Result<_, _>>()?;
                let mut s = 0.0;
                for p in 0..x.len() {
                    for q in 0..x.len() {
                        s += x[p] * v[(p, q)] * x[q];
                    }
                }
                s.max(0.0).sqrt()
            }
            "hazard" if in_st => {
                label = "Predicted hazard".into();
                lnh_cumh(dist, aft, tc[i], a[0], &a[1..]).0.exp()
            }
            "surv" | "csurv" if in_st => {
                label = "Predicted S(t|t0)".into();
                surv(dist, aft, tc[i], &a) / surv(dist, aft, t0c[i], &a)
            }
            "csnell" | "mgale" | "deviance" if in_st && sample.get(i).copied().unwrap_or(false) => {
                let h = lnh_cumh(dist, aft, tc[i], a[0], &a[1..]).1 - if t0c[i] > 0.0 { lnh_cumh(dist, aft, t0c[i], a[0], &a[1..]).1 } else { 0.0 };
                h
            }
            _ => SYSMISS,
        };
    }
    drop(env);
    // residuals: aggregate per subject unless partial
    if matches!(stat.as_str(), "csnell" | "mgale" | "deviance" | "csurv") {
        let partial = cmd.has_opt("partial", 4);
        let id = if partial { None } else { st_id(ds) };
        let mut groups: Vec<Vec<usize>> = match id.and_then(|v| util::col(ds, &v).ok()) {
            Some(c) => {
                let mut m: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
                for i in 0..n {
                    if !is_missing(vals[i]) {
                        m.entry(c[i].to_bits()).or_default().push(i);
                    }
                }
                m.into_values().collect()
            }
            None => (0..n).filter(|&i| !is_missing(vals[i])).map(|i| vec![i]).collect(),
        };
        let mut nv = vec![SYSMISS; n];
        for g in groups.iter_mut() {
            g.sort_by(|&a, &c| tc[a].total_cmp(&tc[c]));
            let last = *g.last().unwrap();
            let dsum: f64 = g.iter().map(|&i| dc[i]).sum();
            if stat == "csurv" {
                // running product over the subject's records
                let mut p = 1.0;
                for &i in g.iter() {
                    p *= vals[i];
                    nv[i] = p;
                }
                continue;
            }
            let hsum: f64 = g.iter().map(|&i| vals[i]).sum();
            nv[last] = match stat.as_str() {
                "csnell" => hsum,
                "mgale" => dsum - hsum,
                _ => {
                    let m = dsum - hsum;
                    let inner = if dsum > 0.0 { m + dsum * (dsum - m).ln() } else { m };
                    m.signum() * (-2.0 * inner).max(0.0).sqrt()
                }
            };
        }
        vals = nv;
        label = match stat.as_str() {
            "csnell" => "Cox–Snell residual",
            "mgale" => "Martingale",
            "deviance" => "Deviance residual",
            _ => "Predicted S(t|t0), cumulative",
        }
        .into();
    }
    let nmiss = vals.iter().filter(|x| is_missing(**x)).count();
    let mut var = util::new_double(&names[0], vals);
    var.var_type = ferrostat_core::VarType::Float;
    var.label = label;
    ds.add_variable(var).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(Some(lines.join("\n")))
}

/// Curve for stcurve after streg: survival, failure, hazard or cumhaz on a grid.
pub fn curve(ds: &DataSet, st: &Store, what: &str, at: &[(String, f64)], cmd: &Command) -> Result<Vec<(f64, f64)>, String> {
    let (dist, aft) = model_of(st).ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let xv = curve_covariates(ds, st, &b.colnames, at)?;
    let env = Env::with_store(ds, st);
    let a = eq_values(ds, &env, &b, 0, Some(&xv))?;
    let tc = util::col(ds, "_t")?;
    let sample = st.e.sample.clone().unwrap_or_default();
    let tmax = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).map(|i| tc[i]).fold(0.0, f64::max);
    let (lo, hi) = match cmd.opt("range", 1) {
        Some(r) => {
            let v = crate::margins::numlist(r)?;
            (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(tmax))
        }
        None => (0.0, tmax),
    };
    Ok((0..=100)
        .filter_map(|q| {
            let t = lo + (hi - lo) * q as f64 / 100.0;
            if t <= 0.0 {
                return match what {
                    "survival" => Some((0.0, 1.0)),
                    "failure" | "cumhaz" => Some((0.0, 0.0)),
                    _ => None,
                };
            }
            let (lh, h) = lnh_cumh(dist, aft, t, a[0], &a[1..]);
            Some((
                t,
                match what {
                    "survival" => (-h).exp(),
                    "failure" => 1.0 - (-h).exp(),
                    "cumhaz" => h,
                    _ => lh.exp(),
                },
            ))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hazards_integrate_to_cumulative() {
        // d H / dt = h for every distribution and metric
        for (dist, aft, anc) in [
            (Dist::Exponential, false, vec![]),
            (Dist::Weibull, true, vec![0.3]),
            (Dist::Gompertz, false, vec![0.2]),
            (Dist::Loglogistic, true, vec![-0.4]),
            (Dist::Lognormal, true, vec![0.1]),
            (Dist::Ggamma, true, vec![0.2, 0.7]),
            (Dist::Ggamma, true, vec![0.2, -0.5]),
        ] {
            let t = 1.7;
            let e = 1e-6;
            let (lh, _) = lnh_cumh(dist, aft, t, 0.4, &anc);
            let dh = (lnh_cumh(dist, aft, t + e, 0.4, &anc).1 - lnh_cumh(dist, aft, t - e, 0.4, &anc).1) / (2.0 * e);
            assert!((lh.exp() - dh).abs() < 1e-6 * dh.abs().max(1.0), "{dist:?}: {} vs {dh}", lh.exp());
        }
    }

    #[test]
    fn ggamma_nests_weibull_and_lognormal() {
        let (t, eta, lsig) = (2.3, 0.7, -0.2);
        let g = lnh_cumh(Dist::Ggamma, true, t, eta, &[lsig, 1.0]);
        let w = lnh_cumh(Dist::Weibull, true, t, eta, &[-lsig]);
        assert!((g.0 - w.0).abs() < 1e-9 && (g.1 - w.1).abs() < 1e-9, "{g:?} {w:?}");
        let g = lnh_cumh(Dist::Ggamma, true, t, eta, &[lsig, 1e-4]);
        let l = lnh_cumh(Dist::Lognormal, true, t, eta, &[lsig]);
        assert!((g.0 - l.0).abs() < 1e-3 && (g.1 - l.1).abs() < 1e-3, "{g:?} {l:?}");
    }

    #[test]
    fn shared_gamma_reduces_to_unshared() {
        // one record: D ln theta + lgamma(1/theta + D) - lgamma(1/theta) ... equals the
        // observation-level formula
        let (a, b, th) = (0.8f64, 0.1f64, 0.6f64);
        let f1 = frailty_f(Frailty::Gamma, 1.0, a, b, th.ln());
        let direct = -(1.0 / th) * (th * a).ln_1p() + (1.0 / th) * (th * b).ln_1p() - (th * a).ln_1p();
        assert!((f1 - direct).abs() < 1e-12);
        let g1 = frailty_f(Frailty::InvGauss, 1.0, a, b, th.ln());
        let ua = (1.0 + 2.0 * th * a).sqrt();
        let ub = (1.0 + 2.0 * th * b).sqrt();
        let direct = (1.0 - ua) / th - (1.0 - ub) / th - ua.ln();
        assert!((g1 - direct).abs() < 1e-12);
    }
}

