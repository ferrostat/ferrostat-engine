//! `demandsys`: systems of consumer demand, and `estat elasticities`.
//!
//! Models (budget shares w_i, prices p_j, total expenditure m, K goods,
//! demographic variables z_s):
//!
//! * `aids` (almost ideal demand system): w_i = α_i + Σ_j γ_ij ln p_j +
//!   β_i ln(m / a(p)), with ln a(p) = α_0 + Σ_k α_k ln p_k +
//!   ½ Σ_k Σ_j γ_kj ln p_k ln p_j.
//! * `quaids` (quadratic AIDS): adds λ_i / b(p) · [ln(m / a(p))]², with
//!   b(p) = Π_k p_k^β_k.
//! * `translog` (indirect translog): w_i = (α_i + Σ_j β_ij ln(p_j/m)) /
//!   (-1 + Σ_j β_Mj ln(p_j/m)), β_Mj = Σ_i β_ij, Σ_i α_i = -1.
//! * `rotterdam` (absolute-price Rotterdam model, on time-series data set
//!   with `tsset`): w̄_it Δln q_it = a_i + θ_i ΔlnQ_t + Σ_j π_ij Δln p_jt,
//!   with w̄ the average share of t and t-1, q_i = w_i m / p_i and the
//!   Divisia volume index ΔlnQ_t = Σ_i w̄_it Δln q_it; Σ θ_i = 1. The
//!   constants a_i are dropped with `noconstant`.
//!
//! `demographics(varlist)` shifts the intercepts by translation:
//! α_i(z) = α_i + Σ_s η_is z_s with Σ_i η_is = 0 (not for `rotterdam`).
//!
//! Adding-up holds by construction; homogeneity (Σ_j γ_ij = 0) and
//! symmetry (γ_ij = γ_ji) are imposed unless `nohomogeneity` or
//! `nosymmetry` is given. The last equation is dropped and the remaining
//! K-1 equations are estimated jointly by maximum likelihood under
//! multivariate normal errors, computed by iterated feasible GLS
//! (Gauss-Newton steps on the generalized sum of squares, with the error
//! covariance re-estimated after each step until the concentrated
//! log-likelihood -n/2 ln|Σ| converges). Parameters of the dropped
//! equation follow from the restrictions; their standard errors from the
//! linear map.
//!
//! Elasticities are derivatives of the fitted share functions at the means
//! of the log prices, log expenditure and demographics: expenditure
//! η_i = 1 + ∂ln w_i/∂ln m, uncompensated ε_ij = -δ_ij + ∂ln w_i/∂ln p_j,
//! compensated ε_ij + η_i w_j. For the Rotterdam model, at the mean
//! shares: η_i = θ_i / w_i, compensated π_ij / w_i, uncompensated
//! (π_ij - θ_i w_j) / w_i.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::{est, util, varlist};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Model {
    Aids,
    Quaids,
    Translog,
    Rotterdam,
}

impl Model {
    fn parse(s: &str) -> Option<Model> {
        Some(match s {
            "aids" => Model::Aids,
            "quaids" => Model::Quaids,
            "translog" => Model::Translog,
            "rotterdam" => Model::Rotterdam,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Model::Aids => "aids",
            Model::Quaids => "quaids",
            Model::Translog => "translog",
            Model::Rotterdam => "rotterdam",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Model::Aids => "Almost ideal demand system",
            Model::Quaids => "Quadratic almost ideal demand system",
            Model::Translog => "Translog demand system",
            Model::Rotterdam => "Rotterdam demand system",
        }
    }
}

/// The structural parameters of a model with K goods (Rotterdam: alpha =
/// constants a, beta = θ, g = π).
#[derive(Clone, Debug)]
struct Params {
    alpha: Vec<f64>,
    /// demographic shifts, K x nz row-major
    eta: Vec<f64>,
    beta: Vec<f64>,
    /// gamma (AIDS, QUAIDS), the translog β_ij or the Rotterdam π_ij, K x K
    g: Vec<f64>,
    lambda: Vec<f64>,
}

#[derive(Clone)]
struct Spec {
    model: Model,
    k: usize,
    nz: usize,
    homog: bool,
    sym: bool,
    alpha0: f64,
    /// intercepts (always, except Rotterdam with noconstant)
    cons: bool,
}

impl Spec {
    fn n_gamma(&self) -> usize {
        let (k, km) = (self.k, self.k - 1);
        if self.model == Model::Translog {
            return if self.sym { k * (k + 1) / 2 } else { k * k };
        }
        if self.homog {
            if self.sym { km * k / 2 } else { km * km }
        } else {
            km * k
        }
    }

    /// Free parameter count.
    fn n_free(&self) -> usize {
        let km = self.k - 1;
        let a = if self.cons { km } else { 0 };
        let b = if self.model == Model::Translog { 0 } else { km };
        let l = if self.model == Model::Quaids { km } else { 0 };
        a + km * self.nz + b + self.n_gamma() + l
    }

    /// Structural parameters from the free ones.
    fn unpack(&self, th: &[f64]) -> Params {
        let (k, km, nz) = (self.k, self.k - 1, self.nz);
        let mut pos = 0;
        let mut take = |n: usize| {
            let v = th[pos..pos + n].to_vec();
            pos += n;
            v
        };
        let alpha_sum = match self.model {
            Model::Translog => -1.0,
            Model::Rotterdam => 0.0,
            _ => 1.0,
        };
        let mut alpha = if self.cons { take(km) } else { vec![0.0; km] };
        alpha.push(alpha_sum - alpha.iter().sum::<f64>());
        let mut eta = take(km * nz);
        for s in 0..nz {
            eta.push(-(0..km).map(|i| eta[i * nz + s]).sum::<f64>());
        }
        let beta = if self.model == Model::Translog {
            Vec::new()
        } else {
            let mut b = take(km);
            let sum = if self.model == Model::Rotterdam { 1.0 } else { 0.0 };
            b.push(sum - b.iter().sum::<f64>());
            b
        };
        let mut g = vec![0.0; k * k];
        if self.model == Model::Translog {
            if self.sym {
                for i in 0..k {
                    for j in i..k {
                        let x = take(1)[0];
                        g[i * k + j] = x;
                        g[j * k + i] = x;
                    }
                }
            } else {
                g = take(k * k);
            }
        } else if self.homog && self.sym {
            for i in 0..km {
                for j in i..km {
                    let x = take(1)[0];
                    g[i * k + j] = x;
                    g[j * k + i] = x;
                }
            }
            for i in 0..km {
                let s: f64 = (0..km).map(|j| g[i * k + j]).sum();
                g[i * k + km] = -s;
                g[km * k + i] = -s;
            }
            g[km * k + km] = -(0..km).map(|j| g[km * k + j]).sum::<f64>();
        } else {
            for i in 0..km {
                let cols = if self.homog { km } else { k };
                for j in 0..cols {
                    g[i * k + j] = take(1)[0];
                }
                if self.homog {
                    g[i * k + km] = -(0..km).map(|j| g[i * k + j]).sum::<f64>();
                }
            }
            for j in 0..k {
                g[km * k + j] = -(0..km).map(|i| g[i * k + j]).sum::<f64>();
            }
        }
        let lambda = if self.model == Model::Quaids {
            let mut l = take(km);
            l.push(-l.iter().sum::<f64>());
            l
        } else {
            Vec::new()
        };
        Params { alpha, eta, beta, g, lambda }
    }

    /// Fitted shares at log prices `lp`, log expenditure `lm` and
    /// demographics `z` (Rotterdam: fitted w̄ Δln q at price changes `lp`
    /// and volume index `lm`).
    fn shares(&self, p: &Params, lp: &[f64], lm: f64, z: &[f64]) -> Vec<f64> {
        let (k, nz) = (self.k, self.nz);
        let al: Vec<f64> = (0..k).map(|i| p.alpha[i] + (0..nz).map(|s| p.eta[i * nz + s] * z[s]).sum::<f64>()).collect();
        match self.model {
            Model::Rotterdam => (0..k).map(|i| al[i] + p.beta[i] * lm + (0..k).map(|j| p.g[i * k + j] * lp[j]).sum::<f64>()).collect(),
            Model::Translog => {
                let lr: Vec<f64> = lp.iter().map(|x| x - lm).collect();
                let den = -1.0 + (0..k).map(|j| (0..k).map(|i| p.g[i * k + j]).sum::<f64>() * lr[j]).sum::<f64>();
                (0..k).map(|i| (al[i] + (0..k).map(|j| p.g[i * k + j] * lr[j]).sum::<f64>()) / den).collect()
            }
            _ => {
                let mut lna = self.alpha0;
                for a in 0..k {
                    lna += al[a] * lp[a];
                    for b in 0..k {
                        lna += 0.5 * p.g[a * k + b] * lp[a] * lp[b];
                    }
                }
                let r = lm - lna;
                let lnb: f64 = (0..k).map(|a| p.beta[a] * lp[a]).sum();
                (0..k)
                    .map(|i| {
                        let mut w = al[i] + (0..k).map(|j| p.g[i * k + j] * lp[j]).sum::<f64>() + p.beta[i] * r;
                        if self.model == Model::Quaids {
                            w += p.lambda[i] / lnb.exp() * r * r;
                        }
                        w
                    })
                    .collect()
            }
        }
    }

    /// Names (equation:coefficient) of the full parameter vector.
    fn names(&self, shares: &[String], prices: &[String], demo: &[String]) -> Vec<String> {
        let k = self.k;
        let rot = self.model == Model::Rotterdam;
        let mut v: Vec<String> = Vec::new();
        if self.cons {
            v.extend(shares.iter().map(|s| format!("{}:{s}", if rot { "a" } else { "alpha" })));
        }
        for s in shares {
            for z in demo {
                v.push(format!("eta:{s}_{z}"));
            }
        }
        if self.model != Model::Translog {
            v.extend(shares.iter().map(|s| format!("{}:{s}", if rot { "theta" } else { "beta" })));
        }
        let gname = match self.model {
            Model::Translog => "beta",
            Model::Rotterdam => "pi",
            _ => "gamma",
        };
        for i in 0..k {
            for j in 0..k {
                if self.sym && j < i {
                    continue;
                }
                v.push(format!("{gname}:{}_{}", shares[i], prices[j]));
            }
        }
        if self.model == Model::Quaids {
            v.extend(shares.iter().map(|s| format!("lambda:{s}")));
        }
        v
    }

    /// Full parameter vector in the order of `names`.
    fn full(&self, p: &Params) -> Vec<f64> {
        let k = self.k;
        let mut v = if self.cons { p.alpha.clone() } else { Vec::new() };
        v.extend(p.eta.iter().copied());
        v.extend(p.beta.iter().copied());
        for i in 0..k {
            for j in 0..k {
                if self.sym && j < i {
                    continue;
                }
                v.push(p.g[i * k + j]);
            }
        }
        v.extend(p.lambda.iter().copied());
        v
    }

    /// Structural parameters from a full vector (for postestimation).
    fn from_full(&self, f: &[f64]) -> Params {
        let k = self.k;
        let mut pos = 0;
        let mut take = |n: usize| {
            let v = f[pos..pos + n].to_vec();
            pos += n;
            v
        };
        let alpha = if self.cons { take(k) } else { vec![0.0; k] };
        let eta = take(k * self.nz);
        let beta = if self.model != Model::Translog { take(k) } else { Vec::new() };
        let mut g = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                if self.sym && j < i {
                    continue;
                }
                let x = take(1)[0];
                g[i * k + j] = x;
                if self.sym {
                    g[j * k + i] = x;
                }
            }
        }
        let lambda = if self.model == Model::Quaids { take(k) } else { Vec::new() };
        Params { alpha, eta, beta, g, lambda }
    }
}

struct Data {
    w: Vec<Vec<f64>>,
    lp: Vec<Vec<f64>>,
    lm: Vec<f64>,
    z: Vec<Vec<f64>>,
}

/// Residuals of the first K-1 equations, observation-major.
fn residuals(sp: &Spec, d: &Data, th: &[f64]) -> Vec<f64> {
    let p = sp.unpack(th);
    let km = sp.k - 1;
    let mut r = Vec::with_capacity(d.lm.len() * km);
    for t in 0..d.lm.len() {
        let w = sp.shares(&p, &d.lp[t], d.lm[t], &d.z[t]);
        for i in 0..km {
            r.push(d.w[t][i] - w[i]);
        }
    }
    r
}

fn sigma_of(r: &[f64], n: usize, km: usize) -> Matrix {
    let mut s = Matrix::zeros(km, km);
    for t in 0..n {
        for a in 0..km {
            for b in 0..km {
                s[(a, b)] += r[t * km + a] * r[t * km + b];
            }
        }
    }
    s.scale(1.0 / n as f64)
}

/// ML by iterated FGLS: (θ, V(θ), Σ, log likelihood, iterations).
fn fit(sp: &Spec, d: &Data, th0: Vec<f64>) -> Result<(Vec<f64>, Matrix, Matrix, f64, usize), String> {
    let n = d.lm.len();
    let km = sp.k - 1;
    let p = th0.len();
    let mut th = th0;
    let mut r = residuals(sp, d, &th);
    let mut sig = Matrix::identity(km);
    let obj = |r: &[f64]| -> f64 {
        let s = sigma_of(r, n, km);
        let det = s.det();
        if det > 0.0 { det.ln() } else { f64::INFINITY }
    };
    let mut cur = f64::INFINITY;
    let mut iters = 0;
    let mut info = Matrix::zeros(p, p);
    for it in 0..300 {
        iters = it + 1;
        let sinv = sig.inverse().map_err(|_| rc_err(430, "error covariance is singular"))?;
        // Jacobian of the fitted values (minus the residuals)
        let mut jac: Vec<Vec<f64>> = Vec::with_capacity(p);
        for c in 0..p {
            let h = 1e-6 * th[c].abs().max(1.0);
            let mut tp = th.clone();
            tp[c] += h;
            let rp = residuals(sp, d, &tp);
            let mut tm = th.clone();
            tm[c] -= h;
            let rm = residuals(sp, d, &tm);
            jac.push(rm.iter().zip(&rp).map(|(a, b)| (a - b) / (2.0 * h)).collect());
        }
        info = Matrix::zeros(p, p);
        let mut g = vec![0.0; p];
        for t in 0..n {
            for a in 0..p {
                // J_t' S
                let js: Vec<f64> = (0..km).map(|e| (0..km).map(|f| jac[a][t * km + f] * sinv[(f, e)]).sum()).collect();
                for e in 0..km {
                    g[a] += js[e] * r[t * km + e];
                }
                for b in a..p {
                    let v: f64 = (0..km).map(|e| js[e] * jac[b][t * km + e]).sum();
                    info[(a, b)] += v;
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                info[(a, b)] = info[(b, a)];
            }
        }
        let inv = info.inverse().map_err(|_| rc_err(430, "demand system not identified: singular information matrix"))?;
        let step: Vec<f64> = (0..p).map(|a| (0..p).map(|b| inv[(a, b)] * g[b]).sum()).collect();
        // step halving on the concentrated likelihood
        let mut lam = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let tn: Vec<f64> = th.iter().zip(&step).map(|(a, s)| a + lam * s).collect();
            let rn = residuals(sp, d, &tn);
            let o = obj(&rn);
            if o.is_finite() && (o <= cur + 1e-12 || !cur.is_finite()) {
                th = tn;
                r = rn;
                accepted = true;
                break;
            }
            lam *= 0.5;
        }
        let new = obj(&r);
        sig = sigma_of(&r, n, km);
        let small = step.iter().map(|s| (lam * s).abs()).fold(0.0, f64::max) < 1e-9;
        if !accepted || ((cur - new).abs() < 1e-11 && small) {
            cur = new;
            break;
        }
        cur = new;
    }
    if !cur.is_finite() {
        return Err(rc_err(430, "demand system: convergence not achieved"));
    }
    let v = info.inverse().map_err(|_| rc_err(430, "singular information matrix"))?;
    let ll = -(n as f64) * km as f64 / 2.0 * (1.0 + (2.0 * std::f64::consts::PI).ln()) - n as f64 / 2.0 * cur;
    Ok((th, v, sig, ll, iters))
}

fn opt_vars(ds: &DataSet, cmd: &Command, name: &str, min: usize) -> Result<Vec<String>, String> {
    cmd.opt(name, min).map(|v| varlist::expand(ds, v)).transpose().map(|v| v.unwrap_or_default())
}

pub struct Demandsys;

impl CommandHandler for Demandsys {
    fn name(&self) -> &'static str {
        "demandsys"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (mname, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let model = Model::parse(&mname.to_lowercase()).ok_or_else(|| rc_err(198, format!("demand system {mname} unknown; aids, quaids, translog or rotterdam")))?;
        let shares = varlist::expand(ds, rest)?;
        let k = shares.len();
        if k < 2 {
            return Err(rc_err(102, "at least two budget shares required"));
        }
        let prices = opt_vars(ds, cmd, "prices", 2)?;
        if prices.len() != k {
            return Err(rc_err(198, "prices() requires one price per budget share"));
        }
        let expv = opt_vars(ds, cmd, "expenditure", 3)?;
        if expv.len() != 1 {
            return Err(rc_err(198, "expenditure() requires one variable"));
        }
        let demo = opt_vars(ds, cmd, "demographics", 4)?;
        if model == Model::Rotterdam && !demo.is_empty() {
            return Err(rc_err(198, "demographics() not allowed with rotterdam"));
        }
        let homog = !cmd.has_opt("nohomogeneity", 6);
        let sym = (homog || model == Model::Translog) && !cmd.has_opt("nosymmetry", 5);
        let alpha0: f64 = cmd.opt("alpha0", 6).map(|s| s.trim().parse::<f64>().map_err(|_| rc_err(198, "alpha0() must be a number"))).transpose()?.unwrap_or(0.0);
        let cons = !(model == Model::Rotterdam && cmd.has_opt("noconstant", 5));
        let sp = Spec { model, k, nz: demo.len(), homog, sym, alpha0, cons };
        let level = util::level(cmd, st)?;
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = shares.clone();
        all.extend(prices.iter().cloned());
        all.extend(expv.iter().cloned());
        all.extend(demo.iter().cloned());
        util::markout(ds, &mut mask, &all);
        let wc: Vec<Vec<f64>> = shares.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let pc: Vec<Vec<f64>> = prices.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mc = util::col(ds, &expv[0])?;
        let zc: Vec<Vec<f64>> = demo.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        for i in 0..ds.n_obs {
            if mask[i] && (pc.iter().any(|c| c[i] <= 0.0) || mc[i] <= 0.0 || (model == Model::Rotterdam && wc.iter().any(|c| c[i] <= 0.0))) {
                mask[i] = false;
            }
        }
        for i in util::rows(&mask) {
            let s: f64 = wc.iter().map(|c| c[i]).sum();
            if (s - 1.0).abs() > 1e-4 {
                return Err(rc_err(459, format!("budget shares do not sum to one in observation {}", i + 1)));
            }
        }
        let d = if model == Model::Rotterdam {
            // pairs (t, t-1) of the tsset time variable within panels
            let (panel, time) = st.tsset.clone().ok_or_else(|| rc_err(111, "rotterdam requires time-series data: use tsset"))?;
            let tv = util::col(ds, &time)?;
            let pv = panel.as_ref().map(|p| util::col(ds, p)).transpose()?;
            let key = |i: usize| (pv.as_ref().map_or(0.0, |p| p[i]).to_bits(), tv[i].to_bits());
            let mut at: std::collections::HashMap<(u64, u64), usize> = Default::default();
            for i in util::rows(&mask) {
                at.insert(key(i), i);
            }
            let mut d = Data { w: Vec::new(), lp: Vec::new(), lm: Vec::new(), z: Vec::new() };
            let mut used = vec![false; ds.n_obs];
            for i in util::rows(&mask) {
                let prev = (pv.as_ref().map_or(0.0, |p| p[i]).to_bits(), (tv[i] - 1.0).to_bits());
                let Some(&j) = at.get(&prev) else { continue };
                let dlp: Vec<f64> = pc.iter().map(|c| c[i].ln() - c[j].ln()).collect();
                let dlm = mc[i].ln() - mc[j].ln();
                let y: Vec<f64> = (0..k).map(|g| {
                    let wbar = 0.5 * (wc[g][i] + wc[g][j]);
                    wbar * (wc[g][i].ln() - wc[g][j].ln() + dlm - dlp[g])
                }).collect();
                d.lm.push(y.iter().sum());
                d.w.push(y);
                d.lp.push(dlp);
                d.z.push(Vec::new());
                used[i] = true;
            }
            mask = used;
            d
        } else {
            let rows = util::rows(&mask);
            Data {
                w: rows.iter().map(|&i| wc.iter().map(|c| c[i]).collect()).collect(),
                lp: rows.iter().map(|&i| pc.iter().map(|c| c[i].ln()).collect()).collect(),
                lm: rows.iter().map(|&i| mc[i].ln()).collect(),
                z: rows.iter().map(|&i| zc.iter().map(|c| c[i]).collect()).collect(),
            }
        };
        let n = d.lm.len();
        if n <= sp.n_free() / (k - 1) + 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // starting values: constant shares (Rotterdam: θ at the mean shares)
        let mean_w: Vec<f64> = if model == Model::Rotterdam {
            (0..k).map(|i| util::rows(&mask).iter().map(|&t| wc[i][t]).sum::<f64>() / n as f64).collect()
        } else {
            (0..k).map(|i| d.w.iter().map(|w| w[i]).sum::<f64>() / n as f64).collect()
        };
        let mut th0 = vec![0.0; sp.n_free()];
        let km = k - 1;
        let a_len = if cons { km } else { 0 };
        for i in 0..km {
            match model {
                Model::Translog => th0[i] = -mean_w[i],
                Model::Rotterdam => th0[a_len + i] = mean_w[i],
                _ => th0[i] = mean_w[i],
            }
        }
        // translog: away from β = 0, where the direction β ∝ αα' is not
        // identified to first order
        if model == Model::Translog {
            let mut pos = km + km * sp.nz;
            for i in 0..k {
                for j in 0..k {
                    if sym && j < i {
                        continue;
                    }
                    if i == j {
                        th0[pos] = -0.05 * mean_w[i];
                    }
                    pos += 1;
                }
            }
        }
        let (th, vth, sig, ll, iters) = fit(&sp, &d, th0)?;
        // full parameter vector and its covariance through the linear map
        let par = sp.unpack(&th);
        let full = sp.full(&par);
        let names = sp.names(&shares, &prices, &demo);
        let pf = full.len();
        let p = th.len();
        let mut a = Matrix::zeros(pf, p);
        for c in 0..p {
            let mut t1 = th.clone();
            t1[c] += 1.0;
            let f1 = sp.full(&sp.unpack(&t1));
            for r in 0..pf {
                a[(r, c)] = f1[r] - full[r];
            }
        }
        let vf = a.mul(&vth)?.mul(&a.transpose())?;
        let vf = Matrix::from_rows(&(0..pf).map(|i| (0..pf).map(|j| 0.5 * (vf[(i, j)] + vf[(j, i)])).collect()).collect::<Vec<_>>());
        // display
        let mut out = vec![String::new(), sp.model.title().to_string(), String::new()];
        out.push(format!("{:<48}Number of obs     = {:>10}", "", n));
        out.push(format!("{:<48}Number of goods   = {:>10}", "", k));
        out.push(format!("Log likelihood = {:<31}Iterations        = {:>10}", fmt_g(ll, 9), iters));
        let restr: Vec<&str> = [(homog && model != Model::Translog).then_some("homogeneity"), sym.then_some("symmetry")].into_iter().flatten().collect();
        out.push(format!("Restrictions: adding-up{}", restr.iter().map(|r| format!(", {r}")).collect::<String>()));
        if !demo.is_empty() {
            out.push(format!("Demographics: {} (translating)", demo.join(" ")));
        }
        out.push(String::new());
        let omitted = vec![false; pf];
        let ts = est::TableSpec { depvar: "", names: &names, b: &full, v: &vf, omitted: &omitted, is_base: &omitted, df: None, level, eform: None, coef_title: "Coefficient" };
        out.extend(est::coef_table(&ts));
        if matches!(model, Model::Aids | Model::Quaids) {
            out.push(format!("Note: alpha_0 = {} (ln a(p) intercept).", fmt_g(alpha0, 8)));
        }
        // e()
        let e = st.begin_e();
        e.set_str("cmd", "demandsys");
        e.set_str("model", model.name());
        e.set_str("title", model.title());
        e.set_str("depvar", shares.join(" "));
        e.set_str("shares", shares.join(" "));
        e.set_str("prices", prices.join(" "));
        e.set_str("expenditure", expv[0].clone());
        e.set_str("demographics", demo.join(" "));
        e.set_str("homogeneity", if homog { "imposed" } else { "" });
        e.set_str("symmetry", if sym { "imposed" } else { "" });
        e.set_str("constant", if cons { "" } else { "noconstant" });
        e.set_str("vce", "oim");
        e.set_str("properties", "b V");
        e.set("N", n as f64);
        e.set("ll", ll);
        e.set("k_eq", (k - 1) as f64);
        e.set("ngoods", k as f64);
        e.set("alpha0", alpha0);
        e.set("converged", 1.0);
        e.set("ic", iters as f64);
        e.set("rank", p as f64);
        e.set_mat("b", Matrix::row_vector(&full).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", vf.with_names(names.clone(), names));
        let sn: Vec<String> = shares[..k - 1].to_vec();
        e.set_mat("Sigma", sig.with_names(sn.clone(), sn));
        e.sample = Some(mask);
        Ok(out.join("\n"))
    }
}

struct Fitted {
    sp: Spec,
    shares: Vec<String>,
    prices: Vec<String>,
    expv: String,
    demo: Vec<String>,
    par: Params,
}

fn spec_from_e(e: &Results) -> Result<Fitted, String> {
    let model = e.macros.get("model").and_then(|m| Model::parse(m)).ok_or_else(|| rc_err(301, "demandsys results not found"))?;
    let words = |k: &str| -> Vec<String> { e.macros.get(k).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default() };
    let shares = words("shares");
    let prices = words("prices");
    let demo = words("demographics");
    let sp = Spec {
        model,
        k: shares.len(),
        nz: demo.len(),
        homog: e.macros.get("homogeneity").is_some_and(|s| !s.is_empty()),
        sym: e.macros.get("symmetry").is_some_and(|s| !s.is_empty()),
        alpha0: e.get("alpha0").unwrap_or(0.0),
        cons: e.macros.get("constant").map_or(true, |s| s.is_empty()),
    };
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let par = sp.from_full(&b.data[..b.cols]);
    Ok(Fitted { sp, shares, prices, expv: e.macros.get("expenditure").cloned().unwrap_or_default(), demo, par })
}

/// Elasticities at log prices `lp`, log expenditure `lm` and demographics
/// `z`: (shares, expenditure, uncompensated K x K, compensated K x K).
fn elasticities(sp: &Spec, p: &Params, lp: &[f64], lm: f64, z: &[f64]) -> (Vec<f64>, Vec<f64>, Matrix, Matrix) {
    let k = sp.k;
    let w = sp.shares(p, lp, lm, z);
    let h = 1e-5;
    let dm: Vec<f64> = {
        let a = sp.shares(p, lp, lm + h, z);
        let b = sp.shares(p, lp, lm - h, z);
        (0..k).map(|i| (a[i] - b[i]) / (2.0 * h)).collect()
    };
    let eta: Vec<f64> = (0..k).map(|i| 1.0 + dm[i] / w[i]).collect();
    let mut un = Matrix::zeros(k, k);
    for j in 0..k {
        let mut up = lp.to_vec();
        up[j] += h;
        let mut dn = lp.to_vec();
        dn[j] -= h;
        let (a, b) = (sp.shares(p, &up, lm, z), sp.shares(p, &dn, lm, z));
        for i in 0..k {
            un[(i, j)] = -if i == j { 1.0 } else { 0.0 } + (a[i] - b[i]) / (2.0 * h) / w[i];
        }
    }
    let mut co = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            co[(i, j)] = un[(i, j)] + eta[i] * w[j];
        }
    }
    (w, eta, un, co)
}

/// Rotterdam elasticities at the shares `w`.
fn rotterdam_elasticities(sp: &Spec, p: &Params, w: &[f64]) -> (Vec<f64>, Matrix, Matrix) {
    let k = sp.k;
    let eta: Vec<f64> = (0..k).map(|i| p.beta[i] / w[i]).collect();
    let mut co = Matrix::zeros(k, k);
    let mut un = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            co[(i, j)] = p.g[i * k + j] / w[i];
            un[(i, j)] = co[(i, j)] - eta[i] * w[j];
        }
    }
    (eta, un, co)
}

/// `estat elasticities [, expenditure uncompensated compensated]` after
/// `demandsys`: elasticities at the sample means of log prices, log
/// expenditure and demographics (Rotterdam: at the mean shares).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("demandsys") {
        return None;
    }
    if !(sub.len() >= 4 && "elasticities".starts_with(sub)) {
        return None;
    }
    Some((|| {
        let f = spec_from_e(&st.e)?;
        let mask = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
        let rows = util::rows(&mask);
        let mean_of = |v: &str, log: bool| -> Result<f64, String> {
            let c = util::col(ds, v)?;
            let xs: Vec<f64> = rows.iter().map(|&i| c[i]).filter(|x| !is_missing(*x) && (!log || *x > 0.0)).map(|x| if log { x.ln() } else { x }).collect();
            Ok(xs.iter().sum::<f64>() / xs.len() as f64)
        };
        let (w, eta, un, co, title) = if f.sp.model == Model::Rotterdam {
            let w: Vec<f64> = f.shares.iter().map(|s| mean_of(s, false)).collect::<Result<_, _>>()?;
            let (eta, un, co) = rotterdam_elasticities(&f.sp, &f.par, &w);
            (w, eta, un, co, "Elasticities at the mean budget shares")
        } else {
            let lp: Vec<f64> = f.prices.iter().map(|p| mean_of(p, true)).collect::<Result<_, _>>()?;
            let lm = mean_of(&f.expv, true)?;
            let z: Vec<f64> = f.demo.iter().map(|v| mean_of(v, false)).collect::<Result<_, _>>()?;
            let (w, eta, un, co) = elasticities(&f.sp, &f.par, &lp, lm, &z);
            (w, eta, un, co, "Elasticities at the means of log prices and log expenditure")
        };
        let (shares, prices) = (&f.shares, &f.prices);
        let any = ["expenditure", "uncompensated", "compensated"].iter().any(|o| cmd.has_opt(o, 3));
        let show = |o: &str| !any || cmd.has_opt(o, 3);
        let mut out = vec![String::new(), title.to_string()];
        let row = |name: &str, v: &[f64]| format!("{:>12} |{}", name, v.iter().map(|x| format!(" {:>10}", fmt_g(*x, 7))).collect::<String>());
        let head = |cols: &[String]| format!("{:>12} |{}", "", cols.iter().map(|c| format!(" {:>10}", crate::summarize::abbrev(c, 10))).collect::<String>());
        out.push(String::new());
        out.push(head(shares));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11 * shares.len())));
        out.push(row(if f.sp.model == Model::Rotterdam { "mean share" } else { "fitted share" }, &w));
        if show("expenditure") {
            out.push(row("expenditure", &eta));
        }
        for (title, m, on) in [("Uncompensated price elasticities", &un, show("uncompensated")), ("Compensated price elasticities", &co, show("compensated"))] {
            if !on {
                continue;
            }
            out.push(String::new());
            out.push(title.to_string());
            out.push(head(prices));
            out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11 * prices.len())));
            for i in 0..f.sp.k {
                out.push(row(&shares[i], m.row(i)));
            }
        }
        let mut r = Results::new();
        r.set_mat("shares", Matrix::row_vector(&w).with_names(vec!["share".into()], shares.clone()));
        r.set_mat("expenditure", Matrix::row_vector(&eta).with_names(vec!["expenditure".into()], shares.clone()));
        r.set_mat("uncompensated", un.with_names(shares.clone(), prices.clone()));
        r.set_mat("compensated", co.with_names(shares.clone(), prices.clone()));
        st.r = r;
        Ok(out.join("\n"))
    })())
}

/// `predict newvar [, equation(share)]` after `demandsys`: fitted share.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("demandsys") {
        return None;
    }
    Some((|| {
        let f = spec_from_e(&st.e)?;
        if f.sp.model == Model::Rotterdam {
            return Err(rc_err(198, "predict is not available after demandsys rotterdam"));
        }
        let name = cmd.main.split_whitespace().last().ok_or_else(|| rc_err(198, "new variable name required"))?.to_string();
        let eq = cmd.opt("equation", 2).map(|s| s.trim().trim_start_matches('#').to_string()).unwrap_or_else(|| f.shares[0].clone());
        let i = f.shares.iter().position(|s| *s == eq).or_else(|| eq.parse::<usize>().ok().filter(|k| *k >= 1 && *k <= f.sp.k).map(|k| k - 1)).ok_or_else(|| rc_err(198, format!("equation {eq} not found")))?;
        let pc: Vec<Vec<f64>> = f.prices.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mc = util::col(ds, &f.expv)?;
        let zc: Vec<Vec<f64>> = f.demo.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mask = util::touse(ds, st, cmd)?;
        let vals: Vec<f64> = (0..ds.n_obs)
            .map(|t| {
                if !mask[t] || mc[t] <= 0.0 || is_missing(mc[t]) || pc.iter().any(|c| c[t] <= 0.0 || is_missing(c[t])) || zc.iter().any(|c| is_missing(c[t])) {
                    return ferrostat_core::missing::SYSMISS;
                }
                let lp: Vec<f64> = pc.iter().map(|c| c[t].ln()).collect();
                let z: Vec<f64> = zc.iter().map(|c| c[t]).collect();
                f.sp.shares(&f.par, &lp, mc[t].ln(), &z)[i]
            })
            .collect();
        if ds.get_var(&name).is_some() {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
        Ok(String::new())
    })())
}
