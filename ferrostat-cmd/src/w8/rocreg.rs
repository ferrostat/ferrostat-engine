//! `rocreg`: receiver operating characteristic regression, and
//! `rocregplot`.
//!
//! Every method starts from the *placement value* of each case: the
//! survivor function of the (covariate-adjusted) control distribution at
//! the case's classifier value, u = P(Y_control >= y_case) (ties counted
//! one half with `tiecorrected`). The ROC curve is the distribution
//! function of the placement values: ROC(t) = P(u <= t).
//!
//! * nonparametric (default): AUC = 1 - mean(u), ROC(t), the inverse
//!   ROC (false-positive rate reaching a true-positive rate) and the partial
//!   area int_0^t ROC(s) ds; standard errors and confidence intervals by
//!   bootstrap, resampling cases and controls separately (or clusters).
//! * `probit`: the binormal ROC model ROC(t) = Φ(i_cons + x'β + s_cons
//!   Φ⁻¹(t)) fitted as a probit regression of I(u <= t_k) on Φ⁻¹(t_k) and
//!   the ROC covariates over `fprpts()` false-positive rates (the GLM
//!   approach of Pepe 2000 and Alonso-Pepe), with bootstrap inference.
//! * `probit ml`: fully parametric maximum likelihood; controls
//!   Y = γ0 + z'γ + σ0 ε, cases Y = γ0 + z'γ + δ0 + x'δ + σ1 ε, ε ~ N(0,1),
//!   so ROC(t|x) = Φ((δ0 + x'δ)/σ1 + (σ0/σ1) Φ⁻¹(t)); estimated by
//!   iterated weighted least squares (the exact ML for this model), with
//!   delta-method standard errors from the information matrix.
//!
//! Covariate adjustment of the control distribution (`ctrlcov()`): within
//! strata of the covariates (`ctrlmodel(strata)`) or by a linear model of
//! the controls (`ctrlmodel(linear)`, the default) whose standardized
//! residuals are compared, empirically or under normality (`pvc(normal)`).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::varlist;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Stat {
    Auc,
    Roc(f64),
    InvRoc(f64),
    Pauc(f64),
}

impl Stat {
    fn name(&self, k: usize) -> String {
        match self {
            Stat::Auc => "AUC".into(),
            Stat::Roc(_) => format!("roc_{k}"),
            Stat::InvRoc(_) => format!("invroc_{k}"),
            Stat::Pauc(_) => format!("pauc_{k}"),
        }
    }
}

#[derive(Clone)]
struct Opts {
    ctrlcov: Vec<String>,
    strata: bool,
    pvc_normal: bool,
    tiecorr: bool,
    probit: bool,
    ml: bool,
    roccov: Vec<String>,
    fprpts: usize,
    stats: Vec<Stat>,
    breps: usize,
    boot: bool,
    cluster: Option<String>,
}

/// One observation of the estimation sample.
#[derive(Clone)]
struct Obs {
    case: bool,
    y: f64,
    z: Vec<f64>,
    x: Vec<f64>,
    cl: f64,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Least squares of y on [1, columns]; returns coefficients and the
/// residual standard deviation.
fn ols(rows: &[(f64, Vec<f64>)]) -> Result<(Vec<f64>, f64), String> {
    let k = rows.first().map_or(1, |r| r.1.len() + 1);
    let n = rows.len();
    if n <= k {
        return Err(rc_err(2001, "insufficient control observations for the control model"));
    }
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for (y, z) in rows {
        let x: Vec<f64> = std::iter::once(1.0).chain(z.iter().copied()).collect();
        for a in 0..k {
            xty[a] += x[a] * y;
            for b in 0..k {
                xtx[(a, b)] += x[a] * x[b];
            }
        }
    }
    let inv = xtx.inverse().map_err(|_| rc_err(498, "control covariates are collinear"))?;
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv[(a, b)] * xty[b]).sum()).collect();
    let rss: f64 = rows
        .iter()
        .map(|(y, z)| {
            let f = beta[0] + z.iter().zip(&beta[1..]).map(|(a, b)| a * b).sum::<f64>();
            (y - f).powi(2)
        })
        .sum();
    Ok((beta, (rss / (n - k) as f64).sqrt()))
}

/// Survivor function of `ctrl` at x (ties one half when `tie`).
fn surv(ctrl: &[f64], x: f64, tie: bool) -> f64 {
    let gt = ctrl.iter().filter(|&&c| c > x).count() as f64;
    let eq = ctrl.iter().filter(|&&c| c == x).count() as f64;
    (gt + if tie { 0.5 * eq } else { eq }) / ctrl.len() as f64
}

/// Placement values of the cases among `idx` (in the order of the cases).
fn placements(obs: &[Obs], idx: &[usize], o: &Opts) -> Result<Vec<f64>, String> {
    let ctrl: Vec<usize> = idx.iter().copied().filter(|&i| !obs[i].case).collect();
    let cases: Vec<usize> = idx.iter().copied().filter(|&i| obs[i].case).collect();
    if ctrl.len() < 2 || cases.is_empty() {
        return Err(rc_err(2000, "insufficient observations: cases and controls are required"));
    }
    if o.ctrlcov.is_empty() {
        let cy: Vec<f64> = ctrl.iter().map(|&i| obs[i].y).collect();
        if o.pvc_normal {
            let m = mean(&cy);
            let sd = (cy.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (cy.len() - 1) as f64).sqrt();
            return Ok(cases.iter().map(|&i| 1.0 - dist::normal((obs[i].y - m) / sd)).collect());
        }
        return Ok(cases.iter().map(|&i| surv(&cy, obs[i].y, o.tiecorr)).collect());
    }
    if o.strata {
        let mut out = Vec::with_capacity(cases.len());
        for &i in &cases {
            let cy: Vec<f64> = ctrl.iter().filter(|&&j| obs[j].z == obs[i].z).map(|&j| obs[j].y).collect();
            if cy.is_empty() {
                return Err(rc_err(2000, "a case covariate stratum has no controls"));
            }
            if o.pvc_normal {
                if cy.len() < 2 {
                    return Err(rc_err(2001, "a control stratum has fewer than two observations"));
                }
                let m = mean(&cy);
                let sd = (cy.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (cy.len() - 1) as f64).sqrt();
                out.push(1.0 - dist::normal((obs[i].y - m) / sd));
            } else {
                out.push(surv(&cy, obs[i].y, o.tiecorr));
            }
        }
        return Ok(out);
    }
    let rows: Vec<(f64, Vec<f64>)> = ctrl.iter().map(|&i| (obs[i].y, obs[i].z.clone())).collect();
    let (beta, sd) = ols(&rows)?;
    let std = |i: usize| (obs[i].y - beta[0] - obs[i].z.iter().zip(&beta[1..]).map(|(a, b)| a * b).sum::<f64>()) / sd;
    if o.pvc_normal {
        return Ok(cases.iter().map(|&i| 1.0 - dist::normal(std(i))).collect());
    }
    let cr: Vec<f64> = ctrl.iter().map(|&i| std(i)).collect();
    Ok(cases.iter().map(|&i| surv(&cr, std(i), o.tiecorr)).collect())
}

fn quantile_sorted(v: &[f64], p: f64) -> f64 {
    let n = v.len();
    let k = ((p * n as f64).ceil() as usize).clamp(1, n);
    v[k - 1]
}

/// Nonparametric statistics from placement values.
fn np_stats(u: &[f64], stats: &[Stat]) -> Vec<f64> {
    let n = u.len() as f64;
    let mut s = u.to_vec();
    s.sort_by(f64::total_cmp);
    stats
        .iter()
        .map(|st| match *st {
            Stat::Auc => 1.0 - mean(u),
            Stat::Roc(t) => u.iter().filter(|&&x| x <= t).count() as f64 / n,
            Stat::InvRoc(r) => quantile_sorted(&s, r),
            Stat::Pauc(t) => u.iter().map(|&x| (t - x).max(0.0)).sum::<f64>() / n,
        })
        .collect()
}

/// Probit regression by Fisher scoring: (coefficients, covariance).
fn probit(rows: &[(f64, Vec<f64>)]) -> Result<(Vec<f64>, Matrix), String> {
    let k = rows[0].1.len();
    let mut b = vec![0.0; k];
    let mut info = Matrix::zeros(k, k);
    for _ in 0..200 {
        let mut g = vec![0.0; k];
        info = Matrix::zeros(k, k);
        for (y, x) in rows {
            let xb: f64 = x.iter().zip(&b).map(|(a, c)| a * c).sum();
            let p = dist::normal(xb).clamp(1e-12, 1.0 - 1e-12);
            let f = dist::normalden(xb);
            let r = if *y > 0.5 { f / p } else { -f / (1.0 - p) };
            let w = f * f / (p * (1.0 - p));
            for a in 0..k {
                g[a] += r * x[a];
                for c in 0..k {
                    info[(a, c)] += w * x[a] * x[c];
                }
            }
        }
        let inv = info.inverse().map_err(|_| rc_err(430, "probit ROC model: information matrix is singular"))?;
        let step: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * g[c]).sum()).collect();
        let mx = step.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        let damp = if mx > 5.0 { 5.0 / mx } else { 1.0 };
        for a in 0..k {
            b[a] += damp * step[a];
        }
        if mx < 1e-10 {
            break;
        }
    }
    if b.iter().any(|x| !x.is_finite()) {
        return Err(rc_err(430, "probit ROC model did not converge"));
    }
    let v = info.inverse().unwrap_or_else(|_| Matrix::zeros(k, k));
    Ok((b, v))
}

/// Parametric (GLM) estimates: roccov coefficients, i_cons, s_cons, and the
/// AUC when there are no ROC covariates.
fn glm_stats(obs: &[Obs], idx: &[usize], u: &[f64], o: &Opts) -> Result<Vec<f64>, String> {
    let cases: Vec<usize> = idx.iter().copied().filter(|&i| obs[i].case).collect();
    let ts: Vec<f64> = (1..=o.fprpts).map(|k| k as f64 / (o.fprpts + 1) as f64).collect();
    let mut rows = Vec::with_capacity(cases.len() * ts.len());
    for (ci, &i) in cases.iter().enumerate() {
        for &t in &ts {
            let mut x = obs[i].x.clone();
            x.push(1.0);
            x.push(dist::invnormal(t));
            rows.push((if u[ci] <= t { 1.0 } else { 0.0 }, x));
        }
    }
    let (b, _) = probit(&rows)?;
    let mut out = b.clone();
    if o.roccov.is_empty() {
        let (a, s) = (b[b.len() - 2], b[b.len() - 1]);
        out.push(dist::normal(a / (1.0 + s * s).sqrt()));
    }
    Ok(out)
}

fn stat_names(o: &Opts) -> Vec<String> {
    if o.probit {
        let mut v: Vec<String> = o.roccov.clone();
        v.push("i_cons".into());
        v.push("s_cons".into());
        if o.roccov.is_empty() {
            v.push("AUC".into());
        }
        return v;
    }
    let mut cnt = [0usize; 4];
    o.stats
        .iter()
        .map(|s| {
            let j = match s {
                Stat::Auc => 0,
                Stat::Roc(_) => 1,
                Stat::InvRoc(_) => 2,
                Stat::Pauc(_) => 3,
            };
            cnt[j] += 1;
            s.name(cnt[j])
        })
        .collect()
}

fn estimates(obs: &[Obs], idx: &[usize], o: &Opts) -> Result<Vec<f64>, String> {
    let u = placements(obs, idx, o)?;
    if o.probit {
        glm_stats(obs, idx, &u, o)
    } else {
        Ok(np_stats(&u, &o.stats))
    }
}

/// One bootstrap sample: cases and controls resampled separately, or whole
/// clusters within case status.
fn resample(st: &Store, obs: &[Obs], o: &Opts) -> Vec<usize> {
    let mut rng = st.rng.borrow_mut();
    let mut out = Vec::with_capacity(obs.len());
    for status in [false, true] {
        let grp: Vec<usize> = (0..obs.len()).filter(|&i| obs[i].case == status).collect();
        if o.cluster.is_some() {
            let mut ids: Vec<f64> = grp.iter().map(|&i| obs[i].cl).collect();
            ids.sort_by(f64::total_cmp);
            ids.dedup();
            for _ in 0..ids.len() {
                let c = ids[rng.uniform_int(0.0, ids.len() as f64 - 1.0) as usize];
                out.extend(grp.iter().copied().filter(|&i| obs[i].cl == c));
            }
        } else {
            for _ in 0..grp.len() {
                out.push(grp[rng.uniform_int(0.0, grp.len() as f64 - 1.0) as usize]);
            }
        }
    }
    out
}

struct Boot {
    bias: Vec<f64>,
    se: Vec<f64>,
    normal: Vec<(f64, f64)>,
    pct: Vec<(f64, f64)>,
    bc: Vec<(f64, f64)>,
    v: Matrix,
    reps: usize,
}

fn bootstrap(st: &Store, obs: &[Obs], o: &Opts, est: &[f64], level: f64) -> Result<Boot, String> {
    let k = est.len();
    let mut draws: Vec<Vec<f64>> = Vec::with_capacity(o.breps);
    for _ in 0..o.breps {
        let idx = resample(st, obs, o);
        if let Ok(e) = estimates(obs, &idx, o) {
            if e.len() == k && e.iter().all(|x| x.is_finite()) {
                draws.push(e);
            }
        }
    }
    let b = draws.len();
    if b < 2 {
        return Err(rc_err(2000, "insufficient bootstrap replications"));
    }
    let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let alpha = (1.0 - level / 100.0) / 2.0;
    let mut out = Boot { bias: vec![0.0; k], se: vec![0.0; k], normal: Vec::new(), pct: Vec::new(), bc: Vec::new(), v: Matrix::zeros(k, k), reps: b };
    let m: Vec<f64> = (0..k).map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / b as f64).collect();
    for a in 0..k {
        for c in 0..k {
            out.v[(a, c)] = draws.iter().map(|d| (d[a] - m[a]) * (d[c] - m[c])).sum::<f64>() / (b - 1) as f64;
        }
    }
    for j in 0..k {
        out.bias[j] = m[j] - est[j];
        out.se[j] = out.v[(j, j)].sqrt();
        out.normal.push((est[j] - z * out.se[j], est[j] + z * out.se[j]));
        let mut s: Vec<f64> = draws.iter().map(|d| d[j]).collect();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let h = p * (b - 1) as f64;
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(b - 1);
            s[lo] + (h - lo as f64) * (s[hi] - s[lo])
        };
        out.pct.push((q(alpha), q(1.0 - alpha)));
        let below = s.iter().filter(|&&x| x < est[j]).count() as f64 + 0.5 * s.iter().filter(|&&x| x == est[j]).count() as f64;
        let p0 = (below / b as f64).clamp(0.5 / b as f64, 1.0 - 0.5 / b as f64);
        let z0 = dist::invnormal(p0);
        out.bc.push((q(dist::normal(2.0 * z0 - z)), q(dist::normal(2.0 * z0 + z))));
    }
    Ok(out)
}

fn parse_opts(ds: &DataSet, cmd: &Command) -> Result<Opts, String> {
    let list = |n: &str, min: usize| -> Result<Vec<String>, String> { cmd.opt(n, min).map(|v| varlist::expand(ds, v)).transpose().map(|v| v.unwrap_or_default()) };
    let nums = |n: &str, min: usize| -> Result<Vec<f64>, String> { cmd.opt(n, min).map(crate::margins::numlist).transpose().map(|v| v.unwrap_or_default()) };
    let probit = cmd.has_opt("probit", 4);
    let ml = cmd.has_opt("ml", 2);
    let mut stats = Vec::new();
    for t in nums("roc", 3)? {
        stats.push(Stat::Roc(t));
    }
    for t in nums("invroc", 4)? {
        stats.push(Stat::InvRoc(t));
    }
    for t in nums("pauc", 4)? {
        stats.push(Stat::Pauc(t));
    }
    for s in &stats {
        let (Stat::Roc(t) | Stat::InvRoc(t) | Stat::Pauc(t)) = *s else { continue };
        if !(t > 0.0 && t < 1.0) && !(matches!(s, Stat::Pauc(_)) && t == 1.0) {
            return Err(rc_err(125, "roc(), invroc() and pauc() values must be between 0 and 1"));
        }
    }
    if stats.is_empty() || cmd.has_opt("auc", 3) {
        stats.insert(0, Stat::Auc);
    }
    let ctrlmodel = cmd.opt("ctrlmodel", 5).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "linear".into());
    if !matches!(ctrlmodel.as_str(), "linear" | "strata") {
        return Err(rc_err(198, format!("ctrlmodel({ctrlmodel}) invalid; strata or linear")));
    }
    let pvc = cmd.opt("pvc", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "empirical".into());
    if !matches!(pvc.as_str(), "empirical" | "normal") {
        return Err(rc_err(198, format!("pvc({pvc}) invalid; empirical or normal")));
    }
    let o = Opts {
        ctrlcov: list("ctrlcov", 5)?,
        strata: ctrlmodel == "strata",
        pvc_normal: pvc == "normal",
        tiecorr: cmd.has_opt("tiecorrected", 3),
        probit,
        ml,
        roccov: list("roccov", 4)?,
        fprpts: cmd.opt("fprpts", 4).map(|s| s.trim().parse::<usize>().map_err(|_| rc_err(198, "fprpts() must be a positive integer"))).transpose()?.unwrap_or(10),
        stats,
        breps: cmd.opt("breps", 3).map(|s| s.trim().parse::<usize>().map_err(|_| rc_err(198, "breps() must be a positive integer"))).transpose()?.unwrap_or(1000),
        boot: !cmd.has_opt("nobootstrap", 6),
        cluster: cmd.opt("cluster", 2).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
    };
    if o.ml && !o.probit {
        return Err(rc_err(198, "option ml requires probit"));
    }
    if !o.roccov.is_empty() && !o.probit {
        return Err(rc_err(198, "roccov() requires probit"));
    }
    if o.fprpts < 2 {
        return Err(rc_err(198, "fprpts() must be at least 2"));
    }
    Ok(o)
}

/// Estimation sample for one classifier.
fn sample(ds: &DataSet, mask: &[bool], refvar: &str, class: &str, o: &Opts) -> Result<(Vec<Obs>, Vec<bool>), String> {
    let r = util::col(ds, refvar)?;
    let y = util::col(ds, class)?;
    let zc: Vec<Vec<f64>> = o.ctrlcov.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let xc: Vec<Vec<f64>> = o.roccov.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let cl = match &o.cluster {
        Some(c) => Some(util::col(ds, c)?),
        None => None,
    };
    let mut used = vec![false; ds.n_obs];
    let mut obs = Vec::new();
    for i in 0..ds.n_obs {
        if !mask[i] || is_missing(r[i]) || is_missing(y[i]) || zc.iter().chain(&xc).any(|c| is_missing(c[i])) || cl.as_ref().is_some_and(|c| is_missing(c[i])) {
            continue;
        }
        used[i] = true;
        obs.push(Obs { case: r[i] != 0.0, y: y[i], z: zc.iter().map(|c| c[i]).collect(), x: xc.iter().map(|c| c[i]).collect(), cl: cl.as_ref().map_or(i as f64, |c| c[i]) });
    }
    if !obs.iter().any(|o| o.case) || !obs.iter().any(|o| !o.case) {
        return Err(rc_err(2000, format!("{refvar} does not vary: cases and controls are required")));
    }
    Ok((obs, used))
}

/// Parametric ML fit for one classifier: (parameter names, ROC-scale
/// estimates, their covariance).
fn ml_fit(obs: &[Obs], o: &Opts) -> Result<(Vec<String>, Vec<f64>, Matrix, f64), String> {
    let (nz, nx) = (o.ctrlcov.len(), o.roccov.len());
    // regressors: 1, z (common), case, case*x
    let design = |ob: &Obs| -> Vec<f64> {
        let c = if ob.case { 1.0 } else { 0.0 };
        let mut v = vec![1.0];
        v.extend(ob.z.iter().copied());
        v.push(c);
        v.extend(ob.x.iter().map(|x| x * c));
        v
    };
    let k = 2 + nz + nx;
    let n0 = obs.iter().filter(|o| !o.case).count() as f64;
    let n1 = obs.iter().filter(|o| o.case).count() as f64;
    if n0 < 2.0 || n1 < 2.0 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let (mut s0, mut s1) = (1.0f64, 1.0f64);
    let mut beta = vec![0.0; k];
    let mut xtx = Matrix::zeros(k, k);
    let mut ll_old = f64::NEG_INFINITY;
    for _ in 0..500 {
        xtx = Matrix::zeros(k, k);
        let mut xty = vec![0.0; k];
        for ob in obs {
            let w = 1.0 / if ob.case { s1 * s1 } else { s0 * s0 };
            let x = design(ob);
            for a in 0..k {
                xty[a] += w * x[a] * ob.y;
                for b in 0..k {
                    xtx[(a, b)] += w * x[a] * x[b];
                }
            }
        }
        let inv = xtx.inverse().map_err(|_| rc_err(498, "the parametric ROC model is not identified (collinear covariates)"))?;
        beta = (0..k).map(|a| (0..k).map(|b| inv[(a, b)] * xty[b]).sum()).collect();
        let (mut r0, mut r1) = (0.0, 0.0);
        for ob in obs {
            let e = ob.y - design(ob).iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>();
            if ob.case {
                r1 += e * e;
            } else {
                r0 += e * e;
            }
        }
        s0 = (r0 / n0).sqrt();
        s1 = (r1 / n1).sqrt();
        if s0 == 0.0 || s1 == 0.0 {
            return Err(rc_err(430, "a group has no variation around the model"));
        }
        let ll = -0.5 * (n0 + n1) * (2.0 * std::f64::consts::PI).ln() - n0 * s0.ln() - n1 * s1.ln() - 0.5 * (n0 + n1);
        if (ll - ll_old).abs() < 1e-12 * ll.abs().max(1.0) {
            ll_old = ll;
            break;
        }
        ll_old = ll;
    }
    // information: beta block X'WX, ln sigma_g block 2 n_g
    let m = k + 2;
    let mut vtheta = Matrix::zeros(m, m);
    let inv = xtx.inverse().map_err(|_| rc_err(498, "singular information matrix"))?;
    for a in 0..k {
        for b in 0..k {
            vtheta[(a, b)] = inv[(a, b)];
        }
    }
    vtheta[(k, k)] = 1.0 / (2.0 * n0);
    vtheta[(k + 1, k + 1)] = 1.0 / (2.0 * n1);
    let mut theta = beta.clone();
    theta.push(s0.ln());
    theta.push(s1.ln());
    // ROC-scale parameters: x coefficients/σ1, i_cons = δ0/σ1, s_cons = σ0/σ1
    let g = |th: &[f64]| -> Vec<f64> {
        let s1 = th[k + 1].exp();
        let s0 = th[k].exp();
        let mut v: Vec<f64> = (0..nx).map(|j| th[2 + nz + j] / s1).collect();
        let a = th[1 + nz] / s1;
        let b = s0 / s1;
        v.push(a);
        v.push(b);
        if nx == 0 {
            v.push(dist::normal(a / (1.0 + b * b).sqrt()));
        }
        v
    };
    let est = g(&theta);
    let p = est.len();
    let mut jac = Matrix::zeros(p, m);
    for c in 0..m {
        let h = 1e-6 * theta[c].abs().max(1.0);
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[c] += h;
        tm[c] -= h;
        let (fp, fm) = (g(&tp), g(&tm));
        for r in 0..p {
            jac[(r, c)] = (fp[r] - fm[r]) / (2.0 * h);
        }
    }
    let v = jac.mul(&vtheta)?.mul(&jac.transpose())?;
    let mut names: Vec<String> = o.roccov.clone();
    names.push("i_cons".into());
    names.push("s_cons".into());
    if nx == 0 {
        names.push("AUC".into());
    }
    Ok((names, est, v, ll_old))
}

pub struct Rocreg;

impl CommandHandler for Rocreg {
    fn name(&self) -> &'static str {
        "rocreg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "rocreg refvar classvar [classvar ...]"));
        }
        let o = parse_opts(ds, cmd)?;
        let level = util::level(cmd, st)?;
        if let Some(s) = cmd.opt("bseed", 3) {
            let s: u64 = s.trim().parse().map_err(|_| rc_err(198, "bseed() must be a nonnegative integer"))?;
            st.rng.borrow_mut().reseed(s);
        }
        let mask = util::touse(ds, st, cmd)?;
        let refvar = vars[0].clone();
        let classes = vars[1..].to_vec();
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut out: Vec<String> = Vec::new();
        let mut all_names: Vec<String> = Vec::new();
        let mut all_b: Vec<f64> = Vec::new();
        let mut blocks: Vec<Matrix> = Vec::new();
        let mut ci_n: Vec<(f64, f64)> = Vec::new();
        let mut ci_p: Vec<(f64, f64)> = Vec::new();
        let mut ci_bc: Vec<(f64, f64)> = Vec::new();
        let mut biases: Vec<f64> = Vec::new();
        let mut sample_all = vec![false; ds.n_obs];
        let mut n_obs = 0usize;
        let (mut n_case, mut n_ctrl) = (0usize, 0usize);
        let mut reps = 0usize;
        let mut ll_total = 0.0;
        let title = if o.ml {
            "Parametric ROC estimation (maximum likelihood)"
        } else if o.probit {
            "Parametric ROC estimation (GLM, probit link)"
        } else {
            "Nonparametric ROC estimation"
        };
        out.push(String::new());
        for class in &classes {
            let (obs, used) = sample(ds, &mask, &refvar, class, &o)?;
            for i in 0..ds.n_obs {
                sample_all[i] |= used[i];
            }
            n_obs = n_obs.max(obs.len());
            n_case = obs.iter().filter(|o| o.case).count();
            n_ctrl = obs.len() - n_case;
            let names = if o.ml { Vec::new() } else { stat_names(&o) };
            if o.ml {
                let (names, est, v, ll) = ml_fit(&obs, &o)?;
                ll_total += ll;
                out.push(format!("Status    : {refvar}"));
                out.push(format!("Classifier: {class}"));
                if !o.ctrlcov.is_empty() {
                    out.push(format!("Control covariates: {}", o.ctrlcov.join(" ")));
                }
                out.push(String::new());
                out.push(format!("{:>13}|{:>11} {:>10} {:>8} {:>7}   {:>22}", class, "Coef.", "Std. err.", "z", "P>|z|", format!("[{}% conf. interval]", fmt_g(level, 4))));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
                for (j, nm) in names.iter().enumerate() {
                    let se = v[(j, j)].max(0.0).sqrt();
                    let zz = est[j] / se;
                    out.push(format!("{:>13}|{:>11} {:>10} {:>8.2} {:>7.3}   {:>10} {:>10}", nm, fmt_g(est[j], 7), fmt_g(se, 7), zz, 2.0 * dist::normal(-zz.abs()), fmt_g(est[j] - z * se, 7), fmt_g(est[j] + z * se, 7)));
                    ci_n.push((est[j] - z * se, est[j] + z * se));
                    all_names.push(format!("{class}:{nm}"));
                }
                out.push("-".repeat(78));
                all_b.extend(est);
                blocks.push(v);
                continue;
            }
            let all: Vec<usize> = (0..obs.len()).collect();
            let est = estimates(&obs, &all, &o)?;
            out.push(format!("Status    : {refvar}"));
            out.push(format!("Classifier: {class}"));
            if !o.ctrlcov.is_empty() {
                out.push(format!("Control covariates: {} ({}; {})", o.ctrlcov.join(" "), if o.strata { "strata" } else { "linear model" }, if o.pvc_normal { "normal" } else { "empirical" }));
            }
            out.push(String::new());
            if o.boot {
                let b = bootstrap(st, &obs, &o, &est, level)?;
                reps = b.reps;
                out.push(format!("{:>13}|{:>11} {:>10} {:>10}   {:>22}", "", "Observed", "", "Bootstrap", ""));
                out.push(format!("{:>13}|{:>11} {:>10} {:>10}   {:>22}", class, "coef.", "Bias", "std. err.", format!("[{}% conf. interval]", fmt_g(level, 4))));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
                for (j, nm) in names.iter().enumerate() {
                    let lab = match o.stats.get(j) {
                        Some(Stat::Roc(t)) if !o.probit => format!("{nm} ({})", fmt_g(*t, 4)),
                        Some(Stat::InvRoc(t)) if !o.probit => format!("{nm} ({})", fmt_g(*t, 4)),
                        Some(Stat::Pauc(t)) if !o.probit => format!("{nm} ({})", fmt_g(*t, 4)),
                        _ => nm.clone(),
                    };
                    out.push(format!("{:>13}|{:>11} {:>10} {:>10}   {:>10} {:>10}  (N)", lab, fmt_g(est[j], 7), fmt_g(b.bias[j], 7), fmt_g(b.se[j], 7), fmt_g(b.normal[j].0, 7), fmt_g(b.normal[j].1, 7)));
                    out.push(format!("{:>13}|{:>11} {:>10} {:>10}   {:>10} {:>10}  (P)", "", "", "", "", fmt_g(b.pct[j].0, 7), fmt_g(b.pct[j].1, 7)));
                    out.push(format!("{:>13}|{:>11} {:>10} {:>10}   {:>10} {:>10} (BC)", "", "", "", "", fmt_g(b.bc[j].0, 7), fmt_g(b.bc[j].1, 7)));
                }
                out.push("-".repeat(78));
                ci_n.extend(b.normal.iter().copied());
                ci_p.extend(b.pct.iter().copied());
                ci_bc.extend(b.bc.iter().copied());
                biases.extend(b.bias.iter().copied());
                blocks.push(b.v);
            } else {
                out.push(format!("{:>13}|{:>11}", class, "Estimate"));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(20)));
                for (j, nm) in names.iter().enumerate() {
                    out.push(format!("{:>13}|{:>11}", nm, fmt_g(est[j], 7)));
                }
                out.push("-".repeat(34));
                blocks.push(Matrix::zeros(est.len(), est.len()));
            }
            for nm in &names {
                all_names.push(format!("{class}:{nm}"));
            }
            all_b.extend(est);
        }
        // header
        let mut head = vec![String::new()];
        head.push(format!("{:<50}Number of obs     = {:>9}", if o.boot && !o.ml { "Bootstrap results" } else { "" }, n_obs));
        if o.boot && !o.ml {
            head.push(format!("{:<50}Replications      = {:>9}", "", reps));
        }
        head.push(String::new());
        head.push(title.to_string());
        if !o.ml {
            head.push(format!("Control standardization: {}", if o.pvc_normal { "normal" } else { "empirical" }));
        }
        head.push(format!("ROC method             : {}", if o.probit { "parametric" } else { "empirical" }));
        head.extend(out);
        // e() results
        let k = all_b.len();
        let mut v = Matrix::zeros(k, k);
        let mut off = 0;
        for b in &blocks {
            for a in 0..b.rows {
                for c in 0..b.cols {
                    v[(off + a, off + c)] = b[(a, c)];
                }
            }
            off += b.rows;
        }
        let e = st.begin_e();
        e.set_str("cmd", "rocreg");
        e.set_str("refvar", refvar.clone());
        e.set_str("classvars", classes.join(" "));
        e.set_str("ctrlcov", o.ctrlcov.join(" "));
        e.set_str("ctrlmodel", if o.ctrlcov.is_empty() { "" } else if o.strata { "strata" } else { "linear" });
        e.set_str("pvc", if o.pvc_normal { "normal" } else { "empirical" });
        e.set_str("roccov", o.roccov.join(" "));
        e.set_str("probspec", if o.probit { "parametric" } else { "nonparametric" });
        e.set_str("roc_method", if o.ml { "ml" } else if o.probit { "glm" } else { "empirical" });
        e.set_str("tiecorrected", if o.tiecorr { "tiecorrected" } else { "" });
        e.set_str("title", title);
        e.set_str("properties", "b V");
        let pts: Vec<String> = o.stats.iter().map(|s| match s {
            Stat::Auc => "auc".to_string(),
            Stat::Roc(t) => format!("roc {}", fmt_g(*t, 8)),
            Stat::InvRoc(t) => format!("invroc {}", fmt_g(*t, 8)),
            Stat::Pauc(t) => format!("pauc {}", fmt_g(*t, 8)),
        }).collect();
        e.set_str("stats", pts.join(";"));
        e.set("N", n_obs as f64);
        e.set("N_cases", n_case as f64);
        e.set("N_controls", n_ctrl as f64);
        e.set("level", level);
        if o.probit && !o.ml {
            e.set("fprpts", o.fprpts as f64);
        }
        if o.ml {
            e.set("ll", ll_total);
        }
        if o.boot && !o.ml {
            e.set("N_reps", reps as f64);
            e.set_str("vce", "bootstrap");
            e.set_str("vcetype", "Bootstrap");
        } else if o.ml {
            e.set_str("vce", "oim");
        }
        let cn = all_names.clone();
        e.set_mat("b", Matrix::row_vector(&all_b).with_names(vec!["y1".into()], cn.clone()));
        e.set_mat("V", v.with_names(cn.clone(), cn.clone()));
        let ci = |c: &[(f64, f64)]| Matrix::from_rows(&[c.iter().map(|x| x.0).collect(), c.iter().map(|x| x.1).collect()]).with_names(vec!["ll".into(), "ul".into()], cn.clone());
        if !ci_n.is_empty() {
            e.set_mat("ci_normal", ci(&ci_n));
        }
        if !ci_p.is_empty() {
            e.set_mat("ci_percentile", ci(&ci_p));
            e.set_mat("ci_bc", ci(&ci_bc));
            e.set_mat("bias", Matrix::row_vector(&biases).with_names(vec!["y1".into()], cn.clone()));
        }
        e.sample = Some(sample_all);
        Ok(head.join("\n"))
    }
}

/// `rocregplot [, classvars(varlist) at(numlist) norefline nopoints]`:
/// after nonparametric `rocreg`, the empirical ROC curve of each classifier
/// (from its placement values); after `probit`, the fitted binormal curve
/// (ROC covariates at `at()`, default zero) with the empirical points.
pub struct Rocregplot;

impl CommandHandler for Rocregplot {
    fn name(&self) -> &'static str {
        "rocregplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if st.e.macros.get("cmd").map(String::as_str) != Some("rocreg") {
            return Err(rc_err(301, "last estimates not found (rocregplot requires rocreg)"));
        }
        let e = st.e.clone();
        let mac = |k: &str| e.macros.get(k).cloned().unwrap_or_default();
        let split = |s: String| -> Vec<String> { s.split_whitespace().map(String::from).collect() };
        let o = Opts {
            ctrlcov: split(mac("ctrlcov")),
            strata: mac("ctrlmodel") == "strata",
            pvc_normal: mac("pvc") == "normal",
            tiecorr: !mac("tiecorrected").is_empty(),
            probit: mac("probspec") == "parametric",
            ml: mac("roc_method") == "ml",
            roccov: split(mac("roccov")),
            fprpts: e.get("fprpts").unwrap_or(10.0) as usize,
            stats: vec![Stat::Auc],
            breps: 0,
            boot: false,
            cluster: None,
        };
        let refvar = mac("refvar");
        let mut classes = split(mac("classvars"));
        if let Some(c) = cmd.opt("classvars", 5) {
            let want = varlist::expand(ds, c)?;
            classes.retain(|x| want.contains(x));
        }
        let at: Vec<f64> = match cmd.opt("at", 2) {
            Some(a) => crate::margins::numlist(a)?,
            None => vec![0.0; o.roccov.len()],
        };
        if at.len() != o.roccov.len() {
            return Err(rc_err(198, "at() requires one value per ROC covariate"));
        }
        let mask: Vec<bool> = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
        let b = e.matrices.get("b").cloned();
        let mut layers = Vec::new();
        for class in &classes {
            let (obs, _) = sample(ds, &mask, &refvar, class, &o)?;
            let mut u = if o.ml { Vec::new() } else { placements(&obs, &(0..obs.len()).collect::<Vec<_>>(), &o)? };
            u.sort_by(f64::total_cmp);
            let n = u.len() as f64;
            if !u.is_empty() && !cmd.has_opt("nopoints", 5) {
                let mut pts = vec![(0.0, 0.0)];
                for (i, &x) in u.iter().enumerate() {
                    pts.push((x, (i + 1) as f64 / n));
                }
                pts.push((1.0, 1.0));
                let kind = if o.probit { LayerKind::Scatter } else { LayerKind::Line };
                layers.push(Layer { kind, points: pts, label: format!("{class} (empirical)"), ..Default::default() });
            }
            if o.probit {
                let b = b.as_ref().ok_or_else(|| rc_err(301, "e(b) not found"))?;
                let get = |n: &str| b.col_index(&format!("{class}:{n}")).map(|j| b.get(0, j)).unwrap_or(0.0);
                let a = get("i_cons") + o.roccov.iter().zip(&at).map(|(v, x)| get(v) * x).sum::<f64>();
                let s = get("s_cons");
                let mut pts = vec![(0.0, 0.0)];
                for k in 1..200 {
                    let t = k as f64 / 200.0;
                    pts.push((t, dist::normal(a + s * dist::invnormal(t))));
                }
                pts.push((1.0, 1.0));
                layers.push(Layer { kind: LayerKind::Line, points: pts, label: format!("{class} (fitted)"), ..Default::default() });
            }
        }
        if !cmd.has_opt("norefline", 6) {
            layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (1.0, 1.0)], label: "Reference".into(), ..Default::default() });
        }
        let mut g = Graph { layers, xtitle: "False-positive rate".into(), ytitle: "True-positive rate".into(), legend: true, ..Default::default() };
        let _ = SYSMISS;
        graph::finish(&mut g, cmd, st)
    }
}
