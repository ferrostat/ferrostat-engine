//! `xtgee`: generalized estimating equations (population-averaged panel
//! models) with Gaussian, binomial, Poisson, negative binomial and gamma
//! families, the usual links and independent, exchangeable, AR(k),
//! stationary(k) or unstructured working correlation (Liang and Zeger,
//! moment estimators of the correlation divided by N as by default).

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Family {
    Gaussian,
    Binomial,
    Poisson,
    Nbinomial(f64),
    Gamma,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Link {
    Identity,
    Log,
    Logit,
    Probit,
    Cloglog,
    Reciprocal,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Corr {
    Independent,
    Exchangeable,
    Ar(usize),
    Stationary(usize),
    Unstructured,
}

impl Family {
    fn var(self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Nbinomial(k) => mu + k * mu * mu,
            Family::Gamma => mu * mu,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Family::Gaussian => "Gaussian",
            Family::Binomial => "Binomial",
            Family::Poisson => "Poisson",
            Family::Nbinomial(_) => "Negative binomial",
            Family::Gamma => "Gamma",
        }
    }
    fn fixed_scale(self) -> bool {
        matches!(self, Family::Binomial | Family::Poisson | Family::Nbinomial(_))
    }
}

impl Link {
    /// (mu, dmu/deta)
    fn inv(self, eta: f64) -> (f64, f64) {
        match self {
            Link::Identity => (eta, 1.0),
            Link::Log => (eta.exp(), eta.exp()),
            Link::Logit => {
                let p = dist::invlogit(eta);
                (p, p * (1.0 - p))
            }
            Link::Probit => (dist::normal(eta), dist::normalden(eta)),
            Link::Cloglog => {
                let e = eta.exp();
                (1.0 - (-e).exp(), e * (-e).exp())
            }
            Link::Reciprocal => (1.0 / eta, -1.0 / (eta * eta)),
        }
    }
    fn of(self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Log => mu.ln(),
            Link::Logit => dist::logit(mu),
            Link::Probit => dist::invnormal(mu),
            Link::Cloglog => (-(1.0 - mu).ln()).ln(),
            Link::Reciprocal => 1.0 / mu,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Link::Identity => "Identity",
            Link::Log => "Log",
            Link::Logit => "Logit",
            Link::Probit => "Probit",
            Link::Cloglog => "Cloglog",
            Link::Reciprocal => "Reciprocal",
        }
    }
}

pub struct GeeOut {
    pub b: Vec<f64>,
    pub v: Matrix,
    pub v_model: Matrix,
    pub phi: f64,
    pub r: Matrix,
    pub iters: usize,
    pub n_clust: usize,
    pub dev: f64,
    pub chi2_dev: f64,
}

/// GEE fit. `pos[t]` is the time position (0-based) of observation t.
#[allow(clippy::too_many_arguments)]
pub fn gee_fit(x: &[&[f64]], y: &[f64], off: &[f64], groups: &[Vec<usize>], pos: &[usize], tmax: usize, fam: Family, link: Link, corr: Corr, nmp: bool, robust: Option<&[f64]>, fixed_r: Option<&Matrix>) -> Result<GeeOut, String> {
    let k = x.len();
    let n: usize = groups.iter().map(|g| g.len()).sum();
    let ybar = y.iter().sum::<f64>() / n as f64;
    let eta_of = |b: &[f64], t: usize| off[t] + x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>();
    // starting values: IRLS GLM fit
    let mut b = vec![0.0; k];
    let mut eta0: Vec<f64> = (0..y.len()).map(|t| {
        let m = match fam {
            Family::Binomial => (y[t] + 0.5) / 2.0,
            Family::Gaussian => y[t],
            _ => (y[t] + ybar) / 2.0,
        };
        let m = if m <= 0.0 && link != Link::Identity { ybar.max(0.1) } else { m };
        link.of(m) - off[t]
    })
    .collect();
    let idx: Vec<usize> = groups.iter().flatten().cloned().collect();
    for _ in 0..50 {
        let mut a = Matrix::zeros(k, k);
        let mut u = vec![0.0; k];
        for &t in &idx {
            let e = eta0[t] + off[t];
            let (mu, dm) = link.inv(e);
            let w = dm * dm / fam.var(mu).max(1e-300);
            let z = eta0[t] + (y[t] - mu) / dm;
            for i in 0..k {
                u[i] += w * x[i][t] * z;
                for j in 0..k {
                    a[(i, j)] += w * x[i][t] * x[j][t];
                }
            }
        }
        let (ai, _) = a.sym_inverse(1e-14);
        let nb: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * u[j]).sum()).collect();
        let d = nb.iter().zip(&b).map(|(p, q)| (p - q).abs() / (q.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        for &t in &idx {
            eta0[t] = eta_of(&b, t) - off[t];
        }
        if d < 1e-10 || !b.iter().all(|v| v.is_finite()) {
            break;
        }
    }
    if !b.iter().all(|v| v.is_finite()) {
        b = vec![0.0; k];
    }
    let p_adj = if nmp { k as f64 } else { 0.0 };
    let mut iters = 0;
    // pieces for a panel given R
    let panel = |b: &[f64], r: &Matrix, g: &[usize]| -> (Matrix, Vec<f64>, Vec<f64>) {
        let m = g.len();
        let mut rr = Matrix::zeros(m, m);
        for (a, &s) in g.iter().enumerate() {
            for (c, &t) in g.iter().enumerate() {
                rr[(a, c)] = r[(pos[s], pos[t])];
            }
        }
        let ri = rr.inverse().unwrap_or_else(|_| rr.sym_inverse(1e-14).0);
        let mut d = vec![vec![0.0; k]; m];
        let mut e = vec![0.0; m];
        for (a, &t) in g.iter().enumerate() {
            let (mu, dm) = link.inv(eta_of(b, t));
            let sv = fam.var(mu).max(1e-300).sqrt();
            e[a] = (y[t] - mu) / sv;
            for j in 0..k {
                d[a][j] = dm / sv * x[j][t];
            }
        }
        let mut am = Matrix::zeros(k, k);
        let mut u = vec![0.0; k];
        for a in 0..m {
            for c in 0..m {
                let w = ri[(a, c)];
                if w == 0.0 {
                    continue;
                }
                for i in 0..k {
                    u[i] += d[a][i] * w * e[c];
                    for j in 0..k {
                        am[(i, j)] += d[a][i] * w * d[c][j];
                    }
                }
            }
        }
        (am, u, e)
    };
    // working correlation from Pearson residuals
    let estimate_r = |b: &[f64]| -> (Matrix, f64) {
        let mut res: Vec<Vec<(usize, f64)>> = Vec::with_capacity(groups.len());
        let mut ss = 0.0;
        for g in groups {
            let v: Vec<(usize, f64)> = g.iter().map(|&t| {
                let (mu, _) = link.inv(eta_of(b, t));
                (pos[t], (y[t] - mu) / fam.var(mu).max(1e-300).sqrt())
            })
            .collect();
            ss += v.iter().map(|(_, e)| e * e).sum::<f64>();
            res.push(v);
        }
        let phi = ss / (n as f64 - p_adj);
        let mut r = Matrix::identity(tmax);
        match corr {
            Corr::Independent => {}
            Corr::Exchangeable => {
                let (mut sx, mut np) = (0.0, 0.0);
                for v in &res {
                    let s: f64 = v.iter().map(|x| x.1).sum();
                    let q: f64 = v.iter().map(|x| x.1 * x.1).sum();
                    sx += s * s - q;
                    np += (v.len() * (v.len() - 1)) as f64;
                }
                let a = if np > 0.0 { sx / (np - p_adj) / phi } else { 0.0 };
                for i in 0..tmax {
                    for j in 0..tmax {
                        if i != j {
                            r[(i, j)] = a;
                        }
                    }
                }
            }
            Corr::Ar(kk) | Corr::Stationary(kk) => {
                let mut rho = vec![1.0; kk + 1];
                for (lag, rl) in rho.iter_mut().enumerate().skip(1) {
                    let (mut s, mut c) = (0.0, 0.0);
                    for v in &res {
                        for a in 0..v.len() {
                            for bb in 0..v.len() {
                                if v[bb].0 == v[a].0 + lag {
                                    s += v[a].1 * v[bb].1;
                                    c += 1.0;
                                }
                            }
                        }
                    }
                    *rl = if c > 0.0 { s / (c - p_adj).max(1.0) / phi } else { 0.0 };
                }
                let full: Vec<f64> = if let Corr::Ar(_) = corr {
                    // Yule-Walker: AR coefficients, then implied autocorrelations
                    let mut tm = Matrix::zeros(kk, kk);
                    for i in 0..kk {
                        for j in 0..kk {
                            tm[(i, j)] = rho[(i as i64 - j as i64).unsigned_abs() as usize];
                        }
                    }
                    let ti = tm.inverse().unwrap_or_else(|_| Matrix::identity(kk));
                    let phis: Vec<f64> = (0..kk).map(|i| (0..kk).map(|j| ti[(i, j)] * rho[j + 1]).sum()).collect();
                    let mut ac = rho.clone();
                    for l in kk + 1..tmax.max(kk + 1) {
                        let v: f64 = (0..kk).map(|j| phis[j] * ac[l - 1 - j]).sum();
                        ac.push(v);
                    }
                    ac
                } else {
                    let mut ac = rho.clone();
                    ac.resize(tmax.max(kk + 1), 0.0);
                    ac
                };
                for i in 0..tmax {
                    for j in 0..tmax {
                        r[(i, j)] = full[(i as i64 - j as i64).unsigned_abs() as usize];
                    }
                }
            }
            Corr::Unstructured => {
                let mut s = Matrix::zeros(tmax, tmax);
                let mut c = Matrix::zeros(tmax, tmax);
                for v in &res {
                    for a in v {
                        for bb in v {
                            s[(a.0, bb.0)] += a.1 * bb.1;
                            c[(a.0, bb.0)] += 1.0;
                        }
                    }
                }
                for i in 0..tmax {
                    for j in 0..tmax {
                        if i != j && c[(i, j)] > 0.0 {
                            r[(i, j)] = s[(i, j)] / (c[(i, j)] - p_adj).max(1.0) / phi;
                        }
                    }
                }
            }
        }
        (r, phi)
    };
    for it in 0..200 {
        iters = it + 1;
        let (r, _) = match fixed_r {
            Some(m) => (m.clone(), 1.0),
            None => estimate_r(&b),
        };
        let mut a = Matrix::zeros(k, k);
        let mut u = vec![0.0; k];
        for g in groups {
            let (ag, ug, _) = panel(&b, &r, g);
            for i in 0..k {
                u[i] += ug[i];
                for j in 0..k {
                    a[(i, j)] += ag[(i, j)];
                }
            }
        }
        let (ai, _) = a.sym_inverse(1e-14);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * u[j]).sum()).collect();
        let dif = step.iter().zip(&b).map(|(s, bb)| s.abs() / (bb.abs() + 1.0)).fold(0.0, f64::max);
        for i in 0..k {
            b[i] += step[i];
        }
        if !b.iter().all(|v| v.is_finite()) {
            return Err(rc_err(430, "convergence not achieved"));
        }
        if dif < 1e-10 {
            break;
        }
    }
    let (nr, nphi) = match fixed_r {
        Some(m) => (m.clone(), estimate_r(&b).1),
        None => estimate_r(&b),
    };
    let r = nr;
    let phi = if fam.fixed_scale() { 1.0 } else { nphi };
    let mut a = Matrix::zeros(k, k);
    let mut us = Vec::new();
    let (mut dev, mut chi2_dev) = (0.0, 0.0);
    for g in groups {
        let (ag, ug, e) = panel(&b, &r, g);
        chi2_dev += e.iter().map(|v| v * v).sum::<f64>();
        for i in 0..k {
            for j in 0..k {
                a[(i, j)] += ag[(i, j)];
            }
        }
        us.push(ug);
        for &t in g {
            let (mu, _) = link.inv(eta_of(&b, t));
            dev += match fam {
                Family::Gaussian => (y[t] - mu).powi(2),
                Family::Poisson => 2.0 * (if y[t] > 0.0 { y[t] * (y[t] / mu).ln() } else { 0.0 } - (y[t] - mu)),
                Family::Binomial => -2.0 * (if y[t] > 0.0 { mu.ln() } else { (1.0 - mu).ln() }),
                _ => 0.0,
            };
        }
    }
    let (ai, _) = a.sym_inverse(1e-14);
    let v_model = ai.scale(phi);
    let (v, ncl) = match robust {
        Some(cl) => est::sandwich(&ai, &us, Some(cl), 1.0),
        None => (v_model.clone(), groups.len()),
    };
    Ok(GeeOut { b, v, v_model, phi, r, iters, n_clust: ncl, dev, chi2_dev })
}

fn parse_family(cmd: &Command) -> Result<Family, String> {
    let f = cmd.opt("family", 1).unwrap_or("gaussian").trim().to_lowercase();
    let mut w = f.split_whitespace();
    let name = w.next().unwrap_or("gaussian");
    let arg = w.next().and_then(|s| s.parse::<f64>().ok());
    let is = |full: &str, min: usize| name.len() >= min && full.starts_with(name);
    Ok(if is("gaussian", 3) || is("normal", 3) {
        Family::Gaussian
    } else if is("binomial", 1) {
        if arg.is_some_and(|a| a != 1.0) {
            return Err(rc_err(198, "family(binomial #) with # > 1 not supported"));
        }
        Family::Binomial
    } else if is("poisson", 1) {
        Family::Poisson
    } else if is("nbinomial", 2) {
        Family::Nbinomial(arg.unwrap_or(1.0))
    } else if is("gamma", 3) {
        Family::Gamma
    } else {
        return Err(rc_err(198, format!("unknown family() {name}")));
    })
}

fn parse_link(cmd: &Command, fam: Family) -> Result<Link, String> {
    let l = match cmd.opt("link", 1) {
        Some(s) => s.trim().to_lowercase(),
        None => {
            return Ok(match fam {
                Family::Gaussian => Link::Identity,
                Family::Binomial => Link::Logit,
                Family::Gamma => Link::Reciprocal,
                _ => Link::Log,
            })
        }
    };
    let is = |full: &str, min: usize| l.len() >= min && full.starts_with(l.as_str());
    Ok(if is("identity", 1) {
        Link::Identity
    } else if is("logit", 4) {
        Link::Logit
    } else if is("log", 3) {
        Link::Log
    } else if is("probit", 1) {
        Link::Probit
    } else if is("cloglog", 1) {
        Link::Cloglog
    } else if is("reciprocal", 3) || l == "power -1" {
        Link::Reciprocal
    } else {
        return Err(rc_err(198, format!("unknown link() {l}")));
    })
}

pub fn parse_corr(cmd: &Command) -> Result<Corr, String> {
    let c = cmd.opt("corr", 1).or(cmd.opt("correlation", 4)).unwrap_or("exchangeable").trim().to_lowercase();
    let mut w = c.split_whitespace();
    let name = w.next().unwrap_or("exchangeable");
    let arg = w.next().and_then(|s| s.parse::<usize>().ok());
    let name = name.split('(').next().unwrap_or(name);
    let paren = c.split_once('(').and_then(|(_, r)| r.trim_end_matches(')').trim().parse::<usize>().ok());
    let arg = arg.or(paren);
    let is = |full: &str, min: usize| name.len() >= min && full.starts_with(name);
    Ok(if is("independent", 3) {
        Corr::Independent
    } else if is("exchangeable", 3) {
        Corr::Exchangeable
    } else if is("ar", 2) {
        Corr::Ar(arg.unwrap_or(1).max(1))
    } else if is("stationary", 3) {
        Corr::Stationary(arg.unwrap_or(1).max(1))
    } else if is("unstructured", 3) {
        Corr::Unstructured
    } else if is("fixed", 3) {
        return Err(rc_err(198, "corr(fixed) not supported"));
    } else {
        return Err(rc_err(198, format!("unknown corr() {name}")));
    })
}

pub struct XtGee;

impl CommandHandler for XtGee {
    fn name(&self) -> &'static str {
        "xtgee"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("xtgee") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(render(&st.e, cmd.has_opt("eform", 2), util::level(cmd, st)?));
        }
        let fam = parse_family(cmd)?;
        let link = parse_link(cmd, fam)?;
        let corr = parse_corr(cmd)?;
        let nmp = cmd.has_opt("nmp", 3);
        let p = crate::xtnl::prepare(ds, st, cmd, cmd.has_opt("noconstant", 3))?;
        // time positions
        let needs_t = matches!(corr, Corr::Ar(_) | Corr::Stationary(_) | Corr::Unstructured);
        let tvar = match cmd.opt("t", 1) {
            Some(t) => Some(crate::varlist::resolve_name(ds, t.trim())?),
            None => st.tsset.as_ref().map(|(_, t)| t.clone()).filter(|t| !t.is_empty()),
        };
        let n = p.d.n();
        let (pos, tmax) = match (&tvar, needs_t) {
            (Some(tv), _) => {
                let c = util::col(ds, tv)?;
                let tv: Vec<f64> = p.d.rows.iter().map(|&i| c[i]).collect();
                let mut times = tv.clone();
                times.sort_by(f64::total_cmp);
                times.dedup();
                let pos: Vec<usize> = tv.iter().map(|t| times.iter().position(|x| x == t).unwrap()).collect();
                // equally spaced positions when time is integer-valued
                let t0 = times[0];
                let int = times.iter().all(|t| (t - t0).fract() == 0.0) && times.len() > 1;
                if int {
                    let mut gcd = 0i64;
                    for t in &times {
                        let mut a = (t - t0) as i64;
                        let mut b = gcd;
                        while b != 0 {
                            let r = a % b;
                            a = b;
                            b = r;
                        }
                        gcd = a.abs();
                    }
                    let gcd = gcd.max(1);
                    let pos: Vec<usize> = tv.iter().map(|t| ((t - t0) as i64 / gcd) as usize).collect();
                    let tm = pos.iter().max().unwrap() + 1;
                    (pos, tm)
                } else {
                    (pos, times.len())
                }
            }
            (None, true) => return Err(rc_err(459, "time variable not set; use xtset panelvar timevar")),
            (None, false) => {
                let mut pos = vec![0usize; n];
                for g in &p.groups {
                    for (a, &t) in g.iter().enumerate() {
                        pos[t] = a;
                    }
                }
                let tm = p.groups.iter().map(|g| g.len()).max().unwrap_or(1);
                (pos, tm)
            }
        };
        // duplicate time within panel is an error when time matters
        if needs_t {
            for g in &p.groups {
                let mut s: Vec<usize> = g.iter().map(|&t| pos[t]).collect();
                s.sort();
                let l = s.len();
                s.dedup();
                if s.len() != l {
                    return Err(rc_err(451, "repeated time values within panel"));
                }
            }
        }
        if fam == Family::Binomial && p.d.y.iter().any(|&y| !(0.0..=1.0).contains(&y)) {
            return Err(rc_err(2000, format!("{} must be 0/1 for family(binomial)", p.dep)));
        }
        let mut y = p.d.y.clone();
        if fam == Family::Binomial {
            for v in y.iter_mut() {
                *v = if *v != 0.0 { 1.0 } else { 0.0 };
            }
        }
        let act = p.d.active();
        let cols: Vec<&[f64]> = act.iter().map(|&j| p.d.cols[j].as_slice()).collect();
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let cl: Option<Vec<f64>> = match &vce {
            Vce::Robust => Some((0..p.groups.len()).map(|g| g as f64).collect()),
            Vce::Cluster(c) => {
                let col = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
                Some(p.groups.iter().map(|g| col[p.d.rows[g[0]]]).collect())
            }
            _ => None,
        };
        let fit = gee_fit(&cols, &y, &p.off, &p.groups, &pos, tmax, fam, link, corr, nmp, cl.as_deref(), None)?;
        let k = p.d.k();
        let mut bfull = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            bfull[j] = fit.b[a];
        }
        let v = est::expand_v(&fit.v, &act, k);
        let (chi2, df_m) = est::wald_all(&p.d, &bfull, &v);
        let sizes: Vec<usize> = p.groups.iter().map(|g| g.len()).collect();
        let mn = *sizes.iter().min().unwrap_or(&0) as f64;
        let mx = *sizes.iter().max().unwrap_or(&0) as f64;
        let av = n as f64 / sizes.len().max(1) as f64;
        let mut d = p.d.clone();
        d.names = d.names.iter().map(|nm| format!("{}:{nm}", p.dep)).collect();
        let corr_name = match corr {
            Corr::Independent => "independent".to_string(),
            Corr::Exchangeable => "exchangeable".to_string(),
            Corr::Ar(k) => format!("AR({k})"),
            Corr::Stationary(k) => format!("stationary({k})"),
            Corr::Unstructured => "unstructured".to_string(),
        };
        let tn: Vec<String> = (1..=tmax).map(|i| format!("c{i}")).collect();
        let rn: Vec<String> = (1..=tmax).map(|i| format!("r{i}")).collect();
        let robust = cl.is_some();
        let clustvar = match &vce {
            Vce::Cluster(c) => c.clone(),
            _ => p.ivar.clone(),
        };
        let mut iter_log = vec![String::new()];
        for i in 1..=fit.iters {
            iter_log.push(format!("Iteration {i}: Tolerance = {}", if i == fit.iters { "converged".into() } else { fmt_g(10f64.powi(-(i as i32)), 9) }));
        }
        let head = iter_log.join("\n");
        let ng = p.groups.len();
        let level = p.level;
        let ivar = p.ivar.clone();
        let tvn = tvar.clone().unwrap_or_default();
        let offset = p.offset.clone();
        let vm = est::expand_v(&fit.v_model, &act, k);
        est::post(st, "xtgee", &d, &bfull, &v, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("phi", fit.phi);
            e.set("deviance", fit.dev);
            e.set("chi2_dev", fit.chi2_dev);
            e.set("dispers", fit.chi2_dev / n as f64);
            e.set("tol", 1e-6);
            e.set("rank", act.len() as f64);
            if robust {
                e.set("N_clust", fit.n_clust as f64);
            }
            e.set_mat("R", fit.r.clone().with_names(rn.clone(), tn.clone()));
            e.set_mat("V_modelbased", vm.clone());
            e.set_str("model", "pa");
            e.set_str("family", fam.name());
            e.set_str("link", link.name());
            e.set_str("corr", corr_name.clone());
            e.set_str("scale", if fam.fixed_scale() { "1" } else { "phi" });
            e.set_str("ivar", ivar.clone());
            if !tvn.is_empty() {
                e.set_str("tvar", tvn.clone());
            }
            e.set_str("title", "GEE population-averaged model");
            e.set_str("chi2type", "Wald");
            e.set_str("vce", if robust { "robust" } else { "conventional" });
            if robust {
                e.set_str("vcetype", "Robust");
                e.set_str("clustvar", clustvar.clone());
            }
            if let Some(o) = &offset {
                e.set_str("offset", o.clone());
            }
            e.set_str("predict", "xtgee_p");
            e.set_str("_head", head.clone());
        });
        let eform = cmd.has_opt("eform", 2) || cmd.has_opt("or", 2) || cmd.has_opt("irr", 3);
        Ok(format!("{head}\n{}", render(&st.e, eform, level)))
    }
}

fn render(e: &Results, eform: bool, level: f64) -> String {
    use crate::xtnl::{header_block, hl, hv};
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let mac = |s: &str| e.macros.get(s).cloned().unwrap_or_default();
    let mut h = vec![hl(""), hv("GEE population-averaged model", "Number of obs", util::commas(get("N") as usize))];
    let gv = if mac("tvar").is_empty() { format!("Group variable: {}", mac("ivar")) } else { format!("Group and time vars: {} {}", mac("ivar"), mac("tvar")) };
    h.push(hv(&gv, "Number of groups", util::commas(get("N_g") as usize)));
    h.push(hv(&format!("Family: {}", mac("family")), "Obs per group:", String::new()));
    h.push(hv(&format!("Link:   {}", mac("link")), &format!("{:>16}", "min"), util::commas(get("g_min") as usize)));
    h.push(hv(&format!("Correlation: {}", mac("corr")), &format!("{:>16}", "avg"), format!("{:.1}", get("g_avg"))));
    h.push(hv("", &format!("{:>16}", "max"), util::commas(get("g_max") as usize)));
    h.push(hv("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
    h.push(hv(&format!("Scale parameter = {}", fmt_g(get("phi"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    let mut out = header_block(&h);
    if !mac("vcetype").is_empty() {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for clustering on {})", mac("clustvar"))));
    }
    out.push(String::new());
    let b = e.matrices.get("b").cloned().unwrap();
    let v = e.matrices.get("V").cloned().unwrap();
    let names: Vec<String> = b.colnames.iter().map(|n| n.split_once(':').map(|x| x.1.to_string()).unwrap_or_else(|| n.clone())).collect();
    let is_base: Vec<bool> = names.iter().map(|c| c.split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = names.iter().zip(&is_base).map(|(c, ib)| *ib || c.starts_with("o.")).collect();
    let ef = if eform {
        Some(match mac("link").as_str() {
            "Logit" => "Odds ratio",
            "Log" => "IRR",
            _ => "exp(b)",
        })
    } else {
        None
    };
    let mut tbl = est::coef_table(&est::TableSpec { depvar: &mac("depvar"), names: &names, b: &b.data, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: ef, coef_title: "Coefficient" });
    if !mac("vcetype").is_empty() {
        let lw = tbl.last().map(|b| b.len()).unwrap_or(78).saturating_sub(66);
        tbl.insert(1, format!("{:>lw$} |{:>23}", "", "Robust"));
    }
    out.extend(tbl);
    out.join("\n")
}

/// `estat wcorrelation` after xtgee: the estimated working correlation.
pub fn wcorrelation(st: &Store) -> Result<String, String> {
    let r = st.e.matrices.get("R").ok_or_else(|| rc_err(321, "requested action not valid after most recent estimation command"))?;
    let mut out = vec![String::new(), "Estimated within-panel correlation matrix R:".to_string(), String::new()];
    let t = r.rows;
    out.push(format!("{:>6} {}", "", (1..=t).map(|j| format!("{:>9}", format!("c{j}"))).collect::<Vec<_>>().join("")));
    for i in 0..t {
        out.push(format!("{:>6} {}", format!("r{}", i + 1), (0..=i).map(|j| format!("{:>9.4}", r[(i, j)])).collect::<Vec<_>>().join("")));
    }
    Ok(out.join("\n"))
}
