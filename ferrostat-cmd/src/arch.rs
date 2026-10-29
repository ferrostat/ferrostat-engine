//! `arch`: autoregressive conditional heteroskedasticity models (ARCH,
//! GARCH, threshold, simple asymmetric, exponential and asymmetric power
//! GARCH, ARCH-in-mean, ARMA mean disturbances) estimated by conditional
//! maximum likelihood with Gaussian, Student t or generalized error
//! innovations; `predict` after `arch` (xb, y, variance, residuals).
//!
//! Mean:      y_t = x_t b + sum_l psi_l sigma2_{t-l} + mu_t,
//!            mu_t = sum rho_l mu_{t-l} + sum theta_l e_{t-l} + e_t
//! Variance:  sigma2_t = g0 + sum a_l e2_{t-l} + sum ta_l e2_{t-l} 1(e_{t-l}>0)
//!                      + sum sa_l e_{t-l} + sum g_l sigma2_{t-l}
//! EGARCH:    ln sigma2_t = g0 + sum ea_l z_{t-l} + sum eaa_l (|z_{t-l}| - sqrt(2/pi)) + sum eg_l ln sigma2_{t-l}
//! APARCH:    sigma_t^p = g0 + sum ap_l (|e| + ae_l e)_{t-l}^p + sum pg_l sigma_{t-l}^p
//! Presample values of e2 and sigma2 are the mean squared residual of the
//! mean equation at the current parameters (`arch0(xb)`); presample
//! disturbances are zero.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::ts_ml::{self, parse_lags, Post, LN2PI};
use crate::tsops;
use crate::tsreg::{f2, f4, hline};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dist {
    Normal,
    T,
    Ged,
}

/// The model terms (lags of each component).
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub archm: Vec<usize>, // 0 = contemporaneous
    pub ar: Vec<usize>,
    pub ma: Vec<usize>,
    pub arch: Vec<usize>,
    pub tarch: Vec<usize>,
    pub saarch: Vec<usize>,
    pub earch: Vec<usize>,
    pub egarch: Vec<usize>,
    pub aparch: Vec<usize>,
    pub pgarch: Vec<usize>,
    pub garch: Vec<usize>,
}

impl Spec {
    fn n_var(&self) -> usize {
        self.arch.len() + self.tarch.len() + self.saarch.len() + 2 * self.earch.len() + self.egarch.len() + 2 * self.aparch.len() + self.pgarch.len() + self.garch.len()
    }
    fn is_egarch(&self) -> bool {
        !self.earch.is_empty() || !self.egarch.is_empty()
    }
    fn is_power(&self) -> bool {
        !self.aparch.is_empty() || !self.pgarch.is_empty()
    }
    fn max_lag(&self) -> usize {
        [&self.archm, &self.ar, &self.ma, &self.arch, &self.tarch, &self.saarch, &self.earch, &self.egarch, &self.aparch, &self.pgarch, &self.garch].iter().flat_map(|v| v.iter()).copied().max().unwrap_or(0)
    }
    fn to_text(&self) -> String {
        let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            j(&self.archm), j(&self.ar), j(&self.ma), j(&self.arch), j(&self.tarch), j(&self.saarch), j(&self.earch), j(&self.egarch), j(&self.aparch), j(&self.pgarch), j(&self.garch)
        )
    }
    fn from_text(s: &str) -> Spec {
        let p: Vec<Vec<usize>> = s.split('|').map(|x| x.split(',').filter_map(|v| v.parse().ok()).collect()).collect();
        let g = |i: usize| p.get(i).cloned().unwrap_or_default();
        Spec { archm: g(0), ar: g(1), ma: g(2), arch: g(3), tarch: g(4), saarch: g(5), earch: g(6), egarch: g(7), aparch: g(8), pgarch: g(9), garch: g(10) }
    }
}

/// Layout of the parameter vector.
#[derive(Clone, Debug)]
pub struct Layout {
    pub kx: usize,
    pub spec: Spec,
    pub dist: Dist,
}

impl Layout {
    fn i_archm(&self) -> usize {
        self.kx
    }
    fn i_ar(&self) -> usize {
        self.i_archm() + self.spec.archm.len()
    }
    fn i_ma(&self) -> usize {
        self.i_ar() + self.spec.ar.len()
    }
    fn i_var(&self) -> usize {
        self.i_ma() + self.spec.ma.len()
    }
    fn i_g0(&self) -> usize {
        self.i_var() + self.spec.n_var()
    }
    fn i_power(&self) -> usize {
        self.i_g0() + 1
    }
    fn i_dist(&self) -> usize {
        self.i_power() + usize::from(self.spec.is_power())
    }
    pub fn npar(&self) -> usize {
        self.i_dist() + usize::from(self.dist != Dist::Normal)
    }

    /// Coefficient names in e(b) order.
    fn names(&self, eq: &str, xnames: &[String]) -> Vec<String> {
        let lag = |l: usize, w: &str| if l == 1 { format!("L.{w}") } else { format!("L{l}.{w}") };
        let mut out: Vec<String> = xnames.iter().map(|n| format!("{eq}:{n}")).collect();
        let s = &self.spec;
        out.extend(s.archm.iter().map(|&l| if l == 0 { "ARCHM:sigma2".to_string() } else { format!("ARCHM:{}", lag(l, "sigma2")) }));
        out.extend(s.ar.iter().map(|&l| format!("ARMA:{}", lag(l, "ar"))));
        out.extend(s.ma.iter().map(|&l| format!("ARMA:{}", lag(l, "ma"))));
        out.extend(s.arch.iter().map(|&l| format!("ARCH:{}", lag(l, "arch"))));
        out.extend(s.tarch.iter().map(|&l| format!("ARCH:{}", lag(l, "tarch"))));
        out.extend(s.saarch.iter().map(|&l| format!("ARCH:{}", lag(l, "saarch"))));
        out.extend(s.earch.iter().map(|&l| format!("ARCH:{}", lag(l, "earch"))));
        out.extend(s.earch.iter().map(|&l| format!("ARCH:{}", lag(l, "earch_a"))));
        out.extend(s.egarch.iter().map(|&l| format!("ARCH:{}", lag(l, "egarch"))));
        out.extend(s.aparch.iter().map(|&l| format!("ARCH:{}", lag(l, "aparch"))));
        out.extend(s.aparch.iter().map(|&l| format!("ARCH:{}", lag(l, "aparch_e"))));
        out.extend(s.pgarch.iter().map(|&l| format!("ARCH:{}", lag(l, "pgarch"))));
        out.extend(s.garch.iter().map(|&l| format!("ARCH:{}", lag(l, "garch"))));
        out.push("ARCH:_cons".into());
        if s.is_power() {
            out.push("POWER:power".into());
        }
        match self.dist {
            Dist::T => out.push("/lndfm2".into()),
            Dist::Ged => out.push("/lnshape".into()),
            Dist::Normal => {}
        }
        out
    }
}

/// Filtered series at parameters `th`: (e_t, sigma2_t, mean prediction xb_t + archm + ARMA).
pub struct Filtered {
    pub e: Vec<f64>,
    pub s2: Vec<f64>,
    pub pred: Vec<f64>,
}

/// Run the recursions over `n_fit` estimation observations followed by
/// any extra periods where y is unknown (forecast with e = 0).
pub fn filter(lay: &Layout, th: &[f64], y: &[Option<f64>], x: &[Vec<f64>]) -> Option<Filtered> {
    let s = &lay.spec;
    let n = y.len();
    let b = &th[..lay.kx];
    let xb: Vec<f64> = x.iter().map(|r| ts_ml::dot(r, b)).collect();
    // priming value: mean squared residual of the mean equation (ARMA included, ARCH-M excluded)
    let arma = |u: &[f64], e: &[f64], t: usize| -> f64 {
        let mut v = 0.0;
        for (j, &l) in s.ar.iter().enumerate() {
            if t >= l {
                v += th[lay.i_ar() + j] * u[t - l];
            }
        }
        for (j, &l) in s.ma.iter().enumerate() {
            if t >= l {
                v += th[lay.i_ma() + j] * e[t - l];
            }
        }
        v
    };
    let mut s0 = 0.0;
    let mut cnt = 0.0;
    {
        let mut u = vec![0.0; n];
        let mut e = vec![0.0; n];
        for t in 0..n {
            if let Some(yv) = y[t] {
                u[t] = yv - xb[t];
                e[t] = u[t] - arma(&u, &e, t);
                s0 += e[t] * e[t];
                cnt += 1.0;
            }
        }
    }
    if cnt == 0.0 {
        return None;
    }
    s0 /= cnt;
    if !(s0 > 0.0) {
        s0 = 1e-12;
    }
    let p = if s.is_power() { th[lay.i_power()] } else { 2.0 };
    if s.is_power() && !(p > 0.0) {
        return None;
    }
    let mut e = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut s2 = vec![0.0; n];
    let mut pred = vec![0.0; n];
    let ez = (2.0 / std::f64::consts::PI).sqrt();
    for t in 0..n {
        let get_e = |l: usize| -> Option<f64> { if t >= l { Some(e[t - l]) } else { None } };
        let get_s2 = |l: usize| -> f64 { if t >= l { s2[t - l] } else { s0 } };
        let mut k = lay.i_var();
        let v;
        if s.is_egarch() {
            let mut lv = th[lay.i_g0()];
            for &l in &s.earch {
                let z = get_e(l).map_or(0.0, |ev| ev / get_s2(l).sqrt());
                lv += th[k] * z;
                k += 1;
            }
            for &l in &s.earch {
                let z = get_e(l).map_or(ez, |ev| (ev / get_s2(l).sqrt()).abs());
                lv += th[k] * (z - ez);
                k += 1;
            }
            for &l in &s.egarch {
                lv += th[k] * get_s2(l).ln();
                k += 1;
            }
            // additive terms are also allowed alongside egarch
            v = lv.exp();
        } else {
            let mut sv = th[lay.i_g0()];
            for &l in &s.arch {
                sv += th[k] * get_e(l).map_or(s0, |ev| ev * ev);
                k += 1;
            }
            for &l in &s.tarch {
                sv += th[k] * get_e(l).map_or(0.5 * s0, |ev| if ev > 0.0 { ev * ev } else { 0.0 });
                k += 1;
            }
            for &l in &s.saarch {
                sv += th[k] * get_e(l).unwrap_or(0.0);
                k += 1;
            }
            if s.is_power() {
                let mut pv = sv;
                let ap0 = k;
                let na = s.aparch.len();
                for (j, &l) in s.aparch.iter().enumerate() {
                    let gam = th[ap0 + na + j];
                    let term = match get_e(l) {
                        Some(ev) => (ev.abs() + gam * ev).abs().powf(p),
                        None => s0.powf(p / 2.0),
                    };
                    pv += th[ap0 + j] * term;
                }
                k += 2 * na;
                for &l in &s.pgarch {
                    pv += th[k] * get_s2(l).powf(p / 2.0);
                    k += 1;
                }
                for &l in &s.garch {
                    pv += th[k] * get_s2(l);
                    k += 1;
                }
                if !(pv > 0.0) {
                    return None;
                }
                v = pv.powf(2.0 / p);
            } else {
                for &l in &s.garch {
                    sv += th[k] * get_s2(l);
                    k += 1;
                }
                v = sv;
            }
        }
        if !(v > 0.0) || !v.is_finite() {
            return None;
        }
        let mut m = xb[t];
        for (j, &l) in s.archm.iter().enumerate() {
            m += th[lay.i_archm() + j] * if l == 0 { v } else { get_s2(l) };
        }
        let am = arma(&u, &e, t);
        s2[t] = v;
        pred[t] = m + am;
        match y[t] {
            Some(yv) => {
                u[t] = yv - m;
                e[t] = u[t] - am;
            }
            None => {
                u[t] = am;
                e[t] = 0.0;
            }
        }
    }
    Some(Filtered { e, s2, pred })
}

/// Log density of e with conditional variance s2.
fn lden(dist: Dist, e: f64, s2: f64, par: f64) -> f64 {
    match dist {
        Dist::Normal => -0.5 * (LN2PI + s2.ln() + e * e / s2),
        Dist::T => {
            let nu = par.exp() + 2.0;
            dist::lngamma((nu + 1.0) / 2.0) - dist::lngamma(nu / 2.0) - 0.5 * (std::f64::consts::PI * (nu - 2.0) * s2).ln() - (nu + 1.0) / 2.0 * (1.0 + e * e / ((nu - 2.0) * s2)).ln()
        }
        Dist::Ged => {
            let sh = par.exp();
            let lam = (2f64.powf(-2.0 / sh) * dist::lngamma(1.0 / sh).exp() / dist::lngamma(3.0 / sh).exp()).sqrt();
            let sd = s2.sqrt();
            sh.ln() - (1.0 + 1.0 / sh) * 2f64.ln() - dist::lngamma(1.0 / sh) - (lam * sd).ln() - 0.5 * (e.abs() / (lam * sd)).powf(sh)
        }
    }
}

fn parse_dist(cmd: &Command) -> Result<(Dist, Option<f64>), String> {
    let Some(d) = cmd.opt("distribution", 4) else { return Ok((Dist::Normal, None)) };
    let mut w = d.split_whitespace();
    let kind = w.next().unwrap_or("").to_lowercase();
    let fixed = w.next().and_then(|v| v.parse::<f64>().ok());
    Ok(match kind.as_str() {
        "gaussian" | "normal" => (Dist::Normal, None),
        "t" => (Dist::T, fixed),
        "ged" => (Dist::Ged, fixed),
        o => return Err(rc_err(198, format!("distribution({o}) not allowed"))),
    })
}

fn parse_spec(cmd: &Command) -> Result<Spec, String> {
    let l = |name: &str, min: usize| -> Result<Vec<usize>, String> { cmd.opt(name, min).map_or(Ok(vec![]), parse_lags) };
    let mut s = Spec { ar: l("ar", 2)?, ma: l("ma", 2)?, arch: l("arch", 4)?, tarch: l("tarch", 5)?, saarch: l("saarch", 6)?, earch: l("earch", 5)?, egarch: l("egarch", 6)?, aparch: l("aparch", 6)?, pgarch: l("pgarch", 6)?, garch: l("garch", 5)?, archm: vec![] };
    if let Some(a) = cmd.opt("arima", 5) {
        let v: Vec<usize> = a.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 3 {
            s.ar = (1..=v[0]).collect();
            s.ma = (1..=v[2]).collect();
        }
    }
    if cmd.has_opt("archm", 5) || cmd.opt("archmlags", 6).is_some() {
        s.archm = vec![0];
        if let Some(a) = cmd.opt("archmlags", 6) {
            s.archm.extend(parse_lags(a)?);
            if !cmd.has_opt("archm", 5) {
                s.archm.remove(0);
            }
        }
    }
    // egarch() alone implies earch with the same lags in the reference
    if !s.egarch.is_empty() && s.earch.is_empty() {
        s.earch = s.egarch.clone();
    }
    if s.is_egarch() && (!s.arch.is_empty() || !s.garch.is_empty() || !s.tarch.is_empty() || !s.saarch.is_empty() || s.is_power()) {
        return Err(rc_err(198, "egarch terms cannot be combined with other ARCH terms"));
    }
    Ok(s)
}

pub struct Arch;

impl CommandHandler for Arch {
    fn name(&self) -> &'static str {
        "arch"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "arch");
        }
        let spec = parse_spec(cmd)?;
        let (dist, fixed) = parse_dist(cmd)?;
        let samp = ts_ml::ts_sample(ds, st, cmd, &cmd.main, cmd.has_opt("noconstant", 4), vec![])?;
        let kx = samp.kx();
        let lay = Layout { kx, spec: spec.clone(), dist };
        let n = samp.n();
        if n <= lay.npar() + 1 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let y: Vec<Option<f64>> = samp.y.iter().map(|v| Some(*v)).collect();
        let fixed_par = fixed.map(|v| match dist {
            Dist::T => (v - 2.0).max(1e-8).ln(),
            _ => v.max(1e-8).ln(),
        });
        let llf = |th: &[f64]| -> Option<Vec<f64>> {
            let mut full;
            let th = if let Some(fp) = fixed_par {
                full = th.to_vec();
                full.push(fp);
                &full[..]
            } else {
                th
            };
            let f = filter(&lay, th, &y, &samp.x)?;
            let par = if dist == Dist::Normal { 0.0 } else { th[lay.i_dist()] };
            let v: Vec<f64> = (0..n).map(|t| lden(dist, f.e[t], f.s2[t], par)).collect();
            if v.iter().all(|x| x.is_finite()) { Some(v) } else { None }
        };
        let k_est = lay.npar() - usize::from(fixed_par.is_some());
        // starting values
        let beta = ts_ml::lsq(&samp.x, &samp.y);
        let resid: Vec<f64> = (0..n).map(|t| samp.y[t] - samp.xb(t, &beta)).collect();
        let s2 = resid.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let mk = |a0: f64, g0: f64| -> Vec<f64> {
            let mut th = beta.clone();
            th.extend(spec.archm.iter().map(|_| 0.0));
            th.extend(spec.ar.iter().map(|_| 0.1));
            th.extend(spec.ma.iter().map(|_| 0.0));
            let na = spec.arch.len() + spec.tarch.len() + spec.aparch.len();
            let ng = spec.garch.len() + spec.pgarch.len();
            let mut sum = 0.0;
            for _ in &spec.arch {
                th.push(a0 / na.max(1) as f64);
                sum += a0 / na.max(1) as f64;
            }
            for _ in &spec.tarch {
                th.push(a0 / na.max(1) as f64);
                sum += 0.5 * a0 / na.max(1) as f64;
            }
            th.extend(spec.saarch.iter().map(|_| 0.0));
            th.extend(spec.earch.iter().map(|_| 0.0));
            th.extend(spec.earch.iter().map(|_| a0));
            th.extend(spec.egarch.iter().map(|_| g0 / spec.egarch.len() as f64));
            for _ in &spec.aparch {
                th.push(a0 / na.max(1) as f64);
                sum += a0 / na.max(1) as f64;
            }
            th.extend(spec.aparch.iter().map(|_| 0.0));
            for _ in &spec.pgarch {
                th.push(g0 / ng.max(1) as f64);
                sum += g0 / ng.max(1) as f64;
            }
            for _ in &spec.garch {
                th.push(g0 / ng.max(1) as f64);
                sum += g0 / ng.max(1) as f64;
            }
            if spec.is_egarch() {
                th.push(s2.ln() * (1.0 - g0));
            } else if spec.is_power() {
                th.push(s2.sqrt().powf(1.5) * (1.0 - sum).max(0.05));
                th.push(1.5);
            } else {
                th.push(s2 * (1.0 - sum).max(0.05));
            }
            if dist != Dist::Normal && fixed_par.is_none() {
                th.push(if dist == Dist::T { 8f64.ln() } else { 0.5f64.ln() + 1.0 });
            }
            th
        };
        let mut th0 = mk(0.1, if spec.is_egarch() { 0.5 } else { 0.7 });
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (a, g) in [(0.05, 0.9), (0.2, 0.5), (0.3, 0.3), (0.1, 0.0)] {
            let c = mk(a, g);
            let l = ts_ml::ll_sum(&llf, &c);
            if l > best {
                best = l;
                th0 = c;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, k_est)? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(100.0) as usize;
        let tech = ts_ml::tech_from(cmd, ts_ml::Tech::BhhhBfgs);
        let opt = ts_ml::optimize(&llf, th0, maxiter, tech, "log likelihood", true)?;
        let th = opt.th.clone();
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "opg")?;
        let eq = util::base_name(&samp.depvar).to_string();
        let eq = eq.rsplit('.').next().unwrap_or(&eq).to_string();
        let mut names = lay.names(&eq, &samp.xnames);
        if fixed_par.is_some() {
            names.pop();
        }
        // model test: regressors other than _cons, ARCH-M and ARMA terms
        let test_idx: Vec<usize> = (0..lay.i_var()).filter(|&j| !(j < kx && samp.xnames[j] == "_cons")).collect();
        let (chi2, df_m) = ts_ml::wald(&th, &v, &test_idx);
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let dname = match dist {
            Dist::Normal => "Gaussian".to_string(),
            Dist::T => "t".to_string(),
            Dist::Ged => "GED".to_string(),
        };
        let pchi = if df_m > 0.0 { dist::chi2tail(df_m, chi2) } else { SYSMISS };
        let header = vec![
            hline(&sline, "Number of obs", &fmt_g(n as f64, 10)),
            hline(&format!("Distribution: {dname}"), &format!("Wald chi2({})", if df_m > 0.0 { df_m.to_string() } else { ".".into() }), &f2(chi2)),
            hline(&format!("Log likelihood = {}", fmt_g(opt.ll, 8)), "Prob > chi2", &f4(pchi)),
        ];
        let title = "ARCH family regression";
        let full_title = if !spec.ar.is_empty() || !spec.ma.is_empty() { "ARCH family regression -- ARMA disturbances" } else { title };
        let mut notes = Vec::new();
        if let (Dist::T, None) = (dist, fixed_par) {
            let lt = th[lay.i_dist()];
            notes.push(format!("{:>12}df = {}", "", fmt_g(lt.exp() + 2.0, 9)));
        }
        let text = ts_ml::post(
            st,
            cmd,
            Post { title: full_title, cmd: "arch", depvar: &samp.depvar, names, b: th.clone(), v, ll: opt.ll, n, header, vcetype, vce: vce_name, log: opt.log, notes, converged: opt.converged, iters: opt.iters, df: None },
            Some(&samp.mask),
        )?;
        let e = &mut st.e;
        e.set("N_gaps", samp.n_gaps as f64);
        e.set("k_eq_model", 1.0);
        e.set("k_dv", 1.0);
        e.set("k1", kx as f64);
        e.set("df_m", df_m);
        e.set("chi2", chi2);
        e.set("p", pchi);
        e.set("tmin", samp.tmin());
        e.set("tmax", samp.tmax());
        e.set("archmax", spec.max_lag() as f64);
        e.set("power", if spec.is_power() { th[lay.i_power()] } else { 2.0 });
        if dist == Dist::T {
            e.set("tdf", fixed.unwrap_or_else(|| th[lay.i_dist()].exp() + 2.0));
        }
        if dist == Dist::Ged {
            e.set("shape", fixed.unwrap_or_else(|| th[lay.i_dist()].exp()));
        }
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("chi2type", "Wald");
        e.set_str("dist", match dist { Dist::Normal => "gaussian", Dist::T => "t", Dist::Ged => "ged" });
        e.set_str("covariates", samp.xnames.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
        let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
        e.set_str("arch", j(&spec.arch));
        e.set_str("garch", j(&spec.garch));
        e.set_str("tarch", j(&spec.tarch));
        e.set_str("saarch", j(&spec.saarch));
        e.set_str("egarch", j(&spec.egarch));
        e.set_str("aparch", j(&spec.aparch));
        e.set_str("ar", j(&spec.ar));
        e.set_str("ma", j(&spec.ma));
        e.set_str("archm", if spec.archm.is_empty() { "" } else { "archm" });
        e.set_str("tech", "bhhh");
        e.set_str("predict", "arch_p");
        e.set_str("_spec", spec.to_text());
        e.set_str("_title", title);
        e.set_str("_fixdist", fixed_par.map(|v| v.to_string()).unwrap_or_default());
        Ok(text)
    }
}

/// `predict` after `arch`: xb (default), y, variance (het), residuals, yresiduals; dynamic().
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let spec = Spec::from_text(e.macros.get("_spec").map(String::as_str).unwrap_or(""));
    let dist = match e.macros.get("dist").map(String::as_str) {
        Some("t") => Dist::T,
        Some("ged") => Dist::Ged,
        _ => Dist::Normal,
    };
    let kx = e.get("k1").unwrap_or(0.0) as usize;
    let mut th: Vec<f64> = b.data[..b.cols].to_vec();
    if let Some(v) = e.macros.get("_fixdist").and_then(|s| s.parse::<f64>().ok()) {
        th.push(v);
    }
    let lay = Layout { kx, spec, dist };
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let xnames: Vec<String> = b.colnames[..kx].iter().map(|c| c.split_once(':').map_or(c.clone(), |x| x.1.to_string())).collect();
    let newvar = ts_ml::predict_name(cmd)?;
    let mut stat = "xb";
    let mut explicit = false;
    for (n, _) in &cmd.opts {
        let n = n.to_lowercase();
        let s = match n.as_str() {
            "xb" => "xb",
            "y" => "y",
            "variance" | "var" | "het" => "variance",
            _ if !n.is_empty() && "residuals".starts_with(n.as_str()) => "residuals",
            _ if n.len() >= 2 && "yresiduals".starts_with(n.as_str()) => "yresiduals",
            _ => continue,
        };
        stat = s;
        explicit = true;
    }
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let mask = util::touse(ds, st, cmd)?;
    let wv = tsops::values(ds, st, &dep)?;
    let xv: Vec<Vec<f64>> = xnames.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?;
    let tmin = e.get("tmin").unwrap_or(0.0);
    let dyn_t = match cmd.opt("dynamic", 3) {
        Some(t) => {
            let ex = ferrostat_parser::parser::parse_expression(t).map_err(|e| rc_err(198, e.message))?;
            Some(crate::eval::Env::with_store(ds, st).eval(&ex, 0)?.num()?)
        }
        None => None,
    };
    // time-ordered rows from tmin on with all covariates present
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(tc[i]) && tc[i] >= tmin && xv.iter().all(|c| !is_missing(c[i]))).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    let y: Vec<Option<f64>> = rows.iter().map(|&i| if is_missing(wv[i]) || dyn_t.is_some_and(|d| tc[i] >= d) { None } else { Some(wv[i]) }).collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|&i| xv.iter().map(|c| c[i]).collect()).collect();
    let f = filter(&lay, &th, &y, &x).ok_or_else(|| rc_err(498, "variance could not be computed"))?;
    let mut out = vec![SYSMISS; ds.n_obs];
    for (p, &i) in rows.iter().enumerate() {
        out[i] = match stat {
            "variance" => f.s2[p],
            "residuals" | "yresiduals" => {
                if is_missing(wv[i]) {
                    SYSMISS
                } else {
                    wv[i] - f.pred[p]
                }
            }
            _ => f.pred[p],
        };
    }
    if stat == "y" || stat == "yresiduals" {
        // undo the difference operator of the dependent variable
        if let Some((op, var)) = dep.rsplit_once('.') {
            let poly = tsops::lag_poly(op).unwrap_or(vec![(0, 1.0)]);
            let yv = util::col(ds, var)?;
            let pos_of: std::collections::HashMap<i64, usize> = (0..ds.n_obs).filter(|&i| !is_missing(tc[i])).map(|i| (tc[i].round() as i64, i)).collect();
            let mut lev = vec![SYSMISS; ds.n_obs];
            for &i in &rows {
                let mut v = if stat == "y" { out[i] } else { f.pred[rows.iter().position(|&r| r == i).unwrap()] };
                let mut ok = !is_missing(v);
                for &(l, c) in &poly {
                    if l == 0 {
                        continue;
                    }
                    let tl = tc[i].round() as i64 - l;
                    match pos_of.get(&tl).map(|&r| yv[r]).filter(|v| !is_missing(*v)) {
                        Some(yy) => v -= c * yy,
                        None => ok = false,
                    }
                }
                if ok {
                    lev[i] = v;
                }
            }
            for &i in &rows {
                out[i] = if stat == "y" { lev[i] } else if is_missing(yv[i]) || is_missing(lev[i]) { SYSMISS } else { yv[i] - lev[i] };
            }
        }
    }
    let note = if explicit { "" } else { "(option xb assumed; predicted values)" };
    ts_ml::add_prediction(ds, &newvar, out, &mask, note)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garch_filter_recursion() {
        let lay = Layout { kx: 1, spec: Spec { arch: vec![1], garch: vec![1], ..Default::default() }, dist: Dist::Normal };
        let y = vec![Some(1.0), Some(-1.0), Some(2.0)];
        let x = vec![vec![1.0]; 3];
        let th = [0.0, 0.2, 0.5, 0.1];
        let f = filter(&lay, &th, &y, &x).unwrap();
        let s0 = 2.0;
        assert!((f.s2[0] - (0.1 + 0.2 * s0 + 0.5 * s0)).abs() < 1e-12);
        assert!((f.s2[1] - (0.1 + 0.2 * 1.0 + 0.5 * f.s2[0])).abs() < 1e-12);
    }
}
