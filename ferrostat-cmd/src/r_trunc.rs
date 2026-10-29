//! Truncated-sample regression models: `truncreg` (normal), `tpoisson`
//! (Poisson) and `tnbreg` (negative binomial, mean or constant dispersion).
//!
//! Observations outside the truncation range are dropped; the likelihood of
//! each remaining observation is its density divided by the probability of
//! the observable range.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::{gamma_lr, ln_gamma};

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels::{self as xm, Row};

pub fn handlers() -> xm::Handlers {
    vec![("truncreg", Box::new(Truncreg)), ("tpoisson", Box::new(Tpoisson)), ("tnbreg", Box::new(Tnbreg))]
}

/// Header in the current layout: left texts at column 0, right-hand
/// `key = value` pairs starting at column 56 with aligned keys and values.
pub fn header_new(left: &[String], right: &[(String, String)]) -> Vec<String> {
    let kw = right.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    let vw = right.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    let n = left.len().max(right.len());
    let mut out = Vec::new();
    for i in 0..n {
        let l = left.get(i).cloned().unwrap_or_default();
        match right.get(i) {
            Some((k, v)) => out.push(format!("{:<56}{:<kw$} = {:>vw$}", l, k, v)),
            None => out.push(l),
        }
    }
    out
}

/// A truncation limit: a number or a variable.
#[derive(Clone)]
pub enum Lim {
    Num(f64),
    Var(String),
}

impl Lim {
    fn text(&self) -> String {
        match self {
            Lim::Num(v) => fmt_g(*v, 8),
            Lim::Var(s) => s.clone(),
        }
    }
}

fn parse_lim(ds: &DataSet, cmd: &Command, name: &str) -> Result<Option<Lim>, String> {
    match cmd.opt(name, 2) {
        None => Ok(None),
        Some(s) => {
            let s = s.trim();
            match s.parse::<f64>() {
                Ok(v) => Ok(Some(Lim::Num(v))),
                Err(_) => Ok(Some(Lim::Var(xm::var_name(ds, s)?))),
            }
        }
    }
}

fn lim_values(ds: &DataSet, d: &Design, l: &Option<Lim>, def: f64) -> Result<Vec<f64>, String> {
    Ok(match l {
        None => vec![def; d.n()],
        Some(Lim::Num(v)) => vec![*v; d.n()],
        Some(Lim::Var(v)) => {
            let c = util::col(ds, v)?;
            d.rows.iter().map(|&i| if is_missing(c[i]) { def } else { c[i] }).collect()
        }
    })
}

fn lim_vars(l: &[&Option<Lim>]) -> Vec<String> {
    l.iter().filter_map(|x| if let Some(Lim::Var(v)) = x { Some(v.clone()) } else { None }).collect()
}

/// Offset over the sample rows from `offset()` / `exposure()`.
pub fn offset_of(ds: &DataSet, d: &Design, cmd: &Command) -> Result<Vec<f64>, String> {
    let mut off = vec![0.0; d.n()];
    if let Some(o) = cmd.opt("offset", 3) {
        let c = util::col(ds, &xm::var_name(ds, o)?)?;
        for (t, &i) in d.rows.iter().enumerate() {
            off[t] += c[i];
        }
    }
    if let Some(o) = cmd.opt("exposure", 1) {
        let c = util::col(ds, &xm::var_name(ds, o)?)?;
        for (t, &i) in d.rows.iter().enumerate() {
            if !(c[i] > 0.0) {
                return Err(rc_err(459, "exposure must be positive"));
            }
            off[t] += c[i].ln();
        }
    }
    Ok(off)
}

fn offset_vars(ds: &DataSet, cmd: &Command) -> Vec<String> {
    let mut v = Vec::new();
    for o in ["offset", "exposure"] {
        if let Some(s) = cmd.opt(o, if o == "offset" { 3 } else { 1 }) {
            if let Ok(n) = xm::var_name(ds, s) {
                v.push(n);
            }
        }
    }
    v
}

/// Replay the stored output of the last estimation of `verb`.
pub fn replay(st: &Store, verb: &str) -> Result<String, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
        return Err(rc_err(301, "last estimates not found"));
    }
    est::stash_table_from_e(&st.e, st.level);
    Ok(st.e.macros.get("_output").cloned().unwrap_or_default())
}

/// Replace auxiliary parameter `j` (estimated on the log scale) by its
/// exponential, with delta-method variances, and rename it.
fn exp_aux(fit: &mut xm::Fit, name: &str, new: &str) {
    let Some(j) = fit.idx(&format!("/:{name}")) else { return };
    let e = fit.b[j].exp();
    fit.b[j] = e;
    let k = fit.b.len();
    for i in 0..k {
        fit.v[(i, j)] *= e;
        fit.v[(j, i)] *= e;
    }
    fit.names[j] = format!("/:{new}");
}

// ---------------------------------------------------------------------------
// truncreg
// ---------------------------------------------------------------------------

pub struct Truncreg;

impl CommandHandler for Truncreg {
    fn name(&self) -> &'static str {
        "truncreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "truncreg");
        }
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let ll = parse_lim(ds, cmd, "ll")?;
        let ul = parse_lim(ds, cmd, "ul")?;
        let mut extra = lim_vars(&[&ll, &ul]);
        extra.extend(offset_vars(ds, cmd));
        let d0 = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fpia", extra })?;
        let lo = lim_values(ds, &d0, &ll, f64::NEG_INFINITY)?;
        let hi = lim_values(ds, &d0, &ul, f64::INFINITY)?;
        let keep: Vec<bool> = (0..d0.n()).map(|t| d0.y[t] > lo[t] && d0.y[t] < hi[t]).collect();
        let ntr = keep.iter().filter(|k| !**k).count();
        let mut d = xm::subset(&d0, &keep);
        est::drop_collinear(&mut d);
        if d.n() == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let lo: Vec<f64> = lo.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect();
        let hi: Vec<f64> = hi.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect();
        let off = offset_of(ds, &d, cmd)?;
        let y = d.y.clone();
        let f = |t: usize, v: &[f64]| -> f64 {
            let s = v[1].exp();
            let xb = v[0] + off[t];
            let z = (y[t] - xb) / s;
            let a = (lo[t] - xb) / s;
            let b = (hi[t] - xb) / s;
            // probability of the observable range, computed in the more
            // accurate tail
            let p = if a > 0.0 { xm::ncdf(-a) - xm::ncdf(-b) } else { xm::ncdf(b) - xm::ncdf(a) };
            -0.5 * z * z - 0.918_938_533_204_672_8 - v[1] - p.max(1e-300).ln()
        };
        let g = xm::numeric(&f);
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let ycent: Vec<f64> = y.iter().zip(&off).map(|(a, o)| a - o).collect();
        let (mut b0, s0) = xm::ols_start(&d, &ycent);
        b0.push(s0.max(1e-8).ln());
        let fw = d.wkind == Some(WeightKind::F);
        let mut fit = xm::ml_fit(&[("", &d)], &["lnsigma"], &g, b0, &d.w, fw, &vce, cluster.as_deref())?;
        exp_aux(&mut fit, "lnsigma", "sigma");
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        let mut out = Vec::new();
        out.push(format!("({} obs truncated)", util::commas(ntr)));
        out.push(String::new());
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
        out.push(String::new());
        let lt = ll.as_ref().map_or("-inf".to_string(), |l| l.text());
        let ut = ul.as_ref().map_or("+inf".to_string(), |l| l.text());
        let w = lt.len().max(ut.len());
        out.push("Truncated regression".into());
        out.push(crate::choice_mixed::hrow(&format!("Limit: Lower = {lt:>w$}"), "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow(&format!("       Upper = {ut:>w$}"), &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)), "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        if let (Some(gc), Vce::Cluster(c)) = (fit.n_clust, &vce) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(gc))));
        }
        out.push(String::new());
        let level = util::level(cmd, st)?;
        out.extend(xm::table(&dep, &fit, level, None, &[]));
        let text = out.join("\n");
        let sigma = fit.coef("/:sigma");
        let llopt = ll.as_ref().map(|l| l.text());
        let ulopt = ul.as_ref().map(|l| l.text());
        xm::post_fit(st, "truncreg", &d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("sigma", sigma);
            e.set("N_bf", ntr as f64);
            if let Some(l) = &llopt {
                e.set_str("llopt", l.clone());
            }
            if let Some(u) = &ulopt {
                e.set_str("ulopt", u.clone());
            }
            e.set_str("title", "Truncated regression");
            e.set_str("predict", "truncreg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// count models
// ---------------------------------------------------------------------------

/// ln P(Y = y) for the negative binomial: mean dispersion (NB2, `la` =
/// ln alpha) or constant dispersion (NB1, `la` = ln delta).
pub fn nb_lnp(constant: bool, y: f64, mu: f64, la: f64) -> f64 {
    let a = la.exp();
    if constant {
        let m = mu / a;
        ln_gamma(y + m) - ln_gamma(m) - ln_gamma(y + 1.0) - (m + y) * a.ln_1p() + y * la
    } else {
        let m = 1.0 / a;
        ln_gamma(y + m) - ln_gamma(m) - ln_gamma(y + 1.0) - (m + y) * (a * mu).ln_1p() + y * (a * mu).ln()
    }
}

/// ln P(Y > a) for a Poisson mean `mu`.
fn pois_ln_gt(a: f64, mu: f64) -> f64 {
    if a < 0.0 {
        return 0.0;
    }
    let k = a.floor();
    if k == 0.0 {
        return if mu < 1e-6 { mu.ln() + (-mu / 2.0).ln_1p() } else { (-(-mu).exp_m1()).ln() };
    }
    gamma_lr(k + 1.0, mu).max(1e-300).ln()
}

/// ln P(lo < Y < hi) for the Poisson.
fn pois_ln_range(lo: f64, hi: f64, mu: f64) -> f64 {
    let a = pois_ln_gt(lo, mu);
    if hi.is_infinite() {
        return a;
    }
    // P(Y >= hi) = P(Y > hi - 1)
    let b = pois_ln_gt(hi.ceil() - 1.0, mu);
    let d = a.exp() - b.exp();
    d.max(1e-300).ln()
}

/// ln P(lo < Y < hi) for the negative binomial.
fn nb_ln_range(constant: bool, lo: f64, hi: f64, mu: f64, la: f64) -> f64 {
    let cdf = |k: f64| -> f64 {
        // P(Y <= k)
        let mut s = 0.0;
        let mut j = 0.0;
        while j <= k {
            s += nb_lnp(constant, j, mu, la).exp();
            j += 1.0;
        }
        s
    };
    let pl = if lo < 0.0 { 0.0 } else { cdf(lo.floor()) };
    let ph = if hi.is_infinite() { 1.0 } else { cdf(hi.ceil() - 1.0) };
    let p = ph - pl;
    if p < 1e-12 && hi.is_infinite() {
        // upper tail by direct summation for accuracy
        let mut s = 0.0;
        let mut j = lo.floor() + 1.0;
        for _ in 0..10000 {
            let t = nb_lnp(constant, j, mu, la).exp();
            s += t;
            if t < 1e-17 * s {
                break;
            }
            j += 1.0;
        }
        return s.max(1e-300).ln();
    }
    p.max(1e-300).ln()
}

struct CountSetup {
    dep: String,
    d: Design,
    lo: Vec<f64>,
    hi: Vec<f64>,
    off: Vec<f64>,
    ll: Option<Lim>,
    ul: Option<Lim>,
    vce: Vce,
    cluster: Option<Vec<f64>>,
}

fn count_setup(ds: &DataSet, st: &Store, cmd: &Command) -> Result<CountSetup, String> {
    let (dep, indep) = xm::first_word(&cmd.main);
    let dep = xm::var_name(ds, &dep)?;
    let ll = parse_lim(ds, cmd, "ll")?.or(Some(Lim::Num(0.0)));
    let ul = parse_lim(ds, cmd, "ul")?;
    let mut extra = lim_vars(&[&ll, &ul]);
    extra.extend(offset_vars(ds, cmd));
    let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fpi", extra })?;
    let lo = lim_values(ds, &d, &ll, -1.0)?;
    let hi = lim_values(ds, &d, &ul, f64::INFINITY)?;
    for t in 0..d.n() {
        if d.y[t] < 0.0 || d.y[t].fract() != 0.0 {
            return Err(rc_err(459, format!("{dep} must be a nonnegative integer")));
        }
        if d.y[t] <= lo[t] || d.y[t] >= hi[t] {
            return Err(rc_err(459, format!("observations with {dep} outside the truncation points not allowed")));
        }
    }
    let off = offset_of(ds, &d, cmd)?;
    let vce = xm::ml_vce_opt(cmd, &d)?;
    let cluster = est::cluster_values(ds, &d, &vce)?;
    Ok(CountSetup { dep, d, lo, hi, off, ll, ul, vce, cluster })
}

fn count_test(fit: &xm::Fit, robust: bool, ll0: f64) -> (&'static str, f64, f64) {
    let (w, df) = fit.wald_eq(0);
    if robust || is_missing(ll0) {
        ("Wald", w, df)
    } else {
        ("LR", 2.0 * (fit.ll() - ll0), df)
    }
}

pub struct Tpoisson;

impl CommandHandler for Tpoisson {
    fn name(&self) -> &'static str {
        "tpoisson"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "tpoisson");
        }
        let s = count_setup(ds, st, cmd)?;
        let d = &s.d;
        let y = d.y.clone();
        let (lo, hi, off) = (&s.lo, &s.hi, &s.off);
        let f = |t: usize, v: &[f64]| -> f64 {
            let e = v[0] + off[t];
            let mu = e.exp();
            y[t] * e - mu - ln_gamma(y[t] + 1.0) - pois_ln_range(lo[t], hi[t], mu)
        };
        let g = xm::numeric(&f);
        let fw = d.wkind == Some(WeightKind::F);
        let ly: Vec<f64> = y.iter().zip(off).map(|(v, o)| v.max(0.5).ln() - o).collect();
        let (b0, _) = xm::ols_start(d, &ly);
        let fit = xm::ml_fit(&[("", d)], &[], &g, b0, &d.w, fw, &s.vce, s.cluster.as_deref())?;
        let mut ll0 = SYSMISS;
        if let Some(d0) = xm::const_only(d) {
            let m = util::mean(&ly);
            if let Ok(f0) = xm::ml_fit(&[("", &d0)], &[], &g, vec![m], &d.w, fw, &Vce::Oim, None) {
                ll0 = f0.ll();
            }
        }
        let robust = s.vce != Vce::Oim;
        let (tn, chi2, df) = count_test(&fit, robust, ll0);
        let r2p = if is_missing(ll0) { SYSMISS } else { 1.0 - fit.ll() / ll0 };
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 5) {
            out.extend(est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
            out.push(String::new());
        }
        let hrow = crate::choice_mixed::hrow;
        let lt = s.ll.as_ref().map_or("0".into(), |l| l.text());
        let ut = s.ul.as_ref().map_or("+inf".into(), |l| l.text());
        out.push(hrow("Truncated Poisson regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(hrow(&format!("Limits:  lower = {lt:>10}"), &format!("{tn} chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(hrow(&format!("         upper = {ut:>10}"), "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)), "Pseudo R2", xm::f4(r2p)));
        if let (Some(gc), Vce::Cluster(c)) = (fit.n_clust, &s.vce) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(gc))));
        }
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let eform = if cmd.has_opt("irr", 3) { Some("IRR") } else { None };
        out.extend(xm::table(&s.dep, &fit, level, eform, &[]));
        let text = out.join("\n");
        let (llopt, ulopt) = (s.ll.as_ref().map(|l| l.text()), s.ul.as_ref().map(|l| l.text()));
        let offset = cmd.opt("offset", 3).map(|o| o.trim().to_string()).or(cmd.opt("exposure", 1).map(|o| format!("ln({})", o.trim())));
        xm::post_fit(st, "tpoisson", d, &fit, &s.vce, &|e: &mut Results| {
            xm::set_test(e, tn, chi2, df);
            e.set("ll_0", ll0);
            e.set("r2_p", r2p);
            e.set_str("llopt", llopt.clone().unwrap_or_else(|| "0".into()));
            if let Some(u) = &ulopt {
                e.set_str("ulopt", u.clone());
            }
            if let Some(o) = &offset {
                e.set_str("offset", o.clone());
            }
            e.set_str("title", "Truncated Poisson regression");
            e.set_str("predict", "tpoisson_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

pub struct Tnbreg;

impl CommandHandler for Tnbreg {
    fn name(&self) -> &'static str {
        "tnbreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "tnbreg");
        }
        let constant = matches!(cmd.opt("dispersion", 1).map(|s| s.trim().to_lowercase()), Some(ref s) if s.starts_with("cons"));
        let s = count_setup(ds, st, cmd)?;
        let d = &s.d;
        let y = d.y.clone();
        let (lo, hi, off) = (&s.lo, &s.hi, &s.off);
        let f = |t: usize, v: &[f64]| -> f64 {
            let mu = (v[0] + off[t]).exp();
            nb_lnp(constant, y[t], mu, v[1]) - nb_ln_range(constant, lo[t], hi[t], mu, v[1])
        };
        let g = xm::numeric(&f);
        let fw = d.wkind == Some(WeightKind::F);
        // starting values from the truncated Poisson
        let fp = |t: usize, v: &[f64]| -> f64 {
            let e = v[0] + off[t];
            let mu = e.exp();
            y[t] * e - mu - ln_gamma(y[t] + 1.0) - pois_ln_range(lo[t], hi[t], mu)
        };
        let gp = xm::numeric(&fp);
        let ly: Vec<f64> = y.iter().zip(off).map(|(v, o)| v.max(0.5).ln() - o).collect();
        let (bp0, _) = xm::ols_start(d, &ly);
        let pfit = xm::ml_fit(&[("", d)], &[], &gp, bp0, &d.w, fw, &Vce::Oim, None).ok();
        let mut b0 = pfit.as_ref().map_or_else(|| xm::ols_start(d, &ly).0, |p| p.bt.clone());
        b0.push(-1.0);
        let aux = if constant { "lndelta" } else { "lnalpha" };
        let mut fit = xm::ml_fit(&[("", d)], &[aux], &g, b0, &d.w, fw, &s.vce, s.cluster.as_deref())?;
        let mut ll0 = SYSMISS;
        if let Some(d0) = xm::const_only(d) {
            let m = util::mean(&ly);
            if let Ok(f0) = xm::ml_fit(&[("", &d0)], &[aux], &g, vec![m, fit.coef(&format!("/:{aux}"))], &d.w, fw, &Vce::Oim, None) {
                ll0 = f0.ll();
            }
        }
        let robust = s.vce != Vce::Oim;
        let (tn, chi2, df) = count_test(&fit, robust, ll0);
        let r2p = if is_missing(ll0) { SYSMISS } else { 1.0 - fit.ll() / ll0 };
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 5) {
            out.extend(est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
            out.push(String::new());
        }
        let trunc = match (&s.ll, &s.ul) {
            (Some(Lim::Num(a)), None) => format!("Truncation point = {}", fmt_g(*a, 8)),
            (Some(l), None) => format!("Truncation points: {}", l.text()),
            (l, Some(u)) => format!("Truncation points: {} {}", l.as_ref().map_or("0".into(), |x| x.text()), u.text()),
            (None, None) => "Truncation point = 0".into(),
        };
        let left = vec![
            if constant { "Truncated negative binomial regression".to_string() } else { "Truncated negative binomial regression".to_string() },
            trunc,
            format!("Dispersion: {}", if constant { "constant" } else { "mean" }),
            format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)),
        ];
        let right = vec![
            ("Number of obs".to_string(), util::commas(d.n_eff() as usize)),
            (format!("{tn} chi2({})", fmt_g(df, 9)), xm::f2(chi2)),
            ("Prob > chi2".to_string(), xm::f4(xm::chi2p(df, chi2))),
            ("Pseudo R2".to_string(), xm::f4(r2p)),
        ];
        out.extend(header_new(&left, &right));
        if let (Some(gc), Vce::Cluster(c)) = (fit.n_clust, &s.vce) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(gc))));
        }
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let eform = if cmd.has_opt("irr", 3) { Some("IRR") } else { None };
        // the dispersion parameter is shown with its interval only
        let j = fit.idx(&format!("/:{aux}")).unwrap();
        let (la, se) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
        let crit = crate::choice_mixed::zcrit(level);
        let alias = if constant { "delta" } else { "alpha" };
        let rows = vec![Row::Ci(format!("/{aux}"), la, se, la - crit * se, la + crit * se), Row::Tr(alias.to_string(), la.exp(), la.exp() * se, la, se, f64::exp)];
        let saved_names = fit.names.clone();
        fit.names.truncate(j);
        let mut tbl = xm::table(&s.dep, &fit, level, eform, &rows);
        fit.names = saved_names;
        // r(table) with all parameters
        let full = est::table_matrix(&est::TableSpec { depvar: &s.dep, names: &fit.names, b: &fit.b, v: &fit.v, omitted: &fit.omitted, is_base: &fit.is_base, df: None, level, eform: None, coef_title: "Coefficient" });
        est::stash_table(full, level);
        out.append(&mut tbl);
        let chibar = pfit.as_ref().map(|p| (2.0 * (fit.ll() - p.ll())).max(0.0));
        if !robust && !constant {
            if let Some(c) = chibar {
                out.push(format!("LR test of alpha=0: chibar2(01) = {:.2}{:>20}Prob >= chibar2 = {:.3}", c, "", 0.5 * dist::chi2tail(1.0, c)));
            }
        }
        let text = out.join("\n");
        let llopt = s.ll.as_ref().map(|l| l.text());
        let ulopt = s.ul.as_ref().map(|l| l.text());
        let disp = la.exp();
        xm::post_fit(st, "tnbreg", d, &fit, &s.vce, &|e: &mut Results| {
            xm::set_test(e, tn, chi2, df);
            e.set("ll_0", ll0);
            e.set("r2_p", r2p);
            e.set(alias, disp);
            if let Some(c) = chibar {
                e.set("chi2_c", c);
            }
            e.set_str("llopt", llopt.clone().unwrap_or_else(|| "0".into()));
            if let Some(u) = &ulopt {
                e.set_str("ulopt", u.clone());
            }
            e.set_str("dispers", if constant { "constant" } else { "mean" });
            e.set_str("title", "Truncated negative binomial regression");
            e.set_str("predict", "tnbreg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// `predict` after the commands of this module.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    let llopt = st.e.macros.get("llopt").cloned().unwrap_or_default();
    let ulopt = st.e.macros.get("ulopt").cloned().unwrap_or_default();
    let lim_at = |ds: &DataSet, s: &str, row: usize, def: f64| -> f64 {
        if s.is_empty() {
            return def;
        }
        match s.parse::<f64>() {
            Ok(v) => v,
            Err(_) => util::col(ds, s).map(|c| c[row]).unwrap_or(def),
        }
    };
    let dsc = ds.clone();
    match ecmd {
        "truncreg" => Some(xm::generic_predict(ds, st, cmd, &["xb", "e", "ystar", "stdp", "stdf"], "(option xb assumed; fitted values)", &|s, c| {
            let sig = (c.aux)("sigma");
            let a = lim_at(&dsc, &llopt, c.row, f64::NEG_INFINITY);
            let b = lim_at(&dsc, &ulopt, c.row, f64::INFINITY);
            let xb = c.eta[0];
            let (za, zb) = ((a - xb) / sig, (b - xb) / sig);
            let pdf = |z: f64| if z.is_infinite() { 0.0 } else { dist::normalden(z) };
            let p = xm::ncdf(zb) - xm::ncdf(za);
            match s {
                "xb" => xb,
                "e" => xb + sig * (pdf(za) - pdf(zb)) / p,
                "ystar" => {
                    // E(y*) with y* = max(a, min(y, b))
                    let fa = if a.is_finite() { a * xm::ncdf(za) } else { 0.0 };
                    let fb = if b.is_finite() { b * (1.0 - xm::ncdf(zb)) } else { 0.0 };
                    fa + fb + xb * p + sig * (pdf(za) - pdf(zb))
                }
                _ => SYSMISS,
            }
        })),
        "tpoisson" | "tnbreg" => {
            let constant = st.e.macros.get("dispers").is_some_and(|s| s == "constant");
            let nb = ecmd == "tnbreg";
            Some(xm::generic_predict(ds, st, cmd, &["n", "ir", "cm", "xb", "stdp"], "(option n assumed; predicted number of events)", &|s, c| {
                let mu = c.eta[0].exp();
                match s {
                    "n" | "ir" => mu,
                    "xb" => c.eta[0],
                    "cm" => {
                        let a = lim_at(&dsc, &llopt, c.row, 0.0);
                        let b = lim_at(&dsc, &ulopt, c.row, f64::INFINITY);
                        if nb {
                            let la = (c.aux)(if constant { "lndelta" } else { "lnalpha" });
                            let lp = nb_ln_range(constant, a, b, mu, la);
                            // E(Y 1{Y <= a}) and above b subtracted from mu
                            let mut low = 0.0;
                            let mut j = 0.0;
                            while j <= a {
                                low += j * nb_lnp(constant, j, mu, la).exp();
                                j += 1.0;
                            }
                            let mut high = 0.0;
                            if b.is_finite() {
                                let mut j = b.ceil();
                                for _ in 0..100000 {
                                    let t = j * nb_lnp(constant, j, mu, la).exp();
                                    high += t;
                                    if t < 1e-15 {
                                        break;
                                    }
                                    j += 1.0;
                                }
                            }
                            (mu - low - high) / lp.exp()
                        } else {
                            // E(Y | a < Y < b) = mu P(a-1 < Y < b-1) / P(a < Y < b)
                            mu * (pois_ln_range(a - 1.0, b - 1.0, mu) - pois_ln_range(a, b, mu)).exp()
                        }
                    }
                    _ => SYSMISS,
                }
            }))
        }
        _ => None,
    }
}
