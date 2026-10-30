//! Bayesian model averaging for linear regression.
//!
//! `bmaregress y x1 ... xk [(z1 z2) ...] [(w1 w2, always)]` averages over
//! the linear models formed by subsets of the candidate predictors (a
//! parenthesized group enters and leaves together; `, always` marks
//! predictors kept in every model). Coefficients get Zellner's g-prior
//! with flat priors on the intercept and the error variance, so that with
//! k predictors and fit R^2 a model's marginal likelihood is
//!
//!   p(y | M, g) ∝ (1 + g)^((n - 1 - k)/2) [1 + g (1 - R^2)]^(-(n - 1)/2)
//!
//! g priors: `gprior(fixed #)`, `uip` (g = n), `ric` (g = K^2), `bric`
//! (g = max(n, K^2), default), `hyper [#]` (hyper-g with parameter a,
//! default 3) and `zs` (Zellner-Siow, g ~ inverse gamma(1/2, n/2)); the
//! last two are integrated numerically. Model priors: `mprior(uniform)`
//! (default), `binomial #`, `betabinomial # #`. With up to 15 candidate
//! terms all models are enumerated; otherwise (or with `sampling`) an
//! MC3 sampler of `mcmcsize()` draws after `burnin()` explores the model
//! space using the session random-number generator (`rseed()`).
//!
//! Postestimation: `bmastats models|msize|pip|jointness|lps`, `bmagraph
//! pmp|msize|varmap|coefdensity`, `bmapredict`, `bmacoefsample`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    vec![
        ("bmaregress", Box::new(Bma { verb: "bmaregress" })),
        ("bmastats", Box::new(Bma { verb: "bmastats" })),
        ("bmagraph", Box::new(Bma { verb: "bmagraph" })),
        ("bmapredict", Box::new(Bma { verb: "bmapredict" })),
        ("bmacoefsample", Box::new(Bma { verb: "bmacoefsample" })),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub enum GPrior {
    Fixed(f64),
    Hyper(f64),
    Zs,
}

#[derive(Clone, Debug)]
pub enum MPrior {
    Uniform,
    Binomial(f64),
    BetaBinomial(f64, f64),
}

/// Data and prior settings of a BMA problem.
pub struct Problem {
    pub n: usize,
    /// centred columns of all predictors (always ones first)
    pub x: Vec<Vec<f64>>,
    pub names: Vec<String>,
    pub y: Vec<f64>,
    pub ybar: f64,
    pub xbar: Vec<f64>,
    pub tss: f64,
    /// candidate terms: column indices of each
    pub terms: Vec<Vec<usize>>,
    pub always: Vec<usize>,
    pub g: GPrior,
    pub mp: MPrior,
}

/// A fitted model.
#[derive(Clone, Debug)]
pub struct Fit {
    pub cols: Vec<usize>,
    pub r2: f64,
    pub bhat: Vec<f64>,
    pub xtxi: Matrix,
    pub logml: f64,
    /// posterior mean of the shrinkage factor g/(1+g)
    pub delta: f64,
}

fn ln_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

impl Problem {
    pub fn kterms(&self) -> usize {
        self.terms.len()
    }

    fn g_fixed(&self) -> Option<f64> {
        match self.g {
            GPrior::Fixed(g) => Some(g),
            _ => None,
        }
    }

    /// log model prior of a model with `k` included terms.
    pub fn log_mprior(&self, k: usize) -> f64 {
        let kk = self.kterms() as f64;
        let k = k as f64;
        match self.mp {
            MPrior::Uniform => -kk * std::f64::consts::LN_2,
            MPrior::Binomial(t) => k * t.ln() + (kk - k) * (1.0 - t).ln(),
            MPrior::BetaBinomial(a, b) => lnbeta(a + k, b + kk - k) - lnbeta(a, b),
        }
    }

    /// Prior probability of model size k.
    pub fn size_prior(&self, k: usize) -> f64 {
        let kk = self.kterms() as f64;
        (dist::lngamma(kk + 1.0) - dist::lngamma(k as f64 + 1.0) - dist::lngamma(kk - k as f64 + 1.0) + self.log_mprior(k)).exp()
    }

    pub fn fit(&self, inc: &[bool]) -> Option<Fit> {
        let mut cols: Vec<usize> = self.always.clone();
        for (t, on) in inc.iter().enumerate() {
            if *on {
                cols.extend(self.terms[t].iter().copied());
            }
        }
        let k = cols.len();
        let n = self.n as f64;
        let (r2, bhat, xtxi) = if k == 0 {
            (0.0, vec![], Matrix::zeros(0, 0))
        } else {
            let mut xtx = Matrix::zeros(k, k);
            let mut xty = vec![0.0; k];
            for (a, &ca) in cols.iter().enumerate() {
                for (b, &cb) in cols.iter().enumerate().take(a + 1) {
                    let v: f64 = self.x[ca].iter().zip(&self.x[cb]).map(|(p, q)| p * q).sum();
                    xtx[(a, b)] = v;
                    xtx[(b, a)] = v;
                }
                xty[a] = self.x[ca].iter().zip(&self.y).map(|(p, q)| p * q).sum();
            }
            let xtxi = xtx.inverse().ok()?;
            let b: Vec<f64> = (0..k).map(|i| (0..k).map(|j| xtxi[(i, j)] * xty[j]).sum()).collect();
            let ess: f64 = b.iter().zip(&xty).map(|(p, q)| p * q).sum();
            ((ess / self.tss).clamp(0.0, 1.0 - 1e-15), b, xtxi)
        };
        if k as f64 >= n - 1.0 {
            return None;
        }
        let kf = k as f64;
        let lml = |g: f64| 0.5 * (n - 1.0 - kf) * g.ln_1p() - 0.5 * (n - 1.0) * (g * (1.0 - r2)).ln_1p();
        let (logml, delta) = match &self.g {
            GPrior::Fixed(g) => (lml(*g), g / (1.0 + g)),
            other => {
                // integrate over s = ln g on a grid (trapezoid in log space)
                let lprior = |g: f64| -> f64 {
                    match other {
                        GPrior::Hyper(a) => ((a - 2.0) / 2.0).ln() - a / 2.0 * g.ln_1p(),
                        _ => {
                            // inverse gamma(1/2, n/2)
                            0.5 * (n / 2.0).ln() - dist::lngamma(0.5) - 1.5 * g.ln() - n / (2.0 * g)
                        }
                    }
                };
                let (lo, hi, m) = (-25.0f64, 40.0f64, 4000);
                let h = (hi - lo) / m as f64;
                let mut acc = f64::NEG_INFINITY;
                let mut accd = f64::NEG_INFINITY;
                for i in 0..=m {
                    let s = lo + i as f64 * h;
                    let g = s.exp();
                    let wt = if i == 0 || i == m { 0.5 } else { 1.0 };
                    let lv = lml(g) + lprior(g) + s + (wt * h).ln();
                    acc = ln_add(acc, lv);
                    accd = ln_add(accd, lv + (g / (1.0 + g)).ln());
                }
                (acc, (accd - acc).exp())
            }
        };
        Some(Fit { cols, r2, bhat, xtxi, logml, delta })
    }

    pub fn log_post(&self, inc: &[bool], f: &Fit) -> f64 {
        f.logml + self.log_mprior(inc.iter().filter(|b| **b).count())
    }

    /// Posterior mean and variance of the coefficients of a model (over all columns).
    pub fn moments(&self, f: &Fit) -> (Vec<f64>, Vec<f64>) {
        let p = self.x.len();
        let mut m = vec![0.0; p];
        let mut v = vec![0.0; p];
        let n = self.n as f64;
        let s = self.tss * (1.0 - f.delta * f.r2);
        for (a, &c) in f.cols.iter().enumerate() {
            m[c] = f.delta * f.bhat[a];
            v[c] = f.delta * s / (n - 3.0) * f.xtxi[(a, a)];
        }
        (m, v)
    }

    /// Posterior predictive scale (residual variance) of a model.
    pub fn sigma2(&self, f: &Fit) -> f64 {
        self.tss * (1.0 - f.delta * f.r2) / (self.n as f64 - 3.0)
    }
}

fn lnbeta(a: f64, b: f64) -> f64 {
    dist::lngamma(a) + dist::lngamma(b) - dist::lngamma(a + b)
}

/// Parse `y x1 x2 (x3 x4) (z, always)`: (depvar, terms, always).
fn parse_spec(ds: &DataSet, main: &str) -> Result<(String, Vec<Vec<String>>, Vec<String>), String> {
    let (dep, rest) = crate::xmodels::first_word(main);
    let dep = crate::xmodels::var_name(ds, &dep)?;
    let mut terms: Vec<Vec<String>> = Vec::new();
    let mut always = Vec::new();
    // walk the text in order: words outside parentheses are single terms
    let mut depth = 0;
    let mut cur = String::new();
    let mut chunks: Vec<(bool, String)> = Vec::new();
    for ch in rest.chars() {
        match ch {
            '(' => {
                if depth == 0 && !cur.trim().is_empty() {
                    chunks.push((false, std::mem::take(&mut cur)));
                }
                if depth == 0 {
                    cur.clear();
                } else {
                    cur.push(ch);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    chunks.push((true, std::mem::take(&mut cur)));
                } else {
                    cur.push(ch);
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        chunks.push((false, cur));
    }
    for (grp, g) in chunks {
        if grp {
            let (vl, o) = g.split_once(',').map_or((g.as_str(), ""), |(a, b)| (a, b));
            let vars = crate::varlist::expand(ds, vl).map_err(|e| rc_err(111, e))?;
            if o.trim().starts_with("always") {
                always.extend(vars);
            } else if !vars.is_empty() {
                terms.push(vars);
            }
        } else {
            for v in crate::varlist::expand(ds, &g).map_err(|e| rc_err(111, e))? {
                terms.push(vec![v]);
            }
        }
    }
    if terms.is_empty() {
        return Err(rc_err(102, "at least one candidate predictor required"));
    }
    Ok((dep, terms, always))
}

fn parse_g(cmd: &Command) -> Result<Option<GPrior>, String> {
    let Some(t) = cmd.opt("gprior", 2) else { return Ok(None) };
    let w: Vec<String> = t.split_whitespace().map(|s| s.to_lowercase()).collect();
    let num = |i: usize| -> Result<Option<f64>, String> { w.get(i).map(|s| s.parse::<f64>().map_err(|_| rc_err(198, "invalid gprior()"))).transpose() };
    Ok(Some(match w.first().map(String::as_str) {
        Some("fixed") => GPrior::Fixed(num(1)?.ok_or_else(|| rc_err(198, "gprior(fixed #) requires a value"))?),
        Some("hyper") => GPrior::Hyper(num(1)?.unwrap_or(3.0)),
        Some("zs") | Some("zellnersiow") => GPrior::Zs,
        Some(s @ ("uip" | "ric" | "bric")) => return Ok(Some(GPrior::Fixed(f64::from(match s { "uip" => -1, "ric" => -2, _ => -3 })))),
        _ => return Err(rc_err(198, "gprior() must be fixed #, uip, ric, bric, hyper [#] or zs")),
    }))
}

/// Build the problem from the data.
fn problem(ds: &DataSet, st: &Store, cmd: &Command, main: &str, ifmask: Option<&[bool]>) -> Result<(Problem, Vec<bool>, Vec<String>), String> {
    let (dep, terms, always) = parse_spec(ds, main)?;
    let mut mask = match ifmask {
        Some(m) => m.to_vec(),
        None => util::touse(ds, st, cmd)?,
    };
    let mut allv: Vec<String> = vec![dep.clone()];
    allv.extend(always.iter().cloned());
    for t in &terms {
        allv.extend(t.iter().cloned());
    }
    util::markout(ds, &mut mask, &allv);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let n = rows.len();
    let ycol = util::col(ds, &dep)?;
    let y0: Vec<f64> = rows.iter().map(|&r| ycol[r]).collect();
    let ybar = util::mean(&y0);
    let y: Vec<f64> = y0.iter().map(|v| v - ybar).collect();
    let tss: f64 = y.iter().map(|v| v * v).sum();
    let mut names = always.clone();
    let mut tcols = Vec::new();
    for t in &terms {
        let mut c = Vec::new();
        for v in t {
            c.push(names.len());
            names.push(v.clone());
        }
        tcols.push(c);
    }
    let mut x = Vec::new();
    let mut xbar = Vec::new();
    for v in &names {
        let c = util::col(ds, v)?;
        let col: Vec<f64> = rows.iter().map(|&r| c[r]).collect();
        let m = util::mean(&col);
        xbar.push(m);
        x.push(col.iter().map(|a| a - m).collect());
    }
    let kk = terms.len() as f64;
    let g = match parse_g(cmd)? {
        Some(GPrior::Fixed(v)) if v == -1.0 => GPrior::Fixed(n as f64),
        Some(GPrior::Fixed(v)) if v == -2.0 => GPrior::Fixed(kk * kk),
        Some(GPrior::Fixed(v)) if v == -3.0 => GPrior::Fixed((n as f64).max(kk * kk)),
        Some(g) => g,
        None => GPrior::Fixed((n as f64).max(kk * kk)),
    };
    if let GPrior::Fixed(v) = g {
        if !(v > 0.0) {
            return Err(rc_err(198, "g must be positive"));
        }
    }
    if let GPrior::Hyper(a) = g {
        if !(a > 2.0) {
            return Err(rc_err(198, "hyper-g parameter must exceed 2"));
        }
    }
    let mp = match cmd.opt("mprior", 2) {
        None => MPrior::Uniform,
        Some(t) => {
            let w: Vec<String> = t.split_whitespace().map(|s| s.to_lowercase()).collect();
            let num = |i: usize, d: f64| -> Result<f64, String> { w.get(i).map_or(Ok(d), |s| s.parse::<f64>().map_err(|_| rc_err(198, "invalid mprior()"))) };
            match w.first().map(String::as_str) {
                Some("uniform") => MPrior::Uniform,
                Some("binomial") => {
                    let p = num(1, 0.5)?;
                    if !(p > 0.0 && p < 1.0) {
                        return Err(rc_err(198, "binomial inclusion probability must be in (0, 1)"));
                    }
                    MPrior::Binomial(p)
                }
                Some("betabinomial") => MPrior::BetaBinomial(num(1, 1.0)?, num(2, 1.0)?),
                _ => return Err(rc_err(198, "mprior() must be uniform, binomial # or betabinomial # #")),
            }
        }
    };
    if n < names.len() + 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    if tss <= 0.0 {
        return Err(rc_err(459, format!("{dep} does not vary")));
    }
    let always_idx: Vec<usize> = (0..always.len()).collect();
    Ok((Problem { n, x, names, y, ybar, xbar, tss, terms: tcols, always: always_idx, g, mp }, mask, terms.iter().map(|t| t.join(" ")).collect()))
}

fn bits(m: u64, k: usize) -> Vec<bool> {
    (0..k).map(|j| m >> j & 1 == 1).collect()
}

/// Visited models: inclusion bits, log posterior, posterior probability.
pub struct Models {
    pub inc: Vec<Vec<bool>>,
    pub logpost: Vec<f64>,
    pub pmp: Vec<f64>,
    pub sampled: bool,
    pub accept: f64,
}

fn explore(p: &Problem, sampling: bool, size: usize, burn: usize, st: &Store) -> Result<Models, String> {
    let k = p.kterms();
    if !sampling {
        if k > 20 {
            return Err(rc_err(198, "too many candidate terms to enumerate; use sampling"));
        }
        let mut inc = Vec::new();
        let mut lp = Vec::new();
        for m in 0..(1u64 << k) {
            let b = bits(m, k);
            if let Some(f) = p.fit(&b) {
                lp.push(p.log_post(&b, &f));
                inc.push(b);
            }
        }
        let mx = lp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let tot: f64 = lp.iter().map(|v| (v - mx).exp()).sum();
        let pmp = lp.iter().map(|v| (v - mx).exp() / tot).collect();
        return Ok(Models { inc, logpost: lp, pmp, sampled: false, accept: 1.0 });
    }
    let mut rng = st.rng.borrow_mut();
    let mut cur = vec![false; k];
    let f0 = p.fit(&cur).ok_or_else(|| rc_err(430, "null model could not be fit"))?;
    let mut lcur = p.log_post(&cur, &f0);
    let mut seen: std::collections::HashMap<Vec<bool>, (f64, usize)> = Default::default();
    let mut acc = 0usize;
    for it in 0..(burn + size) {
        let j = ((rng.uniform() * k as f64) as usize).min(k - 1);
        let mut prop = cur.clone();
        prop[j] = !prop[j];
        if let Some(f) = p.fit(&prop) {
            let lp = p.log_post(&prop, &f);
            if rng.uniform().ln() < lp - lcur {
                cur = prop;
                lcur = lp;
                if it >= burn {
                    acc += 1;
                }
            }
        }
        if it >= burn {
            let e = seen.entry(cur.clone()).or_insert((lcur, 0));
            e.1 += 1;
        }
    }
    let mut v: Vec<(Vec<bool>, f64, usize)> = seen.into_iter().map(|(a, (b, c))| (a, b, c)).collect();
    v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.total_cmp(&a.1)));
    let tot = size as f64;
    Ok(Models { pmp: v.iter().map(|x| x.2 as f64 / tot).collect(), logpost: v.iter().map(|x| x.1).collect(), inc: v.into_iter().map(|x| x.0).collect(), sampled: true, accept: acc as f64 / tot })
}

pub struct Bma {
    pub verb: &'static str,
}

impl CommandHandler for Bma {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        match self.verb {
            "bmaregress" => regress(ds, st, cmd),
            "bmastats" => stats(ds, st, cmd),
            "bmagraph" => graph(ds, st, cmd),
            "bmapredict" => predict(ds, st, cmd),
            _ => coefsample(ds, st, cmd),
        }
    }
}

fn gprior_label(g: &GPrior) -> String {
    match g {
        GPrior::Fixed(v) => format!("Fixed, g = {}", fmt_g(*v, 9)),
        GPrior::Hyper(a) => format!("Hyper-g, a = {}", fmt_g(*a, 6)),
        GPrior::Zs => "Zellner-Siow".into(),
    }
}

fn regress(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    if cmd.main.trim().is_empty() {
        return st.e.macros.get("_output").cloned().filter(|_| st.e.macros.get("cmd").is_some_and(|c| c == "bmaregress")).ok_or_else(|| rc_err(301, "last estimates not found"));
    }
    if let Some(s) = util::opt_num(cmd, "rseed", 5)? {
        st.rng.borrow_mut().reseed(s as u64);
    }
    let (p, mask, term_names) = problem(ds, st, cmd, cmd.main.trim(), None)?;
    let k = p.kterms();
    let sampling = cmd.has_opt("sampling", 4) || (k > 15 && !cmd.has_opt("enumeration", 4));
    let size = util::opt_num(cmd, "mcmcsize", 5)?.unwrap_or(10000.0) as usize;
    let burn = util::opt_num(cmd, "burnin", 4)?.unwrap_or(2500.0) as usize;
    let models = explore(&p, sampling, size, burn, st)?;
    // posterior moments
    let np = p.names.len();
    let mut mean = vec![0.0; np];
    let mut second = vec![0.0; np];
    let mut pip = vec![0.0; np];
    let mut msize = 0.0;
    let mut m2 = 0.0;
    let mut cons = 0.0;
    for (m, inc) in models.inc.iter().enumerate() {
        let w = models.pmp[m];
        if w == 0.0 {
            continue;
        }
        let f = p.fit(inc).unwrap();
        let (mu, var) = p.moments(&f);
        for j in 0..np {
            mean[j] += w * mu[j];
            second[j] += w * (var[j] + mu[j] * mu[j]);
        }
        for &c in &f.cols {
            pip[c] += w;
        }
        let sz = inc.iter().filter(|b| **b).count() as f64;
        msize += w * sz;
        m2 += w * sz * sz;
        cons += w * (p.ybar - (0..np).map(|j| mu[j] * p.xbar[j]).sum::<f64>());
    }
    let sd: Vec<f64> = (0..np).map(|j| (second[j] - mean[j] * mean[j]).max(0.0).sqrt()).collect();
    let prior_size: f64 = (0..=k).map(|s| s as f64 * p.size_prior(s)).sum();
    let mut out = vec![String::new()];
    let hdr = |a: &str, b: &str, v: String| format!("{:<45}{:<22}= {:>9}", a, b, v);
    out.push(hdr("Bayesian model averaging", "No. of obs", util::commas(p.n)));
    out.push(hdr("Linear regression", "No. of predictors", k.to_string()));
    out.push(hdr(if models.sampled { "Model enumeration: no (MC3 sampling)" } else { "Model enumeration: full" }, if models.sampled { "Models visited" } else { "No. of models" }, models.inc.len().to_string()));
    out.push(hdr("", "For CPMP >= .9", {
        let mut srt = models.pmp.clone();
        srt.sort_by(|a, b| b.total_cmp(a));
        let mut c = 0.0;
        let mut n9 = 0;
        for v in srt {
            n9 += 1;
            c += v;
            if c >= 0.9 {
                break;
            }
        }
        n9.to_string()
    }));
    out.push(hdr(&format!("Priors:"), "Mean model size", format!("{:.3}", msize)));
    out.push(format!("  Models: {}", match p.mp { MPrior::Uniform => "Uniform".to_string(), MPrior::Binomial(t) => format!("Binomial, IP = {}", fmt_g(t, 6)), MPrior::BetaBinomial(a, b) => format!("Beta-binomial({}, {})", fmt_g(a, 6), fmt_g(b, 6)) }));
    out.push(format!("   Cons.: Noninformative"));
    out.push(format!("   Coef.: Zellner's g"));
    out.push(format!("       g: {}", gprior_label(&p.g)));
    out.push(format!("  sigma2: Noninformative"));
    if models.sampled {
        out.push(format!("Sampling correlation / acceptance rate: {:.4}", models.accept));
    }
    out.push(String::new());
    out.push("-".repeat(62));
    out.push(format!("{:>12} |       Mean        Std. dev.     Group       PIP", p_dep(&cmd.main)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(48)));
    // order by PIP (descending) unless inputorder
    let mut order: Vec<usize> = (0..np).collect();
    if !cmd.has_opt("inputorder", 3) {
        order.sort_by(|&a, &b| pip[b].total_cmp(&pip[a]));
    }
    let group_of = |j: usize| -> String {
        if j < p.always.len() {
            return "always".into();
        }
        p.terms.iter().position(|t| t.contains(&j)).map_or(String::new(), |g| (g + 1).to_string())
    };
    for &j in &order {
        out.push(format!("{:>12} | {:>10}   {:>10}   {:>7}   {:>9}", crate::summarize::abbrev(&p.names[j], 12), fmt_g(mean[j], 9), fmt_g(sd[j], 9), group_of(j), format!("{:.5}", pip[j])));
    }
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(48)));
    out.push(format!("{:>12} | {:>10}   {:>10}", "Always", "", ""));
    out.push(format!("{:>12} | {:>10}   {:>10}   {:>7}   {:>9}", "_cons", fmt_g(cons, 9), "", "", "1"));
    out.push("-".repeat(62));
    out.push("Note: Coefficient posterior means and std. dev. are averaged over all models.".into());
    let text = out.join("\n");
    // post
    let mut names = p.names.clone();
    names.push("_cons".into());
    let mut b = mean.clone();
    b.push(cons);
    let mut sdv = sd.clone();
    sdv.push(SYSMISS);
    let mut pipv = pip.clone();
    pipv.push(1.0);
    let dep = parse_spec(ds, cmd.main.trim())?.0;
    let e = st.begin_e();
    e.set_str("cmd", "bmaregress");
    e.set_str("cmdline", format!("bmaregress {}", cmd.raw.trim_start_matches("bmaregress").trim()));
    e.set_str("depvar", dep);
    e.set_str("title", "Bayesian model averaging");
    e.set_str("_spec", cmd.main.trim());
    e.set_str("_gprior", match p.g { GPrior::Fixed(v) => format!("fixed {v:e}"), GPrior::Hyper(a) => format!("hyper {a:e}"), GPrior::Zs => "zs".into() });
    e.set_str("_mprior", match p.mp { MPrior::Uniform => "uniform".into(), MPrior::Binomial(t) => format!("binomial {t:e}"), MPrior::BetaBinomial(a, c) => format!("betabinomial {a:e} {c:e}") });
    e.set_str("terms", term_names.join(";"));
    e.set_str("_output", text.clone());
    e.set("N", p.n as f64);
    e.set("K", k as f64);
    e.set("N_models", models.inc.len() as f64);
    e.set("msize", msize);
    e.set("msize_sd", (m2 - msize * msize).max(0.0).sqrt());
    e.set("prior_msize", prior_size);
    e.set("sampling", if models.sampled { 1.0 } else { 0.0 });
    if let GPrior::Fixed(g) = p.g {
        e.set("g", g);
    }
    e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("sd", Matrix::row_vector(&sdv).with_names(vec!["sd".into()], names.clone()));
    e.set_mat("pip", Matrix::row_vector(&pipv).with_names(vec!["pip".into()], names.clone()));
    // models matrix (top 2000 by probability): inclusion bits, pmp, log posterior
    let mut idx: Vec<usize> = (0..models.inc.len()).collect();
    idx.sort_by(|&a, &c| models.pmp[c].total_cmp(&models.pmp[a]).then(models.logpost[c].total_cmp(&models.logpost[a])));
    idx.truncate(2000);
    let rows: Vec<Vec<f64>> = idx.iter().map(|&m| {
        let mut r: Vec<f64> = models.inc[m].iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
        r.push(models.pmp[m]);
        r.push(models.logpost[m]);
        r
    }).collect();
    let mut cn: Vec<String> = (1..=k).map(|t| format!("t{t}")).collect();
    cn.push("pmp".into());
    cn.push("logpost".into());
    e.set_mat("models", Matrix::from_rows(&rows).with_names((1..=rows.len()).map(|i| i.to_string()).collect(), cn));
    e.sample = Some(mask);
    let _ = (p.g_fixed(), dist::normal(0.0));
    Ok(text)
}

fn p_dep(main: &str) -> String {
    crate::summarize::abbrev(main.split_whitespace().next().unwrap_or(""), 12)
}

/// Rebuild the problem and the stored models after bmaregress.
fn stored(ds: &DataSet, st: &Store) -> Result<(Problem, Vec<Vec<bool>>, Vec<f64>), String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("bmaregress") {
        return Err(rc_err(301, "last estimates not found; run bmaregress first"));
    }
    let spec = st.e.macros.get("_spec").cloned().unwrap_or_default();
    let mut opts = vec![format!("gprior({})", st.e.macros.get("_gprior").cloned().unwrap_or_default()), format!("mprior({})", st.e.macros.get("_mprior").cloned().unwrap_or_default())];
    opts.retain(|o| !o.ends_with("()"));
    let line = format!("bmaregress {spec}, {}", opts.join(" "));
    let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
    let mask = st.e.sample.clone();
    let (p, _, _) = problem(ds, st, &c, &spec, mask.as_deref())?;
    let m = st.e.matrices.get("models").cloned().ok_or_else(|| rc_err(301, "e(models) not found"))?;
    let k = p.kterms();
    let inc: Vec<Vec<bool>> = (0..m.rows).map(|r| (0..k).map(|j| m[(r, j)] == 1.0).collect()).collect();
    let pmp: Vec<f64> = (0..m.rows).map(|r| m[(r, k)]).collect();
    Ok((p, inc, pmp))
}

fn model_vars(p: &Problem, inc: &[bool]) -> String {
    let mut v: Vec<String> = p.always.iter().map(|&j| p.names[j].clone()).collect();
    for (t, on) in inc.iter().enumerate() {
        if *on {
            v.extend(p.terms[t].iter().map(|&j| p.names[j].clone()));
        }
    }
    if v.is_empty() { "(constant only)".into() } else { v.join(" ") }
}

fn stats(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let (p, inc, pmp) = stored(ds, st)?;
    let k = p.kterms();
    let mut out = vec![String::new()];
    match sub.to_lowercase().as_str() {
        "models" | "model" => {
            let nshow = util::opt_num(cmd, "top", 3)?.unwrap_or(5.0) as usize;
            out.push("Computing model probabilities ...".into());
            out.push(String::new());
            out.push(format!("Model summary{:>40}Number of models:", ""));
            out.push(format!("{:>53}Visited = {}", "", inc.len()));
            out.push(format!("{:>53}Reported = {}", "", nshow.min(inc.len())));
            out.push(String::new());
            out.push("-".repeat(70));
            out.push(format!("{:>10} | {:>10} {:>10} {:>6}  Variables", "Rank", "PMP", "CPMP", "Size"));
            out.push(format!("{}+{}", "-".repeat(11), "-".repeat(58)));
            let mut c = 0.0;
            let mut rows = Vec::new();
            for r in 0..nshow.min(inc.len()) {
                c += pmp[r];
                let sz = inc[r].iter().filter(|b| **b).count();
                out.push(format!("{:>10} | {:>10.4} {:>10.4} {:>6}  {}", r + 1, pmp[r], c, sz, model_vars(&p, &inc[r])));
                rows.push(vec![pmp[r], c, sz as f64]);
            }
            out.push("-".repeat(70));
            st.begin_r().set_mat("modelstats", Matrix::from_rows(&rows).with_names((1..=rows.len()).map(|i| i.to_string()).collect(), vec!["PMP".into(), "CPMP".into(), "Size".into()]));
        }
        "msize" => {
            let mut post = vec![0.0; k + 1];
            for (m, b) in inc.iter().enumerate() {
                post[b.iter().filter(|x| **x).count()] += pmp[m];
            }
            let prior: Vec<f64> = (0..=k).map(|s| p.size_prior(s)).collect();
            let pm: f64 = (0..=k).map(|s| s as f64 * prior[s]).sum();
            let qm: f64 = (0..=k).map(|s| s as f64 * post[s]).sum();
            out.push("Model-size summary".into());
            out.push(String::new());
            out.push(format!("{:>14} | {:>10} {:>10}", "Model size", "Prior", "Posterior"));
            out.push(format!("{}+{}", "-".repeat(15), "-".repeat(22)));
            out.push(format!("{:>14} | {:>10.4} {:>10.4}", "Mean", pm, qm));
            for s in 0..=k {
                out.push(format!("{:>14} | {:>10.4} {:>10.4}", s, prior[s], post[s]));
            }
            let r = st.begin_r();
            r.set("prior_mean", pm);
            r.set("post_mean", qm);
            r.set_mat("msize", Matrix::from_rows(&(0..=k).map(|s| vec![prior[s], post[s]]).collect::<Vec<_>>()).with_names((0..=k).map(|s| s.to_string()).collect(), vec!["prior".into(), "posterior".into()]));
        }
        "pip" => {
            let mut pip = vec![0.0; k];
            for (m, b) in inc.iter().enumerate() {
                for t in 0..k {
                    if b[t] {
                        pip[t] += pmp[m];
                    }
                }
            }
            let tn: Vec<String> = p.terms.iter().map(|t| t.iter().map(|&j| p.names[j].clone()).collect::<Vec<_>>().join(" ")).collect();
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| pip[b].total_cmp(&pip[a]));
            out.push("Posterior inclusion probabilities".into());
            out.push(String::new());
            out.push(format!("{:>20} | {:>8}", "", "PIP"));
            out.push(format!("{}+{}", "-".repeat(21), "-".repeat(10)));
            for &t in &order {
                out.push(format!("{:>20} | {:>8.5}", crate::summarize::abbrev(&tn[t], 20), pip[t]));
            }
            st.begin_r().set_mat("pip", Matrix::row_vector(&pip).with_names(vec!["pip".into()], tn));
        }
        "jointness" | "joint" => {
            let names: Vec<String> = p.terms.iter().map(|t| p.names[t[0]].clone()).collect();
            let sel: Vec<usize> = if rest.trim().is_empty() {
                (0..k).collect()
            } else {
                crate::varlist::expand(ds, &rest).map_err(|e| rc_err(111, e))?.iter().map(|v| names.iter().position(|n| n == v).ok_or_else(|| rc_err(198, format!("{v} is not a candidate predictor")))).collect::<Result<_, _>>()?
            };
            if sel.len() < 2 {
                return Err(rc_err(198, "at least two predictors required"));
            }
            out.push("Jointness of predictor pairs".into());
            out.push(String::new());
            out.push(format!("{:>24} | {:>12} {:>12} {:>12}", "Pair", "P(both)", "Doppelhofer", "Ley-Steel"));
            out.push(format!("{}+{}", "-".repeat(25), "-".repeat(40)));
            let mut rows = Vec::new();
            let mut rn = Vec::new();
            for a in 0..sel.len() {
                for b in a + 1..sel.len() {
                    let (i, j) = (sel[a], sel[b]);
                    let (mut both, mut oi, mut oj, mut none) = (0.0, 0.0, 0.0, 0.0);
                    for (m, x) in inc.iter().enumerate() {
                        match (x[i], x[j]) {
                            (true, true) => both += pmp[m],
                            (true, false) => oi += pmp[m],
                            (false, true) => oj += pmp[m],
                            _ => none += pmp[m],
                        }
                    }
                    let dw = (both * none / (oi * oj)).ln();
                    let ls = both / (oi + oj);
                    let pair = format!("{} {}", names[i], names[j]);
                    out.push(format!("{:>24} | {:>12.4} {:>12} {:>12}", crate::summarize::abbrev(&pair, 24), both, fmt_g(if dw.is_finite() { dw } else { SYSMISS }, 9), fmt_g(if ls.is_finite() { ls } else { SYSMISS }, 9)));
                    rows.push(vec![both, if dw.is_finite() { dw } else { SYSMISS }, if ls.is_finite() { ls } else { SYSMISS }]);
                    rn.push(pair.replace(' ', "_"));
                }
            }
            st.begin_r().set_mat("jointness", Matrix::from_rows(&rows).with_names(rn, vec!["both".into(), "dw".into(), "ls".into()]));
        }
        "lps" => {
            // log predictive score over the estimation (or if/in) sample
            let mask = if cmd.if_expr.is_some() || cmd.in_range.is_some() { util::touse(ds, st, cmd)? } else { st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]) };
            let dens = predictive(ds, st, &p, &inc, &pmp, &mask, true)?;
            let lps: Vec<f64> = dens.iter().filter(|d| !is_missing(**d)).map(|d| -d.ln()).collect();
            let mean = util::mean(&lps);
            let mn = lps.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            out.push("Log predictive-score (LPS)".into());
            out.push(String::new());
            out.push(format!("{:>10} | {:>10} {:>10} {:>10}", "", "Mean", "Minimum", "Maximum"));
            out.push(format!("{}+{}", "-".repeat(11), "-".repeat(33)));
            out.push(format!("{:>10} | {:>10.4} {:>10.4} {:>10.4}", "LPS", mean, mn, mx));
            let r = st.begin_r();
            r.set("mean", mean);
            r.set("min", mn);
            r.set("max", mx);
            r.set("N", lps.len() as f64);
        }
        o => return Err(rc_err(198, format!("bmastats {o} is unrecognized; use models, msize, pip, jointness or lps"))),
    }
    Ok(out.join("\n"))
}

/// BMA predictive means (or densities of the observed outcome) for rows in `mask`.
fn predictive(ds: &DataSet, st: &Store, p: &Problem, inc: &[Vec<bool>], pmp: &[f64], mask: &[bool], density: bool) -> Result<Vec<f64>, String> {
    let _ = st;
    let cols: Vec<Vec<f64>> = p.names.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let dep = p_dep_name(st);
    let yc = util::col(ds, &dep)?;
    let tot: f64 = pmp.iter().sum();
    let fits: Vec<(f64, Fit)> = inc.iter().zip(pmp).filter(|(_, w)| **w > 0.0).filter_map(|(b, w)| p.fit(b).map(|f| (*w / tot, f))).collect();
    let mut out = vec![SYSMISS; ds.n_obs];
    for i in 0..ds.n_obs {
        if !mask[i] || cols.iter().any(|c| is_missing(c[i])) {
            continue;
        }
        let xc: Vec<f64> = (0..p.names.len()).map(|j| cols[j][i] - p.xbar[j]).collect();
        let mut acc = 0.0;
        for (w, f) in &fits {
            let (mu, _) = p.moments(f);
            let m = p.ybar + xc.iter().zip(&mu).map(|(a, b)| a * b).sum::<f64>();
            if density {
                if is_missing(yc[i]) {
                    acc = SYSMISS;
                    break;
                }
                // predictive variance: sigma2 (1 + 1/n + delta x'(X'X)^-1 x)
                let mut h = 1.0 / p.n as f64;
                for (a, &ca) in f.cols.iter().enumerate() {
                    for (b, &cb) in f.cols.iter().enumerate() {
                        h += f.delta * xc[ca] * f.xtxi[(a, b)] * xc[cb];
                    }
                }
                let s = (p.sigma2(f) * (1.0 + h)).sqrt();
                acc += w * dist::normalden((yc[i] - m) / s) / s;
            } else {
                acc += w * m;
            }
        }
        out[i] = acc;
    }
    Ok(out)
}

fn p_dep_name(st: &Store) -> String {
    st.e.macros.get("depvar").cloned().unwrap_or_default()
}

fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let name = cmd.main.split_whitespace().last().ok_or_else(|| rc_err(100, "new variable name required"))?.to_string();
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let (p, inc, pmp) = stored(ds, st)?;
    let mask = util::touse(ds, st, cmd)?;
    let v = predictive(ds, st, &p, &inc, &pmp, &mask, false)?;
    let nmiss = v.iter().filter(|x| is_missing(**x)).count();
    ds.add_variable(util::new_double(&name, v)).map_err(|e| rc_err(110, e))?;
    Ok(if nmiss > 0 { format!("({nmiss} missing value{} generated)", if nmiss == 1 { "" } else { "s" }) } else { String::new() })
}

fn graph(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let (p, inc, pmp) = stored(ds, st)?;
    let k = p.kterms();
    let mut g = Graph::default();
    match sub.to_lowercase().as_str() {
        "pmp" => {
            let n = inc.len().min(util::opt_num(cmd, "top", 3)?.unwrap_or(100.0) as usize);
            let mut c = 0.0;
            let cum: Vec<(f64, f64)> = (0..n).map(|r| {
                c += pmp[r];
                ((r + 1) as f64, c)
            }).collect();
            g.layers.push(Layer { kind: LayerKind::Line, points: cum, label: "CPMP".into(), ..Default::default() });
            g.layers.push(Layer { kind: LayerKind::Scatter, points: (0..n).map(|r| ((r + 1) as f64, pmp[r])).collect(), label: "PMP".into(), ..Default::default() });
            g.xtitle = "Rank".into();
            g.ytitle = "Posterior model probability".into();
            g.title = "Posterior model probabilities".into();
            g.legend = true;
        }
        "msize" => {
            let mut post = vec![0.0; k + 1];
            for (m, b) in inc.iter().enumerate() {
                post[b.iter().filter(|x| **x).count()] += pmp[m];
            }
            g.layers.push(Layer { kind: LayerKind::Connected, points: (0..=k).map(|s| (s as f64, p.size_prior(s))).collect(), label: "Prior".into(), ..Default::default() });
            g.layers.push(Layer { kind: LayerKind::Connected, points: (0..=k).map(|s| (s as f64, post[s])).collect(), label: "Posterior".into(), ..Default::default() });
            g.xtitle = "Model size".into();
            g.ytitle = "Probability".into();
            g.title = "Model-size distribution".into();
            g.legend = true;
        }
        "varmap" => {
            let n = inc.len().min(util::opt_num(cmd, "top", 3)?.unwrap_or(100.0) as usize);
            let mut cells = Vec::new();
            let mut colors = Vec::new();
            let mut x0 = 0.0;
            for r in 0..n {
                for t in 0..k {
                    if inc[r][t] {
                        cells.push([x0, t as f64, x0 + pmp[r], t as f64 + 0.9]);
                        colors.push((0, 114, 178));
                    }
                }
                x0 += pmp[r];
            }
            g.layers.push(Layer { kind: LayerKind::Surface, cells, cell_colors: colors, label: "Included".into(), ..Default::default() });
            g.ylabels = (0..k).map(|t| (t as f64 + 0.45, p.terms[t].iter().map(|&j| p.names[j].clone()).collect::<Vec<_>>().join(" "))).collect();
            g.xtitle = "Cumulative posterior model probability".into();
            g.title = "Variable-inclusion map".into();
        }
        "coefdensity" | "coefd" => {
            let vars = crate::varlist::expand(ds, &rest).map_err(|e| rc_err(111, e))?;
            if vars.is_empty() {
                return Err(rc_err(100, "varlist required"));
            }
            let mut zero_mass = Vec::new();
            for v in &vars {
                let j = p.names.iter().position(|n| n == v).ok_or_else(|| rc_err(198, format!("{v} is not a predictor")))?;
                // mixture of the models' normal approximations, weighted by PMP
                let comps: Vec<(f64, f64, f64)> = inc.iter().zip(&pmp).filter(|(_, w)| **w > 0.0).filter_map(|(b, w)| {
                    let f = p.fit(b)?;
                    let (m, var) = p.moments(&f);
                    if f.cols.contains(&j) { Some((*w, m[j], var[j].sqrt())) } else { None }
                }).collect();
                let pin: f64 = comps.iter().map(|c| c.0).sum();
                zero_mass.push(1.0 - pin);
                if comps.is_empty() {
                    continue;
                }
                let lo = comps.iter().map(|c| c.1 - 4.0 * c.2).fold(f64::INFINITY, f64::min);
                let hi = comps.iter().map(|c| c.1 + 4.0 * c.2).fold(f64::NEG_INFINITY, f64::max);
                let pts: Vec<(f64, f64)> = (0..=200).map(|i| {
                    let x = lo + (hi - lo) * i as f64 / 200.0;
                    (x, comps.iter().map(|(w, m, s)| w * dist::normalden((x - m) / s) / s).sum::<f64>())
                }).collect();
                g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: v.clone(), ..Default::default() });
            }
            g.xlines.push(0.0);
            g.xtitle = "Coefficient".into();
            g.ytitle = "Posterior density (continuous part)".into();
            g.title = "Coefficient posterior densities".into();
            g.legend = vars.len() > 1;
            st.begin_r().set_mat("P0", Matrix::row_vector(&zero_mass).with_names(vec!["P(=0)".into()], vars.clone()));
        }
        o => return Err(rc_err(198, format!("bmagraph {o} is unrecognized; use pmp, msize, varmap or coefdensity"))),
    }
    g.command = format!("bmagraph {}", cmd.main.trim());
    crate::graph::finish(&mut g, cmd, st)
}

/// Draw a sample of coefficients from the BMA posterior (normal
/// approximation within each model), store it in e(coefsample) and
/// summarize it.
fn coefsample(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (p, inc, pmp) = stored(ds, st)?;
    if let Some(s) = util::opt_num(cmd, "rseed", 5)? {
        st.rng.borrow_mut().reseed(s as u64);
    }
    let size = util::opt_num(cmd, "mcmcsize", 5)?.unwrap_or(10000.0) as usize;
    let tot: f64 = pmp.iter().sum();
    let fits: Vec<(f64, Vec<f64>, Vec<f64>)> = inc.iter().zip(&pmp).filter(|(_, w)| **w > 0.0).filter_map(|(b, w)| {
        let f = p.fit(b)?;
        let (m, v) = p.moments(&f);
        Some((*w / tot, m, v))
    }).collect();
    let np = p.names.len();
    let mut draws = Vec::with_capacity(size);
    {
        let mut rng = st.rng.borrow_mut();
        for _ in 0..size {
            let u = rng.uniform();
            let mut c = 0.0;
            let mut pick = fits.len() - 1;
            for (i, f) in fits.iter().enumerate() {
                c += f.0;
                if u < c {
                    pick = i;
                    break;
                }
            }
            let (_, m, v) = &fits[pick];
            draws.push((0..np).map(|j| if v[j] > 0.0 { m[j] + v[j].sqrt() * rng.normal() } else { 0.0 }).collect::<Vec<f64>>());
        }
    }
    let mut out = vec![String::new(), format!("Simulated {} draws of the BMA coefficient posterior", util::commas(size)), String::new()];
    out.push(format!("{:>12} | {:>10} {:>10} {:>10} {:>10}  [95% cred. interval]", "", "Mean", "Std. dev.", "Median", "P(=0)"));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(70)));
    let mut summ = Vec::new();
    for j in 0..np {
        let mut col: Vec<f64> = draws.iter().map(|d| d[j]).collect();
        let m = util::mean(&col);
        let s = util::variance(&col).sqrt();
        let p0 = col.iter().filter(|x| **x == 0.0).count() as f64 / size as f64;
        col.sort_by(|a, b| a.total_cmp(b));
        let q = |pr: f64| util::pctile_sorted(&col, pr);
        out.push(format!("{:>12} | {:>10} {:>10} {:>10} {:>10.4}  {:>10} {:>10}", crate::summarize::abbrev(&p.names[j], 12), fmt_g(m, 8), fmt_g(s, 8), fmt_g(q(50.0), 8), p0, fmt_g(q(2.5), 8), fmt_g(q(97.5), 8)));
        summ.push(vec![m, s, q(50.0), q(2.5), q(97.5), p0]);
    }
    st.e.set_mat("coefsample", Matrix::from_rows(&draws).with_names((1..=size).map(|i| i.to_string()).collect(), p.names.clone()));
    st.e.set("mcmcsize", size as f64);
    st.begin_r().set_mat("summary", Matrix::from_rows(&summ).with_names(p.names.clone(), ["mean", "sd", "median", "lb", "ub", "p0"].iter().map(|s| s.to_string()).collect()));
    Ok(out.join("\n"))
}
