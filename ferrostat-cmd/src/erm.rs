//! Extended regression models: `eregress`, `eprobit`, `eoprobit`,
//! `eintreg` (linear, probit, ordered probit and interval outcomes with
//! endogenous covariates, endogenous or exogenous binary treatment and
//! probit sample selection), and their random-effects panel versions
//! `xteregress`, `xteprobit`, `xteoprobit`, `xteintreg` and `xtheckman`.
//!
//! Every equation has a latent Gaussian error; the errors are jointly
//! normal with unrestricted correlations. An observation contributes the
//! joint density of its continuous components times the probability that
//! its discrete components (binary, ordinal, censored) fall in their
//! observed intervals conditional on the continuous ones, so at most two
//! discrete components are allowed. Variances are estimated as log
//! standard deviations and correlations as Fisher z, and reported on the
//! natural scale (delta method).
//!
//! The panel versions add a random intercept to each equation, jointly
//! normal across equations, integrated out by adaptive Gauss-Hermite
//! quadrature (mode and curvature of each panel's integrand, product
//! rule), and fitted with numerical derivatives of the panel
//! log likelihoods.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, Likelihood, MlResult, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels as xm;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = Vec::new();
    for (verb, panel) in [
        ("eregress", false),
        ("eprobit", false),
        ("eoprobit", false),
        ("eintreg", false),
        ("xteregress", true),
        ("xteprobit", true),
        ("xteoprobit", true),
        ("xteintreg", true),
        ("xtheckman", true),
    ] {
        v.push((verb, Box::new(Erm { verb, panel })));
    }
    v
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Outcome {
    Linear,
    Probit,
    Oprobit,
    Interval,
}

/// Observed state of one component for one observation.
#[derive(Clone, Copy, Debug)]
enum Ob {
    Point(f64),
    /// latent in (lo, hi)
    Range(f64, f64),
    /// ordinal category (cutpoints in aux)
    Cat(usize),
    Absent,
}

/// The likelihood structure shared by the cross-section and panel fits.
struct Model {
    m: usize,
    /// aux index of the log standard deviation of each component (None: 1)
    lnsd: Vec<Option<usize>>,
    /// aux index of the Fisher z of each pair (i < j)
    corr: Vec<Vec<usize>>,
    ncut: usize,
    obs: Vec<Vec<Ob>>,
    patterns: Vec<(Vec<usize>, Vec<usize>)>,
    pat_of: Vec<usize>,
}

/// Standard normal CDF that accepts infinite arguments.
fn ncdf(x: f64) -> f64 {
    if x == f64::INFINITY {
        1.0
    } else if x == f64::NEG_INFINITY {
        0.0
    } else {
        dist::normal(x)
    }
}

fn rect1(a: f64, b: f64) -> f64 {
    // upper tail computed directly for accuracy
    if b == f64::INFINITY {
        ncdf(-a)
    } else {
        (ncdf(b) - ncdf(a)).max(0.0)
    }
}

fn cdf2(x: f64, y: f64, r: f64) -> f64 {
    if x == f64::NEG_INFINITY || y == f64::NEG_INFINITY {
        0.0
    } else if x == f64::INFINITY && y == f64::INFINITY {
        1.0
    } else if x == f64::INFINITY {
        ncdf(y)
    } else if y == f64::INFINITY {
        ncdf(x)
    } else {
        crate::choice_mixed::bvn(x, y, r)
    }
}

fn rect2(a1: f64, b1: f64, a2: f64, b2: f64, r: f64) -> f64 {
    (cdf2(b1, b2, r) - cdf2(a1, b2, r) - cdf2(b1, a2, r) + cdf2(a1, a2, r)).max(0.0)
}

/// Cholesky factor (lower) of a small SPD matrix.
pub(crate) fn chol(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve L L' x = b.
fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

impl Model {
    fn sigma(&self, aux: &[f64]) -> Vec<Vec<f64>> {
        let m = self.m;
        let sd: Vec<f64> = self.lnsd.iter().map(|o| o.map_or(1.0, |j| aux[j].exp())).collect();
        let mut s = vec![vec![0.0; m]; m];
        for i in 0..m {
            s[i][i] = sd[i] * sd[i];
            for j in 0..i {
                let r = aux[self.corr[j][i]].tanh();
                s[i][j] = r * sd[i] * sd[j];
                s[j][i] = s[i][j];
            }
        }
        s
    }

    /// Patterns of continuous and discrete components and the pattern of each observation.
    fn with_patterns(mut self) -> Model {
        let mut pats: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut of = Vec::with_capacity(self.obs.len());
        for ob in &self.obs {
            let mut c = Vec::new();
            let mut d = Vec::new();
            for (j, o) in ob.iter().enumerate() {
                match o {
                    Ob::Point(_) => c.push(j),
                    Ob::Range(..) | Ob::Cat(_) => d.push(j),
                    Ob::Absent => {}
                }
            }
            let key = (c, d);
            let k = match pats.iter().position(|p| *p == key) {
                Some(k) => k,
                None => {
                    pats.push(key);
                    pats.len() - 1
                }
            };
            of.push(k);
        }
        self.patterns = pats;
        self.pat_of = of;
        self
    }

    /// Conditional-distribution pieces of every pattern (None: covariance not PD).
    fn prep(&self, aux: &[f64]) -> Option<Vec<PatPrep>> {
        let sig = self.sigma(aux);
        chol(&sig)?;
        self.patterns.iter().map(|(c, d)| prep_pattern(&sig, c, d)).collect()
    }

    /// Log likelihood of observation t given the linear indexes and aux.
    fn ll(&self, t: usize, eta: &[f64], aux: &[f64]) -> f64 {
        let sig = self.sigma(aux);
        let (c, d) = &self.patterns[self.pat_of[t]];
        match prep_pattern(&sig, c, d) {
            Some(p) => self.ll_prep(t, eta, aux, &p),
            None => f64::NEG_INFINITY,
        }
    }

    fn ll_prep(&self, t: usize, eta: &[f64], aux: &[f64], pp: &PatPrep) -> f64 {
        let ob = &self.obs[t];
        let (c, d) = &self.patterns[self.pat_of[t]];
        let mut ll = 0.0;
        let mut mu = [0.0f64; 8];
        if !c.is_empty() {
            let nc = c.len();
            let mut z = [0.0f64; 8];
            let mut q = 0.0;
            for i in 0..nc {
                let u = if let Ob::Point(v) = ob[c[i]] { v - eta[c[i]] } else { 0.0 };
                let mut zi = u;
                for k in 0..i {
                    zi -= pp.l[i][k] * z[k];
                }
                zi /= pp.l[i][i];
                z[i] = zi;
                q += zi * zi;
            }
            ll += -0.5 * (nc as f64 * (2.0 * std::f64::consts::PI).ln() + pp.logdet + q);
            // mu = A u with A = S_dc S_cc^-1 = (S_dc L'^-1) L^-1, B = S_dc L'^-1
            for p in 0..d.len() {
                mu[p] = (0..nc).map(|k| pp.b[p][k] * z[k]).sum();
            }
        }
        if d.is_empty() {
            return ll;
        }
        let mut bounds = [(0.0f64, 0.0f64); 2];
        for (p, &j) in d.iter().enumerate().take(2) {
            let (lo, hi) = match ob[j] {
                Ob::Range(a, b) => (a, b),
                Ob::Cat(k) => (if k == 0 { f64::NEG_INFINITY } else { aux[k - 1] }, if k >= self.ncut { f64::INFINITY } else { aux[k] }),
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            };
            let s = pp.sd[p];
            bounds[p] = ((lo - eta[j] - mu[p]) / s, (hi - eta[j] - mu[p]) / s);
        }
        let p = match d.len() {
            1 => rect1(bounds[0].0, bounds[0].1),
            2 => rect2(bounds[0].0, bounds[0].1, bounds[1].0, bounds[1].1, pp.r),
            _ => return f64::NEG_INFINITY,
        };
        ll + p.max(1e-300).ln()
    }
}

/// Pieces of the conditional distribution of the discrete components given
/// the continuous ones for one pattern.
pub(crate) struct PatPrep {
    /// Cholesky factor of S_cc
    l: Vec<Vec<f64>>,
    logdet: f64,
    /// S_dc L'^-1 (so that the conditional mean is B z with z = L^-1 u)
    b: Vec<Vec<f64>>,
    /// conditional standard deviations and correlation of the discrete components
    sd: Vec<f64>,
    r: f64,
}

fn prep_pattern(sig: &[Vec<f64>], c: &[usize], d: &[usize]) -> Option<PatPrep> {
    let scc: Vec<Vec<f64>> = c.iter().map(|&a| c.iter().map(|&b| sig[a][b]).collect()).collect();
    let l = if c.is_empty() { Vec::new() } else { chol(&scc)? };
    let logdet = (0..c.len()).map(|i| l[i][i].ln()).sum::<f64>() * 2.0;
    // B = S_dc L'^-1: solve L B' = S_cd
    let b: Vec<Vec<f64>> = d
        .iter()
        .map(|&dj| {
            let mut z = vec![0.0; c.len()];
            for i in 0..c.len() {
                z[i] = (sig[dj][c[i]] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i];
            }
            z
        })
        .collect();
    let cov = |p: usize, q: usize| sig[d[p]][d[q]] - b[p].iter().zip(&b[q]).map(|(x, y)| x * y).sum::<f64>();
    let sd: Vec<f64> = (0..d.len()).map(|p| cov(p, p).max(1e-300).sqrt()).collect();
    let r = if d.len() == 2 { (cov(0, 1) / (sd[0] * sd[1])).clamp(-0.999999, 0.999999) } else { 0.0 };
    Some(PatPrep { l, logdet, b, sd, r })
}

struct Erm {
    verb: &'static str,
    panel: bool,
}

/// One `name(... = ..., subopts)` option split into (lhs, rhs, subopts).
fn opt_parts(s: &str) -> (String, Option<String>, String) {
    let (body, sub) = match s.split_once(',') {
        Some((a, b)) => (a.to_string(), b.to_lowercase()),
        None => (s.to_string(), String::new()),
    };
    match body.split_once('=') {
        Some((a, b)) => (a.trim().to_string(), Some(b.trim().to_string()), sub),
        None => (body.trim().to_string(), None, sub),
    }
}

fn all_opts(cmd: &Command, name: &str, min: usize) -> Vec<String> {
    cmd.opts
        .iter()
        .filter(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min && name.starts_with(n.as_str())
        })
        .map(|(_, a)| a.clone().unwrap_or_default())
        .collect()
}

fn has_word(sub: &str, w: &str, min: usize) -> bool {
    sub.split(|c: char| c.is_whitespace() || c == ',').any(|x| x.len() >= min && w.starts_with(x))
}

#[derive(Clone)]
struct EqSpec {
    name: String,
    /// dependent variable read as data (None: outcome handled separately)
    dep: Option<String>,
    indep: String,
    nocons: bool,
    kind: EqKind,
}

#[derive(Clone, Copy, PartialEq)]
enum EqKind {
    Outcome,
    EndogCont,
    EndogProbit,
    Treat,
    Select,
}

struct Prepared {
    designs: Vec<Design>,
    specs: Vec<EqSpec>,
    model: Model,
    aux_names: Vec<String>,
    b0: Vec<f64>,
    outcome: Outcome,
    deps: Vec<String>,
    cats: Vec<f64>,
    n_sel: Option<(usize, usize)>,
    /// panel id per observation (panel versions)
    panel: Option<Vec<f64>>,
    treat: Option<String>,
}

impl Erm {
    fn outcome(&self) -> Outcome {
        match self.verb.trim_start_matches("xt") {
            "eprobit" => Outcome::Probit,
            "eoprobit" => Outcome::Oprobit,
            "eintreg" => Outcome::Interval,
            _ => Outcome::Linear,
        }
    }

    fn title(&self, p: &Prepared) -> String {
        let base = match p.outcome {
            Outcome::Linear => "linear regression",
            Outcome::Probit => "probit regression",
            Outcome::Oprobit => "ordered probit regression",
            Outcome::Interval => "interval regression",
        };
        if self.verb == "xtheckman" {
            return "Random-effects regression with selection".into();
        }
        if self.panel {
            format!("Extended {base} with random effects")
        } else {
            let mut t = format!("Extended {base}");
            t[..1].make_ascii_uppercase();
            t
        }
    }

    fn prepare(&self, ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<Prepared, String> {
        let outcome = self.outcome();
        let main = cmd.main.trim();
        let (deps, xs): (Vec<String>, String) = if outcome == Outcome::Interval {
            let w: Vec<&str> = main.split_whitespace().collect();
            if w.len() < 2 {
                return Err(rc_err(100, "two dependent variables required"));
            }
            (vec![xm::var_name(ds, w[0])?, xm::var_name(ds, w[1])?], w[2..].join(" "))
        } else {
            let (d, x) = xm::first_word(main);
            if d.is_empty() {
                return Err(rc_err(100, "varlist required"));
            }
            (vec![xm::var_name(ds, &d)?], x)
        };
        let nocons = cmd.has_opt("noconstant", 3) || outcome == Outcome::Oprobit;
        let mut specs: Vec<EqSpec> = Vec::new();
        let mut main_x: Vec<String> = crate::tsops::expand_text(&xs).split_whitespace().map(String::from).collect();
        // endogenous covariates
        let mut endog: Vec<EqSpec> = Vec::new();
        for o in all_opts(cmd, "endogenous", 5) {
            let (lhs, rhs, sub) = opt_parts(&o);
            let rhs = rhs.ok_or_else(|| rc_err(198, "endogenous() requires depvars = varlist"))?;
            if has_word(&sub, "oprobit", 2) {
                return Err(rc_err(198, "ordered probit endogenous covariates are not supported"));
            }
            let probit = has_word(&sub, "probit", 2);
            for w in lhs.split_whitespace() {
                let w = xm::var_name(ds, w)?;
                if !has_word(&sub, "nomain", 4) {
                    main_x.push(w.clone());
                }
                endog.push(EqSpec { name: w.clone(), dep: Some(w), indep: rhs.clone(), nocons: has_word(&sub, "noconstant", 3), kind: if probit { EqKind::EndogProbit } else { EqKind::EndogCont } });
            }
        }
        // treatment
        let mut treat: Option<EqSpec> = None;
        let mut interact: Option<String> = None;
        if let Some(o) = cmd.opt("entreat", 3) {
            let (lhs, rhs, sub) = opt_parts(o);
            let tv = xm::var_name(ds, &lhs)?;
            if has_word(&sub, "pocorrelation", 3) {
                return Err(rc_err(198, "pocorrelation is not supported"));
            }
            if !has_word(&sub, "nointeract", 5) {
                interact = Some(tv.clone());
            } else {
                main_x.push(format!("i.{tv}"));
            }
            treat = Some(EqSpec { name: tv.clone(), dep: Some(tv), indep: rhs.unwrap_or_default(), nocons: has_word(&sub, "noconstant", 3), kind: EqKind::Treat });
        }
        if let Some(o) = cmd.opt("extreat", 3) {
            if treat.is_some() {
                return Err(rc_err(198, "entreat() and extreat() may not be combined"));
            }
            let (lhs, _, sub) = opt_parts(o);
            let tv = xm::var_name(ds, &lhs)?;
            if !has_word(&sub, "nointeract", 5) {
                interact = Some(tv);
            } else {
                main_x.push(format!("i.{tv}"));
            }
        }
        let mut select: Option<EqSpec> = None;
        if let Some(o) = cmd.opt("select", 3) {
            let (lhs, rhs, sub) = opt_parts(o);
            if has_word(&sub, "tobit", 3) {
                return Err(rc_err(198, "tobit selection is not supported"));
            }
            let (sdep, z) = match rhs {
                Some(r) => (Some(xm::var_name(ds, &lhs)?), r),
                None => (None, lhs),
            };
            select = Some(EqSpec { name: sdep.clone().unwrap_or_else(|| "select".into()), dep: sdep, indep: z, nocons: has_word(&sub, "noconstant", 3), kind: EqKind::Select });
        }
        if self.verb == "xtheckman" && select.is_none() {
            return Err(rc_err(198, "option select() required"));
        }
        // outcome regressors (interacted with the treatment)
        let outcome_indep = match &interact {
            Some(tv) => {
                let mut t: Vec<String> = Vec::new();
                if !nocons {
                    t.push(format!("ibn.{tv}"));
                }
                for x in &main_x {
                    if x.contains('.') && x.split('.').next().is_some_and(|p| p.to_lowercase().starts_with('i')) {
                        t.push(format!("ibn.{tv}#{x}"));
                    } else {
                        t.push(format!("ibn.{tv}#c.{x}"));
                    }
                }
                t.join(" ")
            }
            None => main_x.join(" "),
        };
        let out_nocons = nocons || interact.is_some();
        specs.push(EqSpec { name: deps[0].clone(), dep: None, indep: outcome_indep, nocons: out_nocons, kind: EqKind::Outcome });
        specs.extend(endog);
        if let Some(t) = treat.clone() {
            specs.push(t);
        }
        if let Some(s) = select.clone() {
            specs.push(s);
        }
        // panel variable
        let pvar = if self.panel { Some(crate::xtnl::panel_var(ds, st, cmd)?) } else { None };
        let mut extra: Vec<String> = Vec::new();
        if let Some(p) = &pvar {
            extra.push(p.clone());
        }
        if let Some(tv) = &interact {
            extra.push(tv.clone());
        }
        // outcome may be missing when selection is modelled or censored (interval)
        let dep_in_design = select.is_none() && outcome != Outcome::Interval;
        let spec_tuples: Vec<(Option<&str>, &str, bool)> = specs
            .iter()
            .map(|s| {
                let dep = if s.kind == EqKind::Outcome && dep_in_design { Some(deps[0].as_str()) } else { s.dep.as_deref() };
                (dep, s.indep.as_str(), s.nocons)
            })
            .collect();
        let mut designs = xm::build_eqs(ds, st, cmd, &spec_tuples, "fpi", &extra)?;
        let n0 = designs[0].n();
        if designs.iter().any(|d| d.rows != designs[0].rows) {
            return Err(rc_err(459, "equations have different estimation samples"));
        }
        // outcome values
        let ycols: Vec<Vec<f64>> = deps.iter().map(|d| util::col(ds, d)).collect::<Result<_, _>>()?;
        let rows = designs[0].rows.clone();
        let sel_idx = specs.iter().position(|s| s.kind == EqKind::Select);
        let sel: Vec<bool> = match (sel_idx, &select) {
            (Some(j), Some(s)) if s.dep.is_some() => designs[j].y.iter().map(|v| *v != 0.0).collect(),
            (Some(_), _) => rows.iter().map(|&r| !is_missing(ycols[0][r])).collect(),
            _ => vec![true; n0],
        };
        let mut keep = vec![true; n0];
        for t in 0..n0 {
            let r = rows[t];
            let miss = if outcome == Outcome::Interval { is_missing(ycols[0][r]) && is_missing(ycols[1][r]) } else { is_missing(ycols[0][r]) };
            if sel[t] && miss {
                if select.is_some() {
                    return Err(rc_err(459, format!("{} missing for selected observations", deps[0])));
                }
                keep[t] = false;
            }
        }
        let (designs2, sel2) = if keep.iter().all(|k| *k) {
            (designs, sel)
        } else {
            let d2: Vec<Design> = designs.iter().map(|d| xm::subset(d, &keep)).collect();
            (d2, sel.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| s).collect())
        };
        designs = designs2;
        let sel = sel2;
        let n = designs[0].n();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let rows = designs[0].rows.clone();
        // ordinal categories
        let mut cats: Vec<f64> = Vec::new();
        if outcome == Outcome::Oprobit {
            cats = (0..n).filter(|&t| sel[t]).map(|t| ycols[0][rows[t]]).collect();
            cats.sort_by(f64::total_cmp);
            cats.dedup();
            if cats.len() < 2 {
                return Err(rc_err(2000, "outcome does not vary"));
            }
        }
        let ncut = cats.len().saturating_sub(1);
        // components and aux layout
        let m = specs.len();
        let mut aux_names: Vec<String> = (1..=ncut).map(|i| format!("cut{i}")).collect();
        let mut lnsd = vec![None; m];
        for (j, s) in specs.iter().enumerate() {
            let free = match s.kind {
                EqKind::Outcome => matches!(outcome, Outcome::Linear | Outcome::Interval),
                EqKind::EndogCont => true,
                _ => false,
            };
            if free {
                lnsd[j] = Some(aux_names.len());
                aux_names.push(format!("lnsd:{}", s.name));
            }
        }
        let mut corr = vec![vec![usize::MAX; m]; m];
        for j in 1..m {
            for i in 0..j {
                corr[i][j] = aux_names.len();
                aux_names.push(format!("athrho:{}:{}", specs[j].name, specs[i].name));
            }
        }
        let discrete = specs
            .iter()
            .filter(|s| match s.kind {
                EqKind::Outcome => matches!(outcome, Outcome::Probit | Outcome::Oprobit | Outcome::Interval),
                EqKind::EndogCont => false,
                _ => true,
            })
            .count();
        if discrete > 2 {
            return Err(rc_err(198, "at most two binary, ordinal or censored equations may be combined"));
        }
        let mut obs = Vec::with_capacity(n);
        for t in 0..n {
            let r = rows[t];
            let mut o = Vec::with_capacity(m);
            for (j, s) in specs.iter().enumerate() {
                let ob = match s.kind {
                    EqKind::Outcome => {
                        if !sel[t] {
                            Ob::Absent
                        } else {
                            match outcome {
                                Outcome::Linear => Ob::Point(ycols[0][r]),
                                Outcome::Probit => {
                                    if ycols[0][r] != 0.0 {
                                        Ob::Range(0.0, f64::INFINITY)
                                    } else {
                                        Ob::Range(f64::NEG_INFINITY, 0.0)
                                    }
                                }
                                Outcome::Oprobit => Ob::Cat(cats.iter().position(|c| *c == ycols[0][r]).unwrap()),
                                Outcome::Interval => {
                                    let (lo, hi) = (ycols[0][r], ycols[1][r]);
                                    if !is_missing(lo) && !is_missing(hi) && lo == hi {
                                        Ob::Point(lo)
                                    } else {
                                        if !is_missing(lo) && !is_missing(hi) && lo > hi {
                                            return Err(rc_err(459, format!("{} > {} in some observations", deps[0], deps[1])));
                                        }
                                        Ob::Range(if is_missing(lo) { f64::NEG_INFINITY } else { lo }, if is_missing(hi) { f64::INFINITY } else { hi })
                                    }
                                }
                            }
                        }
                    }
                    EqKind::EndogCont => Ob::Point(designs[j].y[t]),
                    EqKind::EndogProbit | EqKind::Treat | EqKind::Select => {
                        let v = if s.kind == EqKind::Select { if sel[t] { 1.0 } else { 0.0 } } else { designs[j].y[t] };
                        if v != 0.0 {
                            Ob::Range(0.0, f64::INFINITY)
                        } else {
                            Ob::Range(f64::NEG_INFINITY, 0.0)
                        }
                    }
                };
                o.push(ob);
            }
            obs.push(o);
        }
        // starting values
        let mut b0: Vec<f64> = Vec::new();
        let mut aux0 = vec![0.0; aux_names.len()];
        for (j, s) in specs.iter().enumerate() {
            let d = &designs[j];
            let act = d.active();
            let cols: Vec<&[f64]> = act.iter().map(|&c| d.cols[c].as_slice()).collect();
            match s.kind {
                EqKind::Outcome => {
                    let keep: Vec<bool> = sel.clone();
                    let ds1 = xm::subset(d, &keep);
                    let yv: Vec<f64> = (0..n).filter(|&t| sel[t]).map(|t| match obs[t][0] {
                        Ob::Point(v) => v,
                        Ob::Range(a, b) => {
                            if a.is_finite() && b.is_finite() {
                                (a + b) / 2.0
                            } else if a.is_finite() {
                                if outcome == Outcome::Probit { 1.0 } else { a }
                            } else if outcome == Outcome::Probit {
                                0.0
                            } else {
                                b
                            }
                        }
                        Ob::Cat(k) => k as f64,
                        Ob::Absent => 0.0,
                    }).collect();
                    match outcome {
                        Outcome::Linear | Outcome::Interval => {
                            let (b, s0) = xm::ols_start(&ds1, &yv);
                            b0.extend(b);
                            if let Some(a) = lnsd[j] {
                                aux0[a] = s0.max(1e-3).ln();
                            }
                        }
                        Outcome::Probit => {
                            let c1: Vec<&[f64]> = act.iter().map(|&c| ds1.cols[c].as_slice()).collect();
                            match crate::heckman::probit_fit(&c1, &yv) {
                                Ok(r) => b0.extend(r.b),
                                Err(_) => b0.extend(vec![0.0; act.len()]),
                            }
                        }
                        Outcome::Oprobit => {
                            b0.extend(vec![0.0; act.len()]);
                            let n1 = yv.len() as f64;
                            let mut cum = 0.0;
                            for c in 0..ncut {
                                cum += yv.iter().filter(|v| **v as usize == c).count() as f64;
                                aux0[c] = dist::invnormal((cum / n1).clamp(1e-4, 1.0 - 1e-4));
                            }
                        }
                    }
                }
                EqKind::EndogCont => {
                    let (b, s0) = xm::ols_start(d, &d.y);
                    b0.extend(b);
                    if let Some(a) = lnsd[j] {
                        aux0[a] = s0.max(1e-3).ln();
                    }
                }
                _ => {
                    let yv: Vec<f64> = (0..n).map(|t| if let Ob::Range(a, _) = obs[t][j] { if a == 0.0 { 1.0 } else { 0.0 } } else { 0.0 }).collect();
                    match crate::heckman::probit_fit(&cols, &yv) {
                        Ok(r) => b0.extend(r.b),
                        Err(_) => b0.extend(vec![0.0; act.len()]),
                    }
                }
            }
        }
        b0.extend(aux0);
        let n_sel = select.as_ref().map(|_| (sel.iter().filter(|s| **s).count(), sel.iter().filter(|s| !**s).count()));
        let panel = match &pvar {
            Some(p) => {
                let c = util::col(ds, p)?;
                Some(rows.iter().map(|&r| c[r]).collect())
            }
            None => None,
        };
        Ok(Prepared {
            designs,
            specs,
            model: Model { m, lnsd, corr, ncut, obs, patterns: Vec::new(), pat_of: Vec::new() }.with_patterns(),
            aux_names,
            b0,
            outcome,
            deps,
            cats,
            n_sel,
            panel,
            treat: interact.or(treat.map(|t| t.name)),
        })
    }
}

// ---------------------------------------------------------------------------
// random-effects panel likelihood
// ---------------------------------------------------------------------------

/// Observation-level model for the random-effects panel likelihood: a
/// per-parameter-vector preparation and the log likelihood of one
/// observation given (shifted) linear indexes.
pub(crate) trait PanelObs {
    type Prep;
    fn n_obs(&self) -> usize;
    fn prep(&self, aux: &[f64]) -> Option<Self::Prep>;
    fn ll(&self, t: usize, eta: &[f64], aux: &[f64], p: &Self::Prep) -> f64;
}

impl PanelObs for Model {
    type Prep = Vec<PatPrep>;
    fn n_obs(&self) -> usize {
        self.obs.len()
    }
    fn prep(&self, aux: &[f64]) -> Option<Vec<PatPrep>> {
        Model::prep(self, aux)
    }
    fn ll(&self, t: usize, eta: &[f64], aux: &[f64], p: &Vec<PatPrep>) -> f64 {
        self.ll_prep(t, eta, aux, &p[self.pat_of[t]])
    }
}

/// Random-effects panel likelihood: a random intercept in every linear
/// index, jointly normal (unstructured), integrated by adaptive
/// Gauss-Hermite quadrature.
pub(crate) struct PanelLik<'a, M: PanelObs> {
    pub model: &'a M,
    pub x: Vec<Vec<&'a [f64]>>,
    pub naux: usize,
    pub r: usize,
    pub panels: Vec<Vec<usize>>,
    /// standard-normal Gauss-Hermite nodes and weights
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

/// Standard-normal Gauss-Hermite rule with q points.
pub(crate) fn gh_normal(q: usize) -> (Vec<f64>, Vec<f64>) {
    let (gx, gw) = crate::melogit::gauss_hermite(q.max(1));
    (gx.iter().map(|x| x * 2f64.sqrt()).collect(), gw.iter().map(|w| w / std::f64::consts::PI.sqrt()).collect())
}

impl<M: PanelObs> PanelLik<'_, M> {
    fn ncoef(&self) -> usize {
        self.x.iter().map(|c| c.len()).sum()
    }

    fn eta(&self, b: &[f64]) -> Vec<Vec<f64>> {
        let n = self.model.n_obs();
        let mut out = vec![vec![0.0; self.x.len()]; n];
        let mut p = 0;
        for (e, cols) in self.x.iter().enumerate() {
            for c in cols {
                for t in 0..n {
                    out[t][e] += c[t] * b[p];
                }
                p += 1;
            }
        }
        out
    }

    /// Random-effects covariance (lower Cholesky factor) from its parameters.
    fn re_chol(&self, b: &[f64]) -> Option<Vec<Vec<f64>>> {
        let off = self.ncoef() + self.naux;
        let r = self.r;
        let sd: Vec<f64> = (0..r).map(|i| b[off + i].exp()).collect();
        let mut s = vec![vec![0.0; r]; r];
        let mut k = off + r;
        for j in 0..r {
            s[j][j] = sd[j] * sd[j];
        }
        for j in 1..r {
            for i in 0..j {
                let rho = b[k].tanh();
                s[i][j] = rho * sd[i] * sd[j];
                s[j][i] = s[i][j];
                k += 1;
            }
        }
        chol(&s)
    }

    /// Log of the panel integrand at random effects `u`.
    fn h(&self, g: usize, u: &[f64], eta: &[Vec<f64>], aux: &[f64], preps: &M::Prep, lu: &[Vec<f64>]) -> f64 {
        let mut s = 0.0;
        let mut e = vec![0.0; self.x.len()];
        for &t in &self.panels[g] {
            for j in 0..e.len() {
                e[j] = eta[t][j] + u[j];
            }
            s += self.model.ll(t, &e, aux, preps);
        }
        // prior: N(0, L L')
        let z = {
            let r = u.len();
            let mut z = vec![0.0; r];
            for i in 0..r {
                z[i] = (u[i] - (0..i).map(|k| lu[i][k] * z[k]).sum::<f64>()) / lu[i][i];
            }
            z
        };
        let logdet: f64 = (0..u.len()).map(|i| lu[i][i].ln()).sum();
        s - 0.5 * z.iter().map(|v| v * v).sum::<f64>() - logdet - 0.5 * u.len() as f64 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Adaptive quadrature points per panel: (mode, Cholesky of the curvature inverse).
    fn adapt(&self, b: &[f64]) -> Option<Vec<(Vec<f64>, Vec<Vec<f64>>)>> {
        let lu = self.re_chol(b)?;
        let nc = self.ncoef();
        let aux = &b[nc..nc + self.naux];
        let preps = self.model.prep(aux)?;
        let eta = self.eta(b);
        let r = self.r;
        let mut out = Vec::with_capacity(self.panels.len());
        for g in 0..self.panels.len() {
            let f = |u: &[f64]| self.h(g, u, &eta, aux, &preps, &lu);
            let mut u = vec![0.0; r];
            let mut fu = f(&u);
            let hstep = 1e-4;
            let mut hess = vec![vec![0.0; r]; r];
            for _ in 0..30 {
                let (_, grad, hs) = crate::zip::num_derivs(&f, &u);
                hess = hs;
                let neg: Vec<Vec<f64>> = hess.iter().map(|row| row.iter().map(|v| -v).collect()).collect();
                let step = match chol(&neg) {
                    Some(l) => chol_solve(&l, &grad),
                    None => grad.iter().map(|g| g * 0.1).collect(),
                };
                let mut lam = 1.0;
                let mut moved = false;
                for _ in 0..20 {
                    let nu: Vec<f64> = u.iter().zip(&step).map(|(a, s)| a + lam * s).collect();
                    let fnu = f(&nu);
                    if fnu.is_finite() && fnu >= fu - 1e-12 {
                        let ch = nu.iter().zip(&u).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
                        u = nu;
                        fu = fnu;
                        moved = ch > 1e-8;
                        break;
                    }
                    lam /= 2.0;
                }
                if !moved {
                    break;
                }
            }
            let _ = hstep;
            let neg: Vec<Vec<f64>> = hess.iter().map(|row| row.iter().map(|v| -v).collect()).collect();
            let l = match chol(&neg) {
                Some(l) => {
                    // inverse of L L' -> Cholesky of the covariance
                    let mut inv = vec![vec![0.0; r]; r];
                    for c in 0..r {
                        let mut e = vec![0.0; r];
                        e[c] = 1.0;
                        let col = chol_solve(&l, &e);
                        for i in 0..r {
                            inv[i][c] = col[i];
                        }
                    }
                    chol(&inv).unwrap_or_else(|| lu.clone())
                }
                None => lu.clone(),
            };
            out.push((u, l));
        }
        Some(out)
    }

    /// Log likelihood of every panel at `b` with fixed quadrature points.
    fn panel_lls(&self, b: &[f64], ad: &[(Vec<f64>, Vec<Vec<f64>>)]) -> Option<Vec<f64>> {
        let lu = self.re_chol(b)?;
        let nc = self.ncoef();
        let aux = &b[nc..nc + self.naux];
        let preps = self.model.prep(aux)?;
        let eta = self.eta(b);
        let r = self.r;
        let q = self.nodes.len();
        let total = q.pow(r as u32);
        let mut out = Vec::with_capacity(self.panels.len());
        for (g, (mu, l)) in ad.iter().enumerate() {
            let logdet: f64 = (0..r).map(|i| l[i][i].ln()).sum();
            let mut terms = Vec::with_capacity(total);
            for idx in 0..total {
                let mut z = vec![0.0; r];
                let mut lw = 0.0;
                let mut k = idx;
                for d in 0..r {
                    let a = k % q;
                    k /= q;
                    z[d] = self.nodes[a];
                    lw += self.weights[a].ln() + 0.5 * z[d] * z[d] + 0.5 * (2.0 * std::f64::consts::PI).ln();
                }
                let u: Vec<f64> = (0..r).map(|i| mu[i] + (0..=i).map(|c| l[i][c] * z[c]).sum::<f64>()).collect();
                terms.push(lw + self.h(g, &u, &eta, aux, &preps, &lu));
            }
            let mx = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if !mx.is_finite() {
                return None;
            }
            out.push(mx + terms.iter().map(|v| (v - mx).exp()).sum::<f64>().ln() + logdet);
        }
        Some(out)
    }

    fn scores(&self, b: &[f64], ad: &[(Vec<f64>, Vec<Vec<f64>>)]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let base = self.panel_lls(b, ad)?;
        let k = b.len();
        let g = base.len();
        let mut sc = vec![vec![0.0; k]; g];
        for j in 0..k {
            let h = 1e-5 * b[j].abs().max(1.0);
            let mut bp = b.to_vec();
            let mut bm = b.to_vec();
            bp[j] += h;
            bm[j] -= h;
            let lp = self.panel_lls(&bp, ad)?;
            let lm = self.panel_lls(&bm, ad)?;
            for i in 0..g {
                sc[i][j] = (lp[i] - lm[i]) / (2.0 * h);
            }
        }
        Some((base, sc))
    }
}

impl<M: PanelObs> Likelihood for PanelLik<'_, M> {
    fn k(&self) -> usize {
        self.ncoef() + self.naux + self.r + self.r * (self.r - 1) / 2
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.k();
        let bad = || (f64::NEG_INFINITY, vec![0.0; k], Matrix::identity(k), Vec::new());
        let Some(ad) = self.adapt(b) else { return bad() };
        let Some((base, sc)) = self.scores(b, &ad) else { return bad() };
        let ll: f64 = base.iter().sum();
        let grad: Vec<f64> = (0..k).map(|j| sc.iter().map(|s| s[j]).sum()).collect();
        let mut info = Matrix::zeros(k, k);
        if need_scores {
            // numerical Hessian from differences of the gradient
            for j in 0..k {
                let h = 1e-4 * b[j].abs().max(1.0);
                let mut bp = b.to_vec();
                let mut bm = b.to_vec();
                bp[j] += h;
                bm[j] -= h;
                let gp = self.scores(&bp, &ad).map(|x| (0..k).map(|c| x.1.iter().map(|s| s[c]).sum::<f64>()).collect::<Vec<f64>>());
                let gm = self.scores(&bm, &ad).map(|x| (0..k).map(|c| x.1.iter().map(|s| s[c]).sum::<f64>()).collect::<Vec<f64>>());
                if let (Some(gp), Some(gm)) = (gp, gm) {
                    for c in 0..k {
                        info[(j, c)] = -(gp[c] - gm[c]) / (2.0 * h);
                    }
                }
            }
            for a in 0..k {
                for c in 0..a {
                    let v = 0.5 * (info[(a, c)] + info[(c, a)]);
                    info[(a, c)] = v;
                    info[(c, a)] = v;
                }
            }
        } else {
            for s in &sc {
                for a in 0..k {
                    for c in 0..k {
                        info[(a, c)] += s[a] * s[c];
                    }
                }
            }
        }
        (ll, grad, info, sc)
    }
}

// ---------------------------------------------------------------------------
// command
// ---------------------------------------------------------------------------

/// Natural-scale display of the ancillary parameters: names and the
/// Jacobian of (value) with respect to the estimated (transformed) value.
pub(crate) fn natural_aux(name: &str, v: f64) -> (String, f64, f64) {
    if let Some(n) = name.strip_prefix("lnsd:") {
        let var = (2.0 * v).exp();
        (format!("var(e.{n})"), var, 2.0 * var)
    } else if let Some(rest) = name.strip_prefix("athrho:") {
        let (a, b) = rest.split_once(':').unwrap_or((rest, ""));
        let r = v.tanh();
        (format!("corr(e.{a},e.{b})"), r, 1.0 - r * r)
    } else if let Some(rest) = name.strip_prefix("relnsd:") {
        let var = (2.0 * v).exp();
        (format!("var({rest}[{}])", "panel"), var, 2.0 * var)
    } else if let Some(rest) = name.strip_prefix("reathrho:") {
        let r = v.tanh();
        (format!("corr({rest})"), r, 1.0 - r * r)
    } else {
        (name.to_string(), v, 1.0)
    }
}

/// Convert the ancillary parameters of a fit (log standard deviations,
/// Fisher z) to variances and correlations, with the delta-method variance.
pub(crate) fn naturalize(fit: &mut xm::Fit) {
    let ncoef = fit.names.iter().filter(|nm| !nm.starts_with("/:")).count();
    let mut jac = vec![1.0; fit.names.len()];
    let mut nat_names = fit.names.clone();
    for j in ncoef..fit.names.len() {
        let raw = fit.names[j][2..].to_string();
        let (nm, val, dj) = natural_aux(&raw, fit.b[j]);
        nat_names[j] = format!("/:{nm}");
        fit.b[j] = val;
        jac[j] = dj;
    }
    let k = fit.names.len();
    for a in 0..k {
        for c in 0..k {
            fit.v[(a, c)] *= jac[a] * jac[c];
        }
    }
    let pos = fit.pos.clone();
    for (ai, &a) in pos.iter().enumerate() {
        for (ci, &c) in pos.iter().enumerate() {
            fit.vt[(ai, ci)] = fit.v[(a, c)];
        }
        fit.bt[ai] = fit.b[a];
    }
    fit.names = nat_names;
}

impl CommandHandler for Erm {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return xm_replay(st, self.verb);
        }
        let p = self.prepare(ds, st, cmd)?;
        let d0 = &p.designs[0];
        let n = d0.n();
        let level = util::level(cmd, st)?;
        let mut vce = xm::ml_vce_opt(cmd, d0)?;
        let eq_names: Vec<String> = p.specs.iter().map(|s| s.name.clone()).collect();
        let eqs: Vec<(&str, &Design)> = eq_names.iter().map(|s| s.as_str()).zip(p.designs.iter()).collect();
        let model = &p.model;
        let mut aux_names = p.aux_names.clone();
        let (fit, ng) = if !self.panel {
            let cluster = est::cluster_values(ds, d0, &vce)?;
            let nc = eqs.len();
            let f = |t: usize, v: &[f64]| model.ll(t, &v[..nc], &v[nc..]);
            let g = xm::numeric(&f);
            let auxr: Vec<&str> = aux_names.iter().map(|s| s.as_str()).collect();
            (xm::ml_fit(&eqs, &auxr, &g, p.b0.clone(), &d0.w, d0.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?, None)
        } else {
            if d0.wkind.is_some() {
                return Err(rc_err(101, "weights not allowed"));
            }
            let pid = p.panel.clone().unwrap();
            let mut keys: Vec<f64> = pid.clone();
            keys.sort_by(f64::total_cmp);
            keys.dedup();
            let mut panels = vec![Vec::new(); keys.len()];
            for (t, v) in pid.iter().enumerate() {
                panels[keys.binary_search_by(|k| k.total_cmp(v)).unwrap()].push(t);
            }
            let r = eqs.len();
            let q = util::opt_num(cmd, "intpoints", 4)?.map(|v| v as usize).unwrap_or(match r {
                1 => 12,
                2 => 7,
                _ => 5,
            });
            let (gx, gw) = crate::melogit::gauss_hermite(q.max(1));
            let nodes: Vec<f64> = gx.iter().map(|x| x * 2f64.sqrt()).collect();
            let weights: Vec<f64> = gw.iter().map(|w| w / std::f64::consts::PI.sqrt()).collect();
            let acts: Vec<Vec<usize>> = p.designs.iter().map(|d| d.active()).collect();
            let x: Vec<Vec<&[f64]>> = p.designs.iter().zip(&acts).map(|(d, a)| a.iter().map(|&j| d.cols[j].as_slice()).collect()).collect();
            let lik = PanelLik { model, x, naux: aux_names.len(), r, panels, nodes, weights };
            let mut b0 = p.b0.clone();
            for _ in 0..r {
                b0.push((0.5f64).ln());
            }
            for _ in 0..r * (r - 1) / 2 {
                b0.push(0.0);
            }
            let res = xm::maximize(&lik, b0, 200)?;
            for s in &p.specs {
                aux_names.push(format!("relnsd:{}", s.name));
            }
            for j in 1..r {
                for i in 0..j {
                    aux_names.push(format!("reathrho:{}[panel],{}[panel]", p.specs[j].name, p.specs[i].name));
                }
            }
            let ngf = lik.panels.len();
            if matches!(vce, Vce::Cluster(_)) {
                return Err(rc_err(198, "vce(cluster) not allowed; vce(robust) clusters on the panel variable"));
            }
            let fit_vce = Vce::Oim;
            let auxr: Vec<&str> = aux_names.iter().map(|s| s.as_str()).collect();
            let mut fit = xm::finish_fit(&eqs, &auxr, res, &fit_vce, None)?;
            if vce == Vce::Robust {
                let (hinv, _) = fit.res.info.sym_inverse(1e-14);
                let g = ngf as f64;
                let (vr, _) = est::sandwich(&hinv, &fit.res.scores, None, g / (g - 1.0));
                fit.vt = vr;
                fit.v = est::expand_v(&fit.vt, &fit.pos, fit.names.len());
                fit.n_clust = Some(ngf);
                vce = Vce::Cluster(crate::xtnl::panel_var(ds, st, cmd)?);
            }
            (fit, Some(ngf))
        };
        let mut fit = fit;
        naturalize(&mut fit);
        if let Some(pn) = &p.panel {
            let _ = pn;
            let pv = crate::xtnl::panel_var(ds, st, cmd)?;
            for nm in fit.names.iter_mut() {
                *nm = nm.replace("[panel]", &format!("[{pv}]"));
            }
        }
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        let title = self.title(&p);
        let mut out = est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" });
        out.push(String::new());
        let mut lines: Vec<(String, String, String)> = Vec::new();
        if let Some((s1, s0)) = p.n_sel {
            lines.push((String::new(), "Selected".into(), util::commas(s1)));
            lines.push((String::new(), "Nonselected".into(), util::commas(s0)));
        }
        if let Some(g) = ng {
            let pv = crate::xtnl::panel_var(ds, st, cmd)?;
            lines.push((format!("Group variable: {pv}"), "Number of groups".into(), util::commas(g)));
        }
        let ncl = match (&vce, fit.n_clust) {
            (Vce::Cluster(c), Some(g)) => Some((g, c.clone())),
            _ => None,
        };
        out.extend(xm::Header { title: &title, n: d0.n_eff(), lines, test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: ncl }.render());
        out.extend(xm::table("", &fit, level, None, &[]));
        let text = out.join("\n");
        let depvar = {
            let mut v = p.deps.clone();
            for s in p.specs.iter().skip(1) {
                if let Some(dv) = &s.dep {
                    v.push(dv.clone());
                }
            }
            v.join(" ")
        };
        let verb = self.verb;
        let cats = p.cats.clone();
        let n_sel = p.n_sel;
        let treat = p.treat.clone();
        let ll = fit.ll();
        xm::post_fit(st, verb, d0, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set_str("depvar", depvar.clone());
            e.set_str("title", title.clone());
            e.set_str("_output", text.clone());
            e.set("ll", ll);
            e.set("N", n as f64);
            if !cats.is_empty() {
                e.set("k_cat", cats.len() as f64);
                e.set_mat("cat", Matrix::row_vector(&cats));
            }
            if let Some((a, b)) = n_sel {
                e.set("N_selected", a as f64);
                e.set("N_nonselected", b as f64);
            }
            if let Some(g) = ng {
                e.set("N_g", g as f64);
            }
            if let Some(t) = &treat {
                e.set_str("treatvar", t.clone());
            }
            e.set_str("marginsok", "xb default");
        });
        if let Some(pv) = p.panel.as_ref().map(|_| crate::xtnl::panel_var(ds, st, cmd)) {
            st.e.set_str("ivar", pv?);
        }
        Ok(text)
    }
}

fn xm_replay(st: &Store, verb: &str) -> Result<String, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
        return Err(rc_err(301, "last estimates not found"));
    }
    crate::est::stash_table_from_e(&st.e, st.level);
    Ok(st.e.macros.get("_output").cloned().unwrap_or_default())
}

/// `predict` after the extended regression models (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let probit = matches!(ecmd.as_str(), "eprobit" | "xteprobit");
    if !matches!(ecmd.as_str(), "eregress" | "eprobit" | "eoprobit" | "eintreg" | "xteregress" | "xteprobit" | "xteoprobit" | "xteintreg" | "xtheckman") {
        return None;
    }
    let stats: &[&str] = if probit { &["pr", "xb"] } else { &["xb", "pr"] };
    Some(xm::generic_predict(ds, st, cmd, stats, if probit { "(option pr assumed; Pr(outcome))" } else { "(option xb assumed; linear prediction)" }, &|s, c| match s {
        "xb" => c.eta[0],
        "pr" if probit => dist::normal(c.eta[0]),
        "pr" => {
            let sel = c.eta.last().copied().unwrap_or(SYSMISS);
            if is_missing(sel) { SYSMISS } else { dist::normal(sel) }
        }
        _ => SYSMISS,
    }))
}

#[allow(dead_code)]
fn unused(_: MlResult) -> String {
    fmt_g(0.0, 9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_likelihood_reduces_to_marginals_without_correlation() {
        // two components: continuous (sd 2) and binary, independent
        let model = Model { m: 2, lnsd: vec![Some(0), None], corr: vec![vec![usize::MAX, 1], vec![usize::MAX; 2]], ncut: 0, obs: vec![vec![Ob::Point(1.0), Ob::Range(0.0, f64::INFINITY)]], patterns: Vec::new(), pat_of: Vec::new() }.with_patterns();
        let aux = [2f64.ln(), 0.0];
        let ll = model.ll(0, &[0.5, 0.3], &aux);
        let want = (dist::normalden(0.25) / 2.0).ln() + dist::normal(0.3).ln();
        assert!((ll - want).abs() < 1e-10, "{ll} {want}");
    }

    #[test]
    fn rectangle_probabilities() {
        assert!((rect2(f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, 0.3) - 1.0).abs() < 1e-12);
        assert!((rect2(0.0, f64::INFINITY, 0.0, f64::INFINITY, 0.0) - 0.25).abs() < 1e-7);
    }
}
