//! Bayesian model: parameters, equations, likelihoods and priors, with the
//! log-posterior evaluated in full or locally for a block of parameters.

use ferrostat_core::missing::is_missing;

use crate::dist;
use crate::matrix::Matrix;
use crate::rng::Rng;

use super::expr::{Ctx, CE};
use super::prior::{self, Support, Uni};

const LN2PI: f64 = 1.8378770664093453;

#[derive(Clone, Debug)]
pub struct Param {
    /// full name: `eq:name` or `name`
    pub name: String,
    pub support: Support,
    /// random-effects term this parameter belongs to (index into Model::re_terms)
    pub re: Option<usize>,
    /// matrix parameter this element belongs to: (matrix, row, col)
    pub mat: Option<(usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub enum Arg {
    Num(f64),
    Par(usize),
    Ex(CE),
}

impl Arg {
    pub fn val(&self, th: &[f64]) -> f64 {
        match self {
            Arg::Num(v) => *v,
            Arg::Par(j) => th[*j],
            Arg::Ex(e) => e.eval(&Ctx { cols: &[], theta: th, xb: &[], row: 0 }),
        }
    }
    pub fn params(&self, out: &mut Vec<usize>) {
        match self {
            Arg::Par(j) => out.push(*j),
            Arg::Ex(e) => e.params(out),
            Arg::Num(_) => {}
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Arg::Num(v) => Some(*v),
            Arg::Ex(e) => e.constant(),
            _ => None,
        }
    }
    pub fn label(&self, m: &Model) -> String {
        match self {
            Arg::Num(v) => crate::format::fmt_g(*v, 8).trim().to_string(),
            Arg::Par(j) => format!("{{{}}}", m.params[*j].name),
            Arg::Ex(_) => "expr".into(),
        }
    }
}

/// A random-effects term `U[id]` (optionally multiplying a variable).
#[derive(Clone, Debug)]
pub struct ReTerm {
    pub name: String,
    pub groupvar: String,
    /// parameter index of each sample row's effect
    pub gidx: Vec<usize>,
    /// slope variable values (None = intercept)
    pub z: Option<Vec<f64>>,
    /// parameters of the levels, in level order, and the levels
    pub params: Vec<usize>,
    pub levels: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Eq {
    pub name: String,
    pub depvar: String,
    pub y: Vec<f64>,
    pub xnames: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub coefs: Vec<usize>,
    pub offset: Vec<f64>,
    /// substitutable expression for the mean (nonlinear equation)
    pub nl: Option<CE>,
    pub re: Vec<usize>,
}

/// Log likelihood (or log posterior) computed by a user program:
/// (parameters, parameter names, linear forms per equation over the sample,
/// dataset rows of the sample) -> value.
pub type CbFn = dyn FnMut(&[f64], &[String], &[Vec<f64>], &[usize]) -> f64;

#[derive(Clone)]
pub struct Callback(pub std::rc::Rc<std::cell::RefCell<Box<CbFn>>>);

impl std::fmt::Debug for Callback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Callback")
    }
}

thread_local! {
    /// Set by the program interpreter before `bayesmh ..., evaluator()` runs.
    pub static CALLBACK: std::cell::RefCell<Option<Callback>> = const { std::cell::RefCell::new(None) };
}

#[derive(Clone, Debug)]
pub enum Fam {
    Normal(Arg),
    Logit,
    Probit,
    Cloglog,
    Poisson,
    /// binomial with logit link and number of trials
    BinLogit(Vec<f64>),
    /// ordered outcome: cutpoint parameters and outcome category of each row
    Ologit(Vec<usize>, Vec<usize>),
    Oprobit(Vec<usize>, Vec<usize>),
    /// multinomial logit: outcome index per row (0 = base) ; eqs are the non-base outcomes
    Mlogit(Vec<usize>),
    Llf(CE),
    DNormal(CE, Arg),
    DBernoulli(CE),
    DPoisson(CE),
    DBinomial(CE, CE),
    DExponential(CE),
    /// negative binomial (mean dispersion), parameter ln(alpha)
    NBreg(Arg),
    /// censored normal: lower, upper limits per row, standard deviation (variance) arg
    Tobit(Vec<f64>, Vec<f64>, Arg),
    /// beta regression with logit mean and log precision ln(phi)
    Beta(Arg),
    /// interval-censored normal: lower and upper bounds (missing = open), variance
    Interval(Vec<f64>, Vec<f64>, Arg),
    /// truncated normal: truncation points (empty = none), variance
    Trunc(Vec<f64>, Vec<f64>, Arg),
    /// zero-truncated (at ll) Poisson
    TPoisson(f64),
    /// zero-inflated Poisson (None) / negative binomial (ln alpha); eqs = [count, inflation]
    Zip(Option<Arg>),
    /// asymmetric Laplace (quantile regression): quantile, scale
    Ald(f64, Arg),
    /// negative binomial with ln(alpha) given by an expression of the data
    GNBreg(CE),
    /// fractional response (quasi-Bernoulli) logit / probit
    FracLogit,
    FracProbit,
    /// weibull PH (ln p), exponential: depvar is time, `fail` indicator
    Weibull(Arg, Vec<f64>, Vec<f64>),
    Exponential(Vec<f64>, Vec<f64>),
    /// generalized linear model: family, link, (binomial denominators)
    Glm(String, String, Vec<f64>),
    /// quadratic approximation around a maximum-likelihood fit
    Quad { b: Vec<f64>, vinv: Matrix, llmax: f64, idx: Vec<usize> },
    /// multivariate normal over the equations, covariance matrix parameter
    MvNormal(usize),
    /// user program; true when it returns the log posterior
    Callback(Callback, bool),
}

impl Fam {
    pub fn label(&self) -> &'static str {
        match self {
            Fam::Normal(_) | Fam::DNormal(..) => "normal",
            Fam::Logit => "logit",
            Fam::Probit => "probit",
            Fam::Cloglog => "cloglog",
            Fam::Poisson | Fam::DPoisson(_) => "poisson",
            Fam::BinLogit(_) => "binlogit",
            Fam::Ologit(..) => "ologit",
            Fam::Oprobit(..) => "oprobit",
            Fam::Mlogit(_) => "mlogit",
            Fam::Llf(_) => "llf",
            Fam::DBernoulli(_) => "bernoulli",
            Fam::DBinomial(..) => "binomial",
            Fam::DExponential(_) => "exponential",
            Fam::NBreg(_) => "nbreg",
            Fam::Tobit(..) => "tobit",
            Fam::Beta(_) => "beta",
            Fam::Ald(..) => "asymmetric Laplace",
            Fam::Interval(..) => "interval",
            Fam::Trunc(..) => "truncated normal",
            Fam::TPoisson(_) => "truncated Poisson",
            Fam::Zip(a) => if a.is_some() { "zinb" } else { "zip" },
            Fam::GNBreg(_) => "gnbreg",
            Fam::FracLogit => "fraclogit",
            Fam::FracProbit => "fracprobit",
            Fam::Weibull(..) => "weibull",
            Fam::Exponential(..) => "exponential",
            Fam::Glm(..) => "glm",
            Fam::Quad { .. } => "approximate",
            Fam::MvNormal(_) => "mvnormal",
            Fam::Callback(_, post) => if *post { "evaluator" } else { "llevaluator" },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Lik {
    pub eqs: Vec<usize>,
    pub fam: Fam,
}

#[derive(Clone, Debug)]
pub enum Cov {
    Fixed(Matrix),
    Mat(usize),
    ScaledI(Arg),
    /// fixed matrix times a scalar (Zellner's g-prior: g (X'X)^-1 times a variance)
    Scaled(Matrix, Arg),
}

#[derive(Clone, Debug)]
pub enum PK {
    Uni(Uni, Vec<Arg>),
    MvNormal(Vec<Arg>, Cov),
    Wishart(f64, Matrix, usize),
    IWishart(f64, Matrix, usize),
    Dirichlet(Vec<Arg>),
    Density(CE),
    LogDensity(CE),
}

#[derive(Clone, Debug)]
pub struct Prior {
    pub params: Vec<usize>,
    pub kind: PK,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct MatParam {
    pub name: String,
    pub d: usize,
    /// element parameters, row-major lower triangle (i >= j)
    pub idx: Vec<usize>,
}

impl MatParam {
    pub fn matrix(&self, th: &[f64]) -> Matrix {
        let mut m = Matrix::zeros(self.d, self.d);
        let mut k = 0;
        for i in 0..self.d {
            for j in 0..=i {
                m[(i, j)] = th[self.idx[k]];
                m[(j, i)] = th[self.idx[k]];
                k += 1;
            }
        }
        m
    }
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub params: Vec<Param>,
    pub eqs: Vec<Eq>,
    pub liks: Vec<Lik>,
    pub priors: Vec<Prior>,
    pub mats: Vec<MatParam>,
    pub re_terms: Vec<ReTerm>,
    /// data columns over the sample rows (read by expressions)
    pub cols: Vec<Vec<f64>>,
    pub colnames: Vec<String>,
    /// number of sample rows
    pub n: usize,
    /// sample marker over the dataset
    pub mask: Vec<bool>,
    /// dependency lists, filled by `finalize`
    pub prior_deps: Vec<Vec<usize>>,
    pub lik_deps: Vec<Vec<usize>>,
    /// rows of each RE parameter (sample positions)
    pub re_rows: Vec<Vec<usize>>,
}

impl Model {
    pub fn find(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    pub fn add_param(&mut self, name: &str) -> usize {
        if let Some(j) = self.find(name) {
            return j;
        }
        self.params.push(Param { name: name.to_string(), support: Support::Real, re: None, mat: None });
        self.params.len() - 1
    }

    pub fn col_index(&mut self, name: &str, vals: Vec<f64>) -> usize {
        if let Some(j) = self.colnames.iter().position(|c| c == name) {
            return j;
        }
        self.colnames.push(name.to_string());
        self.cols.push(vals);
        self.cols.len() - 1
    }

    /// Compute dependency lists and per-parameter row sets.
    pub fn finalize(&mut self) {
        self.prior_deps = self.priors.iter().map(|p| {
            let mut d = p.params.clone();
            match &p.kind {
                PK::Uni(_, a) | PK::Dirichlet(a) => a.iter().for_each(|x| x.params(&mut d)),
                PK::MvNormal(m, c) => {
                    m.iter().for_each(|x| x.params(&mut d));
                    match c {
                        Cov::Mat(k) => d.extend(self.mats[*k].idx.iter().copied()),
                        Cov::ScaledI(a) | Cov::Scaled(_, a) => a.params(&mut d),
                        Cov::Fixed(_) => {}
                    }
                }
                PK::Wishart(_, _, k) | PK::IWishart(_, _, k) => d.extend(self.mats[*k].idx.iter().copied()),
                PK::Density(e) | PK::LogDensity(e) => e.params(&mut d),
            }
            d.sort();
            d.dedup();
            d
        }).collect();
        let lik_deps: Vec<Vec<usize>> = self.liks.iter().map(|l| {
            let mut d = Vec::new();
            let mut eqs = l.eqs.clone();
            let mut more = Vec::new();
            match &l.fam {
                Fam::Llf(e) | Fam::DBernoulli(e) | Fam::DPoisson(e) | Fam::DExponential(e) => {
                    e.params(&mut d);
                    e.xbs(&mut more);
                }
                Fam::DNormal(e, a) => {
                    e.params(&mut d);
                    e.xbs(&mut more);
                    a.params(&mut d);
                }
                Fam::DBinomial(a, b) => {
                    a.params(&mut d);
                    b.params(&mut d);
                    a.xbs(&mut more);
                }
                Fam::Normal(a) | Fam::NBreg(a) | Fam::Tobit(_, _, a) | Fam::Beta(a) | Fam::Weibull(a, _, _) | Fam::Ald(_, a) => a.params(&mut d),
                Fam::GNBreg(e) => e.params(&mut d),
                Fam::Interval(_, _, a) | Fam::Trunc(_, _, a) | Fam::Zip(Some(a)) => a.params(&mut d),
                Fam::Ologit(c, _) | Fam::Oprobit(c, _) => d.extend(c.iter().copied()),
                Fam::Quad { idx, .. } => d.extend(idx.iter().copied()),
                Fam::MvNormal(k) => d.extend(self.mats[*k].idx.iter().copied()),
                Fam::Callback(..) => d.extend(0..self.params.len()),
                _ => {}
            }
            eqs.extend(more);
            for &k in &eqs {
                let e = &self.eqs[k];
                d.extend(e.coefs.iter().copied());
                if let Some(nl) = &e.nl {
                    nl.params(&mut d);
                }
                for &t in &e.re {
                    d.extend(self.re_terms[t].params.iter().copied());
                }
            }
            d.sort();
            d.dedup();
            d
        }).collect();
        self.lik_deps = lik_deps;
        self.re_rows = vec![Vec::new(); self.params.len()];
        for t in &self.re_terms {
            for (r, &j) in t.gidx.iter().enumerate() {
                self.re_rows[j].push(r);
            }
        }
    }

    /// Equations whose linear forms a likelihood reads.
    fn lik_eqs(&self, l: &Lik) -> Vec<usize> {
        let mut eqs = l.eqs.clone();
        match &l.fam {
            Fam::Llf(e) | Fam::DBernoulli(e) | Fam::DPoisson(e) | Fam::DExponential(e) | Fam::DNormal(e, _) | Fam::DBinomial(e, _) => e.xbs(&mut eqs),
            _ => {}
        }
        eqs.sort();
        eqs.dedup();
        eqs
    }

    /// Linear form (or substitutable expression) of equation k at sample row r.
    pub fn xb(&self, k: usize, th: &[f64], r: usize, xbuf: &[f64]) -> f64 {
        let e = &self.eqs[k];
        let mut s = if let Some(nl) = &e.nl {
            nl.eval(&Ctx { cols: &self.cols, theta: th, xb: xbuf, row: r })
        } else {
            let mut s = 0.0;
            for (j, &c) in e.coefs.iter().enumerate() {
                s += e.x[j][r] * th[c];
            }
            s
        };
        if !e.offset.is_empty() {
            s += e.offset[r];
        }
        for &t in &e.re {
            let rt = &self.re_terms[t];
            let u = th[rt.gidx[r]];
            s += match &rt.z {
                Some(z) => u * z[r],
                None => u,
            };
        }
        s
    }

    /// Log likelihood of likelihood `li` over `rows` (None = all rows).
    pub fn lik_ll(&self, li: usize, th: &[f64], rows: Option<&[usize]>) -> f64 {
        let l = &self.liks[li];
        if let Fam::Quad { b, vinv, llmax, idx } = &l.fam {
            let d: Vec<f64> = idx.iter().zip(b).map(|(&j, bb)| th[j] - bb).collect();
            let mut q = 0.0;
            for i in 0..d.len() {
                for j in 0..d.len() {
                    q += d[i] * vinv[(i, j)] * d[j];
                }
            }
            return llmax - 0.5 * q;
        }
        if let Fam::Callback(cb, _) = &l.fam {
            let mut xbuf = vec![0.0; self.eqs.len()];
            let mut xbs = vec![vec![0.0; self.n]; self.eqs.len()];
            for r in 0..self.n {
                for k in 0..self.eqs.len() {
                    xbuf[k] = self.xb(k, th, r, &xbuf);
                    xbs[k][r] = xbuf[k];
                }
            }
            let rows: Vec<usize> = (0..self.mask.len()).filter(|&i| self.mask[i]).collect();
            let names: Vec<String> = self.params.iter().map(|p| p.name.clone()).collect();
            let v = (cb.0.borrow_mut())(th, &names, &xbs, &rows);
            return if v.is_finite() { v } else { f64::NEG_INFINITY };
        }
        let eqs = self.lik_eqs(l);
        let mut xbuf = vec![0.0; self.eqs.len()];
        let mut total = 0.0;
        let all: Vec<usize>;
        let rows = match rows {
            Some(r) => r,
            None => {
                all = (0..self.n).collect();
                &all
            }
        };
        // per-likelihood constants
        let pre: f64 = match &l.fam {
            Fam::Normal(a) => a.val(th),
            Fam::NBreg(a) | Fam::Tobit(_, _, a) | Fam::Beta(a) | Fam::Weibull(a, _, _) | Fam::Ald(_, a) | Fam::Interval(_, _, a) | Fam::Trunc(_, _, a) | Fam::Zip(Some(a)) => a.val(th),
            _ => 0.0,
        };
        if matches!(l.fam, Fam::Normal(_)) && pre <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let mvn = if let Fam::MvNormal(k) = &l.fam {
            let s = self.mats[*k].matrix(th);
            match s.cholesky() {
                Ok(c) => Some(c),
                Err(_) => return f64::NEG_INFINITY,
            }
        } else {
            None
        };
        for &r in rows {
            for &k in &eqs {
                // nonlinear equations may read earlier linear forms
                xbuf[k] = self.xb(k, th, r, &xbuf);
            }
            let v = self.row_ll(l, th, r, &xbuf, pre, mvn.as_ref());
            if !v.is_finite() {
                return f64::NEG_INFINITY;
            }
            total += v;
        }
        total
    }

    fn row_ll(&self, l: &Lik, th: &[f64], r: usize, xb: &[f64], pre: f64, mvn: Option<&Matrix>) -> f64 {
        let e0 = &self.eqs[l.eqs.first().copied().unwrap_or(0)];
        let y = e0.y.get(r).copied().unwrap_or(0.0);
        let eta = l.eqs.first().map(|&k| xb[k]).unwrap_or(0.0);
        let ctx = || Ctx { cols: &self.cols, theta: th, xb, row: r };
        match &l.fam {
            Fam::Normal(_) => prior::ln_normal(y, eta, pre),
            Fam::Logit => {
                if y != 0.0 { -ln1pexp(-eta) } else { -ln1pexp(eta) }
            }
            Fam::Probit => {
                if y != 0.0 { dist::lnnormal(eta) } else { dist::lnnormal(-eta) }
            }
            Fam::Cloglog => {
                let p0 = -eta.exp(); // ln(1-p)
                if y != 0.0 { (-p0.exp()).ln_1p() } else { p0 }
            }
            Fam::Poisson => y * eta - eta.exp() - dist::lnfactorial(y),
            Fam::BinLogit(nt) => {
                let n = nt[r];
                dist::lnfactorial(n) - dist::lnfactorial(y) - dist::lnfactorial(n - y) - y * ln1pexp(-eta) - (n - y) * ln1pexp(eta)
            }
            Fam::Ologit(cuts, cat) | Fam::Oprobit(cuts, cat) => {
                let c = cat[r];
                let k = cuts.len();
                let f = |z: f64| if matches!(l.fam, Fam::Ologit(..)) { dist::invlogit(z) } else { dist::normal(z) };
                for j in 1..k {
                    if th[cuts[j]] <= th[cuts[j - 1]] {
                        return f64::NEG_INFINITY;
                    }
                }
                let hi = if c < k { f(th[cuts[c]] - eta) } else { 1.0 };
                let lo = if c > 0 { f(th[cuts[c - 1]] - eta) } else { 0.0 };
                (hi - lo).ln()
            }
            Fam::Mlogit(cat) => {
                let mut m = 0.0f64;
                for &k in &l.eqs {
                    m = m.max(xb[k]);
                }
                let mut den = (-m).exp();
                for &k in &l.eqs {
                    den += (xb[k] - m).exp();
                }
                let c = cat[r];
                let num = if c == 0 { -m } else { xb[l.eqs[c - 1]] - m };
                num - den.ln()
            }
            Fam::Llf(e) => e.eval(&ctx()),
            Fam::DNormal(m, v) => prior::ln_normal(y, m.eval(&ctx()), v.val(th)),
            Fam::DBernoulli(p) => {
                let p = p.eval(&ctx());
                if !(0.0..=1.0).contains(&p) { f64::NEG_INFINITY } else if y != 0.0 { p.ln() } else { (1.0 - p).ln() }
            }
            Fam::DPoisson(mu) => {
                let mu = mu.eval(&ctx());
                if mu <= 0.0 { f64::NEG_INFINITY } else { y * mu.ln() - mu - dist::lnfactorial(y) }
            }
            Fam::DBinomial(p, n) => {
                let (p, n) = (p.eval(&ctx()), n.eval(&ctx()));
                if !(0.0..=1.0).contains(&p) {
                    return f64::NEG_INFINITY;
                }
                dist::lnfactorial(n) - dist::lnfactorial(y) - dist::lnfactorial(n - y) + y * p.ln() + (n - y) * (1.0 - p).ln()
            }
            Fam::DExponential(m) => {
                let m = m.eval(&ctx());
                if m <= 0.0 || y < 0.0 { f64::NEG_INFINITY } else { -m.ln() - y / m }
            }
            Fam::NBreg(_) => {
                let a = pre.exp();
                let mu = eta.exp();
                let m = 1.0 / a;
                nb_ll(y, m, mu)
            }
            Fam::Tobit(lo, hi, _) => {
                let s = pre.sqrt();
                if !(pre > 0.0) {
                    return f64::NEG_INFINITY;
                }
                if !lo.is_empty() && y <= lo[r] {
                    dist::lnnormal((lo[r] - eta) / s)
                } else if !hi.is_empty() && y >= hi[r] {
                    dist::lnnormal((eta - hi[r]) / s)
                } else {
                    prior::ln_normal(y, eta, pre)
                }
            }
            Fam::Interval(lo, hi, _) => {
                if !(pre > 0.0) {
                    return f64::NEG_INFINITY;
                }
                let s = pre.sqrt();
                let (a, b) = (lo[r], hi[r]);
                match (is_missing(a), is_missing(b)) {
                    (true, true) => 0.0,
                    (true, false) => dist::lnnormal((b - eta) / s),
                    (false, true) => dist::lnnormal((eta - a) / s),
                    (false, false) if a == b => prior::ln_normal(a, eta, pre),
                    _ => (dist::normal((b - eta) / s) - dist::normal((a - eta) / s)).max(1e-300).ln(),
                }
            }
            Fam::Trunc(ll, ul, _) => {
                if !(pre > 0.0) {
                    return f64::NEG_INFINITY;
                }
                let s = pre.sqrt();
                let up = if ul.is_empty() { 1.0 } else { dist::normal((ul[r] - eta) / s) };
                let lo = if ll.is_empty() { 0.0 } else { dist::normal((ll[r] - eta) / s) };
                prior::ln_normal(y, eta, pre) - (up - lo).max(1e-300).ln()
            }
            Fam::TPoisson(ll) => {
                let mu = eta.exp();
                let p0 = dist::poisson(mu, *ll);
                y * eta - mu - dist::lnfactorial(y) - (1.0 - p0).max(1e-300).ln()
            }
            Fam::Zip(lna) => {
                let pi = dist::invlogit(xb[l.eqs[1]]);
                let mu = eta.exp();
                let lc0 = match lna {
                    None => -mu,
                    Some(_) => {
                        let m = 1.0 / pre.exp();
                        -m * (mu / m).ln_1p()
                    }
                };
                if y == 0.0 {
                    (pi + (1.0 - pi) * lc0.exp()).max(1e-300).ln()
                } else {
                    let lc = match lna {
                        None => y * eta - mu - dist::lnfactorial(y),
                        Some(_) => nb_ll(y, 1.0 / pre.exp(), mu),
                    };
                    (1.0 - pi).ln() + lc
                }
            }
            Fam::Ald(q, _) => {
                if !(pre > 0.0) {
                    return f64::NEG_INFINITY;
                }
                let u = (y - eta) / pre;
                (q * (1.0 - q) / pre).ln() - u * (q - if u < 0.0 { 1.0 } else { 0.0 })
            }
            Fam::GNBreg(e) => {
                let a = e.eval(&ctx()).exp();
                let mu = eta.exp();
                let m = 1.0 / a;
                nb_ll(y, m, mu)
            }
            Fam::Beta(_) => {
                let phi = pre.exp();
                let mu = dist::invlogit(eta);
                let (a, b) = (mu * phi, (1.0 - mu) * phi);
                prior::ln_beta(y, a, b)
            }
            Fam::FracLogit => {
                let p = dist::invlogit(eta).clamp(1e-300, 1.0 - 1e-16);
                y * p.ln() + (1.0 - y) * (1.0 - p).ln()
            }
            Fam::FracProbit => y * dist::lnnormal(eta) + (1.0 - y) * dist::lnnormal(-eta),
            Fam::Weibull(_, fail, t0) => {
                let p = pre.exp();
                let lam = eta.exp();
                let t = y;
                let h0 = if t0.is_empty() { 0.0 } else { lam * t0[r].powf(p) };
                fail[r] * (eta + p.ln() + (p - 1.0) * t.ln()) - lam * t.powf(p) + h0
            }
            Fam::Exponential(fail, t0) => {
                let lam = eta.exp();
                let h0 = if t0.is_empty() { 0.0 } else { lam * t0[r] };
                fail[r] * eta - lam * y + h0
            }
            Fam::Glm(fam, link, nt) => glm_ll(fam, link, y, eta, nt.get(r).copied().unwrap_or(1.0)),
            Fam::Quad { .. } | Fam::Callback(..) => 0.0,
            Fam::MvNormal(_) => {
                let c = mvn.unwrap();
                let d = l.eqs.len();
                let mut z = vec![0.0; d];
                let mut ld = 0.0;
                for i in 0..d {
                    let ei = &self.eqs[l.eqs[i]];
                    let mut v = ei.y[r] - xb[l.eqs[i]];
                    for j in 0..i {
                        v -= c[(i, j)] * z[j];
                    }
                    z[i] = v / c[(i, i)];
                    ld += c[(i, i)].ln();
                }
                -0.5 * d as f64 * LN2PI - ld - 0.5 * z.iter().map(|v| v * v).sum::<f64>()
            }
        }
    }

    /// Log density of prior `pi`.
    pub fn prior_lp(&self, pi: usize, th: &[f64]) -> f64 {
        let p = &self.priors[pi];
        match &p.kind {
            PK::Uni(u, args) => {
                let a: Vec<f64> = args.iter().map(|x| x.val(th)).collect();
                p.params.iter().map(|&j| u.ln_pdf(th[j], &a)).sum()
            }
            PK::MvNormal(m, cov) => {
                let x: Vec<f64> = p.params.iter().map(|&j| th[j]).collect();
                let d = x.len();
                let mean: Vec<f64> = if m.len() == d { m.iter().map(|a| a.val(th)).collect() } else if m.len() == 1 { vec![m[0].val(th); d] } else { vec![0.0; d] };
                let s = match cov {
                    Cov::Fixed(s) => s.clone(),
                    Cov::Mat(k) => self.mats[*k].matrix(th),
                    Cov::ScaledI(a) => Matrix::identity(d).scale(a.val(th)),
                    Cov::Scaled(s, a) => s.scale(a.val(th)),
                };
                if d % s.rows == 0 && d > s.rows {
                    // several vectors sharing a covariance (e.g. random effects)
                    let q = s.rows;
                    (0..d / q).map(|g| prior::ln_mvnormal(&x[g * q..(g + 1) * q], &mean[g * q..(g + 1) * q], &s)).sum()
                } else {
                    prior::ln_mvnormal(&x, &mean, &s)
                }
            }
            PK::Wishart(df, v, k) => prior::ln_wishart(&self.mats[*k].matrix(th), *df, v),
            PK::IWishart(df, v, k) => prior::ln_iwishart(&self.mats[*k].matrix(th), *df, v),
            PK::Dirichlet(a) => {
                let x: Vec<f64> = p.params.iter().map(|&j| th[j]).collect();
                let a: Vec<f64> = a.iter().map(|v| v.val(th)).collect();
                prior::ln_dirichlet(&x, &a)
            }
            PK::Density(e) => {
                let v = e.eval(&Ctx { cols: &[], theta: th, xb: &[], row: 0 });
                if v > 0.0 { v.ln() } else { f64::NEG_INFINITY }
            }
            PK::LogDensity(e) => e.eval(&Ctx { cols: &[], theta: th, xb: &[], row: 0 }),
        }
    }

    pub fn log_prior(&self, th: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..self.priors.len() {
            let v = self.prior_lp(i, th);
            if !v.is_finite() {
                return f64::NEG_INFINITY;
            }
            s += v;
        }
        s
    }

    pub fn loglik(&self, th: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..self.liks.len() {
            let v = self.lik_ll(i, th, None);
            if !v.is_finite() {
                return f64::NEG_INFINITY;
            }
            s += v;
        }
        s
    }

    pub fn log_post(&self, th: &[f64]) -> f64 {
        let p = self.log_prior(th);
        if !p.is_finite() {
            return p;
        }
        p + self.loglik(th)
    }

    /// The part of the log posterior that depends on the parameters in `block`.
    pub fn local_post(&self, block: &[usize], th: &[f64]) -> f64 {
        let touches = |d: &[usize]| block.iter().any(|j| d.binary_search(j).is_ok());
        let mut s = 0.0;
        for (i, d) in self.prior_deps.iter().enumerate() {
            if touches(d) {
                let v = self.prior_lp(i, th);
                if !v.is_finite() {
                    return f64::NEG_INFINITY;
                }
                s += v;
            }
        }
        // rows: random-effect blocks of a single term only touch their group's rows
        let rows: Option<Vec<usize>> = if block.iter().all(|&j| self.params[j].re.is_some() && !self.re_rows[j].is_empty()) {
            let mut r: Vec<usize> = block.iter().flat_map(|&j| self.re_rows[j].iter().copied()).collect();
            r.sort();
            r.dedup();
            Some(r)
        } else {
            None
        };
        for (i, d) in self.lik_deps.iter().enumerate() {
            if touches(d) {
                let v = self.lik_ll(i, th, rows.as_deref());
                if !v.is_finite() {
                    return f64::NEG_INFINITY;
                }
                s += v;
            }
        }
        s
    }

    /// Expected outcome of likelihood `li` at sample row r.
    pub fn mu(&self, li: usize, th: &[f64], r: usize) -> Option<f64> {
        let l = &self.liks[li];
        let eqs = self.lik_eqs(l);
        let mut xbuf = vec![0.0; self.eqs.len()];
        for &k in &eqs {
            xbuf[k] = self.xb(k, th, r, &xbuf);
        }
        let eta = l.eqs.first().map(|&k| xbuf[k]).unwrap_or(0.0);
        let ctx = Ctx { cols: &self.cols, theta: th, xb: &xbuf, row: r };
        Some(match &l.fam {
            Fam::Normal(_) | Fam::Tobit(..) => eta,
            Fam::DNormal(m, _) => m.eval(&ctx),
            Fam::Logit | Fam::FracLogit | Fam::Beta(_) => dist::invlogit(eta),
            Fam::Probit | Fam::FracProbit => dist::normal(eta),
            Fam::Cloglog => 1.0 - (-eta.exp()).exp(),
            Fam::Poisson | Fam::NBreg(_) | Fam::GNBreg(_) => eta.exp(),
            Fam::BinLogit(nt) => nt[r] * dist::invlogit(eta),
            Fam::DPoisson(m) | Fam::DBernoulli(m) | Fam::DExponential(m) => m.eval(&ctx),
            Fam::DBinomial(p, n) => p.eval(&ctx) * n.eval(&ctx),
            _ => eta,
        })
    }

    /// Simulate the outcome of likelihood `li` at sample row r (posterior predictive).
    pub fn simulate(&self, li: usize, th: &[f64], r: usize, rng: &mut Rng) -> Option<f64> {
        let l = &self.liks[li];
        let eqs = self.lik_eqs(l);
        let mut xbuf = vec![0.0; self.eqs.len()];
        for &k in &eqs {
            xbuf[k] = self.xb(k, th, r, &xbuf);
        }
        let eta = l.eqs.first().map(|&k| xbuf[k]).unwrap_or(0.0);
        let ctx = Ctx { cols: &self.cols, theta: th, xb: &xbuf, row: r };
        Some(match &l.fam {
            Fam::Normal(a) => eta + a.val(th).max(0.0).sqrt() * rng.normal(),
            Fam::DNormal(m, a) => m.eval(&ctx) + a.val(th).max(0.0).sqrt() * rng.normal(),
            Fam::Logit | Fam::FracLogit => (rng.uniform() < dist::invlogit(eta)) as i32 as f64,
            Fam::Probit | Fam::FracProbit => (rng.uniform() < dist::normal(eta)) as i32 as f64,
            Fam::Cloglog => (rng.uniform() < 1.0 - (-eta.exp()).exp()) as i32 as f64,
            Fam::Poisson => rng.poisson(eta.exp()),
            Fam::DPoisson(m) => rng.poisson(m.eval(&ctx)),
            Fam::DBernoulli(p) => (rng.uniform() < p.eval(&ctx)) as i32 as f64,
            Fam::DExponential(m) => rng.exponential(m.eval(&ctx)),
            Fam::DBinomial(p, n) => rng.binomial(n.eval(&ctx), p.eval(&ctx)),
            Fam::BinLogit(nt) => rng.binomial(nt[r], dist::invlogit(eta)),
            Fam::NBreg(a) => {
                let al = a.val(th).exp();
                let g = rng.gamma(1.0 / al) * al;
                rng.poisson(eta.exp() * g)
            }
            Fam::Ologit(cuts, _) | Fam::Oprobit(cuts, _) => {
                let u = rng.uniform();
                let f = |z: f64| if matches!(l.fam, Fam::Ologit(..)) { dist::invlogit(z) } else { dist::normal(z) };
                let mut c = cuts.len();
                for (j, &cj) in cuts.iter().enumerate() {
                    if u < f(th[cj] - eta) {
                        c = j;
                        break;
                    }
                }
                (c + 1) as f64
            }
            Fam::Mlogit(_) => {
                let mut w = vec![1.0];
                for &k in &l.eqs {
                    w.push(xbuf[k].exp());
                }
                let s: f64 = w.iter().sum();
                let mut u = rng.uniform() * s;
                let mut c = 0;
                for (j, v) in w.iter().enumerate() {
                    if u < *v {
                        c = j;
                        break;
                    }
                    u -= v;
                    c = j;
                }
                c as f64
            }
            Fam::Tobit(lo, hi, a) => {
                let mut v = eta + a.val(th).max(0.0).sqrt() * rng.normal();
                if !lo.is_empty() && v < lo[r] {
                    v = lo[r];
                }
                if !hi.is_empty() && v > hi[r] {
                    v = hi[r];
                }
                v
            }
            Fam::Beta(a) => {
                let phi = a.val(th).exp();
                let mu = dist::invlogit(eta);
                rng.beta(mu * phi, (1.0 - mu) * phi)
            }
            Fam::Glm(..) | Fam::Ald(..) | Fam::Trunc(..) | Fam::Interval(..) => eta,
            Fam::TPoisson(ll) => {
                let mu = eta.exp();
                let mut v = rng.poisson(mu);
                let mut tries = 0;
                while v <= *ll && tries < 1000 {
                    v = rng.poisson(mu);
                    tries += 1;
                }
                v
            }
            Fam::Zip(lna) => {
                let pi = dist::invlogit(xbuf[l.eqs[1]]);
                if rng.uniform() < pi {
                    0.0
                } else {
                    let mu = eta.exp();
                    match lna {
                        None => rng.poisson(mu),
                        Some(a) => {
                            let al = a.val(th).exp();
                            { let g = rng.gamma(1.0 / al) * al; rng.poisson(mu * g) }
                        }
                    }
                }
            }
            Fam::GNBreg(e) => {
                let al = e.eval(&ctx).exp();
                let g = rng.gamma(1.0 / al) * al;
                rng.poisson(eta.exp() * g)
            }
            Fam::Weibull(..) | Fam::Exponential(..) => eta.exp(),
            _ => return None,
        })
    }
}

/// Negative binomial log probability with size m and mean mu, stable for large m.
fn nb_ll(y: f64, m: f64, mu: f64) -> f64 {
    let lg = if y >= 0.0 && y < 1000.0 && y.fract() == 0.0 {
        (0..y as usize).map(|k| (m + k as f64).ln()).sum::<f64>()
    } else {
        dist::lngamma(y + m) - dist::lngamma(m)
    };
    // m ln(m/(m+mu)) = -m ln(1 + mu/m)
    lg - dist::lnfactorial(y) - m * (mu / m).ln_1p() + y * (mu / (m + mu)).ln()
}

pub fn ln1pexp(x: f64) -> f64 {
    if x > 35.0 { x } else if x < -35.0 { x.exp() } else { x.exp().ln_1p() }
}

fn glm_ll(fam: &str, link: &str, y: f64, eta: f64, n: f64) -> f64 {
    let mu = match link {
        "log" => eta.exp(),
        "logit" => dist::invlogit(eta),
        "probit" => dist::normal(eta),
        "cloglog" => 1.0 - (-eta.exp()).exp(),
        "inverse" | "power -1" => 1.0 / eta,
        _ => eta,
    };
    match fam {
        "binomial" => {
            if !(0.0..=1.0).contains(&mu) {
                return f64::NEG_INFINITY;
            }
            let p = mu.clamp(1e-300, 1.0 - 1e-16);
            dist::lnfactorial(n) - dist::lnfactorial(y) - dist::lnfactorial(n - y) + y * p.ln() + (n - y) * (1.0 - p).ln()
        }
        "poisson" => {
            if mu <= 0.0 { f64::NEG_INFINITY } else { y * mu.ln() - mu - dist::lnfactorial(y) }
        }
        "gamma" => {
            if mu <= 0.0 { f64::NEG_INFINITY } else { -y / mu - mu.ln() }
        }
        _ => -0.5 * (y - mu) * (y - mu),
    }
}
