//! `bayesmh`: specification parsing and model construction.
//!
//! Syntax handled:
//!
//! ```text
//! bayesmh depvar [indepvars] [U[id] x#V[id] ...] [if] [in], likelihood(spec) prior(...) ...
//! bayesmh depvar = (expression) [if] [in], likelihood(spec) prior(...) ...
//! bayesmh (eq1) (eq2) ... [if] [in], [likelihood(spec)] prior(...) ...
//!     eq: ([name:] depvar [indepvars] [, likelihood(spec) noconstant])
//! bayesmh, likelihood(llf(expression)) prior(...)        (no outcome)
//! ```
//!
//! Likelihoods: normal(var), logit, probit, cloglog, poisson, binlogit(n),
//! ologit, oprobit, mlogit, llf(expr), dnormal(mean, var), dbernoulli(p),
//! dpoisson(mu), dbinomial(p, n), dexponential(mean), mvnormal({S,m}).
//! Priors: normal t lognormal uniform gamma igamma exponential beta chi2
//! cauchy laplace pareto jeffreys flat bernoulli geometric poisson index
//! mvnormal mvnormal0 zellnersg zellnersg0 wishart iwishart dirichlet
//! density logdensity.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;

use crate::dist;
use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;

use super::expr::{self, Resolver, CE};
use super::model::{Arg, Cov, Eq, Fam, Lik, MatParam, Model, Param, Prior, ReTerm, PK};
use super::prior::{Support, Uni};
use super::text;

/// Equation as written.
#[derive(Clone, Debug, Default)]
pub struct EqSpec {
    pub name: Option<String>,
    pub dep: String,
    pub indep: String,
    /// (term name, group variable, slope variable)
    pub re: Vec<(String, String, Option<String>)>,
    pub nl: Option<String>,
    pub lik: Option<String>,
    pub noconst: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub eqs: Vec<EqSpec>,
    pub lik: Option<String>,
    pub priors: Vec<String>,
    /// default priors (prefix): apply only to parameters without a user prior
    pub dpriors: Vec<String>,
    pub blocks: Vec<String>,
    pub initial: Vec<String>,
    pub ifin: String,
    pub opts: Vec<(String, Option<String>)>,
}

impl Spec {
    pub fn opt(&self, full: &str, min: usize) -> Option<&Option<String>> {
        self.opts.iter().rev().find(|(n, _)| text::abbrev(n, full, min)).map(|x| &x.1)
    }
    pub fn has(&self, full: &str, min: usize) -> bool {
        self.opt(full, min).is_some()
    }
    pub fn num(&self, full: &str, min: usize) -> Result<Option<f64>, String> {
        match self.opt(full, min) {
            Some(Some(s)) => s.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("option {full}() invalid"))),
            Some(None) => Err(rc_err(198, format!("option {full}() requires an argument"))),
            None => Ok(None),
        }
    }
}

fn is_re_token(w: &str) -> bool {
    w.ends_with(']') && w.contains('[') && !w.starts_with('[')
}

fn parse_re(w: &str) -> (String, String, Option<String>) {
    let (slope, term) = match w.split_once('#') {
        Some((a, b)) if b.contains('[') => (Some(a.trim_start_matches("c.").to_string()), b),
        Some((a, b)) => (Some(b.trim_start_matches("c.").to_string()), a),
        None => (None, w),
    };
    let i = term.find('[').unwrap();
    (term[..i].to_string(), term[i + 1..term.len() - 1].to_string(), slope)
}

fn parse_eq(body: &str) -> Result<EqSpec, String> {
    let mut e = EqSpec::default();
    let (main, opts) = match text::find_top(body, ',') {
        Some(i) => (&body[..i], &body[i + 1..]),
        None => (body, ""),
    };
    for (n, a) in text::options(opts) {
        if text::abbrev(&n, "likelihood", 2) {
            e.lik = a;
        } else if text::abbrev(&n, "noconstant", 4) {
            e.noconst = true;
        }
    }
    let mut main = main.trim().to_string();
    // optional "name:" prefix
    if let Some(i) = text::find_top(&main, ':') {
        let nm = main[..i].trim();
        if !nm.contains(char::is_whitespace) && !nm.is_empty() {
            e.name = Some(nm.to_string());
            main = main[i + 1..].trim().to_string();
        }
    }
    if let Some(i) = text::find_top(&main, '=') {
        e.dep = main[..i].trim().to_string();
        let mut rhs = main[i + 1..].trim().to_string();
        if rhs.starts_with('(') && rhs.ends_with(')') && text::find_top(&rhs[1..rhs.len() - 1], ')').is_none() {
            rhs = rhs[1..rhs.len() - 1].to_string();
        }
        e.nl = Some(rhs);
        return Ok(e);
    }
    let ws = text::words(&main);
    let mut it = ws.into_iter();
    e.dep = it.next().unwrap_or_default();
    let mut ind = Vec::new();
    for w in it {
        if is_re_token(&w) {
            e.re.push(parse_re(&w));
        } else {
            ind.push(w);
        }
    }
    e.indep = ind.join(" ");
    Ok(e)
}

/// Parse the text after the verb.
pub fn parse(rest: &str) -> Result<Spec, String> {
    let mut sp = Spec::default();
    let (main, opts) = match text::find_top(rest, ',') {
        Some(i) => (rest[..i].trim(), rest[i + 1..].trim()),
        None => (rest.trim(), ""),
    };
    for (n, a) in text::options(opts) {
        if text::abbrev(&n, "likelihood", 2) {
            sp.lik = a;
        } else if n == "prior" {
            sp.priors.push(a.unwrap_or_default());
        } else if n == "_dprior" {
            sp.dpriors.push(a.unwrap_or_default());
        } else if text::abbrev(&n, "block", 5) {
            sp.blocks.push(a.unwrap_or_default());
        } else if text::abbrev(&n, "initial", 4) {
            sp.initial.push(a.unwrap_or_default());
        } else {
            sp.opts.push((n, a));
        }
    }
    let ws = text::words(main);
    let mut k = 0;
    let mut eq_words = Vec::new();
    while k < ws.len() && ws[k] != "if" && ws[k] != "in" {
        eq_words.push(ws[k].clone());
        k += 1;
    }
    sp.ifin = ws[k..].join(" ");
    if eq_words.first().is_some_and(|w| w.starts_with('(')) {
        for w in &eq_words {
            if !(w.starts_with('(') && w.ends_with(')')) {
                return Err(rc_err(198, format!("invalid equation specification {w}")));
            }
            sp.eqs.push(parse_eq(&w[1..w.len() - 1])?);
        }
    } else if !eq_words.is_empty() {
        sp.eqs.push(parse_eq(&eq_words.join(" "))?);
    }
    Ok(sp)
}

// ---------------------------------------------------------------------------
// name resolution
// ---------------------------------------------------------------------------

struct Res<'a> {
    m: &'a mut Model,
    ds: &'a DataSet,
    st: &'a Store,
    data: bool,
}

impl<'a> Resolver for Res<'a> {
    fn brace(&mut self, t: &str) -> Result<CE, String> {
        let t = t.trim();
        if t.contains(',') {
            return Err(rc_err(198, format!("matrix parameter {{{t}}} not allowed in a scalar expression")));
        }
        if let Some(eq) = t.strip_suffix(':') {
            let k = self.m.eqs.iter().position(|e| e.name == eq.trim()).ok_or_else(|| rc_err(198, format!("equation {eq} not found")))?;
            return Ok(CE::Xb(k));
        }
        let name = if let Some((e, p)) = t.split_once(':') { format!("{}:{}", e.trim(), p.trim()) } else { t.to_string() };
        if let Some((nm, init)) = name.split_once('=') {
            let j = self.m.add_param(nm.trim());
            let _ = init;
            return Ok(CE::Par(j));
        }
        Ok(CE::Par(self.m.add_param(&name)))
    }
    fn name(&mut self, n: &str) -> Result<CE, String> {
        if self.data {
            if let Ok(v) = crate::varlist::resolve_name(self.ds, n) {
                if self.ds.get_var(&v).is_some_and(|x| !x.var_type.is_string()) {
                    return Ok(CE::Col(self.m.col_index(&v, Vec::new())));
                }
            }
        }
        if let Some(v) = self.st.scalars.get(n) {
            return Ok(CE::Num(*v));
        }
        Err(rc_err(111, format!("{n} not found")))
    }
}

/// `{y:x}` / `{var}` style default initial values `{name=val}` inside braces.
fn inline_inits(s: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for b in text::braces(s) {
        if let Some((n, v)) = b.split_once('=') {
            if let Ok(x) = v.trim().parse::<f64>() {
                out.push((n.trim().to_string(), x));
            }
        }
    }
    out
}

fn strip_inline_inits(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    let mut skip = false;
    for c in s.chars() {
        match c {
            '{' => {
                depth += 1;
                out.push(c);
            }
            '}' => {
                depth -= 1;
                skip = false;
                out.push(c);
            }
            '=' if depth > 0 => skip = true,
            _ if skip => {}
            _ => out.push(c),
        }
    }
    out
}

fn compile_arg(s: &str, m: &mut Model, ds: &DataSet, st: &Store, data: bool) -> Result<Arg, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Ok(Arg::Num(v));
    }
    let mut r = Res { m, ds, st, data };
    let e = expr::compile(s, &mut r)?;
    Ok(match e {
        CE::Num(v) => Arg::Num(v),
        CE::Par(j) => Arg::Par(j),
        e => match e.constant() {
            Some(v) => Arg::Num(v),
            None => Arg::Ex(e),
        },
    })
}

fn compile_ce(s: &str, m: &mut Model, ds: &DataSet, st: &Store) -> Result<CE, String> {
    let mut r = Res { m, ds, st, data: true };
    expr::compile(s, &mut r)
}

/// Resolve a parameter list `{y:x1 x2} {var} {U} {S,m}`.
pub fn resolve_refs(m: &mut Model, s: &str, create: bool) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    let groups = text::braces(s);
    if groups.is_empty() {
        return Err(rc_err(198, format!("parameter specification {s} invalid; parameters are written in braces")));
    }
    for g in groups {
        let g = g.trim().to_string();
        if let Some((nm, _)) = g.split_once(',') {
            let k = m.mats.iter().position(|x| x.name == nm.trim()).ok_or_else(|| rc_err(198, format!("matrix parameter {nm} not found")))?;
            out.extend(m.mats[k].idx.iter().copied());
            continue;
        }
        if let Some((eq, rest)) = g.split_once(':') {
            let eq = eq.trim();
            let rest = rest.trim();
            if rest.is_empty() {
                let pre = format!("{eq}:");
                let mut found = false;
                for (j, p) in m.params.iter().enumerate() {
                    if p.name.starts_with(&pre) {
                        out.push(j);
                        found = true;
                    }
                }
                // equations created from one outcome (mlogit) share the depvar
                for e in &m.eqs {
                    if e.depvar == eq && e.name != eq {
                        out.extend(e.coefs.iter().copied());
                        found = true;
                    }
                }
                if !found {
                    return Err(rc_err(198, format!("equation {eq} not found")));
                }
                continue;
            }
            for w in rest.split_whitespace() {
                let name = format!("{eq}:{w}");
                match m.find(&name) {
                    Some(j) => out.push(j),
                    None if create => out.push(m.add_param(&name)),
                    None => return Err(rc_err(198, format!("parameter {{{name}}} not found"))),
                }
            }
            continue;
        }
        for w in g.split_whitespace() {
            if let Some(t) = m.re_terms.iter().position(|t| t.name == w) {
                out.extend(m.re_terms[t].params.iter().copied());
                continue;
            }
            if let Some(k) = m.mats.iter().position(|x| x.name == w) {
                out.extend(m.mats[k].idx.iter().copied());
                continue;
            }
            match m.find(w) {
                Some(j) => out.push(j),
                None if create => out.push(m.add_param(w)),
                None => return Err(rc_err(198, format!("parameter {{{w}}} not found"))),
            }
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    out.retain(|j| seen.insert(*j));
    Ok(out)
}

fn named_matrix(s: &str, st: &Store, d: usize) -> Result<Matrix, String> {
    let s = s.trim();
    // scalar multiples: 100*I(3), I(2)*5
    if let Some(i) = text::find_top(s, '*') {
        let (a, b) = (s[..i].trim(), s[i + 1..].trim());
        if let Ok(v) = a.parse::<f64>() {
            return Ok(named_matrix(b, st, d)?.scale(v));
        }
        if let Ok(v) = b.parse::<f64>() {
            return Ok(named_matrix(a, st, d)?.scale(v));
        }
    }
    if let Some(inner) = s.strip_prefix("I(").and_then(|x| x.strip_suffix(')')) {
        let k: usize = inner.trim().parse().map_err(|_| rc_err(198, format!("invalid {s}")))?;
        return Ok(Matrix::identity(k));
    }
    if let Some(m) = st.matrices.get(s) {
        return Ok(m.clone());
    }
    if let Ok(v) = s.parse::<f64>() {
        return Ok(Matrix::identity(d.max(1)).scale(v));
    }
    Err(rc_err(111, format!("matrix {s} not found")))
}

/// Declare matrix parameters met as `{S,m}` / `{S, matrix}`.
fn declare_matrices(m: &mut Model, s: &str, dims: &BTreeMap<String, usize>) {
    for g in text::braces(s) {
        if let Some((nm, kind)) = g.split_once(',') {
            let nm = nm.trim().to_string();
            let kind = kind.trim();
            if !(kind.starts_with('m')) || m.mats.iter().any(|x| x.name == nm) {
                continue;
            }
            let d = dims.get(&nm).copied().unwrap_or(2);
            let mi = m.mats.len();
            let mut idx = Vec::new();
            for i in 0..d {
                for j in 0..=i {
                    let pj = m.params.len();
                    m.params.push(Param { name: format!("{nm}_{}_{}", i + 1, j + 1), support: Support::Real, re: None, mat: Some((mi, i, j)) });
                    idx.push(pj);
                }
            }
            m.mats.push(MatParam { name: nm, d, idx });
        }
    }
}

// ---------------------------------------------------------------------------
// building
// ---------------------------------------------------------------------------

enum LikPending {
    Ready(Fam),
    Normal(Arg),
    Ologit(bool),
    Mlogit,
    Llf(CE),
    D(String, Vec<CE>, Option<Arg>),
    BinLogit(String),
    MvNormal(usize),
    NBreg(Arg),
    Tobit(Option<String>, Option<String>, Arg),
    Beta(Arg),
    Ald(f64, Arg),
    GNBreg(CE),
    Interval(String, String, Arg),
    Trunc(Option<String>, Option<String>, Arg),
    Weibull(Arg, String, Option<String>),
    Exponential(String, Option<String>),
    Glm(String, String, Option<String>),
    Quad(String, String, f64),
}

fn lik_pending(spec: &str, m: &mut Model, ds: &DataSet, st: &Store) -> Result<(LikPending, Vec<(String, Option<String>)>), String> {
    let (head, sub) = match text::find_top(spec, ',') {
        Some(i) => (spec[..i].trim(), text::options(&spec[i + 1..])),
        None => (spec.trim(), Vec::new()),
    };
    let (name, args) = text::call(head);
    let args: Vec<String> = args.map(|a| text::split_top(&a, ',')).unwrap_or_default();
    let a0 = |m: &mut Model| -> Result<Arg, String> { compile_arg(args.first().ok_or_else(|| rc_err(198, format!("{name}() requires an argument")))?, m, ds, st, false) };
    let p = match name.as_str() {
        "normal" => LikPending::Normal(a0(m)?),
        "logit" | "logistic" => LikPending::Ready(Fam::Logit),
        "probit" => LikPending::Ready(Fam::Probit),
        "cloglog" => LikPending::Ready(Fam::Cloglog),
        "poisson" => LikPending::Ready(Fam::Poisson),
        "ologit" => LikPending::Ologit(true),
        "oprobit" => LikPending::Ologit(false),
        "mlogit" => LikPending::Mlogit,
        "binlogit" => LikPending::BinLogit(args.first().cloned().unwrap_or_else(|| "1".into())),
        "llf" => LikPending::Llf(compile_ce(args.join(","). as_str(), m, ds, st)?),
        "dnormal" | "dbernoulli" | "dpoisson" | "dbinomial" | "dexponential" => {
            let mut ces = Vec::new();
            let mut var = None;
            for (k, a) in args.iter().enumerate() {
                if name == "dnormal" && k == 1 {
                    var = Some(compile_arg(a, m, ds, st, false)?);
                } else {
                    ces.push(compile_ce(a, m, ds, st)?);
                }
            }
            LikPending::D(name.clone(), ces, var)
        }
        "mvnormal" => {
            let g = text::braces(args.first().map(String::as_str).unwrap_or(""));
            let nm = g.first().and_then(|x| x.split(',').next()).map(|x| x.trim().to_string()).ok_or_else(|| rc_err(198, "mvnormal() requires a matrix parameter {Sigma,m}"))?;
            let k = m.mats.iter().position(|x| x.name == nm).ok_or_else(|| rc_err(198, format!("matrix parameter {nm} not declared")))?;
            LikPending::MvNormal(k)
        }
        "_nbreg" => LikPending::NBreg(a0(m)?),
        "_tobit" => {
            let ll = args.first().filter(|s| !s.trim().is_empty() && s.trim() != ".").cloned();
            let ul = args.get(1).filter(|s| !s.trim().is_empty() && s.trim() != ".").cloned();
            LikPending::Tobit(ll, ul, compile_arg(args.get(2).map(String::as_str).unwrap_or("1"), m, ds, st, false)?)
        }
        "_beta" => LikPending::Beta(a0(m)?),
        "_fraclogit" => LikPending::Ready(Fam::FracLogit),
        "_interval" => LikPending::Interval(args.first().cloned().unwrap_or_default(), args.get(1).cloned().unwrap_or_default(), compile_arg(args.get(2).map(String::as_str).unwrap_or("1"), m, ds, st, false)?),
        "_trunc" => {
            let ll = args.first().filter(|s| !s.trim().is_empty() && s.trim() != ".").cloned();
            let ul = args.get(1).filter(|s| !s.trim().is_empty() && s.trim() != ".").cloned();
            LikPending::Trunc(ll, ul, compile_arg(args.get(2).map(String::as_str).unwrap_or("1"), m, ds, st, false)?)
        }
        "_tpoisson" => LikPending::Ready(Fam::TPoisson(args.first().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0))),
        "_zip" => LikPending::Ready(Fam::Zip(None)),
        "_zinb" => LikPending::Ready(Fam::Zip(Some(a0(m)?))),
        "_ald" => LikPending::Ald(args.first().and_then(|s| s.trim().parse().ok()).unwrap_or(0.5), compile_arg(args.get(1).map(String::as_str).unwrap_or("1"), m, ds, st, false)?),
        "_gnbreg" => LikPending::GNBreg(compile_ce(&args.join(","), m, ds, st)?),
        "_fracprobit" => LikPending::Ready(Fam::FracProbit),
        "_weibull" => LikPending::Weibull(a0(m)?, args.get(1).cloned().unwrap_or_default(), args.get(2).cloned()),
        "_exponential" => LikPending::Exponential(args.first().cloned().unwrap_or_default(), args.get(1).cloned()),
        "_glm" => LikPending::Glm(args.first().cloned().unwrap_or_default(), args.get(1).cloned().unwrap_or_default(), args.get(2).cloned()),
        "_quad" => LikPending::Quad(args.first().cloned().unwrap_or_default(), args.get(1).cloned().unwrap_or_default(), args.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0)),
        o => return Err(rc_err(198, format!("likelihood model {o} not supported"))),
    };
    Ok((p, sub))
}

fn col_over(ds: &DataSet, name: &str, rows: &[usize]) -> Result<Vec<f64>, String> {
    let v = crate::util::col(ds, name)?;
    Ok(rows.iter().map(|&i| v[i]).collect())
}

fn num_or_col(ds: &DataSet, s: &str, rows: &[usize]) -> Result<Vec<f64>, String> {
    if let Ok(v) = s.trim().parse::<f64>() {
        return Ok(vec![v; rows.len()]);
    }
    let nm = crate::varlist::resolve_name(ds, s.trim())?;
    col_over(ds, &nm, rows)
}

/// Result of building: the model, initial values and settings.
pub struct Built {
    pub model: Model,
    pub init: Vec<f64>,
    pub user_init: Vec<bool>,
    pub settings: super::mcmc::Settings,
    pub spec: Spec,
    pub title: String,
}

pub fn build(ds: &DataSet, st: &Store, rest: &str) -> Result<Built, String> {
    let rest_clean = strip_inline_inits(rest);
    let inl = inline_inits(rest);
    let sp = parse(&rest_clean)?;
    let mut m = Model::default();
    // matrix dimensions from priors: wishart(d, ...), iwishart(d, ...), mvnormal(d, ...)
    let mut dims = BTreeMap::new();
    for p in sp.priors.iter().chain(&sp.dpriors) {
        if let Some(i) = text::find_top(p, ',') {
            let refs = &p[..i];
            let (dn, da) = text::call(p[i + 1..].trim());
            if matches!(dn.as_str(), "wishart" | "iwishart") {
                if let Some(d) = da.as_ref().and_then(|a| text::split_top(a, ',').first().and_then(|x| x.trim().parse::<usize>().ok())) {
                    for g in text::braces(refs) {
                        let nm = g.split(',').next().unwrap_or("").trim().to_string();
                        dims.insert(nm, d);
                    }
                }
            }
        }
    }
    let all_text = format!("{} {} {}", sp.lik.clone().unwrap_or_default(), format!("{} {}", sp.priors.join(" "), sp.dpriors.join(" ")), sp.eqs.iter().filter_map(|e| e.lik.clone()).collect::<Vec<_>>().join(" "));
    declare_matrices(&mut m, &all_text, &dims);
    // if/in mask
    let n_all = ds.n_obs;
    let ifin_cmd = ferrostat_parser::parser::parse_line(&format!("regress {}", sp.ifin)).map_err(|e| rc_err(198, e.message))?;
    let mut mask = crate::util::touse(ds, st, &ifin_cmd)?;
    if sp.has("_esample", 4) {
        if let Some(es) = &st.e.sample {
            for (i, mk) in mask.iter_mut().enumerate() {
                *mk &= es.get(i).copied().unwrap_or(false);
            }
        }
    }
    // equations: designs
    struct Pre {
        name: String,
        dep: String,
        design: Option<est::Design>,
        nl: Option<CE>,
        re: Vec<(String, String, Option<String>)>,
        lik: Option<String>,
    }
    let mut pres: Vec<Pre> = Vec::new();
    let global_noconst = sp.has("noconstant", 4);
    for e in &sp.eqs {
        let dep = if e.dep.is_empty() || e.dep == "_" { String::new() } else { crate::varlist::resolve_name(ds, &e.dep).unwrap_or(e.dep.clone()) };
        let name = e.name.clone().unwrap_or_else(|| dep.clone());
        if e.nl.is_none() && dep.is_empty() && e.dep == "_" {
            // linear form without an outcome (e.g. an inflation equation)
            let c = ferrostat_parser::parser::parse_line(&format!("regress {} {}", e.indep, sp.ifin)).map_err(|e| rc_err(198, e.message))?;
            let d = est::build(ds, st, &c, DesignSpec { depvar: None, indep: &e.indep, noconstant: e.noconst || global_noconst, weights: "", extra: Vec::new() })?;
            for i in 0..n_all {
                mask[i] &= d.mask[i];
            }
            pres.push(Pre { name, dep, design: Some(d), nl: None, re: e.re.clone(), lik: e.lik.clone() });
            continue;
        }
        if e.nl.is_none() {
            let c = ferrostat_parser::parser::parse_line(&format!("regress {} {} {}", dep, e.indep, sp.ifin)).map_err(|e| rc_err(198, e.message))?;
            let extra: Vec<String> = e.re.iter().flat_map(|r| std::iter::once(r.1.clone()).chain(r.2.clone())).collect();
            // outcome-only distributions (dpoisson(), llf(), ...) have no linear form
            let lk = e.lik.clone().or(sp.lik.clone()).unwrap_or_default();
            let lname = text::call(text::split_top(&lk, ',')[0].as_str()).0;
            let no_xb = e.indep.trim().is_empty() && (lname.starts_with('d') || lname == "llf");
            let d = est::build(ds, st, &c, DesignSpec { depvar: Some(&dep), indep: &e.indep, noconstant: e.noconst || global_noconst || no_xb, weights: "", extra })?;
            for i in 0..n_all {
                mask[i] &= d.mask[i];
            }
            pres.push(Pre { name, dep, design: Some(d), nl: None, re: e.re.clone(), lik: e.lik.clone() });
        } else {
            if !dep.is_empty() {
                let y = crate::util::col(ds, &dep)?;
                for i in 0..n_all {
                    mask[i] &= !is_missing(y[i]);
                }
            }
            pres.push(Pre { name, dep, design: None, nl: None, re: e.re.clone(), lik: e.lik.clone() });
        }
    }
    // equations exist (empty) so that {eq:} references resolve
    for p in &pres {
        m.eqs.push(Eq { name: p.name.clone(), depvar: p.dep.clone(), y: Vec::new(), xnames: Vec::new(), x: Vec::new(), coefs: Vec::new(), offset: Vec::new(), nl: None, re: Vec::new() });
    }
    // coefficients
    let mut is_mlogit = vec![false; pres.len()];
    for (k, p) in pres.iter().enumerate() {
        let lk = p.lik.clone().or(sp.lik.clone()).unwrap_or_default();
        is_mlogit[k] = text::call(text::split_top(&lk, ',')[0].as_str()).0 == "mlogit";
        if let Some(d) = &p.design {
            if !is_mlogit[k] {
                let mut coefs = Vec::new();
                let mut names = Vec::new();
                for j in d.active() {
                    names.push(d.names[j].clone());
                    coefs.push(m.add_param(&format!("{}:{}", p.name, d.names[j])));
                }
                m.eqs[k].coefs = coefs;
                m.eqs[k].xnames = names;
            }
        }
    }
    // substitutable expressions
    for (k, e) in sp.eqs.iter().enumerate() {
        if let Some(nl) = &e.nl {
            let ce = compile_ce(nl, &mut m, ds, st)?;
            pres[k].nl = Some(ce);
        }
    }
    // likelihoods (compile)
    let mut pend: Vec<(Vec<usize>, LikPending, Vec<(String, Option<String>)>)> = Vec::new();
    let mut global: Option<(LikPending, Vec<(String, Option<String>)>)> = None;
    for k in 0..pres.len() {
        if let Some(l) = pres[k].lik.clone() {
            let (p, sub) = lik_pending(&l, &mut m, ds, st)?;
            pend.push((vec![k], p, sub));
        }
    }
    if let Some(l) = &sp.lik {
        global = Some(lik_pending(l, &mut m, ds, st)?);
    }
    // user evaluator programs: llevaluator(prog [, parameters({a} {b})]) / evaluator(...)
    let mut evaluator_post = false;
    for (on, post) in [("llevaluator", false), ("evaluator", true)] {
        if let Some(Some(a)) = sp.opt(on, if post { 4 } else { 6 }) {
            if on == "evaluator" && sp.opts.iter().any(|(n, _)| n.starts_with("llev")) {
                continue;
            }
            let cb = super::model::CALLBACK.with(|c| c.borrow().clone()).ok_or_else(|| rc_err(198, format!("{on}() requires the program interpreter")))?;
            if let Some(i) = text::find_top(a, ',') {
                for (sn, sa) in text::options(&a[i + 1..]) {
                    if text::abbrev(&sn, "parameters", 5) {
                        let refs = sa.unwrap_or_default();
                        resolve_refs(&mut m, &refs, true)?;
                    }
                }
            }
            evaluator_post = post;
            global = Some((LikPending::Ready(Fam::Callback(cb, post)), Vec::new()));
        }
    }
    // columns referenced by expressions: mark out missing
    for cn in m.colnames.clone() {
        let v = crate::util::col(ds, &cn)?;
        for i in 0..n_all {
            mask[i] &= !is_missing(v[i]);
        }
    }
    // sub-option variables (exposure/offset) and RE group variables
    let mut extra_vars: Vec<String> = Vec::new();
    for p in &pres {
        for r in &p.re {
            extra_vars.push(r.1.clone());
            if let Some(s) = &r.2 {
                extra_vars.push(s.clone());
            }
        }
    }
    for v in &extra_vars {
        let nm = crate::varlist::resolve_name(ds, v)?;
        let c = crate::util::col(ds, &nm)?;
        for i in 0..n_all {
            mask[i] &= !is_missing(c[i]);
        }
    }
    let rows: Vec<usize> = (0..n_all).filter(|&i| mask[i]).collect();
    if rows.is_empty() && !pres.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let n = if pres.is_empty() && m.colnames.is_empty() { if global.as_ref().is_some_and(|g| matches!(g.0, LikPending::Llf(_))) { 1 } else { 0 } } else { rows.len() };
    m.n = if pres.is_empty() && m.colnames.is_empty() { n } else { rows.len() };
    let rows: Vec<usize> = if pres.is_empty() && m.colnames.is_empty() { (0..m.n).collect() } else { rows };
    m.mask = mask.clone();
    // fill columns
    for (j, cn) in m.colnames.clone().iter().enumerate() {
        m.cols[j] = col_over(ds, cn, &rows)?;
    }
    // fill equations
    for (k, p) in pres.iter().enumerate() {
        if !p.dep.is_empty() {
            m.eqs[k].y = col_over(ds, &p.dep, &rows)?;
        }
        if let Some(d) = &p.design {
            let pos: std::collections::HashMap<usize, usize> = d.rows.iter().enumerate().map(|(t, &r)| (r, t)).collect();
            let act = d.active();
            m.eqs[k].x = act.iter().map(|&j| rows.iter().map(|r| d.cols[j][pos[r]]).collect()).collect();
        }
        m.eqs[k].nl = p.nl.clone();
    }
    // random-effects terms
    for (k, p) in pres.iter().enumerate() {
        for (tn, gv, slope) in &p.re {
            let gvn = crate::varlist::resolve_name(ds, gv)?;
            let g = col_over(ds, &gvn, &rows)?;
            let z = match slope {
                Some(s) => Some(col_over(ds, &crate::varlist::resolve_name(ds, s)?, &rows)?),
                None => None,
            };
            let ti = match m.re_terms.iter().position(|t| &t.name == tn) {
                Some(t) => t,
                None => {
                    let mut lv = g.clone();
                    lv.sort_by(f64::total_cmp);
                    lv.dedup();
                    let ti = m.re_terms.len();
                    let mut params = Vec::new();
                    for l in &lv {
                        let pj = m.params.len();
                        m.params.push(Param { name: format!("{tn}[{}]", fmt_g(*l, 10)), support: Support::Real, re: Some(ti), mat: None });
                        params.push(pj);
                    }
                    m.re_terms.push(ReTerm { name: tn.clone(), groupvar: gvn.clone(), gidx: Vec::new(), z: None, params, levels: lv });
                    ti
                }
            };
            let t = &mut m.re_terms[ti];
            t.gidx = g.iter().map(|v| t.params[t.levels.iter().position(|l| l == v).unwrap_or(0)]).collect();
            t.z = z;
            m.eqs[k].re.push(ti);
        }
    }
    // build likelihoods
    let mut title = String::new();
    let mut add_lik = |m: &mut Model, eqs: Vec<usize>, p: LikPending, sub: &[(String, Option<String>)], title: &mut String| -> Result<(), String> {
        // offset / exposure
        for (sn, sa) in sub {
            if let Some(a) = sa {
                if text::abbrev(sn, "offset", 3) || text::abbrev(sn, "exposure", 3) {
                    let v = num_or_col(ds, a, &rows)?;
                    let v: Vec<f64> = if sn.starts_with("exp") { v.iter().map(|x| x.ln()).collect() } else { v };
                    for &k in &eqs {
                        m.eqs[k].offset = v.clone();
                    }
                }
            }
        }
        let fam = match p {
            LikPending::Ready(f) => f,
            LikPending::Normal(a) => Fam::Normal(a),
            LikPending::Ologit(lg) => {
                let k = eqs[0];
                let y = m.eqs[k].y.clone();
                let mut lv = y.clone();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                if lv.len() < 2 {
                    return Err(rc_err(148, "outcome does not vary"));
                }
                // ordinal models have no intercept
                if let Some(c) = m.eqs[k].xnames.iter().position(|x| x == "_cons") {
                    m.eqs[k].xnames.remove(c);
                    m.eqs[k].x.remove(c);
                    let pj = m.eqs[k].coefs.remove(c);
                    m.params[pj].name = "__dropped".into();
                }
                let cat: Vec<usize> = y.iter().map(|v| lv.iter().position(|l| l == v).unwrap()).collect();
                let mut cuts = Vec::new();
                for c in 1..lv.len() {
                    cuts.push(m.add_param(&format!("cut{c}")));
                }
                if lg { Fam::Ologit(cuts, cat) } else { Fam::Oprobit(cuts, cat) }
            }
            LikPending::Mlogit => {
                // one equation per outcome other than the base (the lowest)
                let k = eqs[0];
                let y = m.eqs[k].y.clone();
                let mut lv = y.clone();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                let base_pos = sub.iter().find(|(n, _)| text::abbrev(n, "baseoutcome", 4)).and_then(|(_, a)| a.as_ref()).and_then(|a| a.trim().parse::<f64>().ok()).and_then(|b| lv.iter().position(|l| *l == b)).unwrap_or(0);
                let cat: Vec<usize> = y.iter().map(|v| {
                    let p = lv.iter().position(|l| l == v).unwrap();
                    if p == base_pos { 0 } else if p < base_pos { p + 1 } else { p }
                }).collect();
                let d = pres[k].design.as_ref().unwrap();
                let act = d.active();
                let template = m.eqs[k].clone();
                let mut new_eqs = Vec::new();
                for (p, l) in lv.iter().enumerate() {
                    if p == base_pos {
                        continue;
                    }
                    let mut e = template.clone();
                    e.name = fmt_g(*l, 10);
                    e.coefs = act.iter().map(|&j| m.add_param(&format!("{}:{}", e.name, d.names[j]))).collect();
                    e.xnames = act.iter().map(|&j| d.names[j].clone()).collect();
                    new_eqs.push(e);
                }
                // replace the template with the first outcome equation, append the rest
                let mut idx = vec![k];
                let mut it = new_eqs.into_iter();
                if let Some(first) = it.next() {
                    m.eqs[k] = first;
                }
                for e in it {
                    m.eqs.push(e);
                    idx.push(m.eqs.len() - 1);
                }
                m.liks.push(Lik { eqs: idx, fam: Fam::Mlogit(cat) });
                *title = "multinomial logistic regression".into();
                return Ok(());
            }
            LikPending::Llf(e) => Fam::Llf(e),
            LikPending::D(nm, ces, var) => {
                let mut it = ces.into_iter();
                let a = it.next().ok_or_else(|| rc_err(198, format!("{nm}() requires arguments")))?;
                match nm.as_str() {
                    "dnormal" => Fam::DNormal(a, var.unwrap_or(Arg::Num(1.0))),
                    "dbernoulli" => Fam::DBernoulli(a),
                    "dpoisson" => Fam::DPoisson(a),
                    "dexponential" => Fam::DExponential(a),
                    _ => Fam::DBinomial(a, it.next().unwrap_or(CE::Num(1.0))),
                }
            }
            LikPending::BinLogit(s) => Fam::BinLogit(num_or_col(ds, &s, &rows)?),
            LikPending::MvNormal(k) => Fam::MvNormal(k),
            LikPending::NBreg(a) => Fam::NBreg(a),
            LikPending::Tobit(ll, ul, a) => Fam::Tobit(
                match ll { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() },
                match ul { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() },
                a,
            ),
            LikPending::Beta(a) => Fam::Beta(a),
            LikPending::Ald(q, a) => Fam::Ald(q, a),
            LikPending::GNBreg(e) => Fam::GNBreg(e),
            LikPending::Interval(lo, hi, a) => Fam::Interval(num_or_col(ds, &lo, &rows)?, num_or_col(ds, &hi, &rows)?, a),
            LikPending::Trunc(ll, ul, a) => Fam::Trunc(
                match ll { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() },
                match ul { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() },
                a,
            ),
            LikPending::Weibull(a, f, t0) => Fam::Weibull(a, num_or_col(ds, &f, &rows)?, match t0 { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() }),
            LikPending::Exponential(f, t0) => Fam::Exponential(num_or_col(ds, &f, &rows)?, match t0 { Some(s) => num_or_col(ds, &s, &rows)?, None => Vec::new() }),
            LikPending::Glm(f, l, nt) => Fam::Glm(f, l, match nt { Some(s) => num_or_col(ds, &s, &rows)?, None => vec![1.0; rows.len()] }),
            LikPending::Quad(bn, vn, ll) => {
                let b = st.matrices.get(bn.trim()).or_else(|| st.e.matrices.get(bn.trim())).ok_or_else(|| rc_err(111, format!("matrix {bn} not found")))?.clone();
                let v = st.matrices.get(vn.trim()).or_else(|| st.e.matrices.get(vn.trim())).ok_or_else(|| rc_err(111, format!("matrix {vn} not found")))?.clone();
                let (vinv, _) = v.sym_inverse(1e-12);
                let idx: Vec<usize> = b.colnames.iter().map(|c| m.add_param(c)).collect();
                Fam::Quad { b: (0..b.cols).map(|j| b.get(0, j)).collect(), vinv, llmax: ll, idx }
            }
        };
        if title.is_empty() {
            *title = match &fam {
                Fam::Normal(_) => "normal regression".into(),
                Fam::Logit => "logistic regression".into(),
                Fam::Probit => "probit regression".into(),
                Fam::Poisson => "Poisson regression".into(),
                Fam::Ologit(..) => "ordered logistic regression".into(),
                Fam::Oprobit(..) => "ordered probit regression".into(),
                _ => "regression".into(),
            };
        }
        m.liks.push(Lik { eqs, fam });
        Ok(())
    };
    for (eqs, p, sub) in pend {
        add_lik(&mut m, eqs, p, &sub, &mut title)?;
    }
    if let Some((p, sub)) = global {
        let without: Vec<usize> = (0..pres.len()).filter(|k| pres[*k].lik.is_none()).collect();
        match p {
            LikPending::MvNormal(_) => add_lik(&mut m, without, p, &sub, &mut title)?,
            LikPending::Ready(Fam::Callback(..)) | LikPending::Ready(Fam::Zip(_)) => add_lik(&mut m, (0..pres.len()).collect(), p, &sub, &mut title)?,
            LikPending::Llf(_) if without.is_empty() => add_lik(&mut m, Vec::new(), p, &sub, &mut title)?,
            p if without.len() <= 1 => add_lik(&mut m, without, p, &sub, &mut title)?,
            p => {
                // the same family for each equation
                let tmpl = p;
                let first = without[0];
                let rest_eqs = without[1..].to_vec();
                let shared = match &tmpl {
                    LikPending::Normal(a) => Some(a.clone()),
                    _ => None,
                };
                let simple = match &tmpl {
                    LikPending::Ready(f) => Some(f.clone()),
                    _ => None,
                };
                add_lik(&mut m, vec![first], tmpl, &sub, &mut title)?;
                for k in rest_eqs {
                    let p2 = if let Some(a) = &shared { LikPending::Normal(a.clone()) } else if let Some(f) = &simple { LikPending::Ready(f.clone()) } else { return Err(rc_err(198, "specify likelihood() within each equation")) };
                    add_lik(&mut m, vec![k], p2, &sub, &mut title)?;
                }
            }
        }
    }
    for (k, p) in pres.iter().enumerate() {
        let _ = (k, p);
    }
    // drop parameters removed from equations (ordinal intercepts)
    remove_dropped(&mut m);
    // priors (user priors first; default priors for the remaining parameters)
    let mut covered0 = vec![false; 0];
    let all_priors: Vec<(String, bool)> = sp.priors.iter().map(|p| (p.clone(), false)).chain(sp.dpriors.iter().map(|p| (p.clone(), true))).collect();
    for (pt, is_default) in &all_priors {
        let i = text::find_top(pt, ',').ok_or_else(|| rc_err(198, format!("prior({pt}) invalid")))?;
        let mut refs = resolve_refs(&mut m, &pt[..i], true)?;
        if covered0.len() < m.params.len() {
            covered0.resize(m.params.len(), false);
        }
        if *is_default {
            refs.retain(|&j| !covered0[j]);
            if refs.is_empty() {
                continue;
            }
        }
        for &j in &refs {
            covered0[j] = true;
        }
        let dspec = pt[i + 1..].trim();
        let (dn, da) = text::call(dspec);
        let args: Vec<String> = da.map(|a| text::split_top(&a, ',')).unwrap_or_default();
        let kind = make_prior(&mut m, &dn, &args, &refs, ds, st)?;
        // support from constant arguments
        if let PK::Uni(u, a) = &kind {
            let consts: Vec<f64> = a.iter().map(|x| x.num().unwrap_or(f64::NAN)).collect();
            let sup = u.support(&consts);
            let sup = match sup {
                Support::Interval(a, b) if !(a.is_finite() && b.is_finite()) => Support::Real,
                Support::Lower(l) if !l.is_finite() => Support::Real,
                s => s,
            };
            for &j in &refs {
                if m.params[j].mat.is_none() {
                    m.params[j].support = sup;
                }
            }
        }
        if let PK::Dirichlet(_) = &kind {
            for &j in &refs {
                m.params[j].support = Support::Interval(0.0, 1.0);
            }
        }
        m.priors.push(Prior { params: refs, kind, label: dspec.to_string() });
    }
    // every parameter needs a prior
    let mut covered = vec![false; m.params.len()];
    for p in &m.priors {
        for &j in &p.params {
            covered[j] = true;
        }
    }
    if let Some(j) = (0..m.params.len()).find(|&j| !covered[j] && !evaluator_post) {
        return Err(rc_err(198, format!("parameter {{{}}} not found in any of the prior() options; every model parameter needs a prior", m.params[j].name)));
    }
    m.finalize();
    // initial values
    let mut init: Vec<f64> = m.params.iter().map(|p| match p.support {
        Support::Real => 0.0,
        Support::Lower(l) => l + 1.0,
        Support::Interval(a, b) => (a + b) / 2.0,
        Support::Discrete(lo, hi) => if hi.is_finite() { ((lo + hi) / 2.0).floor() } else { lo.max(0.0) },
    }).collect();
    for mp in &m.mats {
        for (k, &j) in mp.idx.iter().enumerate() {
            let _ = k;
            let (_, i, jj) = m.params[j].mat.unwrap();
            init[j] = if i == jj { 1.0 } else { 0.0 };
        }
    }
    // Dirichlet: start at the simplex centre
    for p in &m.priors {
        if let PK::Dirichlet(_) = p.kind {
            for &j in &p.params {
                init[j] = 1.0 / p.params.len() as f64;
            }
        }
    }
    // cutpoints start ordered
    for l in &m.liks {
        if let Fam::Ologit(c, _) | Fam::Oprobit(c, _) = &l.fam {
            let k = c.len() as f64;
            for (t, &j) in c.iter().enumerate() {
                init[j] = -1.0 + 2.0 * (t as f64 + 0.5) / k;
            }
        }
    }
    // a normal outcome: start the constant at the mean and the variance at the variance
    for l in &m.liks {
        if let Fam::Normal(a) = &l.fam {
            let e = &m.eqs[l.eqs[0]];
            if e.nl.is_none() && !e.y.is_empty() {
                let mu = super::stats::mean(&e.y);
                if let Some(c) = e.xnames.iter().position(|x| x == "_cons") {
                    init[e.coefs[c]] = mu;
                }
                if let Arg::Par(j) = a {
                    init[*j] = super::stats::var(&e.y).max(1e-6);
                }
            }
        }
    }
    let mut user_init = vec![false; m.params.len()];
    for (nm, v) in &inl {
        if let Ok(js) = resolve_refs(&mut m, &format!("{{{nm}}}"), false) {
            for j in js {
                init[j] = *v;
                user_init[j] = true;
            }
        }
    }
    for t in &sp.initial {
        let ws = text::words(t);
        let mut k = 0;
        while k + 1 < ws.len() {
            let r = if ws[k].starts_with('{') { ws[k].clone() } else { format!("{{{}}}", ws[k]) };
            let js = resolve_refs(&mut m, &r, false)?;
            let v: f64 = ws[k + 1].parse().or_else(|_| st.scalars.get(ws[k + 1].as_str()).copied().ok_or(())).map_err(|_| rc_err(198, format!("initial(): invalid value {}", ws[k + 1])))?;
            for j in js {
                init[j] = v;
                user_init[j] = true;
            }
            k += 2;
        }
    }
    // settings
    let mut set = super::mcmc::Settings::default();
    if let Some(v) = sp.num("mcmcsize", 4)? {
        set.mcmcsize = (v as usize).max(1);
    }
    if let Some(v) = sp.num("burnin", 4)? {
        set.burnin = v as usize;
    }
    if let Some(v) = sp.num("thinning", 4)? {
        set.thin = (v as usize).max(1);
    }
    if let Some(v) = sp.num("nchains", 6)? {
        set.nchains = (v as usize).max(1);
    }
    if let Some(Some(a)) = sp.opt("adaptation", 5) {
        for (n, v) in text::options(a) {
            let x = v.as_deref().and_then(|s| s.trim().parse::<f64>().ok());
            match (n.as_str(), x) {
                ("every", Some(x)) => set.every = (x as usize).max(1),
                ("maxiter", Some(x)) => set.maxiter = x as usize,
                ("tarate", Some(x)) => set.tarate = Some(x),
                _ => {}
            }
        }
    }
    set.gibbs_all = sp.has("gibbs", 5);
    for b in &sp.blocks {
        let (refs, bo) = match text::find_top(b, ',') {
            Some(i) => (&b[..i], text::options(&b[i + 1..])),
            None => (b.as_str(), Vec::new()),
        };
        let js = resolve_refs(&mut m, refs, false)?;
        let g = bo.iter().any(|(n, _)| n == "gibbs");
        let split = bo.iter().any(|(n, _)| n == "split");
        set.blocks.push((js, g, split));
    }
    if title.is_empty() {
        title = "model".into();
    }
    Ok(Built { model: m, init, user_init, settings: set, spec: sp, title })
}

fn remove_dropped(m: &mut Model) {
    let Some(j) = m.params.iter().position(|p| p.name == "__dropped") else { return };
    m.params.remove(j);
    let fix = |x: &mut usize| {
        if *x > j {
            *x -= 1;
        }
    };
    for e in m.eqs.iter_mut() {
        e.coefs.iter_mut().for_each(fix);
    }
    for l in m.liks.iter_mut() {
        match &mut l.fam {
            Fam::Ologit(c, _) | Fam::Oprobit(c, _) => c.iter_mut().for_each(fix),
            Fam::Normal(Arg::Par(p)) | Fam::NBreg(Arg::Par(p)) | Fam::Beta(Arg::Par(p)) => fix(p),
            Fam::Quad { idx, .. } => idx.iter_mut().for_each(fix),
            _ => {}
        }
    }
    for t in m.re_terms.iter_mut() {
        t.params.iter_mut().for_each(fix);
        t.gidx.iter_mut().for_each(fix);
    }
    for mp in m.mats.iter_mut() {
        mp.idx.iter_mut().for_each(fix);
    }
    remove_dropped(m);
}

fn make_prior(m: &mut Model, dn: &str, args: &[String], refs: &[usize], ds: &DataSet, st: &Store) -> Result<PK, String> {
    let carg = |m: &mut Model, s: &str| compile_arg(s, m, ds, st, false);
    if let Some(u) = Uni::from_name(dn) {
        if let Some(k) = u.nargs() {
            if args.iter().filter(|a| !a.trim().is_empty()).count() != k {
                return Err(rc_err(198, format!("{dn}() requires {k} argument{}", if k == 1 { "" } else { "s" })));
            }
        }
        let a = args.iter().filter(|a| !a.trim().is_empty()).map(|s| carg(m, s)).collect::<Result<Vec<_>, _>>()?;
        return Ok(PK::Uni(u, a));
    }
    let d = refs.len();
    Ok(match dn {
        "mvnormal" | "mvn" | "mvnormal0" | "mvn0" => {
            let zero = dn.ends_with('0');
            let dim: usize = args.first().and_then(|s| s.trim().parse().ok()).ok_or_else(|| rc_err(198, format!("{dn}() requires the dimension")))?;
            let covs = args.last().cloned().unwrap_or_default();
            let cov = if covs.contains('{') {
                let nm = text::braces(&covs).first().and_then(|g| g.split(',').next().map(|x| x.trim().to_string())).unwrap_or_default();
                match m.mats.iter().position(|x| x.name == nm) {
                    Some(k) => Cov::Mat(k),
                    None => {
                        let a = carg(m, &covs)?;
                        Cov::ScaledI(a)
                    }
                }
            } else {
                Cov::Fixed(named_matrix(&covs, st, dim)?)
            };
            let means: Vec<Arg> = if zero {
                vec![Arg::Num(0.0)]
            } else {
                let mid = &args[1..args.len().saturating_sub(1)];
                if mid.len() == 1 && st.matrices.contains_key(mid[0].trim()) {
                    let mm = &st.matrices[mid[0].trim()];
                    (0..mm.rows * mm.cols).map(|i| Arg::Num(mm.get(i / mm.cols, i % mm.cols))).collect()
                } else {
                    mid.iter().map(|s| carg(m, s)).collect::<Result<Vec<_>, _>>()?
                }
            };
            let _ = dim;
            PK::MvNormal(means, cov)
        }
        "zellnersg" | "zellnersg0" => {
            let zero = dn.ends_with('0');
            let g: f64 = args.get(1).and_then(|s| s.trim().parse().ok()).ok_or_else(|| rc_err(198, "zellnersg() requires dimension, g and variance"))?;
            let var = carg(m, args.last().unwrap())?;
            // (X'X)^-1 of the equation holding the first parameter
            let eq = m.eqs.iter().position(|e| e.coefs.contains(&refs[0])).ok_or_else(|| rc_err(198, "zellnersg() applies to regression coefficients"))?;
            let e = &m.eqs[eq];
            let cols: Vec<usize> = refs.iter().map(|j| e.coefs.iter().position(|c| c == j).ok_or_else(|| rc_err(198, "zellnersg() parameters must belong to one equation"))).collect::<Result<_, _>>()?;
            let mut xtx = Matrix::zeros(d, d);
            for a in 0..d {
                for b in 0..d {
                    xtx[(a, b)] = (0..m.n).map(|r| e.x[cols[a]][r] * e.x[cols[b]][r]).sum();
                }
            }
            let inv = xtx.inverse().map_err(|_| rc_err(506, "zellnersg(): X'X not invertible"))?.scale(g);
            let means: Vec<Arg> = if zero { vec![Arg::Num(0.0)] } else { args[2..args.len() - 1].iter().map(|s| carg(m, s)).collect::<Result<Vec<_>, _>>()? };
            // covariance g * var * (X'X)^-1: fold var into a matrix-scaled arg
            match var {
                Arg::Num(v) => PK::MvNormal(means, Cov::Fixed(inv.scale(v))),
                a => {
                    // represent as density through an expression is not possible; use ScaledI on a whitened scale
                    let k = m.mats.len();
                    let _ = k;
                    PK::MvNormal(means, Cov::Fixed(inv)).with_scale(a)
                }
            }
        }
        "wishart" | "iwishart" => {
            let dim: usize = args.first().and_then(|s| s.trim().parse().ok()).ok_or_else(|| rc_err(198, format!("{dn}() requires the dimension")))?;
            let df: f64 = args.get(1).and_then(|s| s.trim().parse().ok()).ok_or_else(|| rc_err(198, format!("{dn}() requires degrees of freedom")))?;
            let v = named_matrix(args.get(2).map(String::as_str).unwrap_or("I(1)"), st, dim)?;
            let k = m.mats.iter().position(|mp| mp.idx.first() == refs.first()).ok_or_else(|| rc_err(198, format!("{dn}() applies to a matrix parameter {{S,matrix}}")))?;
            if dn == "wishart" { PK::Wishart(df, v, k) } else { PK::IWishart(df, v, k) }
        }
        "dirichlet" => PK::Dirichlet(args.iter().map(|s| carg(m, s)).collect::<Result<Vec<_>, _>>()?),
        "density" => PK::Density(compile_ce(&args.join(","), m, ds, st)?),
        "logdensity" => PK::LogDensity(compile_ce(&args.join(","), m, ds, st)?),
        o => return Err(rc_err(198, format!("prior distribution {o}() not supported"))),
    })
}

trait WithScale {
    fn with_scale(self, a: Arg) -> PK;
}

impl WithScale for PK {
    /// mvnormal with covariance (fixed matrix) x scalar parameter
    fn with_scale(self, a: Arg) -> PK {
        match self {
            PK::MvNormal(means, Cov::Fixed(s)) => PK::MvNormal(means, Cov::Scaled(s, a)),
            o => o,
        }
    }
}

/// Log density helpers re-exported for the prefix builder.
pub fn lnf(x: f64) -> f64 {
    dist::lnfactorial(x)
}
