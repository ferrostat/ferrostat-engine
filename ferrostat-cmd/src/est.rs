//! Estimation framework: design matrices (factor variables, interactions,
//! time-series operators), estimation samples, variance estimators,
//! maximum-likelihood optimisation and coefficient tables.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind, Weights};

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Base {
    First,
    Last,
    Freq,
    Value(f64),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Factor {
    /// Continuous variable or time-series expression (`x`, `L.x`, `c.x`).
    Cont(String),
    /// Categorical variable with base specification.
    Cat(String, Base),
    /// Single level of a categorical variable (`3.g`).
    Level(String, f64),
}

impl Factor {
    fn var(&self) -> &str {
        match self {
            Factor::Cont(v) => crate::util::base_name(v),
            Factor::Cat(v, _) | Factor::Level(v, _) => v,
        }
    }
}

/// Parse one factor token like `i.g`, `ib3.g`, `ib(last).g`, `c.x`, `L.x`, `2.g`.
fn parse_factor(tok: &str) -> Result<Factor, String> {
    let (op, var) = match tok.rsplit_once('.') {
        Some((o, v)) if !o.is_empty() => (o.to_string(), v.to_string()),
        _ => return Ok(Factor::Cont(tok.to_string())),
    };
    let lo = op.to_lowercase();
    if lo == "c" {
        return Ok(Factor::Cont(var));
    }
    if lo == "i" {
        return Ok(Factor::Cat(var, Base::First));
    }
    if lo == "ibn" {
        return Ok(Factor::Cat(var, Base::None));
    }
    if let Some(b) = lo.strip_prefix("ib") {
        let b = b.trim_start_matches('(').trim_end_matches(')');
        let base = match b {
            "last" => Base::Last,
            "first" => Base::First,
            "freq" => Base::Freq,
            n => Base::Value(n.parse().map_err(|_| rc_err(198, format!("{tok}: invalid base")))?),
        };
        return Ok(Factor::Cat(var, base));
    }
    if let Ok(v) = lo.trim_end_matches(['b', 'o', 'n']).parse::<f64>() {
        return Ok(Factor::Level(var, v));
    }
    // time-series operator: keep whole token as continuous expression
    Ok(Factor::Cont(tok.to_string()))
}

/// A model term: product of factors.
pub type Term = Vec<Factor>;

/// Parse an indepvars list into terms, expanding `a##b`.
pub fn parse_terms(ds: &DataSet, text: &str) -> Result<Vec<Term>, String> {
    let mut terms: Vec<Term> = Vec::new();
    for raw in text.split_whitespace() {
        // i.(a b) groups
        let raw = raw.to_string();
        if raw.contains("##") {
            let parts: Vec<&str> = raw.split("##").collect();
            let facs: Vec<Factor> = parts.iter().map(|p| expand_inter(ds, p)).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
            // all non-empty subsets in order of size
            let k = facs.len();
            for size in 1..=k {
                for mask in 0..(1u32 << k) {
                    if mask.count_ones() as usize == size {
                        let term: Term = (0..k).filter(|i| mask & (1 << i) != 0).map(|i| facs[i].clone()).collect();
                        terms.push(term);
                    }
                }
            }
        } else if raw.contains('#') {
            let facs: Vec<Factor> = raw.split('#').map(|p| expand_inter(ds, p)).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
            terms.push(facs);
        } else {
            for f in expand_one(ds, &raw)? {
                terms.push(vec![f]);
            }
        }
    }
    Ok(terms)
}

/// Inside interactions, bare variable names are categorical.
fn expand_inter(ds: &DataSet, tok: &str) -> Result<Vec<Factor>, String> {
    let f = expand_one(ds, tok)?;
    if tok.contains('.') {
        return Ok(f);
    }
    Ok(f.into_iter().map(|x| match x { Factor::Cont(v) => Factor::Cat(v, Base::First), o => o }).collect())
}

/// Expand one token (may contain wildcards/abbreviations) into factors.
fn expand_one(ds: &DataSet, tok: &str) -> Result<Vec<Factor>, String> {
    let (prefix, var) = match tok.rsplit_once('.') {
        Some((o, v)) if !o.is_empty() => (Some(o), v),
        _ => (None, tok),
    };
    // interactions default to categorical for bare names
    let names = crate::varlist::expand(ds, var)?;
    names
        .into_iter()
        .map(|n| match prefix {
            Some(p) => parse_factor(&format!("{p}.{n}")),
            None => Ok(Factor::Cont(n)),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Design matrix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Design {
    pub names: Vec<String>,
    /// columns over sample rows
    pub cols: Vec<Vec<f64>>,
    /// base levels and collinear columns: coefficient fixed at 0
    pub omitted: Vec<bool>,
    pub is_base: Vec<bool>,
    pub rows: Vec<usize>,
    pub y: Vec<f64>,
    pub w: Vec<f64>,
    pub wkind: Option<WeightKind>,
    pub has_cons: bool,
    pub depvar: String,
    pub mask: Vec<bool>,
    pub notes: Vec<String>,
    pub cluster: Option<Vec<f64>>,
}

impl Design {
    pub fn k(&self) -> usize {
        self.cols.len()
    }
    pub fn n(&self) -> usize {
        self.rows.len()
    }
    /// Effective number of observations (sum of fweights).
    pub fn n_eff(&self) -> f64 {
        match self.wkind {
            Some(WeightKind::F) => self.w.iter().sum(),
            _ => self.rows.len() as f64,
        }
    }
    pub fn active(&self) -> Vec<usize> {
        (0..self.k()).filter(|&j| !self.omitted[j]).collect()
    }
    pub fn row(&self, t: usize) -> Vec<f64> {
        self.cols.iter().map(|c| c[t]).collect()
    }
    pub fn xb(&self, b: &[f64], t: usize) -> f64 {
        self.cols.iter().zip(b).map(|(c, bj)| if *bj == 0.0 { 0.0 } else { c[t] * bj }).sum()
    }
}

fn factor_values(ds: &DataSet, st: &Store, var: &str, rows_all: usize) -> Result<Vec<f64>, String> {
    if ds.var_index.contains_key(var) {
        let v = ds.get_var(var).unwrap();
        if v.var_type.is_string() {
            return Err(rc_err(109, format!("{var}: string variables not allowed")));
        }
        return Ok(v.data.iter().map(|x| x.num()).collect());
    }
    // time-series expression such as L.x
    let e = ferrostat_parser::parser::parse_expression(var).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st);
    (0..rows_all).map(|i| env.eval(&e, i)?.num()).collect()
}

pub struct DesignSpec<'a> {
    pub depvar: Option<&'a str>,
    pub indep: &'a str,
    pub noconstant: bool,
    pub weights: &'a str,
    /// Additional variables to mark out (e.g. cluster, panel).
    pub extra: Vec<String>,
}

/// Build the design for a command like `regress y x1 x2 [if] [in] [w]`.
pub fn build(ds: &DataSet, st: &Store, cmd: &Command, spec: DesignSpec) -> Result<Design, String> {
    let n_all = ds.n_obs;
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, spec.weights)?;
    let terms = parse_terms(ds, spec.indep)?;
    // depvar
    let y_all = match spec.depvar {
        Some(d) => factor_values(ds, st, d, n_all)?,
        None => vec![0.0; n_all],
    };
    if spec.depvar.is_some() {
        for i in 0..n_all {
            if is_missing(y_all[i]) {
                mask[i] = false;
            }
        }
    }
    // mark out missing in all factors
    let mut cache: std::collections::HashMap<String, Vec<f64>> = Default::default();
    for t in &terms {
        for f in t {
            let key = match f {
                Factor::Cont(v) => v.clone(),
                Factor::Cat(v, _) | Factor::Level(v, _) => v.clone(),
            };
            if !cache.contains_key(&key) {
                let vals = factor_values(ds, st, &key, n_all)?;
                cache.insert(key.clone(), vals);
            }
            let vals = &cache[&key];
            for i in 0..n_all {
                if is_missing(vals[i]) {
                    mask[i] = false;
                }
            }
            if let Factor::Cat(v, _) = f {
                for i in 0..n_all {
                    if mask[i] && (vals[i] < 0.0 || vals[i].fract() != 0.0) {
                        return Err(rc_err(452, format!("factor variables may not contain noninteger or negative values ({v})")));
                    }
                }
            }
        }
    }
    util::markout(ds, &mut mask, &spec.extra);
    let rows = util::rows(&mask);
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    // levels of categorical factors
    let mut levels: std::collections::HashMap<String, (Vec<f64>, Option<f64>)> = Default::default();
    for t in &terms {
        for f in t {
            if let Factor::Cat(v, base) = f {
                let vals = &cache[v];
                let mut lv: Vec<f64> = rows.iter().map(|&i| vals[i]).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                let b = match base {
                    Base::First => lv.first().copied(),
                    Base::Last => lv.last().copied(),
                    Base::Value(x) => Some(*x),
                    Base::None => None,
                    Base::Freq => {
                        let mut best = (0usize, lv[0]);
                        for &l in &lv {
                            let c = rows.iter().filter(|&&i| vals[i] == l).count();
                            if c > best.0 {
                                best = (c, l);
                            }
                        }
                        Some(best.1)
                    }
                };
                levels.insert(v.clone(), (lv, b));
            }
        }
    }
    // build columns
    let mut names = Vec::new();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut is_base = Vec::new();
    for t in &terms {
        // each factor yields a list of (name part, values, base?)
        let mut acc: Vec<(String, Vec<f64>, bool)> = vec![(String::new(), vec![1.0; rows.len()], false)];
        for f in t {
            let parts: Vec<(String, Vec<f64>, bool)> = match f {
                Factor::Cont(v) => {
                    let vals = &cache[v];
                    let name = if t.len() > 1 { format!("c.{v}") } else { v.clone() };
                    vec![(name, rows.iter().map(|&i| vals[i]).collect(), false)]
                }
                Factor::Level(v, l) => {
                    let vals = &cache[v];
                    vec![(format!("{}.{v}", fmt_g(*l, 12)), rows.iter().map(|&i| if vals[i] == *l { 1.0 } else { 0.0 }).collect(), false)]
                }
                Factor::Cat(v, _) => {
                    let (lv, b) = &levels[v];
                    let vals = &cache[v];
                    lv.iter()
                        .map(|&l| {
                            let base = Some(l) == *b;
                            (
                                format!("{}{}.{v}", fmt_g(l, 12), if base { "b" } else { "" }),
                                rows.iter().map(|&i| if vals[i] == l { 1.0 } else { 0.0 }).collect(),
                                base,
                            )
                        })
                        .collect()
                }
            };
            let mut next = Vec::new();
            for (an, av, ab) in &acc {
                for (pn, pv, pb) in &parts {
                    let name = if an.is_empty() { pn.clone() } else { format!("{an}#{pn}") };
                    let vals: Vec<f64> = av.iter().zip(pv).map(|(a, b)| a * b).collect();
                    next.push((name, vals, *ab || *pb));
                }
            }
            acc = next;
        }
        for (n, v, b) in acc {
            if names.contains(&n) {
                continue;
            }
            names.push(n);
            cols.push(v);
            is_base.push(b);
        }
    }
    if !spec.noconstant {
        names.push("_cons".into());
        cols.push(vec![1.0; rows.len()]);
        is_base.push(false);
    }
    let y: Vec<f64> = rows.iter().map(|&i| y_all[i]).collect();
    let ww: Vec<f64> = match &w {
        Some(w) => {
            let raw: Vec<f64> = rows.iter().map(|&i| w.w[i]).collect();
            match w.kind {
                // analytic and sampling weights are normalised to sum to N
                WeightKind::A | WeightKind::P => {
                    let s: f64 = raw.iter().sum();
                    raw.iter().map(|x| x * rows.len() as f64 / s).collect()
                }
                _ => raw,
            }
        }
        None => vec![1.0; rows.len()],
    };
    let omitted = is_base.clone();
    let mut d = Design {
        names,
        cols,
        omitted,
        is_base,
        rows,
        y,
        w: ww,
        wkind: w.as_ref().map(|w| w.kind),
        has_cons: !spec.noconstant,
        depvar: spec.depvar.unwrap_or("").to_string(),
        mask,
        notes: Vec::new(),
        cluster: None,
    };
    drop_collinear(&mut d);
    Ok(d)
}

/// Mark columns that are linear combinations of earlier ones as omitted.
pub fn drop_collinear(d: &mut Design) {
    let act: Vec<usize> = (0..d.k()).filter(|&j| !d.omitted[j]).collect();
    if act.is_empty() {
        return;
    }
    // put the constant first so other columns are dropped instead
    let mut order = act.clone();
    if d.has_cons {
        let c = d.k() - 1;
        order.retain(|&j| j != c);
        order.insert(0, c);
    }
    let xtx = cross(d, &order);
    let (_, dropped) = xtx.sym_inverse(1e-13 * 1000.0);
    for idx in dropped {
        let j = order[idx];
        d.omitted[j] = true;
        if !d.is_base[j] {
            d.notes.push(format!("note: {} omitted because of collinearity.", d.names[j]));
            d.names[j] = omit_name(&d.names[j]);
        }
    }
}

fn omit_name(n: &str) -> String {
    if n == "_cons" {
        return n.to_string();
    }
    match n.split_once('.') {
        Some((a, b)) if a.chars().all(|c| c.is_ascii_digit()) => format!("{a}o.{b}"),
        _ => format!("o.{n}"),
    }
}

/// X'WX over a subset of columns (with centering-free accumulation).
pub fn cross(d: &Design, cols: &[usize]) -> Matrix {
    let k = cols.len();
    let mut m = Matrix::zeros(k, k);
    for t in 0..d.n() {
        let w = d.w[t];
        for a in 0..k {
            let xa = d.cols[cols[a]][t] * w;
            if xa == 0.0 {
                continue;
            }
            for b in 0..=a {
                m[(a, b)] += xa * d.cols[cols[b]][t];
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            m[(b, a)] = m[(a, b)];
        }
    }
    m
}

// ---------------------------------------------------------------------------
// VCE
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Vce {
    Ols,
    Robust,
    Cluster(String),
    Oim,
}

pub fn parse_vce(cmd: &Command, default: Vce) -> Result<Vce, String> {
    if let Some(v) = cmd.opt("vce", 3) {
        let w: Vec<&str> = v.split_whitespace().collect();
        return match w.first().map(|s| s.to_lowercase()).as_deref() {
            Some("robust") | Some("r") | Some("hc1") => Ok(Vce::Robust),
            Some("cluster") | Some("cl") => Ok(Vce::Cluster(w.get(1).ok_or_else(|| rc_err(198, "vce(cluster clustvar) requires a variable"))?.to_string())),
            Some("ols") => Ok(Vce::Ols),
            Some("oim") | Some("opg") => Ok(Vce::Oim),
            Some(o) => Err(rc_err(198, format!("vce({o}) not allowed"))),
            None => Ok(default),
        };
    }
    if cmd.has_opt("robust", 1) {
        return Ok(Vce::Robust);
    }
    if let Some(c) = cmd.opt("cluster", 2) {
        return Ok(Vce::Cluster(c.trim().to_string()));
    }
    Ok(default)
}

/// Sandwich estimator: V = A (sum over clusters of s_g s_g') A * scale.
/// `scores[t]` is the per-observation score vector (gradient contribution).
pub fn sandwich(bread: &Matrix, scores: &[Vec<f64>], cluster: Option<&[f64]>, scale: f64) -> (Matrix, usize) {
    let k = bread.rows;
    let mut meat = Matrix::zeros(k, k);
    let mut ng = scores.len();
    match cluster {
        Some(cl) => {
            let mut sums: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
            for (t, s) in scores.iter().enumerate() {
                let e = sums.entry(cl[t].to_bits()).or_insert_with(|| vec![0.0; k]);
                for j in 0..k {
                    e[j] += s[j];
                }
            }
            ng = sums.len();
            for s in sums.values() {
                for a in 0..k {
                    for b in 0..k {
                        meat[(a, b)] += s[a] * s[b];
                    }
                }
            }
        }
        None => {
            for s in scores {
                for a in 0..k {
                    if s[a] == 0.0 {
                        continue;
                    }
                    for b in 0..k {
                        meat[(a, b)] += s[a] * s[b];
                    }
                }
            }
        }
    }
    let v = bread.mul(&meat).unwrap().mul(bread).unwrap().scale(scale);
    (v, ng)
}

/// Expand a VCE on active columns to the full k x k matrix.
pub fn expand_v(vact: &Matrix, act: &[usize], k: usize) -> Matrix {
    let mut v = Matrix::zeros(k, k);
    for (a, &i) in act.iter().enumerate() {
        for (b, &j) in act.iter().enumerate() {
            v[(i, j)] = vact[(a, b)];
        }
    }
    v
}

// ---------------------------------------------------------------------------
// OLS
// ---------------------------------------------------------------------------

pub struct OlsFit {
    pub b: Vec<f64>,
    pub v: Matrix,
    pub rss: f64,
    pub tss: f64,
    pub df_r: f64,
    pub df_m: f64,
    pub resid: Vec<f64>,
    pub xtx_inv: Matrix,
    pub n_clust: Option<usize>,
}

pub fn ols(d: &Design, vce: &Vce, cluster: Option<&[f64]>) -> Result<OlsFit, String> {
    let act = d.active();
    let k = d.k();
    let xtx = cross(d, &act);
    let (inv_a, _) = xtx.sym_inverse(1e-14);
    let mut xty = vec![0.0; act.len()];
    for t in 0..d.n() {
        let wy = d.w[t] * d.y[t];
        for (a, &j) in act.iter().enumerate() {
            xty[a] += d.cols[j][t] * wy;
        }
    }
    let ba: Vec<f64> = (0..act.len()).map(|a| (0..act.len()).map(|c| inv_a[(a, c)] * xty[c]).sum()).collect();
    let mut b = vec![0.0; k];
    for (a, &j) in act.iter().enumerate() {
        b[j] = ba[a];
    }
    let resid: Vec<f64> = (0..d.n()).map(|t| d.y[t] - d.xb(&b, t)).collect();
    let rss: f64 = (0..d.n()).map(|t| d.w[t] * resid[t] * resid[t]).sum();
    let sw: f64 = d.w.iter().sum();
    let ybar = (0..d.n()).map(|t| d.w[t] * d.y[t]).sum::<f64>() / sw;
    let tss: f64 = if d.has_cons {
        (0..d.n()).map(|t| d.w[t] * (d.y[t] - ybar).powi(2)).sum()
    } else {
        (0..d.n()).map(|t| d.w[t] * d.y[t] * d.y[t]).sum()
    };
    let n_eff = d.n_eff();
    let rank = act.len() as f64;
    let df_r = n_eff - rank;
    let df_m = rank - if d.has_cons { 1.0 } else { 0.0 };
    let (vact, n_clust) = match vce {
        Vce::Ols | Vce::Oim => (inv_a.scale(rss / df_r), None),
        Vce::Robust | Vce::Cluster(_) => {
            let scores: Vec<Vec<f64>> = (0..d.n())
                .map(|t| {
                    let f = d.w[t] * resid[t];
                    let fw = if d.wkind == Some(WeightKind::F) { d.w[t].sqrt() } else { 1.0 };
                    act.iter().map(|&j| d.cols[j][t] * f / fw).collect()
                })
                .collect();
            match cluster {
                Some(cl) => {
                    let (v, g) = sandwich(&inv_a, &scores, Some(cl), 1.0);
                    let g_f = g as f64;
                    let scale = (g_f / (g_f - 1.0)) * ((n_eff - 1.0) / (n_eff - rank));
                    (v.scale(scale), Some(g))
                }
                None => {
                    let (v, _) = sandwich(&inv_a, &scores, None, 1.0);
                    (v.scale(n_eff / (n_eff - rank)), None)
                }
            }
        }
    };
    Ok(OlsFit {
        v: expand_v(&vact, &act, k),
        b,
        rss,
        tss,
        df_r,
        df_m,
        resid,
        xtx_inv: expand_v(&inv_a, &act, k),
        n_clust,
    })
}

// ---------------------------------------------------------------------------
// Maximum likelihood
// ---------------------------------------------------------------------------

/// A model's log likelihood with analytic gradient and Hessian.
pub trait Likelihood {
    /// Number of parameters.
    fn k(&self) -> usize;
    /// (ll, gradient, negative Hessian (information), per-obs scores)
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>);
}

pub struct MlResult {
    pub b: Vec<f64>,
    pub ll: f64,
    pub info: Matrix,
    pub scores: Vec<Vec<f64>>,
    pub iterations: Vec<f64>,
    pub converged: bool,
}

/// Newton-Raphson with step halving.
pub fn maximize(model: &dyn Likelihood, b0: Vec<f64>, max_iter: usize) -> Result<MlResult, String> {
    let k = model.k();
    let mut b = b0;
    let (mut ll, mut g, mut h, _) = model.eval(&b, false);
    let mut log = vec![ll];
    let mut converged = false;
    for _ in 0..max_iter {
        let (hinv, _) = h.sym_inverse(1e-14);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum()).collect();
        let mut lambda = 1.0;
        let mut accepted = false;
        let mut nb = b.clone();
        let mut res = (ll, g.clone(), h.clone(), Vec::new());
        for _ in 0..40 {
            nb = b.iter().zip(&step).map(|(x, s)| x + lambda * s).collect();
            res = model.eval(&nb, false);
            if res.0.is_finite() && res.0 >= ll - 1e-12 * ll.abs().max(1.0) {
                accepted = true;
                break;
            }
            lambda /= 2.0;
        }
        if !accepted {
            break;
        }
        let change = (res.0 - ll).abs();
        let bchange = b.iter().zip(&nb).map(|(a, c)| (a - c).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        ll = res.0;
        g = res.1;
        h = res.2;
        log.push(ll);
        let gnorm: f64 = {
            let (hi, _) = h.sym_inverse(1e-14);
            (0..k).map(|i| (0..k).map(|j| g[i] * hi[(i, j)] * g[j]).sum::<f64>()).sum::<f64>()
        };
        if (change < 1e-10 * ll.abs().max(1.0) && bchange < 1e-8) || gnorm.abs() < 1e-14 {
            converged = true;
            break;
        }
    }
    let (ll, _, info, scores) = model.eval(&b, true);
    Ok(MlResult { b, ll, info, scores, iterations: log, converged })
}

pub fn iteration_log(it: &[f64], label: &str) -> Vec<String> {
    it.iter().enumerate().map(|(i, ll)| format!("Iteration {i}:  {label} = {}", fmt_g(*ll, 10))).collect()
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

pub struct TableSpec<'a> {
    pub depvar: &'a str,
    pub names: &'a [String],
    pub b: &'a [f64],
    pub v: &'a Matrix,
    pub omitted: &'a [bool],
    pub is_base: &'a [bool],
    /// t distribution with these df; None = normal
    pub df: Option<f64>,
    pub level: f64,
    pub eform: Option<&'a str>,
    pub coef_title: &'a str,
}

fn cell(v: f64) -> String {
    fmt_g(v, 8)
}

/// Stata-style coefficient table.
pub fn coef_table(s: &TableSpec) -> Vec<String> {
    let stat = if s.df.is_some() { "t" } else { "z" };
    // label column: at least 12, widened for long names (max 32)
    let mut lw = 12usize.max(s.depvar.chars().count().min(32));
    for (j, name) in s.names.iter().enumerate() {
        if s.is_base[j] {
            continue;
        }
        let nm = name.rsplit(':').next().unwrap_or(name);
        let (label, header) = display_name(nm);
        lw = lw.max(label.chars().count()).max(header.map_or(0, |h| h.chars().count()));
    }
    let lw = lw.min(32);
    let ab = |x: &str| crate::summarize::abbrev(x, lw);
    let mut out = Vec::new();
    let bar = "-".repeat(lw + 66);
    out.push(bar.clone());
    let title = s.eform.unwrap_or(s.coef_title);
    out.push(format!(
        "{:>lw$} | {:>11}  Std. err.  {:>6}    P>|{}|     [{}% conf. interval]",
        ab(s.depvar),
        title,
        stat,
        stat,
        fmt_g(s.level, 4)
    ));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
    let crit = match s.df {
        Some(df) => dist::invttail(df, (1.0 - s.level / 100.0) / 2.0),
        None => dist::invnormal(1.0 - (1.0 - s.level / 100.0) / 2.0),
    };
    let mut last_header = String::new();
    let mut last_eq = String::new();
    for (j, name) in s.names.iter().enumerate() {
        if s.is_base[j] {
            continue;
        }
        let (eq, nm) = match name.split_once(':') {
            Some((e, n)) => (e.to_string(), n.to_string()),
            None => (String::new(), name.clone()),
        };
        if !eq.is_empty() && eq != last_eq {
            if !last_eq.is_empty() {
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            }
            out.push(format!("{:<lw$} |", ab(&eq)));
            last_eq = eq.clone();
            last_header.clear();
        }
        let (label, header) = display_name(&nm);
        if let Some(h) = header {
            if h != last_header {
                let after_rule = out.last().map_or(true, |l| l.ends_with(&"-".repeat(10)) || l.trim_end().ends_with('|'));
                if !after_rule {
                    out.push(format!("{:>lw$} |", ""));
                }
                out.push(format!("{:>lw$} |", ab(&h)));
                last_header = h;
            }
        } else if !last_header.is_empty() {
            out.push(format!("{:>lw$} |", ""));
            last_header.clear();
        }
        let b = s.b[j];
        let se = s.v[(j, j)].max(0.0).sqrt();
        if s.omitted[j] {
            out.push(format!("{:>lw$} | {:>10}  (omitted)", ab(&label), if s.eform.is_some() { "1" } else { "0" }));
            continue;
        }
        let t = b / se;
        let p = match s.df {
            Some(df) => 2.0 * dist::ttail(df, t.abs()),
            None => 2.0 * (1.0 - dist::normal(t.abs())),
        };
        let (est, sev, lo, hi) = if s.eform.is_some() {
            (b.exp(), b.exp() * se, (b - crit * se).exp(), (b + crit * se).exp())
        } else {
            (b, se, b - crit * se, b + crit * se)
        };
        out.push(format!(
            "{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}",
            ab(&label),
            cell(est),
            cell(sev),
            t,
            p,
            cell(lo),
            cell(hi)
        ));
    }
    out.push(bar);
    out
}

/// Row label and optional group header for a coefficient name:
/// "2.g" -> ("2", "g"); "L.x" -> ("L1.", "x"); "2.g#c.x" -> ("2#c.x", "g#c.x").
fn display_name(n: &str) -> (String, Option<String>) {
    let clean = |s: &str| s.trim_end_matches(['b', 'o', 'n']).to_string();
    let n = n.strip_prefix("o.").unwrap_or(n);
    if n.contains('#') {
        let parts: Vec<&str> = n.split('#').collect();
        let mut header = Vec::new();
        let mut label = Vec::new();
        for p in &parts {
            match p.split_once('.') {
                Some((l, v)) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => {
                    header.push(v.to_string());
                    label.push(clean(l));
                }
                Some(("c", v)) => {
                    header.push(format!("c.{v}"));
                }
                _ => header.push(p.to_string()),
            }
        }
        if label.is_empty() {
            return (n.to_string(), None);
        }
        return (label.join("#"), Some(header.join("#")));
    }
    match n.split_once('.') {
        Some((l, v)) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => (clean(l), Some(v.to_string())),
        Some((op, v)) if op.chars().all(|c| "LFDSlfds0123456789".contains(c)) => {
            let up = op.to_uppercase();
            let label = if up.len() == 1 { format!("{up}1.") } else { format!("{up}.") };
            (label, Some(v.to_string()))
        }
        _ => (n.to_string(), None),
    }
}

/// Header block with right-aligned statistics.
pub fn header(title: &str, stats: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    lines.push(title.to_string());
    for (i, (k, v)) in stats.iter().enumerate() {
        let right = format!("{:<15} = {:>10}", k, v);
        if i == 0 {
            lines[0] = format!("{:<52}{}", title, right);
        } else {
            lines.push(format!("{:<52}{}", "", right));
        }
    }
    out.extend(lines);
    out
}

/// Store the standard e() results.
pub fn post(st: &mut Store, cmd: &str, d: &Design, b: &[f64], v: &Matrix, extra: &dyn Fn(&mut Results)) {
    let names = d.names.clone();
    let e = st.begin_e();
    e.set("N", d.n_eff());
    e.set("rank", d.active().len() as f64);
    e.set_str("cmd", cmd);
    e.set_str("depvar", d.depvar.clone());
    e.set_str("properties", "b V");
    e.set_mat("b", Matrix::row_vector(b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", v.clone().with_names(names.clone(), names));
    e.sample = Some(d.mask.clone());
    extra(e);
}

/// Model-level significance test (Wald) of all non-constant coefficients.
pub fn wald_all(d: &Design, b: &[f64], v: &Matrix) -> (f64, f64) {
    let idx: Vec<usize> = d.active().into_iter().filter(|&j| d.names[j] != "_cons" && !d.names[j].starts_with('/')).collect();
    let k = idx.len();
    if k == 0 {
        return (0.0, 0.0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-14);
    let bs: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
    let w: f64 = (0..k).map(|a| (0..k).map(|c| bs[a] * vi[(a, c)] * bs[c]).sum::<f64>()).sum();
    (w, (k - dropped.len()) as f64)
}

pub fn cluster_values(ds: &DataSet, d: &Design, vce: &Vce) -> Result<Option<Vec<f64>>, String> {
    match vce {
        Vce::Cluster(v) => {
            let name = crate::varlist::resolve_name(ds, v)?;
            let var = ds.get_var(&name).unwrap();
            Ok(Some(
                d.rows
                    .iter()
                    .map(|&i| match &var.data[i] {
                        Value::Str(s) => {
                            // hash strings to numbers for grouping
                            s.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211)) as f64
                        }
                        x => x.num(),
                    })
                    .collect(),
            ))
        }
        _ => Ok(None),
    }
}

pub fn vce_label(vce: &Vce) -> (&'static str, &'static str) {
    match vce {
        Vce::Robust => ("robust", "Robust"),
        Vce::Cluster(_) => ("cluster", "Robust"),
        Vce::Oim => ("oim", ""),
        Vce::Ols => ("ols", ""),
    }
}

pub fn missing_or(v: f64) -> f64 {
    if v.is_finite() { v } else { SYSMISS }
}

/// Use a previous fit's weights: helpers for models.
pub fn weights_of(d: &Design) -> Option<Weights> {
    d.wkind.map(|k| Weights { kind: k, w: d.w.clone(), expr: String::new() })
}
