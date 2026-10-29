//! Further estimators: the command list, the `predict` hook and shared
//! helpers (multi-equation maximum likelihood, coefficient tables with
//! ancillary parameters, linear constraints).
//!
//! The estimators live in `xm_lin` (interval, constrained, errors-in-variables,
//! heteroskedastic linear, systems of equations), `xm_bin` (binary,
//! fractional, count and ordinal variants), `xm_iv` (endogenous regressors),
//! `xm_sel` (selection, hurdle and frontier models), `xm_gmm` (generalized
//! method of moments), `xm_exact` (exact conditional inference), `xm_np`
//! (nonparametric regression) and `xm_roc` (ROC analysis and small tools).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::choice_mixed::{hrow, row_ci, row_z, sep, zcrit};
use crate::commands::CommandHandler;
use crate::est::{self, Design, MlResult, Vce};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::zip::{IndexFn, MultiIndex};

pub type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

/// Commands implemented in this family.
pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.extend(crate::xm_lin::handlers());
    v.extend(crate::xm_bin::handlers());
    v.extend(crate::xm_iv::handlers());
    v.extend(crate::xm_sel::handlers());
    v.extend(crate::xm_gmm::handlers());
    v.extend(crate::xm_exact::handlers());
    v.extend(crate::xm_np::handlers());
    v.extend(crate::xm_roc::handlers());
    v
}

/// `predict` after the commands of this family (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if let Some(r) = crate::xm_lin::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::xm_bin::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::xm_iv::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::xm_sel::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::xm_gmm::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::xm_np::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// command-line helpers
// ---------------------------------------------------------------------------

/// Split `dep rest` into the first word and the remainder.
pub fn first_word(main: &str) -> (String, String) {
    let t = main.trim();
    match t.split_once(char::is_whitespace) {
        Some((a, b)) => (a.to_string(), b.trim().to_string()),
        None => (t.to_string(), String::new()),
    }
}

/// Resolve a (possibly abbreviated) variable name.
pub fn var_name(ds: &DataSet, v: &str) -> Result<String, String> {
    crate::varlist::resolve_name(ds, v.trim())
}

/// Text inside the first level of parentheses: `(a) (b = c)` -> ["a", "b = c"].
pub fn paren_groups(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for ch in text.chars() {
        match ch {
            '(' => {
                if depth > 0 {
                    cur.push(ch);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            _ if depth > 0 => cur.push(ch),
            _ => {}
        }
    }
    out
}

/// Text outside parentheses.
pub fn outside_parens(text: &str) -> String {
    let mut depth = 0;
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// `het(varlist[, noconstant])`-style option: (varlist, suboptions).
pub fn opt_with_sub(cmd: &Command, name: &str, min: usize) -> Option<(String, String)> {
    cmd.opt(name, min).map(|s| match s.split_once(',') {
        Some((a, b)) => (a.trim().to_string(), b.trim().to_lowercase()),
        None => (s.trim().to_string(), String::new()),
    })
}

/// `(y2 = z1 z2)` groups of instrumental-variable syntax: returns the
/// text outside the group plus (endogenous, instruments).
pub fn iv_parts(main: &str) -> Result<(String, String, String), String> {
    let groups = paren_groups(main);
    let g = groups.iter().find(|g| g.contains('=')).ok_or_else(|| rc_err(198, "equation for endogenous regressors required: (varlist = varlist)"))?;
    let (a, b) = g.split_once('=').unwrap();
    Ok((outside_parens(main), a.trim().to_string(), b.trim().to_string()))
}

/// Default VCE for ML with weights: pweights imply robust.
pub fn ml_vce_opt(cmd: &Command, d: &Design) -> Result<Vce, String> {
    let v = est::parse_vce(cmd, Vce::Oim)?;
    Ok(if d.wkind == Some(WeightKind::P) && matches!(v, Vce::Oim | Vce::Ols) { Vce::Robust } else { v })
}

/// Keep only the sample rows flagged in `keep`.
pub fn subset(d: &Design, keep: &[bool]) -> Design {
    let mut o = d.clone();
    let pick = |v: &Vec<f64>| -> Vec<f64> { v.iter().zip(keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect() };
    o.cols = d.cols.iter().map(pick).collect();
    o.y = pick(&d.y);
    o.w = pick(&d.w);
    o.cluster = d.cluster.as_ref().map(pick);
    o.rows = d.rows.iter().zip(keep).filter(|(_, k)| **k).map(|(r, _)| *r).collect();
    for (r, k) in d.rows.iter().zip(keep) {
        if !k {
            o.mask[*r] = false;
        }
    }
    o
}

/// Design with only the constant active (for constant-only fits).
pub fn const_only(d: &Design) -> Option<Design> {
    let c = d.names.iter().position(|n| n == "_cons")?;
    let mut o = d.clone();
    for j in 0..o.k() {
        o.omitted[j] = j != c;
    }
    Some(o)
}

/// Build designs for several equations over a common estimation sample:
/// each equation's variables mark out the others.
pub fn build_eqs(ds: &DataSet, st: &Store, cmd: &Command, specs: &[(Option<&str>, &str, bool)], weights: &str, extra: &[String]) -> Result<Vec<Design>, String> {
    let mut all: Vec<String> = extra.to_vec();
    for (dep, x, _) in specs {
        if let Some(d) = dep {
            all.extend(crate::zip::base_vars(ds, d));
        }
        all.extend(crate::zip::base_vars(ds, x));
    }
    let mut out = Vec::new();
    for (dep, x, nocons) in specs {
        out.push(est::build(ds, st, cmd, est::DesignSpec { depvar: *dep, indep: x, noconstant: *nocons, weights, extra: all.clone() })?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// maximum likelihood over several linear indexes
// ---------------------------------------------------------------------------

/// A fitted multi-equation model, with coefficients over all design
/// columns (omitted ones fixed at zero) followed by ancillary parameters.
pub struct Fit {
    pub names: Vec<String>,
    pub b: Vec<f64>,
    pub v: Matrix,
    pub omitted: Vec<bool>,
    pub is_base: Vec<bool>,
    pub res: MlResult,
    /// parameters as estimated (active only) and their variance
    pub bt: Vec<f64>,
    pub vt: Matrix,
    /// full position of each estimated parameter
    pub pos: Vec<usize>,
    pub n_clust: Option<usize>,
    pub neq: usize,
    pub naux: usize,
}

impl Fit {
    pub fn ll(&self) -> f64 {
        self.res.ll
    }
    /// Index of a parameter in the full vector.
    pub fn idx(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
    pub fn coef(&self, name: &str) -> f64 {
        self.idx(name).map_or(SYSMISS, |j| self.b[j])
    }
    pub fn se(&self, name: &str) -> f64 {
        self.idx(name).map_or(SYSMISS, |j| self.v[(j, j)].max(0.0).sqrt())
    }
    /// Wald test that the non-constant coefficients of equation `e` are zero.
    pub fn wald_eq(&self, e: usize) -> (f64, f64) {
        let idx: Vec<usize> = (0..self.pos.len()).filter(|&a| self.eq_of(self.pos[a]) == Some(e) && !self.names[self.pos[a]].ends_with("_cons")).collect();
        crate::heckman::wald(&self.bt, &self.vt, &idx)
    }
/// Equation number of a full-vector parameter (None: ancillary).
pub fn eq_of(&self, j: usize) -> Option<usize> {
    if self.names[j].starts_with("/:") {
        return None;
    }
    let eq = self.names[j].split_once(':').map(|x| x.0);
    let mut seen: Vec<Option<&str>> = Vec::new();
    for n in &self.names {
        if n.starts_with("/:") {
            continue;
        }
        let e = n.split_once(':').map(|x| x.0);
        if !seen.contains(&e) {
            seen.push(e);
        }
    }
    seen.iter().position(|x| *x == eq)
}
}

/// Fit by maximum likelihood. `eqs` are (equation name, design) pairs (an
/// empty name keeps plain coefficient names); `b0` gives starting values
/// over the active columns of each equation followed by the ancillary
/// parameters.
#[allow(clippy::too_many_arguments)]
pub fn ml_fit(eqs: &[(&str, &Design)], aux: &[&str], f: &IndexFn, b0: Vec<f64>, w: &[f64], fweight: bool, vce: &Vce, cluster: Option<&[f64]>) -> Result<Fit, String> {
    let acts: Vec<Vec<usize>> = eqs.iter().map(|(_, d)| d.active()).collect();
    let x: Vec<Vec<&[f64]>> = eqs.iter().zip(&acts).map(|((_, d), a)| a.iter().map(|&j| d.cols[j].as_slice()).collect()).collect();
    let m = MultiIndex { x, naux: aux.len(), w, fweight, f };
    let res = maximize(&m, b0, 300)?;
    finish_fit(eqs, aux, res, vce, cluster)
}

/// Newton-Raphson with step halving and a ridge for non-concave regions;
/// stops when the parameters settle, or when the log likelihood stops
/// changing for several iterations (flat directions).
pub fn maximize(model: &dyn crate::est::Likelihood, b0: Vec<f64>, max_iter: usize) -> Result<MlResult, String> {
    let k = model.k();
    let mut b = b0;
    let (mut ll, mut g, mut h, _) = model.eval(&b, false);
    if !ll.is_finite() {
        return Err(rc_err(430, "initial values not feasible"));
    }
    let mut log = vec![ll];
    let mut converged = false;
    let mut flat = 0;
    for _ in 0..max_iter {
        let scale: f64 = (0..k).map(|i| h[(i, i)].abs()).fold(0.0, f64::max).max(1e-8);
        let mut ridge = 0.0;
        let step = loop {
            let mut hr = h.clone();
            for i in 0..k {
                hr[(i, i)] += ridge;
            }
            if hr.cholesky().is_ok() || ridge > 1e12 * scale {
                let (hinv, _) = hr.sym_inverse(1e-14);
                break (0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum::<f64>()).collect::<Vec<f64>>();
            }
            ridge = if ridge == 0.0 { 1e-6 * scale } else { ridge * 10.0 };
        };
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..50 {
            let nb: Vec<f64> = b.iter().zip(&step).map(|(x, s)| x + lambda * s).collect();
            let r = model.eval(&nb, false);
            if r.0.is_finite() && r.0 >= ll - 1e-12 * ll.abs().max(1.0) {
                accepted = Some((nb, r));
                break;
            }
            lambda /= 2.0;
        }
        let Some((nb, r)) = accepted else {
            converged = h.cholesky().is_ok();
            break;
        };
        let change = (r.0 - ll).abs();
        let bchange = b.iter().zip(&nb).map(|(a, c)| (a - c).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        ll = r.0;
        g = r.1;
        h = r.2;
        log.push(ll);
        let pd = h.cholesky().is_ok();
        if pd && change < 1e-10 * ll.abs().max(1.0) && bchange < 1e-7 {
            converged = true;
            break;
        }
        // scaled gradient criterion
        if pd {
            let (hi, _) = h.sym_inverse(1e-14);
            let dec: f64 = (0..k).map(|i| (0..k).map(|j| g[i] * hi[(i, j)] * g[j]).sum::<f64>()).sum();
            if dec.abs() < 1e-14 {
                converged = true;
                break;
            }
        }
        flat = if change < 1e-9 * ll.abs().max(1.0) { flat + 1 } else { 0 };
        if flat >= 8 {
            converged = false;
            break;
        }
    }
    let (ll, _, info, scores) = model.eval(&b, true);
    Ok(MlResult { b, ll, info, scores, iterations: log, converged })
}

/// Assemble a `Fit` from an ML result over the active columns.
pub fn finish_fit(eqs: &[(&str, &Design)], aux: &[&str], res: MlResult, vce: &Vce, cluster: Option<&[f64]>) -> Result<Fit, String> {
    let n = res.scores.len().max(1) as f64;
    let (vt, n_clust) = crate::zip::ml_vce(&res, vce, cluster, n);
    let mut names = Vec::new();
    let mut omitted = Vec::new();
    let mut is_base = Vec::new();
    let mut pos = Vec::new();
    for (e, d) in eqs {
        for j in 0..d.k() {
            if !d.omitted[j] {
                pos.push(names.len());
            }
            names.push(if e.is_empty() { d.names[j].clone() } else { format!("{e}:{}", d.names[j]) });
            omitted.push(d.omitted[j]);
            is_base.push(d.is_base[j]);
        }
    }
    for a in aux {
        pos.push(names.len());
        names.push(format!("/:{a}"));
        omitted.push(false);
        is_base.push(false);
    }
    let k = names.len();
    let mut b = vec![0.0; k];
    for (a, &j) in pos.iter().enumerate() {
        b[j] = res.b[a];
    }
    let v = est::expand_v(&vt, &pos, k);
    Ok(Fit { names, b, v, omitted, is_base, bt: res.b.clone(), vt, pos, n_clust, neq: eqs.len(), naux: aux.len(), res })
}

/// Per-observation index function from a log-likelihood closure, with
/// numerical derivatives.
pub fn numeric<'a>(l: &'a (dyn Fn(usize, &[f64]) -> f64 + 'a)) -> impl Fn(usize, &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) + 'a {
    move |t: usize, x: &[f64]| crate::zip::num_derivs(&|v: &[f64]| l(t, v), x)
}

/// Per-observation index function from an analytic gradient closure
/// (Hessian by differencing the gradient).
pub fn with_grad<'a>(l: &'a (dyn Fn(usize, &[f64]) -> (f64, Vec<f64>) + 'a)) -> impl Fn(usize, &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) + 'a {
    move |t: usize, x: &[f64]| {
        let (v, g) = l(t, x);
        (v, g, crate::choice_mixed::num_hess(&|u: &[f64]| l(t, u).1, x))
    }
}

/// Least-squares coefficients of `y` on the active columns of `d`.
pub fn ols_start(d: &Design, y: &[f64]) -> (Vec<f64>, f64) {
    let mut d2 = d.clone();
    d2.y = y.to_vec();
    match est::ols(&d2, &Vce::Ols, None) {
        Ok(f) => {
            let act = d.active();
            let s = (f.rss / d.n().max(1) as f64).sqrt();
            (act.iter().map(|&j| f.b[j]).collect(), s)
        }
        Err(_) => (vec![0.0; d.active().len()], 1.0),
    }
}

// ---------------------------------------------------------------------------
// output
// ---------------------------------------------------------------------------

/// Extra rows below a coefficient table.
pub enum Row {
    /// label, estimate, standard error: with z test
    Z(String, f64, f64),
    /// label, estimate, standard error, interval
    Ci(String, f64, f64, f64, f64),
    /// transformed parameter: label, estimate, std. err. on the original
    /// scale and its interval from the untransformed (b, se) by `g`
    Tr(String, f64, f64, f64, f64, fn(f64) -> f64),
    Sep,
}

/// Coefficient table of a fit: equations, then ancillary parameters (as
/// `/name` rows with tests) and `extra` rows.
pub fn table(depvar: &str, fit: &Fit, level: f64, eform: Option<&str>, extra: &[Row]) -> Vec<String> {
    let ncoef = fit.names.iter().filter(|n| !n.starts_with("/:")).count();
    let mut tbl = est::coef_table(&est::TableSpec {
        depvar,
        names: &fit.names[..ncoef],
        b: &fit.b[..ncoef],
        v: &fit.v,
        omitted: &fit.omitted[..ncoef],
        is_base: &fit.is_base[..ncoef],
        df: None,
        level,
        eform,
        coef_title: "Coefficient",
    });
    // store r(table) with every parameter
    let mut full = est::table_matrix(&est::TableSpec { depvar, names: &fit.names, b: &fit.b, v: &fit.v, omitted: &fit.omitted, is_base: &fit.is_base, df: None, level, eform: None, coef_title: "Coefficient" });
    for (j, n) in fit.names.iter().enumerate() {
        if n.starts_with("/:") && full.rows > 3 {
            let se = fit.v[(j, j)].max(0.0).sqrt();
            let z = fit.b[j] / se;
            full[(2, j)] = z;
            full[(3, j)] = 2.0 * crate::dist::normal(-z.abs());
        }
    }
    est::stash_table(full, level);
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    let crit = zcrit(level);
    let auxes: Vec<usize> = (ncoef..fit.names.len()).collect();
    if !auxes.is_empty() || !extra.is_empty() {
        tbl.push(sep(lw));
    }
    for &j in &auxes {
        tbl.push(row_z(lw, &format!("/{}", &fit.names[j][2..]), fit.b[j], fit.v[(j, j)].max(0.0).sqrt(), crit));
    }
    tbl.extend(extra_rows(lw, level, extra, !auxes.is_empty()));
    tbl.push(bar);
    tbl
}

/// Render extra rows (`Row`) at label width `lw`.
pub fn extra_rows(lw: usize, level: f64, extra: &[Row], after_aux: bool) -> Vec<String> {
    let crit = zcrit(level);
    let mut out = Vec::new();
    let mut first = true;
    for r in extra {
        match r {
            Row::Z(l, b, se) => out.push(row_z(lw, l, *b, *se, crit)),
            Row::Ci(l, b, se, lo, hi) => {
                if first && after_aux {
                    out.push(sep(lw));
                }
                out.push(row_ci(lw, l, *b, *se, *lo, *hi))
            }
            Row::Tr(l, b, se, b0, se0, g) => {
                if first && after_aux {
                    out.push(sep(lw));
                }
                out.push(row_ci(lw, l, *b, *se, g(b0 - crit * se0), g(b0 + crit * se0)))
            }
            Row::Sep => out.push(sep(lw)),
        }
        first = false;
    }
    out
}

/// Header lines: title with N, optional extra lines, then the model test
/// and log likelihood.
pub struct Header<'a> {
    pub title: &'a str,
    pub n: f64,
    pub lines: Vec<(String, String, String)>,
    pub test: Option<(String, f64, f64)>,
    pub ll: Option<f64>,
    pub robust: bool,
    pub n_clust: Option<(usize, String)>,
}

impl Header<'_> {
    pub fn render(&self) -> Vec<String> {
        let mut out = vec![hrow(self.title, "Number of obs", util::commas(self.n.round() as usize))];
        for (l, k, v) in &self.lines {
            out.push(hrow(l, k, v.clone()));
        }
        let (tk, tv, pv) = match &self.test {
            Some((name, chi2, df)) => {
                let p = chi2p(*df, *chi2);
                (format!("{name} chi2({})", fmt_g(*df, 9)), f2(*chi2), f4(p))
            }
            None => (String::new(), String::new(), String::new()),
        };
        let lltxt = self.ll.map(|ll| format!("Log {}likelihood = {}", if self.robust { "pseudo" } else { "" }, fmt_g(ll, 9))).unwrap_or_default();
        if self.test.is_some() {
            out.push(hrow("", &tk, tv));
            out.push(hrow(&lltxt, "Prob > chi2", pv));
        } else if !lltxt.is_empty() {
            out.push(lltxt);
        }
        if let Some((g, var)) = &self.n_clust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {var})", util::commas(*g))));
        }
        out.push(String::new());
        out
    }
}

pub fn f2(v: f64) -> String {
    crate::choice_mixed::f2(v)
}
pub fn f4(v: f64) -> String {
    crate::choice_mixed::f4(v)
}

/// Store a fit in e(): b, V, sample and the usual ML scalars.
pub fn post_fit(st: &mut Store, verb: &str, d: &Design, fit: &Fit, vce: &Vce, extra: &dyn Fn(&mut Results)) {
    let mut dp = d.clone();
    dp.names = fit.names.clone();
    dp.omitted = fit.omitted.clone();
    dp.is_base = fit.is_base.clone();
    let res = &fit.res;
    let n_clust = fit.n_clust;
    let (neq, naux, np) = (fit.neq, fit.naux, fit.pos.len());
    est::post(st, verb, &dp, &fit.b, &fit.v, &|e: &mut Results| {
        e.set("ll", res.ll);
        e.set("k", np as f64);
        e.set("rank", np as f64);
        e.set("k_eq", (neq + naux) as f64);
        e.set("k_aux", naux as f64);
        e.set("ic", res.iterations.len().saturating_sub(1) as f64);
        e.set("converged", if res.converged { 1.0 } else { 0.0 });
        let (v, t) = est::vce_label(vce);
        e.set_str("vce", v);
        if !t.is_empty() {
            e.set_str("vcetype", t);
        }
        if let (Some(g), Vce::Cluster(c)) = (n_clust, vce) {
            e.set("N_clust", g as f64);
            e.set_str("clustvar", c.clone());
        }
        e.set_str("opt", "moptimize");
        e.set_str("ml_method", "lf2");
        extra(e);
    });
}

/// Store model-test scalars.
pub fn set_test(e: &mut Results, kind: &str, chi2: f64, df: f64) {
    e.set("chi2", chi2);
    e.set("df_m", df);
    e.set("p", chi2p(df, chi2));
    e.set_str("chi2type", kind);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// Context passed to a model's prediction function for one observation.
pub struct PredCtx<'a> {
    /// linear indexes of the equations (in order of e(b))
    pub eta: Vec<f64>,
    /// ancillary parameter by name (`/name`), SYSMISS if absent
    pub aux: &'a dyn Fn(&str) -> f64,
    /// dependent variable value(s) for the observation
    pub y: Vec<f64>,
    pub row: usize,
}

/// Equations named in e(b), in order (`None` for plain names).
pub fn eq_names(b: &Matrix) -> Vec<Option<String>> {
    let mut v: Vec<Option<String>> = Vec::new();
    for c in &b.colnames {
        if c.starts_with("/:") || c.starts_with('/') {
            continue;
        }
        let e = c.split_once(':').map(|x| x.0.to_string());
        if !v.contains(&e) {
            v.push(e);
        }
    }
    v
}

/// Generic `predict newvar [, stat]`: `stats` lists the statistics the
/// model accepts (the first is the default, announced by `default_msg`);
/// `f` computes the statistic for one observation.
pub fn generic_predict(ds: &mut DataSet, st: &Store, cmd: &Command, stats: &[&str], default_msg: &str, f: &dyn Fn(&str, &PredCtx) -> f64) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mut stat = None;
    for (o, _) in &cmd.opts {
        let o = o.to_lowercase();
        if let Some(s) = stats.iter().find(|s| **s == o || (o.len() >= 2 && s.starts_with(o.as_str()))) {
            stat = Some(s.to_string());
        } else if !matches!(o.as_str(), "nooffset" | "eq" | "equation" | "outcome") {
            return Err(rc_err(198, format!("option {o} not allowed")));
        }
    }
    let (stat, msg) = match stat {
        Some(s) => (s, String::new()),
        None => (stats[0].to_string(), default_msg.to_string()),
    };
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let eqs = eq_names(&b);
    let mut deps: Vec<String> = st.e.macros.get("depvar").map(|d| d.split_whitespace().map(String::from).collect()).unwrap_or_default();
    deps.retain(|d| ds.var_index.contains_key(d));
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let env = Env::with_store(ds, st);
    let auxf = |nm: &str| b.col_index(&format!("/:{nm}")).or_else(|| b.col_index(&format!("{nm}:_cons"))).map_or(SYSMISS, |j| b.get(0, j));
    let ycols: Vec<Vec<f64>> = deps.iter().map(|d| util::col(ds, d)).collect::<Result<_, _>>()?;
    let mut vals = vec![SYSMISS; n];
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        if stat == "stdp" {
            let eq = eqs.first().cloned().flatten();
            let x: Vec<f64> = b
                .colnames
                .iter()
                .map(|c| if c.split_once(':').map(|x| x.0.to_string()) == eq && !c.starts_with('/') { crate::postest::coef_value(&env, c, i) } else { Ok(0.0) })
                .collect::<Result<_, _>>()?;
            if x.iter().any(|v| is_missing(*v)) {
                continue;
            }
            let k = x.len();
            vals[i] = (0..k).map(|a| (0..k).map(|c| x[a] * v[(a, c)] * x[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt();
            continue;
        }
        let mut eta = Vec::with_capacity(eqs.len());
        let mut miss = false;
        for e in &eqs {
            let x = crate::postest::xb_row(&env, &b, e.as_deref(), i)?;
            if is_missing(x) {
                miss = true;
            }
            eta.push(x);
        }
        if miss && !matches!(stat.as_str(), "xb") {
            continue;
        }
        if miss && is_missing(eta[0]) {
            continue;
        }
        let ctx = PredCtx { eta, aux: &auxf, y: ycols.iter().map(|c| c[i]).collect(), row: i };
        let r = f(&stat, &ctx);
        vals[i] = if r.is_finite() { r } else { SYSMISS };
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// linear constraints
// ---------------------------------------------------------------------------

/// Parse a linear constraint `lhs = rhs` over coefficient `names` into
/// (coefficients r, constant c) with r'b = c.
pub fn parse_constraint(spec: &str, names: &[String]) -> Result<(Vec<f64>, f64), String> {
    let (l, r) = spec.split_once('=').ok_or_else(|| rc_err(198, format!("constraint {spec}: = required")))?;
    let mut coef = vec![0.0; names.len()];
    let mut cons = 0.0;
    for (side, sign) in [(l, 1.0), (r, -1.0)] {
        for (s, term) in linear_terms(side)? {
            let s = s * sign;
            match term {
                None => cons -= s,
                Some(t) => {
                    let j = find_coef(&t, names).ok_or_else(|| rc_err(111, format!("{t} not found")))?;
                    coef[j] += s;
                }
            }
        }
    }
    Ok((coef, cons))
}

/// Coefficient lookup for constraint terms: `x`, `[eq]x`, `eq:x`, `_b[x]`, `/name`.
pub fn find_coef(t: &str, names: &[String]) -> Option<usize> {
    let t = t.trim();
    let t = t.strip_prefix("_b").unwrap_or(t);
    let t = if let Some(r) = t.strip_prefix('[') {
        let (e, rest) = r.split_once(']')?;
        let rest = rest.trim();
        if rest.is_empty() {
            e.to_string()
        } else {
            format!("{e}:{rest}")
        }
    } else {
        t.to_string()
    };
    let t = t.trim().to_string();
    if let Some(j) = names.iter().position(|n| *n == t) {
        return Some(j);
    }
    if let Some(a) = t.strip_prefix('/') {
        if let Some(j) = names.iter().position(|n| *n == format!("/:{a}") || *n == format!("{a}:_cons")) {
            return Some(j);
        }
    }
    // plain name: first equation containing it
    names.iter().position(|n| n.rsplit(':').next() == Some(t.as_str()) && !n.starts_with("/:"))
}

/// Split a linear expression into signed terms: (multiplier, name or None for a constant).
fn linear_terms(s: &str) -> Result<Vec<(f64, Option<String>)>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    let mut sign = 1.0;
    let flush = |cur: &mut String, sign: f64, out: &mut Vec<(f64, Option<String>)>| -> Result<(), String> {
        let t = cur.trim().to_string();
        cur.clear();
        if t.is_empty() {
            return Ok(());
        }
        let mut mult = sign;
        let mut name: Option<String> = None;
        for f in t.split('*') {
            let f = f.trim();
            if let Ok(v) = f.parse::<f64>() {
                mult *= v;
            } else if name.is_none() {
                name = Some(f.to_string());
            } else {
                return Err(rc_err(198, "nonlinear constraint"));
            }
        }
        out.push((mult, name));
        Ok(())
    };
    for ch in s.chars() {
        match ch {
            '[' | '(' => {
                depth += 1;
                cur.push(ch)
            }
            ']' | ')' => {
                depth -= 1;
                cur.push(ch)
            }
            '+' | '-' if depth == 0 && !(cur.trim_end().ends_with(['e', 'E']) && cur.trim().chars().next().is_some_and(|c| c.is_ascii_digit())) => {
                if cur.trim().is_empty() {
                    if ch == '-' {
                        sign = -sign;
                    }
                    continue;
                }
                flush(&mut cur, sign, &mut out)?;
                sign = if ch == '-' { -1.0 } else { 1.0 };
            }
            _ => cur.push(ch),
        }
    }
    flush(&mut cur, sign, &mut out)?;
    Ok(out)
}

/// The constraints named by `constraints(numlist)`: (number, spec).
pub fn constraint_specs(st: &Store, cmd: &Command) -> Result<Vec<(usize, String)>, String> {
    let Some(list) = cmd.opt("constraints", 1) else { return Ok(vec![]) };
    let mut out = Vec::new();
    for tok in list.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let nums: Vec<usize> = match tok.split_once(['/', '-']) {
            Some((a, b)) => {
                let a: usize = a.parse().map_err(|_| rc_err(198, "invalid numlist"))?;
                let b: usize = b.parse().map_err(|_| rc_err(198, "invalid numlist"))?;
                (a..=b).collect()
            }
            None => vec![tok.parse().map_err(|_| rc_err(198, "invalid numlist"))?],
        };
        for n in nums {
            let spec = st.settings.get(&format!("_constraint.{n}")).ok_or_else(|| rc_err(111, format!("constraint {n} not found")))?;
            out.push((n, spec.clone()));
        }
    }
    Ok(out)
}

/// Display form of a constraint row: ` ( 1)  x1 - x2 = 0`.
pub fn constraint_text(k: usize, r: &[f64], c: f64, names: &[String]) -> String {
    let mut s = String::new();
    for (j, &a) in r.iter().enumerate() {
        if a == 0.0 {
            continue;
        }
        let nm = names[j].strip_prefix("/:").map(|x| format!("/{x}")).unwrap_or_else(|| {
            match names[j].split_once(':') {
                Some((e, n)) => format!("[{e}]{n}"),
                None => names[j].clone(),
            }
        });
        let mag = a.abs();
        let term = if (mag - 1.0).abs() < 1e-15 { nm } else { format!("{}*{nm}", fmt_g(mag, 8)) };
        if s.is_empty() {
            s = if a < 0.0 { format!("- {term}") } else { term };
        } else {
            s.push_str(if a < 0.0 { " - " } else { " + " });
            s.push_str(&term);
        }
    }
    format!(" ({k:>2})  {s} = {}", fmt_g(c, 8))
}

/// Constraint matrices R (q x k) and r from `constraints()`.
pub fn constraint_matrix(st: &Store, cmd: &Command, names: &[String]) -> Result<(Matrix, Vec<f64>, Vec<String>), String> {
    let specs = constraint_specs(st, cmd)?;
    let k = names.len();
    let mut rm = Matrix::zeros(specs.len(), k);
    let mut rv = Vec::new();
    let mut text = Vec::new();
    for (i, (_, s)) in specs.iter().enumerate() {
        let (r, c) = parse_constraint(s, names)?;
        for j in 0..k {
            rm[(i, j)] = r[j];
        }
        rv.push(c);
        text.push(constraint_text(i + 1, &r, c, names));
    }
    Ok((rm, rv, text))
}

/// Null space parameterisation of R b = r: b = T g + b0 (T is k x (k - q)).
pub fn constraint_basis(rm: &Matrix, rv: &[f64]) -> Result<(Matrix, Vec<f64>), String> {
    let (q, k) = (rm.rows, rm.cols);
    // particular solution b0 = R'(RR')^-1 r
    let rrt = rm.mul(&rm.transpose())?;
    let (inv, dropped) = rrt.sym_inverse(1e-12);
    if !dropped.is_empty() {
        return Err(rc_err(412, "redundant or inconsistent constraints"));
    }
    let tmp: Vec<f64> = (0..q).map(|a| (0..q).map(|c| inv[(a, c)] * rv[c]).sum()).collect();
    let b0: Vec<f64> = (0..k).map(|j| (0..q).map(|a| rm[(a, j)] * tmp[a]).sum()).collect();
    // basis of the null space from the eigenvectors of R'R with zero eigenvalues
    let rtr = rm.transpose().mul(rm)?;
    let (vals, vecs) = rtr.sym_eigen();
    let mx = vals.iter().cloned().fold(0.0, f64::max).max(1e-300);
    let free: Vec<usize> = (0..k).filter(|&i| vals[i].abs() < 1e-10 * mx).collect();
    let mut t = Matrix::zeros(k, free.len());
    for (c, &i) in free.iter().enumerate() {
        for j in 0..k {
            t[(j, c)] = vecs[(j, i)];
        }
    }
    Ok((t, b0))
}

/// Standard normal CDF that also accepts infinite arguments.
pub fn ncdf(z: f64) -> f64 {
    if z == f64::INFINITY {
        1.0
    } else if z == f64::NEG_INFINITY {
        0.0
    } else {
        crate::dist::normal(z)
    }
}

/// Chi-squared upper tail that tolerates missing or non-finite statistics.
pub fn chi2p(df: f64, x: f64) -> f64 {
    if !x.is_finite() || !df.is_finite() || is_missing(x) {
        SYSMISS
    } else {
        crate::dist::chi2tail(df, x)
    }
}
