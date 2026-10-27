//! `reghdfe`: linear regression absorbing several high-dimensional fixed
//! effects (and `ivreghdfe`, its two-stage least squares variant).
//!
//! The fixed effects are partialled out of the dependent variable and the
//! regressors (see `hdfe`), then OLS (or 2SLS) runs on the residualised
//! data. Degrees of freedom absorbed by the fixed effects are counted
//! exactly for the first two sets (connected components of their bipartite
//! graph), conservatively for further sets; fixed effects nested within a
//! cluster variable do not use degrees of freedom.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec};
use crate::eval::rc_err;
use crate::hdfe::{self, Factor};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// One absorbed term as written (`name=i.a#i.b`, `a##c.x`).
#[derive(Clone, Debug)]
struct AbsSpec {
    save: Option<String>,
    ivars: Vec<String>,
    cvars: Vec<String>,
    intercept: bool,
}

fn parse_absvars(ds: &DataSet, text: &str) -> Result<Vec<AbsSpec>, String> {
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        let (save, body) = match tok.split_once('=') {
            Some((n, b)) => {
                crate::varlist::check_name(n)?;
                (Some(n.to_string()), b)
            }
            None => (None, tok),
        };
        let both = body.contains("##");
        let parts: Vec<&str> = if both { body.split("##").collect() } else { body.split('#').collect() };
        let mut ivars = Vec::new();
        let mut cvars = Vec::new();
        for p in parts {
            let lo = p.to_lowercase();
            if let Some(v) = lo.strip_prefix("c.").map(|_| &p[2..]) {
                cvars.push(crate::varlist::resolve_name(ds, v)?);
            } else {
                let v = if lo.starts_with("i.") { &p[2..] } else { p };
                if v.starts_with('(') || v.is_empty() {
                    return Err(rc_err(198, format!("{tok}: invalid absorb() term")));
                }
                ivars.push(crate::varlist::resolve_name(ds, v)?);
            }
        }
        if ivars.is_empty() {
            return Err(rc_err(198, format!("{tok}: absorb() terms need a categorical variable")));
        }
        out.push(AbsSpec { save, intercept: cvars.is_empty() || both, ivars, cvars });
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
enum VceKind {
    Unadjusted,
    Robust,
    Cluster(Vec<String>),
}

fn parse_vce(ds: &DataSet, cmd: &Command) -> Result<VceKind, String> {
    let cl = |list: &str| -> Result<VceKind, String> {
        let mut vars = Vec::new();
        for v in list.split_whitespace() {
            // a#b clusters on the interaction
            let parts: Result<Vec<String>, String> = v.split('#').map(|p| crate::varlist::resolve_name(ds, p.trim_start_matches("i."))).collect();
            vars.push(parts?.join("#"));
        }
        if vars.is_empty() {
            return Err(rc_err(198, "vce(cluster clustvar) requires a variable"));
        }
        Ok(VceKind::Cluster(vars))
    };
    if let Some(v) = cmd.opt("vce", 3) {
        let v = v.trim();
        let (first, rest) = v.split_once(char::is_whitespace).unwrap_or((v, ""));
        return match first.to_lowercase().as_str() {
            "robust" | "r" | "hc1" => Ok(VceKind::Robust),
            "unadjusted" | "ols" | "un" | "unadj" => Ok(VceKind::Unadjusted),
            "cluster" | "cl" | "clu" | "clus" => cl(rest),
            o => Err(rc_err(198, format!("vce({o}) not allowed"))),
        };
    }
    if let Some(c) = cmd.opt("cluster", 2) {
        return cl(c);
    }
    if cmd.has_opt("robust", 1) {
        return Ok(VceKind::Robust);
    }
    Ok(VceKind::Unadjusted)
}

/// Numeric or string column over the given rows as group codes.
fn var_codes(ds: &DataSet, name: &str, rows: &[usize]) -> Result<(Vec<u32>, usize), String> {
    let var = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if var.var_type.is_string() {
        let keys: Vec<&str> = rows.iter().map(|&i| var.str_at(i)).collect();
        Ok(hdfe::codes_str(&keys))
    } else {
        let keys: Vec<f64> = rows.iter().map(|&i| var.data[i].num()).collect();
        Ok(hdfe::codes_f64(&keys))
    }
}

/// Codes of the interaction of several variables (`a#b`).
fn multi_codes(ds: &DataSet, names: &[String], rows: &[usize]) -> Result<(Vec<u32>, usize), String> {
    let (mut c, mut g) = var_codes(ds, &names[0], rows)?;
    for n in &names[1..] {
        let (c2, g2) = var_codes(ds, n, rows)?;
        let r = hdfe::combine(&c, g, &c2, g2);
        c = r.0;
        g = r.1;
    }
    Ok((c, g))
}

fn num_col(ds: &DataSet, name: &str, rows: &[usize]) -> Result<Vec<f64>, String> {
    let var = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if var.var_type.is_string() {
        return Err(rc_err(109, format!("{name}: string variables not allowed")));
    }
    Ok(rows.iter().map(|&i| var.data[i].num()).collect())
}

/// Keep the flagged observations of a design.
fn subset_design(d: &mut Design, keep: &[bool]) {
    let pick = |v: &Vec<f64>| -> Vec<f64> { v.iter().zip(keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect() };
    for (t, &k) in keep.iter().enumerate() {
        if !k {
            d.mask[d.rows[t]] = false;
        }
    }
    d.rows = d.rows.iter().zip(keep).filter(|(_, k)| **k).map(|(r, _)| *r).collect();
    d.y = pick(&d.y);
    d.w = pick(&d.w);
    for c in d.cols.iter_mut() {
        *c = pick(c);
    }
    // analytic and sampling weights stay normalised to the sample size
    if matches!(d.wkind, Some(WeightKind::A) | Some(WeightKind::P)) {
        let s: f64 = d.w.iter().sum();
        let n = d.rows.len() as f64;
        for w in d.w.iter_mut() {
            *w *= n / s;
        }
    }
}

fn omit_name(n: &str) -> String {
    match n.split_once('.') {
        Some((a, b)) if a.chars().all(|c| c.is_ascii_digit()) => format!("{a}o.{b}"),
        _ => format!("o.{n}"),
    }
}

fn wdot(a: &[f64], b: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        Some(w) => a.iter().zip(b).zip(w).map(|((x, y), wi)| x * y * wi).sum(),
        None => a.iter().zip(b).map(|(x, y)| x * y).sum(),
    }
}

fn wmean(a: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        Some(w) => a.iter().zip(w).map(|(x, wi)| x * wi).sum::<f64>() / w.iter().sum::<f64>(),
        None => a.iter().sum::<f64>() / a.len() as f64,
    }
}

// ---------------------------------------------------------------------------
// Shared estimation pieces
// ---------------------------------------------------------------------------

/// Common setup: sample, fixed effects, clusters.
struct Setup {
    d: Design,
    factors: Vec<Factor>,
    specs: Vec<AbsSpec>,
    vce: VceKind,
    clusters: Vec<(String, Vec<u32>, usize)>,
    singletons: usize,
    noabsorb: bool,
    tol: f64,
    maxiter: usize,
    w: Option<Vec<f64>>,
}

fn setup(ds: &DataSet, st: &Store, cmd: &Command, dep: &str, indep: &str, extra_vars: &[String]) -> Result<Setup, String> {
    let noabsorb = cmd.has_opt("noabsorb", 3);
    let absorb = cmd.opt("absorb", 1).map(str::to_string);
    if absorb.is_none() && !noabsorb {
        return Err(rc_err(198, "option absorb() required (or noabsorb)"));
    }
    let (abs_text, abs_opts) = match &absorb {
        Some(a) => match a.split_once(',') {
            Some((l, r)) => (l.trim().to_string(), r.trim().to_lowercase()),
            None => (a.trim().to_string(), String::new()),
        },
        None => (String::new(), String::new()),
    };
    let savefe = abs_opts.split_whitespace().any(|o| o.starts_with("savefe"));
    let mut specs = parse_absvars(ds, &abs_text)?;
    if specs.is_empty() && !noabsorb {
        return Err(rc_err(198, "absorb() requires at least one variable"));
    }
    if savefe {
        for (k, s) in specs.iter_mut().enumerate() {
            if s.save.is_none() {
                s.save = Some(format!("__hdfe{}__", k + 1));
            }
        }
    }
    let vce = parse_vce(ds, cmd)?;
    let mut extra: Vec<String> = extra_vars.to_vec();
    for s in &specs {
        extra.extend(s.ivars.iter().cloned());
        extra.extend(s.cvars.iter().cloned());
    }
    if let VceKind::Cluster(cs) = &vce {
        for c in cs {
            extra.extend(c.split('#').map(str::to_string));
        }
    }
    let tol = util::opt_num(cmd, "tolerance", 3)?.unwrap_or(1e-8);
    let maxiter = util::opt_num(cmd, "maxiterations", 4)?.map_or(16_000, |m| m.max(1.0) as usize);
    let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(dep), indep, noconstant: false, weights: "fap", extra })?;
    // fixed effects over the sample
    let mut factors = Vec::new();
    for s in &specs {
        let (ids, ng) = multi_codes(ds, &s.ivars, &d.rows)?;
        let slopes = s.cvars.iter().map(|c| num_col(ds, c, &d.rows)).collect::<Result<Vec<_>, _>>()?;
        let mut f = Factor::new(s.ivars.join("#"), ids, ng, s.intercept, slopes, s.cvars.clone());
        f.save_as = s.save.clone();
        factors.push(f);
    }
    if noabsorb {
        factors.push(Factor::new("_cons".into(), vec![0; d.n()], 1, true, vec![], vec![]));
    }
    // singletons
    let mut singletons = 0;
    if !cmd.has_opt("keepsingletons", 5) && !noabsorb {
        let (keep, dropped) = hdfe::singletons(&factors, d.n());
        if dropped > 0 {
            if dropped == d.n() {
                return Err(rc_err(2001, "insufficient observations (all observations are singletons)"));
            }
            subset_design(&mut d, &keep);
            for f in factors.iter_mut() {
                f.subset(&keep);
            }
            singletons = dropped;
        }
    }
    let clusters = match &vce {
        VceKind::Cluster(cs) => cs
            .iter()
            .map(|c| {
                let names: Vec<String> = c.split('#').map(str::to_string).collect();
                multi_codes(ds, &names, &d.rows).map(|(ids, g)| (c.clone(), ids, g))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let w = d.wkind.map(|_| d.w.clone());
    for f in factors.iter_mut() {
        f.prepare(w.as_deref());
    }
    Ok(Setup { d, factors, specs, vce, clusters, singletons, noabsorb, tol, maxiter, w })
}

/// Rows of the absorbed degrees-of-freedom table.
struct DofRow {
    label: String,
    cats: usize,
    redundant: usize,
    mark: char,
}

/// Degrees of freedom used by the fixed effects; also whether the
/// intercept is part of the absorbed coefficients.
fn absorbed_dof(s: &Setup) -> (Vec<DofRow>, bool) {
    let w = s.w.as_deref();
    let mut rows = Vec::new();
    let mut first: Option<usize> = None;
    let mut second = false;
    for (k, f) in s.factors.iter().enumerate() {
        if f.intercept {
            let nested_cl = !s.noabsorb && s.clusters.iter().any(|(_, c, _)| hdfe::nested(&f.ids, f.ng, c));
            let (red, mark) = if nested_cl {
                (f.ng, '*')
            } else if first.is_none() {
                first = Some(k);
                (0, ' ')
            } else if !second {
                second = true;
                let a = &s.factors[first.unwrap()];
                let comps = hdfe::components(&a.ids, a.ng, &f.ids, f.ng);
                (comps, ' ')
            } else {
                (1.min(f.ng), '?')
            };
            rows.push(DofRow { label: f.label.clone(), cats: f.ng, redundant: red, mark });
        }
        for (j, sn) in f.slope_names.iter().enumerate() {
            let red = hdfe::degenerate_slope_groups(f, j, w);
            rows.push(DofRow { label: format!("{}#c.{}", f.label, sn), cats: f.ng, redundant: red, mark: ' ' });
        }
    }
    (rows, first.is_some())
}

fn dof_table(rows: &[DofRow]) -> Vec<String> {
    let mut out = vec![
        "Absorbed degrees of freedom:".to_string(),
        format!("{}+", "-".repeat(53)),
        " Absorbed FE | Categories  - Redundant  = Num. Coefs |".to_string(),
        format!("{}+{}|", "-".repeat(13), "-".repeat(39)),
    ];
    for r in rows {
        let label = crate::summarize::abbrev(&r.label, 12);
        out.push(format!("{:>12} |{:>10}{:>12}{:>12}    {}|", label, r.cats, r.redundant, r.cats - r.redundant, r.mark));
    }
    out.push(format!("{}+", "-".repeat(53)));
    if rows.iter().any(|r| r.mark == '?') {
        out.push("? = number of redundant parameters may be higher".into());
    }
    if rows.iter().any(|r| r.mark == '*') {
        out.push("* = FE nested within cluster; treated as redundant for DoF computation".into());
    }
    out
}

/// Sandwich variance for coefficient columns `z` with scores `u` (one per
/// observation, already including weights).
fn meat(z: &[Vec<f64>], u: &[f64], cl: Option<(&[u32], usize)>) -> Matrix {
    let k = z.len();
    let mut m = Matrix::zeros(k, k);
    match cl {
        None => {
            for i in 0..u.len() {
                let ui = u[i];
                if ui == 0.0 {
                    continue;
                }
                let u2 = ui * ui;
                for a in 0..k {
                    let za = z[a][i] * u2;
                    for b in 0..=a {
                        m[(a, b)] += za * z[b][i];
                    }
                }
            }
        }
        Some((ids, g)) => {
            let mut s = vec![0.0; g * k];
            for i in 0..u.len() {
                let base = ids[i] as usize * k;
                for a in 0..k {
                    s[base + a] += z[a][i] * u[i];
                }
            }
            for gi in 0..g {
                let sg = &s[gi * k..(gi + 1) * k];
                for a in 0..k {
                    for b in 0..=a {
                        m[(a, b)] += sg[a] * sg[b];
                    }
                }
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

/// Cameron-Gelbach-Miller multiway clustering: inclusion-exclusion over the
/// intersections of the cluster variables.
fn cluster_meat(z: &[Vec<f64>], u: &[f64], clusters: &[(String, Vec<u32>, usize)]) -> Matrix {
    let k = z.len();
    let nc = clusters.len();
    let mut total = Matrix::zeros(k, k);
    for mask in 1u32..(1 << nc) {
        let mut ids: Option<(Vec<u32>, usize)> = None;
        for (j, (_, c, g)) in clusters.iter().enumerate() {
            if mask & (1 << j) == 0 {
                continue;
            }
            ids = Some(match ids {
                None => (c.clone(), *g),
                Some((a, ga)) => hdfe::combine(&a, ga, c, *g),
            });
        }
        let (ids, g) = ids.unwrap();
        let m = meat(z, u, Some((&ids, g)));
        let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
        total = total.add(&m.scale(sign)).unwrap();
    }
    total
}

/// Replace negative eigenvalues by zero (non-PSD multiway variance).
fn psd_fix(v: &Matrix) -> (Matrix, bool) {
    let (vals, vecs) = v.sym_eigen();
    if vals.iter().all(|&l| l >= 0.0) {
        return (v.clone(), false);
    }
    let k = v.rows;
    let mut out = Matrix::zeros(k, k);
    for (e, &l) in vals.iter().enumerate() {
        if l <= 0.0 {
            continue;
        }
        for a in 0..k {
            for b in 0..k {
                out[(a, b)] += l * vecs[(a, e)] * vecs[(b, e)];
            }
        }
    }
    (out, true)
}

/// Number formatted like `%10.0fc`.
fn commas(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        let s = util::commas(v.abs() as usize);
        if v < 0.0 { format!("-{s}") } else { s }
    } else {
        format!("{v:.0}")
    }
}

fn f4(v: f64) -> String {
    if is_missing(v) || !v.is_finite() { ".".into() } else { format!("{v:.4}") }
}

fn f2(v: f64) -> String {
    if is_missing(v) || !v.is_finite() { ".".into() } else { format!("{v:.2}") }
}

/// Insert the "Robust" caption above the standard-error column.
fn robust_caption(table: &mut Vec<String>) {
    if table.len() < 2 {
        return;
    }
    let lw = table[1].find(" | ").unwrap_or(12);
    table[1] = table[1].replacen("Std. err.", "std. err.", 1);
    table.insert(1, format!("{:>lw$} |               Robust", ""));
}

/// Header lines: left text and right statistic, reghdfe layout.
fn header_line(left: &str, label: &str, value: &str) -> String {
    if label.is_empty() {
        return left.to_string();
    }
    format!("{:<50}{:<15} = {:>10}", left, label, value)
}

/// Variance, F test and degrees of freedom for a fitted model.
struct Inference {
    v: Matrix,
    df_r: f64,
    n_clust: Vec<usize>,
    psd_fixed: bool,
}

#[allow(clippy::too_many_arguments)]
fn inference(s: &Setup, z: &[Vec<f64>], inv: &Matrix, resid: &[f64], rss: f64, n: f64, kk: f64) -> Inference {
    let w = s.w.as_deref();
    let fw = s.d.wkind == Some(WeightKind::F);
    match &s.vce {
        VceKind::Unadjusted => Inference { v: inv.scale(rss / (n - kk)), df_r: n - kk, n_clust: vec![], psd_fixed: false },
        VceKind::Robust => {
            let u: Vec<f64> = (0..resid.len()).map(|i| match w { Some(w) => if fw { w[i].sqrt() * resid[i] } else { w[i] * resid[i] }, None => resid[i] }).collect();
            let m = meat(z, &u, None);
            let v = inv.mul(&m).unwrap().mul(inv).unwrap().scale(n / (n - kk));
            Inference { v, df_r: n - kk, n_clust: vec![], psd_fixed: false }
        }
        VceKind::Cluster(_) => {
            let u: Vec<f64> = (0..resid.len()).map(|i| w.map_or(1.0, |w| w[i]) * resid[i]).collect();
            let m = cluster_meat(z, &u, &s.clusters);
            let g = s.clusters.iter().map(|c| c.2).min().unwrap_or(1) as f64;
            let q = (n - 1.0) / (n - kk) * g / (g - 1.0);
            let v = inv.mul(&m).unwrap().mul(inv).unwrap().scale(q);
            let (v, fixed) = if s.clusters.len() > 1 { psd_fix(&v) } else { (v, false) };
            Inference { v, df_r: g - 1.0, n_clust: s.clusters.iter().map(|c| c.2).collect(), psd_fixed: fixed }
        }
    }
}

/// Wald F statistic of the listed coefficients.
fn wald_f(b: &[f64], v: &Matrix, idx: &[usize]) -> (f64, f64) {
    let k = idx.len();
    if k == 0 {
        return (SYSMISS, 0.0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-12);
    let q = (k - dropped.len()) as f64;
    if q == 0.0 {
        return (SYSMISS, 0.0);
    }
    let wv: f64 = (0..k).map(|a| (0..k).map(|c| b[idx[a]] * vi[(a, c)] * b[idx[c]]).sum::<f64>()).sum();
    (wv / q, q)
}

fn add_double(ds: &mut DataSet, name: &str, rows: &[usize], vals: &[f64], label: &str) -> Result<(), String> {
    if ds.var_index.contains_key(name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mut all = vec![SYSMISS; ds.n_obs];
    for (&r, &v) in rows.iter().zip(vals) {
        all[r] = v;
    }
    let mut var = util::new_double(name, all);
    var.label = label.to_string();
    ds.add_variable(var)
}

/// Save residuals and fixed-effect estimates as requested.
fn save_outputs(ds: &mut DataSet, s: &Setup, cmd: &Command, fitted_xb: &[f64], resid: &[f64]) -> Result<Option<String>, String> {
    let resid_name = cmd.opts.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= 5 && "residuals".starts_with(n.as_str())
    });
    let resid_name = resid_name.map(|(_, a)| a.as_deref().map(str::trim).filter(|a| !a.is_empty()).unwrap_or("_reghdfe_resid").to_string());
    if let Some(rn) = &resid_name {
        if ds.var_index.contains_key(rn.as_str()) && rn == "_reghdfe_resid" {
            ds.drop_variable(rn)?;
        }
        crate::varlist::check_name(rn)?;
        add_double(ds, rn, &s.d.rows, resid, "Residuals")?;
    }
    if s.factors.iter().any(|f| f.save_as.is_some()) {
        let dsum: Vec<f64> = (0..resid.len()).map(|i| s.d.y[i] - fitted_xb[i] - resid[i]).collect();
        let parts = hdfe::recover(&s.factors, &dsum, s.w.as_deref(), 1e-10, 10_000);
        for (f, comp) in s.factors.iter().zip(&parts) {
            let Some(name) = &f.save_as else { continue };
            for (a, vals) in comp.iter().enumerate() {
                let slope_ix = a as isize - f.intercept as isize;
                let (vname, label) = if slope_ix < 0 {
                    (name.clone(), format!("[FE] {}", f.label))
                } else if !f.intercept && f.slopes.len() == 1 {
                    (name.clone(), format!("[FE] {}#c.{}", f.label, f.slope_names[0]))
                } else {
                    (format!("{name}_Slope{}", slope_ix + 1), format!("[FE] {}#c.{}", f.label, f.slope_names[slope_ix as usize]))
                };
                if vname.starts_with("__hdfe") && ds.var_index.contains_key(vname.as_str()) {
                    ds.drop_variable(&vname)?;
                }
                add_double(ds, &vname, &s.d.rows, vals, &label)?;
            }
        }
    }
    Ok(resid_name)
}

// ---------------------------------------------------------------------------
// reghdfe
// ---------------------------------------------------------------------------

pub struct Reghdfe;

impl CommandHandler for Reghdfe {
    fn name(&self) -> &'static str {
        "reghdfe"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "reghdfe");
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let dep = if dep.contains('.') { dep.to_string() } else { crate::varlist::resolve_name(ds, dep)? };
        let s = setup(ds, st, cmd, &dep, indep.trim(), &[])?;
        let level = util::level(cmd, st)?;
        let noconstant = cmd.has_opt("noconstant", 5);
        let w = s.w.as_deref();
        let d = &s.d;
        let n_obs = d.n();
        let n = d.n_eff();
        let cons = d.k() - 1; // build() always appends _cons
        let intercept_fe = s.factors.iter().any(|f| f.intercept);
        let mut omitted = d.omitted.clone();
        let mut names = d.names.clone();
        let mut notes: Vec<String> = d.notes.clone();
        // regressors to partial out (the constant only when no fixed effect
        // carries an intercept)
        let xcols: Vec<usize> = (0..d.k()).filter(|&j| !omitted[j] && j != cons).collect();
        let explicit_cons = !intercept_fe && !noconstant;
        let ybar = wmean(&d.y, w);
        let tss = d.y.iter().enumerate().map(|(i, y)| w.map_or(1.0, |w| w[i]) * (y - ybar).powi(2)).sum::<f64>();
        let xbar: Vec<f64> = xcols.iter().map(|&j| wmean(&d.cols[j], w)).collect();
        let xss: Vec<f64> = xcols.iter().zip(&xbar).map(|(&j, m)| d.cols[j].iter().enumerate().map(|(i, x)| w.map_or(1.0, |w| w[i]) * (x - m).powi(2)).sum()).collect();
        let mut work: Vec<Vec<f64>> = Vec::with_capacity(xcols.len() + 2);
        work.push(d.y.clone());
        for &j in &xcols {
            work.push(d.cols[j].clone());
        }
        if explicit_cons {
            work.push(vec![1.0; n_obs]);
        }
        let conv = {
            let mut refs: Vec<&mut Vec<f64>> = work.iter_mut().collect();
            hdfe::partial_out_all(&s.factors, &mut refs, w, s.tol, s.maxiter)
        };
        if conv.iter().any(|c| !c.converged) {
            return Err(rc_err(430, format!("convergence not achieved in {} iterations; try a larger maxiterations() or tolerance()", s.maxiter)));
        }
        let iters = conv.iter().map(|c| c.iterations).max().unwrap_or(1);
        let mut ytil = work.remove(0);
        let tss_within = wdot(&ytil, &ytil, w);
        // drop regressors explained by the fixed effects
        let mut zcols: Vec<(usize, Vec<f64>)> = Vec::new();
        for (p, (&j, col)) in xcols.iter().zip(work.drain(..xcols.len())).enumerate() {
            let ss = wdot(&col, &col, w);
            if ss <= 1e-9 * xss[p].max(1e-300) || xss[p] == 0.0 {
                omitted[j] = true;
                notes.push(format!("note: {} is probably collinear with the fixed effects (all partialled-out values are close to zero; tol = 1.0e-09)", names[j]));
                names[j] = omit_name(&names[j]);
            } else {
                zcols.push((j, col));
            }
        }
        if explicit_cons {
            let c = work.pop().unwrap();
            zcols.push((cons, c));
        }
        let add_back = intercept_fe && !noconstant;
        if add_back {
            for (j, col) in zcols.iter_mut() {
                let m = xbar[xcols.iter().position(|x| x == j).unwrap()];
                for v in col.iter_mut() {
                    *v += m;
                }
            }
            for v in ytil.iter_mut() {
                *v += ybar;
            }
            zcols.push((cons, vec![1.0; n_obs]));
        }
        if !explicit_cons && !add_back {
            omitted[cons] = true;
        }
        // X'WX and collinearity among the remaining regressors
        let k = zcols.len();
        let mut xtx = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..=a {
                let v = wdot(&zcols[a].1, &zcols[b].1, w);
                xtx[(a, b)] = v;
                xtx[(b, a)] = v;
            }
        }
        // the constant first so that regressors are dropped instead
        let mut order: Vec<usize> = (0..k).collect();
        if let Some(pc) = zcols.iter().position(|(j, _)| *j == cons) {
            order.retain(|&a| a != pc);
            order.insert(0, pc);
        }
        let mut xo = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                xo[(a, b)] = xtx[(order[a], order[b])];
            }
        }
        let (_, dropped) = xo.sym_inverse(1e-10);
        if !dropped.is_empty() {
            let drop_set: Vec<usize> = dropped.iter().map(|&a| order[a]).collect();
            let mut kept = Vec::new();
            for (a, zc) in zcols.into_iter().enumerate() {
                if drop_set.contains(&a) {
                    omitted[zc.0] = true;
                    notes.push(format!("note: {} omitted because of collinearity.", names[zc.0]));
                    names[zc.0] = omit_name(&names[zc.0]);
                } else {
                    kept.push(zc);
                }
            }
            zcols = kept;
        }
        let k = zcols.len();
        let mut xtx = Matrix::zeros(k, k);
        let mut xty = vec![0.0; k];
        for a in 0..k {
            xty[a] = wdot(&zcols[a].1, &ytil, w);
            for b in 0..=a {
                let v = wdot(&zcols[a].1, &zcols[b].1, w);
                xtx[(a, b)] = v;
                xtx[(b, a)] = v;
            }
        }
        let (inv, _) = xtx.sym_inverse(1e-14);
        let bz: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
        let mut resid = ytil.clone();
        for (a, (_, col)) in zcols.iter().enumerate() {
            let ba = bz[a];
            for (r, x) in resid.iter_mut().zip(col) {
                *r -= ba * x;
            }
        }
        let rss = wdot(&resid, &resid, w);
        // degrees of freedom
        let (dof_rows, intercept_counted) = absorbed_dof(&s);
        let df_a: usize = dof_rows.iter().map(|r| r.cats - r.redundant).sum();
        let df_m = zcols.iter().filter(|(j, _)| *j != cons).count() as f64;
        let extra = if explicit_cons || (intercept_fe && !intercept_counted) { 1.0 } else { 0.0 };
        let kk = df_m + df_a as f64 + extra;
        if n - kk <= 0.0 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let zs: Vec<Vec<f64>> = zcols.iter().map(|(_, c)| c.clone()).collect();
        let inf = inference(&s, &zs, &inv, &resid, rss, n, kk);
        drop(zs);
        // full coefficient vector and variance
        let kfull = d.k();
        let mut b = vec![0.0; kfull];
        let act: Vec<usize> = zcols.iter().map(|(j, _)| *j).collect();
        for (a, &j) in act.iter().enumerate() {
            b[j] = bz[a];
        }
        let v = est::expand_v(&inf.v, &act, kfull);
        let xidx: Vec<usize> = act.iter().copied().filter(|&j| j != cons).collect();
        let (f, _) = wald_f(&b, &v, &xidx);
        let df_r = inf.df_r;
        let pf = if is_missing(f) { SYSMISS } else { dist::ftail(df_m, df_r, f) };
        let r2 = 1.0 - rss / tss;
        let r2_a = 1.0 - (rss / (n - kk)) / (tss / (n - 1.0));
        let r2_within = 1.0 - rss / tss_within;
        let r2_a_within = 1.0 - (rss / (n - kk)) / (tss_within / (n - kk + df_m - 1.0).max(1.0));
        let rmse = (rss / (n - kk)).sqrt();
        // ---- output -------------------------------------------------------
        let mut out: Vec<String> = Vec::new();
        if s.singletons > 0 {
            out.push(format!("(dropped {} singleton observations)", util::commas(s.singletons)));
        }
        out.push(format!("(MWFE estimator converged in {iters} iterations)"));
        out.extend(notes.iter().cloned());
        if inf.psd_fixed {
            out.push("Warning: VCV matrix was non-positive semi-definite; adjustment from Cameron, Gelbach & Miller applied.".into());
        }
        let nabs = s.factors.len();
        let title2 = format!("Absorbing {} HDFE group{}", nabs, if nabs == 1 { "" } else { "s" });
        let robust = s.vce != VceKind::Unadjusted;
        if !cmd.has_opt("noheader", 5) {
            out.push(String::new());
            let fl = format!("F({:>4},{:>7})", df_m, commas(df_r));
            let mut left_fixed: Vec<String> = vec!["HDFE Linear regression".into(), title2.clone(), if robust { "Statistics robust to heteroskedasticity".into() } else { String::new() }];
            left_fixed.extend(std::iter::repeat_n(String::new(), 4));
            // cluster counts take the left column from "Within R-sq." on
            for (c, (name, _, g)) in s.clusters.iter().enumerate() {
                let l = format!("{:<28} = {:>10}", format!("Number of clusters ({name})"), commas(*g as f64));
                if 5 + c < left_fixed.len() {
                    left_fixed[5 + c] = l;
                } else {
                    left_fixed.push(l);
                }
            }
            let right = [
                ("Number of obs", commas(n)),
                (fl.as_str(), f2(f)),
                ("Prob > F", f4(pf)),
                ("R-squared", f4(r2)),
                ("Adj R-squared", f4(r2_a)),
                ("Within R-sq.", f4(r2_within)),
                ("Root MSE", f4(rmse)),
            ];
            for (i, l) in left_fixed.iter().enumerate() {
                match right.get(i) {
                    Some((lab, val)) => out.push(header_line(l, lab, val)),
                    None => out.push(l.clone()),
                }
            }
            if let VceKind::Cluster(cs) = &s.vce {
                out.push(String::new());
                let g = inf.n_clust.iter().min().copied().unwrap_or(0);
                out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), cs.join(" "))));
            }
            if !robust || !matches!(s.vce, VceKind::Cluster(_)) {
                out.push(String::new());
            }
        }
        let tbl_names: Vec<String> = names.clone();
        let mut show_names = tbl_names.clone();
        let mut show_omitted = omitted.clone();
        let mut show_base = d.is_base.clone();
        if noconstant || omitted[cons] {
            // the constant is not reported
            show_names.truncate(kfull - 1);
            show_omitted.truncate(kfull - 1);
            show_base.truncate(kfull - 1);
        }
        let kshow = show_names.len();
        if !cmd.has_opt("notable", 5) {
            let mut table = est::coef_table(&est::TableSpec {
                depvar: &dep,
                names: &show_names,
                b: &b[..kshow],
                v: &v,
                omitted: &show_omitted,
                is_base: &show_base,
                df: Some(df_r),
                level,
                eform: None,
                coef_title: "Coefficient",
            });
            if robust {
                robust_caption(&mut table);
            }
            out.extend(table);
        }
        if !cmd.has_opt("nofootnote", 5) {
            out.push(String::new());
            out.extend(dof_table(&dof_rows));
        }
        let text = out.join("\n");
        // ---- saved variables ----------------------------------------------
        let xb_fit: Vec<f64> = (0..n_obs).map(|i| (0..kfull).filter(|&j| !omitted[j] && b[j] != 0.0).map(|j| b[j] * d.cols[j][i]).sum()).collect();
        let resid_name = save_outputs(ds, &s, cmd, &xb_fit, &resid)?;
        st.data_changed = true;
        // ---- e() ------------------------------------------------------------
        let mut dpost = d.clone();
        dpost.names = show_names.clone();
        dpost.omitted = show_omitted.clone();
        dpost.cols.truncate(kshow);
        let bpost = b[..kshow].to_vec();
        let vpost = {
            let mut m = Matrix::zeros(kshow, kshow);
            for a in 0..kshow {
                for c in 0..kshow {
                    m[(a, c)] = v[(a, c)];
                }
            }
            m
        };
        let absvars: Vec<String> = s.specs.iter().map(|sp| sp.ivars.join("#")).collect();
        let extended: Vec<String> = dof_rows.iter().map(|r| r.label.clone()).collect();
        let redundant: usize = dof_rows.iter().map(|r| r.redundant).sum();
        let nested: usize = dof_rows.iter().filter(|r| r.mark == '*').map(|r| r.redundant).sum();
        let dof_m = {
            let mut m = Matrix::zeros(dof_rows.len(), 3);
            for (r, row) in dof_rows.iter().enumerate() {
                m[(r, 0)] = row.cats as f64;
                m[(r, 1)] = row.redundant as f64;
                m[(r, 2)] = (row.cats - row.redundant) as f64;
            }
            m.with_names(extended.clone(), vec!["categories".into(), "redundant".into(), "num_coefs".into()])
        };
        let indepvars: Vec<String> = show_names.iter().filter(|n| *n != "_cons").cloned().collect();
        let vce = s.vce.clone();
        let clusters: Vec<(String, usize)> = s.clusters.iter().map(|(c, _, g)| (c.clone(), *g)).collect();
        let wkind = d.wkind;
        est::post(st, "reghdfe", &dpost, &bpost, &vpost, &|e: &mut Results| {
            e.set("N", n);
            e.set("df_r", df_r);
            e.set("df_m", df_m);
            e.set("df_a", df_a as f64);
            e.set("df_a_initial", dof_rows.iter().map(|r| r.cats).sum::<usize>() as f64);
            e.set("df_a_redundant", redundant as f64);
            e.set("df_a_nested", nested as f64);
            e.set("N_hdfe", nabs as f64);
            e.set("N_hdfe_extended", dof_rows.len() as f64);
            e.set("r2", r2);
            e.set("r2_a", r2_a);
            e.set("r2_within", r2_within);
            e.set("r2_a_within", r2_a_within);
            e.set("rmse", rmse);
            e.set("rss", rss);
            e.set("tss", tss);
            e.set("tss_within", tss_within);
            e.set("mss", tss - rss);
            e.set("F", est::missing_or(f));
            e.set("ll", -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (rss / n).ln() + 1.0));
            e.set("ll_0", -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (tss_within / n).ln() + 1.0));
            e.set("ic", iters as f64);
            e.set("converged", 1.0);
            e.set("num_singletons", s.singletons as f64);
            e.set("drop_singletons", s.singletons as f64);
            e.set("report_constant", if noconstant { 0.0 } else { 1.0 });
            e.set("rank", act.len() as f64);
            e.set_str("title", "HDFE Linear regression");
            e.set_str("title2", title2.clone());
            e.set_str("absvars", if s.noabsorb { "_cons".to_string() } else { absvars.join(" ") });
            e.set_str("extended_absvars", extended.join(" "));
            e.set_str("indepvars", indepvars.join(" "));
            e.set_str("model", "ols");
            e.set_str("predict", "reghdfe_p");
            e.set_str("estat_cmd", "reghdfe_estat");
            e.set_str("marginsok", "Residuals XB D XBD");
            e.set_str("dofmethod", "pairwise clusters continuous");
            e.set_str("cmdline", format!("reghdfe {}", cmd.raw.trim()));
            e.set_mat("dof_table", dof_m.clone());
            match &vce {
                VceKind::Unadjusted => e.set_str("vce", "unadjusted"),
                VceKind::Robust => {
                    e.set_str("vce", "robust");
                    e.set_str("vcetype", "Robust");
                }
                VceKind::Cluster(cs) => {
                    e.set_str("vce", "cluster");
                    e.set_str("vcetype", "Robust");
                    e.set_str("clustvar", cs.join(" "));
                    e.set("N_clustervars", cs.len() as f64);
                    e.set("N_clust", clusters.iter().map(|c| c.1).min().unwrap_or(0) as f64);
                    for (i, (c, g)) in clusters.iter().enumerate() {
                        e.set(&format!("N_clust{}", i + 1), *g as f64);
                        e.set_str(&format!("clustvar{}", i + 1), c.clone());
                    }
                }
            }
            if let Some(r) = &resid_name {
                e.set_str("resid", r.clone());
            }
            if let Some(k) = wkind {
                e.set_str("wtype", match k { WeightKind::F => "fweight", WeightKind::A => "aweight", WeightKind::P => "pweight", _ => "iweight" });
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict after reghdfe
// ---------------------------------------------------------------------------

/// `predict` statistics that need the fixed effects: `d`, `xbd`,
/// `residuals` and `dresiduals`. Returns None for the others (xb, stdp),
/// which the generic linear prediction handles.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let stat = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).find_map(|n| match n.as_str() {
        "d" => Some("d"),
        "xbd" => Some("xbd"),
        "dresiduals" | "dresid" | "dres" => Some("dresiduals"),
        o if o.len() >= 1 && "residuals".starts_with(o) => Some("residuals"),
        _ => None,
    });
    let Some(stat) = stat else { return Ok(None) };
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let (ty, name) = match words.as_slice() {
        [t, n] if VarType::from_type_str(t).is_some() => (VarType::from_type_str(t), n.to_string()),
        [n] => (None, n.to_string()),
        _ => return Err(rc_err(198, "one new variable name required")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let resid_var = st.e.macros.get("resid").cloned().filter(|r| ds.var_index.contains_key(r)).ok_or_else(|| rc_err(198, "you must add the residuals() option to reghdfe before running this prediction"))?;
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let mask = util::touse(ds, st, cmd)?;
    let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let env = crate::eval::Env::with_store(ds, st);
    let dep_expr = ferrostat_parser::parser::parse_expression(&dep).map_err(|e| rc_err(198, e.message))?;
    let r = util::col(ds, &resid_var)?;
    let mut vals = vec![SYSMISS; ds.n_obs];
    for i in 0..ds.n_obs {
        if !mask[i] || !sample.get(i).copied().unwrap_or(false) || is_missing(r[i]) {
            continue;
        }
        let y = env.eval(&dep_expr, i)?.num()?;
        let xb = crate::postest::xb_row(&env, &b, None, i)?;
        vals[i] = match stat {
            "residuals" => r[i],
            "xbd" => y - r[i],
            "d" => y - r[i] - xb,
            _ => y - xb,
        };
    }
    drop(env);
    let t = ty.unwrap_or(VarType::Double);
    let mut var = ferrostat_core::Variable::new(&name, t);
    let nmiss = vals.iter().filter(|x| is_missing(**x)).count();
    var.data = vals.iter().map(|&x| if is_missing(x) { Value::Double(SYSMISS) } else { crate::data::store_num(t, x) }).collect();
    var.label = match stat {
        "d" => "d[absorbed fixed effects]".into(),
        "xbd" => "Xb + d[absorbed fixed effects]".into(),
        "dresiduals" => "d + Residuals".into(),
        _ => "Residuals".into(),
    };
    ds.add_variable(var)?;
    st.data_changed = true;
    Ok(Some(if nmiss > 0 { format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }) } else { String::new() }))
}

// ---------------------------------------------------------------------------
// ivreghdfe: two-stage least squares absorbing fixed effects
// ---------------------------------------------------------------------------

pub struct Ivreghdfe;

/// Weighted cross products of column lists.
fn cross_cols(a: &[&[f64]], b: &[&[f64]], w: Option<&[f64]>) -> Matrix {
    let mut m = Matrix::zeros(a.len(), b.len());
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            m[(i, j)] = wdot(x, y, w);
        }
    }
    m
}

impl CommandHandler for Ivreghdfe {
    fn name(&self) -> &'static str {
        "ivreghdfe"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "ivreghdfe");
        }
        let main = cmd.main.trim();
        let open = main.find('(').ok_or_else(|| rc_err(198, "(varlist2 = instlist) required"))?;
        let close = main.rfind(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
        let (endog_t, inst_t) = main[open + 1..close].split_once('=').ok_or_else(|| rc_err(198, "= required inside parentheses"))?;
        let outside = format!("{} {}", &main[..open], &main[close + 1..]);
        let mut words = outside.split_whitespace();
        let dep = crate::varlist::resolve_name(ds, words.next().ok_or_else(|| rc_err(100, "varlist required"))?)?;
        let expand = |t: &str| -> Result<Vec<String>, String> {
            let mut v = Vec::new();
            for w in t.split_whitespace() {
                v.extend(crate::varlist::expand(ds, w)?);
            }
            Ok(v)
        };
        let exog = expand(&words.collect::<Vec<_>>().join(" "))?;
        let endog = expand(endog_t)?;
        let inst = expand(inst_t)?;
        if endog.is_empty() || inst.len() < endog.len() {
            return Err(rc_err(481, "equation not identified; must have at least as many instruments not in the regression as there are instrumented variables"));
        }
        let all: Vec<String> = exog.iter().chain(&endog).chain(&inst).cloned().collect();
        let s = setup(ds, st, cmd, &dep, &all.join(" "), &[])?;
        let small = cmd.has_opt("small", 3);
        let level = util::level(cmd, st)?;
        let w = s.w.as_deref();
        let d = &s.d;
        let n = d.n_eff();
        let col_of = |name: &str| d.names.iter().position(|x| x == name).filter(|&j| !d.omitted[j]);
        let pick = |names: &[String]| -> Vec<(String, usize)> { names.iter().filter_map(|nm| col_of(nm).map(|j| (nm.clone(), j))).collect() };
        let (xe, xn, zi) = (pick(&exog), pick(&endog), pick(&inst));
        let mut uniq: Vec<usize> = xe.iter().chain(&xn).chain(&zi).map(|p| p.1).collect();
        uniq.sort();
        uniq.dedup();
        let mut work: Vec<Vec<f64>> = std::iter::once(d.y.clone()).chain(uniq.iter().map(|&j| d.cols[j].clone())).collect();
        let conv = {
            let mut refs: Vec<&mut Vec<f64>> = work.iter_mut().collect();
            hdfe::partial_out_all(&s.factors, &mut refs, w, s.tol, s.maxiter)
        };
        if conv.iter().any(|c| !c.converged) {
            return Err(rc_err(430, "convergence not achieved"));
        }
        let iters = conv.iter().map(|c| c.iterations).max().unwrap_or(1);
        let ytil = &work[0];
        let get = |j: usize| -> &[f64] { &work[1 + uniq.iter().position(|&u| u == j).unwrap()] };
        let xnames: Vec<String> = xn.iter().chain(&xe).map(|p| p.0.clone()).collect();
        let xs: Vec<&[f64]> = xn.iter().chain(&xe).map(|p| get(p.1)).collect();
        let zs: Vec<&[f64]> = zi.iter().chain(&xe).map(|p| get(p.1)).collect();
        let k = xs.len();
        // first stage: X̂ = Z (Z'Z)^-1 Z'X
        let (zzi, _) = cross_cols(&zs, &zs, w).sym_inverse(1e-12);
        let pi = zzi.mul(&cross_cols(&zs, &xs, w)).unwrap();
        let n_obs = ytil.len();
        let xhat: Vec<Vec<f64>> = (0..k)
            .map(|c| {
                let mut v = vec![0.0; n_obs];
                for (r, z) in zs.iter().enumerate() {
                    let p = pi[(r, c)];
                    for (vi, zi) in v.iter_mut().zip(z.iter()) {
                        *vi += p * zi;
                    }
                }
                v
            })
            .collect();
        let xh: Vec<&[f64]> = xhat.iter().map(|v| v.as_slice()).collect();
        let (bread, dropped) = cross_cols(&xh, &xh, w).sym_inverse(1e-12);
        if !dropped.is_empty() {
            return Err(rc_err(481, "equation not identified (collinear instruments or regressors)"));
        }
        let xhy: Vec<f64> = xh.iter().map(|x| wdot(x, ytil, w)).collect();
        let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| bread[(a, c)] * xhy[c]).sum()).collect();
        let mut resid = ytil.clone();
        for (a, x) in xs.iter().enumerate() {
            for (r, xi) in resid.iter_mut().zip(x.iter()) {
                *r -= b[a] * xi;
            }
        }
        let rss = wdot(&resid, &resid, w);
        let tss_within = wdot(ytil, ytil, w);
        let (dof_rows, intercept_counted) = absorbed_dof(&s);
        let df_a = dof_rows.iter().map(|r| r.cats - r.redundant).sum::<usize>() as f64 + if intercept_counted { 0.0 } else { 1.0 };
        let kk = k as f64 + df_a;
        if n - kk <= 0.0 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let fw = d.wkind == Some(WeightKind::F);
        let xhc: Vec<Vec<f64>> = xhat.clone();
        let (v, df_r, n_clust) = match &s.vce {
            VceKind::Unadjusted => {
                let s2 = rss / if small { n - kk } else { n - df_a };
                (bread.scale(s2), n - kk, None)
            }
            VceKind::Robust => {
                let u: Vec<f64> = (0..n_obs).map(|i| match w { Some(w) => if fw { w[i].sqrt() * resid[i] } else { w[i] * resid[i] }, None => resid[i] }).collect();
                let m = meat(&xhc, &u, None);
                let q = if small { n / (n - kk) } else { n / (n - df_a) };
                (bread.mul(&m).unwrap().mul(&bread).unwrap().scale(q), n - kk, None)
            }
            VceKind::Cluster(_) => {
                let u: Vec<f64> = (0..n_obs).map(|i| w.map_or(1.0, |w| w[i]) * resid[i]).collect();
                let m = cluster_meat(&xhc, &u, &s.clusters);
                let g = s.clusters.iter().map(|c| c.2).min().unwrap_or(1) as f64;
                let q = if small { (n - 1.0) / (n - kk) * g / (g - 1.0) } else { 1.0 };
                let v = bread.mul(&m).unwrap().mul(&bread).unwrap().scale(q);
                let v = if s.clusters.len() > 1 { psd_fix(&v).0 } else { v };
                (v, g - 1.0, Some(g as usize))
            }
        };
        let idx: Vec<usize> = (0..k).collect();
        let (wf, q) = wald_f(&b, &v, &idx);
        let (stat_label, stat, pstat) = if small {
            (format!("F({:>3},{:>6})", q, commas(df_r)), wf, dist::ftail(q, df_r, wf))
        } else {
            (format!("Wald chi2({q})"), wf * q, dist::chi2tail(q, wf * q))
        };
        let r2 = 1.0 - rss / tss_within;
        let rmse = (rss / if small { n - kk } else { n - df_a }).sqrt();
        let mut out = vec![format!("(MWFE estimator converged in {iters} iterations)")];
        if s.singletons > 0 {
            out.insert(0, format!("(dropped {} singleton observations)", util::commas(s.singletons)));
        }
        out.push(String::new());
        out.push("IV (2SLS) estimation".into());
        out.push("--------------------".into());
        out.push(String::new());
        out.push(match s.vce {
            VceKind::Unadjusted => "Statistics consistent for homoskedasticity only".into(),
            VceKind::Robust => "Statistics robust to heteroskedasticity".into(),
            VceKind::Cluster(ref c) => format!("Statistics robust to heteroskedasticity and clustering on {}", c.join(" ")),
        });
        out.push(String::new());
        let nabs = s.factors.len();
        out.push(header_line(&format!("Absorbing {} HDFE group{}", nabs, if nabs == 1 { "" } else { "s" }), "Number of obs", &commas(n)));
        out.push(header_line("", &stat_label, &f2(stat)));
        out.push(header_line("", if small { "Prob > F" } else { "Prob > chi2" }, &f4(pstat)));
        out.push(header_line("", "Centered R2", &f4(r2)));
        out.push(header_line("", "Root MSE", &f4(rmse)));
        if let Some(g) = n_clust {
            out.push(header_line(&format!("Number of clusters = {}", util::commas(g)), "", ""));
        }
        out.push(String::new());
        let omitted = vec![false; k];
        let mut table = est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &xnames,
            b: &b,
            v: &v,
            omitted: &omitted,
            is_base: &omitted,
            df: if small { Some(df_r) } else { None },
            level,
            eform: None,
            coef_title: "Coefficient",
        });
        if s.vce != VceKind::Unadjusted {
            robust_caption(&mut table);
        }
        out.extend(table);
        let names = |v: &[(String, usize)]| v.iter().map(|p| p.0.clone()).collect::<Vec<_>>().join(" ");
        out.push(format!("Instrumented:         {}", names(&xn)));
        if !xe.is_empty() {
            out.push(format!("Included instruments: {}", names(&xe)));
        }
        out.push(format!("Excluded instruments: {}", names(&zi)));
        out.push(String::new());
        out.extend(dof_table(&dof_rows));
        let text = out.join("\n");
        let mut dpost = d.clone();
        dpost.names = xnames.clone();
        dpost.omitted = vec![false; k];
        dpost.is_base = vec![false; k];
        dpost.cols = xn.iter().chain(&xe).map(|p| d.cols[p.1].clone()).collect();
        dpost.has_cons = false;
        let absvars: Vec<String> = s.specs.iter().map(|sp| sp.ivars.join("#")).collect();
        let vce = s.vce.clone();
        let (instd, inexog, exexog) = (names(&xn), names(&xe), names(&zi));
        est::post(st, "ivreghdfe", &dpost, &b, &v, &|e: &mut Results| {
            e.set("N", n);
            e.set("df_r", df_r);
            e.set("df_m", q);
            e.set("df_a", df_a);
            e.set("N_hdfe", nabs as f64);
            e.set("rss", rss);
            e.set("r2", r2);
            e.set("rmse", rmse);
            if small {
                e.set("F", est::missing_or(wf));
            } else {
                e.set("chi2", est::missing_or(wf * q));
            }
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            e.set("small", if small { 1.0 } else { 0.0 });
            e.set_str("model", "iv");
            e.set_str("title", "IV (2SLS) estimation");
            e.set_str("absvars", absvars.join(" "));
            e.set_str("instd", instd.clone());
            e.set_str("inexog", inexog.clone());
            e.set_str("exexog", exexog.clone());
            e.set_str("insts", format!("{exexog} {inexog}").trim().to_string());
            e.set_str("vce", match &vce { VceKind::Unadjusted => "unadjusted", VceKind::Robust => "robust", VceKind::Cluster(_) => "cluster" });
            if let VceKind::Cluster(c) = &vce {
                e.set_str("clustvar", c.join(" "));
            }
            if vce != VceKind::Unadjusted {
                e.set_str("vcetype", "Robust");
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
