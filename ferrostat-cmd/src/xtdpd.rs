//! Dynamic panel-data GMM: `xtabond` (Arellano-Bond difference GMM),
//! `xtdpdsys` (Arellano-Bover / Blundell-Bond system GMM) and the
//! community `xtabond2` syntax, with one- and two-step estimators,
//! Windmeijer-corrected robust standard errors, Arellano-Bond tests for
//! serial correlation of the differenced residuals and Sargan / Hansen
//! tests of the overidentifying restrictions (`estat abond`,
//! `estat sargan`).
//!
//! The estimator follows Arellano and Bond (1991), Blundell and Bond (1998),
//! Windmeijer (2005) and Roodman (2009, "How to do xtabond2"); the
//! organisation of the moment conditions and the robust AR-test variance
//! follow xtabond2 by David Roodman (GPL-3.0-or-later).

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xtnl::{hl, hv};

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("xtabond", Box::new(Dpd { verb: "xtabond" })), ("xtdpdsys", Box::new(Dpd { verb: "xtdpdsys" })), ("xtabond2", Box::new(Dpd { verb: "xtabond2" }))]
}

// ---------------------------------------------------------------------------
// GMM engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Row {
    /// 0 = transformed (differenced) equation, 1 = levels
    eq: u8,
    t: i64,
    y: f64,
    x: Vec<f64>,
    z: Vec<f64>,
    data_row: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum HMode {
    H1,
    H2,
    H3,
}

fn h_entry(a: &Row, b: &Row, mode: HMode) -> f64 {
    if mode == HMode::H1 {
        return if a.eq == b.eq && a.t == b.t { 1.0 } else { 0.0 };
    }
    match (a.eq, b.eq) {
        (0, 0) => {
            if a.t == b.t {
                2.0
            } else if (a.t - b.t).abs() == 1 {
                -1.0
            } else {
                0.0
            }
        }
        (1, 1) => {
            if a.t == b.t { 1.0 } else { 0.0 }
        }
        _ if mode == HMode::H2 => 0.0,
        (0, 1) => {
            if b.t == a.t {
                1.0
            } else if b.t == a.t - 1 {
                -1.0
            } else {
                0.0
            }
        }
        _ => h_entry(b, a, mode),
    }
}

struct GmmOut {
    b: Vec<f64>,
    /// reported variance (robust / corrected when requested)
    v: Matrix,
    sargan: f64,
    hansen: f64,
    sig2: f64,
    ar: Vec<(f64, f64)>,
    e1: Vec<Vec<f64>>,
}

fn sym_inv(m: &Matrix) -> Matrix {
    let mut s = m.clone();
    for i in 0..m.rows {
        for j in 0..m.cols {
            s[(i, j)] = 0.5 * (m[(i, j)] + m[(j, i)]);
        }
    }
    s.sym_inverse(1e-12).0
}

fn mv(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// GMM estimation over panels of rows. `n_main` is the reported number of
/// observations (used for sigma^2).
fn gmm(groups: &[Vec<Row>], k: usize, j: usize, twostep: bool, robust: bool, hmode: HMode, n_main: usize, artests: usize, dfk: usize) -> Result<GmmOut, String> {
    // official commands divide by N - k, xtabond2 by N
    let n_div = (n_main - dfk) as f64;
    // cross products
    let mut zx = Matrix::zeros(k, j); // X'Z
    let mut zy = vec![0.0; j];
    let mut a1 = Matrix::zeros(j, j);
    for g in groups {
        for r in g {
            for c in 0..j {
                let zc = r.z[c];
                if zc == 0.0 {
                    continue;
                }
                zy[c] += zc * r.y;
                for a in 0..k {
                    zx[(a, c)] += r.x[a] * zc;
                }
            }
        }
        for ra in g {
            for rb in g {
                let h = h_entry(ra, rb, hmode);
                if h == 0.0 {
                    continue;
                }
                for c in 0..j {
                    let za = ra.z[c] * h;
                    if za == 0.0 {
                        continue;
                    }
                    for d in 0..j {
                        a1[(c, d)] += za * rb.z[d];
                    }
                }
            }
        }
    }
    let a1 = sym_inv(&a1);
    let est = |w: &Matrix| -> (Vec<f64>, Matrix) {
        let zxa = zx.mul(w).unwrap(); // k x j
        let m = zxa.mul(&zx.transpose()).unwrap();
        let v = sym_inv(&m);
        let b = mv(&v, &mv(&zxa, &zy));
        (b, v)
    };
    let resid = |b: &[f64]| -> Vec<Vec<f64>> { groups.iter().map(|g| g.iter().map(|r| r.y - dot(&r.x, b)).collect()).collect() };
    let zte = |e: &[Vec<f64>]| -> Vec<f64> {
        let mut s = vec![0.0; j];
        for (g, eg) in groups.iter().zip(e) {
            for (r, ev) in g.iter().zip(eg) {
                for c in 0..j {
                    s[c] += r.z[c] * ev;
                }
            }
        }
        s
    };
    let (b1, mut v1) = est(&a1);
    let e1 = resid(&b1);
    let ssr: f64 = groups.iter().zip(&e1).map(|(g, eg)| g.iter().zip(eg).filter(|(r, _)| r.eq == if hmode == HMode::H1 { 1 } else { 0 } || hmode == HMode::H1).map(|(_, e)| e * e).sum::<f64>()).sum();
    let sig2 = ssr / n_div / if hmode == HMode::H1 { 1.0 } else { 2.0 };
    let a1s = a1.scale(1.0 / sig2);
    v1 = v1.scale(sig2);
    let ze1 = zte(&e1);
    let sargan = dot(&ze1, &mv(&a1s, &ze1));
    // robust weighting matrix from one-step residuals
    let mut a2raw = Matrix::zeros(j, j);
    let zi_e: Vec<Vec<f64>> = groups
        .iter()
        .zip(&e1)
        .map(|(g, eg)| {
            let mut s = vec![0.0; j];
            for (r, ev) in g.iter().zip(eg) {
                for c in 0..j {
                    s[c] += r.z[c] * ev;
                }
            }
            s
        })
        .collect();
    for s in &zi_e {
        for c in 0..j {
            if s[c] == 0.0 {
                continue;
            }
            for d in 0..j {
                a2raw[(c, d)] += s[c] * s[d];
            }
        }
    }
    let vzxa1 = v1.mul(&zx).unwrap().mul(&a1s).unwrap(); // k x j
    let v1robust = vzxa1.mul(&a2raw).unwrap().mul(&vzxa1.transpose()).unwrap();
    let a2 = sym_inv(&a2raw);
    let (b2, v2) = est(&a2);
    let e2 = resid(&b2);
    let ze2 = zte(&e2);
    let a2ze = mv(&a2, &ze2);
    let hansen = dot(&ze2, &a2ze);
    let (b, v_plain, v_rep, e, a_used, sig2_out) = if twostep {
        let mut sig2b = 0.0;
        for (g, eg) in groups.iter().zip(&e2) {
            for (r, ev) in g.iter().zip(eg) {
                if r.eq == 0 || hmode == HMode::H1 {
                    sig2b += ev * ev;
                }
            }
        }
        sig2b /= n_div * if hmode == HMode::H1 { 1.0 } else { 2.0 };
        let vrep = if robust {
            // Windmeijer (2005) finite-sample correction
            let mut dsum = Matrix::zeros(j, k);
            for (g, s1) in groups.iter().zip(&zi_e) {
                let mut zxi = Matrix::zeros(j, k);
                for r in g {
                    for c in 0..j {
                        if r.z[c] == 0.0 {
                            continue;
                        }
                        for a in 0..k {
                            zxi[(c, a)] += r.z[c] * r.x[a];
                        }
                    }
                }
                let sc = dot(s1, &a2ze);
                let row: Vec<f64> = (0..k).map(|a| (0..j).map(|c| a2ze[c] * zxi[(c, a)]).sum()).collect();
                for c in 0..j {
                    for a in 0..k {
                        dsum[(c, a)] += sc * zxi[(c, a)] + s1[c] * row[a];
                    }
                }
            }
            let vzxa2 = v2.mul(&zx).unwrap().mul(&a2).unwrap();
            let dm = vzxa2.mul(&dsum).unwrap();
            let t1 = dm.mul(&v1robust).unwrap().mul(&dm.transpose()).unwrap();
            let t2 = dm.mul(&v2).unwrap();
            let mut vr = Matrix::zeros(k, k);
            for a in 0..k {
                for c in 0..k {
                    vr[(a, c)] = v2[(a, c)] + t1[(a, c)] + t2[(a, c)] + t2[(c, a)];
                }
            }
            vr
        } else {
            v2.clone()
        };
        (b2, v2.clone(), vrep, e2, a2.clone(), sig2b)
    } else {
        let vrep = if robust { v1robust.clone() } else { v1.clone() };
        (b1, v1.clone(), vrep, e1.clone(), a1s.clone(), sig2)
    };
    // Arellano-Bond tests on the differenced residuals
    let m2vzxa = v_plain.mul(&zx).unwrap().mul(&a_used).unwrap().scale(-2.0); // k x j
    let mut ar = Vec::new();
    for l in 1..=artests {
        let mut ew = 0.0;
        let mut whw = 0.0;
        let mut zhw = vec![0.0; j];
        let mut xw = vec![0.0; k];
        for (g, eg) in groups.iter().zip(&e) {
            // lagged residual for each differenced row
            let tidx: HashMap<i64, usize> = g.iter().enumerate().filter(|(_, r)| r.eq == 0).map(|(i, r)| (r.t, i)).collect();
            let w: Vec<f64> = g.iter().map(|r| if r.eq == 0 { tidx.get(&(r.t - l as i64)).map_or(0.0, |&i| eg[i]) } else { 0.0 }).collect();
            let ewi: f64 = g.iter().zip(eg).zip(&w).filter(|((r, _), _)| r.eq == 0).map(|((_, e), w)| e * w).sum();
            ew += ewi;
            for (r, wv) in g.iter().zip(&w) {
                if r.eq == 0 && *wv != 0.0 {
                    for a in 0..k {
                        xw[a] += wv * r.x[a];
                    }
                }
            }
            if !twostep && !robust {
                for (ra, wa) in g.iter().zip(&w) {
                    for (rb, wb) in g.iter().zip(&w) {
                        let h = h_entry(ra, rb, hmode) * sig2;
                        if h == 0.0 {
                            continue;
                        }
                        whw += wa * h * wb;
                        if *wb != 0.0 {
                            for c in 0..j {
                                zhw[c] += ra.z[c] * h * wb;
                            }
                        }
                    }
                }
            } else {
                whw += ewi * ewi;
                for (r, ev) in g.iter().zip(eg) {
                    for c in 0..j {
                        zhw[c] += r.z[c] * ev * ewi;
                    }
                }
            }
        }
        if ew == 0.0 {
            ar.push((SYSMISS, SYSMISS));
            continue;
        }
        let t1 = dot(&xw, &mv(&m2vzxa, &zhw));
        let t2 = dot(&xw, &mv(&v_rep, &xw));
        let d = whw + t1 + t2;
        let z = ew / d.sqrt();
        ar.push((z, 2.0 * dist::normal(-z.abs())));
    }
    let _ = sig2_out;
    Ok(GmmOut { b, v: v_rep, sargan, hansen, sig2, ar, e1 })
}

// ---------------------------------------------------------------------------
// model specification
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Eqs {
    Diff,
    Level,
    Both,
}

#[derive(Clone, Debug)]
struct GmmSpec {
    vars: Vec<String>,
    lo: i64,
    hi: Option<i64>,
    collapse: bool,
    eq: Eqs,
    display: String,
}

#[derive(Clone, Debug)]
struct IvSpec {
    vars: Vec<String>,
    eq: Eqs,
    passthru: bool,
}

struct Spec {
    dep: String,
    regs: Vec<String>,
    cons: bool,
    gmm: Vec<GmmSpec>,
    iv: Vec<IvSpec>,
    system: bool,
    /// system rows only where a differenced row exists (official commands)
    level_rows_follow_diff: bool,
    /// constant column in the differenced equation (xtabond)
    diff_cons: bool,
}

pub struct Dpd {
    pub verb: &'static str,
}

/// Expand a varlist with time-series operator groups into variable terms.
fn terms(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    let t = crate::tsops::expand_text(text);
    let mut out = Vec::new();
    for v in crate::varlist::expand(ds, &t)? {
        // normalise L1. to L.
        let v = if let Some(rest) = v.strip_prefix("L1.") { format!("L.{rest}") } else { v };
        if !out.contains(&v) {
            out.push(v);
        }
    }
    Ok(out)
}

/// Option text split into (varlist, suboptions).
fn split_sub(arg: &str) -> (String, Vec<(String, Option<String>)>) {
    match arg.split_once(',') {
        Some((a, b)) => (a.trim().to_string(), ferrostat_parser::parser::split_options(b)),
        None => (arg.trim().to_string(), vec![]),
    }
}

fn sub_has(subs: &[(String, Option<String>)], name: &str, min: usize) -> Option<Option<String>> {
    subs.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= min && name.starts_with(n.as_str())
    }).map(|(_, a)| a.clone())
}

fn parse_eq(a: &str) -> Result<Eqs, String> {
    let a = a.trim().to_lowercase();
    if "diff".starts_with(a.as_str()) && !a.is_empty() || a == "difference" {
        Ok(Eqs::Diff)
    } else if "level".starts_with(a.as_str()) && !a.is_empty() || a == "levels" {
        Ok(Eqs::Level)
    } else if "both".starts_with(a.as_str()) && !a.is_empty() {
        Ok(Eqs::Both)
    } else {
        Err(rc_err(198, format!("equation({a}) invalid")))
    }
}

/// Display form of a term's difference: `L.w` -> `LD.w`, `w` -> `D.w`.
fn dname(v: &str) -> String {
    match v.split_once('.') {
        Some((op, var)) if op.chars().all(|c| c.is_ascii_alphanumeric()) && op.to_uppercase().starts_with('L') => format!("{}D.{var}", op.to_uppercase()),
        _ => format!("D.{v}"),
    }
}

impl Dpd {
    fn spec(&self, ds: &DataSet, cmd: &Command) -> Result<Spec, String> {
        let main = cmd.main.trim();
        let (dep, rest) = main.split_once(char::is_whitespace).map(|(a, b)| (a.to_string(), b.trim().to_string())).unwrap_or((main.to_string(), String::new()));
        if dep.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let dep = crate::varlist::resolve_name(ds, &dep)?;
        if self.verb == "xtabond2" {
            let regs = terms(ds, &rest)?;
            let mut gmm = Vec::new();
            let mut iv = Vec::new();
            let nolevel = cmd.has_opt("noleveleq", 5);
            for (name, arg) in &cmd.opts {
                let n = name.to_lowercase();
                let arg = arg.clone().unwrap_or_default();
                if n.starts_with("gmm") {
                    let (vl, subs) = split_sub(&arg);
                    let vars = terms(ds, &vl)?;
                    let (mut lo, mut hi) = (1i64, None);
                    if let Some(Some(l)) = sub_has(&subs, "laglimits", 3) {
                        let w: Vec<&str> = l.split_whitespace().collect();
                        if w.len() != 2 {
                            return Err(rc_err(198, "laglimits() requires two arguments"));
                        }
                        let p = |s: &str| -> Result<Option<i64>, String> { if s == "." { Ok(None) } else { s.parse::<i64>().map(Some).map_err(|_| rc_err(198, "laglimits() invalid")) } };
                        let (a, b) = (p(w[0])?, p(w[1])?);
                        lo = a.unwrap_or(1);
                        hi = b;
                        if let (Some(x), Some(y)) = (a, b) {
                            if x > y {
                                lo = y;
                                hi = Some(x);
                            }
                        }
                    }
                    let collapse = sub_has(&subs, "collapse", 1).is_some();
                    let eq = match sub_has(&subs, "equation", 2) {
                        Some(Some(e)) => parse_eq(&e)?,
                        _ => Eqs::Both,
                    };
                    let eq = if nolevel { Eqs::Diff } else { eq };
                    let disp = if vars.len() > 1 { format!("({})", vars.join(" ")) } else { vars.join(" ") };
                    gmm.push(GmmSpec { vars, lo, hi, collapse, eq, display: disp });
                } else if n.starts_with("iv") {
                    let (vl, subs) = split_sub(&arg);
                    let vars = terms(ds, &vl)?;
                    let eq = match sub_has(&subs, "equation", 2) {
                        Some(Some(e)) => parse_eq(&e)?,
                        _ => Eqs::Both,
                    };
                    let eq = if nolevel { Eqs::Diff } else { eq };
                    iv.push(IvSpec { vars, eq, passthru: sub_has(&subs, "passthru", 1).is_some() });
                }
            }
            if gmm.is_empty() && iv.is_empty() {
                return Err(rc_err(198, "no instruments specified; use gmmstyle() and/or ivstyle()"));
            }
            if cmd.has_opt("orthogonal", 3) {
                return Err(rc_err(198, "option orthogonal not supported"));
            }
            let cons = !nolevel && !cmd.has_opt("noconstant", 3);
            if cons {
                iv.push(IvSpec { vars: vec!["_cons".into()], eq: Eqs::Level, passthru: true });
            }
            return Ok(Spec { dep, regs, cons, gmm, iv, system: !nolevel, level_rows_follow_diff: false, diff_cons: false });
        }
        // official syntax
        let p = util::opt_num(cmd, "lags", 1)?.unwrap_or(1.0) as i64;
        if p < 1 {
            return Err(rc_err(198, "lags() must be positive"));
        }
        let maxl = util::opt_num(cmd, "maxldep", 4)?.map(|v| v as i64);
        let maxlags = util::opt_num(cmd, "maxlags", 4)?.map(|v| v as i64);
        let mut regs: Vec<String> = (1..=p).map(|l| if l == 1 { format!("L.{dep}") } else { format!("L{l}.{dep}") }).collect();
        let exog = terms(ds, &rest)?;
        let mut gmm = vec![GmmSpec { vars: vec![dep.clone()], lo: 2, hi: maxl.map(|m| m + 1), collapse: false, eq: Eqs::Diff, display: dep.clone() }];
        let mut pre_regs = Vec::new();
        for (opt, lo) in [("pre", 1i64), ("endogenous", 2i64)] {
            let mut found = Vec::new();
            for (n, a) in &cmd.opts {
                let nl = n.to_lowercase();
                let hit = if opt == "pre" { nl.len() >= 3 && "predetermined".starts_with(nl.as_str()) } else { nl.len() >= 3 && "endogenous".starts_with(nl.as_str()) };
                if hit {
                    found.push(a.clone().unwrap_or_default());
                }
            }
            for a in found {
                let (vl, subs) = split_sub(&a);
                let vars = terms(ds, &vl)?;
                let (mut plags, mut pmax) = (0i64, maxlags);
                if let Some(Some(ls)) = sub_has(&subs, "lagstruct", 4) {
                    let w: Vec<&str> = ls.split(',').map(|s| s.trim()).collect();
                    plags = w.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                    pmax = w.get(1).and_then(|s| if *s == "." { None } else { s.parse().ok() });
                }
                for v in &vars {
                    for l in 0..=plags {
                        pre_regs.push(if l == 0 { v.clone() } else if l == 1 { format!("L.{v}") } else { format!("L{l}.{v}") });
                    }
                    gmm.push(GmmSpec { vars: vec![v.clone()], lo, hi: pmax.map(|m| m + lo - 1), collapse: false, eq: Eqs::Diff, display: v.clone() });
                }
            }
        }
        regs.extend(exog.iter().cloned());
        regs.extend(pre_regs);
        let mut ivv = exog.clone();
        if let Some(a) = cmd.opt("inst", 3) {
            ivv.extend(terms(ds, a)?);
        }
        let cons = !cmd.has_opt("noconstant", 3);
        let system = self.verb == "xtdpdsys";
        let mut iv = vec![IvSpec { vars: ivv, eq: Eqs::Diff, passthru: false }];
        if system {
            gmm.push(GmmSpec { vars: vec![dep.clone()], lo: 2, hi: Some(2), collapse: false, eq: Eqs::Level, display: dep.clone() });
            if cons {
                iv.push(IvSpec { vars: vec!["_cons".into()], eq: Eqs::Level, passthru: true });
            }
        } else if cons {
            iv.push(IvSpec { vars: vec!["_cons".into()], eq: Eqs::Diff, passthru: true });
        }
        Ok(Spec { dep, regs, cons, gmm, iv, system, level_rows_follow_diff: true, diff_cons: cons && !system })
    }
}

/// Values of a term for every observation of the dataset.
fn series(ds: &DataSet, st: &Store, term: &str) -> Result<Vec<f64>, String> {
    if term == "_cons" {
        return Ok(vec![1.0; ds.n_obs]);
    }
    if let Some(v) = ds.get_var(term) {
        return Ok(v.data.iter().map(|x| x.num()).collect());
    }
    let e = ferrostat_parser::parser::parse_expression(term).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st);
    (0..ds.n_obs).map(|i| env.eval(&e, i)?.num()).collect()
}

impl CommandHandler for Dpd {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            crate::est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        if cmd.weight.is_some() {
            return Err(rc_err(101, "weights not allowed"));
        }
        let (ivar, tvar) = match &st.tsset {
            Some((Some(p), t)) if !t.is_empty() => (p.clone(), t.clone()),
            _ => return Err(rc_err(459, "panel variable and time variable not set; use xtset panelvar timevar")),
        };
        let spec = self.spec(ds, cmd)?;
        let twostep = cmd.has_opt("twostep", 3);
        let robust = match crate::est::parse_vce(cmd, crate::est::Vce::Oim)? {
            crate::est::Vce::Robust | crate::est::Vce::Cluster(_) => true,
            _ => false,
        };
        let artests = util::opt_num(cmd, "artests", 2)?.unwrap_or(2.0) as usize;
        let hmode = match util::opt_num(cmd, "h", 1)? {
            Some(1.0) => HMode::H1,
            Some(2.0) => HMode::H2,
            _ => HMode::H3,
        };
        let hmode = if spec.system { hmode } else if hmode == HMode::H1 { HMode::H1 } else { HMode::H2 };
        let level = util::level(cmd, st)?;
        let touse = util::touse(ds, st, cmd)?;
        let pc = util::col(ds, &ivar)?;
        let tc = util::col(ds, &tvar)?;
        // (panel, time) -> observation
        let mut keys: Vec<f64> = (0..ds.n_obs).map(|i| pc[i]).filter(|v| !is_missing(*v)).collect();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let mut at: HashMap<(usize, i64), usize> = HashMap::new();
        for i in 0..ds.n_obs {
            if is_missing(pc[i]) || is_missing(tc[i]) {
                continue;
            }
            let g = keys.binary_search_by(|k| k.total_cmp(&pc[i])).unwrap();
            at.insert((g, tc[i].round() as i64), i);
        }
        // series of every term involved
        let mut cache: HashMap<String, Vec<f64>> = HashMap::new();
        let mut need: Vec<String> = vec![spec.dep.clone()];
        need.extend(spec.regs.iter().cloned());
        for g in &spec.gmm {
            need.extend(g.vars.iter().cloned());
        }
        for v in &spec.iv {
            need.extend(v.vars.iter().cloned());
        }
        for t in need {
            if !cache.contains_key(&t) {
                let s = series(ds, st, &t)?;
                cache.insert(t, s);
            }
        }
        let val = |term: &str, g: usize, t: i64| -> f64 {
            match at.get(&(g, t)) {
                Some(&i) => cache[term][i],
                None => SYSMISS,
            }
        };
        let k0 = spec.regs.len() + spec.cons as usize;
        let mut groups_rows: Vec<Vec<Row>> = Vec::new();
        let mut col_index: BTreeMap<String, usize> = BTreeMap::new();
        let mut n_main = 0usize;
        let mut per_group = Vec::new();
        let mut esample = vec![false; ds.n_obs];
        // observations ordered by panel then time
        let mut obs_by_g: Vec<Vec<(i64, usize)>> = vec![Vec::new(); keys.len()];
        for (&(g, t), &i) in &at {
            obs_by_g[g].push((t, i));
        }
        for v in obs_by_g.iter_mut() {
            v.sort();
        }
        let mut zrows: Vec<Vec<Vec<(usize, f64)>>> = Vec::new();
        for (g, obs) in obs_by_g.iter().enumerate() {
            let mut rows: Vec<Row> = Vec::new();
            let mut zr: Vec<Vec<(usize, f64)>> = Vec::new();
            let mut cnt = 0;
            for &(t, i) in obs {
                if !touse[i] {
                    continue;
                }
                let lv: Vec<f64> = spec.regs.iter().map(|r| val(r, g, t)).collect();
                let y = val(&spec.dep, g, t);
                let ok_level = !is_missing(y) && lv.iter().all(|v| !is_missing(*v));
                if !ok_level {
                    continue;
                }
                let y1 = val(&spec.dep, g, t - 1);
                let lv1: Vec<f64> = spec.regs.iter().map(|r| val(r, g, t - 1)).collect();
                let ok_diff = !is_missing(y1) && lv1.iter().all(|v| !is_missing(*v)) && at.get(&(g, t - 1)).is_some();
                let mut make = |eq: u8| -> (Row, Vec<(usize, f64)>) {
                    let mut x: Vec<f64> = if eq == 0 { lv.iter().zip(&lv1).map(|(a, b)| a - b).collect() } else { lv.clone() };
                    if spec.cons {
                        x.push(if eq == 0 { if spec.diff_cons { 1.0 } else { 0.0 } } else { 1.0 });
                    }
                    let yy = if eq == 0 { y - y1 } else { y };
                    let mut z: Vec<(usize, f64)> = Vec::new();
                    let mut put = |key: String, v: f64, z: &mut Vec<(usize, f64)>| {
                        if v == 0.0 || is_missing(v) {
                            return;
                        }
                        let n = col_index.len();
                        let c = *col_index.entry(key).or_insert(n);
                        z.push((c, v));
                    };
                    for (si, gs) in spec.gmm.iter().enumerate() {
                        let in_eq = match (gs.eq, eq) {
                            (Eqs::Both, _) => true,
                            (Eqs::Diff, 0) => true,
                            (Eqs::Level, 1) => true,
                            _ => false,
                        };
                        if !in_eq {
                            continue;
                        }
                        for var in &gs.vars {
                            if eq == 0 {
                                let mut l = gs.lo;
                                loop {
                                    if let Some(h) = gs.hi {
                                        if l > h {
                                            break;
                                        }
                                    }
                                    if t - l < -100000 || (gs.hi.is_none() && obs.first().map_or(true, |f| t - l < f.0 - 50)) {
                                        break;
                                    }
                                    let v = val(var, g, t - l);
                                    let key = if gs.collapse { format!("g{si}|{var}|d|L{l}") } else { format!("g{si}|{var}|d|{t}|L{l}") };
                                    put(key, v, &mut z);
                                    l += 1;
                                }
                            } else {
                                let l = gs.lo - 1;
                                let v = val(var, g, t - l) - val(var, g, t - l - 1);
                                let v = if is_missing(val(var, g, t - l)) || is_missing(val(var, g, t - l - 1)) { SYSMISS } else { v };
                                let key = if gs.collapse { format!("g{si}|{var}|l|L{l}") } else { format!("g{si}|{var}|l|{t}|L{l}") };
                                put(key, v, &mut z);
                            }
                        }
                    }
                    for (si, ivs) in spec.iv.iter().enumerate() {
                        let in_eq = match (ivs.eq, eq) {
                            (Eqs::Both, _) => true,
                            (Eqs::Diff, 0) => true,
                            (Eqs::Level, 1) => true,
                            _ => false,
                        };
                        if !in_eq {
                            continue;
                        }
                        for var in &ivs.vars {
                            let v = if eq == 0 && !ivs.passthru {
                                let (a, b) = (val(var, g, t), val(var, g, t - 1));
                                if var == "_cons" { 0.0 } else if is_missing(a) || is_missing(b) { SYSMISS } else { a - b }
                            } else if var == "_cons" {
                                1.0
                            } else {
                                val(var, g, t)
                            };
                            put(format!("i{si}|{var}"), v, &mut z);
                        }
                    }
                    (Row { eq, t, y: yy, x, z: Vec::new(), data_row: i }, z)
                };
                let mut main_here = false;
                if ok_diff {
                    let (r, z) = make(0);
                    rows.push(r);
                    zr.push(z);
                    if !spec.system {
                        main_here = true;
                    }
                }
                if spec.system && (ok_diff || !spec.level_rows_follow_diff) {
                    let (r, z) = make(1);
                    rows.push(r);
                    zr.push(z);
                    main_here = true;
                }
                if main_here {
                    cnt += 1;
                    n_main += 1;
                    esample[i] = true;
                }
            }
            if cnt > 0 {
                per_group.push(cnt);
                groups_rows.push(rows);
                zrows.push(zr);
            }
        }
        if n_main == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        // dense instrument matrix, dropping all-zero and collinear columns
        let j0 = col_index.len();
        for (rows, zr) in groups_rows.iter_mut().zip(&zrows) {
            for (r, z) in rows.iter_mut().zip(zr) {
                r.z = vec![0.0; j0];
                for &(c, v) in z {
                    r.z[c] = v;
                }
            }
        }
        let mut ztz = Matrix::zeros(j0, j0);
        for rows in &groups_rows {
            for r in rows {
                for a in 0..j0 {
                    if r.z[a] == 0.0 {
                        continue;
                    }
                    for b in 0..j0 {
                        ztz[(a, b)] += r.z[a] * r.z[b];
                    }
                }
            }
        }
        // scale to unit diagonal before the rank check
        let dg: Vec<f64> = (0..j0).map(|a| ztz[(a, a)].sqrt().max(1e-150)).collect();
        let mut zs = ztz.clone();
        for a in 0..j0 {
            for b in 0..j0 {
                zs[(a, b)] = zs[(a, b)] / dg[a] / dg[b];
            }
        }
        let (_, dropped) = zs.sym_inverse(1e-10);
        let keep: Vec<usize> = (0..j0).filter(|c| !dropped.contains(c)).collect();
        let j = keep.len();
        for rows in groups_rows.iter_mut() {
            for r in rows.iter_mut() {
                r.z = keep.iter().map(|&c| r.z[c]).collect();
            }
        }
        // regressors: drop collinear columns
        let mut names: Vec<String> = spec.regs.clone();
        if spec.cons {
            names.push("_cons".into());
        }
        let mut xtx = Matrix::zeros(k0, k0);
        for rows in &groups_rows {
            for r in rows {
                for a in 0..k0 {
                    for b in 0..k0 {
                        xtx[(a, b)] += r.x[a] * r.x[b];
                    }
                }
            }
        }
        // rank check with the constant first (as _rmcoll keeps it), zero
        // columns dropped up front
        let mut order: Vec<usize> = Vec::new();
        if spec.cons {
            order.push(k0 - 1);
        }
        order.extend((0..k0).filter(|&c| !(spec.cons && c == k0 - 1)));
        let nz: Vec<usize> = order.iter().cloned().filter(|&c| xtx[(c, c)] > 0.0).collect();
        let mut sc = Matrix::zeros(nz.len(), nz.len());
        for (a, &ca) in nz.iter().enumerate() {
            for (b, &cb) in nz.iter().enumerate() {
                sc[(a, b)] = xtx[(ca, cb)] / xtx[(ca, ca)].sqrt() / xtx[(cb, cb)].sqrt();
            }
        }
        let (_, dr) = sc.sym_inverse(1e-10);
        let dropped_set: Vec<usize> = (0..k0).filter(|c| !nz.contains(c)).chain(dr.iter().map(|&a| nz[a])).collect();
        let xkeep: Vec<usize> = (0..k0).filter(|c| !dropped_set.contains(c)).collect();
        let mut notes = Vec::new();
        for c in 0..k0 {
            if dropped_set.contains(&c) {
                notes.push(if self.verb == "xtabond2" { format!("{} dropped due to collinearity", names[c]) } else { format!("note: {} dropped because of collinearity.", names[c]) });
            }
        }
        let k = xkeep.len();
        for rows in groups_rows.iter_mut() {
            for r in rows.iter_mut() {
                r.x = xkeep.iter().map(|&c| r.x[c]).collect();
            }
        }
        let anames: Vec<String> = xkeep.iter().map(|&c| names[c].clone()).collect();
        if j < k {
            return Err(rc_err(481, "equation not identified; must have at least as many instruments as regressors"));
        }
        let dfk = if self.verb == "xtabond2" { 0 } else { k };
        let res = gmm(&groups_rows, k, j, twostep, robust, hmode, n_main, artests.max(2), dfk)?;
        // Wald test of all coefficients except the constant
        let idx: Vec<usize> = (0..k).filter(|&a| anames[a] != "_cons").collect();
        let mut vs = Matrix::zeros(idx.len(), idx.len());
        for (a, &i) in idx.iter().enumerate() {
            for (c, &jj) in idx.iter().enumerate() {
                vs[(a, c)] = res.v[(i, jj)];
            }
        }
        let (vi, dr) = vs.sym_inverse(1e-14);
        let bs: Vec<f64> = idx.iter().map(|&i| res.b[i]).collect();
        let chi2 = dot(&bs, &mv(&vi, &bs));
        let df_m = (idx.len() - dr.len()) as f64;
        let ng = groups_rows.len();
        let (gmin, gmax) = (*per_group.iter().min().unwrap(), *per_group.iter().max().unwrap());
        let gavg = n_main as f64 / ng as f64;
        let sar_df = (j - k) as f64;
        let vcetype = if robust {
            if twostep {
                if self.verb == "xtabond2" { "Corrected" } else { "WC-Robust" }
            } else {
                "Robust"
            }
        } else {
            ""
        };
        // ---------------------------------------------------------- output
        let mut out: Vec<String> = Vec::new();
        out.extend(notes.iter().cloned());
        let table = crate::est::coef_table(&crate::est::TableSpec {
            depvar: &spec.dep,
            names: &anames,
            b: &res.b,
            v: &res.v,
            omitted: &vec![false; k],
            is_base: &vec![false; k],
            df: None,
            level,
            eform: None,
            coef_title: "Coefficient",
        });
        let lw = table.last().map_or(12, |l| l.len().saturating_sub(66));
        let mut table = table;
        if !vcetype.is_empty() {
            table.insert(1, format!("{:>lw$} |{:>21}", "", vcetype));
        }
        let steps = if twostep { "Two-step" } else { "One-step" };
        // instrument listing
        let mut dlist_gmm = Vec::new();
        let mut llist_gmm = Vec::new();
        for gs in &spec.gmm {
            if matches!(gs.eq, Eqs::Diff | Eqs::Both) {
                let range = match gs.hi {
                    Some(h) if h == gs.lo => if gs.lo == 1 { "L".to_string() } else { format!("L{}", gs.lo) },
                    Some(h) => format!("L({}/{})", gs.lo, h),
                    None => format!("L({}/.)", gs.lo),
                };
                dlist_gmm.push(format!("{range}.{}{}", gs.display, if gs.collapse { " collapsed" } else { "" }));
            }
            if matches!(gs.eq, Eqs::Level | Eqs::Both) {
                let l = gs.lo - 1;
                let lag = match l {
                    0 => String::new(),
                    1 => "L".into(),
                    l => format!("L{l}"),
                };
                llist_gmm.push(format!("{lag}D.{}{}", gs.display, if gs.collapse { " collapsed" } else { "" }));
            }
        }
        let mut dlist_iv = Vec::new();
        let mut llist_iv = Vec::new();
        for ivs in &spec.iv {
            for v in &ivs.vars {
                if matches!(ivs.eq, Eqs::Diff | Eqs::Both) && v != "_cons" {
                    dlist_iv.push(if ivs.passthru { v.clone() } else { dname(v) });
                } else if v == "_cons" && ivs.eq == Eqs::Diff {
                    dlist_iv.push("_cons".into());
                }
                if matches!(ivs.eq, Eqs::Level | Eqs::Both) {
                    llist_iv.push(v.clone());
                }
            }
        }
        let wrap = |label: &str, items: &[String], indent: usize| -> Vec<String> {
            let mut lines = Vec::new();
            let mut cur = format!("{:indent$}{label}", "");
            let start = cur.len();
            for it in items {
                if cur.len() + it.len() + 1 > 78 && cur.len() > start {
                    lines.push(cur);
                    cur = " ".repeat(start);
                }
                if !cur.ends_with(' ') {
                    cur.push(' ');
                }
                cur.push_str(it);
            }
            lines.push(cur);
            lines
        };
        if self.verb == "xtabond2" {
            out.push(String::new());
            out.push(format!("Dynamic panel-data estimation, {}-step {} GMM", if twostep { "two" } else { "one" }, if spec.system { "system" } else { "difference" }));
            out.push("-".repeat(78));
            out.push(format!("{:<48}{} = {:>9}", format!("Group variable: {}", crate::summarize::abbrev(&ivar, 12)), "Number of obs     ", n_main));
            out.push(format!("{:<48}{} = {:>9}", format!("Time variable : {}", crate::summarize::abbrev(&tvar, 12)), "Number of groups  ", ng));
            out.push(format!("{:<48}Obs per group: min = {:>9}", format!("Number of instruments = {j}"), gmin));
            out.push(format!("{:<14}= {:>9.2}{:>38}{:>9.2}", format!("Wald chi2({})", df_m), chi2, "avg = ", gavg));
            out.push(format!("{:<14}={:>10.3}{:>38}{:>9}", "Prob > chi2", dist::chi2tail(df_m, chi2), "max = ", gmax));
            out.extend(table);
            if twostep && vcetype != "Corrected" {
                out.push("Warning: Uncorrected two-step standard errors are unreliable.".into());
                out.push(String::new());
            }
            out.push(format!("Instruments for {} equation", if spec.system { "first differences" } else { "first differences" }));
            if !dlist_iv.is_empty() {
                out.push("  Standard".into());
                out.extend(wrap("", &[format!("D.({})", spec.iv.iter().filter(|s| matches!(s.eq, Eqs::Diff | Eqs::Both) && !s.passthru).flat_map(|s| s.vars.iter().cloned()).collect::<Vec<_>>().join(" "))], 4));
            }
            if !dlist_gmm.is_empty() {
                out.push("  GMM-type (missing=0, separate instruments for each period unless collapsed)".into());
                for gs in &dlist_gmm {
                    out.push(format!("    {gs}"));
                }
            }
            if spec.system {
                out.push("Instruments for levels equation".into());
                if !llist_iv.is_empty() {
                    out.push("  Standard".into());
                    out.extend(wrap("", &llist_iv, 4));
                }
                if !llist_gmm.is_empty() {
                    out.push("  GMM-type (missing=0, separate instruments for each period unless collapsed)".into());
                    for gs in &llist_gmm {
                        out.push(format!("    {gs}"));
                    }
                }
            }
            out.push("-".repeat(78));
            for (l, (z, p)) in res.ar.iter().enumerate().take(artests) {
                out.push(format!("{:<51}z = {:>6.2}  Pr > z = {:>6.3}", format!("Arellano-Bond test for AR({}) in first differences:", l + 1), z, p));
            }
            out.push("-".repeat(78));
            out.push(format!("{:<48}={:>7.2}{:>3}Prob > chi2 = {:>6.3}", format!("Sargan test of overid. restrictions: chi2({})", sar_df), res.sargan, "", dist::chi2tail(sar_df, res.sargan)));
            out.push("  (Not robust, but not weakened by many instruments.)".into());
            if twostep || robust {
                out.push(format!("{:<48}={:>7.2}{:>3}Prob > chi2 = {:>6.3}", format!("Hansen test of overid. restrictions: chi2({})", sar_df), res.hansen, "", dist::chi2tail(sar_df, res.hansen)));
                out.push("  (Robust, but weakened by many instruments.)".into());
            }
        } else {
            let title = if spec.system { "System dynamic panel-data estimation" } else { "Arellano\u{2013}Bond dynamic panel-data estimation" };
            let mut h = vec![hl(""), hv(title, "Number of obs", util::commas(n_main)), hv(&format!("Group variable: {ivar}"), "Number of groups", util::commas(ng)), hl(&format!("Time variable: {tvar}"))];
            h.push(hv("", "Obs per group:", String::new()));
            h.push(hv("", &format!("{:>16}", "min"), util::commas(gmin)));
            h.push(hv("", &format!("{:>16}", "avg"), fmt_g(gavg, 8)));
            h.push(hv("", &format!("{:>16}", "max"), util::commas(gmax)));
            h.push(hl(""));
            h.push(hv(&format!("Number of instruments = {:>6}", j), &format!("Wald chi2({})", fmt_g(df_m, 9)), format!("{:.2}", chi2)));
            h.push(hv("", "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))));
            out.extend(crate::xtnl::header_block(&h));
            out.push(format!("{steps} results"));
            if robust {
                out.push(format!("{:>78}", format!("(Std. err. adjusted for clustering on {ivar})")));
            }
            out.extend(table);
            out.push("Instruments for differenced equation".into());
            let dg: Vec<String> = dlist_gmm.clone();
            out.extend(wrap("GMM-type:", &dg, 8));
            if !dlist_iv.is_empty() {
                out.extend(wrap("Standard:", &dlist_iv, 8));
            }
            if spec.system {
                out.push("Instruments for level equation".into());
                if !llist_gmm.is_empty() {
                    out.extend(wrap("GMM-type:", &llist_gmm, 8));
                }
                if !llist_iv.is_empty() {
                    out.extend(wrap("Standard:", &llist_iv, 8));
                }
            }
        }
        let text = out.join("\n");
        // ---------------------------------------------------------- e()
        let verb = self.verb;
        let e = st.begin_e();
        e.set("N", n_main as f64);
        e.set("N_g", ng as f64);
        e.set("g_min", gmin as f64);
        e.set("g_avg", gavg);
        e.set("g_max", gmax as f64);
        e.set("chi2", chi2);
        e.set("df_m", df_m);
        e.set("p", dist::chi2tail(df_m, chi2));
        e.set("rank", k as f64);
        e.set("sig2", res.sig2);
        e.set("artests", artests as f64);
        if verb == "xtabond2" {
            e.set("j", j as f64);
            e.set("j0", j0 as f64);
            e.set("sargan", res.sargan);
            e.set("sar_df", sar_df);
            e.set("sarganp", dist::chi2tail(sar_df, res.sargan));
            if twostep || robust {
                e.set("hansen", res.hansen);
                e.set("hansen_df", sar_df);
                e.set("hansenp", dist::chi2tail(sar_df, res.hansen));
            }
            e.set("chi2p", dist::chi2tail(df_m, chi2));
            e.set("h", match hmode {
                HMode::H1 => 1.0,
                HMode::H2 => 2.0,
                HMode::H3 => 3.0,
            });
            e.set_str("esttype", if spec.system { "system" } else { "difference" });
            e.set_str("transform", "first differences");
            e.set_str("artype", "first differences");
            if twostep {
                e.set_str("twostep", "twostep");
            }
        } else {
            e.set("zrank", j as f64);
            e.set("sargan", if twostep { res.hansen } else { res.sargan });
            e.set("sar_df", sar_df);
            e.set_str("twostep", if twostep { "twostep" } else { "" });
        }
        for (l, (z, p)) in res.ar.iter().enumerate() {
            e.set(&format!("ar{}", l + 1), *z);
            e.set(&format!("ar{}p", l + 1), *p);
            if l < artests {
                e.set(&format!("arm{}", l + 1), *z);
            }
        }
        e.set("_sargan1", res.sargan);
        e.set("_hansen", res.hansen);
        e.set_str("cmd", verb);
        e.set_str("depvar", spec.dep.clone());
        e.set_str("ivar", ivar.clone());
        e.set_str("tvar", tvar.clone());
        e.set_str("vce", if robust { "robust" } else { "gmm" });
        if !vcetype.is_empty() {
            e.set_str("vcetype", vcetype);
        }
        e.set_str("_steps", if twostep { "2" } else { "1" });
        e.set_str("properties", "b V");
        e.set_str("predict", if verb == "xtabond2" { "xtab2_p" } else { "xtdpd_p" });
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&res.b).with_names(vec!["y1".into()], anames.clone()));
        e.set_mat("V", res.v.clone().with_names(anames.clone(), anames.clone()));
        e.sample = Some(esample);
        let _ = res.e1.len();
        let _ = groups_rows.iter().map(|g| g.iter().map(|r| r.data_row).count()).sum::<usize>();
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// estat abond / estat sargan
// ---------------------------------------------------------------------------

/// `estat abond` and `estat sargan` after xtabond / xtdpdsys / xtabond2;
/// None for other subcommands or estimators.
pub fn estat(st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if !matches!(ecmd.as_str(), "xtabond" | "xtdpdsys" | "xtabond2") {
        return None;
    }
    match sub {
        "abond" => {
            let want = match util::opt_num(cmd, "artests", 2) {
                Ok(v) => v.map(|v| v as usize).unwrap_or_else(|| st.e.get("artests").unwrap_or(2.0) as usize),
                Err(e) => return Some(Err(e)),
            };
            let mut out = vec![String::new(), "Arellano\u{2013}Bond test for zero autocorrelation in first-differenced errors".to_string(), "  +-----------------------+".into(), "  |Order |  z     Prob > z|".into(), "  |------+----------------|".into()];
            let mut r = Results::new();
            for l in 1..=want {
                let (Some(z), Some(p)) = (st.e.get(&format!("ar{l}")), st.e.get(&format!("ar{l}p"))) else {
                    return Some(Err(rc_err(198, format!("artests({want}) not available; refit with artests({want})"))));
                };
                out.push(format!("  |{:>4}  |{:>7} {:>7} |", l, fmt_g(z, 7), format!("{:.4}", p)));
                r.set(&format!("z{l}"), z);
                r.set(&format!("p{l}"), p);
            }
            out.push("  +-----------------------+".into());
            out.push("   H0: No autocorrelation ".into());
            let mut m = Matrix::zeros(want, 2);
            for l in 1..=want {
                m[(l - 1, 0)] = st.e.get(&format!("ar{l}")).unwrap_or(SYSMISS);
                m[(l - 1, 1)] = st.e.get(&format!("ar{l}p")).unwrap_or(SYSMISS);
            }
            r.set_mat("arm", m.with_names((1..=want).map(|l| l.to_string()).collect(), vec!["z".into(), "p".into()]));
            r.set("artests", want as f64);
            st.r = r;
            Some(Ok(out.join("\n")))
        }
        "sargan" => {
            let twostep = st.e.macros.get("_steps").map(String::as_str) == Some("2");
            let robust = st.e.macros.get("vce").map(String::as_str) == Some("robust");
            let df = st.e.get("sar_df").unwrap_or(SYSMISS);
            let s = if twostep { st.e.get("_hansen") } else { st.e.get("_sargan1") }.unwrap_or(SYSMISS);
            let p = dist::chi2tail(df, s);
            let mut out = vec![String::new(), "Sargan test of overidentifying restrictions".to_string(), "H0: Overidentifying restrictions are valid".into(), String::new()];
            out.push(format!("        chi2({})     = {:>9}", fmt_g(df, 9), fmt_g(s, 8)));
            out.push(format!("        Prob > chi2  = {:>9.4}", p));
            if robust && !twostep {
                out.insert(1, "Note: the Sargan test is not robust to heteroskedasticity; refit without vce(robust).".into());
            }
            let mut r = Results::new();
            r.set("chi2", s);
            r.set("df", df);
            r.set("p", p);
            st.r = r;
            Some(Ok(out.join("\n")))
        }
        _ => None,
    }
}
