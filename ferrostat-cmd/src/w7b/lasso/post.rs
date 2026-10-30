//! Postestimation for lasso fits: `lassoinfo`, `lassoknots`, `lassocoef`,
//! `lassogof`, `lassoselect`, `cvplot`, `coefpath`, `bicplot` and
//! `predict`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::core::{Family, Problem};
use super::fit::{self, Lasso};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub struct Post {
    pub verb: &'static str,
}

impl CommandHandler for Post {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        match self.verb {
            "lassoinfo" => lassoinfo(st, cmd),
            "lassoknots" => lassoknots(st, cmd),
            "lassocoef" => lassocoef(ds, st, cmd),
            "lassogof" => lassogof(ds, st, cmd),
            "lassoselect" => lassoselect(ds, st, cmd),
            "cvplot" | "bicplot" | "coefpath" => plot(ds, st, cmd, self.verb),
            _ => Err(rc_err(199, "unrecognized command")),
        }
    }
}

/// Results named in a namelist (`.` is the current estimates).
fn named(st: &Store, name: &str) -> Result<Results, String> {
    if name == "." {
        return Ok(st.e.clone());
    }
    st.estimates.get(name).cloned().ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))
}

fn need_lasso(e: &Results) -> Result<(), String> {
    if e.macros.contains_key("_lasso_list") {
        Ok(())
    } else {
        Err(rc_err(301, "last estimates not found; a lasso fit is required"))
    }
}

// ---------------------------------------------------------------------------
// lassoinfo
// ---------------------------------------------------------------------------

fn lassoinfo(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let names: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
    let names = if names.is_empty() { vec![".".to_string()] } else { names };
    let mut out = Vec::new();
    let mut total = 0;
    for nm in &names {
        let e = named(st, nm)?;
        need_lasso(&e)?;
        let ecmd = e.macros.get("cmd").cloned().unwrap_or_default();
        out.push(String::new());
        out.push(format!("    Estimate: {}", if nm == "." { "active results" } else { nm.as_str() }));
        out.push(format!("     Command: {ecmd}"));
        out.push(String::new());
        out.push("------------------------------------------------------------------------------".into());
        out.push("             |                                                     No. of".into());
        out.push("             |                                   Selection        selected".into());
        out.push("    Variable |      Model        Selection        criterion  lambda  variables".into());
        out.push("-------------+----------------------------------------------------------------".into());
        for (lab, l) in fit::load_all(&e) {
            total += 1;
            let (crit, lam, k) = match l.selected() {
                Some(p) => (match l.how.as_str() {
                    "cv" => "CV min.",
                    "serule" => "CV 1se",
                    "bic" => "BIC min.",
                    "user" => "user",
                    _ if l.selection == "plugin" => "plugin",
                    _ => "",
                }, fmt_g(p.lambda, 6), p.df.to_string()),
                None => ("", ".".into(), ".".into()),
            };
            out.push(format!("{:>12} | {:>10} {:>16} {:>16} {:>7} {:>10}", abbrev(&lab, 12), l.family.name(), sel_name(&l), crit, lam, k));
        }
        out.push("------------------------------------------------------------------------------".into());
    }
    let mut r = Results::new();
    r.set("N_lassos", total as f64);
    st.r = r;
    Ok(out.join("\n"))
}

fn sel_name(l: &Lasso) -> &'static str {
    match l.selection.as_str() {
        "cv" => "cv",
        "adaptive" => "adaptive",
        "plugin" => "plugin",
        "bic" => "bic",
        _ => "none",
    }
}

fn abbrev(s: &str, w: usize) -> String {
    if s.len() <= w {
        s.to_string()
    } else {
        format!("{}~{}", &s[..w / 2 - 1], &s[s.len() - (w - w / 2)..])
    }
}

// ---------------------------------------------------------------------------
// lassoknots
// ---------------------------------------------------------------------------

fn lassoknots(st: &mut Store, cmd: &Command) -> Result<String, String> {
    need_lasso(&st.e)?;
    let (_, l) = fit::load_one(&st.e, cmd)?;
    let disp = cmd.opt("display", 4).unwrap_or("").to_lowercase();
    let cv = l.pts.iter().any(|p| p.cvm.is_finite());
    let elastic = l.kind == "elasticnet";
    let mut cols: Vec<&str> = disp.split_whitespace().collect();
    if cols.is_empty() {
        cols = if cv {
            vec!["nonzero", "cvmpe"]
        } else if l.selection == "bic" {
            vec!["nonzero", "bic"]
        } else {
            vec!["nonzero"]
        };
    }
    let lin = l.family == Family::Linear;
    let mut out = vec![String::new()];
    let mut head = String::new();
    if elastic {
        head.push_str(&format!("{:>10} ", "alpha"));
    }
    head.push_str(&format!("{:>8} | {:>11}", "ID", "lambda"));
    let mut widths: Vec<usize> = Vec::new();
    for c in &cols {
        let lab = match *c {
            "nonzero" => "nonzero",
            "cvmpe" | "cvmd" => if lin { "CV mean pred. err." } else { "CV mean deviance" },
            "osr2" => if lin { "Out-of-sample R2" } else { "Out-sample dev. ratio" },
            "r2" | "rsquared" => if lin { "In-sample R2" } else { "In-sample dev. ratio" },
            "bic" => "BIC",
            "l1" => "L1 norm",
            "l2" => "L2 norm",
            o => return Err(rc_err(198, format!("display({o}) not allowed"))),
        };
        head.push_str(&format!(" {:>w$}", lab, w = lab.len().max(13)));
        widths.push(lab.len().max(13));
    }
    head.push_str("   Variables (A)dded, (R)emoved, or left (U)nchanged");
    let w = head.len().min(120);
    out.push("-".repeat(w));
    out.push(head);
    out.push("-".repeat(w));
    let mut prev: Vec<bool> = vec![false; l.names.len()];
    let mut prev_alpha = f64::NAN;
    let mut mat = Vec::new();
    for (k, p) in l.pts.iter().enumerate() {
        if p.alpha != prev_alpha {
            prev = vec![false; l.names.len()];
            prev_alpha = p.alpha;
        }
        let now: Vec<bool> = p.b.iter().map(|b| *b != 0.0).collect();
        let changes: Vec<String> = (0..l.names.len())
            .filter(|&j| now[j] != prev[j])
            .map(|j| format!("{} {}", if now[j] { "A" } else { "R" }, l.names[j]))
            .collect();
        let first = k == 0 || l.pts[k - 1].alpha != p.alpha;
        if changes.is_empty() && !first && Some(k) != l.sel && k + 1 != l.pts.len() {
            prev = now;
            continue;
        }
        let star = if Some(k) == l.sel { "*" } else { " " };
        let mut line = String::new();
        if elastic {
            line.push_str(&format!("{:>10} ", fmt_g(p.alpha, 6)));
        }
        line.push_str(&format!("{star}{:>7} | {:>11}", k + 1, fmt_g(p.lambda, 7)));
        let l1: f64 = p.b.iter().map(|b| b.abs()).sum();
        let l2: f64 = p.b.iter().map(|b| b * b).sum::<f64>().sqrt();
        let mut row = vec![(k + 1) as f64, p.lambda, p.df as f64];
        for c in &cols {
            let v = match *c {
                "nonzero" => p.df as f64,
                "cvmpe" | "cvmd" => p.cvm,
                "osr2" => 1.0 - p.cvm / l.dev0,
                "r2" | "rsquared" => 1.0 - p.dev / l.dev0,
                "bic" => p.bic,
                "l1" => l1,
                _ => l2,
            };
            row.push(v);
            let wd = widths[row.len() - 4];
            line.push_str(&format!(" {:>wd$}", if v.is_finite() { fmt_g(v, 8) } else { ".".into() }));
        }
        mat.push(row);
        let ch = if changes.is_empty() { "U".to_string() } else { changes.join(" ") };
        line.push_str(&format!("   {ch}"));
        out.push(line);
        prev = now;
    }
    out.push("-".repeat(w));
    if l.sel.is_some() {
        out.push("* lambda selected.".into());
    }
    let mut r = Results::new();
    let rows = mat.len();
    let ncol = mat.first().map_or(0, |r| r.len());
    let mut m = Matrix::zeros(rows, ncol);
    for (i, rr) in mat.iter().enumerate() {
        for (j, v) in rr.iter().enumerate() {
            m[(i, j)] = if v.is_finite() { *v } else { SYSMISS };
        }
    }
    r.set_mat("table", m);
    st.r = r;
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// lassocoef
// ---------------------------------------------------------------------------

/// Items of a lassocoef namelist: (results name, for() variable).
fn coef_items(main: &str) -> Vec<(String, Option<String>)> {
    let mut items = Vec::new();
    let mut rest = main.trim();
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('(') {
            let end = r.find(')').map(|e| {
                // include nested for(...) parenthesis
                let mut depth = 1;
                let mut pos = 0;
                for (i, c) in r.char_indices() {
                    if c == '(' {
                        depth += 1;
                    } else if c == ')' {
                        depth -= 1;
                        if depth == 0 {
                            pos = i;
                            break;
                        }
                    }
                }
                let _ = e;
                pos
            });
            let end = end.unwrap_or(r.len());
            let inner = &r[..end];
            let (nm, opts) = inner.split_once(',').unwrap_or((inner, ""));
            let f = opts.trim().strip_prefix("for(").and_then(|x| x.strip_suffix(')')).map(|x| x.trim().to_string());
            items.push((nm.trim().to_string(), f));
            rest = r.get(end + 1..).unwrap_or("").trim_start();
        } else {
            let (w, r) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            items.push((w.to_string(), None));
            rest = r.trim_start();
        }
    }
    if items.is_empty() {
        items.push((".".into(), None));
    }
    items
}

fn lasso_for(e: &Results, f: &Option<String>) -> Result<Lasso, String> {
    need_lasso(e)?;
    let all = fit::load_all(e);
    match f {
        Some(v) => all.into_iter().find(|(l, _)| l == v).map(|(_, x)| x).ok_or_else(|| rc_err(198, format!("for({v}): no lasso for {v}"))),
        None => all.into_iter().next().map(|(_, x)| x).ok_or_else(|| rc_err(301, "no lasso found")),
    }
}

fn lassocoef(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let items = coef_items(&cmd.main);
    let disp = cmd.opt("display", 4).unwrap_or("x").to_lowercase();
    let (what, how) = match disp.split_once(',') {
        Some((a, b)) => (a.trim().to_string(), b.trim().to_string()),
        None => (disp.trim().to_string(), String::new()),
    };
    let coef = what.starts_with("coef");
    let standardized = how.starts_with("stand");
    let postsel = how.starts_with("post");
    let mut cols: Vec<(String, Vec<(String, f64)>)> = Vec::new();
    for (nm, f) in &items {
        let e = named(st, nm)?;
        let l = lasso_for(&e, f)?;
        let k = l.sel.ok_or_else(|| rc_err(498, "no lambda selected; use lassoselect"))?;
        let p = &l.pts[k];
        let mut vals: Vec<(String, f64)> = Vec::new();
        let (b0, b) = if coef && (postsel || standardized) {
            let mask = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
            let (_, x, y) = fit::columns(ds, st, &l, &mask)?;
            let prob = Problem { family: l.family, x: x.clone(), y: fit::outcome(l.family, &y, &l.depvar).unwrap_or(y), w: vec![1.0; x.first().map_or(0, |c| c.len())], offset: None, loadings: l.loadings.clone(), alpha: 1.0, standardize: true };
            let (c0, cb) = if postsel { fit::postsel(&l, k, &prob) } else { (p.b0, p.b.clone()) };
            if standardized {
                let sds: Vec<f64> = x.iter().map(|c| {
                    let m = util::mean(c);
                    (c.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / c.len().max(1) as f64).sqrt()
                }).collect();
                (f64::NAN, cb.iter().zip(&sds).map(|(b, s)| b * s).collect())
            } else {
                (c0, cb)
            }
        } else {
            (p.b0, p.b.clone())
        };
        for j in 0..l.names.len() {
            if p.b[j] != 0.0 || (l.always[j] && postsel) {
                vals.push((l.names[j].clone(), b[j]));
            }
        }
        if b0.is_finite() {
            vals.push(("_cons".into(), b0));
        }
        let label = match (nm.as_str(), f) {
            (".", Some(v)) => v.clone(),
            (".", None) => "active".into(),
            (n, Some(v)) => format!("{n} {v}"),
            (n, None) => n.to_string(),
        };
        cols.push((label, vals));
    }
    let mut rows: Vec<String> = Vec::new();
    for (_, v) in &cols {
        for (n, _) in v {
            if n != "_cons" && !rows.contains(n) {
                rows.push(n.clone());
            }
        }
    }
    if coef {
        rows.push("_cons".into());
    }
    let w = cols.len() * 13 + 14;
    let mut out = vec![String::new(), "-".repeat(w)];
    let mut h = format!("{:>13}|", "");
    for (lab, _) in &cols {
        h.push_str(&format!(" {:>12}", abbrev(lab, 12)));
    }
    out.push(h);
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(w - 14)));
    let mut m = Matrix::zeros(rows.len(), cols.len());
    for (i, r) in rows.iter().enumerate() {
        let mut line = format!("{:>12} |", abbrev(r, 12));
        for (j, (_, v)) in cols.iter().enumerate() {
            let cell = v.iter().find(|(n, _)| n == r).map(|(_, b)| *b);
            m[(i, j)] = cell.unwrap_or(if coef { 0.0 } else { SYSMISS });
            let txt = match (cell, coef) {
                (Some(b), true) => fmt_g(b, 9),
                (Some(_), false) => "x".into(),
                (None, _) => String::new(),
            };
            line.push_str(&format!(" {txt:>12}"));
        }
        out.push(line);
    }
    out.push("-".repeat(w));
    if !coef && !cmd.has_opt("nolegend", 5) {
        out.push("Legend:".into());
        out.push("  x - estimated".into());
    }
    let mut r = Results::new();
    r.set_mat("coef", m.with_names(rows, cols.iter().map(|c| c.0.clone()).collect()));
    st.r = r;
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// lassogof
// ---------------------------------------------------------------------------

/// Goodness of fit of a stored lasso over rows in `mask`: (stat1, stat2,
/// N) = MSE and R-squared (linear) or deviance and deviance ratio.
pub fn gof(ds: &DataSet, st: &Store, l: &Lasso, mask: &[bool], fit_mask: &[bool], postsel: bool) -> Result<(f64, f64, usize), String> {
    let k = l.sel.ok_or_else(|| rc_err(498, "no lambda selected; use lassoselect"))?;
    let p = &l.pts[k];
    let (b0, b) = if postsel {
        let (_, x, y) = fit::columns(ds, st, l, fit_mask)?;
        let prob = Problem { family: l.family, x: x.clone(), y: fit::outcome(l.family, &y, &l.depvar)?, w: vec![1.0; y.len()], offset: None, loadings: l.loadings.clone(), alpha: 1.0, standardize: true };
        fit::postsel(l, k, &prob)
    } else {
        (p.b0, p.b.clone())
    };
    let (_, x, y) = fit::columns(ds, st, l, mask)?;
    let mut y2 = Vec::new();
    let mut eta = Vec::new();
    for i in 0..y.len() {
        if is_missing(y[i]) {
            continue;
        }
        y2.push(if matches!(l.family, Family::Logit | Family::Probit) { if y[i] != 0.0 { 1.0 } else { 0.0 } } else { y[i] });
        eta.push(b0 + b.iter().zip(&x).map(|(c, col)| c * col[i]).sum::<f64>());
    }
    let n = y2.len();
    if n == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    let dev: f64 = y2.iter().zip(&eta).map(|(a, e)| l.family.dev(*a, *e)).sum::<f64>() / n as f64;
    let ybar = util::mean(&y2);
    let e0 = match l.family {
        Family::Linear => ybar,
        Family::Logit => (ybar / (1.0 - ybar)).ln(),
        Family::Probit => crate::dist::invnormal(ybar),
        Family::Poisson => ybar.ln(),
    };
    let dev0: f64 = y2.iter().map(|a| l.family.dev(*a, e0)).sum::<f64>() / n as f64;
    Ok((dev, 1.0 - dev / dev0, n))
}

fn lassogof(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let names: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
    let names = if names.is_empty() { vec![".".to_string()] } else { names };
    let postsel = cmd.has_opt("postselection", 4);
    let has_ifin = cmd.if_expr.is_some() || cmd.in_range.is_some();
    let over = cmd.opt("over", 4).map(|s| s.trim().to_string());
    let mut out = vec![String::new()];
    out.push(format!("{} coefficients", if postsel { "Postselection" } else { "Penalized" }));
    out.push("-------------------------------------------------".into());
    let mut rows: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut lin = true;
    for nm in &names {
        let e = named(st, nm)?;
        need_lasso(&e)?;
        let l = lasso_for(&e, &cmd.opt("for", 3).map(|s| s.trim().to_string()))?;
        lin = l.family == Family::Linear;
        let sample = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
        let touse = util::touse(ds, st, cmd)?;
        let base: Vec<bool> = if has_ifin { touse.clone() } else { sample.clone() };
        match &over {
            Some(v) => {
                let c = util::col(ds, v)?;
                let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| base[i] && !is_missing(c[i])).map(|i| c[i]).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                for g in lv {
                    let m: Vec<bool> = (0..ds.n_obs).map(|i| base[i] && c[i] == g).collect();
                    let (a, b, n) = gof(ds, st, &l, &m, &sample, postsel)?;
                    rows.push((format!("{} {}", if nm == "." { "" } else { nm }, fmt_g(g, 8)).trim().to_string(), a, b, n));
                }
            }
            None => {
                let (a, b, n) = gof(ds, st, &l, &base, &sample, postsel)?;
                rows.push((if nm == "." { "active".into() } else { nm.clone() }, a, b, n));
            }
        }
    }
    out.push(format!("{:>12} | {:>12} {:>12} {:>10}", "Name", if lin { "MSE" } else { "Deviance" }, if lin { "R-squared" } else { "Dev. ratio" }, "Obs"));
    out.push("-------------+-----------------------------------".into());
    let mut m = Matrix::zeros(rows.len(), 3);
    for (i, (n, a, b, k)) in rows.iter().enumerate() {
        out.push(format!("{:>12} | {:>12} {:>12} {:>10}", abbrev(n, 12), fmt_g(*a, 8), format!("{b:.4}"), util::commas(*k)));
        m[(i, 0)] = *a;
        m[(i, 1)] = *b;
        m[(i, 2)] = *k as f64;
    }
    out.push("-------------------------------------------------".into());
    let mut r = Results::new();
    r.set_mat("table", m.with_names(rows.iter().map(|r| r.0.replace(' ', "_")).collect(), if lin { vec!["MSE".into(), "R-squared".into(), "Obs".into()] } else { vec!["deviance".into(), "deviance_ratio".into(), "Obs".into()] }));
    st.r = r;
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// lassoselect
// ---------------------------------------------------------------------------

fn lassoselect(_ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    need_lasso(&st.e)?;
    let (k, mut l) = fit::load_one(&st.e, cmd)?;
    let text = cmd.main.replace('=', " = ");
    let w: Vec<&str> = text.split_whitespace().collect();
    let mut alpha: Option<f64> = None;
    let mut id: Option<usize> = None;
    let mut lambda: Option<f64> = None;
    let mut i = 0;
    while i < w.len() {
        let key = w[i].to_lowercase();
        if w.get(i + 1) != Some(&"=") {
            return Err(rc_err(198, "invalid syntax"));
        }
        let v: f64 = w.get(i + 2).and_then(|s| s.parse().ok()).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        match key.as_str() {
            "id" => id = Some(v as usize),
            "lambda" => lambda = Some(v),
            "alpha" => alpha = Some(v),
            o => return Err(rc_err(198, format!("{o} not allowed"))),
        }
        i += 3;
    }
    let pick = match (id, lambda) {
        (Some(i), _) => {
            if i == 0 || i > l.pts.len() {
                return Err(rc_err(198, format!("ID {i} not found")));
            }
            i - 1
        }
        (None, Some(lam)) => {
            let a = alpha.or_else(|| l.pts.first().map(|p| p.alpha)).unwrap_or(1.0);
            let cands: Vec<usize> = (0..l.pts.len()).filter(|&j| (l.pts[j].alpha - a).abs() < 1e-9).collect();
            if cands.is_empty() {
                return Err(rc_err(198, format!("alpha = {} not found", fmt_g(a, 6))));
            }
            let lt = lam.max(1e-300).ln();
            *cands.iter().min_by(|&&x, &&y| (l.pts[x].lambda.ln() - lt).abs().total_cmp(&(l.pts[y].lambda.ln() - lt).abs())).unwrap()
        }
        _ => return Err(rc_err(198, "id = # or lambda = # required")),
    };
    l.sel = Some(pick);
    l.how = "user".into();
    st.e.set_str(&format!("_lasso{k}"), l.encode());
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if matches!(ecmd.as_str(), "lasso" | "elasticnet" | "sqrtlasso") {
        super::cmd::set_selected(&mut st.e, &l);
    }
    let p = &l.pts[pick];
    let mut out = vec![format!("ID = {}  lambda = {} selected", pick + 1, fmt_g(p.lambda, 7))];
    if !matches!(ecmd.as_str(), "lasso" | "elasticnet" | "sqrtlasso") {
        out.push(format!("(type {ecmd}, reestimate  to refit with the new selection)"));
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------

fn plot(ds: &mut DataSet, st: &mut Store, cmd: &Command, verb: &str) -> Result<String, String> {
    need_lasso(&st.e)?;
    let (_, l) = fit::load_one(&st.e, cmd)?;
    let a0 = l.selected().map(|p| p.alpha).or(l.pts.first().map(|p| p.alpha)).unwrap_or(1.0);
    let pts: Vec<&fit::Pt> = l.pts.iter().filter(|p| p.alpha == a0).collect();
    let mut g = Graph { legend: false, ..Default::default() };
    match verb {
        "cvplot" | "bicplot" => {
            let cv = verb == "cvplot";
            if cv && !pts.iter().any(|p| p.cvm.is_finite()) {
                return Err(rc_err(498, "cvplot requires selection(cv) or selection(adaptive)"));
            }
            let val = |p: &fit::Pt| if cv { p.cvm } else { p.bic };
            let line: Vec<(f64, f64)> = pts.iter().filter(|p| val(p).is_finite()).map(|p| (p.lambda.ln(), val(p))).collect();
            g.layers.push(Layer { kind: LayerKind::Connected, points: line, label: if cv { "CV function".into() } else { "BIC".into() }, ..Default::default() });
            g.xtitle = "ln(lambda)".into();
            g.ytitle = if cv { if l.family == Family::Linear { "Cross-validation function (mean prediction error)".into() } else { "Cross-validation function (mean deviance)".into() } } else { "Bayesian information criterion".into() };
            g.title = if cv { "Cross-validation plot".into() } else { "BIC plot".into() };
            if let Some(p) = l.selected() {
                g.xlines.push(p.lambda.ln());
            }
        }
        _ => {
            let mask = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
            let (_, x, _) = fit::columns(ds, st, &l, &mask)?;
            let sds: Vec<f64> = x.iter().map(|c| {
                let m = util::mean(c);
                (c.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / c.len().max(1) as f64).sqrt()
            }).collect();
            let xunits = cmd.opt("xunits", 2).unwrap_or("l1norm").trim().to_lowercase();
            let xs: Vec<f64> = pts
                .iter()
                .map(|p| match xunits.as_str() {
                    "lnlambda" => p.lambda.ln(),
                    _ => p.b.iter().zip(&sds).map(|(b, s)| (b * s).abs()).sum(),
                })
                .collect();
            let maxl1 = xs.iter().cloned().fold(0.0, f64::max).max(1e-300);
            for j in 0..l.names.len() {
                if pts.iter().all(|p| p.b[j] == 0.0) {
                    continue;
                }
                let line: Vec<(f64, f64)> = pts.iter().zip(&xs).map(|(p, x)| (if xunits == "rl1norm" { x / maxl1 } else { *x }, p.b[j] * sds[j])).collect();
                g.layers.push(Layer { kind: LayerKind::Line, points: line, label: l.names[j].clone(), ..Default::default() });
            }
            g.xtitle = match xunits.as_str() {
                "lnlambda" => "ln(lambda)".into(),
                "rl1norm" => "Relative L1 norm of standardized coefficient vector".into(),
                _ => "L1 norm of standardized coefficient vector".into(),
            };
            g.ytitle = "Standardized coefficients".into();
            g.title = "Coefficient paths".into();
            g.legend = cmd.has_opt("legend", 3);
        }
    }
    g.command = format!("{verb} {}", cmd.main).trim().to_string();
    let extra = crate::graph::finish(&mut g, cmd, st)?;
    Ok(extra)
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
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
    let (_, l) = fit::load_one(&st.e, cmd)?;
    let k = l.sel.ok_or_else(|| rc_err(498, "no lambda selected; use lassoselect"))?;
    let postsel = cmd.has_opt("postselection", 4);
    let mut stat = match l.family {
        Family::Linear => "xb",
        Family::Logit | Family::Probit => "pr",
        Family::Poisson => "n",
    };
    let mut announced = true;
    for (o, _) in &cmd.opts {
        match o.to_lowercase().as_str() {
            "xb" => {
                stat = "xb";
                announced = false
            }
            "pr" => {
                stat = "pr";
                announced = false
            }
            "n" => {
                stat = "n";
                announced = false
            }
            "ir" => {
                stat = "ir";
                announced = false
            }
            "penalized" | "postselection" | "pen" | "post" | "postsel" | "nooffset" => {}
            o => return Err(rc_err(198, format!("option {o} not allowed"))),
        }
    }
    let p = &l.pts[k];
    let (b0, b) = if postsel {
        let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
        let (_, x, y) = fit::columns(ds, st, &l, &sample)?;
        let prob = Problem { family: l.family, x, y: fit::outcome(l.family, &y, &l.depvar)?, w: vec![1.0; y.len()], offset: None, loadings: l.loadings.clone(), alpha: 1.0, standardize: true };
        fit::postsel(&l, k, &prob)
    } else {
        (p.b0, p.b.clone())
    };
    let mask = util::touse(ds, st, cmd)?;
    let env = crate::eval::Env::with_store(ds, st);
    let mut vals = vec![SYSMISS; ds.n_obs];
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let mut eta = b0;
        let mut miss = false;
        for j in 0..l.names.len() {
            if b[j] == 0.0 {
                continue;
            }
            let v = crate::postest::coef_value(&env, &l.names[j], i)?;
            if is_missing(v) {
                miss = true;
                break;
            }
            eta += b[j] * v;
        }
        if miss {
            continue;
        }
        vals[i] = match stat {
            "xb" => eta,
            "pr" => l.family.mean(eta),
            "n" | "ir" => l.family.mean(eta),
            _ => eta,
        };
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut out = Vec::new();
    if announced {
        let what = match stat {
            "xb" => "linear prediction".to_string(),
            "pr" => format!("Pr({})", l.depvar),
            _ => "number of events".to_string(),
        };
        out.push(format!("(options {stat} {} assumed; {what})", if postsel { "postselection" } else { "penalized" }));
    }
    if nmiss > 0 {
        out.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(out.join("\n"))
}
