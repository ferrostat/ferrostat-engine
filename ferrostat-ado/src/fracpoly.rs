//! Fractional polynomials: the `fp` prefix (with `fp generate`) and
//! multivariable fractional polynomials (`mfp`).
//!
//! A fractional polynomial of degree m in x > 0 uses powers p1 <= ... <= pm
//! from a small set (default -2, -1, -.5, 0, .5, 1, 2, 3); power 0 means
//! ln x and a repeated power multiplies the previous term by ln x
//! (Royston and Altman 1994). Models are compared by deviance.

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_parser::parser::{find_top_level, split_options, split_top_level_words};

use crate::interpreter::{split_verb, AdoInterpreter, Flow};
use crate::resample::{self, arg, has, num_arg};

pub(crate) fn intercept(me: &mut AdoInterpreter, verb: &str, rest: &str) -> Result<Option<Flow>, String> {
    match verb {
        "fp" => {
            let r = rest.trim_start();
            let first = r.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").to_lowercase();
            if !r.starts_with('<') && !first.is_empty() && "generate".starts_with(first.as_str()) && first.len() >= 3 {
                fp_generate(me, r[first.len()..].trim())?;
            } else if first == "plot" || first == "predict" {
                fp_post(me, &first, r[first.len()..].trim())?;
            } else {
                fp_prefix(me, r)?;
            }
            Ok(Some(Flow::Next))
        }
        "mfp" => {
            mfp(me, rest)?;
            Ok(Some(Flow::Next))
        }
        _ => Ok(None),
    }
}

const DEFAULT_POWERS: [f64; 8] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];

fn pstr(p: f64) -> String {
    fmt_g(p, 8)
}

/// Fractional polynomial terms of x for the given powers.
pub(crate) fn fp_terms(x: f64, powers: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(powers.len());
    let mut prev: Option<(f64, f64)> = None; // (power, last term)
    for &p in powers {
        let base = if p == 0.0 { x.ln() } else { x.powf(p) };
        let t = match prev {
            Some((pp, last)) if pp == p => last * x.ln(),
            _ => base,
        };
        out.push(t);
        prev = Some((p, t));
    }
    out
}

/// All power combinations of degree m (nondecreasing).
fn combos(set: &[f64], m: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    fn rec(set: &[f64], m: usize, start: usize, cur: &mut Vec<f64>, out: &mut Vec<Vec<f64>>) {
        if cur.len() == m {
            out.push(cur.clone());
            return;
        }
        for i in start..set.len() {
            cur.push(set[i]);
            rec(set, m, i, cur, out);
            cur.pop();
        }
    }
    rec(set, m, 0, &mut cur, &mut out);
    out
}

/// Scaling of a term: x' = (x + a) / b.
#[derive(Clone, Copy)]
struct Scale {
    a: f64,
    b: f64,
}

fn auto_scale(x: &[f64]) -> Scale {
    let mut v: Vec<f64> = x.iter().cloned().filter(|z| !is_missing(*z)).collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    if v.is_empty() {
        return Scale { a: 0.0, b: 1.0 };
    }
    let (mn, mx) = (v[0], v[v.len() - 1]);
    let lr = (mx - mn).max(1e-300).log10();
    let b = 10f64.powf(lr.signum() * lr.abs().floor());
    let a = if mn <= 0.0 {
        let gap = v.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
        let gap = if gap.is_finite() { gap } else { 1.0 };
        -mn + gap
    } else {
        0.0
    };
    Scale { a, b }
}

struct FpSpec {
    term: String,
    scale: Scale,
    center: Option<f64>,
    zero: bool,
    catzero: bool,
}

impl FpSpec {
    /// Transformed column(s) for powers over all observations.
    fn columns(&self, x: &[f64], powers: &[f64]) -> Vec<Vec<f64>> {
        let k = powers.len();
        let mut cols = vec![vec![SYSMISS; x.len()]; k];
        let cvals = self.center.map(|c| fp_terms((c + self.scale.a) / self.scale.b, powers));
        for (i, &xv) in x.iter().enumerate() {
            if is_missing(xv) {
                continue;
            }
            let s = (xv + self.scale.a) / self.scale.b;
            if s <= 0.0 {
                if self.zero || self.catzero {
                    for c in cols.iter_mut() {
                        c[i] = 0.0;
                    }
                }
                continue;
            }
            let t = fp_terms(s, powers);
            for j in 0..k {
                cols[j][i] = t[j] - cvals.as_ref().map_or(0.0, |c| c[j]);
            }
        }
        cols
    }
}

fn set_var(me: &mut AdoInterpreter, name: &str, vals: Vec<f64>) -> Result<(), String> {
    let ds = &mut me.engine.dataset;
    if ds.var_index.contains_key(name) {
        ds.drop_variable(name).map_err(|e| rc_err(110, e))?;
    }
    ds.add_variable(util::new_double(name, vals)).map_err(|e| rc_err(110, e))
}

fn drop_vars(me: &mut AdoInterpreter, names: &[String]) {
    for n in names {
        if me.engine.dataset.var_index.contains_key(n) {
            let _ = me.engine.dataset.drop_variable(n);
        }
    }
}

/// Deviance, residual s.d. and N of the last estimation.
fn fit_stats(me: &AdoInterpreter) -> (f64, f64, f64, bool) {
    let e = &me.engine.store.e;
    let n = e.get("N").unwrap_or(SYSMISS);
    let is_reg = e.macros.get("cmd").is_some_and(|c| c == "regress");
    if is_reg {
        let rss = e.get("rss").unwrap_or(SYSMISS);
        let dev = n * (1.0 + (2.0 * std::f64::consts::PI).ln() + (rss / n).ln());
        (dev, e.get("rmse").unwrap_or(SYSMISS), n, true)
    } else {
        (-2.0 * e.get("ll").unwrap_or(SYSMISS), SYSMISS, n, false)
    }
}

fn parse_numlist(s: &str) -> Result<Vec<f64>, String> {
    s.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).map(|t| t.parse::<f64>().map_err(|_| rc_err(198, format!("{t} invalid in numlist")))).collect()
}

fn fp_prefix(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "fp: estimation command required"))?;
    let head = rest[..colon].trim();
    let est_cmd = rest[colon + 1..].trim().to_string();
    let (termpart, optpart) = match find_top_level(head, ',') {
        Some(i) => (head[..i].trim(), head[i + 1..].trim()),
        None => (head, ""),
    };
    let term = termpart.trim().trim_start_matches('<').trim_end_matches('>').trim().to_string();
    let term = varlist::resolve_name(&me.engine.dataset, &term)?;
    if !est_cmd.contains(&format!("<{}>", termpart.trim().trim_start_matches('<').trim_end_matches('>').trim())) {
        return Err(rc_err(198, format!("<{term}> not found in estimation command")));
    }
    let placeholder = format!("<{}>", termpart.trim().trim_start_matches('<').trim_end_matches('>').trim());
    let opts = split_options(optpart);
    let x = util::col(&me.engine.dataset, &term)?;
    let scale = if let Some(s) = arg(&opts, "scale", 2) {
        let v = parse_numlist(&s)?;
        if v.len() != 2 {
            return Err(rc_err(198, "scale() requires two numbers"));
        }
        Scale { a: v[0], b: v[1] }
    } else if has(&opts, "scale", 2) {
        auto_scale(&x)
    } else {
        Scale { a: 0.0, b: 1.0 }
    };
    let zero = has(&opts, "zero", 4);
    let catzero = has(&opts, "catzero", 4);
    let center = if let Some(c) = arg(&opts, "center", 3) {
        Some(c.trim().parse::<f64>().map_err(|_| rc_err(198, "center() invalid"))?)
    } else if has(&opts, "center", 3) {
        let v: Vec<f64> = x.iter().cloned().filter(|z| !is_missing(*z)).collect();
        Some(util::mean(&v))
    } else {
        None
    };
    let spec = FpSpec { term: term.clone(), scale, center, zero, catzero };
    if !zero && !catzero && x.iter().any(|v| !is_missing(*v) && (v + scale.a) / scale.b <= 0.0) {
        return Err(rc_err(411, format!("{term} has nonpositive values; use the scale, zero or catzero option")));
    }
    let replace = has(&opts, "replace", 3);
    let dim: usize = num_arg(&opts, "dimension", 3)?.map_or(2, |d| d as usize).max(1);
    let powers: Vec<f64> = match arg(&opts, "powers", 3) {
        Some(p) => parse_numlist(&p)?,
        None => DEFAULT_POWERS.to_vec(),
    };
    let fixed: Option<Vec<f64>> = arg(&opts, "fp", 2).map(|p| parse_numlist(&p)).transpose()?.map(|mut v| {
        v.sort_by(f64::total_cmp);
        v
    });
    let zname = format!("{term}_0");
    let names_for = |k: usize| -> Vec<String> { (1..=k).map(|i| format!("{term}_{i}")).collect() };
    let maxk = fixed.as_ref().map_or(dim, |f| f.len());
    if !replace {
        for n in names_for(maxk) {
            if me.engine.dataset.var_index.contains_key(&n) {
                return Err(rc_err(110, format!("variable {n} already defined; use the replace option")));
            }
        }
    }
    let line_for = |names: &[String]| -> String {
        let mut all = names.to_vec();
        if catzero {
            all.insert(0, zname.clone());
        }
        est_cmd.replace(&placeholder, &all.join(" "))
    };
    if catzero {
        let zv: Vec<f64> = x.iter().map(|v| if is_missing(*v) { SYSMISS } else if (v + scale.a) / scale.b <= 0.0 { 1.0 } else { 0.0 }).collect();
        set_var(me, &zname, zv)?;
    }
    let generate = |me: &mut AdoInterpreter, pw: &[f64]| -> Result<Vec<String>, String> {
        let names = names_for(pw.len());
        for (n, c) in names.iter().zip(spec.columns(&x, pw)) {
            set_var(me, n, c)?;
        }
        Ok(names)
    };
    if let Some(pw) = fixed {
        drop_vars(me, &names_for(dim.max(pw.len()) + 2));
        let names = generate(me, &pw)?;
        let line = line_for(&names);
        me.emit(&format!("-> {line}"));
        me.exec_text(&line, None)?;
        set_fp_results(me, &term, &pw, scale, center);
        return Ok(());
    }
    // the search: omitted, linear, then the best model of each degree
    let mut models = 0usize;
    for m in 1..=dim {
        models += combos(&powers, m).len();
    }
    me.emit(&format!("(fitting {} models)", util::commas(models)));
    me.emit("(....10%....20%....30%....40%....50%....60%....70%....80%....90%....100%)");
    drop_vars(me, &names_for(dim + 2));
    // omitted: the term removed from the command
    let omit_line = line_for(&[]);
    let omit_line = if catzero { omit_line.replace(&zname, "") } else { omit_line };
    resample::run_cmd(me, &omit_line, false)?;
    let (dev0, rsd0, n, is_reg) = fit_stats(me);
    let k_other = me.engine.store.e.get("rank").unwrap_or(0.0);
    let names = generate(me, &[1.0])?;
    resample::run_cmd(me, &line_for(&names), false)?;
    let (dev_lin, rsd_lin, _, _) = fit_stats(me);
    let mut best: Vec<(Vec<f64>, f64, f64)> = Vec::new();
    for m in 1..=dim {
        let mut b: Option<(Vec<f64>, f64, f64)> = None;
        for pw in combos(&powers, m) {
            drop_vars(me, &names_for(dim + 2));
            let names = generate(me, &pw)?;
            if resample::run_cmd(me, &line_for(&names), false).is_err() {
                continue;
            }
            let (dev, rsd, _, _) = fit_stats(me);
            if dev.is_finite() && b.as_ref().is_none_or(|x| dev < x.1 - 1e-10) {
                b = Some((pw.clone(), dev, rsd));
            }
        }
        best.push(b.ok_or_else(|| rc_err(430, "no fractional polynomial model could be fit"))?);
    }
    let (bp, bdev, _) = best[dim - 1].clone();
    let df_best = 2.0 * dim as f64;
    let dfres = n - k_other - df_best;
    let ptest = |dev: f64, df: f64| -> f64 {
        if df <= 0.0 {
            return SYSMISS;
        }
        let d = dev - bdev;
        if is_reg {
            let f = ((d / n).exp() - 1.0) * dfres / df;
            dist::ftail(df, dfres, f.max(0.0))
        } else {
            dist::chi2tail(df, d.max(0.0))
        }
    };
    let bar = "-".repeat(78);
    let mut out = vec![String::new(), "Fractional polynomial comparisons:".to_string(), bar.clone()];
    out.push(format!("{:>12} |   Test{:>14}Residual   Deviance", "", if is_reg { "" } else { "" }));
    out.push(format!("{:>12} |     df   Deviance   std. dev.      diff.       P   Powers", ferrostat_cmd::summarize::abbrev(&term, 12)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    let row = |label: &str, df: f64, dev: f64, rsd: f64, pw: &str| -> String {
        let p = ptest(dev, df);
        format!(
            "{:>12} | {:>6} {:>10.3} {:>10} {:>10.3} {:>8}   {}",
            label,
            fmt_g(df, 6),
            dev,
            if is_missing(rsd) { String::new() } else { format!("{rsd:.3}") },
            dev - bdev,
            if is_missing(p) { "--".to_string() } else { format!("{p:.3}") },
            pw
        )
        .trim_end()
        .to_string()
    };
    out.push(row("omitted", df_best, dev0, rsd0, ""));
    out.push(row("linear", df_best - 1.0, dev_lin, rsd_lin, "1"));
    for (m, (pw, dev, rsd)) in best.iter().enumerate() {
        let txt: Vec<String> = pw.iter().map(|p| pstr(*p)).collect();
        out.push(row(&format!("m = {}", m + 1), df_best - 2.0 * (m + 1) as f64, *dev, *rsd, &txt.join(" ")));
    }
    out.push(bar);
    out.push(format!("Note: Test df is degrees of freedom, and P = P > {} is sig. level for tests", if is_reg { "F" } else { "chi2" }));
    out.push(format!("      comparing models vs. model with m = {dim} based on deviance difference,"));
    out.push(if is_reg { format!("      F(df, {}).", fmt_g(dfres, 9)) } else { "      chi2(df).".to_string() });
    me.emit(&out.join("\n"));
    let _ = (rsd0, rsd_lin);
    drop_vars(me, &names_for(dim + 2));
    let names = generate(me, &bp)?;
    me.exec_text(&line_for(&names), None)?;
    set_fp_results(me, &term, &bp, scale, center);
    Ok(())
}

fn set_fp_results(me: &mut AdoInterpreter, term: &str, pw: &[f64], scale: Scale, center: Option<f64>) {
    let e = &mut me.engine.store.e;
    e.set_str("fp_cmd", "fp, search:");
    e.set_str("fp_cmd2", format!("fp <{term}>"));
    e.set_str("fp_terms", term.to_string());
    e.set_str("fp_powers", pw.iter().map(|p| pstr(*p)).collect::<Vec<_>>().join(" "));
    e.set("fp_scale_a", scale.a);
    e.set("fp_scale_b", scale.b);
    if let Some(c) = center {
        e.set("fp_center_mean", c);
    }
}

/// `fp generate [type] [newvar =] varname^(numlist) [, scale() center() replace]`.
fn fp_generate(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (main, optpart) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim(), rest[i + 1..].trim()),
        None => (rest.trim(), ""),
    };
    let main = main.trim_start_matches("double ").trim_start_matches("float ").trim();
    let (newname, expr) = match main.split_once('=') {
        Some((a, b)) => (Some(a.trim().to_string()), b.trim()),
        None => (None, main),
    };
    let (var, pw) = expr.split_once('^').ok_or_else(|| rc_err(198, "varname^(powers) required"))?;
    let var = varlist::resolve_name(&me.engine.dataset, var.trim())?;
    let mut powers = parse_numlist(pw.trim().trim_start_matches('(').trim_end_matches(')'))?;
    powers.sort_by(f64::total_cmp);
    let opts = split_options(optpart);
    let x = util::col(&me.engine.dataset, &var)?;
    let scale = match arg(&opts, "scale", 2) {
        Some(s) => {
            let v = parse_numlist(&s)?;
            Scale { a: v.first().copied().unwrap_or(0.0), b: v.get(1).copied().unwrap_or(1.0) }
        }
        None if has(&opts, "scale", 2) => auto_scale(&x),
        None => Scale { a: 0.0, b: 1.0 },
    };
    let center = match arg(&opts, "center", 3) {
        Some(c) => Some(c.trim().parse::<f64>().map_err(|_| rc_err(198, "center() invalid"))?),
        None if has(&opts, "center", 3) => Some(util::mean(&x.iter().cloned().filter(|v| !is_missing(*v)).collect::<Vec<_>>())),
        None => None,
    };
    let spec = FpSpec { term: var.clone(), scale, center, zero: has(&opts, "zero", 4), catzero: has(&opts, "catzero", 4) };
    if !spec.zero && !spec.catzero && x.iter().any(|v| !is_missing(*v) && (v + scale.a) / scale.b <= 0.0) {
        return Err(rc_err(411, format!("{var} has nonpositive values")));
    }
    let stub = newname.unwrap_or_else(|| var.clone());
    let replace = has(&opts, "replace", 3);
    let cols = spec.columns(&x, &powers);
    let _ = &spec.term;
    for (i, c) in cols.into_iter().enumerate() {
        let n = format!("{stub}_{}", i + 1);
        if me.engine.dataset.var_index.contains_key(&n) && !replace {
            return Err(rc_err(110, format!("variable {n} already defined")));
        }
        set_var(me, &n, c)?;
    }
    Ok(())
}

/// `fp predict newvar` (the fractional polynomial part of the linear
/// predictor) and `fp plot` (component-plus-residual plot).
fn fp_post(me: &mut AdoInterpreter, sub: &str, rest: &str) -> Result<(), String> {
    let term = me.engine.store.e.macros.get("fp_terms").cloned().ok_or_else(|| rc_err(301, "fp results not found"))?;
    let b = me.engine.store.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let cols: Vec<(usize, String)> = b.colnames.iter().enumerate().filter(|(_, c)| c.rsplit(':').next().is_some_and(|n| n.starts_with(&format!("{term}_")))).map(|(j, c)| (j, c.rsplit(':').next().unwrap().to_string())).collect();
    let n = me.engine.dataset.n_obs;
    let mut fpv = vec![0.0; n];
    for (j, c) in &cols {
        let v = util::col(&me.engine.dataset, c)?;
        for i in 0..n {
            fpv[i] = if is_missing(v[i]) || is_missing(fpv[i]) { SYSMISS } else { fpv[i] + b.get(0, *j) * v[i] };
        }
    }
    if sub == "predict" {
        let (main, _) = match find_top_level(rest, ',') {
            Some(i) => (rest[..i].trim(), rest[i + 1..].trim()),
            None => (rest.trim(), ""),
        };
        let words = split_top_level_words(main);
        let name = words.iter().find(|w| !matches!(w.as_str(), "double" | "float")).cloned().ok_or_else(|| rc_err(100, "new variable name required"))?;
        let mask = me.engine.store.e.sample.clone().unwrap_or_else(|| vec![true; n]);
        let vals: Vec<f64> = (0..n).map(|i| if mask.get(i).copied().unwrap_or(true) { fpv[i] } else { fpv[i] }).collect();
        set_var(me, &name, vals)?;
        return Ok(());
    }
    // plot: fitted function against the term
    let tmp = "__fp_fit";
    set_var(me, tmp, fpv)?;
    let r = me.exec_text(&format!("twoway line {tmp} {term}, sort ytitle(\"Partial predictor\")"), None);
    let _ = me.engine.dataset.drop_variable(tmp);
    r.map(|_| ())
}

// ---------------------------------------------------------------------------
// mfp
// ---------------------------------------------------------------------------

struct MfpVar {
    name: String,
    df: usize,
    select: f64,
    alpha: f64,
    /// current powers (empty: excluded; [1]: linear)
    powers: Vec<f64>,
    scale: Scale,
    vars: Vec<String>,
}

fn per_var(spec: &Option<String>, name: &str) -> Option<String> {
    // "4, x1:2 x2:1" style lists: a default and per-variable values
    let s = spec.as_ref()?;
    let mut default = None;
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        for tok in part.split_whitespace() {
            match tok.split_once(':') {
                Some((v, val)) => {
                    if v.split_whitespace().any(|x| x == name) {
                        return Some(val.to_string());
                    }
                }
                None => default = Some(tok.to_string()),
            }
        }
        if let Some((vars, val)) = part.rsplit_once(':') {
            if vars.split_whitespace().any(|x| x == name) {
                return Some(val.trim().to_string());
            }
        }
    }
    default
}

fn mfp(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "mfp: estimation command required"))?;
    let head = rest[..colon].trim().trim_start_matches(',');
    let opts = split_options(head);
    let est = rest[colon + 1..].trim();
    let (verb, args) = split_verb(est);
    // depvar(s) then covariates up to if/in/weight/options
    let (vpart, tail) = {
        let mut cut = args.len();
        for key in [" if ", " in ", "[", ","] {
            if let Some(i) = args.find(key) {
                cut = cut.min(i);
            }
        }
        (args[..cut].trim().to_string(), args[cut..].to_string())
    };
    let words: Vec<String> = split_top_level_words(&vpart);
    let ndep = if matches!(verb.as_str(), "stcox" | "streg") { 0 } else { 1 };
    if words.len() <= ndep {
        return Err(rc_err(102, "too few variables specified"));
    }
    let deps: Vec<String> = words[..ndep].to_vec();
    let alpha_s = arg(&opts, "alpha", 1);
    let select_s = arg(&opts, "select", 3);
    let df_s = arg(&opts, "df", 2);
    let powers_s = arg(&opts, "xpowers", 2);
    let cycles: usize = num_arg(&opts, "cycles", 2)?.map_or(5, |c| c as usize);
    let mut items: Vec<MfpVar> = Vec::new();
    for w in &words[ndep..] {
        let (grouped, names) = if w.starts_with('(') {
            (true, varlist::expand(&me.engine.dataset, w.trim_matches(|c| c == '(' || c == ')'))?)
        } else {
            (false, varlist::expand(&me.engine.dataset, w)?)
        };
        let units: Vec<Vec<String>> = if grouped { vec![names] } else { names.into_iter().map(|n| vec![n]).collect() };
        for u in units {
            let name = u.join(" ");
            let x = util::col(&me.engine.dataset, &u[0])?;
            let mut lv: Vec<f64> = x.iter().cloned().filter(|v| !is_missing(*v)).collect();
            lv.sort_by(f64::total_cmp);
            lv.dedup();
            let default_df = if u.len() > 1 || lv.len() <= 2 { 1 } else if lv.len() <= 3 { 2 } else { 4 };
            let df = per_var(&df_s, &name).and_then(|s| s.parse::<usize>().ok()).unwrap_or(default_df);
            let select = per_var(&select_s, &name).and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
            let alpha = per_var(&alpha_s, &name).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.05);
            let scale = if u.len() == 1 && lv.first().is_some_and(|m| *m <= 0.0) && df > 1 { auto_scale(&x) } else { Scale { a: 0.0, b: 1.0 } };
            items.push(MfpVar { name, df, select, alpha, powers: vec![1.0], scale, vars: u });
        }
    }
    let powers: Vec<f64> = match &powers_s {
        Some(p) if !p.contains(':') => parse_numlist(p)?,
        _ => DEFAULT_POWERS.to_vec(),
    };
    let tname = |v: &str, k: usize| -> String {
        let s: String = v.chars().take(25).collect();
        format!("I{s}__{k}")
    };
    // build the variables of one item for given powers; returns their names
    let make = |me: &mut AdoInterpreter, it: &MfpVar, pw: &[f64]| -> Result<Vec<String>, String> {
        if it.vars.len() > 1 || it.df == 1 {
            return Ok(if pw.is_empty() { vec![] } else { it.vars.clone() });
        }
        if pw == [1.0] && it.scale.a == 0.0 && it.scale.b == 1.0 {
            return Ok(it.vars.clone());
        }
        let x = util::col(&me.engine.dataset, &it.vars[0])?;
        let spec = FpSpec { term: it.vars[0].clone(), scale: it.scale, center: None, zero: false, catzero: false };
        let mut names = Vec::new();
        for (k, c) in spec.columns(&x, pw).into_iter().enumerate() {
            let n = tname(&it.vars[0], k + 1);
            set_var(me, &n, c)?;
            names.push(n);
        }
        Ok(names)
    };
    let fit = |me: &mut AdoInterpreter, items: &[MfpVar], override_i: Option<(usize, &[f64])>| -> Result<(f64, f64, f64, bool), String> {
        let mut cov = Vec::new();
        for (i, it) in items.iter().enumerate() {
            let pw: Vec<f64> = match override_i {
                Some((j, p)) if j == i => p.to_vec(),
                _ => it.powers.clone(),
            };
            if pw.is_empty() {
                continue;
            }
            let names = if it.vars.len() == 1 && it.df > 1 && !(pw == [1.0] && it.scale.a == 0.0 && it.scale.b == 1.0) {
                let x = util::col(&me.engine.dataset, &it.vars[0])?;
                let spec = FpSpec { term: it.vars[0].clone(), scale: it.scale, center: None, zero: false, catzero: false };
                let mut names = Vec::new();
                for (k, c) in spec.columns(&x, &pw).into_iter().enumerate() {
                    let n = format!("__mfp{}_{}", i, k + 1);
                    set_var(me, &n, c)?;
                    names.push(n);
                }
                names
            } else {
                it.vars.clone()
            };
            cov.extend(names);
        }
        let line = format!("{verb} {} {}{}", deps.join(" "), cov.join(" "), tail);
        resample::run_cmd(me, &line, false)?;
        let r = fit_stats(me);
        Ok(r)
    };
    let cleanup = |me: &mut AdoInterpreter| {
        let names: Vec<String> = me.engine.dataset.varlist().into_iter().filter(|n| n.starts_with("__mfp")).collect();
        drop_vars(me, &names);
    };
    // order: by the p-value of each term in the full linear model
    let res = (|| -> Result<usize, String> {
        let (dev_full, _, n, is_reg) = fit(me, &items, None)?;
        let _ = dev_full;
        let mut order: Vec<(usize, f64)> = Vec::new();
        for i in 0..items.len() {
            let (d0, _, _, _) = fit(me, &items, Some((i, &[])))?;
            let (d1, _, _, _) = fit(me, &items, None)?;
            order.push((i, d0 - d1));
        }
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        let order: Vec<usize> = order.into_iter().map(|x| x.0).collect();
        let pval = |d: f64, df: f64, dfres: f64| -> f64 {
            if is_reg {
                let f = ((d / n).exp() - 1.0) * dfres / df;
                dist::ftail(df, dfres, f.max(0.0))
            } else {
                dist::chi2tail(df, d.max(0.0))
            }
        };
        let mut ncycles = 0;
        for _ in 0..cycles {
            ncycles += 1;
            let mut changed = false;
            for &i in &order {
                let it = &items[i];
                let (d_null, _, _, _) = fit(me, &items, Some((i, &[])))?;
                let rank_null = me.engine.store.e.get("rank").unwrap_or(0.0);
                let new_pw: Vec<f64> = if it.df == 1 || it.vars.len() > 1 {
                    let (d_lin, _, _, _) = fit(me, &items, Some((i, &[1.0])))?;
                    let df = it.vars.len() as f64;
                    if pval(d_null - d_lin, df, n - rank_null - df) > it.select { vec![] } else { vec![1.0] }
                } else {
                    let m_max = it.df / 2;
                    let mut best_m: Vec<(Vec<f64>, f64)> = Vec::new();
                    for m in 1..=m_max {
                        let mut b: Option<(Vec<f64>, f64)> = None;
                        for pw in combos(&powers, m) {
                            if let Ok((d, _, _, _)) = fit(me, &items, Some((i, &pw))) {
                                if d.is_finite() && b.as_ref().is_none_or(|x| d < x.1 - 1e-10) {
                                    b = Some((pw, d));
                                }
                            }
                        }
                        if let Some(b) = b {
                            best_m.push(b);
                        }
                    }
                    let (top_pw, top_d) = best_m.last().cloned().ok_or_else(|| rc_err(430, "no model could be fit"))?;
                    let dfr = n - rank_null - 2.0 * m_max as f64;
                    let (d_lin, _, _, _) = fit(me, &items, Some((i, &[1.0])))?;
                    if pval(d_null - top_d, 2.0 * m_max as f64, dfr) > it.select {
                        vec![]
                    } else if pval(d_lin - top_d, 2.0 * m_max as f64 - 1.0, dfr) > it.alpha {
                        vec![1.0]
                    } else {
                        // step down through lower degrees
                        let mut chosen = top_pw.clone();
                        for m in (1..m_max).rev() {
                            let (pw_m, d_m) = &best_m[m - 1];
                            if pval(d_m - top_d, 2.0 * (m_max - m) as f64, dfr) > it.alpha {
                                chosen = pw_m.clone();
                            } else {
                                break;
                            }
                        }
                        chosen
                    }
                };
                if new_pw != items[i].powers {
                    items[i].powers = new_pw;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(ncycles)
    })();
    cleanup(me);
    let ncycles = res?;
    // final model with named transformed variables
    let mut cov = Vec::new();
    let mut gen_lines = Vec::new();
    for it in &items {
        if it.powers.is_empty() {
            continue;
        }
        let names = make(me, it, &it.powers)?;
        if names.first() != it.vars.first() {
            for (k, n) in names.iter().enumerate() {
                let p = it.powers[k];
                let base = if it.scale.a == 0.0 && it.scale.b == 1.0 { it.vars[0].clone() } else { format!("(({}+{})/{})", it.vars[0], fmt_g(it.scale.a, 8), fmt_g(it.scale.b, 8)) };
                let f = if p == 0.0 { format!("ln({base})") } else { format!("{base}^{}", pstr(p)) };
                gen_lines.push(format!("-> gen double {n} = {f}"));
            }
        }
        cov.extend(names);
    }
    let mut out = vec![String::new(), format!("Fractional polynomial fitting algorithm converged after {ncycles} cycle{}.", if ncycles == 1 { "" } else { "s" }), String::new(), "Transformations of covariates:".to_string(), String::new()];
    out.extend(gen_lines);
    out.push(String::new());
    out.push(format!("Final multivariable fractional polynomial model for {}", deps.first().cloned().unwrap_or_else(|| "_t".into())));
    let bar = "-".repeat(70);
    out.push(bar.clone());
    out.push("    Variable |    -----Initial-----          -----Final-----".to_string());
    out.push("             |   df     Select   Alpha    Status    df    Powers".to_string());
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(56)));
    for it in &items {
        let fdf = if it.powers.is_empty() { 0 } else if it.powers == [1.0] { 1 } else { 2 * it.powers.len() };
        let pw = if it.powers.is_empty() { ".".to_string() } else { it.powers.iter().map(|p| pstr(*p)).collect::<Vec<_>>().join(" ") };
        out.push(format!("{:>12} | {:>4}     {:.4}   {:.4}    {:>4}   {:>4}    {}", ferrostat_cmd::summarize::abbrev(&it.name, 12), it.df, it.select, it.alpha, if it.powers.is_empty() { "out" } else { "in" }, fdf, pw));
    }
    out.push(bar);
    me.emit(&out.join("\n"));
    let line = format!("{verb} {} {}{}", deps.join(" "), cov.join(" "), tail);
    me.exec_text(&line, None)?;
    let e = &mut me.engine.store.e;
    e.set_str("fp_cmd", "mfp");
    e.set_str("fp_fvl", cov.join(" "));
    e.set("fp_nx", items.len() as f64);
    for (i, it) in items.iter().enumerate() {
        e.set_str(&format!("fp_x{}", i + 1), it.name.clone());
        e.set_str(&format!("fp_k{}", i + 1), if it.powers.is_empty() { ".".into() } else { it.powers.iter().map(|p| pstr(*p)).collect::<Vec<_>>().join(" ") });
    }
    Ok(())
}
