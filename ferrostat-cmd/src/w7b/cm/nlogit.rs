//! Nested logit: `nlogit` (two-level, RUM-consistent parameterization,
//! full information maximum likelihood), `nlogitgen` (build a nesting
//! variable from groups of alternatives) and `nlogittree` (display the
//! tree).
//!
//! Syntax:
//!   nlogit depvar [indepvars] [if] [in] || nestvar: [varlist] [, base(#|lbl) estconst]
//!          || altvar: [varlist] [, noconstant base(#|lbl)] , case(varname) [options]
//!
//! With dissimilarity parameters tau_m, an alternative j in nest m has
//! P(j) = exp(V_j/tau_m) / S_m * S_m^tau_m * exp(W_m) / sum_l S_l^tau_l exp(W_l),
//! S_m = sum over the nest of exp(V/tau_m). Nests with a single
//! alternative keep tau = 1 (not identified).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, ValueLabel, Value, VarType};
use ferrostat_parser::Command;

use super::data::{self, CmData, Need, UnitLik};
use crate::commands::CommandHandler;
use crate::est::{self, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub struct Nlogit;
pub struct Nlogitgen;
pub struct Nlogittree;

/// Split at a separator outside parentheses.
fn split_top(s: &str, sep: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        let c = b[i] as char;
        if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
        }
        if depth == 0 && s[i..].starts_with(sep) {
            out.push(std::mem::take(&mut cur));
            i += sep.len();
            continue;
        }
        cur.push(c);
        i += 1;
    }
    out.push(cur);
    out
}

/// Text after the verb in a raw command line.
fn after_verb(raw: &str) -> String {
    let t = raw.trim_start();
    match t.find(char::is_whitespace) {
        Some(p) => t[p..].trim().to_string(),
        None => String::new(),
    }
}

/// `name(arg)` options from a text of options.
fn opt_in(text: &str, name: &str, min: usize) -> Option<String> {
    for (n, a) in ferrostat_parser::parser::split_options(text) {
        let n = n.to_lowercase();
        if n.len() >= min && name.starts_with(n.as_str()) {
            return Some(a.unwrap_or_default());
        }
    }
    None
}

struct Level {
    var: String,
    vars: String,
    opts: String,
}

fn parse_level(seg: &str) -> Result<Level, String> {
    let (head, opts) = match split_top(seg, ",").as_slice() {
        [h] => (h.clone(), String::new()),
        [h, rest @ ..] => (h.clone(), rest.join(" ")),
        _ => unreachable!(),
    };
    let (var, vars) = head.split_once(':').ok_or_else(|| rc_err(198, format!("invalid level specification: {seg}")))?;
    Ok(Level { var: var.trim().to_string(), vars: vars.trim().to_string(), opts })
}

pub struct Nest {
    /// nest index of each alternative
    nest_of: Vec<usize>,
    labels: Vec<String>,
    /// nests with an estimated dissimilarity parameter, in order
    free: Vec<usize>,
    pub nestvar: String,
}

/// Parse the command, prepare the data with nest-level columns.
fn build(ds: &DataSet, st: &Store, raw: &str) -> Result<(Command, CmData, Nest), String> {
    let body = after_verb(raw);
    let segs = split_top(&body, "||");
    if segs.len() < 3 {
        return Err(rc_err(198, "nlogit requires a nesting level and the alternatives level: depvar indepvars || nestvar: || altvar:"));
    }
    if segs.len() > 3 {
        return Err(rc_err(198, "only one nesting level is supported"));
    }
    let seg0 = segs[0].trim();
    let mid = parse_level(&segs[1])?;
    let last = parse_level(&segs[2])?;
    let mut opts = last.opts.clone();
    if let Some(b) = opt_in(&last.opts, "base", 4) {
        opts = opts.replacen(&format!("base({b})"), "", 1);
        opts.push_str(&format!(" basealternative({b})"));
    }
    let mut line = format!("nlogit {seg0}, alternatives({})", last.var);
    if !last.vars.is_empty() {
        line.push_str(&format!(" casevars({})", last.vars));
    }
    line.push(' ');
    line.push_str(&opts);
    let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, format!("{e:?}")))?;
    let mut d = data::prepare(ds, st, &c, Need::One)?;
    // nests
    let nestvar = crate::varlist::resolve_name(ds, &mid.var)?;
    let nv = ds.get_var(&nestvar).ok_or_else(|| rc_err(111, format!("variable {nestvar} not found")))?.clone();
    let nc = util::col(ds, &nestvar)?;
    let mut nests: Vec<f64> = d.d.rows.iter().map(|&r| nc[r]).collect();
    if nests.iter().any(|v| is_missing(*v)) {
        return Err(rc_err(416, format!("missing values in {nestvar}")));
    }
    nests.sort_by(f64::total_cmp);
    nests.dedup();
    let mut nest_of = vec![usize::MAX; d.alts.len()];
    let mut labels = vec![String::new(); nests.len()];
    for (t, &r) in d.d.rows.iter().enumerate() {
        let m = nests.binary_search_by(|k| k.total_cmp(&nc[r])).unwrap();
        let a = d.alt_of[t];
        if nest_of[a] != usize::MAX && nest_of[a] != m {
            return Err(rc_err(459, format!("alternatives must be nested within {nestvar}")));
        }
        nest_of[a] = m;
        labels[m] = util::cell_text(ds, &nv, r, true).replace(char::is_whitespace, "_");
    }
    // nest-level variables
    let nvars = if mid.vars.is_empty() { vec![] } else { crate::varlist::expand(ds, &mid.vars)? };
    let estconst = opt_in(&mid.opts, "estconst", 4).is_some();
    let base_nest = match opt_in(&mid.opts, "base", 4) {
        Some(b) => {
            let b = b.trim().to_string();
            match b.parse::<f64>().ok().and_then(|x| nests.iter().position(|n| *n == x)) {
                Some(m) => m,
                None => labels.iter().position(|l| *l == b).ok_or_else(|| rc_err(198, format!("base({b}) not found")))?,
            }
        }
        None => 0,
    };
    if !nvars.is_empty() || estconst {
        let cols: Vec<Vec<f64>> = nvars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mut names = Vec::new();
        let mut newc = Vec::new();
        let mut isc = Vec::new();
        for m in 0..nests.len() {
            if m == base_nest {
                continue;
            }
            let dummy: Vec<f64> = d.alt_of.iter().map(|&a| if nest_of[a] == m { 1.0 } else { 0.0 }).collect();
            for (cv, cn) in cols.iter().zip(&nvars) {
                names.push(format!("{}:{cn}", labels[m]));
                newc.push(d.d.rows.iter().zip(&dummy).map(|(&r, dm)| dm * cv[r]).collect::<Vec<f64>>());
                isc.push(false);
            }
            if estconst {
                names.push(format!("{}:_cons", labels[m]));
                newc.push(dummy);
                isc.push(true);
            }
        }
        // nest equations come before the alternative equations
        let k_as = d.n_as;
        let tail_n: Vec<String> = d.names.drain(k_as..).collect();
        let tail_c: Vec<Vec<f64>> = d.cols.drain(k_as..).collect();
        let tail_i: Vec<bool> = d.is_cons.drain(k_as..).collect();
        d.names.extend(names);
        d.cols.extend(newc);
        d.is_cons.extend(isc);
        d.names.extend(tail_n);
        d.cols.extend(tail_c);
        d.is_cons.extend(tail_i);
    }
    let free: Vec<usize> = (0..nests.len()).filter(|&m| nest_of.iter().filter(|&&n| n == m).count() >= 2).collect();
    Ok((c, d, Nest { nest_of, labels, free, nestvar }))
}

/// Per-row (P(j), P(j | nest), inclusive value of the nest) for a case.
fn case_probs(d: &CmData, nest: &Nest, g: &[usize], beta: &[f64], tau: &[f64]) -> Vec<(f64, f64, f64)> {
    let nn = nest.labels.len();
    let v: Vec<f64> = g.iter().map(|&t| d.xb(beta, t)).collect();
    let mut iv = vec![f64::NEG_INFINITY; nn];
    for m in 0..nn {
        let tm = tau[m];
        let s: Vec<f64> = g.iter().zip(&v).filter(|(t, _)| nest.nest_of[d.alt_of[**t]] == m).map(|(_, vv)| vv / tm).collect();
        if s.is_empty() {
            continue;
        }
        let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        iv[m] = mx + s.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
    }
    let top: Vec<f64> = (0..nn).map(|m| if iv[m].is_finite() { tau[m] * iv[m] } else { f64::NEG_INFINITY }).collect();
    let mx = top.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let den: f64 = top.iter().map(|x| (x - mx).exp()).sum();
    g.iter()
        .zip(&v)
        .map(|(&t, vv)| {
            let m = nest.nest_of[d.alt_of[t]];
            let cond = (vv / tau[m] - iv[m]).exp();
            let pn = (top[m] - mx).exp() / den;
            (cond * pn, cond, iv[m])
        })
        .collect()
}

fn taus(nest: &Nest, aux: &[f64]) -> Vec<f64> {
    let mut t = vec![1.0; nest.labels.len()];
    for (i, &m) in nest.free.iter().enumerate() {
        t[m] = aux[i];
    }
    t
}

impl CommandHandler for Nlogit {
    fn name(&self) -> &'static str {
        "nlogit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() && !cmd.raw.contains("||") {
            if st.e.macros.get("cmd").map(String::as_str) != Some("nlogit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (c, d, nest) = build(ds, st, &cmd.raw)?;
        let k = d.cols.len();
        if k == 0 {
            return Err(rc_err(102, "no covariates or alternative-specific constants to estimate"));
        }
        let level = util::level(&c, st)?;
        let vce = est::parse_vce(&c, Vce::Oim)?;
        let mut names = d.names.clone();
        for &m in &nest.free {
            names.push(format!("/:{}_tau", nest.labels[m]));
        }
        let start = super::models::clogit_fit(&d)?;
        let ll_c = start.ll;
        let mut b0 = start.b.clone();
        b0.extend(vec![1.0; nest.free.len()]);
        let chosen: Vec<usize> = d.cases.iter().map(|g| g.iter().position(|&t| d.y_of(t) != 0.0).unwrap()).collect();
        let ff = |c: usize, th: &[f64]| {
            let tau = taus(&nest, &th[k..]);
            if tau.iter().any(|t| !(*t > 1e-3)) {
                return f64::NEG_INFINITY;
            }
            let p = case_probs(&d, &nest, &d.cases[c], &th[..k], &tau);
            p[chosen[c]].0.max(1e-300).ln()
        };
        let lik = UnitLik { k: names.len(), units: d.cases.len(), f: &ff, grad: None };
        let cluster = data::case_cluster(ds, &d, &vce, false)?;
        let neq = data::count_eqs(&d.names);
        let fit = data::fit_units(&lik, b0, &names, neq, nest.free.len(), &vce, cluster.as_deref())?;
        let widx: Vec<usize> = (0..k).filter(|&j| !d.is_cons[j]).collect();
        let (chi2, df_m) = crate::heckman::wald(&fit.b, &fit.v, &widx);
        let ll = fit.ll();
        let lr = (2.0 * (ll - ll_c)).max(0.0);
        let lr_df = nest.free.len() as f64;
        let robust = vce != Vce::Oim;
        let (amin, aavg, amax) = d.alt_sizes();
        let hr = crate::choice_mixed::hrow;
        let ttl = "RUM-consistent nested logit regression";
        let mut out = vec![String::new()];
        out.extend(est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
        out.push(String::new());
        out.push(hr(ttl, "Number of obs", util::commas(d.n())));
        out.push(hr(&format!("Case variable: {}", d.set.case), "Number of cases", util::commas(d.cases.len())));
        out.push(String::new());
        out.push(hr(&format!("Alternative variable: {}", d.set.alt.clone().unwrap_or_default()), "Alts per case: min", util::commas(amin)));
        out.push(hr("", "               avg", format!("{aavg:.1}")));
        out.push(hr("", "               max", util::commas(amax)));
        out.push(String::new());
        out.push(hr("", &format!("Wald chi2({})", fmt_g(df_m, 9)), xm::f2(chi2)));
        out.push(hr(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9)), "Prob > chi2", xm::f4(xm::chi2p(df_m, chi2))));
        out.push(String::new());
        out.extend(xm::table(&d.depvar, &fit, level, None, &[]));
        out.push(format!("LR test for IIA (tau=1): chi2({}) = {}   Prob > chi2 = {}", fmt_g(lr_df, 9), xm::f2(lr), xm::f4(xm::chi2p(lr_df, lr))));
        let text = out.join("\n");
        let raw = cmd.raw.clone();
        let ncase = d.cases.len();
        let alts = d.alts.clone();
        let altl = d.alt_labels.clone();
        let set = d.set.clone();
        let depvar = d.depvar.clone();
        let nestvar = nest.nestvar.clone();
        xm::post_fit(st, "nlogit", &d.d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df_m);
            e.set("ll", ll);
            e.set("ll_c", ll_c);
            e.set("chi2_c", lr);
            e.set("df_c", lr_df);
            e.set("p_c", xm::chi2p(lr_df, lr));
            e.set("N_case", ncase as f64);
            e.set("k_alt", alts.len() as f64);
            e.set("alt_min", amin as f64);
            e.set("alt_avg", aavg);
            e.set("alt_max", amax as f64);
            e.set("levels", 2.0);
            e.set_mat("altvals", Matrix::row_vector(&alts));
            e.set_str("alteqs", altl.join(" "));
            e.set_str("altvar", set.alt.clone().unwrap_or_default());
            e.set_str("case", set.case.clone());
            e.set_str("depvar", depvar.clone());
            e.set_str("nestvar", nestvar.clone());
            e.set_str("title", ttl);
            e.set_str("predict", "nlogit_p");
            e.set_str("_cmdline", raw.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

impl CmData {
    fn y_of(&self, t: usize) -> f64 {
        self.d.y[t]
    }
}

/// `predict` after nlogit: pr (default), condp, iv, xb.
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
    let raw = st.e.macros.get("_cmdline").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let (_, d, nest) = build(ds, st, &raw)?;
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let th: Vec<f64> = (0..b.cols).map(|j| b.get(0, j)).collect();
    let k = d.cols.len();
    let tau = taus(&nest, &th[k..]);
    let stat = if cmd.has_opt("xb", 2) {
        "xb"
    } else if cmd.has_opt("condp", 4) {
        "condp"
    } else if cmd.has_opt("iv", 2) {
        "iv"
    } else {
        "pr"
    };
    let mut vs = vec![SYSMISS; d.n()];
    for g in &d.cases {
        let p = case_probs(&d, &nest, g, &th[..k], &tau);
        for (i, &t) in g.iter().enumerate() {
            vs[t] = match stat {
                "xb" => d.xb(&th[..k], t),
                "condp" => p[i].1,
                "iv" => p[i].2,
                _ => p[i].0,
            };
        }
    }
    let mask = util::touse(ds, st, cmd)?;
    let mut vals = vec![SYSMISS; ds.n_obs];
    for (t, &r) in d.d.rows.iter().enumerate() {
        if mask[r] {
            vals[r] = vs[t];
        }
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if stat == "pr" && !cmd.has_opt("pr", 2) {
        lines.push("(option pr assumed)".to_string());
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// nlogitgen / nlogittree
// ---------------------------------------------------------------------------

impl CommandHandler for Nlogitgen {
    fn name(&self) -> &'static str {
        "nlogitgen"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let body = after_verb(&cmd.raw);
        let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "syntax: nlogitgen newvar = altvar(branchlist)"))?;
        let newvar = lhs.trim().to_string();
        crate::varlist::check_name(&newvar)?;
        if ds.var_index.contains_key(&newvar) {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        let rhs = rhs.trim();
        let open = rhs.find('(').ok_or_else(|| rc_err(198, "branch list required"))?;
        let close = rhs.rfind(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
        let altvar = crate::varlist::resolve_name(ds, rhs[..open].trim())?;
        let av = ds.get_var(&altvar).unwrap().clone();
        let ac = util::col(ds, &altvar)?;
        let vl = ds.value_labels.get(&av.value_label_name).cloned();
        let lookup = |s: &str| -> Result<f64, String> {
            if let Ok(x) = s.parse::<f64>() {
                return Ok(x);
            }
            vl.as_ref().and_then(|l| l.value_of(s)).ok_or_else(|| rc_err(198, format!("alternative {s} not found in {altvar}")))
        };
        let mut branches: Vec<(String, Vec<f64>)> = Vec::new();
        for b in split_top(&rhs[open + 1..close], ",") {
            let b = b.trim();
            if b.is_empty() {
                continue;
            }
            let (lab, list) = match b.split_once(':') {
                Some((l, r)) => (Some(l.trim().to_string()), r.to_string()),
                None => (None, b.to_string()),
            };
            let items: Vec<String> = list.split('|').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            let vals: Vec<f64> = items.iter().map(|s| lookup(s)).collect::<Result<_, _>>()?;
            let lab = lab.unwrap_or_else(|| items.join("_"));
            branches.push((lab, vals));
        }
        if branches.len() < 2 {
            return Err(rc_err(198, "at least two branches required"));
        }
        let mut vals = vec![SYSMISS; ds.n_obs];
        for i in 0..ds.n_obs {
            if is_missing(ac[i]) {
                continue;
            }
            let pos: Vec<usize> = (0..branches.len()).filter(|&b| branches[b].1.contains(&ac[i])).collect();
            match pos.as_slice() {
                [b] => vals[i] = (b + 1) as f64,
                [] => {}
                _ => return Err(rc_err(198, format!("alternative {} appears in more than one branch", fmt_g(ac[i], 9)))),
            }
        }
        let mut v = util::new_double(&newvar, vals.clone());
        v.var_type = VarType::Byte;
        v.data = vals.iter().map(|x| if is_missing(*x) { Value::Missing } else { Value::Byte(*x as i8) }).collect();
        let mut lab = ValueLabel::new(format!("lb_{newvar}"));
        for (i, (l, _)) in branches.iter().enumerate() {
            lab.add((i + 1) as f64, l.clone());
        }
        ds.value_labels.insert(format!("lb_{newvar}"), lab);
        v.value_label_name = format!("lb_{newvar}");
        ds.add_variable(v)?;
        let mut out = vec![format!("New variable {newvar} is generated with {} groups", branches.len()), format!("label list lb_{newvar}"), format!("lb_{newvar}:")];
        for (i, (l, _)) in branches.iter().enumerate() {
            out.push(format!("{:>11} {l}", i + 1));
        }
        if cmd.has_opt("nolog", 3) {
            return Ok(String::new());
        }
        Ok(out.join("\n"))
    }
}

impl CommandHandler for Nlogittree {
    fn name(&self) -> &'static str {
        "nlogittree"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified: altvar nestvar"));
        }
        let alt = &vars[0];
        let nestv = &vars[vars.len() - 1];
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let ch = match cmd.opt("choice", 3) {
            Some(c) => Some(util::col(ds, &crate::varlist::resolve_name(ds, c.trim())?)?),
            None => None,
        };
        let ac = util::col(ds, alt)?;
        let nc = util::col(ds, nestv)?;
        let av = ds.get_var(alt).unwrap().clone();
        let nv = ds.get_var(nestv).unwrap().clone();
        let nolab = cmd.has_opt("nolabel", 3);
        // nest -> alternatives -> (N, chosen)
        let mut tree: std::collections::BTreeMap<u64, (String, std::collections::BTreeMap<u64, (String, usize, usize)>)> = Default::default();
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            let ne = tree.entry((nc[i] + 1e15).to_bits()).or_insert((util::cell_text(ds, &nv, i, !nolab), Default::default()));
            let ae = ne.1.entry((ac[i] + 1e15).to_bits()).or_insert((util::cell_text(ds, &av, i, !nolab), 0, 0));
            ae.1 += 1;
            if let Some(c) = &ch {
                if c[i] != 0.0 && !is_missing(c[i]) {
                    ae.2 += 1;
                }
            }
        }
        let mut out = vec![String::new(), "tree structure specified for the nested logit model".to_string(), String::new()];
        let kcol = if ch.is_some() { format!(" {:>6}", "k") } else { String::new() };
        out.push(format!(" {:<12} {:>6}     {:<14} {:>6}{kcol}", crate::summarize::abbrev(nestv, 12), "N", crate::summarize::abbrev(alt, 14), "N"));
        out.push("-".repeat(50 + kcol.len()));
        let (mut tn, mut tk) = (0usize, 0usize);
        for (_, (nl, alts)) in &tree {
            let nsum: usize = alts.values().map(|a| a.1).sum();
            for (j, (_, (al, n, k))) in alts.iter().enumerate() {
                let kc = if ch.is_some() { format!(" {:>6}", k) } else { String::new() };
                if j == 0 {
                    out.push(format!(" {:<12} {:>6} --- {:<14} {:>6}{kc}", crate::summarize::abbrev(nl, 12), nsum, crate::summarize::abbrev(al, 14), n));
                } else {
                    out.push(format!(" {:<12} {:>6} +-- {:<14} {:>6}{kc}", "", "", crate::summarize::abbrev(al, 14), n));
                }
                tn += n;
                tk += k;
            }
        }
        out.push("-".repeat(50 + kcol.len()));
        out.push(format!(" {:>40} {:>6}{}", "total", tn, if ch.is_some() { format!(" {:>6}", tk) } else { String::new() }));
        if ch.is_some() {
            out.push(String::new());
            out.push("k = number of times alternative is chosen".to_string());
        }
        out.push("N = number of observations at each level".to_string());
        Ok(out.join("\n"))
    }
}
