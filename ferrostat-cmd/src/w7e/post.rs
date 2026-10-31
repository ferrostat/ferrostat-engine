//! Postestimation after Bayesian estimation: `bayesstats`, `bayestest`,
//! `bayesgraph`, `bayespredict`, `bayesreps` and `estat`.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

use super::expr::{self, Resolver, CE};
use super::run::num9;
use super::stats;
use super::text;

/// The simulation sample of the current results.
pub struct Draws {
    pub names: Vec<String>,
    /// one vector per parameter
    pub cols: Vec<Vec<f64>>,
    pub ll: Vec<f64>,
    pub chain: Vec<usize>,
    pub nchains: usize,
}

pub fn draws_of(e: &Results) -> Result<Draws, String> {
    let m = e.matrices.get("_mcmc").ok_or_else(|| rc_err(301, "last estimation results not bayes; use bayesmh or the bayes prefix"))?;
    let k = m.cols - 3;
    let names = m.colnames[..k].to_vec();
    let cols = (0..k).map(|j| (0..m.rows).map(|r| m.get(r, j)).collect()).collect();
    let ll = (0..m.rows).map(|r| m.get(r, k)).collect();
    let chain: Vec<usize> = (0..m.rows).map(|r| m.get(r, k + 2) as usize - 1).collect();
    let nchains = chain.iter().copied().max().map_or(1, |c| c + 1);
    Ok(Draws { names, cols, ll, chain, nchains })
}

impl Draws {
    fn idx(&self, n: &str) -> Option<usize> {
        self.names.iter().position(|x| x == n)
    }
}

/// A requested quantity: a parameter or an expression of parameters.
pub struct Item {
    pub label: String,
    pub vals: Vec<f64>,
}

struct DRes<'a> {
    d: &'a Draws,
}

impl<'a> Resolver for DRes<'a> {
    fn brace(&mut self, t: &str) -> Result<CE, String> {
        let t = t.trim();
        let name = if let Some((e, p)) = t.split_once(':') { format!("{}:{}", e.trim(), p.trim()) } else { t.to_string() };
        self.d.idx(&name).map(CE::Par).ok_or_else(|| rc_err(198, format!("parameter {{{name}}} not found")))
    }
    fn name(&mut self, n: &str) -> Result<CE, String> {
        Err(rc_err(111, format!("{n} not found")))
    }
}

/// Expand `{y:} {var} (label: expr) _all` into items.
pub fn items(d: &Draws, e: &Results, spec: &str) -> Result<Vec<Item>, String> {
    let spec = spec.trim();
    let mut out = Vec::new();
    let shown: Vec<String> = e.macros.get("parnames").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let col = |j: usize| Item { label: d.names[j].clone(), vals: d.cols[j].clone() };
    if spec.is_empty() || spec == "_all" {
        for n in &shown {
            if let Some(j) = d.idx(n) {
                out.push(col(j));
            }
        }
        if spec == "_all" {
            for (j, n) in d.names.iter().enumerate() {
                if !shown.contains(n) {
                    out.push(col(j));
                }
            }
        }
        return Ok(out);
    }
    for w in text::words(spec) {
        if w.starts_with('(') && w.ends_with(')') {
            let inner = &w[1..w.len() - 1];
            let (label, ex) = match text::find_top(inner, ':') {
                Some(i) if !inner[..i].contains('{') => (inner[..i].trim().to_string(), inner[i + 1..].trim().to_string()),
                _ => (format!("expr{}", out.len() + 1), inner.to_string()),
            };
            let ce = expr::compile(&ex, &mut DRes { d })?;
            let n = d.ll.len();
            let vals: Vec<f64> = (0..n).map(|r| {
                let th: Vec<f64> = d.cols.iter().map(|c| c[r]).collect();
                ce.eval(&expr::Ctx { cols: &[], theta: &th, xb: &[], row: 0 })
            }).collect();
            out.push(Item { label, vals });
            continue;
        }
        if w == "_all" {
            for (j, _) in d.names.iter().enumerate() {
                out.push(col(j));
            }
            continue;
        }
        let groups = if w.contains('{') { text::braces(&w) } else { vec![w.clone()] };
        for g in groups {
            let g = g.trim();
            if let Some(eq) = g.strip_suffix(':') {
                let pre = format!("{}:", eq.trim());
                let mut any = false;
                for (j, n) in d.names.iter().enumerate() {
                    if n.starts_with(&pre) {
                        out.push(col(j));
                        any = true;
                    }
                }
                if !any {
                    return Err(rc_err(198, format!("equation {eq} not found")));
                }
                continue;
            }
            if let Some((eq, rest)) = g.split_once(':') {
                for p in rest.split_whitespace() {
                    let n = format!("{}:{p}", eq.trim());
                    let j = d.idx(&n).ok_or_else(|| rc_err(198, format!("parameter {{{n}}} not found")))?;
                    out.push(col(j));
                }
                continue;
            }
            if let Some((mn, _)) = g.split_once(',') {
                let pre = format!("{}_", mn.trim());
                for (j, n) in d.names.iter().enumerate() {
                    if n.starts_with(&pre) {
                        out.push(col(j));
                    }
                }
                continue;
            }
            for p in g.split_whitespace() {
                if let Some(j) = d.idx(p) {
                    out.push(col(j));
                    continue;
                }
                // random effects term, or equation without the colon
                let pre_re = format!("{p}[");
                let pre_eq = format!("{p}:");
                let mut any = false;
                for (j, n) in d.names.iter().enumerate() {
                    if n.starts_with(&pre_re) || n.starts_with(&pre_eq) {
                        out.push(col(j));
                        any = true;
                    }
                }
                if !any {
                    return Err(rc_err(198, format!("parameter {{{p}}} not found")));
                }
            }
        }
    }
    Ok(out)
}

fn split_rest(rest: &str) -> (String, Vec<(String, Option<String>)>) {
    match text::find_top(rest, ',') {
        Some(i) => (rest[..i].trim().to_string(), text::options(&rest[i + 1..])),
        None => (rest.trim().to_string(), Vec::new()),
    }
}

fn opt<'a>(o: &'a [(String, Option<String>)], full: &str, min: usize) -> Option<&'a Option<String>> {
    o.iter().rev().find(|(n, _)| text::abbrev(n, full, min)).map(|x| &x.1)
}

fn optnum(o: &[(String, Option<String>)], full: &str, min: usize) -> Result<Option<f64>, String> {
    match opt(o, full, min) {
        Some(Some(s)) => s.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("option {full}() invalid"))),
        Some(None) => Err(rc_err(198, format!("option {full}() requires a value"))),
        None => Ok(None),
    }
}

fn level_of(st: &Store, o: &[(String, Option<String>)]) -> Result<f64, String> {
    Ok(match optnum(o, "clevel", 2)? {
        Some(v) => v,
        None => st.settings.get("clevel").and_then(|v| v.parse().ok()).or_else(|| st.e.get("clevel")).unwrap_or(st.level),
    })
}

fn split_label(n: &str) -> (Option<String>, String) {
    match n.split_once(':') {
        Some((e, p)) if !n.contains('[') || n.find(':').unwrap() < n.find('[').unwrap() => (Some(e.to_string()), p.to_string()),
        _ => (None, n.to_string()),
    }
}

/// Generic table: label column plus value columns, grouped by equation.
fn grouped_table(labels: &[String], head: &[String], rows: &[Vec<String>], widths: &[usize]) -> Vec<String> {
    let lw = 12usize.max(labels.iter().map(|l| split_label(l).1.len().min(32)).max().unwrap_or(0));
    let tw: usize = widths.iter().map(|w| w + 1).sum::<usize>();
    let mut out = vec!["-".repeat(lw + 2 + tw)];
    for h in head {
        out.push(format!("{:>lw$} |{}", "", h));
    }
    let sep = format!("{}+{}", "-".repeat(lw + 1), "-".repeat(tw));
    out.push(sep.clone());
    let mut last: Option<String> = None;
    for (i, l) in labels.iter().enumerate() {
        let (eq, nm) = split_label(l);
        if eq != last {
            if i > 0 {
                out.push(sep.clone());
            }
            if let Some(e) = &eq {
                out.push(format!("{:<lw$} |", e));
            }
            last = eq;
        }
        let mut line = format!("{:>lw$} |", if nm.len() > lw { format!("{}~{}", &nm[..lw - 3], &nm[nm.len() - 2..]) } else { nm });
        for (c, w) in rows[i].iter().zip(widths) {
            line.push_str(&format!(" {:>w$}", c, w = *w));
        }
        out.push(line);
    }
    out.push("-".repeat(lw + 2 + tw));
    out
}

fn mcmc_header(st: &Store) -> Vec<String> {
    let e = &st.e;
    let g = |k: &str| e.get(k).unwrap_or(0.0);
    let mut v = vec![
        format!("{:>47}{:<17}= {:>10}", "", "MCMC iterations", util::commas(g("mcmciter") as usize)),
        format!("{:>47}{:<17}= {:>10}", "", "Burn-in", util::commas(g("burnin") as usize)),
        format!("{:>47}{:<17}= {:>10}", "", "MCMC sample size", util::commas(g("mcmcsize") as usize)),
    ];
    if g("nchains") > 1.0 {
        v.push(format!("{:>47}{:<17}= {:>10}", "", "Number of chains", g("nchains")));
    }
    v
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

pub fn replay(_ds: &mut DataSet, st: &mut Store, _rest: &str) -> Result<String, String> {
    if !super::is_bayes(st) {
        return Err(rc_err(301, "last estimates not found"));
    }
    Ok(st.e.macros.get("_output").cloned().unwrap_or_default())
}

fn sub_and_rest(rest: &str) -> (String, String) {
    let r = rest.trim();
    let end = r.find(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == '{').unwrap_or(r.len());
    (r[..end].to_lowercase(), r[end..].trim().to_string())
}

pub fn dispatch(verb: &str, ds: &mut DataSet, st: &mut Store, rest: &str, cmd: &Command) -> Result<String, String> {
    match verb {
        "bayesstats" => {
            let (sub, r) = sub_and_rest(rest);
            match sub.as_str() {
                s if text::abbrev(s, "summary", 3) => summary(st, &r),
                "ess" => ess(st, &r),
                "ic" => ic(st, &r),
                s if text::abbrev(s, "grubin", 2) => grubin(st, &r),
                s if text::abbrev(s, "ppvalues", 3) => super::predict::ppvalues(ds, st, &r),
                o => Err(rc_err(198, format!("bayesstats {o} is unrecognized"))),
            }
        }
        "bayestest" => {
            let (sub, r) = sub_and_rest(rest);
            match sub.as_str() {
                s if text::abbrev(s, "model", 3) => test_model(st, &r),
                s if text::abbrev(s, "interval", 3) => test_interval(st, &r),
                o => Err(rc_err(198, format!("bayestest {o} is unrecognized"))),
            }
        }
        "bayesgraph" => {
            let (sub, r) = sub_and_rest(rest);
            graph(st, &sub, &r, cmd)
        }
        "bayespredict" => super::predict::bayespredict(ds, st, rest),
        "bayesreps" => super::predict::bayesreps(ds, st, rest),
        "bayesirf" | "bayesfcast" | "bayesvarstable" => super::var::dispatch(verb, ds, st, rest, cmd),
        "bayesselect" => super::select::bayesselect(ds, st, rest),
        o => Err(rc_err(199, format!("command {o} is unrecognized"))),
    }
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let rest = super::after_verb(cmd);
    let r = rest.trim_start().strip_prefix(sub).unwrap_or(&rest).trim().to_string();
    let _ = ds;
    Some(match sub {
        "ic" => ic(st, &r),
        s if text::abbrev(s, "summary", 3) => summary(st, &r),
        "ess" => ess(st, &r),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// bayesstats
// ---------------------------------------------------------------------------

fn summary(st: &mut Store, rest: &str) -> Result<String, String> {
    let d = draws_of(&st.e)?;
    let (spec, o) = split_rest(rest);
    let level = level_of(st, &o)?;
    let use_hpd = opt(&o, "hpd", 3).is_some();
    let batch = optnum(&o, "batch", 5)?.unwrap_or(0.0) as usize;
    let its = items(&d, &st.e, &spec)?;
    let mut out = vec![String::new(), "Posterior summary statistics".to_string()];
    out.extend(mcmc_header(st));
    out.push(String::new());
    let labels: Vec<String> = its.iter().map(|i| i.label.clone()).collect();
    let mut rows = Vec::new();
    let k = its.len();
    let mut mat = Matrix::zeros(k, 6);
    for (r, it) in its.iter().enumerate() {
        let s = stats::summarize(&it.vals, level, use_hpd, batch);
        rows.push(vec![num9(s.mean), num9(s.sd), num9(s.mcse), num9(s.median), num9(s.lo), num9(s.hi)]);
        for (c, v) in [s.mean, s.sd, s.mcse, s.median, s.lo, s.hi].into_iter().enumerate() {
            mat[(r, c)] = v;
        }
    }
    let head = vec![
        format!("{:>64}", if use_hpd { "HPD" } else { "Equal-tailed" }),
        format!("      Mean   Std. dev.     MCSE     Median  [{}% cred. interval]", fmt_g(level, 4)),
    ];
    out.extend(grouped_table(&labels, &head, &rows, &[9, 10, 9, 10, 10, 10]));
    let r = st.begin_r();
    r.set_mat("summary", mat.with_names(labels, vec!["mean".into(), "sd".into(), "mcse".into(), "median".into(), "crlb".into(), "crub".into()]));
    r.set("clevel", level);
    r.set_str("hpd", if use_hpd { "hpd" } else { "" });
    Ok(out.join("\n"))
}

fn ess(st: &mut Store, rest: &str) -> Result<String, String> {
    let d = draws_of(&st.e)?;
    let (spec, _o) = split_rest(rest);
    let its = items(&d, &st.e, &spec)?;
    let n = d.ll.len() as f64;
    let mut out = vec![String::new(), "Efficiency summaries".to_string()];
    out.extend(mcmc_header(st));
    let labels: Vec<String> = its.iter().map(|i| i.label.clone()).collect();
    let mut rows = Vec::new();
    let mut mat = Matrix::zeros(its.len(), 3);
    let mut effs = Vec::new();
    for (r, it) in its.iter().enumerate() {
        let e = stats::ess(&it.vals);
        let corr = n / e;
        let eff = e / n;
        effs.push(eff);
        rows.push(vec![format!("{e:.2}"), format!("{corr:.2}"), format!("{eff:.4}").trim_start_matches('0').to_string()]);
        mat[(r, 0)] = e;
        mat[(r, 1)] = corr;
        mat[(r, 2)] = eff;
    }
    out.push(format!("{:>47}{:<17}= {:>10}", "", "Efficiency:  min", format!("{:.4}", effs.iter().copied().fold(f64::INFINITY, f64::min)).trim_start_matches('0')));
    out.push(format!("{:>47}{:<17}= {:>10}", "", "             avg", format!("{:.4}", stats::mean(&effs)).trim_start_matches('0')));
    out.push(format!("{:>47}{:<17}= {:>10}", "", "             max", format!("{:.4}", effs.iter().copied().fold(0.0, f64::max)).trim_start_matches('0')));
    out.push(String::new());
    out.extend(grouped_table(&labels, &["        ESS   Corr. time    Efficiency".to_string()], &rows, &[10, 12, 13]));
    let r = st.begin_r();
    r.set_mat("ess", mat.with_names(labels, vec!["ess".into(), "corrtime".into(), "efficiency".into()]));
    r.set("skip", 0.0);
    Ok(out.join("\n"))
}

fn grubin(st: &mut Store, rest: &str) -> Result<String, String> {
    let d = draws_of(&st.e)?;
    if d.nchains < 2 {
        return Err(rc_err(198, "Gelman-Rubin diagnostics require multiple chains; specify nchains() at estimation"));
    }
    let (spec, o) = split_rest(rest);
    let its = items(&d, &st.e, &spec)?;
    let labels: Vec<String> = its.iter().map(|i| i.label.clone()).collect();
    let mut rows = Vec::new();
    let mut rc = Vec::new();
    for it in &its {
        let chains: Vec<Vec<f64>> = (0..d.nchains).map(|c| it.vals.iter().zip(&d.chain).filter(|(_, ch)| **ch == c).map(|(v, _)| *v).collect()).collect();
        let r = stats::grubin(&chains);
        rc.push(r);
        rows.push(vec![format!("{r:.6}")]);
    }
    let mut out = vec![String::new(), "Gelman-Rubin convergence diagnostic".to_string(), String::new()];
    out.push(format!("Number of chains     = {:>10}", d.nchains));
    out.push(format!("MCMC size, per chain = {:>10}", util::commas(d.ll.len() / d.nchains)));
    let rmax = rc.iter().copied().fold(0.0, f64::max);
    out.push(format!("Max Gelman-Rubin Rc  = {:>10.6}", rmax));
    out.push(String::new());
    out.extend(grouped_table(&labels, &["        Rc".to_string()], &rows, &[10]));
    if rmax > 1.1 && opt(&o, "nonote", 3).is_none() {
        out.push("Note: Rc values greater than 1.1 may indicate nonconvergence.".into());
    }
    let r = st.begin_r();
    r.set_mat("Rc", Matrix::row_vector(&rc).with_names(vec!["Rc".into()], labels));
    r.set("Rc_max", rmax);
    r.set("nchains", d.nchains as f64);
    Ok(out.join("\n"))
}

fn lookup_est<'a>(st: &'a Store, name: &str) -> Result<&'a Results, String> {
    if name == "." {
        return Ok(&st.e);
    }
    st.estimates.get(name).ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))
}

fn ic(st: &mut Store, rest: &str) -> Result<String, String> {
    let (names, o) = split_rest(rest);
    let names: Vec<String> = if names.trim().is_empty() { vec![".".to_string()] } else { names.split_whitespace().map(String::from).collect() };
    let mut rows = Vec::new();
    let mut mat = Matrix::zeros(names.len(), 3);
    let mut base = f64::NAN;
    let use_bf = opt(&o, "bayesfactor", 3).is_some();
    let mut labels = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let e = lookup_est(st, n)?;
        if !e.matrices.contains_key("_mcmc") {
            return Err(rc_err(301, format!("{n} is not a Bayesian estimation result")));
        }
        let dic = e.get("dic").unwrap_or(SYSMISS);
        let lml = e.get("lml_lm").unwrap_or(SYSMISS);
        if i == 0 {
            base = lml;
        }
        let bf = if ferrostat_core::missing::is_missing(lml) || ferrostat_core::missing::is_missing(base) { SYSMISS } else if use_bf { (lml - base).exp() } else { lml - base };
        rows.push(vec![num9(dic), num9(lml), if i == 0 || names.len() == 1 { ".".into() } else { num9(bf) }]);
        mat[(i, 0)] = dic;
        mat[(i, 1)] = lml;
        mat[(i, 2)] = if i == 0 { SYSMISS } else { bf };
        labels.push(if n == "." { "active".to_string() } else { n.clone() });
    }
    let mut out = vec![String::new(), "Bayesian information criteria".to_string(), String::new()];
    let bfh = if use_bf { "BF" } else { "log(BF)" };
    out.extend(grouped_table(&labels, &[format!("{:>10} {:>10} {:>10}", "DIC", "log(ML)", bfh)], &rows, &[10, 10, 10]));
    out.push("Note: Marginal likelihood (ML) is computed using Laplace-Metropolis approximation.".into());
    let r = st.begin_r();
    r.set_mat("ic", mat.with_names(labels, vec!["DIC".into(), "log(ML)".into(), if use_bf { "BF".into() } else { "log(BF)".into() }]));
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// bayestest
// ---------------------------------------------------------------------------

fn test_model(st: &mut Store, rest: &str) -> Result<String, String> {
    let (names, o) = split_rest(rest);
    let names: Vec<String> = if names.trim().is_empty() { vec![".".to_string()] } else { names.split_whitespace().map(String::from).collect() };
    let k = names.len();
    let priors: Vec<f64> = match opt(&o, "prior", 5) {
        Some(Some(p)) => p.split(|c: char| c.is_whitespace() || c == ',').filter(|s| !s.is_empty()).map(|s| s.parse::<f64>().map_err(|_| rc_err(198, "prior() invalid"))).collect::<Result<_, _>>()?,
        _ => vec![1.0 / k as f64; k],
    };
    if priors.len() != k {
        return Err(rc_err(198, "prior() must list one probability per model"));
    }
    let mut lml = Vec::new();
    for n in &names {
        let e = lookup_est(st, n)?;
        let v = e.get("lml_lm").unwrap_or(SYSMISS);
        if ferrostat_core::missing::is_missing(v) {
            return Err(rc_err(498, format!("log marginal likelihood of {n} not available (improper priors)")));
        }
        lml.push(v);
    }
    let mx = lml.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = lml.iter().zip(&priors).map(|(l, p)| (l - mx).exp() * p).collect();
    let s: f64 = w.iter().sum();
    let post: Vec<f64> = w.iter().map(|v| v / s).collect();
    let labels: Vec<String> = names.iter().map(|n| if n == "." { "active".to_string() } else { n.clone() }).collect();
    let rows: Vec<Vec<String>> = (0..k).map(|i| vec![num9(lml[i]), format!("{:.4}", priors[i]), format!("{:.4}", post[i])]).collect();
    let mut out = vec![String::new(), "Bayesian model tests".to_string(), String::new()];
    out.extend(grouped_table(&labels, &[format!("{:>10} {:>10} {:>10}", "log(ML)", "P(M)", "P(M|y)")], &rows, &[10, 10, 10]));
    out.push("Note: Marginal likelihood (ML) is computed using Laplace-Metropolis approximation.".into());
    let mut mat = Matrix::zeros(k, 3);
    for i in 0..k {
        mat[(i, 0)] = lml[i];
        mat[(i, 1)] = priors[i];
        mat[(i, 2)] = post[i];
    }
    let r = st.begin_r();
    r.set_mat("test", mat.with_names(labels, vec!["log(ML)".into(), "P(M)".into(), "P(M|y)".into()]));
    Ok(out.join("\n"))
}

fn test_interval(st: &mut Store, rest: &str) -> Result<String, String> {
    let d = draws_of(&st.e)?;
    // forms: ({p}, lower(#) upper(#)) ...   or   {p}, lower(#) upper(#)
    let mut specs: Vec<(String, Option<f64>, Option<f64>)> = Vec::new();
    let r = rest.trim();
    let parse_bounds = |o: &[(String, Option<String>)]| -> Result<(Option<f64>, Option<f64>), String> { Ok((optnum(o, "lower", 1)?, optnum(o, "upper", 1)?)) };
    let (main, gopts) = split_rest(r);
    if main.starts_with('(') {
        for w in text::words(&main) {
            let inner = w.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(&w);
            let (p, o) = split_rest(inner);
            let (lo, hi) = parse_bounds(&o)?;
            specs.push((p, lo, hi));
        }
    } else {
        let (lo, hi) = parse_bounds(&gopts)?;
        specs.push((main.clone(), lo, hi));
    }
    let mut out = vec![String::new(), "Interval tests".to_string()];
    out.extend(mcmc_header(st));
    out.push(String::new());
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    let mut mat = Matrix::zeros(0, 0);
    let mut matrows = Vec::new();
    for (p, lo, hi) in &specs {
        if lo.is_none() && hi.is_none() {
            return Err(rc_err(198, "lower() or upper() required"));
        }
        let its = items(&d, &st.e, p)?;
        for it in its {
            let ind: Vec<f64> = it.vals.iter().map(|v| (lo.map_or(true, |l| *v > l) && hi.map_or(true, |h| *v < h)) as i32 as f64).collect();
            let s = stats::summarize(&ind, 95.0, false, 0);
            let cond = format!("{} < {} < {}", lo.map_or("-inf".into(), |v| fmt_g(v, 9)), it.label, hi.map_or("+inf".into(), |v| fmt_g(v, 9)));
            out.push(format!("  prob{} : {cond}", labels.len() + 1));
            labels.push(it.label.clone());
            rows.push(vec![format!("{:.5}", s.mean), format!("{:.5}", s.sd), format!("{:.7}", s.mcse)]);
            matrows.push(vec![s.mean, s.sd, s.mcse]);
        }
    }
    out.push(String::new());
    out.extend(grouped_table(&labels, &[format!("{:>10} {:>11} {:>10}", "Mean", "Std. dev.", "MCSE")], &rows, &[10, 11, 10]));
    if !matrows.is_empty() {
        mat = Matrix::from_rows(&matrows);
    }
    let r = st.begin_r();
    r.set_mat("summary", mat.with_names(labels, vec!["mean".into(), "sd".into(), "mcse".into()]));
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// bayesgraph
// ---------------------------------------------------------------------------

fn kde(x: &[f64], npts: usize) -> Vec<(f64, f64)> {
    let n = x.len() as f64;
    let sd = stats::var(x).sqrt();
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let iqr = stats::quantile(&s, 0.75) - stats::quantile(&s, 0.25);
    let a = if iqr > 0.0 { sd.min(iqr / 1.349) } else { sd };
    let h = (0.9 * a * n.powf(-0.2)).max(1e-12);
    let lo = s[0] - 3.0 * h;
    let hi = s[s.len() - 1] + 3.0 * h;
    // bin then smooth for speed
    let nb = 512;
    let bw = (hi - lo) / nb as f64;
    let mut bins = vec![0.0; nb];
    for v in x {
        let k = (((v - lo) / bw) as usize).min(nb - 1);
        bins[k] += 1.0;
    }
    (0..npts).map(|i| {
        let t = lo + (hi - lo) * i as f64 / (npts - 1) as f64;
        let mut f = 0.0;
        for (k, c) in bins.iter().enumerate() {
            if *c > 0.0 {
                let c0 = lo + (k as f64 + 0.5) * bw;
                let z = (t - c0) / h;
                if z.abs() < 6.0 {
                    f += c * (-0.5 * z * z).exp();
                }
            }
        }
        (t, f / (n * h * (2.0 * std::f64::consts::PI).sqrt()))
    }).collect()
}

fn trace_graph(label: &str, x: &[f64]) -> Graph {
    let mut g = Graph::default();
    g.layers.push(Layer { kind: LayerKind::Line, points: x.iter().enumerate().map(|(i, v)| ((i + 1) as f64, *v)).collect(), label: label.into(), ..Default::default() });
    g.title = "Trace".into();
    g.xtitle = "Iteration number".into();
    g
}

fn hist_graph(label: &str, x: &[f64]) -> Graph {
    let mut g = Graph::default();
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let (lo, hi) = (s[0], s[s.len() - 1]);
    let nb = ((x.len() as f64).sqrt().min(50.0).max(5.0)) as usize;
    let w = ((hi - lo) / nb as f64).max(1e-12);
    let mut c = vec![0.0; nb];
    for v in x {
        c[(((v - lo) / w) as usize).min(nb - 1)] += 1.0;
    }
    let n = x.len() as f64;
    g.layers.push(Layer { kind: LayerKind::Bar, points: (0..nb).map(|k| (lo + (k as f64 + 0.5) * w, c[k] / (n * w))).collect(), width: w, label: label.into(), ..Default::default() });
    g.title = "Histogram".into();
    g.ytitle = "Density".into();
    g.xtitle = label.into();
    g
}

fn ac_graph(label: &str, x: &[f64], lags: usize) -> Graph {
    let mut g = Graph::default();
    let pts: Vec<(f64, f64)> = (0..=lags.min(x.len().saturating_sub(1))).map(|k| (k as f64, stats::acf(x, k))).collect();
    g.layers.push(Layer { kind: LayerKind::Range(crate::graph::RangeStyle::Spike), ranges: pts.iter().map(|(k, a)| (*k, 0.0, *a)).collect(), label: label.into(), ..Default::default() });
    g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, label: label.into(), ..Default::default() });
    g.title = "Autocorrelation".into();
    g.xtitle = "Lag".into();
    g.ytitle = "Autocorrelation".into();
    g
}

fn density_graph(label: &str, x: &[f64]) -> Graph {
    let mut g = Graph::default();
    g.layers.push(Layer { kind: LayerKind::Line, points: kde(x, 200), label: "overall".into(), ..Default::default() });
    let h = x.len() / 2;
    if h > 10 {
        g.layers.push(Layer { kind: LayerKind::Line, points: kde(&x[..h], 200), label: "1-half".into(), pattern: "dash".into(), ..Default::default() });
        g.layers.push(Layer { kind: LayerKind::Line, points: kde(&x[h..], 200), label: "2-half".into(), pattern: "dot".into(), ..Default::default() });
        g.legend = true;
    }
    g.title = "Density".into();
    g.xtitle = label.into();
    g
}

fn cusum_graph(label: &str, x: &[f64]) -> Graph {
    let m = stats::mean(x);
    let mut s = 0.0;
    let pts: Vec<(f64, f64)> = x.iter().enumerate().map(|(i, v)| {
        s += v - m;
        ((i + 1) as f64, s)
    }).collect();
    let mut g = Graph::default();
    g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: label.into(), ..Default::default() });
    g.ylines.push(0.0);
    g.title = format!("Cusum of {label}");
    g.xtitle = "Iteration number".into();
    g
}

fn graph(st: &mut Store, sub: &str, rest: &str, cmd: &Command) -> Result<String, String> {
    let d = draws_of(&st.e)?;
    let (spec, o) = split_rest(rest);
    let its = items(&d, &st.e, &spec)?;
    if its.is_empty() {
        return Err(rc_err(198, "no parameters specified"));
    }
    let lags = optnum(&o, "lags", 3)?.unwrap_or(40.0) as usize;
    let mut g = Graph::default();
    let one = |it: &Item| -> Result<Graph, String> {
        Ok(match sub {
            s if text::abbrev(s, "diagnostics", 4) => {
                let mut gg = Graph::default();
                gg.children = vec![trace_graph(&it.label, &it.vals), hist_graph(&it.label, &it.vals), ac_graph(&it.label, &it.vals, lags), density_graph(&it.label, &it.vals)];
                gg.cols = 2;
                gg.title = it.label.clone();
                gg
            }
            s if text::abbrev(s, "trace", 2) => trace_graph(&it.label, &it.vals),
            "ac" => ac_graph(&it.label, &it.vals, lags),
            s if text::abbrev(s, "histogram", 4) => hist_graph(&it.label, &it.vals),
            s if text::abbrev(s, "kdensity", 4) => density_graph(&it.label, &it.vals),
            s if text::abbrev(s, "cusum", 2) => cusum_graph(&it.label, &it.vals),
            o => return Err(rc_err(198, format!("bayesgraph {o} is unrecognized"))),
        })
    };
    if sub == "matrix" {
        let k = its.len();
        if k < 2 {
            return Err(rc_err(198, "bayesgraph matrix requires at least two parameters"));
        }
        let step = (its[0].vals.len() / 1000).max(1);
        for a in 0..k {
            for b in 0..k {
                let mut c = Graph::default();
                if a == b {
                    c.center_text = its[a].label.clone();
                } else {
                    c.layers.push(Layer { kind: LayerKind::Scatter, points: its[b].vals.iter().zip(&its[a].vals).step_by(step).map(|(x, y)| (*x, *y)).collect(), ..Default::default() });
                }
                g.children.push(c);
            }
        }
        g.cols = k;
    } else if its.len() == 1 {
        g = one(&its[0])?;
        if g.title.is_empty() || !g.children.is_empty() {
            g.title = its[0].label.clone();
        } else {
            g.title = format!("{} of {}", g.title, its[0].label);
        }
    } else {
        for it in &its {
            let mut c = one(it)?;
            c.title = if c.children.is_empty() { format!("{} of {}", c.title, it.label) } else { it.label.clone() };
            g.children.push(c);
        }
        g.cols = if its.len() > 1 && !sub.starts_with("diag") { 2 } else { 1 };
    }
    g.command = format!("bayesgraph {sub} {rest}");
    crate::graph::finish(&mut g, cmd, st)
}
