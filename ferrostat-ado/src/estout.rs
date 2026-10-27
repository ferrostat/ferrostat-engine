//! Publication-style tables from stored results (Ferrostat's own
//! implementation of the community estout package): `eststo` stores models,
//! `estpost` posts descriptive statistics as e() row vectors, `estadd` adds
//! results, and `esttab` / `estout` lay them out on screen or in
//! fixed/tab/CSV/RTF/LaTeX/HTML files.

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::{rc_err, Env};
use ferrostat_cmd::format::{fmt_g, fmt_trim};
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::summarize::{summarize_values, Summary};
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{vfs, DataSet, Value};
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options, split_top_level_words};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

type Opts = Vec<(String, Option<String>)>;

/// Option `name`, accepted when written with at least `min` characters.
fn opt<'a>(opts: &'a Opts, name: &str, min: usize) -> Option<&'a Option<String>> {
    opts.iter()
        .find(|(o, _)| {
            let o = o.to_lowercase();
            o.len() >= min.min(name.len()) && name.starts_with(o.as_str())
        })
        .map(|(_, a)| a)
}

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opt(opts, name, min).is_some()
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opt(opts, name, min).and_then(|a| a.clone())
}

fn split_main(r: &str) -> (String, Opts) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

/// Words of an option argument, unquoted.
fn words(s: &str) -> Vec<String> {
    split_top_level_words(s).iter().map(|w| unquote(w)).collect()
}

/// `*` / `?` wildcard match.
fn glob(pat: &str, s: &str) -> bool {
    fn rec(p: &[char], s: &[char]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some('*') => (0..=s.len()).any(|k| rec(&p[1..], &s[k..])),
            Some('?') => !s.is_empty() && rec(&p[1..], &s[1..]),
            Some(c) => s.first() == Some(c) && rec(&p[1..], &s[1..]),
        }
    }
    let p: Vec<char> = pat.chars().collect();
    let s: Vec<char> = s.chars().collect();
    rec(&p, &s)
}

fn missing_to(v: f64) -> f64 {
    if is_missing(v) {
        SYSMISS
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// eststo
// ---------------------------------------------------------------------------

/// Names stored by `eststo`, in storage order (kept in the global `eststo`).
fn eststo_names(me: &AdoInterpreter) -> Vec<String> {
    me.engine.store.globals.get("eststo").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

fn set_eststo_names(me: &mut AdoInterpreter, names: &[String]) {
    if names.is_empty() {
        me.engine.store.globals.remove("eststo");
    } else {
        me.engine.store.globals.insert("eststo".into(), names.join(" "));
    }
}

pub fn eststo(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (head, inner) = match find_top_level(r, ':') {
        Some(i) => (r[..i].to_string(), Some(r[i + 1..].trim().to_string())),
        None => (r.to_string(), None),
    };
    let (main, opts) = split_main(&head);
    let w: Vec<String> = main.split_whitespace().map(String::from).collect();
    let mut names = eststo_names(me);
    if inner.is_none() {
        match w.first().map(String::as_str) {
            Some("clear") => {
                for n in &names {
                    me.engine.store.estimates.remove(n);
                }
                set_eststo_names(me, &[]);
                me.engine.store.globals.remove("eststo_counter");
                return Ok(String::new());
            }
            Some("dir") => {
                let st = &me.engine.store;
                return Ok(names
                    .iter()
                    .filter_map(|n| st.estimates.get(n).map(|e| format!("{:<14} {}", n, e.macros.get("cmd").cloned().unwrap_or_default())))
                    .collect::<Vec<_>>()
                    .join("\n"));
            }
            Some("drop") => {
                let pats = &w[1..];
                let (gone, keep): (Vec<String>, Vec<String>) = names.into_iter().partition(|n| pats.iter().any(|p| glob(p, n)));
                for n in &gone {
                    me.engine.store.estimates.remove(n);
                }
                set_eststo_names(me, &keep);
                return Ok(String::new());
            }
            _ => {}
        }
    }
    if let Some(cmd) = &inner {
        me.exec_text(cmd, None)?;
    }
    if me.engine.store.e.is_empty() {
        return Err(rc_err(301, "last estimates not found"));
    }
    let (name, note) = match w.first() {
        Some(n) => (n.clone(), String::new()),
        None => {
            let k = me.engine.store.globals.get("eststo_counter").and_then(|c| c.parse::<usize>().ok()).unwrap_or(0) + 1;
            me.engine.store.globals.insert("eststo_counter".into(), k.to_string());
            let n = format!("est{k}");
            let note = format!("({n} stored)");
            (n, note)
        }
    };
    if let Some(t) = arg(&opts, "title", 2) {
        me.engine.store.e.set_str("_estimates_title", unquote(&t));
    }
    if let Some(a) = arg(&opts, "addscalars", 4) {
        let w = split_top_level_words(&a);
        for pair in w.chunks(2) {
            if pair.len() == 2 {
                let v = eval_num(me, &pair[1])?;
                me.engine.store.e.set(&pair[0], v);
            }
        }
    }
    me.engine.store.e.set_str("_estimates_name", name.clone());
    let e = me.engine.store.e.clone();
    me.engine.store.estimates.insert(name.clone(), e);
    me.engine.store.estimates_order.retain(|n| *n != name);
    me.engine.store.estimates_order.push(name.clone());
    names.retain(|n| *n != name);
    names.push(name);
    set_eststo_names(me, &names);
    Ok(note)
}

fn eval_num(me: &AdoInterpreter, text: &str) -> Result<f64, String> {
    let e = parse_expression(text).map_err(|e| rc_err(198, e.message))?;
    Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?.num()
}

// ---------------------------------------------------------------------------
// estadd
// ---------------------------------------------------------------------------

pub fn estadd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, targets) = match find_top_level(r, ':') {
        Some(i) => (r[..i].trim().to_string(), Some(r[i + 1..].split_whitespace().map(String::from).collect::<Vec<_>>())),
        None => (r.trim().to_string(), None),
    };
    let (sub, rest) = body.split_once(char::is_whitespace).map(|(a, b)| (a.to_lowercase(), b.trim().to_string())).unwrap_or((body.to_lowercase(), String::new()));
    // name = value, name value, or r(name)
    let name_value = |rest: &str| -> (String, String) {
        let rest = rest.trim();
        if let Some((n, v)) = rest.split_once('=') {
            (n.trim().to_string(), v.trim().to_string())
        } else {
            match rest.split_once(char::is_whitespace) {
                Some((n, v)) => (n.to_string(), v.trim().to_string()),
                None => {
                    // `estadd scalar r(mean)` adds e(mean)
                    let n = rest.trim_start_matches("r(").trim_end_matches(')').to_string();
                    (n, rest.to_string())
                }
            }
        }
    };
    enum Add {
        Scalar(String, f64),
        Local(String, String),
        Matrix(String, Matrix),
        Many(Vec<(String, f64)>),
    }
    let add = match sub.as_str() {
        "scalar" | "sca" => {
            let (n, v) = name_value(&rest);
            Add::Scalar(n, eval_num(me, &v)?)
        }
        "local" | "loc" => {
            let (n, v) = match rest.split_once(char::is_whitespace) {
                Some((n, v)) => (n.to_string(), unquote(v.trim())),
                None => (rest.clone(), String::new()),
            };
            Add::Local(n, v)
        }
        "matrix" | "mat" => {
            let (n, v) = name_value(&rest);
            let st = &me.engine.store;
            let m = if let Some(x) = v.strip_prefix("e(").and_then(|x| x.strip_suffix(')')) {
                st.e.matrices.get(x).cloned()
            } else if let Some(x) = v.strip_prefix("r(").and_then(|x| x.strip_suffix(')')) {
                st.r.matrices.get(x).cloned()
            } else {
                st.matrices.get(&v).cloned()
            };
            Add::Matrix(n, m.ok_or_else(|| rc_err(111, format!("matrix {v} not found")))?)
        }
        "ysumm" => {
            let e = &me.engine.store.e;
            let dep = e.macros.get("depvar").and_then(|d| d.split_whitespace().next().map(String::from)).ok_or_else(|| rc_err(301, "e(depvar) not found"))?;
            let col = util::col(&me.engine.dataset, &dep)?;
            let x: Vec<f64> = col.iter().enumerate().filter(|(i, v)| !is_missing(**v) && e.sample.as_ref().map_or(true, |s| s.get(*i).copied().unwrap_or(false))).map(|(_, v)| *v).collect();
            let s = summarize_values(&x, None, None);
            Add::Many(vec![("ymean".into(), s.mean), ("ysd".into(), s.sd()), ("ymin".into(), s.min), ("ymax".into(), s.max)])
        }
        other => return Err(rc_err(198, format!("estadd {other}: invalid subcommand"))),
    };
    let apply = |e: &mut Results| match &add {
        Add::Scalar(n, v) => e.set(n, *v),
        Add::Local(n, v) => e.set_str(n, v.clone()),
        Add::Matrix(n, m) => e.set_mat(n, m.clone()),
        Add::Many(v) => {
            for (n, x) in v {
                e.set(n, *x);
            }
        }
    };
    let st = &mut me.engine.store;
    match &targets {
        Some(t) => {
            let keys: Vec<String> = st.estimates.keys().filter(|k| t.iter().any(|p| glob(p, k))).cloned().collect();
            if keys.is_empty() {
                return Err(rc_err(111, format!("estimation result {} not found", t.join(" "))));
            }
            for k in keys {
                apply(st.estimates.get_mut(&k).unwrap());
            }
        }
        None => {
            if st.e.is_empty() {
                return Err(rc_err(301, "last estimates not found"));
            }
            apply(&mut st.e);
            // results stored by name are kept in sync with the active copy
            if let Some(n) = st.e.macros.get("_estimates_name").cloned() {
                if st.estimates.contains_key(&n) {
                    let e = st.e.clone();
                    st.estimates.insert(n, e);
                }
            }
        }
    }
    Ok(match add {
        Add::Scalar(n, v) => format!("\nadded scalar:\n{:>20} =  {}", format!("e({n})"), fmt_g(v, 9)),
        Add::Local(n, v) => format!("\nadded macro:\n{:>20} : \"{}\"", format!("e({n})"), v),
        Add::Matrix(n, m) => format!("\nadded matrix:\n{:>20} :  {} x {}", format!("e({n})"), m.rows, m.cols),
        Add::Many(v) => format!("\nadded scalars:\n{}", v.iter().map(|(n, x)| format!("{:>20} =  {}", format!("e({n})"), fmt_g(*x, 9))).collect::<Vec<_>>().join("\n")),
    })
}

// ---------------------------------------------------------------------------
// estpost
// ---------------------------------------------------------------------------

pub fn estpost(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let (sub, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a.to_lowercase(), b.trim().to_string())).unwrap_or((r.to_lowercase(), String::new()));
    let is = |full: &str, min: usize| sub.len() >= min && full.starts_with(sub.as_str());
    if is("summarize", 2) {
        post_summarize(me, &rest)
    } else if is("ttest", 5) {
        post_ttest(me, &rest)
    } else if is("tabstat", 4) {
        post_tabstat(me, &rest)
    } else if is("tabulate", 3) {
        post_tabulate(me, &rest)
    } else {
        Err(rc_err(199, format!("estpost {sub}: unsupported subcommand")))
    }
}

/// Variables named in `main` and the text after the varlist (if/in/weights).
fn varlist_and_tail(me: &AdoInterpreter, verb: &str, main: &str) -> Result<(Vec<String>, String), String> {
    let cmd = parse_line(&format!("{verb} {main}")).map_err(|e| rc_err(198, e.message))?;
    let vl = cmd.main.trim();
    let tail = match main.find(vl) {
        Some(p) if !vl.is_empty() => main[p + vl.len()..].to_string(),
        _ => main.to_string(),
    };
    let ds = &me.engine.dataset;
    let vars = if vl.is_empty() {
        ds.vars.iter().filter(|v| !v.var_type.is_string()).map(|v| v.name.clone()).collect()
    } else {
        varlist::expand(ds, vl)?
    };
    Ok((vars, tail))
}

/// Add `!missing(vars)` to the condition in `tail` (listwise deletion).
fn listwise_tail(tail: &str, vars: &[String]) -> String {
    let cond = format!("!missing({})", vars.join(","));
    match tail.find(" if ").or_else(|| tail.trim_start().starts_with("if ").then_some(tail.len() - tail.trim_start().len())) {
        Some(p) => {
            let start = tail[p..].find("if").unwrap() + p + 2;
            let rest = &tail[start..];
            let end = rest.find(" in ").or_else(|| rest.find('[')).unwrap_or(rest.len());
            format!("{} if ({}) & {}{}", &tail[..p], rest[..end].trim(), cond, &rest[end..])
        }
        None => format!(" if {cond}{tail}"),
    }
}

/// Listing of posted row vectors: one line per column name.
fn posted_listing(cols: &[String], stats: &[(String, Vec<f64>)]) -> String {
    let mut out = vec![String::new()];
    out.push(format!("{:>12} |{}", "", stats.iter().map(|(n, _)| format!("{:>12}", format!("e({n})"))).collect::<String>()));
    out.push(format!("-------------+{}", "-".repeat(12 * stats.len())));
    for (j, c) in cols.iter().enumerate() {
        out.push(format!("{:>12} |{}", ferrostat_cmd::summarize::abbrev(c, 12), stats.iter().map(|(_, v)| format!("{:>12}", fmt_g(v[j], 9))).collect::<String>()));
    }
    out.join("\n")
}

fn post(me: &mut AdoInterpreter, subcmd: &str, cols: &[String], stats: &[(String, Vec<f64>)], n: f64) {
    let mut e = Results::new();
    for (name, v) in stats {
        let m = Matrix::row_vector(&v.iter().map(|x| missing_to(*x)).collect::<Vec<_>>()).with_names(vec![name.clone()], cols.to_vec());
        e.set_mat(name, m);
    }
    e.set("N", n);
    e.set_str("cmd", "estpost");
    e.set_str("subcmd", subcmd);
    me.engine.store.e = e;
}

fn post_summarize(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (main, opts) = split_main(rest);
    let detail = has(&opts, "detail", 1);
    let (vars, mut tail) = varlist_and_tail(me, "summarize", &main)?;
    if has(&opts, "listwise", 4) {
        tail = listwise_tail(&tail, &vars);
    }
    let mut names = vec!["count", "sum_w", "mean", "Var", "sd", "min", "max", "sum"];
    if detail {
        names.extend(["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "skewness", "kurtosis"]);
    }
    let mut stats: Vec<(String, Vec<f64>)> = names.iter().map(|n| (n.to_string(), Vec::new())).collect();
    let mut nmax: f64 = 0.0;
    for v in &vars {
        let line = format!("summarize {v} {tail}{}", if detail { ", detail" } else { "" });
        let cmd = parse_line(&line).map_err(|e| rc_err(198, e.message))?;
        me.engine.execute(&cmd)?;
        let rr = &me.engine.store.r;
        for (n, vals) in stats.iter_mut() {
            let key = if n == "count" { "N" } else { n.as_str() };
            vals.push(rr.get(key).unwrap_or(SYSMISS));
        }
        nmax = nmax.max(rr.get("N").unwrap_or(0.0));
    }
    let shown: Vec<(String, Vec<f64>)> = stats.iter().filter(|(n, _)| detail || ["count", "mean", "sd", "min", "max"].contains(&n.as_str())).cloned().collect();
    post(me, "summarize", &vars, &stats, nmax);
    Ok(posted_listing(&vars, &shown))
}

fn post_ttest(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (main, opts) = split_main(rest);
    let by = arg(&opts, "by", 2).ok_or_else(|| rc_err(198, "option by() required"))?;
    let (vars, mut tail) = varlist_and_tail(me, "summarize", &main)?;
    if has(&opts, "listwise", 4) {
        tail = listwise_tail(&tail, &vars);
    }
    let extra: String = ["unequal", "welch"].iter().filter(|o| has(&opts, o, 2)).map(|o| format!(" {o}")).collect();
    let names = ["b", "count", "se", "t", "df_t", "p_l", "p", "p_u", "N_1", "mu_1", "sd_1", "N_2", "mu_2", "sd_2"];
    let mut stats: Vec<(String, Vec<f64>)> = names.iter().map(|n| (n.to_string(), Vec::new())).collect();
    let mut ntot: f64 = 0.0;
    for v in &vars {
        let cmd = parse_line(&format!("ttest {v} {tail}, by({by}){extra}")).map_err(|e| rc_err(198, e.message))?;
        me.engine.execute(&cmd)?;
        let rr = &me.engine.store.r;
        let g = |k: &str| rr.get(k).unwrap_or(SYSMISS);
        for (n, vals) in stats.iter_mut() {
            vals.push(match n.as_str() {
                "b" => g("mu_1") - g("mu_2"),
                "count" => g("N_1") + g("N_2"),
                k => g(k),
            });
        }
        ntot = ntot.max(g("N_1") + g("N_2"));
    }
    let shown: Vec<(String, Vec<f64>)> = stats.iter().filter(|(n, _)| ["b", "count", "t", "df_t", "p"].contains(&n.as_str())).cloned().collect();
    post(me, "ttest", &vars, &stats, ntot);
    let se: Vec<f64> = stats.iter().find(|s| s.0 == "se").map(|s| s.1.clone()).unwrap_or_default();
    let mut vm = Matrix::from_diag(&se.iter().map(|s| s * s).collect::<Vec<_>>());
    vm.rownames = vars.clone();
    vm.colnames = vars.clone();
    me.engine.store.e.set_mat("V", vm);
    me.engine.store.e.set_str("byvar", by.trim());
    Ok(posted_listing(&vars, &shown))
}

/// One statistic of a summary, by its tabstat name.
fn summary_stat(name: &str, s: &Summary) -> Result<f64, String> {
    if s.n == 0.0 {
        return Ok(if matches!(name, "count" | "n" | "sum") { 0.0 } else { SYSMISS });
    }
    Ok(match name {
        "mean" => s.mean,
        "count" | "n" => s.n,
        "sum" => s.sum,
        "max" => s.max,
        "min" => s.min,
        "range" | "r" => s.max - s.min,
        "sd" => s.sd(),
        "variance" | "var" => s.var,
        "cv" => s.sd() / s.mean,
        "semean" => s.sd() / s.n.sqrt(),
        "skewness" => s.moments().0,
        "kurtosis" => s.moments().1,
        "median" | "p50" => s.pctile(50.0),
        "iqr" => s.pctile(75.0) - s.pctile(25.0),
        p if p.starts_with('p') => s.pctile(p[1..].parse().map_err(|_| rc_err(198, format!("{p} invalid statistic")))?),
        other => return Err(rc_err(198, format!("{other} invalid statistic"))),
    })
}

/// Distinct values of `var` among selected observations: (name, label,
/// observation -> group index).
fn levels(ds: &DataSet, var: &str, mask: &[bool], with_missing: bool) -> Result<(Vec<String>, Vec<String>, Vec<Option<usize>>), String> {
    let v = ds.get_var(var).ok_or_else(|| rc_err(111, format!("variable {var} not found")))?;
    let n = ds.n_obs;
    let mut keys: Vec<(f64, String)> = Vec::new();
    let mut obs_key: Vec<Option<(f64, String)>> = vec![None; n];
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let k = match &v.data[i] {
            Value::Str(_) | Value::StrL(..) => {
                let s = util::cell_text(ds, v, i, false);
                if s.is_empty() && !with_missing {
                    continue;
                }
                (0.0, s)
            }
            x => {
                let f = x.num();
                if is_missing(f) && !with_missing {
                    continue;
                }
                (f, String::new())
            }
        };
        if !keys.contains(&k) {
            keys.push(k.clone());
        }
        obs_key[i] = Some(k);
    }
    keys.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let vl = ds.value_labels.get(&v.value_label_name);
    let mut names = Vec::new();
    let mut labels = Vec::new();
    for (f, s) in &keys {
        if v.var_type.is_string() {
            names.push(s.replace(|c: char| c.is_whitespace() || c == ':', "_"));
            labels.push(s.clone());
        } else {
            let t = if is_missing(*f) { ".".to_string() } else { fmt_g(*f, 9).trim().to_string() };
            labels.push(vl.and_then(|l| l.lookup(*f)).map(String::from).unwrap_or_else(|| t.clone()));
            names.push(t);
        }
    }
    let idx = obs_key.into_iter().map(|k| k.and_then(|k| keys.iter().position(|x| *x == k))).collect();
    Ok((names, labels, idx))
}

fn mask_for(me: &AdoInterpreter, verb: &str, main: &str) -> Result<Vec<bool>, String> {
    let cmd = parse_line(&format!("{verb} {main}")).map_err(|e| rc_err(198, e.message))?;
    util::touse(&me.engine.dataset, &me.engine.store, &cmd)
}

fn label_macro(names: &[String], labels: &[String]) -> String {
    names.iter().zip(labels).map(|(n, l)| format!("{n} \"{l}\"")).collect::<Vec<_>>().join(" ")
}

fn post_tabstat(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (main, opts) = split_main(rest);
    let stats: Vec<String> = arg(&opts, "statistics", 1).map(|s| s.split_whitespace().map(|x| x.to_lowercase()).collect()).unwrap_or_else(|| vec!["mean".into()]);
    let (vars, _) = varlist_and_tail(me, "summarize", &main)?;
    let mut mask = mask_for(me, "summarize", &main)?;
    if has(&opts, "listwise", 4) {
        util::markout(&me.engine.dataset, &mut mask, &vars);
    }
    let by_stats = arg(&opts, "columns", 1).map_or(false, |c| c.trim().starts_with('s'));
    let ds = &me.engine.dataset;
    // groups: (equation name, mask)
    let mut groups: Vec<(String, Vec<bool>)> = Vec::new();
    let mut labels = String::new();
    if let Some(by) = arg(&opts, "by", 2) {
        let by = varlist::expand(ds, &by)?.into_iter().next().unwrap_or_default();
        let (names, labs, idx) = levels(ds, &by, &mask, has(&opts, "missing", 1))?;
        for (k, n) in names.iter().enumerate() {
            groups.push((labs[k].replace(|c: char| c.is_whitespace() || c == ':', "_").to_string().replace("__", "_"), idx.iter().map(|g| *g == Some(k)).collect()));
            let _ = n;
        }
        labels = label_macro(&names, &labs);
        if !has(&opts, "nototal", 3) {
            groups.push(("Total".into(), mask.clone()));
        }
    } else {
        groups.push((String::new(), mask.clone()));
    }
    let mut cols = Vec::new();
    let mut table: Vec<(String, Vec<f64>)> = if by_stats { vars.iter().map(|v| (v.clone(), Vec::new())).collect() } else { stats.iter().map(|s| (s.clone(), Vec::new())).collect() };
    for (eq, gm) in &groups {
        let prefix = if eq.is_empty() { String::new() } else { format!("{eq}:") };
        let sums: Vec<Summary> = vars
            .iter()
            .map(|v| {
                let c = util::col(ds, v)?;
                let x: Vec<f64> = c.iter().zip(gm).filter(|(x, m)| **m && !is_missing(**x)).map(|(x, _)| *x).collect();
                Ok(summarize_values(&x, None, None))
            })
            .collect::<Result<_, String>>()?;
        if by_stats {
            for s in &stats {
                cols.push(format!("{prefix}{s}"));
            }
            for (k, (_, vals)) in table.iter_mut().enumerate() {
                for s in &stats {
                    vals.push(summary_stat(s, &sums[k])?);
                }
            }
        } else {
            for v in &vars {
                cols.push(format!("{prefix}{v}"));
            }
            for (s, vals) in table.iter_mut() {
                for sm in &sums {
                    vals.push(summary_stat(s, sm)?);
                }
            }
        }
    }
    let n = mask.iter().filter(|m| **m).count() as f64;
    post(me, "tabstat", &cols, &table, n);
    if !labels.is_empty() {
        me.engine.store.e.set_str("labels", labels);
    }
    Ok(posted_listing(&cols, &table))
}

fn post_tabulate(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (main, opts) = split_main(rest);
    let (vars, _) = varlist_and_tail(me, "summarize", &main)?;
    let mask = mask_for(me, "summarize", &main)?;
    let miss = has(&opts, "missing", 1);
    let total = !has(&opts, "nototal", 3);
    let ds = &me.engine.dataset;
    match vars.as_slice() {
        [v] => {
            let (mut names, labels, idx) = levels(ds, v, &mask, miss)?;
            let mut counts = vec![0.0; names.len()];
            for g in idx.iter().flatten() {
                counts[*g] += 1.0;
            }
            let n: f64 = counts.iter().sum();
            let mut pct: Vec<f64> = counts.iter().map(|c| 100.0 * c / n).collect();
            let mut cum = Vec::new();
            let mut acc = 0.0;
            for p in &pct {
                acc += p;
                cum.push(acc);
            }
            let lm = label_macro(&names, &labels);
            if total {
                names.push("Total".into());
                counts.push(n);
                pct.push(100.0);
                cum.push(SYSMISS);
            }
            let stats = vec![("b".to_string(), counts), ("pct".to_string(), pct), ("cumpct".to_string(), cum)];
            post(me, "tabulate", &names, &stats, n);
            let e = &mut me.engine.store.e;
            e.set_str("labels", lm);
            e.set_str("depvar", v.clone());
            Ok(posted_listing(&names, &stats))
        }
        [rv, cv] => {
            let (rn, rl, ri) = levels(ds, rv, &mask, miss)?;
            let (cn, cl, ci) = levels(ds, cv, &mask, miss)?;
            let (nr, nc) = (rn.len(), cn.len());
            let mut cnt = vec![vec![0.0; nc + 1]; nr + 1];
            for i in 0..ds.n_obs {
                if let (Some(r), Some(c)) = (ri[i], ci[i]) {
                    cnt[r][c] += 1.0;
                    cnt[r][nc] += 1.0;
                    cnt[nr][c] += 1.0;
                    cnt[nr][nc] += 1.0;
                }
            }
            let n = cnt[nr][nc];
            let (rlim, clim) = if total { (nr + 1, nc + 1) } else { (nr, nc) };
            let mut names = Vec::new();
            let mut b = Vec::new();
            let mut pct = Vec::new();
            let mut colpct = Vec::new();
            let mut rowpct = Vec::new();
            for c in 0..clim {
                let eq = if c == nc { "Total".to_string() } else { cn[c].clone() };
                for r in 0..rlim {
                    let row = if r == nr { "Total".to_string() } else { rn[r].clone() };
                    names.push(format!("{eq}:{row}"));
                    let x = cnt[r][c];
                    b.push(x);
                    pct.push(100.0 * x / n);
                    colpct.push(100.0 * x / cnt[nr][c]);
                    rowpct.push(100.0 * x / cnt[r][nc]);
                }
            }
            let stats = vec![("b".to_string(), b), ("pct".to_string(), pct), ("colpct".to_string(), colpct), ("rowpct".to_string(), rowpct)];
            post(me, "tabulate", &names, &stats, n);
            let e = &mut me.engine.store.e;
            e.set_str("labels", label_macro(&rn, &rl));
            e.set_str("eqlabels", label_macro(&cn, &cl));
            e.set_str("rowvar", rv.clone());
            e.set_str("colvar", cv.clone());
            Ok(posted_listing(&names, &stats))
        }
        _ => Err(rc_err(198, "estpost tabulate: one or two variables required")),
    }
}

// ---------------------------------------------------------------------------
// esttab / estout
// ---------------------------------------------------------------------------

/// One statistic shown in each coefficient cell (`cells()` element).
#[derive(Debug, Clone, Default)]
struct Elem {
    stat: String,
    fmt: Vec<String>,
    par: Option<(String, String)>,
    star: bool,
    label: Option<String>,
}

fn parse_elem(w: &str) -> Elem {
    let (name, inner) = match w.find('(') {
        Some(i) => (&w[..i], w[i + 1..].strip_suffix(')').unwrap_or(&w[i + 1..])),
        None => (w, ""),
    };
    let mut el = Elem { stat: name.trim().to_string(), ..Default::default() };
    for (o, a) in split_options(inner) {
        match o.to_lowercase().as_str() {
            "fmt" | "f" => el.fmt = words(&a.unwrap_or_default()),
            "par" | "parentheses" => {
                let w = a.map(|a| words(&a)).unwrap_or_default();
                el.par = Some(match w.as_slice() {
                    [l, r] => (l.clone(), r.clone()),
                    _ => ("(".into(), ")".into()),
                });
            }
            "brackets" | "bracket" => el.par = Some(("[".into(), "]".into())),
            "star" => el.star = true,
            "label" | "l" => el.label = a.map(|a| unquote(&a)),
            _ => {}
        }
    }
    el
}

/// `cells()` specification: rows of elements; a quoted group shares one row.
fn parse_cells(spec: &str) -> Vec<Vec<Elem>> {
    let mut rows = Vec::new();
    for w in split_top_level_words(spec) {
        if w == "none" {
            continue;
        }
        if w.starts_with('"') || w.starts_with("`\"") {
            let row: Vec<Elem> = split_top_level_words(&unquote(&w)).iter().map(|e| parse_elem(e)).collect();
            if !row.is_empty() {
                rows.push(row);
            }
        } else {
            rows.push(vec![parse_elem(&w)]);
        }
    }
    rows
}

#[derive(Debug, Clone)]
struct StatRow {
    name: String,
    label: String,
    fmt: Option<String>,
}

struct Spec {
    cells: Vec<Vec<Elem>>,
    stats: Vec<StatRow>,
    stars: Vec<(String, f64)>,
    numbers: bool,
    titles: Vec<String>,
    collabels: Option<Vec<String>>,
    label: bool,
    keep: Vec<String>,
    drop: Vec<String>,
    order: Vec<String>,
    coeflabels: Vec<(String, String)>,
    eqlabels: Vec<String>,
    title: String,
    notes: Vec<String>,
    gaps: bool,
    unstack: bool,
    level: f64,
    compress: bool,
    lines: bool,
    stat_default_fmt: &'static str,
}

/// Number formatting: `#` decimals, `a#` adaptive, or a display format.
fn fmt_num(v: f64, f: &str) -> String {
    if is_missing(v) {
        return ".".into();
    }
    let f = f.trim();
    if let Ok(d) = f.parse::<usize>() {
        return format!("{:.*}", d, v);
    }
    if let Some(a) = f.strip_prefix('a') {
        let sig: i32 = a.parse().unwrap_or(3);
        if v == 0.0 {
            return "0".into();
        }
        let d = (sig - v.abs().log10().trunc() as i32).clamp(1, 9);
        return format!("{:.*}", d as usize, v);
    }
    if f.starts_with('%') {
        return fmt_trim(v, f);
    }
    fmt_g(v, 9).trim().to_string()
}

fn mat_get(e: &Results, name: &str, key: &str) -> Option<f64> {
    let m = e.matrices.get(name)?;
    let bare = key.rsplit(':').next().unwrap_or(key);
    let j = m.colnames.iter().position(|c| c == key).or_else(|| if key.contains(':') { None } else { m.colnames.iter().position(|c| c.rsplit(':').next() == Some(bare)) })?;
    (m.rows >= 1).then(|| m.get(0, j))
}

fn var_get(e: &Results, key: &str) -> Option<f64> {
    let m = e.matrices.get("V")?;
    let j = m.colnames.iter().position(|c| c == key)?;
    Some(m.get(j, j))
}

fn is_base(key: &str) -> bool {
    let bare = key.rsplit(':').next().unwrap_or(key);
    bare.split('#').all(|p| p.split('.').next().map_or(false, |x| x.ends_with('b') && x.len() > 1 && x[..x.len() - 1].chars().all(|c| c.is_ascii_digit())))
}

fn is_omitted(key: &str) -> bool {
    let bare = key.rsplit(':').next().unwrap_or(key);
    bare.split('#').any(|p| p.split('.').next().map_or(false, |x| x.ends_with('o') && p.contains('.')))
}

/// Value of statistic `stat` for coefficient `key` of one model.
fn coef_value(e: &Results, stat: &str, key: &str, level: f64) -> Option<f64> {
    let b = || mat_get(e, "b", key);
    let se = || mat_get(e, "se", key).or_else(|| var_get(e, key).map(|v| v.max(0.0).sqrt()));
    let base = is_base(key) && e.matrices.contains_key("V");
    if base && stat != "b" && stat != "_b" {
        return b().map(|_| SYSMISS);
    }
    let df = e.get("df_r");
    let crit = || match df {
        Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    match stat {
        "b" | "_b" => b(),
        "se" | "_se" => se(),
        "var" => var_get(e, key),
        "t" | "z" => mat_get(e, stat, key).or_else(|| Some(b()? / se()?)),
        "p" => mat_get(e, "p", key).or_else(|| {
            let t = b()? / se()?;
            if !t.is_finite() {
                return Some(SYSMISS);
            }
            Some(match df {
                Some(d) => 2.0 * dist::ttail(d, t.abs()),
                None => 2.0 * (1.0 - dist::normal(t.abs())),
            })
        }),
        "ci_l" => Some(b()? - crit() * se()?),
        "ci_u" => Some(b()? + crit() * se()?),
        other => mat_get(e, other, key),
    }
}

fn stars_for(p: Option<f64>, stars: &[(String, f64)]) -> String {
    let Some(p) = p.filter(|p| !is_missing(*p)) else { return String::new() };
    stars.iter().filter(|(_, t)| p < *t).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(s, _)| s.clone()).unwrap_or_default()
}

/// A table cell: text and star symbols.
type Cell = (String, String);

#[derive(Default)]
struct Grid {
    title: String,
    ncols: usize,
    head: Vec<(String, Vec<Cell>)>,
    body: Vec<(String, Vec<Cell>)>,
    foot: Vec<(String, Vec<Cell>)>,
    notes: Vec<String>,
    /// LaTeX/HTML/RTF: spans of model columns, by first column
    spans: Vec<(usize, usize)>,
}

struct Col {
    m: usize,
    eq: Option<String>,
    sub: usize,
}

fn coef_label(ds: &DataSet, spec: &Spec, key: &str, extra: &[(String, String)]) -> String {
    let bare = key.rsplit(':').next().unwrap_or(key);
    if let Some((_, l)) = spec.coeflabels.iter().chain(extra).find(|(n, _)| n == key || n == bare) {
        return l.clone();
    }
    if !spec.label {
        return bare.to_string();
    }
    if bare == "_cons" {
        return "Constant".into();
    }
    let part_label = |p: &str| -> String {
        if let Some((lev, var)) = p.split_once('.') {
            let vname = var.to_string();
            if let Some(v) = ds.get_var(&vname) {
                let num: String = lev.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
                if let Ok(x) = num.parse::<f64>() {
                    if let Some(l) = ds.value_labels.get(&v.value_label_name).and_then(|vl| vl.lookup(x)) {
                        return l.to_string();
                    }
                }
                if lev.starts_with('c') {
                    return if v.label.is_empty() { vname } else { v.label.clone() };
                }
            }
            return p.to_string();
        }
        match ds.get_var(p) {
            Some(v) if !v.label.is_empty() => v.label.clone(),
            _ => p.to_string(),
        }
    };
    bare.split('#').map(part_label).collect::<Vec<_>>().join(" # ")
}

fn stat_text(e: &Results, s: &StatRow, default_fmt: &str) -> String {
    let v = match s.name.as_str() {
        "aic" | "bic" if e.get(&s.name).is_none() => {
            let (ll, k, n) = (e.get("ll"), e.get("rank").or_else(|| e.get("df_m").map(|d| d + 1.0)), e.get("N"));
            match (ll, k, n) {
                (Some(ll), Some(k), Some(n)) => Some(if s.name == "aic" { -2.0 * ll + 2.0 * k } else { -2.0 * ll + k * n.ln() }),
                _ => None,
            }
        }
        n => e.get(n),
    };
    match v {
        Some(v) => match &s.fmt {
            Some(f) => fmt_num(v, f),
            None if v.fract() == 0.0 && v.abs() < 1e15 => format!("{}", v as i64),
            None => fmt_num(v, default_fmt),
        },
        None => e.macros.get(&s.name).cloned().unwrap_or_default(),
    }
}

fn build_grid(ds: &DataSet, models: &[(String, Results)], spec: &Spec) -> Grid {
    let mut g = Grid { title: spec.title.clone(), notes: spec.notes.clone(), ..Default::default() };
    let width = spec.cells.iter().map(|r| r.len()).max().unwrap_or(0);
    // coefficient names of each model
    let model_keys: Vec<Vec<String>> = models
        .iter()
        .map(|(_, e)| {
            let src = spec.cells.iter().flatten().find_map(|el| e.matrices.get(&el.stat)).or_else(|| e.matrices.get("b"));
            let keys: Vec<String> = src.map(|m| m.colnames.clone()).unwrap_or_default();
            keys.into_iter()
                .filter(|k| !is_omitted(k))
                .filter(|k| {
                    let bare = k.rsplit(':').next().unwrap_or(k);
                    let m = |p: &String| glob(p, k) || glob(p, bare);
                    (spec.keep.is_empty() || spec.keep.iter().any(m)) && !spec.drop.iter().any(m)
                })
                .collect()
        })
        .collect();
    let eq_of = |k: &str| k.rsplit_once(':').map(|(e, _)| e.to_string()).unwrap_or_default();
    let bare_of = |k: &str| k.rsplit(':').next().unwrap_or(k).to_string();
    let mut keys: Vec<String> = Vec::new();
    let mut cols: Vec<Col> = Vec::new();
    for (m, mk) in model_keys.iter().enumerate() {
        let first = cols.len();
        if spec.unstack {
            let mut eqs: Vec<String> = Vec::new();
            for k in mk {
                let eq = eq_of(k);
                if !eqs.contains(&eq) {
                    eqs.push(eq);
                }
                let b = bare_of(k);
                if !keys.contains(&b) {
                    keys.push(b);
                }
            }
            for eq in eqs {
                for sub in 0..width.max(1) {
                    cols.push(Col { m, eq: Some(eq.clone()), sub });
                }
            }
        } else {
            for k in mk {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
            for sub in 0..width.max(1) {
                cols.push(Col { m, eq: None, sub });
            }
        }
        if cols.len() > first {
            g.spans.push((first, cols.len() - first));
        }
    }
    // group by equation, then apply order()
    if !spec.unstack {
        let mut eqs: Vec<String> = Vec::new();
        for k in &keys {
            let e = eq_of(k);
            if !eqs.contains(&e) {
                eqs.push(e);
            }
        }
        keys.sort_by_key(|k| eqs.iter().position(|e| *e == eq_of(k)).unwrap_or(0));
    }
    if !spec.order.is_empty() {
        let mut ordered = Vec::new();
        for p in &spec.order {
            for k in &keys {
                if (glob(p, k) || glob(p, &bare_of(k))) && !ordered.contains(k) {
                    ordered.push(k.clone());
                }
            }
        }
        let rest: Vec<String> = keys.iter().filter(|k| !ordered.contains(k)).cloned().collect();
        ordered.extend(rest);
        keys = ordered;
    }
    g.ncols = cols.len();
    let blank = || vec![(String::new(), String::new()); cols.len()];
    // header
    let first_cols: Vec<usize> = (0..models.len()).map(|m| cols.iter().position(|c| c.m == m).unwrap_or(usize::MAX)).collect();
    let model_row = |texts: &[String]| -> Vec<Cell> {
        let mut row = blank();
        for (m, t) in texts.iter().enumerate() {
            if let Some(c) = row.get_mut(first_cols[m]) {
                c.0 = t.clone();
            }
        }
        row
    };
    if spec.numbers {
        g.head.push((String::new(), model_row(&(1..=models.len()).map(|i| format!("({i})")).collect::<Vec<_>>())));
    }
    if spec.titles.iter().any(|t| !t.is_empty()) {
        g.head.push((String::new(), model_row(&spec.titles)));
    }
    if spec.unstack {
        let mut row = blank();
        let mut last: Option<(usize, String)> = None;
        for (j, c) in cols.iter().enumerate() {
            let eq = c.eq.clone().unwrap_or_default();
            if last.as_ref() != Some(&(c.m, eq.clone())) {
                let lab = eq_label(&models[c.m].1, &eq, spec);
                row[j].0 = lab;
                last = Some((c.m, eq));
            }
        }
        if row.iter().any(|c| !c.0.is_empty()) {
            g.head.push((String::new(), row));
        }
    }
    let show_collabels = match &spec.collabels {
        Some(l) => !l.is_empty(),
        None => width > 1,
    };
    if show_collabels {
        let mut row = blank();
        for (j, c) in cols.iter().enumerate() {
            let el = spec.cells.iter().find_map(|r| r.get(c.sub));
            row[j].0 = match &spec.collabels {
                Some(l) if !l.is_empty() => l.get(c.sub).cloned().unwrap_or_default(),
                _ => el.map(|e| e.label.clone().unwrap_or_else(|| e.stat.clone())).unwrap_or_default(),
            };
        }
        g.head.push((String::new(), row));
    }
    // body
    let auto_labels: Vec<(String, String)> = if spec.label {
        models.iter().filter_map(|(_, e)| e.macros.get("labels")).flat_map(|l| words(l).chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect::<Vec<_>>()).collect()
    } else {
        Vec::new()
    };
    let distinct_eqs = {
        let mut v: Vec<String> = keys.iter().map(|k| eq_of(k)).collect();
        v.dedup();
        v
    };
    let show_eqs = !spec.unstack && distinct_eqs.len() > 1;
    let mut last_eq: Option<String> = None;
    for key in &keys {
        if show_eqs {
            let eq = eq_of(key);
            if last_eq.as_ref() != Some(&eq) {
                let lab = models.iter().find(|(_, e)| e.matrices.contains_key("b") || !e.matrices.is_empty()).map(|(_, e)| eq_label(e, &eq, spec)).unwrap_or(eq.clone());
                g.body.push((lab, blank()));
                last_eq = Some(eq);
            }
        }
        let label = coef_label(ds, spec, key, &auto_labels);
        let ri = keys.iter().position(|k| k == key).unwrap_or(0);
        for (r, cells_row) in spec.cells.iter().enumerate() {
            let mut row = blank();
            for (j, c) in cols.iter().enumerate() {
                let Some(el) = cells_row.get(c.sub) else { continue };
                let e = &models[c.m].1;
                let full = match &c.eq {
                    Some(eq) if !eq.is_empty() => format!("{eq}:{key}"),
                    _ => key.clone(),
                };
                let f = el.fmt.get(ri).or(el.fmt.last()).cloned().unwrap_or_else(|| "%9.0g".into());
                let text = if el.stat == "ci" {
                    match (coef_value(e, "ci_l", &full, spec.level), coef_value(e, "ci_u", &full, spec.level)) {
                        (Some(l), Some(u)) => Some(format!("{},{}", fmt_num(l, &f), fmt_num(u, &f))),
                        _ => None,
                    }
                } else {
                    coef_value(e, &el.stat, &full, spec.level).map(|v| fmt_num(v, &f))
                };
                let Some(text) = text else { continue };
                let text = match &el.par {
                    Some((l, rr)) => format!("{l}{text}{rr}"),
                    None => text,
                };
                let st = if el.star { stars_for(coef_value(e, "p", &full, spec.level), &spec.stars) } else { String::new() };
                row[j] = (text, st);
            }
            if r > 0 && row.iter().all(|c| c.0.is_empty()) {
                continue;
            }
            g.body.push((if r == 0 { label.clone() } else { String::new() }, row));
        }
        if spec.gaps {
            g.body.push((String::new(), blank()));
        }
    }
    if spec.gaps && g.body.last().map_or(false, |r| r.0.is_empty() && r.1.iter().all(|c| c.0.is_empty())) {
        g.body.pop();
    }
    // statistics
    for s in &spec.stats {
        let texts: Vec<String> = models.iter().map(|(_, e)| stat_text(e, s, spec.stat_default_fmt)).collect();
        if texts.iter().all(|t| t.is_empty()) {
            continue;
        }
        g.foot.push((s.label.clone(), model_row(&texts)));
    }
    g
}

fn eq_label(e: &Results, eq: &str, spec: &Spec) -> String {
    if let Some((_, l)) = spec.eqlabels.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).find(|(n, _)| n == eq) {
        return l;
    }
    if spec.label {
        if let Some(l) = e.macros.get("eqlabels") {
            let w = words(l);
            if let Some(p) = w.chunks(2).find(|c| c.len() == 2 && c[0] == eq) {
                return p[1].clone();
            }
        }
    }
    eq.to_string()
}

// ---- renderers --------------------------------------------------------------

fn pad_left(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(w - n))
    }
}

fn pad_right(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(w - n))
    }
}

fn render_fixed(g: &Grid, spec: &Spec) -> String {
    let all = || g.head.iter().chain(&g.body).chain(&g.foot);
    let lw = all().map(|r| r.0.chars().count() + 1).max().unwrap_or(0).max(if spec.label { 20 } else { 12 });
    let minw = if spec.compress { 8 } else { 12 };
    let sw = all().flat_map(|r| r.1.iter().map(|c| c.1.chars().count())).max().unwrap_or(0);
    let nw: Vec<usize> = (0..g.ncols).map(|j| all().map(|r| r.1[j].0.chars().count()).max().unwrap_or(0).max(minw)).collect();
    let total = lw + nw.iter().map(|w| w + 1 + sw).sum::<usize>();
    let rule = "-".repeat(total);
    let line = |(label, cells): &(String, Vec<Cell>)| -> String {
        let mut s = pad_right(label, lw);
        for (j, c) in cells.iter().enumerate() {
            s.push(' ');
            s.push_str(&pad_left(&c.0, nw[j]));
            s.push_str(&pad_right(&c.1, sw));
        }
        s.trim_end().to_string()
    };
    let mut out = Vec::new();
    if !g.title.is_empty() {
        out.push(g.title.clone());
    }
    if spec.lines {
        out.push(rule.clone());
    }
    out.extend(g.head.iter().map(line));
    if spec.lines && !g.head.is_empty() {
        out.push(rule.clone());
    }
    out.extend(g.body.iter().map(line));
    if !g.foot.is_empty() {
        if spec.lines {
            out.push(rule.clone());
        }
        out.extend(g.foot.iter().map(line));
    }
    if spec.lines {
        out.push(rule);
    }
    out.extend(g.notes.iter().cloned());
    out.join("\n")
}

fn render_tab(g: &Grid) -> String {
    let mut out = Vec::new();
    if !g.title.is_empty() {
        out.push(g.title.clone());
    }
    for (label, cells) in g.head.iter().chain(&g.body).chain(&g.foot) {
        let mut s = label.clone();
        for c in cells {
            s.push('\t');
            s.push_str(&c.0);
            s.push_str(&c.1);
        }
        out.push(s);
    }
    out.extend(g.notes.iter().cloned());
    out.join("\n") + "\n"
}

fn render_csv(g: &Grid, plain: bool, sep: char) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
    let mut out = Vec::new();
    if !g.title.is_empty() {
        out.push(q(&g.title));
    }
    for (label, cells) in g.head.iter().chain(&g.body).chain(&g.foot) {
        let mut s = q(label);
        for c in cells {
            s.push(sep);
            let t = format!("{}{}", c.0, c.1);
            if plain || t.is_empty() {
                s.push_str(&q(&t));
            } else {
                s.push('=');
                s.push_str(&q(&t));
            }
        }
        out.push(s);
    }
    out.extend(g.notes.iter().map(|n| q(n)));
    out.join("\n") + "\n"
}

fn rtf_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '\\' | '{' | '}' => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 128 => o.push(c),
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    o.push_str(&format!("\\u{}?", *u as i16));
                }
            }
        }
    }
    o
}

const RTF_PROLOG: &str = "{\\rtf1\\ansi\\deff0 {\\fonttbl{\\f0\\froman Times New Roman;}}\n\\deflang1033\\plain\\fs24\n";

/// Body of an RTF document holding the table (without prolog and final brace).
fn render_rtf_body(g: &Grid) -> String {
    let lw = 2400;
    let cw = 1300;
    let mut out = String::new();
    if !g.title.is_empty() {
        out.push_str(&format!("{{\\pard\\keepn\\ql {}\\par}}\n", rtf_escape(&g.title)));
    }
    let rows: Vec<(&(String, Vec<Cell>), &str)> = g.head.iter().map(|r| (r, "h")).chain(g.body.iter().map(|r| (r, "b"))).chain(g.foot.iter().map(|r| (r, "f"))).collect();
    let nrows = rows.len();
    for (i, ((label, cells), part)) in rows.iter().enumerate() {
        let top = i == 0 || (*part == "b" && i == g.head.len()) || (*part == "f" && i == g.head.len() + g.body.len());
        let bottom = i + 1 == nrows;
        let border = |s: &mut String| {
            if top {
                s.push_str("\\clbrdrt\\brdrw10\\brdrs");
            }
            if bottom {
                s.push_str("\\clbrdrb\\brdrw10\\brdrs");
            }
        };
        let mut def = String::from("\\trowd\\trgaph60\\trleft0\\trqc");
        border(&mut def);
        def.push_str(&format!("\\cellx{lw}"));
        for j in 0..cells.len() {
            border(&mut def);
            def.push_str(&format!("\\cellx{}", lw + cw * (j + 1)));
        }
        out.push_str(&def);
        out.push('\n');
        out.push_str(&format!("\\pard\\intbl\\ql {{{}}}\\cell", rtf_escape(label)));
        for c in cells.iter() {
            let stars = if c.1.is_empty() { String::new() } else { format!("{{\\super {}}}", rtf_escape(&c.1)) };
            out.push_str(&format!("\\pard\\intbl\\qc {{{}{}}}\\cell", rtf_escape(&c.0), stars));
        }
        out.push_str("\\row\n");
    }
    for n in &g.notes {
        out.push_str(&format!("{{\\pard\\ql\\fs20 {}\\par}}\n", rtf_escape(n)));
    }
    out
}

fn tex_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '_' | '%' | '&' | '#' | '$' | '{' | '}' => {
                o.push('\\');
                o.push(c);
            }
            c => o.push(c),
        }
    }
    o
}

fn render_tex(g: &Grid, booktabs: bool, fragment: bool) -> String {
    let (top, mid, bot) = if booktabs { ("\\toprule", "\\midrule", "\\bottomrule") } else { ("\\hline\\hline", "\\hline", "\\hline\\hline") };
    let row = |(label, cells): &(String, Vec<Cell>)| -> String {
        let mut s = tex_escape(label);
        for c in cells {
            s.push_str(" & ");
            s.push_str(&tex_escape(&c.0));
            if !c.1.is_empty() {
                s.push_str(&format!("\\sym{{{}}}", c.1));
            }
        }
        s + " \\\\"
    };
    let mut out = Vec::new();
    if !fragment {
        out.push("{".to_string());
        out.push("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}".to_string());
        if !g.title.is_empty() {
            out.push(format!("\\begin{{table}}[htbp]\\centering\n\\caption{{{}}}", tex_escape(&g.title)));
        }
        out.push(format!("\\begin{{tabular}}{{l*{{{}}}{{c}}}}", g.ncols));
        out.push(top.to_string());
    }
    out.extend(g.head.iter().map(row));
    if !g.head.is_empty() {
        out.push(mid.to_string());
    }
    out.extend(g.body.iter().map(row));
    if !g.foot.is_empty() {
        out.push(mid.to_string());
        out.extend(g.foot.iter().map(row));
    }
    if !fragment {
        out.push(bot.to_string());
        for n in &g.notes {
            out.push(format!("\\multicolumn{{{}}}{{l}}{{\\footnotesize {}}}\\\\", g.ncols + 1, tex_escape(n)));
        }
        out.push("\\end{tabular}".to_string());
        if !g.title.is_empty() {
            out.push("\\end{table}".to_string());
        }
        out.push("}".to_string());
    }
    out.join("\n") + "\n"
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn render_html(g: &Grid) -> String {
    let n = g.ncols + 1;
    let mut out = vec!["<table border=\"0\" width=\"*\">".to_string()];
    if !g.title.is_empty() {
        out.push(format!("<caption>{}</caption>", html_escape(&g.title)));
    }
    let rule = format!("<tr><td colspan={n}><hr></td></tr>");
    let row = |(label, cells): &(String, Vec<Cell>)| -> String {
        let mut s = format!("<tr><td>{}</td>", html_escape(label));
        for c in cells {
            let st = if c.1.is_empty() { String::new() } else { format!("<sup>{}</sup>", html_escape(&c.1)) };
            s.push_str(&format!("<td align=\"center\">{}{}</td>", html_escape(&c.0), st));
        }
        s + "</tr>"
    };
    out.push(rule.clone());
    out.extend(g.head.iter().map(row));
    out.push(rule.clone());
    out.extend(g.body.iter().map(row));
    if !g.foot.is_empty() {
        out.push(rule.clone());
        out.extend(g.foot.iter().map(row));
    }
    out.push(rule);
    for note in &g.notes {
        out.push(format!("<tr><td colspan={n}>{}</td></tr>", html_escape(note)));
    }
    out.push("</table>".to_string());
    out.join("\n") + "\n"
}

// ---- command ---------------------------------------------------------------

/// Models named on the command line (wildcards allowed); the `eststo`
/// collection or the active results when none are given.
fn select_models(me: &AdoInterpreter, names: &[String]) -> Result<Vec<(String, Results)>, String> {
    let st = &me.engine.store;
    let mut out = Vec::new();
    if names.is_empty() {
        for n in eststo_names(me) {
            if let Some(e) = st.estimates.get(&n) {
                out.push((n, e.clone()));
            }
        }
        if out.is_empty() {
            if st.e.is_empty() {
                return Err(rc_err(301, "last estimates not found"));
            }
            out.push((".".into(), st.e.clone()));
        }
        return Ok(out);
    }
    let mut all = eststo_names(me);
    for k in st.estimates.keys() {
        if !all.contains(k) {
            all.push(k.clone());
        }
    }
    for n in names {
        if n == "." {
            if st.e.is_empty() {
                return Err(rc_err(301, "last estimates not found"));
            }
            out.push((".".into(), st.e.clone()));
        } else if n.contains('*') || n.contains('?') {
            for k in all.iter().filter(|k| glob(n, k)) {
                out.push((k.clone(), st.estimates[k].clone()));
            }
        } else {
            let e = st.estimates.get(n).ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?;
            out.push((n.clone(), e.clone()));
        }
    }
    if out.is_empty() {
        return Err(rc_err(111, "no estimates found"));
    }
    Ok(out)
}

/// `esttab` (`tab == true`) and `estout`.
pub fn table_cmd(me: &mut AdoInterpreter, r: &str, esttab: bool) -> Result<String, String> {
    let (main, opts) = split_main(r);
    let (names_text, path) = match main.find(" using ").map(|i| i + 1).or_else(|| main.starts_with("using ").then_some(0)) {
        Some(i) => (main[..i].trim().to_string(), Some(unquote(main[i + 5..].trim()))),
        None => (main.clone(), None),
    };
    let names: Vec<String> = names_text.split_whitespace().map(String::from).collect();
    let models = select_models(me, &names)?;
    let ds = &me.engine.dataset;
    let label = has(&opts, "label", 3);
    let plain = has(&opts, "plain", 5);
    let level = arg(&opts, "level", 3).and_then(|l| l.trim().parse().ok()).unwrap_or(me.engine.store.level);
    // stars
    let mut stars: Vec<(String, f64)> = vec![("*".into(), 0.05), ("**".into(), 0.01), ("***".into(), 0.001)];
    if let Some(s) = arg(&opts, "star", 4) {
        let w = words(&s);
        let parsed: Vec<(String, f64)> = w.chunks(2).filter(|c| c.len() == 2).filter_map(|c| c[1].parse().ok().map(|t| (c[0].clone(), t))).collect();
        if !parsed.is_empty() {
            stars = parsed;
        }
    }
    let user_cells = arg(&opts, "cells", 4);
    let mut notes = Vec::new();
    let cells = match &user_cells {
        Some(c) => parse_cells(c),
        None if esttab => {
            let aux = ["se", "t", "z", "p", "ci"].into_iter().find(|a| has(&opts, a, a.len())).unwrap_or("t");
            let show_aux = !has(&opts, "not", 3) && !(has(&opts, "nose", 4) && aux == "se");
            let star_on = !has(&opts, "nostar", 5) && !plain || has(&opts, "star", 4);
            let bfmt = arg(&opts, "b", 1).map(|f| f.trim().to_string()).unwrap_or_else(|| if plain { "%9.0g".into() } else { "a3".into() });
            let auxfmt = arg(&opts, aux, aux.len()).map(|f| f.trim().to_string()).unwrap_or_else(|| match aux {
                "t" | "z" => "2".into(),
                "p" => "3".into(),
                _ => bfmt.clone(),
            });
            let b = Elem { stat: "b".into(), fmt: vec![bfmt], star: star_on, ..Default::default() };
            let a = Elem {
                stat: aux.into(),
                fmt: vec![auxfmt],
                par: if plain { None } else if aux == "ci" { Some(("[".into(), "]".into())) } else { Some(("(".into(), ")".into())) },
                ..Default::default()
            };
            if show_aux {
                notes.push(match aux {
                    "se" => "Standard errors in parentheses".to_string(),
                    "p" => "p-values in parentheses".to_string(),
                    "z" => "z statistics in parentheses".to_string(),
                    "ci" => format!("{}% confidence intervals in brackets", fmt_g(level, 9).trim()),
                    _ => "t statistics in parentheses".to_string(),
                });
            }
            if has(&opts, "wide", 4) && show_aux {
                vec![vec![b, a]]
            } else if show_aux {
                vec![vec![b], vec![a]]
            } else {
                vec![vec![b]]
            }
        }
        None => vec![vec![Elem { stat: "b".into(), ..Default::default() }]],
    };
    if cells.iter().flatten().any(|e| e.star) {
        let mut sorted = stars.clone();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        notes.push(sorted.iter().map(|(s, t)| format!("{s} p<{t}")).collect::<Vec<_>>().join(", "));
    }
    if has(&opts, "nonotes", 6) || !esttab {
        notes.clear();
    }
    if let Some(n) = arg(&opts, "addnotes", 4) {
        notes.extend(words(&n));
    }
    // statistics
    let mut stats: Vec<StatRow> = Vec::new();
    let stat_label = |n: &str| -> String {
        match (n, label) {
            ("N", true) => "Observations".into(),
            ("r2", true) => "R-squared".into(),
            ("r2_a", true) => "Adjusted R-squared".into(),
            ("r2_p", true) => "Pseudo R-squared".into(),
            ("r2", false) if esttab => "R-sq".into(),
            ("r2_a", false) if esttab => "adj. R-sq".into(),
            ("r2_p", false) if esttab => "pseudo R-sq".into(),
            ("aic", _) => "AIC".into(),
            ("bic", _) => "BIC".into(),
            _ => n.to_string(),
        }
    };
    if let Some(s) = arg(&opts, "stats", 4).or_else(|| arg(&opts, "scalars", 3)) {
        let (list, sub) = split_main(&s);
        let fmts = sub.iter().find(|(o, _)| o.starts_with("f")).and_then(|(_, a)| a.clone()).map(|a| words(&a)).unwrap_or_default();
        let labs = sub.iter().find(|(o, _)| o.starts_with("l")).and_then(|(_, a)| a.clone()).map(|a| words(&a)).unwrap_or_default();
        for (k, n) in words(&list).iter().enumerate() {
            stats.push(StatRow { name: n.clone(), label: labs.get(k).cloned().unwrap_or_else(|| stat_label(n)), fmt: fmts.get(k).or(fmts.last()).cloned() });
        }
    } else if esttab {
        if !has(&opts, "noobs", 5) {
            stats.push(StatRow { name: "N".into(), label: stat_label("N"), fmt: None });
        }
        for (o, n, d) in [("r2", "r2", "3"), ("ar2", "r2_a", "3"), ("pr2", "r2_p", "3"), ("aic", "aic", "1"), ("bic", "bic", "1")] {
            if let Some(a) = opt(&opts, o, o.len()) {
                let f = a.clone().map(|x| x.trim().to_string()).unwrap_or_else(|| d.to_string());
                stats.push(StatRow { name: n.into(), label: stat_label(n), fmt: Some(f) });
            }
        }
        if has(&opts, "obslast", 4) && stats.first().map_or(false, |s| s.name == "N") {
            let n = stats.remove(0);
            stats.push(n);
        }
    }
    // model titles
    let titles: Vec<String> = if has(&opts, "nomtitles", 5) || arg(&opts, "mlabels", 4).map_or(false, |a| a.trim() == "none") {
        Vec::new()
    } else if let Some(t) = arg(&opts, "mtitles", 3).or_else(|| arg(&opts, "mlabels", 4)) {
        words(&t)
    } else if has(&opts, "mtitles", 3) || !esttab {
        models.iter().map(|(n, e)| e.macros.get("_estimates_title").cloned().unwrap_or_else(|| n.clone())).collect()
    } else if has(&opts, "nodepvars", 5) {
        Vec::new()
    } else {
        models
            .iter()
            .map(|(n, e)| match e.macros.get("depvar") {
                Some(d) if label => ds.get_var(d).filter(|v| !v.label.is_empty()).map(|v| v.label.clone()).unwrap_or_else(|| d.clone()),
                Some(d) => d.clone(),
                None if n != "." => n.clone(),
                None => String::new(),
            })
            .collect()
    };
    let list = |n: &str, min: usize| arg(&opts, n, min).map(|a| words(&a)).unwrap_or_default();
    let pairs = |n: &str, min: usize| -> Vec<(String, String)> { list(n, min).chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect() };
    let mut drop = list("drop", 4);
    if has(&opts, "noconstant", 6) {
        drop.push("_cons".into());
    }
    let mut coeflabels = pairs("coeflabels", 5);
    coeflabels.extend(pairs("varlabels", 4));
    let spec = Spec {
        cells,
        stats,
        stars,
        numbers: if esttab { !has(&opts, "nonumbers", 5) } else { has(&opts, "numbers", 3) },
        titles,
        // the default esttab cells (also with `wide`) carry no column labels
        collabels: arg(&opts, "collabels", 4).map(|a| if a.trim() == "none" { Vec::new() } else { words(&a) }).or_else(|| (esttab && user_cells.is_none()).then(Vec::new)),
        label,
        keep: list("keep", 4),
        drop,
        order: list("order", 3),
        coeflabels,
        eqlabels: list("eqlabels", 3),
        title: arg(&opts, "title", 2).map(|t| unquote(&t)).unwrap_or_default(),
        notes,
        gaps: has(&opts, "gaps", 4),
        unstack: has(&opts, "unstack", 3),
        level,
        compress: has(&opts, "compress", 4),
        lines: esttab && !has(&opts, "nolines", 6),
        stat_default_fmt: if esttab { "3" } else { "%9.0g" },
    };
    let mut spec = spec;
    if !esttab && spec.collabels.is_none() {
        // estout shows the statistic names above the columns
        let width = spec.cells.iter().map(|r| r.len()).max().unwrap_or(0);
        if width > 0 {
            spec.collabels = Some((0..width).map(|k| spec.cells.iter().find_map(|r| r.get(k)).map(|e| e.label.clone().unwrap_or_else(|| e.stat.clone())).unwrap_or_default()).collect());
        }
    }
    let grid = build_grid(&me.engine.dataset, &models, &spec);
    // output format
    let mode = ["rtf", "csv", "scsv", "tex", "html", "fixed", "tab", "smcl", "md"].into_iter().find(|m| has(&opts, m, m.len())).map(String::from).or_else(|| arg(&opts, "style", 3).map(|s| s.trim().to_lowercase()));
    let Some(path) = path else {
        let shown = match mode.as_deref() {
            Some("tab") => render_tab(&grid),
            Some("tex") => render_tex(&grid, has(&opts, "booktabs", 4), has(&opts, "fragment", 4)),
            Some("html") => render_html(&grid),
            Some("csv") => render_csv(&grid, plain, ','),
            Some("scsv") => render_csv(&grid, plain, ';'),
            _ if !esttab && mode.is_none() => render_tab(&grid),
            _ => render_fixed(&grid, &spec),
        };
        return Ok(format!("\n{}", shown.trim_end()));
    };
    let ext = std::path::Path::new(&path).extension().map(|e| e.to_string_lossy().to_lowercase());
    let mode = mode.unwrap_or_else(|| match ext.as_deref() {
        Some("rtf") | Some("doc") => "rtf".into(),
        Some("csv") => "csv".into(),
        Some("tex") => "tex".into(),
        Some("html") | Some("htm") => "html".into(),
        Some("smcl") => "smcl".into(),
        _ if esttab => "fixed".into(),
        _ => "tab".into(),
    });
    let path = if ext.is_some() {
        path
    } else {
        let suf = match mode.as_str() {
            "rtf" => "rtf",
            "csv" | "scsv" => "csv",
            "tex" => "tex",
            "html" => "html",
            "smcl" => "smcl",
            _ => "txt",
        };
        format!("{path}.{suf}")
    };
    let append = has(&opts, "append", 1);
    let exists = vfs::exists(&path);
    if exists && !append && !has(&opts, "replace", 1) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let old = if append && exists { vfs::read_to_string(&path).unwrap_or_default() } else { String::new() };
    let content = match mode.as_str() {
        "rtf" => {
            let body = render_rtf_body(&grid);
            let t = old.trim_end();
            if t.starts_with("{\\rtf") && t.ends_with('}') {
                format!("{}{{\\pard\\par}}\n{body}}}\n", &t[..t.len() - 1])
            } else {
                format!("{old}{RTF_PROLOG}{body}}}\n")
            }
        }
        "csv" => old + &render_csv(&grid, plain, ','),
        "scsv" => old + &render_csv(&grid, plain, ';'),
        "tex" => old + &render_tex(&grid, has(&opts, "booktabs", 4), has(&opts, "fragment", 4)),
        "html" => old + &render_html(&grid),
        "tab" => old + &render_tab(&grid),
        _ => old + &render_fixed(&grid, &spec) + "\n",
    };
    vfs::write(&path, content.as_bytes()).map_err(|e| rc_err(603, format!("file {path} could not be opened: {e}")))?;
    Ok(format!("(output written to {path})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_format() {
        assert_eq!(fmt_num(1.7469, "a3"), "1.747");
        assert_eq!(fmt_num(-49.512, "a3"), "-49.51");
        assert_eq!(fmt_num(3673.06, "a3"), "3673.1");
        assert_eq!(fmt_num(0.01193, "a3"), "0.0119");
        assert_eq!(fmt_num(2.5, "2"), "2.50");
    }

    #[test]
    fn cells_spec() {
        let c = parse_cells("mean(fmt(2)) sd(fmt(2) par)");
        assert_eq!(c.len(), 2);
        assert_eq!(c[1][0].par, Some(("(".to_string(), ")".to_string())));
        let c = parse_cells("\"N_1 mu_1(fmt(2)) b(fmt(2) star) p(fmt(4))\"");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 4);
        assert!(c[0][2].star);
        assert_eq!(c[0][3].fmt, vec!["4".to_string()]);
    }

    #[test]
    fn wildcards() {
        assert!(glob("est*", "est12"));
        assert!(!glob("a?", "abc"));
    }
}
