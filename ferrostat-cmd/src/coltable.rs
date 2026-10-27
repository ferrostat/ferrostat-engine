//! `table` (current syntax): computes the statistics of every cell, puts
//! them in a collection (default name `Table`) and shows its layout.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, split_options};
use ferrostat_parser::Command;

use crate::collect::{CellOpts, Cond, Layout, Style, Term};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::table::{canonical_stat, eval_stat, split_specs, stat_title, Stat};
use crate::util::{self, WeightKind, Weights};
use crate::varlist;

/// A requested statistic.
#[derive(Clone, Debug)]
struct TStat {
    name: String,
    var: Option<String>,
    across: Option<String>,
    fv: bool,
}

fn fv_name(s: &str) -> Option<&'static str> {
    Some(match s.to_lowercase().as_str() {
        "fvfrequency" | "fvfreq" | "fvfrequ" | "fvfrequen" => "fvfrequency",
        "fvpercent" | "fvperc" | "fvper" => "fvpercent",
        "fvproportion" | "fvprop" | "fvproport" => "fvproportion",
        "fvrawfrequency" | "fvrawfreq" => "fvrawfrequency",
        "fvrawpercent" | "fvrawperc" => "fvrawpercent",
        "fvrawproportion" | "fvrawprop" => "fvrawproportion",
        _ => return None,
    })
}

fn stat_name(s: &str) -> Result<String, String> {
    if let Some(f) = fv_name(s) {
        return Ok(f.into());
    }
    match s.to_lowercase().as_str() {
        "prop" | "proportion" | "propor" | "proport" => return Ok("proportion".into()),
        "rawfrequency" | "rawfreq" => return Ok("rawfrequency".into()),
        "rawpercent" | "rawperc" => return Ok("rawpercent".into()),
        "rawproportion" | "rawprop" => return Ok("rawproportion".into()),
        "total" => return Ok("total".into()),
        "rawtotal" => return Ok("rawtotal".into()),
        _ => {}
    }
    canonical_stat(s)
}

/// Label of a statistic as a level of the result dimension.
pub fn result_label(s: &str) -> String {
    match s {
        "fvfrequency" => "Factor-variable frequency".into(),
        "fvpercent" => "Factor-variable percent".into(),
        "fvproportion" => "Factor-variable proportion".into(),
        "fvrawfrequency" => "Factor-variable raw frequency".into(),
        "fvrawpercent" => "Factor-variable raw percent".into(),
        "fvrawproportion" => "Factor-variable raw proportion".into(),
        "proportion" => "Proportion".into(),
        "rawfrequency" => "Raw frequency".into(),
        "rawpercent" => "Raw percent".into(),
        "rawproportion" => "Raw proportion".into(),
        "total" => "Total".into(),
        "rawtotal" => "Raw total".into(),
        "count" => "Number of nonmissing values".into(),
        "semean" => "Standard error of the mean".into(),
        "q1" => "First quartile".into(),
        "q3" => "Third quartile".into(),
        other => stat_title(other),
    }
}

/// Level key of observation `i` of `var`: the value (not its label).
pub fn level_key(ds: &DataSet, var: &str, i: usize) -> String {
    let v = ds.get_var(var).unwrap();
    if v.var_type.is_string() {
        return util::cell_text(ds, v, i, false);
    }
    fmt_g(v.num(i), 12)
}

/// Variable label or name.
pub fn var_label(ds: &DataSet, var: &str) -> String {
    ds.get_var(var).map(|v| if v.label.is_empty() { var.to_string() } else { v.label.clone() }).unwrap_or_else(|| var.to_string())
}

/// Text of a factor level `k` of `var` (value label when there is one).
pub fn value_text(ds: &DataSet, var: &str, k: f64) -> String {
    if let Some(v) = ds.get_var(var) {
        if let Some(l) = ds.value_labels.get(&v.value_label_name).and_then(|vl| vl.lookup(k)) {
            return l.to_string();
        }
    }
    fmt_g(k, 12)
}

/// Sorted distinct nonmissing values of numeric `var` among `rows`.
pub fn fv_levels(ds: &DataSet, var: &str, rows: &[usize]) -> Vec<f64> {
    let v = ds.get_var(var).unwrap();
    let mut l: Vec<f64> = rows.iter().map(|&i| v.num(i)).filter(|x| !is_missing(*x)).collect();
    l.sort_by(|a, b| a.total_cmp(b));
    l.dedup();
    l
}

fn freq(rows: &[usize], w: Option<&Weights>) -> f64 {
    match w {
        Some(w) if matches!(w.kind, WeightKind::F | WeightKind::I) => rows.iter().map(|&i| w.w[i]).sum(),
        _ => rows.len() as f64,
    }
}

/// Parse a statistic(...) option argument.
fn parse_stat(ds: &DataSet, arg: &str) -> Result<Vec<TStat>, String> {
    let (head, opts) = match find_top_level(arg, ',') {
        Some(i) => (&arg[..i], split_options(&arg[i + 1..])),
        None => (arg, Vec::new()),
    };
    let across = opts.iter().find(|(n, _)| n.to_lowercase().starts_with("across")).and_then(|(_, a)| a.clone()).map(|a| varlist::resolve_name(ds, a.trim())).transpose()?;
    let mut w = head.split_whitespace();
    let name = stat_name(w.next().ok_or_else(|| rc_err(198, "statistic() invalid"))?)?;
    let rest: Vec<&str> = w.collect();
    let mut out = Vec::new();
    if name.starts_with("fv") {
        if rest.is_empty() {
            return Err(rc_err(198, format!("statistic {name} requires a varlist")));
        }
        let text: Vec<String> = rest.iter().map(|r| r.strip_prefix("i.").unwrap_or(r).to_string()).collect();
        for v in varlist::expand(ds, &text.join(" "))? {
            out.push(TStat { name: name.clone(), var: Some(v), across: None, fv: true });
        }
    } else if matches!(name.as_str(), "frequency" | "percent" | "proportion" | "rawfrequency" | "rawpercent" | "rawproportion") {
        out.push(TStat { name, var: None, across, fv: false });
    } else {
        if rest.is_empty() {
            return Err(rc_err(198, format!("statistic {name} requires a varlist")));
        }
        for v in varlist::expand(ds, &rest.join(" "))? {
            out.push(TStat { name: name.clone(), var: Some(v), across: None, fv: false });
        }
    }
    Ok(out)
}

fn opt_matches(n: &str, full: &str, min: usize) -> bool {
    let n = n.to_lowercase();
    n.len() >= min && full.starts_with(&n)
}

/// `%fmt [stat ...]` / `"fmt" [stat ...]` of nformat()/sformat().
fn parse_fmt_opt(a: &str) -> (String, Vec<String>) {
    let t = a.trim();
    let (f, rest) = if t.starts_with('"') || t.starts_with('`') {
        let close = if t.starts_with('`') { t.find("\"'").map(|p| p + 2) } else { t[1..].find('"').map(|p| p + 2) };
        let end = close.unwrap_or(t.len());
        (util::unquote(&t[..end]), t[end..].trim().to_string())
    } else {
        // the format is everything up to the first word naming a statistic
        let words: Vec<&str> = t.split_whitespace().collect();
        let k = (1..words.len()).find(|&k| stat_name(words[k]).is_ok()).unwrap_or(words.len());
        (words[..k].join(" "), words[k..].join(" "))
    };
    (f, rest.split_whitespace().map(|s| stat_name(s).unwrap_or_else(|_| s.to_string())).collect())
}

pub fn run(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let specs = split_specs(&cmd.main);
    if specs.len() > 3 {
        return Err(rc_err(198, "too many dimensions specified"));
    }
    // axes: table variables plus the keywords result / var
    let mut axes: Vec<Vec<String>> = Vec::new();
    for s in &specs {
        let mut a = Vec::new();
        for w in s.split_whitespace() {
            match w {
                "result" | "var" | "across" => a.push(w.to_string()),
                _ => a.extend(varlist::expand(ds, w)?),
            }
        }
        axes.push(a);
    }
    axes.resize(3, Vec::new());
    let table_vars: Vec<String> = axes.iter().flatten().filter(|v| !matches!(v.as_str(), "result" | "var" | "across")).cloned().collect();
    // statistics and formats
    let mut stats: Vec<TStat> = Vec::new();
    let mut edits: Vec<(Vec<Cond>, CellOpts)> = Vec::new();
    for (n, a) in &cmd.opts {
        let a = a.clone().unwrap_or_default();
        if opt_matches(n, "statistic", 4) {
            stats.extend(parse_stat(ds, &a)?);
        } else if opt_matches(n, "nformat", 2) || opt_matches(n, "sformat", 2) {
            let (f, on) = parse_fmt_opt(&a);
            let conds = if on.is_empty() { Vec::new() } else { vec![Cond { dim: "result".into(), levels: Some(on) }] };
            let o = if opt_matches(n, "nformat", 2) { CellOpts { nformat: Some(f), ..Default::default() } } else { CellOpts { sformat: Some(f), ..Default::default() } };
            edits.push((conds, o));
        }
    }
    if stats.is_empty() {
        stats.push(TStat { name: "frequency".into(), var: None, across: None, fv: false });
    }
    for s in &stats {
        if let Some(a) = &s.across {
            if !table_vars.contains(a) {
                return Err(rc_err(198, format!("variable {a} in across() must be a table dimension")));
            }
        }
    }
    let missing = cmd.has_opt("missing", 4);
    let nototals = cmd.has_opt("nototals", 5);
    let total_vars: Option<Vec<String>> = cmd.opt("totals", 3).map(|t| varlist::expand(ds, t.split('#').collect::<Vec<_>>().join(" ").as_str())).transpose()?;
    let wants_total = |v: &str| !nototals && total_vars.as_ref().map_or(true, |t| t.iter().any(|x| x == v));
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "faip")?;
    if !missing {
        util::markout(ds, &mut mask, &table_vars);
    }
    let all_rows = util::rows(&mask);
    let grand = freq(&all_rows, w.as_ref());
    let name = cmd.opt("name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| "Table".into());
    let append = cmd.has_opt("append", 6) && st.collect.map.contains_key(&name);
    let style_name = cmd.opt("style", 3).map(|s| s.trim().split(',').next().unwrap_or("").trim().to_string()).filter(|s| !s.is_empty());
    let mut style = Style::named("table").unwrap();
    if let Some(sn) = &style_name {
        style = Style::named(sn).ok_or_else(|| rc_err(111, format!("style {sn} not found")))?;
    }
    style.cell.extend(edits);
    if append {
        st.collect.current = name.clone();
    } else {
        st.collect.fresh(&name);
    }
    let c = st.collect.cur_mut();
    c.style = style;
    if let Some(t) = cmd.opt("title", 5) {
        c.title = Some(util::unquote(t));
    }
    // dimension levels (sorted values) and labels
    let mut obs_level: Vec<Vec<usize>> = Vec::new();
    let mut level_keys: Vec<Vec<String>> = Vec::new();
    for v in &table_vars {
        let (ids, reps) = util::group_ids(ds, &[v.clone()], &mask);
        let keys: Vec<String> = reps.iter().map(|&r| level_key(ds, v, r)).collect();
        let var = ds.get_var(v).unwrap();
        for (k, &r) in keys.iter().zip(&reps) {
            c.set_level_label(v, k, &util::cell_text(ds, var, r, true));
        }
        c.dim_labels.insert(v.clone(), var_label(ds, v));
        c.set_level_label(v, ".m", "Total");
        let mut lv = keys.clone();
        if wants_total(v) {
            lv.push(".m".into());
        }
        c.levels.insert(v.clone(), lv);
        if !c.dims.contains(v) {
            c.dims.push(v.clone());
        }
        obs_level.push(ids);
        level_keys.push(keys);
    }
    // result and var levels in the order requested
    let mut fv_lv: Vec<Vec<f64>> = Vec::new();
    {
        let mut res_order: Vec<String> = Vec::new();
        let mut var_order: Vec<String> = Vec::new();
        for s in &stats {
            if !res_order.contains(&s.name) {
                res_order.push(s.name.clone());
            }
            c.set_level_label("result", &s.name, &result_label(&s.name));
            let lv = if s.fv { fv_levels(ds, s.var.as_ref().unwrap(), &all_rows) } else { Vec::new() };
            if let Some(v) = &s.var {
                if s.fv {
                    for &k in &lv {
                        let key = format!("{}.{}", fmt_g(k, 12), v);
                        c.set_level_label("var", &key, &value_text(ds, v, k));
                        c.set_group("var", &key, &var_label(ds, v));
                        if !var_order.contains(&key) {
                            var_order.push(key);
                        }
                    }
                } else {
                    c.set_level_label("var", v, &var_label(ds, v));
                    if !var_order.contains(v) {
                        var_order.push(v.clone());
                    }
                }
            }
            fv_lv.push(lv);
        }
        c.levels.insert("result".into(), res_order);
        if !var_order.is_empty() {
            c.levels.insert("var".into(), var_order);
        }
    }
    // every cell: levels of each dimension or its total
    let nd = table_vars.len();
    let mut key: Vec<Option<usize>> = vec![None; nd];
    let mut cells: Vec<(Vec<Option<usize>>, Vec<usize>)> = Vec::new();
    fn rec(d: usize, nd: usize, rows: Vec<usize>, key: &mut Vec<Option<usize>>, obs_level: &[Vec<usize>], nlev: &[usize], tot: &[bool], out: &mut Vec<(Vec<Option<usize>>, Vec<usize>)>) {
        if d == nd {
            out.push((key.clone(), rows));
            return;
        }
        for l in 0..nlev[d] {
            let sub: Vec<usize> = rows.iter().copied().filter(|&i| obs_level[d][i] == l).collect();
            if sub.is_empty() {
                continue;
            }
            key[d] = Some(l);
            rec(d + 1, nd, sub, key, obs_level, nlev, tot, out);
        }
        if tot[d] {
            key[d] = None;
            rec(d + 1, nd, rows, key, obs_level, nlev, tot, out);
        }
        key[d] = None;
    }
    let nlev: Vec<usize> = level_keys.iter().map(Vec::len).collect();
    let tot: Vec<bool> = table_vars.iter().map(|v| wants_total(v)).collect();
    rec(0, nd, all_rows.clone(), &mut key, &obs_level, &nlev, &tot, &mut cells);
    let ds_ref: &DataSet = ds;
    let rows_for = |k: &[Option<usize>]| -> Vec<usize> { all_rows.iter().copied().filter(|&i| k.iter().enumerate().all(|(d, l)| l.map_or(true, |l| obs_level[d][i] == l))).collect() };
    for (k, rows) in &cells {
        let base: Vec<(String, String)> = table_vars.iter().zip(k).enumerate().map(|(d, (v, l))| (v.clone(), l.map_or(".m".to_string(), |l| level_keys[d][l].clone()))).collect();
        for (si, s) in stats.iter().enumerate() {
            let mut tags = base.clone();
            tags.push(("result".into(), s.name.clone()));
            match s.name.as_str() {
                "frequency" | "rawfrequency" => {
                    c.add(tags, crate::collect::Val::Num(freq(rows, w.as_ref())));
                }
                "percent" | "proportion" | "rawpercent" | "rawproportion" => {
                    let den = match &s.across {
                        Some(a) => {
                            let d = table_vars.iter().position(|v| v == a).unwrap();
                            let mut kk = k.clone();
                            kk[d] = None;
                            tags.push(("across".into(), a.clone()));
                            freq(&rows_for(&kk), w.as_ref())
                        }
                        None => grand,
                    };
                    let f = freq(rows, w.as_ref());
                    let scale = if s.name.ends_with("percent") { 100.0 } else { 1.0 };
                    c.add(tags, crate::collect::Val::Num(if den > 0.0 { scale * f / den } else { SYSMISS }));
                }
                n if s.fv => {
                    let v = s.var.as_ref().unwrap();
                    let col = ds_ref.get_var(v).unwrap();
                    let nonmiss: Vec<usize> = rows.iter().copied().filter(|&i| !is_missing(col.num(i))).collect();
                    let den = freq(&nonmiss, w.as_ref());
                    for &lv in &fv_lv[si] {
                        let at: Vec<usize> = nonmiss.iter().copied().filter(|&i| col.num(i) == lv).collect();
                        let f = freq(&at, w.as_ref());
                        let val = match n {
                            "fvfrequency" | "fvrawfrequency" => f,
                            "fvpercent" | "fvrawpercent" => if den > 0.0 { 100.0 * f / den } else { SYSMISS },
                            _ => if den > 0.0 { f / den } else { SYSMISS },
                        };
                        let mut t = tags.clone();
                        t.push(("var".into(), format!("{}.{}", fmt_g(lv, 12), v)));
                        c.add(t, crate::collect::Val::Num(val));
                    }
                }
                _ => {
                    let stat = Stat { name: s.name.clone(), var: s.var.clone() };
                    let val = if s.name == "total" {
                        let col = ds_ref.get_var(s.var.as_ref().unwrap()).unwrap();
                        rows.iter().map(|&i| (col.num(i), w.as_ref().map_or(1.0, |w| w.w[i]))).filter(|(x, _)| !is_missing(*x)).map(|(x, ww)| x * ww).sum()
                    } else if s.name == "rawtotal" {
                        eval_stat(ds_ref, &Stat { name: "rawsum".into(), var: s.var.clone() }, rows, None, grand)
                    } else {
                        eval_stat(ds_ref, &stat, rows, w.as_ref(), grand)
                    };
                    tags.push(("var".into(), s.var.clone().unwrap()));
                    c.add(tags, crate::collect::Val::Num(val));
                }
            }
        }
    }
    // default layout
    let distinct_vars = c.levels.get("var").map_or(0, Vec::len);
    let distinct_results = c.levels.get("result").map_or(0, Vec::len);
    let placed = |k: &str| axes.iter().flatten().any(|x| x == k);
    let (mut rowd, mut cold, tabd) = (axes[0].clone(), axes[1].clone(), axes[2].clone());
    let mut auto: Vec<String> = Vec::new();
    if distinct_vars > 1 && !placed("var") {
        auto.push("var".into());
    }
    if (distinct_results > 1 || cold.is_empty()) && !placed("result") {
        auto.push("result".into());
    }
    if cold.is_empty() {
        cold = auto;
    } else {
        rowd.extend(auto);
    }
    let term = |dims: &[String]| -> Vec<Term> { if dims.is_empty() { Vec::new() } else { vec![dims.iter().map(|d| Cond { dim: d.clone(), levels: None }).collect()] } };
    c.layout = Some(Layout { rows: term(&rowd), cols: term(&cold), tabs: term(&tabd) });
    c.preview()
}
