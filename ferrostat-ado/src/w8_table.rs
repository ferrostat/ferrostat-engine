//! `table ..., command()`: tables of results from estimation and testing
//! commands. Each `command()` option runs a command quietly and collects its
//! results into the table's collection under the dimensions `command`
//! (1, 2, ... in option order), `colname` (coefficient names, with `coleq`
//! for multi-equation models) and `result`; the layout is then built from
//! the row/column/table specifications, so `collect style`, `collect
//! layout` and `collect export` work on the result afterwards.
//!
//! Option syntax: `command([name =] [results :] cmdline)`; results are
//! `_r_b _r_se _r_z _r_z_abs _r_p _r_lb _r_ub _r_ci _r_df` and saved
//! results written `e(name)`, `r(name)` or `name`. Without results an
//! estimation command contributes `_r_b _r_ci`, any other command all of
//! its scalars in r().

use ferrostat_cmd::collect::{Cond, Layout, Style, Term, Val};
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::{dist, util};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_parser::parser::{find_top_level, parse_line};

use crate::interpreter::AdoInterpreter;

const COEF_RESULTS: &[&str] = &["_r_b", "_r_se", "_r_z", "_r_z_abs", "_r_p", "_r_lb", "_r_ub", "_r_ci", "_r_df"];

fn is_opt(n: &str, full: &str, min: usize) -> bool {
    let n = n.to_lowercase();
    n.len() >= min && full.starts_with(&n)
}

/// Does this `table` statement use `command()`?
pub fn has_command(rest: &str) -> bool {
    let Some(c) = find_top_level(rest, ',') else { return false };
    let o = rest[c + 1..].to_lowercase();
    let b = o.as_bytes();
    // an option word comm[and]( at the start of an option
    let mut i = 0;
    while let Some(p) = o[i..].find("comm") {
        let s = i + p;
        let prev = if s == 0 { b' ' } else { b[s - 1] };
        if prev == b' ' || prev == b',' || prev == b')' || prev == b'\t' {
            let tail = &o[s..];
            let word: String = tail.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
            if "command".starts_with(&word) && tail[word.len()..].trim_start().starts_with('(') {
                return true;
            }
        }
        i = s + 4;
    }
    false
}

/// Parenthesized specifications of the varlist part: `(a b) (c) ()`.
fn specs(s: &str) -> Result<Vec<Vec<String>>, String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for ch in s.chars() {
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
                    out.push(cur.split_whitespace().map(String::from).collect());
                    cur.clear();
                } else {
                    cur.push(ch);
                }
            }
            c if depth == 0 && c.is_whitespace() => {
                if !cur.trim().is_empty() {
                    out.push(vec![cur.trim().to_string()]);
                }
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(vec![cur.trim().to_string()]);
    }
    if out.len() > 3 {
        return Err(rc_err(198, "too many dimensions specified"));
    }
    Ok(out)
}

/// One `command()` option.
struct Spec {
    name: Option<String>,
    results: Vec<String>,
    line: String,
}

fn is_result_word(w: &str) -> bool {
    let id = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !s.starts_with(|c: char| c.is_ascii_digit());
    if let Some(inner) = w.strip_prefix("e(").or_else(|| w.strip_prefix("r(")).or_else(|| w.strip_prefix("s(")) {
        return inner.strip_suffix(')').is_some_and(id);
    }
    w.starts_with("_r_") && id(w)
}

fn parse_spec(me: &AdoInterpreter, arg: &str) -> Result<Spec, String> {
    let mut a = arg.trim().to_string();
    let mut name = None;
    // name = (...) or name = cmdline
    if let Some(eq) = find_top_level(&a, '=') {
        let lhs = a[..eq].trim();
        if !lhs.is_empty() && lhs.chars().all(|c| c.is_alphanumeric() || c == '_' || c == ' ') && !a[eq + 1..].starts_with('=') {
            name = Some(lhs.to_string());
            a = a[eq + 1..].trim().to_string();
        }
    }
    if a.starts_with('(') && a.ends_with(')') {
        a = a[1..a.len() - 1].trim().to_string();
    }
    let mut results = Vec::new();
    if let Some(c) = find_top_level(&a, ':') {
        let words: Vec<&str> = a[..c].split_whitespace().collect();
        let plain_ok = |w: &str| w.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !me.engine.is_command(w) && !matches!(w, "by" | "bysort" | "quietly" | "noisily" | "capture" | "xi" | "svy" | "mi" | "nestreg" | "stepwise");
        if !words.is_empty() && words.iter().all(|w| is_result_word(w) || plain_ok(w)) {
            results = words.iter().map(|w| w.to_string()).collect();
            a = a[c + 1..].trim().to_string();
        }
    }
    if a.is_empty() {
        return Err(rc_err(198, "command() requires a command"));
    }
    Ok(Spec { name, results, line: a })
}

fn level_of(w: &str) -> String {
    for p in ["e(", "r(", "s("] {
        if let Some(i) = w.strip_prefix(p).and_then(|x| x.strip_suffix(')')) {
            return i.to_string();
        }
    }
    w.to_string()
}

/// `table [(rows)] [(cols)] [(tabs)], command(...) [command(...) ...] options`
pub fn run(me: &mut AdoInterpreter, full: &str) -> Result<String, String> {
    let cmd = parse_line(full).map_err(|e| rc_err(198, e.message))?;
    let axes = specs(&cmd.main)?;
    for w in axes.iter().flatten() {
        if !matches!(w.as_str(), "command" | "result" | "colname" | "coleq") {
            return Err(rc_err(198, format!("{w} not allowed with command(); dimensions are command, colname, coleq and result")));
        }
    }
    let mut specs_v = Vec::new();
    let mut edits: Vec<(Vec<Cond>, ferrostat_cmd::collect::CellOpts)> = Vec::new();
    for (n, a) in &cmd.opts {
        let a = a.clone().unwrap_or_default();
        if is_opt(n, "command", 4) {
            specs_v.push(parse_spec(me, &a)?);
        } else if is_opt(n, "nformat", 2) || is_opt(n, "sformat", 2) {
            let t = a.trim();
            let (f, rest) = if t.starts_with('"') {
                let end = t[1..].find('"').map_or(t.len(), |p| p + 2);
                (util::unquote(&t[..end]), t[end..].trim().to_string())
            } else {
                let mut w = t.split_whitespace();
                (w.next().unwrap_or("").to_string(), w.collect::<Vec<_>>().join(" "))
            };
            let on: Vec<String> = rest.split_whitespace().map(level_of).collect();
            let conds = if on.is_empty() { Vec::new() } else { vec![Cond { dim: "result".into(), levels: Some(on) }] };
            let o = if is_opt(n, "nformat", 2) {
                ferrostat_cmd::collect::CellOpts { nformat: Some(f), ..Default::default() }
            } else {
                ferrostat_cmd::collect::CellOpts { sformat: Some(f), ..Default::default() }
            };
            edits.push((conds, o));
        }
    }
    let level: f64 = match cmd.opt("level", 3) {
        Some(l) => l.trim().parse().map_err(|_| rc_err(198, "invalid level()"))?,
        None => me.engine.store.level,
    };
    let name = cmd.opt("name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| "Table".into());
    let append = cmd.has_opt("append", 6) && me.engine.store.collect.map.contains_key(&name);
    let mut style = Style::named("table").unwrap();
    if let Some(sn) = cmd.opt("style", 3).map(|s| s.trim().split(',').next().unwrap_or("").trim().to_string()).filter(|s| !s.is_empty()) {
        style = Style::named(&sn).ok_or_else(|| rc_err(111, format!("style {sn} not found")))?;
    }
    style.cell.push((vec![Cond { dim: "result".into(), levels: Some(vec!["_r_b".into(), "_r_se".into(), "_r_lb".into(), "_r_ub".into()]) }], ferrostat_cmd::collect::CellOpts { nformat: Some("%9.0g".into()), ..Default::default() }));
    style.cell.extend(edits);

    // run the commands and keep their results
    let crit_for = |df: Option<f64>| match df {
        Some(d) if d > 0.0 && !is_missing(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        _ => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    struct Got {
        items: Vec<(Vec<(String, String)>, Val)>,
        coefs: Vec<(Option<String>, String)>,
    }
    let mut got: Vec<Got> = Vec::new();
    let mut order_results: Vec<String> = Vec::new();
    for sp in &specs_v {
        // markers tell which results the command replaced
        const MARK: &str = "__table_cmd_mark";
        me.engine.store.e.macros.insert(MARK.into(), "1".into());
        me.engine.store.r.macros.insert(MARK.into(), "1".into());
        let res = me.exec_text(&format!("quietly {}", sp.line), None);
        let st = &mut me.engine.store;
        let e_new = st.e.macros.remove(MARK).is_none();
        let r_new = st.r.macros.remove(MARK).is_none();
        res?;
        let (r, e) = (st.r.clone(), st.e.clone());
        let has_b = e_new && e.matrices.contains_key("b");
        let wanted: Vec<String> = if sp.results.is_empty() {
            if has_b {
                vec!["_r_b".into(), "_r_ci".into()]
            } else if r_new {
                let mut k: Vec<String> = r.scalars.keys().cloned().collect();
                k.sort();
                k
            } else {
                let mut k: Vec<String> = e.scalars.keys().cloned().collect();
                k.sort();
                k
            }
        } else {
            sp.results.clone()
        };
        let mut g = Got { items: Vec::new(), coefs: Vec::new() };
        for w in &wanted {
            let lv = level_of(w);
            if !order_results.contains(&lv) {
                order_results.push(lv.clone());
            }
            if COEF_RESULTS.contains(&w.as_str()) {
                let (Some(b), true) = (e.matrices.get("b"), e_new) else {
                    return Err(rc_err(198, format!("result {w} requires an estimation command")));
                };
                let v = e.matrices.get("V");
                let df = e.get("df_r");
                let crit = crit_for(df);
                for (j, cn) in b.colnames.iter().enumerate() {
                    let (eq, coef) = match cn.rsplit_once(':') {
                        Some((q, c)) => (Some(q.to_string()), c.to_string()),
                        None => (None, cn.clone()),
                    };
                    if !g.coefs.contains(&(eq.clone(), coef.clone())) {
                        g.coefs.push((eq.clone(), coef.clone()));
                    }
                    let bv = b.get(0, j);
                    let se = v.filter(|m| m.rows > j).map(|m| m.get(j, j).max(0.0).sqrt()).unwrap_or(SYSMISS);
                    let z = if se > 0.0 && !is_missing(se) { bv / se } else { SYSMISS };
                    let p = if is_missing(z) {
                        SYSMISS
                    } else {
                        match df {
                            Some(d) if d > 0.0 && !is_missing(d) => 2.0 * dist::ttail(d, z.abs()),
                            _ => 2.0 * dist::normal(-z.abs()),
                        }
                    };
                    let (lb, ub) = if is_missing(se) || se == 0.0 { (SYSMISS, SYSMISS) } else { (bv - crit * se, bv + crit * se) };
                    let mut base = Vec::new();
                    if let Some(q) = &eq {
                        base.push(("coleq".to_string(), q.clone()));
                    }
                    base.push(("colname".to_string(), coef.clone()));
                    let mut put = |r: &str, x: f64| {
                        let mut t = base.clone();
                        t.push(("result".into(), r.to_string()));
                        g.items.push((t, Val::Num(x)));
                    };
                    match w.as_str() {
                        "_r_b" => put("_r_b", bv),
                        "_r_se" => put("_r_se", se),
                        "_r_z" => put("_r_z", z),
                        "_r_z_abs" => put("_r_z_abs", if is_missing(z) { z } else { z.abs() }),
                        "_r_p" => put("_r_p", p),
                        "_r_lb" => put("_r_lb", lb),
                        "_r_ub" => put("_r_ub", ub),
                        "_r_df" => put("_r_df", df.unwrap_or(SYSMISS)),
                        _ => {
                            put("_r_lb", lb);
                            put("_r_ub", ub);
                        }
                    }
                }
                continue;
            }
            // a saved scalar (or macro): r() first for r(...) and bare names
            // after an r-class command, e() otherwise
            let from_e = w.starts_with("e(") || (!w.starts_with("r(") && !(r_new && r.scalars.contains_key(&lv)) && e.scalars.contains_key(&lv));
            let src = if from_e { &e } else { &r };
            if let Some(x) = src.scalars.get(&lv) {
                g.items.push((vec![("result".into(), lv.clone())], Val::Num(*x)));
            } else if let Some(m) = src.macros.get(&lv) {
                g.items.push((vec![("result".into(), lv.clone())], Val::Str(m.clone())));
            } else {
                return Err(rc_err(111, format!("result {w} not found after {}", sp.line)));
            }
        }
        got.push(g);
    }

    // build the collection
    let st = &mut me.engine.store;
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
    let base_k = if append { c.levels.get("command").map_or(0, Vec::len) } else { 0 };
    let mut all_coefs: Vec<(Option<String>, String)> = Vec::new();
    for (k, (g, sp)) in got.iter().zip(&specs_v).enumerate() {
        let lv = (base_k + k + 1).to_string();
        c.set_level_label("command", &lv, sp.name.as_deref().unwrap_or(&sp.line));
        for (tags, v) in &g.items {
            let mut t = vec![("command".to_string(), lv.clone())];
            t.extend(tags.iter().cloned());
            c.add(t, v.clone());
        }
        for cf in &g.coefs {
            if !all_coefs.contains(cf) {
                all_coefs.push(cf.clone());
            }
        }
    }
    c.dim_labels.insert("command".into(), "Command".into());
    c.dim_labels.insert("colname".into(), "Covariate names".into());
    c.dim_labels.insert("result".into(), "Result".into());
    // coefficient order: as first seen, the intercept last
    let mut names: Vec<String> = Vec::new();
    for (_, n) in &all_coefs {
        if !names.contains(n) && n != "_cons" {
            names.push(n.clone());
        }
    }
    if all_coefs.iter().any(|(_, n)| n == "_cons") {
        names.push("_cons".into());
        c.set_level_label("colname", "_cons", "Intercept");
    }
    if !names.is_empty() {
        c.levels.insert("colname".into(), names);
    }
    // result order as requested; _r_ci is the composite of the bounds
    let mut res: Vec<String> = Vec::new();
    for r in &order_results {
        if r == "_r_ci" {
            c.composites.insert("_r_ci".into(), vec!["_r_lb".into(), "_r_ub".into()]);
            c.set_level_label("result", "_r_ci", &format!("{}% CI", ferrostat_cmd::format::fmt_g(level, 4)));
        }
        if let Some(l) = ferrostat_cmd::collect::default_result_label(r) {
            if r != "_r_ci" {
                c.set_level_label_default("result", r, l);
            }
        }
        if !res.contains(r) {
            res.push(r.clone());
        }
    }
    if level != 95.0 {
        let lv = ferrostat_cmd::format::fmt_g(level, 4);
        c.set_level_label("result", "_r_lb", &format!("{lv}% lower bound"));
        c.set_level_label("result", "_r_ub", &format!("{lv}% upper bound"));
    }
    c.levels.insert("result".into(), res);
    let ncmd = c.levels.get("command").map_or(0, Vec::len);
    if ncmd > 0 {
        c.levels.insert("command".into(), (1..=ncmd).map(|k| k.to_string()).collect());
    }

    // layout: requested dimensions, then colname on the rows and command
    // and result on the columns where not placed
    let mut axes = axes;
    axes.resize(3, Vec::new());
    let placed = |d: &str, axes: &[Vec<String>]| axes.iter().flatten().any(|x| x == d);
    let mut rows = axes[0].clone();
    let mut cols = axes[1].clone();
    let tabs = axes[2].clone();
    let has_coef = c.levels.contains_key("colname") && c.items.iter().any(|it| it.tag("colname").is_some());
    if has_coef && !placed("colname", &axes) {
        rows.insert(0, "colname".into());
    }
    if c.dims.iter().any(|d| d == "coleq") && !placed("coleq", &axes) {
        let at = rows.iter().position(|d| d == "colname").unwrap_or(rows.len());
        rows.insert(at, "coleq".into());
    }
    for d in ["command", "result"] {
        if !placed(d, &axes) {
            if rows.is_empty() && !has_coef && d == "command" {
                rows.push(d.into());
            } else {
                cols.push(d.into());
            }
        }
    }
    let term = |dims: &[String]| -> Vec<Term> { if dims.is_empty() { Vec::new() } else { vec![dims.iter().map(|d| Cond { dim: d.clone(), levels: None }).collect()] } };
    c.layout = Some(Layout { rows: term(&rows), cols: term(&cols), tabs: term(&tabs) });
    c.preview()
}
