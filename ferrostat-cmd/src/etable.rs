//! `etable`: estimation results side by side, from stored estimates or the
//! active ones.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, split_options};
use ferrostat_parser::Command;

use crate::collect::{default_result_label, CellOpts, Cond, Layout, Stars, Style, Val};
use crate::coltable::{value_text, var_label};
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt;
use crate::store::{Results, Store};
use crate::util;

fn split_arg(a: &str) -> (String, Vec<(String, Option<String>)>) {
    match find_top_level(a, ',') {
        Some(i) => (a[..i].trim().to_string(), split_options(&a[i + 1..])),
        None => (a.trim().to_string(), Vec::new()),
    }
}

fn sub_opt(o: &[(String, Option<String>)], name: &str, min: usize) -> Option<String> {
    o.iter()
        .find(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min && name.starts_with(n.as_str())
        })
        .map(|(_, a)| a.clone().unwrap_or_default())
}

fn cstat_name(s: &str) -> String {
    match s.trim() {
        "_r_t" => "_r_z".into(),
        x => x.to_string(),
    }
}

/// Is a coefficient name a base or omitted level?
fn is_base_or_omitted(name: &str) -> bool {
    name.split('#').any(|p| match p.split_once('.') {
        Some((lev, _)) => {
            let l = lev.trim_end_matches('n');
            l.ends_with('b') || l.ends_with('o')
        }
        None => false,
    })
}

/// `1bn.foreign` -> `1.foreign` (display key).
fn clean_name(name: &str) -> String {
    name.split('#')
        .map(|p| match p.split_once('.') {
            Some((lev, v)) => {
                let l = lev.trim_end_matches('n');
                format!("{l}.{v}")
            }
            None => p.to_string(),
        })
        .collect::<Vec<_>>()
        .join("#")
}

/// (group label, level label) of a coefficient.
fn coef_label(ds: &DataSet, name: &str, varlabel: bool, fvlabel: bool) -> (Option<String>, String) {
    if name == "_cons" {
        return (None, "Intercept".into());
    }
    let vl = |v: &str| if varlabel { var_label(ds, v) } else { v.to_string() };
    let part = |p: &str| -> (Option<String>, String) {
        match p.split_once('.') {
            Some((lev, v)) if lev.chars().all(|c| c.is_ascii_digit() || c == '-') && !lev.is_empty() => {
                let k: f64 = lev.parse().unwrap_or(0.0);
                (Some(vl(v)), if fvlabel { value_text(ds, v, k) } else { lev.to_string() })
            }
            Some(("c", v)) | Some(("co", v)) => (None, vl(v)),
            _ => (None, vl(p)),
        }
    };
    let parts: Vec<&str> = name.split('#').collect();
    if parts.len() == 1 {
        return part(parts[0]);
    }
    let labs: Vec<String> = parts
        .iter()
        .map(|p| {
            let (g, l) = part(p);
            match g {
                Some(g) => format!("{g}={l}"),
                None => l,
            }
        })
        .collect();
    (None, labs.join(" # "))
}

fn model_stat(e: &Results, name: &str) -> Option<f64> {
    match name {
        "aic" | "bic" if e.get(name).is_none() => {
            let ll = e.get("ll")?;
            let k = e.get("rank").or_else(|| e.get("df_m").map(|d| d + 1.0))?;
            let n = e.get("N")?;
            Some(if name == "aic" { -2.0 * ll + 2.0 * k } else { -2.0 * ll + k * n.ln() })
        }
        n => e.get(n),
    }
}

pub fn run(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    // models
    let mut models: Vec<(String, Results)> = Vec::new();
    match cmd.opt("estimates", 3) {
        Some(list) => {
            for n in list.split_whitespace() {
                if n == "." {
                    if st.e.is_empty() {
                        return Err(rc_err(301, "last estimates not found"));
                    }
                    models.push((".".into(), st.e.clone()));
                } else if n.contains('*') {
                    let pre = n.trim_end_matches('*');
                    for k in &st.estimates_order {
                        if k.starts_with(pre) {
                            if let Some(e) = st.estimates.get(k) {
                                models.push((k.clone(), e.clone()));
                            }
                        }
                    }
                } else {
                    let e = st.estimates.get(n).ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?;
                    models.push((n.to_string(), e.clone()));
                }
            }
        }
        None => {
            if st.e.is_empty() || !st.e.matrices.contains_key("b") {
                return Err(rc_err(301, "last estimates not found"));
            }
            models.push((".".into(), st.e.clone()));
        }
    }
    if models.is_empty() {
        return Err(rc_err(111, "no estimates found"));
    }
    let varlabel = !cmd.has_opt("novarlabel", 7);
    let fvlabel = !cmd.has_opt("nofvlabel", 6);
    let column = cmd.opt("column", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "depvar".into());
    let mut cstats: Vec<String> = Vec::new();
    let mut mstats: Vec<String> = Vec::new();
    let mut edits: Vec<(Vec<Cond>, CellOpts)> = Vec::new();
    let mut mlabels: Vec<(String, String)> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut stars_opt: Option<Stars> = None;
    let mut keep: Vec<String> = Vec::new();
    let mut drop: Vec<String> = Vec::new();
    for (n, a) in &cmd.opts {
        let n = n.to_lowercase();
        let a = a.clone().unwrap_or_default();
        if n.len() >= 3 && "cstat".starts_with(n.as_str()) {
            let (name, o) = split_arg(&a);
            let name = cstat_name(&name);
            if let Some(f) = sub_opt(&o, "nformat", 2) {
                edits.push((vec![Cond { dim: "result".into(), levels: Some(vec![name.clone()]) }], CellOpts { nformat: Some(f.trim().to_string()), ..Default::default() }));
            }
            if let Some(f) = sub_opt(&o, "sformat", 2) {
                edits.push((vec![Cond { dim: "result".into(), levels: Some(vec![name.clone()]) }], CellOpts { sformat: Some(util::unquote(&f)), ..Default::default() }));
            }
            cstats.push(name);
        } else if n.len() >= 3 && "mstat".starts_with(n.as_str()) {
            let (name, o) = split_arg(&a);
            if let Some(f) = sub_opt(&o, "nformat", 2) {
                edits.push((vec![Cond { dim: "result".into(), levels: Some(vec![name.clone()]) }], CellOpts { nformat: Some(f.trim().to_string()), ..Default::default() }));
            }
            if let Some(l) = sub_opt(&o, "label", 3) {
                mlabels.push((name.clone(), util::unquote(&l)));
            }
            mstats.push(name);
        } else if n == "note" || n == "notes" {
            notes.push(util::unquote(&a));
        } else if n == "stars" {
            let (rules_text, o) = split_arg(&a);
            let words = crate::collect::split_words(&rules_text);
            let mut rules = Vec::new();
            let mut i = 0;
            while i + 1 < words.len() {
                if let Ok(t) = words[i].parse::<f64>() {
                    rules.push((t, util::unquote(&words[i + 1])));
                }
                i += 2;
            }
            let attach = sub_opt(&o, "attach", 3).map(|x| x.split_whitespace().map(String::from).collect()).unwrap_or_else(|| vec!["_r_b".to_string()]);
            stars_opt = Some(Stars { pvalue: "_r_p".into(), rules, attach, shownote: false });
        } else if n == "keep" {
            keep.extend(a.split_whitespace().map(String::from));
        } else if n == "drop" {
            drop.extend(a.split_whitespace().map(String::from));
        }
    }
    if cstats.is_empty() {
        cstats = vec!["_r_b".into(), "_r_se".into()];
    }
    if mstats.is_empty() {
        mstats = vec!["N".into()];
    }
    let matches_spec = |name: &str, spec: &str| -> bool {
        let spec = spec.trim();
        if spec == name {
            return true;
        }
        // `foreign` or `i.foreign` matches every level of the variable
        let var = spec.strip_prefix("i.").unwrap_or(spec);
        name.split('#').count() == 1 && name.split_once('.').is_some_and(|(_, v)| v == var)
    };
    let c = st.collect.fresh("ETable");
    let mut style = Style::named("etable").unwrap();
    style.cell.extend(edits);
    let showstars = cmd.has_opt("showstars", 9) || stars_opt.is_some();
    if let Some(s) = stars_opt {
        style.stars = Some(s);
    }
    if !showstars {
        style.stars = None;
    } else if let Some(s) = style.stars.as_mut() {
        s.shownote = cmd.has_opt("showstarsnote", 13);
    }
    c.style = style;
    let mut coef_order: Vec<String> = Vec::new();
    let mut has_cons = false;
    for (mi, (name, e)) in models.iter().enumerate() {
        let col = (mi + 1).to_string();
        let header = match column.as_str() {
            "estimates" | "estimate" => name.clone(),
            "index" => col.clone(),
            "title" => e.macros.get("title").cloned().unwrap_or_else(|| name.clone()),
            "dvlabel" => e.macros.get("depvar").map(|d| var_label(ds, d.split_whitespace().next().unwrap_or(""))).unwrap_or_default(),
            _ => e.macros.get("depvar").map(|d| d.split_whitespace().next().unwrap_or("").to_string()).unwrap_or_default(),
        };
        c.set_level_label("etable_depvar", &col, &header);
        let Some(b) = e.matrices.get("b") else { continue };
        let v = e.matrices.get("V");
        let df = e.get("df_r");
        let level = e.get("level").unwrap_or(95.0);
        let crit = match df {
            Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
            None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
        };
        for (j, full) in b.colnames.iter().enumerate() {
            let bare = full.rsplit_once(':').map_or(full.as_str(), |x| x.1);
            if is_base_or_omitted(bare) {
                continue;
            }
            let key = clean_name(bare);
            if !keep.is_empty() && !keep.iter().any(|k| matches_spec(&key, k)) {
                continue;
            }
            if drop.iter().any(|k| matches_spec(&key, k)) {
                continue;
            }
            if key == "_cons" {
                has_cons = true;
            } else if !coef_order.contains(&key) {
                coef_order.push(key.clone());
            }
            let (g, l) = coef_label(ds, &key, varlabel, fvlabel);
            c.set_level_label("colname", &key, &l);
            if let Some(g) = g {
                c.set_group("colname", &key, &g);
            }
            let bv = b.get(0, j);
            let se = v.filter(|m| m.rows > j).map(|m| m.get(j, j).max(0.0).sqrt()).unwrap_or(SYSMISS);
            let z = if se > 0.0 && !is_missing(se) { bv / se } else { SYSMISS };
            let p = if is_missing(z) {
                SYSMISS
            } else {
                match df {
                    Some(d) => 2.0 * dist::ttail(d, z.abs()),
                    None => 2.0 * (1.0 - dist::normal(z.abs())),
                }
            };
            let (lb, ub) = (bv - crit * se, bv + crit * se);
            let tags = |r: &str| vec![("etable_depvar".to_string(), col.clone()), ("colname".to_string(), key.clone()), ("result".to_string(), r.to_string())];
            for (r, val) in [("_r_b", bv), ("_r_se", se), ("_r_z", z), ("_r_z_abs", z.abs()), ("_r_p", p), ("_r_lb", lb), ("_r_ub", ub)] {
                c.add(tags(r), Val::Num(val));
            }
            let f = |x: f64| fmt(x, "%9.3f").trim().to_string();
            c.add(tags("_r_ci"), Val::Str(format!("{} {}", f(lb), f(ub))));
        }
        for m in &mstats {
            if let Some(x) = model_stat(e, m) {
                c.add(vec![("etable_depvar".to_string(), col.clone()), ("result".to_string(), m.clone())], Val::Num(x));
            }
        }
    }
    if has_cons {
        coef_order.push("_cons".into());
    }
    c.levels.insert("colname".into(), coef_order);
    for (m, l) in mlabels {
        c.set_level_label("result", &m, &l);
    }
    for m in &mstats {
        if let Some(l) = default_result_label(m) {
            c.set_level_label_default("result", m, l);
        }
    }
    c.layout = Some(Layout {
        rows: vec![
            vec![Cond { dim: "colname".into(), levels: None }, Cond { dim: "result".into(), levels: Some(cstats) }],
            vec![Cond { dim: "result".into(), levels: Some(mstats) }],
        ],
        cols: vec![vec![Cond { dim: "etable_depvar".into(), levels: None }]],
        tabs: Vec::new(),
    });
    if let Some(t) = cmd.opt("title", 5) {
        c.title = Some(util::unquote(t));
    }
    c.notes.extend(notes);
    c.preview()
}
