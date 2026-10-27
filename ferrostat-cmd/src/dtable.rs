//! `dtable`: a table of descriptive statistics ("Table 1"), optionally by
//! groups with tests of equality across them.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, split_options};
use ferrostat_parser::Command;

use crate::collect::{CellOpts, Cond, HMode, Layout, Style};
use crate::coltable::{fv_levels, level_key, var_label, value_text};
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::table::{canonical_stat, eval_stat, Stat};
use crate::util::{self, WeightKind, Weights};
use crate::varlist;

#[derive(Clone, Debug)]
struct RowVar {
    var: String,
    factor: bool,
    stats: Vec<String>,
    test: Option<String>,
}

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

fn freq(rows: &[usize], w: Option<&Weights>) -> f64 {
    match w {
        Some(w) if matches!(w.kind, WeightKind::F | WeightKind::I) => rows.iter().map(|&i| w.w[i]).sum(),
        _ => rows.len() as f64,
    }
}

fn stat_list(s: &str) -> Result<Vec<String>, String> {
    s.split_whitespace()
        .map(|x| {
            let l = x.to_lowercase();
            Ok(match l.as_str() {
                "fvfrequency" | "fvfreq" => "fvfrequency".to_string(),
                "fvpercent" | "fvperc" => "fvpercent".to_string(),
                "fvproportion" | "fvprop" => "fvproportion".to_string(),
                "iqi" | "rangei" | "idi" => l,
                "proportion" | "prop" => "proportion".into(),
                _ => canonical_stat(&l)?,
            })
        })
        .collect()
}

/// p-value of the one-way ANOVA F test of `x` across `groups`.
fn anova_p(x: &[f64], g: &[usize]) -> f64 {
    let n = x.len() as f64;
    let k = g.iter().copied().max().map_or(0, |m| m + 1);
    let mut sum = vec![0.0; k];
    let mut cnt = vec![0.0; k];
    for (v, &gi) in x.iter().zip(g) {
        sum[gi] += v;
        cnt[gi] += 1.0;
    }
    let groups = cnt.iter().filter(|c| **c > 0.0).count() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let ssb: f64 = (0..k).filter(|&j| cnt[j] > 0.0).map(|j| cnt[j] * (sum[j] / cnt[j] - mean).powi(2)).sum();
    let ssw: f64 = x.iter().zip(g).map(|(v, &gi)| (v - sum[gi] / cnt[gi]).powi(2)).sum();
    if groups < 2.0 || n - groups <= 0.0 || ssw <= 0.0 {
        return SYSMISS;
    }
    let f = (ssb / (groups - 1.0)) / (ssw / (n - groups));
    dist::ftail(groups - 1.0, n - groups, f)
}

/// p-value of Pearson's chi-squared test of independence.
fn pearson_p(a: &[usize], b: &[usize]) -> f64 {
    let ra = a.iter().copied().max().map_or(0, |m| m + 1);
    let rb = b.iter().copied().max().map_or(0, |m| m + 1);
    let mut t = vec![vec![0.0; rb]; ra];
    for (&i, &j) in a.iter().zip(b) {
        t[i][j] += 1.0;
    }
    let n: f64 = a.len() as f64;
    let rs: Vec<f64> = t.iter().map(|r| r.iter().sum()).collect();
    let cs: Vec<f64> = (0..rb).map(|j| t.iter().map(|r| r[j]).sum()).collect();
    let r_used = rs.iter().filter(|x| **x > 0.0).count();
    let c_used = cs.iter().filter(|x| **x > 0.0).count();
    if r_used < 2 || c_used < 2 {
        return SYSMISS;
    }
    let mut chi2 = 0.0;
    for i in 0..ra {
        for j in 0..rb {
            let e = rs[i] * cs[j] / n;
            if e > 0.0 {
                chi2 += (t[i][j] - e).powi(2) / e;
            }
        }
    }
    dist::chi2tail(((r_used - 1) * (c_used - 1)) as f64, chi2)
}

fn and_list(v: &[String]) -> String {
    match v.len() {
        0 => String::new(),
        1 => v[0].clone(),
        2 => format!("{} and {}", v[0], v[1]),
        n => format!("{}, and {}", v[..n - 1].join(", "), v[n - 1]),
    }
}

pub fn run(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    // variables
    let mut rows: Vec<RowVar> = Vec::new();
    let add_vars = |ds: &DataSet, text: &str, factor_default: bool, stats: Option<Vec<String>>, test: Option<String>, rows: &mut Vec<RowVar>| -> Result<(), String> {
        for w in text.split_whitespace() {
            let (factor, name) = match w.strip_prefix("i.").or_else(|| w.strip_prefix("ib.")) {
                Some(n) => (true, n.trim_start_matches('(').trim_end_matches(')').to_string()),
                None => (factor_default, w.strip_prefix("c.").unwrap_or(w).to_string()),
            };
            for v in varlist::expand(ds, &name)? {
                let stats = stats.clone().unwrap_or_else(|| if factor { vec!["fvfrequency".into(), "fvpercent".into()] } else { vec!["mean".into(), "sd".into()] });
                let test = test.clone();
                match rows.iter_mut().find(|r| r.var == v) {
                    Some(r) => {
                        r.factor = factor;
                        r.stats = stats;
                        r.test = test.or(r.test.clone());
                    }
                    None => rows.push(RowVar { var: v, factor, stats, test }),
                }
            }
        }
        Ok(())
    };
    add_vars(ds, &cmd.main, false, None, None, &mut rows)?;
    let mut sample = true;
    let mut sample_stats: Option<Vec<String>> = None;
    let mut sample_title = "N".to_string();
    let mut edits: Vec<(Vec<Cond>, CellOpts)> = Vec::new();
    let mut by: Option<String> = None;
    let mut tests = false;
    let mut by_totals = true;
    let mut notes: Vec<String> = Vec::new();
    for (n, a) in &cmd.opts {
        let n = n.to_lowercase();
        let a = a.clone().unwrap_or_default();
        if n.len() >= 4 && "continuous".starts_with(n.as_str()) {
            let (vars, o) = split_arg(&a);
            let stats = sub_opt(&o, "statistics", 4).map(|s| stat_list(&s)).transpose()?;
            add_vars(ds, &vars, false, stats, sub_opt(&o, "test", 4), &mut rows)?;
        } else if n.len() >= 3 && "factor".starts_with(n.as_str()) {
            let (vars, o) = split_arg(&a);
            let stats = sub_opt(&o, "statistics", 4).map(|s| stat_list(&s)).transpose()?;
            add_vars(ds, &vars, true, stats, sub_opt(&o, "test", 4), &mut rows)?;
        } else if n == "by" {
            let (v, o) = split_arg(&a);
            by = Some(varlist::resolve_name(ds, v.trim())?);
            tests = sub_opt(&o, "tests", 4).is_some() && sub_opt(&o, "notests", 6).is_none();
            by_totals = sub_opt(&o, "nototals", 5).is_none();
        } else if n == "nosample" {
            sample = false;
        } else if n.len() >= 4 && "sample".starts_with(n.as_str()) {
            let (t, o) = split_arg(&a);
            if !t.is_empty() {
                sample_title = util::unquote(&t);
            }
            sample_stats = sub_opt(&o, "statistics", 4).map(|s| stat_list(&s)).transpose()?;
        } else if n.len() >= 2 && ("nformat".starts_with(n.as_str()) || "sformat".starts_with(n.as_str())) {
            let t = a.trim();
            let (f, rest) = if t.starts_with('"') {
                let end = t[1..].find('"').map(|p| p + 2).unwrap_or(t.len());
                (util::unquote(&t[..end]), t[end..].trim().to_string())
            } else {
                let mut sp = t.splitn(2, char::is_whitespace);
                (sp.next().unwrap_or("").to_string(), sp.next().unwrap_or("").trim().to_string())
            };
            let on = stat_list(&rest).unwrap_or_else(|_| rest.split_whitespace().map(String::from).collect());
            let conds = if on.is_empty() { Vec::new() } else { vec![Cond { dim: "result".into(), levels: Some(on) }] };
            let o = if n.starts_with('n') { CellOpts { nformat: Some(f), ..Default::default() } } else { CellOpts { sformat: Some(f), ..Default::default() } };
            edits.push((conds, o));
        } else if n == "note" || n == "notes" {
            notes.push(util::unquote(&a));
        }
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "faip")?;
    if let Some(b) = &by {
        if !cmd.has_opt("missing", 4) {
            util::markout(ds, &mut mask, &[b.clone()]);
        }
    }
    if cmd.has_opt("listwise", 4) && !cmd.has_opt("nolistwise", 6) {
        let vs: Vec<String> = rows.iter().map(|r| r.var.clone()).collect();
        util::markout(ds, &mut mask, &vs);
    }
    let all = util::rows(&mask);
    if all.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    // by groups: (level key, label, rows); then the total
    let mut cols: Vec<(String, Vec<usize>)> = Vec::new();
    let c = st.collect.fresh("DTable");
    let mut style = Style::named("dtable").unwrap();
    style.cell.extend(edits);
    let col_dim = by.clone().unwrap_or_else(|| "_dtable_col".to_string());
    if let Some(b) = &by {
        let (ids, reps) = util::group_ids(ds, &[b.clone()], &mask);
        let var = ds.get_var(b).unwrap();
        for (g, &r) in reps.iter().enumerate() {
            let key = level_key(ds, b, r);
            c.set_level_label(b, &key, &util::cell_text(ds, var, r, true));
            cols.push((key, all.iter().copied().filter(|&i| ids[i] == g).collect()));
        }
        c.dim_labels.insert(b.clone(), var_label(ds, b));
        style.title_mode.insert(b.clone(), HMode::Label);
        style.level_mode.insert("result".into(), HMode::Hide);
        if by_totals {
            cols.push((".m".into(), all.clone()));
            c.set_level_label(b, ".m", "Total");
        }
    } else {
        cols.push(("1".into(), all.clone()));
        c.set_level_label("_dtable_col", "1", "Summary");
        style.level_mode.insert("_dtable_col".into(), HMode::Label);
        style.level_mode.insert("result".into(), HMode::Hide);
    }
    c.style = style;
    let mut comp: Vec<String> = Vec::new();
    let push_comp = |s: &str, comp: &mut Vec<String>| {
        if !comp.iter().any(|x| x == s) {
            comp.push(s.to_string());
        }
    };
    let sample_stats = sample_stats.unwrap_or_else(|| if by.is_some() { vec!["frequency".into(), "percent".into()] } else { vec!["frequency".into()] });
    let grand = freq(&all, w.as_ref());
    // sample row
    if sample {
        c.set_level_label("var", "_N", &sample_title);
        for (key, r) in &cols {
            for s in &sample_stats {
                let v = match s.as_str() {
                    "frequency" => freq(r, w.as_ref()),
                    "percent" => 100.0 * freq(r, w.as_ref()) / grand,
                    "proportion" => freq(r, w.as_ref()) / grand,
                    _ => continue,
                };
                c.num(&[("var", "_N"), (&col_dim, key), ("result", s)], v);
                push_comp(s, &mut comp);
            }
        }
    }
    let mut cont_tested: Vec<String> = Vec::new();
    let mut fact_tested: Vec<String> = Vec::new();
    for rv in &rows {
        let col = ds.get_var(&rv.var).unwrap();
        if rv.factor {
            let levels = fv_levels(ds, &rv.var, &all);
            for &k in &levels {
                let key = format!("{}.{}", fmt_g(k, 12), rv.var);
                c.set_level_label("var", &key, &value_text(ds, &rv.var, k));
                c.set_group("var", &key, &var_label(ds, &rv.var));
            }
            for (ck, r) in &cols {
                let nonmiss: Vec<usize> = r.iter().copied().filter(|&i| !is_missing(col.num(i))).collect();
                let den = freq(&nonmiss, w.as_ref());
                for &k in &levels {
                    let key = format!("{}.{}", fmt_g(k, 12), rv.var);
                    let at: Vec<usize> = nonmiss.iter().copied().filter(|&i| col.num(i) == k).collect();
                    let f = freq(&at, w.as_ref());
                    for s in &rv.stats {
                        let v = match s.as_str() {
                            "fvfrequency" | "frequency" => f,
                            "fvpercent" | "percent" => if den > 0.0 { 100.0 * f / den } else { SYSMISS },
                            "fvproportion" | "proportion" => if den > 0.0 { f / den } else { SYSMISS },
                            _ => continue,
                        };
                        c.num(&[("var", &key), (&col_dim, ck), ("result", s)], v);
                        push_comp(s, &mut comp);
                    }
                }
            }
            if tests && by.is_some() && levels.len() > 1 {
                let b = ds.get_var(by.as_ref().unwrap()).unwrap();
                let bl: Vec<f64> = fv_levels(ds, by.as_ref().unwrap(), &all);
                let pairs: Vec<(usize, usize)> = all
                    .iter()
                    .filter(|&&i| !is_missing(col.num(i)) && !is_missing(b.num(i)))
                    .map(|&i| (levels.iter().position(|&x| x == col.num(i)).unwrap(), bl.iter().position(|&x| x == b.num(i)).unwrap_or(0)))
                    .collect();
                let (a, bb): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
                let p = pearson_p(&a, &bb);
                let key = format!("{}.{}", fmt_g(levels[0], 12), rv.var);
                let tname = rv.test.clone().unwrap_or_else(|| "pearson".into());
                c.num(&[("var", &key), (&col_dim, "_test"), ("result", &tname)], p);
                push_comp(&tname, &mut comp);
                fact_tested.push(rv.var.clone());
            }
        } else {
            c.set_level_label("var", &rv.var, &var_label(ds, &rv.var));
            for (ck, r) in &cols {
                for s in &rv.stats {
                    let parts: Vec<&str> = match s.as_str() {
                        "iqi" => vec!["q1", "q3"],
                        "rangei" => vec!["min", "max"],
                        "idi" => vec!["p10", "p90"],
                        x => vec![x],
                    };
                    for p in parts {
                        let v = eval_stat(ds, &Stat { name: p.to_string(), var: Some(rv.var.clone()) }, r, w.as_ref(), grand);
                        c.num(&[("var", &rv.var), (&col_dim, ck), ("result", p)], v);
                    }
                    if matches!(s.as_str(), "iqi" | "rangei" | "idi") {
                        let els: Vec<String> = match s.as_str() {
                            "iqi" => vec!["q1".into(), "q3".into()],
                            "rangei" => vec!["min".into(), "max".into()],
                            _ => vec!["p10".into(), "p90".into()],
                        };
                        c.composites.insert(s.clone(), els);
                    }
                    push_comp(s, &mut comp);
                }
            }
            if tests && by.is_some() {
                let b = ds.get_var(by.as_ref().unwrap()).unwrap();
                let bl: Vec<f64> = fv_levels(ds, by.as_ref().unwrap(), &all);
                let obs: Vec<usize> = all.iter().copied().filter(|&i| !is_missing(col.num(i)) && !is_missing(b.num(i))).collect();
                let x: Vec<f64> = obs.iter().map(|&i| col.num(i)).collect();
                let g: Vec<usize> = obs.iter().map(|&i| bl.iter().position(|&v| v == b.num(i)).unwrap_or(0)).collect();
                let p = anova_p(&x, &g);
                let tname = rv.test.clone().unwrap_or_else(|| "regress".into());
                c.num(&[("var", &rv.var), (&col_dim, "_test"), ("result", &tname)], p);
                push_comp(&tname, &mut comp);
                cont_tested.push(rv.var.clone());
            }
        }
    }
    if let Some(b) = &by {
        if tests {
            c.set_level_label(b, "_test", "Test");
            let lv = c.levels.entry(b.clone()).or_default();
            lv.retain(|x| x != "_test");
            lv.push("_test".into());
        }
    }
    // the composite of everything shown in a cell
    c.composites.insert("_dtable_stats".into(), comp);
    c.set_level_label("result", "_dtable_stats", "Summary");
    // var order: sample first, then the variables as given
    let mut vorder: Vec<String> = Vec::new();
    if sample {
        vorder.push("_N".into());
    }
    let existing = c.levels.get("var").cloned().unwrap_or_default();
    for rv in &rows {
        for l in &existing {
            if l == &rv.var || l.ends_with(&format!(".{}", rv.var)) {
                if !vorder.contains(l) {
                    vorder.push(l.clone());
                }
            }
        }
    }
    c.levels.insert("var".into(), vorder);
    c.layout = Some(Layout {
        rows: vec![vec![Cond { dim: "var".into(), levels: None }]],
        cols: vec![vec![Cond { dim: col_dim.clone(), levels: None }, Cond { dim: "result".into(), levels: Some(vec!["_dtable_stats".into()]) }]],
        tabs: Vec::new(),
    });
    if let Some(t) = cmd.opt("title", 5) {
        c.title = Some(util::unquote(t));
    }
    c.notes.extend(notes);
    let mut out = Vec::new();
    if let Some(b) = &by {
        let lab = |vs: &[String]| and_list(vs);
        if !cont_tested.is_empty() {
            out.push(format!("note: using test regress across levels of {b} for {}.", lab(&cont_tested)));
        }
        if !fact_tested.is_empty() {
            out.push(format!("note: using test pearson across levels of {b} for {}.", lab(&fact_tested)));
        }
    }
    out.push(c.preview()?);
    Ok(out.join("\n"))
}
