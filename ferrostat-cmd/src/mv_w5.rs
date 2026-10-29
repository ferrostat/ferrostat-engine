//! Further multivariate tools: postestimation plots after ca, mca and mds
//! (`caprojection`, `mcaplot`, `mcaprojection`, `mdsshepard`), the
//! remaining `estat` subcommands after discrim (summarize, covariance,
//! correlations, grdistances, manova) and the cluster programming
//! utilities (`cluster set`, `cluster delete`, `cluster parsedistance`,
//! `cluster measures`).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::graph::{finish, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::mv_cluster::{load_objs, save_objs, ClObj, Measure};
use crate::mvutil;
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// plots
// ---------------------------------------------------------------------------

fn keep_graph_opts(cmd: &Command) -> Command {
    let mut c = cmd.clone();
    c.opts.retain(|(n, _)| ["title", "xtitle", "ytitle", "name", "saving"].iter().any(|k| n.eq_ignore_ascii_case(k)));
    c
}

fn need_cmd(st: &Store, name: &str) -> Result<(), String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some(name) {
        return Err(rc_err(301, "last estimates not found"));
    }
    Ok(())
}

/// Dimensions to show: `dimensions(# #)` or the first two.
fn dims_opt(cmd: &Command, ncol: usize) -> Result<(usize, usize), String> {
    let d: Vec<usize> = match cmd.opt("dimensions", 3) {
        Some(s) => s.split_whitespace().map(|t| t.parse::<usize>().map_err(|_| rc_err(198, "dimensions() invalid"))).collect::<Result<_, _>>()?,
        None => vec![1, 2],
    };
    let a = *d.first().unwrap_or(&1);
    let b = *d.get(1).unwrap_or(&2);
    if a == 0 || b == 0 || a > ncol || b > ncol.max(1) {
        return Err(rc_err(198, "dimensions() out of range"));
    }
    Ok((a - 1, (b - 1).min(ncol.saturating_sub(1))))
}

/// Projection plot: one column of labelled markers per dimension.
fn projection(title: &str, sets: &[(&str, &Matrix)], cmd: &Command, st: &mut Store) -> Result<String, String> {
    let ncol = sets.iter().map(|(_, m)| m.cols).max().unwrap_or(0);
    let k = match cmd.opt("dimensions", 3) {
        Some(s) => s.split_whitespace().filter_map(|t| t.parse::<usize>().ok()).max().unwrap_or(2),
        None => ncol.min(2),
    }
    .min(ncol)
    .max(1);
    let mut g = Graph { title: title.into(), xtitle: "Dimension".into(), ytitle: "Score".into(), legend: sets.len() > 1, ..Default::default() };
    for (si, (lab, m)) in sets.iter().enumerate() {
        let mut pts = Vec::new();
        let mut labels = Vec::new();
        for d in 0..k.min(m.cols) {
            for i in 0..m.rows {
                // rows and columns side by side within a dimension
                pts.push(((d + 1) as f64 + if sets.len() > 1 { 0.2 * (si as f64 - 0.5) } else { 0.0 }, m[(i, d)]));
                labels.push(m.rownames.get(i).cloned().unwrap_or_default());
            }
        }
        g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, mlabels: labels, label: lab.to_string(), hollow: si > 0, ..Default::default() });
    }
    g.command = cmd.raw.clone();
    finish(&mut g, &keep_graph_opts(cmd), st)
}

pub struct Caprojection;
impl CommandHandler for Caprojection {
    fn name(&self) -> &'static str {
        "caprojection"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        need_cmd(st, "ca").or_else(|_| need_cmd(st, "camat"))?;
        let tr = st.e.matrices.get("TR").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let tc = st.e.matrices.get("TC").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let rn = st.e.macros.get("Rname").cloned().unwrap_or_else(|| "rows".into());
        let cn = st.e.macros.get("Cname").cloned().unwrap_or_else(|| "columns".into());
        let mut sets: Vec<(&str, &Matrix)> = Vec::new();
        if !cmd.has_opt("norow", 5) {
            sets.push((&rn, &tr));
        }
        if !cmd.has_opt("nocolumn", 5) {
            sets.push((&cn, &tc));
        }
        projection("CA dimension projection plot", &sets, cmd, st)
    }
}

pub struct Mcaprojection;
impl CommandHandler for Mcaprojection {
    fn name(&self) -> &'static str {
        "mcaprojection"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        need_cmd(st, "mca")?;
        let c = st.e.matrices.get("cGS").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        projection("MCA dimension projection plot", &[("categories", &c)], cmd, st)
    }
}

pub struct Mcaplot;
impl CommandHandler for Mcaplot {
    fn name(&self) -> &'static str {
        "mcaplot"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        need_cmd(st, "mca")?;
        let c = st.e.matrices.get("cGS").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let (a, b) = dims_opt(cmd, c.cols)?;
        let names: Vec<String> = st.e.macros.get("names").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
        let mut g = Graph { title: "MCA coordinate plot".into(), xtitle: format!("Dimension {}", a + 1), ytitle: format!("Dimension {}", b + 1), legend: true, ..Default::default() };
        // categories are listed variable by variable; split them by the
        // category counts of each variable when the data are still in memory
        let counts = category_counts(_ds, &names, st);
        let mut start = 0;
        for (vi, nm) in names.iter().enumerate() {
            let cnt = counts.get(vi).copied().unwrap_or(c.rows - start);
            let end = (start + cnt).min(c.rows);
            let pts: Vec<(f64, f64)> = (start..end).map(|i| (c[(i, a)], c[(i, b)])).collect();
            let lab: Vec<String> = (start..end).map(|i| c.rownames.get(i).cloned().unwrap_or_default()).collect();
            g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, mlabels: lab, label: nm.clone(), ..Default::default() });
            start = end;
        }
        if start < c.rows || names.is_empty() {
            let pts: Vec<(f64, f64)> = (start..c.rows).map(|i| (c[(i, a)], c[(i, b)])).collect();
            let lab: Vec<String> = (start..c.rows).map(|i| c.rownames.get(i).cloned().unwrap_or_default()).collect();
            g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, mlabels: lab, ..Default::default() });
        }
        g.command = cmd.raw.clone();
        finish(&mut g, &keep_graph_opts(cmd), st)
    }
}

fn category_counts(ds: &DataSet, names: &[String], st: &Store) -> Vec<usize> {
    let sample = st.e.sample.clone();
    names
        .iter()
        .map(|v| {
            let c = util::col(ds, v).unwrap_or_default();
            let mut vals: Vec<f64> = c.iter().enumerate().filter(|(i, x)| !is_missing(**x) && sample.as_ref().map_or(true, |s| s.get(*i).copied().unwrap_or(false))).map(|(_, x)| *x).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            vals.len()
        })
        .collect()
}

pub struct Mdsshepard;
impl CommandHandler for Mdsshepard {
    fn name(&self) -> &'static str {
        "mdsshepard"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        need_cmd(st, "mds")?;
        let y = st.e.matrices.get("Y").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let d = st.e.matrices.get("Disparities").or_else(|| st.e.matrices.get("D")).cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let n = y.rows;
        let mut pts = Vec::new();
        for i in 0..n {
            for j in 0..i {
                let dist: f64 = (0..y.cols).map(|k| (y[(i, k)] - y[(j, k)]).powi(2)).sum::<f64>().sqrt();
                pts.push((d[(i, j)], dist));
            }
        }
        let mx = pts.iter().map(|p| p.0.max(p.1)).fold(0.0, f64::max);
        let mut g = Graph { title: "Shepard diagram".into(), xtitle: "Dissimilarities".into(), ytitle: "Distances".into(), ..Default::default() };
        g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, ..Default::default() });
        g.layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (mx, mx)], ..Default::default() });
        g.command = cmd.raw.clone();
        finish(&mut g, &keep_graph_opts(cmd), st)
    }
}

// ---------------------------------------------------------------------------
// estat after discrim
// ---------------------------------------------------------------------------

struct Groups {
    vars: Vec<String>,
    gvar: String,
    glabels: Vec<String>,
    /// per group: observations (rows of x)
    x: Vec<Vec<Vec<f64>>>,
}

fn groups(ds: &DataSet, st: &Store) -> Result<Groups, String> {
    let e = &st.e;
    let vars: Vec<String> = e.macros.get("varlist").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let gvar = e.macros.get("groupvar").cloned().unwrap_or_default();
    let gv = e.matrices.get("groupvalues").map(|m| m.data.clone()).unwrap_or_default();
    let glabels: Vec<String> = e.macros.get("grouplabels").map(|s| s.split('\u{1f}').map(String::from).collect()).unwrap_or_else(|| gv.iter().map(|v| crate::format::fmt_g(*v, 8)).collect());
    let sample = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let gc = util::col(ds, &gvar)?;
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let mut x = vec![Vec::new(); gv.len()];
    for i in 0..ds.n_obs.min(sample.len()) {
        if !sample[i] {
            continue;
        }
        let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
        if row.iter().any(|v| is_missing(*v)) {
            continue;
        }
        if let Some(g) = gv.iter().position(|&v| v == gc[i]) {
            x[g].push(row);
        }
    }
    Ok(Groups { vars, gvar, glabels, x })
}

fn mean_cov(rows: &[Vec<f64>], p: usize) -> (Vec<f64>, Matrix) {
    let n = rows.len() as f64;
    let m: Vec<f64> = (0..p).map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n).collect();
    let mut s = Matrix::zeros(p, p);
    for r in rows {
        for a in 0..p {
            for b in 0..p {
                s[(a, b)] += (r[a] - m[a]) * (r[b] - m[b]);
            }
        }
    }
    (m, s)
}

fn to_corr(s: &Matrix) -> Matrix {
    let mut r = s.clone();
    for a in 0..s.rows {
        for b in 0..s.cols {
            r[(a, b)] = s[(a, b)] / (s[(a, a)] * s[(b, b)]).sqrt();
        }
    }
    r
}

/// Extra estat subcommands after discrim; None when `sub` is not one.
pub fn discrim_estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let known = ["summarize", "su", "sum", "summ", "covariance", "cov", "covar", "correlations", "cor", "corr", "correlation", "grdistances", "grdist", "grd", "manova"];
    if !known.contains(&sub) {
        return None;
    }
    Some(discrim_estat_run(ds, st, sub, cmd))
}

fn discrim_estat_run(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let g = groups(ds, st)?;
    let p = g.vars.len();
    let gn = g.x.len();
    let all: Vec<Vec<f64>> = g.x.iter().flatten().cloned().collect();
    let n = all.len();
    let mut out = vec![String::new()];
    match sub {
        "summarize" | "su" | "sum" | "summ" => {
            out.push("  Estimation sample discrim".into());
            out.push(String::new());
            out.push("  -------------------------------------------------------------".into());
            out.push("      Variable |         Mean   Std. dev.         Min         Max".into());
            out.push("  -------------+-----------------------------------------------".into());
            let gall: Vec<f64> = g.x.iter().enumerate().flat_map(|(k, r)| std::iter::repeat_n(k, r.len())).map(|k| st.e.matrices.get("groupvalues").map(|m| m.data[k]).unwrap_or(k as f64)).collect();
            let mut stats = Matrix::zeros(p + 1, 4);
            let line = |name: &str, v: &[f64], row: usize, stats: &mut Matrix| {
                let mn = util::mean(v);
                let sd = util::variance(v).sqrt();
                let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for (c, x) in [mn, sd, lo, hi].into_iter().enumerate() {
                    stats[(row, c)] = x;
                }
                format!("  {:>12} | {:>12} {:>11} {:>11} {:>11}", crate::summarize::abbrev(name, 12), sig(mn), sig(sd), sig(lo), sig(hi))
            };
            out.push(line(&g.gvar, &gall, 0, &mut stats));
            out.push("  -------------+-----------------------------------------------".into());
            for (j, v) in g.vars.iter().enumerate() {
                let c: Vec<f64> = all.iter().map(|r| r[j]).collect();
                out.push(line(v, &c, j + 1, &mut stats));
            }
            out.push("  -------------------------------------------------------------".into());
            let mut rn = vec![g.gvar.clone()];
            rn.extend(g.vars.iter().cloned());
            st.begin_r().set_mat("stats", stats.with_names(rn, vec!["mean".into(), "sd".into(), "min".into(), "max".into()]));
        }
        "covariance" | "cov" | "covar" | "correlations" | "cor" | "corr" | "correlation" => {
            let corr = sub.starts_with("cor");
            let mut within = Matrix::zeros(p, p);
            let mut per = Vec::new();
            for rows in &g.x {
                let (_, s) = mean_cov(rows, p);
                within = within.add(&s).unwrap_or(within.clone());
                per.push(s.scale(1.0 / (rows.len() as f64 - 1.0).max(1.0)));
            }
            let within = within.scale(1.0 / (n as f64 - gn as f64).max(1.0));
            let (_, tot) = mean_cov(&all, p);
            let tot = tot.scale(1.0 / (n as f64 - 1.0).max(1.0));
            let what = if corr { "correlation" } else { "covariance" };
            let tf = |m: &Matrix| if corr { to_corr(m) } else { m.clone() };
            let r = st.begin_r();
            let show_w = !cmd.has_opt("group", 2) && !cmd.has_opt("total", 3) || cmd.has_opt("within", 3);
            let key = if corr { "Rho" } else { "S" };
            let named = |m: Matrix| m.with_names(g.vars.clone(), g.vars.clone());
            if show_w {
                out.push(format!("Pooled within-group {what} matrix"));
                let m = named(tf(&within));
                out.extend(mvutil::matlist(&m, "", "%9.0g", 10, true, 4));
                out.push(String::new());
                r.set_mat(key, m);
            }
            if cmd.has_opt("group", 2) {
                for (k, s) in per.iter().enumerate() {
                    out.push(format!("Group {what} matrix  {} = {}", g.gvar, g.glabels.get(k).cloned().unwrap_or_default()));
                    let m = named(tf(s));
                    out.extend(mvutil::matlist(&m, "", "%9.0g", 10, true, 4));
                    out.push(String::new());
                    r.set_mat(&format!("{key}_{}", k + 1), m);
                }
            }
            if cmd.has_opt("total", 3) {
                out.push(format!("Total-sample {what} matrix"));
                let m = named(tf(&tot));
                out.extend(mvutil::matlist(&m, "", "%9.0g", 10, true, 4));
                r.set_mat(&format!("{key}_T"), m);
            }
        }
        "grdistances" | "grdist" | "grd" => {
            let mut within = Matrix::zeros(p, p);
            let mut means = Vec::new();
            for rows in &g.x {
                let (m, s) = mean_cov(rows, p);
                within = within.add(&s).unwrap_or(within.clone());
                means.push(m);
            }
            let within = within.scale(1.0 / (n as f64 - gn as f64).max(1.0));
            let wi = mvutil::sym_inv(&within);
            let mut d = Matrix::zeros(gn, gn);
            for a in 0..gn {
                for b in 0..gn {
                    let dd: Vec<f64> = (0..p).map(|j| means[a][j] - means[b][j]).collect();
                    d[(a, b)] = (0..p).map(|i| (0..p).map(|j| dd[i] * wi[(i, j)] * dd[j]).sum::<f64>()).sum();
                }
            }
            let d = d.with_names(g.glabels.clone(), g.glabels.clone());
            out.push("Mahalanobis squared distances between groups".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&d, &g.gvar, "%9.0g", 10, true, 4));
            st.begin_r().set_mat("sqdist", d);
        }
        "manova" => {
            let saved = st.e.clone();
            let sample = saved.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
            let tmp = "__discrim_touse";
            let _ = ds.drop_variable(tmp);
            ds.add_variable(util::new_double(tmp, sample.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()))?;
            let line = format!("manova {} = {} if {tmp}", g.vars.join(" "), g.gvar);
            let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message));
            let r = c.and_then(|c| crate::manova::Manova.execute(ds, st, &c));
            let _ = ds.drop_variable(tmp);
            let m_e = std::mem::replace(&mut st.e, saved);
            let text = r?;
            let rr = st.begin_r();
            for (k, v) in &m_e.scalars {
                rr.set(k, *v);
            }
            for (k, v) in &m_e.matrices {
                rr.set_mat(k, v.clone());
            }
            return Ok(text);
        }
        _ => unreachable!(),
    }
    Ok(out.join("\n"))
}

fn sig(x: f64) -> String {
    if is_missing(x) {
        ".".into()
    } else {
        crate::format::fmt_g(x, 9)
    }
}

// ---------------------------------------------------------------------------
// cluster programming utilities
// ---------------------------------------------------------------------------

/// `cluster set|delete|parsedistance|measures`; None when `sub` is not one.
pub fn cluster_more(ds: &mut DataSet, st: &mut Store, cmd: &Command, sub: &str, rest: &str) -> Option<Result<String, String>> {
    Some(match sub {
        "set" => cluster_set(ds, cmd, rest),
        "delete" | "del" | "dele" | "delet" => cluster_delete(ds, cmd, rest),
        "parsedistance" | "parsedist" => parse_distance(st, rest),
        "measures" | "measure" => cluster_measures(ds, st, cmd, rest),
        _ => return None,
    })
}

fn pick_obj<'a>(objs: &'a mut Vec<ClObj>, rest: &str, add: bool) -> Result<&'a mut ClObj, String> {
    let name = rest.split_whitespace().next().map(String::from);
    let idx = match &name {
        Some(n) => match objs.iter().position(|o| &o.name == n) {
            Some(i) => i,
            None if add => {
                objs.insert(0, ClObj { name: n.clone(), ..Default::default() });
                0
            }
            None => return Err(rc_err(198, format!("{n} not found"))),
        },
        None if !objs.is_empty() => 0,
        None => return Err(rc_err(198, "no cluster solutions defined")),
    };
    Ok(&mut objs[idx])
}

fn tag_val(s: &str) -> Result<(String, String), String> {
    let s = s.trim();
    let (t, v) = s.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "tag and value required"))?;
    Ok((t.to_string(), util::unquote(v.trim())))
}

fn cluster_set(ds: &mut DataSet, cmd: &Command, rest: &str) -> Result<String, String> {
    let mut objs = load_objs(ds);
    let o = pick_obj(&mut objs, rest, cmd.has_opt("addname", 3))?;
    for (k, v) in &cmd.opts {
        let k = k.to_lowercase();
        let v = v.clone().unwrap_or_default();
        match k.as_str() {
            "type" => o.typ = v.trim().to_string(),
            "method" => o.method = v.trim().to_string(),
            "dissimilarity" | "dissim" => {
                o.measure = v.trim().to_string();
                o.sim = false;
            }
            "similarity" | "sim" => {
                o.measure = v.trim().to_string();
                o.sim = true;
            }
            "var" => {
                let (t, name) = tag_val(&v)?;
                o.vars.retain(|(_, r)| *r != t);
                o.vars.push((name, t));
            }
            "char" | "other" => {
                let (t, val) = tag_val(&v)?;
                o.other.retain(|(kk, _)| *kk != t);
                o.other.push((t, val));
            }
            "note" => o.notes.push(util::unquote(v.trim())),
            "addname" => {}
            other => return Err(rc_err(198, format!("option {other} not allowed"))),
        }
    }
    save_objs(ds, &objs);
    Ok(String::new())
}

fn cluster_delete(ds: &mut DataSet, cmd: &Command, rest: &str) -> Result<String, String> {
    let mut objs = load_objs(ds);
    let name = pick_obj(&mut objs, rest, false)?.name.clone();
    let idx = objs.iter().position(|o| o.name == name).unwrap();
    if cmd.has_opt("zap", 3) || cmd.has_opt("delname", 4) {
        let o = objs.remove(idx);
        if cmd.has_opt("zap", 3) {
            for (v, _) in &o.vars {
                let _ = ds.drop_variable(v);
            }
        }
        save_objs(ds, &objs);
        return Ok(String::new());
    }
    let o = &mut objs[idx];
    for (k, v) in &cmd.opts {
        let v = v.clone().unwrap_or_default();
        match k.to_lowercase().as_str() {
            "type" => o.typ.clear(),
            "method" => o.method.clear(),
            "dissimilarity" | "similarity" | "dissim" | "sim" => o.measure.clear(),
            "allnotes" => o.notes.clear(),
            "notes" => {
                let del: Vec<usize> = crate::margins::numlist(&v)?.into_iter().map(|x| x as usize).collect();
                o.notes = o.notes.iter().enumerate().filter(|(i, _)| !del.contains(&(i + 1))).map(|(_, s)| s.clone()).collect();
            }
            "allvars" => o.vars.clear(),
            "allcharzap" | "allvarzap" => {
                for (vv, _) in &o.vars {
                    let _ = ds.drop_variable(vv);
                }
                o.vars.clear();
            }
            "var" => {
                let t = v.trim().to_string();
                o.vars.retain(|(_, r)| *r != t);
            }
            "varzap" => {
                let t = v.trim().to_string();
                if let Some((vv, _)) = o.vars.iter().find(|(_, r)| *r == t).cloned() {
                    let _ = ds.drop_variable(&vv);
                }
                o.vars.retain(|(_, r)| *r != t);
            }
            "allothers" | "allchars" => o.other.clear(),
            "other" | "char" => {
                let t = v.trim().to_string();
                o.other.retain(|(kk, _)| *kk != t);
            }
            other => return Err(rc_err(198, format!("option {other} not allowed"))),
        }
    }
    save_objs(ds, &objs);
    Ok(String::new())
}

fn parse_distance(st: &mut Store, rest: &str) -> Result<String, String> {
    let txt = rest.trim();
    let m = Measure::parse(if txt.is_empty() { "L2" } else { txt })?;
    let full = m.name();
    let (dist, darg) = match full.split_once('(') {
        Some((a, b)) => (a.to_string(), b.trim_end_matches(')').to_string()),
        None => (full.clone(), String::new()),
    };
    let r = st.begin_r();
    r.set_str("dist", dist);
    r.set_str("darg", darg);
    r.set_str("unab", full);
    r.set_str("dtype", if m.is_sim() { "similarity" } else { "dissimilarity" });
    if m.is_binary() {
        r.set_str("binary", "binary");
    }
    Ok(String::new())
}

fn cluster_measures(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let vars = crate::varlist::expand(ds, if rest.trim().is_empty() { "_all" } else { rest })?;
    let cmp: Vec<usize> = crate::margins::numlist(cmd.opt("compare", 3).ok_or_else(|| rc_err(198, "option compare() required"))?)?.into_iter().map(|v| v as usize).collect();
    let gens: Vec<String> = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.split_whitespace().map(String::from).collect();
    if gens.len() != cmp.len() {
        return Err(rc_err(198, "generate() must name one variable per compare() observation"));
    }
    let m = Measure::from_cmd(cmd, Measure::L2)?;
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &vars);
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let rows = util::rows(&mask);
    let ranges: Vec<f64> = cols.iter().map(|c| {
        let v: Vec<f64> = rows.iter().map(|&i| c[i]).collect();
        v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min)
    }).collect();
    let binvar = vec![false; vars.len()];
    for (c, g) in cmp.iter().zip(&gens) {
        if *c == 0 || *c > ds.n_obs {
            return Err(rc_err(198, format!("compare() observation {c} out of range")));
        }
        crate::varlist::check_name(g)?;
        if ds.var_index.contains_key(g) {
            return Err(rc_err(110, format!("variable {g} already defined")));
        }
        let xc: Vec<f64> = cols.iter().map(|col| col[c - 1]).collect();
        let vals: Vec<f64> = (0..ds.n_obs)
            .map(|i| {
                if !mask[i] || xc.iter().any(|v| is_missing(*v)) {
                    return SYSMISS;
                }
                let xi: Vec<f64> = cols.iter().map(|col| col[i]).collect();
                m.eval(&xi, &xc, &ranges, &binvar)
            })
            .collect();
        ds.add_variable(util::new_double(g, vals))?;
    }
    st.data_changed = true;
    Ok(String::new())
}
