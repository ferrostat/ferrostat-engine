//! More small commands: `boxcox`, `lowess`, `lpoly`, `dotplot`,
//! `grmeanby`, `cusum`, `bsample`, `bstat`, `dstdize`, `istdize`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("boxcox", Box::new(Boxcox)),
        ("lowess", Box::new(Lowess)),
        ("lpoly", Box::new(Lpoly)),
        ("dotplot", Box::new(Dotplot)),
        ("grmeanby", Box::new(Grmeanby)),
        ("cusum", Box::new(Cusum)),
        ("bsample", Box::new(Bsample)),
        ("bstat", Box::new(Bstat)),
        ("dstdize", Box::new(Dstdize)),
        ("istdize", Box::new(Istdize)),
    ]
}

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

fn add_var(ds: &mut DataSet, name: &str, vals: &[f64], label: &str) -> Result<(), String> {
    varlist::check_name(name)?;
    if ds.var_index.contains_key(name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mut v = Variable::new(name.to_string(), VarType::Float);
    v.label = label.to_string();
    v.data = vals.iter().map(|&x| util::typed_value(VarType::Float, x)).collect();
    ds.add_variable(v)
}

fn show_graph(g: &mut Graph, cmd: &Command, st: &mut Store) -> Result<String, String> {
    if cmd.has_opt("nograph", 4) {
        return Ok(String::new());
    }
    crate::graph::finish(g, cmd, st)
}

// ---------------------------------------------------------------------------
// boxcox
// ---------------------------------------------------------------------------

/// OLS of y on X (constant last); returns (b, rss).
fn ols(y: &[f64], x: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let n = y.len();
    let k = x.len() + 1;
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for t in 0..n {
        let row: Vec<f64> = x.iter().map(|c| c[t]).chain(std::iter::once(1.0)).collect();
        for a in 0..k {
            xty[a] += row[a] * y[t];
            for c in 0..k {
                xtx[(a, c)] += row[a] * row[c];
            }
        }
    }
    let (inv, _) = xtx.sym_inverse(1e-13);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    let rss = (0..n).map(|t| {
        let f: f64 = x.iter().enumerate().map(|(j, c)| b[j] * c[t]).sum::<f64>() + b[k - 1];
        (y[t] - f).powi(2)
    }).sum();
    (b, rss)
}

fn bc(y: f64, l: f64) -> f64 {
    if l.abs() < 1e-10 { y.ln() } else { (y.powf(l) - 1.0) / l }
}

pub struct Boxcox;

impl CommandHandler for Boxcox {
    fn name(&self) -> &'static str {
        "boxcox"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let model = cmd.opt("model", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "lhsonly".into());
        if !model.starts_with("lhs") {
            return Err(rc_err(198, "only model(lhsonly) is supported"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let yc = util::col(ds, &vars[0])?;
        let y: Vec<f64> = rows.iter().map(|&i| yc[i]).collect();
        if y.iter().any(|v| *v <= 0.0) {
            return Err(rc_err(411, format!("{} contains observations that are not strictly positive", vars[0])));
        }
        let x: Vec<Vec<f64>> = vars[1..].iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
        let n = y.len() as f64;
        let slny: f64 = y.iter().map(|v| v.ln()).sum();
        let prof = |l: f64, x: &[Vec<f64>]| -> f64 {
            let t: Vec<f64> = y.iter().map(|&v| bc(v, l)).collect();
            let (_, rss) = ols(&t, x);
            -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (rss / n).ln() + 1.0) + (l - 1.0) * slny
        };
        // maximise the profile likelihood: bracket then golden section
        let maxim = |x: &[Vec<f64>]| -> f64 {
            let grid: Vec<f64> = (-60..=60).map(|i| i as f64 * 0.1).collect();
            let mut best = grid[0];
            let mut bv = f64::NEG_INFINITY;
            for &g in &grid {
                let v = prof(g, x);
                if v > bv {
                    bv = v;
                    best = g;
                }
            }
            let (mut a, mut b) = (best - 0.1, best + 0.1);
            let gr = (5f64.sqrt() - 1.0) / 2.0;
            for _ in 0..200 {
                let c = b - gr * (b - a);
                let d = a + gr * (b - a);
                if prof(c, x) > prof(d, x) {
                    b = d;
                } else {
                    a = c;
                }
                if b - a < 1e-12 {
                    break;
                }
            }
            0.5 * (a + b)
        };
        let theta = maxim(&x);
        let ll = prof(theta, &x);
        let theta0 = maxim(&[]);
        let ll0 = prof(theta0, &[]);
        let h = 1e-4;
        let d2 = (prof(theta + h, &x) - 2.0 * ll + prof(theta - h, &x)) / (h * h);
        let se = (-1.0 / d2).sqrt();
        let k = x.len();
        let lr = 2.0 * (ll - ll0);
        let plr = dist::chi2tail(k as f64, lr);
        let t: Vec<f64> = y.iter().map(|&v| bc(v, theta)).collect();
        let (b, rss) = ols(&t, &x);
        let sigma = (rss / n).sqrt();
        let level = util::level(cmd, st)?;
        let mut out = vec![String::new(), "Fitting comparison model".to_string(), String::new(), format!("Constant-only model:  Log likelihood = {}", fmt_g(ll0, 10)), String::new(), "Fitting full model".to_string(), String::new()];
        out.push(format!("{:<50}{:<15} = {:>10}", "", "Number of obs", util::commas(n as usize)));
        out.push(format!("{:<50}{:<15} = {:>10.2}", "", format!("LR chi2({k})"), lr));
        out.push(format!("{:<50}{:<15} = {:>10.3}", format!("Log likelihood = {}", fmt_g(ll, 10)), "Prob > chi2", plr));
        out.push(String::new());
        let names = vec!["/theta".to_string()];
        let vm = Matrix::from_rows(&[vec![se * se]]);
        out.extend(crate::est::coef_table(&crate::est::TableSpec { depvar: &vars[0], names: &names, b: &[theta], v: &vm, omitted: &[false], is_base: &[false], df: None, level, eform: None, coef_title: "Coefficient" }));
        out.push(String::new());
        out.push("Estimates of scale-variant parameters".into());
        out.push("-".repeat(28));
        out.push("             |  Coefficient".into());
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(14)));
        out.push(format!("{:<12} |", "Notrans"));
        for (j, v) in vars[1..].iter().enumerate() {
            out.push(format!("{:>12} |  {:>11}", abbrev(v, 12), g9(b[j])));
        }
        out.push(format!("{:>12} |  {:>11}", "_cons", g9(b[k])));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(14)));
        out.push(format!("{:>12} |  {:>11}", "/sigma", g9(sigma)));
        out.push("-".repeat(28));
        out.push(String::new());
        out.push("-".repeat(57));
        out.push("   Test         Restricted     LR statistic      P-value".into());
        out.push("    H0:       log likelihood       chi2       Prob > chi2".into());
        out.push("-".repeat(57));
        let mut res = Results::new();
        for (lab, v, key) in [("theta = -1", -1.0, "ll_tm1"), ("theta =  0", 0.0, "ll_t0"), ("theta =  1", 1.0, "ll_t1")] {
            let l = prof(v, &x);
            let c = 2.0 * (ll - l);
            out.push(format!("{:<12} {:>14}    {:>10.2}     {:>10.3}", lab, fmt_g(l, 10), c, dist::chi2tail(1.0, c)));
            res.set(key, l);
            res.set(&format!("chi2_{}", &key[3..]), c);
        }
        out.push("-".repeat(57));
        let mut bn: Vec<String> = vars[1..].iter().map(|v| format!("Notrans:{v}")).collect();
        bn.push("Notrans:_cons".into());
        bn.push("theta:_cons".into());
        let mut bv = b.clone();
        bv.push(theta);
        let kk = bv.len();
        let mut v = Matrix::zeros(kk, kk);
        v[(kk - 1, kk - 1)] = se * se;
        res.set("N", n);
        res.set("ll", ll);
        res.set("ll0", ll0);
        res.set("chi2", lr);
        res.set("df_m", k as f64);
        res.set("p", plr);
        res.set("rss", rss);
        res.set_str("cmd", "boxcox");
        res.set_str("depvar", vars[0].clone());
        res.set_str("model", "lhsonly");
        res.set_str("properties", "b V");
        res.set_mat("b", Matrix::row_vector(&bv).with_names(vec!["y1".into()], bn.clone()));
        res.set_mat("V", v.with_names(bn.clone(), bn));
        res.sample = Some(mask);
        st.e = res;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// lowess, lpoly
// ---------------------------------------------------------------------------

fn xy(ds: &DataSet, st: &Store, cmd: &Command) -> Result<(Vec<String>, Vec<usize>, Vec<f64>, Vec<f64>), String> {
    let vars = varlist::expand(ds, &cmd.main)?;
    if vars.len() != 2 {
        return Err(rc_err(198, "yvar xvar required"));
    }
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &vars);
    let rows = util::rows(&mask);
    let y = util::col(ds, &vars[0])?;
    let x = util::col(ds, &vars[1])?;
    Ok((vars, rows, y, x))
}

pub struct Lowess;

impl CommandHandler for Lowess {
    fn name(&self) -> &'static str {
        "lowess"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (vars, rows, y, x) = xy(ds, st, cmd)?;
        let bw = util::opt_num(cmd, "bwidth", 2)?.unwrap_or(0.8);
        let pts: Vec<(f64, f64)> = rows.iter().map(|&i| (x[i], y[i])).collect();
        let fit = crate::smooth::lowess(&pts, bw, cmd.has_opt("mean", 4), cmd.has_opt("noweight", 3));
        if let Some(name) = cmd.opt("generate", 3) {
            let mut vals = vec![SYSMISS; ds.n_obs];
            for &i in &rows {
                if let Some(p) = fit.iter().find(|p| p.0 == x[i]) {
                    vals[i] = p.1;
                }
            }
            add_var(ds, name.trim(), &vals, &format!("lowess: {}", vars[0]))?;
            st.data_changed = true;
        }
        let mut g = Graph {
            layers: vec![
                Layer { kind: LayerKind::Scatter, points: pts, label: vars[0].clone(), ..Default::default() },
                Layer { kind: LayerKind::Line, points: fit, label: "Lowess smoother".into(), ..Default::default() },
            ],
            title: "Lowess smoother".into(),
            xtitle: vars[1].clone(),
            ytitle: vars[0].clone(),
            ..Default::default()
        };
        let _ = &mut g;
        let mut r = Results::new();
        r.set("N", rows.len() as f64);
        r.set("bwidth", bw);
        st.r = r;
        show_graph(&mut g, cmd, st)
    }
}

pub struct Lpoly;

impl CommandHandler for Lpoly {
    fn name(&self) -> &'static str {
        "lpoly"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (vars, rows, y, x) = xy(ds, st, cmd)?;
        let kern = crate::kdensity::canonical_kernel(cmd.opt("kernel", 1).map(|s| s.trim()).unwrap_or("epanechnikov"));
        let degree = util::opt_num(cmd, "degree", 3)?.unwrap_or(0.0) as usize;
        let xs: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
        let bw = util::opt_num(cmd, "bwidth", 2)?.unwrap_or_else(|| crate::smooth::rot_bandwidth(&xs));
        let ngrid = util::opt_num(cmd, "n", 1)?.unwrap_or(50f64.min(rows.len() as f64)) as usize;
        let (lo, hi) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let grid: Vec<f64> = match cmd.opt("at", 2) {
            Some(v) => {
                let c = util::col(ds, &varlist::resolve_name(ds, v.trim())?)?;
                c.into_iter().filter(|v| !is_missing(*v)).collect()
            }
            None => (0..ngrid).map(|i| lo + (hi - lo) * i as f64 / (ngrid.max(2) - 1) as f64).collect(),
        };
        let pts: Vec<(f64, f64)> = rows.iter().map(|&i| (x[i], y[i])).collect();
        let fit = crate::smooth::lpoly(&pts, &grid, kern, degree, bw);
        if let Some(names) = cmd.opt("generate", 3) {
            let nm: Vec<&str> = names.split_whitespace().collect();
            let (gx, gy) = match nm.as_slice() {
                [a, b] => (Some(*a), *b),
                [b] => (None, *b),
                _ => return Err(rc_err(198, "generate() invalid")),
            };
            let n = ds.n_obs.max(fit.len());
            if fit.len() > ds.n_obs {
                crate::data::set_obs(ds, fit.len())?;
            }
            let mut vx = vec![SYSMISS; n];
            let mut vy = vec![SYSMISS; n];
            for (i, p) in fit.iter().enumerate() {
                vx[i] = p.0;
                vy[i] = p.1;
            }
            if let Some(a) = gx {
                add_var(ds, a, &vx, "Smoothing grid")?;
            }
            add_var(ds, gy, &vy, &format!("lpoly smooth: {}", vars[0]))?;
            st.data_changed = true;
        }
        let mut r = Results::new();
        r.set("N", rows.len() as f64);
        r.set("bwidth", bw);
        r.set("degree", degree as f64);
        r.set("ngrid", grid.len() as f64);
        r.set_str("kernel", kern);
        st.r = r;
        let mut layers = Vec::new();
        if !cmd.has_opt("noscatter", 5) {
            layers.push(Layer { kind: LayerKind::Scatter, points: pts, label: vars[0].clone(), ..Default::default() });
        }
        layers.push(Layer { kind: LayerKind::Line, points: fit.iter().map(|p| (p.0, p.1)).collect(), label: "lpoly smooth".into(), ..Default::default() });
        let mut g = Graph { layers, title: "Local polynomial smooth".into(), xtitle: vars[1].clone(), ytitle: vars[0].clone(), ..Default::default() };
        show_graph(&mut g, cmd, st)
    }
}

// ---------------------------------------------------------------------------
// dotplot, grmeanby, cusum
// ---------------------------------------------------------------------------

pub struct Dotplot;

impl CommandHandler for Dotplot {
    fn name(&self) -> &'static str {
        "dotplot"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let over = cmd.opt("over", 2).map(|o| varlist::resolve_name(ds, o.trim())).transpose()?;
        let ny = util::opt_num(cmd, "ny", 2)?.unwrap_or(35.0).max(2.0) as usize;
        // columns: one per variable, or one per level of over()
        let mut cols: Vec<(String, Vec<f64>)> = Vec::new();
        if let Some(o) = &over {
            util::markout(ds, &mut mask, &[o.clone()]);
            let y = util::col(ds, &vars[0])?;
            let g = util::col(ds, o)?;
            let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(y[i])).map(|i| g[i]).collect();
            lv.sort_by(f64::total_cmp);
            lv.dedup();
            for l in lv {
                let lab = ds.get_var(o).and_then(|v| ds.value_labels.get(&v.value_label_name)).and_then(|vl| vl.lookup(l).map(String::from)).unwrap_or_else(|| fmt_g(l, 12));
                cols.push((lab, (0..ds.n_obs).filter(|&i| mask[i] && g[i] == l && !is_missing(y[i])).map(|i| y[i]).collect()));
            }
        } else {
            for v in &vars {
                let y = util::col(ds, v)?;
                cols.push((v.clone(), (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(y[i])).map(|i| y[i]).collect()));
            }
        }
        let all: Vec<f64> = cols.iter().flat_map(|c| c.1.iter().copied()).collect();
        if all.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let (lo, hi) = (all.iter().cloned().fold(f64::INFINITY, f64::min), all.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let width = if hi > lo { (hi - lo) / ny as f64 } else { 1.0 };
        let mut pts = Vec::new();
        let mut xl = Vec::new();
        for (ci, (lab, ys)) in cols.iter().enumerate() {
            let mut bins: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for &v in ys {
                bins.entry((((v - lo) / width) as usize).min(ny - 1)).or_default().push(v);
            }
            let maxc = bins.values().map(|b| b.len()).max().unwrap_or(1) as f64;
            for (b, vals) in &bins {
                let m = vals.len() as f64;
                let mid = lo + (*b as f64 + 0.5) * width;
                for k in 0..vals.len() {
                    let off = (k as f64 - (m - 1.0) / 2.0) * 0.8 / maxc.max(1.0);
                    pts.push((ci as f64 + 1.0 + off, mid));
                }
            }
            xl.push((ci as f64 + 1.0, lab.clone()));
        }
        let mut g = Graph { layers: vec![Layer { kind: LayerKind::Scatter, points: pts, label: String::new(), ..Default::default() }], xlabels: xl, ytitle: vars[0].clone(), ..Default::default() };
        show_graph(&mut g, cmd, st)
    }
}

pub struct Grmeanby;

impl CommandHandler for Grmeanby {
    fn name(&self) -> &'static str {
        "grmeanby"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let yv = varlist::resolve_name(ds, cmd.opt("summarize", 3).ok_or_else(|| rc_err(198, "option summarize() required"))?.trim())?;
        let median = cmd.has_opt("median", 3);
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = vars.clone();
        all.push(yv.clone());
        util::markout(ds, &mut mask, &all);
        let y = util::col(ds, &yv)?;
        let stat = |v: &mut Vec<f64>| -> f64 {
            if median {
                v.sort_by(f64::total_cmp);
                util::pctile_sorted(v, 50.0)
            } else {
                util::mean(v)
            }
        };
        let mut overall: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| y[i]).collect();
        let om = stat(&mut overall);
        let mut layers = Vec::new();
        let mut xl = Vec::new();
        for (j, v) in vars.iter().enumerate() {
            let c = util::col(ds, v)?;
            let mut groups: BTreeMap<u64, (f64, Vec<f64>)> = BTreeMap::new();
            for i in 0..ds.n_obs {
                if mask[i] {
                    let key = { let b = c[i].to_bits(); if c[i] >= 0.0 { b ^ (1 << 63) } else { !b } };
                    groups.entry(key).or_insert((c[i], Vec::new())).1.push(y[i]);
                }
            }
            let mut pts = Vec::new();
            let mut labels = Vec::new();
            for (l, ys) in groups.values_mut() {
                pts.push((j as f64 + 1.0, stat(ys)));
                labels.push(fmt_g(*l, 12));
            }
            layers.push(Layer { kind: LayerKind::Connected, points: pts, mlabels: labels, label: v.clone(), ..Default::default() });
            xl.push((j as f64 + 1.0, v.clone()));
        }
        let n = vars.len() as f64;
        layers.push(Layer { kind: LayerKind::Line, points: vec![(0.5, om), (n + 0.5, om)], label: "Overall".into(), ..Default::default() });
        let mut g = Graph {
            layers,
            xlabels: xl,
            title: format!("{} of {}, by categories of the variables", if median { "Medians" } else { "Means" }, yv),
            ytitle: yv.clone(),
            ..Default::default()
        };
        show_graph(&mut g, cmd, st)
    }
}

pub struct Cusum;

impl CommandHandler for Cusum {
    fn name(&self) -> &'static str {
        "cusum"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (vars, rows, y, x) = xy(ds, st, cmd)?;
        let mut r = rows.clone();
        r.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
        let ybar = rows.iter().map(|&i| y[i]).sum::<f64>() / rows.len() as f64;
        let mut cum = 0.0;
        let mut vals = vec![SYSMISS; ds.n_obs];
        let mut pts = Vec::new();
        for &i in &r {
            cum += y[i] - ybar;
            vals[i] = cum;
            pts.push((x[i], cum));
        }
        if let Some(name) = cmd.opt("generate", 1) {
            add_var(ds, name.trim(), &vals, &format!("Cusum of {}", vars[0]))?;
            st.data_changed = true;
        }
        let mut g = Graph { layers: vec![Layer { kind: LayerKind::Line, points: pts, label: format!("Cusum ({})", vars[0]), ..Default::default() }], xtitle: vars[1].clone(), ytitle: format!("Cusum ({})", vars[0]), ylines: vec![0.0], ..Default::default() };
        show_graph(&mut g, cmd, st)
    }
}

// ---------------------------------------------------------------------------
// bsample, bstat
// ---------------------------------------------------------------------------

pub struct Bsample;

impl CommandHandler for Bsample {
    fn name(&self) -> &'static str {
        "bsample"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let rows = util::rows(&mask);
        let cluster: Vec<String> = match cmd.opt("cluster", 2) {
            Some(c) => varlist::expand(ds, c)?,
            None => Vec::new(),
        };
        let strata: Vec<String> = match cmd.opt("strata", 6) {
            Some(c) => varlist::expand(ds, c)?,
            None => Vec::new(),
        };
        let key = |vars: &[String], i: usize| -> String { vars.iter().map(|v| { let var = ds.get_var(v).unwrap(); if var.var_type.is_string() { var.str_at(i).to_string() } else { format!("{:?}", var.num(i)) } }).collect::<Vec<_>>().join("\u{1}") };
        // units within strata
        let mut strat: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut units: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
        let mut order: Vec<(String, String)> = Vec::new();
        for &i in &rows {
            let s = key(&strata, i);
            let u = if cluster.is_empty() { i.to_string() } else { key(&cluster, i) };
            let k = (s.clone(), u.clone());
            if !units.contains_key(&k) {
                strat.entry(s).or_default().push(u);
                order.push(k.clone());
            }
            units.entry(k).or_default().push(i);
        }
        let size = match cmd.main.trim() {
            "" => None,
            t => Some(crate::util::opt_num(&ferrostat_parser::parser::parse_line(&format!("x, n({t})")).map_err(|e| rc_err(198, e.message))?, "n", 1)?.unwrap_or(0.0) as usize),
        };
        let wvar = cmd.opt("weight", 2).map(|s| s.trim().to_string());
        let mut picks: Vec<(usize, f64)> = Vec::new(); // (row, new cluster id)
        let mut counts: BTreeMap<usize, f64> = BTreeMap::new();
        let mut next = 0.0;
        {
            let mut rng = st.rng.borrow_mut();
            for (s, us) in &strat {
                let m = us.len();
                let draws = size.unwrap_or(m);
                if draws > m && !cluster.is_empty() {
                    return Err(rc_err(498, "resample size must not be greater than number of clusters"));
                }
                for _ in 0..draws {
                    let j = ((rng.uniform() * m as f64) as usize).min(m - 1);
                    next += 1.0;
                    for &r in &units[&(s.clone(), us[j].clone())] {
                        picks.push((r, next));
                        *counts.entry(r).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
        if let Some(wv) = wvar {
            // frequency weights instead of replacing the data
            let v = ds.get_var_mut(&wv).ok_or_else(|| rc_err(111, format!("variable {wv} not found")))?;
            v.data = (0..v.data.len()).map(|i| util::typed_value(v.var_type, counts.get(&i).copied().unwrap_or(0.0))).collect();
            st.data_changed = true;
            return Ok(String::new());
        }
        let mut nd = ds.clone();
        for v in nd.vars.iter_mut() {
            let src = v.data.clone();
            v.data = picks.iter().map(|&(r, _)| src[r].clone()).collect();
        }
        nd.n_obs = picks.len();
        nd.sort_by.clear();
        if let Some(id) = cmd.opt("idcluster", 3) {
            let id = id.trim();
            let _ = nd.drop_variable(id);
            let mut v = Variable::new(id.to_string(), VarType::Long);
            v.data = picks.iter().map(|&(_, c)| util::typed_value(VarType::Long, c)).collect();
            nd.add_variable(v)?;
        }
        *ds = nd;
        st.data_changed = true;
        Ok(String::new())
    }
}

pub struct Bstat;

impl CommandHandler for Bstat {
    fn name(&self) -> &'static str {
        "bstat"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.vars.iter().map(|v| v.name.clone()).collect() } else { varlist::expand(ds, &cmd.main)? };
        let level = util::level(cmd, st)?;
        let k = vars.len();
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let observed: Vec<f64> = match cmd.opt("stat", 2) {
            Some(m) => {
                let m = m.trim();
                let mat = st.matrices.get(m).cloned().ok_or_else(|| rc_err(111, format!("matrix {m} not found")))?;
                if mat.data.len() != k {
                    return Err(rc_err(503, "stat() must have one column per variable"));
                }
                mat.data
            }
            None => cols.iter().map(|c| util::mean(&c.iter().copied().filter(|x| !is_missing(*x)).collect::<Vec<_>>())).collect(),
        };
        let complete: Vec<usize> = (0..ds.n_obs).filter(|&i| cols.iter().all(|c| !is_missing(c[i]))).collect();
        let nr = complete.len() as f64;
        let mean: Vec<f64> = cols.iter().map(|c| complete.iter().map(|&i| c[i]).sum::<f64>() / nr).collect();
        let mut v = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                v[(a, b)] = complete.iter().map(|&i| (cols[a][i] - mean[a]) * (cols[b][i] - mean[b])).sum::<f64>() / (nr - 1.0);
            }
        }
        let names: Vec<String> = vars.iter().map(|v| ds.get_var(v).map(|x| x.name.clone()).unwrap()).collect();
        let n = util::opt_num(cmd, "n", 1)?;
        let mut out = vec![String::new(), format!("{:<51}{:<14}= {:>10}", "Bootstrap results", "Number of obs", n.map_or(".".into(), |x| util::commas(x as usize)))];
        out.push(format!("{:<51}{:<14}= {:>10}", "", "Replications", util::commas(nr as usize)));
        out.push(String::new());
        let no = vec![false; k];
        let mut t = crate::est::coef_table(&crate::est::TableSpec { depvar: "", names: &names, b: &observed, v: &v, omitted: &no, is_base: &no, df: None, level, eform: None, coef_title: "Coefficient" });
        if t.len() > 1 {
            t[1] = t[1].replacen("Std. err.", "std. err.", 1).replacen("Coefficient", "coefficient", 1);
            t.insert(1, format!("{}|   Observed   Bootstrap                          Normal-based", " ".repeat(13)));
        }
        out.extend(t);
        let rowv = |x: &[f64]| Matrix::row_vector(x).with_names(vec!["y1".into()], names.clone());
        let mut e = Results::new();
        e.set_str("cmd", "bstat");
        e.set_str("prefix", "bootstrap");
        e.set_str("vce", "bootstrap");
        e.set_str("vcetype", "Bootstrap");
        e.set_str("properties", "b V");
        if let Some(n) = n {
            e.set("N", n);
        }
        e.set("N_reps", nr);
        e.set("level", level);
        e.set_mat("b", rowv(&observed));
        e.set_mat("V", v.clone().with_names(names.clone(), names.clone()));
        e.set_mat("b_bs", rowv(&mean));
        e.set_mat("bias", rowv(&mean.iter().zip(&observed).map(|(a, b)| a - b).collect::<Vec<_>>()));
        e.set_mat("se", rowv(&(0..k).map(|j| v[(j, j)].sqrt()).collect::<Vec<_>>()));
        st.e = e;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// dstdize, istdize
// ---------------------------------------------------------------------------

/// (strata key, label) of each observation.
fn strata_keys(ds: &DataSet, vars: &[String], i: usize) -> String {
    vars.iter().map(|v| { let var = ds.get_var(v).unwrap(); if var.var_type.is_string() { var.str_at(i).to_string() } else { util::cell_text(ds, var, i, true) } }).collect::<Vec<_>>().join(" ")
}

pub struct Dstdize;

impl CommandHandler for Dstdize {
    fn name(&self) -> &'static str {
        "dstdize"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 3 {
            return Err(rc_err(198, "dstdize charvar popvar stratavars, by(groupvars)"));
        }
        let (cv, pv, sv) = (&vars[0], &vars[1], &vars[2..]);
        let by = varlist::expand(ds, cmd.opt("by", 2).ok_or_else(|| rc_err(198, "option by() required"))?)?;
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = vars.clone();
        all.extend(by.iter().cloned());
        util::markout(ds, &mut mask, &all);
        let c = util::col(ds, cv)?;
        let p = util::col(ds, pv)?;
        // per group per stratum: (cases, population)
        let mut cells: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
        let mut std_pop: BTreeMap<String, f64> = BTreeMap::new();
        for i in util::rows(&mask) {
            let g = strata_keys(ds, &by, i);
            let s = strata_keys(ds, sv, i);
            let e = cells.entry(g).or_default().entry(s.clone()).or_insert((0.0, 0.0));
            e.0 += c[i];
            e.1 += p[i];
            *std_pop.entry(s).or_insert(0.0) += p[i];
        }
        let base_group = cmd.opt("base", 2).map(|s| util::unquote(s.trim()));
        if let Some(bg) = &base_group {
            let cell = cells.get(bg).ok_or_else(|| rc_err(111, format!("base({bg}) not found")))?;
            std_pop = cell.iter().map(|(s, v)| (s.clone(), v.1)).collect();
        }
        let tot: f64 = std_pop.values().sum();
        let alpha = 1.0 - level / 100.0;
        let z = dist::invnormal(1.0 - alpha / 2.0);
        let mut out = Vec::new();
        let mut summary = Vec::new();
        let mut adj_all = Vec::new();
        let mut crude_all = Vec::new();
        for (g, strata) in &cells {
            out.push(String::new());
            out.push("-".repeat(70));
            out.push(format!("-> {} = {g}", by.join(" ")));
            out.push(String::new());
            out.push(format!("{:>39}{:>11}{:>21}", "Pop.", "Stratum", "Std. Pop."));
            out.push(format!("{:>20} {:>10} {:>7} {:>9} {:>9} {:>9} {:>7}", "Stratum", "Pop.", "Cases", "Dist.", "Rate[s]", "Dst[P]", "s*P"));
            out.push("-".repeat(70));
            let (mut cases, mut pop, mut adj, mut var) = (0.0, 0.0, 0.0, 0.0);
            for (s, &(cs, ps)) in strata {
                let rate = cs / ps;
                let dst = std_pop.get(s).copied().unwrap_or(0.0) / tot;
                out.push(format!("{:>20} {:>10} {:>7} {:>9.4} {:>9.4} {:>9.4} {:>7.4}", abbrev(s, 20), g9(ps), g9(cs), ps / strata.values().map(|v| v.1).sum::<f64>(), rate, dst, rate * dst));
                cases += cs;
                pop += ps;
                adj += rate * dst;
                var += dst * dst * rate * (1.0 - rate) / ps;
            }
            out.push("-".repeat(70));
            let crude = cases / pop;
            let se = var.sqrt();
            out.push(format!("Totals: {:>13} {:>7}{:>28}{:>11.4}", g9(pop), g9(cases), "Adjusted rate =", adj));
            out.push(format!("{:>55}{:>11}", "Crude rate =", format!("{crude:.4}")));
            out.push(format!("{:>55}  [{:.4}, {:.4}]", format!("{}% conf. interval:", fmt_g(level, 4)), adj - z * se, adj + z * se));
            summary.push(format!("{:>20} | {:>10.4} {:>10.4}   {:>10.4} {:>10.4}", abbrev(g, 20), crude, adj, adj - z * se, adj + z * se));
            adj_all.push(adj);
            crude_all.push(crude);
        }
        out.push(String::new());
        out.push(format!("Summary of study populations:"));
        out.push(format!("{:>20} | {:>10} {:>10}   {:>21}", by.join(" "), "Crude", "Adj_Rate", format!("{}% Conf. Interval", fmt_g(level, 4))));
        out.push("-".repeat(70));
        out.extend(summary);
        let mut r = Results::new();
        r.set("k", cells.len() as f64);
        r.set_mat("adj", Matrix::row_vector(&adj_all));
        r.set_mat("crude", Matrix::row_vector(&crude_all));
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Istdize;

impl CommandHandler for Istdize {
    fn name(&self) -> &'static str {
        "istdize"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // istdize casevar_s popvar_s stratavars using file, popvars(casevar_p popvar_p) [by()]
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 3 {
            return Err(rc_err(198, "istdize casevar popvar stratavars using filename, popvars(casevar popvar)"));
        }
        let file = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let file = util::unquote(&file);
        let path = if file.ends_with(".dta") { file } else { format!("{file}.dta") };
        let bytes = ferrostat_core::vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
        let std = ferrostat_core::dta::DtaReader::from_bytes(bytes).map_err(|e| rc_err(610, e.to_string()))?;
        let pv: Vec<String> = cmd.opt("popvars", 4).ok_or_else(|| rc_err(198, "option popvars() required"))?.split_whitespace().map(String::from).collect();
        if pv.len() != 2 {
            return Err(rc_err(198, "popvars(casevar popvar)"));
        }
        let sv = &vars[2..];
        let mut rates: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        let (sc, sp) = (util::col(&std, &pv[0])?, util::col(&std, &pv[1])?);
        for i in 0..std.n_obs {
            let k = strata_keys(&std, sv, i);
            let e = rates.entry(k).or_insert((0.0, 0.0));
            e.0 += sc[i];
            e.1 += sp[i];
        }
        let by: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let (c, p) = (util::col(ds, &vars[0])?, util::col(ds, &vars[1])?);
        let mut groups: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new(); // observed, expected, pop
        for i in util::rows(&mask) {
            let g = strata_keys(ds, &by, i);
            let s = strata_keys(ds, sv, i);
            let (rc, rp) = rates.get(&s).copied().ok_or_else(|| rc_err(459, format!("stratum {s} not found in the standard population")))?;
            let e = groups.entry(g).or_insert((0.0, 0.0, 0.0));
            e.0 += c[i];
            e.1 += p[i] * rc / rp;
            e.2 += p[i];
        }
        let crude_std = rates.values().map(|v| v.0).sum::<f64>() / rates.values().map(|v| v.1).sum::<f64>();
        let alpha = 1.0 - level / 100.0;
        let mut out = Vec::new();
        let mut smr_all = Vec::new();
        for (g, &(obs, exp, pop)) in &groups {
            let smr = obs / exp;
            let (lo, hi) = (if obs > 0.0 { dist::invgammap(obs, alpha / 2.0) } else { 0.0 } / exp, dist::invgammap(obs + 1.0, 1.0 - alpha / 2.0) / exp);
            out.push(String::new());
            if !by.is_empty() {
                out.push(format!("-> {} = {g}", by.join(" ")));
            }
            out.push(format!("{:<46}{:>10}", "Observed cases:", g9(obs)));
            out.push(format!("{:<46}{:>10.4}", "SMR (Obs/Exp):", smr));
            out.push(format!("{:<46}[{:.4}, {:.4}]", format!("SMR exact {}% conf. interval:", fmt_g(level, 4)), lo, hi));
            out.push(format!("{:<46}{:>10.4}", "Crude rate:", obs / pop));
            out.push(format!("{:<46}{:>10.4}", "Adjusted rate:", smr * crude_std));
            out.push(format!("{:<46}[{:.4}, {:.4}]", format!("{}% conf. interval:", fmt_g(level, 4)), lo * crude_std, hi * crude_std));
            smr_all.push(smr);
        }
        let mut r = Results::new();
        r.set("k", groups.len() as f64);
        r.set_mat("smr", Matrix::row_vector(&smr_all));
        st.r = r;
        Ok(out.join("\n"))
    }
}
