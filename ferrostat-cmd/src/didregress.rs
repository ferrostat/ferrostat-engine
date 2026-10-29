//! `didregress` and `xtdidregress`: classic and two-way fixed-effects
//! difference-in-differences regressions, and their `estat ptrends`,
//! `estat granger` and `estat trendplots`.
//!
//! The ATET is the coefficient on the treatment indicator in a regression
//! with group (or panel) and time fixed effects, with standard errors
//! clustered at the group level. `wildbootstrap` adds a wild cluster
//! bootstrap (Rademacher or Webb weights, null imposed) for the ATET.

use std::collections::BTreeMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::did_imputation::{add_temps, drop_temps, run_line};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

pub struct Didregress {
    pub xt: bool,
}

fn col(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let v = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if v.var_type.is_string() {
        return Err(rc_err(109, format!("{name}: string variables not allowed")));
    }
    Ok(v.data.iter().map(|x| x.num()).collect())
}

/// Split `(y x1 x2) (treat[, continuous])`.
fn parse_eqs(main: &str) -> Result<(String, String), String> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in main.chars() {
        match c {
            '(' => {
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    parts.push(cur.trim().to_string());
                    cur.clear();
                } else {
                    cur.push(c);
                }
            }
            _ if depth > 0 => cur.push(c),
            _ if c.is_whitespace() => {}
            _ => return Err(rc_err(198, "syntax is (ovar [omvarlist]) (tvar)")),
        }
    }
    if parts.len() != 2 {
        return Err(rc_err(198, "syntax is (ovar [omvarlist]) (tvar)"));
    }
    Ok((parts[0].clone(), parts[1].clone()))
}

/// Value label text of `v` for variable `var` (or the number).
fn vlabel(ds: &DataSet, var: &str, v: f64) -> String {
    if let Some(x) = ds.get_var(var) {
        if let Some(t) = ds.value_labels.get(&x.value_label_name).and_then(|vl| vl.lookup(v)) {
            return t.to_string();
        }
    }
    crate::format::fmt_g(v, 12).trim().to_string()
}

impl CommandHandler for Didregress {
    fn name(&self) -> &'static str {
        if self.xt { "xtdidregress" } else { "didregress" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, verb);
        }
        let (oeq, teq) = parse_eqs(&cmd.main)?;
        let ovars = crate::varlist::expand(ds, &oeq)?;
        let (y, xs) = ovars.split_first().ok_or_else(|| rc_err(102, "outcome variable required"))?;
        let teq_var = teq.split(',').next().unwrap_or("").trim().to_string();
        let tvar = crate::varlist::resolve_name(ds, &teq_var)?;
        let group = match cmd.opt("group", 2) {
            Some(g) => crate::varlist::expand(ds, g)?,
            None => return Err(rc_err(198, "option group() required")),
        };
        let time = match cmd.opt("time", 2) {
            Some(t) => Some(crate::varlist::resolve_name(ds, t.trim())?),
            None => match (&st.tsset, self.xt) {
                (Some((_, t)), true) => Some(t.clone()),
                _ => None,
            },
        };
        let time = time.ok_or_else(|| rc_err(198, "option time() required"))?;
        let panel = if self.xt {
            match &st.tsset {
                Some((Some(p), _)) => Some(p.clone()),
                _ => return Err(rc_err(459, "panel variable not set; use xtset")),
            }
        } else {
            None
        };
        let n = ds.n_obs;
        let mut touse = util::touse(ds, st, cmd)?;
        let mut need: Vec<String> = ovars.clone();
        need.push(tvar.clone());
        need.extend(group.iter().cloned());
        need.push(time.clone());
        if let Some(p) = &panel {
            need.push(p.clone());
        }
        util::markout(ds, &mut touse, &need);
        let tr = col(ds, &tvar)?;
        let tcol = col(ds, &time)?;
        let gcols: Vec<Vec<f64>> = group.iter().map(|g| col(ds, g)).collect::<Result<_, _>>()?;
        let gkey = |i: usize| -> Vec<i64> { gcols.iter().map(|c| (c[i] * 1e6).round() as i64).collect() };
        // groups: treated if ever treated; first treatment time
        let mut ginfo: BTreeMap<Vec<i64>, (bool, f64, f64)> = BTreeMap::new();
        for i in 0..n {
            if !touse[i] {
                continue;
            }
            let e = ginfo.entry(gkey(i)).or_insert((false, f64::INFINITY, f64::INFINITY));
            e.2 = e.2.min(tcol[i]);
            if tr[i] != 0.0 {
                e.0 = true;
                e.1 = e.1.min(tcol[i]);
            }
        }
        let ntreated = ginfo.values().filter(|g| g.0).count();
        let ncontrol = ginfo.len() - ntreated;
        if ntreated == 0 || ncontrol == 0 {
            return Err(rc_err(459, "both treated and control groups are required"));
        }
        let (cmin, cmax) = ginfo.values().filter(|g| !g.0).fold((f64::INFINITY, f64::NEG_INFINITY), |a, g| (a.0.min(g.2), a.1.max(g.2)));
        let (tmin, tmax) = ginfo.values().filter(|g| g.0).fold((f64::INFINITY, f64::NEG_INFINITY), |a, g| (a.0.min(g.1), a.1.max(g.1)));
        // regression
        let absorb = match &panel {
            Some(p) => format!("{p} {time}"),
            None => format!("{} {time}", group.join("#")),
        };
        let gname = group.join("#");
        let sel = add_temps(ds, vec![touse.iter().map(|t| if *t { 1.0 } else { 0.0 }).collect()])?;
        let vce = cmd.opt("vce", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| format!("cluster {gname}"));
        let line = format!("reghdfe {y} {tvar} {} if {}==1, absorb({absorb}) vce({vce})", xs.join(" "), sel[0]);
        let r = run_line(ds, st, &crate::reghdfe::Reghdfe, &line);
        drop_temps(ds, &sel);
        r?;
        let eb = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(498, "regression failed"))?;
        let ev = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(498, "regression failed"))?;
        let j = eb.colnames.iter().position(|c| c == &tvar).ok_or_else(|| rc_err(498, "treatment omitted because of collinearity"))?;
        let atet = eb.data[j];
        let se = ev[(j, j)].sqrt();
        let df_r = st.e.get("df_r").unwrap_or(f64::INFINITY);
        let nobs = st.e.get("N").unwrap_or(0.0);
        let n_clust = st.e.get("N_clust");
        let clustvar = st.e.macros.get("clustvar").cloned();
        let level = util::level(cmd, st)?;
        // output
        let mut out = vec!["Number of groups and treatment time".to_string(), String::new()];
        out.push(format!("Time variable: {time}"));
        out.push(format!("Control:       {tvar} = 0"));
        out.push(format!("Treatment:     {tvar} = 1"));
        out.push("-".repeat(35));
        out.push(format!("{:>12} |   Control  Treatment", ""));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(21)));
        out.push(format!("{:<12} |", "Group"));
        out.push(format!("{:>12} | {:>9}  {:>9}", crate::summarize::abbrev(&gname, 12), ncontrol, ntreated));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(21)));
        out.push(format!("{:<12} |", "Time"));
        let g = |v: f64| crate::format::fmt_g(v, 9).trim().to_string();
        out.push(format!("{:>12} | {:>9}  {:>9}", "Minimum", g(cmin), g(tmin)));
        out.push(format!("{:>12} | {:>9}  {:>9}", "Maximum", g(cmax), g(tmax)));
        out.push("-".repeat(35));
        out.push(String::new());
        out.extend(crate::est::header("Difference-in-differences regression", &[("Number of obs".to_string(), util::commas(nobs as usize))]));
        out.push(format!("Data type: {}", if self.xt { "Longitudinal" } else { "Repeated cross-sectional" }));
        out.push(String::new());
        if let (Some(g), Some(c)) = (n_clust, &clustvar) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g as usize))));
        }
        let lab1 = vlabel(ds, &tvar, 1.0);
        let lab0 = vlabel(ds, &tvar, 0.0);
        let names = vec!["ATET:__did_atet__".to_string()];
        let mut v1 = Matrix::zeros(1, 1);
        v1[(0, 0)] = se * se;
        let table = crate::est::coef_table(&crate::est::TableSpec { depvar: y, names: &names, b: &[atet], v: &v1, omitted: &[false], is_base: &[false], df: Some(df_r), level, eform: None, coef_title: "Coefficient" });
        for l in table {
            if let Some(rest) = l.strip_prefix("__did_atet__") {
                out.push(format!("{:>12} |", crate::summarize::abbrev(&tvar, 12)));
                out.push(format!("{:>12}{}", format!("({lab1} vs {lab0})"), rest));
            } else {
                out.push(l.replace("| Coefficient  Std. err.", "| Coefficient  std. err."));
            }
        }
        let adj = if xs.is_empty() { "group effects and time effects" } else { "covariates, group effects, and time effects" };
        out.push(format!("Note: ATET estimate adjusted for {}.", if self.xt && xs.is_empty() { "panel effects and time effects" } else if self.xt { "covariates, panel effects, and time effects" } else { adj }));
        // wild cluster bootstrap
        let mut wb: Option<(f64, f64, f64, usize)> = None;
        if let Some(wopt) = cmd.opts.iter().find(|(n, _)| n.len() >= 5 && "wildbootstrap".starts_with(n.to_lowercase().as_str())).map(|(_, a)| a.clone().unwrap_or_default()) {
            wb = Some(self.wild(ds, st, y, xs, &tvar, &touse, &gcols, panel.as_deref(), &time, &wopt, level)?);
            let (lo, hi, p, reps) = wb.unwrap();
            out.push(format!("Wild-cluster bootstrap ({reps} replications): p-value = {:.4}; [{}% conf. interval] = [{}, {}]", p, crate::format::fmt_g(level, 4), crate::format::fmt_g(lo, 9).trim(), crate::format::fmt_g(hi, 9).trim()));
        }
        let text = out.join("\n");
        // e()
        let e = &mut st.e;
        let bname = vec![format!("ATET:r1vs0.{tvar}")];
        e.set_mat("b", Matrix::row_vector(&[atet]).with_names(vec!["y1".into()], bname.clone()));
        e.set_mat("V", v1.with_names(bname.clone(), bname));
        e.set_str("cmd", verb);
        e.set_str("cmdline", cmd.raw.trim().to_string());
        e.set_str("title", "Difference-in-differences regression");
        e.set_str("depvar", y.clone());
        e.set_str("treatment", tvar.clone());
        e.set_str("groupvars", group.join(" "));
        e.set_str("timevar", time.clone());
        e.set_str("covariates", xs.join(" "));
        e.set_str("datatype", if self.xt { "longitudinal" } else { "repeated cross-sectional" });
        e.set("N_treated_groups", ntreated as f64);
        e.set("N_control_groups", ncontrol as f64);
        if let Some((lo, hi, p, reps)) = wb {
            e.set("p_wild", p);
            e.set("ll_wild", lo);
            e.set("ul_wild", hi);
            e.set("reps", reps as f64);
        }
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

impl Didregress {
    /// Wild cluster bootstrap with the null imposed (WCR): p-value for
    /// H0: ATET = 0 and a percentile-t interval.
    #[allow(clippy::too_many_arguments)]
    fn wild(&self, ds: &mut DataSet, st: &mut Store, y: &str, xs: &[String], tvar: &str, touse: &[bool], gcols: &[Vec<f64>], panel: Option<&str>, time: &str, opts: &str, level: f64) -> Result<(f64, f64, f64, usize), String> {
        use crate::did_imputation::FeModel;
        let n = ds.n_obs;
        let mut reps = 999usize;
        let mut webb = false;
        for (nm, a) in ferrostat_parser::parser::split_options(opts) {
            match nm.to_lowercase().as_str() {
                "reps" => reps = a.unwrap_or_default().trim().parse().map_err(|_| rc_err(198, "invalid reps()"))?,
                "rseed" => {
                    let s: f64 = a.unwrap_or_default().trim().parse().map_err(|_| rc_err(198, "invalid rseed()"))?;
                    st.rng.borrow_mut().reseed(s as u64);
                }
                "errorweight" => webb = a.unwrap_or_default().trim().eq_ignore_ascii_case("webb"),
                _ => {}
            }
        }
        let rows: Vec<usize> = (0..n).filter(|&i| touse[i]).collect();
        let m = rows.len();
        let yv = col(ds, y)?;
        let trv = col(ds, tvar)?;
        let xv: Vec<Vec<f64>> = xs.iter().map(|x| col(ds, x)).collect::<Result<_, _>>()?;
        let key = |c: &[f64]| -> (Vec<usize>, usize) {
            let mut map: BTreeMap<i64, usize> = BTreeMap::new();
            let v: Vec<usize> = rows.iter().map(|&i| {
                let k = (c[i] * 1e6).round() as i64;
                let l = map.len();
                *map.entry(k).or_insert(l)
            }).collect();
            (v, map.len())
        };
        let gcode = {
            let mut map: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
            let v: Vec<usize> = rows.iter().map(|&i| {
                let k: Vec<i64> = gcols.iter().map(|c| (c[i] * 1e6).round() as i64).collect();
                let l = map.len();
                *map.entry(k).or_insert(l)
            }).collect();
            (v, map.len())
        };
        let fe1 = match panel {
            Some(p) => key(&col(ds, p)?),
            None => gcode.clone(),
        };
        let fe2 = key(&col(ds, time)?);
        let s = vec![true; m];
        let w = vec![1.0; m];
        let full = FeModel { fac: vec![fe1.clone(), fe2.clone()], ctl: xv.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect() };
        let with_t = FeModel { fac: vec![fe1, fe2], ctl: { let mut c = full.ctl.clone(); c.push(rows.iter().map(|&i| trv[i]).collect()); c } };
        let yr: Vec<f64> = rows.iter().map(|&i| yv[i]).collect();
        // residualised treatment (FWL) for the ATET
        let dr: Vec<f64> = rows.iter().map(|&i| trv[i]).collect();
        let fd = full.regress(&s, &w, &dr, 1e-13, 100000);
        let dt: Vec<f64> = (0..m).map(|i| dr[i] - fd.fit[i]).collect();
        let dd: f64 = dt.iter().map(|v| v * v).sum();
        // restricted fit (ATET = 0)
        let r0 = full.regress(&s, &w, &yr, 1e-13, 100000);
        let u0: Vec<f64> = (0..m).map(|i| yr[i] - r0.fit[i]).collect();
        let fit0 = r0.fit.clone();
        let ur = with_t.regress(&s, &w, &yr, 1e-13, 100000);
        let (gc, g) = gcode;
        let tstat = |yy: &[f64]| -> (f64, f64) {
            let b: f64 = (0..m).map(|i| dt[i] * yy[i]).sum::<f64>() / dd;
            let f = with_t.regress(&s, &w, yy, 1e-13, 100000);
            let mut sc = vec![0.0; g];
            for i in 0..m {
                sc[gc[i]] += dt[i] * (yy[i] - f.fit[i]);
            }
            let v = sc.iter().map(|x| x * x).sum::<f64>() / (dd * dd);
            (b, b / v.sqrt())
        };
        let _ = ur;
        let (b_hat, t_hat) = tstat(&yr);
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mult: Vec<f64> = {
                let mut rng = st.rng.borrow_mut();
                (0..g).map(|_| {
                    let u = rng.uniform();
                    if webb {
                        let k = (u * 6.0).floor() as usize;
                        [-(1.5f64).sqrt(), -1.0, -(0.5f64).sqrt(), (0.5f64).sqrt(), 1.0, (1.5f64).sqrt()][k.min(5)]
                    } else if u < 0.5 {
                        -1.0
                    } else {
                        1.0
                    }
                }).collect()
            };
            let yb: Vec<f64> = (0..m).map(|i| fit0[i] + u0[i] * mult[gc[i]]).collect();
            ts.push(tstat(&yb).1);
        }
        let p = ts.iter().filter(|t| t.abs() >= t_hat.abs()).count() as f64 / reps as f64;
        let mut abs: Vec<f64> = ts.iter().map(|t| t.abs()).collect();
        abs.sort_by(|a, b| a.total_cmp(b));
        let q = abs[((level / 100.0 * reps as f64).ceil() as usize).clamp(1, reps) - 1];
        let se = b_hat / t_hat;
        Ok((b_hat - q * se, b_hat + q * se, p, reps))
    }
}

/// estat ptrends | granger | trendplots after didregress / xtdidregress.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "ptrends" | "granger" | "trendplots" => Some(estat_impl(ds, st, sub, cmd)),
        _ => None,
    }
}

fn estat_impl(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let e = st.e.clone();
    let y = e.macros.get("depvar").cloned().unwrap_or_default();
    let tvar = e.macros.get("treatment").cloned().unwrap_or_default();
    let time = e.macros.get("timevar").cloned().unwrap_or_default();
    let group: Vec<String> = e.macros.get("groupvars").map(|g| g.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let xs = e.macros.get("covariates").cloned().unwrap_or_default();
    let sample = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let n = ds.n_obs;
    let tr = col(ds, &tvar)?;
    let tc = col(ds, &time)?;
    let yv = col(ds, &y)?;
    let gcols: Vec<Vec<f64>> = group.iter().map(|g| col(ds, g)).collect::<Result<_, _>>()?;
    let gkey = |i: usize| -> Vec<i64> { gcols.iter().map(|c| (c[i] * 1e6).round() as i64).collect() };
    let mut first: BTreeMap<Vec<i64>, f64> = BTreeMap::new();
    for i in 0..n {
        if sample[i] && tr[i] != 0.0 {
            let f = first.entry(gkey(i)).or_insert(f64::INFINITY);
            *f = f.min(tc[i]);
        }
    }
    let ttime: Vec<Option<f64>> = (0..n).map(|i| first.get(&gkey(i)).copied()).collect();
    if sub == "trendplots" {
        use crate::graph::{Graph, Layer, LayerKind};
        let mut by: [BTreeMap<i64, (f64, f64)>; 2] = [BTreeMap::new(), BTreeMap::new()];
        for i in 0..n {
            if sample[i] {
                let k = ttime[i].is_some() as usize;
                let e = by[k].entry((tc[i] * 1e6).round() as i64).or_insert((0.0, 0.0));
                e.0 += yv[i];
                e.1 += 1.0;
            }
        }
        let mut g = Graph::default();
        for (k, lab, colr) in [(0usize, "Control", (0u8, 114u8, 178u8)), (1, "Treatment", (213, 94, 0))] {
            let mut l = Layer { kind: LayerKind::Connected, color: Some(colr), label: lab.into(), ..Default::default() };
            l.points = by[k].iter().map(|(t, (s, c))| (*t as f64 / 1e6, s / c)).collect();
            g.layers.push(l);
        }
        g.legend = true;
        g.title = "Observed means".into();
        g.xtitle = time.clone();
        g.ytitle = y.clone();
        if let Some(t0) = first.values().copied().reduce(f64::min) {
            g.xlines = vec![t0];
        }
        return crate::graph::finish(&mut g, cmd, st);
    }
    // extra regressors on treated groups before treatment
    let mut extra: Vec<Vec<f64>> = Vec::new();
    if sub == "ptrends" {
        extra.push((0..n).map(|i| match ttime[i] { Some(t0) if tc[i] < t0 => tc[i], _ => 0.0 }).collect());
    } else {
        let mut leads: Vec<f64> = (0..n).filter_map(|i| ttime[i].filter(|t0| sample[i] && tc[i] < *t0).map(|t0| t0 - tc[i])).collect();
        leads.sort_by(|a, b| a.total_cmp(b));
        leads.dedup();
        // the earliest pre-treatment period is the reference
        leads.pop();
        for l in &leads {
            extra.push((0..n).map(|i| match ttime[i] { Some(t0) if t0 - tc[i] == *l => 1.0, _ => 0.0 }).collect());
        }
        if extra.is_empty() {
            return Err(rc_err(2000, "not enough pre-treatment periods for the test"));
        }
    }
    let k = extra.len();
    let mut cols = extra;
    cols.push(sample.iter().map(|s| if *s { 1.0 } else { 0.0 }).collect());
    let temps = add_temps(ds, cols)?;
    let absorb = if e.macros.get("cmd").map(String::as_str) == Some("xtdidregress") {
        match &st.tsset {
            Some((Some(p), _)) => format!("{p} {time}"),
            _ => format!("{} {time}", group.join("#")),
        }
    } else {
        format!("{} {time}", group.join("#"))
    };
    let line = format!("reghdfe {y} {tvar} {xs} {} if {}==1, absorb({absorb}) vce(cluster {})", temps[..k].join(" "), temps[k], group.join("#"));
    let saved = st.e.clone();
    let r = run_line(ds, st, &crate::reghdfe::Reghdfe, &line);
    let est = (st.e.matrices.get("b").cloned(), st.e.matrices.get("V").cloned(), st.e.get("df_r"));
    st.e = saved;
    drop_temps(ds, &temps);
    r?;
    let (Some(b), Some(v), Some(df)) = est else { return Err(rc_err(498, "test regression failed")) };
    let idx: Vec<usize> = temps[..k].iter().filter_map(|t| b.colnames.iter().position(|c| c == t)).collect();
    let q = idx.len();
    let mut vq = Matrix::zeros(q, q);
    for a in 0..q {
        for c in 0..q {
            vq[(a, c)] = v[(idx[a], idx[c])];
        }
    }
    let (vi, dr) = vq.sym_inverse(1e-12);
    let bq: Vec<f64> = idx.iter().map(|&j| b.data[j]).collect();
    let df1 = (q - dr.len()) as f64;
    if df1 == 0.0 {
        return Err(rc_err(2000, "not enough pre-treatment periods for the test"));
    }
    let f = (0..q).map(|a| (0..q).map(|c| bq[a] * vi[(a, c)] * bq[c]).sum::<f64>()).sum::<f64>() / df1;
    let p = crate::dist::ftail(df1, df, f);
    let mut out = Vec::new();
    if sub == "ptrends" {
        out.push("Parallel-trends test (pretreatment time period)".to_string());
        out.push("H0: Linear trends are parallel".to_string());
    } else {
        out.push("Granger causality test".to_string());
        out.push("H0: No effect in anticipation of treatment".to_string());
    }
    out.push(String::new());
    out.push(format!("F({}, {}) = {:>7.2}", df1, df, f));
    out.push(format!("Prob > F = {:>7.4}", p));
    let r = st.begin_r();
    r.set("F", f);
    r.set("p", p);
    r.set("df", df1);
    r.set("df_r", df);
    Ok(out.join("\n"))
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("didregress", Box::new(Didregress { xt: false })), ("xtdidregress", Box::new(Didregress { xt: true }))]
}
