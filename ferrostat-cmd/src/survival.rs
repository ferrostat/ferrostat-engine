//! Survival analysis: `stset`, `sts graph` (Kaplan-Meier survivor and
//! Nelson-Aalen cumulative hazard), `strate` (failure rates with exact
//! Poisson intervals) and `ltable` (actuarial life tables).
//!
//! `stset` records the analysis in the variables `_t` (exit time), `_t0`
//! (entry time), `_d` (failure indicator) and `_st` (in the analysis); the
//! other commands read them. References: Kalbfleisch & Prentice, *The
//! Statistical Analysis of Failure Time Data*; Collett, *Modelling Survival
//! Data in Medical Research*.

use ferrostat_core::missing::is_missing;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::store::Store;
use crate::util;
use crate::varlist;

fn set_var(ds: &mut DataSet, name: &str, vt: VarType, vals: &[f64]) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    let mut v = Variable::new(name, vt);
    v.data = vals.iter().map(|&x| if is_missing(x) { Value::Missing } else if vt == VarType::Byte { Value::Byte(x as i8) } else { Value::Double(x) }).collect();
    ds.add_variable(v)
}

pub struct Stset;

impl CommandHandler for Stset {
    fn name(&self) -> &'static str {
        "stset"
    }

    /// stset timevar [if] [, failure(var[==numlist])]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let tvar = varlist::resolve_name(ds, cmd.main.trim())?;
        let mask = util::touse(ds, st, cmd)?;
        let t = util::col(ds, &tvar)?;
        let (fvar, codes): (Option<String>, Vec<f64>) = match cmd.opt("failure", 1) {
            Some(f) => {
                let (v, list) = match f.split_once("==") {
                    Some((v, l)) => (v.trim().to_string(), l.split(|c: char| c == ',' || c.is_whitespace()).filter_map(|x| x.trim().parse().ok()).collect()),
                    None => (f.trim().to_string(), Vec::new()),
                };
                (Some(varlist::resolve_name(ds, &v)?), list)
            }
            None => (None, Vec::new()),
        };
        let fcol = fvar.as_ref().map(|v| util::col(ds, v)).transpose()?;
        let n = ds.n_obs;
        let mut stv = vec![0.0; n];
        let mut tv = vec![ferrostat_core::SYSMISS; n];
        let mut dv = vec![ferrostat_core::SYSMISS; n];
        for i in 0..n {
            if mask[i] && !is_missing(t[i]) && t[i] > 0.0 {
                stv[i] = 1.0;
                tv[i] = t[i];
                dv[i] = match &fcol {
                    None => 1.0,
                    Some(f) if is_missing(f[i]) => 0.0,
                    Some(f) if codes.is_empty() => (f[i] != 0.0) as u8 as f64,
                    Some(f) => codes.contains(&f[i]) as u8 as f64,
                };
            }
        }
        let t0: Vec<f64> = (0..n).map(|i| if stv[i] == 1.0 { 0.0 } else { ferrostat_core::SYSMISS }).collect();
        set_var(ds, "_st", VarType::Byte, &stv)?;
        set_var(ds, "_d", VarType::Byte, &dv)?;
        set_var(ds, "_t", VarType::Double, &tv)?;
        set_var(ds, "_t0", VarType::Double, &t0)?;
        ds.characteristics.entry("_dta".into()).or_default().insert("st_t".into(), tvar.clone());
        if let Some(f) = &fvar {
            ds.characteristics.entry("_dta".into()).or_default().insert("st_d".into(), f.clone());
        }
        let used = stv.iter().filter(|&&s| s == 1.0).count();
        let failures = dv.iter().filter(|&&d| d == 1.0).count();
        let risk: f64 = tv.iter().filter(|x| !is_missing(**x)).sum();
        let fdesc = match &fvar {
            None => "(assumed to fail at time=_t)".to_string(),
            Some(f) if codes.is_empty() => format!("{f} != 0 & {f} < ."),
            Some(f) => format!("{f} == {}", codes.iter().map(|c| fmt_g(*c, 9)).collect::<Vec<_>>().join(" ")),
        };
        Ok(format!(
            "Survival-time data settings\n\n         Failure event: {fdesc}\nObserved time interval: (0, {tvar}]\n     Exit on or before: failure\n\n{:>10}  total observations\n{:>10}  observations excluded\n{:>10}  observations remaining, representing\n{:>10}  failures in single-record/single-failure data\n{:>10}  total analysis time at risk and under observation",
            util::commas(n),
            util::commas(n - used),
            util::commas(used),
            util::commas(failures),
            fmt_g(risk, 12)
        ))
    }
}

/// (t, d, row) of the observations in the st analysis and the command sample.
fn st_data(ds: &DataSet, st: &Store, cmd: &Command) -> Result<Vec<(f64, bool, usize)>, String> {
    if ds.get_var("_t").is_none() || ds.get_var("_d").is_none() {
        return Err(rc_err(119, "data not st; use stset"));
    }
    let mask = util::touse(ds, st, cmd)?;
    let (t, d, s) = (util::col(ds, "_t")?, util::col(ds, "_d")?, util::col(ds, "_st")?);
    Ok((0..ds.n_obs).filter(|&i| mask[i] && s[i] == 1.0 && !is_missing(t[i])).map(|i| (t[i], d[i] == 1.0, i)).collect())
}

/// Groups of a by() variable: (label, observation rows); one group when absent.
fn by_groups(ds: &DataSet, var: Option<&str>, data: &[(f64, bool, usize)]) -> Result<Vec<(String, Vec<(f64, bool, usize)>)>, String> {
    let Some(v) = var else { return Ok(vec![(String::new(), data.to_vec())]) };
    let v = varlist::resolve_name(ds, v.trim())?;
    let c = util::col(ds, &v)?;
    let var = ds.get_var(&v).unwrap();
    let mut levels: Vec<f64> = data.iter().map(|r| c[r.2]).filter(|x| !is_missing(*x)).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    Ok(levels
        .into_iter()
        .map(|lv| {
            let rows: Vec<(f64, bool, usize)> = data.iter().filter(|r| c[r.2] == lv).cloned().collect();
            (format!("{v} = {}", util::cell_text(ds, var, rows[0].2, true)), rows)
        })
        .collect())
}

/// Risk-set table: (time, at risk, failures) at each distinct failure time.
fn risk_table(rows: &[(f64, bool, usize)]) -> Vec<(f64, f64, f64)> {
    let mut times: Vec<f64> = rows.iter().filter(|r| r.1).map(|r| r.0).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times.into_iter().map(|tj| (tj, rows.iter().filter(|r| r.0 >= tj).count() as f64, rows.iter().filter(|r| r.0 == tj && r.1).count() as f64)).collect()
}

pub struct Sts;

impl CommandHandler for Sts {
    fn name(&self) -> &'static str {
        "sts"
    }

    /// sts graph [, by(var) na atrisk]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let sub = cmd.main.split_whitespace().next().unwrap_or("graph");
        if !matches!(sub, "graph" | "gr") {
            return Err(rc_err(198, format!("sts {sub} not supported; use sts graph")));
        }
        let data = st_data(ds, st, cmd)?;
        let na = cmd.has_opt("na", 2) || cmd.has_opt("cumhaz", 4);
        let groups = by_groups(ds, cmd.opt("by", 2).as_deref(), &data)?;
        let mut graph = Graph::default();
        let tmax = data.iter().map(|r| r.0).fold(0.0, f64::max);
        for (label, rows) in &groups {
            // step function starting at 0
            let mut level = if na { 0.0 } else { 1.0 };
            let mut pts = vec![(0.0, level)];
            for (tj, nj, dj) in risk_table(rows) {
                pts.push((tj, level));
                level = if na { level + dj / nj } else { level * (1.0 - dj / nj) };
                pts.push((tj, level));
            }
            let last = rows.iter().map(|r| r.0).fold(0.0, f64::max);
            pts.push((last, level));
            graph.layers.push(Layer { kind: LayerKind::Line, points: pts, label: if label.is_empty() { (if na { "Nelson-Aalen cumulative hazard" } else { "Kaplan-Meier survival" }).into() } else { label.clone() }, ..Default::default() });
            if cmd.has_opt("atrisk", 3) {
                // numbers at risk at six evenly spaced times, written just above the axis
                let mut marks = Layer { kind: LayerKind::Scatter, label: String::new(), hollow: true, ..Default::default() };
                for k in 0..6 {
                    let tk = tmax * k as f64 / 5.0;
                    marks.points.push((tk, if na { 0.0 } else { 0.02 }));
                    marks.sizes.push(0.0);
                    marks.mlabels.push(rows.iter().filter(|r| r.0 >= tk).count().to_string());
                }
                graph.layers.push(marks);
            }
        }
        graph.title = if na { "Nelson-Aalen cumulative hazard estimate".into() } else { "Kaplan-Meier survival estimate".into() };
        graph.xtitle = "Analysis time".into();
        graph.legend = groups.len() > 1;
        if cmd.has_opt("atrisk", 3) {
            // the number-at-risk labels are not legend entries
            graph.legend_order = graph.layers.iter().enumerate().filter(|(_, l)| l.kind == LayerKind::Line).map(|(k, _)| (k + 1, String::new())).collect();
            if groups.len() == 1 {
                graph.legend = false;
            }
        }
        let fail = ds.characteristics.get("_dta").and_then(|c| c.get("st_d")).cloned().unwrap_or_else(|| "_d".into());
        let time = ds.characteristics.get("_dta").and_then(|c| c.get("st_t")).cloned().unwrap_or_else(|| "_t".into());
        let extra = graph::finish(&mut graph, cmd, st)?;
        Ok([format!("        Failure _d: {fail}\n  Analysis time _t: {time}"), extra].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"))
    }
}

/// Exact Poisson confidence interval for a rate with `d` events in `y` time units.
fn poisson_ci(d: f64, y: f64, level: f64) -> (f64, f64) {
    let a = (1.0 - level / 100.0) / 2.0;
    let lo = if d > 0.0 { dist::invchi2(2.0 * d, a) / 2.0 } else { 0.0 };
    let hi = dist::invchi2(2.0 * (d + 1.0), 1.0 - a) / 2.0;
    (lo / y, hi / y)
}

pub struct Strate;

impl CommandHandler for Strate {
    fn name(&self) -> &'static str {
        "strate"
    }

    /// strate [var] [, graph nowhisker recast() ciopts()]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let data = st_data(ds, st, cmd)?;
        let var = cmd.main.split_whitespace().next();
        let groups = by_groups(ds, var, &data)?;
        let level = util::level(cmd, st)?;
        let t0 = util::col(ds, "_t0")?;
        let name = var.map(|v| varlist::resolve_name(ds, v)).transpose()?.unwrap_or_default();
        let mut out = vec![
            "Estimated failure rates".to_string(),
            format!("Number of records = {}", util::commas(data.len())),
            String::new(),
            format!("  {:>10} {:>6} {:>10} {:>10} {:>10} {:>10}", if name.is_empty() { "" } else { name.as_str() }, "D", "Y", "Rate", "Lower", "Upper"),
        ];
        let mut rates = Layer { kind: LayerKind::Scatter, label: "Rate".into(), ..Default::default() };
        let mut cis = Layer { kind: LayerKind::Range(RangeStyle::Spike), label: format!("{}% CI", fmt_g(level, 4)), ..Default::default() };
        let mut labels = Vec::new();
        for (k, (label, rows)) in groups.iter().enumerate() {
            let d = rows.iter().filter(|r| r.1).count() as f64;
            let y: f64 = rows.iter().map(|r| r.0 - t0[r.2]).sum();
            let rate = d / y;
            let (lo, hi) = poisson_ci(d, y, level);
            let lv = label.split(" = ").nth(1).unwrap_or("").to_string();
            out.push(format!("  {:>10} {:>6} {:>10} {:>10.6} {:>10.6} {:>10.6}", lv, d, fmt_g(y, 9), rate, lo, hi));
            let x = k as f64 + 1.0;
            rates.points.push((x, rate));
            cis.ranges.push((x, lo, hi));
            labels.push((x, if lv.is_empty() { "All".to_string() } else { lv }));
        }
        out.push(String::new());
        out.push(" Notes: Rate = D/Y = failures/person-time.".into());
        out.push(format!("        Lower and Upper are bounds of {}% confidence intervals.", fmt_g(level, 4)));
        if cmd.has_opt("graph", 2) {
            if let Some(r) = cmd.opt("recast", 3) {
                rates.kind = if r.trim().starts_with("conn") { LayerKind::Connected } else if r.trim() == "line" { LayerKind::Line } else { LayerKind::Scatter };
            }
            if cmd.opt("ciopts", 3).is_some_and(|c| c.contains("rcap")) {
                cis.kind = LayerKind::Range(RangeStyle::Cap);
            }
            let mut layers = Vec::new();
            if !cmd.has_opt("nowhisker", 3) {
                layers.push(cis);
            }
            layers.push(rates);
            let mut graph = Graph { layers, xlabels: labels, xtitle: name.clone(), ytitle: "Rate".into(), title: "Failure rates and their CIs".into(), ..Default::default() };
            graph.legend = false;
            let extra = graph::finish(&mut graph, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

pub struct Ltable;

impl CommandHandler for Ltable {
    fn name(&self) -> &'static str {
        "ltable"
    }

    /// ltable timevar [deadvar] [, by(var) survival graph noconf overlay intervals(#)]
    /// Actuarial estimator: those lost during an interval count as half
    /// exposed; standard errors by Greenwood's formula; confidence
    /// intervals on the log(-log) scale.
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let tvar = vars.first().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        let mask = util::touse(ds, st, cmd)?;
        let t = util::col(ds, tvar)?;
        let dcol = vars.get(1).map(|v| util::col(ds, v)).transpose()?;
        let data: Vec<(f64, bool, usize)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(t[i])).map(|i| (t[i], dcol.as_ref().map_or(true, |d| d[i] != 0.0 && !is_missing(d[i])), i)).collect();
        let width = util::opt_num(cmd, "intervals", 3)?.unwrap_or(1.0);
        let groups = by_groups(ds, cmd.opt("by", 2).as_deref(), &data)?;
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut out = vec![format!("           Interval     Total   Deaths   Lost   Survival   Std. err.   [{}% conf. int.]", fmt_g(level, 4))];
        let mut children = Vec::new();
        let mut overlay = Graph { xtitle: tvar.clone(), ytitle: "Proportion surviving".into(), legend: true, ..Default::default() };
        for (label, rows) in &groups {
            if !label.is_empty() {
                out.push(label.clone());
            }
            let tmax = rows.iter().map(|r| r.0).fold(0.0, f64::max);
            let nint = (tmax / width).floor() as usize + 1;
            let mut n = rows.len() as f64;
            let (mut s, mut gw) = (1.0f64, 0.0f64);
            let mut curve = vec![(0.0, 1.0)];
            let mut band = Vec::new();
            for j in 0..nint {
                let (a, b) = (j as f64 * width, (j + 1) as f64 * width);
                let inside: Vec<&(f64, bool, usize)> = rows.iter().filter(|r| r.0 >= a && r.0 < b).collect();
                if inside.is_empty() {
                    continue;
                }
                let d = inside.iter().filter(|r| r.1).count() as f64;
                let lost = inside.len() as f64 - d;
                let ne = n - lost / 2.0;
                let q = if ne > 0.0 { d / ne } else { 0.0 };
                s *= 1.0 - q;
                if ne > d {
                    gw += d / (ne * (ne - d));
                }
                let se = s * gw.sqrt();
                // log(-log) interval
                let (lo, hi) = if s > 0.0 && s < 1.0 {
                    let w = z * gw.sqrt() / s.ln().abs();
                    (s.powf((w).exp()), s.powf((-w).exp()))
                } else {
                    (f64::NAN, f64::NAN)
                };
                out.push(format!("  {:>8} {:>8} {:>8} {:>8} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>9.4}", fmt_g(a, 6), fmt_g(b, 6), n, d, lost, s, se, lo, hi));
                curve.push((b, s));
                if lo.is_finite() {
                    band.push((b, lo, hi));
                }
                n -= inside.len() as f64;
            }
            let line = Layer { kind: LayerKind::Line, points: curve, label: if label.is_empty() { "Survival".into() } else { label.clone() }, ..Default::default() };
            let ci = Layer { kind: LayerKind::Range(RangeStyle::Cap), ranges: band, label: format!("{}% CI", fmt_g(level, 4)), ..Default::default() };
            if cmd.has_opt("overlay", 4) {
                overlay.layers.push(line);
            } else {
                let mut g = Graph { xtitle: tvar.clone(), ytitle: "Proportion surviving".into(), title: label.clone(), ..Default::default() };
                if !cmd.has_opt("noconf", 4) {
                    g.layers.push(ci);
                }
                g.layers.push(line);
                children.push(g);
            }
        }
        if cmd.has_opt("graph", 2) {
            let mut graph = if cmd.has_opt("overlay", 4) || children.len() != 1 {
                if cmd.has_opt("overlay", 4) { overlay } else { Graph { cols: children.len().min(3), children, ..Default::default() } }
            } else {
                children.pop().unwrap()
            };
            let extra = graph::finish(&mut graph, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_poisson_interval() {
        // 10 events in 100 time units: the textbook exact 95% interval is (0.04795, 0.18390)
        let (lo, hi) = poisson_ci(10.0, 100.0, 95.0);
        assert!((lo - 0.047954).abs() < 1e-5, "{lo}");
        assert!((hi - 0.183904).abs() < 1e-5, "{hi}");
    }

    #[test]
    fn kaplan_meier_risk_table() {
        // times 1 2 2 3(censored) 4: at t=2 four at risk, two failures
        let rows = vec![(1.0, true, 0), (2.0, true, 1), (2.0, true, 2), (3.0, false, 3), (4.0, true, 4)];
        let rt = risk_table(&rows);
        assert_eq!(rt, vec![(1.0, 5.0, 1.0), (2.0, 4.0, 2.0), (4.0, 1.0, 1.0)]);
        let s: f64 = rt.iter().map(|(_, n, d)| 1.0 - d / n).product();
        assert_eq!(s, 0.0);
        let s2: f64 = rt[..2].iter().map(|(_, n, d)| 1.0 - d / n).product();
        assert!((s2 - 0.4).abs() < 1e-12);
    }
}
