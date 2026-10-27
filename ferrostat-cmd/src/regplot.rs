//! Regression diagnostic plots after a linear regression: residual-versus-
//! fitted (rvfplot), residual-versus-predictor (rvpplot), added-variable
//! (avplot, avplots) and leverage-versus-squared-residual (lvr2plot) plots.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::postest::{coef_value, Predict};
use crate::store::Store;
use crate::util;
use crate::varlist;

pub struct RegPlot;

fn scatter(points: Vec<(f64, f64)>) -> Layer {
    Layer { kind: LayerKind::Scatter, points, ..Default::default() }
}

fn label_of(ds: &DataSet, v: &str) -> String {
    ds.get_var(v).map(|x| if x.label.is_empty() { v.to_string() } else { x.label.clone() }).unwrap_or_else(|| v.to_string())
}

/// Predictions of the last regression (`predict ..., stat`) on the estimation sample.
fn predicted(ds: &DataSet, st: &mut Store, stat: &str) -> Result<Vec<f64>, String> {
    let mut tmp = ds.clone();
    let c = parse_line(&format!("predict double __regplot_tmp, {stat}")).map_err(|e| rc_err(198, e))?;
    Predict.execute(&mut tmp, st, &c)?;
    let mut v = util::col(&tmp, "__regplot_tmp")?;
    if let Some(s) = &st.e.sample {
        for (x, keep) in v.iter_mut().zip(s) {
            if !keep {
                *x = ferrostat_core::missing::SYSMISS;
            }
        }
    }
    Ok(v)
}

/// Residuals of a least-squares fit of `y` on the columns of `x` (rows already complete).
fn ols_resid(y: &[f64], x: &[Vec<f64>]) -> Vec<f64> {
    let k = x.len();
    let n = y.len();
    // normal equations [X'X | X'y], solved by Gauss-Jordan elimination with partial pivoting
    let mut a: Vec<Vec<f64>> = (0..k).map(|r| (0..=k).map(|c| (0..n).map(|i| x[r][i] * if c < k { x[c][i] } else { y[i] }).sum()).collect()).collect();
    let mut keep = vec![true; k];
    for col in 0..k {
        let piv = (col..k).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs())).unwrap();
        if a[piv][col].abs() < 1e-12 {
            keep[col] = false;
            continue;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for c in col..=k {
            a[col][c] /= d;
        }
        for r in 0..k {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for c in col..=k {
                        a[r][c] -= f * a[col][c];
                    }
                }
            }
        }
    }
    (0..n).map(|i| y[i] - (0..k).filter(|&c| keep[c]).map(|c| a[c][k] * x[c][i]).sum::<f64>()).collect()
}

/// Added-variable plot of regressor `var`: residuals of y and of `var`, each
/// regressed on the other regressors, with the fitted slope (the coefficient).
fn avplot(ds: &DataSet, st: &Store, var: &str) -> Result<Graph, String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names: Vec<String> = b.colnames.iter().filter(|n| !n.contains("o.") && *n != "_cons").cloned().collect();
    let pos = names.iter().position(|n| n == var).ok_or_else(|| rc_err(198, format!("{var} is not an independent variable in the last estimation")))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let env = Env::with_store(ds, st);
    let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let ycol = util::col(ds, &dep)?;
    let mut rows = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut y = Vec::new();
    for i in 0..ds.n_obs {
        if !sample[i] || is_missing(ycol[i]) {
            continue;
        }
        let vals: Vec<f64> = names.iter().map(|n| coef_value(&env, n, i)).collect::<Result<_, _>>()?;
        if vals.iter().any(|v| is_missing(*v)) {
            continue;
        }
        rows.push(i);
        y.push(ycol[i]);
        for (c, v) in cols.iter_mut().zip(vals) {
            c.push(v);
        }
    }
    if rows.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let target = cols.remove(pos);
    if b.col_index("_cons").is_some() {
        cols.push(vec![1.0; rows.len()]);
    }
    let ey = ols_resid(&y, &cols);
    let ex = ols_resid(&target, &cols);
    let slope = b.get(0, b.col_index(var).unwrap());
    let lo = ex.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ex.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fit = Layer { kind: LayerKind::Line, points: vec![(lo, slope * lo), (hi, slope * hi)], label: "Fitted values".into(), ..Default::default() };
    Ok(Graph {
        layers: vec![scatter(ex.into_iter().zip(ey).collect()), fit],
        xtitle: format!("e( {var} | X )"),
        ytitle: format!("e( {dep} | X )"),
        ..Default::default()
    })
}

impl CommandHandler for RegPlot {
    fn name(&self) -> &'static str {
        "rvfplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        if ecmd.is_empty() || !st.e.matrices.contains_key("b") {
            return Err(rc_err(301, "last estimates not found"));
        }
        if ecmd != "regress" {
            return Err(rc_err(301, format!("{} not allowed after {ecmd}", cmd.verb)));
        }
        let mut graph = match cmd.verb.as_str() {
            "rvfplot" | "rvpplot" => {
                let res = predicted(ds, st, "residuals")?;
                let (x, xtitle) = if cmd.verb == "rvfplot" {
                    (predicted(ds, st, "xb")?, "Fitted values".to_string())
                } else {
                    let v = varlist::expand(ds, &cmd.main)?;
                    if v.len() != 1 {
                        return Err(rc_err(if v.is_empty() { 102 } else { 103 }, if v.is_empty() { "too few variables specified" } else { "too many variables specified" }));
                    }
                    (util::col(ds, &v[0])?, label_of(ds, &v[0]))
                };
                let pts = (0..ds.n_obs).filter(|&i| !is_missing(res[i]) && !is_missing(x[i])).map(|i| (x[i], res[i])).collect();
                Graph { layers: vec![scatter(pts)], xtitle, ytitle: "Residuals".into(), ..Default::default() }
            }
            "lvr2plot" => {
                let res = predicted(ds, st, "residuals")?;
                let lev = predicted(ds, st, "leverage")?;
                let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(res[i]) && !is_missing(lev[i])).collect();
                let ss: f64 = rows.iter().map(|&i| res[i] * res[i]).sum();
                let pts: Vec<(f64, f64)> = rows.iter().map(|&i| (res[i] * res[i] / ss, lev[i])).collect();
                let n = pts.len().max(1) as f64;
                let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
                let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
                Graph { layers: vec![scatter(pts)], xtitle: "Normalized residual squared".into(), ytitle: "Leverage".into(), xlines: vec![mx], ylines: vec![my], ..Default::default() }
            }
            "avplot" => {
                let v = varlist::expand(ds, &cmd.main)?;
                if v.len() != 1 {
                    return Err(rc_err(if v.is_empty() { 102 } else { 103 }, if v.is_empty() { "too few variables specified" } else { "too many variables specified" }));
                }
                avplot(ds, st, &v[0])?
            }
            _ => {
                // avplots: one added-variable plot per regressor
                let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
                let names: Vec<String> = b.colnames.iter().filter(|n| !n.contains("o.") && *n != "_cons").cloned().collect();
                let children: Vec<Graph> = names.iter().map(|n| avplot(ds, st, n)).collect::<Result<_, _>>()?;
                let cols = (children.len() as f64).sqrt().ceil().max(1.0) as usize;
                Graph { children, cols, ..Default::default() }
            }
        };
        graph::finish(&mut graph, cmd, st)
    }
}
