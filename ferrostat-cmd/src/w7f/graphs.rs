//! `h2omlgraph`: variable importance (split gains and permutation),
//! partial dependence and ICE curves, SHAP summaries and values, ROC and
//! precision-recall curves, score histories and regression diagnostics.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::current;
use super::learn::{Kind, Model};
use super::metrics::{self, Eval};
use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::store::Store;
use crate::util;

fn hbar(g: &mut Graph, names: &[String], vals: &[f64], label: &str) {
    let k = names.len();
    let mut l = Layer { kind: LayerKind::Range(RangeStyle::Bar), label: label.into(), width: 0.7, horizontal: true, ..Default::default() };
    // first name at the top
    l.ranges = vals.iter().enumerate().map(|(i, v)| ((k - i) as f64, 0.0, *v)).collect();
    g.layers.push(l);
    g.ylabels = names.iter().enumerate().map(|(i, n)| ((k - i) as f64, n.clone())).collect();
}

/// Output (probability or value) used by pdp/ice: the positive class for
/// binary models, `class(#)` for multiclass ones.
fn out_index(m: &Model, cmd: &Command) -> Result<usize, String> {
    Ok(match m.kind {
        Kind::Reg => 0,
        Kind::Bin => 1,
        Kind::Multi => (util::opt_num(cmd, "class", 5)?.unwrap_or(1.0) as usize).clamp(1, m.levels.len()) - 1,
    })
}

fn pred_var(m: &Model, k: usize) -> String {
    if m.kind == Kind::Reg { format!("Predicted {}", m.depvar) } else { format!("Pr({} = {})", m.depvar, m.level_labels.get(k).cloned().unwrap_or_default()) }
}

fn grid_values(v: &[f64], n: usize) -> Vec<f64> {
    let mut u: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    u.sort_by(|a, b| a.total_cmp(b));
    u.dedup();
    if u.len() <= n {
        return u;
    }
    (0..n).map(|i| util::pctile_sorted(&u, 100.0 * i as f64 / (n - 1) as f64)).collect()
}

fn eval_for(m: &Model, cmd: &Command) -> Result<(String, Eval), String> {
    super::post::pick_eval(m, cmd)
}

pub fn graph(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (_, m) = current(st)?;
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let sub = sub.to_lowercase();
    let mut g = Graph::default();
    let p = m.enc.preds.len();
    let top = util::opt_num(cmd, "top", 3)?.map_or(p.min(20), |v| (v as usize).clamp(1, p));
    match sub.as_str() {
        "varimp" => {
            let vi = m.varimp();
            let vi = &vi[..top.min(vi.len())];
            let names: Vec<String> = vi.iter().map(|v| v.0.clone()).collect();
            let pct = cmd.has_opt("percent", 4);
            let vals: Vec<f64> = vi.iter().map(|v| if pct { 100.0 * v.3 } else { v.2 }).collect();
            hbar(&mut g, &names, &vals, "Importance");
            g.title = "Variable importance".into();
            g.xtitle = if pct { "Percentage".into() } else { "Scaled importance".into() };
            st.begin_r().set_mat("varimp", Matrix::col_vector(&vals).with_names(names, vec!["importance".into()]));
        }
        "permimp" => {
            let (x, rows, y) = super::post::active_xy(ds, st, &m)?;
            let metric = cmd.opt("metric", 6).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| if m.kind == Kind::Reg { "mse".into() } else { "logloss".into() });
            let hb = metrics::higher_better(&metric);
            let base = Eval::new(y.clone(), m.predict_cols(&x, rows.iter().cloned()), m.classes()).get(&metric);
            let seed = super::seed_opt(cmd, st)?;
            let mut rng = Rng::new(seed);
            let reps = util::opt_num(cmd, "repeats", 3)?.unwrap_or(1.0).max(1.0) as usize;
            let mut imp = vec![0.0; p];
            for j in 0..p {
                for _ in 0..reps {
                    let mut perm = rows.clone();
                    for i in (1..perm.len()).rev() {
                        let t = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
                        perm.swap(i, t);
                    }
                    let pred: Vec<Vec<f64>> = rows.iter().zip(&perm).map(|(&i, &pi)| m.predict_row(&|c| if c == j { x[c][pi] } else { x[c][i] })).collect();
                    let v = Eval::new(y.clone(), pred, m.classes()).get(&metric);
                    imp[j] += (if hb { base - v } else { v - base }) / reps as f64;
                }
            }
            let mut ord: Vec<usize> = (0..p).collect();
            ord.sort_by(|a, b| imp[*b].total_cmp(&imp[*a]));
            ord.truncate(top);
            let names: Vec<String> = ord.iter().map(|&j| m.enc.preds[j].clone()).collect();
            let vals: Vec<f64> = ord.iter().map(|&j| imp[j]).collect();
            hbar(&mut g, &names, &vals, "Permutation importance");
            g.title = "Permutation variable importance".into();
            g.xtitle = format!("Increase in {}", super::post::metric_name(&metric));
            st.begin_r().set_mat("permimp", Matrix::col_vector(&vals).with_names(names, vec!["importance".into()]));
        }
        "pdp" | "ice" => {
            let vars = crate::varlist::expand(ds, &rest).map_err(|e| rc_err(111, e)).or_else(|_| Ok::<_, String>(rest.split_whitespace().map(str::to_string).collect()))?;
            let v = vars.first().ok_or_else(|| rc_err(100, "varname required"))?;
            let j = m.enc.preds.iter().position(|q| q == v).ok_or_else(|| rc_err(198, format!("{v} is not a predictor")))?;
            let k = out_index(&m, cmd)?;
            let (x, rows, _) = super::post::active_xy(ds, st, &m)?;
            let npts = util::opt_num(cmd, "points", 3)?.unwrap_or(20.0).max(2.0) as usize;
            let gv = grid_values(&rows.iter().map(|&i| x[j][i]).collect::<Vec<_>>(), npts);
            let curve = |i: usize| -> Vec<(f64, f64)> { gv.iter().map(|&z| (z, m.predict_row(&|c| if c == j { z } else { x[c][i] })[k])).collect() };
            let mut mean = vec![0.0; gv.len()];
            let nice = if sub == "ice" { util::opt_num(cmd, "obs", 3)?.unwrap_or(20.0) as usize } else { 0 };
            let step = (rows.len() / nice.max(1)).max(1);
            for (r, &i) in rows.iter().enumerate() {
                let c = curve(i);
                for (t, pt) in c.iter().enumerate() {
                    mean[t] += pt.1 / rows.len() as f64;
                }
                if sub == "ice" && r % step == 0 && r / step < nice {
                    g.layers.push(Layer { kind: LayerKind::Line, points: c, label: String::new(), color: Some((170, 170, 170)), ..Default::default() });
                }
            }
            let pts: Vec<(f64, f64)> = gv.iter().cloned().zip(mean.iter().cloned()).collect();
            g.layers.push(Layer { kind: LayerKind::Connected, points: pts, label: "Partial dependence".into(), ..Default::default() });
            g.xtitle = v.clone();
            g.ytitle = pred_var(&m, k);
            g.title = if sub == "ice" { "Individual conditional expectation".into() } else { "Partial dependence".into() };
            g.legend = sub == "ice";
            st.begin_r().set_mat("pdp", Matrix::from_rows(&gv.iter().zip(&mean).map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>()).with_names((1..=gv.len()).map(|i| i.to_string()).collect(), vec![v.clone(), "pd".into()]));
        }
        "shapsummary" | "shapvalues" => {
            let k = super::post::shap_output(&m, util::opt_num(cmd, "class", 5)?)?;
            let (x, rows, _) = super::post::active_xy(ds, st, &m)?;
            if sub == "shapvalues" {
                let obs = util::opt_num(cmd, "obs", 3)?.unwrap_or(1.0) as usize;
                let i = obs.checked_sub(1).ok_or_else(|| rc_err(198, "obs() must be positive"))?;
                let nobs = x.first().map_or(0, |c| c.len());
                if i >= nobs {
                    return Err(rc_err(198, format!("obs() must be between 1 and {nobs}")));
                }
                let phi = m.contrib(&|c| x[c][i], k);
                let mut ord: Vec<usize> = (0..p).collect();
                ord.sort_by(|a, b| phi[*b].abs().total_cmp(&phi[*a].abs()));
                ord.truncate(top);
                let names: Vec<String> = ord.iter().map(|&j| format!("{} = {}", m.enc.preds[j], crate::format::fmt_g(x[j][i], 9).trim())).collect();
                let vals: Vec<f64> = ord.iter().map(|&j| phi[j]).collect();
                hbar(&mut g, &names, &vals, "SHAP value");
                g.xlines.push(0.0);
                g.title = format!("SHAP values for observation {obs}");
                g.xtitle = format!("Contribution (expected value {})", crate::format::fmt_g(phi[p], 9).trim());
                let r = st.begin_r();
                r.set_mat("shapvalues", Matrix::row_vector(&phi).with_names(vec![obs.to_string()], m.enc.preds.iter().cloned().chain(std::iter::once("_bias".to_string())).collect()));
                r.set("expected", phi[p]);
            } else {
                let maxobs = util::opt_num(cmd, "obs", 3)?.unwrap_or(500.0) as usize;
                let step = (rows.len() / maxobs.max(1)).max(1);
                let sel: Vec<usize> = rows.iter().cloned().step_by(step).collect();
                let phis: Vec<Vec<f64>> = sel.iter().map(|&i| m.contrib(&|c| x[c][i], k)).collect();
                let mean_abs: Vec<f64> = (0..p).map(|j| phis.iter().map(|f| f[j].abs()).sum::<f64>() / phis.len().max(1) as f64).collect();
                let mut ord: Vec<usize> = (0..p).collect();
                ord.sort_by(|a, b| mean_abs[*b].total_cmp(&mean_abs[*a]));
                ord.truncate(top);
                let kk = ord.len();
                let mut rng = Rng::new(12345);
                for (pos, &j) in ord.iter().enumerate() {
                    let xs: Vec<f64> = sel.iter().map(|&i| x[j][i]).collect();
                    let (lo, hi) = xs.iter().filter(|v| !v.is_nan()).fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(*v), a.1.max(*v)));
                    let ypos = (kk - pos) as f64;
                    for (r, f) in phis.iter().enumerate() {
                        let t = if hi > lo { (xs[r] - lo) / (hi - lo) } else { 0.5 };
                        let t = if t.is_nan() { 0.5 } else { t };
                        let col = ((30.0 + 200.0 * t) as u8, 60, (230.0 - 200.0 * t) as u8);
                        g.layers.push(Layer { kind: LayerKind::Scatter, points: vec![(f[j], ypos + 0.3 * (rng.uniform() - 0.5))], color: Some(col), ..Default::default() });
                    }
                }
                g.ylabels = ord.iter().enumerate().map(|(pos, &j)| ((kk - pos) as f64, m.enc.preds[j].clone())).collect();
                g.xlines.push(0.0);
                g.title = "SHAP summary".into();
                g.xtitle = "SHAP value (color: low to high predictor value)".into();
                let names: Vec<String> = ord.iter().map(|&j| m.enc.preds[j].clone()).collect();
                st.begin_r().set_mat("shapsummary", Matrix::col_vector(&ord.iter().map(|&j| mean_abs[j]).collect::<Vec<_>>()).with_names(names, vec!["mean_abs_shap".into()]));
            }
        }
        "roc" | "prcurve" => {
            if m.kind != Kind::Bin {
                return Err(rc_err(321, format!("h2omlgraph {sub} is allowed only after binary classification")));
            }
            let (lab, ev) = eval_for(&m, cmd)?;
            let th = metrics::thresholds(&ev.y, &ev.p1());
            if sub == "roc" {
                let mut pts = vec![(0.0, 0.0)];
                pts.extend(th.iter().map(|c| (1.0 - c.specificity(), c.recall())));
                g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: format!("ROC (AUC = {:.4})", ev.get("auc")), ..Default::default() });
                g.layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (1.0, 1.0)], label: "Reference".into(), pattern: "dash".into(), ..Default::default() });
                g.xtitle = "1 - specificity".into();
                g.ytitle = "Sensitivity".into();
                g.title = format!("ROC curve ({lab} data)");
                st.begin_r().set("auc", ev.get("auc"));
            } else {
                let mut pts = vec![(0.0, th.first().map_or(1.0, |c| c.precision()))];
                pts.extend(th.iter().map(|c| (c.recall(), c.precision())));
                g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: format!("PR (AUCPR = {:.4})", ev.get("aucpr")), ..Default::default() });
                g.xtitle = "Recall".into();
                g.ytitle = "Precision".into();
                g.title = format!("Precision-recall curve ({lab} data)");
                st.begin_r().set("aucpr", ev.get("aucpr"));
            }
            g.legend = true;
        }
        "scorehistory" | "scoreh" => {
            let tr: Vec<(f64, f64)> = m.history.iter().map(|s| (s.trees as f64, s.train)).collect();
            g.layers.push(Layer { kind: LayerKind::Line, points: tr, label: "Training".into(), ..Default::default() });
            if m.history.iter().any(|s| !s.valid.is_nan()) {
                g.layers.push(Layer { kind: LayerKind::Line, points: m.history.iter().map(|s| (s.trees as f64, s.valid)).collect(), label: "Validation".into(), ..Default::default() });
            }
            g.xtitle = "Number of trees".into();
            g.ytitle = super::post::metric_name(&m.score_metric).to_string();
            g.title = "Score history".into();
            g.legend = true;
            st.begin_r().set_mat("scorehistory", Matrix::from_rows(&m.history.iter().map(|s| vec![s.trees as f64, s.train, s.valid]).collect::<Vec<_>>()).with_names((1..=m.history.len()).map(|i| i.to_string()).collect(), vec!["trees".into(), "train".into(), "valid".into()]));
        }
        "rvfplot" | "rvpplot" | "residuals" => {
            if m.kind != Kind::Reg {
                return Err(rc_err(321, format!("h2omlgraph {sub} is allowed only after regression")));
            }
            let (x, rows, y) = super::post::active_xy(ds, st, &m)?;
            let fit: Vec<f64> = rows.iter().map(|&i| m.predict_row(&|c| x[c][i])[0]).collect();
            let xs: Vec<f64> = if sub == "rvpplot" {
                let v = rest.split_whitespace().next().ok_or_else(|| rc_err(100, "varname required"))?;
                let j = m.enc.preds.iter().position(|q| q == v).ok_or_else(|| rc_err(198, format!("{v} is not a predictor")))?;
                g.xtitle = v.to_string();
                rows.iter().map(|&i| x[j][i]).collect()
            } else {
                g.xtitle = "Fitted values".into();
                fit.clone()
            };
            g.layers.push(Layer { kind: LayerKind::Scatter, points: xs.iter().zip(y.iter().zip(&fit)).map(|(a, (b, c))| (*a, b - c)).collect(), label: "Residuals".into(), ..Default::default() });
            g.ylines.push(0.0);
            g.ytitle = "Residuals".into();
            g.title = "Residuals".into();
        }
        o => return Err(rc_err(198, format!("h2omlgraph {o} is unrecognized"))),
    }
    if let Some(t) = cmd.opt("title", 2) {
        g.title = util::unquote(t);
    }
    g.command = format!("h2omlgraph {}", cmd.main.trim());
    crate::graph::finish(&mut g, cmd, st)
}
