//! Results, replay and postestimation commands of `h2oml`.

use std::rc::Rc;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::learn::{self, Kind, Method, Model};
use super::metrics::{self, Eval};
use super::{current, current_grid, model_by_id, put_model};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

pub fn dispatch(verb: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    match verb {
        "h2omlest" => est(st, cmd),
        "h2omlestat" => estat(ds, st, cmd),
        "h2omlpredict" => predict(ds, st, cmd),
        "h2omlselect" => select(ds, st, cmd),
        "h2omlgof" => gof(ds, st, cmd),
        "h2omlexplore" => explore(ds, st, cmd),
        "h2omlpostestframe" => postestframe(cmd),
        "h2omltree" => tree(st, cmd),
        "h2omlgraph" => super::graphs::graph(ds, st, cmd),
        _ => Err(rc_err(199, format!("unrecognized command: {verb}"))),
    }
}

fn kind_title(m: &Model) -> &'static str {
    match (m.method, m.kind) {
        (Method::Gbm, Kind::Reg) => "Gradient boosting regression",
        (Method::Gbm, Kind::Bin) => "Gradient boosting binary classification",
        (Method::Gbm, Kind::Multi) => "Gradient boosting multiclass classification",
        (Method::Rf, Kind::Reg) => "Random forest regression",
        (Method::Rf, Kind::Bin) => "Random forest binary classification",
        (Method::Rf, Kind::Multi) => "Random forest multiclass classification",
    }
}

/// The evaluation of a model on "train", "valid" or "cv".
pub fn sample_eval<'a>(m: &'a Model, sample: &str) -> Option<&'a Eval> {
    match sample {
        "valid" => m.valid.as_ref(),
        "cv" => m.cv.as_ref().map(|c| &c.eval),
        _ => Some(&m.train),
    }
}

/// Evaluate a model on an `_h2oframe` frame.
pub fn frame_eval(m: &Model, name: &str) -> Result<Eval, String> {
    let d = super::frame_copy(name)?;
    let x = m.enc.columns(&d)?;
    let ycol = util::col(&d, &m.depvar)?;
    let rows: Vec<usize> = (0..d.n_obs).filter(|&i| !is_missing(ycol[i]) && (m.kind == Kind::Reg || m.levels.contains(&ycol[i]))).collect();
    let y: Vec<f64> = rows.iter().map(|&i| if m.kind == Kind::Reg { ycol[i] } else { m.levels.iter().position(|l| *l == ycol[i]).unwrap() as f64 }).collect();
    let pred = m.predict_cols(&x, rows.iter().cloned());
    Ok(Eval::new(y, pred, m.classes()))
}

/// The sample an estat-type command reports on: `valid`, `cv`, `test`
/// (postestimation frame), `frame(name)`, else the postestimation frame
/// when set, else the training data.
pub fn pick_eval(m: &Model, cmd: &Command) -> Result<(String, Eval), String> {
    if cmd.has_opt("valid", 5) {
        return m.valid.clone().map(|e| ("validation".to_string(), e)).ok_or_else(|| rc_err(198, "no validation results; specify validframe() when fitting"));
    }
    if cmd.has_opt("cv", 2) {
        return m.cv.as_ref().map(|c| ("cross-validation".to_string(), c.eval.clone())).ok_or_else(|| rc_err(198, "no cross-validation results; specify cv() when fitting"));
    }
    if let Some(f) = cmd.opt("frame", 5) {
        return Ok((format!("frame {}", f.trim()), frame_eval(m, f.trim())?));
    }
    if cmd.has_opt("train", 5) {
        return Ok(("training".into(), m.train.clone()));
    }
    if let Some(f) = super::postest_frame() {
        return Ok((format!("frame {f}"), frame_eval(m, &f)?));
    }
    Ok(("training".into(), m.train.clone()))
}

pub fn post_results(st: &mut Store, id: u64, gid: Option<u64>, obs: &[usize], n: usize) -> Result<(), String> {
    let m = model_by_id(id).ok_or_else(|| rc_err(301, "model not found"))?;
    let e = st.begin_e();
    e.set_str("cmd", "h2oml");
    e.set_str("subcmd", m.subcmd());
    e.set_str("method", m.method_name());
    e.set_str("depvar", m.depvar.clone());
    e.set_str("predictors", m.enc.preds.join(" "));
    e.set_str("title", kind_title(&m));
    e.set_str("backend", "native");
    e.set_str("_w7f_id", id.to_string());
    if let Some(g) = gid {
        e.set_str("_w7f_grid", g.to_string());
    }
    e.set("N", m.n_train as f64);
    e.set("ntrees", m.hyper.ntrees as f64);
    e.set("ntrees_actual", m.trees.len() as f64);
    e.set("maxdepth", m.hyper.max_depth as f64);
    e.set("minobsleaf", m.hyper.min_leaf);
    if m.method == Method::Gbm {
        e.set("lrate", m.hyper.lrate);
    }
    e.set("samprate", m.hyper.samprate);
    e.set("seed", m.seed as f64);
    if m.kind != Kind::Reg {
        e.set("k_classes", m.levels.len() as f64);
        e.set_str("classes", m.level_labels.join(" "));
    }
    for (name, v) in &m.train.metrics {
        e.set(name, *v);
    }
    if let Some(v) = &m.valid {
        e.set("N_valid", v.y.len() as f64);
        for (name, x) in &v.metrics {
            e.set(&format!("valid_{name}"), *x);
        }
    }
    if let Some(c) = &m.cv {
        e.set("cv_folds", c.folds as f64);
        for (name, x) in &c.eval.metrics {
            e.set(&format!("cv_{name}"), *x);
        }
    }
    let vi = m.varimp();
    e.set_mat("varimp", Matrix::from_rows(&vi.iter().map(|v| vec![v.1, v.2, v.3]).collect::<Vec<_>>()).with_names(vi.iter().map(|v| v.0.clone()).collect(), vec!["raw".into(), "scaled".into(), "percentage".into()]));
    let mut s = vec![false; n];
    for &i in obs {
        if i < n {
            s[i] = true;
        }
    }
    e.sample = Some(s);
    Ok(())
}

fn num(v: f64) -> String {
    if v.is_nan() { ".".into() } else { fmt_g(v, 9).trim().to_string() }
}

pub fn replay(st: &Store) -> Result<String, String> {
    let (_, m) = current(st)?;
    let mut o = Vec::new();
    o.push(format!("{:<50} Number of obs     = {:>9}", kind_title(&m), m.n_train));
    o.push(format!("{:<50} Backend           = {:>9}", format!("Response: {}", m.depvar), "native"));
    if let Some(v) = &m.valid {
        o.push(format!("{:<50} Validation obs    = {:>9}", "", v.y.len()));
    }
    if let Some(c) = &m.cv {
        o.push(format!("{:<50} Cross-validation  = {:>6} folds", "", c.folds));
    }
    if m.kind != Kind::Reg {
        o.push(format!("Classes: {}", m.level_labels.join(", ")));
    }
    o.push(String::new());
    o.push("Model parameters".into());
    o.push(format!("  Number of trees       = {:>8}   (actual = {})", m.hyper.ntrees, m.trees.len()));
    let depths: Vec<usize> = m.trees.iter().flatten().map(|t| t.depth()).collect();
    let leaves: Vec<usize> = m.trees.iter().flatten().map(|t| t.n_leaves()).collect();
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
    o.push(format!("  Tree depth: max input = {:>8}   (min = {}, avg = {:.2}, max = {})", m.hyper.max_depth, depths.iter().min().unwrap_or(&0), mean(&depths), depths.iter().max().unwrap_or(&0)));
    o.push(format!("  Terminal nodes        :            (min = {}, avg = {:.2}, max = {})", leaves.iter().min().unwrap_or(&0), mean(&leaves), leaves.iter().max().unwrap_or(&0)));
    o.push(format!("  Min. obs. per leaf    = {:>8}", num(m.hyper.min_leaf)));
    if m.method == Method::Gbm {
        o.push(format!("  Learning rate         = {:>8}", num(m.hyper.lrate)));
        if m.hyper.predsamprate < 1.0 {
            o.push(format!("  Pred. sampling rate   = {:>8}", num(m.hyper.predsamprate)));
        }
    } else {
        o.push(format!("  Pred. sampled/split   = {:>8}", if m.hyper.mtry > 0.0 { num(m.hyper.mtry) } else { "default".into() }));
    }
    o.push(format!("  Sampling rate         = {:>8}", num(m.hyper.samprate)));
    if m.hyper.stop > 0 {
        o.push(format!("  Early stopping        : {} rounds of {} (tolerance {})", m.hyper.stop, m.score_metric, num(m.hyper.stop_tol)));
    }
    o.push(format!("  Random-number seed    = {:>8}", m.seed));
    o.push(String::new());
    o.push(metric_table(&m, None));
    Ok(o.join("\n"))
}

pub fn metric_name(n: &str) -> &str {
    match n {
        "mse" => "MSE",
        "rmse" => "RMSE",
        "mae" => "MAE",
        "rmsle" => "RMSLE",
        "deviance" => "Mean res. deviance",
        "r2" => "R-squared",
        "logloss" => "Log loss",
        "auc" => "AUC",
        "aucpr" => "AUCPR",
        "gini" => "Gini coefficient",
        "mpce" => "Mean per-class error",
        "error" => "Misclassification",
        o => o,
    }
}

fn metric_table(m: &Model, extra: Option<(&str, &Eval)>) -> String {
    let mut cols: Vec<(&str, &Eval)> = vec![("Training", &m.train)];
    if let Some(v) = &m.valid {
        cols.push(("Validation", v));
    }
    if let Some(c) = &m.cv {
        cols.push(("Cross-valid.", &c.eval));
    }
    if let Some(x) = extra {
        cols.push(x);
    }
    let mut o = vec!["Metric summary".to_string()];
    let w = 22 + 14 * cols.len();
    o.push("-".repeat(w));
    o.push(format!("{:<22}{}", "Metric", cols.iter().map(|c| format!("{:>14}", c.0)).collect::<String>()));
    o.push("-".repeat(w));
    for (name, _) in &m.train.metrics {
        o.push(format!("{:<22}{}", metric_name(name), cols.iter().map(|c| format!("{:>14}", num(c.1.get(name)))).collect::<String>()));
    }
    o.push("-".repeat(w));
    o.join("\n")
}

fn est(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let rest = rest.trim();
    match sub.to_lowercase().as_str() {
        "store" | "sto" => {
            current(st)?;
            if rest.is_empty() {
                return Err(rc_err(198, "name required"));
            }
            st.estimates.insert(rest.to_string(), st.e.clone());
            st.estimates_order.retain(|n| n != rest);
            st.estimates_order.push(rest.to_string());
            Ok(String::new())
        }
        "restore" | "res" => {
            let e = st.estimates.get(rest).cloned().ok_or_else(|| rc_err(111, format!("estimation result {rest} not found")))?;
            if e.macros.get("cmd").map(String::as_str) != Some("h2oml") {
                return Err(rc_err(301, format!("{rest} are not h2oml results")));
            }
            st.e = e;
            Ok(String::new())
        }
        "dir" | "" => {
            let mut o = vec![format!("{:<14} {:<14} {:>8} {:>8}", "Name", "Command", "Trees", "N"), "-".repeat(48)];
            for n in &st.estimates_order {
                if let Some(e) = st.estimates.get(n) {
                    if e.macros.get("cmd").map(String::as_str) == Some("h2oml") {
                        o.push(format!("{:<14} {:<14} {:>8} {:>8}", n, e.macros.get("subcmd").cloned().unwrap_or_default(), num(e.get("ntrees_actual").unwrap_or(f64::NAN)), num(e.get("N").unwrap_or(f64::NAN))));
                    }
                }
            }
            Ok(o.join("\n"))
        }
        "drop" => {
            for n in rest.split_whitespace() {
                st.estimates.remove(n);
                st.estimates_order.retain(|x| x != n);
            }
            Ok(String::new())
        }
        "clear" => {
            let names: Vec<String> = st.estimates.iter().filter(|(_, e)| e.macros.get("cmd").map(String::as_str) == Some("h2oml")).map(|(n, _)| n.clone()).collect();
            for n in names {
                st.estimates.remove(&n);
                st.estimates_order.retain(|x| *x != n);
            }
            Ok(String::new())
        }
        "replay" => {
            if !rest.is_empty() {
                let e = st.estimates.get(rest).cloned().ok_or_else(|| rc_err(111, format!("estimation result {rest} not found")))?;
                let saved = std::mem::replace(&mut st.e, e);
                let r = replay(st);
                st.e = saved;
                return r;
            }
            replay(st)
        }
        o => Err(rc_err(199, format!("h2omlest {o} is unrecognized"))),
    }
}

fn class_label(m: &Model, k: usize) -> String {
    m.level_labels.get(k).cloned().unwrap_or_else(|| k.to_string())
}

fn estat(_ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (_, m) = current(st)?;
    let (sub, _) = crate::xmodels::first_word(cmd.main.trim());
    let r = st.begin_r();
    match sub.to_lowercase().as_str() {
        "metrics" | "metric" => {
            let pf = super::postest_frame();
            let fe = match &pf {
                Some(f) => Some(frame_eval(&m, f)?),
                None => None,
            };
            let t = match (&pf, &fe) {
                (Some(f), Some(e)) => metric_table(&m, Some((f.as_str(), e))),
                _ => metric_table(&m, None),
            };
            for (n, v) in &m.train.metrics {
                r.set(n, *v);
            }
            if let Some(v) = &m.valid {
                for (n, x) in &v.metrics {
                    r.set(&format!("valid_{n}"), *x);
                }
            }
            if let Some(c) = &m.cv {
                for (n, x) in &c.eval.metrics {
                    r.set(&format!("cv_{n}"), *x);
                }
            }
            if let Some(e) = &fe {
                for (n, x) in &e.metrics {
                    r.set(&format!("test_{n}"), *x);
                }
            }
            Ok(t)
        }
        "confmatrix" | "confm" => {
            if m.kind == Kind::Reg {
                return Err(rc_err(321, "h2omlestat confmatrix is allowed only after classification"));
            }
            let (lab, ev) = pick_eval(&m, cmd)?;
            let k = m.levels.len();
            let (cm, thr) = if m.kind == Kind::Bin {
                let p = ev.p1();
                let t = match util::opt_num(cmd, "threshold", 6)? { Some(t) => t, None => metrics::f1_threshold(&ev.y, &p) };
                let c = metrics::Cut::at(&ev.y, &p, t);
                (vec![vec![c.tn, c.fp], vec![c.fnn, c.tp]], Some(t))
            } else {
                (metrics::confusion(&ev.y, &ev.pred, k), None)
            };
            let mut o = vec![format!("Confusion matrix using {lab} data")];
            if let Some(t) = thr {
                o.push(format!("Threshold = {} (predicted {} when Pr >= threshold)", num(t), class_label(&m, 1)));
            }
            let w = 14;
            o.push(format!("{:<14}{}{:>w$}{:>12}", "Observed", (0..k).map(|c| format!("{:>w$}", class_label(&m, c))).collect::<String>(), "Error", "Rate"));
            o.push("-".repeat(14 + w * (k + 1) + 12));
            let mut tot_err = 0.0;
            let mut col = vec![0.0; k];
            for a in 0..k {
                let rt: f64 = cm[a].iter().sum();
                let err = rt - cm[a][a];
                tot_err += err;
                for b in 0..k {
                    col[b] += cm[a][b];
                }
                o.push(format!("{:<14}{}{:>w$}{:>12}", class_label(&m, a), cm[a].iter().map(|v| format!("{:>w$}", num(*v))).collect::<String>(), num(if rt > 0.0 { err / rt } else { 0.0 }), format!("{}/{}", num(err), num(rt))));
            }
            let n: f64 = col.iter().sum();
            o.push("-".repeat(14 + w * (k + 1) + 12));
            o.push(format!("{:<14}{}{:>w$}{:>12}", "Total", col.iter().map(|v| format!("{:>w$}", num(*v))).collect::<String>(), num(tot_err / n.max(1.0)), format!("{}/{}", num(tot_err), num(n))));
            let names: Vec<String> = (0..k).map(|c| class_label(&m, c)).collect();
            r.set_mat("confmatrix", Matrix::from_rows(&cm).with_names(names.clone(), names));
            r.set("N", n);
            r.set("error", tot_err / n.max(1.0));
            if let Some(t) = thr {
                r.set("threshold", t);
            }
            Ok(o.join("\n"))
        }
        "threshmetric" | "thresh" => {
            if m.kind != Kind::Bin {
                return Err(rc_err(321, "h2omlestat threshmetric is allowed only after binary classification"));
            }
            let (lab, ev) = pick_eval(&m, cmd)?;
            let th = metrics::thresholds(&ev.y, &ev.p1());
            let mut o = vec![format!("Threshold metrics using {lab} data"), "-".repeat(48), format!("{:<26}{:>11}{:>11}", "Metric", "Threshold", "Value"), "-".repeat(48)];
            let mut rows = Vec::new();
            for name in metrics::THRESH {
                let best = th.iter().max_by(|a, b| metrics::thresh_value(a, name).total_cmp(&metrics::thresh_value(b, name)));
                let (t, v) = best.map_or((f64::NAN, f64::NAN), |c| (c.t, metrics::thresh_value(c, name)));
                o.push(format!("{:<26}{:>11}{:>11}", name.replace('_', " "), num(t), num(v)));
                rows.push(vec![t, v]);
            }
            o.push("-".repeat(48));
            r.set_mat("threshmetric", Matrix::from_rows(&rows).with_names(metrics::THRESH.iter().map(|s| s.to_string()).collect(), vec!["threshold".into(), "value".into()]));
            Ok(o.join("\n"))
        }
        "cvsummary" | "cvsum" => {
            let c = m.cv.as_ref().ok_or_else(|| rc_err(198, "no cross-validation results; specify cv() when fitting"))?;
            let k = c.fold_metrics.len();
            let mut o = vec![format!("Cross-validation summary ({} folds)", c.folds)];
            let w = 22 + 12 * (k + 2);
            o.push("-".repeat(w));
            o.push(format!("{:<22}{:>12}{:>12}{}", "Metric", "Mean", "Std. dev.", (1..=k).map(|f| format!("{:>12}", format!("Fold {f}"))).collect::<String>()));
            o.push("-".repeat(w));
            let mut rows = Vec::new();
            let mut names = Vec::new();
            for (name, _) in &m.train.metrics {
                let v: Vec<f64> = c.fold_metrics.iter().map(|fm| fm.iter().find(|x| &x.0 == name).map_or(f64::NAN, |x| x.1)).collect();
                let ok: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
                let mean = if ok.is_empty() { f64::NAN } else { ok.iter().sum::<f64>() / ok.len() as f64 };
                let sd = if ok.len() > 1 { (ok.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (ok.len() - 1) as f64).sqrt() } else { f64::NAN };
                o.push(format!("{:<22}{:>12}{:>12}{}", metric_name(name), num(mean), num(sd), v.iter().map(|x| format!("{:>12}", num(*x))).collect::<String>()));
                let mut row = vec![mean, sd];
                row.extend(v);
                rows.push(row);
                names.push(name.clone());
            }
            o.push("-".repeat(w));
            let mut cn = vec!["mean".to_string(), "sd".to_string()];
            cn.extend((1..=k).map(|f| format!("fold{f}")));
            r.set_mat("cvsummary", Matrix::from_rows(&rows).with_names(names, cn));
            Ok(o.join("\n"))
        }
        "gridsummary" | "gridsum" => {
            let _ = r;
            grid_summary(st)
        }
        "hitratio" | "hit" => {
            if m.kind == Kind::Reg {
                return Err(rc_err(321, "h2omlestat hitratio is allowed only after classification"));
            }
            let (lab, ev) = pick_eval(&m, cmd)?;
            let k = m.levels.len();
            let top = util::opt_num(cmd, "top", 3)?.map_or(k.min(10), |v| (v as usize).clamp(1, k));
            let mut hits = vec![0.0; top];
            for (c, p) in ev.y.iter().zip(&ev.pred) {
                let mut ord: Vec<usize> = (0..k).collect();
                ord.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
                if let Some(pos) = ord.iter().position(|v| *v == *c as usize) {
                    for h in hits.iter_mut().skip(pos) {
                        *h += 1.0;
                    }
                }
            }
            let n = ev.y.len().max(1) as f64;
            let mut o = vec![format!("Top-k hit ratios using {lab} data"), "-".repeat(24), format!("{:>6}{:>18}", "k", "Hit ratio"), "-".repeat(24)];
            for (i, h) in hits.iter().enumerate() {
                o.push(format!("{:>6}{:>18}", i + 1, num(h / n)));
            }
            o.push("-".repeat(24));
            r.set_mat("hitratio", Matrix::col_vector(&hits.iter().map(|h| h / n).collect::<Vec<_>>()));
            Ok(o.join("\n"))
        }
        "aucmulticlass" | "aucmulti" => {
            if m.kind == Kind::Reg {
                return Err(rc_err(321, "h2omlestat aucmulticlass is allowed only after classification"));
            }
            let (lab, ev) = pick_eval(&m, cmd)?;
            let k = m.levels.len();
            let n = ev.y.len().max(1) as f64;
            let mut o = vec![format!("Multiclass AUC using {lab} data"), "One-versus-rest".to_string(), "-".repeat(40)];
            let mut ovr = Vec::new();
            let (mut macro_, mut weighted) = (0.0, 0.0);
            for c in 0..k {
                let yc: Vec<f64> = ev.y.iter().map(|v| if *v as usize == c { 1.0 } else { 0.0 }).collect();
                let pc: Vec<f64> = ev.pred.iter().map(|p| p[c]).collect();
                let a = metrics::auc(&yc, &pc);
                let prev = yc.iter().sum::<f64>() / n;
                macro_ += a / k as f64;
                weighted += a * prev;
                o.push(format!("{:<24}{:>16}", format!("{} vs rest", class_label(&m, c)), num(a)));
                ovr.push(a);
            }
            o.push(format!("{:<24}{:>16}", "Macro average", num(macro_)));
            o.push(format!("{:<24}{:>16}", "Weighted average", num(weighted)));
            o.push("-".repeat(40));
            o.push("One-versus-one".into());
            o.push("-".repeat(40));
            let mut pair_sum = 0.0;
            let mut npair: f64 = 0.0;
            for a in 0..k {
                for b in a + 1..k {
                    let sel: Vec<usize> = (0..ev.y.len()).filter(|&i| ev.y[i] as usize == a || ev.y[i] as usize == b).collect();
                    let ya: Vec<f64> = sel.iter().map(|&i| if ev.y[i] as usize == a { 1.0 } else { 0.0 }).collect();
                    let pa: Vec<f64> = sel.iter().map(|&i| ev.pred[i][a]).collect();
                    let yb: Vec<f64> = ya.iter().map(|v| 1.0 - v).collect();
                    let pb: Vec<f64> = sel.iter().map(|&i| ev.pred[i][b]).collect();
                    let v = 0.5 * (metrics::auc(&ya, &pa) + metrics::auc(&yb, &pb));
                    o.push(format!("{:<24}{:>16}", format!("{} vs {}", class_label(&m, a), class_label(&m, b)), num(v)));
                    if !v.is_nan() {
                        pair_sum += v;
                        npair += 1.0;
                    }
                }
            }
            o.push(format!("{:<24}{:>16}", "Macro average", num(pair_sum / npair.max(1.0))));
            o.push("-".repeat(40));
            r.set_mat("auc_ovr", Matrix::col_vector(&ovr));
            r.set("auc_macro", macro_);
            r.set("auc_weighted", weighted);
            r.set("auc_ovo_macro", pair_sum / npair.max(1.0));
            Ok(o.join("\n"))
        }
        o => Err(rc_err(199, format!("h2omlestat {o} is unrecognized"))),
    }
}

pub fn grid_summary(st: &mut Store) -> Result<String, String> {
    let g = current_grid(st).ok_or_else(|| rc_err(198, "no grid results; specify hyperparameter lists when fitting"))?;
    let (cur, _) = current(st)?;
    let mut o = vec![format!("Grid summary ({} models, sorted by {} {})", g.models.len(), g.sample, metric_name(&g.metric))];
    let w = 6 + 12 * (g.params.len() + 1) + 2;
    o.push("-".repeat(w));
    o.push(format!("{:>6}{}{:>12}", "ID", g.params.iter().map(|p| format!("{:>12}", p)).collect::<String>(), g.metric));
    o.push("-".repeat(w));
    let mut rows = Vec::new();
    for (i, id) in g.models.iter().enumerate() {
        let Some(m) = model_by_id(*id) else { continue };
        let v = sample_eval(&m, &g.sample).map_or(f64::NAN, |e| e.get(&g.metric));
        let ps: Vec<f64> = g.params.iter().map(|p| learn::get_hyper(&m.hyper, p)).collect();
        o.push(format!("{:>5}{}{}{:>12}", i + 1, if *id == cur { "*" } else { " " }, ps.iter().map(|p| format!("{:>12}", num(*p))).collect::<String>(), num(v)));
        let mut row = vec![(i + 1) as f64];
        row.extend(ps);
        row.push(v);
        rows.push(row);
    }
    o.push("-".repeat(w));
    o.push("* selected model".into());
    let mut cn = vec!["id".to_string()];
    cn.extend(g.params.iter().cloned());
    cn.push(g.metric.clone());
    let n = rows.len();
    st.begin_r().set_mat("gridsummary", Matrix::from_rows(&rows).with_names((1..=n).map(|i| i.to_string()).collect(), cn));
    Ok(o.join("\n"))
}

fn select(_ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let g = current_grid(st).ok_or_else(|| rc_err(198, "no grid results to select from"))?;
    let k = match util::opt_num(cmd, "id", 2)? {
        Some(v) => v as usize,
        None => cmd.main.trim().parse::<f64>().map(|v| v as usize).unwrap_or(1),
    };
    if k < 1 || k > g.models.len() {
        return Err(rc_err(198, format!("id() must be between 1 and {}", g.models.len())));
    }
    let id = g.models[k - 1];
    let gid: u64 = st.e.macros.get("_w7f_grid").and_then(|v| v.parse().ok()).unwrap();
    let sample = st.e.sample.clone().unwrap_or_default();
    let obs: Vec<usize> = (0..sample.len()).filter(|&i| sample[i]).collect();
    post_results(st, id, Some(gid), &obs, sample.len())?;
    let mut out = format!("(model {k} of the grid selected)\n\n");
    out.push_str(&replay(st)?);
    Ok(out)
}

fn gof(_ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let names: Vec<String> = cmd.main.split_whitespace().map(str::to_string).collect();
    let mut models: Vec<(String, Rc<Model>)> = Vec::new();
    if names.is_empty() {
        models.push(("active".into(), current(st)?.1));
    }
    for n in &names {
        let e = if n == "." { st.e.clone() } else { st.estimates.get(n).cloned().ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))? };
        let id: u64 = e.macros.get("_w7f_id").and_then(|v| v.parse().ok()).ok_or_else(|| rc_err(301, format!("{n} are not h2oml results")))?;
        models.push((n.clone(), model_by_id(id).ok_or_else(|| rc_err(301, "model not available"))?));
    }
    let mut evals = Vec::new();
    let mut lab = String::new();
    for (_, m) in &models {
        let (l, e) = pick_eval(m, cmd)?;
        lab = l;
        evals.push(e);
    }
    let all: Vec<String> = models[0].1.train.metrics.iter().map(|x| x.0.clone()).collect();
    let w = 22 + 14 * models.len();
    let mut o = vec![format!("Goodness of fit using {lab} data"), "-".repeat(w), format!("{:<22}{}", "Metric", models.iter().map(|(n, _)| format!("{:>14}", n)).collect::<String>()), "-".repeat(w)];
    let mut rows = Vec::new();
    for name in &all {
        let v: Vec<f64> = evals.iter().map(|e| e.get(name)).collect();
        o.push(format!("{:<22}{}", metric_name(name), v.iter().map(|x| format!("{:>14}", num(*x))).collect::<String>()));
        rows.push(v);
    }
    o.push("-".repeat(w));
    st.begin_r().set_mat("gof", Matrix::from_rows(&rows).with_names(all, models.iter().map(|m| m.0.clone()).collect()));
    Ok(o.join("\n"))
}

fn explore(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (_, m) = current(st)?;
    let mut o = Vec::new();
    // marginal summary of a grid by hyperparameter value
    if let Some(g) = current_grid(st) {
        for p in &g.params {
            let mut by: Vec<(f64, Vec<f64>)> = Vec::new();
            for id in &g.models {
                if let Some(gm) = model_by_id(*id) {
                    let hv = learn::get_hyper(&gm.hyper, p);
                    let v = sample_eval(&gm, &g.sample).map_or(f64::NAN, |e| e.get(&g.metric));
                    match by.iter_mut().find(|b| b.0 == hv) {
                        Some(b) => b.1.push(v),
                        None => by.push((hv, vec![v])),
                    }
                }
            }
            by.sort_by(|a, b| a.0.total_cmp(&b.0));
            o.push(format!("{} by {p} ({} {})", g.metric, g.sample, metric_name(&g.metric)));
            o.push(format!("{:>12}{:>10}{:>12}{:>12}{:>12}", p, "models", "best", "mean", "worst"));
            let hb = metrics::higher_better(&g.metric);
            for (hv, vs) in &by {
                let mn = vs.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mean = vs.iter().sum::<f64>() / vs.len() as f64;
                let (b, wv) = if hb { (mx, mn) } else { (mn, mx) };
                o.push(format!("{:>12}{:>10}{:>12}{:>12}{:>12}", num(*hv), vs.len(), num(b), num(mean), num(wv)));
            }
            o.push(String::new());
        }
    }
    // refit with changed hyperparameters and compare
    let mut h = m.hyper.clone();
    let mut changed = Vec::new();
    for name in ["ntrees", "maxdepth", "minobsleaf", "lrate", "lratedecay", "samprate", "predsamprate", "predsampvalue"] {
        if let Some(v) = util::opt_num(cmd, name, name.len())? {
            learn::set_hyper(&mut h, name, v);
            changed.push(format!("{name}({})", num(v)));
        }
    }
    if !changed.is_empty() {
        let sample = st.e.sample.clone().unwrap_or_default();
        let obs: Vec<usize> = (0..sample.len()).filter(|&i| sample[i]).collect();
        let m2 = super::with_active(ds, |d| -> Result<Model, String> {
            let x = m.enc.columns(d)?;
            let ycol = util::col(d, &m.depvar)?;
            let rows: Vec<usize> = obs.iter().cloned().filter(|&i| i < d.n_obs).collect();
            let y = rows.iter().map(|&i| if m.kind == Kind::Reg { ycol[i] } else { m.levels.iter().position(|l| *l == ycol[i]).map_or(f64::NAN, |k| k as f64) }).collect();
            let data = learn::Data { x: x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(), y, obs: rows };
            let spec = learn::FitSpec { method: m.method, kind: m.kind, depvar: m.depvar.clone(), enc: m.enc.clone(), levels: m.levels.clone(), level_labels: m.level_labels.clone(), seed: m.seed, cv_folds: m.cv.as_ref().map_or(0, |c| c.folds), fold_var: None };
            learn::fit(&spec, &h, &data, None)
        })?;
        let w = 50;
        o.push(format!("Exploring {}", changed.join(" ")));
        o.push("-".repeat(w));
        o.push(format!("{:<22}{:>14}{:>14}", "Metric", "Current", "Explored"));
        o.push("-".repeat(w));
        let (s1, s2) = if m.cv.is_some() { (m.cv.as_ref().map(|c| &c.eval).unwrap(), m2.cv.as_ref().map(|c| &c.eval).unwrap_or(&m2.train)) } else { (&m.train, &m2.train) };
        for (name, v) in &s1.metrics {
            o.push(format!("{:<22}{:>14}{:>14}", metric_name(name), num(*v), num(s2.get(name))));
        }
        o.push("-".repeat(w));
        let r = st.begin_r();
        for (name, v) in &s2.metrics {
            r.set(name, *v);
        }
        put_model(m2);
    } else if o.is_empty() {
        o.push(replay(st)?);
    }
    Ok(o.join("\n").trim_end().to_string())
}

fn postestframe(cmd: &Command) -> Result<String, String> {
    let name = cmd.main.trim();
    if name.is_empty() {
        return Ok(match super::postest_frame() {
            Some(f) => format!("Postestimation frame: {f}"),
            None => "Postestimation frame: training data".into(),
        });
    }
    if name == "_reset" || name == "clear" {
        super::ST.with(|s| s.borrow_mut().postest = None);
        return Ok(String::new());
    }
    super::frame_copy(name)?;
    super::ST.with(|s| s.borrow_mut().postest = Some(name.to_string()));
    Ok(format!("(postestimation frame set to {name})"))
}

fn tree(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (_, m) = current(st)?;
    let t = util::opt_num(cmd, "tree", 4)?.unwrap_or(1.0) as usize;
    let k = util::opt_num(cmd, "class", 5)?.unwrap_or(1.0) as usize;
    if t < 1 || t > m.trees.len() {
        return Err(rc_err(198, format!("tree() must be between 1 and {}", m.trees.len())));
    }
    let it = &m.trees[t - 1];
    let tr = if it.len() > 1 { it.get(k.saturating_sub(1)).ok_or_else(|| rc_err(198, "class() out of range"))? } else { &it[0] };
    let out_k = if m.method == Method::Rf && m.kind != Kind::Reg { (k.max(1) - 1).min(m.levels.len() - 1) } else { 0 };
    let mut o = vec![format!("Tree {t} of {} ({} nodes, depth {}, {} leaves)", m.trees.len(), tr.nodes.len(), tr.depth(), tr.n_leaves())];
    fn walk(m: &Model, tr: &super::tree::Tree, id: usize, depth: usize, k: usize, o: &mut Vec<String>, path: &str) {
        let n = &tr.nodes[id];
        let pad = "  ".repeat(depth);
        if n.is_leaf() {
            o.push(format!("{pad}{path}leaf: value = {}, obs = {}", num(n.value[k]), n.cover));
        } else {
            let v = &m.enc.preds[n.feat];
            o.push(format!("{pad}{path}split on {v} at {} (obs = {}, missing go {})", num(n.thr), n.cover, if n.na_left { "left" } else { "right" }));
            walk(m, tr, n.left, depth + 1, k, o, &format!("{v} < {}: ", num(n.thr)));
            walk(m, tr, n.right, depth + 1, k, o, &format!("{v} >= {}: ", num(n.thr)));
        }
    }
    walk(&m, tr, 0, 0, out_k, &mut o, "");
    let r = st.begin_r();
    r.set("nodes", tr.nodes.len() as f64);
    r.set("depth", tr.depth() as f64);
    r.set("leaves", tr.n_leaves() as f64);
    Ok(o.join("\n"))
}

fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (_, m) = current(st)?;
    let names: Vec<String> = cmd.main.split_whitespace().map(|s| s.trim_end_matches('*').to_string()).collect();
    if names.is_empty() {
        return Err(rc_err(198, "new variable name required"));
    }
    let frame = cmd.opt("frame", 5).map(|s| s.trim().to_string());
    let pr = cmd.has_opt("pr", 2);
    let class = cmd.has_opt("class", 5);
    let contrib = cmd.has_opt("contrib", 7) || cmd.has_opt("shap", 4);
    let thr_opt = util::opt_num(cmd, "threshold", 6)?;
    let cls = util::opt_num(cmd, "class", 5)?;
    if let Some(f) = &frame {
        super::frame_copy(f)?;
    }
    super::with_frame(ds, frame.as_deref(), |d| -> Result<String, String> {
        let mask = util::touse(d, st, cmd)?;
        let x = m.enc.columns(d)?;
        let n = d.n_obs;
        for nm in &names {
            if d.get_var(nm).is_some() && !contrib {
                return Err(rc_err(110, format!("variable {nm} already defined")));
            }
        }
        let preds: Vec<Option<Vec<f64>>> = (0..n).map(|i| if mask[i] { Some(m.predict_row(&|j| x[j][i])) } else { None }).collect();
        let mut new: Vec<(String, Vec<f64>)> = Vec::new();
        if contrib {
            let p = m.enc.preds.len();
            let vn: Vec<String> = if names.len() == p + 1 { names.clone() } else {
                let stub = &names[0];
                let mut v: Vec<String> = m.enc.preds.iter().map(|q| format!("{stub}_{q}")).collect();
                v.push(format!("{stub}_bias"));
                v
            };
            let k = shap_output(&m, cls)?;
            let mut cols = vec![vec![SYSMISS; n]; p + 1];
            for i in 0..n {
                if mask[i] {
                    let phi = m.contrib(&|j| x[j][i], k);
                    for j in 0..=p {
                        cols[j][i] = phi[j];
                    }
                }
            }
            for (nm, c) in vn.into_iter().zip(cols) {
                new.push((nm, c));
            }
        } else if m.kind == Kind::Reg {
            new.push((names[0].clone(), preds.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| v[0])).collect()));
        } else if pr {
            let k = m.levels.len();
            if names.len() == 1 && m.kind == Kind::Bin {
                new.push((names[0].clone(), preds.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| v[1])).collect()));
            } else if names.len() == k {
                for (c, nm) in names.iter().enumerate() {
                    new.push((nm.clone(), preds.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| v[c])).collect()));
                }
            } else if names.len() == 1 {
                for c in 0..k {
                    new.push((format!("{}{}", names[0], c + 1), preds.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| v[c])).collect()));
                }
            } else {
                return Err(rc_err(198, format!("specify 1 or {k} new variables with pr")));
            }
        } else {
            let _ = class;
            let thr = if m.kind == Kind::Bin { thr_opt.unwrap_or_else(|| metrics::f1_threshold(&m.train.y, &m.train.p1())) } else { 0.0 };
            new.push((names[0].clone(), preds.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| {
                let c = if m.kind == Kind::Bin { usize::from(v[1] >= thr) } else { metrics::argmax(v) };
                m.levels[c]
            })).collect()));
        }
        let mut nmiss = 0;
        for (nm, vals) in new {
            if d.get_var(&nm).is_some() {
                return Err(rc_err(110, format!("variable {nm} already defined")));
            }
            nmiss += vals.iter().filter(|v| is_missing(**v)).count();
            let mut var = util::new_double(&nm, vals);
            if m.kind != Kind::Reg && !pr && !contrib {
                if let Some(dv) = d.get_var(&m.depvar) {
                    var.value_label_name = dv.value_label_name.clone();
                }
            }
            d.add_variable(var).map_err(|e| rc_err(110, e))?;
        }
        st.data_changed = true;
        Ok(if nmiss > 0 { format!("({nmiss} missing value{} generated)", if nmiss == 1 { "" } else { "s" }) } else { String::new() })
    })
}

/// All-row predictions and the estimation-sample flag on the active data
/// (used by the graphs).
pub fn active_xy(ds: &mut DataSet, st: &Store, m: &Model) -> Result<(Vec<Vec<f64>>, Vec<usize>, Vec<f64>), String> {
    let sample = st.e.sample.clone().unwrap_or_default();
    super::with_active(ds, |d| {
        let x = m.enc.columns(d)?;
        let rows: Vec<usize> = (0..d.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
        let ycol = util::col(d, &m.depvar)?;
        let y = rows.iter().map(|&i| if m.kind == Kind::Reg { ycol[i] } else { m.levels.iter().position(|l| *l == ycol[i]).map_or(f64::NAN, |k| k as f64) }).collect();
        Ok((x, rows, y))
    })
}

/// Output whose SHAP values are computed: the positive class of binary
/// forests, class `class(#)` (1-based) of multiclass models.
pub fn shap_output(m: &Model, class: Option<f64>) -> Result<usize, String> {
    Ok(match m.kind {
        Kind::Reg => 0,
        Kind::Bin => usize::from(m.method == Method::Rf),
        Kind::Multi => {
            let c = class.ok_or_else(|| rc_err(198, "option class(#) required after multiclass classification"))? as usize;
            if c < 1 || c > m.levels.len() {
                return Err(rc_err(198, format!("class() must be between 1 and {}", m.levels.len())));
            }
            c - 1
        }
    })
}
