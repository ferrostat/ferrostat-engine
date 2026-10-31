//! Wave 7f: tree-ensemble machine learning with the `h2o`/`h2oml` command
//! surface, implemented natively (no external cluster): gradient boosting
//! and random forests for regression, binary and multiclass
//! classification, hyperparameter grids, cross-validation and the
//! postestimation commands (`h2omlest`, `h2omlestat`, `h2omlpredict`,
//! `h2omlgraph`, `h2omlselect`, `h2omlgof`, `h2omlexplore`,
//! `h2omlpostestframe`, `h2omltree`).
//!
//! "Frames" created by `_h2oframe` are copies of the data kept in memory;
//! when one is current, `h2oml` and `h2omlpredict` work on it, otherwise on
//! the data in memory.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::xmodels::Handlers;

pub mod learn;
pub mod metrics;
pub mod post;
pub mod graphs;
pub mod tree;

use learn::{Data, Encoding, FitSpec, Hyper, Kind, Method, Model};

/// A tuned grid: the models fitted and how they were ranked.
pub struct Grid {
    pub params: Vec<String>,
    pub models: Vec<u64>,
    pub metric: String,
    pub sample: String,
}

#[derive(Default)]
struct State {
    next: u64,
    models: BTreeMap<u64, Rc<Model>>,
    grids: BTreeMap<u64, Rc<Grid>>,
    frames: BTreeMap<String, DataSet>,
    current: Option<String>,
    postest: Option<String>,
}

thread_local! {
    static ST: RefCell<State> = RefCell::new(State::default());
}

fn new_id() -> u64 {
    ST.with(|s| {
        let mut s = s.borrow_mut();
        s.next += 1;
        s.next
    })
}

pub fn put_model(m: Model) -> u64 {
    let id = new_id();
    ST.with(|s| {
        let mut s = s.borrow_mut();
        s.models.insert(id, Rc::new(m));
        while s.models.len() > 400 {
            let k = *s.models.keys().next().unwrap();
            s.models.remove(&k);
        }
    });
    id
}

pub fn model_by_id(id: u64) -> Option<Rc<Model>> {
    ST.with(|s| s.borrow().models.get(&id).cloned())
}

pub fn grid_by_id(id: u64) -> Option<Rc<Grid>> {
    ST.with(|s| s.borrow().grids.get(&id).cloned())
}

/// The model of the active estimation results.
pub fn current(st: &Store) -> Result<(u64, Rc<Model>), String> {
    let id: u64 = st.e.macros.get("_w7f_id").and_then(|v| v.parse().ok()).ok_or_else(|| rc_err(301, "last h2oml estimates not found"))?;
    model_by_id(id).map(|m| (id, m)).ok_or_else(|| rc_err(301, "h2oml model no longer available"))
}

pub fn current_grid(st: &Store) -> Option<Rc<Grid>> {
    st.e.macros.get("_w7f_grid").and_then(|v| v.parse().ok()).and_then(grid_by_id)
}

/// A copy of an `_h2oframe` frame.
pub fn frame_copy(name: &str) -> Result<DataSet, String> {
    ST.with(|s| s.borrow().frames.get(name).cloned()).ok_or_else(|| rc_err(111, format!("frame {name} not found")))
}

pub fn postest_frame() -> Option<String> {
    ST.with(|s| s.borrow().postest.clone())
}

/// Run `f` on the active data: the current `_h2oframe` frame, or the data in memory.
pub fn with_active<R>(ds: &mut DataSet, f: impl FnOnce(&mut DataSet) -> R) -> R {
    with_frame(ds, None, f)
}

/// Run `f` on frame `name` (None: the current frame, or the data in memory).
pub fn with_frame<R>(ds: &mut DataSet, name: Option<&str>, f: impl FnOnce(&mut DataSet) -> R) -> R {
    let name = name.map(str::to_string).or_else(|| ST.with(|s| s.borrow().current.clone()));
    let Some(name) = name else { return f(ds) };
    let taken = ST.with(|s| s.borrow_mut().frames.remove(&name));
    match taken {
        Some(mut fr) => {
            let r = f(&mut fr);
            ST.with(|s| s.borrow_mut().frames.insert(name, fr));
            r
        }
        None => f(ds),
    }
}

pub struct H2o {
    verb: &'static str,
}

impl CommandHandler for H2o {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        match self.verb {
            "h2o" => h2o(cmd),
            "_h2oframe" => h2oframe(ds, st, cmd),
            "h2oml" => h2oml(ds, st, cmd),
            v => post::dispatch(v, ds, st, cmd),
        }
    }
}

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    for verb in ["h2o", "_h2oframe", "h2oml", "h2omlest", "h2omlestat", "h2omlexplore", "h2omlgraph", "h2omlpredict", "h2omlselect", "h2omlgof", "h2omlpostestframe", "h2omltree"] {
        v.push((verb, Box::new(H2o { verb })));
    }
    v
}

fn h2o(cmd: &Command) -> Result<String, String> {
    let (sub, _) = crate::xmodels::first_word(cmd.main.trim());
    match sub.to_lowercase().as_str() {
        "init" | "connect" | "" => Ok("Connected to the native backend: models are fitted in-process, no cluster is started.".into()),
        "shutdown" | "disconnect" | "close" => Ok("Native backend: nothing to shut down.".into()),
        "query" | "status" | "cluster" => {
            let n = ST.with(|s| s.borrow().frames.len());
            Ok(format!("Backend: native (in-process)\nFrames in memory: {n}"))
        }
        o => Err(rc_err(199, format!("h2o {o} is unrecognized"))),
    }
}

fn h2oframe(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let rest = rest.trim().to_string();
    match sub.to_lowercase().as_str() {
        "put" => {
            let name = cmd.opt("into", 4).map(|s| s.trim().to_string()).ok_or_else(|| rc_err(198, "option into() required"))?;
            let mask = crate::util::touse(ds, st, cmd)?;
            let mut fr = ds.clone();
            if !rest.is_empty() {
                let vars = crate::varlist::expand(ds, &rest).map_err(|e| rc_err(111, e))?;
                fr.keep_variables(&vars);
            }
            if mask.iter().any(|m| !m) {
                fr.keep_if(&mask)?;
            }
            let exists = ST.with(|s| s.borrow().frames.contains_key(&name));
            if exists && !cmd.has_opt("replace", 7) {
                return Err(rc_err(110, format!("frame {name} already exists")));
            }
            let n = fr.n_obs;
            ST.with(|s| {
                let mut s = s.borrow_mut();
                s.frames.insert(name.clone(), fr);
                if cmd.has_opt("current", 7) {
                    s.current = Some(name.clone());
                }
            });
            Ok(format!("(frame {name} created with {n} observations)"))
        }
        "change" | "cwf" => {
            let name = rest.split_whitespace().next().unwrap_or("").to_string();
            frame_copy(&name)?;
            ST.with(|s| s.borrow_mut().current = Some(name));
            Ok(String::new())
        }
        "pwf" => Ok(ST.with(|s| s.borrow().current.clone()).unwrap_or_else(|| "(data in memory)".into())),
        "dir" | "list" => {
            let v: Vec<String> = ST.with(|s| {
                let s = s.borrow();
                s.frames.iter().map(|(n, d)| format!("{}{:<20} {:>8} x {}", if s.current.as_deref() == Some(n.as_str()) { "*" } else { " " }, n, d.n_obs, d.vars.len())).collect()
            });
            Ok(if v.is_empty() { "(no frames)".into() } else { v.join("\n") })
        }
        "drop" => {
            for n in rest.split_whitespace() {
                ST.with(|s| {
                    let mut s = s.borrow_mut();
                    s.frames.remove(n);
                    if s.current.as_deref() == Some(n) {
                        s.current = None;
                    }
                });
            }
            Ok(String::new())
        }
        "unset" | "reset" => {
            ST.with(|s| s.borrow_mut().current = None);
            Ok(String::new())
        }
        "describe" => {
            let name = if rest.is_empty() { ST.with(|s| s.borrow().current.clone()).ok_or_else(|| rc_err(198, "no current frame"))? } else { rest.clone() };
            let fr = frame_copy(&name)?;
            Ok(format!("Frame {name}: {} observations\n{}", fr.n_obs, fr.describe()))
        }
        "get" => {
            let name = cmd.opt("frame", 5).map(|s| s.trim().to_string()).or_else(|| if rest.is_empty() { ST.with(|s| s.borrow().current.clone()) } else { Some(rest.clone()) }).ok_or_else(|| rc_err(198, "frame name required"))?;
            if !cmd.has_opt("clear", 5) && ds.n_obs > 0 && st.data_changed {
                return Err(rc_err(4, "no; data in memory would be lost"));
            }
            *ds = frame_copy(&name)?;
            st.data_changed = false;
            Ok(String::new())
        }
        "split" => {
            let src = if rest.is_empty() { ST.with(|s| s.borrow().current.clone()).ok_or_else(|| rc_err(198, "frame name required"))? } else { rest.clone() };
            let fr = frame_copy(&src)?;
            let into: Vec<String> = cmd.opt("into", 4).unwrap_or("").split_whitespace().map(str::to_string).collect();
            let mut props = crate::margins::numlist(cmd.opt("split", 5).unwrap_or("0.8 0.2"))?;
            if into.len() < 2 || props.len() + 1 < into.len() {
                return Err(rc_err(198, "into() requires two or more frame names and split() their proportions"));
            }
            if props.len() < into.len() {
                props.push(1.0 - props.iter().sum::<f64>());
            }
            let seed = seed_opt(cmd, st)?;
            let mut rng = crate::rng::Rng::new(seed);
            let tot: f64 = props.iter().sum();
            let cut: Vec<f64> = props.iter().scan(0.0, |a, p| { *a += p / tot; Some(*a) }).collect();
            let assign: Vec<usize> = (0..fr.n_obs).map(|_| { let u = rng.uniform(); cut.iter().position(|c| u < *c).unwrap_or(cut.len() - 1) }).collect();
            let mut msg = Vec::new();
            for (k, nm) in into.iter().enumerate() {
                let mut f2 = fr.clone();
                let mask: Vec<bool> = assign.iter().map(|a| *a == k).collect();
                f2.keep_if(&mask)?;
                msg.push(format!("(frame {nm}: {} observations)", f2.n_obs));
                ST.with(|s| s.borrow_mut().frames.insert(nm.clone(), f2));
            }
            Ok(msg.join("\n"))
        }
        "rename" => {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "_h2oframe rename oldname newname"));
            }
            let fr = frame_copy(w[0])?;
            ST.with(|s| {
                let mut s = s.borrow_mut();
                s.frames.remove(w[0]);
                s.frames.insert(w[1].to_string(), fr);
                if s.current.as_deref() == Some(w[0]) {
                    s.current = Some(w[1].to_string());
                }
            });
            Ok(String::new())
        }
        "copy" => {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "_h2oframe copy frame newframe"));
            }
            let fr = frame_copy(w[0])?;
            ST.with(|s| s.borrow_mut().frames.insert(w[1].to_string(), fr));
            Ok(String::new())
        }
        "factor" | "toenum" | "tonumeric" => Ok(String::new()),
        o => Err(rc_err(199, format!("_h2oframe {o} is unrecognized"))),
    }
}

/// `h2orseed(#)` / `rseed(#)`, or a seed drawn from the session RNG.
pub fn seed_opt(cmd: &Command, st: &Store) -> Result<u64, String> {
    for o in ["h2orseed", "rseed"] {
        if let Some(v) = crate::util::opt_num(cmd, o, o.len())? {
            return Ok(v as u64);
        }
    }
    Ok(st.rng.borrow_mut().next_u64() >> 11)
}

const HYPER: [&str; 8] = ["ntrees", "maxdepth", "minobsleaf", "lrate", "lratedecay", "samprate", "predsamprate", "predsampvalue"];

/// `name(text)` inside an option's text.
pub fn inner_opt(text: &str, name: &str) -> Option<String> {
    let low = text.to_lowercase();
    let mut from = 0;
    while let Some(p) = low[from..].find(name) {
        let s = from + p;
        let before_ok = s == 0 || !low.as_bytes()[s - 1].is_ascii_alphanumeric();
        let after = s + name.len();
        if before_ok && low[after..].trim_start().starts_with('(') {
            let open = after + low[after..].find('(').unwrap();
            let mut depth = 0;
            for (i, c) in text[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(text[open + 1..open + i].trim().to_string());
                        }
                    }
                    _ => {}
                }
            }
            return None;
        }
        from = s + name.len();
    }
    None
}

fn h2oml(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
    let sub = sub.to_lowercase();
    if sub.is_empty() {
        return post::replay(st);
    }
    let (method, kind) = match sub.as_str() {
        "gbregress" | "gbmregress" => (Method::Gbm, Kind::Reg),
        "gbbinclass" | "gbmbinclass" => (Method::Gbm, Kind::Bin),
        "gbmulticlass" | "gbmmulticlass" => (Method::Gbm, Kind::Multi),
        "rfregress" => (Method::Rf, Kind::Reg),
        "rfbinclass" => (Method::Rf, Kind::Bin),
        "rfmulticlass" => (Method::Rf, Kind::Multi),
        o => return Err(rc_err(199, format!("h2oml {o} is unrecognized"))),
    };
    let known = ["ntrees", "maxdepth", "minobsleaf", "lrate", "lratedecay", "samprate", "predsamprate", "predsampvalue", "predsamplevalue", "stop", "stopmetric", "stoptolerance", "h2orseed", "rseed", "cv", "validframe", "tune", "grid", "nolog", "noheader", "maxmodels", "metric"];
    for (o, _) in &cmd.opts {
        let o = o.to_lowercase();
        if !known.contains(&o.as_str()) {
            return Err(rc_err(198, format!("option {o} not allowed")));
        }
    }
    if cmd.weight.is_some() {
        return Err(rc_err(101, "weights not allowed"));
    }
    let seed = seed_opt(cmd, st)?;
    let mut base = Hyper::default_for(method);
    let mut lists: Vec<(String, Vec<f64>)> = Vec::new();
    for h in HYPER {
        let raw = cmd.opt(h, h.len()).or_else(|| if h == "predsampvalue" { cmd.opt("predsamplevalue", 15) } else { None });
        if let Some(r) = raw {
            let vals = crate::margins::numlist(r)?;
            if vals.len() == 1 {
                learn::set_hyper(&mut base, h, vals[0]);
            } else {
                lists.push((h.to_string(), vals));
            }
        }
    }
    // stop(#[, stopmetric(name) stoptolerance(#)]) or the separate options
    let stop_txt = cmd.opt("stop", 4).unwrap_or("").to_string();
    let lead = stop_txt.split(',').next().unwrap_or("").trim().to_string();
    if !lead.is_empty() {
        base.stop = lead.parse::<f64>().map_err(|_| rc_err(198, "stop() requires a number of scoring rounds"))?.max(0.0) as usize;
    }
    if let Some(v) = inner_opt(&stop_txt, "stopmetric").or_else(|| cmd.opt("stopmetric", 10).map(str::to_string)) {
        base.stop_metric = v.trim().to_lowercase();
    }
    if let Some(v) = inner_opt(&stop_txt, "stoptolerance").or_else(|| cmd.opt("stoptolerance", 13).map(str::to_string)) {
        base.stop_tol = v.trim().parse().map_err(|_| rc_err(198, "stoptolerance() requires a number"))?;
    }
    let valid_metrics = ["deviance", "mse", "rmse", "mae", "logloss", "auc", "aucpr", "misclassification", "error", "r2", "mpce", "rmsle"];
    if !base.stop_metric.is_empty() && !valid_metrics.contains(&base.stop_metric.as_str()) {
        return Err(rc_err(198, format!("stopmetric({}) not allowed", base.stop_metric)));
    }
    let tune = cmd.opt("tune", 4).map(str::to_string).unwrap_or_default();
    let gridtype = inner_opt(&tune, "grid").or_else(|| cmd.opt("grid", 4).map(str::to_string)).unwrap_or_else(|| "cartesian".into()).to_lowercase();
    let maxmodels = inner_opt(&tune, "maxmodels").or_else(|| cmd.opt("maxmodels", 9).map(str::to_string)).map(|v| v.trim().parse::<f64>().unwrap_or(0.0) as usize).unwrap_or(0);
    let tmetric = inner_opt(&tune, "metric").or_else(|| cmd.opt("metric", 6).map(str::to_string)).map(|s| s.trim().to_lowercase());

    // the data the model is fitted on
    let vars = crate::varlist::expand(ds_active_ref(ds).as_ref().unwrap_or(ds), &rest).map_err(|e| rc_err(111, e))?;
    if vars.len() < 2 {
        return Err(rc_err(102, "too few variables specified"));
    }
    let depvar = vars[0].clone();
    let preds: Vec<String> = vars[1..].to_vec();
    let cvopt = cmd.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case("cv")).map(|(_, a)| a.clone().unwrap_or_default());
    let validname = cmd.opt("validframe", 5).map(|s| s.trim().to_string());
    let prepared = with_active(ds, |d| -> Result<_, String> {
        let mut mask = crate::util::touse(d, st, cmd)?;
        crate::util::markout(d, &mut mask, &[depvar.clone()]);
        let ycol = crate::util::col(d, &depvar)?;
        let enc = Encoding::build(d, &preds, &mask)?;
        let mut fold_var = None;
        let mut cv_folds = 0;
        if let Some(c) = &cvopt {
            let c = c.split(',').next().unwrap_or("").trim();
            if c.is_empty() {
                cv_folds = 3;
            } else if let Ok(k) = c.parse::<f64>() {
                cv_folds = k as usize;
            } else {
                let fv = crate::util::col(d, c)?;
                for (i, m) in mask.iter_mut().enumerate() {
                    if is_missing(fv[i]) {
                        *m = false;
                    }
                }
                fold_var = Some(fv);
            }
        }
        let rows: Vec<usize> = (0..d.n_obs).filter(|&i| mask[i]).collect();
        let x = enc.columns(d)?;
        let (levels, labels) = if kind == Kind::Reg {
            (Vec::new(), Vec::new())
        } else {
            let mut lv: Vec<f64> = rows.iter().map(|&i| ycol[i]).collect();
            lv.sort_by(|a, b| a.total_cmp(b));
            lv.dedup();
            let vl = d.get_var(&depvar).map(|v| v.value_label_name.clone()).unwrap_or_default();
            let labels = lv.iter().map(|v| d.value_labels.get(&vl).and_then(|l| l.lookup(*v).map(str::to_string)).unwrap_or_else(|| crate::format::fmt_g(*v, 9).trim().to_string())).collect();
            (lv, labels)
        };
        if kind == Kind::Bin && levels.len() != 2 {
            return Err(rc_err(450, format!("{depvar} must have exactly two outcomes for binary classification ({} found)", levels.len())));
        }
        if kind == Kind::Multi && levels.len() < 2 {
            return Err(rc_err(148, format!("{depvar} has fewer than two outcomes")));
        }
        let ymap = |v: f64| -> f64 { if kind == Kind::Reg { v } else { levels.iter().position(|l| *l == v).map_or(f64::NAN, |k| k as f64) } };
        let data = Data { x: x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(), y: rows.iter().map(|&i| ymap(ycol[i])).collect(), obs: rows.clone() };
        let fv = fold_var.map(|f| rows.iter().map(|&i| f[i]).collect());
        Ok((data, enc, levels, labels, cv_folds, fv))
    })?;
    let (data, enc, levels, labels, cv_folds, fold_var) = prepared;
    let valid = match &validname {
        Some(vn) => {
            let vd = frame_copy(vn)?;
            let x = enc.columns(&vd)?;
            let ycol = crate::util::col(&vd, &depvar)?;
            let rows: Vec<usize> = (0..vd.n_obs).filter(|&i| !is_missing(ycol[i]) && (kind == Kind::Reg || levels.contains(&ycol[i]))).collect();
            let y = rows.iter().map(|&i| if kind == Kind::Reg { ycol[i] } else { levels.iter().position(|l| *l == ycol[i]).unwrap() as f64 }).collect();
            Some(Data { x: x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(), y, obs: rows })
        }
        None => None,
    };
    let spec = FitSpec { method, kind, depvar: depvar.clone(), enc, levels, level_labels: labels, seed, cv_folds, fold_var };
    if lists.is_empty() {
        let m = learn::fit(&spec, &base, &data, valid.as_ref())?;
        let id = put_model(m);
        post::post_results(st, id, None, &data.obs, ds_len(ds))?;
        return post::replay(st);
    }
    // hyperparameter grid
    let combos = learn::grid_combos(&base, &lists, gridtype.starts_with("rand"), maxmodels, seed);
    let sample = if valid.is_some() { "valid" } else if cv_folds >= 2 || spec.fold_var.is_some() { "cv" } else { "train" };
    let metric = tmetric.unwrap_or_else(|| learn::default_metric(kind).to_string());
    let mut ids = Vec::new();
    let mut scores = Vec::new();
    for h in &combos {
        let m = learn::fit(&spec, h, &data, valid.as_ref())?;
        scores.push(post::sample_eval(&m, sample).map_or(f64::NAN, |e| e.get(&metric)));
        ids.push(put_model(m));
    }
    if scores.iter().all(|v| v.is_nan()) {
        return Err(rc_err(198, format!("metric {metric} is not available for this model")));
    }
    let hb = metrics::higher_better(&metric);
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (scores[a], scores[b]);
        let c = if hb { y.total_cmp(&x) } else { x.total_cmp(&y) };
        if x.is_nan() || y.is_nan() { x.is_nan().cmp(&y.is_nan()) } else { c }
    });
    let models: Vec<u64> = order.iter().map(|&i| ids[i]).collect();
    let grid = Grid { params: lists.iter().map(|l| l.0.clone()).collect(), models: models.clone(), metric, sample: sample.into() };
    let gid = new_id();
    ST.with(|s| s.borrow_mut().grids.insert(gid, Rc::new(grid)));
    post::post_results(st, models[0], Some(gid), &data.obs, ds_len(ds))?;
    let mut out = post::replay(st)?;
    out.push_str("\n\n");
    out.push_str(&post::grid_summary(st)?);
    Ok(out)
}

fn ds_active_ref(_ds: &DataSet) -> Option<DataSet> {
    let cur = ST.with(|s| s.borrow().current.clone())?;
    frame_copy(&cur).ok()
}

fn ds_len(ds: &DataSet) -> usize {
    ds_active_ref(ds).map_or(ds.n_obs, |d| d.n_obs)
}
