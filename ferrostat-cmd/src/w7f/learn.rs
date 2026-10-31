//! Gradient boosting machines (Friedman's gradient boosting with
//! Newton-step leaves) and random forests (Breiman: bootstrap samples and
//! random predictor subsets at each split), with early stopping,
//! k-fold cross-validation and hyperparameter grids.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;

use super::metrics::{higher_better, Eval};
use super::tree::{grow, GrowParams, Tree};
use crate::eval::rc_err;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    Gbm,
    Rf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Reg,
    Bin,
    Multi,
}

/// Hyperparameters of one model.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyper {
    pub ntrees: usize,
    pub max_depth: usize,
    pub min_leaf: f64,
    pub lrate: f64,
    pub lrate_decay: f64,
    /// row sampling rate per tree
    pub samprate: f64,
    /// share of predictors tried at each split (gbm)
    pub predsamprate: f64,
    /// predictors tried at each split (rf; <= 0: default)
    pub mtry: f64,
    pub stop: usize,
    pub stop_metric: String,
    pub stop_tol: f64,
}

impl Hyper {
    pub fn default_for(m: Method) -> Hyper {
        match m {
            Method::Gbm => Hyper { ntrees: 50, max_depth: 5, min_leaf: 10.0, lrate: 0.1, lrate_decay: 1.0, samprate: 1.0, predsamprate: 1.0, mtry: -1.0, stop: 0, stop_metric: String::new(), stop_tol: 1e-3 },
            Method::Rf => Hyper { ntrees: 50, max_depth: 20, min_leaf: 1.0, lrate: 1.0, lrate_decay: 1.0, samprate: 1.0, predsamprate: 1.0, mtry: -1.0, stop: 0, stop_metric: String::new(), stop_tol: 1e-3 },
        }
    }
}

/// Predictors (columns, NaN = missing) and response of a sample.
#[derive(Clone, Debug, Default)]
pub struct Data {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    /// observation numbers in the data set
    pub obs: Vec<usize>,
}

impl Data {
    pub fn n(&self) -> usize {
        self.obs.len()
    }
    pub fn subset(&self, rows: &[usize]) -> Data {
        Data { x: self.x.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(), y: rows.iter().map(|&i| self.y[i]).collect(), obs: rows.iter().map(|&i| self.obs[i]).collect() }
    }
}

/// How predictors are read from a data set (string predictors are coded
/// by the position of their value in the training levels).
#[derive(Clone, Debug, Default)]
pub struct Encoding {
    pub preds: Vec<String>,
    pub str_levels: Vec<Option<Vec<String>>>,
}

impl Encoding {
    pub fn build(ds: &DataSet, preds: &[String], mask: &[bool]) -> Result<Encoding, String> {
        let mut sl = Vec::new();
        for p in preds {
            let v = ds.get_var(p).ok_or_else(|| rc_err(111, format!("variable {p} not found")))?;
            if v.var_type.is_string() {
                let mut lv: Vec<String> = v.data.iter().enumerate().filter(|(i, _)| mask[*i]).map(|(_, x)| x.as_string().unwrap_or("").to_string()).filter(|s| !s.is_empty()).collect();
                lv.sort();
                lv.dedup();
                sl.push(Some(lv));
            } else {
                sl.push(None);
            }
        }
        Ok(Encoding { preds: preds.to_vec(), str_levels: sl })
    }

    /// Predictor columns for all observations.
    pub fn columns(&self, ds: &DataSet) -> Result<Vec<Vec<f64>>, String> {
        let mut out = Vec::new();
        for (p, lv) in self.preds.iter().zip(&self.str_levels) {
            let v = ds.get_var(p).ok_or_else(|| rc_err(111, format!("variable {p} not found")))?;
            let col: Vec<f64> = match lv {
                Some(lv) => v.data.iter().map(|x| lv.binary_search(&x.as_string().unwrap_or("").to_string()).map(|k| k as f64).unwrap_or(f64::NAN)).collect(),
                None => {
                    if v.var_type.is_string() {
                        return Err(rc_err(109, format!("{p}: type mismatch")));
                    }
                    v.data.iter().map(|x| { let f = x.num(); if is_missing(f) { f64::NAN } else { f } }).collect()
                }
            };
            out.push(col);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Cv {
    pub folds: usize,
    pub eval: Eval,
    /// metrics of each held-out fold
    pub fold_metrics: Vec<Vec<(String, f64)>>,
}

/// One scoring step: number of trees, training and validation metric.
#[derive(Clone, Debug)]
pub struct Score {
    pub trees: usize,
    pub train: f64,
    pub valid: f64,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub method: Method,
    pub kind: Kind,
    pub depvar: String,
    pub enc: Encoding,
    /// class values (classification)
    pub levels: Vec<f64>,
    pub level_labels: Vec<String>,
    pub hyper: Hyper,
    pub seed: u64,
    pub init: Vec<f64>,
    /// gbm: per iteration, one tree per output; rf: one multi-output tree
    pub trees: Vec<Vec<Tree>>,
    pub importance: Vec<f64>,
    pub history: Vec<Score>,
    pub score_metric: String,
    pub train: Eval,
    pub valid: Option<Eval>,
    pub cv: Option<Cv>,
    pub n_train: usize,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn softmax(f: &[f64]) -> Vec<f64> {
    let m = f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = f.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

impl Model {
    pub fn classes(&self) -> usize {
        if self.kind == Kind::Reg { 0 } else { self.levels.len() }
    }

    /// Outputs per tree.
    fn outputs(&self) -> usize {
        match (self.method, self.kind) {
            (_, Kind::Reg) | (Method::Gbm, Kind::Bin) => 1,
            _ => self.levels.len(),
        }
    }

    /// Raw score: gbm margins (link scale), rf averaged leaf values.
    pub fn raw(&self, xr: &dyn Fn(usize) -> f64) -> Vec<f64> {
        match self.method {
            Method::Gbm => {
                let mut f = self.init.clone();
                for it in &self.trees {
                    for (k, t) in it.iter().enumerate() {
                        f[k] += t.leaf(xr).value[0];
                    }
                }
                f
            }
            Method::Rf => {
                let k = self.outputs();
                let mut f = vec![0.0; k];
                for it in &self.trees {
                    let v = &it[0].leaf(xr).value;
                    for j in 0..k {
                        f[j] += v[j];
                    }
                }
                let nt = self.trees.len().max(1) as f64;
                f.iter().map(|v| v / nt).collect()
            }
        }
    }

    /// Prediction from the raw score: [value] or class probabilities.
    pub fn link(&self, f: &[f64]) -> Vec<f64> {
        match (self.method, self.kind) {
            (_, Kind::Reg) => vec![f[0]],
            (Method::Gbm, Kind::Bin) => {
                let p = sigmoid(f[0]);
                vec![1.0 - p, p]
            }
            (Method::Gbm, Kind::Multi) => softmax(f),
            (Method::Rf, _) => {
                let s: f64 = f.iter().sum();
                if s > 0.0 { f.iter().map(|v| v / s).collect() } else { vec![1.0 / f.len() as f64; f.len()] }
            }
        }
    }

    pub fn predict_row(&self, xr: &dyn Fn(usize) -> f64) -> Vec<f64> {
        self.link(&self.raw(xr))
    }

    pub fn predict_cols(&self, x: &[Vec<f64>], rows: impl Iterator<Item = usize>) -> Vec<Vec<f64>> {
        rows.map(|i| self.predict_row(&|j| x[j][i])).collect()
    }

    /// SHAP contributions (raw-score scale) of output k for one row; the
    /// last element is the bias (expected raw score).
    pub fn contrib(&self, xr: &dyn Fn(usize) -> f64, k: usize) -> Vec<f64> {
        let p = self.enc.preds.len();
        let mut phi = vec![0.0; p + 1];
        match self.method {
            Method::Gbm => {
                phi[p] = self.init[k];
                for it in &self.trees {
                    it[k].shap(xr, 0, &mut phi[..p]);
                    phi[p] += it[k].expected(0);
                }
            }
            Method::Rf => {
                let nt = self.trees.len().max(1) as f64;
                let mut tmp = vec![0.0; p];
                for it in &self.trees {
                    it[0].shap(xr, k, &mut tmp);
                    phi[p] += it[0].expected(k) / nt;
                }
                for j in 0..p {
                    phi[j] = tmp[j] / nt;
                }
            }
        }
        phi
    }

    /// Relative importances (scaled to a maximum of 1) and percentages.
    pub fn varimp(&self) -> Vec<(String, f64, f64, f64)> {
        let mx = self.importance.iter().cloned().fold(0.0, f64::max);
        let tot: f64 = self.importance.iter().sum();
        let mut v: Vec<(String, f64, f64, f64)> = self.enc.preds.iter().zip(&self.importance).map(|(n, a)| (n.clone(), *a, if mx > 0.0 { a / mx } else { 0.0 }, if tot > 0.0 { a / tot } else { 0.0 })).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    pub fn method_name(&self) -> &'static str {
        match self.method {
            Method::Gbm => "gbm",
            Method::Rf => "rf",
        }
    }

    pub fn subcmd(&self) -> String {
        let a = if self.method == Method::Gbm { "gb" } else { "rf" };
        let b = match self.kind {
            Kind::Reg => "regress",
            Kind::Bin => "binclass",
            Kind::Multi => "multiclass",
        };
        format!("{a}{b}")
    }
}

/// Metric scored during training (early stopping and score history).
pub fn default_metric(kind: Kind) -> &'static str {
    if kind == Kind::Reg { "deviance" } else { "logloss" }
}

fn metric_of(y: &[f64], pred: &[Vec<f64>], classes: usize, m: &str) -> f64 {
    let n = y.len().max(1) as f64;
    match m {
        "deviance" | "mse" if classes == 0 => y.iter().zip(pred).map(|(a, p)| (a - p[0]).powi(2)).sum::<f64>() / n,
        "rmse" if classes == 0 => (y.iter().zip(pred).map(|(a, p)| (a - p[0]).powi(2)).sum::<f64>() / n).sqrt(),
        "mae" if classes == 0 => y.iter().zip(pred).map(|(a, p)| (a - p[0]).abs()).sum::<f64>() / n,
        "logloss" | "deviance" => -y.iter().zip(pred).map(|(c, p)| p[*c as usize].clamp(1e-15, 1.0).ln()).sum::<f64>() / n,
        "misclassification" | "error" => y.iter().zip(pred).filter(|(c, p)| super::metrics::argmax(p) != **c as usize).count() as f64 / n,
        _ => Eval::new(y.to_vec(), pred.to_vec(), classes).get(m),
    }
}

/// Settings of a fit besides the hyperparameters.
pub struct FitSpec {
    pub method: Method,
    pub kind: Kind,
    pub depvar: String,
    pub enc: Encoding,
    pub levels: Vec<f64>,
    pub level_labels: Vec<String>,
    pub seed: u64,
    pub cv_folds: usize,
    pub fold_var: Option<Vec<f64>>,
}

/// Fit a model on `d` (validation data `v` scored if given).
pub fn fit(spec: &FitSpec, hy: &Hyper, d: &Data, v: Option<&Data>) -> Result<Model, String> {
    let mut m = fit_core(spec, hy, d, v, spec.seed)?;
    if spec.cv_folds >= 2 || spec.fold_var.is_some() {
        let n = d.n();
        let fold: Vec<usize> = match &spec.fold_var {
            Some(f) => {
                let mut lv: Vec<f64> = f.clone();
                lv.sort_by(|a, b| a.total_cmp(b));
                lv.dedup();
                f.iter().map(|x| lv.iter().position(|l| l == x).unwrap()).collect()
            }
            None => {
                let mut rng = Rng::new(spec.seed ^ 0x9e37_79b9);
                let mut perm: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
                    perm.swap(i, j);
                }
                let mut f = vec![0; n];
                for (r, &i) in perm.iter().enumerate() {
                    f[i] = r % spec.cv_folds;
                }
                f
            }
        };
        let k = fold.iter().max().map_or(0, |v| v + 1);
        if k < 2 {
            return Err(rc_err(198, "cross-validation requires at least 2 folds"));
        }
        let mut pred = vec![Vec::new(); n];
        let mut fm = Vec::new();
        for f in 0..k {
            let tr: Vec<usize> = (0..n).filter(|&i| fold[i] != f).collect();
            let te: Vec<usize> = (0..n).filter(|&i| fold[i] == f).collect();
            if te.is_empty() || tr.is_empty() {
                continue;
            }
            let dtr = d.subset(&tr);
            let mf = fit_core(spec, hy, &dtr, None, spec.seed.wrapping_add(f as u64 + 1))?;
            let pf = mf.predict_cols(&d.x, te.iter().cloned());
            let yf: Vec<f64> = te.iter().map(|&i| d.y[i]).collect();
            fm.push(Eval::new(yf, pf.clone(), m.classes()).metrics);
            for (i, p) in te.iter().zip(pf) {
                pred[*i] = p;
            }
        }
        m.cv = Some(Cv { folds: k, eval: Eval::new(d.y.clone(), pred, m.classes()), fold_metrics: fm });
    }
    Ok(m)
}

fn fit_core(spec: &FitSpec, hy: &Hyper, d: &Data, v: Option<&Data>, seed: u64) -> Result<Model, String> {
    let n = d.n();
    if n < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let p = d.x.len();
    let nl = spec.levels.len();
    let classes = if spec.kind == Kind::Reg { 0 } else { nl };
    let mut rng = Rng::new(seed);
    let metric = if hy.stop_metric.is_empty() { default_metric(spec.kind).to_string() } else { hy.stop_metric.clone() };
    let mut m = Model {
        method: spec.method,
        kind: spec.kind,
        depvar: spec.depvar.clone(),
        enc: spec.enc.clone(),
        levels: spec.levels.clone(),
        level_labels: spec.level_labels.clone(),
        hyper: hy.clone(),
        seed,
        init: Vec::new(),
        trees: Vec::new(),
        importance: vec![0.0; p],
        history: Vec::new(),
        score_metric: metric.clone(),
        train: Eval::default(),
        valid: None,
        cv: None,
        n_train: n,
    };
    let kout = m.outputs();
    let onehot = |k: usize| -> Vec<f64> { d.y.iter().map(|c| if *c as usize == k { 1.0 } else { 0.0 }).collect() };
    let mut scores: Vec<f64> = Vec::new();
    let stop_check = |scores: &[f64], k: usize, tol: f64, hb: bool| -> bool {
        if k == 0 || scores.len() < 2 * k {
            return false;
        }
        let s: Vec<f64> = scores.iter().map(|v| if hb { -v } else { *v }).collect();
        let ma = |t: usize| s[t + 1 - k..=t].iter().sum::<f64>() / k as f64;
        let last = ma(s.len() - 1);
        let best = (k - 1..s.len() - k).map(ma).fold(f64::INFINITY, f64::min);
        // relative improvement smaller than the tolerance
        !(last < best - tol * best.abs())
    };
    let hb = higher_better(&metric);
    // rf: out-of-bag predictions give the training metrics
    let mut oob_pred: Option<Vec<Vec<f64>>> = None;
    match spec.method {
        Method::Gbm => {
            let init: Vec<f64> = match spec.kind {
                Kind::Reg => vec![d.y.iter().sum::<f64>() / n as f64],
                Kind::Bin => {
                    let pm = (d.y.iter().sum::<f64>() / n as f64).clamp(1e-6, 1.0 - 1e-6);
                    vec![(pm / (1.0 - pm)).ln()]
                }
                Kind::Multi => (0..nl).map(|k| (onehot(k).iter().sum::<f64>() / n as f64).max(1e-6).ln()).collect(),
            };
            m.init = init.clone();
            let mut f: Vec<Vec<f64>> = init.iter().map(|v| vec![*v; n]).collect();
            let mut fv: Vec<Vec<f64>> = v.map_or(Vec::new(), |v| init.iter().map(|a| vec![*a; v.n()]).collect());
            let yk: Vec<Vec<f64>> = if spec.kind == Kind::Multi { (0..nl).map(onehot).collect() } else { Vec::new() };
            let mtry = if hy.predsamprate < 1.0 { Some(((hy.predsamprate * p as f64).ceil() as usize).max(1)) } else { None };
            let mut scale = hy.lrate;
            for _ in 0..hy.ntrees {
                let rows: Vec<usize> = if hy.samprate < 1.0 { (0..n).filter(|_| rng.uniform() < hy.samprate).collect() } else { (0..n).collect() };
                let rows = if rows.len() < 2 { (0..n).collect() } else { rows };
                let prm = GrowParams { max_depth: hy.max_depth, min_leaf: hy.min_leaf, mtry, scale };
                let mut iter = Vec::new();
                match spec.kind {
                    Kind::Reg => {
                        let g: Vec<f64> = (0..n).map(|i| d.y[i] - f[0][i]).collect();
                        iter.push(grow(&d.x, &rows, &[g], &vec![1.0; n], &prm, &mut rng));
                    }
                    Kind::Bin => {
                        let pr: Vec<f64> = f[0].iter().map(|z| sigmoid(*z)).collect();
                        let g: Vec<f64> = (0..n).map(|i| d.y[i] - pr[i]).collect();
                        let h: Vec<f64> = pr.iter().map(|q| (q * (1.0 - q)).max(1e-6)).collect();
                        iter.push(grow(&d.x, &rows, &[g], &h, &prm, &mut rng));
                    }
                    Kind::Multi => {
                        let pr: Vec<Vec<f64>> = (0..n).map(|i| softmax(&f.iter().map(|c| c[i]).collect::<Vec<_>>())).collect();
                        let c = nl as f64 / (nl as f64 - 1.0).max(1.0);
                        for k in 0..nl {
                            let g: Vec<f64> = (0..n).map(|i| yk[k][i] - pr[i][k]).collect();
                            let h: Vec<f64> = (0..n).map(|i| (c * pr[i][k] * (1.0 - pr[i][k])).max(1e-6)).collect();
                            iter.push(grow(&d.x, &rows, &[g], &h, &prm, &mut rng));
                        }
                    }
                }
                for (k, t) in iter.iter().enumerate() {
                    for nd in &t.nodes {
                        if !nd.is_leaf() {
                            m.importance[nd.feat] += nd.gain;
                        }
                    }
                    for i in 0..n {
                        f[k][i] += t.leaf(&|j| d.x[j][i]).value[0];
                    }
                    if let Some(v) = v {
                        for i in 0..v.n() {
                            fv[k][i] += t.leaf(&|j| v.x[j][i]).value[0];
                        }
                    }
                }
                m.trees.push(iter);
                scale *= hy.lrate_decay;
                let pt: Vec<Vec<f64>> = (0..n).map(|i| m.link(&f.iter().map(|c| c[i]).collect::<Vec<_>>())).collect();
                let tr = metric_of(&d.y, &pt, classes, &metric);
                let va = v.map_or(f64::NAN, |v| {
                    let pv: Vec<Vec<f64>> = (0..v.n()).map(|i| m.link(&fv.iter().map(|c| c[i]).collect::<Vec<_>>())).collect();
                    metric_of(&v.y, &pv, classes, &metric)
                });
                m.history.push(Score { trees: m.trees.len(), train: tr, valid: va });
                scores.push(if v.is_some() { va } else { tr });
                if stop_check(&scores, hy.stop, hy.stop_tol, hb) {
                    break;
                }
            }
        }
        Method::Rf => {
            let targets: Vec<Vec<f64>> = if spec.kind == Kind::Reg { vec![d.y.clone()] } else { (0..nl).map(onehot).collect() };
            let mtry = if hy.mtry > 0.0 {
                (hy.mtry as usize).clamp(1, p)
            } else if spec.kind == Kind::Reg {
                (p / 3).max(1)
            } else {
                ((p as f64).sqrt().floor() as usize).max(1)
            };
            let ns = ((hy.samprate * n as f64).round() as usize).max(1);
            let mut oob_sum = vec![vec![0.0; kout]; n];
            let mut oob_cnt = vec![0usize; n];
            let mut vsum = v.map_or(Vec::new(), |v| vec![vec![0.0; kout]; v.n()]);
            let mut insum = vec![vec![0.0; kout]; n];
            let prm = GrowParams { max_depth: hy.max_depth, min_leaf: hy.min_leaf, mtry: Some(mtry), scale: 1.0 };
            for _ in 0..hy.ntrees {
                let mut inbag = vec![false; n];
                let rows: Vec<usize> = (0..ns).map(|_| {
                    let i = ((rng.uniform() * n as f64) as usize).min(n - 1);
                    inbag[i] = true;
                    i
                }).collect();
                let t = grow(&d.x, &rows, &targets, &vec![1.0; n], &prm, &mut rng);
                for nd in &t.nodes {
                    if !nd.is_leaf() {
                        m.importance[nd.feat] += nd.gain;
                    }
                }
                for i in 0..n {
                    let lv = &t.leaf(&|j| d.x[j][i]).value;
                    for k in 0..kout {
                        insum[i][k] += lv[k];
                        if !inbag[i] {
                            oob_sum[i][k] += lv[k];
                        }
                    }
                    if !inbag[i] {
                        oob_cnt[i] += 1;
                    }
                }
                if let Some(v) = v {
                    for i in 0..v.n() {
                        let lv = &t.leaf(&|j| v.x[j][i]).value;
                        for k in 0..kout {
                            vsum[i][k] += lv[k];
                        }
                    }
                }
                m.trees.push(vec![t]);
                let nt = m.trees.len() as f64;
                let pt: Vec<Vec<f64>> = (0..n).map(|i| {
                    if oob_cnt[i] > 0 { m.link(&oob_sum[i].iter().map(|s| s / oob_cnt[i] as f64).collect::<Vec<_>>()) } else { m.link(&insum[i].iter().map(|s| s / nt).collect::<Vec<_>>()) }
                }).collect();
                let tr = metric_of(&d.y, &pt, classes, &metric);
                let va = v.map_or(f64::NAN, |v| {
                    let pv: Vec<Vec<f64>> = (0..v.n()).map(|i| m.link(&vsum[i].iter().map(|s| s / nt).collect::<Vec<_>>())).collect();
                    metric_of(&v.y, &pv, classes, &metric)
                });
                m.history.push(Score { trees: m.trees.len(), train: tr, valid: va });
                scores.push(if v.is_some() { va } else { tr });
                oob_pred = Some(pt);
                if stop_check(&scores, hy.stop, hy.stop_tol, hb) {
                    break;
                }
            }
        }
    }
    let pt = oob_pred.unwrap_or_else(|| m.predict_cols(&d.x, 0..n));
    m.train = Eval::new(d.y.clone(), pt, classes);
    if let Some(v) = v {
        let pv = m.predict_cols(&v.x, 0..v.n());
        m.valid = Some(Eval::new(v.y.clone(), pv, classes));
    }
    Ok(m)
}

/// Hyperparameter combinations of a grid.
pub fn grid_combos(base: &Hyper, lists: &[(String, Vec<f64>)], random: bool, maxmodels: usize, seed: u64) -> Vec<Hyper> {
    let mut combos: Vec<Hyper> = vec![base.clone()];
    for (name, vals) in lists {
        let mut next = Vec::new();
        for h in &combos {
            for v in vals {
                let mut h2 = h.clone();
                set_hyper(&mut h2, name, *v);
                next.push(h2);
            }
        }
        combos = next;
    }
    if random {
        let mut rng = Rng::new(seed ^ 0x5bd1_e995);
        let n = combos.len();
        for i in (1..n).rev() {
            let j = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
            combos.swap(i, j);
        }
    }
    if maxmodels > 0 {
        combos.truncate(maxmodels);
    }
    combos
}

pub fn set_hyper(h: &mut Hyper, name: &str, v: f64) {
    match name {
        "ntrees" => h.ntrees = v.max(1.0) as usize,
        "maxdepth" => h.max_depth = v.max(0.0) as usize,
        "minobsleaf" => h.min_leaf = v.max(1.0),
        "lrate" => h.lrate = v,
        "lratedecay" => h.lrate_decay = v,
        "samprate" => h.samprate = v,
        "predsamprate" => h.predsamprate = v,
        "predsampvalue" => h.mtry = v,
        _ => {}
    }
}

pub fn get_hyper(h: &Hyper, name: &str) -> f64 {
    match name {
        "ntrees" => h.ntrees as f64,
        "maxdepth" => h.max_depth as f64,
        "minobsleaf" => h.min_leaf,
        "lrate" => h.lrate,
        "lratedecay" => h.lrate_decay,
        "samprate" => h.samprate,
        "predsamprate" => h.predsamprate,
        "predsampvalue" => h.mtry,
        _ => f64::NAN,
    }
}
