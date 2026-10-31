//! Performance metrics of regression and classification predictions.

/// How a model is evaluated on a sample: the response (value or class
/// index) and the prediction (value, or class probabilities).
#[derive(Clone, Debug, Default)]
pub struct Eval {
    pub y: Vec<f64>,
    pub pred: Vec<Vec<f64>>,
    pub metrics: Vec<(String, f64)>,
}

impl Eval {
    pub fn new(y: Vec<f64>, pred: Vec<Vec<f64>>, classes: usize) -> Self {
        let metrics = if classes == 0 { reg(&y, &pred) } else if classes == 2 { bin(&y, &pred) } else { multi(&y, &pred, classes) };
        Eval { y, pred, metrics }
    }
    pub fn get(&self, name: &str) -> f64 {
        self.metrics.iter().find(|(n, _)| n == name).map(|m| m.1).unwrap_or(f64::NAN)
    }
    /// probability of the positive (second) class
    pub fn p1(&self) -> Vec<f64> {
        self.pred.iter().map(|p| p[1]).collect()
    }
}

/// Is a larger value of the metric better?
pub fn higher_better(m: &str) -> bool {
    matches!(m, "auc" | "aucpr" | "r2" | "accuracy" | "f1")
}

fn reg(y: &[f64], pred: &[Vec<f64>]) -> Vec<(String, f64)> {
    let n = y.len().max(1) as f64;
    let mse = y.iter().zip(pred).map(|(a, p)| (a - p[0]).powi(2)).sum::<f64>() / n;
    let mae = y.iter().zip(pred).map(|(a, p)| (a - p[0]).abs()).sum::<f64>() / n;
    let ok = y.iter().zip(pred).all(|(a, p)| *a > -1.0 && p[0] > -1.0);
    let rmsle = if ok { (y.iter().zip(pred).map(|(a, p)| ((1.0 + p[0]).ln() - (1.0 + a).ln()).powi(2)).sum::<f64>() / n).sqrt() } else { f64::NAN };
    let my = y.iter().sum::<f64>() / n;
    let vy = y.iter().map(|a| (a - my).powi(2)).sum::<f64>() / n;
    vec![
        ("mse".into(), mse),
        ("rmse".into(), mse.sqrt()),
        ("mae".into(), mae),
        ("rmsle".into(), rmsle),
        ("deviance".into(), mse),
        ("r2".into(), if vy > 0.0 { 1.0 - mse / vy } else { f64::NAN }),
    ]
}

fn logloss(y: &[f64], pred: &[Vec<f64>]) -> f64 {
    let n = y.len().max(1) as f64;
    -y.iter().zip(pred).map(|(c, p)| p[*c as usize].clamp(1e-15, 1.0).ln()).sum::<f64>() / n
}

fn bin(y: &[f64], pred: &[Vec<f64>]) -> Vec<(String, f64)> {
    let n = y.len().max(1) as f64;
    let p: Vec<f64> = pred.iter().map(|v| v[1]).collect();
    let mse = y.iter().zip(&p).map(|(a, q)| (a - q).powi(2)).sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let vy = my * (1.0 - my);
    let a = auc(y, &p);
    let th = thresholds(y, &p);
    let best = th.iter().max_by(|a, b| a.f1().total_cmp(&b.f1()));
    let mpce = best.map_or(f64::NAN, |c| 1.0 - c.mean_acc());
    vec![
        ("logloss".into(), logloss(y, pred)),
        ("auc".into(), a),
        ("aucpr".into(), aucpr(y, &p)),
        ("gini".into(), 2.0 * a - 1.0),
        ("mpce".into(), mpce),
        ("mse".into(), mse),
        ("rmse".into(), mse.sqrt()),
        ("r2".into(), if vy > 0.0 { 1.0 - mse / vy } else { f64::NAN }),
    ]
}

fn multi(y: &[f64], pred: &[Vec<f64>], k: usize) -> Vec<(String, f64)> {
    let n = y.len().max(1) as f64;
    let mse = y.iter().zip(pred).map(|(c, p)| (1.0 - p[*c as usize]).powi(2)).sum::<f64>() / n;
    let cm = confusion(y, pred, k);
    let mut errs = Vec::new();
    for c in 0..k {
        let tot: f64 = cm[c].iter().sum();
        if tot > 0.0 {
            errs.push(1.0 - cm[c][c] / tot);
        }
    }
    let correct: f64 = (0..k).map(|c| cm[c][c]).sum();
    vec![
        ("logloss".into(), logloss(y, pred)),
        ("mpce".into(), errs.iter().sum::<f64>() / errs.len().max(1) as f64),
        ("error".into(), 1.0 - correct / n),
        ("mse".into(), mse),
        ("rmse".into(), mse.sqrt()),
    ]
}

/// Most probable class.
pub fn argmax(p: &[f64]) -> usize {
    let mut b = 0;
    for (i, v) in p.iter().enumerate() {
        if *v > p[b] {
            b = i;
        }
    }
    b
}

/// Confusion counts (rows: observed, columns: predicted by argmax).
pub fn confusion(y: &[f64], pred: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut cm = vec![vec![0.0; k]; k];
    for (c, p) in y.iter().zip(pred) {
        cm[*c as usize][argmax(p)] += 1.0;
    }
    cm
}

/// Area under the ROC curve (Mann-Whitney, ties counted one half).
pub fn auc(y: &[f64], p: &[f64]) -> f64 {
    let mut v: Vec<(f64, bool)> = p.iter().zip(y).map(|(q, a)| (*q, *a > 0.5)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut rank_sum, mut i) = (0.0, 0);
    let npos = v.iter().filter(|a| a.1).count() as f64;
    let nneg = v.len() as f64 - npos;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 {
            j += 1;
        }
        let r = (i + j + 1) as f64 / 2.0;
        rank_sum += r * v[i..j].iter().filter(|a| a.1).count() as f64;
        i = j;
    }
    if npos == 0.0 || nneg == 0.0 {
        return f64::NAN;
    }
    (rank_sum - npos * (npos + 1.0) / 2.0) / (npos * nneg)
}

/// Counts at one threshold (positive when p >= t).
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    pub t: f64,
    pub tp: f64,
    pub fp: f64,
    pub tn: f64,
    pub fnn: f64,
}

impl Cut {
    fn div(a: f64, b: f64) -> f64 {
        if b > 0.0 { a / b } else { 0.0 }
    }
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp > 0.0 { self.tp / (self.tp + self.fp) } else { 1.0 }
    }
    pub fn recall(&self) -> f64 {
        Self::div(self.tp, self.tp + self.fnn)
    }
    pub fn specificity(&self) -> f64 {
        Self::div(self.tn, self.tn + self.fp)
    }
    pub fn fbeta(&self, b: f64) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        Self::div((1.0 + b * b) * p * r, b * b * p + r)
    }
    pub fn f1(&self) -> f64 {
        self.fbeta(1.0)
    }
    pub fn accuracy(&self) -> f64 {
        Self::div(self.tp + self.tn, self.tp + self.tn + self.fp + self.fnn)
    }
    pub fn mean_acc(&self) -> f64 {
        0.5 * (self.recall() + self.specificity())
    }
    pub fn min_acc(&self) -> f64 {
        self.recall().min(self.specificity())
    }
    pub fn mcc(&self) -> f64 {
        let d = ((self.tp + self.fp) * (self.tp + self.fnn) * (self.tn + self.fp) * (self.tn + self.fnn)).sqrt();
        Self::div(self.tp * self.tn - self.fp * self.fnn, d).abs()
    }
    pub fn at(y: &[f64], p: &[f64], t: f64) -> Cut {
        let mut c = Cut { t, tp: 0.0, fp: 0.0, tn: 0.0, fnn: 0.0 };
        for (a, q) in y.iter().zip(p) {
            match (*a > 0.5, *q >= t) {
                (true, true) => c.tp += 1.0,
                (true, false) => c.fnn += 1.0,
                (false, true) => c.fp += 1.0,
                (false, false) => c.tn += 1.0,
            }
        }
        c
    }
}

/// Counts at every distinct predicted probability, from the highest down.
pub fn thresholds(y: &[f64], p: &[f64]) -> Vec<Cut> {
    let mut v: Vec<(f64, bool)> = p.iter().zip(y).map(|(q, a)| (*q, *a > 0.5)).collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let npos = v.iter().filter(|a| a.1).count() as f64;
    let nneg = v.len() as f64 - npos;
    let (mut tp, mut fp) = (0.0, 0.0);
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let t = v[i].0;
        while i < v.len() && v[i].0 == t {
            if v[i].1 { tp += 1.0 } else { fp += 1.0 }
            i += 1;
        }
        out.push(Cut { t, tp, fp, tn: nneg - fp, fnn: npos - tp });
    }
    out
}

/// Area under the precision-recall curve (trapezoid rule).
pub fn aucpr(y: &[f64], p: &[f64]) -> f64 {
    let th = thresholds(y, p);
    let mut area = 0.0;
    let (mut r0, mut p0) = (0.0, th.first().map_or(1.0, |c| c.precision()));
    for c in &th {
        let (r, pr) = (c.recall(), c.precision());
        area += (r - r0) * (pr + p0) / 2.0;
        r0 = r;
        p0 = pr;
    }
    area
}

/// Names and accessors of the threshold metrics.
pub const THRESH: [&str; 10] = ["f1", "f2", "f0point5", "accuracy", "precision", "recall", "specificity", "absolute_mcc", "min_per_class_accuracy", "mean_per_class_accuracy"];

pub fn thresh_value(c: &Cut, m: &str) -> f64 {
    match m {
        "f1" => c.f1(),
        "f2" => c.fbeta(2.0),
        "f0point5" => c.fbeta(0.5),
        "accuracy" => c.accuracy(),
        "precision" => c.precision(),
        "recall" => c.recall(),
        "specificity" => c.specificity(),
        "absolute_mcc" => c.mcc(),
        "min_per_class_accuracy" => c.min_acc(),
        _ => c.mean_acc(),
    }
}

/// Threshold maximizing F1 (0.5 when undefined).
pub fn f1_threshold(y: &[f64], p: &[f64]) -> f64 {
    thresholds(y, p).iter().max_by(|a, b| a.f1().total_cmp(&b.f1())).map_or(0.5, |c| c.t)
}
