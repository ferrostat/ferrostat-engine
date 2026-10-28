//! Discriminant analysis: `discrim lda|qda|logistic|knn`, `candisc`, and
//! their `estat` subcommands (classtable, errorrate, grsummarize, list,
//! loadings, structure, canontest, grmeans, classfunctions) and `predict`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::mv_cluster::Measure;
use crate::mvutil::{self, Boxed};
use crate::store::{Results, Store};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Lda,
    Qda,
    Logistic,
    Knn,
}

impl Kind {
    fn code(self) -> &'static str {
        match self {
            Kind::Lda => "lda",
            Kind::Qda => "qda",
            Kind::Logistic => "logistic",
            Kind::Knn => "knn",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Kind::Lda => "Linear discriminant analysis",
            Kind::Qda => "Quadratic discriminant analysis",
            Kind::Logistic => "Logistic discriminant analysis",
            Kind::Knn => "Kth-nearest-neighbor discriminant analysis",
        }
    }
    fn from(s: &str) -> Option<Kind> {
        Some(match s {
            "lda" => Kind::Lda,
            "qda" => Kind::Qda,
            "logistic" | "logit" => Kind::Logistic,
            "knn" => Kind::Knn,
            _ => return None,
        })
    }
}

/// Model state reconstructed from e().
struct Model {
    kind: Kind,
    vars: Vec<String>,
    gvals: Vec<f64>,
    glabels: Vec<String>,
    priors: Vec<f64>,
    means: Matrix,
    /// pooled within covariance (lda) / per-group covariances (qda)
    w: Matrix,
    covs: Vec<Matrix>,
    /// logistic coefficients: (G-1) x (p+1), base = last group
    lb: Matrix,
    /// knn training data
    train: Vec<Vec<f64>>,
    train_g: Vec<usize>,
    k: usize,
    measure: Measure,
    mahal: bool,
}

fn log_post(m: &Model, x: &[f64], skip: Option<usize>) -> Vec<f64> {
    let gn = m.gvals.len();
    let p = x.len();
    match m.kind {
        Kind::Lda | Kind::Qda => (0..gn)
            .map(|g| {
                let cov = if m.kind == Kind::Lda { &m.w } else { &m.covs[g] };
                let ci = mvutil::sym_inv(cov);
                let d: Vec<f64> = (0..p).map(|j| x[j] - m.means[(g, j)]).collect();
                let q: f64 = (0..p).map(|a| (0..p).map(|b| d[a] * ci[(a, b)] * d[b]).sum::<f64>()).sum();
                let ld = if m.kind == Kind::Qda { mvutil::ln_det(cov) } else { 0.0 };
                -0.5 * q - 0.5 * ld + m.priors[g].ln()
            })
            .collect(),
        Kind::Logistic => {
            let mut s: Vec<f64> = (0..gn - 1).map(|g| m.lb[(g, p)] + (0..p).map(|j| m.lb[(g, j)] * x[j]).sum::<f64>()).collect();
            s.push(0.0);
            s
        }
        Kind::Knn => {
            let wi = if m.mahal { Some(mvutil::sym_inv(&m.w)) } else { None };
            let mut ds: Vec<(f64, usize)> = m
                .train
                .iter()
                .enumerate()
                .filter(|(i, _)| Some(*i) != skip)
                .map(|(i, t)| {
                    let d = match &wi {
                        Some(w) => {
                            let dd: Vec<f64> = (0..p).map(|j| x[j] - t[j]).collect();
                            (0..p).map(|a| (0..p).map(|b| dd[a] * w[(a, b)] * dd[b]).sum::<f64>()).sum::<f64>().sqrt()
                        }
                        None => {
                            let v = m.measure.eval(x, t, &[], &[]);
                            if m.measure.is_sim() { -v } else { v }
                        }
                    };
                    (d, m.train_g[i])
                })
                .collect();
            ds.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut cnt = vec![0.0; gn];
            let kk = m.k.min(ds.len());
            let cut = if kk > 0 { ds[kk - 1].0 } else { 0.0 };
            for &(d, g) in ds.iter().take_while(|(d, _)| *d <= cut + 1e-12) {
                let _ = d;
                cnt[g] += 1.0;
            }
            let ng: Vec<f64> = (0..gn).map(|g| m.train_g.iter().enumerate().filter(|(i, tg)| **tg == g && Some(*i) != skip).count() as f64).collect();
            (0..gn).map(|g| if cnt[g] > 0.0 { (cnt[g] / ng[g].max(1.0) * m.priors[g]).ln() } else { f64::NEG_INFINITY }).collect()
        }
    }
}

fn posterior(lp: &[f64]) -> Vec<f64> {
    let mx = lp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !mx.is_finite() {
        return vec![1.0 / lp.len() as f64; lp.len()];
    }
    let e: Vec<f64> = lp.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn classify(post: &[f64]) -> Option<usize> {
    let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let idx: Vec<usize> = (0..post.len()).filter(|&g| (post[g] - mx).abs() <= 1e-12).collect();
    if idx.len() == 1 { Some(idx[0]) } else { None }
}

/// Fit pieces shared by lda/qda/candisc: group means, pooled covariance.
fn group_stats(x: &[Vec<f64>], g: &[usize], gn: usize) -> (Matrix, Matrix, Vec<Matrix>, Vec<usize>) {
    let p = x[0].len();
    let n = x.len();
    let mut means = Matrix::zeros(gn, p);
    let mut cnt = vec![0usize; gn];
    for t in 0..n {
        cnt[g[t]] += 1;
        for j in 0..p {
            means[(g[t], j)] += x[t][j];
        }
    }
    for k in 0..gn {
        for j in 0..p {
            means[(k, j)] /= cnt[k] as f64;
        }
    }
    let mut e = Matrix::zeros(p, p);
    let mut covs = vec![Matrix::zeros(p, p); gn];
    for t in 0..n {
        for a in 0..p {
            for b in 0..p {
                let v = (x[t][a] - means[(g[t], a)]) * (x[t][b] - means[(g[t], b)]);
                e[(a, b)] += v;
                covs[g[t]][(a, b)] += v;
            }
        }
    }
    for k in 0..gn {
        covs[k] = covs[k].scale(1.0 / (cnt[k] as f64 - 1.0));
    }
    (means, e, covs, cnt)
}

fn mlogit_fit(x: &[Vec<f64>], g: &[usize], gn: usize) -> Matrix {
    // Newton-Raphson for multinomial logit with base = last group
    let n = x.len();
    let p = x[0].len() + 1;
    let k = (gn - 1) * p;
    let mut beta = vec![0.0; k];
    for _ in 0..100 {
        let mut grad = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        for t in 0..n {
            let xt: Vec<f64> = x[t].iter().cloned().chain(std::iter::once(1.0)).collect();
            let mut s: Vec<f64> = (0..gn - 1).map(|c| (0..p).map(|j| beta[c * p + j] * xt[j]).sum()).collect();
            s.push(0.0);
            let pr = posterior(&s);
            for c in 0..gn - 1 {
                let yc = if g[t] == c { 1.0 } else { 0.0 };
                for j in 0..p {
                    grad[c * p + j] += (yc - pr[c]) * xt[j];
                }
                for d in 0..gn - 1 {
                    let w = if c == d { pr[c] * (1.0 - pr[c]) } else { -pr[c] * pr[d] };
                    for a in 0..p {
                        for b in 0..p {
                            h[(c * p + a, d * p + b)] += w * xt[a] * xt[b];
                        }
                    }
                }
            }
        }
        let hi = mvutil::sym_inv(&h);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * grad[j]).sum()).collect();
        let sz: f64 = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for i in 0..k {
            beta[i] += step[i];
        }
        if sz < 1e-10 {
            break;
        }
    }
    let mut m = Matrix::zeros(gn - 1, p);
    for c in 0..gn - 1 {
        for j in 0..p {
            m[(c, j)] = beta[c * p + j];
        }
    }
    m
}

fn model_from_e(ds: &DataSet, e: &Results) -> Result<Model, String> {
    let kind = Kind::from(e.macros.get("subcmd").map(String::as_str).unwrap_or("lda")).unwrap_or(Kind::Lda);
    let vars: Vec<String> = e.macros.get("varlist").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let gm = e.matrices.get("groupvalues").cloned().unwrap_or_default();
    let glabels: Vec<String> = e.macros.get("grouplabels").map(|s| s.split('\u{1f}').map(String::from).collect()).unwrap_or_default();
    let priors = e.matrices.get("grouppriors").map(|m| m.data.clone()).unwrap_or_default();
    let means = e.matrices.get("means").cloned().unwrap_or_default().transpose();
    let w = e.matrices.get("W").cloned().unwrap_or_default();
    let gn = gm.data.len();
    let covs = (0..gn).filter_map(|g| e.matrices.get(&format!("S_{}", g + 1)).cloned()).collect();
    let lb = e.matrices.get("b_logit").cloned().unwrap_or_default();
    let mut train = Vec::new();
    let mut train_g = Vec::new();
    if kind == Kind::Knn {
        let sample = e.sample.clone().unwrap_or_default();
        let gv = e.macros.get("groupvar").cloned().unwrap_or_default();
        let gc = util::col(ds, &gv)?;
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        for i in 0..ds.n_obs.min(sample.len()) {
            if sample[i] {
                train.push(cols.iter().map(|c| c[i]).collect());
                train_g.push(gm.data.iter().position(|&v| v == gc[i]).unwrap_or(0));
            }
        }
    }
    Ok(Model {
        kind,
        vars,
        gvals: gm.data.clone(),
        glabels,
        priors,
        means,
        w,
        covs,
        lb,
        train,
        train_g,
        k: e.get("k_nn").unwrap_or(1.0) as usize,
        measure: e.macros.get("measure").and_then(|m| Measure::parse(m).ok()).unwrap_or(Measure::L2),
        mahal: e.macros.get("mahalanobis").is_some(),
    })
}

/// Classification table lines.
fn class_table(title: &str, glabels: &[String], gvar: &str, truth: &[usize], class: &[Option<usize>], priors: &[f64]) -> (Vec<String>, Matrix) {
    let gn = glabels.len();
    let mut cnt = Matrix::zeros(gn, gn + 1);
    for (t, c) in truth.iter().zip(class) {
        match c {
            Some(c) => cnt[(*t, *c)] += 1.0,
            None => cnt[(*t, gn)] += 1.0,
        }
    }
    let has_unc = (0..gn).any(|g| cnt[(g, gn)] > 0.0);
    let ncol = gn + if has_unc { 1 } else { 0 };
    let mut out = vec![String::new(), title.to_string(), String::new()];
    out.push("+---------+".into());
    out.push("| Key     |".into());
    out.push("|---------|".into());
    out.push("| Number  |".into());
    out.push("| Percent |".into());
    out.push("+---------+".into());
    out.push(String::new());
    let w = 12;
    let mut head = format!("{:>16} |", format!("True {}", crate::summarize::abbrev(gvar, 11)));
    for g in 0..gn {
        head.push_str(&format!(" {:>w$}", crate::summarize::abbrev(&glabels[g], w)));
    }
    if has_unc {
        head.push_str(&format!(" {:>w$}", "Unclassified"));
    }
    head.push_str(&format!(" | {:>8}", "Total"));
    out.push(format!("{:>16} | {:^width$}", "", "Classified", width = (w + 1) * ncol));
    out.push(head);
    let rule = format!("{}+{}+{}", "-".repeat(17), "-".repeat((w + 1) * ncol + 1), "-".repeat(10));
    out.push(rule.clone());
    let mut coltot = vec![0.0; ncol];
    for g in 0..gn {
        let tot: f64 = (0..=gn).map(|c| cnt[(g, c)]).sum();
        let mut l1 = format!("{:>16} |", crate::summarize::abbrev(&glabels[g], 16));
        let mut l2 = format!("{:>16} |", "");
        for c in 0..ncol {
            let v = cnt[(g, c)];
            coltot[c] += v;
            l1.push_str(&format!(" {:>w$}", v));
            l2.push_str(&format!(" {:>w$.2}", 100.0 * v / tot));
        }
        l1.push_str(&format!(" | {:>8}", tot));
        l2.push_str(&format!(" | {:>8.2}", 100.0));
        out.push(l1);
        out.push(l2);
        out.push(format!("{:>16} |{}|", "", " ".repeat((w + 1) * ncol + 1)));
    }
    out.pop();
    out.push(rule.clone());
    let n: f64 = coltot.iter().sum();
    let mut l1 = format!("{:>16} |", "Total");
    let mut l2 = format!("{:>16} |", "");
    for c in 0..ncol {
        l1.push_str(&format!(" {:>w$}", coltot[c]));
        l2.push_str(&format!(" {:>w$.2}", 100.0 * coltot[c] / n));
    }
    l1.push_str(&format!(" | {:>8}", n));
    l2.push_str(&format!(" | {:>8.2}", 100.0));
    out.push(l1);
    out.push(l2);
    out.push(rule.clone());
    let mut l3 = format!("{:>16} |", "Priors");
    for g in 0..gn {
        l3.push_str(&format!(" {:>w$.4}", priors[g]));
    }
    out.push(l3);
    (out, cnt)
}

pub struct Discrim {
    pub candisc: bool,
}

struct Data {
    rows: Vec<usize>,
    x: Vec<Vec<f64>>,
    g: Vec<usize>,
    gvals: Vec<f64>,
    glabels: Vec<String>,
    mask: Vec<bool>,
}

fn load(ds: &DataSet, st: &Store, cmd: &Command, varlist: &str, gvar: &str) -> Result<(Vec<String>, Data), String> {
    let vars = crate::varlist::expand(ds, varlist)?;
    let mut mask = util::touse(ds, st, cmd)?;
    let mut all = vars.clone();
    all.push(gvar.to_string());
    util::markout(ds, &mut mask, &all);
    let rows = util::rows(&mask);
    let gc = util::col(ds, gvar)?;
    let gv = ds.get_var(gvar).unwrap();
    let mut gvals: Vec<f64> = rows.iter().map(|&i| gc[i]).collect();
    gvals.sort_by(|a, b| a.total_cmp(b));
    gvals.dedup();
    if gvals.len() < 2 {
        return Err(rc_err(198, "at least two groups required"));
    }
    let glabels: Vec<String> = gvals.iter().map(|&v| util::cell_text(ds, gv, rows.iter().copied().find(|&i| gc[i] == v).unwrap(), true)).collect();
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let x: Vec<Vec<f64>> = rows.iter().map(|&i| cols.iter().map(|c| c[i]).collect()).collect();
    let g: Vec<usize> = rows.iter().map(|&i| gvals.iter().position(|&v| v == gc[i]).unwrap()).collect();
    Ok((vars, Data { rows, x, g, gvals, glabels, mask }))
}

fn parse_priors(cmd: &Command, cnt: &[usize]) -> Result<Vec<f64>, String> {
    let gn = cnt.len();
    let n: usize = cnt.iter().sum();
    match cmd.opt("priors", 3).map(|s| s.trim().to_lowercase()) {
        None => Ok(vec![1.0 / gn as f64; gn]),
        Some(s) if s.starts_with("eq") => Ok(vec![1.0 / gn as f64; gn]),
        Some(s) if s.starts_with("prop") => Ok(cnt.iter().map(|&c| c as f64 / n as f64).collect()),
        Some(s) => {
            let v: Vec<f64> = s.replace(',', " ").split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "invalid priors()"))).collect::<Result<_, _>>()?;
            if v.len() != gn {
                return Err(rc_err(198, "priors() has the wrong number of elements"));
            }
            let s: f64 = v.iter().sum();
            Ok(v.iter().map(|x| x / s).collect())
        }
    }
}

/// Canonical discriminant analysis pieces: eigenvalues, raw coefficients (p x f).
fn canon(e: &Matrix, means: &Matrix, cnt: &[usize], n: usize) -> (Vec<f64>, Matrix) {
    let p = e.rows;
    let gn = cnt.len();
    let grand: Vec<f64> = (0..p).map(|j| (0..gn).map(|g| means[(g, j)] * cnt[g] as f64).sum::<f64>() / n as f64).collect();
    let mut h = Matrix::zeros(p, p);
    for g in 0..gn {
        for a in 0..p {
            for b in 0..p {
                h[(a, b)] += cnt[g] as f64 * (means[(g, a)] - grand[a]) * (means[(g, b)] - grand[b]);
            }
        }
    }
    let l = e.cholesky().unwrap_or_else(|_| Matrix::identity(p));
    let li = l.inverse().unwrap_or_else(|_| Matrix::identity(p));
    let m = li.mul(&h).unwrap().mul(&li.transpose()).unwrap();
    let (vals, vecs) = mvutil::sym_eig(&m);
    let f = (gn - 1).min(p);
    let w_df = (n - gn) as f64;
    let mut coef = Matrix::zeros(p, f);
    let raw = li.transpose().mul(&vecs).unwrap();
    for k in 0..f {
        // scale: v' (E/(n-g)) v = 1
        let s = w_df.sqrt();
        for j in 0..p {
            coef[(j, k)] = raw[(j, k)] * s;
        }
    }
    // orientation: first group's mean score negative... use positive sum of standardized coefficient? keep last group mean positive
    for k in 0..f {
        let score_last: f64 = (0..p).map(|j| (means[(gn - 1, j)] - grand[j]) * coef[(j, k)]).sum();
        let score_first: f64 = (0..p).map(|j| (means[(0, j)] - grand[j]) * coef[(j, k)]).sum();
        let _ = score_last;
        if score_first > 0.0 {
            for j in 0..p {
                coef[(j, k)] = -coef[(j, k)];
            }
        }
    }
    (vals[..f].to_vec(), coef)
}

impl CommandHandler for Discrim {
    fn name(&self) -> &'static str {
        if self.candisc { "candisc" } else { "discrim" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (kind, varlist) = if self.candisc {
            (Kind::Lda, main.to_string())
        } else {
            let (s, r) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
            match Kind::from(&s.to_lowercase()) {
                Some(k) => (k, r.trim().to_string()),
                None => (Kind::Lda, main.to_string()),
            }
        };
        let gvar = cmd.opt("group", 2).map(|s| s.trim().to_string()).ok_or_else(|| rc_err(198, "option group() required"))?;
        let (vars, d) = load(ds, st, cmd, &varlist, &gvar)?;
        let gn = d.gvals.len();
        let p = vars.len();
        let n = d.rows.len();
        let (means, e, covs, cnt) = group_stats(&d.x, &d.g, gn);
        let priors = parse_priors(cmd, &cnt)?;
        let w = e.scale(1.0 / (n - gn) as f64);
        let lb = if kind == Kind::Logistic { mlogit_fit(&d.x, &d.g, gn) } else { Matrix::zeros(0, 0) };
        let k_nn = if kind == Kind::Knn { mvutil::opt_usize(cmd, "k", 1)?.ok_or_else(|| rc_err(198, "option k() required"))? } else { 0 };
        let measure = Measure::from_cmd(cmd, Measure::L2)?;
        let model = Model {
            kind,
            vars: vars.clone(),
            gvals: d.gvals.clone(),
            glabels: d.glabels.clone(),
            priors: priors.clone(),
            means: means.clone(),
            w: w.clone(),
            covs: covs.clone(),
            lb: lb.clone(),
            train: d.x.clone(),
            train_g: d.g.clone(),
            k: k_nn,
            measure,
            mahal: cmd.has_opt("mahalanobis", 4),
        };
        let class: Vec<Option<usize>> = d.x.iter().map(|x| classify(&posterior(&log_post(&model, x, None)))).collect();
        let gnames: Vec<String> = d.gvals.iter().map(|v| crate::format::fmt_g(*v, 9)).collect();
        let mut out = vec![String::new()];
        let title = if self.candisc { "Canonical linear discriminant analysis" } else { kind.title() };
        out.push(title.to_string());
        let (ev, coef) = if kind == Kind::Lda { canon(&e, &means, &cnt, n) } else { (vec![], Matrix::zeros(0, 0)) };
        if self.candisc {
            out.extend(canontest_lines(&ev, p, gn, n));
        }
        let (tbl, cmat) = class_table("Resubstitution classification summary", &d.glabels, &gvar, &d.g, &class, &priors);
        if self.candisc {
            let sd: Vec<f64> = (0..p).map(|j| w[(j, j)].sqrt()).collect();
            let mut stdc = coef.clone();
            for j in 0..p {
                for k in 0..coef.cols {
                    stdc[(j, k)] *= sd[j];
                }
            }
            out.push(String::new());
            out.push("Standardized canonical discriminant function coefficients".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&stdc.clone().with_names(vars.clone(), mvutil::names("function", coef.cols)), "", "%10.0g", 10, false, 4));
            let wc = w.mul(&coef).unwrap();
            let mut stru = wc.clone();
            for j in 0..p {
                for k in 0..coef.cols {
                    stru[(j, k)] = wc[(j, k)] / sd[j];
                }
            }
            out.push(String::new());
            out.push("Canonical structure".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&stru.with_names(vars.clone(), mvutil::names("function", coef.cols)), "", "%10.0g", 10, false, 4));
            out.push(String::new());
            out.push("Group means on canonical variables".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&grmeans_canon(&means, &coef, &cnt, n).with_names(d.glabels.clone(), mvutil::names("function", coef.cols)), &gvar, "%10.0g", 10, false, 4));
        }
        if !cmd.has_opt("notable", 5) {
            out.extend(tbl);
        }
        if cmd.has_opt("lootable", 4) {
            let loo = loo_class(&model, &d.x, &d.g);
            let (t2, _) = class_table("Leave-one-out classification summary", &d.glabels, &gvar, &d.g, &loo, &priors);
            out.extend(t2);
        }
        let es = st.begin_e();
        es.set("N", n as f64);
        es.set("N_groups", gn as f64);
        es.set("k", p as f64);
        es.set_str("cmd", if self.candisc { "candisc" } else { "discrim" });
        es.set_str("subcmd", kind.code());
        es.set_str("title", title);
        es.set_str("groupvar", gvar.clone());
        es.set_str("grouplabels", d.glabels.join("\u{1f}"));
        es.set_str("varlist", vars.join(" "));
        es.set_str("properties", "nob noV");
        es.set_str("estat_cmd", "discrim_estat");
        es.set_str("predict", "discrim_p");
        es.set_mat("groupvalues", Matrix::row_vector(&d.gvals).with_names(vec!["r1".into()], gnames.clone()));
        es.set_mat("groupcounts", Matrix::row_vector(&cnt.iter().map(|&c| c as f64).collect::<Vec<_>>()).with_names(vec!["r1".into()], gnames.clone()));
        es.set_mat("grouppriors", Matrix::row_vector(&priors).with_names(vec!["r1".into()], gnames.clone()));
        es.set_mat("means", means.transpose().with_names(vars.clone(), gnames.clone()));
        es.set_mat("W", w.clone().with_names(vars.clone(), vars.clone()));
        es.set_mat("ctable", cmat);
        for g in 0..gn {
            es.set_mat(&format!("S_{}", g + 1), covs[g].clone().with_names(vars.clone(), vars.clone()));
        }
        if kind == Kind::Lda {
            es.set("f", ev.len() as f64);
            es.set_mat("Ev", Matrix::row_vector(&ev).with_names(vec!["r1".into()], mvutil::names("function", ev.len())));
            es.set_mat("L_unstd", coef.clone().with_names(vars.clone(), mvutil::names("function", coef.cols)));
            es.set_mat("SSCP_W", e.clone().with_names(vars.clone(), vars.clone()));
        }
        if kind == Kind::Logistic {
            es.set_mat("b_logit", lb);
        }
        if kind == Kind::Knn {
            es.set("k_nn", k_nn as f64);
            es.set_str("measure", measure.name());
            if cmd.has_opt("mahalanobis", 4) {
                es.set_str("mahalanobis", "mahalanobis");
            }
        }
        es.sample = Some(d.mask.clone());
        Ok(out.join("\n"))
    }
}

fn loo_class(model: &Model, x: &[Vec<f64>], g: &[usize]) -> Vec<Option<usize>> {
    let n = x.len();
    let gn = model.gvals.len();
    (0..n)
        .map(|i| {
            if model.kind == Kind::Knn {
                return classify(&posterior(&log_post(model, &x[i], Some(i))));
            }
            let xs: Vec<Vec<f64>> = (0..n).filter(|&t| t != i).map(|t| x[t].clone()).collect();
            let gs: Vec<usize> = (0..n).filter(|&t| t != i).map(|t| g[t]).collect();
            let (means, e, covs, _) = group_stats(&xs, &gs, gn);
            let mut m2 = Model { means, w: e.scale(1.0 / (n - 1 - gn) as f64), covs, lb: model.lb.clone(), train: vec![], train_g: vec![], vars: vec![], gvals: model.gvals.clone(), glabels: vec![], priors: model.priors.clone(), kind: model.kind, k: model.k, measure: model.measure, mahal: model.mahal };
            if model.kind == Kind::Logistic {
                m2.lb = mlogit_fit(&xs, &gs, gn);
            }
            classify(&posterior(&log_post(&m2, &x[i], None)))
        })
        .collect()
}

fn grmeans_canon(means: &Matrix, coef: &Matrix, cnt: &[usize], n: usize) -> Matrix {
    let gn = means.rows;
    let p = means.cols;
    let grand: Vec<f64> = (0..p).map(|j| (0..gn).map(|g| means[(g, j)] * cnt[g] as f64).sum::<f64>() / n as f64).collect();
    let mut m = Matrix::zeros(gn, coef.cols);
    for g in 0..gn {
        for k in 0..coef.cols {
            m[(g, k)] = (0..p).map(|j| (means[(g, j)] - grand[j]) * coef[(j, k)]).sum();
        }
    }
    m
}

fn canontest_lines(ev: &[f64], p: usize, gn: usize, n: usize) -> Vec<String> {
    let mut out = vec![String::new()];
    out.push(format!("{:>6} {:>8} {:>10} {:>18} {:>9}", "", "Canon.", "Eigen-", "Variance", "Like-"));
    out.push(format!("{:>6} {:>8} {:>10} {:>18} {:>9}", "", "", "", "", "lihood"));
    out.push(format!("{:>6} {:>8} {:>10} {:>8} {:>9} {:>9} {:>9} {:>5} {:>7} {:>7}", "Fcn", "Corr.", "value", "Prop.", "Cumul.", "Ratio", "F", "df1", "df2", "Prob>F"));
    out.push("-".repeat(86));
    let tot: f64 = ev.iter().sum();
    let mut cum = 0.0;
    for k in 0..ev.len() {
        cum += ev[k];
        let lam: f64 = ev[k..].iter().map(|l| 1.0 / (1.0 + l)).product();
        let p1 = (p - k) as f64;
        let q1 = (gn - 1 - k) as f64;
        let wv = n as f64 - 1.0 - (p as f64 + gn as f64) / 2.0;
        let t = if p1 * p1 + q1 * q1 - 5.0 > 0.0 { ((p1 * p1 * q1 * q1 - 4.0) / (p1 * p1 + q1 * q1 - 5.0)).sqrt() } else { 1.0 };
        let df1 = p1 * q1;
        let df2 = wv * t - (p1 * q1 - 2.0) / 2.0;
        let lt = lam.powf(1.0 / t);
        let f = (1.0 - lt) / lt * df2 / df1;
        let exact = p1.min(q1) <= 2.0;
        out.push(format!(
            "{:>6} {:>8.4} {:>10} {:>8.4} {:>9.4} {:>9.4} {:>9} {:>5} {:>7} {:>7.4} {}",
            k + 1,
            (ev[k] / (1.0 + ev[k])).sqrt(),
            mvutil::sig(ev[k], 6),
            ev[k] / tot,
            cum / tot,
            lam,
            mvutil::sig(f, 5),
            df1,
            crate::format::fmt_g(df2, 7),
            dist::ftail(df1, df2, f),
            if exact { "e" } else { "a" }
        ));
    }
    out.push("-".repeat(86));
    out.push(" H0: This and smaller canon. corr. are zero;                        e = exact F, a = approximate F".into());
    out
}

// ---------------------------------------------------------------------------
// estat / predict
// ---------------------------------------------------------------------------

fn is_discrim(st: &Store) -> bool {
    matches!(st.e.macros.get("cmd").map(String::as_str), Some("discrim") | Some("candisc"))
}

fn sample_data(ds: &DataSet, st: &Store, cmd: &Command, model: &Model, use_sample: bool) -> Result<(Vec<usize>, Vec<Vec<f64>>, Vec<Option<usize>>), String> {
    let gvar = st.e.macros.get("groupvar").cloned().unwrap_or_default();
    let mut mask = if use_sample { st.e.sample.clone().unwrap_or_default() } else { vec![true; ds.n_obs] };
    let tu = util::touse(ds, st, cmd)?;
    for i in 0..mask.len().min(tu.len()) {
        mask[i] = mask[i] && tu[i];
    }
    util::markout(ds, &mut mask, &model.vars);
    let rows = util::rows(&mask);
    let cols: Vec<Vec<f64>> = model.vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let gc = util::col(ds, &gvar).ok();
    let x = rows.iter().map(|&i| cols.iter().map(|c| c[i]).collect()).collect();
    let g = rows.iter().map(|&i| gc.as_ref().and_then(|c| model.gvals.iter().position(|&v| v == c[i]))).collect();
    Ok((rows, x, g))
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if !is_discrim(st) {
        return None;
    }
    let r = (|| -> Result<Option<String>, String> {
        let mut model = model_from_e(ds, &st.e)?;
        if let Some(pr) = cmd.opt("priors", 3) {
            let cnt: Vec<usize> = st.e.matrices.get("groupcounts").map(|m| m.data.iter().map(|v| *v as usize).collect()).unwrap_or_default();
            let mut c2 = cmd.clone();
            c2.opts = vec![("priors".into(), Some(pr.to_string()))];
            model.priors = parse_priors(&c2, &cnt)?;
        }
        let gvar = st.e.macros.get("groupvar").cloned().unwrap_or_default();
        let gn = model.gvals.len();
        let p = model.vars.len();
        let n = st.e.get("N").unwrap_or(0.0) as usize;
        let mut out = Vec::new();
        match sub {
            "classtable" | "classt" | "classta" | "classtab" | "classtabl" => {
                let (_, x, g) = sample_data(ds, st, cmd, &model, true)?;
                let pairs: Vec<(usize, Vec<f64>)> = x.into_iter().zip(g).filter_map(|(x, g)| g.map(|g| (g, x))).collect();
                let truth: Vec<usize> = pairs.iter().map(|p| p.0).collect();
                let xs: Vec<Vec<f64>> = pairs.iter().map(|p| p.1.clone()).collect();
                let loo = cmd.has_opt("looclass", 3) || cmd.has_opt("loo", 3);
                let class = if loo { loo_class(&model, &xs, &truth) } else { xs.iter().map(|x| classify(&posterior(&log_post(&model, x, None)))).collect() };
                let (t, cm) = class_table(if loo { "Leave-one-out classification table" } else { "Resubstitution classification table" }, &model.glabels, &gvar, &truth, &class, &model.priors);
                out.extend(t);
                st.begin_r().set_mat("ctable", cm);
            }
            "errorrate" | "error" | "err" | "errorr" => {
                let (_, x, g) = sample_data(ds, st, cmd, &model, true)?;
                let pairs: Vec<(usize, Vec<f64>)> = x.into_iter().zip(g).filter_map(|(x, g)| g.map(|g| (g, x))).collect();
                let truth: Vec<usize> = pairs.iter().map(|p| p.0).collect();
                let xs: Vec<Vec<f64>> = pairs.iter().map(|p| p.1.clone()).collect();
                let loo = cmd.has_opt("looclass", 3) || cmd.has_opt("loo", 3);
                let class = if loo { loo_class(&model, &xs, &truth) } else { xs.iter().map(|x| classify(&posterior(&log_post(&model, x, None)))).collect() };
                let mut err = vec![0.0; gn];
                for g in 0..gn {
                    let idx: Vec<usize> = (0..truth.len()).filter(|&t| truth[t] == g && class[t].is_some()).collect();
                    let wrong = idx.iter().filter(|&&t| class[t] != Some(g)).count();
                    err[g] = wrong as f64 / idx.len().max(1) as f64;
                }
                let total: f64 = (0..gn).map(|g| err[g] * model.priors[g]).sum();
                out.push(String::new());
                out.push(format!("Error rate estimated by error count{}", if loo { " (leave-one-out)" } else { "" }));
                out.push(String::new());
                let mut bx = Boxed::new("", vec![model.glabels.iter().map(|l| (crate::summarize::abbrev(l, 11), 12)).collect(), vec![("Total".into(), 10)]]);
                let mut cells: Vec<String> = err.iter().map(|e| format!("{e:.4}")).collect();
                cells.push(format!("{total:.4}"));
                bx.row("Error rate", cells);
                let mut pc: Vec<String> = model.priors.iter().map(|p| format!("{p:.4}")).collect();
                pc.push(String::new());
                bx.row("Priors", pc);
                bx.stub_width = 12;
                out.extend(bx.render());
                let nunc = class.iter().filter(|c| c.is_none()).count();
                if nunc > 0 {
                    out.push(format!("Note: {nunc} observation{} not classified and {} not included in the table.", if nunc == 1 { " was" } else { "s were" }, if nunc == 1 { "is" } else { "are" }));
                }
                let mut m = Matrix::zeros(1, gn + 1);
                for g in 0..gn {
                    m[(0, g)] = err[g];
                }
                m[(0, gn)] = total;
                st.begin_r().set_mat("erate_count", m);
            }
            "grsummarize" | "grsum" | "grsumm" | "grsummarise" => {
                let (_, x, g) = sample_data(ds, st, cmd, &model, true)?;
                out.push(String::new());
                out.push(format!("Estimation sample {}", st.e.macros.get("cmd").cloned().unwrap_or_default()));
                out.push(format!("Summarized by {gvar}"));
                out.push(String::new());
                let mut m = Matrix::zeros(p + 1, gn + 1);
                for j in 0..p {
                    for gg in 0..=gn {
                        let vals: Vec<f64> = x.iter().zip(&g).filter(|(_, t)| gg == gn || **t == Some(gg)).map(|(r, _)| r[j]).collect();
                        m[(j, gg)] = util::mean(&vals);
                    }
                }
                for gg in 0..=gn {
                    m[(p, gg)] = g.iter().filter(|t| gg == gn || **t == Some(gg)).count() as f64;
                }
                let mut rn = model.vars.clone();
                rn.push("N".into());
                let mut cn = model.glabels.clone();
                cn.push("Total".into());
                out.push("Mean".into());
                out.extend(mvutil::matlist(&m.with_names(rn, cn), gvar.as_str(), "%10.0g", 11, false, 4));
            }
            "list" => {
                let (rows, x, g) = sample_data(ds, st, cmd, &model, false)?;
                let mis = cmd.has_opt("misclassified", 3);
                out.push(String::new());
                let mut head = format!("{:>6} | {:>10} {:>10} |", "Obs", "True", "Class.");
                for l in &model.glabels {
                    head.push_str(&format!(" {:>10}", crate::summarize::abbrev(l, 10)));
                }
                out.push(head);
                out.push("-".repeat(32 + 11 * gn));
                for (t, &i) in rows.iter().enumerate() {
                    let post = posterior(&log_post(&model, &x[t], None));
                    let c = classify(&post);
                    let wrong = g[t].is_some() && c != g[t];
                    if mis && !wrong {
                        continue;
                    }
                    let mut line = format!(
                        "{:>6} | {:>10} {:>10} |",
                        i + 1,
                        g[t].map(|v| crate::summarize::abbrev(&model.glabels[v], 8)).unwrap_or_default(),
                        format!("{}{}", c.map(|v| crate::summarize::abbrev(&model.glabels[v], 8)).unwrap_or_default(), if wrong { " *" } else { "  " })
                    );
                    for pp in post {
                        line.push_str(&format!(" {:>10.4}", pp));
                    }
                    out.push(line);
                }
                out.push("-".repeat(32 + 11 * gn));
                out.push("* indicates misclassified observations".into());
            }
            "loadings" | "load" | "loading" | "structure" | "canontest" | "grmeans" | "classfunctions" | "classfun" | "classfunction" | "manova" | "anova" => {
                if model.kind != Kind::Lda {
                    return Err(rc_err(321, format!("estat {sub} not valid after discrim {}", model.kind.code())));
                }
                let coef = st.e.matrices.get("L_unstd").cloned().unwrap_or_default();
                let ev = st.e.matrices.get("Ev").map(|m| m.data.clone()).unwrap_or_default();
                let w = model.w.clone();
                let sd: Vec<f64> = (0..p).map(|j| w[(j, j)].sqrt()).collect();
                let fnames = mvutil::names("function", coef.cols);
                let cnt: Vec<usize> = st.e.matrices.get("groupcounts").map(|m| m.data.iter().map(|v| *v as usize).collect()).unwrap_or_default();
                let grand: Vec<f64> = (0..p).map(|j| (0..gn).map(|g| model.means[(g, j)] * cnt[g] as f64).sum::<f64>() / n as f64).collect();
                let r = st.begin_r();
                match sub {
                    "canontest" => out.extend(canontest_lines(&ev, p, gn, n)),
                    "structure" => {
                        let wc = w.mul(&coef).unwrap();
                        let mut s = wc.clone();
                        for j in 0..p {
                            for k in 0..coef.cols {
                                s[(j, k)] = wc[(j, k)] / sd[j];
                            }
                        }
                        let s = s.with_names(model.vars.clone(), fnames.clone());
                        out.push(String::new());
                        out.push("Canonical structure".into());
                        out.push(String::new());
                        out.extend(mvutil::matlist(&s, "", "%10.0g", 10, false, 4));
                        r.set_mat("canstruct", s);
                    }
                    "grmeans" => {
                        let gm = grmeans_canon(&model.means, &coef, &cnt, n).with_names(model.glabels.clone(), fnames.clone());
                        out.push(String::new());
                        out.push("Group means on canonical variables".into());
                        out.push(String::new());
                        out.extend(mvutil::matlist(&gm, &gvar, "%10.0g", 10, false, 4));
                        r.set_mat("cmeans", gm);
                    }
                    "classfunctions" | "classfun" | "classfunction" => {
                        let wi = mvutil::sym_inv(&w);
                        let mut cf = Matrix::zeros(p + 1, gn);
                        for g in 0..gn {
                            let mw: Vec<f64> = (0..p).map(|a| (0..p).map(|b| wi[(a, b)] * model.means[(g, b)]).sum()).collect();
                            for j in 0..p {
                                cf[(j, g)] = mw[j];
                            }
                            cf[(p, g)] = -0.5 * (0..p).map(|j| mw[j] * model.means[(g, j)]).sum::<f64>() + model.priors[g].ln() * if cmd.has_opt("nopriors", 4) { 0.0 } else { 1.0 };
                        }
                        let mut rn = model.vars.clone();
                        rn.push("_cons".into());
                        let cf = cf.with_names(rn, model.glabels.clone());
                        out.push(String::new());
                        out.push("Classification functions".into());
                        out.push(String::new());
                        out.extend(mvutil::matlist(&cf, "", "%10.0g", 10, false, 4));
                        r.set_mat("C", cf);
                    }
                    "manova" | "anova" => return Ok(None),
                    _ => {
                        let unstd = cmd.has_opt("unstandardized", 3) || cmd.has_opt("all", 3);
                        let std_ = cmd.has_opt("standardized", 3) || cmd.has_opt("all", 3) || !unstd;
                        if unstd {
                            let mut u = Matrix::zeros(p + 1, coef.cols);
                            for k in 0..coef.cols {
                                for j in 0..p {
                                    u[(j, k)] = coef[(j, k)];
                                }
                                u[(p, k)] = -(0..p).map(|j| coef[(j, k)] * grand[j]).sum::<f64>();
                            }
                            let mut rn = model.vars.clone();
                            rn.push("_cons".into());
                            let u = u.with_names(rn, fnames.clone());
                            out.push(String::new());
                            out.push("Canonical discriminant function coefficients".into());
                            out.push(String::new());
                            out.extend(mvutil::matlist(&u, "", "%10.0g", 10, false, 4));
                            r.set_mat("L_unstd", u);
                        }
                        if std_ {
                            let mut s = coef.clone();
                            for j in 0..p {
                                for k in 0..coef.cols {
                                    s[(j, k)] *= sd[j];
                                }
                            }
                            let s = s.with_names(model.vars.clone(), fnames.clone());
                            out.push(String::new());
                            out.push("Standardized canonical discriminant function coefficients".into());
                            out.push(String::new());
                            out.extend(mvutil::matlist(&s, "", "%10.0g", 10, false, 4));
                            r.set_mat("L_std", s);
                        }
                    }
                }
            }
            _ => return Ok(None),
        }
        Ok(Some(out.join("\n")))
    })();
    match r {
        Ok(Some(s)) => Some(Ok(s)),
        Ok(None) => Some(Err(rc_err(321, format!("estat {sub} not implemented after discrim")))),
        Err(e) => Some(Err(e)),
    }
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    if !is_discrim(st) {
        return Ok(None);
    }
    let model = model_from_e(ds, &st.e)?;
    let gn = model.gvals.len();
    let (rows, x, _) = sample_data(ds, st, cmd, &model, false)?;
    let pr = cmd.has_opt("pr", 2);
    let (ty, names) = mvutil::new_var_names(ds, &cmd.main, if pr { gn } else { 1 })?;
    let n = ds.n_obs;
    let posts: Vec<Vec<f64>> = x.iter().map(|xx| posterior(&log_post(&model, xx, None))).collect();
    let mut msg = String::new();
    if pr {
        for (k, nm) in names.iter().enumerate().take(gn) {
            let vals: Vec<f64> = posts.iter().map(|p| p[k]).collect();
            mvutil::put_var(ds, st, nm, ty, n, &rows, &vals, &format!("Pr({})", model.glabels[k]))?;
        }
    } else {
        if !cmd.has_opt("classification", 2) {
            msg = "(option classification assumed; group classification)".into();
        }
        let vals: Vec<f64> = posts.iter().map(|p| classify(p).map(|g| model.gvals[g]).unwrap_or(SYSMISS)).collect();
        mvutil::put_var(ds, st, &names[0], ty, n, &rows, &vals, "Group classification")?;
    }
    st.data_changed = true;
    let _ = is_missing;
    Ok(Some(msg))
}
