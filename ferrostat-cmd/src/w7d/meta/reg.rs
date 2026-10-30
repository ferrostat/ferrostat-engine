//! Meta-regression: univariate (`meta regress`, with `estat bubbleplot`
//! and `predict`), multivariate (`meta mvregress`) and multilevel
//! (`meta meregress`, `meta multilevel`) random-effects models, and their
//! `estat heterogeneity`, `estat recovariance`, `estat sd`, `estat group`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::pool::{self, Model};
use super::summ::opts;
use super::settings;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

const LN2PI: f64 = 1.837_877_066_409_345_5;

fn xtwx(x: &[Vec<f64>], w: &[f64]) -> Matrix {
    let p = x[0].len();
    let mut m = Matrix::zeros(p, p);
    for (r, wi) in x.iter().zip(w) {
        for a in 0..p {
            for b in 0..p {
                m[(a, b)] += wi * r[a] * r[b];
            }
        }
    }
    m
}

fn gls(y: &[f64], x: &[Vec<f64>], w: &[f64]) -> Result<(Vec<f64>, Matrix), String> {
    let p = x[0].len();
    let m = xtwx(x, w);
    let mi = m.inverse().map_err(|_| rc_err(459, "moderators are collinear"))?;
    let mut xy = vec![0.0; p];
    for ((r, wi), yi) in x.iter().zip(w).zip(y) {
        for a in 0..p {
            xy[a] += wi * r[a] * yi;
        }
    }
    let b = (0..p).map(|a| (0..p).map(|c| mi[(a, c)] * xy[c]).sum()).collect();
    Ok((b, mi))
}

/// Residual heterogeneity statistic with fixed weights and the typical
/// within-study variance (k - p) / tr(P).
fn resid_q(y: &[f64], v: &[f64], x: &[Vec<f64>]) -> Result<(f64, f64, f64), String> {
    let w: Vec<f64> = v.iter().map(|a| 1.0 / a).collect();
    let (b, mi) = gls(y, x, &w)?;
    let q: f64 = (0..y.len()).map(|i| w[i] * (y[i] - dot(&x[i], &b)).powi(2)).sum();
    // tr(P) = sum w - tr((X'WX)^-1 X'W^2X)
    let x2 = xtwx(x, &w.iter().map(|a| a * a).collect::<Vec<_>>());
    let tr: f64 = w.iter().sum::<f64>() - mi.mul(&x2)?.trace();
    let df = y.len() as f64 - x[0].len() as f64;
    Ok((q, df, tr))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Between-study variance of a meta-regression.
pub fn mr_tau2(y: &[f64], v: &[f64], x: &[Vec<f64>], method: &str) -> Result<f64, String> {
    let (q, df, tr) = resid_q(y, v, x)?;
    let mm = ((q - df) / tr).max(0.0);
    Ok(match method {
        "ml" | "reml" => {
            let reml = method == "reml";
            let ll = |t: f64| -> f64 {
                let w: Vec<f64> = v.iter().map(|a| 1.0 / (a + t)).collect();
                let Ok((b, mi)) = gls(y, x, &w) else { return f64::NEG_INFINITY };
                let mut l = -0.5 * (0..y.len()).map(|i| -w[i].ln() + w[i] * (y[i] - dot(&x[i], &b)).powi(2)).sum::<f64>();
                if reml {
                    l += 0.5 * mi.det().ln();
                }
                l
            };
            let mut hi = (mm.max(1e-4)) * 8.0;
            while ll(hi * 2.0) > ll(hi) && hi < 1e12 {
                hi *= 2.0;
            }
            let (mut a, mut b) = (0.0, hi * 2.0);
            let gr = 0.618_033_988_749_895;
            for _ in 0..300 {
                let c = b - gr * (b - a);
                let d = a + gr * (b - a);
                if ll(c) > ll(d) {
                    b = d;
                } else {
                    a = c;
                }
                if b - a < 1e-14 * (1.0 + a) {
                    break;
                }
            }
            let t = 0.5 * (a + b);
            if ll(0.0) >= ll(t) { 0.0 } else { t }
        }
        "pm" | "eb" => {
            let g = |t: f64| -> f64 {
                let w: Vec<f64> = v.iter().map(|a| 1.0 / (a + t)).collect();
                match gls(y, x, &w) {
                    Ok((b, _)) => (0..y.len()).map(|i| w[i] * (y[i] - dot(&x[i], &b)).powi(2)).sum::<f64>() - df,
                    Err(_) => 0.0,
                }
            };
            if g(0.0) <= 0.0 {
                0.0
            } else {
                let mut hi = mm.max(1e-4);
                while g(hi) > 0.0 && hi < 1e12 {
                    hi *= 2.0;
                }
                let mut lo = 0.0;
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if g(mid) > 0.0 { lo = mid } else { hi = mid }
                }
                0.5 * (lo + hi)
            }
        }
        _ => mm,
    })
}

/// Fit a meta-regression: (b, V, tau2).
pub fn fit_mr(y: &[f64], v: &[f64], x: &[Vec<f64>], model: Model, method: &str) -> Result<(Vec<f64>, Matrix, f64), String> {
    if y.len() <= x[0].len() {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let t2 = if model == Model::Random { mr_tau2(y, v, x, method)? } else { 0.0 };
    let w: Vec<f64> = v.iter().map(|a| 1.0 / (a + t2)).collect();
    let (b, vb) = gls(y, x, &w)?;
    Ok((b, vb, t2))
}

/// Design of `_meta_es` on the moderators.
fn design(ds: &DataSet, st: &Store, cmd: &Command, mods: &str, nocons: bool) -> Result<(crate::est::Design, Vec<usize>), String> {
    let d = crate::est::build(ds, st, cmd, crate::est::DesignSpec { depvar: Some("_meta_es"), indep: mods, noconstant: nocons, weights: "", extra: vec!["_meta_se".into()] })?;
    let act = d.active();
    if act.is_empty() {
        return Err(rc_err(102, "no moderators and no constant"));
    }
    Ok((d, act))
}

pub fn regress(ds: &mut DataSet, st: &mut Store, cmd: &Command, mods: &str) -> Result<String, String> {
    let s = settings(ds)?;
    let mut o = opts(cmd, st, &s)?;
    if o.method == "mh" {
        o.method = "iv".into();
    }
    let nocons = cmd.has_opt("noconstant", 4);
    let (d, act) = design(ds, st, cmd, mods, nocons)?;
    let sec = util::col(ds, "_meta_se")?;
    let y = d.y.clone();
    let v: Vec<f64> = d.rows.iter().map(|&r| sec[r] * sec[r]).collect();
    let x: Vec<Vec<f64>> = (0..d.n()).map(|t| act.iter().map(|&j| d.cols[j][t]).collect()).collect();
    let (b, vb, t2) = fit_mr(&y, &v, &x, o.model, &o.method)?;
    let k = y.len();
    let (q, df, tr) = resid_q(&y, &v, &x)?;
    let s2 = df / tr;
    let i2 = if o.model == Model::Random { 100.0 * t2 / (t2 + s2) } else { (100.0 * (q - df) / q).max(0.0) };
    let h2 = if o.model == Model::Random { (t2 + s2) / s2 } else { (q / df).max(1.0) };
    // R-squared relative to the constant-only model
    let r2 = if o.model == Model::Random && !nocons && x[0].len() > 1 {
        let t0 = pool::tau2(&y, &v, if o.method == "iv" { "dl" } else { &o.method });
        let t0 = if matches!(o.method.as_str(), "reml" | "ml" | "pm" | "eb") { mr_tau2(&y, &v, &vec![vec![1.0]; k], &o.method)? } else { t0 };
        if t0 > 0.0 { (100.0 * (t0 - t2) / t0).max(0.0) } else { 0.0 }
    } else {
        SYSMISS
    };
    let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
    // model test: all moderators except the constant
    let idx: Vec<usize> = (0..names.len()).filter(|&j| names[j] != "_cons").collect();
    let (chi2, dfm) = crate::ts_ml::wald(&b, &vb, &idx);
    let mut header = vec![
        format!("{:<46}Number of obs  = {:>8}", if o.model == Model::Random { "Random-effects meta-regression" } else if o.model == Model::Common { "Common-effect meta-regression" } else { "Fixed-effects meta-regression" }, k),
        format!("{:<46}{}", format!("Method: {}", pool::method_label(&o.method)), if o.model == Model::Random { format!("Residual heterogeneity:") } else { String::new() }),
    ];
    if o.model == Model::Random {
        header.push(format!("{:<46}{:>14} = {:>8.4}", "", "tau2", t2));
        header.push(format!("{:<46}{:>14} = {:>8.2}", "", "I2 (%)", i2));
        header.push(format!("{:<46}{:>14} = {:>8.2}", "", "H2", h2));
        if !is_missing(r2) {
            header.push(format!("{:<46}{:>14} = {:>8.2}", "", "R-squared (%)", r2));
        }
    }
    if !idx.is_empty() {
        header.push(format!("{:<46}Wald chi2({}) = {:>8.2}", "", dfm as i64, chi2));
        header.push(format!("{:<46}Prob > chi2  = {:>8.4}", "", crate::dist::chi2tail(dfm, chi2)));
    }
    let t0 = header.remove(0);
    let text = crate::ts_ml::post(
        st,
        cmd,
        crate::ts_ml::Post { title: &t0, cmd: "meta regress", depvar: "_meta_es", names: names.clone(), b: b.clone(), v: vb.clone(), ll: SYSMISS, n: k, header, vcetype: "", vce: "gls", log: vec![], notes: vec![format!("Test of residual homogeneity: Q_res = chi2({}) = {:.2}   Prob > Q_res = {:.4}", df as i64, q, crate::dist::chi2tail(df.max(1.0), q))], converged: true, iters: 0, df: None },
        Some(&d.mask),
    )?;
        let e = &mut st.e;
    e.set_str("cmd", "meta regress");
    e.set_str("cmdline", format!("meta {}", cmd.main.trim()));
    e.set_str("indepvars", mods.trim());
    e.set_str("model", o.model.name());
    e.set_str("method", o.method.clone());
    e.set_str("estype", s.es.clone());
    e.set("tau2", t2);
    e.set("I2_res", i2);
    e.set("H2_res", h2);
    if !is_missing(r2) {
        e.set("R2", r2);
    }
    e.set("Q_res", q);
    e.set("df_Q_res", df);
    e.set("p_Q_res", crate::dist::chi2tail(df.max(1.0), q));
    e.set("chi2", chi2);
    e.set("df_m", dfm);
    e.set("N", k as f64);
    if nocons {
        e.set_str("noconstant", "noconstant");
    }
    Ok(text)
}

/// `predict` after meta regress.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] | [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mods = st.e.macros.get("indepvars").cloned().unwrap_or_default();
    let nocons = st.e.macros.contains_key("noconstant");
    let line = format!("regress _meta_es {mods}");
    let c2 = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
    let (d, act) = design(ds, st, &c2, &mods, nocons)?;
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let vb = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
    let t2 = st.e.get("tau2").unwrap_or(0.0);
    let stat = ["xb", "stdp", "fitted", "residuals", "reffects"].into_iter().find(|s| cmd.has_opt(s, 3)).unwrap_or("xb");
    let sec = util::col(ds, "_meta_se")?;
    let mask = util::touse(ds, st, cmd)?;
    let mut out = vec![SYSMISS; ds.n_obs];
    for t in 0..d.n() {
        let r = d.rows[t];
        if !mask[r] {
            continue;
        }
        let x: Vec<f64> = act.iter().map(|&j| d.cols[j][t]).collect();
        let xb = dot(&x, &b.data);
        let re = t2 / (t2 + sec[r] * sec[r]) * (d.y[t] - xb);
        out[r] = match stat {
            "stdp" => (0..x.len()).map(|a| (0..x.len()).map(|c| x[a] * vb[(a, c)] * x[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt(),
            "fitted" => xb + re,
            "residuals" => d.y[t] - xb,
            "reffects" => re,
            _ => xb,
        };
    }
    let nmiss = out.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, out)).map_err(|e| rc_err(110, e))?;
    Ok(if nmiss > 0 { format!("({nmiss} missing value{} generated)", if nmiss == 1 { "" } else { "s" }) } else { String::new() })
}

/// `estat bubbleplot` after meta regress.
fn bubbleplot(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let mods = st.e.macros.get("indepvars").cloned().unwrap_or_default();
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let names = b.colnames.iter().map(|n| n.rsplit(':').next().unwrap_or(n).to_string()).collect::<Vec<_>>();
    let want = cmd.main.split_whitespace().nth(1).map(String::from);
    let xvar = match want {
        Some(w) => crate::xmodels::var_name(ds, &w)?,
        None => names.iter().find(|n| *n != "_cons" && ds.var_index.contains_key(n.as_str())).cloned().ok_or_else(|| rc_err(198, "estat bubbleplot requires a continuous moderator"))?,
    };
    let j = names.iter().position(|n| *n == xvar).ok_or_else(|| rc_err(198, format!("{xvar} is not a moderator")))?;
    let mask = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let x = util::col(ds, &xvar)?;
    let y = util::col(ds, "_meta_es")?;
    let se = util::col(ds, "_meta_se")?;
    let t2 = st.e.get("tau2").unwrap_or(0.0);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(x[i]) && !is_missing(y[i])).collect();
    let w: Vec<f64> = rows.iter().map(|&i| 1.0 / (se[i] * se[i] + t2)).collect();
    let wmax = w.iter().cloned().fold(0.0, f64::max);
    // other moderators at their sample means
    let mut base = 0.0;
    for (c, n) in names.iter().enumerate() {
        if c == j {
            continue;
        }
        base += b.data[c] * if n == "_cons" { 1.0 } else { util::col(ds, n).map(|col| util::mean(&rows.iter().map(|&i| col[i]).collect::<Vec<_>>())).unwrap_or(0.0) };
    }
    let xmin = rows.iter().map(|&i| x[i]).fold(f64::INFINITY, f64::min);
    let xmax = rows.iter().map(|&i| x[i]).fold(f64::NEG_INFINITY, f64::max);
    let mut g = crate::graph::Graph::default();
    g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Scatter, points: rows.iter().map(|&i| (x[i], y[i])).collect(), sizes: w.iter().map(|a| 0.5 + 2.0 * (a / wmax).sqrt()).collect(), hollow: true, label: "Studies".into(), ..Default::default() });
    g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Line, points: vec![(xmin, base + b.data[j] * xmin), (xmax, base + b.data[j] * xmax)], label: "Linear prediction".into(), ..Default::default() });
    g.xtitle = xvar.clone();
    g.ytitle = "Effect size".into();
    g.title = "Bubble plot".into();
    g.legend = true;
    g.command = format!("estat bubbleplot {}", xvar);
    let _ = mods;
    crate::graph::finish(&mut g, cmd, st)
}

// ---------------------------------------------------------------------------
// multivariate meta-regression
// ---------------------------------------------------------------------------

struct Mv {
    y: Vec<Vec<f64>>,
    s: Vec<Matrix>,
    x: Vec<Vec<f64>>,
    d: usize,
}

impl Mv {
    /// Stacked design for study i: d x (d p)
    fn xi(&self, i: usize) -> Matrix {
        let p = self.x[i].len();
        let mut m = Matrix::zeros(self.d, self.d * p);
        for a in 0..self.d {
            for c in 0..p {
                m[(a, a * p + c)] = self.x[i][c];
            }
        }
        m
    }
    /// GLS given Sigma: (beta, V, sum ln|S+Sigma|, quadratic form, ln|X'WX|)
    fn gls(&self, sig: &Matrix) -> Option<(Vec<f64>, Matrix, f64, f64, f64)> {
        let q = self.d * self.x[0].len();
        let mut a = Matrix::zeros(q, q);
        let mut bv = vec![0.0; q];
        let mut ldet = 0.0;
        let mut ws = Vec::new();
        for i in 0..self.y.len() {
            let vi = self.s[i].add(sig).ok()?;
            let det = vi.det();
            if !(det > 0.0) {
                return None;
            }
            ldet += det.ln();
            let wi = vi.inverse().ok()?;
            let xi = self.xi(i);
            let xtw = xi.transpose().mul(&wi).ok()?;
            a = a.add(&xtw.mul(&xi).ok()?).ok()?;
            for r in 0..q {
                bv[r] += (0..self.d).map(|c| xtw[(r, c)] * self.y[i][c]).sum::<f64>();
            }
            ws.push(wi);
        }
        let ai = a.inverse().ok()?;
        let beta: Vec<f64> = (0..q).map(|r| (0..q).map(|c| ai[(r, c)] * bv[c]).sum()).collect();
        let mut quad = 0.0;
        for i in 0..self.y.len() {
            let xi = self.xi(i);
            let res: Vec<f64> = (0..self.d).map(|r| self.y[i][r] - (0..q).map(|c| xi[(r, c)] * beta[c]).sum::<f64>()).collect();
            quad += (0..self.d).map(|r| (0..self.d).map(|c| res[r] * ws[i][(r, c)] * res[c]).sum::<f64>()).sum::<f64>();
        }
        Some((beta, ai, ldet, quad, a.det().ln()))
    }
}

fn chol_to_sigma(th: &[f64], d: usize) -> Matrix {
    let mut l = Matrix::zeros(d, d);
    let mut k = 0;
    for i in 0..d {
        for j in 0..=i {
            l[(i, j)] = if i == j { th[k].exp() } else { th[k] };
            k += 1;
        }
    }
    l.mul(&l.transpose()).unwrap()
}

pub fn mvregress(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (lhs, rhs) = match rest.split_once('=') {
        Some((a, b)) => (a.trim().to_string(), b.trim().to_string()),
        None => (rest.trim().to_string(), String::new()),
    };
    let ys = crate::varlist::expand(ds, &lhs).map_err(|e| rc_err(111, e))?;
    let d = ys.len();
    if d < 2 {
        return Err(rc_err(102, "at least two dependent variables required"));
    }
    let xs = if rhs.is_empty() { vec![] } else { crate::varlist::expand(ds, &rhs).map_err(|e| rc_err(111, e))? };
    let wcov: Vec<String> = match (cmd.opt("wcovvariables", 4), cmd.opt("wsevariables", 4)) {
        (Some(w), _) => crate::varlist::expand(ds, w).map_err(|e| rc_err(111, e))?,
        (None, Some(w)) => crate::varlist::expand(ds, w).map_err(|e| rc_err(111, e))?,
        _ => return Err(rc_err(198, "wcovvariables() or wsevariables() required")),
    };
    let se_only = cmd.opt("wcovvariables", 4).is_none();
    if (se_only && wcov.len() != d) || (!se_only && wcov.len() != d * (d + 1) / 2) {
        return Err(rc_err(198, format!("wcovvariables() requires {} variables (variances and covariances, row by row of the upper triangle)", d * (d + 1) / 2)));
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let all: Vec<String> = ys.iter().chain(&xs).chain(&wcov).cloned().collect();
    util::markout(ds, &mut mask, &all);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let yc: Vec<Vec<f64>> = ys.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let xc: Vec<Vec<f64>> = xs.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let wc: Vec<Vec<f64>> = wcov.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let nocons = cmd.has_opt("noconstant", 4);
    let mut mv = Mv { y: vec![], s: vec![], x: vec![], d };
    let wcorr = util::opt_num(cmd, "wcorrelation", 5)?.unwrap_or(0.0);
    for &r in &rows {
        mv.y.push(yc.iter().map(|c| c[r]).collect());
        let mut x: Vec<f64> = xc.iter().map(|c| c[r]).collect();
        if !nocons {
            x.push(1.0);
        }
        mv.x.push(x);
        let mut s = Matrix::zeros(d, d);
        if se_only {
            for a in 0..d {
                for b in 0..d {
                    s[(a, b)] = wc[a][r] * wc[b][r] * if a == b { 1.0 } else { wcorr };
                }
            }
        } else {
            let mut k = 0;
            for a in 0..d {
                for b in a..d {
                    s[(a, b)] = wc[k][r];
                    s[(b, a)] = wc[k][r];
                    k += 1;
                }
            }
        }
        mv.s.push(s);
    }
    let k = rows.len();
    let p = mv.x[0].len();
    if p == 0 {
        return Err(rc_err(102, "no moderators and no constant"));
    }
    if k <= p + 1 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let model = if cmd.has_opt("fixed", 3) || cmd.has_opt("common", 4) { Model::Fixed } else { Model::Random };
    let method = cmd.opt("random", 4).map(|m| m.trim().to_lowercase()).unwrap_or_else(|| "reml".into());
    let method = match method.as_str() {
        "" | "reml" => "reml",
        "ml" | "mle" => "ml",
        "jwriley" | "mm" => "jwriley",
        o => return Err(rc_err(198, format!("random({o}) not allowed"))),
    };
    let zero = Matrix::zeros(d, d);
    let sigma = if model == Model::Fixed {
        zero.clone()
    } else if method == "jwriley" {
        mm_sigma(&mv)
    } else {
        let reml = method == "reml";
        let npar = d * (d + 1) / 2;
        let f = |th: &[f64]| -> f64 {
            let sg = chol_to_sigma(th, d);
            match mv.gls(&sg) {
                Some((_, _, ld, q, la)) => -0.5 * (ld + q) - if reml { 0.5 * la } else { 0.0 },
                None => f64::NEG_INFINITY,
            }
        };
        let s0 = mm_sigma(&mv);
        let mut th0 = vec![0.0; npar];
        let mut kk = 0;
        for i in 0..d {
            for j in 0..=i {
                th0[kk] = if i == j { (s0[(i, i)].max(1e-4)).sqrt().ln() } else { 0.0 };
                kk += 1;
            }
        }
        let (th, _) = crate::w7d::opt::maximize(&f, &th0);
        chol_to_sigma(&th, d)
    };
    let (beta, vb, ld, quad, la) = mv.gls(&sigma).ok_or_else(|| rc_err(430, "model could not be fit"))?;
    let (_, _, _, qf, _) = mv.gls(&zero).ok_or_else(|| rc_err(430, "model could not be fit"))?;
    let nobs = k * d;
    let df_q = (nobs - d * p) as f64;
    let reml_c = if method == "reml" && model == Model::Random { (nobs - d * p) as f64 } else { nobs as f64 };
    let ll = -0.5 * (reml_c * LN2PI + ld + quad) - if method == "reml" && model == Model::Random { 0.5 * la } else { 0.0 };
    let mut xnames: Vec<String> = xs.clone();
    if !nocons {
        xnames.push("_cons".into());
    }
    let names: Vec<String> = ys.iter().flat_map(|y| xnames.iter().map(move |x| format!("{y}:{x}"))).collect();
    let idx: Vec<usize> = (0..names.len()).filter(|&j| !names[j].ends_with(":_cons")).collect();
    let (chi2, dfm) = crate::ts_ml::wald(&beta, &vb, &idx);
    let mut header = vec![
        format!("{:<46}Number of obs     = {:>7}", format!("Multivariate {} meta-regression", if model == Model::Fixed { "fixed-effects" } else { "random-effects" }), nobs),
        format!("{:<46}Number of studies = {:>7}", format!("Method: {}", match (model, method) { (Model::Fixed, _) => "Fixed", (_, "ml") => "ML", (_, "jwriley") => "Method of moments", _ => "REML" }), k),
    ];
    if !idx.is_empty() {
        header.push(format!("{:<46}Wald chi2({}) = {:>9.2}", "", dfm as i64, chi2));
        header.push(format!("{:<46}Prob > chi2 = {:>9.4}", "", crate::dist::chi2tail(dfm, chi2)));
    }
    if model == Model::Random && method != "jwriley" {
        header.push(format!("Log {}likelihood = {}", if method == "reml" { "restricted-" } else { "" }, fmt_g(ll, 9)));
    }
    let mut notes = vec![format!("Test of homogeneity: Q_M = chi2({}) = {:.2}{:>14}Prob > Q_M = {:.4}", df_q as i64, qf, "", crate::dist::chi2tail(df_q.max(1.0), qf))];
    if model == Model::Random {
        notes.insert(0, String::new());
        let mut re = vec!["Random-effects parameters | Estimate".to_string()];
        for a in 0..d {
            re.push(format!("{:>25} | {:>9.5}", format!("sd({})", ys[a]), sigma[(a, a)].max(0.0).sqrt()));
        }
        for a in 0..d {
            for b in a + 1..d {
                let r = sigma[(a, b)] / (sigma[(a, a)] * sigma[(b, b)]).sqrt();
                re.push(format!("{:>25} | {:>9.5}", format!("corr({},{})", ys[a], ys[b]), if r.is_finite() { r } else { 0.0 }));
            }
        }
        notes.splice(1..1, re);
    }
    let t0 = header.remove(0);
    let text = crate::ts_ml::post(st, cmd, crate::ts_ml::Post { title: &t0, cmd: "meta mvregress", depvar: &ys.join(" "), names, b: beta, v: vb, ll, n: nobs, header, vcetype: "", vce: "gls", log: vec![], notes, converged: true, iters: 0, df: None }, Some(&mask))?;
    let e = &mut st.e;
    e.set_str("cmd", "meta mvregress");
    e.set_str("depvars", ys.join(" "));
    e.set_str("indepvars", xs.join(" "));
    e.set_str("model", model.name());
    e.set_str("method", method);
    e.set_str("wcovvars", wcov.join(" "));
    e.set("N", nobs as f64);
    e.set("N_s", k as f64);
    e.set("d", d as f64);
    e.set("Q_M", qf);
    e.set("df_Q_M", df_q);
    e.set("p_Q_M", crate::dist::chi2tail(df_q.max(1.0), qf));
    e.set_mat("Sigma", sigma.with_names(ys.clone(), ys.clone()));
    // typical within-study variances per outcome, for heterogeneity
    let typ: Vec<f64> = (0..d)
        .map(|a| {
            let v: Vec<f64> = mv.s.iter().map(|s| s[(a, a)]).collect();
            let w: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
            let sw: f64 = w.iter().sum();
            let sw2: f64 = w.iter().map(|x| x * x).sum();
            (k as f64 - 1.0) * sw / (sw * sw - sw2)
        })
        .collect();
    e.set_mat("_typical", Matrix::row_vector(&typ));
    Ok(text)
}

/// Method-of-moments between-study covariance: univariate
/// DerSimonian-Laird variances on the diagonal and moment covariances
/// (cross-products of deviations from the unweighted means less the mean
/// within-study covariance), projected to be positive semidefinite.
fn mm_sigma(mv: &Mv) -> Matrix {
    let d = mv.d;
    let k = mv.y.len() as f64;
    let mut s = Matrix::zeros(d, d);
    let mean: Vec<f64> = (0..d).map(|a| mv.y.iter().map(|r| r[a]).sum::<f64>() / k).collect();
    for a in 0..d {
        let y: Vec<f64> = mv.y.iter().map(|r| r[a]).collect();
        let v: Vec<f64> = mv.s.iter().map(|m| m[(a, a)]).collect();
        s[(a, a)] = pool::tau2(&y, &v, "dl");
    }
    for a in 0..d {
        for b in a + 1..d {
            let c = mv.y.iter().map(|r| (r[a] - mean[a]) * (r[b] - mean[b])).sum::<f64>() / (k - 1.0) - mv.s.iter().map(|m| m[(a, b)]).sum::<f64>() / k;
            let lim = (s[(a, a)] * s[(b, b)]).sqrt();
            let c = c.clamp(-lim, lim);
            s[(a, b)] = c;
            s[(b, a)] = c;
        }
    }
    let (ev, vecs) = s.sym_eigen();
    let mut out = Matrix::zeros(d, d);
    for (l, e) in ev.iter().enumerate() {
        let e = e.max(0.0);
        for a in 0..d {
            for b in 0..d {
                out[(a, b)] += e * vecs[(a, l)] * vecs[(b, l)];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// multilevel meta-regression
// ---------------------------------------------------------------------------

struct Ml {
    y: Vec<f64>,
    v: Vec<f64>,
    x: Vec<Vec<f64>>,
    /// group id per level (outermost first) for each observation
    g: Vec<Vec<usize>>,
    /// observations of each outermost group
    blocks: Vec<Vec<usize>>,
}

impl Ml {
    fn fit(&self, sig2: &[f64], reml: bool) -> Option<(Vec<f64>, Matrix, f64)> {
        let p = self.x[0].len();
        let mut a = Matrix::zeros(p, p);
        let mut bv = vec![0.0; p];
        let mut ldet = 0.0;
        let mut winv = Vec::new();
        for blk in &self.blocks {
            let n = blk.len();
            let mut vm = Matrix::zeros(n, n);
            for (i, &oi) in blk.iter().enumerate() {
                for (j, &oj) in blk.iter().enumerate() {
                    let mut c = if i == j { self.v[oi] } else { 0.0 };
                    for (l, s2) in sig2.iter().enumerate() {
                        if self.g[oi][l] == self.g[oj][l] {
                            c += s2;
                        }
                    }
                    vm[(i, j)] = c;
                }
            }
            let det = vm.det();
            if !(det > 0.0) {
                return None;
            }
            ldet += det.ln();
            let wi = vm.inverse().ok()?;
            for (i, &oi) in blk.iter().enumerate() {
                for (j, &oj) in blk.iter().enumerate() {
                    let w = wi[(i, j)];
                    for r in 0..p {
                        bv[r] += self.x[oi][r] * w * self.y[oj];
                        for c in 0..p {
                            a[(r, c)] += self.x[oi][r] * w * self.x[oj][c];
                        }
                    }
                }
            }
            winv.push(wi);
        }
        let ai = a.inverse().ok()?;
        let beta: Vec<f64> = (0..p).map(|r| (0..p).map(|c| ai[(r, c)] * bv[c]).sum()).collect();
        let mut quad = 0.0;
        for (bi, blk) in self.blocks.iter().enumerate() {
            let res: Vec<f64> = blk.iter().map(|&o| self.y[o] - dot(&self.x[o], &beta)).collect();
            for i in 0..blk.len() {
                for j in 0..blk.len() {
                    quad += res[i] * winv[bi][(i, j)] * res[j];
                }
            }
        }
        let n = self.y.len() as f64;
        let mut ll = -0.5 * (ldet + quad);
        if reml {
            ll -= 0.5 * a.det().ln() + 0.5 * (n - p as f64) * LN2PI;
        } else {
            ll -= 0.5 * n * LN2PI;
        }
        Some((beta, ai, ll))
    }
}

pub fn meregress(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str, multilevel: bool) -> Result<String, String> {
    let parts: Vec<&str> = rest.split("||").collect();
    let fixed_part = parts[0].trim();
    let mut levels: Vec<String> = Vec::new();
    if multilevel {
        let r = cmd.opt("relevels", 3).ok_or_else(|| rc_err(198, "relevels() required"))?;
        levels = crate::varlist::expand(ds, r).map_err(|e| rc_err(111, e))?;
    } else {
        for p in &parts[1..] {
            let lv = p.split(':').next().unwrap_or("").trim();
            if lv.is_empty() {
                return Err(rc_err(198, "random-effects level required: || levelvar:"));
            }
            levels.push(crate::xmodels::var_name(ds, lv)?);
        }
    }
    if levels.is_empty() {
        return Err(rc_err(198, "at least one random-effects level required"));
    }
    let vars = crate::varlist::expand(ds, fixed_part).map_err(|e| rc_err(111, e))?;
    let (esvar, mods): (String, Vec<String>) = match vars.split_first() {
        Some((a, b)) => (a.clone(), b.to_vec()),
        None => match settings(ds) {
            Ok(_) => ("_meta_es".into(), vec![]),
            Err(e) => return Err(e),
        },
    };
    let sevar = match cmd.opt("essevariable", 4) {
        Some(v) => crate::xmodels::var_name(ds, v)?,
        None if ds.var_index.contains_key("_meta_se") => "_meta_se".into(),
        None => return Err(rc_err(198, "essevariable() required")),
    };
    let mut mask = util::touse(ds, st, cmd)?;
    let all: Vec<String> = [esvar.clone(), sevar.clone()].into_iter().chain(mods.clone()).chain(levels.clone()).collect();
    util::markout(ds, &mut mask, &all);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    if rows.len() < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let y = util::col(ds, &esvar)?;
    let se = util::col(ds, &sevar)?;
    let xc: Vec<Vec<f64>> = mods.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let lc: Vec<Vec<String>> = levels
        .iter()
        .map(|v| {
            let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
            Ok((0..ds.n_obs).map(|i| util::cell_text(ds, var, i, false)).collect())
        })
        .collect::<Result<_, String>>()?;
    let nocons = cmd.has_opt("noconstant", 4);
    // group ids: nested combinations of the level variables
    let nl = levels.len();
    let mut keys: Vec<Vec<Vec<String>>> = vec![Vec::new(); nl];
    let mut g = Vec::new();
    for &r in &rows {
        let mut ids = Vec::new();
        for l in 0..nl {
            let key: Vec<String> = (0..=l).map(|m| lc[m][r].clone()).collect();
            let pos = match keys[l].iter().position(|k| *k == key) {
                Some(p) => p,
                None => {
                    keys[l].push(key);
                    keys[l].len() - 1
                }
            };
            ids.push(pos);
        }
        g.push(ids);
    }
    let ng: Vec<usize> = keys.iter().map(|k| k.len()).collect();
    let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); ng[0]];
    for (o, ids) in g.iter().enumerate() {
        blocks[ids[0]].push(o);
    }
    let ml = Ml {
        y: rows.iter().map(|&r| y[r]).collect(),
        v: rows.iter().map(|&r| se[r] * se[r]).collect(),
        x: rows.iter().map(|&r| {
            let mut x: Vec<f64> = xc.iter().map(|c| c[r]).collect();
            if !nocons {
                x.push(1.0);
            }
            x
        }).collect(),
        g,
        blocks,
    };
    if ml.x[0].is_empty() {
        return Err(rc_err(102, "no moderators and no constant"));
    }
    let reml = !cmd.has_opt("mle", 3);
    let f = |th: &[f64]| -> f64 {
        let s2: Vec<f64> = th.iter().map(|t| (2.0 * t).exp()).collect();
        ml.fit(&s2, reml).map_or(f64::NEG_INFINITY, |r| r.2)
    };
    let yv = util::variance(&ml.y);
    let th0 = vec![(yv.max(1e-4) / nl as f64).sqrt().ln(); nl];
    let (th, ll) = crate::w7d::opt::maximize(&f, &th0);
    let s2: Vec<f64> = th.iter().map(|t| (2.0 * t).exp()).collect();
    let (beta, vb, _) = ml.fit(&s2, reml).ok_or_else(|| rc_err(430, "model could not be fit"))?;
    // standard errors of the log standard deviations
    let lnsd_se: Vec<f64> = match crate::w7d::opt::info(&f, &th) {
        Some(h) => {
            let (hi, _) = h.sym_inverse(1e-12);
            (0..nl).map(|j| hi[(j, j)].max(0.0).sqrt()).collect()
        }
        None => vec![SYSMISS; nl],
    };
    let mut xnames = mods.clone();
    if !nocons {
        xnames.push("_cons".into());
    }
    let idx: Vec<usize> = (0..xnames.len()).filter(|&j| xnames[j] != "_cons").collect();
    let (chi2, dfm) = crate::ts_ml::wald(&beta, &vb, &idx);
    let n = rows.len();
    let mut header = vec![format!("{:<46}Number of obs     = {:>7}", "Multilevel REML meta-analysis".replace("REML", if reml { "REML" } else { "ML" }).replace("meta-analysis", if mods.is_empty() { "meta-analysis" } else { "meta-regression" }), n)];
    header.push(String::new());
    header.push("-------------------------------------------------------------".into());
    header.push("                |     No. of       Observations per group".into());
    header.push(" Group variable |     groups    Minimum    Average    Maximum".into());
    header.push("----------------+--------------------------------------------".into());
    let mut grp = Matrix::zeros(nl, 4);
    for l in 0..nl {
        let mut cnt = vec![0usize; ng[l]];
        for ids in &ml.g {
            cnt[ids[l]] += 1;
        }
        let mn = *cnt.iter().min().unwrap_or(&0);
        let mx = *cnt.iter().max().unwrap_or(&0);
        let avg = n as f64 / ng[l] as f64;
        header.push(format!("{:>15} | {:>10} {:>10} {:>10.1} {:>10}", levels[l], ng[l], mn, avg, mx));
        for (c, v) in [ng[l] as f64, mn as f64, avg, mx as f64].iter().enumerate() {
            grp[(l, c)] = *v;
        }
    }
    header.push("-------------------------------------------------------------".into());
    header.push(String::new());
    if !idx.is_empty() {
        header.push(format!("{:<46}Wald chi2({}) = {:>9.2}", "", dfm as i64, chi2));
        header.push(format!("{:<46}Prob > chi2 = {:>9.4}", "", crate::dist::chi2tail(dfm, chi2)));
    }
    header.push(format!("Log {}likelihood = {}", if reml { "restricted-" } else { "" }, fmt_g(ll, 9)));
    let mut notes = vec![String::new(), "Random-effects parameters |   Estimate   Std. err.".to_string()];
    for l in 0..nl {
        let sd = s2[l].sqrt();
        notes.push(format!("{:>25} | {:>10.5}  {:>9.5}", format!("{}: sd(_cons)", levels[l]), sd, sd * lnsd_se[l]));
    }
    let wv: Vec<f64> = ml.v.iter().map(|v| 1.0 / v).collect();
    let (q, dfq, tr) = resid_q(&ml.y, &ml.v, &ml.x)?;
    let _ = wv;
    notes.push(format!("Test of homogeneity: Q_M = chi2({}) = {:.2}{:>14}Prob > Q_M = {:.4}", dfq as i64, q, "", crate::dist::chi2tail(dfq.max(1.0), q)));
    let t0 = header.remove(0);
    let text = crate::ts_ml::post(st, cmd, crate::ts_ml::Post { title: &t0, cmd: "meta meregress", depvar: &esvar, names: xnames, b: beta, v: vb, ll, n, header, vcetype: "", vce: "oim", log: vec![], notes, converged: true, iters: 0, df: None }, Some(&mask))?;
    let e = &mut st.e;
    e.set_str("cmd", if multilevel { "meta multilevel" } else { "meta meregress" });
    e.set_str("relevels", levels.join(" "));
    e.set_str("indepvars", mods.join(" "));
    e.set_str("method", if reml { "reml" } else { "mle" });
    e.set("k_r", nl as f64);
    e.set("Q_M", q);
    e.set("df_Q_M", dfq);
    e.set("p_Q_M", crate::dist::chi2tail(dfq.max(1.0), q));
    e.set("typical", dfq / tr);
    e.set_mat("sigma2", Matrix::row_vector(&s2).with_names(vec!["sigma2".into()], levels.clone()));
    e.set_mat("_lnsd_se", Matrix::row_vector(&lnsd_se));
    e.set_mat("N_g", grp.with_names(levels.clone(), ["groups", "min", "avg", "max"].iter().map(|s| s.to_string()).collect()));
    Ok(text)
}

// ---------------------------------------------------------------------------
// estat after the meta-regressions
// ---------------------------------------------------------------------------

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if !ecmd.starts_with("meta ") {
        if sub == "bubbleplot" || sub == "bubble" {
            return Some(Err(rc_err(321, "estat bubbleplot is available only after meta regress")));
        }
        return None;
    }
    let mv = ecmd == "meta mvregress";
    let me = ecmd == "meta meregress" || ecmd == "meta multilevel";
    let r = match sub {
        "bubbleplot" | "bubble" if ecmd == "meta regress" => bubbleplot(ds, st, cmd),
        "heterogeneity" | "het" | "hete" | "heter" | "hetero" if mv || me => Ok(heterogeneity(st, mv)),
        "recovariance" | "recov" | "recova" if mv || me => Ok(recov(st, mv)),
        "sd" if mv || me => Ok(sd(st, mv)),
        "group" if me => {
            let g = st.e.matrices.get("N_g").cloned().unwrap_or_else(|| Matrix::zeros(0, 4));
            let mut out = vec!["-------------------------------------------------------------".to_string(), "                |     No. of       Observations per group".into(), " Group variable |     groups    Minimum    Average    Maximum".into(), "----------------+--------------------------------------------".into()];
            for (i, l) in g.rownames.iter().enumerate() {
                out.push(format!("{:>15} | {:>10} {:>10} {:>10.1} {:>10}", l, g[(i, 0)], g[(i, 1)], g[(i, 2)], g[(i, 3)]));
            }
            out.push("-------------------------------------------------------------".into());
            st.begin_r().set_mat("N_g", g);
            Ok(out.join("\n"))
        }
        "summarize" | "ic" | "vce" => return None,
        _ => Err(rc_err(321, format!("estat {sub} not valid after {ecmd}"))),
    };
    Some(r)
}

fn heterogeneity(st: &mut Store, mv: bool) -> String {
    let e = st.e.clone();
    let q = e.get("Q_M").unwrap_or(SYSMISS);
    let df = e.get("df_Q_M").unwrap_or(SYSMISS);
    let mut out = vec![String::new()];
    let r = st.begin_r();
    if mv {
        let sig = e.matrices.get("Sigma").cloned().unwrap();
        let typ = e.matrices.get("_typical").cloned().unwrap();
        let ys: Vec<String> = sig.rownames.clone();
        let i2q = if q > 0.0 { (100.0 * (q - df) / q).max(0.0) } else { 0.0 };
        out.push("Method: Cochran's Q and Jackson-White-Riley-style I2 per outcome".into());
        out.push(String::new());
        out.push(format!("Overall: Q_M = {:.2}, df = {}, I2 (%) = {:.2}, H2 = {:.2}", q, df as i64, i2q, (q / df).max(1.0)));
        let mut i2 = Vec::new();
        for (a, y) in ys.iter().enumerate() {
            let t = sig[(a, a)];
            let v = 100.0 * t / (t + typ.data[a]);
            out.push(format!("  {y:>12}: I2 (%) = {:>6.2}", v));
            i2.push(v);
        }
        r.set("Q", q);
        r.set("df_Q", df);
        r.set("I2", i2q);
        r.set_mat("I2_outcomes", Matrix::row_vector(&i2).with_names(vec!["I2".into()], ys));
    } else {
        let s2 = e.matrices.get("sigma2").cloned().unwrap();
        let typ = e.get("typical").unwrap_or(0.0);
        let tot: f64 = s2.data.iter().sum::<f64>();
        out.push("Method: Higgins-Thompson-style multilevel I2".into());
        out.push(String::new());
        let mut v = Vec::new();
        for (l, name) in s2.colnames.iter().enumerate() {
            let i2 = 100.0 * s2.data[l] / (tot + typ);
            out.push(format!("  {:>12}: I2 (%) = {:>6.2}", name, i2));
            v.push(i2);
        }
        let i2t = 100.0 * tot / (tot + typ);
        out.push(format!("  {:>12}: I2 (%) = {:>6.2}", "Total", i2t));
        r.set("I2_total", i2t);
        r.set("Q", q);
        r.set_mat("I2", Matrix::row_vector(&v).with_names(vec!["I2".into()], s2.colnames.clone()));
    }
    out.join("\n")
}

fn recov(st: &mut Store, mv: bool) -> String {
    let e = st.e.clone();
    let mut out = vec![String::new(), "Between-study covariance matrix:".to_string()];
    let r = st.begin_r();
    if mv {
        let sig = e.matrices.get("Sigma").cloned().unwrap();
        let w = 12;
        out.push(format!("{:>w$} {}", "", sig.colnames.iter().map(|c| format!("{c:>12}")).collect::<String>()));
        for (i, rn) in sig.rownames.iter().enumerate() {
            out.push(format!("{:>w$} {}", rn, (0..sig.cols).map(|j| format!("{:>12.6}", sig[(i, j)])).collect::<String>()));
        }
        r.set_mat("cov", sig);
    } else {
        let s2 = e.matrices.get("sigma2").cloned().unwrap();
        for (l, name) in s2.colnames.iter().enumerate() {
            out.push(format!("  {name}: var(_cons) = {:.6}", s2.data[l]));
            r.set_mat(&format!("Cov{}", l + 1), Matrix::from_rows(&[vec![s2.data[l]]]).with_names(vec!["_cons".into()], vec!["_cons".into()]));
        }
    }
    out.join("\n")
}

fn sd(st: &mut Store, mv: bool) -> String {
    let e = st.e.clone();
    let mut out = vec![String::new(), "Random-effects parameters | Estimate".to_string()];
    let r = st.begin_r();
    if mv {
        let sig = e.matrices.get("Sigma").cloned().unwrap();
        let d = sig.rows;
        let mut sds = Vec::new();
        for a in 0..d {
            let s = sig[(a, a)].max(0.0).sqrt();
            out.push(format!("{:>25} | {:>9.5}", format!("sd({})", sig.rownames[a]), s));
            sds.push(s);
        }
        let mut corr = Matrix::identity(d);
        for a in 0..d {
            for b in 0..d {
                if a != b {
                    let c = sig[(a, b)] / (sds[a] * sds[b]);
                    corr[(a, b)] = if c.is_finite() { c } else { 0.0 };
                }
            }
        }
        for a in 0..d {
            for b in a + 1..d {
                out.push(format!("{:>25} | {:>9.5}", format!("corr({},{})", sig.rownames[a], sig.rownames[b]), corr[(a, b)]));
            }
        }
        r.set_mat("sd", Matrix::row_vector(&sds).with_names(vec!["sd".into()], sig.rownames.clone()));
        r.set_mat("corr", corr.with_names(sig.rownames.clone(), sig.rownames.clone()));
    } else {
        let s2 = e.matrices.get("sigma2").cloned().unwrap();
        let sds: Vec<f64> = s2.data.iter().map(|v| v.sqrt()).collect();
        for (l, name) in s2.colnames.iter().enumerate() {
            out.push(format!("{:>25} | {:>9.5}", format!("{name}: sd(_cons)"), sds[l]));
        }
        r.set_mat("sd", Matrix::row_vector(&sds).with_names(vec!["sd".into()], s2.colnames.clone()));
    }
    out.join("\n")
}

pub fn is_meta_regress(st: &Store) -> bool {
    st.e.macros.get("cmd").is_some_and(|c| c == "meta regress")
}
