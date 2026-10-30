//! Financial tools (`fin`).
//!
//! ```text
//! fin                                          list of subcommands
//! fin returns pricevar, generate(newvar) [log]  simple (or log) returns in
//!                                               time order (within panels)
//! fin summarize retvars [, periods(#) rf(#)]    mean, volatility, annualized
//!                                               figures, Sharpe ratio,
//!                                               skewness, kurtosis, maximum drawdown
//! fin portfolio retvars [, rf(#) target(#) noshort points(#) graph]
//!                                               global minimum-variance,
//!                                               tangency and target-return
//!                                               portfolios, efficient frontier
//! fin regress capm retvar mktvar [, rf(var|#)]  CAPM regression of excess returns
//! fin regress fmb retvar factorvars [, lags(#)] Fama-MacBeth two-pass
//!                                               cross-sectional regressions on
//!                                               panel data (xtset or tsset panel time)
//! fin valrisk retvar [, method(historical|parametric|montecarlo)
//!                       level(#) horizon(#) value(#) reps(#) rseed(#)]
//!                                               value at risk and expected shortfall
//! ```

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

pub struct Fin;

impl CommandHandler for Fin {
    fn name(&self) -> &'static str {
        "fin"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
        match sub.to_lowercase().as_str() {
            "" => Ok([
                "Financial tools:",
                "  fin returns     returns from prices",
                "  fin summarize   return and risk summaries",
                "  fin portfolio   mean-variance portfolios and the efficient frontier",
                "  fin regress     capm (CAPM regression) and fmb (Fama-MacBeth regressions)",
                "  fin valrisk     value at risk and expected shortfall",
            ]
            .join("\n")),
            "returns" | "ret" => returns(ds, st, cmd, &rest),
            "summarize" | "sum" | "summ" => summarize(ds, st, cmd, &rest),
            "portfolio" | "port" => portfolio(ds, st, cmd, &rest),
            "regress" | "reg" => {
                let (m, r2) = crate::xmodels::first_word(&rest);
                match m.to_lowercase().as_str() {
                    "capm" => capm(ds, st, cmd, &r2),
                    "fmb" => fmb(ds, st, cmd, &r2),
                    o => Err(rc_err(198, format!("fin regress {o} is unrecognized; use capm or fmb"))),
                }
            }
            "valrisk" | "var" => valrisk(ds, st, cmd, &rest),
            o => Err(rc_err(199, format!("fin {o} is unrecognized"))),
        }
    }
}

/// Rows of the sample in (panel, time) order.
fn ordered_rows(ds: &DataSet, st: &Store, mask: &[bool]) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>), String> {
    let (pv, tv) = st.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset varname ..."))?;
    let t = util::col(ds, &tv)?;
    let p = match pv {
        Some(p) if !p.is_empty() => util::col(ds, &p)?,
        _ => vec![0.0; ds.n_obs],
    };
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(t[i])).collect();
    rows.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(t[a].total_cmp(&t[b])));
    Ok((rows, p, t))
}

fn returns(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let price = crate::xmodels::var_name(ds, rest.trim())?;
    let newv = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "generate() required"))?.trim().to_string();
    let mask = util::touse(ds, st, cmd)?;
    let (rows, p, t) = ordered_rows(ds, st, &mask)?;
    let pr = util::col(ds, &price)?;
    let log = cmd.has_opt("log", 3);
    let mut out = vec![SYSMISS; ds.n_obs];
    for w in rows.windows(2) {
        let (a, b) = (w[0], w[1]);
        if p[a] != p[b] || (t[b] - t[a] - 1.0).abs() > 1e-9 || is_missing(pr[a]) || is_missing(pr[b]) || pr[a] <= 0.0 {
            continue;
        }
        out[b] = if log { (pr[b] / pr[a]).ln() } else { pr[b] / pr[a] - 1.0 };
    }
    let vals: Vec<f64> = out.iter().copied().filter(|v| !is_missing(*v)).collect();
    crate::tsops::add_var(ds, &newv, out)?;
    if let Some(v) = ds.get_var_mut(&newv) {
        v.label = format!("{} returns of {price}", if log { "Log" } else { "Simple" });
    }
    let r = st.begin_r();
    r.set("N", vals.len() as f64);
    r.set("mean", if vals.is_empty() { SYSMISS } else { util::mean(&vals) });
    Ok(format!("({} return{} generated)", vals.len(), if vals.len() == 1 { "" } else { "s" }))
}

fn series(ds: &DataSet, st: &Store, cmd: &Command, rest: &str) -> Result<(Vec<String>, Vec<Vec<f64>>, Vec<usize>), String> {
    let vars = crate::varlist::expand(ds, rest).map_err(|e| rc_err(111, e))?;
    if vars.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &vars);
    let rows: Vec<usize> = match &st.tsset {
        Some(_) => ordered_rows(ds, st, &mask)?.0,
        None => (0..ds.n_obs).filter(|&i| mask[i]).collect(),
    };
    if rows.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&r| c[r]).collect())).collect::<Result<_, _>>()?;
    Ok((vars, cols, rows))
}

fn moments(x: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let m = util::mean(x);
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    (m, util::variance(x).sqrt(), m3 / m2.powf(1.5), m4 / (m2 * m2))
}

/// Largest peak-to-trough fall of the compounded wealth index.
pub fn max_drawdown(r: &[f64]) -> f64 {
    let mut w = 1.0f64;
    let mut peak = 1.0f64;
    let mut dd = 0.0f64;
    for v in r {
        w *= 1.0 + v;
        peak = peak.max(w);
        dd = dd.max(1.0 - w / peak);
    }
    dd
}

fn summarize(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (vars, cols, _) = series(ds, st, cmd, rest)?;
    let per = util::opt_num(cmd, "periods", 3)?.unwrap_or(1.0);
    let rf = util::opt_num(cmd, "rf", 2)?.unwrap_or(0.0);
    let mut out = vec![String::new(), format!("Return summary (periods per year = {}, risk-free rate per period = {})", fmt_g(per, 6), fmt_g(rf, 6)), String::new()];
    out.push(format!("{:>12} | {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}", "Variable", "Obs", "Mean", "Std. dev.", "Ann. mean", "Ann. vol.", "Sharpe", "Skewn.", "Kurt.", "Max DD"));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(90)));
    let mut rows = Vec::new();
    for (v, c) in vars.iter().zip(&cols) {
        let (m, s, sk, ku) = moments(c);
        let sh = (m - rf) / s * per.sqrt();
        let dd = max_drawdown(c);
        out.push(format!("{:>12} | {:>6} {:>9.5} {:>9.5} {:>9.5} {:>9.5} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", crate::summarize::abbrev(v, 12), c.len(), m, s, m * per, s * per.sqrt(), sh, sk, ku, dd));
        rows.push(vec![c.len() as f64, m, s, m * per, s * per.sqrt(), sh, sk, ku, dd]);
    }
    st.begin_r().set_mat("stats", Matrix::from_rows(&rows).with_names(vars, ["N", "mean", "sd", "ann_mean", "ann_sd", "sharpe", "skewness", "kurtosis", "maxdd"].iter().map(|s| s.to_string()).collect()));
    Ok(out.join("\n"))
}

/// Minimum-variance weights for a target return (None: unconstrained in
/// the target) over the assets `idx`, with weights summing to one.
fn minvar(mu: &[f64], sig: &Matrix, idx: &[usize], target: Option<f64>) -> Option<Vec<f64>> {
    let k = idx.len();
    let mut s = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            s[(a, b)] = sig[(i, j)];
        }
    }
    let si = s.inverse().ok()?;
    let one = vec![1.0; k];
    let m: Vec<f64> = idx.iter().map(|&i| mu[i]).collect();
    let sv = |v: &[f64]| -> Vec<f64> { (0..k).map(|a| (0..k).map(|b| si[(a, b)] * v[b]).sum()).collect() };
    let s1 = sv(&one);
    let sm = sv(&m);
    let a: f64 = s1.iter().sum();
    let w = match target {
        None => s1.iter().map(|v| v / a).collect::<Vec<f64>>(),
        Some(t) => {
            let b: f64 = sm.iter().sum();
            let c: f64 = m.iter().zip(&sm).map(|(x, y)| x * y).sum();
            let d = a * c - b * b;
            if d.abs() < 1e-14 {
                if k == 1 && (m[0] - t).abs() < 1e-12 {
                    vec![1.0]
                } else {
                    return None;
                }
            } else {
                (0..k).map(|i| ((c - b * t) * s1[i] + (a * t - b) * sm[i]) / d).collect()
            }
        }
    };
    let mut full = vec![0.0; mu.len()];
    for (a, &i) in idx.iter().enumerate() {
        full[i] = w[a];
    }
    Some(full)
}

/// Minimum variance with optional no-short-sale constraint (exact search
/// over the faces of the simplex for small numbers of assets).
fn minvar_c(mu: &[f64], sig: &Matrix, target: Option<f64>, noshort: bool) -> Option<Vec<f64>> {
    let n = mu.len();
    if !noshort {
        return minvar(mu, sig, &(0..n).collect::<Vec<_>>(), target);
    }
    let var = |w: &[f64]| -> f64 { (0..n).map(|i| (0..n).map(|j| w[i] * sig[(i, j)] * w[j]).sum::<f64>()).sum() };
    let mut best: Option<(f64, Vec<f64>)> = None;
    for m in 1u32..(1 << n) {
        let idx: Vec<usize> = (0..n).filter(|&i| m >> i & 1 == 1).collect();
        if let Some(w) = minvar(mu, sig, &idx, target) {
            if w.iter().all(|v| *v >= -1e-10) {
                let v = var(&w);
                if best.as_ref().is_none_or(|b| v < b.0) {
                    best = Some((v, w));
                }
            }
        }
    }
    best.map(|b| b.1.iter().map(|v| v.max(0.0)).collect())
}

fn portfolio(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (vars, cols, _) = series(ds, st, cmd, rest)?;
    let n = vars.len();
    if n < 2 {
        return Err(rc_err(102, "at least two assets required"));
    }
    let noshort = cmd.has_opt("noshort", 4);
    if noshort && n > 16 {
        return Err(rc_err(198, "noshort allows at most 16 assets"));
    }
    let rf = util::opt_num(cmd, "rf", 2)?.unwrap_or(0.0);
    let mu: Vec<f64> = cols.iter().map(|c| util::mean(c)).collect();
    let t = cols[0].len() as f64;
    let mut sig = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sig[(i, j)] = cols[i].iter().zip(&cols[j]).map(|(a, b)| (a - mu[i]) * (b - mu[j])).sum::<f64>() / (t - 1.0);
        }
    }
    if sig.inverse().is_err() {
        return Err(rc_err(506, "covariance matrix of returns is singular"));
    }
    let stats = |w: &[f64]| -> (f64, f64, f64) {
        let m: f64 = w.iter().zip(&mu).map(|(a, b)| a * b).sum();
        let v: f64 = (0..n).map(|i| (0..n).map(|j| w[i] * sig[(i, j)] * w[j]).sum::<f64>()).sum();
        (m, v.sqrt(), (m - rf) / v.sqrt())
    };
    let gmv = minvar_c(&mu, &sig, None, noshort).ok_or_else(|| rc_err(430, "minimum-variance portfolio could not be computed"))?;
    // tangency: maximize Sharpe ratio
    let tang = if noshort {
        let (lo, hi) = (stats(&gmv).0, mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let mut best = gmv.clone();
        for i in 0..=400 {
            let tr = lo + (hi - lo) * i as f64 / 400.0;
            if let Some(w) = minvar_c(&mu, &sig, Some(tr), true) {
                if stats(&w).2 > stats(&best).2 {
                    best = w;
                }
            }
        }
        best
    } else {
        let si = sig.inverse().unwrap();
        let ex: Vec<f64> = mu.iter().map(|m| m - rf).collect();
        let z: Vec<f64> = (0..n).map(|i| (0..n).map(|j| si[(i, j)] * ex[j]).sum()).collect();
        let s: f64 = z.iter().sum();
        z.iter().map(|v| v / s).collect()
    };
    let target = util::opt_num(cmd, "target", 3)?;
    let tw = match target {
        Some(tg) => Some(minvar_c(&mu, &sig, Some(tg), noshort).ok_or_else(|| rc_err(430, "target return not attainable"))?),
        None => None,
    };
    let mut out = vec![String::new(), format!("Mean-variance portfolios ({} assets, {} periods{}){}", n, t, if noshort { ", no short sales" } else { "" }, if rf != 0.0 { format!(", risk-free rate {}", fmt_g(rf, 6)) } else { String::new() }), String::new()];
    let mut heads = vec!["Min. variance".to_string(), "Tangency".to_string()];
    let mut ws = vec![gmv.clone(), tang.clone()];
    if let Some(w) = &tw {
        heads.push(format!("Target {}", fmt_g(target.unwrap(), 6)));
        ws.push(w.clone());
    }
    out.push(format!("{:>12} | {}", "Asset", heads.iter().map(|h| format!("{h:>15}")).collect::<String>()));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(15 * heads.len())));
    for i in 0..n {
        out.push(format!("{:>12} | {}", crate::summarize::abbrev(&vars[i], 12), ws.iter().map(|w| format!("{:>15.5}", w[i])).collect::<String>()));
    }
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(15 * heads.len())));
    for (lab, k) in [("Mean", 0), ("Std. dev.", 1), ("Sharpe", 2)] {
        out.push(format!("{:>12} | {}", lab, ws.iter().map(|w| {
            let s = stats(w);
            format!("{:>15.5}", [s.0, s.1, s.2][k])
        }).collect::<String>()));
    }
    // efficient frontier from the minimum-variance return to the largest mean
    let npts = util::opt_num(cmd, "points", 3)?.unwrap_or(25.0).max(2.0) as usize;
    let (lo, hi) = (stats(&gmv).0, mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * if noshort { 1.0 } else { 1.5 });
    let mut fr = Vec::new();
    for i in 0..npts {
        let tr = lo + (hi - lo) * i as f64 / (npts - 1) as f64;
        if let Some(w) = minvar_c(&mu, &sig, Some(tr), noshort) {
            let s = stats(&w);
            fr.push(vec![s.0, s.1]);
        }
    }
    let r = st.begin_r();
    r.set_mat("gmv", Matrix::row_vector(&gmv).with_names(vec!["weight".into()], vars.clone()));
    r.set_mat("tangency", Matrix::row_vector(&tang).with_names(vec!["weight".into()], vars.clone()));
    if let Some(w) = &tw {
        r.set_mat("target", Matrix::row_vector(w).with_names(vec!["weight".into()], vars.clone()));
    }
    r.set_mat("frontier", Matrix::from_rows(&fr).with_names((1..=fr.len()).map(|i| i.to_string()).collect(), vec!["mean".into(), "sd".into()]));
    r.set_mat("mu", Matrix::row_vector(&mu).with_names(vec!["mean".into()], vars.clone()));
    r.set_mat("Sigma", sig.clone().with_names(vars.clone(), vars.clone()));
    let (gm, gs, _) = stats(&gmv);
    let (tm, ts, tsh) = stats(&tang);
    r.set("gmv_mean", gm);
    r.set("gmv_sd", gs);
    r.set("tan_mean", tm);
    r.set("tan_sd", ts);
    r.set("sharpe", tsh);
    if cmd.has_opt("graph", 2) {
        let mut g = crate::graph::Graph::default();
        g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Line, points: fr.iter().map(|v| (v[1], v[0])).collect(), label: "Efficient frontier".into(), ..Default::default() });
        g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Scatter, points: (0..n).map(|i| (sig[(i, i)].sqrt(), mu[i])).collect(), mlabels: vars.clone(), label: "Assets".into(), ..Default::default() });
        g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Scatter, points: vec![(gs, gm), (ts, tm)], mlabels: vec!["GMV".into(), "Tangency".into()], label: "Portfolios".into(), ..Default::default() });
        g.xtitle = "Standard deviation".into();
        g.ytitle = "Mean return".into();
        g.title = "Efficient frontier".into();
        g.legend = true;
        g.command = format!("fin portfolio {rest}");
        crate::graph::finish(&mut g, cmd, st)?;
    }
    Ok(out.join("\n"))
}

/// OLS with a constant: (b, V) for columns `x` (constant last).
fn ols(y: &[f64], x: &[Vec<f64>]) -> Result<(Vec<f64>, Matrix, f64), String> {
    let n = y.len();
    let p = x.len() + 1;
    let row = |i: usize| -> Vec<f64> {
        let mut r: Vec<f64> = x.iter().map(|c| c[i]).collect();
        r.push(1.0);
        r
    };
    let mut xtx = Matrix::zeros(p, p);
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let r = row(i);
        for a in 0..p {
            xty[a] += r[a] * y[i];
            for b in 0..p {
                xtx[(a, b)] += r[a] * r[b];
            }
        }
    }
    let xi = xtx.inverse().map_err(|_| rc_err(459, "regressors are collinear"))?;
    let b: Vec<f64> = (0..p).map(|a| (0..p).map(|c| xi[(a, c)] * xty[c]).sum()).collect();
    let rss: f64 = (0..n).map(|i| (y[i] - row(i).iter().zip(&b).map(|(u, v)| u * v).sum::<f64>()).powi(2)).sum();
    let s2 = rss / (n as f64 - p as f64);
    Ok((b, xi.scale(s2), rss))
}

fn table(st: &mut Store, depvar: &str, names: &[String], b: &[f64], v: &Matrix, df: f64) -> Vec<String> {
    let z = vec![false; b.len()];
    crate::est::coef_table(&crate::est::TableSpec { depvar, names, b, v, omitted: &z, is_base: &z, df: Some(df), level: st.level, eform: None, coef_title: "Coefficient" })
}

fn post(st: &mut Store, sub: &str, depvar: &str, names: &[String], b: &[f64], v: &Matrix, n: usize, mask: Vec<bool>) {
    let e = st.begin_e();
    e.set_str("cmd", format!("fin regress {sub}"));
    e.set_str("depvar", depvar);
    e.set("N", n as f64);
    e.set_mat("b", Matrix::row_vector(b).with_names(vec!["y1".into()], names.to_vec()));
    e.set_mat("V", v.clone().with_names(names.to_vec(), names.to_vec()));
    e.sample = Some(mask);
}

fn capm(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let vars = crate::varlist::expand(ds, rest).map_err(|e| rc_err(111, e))?;
    if vars.len() != 2 {
        return Err(rc_err(198, "fin regress capm requires retvar mktvar"));
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let mut need = vars.clone();
    let rfv = match cmd.opt("rf", 2) {
        Some(t) => match t.trim().parse::<f64>() {
            Ok(v) => Err(v),
            Err(_) => {
                let n = crate::xmodels::var_name(ds, t)?;
                need.push(n.clone());
                Ok(n)
            }
        },
        None => Err(0.0),
    };
    util::markout(ds, &mut mask, &need);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    if rows.len() < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let r = util::col(ds, &vars[0])?;
    let m = util::col(ds, &vars[1])?;
    let rf: Vec<f64> = match &rfv {
        Ok(n) => util::col(ds, n)?,
        Err(v) => vec![*v; ds.n_obs],
    };
    let y: Vec<f64> = rows.iter().map(|&i| r[i] - rf[i]).collect();
    let x: Vec<f64> = rows.iter().map(|&i| m[i] - rf[i]).collect();
    let (b, v, rss) = ols(&y, &[x.clone()])?;
    let n = rows.len();
    let tss: f64 = {
        let my = util::mean(&y);
        y.iter().map(|a| (a - my).powi(2)).sum()
    };
    let names = vec![vars[1].clone(), "_cons".into()];
    let mut out = vec![String::new(), format!("{:<48}Number of obs   = {:>8}", "CAPM regression (excess returns)", n)];
    out.push(format!("{:<48}R-squared       = {:>8.4}", "", 1.0 - rss / tss));
    out.push(String::new());
    out.extend(table(st, &vars[0], &names, &b, &v, n as f64 - 2.0));
    out.push(format!("beta = {:.6}  (systematic risk); alpha = {:.6} per period", b[0], b[1]));
    let mut dmask = vec![false; ds.n_obs];
    for &i in &rows {
        dmask[i] = true;
    }
    post(st, "capm", &vars[0], &names, &b, &v, n, dmask);
    let e = &mut st.e;
    e.set("beta", b[0]);
    e.set("alpha", b[1]);
    e.set("r2", 1.0 - rss / tss);
    e.set("sigma_e", (rss / (n as f64 - 2.0)).sqrt());
    Ok(out.join("\n"))
}

fn fmb(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let vars = crate::varlist::expand(ds, rest).map_err(|e| rc_err(111, e))?;
    if vars.len() < 2 {
        return Err(rc_err(198, "fin regress fmb requires retvar and at least one factor variable"));
    }
    let tv = match &st.tsset {
        Some((Some(p), t)) if !p.is_empty() => t.clone(),
        _ => return Err(rc_err(459, "panel variable not set; use xtset panelvar timevar")),
    };
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &vars);
    let t = util::col(ds, &tv)?;
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let mut times: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| t[i]).collect();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    let p = vars.len();
    let mut lam: Vec<Vec<f64>> = Vec::new();
    let mut nobs = 0;
    for &tt in &times {
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && t[i] == tt).collect();
        if rows.len() <= p {
            continue;
        }
        let y: Vec<f64> = rows.iter().map(|&i| cols[0][i]).collect();
        let x: Vec<Vec<f64>> = (1..p).map(|j| rows.iter().map(|&i| cols[j][i]).collect()).collect();
        if let Ok((b, _, _)) = ols(&y, &x) {
            lam.push(b);
            nobs += rows.len();
        }
    }
    let tn = lam.len();
    if tn < 2 {
        return Err(rc_err(2001, "insufficient time periods with cross-sectional regressions"));
    }
    let lags = util::opt_num(cmd, "lags", 3)?.unwrap_or(0.0) as usize;
    let mean: Vec<f64> = (0..p).map(|j| lam.iter().map(|l| l[j]).sum::<f64>() / tn as f64).collect();
    // Newey-West variance of the time-series mean (lags = 0: classical)
    let mut v = Matrix::zeros(p, p);
    for lag in 0..=lags.min(tn - 1) {
        let w = if lag == 0 { 1.0 } else { 1.0 - lag as f64 / (lags as f64 + 1.0) };
        for a in 0..p {
            for b in 0..p {
                let mut s = 0.0;
                for i in lag..tn {
                    s += (lam[i][a] - mean[a]) * (lam[i - lag][b] - mean[b]);
                }
                let s = s / tn as f64;
                if lag == 0 {
                    v[(a, b)] += s;
                } else {
                    v[(a, b)] += w * s;
                    v[(b, a)] += w * s;
                }
            }
        }
    }
    let scale = if lags == 0 { tn as f64 / (tn as f64 - 1.0) } else { 1.0 };
    let v = v.scale(scale / tn as f64);
    let mut names: Vec<String> = vars[1..].to_vec();
    names.push("_cons".into());
    let mut out = vec![String::new(), format!("{:<48}Number of obs     = {:>8}", "Fama-MacBeth regression", nobs)];
    out.push(format!("{:<48}Number of periods = {:>8}", if lags > 0 { format!("Newey-West standard errors, lags({lags})") } else { String::new() }, tn));
    out.push(String::new());
    out.extend(table(st, &vars[0], &names, &mean, &v, tn as f64 - 1.0));
    let mut dmask = mask.clone();
    for (i, m) in dmask.iter_mut().enumerate() {
        *m = *m && !is_missing(t[i]);
    }
    post(st, "fmb", &vars[0], &names, &mean, &v, nobs, dmask);
    st.e.set("T", tn as f64);
    st.e.set("df_r", tn as f64 - 1.0);
    st.e.set_mat("lambda", Matrix::from_rows(&lam).with_names((1..=tn).map(|i| i.to_string()).collect(), names));
    Ok(out.join("\n"))
}

fn valrisk(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let v = crate::xmodels::var_name(ds, rest.trim())?;
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, std::slice::from_ref(&v));
    let c = util::col(ds, &v)?;
    let r: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| c[i]).collect();
    if r.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let level = if cmd.has_opt("level", 1) { util::level(cmd, st)? } else { 95.0 };
    let a = level / 100.0;
    let h = util::opt_num(cmd, "horizon", 3)?.unwrap_or(1.0);
    let value = util::opt_num(cmd, "value", 3)?.unwrap_or(1.0);
    let method = cmd.opt("method", 3).map(|m| m.trim().to_lowercase()).unwrap_or_else(|| "historical".into());
    let (m, s) = (util::mean(&r), util::variance(&r).sqrt());
    let tail = |mut x: Vec<f64>| -> (f64, f64) {
        x.sort_by(|p, q| p.total_cmp(q));
        let n = x.len();
        let q = util::pctile_sorted(&x, 100.0 * (1.0 - a));
        let k = ((n as f64 * (1.0 - a)).ceil() as usize).clamp(1, n);
        let es = x[..k].iter().sum::<f64>() / k as f64;
        (-q, -es)
    };
    let (var, es, lab) = match method.as_str() {
        "historical" | "hist" => {
            let (q, e) = tail(r.clone());
            (q * h.sqrt(), e * h.sqrt(), "Historical simulation")
        }
        "parametric" | "normal" => {
            let z = dist::invnormal(a);
            let mh = m * h;
            let sh = s * h.sqrt();
            (-(mh - z * sh), -(mh - sh * dist::normalden(z) / (1.0 - a)), "Parametric (normal)")
        }
        "montecarlo" | "mc" => {
            if let Some(sd) = util::opt_num(cmd, "rseed", 5)? {
                st.rng.borrow_mut().reseed(sd as u64);
            }
            let reps = util::opt_num(cmd, "reps", 3)?.unwrap_or(10000.0) as usize;
            let sims: Vec<f64> = {
                let mut rng = st.rng.borrow_mut();
                (0..reps).map(|_| m * h + s * h.sqrt() * rng.normal()).collect()
            };
            let (q, e) = tail(sims);
            (q, e, "Monte Carlo (normal)")
        }
        o => return Err(rc_err(198, format!("method({o}) not allowed; use historical, parametric or montecarlo"))),
    };
    let out = vec![
        String::new(),
        format!("Value at risk of {v}: {lab}"),
        format!("  Observations = {}   confidence level = {}%   horizon = {}   value = {}", r.len(), fmt_g(level, 6), fmt_g(h, 6), fmt_g(value, 9)),
        String::new(),
        format!("{:>28} = {:>12}", "Value at risk (VaR)", fmt_g(var * value, 9)),
        format!("{:>28} = {:>12}", "Expected shortfall (ES)", fmt_g(es * value, 9)),
    ];
    let rr = st.begin_r();
    rr.set("VaR", var * value);
    rr.set("ES", es * value);
    rr.set("level", level);
    rr.set("N", r.len() as f64);
    rr.set("mean", m);
    rr.set("sd", s);
    rr.set_str("method", method);
    Ok(out.join("\n"))
}
