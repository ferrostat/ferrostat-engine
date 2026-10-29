//! Panel unit-root tests: `xtunitroot llc | ht | breitung | ips | fisher | hadri`.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::tstests::{mackinnon_p, Det};
use crate::util;
use crate::xt_lin::ls;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    vec![("xtunitroot", Box::new(XtUnitroot))]
}

/// Series of each panel ordered by time (gaps not allowed).
fn panel_series(ds: &DataSet, st: &Store, cmd: &Command, var: &str) -> Result<(Vec<Vec<f64>>, String, String), String> {
    let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
    let tvar = crate::xtnl::time_var(st)?;
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &[ivar.clone(), tvar.clone(), var.to_string()]);
    let p = util::col(ds, &ivar)?;
    let t = util::col(ds, &tvar)?;
    let x = util::col(ds, var)?;
    let mut m: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if mask[i] {
            m.entry((p[i] * 1000.0).round() as i64).or_default().push((t[i], x[i]));
        }
    }
    let mut out = Vec::new();
    for (_, mut v) in m {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        if v.windows(2).any(|w| w[1].0 - w[0].0 != 1.0) {
            return Err(rc_err(498, "xtunitroot requires data without gaps in the time variable"));
        }
        out.push(v.into_iter().map(|x| x.1).collect());
    }
    if out.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    Ok((out, ivar, tvar))
}

/// ADF regression of one series: (t statistic, rho-1 estimate, residuals,
/// regressors (orthogonalised lagged level and differences)).
struct Adf {
    t: f64,
    b: f64,
    resid: Vec<f64>,
    nobs: usize,
}

fn det_cols(n: usize, det: Det, start: usize) -> Vec<Vec<f64>> {
    match det {
        Det::None => vec![],
        Det::Const => vec![vec![1.0; n]],
        Det::Trend => vec![vec![1.0; n], (0..n).map(|i| (start + i) as f64).collect()],
    }
}

fn adf(y: &[f64], p: usize, det: Det) -> Option<Adf> {
    let tlen = y.len();
    if tlen < p + 3 {
        return None;
    }
    let dy: Vec<f64> = (1..tlen).map(|t| y[t] - y[t - 1]).collect();
    let start = p; // index into dy
    let n = dy.len() - start;
    let yv: Vec<f64> = (start..dy.len()).map(|t| dy[t]).collect();
    let mut x = vec![(start..dy.len()).map(|t| y[t]).collect::<Vec<f64>>()];
    for l in 1..=p {
        x.push((start..dy.len()).map(|t| dy[t - l]).collect());
    }
    x.extend(det_cols(n, det, start + 1));
    let k = x.len();
    if n <= k {
        return None;
    }
    let (b, ai, e) = ls(&x, &yv);
    let s2 = e.iter().map(|v| v * v).sum::<f64>() / (n - k) as f64;
    Some(Adf { t: b[0] / (s2 * ai[(0, 0)]).sqrt(), b: b[0], resid: e, nobs: n })
}

fn llc_table(det: Det, t: f64) -> (f64, f64) {
    let ts = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0, 1e9];
    let (mu, sd): ([f64; 13], [f64; 13]) = match det {
        Det::None => ([0.0; 13], [1.0; 13]),
        Det::Const => (
            [-0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521, -0.520, -0.518, -0.509, -0.500],
            [0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782, 0.776, 0.742, 0.707],
        ),
        Det::Trend => (
            [-0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578, -0.571, -0.566, -0.533, -0.500],
            [1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695, 0.603, 0.500],
        ),
    };
    if t <= ts[0] {
        return (mu[0], sd[0]);
    }
    for i in 1..ts.len() {
        if t <= ts[i] {
            let w = if ts[i] > 1e8 { 0.0 } else { (t - ts[i - 1]) / (ts[i] - ts[i - 1]) };
            return (mu[i - 1] + w * (mu[i] - mu[i - 1]), sd[i - 1] + w * (sd[i] - sd[i - 1]));
        }
    }
    (mu[12], sd[12])
}

pub struct XtUnitroot;

impl CommandHandler for XtUnitroot {
    fn name(&self) -> &'static str {
        "xtunitroot"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut words = cmd.main.split_whitespace();
        let test = words.next().ok_or_else(|| rc_err(198, "test required"))?.to_lowercase();
        let var = crate::varlist::resolve_name(ds, words.next().ok_or_else(|| rc_err(100, "varname required"))?)?;
        let (mut ys, _ivar, _tvar) = panel_series(ds, st, cmd, &var)?;
        let det = if cmd.has_opt("trend", 2) {
            Det::Trend
        } else if cmd.has_opt("noconstant", 3) {
            Det::None
        } else {
            Det::Const
        };
        if cmd.has_opt("demean", 3) {
            let tmax = ys.iter().map(|y| y.len()).max().unwrap_or(0);
            let mut means = vec![(0.0, 0.0); tmax];
            for y in &ys {
                for (t, v) in y.iter().enumerate() {
                    means[t].0 += v;
                    means[t].1 += 1.0;
                }
            }
            for y in ys.iter_mut() {
                for (t, v) in y.iter_mut().enumerate() {
                    *v -= means[t].0 / means[t].1;
                }
            }
        }
        let np = ys.len();
        let tlen = ys.iter().map(|y| y.len()).max().unwrap_or(0);
        let balanced = ys.iter().all(|y| y.len() == tlen);
        let lagspec = cmd.opt("lags", 3).unwrap_or("0").trim().to_lowercase();
        let (ic, maxlag): (Option<String>, usize) = match lagspec.split_whitespace().collect::<Vec<_>>().as_slice() {
            [k, m] if ["aic", "bic", "hqic"].contains(k) => (Some(k.to_string()), m.parse().map_err(|_| rc_err(198, "invalid lags()"))?),
            [m] => (None, m.parse().map_err(|_| rc_err(198, "invalid lags()"))?),
            _ => return Err(rc_err(198, "invalid lags()")),
        };
        let plags: Vec<usize> = ys.iter().map(|y| match &ic {
            None => maxlag,
            Some(k) => {
                let mut best = (f64::INFINITY, 0);
                for p in 0..=maxlag {
                    if let Some(a) = adf(&y[maxlag - p..], p, det) {
                        let n = a.nobs as f64;
                        let kk = (1 + p + det as usize) as f64;
                        let rss: f64 = a.resid.iter().map(|v| v * v).sum();
                        let pen = match k.as_str() { "aic" => 2.0 * kk / n, "bic" => kk * n.ln() / n, _ => 2.0 * kk * n.ln().ln() / n };
                        let c = (rss / n).ln() + pen;
                        if c < best.0 { best = (c, p); }
                    }
                }
                best.1
            }
        }).collect();
        let lags = maxlag;
        let avg_lags = plags.iter().sum::<usize>() as f64 / plags.len().max(1) as f64;
        let lag_text = match &ic { None => format!("{lags} lag{}", if lags == 1 { "" } else { "s" }), Some(k) => format!("{:.2} lags average (chosen by {})", avg_lags, k.to_uppercase()) };
        let mut out = Vec::new();
        let r = st.begin_r();
        let head = |title: &str, h0: &str, ha: &str| -> Vec<String> {
            let mut v = vec![String::new(), format!("{title} for {var}"), "-".repeat(title.len() + 5 + var.len())];
            v.push(format!("{:<44}Number of panels  = {:>6}", format!("H0: {h0}"), np));
            v.push(format!("{:<44}{}", format!("Ha: {ha}"), if balanced { format!("Number of periods = {:>6}", tlen) } else { format!("Avg. number of periods = {:.2}", ys.iter().map(|y| y.len()).sum::<usize>() as f64 / np as f64) }));
            v.push(String::new());
            v
        };
        let detl = |v: &mut Vec<String>| {
            v.push(format!("Panel means:  {}", if det == Det::None { "Not included" } else { "Included" }));
            v.push(format!("Time trend:   {}", if det == Det::Trend { "Included" } else { "Not included" }));
        };
        let rule = "-".repeat(78);
        match test.as_str() {
            "ht" => {
                if !balanced {
                    return Err(rc_err(498, "Harris-Tzavalis test requires strongly balanced data"));
                }
                // within regression of y_t on y_{t-1}
                let mut yv = Vec::new();
                let mut xv = Vec::new();
                for y in &ys {
                    let yy: Vec<f64> = y[1..].to_vec();
                    let xx: Vec<f64> = y[..y.len() - 1].to_vec();
                    let n = yy.len();
                    let dc = det_cols(n, det, 1);
                    if dc.is_empty() {
                        yv.extend(yy);
                        xv.extend(xx);
                    } else {
                        let (_, _, ey) = ls(&dc, &yy);
                        let (_, _, ex) = ls(&dc, &xx);
                        yv.extend(ey);
                        xv.extend(ex);
                    }
                }
                let rho = yv.iter().zip(&xv).map(|(a, b)| a * b).sum::<f64>() / xv.iter().map(|b| b * b).sum::<f64>();
                let t = (tlen - 1) as f64;
                let (bb, cc) = match det {
                    Det::None => (0.0, 2.0 / (t * (t - 1.0))),
                    Det::Const => (-3.0 / (t + 1.0), 3.0 * (17.0 * t * t - 20.0 * t + 17.0) / (5.0 * (t - 1.0) * (t + 1.0).powi(3))),
                    Det::Trend => (-15.0 / (2.0 * (t + 2.0)), 15.0 * (193.0 * t * t - 728.0 * t + 1147.0) / (112.0 * (t + 2.0).powi(3) * (t - 2.0))),
                };
                let z = (np as f64).sqrt() * (rho - 1.0 - bb) / cc.sqrt();
                let p = dist::normal(z);
                out = head("Harris-Tzavalis unit-root test", "Panels contain unit roots", "Panels are stationary");
                out.push(format!("{:<44}Asymptotics: N -> Infinity", "AR parameter: Common"));
                out.push(format!("{:<44}{:>13}T Fixed", format!("Panel means:  {}", if det == Det::None { "Not included" } else { "Included" }), ""));
                out.push(format!("Time trend:   {}", if det == Det::Trend { "Included" } else { "Not included" }));
                out.push(String::new());
                out.push(rule.clone());
                out.push("               Statistic           z      p-value".into());
                out.push(rule.clone());
                out.push(format!(" rho             {:>6.4}    {:>8.4}       {:.4}", rho, z, p));
                out.push(rule.clone());
                r.set("rho", rho);
                r.set("z", z);
                r.set("p", p);
                r.set("N_g", np as f64);
                r.set("N_t", tlen as f64);
            }
            "llc" => {
                if !balanced {
                    return Err(rc_err(498, "Levin-Lin-Chu test requires strongly balanced data"));
                }
                let mut ev = Vec::new();
                let mut vv = Vec::new();
                let mut ratio = 0.0;
                let ttil = tlen as f64 - avg_lags - 1.0;
                let kbar = (3.21 * ttil.powf(1.0 / 3.0)).floor() as usize;
                for (y, &lags) in ys.iter().zip(&plags) {
                    let a = adf(y, lags, det).ok_or_else(|| rc_err(2001, "insufficient observations"))?;
                    let dy: Vec<f64> = (1..y.len()).map(|t| y[t] - y[t - 1]).collect();
                    let n = a.nobs;
                    let start = lags;
                    let mut z: Vec<Vec<f64>> = Vec::new();
                    for l in 1..=lags {
                        z.push((start..dy.len()).map(|t| dy[t - l]).collect());
                    }
                    z.extend(det_cols(n, det, start + 1));
                    let yy: Vec<f64> = (start..dy.len()).map(|t| dy[t]).collect();
                    let xl: Vec<f64> = (start..dy.len()).map(|t| y[t]).collect();
                    let (e, v) = if z.is_empty() { (yy.clone(), xl.clone()) } else { (ls(&z, &yy).2, ls(&z, &xl).2) };
                    let sig = (a.resid.iter().map(|x| x * x).sum::<f64>() / (n as f64 - 1.0 - lags as f64 - z.len() as f64 + lags as f64).max(1.0)).sqrt();
                    ev.extend(e.iter().map(|x| x / sig));
                    vv.extend(v.iter().map(|x| x / sig));
                    // long-run variance of the differenced series (deterministics removed)
                    let dd: Vec<f64> = {
                        let dc = det_cols(dy.len(), det, 1);
                        if dc.is_empty() {
                            dy.clone()
                        } else {
                            ls(&dc, &dy).2
                        }
                    };
                    let tt = dd.len() as f64;
                    let mut lr = dd.iter().map(|x| x * x).sum::<f64>() / tt;
                    for l in 1..=kbar {
                        let w = 1.0 - l as f64 / (kbar as f64 + 1.0);
                        let c: f64 = (l..dd.len()).map(|t| dd[t] * dd[t - l]).sum::<f64>() / tt;
                        lr += 2.0 * w * c;
                    }
                    ratio += lr.max(0.0).sqrt() / sig;
                }
                let sn = ratio / np as f64;
                let svv: f64 = vv.iter().map(|x| x * x).sum();
                let delta = ev.iter().zip(&vv).map(|(a, b)| a * b).sum::<f64>() / svv;
                let nt = ev.len() as f64;
                let s2 = ev.iter().zip(&vv).map(|(a, b)| (a - delta * b).powi(2)).sum::<f64>() / (nt - 1.0);
                let se = (s2 / svv).sqrt();
                let tdel = delta / se;
                let (mu, sd) = llc_table(det, ttil);
                let tstar = (tdel - nt * sn * se / s2 * mu) / sd;
                let p = dist::normal(tstar);
                out = head("Levin-Lin-Chu unit-root test", "Panels contain unit roots", "Panels are stationary");
                out.push(format!("{:<44}Asymptotics: N/T -> 0", "AR parameter: Common"));
                detl(&mut out);
                out.push(String::new());
                out.push(format!("ADF regressions: {lag_text}"));
                out.push(format!("LR variance:     Bartlett kernel, {}.00 lags average (chosen by LLC)", kbar));
                out.push(rule.clone());
                out.push("               Statistic      p-value".into());
                out.push(rule.clone());
                out.push(format!(" Unadjusted t   {:>9.4}", tdel));
                out.push(format!(" Adjusted t*    {:>9.4}       {:.4}", tstar, p));
                out.push(rule.clone());
                r.set("delta", delta);
                r.set("tds", tdel);
                r.set("td", tstar);
                r.set("p_tds", p);
                r.set("N_g", np as f64);
                r.set("N_t", tlen as f64);
            }
            "breitung" => {
                if !balanced {
                    return Err(rc_err(498, "Breitung test requires strongly balanced data"));
                }
                let (mut num, mut den, mut ss, mut nn) = (0.0, 0.0, 0.0, 0.0);
                for y in &ys {
                    let t = y.len() - 1;
                    let dy: Vec<f64> = (1..=t).map(|s| y[s] - y[s - 1]).collect();
                    for s in 0..t - 1 {
                        let rem = (t - s - 1) as f64;
                        let fwd: f64 = dy[s + 1..].iter().sum::<f64>() / rem;
                        let ystar = (rem / (rem + 1.0)).sqrt() * (dy[s] - fwd);
                        let xstar = match det {
                            Det::None => y[s],
                            _ => y[s] - y[0],
                        };
                        num += ystar * xstar;
                        den += xstar * xstar;
                        ss += ystar * ystar;
                        nn += 1.0;
                    }
                }
                let s2 = ss / nn;
                let lam = num / (s2 * den).sqrt();
                let p = dist::normal(lam);
                out = head("Breitung unit-root test", "Panels contain unit roots", "Panels are stationary");
                out.push(format!("{:<44}Asymptotics: T,N -> Infinity", "AR parameter: Common"));
                out.push(format!("{:<44}{:>13}sequentially", format!("Panel means:  {}", if det == Det::None { "Not included" } else { "Included" }), ""));
                out.push(format!("Time trend:   {}", if det == Det::Trend { "Included" } else { "Not included" }));
                out.push(String::new());
                out.push(rule.clone());
                out.push("               Statistic      p-value".into());
                out.push(rule.clone());
                out.push(format!(" lambda         {:>9.4}       {:.4}", lam, p));
                out.push(rule.clone());
                r.set("lambda", lam);
                r.set("p", p);
                r.set("N_g", np as f64);
            }
            "ips" => {
                let mut ts = Vec::new();
                let mut tt = Vec::new();
                for (y, &lags) in ys.iter().zip(&plags) {
                    let a = adf(y, lags, det).ok_or_else(|| rc_err(2001, "insufficient observations"))?;
                    ts.push(a.t);
                    // t-tilde: variance estimated with divisor T
                    let k = 1 + lags + match det {
                        Det::None => 0,
                        Det::Const => 1,
                        Det::Trend => 2,
                    };
                    tt.push(a.t * ((a.nobs as f64) / (a.nobs - k) as f64).sqrt().recip());
                    let _ = a.b;
                }
                let tbar = ts.iter().sum::<f64>() / np as f64;
                let ttbar = tt.iter().sum::<f64>() / np as f64;
                let (e, v): (f64, f64) = match det {
                    Det::Trend => (-2.166, 0.565),
                    _ => (-1.533, 0.706),
                };
                let z = (np as f64).sqrt() * (ttbar - e) / v.sqrt();
                let p = dist::normal(z);
                out = head("Im-Pesaran-Shin unit-root test", "All panels contain unit roots", "Some panels are stationary");
                out.push(format!("{:<44}Asymptotics: T,N -> Infinity", "AR parameter: Panel-specific"));
                out.push(format!("{:<44}{:>13}sequentially", format!("Panel means:  {}", if det == Det::None { "Not included" } else { "Included" }), ""));
                out.push(format!("Time trend:   {}", if det == Det::Trend { "Included" } else { "Not included" }));
                out.push(String::new());
                out.push(format!("ADF regressions: {lag_text}"));
                out.push(rule.clone());
                out.push("                  Statistic      p-value".into());
                out.push(rule.clone());
                out.push(format!(" t-bar            {:>9.4}", tbar));
                out.push(format!(" t-tilde-bar      {:>9.4}", ttbar));
                out.push(format!(" Z-t-tilde-bar    {:>9.4}       {:.4}", z, p));
                out.push(rule.clone());
                r.set("tbar", tbar);
                r.set("ttildebar", ttbar);
                r.set("zttildebar", z);
                r.set("p_zttildebar", p);
                r.set("N_g", np as f64);
            }
            "fisher" => {
                let pp = cmd.has_opt("pperron", 2);
                if !pp && !cmd.has_opt("dfuller", 2) {
                    return Err(rc_err(198, "either dfuller or pperron must be specified"));
                }
                let mut pv = Vec::new();
                for (y, &plag) in ys.iter().zip(&plags) {
                    let a = adf(y, if pp { 0 } else { plag }, det).ok_or_else(|| rc_err(2001, "insufficient observations"))?;
                    let stat = if pp {
                        // Phillips-Perron Z(t) with Newey-West bandwidth `lags`
                        let e = &a.resid;
                        let n = e.len() as f64;
                        let g0 = e.iter().map(|x| x * x).sum::<f64>() / n;
                        let mut lam = g0;
                        for l in 1..=lags {
                            let c: f64 = (l..e.len()).map(|t| e[t] * e[t - l]).sum::<f64>() / n;
                            lam += 2.0 * (1.0 - l as f64 / (lags as f64 + 1.0)) * c;
                        }
                        let k = 1 + match det {
                            Det::None => 0,
                            Det::Const => 1,
                            Det::Trend => 2,
                        };
                        let s2 = e.iter().map(|x| x * x).sum::<f64>() / (n - k as f64);
                        let seb = a.b / a.t;
                        (g0 / lam).sqrt() * a.t - (lam - g0) * n * seb / (2.0 * lam.sqrt() * s2.sqrt())
                    } else {
                        a.t
                    };
                    pv.push(mackinnon_p(stat, det).clamp(1e-300, 1.0 - 1e-16));
                }
                let n = np as f64;
                let pch = -2.0 * pv.iter().map(|p| p.ln()).sum::<f64>();
                let zz = pv.iter().map(|p| dist::invnormal(*p)).sum::<f64>() / n.sqrt();
                let lst = pv.iter().map(|p| (p / (1.0 - p)).ln()).sum::<f64>() * (3.0 * (5.0 * n + 4.0) / (std::f64::consts::PI.powi(2) * n * (5.0 * n + 2.0))).sqrt();
                let dfl = 5.0 * n + 4.0;
                let pm = (pch - 2.0 * n) / (4.0 * n).sqrt();
                out = head("Fisher-type unit-root test", "All panels contain unit roots", "At least one panel is stationary");
                out.push(format!("{:<44}Asymptotics: T -> Infinity", "AR parameter: Panel-specific"));
                detl(&mut out);
                out.push("Drift term:   Not included".to_string());
                out.push(String::new());
                out.push(format!("Based on {} tests", if pp { format!("Phillips-Perron, Newey-West lags: {lags}") } else { format!("augmented Dickey-Fuller, lags: {lags}") }));
                out.push(rule.clone());
                out.push(format!("{:<38}Statistic      p-value", ""));
                out.push(rule.clone());
                out.push(format!(" Inverse chi-squared({:<4}){:>4} P  {:>12.4}       {:.4}", 2 * np, "", pch, dist::chi2tail(2.0 * n, pch)));
                out.push(format!(" Inverse normal{:>15} Z  {:>12.4}       {:.4}", "", zz, dist::normal(zz)));
                out.push(format!(" Inverse logit t({:<4}){:>8} L* {:>12.4}       {:.4}", dfl, "", lst, dist::ttail(dfl, -lst)));
                out.push(format!(" Modified inv. chi-squared{:>4} Pm {:>12.4}       {:.4}", "", pm, 1.0 - dist::normal(pm)));
                out.push(rule.clone());
                out.push(" P statistic requires number of panels to be finite.".into());
                out.push(" Other statistics are suitable for finite or infinite number of panels.".into());
                out.push(rule.clone());
                r.set("P", pch);
                r.set("p_P", dist::chi2tail(2.0 * n, pch));
                r.set("Z", zz);
                r.set("p_Z", dist::normal(zz));
                r.set("L", lst);
                r.set("p_L", dist::ttail(dfl, -lst));
                r.set("Pm", pm);
                r.set("p_Pm", 1.0 - dist::normal(pm));
                r.set("N_g", n);
            }
            "hadri" => {
                if !balanced {
                    return Err(rc_err(498, "Hadri test requires strongly balanced data"));
                }
                let dett = if det == Det::None { Det::Const } else { det };
                let robust = cmd.has_opt("robust", 1);
                let mut lmn = Vec::new();
                let mut s2s = Vec::new();
                let mut tot = 0.0;
                let mut cnt = 0.0;
                for y in &ys {
                    let dc = det_cols(y.len(), dett, 1);
                    let e = ls(&dc, y).2;
                    let mut s = 0.0;
                    let mut acc = 0.0;
                    for v in &e {
                        s += v;
                        acc += s * s;
                    }
                    let t = y.len() as f64;
                    lmn.push(acc / (t * t));
                    let s2 = e.iter().map(|v| v * v).sum::<f64>();
                    s2s.push(s2 / (t - dc.len() as f64));
                    tot += s2;
                    cnt += t - dc.len() as f64;
                }
                let pooled = tot / cnt;
                let lm = if robust { lmn.iter().zip(&s2s).map(|(a, b)| a / b).sum::<f64>() / np as f64 } else { lmn.iter().sum::<f64>() / (np as f64 * pooled) };
                let (xi, zeta2): (f64, f64) = if dett == Det::Trend { (1.0 / 15.0, 11.0 / 6300.0) } else { (1.0 / 6.0, 1.0 / 45.0) };
                let z = (np as f64).sqrt() * (lm - xi) / zeta2.sqrt();
                let p = 1.0 - dist::normal(z);
                out = head("Hadri LM test", "All panels are stationary", "Some panels contain unit roots");
                out.push(format!("{:<44}Asymptotics: T, N -> Infinity", "Time trend:         ".to_string() + if dett == Det::Trend { "Included" } else { "Not included" }));
                out.push(format!("{:<44}{:>13}sequentially", format!("Heteroskedasticity: {}", if robust { "Robust" } else { "Not robust" }), ""));
                out.push("LR variance:        (not used)".into());
                out.push(String::new());
                out.push(rule.clone());
                out.push("               Statistic      p-value".into());
                out.push(rule.clone());
                out.push(format!(" z              {:>9.4}       {:.4}", z, p));
                out.push(rule.clone());
                r.set("z", z);
                r.set("p", p);
                r.set("N_g", np as f64);
            }
            _ => return Err(rc_err(198, format!("unknown test {test}"))),
        }
        let _ = fmt_g;
        let _ = is_missing;
        Ok(out.join("\n"))
    }
}
