//! Panel-data cointegration tests: `xtcointtest kao | pedroni | westerlund`.
//!
//! * Kao (1999): Dickey-Fuller type tests on the residuals of the pooled
//!   within regression (homogeneous cointegrating vector), with Kao's
//!   asymptotic adjustments based on the long-run covariance of the
//!   residuals and the differenced regressors.
//! * Pedroni (1999, 2004): residual-based Phillips-Perron rho and t and
//!   augmented Dickey-Fuller t statistics from panel-specific cointegrating
//!   regressions, as group-mean (`ar(panelspecific)`) or pooled
//!   (`ar(same)`) statistics.
//! * Westerlund (2005): variance-ratio statistics from partial sums of the
//!   residuals, for some (`somepanels`) or all (`allpanels`) panels.
//!
//! The Pedroni and Westerlund statistics are standardized with null moments
//! of the per-panel quantities computed by simulating independent random
//! walks of the same length and dimension (fixed internal seed, so results
//! are reproducible and the session's random-number state is untouched);
//! the standardized statistics are compared with the standard normal and
//! large negative values reject the null of no cointegration.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::store::{Results, Store};
use crate::util;
use crate::xt_lin::ls;

pub struct Xtcointtest;

/// One panel: dependent variable and regressors ordered by time.
struct Panel {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
}

fn load(ds: &DataSet, st: &Store, cmd: &Command, vars: &[String]) -> Result<(Vec<Panel>, String, String), String> {
    let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
    let tvar = crate::xtnl::time_var(st)?;
    let mut mask = util::touse(ds, st, cmd)?;
    let mut all = vars.to_vec();
    all.push(ivar.clone());
    all.push(tvar.clone());
    util::markout(ds, &mut mask, &all);
    let p = util::col(ds, &ivar)?;
    let t = util::col(ds, &tvar)?;
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let mut m: BTreeMap<i64, Vec<(f64, usize)>> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if mask[i] && !is_missing(p[i]) {
            m.entry((p[i] * 1000.0).round() as i64).or_default().push((t[i], i));
        }
    }
    let mut out = Vec::new();
    for (_, mut v) in m {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        if v.windows(2).any(|w| w[1].0 - w[0].0 != 1.0) {
            return Err(rc_err(498, "xtcointtest requires data without gaps in the time variable"));
        }
        let rows: Vec<usize> = v.iter().map(|x| x.1).collect();
        out.push(Panel { y: rows.iter().map(|&r| cols[0][r]).collect(), x: (1..cols.len()).map(|j| rows.iter().map(|&r| cols[j][r]).collect()).collect() });
    }
    if out.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    Ok((out, ivar, tvar))
}

/// Bartlett-kernel long-run covariance of the columns of `w` (rows = time).
fn lrcov(w: &[Vec<f64>], bw: f64) -> (Matrix, Matrix) {
    let n = w.len();
    let k = w.first().map_or(0, |r| r.len());
    let mut s = Matrix::zeros(k, k);
    for r in w {
        for a in 0..k {
            for b in 0..k {
                s[(a, b)] += r[a] * r[b] / n as f64;
            }
        }
    }
    let mut o = s.clone();
    let maxj = (bw.floor() as usize).min(n.saturating_sub(1));
    for j in 1..=maxj {
        let wt = 1.0 - j as f64 / (bw + 1.0);
        for t in j..n {
            for a in 0..k {
                for b in 0..k {
                    let g = w[t][a] * w[t - j][b] / n as f64;
                    o[(a, b)] += wt * g;
                    o[(b, a)] += wt * g;
                }
            }
        }
    }
    (s, o)
}

/// Conditional variance a_11 - a_12 a_22^-1 a_21 of a partitioned matrix.
fn cond_var(m: &Matrix) -> f64 {
    let k = m.rows;
    if k == 1 {
        return m[(0, 0)];
    }
    let mut a22 = Matrix::zeros(k - 1, k - 1);
    for i in 1..k {
        for j in 1..k {
            a22[(i - 1, j - 1)] = m[(i, j)];
        }
    }
    let (inv, _) = a22.sym_inverse(1e-12);
    let mut q = 0.0;
    for i in 1..k {
        for j in 1..k {
            q += m[(0, i)] * inv[(i - 1, j - 1)] * m[(j, 0)];
        }
    }
    m[(0, 0)] - q
}

#[derive(Clone, Copy, PartialEq)]
enum LagRule {
    Fixed(usize),
    Aic(usize),
    Bic(usize),
    Hqic(usize),
}

fn lag_rule(cmd: &Command, tbar: f64) -> Result<LagRule, String> {
    let def_max = ((12.0 * (tbar / 100.0).powf(0.25)).floor() as usize).max(1).min(((tbar - 4.0).max(1.0) / 3.0) as usize).max(1);
    match cmd.opt("lags", 3) {
        None => Ok(LagRule::Aic(def_max)),
        Some(s) => {
            let w: Vec<&str> = s.split_whitespace().collect();
            let num = |i: usize| -> Result<usize, String> { w.get(i).map(|v| v.parse::<usize>().map_err(|_| rc_err(198, "lags() invalid"))).transpose().map(|v| v.unwrap_or(def_max)) };
            match w.first().map(|x| x.to_lowercase()) {
                Some(ref a) if a == "aic" => Ok(LagRule::Aic(num(1)?)),
                Some(ref a) if a == "bic" => Ok(LagRule::Bic(num(1)?)),
                Some(ref a) if a == "hqic" => Ok(LagRule::Hqic(num(1)?)),
                Some(_) => Ok(LagRule::Fixed(num(0)?)),
                None => Ok(LagRule::Aic(def_max)),
            }
        }
    }
}

/// ADF regressions of residual series (pooled when `pool`): returns the t
/// statistics of rho - 1 (one per series, or one pooled) and the lag used.
fn adf_t(series: &[Vec<f64>], rule: LagRule, pool: bool) -> Vec<(f64, usize)> {
    let fit = |sers: &[&Vec<f64>], p: usize, pmax: usize| -> Option<(f64, f64, usize)> {
        // common sample: start after pmax lags
        let mut xs: Vec<Vec<f64>> = vec![Vec::new(); p + 1];
        let mut y = Vec::new();
        for e in sers {
            let n = e.len();
            if n < pmax + 3 {
                continue;
            }
            for t in (pmax + 1)..n {
                y.push(e[t] - e[t - 1]);
                xs[0].push(e[t - 1]);
                for l in 1..=p {
                    xs[l].push(e[t - l] - e[t - l - 1]);
                }
            }
        }
        let n = y.len();
        if n <= p + 2 {
            return None;
        }
        let (b, ai, r) = ls(&xs, &y);
        let ssr: f64 = r.iter().map(|v| v * v).sum();
        let s2 = ssr / (n - p - 1) as f64;
        let t = b[0] / (s2 * ai[(0, 0)]).sqrt();
        Some((t, ssr, n))
    };
    let choose = |sers: &[&Vec<f64>]| -> (f64, usize) {
        let (pmax, kind) = match rule {
            LagRule::Fixed(p) => return (fit(sers, p, p).map(|x| x.0).unwrap_or(f64::NAN), p),
            LagRule::Aic(m) => (m, 0),
            LagRule::Bic(m) => (m, 1),
            LagRule::Hqic(m) => (m, 2),
        };
        let mut best = (f64::INFINITY, 0usize);
        for p in 0..=pmax {
            if let Some((_, ssr, n)) = fit(sers, p, pmax) {
                let nf = n as f64;
                let k = (p + 1) as f64;
                let pen = match kind {
                    0 => 2.0 * k,
                    1 => k * nf.ln(),
                    _ => 2.0 * k * nf.ln().ln(),
                };
                let ic = nf * (ssr / nf).ln() + pen;
                if ic < best.0 {
                    best = (ic, p);
                }
            }
        }
        (fit(sers, best.1, best.1).map(|x| x.0).unwrap_or(f64::NAN), best.1)
    };
    if pool {
        let refs: Vec<&Vec<f64>> = series.iter().collect();
        vec![choose(&refs)]
    } else {
        series.iter().map(|s| choose(&[s])).collect()
    }
}

fn nw_bandwidth(t: f64) -> f64 {
    4.0 * (t / 100.0).powf(2.0 / 9.0)
}

fn bandwidth(cmd: &Command, tbar: f64) -> Result<(f64, String), String> {
    match cmd.opt("kernel", 3) {
        None => Ok((nw_bandwidth(tbar), "Newey\u{2013}West".into())),
        Some(s) => {
            let w: Vec<&str> = s.split_whitespace().collect();
            if let Some(k) = w.first() {
                if !"bartlett".starts_with(&k.to_lowercase()) || k.len() < 2 {
                    return Err(rc_err(198, "only the Bartlett kernel is supported"));
                }
            }
            match w.get(1) {
                None => Ok((nw_bandwidth(tbar), "Newey\u{2013}West".into())),
                Some(v) if v.eq_ignore_ascii_case("nwest") => Ok((nw_bandwidth(tbar), "Newey\u{2013}West".into())),
                Some(v) => Ok((v.parse::<f64>().map_err(|_| rc_err(198, "kernel() invalid"))?, "user".into())),
            }
        }
    }
}

/// Cross-sectional demeaning (subtract the period means across panels).
fn demean(panels: &mut [Panel]) {
    let tmax = panels.iter().map(|p| p.y.len()).max().unwrap_or(0);
    let k = panels.first().map_or(0, |p| p.x.len());
    for t in 0..tmax {
        let mut cnt = 0.0;
        let mut my = 0.0;
        let mut mx = vec![0.0; k];
        for p in panels.iter() {
            if t < p.y.len() {
                cnt += 1.0;
                my += p.y[t];
                for j in 0..k {
                    mx[j] += p.x[j][t];
                }
            }
        }
        for p in panels.iter_mut() {
            if t < p.y.len() {
                p.y[t] -= my / cnt;
                for j in 0..k {
                    p.x[j][t] -= mx[j] / cnt;
                }
            }
        }
    }
}

/// Residuals of the panel-specific cointegrating regression of y on a
/// constant (optional trend) and x.
fn coint_resid(p: &Panel, cons: bool, trend: bool) -> Vec<f64> {
    let n = p.y.len();
    let mut xs: Vec<Vec<f64>> = Vec::new();
    if cons {
        xs.push(vec![1.0; n]);
    }
    if trend {
        xs.push((1..=n).map(|t| t as f64).collect());
    }
    xs.extend(p.x.iter().cloned());
    ls(&xs, &p.y).2
}

/// Per-panel Phillips-Perron quantities of a residual series:
/// A = T^-1 sum(e_{t-1} de_t - lambda), B = T^-2 sum e_{t-1}^2, S = long-run variance.
fn pp_parts(e: &[f64], bw: f64) -> (f64, f64, f64) {
    let n = e.len();
    let tt = (n - 1) as f64;
    let lag: Vec<f64> = e[..n - 1].to_vec();
    let cur: Vec<f64> = e[1..].to_vec();
    let g = lag.iter().zip(&cur).map(|(a, b)| a * b).sum::<f64>() / lag.iter().map(|a| a * a).sum::<f64>().max(1e-300);
    let mu: Vec<Vec<f64>> = lag.iter().zip(&cur).map(|(a, b)| vec![b - g * a]).collect();
    let (s, o) = lrcov(&mu, bw);
    let lambda = 0.5 * (o[(0, 0)] - s[(0, 0)]) * tt;
    let a = (lag.iter().zip(&cur).map(|(a, b)| a * (b - a)).sum::<f64>() - lambda) / tt;
    let b = lag.iter().map(|v| v * v).sum::<f64>() / (tt * tt);
    (a, b, o[(0, 0)].max(1e-300))
}

/// Per-panel variance-ratio quantities: (T^-2 sum E_t^2 / T, R / T) where E
/// are partial sums of the residuals and R their sum of squares.
fn vr_parts(e: &[f64]) -> (f64, f64) {
    let n = e.len() as f64;
    let mut c = 0.0;
    let mut se = 0.0;
    for v in e {
        c += v;
        se += c * c;
    }
    let r: f64 = e.iter().map(|v| v * v).sum();
    (se / (n * n * n), r / n)
}

/// Null moments of the per-panel quantities from simulated independent
/// random walks: returns draws of (A, B, S, adf t, VR num, VR den).
fn simulate_null(t: usize, k: usize, cons: bool, trend: bool, bw: f64, adf_p: usize, reps: usize) -> Vec<[f64; 6]> {
    let mut rng = Rng::new(20_051_999 + (t as u64) * 31 + k as u64);
    let mut out = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut walk = || {
            let mut v = 0.0;
            (0..t)
                .map(|_| {
                    v += rng.normal();
                    v
                })
                .collect::<Vec<f64>>()
        };
        let y = walk();
        let x: Vec<Vec<f64>> = (0..k).map(|_| walk()).collect();
        let e = coint_resid(&Panel { y, x }, cons, trend);
        let (a, b, s) = pp_parts(&e, bw);
        let tadf = adf_t(&[e.clone()], LagRule::Fixed(adf_p), false)[0].0;
        let (vn, vd) = vr_parts(&e);
        out.push([a, b, s, tadf, vn, vd]);
    }
    out
}

fn mean_cov(d: &[Vec<f64>]) -> (Vec<f64>, Matrix) {
    let n = d.len() as f64;
    let k = d[0].len();
    let m: Vec<f64> = (0..k).map(|j| d.iter().map(|r| r[j]).sum::<f64>() / n).collect();
    let mut c = Matrix::zeros(k, k);
    for r in d {
        for a in 0..k {
            for b in 0..k {
                c[(a, b)] += (r[a] - m[a]) * (r[b] - m[b]) / (n - 1.0);
            }
        }
    }
    (m, c)
}

/// Standardize sqrt(N) (f(sample means) - f(null means)) by the delta method.
fn standardize(f: &dyn Fn(&[f64]) -> f64, sample: &[f64], null_mean: &[f64], null_cov: &Matrix, n: f64) -> f64 {
    let k = null_mean.len();
    let grad: Vec<f64> = (0..k)
        .map(|j| {
            let h = 1e-6 * null_mean[j].abs().max(1e-6);
            let mut a = null_mean.to_vec();
            let mut b = null_mean.to_vec();
            a[j] += h;
            b[j] -= h;
            (f(&a) - f(&b)) / (2.0 * h)
        })
        .collect();
    let mut v = 0.0;
    for a in 0..k {
        for b in 0..k {
            v += grad[a] * null_cov[(a, b)] * grad[b];
        }
    }
    n.sqrt() * (f(sample) - f(null_mean)) / v.max(1e-300).sqrt()
}

struct Out {
    title: String,
    ha: String,
    info: Vec<(String, String, String, String)>,
    stats: Vec<(String, f64)>,
}

impl CommandHandler for Xtcointtest {
    fn name(&self) -> &'static str {
        "xtcointtest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (test, rest) = main.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "test and varlist required"))?;
        let test = test.to_lowercase();
        let vars = crate::varlist::expand(ds, &crate::tsops::expand_text(rest.trim()))?;
        let vars: Vec<String> = vars.iter().map(|v| crate::tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        // materialize operator terms
        let mut names = Vec::new();
        let mut tmp = Vec::new();
        for (j, v) in vars.iter().enumerate() {
            if ds.get_var(v).is_some() {
                names.push(v.clone());
            } else {
                let ex = ferrostat_parser::parser::parse_expression(v).map_err(|e| rc_err(198, e.message))?;
                let env = crate::eval::Env::with_store(ds, st);
                let vals: Vec<f64> = (0..ds.n_obs).map(|i| env.eval(&ex, i).and_then(|x| x.num()).unwrap_or(ferrostat_core::missing::SYSMISS)).collect();
                let nm = format!("__xtc{j}");
                ds.add_variable(util::new_double(&nm, vals)).map_err(|e| rc_err(198, e))?;
                tmp.push(nm.clone());
                names.push(nm);
            }
        }
        let loaded = load(ds, st, cmd, &names);
        for t in &tmp {
            let _ = ds.drop_variable(t);
        }
        let (mut panels, ivar, tvar) = loaded?;
        let dm = cmd.has_opt("demean", 3);
        if dm {
            demean(&mut panels);
        }
        let ng = panels.len();
        let k = vars.len() - 1;
        let tbar = panels.iter().map(|p| p.y.len()).sum::<usize>() as f64 / ng as f64;
        let tmin = panels.iter().map(|p| p.y.len()).min().unwrap_or(0);
        if tmin < k + 5 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let trend = cmd.has_opt("trend", 3);
        let (bw, bwlab) = bandwidth(cmd, tbar)?;
        let rule = lag_rule(cmd, tbar)?;
        let out = match test.as_str() {
            "kao" => kao(&panels, bw, &bwlab, rule)?,
            "pedroni" | "ped" => {
                let ar_same = match cmd.opt("ar", 2).map(|s| s.trim().to_lowercase()) {
                    None => false,
                    Some(s) if "panelspecific".starts_with(&s) => false,
                    Some(s) if s == "same" => true,
                    Some(s) => return Err(rc_err(198, format!("ar({s}) invalid"))),
                };
                pedroni(&panels, k, trend, ar_same, bw, &bwlab, rule)
            }
            "westerlund" | "wes" => {
                let all = cmd.has_opt("allpanels", 3);
                westerlund(&panels, k, trend, all)
            }
            o => return Err(rc_err(198, format!("unknown test {o}; use kao, pedroni or westerlund"))),
        };
        let mut lines = vec![String::new(), out.title.clone(), "-".repeat(out.title.chars().count())];
        lines.push(format!("H0: No cointegration{:>21}Number of panels       = {:>6}", "", ng));
        let per = if panels.iter().all(|p| p.y.len() == tmin) { format!("{:<23}= {:>6}", "Number of periods", tmin) } else { format!("{:<23}= {:>6.2}", "Avg. number of periods", tbar) };
        lines.push(format!("Ha: {:<37}{per}", out.ha));
        lines.push(String::new());
        for (a, b, c, d) in &out.info {
            lines.push(format!("{:<22}{:<19}{:<18}{}", a, b, c, d).trim_end().to_string());
        }
        lines.push(String::new());
        lines.push("-".repeat(78));
        lines.push(format!("{:<47}{:>9}{:>22}", "", "Statistic", "p-value"));
        lines.push("-".repeat(78));
        let mut sm = Matrix::zeros(out.stats.len(), 2);
        for (i, (lab, v)) in out.stats.iter().enumerate() {
            let p = dist::normal(*v);
            sm.data[i * 2] = *v;
            sm.data[i * 2 + 1] = p;
            lines.push(format!(" {:<46}{:>9.4}{:>22.4}", lab, v, p));
        }
        lines.push("-".repeat(78));
        let mut r = Results::new();
        r.set("N", panels.iter().map(|p| p.y.len()).sum::<usize>() as f64);
        r.set("N_g", ng as f64);
        r.set("N_t", tbar);
        r.set_str("test", test.clone());
        r.set_str("panelvar", ivar);
        r.set_str("timevar", tvar);
        let rn: Vec<String> = out.stats.iter().map(|s| s.0.replace(' ', "_").replace('\u{2013}', "-")).collect();
        r.set_mat("stats", Matrix::row_vector(&out.stats.iter().map(|s| s.1).collect::<Vec<_>>()).with_names(vec!["stat".into()], rn.clone()));
        r.set_mat("p", Matrix::row_vector(&sm.data.chunks(2).map(|c| c[1]).collect::<Vec<_>>()).with_names(vec!["p".into()], rn));
        st.r = r;
        Ok(lines.join("\n"))
    }
}

fn kao(panels: &[Panel], bw: f64, bwlab: &str, rule: LagRule) -> Result<Out, String> {
    let ng = panels.len() as f64;
    let k = panels[0].x.len();
    // within (fixed-effects) pooled regression
    let mut ys = Vec::new();
    let mut xs: Vec<Vec<f64>> = vec![Vec::new(); k];
    for p in panels {
        let my = util::mean(&p.y);
        ys.extend(p.y.iter().map(|v| v - my));
        for j in 0..k {
            let mx = util::mean(&p.x[j]);
            xs[j].extend(p.x[j].iter().map(|v| v - mx));
        }
    }
    let (b, _, _) = ls(&xs, &ys);
    let resid: Vec<Vec<f64>> = panels
        .iter()
        .map(|p| {
            let my = util::mean(&p.y);
            let mx: Vec<f64> = p.x.iter().map(|c| util::mean(c)).collect();
            (0..p.y.len()).map(|t| p.y[t] - my - (0..k).map(|j| b[j] * (p.x[j][t] - mx[j])).sum::<f64>()).collect()
        })
        .collect();
    // DF regression pooled, no constant
    let mut num = 0.0;
    let mut den = 0.0;
    let mut nobs = 0usize;
    for e in &resid {
        for t in 1..e.len() {
            num += e[t] * e[t - 1];
            den += e[t - 1] * e[t - 1];
            nobs += 1;
        }
    }
    let rho = num / den;
    let mut ssr = 0.0;
    for e in &resid {
        for t in 1..e.len() {
            ssr += (e[t] - rho * e[t - 1]).powi(2);
        }
    }
    let se2 = ssr / nobs as f64;
    let t_rho = (rho - 1.0) * den.sqrt() / se2.sqrt();
    let tt = nobs as f64 / ng;
    // long-run covariance of (u, dx), u estimated by the DF-regression residuals
    let mut s_acc = Matrix::zeros(k + 1, k + 1);
    let mut o_acc = Matrix::zeros(k + 1, k + 1);
    for (p, e) in panels.iter().zip(&resid) {
        let w: Vec<Vec<f64>> = (1..e.len()).map(|t| std::iter::once(e[t] - rho * e[t - 1]).chain((0..k).map(|j| p.x[j][t] - p.x[j][t - 1])).collect()).collect();
        let (s, o) = lrcov(&w, bw);
        s_acc = s_acc.add(&s).unwrap();
        o_acc = o_acc.add(&o).unwrap();
    }
    let sv2 = cond_var(&s_acc.scale(1.0 / ng)).max(1e-300);
    let s0v2 = cond_var(&o_acc.scale(1.0 / ng)).max(1e-300);
    let (sv, s0v) = (sv2.sqrt(), s0v2.sqrt());
    let rn = ng.sqrt() * tt * (rho - 1.0);
    let df_rho_star = (rn + 3.0 * ng.sqrt() * sv2 / s0v2) / (3.0 + 36.0 * sv2 * sv2 / (5.0 * s0v2 * s0v2)).sqrt();
    let adj_den = (s0v2 / (2.0 * sv2) + 3.0 * sv2 / (10.0 * s0v2)).sqrt();
    let df_t_star = (t_rho + (6.0 * ng).sqrt() * sv / (2.0 * s0v)) / adj_den;
    let (t_adf, p_used) = adf_t(&resid, rule, true)[0];
    let adf = (t_adf + (6.0 * ng).sqrt() * sv / (2.0 * s0v)) / adj_den;
    let df_rho = (rn + 3.0 * ng.sqrt()) / 10.2f64.sqrt();
    let df_t = 1.25f64.sqrt() * t_rho + (1.875 * ng).sqrt();
    Ok(Out {
        title: "Kao test for cointegration".into(),
        ha: "All panels are cointegrated".into(),
        info: vec![
            ("Cointegrating vector:".into(), "Same".into(), String::new(), String::new()),
            ("Panel means:".into(), "Included".into(), "Kernel:".into(), "Bartlett".into()),
            ("Time trend:".into(), "Not included".into(), "Lags:".into(), format!("{bw:.2} ({bwlab})")),
            ("AR parameter:".into(), "Same".into(), "Augmented lags:".into(), format!("{p_used}{}", lag_note(rule))),
        ],
        stats: vec![
            ("Modified Dickey\u{2013}Fuller t".into(), df_rho_star),
            ("Dickey\u{2013}Fuller t".into(), df_t_star),
            ("Augmented Dickey\u{2013}Fuller t".into(), adf),
            ("Unadjusted modified Dickey\u{2013}Fuller t".into(), df_rho),
            ("Unadjusted Dickey\u{2013}Fuller t".into(), df_t),
        ],
    })
}

fn lag_note(rule: LagRule) -> &'static str {
    match rule {
        LagRule::Fixed(_) => "",
        LagRule::Aic(_) => " (AIC)",
        LagRule::Bic(_) => " (BIC)",
        LagRule::Hqic(_) => " (HQIC)",
    }
}

const SIM_REPS: usize = 2000;

fn pedroni(panels: &[Panel], k: usize, trend: bool, same: bool, bw: f64, bwlab: &str, rule: LagRule) -> Out {
    let ng = panels.len() as f64;
    let resid: Vec<Vec<f64>> = panels.iter().map(|p| coint_resid(p, true, trend)).collect();
    let parts: Vec<(f64, f64, f64)> = resid.iter().map(|e| pp_parts(e, bw)).collect();
    let adfs = adf_t(&resid, rule, false);
    let tlen = (panels.iter().map(|p| p.y.len()).sum::<usize>() as f64 / ng).round() as usize;
    let p_mode = {
        let mut ps: Vec<usize> = adfs.iter().map(|a| a.1).collect();
        ps.sort();
        ps[ps.len() / 2]
    };
    let sims = simulate_null(tlen, k, true, trend, bw, p_mode, SIM_REPS);
    let tt = (tlen - 1) as f64;
    let stats = if !same {
        // group-mean statistics: per-panel rho, t, adf
        let rho_i = |a: f64, b: f64| tt * a / b;
        let t_i = |a: f64, b: f64, s: f64| a / (s * b).sqrt();
        let std_group = |obs: Vec<f64>, null: Vec<f64>| -> f64 {
            let null: Vec<f64> = null.into_iter().filter(|v| v.is_finite()).collect();
            let m = util::mean(&null);
            let v = util::variance(&null);
            let o: Vec<f64> = obs.into_iter().filter(|v| v.is_finite()).collect();
            let n = o.len() as f64;
            (o.iter().sum::<f64>() / n.sqrt() - m * n.sqrt()) / v.sqrt()
        };
        let z_rho = std_group(parts.iter().map(|p| rho_i(p.0, p.1)).collect(), sims.iter().map(|d| rho_i(d[0], d[1])).collect());
        let z_t = std_group(parts.iter().map(|p| t_i(p.0, p.1, p.2)).collect(), sims.iter().map(|d| t_i(d[0], d[1], d[2])).collect());
        let z_adf = std_group(adfs.iter().map(|a| a.0).collect(), sims.iter().map(|d| d[3]).collect());
        vec![z_rho, z_t, z_adf]
    } else {
        // pooled (within-dimension) statistics, standardized by the delta method
        let d: Vec<Vec<f64>> = sims.iter().filter(|d| d.iter().all(|v| v.is_finite())).map(|d| vec![d[0], d[1], d[2], d[3]]).collect();
        let (m, c) = mean_cov(&d);
        let sm = vec![
            parts.iter().map(|p| p.0).sum::<f64>() / ng,
            parts.iter().map(|p| p.1).sum::<f64>() / ng,
            parts.iter().map(|p| p.2).sum::<f64>() / ng,
            adfs.iter().map(|a| a.0).filter(|v| v.is_finite()).sum::<f64>() / ng,
        ];
        let z_rho = standardize(&|v: &[f64]| v[0] / v[1], &sm, &m, &c, ng);
        let z_t = standardize(&|v: &[f64]| v[0] / (v[2] * v[1]).sqrt(), &sm, &m, &c, ng);
        let z_adf = standardize(&|v: &[f64]| v[3], &sm, &m, &c, ng);
        vec![z_rho, z_t, z_adf]
    };
    Out {
        title: "Pedroni test for cointegration".into(),
        ha: "All panels are cointegrated".into(),
        info: vec![
            ("Cointegrating vector:".into(), "Panel specific".into(), String::new(), String::new()),
            ("Panel means:".into(), "Included".into(), "Kernel:".into(), "Bartlett".into()),
            ("Time trend:".into(), if trend { "Included" } else { "Not included" }.into(), "Lags:".into(), format!("{bw:.2} ({bwlab})")),
            ("AR parameter:".into(), if same { "Same" } else { "Panel specific" }.into(), "Augmented lags:".into(), format!("{p_mode}{}", lag_note(rule))),
        ],
        stats: vec![
            ("Modified Phillips\u{2013}Perron t".into(), stats[0]),
            ("Phillips\u{2013}Perron t".into(), stats[1]),
            ("Augmented Dickey\u{2013}Fuller t".into(), stats[2]),
        ],
    }
}

fn westerlund(panels: &[Panel], k: usize, trend: bool, all: bool) -> Out {
    let ng = panels.len() as f64;
    let resid: Vec<Vec<f64>> = panels.iter().map(|p| coint_resid(p, true, trend)).collect();
    let parts: Vec<(f64, f64)> = resid.iter().map(|e| vr_parts(e)).collect();
    let tlen = (panels.iter().map(|p| p.y.len()).sum::<usize>() as f64 / ng).round() as usize;
    let sims = simulate_null(tlen, k, true, trend, 1.0, 0, SIM_REPS);
    let z = if all {
        let d: Vec<Vec<f64>> = sims.iter().map(|d| vec![d[4], d[5]]).collect();
        let (m, c) = mean_cov(&d);
        let sm = vec![parts.iter().map(|p| p.0).sum::<f64>() / ng, parts.iter().map(|p| p.1).sum::<f64>() / ng];
        standardize(&|v: &[f64]| v[0] / v[1], &sm, &m, &c, ng)
    } else {
        let null: Vec<f64> = sims.iter().map(|d| d[4] / d[5]).collect();
        let (m, v) = (util::mean(&null), util::variance(&null));
        let o: Vec<f64> = parts.iter().map(|p| p.0 / p.1).collect();
        (o.iter().sum::<f64>() / ng.sqrt() - m * ng.sqrt()) / v.sqrt()
    };
    Out {
        title: "Westerlund test for cointegration".into(),
        ha: if all { "All panels are cointegrated".into() } else { "Some panels are cointegrated".into() },
        info: vec![
            ("Cointegrating vector:".into(), "Panel specific".into(), String::new(), String::new()),
            ("Panel means:".into(), "Included".into(), String::new(), String::new()),
            ("Time trend:".into(), if trend { "Included" } else { "Not included" }.into(), String::new(), String::new()),
            ("AR parameter:".into(), if all { "Same" } else { "Panel specific" }.into(), String::new(), String::new()),
        ],
        stats: vec![("Variance ratio".into(), z)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_run_variance_of_white_noise_close_to_variance() {
        let mut rng = Rng::new(7);
        let w: Vec<Vec<f64>> = (0..4000).map(|_| vec![rng.normal()]).collect();
        let (s, o) = lrcov(&w, 3.0);
        assert!((s[(0, 0)] - 1.0).abs() < 0.08);
        assert!((o[(0, 0)] - s[(0, 0)]).abs() < 0.1);
    }

    #[test]
    fn variance_ratio_parts_of_constant_series() {
        let (a, b) = vr_parts(&[1.0, 1.0, 1.0, 1.0]);
        // partial sums 1,2,3,4 -> 30 / 64 ; R/T = 1
        assert!((a - 30.0 / 64.0).abs() < 1e-12 && (b - 1.0).abs() < 1e-12);
    }
}
