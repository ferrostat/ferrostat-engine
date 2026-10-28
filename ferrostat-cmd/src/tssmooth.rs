//! `tssmooth` (moving averages, exponential, double-exponential and
//! Holt–Winters smoothers, Tukey's nonlinear smoothers) and `tsfilter`
//! (Hodrick–Prescott, Baxter–King, Christiano–Fitzgerald and Butterworth
//! filters).
//!
//! References: Montgomery, Johnson & Gardiner, *Forecasting and Time Series
//! Analysis*, 1990; Velleman, "Definition and comparison of robust nonlinear
//! data smoothing algorithms", *JASA* 75 (1980); Hodrick & Prescott,
//! *JMCB* 29 (1997); Baxter & King, *REStat* 81 (1999); Christiano &
//! Fitzgerald, *IER* 44 (2003); Pollock, *J. Econometrics* 2000.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::tsops;
use crate::util;

/// `[type] newvar = exp` -> (newvar, exp)
fn assignment(text: &str) -> Result<(String, String), String> {
    let (l, r) = text.split_once('=').ok_or_else(|| rc_err(198, "= exp required"))?;
    let words: Vec<&str> = l.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [t, n] if ferrostat_core::VarType::from_type_str(t).is_some() => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    Ok((name, r.trim().to_string()))
}

/// Time-ordered sample of an expression.
fn sample(ds: &DataSet, st: &Store, cmd: &Command, exp: &str) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>), String> {
    let x = tsops::values(ds, st, exp)?;
    let mask = util::touse(ds, st, cmd)?;
    let (rows, times) = tsops::ordered_rows(ds, st, &mask, &[&x])?;
    if rows.len() < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let v = rows.iter().map(|&i| x[i]).collect();
    Ok((rows, times, v))
}

/// Write a new variable (or replace) with values at `rows`.
fn write_var(ds: &mut DataSet, name: &str, rows: &[usize], vals: &[f64], replace: bool) -> Result<(), String> {
    let mut all = vec![SYSMISS; ds.n_obs];
    for (&i, &v) in rows.iter().zip(vals) {
        all[i] = if v.is_finite() { v } else { SYSMISS };
    }
    if ds.var_index.contains_key(name) {
        if !replace {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        let idx = ds.var_index[name];
        ds.vars.remove(idx);
        ds.var_index.clear();
        for (i, v) in ds.vars.iter().enumerate() {
            ds.var_index.insert(v.name.clone(), i);
        }
    }
    tsops::add_var(ds, name, all)
}

/// Append `h` observations after the last time period; returns their rows.
fn extend_time(ds: &mut DataSet, st: &Store, last_t: f64, h: usize) -> Result<Vec<usize>, String> {
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let mut rows = Vec::new();
    for k in 1..=h {
        let t = last_t + k as f64;
        if let Some(i) = (0..ds.n_obs).find(|&i| tc[i] == t) {
            rows.push(i);
            continue;
        }
        let n = ds.n_obs;
        crate::data::set_obs(ds, n + 1)?;
        let ti = ds.var_index[&tvar];
        let ty = ds.vars[ti].var_type;
        ds.vars[ti].set(n, util::typed_value(ty, t));
        rows.push(n);
    }
    Ok(rows)
}

fn golden(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if (b - a).abs() < 1e-9 {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// Nelder–Mead minimisation inside the unit cube.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64]) -> Vec<f64> {
    let n = x0.len();
    let clampv = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| v.clamp(1e-6, 1.0 - 1e-6)).collect() };
    let mut simplex: Vec<Vec<f64>> = vec![clampv(x0)];
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] = if p[i] > 0.5 { p[i] - 0.2 } else { p[i] + 0.2 };
        simplex.push(clampv(&p));
    }
    let mut vals: Vec<f64> = simplex.iter().map(|p| f(p)).collect();
    for _ in 0..2000 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() <= 1e-12 * (vals[0].abs() + 1e-12) && simplex.iter().all(|p| p.iter().zip(&simplex[0]).all(|(a, b)| (a - b).abs() < 1e-7)) {
            break;
        }
        let cen: Vec<f64> = (0..n).map(|j| simplex[..n].iter().map(|p| p[j]).sum::<f64>() / n as f64).collect();
        let pt = |t: f64| -> Vec<f64> { clampv(&(0..n).map(|j| cen[j] + t * (simplex[n][j] - cen[j])).collect::<Vec<_>>()) };
        let xr = pt(-1.0);
        let fr = f(&xr);
        if fr < vals[0] {
            let xe = pt(-2.0);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                vals[n] = fe;
            } else {
                simplex[n] = xr;
                vals[n] = fr;
            }
        } else if fr < vals[n - 1] {
            simplex[n] = xr;
            vals[n] = fr;
        } else {
            let xc = pt(0.5);
            let fc = f(&xc);
            if fc < vals[n] {
                simplex[n] = xc;
                vals[n] = fc;
            } else {
                for i in 1..=n {
                    simplex[i] = clampv(&(0..n).map(|j| simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])).collect::<Vec<_>>());
                    vals[i] = f(&simplex[i]);
                }
            }
        }
    }
    let best = (0..=n).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    simplex[best].clone()
}

/// Intercept (at t = 0) and slope of a regression of x_1..x_m on t = 1..m.
fn trend_fit(x: &[f64]) -> (f64, f64) {
    let m = x.len() as f64;
    let tb = (m + 1.0) / 2.0;
    let xb = x.iter().sum::<f64>() / m;
    let sxy: f64 = x.iter().enumerate().map(|(i, v)| ((i + 1) as f64 - tb) * (v - xb)).sum();
    let sxx: f64 = (1..=x.len()).map(|i| (i as f64 - tb).powi(2)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (xb - b * tb, b)
}

fn parms(cmd: &Command, n: usize) -> Result<Option<Vec<f64>>, String> {
    let Some(p) = cmd.opt("parms", 1) else { return Ok(None) };
    let v: Vec<f64> = p.split_whitespace().map(|w| w.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "invalid parms()"))?;
    if v.len() != n || v.iter().any(|x| !(*x >= 0.0 && *x <= 1.0)) {
        return Err(rc_err(198, "parms() invalid; parameters must be between 0 and 1"));
    }
    Ok(Some(v))
}

// ---------------------------------------------------------------------------
// recursions
// ---------------------------------------------------------------------------

/// Single exponential smoothing: smoothed series, one-step SSE.
pub fn exponential(x: &[f64], a: f64, s0: f64) -> (Vec<f64>, f64) {
    let mut s = s0;
    let mut sse = 0.0;
    let out = x.iter().map(|&v| {
        sse += (v - s).powi(2);
        s = a * v + (1.0 - a) * s;
        s
    }).collect();
    (out, sse)
}

/// Double exponential smoothing: (S1, S2, SSE of one-step forecasts).
pub fn dexponential(x: &[f64], a: f64, s10: f64, s20: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let (mut s1, mut s2) = (s10, s20);
    let mut sse = 0.0;
    let mut o1 = Vec::new();
    let mut o2 = Vec::new();
    for &v in x {
        let k = a / (1.0 - a);
        let f = (2.0 + k) * s1 - (1.0 + k) * s2;
        sse += (v - f).powi(2);
        s1 = a * v + (1.0 - a) * s1;
        s2 = a * s1 + (1.0 - a) * s2;
        o1.push(s1);
        o2.push(s2);
    }
    (o1, o2, sse)
}

/// Holt–Winters nonseasonal: (one-step predictions, final level, final slope, SSE).
pub fn holt(x: &[f64], a: f64, b: f64, l0: f64, t0: f64) -> (Vec<f64>, f64, f64, f64) {
    let (mut l, mut t) = (l0, t0);
    let mut sse = 0.0;
    let mut pred = Vec::new();
    for &v in x {
        let f = l + t;
        pred.push(f);
        sse += (v - f).powi(2);
        let nl = a * v + (1.0 - a) * (l + t);
        t = b * (nl - l) + (1.0 - b) * t;
        l = nl;
    }
    (pred, l, t, sse)
}

/// Seasonal Holt–Winters (multiplicative or additive).
pub fn winters(x: &[f64], p: &[f64], period: usize, additive: bool) -> (Vec<f64>, f64, f64, Vec<f64>, f64) {
    let (a, b, g) = (p[0], p[1], p[2]);
    let lp = period;
    let m1 = x[..lp].iter().sum::<f64>() / lp as f64;
    let m2 = if x.len() >= 2 * lp { x[lp..2 * lp].iter().sum::<f64>() / lp as f64 } else { m1 };
    let mut t = (m2 - m1) / lp as f64;
    // level at the end of the first season
    let mut l = m1 + t * (lp as f64 - 1.0) / 2.0;
    let mut s: Vec<f64> = (0..lp)
        .map(|j| {
            let trend_j = m1 + t * (j as f64 - (lp as f64 - 1.0) / 2.0);
            if additive { x[j] - trend_j } else { x[j] / trend_j }
        })
        .collect();
    let mut pred: Vec<f64> = x[..lp].to_vec();
    let mut sse = 0.0;
    for (i, &v) in x.iter().enumerate().skip(lp) {
        let si = s[i % lp];
        let f = if additive { l + t + si } else { (l + t) * si };
        pred.push(f);
        sse += (v - f).powi(2);
        let nl = if additive { a * (v - si) + (1.0 - a) * (l + t) } else { a * v / si + (1.0 - a) * (l + t) };
        t = b * (nl - l) + (1.0 - b) * t;
        l = nl;
        s[i % lp] = if additive { g * (v - l) + (1.0 - g) * si } else { g * v / l + (1.0 - g) * si };
    }
    (pred, l, t, s, sse)
}

// ---------------------------------------------------------------------------
// Tukey nonlinear smoothers
// ---------------------------------------------------------------------------

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Running median of span k (odd: centred; even: shifted half a period,
/// the following even span re-centres). Ends use shorter spans.
fn running_median(x: &[f64], k: usize, shift_back: bool) -> Vec<f64> {
    let n = x.len();
    if k <= 1 {
        return x.to_vec();
    }
    (0..n)
        .map(|t| {
            let (lo, hi) = if k % 2 == 1 {
                let h = k / 2;
                let h = h.min(t).min(n - 1 - t);
                (t - h, t + h)
            } else {
                // window [t - k/2, t + k/2 - 1] or [t - k/2 + 1, t + k/2]
                let h = k / 2;
                let (lo, hi) = if shift_back { (t as i64 - h as i64, t as i64 + h as i64 - 1) } else { (t as i64 - h as i64 + 1, t as i64 + h as i64) };
                let r = (t as i64 - lo).min(hi - t as i64).min(t as i64).min((n - 1 - t) as i64).max(0);
                if lo < 0 || hi >= n as i64 {
                    ((t as i64 - r) as usize, (t as i64 + r) as usize)
                } else {
                    (lo as usize, hi as usize)
                }
            };
            median(&mut x[lo..=hi].to_vec())
        })
        .collect()
}

fn hanning(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|t| if t == 0 || t + 1 == n { x[t] } else { (x[t - 1] + 2.0 * x[t] + x[t + 1]) / 4.0 }).collect()
}

fn endpoints(y: &mut [f64], x: &[f64]) {
    let n = y.len();
    if n < 3 {
        return;
    }
    y[0] = median(&mut [x[0], y[1], 3.0 * y[1] - 2.0 * y[2]]);
    y[n - 1] = median(&mut [x[n - 1], y[n - 2], 3.0 * y[n - 2] - 2.0 * y[n - 3]]);
}

fn split(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut y = x.to_vec();
    if n < 6 {
        return y;
    }
    for t in 2..n - 3 {
        if x[t] == x[t + 1] && ((x[t - 1] < x[t] && x[t + 2] < x[t]) || (x[t - 1] > x[t] && x[t + 2] > x[t])) {
            y[t] = median(&mut [x[t], x[t - 1], 3.0 * x[t - 1] - 2.0 * x[t - 2]]);
            y[t + 1] = median(&mut [x[t + 1], x[t + 2], 3.0 * x[t + 2] - 2.0 * x[t + 3]]);
        }
    }
    y
}

/// Apply a smoother specification such as `3RSSH` or `4253EH`.
pub fn tukey(x: &[f64], spec: &str) -> Result<Vec<f64>, String> {
    let chars: Vec<char> = spec.to_uppercase().chars().collect();
    let mut y = x.to_vec();
    let mut even_pending = false;
    let mut i = 0;
    let mut last: Option<char> = None;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '1'..='9' => {
                let k = c.to_digit(10).unwrap() as usize;
                let rep = i + 1 < chars.len() && chars[i + 1] == 'R';
                loop {
                    let ny = if k % 2 == 0 {
                        let r = running_median(&y, k, !even_pending);
                        even_pending = !even_pending;
                        r
                    } else {
                        running_median(&y, k, false)
                    };
                    let same = ny == y;
                    y = ny;
                    if !rep || same {
                        break;
                    }
                }
                if rep {
                    i += 1;
                }
                last = Some(c);
            }
            'S' => {
                y = split(&y);
                // splitting is followed by re-smoothing with the previous median
                if let Some(d) = last.and_then(|d| d.to_digit(10)) {
                    y = running_median(&y, d as usize, false);
                }
            }
            'H' => y = hanning(&y),
            'E' => {
                let x0 = y.clone();
                endpoints(&mut y, &x0);
            }
            'R' => {}
            _ => return Err(rc_err(198, format!("invalid smoother {spec}"))),
        }
        i += 1;
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// tssmooth
// ---------------------------------------------------------------------------

pub struct Tssmooth;

impl CommandHandler for Tssmooth {
    fn name(&self) -> &'static str {
        "tssmooth"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (method, rest) = main.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "smoother and newvar = exp required"))?;
        let method = method.to_lowercase();
        let (newvar, exp) = assignment(rest)?;
        tsops::time_var(st)?;
        let replace = cmd.has_opt("replace", 3);
        if !replace && ds.var_index.contains_key(&newvar) {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        let (rows, times, x) = sample(ds, st, cmd, &exp)?;
        let n = x.len();
        let horizon = util::opt_num(cmd, "forecast", 1)?.unwrap_or(0.0).max(0.0) as usize;
        let samp0 = util::opt_num(cmd, "samp0", 4)?.map(|v| v as usize);
        let s0 = cmd.opt("s0", 2).map(|s| s.split_whitespace().map(|w| w.parse::<f64>()).collect::<Result<Vec<_>, _>>()).transpose().map_err(|_| rc_err(198, "invalid s0()"))?;
        let mut out = Vec::new();
        let tv = tsops::time_var(st).unwrap_or_default();
        let r = st.begin_r();
        r.set("N", n as f64);
        r.set_str("exp", exp.clone());
        r.set_str("timevar", tv);
        let (vals, fc): (Vec<f64>, Vec<f64>) = match method.as_str() {
            "ma" => {
                let (lagw, w0, leadw, text) = ma_weights(cmd)?;
                let mut y = vec![SYSMISS; n];
                for t in 0..n {
                    let mut s = 0.0;
                    let mut ws = 0.0;
                    let mut add = |j: i64, w: f64| {
                        if j >= 0 && (j as usize) < n && w != 0.0 {
                            s += w * x[j as usize];
                            ws += w;
                        }
                    };
                    for (l, w) in lagw.iter().enumerate() {
                        add(t as i64 - (lagw.len() - l) as i64, *w);
                    }
                    add(t as i64, w0);
                    for (l, w) in leadw.iter().enumerate() {
                        add(t as i64 + l as i64 + 1, *w);
                    }
                    if ws > 0.0 {
                        y[t] = s / ws;
                    }
                }
                out.push("The smoother applied was".to_string());
                out.push(format!("     {text}; x(t)= {exp}"));
                let r = &mut st.r;
                r.set("w0", w0);
                for (l, w) in lagw.iter().rev().enumerate() {
                    r.set(&format!("wlag{}", l + 1), *w);
                }
                for (l, w) in leadw.iter().enumerate() {
                    r.set(&format!("wlead{}", l + 1), *w);
                }
                r.set_str("method", "ma");
                (y, vec![])
            }
            "exponential" | "exp" | "exponen" => {
                let m = samp0.unwrap_or((n / 2).max(1)).min(n);
                let init = s0.as_ref().and_then(|v| v.first().copied()).unwrap_or_else(|| x[..m].iter().sum::<f64>() / m as f64);
                let (alpha, fixed) = match parms(cmd, 1)? {
                    Some(p) => (p[0], true),
                    None => (golden(&|a| exponential(&x, a, init).1, 0.0, 1.0), false),
                };
                let (y, sse) = exponential(&x, alpha, init);
                let rmse = (sse / n as f64).sqrt();
                if fixed {
                    out.push(format!("exponential coefficient      = {:>9.4}", alpha));
                    out.push(format!("sum-of-squared residuals     = {:>9}", fmt_g(sse, 9)));
                    out.push(format!("root mean squared error      = {:>9}", fmt_g(rmse, 9)));
                } else {
                    out.push("computing optimal exponential coefficient (0,1)".into());
                    out.push(String::new());
                    out.push(format!("optimal exponential coefficient = {:>9.4}", alpha));
                    out.push(format!("sum-of-squared residuals        = {:>9}", fmt_g(sse, 9)));
                    out.push(format!("root mean squared error         = {:>9}", fmt_g(rmse, 9)));
                }
                let r = &mut st.r;
                r.set("alpha", alpha);
                r.set("rss", sse);
                r.set("rmse", rmse);
                r.set("N_pre", m as f64);
                r.set("s1_0", init);
                r.set_str("method", "exponential");
                let last = *y.last().unwrap();
                (y, vec![last; horizon])
            }
            "dexponential" | "dexp" | "dexpon" => {
                let m = samp0.unwrap_or((n / 2).max(2)).min(n);
                let (a0, b0) = trend_fit(&x[..m]);
                let starts = |a: f64| -> (f64, f64) {
                    match &s0 {
                        Some(v) if v.len() >= 2 => (v[0], v[1]),
                        _ => {
                            let k = (1.0 - a) / a;
                            (a0 - k * b0, a0 - 2.0 * k * b0)
                        }
                    }
                };
                let sse_of = |a: f64| {
                    let (p, q) = starts(a);
                    dexponential(&x, a, p, q).2
                };
                let (alpha, fixed) = match parms(cmd, 1)? {
                    Some(p) => (p[0], true),
                    None => (golden(&sse_of, 1e-4, 1.0 - 1e-4), false),
                };
                let (p, q) = starts(alpha);
                let (s1, s2, sse) = dexponential(&x, alpha, p, q);
                let rmse = (sse / n as f64).sqrt();
                if fixed {
                    out.push(format!("double-exponential coefficient    = {:>9.4}", alpha));
                    out.push(format!("sum-of-squared residuals          = {:>9}", fmt_g(sse, 9)));
                    out.push(format!("root mean squared error           = {:>9}", fmt_g(rmse, 9)));
                } else {
                    out.push("computing optimal double-exponential coefficient (0,1)".into());
                    out.push(String::new());
                    out.push(format!("optimal double-exponential coefficient = {:>9.4}", alpha));
                    out.push(format!("sum-of-squared residuals               = {:>9}", fmt_g(sse, 9)));
                    out.push(format!("root mean squared error                = {:>9}", fmt_g(rmse, 9)));
                }
                let r = &mut st.r;
                r.set("alpha", alpha);
                r.set("rss", sse);
                r.set("rmse", rmse);
                r.set("N_pre", m as f64);
                r.set("s2_0", q);
                r.set("s1_0", p);
                r.set_str("method", "dexponential");
                let (l1, l2) = (*s1.last().unwrap(), *s2.last().unwrap());
                let k = alpha / (1.0 - alpha);
                let fc = (1..=horizon).map(|h| (2.0 * l1 - l2) + k * (l1 - l2) * h as f64).collect();
                (s2, fc)
            }
            "hwinters" | "hw" => {
                let m = samp0.unwrap_or((n / 2).max(2)).min(n);
                let (l0, t0) = match &s0 {
                    Some(v) if v.len() >= 2 => (v[0], v[1]),
                    _ => trend_fit(&x[..m]),
                };
                let (ab, fixed) = match parms(cmd, 2)? {
                    Some(p) => (p, true),
                    None => (nelder_mead(&|p| holt(&x, p[0], p[1], l0, t0).3, &[0.5, 0.5]), false),
                };
                let (pred, l, t, sse) = holt(&x, ab[0], ab[1], l0, t0);
                let rmse = (sse / n as f64).sqrt();
                if !fixed {
                    out.push("Optimal weights:".into());
                    out.push(String::new());
                }
                out.push(format!("{:>36}alpha = {:.4}", "", ab[0]));
                out.push(format!("{:>37}beta = {:.4}", "", ab[1]));
                out.push(format!("{}sum-of-squared residuals = {}", " ".repeat(14), fmt_g(sse, 9)));
                out.push(format!("{}root mean squared error = {}", " ".repeat(15), fmt_g(rmse, 9)));
                let r = &mut st.r;
                r.set("alpha", ab[0]);
                r.set("beta", ab[1]);
                r.set("rss", sse);
                r.set("rmse", rmse);
                r.set("N_pre", m as f64);
                r.set("s2_0", t0);
                r.set("s1_0", l0);
                r.set("constant", l);
                r.set("linear", t);
                r.set_str("method", "hwinters");
                let fc = (1..=horizon).map(|h| l + t * h as f64).collect();
                (pred, fc)
            }
            "shwinters" | "shw" => {
                let period = match util::opt_num(cmd, "period", 3)? {
                    Some(p) => p as usize,
                    None => tsops::periodicity(ds, &tsops::time_var(st)?).filter(|p| *p > 1.0 && *p < 100.0).map(|p| p as usize).ok_or_else(|| rc_err(198, "period() required"))?,
                };
                if n < 2 * period {
                    return Err(rc_err(2001, "insufficient observations"));
                }
                let additive = cmd.has_opt("additive", 3);
                let (p, fixed) = match parms(cmd, 3)? {
                    Some(p) => (p, true),
                    None => (nelder_mead(&|p| winters(&x, p, period, additive).4, &[0.5, 0.5, 0.5]), false),
                };
                let (pred, l, t, s, sse) = winters(&x, &p, period, additive);
                let rmse = (sse / (n - period) as f64).sqrt();
                if !fixed {
                    out.push("Optimal weights:".into());
                    out.push(String::new());
                }
                out.push(format!("{:>36}alpha = {:.4}", "", p[0]));
                out.push(format!("{:>37}beta = {:.4}", "", p[1]));
                out.push(format!("{:>36}gamma = {:.4}", "", p[2]));
                out.push(format!("{}sum-of-squared residuals = {}", " ".repeat(14), fmt_g(sse, 9)));
                out.push(format!("{}root mean squared error = {}", " ".repeat(15), fmt_g(rmse, 9)));
                let r = &mut st.r;
                r.set("alpha", p[0]);
                r.set("beta", p[1]);
                r.set("gamma", p[2]);
                r.set("rss", sse);
                r.set("rmse", rmse);
                r.set("period", period as f64);
                r.set("constant", l);
                r.set("linear", t);
                r.set_str("method", if additive { "additive" } else { "multiplicative" });
                let fc = (1..=horizon).map(|h| {
                    let si = s[(n + h - 1) % period];
                    if additive { l + t * h as f64 + si } else { (l + t * h as f64) * si }
                }).collect();
                (pred, fc)
            }
            "nl" => {
                let sm = cmd.opt("smoother", 3).ok_or_else(|| rc_err(198, "option smoother() required"))?.to_string();
                let (spec, twice) = match sm.split_once(',') {
                    Some((a, b)) if b.trim().eq_ignore_ascii_case("twice") => (a.trim().to_string(), true),
                    _ => (sm.trim().to_string(), false),
                };
                let mut y = tukey(&x, &spec)?;
                if twice {
                    let rough: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
                    let sr = tukey(&rough, &spec)?;
                    y = y.iter().zip(&sr).map(|(a, b)| a + b).collect();
                }
                st.r.set_str("method", sm.clone());
                (y, vec![])
            }
            o => return Err(rc_err(198, format!("{o} invalid smoother"))),
        };
        let gaps = tsops::gaps(&times);
        let mut all_rows = rows.clone();
        let mut all_vals = vals;
        if !fc.is_empty() {
            let ext = extend_time(ds, st, *times.last().unwrap(), fc.len())?;
            all_rows.extend(ext);
            all_vals.extend(fc);
        }
        let _ = gaps;
        write_var(ds, &newvar, &all_rows, &all_vals, replace)?;
        Ok(out.join("\n"))
    }
}

/// Weights of `tssmooth ma`: (lag weights oldest first, current, lead weights, description).
fn ma_weights(cmd: &Command) -> Result<(Vec<f64>, f64, Vec<f64>, String), String> {
    if let Some(w) = cmd.opt("window", 1) {
        let v: Vec<usize> = w.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "invalid window()"))?;
        let (l, c, f) = (v.first().copied().unwrap_or(0), v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0));
        if c > 1 {
            return Err(rc_err(198, "window(): the current-period weight must be 0 or 1"));
        }
        let total = l + c + f;
        if total == 0 {
            return Err(rc_err(198, "window() must include at least one term"));
        }
        let mut terms = Vec::new();
        terms.extend((1..=l).rev().map(|k| format!("x(t-{k})")));
        if c == 1 {
            terms.push("1*x(t)".into());
        }
        terms.extend((1..=f).map(|k| format!("x(t+{k})")));
        let text = format!("(1/{total})*[{}]", terms.join(" + "));
        return Ok((vec![1.0; l], c as f64, vec![1.0; f], text));
    }
    if let Some(w) = cmd.opt("weights", 1) {
        let lt = w.find('<').ok_or_else(|| rc_err(198, "weights(): current weight <#> required"))?;
        let gt = w.find('>').ok_or_else(|| rc_err(198, "weights(): current weight <#> required"))?;
        let nums = |s: &str| -> Result<Vec<f64>, String> { s.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| rc_err(198, "invalid weights()"))).collect() };
        let lag = nums(&w[..lt])?;
        let cur: f64 = w[lt + 1..gt].trim().parse().map_err(|_| rc_err(198, "invalid weights()"))?;
        let lead = nums(&w[gt + 1..])?;
        let total: f64 = lag.iter().chain(&lead).sum::<f64>() + cur;
        let mut terms = Vec::new();
        let l = lag.len();
        terms.extend(lag.iter().enumerate().map(|(i, wt)| format!("{}*x(t-{})", fmt_g(*wt, 8), l - i)));
        terms.push(format!("{}*x(t)", fmt_g(cur, 8)));
        terms.extend(lead.iter().enumerate().map(|(i, wt)| format!("{}*x(t+{})", fmt_g(*wt, 8), i + 1)));
        let text = format!("(1/{})*[{}]", fmt_g(total, 8), terms.join(" + "));
        return Ok((lag, cur, lead, text));
    }
    Err(rc_err(198, "either window() or weights() must be specified"))
}

// ---------------------------------------------------------------------------
// tsfilter
// ---------------------------------------------------------------------------

/// Solve a symmetric positive-definite banded system (bandwidth `bw`, rows
/// as dense vectors of the band: a[i][j] holds A[i][i + j], j = 0..=bw).
fn band_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let bw = a[0].len() - 1;
    // banded Cholesky: L stored as l[i][j] = L[i][i - j]
    let mut l = vec![vec![0.0; bw + 1]; n];
    for i in 0..n {
        for j in (0..=bw.min(i)).rev() {
            // element L[i][i-j]
            let col = i - j;
            let mut s = if j <= bw { a[col][j] } else { 0.0 };
            for k in 1..=bw {
                if col < k || j + k > bw {
                    continue;
                }
                // L[i][col-k] * L[col][col-k]
                s -= l[i][j + k] * l[col][k];
            }
            if j == 0 {
                l[i][0] = s.max(1e-300).sqrt();
            } else {
                l[i][j] = s / l[col][0];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 1..=bw.min(i) {
            s -= l[i][j] * y[i - j];
        }
        y[i] = s / l[i][0];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in 1..=bw {
            if i + j < n {
                s -= l[i + j][j] * x[i + j];
            }
        }
        x[i] = s / l[i][0];
    }
    x
}

fn binom_coefs(order: usize, sign: f64) -> Vec<f64> {
    let mut c = vec![1.0];
    for _ in 0..order {
        let mut n = vec![0.0; c.len() + 1];
        for (i, v) in c.iter().enumerate() {
            n[i] += v;
            n[i + 1] += sign * v;
        }
        c = n;
    }
    c
}

/// Banded Gram matrix of the (T-n) x T operator with rows `c` shifted:
/// G = C C' ((T-n) x (T-n)), returned in band form.
fn gram_band(c: &[f64], m: usize) -> Vec<Vec<f64>> {
    let bw = c.len() - 1;
    (0..m).map(|i| (0..=bw).map(|j| if i + j < m { (0..c.len()).filter(|&k| k + j < c.len()).map(|k| c[k] * c[k + j]).sum() } else { 0.0 }).collect()).collect()
}

/// Butterworth/Hodrick–Prescott high-pass component (Pollock 2000):
/// cycle = lambda Q (Omega_L + lambda Q'Q)^-1 Q'y, with Q' the n-th
/// difference operator and Omega_L from (1+L)^n (identity for HP).
pub fn pollock_cycle(y: &[f64], lambda: f64, order: usize, hp: bool) -> Vec<f64> {
    let t = y.len();
    let m = t - order;
    let d = binom_coefs(order, -1.0);
    let mut a = gram_band(&d, m);
    let low = if hp { vec![1.0] } else { binom_coefs(order, 1.0) };
    let gl = gram_band(&low, m);
    for i in 0..m {
        for j in 0..a[i].len() {
            a[i][j] *= lambda;
            if j < gl[i].len() {
                a[i][j] += gl[i][j];
            }
        }
    }
    // Q'y: n-th differences, (Q'y)_i = sum_k d_k y_{i + n - k}
    let qy: Vec<f64> = (0..m).map(|i| (0..=order).map(|k| d[k] * y[i + order - k]).sum()).collect();
    let bsol = band_solve(&a, &qy);
    // cycle = lambda Q b: (Q b)_t = sum_i d_{i+n-t} b_i
    (0..t)
        .map(|tt| {
            let mut s = 0.0;
            for k in 0..=order {
                // tt = i + order - k
                let i = tt as i64 - order as i64 + k as i64;
                if i >= 0 && (i as usize) < m {
                    s += d[k] * bsol[i as usize];
                }
            }
            lambda * s
        })
        .collect()
}

/// Baxter–King band-pass cycle (missing for the first and last K).
pub fn bk_cycle(y: &[f64], pl: f64, pu: f64, k: usize) -> Vec<f64> {
    let (w1, w2) = (std::f64::consts::TAU / pu, std::f64::consts::TAU / pl);
    let mut b: Vec<f64> = (0..=k).map(|j| if j == 0 { (w2 - w1) / std::f64::consts::PI } else { ((j as f64 * w2).sin() - (j as f64 * w1).sin()) / (std::f64::consts::PI * j as f64) }).collect();
    let sum = b[0] + 2.0 * b[1..].iter().sum::<f64>();
    let adj = sum / (2 * k + 1) as f64;
    for v in b.iter_mut() {
        *v -= adj;
    }
    let n = y.len();
    (0..n).map(|t| if t < k || t + k >= n { f64::NAN } else { b[0] * y[t] + (1..=k).map(|j| b[j] * (y[t - j] + y[t + j])).sum::<f64>() }).collect()
}

/// Christiano–Fitzgerald asymmetric band-pass filter (random-walk version).
pub fn cf_cycle(y0: &[f64], pl: f64, pu: f64, drift: bool) -> Vec<f64> {
    let n = y0.len();
    let y: Vec<f64> = if drift {
        let d = (y0[n - 1] - y0[0]) / (n - 1) as f64;
        y0.iter().enumerate().map(|(t, v)| v - t as f64 * d).collect()
    } else {
        y0.to_vec()
    };
    let (a, b) = (std::f64::consts::TAU / pu, std::f64::consts::TAU / pl);
    let bj = |j: usize| -> f64 { if j == 0 { (b - a) / std::f64::consts::PI } else { ((j as f64 * b).sin() - (j as f64 * a).sin()) / (std::f64::consts::PI * j as f64) } };
    let bv: Vec<f64> = (0..n).map(bj).collect();
    (0..n)
        .map(|t| {
            // t is 0-based; future terms j = 1..n-2-t, past j = 1..t-1
            let mut c = bv[0] * y[t];
            let nf = n - 1 - t; // number of future observations
            let np = t; // number of past observations
            // the end-point weights make the weights sum to zero
            for j in 1..nf {
                c += bv[j] * y[t + j];
            }
            c += (-0.5 * bv[0] - (1..nf).map(|j| bv[j]).sum::<f64>()) * y[n - 1];
            for j in 1..np {
                c += bv[j] * y[t - j];
            }
            c += (-0.5 * bv[0] - (1..np).map(|j| bv[j]).sum::<f64>()) * y[0];
            c
        })
        .collect()
}

pub struct Tsfilter;

impl CommandHandler for Tsfilter {
    fn name(&self) -> &'static str {
        "tsfilter"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (filter, rest) = main.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "filter and newvarlist = varlist required"))?;
        let filter = filter.to_lowercase();
        let (l, r) = rest.split_once('=').ok_or_else(|| rc_err(198, "= varlist required"))?;
        let srcs = crate::tsops::expand_text(r.trim());
        let srcs: Vec<String> = srcs.split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        let mut news: Vec<String> = l.split_whitespace().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).map(String::from).collect();
        if news.len() == 1 && news[0].ends_with('*') {
            let stub = news[0].trim_end_matches('*').to_string();
            news = srcs.iter().map(|s| format!("{stub}{}", s.replace('.', "_"))).collect();
        }
        if news.len() != srcs.len() {
            return Err(rc_err(198, "number of new variables must equal the number of variables to filter"));
        }
        let trends: Vec<String> = match cmd.opt("trend", 3) {
            Some(t) => {
                let v: Vec<String> = t.split_whitespace().map(String::from).collect();
                if v.len() == 1 && v[0].ends_with('*') {
                    let stub = v[0].trim_end_matches('*').to_string();
                    srcs.iter().map(|s| format!("{stub}{}", s.replace('.', "_"))).collect()
                } else {
                    v
                }
            }
            None => vec![],
        };
        let tvar = tsops::time_var(st)?;
        let per = tsops::periodicity(ds, &tvar);
        let replace = cmd.has_opt("replace", 3);
        // defaults scale with the periodicity of the data (quarterly as reference)
        let scale = per.map_or(1.0, |p| p / 4.0);
        let minp = util::opt_num(cmd, "minperiod", 4)?.unwrap_or(match per { Some(p) if p == 1.0 => 2.0, Some(p) => 6.0 * p / 4.0, None => 6.0 });
        let maxp = util::opt_num(cmd, "maxperiod", 4)?.unwrap_or(32.0 * scale);
        let mut out = Vec::new();
        st.begin_r();
        for (vi, src) in srcs.iter().enumerate() {
            let (rows, times, x) = sample(ds, st, cmd, src)?;
            if tsops::gaps(&times) > 0 {
                return Err(rc_err(498, format!("{src} has gaps; tsfilter requires a contiguous sample")));
            }
            let cycle: Vec<f64> = match filter.as_str() {
                "hp" => {
                    let lambda = util::opt_num(cmd, "smooth", 3)?.unwrap_or(1600.0 * scale.powi(4));
                    st.r.set("smooth", lambda);
                    pollock_cycle(&x, lambda, 2, true)
                }
                "bw" => {
                    let order = util::opt_num(cmd, "order", 3)?.unwrap_or(2.0) as usize;
                    let lambda = (1.0 / (std::f64::consts::PI / maxp).tan()).powi(2 * order as i32);
                    st.r.set("order", order as f64);
                    st.r.set("maxperiod", maxp);
                    pollock_cycle(&x, lambda, order, false)
                }
                "bk" => {
                    let k = util::opt_num(cmd, "smaorder", 3)?.unwrap_or(match per { Some(p) => (3.0 * p).round(), None => 12.0 }) as usize;
                    if 2 * k + 1 > x.len() {
                        return Err(rc_err(2001, "insufficient observations for smaorder()"));
                    }
                    st.r.set("smaorder", k as f64);
                    st.r.set("minperiod", minp);
                    st.r.set("maxperiod", maxp);
                    bk_cycle(&x, minp, maxp, k)
                }
                "cf" => {
                    st.r.set("minperiod", minp);
                    st.r.set("maxperiod", maxp);
                    cf_cycle(&x, minp, maxp, cmd.has_opt("drift", 3))
                }
                o => return Err(rc_err(198, format!("{o}: unknown filter"))),
            };
            write_var(ds, &news[vi], &rows, &cycle, replace)?;
            if let Some(tn) = trends.get(vi) {
                let tr: Vec<f64> = x.iter().zip(&cycle).map(|(a, c)| if c.is_finite() { a - c } else { f64::NAN }).collect();
                write_var(ds, tn, &rows, &tr, replace)?;
            }
            out.push(String::new());
        }
        st.r.set_str("method", filter.to_uppercase());
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hp_matches_dense_solution() {
        // (I + lambda K'K) tau = y with K the second-difference matrix
        let y: Vec<f64> = (0..30).map(|t| (t as f64 * 0.7).sin() + 0.05 * t as f64 * t as f64).collect();
        let lam = 100.0;
        let c = pollock_cycle(&y, lam, 2, true);
        let n = y.len();
        let mut a = crate::matrix::Matrix::identity(n);
        for i in 0..n - 2 {
            let d = [1.0, -2.0, 1.0];
            for p in 0..3 {
                for q in 0..3 {
                    a[(i + p, i + q)] += lam * d[p] * d[q];
                }
            }
        }
        let inv = a.inverse().unwrap();
        for t in 0..n {
            let tau: f64 = (0..n).map(|j| inv[(t, j)] * y[j]).sum();
            assert!((y[t] - tau - c[t]).abs() < 1e-8, "{t}");
        }
    }

    #[test]
    fn tukey_running_median() {
        let x = [1.0, 5.0, 2.0, 8.0, 3.0, 3.0, 9.0];
        assert_eq!(tukey(&x, "3").unwrap(), vec![1.0, 2.0, 5.0, 3.0, 3.0, 3.0, 9.0]);
        let h = tukey(&[0.0, 4.0, 0.0], "H").unwrap();
        assert_eq!(h, vec![0.0, 2.0, 0.0]);
    }
}
