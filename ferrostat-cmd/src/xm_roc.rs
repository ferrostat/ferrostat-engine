//! Small tools: `runtest` (test for random order) and `lnskew0`
//! (zero-skewness log transform).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("runtest", Box::new(Runtest)), ("lnskew0", Box::new(Lnskew0))]
}

pub struct Runtest;

impl CommandHandler for Runtest {
    fn name(&self) -> &'static str {
        "runtest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = xm::var_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[v.clone()]);
        let c = util::col(ds, &v)?;
        let x: Vec<f64> = util::rows(&mask).iter().map(|&i| c[i]).collect();
        if x.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let thresh = if cmd.has_opt("mean", 1) {
            util::mean(&x)
        } else {
            match util::opt_num(cmd, "thresh", 1)? {
                Some(t) => t,
                None => {
                    let mut s = x.clone();
                    s.sort_by(f64::total_cmp);
                    util::pctile_sorted(&s, 50.0)
                }
            }
        };
        // observations at the threshold are dropped unless split is given
        let split = cmd.has_opt("split", 1);
        let mut cls: Vec<bool> = Vec::new();
        let mut toggle = false;
        for &v in &x {
            if v == thresh && split {
                cls.push(toggle);
                toggle = !toggle;
            } else if v == thresh && !cmd.has_opt("thresh", 1) && !cmd.has_opt("mean", 1) {
                continue;
            } else {
                cls.push(v > thresh);
            }
        }
        let n = cls.len() as f64;
        let n1 = cls.iter().filter(|b| **b).count() as f64;
        let n0 = n - n1;
        let runs = 1 + cls.windows(2).filter(|w| w[0] != w[1]).count();
        let mu = 2.0 * n0 * n1 / n + 1.0;
        let var = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n * n * (n - 1.0));
        let mut diff = runs as f64 - mu;
        if cmd.has_opt("continuity", 1) {
            diff = if diff.abs() <= 0.5 { 0.0 } else { diff - 0.5 * diff.signum() };
        }
        let z = diff / var.sqrt();
        let p = 2.0 * dist::normal(-z.abs());
        let t = fmt_g(thresh, 8);
        let out = [
            format!(" N({v} <= {t}) = {}", n0),
            format!(" N({v} >  {t}) = {}", n1),
            format!("{:>w$} = {}", "obs", n, w = format!(" N({v} <= {t})").len()),
            format!("{:>w$} = {}", "N(runs)", runs, w = format!(" N({v} <= {t})").len()),
            format!("{:>w$} = {:.2}", "z", z, w = format!(" N({v} <= {t})").len()),
            format!("{:>w$} = {}", "Prob>|z|", fmt_g((p * 100.0).round() / 100.0, 4), w = format!(" N({v} <= {t})").len()),
        ];
        let r = st.begin_r();
        r.set("N", n);
        r.set("N_below", n0);
        r.set("N_above", n1);
        r.set("mean", mu);
        r.set("Var", var);
        r.set("n_runs", runs as f64);
        r.set("z", z);
        r.set("p", p);
        Ok(out.join("\n"))
    }
}

pub struct Lnskew0;

fn skew(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    m3 / m2.powf(1.5)
}

impl CommandHandler for Lnskew0 {
    fn name(&self) -> &'static str {
        "lnskew0"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (lhs, rhs) = cmd.main.split_once('=').ok_or_else(|| rc_err(198, "syntax is lnskew0 newvar = exp"))?;
        let words: Vec<&str> = lhs.split_whitespace().collect();
        let name = words.last().ok_or_else(|| rc_err(198, "new variable name required"))?.to_string();
        crate::varlist::check_name(&name)?;
        if ds.var_index.contains_key(&name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        let e = parse_expression(rhs.trim()).map_err(|e| rc_err(198, e.message))?;
        let mask = util::touse(ds, st, cmd)?;
        let vals = util::eval_col(ds, st, &e, &mask)?;
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(vals[i])).collect();
        let x: Vec<f64> = rows.iter().map(|&i| vals[i]).collect();
        let n = x.len();
        if n < 3 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // sign: ln(x - k) for right skew, ln(-x - k) for left skew
        let s0 = skew(&x);
        let sign = if s0 >= 0.0 { 1.0 } else { -1.0 };
        let sx: Vec<f64> = x.iter().map(|v| sign * v).collect();
        let mn = sx.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = sx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (mx - mn).max(1e-12);
        let f = |k: f64| -> f64 { skew(&sx.iter().map(|v| (v - k).ln()).collect::<Vec<_>>()) };
        // skewness of ln(x - k) increases from very negative (k -> min) to the
        // skewness of x (k -> -inf): find the root, solving f(k) = target
        let solve = |target: f64| -> Option<f64> {
            let mut hi = mn - range * 1e-9;
            let mut lo = mn - range;
            let mut it = 0;
            while f(lo) < target {
                lo = mn - (mn - lo) * 4.0;
                it += 1;
                if it > 60 {
                    return None;
                }
            }
            if f(hi) > target {
                return None;
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if f(mid) > target {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            Some(0.5 * (lo + hi))
        };
        let k = solve(0.0).ok_or_else(|| rc_err(498, "could not find zero-skewness transform"))?;
        let level = util::opt_num(cmd, "level", 1)?;
        let (mut lo, mut hi) = (SYSMISS, SYSMISS);
        if let Some(l) = level {
            let nf = n as f64;
            let se = (6.0 / nf).sqrt();
            let z = dist::invnormal(1.0 - (1.0 - l / 100.0) / 2.0);
            lo = solve(-z * se).unwrap_or(SYSMISS);
            hi = solve(z * se).unwrap_or(SYSMISS);
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
        }
        let mut out_vals = vec![SYSMISS; ds.n_obs];
        for &i in &rows {
            out_vals[i] = (sign * vals[i] - k).ln();
        }
        let sk = skew(&rows.iter().map(|&i| out_vals[i]).collect::<Vec<_>>());
        let lab = if sign > 0.0 { format!("ln({}-k)", rhs.trim()) } else { format!("ln(-{}-k)", rhs.trim()) };
        let mut var = util::new_double(&name, out_vals);
        var.label = if sign > 0.0 { format!("ln({}-{})", rhs.trim(), fmt_g(k, 7)) } else { format!("ln(-{}-{})", rhs.trim(), fmt_g(k, 7)) };
        ds.add_variable(var).map_err(|e| rc_err(110, e))?;
        let ci = if level.is_some() { format!("{:>10}  {:>10}", fmt_g(lo, 7), fmt_g(hi, 7)) } else { format!("{:^22}", "(not calculated)") };
        let lv = level.unwrap_or(95.0);
        let out = [
            format!("       Transform |         k     [{}% conf. interval]       Skewness", fmt_g(lv, 4)),
            "-----------------+--------------------------------------------------".to_string(),
            format!("{:>16} |  {:>9}   {}     {:>9}", crate::summarize::abbrev(&lab, 16), fmt_g(k, 7), ci, fmt_g(sk, 3)),
        ];
        let r = st.begin_r();
        r.set("gamma", k);
        r.set("skewness", sk);
        r.set("lb", lo);
        r.set("ub", hi);
        Ok(out.join("\n"))
    }
}
