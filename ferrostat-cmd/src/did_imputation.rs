//! `did_imputation`: the imputation estimator of Borusyak, Jaravel and
//! Spiess (2024) for staggered treatment adoption, with its companion
//! `event_plot`.
//!
//! Step 1 fits the model of untreated potential outcomes (fixed effects and
//! controls) on the untreated observations; step 2 imputes Y(0) for the
//! treated observations; step 3 averages the individual effects with the
//! estimand weights. Standard errors follow Theorem 3 of the paper: the
//! estimator is linear in the outcomes, the weights on untreated
//! observations come from the normal equations of the step-1 model, and
//! treated residuals are effects net of their (cohort x period) averages
//! computed with the "smart" weights of Section 4.3 (optionally leave-out).

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

/// Factors (codes over the estimation rows) and continuous controls.
pub(crate) struct FeModel {
    pub fac: Vec<(Vec<usize>, usize)>,
    pub ctl: Vec<Vec<f64>>,
}

pub(crate) struct Solved {
    /// fitted values for every row (NaN when a level is not identified)
    pub fit: Vec<f64>,
    pub b: Vec<f64>,
    pub iters: usize,
}

fn codes_of(cols: &[&[f64]], rows: usize) -> (Vec<usize>, usize) {
    let mut map: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let key: Vec<i64> = cols.iter().map(|c| (c[i] * 1e6).round() as i64).collect();
        let n = map.len();
        out.push(*map.entry(key).or_insert(n));
    }
    let n = map.len();
    (out, n)
}

impl FeModel {
    /// Solve the weighted normal equations restricted to rows in `s`,
    /// X_s' W X_s theta = rhs, where rhs is given per factor level and per
    /// control, by Gauss-Seidel sweeps. Returns X theta for all rows.
    pub fn solve(&self, s: &[bool], w: &[f64], rhs_f: &[Vec<f64>], rhs_c: &[f64], tol: f64, maxit: usize) -> Solved {
        let n = s.len();
        let nf = self.fac.len();
        let mut theta: Vec<Vec<f64>> = self.fac.iter().map(|(_, g)| vec![0.0; *g]).collect();
        let mut wsum: Vec<Vec<f64>> = self.fac.iter().map(|(_, g)| vec![0.0; *g]).collect();
        for (f, (c, _)) in self.fac.iter().enumerate() {
            for i in 0..n {
                if s[i] {
                    wsum[f][c[i]] += w[i];
                }
            }
        }
        let kc = self.ctl.len();
        let mut b = vec![0.0; kc];
        let xtx_inv = if kc > 0 {
            let mut m = Matrix::zeros(kc, kc);
            for a in 0..kc {
                for c in 0..=a {
                    let v: f64 = (0..n).filter(|&i| s[i]).map(|i| w[i] * self.ctl[a][i] * self.ctl[c][i]).sum();
                    m[(a, c)] = v;
                    m[(c, a)] = v;
                }
            }
            Some(m.sym_inverse(1e-12).0)
        } else {
            None
        };
        let mut fit = vec![0.0; n];
        let mut iters = 0;
        for it in 0..maxit {
            iters = it + 1;
            let mut change = 0.0f64;
            for f in 0..nf {
                let (codes, g) = &self.fac[f];
                let mut acc = vec![0.0; *g];
                for i in 0..n {
                    if s[i] {
                        acc[codes[i]] += w[i] * (fit[i] - theta[f][codes[i]]);
                    }
                }
                let mut newt = vec![0.0; *g];
                for l in 0..*g {
                    newt[l] = if wsum[f][l] > 0.0 { (rhs_f[f][l] - acc[l]) / wsum[f][l] } else { 0.0 };
                }
                for i in 0..n {
                    let d = newt[codes[i]] - theta[f][codes[i]];
                    if d != 0.0 {
                        fit[i] += d;
                    }
                }
                for l in 0..*g {
                    change = change.max((newt[l] - theta[f][l]).abs());
                }
                theta[f] = newt;
            }
            if let Some(inv) = &xtx_inv {
                let r: Vec<f64> = (0..kc).map(|a| rhs_c[a] - (0..n).filter(|&i| s[i]).map(|i| w[i] * self.ctl[a][i] * (fit[i] - xbc(&self.ctl, &b, i))).sum::<f64>()).collect();
                let nb: Vec<f64> = (0..kc).map(|a| (0..kc).map(|c| inv[(a, c)] * r[c]).sum()).collect();
                for i in 0..n {
                    let d: f64 = (0..kc).map(|a| (nb[a] - b[a]) * self.ctl[a][i]).sum();
                    fit[i] += d;
                }
                for a in 0..kc {
                    change = change.max((nb[a] - b[a]).abs());
                }
                b = nb;
            }
            if nf == 0 || (nf == 1 && kc == 0) {
                if it > 0 {
                    break;
                }
            } else {
                let mag = theta.iter().flatten().chain(b.iter()).fold(0.0f64, |a, v| a.max(v.abs()));
                if change <= tol * (1.0 + mag) {
                    break;
                }
            }
        }
        for (f, (codes, _)) in self.fac.iter().enumerate() {
            for i in 0..n {
                if wsum[f][codes[i]] <= 0.0 {
                    fit[i] = f64::NAN;
                }
            }
        }
        Solved { fit, b, iters }
    }

    /// X' (v) totals per factor level and per control over rows in `s`.
    pub fn xt(&self, s: &[bool], v: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = s.len();
        let rf = self
            .fac
            .iter()
            .map(|(c, g)| {
                let mut a = vec![0.0; *g];
                for i in 0..n {
                    if s[i] {
                        a[c[i]] += v[i];
                    }
                }
                a
            })
            .collect();
        let rc = self.ctl.iter().map(|c| (0..n).filter(|&i| s[i]).map(|i| c[i] * v[i]).sum()).collect();
        (rf, rc)
    }

    /// Weighted least-squares fit of z on the model over rows `s`.
    pub fn regress(&self, s: &[bool], w: &[f64], z: &[f64], tol: f64, maxit: usize) -> Solved {
        let wz: Vec<f64> = (0..z.len()).map(|i| if s[i] { w[i] * z[i] } else { 0.0 }).collect();
        let (rf, rc) = self.xt(s, &wz);
        self.solve(s, w, &rf, &rc, tol, maxit)
    }
}

fn xbc(ctl: &[Vec<f64>], b: &[f64], i: usize) -> f64 {
    ctl.iter().zip(b).map(|(c, bj)| c[i] * bj).sum()
}

fn col(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let v = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if v.var_type.is_string() {
        // string identifiers: map to codes
        let mut map: BTreeMap<String, f64> = BTreeMap::new();
        return Ok((0..ds.n_obs)
            .map(|i| {
                let s = v.str_at(i).to_string();
                let n = map.len() as f64;
                *map.entry(s).or_insert(n)
            })
            .collect());
    }
    Ok(v.data.iter().map(|x| x.num()).collect())
}

/// Run another command on the current data (used to obtain the same
/// degrees-of-freedom corrections as the high-dimensional FE regression).
pub(crate) fn run_line(ds: &mut DataSet, st: &mut Store, h: &dyn CommandHandler, line: &str) -> Result<String, String> {
    let c = ferrostat_parser::parser::parse_line(line).map_err(|e| rc_err(198, e))?;
    h.execute(ds, st, &c)
}

/// Add temporary double variables; returns their names.
pub(crate) fn add_temps(ds: &mut DataSet, cols: Vec<Vec<f64>>) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let mut k = 0;
    for c in cols {
        let name = loop {
            let n = format!("__did{k:04}");
            k += 1;
            if ds.get_var(&n).is_none() {
                break n;
            }
        };
        ds.add_variable(util::new_double(&name, c))?;
        names.push(name);
    }
    Ok(names)
}

pub(crate) fn drop_temps(ds: &mut DataSet, names: &[String]) {
    for n in names {
        let _ = ds.drop_variable(n);
    }
}

pub struct DidImputation;

impl CommandHandler for DidImputation {
    fn name(&self) -> &'static str {
        "did_imputation"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "did_imputation");
        }
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() != 4 {
            return Err(rc_err(198, "syntax: did_imputation Y i t Ei"));
        }
        let (yv, iv, tv, ev) = (&vars[0], &vars[1], &vars[2], &vars[3]);
        for o in ["hetby", "project", "unitcontrols", "timecontrols", "loadweights"] {
            if cmd.has_opt(o, 5) {
                return Err(rc_err(198, format!("option {o}() is not available")));
            }
        }
        let n = ds.n_obs;
        let fe_text = cmd.opt("fe", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| format!("{iv} {tv}"));
        let controls = match cmd.opt("controls", 1) {
            Some(c) => crate::varlist::expand(ds, c)?,
            None => Vec::new(),
        };
        let cluster = match cmd.opt("cluster", 4) {
            Some(c) => crate::varlist::resolve_name(ds, c.trim())?,
            None => iv.clone(),
        };
        let shift = util::opt_num(cmd, "shift", 5)?.unwrap_or(0.0);
        let pretrends = util::opt_num(cmd, "pretrends", 3)?.unwrap_or(0.0) as usize;
        let minn = util::opt_num(cmd, "minn", 4)?.unwrap_or(30.0);
        let sum = cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("sum"));
        let autosample = cmd.has_opt("autosample", 5);
        let leaveout = cmd.has_opt("leaveout", 8);
        let nose = cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("nose"));
        let tol = util::opt_num(cmd, "tol", 3)?.unwrap_or(1e-6).min(1e-6) * 1e-6;
        let maxit = (util::opt_num(cmd, "maxit", 5)?.unwrap_or(100.0) as usize).max(100) * 100;
        let alpha = util::opt_num(cmd, "alpha", 5)?.unwrap_or(0.05);
        let level = 100.0 * (1.0 - alpha);
        if autosample && sum {
            return Err(rc_err(184, "Autosample cannot be combined with sum. Please specify the sample explicitly"));
        }
        // sample
        let mut touse = util::touse(ds, st, cmd)?;
        let mut need = vec![yv.clone(), tv.clone(), iv.clone(), cluster.clone()];
        need.extend(controls.iter().cloned());
        let fe_terms: Vec<Vec<String>> = if fe_text == "." {
            Vec::new()
        } else {
            fe_text.split_whitespace().map(|t| t.split('#').map(|v| crate::varlist::resolve_name(ds, v.trim_start_matches("i."))).collect::<Result<Vec<_>, _>>()).collect::<Result<_, _>>()?
        };
        for t in &fe_terms {
            need.extend(t.iter().cloned());
        }
        need.sort();
        need.dedup();
        for v in &need {
            let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
            for i in 0..n {
                if touse[i] && !var.var_type.is_string() && is_missing(var.data[i].num()) {
                    touse[i] = false;
                }
            }
        }
        let wts = util::weights(ds, st, cmd, &mut touse, "afi")?;
        let mut wei = wts.as_ref().map_or(vec![1.0; n], |w| w.w.clone());
        if wts.is_some() && !sum {
            let (s, c) = (0..n).filter(|&i| touse[i]).fold((0.0, 0.0), |(s, c), i| (s + wei[i], c + 1.0));
            for v in wei.iter_mut() {
                *v *= c / s;
            }
        }
        let y = col(ds, yv)?;
        let t = col(ds, tv)?;
        let ei = col(ds, ev)?;
        let delta = match util::opt_num(cmd, "delta", 5)? {
            Some(d) => d,
            None => match &st.tsset {
                Some((_, tvar)) if tvar == tv => st.settings.get("tsdelta").and_then(|d| d.parse().ok()).unwrap_or(1.0),
                _ => 1.0,
            },
        };
        if delta <= 0.0 {
            return Err(rc_err(198, "A problem has occured with determining delta. Please specify it explicitly."));
        }
        let kk: Vec<f64> = (0..n).map(|i| if is_missing(ei[i]) { SYSMISS } else { (t[i] - ei[i] + shift) / delta }).collect();
        if (0..n).any(|i| touse[i] && !is_missing(kk[i]) && kk[i].fract().abs() > 1e-9) {
            return Err(rc_err(198, "There are non-integer values of the number of periods since treatment. Please check the time dimension of your data."));
        }
        let dtr: Vec<bool> = (0..n).map(|i| !is_missing(kk[i]) && kk[i] >= 0.0).collect();
        if !(0..n).any(|i| touse[i] && !dtr[i]) {
            return Err(rc_err(459, format!("There are no untreated observations, i.e. those with {tv}<{ev} or mi({ev}).\nPlease double-check the period & event time variables.")));
        }
        // estimand weights on treated observations
        let mut wtr: Vec<Vec<f64>> = Vec::new();
        let mut wnames: Vec<String> = Vec::new();
        match cmd.opt("wtr", 3) {
            Some(list) => {
                let wv = crate::varlist::expand(ds, list)?;
                for v in &wv {
                    wtr.push(col(ds, v)?.into_iter().map(|x| if is_missing(x) { 0.0 } else { x }).collect());
                    wnames.push(if wv.len() == 1 { "tau".into() } else { format!("tau_{v}") });
                }
            }
            None => {
                wtr.push(vec![1.0; n]);
                wnames.push("tau".into());
            }
        }
        let mut horizons: Option<Vec<f64>> = match cmd.opt("horizons", 1) {
            Some(h) => Some(crate::margins::numlist(h)?),
            None => None,
        };
        if cmd.has_opt("allhorizons", 4) {
            let mut hs: Vec<f64> = (0..n).filter(|&i| touse[i] && dtr[i] && wtr[0][i] != 0.0).map(|i| kk[i]).collect();
            hs.sort_by(|a, b| a.total_cmp(b));
            hs.dedup();
            horizons = Some(hs);
        }
        if let Some(hs) = &horizons {
            if wtr.len() > 1 {
                return Err(rc_err(184, "Options horizons and allhorizons cannot be combined with multiple wtr variables"));
            }
            let base = wtr.remove(0);
            wnames.clear();
            let mut base = base;
            if cmd.has_opt("hbalance", 4) {
                let mut cnt: BTreeMap<i64, usize> = BTreeMap::new();
                let idc = col(ds, iv)?;
                for i in 0..n {
                    if touse[i] && dtr[i] && hs.contains(&kk[i]) {
                        *cnt.entry((idc[i] * 1e6).round() as i64).or_default() += 1;
                    }
                }
                for i in 0..n {
                    if !(hs.contains(&kk[i]) && cnt.get(&((idc[i] * 1e6).round() as i64)).copied().unwrap_or(0) == hs.len()) {
                        base[i] = 0.0;
                    }
                }
            }
            for h in hs {
                wtr.push((0..n).map(|i| if kk[i] == *h { base[i] } else { 0.0 }).collect());
                wnames.push(format!("tau{}", crate::format::fmt_g(*h, 12).trim()));
            }
        }
        let normalize = |wtr: &mut Vec<f64>, touse: &[bool]| {
            let s: f64 = (0..n).filter(|&i| touse[i] && dtr[i]).map(|i| wtr[i] * wei[i]).sum();
            if s != 0.0 {
                for v in wtr.iter_mut() {
                    *v /= s;
                }
            }
        };
        if !sum {
            for wv in wtr.iter_mut() {
                if (0..n).any(|i| touse[i] && dtr[i] && wv[i] < 0.0) {
                    return Err(rc_err(9, "Negative wtr weights are only allowed if the sum option is specified"));
                }
                normalize(wv, &touse);
            }
        }
        // model of Y(0)
        let fe_cols: Vec<Vec<Vec<f64>>> = fe_terms.iter().map(|t| t.iter().map(|v| col(ds, v)).collect::<Result<Vec<_>, _>>()).collect::<Result<_, _>>()?;
        let ctl_cols: Vec<Vec<f64>> = controls.iter().map(|v| col(ds, v)).collect::<Result<_, _>>()?;
        let mut model = FeModel { fac: Vec::new(), ctl: ctl_cols };
        if fe_terms.is_empty() {
            model.fac.push((vec![0; n], 1));
        }
        for c in &fe_cols {
            let refs: Vec<&[f64]> = c.iter().map(|v| v.as_slice()).collect();
            model.fac.push(codes_of(&refs, n));
        }
        let s0: Vec<bool> = (0..n).map(|i| touse[i] && !dtr[i]).collect();
        // controls explained by the fixed effects on the untreated sample
        let mut controls = controls;
        let mut out: Vec<String> = Vec::new();
        if !controls.is_empty() {
            let only_fe = FeModel { fac: model.fac.clone(), ctl: Vec::new() };
            let mut keep = Vec::new();
            for (j, c) in model.ctl.iter().enumerate() {
                let f = only_fe.regress(&s0, &wei, c, tol, maxit);
                let m = (0..n).filter(|&i| s0[i]).map(|i| c[i]).sum::<f64>() / (s0.iter().filter(|v| **v).count().max(1) as f64);
                let ss: f64 = (0..n).filter(|&i| s0[i]).map(|i| (c[i] - f.fit[i]).powi(2)).sum();
                let ss0: f64 = (0..n).filter(|&i| s0[i]).map(|i| (c[i] - m).powi(2)).sum();
                if ss <= 1e-9 * ss0.max(1e-300) {
                    out.push(format!("note: {} omitted because of collinearity with the fixed effects", controls[j]));
                } else {
                    keep.push(j);
                }
            }
            model.ctl = keep.iter().map(|&j| model.ctl[j].clone()).collect();
            controls = keep.iter().map(|&j| controls[j].clone()).collect();
        }
        let step1 = model.regress(&s0, &wei, &y, tol, maxit);
        let effect: Vec<f64> = (0..n).map(|i| if touse[i] && dtr[i] { y[i] - step1.fit[i] } else { f64::NAN }).collect();
        // imputation failures
        let mut need_imp = vec![false; n];
        for wv in &wtr {
            for i in 0..n {
                if touse[i] && dtr[i] && wv[i] != 0.0 {
                    need_imp[i] = true;
                }
            }
        }
        for i in 0..n {
            touse[i] = touse[i] && (!dtr[i] || need_imp[i]);
        }
        let cannot: Vec<bool> = (0..n).map(|i| need_imp[i] && touse[i] && effect[i].is_nan()).collect();
        let ncannot = cannot.iter().filter(|c| **c).count();
        let mut autosample_drop = Vec::new();
        let mut autosample_trim = Vec::new();
        if ncannot > 0 {
            if ds.get_var("cannot_impute").is_some() {
                ds.drop_variable("cannot_impute")?;
            }
            let mut v = ferrostat_core::Variable::new("cannot_impute", ferrostat_core::VarType::Byte);
            v.data = (0..n).map(|i| ferrostat_core::Value::Byte(cannot[i] as i8)).collect();
            ds.add_variable(v)?;
            st.data_changed = true;
            if !autosample {
                return Err(rc_err(198, format!("Could not impute FE for {ncannot} observations. Those are saved in the cannot_impute variable. Use the autosample option if you would like those observations to be dropped from the sample automatically.")));
            }
            for (j, wv) in wtr.iter_mut().enumerate() {
                let dropped = (0..n).filter(|&i| cannot[i] && wv[i] != 0.0).count();
                if dropped > 0 {
                    for i in 0..n {
                        if cannot[i] {
                            wv[i] = 0.0;
                        }
                    }
                    let remaining = (0..n).any(|i| touse[i] && dtr[i] && !cannot[i] && wv[i] != 0.0);
                    if remaining {
                        let mask: Vec<bool> = (0..n).map(|i| touse[i] && !cannot[i]).collect();
                        normalize(wv, &mask);
                        autosample_trim.push(wnames[j].clone());
                    } else {
                        autosample_drop.push(wnames[j].clone());
                    }
                }
            }
            for i in 0..n {
                if cannot[i] {
                    touse[i] = false;
                }
            }
            if !autosample_drop.is_empty() {
                out.push(format!("Warning: suppressing the following coefficients because FE could not be imputed for any units: {}.", autosample_drop.join(" ")));
            }
            if !autosample_trim.is_empty() {
                out.push(format!("Warning: part of the sample was dropped for the following coefficients because FE could not be imputed: {}.", autosample_trim.join(" ")));
            }
        }
        // effective sample size
        let mut droplist = Vec::new();
        for (j, wv) in wtr.iter_mut().enumerate() {
            let s: f64 = (0..n).filter(|&i| touse[i] && dtr[i]).map(|i| wv[i].abs() * wei[i]).sum();
            if s != 0.0 {
                let hhi: f64 = (0..n).filter(|&i| touse[i] && dtr[i]).map(|i| (wv[i] * wei[i] / s).powi(2)).sum();
                if hhi > 1.0 / minn {
                    droplist.push(wnames[j].clone());
                    wv.iter_mut().for_each(|v| *v = 0.0);
                }
            } else if !autosample_drop.contains(&wnames[j]) {
                droplist.push(wnames[j].clone());
            }
        }
        if !droplist.is_empty() {
            out.push(format!("WARNING: suppressing the following coefficients from estimation because of insufficient effective sample size: {}. To report them nevertheless, set the minn option to a smaller number or 0, but keep in mind that the estimates may be unreliable and their SE may be downward biased.", droplist.join(" ")));
        }
        let ntau = wtr.len();
        let mut b: Vec<f64> = Vec::new();
        let mut nt: Vec<f64> = Vec::new();
        for wv in &wtr {
            b.push((0..n).filter(|&i| touse[i] && dtr[i]).map(|i| effect[i] * wv[i] * wei[i]).sum());
            nt.push((0..n).filter(|&i| touse[i] && dtr[i] && wv[i] != 0.0).count() as f64);
        }
        let s0: Vec<bool> = (0..n).map(|i| touse[i] && !dtr[i]).collect();
        let nc = s0.iter().filter(|v| **v).count() as f64;
        let nall = touse.iter().filter(|v| **v).count() as f64;
        let clv = col(ds, &cluster)?;
        let (clcode, ncl) = codes_of(&[clv.as_slice()], n);
        let resid0: Vec<f64> = (0..n).map(|i| if s0[i] { y[i] - step1.fit[i] } else { 0.0 }).collect();
        let mut eps: Vec<Vec<f64>> = Vec::new();
        let mut niter = 0usize;
        // pre-trend coefficients and their linear weights
        let mut pre_b: Vec<f64> = Vec::new();
        let mut pre_df = SYSMISS;
        if pretrends > 0 {
            let dummies: Vec<Vec<f64>> = (1..=pretrends).map(|h| (0..n).map(|i| if kk[i] == -(h as f64) { 1.0 } else { 0.0 }).collect()).collect();
            // degrees of freedom as in the high-dimensional FE regression
            let mut tmpcols = dummies.clone();
            tmpcols.push(touse.iter().zip(&s0).map(|(_, s)| if *s { 1.0 } else { 0.0 }).collect());
            let names = add_temps(ds, tmpcols)?;
            let fe_abs = if fe_text == "." { "noabsorb".to_string() } else { format!("absorb({})", fe_text) };
            let line = format!("reghdfe {} {} {} if {}==1, {} vce(cluster {})", yv, controls.join(" "), names[..pretrends].join(" "), names[pretrends], fe_abs, cluster);
            let saved_e = st.e.clone();
            let r = run_line(ds, st, &crate::reghdfe::Reghdfe, &line);
            let edf = (st.e.get("N").unwrap_or(0.0), st.e.get("df_m").unwrap_or(0.0), st.e.get("df_a").unwrap_or(0.0), st.e.get("N_clust").unwrap_or(0.0), st.e.get("df_r").unwrap_or(SYSMISS));
            st.e = saved_e;
            drop_temps(ds, &names);
            r?;
            pre_df = edf.4;
            let dof_adj = (edf.0 - 1.0) / (edf.0 - edf.1 - edf.2) * (edf.3 / (edf.3 - 1.0));
            // FWL: residualise the pre dummies on the model
            let ptil: Vec<Vec<f64>> = dummies.iter().map(|dv| {
                let f = model.regress(&s0, &wei, dv, tol, maxit);
                (0..n).map(|i| if s0[i] { dv[i] - f.fit[i] } else { 0.0 }).collect()
            }).collect();
            let ytil: Vec<f64> = resid0.clone();
            let kp = pretrends;
            let mut a = Matrix::zeros(kp, kp);
            for p in 0..kp {
                for q in 0..kp {
                    a[(p, q)] = (0..n).map(|i| wei[i] * ptil[p][i] * ptil[q][i]).sum();
                }
            }
            let (ai, _) = a.sym_inverse(1e-10);
            let pty: Vec<f64> = (0..kp).map(|p| (0..n).map(|i| wei[i] * ptil[p][i] * ytil[i]).sum()).collect();
            pre_b = (0..kp).map(|p| (0..kp).map(|q| ai[(p, q)] * pty[q]).sum()).collect();
            let presid: Vec<f64> = (0..n).map(|i| if s0[i] { ytil[i] - (0..kp).map(|p| pre_b[p] * ptil[p][i]).sum::<f64>() } else { 0.0 }).collect();
            if !nose {
                for h in 0..kp {
                    let mut e = vec![0.0; ncl];
                    for i in 0..n {
                        if s0[i] {
                            let om: f64 = (0..kp).map(|q| ai[(h, q)] * ptil[q][i]).sum::<f64>() * wei[i];
                            e[clcode[i]] += om * presid[i] * dof_adj.sqrt();
                        }
                    }
                    eps.push(e);
                }
            }
        }
        let mut tau_eps: Vec<Vec<f64>> = Vec::new();
        if !nose {
            let aeb: Vec<String> = match cmd.opt("avgeffectsby", 4) {
                Some(a) => crate::varlist::expand(ds, a)?,
                None => vec![ev.clone(), tv.clone()],
            };
            let acols: Vec<Vec<f64>> = aeb.iter().map(|v| col(ds, v)).collect::<Result<_, _>>()?;
            let arefs: Vec<&[f64]> = acols.iter().map(|v| v.as_slice()).collect();
            let (gcode, ng) = codes_of(&arefs, n);
            let mut cg: Vec<&[f64]> = arefs.clone();
            cg.push(clv.as_slice());
            let (cgcode, ncg) = codes_of(&cg, n);
            for (j, wv) in wtr.iter().enumerate() {
                if b[j] == 0.0 && wv.iter().all(|v| *v == 0.0) {
                    tau_eps.push(vec![0.0; ncl]);
                    continue;
                }
                // weights on untreated observations
                let tr: Vec<bool> = (0..n).map(|i| touse[i] && dtr[i]).collect();
                let v1: Vec<f64> = (0..n).map(|i| -wei[i] * wv[i]).collect();
                let (rf, rc) = model.xt(&tr, &v1);
                let sol = model.solve(&s0, &wei, &rf, &rc, tol, maxit);
                niter = niter.max(sol.iters);
                // treated residuals net of smart averages
                let mut cw = vec![0.0; ncg];
                for i in 0..n {
                    if tr[i] {
                        cw[cgcode[i]] += wei[i] * wv[i];
                    }
                }
                let mut den = vec![0.0; ng];
                for i in 0..n {
                    if tr[i] {
                        den[gcode[i]] += cw[cgcode[i]] * wei[i] * wv[i];
                    }
                }
                let mut avg = vec![0.0; ng];
                for i in 0..n {
                    if tr[i] && den[gcode[i]] != 0.0 {
                        avg[gcode[i]] += effect[i] * cw[cgcode[i]] * wei[i] * wv[i] / den[gcode[i]];
                    }
                }
                let mut e = vec![0.0; ncl];
                for i in 0..n {
                    if s0[i] {
                        e[clcode[i]] += wei[i] * sol.fit[i] * resid0[i];
                    } else if tr[i] {
                        let mut r = effect[i] - avg[gcode[i]];
                        if leaveout {
                            let c2 = cw[cgcode[i]].powi(2);
                            let d = den[gcode[i]];
                            if d > 0.0 && c2 / d > 0.99999 {
                                return Err(rc_err(498, format!("Cannot compute leave-out standard errors because of observations for coefficient \"{}\"\nThis most likely happened because there are cohorts with only one unit or cluster (and the default value for avgeffectsby  is used).", wnames[j])));
                            }
                            r *= d / (d - c2);
                        }
                        e[clcode[i]] += wei[i] * wv[i] * r;
                    }
                }
                tau_eps.push(e);
            }
        }
        // controls (coefficients from step 1)
        let kc = controls.len();
        let mut names: Vec<String> = wnames.clone();
        names.extend((1..=pre_b.len()).map(|h| format!("pre{h}")));
        names.extend(controls.iter().cloned());
        let mut ball = b.clone();
        ball.extend(pre_b.iter().copied());
        ball.extend(step1.b.iter().copied());
        let k = names.len();
        let mut v = Matrix::zeros(k, k);
        if !nose {
            let mut all_eps: Vec<Vec<f64>> = tau_eps;
            all_eps.extend(eps);
            if kc > 0 {
                // FWL weights for each control, residualised on the FE
                let ctil: Vec<Vec<f64>> = model.ctl.iter().map(|c| {
                    let only_fe = FeModel { fac: model.fac.clone(), ctl: Vec::new() };
                    let f = only_fe.regress(&s0, &wei, c, tol, maxit);
                    (0..n).map(|i| if s0[i] { c[i] - f.fit[i] } else { 0.0 }).collect()
                }).collect();
                let mut a = Matrix::zeros(kc, kc);
                for p in 0..kc {
                    for q in 0..kc {
                        a[(p, q)] = (0..n).map(|i| wei[i] * ctil[p][i] * ctil[q][i]).sum();
                    }
                }
                let (ai, _) = a.sym_inverse(1e-10);
                let ncl_f = ncl as f64;
                let nobs0 = nc;
                let dfa: f64 = model.fac.iter().map(|(c, g)| {
                    let mut seen = vec![false; *g];
                    for i in 0..n {
                        if s0[i] {
                            seen[c[i]] = true;
                        }
                    }
                    seen.iter().filter(|s| **s).count() as f64
                }).sum::<f64>() - (model.fac.len() as f64 - 1.0).max(0.0);
                let dof_adj = (nobs0 - 1.0) / (nobs0 - kc as f64 - dfa) * (ncl_f / (ncl_f - 1.0));
                for h in 0..kc {
                    let mut e = vec![0.0; ncl];
                    for i in 0..n {
                        if s0[i] {
                            let om: f64 = (0..kc).map(|q| ai[(h, q)] * ctil[q][i]).sum::<f64>() * wei[i];
                            e[clcode[i]] += om * resid0[i] * dof_adj.sqrt();
                        }
                    }
                    all_eps.push(e);
                }
            }
            for a in 0..k {
                for c in 0..=a {
                    let s: f64 = (0..ncl).map(|g| all_eps[a][g] * all_eps[c][g]).sum();
                    v[(a, c)] = s;
                    v[(c, a)] = s;
                }
            }
        }
        // pre-trend joint test
        let mut pre_test = None;
        if pretrends > 0 && !nose {
            let kp = pre_b.len();
            let mut vp = Matrix::zeros(kp, kp);
            for p in 0..kp {
                for q in 0..kp {
                    vp[(p, q)] = v[(ntau + p, ntau + q)];
                }
            }
            let (vi, dr) = vp.sym_inverse(1e-12);
            let q = (kp - dr.len()) as f64;
            let w: f64 = (0..kp).map(|p| (0..kp).map(|r| pre_b[p] * vi[(p, r)] * pre_b[r]).sum::<f64>()).sum();
            let f = w / q;
            pre_test = Some((f, crate::dist::ftail(q, pre_df, f), pre_df));
        }
        // saveestimates
        if let Some(sv) = cmd.opt("saveestimates", 4) {
            let name = sv.trim().to_string();
            if ds.get_var(&name).is_some() {
                ds.drop_variable(&name)?;
            }
            ds.add_variable(util::new_double(&name, (0..n).map(|i| if touse[i] && dtr[i] && !effect[i].is_nan() { effect[i] } else { SYSMISS }).collect()))?;
            st.data_changed = true;
        }
        // output
        out.push(String::new());
        out.extend(crate::est::header("", &[("Number of obs".to_string(), util::commas(nall as usize))]));
        let omitted: Vec<bool> = (0..k).map(|j| j < ntau && (droplist.contains(&names[j]) || autosample_drop.contains(&names[j]))).collect();
        let table = crate::est::coef_table(&crate::est::TableSpec { depvar: yv, names: &names, b: &ball, v: &v, omitted: &omitted, is_base: &vec![false; k], df: None, level, eform: None, coef_title: "Coefficient" });
        out.extend(table);
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("N", nall);
        e.set("Nc", nc);
        e.set_str("cmd", "did_imputation");
        e.set_str("cmdline", cmd.raw.trim().to_string());
        e.set_str("depvar", yv.clone());
        e.set_str("properties", "b V");
        e.set_mat("b", Matrix::row_vector(&ball).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names.clone()));
        e.set_mat("Nt", Matrix::row_vector(&nt).with_names(vec!["r1".into()], wnames.clone()));
        e.set_str("droplist", droplist.join(" "));
        e.set_str("autosample_drop", autosample_drop.join(" "));
        e.set_str("autosample_trim", autosample_trim.join(" "));
        e.set_str("clustvar", cluster.clone());
        e.set("N_clust", ncl as f64);
        if !nose {
            e.set("Niter", niter as f64);
        }
        if let Some((f, p, df)) = pre_test {
            e.set("pre_F", f);
            e.set("pre_p", p);
            e.set("pre_df", df);
        }
        e.sample = Some(touse);
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

/// `event_plot`: plots post-treatment effects (`tau#` coefficients) and
/// pre-trend coefficients (`pre#`) of the current estimates, or the
/// matrices named as `bmatrix#vmatrix` (e.g. `e(b_iw)#e(V_iw)`).
pub struct EventPlot;

impl CommandHandler for EventPlot {
    fn name(&self) -> &'static str {
        "event_plot"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let spec = cmd.main.trim();
        let (bm, vm) = if spec.is_empty() || spec == "." {
            let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            (b, v)
        } else {
            let (bs, vs) = spec.split_once('#').ok_or_else(|| rc_err(198, "specify bmatrix#vmatrix"))?;
            let get = |s: &str| -> Result<Matrix, String> {
                let s = s.trim();
                if let Some(nm) = s.strip_prefix("e(").and_then(|x| x.strip_suffix(')')) {
                    st.e.matrices.get(nm).cloned().ok_or_else(|| rc_err(111, format!("matrix {s} not found")))
                } else {
                    st.matrices.get(s).cloned().ok_or_else(|| rc_err(111, format!("matrix {s} not found")))
                }
            };
            (get(bs)?, get(vs)?)
        };
        let pat = |opt: &str, default: &str| -> (String, String) {
            let p = cmd.opt(opt, 8).map(|s| s.trim().to_string()).unwrap_or_else(|| default.to_string());
            match p.split_once('#') {
                Some((a, b)) => (a.to_string(), b.to_string()),
                None => (p, String::new()),
            }
        };
        let (lag_a, lag_b) = pat("stub_lag", "tau#");
        let (lead_a, lead_b) = pat("stub_lead", "pre#");
        let level = util::level(cmd, st)?;
        let crit = crate::dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut pts = Vec::new();
        for (j, name) in bm.colnames.iter().enumerate() {
            let nm = name.rsplit(':').next().unwrap_or(name);
            let num = |a: &str, b: &str| -> Option<f64> { nm.strip_prefix(a).and_then(|r| r.strip_suffix(b)).and_then(|r| r.parse::<f64>().ok()) };
            let (x, post) = if let Some(h) = num(&lag_a, &lag_b) {
                (h, true)
            } else if let Some(h) = num(&lead_a, &lead_b) {
                (-h, false)
            } else {
                continue;
            };
            let b = bm.data[j];
            let se = if vm.rows == bm.cols { vm[(j, j)].max(0.0).sqrt() } else { vm.data.get(j).copied().unwrap_or(0.0).max(0.0).sqrt() };
            pts.push((x, b, b - crit * se, b + crit * se, post));
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        crate::did::plot_effects(st, cmd, &pts, "Periods since treatment", "", Vec::new())
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("did_imputation", Box::new(DidImputation)), ("event_plot", Box::new(EventPlot)), ("eventstudyinteract", Box::new(crate::esinteract::EventStudyInteract))]
}
