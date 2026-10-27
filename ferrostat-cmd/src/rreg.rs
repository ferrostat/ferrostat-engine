//! `rreg`: robust regression by iteratively reweighted least squares —
//! gross outliers (Cook's D > 1) dropped first, then Huber iterations,
//! then biweight iterations; standard errors from pseudovalues.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub struct Rreg;

/// Weighted least squares on the active columns: (b, (X'WX)^-1).
fn wls(cols: &[&[f64]], y: &[f64], w: &[f64]) -> (Vec<f64>, Matrix) {
    let k = cols.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for t in 0..y.len() {
        if w[t] == 0.0 {
            continue;
        }
        for a in 0..k {
            let xa = cols[a][t] * w[t];
            xty[a] += xa * y[t];
            for c in 0..=a {
                xtx[(a, c)] += xa * cols[c][t];
            }
        }
    }
    for a in 0..k {
        for c in 0..a {
            xtx[(c, a)] = xtx[(a, c)];
        }
    }
    let (inv, _) = xtx.sym_inverse(1e-13);
    let b = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    (b, inv)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return SYSMISS;
    }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Robust scale: median absolute deviation from the median residual / .6745.
fn scale(r: &[f64], keep: &[bool]) -> f64 {
    let mut v: Vec<f64> = r.iter().zip(keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect();
    let m = median(&mut v);
    let mut d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    median(&mut d) / 0.6745
}

impl CommandHandler for Rreg {
    fn name(&self) -> &'static str {
        "rreg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("rreg") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        if cmd.weight.is_some() {
            return Err(rc_err(101, "weights not allowed"));
        }
        let genwt = cmd.opt("genwt", 3).or_else(|| cmd.opt("generate", 3)).map(|s| s.trim().to_string());
        if let Some(g) = &genwt {
            crate::varlist::check_name(g)?;
            if ds.var_index.contains_key(g) {
                return Err(rc_err(110, format!("variable {g} already defined")));
            }
        }
        let tune = util::opt_num(cmd, "tune", 2)?.unwrap_or(7.0);
        if tune < 2.0 {
            return Err(rc_err(198, "tune() must be at least 2"));
        }
        let tol = util::opt_num(cmd, "tolerance", 3)?.unwrap_or(0.01);
        let nolog = cmd.has_opt("nolog", 5);
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra: vec![] })?;
        let act = d.active();
        let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let n = d.n();
        let k = act.len();
        let resid = |b: &[f64]| -> Vec<f64> { (0..n).map(|t| d.y[t] - cols.iter().zip(b).map(|(c, bb)| c[t] * bb).sum::<f64>()).collect() };
        // 1. OLS and Cook's distance: drop gross outliers
        let ones = vec![1.0; n];
        let (b_ols, inv) = wls(&cols, &d.y, &ones);
        let r = resid(&b_ols);
        let s2 = r.iter().map(|x| x * x).sum::<f64>() / (n - k) as f64;
        let mut w = vec![1.0; n];
        let mut out = Vec::new();
        let mut log = Vec::new();
        for t in 0..n {
            let x: Vec<f64> = cols.iter().map(|c| c[t]).collect();
            let h: f64 = (0..k).map(|a| (0..k).map(|c| x[a] * inv[(a, c)] * x[c]).sum::<f64>()).sum();
            let cd = r[t] * r[t] * h / (k as f64 * s2 * (1.0 - h).powi(2));
            if cd > 1.0 || h >= 1.0 - 1e-12 {
                w[t] = 0.0;
            }
        }
        let keep: Vec<bool> = w.iter().map(|&x| x > 0.0).collect();
        let n_used = keep.iter().filter(|&&x| x).count();
        if n_used <= k {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let ch = 1.345;
        let cb = 4.685 * tune / 7.0;
        // start from least squares without the gross outliers
        let mut b = wls(&cols, &d.y, &w).0;
        let mut it = 0;
        // 2. Huber, then 3. biweight iterations
        for biweight in [false, true] {
            loop {
                it += 1;
                let r = resid(&b);
                let s = scale(&r, &keep);
                let mut maxd: f64 = 0.0;
                let mut nw = vec![0.0; n];
                for t in 0..n {
                    if !keep[t] {
                        continue;
                    }
                    let u = if s > 0.0 { r[t] / s } else { 0.0 };
                    nw[t] = if biweight {
                        if u.abs() <= cb { (1.0 - (u / cb).powi(2)).powi(2) } else { 0.0 }
                    } else if u.abs() <= ch {
                        1.0
                    } else {
                        ch / u.abs()
                    };
                    maxd = maxd.max((nw[t] - w[t]).abs());
                }
                w = nw;
                b = wls(&cols, &d.y, &w).0;
                log.push(format!("{:>8} iteration {}:  maximum difference in weights = {}", if biweight { "Biweight" } else { "Huber" }, it, fmt_g(maxd, 9)));
                if maxd < tol || it > 1000 {
                    break;
                }
            }
        }
        if !nolog {
            out.push(String::new());
            out.extend(log);
        }
        // pseudovalue standard errors (biweight psi)
        let r = resid(&b);
        let s = scale(&r, &keep);
        let mut psi_d = 0.0;
        let mut m = 0.0;
        for t in 0..n {
            if !keep[t] {
                continue;
            }
            let u = r[t] / s;
            psi_d += if u.abs() <= cb { (1.0 - (u / cb).powi(2)) * (1.0 - 5.0 * (u / cb).powi(2)) } else { 0.0 };
            m += 1.0;
        }
        let lam = psi_d / m;
        // small-sample correction K = 1 + (p/n) (1 - lambda) / lambda
        let kcorr = 1.0 + (k as f64 / m) * (1.0 - lam) / lam;
        let mut pseudo = vec![0.0; n];
        for t in 0..n {
            let u = r[t] / s;
            let psi = if u.abs() <= cb { u * (1.0 - (u / cb).powi(2)).powi(2) } else { 0.0 };
            pseudo[t] = d.y[t] - r[t] + s * psi / lam;
        }
        let kw: Vec<f64> = keep.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
        let (bp, invp) = wls(&cols, &pseudo, &kw);
        let rp: Vec<f64> = (0..n).map(|t| pseudo[t] - cols.iter().zip(&bp).map(|(c, bb)| c[t] * bb).sum::<f64>()).collect();
        let df_r = (n_used - k) as f64;
        let s2p = rp.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| x * x).sum::<f64>() / df_r;
        let vact = invp.scale(s2p * kcorr * kcorr);
        let kt = d.k();
        let mut bf = vec![0.0; kt];
        for (a, &j) in act.iter().enumerate() {
            bf[j] = b[a];
        }
        let v = est::expand_v(&vact, &act, kt);
        let (wchi, df_m) = est::wald_all(&d, &bf, &v);
        let f = wchi / df_m;
        let pf = dist::ftail(df_m, df_r, f);
        let level = util::level(cmd, st)?;
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "Robust regression", "Number of obs", n_used));
        out.push(format!("{:<52}{:<15} = {:>10}", "", format!("F({}, {})", fmt_g(df_m, 9), fmt_g(df_r, 9)), format!("{f:.2}")));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > F", format!("{pf:.4}")));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &bf, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: Some(df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let mut dpost = d.clone();
        let mut mask = vec![false; ds.n_obs];
        for t in 0..n {
            if keep[t] {
                mask[d.rows[t]] = true;
            }
        }
        dpost.mask = mask;
        est::post(st, "rreg", &dpost, &bf, &v, &|e: &mut Results| {
            e.set("N", n_used as f64);
            e.set("df_m", df_m);
            e.set("df_r", df_r);
            e.set("F", f);
            e.set("p", pf);
            e.set_str("title", "Robust regression");
            e.set_str("predict", "regres_p");
            e.set_str("model", "ols");
            e.set_str("_output", text.clone());
            if let Some(g) = &genwt {
                e.set_str("genwt", g.clone());
            }
        });
        if let Some(g) = genwt {
            let mut vals = vec![SYSMISS; ds.n_obs];
            for t in 0..n {
                if keep[t] {
                    vals[d.rows[t]] = w[t];
                }
            }
            let mut var = Variable::new(&g, VarType::Float);
            var.data = vals.iter().map(|&x| util::typed_value(VarType::Float, x)).collect();
            var.label = "Robust Regression Weight".into();
            ds.add_variable(var)?;
            st.data_changed = true;
        }
        Ok(text)
    }
}
