//! Panel-data diagnostics: `xttest0` (Breusch-Pagan LM test for random
//! effects), `xttest3` (modified Wald test for groupwise heteroskedasticity
//! after fixed effects), `xtserial` (Wooldridge test for serial correlation)
//! and `xtcsd` (Pesaran CD test of cross-sectional independence).
//!
//! Written from the published methods: Breusch and Pagan (1980) with the
//! unbalanced-panel form of Baltagi and Li (1990); Greene (2000, p. 598) for
//! the modified Wald statistic; Wooldridge (2002, sec. 10.4.1) and Drukker
//! (2003) for the serial-correlation test; Pesaran (2004) for CD.

use std::collections::HashMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("xttest0", Box::new(Xttest0)), ("xttest3", Box::new(Xttest3)), ("xtserial", Box::new(Xtserial)), ("xtcsd", Box::new(Xtcsd))]
}

/// Residual pieces after `xtreg`: for every estimation-sample observation,
/// (panel key, row, y - xb). The constant is included in xb.
fn xtreg_resid(ds: &DataSet, st: &Store) -> Result<(Vec<(f64, usize, f64)>, String), String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("xtreg") {
        return Err(rc_err(301, "last estimates not found; this test requires xtreg"));
    }
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let ivar = st.e.macros.get("ivar").cloned().ok_or_else(|| rc_err(301, "panel variable not found"))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let y = util::col(ds, &dep)?;
    let p = util::col(ds, &ivar)?;
    let env = Env::with_store(ds, st);
    let mut out = Vec::new();
    for i in 0..ds.n_obs {
        if !sample.get(i).copied().unwrap_or(false) {
            continue;
        }
        let xb = crate::postest::xb_row(&env, &b, None, i)?;
        if is_missing(xb) || is_missing(y[i]) {
            continue;
        }
        out.push((p[i], i, y[i] - xb));
    }
    Ok((out, ivar))
}

/// Group residuals by panel key (keys in sorted order).
fn by_panel(r: &[(f64, usize, f64)]) -> Vec<Vec<(usize, f64)>> {
    let mut keys: Vec<f64> = r.iter().map(|x| x.0).collect();
    keys.sort_by(f64::total_cmp);
    keys.dedup();
    let mut g = vec![Vec::new(); keys.len()];
    for &(k, i, e) in r {
        let j = keys.binary_search_by(|x| x.total_cmp(&k)).unwrap();
        g[j].push((i, e));
    }
    g
}

pub struct Xttest0;

impl CommandHandler for Xttest0 {
    fn name(&self) -> &'static str {
        "xttest0"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, _cmd: &Command) -> Result<String, String> {
        if st.e.macros.get("model").map(String::as_str) != Some("re") || st.e.macros.get("cmd").map(String::as_str) != Some("xtreg") {
            return Err(rc_err(301, "last estimates not xtreg, re"));
        }
        let (r, ivar) = xtreg_resid(ds, st)?;
        let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
        let g = by_panel(&r);
        let n = r.len() as f64;
        let see: f64 = r.iter().map(|x| x.2 * x.2).sum();
        let ss: f64 = g.iter().map(|p| p.iter().map(|x| x.1).sum::<f64>().powi(2)).sum();
        let st2: f64 = g.iter().map(|p| (p.len() * p.len()) as f64).sum();
        let a = ss / see - 1.0;
        let lm = n * n / (2.0 * (st2 - n)) * a * a;
        let lm = if a > 0.0 { lm } else { 0.0 };
        let p = if lm > 0.0 { 0.5 * dist::chi2tail(1.0, lm) } else { 1.0 };
        let y = util::col(ds, &dep)?;
        let yv: Vec<f64> = r.iter().map(|x| y[x.1]).collect();
        let vy = util::variance(&yv);
        let se = st.e.get("sigma_e").unwrap_or(0.0);
        let su = st.e.get("sigma_u").unwrap_or(0.0);
        let row = |l: &str, v: f64| format!("{:>24} | {:>10}     {:>10}", l, fmt_g(v, 9), fmt_g(v.sqrt(), 9));
        let out = vec![
            String::new(),
            "Breusch and Pagan Lagrangian multiplier test for random effects".to_string(),
            String::new(),
            format!("        {dep}[{ivar},t] = Xb + u[{ivar}] + e[{ivar},t]"),
            String::new(),
            "        Estimated results:".into(),
            format!("{:>24} |       Var     SD = sqrt(Var)", ""),
            format!("{:>24}-+-----------------------------", "---------"),
            row(&crate::summarize::abbrev(&dep, 8), vy),
            row("e", se * se),
            row("u", su * su),
            String::new(),
            "        Test: Var(u) = 0".into(),
            format!("{:>41} = {:>9.2}", "chibar2(01)", lm),
            format!("{:>41} = {:>9.4}", "Prob > chibar2", p),
        ];
        let mut res = Results::new();
        res.set("lm", lm);
        res.set("df", 1.0);
        res.set("p", p);
        st.r = res;
        Ok(out.join("\n"))
    }
}

pub struct Xttest3;

impl CommandHandler for Xttest3 {
    fn name(&self) -> &'static str {
        "xttest3"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, _cmd: &Command) -> Result<String, String> {
        let (r, _) = xtreg_resid(ds, st)?;
        let model = st.e.macros.get("model").cloned().unwrap_or_default();
        if model != "fe" {
            return Err(rc_err(301, "last estimates not xtreg, fe"));
        }
        let g = by_panel(&r);
        // idiosyncratic residuals: remove the panel mean of y - xb
        let e: Vec<Vec<f64>> = g.iter().map(|p| {
            let m = p.iter().map(|x| x.1).sum::<f64>() / p.len() as f64;
            p.iter().map(|x| x.1 - m).collect()
        }).collect();
        let n: usize = e.iter().map(|v| v.len()).sum();
        let s2 = e.iter().flatten().map(|v| v * v).sum::<f64>() / n as f64;
        let mut w = 0.0;
        for v in &e {
            let t = v.len() as f64;
            let si = v.iter().map(|x| x * x).sum::<f64>() / t;
            let vi = if t > 1.0 { v.iter().map(|x| (x * x - si).powi(2)).sum::<f64>() / t / (t - 1.0) } else { 0.0 };
            if vi > 0.0 {
                w += (si - s2).powi(2) / vi;
            }
        }
        let df = g.len() as f64;
        let p = dist::chi2tail(df, w);
        let out = vec![
            String::new(),
            "Modified Wald test for groupwise heteroskedasticity".to_string(),
            "in fixed effect regression model".into(),
            String::new(),
            "H0: sigma(i)^2 = sigma^2 for all i".into(),
            String::new(),
            format!("{:<11}= {:>10.2}", format!("chi2 ({})", df), w),
            format!("{:<10}= {:>11.4}", "Prob>chi2", p),
        ];
        let mut res = Results::new();
        res.set("wald", w);
        res.set("df", df);
        res.set("p", p);
        st.r = res;
        Ok(out.join("\n"))
    }
}

/// OLS with clustered variance (small-sample adjusted).
fn ols_cluster(y: &[f64], x: &[Vec<f64>], cl: &[usize]) -> (Vec<f64>, Matrix, Vec<f64>, usize) {
    let k = x.len();
    let n = y.len();
    let mut a = Matrix::zeros(k, k);
    let mut r = vec![0.0; k];
    for t in 0..n {
        for i in 0..k {
            r[i] += x[i][t] * y[t];
            for j in 0..k {
                a[(i, j)] += x[i][t] * x[j][t];
            }
        }
    }
    let (ai, _) = a.sym_inverse(1e-14);
    let b: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * r[j]).sum()).collect();
    let e: Vec<f64> = (0..n).map(|t| y[t] - (0..k).map(|j| x[j][t] * b[j]).sum::<f64>()).collect();
    let ng = cl.iter().max().map_or(0, |m| m + 1);
    let mut s = vec![vec![0.0; k]; ng];
    for t in 0..n {
        for j in 0..k {
            s[cl[t]][j] += x[j][t] * e[t];
        }
    }
    let mut meat = Matrix::zeros(k, k);
    let mut used = 0;
    for sg in &s {
        if sg.iter().any(|v| *v != 0.0) {
            used += 1;
        }
        for i in 0..k {
            for j in 0..k {
                meat[(i, j)] += sg[i] * sg[j];
            }
        }
    }
    let g = cl.iter().collect::<std::collections::HashSet<_>>().len() as f64;
    let c = g / (g - 1.0) * (n as f64 - 1.0) / (n as f64 - k as f64);
    let v = ai.mul(&meat).unwrap().mul(&ai).unwrap().scale(c);
    let _ = used;
    (b, v, e, g as usize)
}

pub struct Xtserial;

impl CommandHandler for Xtserial {
    fn name(&self) -> &'static str {
        "xtserial"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (ivar, tvar) = match &st.tsset {
            Some((Some(p), t)) if !t.is_empty() => (p.clone(), t.clone()),
            _ => return Err(rc_err(459, "panel variable and time variable not set; use xtset panelvar timevar")),
        };
        let vars = crate::varlist::expand(ds, &crate::tsops::expand_text(cmd.main.trim()))?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let env = Env::with_store(ds, st);
        let series: Vec<Vec<f64>> = vars
            .iter()
            .map(|v| {
                let e = ferrostat_parser::parser::parse_expression(v).map_err(|e| rc_err(198, e.message))?;
                (0..ds.n_obs).map(|i| env.eval(&e, i)?.num()).collect::<Result<Vec<f64>, String>>()
            })
            .collect::<Result<_, _>>()?;
        let pc = util::col(ds, &ivar)?;
        let tc = util::col(ds, &tvar)?;
        let mut at: HashMap<(u64, i64), usize> = HashMap::new();
        for i in 0..ds.n_obs {
            if mask[i] && !is_missing(pc[i]) && !is_missing(tc[i]) && series.iter().all(|s| !is_missing(s[i])) {
                at.insert((pc[i].to_bits(), tc[i].round() as i64), i);
            }
        }
        // first differences
        let mut keys: Vec<f64> = at.keys().map(|k| f64::from_bits(k.0)).collect();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let mut rows: Vec<(usize, i64, usize)> = Vec::new(); // (panel idx, t, obs)
        for (&(p, t), &i) in &at {
            if at.contains_key(&(p, t - 1)) {
                let g = keys.binary_search_by(|x| x.total_cmp(&f64::from_bits(p))).unwrap();
                rows.push((g, t, i));
            }
        }
        rows.sort();
        let n = rows.len();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let d = |s: &Vec<f64>, g: usize, t: i64, i: usize| -> f64 {
            let j = at[&(keys[g].to_bits(), t - 1)];
            s[i] - s[j]
        };
        let y: Vec<f64> = rows.iter().map(|&(g, t, i)| d(&series[0], g, t, i)).collect();
        let x: Vec<Vec<f64>> = series[1..].iter().map(|s| rows.iter().map(|&(g, t, i)| d(s, g, t, i)).collect()).collect();
        let cl: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let (_, _, e, _) = ols_cluster(&y, &x, &cl);
        // regress e_t on e_{t-1}
        let idx: HashMap<(usize, i64), usize> = rows.iter().enumerate().map(|(k, r)| ((r.0, r.1), k)).collect();
        let mut y2 = Vec::new();
        let mut x2 = Vec::new();
        let mut cl2 = Vec::new();
        for (k, r) in rows.iter().enumerate() {
            if let Some(&j) = idx.get(&(r.0, r.1 - 1)) {
                y2.push(e[k]);
                x2.push(e[j]);
                cl2.push(r.0);
            }
        }
        // compact cluster ids
        let mut ids: Vec<usize> = cl2.clone();
        ids.sort();
        ids.dedup();
        let cl2: Vec<usize> = cl2.iter().map(|c| ids.binary_search(c).unwrap()).collect();
        let (b, v, _, g) = ols_cluster(&y2, &[x2], &cl2);
        let f = (b[0] + 0.5).powi(2) / v[(0, 0)];
        let df2 = g as f64 - 1.0;
        let p = dist::ftail(1.0, df2, f);
        let out = vec![
            String::new(),
            "Wooldridge test for autocorrelation in panel data".to_string(),
            "H0: no first-order autocorrelation".into(),
            format!("    F(  1, {:>7}) = {:>10.3}", df2, f),
            format!("           Prob > F = {:>11.4}", p),
        ];
        let mut res = Results::new();
        res.set("corr", b[0]);
        res.set("df_r", df2);
        res.set("df", 1.0);
        res.set("p", p);
        res.set("F", f);
        res.set("N", n as f64);
        st.r = res;
        Ok(out.join("\n"))
    }
}

pub struct Xtcsd;

impl CommandHandler for Xtcsd {
    fn name(&self) -> &'static str {
        "xtcsd"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !cmd.has_opt("pesaran", 3) {
            return Err(rc_err(198, "one of pesaran, friedman or frees must be specified; only pesaran is supported"));
        }
        let tvar = st.tsset.as_ref().map(|t| t.1.clone()).filter(|t| !t.is_empty()).ok_or_else(|| rc_err(459, "time variable not set"))?;
        let (r, _) = xtreg_resid(ds, st)?;
        let model = st.e.macros.get("model").cloned().unwrap_or_default();
        if model != "fe" && model != "re" {
            return Err(rc_err(198, "xtcsd requires xtreg, fe or xtreg, re"));
        }
        let se2 = st.e.get("sigma_e").unwrap_or(0.0).powi(2);
        let su2 = st.e.get("sigma_u").unwrap_or(0.0).powi(2);
        let tc = util::col(ds, &tvar)?;
        let g = by_panel(&r);
        // idiosyncratic residuals by time
        let res: Vec<HashMap<i64, f64>> = g
            .iter()
            .map(|p| {
                let t = p.len() as f64;
                let m = p.iter().map(|x| x.1).sum::<f64>() / t;
                let u = if model == "fe" { m } else { t * su2 / (t * su2 + se2) * m };
                p.iter().map(|&(i, e)| (tc[i].round() as i64, e - u)).collect()
            })
            .collect();
        let n = res.len();
        let (mut cd, mut abs_sum, mut npair) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            for j in i + 1..n {
                let common: Vec<(f64, f64)> = res[i].iter().filter_map(|(t, a)| res[j].get(t).map(|b| (*a, *b))).collect();
                let tij = common.len();
                if tij < 3 {
                    continue;
                }
                let ma = common.iter().map(|c| c.0).sum::<f64>() / tij as f64;
                let mb = common.iter().map(|c| c.1).sum::<f64>() / tij as f64;
                let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
                for (a, b) in &common {
                    sab += (a - ma) * (b - mb);
                    saa += (a - ma).powi(2);
                    sbb += (b - mb).powi(2);
                }
                let rho = sab / (saa * sbb).sqrt();
                if !rho.is_finite() {
                    continue;
                }
                cd += (tij as f64).sqrt() * rho;
                abs_sum += rho.abs();
                npair += 1.0;
            }
        }
        let nn = n as f64;
        let cd = (2.0 / (nn * (nn - 1.0))).sqrt() * cd;
        let p = 2.0 * dist::normal(-cd.abs());
        let absm = abs_sum / npair.max(1.0);
        let out = vec![
            String::new(),
            format!("Pesaran's test of cross sectional independence = {:>9.3}, Pr = {:.4}", cd, p),
            String::new(),
            format!("Average absolute value of the off-diagonal elements = {:>9.3}", absm),
        ];
        let mut rr = Results::new();
        rr.set("pesaran", cd);
        rr.set("p", p);
        rr.set("abs", absm);
        st.r = rr;
        Ok(out.join("\n"))
    }
}
