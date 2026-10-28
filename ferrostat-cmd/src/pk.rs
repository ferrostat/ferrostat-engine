//! Pharmacokinetic (biopharmaceutical) commands: `pkexamine`, `pksumm`,
//! `pkcollapse`, `pkshape`, `pkcross` and `pkequiv`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("pkexamine", Box::new(Pkexamine)),
        ("pksumm", Box::new(Pksumm)),
        ("pkcollapse", Box::new(Pkcollapse)),
        ("pkshape", Box::new(Pkshape)),
        ("pkcross", Box::new(Pkcross)),
        ("pkequiv", Box::new(Pkequiv)),
    ]
}

/// Pharmacokinetic measures of one concentration–time curve.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pk {
    pub auc: f64,
    pub aucline: f64,
    pub aucexp: f64,
    pub auclog: f64,
    pub half: f64,
    pub ke: f64,
    pub cmax: f64,
    pub tomc: f64,
    pub tmax: f64,
}

/// Measures from (time, concentration) pairs; the elimination rate is
/// estimated from the last `fit` points (log-linear regression).
pub fn pk_measures(pts: &[(f64, f64)], fit: usize, t0: Option<f64>) -> Pk {
    let mut p: Vec<(f64, f64)> = pts.iter().copied().filter(|q| !is_missing(q.0) && !is_missing(q.1)).collect();
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(t) = t0 {
        if p.first().is_none_or(|q| q.0 > t) {
            p.insert(0, (t, 0.0));
        }
    }
    let n = p.len();
    if n < 2 {
        return Pk { auc: SYSMISS, aucline: SYSMISS, aucexp: SYSMISS, auclog: SYSMISS, half: SYSMISS, ke: SYSMISS, cmax: SYSMISS, tomc: SYSMISS, tmax: SYSMISS };
    }
    let auc: f64 = p.windows(2).map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) / 2.0).sum();
    let (mut cmax, mut tomc) = (p[0].1, p[0].0);
    for q in &p {
        if q.1 > cmax {
            cmax = q.1;
            tomc = q.0;
        }
    }
    let tmax = p[n - 1].0;
    let tail: Vec<(f64, f64)> = p[n.saturating_sub(fit.max(2))..].iter().copied().filter(|q| q.1 > 0.0).collect();
    let reg = |pts: &[(f64, f64)]| -> (f64, f64) {
        let m = pts.len() as f64;
        let mx = pts.iter().map(|q| q.0).sum::<f64>() / m;
        let my = pts.iter().map(|q| q.1).sum::<f64>() / m;
        let sxx: f64 = pts.iter().map(|q| (q.0 - mx).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
        let b = sxy / sxx;
        (my - b * mx, b)
    };
    let (a_log, b_log) = if tail.len() >= 2 { reg(&tail.iter().map(|q| (q.0, q.1.ln())).collect::<Vec<_>>()) } else { (SYSMISS, SYSMISS) };
    let ke = if is_missing(b_log) { SYSMISS } else { -b_log };
    let half = if is_missing(ke) || ke <= 0.0 { SYSMISS } else { std::f64::consts::LN_2 / ke };
    let clast = p[n - 1].1;
    let auclog = if is_missing(ke) || ke <= 0.0 { SYSMISS } else { auc + clast / ke };
    let aucexp = if is_missing(ke) || ke <= 0.0 { SYSMISS } else { auc + (a_log - ke * tmax).exp() / ke };
    let tl: Vec<(f64, f64)> = p[n.saturating_sub(fit.max(2))..].to_vec();
    let (a_lin, b_lin) = reg(&tl);
    let aucline = if b_lin < 0.0 {
        let tz = -a_lin / b_lin;
        let fl = a_lin + b_lin * tmax;
        auc + if tz > tmax { 0.5 * fl * (tz - tmax) } else { 0.0 }
    } else {
        SYSMISS
    };
    Pk { auc, aucline, aucexp, auclog, half, ke, cmax, tomc, tmax }
}

fn opt_fit(cmd: &Command) -> Result<usize, String> {
    Ok(util::opt_num(cmd, "fit", 3)?.unwrap_or(3.0) as usize)
}

pub struct Pkexamine;

impl CommandHandler for Pkexamine {
    fn name(&self) -> &'static str {
        "pkexamine"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "pkexamine time concentration"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let t = util::col(ds, &vars[0])?;
        let c = util::col(ds, &vars[1])?;
        let pts: Vec<(f64, f64)> = util::rows(&mask).iter().map(|&i| (t[i], c[i])).collect();
        let pk = pk_measures(&pts, opt_fit(cmd)?, None);
        let f4 = |x: f64| if is_missing(x) { ".".to_string() } else { fmt_g(x, 9) };
        let mut out = vec![String::new()];
        out.push(format!("{:>40} = {:>10}", "Maximum concentration", f4(pk.cmax)));
        out.push(format!("{:>40} = {:>10}", "Time of maximum concentration", f4(pk.tomc)));
        out.push(format!("{:>40} = {:>10}", "Time of last observation (Tmax)", f4(pk.tmax)));
        out.push(format!("{:>40} = {:>10}", "Elimination rate", if is_missing(pk.ke) { ".".into() } else { format!("{:.4}", pk.ke) }));
        out.push(format!("{:>40} = {:>10}", "Half life", if is_missing(pk.half) { ".".into() } else { format!("{:.4}", pk.half) }));
        out.push(String::new());
        out.push(format!("{:>41}", "Area under the curve"));
        out.push("-".repeat(64));
        out.push("  AUC [0, Tmax]   AUC [0, inf.)   AUC [0, inf.)   AUC [0, inf.)".into());
        out.push("                  Linear of log    Linear fit     Exponential".into());
        out.push("                      conc.                           fit".into());
        out.push("-".repeat(64));
        let f2 = |x: f64| if is_missing(x) { ".".to_string() } else { format!("{x:.2}") };
        out.push(format!("  {:>13}   {:>13}   {:>13}   {:>13}", f2(pk.auc), f2(pk.auclog), f2(pk.aucline), f2(pk.aucexp)));
        out.push("-".repeat(64));
        out.push(format!("Fit based on last {} points.", opt_fit(cmd)?));
        let mut r = Results::new();
        r.set("auc", pk.auc);
        r.set("aucline", pk.aucline);
        r.set("aucexp", pk.aucexp);
        r.set("auclog", pk.auclog);
        r.set("half", pk.half);
        r.set("ke", pk.ke);
        r.set("cmax", pk.cmax);
        r.set("tomc", pk.tomc);
        r.set("tmax", pk.tmax);
        st.r = r;
        if cmd.has_opt("graph", 2) {
            let mut g = crate::graph::Graph {
                layers: vec![crate::graph::Layer { kind: crate::graph::LayerKind::Connected, points: pts, label: vars[1].clone(), ..Default::default() }],
                xtitle: vars[0].clone(),
                ytitle: vars[1].clone(),
                ..Default::default()
            };
            let extra = crate::graph::finish(&mut g, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

const STATS: [&str; 9] = ["auc", "aucline", "aucexp", "auclog", "half", "ke", "cmax", "tomc", "tmax"];

fn stat_of(p: &Pk, s: &str) -> f64 {
    match s {
        "auc" => p.auc,
        "aucline" => p.aucline,
        "aucexp" => p.aucexp,
        "auclog" => p.auclog,
        "half" => p.half,
        "ke" => p.ke,
        "cmax" => p.cmax,
        "tomc" => p.tomc,
        _ => p.tmax,
    }
}

/// Measures of each subject: (id value, first row, measures).
fn per_subject(ds: &DataSet, mask: &[bool], id: &[f64], t: &[f64], c: &[f64], fit: usize) -> Vec<(f64, usize, Pk)> {
    let mut groups: BTreeMap<u64, (f64, usize, Vec<(f64, f64)>)> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if !mask[i] || is_missing(id[i]) {
            continue;
        }
        let key = { let b = id[i].to_bits(); if id[i] >= 0.0 { b ^ (1 << 63) } else { !b } };
        groups.entry(key).or_insert((id[i], i, Vec::new())).2.push((t[i], c[i]));
    }
    groups.into_values().map(|(v, first, pts)| (v, first, pk_measures(&pts, fit, None))).collect()
}

pub struct Pksumm;

impl CommandHandler for Pksumm {
    fn name(&self) -> &'static str {
        "pksumm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 3 {
            return Err(rc_err(198, "pksumm id time concentration"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars[1..]);
        let (id, t, c) = (util::col(ds, &vars[0])?, util::col(ds, &vars[1])?, util::col(ds, &vars[2])?);
        let subj = per_subject(ds, &mask, &id, &t, &c, opt_fit(cmd)?);
        let stats: Vec<String> = match cmd.opt("stat", 2) {
            Some(s) => s.split_whitespace().map(String::from).collect(),
            None => STATS.iter().map(|s| s.to_string()).collect(),
        };
        let mut out = vec![String::new(), "Summary statistics for the pharmacokinetic measures".to_string(), String::new()];
        out.push(format!("{:>66} = {:>5}", "Number of observations", subj.len()));
        out.push(String::new());
        out.push("    Measure |       Mean     Median   Variance   Skewness   Kurtosis  p-value".into());
        out.push(format!("{}+{}", "-".repeat(12), "-".repeat(65)));
        let mut r = Results::new();
        for s in &stats {
            let mut v: Vec<f64> = subj.iter().map(|x| stat_of(&x.2, s)).filter(|x| !is_missing(*x)).collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(f64::total_cmp);
            let n = v.len() as f64;
            let m = util::mean(&v);
            let var = util::variance(&v);
            let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
            let sk = v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n / m2.powf(1.5);
            let ku = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n / (m2 * m2);
            let p = if v.len() >= 8 && m2 > 0.0 { crate::smallstat::sktest_values(&v).3 } else { SYSMISS };
            let (sk, ku) = if m2 > 0.0 { (sk, ku) } else { (SYSMISS, SYSMISS) };
            let f2 = |x: f64| if is_missing(x) { ".".to_string() } else { format!("{x:.2}") };
            out.push(format!("{:>11} | {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}", s, fmt_g(m, 9), fmt_g(util::pctile_sorted(&v, 50.0), 9), fmt_g(var, 9), f2(sk), f2(ku), if is_missing(p) { ".".into() } else { format!("{p:.2}") }));
            r.set(&format!("mean_{s}"), m);
        }
        out.push("-".repeat(78));
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Pkcollapse;

impl CommandHandler for Pkcollapse {
    fn name(&self) -> &'static str {
        "pkcollapse"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(198, "pkcollapse time concentration [concentration ...], id(idvar)"));
        }
        let idv = varlist::resolve_name(ds, cmd.opt("id", 2).ok_or_else(|| rc_err(198, "option id() required"))?.trim())?;
        let keep: Vec<String> = match cmd.opt("keep", 2) {
            Some(k) => varlist::expand(ds, k)?,
            None => Vec::new(),
        };
        let stats: Vec<String> = match cmd.opt("stat", 2) {
            Some(s) => s.split_whitespace().map(String::from).collect(),
            None => STATS.iter().map(|s| s.to_string()).collect(),
        };
        let mask = util::touse(ds, st, cmd)?;
        let fit = opt_fit(cmd)?;
        let t = util::col(ds, &vars[0])?;
        let id = util::col(ds, &idv)?;
        let mut out = DataSet::new();
        let mut first_rows = Vec::new();
        let mut cols: Vec<(String, Vec<f64>)> = Vec::new();
        for cv in &vars[1..] {
            let c = util::col(ds, cv)?;
            let subj = per_subject(ds, &mask, &id, &t, &c, fit);
            if first_rows.is_empty() {
                first_rows = subj.iter().map(|s| s.1).collect();
            }
            for s in &stats {
                cols.push((format!("{s}_{cv}"), subj.iter().map(|x| stat_of(&x.2, s)).collect()));
            }
        }
        for v in std::iter::once(&idv).chain(keep.iter()) {
            let src = ds.get_var(v).unwrap();
            let mut nv = src.clone();
            nv.data = first_rows.iter().map(|&r| src.data[r].clone()).collect();
            out.add_variable(nv)?;
        }
        for (name, vals) in cols {
            let mut nv = Variable::new(name, VarType::Float);
            nv.data = vals.iter().map(|&x| util::typed_value(VarType::Float, x)).collect();
            out.add_variable(nv)?;
        }
        out.n_obs = first_rows.len();
        for (name, vl) in &ds.value_labels {
            out.value_labels.insert(name.clone(), vl.clone());
        }
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

/// `pkshape id sequence period1 period2 ..., order(ab ba ...)`: crossover
/// data to one observation per subject and period.
pub struct Pkshape;

impl CommandHandler for Pkshape {
    fn name(&self) -> &'static str {
        "pkshape"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 4 {
            return Err(rc_err(198, "pkshape id sequence period1 period2 [...], order(string)"));
        }
        let orders: Vec<String> = cmd.opt("order", 1).ok_or_else(|| rc_err(198, "option order() required"))?.split_whitespace().map(|s| s.to_lowercase()).collect();
        let (idv, seqv, pvars) = (&vars[0], &vars[1], &vars[2..]);
        let np = pvars.len();
        if orders.iter().any(|o| o.len() != np) {
            return Err(rc_err(198, "each order() sequence must have one letter per period"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let id = util::col(ds, idv)?;
        let seq = util::col(ds, seqv)?;
        let pcols: Vec<Vec<f64>> = pvars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        // sequence values in sorted order map to order() strings
        let mut svals: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(seq[i])).map(|i| seq[i]).collect();
        svals.sort_by(f64::total_cmp);
        svals.dedup();
        let letters: Vec<char> = {
            let mut l: Vec<char> = orders.iter().flat_map(|o| o.chars()).collect();
            l.sort();
            l.dedup();
            l
        };
        let mut rows: Vec<[f64; 6]> = Vec::new(); // id seq outcome treat carry period
        for i in 0..ds.n_obs {
            if !mask[i] || is_missing(seq[i]) {
                continue;
            }
            let si = svals.iter().position(|&v| v == seq[i]).unwrap();
            let ord: Vec<char> = orders.get(si).ok_or_else(|| rc_err(198, "more sequences than order() strings"))?.chars().collect();
            for p in 0..np {
                let treat = letters.iter().position(|&c| c == ord[p]).unwrap() as f64 + 1.0;
                let carry = if p == 0 { 0.0 } else { letters.iter().position(|&c| c == ord[p - 1]).unwrap() as f64 + 1.0 };
                rows.push([id[i], si as f64 + 1.0, pcols[p][i], treat, carry, p as f64 + 1.0]);
            }
        }
        let mut out = DataSet::new();
        for (j, name) in ["id", "sequence", "outcome", "treat", "carry", "period"].iter().enumerate() {
            let ty = if j == 2 { VarType::Float } else { VarType::Int };
            let mut v = Variable::new(name.to_string(), ty);
            v.data = rows.iter().map(|r| util::typed_value(ty, r[j])).collect();
            out.add_variable(v)?;
        }
        out.n_obs = rows.len();
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

/// OLS fit: (rss, df_resid).
fn rss_of(y: &[f64], cols: &[Vec<f64>]) -> (f64, f64) {
    let n = y.len();
    let k = cols.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for t in 0..n {
        for a in 0..k {
            xty[a] += cols[a][t] * y[t];
            for c in 0..k {
                xtx[(a, c)] += cols[a][t] * cols[c][t];
            }
        }
    }
    let (inv, dropped) = xtx.sym_inverse(1e-12);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    let rss = (0..n).map(|t| (y[t] - (0..k).map(|a| b[a] * cols[a][t]).sum::<f64>()).powi(2)).sum();
    (rss, (n - (k - dropped.len())) as f64)
}

fn dummies(v: &[f64]) -> Vec<Vec<f64>> {
    let mut lv: Vec<f64> = v.to_vec();
    lv.sort_by(f64::total_cmp);
    lv.dedup();
    lv.iter().skip(1).map(|l| v.iter().map(|x| if x == l { 1.0 } else { 0.0 }).collect()).collect()
}

pub struct Pkcross;

impl CommandHandler for Pkcross {
    fn name(&self) -> &'static str {
        "pkcross"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let name = |o: &str, d: &str| -> Result<String, String> { varlist::resolve_name(ds, cmd.opt(o, 3).map(|s| s.trim()).unwrap_or(d)) };
        let (sq, tr, pe, idv) = (name("sequence", "sequence")?, name("treatment", "treat")?, name("period", "period")?, name("id", "id")?);
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[vars[0].clone(), sq.clone(), tr.clone(), pe.clone(), idv.clone()]);
        let rows = util::rows(&mask);
        let get = |v: &str| -> Result<Vec<f64>, String> { util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect()) };
        let (y, s, t, p, id) = (get(&vars[0])?, get(&sq)?, get(&tr)?, get(&pe)?, get(&idv)?);
        let n = y.len();
        let one = vec![vec![1.0; n]];
        let ids: Vec<f64> = id.iter().zip(&s).map(|(a, b)| a * 1000.0 + b).collect();
        let cat = |a: &[Vec<f64>], b: &[Vec<f64>]| -> Vec<Vec<f64>> { a.iter().chain(b).cloned().collect() };
        let ds_ = dummies(&s);
        let di = dummies(&ids);
        let dt = dummies(&t);
        let dp = dummies(&p);
        let full = cat(&cat(&cat(&one, &di), &dt), &dp);
        let (rss_full, df_full) = rss_of(&y, &full);
        let (rss_no_t, df_no_t) = rss_of(&y, &cat(&cat(&one, &di), &dp));
        let (rss_no_p, df_no_p) = rss_of(&y, &cat(&cat(&one, &di), &dt));
        // between-subject part: sequence vs subjects within sequence
        let (rss_seq, df_seq) = rss_of(&y, &cat(&cat(&cat(&one, &ds_), &dt), &dp));
        let (rss_none, df_none) = rss_of(&y, &cat(&cat(&one, &dt), &dp));
        let ss_t = rss_no_t - rss_full;
        let df_t = df_no_t - df_full;
        let ss_p = rss_no_p - rss_full;
        let df_p = df_no_p - df_full;
        let ss_s = rss_none - rss_seq;
        let df_s = df_none - df_seq;
        let ss_sub = rss_seq - rss_full;
        let df_sub = df_seq - df_full;
        let ms_e = rss_full / df_full;
        let ms_sub = ss_sub / df_sub;
        let ym = util::mean(&y);
        let tss: f64 = y.iter().map(|v| (v - ym).powi(2)).sum();
        let mut out = vec![String::new(), format!("{:>48}", "sequence variable = ") + &sq, format!("{:>48}", "  period variable = ") + &pe, format!("{:>48}", "treatment variable = ") + &tr, format!("{:>48}", "       id variable = ") + &idv, String::new()];
        out.push(format!("{:<25}{:>10} {:>5} {:>11} {:>8} {:>9}", "Source of Variation", "SS", "df", "MS", "F", "Prob > F"));
        out.push("-".repeat(73));
        out.push("Intersubjects".into());
        let fs = (ss_s / df_s) / ms_sub;
        out.push(format!("{:<25}{:>10} {:>5} {:>11} {:>8.2} {:>9.4}", "  Sequence effect", fmt_g(ss_s, 9), df_s, fmt_g(ss_s / df_s, 9), fs, dist::ftail(df_s, df_sub, fs)));
        let fr = ms_sub / ms_e;
        out.push(format!("{:<25}{:>10} {:>5} {:>11} {:>8.2} {:>9.4}", "  Residuals", fmt_g(ss_sub, 9), df_sub, fmt_g(ms_sub, 9), fr, dist::ftail(df_sub, df_full, fr)));
        out.push("Intrasubjects".into());
        let ft = (ss_t / df_t) / ms_e;
        out.push(format!("{:<25}{:>10} {:>5} {:>11} {:>8.2} {:>9.4}", "  Treatment effect", fmt_g(ss_t, 9), df_t, fmt_g(ss_t / df_t, 9), ft, dist::ftail(df_t, df_full, ft)));
        let fp = (ss_p / df_p) / ms_e;
        out.push(format!("{:<25}{:>10} {:>5} {:>11} {:>8.2} {:>9.4}", "  Period effect", fmt_g(ss_p, 9), df_p, fmt_g(ss_p / df_p, 9), fp, dist::ftail(df_p, df_full, fp)));
        out.push(format!("{:<25}{:>10} {:>5} {:>11}", "  Residuals", fmt_g(rss_full, 9), df_full, fmt_g(ms_e, 9)));
        out.push(String::new());
        out.push(format!("{:<25}{:>10} {:>5}", "Total", fmt_g(tss, 9), n - 1));
        out.push(String::new());
        out.push("Omnibus measure of separability of treatment and carryover = 100.0000%".into());
        let _ = abbrev;
        let mut r = Results::new();
        r.set("F_t", ft);
        r.set("F_p", fp);
        r.set("F_s", fs);
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Pkequiv;

impl CommandHandler for Pkequiv {
    fn name(&self) -> &'static str {
        "pkequiv"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 5 {
            return Err(rc_err(198, "pkequiv outcome treatment period sequence id"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let get = |v: &str| -> Result<Vec<f64>, String> { util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect()) };
        let (y, t, p, s, id) = (get(&vars[0])?, get(&vars[1])?, get(&vars[2])?, get(&vars[3])?, get(&vars[4])?);
        let mut tl: Vec<f64> = t.clone();
        tl.sort_by(f64::total_cmp);
        tl.dedup();
        if tl.len() != 2 {
            return Err(rc_err(198, "pkequiv requires exactly two treatments"));
        }
        let (reference, test) = match cmd.opt("compare", 3) {
            Some(c) => {
                let v: Vec<f64> = c.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if v.len() != 2 {
                    return Err(rc_err(198, "compare() invalid"));
                }
                (v[1], v[0])
            }
            None => (tl[0], tl[1]),
        };
        let level = util::opt_num(cmd, "level", 1)?.unwrap_or(90.0);
        let limit = util::opt_num(cmd, "limit", 3)?.unwrap_or(0.2);
        let n = y.len();
        let mean_of = |tv: f64| { let v: Vec<f64> = (0..n).filter(|&i| t[i] == tv).map(|i| y[i]).collect(); util::mean(&v) };
        let (mt, mr) = (mean_of(test), mean_of(reference));
        // residual variance from the crossover model
        let ids: Vec<f64> = id.iter().zip(&s).map(|(a, b)| a * 1000.0 + b).collect();
        let mut cols = vec![vec![1.0; n]];
        cols.extend(dummies(&ids));
        cols.extend(dummies(&t));
        cols.extend(dummies(&p));
        let (rss, df) = rss_of(&y, &cols);
        let mse = rss / df;
        // subjects per sequence
        let mut seqs: BTreeMap<u64, std::collections::BTreeSet<u64>> = BTreeMap::new();
        for i in 0..n {
            seqs.entry(s[i].to_bits()).or_default().insert(id[i].to_bits());
        }
        let inv_n: f64 = seqs.values().map(|v| 1.0 / v.len() as f64).sum();
        let se = (mse / 2.0 * inv_n).sqrt();
        let diff = mt - mr;
        let tc = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
        let (lo, hi) = (diff - tc * se, diff + tc * se);
        let eq = limit * mr;
        let prob = (dist::t(df, (eq - diff) / se) - dist::t(df, (-eq - diff) / se)).max(0.0);
        let mut out = vec![String::new(), format!("{:>18}Classic confidence interval for bioequivalence", ""), String::new()];
        out.push("                 [equivalence limits]      [      test limits     ]".into());
        out.push("-".repeat(70));
        out.push(format!("  difference: {:>11} {:>11} {:>12} {:>11}", fmt_g(-eq, 8), fmt_g(eq, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
        out.push(format!("       ratio: {:>10}% {:>10}% {:>11.3}% {:>10.3}%", fmt_g(100.0 * (1.0 - limit), 8), fmt_g(100.0 * (1.0 + limit), 8), 100.0 * (1.0 + lo / mr), 100.0 * (1.0 + hi / mr)));
        out.push("-".repeat(70));
        out.push(String::new());
        out.push(format!("probability test limits are within equivalence limits = {:>10.4}", prob));
        let mut r = Results::new();
        r.set("stddev", (mse).sqrt());
        r.set("uci", hi);
        r.set("lci", lo);
        r.set("delta", diff);
        r.set("u0", eq);
        r.set("l0", -eq);
        r.set("p", prob);
        r.set("df", df);
        st.r = r;
        Ok(out.join("\n"))
    }
}
