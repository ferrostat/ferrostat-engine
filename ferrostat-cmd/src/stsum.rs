//! Survival summaries over stset data: `sts list`, `sts test`,
//! `sts generate`, `stsum`, `stci`, `stdescribe`, `stptime`, and the
//! record-splitting commands `stsplit` and `stjoin`.
//!
//! Kaplan-Meier with delayed entry (a record is at risk at t when
//! `_t0 < t <= _t`), Greenwood standard errors and log(-log) intervals;
//! Nelson-Aalen cumulative hazard with log-transformed intervals. The
//! k-sample tests are the weighted log-rank family (Klein & Moeschberger,
//! *Survival Analysis*, section 7.3).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::survival::{gc, st_header, st_id, st_obs, StObs};
use crate::util;
use crate::varlist;

// ---------------------------------------------------------------------------
// Kaplan-Meier / Nelson-Aalen
// ---------------------------------------------------------------------------

/// One step of the product-limit estimate at a distinct exit time.
#[derive(Debug, Clone, Copy)]
pub struct KmStep {
    pub t: f64,
    /// number at risk (weighted)
    pub n: f64,
    pub d: f64,
    /// censored at t
    pub c: f64,
    /// entering since the previous exit time (delayed entry)
    pub ent: f64,
    pub s: f64,
    /// Greenwood sum of d/(n(n-d))
    pub gw: f64,
    /// Nelson-Aalen cumulative hazard and its variance
    pub h: f64,
    pub hv: f64,
}

/// Sorted-array counter: number of values >= x.
fn count_ge(sorted: &[(f64, f64)], x: f64) -> f64 {
    let k = sorted.partition_point(|p| p.0 < x);
    sorted[k..].iter().map(|p| p.1).sum()
}

/// Product-limit table over all distinct exit times.
pub fn km(obs: &[StObs]) -> Vec<KmStep> {
    let mut ts: Vec<(f64, f64)> = obs.iter().map(|o| (o.t, o.w)).collect();
    let mut t0s: Vec<(f64, f64)> = obs.iter().map(|o| (o.t0, o.w)).collect();
    ts.sort_by(|a, b| a.0.total_cmp(&b.0));
    t0s.sort_by(|a, b| a.0.total_cmp(&b.0));
    // suffix sums for fast at-risk counts
    let suffix = |v: &[(f64, f64)]| {
        let mut s = vec![0.0; v.len() + 1];
        for i in (0..v.len()).rev() {
            s[i] = s[i + 1] + v[i].1;
        }
        s
    };
    let (st, st0) = (suffix(&ts), suffix(&t0s));
    let ge = |v: &[(f64, f64)], s: &[f64], x: f64| s[v.partition_point(|p| p.0 < x)];
    let mut times: Vec<f64> = ts.iter().map(|p| p.0).collect();
    times.dedup();
    let mut by_t: std::collections::BTreeMap<u64, (f64, f64)> = Default::default();
    for o in obs {
        let e = by_t.entry(o.t.to_bits()).or_default();
        if o.d {
            e.0 += o.w;
        } else {
            e.1 += o.w;
        }
    }
    let (mut s, mut gw, mut h, mut hv) = (1.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut prev = f64::NEG_INFINITY;
    let mut out = Vec::with_capacity(times.len());
    for &t in &times {
        // at risk: t0 < t <= t_i  ==  #(t_i >= t) - #(t0_i >= t)
        let n = ge(&ts, &st, t) - ge(&t0s, &st0, t);
        let (d, c) = by_t[&t.to_bits()];
        let ent: f64 = t0s.iter().filter(|p| p.0 > 0.0 && p.0 >= prev && p.0 < t).map(|p| p.1).sum();
        if d > 0.0 && n > 0.0 {
            s *= 1.0 - d / n;
            if n > d {
                gw += d / (n * (n - d));
            }
            h += d / n;
            hv += d / (n * n);
        }
        out.push(KmStep { t, n, d, c, ent, s, gw, h, hv });
        prev = t;
    }
    let _ = count_ge;
    out
}

/// Survivor function at time t (right-continuous step function).
pub fn s_at(steps: &[KmStep], t: f64) -> (f64, f64) {
    let k = steps.partition_point(|s| s.t <= t);
    if k == 0 { (1.0, 0.0) } else { (steps[k - 1].s, steps[k - 1].gw) }
}

/// log(-log) confidence interval of a survivor estimate.
pub fn s_ci(s: f64, gw: f64, z: f64) -> (f64, f64) {
    if s <= 0.0 || s >= 1.0 || gw <= 0.0 {
        return (SYSMISS, SYSMISS);
    }
    let sig = gw.sqrt() / s.ln().abs();
    (s.powf((z * sig).exp()), s.powf((-z * sig).exp()))
}

/// Log-transformed interval of the Nelson-Aalen estimate.
fn h_ci(h: f64, hv: f64, z: f64) -> (f64, f64) {
    if h <= 0.0 {
        return (SYSMISS, SYSMISS);
    }
    let f = (z * hv.sqrt() / h).exp();
    (h / f, h * f)
}

/// p-th percentile of survival time (smallest t with S(t) <= 1 - p).
pub fn km_percentile(steps: &[KmStep], p: f64) -> f64 {
    steps.iter().find(|s| s.d > 0.0 && s.s <= 1.0 - p + 1e-12).map_or(SYSMISS, |s| s.t)
}

fn zcrit(level: f64) -> f64 {
    dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0)
}

/// Groups of observations by the values of `vars` (labels "a = 1, b = 2").
fn groups(ds: &DataSet, vars: &[String], obs: &[StObs]) -> Result<Vec<(Vec<f64>, String, Vec<StObs>)>, String> {
    if vars.is_empty() {
        return Ok(vec![(Vec::new(), String::new(), obs.to_vec())]);
    }
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let mut m: std::collections::BTreeMap<Vec<u64>, Vec<StObs>> = Default::default();
    let key = |row: usize| -> Option<Vec<f64>> {
        let k: Vec<f64> = cols.iter().map(|c| c[row]).collect();
        if k.iter().any(|x| is_missing(*x)) { None } else { Some(k) }
    };
    for o in obs {
        if let Some(k) = key(o.row) {
            m.entry(k.iter().map(|x| order_bits(*x)).collect()).or_default().push(*o);
        }
    }
    let mut out = Vec::new();
    for (_, rows) in m {
        let k = key(rows[0].row).unwrap();
        let label = vars
            .iter()
            .map(|v| {
                let var = ds.get_var(v).unwrap();
                format!("{v} = {}", util::cell_text(ds, var, rows[0].row, true))
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.push((k, label, rows));
    }
    Ok(out)
}

/// Bits of a float that sort in numeric order.
fn order_bits(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

fn by_vars(ds: &DataSet, cmd: &Command) -> Result<Vec<String>, String> {
    match cmd.opt("by", 2) {
        Some(v) => varlist::expand(ds, v),
        None => Ok(Vec::new()),
    }
}

fn header_or_not(ds: &DataSet, cmd: &Command) -> Vec<String> {
    if cmd.has_opt("noshow", 4) { Vec::new() } else { vec![st_header(ds), String::new()] }
}

// ---------------------------------------------------------------------------
// sts list / generate / test
// ---------------------------------------------------------------------------

/// `sts list|test|generate`.
pub fn sts(ds: &mut DataSet, st: &mut Store, cmd: &Command, sub: &str) -> Result<String, String> {
    match sub {
        "list" | "li" => sts_list(ds, st, cmd),
        "test" | "t" => sts_test(ds, st, cmd),
        "generate" | "gen" | "g" => sts_generate(ds, st, cmd),
        _ => Err(rc_err(198, format!("sts {sub}: invalid subcommand"))),
    }
}

fn sts_list(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let obs = st_obs(ds, st, cmd)?;
    let level = util::level(cmd, st)?;
    let z = zcrit(level);
    let cumhaz = cmd.has_opt("cumhaz", 4) || cmd.has_opt("na", 2);
    let failure = cmd.has_opt("failure", 4);
    let at: Option<Vec<f64>> = cmd.opt("at", 2).map(crate::margins::numlist).transpose()?;
    let vars = by_vars(ds, cmd)?;
    let gs = groups(ds, &vars, &obs)?;
    let delayed = obs.iter().any(|o| o.t0 > 0.0);
    let mut out = header_or_not(ds, cmd);
    out.push(if cumhaz { "Nelson–Aalen cumulative hazard function".into() } else if failure { "Kaplan–Meier failure function".into() } else { "Kaplan–Meier survivor function".into() });
    if !vars.is_empty() {
        out.push(format!("By variable: {}", vars.join(" ")));
    }
    out.push(String::new());
    let lost = if delayed { "Net lost" } else { "Lost" };
    let (c1, c2) = if cumhaz { ("Nelson–Aalen", "cum. haz.") } else if failure { ("Failure", "function") } else { ("Survivor", "function") };
    if at.is_some() {
        out.push(format!("{:>6} {:>9} {:>6} {:>8} {:>10} {:>9}", "", "Beg.", "", "", c1, "Std."));
        out.push(format!("{:>6} {:>9} {:>6} {:>8} {:>10} {:>9}     [{}% conf. int.]", "Time", "total", "Fail", lost, c2, "error", fmt_g(level, 4)));
    } else {
        out.push(format!("{:>6} {:>9} {:>6} {:>8} {:>10} {:>9}", "", "At", "", "", c1, "Std."));
        out.push(format!("{:>6} {:>9} {:>6} {:>8} {:>10} {:>9}     [{}% conf. int.]", "Time", "risk", "Fail", lost, c2, "error", fmt_g(level, 4)));
    }
    let bar = "-".repeat(79);
    out.push(bar.clone());
    let row = |t: f64, n: f64, d: f64, c: f64, step: Option<(f64, f64, f64, f64)>| -> String {
        // step: (s, gw, h, hv)
        let (s, gw, h, hv) = step.unwrap_or((1.0, 0.0, 0.0, 0.0));
        let (est, se, lo, hi) = if cumhaz {
            let (lo, hi) = h_ci(h, hv, z);
            (h, hv.sqrt(), lo, hi)
        } else {
            let (lo, hi) = s_ci(s, gw, z);
            if failure { (1.0 - s, s * gw.sqrt(), if is_missing(hi) { hi } else { 1.0 - hi }, if is_missing(lo) { lo } else { 1.0 - lo }) } else { (s, s * gw.sqrt(), lo, hi) }
        };
        let f4 = |x: f64| if is_missing(x) { ".".to_string() } else { format!("{x:.4}") };
        format!("{:>6} {:>9} {:>6} {:>8} {:>10} {:>9} {:>10} {:>9}", fmt_g(t, 6), fmt_g(n, 9), fmt_g(d, 6), fmt_g(c, 8), f4(est), f4(se), f4(lo), f4(hi))
    };
    for (_, label, rows) in &gs {
        if !label.is_empty() {
            out.push(label.clone());
        }
        let steps = km(rows);
        match &at {
            None => {
                for s in &steps {
                    out.push(row(s.t, s.n, s.d, s.c - s.ent, Some((s.s, s.gw, s.h, s.hv))));
                }
            }
            Some(ts) => {
                let mut prev = f64::NEG_INFINITY;
                for &t in ts {
                    let inside: Vec<&KmStep> = steps.iter().filter(|s| s.t > prev && s.t <= t).collect();
                    let beg = steps.iter().find(|s| s.t > prev).map_or(0.0, |s| s.n);
                    let d: f64 = inside.iter().map(|s| s.d).sum();
                    let c: f64 = inside.iter().map(|s| s.c - s.ent).sum();
                    let k = steps.partition_point(|s| s.t <= t);
                    let st = if k == 0 { None } else { let s = steps[k - 1]; Some((s.s, s.gw, s.h, s.hv)) };
                    out.push(row(t, beg, d, c, st));
                    prev = t;
                }
            }
        }
    }
    out.push(bar);
    if at.is_some() {
        out.push("Note: Survivor function is calculated over full data and evaluated at".into());
        out.push("      indicated times; it is not calculated from aggregates shown at left.".into());
    }
    Ok(out.join("\n"))
}

fn sts_generate(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    // sts generate newvar = s [newvar = se(s) ...] [, by()]
    let text = cmd.main.trim_start();
    let text = text.split_once(char::is_whitespace).map_or("", |(_, r)| r);
    let obs = st_obs(ds, st, cmd)?;
    let level = util::level(cmd, st)?;
    let z = zcrit(level);
    let vars = by_vars(ds, cmd)?;
    let gs = groups(ds, &vars, &obs)?;
    // parse "a = s b = se(s)" pairs
    let mut specs: Vec<(String, String)> = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        let (name, r) = rest.split_once('=').ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let r = r.trim_start();
        let end = {
            let mut depth = 0i32;
            let mut e = r.len();
            for (i, c) in r.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    c if c.is_whitespace() && depth == 0 => {
                        e = i;
                        break;
                    }
                    _ => {}
                }
            }
            e
        };
        specs.push((name.trim().to_string(), r[..end].replace(' ', "").to_lowercase()));
        rest = r[end..].trim();
    }
    if specs.is_empty() {
        return Err(rc_err(198, "nothing to generate"));
    }
    for (name, _) in &specs {
        varlist::check_name(name)?;
        if ds.var_index.contains_key(name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
    }
    let n = ds.n_obs;
    let mut cols: Vec<(String, String, Vec<f64>)> = specs.iter().map(|(a, b)| (a.clone(), b.clone(), vec![SYSMISS; n])).collect();
    for (_, _, rows) in &gs {
        let steps = km(rows);
        for o in rows {
            let k = steps.partition_point(|s| s.t <= o.t);
            let s = steps[k - 1];
            for (_, what, col) in cols.iter_mut() {
                col[o.row] = match what.as_str() {
                    "s" => s.s,
                    "se(s)" => s.s * s.gw.sqrt(),
                    "lb(s)" => s_ci(s.s, s.gw, z).0,
                    "ub(s)" => s_ci(s.s, s.gw, z).1,
                    "f" => 1.0 - s.s,
                    "na" => s.h,
                    "se(na)" => s.hv.sqrt(),
                    "lb(na)" => h_ci(s.h, s.hv, z).0,
                    "ub(na)" => h_ci(s.h, s.hv, z).1,
                    "h" => if s.n > 0.0 { s.d / s.n } else { 0.0 },
                    "n" => s.n,
                    "d" => s.d,
                    w => return Err(rc_err(198, format!("{w} invalid"))),
                };
            }
        }
    }
    for (name, what, col) in cols {
        let mut v = util::new_double(&name, col);
        v.var_type = VarType::Float;
        v.label = match what.as_str() {
            "s" => "S(t+0)".into(),
            "na" => "Nelson–Aalen cumulative hazard".into(),
            w => w.to_string(),
        };
        ds.add_variable(v)?;
    }
    Ok(String::new())
}

/// Weights of the log-rank family at one failure time.
#[derive(Clone, Copy, PartialEq)]
enum TestKind {
    Logrank,
    Wilcoxon,
    Tware,
    Peto,
    Fh(f64, f64),
}

fn sts_test(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().skip(1).collect();
    let vars = varlist::expand(ds, &words.join(" "))?;
    if vars.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let obs = st_obs(ds, st, cmd)?;
    let kind = if cmd.has_opt("wilcoxon", 1) {
        TestKind::Wilcoxon
    } else if cmd.has_opt("tware", 2) {
        TestKind::Tware
    } else if cmd.has_opt("peto", 1) {
        TestKind::Peto
    } else if let Some(f) = cmd.opt("fh", 2) {
        let v = crate::margins::numlist(f)?;
        TestKind::Fh(v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0))
    } else {
        TestKind::Logrank
    };
    let strata: Vec<String> = match cmd.opt("strata", 3) {
        Some(s) => varlist::expand(ds, s)?,
        None => Vec::new(),
    };
    let trend = cmd.has_opt("trend", 2);
    let gcols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let obs: Vec<StObs> = obs.into_iter().filter(|o| gcols.iter().all(|c| !is_missing(c[o.row]))).collect();
    let gs = groups(ds, &vars, &obs)?;
    let ng = gs.len();
    if ng < 2 {
        return Err(rc_err(2000, "only one group; nothing to compare"));
    }
    let gidx = |row: usize| gs.iter().position(|(k, _, _)| k.iter().zip(&gcols).all(|(a, c)| c[row] == *a)).unwrap();
    let og: Vec<usize> = obs.iter().map(|o| gidx(o.row)).collect();
    let sgs = groups(ds, &strata, &obs)?;
    let mut u = vec![0.0; ng];
    let mut v = Matrix::zeros(ng, ng);
    let mut observed = vec![0.0; ng];
    let mut expected = vec![0.0; ng];
    let mut ranks = vec![0.0; ng];
    let pos: std::collections::HashMap<usize, usize> = obs.iter().enumerate().map(|(k, o)| (o.row, k)).collect();
    for (_, _, srows) in &sgs {
        let mut times: Vec<f64> = srows.iter().filter(|o| o.d).map(|o| o.t).collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let mut s_pool = 1.0f64; // left-continuous pooled KM
        let mut s_peto = 1.0f64;
        for &t in &times {
            let mut nj = vec![0.0; ng];
            let mut dj = vec![0.0; ng];
            for o in srows {
                let g = og[pos[&o.row]];
                if o.t0 < t && o.t >= t {
                    nj[g] += o.w;
                    if o.t == t && o.d {
                        dj[g] += o.w;
                    }
                }
            }
            let n: f64 = nj.iter().sum();
            let d: f64 = dj.iter().sum();
            if n <= 0.0 {
                continue;
            }
            s_peto *= 1.0 - d / (n + 1.0);
            let w = match kind {
                TestKind::Logrank => 1.0,
                TestKind::Wilcoxon => n,
                TestKind::Tware => n.sqrt(),
                TestKind::Peto => s_peto,
                TestKind::Fh(p, q) => s_pool.powf(p) * (1.0 - s_pool).powf(q),
            };
            for g in 0..ng {
                let e = nj[g] * d / n;
                observed[g] += dj[g];
                expected[g] += e;
                u[g] += w * (dj[g] - e);
                ranks[g] += w * (dj[g] - e);
                if n > 1.0 {
                    for h in 0..ng {
                        let delta = if g == h { 1.0 } else { 0.0 };
                        v[(g, h)] += w * w * d * (n - d) / (n - 1.0) * nj[g] / n * (delta - nj[h] / n);
                    }
                }
            }
            s_pool *= 1.0 - d / n;
        }
    }
    let (chi2, df) = if trend {
        let sc: Vec<f64> = gs.iter().map(|(k, _, _)| k[0]).collect();
        let num: f64 = sc.iter().zip(&u).map(|(a, b)| a * b).sum();
        let den: f64 = (0..ng).map(|a| (0..ng).map(|b| sc[a] * v[(a, b)] * sc[b]).sum::<f64>()).sum();
        (num * num / den, 1.0)
    } else {
        let k = ng - 1;
        let mut vs = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                vs[(a, b)] = v[(a, b)];
            }
        }
        let (vi, dropped) = vs.sym_inverse(1e-12);
        let chi2: f64 = (0..k).map(|a| (0..k).map(|b| u[a] * vi[(a, b)] * u[b]).sum::<f64>()).sum();
        (chi2, (k - dropped.len()) as f64)
    };
    let p = dist::chi2tail(df, chi2);
    let title = match kind {
        TestKind::Logrank => "Log-rank test".to_string(),
        TestKind::Wilcoxon => "Wilcoxon–Breslow–Gehan test".to_string(),
        TestKind::Tware => "Tarone–Ware test".to_string(),
        TestKind::Peto => "Peto–Peto–Prentice test".to_string(),
        TestKind::Fh(p, q) => format!("Fleming–Harrington test, p = {}, q = {}", fmt_g(p, 6), fmt_g(q, 6)),
    };
    let mut out = header_or_not(ds, cmd);
    out.push("Equality of survivor functions".into());
    out.push(if strata.is_empty() { title } else { format!("Stratified {}{}", title[..1].to_lowercase(), &title[1..]) });
    if !strata.is_empty() {
        out.push(format!("Stratified by {}", strata.join(" ")));
    }
    out.push(String::new());
    let lab = vars.join(" ");
    let lw = lab.len().max(gs.iter().map(|(_, l, _)| level_text(l).len()).max().unwrap_or(1)).max(5);
    let with_ranks = kind != TestKind::Logrank;
    let extra_h = if with_ranks { ("        Sum of", "         ranks") } else { ("", "") };
    out.push(format!("{:<lw$} |  Observed       Expected{}", "", extra_h.0));
    out.push(format!("{:<lw$} |    events         events{}", lab, extra_h.1));
    let rule = format!("{}+{}", "-".repeat(lw + 1), "-".repeat(if with_ranks { 39 } else { 25 }));
    out.push(rule.clone());
    for (g, (_, l, _)) in gs.iter().enumerate() {
        let rk = if with_ranks { format!("{:>14}", fmt_g(ranks[g], 8)) } else { String::new() };
        out.push(format!("{:>lw$} | {:>9} {:>14.2}{rk}", level_text(l), fmt_g(observed[g], 9), expected[g]));
    }
    out.push(rule);
    let tr = if with_ranks { format!("{:>14}", 0) } else { String::new() };
    out.push(format!("{:<lw$} | {:>9} {:>14.2}{tr}", "Total", fmt_g(observed.iter().sum(), 9), expected.iter().sum::<f64>()));
    out.push(String::new());
    if trend {
        out.push("Test for trend of survivor functions".into());
        out.push(String::new());
    }
    out.push(format!("{:>25} = {:>6.2}", format!("chi2({})", fmt_g(df, 4)), chi2));
    out.push(format!("{:>25} = {:>6.4}", "Pr>chi2", p));
    let mut r = Results::new();
    r.set("chi2", chi2);
    r.set("df", df);
    r.set("p", p);
    st.r = r;
    Ok(out.join("\n"))
}

/// "drug = 1, x = 2" -> "1 2" style row label for single by-variable tables.
fn level_text(label: &str) -> String {
    label.split(", ").map(|p| p.split_once(" = ").map_or(p, |(_, v)| v)).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// stsum / stci / stptime / stdescribe
// ---------------------------------------------------------------------------

pub struct Stsum;

impl CommandHandler for Stsum {
    fn name(&self) -> &'static str {
        "stsum"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let obs = st_obs(ds, st, cmd)?;
        let vars = by_vars(ds, cmd)?;
        let id = st_id(ds);
        let mut out = header_or_not(ds, cmd);
        let lw = vars.join(" ").len().max(8);
        out.push(format!("{:>lw$} |               Incidence     Number of   |------ Survival time -----|", ""));
        out.push(format!("{:>lw$} | Time at risk       rate      subjects        25%       50%       75%", vars.join(" ")));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(69)));
        let mut gs = if vars.is_empty() { Vec::new() } else { groups(ds, &vars, &obs)? };
        gs.push((Vec::new(), "Total".into(), obs.clone()));
        let mut r = Results::new();
        let ng = gs.len();
        for (k, (_, label, rows)) in gs.iter().enumerate() {
            if k == ng - 1 && ng > 1 {
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(69)));
            }
            let risk: f64 = rows.iter().map(|o| (o.t - o.t0) * o.w).sum();
            let fails: f64 = rows.iter().filter(|o| o.d).map(|o| o.w).sum();
            let nsub = subjects(ds, &id, rows);
            let steps = km(rows);
            let q: Vec<f64> = [0.25, 0.5, 0.75].iter().map(|p| km_percentile(&steps, *p)).collect();
            out.push(format!(
                "{:>lw$} | {:>12} {:>10} {:>13} {:>10} {:>9} {:>9}",
                level_text(label),
                gc(risk, 10),
                fmt_g(fails / risk, 8),
                gc(nsub, 10),
                fmt_g(q[0], 9),
                fmt_g(q[1], 9),
                fmt_g(q[2], 9)
            ));
            if k == ng - 1 {
                r.set("risk", risk);
                r.set("ir", fails / risk);
                r.set("N_sub", nsub);
                r.set("p25", q[0]);
                r.set("p50", q[1]);
                r.set("p75", q[2]);
            }
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn subjects(ds: &DataSet, id: &Option<String>, rows: &[StObs]) -> f64 {
    match id.as_ref().and_then(|v| util::col(ds, v).ok()) {
        Some(c) => {
            let mut k: Vec<u64> = rows.iter().map(|o| c[o.row].to_bits()).collect();
            k.sort();
            k.dedup();
            k.len() as f64
        }
        None => rows.iter().map(|o| o.w).sum(),
    }
}

pub struct Stci;

impl CommandHandler for Stci {
    fn name(&self) -> &'static str {
        "stci"
    }

    /// stci [, by() p(#) rmean level(#)]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let obs = st_obs(ds, st, cmd)?;
        let vars = by_vars(ds, cmd)?;
        let id = st_id(ds);
        let level = util::level(cmd, st)?;
        let z = zcrit(level);
        let p = util::opt_num(cmd, "p", 1)?.unwrap_or(50.0) / 100.0;
        let rmean = cmd.has_opt("rmean", 2);
        let mut out = header_or_not(ds, cmd);
        let lw = vars.join(" ").len().max(8);
        let (top, what) = if rmean { ("Restricted".to_string(), "mean".to_string()) } else { (String::new(), format!("{}%", fmt_g(p * 100.0, 6))) };
        out.push(format!("{:>lw$} |    Number of {:>11}", "", top));
        out.push(format!("{:>lw$} |     subjects {:>11}   Std. err.     [{}% conf. interval]", vars.join(" "), what, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let mut gs = if vars.is_empty() { Vec::new() } else { groups(ds, &vars, &obs)? };
        gs.push((Vec::new(), "Total".into(), obs.clone()));
        let ng = gs.len();
        let mut r = Results::new();
        for (k, (_, label, rows)) in gs.iter().enumerate() {
            if k == ng - 1 && ng > 1 {
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            }
            let steps = km(rows);
            let nsub = subjects(ds, &id, rows);
            let (est, se, lo, hi) = if rmean { restricted_mean(&steps, z) } else { percentile_ci(&steps, p, z) };
            let star = if rmean && steps.last().is_some_and(|s| s.d == 0.0 || s.s > 0.0) { "(*)" } else { "" };
            out.push(format!("{:>lw$} | {:>12} {:>11}{:<3}{:>9} {:>11} {:>10}", level_text(label), gc(nsub, 10), fmt_g(est, 8), star, fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
            if k == ng - 1 {
                r.set("N_sub", nsub);
                if rmean {
                    r.set("rmean", est);
                } else {
                    r.set(&format!("p{}", fmt_g(p * 100.0, 6)), est);
                }
                r.set("se", se);
                r.set("lb", lo);
                r.set("ub", hi);
            }
        }
        if rmean {
            out.push("(*) largest observed analysis time is censored, mean is underestimated".into());
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

/// Percentile with the Brookmeyer-Crowley interval (inverted log(-log)
/// band) and Collett's density-based standard error.
fn percentile_ci(steps: &[KmStep], p: f64, z: f64) -> (f64, f64, f64, f64) {
    let target = 1.0 - p;
    let est = km_percentile(steps, p);
    let mut lo = SYSMISS;
    let mut hi = SYSMISS;
    for s in steps.iter().filter(|s| s.d > 0.0) {
        let (l, u) = s_ci(s.s, s.gw, z);
        let (l, u) = if s.s >= 1.0 { (1.0, 1.0) } else if s.s <= 0.0 { (0.0, 0.0) } else { (l, u) };
        if is_missing(lo) && !is_missing(l) && l <= target + 1e-12 {
            lo = s.t;
        }
        if is_missing(hi) && !is_missing(u) && u <= target + 1e-12 {
            hi = s.t;
        }
    }
    // Collett: f = (S(u) - S(l)) / (l - u), u = largest t with S >= target+.05,
    // l = smallest t with S <= target-.05
    let se = if is_missing(est) {
        SYSMISS
    } else {
        let u = steps.iter().filter(|s| s.s >= target + 0.05).last().map_or(0.0, |s| s.t);
        let su = steps.iter().filter(|s| s.t <= u).last().map_or(1.0, |s| s.s);
        let l = steps.iter().find(|s| s.s <= target - 0.05);
        match l {
            Some(l) if l.t > u => {
                let f = (su - l.s) / (l.t - u);
                let (s, gw) = s_at(steps, est);
                if f > 0.0 { s * gw.sqrt() / f } else { SYSMISS }
            }
            _ => SYSMISS,
        }
    };
    (est, se, lo, hi)
}

/// Area under the Kaplan-Meier curve up to the last exit time.
fn restricted_mean(steps: &[KmStep], z: f64) -> (f64, f64, f64, f64) {
    let mut area = 0.0;
    let (mut prev_t, mut prev_s) = (0.0, 1.0);
    let mut pieces = Vec::new();
    for s in steps {
        area += prev_s * (s.t - prev_t);
        pieces.push((s.t, s.n, s.d));
        prev_t = s.t;
        prev_s = s.s;
    }
    // variance: sum over failure times of A_j^2 d/(n(n-d)), A_j = area after t_j
    let mut var = 0.0;
    for (j, s) in steps.iter().enumerate() {
        if s.d == 0.0 || s.n <= s.d {
            continue;
        }
        let mut a = 0.0;
        let mut ps = s.s;
        let mut pt = s.t;
        for s2 in &steps[j + 1..] {
            a += ps * (s2.t - pt);
            pt = s2.t;
            ps = s2.s;
        }
        var += a * a * s.d / (s.n * (s.n - s.d));
    }
    let se = var.sqrt();
    (area, se, area - z * se, area + z * se)
}

pub struct Stptime;

impl CommandHandler for Stptime {
    fn name(&self) -> &'static str {
        "stptime"
    }

    /// stptime [, by(var) per(#) at(numlist) trim level(#)]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let obs = st_obs(ds, st, cmd)?;
        let per = util::opt_num(cmd, "per", 2)?.unwrap_or(1.0);
        let level = util::level(cmd, st)?;
        let vars = by_vars(ds, cmd)?;
        let at: Option<Vec<f64>> = cmd.opt("at", 2).map(crate::margins::numlist).transpose()?;
        let mut out = header_or_not(ds, cmd);
        let first = if vars.is_empty() { if at.is_some() { "_group".to_string() } else { "Cohort".to_string() } } else { vars.join(" ") };
        let lw = first.len().max(6);
        let rate_head = if per == 1.0 { "Rate".to_string() } else { format!("Rate*{}", fmt_g(per, 8)) };
        out.push(format!("{:>lw$} |  Person-time   Failures {:>11}   [{}% conf. interval]", first, rate_head, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(59)));
        let mut rows_out: Vec<(String, f64, f64)> = Vec::new();
        match &at {
            Some(cuts) => {
                let mut edges = cuts.clone();
                edges.sort_by(f64::total_cmp);
                for w in edges.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    let mut y = 0.0;
                    let mut d = 0.0;
                    for o in &obs {
                        let lo = o.t0.max(a);
                        let hi = o.t.min(b);
                        if hi > lo {
                            y += (hi - lo) * o.w;
                        }
                        if o.d && o.t > a && o.t <= b {
                            d += o.w;
                        }
                    }
                    rows_out.push((format!("({}-{}]", fmt_g(a, 6), fmt_g(b, 6)), y, d));
                }
            }
            None => {
                if !vars.is_empty() {
                    for (_, l, rows) in groups(ds, &vars, &obs)? {
                        rows_out.push((level_text(&l), rows.iter().map(|o| (o.t - o.t0) * o.w).sum(), rows.iter().filter(|o| o.d).map(|o| o.w).sum()));
                    }
                }
            }
        }
        let total = ("Total".to_string(), obs.iter().map(|o| (o.t - o.t0) * o.w).sum::<f64>(), obs.iter().filter(|o| o.d).map(|o| o.w).sum::<f64>());
        let line = |(l, y, d): &(String, f64, f64)| {
            let (lo, hi) = crate::survival::poisson_ci_pub(*d, *y, level);
            format!("{:>lw$} | {:>12} {:>10} {:>11} {:>10} {:>10}", l, gc(*y, 10), fmt_g(*d, 9), fmt_g(d / y * per, 8), fmt_g(lo * per, 8), fmt_g(hi * per, 8))
        };
        for r in &rows_out {
            out.push(line(r));
        }
        if !rows_out.is_empty() {
            out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(59)));
        }
        out.push(line(&total));
        let (lo, hi) = crate::survival::poisson_ci_pub(total.2, total.1, level);
        let mut r = Results::new();
        r.set("ptime", total.1);
        r.set("failures", total.2);
        r.set("rate", total.2 / total.1 * per);
        r.set("lb", lo * per);
        r.set("ub", hi * per);
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Stdescribe;

impl CommandHandler for Stdescribe {
    fn name(&self) -> &'static str {
        "stdescribe"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let obs = st_obs(ds, st, cmd)?;
        let id = st_id(ds);
        // per-subject aggregates
        let idc = id.as_ref().map(|v| util::col(ds, v)).transpose()?;
        let mut subj: std::collections::BTreeMap<u64, Vec<StObs>> = Default::default();
        for (k, o) in obs.iter().enumerate() {
            let key = idc.as_ref().map_or(k as u64, |c| order_bits(c[o.row]));
            subj.entry(key).or_default().push(*o);
        }
        let mut nrec = Vec::new();
        let mut entry = Vec::new();
        let mut exit = Vec::new();
        let mut gap = Vec::new();
        let mut risk = Vec::new();
        let mut fail = Vec::new();
        for recs in subj.values_mut() {
            recs.sort_by(|a, b| a.t0.total_cmp(&b.t0));
            nrec.push(recs.len() as f64);
            entry.push(recs[0].t0);
            exit.push(recs.iter().map(|o| o.t).fold(f64::NEG_INFINITY, f64::max));
            let mut g = 0.0;
            for w in recs.windows(2) {
                if w[1].t0 > w[0].t {
                    g += w[1].t0 - w[0].t;
                }
            }
            gap.push(g);
            risk.push(recs.iter().map(|o| o.t - o.t0).sum::<f64>());
            fail.push(recs.iter().filter(|o| o.d).count() as f64);
        }
        let stats = |v: &[f64]| -> (f64, f64, f64, f64, f64) {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            let n = s.len();
            let med = if n == 0 { SYSMISS } else if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
            (s.iter().sum(), s.iter().sum::<f64>() / n as f64, s[0], med, s[n - 1])
        };
        if subj.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let mut out = header_or_not(ds, cmd);
        out.push(format!("{:<26}{:>36}", "", "|-------------- Per subject --------------|"));
        out.push(format!("{:<20}{:>10}{:>12}{:>12}{:>12}{:>12}", "Category", "Total", "Mean", "Min", "Median", "Max"));
        out.push("-".repeat(78));
        let full = |label: &str, v: &[f64], total: bool| {
            let (s, m, mn, md, mx) = stats(v);
            format!("{:<20}{:>10}{:>12}{:>12}{:>12}{:>12}", label, if total { gc(s, 9) } else { String::new() }, fmt_g(m, 8), fmt_g(mn, 8), fmt_g(md, 8), fmt_g(mx, 8))
        };
        let nsub = subj.len() as f64;
        out.push(format!("{:<20}{:>10}", "Number of subjects", gc(nsub, 9)));
        out.push(full("Number of records", &nrec, true));
        out.push(String::new());
        out.push(full("Entry time (first)", &entry, false));
        out.push(full("Exit time (final)", &exit, false));
        out.push(String::new());
        let ngap = gap.iter().filter(|g| **g > 0.0).count() as f64;
        out.push(format!("{:<20}{:>10}", "Subjects with gap", gc(ngap, 9)));
        if ngap > 0.0 {
            out.push(full("Time on gap", &gap, true));
        } else {
            out.push(format!("{:<20}{:>10}", "Time on gap", 0));
        }
        out.push(full("Time at risk", &risk, true));
        out.push(String::new());
        out.push(full("Failures", &fail, true));
        out.push("-".repeat(78));
        let mut r = Results::new();
        r.set("N_sub", nsub);
        r.set("N_total", obs.len() as f64);
        r.set("N_min", stats(&nrec).2);
        r.set("N_mean", stats(&nrec).1);
        r.set("N_med", stats(&nrec).3);
        r.set("N_max", stats(&nrec).4);
        r.set("t0_min", stats(&entry).2);
        r.set("t0_max", stats(&entry).4);
        r.set("t1_min", stats(&exit).2);
        r.set("t1_max", stats(&exit).4);
        r.set("N_gap", ngap);
        r.set("gap", gap.iter().sum());
        r.set("tr", stats(&risk).0);
        r.set("tr_mean", stats(&risk).1);
        r.set("N_fail", stats(&fail).0);
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// stsplit / stjoin
// ---------------------------------------------------------------------------

/// Rebuild the dataset from a list of source rows (rows may repeat).
fn take_rows(ds: &mut DataSet, idx: &[usize]) {
    for v in ds.vars.iter_mut() {
        v.data = idx.iter().map(|&i| v.data[i].clone()).collect();
    }
    ds.n_obs = idx.len();
    ds.sort_by.clear();
}

fn set_num(ds: &mut DataSet, var: &str, row: usize, x: f64) {
    if let Some(v) = ds.get_var_mut(var) {
        let ty = v.var_type;
        v.data[row] = if is_missing(x) { Value::Missing } else { util::typed_value(ty, x) };
        if !util::fits(ty, x) && !is_missing(x) {
            v.var_type = VarType::Double;
            v.data[row] = Value::Double(x);
        }
    }
}

/// Make a variable double so that split times are stored exactly.
fn promote_double(ds: &mut DataSet, var: &str) {
    if let Some(v) = ds.get_var_mut(var) {
        if !v.var_type.is_string() && v.var_type != VarType::Double {
            v.var_type = VarType::Double;
            v.data = v.data.iter().map(|x| if is_missing(x.num()) { x.clone() } else { Value::Double(x.num()) }).collect();
        }
    }
}

fn sort_id_t(ds: &mut DataSet, id: &str) -> Result<(), String> {
    ds.sort(&[id.to_string(), "_t".to_string()], true)?;
    ds.sort_by = vec![id.to_string(), "_t".to_string()];
    Ok(())
}

pub struct Stsplit;

impl CommandHandler for Stsplit {
    fn name(&self) -> &'static str {
        "stsplit"
    }

    /// stsplit newvar [if], {at(numlist) | every(#)}
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let id = st_id(ds).ok_or_else(|| rc_err(459, "stsplit requires that you have previously stset an id() variable"))?;
        let newvar = cmd.main.trim().to_string();
        varlist::check_name(&newvar)?;
        if ds.var_index.contains_key(&newvar) {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        let obs = st_obs(ds, st, cmd)?;
        let tmax = obs.iter().map(|o| o.t).fold(0.0, f64::max);
        let mut cuts: Vec<f64> = if let Some(a) = cmd.opt("at", 2) {
            crate::margins::numlist(a)?
        } else if let Some(e) = util::opt_num(cmd, "every", 2)? {
            if e <= 0.0 {
                return Err(rc_err(198, "every() must be positive"));
            }
            (1..).map(|k| k as f64 * e).take_while(|c| *c < tmax).collect()
        } else {
            return Err(rc_err(198, "at() or every() required"));
        };
        cuts.retain(|c| *c > 0.0 && !is_missing(*c));
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let bt = crate::survival::st_char(ds, "st_bt").unwrap_or_else(|| "_t".into());
        let bd = crate::survival::st_char(ds, "st_bd");
        let scale = crate::survival::st_char(ds, "st_s").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
        for v in ["_t", "_t0", bt.as_str()] {
            promote_double(ds, v);
        }
        let n = ds.n_obs;
        let btc = util::col(ds, &bt)?;
        let tc = util::col(ds, "_t")?;
        let mut idx: Vec<usize> = (0..n).collect();
        // (row in new data, t0, t, d, interval, base time)
        let mut pieces: Vec<(usize, f64, f64, Option<bool>, f64, f64)> = Vec::new();
        let in_st: std::collections::HashMap<usize, StObs> = obs.iter().map(|o| (o.row, *o)).collect();
        let mut added = 0usize;
        for i in 0..n {
            let Some(o) = in_st.get(&i) else { continue };
            let lower = |x: f64| cuts.iter().filter(|c| **c <= x).last().copied().unwrap_or(0.0);
            let inner: Vec<f64> = cuts.iter().copied().filter(|c| *c > o.t0 && *c < o.t).collect();
            // analysis time -> original time units
            let origin = btc[i] - tc[i] * scale;
            let mut a = o.t0;
            let mut bounds = inner.clone();
            bounds.push(o.t);
            for (k, &b) in bounds.iter().enumerate() {
                let last = k == bounds.len() - 1;
                let row = if k == 0 { i } else {
                    idx.push(i);
                    added += 1;
                    idx.len() - 1
                };
                pieces.push((row, a, b, if last { Some(o.d) } else { Some(false) }, lower(a), origin + b * scale));
                a = b;
            }
        }
        take_rows(ds, &idx);
        let mut nv = vec![SYSMISS; ds.n_obs];
        for &(row, t0, t, d, lo, base) in &pieces {
            set_num(ds, "_t0", row, t0);
            set_num(ds, "_t", row, t);
            set_num(ds, &bt, row, base);
            if let Some(d) = d {
                set_num(ds, "_d", row, d as u8 as f64);
                if !d {
                    if let Some(f) = &bd {
                        set_num(ds, f, row, SYSMISS);
                    }
                }
            }
            nv[row] = lo;
        }
        let mut v = Variable::new(&newvar, VarType::Float);
        v.data = nv.iter().map(|&x| if is_missing(x) { Value::Missing } else { Value::Float(x as f32) }).collect();
        ds.add_variable(v)?;
        // later stset calls rebuild _t0 from the previous record's end time
        sort_id_t(ds, &id)?;
        Ok(format!("({} observation{} (episode{}) created)", util::commas(added), if added == 1 { "" } else { "s" }, if added == 1 { "" } else { "s" }))
    }
}

pub struct Stjoin;

impl CommandHandler for Stjoin {
    fn name(&self) -> &'static str {
        "stjoin"
    }

    /// stjoin: join adjacent records of a subject that carry the same values
    /// of all other variables and no failure in between.
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let id = st_id(ds).ok_or_else(|| rc_err(459, "stjoin requires that you have previously stset an id() variable"))?;
        let _ = cmd;
        crate::survival::require_st(ds)?;
        sort_id_t(ds, &id)?;
        let bt = crate::survival::st_char(ds, "st_bt").unwrap_or_else(|| "_t".into());
        let bd = crate::survival::st_char(ds, "st_bd");
        let skip: Vec<String> = ["_t", "_t0", "_d", "_st", bt.as_str()].iter().map(|s| s.to_string()).chain(bd.clone()).collect();
        let others: Vec<usize> = ds.vars.iter().enumerate().filter(|(_, v)| !skip.contains(&v.name)).map(|(k, _)| k).collect();
        let (t, t0, d, s) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?, util::col(ds, "_st")?);
        let idc = util::col(ds, &id)?;
        let n = ds.n_obs;
        let mut keep = vec![true; n];
        // (kept row, its current end time and failure flag)
        let mut last: Option<(usize, f64, f64)> = None;
        let mut updates: Vec<(usize, usize)> = Vec::new(); // (kept row, absorbed row)
        for i in 0..n {
            if s[i] != 1.0 {
                last = None;
                continue;
            }
            if let Some((p, end, dp)) = last {
                let same = idc[p] == idc[i] && end == t0[i] && dp == 0.0 && others.iter().all(|&k| ds.vars[k].data[p] == ds.vars[k].data[i]);
                if same {
                    keep[i] = false;
                    updates.push((p, i));
                    last = Some((p, t[i], d[i]));
                    continue;
                }
            }
            last = Some((i, t[i], d[i]));
        }
        // the kept record takes the end time and outcome of the absorbed one
        for &(p, i) in &updates {
            for name in ["_t", "_d", bt.as_str()].iter().copied().chain(bd.as_deref()) {
                if let Some(v) = ds.get_var_mut(name) {
                    v.data[p] = v.data[i].clone();
                }
            }
        }
        let joined = updates.len();
        let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        take_rows(ds, &idx);
        Ok(format!("({} obs. eliminated)", util::commas(joined)))
    }
}
