//! `cem` (coarsened exact matching) and `imb` (multivariate L1 imbalance),
//! after Iacus, King & Porro (2012, Political Analysis 20:1-24) and
//! Blackwell, Iacus, King & Porro (2009, Stata Journal 9:524-546);
//! reimplemented from the published method descriptions.
//!
//!   cem var1 [(#bins | cutpoints | rule)] var2 ... [if] [in], treatment(var)
//!       [showbreaks autocuts(sturges|fd|scott) k2k noimbal]
//!   imb varlist [if] [in], treatment(var) [useweights(var)]
//!
//! Creates cem_strata, cem_matched and cem_weights.

use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::psmatch2::put_var;
use crate::store::Store;
use crate::{util, varlist};

/// Automatic number of bins (or equal-width cutpoints) for one variable.
fn auto_breaks(x: &[f64], rule: &str) -> Vec<f64> {
    let n = x.len() as f64;
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo {
        return vec![lo, hi];
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let k = match rule {
        "scott" => {
            let h = 3.5 * util::variance(x).sqrt() * n.powf(-1.0 / 3.0);
            ((hi - lo) / h).ceil()
        }
        "fd" => {
            let iqr = util::pctile_sorted(&s, 75.0) - util::pctile_sorted(&s, 25.0);
            let h = 2.0 * iqr * n.powf(-1.0 / 3.0);
            if h > 0.0 { ((hi - lo) / h).ceil() } else { (n.log2() + 1.0).ceil() }
        }
        _ => (n.log2() + 1.0).ceil(),
    }
    .max(1.0) as usize;
    equal_breaks(lo, hi, k)
}

fn equal_breaks(lo: f64, hi: f64, k: usize) -> Vec<f64> {
    (0..=k).map(|j| lo + (hi - lo) * j as f64 / k as f64).collect()
}

/// Bin of `v` given increasing cutpoints (right-closed, lowest included).
fn bin_of(v: f64, br: &[f64]) -> i64 {
    if br.is_empty() {
        return v.to_bits() as i64;
    }
    br.iter().skip(1).take(br.len().saturating_sub(2)).filter(|&&c| v > c).count() as i64
}

/// Multivariate L1 distance between treated and (weighted) controls on the
/// cells defined by `breaks`.
fn l1(cols: &[Vec<f64>], t: &[bool], w: &[f64], breaks: &[Vec<f64>]) -> f64 {
    let mut cells: HashMap<Vec<i64>, (f64, f64)> = HashMap::new();
    let (mut s1, mut s0) = (0.0, 0.0);
    for i in 0..t.len() {
        if w[i] <= 0.0 {
            continue;
        }
        let key: Vec<i64> = cols.iter().zip(breaks).map(|(c, b)| bin_of(c[i], b)).collect();
        let e = cells.entry(key).or_default();
        if t[i] {
            e.0 += w[i];
            s1 += w[i];
        } else {
            e.1 += w[i];
            s0 += w[i];
        }
    }
    0.5 * cells.values().map(|(a, b)| (a / s1 - b / s0).abs()).sum::<f64>()
}

/// The imbalance report: multivariate L1 and univariate statistics.
fn imbalance(out: &mut Vec<String>, names: &[String], cols: &[Vec<f64>], t: &[bool], w: &[f64], st: &mut Store) {
    let breaks: Vec<Vec<f64>> = cols.iter().map(|c| auto_breaks(c, "scott")).collect();
    let ml1 = l1(cols, t, w, &breaks);
    out.push(String::new());
    out.push(format!("Multivariate L1 distance: {}", fmt_g(ml1, 8)));
    out.push(String::new());
    out.push("Univariate imbalance:".into());
    out.push(String::new());
    out.push(format!("{:>12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "L1", "mean", "min", "25%", "50%", "75%", "max"));
    let mut tab = crate::matrix::Matrix::zeros(names.len(), 7);
    for (k, (name, c)) in names.iter().zip(cols).enumerate() {
        let u1 = l1(std::slice::from_ref(c), t, w, std::slice::from_ref(&breaks[k]));
        let mut p1: Vec<(f64, f64)> = (0..t.len()).filter(|&i| t[i] && w[i] > 0.0).map(|i| (c[i], w[i])).collect();
        let mut p0: Vec<(f64, f64)> = (0..t.len()).filter(|&i| !t[i] && w[i] > 0.0).map(|i| (c[i], w[i])).collect();
        let wm = |p: &[(f64, f64)]| p.iter().map(|x| x.0 * x.1).sum::<f64>() / p.iter().map(|x| x.1).sum::<f64>();
        let mean = wm(&p1) - wm(&p0);
        let mn = |p: &[(f64, f64)]| p.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
        let mx = |p: &[(f64, f64)]| p.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
        let q = |p: &mut Vec<(f64, f64)>, pc: f64| util::pctile_weighted(p, pc);
        let vals = [u1, mean, mn(&p1) - mn(&p0), q(&mut p1, 25.0) - q(&mut p0, 25.0), q(&mut p1, 50.0) - q(&mut p0, 50.0), q(&mut p1, 75.0) - q(&mut p0, 75.0), mx(&p1) - mx(&p0)];
        for (j, v) in vals.iter().enumerate() {
            tab[(k, j)] = *v;
        }
        out.push(format!("{:>12} {}", crate::summarize::abbrev(name, 12), vals.iter().map(|v| format!("{:>9}", fmt_g(*v, 5))).collect::<Vec<_>>().join(" ")));
    }
    let tab = tab.with_names(names.to_vec(), ["L1", "mean", "min", "25%", "50%", "75%", "max"].iter().map(|s| s.to_string()).collect());
    st.r.set("L1", ml1);
    st.r.set_mat("imbal", tab);
}

pub struct Cem;

impl CommandHandler for Cem {
    fn name(&self) -> &'static str {
        "cem"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let tvar = varlist::resolve_name(ds, cmd.opt("treatment", 2).ok_or_else(|| rc_err(198, "option treatment() required"))?.trim())?;
        let rule = cmd.opt("autocuts", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "sturges".into());
        // variables with optional coarsening specifications
        let mut specs: Vec<(String, Option<String>)> = Vec::new();
        for w in split_top_level_words(&cmd.main) {
            if let Some(inner) = w.strip_prefix('(') {
                let inner = inner.trim_end_matches(')').trim().to_string();
                let last = specs.last_mut().ok_or_else(|| rc_err(198, "cutpoints must follow a variable"))?;
                last.1 = Some(inner);
            } else if let Some(i) = w.find('(') {
                let name = &w[..i];
                for v in varlist::expand(ds, name)? {
                    specs.push((v, Some(w[i + 1..].trim_end_matches(')').to_string())));
                }
            } else {
                for v in varlist::expand(ds, &w)? {
                    specs.push((v, None));
                }
            }
        }
        if specs.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let names: Vec<String> = specs.iter().map(|s| s.0.clone()).collect();
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = names.clone();
        need.push(tvar.clone());
        util::markout(ds, &mut mask, &need);
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let tcol = util::col(ds, &tvar)?;
        let t: Vec<bool> = rows.iter().map(|&r| tcol[r] != 0.0).collect();
        let cols: Vec<Vec<f64>> = names.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&r| c[r]).collect())).collect::<Result<_, _>>()?;
        let mut breaks = Vec::new();
        for ((name, spec), c) in specs.iter().zip(&cols) {
            let b = match spec.as_deref().map(str::trim) {
                None => auto_breaks(c, &rule),
                Some(s) if s.starts_with('#') => {
                    let k: usize = s[1..].trim().parse().map_err(|_| rc_err(198, format!("invalid bin count for {name}")))?;
                    if k == 0 {
                        Vec::new()
                    } else {
                        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        equal_breaks(lo, hi, k)
                    }
                }
                Some(s) if s.chars().next().is_some_and(|ch| ch.is_ascii_alphabetic()) => auto_breaks(c, s),
                Some(s) => {
                    let mut v = crate::margins::numlist(s)?;
                    v.sort_by(f64::total_cmp);
                    // user cutpoints are interior; add the range ends
                    let lo = c.iter().copied().fold(f64::INFINITY, f64::min).min(v[0]);
                    let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(*v.last().unwrap());
                    let mut b = vec![lo];
                    b.extend(v.into_iter().filter(|x| *x > lo && *x < hi));
                    b.push(hi);
                    b
                }
            };
            breaks.push(b);
        }
        let n = rows.len();
        let keys: Vec<Vec<i64>> = (0..n).map(|i| cols.iter().zip(&breaks).map(|(c, b)| bin_of(c[i], b)).collect()).collect();
        let mut strata: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut sid = vec![0usize; n];
        for i in 0..n {
            let next = strata.len() + 1;
            sid[i] = *strata.entry(keys[i].clone()).or_insert(next);
        }
        let ns = strata.len();
        let mut cnt = vec![(0.0, 0.0); ns + 1];
        for i in 0..n {
            if t[i] {
                cnt[sid[i]].0 += 1.0;
            } else {
                cnt[sid[i]].1 += 1.0;
            }
        }
        let matched: Vec<bool> = (0..n).map(|i| cnt[sid[i]].0 > 0.0 && cnt[sid[i]].1 > 0.0).collect();
        let mut wts = vec![0.0; n];
        if cmd.has_opt("k2k", 3) {
            // one-to-one within strata: greedy closest pairs (Euclidean on standardized covariates)
            let sd: Vec<f64> = cols.iter().map(|c| util::variance(c).sqrt().max(1e-300)).collect();
            let mut by: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
            for i in (0..n).filter(|&i| matched[i]) {
                let e = by.entry(sid[i]).or_default();
                if t[i] { e.0.push(i) } else { e.1.push(i) }
            }
            for (_, (tr, co)) in by {
                let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
                for &a in &tr {
                    for &b in &co {
                        let d: f64 = cols.iter().zip(&sd).map(|(c, s)| ((c[a] - c[b]) / s).powi(2)).sum();
                        pairs.push((d, a, b));
                    }
                }
                pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
                let mut used = std::collections::HashSet::new();
                for (_, a, b) in pairs {
                    if !used.contains(&a) && !used.contains(&b) {
                        used.insert(a);
                        used.insert(b);
                        wts[a] = 1.0;
                        wts[b] = 1.0;
                    }
                }
            }
        } else {
            let mt = (0..n).filter(|&i| matched[i] && t[i]).count() as f64;
            let mc = (0..n).filter(|&i| matched[i] && !t[i]).count() as f64;
            for i in 0..n {
                if matched[i] {
                    wts[i] = if t[i] { 1.0 } else { (mc / mt) * cnt[sid[i]].0 / cnt[sid[i]].1 };
                }
            }
        }
        let mstrata = (1..=ns).filter(|&s| cnt[s].0 > 0.0 && cnt[s].1 > 0.0).count();
        let mut out = Vec::new();
        if cmd.has_opt("showbreaks", 4) {
            out.push("Cutpoints:".into());
            for (name, b) in names.iter().zip(&breaks) {
                out.push(format!("{:>12}: {}", name, if b.is_empty() { "(none)".into() } else { b.iter().map(|v| fmt_g(*v, 6).trim().to_string()).collect::<Vec<_>>().join(" ") }));
            }
            out.push(String::new());
        }
        out.push("Matching Summary:".into());
        out.push("-----------------".into());
        out.push(format!("Number of strata: {ns}"));
        out.push(format!("Number of matched strata: {mstrata}"));
        out.push(String::new());
        let c01 = |f: &dyn Fn(usize) -> bool, tv: bool| (0..n).filter(|&i| t[i] == tv && f(i)).count();
        let k2k = cmd.has_opt("k2k", 3);
        let m_of = |i: usize| if k2k { wts[i] > 0.0 } else { matched[i] };
        out.push(format!("{:>10} {:>7} {:>7}", "", "0", "1"));
        out.push(format!("{:>10} {:>7} {:>7}", "All", c01(&|_| true, false), c01(&|_| true, true)));
        out.push(format!("{:>10} {:>7} {:>7}", "Matched", c01(&m_of, false), c01(&m_of, true)));
        out.push(format!("{:>10} {:>7} {:>7}", "Unmatched", c01(&|i| !m_of(i), false), c01(&|i| !m_of(i), true)));
        out.push(String::new());
        st.begin_r();
        if !cmd.has_opt("noimbal", 5) {
            imbalance(&mut out, &names, &cols, &t, &wts, st);
        }
        st.r.set("n_strata", ns as f64);
        st.r.set("n_mstrata", mstrata as f64);
        st.r.set("n_matched_t", c01(&m_of, true) as f64);
        st.r.set("n_matched_c", c01(&m_of, false) as f64);
        let mut sv = vec![SYSMISS; ds.n_obs];
        let mut mv = vec![SYSMISS; ds.n_obs];
        let mut wv = vec![SYSMISS; ds.n_obs];
        for (k, &r) in rows.iter().enumerate() {
            sv[r] = sid[k] as f64;
            mv[r] = if m_of(k) { 1.0 } else { 0.0 };
            wv[r] = wts[k];
        }
        put_var(ds, "cem_strata", sv)?;
        put_var(ds, "cem_matched", mv)?;
        put_var(ds, "cem_weights", wv)?;
        st.data_changed = true;
        Ok(out.join("\n"))
    }
}

pub struct Imb;

impl CommandHandler for Imb {
    fn name(&self) -> &'static str {
        "imb"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let tvar = varlist::resolve_name(ds, cmd.opt("treatment", 2).ok_or_else(|| rc_err(198, "option treatment() required"))?.trim())?;
        let names = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = names.clone();
        need.push(tvar.clone());
        util::markout(ds, &mut mask, &need);
        let wcol = match cmd.opt("useweights", 4) {
            Some(w) => Some(util::col(ds, &varlist::resolve_name(ds, w.trim())?)?),
            None => None,
        };
        let rows: Vec<usize> = util::rows(&mask).into_iter().filter(|&r| wcol.as_ref().is_none_or(|w| !is_missing(w[r]))).collect();
        let tcol = util::col(ds, &tvar)?;
        let t: Vec<bool> = rows.iter().map(|&r| tcol[r] != 0.0).collect();
        let w: Vec<f64> = rows.iter().map(|&r| wcol.as_ref().map_or(1.0, |c| c[r])).collect();
        let cols: Vec<Vec<f64>> = names.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&r| c[r]).collect())).collect::<Result<_, _>>()?;
        let mut out = Vec::new();
        st.begin_r();
        imbalance(&mut out, &names, &cols, &t, &w, st);
        Ok(out.join("\n"))
    }
}
