//! `psmatch2`, `pstest`, `psgraph`: propensity-score matching in the style
//! of the community package by Leuven & Sianesi (2003), reimplemented from
//! its documentation and the methods literature (Rosenbaum & Rubin 1983,
//! 1985; Heckman, Ichimura & Todd 1997, 1998; Rubin 2001 for B and R).
//!
//!   psmatch2 tvar [indepvars] [if] [in] [, outcome(varlist) pscore(var)
//!       neighbor(#) radius caliper(#) kernel kerneltype(type) bwidth(#)
//!       common noreplacement descending ties logit ate]
//!
//! Creates _pscore _treated _support _weight _id _n1.._nk _nn _pdif and
//! _<outcome> (the matched counterfactual outcome of the treated).

use std::cell::RefCell;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::store::Store;
use crate::teffects::{self as te, OKind};
use crate::{util, varlist};

/// Replace or create a double variable.
pub(crate) fn put_var(ds: &mut DataSet, name: &str, vals: Vec<f64>) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    ds.add_variable(util::new_double(name, vals))
}

#[derive(Clone)]
struct PsState {
    vars: Vec<String>,
    logit: bool,
}

thread_local! {
    static PS: RefCell<Option<PsState>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, PartialEq)]
enum Kern {
    Epan,
    Normal,
    Biweight,
    Uniform,
    Tricube,
}

fn kern(k: Kern, u: f64) -> f64 {
    let a = u.abs();
    match k {
        Kern::Normal => (-0.5 * u * u).exp(),
        _ if a >= 1.0 => 0.0,
        Kern::Epan => 1.0 - u * u,
        Kern::Biweight => (1.0 - u * u).powi(2),
        Kern::Uniform => 1.0,
        Kern::Tricube => (1.0 - a * a * a).powi(3),
    }
}

/// Matching weights of `units` against `pool` on the score `p`.
/// Returns, for each unit, a list of (pool index, weight) summing to 1;
/// empty when the unit cannot be matched.
#[allow(clippy::too_many_arguments)]
fn match_units(units: &[usize], pool: &[usize], p: &[f64], k: usize, ties: bool, caliper: Option<f64>, radius: bool, kernel: Option<(Kern, f64)>, norepl: bool) -> Vec<Vec<(usize, f64)>> {
    let mut used = vec![false; p.len()];
    let mut sorted: Vec<usize> = pool.to_vec();
    sorted.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let sp: Vec<f64> = sorted.iter().map(|&j| p[j]).collect();
    let mut out = Vec::with_capacity(units.len());
    for &i in units {
        if let Some((kt, bw)) = kernel {
            let mut m: Vec<(usize, f64)> = Vec::new();
            for &j in pool {
                let wv = kern(kt, (p[j] - p[i]) / bw);
                if wv > 0.0 && caliper.is_none_or(|c| (p[j] - p[i]).abs() <= c) {
                    m.push((j, wv));
                }
            }
            let s: f64 = m.iter().map(|x| x.1).sum();
            m.iter_mut().for_each(|x| x.1 /= s);
            out.push(if s > 0.0 { m } else { Vec::new() });
            continue;
        }
        if radius {
            let c = caliper.unwrap_or(0.0);
            let m: Vec<usize> = pool.iter().copied().filter(|&j| (p[j] - p[i]).abs() <= c).collect();
            let nm = m.len() as f64;
            out.push(m.into_iter().map(|j| (j, 1.0 / nm)).collect());
            continue;
        }
        // nearest neighbours: expand outwards from the insertion point
        let pos = sp.partition_point(|&v| v < p[i]);
        let (mut lo, mut hi) = (pos as isize - 1, pos);
        let mut cand: Vec<(f64, usize)> = Vec::new();
        let need = k;
        loop {
            let dl = if lo >= 0 { Some(p[i] - sp[lo as usize]) } else { None };
            let dh = if hi < sp.len() { Some(sp[hi] - p[i]) } else { None };
            let take_low = match (dl, dh) {
                (None, None) => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(a), Some(b)) => a <= b,
            };
            let (d, j) = if take_low {
                lo -= 1;
                (dl.unwrap(), sorted[(lo + 1) as usize])
            } else {
                hi += 1;
                (dh.unwrap(), sorted[hi - 1])
            };
            if norepl && used[j] {
                continue;
            }
            let cnt = cand.len();
            if cnt >= need {
                let last = cand[cnt - 1].0;
                if !(ties && (d - last).abs() <= 1e-12 * last.abs().max(1e-300)) {
                    break;
                }
            }
            cand.push((d, j));
        }
        if let Some(c) = caliper {
            cand.retain(|x| x.0 <= c);
        }
        if norepl {
            cand.truncate(1);
            for x in &cand {
                used[x.1] = true;
            }
        }
        let nm = cand.len() as f64;
        out.push(cand.into_iter().map(|(_, j)| (j, 1.0 / nm)).collect());
    }
    out
}

fn wmean(x: &[f64], w: &[f64]) -> f64 {
    x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
}

fn wvar(x: &[f64], w: &[f64]) -> f64 {
    let m = wmean(x, w);
    let sw: f64 = w.iter().sum();
    let npos = w.iter().filter(|&&v| v > 0.0).count() as f64;
    if npos < 2.0 {
        return 0.0;
    }
    x.iter().zip(w).map(|(a, b)| b * (a - m).powi(2)).sum::<f64>() / sw * npos / (npos - 1.0)
}

pub struct Psmatch2;

impl CommandHandler for Psmatch2 {
    fn name(&self) -> &'static str {
        "psmatch2"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words: Vec<String> = cmd.main.split_whitespace().map(str::to_string).collect();
        let tvar = varlist::resolve_name(ds, words.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let xterms: Vec<String> = words[1..].to_vec();
        let outcomes = match cmd.opt("outcome", 3) {
            Some(o) => varlist::expand(ds, o)?,
            None => Vec::new(),
        };
        let logit = cmd.has_opt("logit", 5);
        let k = util::opt_num(cmd, "neighbor", 1)?.unwrap_or(1.0).max(1.0) as usize;
        let caliper = util::opt_num(cmd, "caliper", 3)?;
        let radius = cmd.has_opt("radius", 3);
        if radius && caliper.is_none() {
            return Err(rc_err(198, "radius requires caliper()"));
        }
        let kernel = if cmd.has_opt("kernel", 1) {
            let kt = match cmd.opt("kerneltype", 7).map(|s| s.trim().to_lowercase()).as_deref() {
                None | Some("epan") => Kern::Epan,
                Some("normal") => Kern::Normal,
                Some("biweight") => Kern::Biweight,
                Some("uniform") => Kern::Uniform,
                Some("tricube") => Kern::Tricube,
                Some(o) => return Err(rc_err(198, format!("kerneltype({o}) not allowed"))),
            };
            Some((kt, util::opt_num(cmd, "bwidth", 2)?.unwrap_or(0.06)))
        } else {
            None
        };
        let norepl = cmd.has_opt("noreplacement", 5);
        if norepl && k > 1 {
            return Err(rc_err(198, "noreplacement is allowed only with one neighbor"));
        }
        let ties = cmd.has_opt("ties", 4);
        let common = cmd.has_opt("common", 3);
        let ate = cmd.has_opt("ate", 3);

        // sample and propensity score
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = vec![tvar.clone()];
        need.extend(te::base_vars(ds, &xterms)?);
        if let Some(ps) = cmd.opt("pscore", 2) {
            need.push(varlist::resolve_name(ds, ps.trim())?);
        }
        util::markout(ds, &mut mask, &need);
        let mut out = Vec::new();
        let tcol = util::col(ds, &tvar)?;
        let pfull: Vec<f64> = if let Some(ps) = cmd.opt("pscore", 2) {
            if !xterms.is_empty() {
                return Err(rc_err(198, "pscore() and indepvars may not be combined"));
            }
            util::col(ds, &varlist::resolve_name(ds, ps.trim())?)?
        } else {
            if xterms.is_empty() {
                return Err(rc_err(198, "indepvars or pscore() required"));
            }
            let verb = if logit { "logit" } else { "probit" };
            let line = format!("{verb} {tvar} {}{}", xterms.join(" "), te::if_in_tail(cmd));
            let mc = parse_line(&line).map_err(|e| rc_err(198, e.message))?;
            let model = crate::models::BinaryModel { verb: if logit { "logit" } else { "probit" }, link: if logit { crate::models::Link::Logit } else { crate::models::Link::Probit } };
            let txt = model.execute(ds, st, &mc)?;
            if !cmd.has_opt("quietly", 1) {
                out.push(txt);
            }
            let d = te::sub_design(ds, st, cmd, &tvar, &xterms, false, vec![])?;
            let x = te::design_rows(&d);
            let link = if logit { te::Link::Logit } else { te::Link::Probit };
            let g = te::fit_binary(&x, &d.y.iter().map(|&v| if v != 0.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>(), link)?;
            let mut p = vec![SYSMISS; ds.n_obs];
            for (r, xi) in d.rows.iter().zip(&x) {
                p[*r] = te::cdf(link, te::dot(xi, &g));
            }
            p
        };
        for i in 0..ds.n_obs {
            if is_missing(pfull[i]) || is_missing(tcol[i]) {
                mask[i] = false;
            }
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let treated = |r: usize| tcol[r] != 0.0;
        let tr: Vec<usize> = rows.iter().copied().filter(|&r| treated(r)).collect();
        let co: Vec<usize> = rows.iter().copied().filter(|&r| !treated(r)).collect();
        if tr.is_empty() || co.is_empty() {
            return Err(rc_err(2000, "both treated and control observations are required"));
        }
        // common support
        let mut support = vec![SYSMISS; ds.n_obs];
        for &r in &rows {
            support[r] = 1.0;
        }
        if common {
            let (cmin, cmax) = co.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &r| (a.0.min(pfull[r]), a.1.max(pfull[r])));
            for &r in &tr {
                if pfull[r] < cmin || pfull[r] > cmax {
                    support[r] = 0.0;
                }
            }
            if ate {
                let (tmin, tmax) = tr.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &r| (a.0.min(pfull[r]), a.1.max(pfull[r])));
                for &r in &co {
                    if pfull[r] < tmin || pfull[r] > tmax {
                        support[r] = 0.0;
                    }
                }
            }
        }
        let mut units: Vec<usize> = tr.iter().copied().filter(|&r| support[r] == 1.0).collect();
        if cmd.has_opt("descending", 4) {
            units.sort_by(|&a, &b| pfull[b].total_cmp(&pfull[a]));
        }
        let pool: Vec<usize> = co.iter().copied().filter(|&r| support[r] == 1.0).collect();
        let m_att = match_units(&units, &pool, &pfull, k, ties, caliper, radius, kernel, norepl);
        let mut weight = vec![SYSMISS; ds.n_obs];
        let mut nn = vec![SYSMISS; ds.n_obs];
        let mut pdif = vec![SYSMISS; ds.n_obs];
        let maxk = m_att.iter().map(|m| m.len()).max().unwrap_or(0).min(if kernel.is_some() || radius { 0 } else { usize::MAX });
        let mut ncols = vec![vec![SYSMISS; ds.n_obs]; maxk];
        let id: Vec<f64> = (0..ds.n_obs).map(|i| (i + 1) as f64).collect();
        for (&i, m) in units.iter().zip(&m_att) {
            if m.is_empty() {
                support[i] = 0.0;
                continue;
            }
            weight[i] = 1.0;
            nn[i] = m.len() as f64;
            if kernel.is_none() && !radius {
                for (c, (j, _)) in m.iter().enumerate() {
                    ncols[c][i] = id[*j];
                }
                pdif[i] = (pfull[i] - pfull[m[0].0]).abs();
            }
            for &(j, wv) in m {
                weight[j] = if is_missing(weight[j]) { wv } else { weight[j] + wv };
            }
        }
        let on: Vec<usize> = units.iter().copied().zip(&m_att).filter(|(_, m)| !m.is_empty()).map(|(i, _)| i).collect();
        let m_on: Vec<&Vec<(usize, f64)>> = m_att.iter().filter(|m| !m.is_empty()).collect();
        // ATU: controls matched to the treated
        let m_atu = if ate {
            let tpool: Vec<usize> = tr.iter().copied().filter(|&r| support[r] == 1.0).collect();
            let cu: Vec<usize> = co.iter().copied().filter(|&r| support[r] == 1.0).collect();
            Some((cu.clone(), match_units(&cu, &tpool, &pfull, k, ties, caliper, radius, kernel, false)))
        } else {
            None
        };

        // results for each outcome
        let rule = format!("{}+{}", "-".repeat(28), "-".repeat(59));
        if !outcomes.is_empty() {
            out.push("-".repeat(88));
            out.push(format!("{:>16} {:>10} |{:>11}{:>13}{:>13}{:>13}{:>9}", "Variable", "Sample", "Treated", "Controls", "Difference", "S.E.", "T-stat"));
            out.push(rule.clone());
        }
        let r = st.begin_r();
        let mut first = true;
        let mut cf_vars = Vec::new();
        for ov in &outcomes {
            let y = util::col(ds, ov)?;
            let ok = |i: usize| !is_missing(y[i]);
            let y1: Vec<f64> = tr.iter().copied().filter(|&i| ok(i)).map(|i| y[i]).collect();
            let y0: Vec<f64> = co.iter().copied().filter(|&i| ok(i)).map(|i| y[i]).collect();
            let (m1, m0) = (util::mean(&y1), util::mean(&y0));
            let (n1, n0) = (y1.len() as f64, y0.len() as f64);
            let sp = (((n1 - 1.0) * util::variance(&y1) + (n0 - 1.0) * util::variance(&y0)) / (n1 + n0 - 2.0) * (1.0 / n1 + 1.0 / n0)).sqrt();
            // ATT
            let mut cf = vec![SYSMISS; ds.n_obs];
            let mut sum1 = 0.0;
            let mut sum0 = 0.0;
            let mut cnt = 0.0;
            let mut cw = std::collections::HashMap::<usize, f64>::new();
            let mut yt = Vec::new();
            for (&i, m) in on.iter().zip(&m_on) {
                if !ok(i) || m.iter().any(|(j, _)| !ok(*j)) {
                    continue;
                }
                let c: f64 = m.iter().map(|(j, w)| w * y[*j]).sum();
                cf[i] = c;
                sum1 += y[i];
                sum0 += c;
                cnt += 1.0;
                yt.push(y[i]);
                for &(j, w) in m.iter() {
                    *cw.entry(j).or_default() += w;
                }
            }
            let att = (sum1 - sum0) / cnt;
            let (cy, cwts): (Vec<f64>, Vec<f64>) = cw.iter().map(|(&j, &w)| (y[j], w)).unzip();
            let se_att = (util::variance(&yt) / cnt + wvar(&cy, &cwts) * cwts.iter().map(|w| w * w).sum::<f64>() / (cnt * cnt)).sqrt();
            let line = |name: &str, lab: &str, a: f64, b: f64, se: Option<f64>| -> String {
                let d = a - b;
                match se {
                    Some(s) => format!("{:>16} {:>10} |{:>11}{:>13}{:>13}{:>13}{:>9.2}", name, lab, fmt_g(a, 9), fmt_g(b, 9), fmt_g(d, 9), fmt_g(s, 9), d / s),
                    None => format!("{:>16} {:>10} |{:>11}{:>13}{:>13}{:>13}{:>9}", name, lab, fmt_g(a, 9), fmt_g(b, 9), fmt_g(d, 9), ".", "."),
                }
            };
            out.push(line(&crate::summarize::abbrev(ov, 16), "Unmatched", m1, m0, Some(sp)));
            let se_shown = if kernel.is_some() { None } else { Some(se_att) };
            out.push(line("", "ATT", sum1 / cnt, sum0 / cnt, se_shown));
            let mut atu = f64::NAN;
            if let Some((cu, mu)) = &m_atu {
                let mut s = 0.0;
                let mut c = 0.0;
                let mut s1 = 0.0;
                for (&i, m) in cu.iter().zip(mu) {
                    if m.is_empty() || !ok(i) || m.iter().any(|(j, _)| !ok(*j)) {
                        continue;
                    }
                    let v: f64 = m.iter().map(|(j, w)| w * y[*j]).sum();
                    s += v - y[i];
                    s1 += y[i];
                    c += 1.0;
                }
                atu = s / c;
                out.push(line("", "ATU", (s1 + s) / c, s1 / c, None));
                let ate_v = (att * cnt + atu * c) / (cnt + c);
                out.push(format!("{:>16} {:>10} |{:>11}{:>13}{:>13}{:>13}{:>9}", "", "ATE", "", "", fmt_g(ate_v, 9), ".", "."));
                r.set(&format!("atu_{ov}"), atu);
                r.set(&format!("ate_{ov}"), ate_v);
                if first {
                    r.set("atu", atu);
                    r.set("ate", ate_v);
                }
            }
            out.push(rule.clone());
            r.set(&format!("att_{ov}"), att);
            r.set(&format!("seatt_{ov}"), se_att);
            if first {
                r.set("att", att);
                r.set("seatt", se_att);
                first = false;
            }
            let _ = atu;
            cf_vars.push((format!("_{ov}"), cf));
        }
        if !outcomes.is_empty() {
            out.push("Note: S.E. does not take into account that the propensity score is estimated.".into());
        }
        // support table
        out.push(String::new());
        let cnt = |rs: &[usize], s: f64| rs.iter().filter(|&&r| support[r] == s).count();
        let off_t = cnt(&tr, 0.0);
        let off_c = cnt(&co, 0.0);
        out.push(format!("{:>10} | {:<20} |", "psmatch2:", " Common support"));
        out.push(format!("{:>10} | {:>9} {:>10} | {:>10}", "Treatment", "Off suppo", "On suppor", "Total"));
        out.push(format!("{}+{}+{}", "-".repeat(11), "-".repeat(22), "-".repeat(11)));
        out.push(format!("{:>10} | {:>9} {:>10} | {:>10}", "Untreated", util::commas(off_c), util::commas(co.len() - off_c), util::commas(co.len())));
        out.push(format!("{:>10} | {:>9} {:>10} | {:>10}", "Treated", util::commas(off_t), util::commas(tr.len() - off_t), util::commas(tr.len())));
        out.push(format!("{}+{}+{}", "-".repeat(11), "-".repeat(22), "-".repeat(11)));
        out.push(format!("{:>10} | {:>9} {:>10} | {:>10}", "Total", util::commas(off_c + off_t), util::commas(rows.len() - off_c - off_t), util::commas(rows.len())));

        // variables
        let treat_v: Vec<f64> = (0..ds.n_obs).map(|i| if mask[i] { if treated(i) { 1.0 } else { 0.0 } } else { SYSMISS }).collect();
        let ps_v: Vec<f64> = (0..ds.n_obs).map(|i| if mask[i] { pfull[i] } else { SYSMISS }).collect();
        put_var(ds, "_pscore", ps_v)?;
        put_var(ds, "_treated", treat_v)?;
        put_var(ds, "_support", support)?;
        put_var(ds, "_weight", weight)?;
        put_var(ds, "_id", id)?;
        let mut c = 1;
        while ds.get_var(&format!("_n{c}")).is_some() {
            ds.drop_variable(&format!("_n{c}"))?;
            c += 1;
        }
        for (c, col) in ncols.into_iter().enumerate() {
            put_var(ds, &format!("_n{}", c + 1), col)?;
        }
        put_var(ds, "_nn", nn)?;
        put_var(ds, "_pdif", pdif)?;
        for (name, v) in cf_vars {
            put_var(ds, &name, v)?;
        }
        if let Some(v) = ds.get_var_mut("_pscore") {
            v.label = "psmatch2: Propensity Score".into();
        }
        if let Some(v) = ds.get_var_mut("_weight") {
            v.label = "psmatch2: weight of matched controls".into();
        }
        let vars = te::base_vars(ds, &xterms)?;
        PS.with(|s| *s.borrow_mut() = Some(PsState { vars, logit }));
        st.data_changed = true;
        Ok(out.join("\n"))
    }
}

/// Weighted OLS of x on [t, 1]: t statistic and p-value of the t coefficient.
fn wttest(x: &[f64], t: &[f64], w: &[f64]) -> (f64, f64) {
    let n = x.iter().zip(w).filter(|(_, wi)| **wi > 0.0).count() as f64;
    let sw: f64 = w.iter().sum();
    let wn: Vec<f64> = w.iter().map(|v| v * n / sw).collect();
    let (mut s00, mut s01, mut s11, mut sy0, mut sy1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..x.len() {
        s00 += wn[i];
        s01 += wn[i] * t[i];
        s11 += wn[i] * t[i] * t[i];
        sy0 += wn[i] * x[i];
        sy1 += wn[i] * t[i] * x[i];
    }
    let det = s00 * s11 - s01 * s01;
    let b1 = (s00 * sy1 - s01 * sy0) / det;
    let b0 = (sy0 - s01 * b1) / s00;
    let rss: f64 = (0..x.len()).map(|i| wn[i] * (x[i] - b0 - b1 * t[i]).powi(2)).sum();
    let s2 = rss / (n - 2.0);
    let se = (s2 * s00 / det).sqrt();
    let tt = b1 / se;
    (tt, 2.0 * dist::ttail(n - 2.0, tt.abs()))
}

pub struct Pstest;

impl CommandHandler for Pstest {
    fn name(&self) -> &'static str {
        "pstest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let state = PS.with(|s| s.borrow().clone());
        let vars = if cmd.main.trim().is_empty() {
            state.as_ref().map(|s| s.vars.clone()).filter(|v| !v.is_empty()).ok_or_else(|| rc_err(198, "varlist required"))?
        } else {
            varlist::expand(ds, &cmd.main)?
        };
        let tv = cmd.opt("treated", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| "_treated".into());
        let wv = cmd.opt("mweight", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| "_weight".into());
        let sv = cmd.opt("support", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| "_support".into());
        let raw = cmd.has_opt("raw", 3);
        let both = cmd.has_opt("both", 4) || raw;
        let t = util::col(ds, &tv)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        for i in 0..ds.n_obs {
            if is_missing(t[i]) {
                mask[i] = false;
            }
        }
        let wcol = if raw { vec![1.0; ds.n_obs] } else { util::col(ds, &wv)? };
        let scol = if raw { vec![1.0; ds.n_obs] } else { util::col(ds, &sv).unwrap_or_else(|_| vec![1.0; ds.n_obs]) };
        let rows = util::rows(&mask);
        let mrows: Vec<usize> = rows.iter().copied().filter(|&i| !is_missing(wcol[i]) && wcol[i] > 0.0 && (t[i] == 0.0 || scol[i] == 1.0)).collect();
        let mut out = Vec::new();
        out.push("-".repeat(78));
        out.push(format!("{:<16}{:>9}|{:^28}|{:^15}|{:>8}", "", "", "Mean", "t-test", " V(T)/"));
        out.push(format!("{:<16}{:>9}| {:>8} {:>8} {:>8}  |{:>7} {:>6} |{:>8}", "Variable", "Sample", "Treated", "Control", "%bias", "t", "p>|t|", " V(C)"));
        out.push(format!("{}+{}+{}+{}", "-".repeat(25), "-".repeat(28), "-".repeat(15), "-".repeat(8)));
        let mut biases_u = Vec::new();
        let mut biases_m = Vec::new();
        for v in &vars {
            let x = util::col(ds, v)?;
            let binary = rows.iter().all(|&i| x[i] == 0.0 || x[i] == 1.0);
            let sub = |rs: &[usize], wts: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
                let xs: Vec<f64> = rs.iter().map(|&i| x[i]).collect();
                let ts: Vec<f64> = rs.iter().map(|&i| if t[i] != 0.0 { 1.0 } else { 0.0 }).collect();
                let ws: Vec<f64> = rs.iter().map(|&i| if wts { wcol[i] } else { 1.0 }).collect();
                (xs, ts, ws)
            };
            let stats = |xs: &[f64], ts: &[f64], ws: &[f64]| -> (f64, f64, f64, f64) {
                let w1: Vec<f64> = ts.iter().zip(ws).map(|(t, w)| t * w).collect();
                let w0: Vec<f64> = ts.iter().zip(ws).map(|(t, w)| (1.0 - t) * w).collect();
                (wmean(xs, &w1), wmean(xs, &w0), wvar(xs, &w1), wvar(xs, &w0))
            };
            let (xu, tu, wu) = sub(&rows, false);
            let (m1u, m0u, v1u, v0u) = stats(&xu, &tu, &wu);
            let sd = ((v1u + v0u) / 2.0).sqrt();
            let bu = 100.0 * (m1u - m0u) / sd;
            let (xm, tm, wm) = sub(&mrows, true);
            let (m1m, m0m, v1m, v0m) = stats(&xm, &tm, &wm);
            let bm = 100.0 * (m1m - m0m) / sd;
            let ratio = |a: f64, b: f64| if binary { ".".to_string() } else { format!("{:.2}", a / b) };
            if both {
                let (tt, pp) = wttest(&xu, &tu, &wu);
                out.push(format!("{:<16}{:>9}| {:>8} {:>8} {:>8.1}  |{:>7.2} {:>6.3} |{:>8}", crate::summarize::abbrev(v, 16), "U", fmt_g(m1u, 5), fmt_g(m0u, 5), bu, tt, pp, ratio(v1u, v0u)));
            }
            if !raw {
                let (tt, pp) = wttest(&xm, &tm, &wm);
                let red = 100.0 * (1.0 - (bm.abs() / bu.abs()));
                let lab = if both { "M".to_string() } else { String::new() };
                out.push(format!("{:<16}{:>9}| {:>8} {:>8} {:>8.1}  |{:>7.2} {:>6.3} |{:>8}", if both { String::new() } else { crate::summarize::abbrev(v, 16) }, lab, fmt_g(m1m, 5), fmt_g(m0m, 5), bm, tt, pp, ratio(v1m, v0m)));
                if both {
                    out.push(format!("{:<16}{:>9}| {:>28} |{:>15}|", "", "", format!("%reduct |bias| {:>6.1}", red), ""));
                }
            }
            biases_u.push(bu.abs());
            biases_m.push(bm.abs());
        }
        out.push("-".repeat(78));
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            util::pctile_sorted(&s, 50.0)
        };
        // summary: pseudo-R2 and LR of the score model in each sample, Rubin's B and R
        let logit = state.as_ref().is_none_or(|s| s.logit);
        let summary = |rs: &[usize], wts: bool| -> Result<(f64, f64, f64, f64, f64), String> {
            let xr: Vec<Vec<f64>> = rs.iter().map(|&i| vars.iter().map(|v| ds.get_var(v).unwrap().num(i)).chain(std::iter::once(1.0)).collect()).collect();
            let ty: Vec<f64> = rs.iter().map(|&i| if t[i] != 0.0 { 1.0 } else { 0.0 }).collect();
            let ww: Vec<f64> = rs.iter().map(|&i| if wts { wcol[i] } else { 1.0 }).collect();
            let kind = if logit { OKind::Logit } else { OKind::Probit };
            let b = te::fit_glm(&xr, &ty, &ww, kind)?;
            let link = if logit { te::Link::Logit } else { te::Link::Probit };
            let ll = |pr: &dyn Fn(usize) -> f64| -> f64 { (0..rs.len()).map(|i| ww[i] * if ty[i] == 1.0 { pr(i).ln() } else { (1.0 - pr(i)).ln() }).sum() };
            let ll1 = ll(&|i| te::cdf(link, te::dot(&xr[i], &b)).clamp(1e-300, 1.0 - 1e-16));
            let pbar = wmean(&ty, &ww);
            let ll0 = ll(&|_| pbar);
            let idx: Vec<f64> = xr.iter().map(|r| te::dot(r, &b)).collect();
            let w1: Vec<f64> = ty.iter().zip(&ww).map(|(t, w)| t * w).collect();
            let w0: Vec<f64> = ty.iter().zip(&ww).map(|(t, w)| (1.0 - t) * w).collect();
            let (v1, v0) = (wvar(&idx, &w1), wvar(&idx, &w0));
            let bb = 100.0 * (wmean(&idx, &w1) - wmean(&idx, &w0)).abs() / ((v1 + v0) / 2.0).sqrt();
            Ok((1.0 - ll1 / ll0, 2.0 * (ll1 - ll0), dist::chi2tail(vars.len() as f64, 2.0 * (ll1 - ll0)), bb, v1 / v0))
        };
        out.push(String::new());
        out.push("-".repeat(78));
        out.push(format!("{:>9} | {:>7} {:>9} {:>7} {:>9} {:>8} {:>7} {:>7}", "Sample", "Ps R2", "LR chi2", "p>chi2", "MeanBias", "MedBias", "B", "R"));
        out.push(format!("{}+{}", "-".repeat(10), "-".repeat(67)));
        let r = st.begin_r();
        let mut emit = |lab: &str, rs: &[usize], wts: bool, bias: &[f64]| -> Result<(), String> {
            let (r2, lr, p, b, rr) = summary(rs, wts)?;
            let mb = bias.iter().sum::<f64>() / bias.len() as f64;
            out.push(format!("{:>9} | {:>7.3} {:>9.2} {:>7.3} {:>9.1} {:>8.1} {:>7.1} {:>7.2}", lab, r2, lr, p, mb, med(bias), b, rr));
            let sfx = if lab == "Raw" || lab == "Unmatched" { "bef" } else { "aft" };
            r.set(&format!("r2{sfx}"), r2);
            r.set(&format!("chiprob{sfx}"), p);
            r.set(&format!("meanbias{sfx}"), mb);
            r.set(&format!("medbias{sfx}"), med(bias));
            r.set(&format!("B{sfx}"), b);
            r.set(&format!("R{sfx}"), rr);
            Ok(())
        };
        if both || raw {
            emit("Unmatched", &rows, false, &biases_u)?;
        }
        if !raw {
            emit("Matched", &mrows, true, &biases_m)?;
        }
        out.push("-".repeat(78));
        Ok(out.join("\n"))
    }
}

pub struct Psgraph;

impl CommandHandler for Psgraph {
    fn name(&self) -> &'static str {
        "psgraph"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let tv = cmd.opt("treated", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| "_treated".into());
        let pv = cmd.opt("pscore", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| "_pscore".into());
        let sv = cmd.opt("support", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| "_support".into());
        let t = util::col(ds, &tv)?;
        let p = util::col(ds, &pv)?;
        let s = util::col(ds, &sv).unwrap_or_else(|_| vec![1.0; ds.n_obs]);
        let bins = util::opt_num(cmd, "bin", 3)?.unwrap_or(20.0).max(1.0) as usize;
        let idx: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(p[i]) && !is_missing(t[i])).collect();
        let lo = idx.iter().map(|&i| p[i]).fold(f64::INFINITY, f64::min);
        let hi = idx.iter().map(|&i| p[i]).fold(f64::NEG_INFINITY, f64::max);
        let width = ((hi - lo) / bins as f64).max(1e-12);
        let hist = |f: &dyn Fn(usize) -> bool, sign: f64| -> Vec<(f64, f64)> {
            let mut c = vec![0.0; bins];
            for &i in &idx {
                if f(i) {
                    let b = (((p[i] - lo) / width) as usize).min(bins - 1);
                    c[b] += 1.0;
                }
            }
            c.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(b, v)| (lo + (b as f64 + 0.5) * width, sign * v)).collect()
        };
        let mk = |pts: Vec<(f64, f64)>, lab: &str, col: (u8, u8, u8)| Layer { kind: LayerKind::Bar, points: pts, label: lab.into(), width: width * 0.9, color: Some(col), ..Default::default() };
        let layers = vec![
            mk(hist(&|i| t[i] == 0.0, -1.0), "Untreated", (31, 119, 180)),
            mk(hist(&|i| t[i] != 0.0 && s[i] == 1.0, 1.0), "Treated: On support", (214, 39, 40)),
            mk(hist(&|i| t[i] != 0.0 && s[i] != 1.0, 1.0), "Treated: Off support", (44, 160, 44)),
        ];
        let mut g = Graph { xtitle: "Propensity Score".into(), layers, legend: true, ylines: vec![0.0], ..Default::default() };
        graph::finish(&mut g, cmd, st)
    }
}
