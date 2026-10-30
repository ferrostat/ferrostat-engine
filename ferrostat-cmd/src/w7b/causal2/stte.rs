//! Treatment effects on survival time after `stset`.
//!
//!   stteffects ra    (omvars [, weibull|exponential noconstant]) (tvar) [if] [in] [, ate|atet|pomeans control(#) tlevel(#)]
//!   stteffects ipw   (tvar tmvars [, logit|probit]) (cmvars [, weibull|exponential]) [...]
//!   stteffects ipwra (omvars [, omodel]) (tvar tmvars [, logit|probit]) [(cmvars [, cmodel])] [...]
//!   stteffects wra   (omvars [, omodel]) (tvar) (cmvars [, cmodel]) [...]
//!
//! The effects are contrasts of mean survival times. Outcome models are
//! Weibull (or exponential) proportional-hazards regressions fitted per
//! treatment level; the mean time of an observation under level l is
//! Gamma(1 + 1/p) exp(-x'b/p). `ra` averages the predicted means; `ipw`
//! weights the observed failure times by the inverse of the treatment
//! probability times the probability of remaining uncensored (from a
//! Weibull model of the censoring time); `ipwra` fits the outcome models
//! with inverse treatment-probability weights (and, with a censoring model,
//! on failures weighted by the inverse censoring survival); `wra` fits the
//! outcome models on failures weighted by the inverse censoring survival.
//! Robust standard errors stack all estimating equations. Delayed entry is
//! not supported; `ipw`/`ipwra` need a binary treatment.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::{dot, fit, fmtv, groups, levels, sandwich, score, solve_block};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::store::{Results, Store};
use crate::teffects::{base_vars, design_rows, effects_table, level_label, model, sub_design, wrap12, Block};
use crate::util;
use crate::varlist;
use crate::w7b::lasso::core::Family;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    vec![("stteffects", Box::new(Stteffects))]
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Ra,
    Ipw,
    Ipwra,
    Wra,
}

/// Weibull log likelihood and its derivatives with respect to the index
/// and ln p (`exp_only`: p fixed at 1).
fn wb(t: f64, d: f64, xb: f64, a: f64) -> (f64, f64, f64) {
    let p = a.exp();
    let lt = t.max(1e-300).ln();
    let h = (xb + p * lt).min(700.0).exp();
    let ll = d * (xb + a + (p - 1.0) * lt) - h;
    (ll, d - h, d * (1.0 + p * lt) - h * p * lt)
}

/// Weighted Weibull ML on rows `x` (index) with ln p free unless `expo`.
fn fit_weibull(x: &[Vec<f64>], t: &[f64], d: &[f64], w: &[f64], expo: bool) -> Result<(Vec<f64>, f64), String> {
    let k = x[0].len();
    let np = k + usize::from(!expo);
    let sw: f64 = w.iter().sum();
    if sw <= 0.0 {
        return Err(rc_err(2000, "no observations for an outcome model"));
    }
    let sd: f64 = (0..t.len()).map(|i| w[i] * d[i]).sum();
    let st: f64 = (0..t.len()).map(|i| w[i] * t[i]).sum();
    if sd <= 0.0 {
        return Err(rc_err(459, "no failures for an outcome model"));
    }
    let mut th = vec![0.0; np];
    if let Some(c) = (0..k).find(|&a| x.iter().zip(w).all(|(r, &wi)| wi == 0.0 || r[a] == 1.0)) {
        th[c] = (sd / st).ln();
    }
    let ll_of = |th: &[f64]| -> f64 {
        let a = if expo { 0.0 } else { th[k] };
        (0..t.len()).filter(|&i| w[i] != 0.0).map(|i| w[i] * wb(t[i], d[i], dot(&x[i], &th[..k]), a).0).sum()
    };
    let grad = |th: &[f64]| -> Vec<f64> {
        let a = if expo { 0.0 } else { th[k] };
        let mut g = vec![0.0; np];
        for i in 0..t.len() {
            if w[i] == 0.0 {
                continue;
            }
            let (_, s1, s2) = wb(t[i], d[i], dot(&x[i], &th[..k]), a);
            for c in 0..k {
                g[c] += w[i] * s1 * x[i][c];
            }
            if !expo {
                g[k] += w[i] * s2;
            }
        }
        g
    };
    let mut ll = ll_of(&th);
    for _ in 0..200 {
        let g = grad(&th);
        let h = crate::choice_mixed::num_hess(&|v: &[f64]| grad(v), &th);
        let neg: Vec<Vec<f64>> = (0..np).map(|r| (0..np).map(|c| -h[r][c]).collect()).collect();
        let step = crate::teffects::solve(&neg, &g).ok_or_else(|| rc_err(459, "outcome model: collinear covariates"))?;
        let mut lam = 1.0;
        let mut moved = false;
        for _ in 0..40 {
            let nt: Vec<f64> = th.iter().zip(&step).map(|(a, s)| a + lam * s).collect();
            let nl = ll_of(&nt);
            if nl.is_finite() && nl >= ll - 1e-12 * ll.abs().max(1.0) {
                let ch = (nl - ll).abs();
                th = nt;
                ll = nl;
                moved = ch > 1e-12 * ll.abs().max(1.0);
                break;
            }
            lam /= 2.0;
        }
        if !moved && step.iter().map(|v| v.abs()).fold(0.0, f64::max) * lam < 1e-9 {
            break;
        }
        if !moved {
            break;
        }
    }
    let a = if expo { 0.0 } else { th[k] };
    Ok((th[..k].to_vec(), a))
}

fn wb_mean(xb: f64, a: f64) -> f64 {
    let p = a.exp();
    dist::lngamma(1.0 + 1.0 / p).exp() * (-xb / p).exp()
}

fn wb_surv(t: f64, xb: f64, a: f64) -> f64 {
    (-(xb + a.exp() * t.max(1e-300).ln()).min(700.0).exp()).exp()
}

pub struct Stteffects;

impl CommandHandler for Stteffects {
    fn name(&self) -> &'static str {
        "stteffects"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("stteffects") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        if !ds.var_index.contains_key("_t") || !ds.var_index.contains_key("_d") {
            return Err(rc_err(119, "data not st; use stset"));
        }
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let method = match sub {
            "ra" => Method::Ra,
            "ipw" => Method::Ipw,
            "ipwra" => Method::Ipwra,
            "wra" => Method::Wra,
            _ => return Err(rc_err(198, format!("stteffects {sub}: unknown estimator"))),
        };
        let g = groups(rest);
        // parse groups by method
        let (ogrp, tgrp, cgrp): (Option<&String>, &String, Option<&String>) = match method {
            Method::Ra if g.len() == 2 => (Some(&g[0]), &g[1], None),
            Method::Ipw if g.len() == 2 => (None, &g[0], Some(&g[1])),
            Method::Ipwra if g.len() == 2 || g.len() == 3 => (Some(&g[0]), &g[1], g.get(2)),
            Method::Wra if g.len() == 3 => (Some(&g[0]), &g[1], Some(&g[2])),
            _ => return Err(rc_err(198, format!("stteffects {sub}: wrong number of models"))),
        };
        let (ox, oopts) = ogrp.map(|s| model(s)).unwrap_or_default();
        let (tm, topts) = model(tgrp);
        let (cx, copts) = cgrp.map(|s| model(s)).unwrap_or_default();
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let tx = tm[1..].to_vec();
        if matches!(method, Method::Ra | Method::Wra) && !tx.is_empty() {
            return Err(rc_err(198, "the treatment model takes no covariates with this estimator"));
        }
        let oexpo = oopts.contains("exp");
        let cexpo = copts.contains("exp");
        if oopts.contains("lnormal") || oopts.contains("gamma") || copts.contains("lnormal") || copts.contains("gamma") {
            return Err(rc_err(198, "only weibull and exponential models are available"));
        }
        let tlink = if topts.contains("probit") { Family::Probit } else { Family::Logit };
        let stat = crate::teffects::parse_stat(cmd);
        let atet = stat == crate::teffects::Stat::Atet;
        let level = util::level(cmd, st)?;
        // sample
        let mut all = vec!["_t".to_string(), "_d".to_string(), tvar.clone()];
        all.extend(base_vars(ds, &ox)?);
        all.extend(base_vars(ds, &tx)?);
        all.extend(base_vars(ds, &cx)?);
        let od = sub_design(ds, st, cmd, "_t", &ox, oopts.contains("nocons"), all.clone())?;
        let td = sub_design(ds, st, cmd, &tvar, &tx, topts.contains("nocons"), all.clone())?;
        let cd = sub_design(ds, st, cmd, "_t", &cx, copts.contains("nocons"), all)?;
        if od.rows != td.rows || od.rows != cd.rows {
            return Err(rc_err(459, "models use different samples"));
        }
        let stcol = if ds.var_index.contains_key("_st") { Some(util::col(ds, "_st")?) } else { None };
        let t0col = if ds.var_index.contains_key("_t0") { Some(util::col(ds, "_t0")?) } else { None };
        let dcol = util::col(ds, "_d")?;
        let keep: Vec<bool> = od.rows.iter().map(|&r| stcol.as_ref().is_none_or(|c| c[r] == 1.0)).collect();
        let pick = |d: &crate::est::Design| crate::xmodels::subset(d, &keep);
        let (od, td, cd) = (pick(&od), pick(&td), pick(&cd));
        let rows = od.rows.clone();
        let n = rows.len();
        if n < 5 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        if let Some(t0) = &t0col {
            if rows.iter().any(|&r| !is_missing(t0[r]) && t0[r] > 0.0) {
                return Err(rc_err(198, "delayed entry is not supported"));
            }
        }
        let tt = od.y.clone();
        let dd: Vec<f64> = rows.iter().map(|&r| if dcol[r] != 0.0 { 1.0 } else { 0.0 }).collect();
        let traw = td.y.clone();
        let lv = levels(&traw, &tvar)?;
        let ctrl = util::opt_num(cmd, "control", 4)?.unwrap_or(lv[0]);
        let ci = lv.iter().position(|v| *v == ctrl).ok_or_else(|| rc_err(459, "control() is not a treatment level"))?;
        let mut order = vec![ci];
        order.extend((0..lv.len()).filter(|&l| l != ci));
        let lvals: Vec<f64> = order.iter().map(|&o| lv[o]).collect();
        let nl = lvals.len();
        let has_t = matches!(method, Method::Ipw | Method::Ipwra);
        if has_t && nl != 2 {
            return Err(rc_err(198, "ipw and ipwra require a binary treatment"));
        }
        let trl = match util::opt_num(cmd, "tlevel", 4)? {
            Some(v) => lvals.iter().position(|x| *x == v).filter(|&p| p > 0).ok_or_else(|| rc_err(459, "tlevel() is not a treatment level"))?,
            None => nl - 1,
        };
        let lev: Vec<usize> = traw.iter().map(|v| lvals.iter().position(|l| l == v).unwrap()).collect();
        let has_o = method != Method::Ipw;
        let has_c = cgrp.is_some();
        let cw_mode = method == Method::Wra || (method == Method::Ipwra && has_c);
        let xo = design_rows(&od);
        let xt = design_rows(&td);
        let xc = design_rows(&cd);
        let (ko, kt, kc) = (xo[0].len(), xt[0].len(), xc[0].len());
        let kop = ko + usize::from(!oexpo);
        let kcp = kc + usize::from(!cexpo);
        // layout: outcome (per level) | treatment | censoring | POMs
        let og = if has_o { nl * kop } else { 0 };
        let oc = og + if has_t { kt } else { 0 };
        let op = oc + if has_c { kcp } else { 0 };
        let np = op + nl;
        let mut theta = vec![0.0; np];
        let ones = vec![1.0; n];
        if has_t {
            let t01: Vec<f64> = lev.iter().map(|&l| l as f64).collect();
            let gm = fit(&xt, &t01, &ones, tlink)?;
            theta[og..og + kt].copy_from_slice(&gm);
        }
        if has_c {
            let dc: Vec<f64> = dd.iter().map(|d| 1.0 - d).collect();
            let (b, a) = fit_weibull(&xc, &tt, &dc, &ones, cexpo)?;
            theta[oc..oc + kc].copy_from_slice(&b);
            if !cexpo {
                theta[oc + kc] = a;
            }
        }
        let ptreat = |th: &[f64], i: usize| -> f64 { if has_t { tlink.mean(dot(&xt[i], &th[og..og + kt])).clamp(1e-12, 1.0 - 1e-12) } else { 0.5 } };
        let sc = |th: &[f64], i: usize| -> f64 {
            if !has_c {
                return 1.0;
            }
            let a = if cexpo { 0.0 } else { th[oc + kc] };
            wb_surv(tt[i], dot(&xc[i], &th[oc..oc + kc]), a).max(1e-10)
        };
        let ow = |th: &[f64], i: usize| -> f64 {
            // outcome-model weight of observation i (own level)
            let mut w = 1.0;
            if method == Method::Ipwra {
                let p1 = ptreat(th, i);
                let pl = if lev[i] == 1 { p1 } else { 1.0 - p1 };
                let ptr = if trl == 1 { p1 } else { 1.0 - p1 };
                w *= if atet { ptr / pl } else { 1.0 / pl };
            }
            if cw_mode {
                w *= if dd[i] == 1.0 { 1.0 / sc(th, i) } else { 0.0 };
            }
            w
        };
        if has_o {
            for l in 0..nl {
                let w: Vec<f64> = (0..n).map(|i| if lev[i] == l { ow(&theta, i) } else { 0.0 }).collect();
                let dl: Vec<f64> = if cw_mode { vec![1.0; n] } else { dd.clone() };
                let (b, a) = fit_weibull(&xo, &tt, &dl, &w, oexpo)?;
                theta[l * kop..l * kop + ko].copy_from_slice(&b);
                if !oexpo {
                    theta[l * kop + ko] = a;
                }
            }
        }
        let moments = |th: &[f64], i: usize, m: &mut [f64]| {
            if has_o {
                let l = lev[i];
                let w = ow(th, i);
                if w != 0.0 {
                    let a = if oexpo { 0.0 } else { th[l * kop + ko] };
                    let d = if cw_mode { 1.0 } else { dd[i] };
                    let (_, s1, s2) = wb(tt[i], d, dot(&xo[i], &th[l * kop..l * kop + ko]), a);
                    for c in 0..ko {
                        m[l * kop + c] = w * s1 * xo[i][c];
                    }
                    if !oexpo {
                        m[l * kop + ko] = w * s2;
                    }
                }
            }
            if has_t {
                let s = score(tlink, lev[i] as f64, dot(&xt[i], &th[og..og + kt]));
                for c in 0..kt {
                    m[og + c] = s * xt[i][c];
                }
            }
            if has_c {
                let a = if cexpo { 0.0 } else { th[oc + kc] };
                let (_, s1, s2) = wb(tt[i], 1.0 - dd[i], dot(&xc[i], &th[oc..oc + kc]), a);
                for c in 0..kc {
                    m[oc + c] = s1 * xc[i][c];
                }
                if !cexpo {
                    m[oc + kc] = s2;
                }
            }
            for l in 0..nl {
                let pom = th[op + l];
                m[op + l] = if has_o {
                    let a = if oexpo { 0.0 } else { th[l * kop + ko] };
                    let mu = wb_mean(dot(&xo[i], &th[l * kop..l * kop + ko]), a);
                    let s = if atet { if lev[i] == trl { 1.0 } else { 0.0 } } else { 1.0 };
                    s * (mu - pom)
                } else {
                    let p1 = ptreat(th, i);
                    let pl = if l == 1 { p1 } else { 1.0 - p1 };
                    let ptr = if trl == 1 { p1 } else { 1.0 - p1 };
                    let ind = if lev[i] == l { 1.0 } else { 0.0 };
                    let w = if atet { ptr / pl } else { 1.0 / pl };
                    ind * dd[i] * w / sc(th, i) * (tt[i] - pom)
                };
            }
        };
        solve_block(&mut theta, n, np, op, &moments)?;
        let v = sandwich(&theta, n, np, &moments)?;
        let eff = if atet { "ATET" } else { "ATE" };
        let mut names = Vec::new();
        let mut cm = Vec::new();
        let mut blocks = Vec::new();
        let lab = |l: usize| level_label(ds, &tvar, lvals[l]);
        if stat == crate::teffects::Stat::Pomeans {
            let mut bl = Block { title: "POmeans".into(), rows: Vec::new() };
            for l in 0..nl {
                let mut c = vec![0.0; nl];
                c[l] = 1.0;
                cm.push(c);
                names.push(format!("POmeans:{}.{tvar}", fmtv(lvals[l])));
                bl.rows.push((vec![crate::summarize::abbrev(&lab(l), 12)], 0.0, 0.0));
            }
            blocks.push(bl);
        } else {
            let mut bl = Block { title: eff.into(), rows: Vec::new() };
            for l in 1..nl {
                let mut c = vec![0.0; nl];
                c[l] = 1.0;
                c[0] = -1.0;
                cm.push(c);
                names.push(format!("{eff}:r{}vs{}.{tvar}", fmtv(lvals[l]), fmtv(lvals[0])));
                bl.rows.push((wrap12(&format!("({} vs {})", lab(l), lab(0))), 0.0, 0.0));
            }
            blocks.push(bl);
            let mut c = vec![0.0; nl];
            c[0] = 1.0;
            cm.push(c);
            names.push(format!("POmean:{}.{tvar}", fmtv(lvals[0])));
            blocks.push(Block { title: "POmean".into(), rows: vec![(vec![crate::summarize::abbrev(&lab(0), 12)], 0.0, 0.0)] });
        }
        let (b, vv) = super::combine(&cm, &theta, &v, op);
        let mut q = 0;
        for bl in blocks.iter_mut() {
            for r in bl.rows.iter_mut() {
                r.1 = b[q];
                r.2 = vv[q][q];
                q += 1;
            }
        }
        let est_name = match method {
            Method::Ra => "regression adjustment",
            Method::Ipw => "inverse-probability weights",
            Method::Ipwra => "IPW regression adjustment",
            Method::Wra => "weighted regression adjustment",
        };
        let oname = if oexpo { "exponential" } else { "weibull" };
        let cname = if !has_c { "none" } else if cexpo { "exponential" } else { "weibull" };
        let tname = if !has_t { "none" } else if tlink == Family::Probit { "probit" } else { "logit" };
        let nf = dd.iter().filter(|d| **d == 1.0).count();
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Survival treatment-effects estimation", util::commas(n)));
        out.push(format!("Estimator      : {est_name}{:>width$}Failures          = {:>10}", "", util::commas(nf), width = 31usize.saturating_sub(est_name.len())));
        out.push(format!("Outcome model  : {}", if has_o { oname } else { "weighted mean" }));
        out.push(format!("Treatment model: {tname}"));
        out.push(format!("Censoring model: {cname}"));
        effects_table(&mut out, "_t", &tvar, "Robust", level, &blocks);
        let text = out.join("\n");
        let sub_s = sub.to_string();
        super::post(st, ds, &rows, "stteffects", &b, &vv, &names, &|e: &mut Results| {
            e.set_str("subcmd", sub_s.clone());
            e.set_str("depvar", "_t");
            e.set_str("tvar", tvar.clone());
            e.set_str("stat", match stat {
                crate::teffects::Stat::Ate => "ate",
                crate::teffects::Stat::Atet => "atet",
                crate::teffects::Stat::Pomeans => "pomeans",
            });
            e.set_str("omodel", if has_o { oname } else { "none" });
            e.set_str("tmodel", tname);
            e.set_str("cmodel", cname);
            e.set_str("title", "Survival treatment-effects estimation");
            e.set("N_fail", nf as f64);
            e.set("k_levels", nl as f64);
            e.set("control", lvals[0]);
            e.set("treated", lvals[trl]);
            e.set("level", level);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
