//! Local average treatment effects with a binary instrument.
//!
//!   lateffects ra|ipw|aipw|ipwra (ovar [omvars] [, linear|logit|probit|poisson])
//!        (tvar [tmvars] [, logit|probit|linear]) (zvar [zmvars] [, logit|probit])
//!        [if] [in] [, late|pomeans level(#)]
//!   lateoverlap
//!   latebalance summarize [varlist] | density varname | overid
//!
//! Under monotonicity and instrument validity given the covariates, the
//! effect among compliers is the ratio of two instrument contrasts,
//! E[Y(z=1)] - E[Y(z=0)] over E[D(z=1)] - E[D(z=0)] (Frolich 2007). Each
//! contrast is estimated by regression adjustment (models of Y and D within
//! instrument arms), inverse weighting by the instrument propensity,
//! augmented IPW, or IPW-weighted regression adjustment. The complier
//! potential-outcome means are the contrasts of D*Y and (1-D)*Y over the
//! same denominator. All parameters are stacked in one system of
//! estimating equations for robust standard errors.

use std::cell::RefCell;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::{dot, fam_label, family_of, fit, fmtv, groups, levels, sandwich, score, solve_block};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::teffects::{base_vars, design_rows, effects_table, level_label, model, sub_design, wrap12, Block};
use crate::util;
use crate::varlist;
use crate::w7b::lasso::core::Family;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    vec![("lateffects", Box::new(Lateffects)), ("lateoverlap", Box::new(Lateoverlap)), ("latebalance", Box::new(Latebalance))]
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Ra,
    Ipw,
    Aipw,
    Ipwra,
}

/// What `lateoverlap` / `latebalance` need from the last fit.
#[derive(Clone)]
struct LateState {
    rows: Vec<usize>,
    z: Vec<f64>,
    p1: Vec<f64>,
    zvar: String,
    zcovs: Vec<String>,
    has_model: bool,
}

thread_local! {
    static STATE: RefCell<Option<LateState>> = const { RefCell::new(None) };
}

fn state(st: &Store) -> Result<LateState, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("lateffects") {
        return Err(rc_err(301, "last estimates not found"));
    }
    STATE.with(|s| s.borrow().clone()).ok_or_else(|| rc_err(301, "last estimates not found"))
}

pub struct Lateffects;

impl CommandHandler for Lateffects {
    fn name(&self) -> &'static str {
        "lateffects"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("lateffects") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let method = match sub {
            "ra" => Method::Ra,
            "ipw" => Method::Ipw,
            "aipw" => Method::Aipw,
            "ipwra" => Method::Ipwra,
            _ => return Err(rc_err(198, format!("lateffects {sub}: unknown estimator"))),
        };
        let g = groups(rest);
        if g.len() != 3 {
            return Err(rc_err(198, "lateffects requires an outcome, a treatment and an instrument model in parentheses"));
        }
        let (om, oopts) = model(&g[0]);
        let (tm, topts) = model(&g[1]);
        let (zm, zopts) = model(&g[2]);
        let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let zvar = varlist::resolve_name(ds, zm.first().ok_or_else(|| rc_err(198, "instrument required"))?)?;
        let (ox, tx, zx) = (om[1..].to_vec(), tm[1..].to_vec(), zm[1..].to_vec());
        if method == Method::Ipw && (!ox.is_empty() || !tx.is_empty()) {
            return Err(rc_err(198, "ipw: the outcome and treatment models take no covariates"));
        }
        if method == Method::Ra && !zx.is_empty() {
            return Err(rc_err(198, "ra: the instrument model takes no covariates"));
        }
        let fo = family_of(&oopts, Family::Linear);
        let ft = family_of(&topts, Family::Logit);
        let fz = if zopts.contains("probit") { Family::Probit } else { Family::Logit };
        let pomeans = cmd.has_opt("pomeans", 3) || cmd.opt("stat", 4).map(str::trim) == Some("pomeans");
        if cmd.has_opt("latt", 4) {
            return Err(rc_err(198, "latt is not available; use late or pomeans"));
        }
        let level = util::level(cmd, st)?;

        let mut all = vec![ovar.clone(), tvar.clone(), zvar.clone()];
        all.extend(base_vars(ds, &ox)?);
        all.extend(base_vars(ds, &tx)?);
        all.extend(base_vars(ds, &zx)?);
        let od = sub_design(ds, st, cmd, &ovar, &ox, oopts.contains("nocons"), all.clone())?;
        let td = sub_design(ds, st, cmd, &tvar, &tx, topts.contains("nocons"), all.clone())?;
        let zd = sub_design(ds, st, cmd, &zvar, &zx, zopts.contains("nocons"), all)?;
        if od.rows != td.rows || od.rows != zd.rows {
            return Err(rc_err(459, "models use different samples"));
        }
        let n = od.rows.len();
        if n < 5 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let y = od.y.clone();
        let d = td.y.clone();
        let z = zd.y.clone();
        let dl = levels(&d, &tvar)?;
        let zl = levels(&z, &zvar)?;
        if dl != [0.0, 1.0] || zl != [0.0, 1.0] {
            return Err(rc_err(459, "treatment and instrument must be 0/1 variables"));
        }
        let xo = design_rows(&od);
        let xt = design_rows(&td);
        let xz = design_rows(&zd);
        let (ko, kt, kz) = (xo[0].len(), xt[0].len(), xz[0].len());
        // four outcomes: Y, D, D*Y, (1-D)*Y
        let vals: Vec<[f64; 4]> = (0..n).map(|i| [y[i], d[i], d[i] * y[i], (1.0 - d[i]) * y[i]]).collect();
        let has_o = method != Method::Ipw;
        let has_z = method != Method::Ra;
        // instrument model
        let gamma = if has_z { fit(&xz, &z, &vec![1.0; n], fz)? } else { Vec::new() };
        let p1 = |g: &[f64], i: usize| -> f64 { if has_z { fz.mean(dot(&xz[i], g)).clamp(1e-10, 1.0 - 1e-10) } else { 0.5 } };
        // outcome-type models per (outcome k, arm a): families and designs
        let fam_k = |k: usize, a: usize| -> Family {
            // constant outcomes within an arm are fitted linearly (exact)
            let base = if k == 1 { ft } else { fo };
            let mut first = None;
            let mut constant = true;
            for i in 0..n {
                if z[i] == a as f64 {
                    match first {
                        None => first = Some(vals[i][k]),
                        Some(f) if f != vals[i][k] => {
                            constant = false;
                            break;
                        }
                        _ => {}
                    }
                }
            }
            if constant { Family::Linear } else { base }
        };
        let fams: Vec<[Family; 2]> = (0..4).map(|k| [fam_k(k, 0), fam_k(k, 1)]).collect();
        let kk = |k: usize| if k == 1 { kt } else { ko };
        let xsk: [&Vec<Vec<f64>>; 4] = [&xo, &xt, &xo, &xo];
        // layout: models (k, a) | gamma | pom (k, a) | late, cpo1, cpo0
        let mut off = Vec::new();
        let mut np = 0;
        if has_o {
            for k in 0..4 {
                let mut r = [0; 2];
                for a in 0..2 {
                    r[a] = np;
                    np += kk(k);
                }
                off.push(r);
            }
        }
        let og = np;
        if has_z {
            np += kz;
        }
        let op = np;
        np += 8;
        let ol = np;
        np += 3;
        let ipw_w = |pz: f64, i: usize| -> f64 {
            if method != Method::Ipwra {
                1.0
            } else if z[i] == 1.0 {
                1.0 / pz
            } else {
                1.0 / (1.0 - pz)
            }
        };
        let mut theta = vec![0.0; np];
        if has_o {
            for k in 0..4 {
                for a in 0..2 {
                    let w: Vec<f64> = (0..n).map(|i| if z[i] == a as f64 { if has_z { ipw_w(p1(&gamma, i), i) } else { 1.0 } } else { 0.0 }).collect();
                    let rows: Vec<Vec<f64>> = (0..n).map(|i| xsk[k][i].clone()).collect();
                    let v: Vec<f64> = vals.iter().map(|v| v[k]).collect();
                    let b = fit(&rows, &v, &w, fams[k][a])?;
                    theta[off[k][a]..off[k][a] + kk(k)].copy_from_slice(&b);
                }
            }
        }
        theta[og..og + gamma.len()].copy_from_slice(&gamma);
        let moments = |th: &[f64], i: usize, m: &mut [f64]| {
            let pz = if has_z { p1(&th[og..og + kz], i) } else { 0.5 };
            if has_o {
                let w = ipw_w(pz, i);
                for k in 0..4 {
                    let a = z[i] as usize;
                    let o = off[k][a];
                    let s = w * score(fams[k][a], vals[i][k], dot(&xsk[k][i], &th[o..o + kk(k)]));
                    for (c, x) in xsk[k][i].iter().enumerate() {
                        m[o + c] = x * s;
                    }
                }
            }
            if has_z {
                let s = score(fz, z[i], dot(&xz[i], &th[og..og + kz]));
                for c in 0..kz {
                    m[og + c] = xz[i][c] * s;
                }
            }
            for k in 0..4 {
                for a in 0..2 {
                    let pom = th[op + 2 * k + a];
                    let ind = if z[i] == a as f64 { 1.0 } else { 0.0 };
                    let pa = if a == 1 { pz } else { 1.0 - pz };
                    let mu = if has_o {
                        let o = off[k][a];
                        fams[k][a].mean(dot(&xsk[k][i], &th[o..o + kk(k)]))
                    } else {
                        0.0
                    };
                    m[op + 2 * k + a] = match method {
                        Method::Ra | Method::Ipwra => mu - pom,
                        Method::Ipw => ind / pa * (vals[i][k] - pom),
                        Method::Aipw => mu + ind * (vals[i][k] - mu) / pa - pom,
                    };
                }
            }
            let dlt = |k: usize| th[op + 2 * k + 1] - th[op + 2 * k];
            let den = dlt(1);
            m[ol] = dlt(0) - th[ol] * den;
            m[ol + 1] = dlt(2) - th[ol + 1] * den;
            m[ol + 2] = -dlt(3) - th[ol + 2] * den;
        };
        // a nonzero first-stage contrast keeps the ratio equations solvable
        theta[op + 3] = 1.0;
        solve_block(&mut theta, n, np, op, &moments)?;
        let den = theta[op + 3] - theta[op + 2];
        if den.abs() < 1e-8 {
            return Err(rc_err(459, "the instrument does not shift the treatment (no compliers)"));
        }
        let v = sandwich(&theta, n, np, &moments)?;
        let lab = |v: f64| crate::summarize::abbrev(&level_label(ds, &tvar, v), 12);
        let (names, idx, blocks) = if pomeans {
            (
                vec![format!("cPOmeans:0.{tvar}"), format!("cPOmeans:1.{tvar}")],
                vec![ol + 2, ol + 1],
                vec![Block { title: "cPOmeans".into(), rows: vec![(vec![lab(0.0)], 0.0, 0.0), (vec![lab(1.0)], 0.0, 0.0)] }],
            )
        } else {
            (
                vec![format!("LATE:r1vs0.{tvar}"), format!("cPOmean:0.{tvar}")],
                vec![ol, ol + 2],
                vec![
                    Block { title: "LATE".into(), rows: vec![(wrap12(&format!("({} vs {})", level_label(ds, &tvar, 1.0), level_label(ds, &tvar, 0.0))), 0.0, 0.0)] },
                    Block { title: "cPOmean".into(), rows: vec![(vec![lab(0.0)], 0.0, 0.0)] },
                ],
            )
        };
        let b: Vec<f64> = idx.iter().map(|&j| theta[j]).collect();
        let vv: Vec<Vec<f64>> = idx.iter().map(|&r| idx.iter().map(|&c| v[r][c]).collect()).collect();
        let mut blocks = blocks;
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
            Method::Aipw => "augmented IPW",
            Method::Ipwra => "IPW regression adjustment",
        };
        let share = den;
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Local average treatment effects", util::commas(n)));
        out.push(format!("Estimator       : {est_name}"));
        out.push(format!("Outcome model   : {}", if has_o { fam_label(fo) } else { "weighted mean" }));
        out.push(format!("Treatment model : {}", if has_o { fam_label(ft) } else { "weighted mean" }));
        out.push(format!("Instrument model: {}", if has_z { fam_label(fz) } else { "none" }));
        out.push(format!("Instrument      : {zvar}{:>30}Compliers   = {:>10}", "", crate::format::fmt_g(share, 6).trim()));
        effects_table(&mut out, &ovar, &tvar, "Robust", level, &blocks);
        let text = out.join("\n");
        let rows = od.rows.clone();
        let p1v: Vec<f64> = (0..n).map(|i| p1(&gamma, i)).collect();
        STATE.with(|s| {
            *s.borrow_mut() = Some(LateState { rows: rows.clone(), z: z.clone(), p1: p1v, zvar: zvar.clone(), zcovs: base_vars(ds, &zx).unwrap_or_default(), has_model: has_z });
        });
        let sub_s = sub.to_string();
        super::post(st, ds, &rows, "lateffects", &b, &vv, &names, &|e: &mut Results| {
            e.set_str("subcmd", sub_s.clone());
            e.set_str("depvar", ovar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("ivar", zvar.clone());
            e.set_str("stat", if pomeans { "pomeans" } else { "late" });
            e.set_str("title", "Local average treatment effects");
            e.set_str("omodel", fam_label(fo));
            e.set_str("tmodel", fam_label(ft));
            e.set_str("imodel", if has_z { fam_label(fz) } else { "none" });
            e.set("compliers", share);
            e.set("level", level);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// Densities of the instrument propensity by instrument group.
pub struct Lateoverlap;

impl CommandHandler for Lateoverlap {
    fn name(&self) -> &'static str {
        "lateoverlap"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let s = state(st)?;
        if !s.has_model {
            return Err(rc_err(321, "lateoverlap requires an estimator with an instrument model"));
        }
        let mut layers = Vec::new();
        for a in [0.0, 1.0] {
            let x: Vec<f64> = (0..s.z.len()).filter(|&i| s.z[i] == a).map(|i| s.p1[i]).collect();
            if x.len() < 2 {
                continue;
            }
            let w = vec![1.0; x.len()];
            layers.push(Layer { kind: LayerKind::Line, points: crate::te_match::kde(&x, &w, 0.0, 1.0, 200), label: format!("{}={}", s.zvar, fmtv(a)), ..Default::default() });
        }
        let mut g = Graph { xtitle: format!("Propensity score, {}=1", s.zvar), ytitle: "Density".into(), layers, legend: true, ..Default::default() };
        crate::graph::finish(&mut g, cmd, st)
    }
}

/// Balance of covariates across instrument groups, raw and weighted by the
/// inverse instrument propensity.
pub struct Latebalance;

fn wstats(x: &[f64], w: &[f64]) -> (f64, f64) {
    let sw: f64 = w.iter().sum();
    let m = x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let v = x.iter().zip(w).map(|(a, b)| b * (a - m) * (a - m)).sum::<f64>() / sw;
    let nn = x.len() as f64;
    (m, v * nn / (nn - 1.0).max(1.0))
}

impl CommandHandler for Latebalance {
    fn name(&self) -> &'static str {
        "latebalance"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let s = state(st)?;
        if !s.has_model {
            return Err(rc_err(321, "latebalance requires an estimator with an instrument model"));
        }
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let wt: Vec<f64> = (0..s.z.len()).map(|i| if s.z[i] == 1.0 { 1.0 / s.p1[i] } else { 1.0 / (1.0 - s.p1[i]) }).collect();
        let column = |v: &str| -> Result<Vec<f64>, String> {
            let c = util::col(ds, v)?;
            Ok(s.rows.iter().map(|&r| c[r]).collect())
        };
        match sub {
            "" | "summarize" | "sum" | "summ" => {
                let vars: Vec<String> = if rest.trim().is_empty() { s.zcovs.clone() } else { varlist::expand(ds, rest.trim())? };
                if vars.is_empty() {
                    return Err(rc_err(198, "no covariates to summarize"));
                }
                let n1 = s.z.iter().filter(|v| **v == 1.0).count();
                let mut out = vec![String::new(), "Covariate balance summary".into(), String::new()];
                out.push(format!("{:>40}{:>10}{:>12}", "", "Raw", "Weighted"));
                out.push(format!("{:<28}{:>12}{:>10}{:>12}", "Number of obs", "", util::commas(s.z.len()), util::commas(s.z.len())));
                out.push(format!("{:<28}{:>12}{:>10}{:>12}", format!("{}=1 obs", s.zvar), "", util::commas(n1), fmtv((0..s.z.len()).filter(|&i| s.z[i] == 1.0).map(|i| wt[i]).sum::<f64>().round())));
                out.push(String::new());
                out.push("-".repeat(61));
                out.push(format!("{:>12} |{:>23}{:>23}", "", "Standardized diff.", "Variance ratio"));
                out.push(format!("{:>12} |{:>12}{:>11}{:>12}{:>11}", "", "Raw", "Weighted", "Raw", "Weighted"));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(47)));
                let mut m = Matrix::zeros(vars.len(), 4);
                for (r, v) in vars.iter().enumerate() {
                    let x = column(v)?;
                    let pick = |a: f64, w: bool| -> (Vec<f64>, Vec<f64>) {
                        let idx: Vec<usize> = (0..x.len()).filter(|&i| s.z[i] == a).collect();
                        (idx.iter().map(|&i| x[i]).collect(), idx.iter().map(|&i| if w { wt[i] } else { 1.0 }).collect())
                    };
                    let mut row = [0.0; 4];
                    for (c, w) in [false, true].into_iter().enumerate() {
                        let (x1, w1) = pick(1.0, w);
                        let (x0, w0) = pick(0.0, w);
                        let (m1, v1) = wstats(&x1, &w1);
                        let (m0, v0) = wstats(&x0, &w0);
                        row[c] = (m1 - m0) / ((v1 + v0) / 2.0).sqrt();
                        row[2 + c] = v1 / v0;
                    }
                    for c in 0..4 {
                        m[(r, c)] = row[c];
                    }
                    out.push(format!("{:>12} |{:>12.7}{:>11.7}{:>12.7}{:>11.7}", crate::summarize::abbrev(v, 12), row[0], row[1], row[2], row[3]));
                }
                out.push("-".repeat(61));
                let m = m.with_names(vars.clone(), vec!["sdiff_raw".into(), "sdiff_wt".into(), "vratio_raw".into(), "vratio_wt".into()]);
                let mut r = Results::default();
                r.set_mat("table", m);
                r.set("N", s.z.len() as f64);
                st.r = r;
                Ok(out.join("\n"))
            }
            "density" | "dens" => {
                let v = varlist::resolve_name(ds, rest.split_whitespace().next().ok_or_else(|| rc_err(100, "varname required"))?)?;
                let x = column(&v)?;
                let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &b| (a.0.min(b), a.1.max(b)));
                let pad = (hi - lo) * 0.1;
                let mut layers = Vec::new();
                for (w, tag) in [(false, "Raw"), (true, "Weighted")] {
                    for a in [0.0, 1.0] {
                        let idx: Vec<usize> = (0..x.len()).filter(|&i| s.z[i] == a).collect();
                        let xs: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
                        let ws: Vec<f64> = idx.iter().map(|&i| if w { wt[i] } else { 1.0 }).collect();
                        if xs.len() < 2 {
                            continue;
                        }
                        layers.push(Layer { kind: LayerKind::Line, points: crate::te_match::kde(&xs, &ws, lo - pad, hi + pad, 200), label: format!("{tag}, {}={}", s.zvar, fmtv(a)), ..Default::default() });
                    }
                }
                let mut g = Graph { title: "Balance plot".into(), xtitle: v.clone(), ytitle: "Density".into(), layers, legend: true, ..Default::default() };
                crate::graph::finish(&mut g, cmd, st)
            }
            "overid" => {
                let vars = s.zcovs.clone();
                if vars.is_empty() {
                    return Err(rc_err(198, "overid requires covariates in the instrument model"));
                }
                let cols: Vec<Vec<f64>> = vars.iter().map(|v| column(v)).collect::<Result<_, _>>()?;
                let n = s.z.len();
                let k = cols.len();
                let mi: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        let h = s.z[i] / s.p1[i] - (1.0 - s.z[i]) / (1.0 - s.p1[i]);
                        cols.iter().map(|c| h * c[i]).collect()
                    })
                    .collect();
                let mbar: Vec<f64> = (0..k).map(|a| mi.iter().map(|r| r[a]).sum::<f64>() / n as f64).collect();
                let sm: Vec<Vec<f64>> = (0..k).map(|a| (0..k).map(|b| mi.iter().map(|r| (r[a] - mbar[a]) * (r[b] - mbar[b])).sum::<f64>() / n as f64).collect()).collect();
                let sinv = crate::teffects::invert(&sm).ok_or_else(|| rc_err(459, "balance moments are collinear"))?;
                let chi2: f64 = n as f64 * (0..k).map(|a| (0..k).map(|b| mbar[a] * sinv[a][b] * mbar[b]).sum::<f64>()).sum::<f64>();
                let p = crate::xmodels::chi2p(k as f64, chi2);
                let mut r = Results::default();
                r.set("chi2", chi2);
                r.set("df", k as f64);
                r.set("p", p);
                st.r = r;
                Ok(format!(
                    "\nOveridentification test for covariate balance\n         H0: Covariates are balanced:\n\n         chi2({k})     = {:>10}\n         Prob > chi2 = {:>10.4}",
                    fmtv((chi2 * 1e4).round() / 1e4),
                    p
                ))
            }
            _ => Err(rc_err(198, format!("latebalance {sub}: unknown subcommand"))),
        }
    }
}
