//! Conditional average treatment effects (`cate`) and doubly robust
//! treatment effects with lasso-selected models (`telasso`).
//!
//!   cate po|aipw (ovar catevars) (tvar) [if] [in] [, controls(varlist)
//!        omethod(linear|lasso) tmethod(logit|lasso) xfolds(#) rseed(#) level(#)]
//!   telasso (ovar omvars [, linear|logit|probit|poisson]) (tvar tmvars [, logit|probit])
//!        [if] [in] [, ate|pomeans level(#)]
//!
//! `cate` cross-fits the nuisance functions (outcome mean, treatment
//! propensity; outcome means per arm for `aipw`) over random folds drawn
//! from the session generator, then estimates a CATE that is linear in the
//! `catevars`: `po` by the residual-on-residual (R-learner) regression,
//! `aipw` by regressing the doubly robust pseudo-outcome on the covariates
//! (DR-learner). The ATE is the mean of the doubly robust score.
//! Postestimation: `predict` (cate, or the score `psi`), `estat ate`,
//! `estat projection [varlist]` (robust regression of the score) and
//! `estat heterogeneity` (Wald test that the CATE is constant).
//!
//! Gap: the nonparametric generalized random forest CATE is not
//! implemented; only the parametric (linear-in-covariates) CATE is
//! available, and individual effects are its fitted values.
//!
//! `telasso` selects the covariates of the outcome models (per treatment
//! arm, union across arms) and of the logit/probit treatment model by
//! plug-in lasso, refits the selected models without penalty and reports
//! augmented-IPW effects with standard errors from the stacked estimating
//! equations of the refitted models.

use std::cell::RefCell;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::{dot, fam_label, family_of, fit, fmtv, groups, levels, sandwich, score, solve_block};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::teffects::{base_vars, design_rows, effects_table, level_label, model, sub_design, wrap12, Block};
use crate::util;
use crate::varlist;
use crate::w7b::lasso::core::{self, Family, Problem};
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    vec![("cate", Box::new(Cate)), ("telasso", Box::new(Telasso))]
}

/// Nuisance fit on the `train` rows: (intercept, slopes over `cols`).
fn fit_nuis(cols: &[Vec<f64>], y: &[f64], train: &[bool], fam: Family, lasso: bool) -> Result<(f64, Vec<f64>), String> {
    let idx: Vec<usize> = (0..y.len()).filter(|&i| train[i]).collect();
    let xs: Vec<Vec<f64>> = cols.iter().map(|c| idx.iter().map(|&i| c[i]).collect()).collect();
    let yy: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
    let w = vec![1.0; idx.len()];
    if lasso && !cols.is_empty() {
        let p = Problem { family: fam, x: xs, y: yy, w, offset: None, loadings: vec![1.0; cols.len()], alpha: 1.0, standardize: true };
        let (_, _, pt) = core::plugin(&p, false);
        let sel: Vec<usize> = (0..cols.len()).filter(|&j| pt.b[j] != 0.0).collect();
        return Ok(core::refit(&p, &sel));
    }
    let rows: Vec<Vec<f64>> = (0..idx.len()).map(|r| xs.iter().map(|c| c[r]).chain(std::iter::once(1.0)).collect()).collect();
    let b = fit(&rows, &yy, &w, fam)?;
    let k = cols.len();
    Ok((b[k], b[..k].to_vec()))
}

fn pred(b: &(f64, Vec<f64>), cols: &[Vec<f64>], i: usize) -> f64 {
    b.0 + b.1.iter().zip(cols).map(|(a, c)| a * c[i]).sum::<f64>()
}

/// OLS of `y` on rows `z` with HC0 covariance.
fn ols_robust(z: &[Vec<f64>], y: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let k = z[0].len();
    let n = y.len();
    let mut a = vec![vec![0.0; k]; k];
    let mut g = vec![0.0; k];
    for i in 0..n {
        for r in 0..k {
            g[r] += z[i][r] * y[i];
            for c in 0..k {
                a[r][c] += z[i][r] * z[i][c];
            }
        }
    }
    let b = crate::teffects::solve(&a, &g).ok_or_else(|| rc_err(459, "collinear covariates"))?;
    let ai = crate::teffects::invert(&a).ok_or_else(|| rc_err(459, "collinear covariates"))?;
    let mut m = vec![vec![0.0; k]; k];
    for i in 0..n {
        let e = y[i] - dot(&z[i], &b);
        for r in 0..k {
            for c in 0..k {
                m[r][c] += e * e * z[i][r] * z[i][c];
            }
        }
    }
    let nn = n as f64;
    let adj = nn / (nn - k as f64).max(1.0);
    let v: Vec<Vec<f64>> = (0..k).map(|r| (0..k).map(|c| adj * (0..k).map(|p| (0..k).map(|q| ai[r][p] * m[p][q] * ai[q][c]).sum::<f64>()).sum::<f64>()).collect()).collect();
    Ok((b, v))
}

/// Coefficient rows (label, b, se, z, p, ci).
fn coef_rows(out: &mut Vec<String>, title: &str, names: &[String], b: &[f64], v: &[Vec<f64>], level: f64) {
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    out.push("-".repeat(78));
    out.push(format!("{:>12} |{:>17}", "", "Robust"));
    out.push(format!("{:>12} | Coefficient  std. err.      z    P>|z|     [{}% conf. interval]", crate::summarize::abbrev(title, 12), fmtv(level)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    for (j, nm) in names.iter().enumerate() {
        let se = v[j][j].max(0.0).sqrt();
        let z = b[j] / se;
        let p = 2.0 * (1.0 - dist::normal(z.abs()));
        out.push(format!(
            "{:>12} |  {:>9}  {:>9}  {:>7.2}   {:>5.3}    {:>9}   {:>9}",
            crate::summarize::abbrev(nm, 12),
            crate::format::fmt_g(b[j], 7),
            crate::format::fmt_g(se, 7),
            z,
            p,
            crate::format::fmt_g(b[j] - crit * se, 7),
            crate::format::fmt_g(b[j] + crit * se, 7)
        ));
    }
    out.push("-".repeat(78));
}

#[derive(Clone)]
struct CateState {
    rows: Vec<usize>,
    psi: Vec<f64>,
}

thread_local! {
    static STATE: RefCell<Option<CateState>> = const { RefCell::new(None) };
}

pub struct Cate;

impl CommandHandler for Cate {
    fn name(&self) -> &'static str {
        "cate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("cate") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let aipw = match sub {
            "po" => false,
            "aipw" => true,
            _ => return Err(rc_err(198, format!("cate {sub}: unknown estimator; use po or aipw"))),
        };
        let g = groups(rest);
        if g.len() != 2 {
            return Err(rc_err(198, "cate requires (ovar catevarlist) (tvar)"));
        }
        let (om, _) = model(&g[0]);
        let (tm, _) = model(&g[1]);
        let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let cvars: Vec<String> = om[1..].to_vec();
        let controls: Vec<String> = match cmd.opt("controls", 4) {
            Some(c) => c.split_whitespace().map(String::from).collect(),
            None => Vec::new(),
        };
        let olasso = match cmd.opt("omethod", 2).map(|s| s.trim().to_lowercase()) {
            None => false,
            Some(s) if s == "linear" || s == "regress" => false,
            Some(s) if s == "lasso" => true,
            Some(s) => return Err(rc_err(198, format!("omethod({s}) not available; use linear or lasso (random forests are not implemented)"))),
        };
        let tlasso = match cmd.opt("tmethod", 2).map(|s| s.trim().to_lowercase()) {
            None => false,
            Some(s) if s == "logit" => false,
            Some(s) if s == "lasso" => true,
            Some(s) => return Err(rc_err(198, format!("tmethod({s}) not available; use logit or lasso (random forests are not implemented)"))),
        };
        let k_folds = util::opt_num(cmd, "xfolds", 2)?.map(|v| v as usize).unwrap_or(5).max(2);
        if let Some(s) = util::opt_num(cmd, "rseed", 5)? {
            st.rng.borrow_mut().reseed(s as u64);
        }
        let level = util::level(cmd, st)?;
        // nuisance covariates: catevars and controls
        let mut wterms = cvars.clone();
        for c in &controls {
            if !wterms.contains(c) {
                wterms.push(c.clone());
            }
        }
        let mut extra = vec![ovar.clone(), tvar.clone()];
        extra.extend(base_vars(ds, &wterms)?);
        let wd = sub_design(ds, st, cmd, &ovar, &wterms, false, extra.clone())?;
        let cd = sub_design(ds, st, cmd, &tvar, &cvars, false, extra)?;
        if wd.rows != cd.rows {
            return Err(rc_err(459, "models use different samples"));
        }
        let n = wd.rows.len();
        if n < 2 * k_folds + 4 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let y = wd.y.clone();
        let t = cd.y.clone();
        if levels(&t, &tvar)? != [0.0, 1.0] {
            return Err(rc_err(459, format!("{tvar} must be a 0/1 treatment")));
        }
        let wcols: Vec<Vec<f64>> = wd.active().into_iter().filter(|&j| wd.names[j] != "_cons").map(|j| wd.cols[j].clone()).collect();
        let zact: Vec<usize> = cd.active();
        let znames: Vec<String> = zact.iter().map(|&j| cd.names[j].clone()).collect();
        let zrows = design_rows(&cd);
        // folds
        let fold = {
            let mut rng = st.rng.borrow_mut();
            core::folds(n, k_folds, &mut || rng.uniform())
        };
        let mut m_hat = vec![0.0; n];
        let mut e_hat = vec![0.0; n];
        let mut mu1 = vec![0.0; n];
        let mut mu0 = vec![0.0; n];
        for f in 1..=k_folds {
            let train: Vec<bool> = fold.iter().map(|&g| g != f).collect();
            let eb = fit_nuis(&wcols, &t, &train, Family::Logit, tlasso)?;
            let (bm, b1, b0) = if aipw {
                let tr1: Vec<bool> = (0..n).map(|i| train[i] && t[i] == 1.0).collect();
                let tr0: Vec<bool> = (0..n).map(|i| train[i] && t[i] == 0.0).collect();
                (None, Some(fit_nuis(&wcols, &y, &tr1, Family::Linear, olasso)?), Some(fit_nuis(&wcols, &y, &tr0, Family::Linear, olasso)?))
            } else {
                (Some(fit_nuis(&wcols, &y, &train, Family::Linear, olasso)?), None, None)
            };
            for i in 0..n {
                if fold[i] != f {
                    continue;
                }
                e_hat[i] = Family::Logit.mean(pred(&eb, &wcols, i)).clamp(1e-3, 1.0 - 1e-3);
                if let Some(b) = &bm {
                    m_hat[i] = pred(b, &wcols, i);
                }
                if let (Some(a1), Some(a0)) = (&b1, &b0) {
                    mu1[i] = pred(a1, &wcols, i);
                    mu0[i] = pred(a0, &wcols, i);
                    m_hat[i] = e_hat[i] * mu1[i] + (1.0 - e_hat[i]) * mu0[i];
                }
            }
        }
        // CATE coefficients and the doubly robust score
        let (beta, vbeta, psi) = if aipw {
            let phi: Vec<f64> = (0..n).map(|i| mu1[i] - mu0[i] + t[i] * (y[i] - mu1[i]) / e_hat[i] - (1.0 - t[i]) * (y[i] - mu0[i]) / (1.0 - e_hat[i])).collect();
            let (b, v) = ols_robust(&zrows, &phi)?;
            (b, v, phi)
        } else {
            let tt: Vec<f64> = (0..n).map(|i| t[i] - e_hat[i]).collect();
            let yt: Vec<f64> = (0..n).map(|i| y[i] - m_hat[i]).collect();
            let zz: Vec<Vec<f64>> = (0..n).map(|i| zrows[i].iter().map(|v| v * tt[i]).collect()).collect();
            let (b, v) = ols_robust(&zz, &yt)?;
            let psi: Vec<f64> = (0..n)
                .map(|i| {
                    let tau = dot(&zrows[i], &b);
                    tau + tt[i] / (e_hat[i] * (1.0 - e_hat[i])) * (yt[i] - tt[i] * tau)
                })
                .collect();
            (b, v, psi)
        };
        let ate = psi.iter().sum::<f64>() / n as f64;
        let var_ate = psi.iter().map(|p| (p - ate) * (p - ate)).sum::<f64>() / (n as f64 * (n as f64 - 1.0));
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Conditional average treatment effects", util::commas(n)));
        out.push(format!("Estimator       : {}", if aipw { "augmented inverse-probability weighting" } else { "partialing out" }));
        out.push(format!("Outcome model   : {}", if olasso { "linear lasso" } else { "linear" }));
        out.push(format!("Treatment model : {}", if tlasso { "logit lasso" } else { "logit" }));
        out.push(format!("CATE model      : linear in covariates{:>12}Cross-fit folds   = {:>10}", "", k_folds));
        let blocks = vec![Block { title: "ATE".into(), rows: vec![(wrap12(&format!("({} vs {})", level_label(ds, &tvar, 1.0), level_label(ds, &tvar, 0.0))), ate, var_ate)] }];
        effects_table(&mut out, &ovar, &tvar, "Robust", level, &blocks);
        out.push(String::new());
        out.push("Linear projection of the CATE".into());
        coef_rows(&mut out, "CATE", &znames, &beta, &vbeta, level);
        let text = out.join("\n");
        let rows = wd.rows.clone();
        STATE.with(|s| *s.borrow_mut() = Some(CateState { rows: rows.clone(), psi: psi.clone() }));
        let names = vec![format!("ATE:r1vs0.{tvar}")];
        let k = beta.len();
        let mut bm = Matrix::row_vector(&beta).with_names(vec!["y1".into()], znames.clone());
        bm.rownames = vec!["y1".into()];
        let mut vm = Matrix::zeros(k, k);
        for r in 0..k {
            for c in 0..k {
                vm[(r, c)] = vbeta[r][c];
            }
        }
        let vm = vm.with_names(znames.clone(), znames.clone());
        let cv = cvars.join(" ");
        super::post(st, ds, &rows, "cate", &[ate], &[vec![var_ate]], &names, &|e: &mut Results| {
            e.set_str("subcmd", if aipw { "aipw" } else { "po" });
            e.set_str("depvar", ovar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("catevars", cv.clone());
            e.set_str("controls", controls.join(" "));
            e.set_str("title", "Conditional average treatment effects");
            e.set_str("omethod", if olasso { "lasso" } else { "linear" });
            e.set_str("tmethod", if tlasso { "lasso" } else { "logit" });
            e.set("k_folds", k_folds as f64);
            e.set("level", level);
            e.set_mat("b_cate", bm.clone());
            e.set_mat("V_cate", vm.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

fn need_state(_st: &Store) -> Result<CateState, String> {
    STATE.with(|s| s.borrow().clone()).ok_or_else(|| rc_err(301, "last estimates not found"))
}

/// `predict newvar [, cate | psi]` after `cate`.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mut vals = vec![SYSMISS; ds.n_obs];
    let mask = util::touse(ds, st, cmd)?;
    if cmd.has_opt("psi", 3) {
        let s = need_state(st)?;
        for (k, &r) in s.rows.iter().enumerate() {
            if r < vals.len() && mask[r] {
                vals[r] = s.psi[k];
            }
        }
    } else {
        let b = st.e.matrices.get("b_cate").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let env = crate::eval::Env::with_store(ds, st);
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            let mut acc = 0.0;
            let mut ok = true;
            for (j, c) in b.colnames.iter().enumerate() {
                let x = crate::postest::coef_value(&env, c, i)?;
                if is_missing(x) {
                    ok = false;
                    break;
                }
                acc += x * b.get(0, j);
            }
            if ok {
                vals[i] = acc;
            }
        }
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut out = String::new();
    if nmiss > 0 {
        out = format!("({} missing value{} generated)", nmiss, if nmiss == 1 { "" } else { "s" });
    }
    st.data_changed = true;
    Ok(out)
}

/// `estat ate | projection [varlist] | heterogeneity` after `cate`.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let level = util::level(cmd, st)?;
    match sub {
        "ate" => {
            let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let mut out = vec![String::new()];
            coef_rows(&mut out, "", &["ATE".to_string()], &[b.get(0, 0)], &[vec![v.get(0, 0)]], level);
            Ok(out.join("\n"))
        }
        "projection" | "proj" => {
            let s = need_state(st)?;
            let rest = cmd.main.trim().split_once(char::is_whitespace).map(|x| x.1.trim().to_string()).unwrap_or_default();
            let rest = rest.split(',').next().unwrap_or("").trim().to_string();
            let vars: Vec<String> = if rest.is_empty() {
                st.e.macros.get("catevars").map(|c| c.split_whitespace().map(String::from).collect()).unwrap_or_default()
            } else {
                varlist::expand(ds, &rest)?
            };
            let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
            let mut zr = Vec::new();
            let mut yy = Vec::new();
            for (k, &r) in s.rows.iter().enumerate() {
                let row: Vec<f64> = cols.iter().map(|c| c[r]).chain(std::iter::once(1.0)).collect();
                if row.iter().any(|v| is_missing(*v)) {
                    continue;
                }
                zr.push(row);
                yy.push(s.psi[k]);
            }
            let (b, v) = ols_robust(&zr, &yy)?;
            let mut names = vars.clone();
            names.push("_cons".into());
            let mut out = vec![String::new(), format!("Linear projection of the CATE{:>30}Number of obs = {:>8}", "", util::commas(yy.len()))];
            coef_rows(&mut out, "CATE", &names, &b, &v, level);
            let k = b.len();
            let idx: Vec<usize> = (0..k - 1).collect();
            let (chi2, df) = wald(&b, &v, &idx);
            out.push(format!("Wald chi2({df}) = {:.2}   Prob > chi2 = {:.4}", chi2, crate::xmodels::chi2p(df as f64, chi2)));
            let mut r = Results::default();
            r.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
            r.set("chi2", chi2);
            r.set("df", df as f64);
            r.set("p", crate::xmodels::chi2p(df as f64, chi2));
            st.r = r;
            Ok(out.join("\n"))
        }
        "heterogeneity" | "het" => {
            let b = st.e.matrices.get("b_cate").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let v = st.e.matrices.get("V_cate").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let k = b.cols;
            let bb: Vec<f64> = (0..k).map(|j| b.get(0, j)).collect();
            let vv: Vec<Vec<f64>> = (0..k).map(|r| (0..k).map(|c| v.get(r, c)).collect()).collect();
            let idx: Vec<usize> = (0..k).filter(|&j| b.colnames[j] != "_cons").collect();
            if idx.is_empty() {
                return Err(rc_err(198, "no CATE covariates to test"));
            }
            let (chi2, df) = wald(&bb, &vv, &idx);
            let p = crate::xmodels::chi2p(df as f64, chi2);
            let mut r = Results::default();
            r.set("chi2", chi2);
            r.set("df", df as f64);
            r.set("p", p);
            st.r = r;
            Ok(format!("\nTest of treatment-effect heterogeneity\nH0: Treatment effects are homogeneous\n\n    chi2({df}) = {:>10.2}\n    Prob > chi2 = {:>7.4}", chi2, p))
        }
        _ => Err(rc_err(321, format!("estat {sub} not valid after cate"))),
    }
}

fn wald(b: &[f64], v: &[Vec<f64>], idx: &[usize]) -> (f64, usize) {
    let k = idx.len();
    if k == 0 {
        return (0.0, 0);
    }
    let sub: Vec<Vec<f64>> = idx.iter().map(|&r| idx.iter().map(|&c| v[r][c]).collect()).collect();
    let bs: Vec<f64> = idx.iter().map(|&j| b[j]).collect();
    match crate::teffects::invert(&sub) {
        Some(inv) => ((0..k).map(|r| (0..k).map(|c| bs[r] * inv[r][c] * bs[c]).sum::<f64>()).sum(), k),
        None => (SYSMISS, k),
    }
}

// ---------------------------------------------------------------------------
// telasso
// ---------------------------------------------------------------------------

pub struct Telasso;

impl CommandHandler for Telasso {
    fn name(&self) -> &'static str {
        "telasso"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("telasso") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        let g = groups(main);
        if g.len() != 2 {
            return Err(rc_err(198, "telasso requires (ovar omvarlist) (tvar tmvarlist)"));
        }
        let (om, oopts) = model(&g[0]);
        let (tm, topts) = model(&g[1]);
        let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let (ox, tx) = (om[1..].to_vec(), tm[1..].to_vec());
        let fo = family_of(&oopts, Family::Linear);
        let ft = if topts.contains("probit") { Family::Probit } else { Family::Logit };
        if cmd.has_opt("atet", 4) {
            return Err(rc_err(198, "atet is not available with telasso; use ate or pomeans"));
        }
        let pomeans = cmd.has_opt("pomeans", 3);
        let level = util::level(cmd, st)?;
        let mut all = vec![ovar.clone(), tvar.clone()];
        all.extend(base_vars(ds, &ox)?);
        all.extend(base_vars(ds, &tx)?);
        let od = sub_design(ds, st, cmd, &ovar, &ox, false, all.clone())?;
        let td = sub_design(ds, st, cmd, &tvar, &tx, false, all)?;
        if od.rows != td.rows {
            return Err(rc_err(459, "models use different samples"));
        }
        let n = od.rows.len();
        let y = od.y.clone();
        let t = td.y.clone();
        if levels(&t, &tvar)? != [0.0, 1.0] {
            return Err(rc_err(459, format!("{tvar} must be a 0/1 treatment")));
        }
        let ocand: Vec<usize> = od.active().into_iter().filter(|&j| od.names[j] != "_cons").collect();
        let tcand: Vec<usize> = td.active().into_iter().filter(|&j| td.names[j] != "_cons").collect();
        // lasso selection
        let select = |cols: Vec<Vec<f64>>, yy: Vec<f64>, fam: Family| -> Vec<usize> {
            if cols.is_empty() {
                return Vec::new();
            }
            let k = cols.len();
            let p = Problem { family: fam, w: vec![1.0; yy.len()], x: cols, y: yy, offset: None, loadings: vec![1.0; k], alpha: 1.0, standardize: true };
            let (_, _, pt) = core::plugin(&p, false);
            (0..k).filter(|&j| pt.b[j] != 0.0).collect()
        };
        let mut osel: Vec<usize> = Vec::new();
        for a in [0.0, 1.0] {
            let idx: Vec<usize> = (0..n).filter(|&i| t[i] == a).collect();
            let cols: Vec<Vec<f64>> = ocand.iter().map(|&j| idx.iter().map(|&i| od.cols[j][i]).collect()).collect();
            let yy: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
            for s in select(cols, yy, fo) {
                if !osel.contains(&s) {
                    osel.push(s);
                }
            }
        }
        osel.sort();
        let tsel = select(tcand.iter().map(|&j| td.cols[j].clone()).collect(), t.clone(), ft);
        let xo: Vec<Vec<f64>> = (0..n).map(|i| osel.iter().map(|&s| od.cols[ocand[s]][i]).chain(std::iter::once(1.0)).collect()).collect();
        let xt: Vec<Vec<f64>> = (0..n).map(|i| tsel.iter().map(|&s| td.cols[tcand[s]][i]).chain(std::iter::once(1.0)).collect()).collect();
        let (ko, kt) = (xo[0].len(), xt[0].len());
        // refits and the AIPW system: b0 | b1 | gamma | pom0 pom1
        let mut theta = vec![0.0; 2 * ko + kt + 2];
        for a in 0..2 {
            let w: Vec<f64> = (0..n).map(|i| if t[i] == a as f64 { 1.0 } else { 0.0 }).collect();
            let b = fit(&xo, &y, &w, fo)?;
            theta[a * ko..(a + 1) * ko].copy_from_slice(&b);
        }
        let gm = fit(&xt, &t, &vec![1.0; n], ft)?;
        let og = 2 * ko;
        let op = og + kt;
        theta[og..op].copy_from_slice(&gm);
        let np = op + 2;
        let moments = |th: &[f64], i: usize, m: &mut [f64]| {
            let p1 = ft.mean(dot(&xt[i], &th[og..op])).clamp(1e-12, 1.0 - 1e-12);
            for a in 0..2 {
                let eta = dot(&xo[i], &th[a * ko..(a + 1) * ko]);
                let ind = if t[i] == a as f64 { 1.0 } else { 0.0 };
                let s = ind * score(fo, y[i], eta);
                for c in 0..ko {
                    m[a * ko + c] = xo[i][c] * s;
                }
                let pa = if a == 1 { p1 } else { 1.0 - p1 };
                let mu = fo.mean(eta);
                m[op + a] = mu + ind * (y[i] - mu) / pa - th[op + a];
            }
            let s = score(ft, t[i], dot(&xt[i], &th[og..op]));
            for c in 0..kt {
                m[og + c] = xt[i][c] * s;
            }
        };
        solve_block(&mut theta, n, np, op, &moments)?;
        let v = sandwich(&theta, n, np, &moments)?;
        let lab = |v: f64| crate::summarize::abbrev(&level_label(ds, &tvar, v), 12);
        let (names, cm, mut blocks) = if pomeans {
            (
                vec![format!("POmeans:0.{tvar}"), format!("POmeans:1.{tvar}")],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![Block { title: "POmeans".into(), rows: vec![(vec![lab(0.0)], 0.0, 0.0), (vec![lab(1.0)], 0.0, 0.0)] }],
            )
        } else {
            (
                vec![format!("ATE:r1vs0.{tvar}"), format!("POmean:0.{tvar}")],
                vec![vec![-1.0, 1.0], vec![1.0, 0.0]],
                vec![
                    Block { title: "ATE".into(), rows: vec![(wrap12(&format!("({} vs {})", level_label(ds, &tvar, 1.0), level_label(ds, &tvar, 0.0))), 0.0, 0.0)] },
                    Block { title: "POmean".into(), rows: vec![(vec![lab(0.0)], 0.0, 0.0)] },
                ],
            )
        };
        let (b, vv) = super::combine(&cm, &theta, &v, op);
        let mut q = 0;
        for bl in blocks.iter_mut() {
            for r in bl.rows.iter_mut() {
                r.1 = b[q];
                r.2 = vv[q][q];
                q += 1;
            }
        }
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Treatment-effects estimation", util::commas(n)));
        out.push("Estimator      : augmented IPW".into());
        out.push(format!("Outcome model  : {} by ML, lasso-selected", fam_label(fo)));
        out.push(format!("Treatment model: {}, lasso-selected", fam_label(ft)));
        out.push(format!("Selected       : {} of {} outcome covariates, {} of {} treatment covariates", osel.len(), ocand.len(), tsel.len(), tcand.len()));
        effects_table(&mut out, &ovar, &tvar, "Robust", level, &blocks);
        let text = out.join("\n");
        let rows = od.rows.clone();
        let osel_names: Vec<String> = osel.iter().map(|&s| od.names[ocand[s]].clone()).collect();
        let tsel_names: Vec<String> = tsel.iter().map(|&s| td.names[tcand[s]].clone()).collect();
        let (nos, nts, noc, ntc) = (osel.len(), tsel.len(), ocand.len(), tcand.len());
        super::post(st, ds, &rows, "telasso", &b, &vv, &names, &|e: &mut Results| {
            e.set_str("depvar", ovar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("stat", if pomeans { "pomeans" } else { "ate" });
            e.set_str("title", "Treatment-effects estimation");
            e.set_str("omodel", fam_label(fo));
            e.set_str("tmodel", fam_label(ft));
            e.set_str("othvars_sel", osel_names.join(" "));
            e.set_str("tvars_sel", tsel_names.join(" "));
            e.set("k_othvars_sel", nos as f64);
            e.set("k_tvars_sel", nts as f64);
            e.set("k_othvars", noc as f64);
            e.set("k_tvars", ntc as f64);
            e.set_str("_output", text.clone());
        });
        let _ = fmtv(0.0);
        Ok(text)
    }
}
