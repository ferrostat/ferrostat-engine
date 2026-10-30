//! Regressions with an endogenous binary treatment.
//!
//! `etregress` (linear outcome): y = xb + d*delta + e, with the treatment
//! d = 1(zg + u > 0) and (e, u) bivariate normal, corr(e, u) = rho,
//! var(u) = 1. Maximum likelihood (default), or Heckman's two-step
//! estimator using the generalized residual of a first-stage probit
//! (`twostep`, `cfunction`). `poutcomes` lets the outcome error variance
//! and correlation differ between the treatment groups.
//!
//! `etpoisson` (count outcome): y | d, e ~ Poisson(exp(xb + d*delta + e)),
//! with e normal (sd sigma) and correlated with the treatment error; the
//! likelihood integrates e out by Gauss-Hermite quadrature.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::util::{mills, phi};
use crate::commands::CommandHandler;
use crate::dist::normal;
use crate::est::{self, Design, Vce};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels::{self as xm, Handlers, Row};

pub fn handlers() -> Handlers {
    vec![("etregress", Box::new(Etreg { poisson: false })), ("etpoisson", Box::new(Etreg { poisson: true }))]
}

struct Etreg {
    poisson: bool,
}

/// Parse `treat(tvar = zvars [, noconstant])`.
fn treat_opt(ds: &DataSet, cmd: &Command) -> Result<(String, String, bool), String> {
    let o = cmd.opt("treat", 2).ok_or_else(|| rc_err(198, "option treat() required"))?;
    let (body, sub) = match crate::xmodels::outside_parens(o).find(',') {
        Some(p) => (o[..p].to_string(), o[p + 1..].to_string()),
        None => (o.to_string(), String::new()),
    };
    let (l, r) = body.split_once('=').ok_or_else(|| rc_err(198, "treat() requires tvar = varlist"))?;
    let tv = xm::var_name(ds, l.trim())?;
    let nocons = sub.split_whitespace().any(|w| w.len() >= 3 && "noconstant".starts_with(w));
    Ok((tv, r.trim().to_string(), nocons))
}

fn ln_norm_pdf(e: f64) -> f64 {
    -0.5 * e * e - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn ln_ncdf(z: f64) -> f64 {
    crate::choice_mixed::ln_phi_d(z).0
}

impl CommandHandler for Etreg {
    fn name(&self) -> &'static str {
        if self.poisson {
            "etpoisson"
        } else {
            "etregress"
        }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (dep, xs) = xm::first_word(cmd.main.trim());
        let dep = xm::var_name(ds, &dep)?;
        let (tv, zs, tnocons) = treat_opt(ds, cmd)?;
        let nocons = cmd.has_opt("noconstant", 3);
        let xs_t = format!("{} 1.{tv}", crate::tsops::expand_text(&xs));
        let designs = xm::build_eqs(ds, st, cmd, &[(Some(&dep), &xs_t, nocons), (Some(&tv), &zs, tnocons)], "fpi", &[])?;
        let (d0, d1) = (&designs[0], &designs[1]);
        if d0.rows != d1.rows {
            return Err(rc_err(459, "equations have different estimation samples"));
        }
        let n = d0.n();
        let tval: Vec<f64> = d1.y.clone();
        if tval.iter().any(|v| *v != 0.0 && *v != 1.0) {
            return Err(rc_err(450, format!("{tv} is not a 0/1 variable")));
        }
        if self.poisson && d0.y.iter().any(|v| *v < 0.0) {
            return Err(rc_err(459, format!("{dep} must be greater than or equal to zero")));
        }
        let level = util::level(cmd, st)?;
        let two = !self.poisson && (cmd.has_opt("twostep", 3) || cmd.has_opt("cfunction", 2));
        if two {
            return self.twostep(ds, st, cmd, &designs, &dep, &tv, level);
        }
        let pout = !self.poisson && cmd.has_opt("poutcomes", 3);
        let vce = xm::ml_vce_opt(cmd, d0)?;
        let cluster = est::cluster_values(ds, d0, &vce)?;
        // starting values: OLS / Poisson-ish start and probit
        let act1 = d1.active();
        let (mut b0, s0) = if self.poisson {
            let ly: Vec<f64> = d0.y.iter().map(|v| (v + 0.5).ln()).collect();
            let (b, _) = xm::ols_start(d0, &ly);
            (b, 0.5)
        } else {
            xm::ols_start(d0, &d0.y)
        };
        let c1: Vec<&[f64]> = act1.iter().map(|&c| d1.cols[c].as_slice()).collect();
        let pb = crate::heckman::probit_fit(&c1, &tval).map(|r| r.b).unwrap_or_else(|_| vec![0.0; act1.len()]);
        b0.extend(pb);
        let aux: Vec<&str> = if pout { vec!["athrho0", "lnsigma0", "athrho1", "lnsigma1"] } else { vec!["athrho", "lnsigma"] };
        if pout {
            b0.extend([0.0, s0.max(1e-3).ln(), 0.0, s0.max(1e-3).ln()]);
        } else {
            b0.extend([0.0, s0.max(1e-3).ln()]);
        }
        let y = d0.y.clone();
        let poisson = self.poisson;
        let (gx, gw) = crate::melogit::gauss_hermite(24);
        let nodes: Vec<f64> = gx.iter().map(|x| x * 2f64.sqrt()).collect();
        let lw: Vec<f64> = gw.iter().map(|w| (w / std::f64::consts::PI.sqrt()).ln()).collect();
        let lgy: Vec<f64> = y.iter().map(|v| crate::dist::lngamma(v + 1.0)).collect();
        let f = |t: usize, v: &[f64]| -> f64 {
            let (eta0, eta1) = (v[0], v[1]);
            let d = tval[t];
            let (ath, lns) = if pout && d == 1.0 { (v[4], v[5]) } else { (v[2], v[3]) };
            let rho = ath.clamp(-15.0, 15.0).tanh();
            let sig = lns.clamp(-30.0, 30.0).exp();
            let q = 2.0 * d - 1.0;
            let sr = (1.0 - rho * rho).max(1e-12).sqrt();
            if !poisson {
                let e = (y[t] - eta0) / sig;
                ln_norm_pdf(e) - sig.ln() + ln_ncdf(q * (eta1 + rho * e) / sr)
            } else {
                // integrate over e = sig * s, s standard normal
                let mut terms = Vec::with_capacity(nodes.len());
                for (s, lwk) in nodes.iter().zip(&lw) {
                    let mu = eta0 + sig * s;
                    let lp = y[t] * mu - mu.min(700.0).exp() - lgy[t];
                    let lt = ln_ncdf(q * (eta1 + rho * s) / sr);
                    terms.push(lwk + lp + lt);
                }
                let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                m + terms.iter().map(|a| (a - m).exp()).sum::<f64>().ln()
            }
        };
        let g = xm::numeric(&f);
        let eqs: Vec<(&str, &Design)> = vec![(dep.as_str(), d0), (tv.as_str(), d1)];
        let fit = xm::ml_fit(&eqs, &aux, &g, b0, &d0.w, d0.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        let robust = vce != Vce::Oim;
        // transformed parameters
        let mut extra: Vec<Row> = Vec::new();
        let pairs: Vec<(String, String, String)> = if pout {
            vec![("athrho0".into(), "lnsigma0".into(), "0".into()), ("athrho1".into(), "lnsigma1".into(), "1".into())]
        } else {
            vec![("athrho".into(), "lnsigma".into(), String::new())]
        };
        let mut rho_vals = Vec::new();
        let mut sig_vals = Vec::new();
        for (a, s, suf) in &pairs {
            let ja = fit.idx(&format!("/:{a}")).unwrap();
            let js = fit.idx(&format!("/:{s}")).unwrap();
            let (ba, sa) = (fit.b[ja], fit.v[(ja, ja)].max(0.0).sqrt());
            let (bs, ss) = (fit.b[js], fit.v[(js, js)].max(0.0).sqrt());
            let rho = ba.tanh();
            let sig = bs.exp();
            rho_vals.push(rho);
            sig_vals.push(sig);
            extra.push(Row::Tr(format!("rho{suf}"), rho, sa * (1.0 - rho * rho), ba, sa, f64::tanh));
            extra.push(Row::Tr(format!("sigma{suf}"), sig, ss * sig, bs, ss, f64::exp));
            if !self.poisson {
                // lambda = rho * sigma, delta method
                let lam = rho * sig;
                let ga = (1.0 - rho * rho) * sig;
                let gs = lam;
                let vl = ga * ga * fit.v[(ja, ja)] + gs * gs * fit.v[(js, js)] + 2.0 * ga * gs * fit.v[(ja, js)];
                let sl = vl.max(0.0).sqrt();
                let c = crate::choice_mixed::zcrit(level);
                extra.push(Row::Ci(format!("lambda{suf}"), lam, sl, lam - c * sl, lam + c * sl));
            }
        }
        let (chi2, df) = fit.wald_eq(0);
        let mut out = est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" });
        out.push(String::new());
        let title = if self.poisson { "Poisson regression with endogenous treatment" } else { "Linear regression with endogenous treatment" };
        let ncl = match (&vce, fit.n_clust) {
            (Vce::Cluster(c), Some(g)) => Some((g, c.clone())),
            _ => None,
        };
        out.extend(xm::Header { title, n: d0.n_eff(), lines: Vec::new(), test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: ncl }.render());
        out.extend(xm::table("", &fit, level, None, &extra));
        // test of independent equations
        let idxs: Vec<usize> = pairs.iter().map(|(a, _, _)| fit.pos.iter().position(|&p| fit.names[p] == format!("/:{a}")).unwrap()).collect();
        let (wchi, wdf) = crate::heckman::wald(&fit.bt, &fit.vt, &idxs);
        out.push(format!("Wald test of indep. eqns. (rho = 0): chi2({}) = {:>8}   Prob > chi2 = {}", wdf, xm::f2(wchi), xm::f4(xm::chi2p(wdf, wchi))));
        let text = out.join("\n");
        let ll = fit.ll();
        let depvars = format!("{dep} {tv}");
        let tvc = tv.clone();
        let rv = rho_vals.clone();
        let sv = sig_vals.clone();
        xm::post_fit(st, verb, d0, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set_str("depvar", depvars.clone());
            e.set_str("title", title);
            e.set_str("treat", tvc.clone());
            e.set_str("_output", text.clone());
            e.set("ll", ll);
            e.set("N", n as f64);
            e.set("chi2_c", wchi);
            e.set("p_c", xm::chi2p(wdf, wchi));
            e.set_str("method", "ml");
            e.set_str("predict", verb);
            if rv.len() == 1 {
                e.set("rho", rv[0]);
                e.set("sigma", sv[0]);
                if !poisson {
                    e.set("lambda", rv[0] * sv[0]);
                }
            } else {
                e.set("rho0", rv[0]);
                e.set("sigma0", sv[0]);
                e.set("rho1", rv[1]);
                e.set("sigma1", sv[1]);
            }
        });
        Ok(text)
    }
}

impl Etreg {
    /// Two-step estimator: probit, then least squares of the outcome on the
    /// covariates, the treatment and the generalized residual.
    #[allow(clippy::too_many_arguments)]
    fn twostep(&self, ds: &DataSet, st: &mut Store, cmd: &Command, designs: &[Design], dep: &str, tv: &str, level: f64) -> Result<String, String> {
        let _ = ds;
        let (d0, d1) = (&designs[0], &designs[1]);
        let n = d0.n();
        let act1 = d1.active();
        let c1: Vec<&[f64]> = act1.iter().map(|&c| d1.cols[c].as_slice()).collect();
        let pr = crate::heckman::probit_fit(&c1, &d1.y)?;
        let (vp, _) = pr.info.sym_inverse(1e-14);
        let zg: Vec<f64> = (0..n).map(|t| act1.iter().enumerate().map(|(a, &c)| d1.cols[c][t] * pr.b[a]).sum()).collect();
        let h: Vec<f64> = (0..n)
            .map(|t| {
                if d1.y[t] != 0.0 {
                    mills(zg[t])
                } else {
                    -mills(-zg[t])
                }
            })
            .collect();
        let mut d2 = d0.clone();
        d2.cols.push(h.clone());
        d2.names.push("lambda".into());
        d2.omitted.push(false);
        d2.is_base.push(false);
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let ofit = est::ols(&d2, &vce, None)?;
        let k0 = d0.k();
        let lam = ofit.b[k0];
        let dlt: f64 = (0..n).map(|t| h[t] * (h[t] + zg[t])).sum::<f64>() / n as f64;
        let sig2 = ofit.rss / n as f64 + lam * lam * dlt;
        let sig = sig2.max(1e-300).sqrt();
        let rho = (lam / sig).clamp(-1.0, 1.0);
        // assemble b and V: outcome, treatment, lambda
        let mut names: Vec<String> = d0.names.iter().map(|c| format!("{dep}:{c}")).collect();
        names.extend(d1.names.iter().map(|c| format!("{tv}:{c}")));
        names.push("/:lambda".into());
        let k = names.len();
        let mut b = vec![0.0; k];
        let mut v = Matrix::zeros(k, k);
        for j in 0..k0 {
            b[j] = ofit.b[j];
            for l in 0..k0 {
                v[(j, l)] = ofit.v[(j, l)];
            }
            v[(j, k - 1)] = ofit.v[(j, k0)];
            v[(k - 1, j)] = ofit.v[(k0, j)];
        }
        v[(k - 1, k - 1)] = ofit.v[(k0, k0)];
        b[k - 1] = lam;
        for (a, &c) in act1.iter().enumerate() {
            b[k0 + c] = pr.b[a];
            for (a2, &c2) in act1.iter().enumerate() {
                v[(k0 + c, k0 + c2)] = vp[(a, a2)];
            }
        }
        let mut omitted: Vec<bool> = d0.omitted.clone();
        omitted.extend(d1.omitted.iter().cloned());
        omitted.push(false);
        let mut is_base: Vec<bool> = d0.is_base.clone();
        is_base.extend(d1.is_base.iter().cloned());
        is_base.push(false);
        let ncoef = k - 1;
        let mut out = vec![crate::choice_mixed::hrow("Linear regression with endogenous treatment", "Number of obs", util::commas(n)), "Estimator: two-step".to_string()];
        let (chi2, df) = {
            let idx: Vec<usize> = (0..k0).filter(|&j| !d0.omitted[j] && d0.names[j] != "_cons").collect();
            crate::heckman::wald(&idx.iter().map(|&j| b[j]).collect::<Vec<_>>(), &{
                let mut m = Matrix::zeros(idx.len(), idx.len());
                for (a, &p) in idx.iter().enumerate() {
                    for (c, &q) in idx.iter().enumerate() {
                        m[(a, c)] = v[(p, q)];
                    }
                }
                m
            }, &(0..idx.len()).collect::<Vec<_>>())
        };
        out.push(crate::choice_mixed::hrow("", &format!("Wald chi2({})", crate::format::fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(String::new());
        let mut tbl = est::coef_table(&est::TableSpec { depvar: "", names: &names[..ncoef], b: &b[..ncoef], v: &v, omitted: &omitted[..ncoef], is_base: &is_base[..ncoef], df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(crate::choice_mixed::sep(lw));
        tbl.push(crate::choice_mixed::row_z(lw, "/lambda", lam, v[(k - 1, k - 1)].max(0.0).sqrt(), crate::choice_mixed::zcrit(level)));
        tbl.push(crate::choice_mixed::row_ci(lw, "rho", rho, SYSMISS, SYSMISS, SYSMISS).replace(&crate::format::fmt_g(SYSMISS, 8), "."));
        tbl.push(crate::choice_mixed::row_ci(lw, "sigma", sig, SYSMISS, SYSMISS, SYSMISS).replace(&crate::format::fmt_g(SYSMISS, 8), "."));
        tbl.push(bar);
        out.extend(tbl);
        let text = out.join("\n");
        let mut dp = d0.clone();
        dp.names = names.clone();
        dp.omitted = omitted;
        dp.is_base = is_base;
        let depvars = format!("{dep} {tv}");
        let tvs = tv.to_string();
        let verb = if cmd.has_opt("cfunction", 2) { "cfunction" } else { "twostep" };
        est::post(st, "etregress", &dp, &b, &v, &|e: &mut Results| {
            e.set_str("depvar", depvars.clone());
            e.set_str("treat", tvs.clone());
            e.set_str("method", verb);
            e.set_str("title", "Linear regression with endogenous treatment");
            e.set_str("_output", text.clone());
            e.set("rho", rho);
            e.set("sigma", sig);
            e.set("lambda", lam);
            e.set("chi2", chi2);
            e.set("df_m", df);
            e.set_str("chi2type", "Wald");
        });
        Ok(text)
    }
}

/// `predict` after `etregress` / `etpoisson`.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd != "etregress" && ecmd != "etpoisson" {
        return None;
    }
    let poisson = ecmd == "etpoisson";
    let tv = st.e.macros.get("treat").cloned().unwrap_or_default();
    let dep = st.e.macros.get("depvar").and_then(|d| d.split_whitespace().next().map(String::from)).unwrap_or_default();
    let b = st.e.matrices.get("b").cloned()?;
    let delta = b.col_index(&format!("{dep}:1.{tv}")).map_or(0.0, |j| b.get(0, j));
    let tcol = match util::col(ds, &tv) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    let get = |n: &str| st.e.scalars.get(n).copied().unwrap_or(SYSMISS);
    let (rho, sig) = (get("rho"), get("sigma"));
    let (rho0, sig0, rho1, sig1) = if is_missing(rho) { (get("rho0"), get("sigma0"), get("rho1"), get("sigma1")) } else { (rho, sig, rho, sig) };
    let lam = if st.e.macros.get("method").map(String::as_str) == Some("ml") { SYSMISS } else { get("lambda") };
    let stats: &[&str] = if poisson { &["n", "xb", "xbtrt", "ptrt", "cte", "pr"] } else { &["xb", "yctrt", "ycntrt", "ptrt", "xbtrt", "cte", "stdp"] };
    let msg = if poisson { "(option n assumed; predicted number of events)" } else { "(option xb assumed; fitted values)" };
    Some(xm::generic_predict(ds, st, cmd, stats, msg, &|s, c| {
        let xb = c.eta[0];
        let zg = c.eta.get(1).copied().unwrap_or(SYSMISS);
        let d = tcol[c.row];
        let x1 = xb + delta * (1.0 - d);
        let x0 = xb - delta * d;
        let rs1 = if is_missing(lam) { rho1 * sig1 } else { lam };
        let rs0 = if is_missing(lam) { rho0 * sig0 } else { lam };
        match s {
            "xb" => xb,
            "xbtrt" => zg,
            "ptrt" => normal(zg),
            "yctrt" => x1 + rs1 * phi(zg) / normal(zg).max(1e-300),
            "ycntrt" => x0 - rs0 * phi(zg) / (1.0 - normal(zg)).max(1e-300),
            "cte" if poisson => (x1 + sig1 * sig1 / 2.0).exp() - (x0 + sig0 * sig0 / 2.0).exp(),
            "cte" => {
                // conditional on treatment: E[y1 - y0 | d = 1]
                let m = phi(zg) / normal(zg).max(1e-300);
                x1 - x0 + (rs1 - rs0) * m
            }
            "n" => (xb + sig * sig / 2.0).exp(),
            "pr" => SYSMISS,
            _ => SYSMISS,
        }
    }))
}

/// `estat` after `etregress`: none beyond the generic ones.
pub fn estat(_ds: &mut DataSet, _st: &mut Store, _sub: &str, _cmd: &Command) -> Option<Result<String, String>> {
    None
}
