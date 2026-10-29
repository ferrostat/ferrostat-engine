//! Binary, fractional, count and ordinal variants: `hetprobit`, `scobit`,
//! `fracreg`, `betareg`, `binreg`, `cpoisson`, `hetoprobit` and `mprobit`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::{gamma_ur, ln_gamma};

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xm_lin::pr_bounds;
use crate::xmodels::{self as xm, Header, Row};

pub fn handlers() -> xm::Handlers {
    vec![
        ("hetprobit", Box::new(Hetprobit)),
        ("scobit", Box::new(Scobit)),
        ("fracreg", Box::new(Fracreg)),
        ("betareg", Box::new(Betareg)),
        ("binreg", Box::new(Binreg)),
        ("cpoisson", Box::new(Cpoisson)),
        ("hetoprobit", Box::new(Hetoprobit)),
        ("mprobit", Box::new(Mprobit)),
    ]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    let link = st.e.macros.get("link").cloned().unwrap_or_default();
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let prmsg = format!("(option pr assumed; Pr({dep}))");
    match ecmd {
        "hetprobit" => Some(xm::generic_predict(ds, st, cmd, &["pr", "xb", "sigma", "stdp"], &prmsg, &|s, c| match s {
            "pr" => dist::normal(c.eta[0] / c.eta.get(1).map_or(1.0, |z| z.exp())),
            "xb" => c.eta[0],
            "sigma" => c.eta.get(1).map_or(1.0, |z| z.exp()),
            _ => SYSMISS,
        })),
        "scobit" => Some(xm::generic_predict(ds, st, cmd, &["pr", "xb", "stdp"], &prmsg, &|s, c| match s {
            "pr" => 1.0 - (1.0 + c.eta[0].exp()).powf(-(c.aux)("lnalpha").exp()),
            "xb" => c.eta[0],
            _ => SYSMISS,
        })),
        "fracreg" => Some(xm::generic_predict(ds, st, cmd, &["cm", "xb", "sigma", "stdp"], "(option cm assumed; conditional mean)", &|s, c| match s {
            "cm" => match link.as_str() {
                "logit" => dist::invlogit(c.eta[0]),
                "hetprobit" => dist::normal(c.eta[0] / c.eta.get(1).map_or(1.0, |z| z.exp())),
                _ => dist::normal(c.eta[0]),
            },
            "xb" => c.eta[0],
            "sigma" => c.eta.get(1).map_or(1.0, |z| z.exp()),
            _ => SYSMISS,
        })),
        "betareg" => Some(xm::generic_predict(ds, st, cmd, &["cmean", "xb", "xbscale", "cvariance", "stdp"], "(option cmean assumed; conditional mean)", &|s, c| {
            let mu = inv_link(&link, c.eta[0]);
            let phi = c.eta.get(1).map_or(SYSMISS, |z| z.exp());
            match s {
                "cmean" => mu,
                "xb" => c.eta[0],
                "xbscale" => c.eta.get(1).copied().unwrap_or(SYSMISS),
                "cvariance" => mu * (1.0 - mu) / (1.0 + phi),
                _ => SYSMISS,
            }
        })),
        "cpoisson" => Some(xm::generic_predict(ds, st, cmd, &["n", "ir", "xb", "stdp"], "(option n assumed; predicted number of events)", &|s, c| match s {
            "n" | "ir" => c.eta[0].exp(),
            "xb" => c.eta[0],
            _ => SYSMISS,
        })),
        "hetoprobit" | "mprobit" => {
            let cats: Vec<f64> = st.e.matrices.get(if ecmd == "mprobit" { "out" } else { "cat" }).map(|m| m.data.clone()).unwrap_or_default();
            let outcome = util::opt_num(cmd, "outcome", 3).ok().flatten();
            let k = outcome.and_then(|o| cats.iter().position(|c| *c == o)).unwrap_or(0);
            let mprob = ecmd == "mprobit";
            let base = st.e.get("i_base").unwrap_or(1.0) as usize - 1;
            let msg = format!("(option pr assumed; predicted probability of outcome {})", cats.get(k).map_or(String::new(), |c| fmt_g(*c, 8)));
            Some(xm::generic_predict(ds, st, cmd, &["pr", "xb", "stdp"], &msg, &|s, c| match s {
                "xb" => c.eta[0],
                _ if mprob => {
                    let mut eta = Vec::new();
                    let mut q = 0;
                    for j in 0..cats.len() {
                        if j == base {
                            eta.push(0.0);
                        } else {
                            eta.push(c.eta[q]);
                            q += 1;
                        }
                    }
                    mprobit_prob(&eta, k)
                }
                _ => {
                    let sig = c.eta.get(1).map_or(1.0, |z| z.exp());
                    let cut = |j: usize| (c.aux)(&format!("cut{}", j + 1));
                    let hi = if k + 1 < cats.len() { dist::normal((cut(k) - c.eta[0]) / sig) } else { 1.0 };
                    let lo = if k > 0 { dist::normal((cut(k - 1) - c.eta[0]) / sig) } else { 0.0 };
                    hi - lo
                }
            }))
        }
        "binreg" => Some(xm::generic_predict(ds, st, cmd, &["mu", "xb", "stdp"], "(option mu assumed; predicted mean)", &|s, c| match s {
            "mu" => {
                let p = match link.as_str() {
                    "log" => c.eta[0].exp(),
                    "logit" => dist::invlogit(c.eta[0]),
                    "logc" => 1.0 - c.eta[0].exp(),
                    _ => c.eta[0],
                };
                p
            }
            "xb" => c.eta[0],
            _ => SYSMISS,
        })),
        _ => {
            let _ = pr_bounds;
            None
        }
    }
}

fn inv_link(link: &str, x: f64) -> f64 {
    match link {
        "probit" => dist::normal(x),
        "cloglog" => 1.0 - (-x.exp()).exp(),
        "loglog" => (-(-x).exp()).exp(),
        _ => dist::invlogit(x),
    }
}

fn ln1p_exp(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}

/// Response design and counts of zero / nonzero outcomes.
fn counts(d: &Design) -> (f64, f64) {
    let mut z = 0.0;
    let mut nz = 0.0;
    for t in 0..d.n() {
        let w = if d.wkind == Some(WeightKind::F) { d.w[t] } else { 1.0 };
        if d.y[t] != 0.0 { nz += w } else { z += w }
    }
    (z, nz)
}

fn cluster_name(vce: &Vce) -> String {
    if let Vce::Cluster(c) = vce { c.clone() } else { String::new() }
}

fn probit_ll(y: f64, z: f64) -> f64 {
    let q = if y != 0.0 { 1.0 } else { -1.0 };
    crate::xtnl::ln_phi(q * z)
}

/// Probit / logit starting values on the active columns.
fn binary_start(d: &Design, probit: bool) -> Result<est::MlResult, String> {
    let act = d.active();
    let x: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
    if probit {
        crate::heckman::probit_fit(&x, &d.y)
    } else {
        let y = d.y.clone();
        let f = move |t: usize, v: &[f64]| {
            let p = dist::invlogit(v[0]);
            let yy = if y[t] != 0.0 { 1.0 } else { 0.0 };
            let l = if yy == 1.0 { -ln1p_exp(-v[0]) } else { -ln1p_exp(v[0]) };
            (l, vec![yy - p], vec![vec![-p * (1.0 - p)]])
        };
        let m = crate::zip::MultiIndex { x: vec![x], naux: 0, w: &d.w, fweight: false, f: &f };
        crate::choice_mixed::maximize_tight(&m, vec![0.0; act.len()], 300)
    }
}

// ---------------------------------------------------------------------------
// hetprobit
// ---------------------------------------------------------------------------

pub struct Hetprobit;

impl CommandHandler for Hetprobit {
    fn name(&self) -> &'static str {
        "hetprobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let het = xm::opt_with_sub(cmd, "het", 3).ok_or_else(|| rc_err(198, "option het() required"))?;
        let ds_ = &*ds;
        let des = xm::build_eqs(ds_, st, cmd, &[(Some(dep.as_str()), indep.as_str(), cmd.has_opt("noconstant", 3)), (None, het.0.as_str(), true)], "fpi", &[])?;
        let (d, dz) = (&des[0], &des[1]);
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let pr = binary_start(d, true)?;
        let mut out = vec!["Fitting probit model:".to_string(), String::new()];
        out.extend(est::iteration_log(&pr.iterations, "Log likelihood"));
        out.push(String::new());
        let y = d.y.clone();
        let f = |t: usize, v: &[f64]| probit_ll(y[t], v[0] / v[1].exp());
        let g = xm::numeric(&f);
        let mut b0 = pr.b.clone();
        b0.extend(vec![0.0; dz.active().len()]);
        let fit = xm::ml_fit(&[(dep.as_str(), d), ("lnsigma", dz)], &[], &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&fit.res.iterations, "Log likelihood"));
        out.push(String::new());
        let (z0, z1) = counts(d);
        let (chi2, df) = fit.wald_eq(0);
        let robust = vce != Vce::Oim;
        let (hw, hdf) = fit.wald_eq(1);
        let (ct, cv) = if robust || cmd.has_opt("waldhet", 5) { ("Wald", hw) } else { ("LR", 2.0 * (fit.ll() - pr.ll)) };
        out.extend(
            Header {
                title: "Heteroskedastic probit model",
                n: d.n_eff(),
                lines: vec![(String::new(), "Zero outcomes".into(), util::commas(z0 as usize)), (String::new(), "Nonzero outcomes".into(), util::commas(z1 as usize)), (String::new(), String::new(), String::new())],
                test: Some(("Wald".into(), chi2, df)),
                ll: Some(fit.ll()),
                robust,
                n_clust: fit.n_clust.map(|g| (g, cluster_name(&vce))),
            }
            .render()
            .into_iter()
            .filter(|l| !l.trim_end().ends_with('='))
            .collect::<Vec<_>>(),
        );
        let level = util::level(cmd, st)?;
        out.extend(xm::table("", &fit, level, None, &[]));
        out.push(format!("{ct} test of lnsigma=0: chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(hdf, 9), xm::f2(cv), xm::f4(xm::chi2p(hdf, cv))));
        let text = out.join("\n");
        let ll_c = pr.ll;
        xm::post_fit(st, "hetprobit", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("N_f", z0);
            e.set("N_s", z1);
            e.set("ll_c", ll_c);
            e.set("chi2_c", cv);
            e.set("df_m_c", hdf);
            e.set("p_c", xm::chi2p(hdf, cv));
            e.set_str("chi2_ct", ct);
            e.set_str("title", "Heteroskedastic probit model");
            e.set_str("predict", "hetprobit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// scobit
// ---------------------------------------------------------------------------

fn scobit_ll(y: f64, xb: f64, lna: f64) -> f64 {
    let a = lna.exp();
    let l1 = ln1p_exp(xb);
    if y != 0.0 {
        // ln(1 - exp(-a ln(1+e^xb)))
        let u = -a * l1;
        if u > -1e-300 { (-u).max(1e-300).ln() } else { (-u.exp_m1()).ln() }
    } else {
        -a * l1
    }
}

pub struct Scobit;

impl CommandHandler for Scobit {
    fn name(&self) -> &'static str {
        "scobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fpi", extra: vec![] })?;
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let lg = binary_start(&d, false)?;
        let mut out = vec!["Fitting logistic model:".to_string(), String::new()];
        out.extend(est::iteration_log(&lg.iterations, "Log likelihood"));
        out.push(String::new());
        let y = d.y.clone();
        let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>) {
            let a = v[1].exp();
            let l1 = ln1p_exp(v[0]);
            let p = dist::invlogit(v[0]);
            if y[t] != 0.0 {
                let e = (-a * l1).exp();
                let r = e / (1.0 - e).max(1e-300);
                (scobit_ll(y[t], v[0], v[1]), vec![r * a * p, r * a * l1])
            } else {
                (-a * l1, vec![-a * p, -a * l1])
            }
        };
        let g = xm::with_grad(&f);
        let mut b0 = lg.b.clone();
        b0.push(0.0);
        let fit = xm::ml_fit(&[("", &d)], &["lnalpha"], &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&fit.res.iterations, "Log likelihood"));
        out.push(String::new());
        let (z0, z1) = counts(&d);
        let robust = vce != Vce::Oim;
        let ll = fit.ll();
        out.push(crate::choice_mixed::hrow("Skewed logistic regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow("", "Zero outcomes", util::commas(z0 as usize)));
        out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9)), "Nonzero outcomes", util::commas(z1 as usize)));
        if let Some(gc) = fit.n_clust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(gc), cluster_name(&vce))));
        }
        out.push(String::new());
        let j = fit.idx("/:lnalpha").unwrap();
        let (la, se) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
        let alpha = la.exp();
        let level = util::level(cmd, st)?;
        out.extend(xm::table(&dep, &fit, level, None, &[Row::Tr("alpha".into(), alpha, alpha * se, la, se, f64::exp)]));
        let lr = 2.0 * (ll - lg.ll);
        if !robust {
            out.push(format!("LR test of alpha=1: chi2(1) = {}   Prob > chi2 = {}", xm::f2(lr.max(0.0)), xm::f4(xm::chi2p(1.0, lr.max(0.0)))));
            out.push("Note: Likelihood-ratio tests are recommended for inference with scobit models.".into());
        }
        let text = out.join("\n");
        let (chi2, df) = fit.wald_eq(0);
        xm::post_fit(st, "scobit", &d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("N_f", z0);
            e.set("N_s", z1);
            e.set("alpha", alpha);
            e.set("ll_c", lg.ll);
            e.set("chi2_c", lr);
            e.set_str("title", "Skewed logistic regression");
            e.set_str("predict", "scobit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// fracreg
// ---------------------------------------------------------------------------

pub struct Fracreg;

fn frac_ll(y: f64, p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let mut l = 0.0;
    if y > 0.0 {
        l += y * p.ln();
    }
    if y < 1.0 {
        l += (1.0 - y) * (1.0 - p).max(1e-300).ln();
    }
    l
}

/// Fractional-response quasi log likelihood with the probit or logit
/// probability computed on the stable side.
fn frac_ll_index(y: f64, z: f64, logit: bool) -> f64 {
    let (lp, lq) = if logit { (-ln1p_exp(-z), -ln1p_exp(z)) } else { (crate::xtnl::ln_phi(z), crate::xtnl::ln_phi(-z)) };
    let mut l = 0.0;
    if y > 0.0 {
        l += y * lp;
    }
    if y < 1.0 {
        l += (1.0 - y) * lq;
    }
    l
}

impl CommandHandler for Fracreg {
    fn name(&self) -> &'static str {
        "fracreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (model, rest) = xm::first_word(&cmd.main);
        let model = model.to_lowercase();
        if !matches!(model.as_str(), "logit" | "probit" | "hetprobit") {
            return Err(rc_err(198, "fracreg: model must be logit, probit or hetprobit"));
        }
        let (dep, indep) = xm::first_word(&rest);
        let dep = xm::var_name(ds, &dep)?;
        let het = xm::opt_with_sub(cmd, "het", 3);
        if model == "hetprobit" && het.is_none() {
            return Err(rc_err(198, "option het() required"));
        }
        let hv = het.clone().unwrap_or_default().0;
        let ds_ = &*ds;
        let mut specs = vec![(Some(dep.as_str()), indep.as_str(), cmd.has_opt("noconstant", 3))];
        if model == "hetprobit" {
            specs.push((None, hv.as_str(), true));
        }
        let des = xm::build_eqs(ds_, st, cmd, &specs, "fpi", &[])?;
        let d = &des[0];
        if d.y.iter().any(|v| *v < 0.0 || *v > 1.0) {
            return Err(rc_err(459, format!("{dep} has values outside [0,1]")));
        }
        let mut vce = est::parse_vce(cmd, Vce::Robust)?;
        if vce == Vce::Ols {
            vce = Vce::Robust;
        }
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let logit = model == "logit";
        let y = d.y.clone();
        let fw = d.wkind == Some(WeightKind::F);
        let fit = if model == "hetprobit" {
            let f = |t: usize, v: &[f64]| frac_ll_index(y[t], v[0] / v[1].exp(), false);
            let g = xm::numeric(&f);
            let base = {
                let f0 = |t: usize, v: &[f64]| frac_ll_index(y[t], v[0], false);
                let g0 = xm::numeric(&f0);
                xm::ml_fit(&[(dep.as_str(), d)], &[], &g0, vec![0.0; d.active().len()], &d.w, fw, &Vce::Oim, None)?
            };
            let mut b0 = base.res.b.clone();
            b0.extend(vec![0.0; des[1].active().len()]);
            xm::ml_fit(&[(dep.as_str(), d), ("lnsigma", &des[1])], &[], &g, b0, &d.w, fw, &vce, cluster.as_deref())?
        } else {
            let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>) {
                let z = v[0];
                let l = frac_ll_index(y[t], z, logit);
                let gd = if logit {
                    y[t] - dist::invlogit(z)
                } else {
                    let (_, m1) = crate::choice_mixed::ln_phi_d(z);
                    let (_, m0) = crate::choice_mixed::ln_phi_d(-z);
                    y[t] * m1 - (1.0 - y[t]) * m0
                };
                (l, vec![gd])
            };
            let g = xm::with_grad(&f);
            xm::ml_fit(&[("", d)], &[], &g, vec![0.0; d.active().len()], &d.w, fw, &vce, cluster.as_deref())?
        };
        // constant-only quasi likelihood
        let sw: f64 = d.w.iter().sum();
        let ybar = (0..d.n()).map(|t| d.w[t] * y[t]).sum::<f64>() / sw;
        let ll0: f64 = (0..d.n()).map(|t| d.w[t] * frac_ll(y[t], ybar)).sum();
        let r2p = 1.0 - fit.ll() / ll0;
        let (chi2, df) = fit.wald_eq(0);
        let title = match model.as_str() {
            "logit" => "Fractional logistic regression",
            "probit" => "Fractional probit regression",
            _ => "Fractional heteroskedastic probit regression",
        };
        let mut out = est::iteration_log(&fit.res.iterations, "Log pseudolikelihood");
        out.push(String::new());
        out.push(crate::choice_mixed::hrow(title, "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(crate::choice_mixed::hrow(&format!("Log pseudolikelihood = {}", fmt_g(fit.ll(), 9)), "Pseudo R2", xm::f4(r2p)));
        if let Some(gc) = fit.n_clust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(gc), cluster_name(&vce))));
        }
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let eform = if logit && cmd.has_opt("or", 2) { Some("Odds ratio") } else { None };
        out.extend(xm::table(&dep, &fit, level, eform, &[]));
        let text = out.join("\n");
        xm::post_fit(st, "fracreg", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("ll_0", ll0);
            e.set("r2_p", r2p);
            e.set_str("model", model.clone());
            e.set_str("link", model.clone());
            e.set_str("title", title);
            e.set_str("predict", "fracreg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// betareg
// ---------------------------------------------------------------------------

pub struct Betareg;

fn beta_ll(y: f64, mu: f64, phi: f64) -> f64 {
    let mu = mu.clamp(1e-12, 1.0 - 1e-12);
    let a = mu * phi;
    let b = (1.0 - mu) * phi;
    ln_gamma(phi) - ln_gamma(a) - ln_gamma(b) + (a - 1.0) * y.ln() + (b - 1.0) * (1.0 - y).ln()
}

impl CommandHandler for Betareg {
    fn name(&self) -> &'static str {
        "betareg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let link = cmd.opt("link", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "logit".into());
        if !matches!(link.as_str(), "logit" | "probit" | "cloglog" | "loglog") {
            return Err(rc_err(198, "link() must be logit, probit, cloglog or loglog"));
        }
        let sv = xm::opt_with_sub(cmd, "scale", 2).unwrap_or_default();
        let ds_ = &*ds;
        let des = xm::build_eqs(ds_, st, cmd, &[(Some(dep.as_str()), indep.as_str(), cmd.has_opt("noconstant", 3)), (None, sv.0.as_str(), sv.1.contains("nocons"))], "fpi", &[])?;
        let (d, dz) = (&des[0], &des[1]);
        if d.y.iter().any(|v| *v <= 0.0 || *v >= 1.0) {
            return Err(rc_err(459, format!("{dep} must be greater than zero and less than one")));
        }
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let y = d.y.clone();
        let lk = link.clone();
        let f = move |t: usize, v: &[f64]| beta_ll(y[t], inv_link(&lk, v[0]), v[1].exp());
        let g = xm::numeric(&f);
        let fw = d.wkind == Some(WeightKind::F);
        // start: OLS of the link-transformed outcome, method-of-moments precision
        let ty: Vec<f64> = d
            .y
            .iter()
            .map(|&p| match link.as_str() {
                "probit" => dist::invnormal(p),
                "cloglog" => (-(1.0 - p).ln()).ln(),
                "loglog" => -(-p.ln()).ln(),
                _ => (p / (1.0 - p)).ln(),
            })
            .collect();
        let (bs, _) = xm::ols_start(d, &ty);
        let m = util::mean(&d.y);
        let var = util::variance(&d.y);
        let phi0 = (m * (1.0 - m) / var - 1.0).max(0.5);
        let mut b0 = bs.clone();
        let zact = dz.active();
        let mut gz = vec![0.0; zact.len()];
        if let Some(c) = zact.iter().position(|&j| dz.names[j] == "_cons") {
            gz[c] = phi0.ln();
        }
        b0.extend(gz.iter().cloned());
        let fit = xm::ml_fit(&[(dep.as_str(), d), ("scale", dz)], &[], &g, b0, &d.w, fw, &vce, cluster.as_deref())?;
        // constant-only mean model for the LR test
        let mut ll0 = SYSMISS;
        if let Some(d0) = xm::const_only(d) {
            let mut b00 = vec![match link.as_str() {
                "probit" => dist::invnormal(m),
                _ => (m / (1.0 - m)).ln(),
            }];
            b00.extend(gz.iter().cloned());
            if let Ok(f0) = xm::ml_fit(&[(dep.as_str(), &d0), ("scale", dz)], &[], &g, b00, &d.w, fw, &Vce::Oim, None) {
                ll0 = f0.ll();
            }
        }
        let robust = vce != Vce::Oim;
        let (wchi, df) = fit.wald_eq(0);
        let (tn, chi2) = if robust || is_missing(ll0) { ("Wald", wchi) } else { ("LR", 2.0 * (fit.ll() - ll0)) };
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(crate::choice_mixed::hrow("Beta regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow("", &format!("{tn} chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(String::new());
        let ltxt = match link.as_str() {
            "probit" => "g(u) = invnorm(u)          [Probit]",
            "cloglog" => "g(u) = log(-log(1-u))      [Complementary log-log]",
            "loglog" => "g(u) = -log(-log(u))       [Log-log]",
            _ => "g(u) = log(u/(1-u))       [Logit]",
        };
        out.push(format!("Link function  : {ltxt}"));
        out.push("Slink function : g(u) = log(u)             [Log]".into());
        out.push(String::new());
        out.push(format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)));
        out.push(String::new());
        let level = util::level(cmd, st)?;
        out.extend(xm::table("", &fit, level, None, &[]));
        let text = out.join("\n");
        xm::post_fit(st, "betareg", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, tn, chi2, df);
            e.set("ll_0", ll0);
            e.set_str("link", link.clone());
            e.set_str("slink", "log");
            e.set_str("title", "Beta regression");
            e.set_str("predict", "betareg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// binreg
// ---------------------------------------------------------------------------

pub struct Binreg;

impl CommandHandler for Binreg {
    fn name(&self) -> &'static str {
        "binreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let (link, label, note) = if cmd.has_opt("rr", 2) {
            ("log", "Risk ratio", Some("Note: _cons estimates baseline risk."))
        } else if cmd.has_opt("hr", 2) {
            ("logc", "Hlth ratio", Some("Note: _cons estimates baseline health (probability of no disease)."))
        } else if cmd.has_opt("rd", 2) {
            ("identity", "Risk diff.", None)
        } else {
            ("logit", "Odds ratio", Some("Note: _cons estimates baseline odds."))
        };
        let nopt = cmd.opt("n", 1).map(|s| s.trim().to_string());
        let mut extra = Vec::new();
        if let Some(n) = &nopt {
            if n.parse::<f64>().is_err() {
                extra.push(xm::var_name(ds, n)?);
            }
        }
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fai", extra: extra.clone() })?;
        let nn: Vec<f64> = match &nopt {
            None => vec![1.0; d.n()],
            Some(s) => match s.parse::<f64>() {
                Ok(v) => vec![v; d.n()],
                Err(_) => {
                    let c = util::col(ds, &extra[0])?;
                    d.rows.iter().map(|&i| c[i]).collect()
                }
            },
        };
        for t in 0..d.n() {
            if d.y[t] < 0.0 || d.y[t] > nn[t] {
                return Err(rc_err(499, format!("{dep} must be in [0, n]")));
            }
        }
        let y = d.y.clone();
        let mu_of = |eta: f64| -> f64 {
            match link {
                "log" => eta.exp(),
                "logc" => 1.0 - eta.exp(),
                "identity" => eta,
                _ => dist::invlogit(eta),
            }
        };
        let dmu = |eta: f64| -> f64 {
            match link {
                "log" => eta.exp(),
                "logc" => -eta.exp(),
                "identity" => 1.0,
                _ => {
                    let p = dist::invlogit(eta);
                    p * (1.0 - p)
                }
            }
        };
        // IRLS (Fisher scoring) on proportions
        let act = d.active();
        let ka = act.len();
        let n = d.n();
        let mut eta: Vec<f64> = (0..n)
            .map(|t| {
                let p = ((y[t] + 0.5) / (nn[t] + 1.0)).clamp(1e-4, 1.0 - 1e-4);
                match link {
                    "log" => p.ln(),
                    "logc" => (1.0 - p).ln(),
                    "identity" => p,
                    _ => (p / (1.0 - p)).ln(),
                }
            })
            .collect();
        let mut b = vec![0.0; ka];
        let mut info;
        let dev_of = |eta: &[f64]| -> f64 {
            (0..n)
                .map(|t| {
                    let mu = mu_of(eta[t]) * nn[t];
                    let (yy, m) = (y[t], nn[t]);
                    let a = if yy > 0.0 { yy * (yy / mu).ln() } else { 0.0 };
                    let c = if m - yy > 0.0 { (m - yy) * ((m - yy) / (m - mu)).ln() } else { 0.0 };
                    2.0 * d.w[t] * (a + c)
                })
                .sum()
        };
        let mut dev_old = f64::INFINITY;
        let mut log = Vec::new();
        for _ in 0..200 {
            info = Matrix::zeros(ka, ka);
            let mut rhs = vec![0.0; ka];
            for t in 0..n {
                let p = mu_of(eta[t]).clamp(1e-10, 1.0 - 1e-10);
                let g = dmu(eta[t]);
                let wt = d.w[t] * nn[t] * g * g / (p * (1.0 - p));
                let z = eta[t] + (y[t] / nn[t] - p) / g;
                for a in 0..ka {
                    let xa = d.cols[act[a]][t];
                    rhs[a] += wt * xa * z;
                    for c in 0..ka {
                        info[(a, c)] += wt * xa * d.cols[act[c]][t];
                    }
                }
            }
            let (ii, _) = info.sym_inverse(1e-14);
            b = (0..ka).map(|a| (0..ka).map(|c| ii[(a, c)] * rhs[c]).sum()).collect();
            for t in 0..n {
                let e: f64 = (0..ka).map(|a| d.cols[act[a]][t] * b[a]).sum();
                // keep the mean inside (0,1)
                eta[t] = match link {
                    "log" => e.min(-1e-8),
                    "logc" => e.min(-1e-8),
                    "identity" => e.clamp(1e-8, 1.0 - 1e-8),
                    _ => e,
                };
            }
            let dev = dev_of(&eta);
            log.push(dev);
            if (dev_old - dev).abs() < 1e-8 * (dev.abs() + 1e-8) {
                break;
            }
            dev_old = dev;
        }
        // final information at the estimate
        info = Matrix::zeros(ka, ka);
        let mut scores = Vec::new();
        let mut pearson = 0.0;
        for t in 0..n {
            let p = mu_of(eta[t]).clamp(1e-10, 1.0 - 1e-10);
            let g = dmu(eta[t]);
            let wt = d.w[t] * nn[t] * g * g / (p * (1.0 - p));
            pearson += d.w[t] * (y[t] - nn[t] * p).powi(2) / (nn[t] * p * (1.0 - p));
            for a in 0..ka {
                for c in 0..ka {
                    info[(a, c)] += wt * d.cols[act[a]][t] * d.cols[act[c]][t];
                }
            }
            let s = (y[t] - nn[t] * p) * g / (p * (1.0 - p));
            scores.push(act.iter().map(|&j| d.w[t] * s * d.cols[j][t]).collect::<Vec<f64>>());
        }
        let dev = dev_of(&eta);
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let res = est::MlResult { b: b.clone(), ll: SYSMISS, info, scores, iterations: log.clone(), converged: true };
        let fit = xm::finish_fit(&[("", &d)], &[], res, &vce, cluster.as_deref())?;
        let df_r = d.n_eff() - ka as f64;
        let ll: f64 = (0..n)
            .map(|t| {
                let p = mu_of(eta[t]).clamp(1e-300, 1.0 - 1e-16);
                d.w[t] * (ln_gamma(nn[t] + 1.0) - ln_gamma(y[t] + 1.0) - ln_gamma(nn[t] - y[t] + 1.0) + y[t] * p.ln() + (nn[t] - y[t]) * (1.0 - p).max(1e-300).ln())
            })
            .sum();
        let bic = dev - df_r * d.n_eff().ln();
        let mut out = Vec::new();
        for (i, dv) in log.iter().enumerate() {
            out.push(format!("Iteration {}:  Deviance = {}", i + 1, fmt_g(*dv, 9)));
        }
        out.push(String::new());
        let nlab = nopt.clone().unwrap_or_else(|| "1".into());
        let lk = match link {
            "log" => format!("g(u) = ln(u/{nlab})              [Log]"),
            "logc" => format!("g(u) = ln(1-u/{nlab})            [Log complement]"),
            "identity" => format!("g(u) = u/{nlab}                  [Identity]"),
            _ => format!("g(u) = ln(u/({nlab}-u))          [Logit]"),
        };
        out.push(format!("{:<48}{:<17} = {:>10}", "Generalized linear models", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(format!("{:<48}{:<17} = {:>10}", "Optimization     : MQL Fisher scoring", "Residual df", fmt_g(df_r, 9)));
        out.push(format!("{:<48}{:<17} = {:>10}", "                   (IRLS EIM)", "Scale parameter", "1"));
        out.push(format!("{:<48}{:<17} = {:>10}", format!("Deviance         = {:>15}", fmt_g(dev, 9)), "(1/df) Deviance", fmt_g(dev / df_r, 7)));
        out.push(format!("{:<48}{:<17} = {:>10}", format!("Pearson          = {:>15}", fmt_g(pearson, 10)), "(1/df) Pearson", fmt_g(pearson / df_r, 7)));
        out.push(String::new());
        out.push(format!("Variance function: V(u) = u*(1-u/{nlab})          [Binomial]"));
        out.push(format!("Link function    : {lk}"));
        out.push(String::new());
        out.push(format!("{:<48}{:<17} = {:>10}", "", "BIC", fmt_g(bic, 7)));
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let coefs = cmd.has_opt("coefficients", 4);
        let eform = if link == "identity" || coefs { None } else { Some(label) };
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &fit.names, b: &fit.b, v: &fit.v, omitted: &fit.omitted, is_base: &fit.is_base, df: None, level, eform, coef_title: if link == "identity" { "Risk diff." } else { "Coefficient" } });
        if link == "identity" && tbl.len() > 1 {
            tbl[1] = tbl[1].replacen("Coefficient", " Risk diff.", 1);
        }
        out.extend(tbl);
        if let (Some(nt), true) = (note, eform.is_some()) {
            out.push(nt.to_string());
        }
        let text = out.join("\n");
        let lk2 = link.to_string();
        xm::post_fit(st, "binreg", &d, &fit, &vce, &|e: &mut Results| {
            e.set("ll", ll);
            e.set("deviance", dev);
            e.set("deviance_p", pearson);
            e.set("df", df_r);
            e.set("bic", bic);
            e.set("phi", 1.0);
            e.set_str("link", lk2.clone());
            e.set_str("title", "Generalized linear models");
            e.set_str("predict", "binreg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// cpoisson
// ---------------------------------------------------------------------------

/// ln P(Y <= k) for Y ~ Poisson(mu).
fn ln_pois_cdf(k: f64, mu: f64) -> f64 {
    if k < 0.0 {
        return f64::NEG_INFINITY;
    }
    gamma_ur(k.floor() + 1.0, mu).max(1e-300).ln()
}

pub struct Cpoisson;

impl CommandHandler for Cpoisson {
    fn name(&self) -> &'static str {
        "cpoisson"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let lim = |name: &str| -> Result<Option<(Option<f64>, Option<String>)>, String> {
            match cmd.opt(name, 2) {
                None => Ok(None),
                Some(s) => match s.trim().parse::<f64>() {
                    Ok(v) => Ok(Some((Some(v), None))),
                    Err(_) => Ok(Some((None, Some(xm::var_name(ds, s)?)))),
                },
            }
        };
        let llim = lim("ll")?;
        let ulim = lim("ul")?;
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fpi", extra: vec![] })?;
        let n = d.n();
        let col_of = |l: &Option<(Option<f64>, Option<String>)>, def: f64| -> Result<Vec<f64>, String> {
            Ok(match l {
                None => vec![def; n],
                Some((Some(v), _)) => vec![*v; n],
                Some((None, Some(v))) => {
                    let c = util::col(ds, v)?;
                    d.rows.iter().map(|&i| if is_missing(c[i]) { def } else { c[i] }).collect()
                }
                _ => vec![def; n],
            })
        };
        let lo = col_of(&llim, 0.0)?;
        let hi = col_of(&ulim, f64::INFINITY)?;
        let y = d.y.clone();
        let (mut nu, mut nl, mut nr) = (0.0, 0.0, 0.0);
        let mut kind = vec![0u8; n];
        for t in 0..n {
            if y[t] < 0.0 {
                return Err(rc_err(459, format!("{dep} must be nonnegative")));
            }
            let w = if d.wkind == Some(WeightKind::F) { d.w[t] } else { 1.0 };
            if llim.is_some() && y[t] <= lo[t] {
                kind[t] = 1;
                nl += w;
            } else if y[t] >= hi[t] {
                kind[t] = 2;
                nr += w;
            } else {
                nu += w;
            }
        }
        let f = |t: usize, v: &[f64]| -> f64 {
            let mu = v[0].exp();
            match kind[t] {
                1 => ln_pois_cdf(lo[t], mu),
                2 => {
                    // ln P(Y >= hi) = ln P(hi; mu) lower regularized gamma
                    statrs::function::gamma::gamma_lr(hi[t].ceil(), mu).max(1e-300).ln()
                }
                _ => y[t] * v[0] - mu - ln_gamma(y[t] + 1.0),
            }
        };
        let g = xm::numeric(&f);
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let ly: Vec<f64> = y.iter().map(|v| (v + 0.5).ln()).collect();
        let (b0, _) = xm::ols_start(&d, &ly);
        let fw = d.wkind == Some(WeightKind::F);
        let fit = xm::ml_fit(&[("", &d)], &[], &g, b0, &d.w, fw, &vce, cluster.as_deref())?;
        let mut ll0 = SYSMISS;
        if let Some(d0) = xm::const_only(&d) {
            let m = (util::mean(&y) + 0.1).ln();
            if let Ok(f0) = xm::ml_fit(&[("", &d0)], &[], &g, vec![m], &d.w, fw, &Vce::Oim, None) {
                ll0 = f0.ll();
            }
        }
        let robust = vce != Vce::Oim;
        let (wchi, df) = fit.wald_eq(0);
        let (tn, chi2) = if robust || is_missing(ll0) { ("Wald", wchi) } else { ("LR", 2.0 * (fit.ll() - ll0)) };
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        let lname = |l: &Option<(Option<f64>, Option<String>)>, def: &str| match l {
            None => def.to_string(),
            Some((Some(v), _)) => fmt_g(*v, 8),
            Some((_, Some(v))) => v.clone(),
            _ => def.to_string(),
        };
        out.push(crate::choice_mixed::hrow("Censored Poisson regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow("", "Uncensored", util::commas(nu as usize)));
        out.push(crate::choice_mixed::hrow(&format!("Limits: Lower = {}", lname(&llim, "0")), "Left-censored", util::commas(nl as usize)));
        out.push(crate::choice_mixed::hrow(&format!("        Upper = {}", lname(&ulim, "+inf")), "Right-censored", util::commas(nr as usize)));
        out.push(String::new());
        out.push(crate::choice_mixed::hrow("", &format!("{tn} chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)), "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let eform = if cmd.has_opt("irr", 3) { Some("IRR") } else { None };
        out.extend(xm::table(&dep, &fit, level, eform, &[]));
        let text = out.join("\n");
        xm::post_fit(st, "cpoisson", &d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, tn, chi2, df);
            e.set("ll_0", ll0);
            e.set("N_unc", nu);
            e.set("N_lc", nl);
            e.set("N_rc", nr);
            e.set_str("title", "Censored Poisson regression");
            e.set_str("predict", "cpoisson_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// hetoprobit
// ---------------------------------------------------------------------------

pub struct Hetoprobit;

fn oprobit_ll(k: usize, ncat: usize, xb: f64, sig: f64, cuts: &[f64]) -> f64 {
    let hi = if k + 1 < ncat { (cuts[k] - xb) / sig } else { f64::INFINITY };
    let lo = if k > 0 { (cuts[k - 1] - xb) / sig } else { f64::NEG_INFINITY };
    let p = if lo > 0.0 { xm::ncdf(-lo) - xm::ncdf(-hi) } else { xm::ncdf(hi) - xm::ncdf(lo) };
    p.max(1e-300).ln()
}

impl CommandHandler for Hetoprobit {
    fn name(&self) -> &'static str {
        "hetoprobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let het = xm::opt_with_sub(cmd, "het", 3).ok_or_else(|| rc_err(198, "option het() required"))?;
        let ds_ = &*ds;
        let des = xm::build_eqs(ds_, st, cmd, &[(Some(dep.as_str()), indep.as_str(), true), (None, het.0.as_str(), true)], "fpi", &[])?;
        let (d, dz) = (&des[0], &des[1]);
        let mut cats: Vec<f64> = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let nc = cats.len();
        if nc < 2 {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        let kidx: Vec<usize> = d.y.iter().map(|v| cats.iter().position(|c| c == v).unwrap()).collect();
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let fw = d.wkind == Some(WeightKind::F);
        // ordered probit start
        let f0 = |t: usize, v: &[f64]| oprobit_ll(kidx[t], nc, v[0], 1.0, &v[1..]);
        let g0 = xm::numeric(&f0);
        let cutnames: Vec<String> = (1..nc).map(|i| format!("cut{i}")).collect();
        let cutrefs: Vec<&str> = cutnames.iter().map(|s| s.as_str()).collect();
        let sw: f64 = d.w.iter().sum();
        let mut b0 = vec![0.0; d.active().len()];
        let mut cum = 0.0;
        for c in 0..nc - 1 {
            cum += (0..d.n()).filter(|&t| kidx[t] == c).map(|t| d.w[t]).sum::<f64>();
            b0.push(dist::invnormal((cum / sw).clamp(1e-6, 1.0 - 1e-6)));
        }
        let op = xm::ml_fit(&[("", d)], &cutrefs, &g0, b0, &d.w, fw, &Vce::Oim, None)?;
        let mut out = vec!["Fitting ordered probit model:".to_string(), String::new()];
        out.extend(est::iteration_log(&op.res.iterations, "Log likelihood"));
        out.push(String::new());
        let f = |t: usize, v: &[f64]| oprobit_ll(kidx[t], nc, v[0], v[1].exp(), &v[2..]);
        let g = xm::numeric(&f);
        let kx = d.active().len();
        let mut b1: Vec<f64> = op.res.b[..kx].to_vec();
        b1.extend(vec![0.0; dz.active().len()]);
        b1.extend(op.res.b[kx..].iter().cloned());
        let fit = xm::ml_fit(&[(dep.as_str(), d), ("lnsigma", dz)], &cutrefs, &g, b1, &d.w, fw, &vce, cluster.as_deref())?;
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&fit.res.iterations, "Log likelihood"));
        out.push(String::new());
        let robust = vce != Vce::Oim;
        // model test: LR against the constant-only (cutpoints) model with het
        let (wchi, df) = fit.wald_eq(0);
        // model test: LR against the model with cutpoints and the variance equation only
        let ll0 = {
            let mut d0 = d.clone();
            for o in d0.omitted.iter_mut() {
                *o = true;
            }
            let mut cum = 0.0;
            let mut b00: Vec<f64> = fit.res.b[kx..kx + dz.active().len()].to_vec();
            for c in 0..nc - 1 {
                cum += (0..d.n()).filter(|&t| kidx[t] == c).map(|t| d.w[t]).sum::<f64>();
                b00.push(dist::invnormal(cum / sw));
            }
            let f0 = |t: usize, v: &[f64]| oprobit_ll(kidx[t], nc, v[0], v[1].exp(), &v[2..]);
            let g0 = xm::numeric(&f0);
            xm::ml_fit(&[("", &d0), ("lnsigma", dz)], &cutrefs, &g0, b00, &d.w, fw, &Vce::Oim, None).map(|f| f.ll()).unwrap_or(SYSMISS)
        };
        let (tn, chi2) = if robust { ("Wald", wchi) } else { ("LR", 2.0 * (fit.ll() - ll0)) };
        let _ = tn;
        out.extend(Header { title: "Heteroskedastic ordered probit regression", n: d.n_eff(), lines: vec![], test: Some((if robust { "Wald" } else { "LR" }.into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: fit.n_clust.map(|g| (g, cluster_name(&vce))) }.render());
        let level = util::level(cmd, st)?;
        // cutpoints displayed without tests: render as Ci rows
        let crit = crate::choice_mixed::zcrit(level);
        let mut fit2 = xm::Fit { names: fit.names.clone(), b: fit.b.clone(), v: fit.v.clone(), omitted: fit.omitted.clone(), is_base: fit.is_base.clone(), res: est::MlResult { b: fit.res.b.clone(), ll: fit.res.ll, info: fit.res.info.clone(), scores: vec![], iterations: vec![], converged: true }, bt: vec![], vt: Matrix::zeros(0, 0), pos: vec![], n_clust: None, neq: 2, naux: 0 };
        let ncoef = fit.names.iter().filter(|n| !n.starts_with("/:")).count();
        fit2.names.truncate(ncoef);
        let mut extra = Vec::new();
        for j in ncoef..fit.names.len() {
            let (bj, se) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
            extra.push(Row::Ci(format!("/{}", &fit.names[j][2..]), bj, se, bj - crit * se, bj + crit * se));
        }
        let mut tbl = xm::table("", &fit2, level, None, &[]);
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(crate::choice_mixed::sep(lw));
        tbl.extend(xm::extra_rows(lw, level, &extra, false));
        tbl.push(bar);
        out.extend(tbl);
        let (hw, hdf) = fit.wald_eq(1);
        let (ct, cv) = if robust || cmd.has_opt("waldhet", 5) { ("Wald", hw) } else { ("LR", 2.0 * (fit.ll() - op.ll())) };
        out.push(format!("{ct} test of lnsigma=0: chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(hdf, 9), xm::f2(cv), xm::f4(xm::chi2p(hdf, cv))));
        let text = out.join("\n");
        let catm = Matrix::row_vector(&cats);
        xm::post_fit(st, "hetoprobit", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, if robust { "Wald" } else { "LR" }, chi2, df);
            e.set("ll_0", ll0);
            e.set("k_cat", nc as f64);
            e.set("chi2_c", cv);
            e.set("df_m_c", hdf);
            e.set_mat("cat", catm.clone());
            e.set_str("title", "Heteroskedastic ordered probit regression");
            e.set_str("predict", "hetoprobit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// mprobit
// ---------------------------------------------------------------------------

/// Gauss-Hermite nodes/weights for integrals against the standard normal.
fn gh_normal(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = Matrix::zeros(n, n);
    for i in 1..n {
        let b = (i as f64).sqrt();
        j[(i, i - 1)] = b;
        j[(i - 1, i)] = b;
    }
    let (vals, vecs) = j.sym_eigen();
    let mut p: Vec<(f64, f64)> = (0..n).map(|k| (vals[k], vecs[(0, k)] * vecs[(0, k)])).collect();
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    (p.iter().map(|x| x.0).collect(), p.iter().map(|x| x.1).collect())
}

thread_local! {
    static GH: (Vec<f64>, Vec<f64>) = gh_normal(40);
}

/// Pr(choose k) with independent standard normal latent errors.
pub fn mprobit_prob(eta: &[f64], k: usize) -> f64 {
    GH.with(|(x, w)| {
        let mut s = 0.0;
        for (e, wt) in x.iter().zip(w) {
            let mut p = *wt;
            for (j, ej) in eta.iter().enumerate() {
                if j != k {
                    p *= dist::normal(e + eta[k] - ej);
                }
            }
            s += p;
        }
        s
    })
}

pub struct Mprobit;

impl CommandHandler for Mprobit {
    fn name(&self) -> &'static str {
        "mprobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fpi", extra: vec![] })?;
        let mut cats: Vec<f64> = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let nc = cats.len();
        if nc < 2 {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        let kidx: Vec<usize> = d.y.iter().map(|v| cats.iter().position(|c| c == v).unwrap()).collect();
        let freq: Vec<f64> = (0..nc).map(|c| (0..d.n()).filter(|&t| kidx[t] == c).map(|t| d.w[t]).sum()).collect();
        let base = match util::opt_num(cmd, "baseoutcome", 4)? {
            Some(b) => cats.iter().position(|c| *c == b).ok_or_else(|| rc_err(198, "baseoutcome() not found"))?,
            None => (0..nc).fold(0, |a, c| if freq[c] > freq[a] { c } else { a }),
        };
        let probitparam = cmd.has_opt("probitparam", 6);
        let scale = if probitparam { std::f64::consts::SQRT_2 } else { 1.0 };
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let f = |t: usize, v: &[f64]| -> f64 {
            let mut eta = Vec::with_capacity(nc);
            let mut q = 0;
            for j in 0..nc {
                if j == base {
                    eta.push(0.0);
                } else {
                    eta.push(v[q] * scale);
                    q += 1;
                }
            }
            mprobit_prob(&eta, kidx[t]).max(1e-300).ln()
        };
        let g = xm::numeric(&f);
        let labels: Vec<String> = cats.iter().map(|c| outcome_label(ds, &dep, *c)).collect();
        let eqn: Vec<String> = (0..nc).filter(|&j| j != base).map(|j| labels[j].clone()).collect();
        let eqs: Vec<(&str, &Design)> = eqn.iter().map(|e| (e.as_str(), &d)).collect();
        let ka = d.active().len();
        let mut b0 = Vec::new();
        for j in (0..nc).filter(|&j| j != base) {
            let mut bj = vec![0.0; ka];
            if let Some(c) = d.active().iter().position(|&i| d.names[i] == "_cons") {
                bj[c] = (freq[j] / freq[base]).ln() * 0.6 / scale;
            }
            b0.extend(bj);
        }
        let fit = xm::ml_fit(&eqs, &[], &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        let robust = vce != Vce::Oim;
        // Wald test of all non-constant coefficients
        let idx: Vec<usize> = (0..fit.pos.len()).filter(|&a| !fit.names[fit.pos[a]].ends_with(":_cons")).collect();
        let (chi2, df) = crate::heckman::wald(&fit.bt, &fit.vt, &idx);
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        out.extend(Header { title: "Multinomial probit regression", n: d.n_eff(), lines: vec![], test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: fit.n_clust.map(|g| (g, cluster_name(&vce))) }.render());
        let level = util::level(cmd, st)?;
        let mut tbl = xm::table(&dep, &fit, level, None, &[]);
        // base outcome line after the header rule
        let lw = tbl.last().map_or(12, |b| b.len().saturating_sub(66));
        tbl.insert(3, format!("{:<lw$} |  (base outcome)", crate::summarize::abbrev(&labels[base], lw)));
        tbl.insert(4, format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        out.extend(tbl);
        let text = out.join("\n");
        let outm = Matrix::row_vector(&cats);
        xm::post_fit(st, "mprobit", &d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("k_out", nc as f64);
            e.set("i_base", (base + 1) as f64);
            e.set("k_eq", (nc - 1) as f64);
            e.set_mat("out", outm.clone());
            e.set_str("title", "Multinomial probit regression");
            e.set_str("predict", "mprobit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

fn outcome_label(ds: &DataSet, dep: &str, v: f64) -> String {
    if let Some(var) = ds.get_var(dep) {
        if !var.value_label_name.is_empty() {
            if let Some(l) = ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(v)) {
                return l.replace(' ', "_");
            }
        }
    }
    fmt_g(v, 12)
}
