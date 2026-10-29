//! Models with continuous endogenous regressors: `ivprobit` (maximum
//! likelihood and Newey's two-step estimator), `ivtobit`, `ivfprobit`,
//! `ivpoisson` (GMM and control function) and the control-function
//! estimators `cfregress` and `cfprobit`.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xm_gmm::{self as gm, SKind};
use crate::xmodels::{self as xm, Header, Row};

pub fn handlers() -> xm::Handlers {
    vec![
        ("ivprobit", Box::new(IvMl { kind: Kind::Probit })),
        ("ivtobit", Box::new(IvMl { kind: Kind::Tobit })),
        ("ivfprobit", Box::new(IvMl { kind: Kind::Frac })),
        ("ivpoisson", Box::new(Ivpoisson)),
        ("cfregress", Box::new(Cf { probit: false })),
        ("cfprobit", Box::new(Cf { probit: true })),
    ]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    match ecmd {
        "ivprobit" | "ivfprobit" | "cfprobit" => {
            let twostep = st.e.macros.get("method").is_some_and(|m| m == "twostep");
            let msg = if ecmd == "ivfprobit" { "(option cm assumed; conditional mean)".to_string() } else { format!("(option pr assumed; Pr({dep}))") };
            Some(xm::generic_predict(ds, st, cmd, &["pr", "cm", "xb", "stdp"], &msg, &|s, c| match s {
                "xb" => c.eta[0],
                _ if twostep => SYSMISS,
                _ => dist::normal(c.eta[0]),
            }))
        }
        "ivtobit" => Some(xm::generic_predict(ds, st, cmd, &["xb", "stdp"], "(option xb assumed; fitted values)", &|_, c| c.eta[0])),
        "ivpoisson" => Some(xm::generic_predict(ds, st, cmd, &["n", "xb", "stdp"], "(option n assumed; predicted number of events)", &|s, c| if s == "xb" { c.eta[0] } else { c.eta[0].exp() })),
        "cfregress" => Some(xm::generic_predict(ds, st, cmd, &["xb", "stdp"], "(option xb assumed; fitted values)", &|_, c| c.eta[0])),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

pub struct IvData {
    pub dep: String,
    pub endog: Vec<String>,
    pub exog: String,
    pub inst: String,
    /// dep on endog + exog
    pub d: Design,
    /// exog + instruments (first-stage regressors)
    pub dz: Design,
    /// endogenous variable values over the sample
    pub y2: Vec<Vec<f64>>,
}

pub fn iv_data(ds: &DataSet, st: &Store, cmd: &Command, main: &str, weights: &str) -> Result<IvData, String> {
    let (outside, endog, inst) = xm::iv_parts(main)?;
    let (dep, exog) = xm::first_word(&outside);
    let dep = xm::var_name(ds, &dep)?;
    let endog_v = crate::varlist::expand(ds, &endog)?;
    let nocons = cmd.has_opt("noconstant", 3);
    let xtext = format!("{} {}", endog_v.join(" "), exog);
    let ztext = format!("{} {}", exog, inst);
    let des = xm::build_eqs(ds, st, cmd, &[(Some(dep.as_str()), xtext.as_str(), nocons), (None, ztext.as_str(), nocons)], weights, &endog_v)?;
    let d = des[0].clone();
    let dz = des[1].clone();
    let mut y2 = Vec::new();
    for v in &endog_v {
        let j = d.names.iter().position(|n| n == v).ok_or_else(|| rc_err(198, format!("{v} not found")))?;
        y2.push(d.cols[j].clone());
    }
    let ka = d.active().len();
    let kz = dz.active().len();
    if kz < ka {
        return Err(rc_err(481, "equation not identified; must have at least as many instruments not in the regression as there are instrumented variables"));
    }
    Ok(IvData { dep, endog: endog_v, exog: exog.trim().to_string(), inst: inst.trim().to_string(), d, dz, y2 })
}

/// OLS of each endogenous variable on the first-stage regressors.
fn first_stage(iv: &IvData) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut pis = Vec::new();
    let mut res = Vec::new();
    let zact = iv.dz.active();
    for y in &iv.y2 {
        let (b, _) = xm::ols_start(&iv.dz, y);
        let r: Vec<f64> = (0..iv.dz.n()).map(|t| y[t] - zact.iter().enumerate().map(|(a, &j)| iv.dz.cols[j][t] * b[a]).sum::<f64>()).collect();
        pis.push(b);
        res.push(r);
    }
    (pis, res)
}

/// Exogenous variable names for display (first-stage regressors, no constant).
fn exog_list(dz: &Design) -> String {
    dz.active().iter().filter(|&&j| dz.names[j] != "_cons").map(|&j| dz.names[j].clone()).collect::<Vec<_>>().join(" ")
}

/// Lower-triangular L from parameters (row-wise, log diagonal).
fn chol_of(p: usize, lp: &[f64]) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; p]; p];
    let mut k = 0;
    for i in 0..p {
        for j in 0..=i {
            l[i][j] = if i == j { lp[k].exp() } else { lp[k] };
            k += 1;
        }
    }
    l
}

fn sigma_of(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = l.len();
    let mut s = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            s[i][j] = (0..p).map(|k| l[i][k] * l[j][k]).sum();
        }
    }
    s
}

fn mvn_ll(v: &[f64], l: &[Vec<f64>]) -> f64 {
    let p = v.len();
    let mut u = vec![0.0; p];
    let mut ld = 0.0;
    for i in 0..p {
        let s: f64 = (0..i).map(|k| l[i][k] * u[k]).sum();
        u[i] = (v[i] - s) / l[i][i];
        ld += l[i][i].ln();
    }
    -0.5 * p as f64 * (2.0 * std::f64::consts::PI).ln() - ld - 0.5 * u.iter().map(|x| x * x).sum::<f64>()
}

/// Parameters of L from a covariance matrix.
fn lpar_of(s: &Matrix) -> Vec<f64> {
    let p = s.rows;
    let l = s.cholesky().unwrap_or_else(|_| Matrix::identity(p));
    let mut out = Vec::new();
    for i in 0..p {
        for j in 0..=i {
            out.push(if i == j { l[(i, j)].max(1e-8).ln() } else { l[(i, j)] });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ivprobit / ivtobit / ivfprobit (maximum likelihood)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Probit,
    Tobit,
    Frac,
}

pub struct IvMl {
    pub kind: Kind,
}

fn limit_opt(ds: &DataSet, cmd: &Command, name: &str, d: &Design, def: f64) -> Result<(Vec<f64>, Option<String>), String> {
    match cmd.opt(name, 2) {
        None => {
            if cmd.has_opt(name, 2) {
                let ext = if name == "ll" { d.y.iter().cloned().fold(f64::INFINITY, f64::min) } else { d.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
                Ok((vec![ext; d.n()], Some(fmt_g(ext, 8))))
            } else {
                Ok((vec![def; d.n()], None))
            }
        }
        Some(s) => match s.trim().parse::<f64>() {
            Ok(v) => Ok((vec![v; d.n()], Some(fmt_g(v, 8)))),
            Err(_) => {
                let v = xm::var_name(ds, s)?;
                let c = util::col(ds, &v)?;
                Ok((d.rows.iter().map(|&i| if ferrostat_core::missing::is_missing(c[i]) { def } else { c[i] }).collect(), Some(v)))
            }
        },
    }
}

impl CommandHandler for IvMl {
    fn name(&self) -> &'static str {
        match self.kind {
            Kind::Probit => "ivprobit",
            Kind::Tobit => "ivtobit",
            Kind::Frac => "ivfprobit",
        }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        let ds_ = &*ds;
        let iv = iv_data(ds_, st, cmd, &cmd.main, "fpi")?;
        if self.kind == Kind::Probit && cmd.has_opt("twostep", 3) {
            return newey(ds_, st, cmd, &iv);
        }
        let d = &iv.d;
        let dz = &iv.dz;
        let n = d.n();
        let p = iv.y2.len();
        let kind = self.kind;
        if kind == Kind::Frac && d.y.iter().any(|v| *v < 0.0 || *v > 1.0) {
            return Err(rc_err(459, format!("{} has values outside [0,1]", iv.dep)));
        }
        let mut vce = xm::ml_vce_opt(cmd, d)?;
        if kind == Kind::Frac && vce == Vce::Oim {
            vce = Vce::Robust;
        }
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let (lo, lname) = if kind == Kind::Tobit { limit_opt(ds_, cmd, "ll", d, f64::NEG_INFINITY)? } else { (vec![], None) };
        let (hi, uname) = if kind == Kind::Tobit { limit_opt(ds_, cmd, "ul", d, f64::INFINITY)? } else { (vec![], None) };
        // starting values
        let (pis, vhat) = first_stage(&iv);
        let mut s22 = Matrix::zeros(p, p);
        for a in 0..p {
            for c in 0..p {
                s22[(a, c)] = (0..n).map(|t| vhat[a][t] * vhat[c][t]).sum::<f64>() / n as f64;
            }
        }
        let act = d.active();
        let kx = act.len();
        let mut xcols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        for v in &vhat {
            xcols.push(v.as_slice());
        }
        let y = d.y.clone();
        let mut out = Vec::new();
        let (start, ll_exog) = match kind {
            Kind::Tobit => {
                let mut dd = d.clone();
                for v in &vhat {
                    dd.cols.push(v.clone());
                    dd.names.push("v".into());
                    dd.omitted.push(false);
                    dd.is_base.push(false);
                }
                let (b, s) = xm::ols_start(&dd, &y);
                let mut st0 = b;
                st0.push(s.max(1e-6).ln());
                (st0, SYSMISS)
            }
            _ => {
                let r = crate::heckman::probit_fit(&xcols, &y.iter().map(|v| if *v > 0.5 { 1.0 } else { 0.0 }).collect::<Vec<_>>())?;
                let xo: Vec<&[f64]> = xcols[..kx].to_vec();
                let r0 = if kind == Kind::Probit { crate::heckman::probit_fit(&xo, &y).map(|r| r.ll).unwrap_or(SYSMISS) } else { SYSMISS };
                (r.b, r0)
            }
        };
        out.push(match kind {
            Kind::Probit => "Fitting exogenous probit model".to_string(),
            Kind::Tobit => "Fitting exogenous tobit model".to_string(),
            Kind::Frac => "Fitting exogenous fractional probit model".to_string(),
        });
        out.push(String::new());
        // parameter vector: eq0 (kx), first stages (p x kz), delta (p), [ln s], L
        let kz = dz.active().len();
        let mut b0: Vec<f64> = start[..kx].to_vec();
        for pi in &pis {
            b0.extend(pi.iter().cloned());
        }
        b0.extend(start[kx..kx + p].iter().cloned());
        if kind == Kind::Tobit {
            b0.push(start[kx + p]);
        }
        b0.extend(lpar_of(&s22));
        let naux = b0.len() - kx - p * kz;
        let nl = p * (p + 1) / 2;
        let f = |t: usize, v: &[f64]| -> f64 {
            let xb = v[0];
            let vv: Vec<f64> = (0..p).map(|j| iv.y2[j][t] - v[1 + j]).collect();
            let a = &v[1 + p..];
            let delta = &a[..p];
            let (lns, lp) = if kind == Kind::Tobit { (a[p], &a[p + 1..p + 1 + nl]) } else { (0.0, &a[p..p + nl]) };
            let l = chol_of(p, lp);
            let m = xb + (0..p).map(|j| vv[j] * delta[j]).sum::<f64>();
            let l1 = match kind {
                Kind::Probit => crate::xtnl::ln_phi(if y[t] != 0.0 { m } else { -m }),
                Kind::Frac => {
                    let mut s = 0.0;
                    if y[t] > 0.0 {
                        s += y[t] * crate::xtnl::ln_phi(m);
                    }
                    if y[t] < 1.0 {
                        s += (1.0 - y[t]) * crate::xtnl::ln_phi(-m);
                    }
                    s
                }
                Kind::Tobit => {
                    let s = lns.exp();
                    if y[t] <= lo[t] {
                        crate::xtnl::ln_phi((lo[t] - m) / s)
                    } else if y[t] >= hi[t] {
                        crate::xtnl::ln_phi((m - hi[t]) / s)
                    } else {
                        dist::lnnormalden((y[t] - m) / s) - lns
                    }
                }
            };
            l1 + mvn_ll(&vv, &l)
        };
        let g = xm::numeric(&f);
        let mut eqs: Vec<(&str, &Design)> = vec![("", d)];
        for _ in 0..p {
            eqs.push(("", dz));
        }
        let auxn: Vec<String> = (0..naux).map(|i| format!("a{i}")).collect();
        let acts: Vec<Vec<usize>> = eqs.iter().map(|(_, dd)| dd.active()).collect();
        let x: Vec<Vec<&[f64]>> = eqs.iter().zip(&acts).map(|((_, dd), a)| a.iter().map(|&j| dd.cols[j].as_slice()).collect()).collect();
        let m = crate::zip::MultiIndex { x, naux, w: &d.w, fweight: d.wkind == Some(WeightKind::F), f: &g };
        let res = crate::choice_mixed::maximize_tight(&m, b0, 300)?;
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, if kind == Kind::Frac { "Log pseudolikelihood" } else { "Log likelihood" }));
        out.push(String::new());
        let (vstar, n_clust) = crate::zip::ml_vce(&res, &vce, cluster.as_deref(), n as f64);
        // structural parameters
        let structural = |th: &[f64]| -> Vec<f64> {
            let a = &th[kx + p * kz..];
            let delta = &a[..p];
            let (lns, lp) = if kind == Kind::Tobit { (a[p], &a[p + 1..p + 1 + nl]) } else { (0.0, &a[p..p + nl]) };
            let s = sigma_of(&chol_of(p, lp));
            let mut out: Vec<f64>;
            let cov: Vec<f64>;
            let su2: f64;
            if kind == Kind::Tobit {
                cov = (0..p).map(|i| (0..p).map(|j| s[i][j] * delta[j]).sum()).collect();
                su2 = (2.0 * lns).exp() + (0..p).map(|i| delta[i] * cov[i]).sum::<f64>();
                out = th[..kx + p * kz].to_vec();
            } else {
                let q: f64 = (0..p).map(|i| (0..p).map(|j| delta[i] * s[i][j] * delta[j]).sum::<f64>()).sum();
                let sc = 1.0 / (1.0 + q).sqrt();
                out = th[..kx].iter().map(|b| b * sc).collect();
                out.extend(th[kx..kx + p * kz].iter().cloned());
                cov = (0..p).map(|i| (0..p).map(|j| s[i][j] * delta[j] * sc).sum()).collect();
                su2 = 1.0;
            }
            for j in 0..p {
                let r = (cov[j] / (su2 * s[j][j]).sqrt()).clamp(-0.999999999, 0.999999999);
                out.push(r.atanh());
            }
            for j in 0..p {
                out.push(0.5 * s[j][j].ln());
            }
            if kind == Kind::Tobit {
                out.push(0.5 * su2.ln());
            }
            out
        };
        let bs = structural(&res.b);
        let ks = bs.len();
        let kt = res.b.len();
        let mut jm = Matrix::zeros(ks, kt);
        for c in 0..kt {
            let h = 1e-6 * (1.0 + res.b[c].abs());
            let mut bp = res.b.clone();
            bp[c] += h;
            let up = structural(&bp);
            bp[c] = res.b[c] - h;
            let dn = structural(&bp);
            for r in 0..ks {
                jm[(r, c)] = (up[r] - dn[r]) / (2.0 * h);
            }
        }
        let vs = jm.mul(&vstar)?.mul(&jm.transpose())?;
        // names
        let suffix = |base: &str, j: usize| if p == 1 { base.to_string() } else { format!("{base}{}", j + 1) };
        let mut aux: Vec<String> = (0..p).map(|j| suffix("athrho", j)).collect();
        aux.extend((0..p).map(|j| suffix("lnsigma", j)));
        if kind == Kind::Tobit {
            aux.push("lnsigma_y".into());
        }
        let auxr: Vec<&str> = aux.iter().map(|s| s.as_str()).collect();
        let _ = &auxn;
        let mut eqn: Vec<(&str, &Design)> = vec![(iv.dep.as_str(), d)];
        for e in &iv.endog {
            eqn.push((e.as_str(), dz));
        }
        let ll = res.ll;
        let iters = res.iterations.clone();
        let conv = res.converged;
        let res2 = est::MlResult { b: bs.clone(), ll, info: Matrix::zeros(ks, ks), scores: vec![], iterations: iters, converged: conv };
        let mut fit = xm::finish_fit(&eqn, &auxr, res2, &Vce::Oim, None)?;
        fit.vt = vs.clone();
        fit.v = est::expand_v(&vs, &fit.pos, fit.names.len());
        fit.n_clust = n_clust;
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        let ath_idx: Vec<usize> = (0..p).map(|j| kx + p * kz + j).collect();
        let (exo_chi, exo_df) = crate::heckman::wald(&bs, &vs, &ath_idx);
        let title = match kind {
            Kind::Probit => "Probit model with endogenous regressors",
            Kind::Tobit => "Tobit model with endogenous regressors",
            Kind::Frac => "Fractional probit model with endogenous regressors",
        };
        let mut lines = Vec::new();
        let (mut nu, mut nlc, mut nrc) = (0usize, 0usize, 0usize);
        if kind == Kind::Tobit {
            for t in 0..n {
                if y[t] <= lo[t] {
                    nlc += 1
                } else if y[t] >= hi[t] {
                    nrc += 1
                } else {
                    nu += 1
                }
            }
            lines.push((String::new(), "Uncensored".to_string(), util::commas(nu)));
            lines.push((format!("Limits: Lower = {}", lname.clone().unwrap_or_else(|| "-inf".into())), "Left-censored".to_string(), util::commas(nlc)));
            lines.push((format!("        Upper = {}", uname.clone().unwrap_or_else(|| "+inf".into())), "Right-censored".to_string(), util::commas(nrc)));
            lines.push((String::new(), String::new(), String::new()));
        }
        let mut hl = Header { title, n: d.n_eff(), lines, test: Some(("Wald".into(), chi2, df)), ll: Some(ll), robust, n_clust: n_clust.map(|g| (g, if let Vce::Cluster(c) = &vce { c.clone() } else { String::new() })) }.render();
        for l in hl.iter_mut() {
            if l.trim_end().ends_with('=') {
                *l = String::new();
            }
        }
        out.extend(hl);
        let level = util::level(cmd, st)?;
        let kd = d.k();
        let mut tbl = est::coef_table(&est::TableSpec { depvar: "", names: &d.names, b: &fit.b[..kd], v: &fit.v, omitted: &fit.omitted[..kd], is_base: &fit.is_base[..kd], df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(crate::choice_mixed::sep(lw));
        let mut extra = Vec::new();
        let nb = bs.len();
        let base = kx + p * kz;
        for j in 0..p {
            let (a, se) = (bs[base + j], vs[(base + j, base + j)].max(0.0).sqrt());
            let r = a.tanh();
            extra.push(Row::Tr(format!("corr(e.{},e.{})", iv.endog[j], iv.dep), r, se * (1.0 - r * r), a, se, f64::tanh));
        }
        if kind == Kind::Tobit {
            let i = nb - 1;
            let (a, se) = (bs[i], vs[(i, i)].max(0.0).sqrt());
            extra.push(Row::Tr(format!("sd(e.{})", iv.dep), a.exp(), se * a.exp(), a, se, f64::exp));
        }
        for j in 0..p {
            let i = base + p + j;
            let (a, se) = (bs[i], vs[(i, i)].max(0.0).sqrt());
            extra.push(Row::Tr(format!("sd(e.{})", iv.endog[j]), a.exp(), se * a.exp(), a, se, f64::exp));
        }
        tbl.extend(xm::extra_rows(lw, level, &extra, false));
        tbl.push(bar);
        out.extend(tbl);
        out.push(format!("Wald test of exogeneity (corr = 0): chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(exo_df, 9), xm::f2(exo_chi), xm::f4(xm::chi2p(exo_df, exo_chi))));
        out.push(format!("Endogenous: {}", iv.endog.join(" ")));
        out.push(format!("Exogenous:  {}", exog_list(dz)));
        let text = out.join("\n");
        let endog = iv.endog.join(" ");
        let insts = exog_list(dz);
        let dep = iv.dep.clone();
        xm::post_fit(st, verb, d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("chi2_exog", exo_chi);
            e.set("p_exog", xm::chi2p(exo_df, exo_chi));
            e.set("endog_ct", p as f64);
            e.set("ll_0", ll_exog);
            if p == 1 {
                e.set("rho", bs[base].tanh());
                e.set("sigma", bs[base + 1].exp());
            }
            if kind == Kind::Tobit {
                e.set("N_unc", nu as f64);
                e.set("N_lc", nlc as f64);
                e.set("N_rc", nrc as f64);
            }
            e.set_str("depvar", dep.clone());
            e.set_str("instd", endog.clone());
            e.set_str("insts", insts.clone());
            e.set_str("method", "ml");
            e.set_str("title", title);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// Newey's minimum chi-squared two-step estimator for ivprobit.
fn newey(ds: &DataSet, st: &mut Store, cmd: &Command, iv: &IvData) -> Result<String, String> {
    let d = &iv.d;
    let dz = &iv.dz;
    let n = d.n();
    let p = iv.y2.len();
    let (pis, vhat) = first_stage(iv);
    let zact = dz.active();
    let xact = d.active();
    let kz = zact.len();
    let kx = xact.len();
    let y = d.y.clone();
    // (3): probit of y on all exogenous variables and the residuals
    let mut c3: Vec<&[f64]> = zact.iter().map(|&j| dz.cols[j].as_slice()).collect();
    for v in &vhat {
        c3.push(v.as_slice());
    }
    let r3 = crate::heckman::probit_fit(&c3, &y)?;
    let (v3, _) = r3.info.sym_inverse(1e-14);
    let alpha: Vec<f64> = r3.b[..kz].to_vec();
    let lambda: Vec<f64> = r3.b[kz..].to_vec();
    // 2SIV: probit of y on the structural regressors and the residuals
    let mut c2: Vec<&[f64]> = xact.iter().map(|&j| d.cols[j].as_slice()).collect();
    for v in &vhat {
        c2.push(v.as_slice());
    }
    let r2 = crate::heckman::probit_fit(&c2, &y)?;
    let (v2, _) = r2.info.sym_inverse(1e-14);
    // beta on endogenous variables (their positions among the active x columns)
    let endpos: Vec<usize> = iv.endog.iter().map(|e| xact.iter().position(|&j| d.names[j] == *e).unwrap()).collect();
    let beta: Vec<f64> = endpos.iter().map(|&a| r2.b[a]).collect();
    // regress y2 (lambda - beta) on z: covariance added to J^-1
    let yy: Vec<f64> = (0..n).map(|t| (0..p).map(|j| iv.y2[j][t] * (lambda[j] - beta[j])).sum()).collect();
    let mut dd = dz.clone();
    dd.y = yy;
    let fo = est::ols(&dd, &Vce::Ols, None)?;
    let mut omega = Matrix::zeros(kz, kz);
    for a in 0..kz {
        for c in 0..kz {
            omega[(a, c)] = v3[(a, c)] + fo.v[(zact[a], zact[c])];
        }
    }
    let (oi, _) = omega.sym_inverse(1e-14);
    // D = (Pi, I1): one column per structural active regressor
    let mut dm = Matrix::zeros(kz, kx);
    for (a, &j) in xact.iter().enumerate() {
        if let Some(e) = endpos.iter().position(|&q| q == a) {
            for r in 0..kz {
                dm[(r, a)] = pis[e][r];
            }
        } else {
            let r = zact.iter().position(|&k| dz.names[k] == d.names[j]).ok_or_else(|| rc_err(198, format!("{} must be among the instruments", d.names[j])))?;
            dm[(r, a)] = 1.0;
        }
    }
    let dto = dm.transpose().mul(&oi)?;
    let (vd, _) = dto.mul(&dm)?.sym_inverse(1e-14);
    let rhs: Vec<f64> = (0..kx).map(|a| (0..kz).map(|r| dto[(a, r)] * alpha[r]).sum()).collect();
    let delta: Vec<f64> = (0..kx).map(|a| (0..kx).map(|c| vd[(a, c)] * rhs[c]).sum()).collect();
    let res = est::MlResult { b: delta.clone(), ll: SYSMISS, info: Matrix::zeros(kx, kx), scores: vec![], iterations: vec![], converged: true };
    let mut fit = xm::finish_fit(&[("", d)], &[], res, &Vce::Oim, None)?;
    fit.vt = vd.clone();
    fit.v = est::expand_v(&vd, &fit.pos, fit.names.len());
    let (chi2, df) = fit.wald_eq(0);
    let lidx: Vec<usize> = (kx..kx + p).collect();
    let (ex, exdf) = crate::heckman::wald(&r2.b, &v2, &lidx);
    let level = util::level(cmd, st)?;
    let mut out = vec!["Checking reduced-form model...".to_string(), String::new()];
    out.push(crate::choice_mixed::hrow("Two-step probit with endogenous regressors", "Number of obs", util::commas(n)));
    out.push(crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
    out.push(crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
    out.push(String::new());
    out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &fit.names, b: &fit.b, v: &fit.v, omitted: &fit.omitted, is_base: &fit.is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
    out.push(format!("Wald test of exogeneity: chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(exdf, 9), xm::f2(ex), xm::f4(xm::chi2p(exdf, ex))));
    out.push(format!("Endogenous: {}", iv.endog.join(" ")));
    out.push(format!("Exogenous:  {}", exog_list(dz)));
    let text = out.join("\n");
    let _ = ds;
    let endog = iv.endog.join(" ");
    xm::post_fit(st, "ivprobit", d, &fit, &Vce::Oim, &|e: &mut Results| {
        xm::set_test(e, "Wald", chi2, df);
        e.set("chi2_exog", ex);
        e.set("p_exog", xm::chi2p(exdf, ex));
        e.set_str("instd", endog.clone());
        e.set_str("method", "twostep");
        e.set_str("vce", "twostep");
        e.set_str("title", "Two-step probit with endogenous regressors");
        e.set_str("_output", text.clone());
    });
    Ok(text)
}

// ---------------------------------------------------------------------------
// ivpoisson
// ---------------------------------------------------------------------------

pub struct Ivpoisson;

fn skind(ds: &DataSet, d: &Design, vce: &Vce) -> Result<SKind, String> {
    Ok(match vce {
        Vce::Cluster(_) => SKind::Cluster(est::cluster_values(ds, d, vce)?.unwrap_or_default()),
        _ => SKind::Robust,
    })
}

impl CommandHandler for Ivpoisson {
    fn name(&self) -> &'static str {
        "ivpoisson"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (est_kind, rest) = xm::first_word(&cmd.main);
        let est_kind = est_kind.to_lowercase();
        if !matches!(est_kind.as_str(), "gmm" | "cfunction") {
            return Err(rc_err(198, "ivpoisson: estimator must be gmm or cfunction"));
        }
        let ds_ = &*ds;
        let iv = iv_data(ds_, st, cmd, &rest, "fpi")?;
        let d = &iv.d;
        let dz = &iv.dz;
        let n = d.n();
        let xact = d.active();
        let zact = dz.active();
        let kx = xact.len();
        let kz = zact.len();
        let w = d.w.clone();
        let y = d.y.clone();
        let vce = est::parse_vce(cmd, Vce::Robust)?;
        let sk = skind(ds_, d, &vce)?;
        let mult = cmd.has_opt("multiplicative", 4);
        let xb = |b: &[f64], t: usize| -> f64 { (0..kx).map(|a| d.cols[xact[a]][t] * b[a]).sum() };
        // Poisson start on the structural regressors
        let mut b0 = vec![0.0; kx];
        if let Some(c) = xact.iter().position(|&j| d.names[j] == "_cons") {
            b0[c] = (util::mean(&y) + 1e-3).ln();
        }
        let level = util::level(cmd, st)?;
        let mut out = Vec::new();
        if est_kind == "gmm" {
            let mom = |b: &[f64]| -> Result<Vec<Vec<f64>>, String> {
                Ok((0..n)
                    .map(|t| {
                        let mu = xb(b, t).exp();
                        let u = if mult { y[t] / mu - 1.0 } else { y[t] - mu };
                        zact.iter().map(|&j| dz.cols[j][t] * u).collect()
                    })
                    .collect())
            };
            let zz = {
                let sw: f64 = w.iter().sum();
                let mut m = Matrix::zeros(kz, kz);
                for a in 0..kz {
                    for c in 0..kz {
                        m[(a, c)] = (0..n).map(|t| w[t] * dz.cols[zact[a]][t] * dz.cols[zact[c]][t]).sum::<f64>() / sw;
                    }
                }
                m
            };
            let w1 = gm::weight_of(&zz);
            let (b1, _, log1, _) = gm::minimize(&mom, b0, &w1, &w, 300)?;
            out.push("Step 1".into());
            for (i, q) in log1.iter().enumerate() {
                out.push(format!("Iteration {i}:   GMM criterion Q(b) = {}", fmt_g(*q, 9)));
            }
            let onestep = cmd.has_opt("onestep", 3);
            let (b, wm) = if onestep {
                (b1, w1)
            } else {
                let s1 = gm::s_matrix(&mom(&b1)?, &w, &sk);
                let w2 = gm::weight_of(&s1);
                let (b2, _, log2, _) = gm::minimize(&mom, b1, &w2, &w, 300)?;
                out.push(String::new());
                out.push("Step 2".into());
                for (i, q) in log2.iter().enumerate() {
                    out.push(format!("Iteration {i}:   GMM criterion Q(b) = {}", fmt_g(*q, 9)));
                }
                (b2, w2)
            };
            out.push(String::new());
            let g = gm::jac(&mom, &b, &w)?;
            let s = gm::s_matrix(&mom(&b)?, &w, &sk);
            let sw: f64 = w.iter().sum();
            let v = gm::gmm_v(&g, &wm, &s, sw)?;
            let qv = {
                let gb = gm::gbar(&mom(&b)?, &w);
                (0..kz).map(|a| (0..kz).map(|c| gb[a] * wm[(a, c)] * gb[c]).sum::<f64>()).sum::<f64>()
            };
            let res = est::MlResult { b: b.clone(), ll: SYSMISS, info: Matrix::zeros(kx, kx), scores: vec![], iterations: vec![], converged: true };
            let mut fit = xm::finish_fit(&[("", d)], &[], res, &Vce::Oim, None)?;
            fit.vt = v.clone();
            fit.v = est::expand_v(&v, &fit.pos, fit.names.len());
            out.push("Exponential mean model with endogenous regressors".into());
            out.push(String::new());
            out.push(format!("{:<48}{:<17} = {:>10}", format!("Number of parameters = {kx:>3}"), "Number of obs", util::commas(n)));
            out.push(format!("Number of moments    = {kz:>3}"));
            out.push("Initial weight matrix: Unadjusted".into());
            if !onestep {
                out.push("GMM weight matrix:     Robust".into());
            }
            out.push(String::new());
            out.extend(est::coef_table(&est::TableSpec { depvar: &iv.dep, names: &fit.names, b: &fit.b, v: &fit.v, omitted: &fit.omitted, is_base: &fit.is_base, df: None, level, eform: if cmd.has_opt("irr", 3) { Some("IRR") } else { None }, coef_title: "Coefficient" }));
            out.push(format!("Endogenous: {}", iv.endog.join(" ")));
            out.push(format!("Exogenous:  {}", exog_list(dz)));
            let text = out.join("\n");
            let endog = iv.endog.join(" ");
            let (chi2, df) = fit.wald_eq(0);
            xm::post_fit(st, "ivpoisson", d, &fit, &Vce::Robust, &|e: &mut Results| {
                xm::set_test(e, "Wald", chi2, df);
                e.set("Q", qv);
                e.set("J", sw * qv);
                e.set("J_df", (kz - kx) as f64);
                e.set("n_moments", kz as f64);
                e.set_str("estimator", "gmm");
                e.set_str("additive", if mult { "multiplicative" } else { "additive" });
                e.set_str("instd", endog.clone());
                e.set_str("title", "Exponential mean model with endogenous regressors");
                e.set_str("_output", text.clone());
            });
            return Ok(text);
        }
        // control function: first stages, then Poisson with the residuals
        let p = iv.y2.len();
        let (pis, _) = first_stage(&iv);
        let ktot = p * kz + kx + p;
        let mom = |th: &[f64]| -> Result<Vec<Vec<f64>>, String> {
            Ok((0..n)
                .map(|t| {
                    let mut row = Vec::with_capacity(ktot);
                    let mut vv = Vec::with_capacity(p);
                    for j in 0..p {
                        let pi = &th[j * kz..(j + 1) * kz];
                        let v = iv.y2[j][t] - (0..kz).map(|a| dz.cols[zact[a]][t] * pi[a]).sum::<f64>();
                        for a in 0..kz {
                            row.push(dz.cols[zact[a]][t] * v);
                        }
                        vv.push(v);
                    }
                    let bb = &th[p * kz..];
                    let eta = xb(bb, t) + (0..p).map(|j| bb[kx + j] * vv[j]).sum::<f64>();
                    let u = y[t] - eta.exp();
                    for a in 0..kx {
                        row.push(d.cols[xact[a]][t] * u);
                    }
                    for j in 0..p {
                        row.push(vv[j] * u);
                    }
                    row
                })
                .collect())
        };
        let mut th0: Vec<f64> = pis.iter().flatten().cloned().collect();
        th0.extend(b0.iter().cloned());
        th0.extend(vec![0.0; p]);
        let wid = Matrix::identity(ktot);
        let (th, _, log, _) = gm::minimize(&mom, th0, &wid, &w, 300)?;
        for (i, q) in log.iter().enumerate() {
            out.push(format!("Iteration {i}:   GMM criterion Q(b) = {}", fmt_g(*q, 9)));
        }
        out.push(String::new());
        let v = gm::exact_v(&mom, &th, &w, &sk)?;
        // reorder: main equation, first stages, control-function coefficients
        let mut order: Vec<usize> = (p * kz..p * kz + kx).collect();
        order.extend(0..p * kz);
        order.extend(p * kz + kx..ktot);
        let bo: Vec<f64> = order.iter().map(|&i| th[i]).collect();
        let mut vo = Matrix::zeros(ktot, ktot);
        for (a, &i) in order.iter().enumerate() {
            for (c, &j) in order.iter().enumerate() {
                vo[(a, c)] = v[(i, j)];
            }
        }
        let auxn: Vec<String> = iv.endog.iter().map(|e| format!("c_{e}")).collect();
        let auxr: Vec<&str> = auxn.iter().map(|s| s.as_str()).collect();
        let mut eqn: Vec<(&str, &Design)> = vec![(iv.dep.as_str(), d)];
        for e in &iv.endog {
            eqn.push((e.as_str(), dz));
        }
        let res = est::MlResult { b: bo.clone(), ll: SYSMISS, info: Matrix::zeros(ktot, ktot), scores: vec![], iterations: vec![], converged: true };
        let mut fit = xm::finish_fit(&eqn, &auxr, res, &Vce::Oim, None)?;
        fit.vt = vo.clone();
        fit.v = est::expand_v(&vo, &fit.pos, fit.names.len());
        let sw: f64 = w.iter().sum();
        out.push("Exponential mean model with endogenous regressors".into());
        out.push(String::new());
        out.push(format!("{:<48}{:<17} = {:>10}", format!("Number of parameters = {ktot:>3}"), "Number of obs", util::commas(n)));
        out.push(format!("Number of moments    = {ktot:>3}"));
        out.push(String::new());
        out.extend(xm::table("", &fit, level, None, &[]));
        out.push(format!("Endogenous: {}", iv.endog.join(" ")));
        out.push(format!("Exogenous:  {}", exog_list(dz)));
        let text = out.join("\n");
        let (chi2, df) = fit.wald_eq(0);
        let endog = iv.endog.join(" ");
        let _ = sw;
        xm::post_fit(st, "ivpoisson", d, &fit, &Vce::Robust, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set_str("estimator", "cfunction");
            e.set_str("instd", endog.clone());
            e.set_str("title", "Exponential mean model with endogenous regressors");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// cfregress / cfprobit
// ---------------------------------------------------------------------------

pub struct Cf {
    pub probit: bool,
}

impl CommandHandler for Cf {
    fn name(&self) -> &'static str {
        if self.probit { "cfprobit" } else { "cfregress" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // mainonly(): regressors of the main equation only
        let mut main = cmd.main.clone();
        let mainonly = cmd.opt("mainonly", 5).unwrap_or("").to_string();
        let (outside, endog, inst) = xm::iv_parts(&main)?;
        let inst = inst.split(',').next().unwrap_or("").to_string();
        main = format!("{outside} ({endog} = {inst})");
        let ds_ = &*ds;
        let iv0 = iv_data(ds_, st, cmd, &main, "fpi")?;
        let iv = if mainonly.trim().is_empty() {
            iv0
        } else {
            let (o, en, ins) = xm::iv_parts(&main)?;
            let (dep, exog) = xm::first_word(&o);
            let xtext = format!("{} {} {}", iv0.endog.join(" "), mainonly, exog);
            let ztext = format!("{exog} {ins}");
            let nocons = cmd.has_opt("noconstant", 3);
            let des = xm::build_eqs(ds_, st, cmd, &[(Some(dep.as_str()), xtext.as_str(), nocons), (None, ztext.as_str(), nocons)], "fpi", &iv0.endog)?;
            let _ = en;
            let mut y2 = Vec::new();
            for v in &iv0.endog {
                let j = des[0].names.iter().position(|n| n == v).unwrap();
                y2.push(des[0].cols[j].clone());
            }
            IvData { dep: iv0.dep.clone(), endog: iv0.endog.clone(), exog, inst: ins, d: des[0].clone(), dz: des[1].clone(), y2 }
        };
        let d = &iv.d;
        let dz = &iv.dz;
        let n = d.n();
        let p = iv.y2.len();
        let xact = d.active();
        let zact = dz.active();
        let (kx, kz) = (xact.len(), zact.len());
        let y = d.y.clone();
        let w = d.w.clone();
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let robust = !matches!(vce, Vce::Oim | Vce::Ols);
        let sk = skind(ds_, d, &vce)?;
        let probit = self.probit;
        let (pis, vhat) = first_stage(&iv);
        // main equation on x and the control functions
        let mut cols: Vec<&[f64]> = xact.iter().map(|&j| d.cols[j].as_slice()).collect();
        for v in &vhat {
            cols.push(v.as_slice());
        }
        let bmain: Vec<f64> = if probit {
            crate::heckman::probit_fit(&cols, &y)?.b
        } else {
            let mut dd = d.clone();
            for v in &vhat {
                dd.cols.push(v.clone());
                dd.names.push("v".into());
                dd.omitted.push(false);
                dd.is_base.push(false);
            }
            xm::ols_start(&dd, &y).0
        };
        let ktot = p * kz + kx + p;
        let resid_v = |th: &[f64], t: usize, j: usize| -> f64 { iv.y2[j][t] - (0..kz).map(|a| dz.cols[zact[a]][t] * th[j * kz + a]).sum::<f64>() };
        let mom = |th: &[f64]| -> Result<Vec<Vec<f64>>, String> {
            Ok((0..n)
                .map(|t| {
                    let mut row = Vec::with_capacity(ktot);
                    let vv: Vec<f64> = (0..p).map(|j| resid_v(th, t, j)).collect();
                    for j in 0..p {
                        for a in 0..kz {
                            row.push(dz.cols[zact[a]][t] * vv[j]);
                        }
                    }
                    let bb = &th[p * kz..];
                    let idx = (0..kx).map(|a| d.cols[xact[a]][t] * bb[a]).sum::<f64>() + (0..p).map(|j| bb[kx + j] * vv[j]).sum::<f64>();
                    let s = if probit {
                        let q = if y[t] != 0.0 { 1.0 } else { -1.0 };
                        q * crate::choice_mixed::ln_phi_d(q * idx).1
                    } else {
                        y[t] - idx
                    };
                    for a in 0..kx {
                        row.push(d.cols[xact[a]][t] * s);
                    }
                    for j in 0..p {
                        row.push(vv[j] * s);
                    }
                    row
                })
                .collect())
        };
        let mut th: Vec<f64> = pis.iter().flatten().cloned().collect();
        th.extend(bmain.iter().cloned());
        let sw: f64 = w.iter().sum();
        let v = if robust {
            gm::exact_v(&mom, &th, &w, &sk)?
        } else {
            // conventional: block-diagonal S with the model-based variances
            let g = gm::jac(&mom, &th, &w)?;
            let mut s = Matrix::zeros(ktot, ktot);
            for j in 0..p {
                let s2: f64 = (0..n).map(|t| w[t] * resid_v(&th, t, j).powi(2)).sum::<f64>() / sw;
                for a in 0..kz {
                    for c in 0..kz {
                        s[(j * kz + a, j * kz + c)] = s2 * (0..n).map(|t| w[t] * dz.cols[zact[a]][t] * dz.cols[zact[c]][t]).sum::<f64>() / sw;
                    }
                }
            }
            let o = p * kz;
            if probit {
                // information equality: S = -G for the probit block
                for a in 0..kx + p {
                    for c in 0..kx + p {
                        s[(o + a, o + c)] = -g[(o + a, o + c)];
                    }
                }
            } else {
                // residual variance of the main equation
                let bb = &th[o..];
                let mut rss = 0.0;
                for t in 0..n {
                    let vv: Vec<f64> = (0..p).map(|j| resid_v(&th, t, j)).collect();
                    let idx = (0..kx).map(|a| d.cols[xact[a]][t] * bb[a]).sum::<f64>() + (0..p).map(|j| bb[kx + j] * vv[j]).sum::<f64>();
                    rss += w[t] * (y[t] - idx).powi(2);
                }
                let s2 = rss / sw;
                let xt = |t: usize, a: usize| -> f64 { if a < kx { d.cols[xact[a]][t] } else { resid_v(&th, t, a - kx) } };
                for a in 0..kx + p {
                    for c in 0..kx + p {
                        s[(o + a, o + c)] = s2 * (0..n).map(|t| w[t] * xt(t, a) * xt(t, c)).sum::<f64>() / sw;
                    }
                }
            }
            let wid = Matrix::identity(ktot);
            gm::gmm_v(&g, &wid, &s, sw)?
        };
        // e(b): main equation, control functions (e.dep), first stages
        let mut order: Vec<usize> = (p * kz..p * kz + kx).collect();
        order.extend(p * kz + kx..ktot);
        order.extend(0..p * kz);
        let bo: Vec<f64> = order.iter().map(|&i| th[i]).collect();
        let mut vo = Matrix::zeros(ktot, ktot);
        for (a, &i) in order.iter().enumerate() {
            for (c, &j) in order.iter().enumerate() {
                vo[(a, c)] = v[(i, j)];
            }
        }
        let mut dcf = d.clone();
        dcf.names = iv.endog.iter().map(|e| format!("cf({e})")).collect();
        dcf.cols = vhat.clone();
        dcf.omitted = vec![false; p];
        dcf.is_base = vec![false; p];
        let edep = format!("e.{}", iv.dep);
        let mut eqn: Vec<(&str, &Design)> = vec![(iv.dep.as_str(), d), (edep.as_str(), &dcf)];
        for e in &iv.endog {
            eqn.push((e.as_str(), dz));
        }
        let res = est::MlResult { b: bo.clone(), ll: SYSMISS, info: Matrix::zeros(ktot, ktot), scores: vec![], iterations: vec![], converged: true };
        let mut fit = xm::finish_fit(&eqn, &[], res, &Vce::Oim, None)?;
        fit.vt = vo.clone();
        fit.v = est::expand_v(&vo, &fit.pos, fit.names.len());
        let (chi2, df) = fit.wald_eq(0);
        // R-squared and root MSE of the main equation (linear)
        let mut rss = 0.0;
        let bb = &th[p * kz..];
        for t in 0..n {
            let idx = (0..kx).map(|a| d.cols[xact[a]][t] * bb[a]).sum::<f64>() + (0..p).map(|j| bb[kx + j] * vhat[j][t]).sum::<f64>();
            rss += w[t] * (y[t] - idx).powi(2);
        }
        let ybar = (0..n).map(|t| w[t] * y[t]).sum::<f64>() / sw;
        let tss: f64 = (0..n).map(|t| w[t] * (y[t] - ybar).powi(2)).sum();
        let r2 = 1.0 - rss / tss;
        let rmse = (rss / sw).sqrt();
        let level = util::level(cmd, st)?;
        let title = if probit { "Control-function probit regression" } else { "Control-function linear regression" };
        let mut out = vec![
            crate::choice_mixed::hrow(title, "Number of obs", util::commas(n)),
            crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)),
            crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))),
        ];
        if !probit {
            out.push(crate::choice_mixed::hrow("", "R-squared", xm::f4(r2)));
            out.push(crate::choice_mixed::hrow("", "Root MSE", fmt_g(rmse, 6)));
        }
        out.push(String::new());
        out.push(format!("Endogenous variable model{}:", if p > 1 { "s" } else { "" }));
        for e in &iv.endog {
            out.push(format!("    Linear: {e}"));
        }
        out.push(String::new());
        // display main equation and control functions
        let nshow = d.k() + p;
        out.extend(est::coef_table(&est::TableSpec { depvar: &iv.dep, names: &fit.names[..nshow], b: &fit.b[..nshow], v: &fit.v, omitted: &fit.omitted[..nshow], is_base: &fit.is_base[..nshow], df: None, level, eform: None, coef_title: "Coefficient" }));
        for e in &iv.endog {
            out.push(format!("Instruments for {e}: {}", exog_list(dz).split_whitespace().filter(|v| !d.names.contains(&v.to_string())).collect::<Vec<_>>().join(" ")));
        }
        let text = out.join("\n");
        let endog = iv.endog.join(" ");
        let cidx: Vec<usize> = (kx..kx + p).collect();
        let (ce, cdf) = crate::heckman::wald(&bo, &vo, &cidx);
        let verb = self.name();
        xm::post_fit(st, verb, d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            if !probit {
                e.set("r2", r2);
                e.set("rmse", rmse);
            }
            e.set("k_endog", p as f64);
            e.set("chi2_endog", ce);
            e.set("df_endog", cdf);
            e.set_str("endog", endog.clone());
            e.set_str("title", title);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
