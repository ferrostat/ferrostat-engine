//! `bayesselect`: Bayesian variable selection for linear regression.
//!
//! ```text
//! bayesselect depvar indepvars [if] [in] [, prior(horseshoe|spikeslab) mcmcsize(#)
//!             burnin(#) rseed(#) noconstant clevel(#) hpd]
//! ```
//! Global-local shrinkage (horseshoe: half-Cauchy local and global scales,
//! written as inverse-gamma mixtures) or a continuous spike-and-slab prior
//! (inclusion indicators with a beta(1,1) inclusion probability; spike
//! variance 1/1000 of the slab's). The intercept has a flat prior and the
//! error variance a Jeffreys prior. All conditionals are sampled by Gibbs.
//! Results are stored like `bayesmh` results, so the Bayesian
//! postestimation commands apply; the inclusion summary is in
//! e(inclusion) (spike-and-slab) or e(shrinkage) (horseshoe: 1/(1+lambda^2 tau^2)).

use ferrostat_core::DataSet;

use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::store::Store;

use super::mcmc::{Settings, Sim};
use super::model::{Arg, Eq, Fam, Lik, Model, Param};
use super::prior::Support;
use super::run;
use super::stats;
use super::text;

fn ig(rng: &mut Rng, shape: f64, scale: f64) -> f64 {
    scale / rng.gamma(shape)
}

/// Draw from N(A^-1 b, s2 A^-1).
fn mvn_draw(a: &Matrix, b: &[f64], s2: f64, rng: &mut Rng) -> Option<Vec<f64>> {
    let ai = a.inverse().ok()?;
    let k = b.len();
    let mean: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * b[j]).sum()).collect();
    let l = ai.scale(s2).cholesky().ok()?;
    let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
    Some((0..k).map(|i| mean[i] + (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>()).collect())
}

pub fn bayesselect(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (main, o) = match text::find_top(rest, ',') {
        Some(i) => (rest[..i].trim().to_string(), text::options(&rest[i + 1..])),
        None => (rest.trim().to_string(), Vec::new()),
    };
    let optv = |full: &str, min: usize| o.iter().rev().find(|(n, _)| text::abbrev(n, full, min)).map(|x| x.1.clone());
    let num = |full: &str, min: usize| optv(full, min).flatten().and_then(|s| s.trim().parse::<f64>().ok());
    let prior = optv("prior", 5).flatten().unwrap_or_else(|| "horseshoe".into()).trim().to_lowercase();
    let ss = prior.starts_with("spike") || prior.starts_with("ssvs");
    if !ss && !prior.starts_with("horse") {
        return Err(rc_err(198, format!("prior({prior}) not supported; use horseshoe or spikeslab")));
    }
    let mcmcsize = num("mcmcsize", 4).unwrap_or(10000.0) as usize;
    let burnin = num("burnin", 4).unwrap_or(2500.0) as usize;
    if let Some(s) = num("rseed", 5) {
        st.rng.borrow_mut().reseed(s as u64);
    }
    let level = num("clevel", 2).unwrap_or(st.level);
    let use_hpd = optv("hpd", 3).is_some();
    let nocons = optv("noconstant", 4).is_some();
    // design
    let ws = text::words(&main);
    let (vars, ifin): (Vec<String>, Vec<String>) = {
        let k = ws.iter().position(|w| w == "if" || w == "in").unwrap_or(ws.len());
        (ws[..k].to_vec(), ws[k..].to_vec())
    };
    let dep = vars.first().cloned().ok_or_else(|| rc_err(100, "varlist required"))?;
    let indep = vars[1..].join(" ");
    let c = ferrostat_parser::parser::parse_line(&format!("regress {} {} {}", dep, indep, ifin.join(" "))).map_err(|e| rc_err(198, e.message))?;
    let d = est::build(ds, st, &c, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: nocons, weights: "", extra: Vec::new() })?;
    let act = d.active();
    let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
    let x: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
    let y = d.y.clone();
    let n = y.len();
    let k = x.len();
    let pen: Vec<bool> = names.iter().map(|nm| nm != "_cons").collect();
    let npen = pen.iter().filter(|p| **p).count();
    if npen == 0 {
        return Err(rc_err(102, "no predictors to select"));
    }
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for a in 0..k {
        for b in 0..=a {
            let s: f64 = (0..n).map(|r| x[a][r] * x[b][r]).sum();
            xtx[(a, b)] = s;
            xtx[(b, a)] = s;
        }
        xty[a] = (0..n).map(|r| x[a][r] * y[r]).sum();
    }
    // state
    let mut beta = vec![0.0; k];
    let mut s2 = stats::var(&y).max(1e-8);
    let mut lam2 = vec![1.0; k];
    let mut nu = vec![1.0; k];
    let mut tau2: f64 = 1.0;
    let mut xi = 1.0;
    let mut gam = vec![1.0; k];
    let mut theta: f64 = 0.5;
    let slab: f64 = 10.0;
    let spike = slab / 1000.0;
    let mut draws = Vec::new();
    let mut lls = Vec::new();
    let mut incl = vec![0.0; k];
    let mut shrink = vec![0.0; k];
    {
        let mut rng = st.rng.borrow_mut();
        for it in 0..burnin + mcmcsize {
            // coefficients
            let mut a = xtx.clone();
            for j in 0..k {
                if pen[j] {
                    let v = if ss { if gam[j] > 0.5 { slab } else { spike } } else { lam2[j] * tau2 };
                    a[(j, j)] += 1.0 / v.max(1e-300);
                }
            }
            if let Some(b) = mvn_draw(&a, &xty, s2, &mut rng) {
                beta = b;
            }
            let ssr: f64 = (0..n).map(|r| (y[r] - (0..k).map(|j| x[j][r] * beta[j]).sum::<f64>()).powi(2)).sum();
            let pen_ss: f64 = (0..k).filter(|&j| pen[j]).map(|j| {
                let v = if ss { if gam[j] > 0.5 { slab } else { spike } } else { lam2[j] * tau2 };
                beta[j] * beta[j] / v
            }).sum();
            s2 = ig(&mut rng, (n + npen) as f64 / 2.0, (ssr + pen_ss) / 2.0);
            if ss {
                for j in 0..k {
                    if !pen[j] {
                        continue;
                    }
                    let l1 = theta.ln() - 0.5 * (s2 * slab).ln() - beta[j] * beta[j] / (2.0 * s2 * slab);
                    let l0 = (1.0 - theta).ln() - 0.5 * (s2 * spike).ln() - beta[j] * beta[j] / (2.0 * s2 * spike);
                    let p1 = 1.0 / (1.0 + (l0 - l1).exp());
                    gam[j] = if rng.uniform() < p1 { 1.0 } else { 0.0 };
                }
                let s: f64 = (0..k).filter(|&j| pen[j]).map(|j| gam[j]).sum();
                theta = rng.beta(1.0 + s, 1.0 + npen as f64 - s);
            } else {
                let mut sb = 0.0;
                for j in 0..k {
                    if !pen[j] {
                        continue;
                    }
                    lam2[j] = ig(&mut rng, 1.0, 1.0 / nu[j] + beta[j] * beta[j] / (2.0 * tau2 * s2));
                    nu[j] = ig(&mut rng, 1.0, 1.0 + 1.0 / lam2[j]);
                    sb += beta[j] * beta[j] / lam2[j];
                }
                tau2 = ig(&mut rng, (npen as f64 + 1.0) / 2.0, 1.0 / xi + sb / (2.0 * s2));
                xi = ig(&mut rng, 1.0, 1.0 + 1.0 / tau2);
            }
            if it >= burnin {
                let mut th = beta.clone();
                th.push(s2);
                draws.push(th);
                lls.push(-0.5 * n as f64 * (2.0 * std::f64::consts::PI * s2).ln() - ssr / (2.0 * s2));
                for j in 0..k {
                    if pen[j] {
                        incl[j] += gam[j];
                        shrink[j] += 1.0 / (1.0 + lam2[j] * tau2);
                    }
                }
            }
        }
    }
    let nd = draws.len() as f64;
    // a model shell for the shared table and stored results
    let mut m = Model::default();
    for nm in &names {
        m.params.push(Param { name: format!("{dep}:{nm}"), support: Support::Real, re: None, mat: None });
    }
    m.params.push(Param { name: "sigma2".into(), support: Support::Lower(0.0), re: None, mat: None });
    m.eqs.push(Eq { name: dep.clone(), depvar: dep.clone(), y: y.clone(), xnames: names.clone(), x: x.clone(), coefs: (0..k).collect(), offset: Vec::new(), nl: None, re: Vec::new() });
    m.liks.push(Lik { eqs: vec![0], fam: Fam::Normal(Arg::Par(k)) });
    m.n = n;
    m.mask = d.mask.clone();
    m.finalize();
    let sim = Sim { draws, ll: lls, lp: vec![0.0; nd as usize], chain: vec![0; nd as usize], arate: vec![1.0], blocks: Vec::new(), any_gibbs: true, any_mh: false, start: Vec::new() };
    let idx: Vec<usize> = (0..=k).collect();
    let (tbl, sums) = run::table(&m, &sim, &idx, level, use_hpd, 0);
    let (dic, _, dbar) = run::ic(&m, &sim);
    let effs: Vec<f64> = sums.iter().map(|s| s.ess / nd).collect();
    let (emin, emax) = effs.iter().fold((f64::INFINITY, 0.0f64), |a, &e| (a.0.min(e), a.1.max(e)));
    let eavg = stats::mean(&effs);
    let title = if ss { "variable selection (spike-and-slab prior)" } else { "variable selection (horseshoe prior)" };
    let mut out = vec![
        run::hdr(&format!("Bayesian {title}"), "MCMC iterations", &crate::util::commas(burnin + mcmcsize)),
        run::hdr("Gibbs sampling", "Burn-in", &crate::util::commas(burnin)),
        run::hdr("", "MCMC sample size", &crate::util::commas(mcmcsize)),
        run::hdr("", "Number of obs", &crate::util::commas(n)),
        run::hdr("", "Efficiency:  min", &format!("{emin:.4}")),
        run::hdr("", "             avg", &format!("{eavg:.4}")),
        run::hdr("", "             max", &format!("{emax:.4}")),
        String::new(),
    ];
    out.extend(tbl);
    // inclusion / shrinkage summary
    out.push(String::new());
    out.push(if ss { "Posterior inclusion probabilities".into() } else { "Posterior mean shrinkage factors 1/(1+lambda^2 tau^2) (near 1: shrunk to zero)".to_string() });
    let mut summ = Vec::new();
    let mut snames = Vec::new();
    for j in 0..k {
        if pen[j] {
            let v = if ss { incl[j] / nd } else { shrink[j] / nd };
            out.push(format!("{:>12} | {:>8.4}", names[j], v));
            summ.push(v);
            snames.push(names[j].clone());
        }
    }
    let settings = Settings { mcmcsize, burnin, ..Default::default() };
    run::store_results(st, &m, &sim, &idx, &sums, run::StoreArgs { rest, level, dic, lml: f64::NAN, dbar, arate: 1.0, emin, eavg, emax, rcmax: f64::NAN, rc: &[], settings: &settings, prefix_cmd: None, title });
    st.e.set_str("cmd", "bayesselect");
    st.e.set_str("_spec", String::new());
    st.e.set_str("prior", if ss { "spikeslab" } else { "horseshoe" });
    st.e.set_mat(if ss { "inclusion" } else { "shrinkage" }, Matrix::row_vector(&summ).with_names(vec![if ss { "pip".into() } else { "kappa".into() }], snames));
    st.e.set_str("_output", out.join("\n"));
    Ok(out.join("\n"))
}
