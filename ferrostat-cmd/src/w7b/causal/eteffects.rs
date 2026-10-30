//! `eteffects`: treatment effects with an endogenous binary treatment by
//! the control-function approach.
//!
//!   eteffects (ovar omvarlist [, linear|probit|fractional|exponential noconstant])
//!             (tvar tmvarlist [, noconstant]) [if] [in] [, ate|atet|pomeans aequations vce(robust)]
//!
//! A probit treatment model gives the generalized residual v; within each
//! treatment group the outcome is modelled as g(x'b_k + rho_k v) (least
//! squares, or quasi-likelihood for probit/fractional/exponential means).
//! Potential-outcome means average g over the sample (ATE) or the treated
//! (ATET). All parameters solve one just-identified system of moment
//! conditions; the variance is the GMM sandwich, which accounts for the
//! estimated treatment model.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::util::{gmm_vce, mills, phi, table_matrix, ztable, zrow, ZRow};
use crate::commands::CommandHandler;
use crate::dist::normal;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::teffects::{self as te, OKind};
use crate::util;
use crate::xmodels::{self as xm, Handlers};

pub fn handlers() -> Handlers {
    vec![("eteffects", Box::new(Eteffects))]
}

struct Eteffects;

#[derive(Clone, Copy, PartialEq)]
enum Om {
    Linear,
    Probit,
    Exp,
}

impl Om {
    fn mean(self, z: f64) -> f64 {
        match self {
            Om::Linear => z,
            Om::Probit => normal(z),
            Om::Exp => z.min(700.0).exp(),
        }
    }
    /// quasi-likelihood score factor: d qll / d index = (y - m) * s(z)
    fn sfac(self, z: f64) -> f64 {
        match self {
            Om::Linear | Om::Exp => 1.0,
            Om::Probit => {
                let p = normal(z).clamp(1e-12, 1.0 - 1e-12);
                phi(z) / (p * (1.0 - p))
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl CommandHandler for Eteffects {
    fn name(&self) -> &'static str {
        "eteffects"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("eteffects") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let groups = xm::paren_groups(&cmd.main);
        if groups.len() != 2 {
            return Err(rc_err(198, "eteffects requires (outcome model) (treatment model)"));
        }
        let (ow, oopt) = te::model(&groups[0]);
        let (tw, topt) = te::model(&groups[1]);
        if ow.is_empty() || tw.is_empty() {
            return Err(rc_err(198, "invalid syntax"));
        }
        let has = |o: &str, w: &str, min: usize| o.split(|c: char| c == ',' || c.is_whitespace()).any(|x| x.len() >= min && w.starts_with(x));
        let om = if has(&oopt, "probit", 2) || has(&oopt, "fractional", 4) {
            Om::Probit
        } else if has(&oopt, "exponential", 3) || has(&oopt, "poisson", 3) {
            Om::Exp
        } else {
            Om::Linear
        };
        let ovar = xm::var_name(ds, &ow[0])?;
        let tvar = xm::var_name(ds, &tw[0])?;
        let ox: Vec<String> = ow[1..].to_vec();
        let tx: Vec<String> = tw[1..].to_vec();
        let mut extra = te::base_vars(ds, &ox)?;
        extra.extend(te::base_vars(ds, &tx)?);
        extra.push(ovar.clone());
        extra.push(tvar.clone());
        let dout = te::sub_design(ds, st, cmd, &ovar, &ox, has(&oopt, "noconstant", 3), extra.clone())?;
        let dtr = te::sub_design(ds, st, cmd, &tvar, &tx, has(&topt, "noconstant", 3), extra)?;
        if dout.rows != dtr.rows {
            return Err(rc_err(459, "models have different estimation samples"));
        }
        let xr = te::design_rows(&dout);
        let zr = te::design_rows(&dtr);
        let y = dout.y.clone();
        let t = dtr.y.clone();
        let n = y.len();
        if t.iter().any(|v| *v != 0.0 && *v != 1.0) {
            return Err(rc_err(450, format!("{tvar} must be a 0/1 variable")));
        }
        if om == Om::Probit && y.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(rc_err(459, format!("{ovar} must be between 0 and 1")));
        }
        let stat = if cmd.has_opt("atet", 4) {
            "atet"
        } else if cmd.has_opt("pomeans", 3) {
            "pomeans"
        } else {
            "ate"
        };
        let level = util::level(cmd, st)?;
        // first step: probit
        let gamma = te::fit_binary(&zr, &t, te::Link::Probit)?;
        let gres = |g: &[f64], i: usize| -> f64 {
            let z = dot(&zr[i], g);
            if t[i] != 0.0 {
                mills(z)
            } else {
                -mills(-z)
            }
        };
        let kz = gamma.len();
        let kx = xr[0].len() + 1;
        let rrow = |g: &[f64], i: usize| -> Vec<f64> {
            let mut r = xr[i].clone();
            r.push(gres(g, i));
            r
        };
        // outcome models by group
        let okind = match om {
            Om::Linear => OKind::Linear,
            Om::Probit => OKind::Probit,
            Om::Exp => OKind::Poisson,
        };
        let mut betas: Vec<Vec<f64>> = Vec::new();
        for k in [0.0, 1.0] {
            let idx: Vec<usize> = (0..n).filter(|&i| t[i] == k).collect();
            if idx.len() < kx {
                return Err(rc_err(2001, "insufficient observations in a treatment group"));
            }
            let rows: Vec<Vec<f64>> = idx.iter().map(|&i| rrow(&gamma, i)).collect();
            let yy: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
            betas.push(te::fit_glm(&rows, &yy, &vec![1.0; idx.len()], okind)?);
        }
        let atet = stat == "atet";
        let sel: Vec<f64> = (0..n).map(|i| if atet { t[i] } else { 1.0 }).collect();
        let ns: f64 = sel.iter().sum();
        let pom = |g: &[f64], b: &[f64], i: usize| -> f64 { om.mean(dot(&rrow(g, i), b)) };
        let pm: Vec<f64> = (0..2).map(|k| (0..n).map(|i| sel[i] * pom(&gamma, &betas[k], i)).sum::<f64>() / ns).collect();
        // parameter vector: gamma, beta0, beta1, then (ATE, POM0) or (POM0, POM1)
        let mut theta = gamma.clone();
        theta.extend(betas[0].iter().cloned());
        theta.extend(betas[1].iter().cloned());
        let (e1, e2) = if stat == "pomeans" { (pm[0], pm[1]) } else { (pm[1] - pm[0], pm[0]) };
        theta.push(e1);
        theta.push(e2);
        let ktot = theta.len();
        let mom = |th: &[f64]| -> Vec<Vec<f64>> {
            let g = &th[..kz];
            let b0 = &th[kz..kz + kx];
            let b1 = &th[kz + kx..kz + 2 * kx];
            let (q1, q2) = (th[ktot - 2], th[ktot - 1]);
            (0..n)
                .map(|i| {
                    let mut m = vec![0.0; ktot];
                    let z = dot(&zr[i], g);
                    let p = normal(z).clamp(1e-12, 1.0 - 1e-12);
                    let s = (t[i] - p) * phi(z) / (p * (1.0 - p));
                    for a in 0..kz {
                        m[a] = zr[i][a] * s;
                    }
                    let r = rrow(g, i);
                    let (bk, off) = if t[i] == 0.0 { (b0, kz) } else { (b1, kz + kx) };
                    let e = dot(&r, bk);
                    let u = (y[i] - om.mean(e)) * om.sfac(e);
                    for a in 0..kx {
                        m[off + a] = r[a] * u;
                    }
                    let g0 = om.mean(dot(&r, b0));
                    let g1 = om.mean(dot(&r, b1));
                    if stat == "pomeans" {
                        m[ktot - 2] = sel[i] * (g0 - q1);
                        m[ktot - 1] = sel[i] * (g1 - q2);
                    } else {
                        m[ktot - 2] = sel[i] * (g1 - g0 - q1);
                        m[ktot - 1] = sel[i] * (g0 - q2);
                    }
                    m
                })
                .collect()
        };
        let cl = match crate::est::parse_vce(cmd, crate::est::Vce::Robust)? {
            crate::est::Vce::Cluster(c) => {
                let col = util::col(ds, &c)?;
                Some(dout.rows.iter().map(|&r| col[r]).collect::<Vec<f64>>())
            }
            _ => None,
        };
        let v = gmm_vce(&mom, &theta, cl.as_deref());
        // names
        let mut names: Vec<String> = Vec::new();
        let tn: Vec<String> = dtr.active().iter().map(|&j| dtr.names[j].clone()).collect();
        let on: Vec<String> = dout.active().iter().map(|&j| dout.names[j].clone()).collect();
        for c in &tn {
            names.push(format!("TME1:{c}"));
        }
        for k in 0..2 {
            for c in &on {
                names.push(format!("OME{k}:{c}"));
            }
            names.push(format!("OME{k}:/{tvar}"));
        }
        let (n1, n2) = match stat {
            "pomeans" => (format!("POmeans:0.{tvar}"), format!("POmeans:1.{tvar}")),
            "atet" => (format!("ATET:r1vs0.{tvar}"), format!("POmean:0.{tvar}")),
            _ => (format!("ATE:r1vs0.{tvar}"), format!("POmean:0.{tvar}")),
        };
        // move the effect rows first
        let mut order: Vec<usize> = vec![ktot - 2, ktot - 1];
        order.extend(0..ktot - 2);
        // reorder in output: effects, then (with aequations) the models
        names.push(n1.clone());
        names.push(n2.clone());
        let bfull: Vec<f64> = order.iter().map(|&j| theta[j]).collect();
        let nfull: Vec<String> = order.iter().map(|&j| names[j].clone()).collect();
        let mut vfull = Matrix::zeros(ktot, ktot);
        for (a, &p) in order.iter().enumerate() {
            for (c, &q) in order.iter().enumerate() {
                vfull[(a, c)] = v[(p, q)];
            }
        }
        let se: Vec<f64> = (0..ktot).map(|j| vfull[(j, j)].max(0.0).sqrt()).collect();
        let show = if cmd.has_opt("aequations", 2) { ktot } else { 2 };
        let mut rows: Vec<ZRow> = Vec::new();
        for j in 0..show {
            let (eq, lab) = nfull[j].split_once(':').unwrap();
            let label = match lab.split_once('.') {
                Some((l, _)) if j < 2 => {
                    if l == "r1vs0" {
                        "(1 vs 0)".to_string()
                    } else {
                        l.to_string()
                    }
                }
                _ => lab.trim_start_matches('/').to_string(),
            };
            let eqh = if j < 2 { format!("{eq} {tvar}") } else { eq.to_string() };
            rows.push(zrow(Some(&eqh), &label, bfull[j], se[j]));
        }
        let mut out = vec![
            format!("Endogenous treatment-effects estimation{:>w$}", format!("Number of obs     = {:>10}", util::commas(n)), w = 78 - 39),
            "Outcome model:   ".to_string() + match om {
                Om::Linear => "linear",
                Om::Probit => "probit",
                Om::Exp => "exponential",
            },
            "Treatment model: probit".to_string(),
            String::new(),
        ];
        out.extend(ztable("", "Robust", &rows, level));
        let text = out.join("\n");
        let tm = table_matrix(&nfull[..show].to_vec(), &bfull[..show], &se[..show], level);
        let mut d = dout.clone();
        d.names = nfull.clone();
        d.omitted = vec![false; ktot];
        d.is_base = vec![false; ktot];
        crate::est::post(st, "eteffects", &d, &bfull, &vfull, &|e| {
            e.set_str("depvar", ovar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("stat", stat);
            e.set_str("omodel", match om {
                Om::Linear => "linear",
                Om::Probit => "probit",
                Om::Exp => "exponential",
            });
            e.set_str("title", "Endogenous treatment-effects estimation");
            e.set_str("vce", if cl.is_some() { "cluster" } else { "robust" });
            e.set_str("vcetype", "Robust");
            e.set_str("_output", text.clone());
            e.set("N", n as f64);
            e.set("treated", 1.0);
            e.set("control", 0.0);
        });
        crate::est::stash_table(tm, level);
        Ok(text)
    }
}
