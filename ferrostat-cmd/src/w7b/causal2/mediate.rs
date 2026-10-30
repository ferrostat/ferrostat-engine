//! Causal mediation analysis.
//!
//!   mediate (ovar omvars [, linear|logit|probit|poisson]) (mvar mmvars [, linear|logit|probit|poisson])
//!        (tvar) [if] [in] [, nie nde te pomeans all control(#) tlevel(#) nointeraction level(#)]
//!   estat proportion
//!
//! The mediator model regresses the mediator on treatment indicators and
//! its covariates; the outcome model regresses the outcome on treatment
//! indicators, the mediator, their interaction (unless `nointeraction`)
//! and its covariates. Potential-outcome means E[Y(t, M(t'))] average, over
//! the sample, the outcome model's mean at treatment t integrated over the
//! mediator's conditional distribution at treatment t' (normal by
//! Gauss-Hermite quadrature, Bernoulli or Poisson by summation). For each
//! treatment level l against the control c:
//!
//!   NIE = E[Y(l, M(l))] - E[Y(l, M(c))]
//!   NDE = E[Y(l, M(c))] - E[Y(c, M(c))]
//!   TE  = NIE + NDE
//!
//! Multivalued treatments give one row per level. Standard errors stack
//! the score equations of both models, the mediator variance and the
//! potential-outcome means. Continuous treatments are not supported.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::{dot, fam_label, family_of, fit, fmtv, groups, levels, sandwich, score, solve_block};
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
    vec![("mediate", Box::new(Mediate))]
}

pub struct Mediate;

/// Probability mass / quadrature of the mediator at index `eta`:
/// (values, weights).
fn mediator_support(f: Family, eta: f64, sigma: f64, gh: &(Vec<f64>, Vec<f64>)) -> Vec<(f64, f64)> {
    match f {
        Family::Linear => gh.0.iter().zip(&gh.1).map(|(x, w)| (eta + sigma * x, *w)).collect(),
        Family::Logit | Family::Probit => {
            let p = f.mean(eta);
            vec![(0.0, 1.0 - p), (1.0, p)]
        }
        Family::Poisson => {
            let mu = f.mean(eta).max(1e-300);
            let mut out = Vec::new();
            let mut pk = (-mu).exp();
            let mut cum = 0.0;
            let mut k = 0.0;
            while k < 2000.0 {
                out.push((k, pk));
                cum += pk;
                if cum > 1.0 - 1e-12 && k > mu {
                    break;
                }
                k += 1.0;
                pk *= mu / k;
            }
            out
        }
    }
}

impl CommandHandler for Mediate {
    fn name(&self) -> &'static str {
        "mediate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("mediate") {
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(rc_err(301, "last estimates not found"));
        }
        let g = groups(main);
        if g.len() != 3 {
            return Err(rc_err(198, "mediate requires an outcome, a mediator and a treatment model in parentheses"));
        }
        let (om, oopts) = model(&g[0]);
        let (mm, mopts) = model(&g[1]);
        let (tm, topts) = model(&g[2]);
        if topts.contains("continuous") {
            return Err(rc_err(198, "continuous treatments are not supported"));
        }
        let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
        let mvar = varlist::resolve_name(ds, mm.first().ok_or_else(|| rc_err(198, "mediator required"))?)?;
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let (ox, mx) = (om[1..].to_vec(), mm[1..].to_vec());
        let fo = family_of(&oopts, Family::Linear);
        let fm = family_of(&mopts, Family::Linear);
        let interact = !cmd.has_opt("nointeraction", 5);
        let level = util::level(cmd, st)?;
        let mut want: Vec<&str> = ["nie", "nde", "te"].into_iter().filter(|w| cmd.has_opt(w, 3)).collect();
        let pomeans = cmd.has_opt("pomeans", 3);
        if want.is_empty() && !pomeans || cmd.has_opt("all", 3) {
            want = vec!["nie", "nde", "te"];
        }
        let mut all = vec![ovar.clone(), mvar.clone(), tvar.clone()];
        all.extend(base_vars(ds, &ox)?);
        all.extend(base_vars(ds, &mx)?);
        let od = sub_design(ds, st, cmd, &ovar, &ox, false, all.clone())?;
        let md = sub_design(ds, st, cmd, &mvar, &mx, false, all.clone())?;
        let tcol = util::col(ds, &tvar)?;
        if od.rows != md.rows {
            return Err(rc_err(459, "models use different samples"));
        }
        let n = od.rows.len();
        if n < 5 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let y = od.y.clone();
        let m = md.y.clone();
        let t: Vec<f64> = od.rows.iter().map(|&r| tcol[r]).collect();
        let lv = levels(&t, &tvar)?;
        let ctrl = util::opt_num(cmd, "control", 4)?.unwrap_or(lv[0]);
        let ci = lv.iter().position(|v| *v == ctrl).ok_or_else(|| rc_err(459, "control() is not a treatment level"))?;
        let mut order = vec![ci];
        order.extend((0..lv.len()).filter(|&l| l != ci));
        let lvals: Vec<f64> = order.iter().map(|&o| lv[o]).collect();
        let nl = lvals.len();
        let shown: Vec<usize> = match util::opt_num(cmd, "tlevel", 4)? {
            Some(v) => vec![lvals.iter().position(|x| *x == v).filter(|&p| p > 0).ok_or_else(|| rc_err(459, "tlevel() is not a treatment level"))?],
            None => (1..nl).collect(),
        };
        let lev: Vec<usize> = t.iter().map(|v| lvals.iter().position(|l| l == v).unwrap()).collect();
        let mcov = design_rows(&md);
        let ocov = design_rows(&od);
        let dummies = |l: usize| -> Vec<f64> { (1..nl).map(|k| if k == l { 1.0 } else { 0.0 }).collect() };
        let xm_of = |l: usize, i: usize| -> Vec<f64> { dummies(l).into_iter().chain(mcov[i].iter().copied()).collect() };
        let xo_of = |l: usize, mv: f64, i: usize| -> Vec<f64> {
            let d = dummies(l);
            let mut r = d.clone();
            r.push(mv);
            if interact {
                r.extend(d.iter().map(|v| v * mv));
            }
            r.extend(ocov[i].iter().copied());
            r
        };
        let km = nl - 1 + mcov[0].len();
        let ko = xo_of(0, 0.0, 0).len();
        let has_sig = fm == Family::Linear;
        // pairs (t, t') of potential-outcome means
        let mut pairs: Vec<(usize, usize)> = (0..nl).map(|l| (l, l)).collect();
        for l in 1..nl {
            pairs.push((l, 0));
        }
        let oa = 0;
        let osig = km;
        let ob = km + usize::from(has_sig);
        let op = ob + ko;
        let np = op + pairs.len();
        let xm: Vec<Vec<f64>> = (0..n).map(|i| xm_of(lev[i], i)).collect();
        let xo: Vec<Vec<f64>> = (0..n).map(|i| xo_of(lev[i], m[i], i)).collect();
        let alpha = fit(&xm, &m, &vec![1.0; n], fm)?;
        let beta = fit(&xo, &y, &vec![1.0; n], fo)?;
        let mut theta = vec![0.0; np];
        theta[oa..oa + km].copy_from_slice(&alpha);
        if has_sig {
            let s2 = (0..n).map(|i| (m[i] - dot(&xm[i], &alpha)).powi(2)).sum::<f64>() / n as f64;
            theta[osig] = s2.sqrt();
        }
        theta[ob..ob + ko].copy_from_slice(&beta);
        let gh = {
            let (x, w) = crate::melogit::gauss_hermite(20);
            let s2 = 2f64.sqrt();
            let sp = std::f64::consts::PI.sqrt();
            (x.iter().map(|v| v * s2).collect::<Vec<f64>>(), w.iter().map(|v| v / sp).collect::<Vec<f64>>())
        };
        let moments = |th: &[f64], i: usize, mo: &mut [f64]| {
            let ea = dot(&xm[i], &th[oa..oa + km]);
            let sa = score(fm, m[i], ea);
            for c in 0..km {
                mo[oa + c] = xm[i][c] * sa;
            }
            if has_sig {
                mo[osig] = (m[i] - ea).powi(2) - th[osig] * th[osig];
            }
            let b = &th[ob..ob + ko];
            let sb = score(fo, y[i], dot(&xo[i], b));
            for c in 0..ko {
                mo[ob + c] = xo[i][c] * sb;
            }
            let sig = if has_sig { th[osig].abs() } else { 0.0 };
            for (q, &(tt, tm)) in pairs.iter().enumerate() {
                let em = dot(&xm_of(tm, i), &th[oa..oa + km]);
                let psi = if fm == Family::Linear && fo == Family::Linear {
                    dot(&xo_of(tt, em, i), b)
                } else {
                    mediator_support(fm, em, sig, &gh).iter().map(|(mv, w)| w * fo.mean(dot(&xo_of(tt, *mv, i), b))).sum::<f64>()
                };
                mo[op + q] = psi - th[op + q];
            }
        };
        solve_block(&mut theta, n, np, op, &moments)?;
        let v = sandwich(&theta, n, np, &moments)?;
        let pidx = |a: (usize, usize)| pairs.iter().position(|p| *p == a).unwrap();
        let mut names = Vec::new();
        let mut cm: Vec<Vec<f64>> = Vec::new();
        let mut blocks = Vec::new();
        let np2 = pairs.len();
        let tname = |l: usize| level_label(ds, &tvar, lvals[l]);
        for w in &want {
            let mut bl = Block { title: w.to_uppercase(), rows: Vec::new() };
            for &l in &shown {
                let mut c = vec![0.0; np2];
                match *w {
                    "nie" => {
                        c[pidx((l, l))] += 1.0;
                        c[pidx((l, 0))] -= 1.0;
                    }
                    "nde" => {
                        c[pidx((l, 0))] += 1.0;
                        c[pidx((0, 0))] -= 1.0;
                    }
                    _ => {
                        c[pidx((l, l))] += 1.0;
                        c[pidx((0, 0))] -= 1.0;
                    }
                }
                cm.push(c);
                names.push(format!("{}:r{}vs{}.{tvar}", w.to_uppercase(), fmtv(lvals[l]), fmtv(lvals[0])));
                bl.rows.push((wrap12(&format!("({} vs {})", tname(l), tname(0))), 0.0, 0.0));
            }
            blocks.push(bl);
        }
        if pomeans {
            let mut bl = Block { title: "POmeans".into(), rows: Vec::new() };
            for (q, &(tt, tm)) in pairs.iter().enumerate() {
                let mut c = vec![0.0; np2];
                c[q] = 1.0;
                cm.push(c);
                names.push(format!("POmeans:{}_{}.{tvar}", fmtv(lvals[tt]), fmtv(lvals[tm])));
                bl.rows.push((wrap12(&format!("{} {}", tname(tt), tname(tm))), 0.0, 0.0));
            }
            blocks.push(bl);
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
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Causal mediation analysis", util::commas(n)));
        out.push(format!("Outcome model  : {}{}", fam_label(fo), if interact { " (with treatment-mediator interaction)" } else { "" }));
        out.push(format!("Mediator model : {}", fam_label(fm)));
        out.push(format!("Mediator       : {mvar}"));
        out.push(format!("Treatment type : {}", if nl == 2 { "binary" } else { "multivalued" }));
        effects_table(&mut out, &ovar, &tvar, "Robust", level, &blocks);
        let text = out.join("\n");
        let rows = od.rows.clone();
        super::post(st, ds, &rows, "mediate", &b, &vv, &names, &|e: &mut Results| {
            e.set_str("depvar", ovar.clone());
            e.set_str("mvar", mvar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("omodel", fam_label(fo));
            e.set_str("mmodel", fam_label(fm));
            e.set_str("title", "Causal mediation analysis");
            e.set("k_levels", nl as f64);
            e.set("control", lvals[0]);
            e.set("level", level);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// `estat proportion`: share of the total effect that is mediated.
pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    if !matches!(sub, "proportion" | "prop") {
        return Err(rc_err(321, format!("estat {sub} not valid after mediate")));
    }
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let level = util::level(cmd, st)?;
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut out = vec![String::new(), "Proportion mediated".into(), "-".repeat(78)];
    out.push(format!("{:>12} | Proportion  std. err.      z    P>|z|     [{}% conf. interval]", "", fmtv(level)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    let mut vals = Vec::new();
    for (j, c) in b.colnames.iter().enumerate() {
        let Some(rest) = c.strip_prefix("NIE:") else { continue };
        let Some(k) = b.colnames.iter().position(|x| x == &format!("TE:{rest}")) else { continue };
        let (a, t) = (b.get(0, j), b.get(0, k));
        let r = a / t;
        let g = [1.0 / t, -a / (t * t)];
        let var = g[0] * g[0] * v.get(j, j) + 2.0 * g[0] * g[1] * v.get(j, k) + g[1] * g[1] * v.get(k, k);
        let se = var.max(0.0).sqrt();
        let z = r / se;
        let p = 2.0 * (1.0 - dist::normal(z.abs()));
        let lab = rest.split('.').next().unwrap_or(rest);
        out.push(format!("{:>12} |  {:>9}  {:>9}  {:>7.2}   {:>5.3}    {:>9}   {:>9}", lab, crate::format::fmt_g(r, 7), crate::format::fmt_g(se, 7), z, p, crate::format::fmt_g(r - crit * se, 7), crate::format::fmt_g(r + crit * se, 7)));
        vals.push((r, se));
    }
    if vals.is_empty() {
        return Err(rc_err(321, "estat proportion requires the NIE and TE estimates"));
    }
    out.push("-".repeat(78));
    let mut r = Results::default();
    r.set("prop", vals[0].0);
    r.set("se", vals[0].1);
    st.r = r;
    Ok(out.join("\n"))
}
