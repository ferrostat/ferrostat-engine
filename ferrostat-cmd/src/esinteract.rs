//! `eventstudyinteract`: the interaction-weighted event-study estimator of
//! Sun and Abraham (2021).
//!
//! Cohort-specific effects come from one regression of the outcome on the
//! relative-time indicators interacted with cohort dummies (with absorbed
//! fixed effects, through `reghdfe`); they are averaged with the cohort
//! shares at each relative time. The variance adds the uncertainty of the
//! estimated shares (a heteroskedasticity-robust sandwich of the share
//! regressions) to that of the interacted coefficients.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::did_imputation::{add_temps, drop_temps, run_line};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

pub struct EventStudyInteract;

fn col(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let v = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if v.var_type.is_string() {
        return Err(rc_err(109, format!("{name}: string variables not allowed")));
    }
    Ok(v.data.iter().map(|x| x.num()).collect())
}

impl CommandHandler for EventStudyInteract {
    fn name(&self) -> &'static str {
        "eventstudyinteract"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "eventstudyinteract");
        }
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let (lhs, rel) = vars.split_first().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        if rel.is_empty() {
            return Err(rc_err(102, "relative time indicators required"));
        }
        let absorb = cmd.opt("absorb", 1).ok_or_else(|| rc_err(198, "option absorb() required"))?.trim().to_string();
        let cohort = crate::varlist::resolve_name(ds, cmd.opt("cohort", 6).ok_or_else(|| rc_err(198, "option cohort() required"))?.trim())?;
        let control = crate::varlist::resolve_name(ds, cmd.opt("control_cohort", 9).ok_or_else(|| rc_err(198, "option control_cohort() required"))?.trim())?;
        let covs = match cmd.opt("covariates", 3) {
            Some(c) => crate::varlist::expand(ds, c)?,
            None => Vec::new(),
        };
        let vce = cmd.opt("vce", 3).map(|s| s.trim().to_string());
        let n = ds.n_obs;
        let mut touse = util::touse(ds, st, cmd)?;
        let mut need: Vec<String> = vars.clone();
        need.extend([cohort.clone(), control.clone()]);
        need.extend(covs.iter().cloned());
        for tok in absorb.split_whitespace() {
            let body = tok.split_once('=').map_or(tok, |(_, b)| b);
            for p in body.split('#') {
                let p = p.trim_start_matches("i.").trim_start_matches("c.");
                if !p.is_empty() {
                    need.push(crate::varlist::resolve_name(ds, p)?);
                }
            }
        }
        util::markout(ds, &mut touse, &need);
        let wts = util::weights(ds, st, cmd, &mut touse, "afip")?;
        let w = wts.as_ref().map_or(vec![1.0; n], |w| w.w.clone());
        let cc = col(ds, &cohort)?;
        let ctl = col(ds, &control)?;
        let relc: Vec<Vec<f64>> = rel.iter().map(|v| col(ds, v)).collect::<Result<_, _>>()?;
        // relative-time indicators with the control cohort set to zero
        let nvar: Vec<Vec<f64>> = relc.iter().map(|c| (0..n).map(|i| if ctl[i] == 1.0 { 0.0 } else { c[i] }).collect()).collect();
        let tr: Vec<usize> = (0..n).filter(|&i| touse[i] && ctl[i] == 0.0).collect();
        let mut cohorts: Vec<f64> = tr.iter().map(|&i| cc[i]).filter(|v| !is_missing(*v)).collect();
        cohorts.sort_by(|a, b| a.total_cmp(b));
        cohorts.dedup();
        let nc = cohorts.len();
        let nr = rel.len();
        if nc == 0 {
            return Err(rc_err(2000, "no treated cohorts in the sample"));
        }
        // cohort shares: regress cohort indicators on the indicators (no constant)
        let nt = tr.len();
        let wsum: f64 = tr.iter().map(|&i| w[i]).sum();
        let wn: Vec<f64> = tr.iter().map(|&i| w[i] * nt as f64 / wsum).collect();
        let mut xx = Matrix::zeros(nr, nr);
        for a in 0..nr {
            for b in 0..nr {
                xx[(a, b)] = tr.iter().zip(&wn).map(|(&i, wi)| wi * nvar[a][i] * nvar[b][i]).sum();
            }
        }
        let (xxi, _) = xx.sym_inverse(1e-12);
        let mut ff = Matrix::zeros(nc, nr);
        let mut resid: Vec<Vec<f64>> = Vec::new();
        for (c, &yy) in cohorts.iter().enumerate() {
            let d: Vec<f64> = tr.iter().map(|&i| if cc[i] == yy { 1.0 } else { 0.0 }).collect();
            let xy: Vec<f64> = (0..nr).map(|a| tr.iter().enumerate().map(|(p, &i)| wn[p] * nvar[a][i] * d[p]).sum()).collect();
            let bb: Vec<f64> = (0..nr).map(|a| (0..nr).map(|b| xxi[(a, b)] * xy[b]).sum()).collect();
            for a in 0..nr {
                ff[(c, a)] = bb[a];
            }
            resid.push(tr.iter().enumerate().map(|(p, &i)| d[p] - (0..nr).map(|a| bb[a] * nvar[a][i]).sum::<f64>()).collect());
        }
        let nf = nt as f64;
        let sxxi = xxi.scale(nf);
        // S = (1/N) sum (r_i (x) x_i)(r_i (x) x_i)'
        let dim = nc * nr;
        let mut s = Matrix::zeros(dim, dim);
        for p in 0..nt {
            let i = tr[p];
            let z: Vec<f64> = (0..dim).map(|q| resid[q / nr][p] * nvar[q % nr][i]).collect();
            for a in 0..dim {
                if z[a] == 0.0 {
                    continue;
                }
                for b in 0..dim {
                    s[(a, b)] += wn[p] * z[a] * z[b] / nf;
                }
            }
        }
        let mut k = Matrix::zeros(dim, dim);
        for c in 0..nc {
            for a in 0..nr {
                for b in 0..nr {
                    k[(c * nr + a, c * nr + b)] = sxxi[(a, b)];
                }
            }
        }
        let sigma_ff = k.mul(&s)?.mul(&k)?.scale(1.0 / nf);
        // interacted regression
        let mut tcols: Vec<Vec<f64>> = Vec::new();
        let mut bnames: Vec<String> = Vec::new();
        for (l, nv) in nvar.iter().enumerate() {
            for &yy in &cohorts {
                tcols.push((0..n).map(|i| if cc[i] == yy { nv[i] } else { 0.0 }).collect());
                bnames.push(format!("{}_x_{}", rel[l], crate::format::fmt_g(yy, 12).trim()));
            }
        }
        tcols.push(touse.iter().map(|t| if *t { 1.0 } else { 0.0 }).collect());
        let temps = add_temps(ds, tcols)?;
        let wexp = match (&cmd.weight, &wts) {
            (Some(ws), Some(_)) => {
                let raw = cmd.raw.as_str();
                let a = raw.find('[').unwrap_or(0);
                let b = raw[a..].find(']').map_or(a, |p| a + p + 1);
                let _ = ws;
                raw[a..b].to_string()
            }
            _ => String::new(),
        };
        let line = format!(
            "reghdfe {} {} {} if {}==1 {}, absorb({}){}",
            lhs,
            temps[..dim].join(" "),
            covs.join(" "),
            temps[dim],
            wexp,
            absorb,
            vce.as_ref().map_or(String::new(), |v| format!(" vce({v})"))
        );
        let r = run_line(ds, st, &crate::reghdfe::Reghdfe, &line);
        drop_temps(ds, &temps);
        r?;
        let eb = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(498, "interacted regression failed"))?;
        let ev = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(498, "interacted regression failed"))?;
        let idx = |name: &str| eb.colnames.iter().position(|c| c == name || c.trim_start_matches("o.") == name);
        let mut delta = Matrix::zeros(nc, nr);
        let mut vint = Matrix::zeros(nc, nr);
        let pos: Vec<Option<usize>> = temps[..dim].iter().map(|t| idx(t)).collect();
        for l in 0..nr {
            for c in 0..nc {
                if let Some(j) = pos[l * nc + c] {
                    delta[(c, l)] = eb.data[j];
                    vint[(c, l)] = ev[(j, j)];
                }
            }
        }
        let b_iw: Vec<f64> = (0..nr).map(|l| (0..nc).map(|c| ff[(c, l)] * delta[(c, l)]).sum()).collect();
        let mut v_iw = Matrix::zeros(nr, nr);
        for i in 0..nr {
            for j in 0..nr {
                let mut v = 0.0;
                for a in 0..nc {
                    for b in 0..nc {
                        if let (Some(pa), Some(pb)) = (pos[i * nc + a], pos[j * nc + b]) {
                            v += ff[(a, i)] * ev[(pa, pb)] * ff[(b, j)];
                        }
                        v += delta[(a, i)] * sigma_ff[(a * nr + i, b * nr + j)] * delta[(b, j)];
                    }
                }
                v_iw[(i, j)] = v;
            }
        }
        // rename the interaction coefficients in e(b) and e(V)
        let rename = |names: &[String]| -> Vec<String> {
            names.iter().map(|c| match temps[..dim].iter().position(|t| c == t || c.trim_start_matches("o.") == t) {
                Some(p) => if c.starts_with("o.") { format!("o.{}", bnames[p]) } else { bnames[p].clone() },
                None => c.clone(),
            }).collect()
        };
        let df_r = st.e.get("df_r");
        let nobs = st.e.get("N").unwrap_or(0.0);
        let level = util::level(cmd, st)?;
        let mut out = vec![String::new()];
        out.extend(crate::est::header("IW estimates for dynamic effects", &[("Number of obs".to_string(), util::commas(nobs as usize))]));
        if let Some(nc_) = st.e.get("N_clust") {
            if let Some(cv) = st.e.macros.get("clustvar") {
                out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {cv})", util::commas(nc_ as usize))));
            }
        }
        out.push(String::new());
        let mut vdiag = Matrix::zeros(nr, nr);
        for i in 0..nr {
            vdiag[(i, i)] = v_iw[(i, i)];
        }
        out.extend(crate::est::coef_table(&crate::est::TableSpec { depvar: lhs, names: rel, b: &b_iw, v: &vdiag, omitted: &vec![false; nr], is_base: &vec![false; nr], df: df_r, level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let cohort_names: Vec<String> = cohorts.iter().map(|c| crate::format::fmt_g(*c, 12).trim().to_string()).collect();
        let relv = rel.to_vec();
        let e = &mut st.e;
        if let Some(m) = e.matrices.get_mut("b") {
            m.colnames = rename(&m.colnames);
        }
        if let Some(m) = e.matrices.get_mut("V") {
            m.colnames = rename(&m.colnames);
            m.rownames = rename(&m.rownames);
        }
        e.set_str("cmd", "eventstudyinteract");
        e.set_str("cmdline", format!("eventstudyinteract {}", cmd.raw.trim()));
        e.set_str("title", "IW estimates for dynamic effects");
        e.set_mat("b_iw", Matrix::row_vector(&b_iw).with_names(vec!["y1".into()], relv.clone()));
        e.set_mat("V_iw", v_iw.with_names(relv.clone(), relv.clone()));
        e.set_mat("ff_w", ff.with_names(cohort_names.clone(), relv.clone()));
        e.set_mat("Sigma_ff", sigma_ff);
        e.set_mat("b_interact", delta.with_names(cohort_names.clone(), relv.clone()));
        e.set_mat("V_interact", vint.with_names(cohort_names, relv));
        e.set_str("_output", text.clone());
        Ok(text)
    }
}
