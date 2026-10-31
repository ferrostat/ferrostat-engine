//! `_robust`: the programmer's sandwich variance estimator.
//!
//! `_robust scorevars [weight] [if] [in] [, variance(matname) minus(#)
//! cluster(varname)]` replaces the "bread" D (the conventional variance:
//! `variance()` or e(V)) by D M D · q, where M = Σ u_i u_i' with u_i the
//! score of observation i times its covariates (one score variable per
//! equation of e(b), covariates from the column names, `_cons` = 1), summed
//! within clusters when `cluster()` is given. The small-sample multiplier is
//! q = N/(N-k) with k = `minus()` (default 1), or (N-1)/(N-k) · G/(G-1)
//! with G clusters. Probability and importance weights multiply the scores,
//! frequency weights the terms of M.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util::{self, WeightKind};
use crate::varlist;

pub struct Robust;

impl CommandHandler for Robust {
    fn name(&self) -> &'static str {
        "_robust"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let scores = varlist::expand(ds, &cmd.main)?;
        if scores.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        for o in ["strata", "psu", "fpc", "subpop"] {
            if cmd.has_opt(o, 3) {
                return Err(rc_err(198, format!("option {o}() not supported by _robust; use svy")));
            }
        }
        let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let vname = cmd.opt("variance", 1).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let d = match &vname {
            Some(n) => st.matrices.get(n).cloned().ok_or_else(|| rc_err(111, format!("matrix {n} not found")))?,
            None => st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?,
        };
        let k = b.cols;
        if d.rows != k || d.cols != k {
            return Err(rc_err(503, "conformability error: variance matrix and e(b)"));
        }
        // equations of e(b) in order of appearance
        let mut eqs: Vec<String> = Vec::new();
        let eq_of: Vec<String> = b.colnames.iter().map(|c| c.rsplit_once(':').map_or(String::new(), |x| x.0.to_string())).collect();
        for e in &eq_of {
            if !eqs.contains(e) {
                eqs.push(e.clone());
            }
        }
        if scores.len() != eqs.len() {
            return Err(rc_err(198, format!("{} score variables for {} equations", scores.len(), eqs.len())));
        }
        let minus: f64 = cmd.opt("minus", 1).map(|s| s.trim().parse::<f64>().map_err(|_| rc_err(198, "minus() must be a number"))).transpose()?.unwrap_or(1.0);
        let mut mask = util::touse(ds, st, cmd)?;
        if let Some(s) = &st.e.sample {
            if s.len() == mask.len() && !cmd.has_opt("zeroweight", 4) && cmd.if_raw.is_none() && cmd.in_range.is_none() {
                for i in 0..mask.len() {
                    mask[i] &= s[i];
                }
            }
        }
        let w = util::weights(ds, st, cmd, &mut mask, "fpi")?;
        let sc: Vec<Vec<f64>> = scores.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        // covariate columns
        let mut xcols: Vec<Option<Vec<f64>>> = Vec::with_capacity(k);
        for c in &b.colnames {
            let name = c.rsplit(':').next().unwrap_or(c);
            if name == "_cons" {
                xcols.push(None);
            } else {
                let v = column(ds, name)?;
                xcols.push(Some(v));
            }
        }
        let cl = match cmd.opt("cluster", 2).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            Some(c) => Some((c.clone(), util::col(ds, &c)?)),
            None => None,
        };
        for i in 0..ds.n_obs {
            if mask[i] && (sc.iter().any(|s| is_missing(s[i])) || xcols.iter().flatten().any(|x| is_missing(x[i])) || cl.as_ref().is_some_and(|c| is_missing(c.1[i]))) {
                mask[i] = false;
            }
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let eqidx: Vec<usize> = eq_of.iter().map(|e| eqs.iter().position(|x| x == e).unwrap()).collect();
        let score = |i: usize| -> Vec<f64> {
            let wm = match &w {
                Some(w) if matches!(w.kind, WeightKind::P | WeightKind::I | WeightKind::A) => w.w[i],
                _ => 1.0,
            };
            (0..k).map(|j| wm * sc[eqidx[j]][i] * xcols[j].as_ref().map_or(1.0, |x| x[i])).collect()
        };
        let fw = |i: usize| match &w {
            Some(w) if matches!(w.kind, WeightKind::F) => w.w[i],
            _ => 1.0,
        };
        let n: f64 = rows.iter().map(|&i| fw(i)).sum();
        let mut m = Matrix::zeros(k, k);
        let mut nclust = 0usize;
        match &cl {
            Some((_, c)) => {
                let mut groups: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
                for &i in &rows {
                    let u = score(i);
                    let g = groups.entry(c[i].to_bits()).or_insert_with(|| vec![0.0; k]);
                    for j in 0..k {
                        g[j] += fw(i) * u[j];
                    }
                }
                nclust = groups.len();
                for u in groups.values() {
                    for a in 0..k {
                        for bb in 0..k {
                            m[(a, bb)] += u[a] * u[bb];
                        }
                    }
                }
            }
            None => {
                for &i in &rows {
                    let u = score(i);
                    let f = fw(i);
                    for a in 0..k {
                        for bb in 0..k {
                            m[(a, bb)] += f * u[a] * u[bb];
                        }
                    }
                }
            }
        }
        let q = if cl.is_some() {
            let g = nclust as f64;
            if g < 2.0 {
                return Err(rc_err(459, "at least two clusters required"));
            }
            (n - 1.0) / (n - minus) * g / (g - 1.0)
        } else {
            n / (n - minus)
        };
        let v = d.mul(&m)?.mul(&d)?.scale(q);
        let v = Matrix::from_rows(&(0..k).map(|a| (0..k).map(|c| 0.5 * (v[(a, c)] + v[(c, a)])).collect()).collect::<Vec<_>>()).with_names(d.rownames.clone(), d.colnames.clone());
        match vname {
            Some(nm) => {
                st.matrices.insert(nm, v);
            }
            None => {
                st.e.set_mat("V", v);
                st.e.set_str("vcetype", "Robust");
                match &cl {
                    Some((name, _)) => {
                        st.e.set_str("vce", "cluster");
                        st.e.set_str("clustvar", name.clone());
                        st.e.set("N_clust", nclust as f64);
                    }
                    None => st.e.set_str("vce", "robust"),
                }
            }
        }
        Ok(String::new())
    }
}

/// Values of a coefficient's covariate: variables, `c.`/`o.` prefixes,
/// factor-level indicators (`2.g`, `1b.g`) and interactions of these.
fn column(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let mut out = vec![1.0; ds.n_obs];
    for part in name.split('#') {
        let (lvl, var) = match part.split_once('.') {
            Some((l, v)) => (Some(l), v),
            None => (None, part),
        };
        let x = util::col(ds, var).map_err(|_| rc_err(111, format!("variable {var} not found")))?;
        match lvl {
            Some(l) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => {
                let k: f64 = l.trim_end_matches(['b', 'o', 'n']).parse().unwrap();
                for i in 0..ds.n_obs {
                    out[i] *= if is_missing(x[i]) { x[i] } else if x[i] == k { 1.0 } else { 0.0 };
                }
            }
            Some("c") | Some("o") | Some("co") | None => {
                for i in 0..ds.n_obs {
                    out[i] *= x[i];
                }
            }
            Some(op) => return Err(rc_err(198, format!("operator {op}. not supported by _robust; generate the variable first"))),
        }
    }
    Ok(out)
}
