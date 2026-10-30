//! `ml model ..., svy [subpop()]` (also `vce(linearized)` on svyset data):
//! maximum likelihood with the survey design. The sampling weights of the
//! design become probability weights, observations outside the
//! subpopulation stay in the design with zero scores, and the variance is
//! the linearized sandwich `V = H^-1 * Var(sum of weighted scores) * H^-1`
//! computed by the design code of the `svy` commands.

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::{Results, Store};
use ferrostat_cmd::svy::{self, DesignData, SvySet};
use ferrostat_cmd::util;
use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

#[derive(Debug, Clone)]
pub struct SvyInfo {
    pub set: SvySet,
    /// Rows of the design sample (the estimation sample before `subpop()`).
    pub design_rows: Vec<usize>,
    pub subpop: Option<String>,
}

/// Does this `ml model` ask for the survey design?
fn wants_svy(ds: &DataSet, c: &Command) -> bool {
    if c.has_opt("svy", 3) {
        return true;
    }
    let lin = c.opt("vce", 3).map(|v| v.trim().to_lowercase()).is_some_and(|v| v.len() >= 3 && "linearized".starts_with(v.as_str()));
    (lin || c.has_opt("subpop", 3)) && SvySet::from_chars(ds).is_some()
}

fn subpop_mask(ds: &DataSet, st: &Store, spec: &str) -> Result<Vec<bool>, String> {
    let spec = spec.trim();
    let (var, cond) = if let Some(c) = spec.strip_prefix("if ") {
        ("", Some(c.trim()))
    } else if let Some(i) = spec.find(" if ") {
        (spec[..i].trim(), Some(spec[i + 4..].trim()))
    } else {
        (spec, None)
    };
    let n = ds.n_obs;
    let mut mask = vec![true; n];
    if !var.is_empty() {
        let col = util::col(ds, &ferrostat_cmd::varlist::resolve_name(ds, var)?)?;
        for i in 0..n {
            mask[i] = !is_missing(col[i]) && col[i] != 0.0;
        }
    }
    if let Some(cnd) = cond {
        let e = parse_expression(cnd).map_err(|e| rc_err(198, e.message))?;
        let vals = util::eval_col(ds, st, &e, &vec![true; n])?;
        for i in 0..n {
            mask[i] = mask[i] && !is_missing(vals[i]) && vals[i] != 0.0;
        }
    }
    Ok(mask)
}

/// Restrict the sample to usable design rows and to the subpopulation.
pub fn prepare(ds: &DataSet, st: &Store, c: &Command, mask: &mut [bool]) -> Result<Option<SvyInfo>, String> {
    if !wants_svy(ds, c) {
        return Ok(None);
    }
    let set = svy::current(ds)?;
    if set.is_replicate() || set.vce != "linearized" {
        return Err(rc_err(198, "ml with svy supports linearized designs only"));
    }
    if c.weight.is_some() {
        return Err(rc_err(101, "weights not allowed with svy; the design weights are used"));
    }
    let usable = DesignData::usable(ds, &set);
    for (m, u) in mask.iter_mut().zip(&usable) {
        *m = *m && *u;
    }
    let design_rows = util::rows(mask);
    let subpop = c.opt("subpop", 3).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    if let Some(sp) = &subpop {
        let sm = subpop_mask(ds, st, sp)?;
        for (m, s) in mask.iter_mut().zip(&sm) {
            *m = *m && *s;
        }
    }
    Ok(Some(SvyInfo { set, design_rows, subpop }))
}

/// Design weights over the estimation rows.
pub fn weights(sv: Option<&SvyInfo>, ds: &DataSet, rows: &[usize], w: Vec<f64>, wtype: Option<String>) -> (Vec<f64>, Option<String>) {
    let Some(sv) = sv else { return (w, wtype) };
    match &sv.set.wvar {
        Some(v) => {
            let col = svy::key_col(ds, v);
            (rows.iter().map(|&r| col[r]).collect(), Some("pweight".into()))
        }
        None => (vec![1.0; rows.len()], None),
    }
}

/// Linearized variance from the observation scores.
pub fn variance(ds: &DataSet, sv: &SvyInfo, rows: &[usize], w: &[f64], scores: Option<&Vec<Vec<f64>>>, vinv: &Matrix) -> Result<Matrix, String> {
    let sc = scores.ok_or_else(|| rc_err(198, "svy requires an observation-level method (lf)"))?;
    let k = vinv.rows;
    let pos: std::collections::HashMap<usize, usize> = rows.iter().enumerate().map(|(t, &r)| (r, t)).collect();
    let z: Vec<Vec<f64>> = sv
        .design_rows
        .iter()
        .map(|r| match pos.get(r) {
            Some(&t) => (0..k).map(|a| w[t] * sc[t][a]).collect(),
            None => vec![0.0; k],
        })
        .collect();
    let dd = DesignData::new(ds, &sv.set, &sv.design_rows);
    let meat = dd.variance(&z).ok_or_else(|| rc_err(460, "missing standard errors because of stratum with single sampling unit"))?;
    vinv.mul(&meat)?.mul(vinv)
}

/// Survey results added to `e()`.
pub fn post(e: &mut Results, ds: &DataSet, sv: &SvyInfo, rows: &[usize], w: &[f64]) {
    let dd = DesignData::new(ds, &sv.set, &sv.design_rows);
    let (h, u) = dd.counts();
    let df = dd.df();
    e.set("N", sv.design_rows.len() as f64);
    e.set("N_strata", h as f64);
    e.set("N_psu", u as f64);
    e.set("df_r", df);
    let pop: f64 = dd.w.iter().sum();
    e.set("N_pop", pop);
    if sv.subpop.is_some() {
        e.set("N_sub", rows.len() as f64);
        e.set("N_subpop", w.iter().sum());
        e.set_str("subpop", sv.subpop.clone().unwrap_or_default());
    }
    e.set_str("prefix", "svy");
    e.set_str("vce", "linearized");
    e.set_str("vcetype", "Linearized");
    if let Some(wv) = &sv.set.wvar {
        e.set_str("wtype", "pweight");
        e.set_str("wvar", wv.clone());
        e.set_str("wexp", format!("= {wv}"));
    }
    if let Some(s) = &sv.set.stages[0].strata {
        e.set_str("strata1", s.clone());
    }
    if let Some(s) = &sv.set.stages[0].su {
        e.set_str("su1", s.clone());
    }
    // the design-based Wald test uses an adjusted F
    if let (Some(chi2), Some(dfm)) = (e.get("chi2"), e.get("df_m")) {
        if dfm > 0.0 && df >= dfm {
            let f = chi2 * (df - dfm + 1.0) / (df * dfm);
            e.set("F", f);
            e.set("df_m", dfm);
            e.set("p", ferrostat_cmd::dist::ftail(dfm, df - dfm + 1.0, f));
        }
    }
}
