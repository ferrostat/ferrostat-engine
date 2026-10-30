//! Heterogeneous difference in differences.
//!
//! `hdidregress` (repeated cross sections) and `xthdidregress` (panel data)
//!
//!   [xt]hdidregress aipw|ra|ipw|twfe (ovar [omvarlist]) (tvar [tmvarlist]) [if] [in] [weight],
//!       group(groupvar) [time(timevar) controlgroup(never|notyet) vce(cluster clustvar)]
//!
//! estimate cohort-time average treatment effects on the treated: the
//! cohort of a group is the first period in which `tvar` is 1 (0: never
//! treated). `aipw`, `ra` and `ipw` compare each cohort with the never- (or
//! not-yet-) treated groups between the period before treatment and each
//! later period (doubly robust, regression adjustment, stabilised inverse
//! probability weighting; computed by the `csdid` machinery), clustering by
//! group. `twfe` fits the extended two-way fixed-effects regression with
//! one indicator per treated cohort-period cell (cohort and period effects,
//! covariates entering linearly). `estat aggregation` averages the cells
//! (overall, by cohort, by period, by exposure time); `estat ptrends` tests
//! the pre-treatment cells; `estat atetplot` plots them.
//!
//! `xtswitchdid (ovar) (tvar)` handles treatments that switch on and off:
//! for each pair of consecutive periods it compares groups whose treatment
//! switched on with groups untreated in both periods, and groups whose
//! treatment switched off with groups treated in both, and averages the
//! effects weighted by the number of switchers (de Chaisemartin and
//! D'Haultfoeuille 2020). Standard errors by bootstrap over groups.
//!
//! `gencohort newvar = tvar [if] [in] [, group(groupvar) time(timevar) never(#)]`
//! creates the cohort variable.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::util::{table_matrix, ztable, zrow, ZRow};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::xmodels::{self as xm, Handlers};

pub fn handlers() -> Handlers {
    vec![
        ("hdidregress", Box::new(Hdid { panel: false })),
        ("xthdidregress", Box::new(Hdid { panel: true })),
        ("xtswitchdid", Box::new(Switch)),
        ("gencohort", Box::new(Gencohort)),
    ]
}

fn opt_var(ds: &DataSet, cmd: &Command, name: &str, min: usize) -> Result<Option<String>, String> {
    match cmd.opt(name, min) {
        Some(v) if !v.trim().is_empty() => Ok(Some(crate::varlist::resolve_name(ds, v.trim())?)),
        _ => Ok(None),
    }
}

/// (group variable, time variable) from options or the panel settings.
fn group_time(ds: &DataSet, st: &Store, cmd: &Command, panel: bool) -> Result<(String, String), String> {
    let g = opt_var(ds, cmd, "group", 2)?;
    let t = opt_var(ds, cmd, "time", 2)?;
    let (pv, tv) = match &st.tsset {
        Some((p, t)) => (p.clone(), Some(t.clone())),
        None => (None, None),
    };
    let g = match g {
        Some(g) => g,
        None if panel => pv.ok_or_else(|| rc_err(459, "panel variable not set; use xtset panelvar timevar"))?,
        None => return Err(rc_err(198, "option group() required")),
    };
    let t = match t.or(tv) {
        Some(t) => t,
        None => return Err(rc_err(198, "option time() required")),
    };
    Ok((g, t))
}

/// First period with `tvar` = 1 in each group (0 if never), over `mask`.
fn cohorts(ds: &DataSet, mask: &[bool], g: &str, t: &str, d: &str, never: f64) -> Result<Vec<f64>, String> {
    let gc = util::col(ds, g)?;
    let tc = util::col(ds, t)?;
    let dc = util::col(ds, d)?;
    let mut first: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for i in 0..ds.n_obs {
        if !mask[i] || is_missing(gc[i]) || is_missing(tc[i]) || is_missing(dc[i]) {
            continue;
        }
        let e = first.entry(gc[i].to_bits()).or_insert(f64::INFINITY);
        if dc[i] != 0.0 && tc[i] < *e {
            *e = tc[i];
        }
    }
    Ok((0..ds.n_obs)
        .map(|i| {
            if !mask[i] || is_missing(gc[i]) {
                return SYSMISS;
            }
            match first.get(&gc[i].to_bits()) {
                Some(v) if v.is_finite() => *v,
                Some(_) => never,
                None => SYSMISS,
            }
        })
        .collect())
}

struct Gencohort;

impl CommandHandler for Gencohort {
    fn name(&self) -> &'static str {
        "gencohort"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (newv, tvar) = match main.split_once('=') {
            Some((a, b)) => (a.trim().to_string(), xm::var_name(ds, b.trim())?),
            None => {
                let nv = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "syntax is gencohort newvar = tvar"))?.trim().to_string();
                (nv, xm::var_name(ds, main)?)
            }
        };
        crate::varlist::check_name(&newv)?;
        if ds.var_index.contains_key(&newv) {
            return Err(rc_err(110, format!("variable {newv} already defined")));
        }
        let (g, t) = group_time(ds, st, cmd, true)?;
        let never = util::opt_num(cmd, "never", 3)?.unwrap_or(0.0);
        let mask = util::touse(ds, st, cmd)?;
        let c = cohorts(ds, &mask, &g, &t, &tvar, never)?;
        let nmiss = c.iter().filter(|v| is_missing(**v)).count();
        let mut lv: Vec<f64> = c.iter().filter(|v| !is_missing(**v)).cloned().collect();
        lv.sort_by(f64::total_cmp);
        lv.dedup();
        ds.add_variable(util::new_double(&newv, c)).map_err(|e| rc_err(110, e))?;
        let mut out = vec![format!("(cohort variable {newv} created: {} cohort{})", lv.len(), if lv.len() == 1 { "" } else { "s" })];
        if nmiss > 0 {
            out.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
        }
        Ok(out.join("\n"))
    }
}

struct Hdid {
    panel: bool,
}

/// Parse `method (ovar omvars) (tvar tmvars)`.
fn parse_spec(ds: &DataSet, main: &str) -> Result<(String, String, Vec<String>, String, Vec<String>), String> {
    let (method, rest) = xm::first_word(main);
    let method = method.to_lowercase();
    if !["aipw", "ra", "ipw", "twfe"].contains(&method.as_str()) {
        return Err(rc_err(198, format!("estimator {method} not allowed; use aipw, ra, ipw or twfe")));
    }
    let gr = xm::paren_groups(&rest);
    if gr.len() != 2 {
        return Err(rc_err(198, "syntax is: estimator (ovar [omvarlist]) (tvar [tmvarlist])"));
    }
    let (ow, _) = crate::teffects::model(&gr[0]);
    let (tw, _) = crate::teffects::model(&gr[1]);
    if ow.is_empty() || tw.is_empty() {
        return Err(rc_err(198, "invalid syntax"));
    }
    let ov = xm::var_name(ds, &ow[0])?;
    let tv = xm::var_name(ds, &tw[0])?;
    let ox = crate::varlist::expand(ds, &ow[1..].join(" "))?;
    let tx = crate::varlist::expand(ds, &tw[1..].join(" "))?;
    Ok((method, ov, ox, tv, tx))
}

fn tmpname(ds: &DataSet, stem: &str) -> String {
    let mut k = 0;
    loop {
        let n = format!("__{stem}{k}");
        if !ds.var_index.contains_key(&n) {
            return n;
        }
        k += 1;
    }
}

impl CommandHandler for Hdid {
    fn name(&self) -> &'static str {
        if self.panel {
            "xthdidregress"
        } else {
            "hdidregress"
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
        let (method, ov, ox, tv, tx) = parse_spec(ds, cmd.main.trim())?;
        let (g, t) = group_time(ds, st, cmd, self.panel)?;
        let mut covars: Vec<String> = ox.clone();
        for v in &tx {
            if !covars.contains(v) {
                covars.push(v.clone());
            }
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = vec![ov.clone(), tv.clone(), g.clone(), t.clone()];
        need.extend(covars.iter().cloned());
        util::markout(ds, &mut mask, &need);
        let coh = cohorts(ds, &mask, &g, &t, &tv, 0.0)?;
        let notyet = cmd.opt("controlgroup", 4).map(|s| s.trim().to_lowercase().starts_with("notyet")).unwrap_or(false);
        let level = util::level(cmd, st)?;
        let ivar = if self.panel { st.tsset.as_ref().and_then(|(p, _)| p.clone()).or(Some(g.clone())) } else { None };
        let clust = match crate::est::parse_vce(cmd, crate::est::Vce::Ols)? {
            crate::est::Vce::Cluster(c) => c,
            _ => g.clone(),
        };
        let header = |out: &mut Vec<String>, nobs: f64, ng: usize| {
            let title = "Heterogeneous-treatment-effects regression";
            out.push(format!("{title}{:>w$}", format!("Number of obs     = {:>10}", util::commas(nobs as usize)), w = 78 - title.len()));
            out.push(format!("Estimator:      {}{:>w$}", match method.as_str() {
                "aipw" => "Augmented IPW",
                "ra" => "Regression adjustment",
                "ipw" => "Inverse-probability weighting",
                _ => "Two-way fixed effects",
            }, format!("Number of groups  = {:>10}", util::commas(ng)), w = 78 - 16 - match method.as_str() {
                "aipw" => 13,
                "ra" => 21,
                "ipw" => 29,
                _ => 21,
            }));
            out.push(format!("Panel variable: {}", if self.panel { ivar.clone().unwrap_or_default() } else { "(none)".into() }));
            out.push(format!("Treatment level: {g}"));
            out.push(format!("Control group: {}", if notyet { "Not yet treated" } else { "Never treated" }));
            out.push(String::new());
        };
        let gcol = util::col(ds, &g)?;
        let ng = {
            let mut v: Vec<u64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| gcol[i].to_bits()).collect();
            v.sort();
            v.dedup();
            v.len()
        };
        let mut out = Vec::new();
        if method == "twfe" {
            let res = twfe(ds, &mask, &ov, &covars, &t, &coh, &clust)?;
            let nobs = res.nobs;
            header(&mut out, nobs as f64, ng);
            let names: Vec<String> = res.cells.iter().map(|(cg, _, ct)| format!("{}.{}#{}.{}", fmt_g(*cg, 9), "cohort", fmt_g(*ct, 9), t)).collect();
            let se: Vec<f64> = (0..res.b.len()).map(|j| res.v[(j, j)].max(0.0).sqrt()).collect();
            let rows: Vec<ZRow> = res.cells.iter().enumerate().map(|(j, (cg, _, ct))| zrow(Some(&format!("Cohort {}", fmt_g(*cg, 9))), &fmt_g(*ct, 9), res.b[j], se[j])).collect();
            out.push(format!("ATET{:>w$}", format!("(Std. err. adjusted for {} clusters in {clust})", util::commas(res.nclust)), w = 74));
            out.extend(ztable(&t, "Robust", &rows, level));
            let text = out.join("\n");
            post_cells(st, verb, &method, &ov, &tv, &g, &t, &names, &res.cells, &res.b, &res.p, &res.v, nobs as f64, &text, level, &mask)?;
            st.e.set("N_clust", res.nclust as f64);
            st.e.set_str("clustvar", clust);
            return Ok(text);
        }
        // aipw / ra / ipw through the csdid machinery
        let cname = tmpname(ds, "hdcoh");
        let tname = tmpname(ds, "hdtu");
        ds.add_variable(util::new_double(&cname, coh.clone())).map_err(|e| rc_err(110, e))?;
        ds.add_variable(util::new_double(&tname, mask.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect())).map_err(|e| rc_err(110, e))?;
        let m = match method.as_str() {
            "aipw" => {
                if covars.is_empty() {
                    "reg"
                } else {
                    "dripw"
                }
            }
            "ra" => "reg",
            _ => "stdipw",
        };
        let wt = match &cmd.weight {
            Some(_) => {
                let raw = cmd.raw.clone();
                raw.find('[').and_then(|a| raw[a..].find(']').map(|b| raw[a..a + b + 1].to_string())).unwrap_or_default()
            }
            None => String::new(),
        };
        let line = format!(
            "csdid {ov} {} if {tname} == 1 {wt}, time({t}) gvar({cname}) method({m}){}{} cluster({clust})",
            covars.join(" "),
            ivar.as_ref().map(|i| format!(" ivar({i})")).unwrap_or_default(),
            if notyet { " notyet" } else { "" },
        );
        let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message));
        let r = c.and_then(|c| crate::did::Csdid.execute(ds, st, &c));
        let _ = ds.drop_variable(&cname);
        let _ = ds.drop_variable(&tname);
        let inner = r?;
        let nobs = st.e.get("N").unwrap_or(SYSMISS);
        header(&mut out, nobs, ng);
        // keep the coefficient table of the cohort-time effects
        let lines: Vec<&str> = inner.lines().collect();
        let start = lines.iter().position(|l| l.starts_with("----")).unwrap_or(0);
        out.push("ATET".to_string());
        out.extend(lines[start..].iter().map(|s| s.to_string()));
        let text = out.join("\n").replace(&cname, "cohort");
        let e = &mut st.e;
        e.set_str("cmd", verb);
        e.set_str("estimator", method.clone());
        e.set_str("depvar", ov.clone());
        e.set_str("treatment", tv.clone());
        e.set_str("groupvar", g.clone());
        e.set_str("timevar", t.clone());
        e.set_str("control_group", if notyet { "notyet" } else { "never" });
        e.set_str("title", "Heterogeneous-treatment-effects regression");
        e.set_str("_output", text.clone());
        e.set_str("ggroup", "cohort");
        e.sample = Some(mask);
        Ok(text)
    }
}

struct TwfeRes {
    b: Vec<f64>,
    /// share of observations in each cell (aggregation weights)
    p: Vec<f64>,
    v: Matrix,
    cells: Vec<(f64, f64, f64)>,
    nobs: usize,
    nclust: usize,
}

/// Extended two-way fixed effects: y on cohort and period indicators,
/// covariates and one indicator per treated cohort-period cell; cluster-
/// robust variance.
fn twfe(ds: &DataSet, mask: &[bool], ov: &str, covars: &[String], t: &str, coh: &[f64], clust: &str) -> Result<TwfeRes, String> {
    let y = util::col(ds, ov)?;
    let tc = util::col(ds, t)?;
    let cc = util::col(ds, clust)?;
    let xs: Vec<Vec<f64>> = covars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(coh[i]) && !is_missing(cc[i])).collect();
    let mut gl: Vec<f64> = rows.iter().map(|&i| coh[i]).collect();
    gl.sort_by(f64::total_cmp);
    gl.dedup();
    let mut tl: Vec<f64> = rows.iter().map(|&i| tc[i]).collect();
    tl.sort_by(f64::total_cmp);
    tl.dedup();
    let mut cells: Vec<(f64, f64, f64)> = Vec::new();
    for &g in gl.iter().filter(|g| **g != 0.0) {
        for &tt in &tl {
            if tt >= g && rows.iter().any(|&i| coh[i] == g && tc[i] == tt) {
                cells.push((g, g - 1.0, tt));
            }
        }
    }
    if cells.is_empty() {
        return Err(rc_err(2000, "no treated cohort-period cells"));
    }
    // columns: cells, covariates, cohort dummies (all but first), period dummies (all but first), constant
    let nc = cells.len();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for &(g, _, tt) in &cells {
        cols.push(rows.iter().map(|&i| if coh[i] == g && tc[i] == tt { 1.0 } else { 0.0 }).collect());
    }
    for x in &xs {
        cols.push(rows.iter().map(|&i| x[i]).collect());
    }
    for &g in gl.iter().skip(1) {
        cols.push(rows.iter().map(|&i| if coh[i] == g { 1.0 } else { 0.0 }).collect());
    }
    for &tt in tl.iter().skip(1) {
        cols.push(rows.iter().map(|&i| if tc[i] == tt { 1.0 } else { 0.0 }).collect());
    }
    cols.push(vec![1.0; rows.len()]);
    let yv: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let w = vec![1.0; rows.len()];
    let (b, xtxi) = crate::did_core::wls(&cols, &yv, &w)?;
    let n = rows.len();
    let k = cols.len();
    let resid: Vec<f64> = (0..n).map(|i| yv[i] - (0..k).map(|j| cols[j][i] * b[j]).sum::<f64>()).collect();
    // cluster sums of scores
    let mut keys: Vec<f64> = rows.iter().map(|&i| cc[i]).collect();
    keys.sort_by(f64::total_cmp);
    keys.dedup();
    let g = keys.len();
    let mut sc = vec![vec![0.0; k]; g];
    for (a, &i) in rows.iter().enumerate() {
        let gi = keys.binary_search_by(|v| v.total_cmp(&cc[i])).unwrap();
        for j in 0..k {
            sc[gi][j] += cols[j][a] * resid[a];
        }
    }
    let rank = (0..k).filter(|&j| xtxi[(j, j)] != 0.0).count() as f64;
    let adj = (g as f64 / (g as f64 - 1.0).max(1.0)) * ((n as f64 - 1.0) / (n as f64 - rank).max(1.0));
    let mut meat = Matrix::zeros(k, k);
    for s in &sc {
        for a in 0..k {
            for c in 0..k {
                meat[(a, c)] += s[a] * s[c];
            }
        }
    }
    let mut v = Matrix::zeros(nc, nc);
    for a in 0..nc {
        for c in 0..nc {
            let mut s = 0.0;
            for p in 0..k {
                for q in 0..k {
                    s += xtxi[(a, p)] * meat[(p, q)] * xtxi[(q, c)];
                }
            }
            v[(a, c)] = s * adj;
        }
    }
    let p: Vec<f64> = (0..nc).map(|j| cols[j].iter().sum::<f64>() / n as f64).collect();
    Ok(TwfeRes { b: b[..nc].to_vec(), p, v, cells, nobs: n, nclust: g })
}

#[allow(clippy::too_many_arguments)]
fn post_cells(st: &mut Store, verb: &str, method: &str, ov: &str, tv: &str, g: &str, t: &str, names: &[String], cells: &[(f64, f64, f64)], b: &[f64], p: &[f64], v: &Matrix, nobs: f64, text: &str, level: f64, mask: &[bool]) -> Result<(), String> {
    let se: Vec<f64> = (0..b.len()).map(|j| v[(j, j)].max(0.0).sqrt()).collect();
    let tm = table_matrix(names, b, &se, level);
    let mut cm = Matrix::zeros(cells.len(), 3);
    for (r, c) in cells.iter().enumerate() {
        cm[(r, 0)] = c.0;
        cm[(r, 1)] = c.1;
        cm[(r, 2)] = c.2;
    }
    let cellnames: Vec<String> = cells.iter().map(|c| format!("g{}:t_{}_{}", fmt_g(c.0, 9), fmt_g(c.1, 9), fmt_g(c.2, 9))).collect();
    let e = st.begin_e();
    e.set("N", nobs);
    e.set_str("cmd", verb);
    e.set_str("estimator", method);
    e.set_str("depvar", ov);
    e.set_str("treatment", tv);
    e.set_str("groupvar", g);
    e.set_str("timevar", t);
    e.set_str("properties", "b V");
    e.set_str("title", "Heterogeneous-treatment-effects regression");
    e.set_str("_output", text);
    e.set_mat("b", Matrix::row_vector(b).with_names(vec!["y1".into()], names.to_vec()));
    e.set_mat("V", v.clone().with_names(names.to_vec(), names.to_vec()));
    // theta then cell weights, as the cohort-time aggregation expects
    let k = b.len();
    let mut bt = b.to_vec();
    bt.extend_from_slice(p);
    let mut vt = Matrix::zeros(2 * k, 2 * k);
    for a in 0..k {
        for c in 0..k {
            vt[(a, c)] = v[(a, c)];
        }
    }
    let mut tn = cellnames.clone();
    tn.extend(cellnames.iter().map(|c| format!("w_{c}")));
    e.set_mat("b_attgt", Matrix::row_vector(&bt).with_names(vec!["y1".into()], tn.clone()));
    e.set_mat("V_attgt", vt.with_names(tn.clone(), tn));
    e.set_mat("gtt_cells", cm.with_names(cellnames, vec!["cohort".into(), "t0".into(), "t1".into()]));
    e.sample = Some(mask.to_vec());
    crate::est::stash_table(tm, level);
    Ok(())
}

/// `estat aggregation|ptrends|atetplot` after hdidregress/xthdidregress:
/// computed by the cohort-time aggregation of `csdid`.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd == "xtswitchdid" {
        return None;
    }
    if ecmd != "hdidregress" && ecmd != "xthdidregress" {
        return None;
    }
    let kind = if sub.len() >= 3 && "aggregation".starts_with(sub) {
        if cmd.has_opt("cohort", 3) {
            "group"
        } else if cmd.has_opt("time", 2) {
            "calendar"
        } else if cmd.has_opt("dynamic", 3) {
            "event"
        } else {
            "simple"
        }
    } else if sub.len() >= 3 && "ptrends".starts_with(sub) {
        "pretrend"
    } else if sub == "atetplot" {
        "plot"
    } else {
        return None;
    };
    st.e.set_str("cmd", "csdid");
    let r = if kind == "plot" {
        let line = format!("csdid_plot{}", if cmd.has_opt("cohort", 3) { ", agg(group)" } else { "" });
        ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message)).and_then(|c| crate::did::CsdidPlot.execute(ds, st, &c))
    } else if kind == "pretrend" && !st.e.matrices.get("gtt_cells").is_some_and(|m| (0..m.rows).any(|r| m[(r, 2)] < m[(r, 0)])) {
        Err(rc_err(321, "no pre-treatment cells; estat ptrends requires the aipw, ra or ipw estimators"))
    } else {
        crate::did::estat(ds, st, kind, cmd).unwrap_or_else(|| Err(rc_err(321, "not available")))
    };
    if st.e.macros.get("cmd").map(String::as_str) == Some("csdid") {
        st.e.set_str("cmd", ecmd);
    }
    Some(r.map(|s| s.replace("ATT by group", "ATET by cohort").replace("ATT by Calendar Period", "ATET by time").replace("Average Treatment Effect on Treated", "Overall ATET").replace("ATT by Periods Before and After treatment\nEvent Study:Dynamic effects", "ATET by exposure time")))
}

// ---------------------------------------------------------------------------
// xtswitchdid
// ---------------------------------------------------------------------------

struct Switch;

/// DID_M and its two components from unit-period data:
/// returns (did_m, did_plus, did_minus, n_joiners, n_leavers).
fn did_m(units: &[Vec<(f64, f64, f64)>], periods: &[f64]) -> (f64, f64, f64, f64, f64) {
    let (mut sp, mut np, mut sm, mut nm) = (0.0, 0.0, 0.0, 0.0);
    for w in periods.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let mut grp: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for u in units {
            let a = u.iter().find(|r| r.0 == t0);
            let b = u.iter().find(|r| r.0 == t1);
            if let (Some(a), Some(b)) = (a, b) {
                let k = (a.2 != 0.0) as usize * 2 + (b.2 != 0.0) as usize;
                grp[k].push(b.1 - a.1);
            }
        }
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        // joiners (0 -> 1) vs stayers untreated (0 -> 0)
        if !grp[1].is_empty() && !grp[0].is_empty() {
            let n = grp[1].len() as f64;
            sp += n * (mean(&grp[1]) - mean(&grp[0]));
            np += n;
        }
        // leavers (1 -> 0) vs stayers treated (1 -> 1)
        if !grp[2].is_empty() && !grp[3].is_empty() {
            let n = grp[2].len() as f64;
            sm += n * (mean(&grp[3]) - mean(&grp[2]));
            nm += n;
        }
    }
    let tot = np + nm;
    let dm = if tot > 0.0 { (sp + sm) / tot } else { SYSMISS };
    (dm, if np > 0.0 { sp / np } else { SYSMISS }, if nm > 0.0 { sm / nm } else { SYSMISS }, np, nm)
}

impl CommandHandler for Switch {
    fn name(&self) -> &'static str {
        "xtswitchdid"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("xtswitchdid") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let gr = xm::paren_groups(&cmd.main);
        let (ov, tv) = if gr.len() >= 2 {
            (xm::var_name(ds, gr[0].split_whitespace().next().unwrap_or(""))?, xm::var_name(ds, gr[1].split_whitespace().next().unwrap_or(""))?)
        } else {
            let w: Vec<&str> = cmd.main.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "syntax is: xtswitchdid (ovar) (tvar)"));
            }
            (xm::var_name(ds, w[0])?, xm::var_name(ds, w[1])?)
        };
        let (g, t) = group_time(ds, st, cmd, true)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[ov.clone(), tv.clone(), g.clone(), t.clone()]);
        let (y, d, gc, tc) = (util::col(ds, &ov)?, util::col(ds, &tv)?, util::col(ds, &g)?, util::col(ds, &t)?);
        let mut keys: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| gc[i]).collect();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let mut units: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); keys.len()];
        for i in 0..ds.n_obs {
            if mask[i] {
                let k = keys.binary_search_by(|v| v.total_cmp(&gc[i])).unwrap();
                units[k].push((tc[i], y[i], d[i]));
            }
        }
        let mut periods: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| tc[i]).collect();
        periods.sort_by(f64::total_cmp);
        periods.dedup();
        let (dm, dp, dmn, np, nm) = did_m(&units, &periods);
        if is_missing(dm) {
            return Err(rc_err(2000, "no switching groups with valid comparison groups"));
        }
        // bootstrap over groups
        if let Some(s) = util::opt_num(cmd, "rseed", 2)? {
            st.rng.borrow_mut().reseed(s as u64);
        }
        let reps = util::opt_num(cmd, "reps", 3)?.unwrap_or(100.0) as usize;
        let ng = units.len();
        let mut draws: Vec<[f64; 3]> = Vec::new();
        for _ in 0..reps {
            let sample: Vec<Vec<(f64, f64, f64)>> = {
                let mut rng = st.rng.borrow_mut();
                (0..ng).map(|_| units[((rng.uniform() * ng as f64) as usize).min(ng - 1)].clone()).collect()
            };
            let (a, b, c, _, _) = did_m(&sample, &periods);
            draws.push([a, b, c]);
        }
        let sd = |k: usize| -> f64 {
            let v: Vec<f64> = draws.iter().map(|d| d[k]).filter(|x| !is_missing(*x)).collect();
            if v.len() < 2 {
                return SYSMISS;
            }
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let level = util::level(cmd, st)?;
        let est = [dm, dp, dmn];
        let se = [sd(0), sd(1), sd(2)];
        let labels = ["DID_M", "Joiners", "Leavers"];
        let rows: Vec<ZRow> = (0..3).filter(|&k| !is_missing(est[k])).map(|k| zrow(Some("ATET"), labels[k], est[k], se[k])).collect();
        let nobs = mask.iter().filter(|m| **m).count();
        let mut out = vec![
            "Difference in differences with switching treatment".to_string(),
            format!("{:<48}{:<17} = {:>10}", "", "Number of obs", util::commas(nobs)),
            format!("{:<48}{:<17} = {:>10}", format!("Group variable: {g}"), "Number of groups", util::commas(ng)),
            format!("{:<48}{:<17} = {:>10}", format!("Time variable: {t}"), "Switches on", util::commas(np as usize)),
            format!("{:<48}{:<17} = {:>10}", format!("Bootstrap replications: {reps}"), "Switches off", util::commas(nm as usize)),
            String::new(),
        ];
        out.extend(ztable("", "Bootstrap", &rows, level));
        let text = out.join("\n");
        let names: Vec<String> = (0..3).filter(|&k| !is_missing(est[k])).map(|k| format!("ATET:{}", labels[k])).collect();
        let b: Vec<f64> = (0..3).filter(|&k| !is_missing(est[k])).map(|k| est[k]).collect();
        let s2: Vec<f64> = (0..3).filter(|&k| !is_missing(est[k])).map(|k| se[k]).collect();
        let mut v = Matrix::zeros(b.len(), b.len());
        for j in 0..b.len() {
            v[(j, j)] = s2[j] * s2[j];
        }
        let tm = table_matrix(&names, &b, &s2, level);
        let e = st.begin_e();
        e.set("N", nobs as f64);
        e.set("N_g", ng as f64);
        e.set("N_switch_on", np);
        e.set("N_switch_off", nm);
        e.set("reps", reps as f64);
        e.set_str("cmd", "xtswitchdid");
        e.set_str("depvar", ov);
        e.set_str("treatment", tv);
        e.set_str("vcetype", "Bootstrap");
        e.set_str("properties", "b V");
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.sample = Some(mask);
        crate::est::stash_table(tm, level);
        Ok(text)
    }
}
