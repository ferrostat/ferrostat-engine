//! Meta-analysis: declaring study effect sizes (`meta set`, `meta esize`,
//! `meta psycorr`, `meta update`, `meta query`/`meta describe`, `meta
//! clear`), pooling (`meta summarize`), graphs (`meta forestplot`,
//! `funnelplot`, `labbeplot`, `galbraithplot`), small-study effects
//! (`meta bias`, `meta trimfill`) and meta-regression (`meta regress`,
//! `meta mvregress`, `meta meregress`, `meta multilevel`).
//!
//! Declarations are kept in system variables `_meta_id`, `_meta_es`,
//! `_meta_se`, `_meta_cil`, `_meta_ciu`, `_meta_studylabel` (and
//! `_meta_studysize` when known) and in `_dta[_meta_*]` characteristics.

pub mod bias;
pub mod plots;
pub mod pool;
pub mod reg;
pub mod summ;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::util;
use pool::Model;

pub const VARS: [&str; 7] = ["_meta_id", "_meta_es", "_meta_se", "_meta_cil", "_meta_ciu", "_meta_studylabel", "_meta_studysize"];

/// Declared meta-analysis settings.
#[derive(Clone, Debug)]
pub struct Settings {
    /// generic, continuous, binary or correlation
    pub datatype: String,
    /// effect-size key: generic, mdiff, hedgesg, cohend, glassdelta1/2,
    /// lnor, lnrr, rdiff, lnorpeto, fisherz, corr
    pub es: String,
    pub eslabel: String,
    pub model: Model,
    pub method: String,
    pub level: f64,
    pub datavars: Vec<String>,
    pub zc: String,
    pub studylabel: String,
    pub studysize: String,
    pub esvar: String,
    pub sevar: String,
    pub ifin: String,
    pub cmdline: String,
}

fn chr(ds: &DataSet, k: &str) -> Option<String> {
    ds.characteristics.get("_dta").and_then(|m| m.get(&format!("_meta_{k}"))).cloned()
}

pub fn settings(ds: &DataSet) -> Result<Settings, String> {
    if chr(ds, "marker").is_none() || !ds.var_index.contains_key("_meta_es") {
        return Err(rc_err(119, "data not meta set; use meta set or meta esize to declare your data"));
    }
    let g = |k: &str| chr(ds, k).unwrap_or_default();
    Ok(Settings {
        datatype: g("datatype"),
        es: g("estype"),
        eslabel: g("eslabel"),
        model: Model::parse(&g("model")).unwrap_or(Model::Random),
        method: g("method"),
        level: g("level").parse().unwrap_or(95.0),
        datavars: g("datavars").split_whitespace().map(String::from).collect(),
        zc: g("zcopt"),
        studylabel: g("studylabel"),
        studysize: g("studysize"),
        esvar: g("esvar"),
        sevar: g("sevar"),
        ifin: g("ifin"),
        cmdline: g("cmdline"),
    })
}

fn save_settings(ds: &mut DataSet, s: &Settings) {
    let m = ds.characteristics.entry("_dta".into()).or_default();
    m.retain(|k, _| !k.starts_with("_meta_"));
    let mut put = |k: &str, v: String| {
        m.insert(format!("_meta_{k}"), v);
    };
    put("marker", "_meta_ds_1".into());
    put("datatype", s.datatype.clone());
    put("estype", s.es.clone());
    put("eslabel", s.eslabel.clone());
    put("model", s.model.name().into());
    put("method", s.method.clone());
    put("level", fmt_g(s.level, 8));
    put("datavars", s.datavars.join(" "));
    put("zcopt", s.zc.clone());
    put("studylabel", s.studylabel.clone());
    put("studysize", s.studysize.clone());
    put("esvar", s.esvar.clone());
    put("sevar", s.sevar.clone());
    put("ifin", s.ifin.clone());
    put("cmdline", s.cmdline.clone());
}

pub fn es_label(es: &str) -> &'static str {
    match es {
        "mdiff" => "Mean diff.",
        "hedgesg" => "Hedges's g",
        "cohend" => "Cohen's d",
        "glassdelta1" | "glassdelta2" => "Glass's Delta",
        "lnor" => "Log odds-ratio",
        "lnorpeto" => "Peto's log odds-ratio",
        "lnrr" => "Log risk-ratio",
        "rdiff" => "Risk diff.",
        "fisherz" => "Fisher's z",
        "corr" => "Correlation",
        _ => "Effect size",
    }
}

/// Exponentiated label for `eform`.
pub fn eform_label(es: &str) -> &'static str {
    match es {
        "lnor" | "lnorpeto" => "Odds ratio",
        "lnrr" => "Risk ratio",
        _ => "exp(ES)",
    }
}

/// Model and method from `random[()]`, `common[()]`, `fixed[()]`.
pub fn model_opts(cmd: &Command, default: (Model, String), es: &str) -> Result<(Model, String), String> {
    let mut out = default;
    for (name, m) in [("random", Model::Random), ("common", Model::Common), ("fixed", Model::Fixed)] {
        if cmd.has_opt(name, 4) {
            let arg = cmd.opt(name, 4).unwrap_or("");
            let mut meth = pool::method_name(m, arg)?;
            if meth.is_empty() {
                meth = if matches!(es, "lnor" | "lnrr" | "rdiff") { "mh".into() } else { "iv".into() };
            }
            if meth == "mh" && !matches!(es, "lnor" | "lnrr" | "rdiff") {
                return Err(rc_err(198, "Mantel-Haenszel method is allowed only with log odds ratios, log risk ratios and risk differences"));
            }
            out = (m, meth);
        }
    }
    if es == "lnorpeto" && out.0 != Model::Random {
        out.1 = "iv".into();
    }
    Ok(out)
}

fn set_var(ds: &mut DataSet, name: &str, vals: Vec<f64>) -> Result<(), String> {
    if ds.var_index.contains_key(name) {
        ds.drop_variable(name).map_err(|e| rc_err(198, e))?;
    }
    ds.add_variable(util::new_double(name, vals)).map_err(|e| rc_err(110, e))
}

fn set_str_var(ds: &mut DataSet, name: &str, vals: Vec<String>) -> Result<(), String> {
    if ds.var_index.contains_key(name) {
        ds.drop_variable(name).map_err(|e| rc_err(198, e))?;
    }
    let w = vals.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let mut var = ferrostat_core::Variable::new(name, ferrostat_core::VarType::Str(w.min(2045) as _));
    var.data = vals.into_iter().map(ferrostat_core::Value::Str).collect();
    ds.add_variable(var).map_err(|e| rc_err(110, e))
}

/// Study labels from a variable (value labels used) or "Study #".
fn study_labels(ds: &DataSet, var: &str) -> Vec<String> {
    match ds.get_var(var) {
        Some(v) if !var.is_empty() => (0..ds.n_obs).map(|i| util::cell_text(ds, v, i, true).trim().to_string()).collect(),
        _ => (0..ds.n_obs).map(|i| format!("Study {}", i + 1)).collect(),
    }
}

/// Effect sizes and standard errors from summary data.
pub fn compute_es(ds: &DataSet, s: &Settings, mask: &[bool]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = ds.n_obs;
    let cols: Vec<Vec<f64>> = s.datavars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let mut es = vec![SYSMISS; n];
    let mut se = vec![SYSMISS; n];
    let (zc_val, zc_mode) = zero_cells(&s.zc)?;
    for i in 0..n {
        if !mask[i] || cols.iter().any(|c| is_missing(c[i])) {
            continue;
        }
        let (e, v) = match s.datatype.as_str() {
            "continuous" => {
                let (n1, m1, s1, n2, m2, s2) = (cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i]);
                if n1 < 2.0 || n2 < 2.0 || s1 <= 0.0 || s2 <= 0.0 {
                    return Err(rc_err(459, format!("observation {}: group sizes must be at least 2 and standard deviations positive", i + 1)));
                }
                let nn = n1 + n2;
                let sp = (((n1 - 1.0) * s1 * s1 + (n2 - 1.0) * s2 * s2) / (nn - 2.0)).sqrt();
                match s.es.as_str() {
                    "mdiff" => (m1 - m2, s1 * s1 / n1 + s2 * s2 / n2),
                    "cohend" => {
                        let d = (m1 - m2) / sp;
                        (d, nn / (n1 * n2) + d * d / (2.0 * nn))
                    }
                    "glassdelta1" | "glassdelta2" => {
                        let (sc, nc) = if s.es == "glassdelta1" { (s1, n1) } else { (s2, n2) };
                        let d = (m1 - m2) / sc;
                        (d, nn / (n1 * n2) + d * d / (2.0 * (nc - 1.0)))
                    }
                    _ => {
                        // Hedges's g: small-sample corrected d
                        let j = 1.0 - 3.0 / (4.0 * nn - 9.0);
                        let g = j * (m1 - m2) / sp;
                        (g, nn / (n1 * n2) + g * g / (2.0 * nn))
                    }
                }
            }
            "binary" => {
                let mut c = [cols[0][i], cols[1][i], cols[2][i], cols[3][i]];
                if c.iter().any(|x| *x < 0.0) {
                    return Err(rc_err(459, format!("observation {}: cell counts must be nonnegative", i + 1)));
                }
                let n1 = c[0] + c[1];
                let n2 = c[2] + c[3];
                if n1 <= 0.0 || n2 <= 0.0 {
                    return Err(rc_err(459, format!("observation {}: group sizes must be positive", i + 1)));
                }
                if s.es == "lnorpeto" {
                    let nn = n1 + n2;
                    let m1 = c[0] + c[2];
                    let e = n1 * m1 / nn;
                    let vv = n1 * n2 * m1 * (nn - m1) / (nn * nn * (nn - 1.0));
                    if vv <= 0.0 {
                        continue;
                    }
                    ((c[0] - e) / vv, 1.0 / vv)
                } else {
                    let has0 = c.iter().any(|x| *x == 0.0);
                    if s.es != "rdiff" && zc_mode != "none" && (has0 || zc_mode == "all") {
                        for x in c.iter_mut() {
                            *x += zc_val;
                        }
                    }
                    let (a, b, cc, d) = (c[0], c[1], c[2], c[3]);
                    let (n1, n2) = (a + b, cc + d);
                    match s.es.as_str() {
                        "lnrr" => ((a / n1 / (cc / n2)).ln(), 1.0 / a - 1.0 / n1 + 1.0 / cc - 1.0 / n2),
                        "rdiff" => {
                            let (p1, p2) = (a / n1, cc / n2);
                            (p1 - p2, p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
                        }
                        _ => ((a * d / (b * cc)).ln(), 1.0 / a + 1.0 / b + 1.0 / cc + 1.0 / d),
                    }
                }
            }
            "correlation" => {
                let (r, nn) = (cols[0][i], cols[1][i]);
                if r.abs() >= 1.0 || nn < 4.0 {
                    return Err(rc_err(459, format!("observation {}: correlations must be in (-1, 1) and sample sizes at least 4", i + 1)));
                }
                if s.es == "fisherz" {
                    (r.atanh(), 1.0 / (nn - 3.0))
                } else {
                    (r, (1.0 - r * r).powi(2) / (nn - 1.0))
                }
            }
            _ => continue,
        };
        if e.is_finite() && v.is_finite() && v > 0.0 {
            es[i] = e;
            se[i] = v.sqrt();
        }
    }
    Ok((es, se))
}

/// `zerocells(#[, only0|allif0])` or `zerocells(none)`: (value, mode).
fn zero_cells(spec: &str) -> Result<(f64, String), String> {
    let t = spec.trim().to_lowercase();
    if t.is_empty() {
        return Ok((0.5, "only0".into()));
    }
    if t == "none" {
        return Ok((0.0, "none".into()));
    }
    let (a, b) = t.split_once(',').map_or((t.as_str(), ""), |(a, b)| (a.trim(), b.trim()));
    let v: f64 = a.parse().map_err(|_| rc_err(198, "zerocells() must be a number or none"))?;
    let mode = match b {
        "" | "only0" => "only0",
        "allif0" => "all",
        _ => return Err(rc_err(198, "zerocells() suboption must be only0 or allif0")),
    };
    Ok((v, mode.into()))
}

/// Write the system variables for the declared data.
fn declare(ds: &mut DataSet, st: &Store, cmd: &Command, mut s: Settings, es: Vec<f64>, se: Vec<f64>, mask: &[bool]) -> Result<String, String> {
    let n = ds.n_obs;
    let crit = crate::dist::invnormal(1.0 - (1.0 - s.level / 100.0) / 2.0);
    let ok: Vec<bool> = (0..n).map(|i| mask[i] && !is_missing(es[i]) && !is_missing(se[i])).collect();
    let mut id = vec![SYSMISS; n];
    let mut k = 0;
    for i in 0..n {
        if ok[i] {
            k += 1;
            id[i] = k as f64;
        }
    }
    if k == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    let cil: Vec<f64> = (0..n).map(|i| if ok[i] { es[i] - crit * se[i] } else { SYSMISS }).collect();
    let ciu: Vec<f64> = (0..n).map(|i| if ok[i] { es[i] + crit * se[i] } else { SYSMISS }).collect();
    let labels = study_labels(ds, &s.studylabel);
    let size: Option<Vec<f64>> = if !s.studysize.is_empty() {
        Some(util::col(ds, &s.studysize)?)
    } else if s.datatype == "continuous" {
        let c0 = util::col(ds, &s.datavars[0])?;
        let c3 = util::col(ds, &s.datavars[3])?;
        Some((0..n).map(|i| c0[i] + c3[i]).collect())
    } else if s.datatype == "binary" {
        let cs: Vec<Vec<f64>> = s.datavars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        Some((0..n).map(|i| cs.iter().map(|c| c[i]).sum()).collect())
    } else if s.datatype == "correlation" {
        Some(util::col(ds, &s.datavars[1])?)
    } else {
        None
    };
    set_var(ds, "_meta_id", id)?;
    set_var(ds, "_meta_es", (0..n).map(|i| if ok[i] { es[i] } else { SYSMISS }).collect())?;
    set_var(ds, "_meta_se", (0..n).map(|i| if ok[i] { se[i] } else { SYSMISS }).collect())?;
    set_var(ds, "_meta_cil", cil)?;
    set_var(ds, "_meta_ciu", ciu)?;
    set_str_var(ds, "_meta_studylabel", labels)?;
    if let Some(z) = size {
        set_var(ds, "_meta_studysize", (0..n).map(|i| if ok[i] { z[i] } else { SYSMISS }).collect())?;
    } else if ds.var_index.contains_key("_meta_studysize") {
        let _ = ds.drop_variable("_meta_studysize");
    }
    if s.eslabel.is_empty() {
        s.eslabel = es_label(&s.es).into();
    }
    let _ = (st, cmd);
    save_settings(ds, &s);
    Ok(describe(ds, &s, k))
}

/// The declaration summary printed by meta set / esize / update / query.
fn describe(ds: &DataSet, s: &Settings, k: usize) -> String {
    let _ = ds;
    let mut out = vec![String::new(), "Meta-analysis setting information".to_string(), String::new()];
    out.push(" Study information".into());
    out.push(format!("    No. of studies: {k}"));
    out.push(format!("       Study label: {}", if s.studylabel.is_empty() { "Generic" } else { &s.studylabel }));
    out.push(format!("        Study size: {}", if s.studysize.is_empty() { if s.datatype == "generic" { "N/A".to_string() } else { "_meta_studysize".to_string() } } else { s.studysize.clone() }));
    if !s.datavars.is_empty() {
        out.push(format!("      Summary data: {}", s.datavars.join(" ")));
    }
    out.push(String::new());
    out.push("       Effect size".into());
    out.push(format!("              Type: {}", s.es));
    out.push(format!("             Label: {}", s.eslabel));
    out.push(format!("          Variable: {}", if s.esvar.is_empty() { "_meta_es" } else { &s.esvar }));
    if s.datatype == "binary" && !matches!(s.es.as_str(), "rdiff" | "lnorpeto") {
        let zc = if s.zc.is_empty() { "0.5, only0".to_string() } else { s.zc.clone() };
        out.push(format!("        Zero-cells: {zc}"));
    }
    out.push(String::new());
    out.push("         Precision".into());
    out.push(format!("         Std. err.: {}", if s.sevar.is_empty() { "_meta_se" } else { &s.sevar }));
    out.push(format!("                CI: [_meta_cil, _meta_ciu]"));
    out.push(format!("          CI level: {}%", fmt_g(s.level, 8)));
    out.push(String::new());
    out.push("  Model and method".into());
    out.push(format!("             Model: {}", s.model.title()));
    out.push(format!("            Method: {}", pool::method_label(&s.method)));
    out.join("\n")
}

fn ifin_text(cmd: &Command) -> String {
    let mut t = String::new();
    if let Some(i) = &cmd.if_raw {
        t.push_str(&format!("if {i}"));
    }
    t
}

fn level_opt(cmd: &Command, st: &Store, default: f64) -> Result<f64, String> {
    if cmd.has_opt("level", 1) { util::level(cmd, st) } else { Ok(default) }
}

/// `meta set`, `meta esize`, `meta psycorr`.
fn declare_cmd(ds: &mut DataSet, st: &mut Store, cmd: &Command, sub: &str, rest: &str) -> Result<String, String> {
    let vars = crate::varlist::expand(ds, rest).map_err(|e| rc_err(111, e))?;
    let mask = util::touse(ds, st, cmd)?;
    let studylabel = cmd.opt("studylabel", 6).map(|v| crate::xmodels::var_name(ds, v)).transpose()?.unwrap_or_default();
    let studysize = cmd.opt("studysize", 6).map(|v| crate::xmodels::var_name(ds, v)).transpose()?.unwrap_or_default();
    let eslabel = cmd.opt("eslabel", 3).map(util::unquote).unwrap_or_default();
    let level = level_opt(cmd, st, 95.0)?;
    let mut s = Settings {
        datatype: String::new(),
        es: String::new(),
        eslabel,
        model: Model::Random,
        method: "reml".into(),
        level,
        datavars: Vec::new(),
        zc: cmd.opt("zerocells", 4).unwrap_or("").trim().to_string(),
        studylabel,
        studysize,
        esvar: String::new(),
        sevar: String::new(),
        ifin: ifin_text(cmd),
        cmdline: format!("meta {}", cmd.raw.trim_start_matches("meta").trim()),
    };
    let (es, se) = match sub {
        "set" => {
            s.datatype = "generic".into();
            s.es = "generic".into();
            let c: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
            match vars.len() {
                2 => {
                    s.esvar = vars[0].clone();
                    s.sevar = vars[1].clone();
                    (c[0].clone(), c[1].iter().map(|v| if !is_missing(*v) && *v > 0.0 { *v } else { SYSMISS }).collect())
                }
                3 => {
                    let cl = util::opt_num(cmd, "civarlevel", 4)?.unwrap_or(95.0);
                    let z = crate::dist::invnormal(1.0 - (1.0 - cl / 100.0) / 2.0);
                    let tol = util::opt_num(cmd, "civartolerance", 6)?.unwrap_or(1e-6);
                    s.esvar = vars[0].clone();
                    let mut se = vec![SYSMISS; ds.n_obs];
                    for i in 0..ds.n_obs {
                        if mask[i] && [c[0][i], c[1][i], c[2][i]].iter().all(|v| !is_missing(*v)) {
                            let (lo, hi) = (c[1][i] - c[0][i], c[2][i] - c[0][i]);
                            if ((hi + lo) / (hi - lo)).abs() > tol.max(1e-3) && !cmd.has_opt("civartolerance", 6) {
                                return Err(rc_err(459, format!("observation {}: confidence interval is not symmetric about the effect size", i + 1)));
                            }
                            se[i] = (c[2][i] - c[1][i]) / (2.0 * z);
                        }
                    }
                    (c[0].clone(), se)
                }
                _ => return Err(rc_err(198, "meta set requires esvar sevar or esvar cilvar ciuvar")),
            }
        }
        "esize" => {
            let opt = cmd.opt("esize", 3).unwrap_or("").trim().to_lowercase();
            let (opt, sub_opt) = opt.split_once(',').map_or((opt.clone(), String::new()), |(a, b)| (a.trim().to_string(), b.trim().to_string()));
            match vars.len() {
                6 => {
                    s.datatype = "continuous".into();
                    s.es = match opt.as_str() {
                        "" | "hedgesg" => "hedgesg".into(),
                        "mdiff" => "mdiff".into(),
                        "cohend" => "cohend".into(),
                        "glassdelta" | "glassdelta2" => "glassdelta2".into(),
                        "glassdelta1" => "glassdelta1".into(),
                        o => return Err(rc_err(198, format!("esize({o}) not allowed with continuous data"))),
                    };
                    let _ = sub_opt;
                }
                4 => {
                    s.datatype = "binary".into();
                    s.es = match opt.as_str() {
                        "" | "lnor" => "lnor".into(),
                        "lnrr" => "lnrr".into(),
                        "rdiff" => "rdiff".into(),
                        "lnorpeto" => "lnorpeto".into(),
                        o => return Err(rc_err(198, format!("esize({o}) not allowed with binary data"))),
                    };
                }
                _ => return Err(rc_err(198, "meta esize requires 6 variables (n1 mean1 sd1 n2 mean2 sd2) or 4 variables (n11 n12 n21 n22)")),
            }
            s.datavars = vars.clone();
            compute_es(ds, &s, &mask)?
        }
        _ => {
            // psycorr r n
            if vars.len() != 2 {
                return Err(rc_err(198, "meta psycorr requires rvar nvar"));
            }
            s.datatype = "correlation".into();
            s.es = if cmd.has_opt("fisherz", 3) { "fisherz".into() } else { "corr".into() };
            s.datavars = vars.clone();
            let (mut es, mut se) = compute_es(ds, &s, &mask)?;
            // correction for attenuation by the reliabilities
            let rel = |name: &str| -> Result<Option<Vec<f64>>, String> { cmd.opt(name, 3).map(|v| util::col(ds, &crate::xmodels::var_name(ds, v)?)).transpose() };
            let (rx, ry) = (rel("rxx")?, rel("ryy")?);
            if rx.is_some() || ry.is_some() {
                for i in 0..ds.n_obs {
                    if is_missing(es[i]) {
                        continue;
                    }
                    let a = rx.as_ref().map_or(1.0, |c| c[i]) * ry.as_ref().map_or(1.0, |c| c[i]);
                    if !(a > 0.0 && a <= 1.0) {
                        return Err(rc_err(459, format!("observation {}: reliabilities must be in (0, 1]", i + 1)));
                    }
                    if s.es == "corr" {
                        es[i] /= a.sqrt();
                        se[i] /= a.sqrt();
                    }
                }
            }
            if s.es == "corr" && !cmd.has_opt("random", 4) && !cmd.has_opt("common", 4) && !cmd.has_opt("fixed", 4) {
                s.method = "hs".into();
            }
            (es, se)
        }
    };
    let def = if s.method == "hs" { (Model::Random, "hs".to_string()) } else { (Model::Random, "reml".to_string()) };
    let (m, meth) = model_opts(cmd, def, &s.es)?;
    s.model = m;
    s.method = meth;
    declare(ds, st, cmd, s, es, se, &mask)
}

/// `meta update`: re-declare with changed options.
fn update(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let mut s = settings(ds)?;
    if let Some(v) = cmd.opt("studylabel", 6) {
        s.studylabel = crate::xmodels::var_name(ds, v)?;
    }
    if let Some(v) = cmd.opt("studysize", 6) {
        s.studysize = crate::xmodels::var_name(ds, v)?;
    }
    if let Some(l) = cmd.opt("eslabel", 3) {
        s.eslabel = util::unquote(l);
    }
    s.level = level_opt(cmd, st, s.level)?;
    if let Some(z) = cmd.opt("zerocells", 4) {
        s.zc = z.trim().to_string();
    }
    if let Some(e) = cmd.opt("esize", 3) {
        let e = e.trim().to_lowercase();
        let e = if e == "glassdelta" { "glassdelta2".to_string() } else { e };
        let ok = match s.datatype.as_str() {
            "continuous" => ["mdiff", "hedgesg", "cohend", "glassdelta1", "glassdelta2"].contains(&e.as_str()),
            "binary" => ["lnor", "lnrr", "rdiff", "lnorpeto"].contains(&e.as_str()),
            _ => false,
        };
        if !ok {
            return Err(rc_err(198, format!("esize({e}) not allowed with {} data", s.datatype)));
        }
        s.es = e;
        s.eslabel = es_label(&s.es).into();
    }
    let (m, meth) = model_opts(cmd, (s.model, s.method.clone()), &s.es)?;
    s.model = m;
    s.method = meth;
    let mask: Vec<bool> = {
        let id = util::col(ds, "_meta_id")?;
        let mut m = util::touse(ds, st, cmd)?;
        if cmd.if_expr.is_none() && cmd.in_range.is_none() {
            for i in 0..ds.n_obs {
                m[i] = !is_missing(id[i]);
            }
        }
        m
    };
    let (es, se) = if s.datatype == "generic" {
        (util::col(ds, "_meta_es")?, util::col(ds, "_meta_se")?)
    } else {
        compute_es(ds, &s, &mask)?
    };
    declare(ds, st, cmd, s, es, se, &mask)
}

/// Studies in the current analysis sample: (row, es, se, label).
pub struct Studies {
    pub rows: Vec<usize>,
    pub y: Vec<f64>,
    pub se: Vec<f64>,
    pub labels: Vec<String>,
    pub cells: Option<Vec<[f64; 4]>>,
    pub size: Vec<f64>,
}

impl Studies {
    pub fn v(&self) -> Vec<f64> {
        self.se.iter().map(|s| s * s).collect()
    }
    pub fn k(&self) -> usize {
        self.y.len()
    }
}

pub fn studies(ds: &DataSet, st: &Store, cmd: &Command, s: &Settings, extra: &[String]) -> Result<Studies, String> {
    let mut mask = util::touse(ds, st, cmd)?;
    let mut need: Vec<String> = vec!["_meta_es".into(), "_meta_se".into()];
    need.extend(extra.iter().cloned());
    util::markout(ds, &mut mask, &need);
    let y = util::col(ds, "_meta_es")?;
    let se = util::col(ds, "_meta_se")?;
    let lab: Vec<String> = match ds.get_var("_meta_studylabel") {
        Some(v) => (0..ds.n_obs).map(|i| util::cell_text(ds, v, i, true).trim().to_string()).collect(),
        None => (0..ds.n_obs).map(|i| format!("Study {}", i + 1)).collect(),
    };
    let size = if ds.var_index.contains_key("_meta_studysize") { util::col(ds, "_meta_studysize")? } else { vec![SYSMISS; ds.n_obs] };
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let cells = if s.datatype == "binary" {
        let cs: Vec<Vec<f64>> = s.datavars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        Some(rows.iter().map(|&i| [cs[0][i], cs[1][i], cs[2][i], cs[3][i]]).collect())
    } else {
        None
    };
    Ok(Studies { y: rows.iter().map(|&i| y[i]).collect(), se: rows.iter().map(|&i| se[i]).collect(), labels: rows.iter().map(|&i| lab[i].clone()).collect(), size: rows.iter().map(|&i| size[i]).collect(), cells, rows })
}

pub struct Meta;

impl CommandHandler for Meta {
    fn name(&self) -> &'static str {
        "meta"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (sub, rest) = crate::xmodels::first_word(cmd.main.trim());
        let sub = sub.to_lowercase();
        match sub.as_str() {
            "set" | "esize" | "es" | "esi" | "esiz" | "psycorr" => {
                let s = if sub.starts_with("es") { "esize" } else { sub.as_str() };
                declare_cmd(ds, st, cmd, s, &rest)
            }
            "update" | "upd" | "upda" | "updat" => update(ds, st, cmd),
            "" | "query" | "q" | "qu" | "que" | "quer" | "describe" | "d" | "de" | "des" | "desc" | "descr" | "descri" | "describ" => {
                let s = match settings(ds) {
                    Ok(s) => s,
                    Err(_) if sub.is_empty() || sub.starts_with('q') => return Ok("(data not meta set; use meta set or meta esize to declare as meta data)".into()),
                    Err(e) => return Err(e),
                };
                let id = util::col(ds, "_meta_id")?;
                let k = id.iter().filter(|v| !is_missing(**v)).count();
                let r = st.begin_r();
                r.set("K", k as f64);
                if sub.starts_with('d') {
                    let mut out = vec![String::new(), "  Contains variables declared by meta:".to_string()];
                    for v in VARS {
                        if ds.var_index.contains_key(v) {
                            out.push(format!("    {v}"));
                        }
                    }
                    out.push(describe(ds, &s, k));
                    return Ok(out.join("\n"));
                }
                Ok(format!("-> {}\n{}", s.cmdline, describe(ds, &s, k)))
            }
            "clear" => {
                for v in VARS {
                    if ds.var_index.contains_key(v) {
                        let _ = ds.drop_variable(v);
                    }
                }
                if let Some(m) = ds.characteristics.get_mut("_dta") {
                    m.retain(|k, _| !k.starts_with("_meta_"));
                }
                Ok("(meta settings cleared)".into())
            }
            "summarize" | "sum" | "summ" | "summa" | "summar" | "summari" | "summariz" => summ::summarize(ds, st, cmd),
            "forestplot" | "forest" => plots::forest(ds, st, cmd),
            "funnelplot" | "funnel" => plots::funnel(ds, st, cmd),
            "labbeplot" | "labbe" => plots::labbe(ds, st, cmd),
            "galbraithplot" | "galbraith" => plots::galbraith(ds, st, cmd),
            "bias" => bias::bias(ds, st, cmd, &rest),
            "trimfill" => bias::trimfill(ds, st, cmd),
            "regress" | "reg" | "regr" | "regre" | "regres" => reg::regress(ds, st, cmd, &rest),
            "mvregress" | "mvreg" => reg::mvregress(ds, st, cmd, &rest),
            "meregress" | "mereg" => reg::meregress(ds, st, cmd, &rest, false),
            "multilevel" => reg::meregress(ds, st, cmd, &rest, true),
            o => Err(rc_err(199, format!("meta {o} is unrecognized"))),
        }
    }
}
