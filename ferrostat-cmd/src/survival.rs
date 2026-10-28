//! Survival analysis: `stset`, `sts graph` (Kaplan-Meier survivor and
//! Nelson-Aalen cumulative hazard), `strate` (failure rates with exact
//! Poisson intervals) and `ltable` (actuarial life tables).
//!
//! `stset` records the analysis in the variables `_t` (exit time), `_t0`
//! (entry time), `_d` (failure indicator) and `_st` (in the analysis); the
//! other commands read them. References: Kalbfleisch & Prentice, *The
//! Statistical Analysis of Failure Time Data*; Collett, *Modelling Survival
//! Data in Medical Research*.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::store::Store;
use crate::util;
use crate::varlist;

fn set_var(ds: &mut DataSet, name: &str, vt: VarType, vals: &[f64]) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    let mut v = Variable::new(name, vt);
    v.data = vals.iter().map(|&x| if is_missing(x) { Value::Missing } else if vt == VarType::Byte { Value::Byte(x as i8) } else { Value::Double(x) }).collect();
    ds.add_variable(v)
}

// ---------------------------------------------------------------------------
// st settings shared by the survival commands
// ---------------------------------------------------------------------------

/// One record of the st analysis: entry and exit analysis times, failure
/// flag, observation number and the stset weight (1 when unweighted).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StObs {
    pub t0: f64,
    pub t: f64,
    pub d: bool,
    pub row: usize,
    pub w: f64,
}

/// A `_dta[st_*]` characteristic.
pub fn st_char(ds: &DataSet, key: &str) -> Option<String> {
    ds.characteristics.get("_dta").and_then(|c| c.get(key)).filter(|s| !s.is_empty()).cloned()
}

/// Error unless the data have been stset.
pub fn require_st(ds: &DataSet) -> Result<(), String> {
    if ["_t", "_d", "_st", "_t0"].iter().any(|v| ds.get_var(v).is_none()) {
        return Err(rc_err(119, "data not st; use stset"));
    }
    Ok(())
}

/// The id() variable of stset, if any.
pub fn st_id(ds: &DataSet) -> Option<String> {
    st_char(ds, "st_id").filter(|v| ds.get_var(v).is_some())
}

/// The stset weight: (kind, values) where kind is fweight/pweight/iweight.
pub fn st_weight(ds: &DataSet, st: &Store) -> Result<Option<(String, Vec<f64>)>, String> {
    let (Some(kind), Some(expr)) = (st_char(ds, "st_wt"), st_char(ds, "st_wv")) else { return Ok(None) };
    let e = ferrostat_parser::parser::parse_expression(&expr).map_err(|e| rc_err(198, e.message))?;
    let w = util::eval_col(ds, st, &e, &vec![true; ds.n_obs])?;
    Ok(Some((kind, w)))
}

/// Records in the st analysis among the observations selected by `mask`.
pub fn st_obs_mask(ds: &DataSet, st: &Store, mask: &[bool]) -> Result<Vec<StObs>, String> {
    require_st(ds)?;
    let (t, t0, d, s) = (util::col(ds, "_t")?, util::col(ds, "_t0")?, util::col(ds, "_d")?, util::col(ds, "_st")?);
    let w = st_weight(ds, st)?.map(|(_, w)| w);
    Ok((0..ds.n_obs)
        .filter(|&i| mask[i] && s[i] == 1.0 && !is_missing(t[i]))
        .map(|i| StObs { t0: if is_missing(t0[i]) { 0.0 } else { t0[i] }, t: t[i], d: d[i] == 1.0, row: i, w: w.as_ref().map_or(1.0, |w| w[i]) })
        .collect())
}

/// Records in the st analysis and in the command's if/in sample.
pub fn st_obs(ds: &DataSet, st: &Store, cmd: &Command) -> Result<Vec<StObs>, String> {
    require_st(ds)?;
    let mask = util::touse(ds, st, cmd)?;
    st_obs_mask(ds, st, &mask)
}

/// `%w.0gc`: like fmt_g with thousands separators in the integer part.
pub fn gc(v: f64, width: usize) -> String {
    let s = fmt_g(v, width);
    if s.contains('e') || is_missing(v) {
        return s;
    }
    let (sign, body) = if let Some(b) = s.strip_prefix('-') { ("-", b) } else { ("", s.as_str()) };
    let (int, frac) = match body.split_once('.') {
        Some((a, b)) => (a, format!(".{b}")),
        None => (body, String::new()),
    };
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{sign}{out}{frac}")
}

/// Text describing how analysis time is computed: `t1`, `(t1-origin)/365.25`.
fn analysis_time_text(ds: &DataSet) -> String {
    let bt = st_char(ds, "st_bt").or_else(|| st_char(ds, "st_t").filter(|t| t != "_t")).unwrap_or_else(|| "_t".into());
    let scale = st_char(ds, "st_s").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
    let base = if st_char(ds, "st_orig").is_some() { format!("({bt}-origin)") } else { bt };
    if scale != 1.0 { format!("{base}/{}", fmt_g(scale, 10)) } else { base }
}

/// The failure/analysis-time block printed by the st commands.
pub fn st_header(ds: &DataSet) -> String {
    let mut rows: Vec<(&str, String)> = Vec::new();
    let fail = match st_char(ds, "st_bd").or_else(|| st_char(ds, "st_d").filter(|d| d != "_d")) {
        Some(v) => match st_char(ds, "st_ev") {
            Some(ev) => format!("{v}=={ev}"),
            None => v,
        },
        None => "1 (meaning all fail)".into(),
    };
    rows.push(("Failure _d", fail));
    rows.push(("Analysis time _t", analysis_time_text(ds)));
    if let Some(o) = st_char(ds, "st_orig") {
        rows.push(("Origin", o));
    }
    if let Some(e) = st_char(ds, "st_enter") {
        rows.push(("Enter on or after", e));
    }
    if let Some(e) = st_char(ds, "st_exit") {
        rows.push(("Exit on or before", e));
    }
    if let Some(id) = st_char(ds, "st_id") {
        rows.push(("ID variable", id));
    }
    if let (Some(k), Some(v)) = (st_char(ds, "st_wt"), st_char(ds, "st_wv")) {
        rows.push(("Weight", format!("[{k}={v}]")));
    }
    rows.iter().map(|(l, v)| format!("{l:>18}: {v}")).collect::<Vec<_>>().join("\n")
}

/// How an origin(), enter() or exit() time is specified.
enum TimeSpec {
    /// `time exp`: one value per observation.
    Time(Vec<f64>),
    /// `varname==numlist` (empty list: varname != 0 & varname < .).
    Event(Vec<f64>, Vec<f64>),
    Failure,
    Min,
    Never,
}

fn parse_time_spec(ds: &DataSet, st: &Store, text: &str) -> Result<TimeSpec, String> {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("time ").or_else(|| t.strip_prefix("time(").map(|r| r.trim_end_matches(')'))) {
        let rest = rest.trim();
        if rest == "." {
            return Ok(TimeSpec::Never);
        }
        let e = ferrostat_parser::parser::parse_expression(rest).map_err(|e| rc_err(198, e.message))?;
        return Ok(TimeSpec::Time(util::eval_col(ds, st, &e, &vec![true; ds.n_obs])?));
    }
    if "failure".starts_with(t) && !t.is_empty() {
        return Ok(TimeSpec::Failure);
    }
    if t == "min" {
        return Ok(TimeSpec::Min);
    }
    if t == "." {
        return Ok(TimeSpec::Never);
    }
    let (v, codes) = match t.split_once("==") {
        Some((v, l)) => (v.trim(), crate::margins::numlist(l)?),
        None => (t, Vec::new()),
    };
    let v = varlist::resolve_name(ds, v)?;
    Ok(TimeSpec::Event(util::col(ds, &v)?, codes))
}

fn event_match(vals: &[f64], codes: &[f64], i: usize) -> bool {
    let x = vals[i];
    if is_missing(x) {
        return false;
    }
    if codes.is_empty() { x != 0.0 } else { codes.contains(&x) }
}

/// Remove the st variables and characteristics.
fn st_clear(ds: &mut DataSet) -> Result<(), String> {
    for v in ["_st", "_d", "_t", "_t0"] {
        if ds.get_var(v).is_some() {
            ds.drop_variable(v)?;
        }
    }
    if let Some(c) = ds.characteristics.get_mut("_dta") {
        c.retain(|k, _| !k.starts_with("st_") && k != "_dta");
    }
    Ok(())
}

fn label_var(ds: &mut DataSet, name: &str, label: &str) {
    if let Some(v) = ds.get_var_mut(name) {
        v.label = label.to_string();
    }
}

pub struct Stset;

impl CommandHandler for Stset {
    fn name(&self) -> &'static str {
        "stset"
    }

    /// stset timevar [if] [weight] [, failure(var[==numlist]) id(var)
    ///     origin(time exp|var==numlist|min) enter(time exp|var==numlist)
    ///     exit(failure|time exp|var==numlist) scale(#) time0(var)]
    /// stset, clear
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if cmd.has_opt("clear", 5) {
                st_clear(ds)?;
                return Ok(String::new());
            }
            let raw = st_char(ds, "st_cmd").ok_or_else(|| rc_err(119, "data not st; use stset"))?;
            let c = ferrostat_parser::parser::parse_line(&raw).map_err(|e| rc_err(198, e.message))?;
            return self.execute(ds, st, &c);
        }
        let tvar = varlist::resolve_name(ds, cmd.main.trim())?;
        let n = ds.n_obs;
        let mut mask = util::touse(ds, st, cmd)?;
        let n_if = mask.iter().filter(|m| !**m).count();
        let weights = util::weights(ds, st, cmd, &mut mask, "fpi")?;
        let wtext = cmd.weight.as_ref().and_then(|_| {
            let a = cmd.raw.find('[')?;
            let b = cmd.raw[a..].find(']')? + a;
            let (k, v) = cmd.raw[a + 1..b].split_once('=')?;
            let k = match k.trim() {
                "fw" | "fweight" => "fweight",
                "pw" | "pweight" => "pweight",
                "iw" | "iweight" => "iweight",
                _ => "fweight",
            };
            Some((k.to_string(), v.trim().to_string()))
        });
        let n_weight = mask.iter().filter(|m| !**m).count() - n_if;
        let t = util::col(ds, &tvar)?;
        let (fvar, codes): (Option<String>, Vec<f64>) = match cmd.opt("failure", 1) {
            Some(f) => {
                let (v, list) = match f.split_once("==") {
                    Some((v, l)) => (v.trim().to_string(), crate::margins::numlist(l)?),
                    None => (f.trim().to_string(), Vec::new()),
                };
                (Some(varlist::resolve_name(ds, &v)?), list)
            }
            None => (None, Vec::new()),
        };
        let fcol = fvar.as_ref().map(|v| util::col(ds, v)).transpose()?;
        let idvar = cmd.opt("id", 2).map(|v| varlist::resolve_name(ds, v.trim())).transpose()?;
        let t0var = cmd.opt("time0", 2).map(|v| varlist::resolve_name(ds, v.trim())).transpose()?;
        let t0col = t0var.as_ref().map(|v| util::col(ds, v)).transpose()?;
        let origin = cmd.opt("origin", 2).map(|o| parse_time_spec(ds, st, o)).transpose()?;
        let enter = cmd.opt("enter", 2).map(|o| parse_time_spec(ds, st, o)).transpose()?;
        let exit = cmd.opt("exit", 2).map(|o| parse_time_spec(ds, st, o)).transpose()?.unwrap_or(TimeSpec::Failure);
        let scale = util::opt_num(cmd, "scale", 2)?.unwrap_or(1.0);
        if scale <= 0.0 {
            return Err(rc_err(198, "scale() must be positive"));
        }
        // marking out
        let mut n_tmiss = 0;
        for i in 0..n {
            if mask[i] && is_missing(t[i]) {
                mask[i] = false;
                n_tmiss += 1;
            }
        }
        let mut n_idmiss = 0;
        let idcol = idvar.as_ref().map(|v| util::col(ds, v)).transpose()?;
        if let Some(c) = &idcol {
            for i in 0..n {
                if mask[i] && is_missing(c[i]) {
                    mask[i] = false;
                    n_idmiss += 1;
                }
            }
        }
        // subjects: rows sorted by exit time
        let mut subjects: Vec<Vec<usize>> = match &idcol {
            Some(c) => {
                let mut m: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
                for i in 0..n {
                    if mask[i] {
                        m.entry(c[i].to_bits()).or_default().push(i);
                    }
                }
                m.into_values().collect()
            }
            None => (0..n).filter(|&i| mask[i]).map(|i| vec![i]).collect(),
        };
        for s in subjects.iter_mut() {
            s.sort_by(|&a, &b| t[a].total_cmp(&t[b]).then(a.cmp(&b)));
        }
        let fails = |i: usize| match &fcol {
            None => true,
            Some(f) => event_match(f, &codes, i),
        };
        let first_event = |rows: &[usize], vals: &[f64], codes: &[f64]| rows.iter().find(|&&i| event_match(vals, codes, i)).map(|&i| t[i]);
        let (mut stv, mut tv, mut t0v, mut dv) = (vec![0.0; n], vec![SYSMISS; n], vec![SYSMISS; n], vec![SYSMISS; n]);
        let (mut n_enter, mut n_exit, mut n_orig) = (0usize, 0usize, 0usize);
        let mut n_subj = 0usize;
        for rows in &subjects {
            let org = match &origin {
                None => 0.0,
                Some(TimeSpec::Time(v)) => v[rows[0]],
                Some(TimeSpec::Event(v, c)) => first_event(rows, v, c).unwrap_or(SYSMISS),
                Some(TimeSpec::Min) => rows.iter().filter_map(|&i| t0col.as_ref().map(|c| c[i])).filter(|x| !is_missing(*x)).fold(f64::INFINITY, f64::min).min(0.0),
                Some(_) => 0.0,
            };
            let ent = match &enter {
                None => f64::NEG_INFINITY,
                Some(TimeSpec::Time(v)) => if is_missing(v[rows[0]]) { f64::NEG_INFINITY } else { v[rows[0]] },
                Some(TimeSpec::Event(v, c)) => first_event(rows, v, c).unwrap_or(f64::INFINITY),
                Some(_) => f64::NEG_INFINITY,
            };
            let ext = match &exit {
                TimeSpec::Failure => rows.iter().find(|&&i| fails(i)).map_or(f64::INFINITY, |&i| t[i]),
                TimeSpec::Never | TimeSpec::Min => f64::INFINITY,
                TimeSpec::Time(v) => if is_missing(v[rows[0]]) { f64::INFINITY } else { v[rows[0]] },
                TimeSpec::Event(v, c) => first_event(rows, v, c).unwrap_or(f64::INFINITY),
            };
            let mut used = false;
            for (k, &i) in rows.iter().enumerate() {
                if is_missing(org) {
                    n_orig += 1;
                    continue;
                }
                let raw0 = match &t0col {
                    Some(c) => c[i],
                    None if idcol.is_some() && k > 0 => t[rows[k - 1]],
                    None => org,
                };
                let raw0 = if is_missing(raw0) { org } else { raw0 };
                let begin = raw0.max(org).max(ent);
                if t[i] <= org && origin.is_some() {
                    n_orig += 1;
                    continue;
                }
                if t[i] <= begin {
                    n_enter += 1;
                    continue;
                }
                if begin >= ext {
                    n_exit += 1;
                    continue;
                }
                let (end, d) = if t[i] > ext { (ext, false) } else { (t[i], fails(i)) };
                stv[i] = 1.0;
                tv[i] = (end - org) / scale;
                t0v[i] = (begin - org) / scale;
                dv[i] = d as u8 as f64;
                used = true;
            }
            n_subj += used as usize;
        }
        // store
        st_clear(ds)?;
        set_var(ds, "_st", VarType::Byte, &stv)?;
        set_var(ds, "_d", VarType::Byte, &dv)?;
        set_var(ds, "_t", VarType::Double, &tv)?;
        set_var(ds, "_t0", VarType::Double, &t0v)?;
        label_var(ds, "_st", "1 if record is to be used; 0 otherwise");
        label_var(ds, "_d", "1 if failure; 0 if censored");
        label_var(ds, "_t", "Analysis time when record ends");
        label_var(ds, "_t0", "Analysis time when record begins");
        let spec = |c: &Command, name: &str, min: usize| c.opt(name, min).map(|s| s.trim().to_string());
        {
            let ch = ds.characteristics.entry("_dta".into()).or_default();
            let mut put = |k: &str, v: String| {
                ch.insert(k.into(), v);
            };
            put("_dta", "st".into());
            put("st_ver", "2".into());
            put("st_t", "_t".into());
            put("st_t0", "_t0".into());
            put("st_d", "_d".into());
            put("st_bt", tvar.clone());
            put("st_s", fmt_g(scale, 12));
            put("st_cmd", cmd.raw.clone());
            if let Some(f) = &fvar {
                put("st_bd", f.clone());
                if !codes.is_empty() {
                    put("st_ev", codes.iter().map(|c| fmt_g(*c, 12)).collect::<Vec<_>>().join(" "));
                }
            }
            if let Some(id) = &idvar {
                put("st_id", id.clone());
            }
            if let Some(v) = &t0var {
                put("st_bt0", v.clone());
            }
            if let Some(o) = spec(cmd, "origin", 2) {
                put("st_orig", o);
            }
            if let Some(o) = spec(cmd, "enter", 2) {
                put("st_enter", o);
            }
            if let Some(o) = spec(cmd, "exit", 2) {
                if !"failure".starts_with(o.as_str()) {
                    put("st_exit", o);
                }
            }
            if let Some((k, v)) = &wtext {
                put("st_wt", k.clone());
                put("st_wv", v.clone());
            }
        }
        let _ = weights;
        // report
        let used = stv.iter().filter(|&&s| s == 1.0).count();
        let failures: f64 = dv.iter().filter(|d| !is_missing(**d)).sum();
        let risk: f64 = (0..n).filter(|&i| stv[i] == 1.0).map(|i| tv[i] - t0v[i]).sum();
        let fdesc = match &fvar {
            None => format!("(assumed to fail at time={tvar})"),
            Some(f) if codes.is_empty() => format!("{f}!=0 & {f}<."),
            Some(f) => format!("{f}=={}", codes.iter().map(|c| fmt_g(*c, 9)).collect::<Vec<_>>().join(" ")),
        };
        let mut out = vec!["Survival-time data settings".to_string(), String::new()];
        if let Some(id) = &idvar {
            out.push(format!("{:>22}: {id}", "ID variable"));
        }
        out.push(format!("{:>22}: {fdesc}", "Failure event"));
        let interval = match (&t0var, &idvar) {
            (Some(v), _) => format!("({v}, {tvar}]"),
            (None, Some(_)) => format!("({tvar}[_n-1], {tvar}]"),
            (None, None) => format!("(0, {tvar}]"),
        };
        out.push(format!("{:>22}: {interval}", "Observed time interval"));
        if let Some(e) = spec(cmd, "enter", 2) {
            out.push(format!("{:>22}: {e}", "Enter on or after"));
        }
        let exit_text = match spec(cmd, "exit", 2) {
            Some(e) if !"failure".starts_with(e.as_str()) => e,
            _ => "failure".into(),
        };
        out.push(format!("{:>22}: {exit_text}", "Exit on or before"));
        if origin.is_some() || scale != 1.0 {
            out.push(format!("{:>22}: {}", "Time for analysis", analysis_time_text(ds)));
        }
        if let Some(o) = spec(cmd, "origin", 2) {
            out.push(format!("{:>22}: {o}", "Origin"));
        }
        if let Some((k, v)) = &wtext {
            out.push(format!("{:>22}: [{k}={v}]", "Weight"));
        }
        out.push(String::new());
        let bar = "-".repeat(74);
        out.push(bar.clone());
        out.push(format!("{:>11}  total observations", util::commas(n)));
        let mut excl = Vec::new();
        if n_if > 0 {
            excl.push(format!("{:>11}  ignored at outset because of if/in", util::commas(n_if)));
        }
        if n_weight > 0 {
            excl.push(format!("{:>11}  weights invalid", util::commas(n_weight)));
        }
        if n_tmiss > 0 {
            excl.push(format!("{:>11}  event time missing ({tvar}>=.){:>18}", util::commas(n_tmiss), "PROBABLE ERROR"));
        }
        if n_idmiss > 0 {
            excl.push(format!("{:>11}  missing id ({})", util::commas(n_idmiss), idvar.clone().unwrap_or_default()));
        }
        if n_orig > 0 {
            excl.push(format!("{:>11}  observations end on or before origin()", util::commas(n_orig)));
        }
        if n_enter > 0 {
            excl.push(format!("{:>11}  observations end on or before enter()", util::commas(n_enter)));
        }
        if n_exit > 0 {
            excl.push(format!("{:>11}  observations begin on or after exit", util::commas(n_exit)));
        }
        if excl.is_empty() {
            out.push(format!("{:>11}  exclusions", 0));
        } else {
            out.extend(excl);
        }
        out.push(bar);
        out.push(format!("{:>11}  observations remaining, representing", util::commas(used)));
        if idvar.is_some() {
            out.push(format!("{:>11}  subjects", util::commas(n_subj)));
        }
        let kind = match (&idvar, &exit) {
            (None, _) => "single-record/single-failure data",
            (Some(_), TimeSpec::Failure) => "single-failure-per-subject data",
            (Some(_), _) => "multiple-failure-per-subject data",
        };
        out.push(format!("{:>11}  failures in {kind}", gc(failures, 10)));
        out.push(format!("{:>11}  total analysis time at risk and under observation", gc(risk, 10)));
        let min0 = (0..n).filter(|&i| stv[i] == 1.0).map(|i| t0v[i]).fold(f64::INFINITY, f64::min);
        let maxt = (0..n).filter(|&i| stv[i] == 1.0).map(|i| tv[i]).fold(f64::NEG_INFINITY, f64::max);
        if used > 0 {
            out.push(format!("{:>64}{:>10}", "At risk from t =", 0));
            out.push(format!("{:>64}{:>10}", "Earliest observed entry t =", gc(min0, 9)));
            out.push(format!("{:>64}{:>10}", "Last observed exit t =", gc(maxt, 9)));
        }
        Ok(out.join("\n"))
    }
}

/// (t, d, row) of the observations in the st analysis and the command sample.
fn st_data(ds: &DataSet, st: &Store, cmd: &Command) -> Result<Vec<(f64, bool, usize)>, String> {
    Ok(st_obs(ds, st, cmd)?.into_iter().map(|o| (o.t, o.d, o.row)).collect())
}

/// Groups of a by() variable: (label, observation rows); one group when absent.
fn by_groups(ds: &DataSet, var: Option<&str>, data: &[(f64, bool, usize)]) -> Result<Vec<(String, Vec<(f64, bool, usize)>)>, String> {
    let Some(v) = var else { return Ok(vec![(String::new(), data.to_vec())]) };
    let v = varlist::resolve_name(ds, v.trim())?;
    let c = util::col(ds, &v)?;
    let var = ds.get_var(&v).unwrap();
    let mut levels: Vec<f64> = data.iter().map(|r| c[r.2]).filter(|x| !is_missing(*x)).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    Ok(levels
        .into_iter()
        .map(|lv| {
            let rows: Vec<(f64, bool, usize)> = data.iter().filter(|r| c[r.2] == lv).cloned().collect();
            (format!("{v} = {}", util::cell_text(ds, var, rows[0].2, true)), rows)
        })
        .collect())
}

/// Risk-set table with delayed entry: a record is at risk at `tj` when
/// `_t0 < tj <= _t`.
fn risk_table_t0(rows: &[(f64, bool, usize)], t0: &[f64]) -> Vec<(f64, f64, f64)> {
    let mut times: Vec<f64> = rows.iter().filter(|r| r.1).map(|r| r.0).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times.into_iter().map(|tj| (tj, rows.iter().filter(|r| r.0 >= tj && t0[r.2] < tj).count() as f64, rows.iter().filter(|r| r.0 == tj && r.1).count() as f64)).collect()
}

/// Risk-set table: (time, at risk, failures) at each distinct failure time.
#[cfg(test)]
fn risk_table(rows: &[(f64, bool, usize)]) -> Vec<(f64, f64, f64)> {
    let mut times: Vec<f64> = rows.iter().filter(|r| r.1).map(|r| r.0).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times.into_iter().map(|tj| (tj, rows.iter().filter(|r| r.0 >= tj).count() as f64, rows.iter().filter(|r| r.0 == tj && r.1).count() as f64)).collect()
}

pub struct Sts;

impl CommandHandler for Sts {
    fn name(&self) -> &'static str {
        "sts"
    }

    /// sts graph [, by(var) na atrisk]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let sub = cmd.main.split_whitespace().next().unwrap_or("graph");
        if !matches!(sub, "graph" | "gr") {
            return crate::stsum::sts(ds, st, cmd, sub);
        }
        let data = st_data(ds, st, cmd)?;
        let t0 = util::col(ds, "_t0")?;
        let na = cmd.has_opt("na", 2) || cmd.has_opt("cumhaz", 4);
        let groups = by_groups(ds, cmd.opt("by", 2).as_deref(), &data)?;
        let mut graph = Graph::default();
        let tmax = data.iter().map(|r| r.0).fold(0.0, f64::max);
        for (label, rows) in &groups {
            // step function starting at 0
            let mut level = if na { 0.0 } else { 1.0 };
            let mut pts = vec![(0.0, level)];
            for (tj, nj, dj) in risk_table_t0(rows, &t0) {
                pts.push((tj, level));
                level = if na { level + dj / nj } else { level * (1.0 - dj / nj) };
                pts.push((tj, level));
            }
            let last = rows.iter().map(|r| r.0).fold(0.0, f64::max);
            pts.push((last, level));
            graph.layers.push(Layer { kind: LayerKind::Line, points: pts, label: if label.is_empty() { (if na { "Nelson-Aalen cumulative hazard" } else { "Kaplan-Meier survival" }).into() } else { label.clone() }, ..Default::default() });
            if cmd.has_opt("atrisk", 3) {
                // numbers at risk at six evenly spaced times, written just above the axis
                let mut marks = Layer { kind: LayerKind::Scatter, label: String::new(), hollow: true, ..Default::default() };
                for k in 0..6 {
                    let tk = tmax * k as f64 / 5.0;
                    marks.points.push((tk, if na { 0.0 } else { 0.02 }));
                    marks.sizes.push(0.0);
                    marks.mlabels.push(rows.iter().filter(|r| r.0 >= tk && (t0[r.2] < tk || t0[r.2] <= 0.0)).count().to_string());
                }
                graph.layers.push(marks);
            }
        }
        graph.title = if na { "Nelson-Aalen cumulative hazard estimate".into() } else { "Kaplan-Meier survival estimate".into() };
        graph.xtitle = "Analysis time".into();
        graph.legend = groups.len() > 1;
        if cmd.has_opt("atrisk", 3) {
            // the number-at-risk labels are not legend entries
            graph.legend_order = graph.layers.iter().enumerate().filter(|(_, l)| l.kind == LayerKind::Line).map(|(k, _)| (k + 1, String::new())).collect();
            if groups.len() == 1 {
                graph.legend = false;
            }
        }
        let extra = graph::finish(&mut graph, cmd, st)?;
        Ok([st_header(ds), extra].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"))
    }
}

/// Exact Poisson confidence interval for a rate (for other modules).
pub fn poisson_ci_pub(d: f64, y: f64, level: f64) -> (f64, f64) {
    poisson_ci(d, y, level)
}

/// Exact Poisson confidence interval for a rate with `d` events in `y` time units.
fn poisson_ci(d: f64, y: f64, level: f64) -> (f64, f64) {
    let a = (1.0 - level / 100.0) / 2.0;
    let lo = if d > 0.0 { dist::invchi2(2.0 * d, a) / 2.0 } else { 0.0 };
    let hi = dist::invchi2(2.0 * (d + 1.0), 1.0 - a) / 2.0;
    (lo / y, hi / y)
}

pub struct Strate;

impl CommandHandler for Strate {
    fn name(&self) -> &'static str {
        "strate"
    }

    /// strate [var] [, graph nowhisker recast() ciopts()]
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let data = st_data(ds, st, cmd)?;
        let var = cmd.main.split_whitespace().next();
        let groups = by_groups(ds, var, &data)?;
        let level = util::level(cmd, st)?;
        let t0 = util::col(ds, "_t0")?;
        let name = var.map(|v| varlist::resolve_name(ds, v)).transpose()?.unwrap_or_default();
        let mut out = vec![
            "Estimated failure rates".to_string(),
            format!("Number of records = {}", util::commas(data.len())),
            String::new(),
            format!("  {:>10} {:>6} {:>10} {:>10} {:>10} {:>10}", if name.is_empty() { "" } else { name.as_str() }, "D", "Y", "Rate", "Lower", "Upper"),
        ];
        let mut rates = Layer { kind: LayerKind::Scatter, label: "Rate".into(), ..Default::default() };
        let mut cis = Layer { kind: LayerKind::Range(RangeStyle::Spike), label: format!("{}% CI", fmt_g(level, 4)), ..Default::default() };
        let mut labels = Vec::new();
        for (k, (label, rows)) in groups.iter().enumerate() {
            let d = rows.iter().filter(|r| r.1).count() as f64;
            let y: f64 = rows.iter().map(|r| r.0 - t0[r.2]).sum();
            let rate = d / y;
            let (lo, hi) = poisson_ci(d, y, level);
            let lv = label.split(" = ").nth(1).unwrap_or("").to_string();
            out.push(format!("  {:>10} {:>6} {:>10} {:>10.6} {:>10.6} {:>10.6}", lv, d, fmt_g(y, 9), rate, lo, hi));
            let x = k as f64 + 1.0;
            rates.points.push((x, rate));
            cis.ranges.push((x, lo, hi));
            labels.push((x, if lv.is_empty() { "All".to_string() } else { lv }));
        }
        out.push(String::new());
        out.push(" Notes: Rate = D/Y = failures/person-time.".into());
        out.push(format!("        Lower and Upper are bounds of {}% confidence intervals.", fmt_g(level, 4)));
        if cmd.has_opt("graph", 2) {
            if let Some(r) = cmd.opt("recast", 3) {
                rates.kind = if r.trim().starts_with("conn") { LayerKind::Connected } else if r.trim() == "line" { LayerKind::Line } else { LayerKind::Scatter };
            }
            if cmd.opt("ciopts", 3).is_some_and(|c| c.contains("rcap")) {
                cis.kind = LayerKind::Range(RangeStyle::Cap);
            }
            let mut layers = Vec::new();
            if !cmd.has_opt("nowhisker", 3) {
                layers.push(cis);
            }
            layers.push(rates);
            let mut graph = Graph { layers, xlabels: labels, xtitle: name.clone(), ytitle: "Rate".into(), title: "Failure rates and their CIs".into(), ..Default::default() };
            graph.legend = false;
            let extra = graph::finish(&mut graph, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

pub struct Ltable;

impl CommandHandler for Ltable {
    fn name(&self) -> &'static str {
        "ltable"
    }

    /// ltable timevar [deadvar] [, by(var) survival graph noconf overlay intervals(#)]
    /// Actuarial estimator: those lost during an interval count as half
    /// exposed; standard errors by Greenwood's formula; confidence
    /// intervals on the log(-log) scale.
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let tvar = vars.first().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        let mask = util::touse(ds, st, cmd)?;
        let t = util::col(ds, tvar)?;
        let dcol = vars.get(1).map(|v| util::col(ds, v)).transpose()?;
        let data: Vec<(f64, bool, usize)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(t[i])).map(|i| (t[i], dcol.as_ref().map_or(true, |d| d[i] != 0.0 && !is_missing(d[i])), i)).collect();
        let width = util::opt_num(cmd, "intervals", 3)?.unwrap_or(1.0);
        let groups = by_groups(ds, cmd.opt("by", 2).as_deref(), &data)?;
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut out = vec![format!("           Interval     Total   Deaths   Lost   Survival   Std. err.   [{}% conf. int.]", fmt_g(level, 4))];
        let mut children = Vec::new();
        let mut overlay = Graph { xtitle: tvar.clone(), ytitle: "Proportion surviving".into(), legend: true, ..Default::default() };
        for (label, rows) in &groups {
            if !label.is_empty() {
                out.push(label.clone());
            }
            let tmax = rows.iter().map(|r| r.0).fold(0.0, f64::max);
            let nint = (tmax / width).floor() as usize + 1;
            let mut n = rows.len() as f64;
            let (mut s, mut gw) = (1.0f64, 0.0f64);
            let mut curve = vec![(0.0, 1.0)];
            let mut band = Vec::new();
            for j in 0..nint {
                let (a, b) = (j as f64 * width, (j + 1) as f64 * width);
                let inside: Vec<&(f64, bool, usize)> = rows.iter().filter(|r| r.0 >= a && r.0 < b).collect();
                if inside.is_empty() {
                    continue;
                }
                let d = inside.iter().filter(|r| r.1).count() as f64;
                let lost = inside.len() as f64 - d;
                let ne = n - lost / 2.0;
                let q = if ne > 0.0 { d / ne } else { 0.0 };
                s *= 1.0 - q;
                if ne > d {
                    gw += d / (ne * (ne - d));
                }
                let se = s * gw.sqrt();
                // log(-log) interval
                let (lo, hi) = if s > 0.0 && s < 1.0 {
                    let w = z * gw.sqrt() / s.ln().abs();
                    (s.powf((w).exp()), s.powf((-w).exp()))
                } else {
                    (f64::NAN, f64::NAN)
                };
                out.push(format!("  {:>8} {:>8} {:>8} {:>8} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>9.4}", fmt_g(a, 6), fmt_g(b, 6), n, d, lost, s, se, lo, hi));
                curve.push((b, s));
                if lo.is_finite() {
                    band.push((b, lo, hi));
                }
                n -= inside.len() as f64;
            }
            let line = Layer { kind: LayerKind::Line, points: curve, label: if label.is_empty() { "Survival".into() } else { label.clone() }, ..Default::default() };
            let ci = Layer { kind: LayerKind::Range(RangeStyle::Cap), ranges: band, label: format!("{}% CI", fmt_g(level, 4)), ..Default::default() };
            if cmd.has_opt("overlay", 4) {
                overlay.layers.push(line);
            } else {
                let mut g = Graph { xtitle: tvar.clone(), ytitle: "Proportion surviving".into(), title: label.clone(), ..Default::default() };
                if !cmd.has_opt("noconf", 4) {
                    g.layers.push(ci);
                }
                g.layers.push(line);
                children.push(g);
            }
        }
        if cmd.has_opt("graph", 2) {
            let mut graph = if cmd.has_opt("overlay", 4) || children.len() != 1 {
                if cmd.has_opt("overlay", 4) { overlay } else { Graph { cols: children.len().min(3), children, ..Default::default() } }
            } else {
                children.pop().unwrap()
            };
            let extra = graph::finish(&mut graph, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_poisson_interval() {
        // 10 events in 100 time units: the textbook exact 95% interval is (0.04795, 0.18390)
        let (lo, hi) = poisson_ci(10.0, 100.0, 95.0);
        assert!((lo - 0.047954).abs() < 1e-5, "{lo}");
        assert!((hi - 0.183904).abs() < 1e-5, "{hi}");
    }

    #[test]
    fn kaplan_meier_risk_table() {
        // times 1 2 2 3(censored) 4: at t=2 four at risk, two failures
        let rows = vec![(1.0, true, 0), (2.0, true, 1), (2.0, true, 2), (3.0, false, 3), (4.0, true, 4)];
        let rt = risk_table(&rows);
        assert_eq!(rt, vec![(1.0, 5.0, 1.0), (2.0, 4.0, 2.0), (4.0, 1.0, 1.0)]);
        let s: f64 = rt.iter().map(|(_, n, d)| 1.0 - d / n).product();
        assert_eq!(s, 0.0);
        let s2: f64 = rt[..2].iter().map(|(_, n, d)| 1.0 - d / n).product();
        assert!((s2 - 0.4).abs() < 1e-12);
    }
}
