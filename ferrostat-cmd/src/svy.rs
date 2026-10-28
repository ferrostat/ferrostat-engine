//! Complex survey designs: `svyset`, `svydescribe` and the variance
//! machinery behind the `svy:` prefix (see `svy_est`).
//!
//! The design is kept in the dataset characteristics (`_dta[_svy_*]`), so it
//! survives `save` / `use`.  Variances are computed by Taylor linearization
//! (stratified, multistage, with finite population corrections and the
//! `singleunit()` treatments) or from replicate weights (BRR, jackknife,
//! bootstrap, successive difference replication).

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::parser::{find_top_level, split_options};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::{util, varlist};

/// One sampling stage.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stage {
    /// Sampling unit variable; None = individual observations (`_n`).
    pub su: Option<String>,
    pub strata: Option<String>,
    pub fpc: Option<String>,
}

/// A survey design as declared by `svyset`.
#[derive(Debug, Clone, PartialEq)]
pub struct SvySet {
    pub wtype: Option<String>,
    pub wvar: Option<String>,
    pub stages: Vec<Stage>,
    /// linearized, brr, jackknife, bootstrap, sdr
    pub vce: String,
    /// missing, certainty, scaled, centered
    pub singleunit: String,
    pub brrweight: Vec<String>,
    pub jkrweight: Vec<String>,
    pub bsrweight: Vec<String>,
    pub sdrweight: Vec<String>,
    pub fay: f64,
    pub bsn: f64,
    pub mse: bool,
    pub dof: Option<f64>,
    /// jackknife replicate weights: (stratum, multiplier) per variable;
    /// a missing multiplier means (n_h-1)/n_h
    pub jk: Vec<(f64, Option<f64>)>,
}

impl Default for SvySet {
    fn default() -> Self {
        SvySet {
            wtype: None,
            wvar: None,
            stages: vec![Stage::default()],
            vce: "linearized".into(),
            singleunit: "missing".into(),
            brrweight: vec![],
            jkrweight: vec![],
            bsrweight: vec![],
            sdrweight: vec![],
            fay: 0.0,
            bsn: 1.0,
            mse: false,
            dof: None,
            jk: vec![],
        }
    }
}

fn words(s: Option<&String>) -> Vec<String> {
    s.map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

impl SvySet {
    /// The design stored in the dataset characteristics, if any.
    pub fn from_chars(ds: &DataSet) -> Option<SvySet> {
        let c = ds.characteristics.get("_dta")?;
        c.get("_svy_version")?;
        let mut s = SvySet::default();
        s.wtype = c.get("_svy_wtype").cloned().filter(|x| !x.is_empty());
        s.wvar = c.get("_svy_wvar").cloned().filter(|x| !x.is_empty());
        let nst: usize = c.get("_svy_stages").and_then(|x| x.trim().parse().ok()).unwrap_or(1).max(1);
        s.stages = (1..=nst)
            .map(|k| Stage {
                su: c.get(&format!("_svy_su{k}")).cloned().filter(|x| !x.is_empty() && x != "_n"),
                strata: c.get(&format!("_svy_strata{k}")).cloned().filter(|x| !x.is_empty()),
                fpc: c.get(&format!("_svy_fpc{k}")).cloned().filter(|x| !x.is_empty()),
            })
            .collect();
        if let Some(v) = c.get("_svy_vce") {
            s.vce = v.clone();
        }
        if let Some(v) = c.get("_svy_singleunit") {
            s.singleunit = v.clone();
        }
        s.brrweight = words(c.get("_svy_brrweight"));
        s.jkrweight = words(c.get("_svy_jkrweight"));
        s.bsrweight = words(c.get("_svy_bsrweight"));
        s.sdrweight = words(c.get("_svy_sdrweight"));
        s.fay = c.get("_svy_fay").and_then(|x| x.parse().ok()).unwrap_or(0.0);
        s.bsn = c.get("_svy_bsn").and_then(|x| x.parse().ok()).unwrap_or(1.0);
        s.mse = c.get("_svy_mse").is_some_and(|x| !x.is_empty());
        s.dof = c.get("_svy_dof").and_then(|x| x.parse().ok());
        s.jk = s
            .jkrweight
            .iter()
            .map(|v| {
                let vc = ds.characteristics.get(v);
                let get = |k: &str| vc.and_then(|c| c.get(k)).and_then(|x| x.trim().parse::<f64>().ok());
                (get("jk_stratum").unwrap_or(1.0), get("jk_multiplier"))
            })
            .collect();
        Some(s)
    }

    pub fn clear_chars(ds: &mut DataSet) {
        if let Some(c) = ds.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("_svy_"));
        }
    }

    pub fn to_chars(&self, ds: &mut DataSet) {
        SvySet::clear_chars(ds);
        let c = ds.characteristics.entry("_dta".into()).or_default();
        c.insert("_svy_version".into(), "2".into());
        if let (Some(t), Some(v)) = (&self.wtype, &self.wvar) {
            c.insert("_svy_wtype".into(), t.clone());
            c.insert("_svy_wvar".into(), v.clone());
        }
        c.insert("_svy_stages".into(), self.stages.len().to_string());
        for (k, st) in self.stages.iter().enumerate() {
            c.insert(format!("_svy_su{}", k + 1), st.su.clone().unwrap_or_else(|| "_n".into()));
            if let Some(s) = &st.strata {
                c.insert(format!("_svy_strata{}", k + 1), s.clone());
            }
            if let Some(f) = &st.fpc {
                c.insert(format!("_svy_fpc{}", k + 1), f.clone());
            }
        }
        c.insert("_svy_vce".into(), self.vce.clone());
        if self.singleunit != "missing" || self.is_replicate() {
            c.insert("_svy_singleunit".into(), self.singleunit.clone());
        }
        for (key, list) in [("_svy_brrweight", &self.brrweight), ("_svy_jkrweight", &self.jkrweight), ("_svy_bsrweight", &self.bsrweight), ("_svy_sdrweight", &self.sdrweight)] {
            if !list.is_empty() {
                c.insert(key.into(), list.join(" "));
            }
        }
        if self.fay != 0.0 {
            c.insert("_svy_fay".into(), fmt_g(self.fay, 16));
        }
        if self.bsn != 1.0 {
            c.insert("_svy_bsn".into(), fmt_g(self.bsn, 16));
        }
        if self.mse {
            c.insert("_svy_mse".into(), "mse".into());
        }
        if let Some(d) = self.dof {
            c.insert("_svy_dof".into(), fmt_g(d, 16));
        }
        for (k, v) in self.jkrweight.iter().enumerate() {
            let (h, m) = self.jk.get(k).copied().unwrap_or((1.0, None));
            let vc = ds.characteristics.entry(v.clone()).or_default();
            vc.insert("jk_stratum".into(), fmt_g(h, 16));
            match m {
                Some(m) => vc.insert("jk_multiplier".into(), fmt_g(m, 16)),
                None => vc.remove("jk_multiplier"),
            };
        }
    }

    pub fn is_replicate(&self) -> bool {
        !self.brrweight.is_empty() || !self.jkrweight.is_empty() || !self.bsrweight.is_empty() || !self.sdrweight.is_empty()
    }

    /// Replicate weight variables of the given method.
    pub fn rep_weights(&self, vce: &str) -> &[String] {
        match vce {
            "brr" => &self.brrweight,
            "jackknife" => &self.jkrweight,
            "bootstrap" => &self.bsrweight,
            "sdr" => &self.sdrweight,
            _ => &[],
        }
    }

    /// Variables the design needs (weights, units, strata, fpc, replicate weights).
    pub fn variables(&self) -> Vec<String> {
        let mut v = Vec::new();
        if let Some(w) = &self.wvar {
            v.push(w.clone());
        }
        for s in &self.stages {
            for x in [&s.su, &s.strata, &s.fpc].into_iter().flatten() {
                v.push(x.clone());
            }
        }
        v
    }

    /// The settings block shown by `svyset` and `svydescribe`.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        let wlabel = match self.wtype.as_deref() {
            Some("iweight") => "Importance weights",
            _ => "Sampling weights",
        };
        out.push(format!("{wlabel}: {}", self.wvar.clone().unwrap_or_else(|| "<none>".into())));
        let range = |l: &[String]| if l.len() > 2 { format!("{} .. {}", l[0], l[l.len() - 1]) } else { l.join(" ") };
        for (name, list) in [("BRR weights", &self.brrweight), ("JK weights", &self.jkrweight), ("Bootstrap weights", &self.bsrweight), ("SDR weights", &self.sdrweight)] {
            if !list.is_empty() {
                out.push(format!("{name:>16}: {}", range(list)));
            }
        }
        if !self.brrweight.is_empty() && self.fay != 0.0 {
            out.push(format!("{:>16}: {}", "Fay's adjustment", fmt_g(self.fay, 9)));
        }
        if !self.bsrweight.is_empty() && self.bsn != 1.0 {
            out.push(format!("{:>16}: {}", "Bootstrap mean", fmt_g(self.bsn, 9)));
        }
        out.push(format!("{:>16}: {}", "VCE", self.vce));
        if self.mse {
            out.push(format!("{:>16}: on", "MSE"));
        }
        if let Some(d) = self.dof {
            out.push(format!("{:>16}: {}", "Design df", fmt_g(d, 9)));
        }
        out.push(format!("{:>16}: {}", "Single unit", self.singleunit));
        for (k, s) in self.stages.iter().enumerate() {
            out.push(format!("{:>16}: {}", format!("Strata {}", k + 1), s.strata.clone().unwrap_or_else(|| "<one>".into())));
            out.push(format!("{:>16}: {}", format!("Sampling unit {}", k + 1), s.su.clone().unwrap_or_else(|| "<observations>".into())));
            out.push(format!("{:>16}: {}", format!("FPC {}", k + 1), s.fpc.clone().unwrap_or_else(|| "<zero>".into())));
        }
        out
    }
}

/// The current design or error 119.
pub fn current(ds: &DataSet) -> Result<SvySet, String> {
    SvySet::from_chars(ds).ok_or_else(|| rc_err(119, "data not set up for svy, use svyset"))
}

// ---------------------------------------------------------------------------
// svyset
// ---------------------------------------------------------------------------

/// Split `name(args)` style option text into the name and args of
/// replicate-weight options: `brrweight(varlist)`, `jkrweight(varlist, multiplier(#))`.
fn rep_opt(ds: &DataSet, arg: &str) -> Result<(Vec<String>, Vec<(String, Option<String>)>), String> {
    let (list, sub) = match find_top_level(arg, ',') {
        Some(i) => (&arg[..i], split_options(&arg[i + 1..])),
        None => (arg, Vec::new()),
    };
    Ok((varlist::expand(ds, list)?, sub))
}

fn sub_opt<'a>(opts: &'a [(String, Option<String>)], name: &str, min: usize) -> Option<&'a Option<String>> {
    opts.iter().find(|(n, _)| n.len() >= min && name.starts_with(n.to_lowercase().as_str())).map(|(_, a)| a)
}

fn parse_stage(ds: &DataSet, text: &str, first: bool, set: &mut SvySet) -> Result<Stage, String> {
    let (main, opts) = match find_top_level(text, ',') {
        Some(i) => (text[..i].trim().to_string(), split_options(&text[i + 1..])),
        None => (text.trim().to_string(), Vec::new()),
    };
    // weight clause
    let mut main = main;
    if let Some(a) = main.find('[') {
        let b = main[a..].find(']').map(|x| a + x).ok_or_else(|| rc_err(198, "invalid weight specification"))?;
        let inner = main[a + 1..b].trim().to_string();
        main = format!("{} {}", &main[..a], &main[b + 1..]);
        if !first {
            return Err(rc_err(198, "weights allowed only in the first stage"));
        }
        let (kind, var) = inner.split_once('=').ok_or_else(|| rc_err(198, "invalid weight specification"))?;
        let kind = match kind.trim().to_lowercase().as_str() {
            "pw" | "pweight" => "pweight",
            "iw" | "iweight" => "iweight",
            k => return Err(rc_err(101, format!("{k} not allowed"))),
        };
        let var = varlist::resolve_name(ds, var.trim())?;
        set.wtype = Some(kind.into());
        set.wvar = Some(var);
    }
    let mut stage = Stage::default();
    let su = main.trim();
    if !su.is_empty() && su != "_n" {
        let v = varlist::expand(ds, su)?;
        if v.len() != 1 {
            return Err(rc_err(198, "only one sampling unit variable allowed per stage"));
        }
        stage.su = Some(v[0].clone());
    }
    let opt = |name: &str, min: usize| opts.iter().find(|(n, _)| n.len() >= min && name.starts_with(n.to_lowercase().as_str())).map(|(_, a)| a.clone());
    if let Some(Some(s)) = opt("strata", 3) {
        stage.strata = Some(varlist::resolve_name(ds, s.trim())?);
    }
    if let Some(Some(f)) = opt("fpc", 3) {
        stage.fpc = Some(varlist::resolve_name(ds, f.trim())?);
    }
    if first {
        if let Some(Some(v)) = opt("vce", 3) {
            let v = v.trim().to_lowercase();
            set.vce = match v.as_str() {
                x if x.len() >= 3 && "linearized".starts_with(x) => "linearized",
                "brr" => "brr",
                x if x.len() >= 2 && "jackknife".starts_with(x) => "jackknife",
                x if x.len() >= 2 && "bootstrap".starts_with(x) => "bootstrap",
                "sdr" => "sdr",
                _ => return Err(rc_err(198, format!("vce({v}) not allowed"))),
            }
            .into();
        }
        if let Some(Some(s)) = opt("singleunit", 6) {
            let s = s.trim().to_lowercase();
            set.singleunit = match s.as_str() {
                x if "missing".starts_with(x) && x.len() >= 3 => "missing",
                x if "certainty".starts_with(x) && x.len() >= 3 => "certainty",
                x if "scaled".starts_with(x) && x.len() >= 3 => "scaled",
                x if "centered".starts_with(x) && x.len() >= 3 => "centered",
                _ => return Err(rc_err(198, format!("singleunit({s}) not allowed"))),
            }
            .into();
        }
        if let Some(Some(a)) = opt("brrweight", 3) {
            set.brrweight = rep_opt(ds, &a)?.0;
        }
        if let Some(Some(a)) = opt("jkrweight", 3) {
            let (l, sub) = rep_opt(ds, &a)?;
            let mult = match sub_opt(&sub, "multiplier", 4) {
                Some(Some(m)) => Some(m.trim().parse::<f64>().map_err(|_| rc_err(198, "multiplier() invalid"))?),
                _ => None,
            };
            let strata: Vec<f64> = match sub_opt(&sub, "stratum", 3) {
                Some(Some(s)) => s.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| rc_err(198, "stratum() invalid"))).collect::<Result<_, _>>()?,
                _ => vec![],
            };
            set.jk = (0..l.len()).map(|k| (strata.get(k).or(strata.last()).copied().unwrap_or(1.0), mult)).collect();
            set.jkrweight = l;
        }
        if let Some(Some(a)) = opt("bsrweight", 3) {
            set.bsrweight = rep_opt(ds, &a)?.0;
        }
        if let Some(Some(a)) = opt("sdrweight", 3) {
            set.sdrweight = rep_opt(ds, &a)?.0;
        }
        if let Some(Some(a)) = opt("fay", 3) {
            set.fay = a.trim().parse().map_err(|_| rc_err(198, "fay() invalid"))?;
            if !(0.0..2.0).contains(&set.fay) {
                return Err(rc_err(198, "fay() must be between 0 and 2"));
            }
        }
        if let Some(Some(a)) = opt("bsn", 3) {
            set.bsn = a.trim().parse().map_err(|_| rc_err(198, "bsn() invalid"))?;
        }
        if let Some(Some(a)) = opt("dof", 3) {
            set.dof = Some(a.trim().parse().map_err(|_| rc_err(198, "dof() invalid"))?);
        }
        if opt("mse", 3).is_some() {
            set.mse = true;
        }
        // the variance method follows the replicate weights when not given
        if opt("vce", 3).is_none() {
            if !set.brrweight.is_empty() {
                set.vce = "brr".into();
            } else if !set.jkrweight.is_empty() {
                set.vce = "jackknife".into();
            } else if !set.bsrweight.is_empty() {
                set.vce = "bootstrap".into();
            } else if !set.sdrweight.is_empty() {
                set.vce = "sdr".into();
            }
        }
    }
    Ok(stage)
}

/// Split at top-level `||`.
fn split_stages(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut quote = false;
    let mut cur = String::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match c {
            '"' => quote = !quote,
            '(' | '[' if !quote => depth += 1,
            ')' | ']' if !quote => depth -= 1,
            '|' if !quote && depth == 0 && b.get(i + 1) == Some(&'|') => {
                out.push(std::mem::take(&mut cur));
                i += 2;
                continue;
            }
            _ => {}
        }
        cur.push(c);
        i += 1;
    }
    out.push(cur);
    out
}

pub struct Svyset;

impl CommandHandler for Svyset {
    fn name(&self) -> &'static str {
        "svyset"
    }

    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let end = raw.find(|c: char| c.is_whitespace() || c == ',' || c == '[').unwrap_or(raw.len());
        let text = raw[end..].trim();
        if text.is_empty() {
            return match SvySet::from_chars(ds) {
                Some(s) => Ok(s.lines().join("\n")),
                None => Ok("no survey characteristics are set".into()),
            };
        }
        let stages = split_stages(text);
        if stages.len() == 1 {
            if let Some(i) = find_top_level(text, ',') {
                let o = split_options(&text[i + 1..]);
                if text[..i].trim().is_empty() && o.iter().any(|(n, _)| n == "clear") {
                    SvySet::clear_chars(ds);
                    return Ok(String::new());
                }
            }
        }
        let mut set = SvySet::default();
        let mut parsed = Vec::new();
        for (k, s) in stages.iter().enumerate() {
            parsed.push(parse_stage(ds, s, k == 0, &mut set)?);
        }
        // stages after one with observations as units are not allowed
        if parsed.len() > 1 && parsed[..parsed.len() - 1].iter().any(|s| s.su.is_none()) {
            return Err(rc_err(198, "_n is allowed only in the final stage"));
        }
        set.stages = parsed;
        for w in set.brrweight.iter().chain(&set.jkrweight).chain(&set.bsrweight).chain(&set.sdrweight) {
            if ds.get_var(w).is_none_or(|v| v.var_type.is_string()) {
                return Err(rc_err(109, format!("{w}: replicate weights must be numeric")));
            }
        }
        for v in set.variables() {
            if !ds.var_index.contains_key(&v) {
                return Err(rc_err(111, format!("variable {v} not found")));
            }
        }
        set.to_chars(ds);
        Ok(set.lines().join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Design information on a sample
// ---------------------------------------------------------------------------

/// Numeric key of a variable's values (strings hashed).
pub fn key_col(ds: &DataSet, var: &str) -> Vec<f64> {
    let Some(v) = ds.get_var(var) else { return vec![SYSMISS; ds.n_obs] };
    v.data
        .iter()
        .map(|x| match x {
            Value::Str(s) => s.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211)) as f64 / 1e3,
            Value::StrL(..) => SYSMISS,
            x => x.num(),
        })
        .collect()
}

/// Design columns over the rows of a sample.
#[derive(Debug, Clone)]
pub struct DesignData {
    /// dataset rows in the sample
    pub rows: Vec<usize>,
    /// sampling weight of each row (1 without weights)
    pub w: Vec<f64>,
    /// per stage: (strata key, unit key, fpc)
    pub stages: Vec<(Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>)>,
    pub singleunit: String,
}

impl DesignData {
    /// Mark of rows usable by the design: nonmissing design variables and
    /// positive weights.
    pub fn usable(ds: &DataSet, set: &SvySet) -> Vec<bool> {
        let mut mask = vec![true; ds.n_obs];
        util::markout(ds, &mut mask, &set.variables());
        if let Some(wv) = &set.wvar {
            let w = key_col(ds, wv);
            for i in 0..ds.n_obs {
                if mask[i] && !(w[i] > 0.0) && set.wtype.as_deref() != Some("iweight") {
                    mask[i] = false;
                }
            }
        }
        mask
    }

    pub fn new(ds: &DataSet, set: &SvySet, rows: &[usize]) -> DesignData {
        let w = match &set.wvar {
            Some(v) => {
                let c = key_col(ds, v);
                rows.iter().map(|&i| c[i]).collect()
            }
            None => vec![1.0; rows.len()],
        };
        let pick = |v: &Option<String>| v.as_ref().map(|v| {
            let c = key_col(ds, v);
            rows.iter().map(|&i| c[i]).collect::<Vec<f64>>()
        });
        let stages = set
            .stages
            .iter()
            .map(|s| (pick(&s.strata).unwrap_or_else(|| vec![1.0; rows.len()]), pick(&s.su), pick(&s.fpc)))
            .collect();
        DesignData { rows: rows.to_vec(), w, stages, singleunit: set.singleunit.clone() }
    }

    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Number of strata and of sampling units in the first stage.
    pub fn counts(&self) -> (usize, usize) {
        let (strata, su, _) = &self.stages[0];
        let mut s: BTreeMap<u64, std::collections::BTreeSet<u64>> = BTreeMap::new();
        for t in 0..self.n() {
            let u = su.as_ref().map_or(t as u64, |c| c[t].to_bits());
            s.entry(strata[t].to_bits()).or_default().insert(u);
        }
        (s.len(), s.values().map(|x| x.len()).sum())
    }

    /// Strata with a single sampling unit (first stage).
    pub fn singleton_strata(&self) -> usize {
        let (strata, su, _) = &self.stages[0];
        let mut s: BTreeMap<u64, std::collections::BTreeSet<u64>> = BTreeMap::new();
        for t in 0..self.n() {
            let u = su.as_ref().map_or(t as u64, |c| c[t].to_bits());
            s.entry(strata[t].to_bits()).or_default().insert(u);
        }
        s.values().filter(|x| x.len() == 1).count()
    }

    /// Design degrees of freedom: units minus strata in the first stage.
    pub fn df(&self) -> f64 {
        let (h, u) = self.counts();
        (u as f64 - h as f64).max(1.0)
    }

    /// Linearized variance of totals of the influence values `z[t]` (one
    /// vector per sample row, already weighted).  None when a stratum has
    /// a single unit and `singleunit(missing)` is in effect.
    pub fn variance(&self, z: &[Vec<f64>]) -> Option<Matrix> {
        let k = z.first().map_or(0, |r| r.len());
        let idx: Vec<usize> = (0..self.n()).collect();
        let mut single = false;
        let mut v = self.stage_var(0, &idx, z, k, &mut single)?;
        if self.singleunit == "scaled" {
            let (h, _) = self.counts();
            let hs = self.singleton_strata();
            if hs > 0 && hs < h {
                v = v.scale(h as f64 / (h - hs) as f64);
            }
        }
        Some(v)
    }

    fn stage_var(&self, s: usize, idx: &[usize], z: &[Vec<f64>], k: usize, single: &mut bool) -> Option<Matrix> {
        let (strata, su, fpc) = &self.stages[s];
        let mut v = Matrix::zeros(k, k);
        // stratum -> unit -> rows
        let mut groups: BTreeMap<u64, BTreeMap<u64, Vec<usize>>> = BTreeMap::new();
        for &t in idx {
            let u = su.as_ref().map_or(t as u64, |c| c[t].to_bits());
            groups.entry(strata[t].to_bits()).or_default().entry(u).or_default().push(t);
        }
        // unit totals, also the grand mean of units for singleunit(centered)
        let mut totals: Vec<(u64, Vec<Vec<f64>>, Vec<Vec<usize>>)> = Vec::new();
        let mut grand = vec![0.0; k];
        let mut nunits = 0usize;
        for (h, units) in &groups {
            let mut tot = Vec::new();
            let mut members = Vec::new();
            for rows in units.values() {
                let mut zt = vec![0.0; k];
                for &t in rows {
                    for j in 0..k {
                        zt[j] += z[t][j];
                    }
                }
                for j in 0..k {
                    grand[j] += zt[j];
                }
                nunits += 1;
                tot.push(zt);
                members.push(rows.clone());
            }
            totals.push((*h, tot, members));
        }
        if nunits > 0 {
            for g in grand.iter_mut() {
                *g /= nunits as f64;
            }
        }
        for (_, tot, members) in &totals {
            let nh = tot.len();
            let first = members[0][0];
            // sampling fraction from the fpc: a rate when <= 1, else a population count
            let f = match fpc {
                Some(c) => {
                    let x = c[first];
                    if is_missing(x) {
                        0.0
                    } else if x <= 1.0 {
                        x
                    } else {
                        nh as f64 / x
                    }
                }
                None => 0.0,
            };
            if nh == 1 {
                if f >= 1.0 {
                    continue;
                }
                match self.singleunit.as_str() {
                    "certainty" | "scaled" => {}
                    "centered" => {
                        let d: Vec<f64> = (0..k).map(|j| tot[0][j] - grand[j]).collect();
                        for a in 0..k {
                            for b in 0..k {
                                v[(a, b)] += (1.0 - f) * d[a] * d[b];
                            }
                        }
                    }
                    _ => {
                        *single = true;
                        return None;
                    }
                }
            } else {
                let mean: Vec<f64> = (0..k).map(|j| tot.iter().map(|t| t[j]).sum::<f64>() / nh as f64).collect();
                let c = (1.0 - f) * nh as f64 / (nh as f64 - 1.0);
                for t in tot {
                    for a in 0..k {
                        let da = t[a] - mean[a];
                        if da == 0.0 {
                            continue;
                        }
                        for b in 0..k {
                            v[(a, b)] += c * da * (t[b] - mean[b]);
                        }
                    }
                }
            }
            // later stages contribute when units were sampled without replacement
            if f > 0.0 && s + 1 < self.stages.len() && su.is_some() {
                for rows in members {
                    let sub = self.stage_var(s + 1, rows, z, k, single)?;
                    v = v.add(&sub.scale(f)).unwrap();
                }
            }
        }
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// svydescribe
// ---------------------------------------------------------------------------

pub struct Svydescribe;

impl CommandHandler for Svydescribe {
    fn name(&self) -> &'static str {
        "svydescribe"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let set = current(ds)?;
        let vars = if cmd.main.trim().is_empty() { Vec::new() } else { varlist::expand(ds, &cmd.main)? };
        let stage: usize = util::opt_num(cmd, "stage", 3)?.map(|x| x as usize).unwrap_or(1).clamp(1, set.stages.len());
        let mut mask = util::touse(ds, st, cmd)?;
        let usable = DesignData::usable(ds, &set);
        for i in 0..ds.n_obs {
            mask[i] &= usable[i];
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        // observations with all variables nonmissing
        let complete: Vec<bool> = rows
            .iter()
            .map(|&i| vars.iter().all(|v| ds.get_var(v).is_some_and(|x| !x.data[i].is_missing())))
            .collect();
        let s = &set.stages[stage - 1];
        let strata = s.strata.as_ref().map(|v| key_col(ds, v));
        let su = s.su.as_ref().map(|v| key_col(ds, v));
        // stratum -> unit -> (obs, complete obs)
        let mut g: BTreeMap<(i64, u64), (f64, BTreeMap<u64, (usize, usize)>)> = BTreeMap::new();
        for (t, &i) in rows.iter().enumerate() {
            let h = strata.as_ref().map_or(1.0, |c| c[i]);
            let key = ((h * 1e6).round() as i64, h.to_bits());
            let u = su.as_ref().map_or(i as u64, |c| c[i].to_bits());
            let e = g.entry(key).or_insert((h, BTreeMap::new())).1.entry(u).or_insert((0, 0));
            e.0 += 1;
            if complete[t] {
                e.1 += 1;
            }
        }
        let mut out = vec![format!("Survey: Describing stage {stage} sampling units"), String::new()];
        out.extend(set.lines());
        out.push(String::new());
        let unit_word = if s.su.is_some() { "units" } else { "obs" };
        out.push(format!("{:>36}Number of obs per unit", ""));
        out.push(format!(" Stratum   # {unit_word:<5}     # obs       Min        Mean       Max"));
        let rule = "-".repeat(64);
        out.push(rule.clone());
        let (mut tu, mut to, mut mn, mut mx) = (0usize, 0usize, usize::MAX, 0usize);
        let mut no_complete_units = 0usize;
        for (h, units) in g.values() {
            let counts: Vec<usize> = units.values().map(|x| if vars.is_empty() { x.0 } else { x.1 }).collect();
            let n: usize = counts.iter().sum();
            let nu = counts.iter().filter(|&&c| c > 0).count();
            no_complete_units += counts.len() - nu;
            let lo = counts.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
            let hi = counts.iter().copied().max().unwrap_or(0);
            tu += nu;
            to += n;
            mn = mn.min(lo);
            mx = mx.max(hi);
            let label = if strata.is_some() { fmt_g(*h, 8) } else { "1".into() };
            let mark = if nu == 1 { "*" } else { " " };
            out.push(format!("{:>8}{} {:>8}  {:>8}  {:>8}  {:>10.1}  {:>8}", label, mark, nu, util::commas(n), util::commas(lo), if nu > 0 { n as f64 / nu as f64 } else { 0.0 }, util::commas(hi)));
        }
        out.push(rule.clone());
        out.push(format!("{:>8}  {:>8}  {:>8}  {:>8}  {:>10.1}  {:>8}", g.len(), tu, util::commas(to), util::commas(if mn == usize::MAX { 0 } else { mn }), if tu > 0 { to as f64 / tu as f64 } else { 0.0 }, util::commas(mx)));
        if g.values().any(|(_, u)| u.len() == 1) {
            out.push("* Stratum with a single sampling unit.".into());
        }
        let _ = no_complete_units;
        let r = st.begin_r();
        r.set("N_strata", g.len() as f64);
        r.set("N_units", tu as f64);
        r.set("N", to as f64);
        r.set("stage", stage as f64);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// svymarkout
// ---------------------------------------------------------------------------

/// `svymarkout markvar`: set markvar to zero in observations with missing
/// survey design variables (or nonpositive sampling weights).
pub struct Svymarkout;

impl CommandHandler for Svymarkout {
    fn name(&self) -> &'static str {
        "svymarkout"
    }

    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = varlist::resolve_name(ds, cmd.main.trim())?;
        let set = current(ds)?;
        let mut mask = DesignData::usable(ds, &set);
        let reps: Vec<String> = set.brrweight.iter().chain(&set.jkrweight).chain(&set.bsrweight).chain(&set.sdrweight).cloned().collect();
        util::markout(ds, &mut mask, &reps);
        let var = ds.get_var_mut(&name).unwrap();
        if var.var_type.is_string() {
            return Err(rc_err(109, "type mismatch"));
        }
        let ty = var.var_type;
        for (i, ok) in mask.iter().enumerate() {
            if !ok {
                var.data[i] = util::typed_value(ty, 0.0);
            }
        }
        Ok(String::new())
    }
}
