//! Multiple imputation (`mi set`, `mi register`, `mi describe`, `mi passive`,
//! `mi xeq`, `mi extract`, `mi convert`, `mi estimate` with Rubin's rules)
//! and the hook for the `svy:` prefix and `estat` after it.
//!
//! The mi data live in the dataset in one of the styles wide, mlong or
//! flong, described by `_dta[_mi_*]` characteristics.  Internally they are
//! loaded into `MiData` (the original data plus the imputed columns of
//! every imputation), which makes converting styles, extracting a
//! completed dataset and adding imputations straightforward.

use std::collections::BTreeMap;

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, split_options};

use crate::interpreter::{split_verb, AdoInterpreter, Flow};
use crate::resample::run_cmd;

type Opts = Vec<(String, Option<String>)>;

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str()))
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opts.iter().find(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn intercept(me: &mut AdoInterpreter, verb_raw: &str, rest: &str, _text: &str) -> Result<Option<Flow>, String> {
    let v = verb_raw.to_lowercase();
    if me.engine.store.programs.contains_key(verb_raw) {
        return Ok(None);
    }
    if v == "svy" {
        let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "svy: a colon followed by a command is required"))?;
        let head = rest[..colon].to_string();
        let command = rest[colon + 1..].trim().to_string();
        let e = &mut me.engine;
        let out = ferrostat_cmd::svy_est::run(&mut e.dataset, &mut e.store, &head, &command)?;
        me.emit(&out);
        return Ok(Some(Flow::Next));
    }
    if v == "estat" {
        let r = rest.trim();
        let sub = r.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("");
        let opts = r[sub.len()..].trim().trim_start_matches(',');
        let e = &mut me.engine;
        if let Some(out) = ferrostat_cmd::svy_est::estat(&e.dataset, &mut e.store, sub, opts)? {
            me.emit(&out);
            return Ok(Some(Flow::Next));
        }
        return Ok(None);
    }
    if v == "mi" {
        let r = rest.trim();
        let end = r.find(|c: char| c.is_whitespace() || c == ',' || c == ':').unwrap_or(r.len());
        let sub = r[..end].to_lowercase();
        let args = r[end..].to_string();
        mi_command(me, &sub, &args)?;
        return Ok(Some(Flow::Next));
    }
    Ok(None)
}

fn mi_command(me: &mut AdoInterpreter, sub: &str, args: &str) -> Result<(), String> {
    if let Some(r) = crate::mi_more::command(me, sub, args) { return r; }
    let is = |full: &str, min: usize| sub.len() >= min && full.starts_with(sub);
    if is("set", 3) {
        return mi_set(me, args);
    }
    if is("register", 3) {
        return mi_register(me, args, true);
    }
    if is("unregister", 5) {
        return mi_register(me, args, false);
    }
    if is("describe", 1) {
        return mi_describe(me);
    }
    if is("query", 1) {
        let md = MiData::load(&me.engine.dataset)?;
        me.emit(&format!("data mi set {}, M = {}", md.style, md.m));
        let r = me.engine.store.begin_r();
        r.set("M", md.m as f64);
        r.set_str("style", md.style.clone());
        return Ok(());
    }
    if is("convert", 4) {
        let style = args.split(',').next().unwrap_or("").trim().to_lowercase();
        let style = norm_style(&style)?;
        let md = MiData::load(&me.engine.dataset)?;
        me.engine.dataset = md.store(&style)?;
        me.engine.store.data_changed = true;
        return Ok(());
    }
    if is("extract", 3) {
        let m: usize = args.split(',').next().unwrap_or("").trim().parse().map_err(|_| rc_err(198, "mi extract: imputation number required"))?;
        let md = MiData::load(&me.engine.dataset)?;
        if m > md.m {
            return Err(rc_err(198, format!("imputation {m} not found; M = {}", md.m)));
        }
        me.engine.dataset = md.extract(m);
        me.engine.store.data_changed = true;
        return Ok(());
    }
    if is("update", 3) {
        let md = MiData::load(&me.engine.dataset)?;
        me.engine.dataset = md.store(&md.style)?;
        return Ok(());
    }
    if is("unset", 5) {
        let md = MiData::load(&me.engine.dataset)?;
        me.engine.dataset = md.extract(0);
        me.engine.store.data_changed = true;
        return Ok(());
    }
    if is("passive", 3) {
        return mi_passive(me, args);
    }
    if is("xeq", 3) {
        return mi_xeq(me, args);
    }
    if is("estimate", 3) {
        return mi_estimate(me, args);
    }
    if is("impute", 3) {
        return crate::mi_impute::impute(me, args);
    }
    if is("rename", 3) {
        return mi_rename(me, args);
    }
    if is("reset", 5) {
        return mi_reset(me, args);
    }
    if is("test", 4) {
        return mi_test(me, args);
    }
    if is("varying", 3) {
        return mi_varying(me);
    }
    if is("misstable", 5) {
        let md = MiData::load(&me.engine.dataset)?;
        let saved = std::mem::replace(&mut me.engine.dataset, md.extract(0));
        let r = run_cmd(me, &format!("misstable {}", args.trim()), true);
        me.engine.dataset = saved;
        return r;
    }
    Err(rc_err(199, format!("mi {sub}: unrecognized subcommand")))
}

fn norm_style(s: &str) -> Result<String, String> {
    match s {
        "wide" | "mlong" | "flong" => Ok(s.to_string()),
        "flongsep" => Err(rc_err(198, "style flongsep is not supported; use flong")),
        _ => Err(rc_err(198, format!("style {s} invalid; use wide, mlong or flong"))),
    }
}

// ---------------------------------------------------------------------------
// The mi data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct MiData {
    pub style: String,
    pub m: usize,
    pub ivars: Vec<String>,
    pub pvars: Vec<String>,
    pub rvars: Vec<String>,
    /// the original data (m = 0) without system variables
    pub base: DataSet,
    /// imputed and passive columns of imputations 1..M
    pub imp: Vec<BTreeMap<String, Vec<f64>>>,
    /// flongsep data name
    pub name: String,
}

fn char_words(ds: &DataSet, name: &str) -> Vec<String> {
    ds.characteristics.get("_dta").and_then(|c| c.get(name)).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

pub(crate) fn take_rows(ds: &DataSet, idx: &[usize]) -> DataSet {
    let mut out = crate::resample::empty_shell(ds);
    for (v, src) in out.vars.iter_mut().zip(&ds.vars) {
        v.data = idx.iter().map(|&i| src.data[i].clone()).collect();
    }
    out.n_obs = idx.len();
    out
}

pub(crate) fn set_num_col(ds: &mut DataSet, name: &str, vals: &[f64]) -> Result<(), String> {
    if !ds.var_index.contains_key(name) {
        let mut v = Variable::new(name.to_string(), VarType::Double);
        v.data = vals.iter().map(|&x| util::typed_value(VarType::Double, x)).collect();
        while v.data.len() < ds.n_obs {
            v.push_missing();
        }
        return ds.add_variable(v);
    }
    let var = ds.get_var_mut(name).unwrap();
    let mut ty = var.var_type;
    if vals.iter().any(|&x| !is_missing(x) && !util::fits(ty, x)) {
        ty = if vals.iter().all(|&x| is_missing(x) || (x as f32) as f64 == x) && ty != VarType::Double { VarType::Float } else { VarType::Double };
        var.var_type = ty;
        if !var.format.starts_with("%9") && !var.format.contains('g') {
            var.format = "%9.0g".into();
        }
    }
    var.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
    Ok(())
}

pub(crate) fn col(ds: &DataSet, name: &str) -> Vec<f64> {
    match ds.get_var(name) {
        Some(v) => (0..ds.n_obs).map(|i| v.num(i)).collect(),
        None => vec![SYSMISS; ds.n_obs],
    }
}

const SYS: [&str; 3] = ["_mi_m", "_mi_id", "_mi_miss"];

impl MiData {
    pub fn is_set(ds: &DataSet) -> bool {
        ds.characteristics.get("_dta").is_some_and(|c| c.contains_key("_mi_style"))
    }

    pub fn load(ds: &DataSet) -> Result<MiData, String> {
        let c = ds.characteristics.get("_dta").ok_or_else(|| rc_err(119, "no; data not mi set"))?;
        let style = c.get("_mi_style").cloned().ok_or_else(|| rc_err(119, "no; data not mi set"))?;
        let m: usize = c.get("_mi_M").and_then(|x| x.trim().parse().ok()).unwrap_or(0);
        let ivars = char_words(ds, "_mi_ivars");
        let pvars = char_words(ds, "_mi_pvars");
        let rvars = char_words(ds, "_mi_rvars");
        let ipv: Vec<String> = ivars.iter().chain(&pvars).cloned().collect();
        let mut imp = vec![BTreeMap::new(); m];
        let base = match style.as_str() {
            "wide" => {
                let mut base = ds.clone();
                for k in 1..=m {
                    for v in &ipv {
                        let name = format!("_{k}_{v}");
                        imp[k - 1].insert(v.clone(), col(ds, &name));
                        let _ = base.drop_variable(&name);
                    }
                }
                let _ = base.drop_variable("_mi_miss");
                base
            }
            "mlong" | "flong" => {
                let mm = col(ds, "_mi_m");
                let id = col(ds, "_mi_id");
                let mut rows0: Vec<usize> = (0..ds.n_obs).filter(|&i| mm[i] == 0.0).collect();
                rows0.sort_by(|&a, &b| id[a].total_cmp(&id[b]));
                let mut base = take_rows(ds, &rows0);
                for s in SYS {
                    let _ = base.drop_variable(s);
                }
                let pos: BTreeMap<i64, usize> = rows0.iter().enumerate().map(|(t, &i)| (id[i] as i64, t)).collect();
                let n = rows0.len();
                for (k, slot) in imp.iter_mut().enumerate() {
                    for v in &ipv {
                        slot.insert(v.clone(), col(&base, v));
                    }
                    let _ = (k, n);
                }
                for i in 0..ds.n_obs {
                    let k = mm[i];
                    if k >= 1.0 && (k as usize) <= m {
                        if let Some(&t) = pos.get(&(id[i] as i64)) {
                            for v in &ipv {
                                let val = ds.get_var(v).map_or(SYSMISS, |x| x.num(i));
                                imp[k as usize - 1].get_mut(v).unwrap()[t] = val;
                            }
                        }
                    }
                }
                base
            }
            "flongsep" => crate::mi_more::flongsep_load(ds, &ipv, m, &mut imp)?,
            s => return Err(rc_err(198, format!("mi style {s} not supported"))),
        };
        let name = c.get("_mi_name").cloned().unwrap_or_default();
        let mut md = MiData { style, m, ivars, pvars, rvars, base, imp, name };
        md.base.characteristics.entry("_dta".into()).or_default().retain(|k, _| !k.starts_with("_mi_"));
        md.fill_complete();
        Ok(md)
    }

    /// Imputed columns hold the original values where those are observed.
    fn fill_complete(&mut self) {
        for v in self.ivars.iter().chain(&self.pvars) {
            let orig = col(&self.base, v);
            for slot in self.imp.iter_mut() {
                let c = slot.entry(v.clone()).or_insert_with(|| orig.clone());
                if c.len() != orig.len() {
                    *c = orig.clone();
                }
            }
        }
    }

    pub fn new(base: DataSet, style: &str) -> MiData {
        let mut base = base;
        if let Some(c) = base.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("_mi_"));
        }
        MiData { style: style.into(), m: 0, ivars: vec![], pvars: vec![], rvars: vec![], base, imp: vec![], name: String::new() }
    }

    /// Incomplete observations: any imputed variable missing in m = 0.
    pub fn incomplete(&self) -> Vec<bool> {
        let n = self.base.n_obs;
        let mut out = vec![false; n];
        for v in &self.ivars {
            let c = col(&self.base, v);
            for i in 0..n {
                if is_missing(c[i]) {
                    out[i] = true;
                }
            }
        }
        out
    }

    /// The completed dataset of imputation `m` (0 = original data).
    pub fn extract(&self, m: usize) -> DataSet {
        let mut ds = self.base.clone();
        if m >= 1 && m <= self.m {
            for (v, vals) in &self.imp[m - 1] {
                if ds.var_index.contains_key(v) {
                    let _ = set_num_col(&mut ds, v, vals);
                }
            }
        }
        if let Some(c) = ds.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("_mi_"));
        }
        ds
    }

    pub fn store(&self, style: &str) -> Result<DataSet, String> {
        let n = self.base.n_obs;
        let inc = self.incomplete();
        let miss: Vec<f64> = inc.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let ipv: Vec<String> = self.ivars.iter().chain(&self.pvars).cloned().collect();
        let mut ds = match style {
            "wide" => {
                let mut ds = self.base.clone();
                for v in &ipv {
                    for k in 1..=self.m {
                        let vals = self.imp[k - 1].get(v).cloned().unwrap_or_else(|| col(&self.base, v));
                        let name = format!("_{k}_{v}");
                        let _ = ds.drop_variable(&name);
                        let src = self.base.get_var(v).unwrap();
                        let mut nv = Variable::new(name.clone(), src.var_type);
                        nv.format = src.format.clone();
                        nv.label = src.label.clone();
                        nv.value_label_name = src.value_label_name.clone();
                        ds.add_variable(nv)?;
                        set_num_col(&mut ds, &name, &vals)?;
                    }
                }
                let _ = ds.drop_variable("_mi_miss");
                let mut mv = Variable::new("_mi_miss", VarType::Byte);
                mv.data = miss.iter().map(|&x| util::typed_value(VarType::Byte, x)).collect();
                ds.add_variable(mv)?;
                ds
            }
            "mlong" | "flong" => {
                let mut idx: Vec<usize> = (0..n).collect();
                let mut mcol: Vec<f64> = vec![0.0; n];
                let mut ids: Vec<f64> = (1..=n).map(|x| x as f64).collect();
                let mut mi_miss: Vec<f64> = miss.clone();
                let mut src_m: Vec<usize> = vec![0; n];
                for k in 1..=self.m {
                    for i in 0..n {
                        if style == "flong" || inc[i] {
                            idx.push(i);
                            mcol.push(k as f64);
                            ids.push((i + 1) as f64);
                            mi_miss.push(SYSMISS);
                            src_m.push(k);
                        }
                    }
                }
                let mut ds = take_rows(&self.base, &idx);
                for v in &ipv {
                    let vals: Vec<f64> = idx
                        .iter()
                        .zip(&src_m)
                        .map(|(&i, &k)| if k == 0 { self.base.get_var(v).map_or(SYSMISS, |x| x.num(i)) } else { self.imp[k - 1].get(v).map_or(SYSMISS, |c| c[i]) })
                        .collect();
                    set_num_col(&mut ds, v, &vals)?;
                }
                for (name, vals, ty, lab) in [
                    ("_mi_m", mcol, VarType::Int, "Imputation number"),
                    ("_mi_id", ids, VarType::Long, "Observation ID"),
                    ("_mi_miss", mi_miss, VarType::Byte, "Incomplete observation"),
                ] {
                    let _ = ds.drop_variable(name);
                    let mut v = Variable::new(name, ty);
                    v.label = lab.into();
                    v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
                    ds.add_variable(v)?;
                }
                ds.sort_by = vec!["_mi_m".into(), "_mi_id".into()];
                ds
            }
            "flongsep" => crate::mi_more::flongsep_store(self, &miss)?,
            s => return Err(rc_err(198, format!("style {s} invalid"))),
        };
        let c = ds.characteristics.entry("_dta".into()).or_default();
        c.retain(|k, _| !k.starts_with("_mi_"));
        if style == "flongsep" {
            c.insert("_mi_name".into(), self.name.clone());
        }
        c.insert("_mi_style".into(), style.into());
        c.insert("_mi_M".into(), self.m.to_string());
        c.insert("_mi_N".into(), n.to_string());
        c.insert("_mi_n".into(), inc.iter().filter(|&&b| b).count().to_string());
        c.insert("_mi_marker".into(), "_mi_ds_1".into());
        c.insert("_mi_update".into(), "0".into());
        for (k, l) in [("_mi_ivars", &self.ivars), ("_mi_pvars", &self.pvars), ("_mi_rvars", &self.rvars)] {
            if !l.is_empty() {
                c.insert(k.into(), l.join(" "));
            }
        }
        Ok(ds)
    }
}

fn load_or_err(me: &AdoInterpreter) -> Result<MiData, String> {
    if !MiData::is_set(&me.engine.dataset) {
        return Err(rc_err(119, "no; data not mi set"));
    }
    MiData::load(&me.engine.dataset)
}

pub(crate) fn save(me: &mut AdoInterpreter, md: &MiData) -> Result<(), String> {
    me.engine.dataset = md.store(&md.style)?;
    me.engine.store.data_changed = true;
    Ok(())
}

// ---------------------------------------------------------------------------
// mi set / register / describe
// ---------------------------------------------------------------------------

fn mi_set(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let a = args.trim().to_lowercase();
    let first = a.split_whitespace().next().unwrap_or("");
    if first == "m" || first.starts_with("m ") || a.starts_with("m ") || a.starts_with("m=") {
        // mi set M = #, += #, -= #
        let mut md = load_or_err(me)?;
        let expr = a.trim_start_matches('m').trim();
        let new_m = if let Some(x) = expr.strip_prefix("+=") {
            md.m + x.trim().parse::<usize>().map_err(|_| rc_err(198, "invalid M"))?
        } else if let Some(x) = expr.strip_prefix("-=") {
            md.m.saturating_sub(x.trim().parse::<usize>().map_err(|_| rc_err(198, "invalid M"))?)
        } else {
            expr.trim_start_matches('=').trim().parse::<usize>().map_err(|_| rc_err(198, "invalid M"))?
        };
        while md.imp.len() < new_m {
            let mut slot = BTreeMap::new();
            for v in md.ivars.iter().chain(&md.pvars) {
                slot.insert(v.clone(), col(&md.base, v));
            }
            md.imp.push(slot);
        }
        md.imp.truncate(new_m);
        md.m = new_m;
        return save(me, &md);
    }
    let style = norm_style(first)?;
    if MiData::is_set(&me.engine.dataset) {
        return Err(rc_err(119, "data already mi set"));
    }
    for s in SYS {
        if me.engine.dataset.var_index.contains_key(s) {
            return Err(rc_err(110, format!("variable {s} already defined")));
        }
    }
    let md = MiData::new(me.engine.dataset.clone(), &style);
    save(me, &md)
}

fn mi_register(me: &mut AdoInterpreter, args: &str, reg: bool) -> Result<(), String> {
    let mut md = load_or_err(me)?;
    let a = args.trim();
    let (kind, list) = if reg { a.split_once(char::is_whitespace).map(|(k, l)| (k.to_lowercase(), l.to_string())).ok_or_else(|| rc_err(198, "mi register imputed|passive|regular varlist"))? } else { (String::new(), a.to_string()) };
    let vars = varlist::expand(&md.base, &list)?;
    for v in &vars {
        if SYS.contains(&v.as_str()) {
            return Err(rc_err(198, format!("{v} is a system variable")));
        }
        md.ivars.retain(|x| x != v);
        md.pvars.retain(|x| x != v);
        md.rvars.retain(|x| x != v);
        for slot in md.imp.iter_mut() {
            slot.remove(v);
        }
    }
    if reg {
        let k = kind.as_str();
        let target = if k.len() >= 3 && "imputed".starts_with(k) {
            &mut md.ivars
        } else if k.len() >= 3 && "passive".starts_with(k) {
            &mut md.pvars
        } else if k.len() >= 3 && "regular".starts_with(k) {
            &mut md.rvars
        } else {
            return Err(rc_err(198, format!("mi register: {kind} invalid")));
        };
        for v in &vars {
            if !target.contains(v) {
                target.push(v.clone());
            }
        }
        md.fill_complete();
    }
    save(me, &md)
}

fn mi_describe(me: &mut AdoInterpreter) -> Result<(), String> {
    let md = load_or_err(me)?;
    let n = md.base.n_obs;
    let inc = md.incomplete();
    let ninc = inc.iter().filter(|&&b| b).count();
    let mut out = vec![String::new(), format!("  Style:  {}", md.style), String::new()];
    out.push(format!("  Obs.:   complete     {:>10}", util::commas(n - ninc)));
    out.push(format!("          incomplete   {:>10}  (M = {} imputation{})", util::commas(ninc), md.m, if md.m == 1 { "" } else { "s" }));
    out.push(format!("          {}", "-".repeat(21)));
    out.push(format!("          total        {:>10}", util::commas(n)));
    out.push(String::new());
    let miss_count = |v: &str| col(&md.base, v).iter().filter(|x| is_missing(**x)).count();
    let list = |vars: &[String], counts: bool| -> String {
        if counts {
            vars.iter().map(|v| format!("{v}({})", miss_count(v))).collect::<Vec<_>>().join(" ")
        } else {
            vars.join(" ")
        }
    };
    out.push(format!("  Vars.:  imputed:  {}{}", md.ivars.len(), if md.ivars.is_empty() { String::new() } else { format!("; {}", list(&md.ivars, true)) }));
    out.push(String::new());
    out.push(format!("          passive:  {}{}", md.pvars.len(), if md.pvars.is_empty() { String::new() } else { format!("; {}", list(&md.pvars, true)) }));
    out.push(String::new());
    out.push(format!("          regular:  {}{}", md.rvars.len(), if md.rvars.is_empty() { String::new() } else { format!("; {}", list(&md.rvars, false)) }));
    out.push(String::new());
    let sys: Vec<&str> = match md.style.as_str() {
        "wide" => vec!["_mi_miss"],
        _ => vec!["_mi_m", "_mi_id", "_mi_miss"],
    };
    out.push(format!("          system:   {}; {}", sys.len(), sys.join(" ")));
    out.push(String::new());
    let unreg: Vec<String> = md.base.vars.iter().map(|v| v.name.clone()).filter(|v| !md.ivars.contains(v) && !md.pvars.contains(v) && !md.rvars.contains(v)).collect();
    if unreg.is_empty() {
        out.push("         (there are no unregistered variables)".into());
    } else {
        out.push(format!("          (there {} {} unregistered variable{}{})", if unreg.len() == 1 { "is" } else { "are" }, unreg.len(), if unreg.len() == 1 { "" } else { "s" }, if unreg.len() <= 10 { format!("; {}", unreg.join(" ")) } else { String::new() }));
    }
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set("M", md.m as f64);
    r.set("N", n as f64);
    r.set("N_incomplete", ninc as f64);
    r.set("N_complete", (n - ninc) as f64);
    r.set_str("style", md.style.clone());
    r.set_str("ivars", md.ivars.join(" "));
    r.set_str("pvars", md.pvars.join(" "));
    r.set_str("rvars", md.rvars.join(" "));
    Ok(())
}

// ---------------------------------------------------------------------------
// mi passive / mi xeq
// ---------------------------------------------------------------------------

fn mi_passive(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let a = args.trim().trim_start_matches(':').trim();
    let a = if a.starts_with(',') || a.starts_with(':') { a } else { a };
    let colon = find_top_level(a, ':');
    let command = match colon {
        Some(i) => a[i + 1..].trim(),
        None => a,
    };
    let mut md = load_or_err(me)?;
    let (verb, rest) = split_verb(command);
    let vr = me.engine.resolve_verb(&verb.to_lowercase());
    if !matches!(vr.as_str(), "generate" | "replace" | "egen") {
        return Err(rc_err(198, "mi passive: only generate, replace and egen are allowed"));
    }
    // the variable being created / changed
    let r = rest.trim();
    let r = r.split_once(char::is_whitespace).filter(|(t, _)| VarType::from_type_str(t).is_some()).map(|x| x.1).unwrap_or(r);
    let name = r.split(|c: char| c == '=' || c.is_whitespace()).next().unwrap_or("").trim().to_string();
    if name.is_empty() {
        return Err(rc_err(198, "mi passive: variable name required"));
    }
    let saved = me.engine.dataset.clone();
    let saved_changed = me.engine.store.data_changed;
    let mut lines = Vec::new();
    let mut base_after: Option<DataSet> = None;
    for k in 0..=md.m {
        me.engine.dataset = md.extract(k);
        let res = run_cmd(me, command, false);
        if let Err(e) = res {
            me.engine.dataset = saved;
            me.engine.store.data_changed = saved_changed;
            return Err(e);
        }
        lines.push(format!("m={k}:"));
        if vr != "replace" {
            let nm = col(&me.engine.dataset, &name).iter().filter(|x| is_missing(**x)).count();
            if nm > 0 {
                lines.push(format!("({} missing value{} generated)", util::commas(nm), if nm == 1 { "" } else { "s" }));
            }
        }
        if k == 0 {
            base_after = Some(me.engine.dataset.clone());
        } else {
            let vals = col(&me.engine.dataset, &name);
            md.imp[k - 1].insert(name.clone(), vals);
        }
    }
    md.base = base_after.unwrap();
    if !md.pvars.contains(&name) && !md.ivars.contains(&name) {
        md.pvars.push(name.clone());
        md.rvars.retain(|x| *x != name);
    }
    save(me, &md)?;
    me.emit(&lines.join("\n"));
    Ok(())
}

pub(crate) fn parse_numlist_m(s: &str, m: usize) -> Result<Vec<usize>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok((0..=m).collect());
    }
    let mut out = Vec::new();
    for tok in s.replace(',', " ").split_whitespace() {
        if let Some((a, b)) = tok.split_once('/') {
            let a: usize = a.parse().map_err(|_| rc_err(121, "invalid numlist"))?;
            let b: usize = b.parse().map_err(|_| rc_err(121, "invalid numlist"))?;
            out.extend(a..=b);
        } else {
            out.push(tok.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
        }
    }
    if out.iter().any(|&k| k > m) {
        return Err(rc_err(125, format!("imputation numbers must be between 0 and {m}")));
    }
    Ok(out)
}

fn mi_xeq(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let colon = find_top_level(args, ':').ok_or_else(|| rc_err(198, "mi xeq: colon required"))?;
    let md = load_or_err(me)?;
    let list = parse_numlist_m(&args[..colon], md.m)?;
    let cmds: Vec<String> = args[colon + 1..].split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let saved = me.engine.dataset.clone();
    let saved_changed = me.engine.store.data_changed;
    let mut result = Ok(());
    for k in list {
        me.engine.dataset = md.extract(k);
        me.emit(&format!("\n{} data:", if k == 0 { "m=0".to_string() } else { format!("m={k}") }));
        for c in &cmds {
            me.emit(&format!("-> {c}"));
            if let Err(e) = run_cmd(me, c, true) {
                result = Err(e);
                break;
            }
        }
        if result.is_err() {
            break;
        }
    }
    me.engine.dataset = saved;
    me.engine.store.data_changed = saved_changed;
    result
}

// ---------------------------------------------------------------------------
// mi estimate
// ---------------------------------------------------------------------------

/// Results of combining estimates over imputations by Rubin's rules.
#[derive(Debug, Clone)]
pub struct Rubin {
    pub b: Vec<f64>,
    pub t: Matrix,
    pub w: Matrix,
    pub bm: Matrix,
    pub df: Vec<f64>,
    pub rvi: Vec<f64>,
    pub fmi: Vec<f64>,
    pub re: Vec<f64>,
}

/// Rubin's combination rules with the Barnard-Rubin small-sample degrees
/// of freedom when the complete-data df `dfc` is given.
pub fn rubin(bs: &[Vec<f64>], vs: &[Matrix], dfc: Option<f64>) -> Rubin {
    let m = bs.len();
    let k = bs[0].len();
    let mf = m as f64;
    let b: Vec<f64> = (0..k).map(|j| bs.iter().map(|x| x[j]).sum::<f64>() / mf).collect();
    let mut w = Matrix::zeros(k, k);
    for v in vs {
        w = w.add(v).unwrap();
    }
    let w = w.scale(1.0 / mf);
    let mut bm = Matrix::zeros(k, k);
    if m > 1 {
        for x in bs {
            for a in 0..k {
                for c in 0..k {
                    bm[(a, c)] += (x[a] - b[a]) * (x[c] - b[c]);
                }
            }
        }
        bm = bm.scale(1.0 / (mf - 1.0));
    }
    let t = w.add(&bm.scale(1.0 + 1.0 / mf)).unwrap();
    let mut df = vec![SYSMISS; k];
    let mut rvi = vec![0.0; k];
    let mut fmi = vec![0.0; k];
    let mut re = vec![1.0; k];
    for j in 0..k {
        let u = w[(j, j)];
        let bb = bm[(j, j)];
        let tt = t[(j, j)];
        if tt <= 0.0 {
            continue;
        }
        let r = if u > 0.0 { (1.0 + 1.0 / mf) * bb / u } else { SYSMISS };
        rvi[j] = r;
        let nu_m = if bb > 0.0 && m > 1 { (mf - 1.0) * (1.0 + 1.0 / r).powi(2) } else { f64::INFINITY };
        let nu = match dfc {
            Some(dc) => {
                let gamma = (1.0 + 1.0 / mf) * bb / tt;
                let nu_obs = dc * (dc + 1.0) / (dc + 3.0) * (1.0 - gamma);
                if nu_m.is_infinite() { nu_obs } else { 1.0 / (1.0 / nu_m + 1.0 / nu_obs) }
            }
            None => nu_m,
        };
        df[j] = if nu.is_finite() { nu } else { SYSMISS };
        let lam = if nu.is_finite() { (r + 2.0 / (nu + 3.0)) / (r + 1.0) } else { r / (r + 1.0) };
        fmi[j] = lam;
        re[j] = 1.0 / (1.0 + lam / mf);
    }
    Rubin { b, t, w, bm, df, rvi, fmi, re }
}

/// Model F test assuming equal fractions of missing information
/// (Li, Raghunathan and Rubin), with Reiter's small-sample df.
pub fn mi_ftest(r: &Rubin, idx: &[usize], m: usize, dfc: Option<f64>) -> Option<(f64, f64, f64, f64)> {
    let q = idx.len();
    if q == 0 {
        return None;
    }
    let sub = |x: &Matrix| {
        let mut s = Matrix::zeros(q, q);
        for (a, &i) in idx.iter().enumerate() {
            for (c, &j) in idx.iter().enumerate() {
                s[(a, c)] = x[(i, j)];
            }
        }
        s
    };
    let (wi, dropped) = sub(&r.w).sym_inverse(1e-12);
    if !dropped.is_empty() {
        return None;
    }
    let bsub = sub(&r.bm);
    let mf = m as f64;
    let qf = q as f64;
    let rbar = (1.0 + 1.0 / mf) * bsub.mul(&wi).unwrap().trace() / qf;
    let bq: Vec<f64> = idx.iter().map(|&i| r.b[i]).collect();
    let wald: f64 = (0..q).map(|a| (0..q).map(|c| bq[a] * wi[(a, c)] * bq[c]).sum::<f64>()).sum();
    let f = wald / (qf * (1.0 + rbar));
    let t = qf * (mf - 1.0);
    let mut nu2 = if rbar <= 0.0 {
        f64::INFINITY
    } else if t > 4.0 {
        4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / rbar).powi(2)
    } else {
        t * (1.0 + 1.0 / qf) * (1.0 + 1.0 / rbar).powi(2) / 2.0
    };
    if let Some(dc) = dfc {
        let vstar = dc * (dc + 1.0) / (dc + 3.0);
        if t > 4.0 && rbar > 0.0 {
            let a = rbar * t / (t - 2.0);
            let d1 = vstar - 4.0 * (1.0 + a);
            if d1 > 0.0 {
                let inner = 1.0 / d1 + (1.0 / (t - 4.0)) * (a * a * (vstar - 2.0 * (1.0 + a))) / ((1.0 + a).powi(2) * d1);
                nu2 = 4.0 + 1.0 / inner;
            }
        } else if !nu2.is_finite() {
            nu2 = vstar;
        }
    }
    let p = if nu2.is_finite() { dist::ftail(qf, nu2, f) } else { dist::chi2tail(qf, f * qf) };
    Some((f, qf, nu2, p))
}

fn mi_estimate(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    // `mi estimate using file`: estimates saved by `mi estimate, saving()`
    let pre = crate::mi_more::estimate_using(args)?;
    let (head, command) = match &pre {
        Some(p) => (p.opts.clone(), p.command.clone()),
        None => {
            let colon = find_top_level(args, ':').ok_or_else(|| rc_err(198, "mi estimate: colon followed by an estimation command required"))?;
            (args[..colon].trim().trim_start_matches(',').to_string(), args[colon + 1..].trim().to_string())
        }
    };
    let opts: Opts = split_options(&head);
    if command.is_empty() {
        return Err(rc_err(198, "estimation command required"));
    }
    let md = match &pre {
        Some(p) => MiData { m: p.ests.len(), ..MiData::new(DataSet::new(), "mlong") },
        None => load_or_err(me)?,
    };
    if md.m == 0 {
        return Err(rc_err(2000, "no imputations"));
    }
    let mut list: Vec<usize> = match arg(&opts, "imputations", 3) {
        Some(l) => parse_numlist_m(&l, md.m)?.into_iter().filter(|&k| k >= 1).collect(),
        None => (1..=md.m).collect(),
    };
    if let Some(n) = arg(&opts, "nimputations", 3) {
        let n: usize = n.trim().parse().map_err(|_| rc_err(198, "nimputations() invalid"))?;
        list.truncate(n);
    }
    let level = match arg(&opts, "level", 1) {
        Some(l) => l.trim().parse::<f64>().map_err(|_| rc_err(198, "level() invalid"))?,
        None => me.engine.store.level,
    };
    let dots = has(&opts, "dots", 3);
    let noisily = has(&opts, "noisily", 3);
    let saved = me.engine.dataset.clone();
    let saved_changed = me.engine.store.data_changed;
    let mut bs = Vec::new();
    let mut vs = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut first_e: Option<Results> = None;
    let mut ns = Vec::new();
    let mut marks = String::new();
    let mut all_e: Vec<Results> = Vec::new();
    for &k in &list {
        let e = if let Some(p) = &pre {
            p.ests[k - 1].clone()
        } else {
        me.engine.dataset = md.extract(k);
        me.engine.store.e = Results::new();
        let r = run_cmd(me, &command, noisily);
        if let Err(e) = r {
            me.engine.dataset = saved;
            me.engine.store.data_changed = saved_changed;
            return Err(rc_err(crate::interpreter::rc_of(&e), format!("{}\nan error occurred when mi estimate executed {} on m={k}", crate::interpreter::err_message(&e), split_verb(&command).0)));
        }
        me.engine.store.e.clone()
        };
        all_e.push(e.clone());
        let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "estimation command did not post e(b)"))?;
        let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "estimation command did not post e(V)"))?;
        if names.is_empty() {
            names = b.colnames.clone();
        } else if names != b.colnames {
            me.engine.dataset = saved;
            return Err(rc_err(459, "estimation sample varies between imputations; coefficients differ"));
        }
        let nk = e.get("N").unwrap_or(SYSMISS);
        if let Some(&n0) = ns.first() {
            if nk != n0 && !has(&opts, "esampvaryok", 4) {
                me.engine.dataset = saved;
                me.engine.store.data_changed = saved_changed;
                return Err(rc_err(459, format!("estimation sample varies between m={} and m={k}; check that all imputed variables are imputed in every imputation", list[0])));
            }
        }
        bs.push(b.data.clone());
        vs.push(v.clone());
        ns.push(nk);
        if first_e.is_none() {
            first_e = Some(e);
        }
        marks.push('.');
    }
    me.engine.dataset = saved;
    me.engine.store.data_changed = saved_changed;
    if let Some(sv) = arg(&opts, "saving", 3) {
        crate::mi_more::save_estimates(&sv, &list, &all_e, &command)?;
    }
    let e0 = first_e.unwrap();
    let dfc = e0.get("df_r");
    let mlen = bs.len();
    let r = rubin(&bs, &vs, dfc);
    let k = names.len();
    // coefficients fixed at zero (base levels, omitted) are left out of the summaries
    let active: Vec<usize> = (0..k).filter(|&j| r.t[(j, j)] > 0.0).collect();
    let test_idx: Vec<usize> = active.iter().copied().filter(|&j| !names[j].ends_with("_cons") && !names[j].starts_with("/")).collect();
    let ft = mi_ftest(&r, &test_idx, mlen, dfc);
    let dfs: Vec<f64> = active.iter().map(|&j| r.df[j]).filter(|d| !is_missing(*d)).collect();
    let (dmin, dmax, davg) = if dfs.is_empty() {
        (SYSMISS, SYSMISS, SYSMISS)
    } else {
        (dfs.iter().cloned().fold(f64::INFINITY, f64::min), dfs.iter().cloned().fold(0.0, f64::max), dfs.iter().sum::<f64>() / dfs.len() as f64)
    };
    // average relative variance increase: (1 + 1/M) tr(B W^-1) / k
    let avg_rvi = {
        let ka = active.len();
        let mut ws = Matrix::zeros(ka, ka);
        let mut bsub = Matrix::zeros(ka, ka);
        for (a, &i) in active.iter().enumerate() {
            for (c, &j) in active.iter().enumerate() {
                ws[(a, c)] = r.w[(i, j)];
                bsub[(a, c)] = r.bm[(i, j)];
            }
        }
        if ka == 0 { 0.0 } else { (1.0 + 1.0 / mlen as f64) * bsub.mul(&ws.sym_inverse(1e-12).0).unwrap().trace() / ka as f64 }
    };
    let max_fmi = active.iter().map(|&j| r.fmi[j]).fold(0.0, f64::max);
    let n = ns.iter().cloned().fold(f64::INFINITY, f64::min);

    // ---- output
    let mut out = Vec::new();
    if dots {
        out.push(format!("Imputations ({mlen}):"));
        let mut s = String::from("  ");
        for (i, c) in marks.chars().enumerate() {
            if (i + 1) % 10 == 0 {
                s.push_str(&(i + 1).to_string());
            } else {
                s.push(c);
            }
        }
        s.push_str(" done");
        out.push(s);
    }
    out.push(String::new());
    let title = e0.macros.get("title").cloned().unwrap_or_default();
    let mut rows: Vec<(String, String, String)> = vec![("Multiple-imputation estimates".into(), "Imputations".into(), mlen.to_string())];
    rows.push((title.clone(), "Number of obs".into(), util::commas(n as usize)));
    rows.push((String::new(), "Average RVI".into(), format!("{:.4}", avg_rvi)));
    rows.push((String::new(), "Largest FMI".into(), format!("{:.4}", max_fmi)));
    if let Some(dc) = dfc {
        rows.push((String::new(), "Complete DF".into(), fmt_g(dc, 9)));
    }
    let adj = if dfc.is_some() { "Small sample" } else { "Large sample" };
    let df2 = |x: f64| if is_missing(x) { ".".to_string() } else { format!("{:.2}", x) };
    rows.push((format!("DF adjustment:   {adj}"), "DF:     min".into(), df2(dmin)));
    rows.push((String::new(), "        avg".into(), df2(davg)));
    rows.push((String::new(), "        max".into(), df2(dmax)));
    if let Some((f, q, nu2, p)) = ft {
        let lab = if nu2.is_finite() { format!("F({:>4},{:>7.1})", q as usize, nu2) } else { format!("chi2({})", q as usize) };
        rows.push(("Model F test:       Equal FMI".into(), lab, format!("{:.2}", f)));
        let wvce = e0.macros.get("vce").map(|v| v.to_uppercase()).unwrap_or_else(|| "OIM".into());
        let wvce = match wvce.as_str() {
            "ROBUST" => "Robust".to_string(),
            "CLUSTER" => "Robust".to_string(),
            x => x.to_string(),
        };
        rows.push((format!("Within VCE type: {:>12}", wvce), "Prob > F".into(), format!("{:.4}", p)));
    }
    for (l, lab, v) in &rows {
        out.push(format!("{:<48}{:<18}= {:>10}", l, lab, v));
    }
    out.push(String::new());
    // coefficient table with per-coefficient degrees of freedom
    let depvar = e0.macros.get("depvar").cloned().unwrap_or_default();
    let lw = 12usize.max(names.iter().map(|n| n.rsplit(':').next().unwrap_or(n).len()).max().unwrap_or(0)).min(32);
    let bar = "-".repeat(lw + 66);
    out.push(bar.clone());
    out.push(format!("{:>lw$} | Coefficient  Std. err.      t    P>|t|     [{}% conf. interval]", ferrostat_cmd::summarize::abbrev(depvar.split_whitespace().next().unwrap_or(""), lw), fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
    let mut last_eq = String::new();
    let mut table = Matrix::zeros(9, k);
    for j in 0..k {
        let (eq, nm) = match names[j].split_once(':') {
            Some((a, b)) => (a.to_string(), b.to_string()),
            None => (String::new(), names[j].clone()),
        };
        if !eq.is_empty() && eq != "/" && eq != last_eq {
            if !last_eq.is_empty() {
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            }
            out.push(format!("{:<lw$} |", ferrostat_cmd::summarize::abbrev(&eq, lw)));
            last_eq = eq.clone();
        }
        let label = if eq == "/" { format!("/{nm}") } else { nm.clone() };
        let se = r.t[(j, j)].max(0.0).sqrt();
        if se == 0.0 {
            let is_base = nm.split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b')));
            if is_base {
                out.push(format!("{:>lw$} |          0  (base)", ferrostat_cmd::summarize::abbrev(&label, lw)));
            } else {
                out.push(format!("{:>lw$} |          0  (omitted)", ferrostat_cmd::summarize::abbrev(&label, lw)));
            }
            for i in 0..9 {
                table[(i, j)] = SYSMISS;
            }
            table[(0, j)] = 0.0;
            continue;
        }
        let df = r.df[j];
        let tstat = r.b[j] / se;
        let (p, crit) = if is_missing(df) {
            (2.0 * dist::normal(-tstat.abs()), dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0))
        } else {
            (2.0 * dist::ttail(df, tstat.abs()), dist::invttail(df, (1.0 - level / 100.0) / 2.0))
        };
        let lo = r.b[j] - crit * se;
        let hi = r.b[j] + crit * se;
        let g = |x: f64| fmt_g(x, 8);
        out.push(format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", ferrostat_cmd::summarize::abbrev(&label, lw), g(r.b[j]), g(se), tstat, p, g(lo), g(hi)));
        for (i, x) in [r.b[j], se, tstat, p, lo, hi, df, crit, 0.0].into_iter().enumerate() {
            table[(i, j)] = x;
        }
    }
    out.push(bar);
    me.emit(&out.join("\n"));

    // ---- e()
    let rowv = |vals: &[f64]| Matrix::row_vector(vals).with_names(vec!["y1".into()], names.clone());
    let sq = |m: &Matrix| m.clone().with_names(names.clone(), names.clone());
    let mut e = Results::new();
    for (key, val) in &e0.macros {
        if matches!(key.as_str(), "cmd" | "depvar" | "title" | "vce" | "vcetype" | "wtype" | "wexp" | "clustvar" | "eqnames") {
            e.set_str(key, val.clone());
        }
    }
    let cmd = e0.macros.get("cmd").cloned().unwrap_or_default();
    e.set_str("cmd_mi", cmd.clone());
    e.set_str("mi", "mi");
    e.set_str("prefix", "mi estimate");
    e.set_str("cmdline_mi", format!("mi estimate: {command}"));
    e.set_str("title_mi", "Multiple-imputation estimates");
    e.set_str("dfadjust_mi", adj);
    e.set_str("modeltest_mi", "Equal FMI");
    e.set_str("wvce_mi", e0.macros.get("vce").cloned().unwrap_or_default());
    e.set_str("properties", "b V");
    e.set_str("m_est_mi", list.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    e.set("M_mi", mlen as f64);
    e.set("N_mi", n);
    e.set("N", n);
    e.set("k_eq", e0.get("k_eq").unwrap_or(1.0));
    e.set("level", level);
    if let Some(dc) = dfc {
        e.set("df_r", dc);
    }
    e.set("rvi_avg_mi", avg_rvi);
    e.set("fmi_max_mi", max_fmi);
    e.set("df_avg_mi", davg);
    e.set("df_min_mi", dmin);
    e.set("df_max_mi", dmax);
    if let Some((f, q, nu2, p)) = ft {
        e.set("F_mi", f);
        e.set("df_m_mi", q);
        e.set("df_r_mi", if nu2.is_finite() { nu2 } else { SYSMISS });
        e.set("p_mi", p);
        let wi = {
            let qn = test_idx.len();
            let mut s = Matrix::zeros(qn, qn);
            for (a, &i) in test_idx.iter().enumerate() {
                for (c, &jj) in test_idx.iter().enumerate() {
                    s[(a, c)] = r.bm[(i, jj)];
                }
            }
            s
        };
        let _ = wi;
        e.set("rvi_avg_F_mi", avg_rvi);
    }
    e.set_mat("b", rowv(&r.b));
    e.set_mat("V", sq(&r.t));
    e.set_mat("b_mi", rowv(&r.b));
    e.set_mat("V_mi", sq(&r.t));
    e.set_mat("W_mi", sq(&r.w));
    e.set_mat("B_mi", sq(&r.bm));
    e.set_mat("df_mi", rowv(&r.df));
    e.set_mat("rvi_mi", rowv(&r.rvi));
    e.set_mat("fmi_mi", rowv(&r.fmi));
    e.set_mat("re_mi", rowv(&r.re));
    me.engine.store.e = e;
    let mut rt = Results::new();
    rt.set_mat("table", table.with_names(["b", "se", "t", "pvalue", "ll", "ul", "df", "crit", "eform"].iter().map(|s| s.to_string()).collect(), names.clone()));
    rt.set("level", level);
    me.engine.store.r = rt;
    Ok(())
}

// ---------------------------------------------------------------------------
// mi rename / reset / varying / test
// ---------------------------------------------------------------------------

fn mi_rename(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let w: Vec<&str> = args.split(',').next().unwrap_or("").split_whitespace().collect();
    if w.len() != 2 {
        return Err(rc_err(198, "mi rename oldname newname"));
    }
    let mut md = load_or_err(me)?;
    let old = varlist::resolve_name(&md.base, w[0])?;
    let new = w[1].to_string();
    if md.base.var_index.contains_key(&new) {
        return Err(rc_err(110, format!("variable {new} already defined")));
    }
    let idx = md.base.var_index[&old];
    md.base.vars[idx].name = new.clone();
    md.base.var_index.remove(&old);
    md.base.var_index.insert(new.clone(), idx);
    for l in [&mut md.ivars, &mut md.pvars, &mut md.rvars] {
        for v in l.iter_mut() {
            if *v == old {
                *v = new.clone();
            }
        }
    }
    for slot in md.imp.iter_mut() {
        if let Some(c) = slot.remove(&old) {
            slot.insert(new.clone(), c);
        }
    }
    save(me, &md)
}

/// `mi reset varlist [, m(numlist)]`: imputed values back to the original ones.
fn mi_reset(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (list, opts): (&str, Opts) = match find_top_level(args, ',') {
        Some(i) => (&args[..i], split_options(&args[i + 1..])),
        None => (args, Vec::new()),
    };
    let mut md = load_or_err(me)?;
    let vars = varlist::expand(&md.base, list)?;
    let ms = match arg(&opts, "m", 1) {
        Some(l) => parse_numlist_m(&l, md.m)?,
        None => (1..=md.m).collect(),
    };
    for v in &vars {
        if !md.ivars.contains(v) && !md.pvars.contains(v) {
            return Err(rc_err(198, format!("{v} is not an imputed or passive variable")));
        }
        let orig = col(&md.base, v);
        for &k in ms.iter().filter(|&&k| k >= 1) {
            md.imp[k - 1].insert(v.clone(), orig.clone());
        }
    }
    save(me, &md)
}

/// `mi varying`: imputed and passive variables whose values differ across m
/// in complete observations (they should not).
fn mi_varying(me: &mut AdoInterpreter) -> Result<(), String> {
    let md = load_or_err(me)?;
    let inc = md.incomplete();
    let mut bad = Vec::new();
    for v in md.ivars.iter().chain(&md.pvars) {
        let orig = col(&md.base, v);
        let varies = md.imp.iter().any(|slot| slot.get(v).is_some_and(|c| (0..orig.len()).any(|i| !inc[i] && !(c[i] == orig[i] || (is_missing(c[i]) && is_missing(orig[i]))))));
        if varies {
            bad.push(v.clone());
        }
    }
    let mut out = vec![String::new(), "           Possible problem   variable names".to_string(), format!("  {}", "-".repeat(60))];
    out.push(format!("  imputed or passive varying in complete obs: {}", if bad.is_empty() { "(none)".into() } else { bad.join(" ") }));
    out.push(format!("  {}", "-".repeat(60)));
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set_str("vars", bad.join(" "));
    Ok(())
}

/// `mi test coeflist`: test that coefficients are zero after `mi estimate`
/// assuming equal fractions of missing information.
fn mi_test(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let e = me.engine.store.e.clone();
    if e.macros.get("mi").map(String::as_str) != Some("mi") {
        return Err(rc_err(301, "last estimates not found; mi estimate required"));
    }
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let w = e.matrices.get("W_mi").ok_or_else(|| rc_err(301, "e(W_mi) not found"))?;
    let bm = e.matrices.get("B_mi").ok_or_else(|| rc_err(301, "e(B_mi) not found"))?;
    let m = e.get("M_mi").unwrap_or(1.0) as usize;
    let names = b.colnames.clone();
    let list = args.split(',').next().unwrap_or("").trim().to_string();
    let wanted: Vec<String> = if list.is_empty() {
        names.iter().filter(|n| !n.ends_with("_cons")).cloned().collect()
    } else {
        list.split_whitespace().map(String::from).collect()
    };
    let mut idx = Vec::new();
    for wn in &wanted {
        let j = names.iter().position(|n| n == wn || n.rsplit(':').next() == Some(wn.as_str())).ok_or_else(|| rc_err(111, format!("{wn} not found")))?;
        idx.push(j);
    }
    let k = names.len();
    let t = w.add(&bm.scale(1.0 + 1.0 / m as f64)).unwrap();
    let rr = Rubin { b: b.data.clone(), t, w: w.clone(), bm: bm.clone(), df: vec![SYSMISS; k], rvi: vec![0.0; k], fmi: vec![0.0; k], re: vec![1.0; k] };
    let (f, q, nu2, p) = mi_ftest(&rr, &idx, m, e.get("df_r")).ok_or_else(|| rc_err(504, "matrix has missing values or is singular"))?;
    let mut out = vec!["note: assuming equal fractions of missing information".to_string(), String::new()];
    for (i, &j) in idx.iter().enumerate() {
        out.push(format!(" ({:>2})  {} = 0", i + 1, names[j]));
    }
    out.push(String::new());
    out.push(format!("       F({:>3},{:>7.1}) = {:>8.2}", q as usize, nu2, f));
    out.push(format!("            Prob > F = {:>8.4}", p));
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set("F", f);
    r.set("df", q);
    r.set("df_r", nu2);
    r.set("p", p);
    Ok(())
}
