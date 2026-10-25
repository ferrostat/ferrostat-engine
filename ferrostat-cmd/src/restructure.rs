//! Changing the shape of the data: expand, fillin, contract, collapse,
//! reshape, merge, append, joinby, cross, xpose, sample.

use std::collections::HashMap;

use ferrostat_core::dta::DtaReader;
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, ValueLabel, VarType, Variable};
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::store::Store;
use crate::util::{self, WeightKind};
use crate::varlist;

fn plural(n: usize, w: &str) -> String {
    format!("{} {w}{}", crate::util::commas(n), if n == 1 { "" } else { "s" })
}

/// Build a dataset from selected observations (in the given order).
pub fn take_rows(ds: &DataSet, idx: &[usize]) -> DataSet {
    let mut out = ds.clone();
    for v in out.vars.iter_mut() {
        v.data = idx.iter().map(|&i| v.data[i].clone()).collect();
    }
    out.n_obs = idx.len();
    out.sort_by.clear();
    out
}

pub struct Expand;

impl CommandHandler for Expand {
    fn name(&self) -> &'static str {
        "expand"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let text = cmd.main.trim().trim_start_matches('=').trim();
        let e = ferrostat_parser::parser::parse_expression(text).map_err(|e| rc_err(198, e.message))?;
        let mask = util::touse(ds, st, cmd)?;
        let env = Env::with_store(ds, st);
        let n = ds.n_obs;
        let mut extra = Vec::new();
        for i in 0..n {
            if mask[i] {
                let k = env.eval(&e, i)?.num()?;
                if !is_missing(k) && k >= 2.0 {
                    for _ in 1..(k.floor() as usize) {
                        extra.push(i);
                    }
                }
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.extend(&extra);
        let added = extra.len();
        *ds = take_rows(ds, &idx);
        if let Some(g) = cmd.opt("generate", 3) {
            let mut vals = vec![0.0; n];
            vals.extend(std::iter::repeat(1.0).take(added));
            let mut v = Variable::new(g.trim(), VarType::Byte);
            v.data = vals.into_iter().map(|x| Value::Byte(x as i8)).collect();
            ds.add_variable(v)?;
        }
        st.data_changed = true;
        Ok(format!("({} created)", plural(added, "observation")))
    }
}

pub struct Fillin;

impl CommandHandler for Fillin {
    fn name(&self) -> &'static str {
        "fillin"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        if ds.var_index.contains_key("_fillin") {
            return Err(rc_err(110, "variable _fillin already defined"));
        }
        // distinct values of each variable
        let mut levels: Vec<Vec<Value>> = Vec::new();
        for v in &vars {
            let var = ds.get_var(v).unwrap();
            let mut vals: Vec<Value> = var.data.clone();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup_by(|a, b| a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal));
            levels.push(vals);
        }
        let existing: std::collections::HashSet<Vec<String>> = (0..ds.n_obs)
            .map(|i| vars.iter().map(|v| val_key(&ds.get_var(v).unwrap().data[i])).collect())
            .collect();
        let n0 = ds.n_obs;
        let mut fill = vec![0i8; n0];
        // cartesian product
        let mut counter = vec![0usize; vars.len()];
        loop {
            let combo: Vec<String> = counter.iter().enumerate().map(|(k, &j)| val_key(&levels[k][j])).collect();
            if !existing.contains(&combo) {
                ds.add_observation();
                let i = ds.n_obs - 1;
                for (k, v) in vars.iter().enumerate() {
                    let vi = ds.var_index[v];
                    ds.vars[vi].data[i] = levels[k][counter[k]].clone();
                }
                fill.push(1);
            }
            let mut d = vars.len();
            loop {
                if d == 0 {
                    break;
                }
                d -= 1;
                counter[d] += 1;
                if counter[d] < levels[d].len() {
                    break;
                }
                counter[d] = 0;
                if d == 0 {
                    d = usize::MAX;
                    break;
                }
            }
            if d == usize::MAX {
                break;
            }
        }
        let mut f = Variable::new("_fillin", VarType::Byte);
        f.data = fill.into_iter().map(Value::Byte).collect();
        ds.add_variable(f)?;
        ds.sort(&vars, true)?;
        Ok(String::new())
    }
}

fn val_key(v: &Value) -> String {
    match v {
        Value::Str(s) => format!("s{s}"),
        x => format!("n{:e}", x.num()),
    }
}

pub struct Contract;

impl CommandHandler for Contract {
    fn name(&self) -> &'static str {
        "contract"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "f")?;
        if cmd.has_opt("nomiss", 3) {
            util::markout(ds, &mut mask, &vars);
        }
        let (ids, reps) = util::group_ids(ds, &vars, &mask);
        let mut freq = vec![0.0; reps.len()];
        for i in 0..ds.n_obs {
            if ids[i] != usize::MAX {
                freq[ids[i]] += w.as_ref().map_or(1.0, |w| w.w[i]);
            }
        }
        let mut out = take_rows(ds, &reps);
        out.keep_variables(&vars);
        let fname = cmd.opt("freq", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| "_freq".into());
        let total: f64 = freq.iter().sum();
        let mut fv = Variable::new(&fname, VarType::Long);
        fv.format = "%8.0g".into();
        fv.data = freq.iter().map(|&f| Value::Long(f as i32)).collect();
        out.add_variable(fv)?;
        if let Some(p) = cmd.opt("percent", 2) {
            out.add_variable(util::new_double(p.trim(), freq.iter().map(|f| 100.0 * f / total).collect()))?;
        }
        if let Some(c) = cmd.opt("cfreq", 2) {
            let mut acc = 0.0;
            out.add_variable(util::new_double(c.trim(), freq.iter().map(|f| { acc += f; acc }).collect()))?;
        }
        if let Some(c) = cmd.opt("cpercent", 2) {
            let mut acc = 0.0;
            out.add_variable(util::new_double(c.trim(), freq.iter().map(|f| { acc += f; 100.0 * acc / total }).collect()))?;
        }
        out.sort_by = vars;
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- collapse -----------------------------------------------------------------

fn stat_value(stat: &str, vals: &mut Vec<(f64, f64)>, wk: Option<WeightKind>) -> f64 {
    // vals: (x, w) for nonmissing x
    let n = vals.len();
    if n == 0 {
        return match stat {
            "count" | "sum" | "rawsum" | "percent" => 0.0,
            _ => SYSMISS,
        };
    }
    let sw: f64 = vals.iter().map(|v| v.1).sum();
    let mean = vals.iter().map(|v| v.0 * v.1).sum::<f64>() / sw;
    let var = || {
        let ss: f64 = vals.iter().map(|v| v.1 * (v.0 - mean).powi(2)).sum();
        let denom = match wk {
            Some(WeightKind::F) => sw - 1.0,
            Some(_) => (n as f64 - 1.0) * sw / n as f64,
            None => n as f64 - 1.0,
        };
        if denom > 0.0 { ss / denom } else { SYSMISS }
    };
    match stat {
        "mean" => mean,
        "sum" => match wk {
            Some(WeightKind::A) => vals.iter().map(|v| v.0 * v.1).sum::<f64>() * n as f64 / sw,
            _ => vals.iter().map(|v| v.0 * v.1).sum(),
        },
        "rawsum" => vals.iter().map(|v| v.0).sum(),
        "count" => match wk {
            Some(WeightKind::F) | Some(WeightKind::I) => sw,
            _ => n as f64,
        },
        "max" => vals.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max),
        "min" => vals.iter().map(|v| v.0).fold(f64::INFINITY, f64::min),
        "sd" => {
            let v = var();
            if is_missing(v) { v } else { v.sqrt() }
        }
        "variance" | "var" => var(),
        "semean" => {
            let v = var();
            if is_missing(v) { v } else { (v / sw.max(1.0).min(if wk == Some(WeightKind::F) { sw } else { n as f64 })).sqrt() }
        }
        "sebinomial" => (mean * (1.0 - mean) / n as f64).sqrt(),
        "sepoisson" => (mean / n as f64).sqrt(),
        "first" | "firstnm" => vals[0].0,
        "last" | "lastnm" => vals[n - 1].0,
        "median" => util::pctile_weighted(vals, 50.0),
        "iqr" => {
            let a = util::pctile_weighted(vals, 75.0);
            let b = util::pctile_weighted(vals, 25.0);
            a - b
        }
        s if s.starts_with('p') && s[1..].parse::<f64>().is_ok() => util::pctile_weighted(vals, s[1..].parse().unwrap()),
        _ => SYSMISS,
    }
}

pub struct Collapse;

impl CommandHandler for Collapse {
    fn name(&self) -> &'static str {
        "collapse"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // targets: (stat, newname, source)
        let mut targets: Vec<(String, String, String)> = Vec::new();
        let mut stat = "mean".to_string();
        let words = split_top_level_words(&cmd.main.replace(" = ", "=").replace("= ", "=").replace(" =", "="));
        for w in words {
            if w.starts_with('(') && w.ends_with(')') {
                stat = w[1..w.len() - 1].trim().to_lowercase();
                if stat == "median" {
                    stat = "median".into();
                }
                continue;
            }
            if let Some((new, src)) = w.split_once('=') {
                let src = varlist::resolve_name(ds, src)?;
                targets.push((stat.clone(), new.to_string(), src));
            } else {
                for v in varlist::expand(ds, &w)? {
                    targets.push((stat.clone(), v.clone(), v));
                }
            }
        }
        if targets.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let by = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "faip")?;
        if cmd.has_opt("cw", 2) {
            let srcs: Vec<String> = targets.iter().map(|t| t.2.clone()).collect();
            util::markout(ds, &mut mask, &srcs);
        }
        let (ids, reps) = util::group_ids(ds, &by, &mask);
        let ng = reps.len();
        let mut out = take_rows(ds, &reps);
        out.keep_variables(&by);
        // keep by-variable order as given
        for (stat, new, src) in &targets {
            let col = util::col(ds, src)?;
            let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ng];
            let mut total_w = 0.0;
            for i in 0..ds.n_obs {
                if ids[i] == usize::MAX {
                    continue;
                }
                let wi = w.as_ref().map_or(1.0, |w| w.w[i]);
                if !is_missing(col[i]) || stat == "first" || stat == "last" {
                    if stat == "first" || stat == "last" {
                        buckets[ids[i]].push((col[i], wi));
                    } else {
                        buckets[ids[i]].push((col[i], wi));
                        total_w += wi;
                    }
                }
            }
            let vals: Vec<f64> = buckets
                .iter_mut()
                .map(|b| {
                    if stat == "percent" {
                        let s: f64 = b.iter().map(|v| v.1).sum();
                        100.0 * s / total_w
                    } else {
                        stat_value(stat, b, w.as_ref().map(|w| w.kind))
                    }
                })
                .collect();
            let mut v = util::new_double(new, vals);
            if stat == "count" {
                v.var_type = VarType::Long;
                v.data = v.data.iter().map(|x| util::typed_value(VarType::Long, x.num())).collect();
            } else {
                v.var_type = if ds.get_var(src).unwrap().var_type == VarType::Double || matches!(stat.as_str(), "sum" | "rawsum") { VarType::Double } else { VarType::Float };
                if v.var_type == VarType::Float {
                    v.data = v.data.iter().map(|x| crate::data::store_num(VarType::Float, x.num())).collect();
                }
            }
            v.label = format!("({stat}) {}", { let l = &ds.get_var(src).unwrap().label; if l.is_empty() { src.clone() } else { l.clone() } });
            if out.var_index.contains_key(new) {
                return Err(rc_err(110, format!("{new} already defined")));
            }
            out.add_variable(v)?;
        }
        out.sort_by = by;
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- reshape --------------------------------------------------------------------

pub struct Reshape;

impl CommandHandler for Reshape {
    fn name(&self) -> &'static str {
        "reshape"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut w = cmd.main.split_whitespace();
        let dir = w.next().ok_or_else(|| rc_err(198, "long or wide required"))?.to_lowercase();
        let stubs: Vec<String> = w.map(String::from).collect();
        let i_vars = varlist::expand(ds, cmd.opt("i", 1).ok_or_else(|| rc_err(198, "option i() required"))?)?;
        let jspec = cmd.opt("j", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| "_j".into());
        let mut jparts = jspec.split_whitespace();
        let jname = jparts.next().unwrap().to_string();
        let jvalues: Vec<String> = jparts.map(String::from).collect();
        let is_string = cmd.has_opt("string", 3);
        let out = match dir.as_str() {
            "long" => reshape_long(ds, &stubs, &i_vars, &jname, &jvalues, is_string)?,
            "wide" => reshape_wide(ds, &stubs, &i_vars, &jname)?,
            _ => return Err(rc_err(198, format!("reshape {dir} invalid"))),
        };
        let before = (ds.n_obs, ds.n_vars());
        *ds = out;
        st.data_changed = true;
        Ok(format!(
            "(j = {})\n\nData                               {:>8}   ->   {}\n{}\nNumber of observations     {:>16}   ->   {}\nNumber of variables        {:>16}   ->   {}",
            jname, if dir == "long" { "Wide" } else { "Long" }, if dir == "long" { "Long" } else { "Wide" },
            "-".repeat(70), before.0, ds.n_obs, before.1, ds.n_vars()
        ))
    }
}

fn stub_parts(stub: &str, name: &str) -> Option<String> {
    let (pre, post) = match stub.split_once('@') {
        Some((a, b)) => (a, b),
        None => (stub, ""),
    };
    if name.len() > pre.len() + post.len() && name.starts_with(pre) && name.ends_with(post) {
        Some(name[pre.len()..name.len() - post.len()].to_string())
    } else {
        None
    }
}

fn stub_name(stub: &str) -> String {
    stub.replace('@', "")
}

fn stub_var(stub: &str, j: &str) -> String {
    if stub.contains('@') { stub.replace('@', j) } else { format!("{stub}{j}") }
}

fn reshape_long(ds: &DataSet, stubs: &[String], i_vars: &[String], jname: &str, jvals: &[String], is_string: bool) -> Result<DataSet, String> {
    // collect j values
    let mut js: Vec<String> = Vec::new();
    let mut stub_cols: Vec<String> = Vec::new();
    for s in stubs {
        for v in &ds.vars {
            if let Some(j) = stub_parts(s, &v.name) {
                if !is_string && j.parse::<f64>().is_err() {
                    continue;
                }
                if !i_vars.contains(&v.name) {
                    stub_cols.push(v.name.clone());
                    if !js.contains(&j) {
                        js.push(j);
                    }
                }
            }
        }
    }
    if !jvals.is_empty() {
        js = jvals.to_vec();
    }
    if js.is_empty() {
        return Err(rc_err(111, format!("no xij variables found; variables {} not found", stubs.join(" "))));
    }
    if is_string {
        js.sort();
    } else {
        js.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()));
    }
    // unique i
    {
        let mut seen = std::collections::HashSet::new();
        for r in 0..ds.n_obs {
            let k: Vec<String> = i_vars.iter().map(|v| val_key(&ds.get_var(v).unwrap().data[r])).collect();
            if !seen.insert(k) {
                return Err(rc_err(9, format!("variable {} does not uniquely identify the observations", i_vars.join(" "))));
            }
        }
    }
    let others: Vec<&Variable> = ds.vars.iter().filter(|v| !stub_cols.contains(&v.name)).collect();
    let mut out = DataSet::new();
    let mut order: Vec<usize> = (0..ds.n_obs).collect();
    {
        let cols: Vec<&Variable> = i_vars.iter().map(|v| ds.get_var(v).unwrap()).collect();
        order.sort_by(|&a, &b| {
            for c in &cols {
                let o = c.data[a].partial_cmp(&c.data[b]).unwrap_or(std::cmp::Ordering::Equal);
                if o != std::cmp::Ordering::Equal {
                    return o;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    // i vars first, then j, then stubs, then the rest (Stata order: i j stubs others)
    let mut layout: Vec<Variable> = Vec::new();
    for iv in i_vars {
        let mut v = ds.get_var(iv).unwrap().clone();
        v.data.clear();
        layout.push(v);
    }
    let jvar = if is_string {
        let w = js.iter().map(|s| s.len()).max().unwrap_or(1);
        Variable::new(jname, VarType::Str(w))
    } else {
        let vals: Vec<f64> = js.iter().map(|s| s.parse().unwrap()).collect();
        Variable::new(jname, util::best_numeric_type(&vals))
    };
    layout.push(jvar);
    for s in stubs {
        let sn = stub_name(s);
        // type: widest of the source columns
        let mut ty: Option<VarType> = None;
        for j in &js {
            if let Some(v) = ds.get_var(&stub_var(s, j)) {
                ty = Some(match ty {
                    None => v.var_type,
                    Some(t) if t.is_string() || v.var_type.is_string() => {
                        let w = |t: VarType| if let VarType::Str(w) = t { w } else { 0 };
                        VarType::Str(w(t).max(w(v.var_type)).max(1))
                    }
                    Some(t) => util::promote(t, v.var_type),
                });
            }
        }
        let mut v = Variable::new(&sn, ty.unwrap_or(VarType::Float));
        if let Some(src) = js.iter().find_map(|j| ds.get_var(&stub_var(s, j))) {
            v.format = src.format.clone();
            v.value_label_name = src.value_label_name.clone();
        }
        layout.push(v);
    }
    for o in &others {
        if !i_vars.contains(&o.name) {
            let mut v = (*o).clone();
            v.data.clear();
            layout.push(v);
        }
    }
    for &r in &order {
        for j in &js {
            for v in layout.iter_mut() {
                let val = if i_vars.contains(&v.name) {
                    ds.get_var(&v.name).unwrap().data[r].clone()
                } else if v.name == jname {
                    if is_string { Value::Str(j.clone()) } else { util::typed_value(v.var_type, j.parse().unwrap()) }
                } else if let Some(s) = stubs.iter().find(|s| stub_name(s) == v.name) {
                    match ds.get_var(&stub_var(s, j)) {
                        Some(src) => src.data[r].clone(),
                        None => if v.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) },
                    }
                } else {
                    ds.get_var(&v.name).unwrap().data[r].clone()
                };
                v.data.push(val);
            }
        }
    }
    let n = order.len() * js.len();
    for v in layout {
        out.add_variable(v)?;
    }
    out.n_obs = n;
    out.value_labels = ds.value_labels.clone();
    out.datalabel = ds.datalabel.clone();
    let mut sb = i_vars.to_vec();
    sb.push(jname.to_string());
    out.sort_by = sb;
    Ok(out)
}

fn reshape_wide(ds: &DataSet, stubs: &[String], i_vars: &[String], jname: &str) -> Result<DataSet, String> {
    let jv = ds.get_var(jname).ok_or_else(|| rc_err(111, format!("variable {jname} not found")))?;
    let is_string = jv.var_type.is_string();
    let jtext = |r: usize| -> String {
        match &jv.data[r] {
            Value::Str(s) => s.clone(),
            x => crate::format::fmt_g(x.num(), 17),
        }
    };
    let mut js: Vec<(Value, String)> = (0..ds.n_obs).map(|r| (jv.data[r].clone(), jtext(r))).collect();
    js.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    js.dedup_by(|a, b| a.1 == b.1);
    let all = vec![true; ds.n_obs];
    let (ids, reps) = util::group_ids(ds, i_vars, &all);
    let stub_names: Vec<String> = stubs.iter().map(|s| stub_name(s)).collect();
    for s in &stub_names {
        if ds.get_var(s).is_none() {
            return Err(rc_err(111, format!("variable {s} not found")));
        }
    }
    let others: Vec<&Variable> = ds.vars.iter().filter(|v| !stub_names.contains(&v.name) && v.name != jname && !i_vars.contains(&v.name)).collect();
    // check constancy of others within i
    for o in &others {
        for r in 0..ds.n_obs {
            let rep = reps[ids[r]];
            if o.data[r].partial_cmp(&o.data[rep]) != Some(std::cmp::Ordering::Equal) && !(o.data[r].is_missing() && o.data[rep].is_missing()) {
                return Err(rc_err(9, format!("variable {} not constant within {}", o.name, i_vars.join(" "))));
            }
        }
    }
    // check uniqueness of (i, j)
    let mut seen = std::collections::HashSet::new();
    for r in 0..ds.n_obs {
        if !seen.insert((ids[r], jtext(r))) {
            return Err(rc_err(9, format!("values of variable {jname} not unique within {}", i_vars.join(" "))));
        }
    }
    let mut out = take_rows(ds, &reps);
    out.keep_variables(&i_vars.iter().cloned().chain(others.iter().map(|o| o.name.clone())).collect::<Vec<_>>());
    let ng = reps.len();
    for (s, sn) in stubs.iter().zip(&stub_names) {
        let src = ds.get_var(sn).unwrap();
        for (_, jt) in &js {
            let name = stub_var(s, &jt.replace('-', "_").replace('.', "_"));
            let mut v = Variable::new(&name, src.var_type);
            v.format = src.format.clone();
            v.value_label_name = src.value_label_name.clone();
            v.label = format!("{jt} {}", src.label).trim().to_string();
            v.data = vec![if src.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) }; ng];
            for r in 0..ds.n_obs {
                if &jtext(r) == jt {
                    v.data[ids[r]] = src.data[r].clone();
                }
            }
            out.add_variable(v)?;
        }
    }
    let _ = is_string;
    out.sort_by = i_vars.to_vec();
    Ok(out)
}

// ---- append / merge -------------------------------------------------------------

fn load_using(path: &str) -> Result<DataSet, String> {
    let p = if ferrostat_core::vfs::exists(path) || path.ends_with(".dta") { path.to_string() } else { format!("{path}.dta") };
    DtaReader::open(&p).map_err(|e| rc_err(601, format!("file {p} not found or not a Stata dataset ({e})")))
}

/// Append `other` to `ds`, harmonising types. Returns number of added obs.
pub fn append_ds(ds: &mut DataSet, mut other: DataSet, force: bool, keep: Option<&[String]>) -> Result<usize, String> {
    if let Some(k) = keep {
        other.keep_variables(k);
    }
    let n0 = ds.n_obs;
    for ov in &other.vars {
        match ds.var_index.get(&ov.name) {
            Some(&i) => {
                let mv = &mut ds.vars[i];
                if mv.var_type.is_string() != ov.var_type.is_string() {
                    if !force {
                        return Err(rc_err(106, format!("variable {} is {} in master but {} in using data", ov.name, if mv.var_type.is_string() { "str" } else { "numeric" }, if ov.var_type.is_string() { "str" } else { "numeric" })));
                    }
                } else if mv.var_type.is_numeric() {
                    mv.var_type = util::promote(mv.var_type, ov.var_type);
                } else if let (VarType::Str(a), VarType::Str(b)) = (mv.var_type, ov.var_type) {
                    mv.var_type = VarType::Str(a.max(b));
                } else {
                    mv.var_type = VarType::StrL;
                }
            }
            None => {
                let mut v = ov.clone();
                v.data = vec![if v.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) }; n0];
                ds.add_variable(v)?;
            }
        }
    }
    for v in ds.vars.iter_mut() {
        match other.get_var(&v.name) {
            Some(ov) if ov.var_type.is_string() == v.var_type.is_string() => v.data.extend(ov.data.iter().cloned()),
            _ => {
                let fill = if v.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) };
                v.data.extend(std::iter::repeat(fill).take(other.n_obs));
            }
        }
    }
    for (k, vl) in other.value_labels {
        ds.value_labels.entry(k).or_insert(vl);
    }
    ds.n_obs += other.n_obs;
    ds.sort_by.clear();
    Ok(other.n_obs)
}

pub struct Append;

impl CommandHandler for Append {
    fn name(&self) -> &'static str {
        "append"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let files = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let gen_opt = cmd.opt("generate", 3).map(|s| s.trim().to_string());
        let keep = match cmd.opt("keep", 4) {
            Some(k) => Some(k.split_whitespace().map(String::from).collect::<Vec<_>>()),
            None => None,
        };
        let mut source = vec![0.0; ds.n_obs];
        for (k, f) in split_top_level_words(&files).iter().enumerate() {
            let other = load_using(&util::unquote(f))?;
            let added = append_ds(ds, other, cmd.has_opt("force", 5), keep.as_deref())?;
            source.extend(std::iter::repeat((k + 1) as f64).take(added));
        }
        if let Some(g) = gen_opt {
            let mut v = Variable::new(&g, VarType::Byte);
            v.data = source.into_iter().map(|x| Value::Byte(x as i8)).collect();
            ds.add_variable(v)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

pub struct Merge;

impl CommandHandler for Merge {
    fn name(&self) -> &'static str {
        "merge"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (kind, keytext) = main.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let kind = kind.replace(' ', "").to_lowercase();
        if !matches!(kind.as_str(), "1:1" | "m:1" | "1:m" | "m:m") {
            return Err(rc_err(198, format!("{kind}: invalid merge type; use 1:1, m:1, 1:m or m:m")));
        }
        let keys: Vec<String> = if keytext.trim() == "_n" { Vec::new() } else { varlist::expand(ds, keytext)? };
        let path = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let mut using = load_using(&path)?;
        if let Some(ku) = cmd.opt("keepusing", 5) {
            let mut keepv = varlist::expand(&using, ku)?;
            keepv.extend(keys.iter().cloned());
            using.keep_variables(&keepv);
        }
        for k in &keys {
            let u = using.get_var(k).ok_or_else(|| rc_err(111, format!("variable {k} not found in using data")))?;
            if u.var_type.is_string() != ds.get_var(k).unwrap().var_type.is_string() {
                return Err(rc_err(106, format!("key variable {k} is str in one dataset and numeric in the other")));
            }
        }
        let gen_opt = if cmd.has_opt("nogenerate", 5) { None } else { Some(cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| "_merge".into())) };
        if let Some(g) = &gen_opt {
            if ds.var_index.contains_key(g) {
                return Err(rc_err(110, format!("variable {g} already defined")));
            }
        }
        let keyf = |d: &DataSet, i: usize| -> Vec<String> { keys.iter().map(|k| val_key(&d.get_var(k).unwrap().data[i])).collect() };
        // uniqueness checks
        let unique = |d: &DataSet, who: &str| -> Result<(), String> {
            let mut s = std::collections::HashSet::new();
            for i in 0..d.n_obs {
                if !s.insert(keyf(d, i)) {
                    return Err(rc_err(459, format!("variable{} {} do{} not uniquely identify observations in the {} data", if keys.len() > 1 { "s" } else { "" }, keys.join(" "), if keys.len() > 1 { "" } else { "es" }, who)));
                }
            }
            Ok(())
        };
        if kind.starts_with('1') {
            unique(ds, "master")?;
        }
        if kind.ends_with('1') {
            unique(&using, "using")?;
        }
        // pairs (master idx or None, using idx or None)
        let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
        if keys.is_empty() {
            for t in 0..ds.n_obs.max(using.n_obs) {
                pairs.push(((t < ds.n_obs).then_some(t), (t < using.n_obs).then_some(t)));
            }
        } else if kind == "m:m" {
            let mut groups_u: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            for j in 0..using.n_obs {
                groups_u.entry(keyf(&using, j)).or_default().push(j);
            }
            let mut groups_m: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            let mut order_m: Vec<Vec<String>> = Vec::new();
            for i in 0..ds.n_obs {
                let k = keyf(ds, i);
                if !groups_m.contains_key(&k) {
                    order_m.push(k.clone());
                }
                groups_m.entry(k).or_default().push(i);
            }
            for k in &order_m {
                let m = &groups_m[k];
                let u = groups_u.remove(k).unwrap_or_default();
                if u.is_empty() {
                    pairs.extend(m.iter().map(|&i| (Some(i), None)));
                    continue;
                }
                for t in 0..m.len().max(u.len()) {
                    pairs.push((Some(m[t.min(m.len() - 1)]), Some(u[t.min(u.len() - 1)])));
                }
            }
            let mut rest: Vec<(Vec<String>, Vec<usize>)> = groups_u.into_iter().collect();
            rest.sort_by_key(|(_, v)| v[0]);
            for (_, u) in rest {
                pairs.extend(u.into_iter().map(|j| (None, Some(j))));
            }
        } else {
            let mut umap: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            for j in 0..using.n_obs {
                umap.entry(keyf(&using, j)).or_default().push(j);
            }
            let mut matched_u = vec![false; using.n_obs];
            for i in 0..ds.n_obs {
                match umap.get(&keyf(ds, i)) {
                    Some(us) => {
                        for &j in us {
                            pairs.push((Some(i), Some(j)));
                            matched_u[j] = true;
                        }
                    }
                    None => pairs.push((Some(i), None)),
                }
            }
            for j in 0..using.n_obs {
                if !matched_u[j] {
                    pairs.push((None, Some(j)));
                }
            }
        }
        // build result
        let update = cmd.has_opt("update", 3);
        let replace = cmd.has_opt("replace", 3);
        let mut out = DataSet::new();
        out.datalabel = ds.datalabel.clone();
        out.value_labels = ds.value_labels.clone();
        for (k, vl) in &using.value_labels {
            out.value_labels.entry(k.clone()).or_insert(vl.clone());
        }
        let mut result_code = Vec::with_capacity(pairs.len());
        let mut cols: Vec<Variable> = ds.vars.iter().map(|v| { let mut c = v.clone(); c.data.clear(); c }).collect();
        for uv in &using.vars {
            if !ds.var_index.contains_key(&uv.name) {
                let mut c = uv.clone();
                c.data.clear();
                cols.push(c);
            } else if let Some(c) = cols.iter_mut().find(|c| c.name == uv.name) {
                if c.var_type.is_numeric() && uv.var_type.is_numeric() {
                    c.var_type = util::promote(c.var_type, uv.var_type);
                }
            }
        }
        for &(m, u) in &pairs {
            let mut code = match (m, u) {
                (Some(_), Some(_)) => 3.0,
                (Some(_), None) => 1.0,
                _ => 2.0,
            };
            let mut nonmissing_conflict = false;
            let mut updated = false;
            for c in cols.iter_mut() {
                let mv = m.and_then(|i| ds.get_var(&c.name).map(|v| v.data[i].clone()));
                let uv = u.and_then(|j| using.get_var(&c.name).map(|v| v.data[j].clone()));
                let is_key = keys.contains(&c.name);
                let val = match (mv, uv) {
                    (Some(a), Some(b)) if !is_key && (update || replace) => {
                        let a_miss = a.is_missing() || a.as_string() == Some("");
                        let b_miss = b.is_missing() || b.as_string() == Some("");
                        if a_miss && !b_miss {
                            updated = true;
                            b
                        } else if replace && !b_miss {
                            if a.partial_cmp(&b) != Some(std::cmp::Ordering::Equal) {
                                nonmissing_conflict = true;
                            }
                            b
                        } else {
                            if !b_miss && a.partial_cmp(&b) != Some(std::cmp::Ordering::Equal) {
                                nonmissing_conflict = true;
                            }
                            a
                        }
                    }
                    (Some(a), _) => a,
                    (None, Some(b)) => b,
                    (None, None) => if c.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) },
                };
                c.data.push(val);
            }
            if code == 3.0 && (update || replace) {
                if nonmissing_conflict {
                    code = 5.0;
                } else if updated {
                    code = 4.0;
                }
            }
            result_code.push(code);
        }
        let n = pairs.len();
        for c in cols {
            out.add_variable(c)?;
        }
        out.n_obs = n;
        // keep()
        let keep_codes: Option<Vec<f64>> = cmd.opt("keep", 4).map(|k| parse_results(k));
        let assert_codes: Option<Vec<f64>> = cmd.opt("assert", 3).map(|k| parse_results(k));
        if let Some(a) = &assert_codes {
            if result_code.iter().any(|c| !a.contains(c) && !(a.contains(&3.0) && *c >= 3.0)) {
                return Err(rc_err(9, "merge:  after merge, not all observations from master or matched\n(merge assertion is false)"));
            }
        }
        let mut gv = Variable::new(gen_opt.clone().unwrap_or_else(|| "_merge".into()), VarType::Byte);
        gv.data = result_code.iter().map(|&c| Value::Byte(c as i8)).collect();
        gv.value_label_name = "_merge".into();
        let mut vl = ValueLabel::new("_merge");
        vl.add(1.0, "Master only (1)");
        vl.add(2.0, "Using only (2)");
        vl.add(3.0, "Matched (3)");
        vl.add(4.0, "Missing updated (4)");
        vl.add(5.0, "Nonmissing conflict (5)");
        let counts = |c: f64| result_code.iter().filter(|&&x| x == c).count();
        let report = format!(
            "    Result                      Number of obs\n    -----------------------------------------\n    Not matched                    {:>10}\n        from master                {:>10}  (_merge==1)\n        from using                 {:>10}  (_merge==2)\n\n    Matched                        {:>10}  (_merge==3)\n    -----------------------------------------",
            counts(1.0) + counts(2.0), counts(1.0), counts(2.0), counts(3.0) + counts(4.0) + counts(5.0)
        );
        if gen_opt.is_some() {
            out.value_labels.insert("_merge".into(), vl);
            out.add_variable(gv)?;
        }
        if let Some(k) = keep_codes {
            let mask: Vec<bool> = result_code.iter().map(|c| k.contains(c) || (k.contains(&3.0) && *c >= 3.0)).collect();
            out.keep_if(&mask)?;
        }
        if !keys.is_empty() {
            out.sort(&keys, true)?;
        }
        *ds = out;
        st.data_changed = true;
        Ok(if cmd.has_opt("noreport", 5) { String::new() } else { report })
    }
}

fn parse_results(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|w| match w.to_lowercase().as_str() {
            "1" | "master" | "mas" | "master_only" => Some(1.0),
            "2" | "using" | "us" | "usi" => Some(2.0),
            "3" | "match" | "mat" | "matched" => Some(3.0),
            "4" | "match_update" => Some(4.0),
            "5" | "match_conflict" => Some(5.0),
            _ => None,
        })
        .collect()
}

pub struct Joinby;

impl CommandHandler for Joinby {
    fn name(&self) -> &'static str {
        "joinby"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let keys = if cmd.main.trim().is_empty() {
            let u = load_using(cmd.using.as_deref().unwrap_or(""))?;
            ds.varlist().into_iter().filter(|v| u.var_index.contains_key(v)).collect()
        } else {
            varlist::expand(ds, &cmd.main)?
        };
        let using = load_using(cmd.using.as_deref().ok_or_else(|| rc_err(100, "using required"))?)?;
        let keyf = |d: &DataSet, i: usize| -> Vec<String> { keys.iter().map(|k| val_key(&d.get_var(k).unwrap().data[i])).collect() };
        let mut umap: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for j in 0..using.n_obs {
            umap.entry(keyf(&using, j)).or_default().push(j);
        }
        let mut out_cols: Vec<Variable> = ds.vars.iter().map(|v| { let mut c = v.clone(); c.data.clear(); c }).collect();
        for uv in &using.vars {
            if !ds.var_index.contains_key(&uv.name) {
                let mut c = uv.clone();
                c.data.clear();
                out_cols.push(c);
            }
        }
        let mut n = 0;
        for i in 0..ds.n_obs {
            if let Some(us) = umap.get(&keyf(ds, i)) {
                for &j in us {
                    for c in out_cols.iter_mut() {
                        let v = match ds.get_var(&c.name) {
                            Some(m) => m.data[i].clone(),
                            None => using.get_var(&c.name).unwrap().data[j].clone(),
                        };
                        c.data.push(v);
                    }
                    n += 1;
                }
            }
        }
        let mut out = DataSet::new();
        for c in out_cols {
            out.add_variable(c)?;
        }
        out.n_obs = n;
        out.value_labels = ds.value_labels.clone();
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

pub struct Cross;

impl CommandHandler for Cross {
    fn name(&self) -> &'static str {
        "cross"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let using = load_using(cmd.using.as_deref().ok_or_else(|| rc_err(100, "using required"))?)?;
        for v in &using.vars {
            if ds.var_index.contains_key(&v.name) {
                return Err(rc_err(110, format!("variable {} in both datasets", v.name)));
            }
        }
        let mut out = DataSet::new();
        for v in &ds.vars {
            let mut c = v.clone();
            c.data = (0..ds.n_obs).flat_map(|i| std::iter::repeat(v.data[i].clone()).take(using.n_obs)).collect();
            out.add_variable(c)?;
        }
        for v in &using.vars {
            let mut c = v.clone();
            c.data = (0..ds.n_obs).flat_map(|_| v.data.iter().cloned()).collect();
            out.add_variable(c)?;
        }
        out.n_obs = ds.n_obs * using.n_obs;
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

pub struct Xpose;

impl CommandHandler for Xpose {
    fn name(&self) -> &'static str {
        "xpose"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !cmd.has_opt("clear", 5) {
            return Err(rc_err(198, "option clear required"));
        }
        let numeric: Vec<&Variable> = ds.vars.iter().filter(|v| v.var_type.is_numeric()).collect();
        let mut out = DataSet::new();
        for i in 0..ds.n_obs {
            out.add_variable(util::new_double(&format!("v{}", i + 1), numeric.iter().map(|v| v.num(i)).collect()))?;
        }
        out.n_obs = numeric.len();
        if cmd.has_opt("varname", 4) {
            let mut v = Variable::new("_varname", VarType::Str(32));
            v.data = numeric.iter().map(|x| Value::Str(x.name.clone())).collect();
            out.add_variable(v)?;
        }
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

pub struct Sample;

impl CommandHandler for Sample {
    fn name(&self) -> &'static str {
        "sample"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let amount: f64 = cmd.main.trim().parse().map_err(|_| rc_err(198, "invalid number"))?;
        let count = cmd.has_opt("count", 3);
        let mask = util::touse(ds, st, cmd)?;
        let by = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let (ids, reps) = util::group_ids(ds, &by, &mask);
        let mut keep = vec![false; ds.n_obs];
        for (i, k) in keep.iter_mut().enumerate() {
            if !mask[i] {
                *k = true;
            }
        }
        for g in 0..reps.len() {
            let mut members: Vec<usize> = (0..ds.n_obs).filter(|&i| ids[i] == g).collect();
            let take = if count { amount as usize } else { (members.len() as f64 * amount / 100.0).round() as usize };
            // partial Fisher-Yates shuffle
            let mut rng = st.rng.borrow_mut();
            for k in 0..members.len().min(take) {
                let j = k + ((members.len() - k) as f64 * rng.uniform()) as usize;
                members.swap(k, j);
            }
            for &i in members.iter().take(take) {
                keep[i] = true;
            }
        }
        let before = ds.n_obs;
        ds.keep_if(&keep)?;
        st.data_changed = true;
        Ok(format!("({} deleted)", plural(before - ds.n_obs, "observation")))
    }
}
