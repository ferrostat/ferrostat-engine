//! Native, clean-room versions of popular community data-management
//! packages: the gtools family (gcollapse, gegen, gquantiles, gstats, gtop,
//! gunique, gdistinct, glevelsof, gisid, gduplicates, hashsort, greshape) and
//! the ftools family (fcollapse, fegen, fisid, fsort, join, flevelsof), plus
//! the egenmore-style functions used by `egen`/`gegen`.
//!
//! Many commands delegate to the built-in command of the same meaning after
//! translating options; the rest are implemented here.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, split_top_level_words};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::restructure::{stat_value, take_rows};
use crate::store::Store;
use crate::util::{self, WeightKind};
use crate::{data, egen, listing, restructure, summarize, varlist};

// ---- option helpers ---------------------------------------------------------

pub(crate) fn strip(c: &mut Command, names: &[&str]) {
    c.opts.retain(|(n, _)| !names.contains(&n.to_lowercase().as_str()));
}

/// Remove and return the first option whose name is one of `names`.
pub(crate) fn take_any(c: &mut Command, names: &[&str]) -> Option<Option<String>> {
    let p = c.opts.iter().position(|(n, _)| names.contains(&n.to_lowercase().as_str()))?;
    Some(c.opts.remove(p).1)
}

pub(crate) fn put(c: &mut Command, name: &str, val: Option<&str>) {
    c.opts.push((name.to_string(), val.map(|s| s.to_string())));
}

const IGNORED: &[&str] = &["verbose", "benchmark", "bench", "fast", "forcestrl", "nods", "ds", "compress", "forceio", "unsorted", "oncollision", "debug_level", "hashlib", "nodots", "noi"];

fn run(target: &str, ds: &mut DataSet, st: &mut Store, c: &Command) -> Result<String, String> {
    match target {
        "isid" => listing::Isid.execute(ds, st, c),
        "duplicates" => listing::Duplicates.execute(ds, st, c),
        "levelsof" => listing::Levelsof.execute(ds, st, c),
        "sort" => data::Sort.execute(ds, st, c),
        "gsort" => data::Gsort.execute(ds, st, c),
        "reshape" => restructure::Reshape.execute(ds, st, c),
        "merge" => restructure::Merge.execute(ds, st, c),
        "summarize" => summarize::Summarize.execute(ds, st, c),
        "tabstat" => summarize::Tabstat.execute(ds, st, c),
        "distinct" => crate::community::Distinct.execute(ds, st, c),
        "winsor2" => crate::community::Winsor2.execute(ds, st, c),
        "egen" => egen::Egen.execute(ds, st, c),
        t => Err(rc_err(199, format!("command {t} is unrecognized"))),
    }
}

/// Commands that map directly onto a built-in command.
pub struct Delegate {
    pub name: &'static str,
    pub target: &'static str,
}

impl CommandHandler for Delegate {
    fn name(&self) -> &'static str {
        self.name
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        strip(&mut c, IGNORED);
        c.verb = self.target.to_string();
        if self.target == "sort" && cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("generate") || n.eq_ignore_ascii_case("gen")) {
            c.verb = "gsort".into();
            if let Some(g) = take_any(&mut c, &["gen", "generate"]) {
                put(&mut c, "generate", g.as_deref());
            }
            return run("gsort", ds, st, &c);
        }
        if self.name == "hashsort" {
            strip(&mut c, &["sortgen", "mfirst_"]);
            if let Some(g) = take_any(&mut c, &["gen", "generate"]) {
                put(&mut c, "generate", g.as_deref());
            }
        }
        run(self.target, ds, st, &c)
    }
}

// ---- statistics shared by gcollapse / gegen -------------------------------------

pub(crate) fn valid_stat(s: &str) -> bool {
    matches!(
        s,
        "mean" | "sd" | "sum" | "count" | "min" | "max" | "median" | "iqr" | "first" | "last" | "firstnm" | "lastnm" | "nunique" | "nmissing" | "skewness" | "kurtosis" | "geomean" | "cv" | "semean" | "sebinomial" | "sepoisson" | "rawsum" | "percent" | "variance" | "var" | "range"
    ) || s.strip_prefix('p').is_some_and(|r| r.parse::<f64>().is_ok())
        || ["select", "-select", "rawselect", "-rawselect"].iter().any(|p| s.strip_prefix(p).is_some_and(|r| r.parse::<usize>().is_ok()))
}

/// One group statistic. `vals` holds nonmissing (x, weight) pairs, `all`
/// every value of the group in data order.
pub(crate) fn gstat(stat: &str, vals: &mut Vec<(f64, f64)>, all: &[f64], wk: Option<WeightKind>) -> f64 {
    match stat {
        "nmissing" => return (all.len() - vals.len()) as f64,
        "nunique" => {
            let mut x: Vec<u64> = vals.iter().map(|v| v.0.to_bits()).collect();
            x.sort_unstable();
            x.dedup();
            return x.len() as f64;
        }
        "first" => return all.first().copied().unwrap_or(SYSMISS),
        "last" => return all.last().copied().unwrap_or(SYSMISS),
        "range" => {
            if vals.is_empty() {
                return SYSMISS;
            }
            let mx = vals.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
            let mn = vals.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
            return mx - mn;
        }
        _ => {}
    }
    if vals.is_empty() {
        return stat_value(stat, vals, wk);
    }
    match stat {
        "skewness" | "kurtosis" => {
            let x: Vec<f64> = vals.iter().map(|v| v.0).collect();
            let m = egen::moments(&x);
            if stat == "skewness" { m.0 } else { m.1 }
        }
        "geomean" => {
            if vals.iter().any(|v| v.0 <= 0.0) {
                SYSMISS
            } else {
                (vals.iter().map(|v| v.0.ln()).sum::<f64>() / vals.len() as f64).exp()
            }
        }
        "cv" => {
            let sd = stat_value("sd", vals, wk);
            let m = stat_value("mean", vals, wk);
            if is_missing(sd) || m == 0.0 { SYSMISS } else { sd / m }
        }
        s if ["select", "-select", "rawselect", "-rawselect"].iter().any(|p| s.strip_prefix(p).is_some_and(|r| r.parse::<usize>().is_ok())) => {
            let neg = s.starts_with('-');
            let k: usize = s.trim_start_matches(|c: char| !c.is_ascii_digit()).parse().unwrap_or(0);
            let mut x: Vec<f64> = vals.iter().map(|v| v.0).collect();
            x.sort_by(f64::total_cmp);
            if k == 0 || k > x.len() {
                SYSMISS
            } else if neg {
                x[x.len() - k]
            } else {
                x[k - 1]
            }
        }
        "firstnm" | "lastnm" | "first" | "last" => stat_value(stat, vals, wk),
        _ => stat_value(stat, vals, wk),
    }
}

// ---- gcollapse / fcollapse --------------------------------------------------------

enum Res {
    Num(Vec<f64>),
    Str(Vec<String>),
}

fn pretty(stat: &str) -> String {
    match stat {
        "mean" => "Mean".into(),
        "sd" => "Std. Dev.".into(),
        "sum" => "Sum".into(),
        "count" => "Count".into(),
        "min" => "Min".into(),
        "max" => "Max".into(),
        "median" => "Median".into(),
        "iqr" => "IQR".into(),
        "first" => "First".into(),
        "last" => "Last".into(),
        "nunique" => "N Unique".into(),
        "nmissing" => "N Missing".into(),
        s => s.to_string(),
    }
}

fn make_label(fmt: Option<&str>, stat: &str, src_label: &str) -> String {
    let f = fmt.map(|s| s.to_string()).unwrap_or_else(|| "(#stat#) #sourcelabel#".into());
    f.replace("#stat:pretty#", &pretty(stat))
        .replace("#STAT#", &stat.to_uppercase())
        .replace("#Stat#", &{
            let mut c = stat.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        })
        .replace("#stat#", stat)
        .replace("#sourcelabel#", src_label)
}

pub struct Gcollapse {
    pub name: &'static str,
}

impl CommandHandler for Gcollapse {
    fn name(&self) -> &'static str {
        self.name
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        let merge = take_any(&mut c, &["merge"]).is_some();
        let freq = take_any(&mut c, &["freq"]).map(|v| v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| "_freq".into()));
        let labelformat = take_any(&mut c, &["labelformat"]).flatten();
        strip(&mut c, IGNORED);
        strip(&mut c, &["double", "forcemath", "sumcheck", "wild", "smart", "pre", "replace", "nods", "freeze", "dropmiss"]);
        let cmd = &c;
        let mut targets: Vec<(String, String, String)> = Vec::new();
        let mut stat = "mean".to_string();
        for w in split_top_level_words(&cmd.main.replace(" = ", "=").replace("= ", "=").replace(" =", "=")) {
            if w.starts_with('(') && w.ends_with(')') {
                stat = w[1..w.len() - 1].trim().to_lowercase();
                if stat == "avg" {
                    stat = "mean".into();
                }
                if stat == "stdev" {
                    stat = "sd".into();
                }
                if !valid_stat(&stat) {
                    return Err(rc_err(198, format!("unknown stat {stat}")));
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
        let mut seen = std::collections::HashSet::new();
        for t in &targets {
            if !seen.insert(t.1.clone()) {
                return Err(rc_err(110, format!("{} already defined", t.1)));
            }
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
        let wk = w.as_ref().map(|w| w.kind);
        let n = ds.n_obs;
        let mut results: Vec<(Res, VarType, String, String)> = Vec::new();
        for (stat, new, src) in &targets {
            let var = ds.get_var(src).unwrap();
            let is_str = var.var_type.is_string();
            let label = make_label(labelformat.as_deref(), stat, if var.label.is_empty() { src } else { &var.label });
            if is_str {
                if !matches!(stat.as_str(), "first" | "last" | "firstnm" | "lastnm" | "count" | "nunique" | "nmissing") {
                    return Err(rc_err(109, format!("{src}: string variable not allowed with ({stat})")));
                }
                let mut b: Vec<Vec<String>> = vec![Vec::new(); ng];
                for i in 0..n {
                    if ids[i] != usize::MAX {
                        b[ids[i]].push(util::cell_text(ds, var, i, false));
                    }
                }
                if matches!(stat.as_str(), "first" | "last" | "firstnm" | "lastnm") {
                    let r = b
                        .iter()
                        .map(|g| {
                            let nm: Vec<&String> = g.iter().filter(|s| !s.is_empty()).collect();
                            match stat.as_str() {
                                "first" => g.first().cloned().unwrap_or_default(),
                                "last" => g.last().cloned().unwrap_or_default(),
                                "firstnm" => nm.first().map(|s| s.to_string()).unwrap_or_default(),
                                _ => nm.last().map(|s| s.to_string()).unwrap_or_default(),
                            }
                        })
                        .collect();
                    results.push((Res::Str(r), var.var_type, new.clone(), label));
                } else {
                    let r = b
                        .iter()
                        .map(|g| {
                            let nm: Vec<&String> = g.iter().filter(|s| !s.is_empty()).collect();
                            match stat.as_str() {
                                "count" => nm.len() as f64,
                                "nmissing" => (g.len() - nm.len()) as f64,
                                _ => {
                                    let mut u: Vec<&&String> = nm.iter().collect();
                                    u.sort();
                                    u.dedup();
                                    u.len() as f64
                                }
                            }
                        })
                        .collect();
                    results.push((Res::Num(r), VarType::Long, new.clone(), label));
                }
                continue;
            }
            let col = util::col(ds, src)?;
            let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ng];
            let mut alls: Vec<Vec<f64>> = vec![Vec::new(); ng];
            let mut total_w = 0.0;
            for i in 0..n {
                if ids[i] == usize::MAX {
                    continue;
                }
                let wi = w.as_ref().map_or(1.0, |w| w.w[i]);
                alls[ids[i]].push(col[i]);
                if !is_missing(col[i]) {
                    buckets[ids[i]].push((col[i], wi));
                    total_w += wi;
                }
            }
            let rawstat = stat.starts_with("rawselect") || stat == "rawsum";
            let vals: Vec<f64> = (0..ng)
                .map(|g| {
                    if stat == "percent" {
                        100.0 * buckets[g].iter().map(|v| v.1).sum::<f64>() / total_w
                    } else {
                        let mut b = std::mem::take(&mut buckets[g]);
                        gstat(stat, &mut b, &alls[g], if rawstat { None } else { wk })
                    }
                })
                .collect();
            let ty = match stat.as_str() {
                "count" | "nunique" | "nmissing" => VarType::Long,
                "first" | "last" | "firstnm" | "lastnm" | "min" | "max" => var.var_type,
                s if s.contains("select") => var.var_type,
                "sum" | "rawsum" => VarType::Double,
                _ => {
                    if var.var_type == VarType::Double { VarType::Double } else { VarType::Float }
                }
            };
            results.push((Res::Num(vals), ty, new.clone(), label));
        }
        let freq_vals: Option<Vec<f64>> = freq.as_ref().map(|_| {
            let mut f = vec![0.0; ng];
            for i in 0..n {
                if ids[i] != usize::MAX {
                    f[ids[i]] += w.as_ref().map_or(1.0, |w| if w.kind == WeightKind::F { w.w[i] } else { 1.0 });
                }
            }
            f
        });
        let mk = |res: &Res, ty: VarType, name: &str, label: &str, idx: &dyn Fn(usize) -> Option<usize>, len: usize| -> Variable {
            match res {
                Res::Num(v) => {
                    let mut var = Variable::new(name, ty);
                    var.label = label.to_string();
                    var.data = (0..len).map(|i| data::store_num(ty, idx(i).map_or(SYSMISS, |g| v[g]))).collect();
                    var
                }
                Res::Str(v) => {
                    let wd = v.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                    let mut var = Variable::new(name, util::str_type_for(wd));
                    var.label = label.to_string();
                    var.data = (0..len).map(|i| Value::Str(idx(i).map(|g| v[g].clone()).unwrap_or_default())).collect();
                    var
                }
            }
        };
        if merge {
            for (_, _, new, _) in &results {
                if ds.var_index.contains_key(new) {
                    return Err(rc_err(110, format!("{new} already defined")));
                }
            }
            let idx = |i: usize| if ids[i] == usize::MAX { None } else { Some(ids[i]) };
            let mut newvars = Vec::new();
            for (res, ty, name, label) in &results {
                newvars.push(mk(res, *ty, name, label, &idx, n));
            }
            if let (Some(f), Some(fv)) = (&freq, &freq_vals) {
                newvars.push(mk(&Res::Num(fv.clone()), VarType::Long, f, "", &idx, n));
            }
            for v in newvars {
                ds.add_variable(v)?;
            }
            st.data_changed = true;
            return Ok(String::new());
        }
        let mut out = take_rows(ds, &reps);
        out.keep_variables(&by);
        let idx = |i: usize| Some(i);
        for (res, ty, name, label) in &results {
            if out.var_index.contains_key(name) {
                return Err(rc_err(110, format!("{name} already defined")));
            }
            out.add_variable(mk(res, *ty, name, label, &idx, ng))?;
        }
        if let (Some(f), Some(fv)) = (&freq, &freq_vals) {
            out.add_variable(mk(&Res::Num(fv.clone()), VarType::Long, f, "", &idx, ng))?;
        }
        out.sort_by = by;
        *ds = out;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- gegen / fegen ------------------------------------------------------------------

pub struct Gegen {
    pub name: &'static str,
}

impl CommandHandler for Gegen {
    fn name(&self) -> &'static str {
        self.name
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        c.verb = "egen".into();
        strip(&mut c, IGNORED);
        strip(&mut c, &["double", "nods", "counts", "p", "unsorted"]);
        let replace = take_any(&mut c, &["replace"]).is_some();
        let eq = find_top_level(&c.main, '=').ok_or_else(|| rc_err(198, "syntax: gegen newvar = fcn(arguments)"))?;
        let lhs: Vec<String> = c.main[..eq].split_whitespace().map(|s| s.to_string()).collect();
        let newvar = lhs.last().cloned().ok_or_else(|| rc_err(198, "invalid syntax"))?;
        if replace && ds.var_index.contains_key(&newvar) {
            let tmp = "__gg_tmp".to_string();
            let mut l2 = lhs.clone();
            *l2.last_mut().unwrap() = tmp.clone();
            c.main = format!("{} {}", l2.join(" "), &c.main[eq..]);
            let r = egen::Egen.execute(ds, st, &c)?;
            let pos = ds.var_index[&newvar];
            let mut v = ds.get_var(&tmp).unwrap().clone();
            v.name = newvar.clone();
            ds.drop_variable(&tmp)?;
            ds.vars[pos] = v;
            return Ok(r);
        }
        egen::Egen.execute(ds, st, &c)
    }
}

pub enum Extra {
    Num(Vec<f64>),
    Str(Vec<String>),
}

/// egen functions beyond the built-in set: gtools extras and egenmore
/// functions (nvals, rall, rany, rcount, outside, record). None when unknown.
pub fn egen_extra(func: &str, args: &str, ds: &DataSet, st: &Store, cmd: &Command, mask: &[bool], by: &[String]) -> Result<Option<Extra>, String> {
    let n = ds.n_obs;
    match func {
        "rall" | "rany" | "rcount" => {
            let vars = varlist::expand(ds, args)?;
            let cond = cmd.opt("cond", 2).ok_or_else(|| rc_err(198, "option cond() required"))?.to_string();
            let env = Env::with_store(ds, st);
            let exprs: Vec<_> = vars
                .iter()
                .map(|v| parse_expression(&cond.replace('@', &format!("({v})"))).map_err(|e| rc_err(198, e.message)))
                .collect::<Result<_, _>>()?;
            let mut out = vec![SYSMISS; n];
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                let mut k = 0;
                for e in &exprs {
                    let v = env.eval(e, i)?.num()?;
                    if !is_missing(v) && v != 0.0 {
                        k += 1;
                    }
                }
                out[i] = match func {
                    "rall" => (k == exprs.len()) as u8 as f64,
                    "rany" => (k > 0) as u8 as f64,
                    _ => k as f64,
                };
            }
            Ok(Some(Extra::Num(out)))
        }
        "id" => {
            let vars = varlist::expand(ds, args)?;
            let (ids, _) = util::group_ids(ds, &vars, mask);
            Ok(Some(Extra::Num(ids.iter().map(|&g| if g == usize::MAX { SYSMISS } else { g as f64 + 1.0 }).collect())))
        }
        "nvals" | "nunique" | "nmissing" | "first" | "last" | "firstnm" | "lastnm" | "skewness" | "kurtosis" | "geomean" | "cv" | "sebinomial" | "sepoisson" | "rawsum" | "variance" | "var" | "range" | "record" | "outside" | "percent" => {
            let (ids, ng) = egen::by_ids(ds, st, by, mask);
            let a = args.trim();
            let strvar = ds.get_var(a).filter(|v| v.var_type.is_string());
            let func = if func == "nvals" { "nunique" } else { func };
            if let Some(sv) = strvar {
                let mut b: Vec<Vec<String>> = vec![Vec::new(); ng];
                for i in 0..n {
                    if ids[i] != usize::MAX {
                        b[ids[i]].push(util::cell_text(ds, sv, i, false));
                    }
                }
                return match func {
                    "first" | "last" | "firstnm" | "lastnm" => {
                        let r: Vec<String> = b
                            .iter()
                            .map(|g| {
                                let nm: Vec<&String> = g.iter().filter(|s| !s.is_empty()).collect();
                                match func {
                                    "first" => g.first().cloned().unwrap_or_default(),
                                    "last" => g.last().cloned().unwrap_or_default(),
                                    "firstnm" => nm.first().map(|s| s.to_string()).unwrap_or_default(),
                                    _ => nm.last().map(|s| s.to_string()).unwrap_or_default(),
                                }
                            })
                            .collect();
                        Ok(Some(Extra::Str((0..n).map(|i| if ids[i] == usize::MAX { String::new() } else { r[ids[i]].clone() }).collect())))
                    }
                    "nunique" | "nmissing" => {
                        let r: Vec<f64> = b
                            .iter()
                            .map(|g| {
                                let mut nm: Vec<&String> = g.iter().filter(|s| !s.is_empty()).collect();
                                if func == "nmissing" {
                                    return (g.len() - nm.len()) as f64;
                                }
                                nm.sort();
                                nm.dedup();
                                nm.len() as f64
                            })
                            .collect();
                        Ok(Some(Extra::Num((0..n).map(|i| if ids[i] == usize::MAX { SYSMISS } else { r[ids[i]] }).collect())))
                    }
                    _ => Err(rc_err(109, "type mismatch")),
                };
            }
            let e = parse_expression(a).map_err(|e| rc_err(198, e.message))?;
            let env = Env::with_store(ds, st);
            let vals: Vec<f64> = (0..n).map(|i| if mask[i] { env.eval(&e, i).and_then(|v| v.num()) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?;
            if func == "record" {
                let min = cmd.has_opt("min", 3);
                let mut order: Vec<usize> = (0..n).filter(|&i| ids[i] != usize::MAX).collect();
                if let Some(o) = cmd.opt("order", 3) {
                    let ov = varlist::expand(ds, o)?;
                    let cols: Vec<Vec<f64>> = ov.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
                    order.sort_by(|&x, &y| cols.iter().map(|c| c[x].total_cmp(&c[y])).find(|o| o.is_ne()).unwrap_or(x.cmp(&y)));
                }
                let mut best = vec![SYSMISS; ng];
                let mut out = vec![SYSMISS; n];
                for i in order {
                    let g = ids[i];
                    let x = vals[i];
                    if !is_missing(x) && (is_missing(best[g]) || (min && x < best[g]) || (!min && x > best[g])) {
                        best[g] = x;
                    }
                    out[i] = best[g];
                }
                return Ok(Some(Extra::Num(out)));
            }
            if func == "outside" {
                let lo_o = util::opt_num(cmd, "lower", 3)?;
                let hi_o = util::opt_num(cmd, "upper", 3)?;
                let mut b: Vec<Vec<f64>> = vec![Vec::new(); ng];
                for i in 0..n {
                    if ids[i] != usize::MAX && !is_missing(vals[i]) {
                        b[ids[i]].push(vals[i]);
                    }
                }
                let fences: Vec<(f64, f64)> = b
                    .iter_mut()
                    .map(|g| {
                        g.sort_by(f64::total_cmp);
                        let (q1, q3) = (util::pctile_sorted(g, 25.0), util::pctile_sorted(g, 75.0));
                        (lo_o.unwrap_or(q1 - 1.5 * (q3 - q1)), hi_o.unwrap_or(q3 + 1.5 * (q3 - q1)))
                    })
                    .collect();
                let out = (0..n).map(|i| if ids[i] == usize::MAX || is_missing(vals[i]) { SYSMISS } else { let (l, h) = fences[ids[i]]; (vals[i] < l || vals[i] > h) as u8 as f64 }).collect();
                return Ok(Some(Extra::Num(out)));
            }
            let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ng];
            let mut alls: Vec<Vec<f64>> = vec![Vec::new(); ng];
            for i in 0..n {
                if ids[i] != usize::MAX {
                    alls[ids[i]].push(vals[i]);
                    if !is_missing(vals[i]) {
                        buckets[ids[i]].push((vals[i], 1.0));
                    }
                }
            }
            let tot: f64 = buckets.iter().map(|b| b.len() as f64).sum();
            let res: Vec<f64> = (0..ng)
                .map(|g| {
                    if func == "percent" {
                        return 100.0 * buckets[g].len() as f64 / tot;
                    }
                    let mut b = std::mem::take(&mut buckets[g]);
                    gstat(func, &mut b, &alls[g], None)
                })
                .collect();
            Ok(Some(Extra::Num((0..n).map(|i| if ids[i] == usize::MAX { SYSMISS } else { res[ids[i]] }).collect())))
        }
        f if f.strip_prefix('p').is_some_and(|r| r.parse::<f64>().is_ok()) || f.contains("select") => {
            let (ids, ng) = egen::by_ids(ds, st, by, mask);
            let e = parse_expression(args.trim()).map_err(|e| rc_err(198, e.message))?;
            let env = Env::with_store(ds, st);
            let vals: Vec<f64> = (0..n).map(|i| if mask[i] { env.eval(&e, i).and_then(|v| v.num()) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?;
            let mut b: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ng];
            for i in 0..n {
                if ids[i] != usize::MAX && !is_missing(vals[i]) {
                    b[ids[i]].push((vals[i], 1.0));
                }
            }
            let res: Vec<f64> = b.iter_mut().map(|g| gstat(f, g, &[], None)).collect();
            Ok(Some(Extra::Num((0..n).map(|i| if ids[i] == usize::MAX { SYSMISS } else { res[ids[i]] }).collect())))
        }
        _ => Ok(None),
    }
}

// ---- glevelsof / flevelsof ----------------------------------------------------------

pub struct Glevelsof {
    pub name: &'static str,
}

impl CommandHandler for Glevelsof {
    fn name(&self) -> &'static str {
        self.name
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        if !cmd.has_opt("missing", 4) {
            util::markout(ds, &mut mask, &vars);
        }
        let sep = cmd.opt("separate", 3).map(util::unquote).unwrap_or_else(|| " ".into());
        let colsep = cmd.opt("colseparate", 4).map(util::unquote).unwrap_or_else(|| " ".into());
        let clean = cmd.has_opt("clean", 5);
        let (_, reps) = util::group_ids(ds, &vars, &mask);
        let rows: Vec<String> = reps
            .iter()
            .map(|&i| {
                vars.iter()
                    .map(|v| {
                        let var = ds.get_var(v).unwrap();
                        if var.var_type.is_string() {
                            let s = var.str_at(i).to_string();
                            if clean { s } else { format!("`\"{s}\"'") }
                        } else {
                            crate::format::fmt_g(var.num(i), 17)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(&colsep)
            })
            .collect();
        let text = rows.join(&sep);
        let nobs = mask.iter().filter(|&&m| m).count();
        let r = st.begin_r();
        r.set_str("levels", text.clone());
        r.set("J", rows.len() as f64);
        r.set("N", nobs as f64);
        r.set("r", rows.len() as f64);
        if let Some(l) = cmd.opt("local", 3) {
            st.set_local(l.trim(), text.clone());
        }
        Ok(if cmd.has_opt("silent", 3) { String::new() } else { text })
    }
}

// ---- gunique ------------------------------------------------------------------------

pub struct Gunique;

impl CommandHandler for Gunique {
    fn name(&self) -> &'static str {
        "gunique"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mask = util::touse(ds, st, cmd)?;
        let (ids, reps) = util::group_ids(ds, &vars, &mask);
        let mut sizes = vec![0usize; reps.len()];
        for &g in ids.iter().filter(|&&g| g != usize::MAX) {
            sizes[g] += 1;
        }
        let nn = mask.iter().filter(|&&m| m).count();
        let (mn, mx) = (sizes.iter().min().copied().unwrap_or(0), sizes.iter().max().copied().unwrap_or(0));
        if let Some(g) = cmd.opt("generate", 3).or_else(|| cmd.opt("gen", 3)) {
            let name = g.trim();
            let vals: Vec<f64> = ids.iter().map(|&g| if g == usize::MAX { SYSMISS } else { g as f64 + 1.0 }).collect();
            if cmd.has_opt("replace", 7) && ds.var_index.contains_key(name) {
                ds.drop_variable(name)?;
            }
            ds.add_variable(util::new_double(name, vals))?;
        }
        if let Some(g) = cmd.opt("counts", 3) {
            let vals: Vec<f64> = ids.iter().map(|&g| if g == usize::MAX { SYSMISS } else { sizes[g] as f64 }).collect();
            ds.add_variable(util::new_double(g.trim(), vals))?;
        }
        let r = st.begin_r();
        r.set("N", nn as f64);
        r.set("J", reps.len() as f64);
        r.set("minJ", mn as f64);
        r.set("maxJ", mx as f64);
        st.data_changed = true;
        Ok(format!("N = {}; {} unique values of {}", util::commas(nn), util::commas(reps.len()), vars.join(" ")))
    }
}

// ---- gtop ---------------------------------------------------------------------------

pub struct Gtop;

impl CommandHandler for Gtop {
    fn name(&self) -> &'static str {
        "gtop"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        if cmd.has_opt("nomissing", 3) || !cmd.has_opt("missrow", 4) && !cmd.has_opt("groupmiss", 5) {
            util::markout(ds, &mut mask, &vars);
        }
        let ntop = util::opt_num(cmd, "ntop", 2)?.unwrap_or(10.0) as usize;
        let (ids, reps) = util::group_ids(ds, &vars, &mask);
        let mut cnt = vec![0usize; reps.len()];
        for &g in ids.iter().filter(|&&g| g != usize::MAX) {
            cnt[g] += 1;
        }
        let total: usize = cnt.iter().sum();
        let mut order: Vec<usize> = (0..reps.len()).collect();
        order.sort_by(|&a, &b| cnt[b].cmp(&cnt[a]).then(a.cmp(&b)));
        order.truncate(ntop);
        let label = |g: usize| vars.iter().map(|v| util::cell_text(ds, ds.get_var(v).unwrap(), reps[g], true)).collect::<Vec<_>>().join(" ");
        let w = order.iter().map(|&g| label(g).len()).max().unwrap_or(1).max(vars.join(" ").len());
        let mut out = vec![format!("   | {:w$} | {:>8} {:>8} {:>9} {:>13}", vars.join(" "), "N", "Cum", "Pct (%)", "Cum Pct (%)")];
        let (mut cum, mut levels) = (0usize, Vec::new());
        for (k, &g) in order.iter().enumerate() {
            cum += cnt[g];
            out.push(format!("{:>2} | {:w$} | {:>8} {:>8} {:>9.3} {:>13.3}", k + 1, label(g), cnt[g], cum, 100.0 * cnt[g] as f64 / total as f64, 100.0 * cum as f64 / total as f64));
            levels.push(format!("`\"{}\"'", label(g)));
        }
        let r = st.begin_r();
        r.set("N", total as f64);
        r.set("J", reps.len() as f64);
        r.set_str("levels", levels.join(" "));
        Ok(out.join("\n"))
    }
}

// ---- gquantiles / fasterxtile ----------------------------------------------------------

pub struct Gquantiles {
    pub name: &'static str,
}

impl CommandHandler for Gquantiles {
    fn name(&self) -> &'static str {
        self.name
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        strip(&mut c, IGNORED);
        let xtile = take_any(&mut c, &["xtile"]).is_some() || self.name == "fasterxtile";
        let pctile = take_any(&mut c, &["pctile"]).is_some();
        let _pct = take_any(&mut c, &["_pctile"]).is_some();
        let (newvar, expr) = match find_top_level(&c.main, '=') {
            Some(eq) => (Some(c.main[..eq].trim().to_string()), c.main[eq + 1..].trim().to_string()),
            None => (None, c.main.trim().to_string()),
        };
        let mask = util::touse(ds, st, &c)?;
        let e = parse_expression(&expr).map_err(|e| rc_err(198, e.message))?;
        let x = util::eval_col(ds, st, &e, &mask)?;
        let by = match c.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let nq = take_any(&mut c, &["nquantiles", "nq", "n"]).flatten();
        let quantiles = take_any(&mut c, &["quantiles", "q"]).flatten();
        let cutoffs = take_any(&mut c, &["cutoffs"]).flatten();
        let cutvar = take_any(&mut c, &["cutpoints"]).flatten();
        let genp = take_any(&mut c, &["genp"]).flatten();
        let replace = take_any(&mut c, &["replace"]).is_some();
        let pcts: Vec<f64> = if let Some(q) = quantiles {
            q.split_whitespace().map(|s| s.parse::<f64>().map_err(|_| rc_err(198, "quantiles() invalid"))).collect::<Result<_, _>>()?
        } else {
            let k: usize = nq.as_deref().map(|s| s.trim().parse().map_err(|_| rc_err(198, "nquantiles() invalid"))).transpose()?.unwrap_or(2);
            if k < 2 {
                return Err(rc_err(198, "nquantiles() must be >= 2"));
            }
            (1..k).map(|i| 100.0 * i as f64 / k as f64).collect()
        };
        let fixed: Option<Vec<f64>> = if let Some(cu) = cutoffs {
            Some(cu.split_whitespace().map(|s| s.parse::<f64>().map_err(|_| rc_err(198, "cutoffs() invalid"))).collect::<Result<_, _>>()?)
        } else if let Some(v) = cutvar {
            let col = util::col(ds, v.trim())?;
            let mut f: Vec<f64> = col.into_iter().filter(|v| !is_missing(*v)).collect();
            f.sort_by(f64::total_cmp);
            f.dedup();
            Some(f)
        } else {
            None
        };
        let (ids, ng) = if by.is_empty() { (mask.iter().map(|&m| if m { 0 } else { usize::MAX }).collect::<Vec<_>>(), 1) } else {
            let (i, r) = util::group_ids(ds, &by, &mask);
            (i, r.len())
        };
        let cuts_for = |g: usize| -> Vec<f64> {
            if let Some(f) = &fixed {
                return f.clone();
            }
            let mut v: Vec<f64> = (0..x.len()).filter(|&i| ids[i] == g && !is_missing(x[i])).map(|i| x[i]).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() { Vec::new() } else { pcts.iter().map(|&p| util::pctile_sorted(&v, p)).collect() }
        };
        if newvar.is_none() || (!xtile && !pctile) {
            let cuts = cuts_for(0);
            let r = st.begin_r();
            for (k, c) in cuts.iter().enumerate() {
                r.set(&format!("r{}", k + 1), *c);
            }
            return Ok(String::new());
        }
        let name = newvar.unwrap();
        if replace && ds.var_index.contains_key(&name) {
            ds.drop_variable(&name)?;
        }
        if xtile {
            let cuts: Vec<Vec<f64>> = (0..ng).map(&cuts_for).collect();
            let out: Vec<f64> = (0..ds.n_obs).map(|i| if ids[i] == usize::MAX || is_missing(x[i]) { SYSMISS } else { 1.0 + cuts[ids[i]].iter().filter(|&&c| c < x[i]).count() as f64 }).collect();
            let t = util::best_numeric_type(&out);
            let mut v = Variable::new(&name, t);
            v.data = out.iter().map(|&o| data::store_num(t, o)).collect();
            ds.add_variable(v)?;
        } else {
            let cuts = cuts_for(0);
            let n = ds.n_obs;
            let mut v = Variable::new(&name, VarType::Double);
            v.data = (0..n).map(|i| Value::Double(cuts.get(i).copied().unwrap_or(SYSMISS))).collect();
            ds.add_variable(v)?;
            if let Some(g) = genp {
                let mut p = Variable::new(g.trim(), VarType::Double);
                p.data = (0..n).map(|i| Value::Double(pcts.get(i).copied().unwrap_or(SYSMISS))).collect();
                ds.add_variable(p)?;
            }
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- gstats -----------------------------------------------------------------------------

pub struct Gstats;

impl CommandHandler for Gstats {
    fn name(&self) -> &'static str {
        "gstats"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        strip(&mut c, IGNORED);
        let main = c.main.trim().to_string();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main.as_str(), ""));
        c.main = rest.trim().to_string();
        match sub.to_lowercase().as_str() {
            "sum" | "summarize" | "summ" => {
                if let Some(by) = take_any(&mut c, &["by"]).flatten() {
                    let bv = varlist::expand(ds, &by)?;
                    let mask = util::touse(ds, st, &c)?;
                    let (ids, reps) = util::group_ids(ds, &bv, &mask);
                    let mut out = Vec::new();
                    c.if_expr = None;
                    c.if_raw = None;
                    c.in_range = None;
                    for (g, &rep) in reps.iter().enumerate() {
                        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| ids[i] == g).collect();
                        let mut sub = take_rows(ds, &rows);
                        let lab: Vec<String> = bv.iter().map(|v| format!("{} = {}", v, util::cell_text(ds, ds.get_var(v).unwrap(), rep, true))).collect();
                        out.push(format!("\n-> {}", lab.join(", ")));
                        out.push(summarize::Summarize.execute(&mut sub, st, &c)?);
                    }
                    return Ok(out.join("\n"));
                }
                summarize::Summarize.execute(ds, st, &c)
            }
            "tabstat" => summarize::Tabstat.execute(ds, st, &c),
            "winsor" => {
                let vars = varlist::expand(ds, &c.main)?;
                let genv = take_any(&mut c, &["generate", "gen"]).flatten();
                let prefix = take_any(&mut c, &["prefix"]).flatten();
                if genv.is_some() || prefix.is_some() {
                    let names: Vec<String> = match (&genv, &prefix) {
                        (Some(g), _) => g.split_whitespace().map(|s| s.to_string()).collect(),
                        (_, Some(p)) => vars.iter().map(|v| format!("{}{v}", p.trim())).collect(),
                        _ => unreachable!(),
                    };
                    if names.len() != vars.len() {
                        return Err(rc_err(198, "generate() must name one variable per input"));
                    }
                    for (v, nm) in vars.iter().zip(&names) {
                        let mut nv = ds.get_var(v).unwrap().clone();
                        nv.name = nm.clone();
                        ds.add_variable(nv)?;
                    }
                    c.main = names.join(" ");
                    put(&mut c, "replace", None);
                    strip(&mut c, &["suffix"]);
                }
                crate::community::Winsor2.execute(ds, st, &c)
            }
            "hdfe" => hdfe_basic(ds, st, &c),
            other => Err(rc_err(198, format!("gstats {other}: invalid subcommand"))),
        }
    }
}

fn hdfe_basic(ds: &mut DataSet, st: &mut Store, c: &Command) -> Result<String, String> {
    let vars = varlist::expand(ds, &c.main)?;
    let abs = varlist::expand(ds, c.opt("absorb", 3).ok_or_else(|| rc_err(198, "option absorb() required"))?)?;
    let mut mask = util::touse(ds, st, c)?;
    util::markout(ds, &mut mask, &vars);
    util::markout(ds, &mut mask, &abs);
    let n = ds.n_obs;
    let mut factors: Vec<Vec<usize>> = Vec::new();
    for a in &abs {
        let (ids, _) = util::group_ids(ds, std::slice::from_ref(a), &mask);
        factors.push(ids);
    }
    let genv = c.opt("generate", 3).or_else(|| c.opt("gen", 3)).map(|s| s.split_whitespace().map(|x| x.to_string()).collect::<Vec<_>>());
    let prefix = c.opt("prefix", 3).map(|s| s.trim().to_string());
    let replace = c.has_opt("replace", 7);
    let targets: Vec<String> = match (&genv, &prefix) {
        (Some(g), _) => g.clone(),
        (_, Some(p)) => vars.iter().map(|v| format!("{p}{v}")).collect(),
        _ if replace => vars.clone(),
        _ => return Err(rc_err(198, "specify generate(), prefix() or replace")),
    };
    if targets.len() != vars.len() {
        return Err(rc_err(198, "generate() must name one variable per input"));
    }
    let mut results = Vec::new();
    for v in &vars {
        let col = util::col(ds, v)?;
        let mut r: Vec<f64> = (0..n).map(|i| if mask[i] { col[i] } else { 0.0 }).collect();
        for _ in 0..10_000 {
            let mut maxd: f64 = 0.0;
            for f in &factors {
                let ng = f.iter().filter(|&&g| g != usize::MAX).max().map_or(0, |m| m + 1);
                let (mut s, mut k) = (vec![0.0; ng], vec![0.0; ng]);
                for i in 0..n {
                    if f[i] != usize::MAX {
                        s[f[i]] += r[i];
                        k[f[i]] += 1.0;
                    }
                }
                for i in 0..n {
                    if f[i] != usize::MAX {
                        let m = s[f[i]] / k[f[i]];
                        r[i] -= m;
                        maxd = maxd.max(m.abs());
                    }
                }
            }
            if maxd < 1e-12 {
                break;
            }
        }
        results.push(r);
    }
    for ((v, t), r) in vars.iter().zip(&targets).zip(results) {
        let vals: Vec<f64> = (0..n).map(|i| if mask[i] { r[i] } else { SYSMISS }).collect();
        let mut nv = util::new_double(t, vals);
        nv.label = ds.get_var(v).unwrap().label.clone();
        if ds.var_index.contains_key(t) {
            let pos = ds.var_index[t];
            ds.vars[pos].var_type = VarType::Double;
            ds.vars[pos].data = nv.data;
        } else {
            ds.add_variable(nv)?;
        }
    }
    st.data_changed = true;
    Ok(String::new())
}

// ---- greshape / join --------------------------------------------------------------------

pub struct Greshape;

impl CommandHandler for Greshape {
    fn name(&self) -> &'static str {
        "greshape"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        c.verb = "reshape".into();
        strip(&mut c, IGNORED);
        strip(&mut c, &["nochecks", "dropmiss", "nodupcheck", "nomisscheck", "xi", "match", "fast", "nodots", "cw"]);
        if let Some(b) = take_any(&mut c, &["by"]) {
            put(&mut c, "i", b.as_deref());
        }
        if let Some(k) = take_any(&mut c, &["keys", "key"]) {
            put(&mut c, "j", k.as_deref());
        }
        restructure::Reshape.execute(ds, st, &c)
    }
}

pub struct Join;

impl CommandHandler for Join {
    fn name(&self) -> &'static str {
        "join"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        c.verb = "merge".into();
        strip(&mut c, IGNORED);
        strip(&mut c, &["noreport", "into"]);
        let from = take_any(&mut c, &["from"]).flatten().ok_or_else(|| rc_err(198, "option from() required"))?;
        let by = take_any(&mut c, &["by"]).flatten().ok_or_else(|| rc_err(198, "option by() required"))?;
        let unique_master = take_any(&mut c, &["uniquemaster", "unique"]).is_some();
        let kind = if unique_master { "1:m" } else { "m:1" };
        if let Some(g) = take_any(&mut c, &["gen", "generate"]) {
            put(&mut c, "generate", g.as_deref());
        }
        if let Some(_) = take_any(&mut c, &["nogen", "nogenerate"]) {
            put(&mut c, "nogenerate", None);
        }
        let keepusing = c.main.trim().to_string();
        if !keepusing.is_empty() {
            put(&mut c, "keepusing", Some(&keepusing));
        }
        c.main = format!("{kind} {by}");
        c.using = Some(util::unquote(from.trim()));
        restructure::Merge.execute(ds, st, &c)
    }
}
