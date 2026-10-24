//! `egen`: extended generate functions.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env, Val};
use crate::store::Store;
use crate::util::{self, pctile_sorted};
use crate::varlist;

pub struct Egen;

/// Split `fcn(args)` into name and the raw argument text.
fn split_call(s: &str) -> Result<(String, String), String> {
    let s = s.trim();
    let open = s.find('(').ok_or_else(|| rc_err(198, "invalid egen function call"))?;
    let close = s.rfind(')').ok_or_else(|| rc_err(198, "invalid egen function call"))?;
    Ok((s[..open].trim().to_lowercase(), s[open + 1..close].to_string()))
}

fn plural(n: usize) -> String {
    format!("({} missing value{} generated)", crate::util::commas(n), if n == 1 { "" } else { "s" })
}

/// Per-observation group id according to by-variables (or the by prefix).
fn by_ids(ds: &DataSet, st: &Store, by: &[String], mask: &[bool]) -> (Vec<usize>, usize) {
    if by.is_empty() {
        if let Some(g) = &st.by_groups {
            // by prefix: groups are contiguous ranges
            let mut ids = vec![usize::MAX; ds.n_obs];
            let mut k = 0;
            let mut last = usize::MAX;
            for i in 0..ds.n_obs {
                if g.start[i] != last {
                    last = g.start[i];
                    k += 1;
                }
                if mask[i] {
                    ids[i] = k - 1;
                }
            }
            return (ids, k);
        }
        let ids = mask.iter().map(|&m| if m { 0 } else { usize::MAX }).collect();
        return (ids, 1);
    }
    let (ids, reps) = util::group_ids(ds, by, mask);
    (ids, reps.len())
}

fn numeric_vals(ds: &DataSet, st: &Store, expr: &str, mask: &[bool]) -> Result<Vec<f64>, String> {
    let e = parse_expression(expr).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st);
    (0..ds.n_obs)
        .map(|i| if mask[i] { env.eval(&e, i)?.num() } else { Ok(SYSMISS) })
        .collect()
}

impl CommandHandler for Egen {
    fn name(&self) -> &'static str {
        "egen"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let eq = find_top_level(&cmd.main, '=').ok_or_else(|| rc_err(198, "egen syntax: egen newvar = fcn(arguments)"))?;
        let lhs: Vec<&str> = cmd.main[..eq].split_whitespace().collect();
        let (ty, newvar) = match lhs.as_slice() {
            [n] => (None, n.to_string()),
            [t, n] => (Some(crate::data::parse_type(t)?), n.to_string()),
            _ => return Err(rc_err(198, "invalid syntax")),
        };
        varlist::check_name(&newvar)?;
        if ds.var_index.contains_key(&newvar) {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        let (func, args) = split_call(&cmd.main[eq + 1..])?;
        let mask = util::touse(ds, st, cmd)?;
        let by: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let mut is_str = false;
        let n = ds.n_obs;
        let mut out = vec![SYSMISS; n];
        let mut sout = vec![String::new(); n];
        let missing_opt = cmd.has_opt("missing", 3);

        // group-statistic helper
        let group_stat = |vals: &[f64], f: &dyn Fn(&mut Vec<f64>) -> f64| -> Vec<f64> {
            let (ids, ng) = by_ids(ds, st, &by, &mask);
            let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); ng];
            for i in 0..n {
                if ids[i] != usize::MAX && !is_missing(vals[i]) {
                    buckets[ids[i]].push(vals[i]);
                }
            }
            let res: Vec<f64> = buckets.iter_mut().map(|b| f(b)).collect();
            (0..n).map(|i| if ids[i] == usize::MAX { SYSMISS } else { res[ids[i]] }).collect()
        };

        match func.as_str() {
            "mean" | "total" | "sum" | "sd" | "count" | "max" | "min" | "median" | "iqr" | "pctile" | "skew" | "kurt" | "mad" | "mdev" | "mode" | "std" | "pc" | "rank" | "semean" => {
                let vals = numeric_vals(ds, st, &args, &mask)?;
                out = match func.as_str() {
                    "mean" => group_stat(&vals, &|b| if b.is_empty() { SYSMISS } else { util::mean(b) }),
                    "total" | "sum" => group_stat(&vals, &|b| b.iter().sum()),
                    "count" => group_stat(&vals, &|b| b.len() as f64),
                    "sd" => group_stat(&vals, &|b| if b.len() < 2 { SYSMISS } else { util::variance(b).sqrt() }),
                    "semean" => group_stat(&vals, &|b| if b.len() < 2 { SYSMISS } else { (util::variance(b) / b.len() as f64).sqrt() }),
                    "max" => group_stat(&vals, &|b| b.iter().cloned().fold(SYSMISS, |a, x| if is_missing(a) { x } else { a.max(x) })),
                    "min" => group_stat(&vals, &|b| b.iter().cloned().fold(SYSMISS, f64::min)),
                    "median" => group_stat(&vals, &|b| { b.sort_by(f64::total_cmp); pctile_sorted(b, 50.0) }),
                    "iqr" => group_stat(&vals, &|b| { b.sort_by(f64::total_cmp); pctile_sorted(b, 75.0) - pctile_sorted(b, 25.0) }),
                    "pctile" => {
                        let p = util::opt_num(cmd, "p", 1)?.unwrap_or(50.0);
                        group_stat(&vals, &|b| { b.sort_by(f64::total_cmp); pctile_sorted(b, p) })
                    }
                    "skew" => group_stat(&vals, &|b| moments(b).0),
                    "kurt" => group_stat(&vals, &|b| moments(b).1),
                    "mad" => group_stat(&vals, &|b| {
                        b.sort_by(f64::total_cmp);
                        let m = pctile_sorted(b, 50.0);
                        let mut d: Vec<f64> = b.iter().map(|x| (x - m).abs()).collect();
                        d.sort_by(f64::total_cmp);
                        pctile_sorted(&d, 50.0)
                    }),
                    "mdev" => group_stat(&vals, &|b| { let m = util::mean(b); b.iter().map(|x| (x - m).abs()).sum::<f64>() / b.len() as f64 }),
                    "mode" => {
                        let minmode = cmd.has_opt("minmode", 3);
                        let maxmode = cmd.has_opt("maxmode", 3);
                        group_stat(&vals, &|b| {
                            b.sort_by(f64::total_cmp);
                            let mut best: Vec<f64> = Vec::new();
                            let mut bestc = 0;
                            let mut i = 0;
                            while i < b.len() {
                                let mut j = i;
                                while j < b.len() && b[j] == b[i] {
                                    j += 1;
                                }
                                if j - i > bestc {
                                    bestc = j - i;
                                    best = vec![b[i]];
                                } else if j - i == bestc {
                                    best.push(b[i]);
                                }
                                i = j;
                            }
                            if best.len() == 1 || minmode { best.first().copied().unwrap_or(SYSMISS) } else if maxmode { *best.last().unwrap() } else { SYSMISS }
                        })
                    }
                    "std" => {
                        let m = group_stat(&vals, &|b| util::mean(b));
                        let s = group_stat(&vals, &|b| if b.len() < 2 { SYSMISS } else { util::variance(b).sqrt() });
                        let mean_t = util::opt_num(cmd, "mean", 1)?.unwrap_or(0.0);
                        let sd_t = util::opt_num(cmd, "sd", 2)?.unwrap_or(1.0);
                        (0..n).map(|i| if is_missing(vals[i]) || is_missing(s[i]) { SYSMISS } else { mean_t + sd_t * (vals[i] - m[i]) / s[i] }).collect()
                    }
                    "pc" => {
                        let t = group_stat(&vals, &|b| b.iter().sum());
                        let prop = cmd.has_opt("prop", 4);
                        (0..n).map(|i| if is_missing(vals[i]) { SYSMISS } else { vals[i] / t[i] * if prop { 1.0 } else { 100.0 } }).collect()
                    }
                    "rank" => {
                        let (ids, ng) = by_ids(ds, st, &by, &mask);
                        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ng];
                        for i in 0..n {
                            if ids[i] != usize::MAX && !is_missing(vals[i]) {
                                groups[ids[i]].push(i);
                            }
                        }
                        let field = cmd.has_opt("field", 1);
                        let track = cmd.has_opt("track", 1);
                        let unique = cmd.has_opt("unique", 1);
                        let mut r = vec![SYSMISS; n];
                        for g in groups {
                            let mut idx = g.clone();
                            if field {
                                idx.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
                            } else {
                                idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]).then(a.cmp(&b)));
                            }
                            let mut k = 0;
                            while k < idx.len() {
                                let mut j = k;
                                while j < idx.len() && vals[idx[j]] == vals[idx[k]] {
                                    j += 1;
                                }
                                for t in k..j {
                                    r[idx[t]] = if unique {
                                        (t + 1) as f64
                                    } else if field || track {
                                        (k + 1) as f64
                                    } else {
                                        (k + j + 1) as f64 / 2.0
                                    };
                                }
                                k = j;
                            }
                        }
                        r
                    }
                    _ => unreachable!(),
                };
                if cmd.has_opt("missing", 3) && func == "total" {
                    // all-missing groups give missing
                }
            }
            "rowmean" | "rowtotal" | "rowmax" | "rowmin" | "rowmiss" | "rownonmiss" | "rowsd" | "rowmedian" | "rowfirst" | "rowlast" | "rowpctile" | "anycount" | "anymatch" | "anyvalue" => {
                let vars = varlist::expand(ds, &args)?;
                let cols: Vec<&Variable> = vars.iter().map(|v| ds.get_var(v).unwrap()).collect();
                let values: Vec<f64> = match cmd.opt("values", 1) {
                    Some(v) => v.split_whitespace().filter_map(|x| x.parse().ok()).collect(),
                    None => Vec::new(),
                };
                let p = util::opt_num(cmd, "p", 1)?.unwrap_or(50.0);
                for i in 0..n {
                    if !mask[i] {
                        continue;
                    }
                    let row: Vec<f64> = cols.iter().map(|c| c.num(i)).collect();
                    let ok: Vec<f64> = row.iter().cloned().filter(|x| !is_missing(*x)).collect();
                    let miss = cols.iter().filter(|c| c.data[i].is_missing() || (c.var_type.is_string() && c.str_at(i).is_empty())).count();
                    out[i] = match func.as_str() {
                        "rowmean" => if ok.is_empty() { SYSMISS } else { util::mean(&ok) },
                        "rowtotal" => if ok.is_empty() && missing_opt { SYSMISS } else { ok.iter().sum() },
                        "rowmax" => ok.iter().cloned().fold(SYSMISS, |a, x| if is_missing(a) { x } else { a.max(x) }),
                        "rowmin" => ok.iter().cloned().fold(SYSMISS, f64::min),
                        "rowmiss" => miss as f64,
                        "rownonmiss" => (cols.len() - miss) as f64,
                        "rowsd" => if ok.len() < 2 { SYSMISS } else { util::variance(&ok).sqrt() },
                        "rowmedian" | "rowpctile" => {
                            let mut s = ok.clone();
                            s.sort_by(f64::total_cmp);
                            pctile_sorted(&s, if func == "rowmedian" { 50.0 } else { p })
                        }
                        "rowfirst" => ok.first().copied().unwrap_or(SYSMISS),
                        "rowlast" => ok.last().copied().unwrap_or(SYSMISS),
                        "anycount" => row.iter().filter(|x| values.contains(x)).count() as f64,
                        "anymatch" => if row.iter().any(|x| values.contains(x)) { 1.0 } else { 0.0 },
                        "anyvalue" => {
                            let v = row[0];
                            if values.contains(&v) { v } else { SYSMISS }
                        }
                        _ => unreachable!(),
                    };
                }
            }
            "group" => {
                let vars = varlist::expand(ds, &args)?;
                let mut m = mask.clone();
                if !missing_opt {
                    util::markout(ds, &mut m, &vars);
                }
                let (ids, reps) = util::group_ids(ds, &vars, &m);
                for i in 0..n {
                    if ids[i] != usize::MAX {
                        out[i] = (ids[i] + 1) as f64;
                    }
                }
                if cmd.has_opt("label", 3) && vars.len() == 1 {
                    let v = ds.get_var(&vars[0]).unwrap();
                    let mut vl = ferrostat_core::ValueLabel::new(&newvar);
                    for (k, &r) in reps.iter().enumerate() {
                        vl.add((k + 1) as f64, util::cell_text(ds, v, r, true));
                    }
                    ds.value_labels.insert(newvar.clone(), vl);
                }
            }
            "tag" => {
                let vars = varlist::expand(ds, &args)?;
                let mut m = mask.clone();
                if !missing_opt {
                    util::markout(ds, &mut m, &vars);
                }
                let (ids, reps) = util::group_ids(ds, &vars, &m);
                out = vec![0.0; n];
                for &r in &reps {
                    out[r] = 1.0;
                }
                let _ = ids;
            }
            "seq" => {
                let from = util::opt_num(cmd, "from", 1)?.unwrap_or(1.0);
                let to = util::opt_num(cmd, "to", 1)?;
                let block = util::opt_num(cmd, "block", 1)?.unwrap_or(1.0).max(1.0) as usize;
                let (ids, _) = by_ids(ds, st, &by, &mask);
                let mut counters: std::collections::HashMap<usize, usize> = Default::default();
                for i in 0..n {
                    if ids[i] == usize::MAX {
                        continue;
                    }
                    let c = counters.entry(ids[i]).or_insert(0);
                    let k = (*c / block) as f64;
                    *c += 1;
                    out[i] = match to {
                        Some(t) if t >= from => from + k % (t - from + 1.0),
                        Some(t) => from - k % (from - t + 1.0),
                        None => from + k,
                    };
                }
            }
            "fill" => {
                let nums: Vec<f64> = args.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if nums.len() < 2 {
                    return Err(rc_err(198, "fill() requires at least two numbers"));
                }
                let step = nums[1] - nums[0];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = nums[0] + step * i as f64;
                }
            }
            "diff" => {
                let vars = varlist::expand(ds, &args)?;
                let cols: Vec<&Variable> = vars.iter().map(|v| ds.get_var(v).unwrap()).collect();
                for i in 0..n {
                    if mask[i] {
                        let first = &cols[0].data[i];
                        out[i] = if cols.iter().any(|c| c.data[i].partial_cmp(first) != Some(std::cmp::Ordering::Equal)) { 1.0 } else { 0.0 };
                    }
                }
            }
            "cut" => {
                let vals = numeric_vals(ds, st, &args, &mask)?;
                let icodes = cmd.has_opt("icodes", 1);
                let breaks: Vec<f64> = if let Some(at) = cmd.opt("at", 2) {
                    at.split(|c: char| c == ',' || c.is_whitespace()).filter_map(|x| x.parse().ok()).collect()
                } else if let Some(gk) = util::opt_num(cmd, "group", 1)? {
                    let mut s: Vec<f64> = vals.iter().cloned().filter(|x| !is_missing(*x)).collect();
                    s.sort_by(f64::total_cmp);
                    let mut b: Vec<f64> = (0..gk as usize).map(|k| s[(k * s.len()) / gk as usize]).collect();
                    b.push(s.last().copied().unwrap_or(0.0) + 1.0);
                    b.dedup();
                    b
                } else {
                    return Err(rc_err(198, "at() or group() required"));
                };
                for i in 0..n {
                    let x = vals[i];
                    if is_missing(x) {
                        continue;
                    }
                    for k in 0..breaks.len().saturating_sub(1) {
                        if x >= breaks[k] && x < breaks[k + 1] {
                            out[i] = if icodes { k as f64 } else { breaks[k] };
                        }
                    }
                }
            }
            "concat" => {
                is_str = true;
                let vars = varlist::expand(ds, &args)?;
                let punct = cmd.opt("punct", 1).map(util::unquote).unwrap_or_default();
                let fmt = cmd.opt("format", 1).map(|s| s.trim().to_string());
                for i in 0..n {
                    if mask[i] {
                        let parts: Vec<String> = vars
                            .iter()
                            .map(|v| {
                                let var = ds.get_var(v).unwrap();
                                match (&var.data[i], &fmt) {
                                    (Value::Str(s), _) => s.clone(),
                                    (x, Some(f)) => crate::format::fmt_trim(x.num(), f),
                                    (x, None) => if x.is_missing() { String::new() } else { crate::format::fmt_g(x.num(), 17) },
                                }
                            })
                            .collect();
                        sout[i] = parts.join(&punct);
                    }
                }
            }
            "ends" => {
                is_str = true;
                let v = varlist::resolve_name(ds, args.trim())?;
                let var = ds.get_var(&v).unwrap();
                let punct = cmd.opt("punct", 1).map(util::unquote).unwrap_or_else(|| " ".into());
                for i in 0..n {
                    if mask[i] {
                        let s = var.str_at(i).trim();
                        sout[i] = if cmd.has_opt("last", 1) {
                            s.rsplit(punct.as_str()).next().unwrap_or("").to_string()
                        } else if cmd.has_opt("tail", 1) {
                            s.split_once(punct.as_str()).map(|x| x.1.to_string()).unwrap_or_default()
                        } else {
                            s.split(punct.as_str()).next().unwrap_or("").to_string()
                        };
                    }
                }
            }
            "mtr" => return Err(rc_err(198, "egen mtr() not supported")),
            _ => {
                return Err(rc_err(133, format!("unknown egen function {func}()")));
            }
        }
        let var = if is_str {
            let w = sout.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let mut v = Variable::new(&newvar, util::str_type_for(w));
            v.data = sout.into_iter().map(Value::Str).collect();
            v
        } else {
            let t = ty.unwrap_or(match func.as_str() {
                "group" | "tag" | "count" | "rowmiss" | "rownonmiss" | "seq" | "anycount" | "anymatch" | "diff" => {
                    let b = util::best_numeric_type(&out);
                    if b == VarType::Byte && func == "count" { VarType::Long } else { b }
                }
                "total" | "sum" | "rowtotal" => VarType::Double,
                _ => st.default_type,
            });
            let mut v = Variable::new(&newvar, t);
            v.data = out.iter().map(|&x| crate::data::store_num(t, x)).collect();
            if func == "group" && ds.value_labels.contains_key(&newvar) {
                v.value_label_name = newvar.clone();
            }
            v
        };
        let nmiss = var.data.iter().filter(|x| x.is_missing() || x.as_string() == Some("")).count();
        ds.add_variable(var)?;
        st.data_changed = true;
        let _ = Val::Num(0.0);
        Ok(if nmiss > 0 { plural(nmiss) } else { String::new() })
    }
}

/// (skewness, kurtosis) with population moments.
pub fn moments(b: &[f64]) -> (f64, f64) {
    if b.len() < 2 {
        return (SYSMISS, SYSMISS);
    }
    let n = b.len() as f64;
    let m = b.iter().sum::<f64>() / n;
    let m2 = b.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = b.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    let m4 = b.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return (SYSMISS, SYSMISS);
    }
    (m3 / m2.powf(1.5), m4 / (m2 * m2))
}
