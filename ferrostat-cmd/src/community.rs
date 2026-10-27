//! Small community commands, reimplemented from their documented behaviour:
//! `distinct` (distinct values per variable or jointly), `mdesc` (missing
//! values table) and `winsor2` (winsorizing or trimming at percentiles).

use std::collections::HashSet;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::{util, varlist};

/// A hashable key for one observation's value.
fn key(ds: &DataSet, var: &str, i: usize) -> String {
    let v = ds.get_var(var).unwrap();
    if v.var_type.is_string() { format!("s{}", util::cell_text(ds, v, i, false)) } else { format!("n{:?}", v.num(i)) }
}

fn is_miss(ds: &DataSet, var: &str, i: usize) -> bool {
    let v = ds.get_var(var).unwrap();
    if v.var_type.is_string() { util::cell_text(ds, v, i, false).is_empty() } else { is_missing(v.num(i)) }
}

pub struct Distinct;

impl CommandHandler for Distinct {
    fn name(&self) -> &'static str {
        "distinct"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mask = util::touse(ds, st, cmd)?;
        let missing = cmd.has_opt("missing", 2);
        let joint = cmd.has_opt("joint", 1);
        let min = cmd.opt("minimum", 3).and_then(|v| v.trim().parse::<usize>().ok());
        let max = cmd.opt("maximum", 3).and_then(|v| v.trim().parse::<usize>().ok());
        let rows = util::rows(&mask);
        let count = |vs: &[String]| -> (usize, usize) {
            let mut seen = HashSet::new();
            let mut n = 0;
            for &i in &rows {
                if !missing && vs.iter().any(|v| is_miss(ds, v, i)) {
                    continue;
                }
                n += 1;
                seen.insert(vs.iter().map(|v| key(ds, v, i)).collect::<Vec<_>>().join("\u{1}"));
            }
            (n, seen.len())
        };
        let mut lines: Vec<(String, usize, usize)> = Vec::new();
        if joint {
            let (n, k) = count(&vars);
            lines.push(("(jointly)".into(), n, k));
        } else {
            for v in &vars {
                let (n, k) = count(std::slice::from_ref(v));
                lines.push((v.clone(), n, k));
            }
        }
        let shown: Vec<&(String, usize, usize)> = lines.iter().filter(|(_, _, k)| min.is_none_or(|m| *k >= m) && max.is_none_or(|m| *k <= m)).collect();
        let w = shown.iter().map(|l| l.0.len()).max().unwrap_or(5).max(5);
        let mut out = vec![
            String::new(),
            format!("{:w$}|        Observations", "", w = w + 2),
            format!("{:w$}|      total   distinct", "", w = w + 2),
            format!("{}+----------------------", "-".repeat(w + 2)),
        ];
        for (name, n, k) in &shown {
            out.push(format!(" {name:>w$} |  {:>9}  {:>9}", util::commas(*n), util::commas(*k), w = w));
        }
        if let Some((_, n, k)) = lines.last() {
            let r = st.begin_r();
            r.set("N", *n as f64);
            r.set("ndistinct", *k as f64);
        }
        Ok(out.join("\n"))
    }
}

pub struct Mdesc;

impl CommandHandler for Mdesc {
    fn name(&self) -> &'static str {
        "mdesc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mask = util::touse(ds, st, cmd)?;
        let rows = util::rows(&mask);
        let total = rows.len();
        let mut out = vec![
            String::new(),
            "    Variable    |     Missing          Total     Percent Missing".to_string(),
            "----------------+-----------------------------------------------".to_string(),
        ];
        let mut last = (0usize, 0.0);
        for v in &vars {
            let miss = rows.iter().filter(|&&i| is_miss(ds, v, i)).count();
            let pct = if total > 0 { 100.0 * miss as f64 / total as f64 } else { SYSMISS };
            out.push(format!("{:>15} | {:>11}    {:>11}    {:>11.2}", crate::summarize::abbrev(v, 15), util::commas(miss), util::commas(total), pct));
            last = (miss, pct);
        }
        out.push("----------------+-----------------------------------------------".to_string());
        let r = st.begin_r();
        r.set("miss", last.0 as f64);
        r.set("notmiss", (total - last.0) as f64);
        r.set("total", total as f64);
        r.set("percent", last.1);
        Ok(out.join("\n"))
    }
}

pub struct Winsor2;

impl CommandHandler for Winsor2 {
    fn name(&self) -> &'static str {
        "winsor2"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let trim = cmd.has_opt("trim", 2);
        let replace = cmd.has_opt("replace", 7);
        let suffix = cmd.opt("suffix", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| if trim { "_tr".into() } else { "_w".into() });
        if replace && cmd.opt("suffix", 1).is_some() {
            return Err(rc_err(198, "options suffix() and replace may not be combined"));
        }
        let (lo, hi) = match cmd.opt("cuts", 1) {
            Some(c) => {
                let v: Vec<f64> = c.split(|ch: char| ch.is_whitespace() || ch == ',').filter(|x| !x.is_empty()).map(|x| x.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| rc_err(198, "cuts() invalid"))?;
                if v.len() != 2 || !(0.0..=100.0).contains(&v[0]) || !(0.0..=100.0).contains(&v[1]) || v[0] > v[1] {
                    return Err(rc_err(198, "cuts(#1 #2) requires 0 <= #1 <= #2 <= 100"));
                }
                (v[0], v[1])
            }
            None => (1.0, 99.0),
        };
        let by: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let mask = util::touse(ds, st, cmd)?;
        let (gids, greps) = if by.is_empty() { (vec![0; ds.n_obs], vec![0]) } else { util::group_ids(ds, &by, &mask) };
        for v in &vars {
            let src = ds.get_var(v).unwrap().clone();
            if src.var_type.is_string() {
                return Err(rc_err(109, format!("{v}: string variables not allowed")));
            }
            let target = if replace { v.clone() } else { format!("{v}{suffix}") };
            if !replace && ds.get_var(&target).is_some() {
                return Err(rc_err(110, format!("{target} already defined")));
            }
            // cut points per group
            let mut vals: Vec<Vec<f64>> = vec![Vec::new(); greps.len()];
            for i in 0..ds.n_obs {
                let x = src.num(i);
                if mask[i] && !is_missing(x) {
                    vals[gids[i]].push(x);
                }
            }
            let cuts: Vec<(f64, f64)> = vals
                .iter_mut()
                .map(|g| {
                    g.sort_by(|a, b| a.total_cmp(b));
                    (util::pctile_sorted(g, lo), util::pctile_sorted(g, hi))
                })
                .collect();
            let data: Vec<f64> = (0..ds.n_obs)
                .map(|i| {
                    let x = src.num(i);
                    if !mask[i] {
                        return if replace { x } else { SYSMISS };
                    }
                    if is_missing(x) {
                        return x;
                    }
                    let (a, b) = cuts[gids[i]];
                    if trim {
                        if x < a || x > b { SYSMISS } else { x }
                    } else {
                        x.clamp(a, b)
                    }
                })
                .collect();
            let ty = if src.var_type == VarType::Double { VarType::Double } else { VarType::Float };
            if replace {
                let var = ds.get_var_mut(v).unwrap();
                if !matches!(var.var_type, VarType::Double | VarType::Float) {
                    var.var_type = ty;
                }
                let t = var.var_type;
                var.data = data.iter().map(|&x| crate::data::store_num(t, x)).collect();
            } else {
                let mut var = Variable::new(&target, ty);
                var.format = src.format.clone();
                var.label = format!("{}{}", if src.label.is_empty() { v.clone() } else { src.label.clone() }, if trim { format!("-Trim({lo},{hi})") } else { format!("-Winsor({lo},{hi})") });
                var.data = data.iter().map(|&x| crate::data::store_num(ty, x)).collect();
                ds.add_variable(var)?;
            }
        }
        st.data_changed = true;
        Ok(String::new())
    }
}
