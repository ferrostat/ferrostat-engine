//! Describing and listing data: describe, count, list, duplicates, isid,
//! levelsof, codebook, misstable.

use ferrostat_core::missing::is_missing;
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::{Results, Store};
use crate::util::{self, cell_text};
use crate::varlist;

pub struct Describe;

impl CommandHandler for Describe {
    fn name(&self) -> &'static str {
        "describe"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if let Some(path) = &cmd.using {
            let path = if std::path::Path::new(path).extension().is_some() { path.clone() } else { format!("{path}.dta") };
            let other = ferrostat_core::dta::DtaReader::open(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
            let vars = if cmd.main.trim().is_empty() { other.varlist() } else { varlist::expand(&other, &cmd.main)? };
            if cmd.has_opt("replace", 7) {
                return describe_replace(ds, &other, &vars, st);
            }
            return describe(&other, &vars, cmd, st, Some(&path));
        }
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        if cmd.has_opt("replace", 7) {
            let src = ds.clone();
            return describe_replace(ds, &src, &vars, st);
        }
        describe(ds, &vars, cmd, st, None)
    }
}

/// `describe, replace`: replace the data with one observation per described
/// variable (position, name, type, isnumeric, format, vallab, varlab).
fn describe_replace(ds: &mut DataSet, src: &DataSet, vars: &[String], st: &mut Store) -> Result<String, String> {
    use ferrostat_core::{VarType, Variable};
    let mut rows = Vec::new();
    for v in vars {
        let k = src.vars.iter().position(|x| &x.name == v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
        rows.push((k + 1, &src.vars[k]));
    }
    let str_col = |name: &str, label: &str, vals: Vec<String>| {
        let w = vals.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let mut var = Variable::new(name, VarType::Str(w));
        var.format = format!("%{w}s");
        var.label = label.to_string();
        var.data = vals.into_iter().map(Value::Str).collect();
        var
    };
    let mut out = DataSet::new();
    let mut pos = Variable::new("position", VarType::Int);
    pos.format = "%8.0g".into();
    pos.label = "Variable number".into();
    pos.data = rows.iter().map(|(k, _)| Value::Int(*k as i16)).collect();
    let mut isnum = Variable::new("isnumeric", VarType::Byte);
    isnum.format = "%8.0g".into();
    isnum.label = "Flag for numeric variables".into();
    isnum.data = rows.iter().map(|(_, v)| Value::Byte(v.var_type.is_numeric() as i8)).collect();
    let cols = vec![
        pos,
        str_col("name", "Variable name", rows.iter().map(|(_, v)| v.name.clone()).collect()),
        str_col("type", "Storage type", rows.iter().map(|(_, v)| v.var_type.type_str()).collect()),
        isnum,
        str_col("format", "Display format", rows.iter().map(|(_, v)| v.format.clone()).collect()),
        str_col("vallab", "Value label", rows.iter().map(|(_, v)| v.value_label_name.clone()).collect()),
        str_col("varlab", "Variable label", rows.iter().map(|(_, v)| v.label.clone()).collect()),
    ];
    out.n_obs = rows.len();
    for c in cols {
        out.add_variable(c)?;
    }
    *ds = out;
    st.data_changed = true;
    st.filename.clear();
    Ok(String::new())
}

fn describe(ds: &DataSet, vars: &[String], cmd: &Command, st: &mut Store, file: Option<&str>) -> Result<String, String> {
    let changed = st.data_changed;
    let r = st.begin_r();
    r.set("N", ds.n_obs as f64);
    r.set("k", ds.n_vars() as f64);
    let width: usize = ds.vars.iter().map(|v| v.var_type.storage_width()).sum();
    r.set("width", width as f64);
    r.set("changed", if changed { 1.0 } else { 0.0 });
    st.r.set_str("varlist", vars.join(" "));
    if cmd.has_opt("varlist", 4) {
        return Ok(String::new());
    }
    let mut out = Vec::new();
    if cmd.has_opt("simple", 2) {
        return Ok(vars.join("  "));
    }
    out.push(match file {
        Some(f) => format!("Contains data from {f}"),
        None => "Contains data".to_string(),
    });
    out.push(format!(" Observations:{:>14}{:>20}{}", ds.n_obs, "", ds.datalabel));
    out.push(format!("    Variables:{:>14}{:>20}{}", ds.n_vars(), "", ds.time_stamp));
    if cmd.has_opt("short", 1) {
        if !ds.sort_by.is_empty() {
            out.push(format!("Sorted by: {}", ds.sort_by.join("  ")));
        }
        return Ok(out.join("\n"));
    }
    // `fullnames` widens the name column instead of abbreviating with `~`;
    // `numbers` prefixes each row with the variable's position
    let full = cmd.has_opt("fullnames", 1);
    let numbers = cmd.has_opt("numbers", 1);
    let nw = if full { vars.iter().map(|v| v.chars().count()).max().unwrap_or(0).max(15) } else { 15 };
    let pre = if numbers { "     " } else { "" };
    let bar = "-".repeat(79 + nw - 15 + pre.len());
    out.push(bar.clone());
    let hw = nw - 1;
    out.push(format!("{}{:<hw$}Storage   Display    Value", if numbers { "  #  " } else { "" }, "Variable"));
    out.push(format!("{pre}{:<hw$}   type    format    label      Variable label", "    name"));
    out.push(bar.clone());
    for v in vars {
        let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
        let num = if numbers { format!("{:>3}  ", ds.vars.iter().position(|x| &x.name == v).map_or(0, |k| k + 1)) } else { String::new() };
        out.push(format!(
            "{num}{:<nw$} {:<7} {:<10} {:<10} {}",
            if !full && var.name.chars().count() > 15 { format!("{}~", var.name.chars().take(14).collect::<String>()) } else { var.name.clone() },
            var.var_type.type_str(),
            var.format,
            var.value_label_name,
            var.label
        ));
    }
    out.push(bar);
    if !ds.sort_by.is_empty() {
        out.push(format!("Sorted by: {}", ds.sort_by.join("  ")));
    }
    Ok(out.join("\n"))
}

pub struct Count;

impl CommandHandler for Count {
    fn name(&self) -> &'static str {
        "count"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let n = mask.iter().filter(|&&m| m).count();
        st.begin_r().set("N", n as f64);
        Ok(format!("  {n}"))
    }
}

pub struct List;

impl CommandHandler for List {
    fn name(&self) -> &'static str {
        "list"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mask = util::touse(ds, st, cmd)?;
        let labels = !cmd.has_opt("nolabel", 3);
        let clean = cmd.has_opt("clean", 5);
        let noobs = cmd.has_opt("noobs", 5);
        let sep: usize = cmd.opt("separator", 3).and_then(|s| s.trim().parse().ok()).unwrap_or(5);
        let rows = util::rows(&mask);
        let mut cols: Vec<Vec<String>> = Vec::new();
        let mut right: Vec<bool> = Vec::new();
        for v in &vars {
            let var = ds.get_var(v).unwrap();
            cols.push(rows.iter().map(|&i| cell_text(ds, var, i, labels)).collect());
            let left_fmt = var.format.starts_with("%-");
            let labelled = labels && !var.value_label_name.is_empty();
            right.push(!(left_fmt || labelled));
        }
        let widths: Vec<usize> = vars
            .iter()
            .zip(&cols)
            .map(|(n, c)| c.iter().map(|s| s.chars().count()).max().unwrap_or(0).max(n.chars().count()))
            .collect();
        let fmt_cell = |s: &str, w: usize, r: bool| if r { format!("{s:>w$}") } else { format!("{s:<w$}") };
        let obsw = rows.last().map_or(1, |r| (r + 1).to_string().len()) + 1;
        let lead = if noobs { String::new() } else { " ".repeat(obsw + 1) };
        let inner: usize = widths.iter().sum::<usize>() + 3 * widths.len() - 1;
        let mut out = Vec::new();
        let header: Vec<String> = vars.iter().zip(&widths).zip(&right).map(|((n, w), r)| fmt_cell(n, *w, *r)).collect();
        if clean {
            out.push(format!("{lead} {}", header.join("   ")));
        } else {
            out.push(format!("{lead}  +{}+", "-".repeat(inner)));
            out.push(format!("{lead}  | {} |", header.join("   ")));
            out.push(format!("{lead}  |{}|", "-".repeat(inner)));
        }
        for (k, &i) in rows.iter().enumerate() {
            if !clean && k > 0 && sep > 0 && k % sep == 0 {
                out.push(format!("{lead}  |{}|", "-".repeat(inner)));
            }
            let cells: Vec<String> = cols.iter().zip(&widths).zip(&right).map(|((c, w), r)| fmt_cell(&c[k], *w, *r)).collect();
            let obs = if noobs { String::new() } else { format!("{:>w$}.", i + 1, w = obsw) };
            if clean {
                out.push(format!("{obs}  {}", cells.join("   ")));
            } else {
                out.push(format!("{obs}  | {} |", cells.join("   ")));
            }
        }
        if !clean {
            out.push(format!("{lead}  +{}+", "-".repeat(inner)));
        }
        Ok(out.join("\n"))
    }
}

/// Key text of observation i over vars (for duplicates/isid/levelsof).
fn key(ds: &DataSet, vars: &[String], i: usize) -> Vec<String> {
    vars.iter()
        .map(|v| {
            let var = ds.get_var(v).unwrap();
            match &var.data[i] {
                Value::Str(s) => s.clone(),
                x => format!("{:e}", x.num()),
            }
        })
        .collect()
}

pub struct Duplicates;

impl CommandHandler for Duplicates {
    fn name(&self) -> &'static str {
        "duplicates"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut words = cmd.main.split_whitespace();
        let sub = words.next().ok_or_else(|| rc_err(198, "subcommand required"))?.to_lowercase();
        let rest: Vec<&str> = words.collect();
        let vars = if rest.is_empty() { ds.varlist() } else { varlist::expand(ds, &rest.join(" "))? };
        let mask = util::touse(ds, st, cmd)?;
        let mut counts: std::collections::HashMap<Vec<String>, usize> = Default::default();
        let mut first: std::collections::HashMap<Vec<String>, usize> = Default::default();
        for i in 0..ds.n_obs {
            if mask[i] {
                let k = key(ds, &vars, i);
                *counts.entry(k.clone()).or_default() += 1;
                first.entry(k).or_insert(i);
            }
        }
        match sub.as_str() {
            "report" => {
                let mut by_copies: std::collections::BTreeMap<usize, usize> = Default::default();
                for c in counts.values() {
                    *by_copies.entry(*c).or_default() += c;
                }
                let mut out = vec![format!("Duplicates in terms of {}", if rest.is_empty() { "all variables".into() } else { vars.join(" ") }), String::new()];
                out.push("--------------------------------------".into());
                out.push("   Copies | Observations       Surplus".into());
                out.push("----------+---------------------------".into());
                for (c, n) in &by_copies {
                    out.push(format!("{:>9} | {:>12} {:>13}", c, n, n - n / c));
                }
                out.push("--------------------------------------".into());
                let unique = counts.len();
                let r = st.begin_r();
                r.set("N", mask.iter().filter(|&&m| m).count() as f64);
                r.set("unique_value", unique as f64);
                Ok(out.join("\n"))
            }
            "list" | "examples" => {
                let mut out = Vec::new();
                for i in 0..ds.n_obs {
                    if mask[i] {
                        let k = key(ds, &vars, i);
                        if counts[&k] > 1 && (sub == "list" || first[&k] == i) {
                            out.push(format!("{:>6}. {}", i + 1, vars.iter().map(|v| cell_text(ds, ds.get_var(v).unwrap(), i, true)).collect::<Vec<_>>().join("  ")));
                        }
                    }
                }
                Ok(out.join("\n"))
            }
            "tag" => {
                let name = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
                let vals: Vec<f64> = (0..ds.n_obs).map(|i| if mask[i] { (counts[&key(ds, &vars, i)] - 1) as f64 } else { ferrostat_core::SYSMISS }).collect();
                let mut v = util::new_double(&name, vals);
                v.var_type = ferrostat_core::VarType::Int;
                v.data = v.data.iter().map(|x| util::typed_value(ferrostat_core::VarType::Int, x.num())).collect();
                ds.add_variable(v)?;
                Ok(String::new())
            }
            "drop" => {
                if rest.is_empty() == false && !cmd.has_opt("force", 5) {
                    return Err(rc_err(198, "force option required with duplicates drop varlist"));
                }
                let keep: Vec<bool> = (0..ds.n_obs).map(|i| !mask[i] || first[&key(ds, &vars, i)] == i).collect();
                let before = ds.n_obs;
                ds.keep_if(&keep)?;
                Ok(format!("({} observation{} deleted)", before - ds.n_obs, if before - ds.n_obs == 1 { "" } else { "s" }))
            }
            _ => Err(rc_err(198, format!("duplicates {sub}: invalid subcommand"))),
        }
    }
}

pub struct Isid;

impl CommandHandler for Isid {
    fn name(&self) -> &'static str {
        "isid"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut seen = std::collections::HashSet::new();
        for i in 0..ds.n_obs {
            let k = key(ds, &vars, i);
            let miss = vars.iter().any(|v| ds.get_var(v).unwrap().data[i].is_missing() || ds.get_var(v).unwrap().str_at(i).is_empty() && ds.get_var(v).unwrap().var_type.is_string());
            if miss && !cmd.has_opt("missok", 4) {
                return Err(rc_err(459, format!("variable{} {} should never be missing", if vars.len() > 1 { "s" } else { "" }, vars.join(" "))));
            }
            if !seen.insert(k) {
                return Err(rc_err(459, format!("variable{} {} do{} not uniquely identify the observations", if vars.len() > 1 { "s" } else { "" }, vars.join(" "), if vars.len() > 1 { "" } else { "es" })));
            }
        }
        if cmd.has_opt("sort", 4) {
            ds.sort(&vars, true)?;
        }
        Ok(String::new())
    }
}

pub struct Levelsof;

impl CommandHandler for Levelsof {
    fn name(&self) -> &'static str {
        "levelsof"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        if !cmd.has_opt("missing", 4) {
            util::markout(ds, &mut mask, &[v.clone()]);
        }
        let var = ds.get_var(&v).unwrap();
        let sep = cmd.opt("separate", 3).map(util::unquote).unwrap_or_else(|| " ".into());
        let clean = cmd.has_opt("clean", 5);
        let levels: Vec<String> = if var.var_type.is_string() {
            let mut s: Vec<String> = util::rows(&mask).iter().map(|&i| var.str_at(i).to_string()).collect();
            s.sort();
            s.dedup();
            s.into_iter().map(|x| if clean && !x.contains(' ') { x } else if clean { x } else { format!("`\"{x}\"'") }).collect()
        } else {
            let mut x: Vec<f64> = util::rows(&mask).iter().map(|&i| var.num(i)).collect();
            x.sort_by(f64::total_cmp);
            x.dedup();
            x.into_iter().map(|v| crate::format::fmt_g(v, 17)).collect()
        };
        let text = levels.join(&sep);
        st.begin_r().set_str("levels", text.clone());
        st.r.set("r", levels.len() as f64);
        if let Some(l) = cmd.opt("local", 3) {
            st.set_local(l.trim(), text.clone());
        }
        Ok(text)
    }
}

pub struct Codebook;

impl CommandHandler for Codebook {
    fn name(&self) -> &'static str {
        "codebook"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mask = util::touse(ds, st, cmd)?;
        let mut out = Vec::new();
        for v in vars {
            let var = ds.get_var(&v).unwrap();
            out.push("-".repeat(79));
            out.push(format!("{:<40}{:>39}", v, var.label));
            out.push("-".repeat(79));
            out.push(format!("{:>20}: {}", "Type", if var.var_type.is_string() { format!("String ({})", var.var_type.type_str()) } else { format!("Numeric ({})", var.var_type.type_str()) }));
            let rows = util::rows(&mask);
            let nmiss = rows.iter().filter(|&&i| var.data[i].is_missing() || (var.var_type.is_string() && var.str_at(i).is_empty())).count();
            let mut uniq: Vec<String> = rows.iter().map(|&i| cell_text(ds, var, i, false)).collect();
            uniq.sort();
            uniq.dedup();
            if !var.var_type.is_string() {
                let x: Vec<f64> = rows.iter().map(|&i| var.num(i)).filter(|x| !is_missing(*x)).collect();
                let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                out.push(format!("{:>20}: [{},{}]{:>20}units: 1", "Range", crate::format::fmt_g(lo, 9), crate::format::fmt_g(hi, 9), ""));
            }
            out.push(format!("{:>20}: {}{:>20}missing .: {}/{}", "Unique values", uniq.len(), "", nmiss, rows.len()));
        }
        Ok(out.join("\n"))
    }
}

pub struct Misstable;

impl CommandHandler for Misstable {
    fn name(&self) -> &'static str {
        "misstable"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut w = cmd.main.split_whitespace();
        let sub = w.next().unwrap_or("summarize").to_lowercase();
        let rest: Vec<&str> = w.collect();
        let vars = if rest.is_empty() { ds.varlist() } else { varlist::expand(ds, &rest.join(" "))? };
        let mask = util::touse(ds, st, cmd)?;
        if !sub.starts_with("sum") {
            return Err(rc_err(198, format!("misstable {sub} not supported")));
        }
        let mut out = vec![
            "                                                               Obs<.".to_string(),
            "                                                +------------------------------".to_string(),
            "               |                                | Unique".to_string(),
            "      Variable |     Obs=.     Obs>.     Obs<.  | values        Min         Max".to_string(),
            "  -------------+--------------------------------+------------------------------".to_string(),
        ];
        for v in vars {
            let var = ds.get_var(&v).unwrap();
            if var.var_type.is_string() {
                continue;
            }
            let x: Vec<f64> = util::rows(&mask).iter().map(|&i| var.num(i)).collect();
            let eq = x.iter().filter(|&&v| ferrostat_core::missing::missing_code(v) == Some(0)).count();
            let gt = x.iter().filter(|&&v| ferrostat_core::missing::missing_code(v).map_or(false, |c| c > 0)).count();
            if eq + gt == 0 && !cmd.has_opt("all", 3) {
                continue;
            }
            let ok: Vec<f64> = x.iter().cloned().filter(|v| !is_missing(*v)).collect();
            let mut u = ok.clone();
            u.sort_by(f64::total_cmp);
            u.dedup();
            let lo = ok.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = ok.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            out.push(format!(
                "  {:>12} | {:>9} {:>9} {:>9}  | {:>6} {:>10} {:>11}",
                v,
                eq,
                if gt > 0 { gt.to_string() } else { String::new() },
                ok.len(),
                u.len(),
                crate::format::fmt_g(lo, 9),
                crate::format::fmt_g(hi, 9)
            ));
        }
        out.push("  -----------------------------------------------------------------------------".into());
        st.r = Results::new();
        Ok(out.join("\n"))
    }
}
