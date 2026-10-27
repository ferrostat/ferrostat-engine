//! `separate varname, by(byvar|exp)`: one new variable per group, holding
//! the values of `varname` for that group and missing elsewhere.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, Variable};
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::util;

pub struct Separate;

impl CommandHandler for Separate {
    fn name(&self) -> &'static str {
        "separate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.len() != 1 {
            return Err(rc_err(if vars.is_empty() { 100 } else { 103 }, if vars.is_empty() { "varlist required" } else { "too many variables specified" }));
        }
        let var = vars[0].clone();
        let by = cmd.opt("by", 2).map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| rc_err(198, "option by() required"))?.to_string();
        let src = ds.get_var(&var).unwrap().clone();
        if src.var_type.is_string() {
            return Err(rc_err(109, "string variables not allowed"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let short = cmd.has_opt("shortlabel", 5);
        let vshort = cmd.has_opt("veryshortlabel", 5);
        let seq = cmd.has_opt("sequential", 3);
        let stub = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| var.clone());
        let n = ds.n_obs;
        let is_var = !by.contains(|c: char| !(c.is_alphanumeric() || c == '_')) && crate::varlist::resolve_name(ds, &by).is_ok();
        // (suffix, label, membership per observation)
        let mut groups: Vec<(String, String, Vec<bool>)> = Vec::new();
        if is_var {
            let byname = crate::varlist::resolve_name(ds, &by)?;
            let bv = ds.get_var(&byname).unwrap();
            let is_str = bv.var_type.is_string();
            let keys: Vec<Option<String>> = (0..n)
                .map(|i| match &bv.data[i] {
                    Value::Str(s) => Some(s.clone()),
                    x => {
                        let v = x.num();
                        if is_missing(v) { None } else { Some(fmt_g(v, 16)) }
                    }
                })
                .collect();
            let mut levels: Vec<(f64, String)> = Vec::new();
            for i in 0..n {
                if let (true, Some(k)) = (mask[i], &keys[i]) {
                    if !levels.iter().any(|(_, l)| l == k) {
                        levels.push((if is_str { 0.0 } else { bv.data[i].num() }, k.clone()));
                    }
                }
            }
            if is_str {
                levels.sort_by(|a, b| a.1.cmp(&b.1));
            } else {
                levels.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            let numeric_names = !is_str && !seq && levels.iter().all(|(v, _)| *v >= 0.0 && v.fract() == 0.0);
            let vl = ds.value_labels.get(&bv.value_label_name);
            for (k, (v, key)) in levels.iter().enumerate() {
                let suffix = if numeric_names { key.clone() } else { (k + 1).to_string() };
                let lab = if is_str { key.clone() } else { vl.and_then(|l| l.lookup(*v)).map(str::to_string).unwrap_or_else(|| key.clone()) };
                let label = if vshort {
                    lab
                } else if short {
                    format!("{byname} == {lab}")
                } else {
                    format!("{var}, {byname} == {lab}")
                };
                let member = (0..n).map(|i| keys[i].as_deref() == Some(key.as_str())).collect();
                groups.push((suffix, label, member));
            }
        } else {
            let e = parse_expression(&by).map_err(|e| rc_err(198, e.message))?;
            let all = vec![true; n];
            let vals = util::eval_col(ds, st, &e, &all)?;
            let not = format!("!({by})");
            for (suffix, truth, text) in [("0", false, not), ("1", true, by.clone())] {
                let label = if short || vshort { text } else { format!("{var}, {text}") };
                let member = vals.iter().map(|&v| (v != 0.0) == truth).collect();
                groups.push((suffix.to_string(), label, member));
            }
        }
        let names: Vec<String> = groups.iter().map(|(s, _, _)| format!("{stub}{s}")).collect();
        for nm in &names {
            crate::varlist::check_name(nm)?;
            if ds.var_index.contains_key(nm) {
                return Err(rc_err(110, format!("variable {nm} already defined")));
            }
        }
        let mut out = vec![
            String::new(),
            "Variable      Storage   Display    Value".to_string(),
            "    name         type    format    label      Variable label".to_string(),
            "-".repeat(79),
        ];
        for ((_, label, member), nm) in groups.iter().zip(&names) {
            let mut v = Variable::new(nm.clone(), src.var_type);
            v.format = src.format.clone();
            v.label = label.chars().take(80).collect();
            v.data = (0..n).map(|i| if mask[i] && member[i] { src.data[i].clone() } else { util::typed_value(src.var_type, SYSMISS) }).collect();
            out.push(format!("{:<15} {:<7} {:<10} {:<10} {}", nm, v.var_type.type_str(), v.format, v.value_label_name, v.label));
            ds.add_variable(v)?;
        }
        st.data_changed = true;
        st.begin_r().set_str("varlist", names.join(" "));
        Ok(out.join("\n"))
    }
}
