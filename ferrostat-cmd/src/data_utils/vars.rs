//! `ds`, `lookfor`, `varmanage` and `labelbook` (`vl` and the frame-set
//! files live in `crate::dm_w3`).

use std::collections::BTreeSet;

use ferrostat_core::{DataSet, Variable};
use ferrostat_parser::Command;

use super::{columns, glob};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("ds", Box::new(Ds)),
        ("lookfor", Box::new(Lookfor)),
        ("varmanage", Box::new(Varmanage)),
        ("labelbook", Box::new(Labelbook)),
    ]
}

fn prefix_of(full: &str, given: &str, min: usize) -> bool {
    given.len() >= min && full.starts_with(given)
}

fn crit_match(var: &Variable, crit: &str) -> Result<bool, String> {
    let crit = crit.trim();
    let (kind, pat) = crit.split_once(char::is_whitespace).unwrap_or((crit, ""));
    let pat = util::unquote(pat.trim());
    let kind = kind.to_lowercase();
    if prefix_of("type", &kind, 1) {
        return Ok(match pat.as_str() {
            "numeric" => var.var_type.is_numeric(),
            "string" => var.var_type.is_string(),
            p => glob(p, &var.var_type.type_str()),
        });
    }
    if prefix_of("varlabel", &kind, 4) {
        return Ok(glob(&pat, &var.label));
    }
    if prefix_of("vallabel", &kind, 4) {
        return Ok(glob(&pat, &var.value_label_name));
    }
    if prefix_of("format", &kind, 1) {
        return Ok(glob(&pat, &var.format));
    }
    Err(rc_err(198, format!("invalid criterion {crit}")))
}

struct Ds;

impl CommandHandler for Ds {
    fn name(&self) -> &'static str {
        "ds"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let base = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let has = cmd.opt("has", 3).map(str::to_string);
        let not = cmd.opt("not", 3).map(str::to_string);
        let mut sel = Vec::new();
        for n in &base {
            let v = ds.get_var(n).ok_or_else(|| rc_err(111, format!("variable {n} not found")))?;
            let mut ok = true;
            if let Some(h) = &has {
                ok &= crit_match(v, h)?;
            }
            if let Some(h) = &not {
                ok &= !crit_match(v, h)?;
            }
            if ok {
                sel.push(n.clone());
            }
        }
        if cmd.has_opt("alpha", 1) {
            sel.sort();
        }
        st.begin_r();
        st.r.set_str("varlist", sel.join(" "));
        if sel.is_empty() {
            return Ok(String::new());
        }
        if cmd.has_opt("detail", 1) {
            let mut out = vec![
                "              storage   display    value".to_string(),
                "variable name   type    format     label      variable label".to_string(),
                "-".repeat(79),
            ];
            for n in &sel {
                let v = ds.get_var(n).unwrap();
                out.push(format!("{:<15} {:<7} {:<10} {:<10} {}", n, v.var_type.type_str(), v.format, v.value_label_name, v.label));
            }
            return Ok(out.join("\n"));
        }
        Ok(columns(&sel))
    }
}

struct Lookfor;

impl CommandHandler for Lookfor {
    fn name(&self) -> &'static str {
        "lookfor"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words = split_words(&cmd.main);
        if words.is_empty() {
            return Err(rc_err(100, "lookfor: string required"));
        }
        let needles: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let mut hit = Vec::new();
        for v in &ds.vars {
            let name = v.name.to_lowercase();
            let label = v.label.to_lowercase();
            if needles.iter().any(|n| name.contains(n) || label.contains(n)) {
                hit.push(v);
            }
        }
        st.begin_r();
        st.r.set_str("varlist", hit.iter().map(|v| v.name.as_str()).collect::<Vec<_>>().join(" "));
        if hit.is_empty() {
            return Ok(String::new());
        }
        let mut out = vec![
            "              storage   display    value".to_string(),
            "variable name   type    format     label      variable label".to_string(),
            "-".repeat(79),
        ];
        for v in hit {
            out.push(format!("{:<15} {:<7} {:<10} {:<10} {}", v.name, v.var_type.type_str(), v.format, v.value_label_name, v.label));
        }
        Ok(out.join("\n"))
    }
}

/// Split on blanks, keeping double-quoted strings together.
pub fn split_words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    for c in s.chars() {
        match c {
            '"' => {
                q = !q;
            }
            c if c.is_whitespace() && !q => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

struct Varmanage;

impl CommandHandler for Varmanage {
    fn name(&self) -> &'static str {
        "varmanage"
    }
    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, _cmd: &Command) -> Result<String, String> {
        Ok("varmanage is an interactive window and is not available here; use rename, label, order and drop".into())
    }
}


// ---------------------------------------------------------------- labelbook

struct Labelbook;

impl CommandHandler for Labelbook {
    fn name(&self) -> &'static str {
        "labelbook"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut names: Vec<String> = if cmd.main.trim().is_empty() {
            ds.value_labels.keys().cloned().collect()
        } else {
            cmd.main.split_whitespace().map(String::from).collect()
        };
        names.sort();
        for n in &names {
            if !ds.value_labels.contains_key(n) {
                return Err(rc_err(111, format!("value label {n} not found")));
            }
        }
        let used = |n: &str| -> Vec<String> { ds.vars.iter().filter(|v| v.value_label_name == n).map(|v| v.name.clone()).collect() };
        let mut notused = Vec::new();
        let mut gaps = Vec::new();
        let mut ambiguous = Vec::new();
        for n in &names {
            let vl = &ds.value_labels[n];
            if used(n).is_empty() {
                notused.push(n.clone());
            }
            let keys: Vec<f64> = vl.mapping.keys().map(|k| k.0).collect();
            if keys.windows(2).any(|w| w[1] - w[0] != 1.0) {
                gaps.push(n.clone());
            }
            let mut seen = BTreeSet::new();
            if vl.mapping.values().any(|l| !seen.insert(l.clone())) {
                ambiguous.push(n.clone());
            }
        }
        st.begin_r();
        st.r.set_str("notused", notused.join(" "));
        st.r.set_str("gaps", gaps.join(" "));
        st.r.set_str("ambiguous", ambiguous.join(" "));
        let mut out = Vec::new();
        let rule = "-".repeat(79);
        if cmd.has_opt("problems", 4) {
            if notused.is_empty() && gaps.is_empty() && ambiguous.is_empty() {
                return Ok("  no potential problems in dataset".into());
            }
            out.push("Potential problems in dataset".to_string());
            out.push(String::new());
            out.push("      potential problem                       value labels".to_string());
            out.push(format!("      {}", "-".repeat(60)));
            if !notused.is_empty() {
                out.push(format!("      Not used by any variables               {}", notused.join(" ")));
            }
            if !gaps.is_empty() {
                out.push(format!("      Gaps in mapped values                   {}", gaps.join(" ")));
            }
            if !ambiguous.is_empty() {
                out.push(format!("      Same label for different values         {}", ambiguous.join(" ")));
            }
            return Ok(out.join("\n"));
        }
        for n in &names {
            let vl = &ds.value_labels[n];
            out.push(rule.clone());
            out.push(format!("value label {n}"));
            out.push(rule.clone());
            out.push(String::new());
            let keys: Vec<f64> = vl.mapping.keys().map(|k| k.0).collect();
            let lens: Vec<usize> = vl.mapping.values().map(|l| l.chars().count()).collect();
            let nonmiss: Vec<f64> = keys.iter().copied().filter(|k| !ferrostat_core::missing::is_missing(*k)).collect();
            out.push("      Values                         Labels".to_string());
            if let (Some(a), Some(b)) = (nonmiss.first(), nonmiss.last()) {
                out.push(format!("        Range:  [{a},{b}]                string length:  [{},{}]", lens.iter().min().unwrap_or(&0), lens.iter().max().unwrap_or(&0)));
            }
            out.push(format!("            N:  {}                  unique at full length:  {}", keys.len(), if ambiguous.contains(n) { "no" } else { "yes" }));
            out.push(String::new());
            out.push("      definition".to_string());
            for (k, l) in &vl.mapping {
                out.push(format!("        {:>10}  {}", crate::format::fmt_g(k.0, 10).trim(), l));
            }
            let u = used(n);
            out.push(String::new());
            out.push(format!("      variables:  {}", if u.is_empty() { "(not used by any variables)".to_string() } else { u.join(" ") }));
            out.push(String::new());
        }
        Ok(out.join("\n"))
    }
}

