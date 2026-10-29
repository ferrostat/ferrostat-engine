//! `vl`: named variable lists. System lists (`vlcategorical`,
//! `vlcontinuous`, `vluncertain`, `vlother`, `vldummy`) classify numeric
//! variables by their values; user lists are created from varlists and set
//! expressions of other lists. Every list is available as the global macro
//! `$name` and is also recorded in the dataset's characteristics, so that
//! `vl rebuild` restores the macros after the data are saved and used again.

use std::collections::{BTreeMap, BTreeSet};

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

const SYSTEM: [(&str, &str); 5] = [
    ("vldummy", "0/1 variables"),
    ("vlcategorical", "categorical variables"),
    ("vlcontinuous", "continuous variables"),
    ("vluncertain", "perhaps continuous, perhaps categorical variables"),
    ("vlother", "all missing or constant variables"),
];

/// Characteristic holding the names of the user lists.
const CH_USER: &str = "vl_user";
const CH_PREFIX: &str = "vl_list_";
const CH_LABEL: &str = "vl_label_";

fn is_system(n: &str) -> bool {
    SYSTEM.iter().any(|(s, _)| *s == n)
}

fn dta(ds: &mut DataSet) -> &mut BTreeMap<String, String> {
    ds.characteristics.entry("_dta".into()).or_default()
}

fn chars(ds: &DataSet) -> Option<&BTreeMap<String, String>> {
    ds.characteristics.get("_dta")
}

fn user_lists(ds: &DataSet) -> Vec<String> {
    chars(ds).and_then(|c| c.get(CH_USER)).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

fn system_lists(ds: &DataSet) -> Vec<String> {
    SYSTEM.iter().filter(|(n, _)| chars(ds).is_some_and(|c| c.contains_key(&format!("{CH_PREFIX}{n}")))).map(|(n, _)| n.to_string()).collect()
}

fn exists(ds: &DataSet, n: &str) -> bool {
    chars(ds).is_some_and(|c| c.contains_key(&format!("{CH_PREFIX}{n}")))
}

fn get(ds: &DataSet, n: &str) -> Vec<String> {
    chars(ds).and_then(|c| c.get(&format!("{CH_PREFIX}{n}"))).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

fn put(ds: &mut DataSet, st: &mut Store, n: &str, vars: &[String]) {
    let text = vars.join(" ");
    dta(ds).insert(format!("{CH_PREFIX}{n}"), text.clone());
    st.globals.insert(n.to_string(), text);
    if !is_system(n) {
        let mut u = user_lists(ds);
        if !u.iter().any(|x| x == n) {
            u.push(n.to_string());
            dta(ds).insert(CH_USER.into(), u.join(" "));
        }
    }
}

fn remove(ds: &mut DataSet, st: &mut Store, n: &str) {
    let c = dta(ds);
    c.remove(&format!("{CH_PREFIX}{n}"));
    c.remove(&format!("{CH_LABEL}{n}"));
    st.globals.remove(n);
    let u: Vec<String> = user_lists(ds).into_iter().filter(|x| x != n).collect();
    dta(ds).insert(CH_USER.into(), u.join(" "));
}

fn label_of(ds: &DataSet, n: &str) -> String {
    if let Some(l) = chars(ds).and_then(|c| c.get(&format!("{CH_LABEL}{n}"))) {
        return l.clone();
    }
    SYSTEM.iter().find(|(s, _)| *s == n).map(|(_, d)| d.to_string()).unwrap_or_default()
}

pub struct Vl;

impl CommandHandler for Vl {
    fn name(&self) -> &'static str {
        "vl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "set" => set(ds, st, cmd, rest),
            "move" => {
                // vl move (varlist) systemlist
                let close = rest.rfind(')').ok_or_else(|| rc_err(198, "vl move (varlist) vlsystemname"))?;
                let vars = varlist::expand(ds, rest[..close].trim_start_matches('('))?;
                let to = rest[close + 1..].trim();
                if !is_system(to) {
                    return Err(rc_err(198, format!("{to} is not a system variable list")));
                }
                for (n, _) in SYSTEM {
                    if n == to || !exists(ds, n) {
                        continue;
                    }
                    let kept: Vec<String> = get(ds, n).into_iter().filter(|v| !vars.contains(v)).collect();
                    put(ds, st, n, &kept);
                }
                let mut l = get(ds, to);
                l.extend(vars.iter().filter(|v| !l.contains(v)).cloned().collect::<Vec<_>>());
                let l: Vec<String> = ds.vars.iter().map(|v| v.name.clone()).filter(|v| l.contains(v)).collect();
                put(ds, st, to, &l);
                Ok(dir_table(ds, &system_lists(ds)))
            }
            "create" | "modify" => {
                let (lhs, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, format!("vl {sub} name = list expression")))?;
                let names: Vec<&str> = lhs.split_whitespace().collect();
                let [name] = names.as_slice() else {
                    return Err(rc_err(198, "one list name required"));
                };
                varlist::check_name(name)?;
                if is_system(name) {
                    return Err(rc_err(198, format!("{name} is a system variable list")));
                }
                let known = exists(ds, name);
                if sub == "create" && known {
                    return Err(rc_err(110, format!("variable list {name} already exists")));
                }
                if sub == "modify" && !known {
                    return Err(rc_err(111, format!("variable list {name} not found")));
                }
                let vars = list_expr(ds, rhs)?;
                put(ds, st, name, &vars);
                let verb = if sub == "create" { "initialized with" } else { "now contains" };
                Ok(format!("note: ${name} {verb} {} variable{}.", vars.len(), if vars.len() == 1 { "" } else { "s" }))
            }
            "label" => {
                let (name, text) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                if !exists(ds, name) {
                    return Err(rc_err(111, format!("variable list {name} not found")));
                }
                dta(ds).insert(format!("{CH_LABEL}{name}"), util::unquote(text.trim()));
                Ok(String::new())
            }
            "substitute" => {
                let (name, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "vl substitute name = factor-variable expression"))?;
                let name = name.trim();
                varlist::check_name(name)?;
                let text = substitute(ds, rhs)?;
                st.globals.insert(name.to_string(), text);
                Ok(String::new())
            }
            "drop" => {
                if let Some(v) = rest.strip_prefix('(') {
                    // remove variables from every list
                    let vars = varlist::expand(ds, v.trim_end_matches(')'))?;
                    let all: Vec<String> = system_lists(ds).into_iter().chain(user_lists(ds)).collect();
                    for n in all {
                        let kept: Vec<String> = get(ds, &n).into_iter().filter(|x| !vars.contains(x)).collect();
                        put(ds, st, &n, &kept);
                    }
                    return Ok(String::new());
                }
                let mut names: Vec<String> = rest.split(',').next().unwrap_or("").split_whitespace().map(String::from).collect();
                if cmd.has_opt("user", 1) {
                    names.extend(user_lists(ds));
                }
                if cmd.has_opt("system", 3) {
                    names.extend(system_lists(ds));
                }
                for n in names {
                    if !exists(ds, &n) {
                        return Err(rc_err(111, format!("variable list {n} not found")));
                    }
                    remove(ds, st, &n);
                }
                Ok(String::new())
            }
            "clear" => {
                let (u, s) = (cmd.has_opt("user", 1), cmd.has_opt("system", 3));
                let mut names = Vec::new();
                if u || !s {
                    names.extend(user_lists(ds));
                }
                if s || !u {
                    names.extend(system_lists(ds));
                }
                for n in names {
                    remove(ds, st, &n);
                }
                if let Some(c) = ds.characteristics.get_mut("_dta") {
                    if c.get(CH_USER).is_some_and(|s| s.trim().is_empty()) {
                        c.remove(CH_USER);
                    }
                }
                Ok(String::new())
            }
            "dir" => {
                let (u, s) = (cmd.has_opt("user", 1), cmd.has_opt("system", 3));
                let mut names = Vec::new();
                if s || !u {
                    names.extend(system_lists(ds));
                }
                if u || !s {
                    names.extend(user_lists(ds));
                }
                if names.is_empty() {
                    return Ok("(no variable lists defined)".into());
                }
                Ok(dir_table(ds, &names))
            }
            "list" => list(ds, st, cmd, rest),
            "rebuild" => {
                let existing: BTreeSet<String> = ds.vars.iter().map(|v| v.name.clone()).collect();
                let all: Vec<String> = system_lists(ds).into_iter().chain(user_lists(ds)).collect();
                let mut dropped = 0;
                for n in &all {
                    let l = get(ds, n);
                    let kept: Vec<String> = l.iter().filter(|x| existing.contains(*x)).cloned().collect();
                    dropped += l.len() - kept.len();
                    put(ds, st, n, &kept);
                }
                st.begin_r().set("k_lists", all.len() as f64);
                Ok(format!("({} variable list{} rebuilt{})", all.len(), if all.len() == 1 { "" } else { "s" }, if dropped > 0 { format!("; {dropped} dropped variable names removed") } else { String::new() }))
            }
            "" => Err(rc_err(198, "vl: subcommand required")),
            other => Err(rc_err(198, format!("vl {other}: invalid subcommand"))),
        }
    }
}

fn set(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let cat = util::opt_num(cmd, "categorical", 3)?.unwrap_or(10.0) as usize;
    let unc = util::opt_num(cmd, "uncertain", 3)?.unwrap_or(100.0) as usize;
    let dummy = cmd.has_opt("dummy", 3) && !cmd.has_opt("nodummy", 5);
    let update = cmd.has_opt("update", 3);
    if cmd.has_opt("clear", 3) {
        for n in system_lists(ds) {
            remove(ds, st, &n);
        }
    }
    let had = !system_lists(ds).is_empty();
    if had && !update && !cmd.has_opt("clear", 3) && rest.is_empty() {
        for n in system_lists(ds) {
            remove(ds, st, &n);
        }
    }
    let mut targets: Vec<String> = if rest.is_empty() { ds.vars.iter().map(|v| v.name.clone()).collect() } else { varlist::expand(ds, rest)? };
    let classified: BTreeSet<String> = system_lists(ds).iter().flat_map(|n| get(ds, n)).collect();
    if update {
        targets.retain(|v| !classified.contains(v));
    }
    let mut lists: BTreeMap<&str, Vec<String>> = SYSTEM.iter().map(|(n, _)| (*n, Vec::new())).collect();
    if had && (update || !rest.is_empty()) {
        for (n, _) in SYSTEM {
            let keep: Vec<String> = get(ds, n).into_iter().filter(|v| !targets.contains(v)).collect();
            lists.insert(n, keep);
        }
    }
    for name in &targets {
        let v = ds.get_var(name).unwrap();
        if v.var_type.is_string() {
            continue;
        }
        let mut distinct: Vec<f64> = (0..v.len()).map(|i| v.num(i)).filter(|x| !ferrostat_core::missing::is_missing(*x)).collect();
        distinct.sort_by(|a, b| a.total_cmp(b));
        distinct.dedup();
        let integer = distinct.iter().all(|x| x.fract() == 0.0);
        let nonneg = distinct.iter().all(|x| *x >= 0.0);
        let which = if distinct.len() <= 1 {
            "vlother"
        } else if dummy && distinct == [0.0, 1.0] {
            "vldummy"
        } else if !(integer && nonneg) {
            "vlcontinuous"
        } else if distinct.len() <= cat {
            "vlcategorical"
        } else if distinct.len() <= unc {
            "vluncertain"
        } else {
            "vlcontinuous"
        };
        lists.get_mut(which).unwrap().push(name.clone());
    }
    for (n, _) in SYSTEM {
        if n == "vldummy" && !dummy && lists[n].is_empty() {
            continue;
        }
        let order: Vec<String> = ds.vars.iter().map(|v| v.name.clone()).filter(|v| lists[n].contains(v)).collect();
        put(ds, st, n, &order);
    }
    let r = st.begin_r();
    for (n, _) in SYSTEM {
        r.set(&format!("k_{}", &n[2..]), lists[n].len() as f64);
    }
    Ok(dir_table(ds, &system_lists(ds)))
}

fn dir_table(ds: &DataSet, names: &[String]) -> String {
    let mut out = vec![
        "-------------------------------------------------------------------------------".to_string(),
        format!("{:<17}{:>8}  {}", "Variable list", "# vars", "Description"),
        "-------------------------------------------------------------------------------".to_string(),
    ];
    let mut prev_sys = None;
    for n in names {
        let sys = is_system(n);
        if prev_sys == Some(true) && !sys {
            out.push(String::new());
        }
        prev_sys = Some(sys);
        out.push(format!("${:<16}{:>8}  {}", n, get(ds, n).len(), label_of(ds, n)));
    }
    out.push("-------------------------------------------------------------------------------".into());
    out.join("\n")
}

fn list(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let names: Vec<String> = if rest.is_empty() {
        let (u, s) = (cmd.has_opt("user", 1), cmd.has_opt("system", 3));
        let mut v = Vec::new();
        if s || !u {
            v.extend(system_lists(ds));
        }
        if u || !s {
            v.extend(user_lists(ds));
        }
        v
    } else if let Some(v) = rest.strip_prefix('(') {
        // which lists contain these variables
        let vars = varlist::expand(ds, v.trim_end_matches(')'))?;
        let all: Vec<String> = system_lists(ds).into_iter().chain(user_lists(ds)).collect();
        let mut out = vec![format!("{:<20}{}", "Variable", "Macro")];
        for var in &vars {
            for n in all.iter().filter(|n| get(ds, n).contains(var)) {
                out.push(format!("{var:<20}${n}"));
            }
        }
        return Ok(out.join("\n"));
    } else {
        rest.split_whitespace().map(|s| s.trim_start_matches('$').to_string()).collect()
    };
    let mut out = vec![format!("{:<20}{:<18}{}", "Variable", "Macro", "Values"), "-".repeat(60)];
    let mut total = 0;
    for n in &names {
        if !exists(ds, n) {
            return Err(rc_err(111, format!("variable list {n} not found")));
        }
        let mut vars = get(ds, n);
        if cmd.has_opt("sort", 2) {
            vars.sort();
        }
        for v in vars {
            total += 1;
            let kind = match n.as_str() {
                "vlcategorical" | "vldummy" => match ds.get_var(&v) {
                    Some(var) => {
                        let mut d: Vec<f64> = (0..var.len()).map(|i| var.num(i)).filter(|x| !ferrostat_core::missing::is_missing(*x)).collect();
                        d.sort_by(|a, b| a.total_cmp(b));
                        d.dedup();
                        format!("{} levels", d.len())
                    }
                    None => String::new(),
                },
                _ => String::new(),
            };
            out.push(format!("{v:<20}${n:<17}{kind}"));
        }
    }
    out.push("-".repeat(60));
    st.begin_r().set("k", total as f64);
    Ok(out.join("\n"))
}

/// Variables of `name`, a user/system list, or of a parenthesized varlist.
fn operand(ds: &DataSet, tok: &str) -> Result<Vec<String>, String> {
    if let Some(v) = tok.strip_prefix('(') {
        let inner = v.strip_suffix(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
        let mut vars = Vec::new();
        for t in inner.split_whitespace() {
            if let Some(g) = t.strip_prefix('$') {
                vars.extend(get(ds, g));
            } else {
                vars.extend(varlist::expand(ds, t)?);
            }
        }
        return Ok(vars);
    }
    let n = tok.trim_start_matches('$');
    if exists(ds, n) {
        return Ok(get(ds, n));
    }
    Err(rc_err(111, format!("variable list {n} not found")))
}

/// `list1 + list2 - (varlist) ...`; a bare varlist is also accepted.
fn list_expr(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    // split into operands at top-level + and -
    let mut terms: Vec<(char, String)> = Vec::new();
    let (mut depth, mut cur, mut op) = (0, String::new(), '+');
    for c in text.trim().chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            '+' | '-' if depth == 0 => {
                terms.push((op, std::mem::take(&mut cur)));
                op = c;
            }
            _ => cur.push(c),
        }
    }
    terms.push((op, cur));
    let mut out: Vec<String> = Vec::new();
    for (op, t) in terms {
        let t = t.trim();
        if t.is_empty() {
            continue;
        }
        let vars = if t.starts_with('(') || (!t.contains(char::is_whitespace) && exists(ds, t.trim_start_matches('$'))) {
            operand(ds, t)?
        } else {
            let mut v = Vec::new();
            for w in t.split_whitespace() {
                if exists(ds, w.trim_start_matches('$')) {
                    v.extend(operand(ds, w)?);
                } else {
                    v.extend(varlist::expand(ds, w)?);
                }
            }
            v
        };
        if op == '+' {
            for v in vars {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        } else {
            out.retain(|x| !vars.contains(x));
        }
    }
    Ok(out)
}

/// Expand `i.list c.list list1#list2` into factor-variable notation.
fn substitute(ds: &DataSet, text: &str) -> Result<String, String> {
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        let double = tok.contains("##");
        let parts: Vec<&str> = if double { tok.split("##").collect() } else { tok.split('#').collect() };
        let mut expanded: Vec<Vec<String>> = Vec::new();
        for p in &parts {
            let (pre, name) = match p.split_once('.') {
                Some((a, b)) => (format!("{a}."), b),
                None => (String::new(), *p),
            };
            let name = name.trim_start_matches('$');
            let vars = if exists(ds, name) { get(ds, name) } else { varlist::expand(ds, name)? };
            expanded.push(vars.into_iter().map(|v| format!("{pre}{v}")).collect());
        }
        let mut combos: Vec<String> = vec![String::new()];
        for e in &expanded {
            let mut next = Vec::new();
            for c in &combos {
                for v in e {
                    next.push(if c.is_empty() { v.clone() } else { format!("{c}{}{v}", if double { "##" } else { "#" }) });
                }
            }
            combos = next;
        }
        out.extend(combos);
    }
    Ok(out.join(" "))
}

