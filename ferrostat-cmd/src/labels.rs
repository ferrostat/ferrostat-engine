//! Labels and type conversions: label, encode/decode, destring/tostring,
//! recode, mvencode/mvdecode, split.

use ferrostat_core::missing::{is_missing, missing_code, missing_name, SYSMISS};
use ferrostat_core::{DataSet, Value, ValueLabel, VarType, Variable};
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util::{self, best_numeric_type, unquote};
use crate::varlist;

fn parse_num(t: &str) -> Option<f64> {
    let t = t.trim();
    if t == "." {
        return Some(SYSMISS);
    }
    if let Some(c) = t.strip_prefix('.') {
        if c.len() == 1 {
            return ferrostat_core::missing::missing_letter(c.chars().next()?);
        }
    }
    t.parse::<f64>().ok()
}

pub struct Label;

impl CommandHandler for Label {
    fn name(&self) -> &'static str {
        "label"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // re-split the raw text: label subcommands need quoted strings intact
        let body = cmd.raw.trim();
        let body = body.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        let (body, opts) = match ferrostat_parser::parser::find_top_level(body, ',') {
            Some(i) => (&body[..i], body[i + 1..].trim().to_lowercase()),
            None => (body, String::new()),
        };
        let words = split_top_level_words(body);
        let sub = words.first().map(|s| s.to_lowercase()).unwrap_or_default();
        let rest = &words[1.min(words.len())..];
        match sub.as_str() {
            "var" | "vari" | "varia" | "variab" | "variabl" | "variable" => {
                let v = varlist::resolve_name(ds, rest.first().ok_or_else(|| rc_err(198, "varname required"))?)?;
                let text = rest.get(1).map(|s| unquote(s)).unwrap_or_default();
                if text.len() > 80 {
                    // labels are truncated at 80 characters
                }
                ds.get_var_mut(&v).unwrap().label = text.chars().take(80).collect();
                Ok(String::new())
            }
            "da" | "dat" | "data" => {
                ds.datalabel = rest.first().map(|s| unquote(s)).unwrap_or_default().chars().take(80).collect();
                Ok(String::new())
            }
            "de" | "def" | "defi" | "defin" | "define" => {
                let name = rest.first().ok_or_else(|| rc_err(198, "label name required"))?.clone();
                varlist::check_name(&name)?;
                let add = opts.contains("add");
                let modify = opts.contains("modify");
                let replace = opts.contains("replace");
                if ds.value_labels.contains_key(&name) && !(add || modify || replace) {
                    return Err(rc_err(110, format!("label {name} already defined")));
                }
                let mut vl = if replace || !ds.value_labels.contains_key(&name) {
                    ValueLabel::new(&name)
                } else {
                    ds.value_labels[&name].clone()
                };
                let pairs = &rest[1..];
                if pairs.len() % 2 != 0 {
                    return Err(rc_err(198, "invalid syntax"));
                }
                for p in pairs.chunks(2) {
                    let v = parse_num(&p[0]).ok_or_else(|| rc_err(198, format!("{} invalid value", p[0])))?;
                    if v.fract() != 0.0 && !is_missing(v) {
                        return Err(rc_err(198, "may not label noninteger values"));
                    }
                    if add && !modify && vl.lookup(v).is_some() {
                        return Err(rc_err(180, format!("{} already labeled", p[0])));
                    }
                    vl.add(v, unquote(&p[1]));
                }
                ds.value_labels.insert(name, vl);
                Ok(String::new())
            }
            "val" | "valu" | "value" | "values" => {
                if rest.is_empty() {
                    return Err(rc_err(100, "varlist required"));
                }
                let (vars_text, lbl) = if rest.len() == 1 {
                    (rest[0].clone(), String::new())
                } else {
                    let last = rest.last().unwrap().clone();
                    let vars = rest[..rest.len() - 1].join(" ");
                    if last == "." { (vars, String::new()) } else { (vars, last) }
                };
                let vars = varlist::expand(ds, &vars_text)?;
                if !lbl.is_empty() && opts.contains("nofix") {
                    // nothing extra
                }
                for v in vars {
                    let var = ds.get_var_mut(&v).unwrap();
                    if var.var_type.is_string() {
                        return Err(rc_err(181, "may not label strings"));
                    }
                    var.value_label_name = lbl.clone();
                }
                Ok(String::new())
            }
            "l" | "li" | "lis" | "list" => {
                let mut names: Vec<String> = if rest.is_empty() { ds.value_labels.keys().cloned().collect() } else { rest.to_vec() };
                names.sort();
                let mut out = Vec::new();
                for n in names {
                    let vl = ds.value_labels.get(&n).ok_or_else(|| rc_err(111, format!("value label {n} not found")))?;
                    out.push(format!("{n}:"));
                    for (v, l) in vl.iter() {
                        let vs = match missing_code(v) {
                            Some(c) => missing_name(c),
                            None => crate::format::fmt_g(v, 12),
                        };
                        out.push(format!("{:>12} {}", vs, l));
                    }
                    st.r = crate::store::Results::new();
                    st.r.set("k", vl.len() as f64);
                    if let Some((lo, _)) = vl.iter().next() {
                        st.r.set("min", lo);
                    }
                    if let Some((hi, _)) = vl.iter().last() {
                        st.r.set("max", hi);
                    }
                }
                Ok(out.join("\n"))
            }
            "dir" => {
                let mut names: Vec<String> = ds.value_labels.keys().cloned().collect();
                names.sort();
                st.r = crate::store::Results::new();
                st.r.set_str("names", names.join(" "));
                Ok(names.join("\n"))
            }
            "drop" => {
                if rest.iter().any(|r| r == "_all") {
                    ds.value_labels.clear();
                } else {
                    for n in rest {
                        if ds.value_labels.remove(n).is_none() {
                            return Err(rc_err(111, format!("value label {n} not found")));
                        }
                    }
                }
                Ok(String::new())
            }
            "copy" => {
                let (a, b) = (rest.first().cloned().unwrap_or_default(), rest.get(1).cloned().unwrap_or_default());
                let mut vl = ds.value_labels.get(&a).cloned().ok_or_else(|| rc_err(111, format!("value label {a} not found")))?;
                vl.name = b.clone();
                ds.value_labels.insert(b, vl);
                Ok(String::new())
            }
            "save" => Ok(String::new()),
            _ => Err(rc_err(198, format!("label {sub}: invalid subcommand"))),
        }
    }
}

pub struct Encode;

impl CommandHandler for Encode {
    fn name(&self) -> &'static str {
        "encode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let src = varlist::resolve_name(ds, cmd.main.trim())?;
        let newname = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        varlist::check_name(&newname)?;
        if ds.var_index.contains_key(&newname) {
            return Err(rc_err(110, format!("variable {newname} already defined")));
        }
        let lname = cmd.opt("label", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| newname.clone());
        let mask = util::touse(ds, st, cmd)?;
        let var = ds.get_var(&src).unwrap();
        if !var.var_type.is_string() {
            return Err(rc_err(108, "not a string variable"));
        }
        let mut vl = ds.value_labels.get(&lname).cloned().unwrap_or_else(|| ValueLabel::new(&lname));
        let mut distinct: Vec<String> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| var.str_at(i).to_string()).filter(|s| !s.is_empty()).collect();
        distinct.sort();
        distinct.dedup();
        let mut next = vl.iter().map(|(v, _)| v).filter(|v| !is_missing(*v)).fold(0.0, f64::max) + 1.0;
        for s in &distinct {
            if vl.value_of(s).is_none() {
                vl.add(next, s.clone());
                next += 1.0;
            }
        }
        let vals: Vec<f64> = (0..ds.n_obs)
            .map(|i| {
                let s = var.str_at(i);
                if !mask[i] || s.is_empty() { SYSMISS } else { vl.value_of(s).unwrap() }
            })
            .collect();
        let ty = best_numeric_type(&vals);
        let ty = if ty == VarType::Byte { VarType::Long } else { ty };
        let mut nv = Variable::new(&newname, ty);
        nv.data = vals.iter().map(|&v| util::typed_value(ty, v)).collect();
        nv.value_label_name = lname.clone();
        nv.label = var.label.clone();
        nv.format = "%8.0g".into();
        ds.value_labels.insert(lname, vl);
        ds.add_variable(nv)?;
        Ok(String::new())
    }
}

pub struct Decode;

impl CommandHandler for Decode {
    fn name(&self) -> &'static str {
        "decode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let src = varlist::resolve_name(ds, cmd.main.trim())?;
        let newname = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        varlist::check_name(&newname)?;
        let mask = util::touse(ds, st, cmd)?;
        let var = ds.get_var(&src).unwrap();
        if var.value_label_name.is_empty() {
            return Err(rc_err(182, format!("{src} not labeled")));
        }
        let vl = ds.value_labels.get(&var.value_label_name).cloned().unwrap_or_default();
        let strs: Vec<String> = (0..ds.n_obs)
            .map(|i| {
                if !mask[i] {
                    return String::new();
                }
                let v = var.num(i);
                vl.lookup(v).map(String::from).unwrap_or_else(|| if is_missing(v) { String::new() } else { crate::format::fmt_g(v, 9) })
            })
            .collect();
        let w = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let mut nv = Variable::new(&newname, util::str_type_for(w));
        nv.format = format!("%{w}s");
        nv.label = var.label.clone();
        nv.data = strs.into_iter().map(Value::Str).collect();
        ds.add_variable(nv)?;
        Ok(String::new())
    }
}

fn opt_newnames(ds: &DataSet, cmd: &Command, vars: &[String]) -> Result<Option<Vec<String>>, String> {
    if let Some(g) = cmd.opt("generate", 3) {
        let names: Vec<String> = g.split_whitespace().map(String::from).collect();
        if names.len() != vars.len() {
            return Err(rc_err(198, "number of variables in generate() must match varlist"));
        }
        for n in &names {
            varlist::check_name(n)?;
            if ds.var_index.contains_key(n) {
                return Err(rc_err(110, format!("variable {n} already defined")));
            }
        }
        return Ok(Some(names));
    }
    Ok(None)
}

fn replace_or_add(ds: &mut DataSet, old: &str, new: Option<&str>, mut v: Variable) -> Result<(), String> {
    match new {
        Some(n) => {
            v.name = n.to_string();
            ds.add_variable(v)
        }
        None => {
            let i = ds.var_index[old];
            v.name = old.to_string();
            let o = &ds.vars[i];
            v.label = o.label.clone();
            ds.vars[i] = v;
            Ok(())
        }
    }
}

pub struct Destring;

impl CommandHandler for Destring {
    fn name(&self) -> &'static str {
        "destring"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let newnames = opt_newnames(ds, cmd, &vars)?;
        if newnames.is_none() && !cmd.has_opt("replace", 7) {
            return Err(rc_err(198, "must specify either generate or replace option"));
        }
        let ignore: String = cmd.opt("ignore", 1).map(unquote).unwrap_or_default();
        let force = cmd.has_opt("force", 5);
        let pct = cmd.has_opt("percent", 3);
        let dpcomma = cmd.has_opt("dpcomma", 3);
        let mut out = Vec::new();
        for (k, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            if !var.var_type.is_string() {
                out.push(format!("{v} already numeric; no replace"));
                continue;
            }
            let mut vals = Vec::with_capacity(ds.n_obs);
            let mut bad = false;
            for i in 0..ds.n_obs {
                let mut s: String = var.str_at(i).chars().filter(|c| !ignore.contains(*c)).collect();
                if dpcomma {
                    s = s.replace('.', "").replace(',', ".");
                }
                let mut s = s.trim().to_string();
                let mut scale = 1.0;
                if pct && s.ends_with('%') {
                    s.pop();
                    scale = 0.01;
                }
                if s.is_empty() {
                    vals.push(SYSMISS);
                    continue;
                }
                match parse_num(&s) {
                    Some(x) => vals.push(if is_missing(x) { x } else { x * scale }),
                    None => {
                        bad = true;
                        vals.push(SYSMISS);
                    }
                }
            }
            if bad && !force {
                out.push(format!("{v}: contains nonnumeric characters; no {}", if newnames.is_some() { "generate" } else { "replace" }));
                continue;
            }
            let ty = if cmd.has_opt("float", 5) { VarType::Float } else { best_numeric_type(&vals) };
            let ty = if ty == VarType::Float && !cmd.has_opt("float", 5) {
                if vals.iter().all(|&x| util::fits(VarType::Float, x)) { VarType::Float } else { VarType::Double }
            } else {
                ty
            };
            let mut nv = Variable::new(v, ty);
            nv.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
            let nn = newnames.as_ref().map(|n| n[k].clone());
            out.push(format!("{v}: all characters numeric; {} as {}", if nn.is_some() { format!("{} generated", nn.clone().unwrap()) } else { "replaced".into() }, ty.type_str()));
            replace_or_add(ds, v, nn.as_deref(), nv)?;
        }
        Ok(out.join("\n"))
    }
}

pub struct Tostring;

impl CommandHandler for Tostring {
    fn name(&self) -> &'static str {
        "tostring"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let newnames = opt_newnames(ds, cmd, &vars)?;
        if newnames.is_none() && !cmd.has_opt("replace", 7) {
            return Err(rc_err(198, "must specify either generate or replace option"));
        }
        let fmt = cmd.opt("format", 3).map(|s| s.trim().to_string());
        let force = cmd.has_opt("force", 5);
        let mut out = Vec::new();
        for (k, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            if var.var_type.is_string() {
                out.push(format!("{v} already string; no replace"));
                continue;
            }
            let strs: Vec<String> = var
                .data
                .iter()
                .map(|x| {
                    let n = x.num();
                    if is_missing(n) {
                        if missing_code(n) == Some(0) { String::new() } else { missing_name(missing_code(n).unwrap()) }
                    } else {
                        match &fmt {
                            Some(f) => crate::format::fmt_trim(n, f),
                            None => format_full(n),
                        }
                    }
                })
                .collect();
            if !force && fmt.is_none() {
                // check round trip
                let lossy = var.data.iter().zip(&strs).any(|(x, s)| !s.is_empty() && s.parse::<f64>().map_or(false, |p| p != x.num()));
                if lossy {
                    out.push(format!("{v} cannot be converted reversibly; no replace"));
                    continue;
                }
            }
            let w = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let mut nv = Variable::new(v, util::str_type_for(w));
            nv.format = format!("%{w}s");
            nv.data = strs.into_iter().map(Value::Str).collect();
            let nn = newnames.as_ref().map(|n| n[k].clone());
            let old_type = var.var_type.type_str();
            replace_or_add(ds, v, nn.as_deref(), nv)?;
            out.push(match &nn {
                Some(n) => format!("{n} generated as str{w}"),
                None => format!("{v} was {old_type} now str{w}"),
            });
        }
        Ok(out.join("\n"))
    }
}

/// Shortest decimal representation that round-trips (like `%17.0g`).
fn format_full(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    for p in 1..=17 {
        let s = format!("{:.*e}", p - 1, x);
        if s.parse::<f64>().ok() == Some(x) {
            let v: f64 = s.parse().unwrap();
            let t = format!("{}", v);
            return t;
        }
    }
    format!("{x}")
}

// ---- recode -----------------------------------------------------------------

enum Lhs {
    Value(f64),
    Range(f64, f64),
    Missing,
    NonMissing,
    Else,
}

struct Rule {
    lhs: Vec<Lhs>,
    /// target value; `None` for `copy` (keep the original value)
    to: Option<f64>,
    label: Option<String>,
}

fn parse_rule(text: &str) -> Result<Rule, String> {
    let bad = || rc_err(198, format!("({text}): invalid rule"));
    let (l, r) = text.split_once('=').ok_or_else(bad)?;
    // right side: `#` or `copy`, optionally followed by a (quoted or bare) label
    let r = r.trim();
    let (head, rest) = match r.find(|c: char| c.is_whitespace() || c == '"' || c == '`') {
        Some(i) => (&r[..i], r[i..].trim()),
        None => (r, ""),
    };
    let to = if head.eq_ignore_ascii_case("copy") { None } else { Some(parse_num(head).ok_or_else(bad)?) };
    let label = if rest.is_empty() { None } else { Some(unquote(rest)) };
    let mut lhs = Vec::new();
    for tok in l.split_whitespace() {
        let t = tok.to_lowercase();
        lhs.push(match t.as_str() {
            "missing" => Lhs::Missing,
            "nonmissing" => Lhs::NonMissing,
            "else" | "*" => Lhs::Else,
            "min" => Lhs::Range(f64::NEG_INFINITY, f64::NEG_INFINITY),
            "max" => Lhs::Range(f64::INFINITY, f64::INFINITY),
            _ => {
                if let Some((a, b)) = t.split_once('/') {
                    let bound = |s: &str| match s {
                        "min" => Some(f64::NEG_INFINITY),
                        "max" => Some(f64::INFINITY),
                        _ => parse_num(s),
                    };
                    Lhs::Range(bound(a).ok_or_else(bad)?, bound(b).ok_or_else(bad)?)
                } else {
                    Lhs::Value(parse_num(&t).ok_or_else(|| rc_err(198, format!("{tok} invalid")))?)
                }
            }
        });
    }
    Ok(Rule { lhs, to, label })
}

pub struct Recode;

impl CommandHandler for Recode {
    fn name(&self) -> &'static str {
        "recode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let mut rules = Vec::new();
        let vars = match main.find('(') {
            Some(first_paren) => {
                let vars = varlist::expand(ds, &main[..first_paren])?;
                for w in split_top_level_words(&main[first_paren..]) {
                    let w = w.trim();
                    let inner = w.strip_prefix('(').and_then(|x| x.strip_suffix(')')).ok_or_else(|| rc_err(198, format!("{w}: rules must be in parentheses")))?;
                    rules.push(parse_rule(inner)?);
                }
                vars
            }
            None => {
                // the old syntax without parentheses: `recode x 0/4=1 5/14=2`
                let words: Vec<&str> = main.split_whitespace().collect();
                let first_rule = words.iter().position(|w| w.contains('=')).ok_or_else(|| rc_err(198, "rules required"))?;
                for w in &words[first_rule..] {
                    rules.push(parse_rule(w)?);
                }
                varlist::expand(ds, &words[..first_rule].join(" "))?
            }
        };
        let mask = util::touse(ds, st, cmd)?;
        let newnames: Option<Vec<String>> = if let Some(p) = cmd.opt("prefix", 3) {
            let names: Vec<String> = vars.iter().map(|v| format!("{}{v}", p.trim())).collect();
            for n in &names {
                varlist::check_name(n)?;
                if ds.var_index.contains_key(n) {
                    return Err(rc_err(110, format!("variable {n} already defined")));
                }
            }
            Some(names)
        } else {
            opt_newnames(ds, cmd, &vars)?
        };
        let copyrest = cmd.has_opt("copyrest", 4);
        let lname = cmd.opt("label", 1).map(|s| s.trim().to_string());
        let mut out = Vec::new();
        for (k, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            if var.var_type.is_string() {
                return Err(rc_err(108, format!("{v} is a string variable")));
            }
            let mut changes = 0;
            let vals: Vec<f64> = (0..ds.n_obs)
                .map(|i| {
                    let x = var.num(i);
                    if !mask[i] {
                        return if newnames.is_some() && !copyrest { SYSMISS } else { x };
                    }
                    for r in &rules {
                        let hit = r.lhs.iter().any(|l| match l {
                            Lhs::Value(a) => x == *a,
                            Lhs::Range(a, b) => {
                                let (lo, hi) = if a <= b { (*a, *b) } else { (*b, *a) };
                                // a bare `min`/`max` matches the observed extreme
                                let (lo, hi) = match (lo, hi) {
                                    (f64::NEG_INFINITY, f64::NEG_INFINITY) => (var_min(var), var_min(var)),
                                    (f64::INFINITY, f64::INFINITY) => (var_max(var), var_max(var)),
                                    p => p,
                                };
                                !is_missing(x) && x >= lo && x <= hi
                            }
                            Lhs::Missing => is_missing(x),
                            Lhs::NonMissing => !is_missing(x),
                            Lhs::Else => true,
                        });
                        if hit {
                            let to = r.to.unwrap_or(x);
                            if to != x {
                                changes += 1;
                            }
                            return to;
                        }
                    }
                    x
                })
                .collect();
            let ty = util::promote(if newnames.is_some() { VarType::Byte } else { var.var_type }, best_numeric_type(&vals));
            let mut nv = Variable::new(v, ty);
            nv.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
            if newnames.is_some() {
                nv.label = if var.label.is_empty() { format!("RECODE of {v}") } else { format!("RECODE of {v} ({})", var.label) };
            } else {
                nv.value_label_name = var.value_label_name.clone();
                nv.format = var.format.clone();
            }
            if rules.iter().any(|r| r.label.is_some()) {
                let name = lname.clone().unwrap_or_else(|| newnames.as_ref().map(|n| n[k].clone()).unwrap_or_else(|| v.clone()));
                let mut vl = ValueLabel::new(&name);
                for r in &rules {
                    if let (Some(l), Some(to)) = (&r.label, r.to) {
                        vl.add(to, l.clone());
                    }
                }
                ds.value_labels.insert(name.clone(), vl);
                nv.value_label_name = name;
            }
            let nn = newnames.as_ref().map(|n| n[k].clone());
            replace_or_add(ds, v, nn.as_deref(), nv)?;
            match &nn {
                Some(n) => {
                    let d = (0..ds.n_obs).filter(|&i| mask[i] && vals[i] != ds.get_var(v).unwrap().num(i)).count();
                    out.push(format!("({d} difference{} between {v} and {n})", if d == 1 { "" } else { "s" }));
                }
                None => out.push(format!("({v}: {changes} change{} made)", if changes == 1 { "" } else { "s" })),
            }
        }
        Ok(out.join("\n"))
    }
}

fn var_min(v: &Variable) -> f64 {
    (0..v.data.len()).map(|i| v.num(i)).filter(|x| !is_missing(*x)).fold(f64::INFINITY, f64::min)
}

fn var_max(v: &Variable) -> f64 {
    (0..v.data.len()).map(|i| v.num(i)).filter(|x| !is_missing(*x)).fold(f64::NEG_INFINITY, f64::max)
}

/// Split an `mv()` specification into its `\`-separated rules.
fn mv_rules(spec: &str) -> Vec<&str> {
    spec.split('\\').map(str::trim).filter(|s| !s.is_empty()).collect()
}

/// Expand a numeric list of values or `a/b` ranges (integers stepped by 1).
fn mv_values(list: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut out = Vec::new();
    for tok in list.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()) {
        if let Some((a, b)) = tok.split_once('/').filter(|(a, _)| !a.is_empty()) {
            let a = parse_num(a).ok_or_else(|| rc_err(121, format!("{tok}: invalid numlist")))?;
            let b = parse_num(b).ok_or_else(|| rc_err(121, format!("{tok}: invalid numlist")))?;
            out.push((a.min(b), a.max(b)));
        } else {
            let x = parse_num(tok).ok_or_else(|| rc_err(121, format!("{tok}: invalid numlist")))?;
            out.push((x, x));
        }
    }
    Ok(out)
}

/// Widen a numeric variable so that `x` can be stored exactly.
fn widen_for(var: &mut Variable, x: f64) {
    let ty = util::promote(var.var_type, best_numeric_type(&[x]));
    if ty != var.var_type {
        let vals: Vec<f64> = (0..var.data.len()).map(|i| var.num(i)).collect();
        var.data = vals.into_iter().map(|v| util::typed_value(ty, v)).collect();
        var.var_type = ty;
    }
}

pub struct Mvencode;

impl CommandHandler for Mvencode {
    fn name(&self) -> &'static str {
        "mvencode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let spec = cmd.opt("mv", 2).ok_or_else(|| rc_err(198, "option mv() required"))?;
        let override_ = cmd.has_opt("override", 1);
        let mask = util::touse(ds, st, cmd)?;
        // forms: `#` or `.a=1 \ .b=2 \ else=9`; `None` stands for "every other missing"
        let mut map: Vec<(Option<f64>, f64)> = Vec::new();
        for rule in mv_rules(spec) {
            let bad = || rc_err(198, format!("mv({spec}): invalid"));
            match rule.split_once('=') {
                Some((a, b)) => {
                    let to = parse_num(b).filter(|x| !is_missing(*x)).ok_or_else(bad)?;
                    let a = a.trim();
                    if a == "else" || a == "_all" {
                        map.push((None, to));
                    } else {
                        for t in a.split_whitespace() {
                            map.push((Some(parse_num(t).filter(|x| is_missing(*x)).ok_or_else(bad)?), to));
                        }
                    }
                }
                None => map.push((None, parse_num(rule).filter(|x| !is_missing(*x)).ok_or_else(bad)?)),
            }
        }
        let mut out = Vec::new();
        for v in vars {
            let i = ds.var_index[&v];
            if ds.vars[i].var_type.is_string() {
                continue;
            }
            let target = |x: f64| map.iter().find(|(k, _)| *k == Some(x)).or_else(|| map.iter().find(|(k, _)| k.is_none())).map(|x| x.1);
            if !override_ {
                let used: Vec<f64> = map.iter().map(|m| m.1).filter(|&to| (0..ds.n_obs).any(|r| mask[r] && ds.vars[i].num(r) == to)).collect();
                if let Some(to) = used.first() {
                    return Err(rc_err(9, format!("{v}: {} already used as a nonmissing value; nothing changed\n(use option override to recode anyway)", fmt_g(*to))));
                }
            }
            let mut n = 0;
            for r in 0..ds.n_obs {
                let x = ds.vars[i].num(r);
                if mask[r] && is_missing(x) {
                    if let Some(to) = target(x) {
                        widen_for(&mut ds.vars[i], to);
                        let ty = ds.vars[i].var_type;
                        ds.vars[i].data[r] = crate::data::store_num(ty, to);
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out.push(format!("{:>12}: {} missing value{} recoded", v, n, if n == 1 { "" } else { "s" }));
            }
        }
        Ok(out.join("\n"))
    }
}

fn fmt_g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 { format!("{}", x as i64) } else { format!("{x}") }
}

pub struct Mvdecode;

impl CommandHandler for Mvdecode {
    fn name(&self) -> &'static str {
        "mvdecode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let spec = cmd.opt("mv", 2).ok_or_else(|| rc_err(198, "option mv() required"))?;
        let mask = util::touse(ds, st, cmd)?;
        // forms: `numlist` (-> .) or `numlist = .a \ numlist = .b ...`
        let mut map: Vec<((f64, f64), f64)> = Vec::new();
        for rule in mv_rules(spec) {
            let (list, to) = match rule.split_once('=') {
                Some((a, b)) => (a, parse_num(b).filter(|x| is_missing(*x)).ok_or_else(|| rc_err(198, format!("mv({spec}): invalid")))?),
                None => (rule, SYSMISS),
            };
            for range in mv_values(list)? {
                map.push((range, to));
            }
        }
        let mut out = Vec::new();
        for v in vars {
            let i = ds.var_index[&v];
            if ds.vars[i].var_type.is_string() {
                continue;
            }
            let mut n = 0;
            for r in 0..ds.n_obs {
                let x = ds.vars[i].num(r);
                if mask[r] && !is_missing(x) {
                    if let Some(&(_, to)) = map.iter().find(|((a, b), _)| x >= *a && x <= *b) {
                        let ty = ds.vars[i].var_type;
                        ds.vars[i].data[r] = crate::data::store_num(ty, to);
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out.push(format!("{:>12}: {} missing value{} generated", v, n, if n == 1 { "" } else { "s" }));
            }
        }
        Ok(out.join("\n"))
    }
}

pub struct Split;

impl CommandHandler for Split {
    fn name(&self) -> &'static str {
        "split"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let stub = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| v.clone());
        let parse: Vec<String> = cmd.opt("parse", 1).map(|p| split_top_level_words(p).iter().map(|s| unquote(s)).collect()).unwrap_or_else(|| vec![" ".into()]);
        let limit = cmd.opt("limit", 1).and_then(|s| s.trim().parse::<usize>().ok());
        let mask = util::touse(ds, st, cmd)?;
        let var = ds.get_var(&v).unwrap();
        let mut pieces: Vec<Vec<String>> = Vec::new();
        for i in 0..ds.n_obs {
            let s = if mask[i] { var.str_at(i).to_string() } else { String::new() };
            let mut parts = vec![s];
            for p in &parse {
                parts = parts.iter().flat_map(|x| x.split(p.as_str()).map(String::from).collect::<Vec<_>>()).collect();
            }
            let mut parts: Vec<String> = parts.into_iter().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
            if let Some(l) = limit {
                parts.truncate(l);
            }
            pieces.push(parts);
        }
        let k = pieces.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut names = Vec::new();
        for j in 0..k {
            let name = format!("{stub}{}", j + 1);
            let strs: Vec<String> = pieces.iter().map(|p| p.get(j).cloned().unwrap_or_default()).collect();
            let w = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let mut nv = Variable::new(&name, util::str_type_for(w));
            nv.data = strs.into_iter().map(Value::Str).collect();
            ds.add_variable(nv)?;
            names.push(name);
        }
        if cmd.has_opt("destring", 3) {
            let c = ferrostat_parser::parser::parse_line(&format!("destring {}, replace", names.join(" "))).unwrap();
            Destring.execute(ds, st, &c)?;
        }
        st.r = crate::store::Results::new();
        st.r.set("nvars", k as f64);
        st.r.set_str("varlist", names.join(" "));
        Ok(format!("variables created as string:\n{}", names.join("  ")))
    }
}
