//! `ds`, `lookfor`, `vl`, `varmanage`, `labelbook` and the frame helpers
//! `frames describe`, `frames save`, `frames use`.

use std::collections::BTreeSet;

use ferrostat_core::dta::{DtaReader, DtaWriter};
use ferrostat_core::{vfs, DataSet, Variable};
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
        ("vl", Box::new(Vl)),
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

// ---------------------------------------------------------------- vl

const SYSTEM: [(&str, &str); 5] = [
    ("vlcategorical", "categorical variables"),
    ("vlcontinuous", "continuous variables"),
    ("vluncertain", "perhaps continuous, perhaps categorical variables"),
    ("vlother", "all missing or constant variables"),
    ("vldummy", "0/1 variables"),
];

struct Vl;

fn glob_list(st: &Store, name: &str) -> Vec<String> {
    st.globals.get(name).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

impl CommandHandler for Vl {
    fn name(&self) -> &'static str {
        "vl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "set" => {
                let cat = util::opt_num(cmd, "categorical", 3)?.unwrap_or(10.0) as usize;
                let unc = util::opt_num(cmd, "uncertain", 3)?.unwrap_or(100.0) as usize;
                let mut lists: [Vec<String>; 5] = Default::default();
                let skip: BTreeSet<String> = if rest.is_empty() { BTreeSet::new() } else { varlist::expand(ds, rest)?.into_iter().collect() };
                let _ = skip;
                for v in &ds.vars {
                    if v.var_type.is_string() {
                        continue;
                    }
                    let vals: Vec<f64> = (0..v.len()).map(|i| v.num(i)).filter(|x| !ferrostat_core::missing::is_missing(*x)).collect();
                    let mut distinct = vals.clone();
                    distinct.sort_by(|a, b| a.total_cmp(b));
                    distinct.dedup();
                    if distinct.len() <= 1 {
                        lists[3].push(v.name.clone());
                        continue;
                    }
                    let integer = distinct.iter().all(|x| x.fract() == 0.0);
                    let nonneg = distinct.iter().all(|x| *x >= 0.0);
                    if distinct.len() == 2 && distinct[0] == 0.0 && distinct[1] == 1.0 {
                        lists[4].push(v.name.clone());
                        lists[0].push(v.name.clone());
                    } else if integer && nonneg && distinct.len() < cat {
                        lists[0].push(v.name.clone());
                    } else if integer && nonneg && distinct.len() < unc {
                        lists[2].push(v.name.clone());
                    } else {
                        lists[1].push(v.name.clone());
                    }
                }
                for (i, (n, _)) in SYSTEM.iter().enumerate() {
                    st.globals.insert(n.to_string(), lists[i].join(" "));
                }
                Ok(vl_table(st))
            }
            "list" => {
                if rest.is_empty() || rest == "system" {
                    return Ok(vl_table(st));
                }
                let mut out = Vec::new();
                for name in rest.split_whitespace() {
                    let l = glob_list(st, name);
                    if !st.globals.contains_key(name) {
                        return Err(rc_err(111, format!("variable list {name} not found")));
                    }
                    out.push(format!("-------------------------------------------------------------------------------\nMacro: ${name}  ({} variables)\n-------------------------------------------------------------------------------", l.len()));
                    out.push(columns(&l));
                }
                Ok(out.join("\n"))
            }
            "create" => {
                // vl create name = (varlist)  /  name = varlist
                let (name, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "vl create name = (varlist)"))?;
                let name = name.trim();
                varlist::check_name(name)?;
                if SYSTEM.iter().any(|(n, _)| *n == name) {
                    return Err(rc_err(198, format!("{name} is a system variable list")));
                }
                let rhs = rhs.trim().trim_start_matches('(').trim_end_matches(')');
                let mut vars = Vec::new();
                for tok in rhs.split_whitespace() {
                    if let Some(g) = tok.strip_prefix('$') {
                        vars.extend(glob_list(st, g));
                    } else {
                        vars.extend(varlist::expand(ds, tok)?);
                    }
                }
                st.globals.insert(name.to_string(), vars.join(" "));
                Ok(format!("{} variables in ${name}", vars.len()))
            }
            "substitute" => {
                let (name, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "vl substitute name = varlist"))?;
                let mut vars = Vec::new();
                for tok in rhs.split_whitespace() {
                    if let Some(g) = tok.strip_prefix('$') {
                        vars.extend(glob_list(st, g));
                    } else {
                        vars.push(tok.to_string());
                    }
                }
                st.globals.insert(name.trim().to_string(), vars.join(" "));
                Ok(String::new())
            }
            "drop" => {
                // vl drop name | vl drop (varlist)
                if let Some(v) = rest.strip_prefix('(') {
                    let vars = varlist::expand(ds, v.trim_end_matches(')'))?;
                    for (n, _) in SYSTEM.iter() {
                        let kept: Vec<String> = glob_list(st, n).into_iter().filter(|x| !vars.contains(x)).collect();
                        if st.globals.contains_key(*n) {
                            st.globals.insert(n.to_string(), kept.join(" "));
                        }
                    }
                } else {
                    for name in rest.split_whitespace() {
                        if SYSTEM.iter().any(|(n, _)| *n == name) {
                            return Err(rc_err(198, format!("cannot drop system variable list {name}")));
                        }
                        st.globals.remove(name);
                    }
                }
                Ok(String::new())
            }
            "rebuild" => {
                let existing: BTreeSet<&str> = ds.vars.iter().map(|v| v.name.as_str()).collect();
                let mut keys: Vec<String> = SYSTEM.iter().map(|(n, _)| n.to_string()).collect();
                keys.extend(st.globals.keys().filter(|k| k.starts_with("vl") && !SYSTEM.iter().any(|(n, _)| n == k)).cloned());
                for k in keys {
                    if let Some(v) = st.globals.get(&k).cloned() {
                        let kept: Vec<&str> = v.split_whitespace().filter(|x| existing.contains(x)).collect();
                        st.globals.insert(k, kept.join(" "));
                    }
                }
                Ok(String::new())
            }
            "" => Err(rc_err(198, "vl: subcommand required")),
            other => Err(rc_err(198, format!("vl {other}: invalid subcommand"))),
        }
    }
}

fn vl_table(st: &Store) -> String {
    let rule = format!("  {}", "-".repeat(66));
    let mut out = vec![rule.clone(), "  Macro's contents".to_string(), rule.clone(), "  Macro            # Vars   Description".to_string(), rule.clone(), "  System".to_string()];
    for (n, d) in SYSTEM.iter() {
        out.push(format!("    {:<16}{:>5}   {}", format!("${n}"), glob_list(st, n).len(), d));
    }
    out.push(rule);
    out.join("\n")
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

// ---------------------------------------------------------------- frame files

const FRAME_MAGIC: &[u8] = b"FERROSTAT-FRAMES\x01\n";

/// `frames describe`, `frames save`, `frames use`; called by the frame command.
pub fn frames_extra(sub: &str, rest: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "describe" | "des" | "desc" => Some(frames_describe(ds, st, rest)),
        "save" => Some(frames_save(ds, st, cmd, rest)),
        "use" => Some(frames_use(ds, st, cmd, rest)),
        _ => None,
    }
}

fn frames_describe(ds: &DataSet, st: &Store, rest: &str) -> Result<String, String> {
    let mut all: Vec<(String, &DataSet, bool)> = vec![(st.cur_frame.clone(), ds, true)];
    all.extend(st.data_frames.iter().map(|(n, f)| (n.clone(), &f.data, false)));
    all.sort_by(|a, b| a.0.cmp(&b.0));
    let want: Vec<&str> = rest.split_whitespace().collect();
    let mut out = Vec::new();
    for w in &want {
        if !all.iter().any(|a| a.0 == *w) {
            return Err(rc_err(111, format!("frame {w} not found")));
        }
    }
    for (n, d, cur) in all.iter().filter(|a| want.is_empty() || want.contains(&a.0.as_str())) {
        out.push(format!("  Frame {n}{}", if *cur { " (current frame)" } else { "" }));
        out.push(format!("    Observations: {:>10}", d.n_obs));
        out.push(format!("       Variables: {:>10}", d.n_vars()));
        if !d.datalabel.is_empty() {
            out.push(format!("      Data label: {}", d.datalabel));
        }
        out.push(String::new());
    }
    Ok(out.join("\n").trim_end().to_string())
}

fn dta_bytes(d: &DataSet) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    DtaWriter::write_release(&mut buf, d, 118).map_err(|e| rc_err(610, format!("{e}")))?;
    Ok(buf)
}

/// Container: magic, u32 frame count, then per frame: u32 name length, name,
/// u64 data length, a complete dataset file.
fn frames_save(ds: &DataSet, st: &Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let path = cmd.using.clone().unwrap_or_else(|| util::unquote(rest.split(',').next().unwrap_or("").trim()));
    if path.is_empty() {
        return Err(rc_err(100, "filename required"));
    }
    let path = if path.contains('.') { path } else { format!("{path}.dtas") };
    if vfs::exists(&path) && !cmd.has_opt("replace", 3) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut frames: Vec<(String, &DataSet)> = vec![(st.cur_frame.clone(), ds)];
    frames.extend(st.data_frames.iter().map(|(n, f)| (n.clone(), &f.data)));
    if let Some(list) = cmd.opt("frames", 3) {
        let want: Vec<&str> = list.split_whitespace().collect();
        for w in &want {
            if !frames.iter().any(|f| f.0 == *w) {
                return Err(rc_err(111, format!("frame {w} not found")));
            }
        }
        frames.retain(|f| want.contains(&f.0.as_str()));
    }
    frames.sort_by(|a, b| a.0.cmp(&b.0));
    let mut buf = FRAME_MAGIC.to_vec();
    buf.extend((frames.len() as u32).to_le_bytes());
    for (n, d) in &frames {
        buf.extend((n.len() as u32).to_le_bytes());
        buf.extend(n.as_bytes());
        let b = dta_bytes(d)?;
        buf.extend((b.len() as u64).to_le_bytes());
        buf.extend(b);
    }
    vfs::write(&path, &buf).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
    Ok(format!("file {path} saved"))
}

fn frames_use(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let path = cmd.using.clone().unwrap_or_else(|| util::unquote(rest.split(',').next().unwrap_or("").trim()));
    let path = if !vfs::exists(&path) && vfs::exists(&format!("{path}.dtas")) { format!("{path}.dtas") } else { path };
    let data = vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    if !data.starts_with(FRAME_MAGIC) {
        return Err(rc_err(610, format!("file {path} is not a frames file")));
    }
    if !cmd.has_opt("clear", 5) && (st.data_changed && ds.n_vars() > 0 || !st.data_frames.is_empty()) {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
    }
    let bad = || rc_err(610, format!("file {path} is damaged"));
    let mut p = FRAME_MAGIC.len();
    let rd32 = |p: &mut usize| -> Result<usize, String> {
        let b = data.get(*p..*p + 4).ok_or_else(bad)?;
        *p += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let count = rd32(&mut p)?;
    let mut loaded: Vec<(String, DataSet)> = Vec::new();
    for _ in 0..count {
        let nl = rd32(&mut p)?;
        let name = String::from_utf8_lossy(data.get(p..p + nl).ok_or_else(bad)?).to_string();
        p += nl;
        let dl = u64::from_le_bytes(data.get(p..p + 8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        p += 8;
        let blob = data.get(p..p + dl).ok_or_else(bad)?.to_vec();
        p += dl;
        let d = DtaReader::from_bytes(blob).map_err(|e| rc_err(610, format!("{e}")))?;
        loaded.push((name, d));
    }
    let want: Option<Vec<String>> = cmd.opt("frames", 3).map(|l| l.split_whitespace().map(String::from).collect());
    if let Some(w) = &want {
        loaded.retain(|l| w.contains(&l.0));
    }
    crate::frames::reset(st);
    *ds = DataSet::new();
    st.tsset = None;
    st.data_changed = false;
    let mut n = 0;
    for (name, d) in loaded {
        n += 1;
        if name == st.cur_frame || (n == 1 && !st.data_frames.contains_key(&name) && name == crate::frames::DEFAULT_FRAME) {
            *ds = d;
        } else {
            st.data_frames.insert(name, crate::frames::Frame { data: d, ..Default::default() });
        }
    }
    Ok(String::new())
}
