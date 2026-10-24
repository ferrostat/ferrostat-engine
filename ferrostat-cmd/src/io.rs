//! File input/output: use, save, saveold, import/export delimited,
//! infile-free text formats (insheet/outsheet aliases).

use std::fs::File;
use std::path::Path;

use csv::{ReaderBuilder, WriterBuilder};
use ferrostat_core::dta::{DtaReader, DtaWriter};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

fn with_ext(path: &str, ext: &str) -> String {
    if Path::new(path).extension().is_some() { path.to_string() } else { format!("{path}.{ext}") }
}

/// Path given as `using` or as the command argument.
fn path_arg(cmd: &Command) -> Option<String> {
    cmd.using.clone().or_else(|| {
        let m = cmd.main.trim();
        (!m.is_empty()).then(|| util::unquote(m))
    })
}

pub struct Use;

impl CommandHandler for Use {
    fn name(&self) -> &'static str {
        "use"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (path, vars_text) = match &cmd.using {
            Some(p) => (p.clone(), cmd.main.trim().to_string()),
            None => (util::unquote(cmd.main.trim()), String::new()),
        };
        if path.is_empty() {
            return Err(rc_err(100, "filename required"));
        }
        if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
            return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
        }
        let path = with_ext(&path, "dta");
        let mut new = DtaReader::open(&path).map_err(|e| match e {
            ferrostat_core::dta::DtaError::Io(_) => rc_err(601, format!("file {path} not found")),
            other => rc_err(610, format!("file {path} not Stata format ({other})")),
        })?;
        let mut sub = cmd.clone();
        sub.main.clear();
        if cmd.if_expr.is_some() || cmd.in_range.is_some() {
            let tmp = Store::new();
            let mask = util::touse(&new, &tmp, &sub)?;
            new.keep_if(&mask)?;
        }
        if !vars_text.is_empty() {
            let keep = varlist::expand(&new, &vars_text)?;
            new.keep_variables(&keep);
        }
        *ds = new;
        st.data_changed = false;
        st.filename = path.clone();
        st.tsset = None;
        Ok(String::new())
    }
}

pub struct Save;

impl CommandHandler for Save {
    fn name(&self) -> &'static str {
        "save"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = match path_arg(cmd) {
            Some(p) => with_ext(&p, "dta"),
            None if !st.filename.is_empty() => st.filename.clone(),
            None => return Err(rc_err(100, "filename required")),
        };
        if Path::new(&path).exists() && !cmd.has_opt("replace", 7) && path_arg(cmd).is_some() {
            return Err(rc_err(602, format!("file {path} already exists")));
        }
        let release = if cmd.verb == "saveold" {
            match util::opt_num(cmd, "version", 1)?.map(|v| v as i32) {
                Some(11) | Some(12) => 115,
                Some(13) => 117,
                Some(14..=18) | None => if cmd.verb == "saveold" && util::opt_num(cmd, "version", 1)?.is_none() { 117 } else { 118 },
                Some(v) => return Err(rc_err(198, format!("version({v}) not allowed"))),
            }
        } else {
            match util::opt_num(cmd, "version", 1)?.map(|v| v as i32) {
                Some(13) => 117,
                Some(11) | Some(12) => 115,
                _ => if ds.n_vars() > 32767 { 119 } else { 118 },
            }
        };
        let mut f = File::create(&path).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
        DtaWriter::write_release(&mut f, ds, release).map_err(|e| rc_err(693, e))?;
        st.data_changed = false;
        st.filename = path.clone();
        Ok(format!("file {path} saved"))
    }
}

/// Number formatting conventions for text import.
#[derive(Clone, Copy)]
pub struct NumLocale {
    pub decimal: char,
    pub group: Option<char>,
}

impl NumLocale {
    pub fn parse(&self, s: &str) -> Option<f64> {
        let t = s.trim();
        if self.decimal == '.' && self.group.is_none() {
            return t.parse::<f64>().ok();
        }
        let mut cleaned = String::with_capacity(t.len());
        for c in t.chars() {
            if Some(c) == self.group {
                continue;
            }
            cleaned.push(if c == self.decimal { '.' } else { c });
        }
        cleaned.parse::<f64>().ok()
    }
}

/// Decimal/group separators from parselocale(), decimalseparator() and
/// groupseparator().
pub fn locale_from(cmd: &Command) -> NumLocale {
    let mut loc = NumLocale { decimal: '.', group: None };
    if let Some(l) = cmd.opt("parselocale", 6) {
        let lang = l.trim().to_lowercase();
        let lang = lang.split(['_', '-']).next().unwrap_or("");
        match lang {
            "it" | "de" | "es" | "pt" | "nl" | "da" | "id" | "tr" | "el" => loc = NumLocale { decimal: ',', group: Some('.') },
            "fr" | "ru" | "pl" | "cs" | "sv" | "fi" | "nb" | "no" | "uk" => loc = NumLocale { decimal: ',', group: Some(' ') },
            "de_ch" => loc = NumLocale { decimal: '.', group: Some('\'') },
            _ => loc = NumLocale { decimal: '.', group: Some(',') },
        }
    }
    let sep = |v: &str| -> Option<char> {
        let u = util::unquote(v);
        u.chars().next()
    };
    if let Some(d) = cmd.opt("decimalseparator", 3).and_then(sep) {
        loc.decimal = d;
        if loc.group == Some(d) {
            loc.group = Some(if d == ',' { '.' } else { ',' });
        }
    }
    if let Some(g) = cmd.opt("groupseparator", 3).and_then(sep) {
        loc.group = Some(g);
    }
    loc
}

/// Infer the narrowest type for a column of text cells.
fn infer_column_locale(cells: &[String], stringcols: bool, loc: NumLocale) -> (VarType, Vec<Value>) {
    let is_missing_text = |s: &str| s.is_empty() || s == "." || s.eq_ignore_ascii_case("na");
    let numeric = !stringcols
        && cells.iter().all(|s| {
            let t = s.trim();
            is_missing_text(t) || loc.parse(t).is_some() || (t.len() == 2 && t.starts_with('.') && t.as_bytes()[1].is_ascii_lowercase())
        })
        && cells.iter().any(|s| !is_missing_text(s.trim()));
    if numeric {
        let vals: Vec<f64> = cells
            .iter()
            .map(|s| {
                let t = s.trim();
                if t.len() == 2 && t.starts_with('.') && t.as_bytes()[1].is_ascii_lowercase() {
                    ferrostat_core::missing::missing_letter(t.chars().nth(1).unwrap()).unwrap()
                } else {
                    loc.parse(t).unwrap_or(SYSMISS)
                }
            })
            .collect();
        let ty = util::best_numeric_type(&vals);
        let ty = if ty == VarType::Float { if vals.iter().all(|&v| util::fits(VarType::Float, v)) { VarType::Float } else { VarType::Double } } else { ty };
        let data = vals.iter().map(|&v| util::typed_value(ty, v)).collect();
        (ty, data)
    } else {
        let w = cells.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        (util::str_type_for(w), cells.iter().map(|s| Value::Str(s.clone())).collect())
    }
}

/// Make a valid variable name from a header cell.
fn clean_name(raw: &str, k: usize, case: &str) -> String {
    let mut s: String = raw.trim().chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    s = s.trim_matches('_').to_string();
    if s.is_empty() || s.chars().next().unwrap().is_ascii_digit() {
        s = format!("v{}", k + 1);
    }
    let s: String = s.chars().take(32).collect();
    match case {
        "preserve" => s,
        "upper" => s.to_uppercase(),
        _ => s.to_lowercase(),
    }
}

pub fn import_delimited(path: &str, cmd: &Command) -> Result<DataSet, String> {
    let delim = match cmd.opt("delimiters", 3) {
        Some(d) => {
            let d = util::unquote(d);
            if d == "\\t" || d == "tab" { b'\t' } else { *d.as_bytes().first().unwrap_or(&b',') }
        }
        None => {
            if path.ends_with(".tsv") || path.ends_with(".tab") { b'\t' } else { b',' }
        }
    };
    let has_header = !cmd.has_opt("varnames", 4) || cmd.opt("varnames", 4).map_or(true, |v| v.trim() != "nonames");
    let file = File::open(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let mut rdr = ReaderBuilder::new().has_headers(false).delimiter(delim).flexible(true).from_reader(file);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for r in rdr.records() {
        let r = r.map_err(|e| rc_err(198, e))?;
        rows.push(r.iter().map(|s| s.to_string()).collect());
    }
    let (header, body) = if has_header && !rows.is_empty() { (Some(rows[0].clone()), &rows[1..]) } else { (None, &rows[..]) };
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let case = cmd.opt("case", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "lower".into());
    let locale = locale_from(cmd);
    let stringcols_all = cmd.has_opt("stringcols", 4) && cmd.opt("stringcols", 4).map_or(false, |s| s.trim() == "_all");
    let mut ds = DataSet::new();
    let mut used = std::collections::HashSet::new();
    for c in 0..ncols {
        let cells: Vec<String> = body.iter().map(|r| r.get(c).cloned().unwrap_or_default()).collect();
        let mut name = clean_name(header.as_ref().and_then(|h| h.get(c)).map(String::as_str).unwrap_or(""), c, &case);
        if header.is_none() {
            name = format!("v{}", c + 1);
        }
        while used.contains(&name) {
            name = format!("{name}_{c}");
        }
        used.insert(name.clone());
        let (ty, data) = infer_column_locale(&cells, stringcols_all, locale);
        let mut v = Variable::new(&name, ty);
        if let Some(h) = header.as_ref().and_then(|h| h.get(c)) {
            if h.trim() != name {
                v.label = h.trim().to_string();
            }
        }
        if ty.is_string() {
            if let VarType::Str(w) = ty {
                v.format = format!("%{w}s");
            }
        }
        v.data = data;
        ds.add_variable(v)?;
    }
    ds.n_obs = body.len();
    Ok(ds)
}

pub struct Import;

impl CommandHandler for Import {
    fn name(&self) -> &'static str {
        "import"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (kind, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let path = cmd.using.clone().unwrap_or_else(|| util::unquote(rest.trim()));
        match kind {
            "delimited" | "delim" | "csv" => {
                if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
                    return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
                }
                let new = import_delimited(&path, cmd)?;
                let msg = format!("({} vars, {} obs)", new.n_vars(), new.n_obs);
                *ds = new;
                st.data_changed = false;
                Ok(msg)
            }
            _ => Err(rc_err(198, format!("import {kind} not supported"))),
        }
    }
}

pub struct Insheet;

impl CommandHandler for Insheet {
    fn name(&self) -> &'static str {
        "insheet"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = path_arg(cmd).ok_or_else(|| rc_err(100, "using required"))?;
        let new = import_delimited(&path, cmd)?;
        let msg = format!("({} vars, {} obs)", new.n_vars(), new.n_obs);
        *ds = new;
        st.data_changed = false;
        Ok(msg)
    }
}

fn cell_out(v: &Value, var: &Variable, ds: &DataSet, labels: bool) -> String {
    match v {
        Value::Str(s) => s.clone(),
        x => {
            let n = x.num();
            if is_missing(n) {
                return String::new();
            }
            if labels && !var.value_label_name.is_empty() {
                if let Some(l) = ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(n)) {
                    return l.to_string();
                }
            }
            if var.format.starts_with("%t") || var.format.starts_with("%d") {
                return crate::format::fmt(n, &var.format).trim().to_string();
            }
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", n as i64)
            } else {
                let s = format!("{}", if matches!(v, Value::Float(_)) { (n as f32) as f64 } else { n });
                if matches!(v, Value::Float(_)) { format!("{}", n as f32) } else { s }
            }
        }
    }
}

pub fn export_delimited(ds: &DataSet, path: &str, vars: &[String], mask: &[bool], cmd: &Command) -> Result<(), String> {
    let delim = match cmd.opt("delimiter", 3) {
        Some(d) => {
            let d = util::unquote(d);
            if d == "tab" || d == "\\t" { b'\t' } else { *d.as_bytes().first().unwrap_or(&b',') }
        }
        None => b',',
    };
    if Path::new(path).exists() && !cmd.has_opt("replace", 7) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let file = File::create(path).map_err(|e| rc_err(603, e))?;
    let mut w = WriterBuilder::new().delimiter(delim).from_writer(file);
    if !cmd.has_opt("novarnames", 3) {
        w.write_record(vars).map_err(|e| rc_err(603, e))?;
    }
    let labels = !cmd.has_opt("nolabel", 3);
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let rec: Vec<String> = vars.iter().map(|v| {
            let var = ds.get_var(v).unwrap();
            cell_out(&var.data[i], var, ds, labels)
        }).collect();
        w.write_record(&rec).map_err(|e| rc_err(603, e))?;
    }
    w.flush().map_err(|e| rc_err(603, e))?;
    Ok(())
}

pub struct Export;

impl CommandHandler for Export {
    fn name(&self) -> &'static str {
        "export"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (kind, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        if !matches!(kind, "delimited" | "delim" | "csv") {
            return Err(rc_err(198, format!("export {kind} not supported")));
        }
        let (vars, path) = match &cmd.using {
            Some(p) => (if rest.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, rest)? }, p.clone()),
            None => (ds.varlist(), util::unquote(rest.trim())),
        };
        let mut sub = cmd.clone();
        sub.main.clear();
        let mask = util::touse(ds, st, &sub)?;
        export_delimited(ds, &path, &vars, &mask, cmd)?;
        Ok(format!("file {path} saved"))
    }
}

pub struct Outsheet;

impl CommandHandler for Outsheet {
    fn name(&self) -> &'static str {
        "outsheet"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mut sub = cmd.clone();
        sub.main.clear();
        let mask = util::touse(ds, st, &sub)?;
        export_delimited(ds, &path, &vars, &mask, cmd)?;
        Ok(String::new())
    }
}
