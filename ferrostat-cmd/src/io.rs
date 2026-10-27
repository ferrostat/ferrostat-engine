//! File input/output: use, save, saveold, import/export delimited,
//! infile-free text formats (insheet/outsheet aliases).

use std::path::Path;

use csv::{ReaderBuilder, WriterBuilder};
use ferrostat_core::dta::{DtaReader, DtaWriter};
use ferrostat_core::vfs;
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
            other => rc_err(610, format!("file {path} not a valid .dta dataset ({other})")),
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
        if vfs::exists(&path) && !cmd.has_opt("replace", 7) && path_arg(cmd).is_some() {
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
        let mut buf = Vec::new();
        DtaWriter::write_release(&mut buf, ds, release).map_err(|e| rc_err(693, e))?;
        vfs::write(&path, &buf).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
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
    let file = std::io::Cursor::new(vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?);
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
            "excel" | "xlsx" => {
                if cmd.has_opt("describe", 3) {
                    return describe_excel(&path);
                }
                if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
                    return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
                }
                let new = import_excel(&path, cmd)?;
                let msg = format!("({} vars, {} obs)", new.n_vars(), crate::util::commas(new.n_obs));
                *ds = new;
                st.data_changed = false;
                Ok(msg)
            }
            _ => Err(rc_err(198, format!("import {kind} not supported"))),
        }
    }
}

fn read_workbook(path: &str) -> Result<crate::xlsx::Workbook, String> {
    if path.to_lowercase().ends_with(".xls") {
        return Err(rc_err(610, format!("{path}: legacy .xls (BIFF) workbooks are not supported; save as .xlsx")));
    }
    let data = vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    crate::xlsx::Workbook::read(&data)
}

fn describe_excel(path: &str) -> Result<String, String> {
    let book = read_workbook(path)?;
    let mut out = vec!["Sheet | Range".to_string(), "------+--------".to_string()];
    for (name, cells) in &book.sheets {
        let range = match (cells.keys().map(|k| k.0).min(), cells.keys().map(|k| k.0).max(), cells.keys().map(|k| k.1).min(), cells.keys().map(|k| k.1).max()) {
            (Some(r0), Some(r1), Some(c0), Some(c1)) => format!("{}{}:{}{}", crate::xlsx::col_letters(c0), r0, crate::xlsx::col_letters(c1), r1),
            _ => "(empty)".into(),
        };
        out.push(format!("{name} | {range}"));
    }
    Ok(out.join("\n"))
}

/// `import excel`: read one worksheet into a dataset.
pub fn import_excel(path: &str, cmd: &Command) -> Result<DataSet, String> {
    use crate::xlsx::Cell;
    // a name without extension means .xls, then .xlsx
    let has_ext = std::path::Path::new(path).extension().is_some();
    let path = if has_ext || ferrostat_core::vfs::exists(path) {
        path.to_string()
    } else {
        [".xls", ".xlsx"].iter().map(|e| format!("{path}{e}")).find(|p| ferrostat_core::vfs::exists(p)).unwrap_or_else(|| path.to_string())
    };
    let path = path.as_str();
    let book = read_workbook(path)?;
    let sheet_name = cmd.opt("sheet", 2).map(|s| util::unquote(s));
    let (_, cells) = match &sheet_name {
        Some(n) => book.sheets.iter().find(|(s, _)| s == n).ok_or_else(|| rc_err(601, format!("worksheet {n} not found")))?,
        None => book.sheets.first().ok_or_else(|| rc_err(610, "workbook has no worksheets"))?,
    };
    // range: cellrange(A1:D20), cellrange(B3:) or the used range
    let (r0, c0, r1, c1) = {
        let used = (
            cells.keys().map(|k| k.0).min().unwrap_or(1),
            cells.keys().map(|k| k.1).min().unwrap_or(1),
            cells.keys().map(|k| k.0).max().unwrap_or(0),
            cells.keys().map(|k| k.1).max().unwrap_or(0),
        );
        match cmd.opt("cellrange", 4) {
            Some(r) => {
                let r = util::unquote(r);
                let (a, b) = r.split_once(':').unwrap_or((&r, ""));
                let (ar, ac) = crate::xlsx::parse_ref(a).ok_or_else(|| rc_err(198, format!("invalid cellrange({r})")))?;
                match crate::xlsx::parse_ref(b) {
                    Some((br, bc)) => (ar, ac, br, bc),
                    None => (ar, ac, used.2, used.3),
                }
            }
            None => (1.max(used.0.min(1)), used.1, used.2, used.3),
        }
    };
    let firstrow = cmd.has_opt("firstrow", 3);
    let allstring = cmd.has_opt("allstring", 4);
    let case = cmd.opt("case", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "preserve".into());
    let data_start = if firstrow { r0 + 1 } else { r0 };
    let mut ds = DataSet::new();
    let mut used_names = std::collections::HashSet::new();
    let text_of = |c: &Cell| -> String {
        match c {
            Cell::Str(s) => s.clone(),
            Cell::Num(x) => format_full(*x),
            Cell::Bool(b) => if *b { "1".into() } else { "0".into() },
            Cell::Date(x) => crate::format::fmt(crate::xlsx::excel_to_td(*x).floor(), "%td").trim().to_string(),
        }
    };
    for col in c0..=c1 {
        let column: Vec<Option<&Cell>> = (data_start..=r1).map(|r| cells.get(&(r, col))).collect();
        let header = if firstrow { cells.get(&(r0, col)).map(&text_of) } else { None };
        if column.iter().all(|c| c.is_none()) && header.as_deref().map_or(true, |h| h.trim().is_empty()) {
            continue; // empty column
        }
        let letters = crate::xlsx::col_letters(col);
        let mut name = match &header {
            Some(h) if !h.trim().is_empty() => excel_name(h, &letters, &case),
            _ => letters.clone(),
        };
        if !firstrow && case == "lower" {
            name = name.to_lowercase();
        }
        while used_names.contains(&name) {
            name = format!("{name}_{letters}");
        }
        used_names.insert(name.clone());
        let numeric = !allstring && column.iter().all(|c| matches!(c, None | Some(Cell::Num(_)) | Some(Cell::Bool(_)) | Some(Cell::Date(_))));
        let dates = numeric && column.iter().any(|c| matches!(c, Some(Cell::Date(_)))) && column.iter().all(|c| matches!(c, None | Some(Cell::Date(_))));
        let mut v;
        if numeric {
            let vals: Vec<f64> = column
                .iter()
                .map(|c| match c {
                    Some(Cell::Num(x)) => *x,
                    Some(Cell::Bool(b)) => *b as i32 as f64,
                    Some(Cell::Date(x)) => {
                        let td = crate::xlsx::excel_to_td(*x);
                        if dates && x.fract() == 0.0 { td } else if dates { td * 86_400_000.0 } else { *x }
                    }
                    _ => SYSMISS,
                })
                .collect();
            let has_time = dates && column.iter().any(|c| matches!(c, Some(Cell::Date(x)) if x.fract() != 0.0));
            let vals: Vec<f64> = if has_time {
                column.iter().map(|c| match c {
                    Some(Cell::Date(x)) => crate::xlsx::excel_to_td(*x) * 86_400_000.0,
                    _ => SYSMISS,
                }).collect()
            } else {
                vals
            };
            let ty = if has_time { VarType::Double } else { util::best_numeric_type(&vals) };
            let ty = if ty == VarType::Float && !vals.iter().all(|&x| util::fits(VarType::Float, x)) { VarType::Double } else { ty };
            v = Variable::new(&name, ty);
            v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
            if dates {
                v.format = if has_time { "%tc".into() } else { "%td".into() };
            }
        } else {
            let strs: Vec<String> = column.iter().map(|c| c.map(&text_of).unwrap_or_default()).collect();
            let w = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            v = Variable::new(&name, util::str_type_for(w));
            if let VarType::Str(w) = v.var_type {
                v.format = format!("%{w}s");
            }
            v.data = strs.into_iter().map(Value::Str).collect();
        }
        if let Some(h) = header {
            if h.trim() != name {
                v.label = h.trim().chars().take(80).collect();
            }
        }
        ds.add_variable(v)?;
    }
    ds.n_obs = if r1 >= data_start { (r1 - data_start + 1) as usize } else { 0 };
    for v in ds.vars.iter_mut() {
        while v.data.len() < ds.n_obs {
            v.push_missing();
        }
    }
    Ok(ds)
}

/// Shortest text representation of a number that round-trips.
fn format_full(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
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
    if vfs::exists(path) && !cmd.has_opt("replace", 7) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut w = WriterBuilder::new().delimiter(delim).from_writer(Vec::new());
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
    let bytes = w.into_inner().map_err(|e| rc_err(603, e.to_string()))?;
    vfs::write(path, &bytes).map_err(|e| rc_err(603, e))?;
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
        if matches!(kind, "excel" | "xlsx") {
            return export_excel(ds, st, cmd, rest);
        }
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

/// `export excel [varlist] using file [if] [in] [, sheet("name"[, modify|replace]) firstrow(variables|varlabels) replace nolabel]`
fn export_excel(ds: &DataSet, st: &Store, cmd: &Command, rest: &str) -> Result<String, String> {
    use crate::xlsx::{Cell, Workbook};
    let (vars, path) = match &cmd.using {
        Some(p) => (if rest.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, rest)? }, p.clone()),
        None => (ds.varlist(), util::unquote(rest.trim())),
    };
    let path = if path.to_lowercase().ends_with(".xlsx") { path } else { format!("{path}.xlsx") };
    let (sheet, sheet_mode) = match cmd.opt("sheet", 2) {
        Some(s) => {
            let parts: Vec<String> = s.split(',').map(|x| x.trim().to_string()).collect();
            (util::unquote(&parts[0]), parts.get(1).cloned().unwrap_or_default())
        }
        None => ("Sheet1".to_string(), String::new()),
    };
    let exists = vfs::exists(&path);
    let modify = cmd.has_opt("sheetmodify", 6) || sheet_mode.starts_with("modify");
    let sheet_replace = cmd.has_opt("sheetreplace", 6) || sheet_mode.starts_with("replace");
    if exists && !cmd.has_opt("replace", 7) && !modify && !sheet_replace {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut book = if exists && (modify || sheet_replace) {
        read_workbook(&path)?
    } else {
        Workbook::default()
    };
    let mut sub = cmd.clone();
    sub.main.clear();
    let mask = util::touse(ds, st, &sub)?;
    let labels = !cmd.has_opt("nolabel", 3);
    let header = cmd.opt("firstrow", 3).map(|s| s.trim().to_lowercase());
    let cells = book.sheet_mut(&sheet);
    if sheet_replace {
        cells.clear();
    }
    let mut row = 1u32;
    if let Some(h) = &header {
        for (j, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            let text = if h.starts_with("varl") && !var.label.is_empty() { var.label.clone() } else { v.clone() };
            cells.insert((1, j as u32 + 1), Cell::Str(text));
        }
        row = 2;
    }
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        for (j, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            let cell = match &var.data[i] {
                Value::Str(s) => {
                    if s.is_empty() {
                        None
                    } else {
                        Some(Cell::Str(s.clone()))
                    }
                }
                x => {
                    let n = x.num();
                    if is_missing(n) {
                        None
                    } else if labels && !var.value_label_name.is_empty() {
                        Some(ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(n)).map(|l| Cell::Str(l.to_string())).unwrap_or(Cell::Num(n)))
                    } else if var.format.starts_with("%td") || var.format.starts_with("%d") {
                        Some(Cell::Date(n + 21916.0))
                    } else if var.format.starts_with("%tc") {
                        Some(Cell::Date(n / 86_400_000.0 + 21916.0))
                    } else {
                        Some(Cell::Num(n))
                    }
                }
            };
            if let Some(c) = cell {
                cells.insert((row, j as u32 + 1), c);
            }
        }
        row += 1;
    }
    vfs::write(&path, &book.to_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("file {path} saved"))
}

/// Variable name from a header cell: characters that are not valid in names
/// are removed; falls back to the column letters.
fn excel_name(raw: &str, letters: &str, case: &str) -> String {
    let mut s: String = raw.trim().chars().filter(|c| c.is_alphanumeric() || *c == '_').collect();
    if s.chars().next().map_or(true, |c| c.is_ascii_digit()) {
        s = if s.is_empty() { letters.to_string() } else { format!("_{s}") };
    }
    let s: String = s.chars().take(32).collect();
    match case {
        "lower" => s.to_lowercase(),
        "upper" => s.to_uppercase(),
        _ => s,
    }
}
