//! `putexcel`: writing cells, expressions and matrices to .xlsx workbooks.
//! Workbooks are read and written by `ferrostat_cmd::xlsx`.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matcmd::eval_matrix;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

pub use ferrostat_cmd::xlsx::{col_letters, parse_ref, Cell, Workbook};

#[derive(Debug, Clone, Default)]
pub struct PutExcel {
    pub path: Option<String>,
    pub sheet: String,
    pub books: BTreeMap<String, Workbook>,
}

fn opt_arg(opts: &[(String, Option<String>)], name: &str) -> Option<String> {
    opts.iter().find(|(n, _)| n.to_lowercase().starts_with(name)).and_then(|(_, a)| a.clone())
}

fn has_opt(opts: &[(String, Option<String>)], name: &str) -> bool {
    opts.iter().any(|(n, _)| n.to_lowercase() == name)
}

pub fn putexcel(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    };
    let first = main.split_whitespace().next().unwrap_or("").to_lowercase();
    if first == "set" {
        let path = unquote(main[3..].trim());
        let path = if path.ends_with(".xlsx") || path.ends_with(".xls") { path } else { format!("{path}.xlsx") };
        let replace = has_opt(&opts, "replace");
        let mut book = if replace {
            Workbook::default()
        } else if let Some(b) = me.putexcel.books.get(&path) {
            b.clone()
        } else if let Ok(bytes) = ferrostat_core::vfs::read(&path) {
            Workbook::read(&bytes).map_err(|_| rc_err(603, format!("{path} could not be read; use replace")))?
        } else {
            Workbook::default()
        };
        let (sheet, sheet_replace) = match opt_arg(&opts, "sheet") {
            Some(s) => {
                let parts: Vec<String> = s.split(',').map(|x| x.trim().to_string()).collect();
                (unquote(&parts[0]), parts.get(1).map_or(false, |x| x.starts_with("replace")))
            }
            None => ("Sheet1".to_string(), false),
        };
        match book.sheets.iter_mut().find(|(n, _)| *n == sheet) {
            Some((_, cells)) => {
                if sheet_replace {
                    cells.clear();
                }
            }
            None => book.sheets.push((sheet.clone(), BTreeMap::new())),
        }
        me.putexcel.path = Some(path.clone());
        me.putexcel.sheet = sheet;
        me.putexcel.books.insert(path, book);
        return Ok(String::new());
    }
    if first == "clear" {
        me.putexcel.path = None;
        return Ok(String::new());
    }
    if first == "describe" {
        return Ok(format!("{}", me.putexcel.path.clone().unwrap_or_default()));
    }
    let path = me.putexcel.path.clone().ok_or_else(|| rc_err(198, "putexcel set required first"))?;
    // assignments: CELL = item ...
    let words = split_top_level_words(&main);
    let mut i = 0;
    let mut writes: Vec<((u32, u32), Cell)> = Vec::new();
    while i < words.len() {
        let target = words[i].clone();
        let (cellref, after_eq) = if let Some((a, b)) = target.split_once('=') {
            (a.to_string(), b.to_string())
        } else {
            if words.get(i + 1).map(String::as_str) != Some("=") {
                return Err(rc_err(198, format!("{target}: = expected")));
            }
            i += 1;
            (target, String::new())
        };
        let cell = cellref.split(':').next().unwrap();
        let (row, col) = parse_ref(cell).ok_or_else(|| rc_err(198, format!("{cell} invalid cell")))?;
        // the value is the next word (or the text after '=')
        let value = if !after_eq.is_empty() {
            after_eq
        } else {
            i += 1;
            words.get(i).cloned().ok_or_else(|| rc_err(198, "value expected"))?
        };
        i += 1;
        // swallow further words that belong to an expression until the next CELL=
        let mut value = value;
        while i < words.len() && !(parse_ref(words[i].split('=').next().unwrap()).is_some() && (words[i].contains('=') || words.get(i + 1).map(String::as_str) == Some("="))) {
            value.push(' ');
            value.push_str(&words[i]);
            i += 1;
        }
        let v = value.trim();
        if let Some(inner) = v.strip_prefix("matrix(").and_then(|x| x.strip_suffix(')')) {
            let m = eval_matrix(&me.engine.dataset, &me.engine.store, inner)?;
            let names = has_opt(&opts, "names");
            let rown = names || has_opt(&opts, "rownames");
            let coln = names || has_opt(&opts, "colnames");
            let (r0, c0) = (row + coln as u32, col + rown as u32);
            if coln {
                for (j, cn) in m.colnames.iter().enumerate() {
                    writes.push(((row, c0 + j as u32), Cell::Str(cn.clone())));
                }
            }
            if rown {
                for (k, rn) in m.rownames.iter().enumerate() {
                    writes.push(((r0 + k as u32, col), Cell::Str(rn.clone())));
                }
            }
            for a in 0..m.rows {
                for b in 0..m.cols {
                    let x = m[(a, b)];
                    writes.push(((r0 + a as u32, c0 + b as u32), if ferrostat_core::is_missing(x) { Cell::Str(String::new()) } else { Cell::Num(x) }));
                }
            }
        } else if v.starts_with('"') || v.starts_with("`\"") {
            writes.push(((row, col), Cell::Str(unquote(v))));
        } else if v.starts_with("image(") || v.starts_with("picture(") {
            // images are not embedded; nothing to write
        } else {
            let e = parse_expression(v).map_err(|e| rc_err(198, e.message))?;
            match Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)? {
                Val::Num(x) => writes.push(((row, col), Cell::Num(x))),
                Val::Str(s) => writes.push(((row, col), Cell::Str(s))),
            }
        }
    }
    let sheet = me.putexcel.sheet.clone();
    let book = me.putexcel.books.entry(path.clone()).or_default();
    let cells = match book.sheets.iter_mut().find(|(n, _)| *n == sheet) {
        Some((_, c)) => c,
        None => {
            book.sheets.push((sheet.clone(), BTreeMap::new()));
            &mut book.sheets.last_mut().unwrap().1
        }
    };
    for (pos, c) in writes {
        cells.insert(pos, c);
    }
    ferrostat_core::vfs::write(&path, &book.to_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("file {path} saved"))
}

