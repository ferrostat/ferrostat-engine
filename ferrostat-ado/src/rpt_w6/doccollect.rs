//! `putdocx collect` and `putpdf collect`: add the table of the current
//! (or a named) collection to the active document.
//!
//! Syntax: `putdocx collect [, name(cname) tablename(tname) title(text) note(text) border(nil)]`.

use ferrostat_cmd::eval::rc_err;

use super::{has, opt_arg, split_main};
use crate::docx::{Block, Cell, Table};
use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

pub fn put_collect(me: &mut AdoInterpreter, pdf: bool, r: &str) -> Result<String, String> {
    let cmd = if pdf { "putpdf" } else { "putdocx" };
    let rest = r.trim_start();
    let rest = &rest["collect".len().min(rest.len())..];
    let (main, o) = split_main(rest);
    if !main.trim().is_empty() {
        return Err(rc_err(198, format!("{cmd} collect: {} not allowed", main.trim())));
    }
    let name = opt_arg(&o, "name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let c = me.engine.store.collect.get(name.as_deref())?;
    let rendered = c.render()?;
    if rendered.tables.is_empty() {
        return Err(rc_err(198, format!("collection {} has no table; use collect layout", c.name)));
    }
    let border = !opt_arg(&o, "border", 3).is_some_and(|b| b.to_lowercase().contains("nil"));
    let title = opt_arg(&o, "title", 5).map(|t| unquote(&t)).or_else(|| rendered.title.clone());
    let mut notes: Vec<String> = rendered.notes.clone();
    if let Some(n) = opt_arg(&o, "note", 4).or_else(|| opt_arg(&o, "notes", 5)) {
        notes.push(unquote(&n));
    }
    let base = opt_arg(&o, "tablename", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let d = (if pdf { me.docs.pdf.as_mut() } else { me.docs.docx.as_mut() }).ok_or_else(|| rc_err(198, format!("document not created; use {cmd} begin")))?;
    let ntab = rendered.tables.len();
    let existing = d.blocks.iter().filter(|b| matches!(b, Block::Table(_))).count();
    for (k, t) in rendered.tables.iter().enumerate() {
        let (g, nh) = t.grid(&rendered.style);
        let cells: Vec<Vec<Cell>> = g
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, s)| Cell {
                        text: s.trim_end().to_string(),
                        bold: i < nh,
                        halign: Some(if j == 0 { "left".into() } else if i < nh { "center".into() } else { "right".into() }),
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        let tname = match (&base, ntab) {
            (Some(b), 1) => b.clone(),
            (Some(b), _) => format!("{b}_{}", k + 1),
            (None, _) => format!("collect{}", existing + k + 1),
        };
        let ttitle = if k == 0 { title.clone() } else { None }.or_else(|| t.heading.clone());
        let tnotes = if k + 1 == ntab { notes.clone() } else { Vec::new() };
        let table = Table { name: tname.clone(), cells, border, title: ttitle, notes: tnotes };
        if let Some(old) = d.table_mut(&tname) {
            *old = table;
        } else {
            d.blocks.push(Block::Table(table));
        }
    }
    if has(&o, "nomsg", 5) {
        return Ok(String::new());
    }
    Ok(String::new())
}
