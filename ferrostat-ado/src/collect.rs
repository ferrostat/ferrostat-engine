//! `collect` (subcommands and the `collect:` prefix), `dtable` and `etable`
//! as seen by the interpreter: the prefix runs a command and collects its
//! results; Word export uses the document writer of this crate.

use ferrostat_cmd::collect::{collect_results, parse_tags, Collection};
use ferrostat_cmd::collect_cmd::{self, export_kind, opt_arg, split_main};
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::util::unquote;
use ferrostat_core::vfs;
use ferrostat_parser::parser::{find_top_level, parse_line};

use crate::docx::{Block, Cell, Docx, Inline, Paragraph, Run, Table};
use crate::interpreter::AdoInterpreter;

/// `collect ...`; `full` is the whole statement.
pub fn collect(me: &mut AdoInterpreter, full: &str) -> Result<String, String> {
    let body = full.trim();
    let after = body.get(7..).unwrap_or("").trim_start();
    let after = if body.len() >= 7 && body[..7].eq_ignore_ascii_case("collect") { after } else { body.split_once(char::is_whitespace).map_or("", |x| x.1).trim_start() };
    if after.starts_with(':') || after.starts_with(',') {
        return prefix(me, after);
    }
    let sub = after.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").to_lowercase();
    if sub.len() >= 3 && "export".starts_with(sub.as_str()) {
        let args = after[sub.len()..].trim();
        return export(me, args, None);
    }
    collect_cmd::run(&mut me.engine.store, after)
}

/// `collect [, tags() name() ...]: command`
fn prefix(me: &mut AdoInterpreter, after: &str) -> Result<String, String> {
    let colon = find_top_level(after, ':').ok_or_else(|| rc_err(198, "collect: colon expected"))?;
    let (_, opts) = split_main(&after[..colon]);
    let command = after[colon + 1..].trim().to_string();
    let tags = match opt_arg(&opts, "tags", 3) {
        Some(t) => parse_tags(&t)?,
        None => Vec::new(),
    };
    let name = opt_arg(&opts, "name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let (r0, e0) = (me.engine.store.r.clone(), me.engine.store.e.clone());
    me.exec_text(&command, None)?;
    let st = &mut me.engine.store;
    let (r, e) = (st.r.clone(), st.e.clone());
    let e_new = e.scalars != e0.scalars || e.macros != e0.macros || e.matrices.len() != e0.matrices.len();
    let r_new = r.scalars != r0.scalars || r.macros != r0.macros || r.matrices.len() != r0.matrices.len();
    if let Some(n) = &name {
        if !st.collect.map.contains_key(n) {
            st.collect.map.insert(n.clone(), Collection::new(n));
        }
        st.collect.current = n.clone();
    }
    let c = st.collect.cur_mut();
    let mut tags = tags;
    if tags.is_empty() {
        c.cmdset += 1;
        tags.push(("cmdset".into(), c.cmdset.to_string()));
    }
    if r_new {
        collect_results(c, &r, false, &tags, None);
    }
    if e_new {
        collect_results(c, &e, true, &tags, None);
    }
    Ok(String::new())
}

/// `collect export file [, as() replace name() ...]`; Word documents here,
/// every other format in the command crate.
fn export(me: &mut AdoInterpreter, args: &str, _default: Option<&str>) -> Result<String, String> {
    let (main, o) = split_main(args);
    let path = unquote(&main);
    if export_kind(&path, &o) != "docx" {
        return collect_cmd::export(&mut me.engine.store, args);
    }
    let replace = o.iter().any(|(n, _)| n.to_lowercase().starts_with("replace") || n.to_lowercase().starts_with("modify"));
    if vfs::exists(&path) && !replace {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let name = opt_arg(&o, "name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let c = me.engine.store.collect.get(name.as_deref())?;
    let r = c.render()?;
    let mut d = Docx::new();
    for (k, t) in r.tables.iter().enumerate() {
        let (g, nh) = t.grid(&r.style);
        let cells: Vec<Vec<Cell>> = g
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, s)| Cell { text: s.clone(), bold: i < nh, halign: Some(if j == 0 { "left".into() } else if i < nh { "center".into() } else { "right".into() }), ..Default::default() })
                    .collect()
            })
            .collect();
        let title = if k == 0 { r.title.clone() } else { None }.or_else(|| t.heading.clone());
        let notes = if k + 1 == r.tables.len() { r.notes.clone() } else { Vec::new() };
        d.blocks.push(Block::Table(Table { name: format!("table{}", k + 1), cells, border: true, title, notes }));
        d.blocks.push(Block::Para(Paragraph { style: None, halign: None, items: vec![Inline::Run(Run::default())] }));
    }
    vfs::write(&path, &d.to_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("(collection {} exported to file {path})", c.name))
}

/// Pull an `export(file [, replace])` option out of a statement.
fn take_export(full: &str) -> (String, Option<String>) {
    let lower = full.to_lowercase();
    let Some(comma) = find_top_level(full, ',') else { return (full.to_string(), None) };
    let mut search = comma;
    while let Some(p) = lower[search..].find("export(") {
        let start = search + p;
        let prev = full[..start].chars().last().unwrap_or(' ');
        if prev.is_whitespace() || prev == ',' {
            // matching parenthesis
            let mut depth = 0;
            for (i, ch) in full[start..].char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            let arg = full[start + 7..start + i].to_string();
                            let rest = format!("{}{}", &full[..start], &full[start + i + 1..]);
                            return (rest, Some(arg));
                        }
                    }
                    _ => {}
                }
            }
            break;
        }
        search = start + 7;
    }
    (full.to_string(), None)
}

/// `dtable` and `etable`.
pub fn dtable_etable(me: &mut AdoInterpreter, verb: &str, full: &str) -> Result<String, String> {
    let (text, export_arg) = take_export(full);
    let cmd = parse_line(&text).map_err(|e| rc_err(198, e.message))?;
    let e = &mut me.engine;
    let out = if verb == "dtable" { ferrostat_cmd::dtable::run(&mut e.dataset, &mut e.store, &cmd)? } else { ferrostat_cmd::etable::run(&mut e.dataset, &mut e.store, &cmd)? };
    if let Some(a) = export_arg {
        let msg = export(me, &a, None)?;
        return Ok(format!("{out}\n{msg}"));
    }
    Ok(out)
}
