//! Programming commands: byable and sortpreserve programs, program
//! properties, trace output, `mark`/`markin`, `unab`/`tsunab`/`fvunab`,
//! `unabcmd`, `fvexpand`, `_rmcoll`/`_rmdcoll`, `makecns`/`matcproc`,
//! estimation-sample signatures, `tabdisp`, `sysdir`/`personal`/`adopath`,
//! `findfile`, `viewsource`, `discard`, `pause`, `window` and the
//! unsupported `plugin`/`javacall`. Files, posting and sersets live in
//! [`crate::progfile`]; the class system in [`crate::progclass`].

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::{Program, Results};
use ferrostat_cmd::{est, format, util, varlist};
use ferrostat_core::missing::is_missing;
use ferrostat_core::vfs;
use ferrostat_core::{Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_line, split_options, split_top_level_words};
use ferrostat_parser::Command;

use crate::interpreter::{AdoInterpreter, Flow};
use crate::macros::{macro_number, unquote};

/// Interpreter state used by these commands.
#[derive(Default)]
pub struct ProgState {
    pub files: crate::progfile::Files,
    pub posts: std::collections::BTreeMap<String, crate::progfile::Post>,
    pub sersets: crate::progfile::Sersets,
    pub class: crate::progclass::ClassState,
    pub pause_on: bool,
    /// Locals to define when the next program starts (byable(onecall)).
    by_locals: Option<Vec<(String, String)>>,
    /// Programs being traced (names), innermost last.
    trace_stack: Vec<String>,
    counter: usize,
}

/// Commands provided here (for command lists and "did you mean").
pub const COMMANDS: &[&str] = &[
    "file", "postfile", "post", "postclose", "postutil", "serset", "sysdir", "personal", "findfile", "tabdisp", "mark", "markin", "fvexpand", "_rmcoll", "rmcoll", "_rmdcoll", "makecns", "matcproc",
    "signestimationsample", "checkestimationsample", "viewsource", "class", "classutil", "classexit", "unabcmd", "discard", "pause", "adopath", "tsunab", "fvunab",
];

/// Split `text, options`.
pub(crate) fn split_opts(r: &str) -> (String, Vec<(String, Option<String>)>) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

pub(crate) fn has_opt(opts: &[(String, Option<String>)], name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| o.len() >= min.min(name.len()) && name.starts_with(o.as_str()))
}

pub(crate) fn opt_arg(opts: &[(String, Option<String>)], name: &str, min: usize) -> Option<String> {
    opts.iter().find(|(o, _)| o.len() >= min.min(name.len()) && name.starts_with(o.as_str())).and_then(|(_, a)| a.clone())
}

/// Entry point from the interpreter's command table.
pub fn command(me: &mut AdoInterpreter, verb: &str, rest: &str, full: &str) -> Result<Option<String>, String> {
    let t = full.trim_start();
    if t.starts_with('.') && t[1..].starts_with(|c: char| c.is_alphabetic() || c == '_') {
        return crate::progclass::statement(me, t).map(Some);
    }
    if t.starts_with(',') && t.contains("inherit(") {
        return crate::progclass::inherit_line(me, t).map(Some);
    }
    let r = rest.trim();
    Ok(Some(match verb {
        "file" => crate::progfile::file_cmd(me, r)?,
        "postfile" => crate::progfile::postfile(me, r)?,
        "post" => crate::progfile::post(me, r)?,
        "postclose" => crate::progfile::postclose(me, r)?,
        "postutil" => crate::progfile::postutil(me, r)?,
        "serset" => crate::progfile::serset(me, r)?,
        "frame" | "frames" if first_word(r) == "post" => crate::progfile::frame_post(me, r[4..].trim())?,
        "sysdir" => sysdir(me, r)?,
        "personal" => personal(me, r)?,
        "adopath" => adopath(me, r)?,
        "findfile" => findfile(me, r)?,
        "viewsource" => viewsource(me, r)?,
        "discard" => discard(me),
        "pause" => pause(me, r)?,
        "mark" | "markin" => mark(me, verb, r)?,
        "unab" | "tsunab" | "fvunab" => unab(me, verb, r)?,
        "unabcmd" => unabcmd(me, r)?,
        "fvexpand" => fvexpand(me, r)?,
        "_rmcoll" | "rmcoll" | "_rmdcoll" => rmcoll(me, verb == "_rmdcoll", r)?,
        "makecns" => makecns(me, r)?,
        "matcproc" => matcproc(me, r)?,
        "signestimationsample" => sign_sample(me, r)?,
        "checkestimationsample" => check_sample(me)?,
        "tabdisp" => tabdisp(me, r)?,
        "window" | "win" | "wind" | "windo" => window(me, r)?,
        "class" => return Err(rc_err(198, "class definitions must be followed by { ... }")),
        "classutil" => crate::progclass::classutil(me, r)?,
        "classexit" => crate::progclass::classexit(me, r)?,
        "plugin" => return Err(rc_err(199, "plugin: compiled plugins are not supported by Ferrostat")),
        "javacall" => return Err(rc_err(199, "javacall: Java plugins are not supported by Ferrostat")),
        _ => return Ok(None),
    }))
}

fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

// ---------------------------------------------------------------------------
// program definition, byable, sortpreserve, trace
// ---------------------------------------------------------------------------

/// Store a program defined by `program [define] name [, options]`.
pub fn define_program(me: &mut AdoInterpreter, name: &str, opts: &str, raw: &[String]) -> Result<(), String> {
    let o = split_options(opts);
    if has_opt(&o, "plugin", 6) {
        return Err(rc_err(199, format!("program {name}: compiled plugins are not supported by Ferrostat")));
    }
    // member programs of a class (`program .name` after a class definition)
    let key = if let Some(member) = name.strip_prefix('.') {
        let class = me.prog.class.cur_class.clone().ok_or_else(|| rc_err(198, format!("program .{member}: no class is being defined")))?;
        format!("{class}.{member}")
    } else {
        if me.engine.store.programs.contains_key(name) {
            return Err(rc_err(110, format!("program {name} already defined")));
        }
        name.to_string()
    };
    if let Some(b) = opt_arg(&o, "byable", 2) {
        let mode = b.split(',').next().unwrap_or("").trim().to_string();
        if !matches!(mode.as_str(), "recall" | "onecall") {
            return Err(rc_err(198, format!("byable({b}) invalid")));
        }
    }
    let class = ["rclass", "eclass", "sclass"].iter().find(|c| has_opt(&o, c, c.len())).map(|c| c.to_string()).unwrap_or_else(|| "nclass".into());
    let byable = has_opt(&o, "byable", 2);
    me.engine.store.programs.insert(key.clone(), Program { name: key, body: raw.to_vec(), class, byable, opts: opts.to_string() });
    Ok(())
}

/// Properties of a program (`: properties name`).
pub fn properties(st: &ferrostat_cmd::store::Store, name: &str) -> String {
    st.programs
        .get(name.trim())
        .and_then(|p| opt_arg(&split_options(&p.opts), "properties", 4))
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// Work done when a program starts; undone by [`program_end`].
pub struct ProgramRun {
    sortvar: Option<(String, Vec<String>)>,
    traced: bool,
}

fn trace_on(me: &AdoInterpreter) -> bool {
    me.engine.store.settings.get("trace").map_or(false, |v| v == "on")
}

fn trace_depth(me: &AdoInterpreter) -> usize {
    me.engine.store.settings.get("tracedepth").and_then(|v| v.parse().ok()).unwrap_or(32000)
}

fn trace_rule(level: usize, word: &str, name: &str) -> String {
    let indent = "  ".repeat(level);
    let tail = format!(" {word} {name} ---");
    let dashes = 78usize.saturating_sub(indent.len() + tail.len()).max(3);
    format!("{indent}{}{tail}", "-".repeat(dashes))
}

pub fn program_start(me: &mut AdoInterpreter, prog: &Program) -> ProgramRun {
    if let Some(locals) = me.prog.by_locals.take() {
        for (k, v) in locals {
            me.engine.store.set_local(&k, v);
        }
    }
    let mut traced = false;
    if trace_on(me) {
        me.prog.trace_stack.push(prog.name.clone());
        let level = me.prog.trace_stack.len();
        if level <= trace_depth(me) {
            let line = trace_rule(level, "begin", &prog.name);
            me.out.push(line.clone());
            me.log_write(&line, false);
        }
        traced = true;
    }
    let mut sortvar = None;
    if has_opt(&split_options(&prog.opts), "sortpreserve", 8) {
        me.prog.counter += 1;
        let name = format!("__sortpreserve{:04}", me.prog.counter);
        let ds = &mut me.engine.dataset;
        let mut v = Variable::new(&name, VarType::Double);
        v.data = (0..ds.n_obs).map(|i| Value::Double(i as f64)).collect();
        if ds.add_variable(v).is_ok() {
            sortvar = Some((name, ds.sort_by.clone()));
        }
    }
    ProgramRun { sortvar, traced }
}

pub fn program_end(me: &mut AdoInterpreter, prog: &Program, run: ProgramRun) {
    if let Some((name, sort_by)) = run.sortvar {
        let ds = &mut me.engine.dataset;
        if ds.get_var(&name).is_some() {
            let _ = ds.sort(&[name.clone()], true);
            let _ = ds.drop_variable(&name);
            ds.sort_by = sort_by;
        }
    }
    if run.traced {
        let level = me.prog.trace_stack.len();
        if level <= trace_depth(me) {
            let line = trace_rule(level, "end", &prog.name);
            me.out.push(line.clone());
            me.log_write(&line, false);
        }
        me.prog.trace_stack.pop();
    }
}

/// `set trace on`: show each line executed inside programs, and its
/// macro-expanded form when that differs.
pub fn trace_line(me: &mut AdoInterpreter, raw: &str, expanded: &str) {
    let level = me.prog.trace_stack.len();
    if level == 0 || !trace_on(me) || level > trace_depth(me) {
        return;
    }
    let indent = "  ".repeat(level);
    let mut lines = vec![format!("{indent}- {}", raw.trim())];
    if raw.trim() != expanded.trim() && me.engine.store.settings.get("traceexpand").map_or(true, |v| v != "off") {
        lines.push(format!("{indent}= {}", expanded.trim()));
    }
    for l in lines {
        me.out.push(l.clone());
        me.log_write(&l, false);
    }
}

/// `by varlist: program`: byable(recall) programs run once per group with
/// `_by()` true; byable(onecall) programs run once with `_byvars` and
/// `_byrc0` defined. Other programs may not be combined with by.
pub fn by_program(me: &mut AdoInterpreter, byvars: &[String], groups: &ferrostat_cmd::eval::Groups, command: &str, opts: &str) -> Result<Option<Flow>, String> {
    let (cverb, _) = crate::interpreter::split_verb(command);
    let Some(prog) = me.engine.store.programs.get(&cverb).cloned() else { return Ok(None) };
    if !prog.byable {
        return Err(rc_err(190, format!("{cverb} may not be combined with by")));
    }
    let popts = split_options(&prog.opts);
    let mode = opt_arg(&popts, "byable", 2).unwrap_or_default();
    let rc0 = opts.contains("rc0");
    let n = me.engine.dataset.n_obs;
    if mode.trim_start().starts_with("onecall") {
        me.prog.by_locals = Some(vec![("_byvars".into(), byvars.join(" ")), ("_byrc0".into(), if rc0 { ", rc0".into() } else { String::new() })]);
        me.engine.store.by_mask = Some(vec![true; n]);
        let r = me.exec_text(command, None);
        me.engine.store.by_mask = None;
        me.prog.by_locals = None;
        return r.map(Some);
    }
    let noheader = mode.contains("noheader");
    let starts: Vec<usize> = (0..n).filter(|&i| groups.start[i] == i).collect();
    let mut result = Ok(Flow::Next);
    for (k, &s) in starts.iter().enumerate() {
        let e = groups.end[s];
        if !noheader {
            let header: Vec<String> = byvars
                .iter()
                .map(|v| {
                    let var = me.engine.dataset.get_var(v).unwrap();
                    format!("{} = {}", v, util::cell_text(&me.engine.dataset, var, s, true))
                })
                .collect();
            me.emit(&format!("\n{}\n-> {}\n", "-".repeat(78), header.join(", ")));
        }
        me.engine.store.by_mask = Some((0..n).map(|i| i >= s && i < e).collect());
        me.engine.store.settings.insert("_byindex".into(), (k + 1).to_string());
        me.engine.store.settings.insert("_bylastcall".into(), if k + 1 == starts.len() { "1" } else { "0" }.into());
        let r = me.exec_text(command, None);
        me.engine.store.by_mask = None;
        if let Err(err) = r {
            if !rc0 {
                result = Err(err);
                break;
            }
        }
    }
    me.engine.store.settings.remove("_byindex");
    me.engine.store.settings.remove("_bylastcall");
    result.map(Some)
}

// ---------------------------------------------------------------------------
// sysdir, personal, adopath, findfile, viewsource
// ---------------------------------------------------------------------------

const SYSDIRS: &[&str] = &["BASE", "SITE", "PLUS", "PERSONAL", "OLDPLACE"];

fn c_str(me: &AdoInterpreter, key: &str) -> String {
    let env = Env::with_store(&me.engine.dataset, &me.engine.store);
    match ferrostat_cmd::eval::c_value(&env, key) {
        Val::Str(s) => s,
        Val::Num(n) => macro_number(n),
    }
}

/// Directory of a codeword (BASE, PLUS, ...), with a trailing separator.
pub fn sysdir_path(me: &AdoInterpreter, code: &str) -> Option<String> {
    let c = code.to_uppercase();
    SYSDIRS.contains(&c.as_str()).then(|| c_str(me, &format!("sysdir_{}", c.to_lowercase())))
}

fn with_sep(p: &str) -> String {
    if p.is_empty() || p.ends_with('/') || p.ends_with('\\') { p.to_string() } else { format!("{p}/") }
}

fn sysdir(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let w = split_top_level_words(r);
    match w.first().map(|s| s.to_lowercase()).as_deref() {
        None | Some("list") => Ok(SYSDIRS.iter().map(|c| format!("{c:>8}:  {}", sysdir_path(me, c).unwrap_or_default())).collect::<Vec<_>>().join("\n")),
        Some("set") => {
            let code = w.get(1).map(|s| s.to_uppercase()).ok_or_else(|| rc_err(198, "sysdir set codeword path"))?;
            if !SYSDIRS.contains(&code.as_str()) {
                return Err(rc_err(198, format!("{code} invalid codeword")));
            }
            let path = w.get(2).map(|p| unquote(p)).ok_or_else(|| rc_err(198, "path required"))?;
            me.engine.store.settings.insert(format!("sysdir_{}", code.to_lowercase()), with_sep(&path));
            Ok(String::new())
        }
        Some(other) => Err(rc_err(198, format!("sysdir {other}: invalid subcommand"))),
    }
}

fn personal(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let dir = sysdir_path(me, "PERSONAL").unwrap_or_default();
    match first_word(r) {
        "" => Ok(format!("your personal ado-directory is {dir}")),
        "dir" => {
            let mut files: Vec<String> = vfs::list(dir.trim_end_matches('/')).map(|l| l.into_iter().filter(|e| !e.1).map(|e| e.0).collect()).unwrap_or_default();
            files.sort();
            Ok(if files.is_empty() { format!("(no files in {dir})") } else { files.join("\n") })
        }
        other => Err(rc_err(198, format!("personal {other}: invalid subcommand"))),
    }
}

/// The ado-path entries (codewords or directories).
fn adopath_entries(me: &AdoInterpreter) -> Vec<String> {
    c_str(me, "adopath").split(';').filter(|s| !s.is_empty()).map(String::from).collect()
}

fn entry_dir(me: &AdoInterpreter, e: &str) -> String {
    sysdir_path(me, e).unwrap_or_else(|| e.to_string())
}

fn adopath(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let mut entries = adopath_entries(me);
    let (op, arg) = if let Some(a) = r.strip_prefix("++") {
        ("++", a)
    } else if let Some(a) = r.strip_prefix('+') {
        ("+", a)
    } else if let Some(a) = r.strip_prefix('-') {
        ("-", a)
    } else {
        ("", r)
    };
    let arg = unquote(arg.trim());
    match op {
        "" => {
            let lines: Vec<String> = entries
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let code = if sysdir_path(me, e).is_some() { format!("({})", e.to_uppercase()) } else { String::new() };
                    format!("  [{}]  {:<10}  \"{}\"", i + 1, code, entry_dir(me, e))
                })
                .collect();
            return Ok(lines.join("\n"));
        }
        "+" | "++" => {
            if arg.is_empty() {
                return Err(rc_err(198, "directory or codeword required"));
            }
            let e = if SYSDIRS.contains(&arg.to_uppercase().as_str()) { arg.to_uppercase() } else { arg.clone() };
            entries.retain(|x| *x != e);
            if op == "+" { entries.push(e) } else { entries.insert(0, e) }
        }
        _ => {
            if let Ok(k) = arg.parse::<usize>() {
                if k == 0 || k > entries.len() {
                    return Err(rc_err(198, format!("adopath entry {k} not found")));
                }
                entries.remove(k - 1);
            } else {
                let before = entries.len();
                entries.retain(|x| !x.eq_ignore_ascii_case(&arg) && entry_dir(me, x) != arg);
                if entries.len() == before {
                    return Err(rc_err(198, format!("{arg} not in adopath")));
                }
            }
        }
    }
    me.engine.store.settings.insert("adopath".into(), entries.join(";"));
    adopath(me, "")
}

/// Search the ado-path (or `path`) for a file; with `descend`, also the
/// letter subdirectory (`plus/r/file.ado`).
pub fn find_on_path(me: &AdoInterpreter, file: &str, path: Option<&str>, descend: bool, all: bool) -> Vec<String> {
    let dirs: Vec<String> = match path {
        Some(p) => p.split(';').map(|e| entry_dir(me, e.trim())).collect(),
        None => adopath_entries(me).iter().map(|e| entry_dir(me, e)).collect(),
    };
    let mut found = Vec::new();
    if file.contains('/') || file.contains('\\') {
        if vfs::exists(file) {
            found.push(file.to_string());
        }
        return found;
    }
    let first = file.chars().next().map(|c| c.to_ascii_lowercase()).unwrap_or('_');
    let letter = if first.is_ascii_alphabetic() { first } else { '_' };
    for d in dirs {
        if d.is_empty() {
            continue;
        }
        let base = if d == "." { String::new() } else { with_sep(&d) };
        let mut cands = vec![format!("{}{file}", if base.is_empty() { "./" } else { &base })];
        if descend {
            cands.push(format!("{}{letter}/{file}", if base.is_empty() { "./" } else { &base }));
        }
        for c in cands {
            if vfs::exists(&c) && !found.contains(&c) {
                found.push(c);
                if !all {
                    return found;
                }
            }
        }
    }
    found
}

fn findfile(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let file = unquote(body.trim());
    if file.is_empty() {
        return Err(rc_err(198, "filename required"));
    }
    let path = opt_arg(&opts, "path", 4).map(|p| unquote(&p));
    let all = has_opt(&opts, "all", 3);
    let found = find_on_path(me, &file, path.as_deref(), !has_opt(&opts, "nodescend", 6), all);
    if found.is_empty() {
        return Err(rc_err(601, format!("file \"{file}\" not found")));
    }
    let res = me.engine.store.begin_r();
    if all {
        res.set_str("fn", found.iter().map(|f| format!("\"{f}\"")).collect::<Vec<_>>().join(" "));
    } else {
        res.set_str("fn", found[0].clone());
    }
    Ok(found.join("\n"))
}

fn viewsource(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let file = unquote(r.trim());
    if file.is_empty() {
        return Err(rc_err(198, "filename required"));
    }
    let found = find_on_path(me, &file, None, true, false);
    let p = found.first().ok_or_else(|| rc_err(601, format!("file \"{file}\" not found")))?;
    Ok(vfs::read_to_string(p).map_err(|_| rc_err(601, format!("file \"{file}\" not found")))?.trim_end().to_string())
}

// ---------------------------------------------------------------------------
// discard, pause, window
// ---------------------------------------------------------------------------

fn discard(me: &mut AdoInterpreter) -> String {
    let st = &mut me.engine.store;
    st.e = Results::new();
    st.r = Results::new();
    st.s = Results::new();
    me.prog.sersets = Default::default();
    me.prog.class.discard();
    String::new()
}

fn pause(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    match r.trim() {
        "on" => me.prog.pause_on = true,
        "off" => me.prog.pause_on = false,
        msg if me.prog.pause_on => {
            // there is no console to read from: report and continue
            let m = if msg.is_empty() { String::new() } else { format!("  {}", unquote(msg)) };
            return Ok(format!("pause:{m}\n(pause ignored: not interactive; continuing)"));
        }
        _ => {}
    }
    Ok(String::new())
}

/// `window fopen|fsave|manage|menu|push|stopbox|dialog|...`: dialogs and
/// menus do not exist here; the commands do nothing (`fopen`/`fsave` leave
/// the macro empty), except `window stopbox stop`, which stops with r(1).
fn window(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let w = split_top_level_words(r);
    let sub = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    match sub.as_str() {
        "fopen" | "fsave" => {
            let mac = w.get(1).ok_or_else(|| rc_err(198, format!("window {sub} macname \"title\" \"filter\"")))?;
            me.engine.store.set_local(mac, String::new());
            Ok(String::new())
        }
        "stopbox" => {
            let kind = w.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
            let text: Vec<String> = w.iter().skip(2).map(|s| unquote(s)).collect();
            match kind.as_str() {
                "stop" => Err(rc_err(1, text.join("\n"))),
                "note" | "rusure" => Ok(text.join("\n")),
                _ => Err(rc_err(198, "window stopbox stop|note|rusure [\"text\" ...]")),
            }
        }
        "manage" | "man" | "menu" | "push" | "dialog" | "control" | "list" => Ok(String::new()),
        "" => Err(rc_err(198, "window subcommand required")),
        other => Err(rc_err(198, format!("window {other}: invalid subcommand"))),
    }
}

// ---------------------------------------------------------------------------
// mark, markin, unab, unabcmd, fvexpand
// ---------------------------------------------------------------------------

fn set_byte_var(me: &mut AdoInterpreter, name: &str, mask: &[bool], create: bool) -> Result<(), String> {
    let ds = &mut me.engine.dataset;
    let data: Vec<Value> = mask.iter().map(|&m| Value::Byte(m as i8)).collect();
    if create {
        let mut v = Variable::new(name, VarType::Byte);
        v.data = data;
        ds.add_variable(v)
    } else {
        let v = ds.get_var_mut(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
        v.data = data;
        v.var_type = VarType::Byte;
        Ok(())
    }
}

/// `mark newvar [if] [in] [weight] [, zeroweight noweight]`,
/// `markin touse [if] [in]`.
fn mark(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    let c: Command = parse_line(&format!("{verb} {r}")).map_err(|e| rc_err(198, e.message))?;
    let name = c.main.trim().to_string();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(rc_err(198, format!("{verb} requires one variable name")));
    }
    let ds = &me.engine.dataset;
    let st = &me.engine.store;
    let mut mask = util::touse(ds, st, &c)?;
    if verb == "mark" {
        if ds.get_var(&name).is_some() {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        varlist::check_name(&name)?;
        if !c.has_opt("noweight", 5) {
            let base = mask.clone();
            if let Some(w) = util::weights(ds, st, &c, &mut mask, "fapi")? {
                let zero = c.has_opt("zeroweight", 4);
                for (i, m) in mask.iter_mut().enumerate() {
                    *m = base[i] && !is_missing(w.w[i]) && (zero || w.w[i] != 0.0);
                }
            }
        }
        set_byte_var(me, &name, &mask, true)?;
    } else {
        let cur = util::col(ds, &name)?;
        for (i, m) in mask.iter_mut().enumerate() {
            *m = *m && cur[i] != 0.0 && !is_missing(cur[i]);
        }
        set_byte_var(me, &name, &mask, false)?;
    }
    Ok(String::new())
}

/// Expand `L(1/2).x` style operator ranges into `L.x L2.x`.
fn expand_op_ranges(tok: &str) -> Result<Vec<String>, String> {
    let Some(open) = tok.find('(') else { return Ok(vec![tok.to_string()]) };
    let Some(close) = tok[open..].find(").").map(|i| i + open) else { return Ok(vec![tok.to_string()]) };
    let op = &tok[..open];
    if op.is_empty() || !op.chars().all(|c| matches!(c.to_ascii_uppercase(), 'L' | 'F' | 'D' | 'S')) {
        return Ok(vec![tok.to_string()]);
    }
    let var = &tok[close + 2..];
    let nums = crate::macros::numlist(&tok[open + 1..close])?;
    Ok(nums
        .iter()
        .map(|&k| {
            let k = k as i64;
            if k == 0 {
                var.to_string()
            } else if k == 1 {
                format!("{op}.{var}")
            } else {
                format!("{op}{k}.{var}")
            }
        })
        .collect())
}

/// `unab`, `tsunab`, `fvunab lmacname : varlist [, min(#) max(#) name(str)]`.
fn unab(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    let colon = r.find(':').ok_or_else(|| rc_err(198, format!("{verb} lmacname : varlist")))?;
    let mac = r[..colon].trim().to_string();
    let (list, opts) = split_opts(&r[colon + 1..]);
    let mut toks = Vec::new();
    for t in split_top_level_words(&list) {
        toks.extend(expand_op_ranges(&t)?);
    }
    let vars = varlist::expand(&me.engine.dataset, &toks.join(" "))?;
    let label = opt_arg(&opts, "name", 1).map(|s| unquote(&s)).unwrap_or_default();
    for v in &vars {
        let (fv, ts) = op_kinds(v);
        match verb {
            "unab" if fv || ts => return Err(rc_err(101, format!("{label}factor-variable and time-series operators not allowed"))),
            "tsunab" if fv => return Err(rc_err(101, format!("{label}factor-variable operators not allowed"))),
            _ => {}
        }
    }
    if let Some(m) = opt_arg(&opts, "min", 3).and_then(|v| v.trim().parse::<usize>().ok()) {
        if vars.len() < m {
            return Err(rc_err(102, format!("{label}too few variables specified")));
        }
    }
    if let Some(m) = opt_arg(&opts, "max", 3).and_then(|v| v.trim().parse::<usize>().ok()) {
        if vars.len() > m {
            return Err(rc_err(103, format!("{label}too many variables specified")));
        }
    }
    me.engine.store.set_local(&mac, vars.join(" "));
    Ok(String::new())
}

/// `unabcmd lmacname : abbreviated-command`.
fn unabcmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let colon = r.find(':').ok_or_else(|| rc_err(198, "unabcmd lmacname : command"))?;
    let mac = r[..colon].trim().to_string();
    let word = r[colon + 1..].trim().to_string();
    let full = me.engine.resolve_verb(&word);
    let known = me.engine.is_command(&full)
        || crate::progcmd::INTERP_COMMANDS.contains(&full.as_str())
        || COMMANDS.contains(&full.as_str())
        || me.engine.store.programs.contains_key(&word)
        || matches!(full.as_str(), "clear" | "set" | "preserve" | "restore" | "sysuse" | "version");
    if !known {
        return Err(rc_err(199, format!("command {word} is unrecognized")));
    }
    me.engine.store.set_local(&mac, full);
    Ok(String::new())
}

/// Undo the `o.` marks that collinearity checks add to names.
fn unomit(n: &str) -> String {
    n.split('#')
        .map(|p| {
            if let Some(r) = p.strip_prefix("o.") {
                return r.to_string();
            }
            if let Some(r) = p.strip_prefix("co.") {
                return format!("c.{r}");
            }
            let digits: String = p.chars().take_while(|c| c.is_ascii_digit()).collect();
            let rest = &p[digits.len()..];
            if !digits.is_empty() {
                if let Some(r) = rest.strip_prefix("bo.") {
                    return format!("{digits}b.{r}");
                }
                if let Some(r) = rest.strip_prefix("o.") {
                    return format!("{digits}.{r}");
                }
            }
            p.to_string()
        })
        .collect::<Vec<_>>()
        .join("#")
}

/// (factor-variable operators present, time-series operators present) in a token list.
fn op_kinds(text: &str) -> (bool, bool) {
    let (mut fv, mut ts) = (false, false);
    for tok in split_top_level_words(text) {
        if tok.contains('#') {
            fv = true;
        }
        for part in tok.split('#') {
            let Some((op, _)) = part.split_once('.') else { continue };
            let op = op.to_lowercase();
            let first = op.chars().next().unwrap_or(' ');
            if matches!(first, 'l' | 'f' | 'd' | 's') && op[1..].chars().all(|c| c.is_ascii_digit() || matches!(c, '(' | ')' | '/' | ' ' | 'l' | 'f' | 'd' | 's')) {
                ts = true;
            } else if !op.is_empty() {
                fv = true;
            }
        }
    }
    (fv, ts)
}

fn has_fv(text: &str) -> bool {
    op_kinds(text).0
}

fn has_ts(text: &str) -> bool {
    op_kinds(text).1
}

/// `fvexpand varlist [if] [in]`: r(varlist) with factor levels spelled out.
fn fvexpand(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let c: Command = parse_line(&format!("fvexpand {r}")).map_err(|e| rc_err(198, e.message))?;
    let text = c.main.trim().to_string();
    let (fv, ts) = (has_fv(&text), has_ts(&text));
    let names: Vec<String> = if fv {
        let d = est::build(&me.engine.dataset, &me.engine.store, &c, est::DesignSpec { depvar: None, indep: &text, noconstant: true, weights: "", extra: vec![] })?;
        d.names.iter().map(|n| unomit(n)).collect()
    } else {
        varlist::expand(&me.engine.dataset, &text)?
    };
    let res = me.engine.store.begin_r();
    res.set_str("varlist", names.join(" "));
    if fv {
        res.set_str("fvops", "true");
    }
    if ts {
        res.set_str("tsops", "true");
    }
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// _rmcoll, _rmdcoll
// ---------------------------------------------------------------------------

/// `_rmcoll varlist [if] [in] [weight] [, noconstant collinear expand forcedrop]`
/// and `_rmdcoll depvar varlist ...`: r(varlist) with omitted variables
/// marked `o.` (or dropped with forcedrop) and r(k_omitted).
fn rmcoll(me: &mut AdoInterpreter, dcoll: bool, r: &str) -> Result<String, String> {
    let c: Command = parse_line(&format!("rmcoll {r}")).map_err(|e| rc_err(198, e.message))?;
    let mut text = c.main.trim().to_string();
    let mut depvar = None;
    if dcoll {
        let w = split_top_level_words(&text);
        let d = w.first().cloned().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        depvar = Some(varlist::expand(&me.engine.dataset, &d)?.join(" "));
        text = w[1..].join(" ");
    }
    let noconstant = c.has_opt("noconstant", 5);
    let fv = has_fv(&text);
    let (ds, st) = (&me.engine.dataset, &me.engine.store);
    if text.is_empty() {
        let res = me.engine.store.begin_r();
        res.set_str("varlist", "");
        res.set("k_omitted", 0.0);
        return Ok(String::new());
    }
    let d = est::build(ds, st, &c, est::DesignSpec { depvar: None, indep: &text, noconstant, weights: "fapiw", extra: depvar.iter().cloned().collect() })?;
    let collinear = c.has_opt("collinear", 4);
    let forcedrop = c.has_opt("forcedrop", 5);
    let mut out = Vec::new();
    let mut notes = Vec::new();
    let mut k_omitted = 0;
    let given = if fv { Vec::new() } else { varlist::expand(ds, &text)? };
    let mut kept_cols = Vec::new();
    for (j, n) in d.names.iter().enumerate() {
        if n == "_cons" {
            continue;
        }
        let omitted = d.omitted[j] && !d.is_base[j];
        let shown = if fv { n.clone() } else { given.get(j).cloned().unwrap_or_else(|| unomit(n)) };
        if omitted && !collinear {
            k_omitted += 1;
            notes.push(format!("note: {} omitted because of collinearity.", unomit(n)));
            if !forcedrop {
                out.push(if fv { n.clone() } else { format!("o.{shown}") });
            }
        } else {
            out.push(if collinear { unomit(&shown) } else { shown });
            if !d.omitted[j] {
                kept_cols.push(j);
            }
        }
    }
    if let Some(dv) = &depvar {
        // the dependent variable may not be a linear function of the regressors
        let y = util::col(ds, dv)?;
        let yv: Vec<f64> = d.rows.iter().map(|&i| y[i]).collect();
        let mut cols: Vec<&Vec<f64>> = kept_cols.iter().map(|&j| &d.cols[j]).collect();
        let ones = vec![1.0; d.rows.len()];
        if !noconstant {
            cols.push(&ones);
        }
        if collinear_with(&yv, &cols) {
            return Err(rc_err(459, format!("{dv} collinear with the independent variables")));
        }
    }
    let res = me.engine.store.begin_r();
    res.set_str("varlist", out.join(" "));
    res.set("k_omitted", k_omitted as f64);
    Ok(notes.join("\n"))
}

/// Is `y` (nearly) an exact linear combination of `cols`?
fn collinear_with(y: &[f64], cols: &[&Vec<f64>]) -> bool {
    let n = y.len();
    let tss: f64 = {
        let m = y.iter().sum::<f64>() / n.max(1) as f64;
        y.iter().map(|v| (v - m).powi(2)).sum::<f64>() + y.iter().map(|v| v * v).sum::<f64>() * 1e-30
    };
    if cols.is_empty() {
        return tss == 0.0;
    }
    let k = cols.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for a in 0..k {
        for b in 0..k {
            xtx[(a, b)] = (0..n).map(|t| cols[a][t] * cols[b][t]).sum();
        }
        xty[a] = (0..n).map(|t| cols[a][t] * y[t]).sum();
    }
    let (inv, _) = xtx.sym_inverse(1e-13);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    let rss: f64 = (0..n).map(|t| (y[t] - (0..k).map(|a| b[a] * cols[a][t]).sum::<f64>()).powi(2)).sum();
    let yy: f64 = y.iter().map(|v| v * v).sum();
    rss <= 1e-12 * yy.max(1e-300)
}

// ---------------------------------------------------------------------------
// makecns, matcproc
// ---------------------------------------------------------------------------

/// `makecns [clist|matname] [, nocnsnotes displaycns r]`: build the
/// constraint matrix [R | r] for the coefficients in e(b) and post it as
/// e(Cns) (or return it in r(Cns) with option `r`).
fn makecns(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let b = me.engine.store.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names: Vec<String> = b.colnames.clone();
    let p = names.len();
    let notes = !has_opt(&opts, "nocnsnotes", 5);
    let display = has_opt(&opts, "displaycns", 7);
    let to_r = opts.iter().any(|(o, _)| o == "r");
    let mut lines = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut used = Vec::new();
    let mut texts = Vec::new();
    let body = body.trim();
    if !body.is_empty() && !body.chars().next().unwrap().is_ascii_digit() && me.engine.store.matrices.contains_key(body) {
        let m = me.engine.store.matrices[body].clone();
        if m.cols != p + 1 {
            return Err(rc_err(503, "conformability error"));
        }
        for i in 0..m.rows {
            rows.push(m.row(i).to_vec());
        }
    } else {
        let mut defined: Vec<(usize, String)> = me.engine.store.settings.iter().filter_map(|(k, s)| k.strip_prefix("_constraint.").and_then(|n| n.parse().ok()).map(|n| (n, s.clone()))).collect();
        defined.sort();
        // constraint lists: `1 2`, `1-3`, `1,4`, `2/5`
        let clist: String = body.replace(',', " ").split_whitespace().map(|t| t.replacen('-', "/", 1)).collect::<Vec<_>>().join(" ");
        let wanted: Vec<usize> = if body.is_empty() { defined.iter().map(|d| d.0).collect() } else { crate::macros::numlist(&clist)?.into_iter().map(|v| v as usize).collect() };
        for k in wanted {
            let Some((_, spec)) = defined.iter().find(|d| d.0 == k) else {
                if notes {
                    lines.push(format!("note: constraint number {k} undefined"));
                }
                continue;
            };
            match ferrostat_cmd::xmodels::parse_constraint(spec, &names) {
                Ok((coef, cons)) => {
                    let mut row = coef;
                    row.push(cons);
                    // drop constraints implied by earlier ones
                    let mut test = rows.clone();
                    test.push(row.clone());
                    if rank(&test, p + 1) < test.len() {
                        if rank(&test.iter().map(|r| r[..p].to_vec()).collect::<Vec<_>>(), p) < test.len() && rank(&test, p + 1) > rank(&test.iter().map(|r| r[..p].to_vec()).collect::<Vec<_>>(), p) {
                            return Err(rc_err(412, format!("constraint {k}: constraints inconsistent")));
                        }
                        if notes {
                            lines.push(format!("note: constraint number {k} dropped"));
                        }
                        continue;
                    }
                    rows.push(row);
                    used.push(k);
                    texts.push(spec.clone());
                }
                Err(e) => {
                    if notes {
                        lines.push(format!("note: constraint number {k} caused error r({})", crate::interpreter::rc_of(&e)));
                    }
                }
            }
        }
    }
    if display {
        for (i, t) in texts.iter().enumerate() {
            lines.push(format!(" ({:>2})  {t}", i + 1));
        }
    }
    let k = rows.len();
    let res_mat = if k > 0 {
        let mut cn = names.clone();
        cn.push("_r".into());
        Some(Matrix::from_rows(&rows).with_names((1..=k).map(|i| format!("r{i}")).collect(), cn))
    } else {
        None
    };
    let res = me.engine.store.begin_r();
    res.set("k", k as f64);
    res.set_str("clist", used.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(" "));
    if let Some(m) = res_mat {
        if to_r {
            res.set_mat("Cns", m);
        } else {
            me.engine.store.e.set_mat("Cns", m);
        }
    }
    Ok(lines.join("\n"))
}

/// Numerical rank of the rows (each of length `p`).
fn rank(rows: &[Vec<f64>], p: usize) -> usize {
    let mut m: Vec<Vec<f64>> = rows.iter().map(|r| r[..p].to_vec()).collect();
    let scale = m.iter().flat_map(|r| r.iter()).fold(0.0f64, |a, b| a.max(b.abs())).max(1e-300);
    let mut rank = 0;
    let mut col = 0;
    while rank < m.len() && col < p {
        let piv = (rank..m.len()).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs())).unwrap();
        if m[piv][col].abs() <= 1e-10 * scale {
            col += 1;
            continue;
        }
        m.swap(rank, piv);
        for i in rank + 1..m.len() {
            let f = m[i][col] / m[rank][col];
            for j in col..p {
                m[i][j] -= f * m[rank][j];
            }
        }
        rank += 1;
        col += 1;
    }
    rank
}

/// `matcproc T a C`: with constraints R b = r in C = [R | r], b = T f + a'
/// for free parameters f (T spans the null space of R).
fn matcproc(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    if w.len() != 3 {
        return Err(rc_err(198, "matcproc T a C"));
    }
    let c = me.engine.store.matrices.get(w[2]).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", w[2])))?;
    if c.cols < 2 {
        return Err(rc_err(503, "conformability error"));
    }
    let p = c.cols - 1;
    let k = c.rows;
    let mut rm = Matrix::zeros(k, p);
    let mut rv = vec![0.0; k];
    for i in 0..k {
        for j in 0..p {
            rm[(i, j)] = c[(i, j)];
        }
        rv[i] = c[(i, p)];
    }
    let names: Vec<String> = c.colnames[..p].to_vec();
    let rtr = rm.transpose().mul(&rm)?;
    let (vals, vecs) = rtr.sym_eigen();
    let tol = vals.first().copied().unwrap_or(0.0).abs().max(1e-300) * 1e-10;
    let free: Vec<usize> = (0..p).filter(|&j| vals[j].abs() <= tol).collect();
    let mut t = Matrix::zeros(p, free.len());
    for (nj, &j) in free.iter().enumerate() {
        for i in 0..p {
            t[(i, nj)] = vecs[(i, j)];
        }
    }
    // minimum-norm particular solution a = R'(RR')^- r
    let rrt = rm.mul(&rm.transpose())?;
    let (g, _) = rrt.sym_inverse(1e-12);
    let gr: Vec<f64> = (0..k).map(|i| (0..k).map(|j| g[(i, j)] * rv[j]).sum()).collect();
    let a: Vec<f64> = (0..p).map(|j| (0..k).map(|i| rm[(i, j)] * gr[i]).sum()).collect();
    let tn: Vec<String> = (1..=free.len()).map(|i| format!("c{i}")).collect();
    me.engine.store.matrices.insert(w[0].to_string(), t.with_names(names.clone(), tn));
    me.engine.store.matrices.insert(w[1].to_string(), Matrix::row_vector(&a).with_names(vec!["r1".into()], names));
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// signestimationsample, checkestimationsample
// ---------------------------------------------------------------------------

fn sample_signature(me: &AdoInterpreter, vars: &[String]) -> Result<String, String> {
    let ds = &me.engine.dataset;
    let sample = me.engine.store.e.sample.clone();
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.as_ref().map_or(true, |s| s.get(i).copied().unwrap_or(false))).collect();
    // FNV-1a over names, types and values
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x84222325cbf29ce4;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h1 = (h1 ^ b as u64).wrapping_mul(0x100000001b3);
            h2 = (h2 ^ b as u64).wrapping_mul(0x1000193).rotate_left(5);
        }
    };
    for v in vars {
        let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
        feed(v.as_bytes());
        for &i in &rows {
            match &var.data[i] {
                Value::Str(s) => feed(s.as_bytes()),
                x => feed(&x.num().to_bits().to_le_bytes()),
            }
        }
    }
    Ok(format!("{}:{}({}):{}:{}", rows.len(), vars.len(), h1 % 100000, h1 >> 32, h2 >> 32))
}

/// `signestimationsample varlist`: store e(datasignature) and e(datasignaturevars).
fn sign_sample(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    if me.engine.store.e.is_empty() {
        return Err(rc_err(301, "last estimates not found"));
    }
    let vars = if r.trim().is_empty() { Vec::new() } else { varlist::expand(&me.engine.dataset, r)? };
    let sig = sample_signature(me, &vars)?;
    let e = &mut me.engine.store.e;
    e.set_str("datasignature", sig);
    e.set_str("datasignaturevars", vars.join(" "));
    Ok(String::new())
}

/// `checkestimationsample`: error 459 when the estimation data changed.
fn check_sample(me: &mut AdoInterpreter) -> Result<String, String> {
    let e = &me.engine.store.e;
    let Some(sig) = e.macros.get("datasignature").cloned() else {
        if e.is_empty() {
            return Err(rc_err(301, "last estimates not found"));
        }
        return Ok(String::new());
    };
    let vars: Vec<String> = e.macros.get("datasignaturevars").map(|v| v.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let ok = vars.iter().all(|v| me.engine.dataset.get_var(v).is_some()) && sample_signature(me, &vars)? == sig;
    if !ok {
        return Err(rc_err(459, "data have changed since estimation"));
    }
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// tabdisp
// ---------------------------------------------------------------------------

/// `tabdisp rowvar [colvar] [if] [in], cellvar(varlist) [format(%fmt) left
/// center missing totals cellwidth(#) stubwidth(#)]`: show the values of
/// cell variables in a table; each cell must come from one observation.
fn tabdisp(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let c: Command = parse_line(&format!("tabdisp {r}")).map_err(|e| rc_err(198, e.message))?;
    let ds = &me.engine.dataset;
    let st = &me.engine.store;
    let dims = varlist::expand(ds, &c.main)?;
    if dims.is_empty() || dims.len() > 3 {
        return Err(rc_err(198, "tabdisp rowvar [colvar [supercolvar]], cellvar(varlist)"));
    }
    if dims.len() == 3 {
        return Err(rc_err(198, "tabdisp: supercolumn variables are not supported"));
    }
    let cells = varlist::expand(ds, c.opt("cellvar", 4).ok_or_else(|| rc_err(198, "option cellvar() required"))?)?;
    if cells.len() > 5 {
        return Err(rc_err(103, "too many cell variables specified"));
    }
    let missing = c.has_opt("missing", 1);
    let fmt = c.opt("format", 1).map(|s| s.trim().to_string());
    let left = c.has_opt("left", 4);
    let center = c.has_opt("center", 3);
    let mask = util::touse(ds, st, &c)?;
    let label = |v: &str, i: usize| -> String {
        let var = ds.get_var(v).unwrap();
        util::cell_text(ds, var, i, true).trim().to_string()
    };
    let key_missing = |v: &str, i: usize| -> bool {
        let var = ds.get_var(v).unwrap();
        match &var.data[i] {
            Value::Str(s) => s.is_empty(),
            x => is_missing(x.num()),
        }
    };
    // ordered levels of the row and column variables
    let order_key = |v: &str, i: usize| -> (f64, String) {
        let var = ds.get_var(v).unwrap();
        match &var.data[i] {
            Value::Str(s) => (0.0, s.clone()),
            x => (x.num(), String::new()),
        }
    };
    let rows_used: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && (missing || dims.iter().all(|d| !key_missing(d, i)))).collect();
    let levels = |v: &str| -> Vec<(f64, String, String)> {
        let mut l: Vec<(f64, String, String)> = rows_used.iter().map(|&i| {
            let (a, b) = order_key(v, i);
            (a, b, label(v, i))
        }).collect();
        l.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        l.dedup_by(|a, b| a.0.to_bits() == b.0.to_bits() && a.1 == b.1);
        l
    };
    let rl = levels(&dims[0]);
    let cl = if dims.len() > 1 { levels(&dims[1]) } else { vec![(0.0, String::new(), String::new())] };
    let mut grid: Vec<Vec<Option<usize>>> = vec![vec![None; cl.len()]; rl.len()];
    for &i in &rows_used {
        let rk = order_key(&dims[0], i);
        let ri = rl.iter().position(|l| l.0.to_bits() == rk.0.to_bits() && l.1 == rk.1).unwrap();
        let ci = if dims.len() > 1 {
            let ck = order_key(&dims[1], i);
            cl.iter().position(|l| l.0.to_bits() == ck.0.to_bits() && l.1 == ck.1).unwrap()
        } else {
            0
        };
        if grid[ri][ci].is_some() {
            return Err(rc_err(459, format!("observations do not uniquely identify cells; {} not unique", dims.join(" and "))));
        }
        grid[ri][ci] = Some(i);
    }
    let cell_text = |v: &str, i: usize| -> String {
        let var = ds.get_var(v).unwrap();
        match (&var.data[i], &fmt) {
            (Value::Str(s), _) => s.clone(),
            (x, Some(f)) => format::fmt(x.num(), f).trim().to_string(),
            (x, None) if is_missing(x.num()) => String::new(),
            _ => util::cell_text(ds, var, i, true).trim().to_string(),
        }
    };
    let mut cw = c.opt("cellwidth", 5).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
    if cw == 0 {
        cw = 9;
        for row in &grid {
            for cell in row.iter().flatten() {
                for v in &cells {
                    cw = cw.max(cell_text(v, *cell).chars().count() + 1);
                }
            }
        }
        for l in &cl {
            cw = cw.max(l.2.chars().count() + 1);
        }
        if dims.len() == 1 {
            for v in &cells {
                cw = cw.max(v.chars().count() + 1);
            }
        }
    }
    let sw = c.opt("stubwidth", 4).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or_else(|| rl.iter().map(|l| l.2.chars().count()).max().unwrap_or(0).max(dims[0].chars().count()).max(8));
    let align = |s: &str, w: usize| -> String {
        let n = s.chars().count();
        if n >= w {
            return s.to_string();
        }
        if left {
            format!("{s}{}", " ".repeat(w - n))
        } else if center {
            let l = (w - n) / 2;
            format!("{}{s}{}", " ".repeat(l), " ".repeat(w - n - l))
        } else {
            format!("{}{s}", " ".repeat(w - n))
        }
    };
    let ncol = cl.len();
    let body_w = 1 + ncol * (cw + 1);
    let rule = "-".repeat(sw + 2 + body_w);
    let mut out = vec![rule.clone()];
    if dims.len() > 1 {
        let title = dims[1].clone();
        let pad = body_w.saturating_sub(title.len()) / 2;
        out.push(format!("{} | {}{}", " ".repeat(sw), " ".repeat(pad), title).trim_end().to_string());
        let heads: Vec<String> = cl.iter().map(|l| align(&l.2, cw)).collect();
        out.push(format!("{:>sw$} | {}", dims[0], heads.join(" ")).trim_end().to_string());
    } else {
        let heads: Vec<String> = cells.iter().take(1).map(|v| align(v, cw)).collect();
        out.push(format!("{:>sw$} | {}", dims[0], heads.join(" ")).trim_end().to_string());
    }
    out.push(format!("{}-+-{}", "-".repeat(sw), "-".repeat(body_w - 1)));
    for (ri, l) in rl.iter().enumerate() {
        for (k, v) in cells.iter().enumerate() {
            let stub = if k == 0 { l.2.clone() } else { String::new() };
            let vals: Vec<String> = (0..ncol).map(|ci| align(&grid[ri][ci].map(|i| cell_text(v, i)).unwrap_or_default(), cw)).collect();
            out.push(format!("{stub:>sw$} | {}", vals.join(" ")).trim_end().to_string());
        }
    }
    out.push(rule);
    Ok(out.join("\n"))
}
