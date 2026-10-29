//! `file` (text and binary file I/O), `postfile`/`post`/`postclose`/
//! `postutil`, `frame post`, and sersets (`serset`, `file sersetwrite`,
//! `file sersetread`). All file access goes through the virtual file system.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::{format, util, varlist};
use ferrostat_core::dta::DtaWriter;
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::vfs;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::{parse_expression, parse_line, split_top_level_words};
use ferrostat_parser::Command;

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;
use crate::progtools::{has_opt, opt_arg, split_opts};

// ---------------------------------------------------------------------------
// file
// ---------------------------------------------------------------------------

pub struct FileHandle {
    path: String,
    read: bool,
    write: bool,
    binary: bool,
    buf: Vec<u8>,
    pos: usize,
    hilo: bool,
}

pub type Files = BTreeMap<String, FileHandle>;

fn handle<'a>(me: &'a mut AdoInterpreter, name: &str) -> Result<&'a mut FileHandle, String> {
    me.prog.files.get_mut(name).ok_or_else(|| rc_err(111, format!("file handle {name} not found")))
}

pub fn file_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (sub, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((r, ""));
    match sub {
        "open" => file_open(me, rest),
        "close" => {
            let name = rest.trim();
            if name == "_all" {
                me.prog.files.clear();
            } else {
                me.prog.files.remove(name).ok_or_else(|| rc_err(111, format!("file handle {name} not found")))?;
            }
            Ok(String::new())
        }
        "write" => {
            let (h, items) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            let (binary, hilo, write) = {
                let f = handle(me, h)?;
                (f.binary, f.hilo, f.write)
            };
            if !write {
                return Err(rc_err(602, format!("file handle {h} not open for writing")));
            }
            let bytes = write_items(me, items, binary, hilo)?;
            write_bytes(handle(me, h)?, &bytes)?;
            Ok(String::new())
        }
        "read" => file_read(me, rest),
        "seek" => {
            let w: Vec<&str> = rest.split_whitespace().collect();
            let (h, what) = (w.first().copied().unwrap_or(""), w.get(1).copied().unwrap_or("query"));
            let f = handle(me, h)?;
            match what {
                "query" => {}
                "tof" => f.pos = 0,
                "eof" => f.pos = f.buf.len(),
                n => {
                    let p: usize = n.parse().map_err(|_| rc_err(198, format!("file seek {h} {n} invalid")))?;
                    if p > f.buf.len() && !f.write {
                        return Err(rc_err(692, "file seek beyond end of file"));
                    }
                    if p > f.buf.len() {
                        f.buf.resize(p, 0);
                    }
                    f.pos = p;
                }
            }
            let loc = f.pos as f64;
            me.engine.store.begin_r().set("loc", loc);
            Ok(String::new())
        }
        "set" => {
            let w: Vec<&str> = rest.split_whitespace().collect();
            let f = handle(me, w.first().copied().unwrap_or(""))?;
            match (w.get(1).copied(), w.get(2).copied()) {
                (Some("byteorder"), Some("hilo" | "1")) => f.hilo = true,
                (Some("byteorder"), Some("lohi" | "2")) => f.hilo = false,
                (Some("write"), _) => f.write = true,
                (Some("read"), _) => f.read = true,
                _ => return Err(rc_err(198, "file set handle byteorder hilo|lohi")),
            }
            Ok(String::new())
        }
        "query" => {
            if me.prog.files.is_empty() {
                return Ok("  (no files open)".into());
            }
            Ok(me
                .prog
                .files
                .iter()
                .map(|(n, f)| {
                    let mode = match (f.read, f.write) {
                        (true, true) => "read/write",
                        (true, false) => "read",
                        _ => "write",
                    };
                    format!("  handle {n}: {} {mode}, position {}, file {}", if f.binary { "binary" } else { "text" }, f.pos, f.path)
                })
                .collect::<Vec<_>>()
                .join("\n"))
        }
        "sersetwrite" => {
            let h = rest.trim();
            let bytes = me.prog.sersets.current_bytes()?;
            write_bytes(handle(me, h)?, &bytes)?;
            Ok(String::new())
        }
        "sersetread" => {
            let h = rest.trim();
            let f = handle(me, h)?;
            let (s, used) = Serset::from_bytes(&f.buf[f.pos..])?;
            f.pos += used;
            let id = me.prog.sersets.add(s);
            serset_results(me, id);
            Ok(String::new())
        }
        "" => Err(rc_err(198, "file subcommand required")),
        other => Err(rc_err(198, format!("file {other}: invalid subcommand"))),
    }
}

fn file_open(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (body, opts) = split_opts(rest);
    let w = split_top_level_words(&body);
    if w.len() < 3 || w[1] != "using" {
        return Err(rc_err(198, "file open handle using filename, read|write"));
    }
    let name = w[0].clone();
    let path = unquote(&w[2..].join(" "));
    if me.prog.files.contains_key(&name) {
        return Err(rc_err(110, format!("file handle {name} already exists")));
    }
    let read = has_opt(&opts, "read", 4);
    let write = has_opt(&opts, "write", 5);
    if !read && !write {
        return Err(rc_err(198, "option read or write required"));
    }
    let binary = has_opt(&opts, "binary", 6);
    let replace = has_opt(&opts, "replace", 7);
    let append = has_opt(&opts, "append", 6);
    let exists = vfs::exists(&path);
    let mut buf = Vec::new();
    let mut pos = 0;
    if read && !write {
        buf = vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    } else if write && !read {
        if exists && !replace && !append {
            return Err(rc_err(602, format!("file {path} already exists")));
        }
        if append && exists {
            buf = vfs::read(&path).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
            pos = buf.len();
        } else {
            vfs::write(&path, &[]).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
        }
    } else {
        if exists && replace {
            vfs::write(&path, &[]).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
        } else if exists {
            buf = vfs::read(&path).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
            if append {
                pos = buf.len();
            }
        } else {
            vfs::write(&path, &[]).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
        }
    }
    let hilo = cfg!(target_endian = "big");
    me.prog.files.insert(name, FileHandle { path, read, write, binary, buf, pos, hilo });
    Ok(String::new())
}

fn write_bytes(f: &mut FileHandle, bytes: &[u8]) -> Result<(), String> {
    let io = |e: std::io::Error| rc_err(693, format!("could not write to {}: {e}", f.path));
    if f.pos == f.buf.len() {
        f.buf.extend_from_slice(bytes);
        f.pos = f.buf.len();
        vfs::append(&f.path, bytes).map_err(io)?;
    } else {
        let end = f.pos + bytes.len();
        if end > f.buf.len() {
            f.buf.resize(end, 0);
        }
        f.buf[f.pos..end].copy_from_slice(bytes);
        f.pos = end;
        vfs::write(&f.path, &f.buf).map_err(io)?;
    }
    Ok(())
}

/// Next directive token: a quoted string, a parenthesised expression, or a word.
fn next_token(s: &str) -> (String, &str) {
    let s = s.trim_start();
    if s.starts_with("`\"") {
        if let Some(end) = crate::macros::compound_end(s) {
            return (s[..end + 2].to_string(), &s[end + 2..]);
        }
    }
    if let Some(r) = s.strip_prefix('"') {
        if let Some(e) = r.find('"') {
            return (s[..e + 2].to_string(), &s[e + 2..]);
        }
    }
    let mut depth = 0i32;
    let mut in_q = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_q = !in_q,
            '(' if !in_q => depth += 1,
            ')' if !in_q => {
                depth -= 1;
                if depth == 0 && s.starts_with('(') {
                    return (s[..=i].to_string(), &s[i + 1..]);
                }
            }
            c if c.is_whitespace() && depth == 0 && !in_q => return (s[..i].to_string(), &s[i..]),
            _ => {}
        }
    }
    (s.to_string(), "")
}

fn string_literal(tok: &str) -> Option<String> {
    if tok.starts_with("`\"") && tok.ends_with("\"'") && tok.len() >= 4 {
        return Some(tok[2..tok.len() - 2].to_string());
    }
    if tok.starts_with('"') && tok.ends_with('"') && tok.len() >= 2 {
        return Some(tok[1..tok.len() - 1].to_string());
    }
    None
}

fn eval_scalar(me: &AdoInterpreter, text: &str) -> Result<Val, String> {
    let e = parse_expression(text).map_err(|e| rc_err(198, format!("{text}: {}", e.message)))?;
    Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)
}

/// Binary number formats: %8z %4z %4b %2b %1b with optional s/u.
fn binary_format(f: &str) -> Option<(usize, char, bool)> {
    let t = f.strip_prefix('%')?;
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    let rest = &t[digits.len()..];
    let n: usize = digits.parse().ok()?;
    match rest {
        "z" if n == 4 || n == 8 => Some((n, 'z', true)),
        "b" | "bs" if matches!(n, 1 | 2 | 4) => Some((n, 'b', true)),
        "bu" if matches!(n, 1 | 2 | 4) => Some((n, 'b', false)),
        "s" => Some((n, 's', false)),
        _ => None,
    }
}

fn encode_number(v: f64, n: usize, kind: char, signed: bool, hilo: bool) -> Vec<u8> {
    let mut b: Vec<u8> = match (kind, n) {
        ('z', 8) => v.to_le_bytes().to_vec(),
        ('z', _) => (v as f32).to_le_bytes().to_vec(),
        (_, 1) => vec![if signed { (v as i8) as u8 } else { v as u8 }],
        (_, 2) => if signed { (v as i16).to_le_bytes().to_vec() } else { (v as u16).to_le_bytes().to_vec() },
        _ => if signed { (v as i32).to_le_bytes().to_vec() } else { (v as u32).to_le_bytes().to_vec() },
    };
    if hilo {
        b.reverse();
    }
    b
}

fn decode_number(bytes: &[u8], kind: char, signed: bool, hilo: bool) -> f64 {
    let mut b = bytes.to_vec();
    if hilo {
        b.reverse();
    }
    match (kind, b.len()) {
        ('z', 8) => f64::from_le_bytes(b.try_into().unwrap()),
        ('z', _) => f32::from_le_bytes(b.try_into().unwrap()) as f64,
        (_, 1) => if signed { b[0] as i8 as f64 } else { b[0] as f64 },
        (_, 2) => {
            let a = [b[0], b[1]];
            if signed { i16::from_le_bytes(a) as f64 } else { u16::from_le_bytes(a) as f64 }
        }
        _ => {
            let a = [b[0], b[1], b[2], b[3]];
            if signed { i32::from_le_bytes(a) as f64 } else { u32::from_le_bytes(a) as f64 }
        }
    }
}

/// The output of `file write` directives.
fn write_items(me: &mut AdoInterpreter, items: &str, binary: bool, hilo: bool) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    let mut rest = items;
    let mut fmt: Option<String> = None;
    let mut dup = 1usize;
    loop {
        let (tok, r) = next_token(rest);
        rest = r;
        if tok.is_empty() {
            break;
        }
        if let Some(text) = string_literal(&tok) {
            let bytes: Vec<u8> = match fmt.take().and_then(|f| binary_format(&f)) {
                Some((n, 's', _)) => {
                    let mut b = text.as_bytes().to_vec();
                    b.resize(n, 0);
                    b
                }
                _ => text.into_bytes(),
            };
            for _ in 0..dup {
                out.extend_from_slice(&bytes);
            }
            dup = 1;
            continue;
        }
        if tok.starts_with('_') {
            let (name, arg) = match tok.find('(') {
                Some(i) => (&tok[..i], tok[i + 1..].trim_end_matches(')').trim()),
                None => (tok.as_str(), ""),
            };
            let n = || -> Result<usize, String> {
                if arg.is_empty() { Ok(1) } else { eval_scalar(me, arg)?.num().map(|v| v.max(0.0) as usize) }
            };
            match name {
                "_n" | "_newline" => out.extend(std::iter::repeat(b'\n').take(n()?)),
                "_tab" | "_t" => out.extend(std::iter::repeat(b'\t').take(n()?)),
                "_skip" | "_s" => out.extend(std::iter::repeat(b' ').take(n()?)),
                "_char" | "_c" => out.push(n()? as u8),
                "_quote" | "_q" => out.push(b'"'),
                "_column" | "_col" => {
                    let cur = out.iter().rev().take_while(|&&b| b != b'\n').count();
                    let want = n()?;
                    if want > cur + 1 {
                        out.extend(std::iter::repeat(b' ').take(want - 1 - cur));
                    }
                }
                "_dup" => dup = n()?,
                "_asis" => {}
                "_request" | "_r" => me.engine.store.set_local(arg, String::new()),
                _ => {
                    // an expression that happens to start with an underscore
                    let v = eval_scalar(me, &tok)?;
                    push_value(&mut out, &v, fmt.take(), binary, hilo)?;
                }
            }
            continue;
        }
        if tok.starts_with('%') {
            fmt = Some(tok);
            continue;
        }
        let v = eval_scalar(me, &tok)?;
        for _ in 0..dup {
            push_value(&mut out, &v, fmt.clone(), binary, hilo)?;
        }
        fmt = None;
        dup = 1;
    }
    Ok(out)
}

fn push_value(out: &mut Vec<u8>, v: &Val, fmt: Option<String>, binary: bool, hilo: bool) -> Result<(), String> {
    if let Some(f) = &fmt {
        if let Some((n, kind, signed)) = binary_format(f) {
            if !binary && kind != 's' {
                return Err(rc_err(198, format!("{f} only allowed with binary files")));
            }
            match (v, kind) {
                (Val::Num(x), 'z' | 'b') => out.extend(encode_number(*x, n, kind, signed, hilo)),
                (Val::Str(s), 's') => {
                    let mut b = s.as_bytes().to_vec();
                    b.resize(n, 0);
                    out.extend(b);
                }
                (Val::Num(x), 's') => out.extend(format::fmt(*x, &format!("%{n}.0g")).into_bytes()),
                _ => return Err(rc_err(109, "type mismatch")),
            }
            return Ok(());
        }
    }
    let text = match (v, &fmt) {
        (Val::Num(n), Some(f)) => format::fmt(*n, f),
        (Val::Num(n), None) => format::display_number(*n),
        (Val::Str(t), Some(f)) => format::fmt_str(t, f),
        (Val::Str(t), None) => t.clone(),
    };
    out.extend(text.into_bytes());
    Ok(())
}

fn file_read(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (h, specs) = rest.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((rest, ""));
    let (binary, hilo, read) = {
        let f = handle(me, h)?;
        (f.binary, f.hilo, f.read)
    };
    if !read {
        return Err(rc_err(198, format!("file handle {h} not open for reading")));
    }
    if !binary {
        let name = specs.trim().to_string();
        let f = handle(me, h)?;
        let (line, status) = if f.pos >= f.buf.len() {
            (String::new(), -1.0)
        } else {
            let rest = &f.buf[f.pos..];
            match rest.iter().position(|&b| b == b'\n') {
                Some(nl) => {
                    let mut l = &rest[..nl];
                    if l.last() == Some(&b'\r') {
                        l = &l[..l.len() - 1];
                    }
                    let s = String::from_utf8_lossy(l).to_string();
                    f.pos += nl + 1;
                    (s, 1.0)
                }
                None => {
                    let s = String::from_utf8_lossy(rest).to_string();
                    f.pos = f.buf.len();
                    (s, 2.0)
                }
            }
        };
        if !name.is_empty() {
            me.engine.store.set_local(&name, line);
        }
        let r = me.engine.store.begin_r();
        r.set("eof", if status < 0.0 { 1.0 } else { 0.0 });
        r.set("status", status);
        return Ok(String::new());
    }
    // binary: %fmt name pairs
    let toks = split_top_level_words(specs);
    let mut eof = false;
    let mut k = 0;
    while k + 1 < toks.len() + 1 && k < toks.len() {
        let f_tok = &toks[k];
        let name = toks.get(k + 1).ok_or_else(|| rc_err(198, "file read: name expected after format"))?.clone();
        k += 2;
        let (n, kind, signed) = binary_format(f_tok).ok_or_else(|| rc_err(198, format!("{f_tok} invalid binary format")))?;
        let f = handle(me, h)?;
        if f.pos + n > f.buf.len() {
            eof = true;
            f.pos = f.buf.len();
            break;
        }
        let bytes = f.buf[f.pos..f.pos + n].to_vec();
        f.pos += n;
        if kind == 's' {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            me.engine.store.set_local(&name, String::from_utf8_lossy(&bytes[..end]).to_string());
        } else {
            let v = decode_number(&bytes, kind, signed, hilo);
            me.engine.store.scalars.insert(name, if v.is_finite() { v } else { SYSMISS });
        }
    }
    let r = me.engine.store.begin_r();
    r.set("eof", if eof { 1.0 } else { 0.0 });
    r.set("status", if eof { -1.0 } else { 0.0 });
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// postfile, post, postclose, postutil, frame post
// ---------------------------------------------------------------------------

pub struct Post {
    path: String,
    data: DataSet,
}

fn default_format(t: VarType) -> String {
    match t {
        VarType::Byte => "%8.0g".into(),
        VarType::Int => "%8.0g".into(),
        VarType::Long => "%12.0g".into(),
        VarType::Float => "%9.0g".into(),
        VarType::Double => "%10.0g".into(),
        VarType::Str(n) => format!("%{}s", n.max(9)),
        VarType::StrL => "%9s".into(),
    }
}

/// Parse `[type] name [type] name ...` (a type applies to the next name or
/// parenthesised group).
fn new_vars(default: VarType, text: &str) -> Result<DataSet, String> {
    let mut ds = DataSet::new();
    let mut ty: Option<VarType> = None;
    for w in split_top_level_words(text) {
        let lw = w.to_lowercase();
        if let Some(t) = VarType::from_type_str(&lw) {
            ty = Some(if t == VarType::StrL { VarType::Str(2045) } else { t });
            continue;
        }
        let names: Vec<String> = w.trim_start_matches('(').trim_end_matches(')').split_whitespace().map(String::from).collect();
        for n in names {
            varlist::check_name(&n)?;
            let t = ty.unwrap_or(default);
            let mut v = Variable::new(&n, t);
            v.format = default_format(t);
            ds.add_variable(v).map_err(|_| rc_err(110, format!("{n} already defined")))?;
        }
        ty = None;
    }
    if ds.n_vars() == 0 {
        return Err(rc_err(100, "varlist required"));
    }
    Ok(ds)
}

fn with_dta(p: &str) -> String {
    if std::path::Path::new(p).extension().is_some() { p.to_string() } else { format!("{p}.dta") }
}

/// `postfile postname newvarlist using filename [, every(#) replace]`
pub fn postfile(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let w = split_top_level_words(&body);
    let u = w.iter().position(|x| x == "using").ok_or_else(|| rc_err(100, "using required"))?;
    if u < 2 {
        return Err(rc_err(198, "postfile postname newvarlist using filename"));
    }
    let name = w[0].clone();
    if me.prog.posts.contains_key(&name) {
        return Err(rc_err(110, format!("postname {name} already in use")));
    }
    let path = with_dta(&unquote(&w[u + 1..].join(" ")));
    if path.is_empty() {
        return Err(rc_err(198, "filename required"));
    }
    if vfs::exists(&path) && !has_opt(&opts, "replace", 7) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let data = new_vars(VarType::Float, &w[1..u].join(" "))?;
    let _ = opt_arg(&opts, "every", 5);
    // create the file now so that errors appear early
    write_dta(&path, &data)?;
    me.prog.posts.insert(name, Post { path, data });
    Ok(String::new())
}

fn write_dta(path: &str, ds: &DataSet) -> Result<(), String> {
    let mut buf = Vec::new();
    DtaWriter::write(&mut buf, ds).map_err(|e| rc_err(693, e))?;
    vfs::write(path, &buf).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))
}

/// Split `(exp) (exp) ...` into expression texts.
fn post_items(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let (tok, r) = next_token(rest);
        rest = r;
        if tok.is_empty() {
            break;
        }
        out.push(tok);
    }
    out
}

/// Append one observation of evaluated expressions to `ds`.
fn append_row(me: &AdoInterpreter, ds: &mut DataSet, items: &[String], who: &str) -> Result<(), String> {
    let k = ds.n_vars();
    if items.len() != k {
        return Err(rc_err(198, format!("{who}: {} expressions specified for {k} variables", items.len())));
    }
    let mut vals = Vec::with_capacity(k);
    for (j, it) in items.iter().enumerate() {
        let v = eval_scalar(me, it).map_err(|e| format!("{}\n{who}: above message corresponds to expression {}, variable {}", e, j + 1, ds.vars[j].name))?;
        let var = &ds.vars[j];
        let cell = match (&v, var.var_type) {
            (Val::Str(s), VarType::Str(n)) => Value::Str(s.chars().take(n).collect()),
            (Val::Str(s), VarType::StrL) => Value::Str(s.clone()),
            (Val::Num(x), t) if t.is_numeric() => util::typed_value(t, *x),
            _ => return Err(rc_err(109, format!("type mismatch\n{who}: above message corresponds to expression {}, variable {}", j + 1, var.name))),
        };
        vals.push(cell);
    }
    for (var, v) in ds.vars.iter_mut().zip(vals) {
        var.data.push(v);
    }
    ds.n_obs += 1;
    Ok(())
}

/// `post postname (exp) (exp) ...`
pub fn post(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (name, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((r, ""));
    let mut p = me.prog.posts.remove(name).ok_or_else(|| rc_err(111, format!("postname {name} not found")))?;
    let res = append_row(me, &mut p.data, &post_items(rest), "post");
    me.prog.posts.insert(name.to_string(), p);
    res.map(|_| String::new())
}

fn close_post(me: &mut AdoInterpreter, name: &str) -> Result<(), String> {
    let p = me.prog.posts.remove(name).ok_or_else(|| rc_err(111, format!("postname {name} not found")))?;
    write_dta(&p.path, &p.data)
}

/// `postclose postname`
pub fn postclose(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let name = r.trim();
    if name.is_empty() {
        return Err(rc_err(198, "postname required"));
    }
    close_post(me, name)?;
    Ok(String::new())
}

/// `postutil dir` / `postutil clear`
pub fn postutil(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    match r.trim() {
        "dir" | "" => {
            if me.prog.posts.is_empty() {
                return Ok("  (no postfiles open)".into());
            }
            Ok(me.prog.posts.iter().map(|(n, p)| format!("  postname {n}:  {}  ({} obs. posted)", p.path, p.data.n_obs)).collect::<Vec<_>>().join("\n"))
        }
        "clear" => {
            let names: Vec<String> = me.prog.posts.keys().cloned().collect();
            for n in names {
                close_post(me, &n)?;
            }
            Ok(String::new())
        }
        other => Err(rc_err(198, format!("postutil {other}: invalid subcommand"))),
    }
}

/// `frame post framename (exp) (exp) ...`: add an observation to a frame.
pub fn frame_post(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (name, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((r, ""));
    let items = post_items(rest);
    if name == me.engine.store.cur_frame {
        let mut ds = std::mem::take(&mut me.engine.dataset);
        let res = append_row(me, &mut ds, &items, "frame post");
        me.engine.dataset = ds;
        res?;
        me.engine.store.data_changed = true;
        return Ok(String::new());
    }
    let mut f = me.engine.store.data_frames.remove(name).ok_or_else(|| rc_err(111, format!("frame {name} not found")))?;
    let res = append_row(me, &mut f.data, &items, "frame post");
    if res.is_ok() {
        f.data_changed = true;
    }
    me.engine.store.data_frames.insert(name.to_string(), f);
    res.map(|_| String::new())
}

// ---------------------------------------------------------------------------
// serset
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Serset {
    names: Vec<String>,
    formats: Vec<String>,
    cols: Vec<Vec<f64>>,
}

impl Serset {
    fn n(&self) -> usize {
        self.cols.first().map_or(0, |c| c.len())
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut b = b"FSERSET1".to_vec();
        b.extend((self.n() as u64).to_le_bytes());
        b.extend((self.names.len() as u64).to_le_bytes());
        for (n, f) in self.names.iter().zip(&self.formats) {
            for s in [n, f] {
                b.extend((s.len() as u32).to_le_bytes());
                b.extend(s.as_bytes());
            }
        }
        for c in &self.cols {
            for v in c {
                b.extend(v.to_le_bytes());
            }
        }
        b
    }

    fn from_bytes(b: &[u8]) -> Result<(Serset, usize), String> {
        let bad = || rc_err(610, "file not a serset");
        if !b.starts_with(b"FSERSET1") {
            return Err(bad());
        }
        let mut p = 8;
        let u64_at = |p: &mut usize| -> Result<u64, String> {
            let v = b.get(*p..*p + 8).ok_or_else(bad)?;
            *p += 8;
            Ok(u64::from_le_bytes(v.try_into().unwrap()))
        };
        let n = u64_at(&mut p)? as usize;
        let k = u64_at(&mut p)? as usize;
        let mut names = Vec::new();
        let mut formats = Vec::new();
        for _ in 0..k {
            for which in 0..2 {
                let l = u32::from_le_bytes(b.get(p..p + 4).ok_or_else(bad)?.try_into().unwrap()) as usize;
                p += 4;
                let s = String::from_utf8_lossy(b.get(p..p + l).ok_or_else(bad)?).to_string();
                p += l;
                if which == 0 { names.push(s) } else { formats.push(s) }
            }
        }
        let mut cols = Vec::new();
        for _ in 0..k {
            let mut c = Vec::with_capacity(n);
            for _ in 0..n {
                c.push(f64::from_le_bytes(b.get(p..p + 8).ok_or_else(bad)?.try_into().unwrap()));
                p += 8;
            }
            cols.push(c);
        }
        Ok((Serset { names, formats, cols }, p))
    }
}

#[derive(Default)]
pub struct Sersets {
    sets: BTreeMap<usize, Serset>,
    current: Option<usize>,
}

impl Sersets {
    fn add(&mut self, s: Serset) -> usize {
        let id = (0..).find(|i| !self.sets.contains_key(i)).unwrap();
        self.sets.insert(id, s);
        self.current = Some(id);
        id
    }

    fn cur(&self) -> Result<(usize, &Serset), String> {
        let id = self.current.filter(|i| self.sets.contains_key(i)).ok_or_else(|| rc_err(111, "serset not found"))?;
        Ok((id, &self.sets[&id]))
    }

    fn current_bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self.cur()?.1.to_bytes())
    }
}

fn serset_results(me: &mut AdoInterpreter, id: usize) {
    let s = &me.prog.sersets.sets[&id];
    let (n, k) = (s.n() as f64, s.names.len() as f64);
    let r = me.engine.store.begin_r();
    r.set("id", id as f64);
    r.set("N", n);
    r.set("k", k);
}

/// `serset create|use|set|sort|summarize|dir|drop|clear|id|reset_id ...`
pub fn serset(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let cut = r.find(|c: char| c.is_whitespace() || c == ',').unwrap_or(r.len());
    let (sub, rest) = (&r[..cut], r[cut..].trim());
    match sub {
        "create" => {
            let c: Command = parse_line(&format!("serset {rest}")).map_err(|e| rc_err(198, e.message))?;
            let ds = &me.engine.dataset;
            let vars = varlist::expand(ds, &c.main)?;
            if vars.is_empty() {
                return Err(rc_err(100, "varlist required"));
            }
            let mask = util::touse(ds, &me.engine.store, &c)?;
            let mut cols: Vec<Vec<f64>> = Vec::new();
            let mut formats = Vec::new();
            for v in &vars {
                let var = ds.get_var(v).unwrap();
                if var.var_type.is_string() {
                    return Err(rc_err(109, format!("{v}: string variables not allowed")));
                }
                formats.push(var.format.clone());
                cols.push((0..ds.n_obs).map(|i| var.num(i)).collect());
            }
            let nothing = c.has_opt("omitnothing", 5);
            let allmiss = c.has_opt("omitallmiss", 5);
            let dupmiss = c.has_opt("omitdupmiss", 5);
            let mut keep: Vec<usize> = Vec::new();
            for i in (0..ds.n_obs).filter(|&i| mask[i]) {
                let miss: Vec<bool> = cols.iter().map(|c| is_missing(c[i])).collect();
                let drop = if nothing || dupmiss {
                    false
                } else if allmiss {
                    miss.iter().all(|&m| m)
                } else {
                    miss.iter().any(|&m| m)
                };
                if dupmiss && miss.iter().all(|&m| m) {
                    if keep.last().map_or(false, |&p| cols.iter().all(|c| is_missing(c[p]))) {
                        continue;
                    }
                }
                if !drop {
                    keep.push(i);
                }
            }
            let mut s = Serset { names: vars.clone(), formats, cols: cols.iter().map(|c| keep.iter().map(|&i| c[i]).collect()).collect() };
            if let Some(sv) = c.opt("sort", 4) {
                sort_serset(&mut s, sv)?;
            }
            let id = me.prog.sersets.add(s);
            serset_results(me, id);
            Ok(String::new())
        }
        "set" => {
            let id: usize = rest.trim().parse().map_err(|_| rc_err(198, "serset set #"))?;
            if !me.prog.sersets.sets.contains_key(&id) {
                return Err(rc_err(111, format!("serset {id} not found")));
            }
            me.prog.sersets.current = Some(id);
            serset_results(me, id);
            Ok(String::new())
        }
        "id" => {
            let (id, _) = me.prog.sersets.cur()?;
            serset_results(me, id);
            Ok(String::new())
        }
        "reset_id" => {
            // renumber the current serset as 0 (other sersets keep their ids)
            let (id, _) = me.prog.sersets.cur()?;
            if id != 0 {
                let s = me.prog.sersets.sets.remove(&id).unwrap();
                me.prog.sersets.sets.remove(&0);
                me.prog.sersets.sets.insert(0, s);
                me.prog.sersets.current = Some(0);
            }
            serset_results(me, 0);
            Ok(String::new())
        }
        "sort" => {
            let (id, _) = me.prog.sersets.cur()?;
            let mut s = me.prog.sersets.sets[&id].clone();
            let by = if rest.is_empty() { s.names.join(" ") } else { rest.to_string() };
            sort_serset(&mut s, &by)?;
            me.prog.sersets.sets.insert(id, s);
            Ok(String::new())
        }
        "summarize" | "sum" => {
            let (_, s) = me.prog.sersets.cur()?;
            let name = rest.split(',').next().unwrap_or("").trim();
            let j = s.names.iter().position(|n| n == name).ok_or_else(|| rc_err(111, format!("{name} not found in serset")))?;
            let vals: Vec<f64> = s.cols[j].iter().copied().filter(|v| !is_missing(*v)).collect();
            let n = vals.len() as f64;
            let mean = vals.iter().sum::<f64>() / n;
            let var = if n > 1.0 { vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) } else { SYSMISS };
            let r = me.engine.store.begin_r();
            r.set("N", n);
            r.set("sum_w", n);
            r.set("sum", vals.iter().sum());
            r.set("mean", if n > 0.0 { mean } else { SYSMISS });
            r.set("Var", var);
            r.set("sd", if is_missing(var) { SYSMISS } else { var.sqrt() });
            r.set("min", vals.iter().copied().fold(f64::INFINITY, f64::min));
            r.set("max", vals.iter().copied().fold(f64::NEG_INFINITY, f64::max));
            Ok(String::new())
        }
        "use" => {
            let (body, opts) = split_opts(rest);
            if me.engine.store.data_changed && me.engine.dataset.n_vars() > 0 && !has_opt(&opts, "clear", 5) {
                return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
            }
            let (id, s) = me.prog.sersets.cur()?;
            let want: Vec<String> = if body.is_empty() { s.names.clone() } else { body.split_whitespace().map(String::from).collect() };
            let mut ds = DataSet::new();
            for w in &want {
                let j = s.names.iter().position(|n| n == w).ok_or_else(|| rc_err(111, format!("{w} not found in serset")))?;
                let mut v = Variable::new(w, VarType::Double);
                v.format = s.formats[j].clone();
                v.data = s.cols[j].iter().map(|&x| Value::Double(x)).collect();
                ds.add_variable(v)?;
            }
            ds.n_obs = s.n();
            me.engine.dataset = ds;
            me.engine.store.data_changed = true;
            serset_results(me, id);
            Ok(String::new())
        }
        "dir" | "" => {
            let mut out = Vec::new();
            for (id, s) in &me.prog.sersets.sets {
                out.push(format!("  {id:>3}.  {} observations on {} variables", s.n(), s.names.len()));
                out.push(format!("        {}", s.names.join(" ")));
            }
            Ok(out.join("\n"))
        }
        "drop" => {
            let w = rest.trim();
            if w.is_empty() || w == "_all" {
                me.prog.sersets = Default::default();
            } else {
                for v in crate::macros::numlist(w)? {
                    me.prog.sersets.sets.remove(&(v as usize));
                }
                if me.prog.sersets.current.map_or(false, |c| !me.prog.sersets.sets.contains_key(&c)) {
                    me.prog.sersets.current = None;
                }
            }
            Ok(String::new())
        }
        "clear" => {
            me.prog.sersets = Default::default();
            Ok(String::new())
        }
        other => Err(rc_err(198, format!("serset {other}: invalid subcommand"))),
    }
}

fn sort_serset(s: &mut Serset, by: &str) -> Result<(), String> {
    let keys: Vec<usize> = by.split_whitespace().map(|v| s.names.iter().position(|n| n == v).ok_or_else(|| rc_err(111, format!("{v} not found in serset")))).collect::<Result<_, _>>()?;
    let mut idx: Vec<usize> = (0..s.n()).collect();
    idx.sort_by(|&a, &b| {
        for &k in &keys {
            let o = s.cols[k][a].total_cmp(&s.cols[k][b]);
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    for c in s.cols.iter_mut() {
        *c = idx.iter().map(|&i| c[i]).collect();
    }
    Ok(())
}
