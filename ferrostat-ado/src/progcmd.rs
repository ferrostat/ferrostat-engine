//! Commands implemented by the interpreter itself: macros, display, scalars,
//! programming commands, files, logs.

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{format, util, varlist};
use ferrostat_core::missing::is_missing;
use ferrostat_core::vfs;
use ferrostat_core::{Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options, split_top_level_words, Parser};

use crate::interpreter::{err_message, AdoInterpreter};
use crate::macros::{self, macro_number, unquote};

/// SSC packages whose functionality Ferrostat provides natively.
pub const NATIVE_PACKAGES: &[&str] = &["outreg2", "xtpmg", "spmap", "shp2dta", "sppack", "spmat", "xtivreg2", "ivreg2", "ranktest"];

pub fn interp_command(me: &mut AdoInterpreter, verb: &str, rest: &str, full: &str) -> Result<Option<String>, String> {
    let r = rest.trim();
    Ok(Some(match verb {
        "local" | "global" => {
            define_macro(me, verb == "global", r)?;
            String::new()
        }
        "macro" => macro_cmd(me, r)?,
        "display" => display(me, r)?,
        "scalar" => scalar(me, r)?,
        "program" => program_cmd(me, r)?,
        "syntax" => {
            syntax(me, r)?;
            String::new()
        }
        "args" => {
            let zero = me.engine.store.local("0").cloned().unwrap_or_default();
            let vals = macros::words(&zero);
            for (k, name) in r.split_whitespace().enumerate() {
                me.engine.store.set_local(name, vals.get(k).map(|v| unquote(v)).unwrap_or_default());
            }
            String::new()
        }
        "tokenize" => {
            let (text, opts) = split_opts(r);
            let text = unquote(&text);
            let parse = opts.iter().find(|(n, _)| n.starts_with("p")).and_then(|(_, a)| a.clone()).map(|a| unquote(&a));
            let toks = tokenize(&text, parse.as_deref());
            let st = &mut me.engine.store;
            for k in 1..100 {
                st.clear_local(&k.to_string());
            }
            for (k, t) in toks.iter().enumerate() {
                st.set_local(&(k + 1).to_string(), t.clone());
            }
            String::new()
        }
        "gettoken" => {
            gettoken(me, r)?;
            String::new()
        }
        "confirm" => {
            confirm(me, r)?;
            String::new()
        }
        "return" | "ereturn" | "sreturn" => returns(me, verb, r)?,
        "creturn" => "c(current_date), c(current_time), c(N), c(k), c(pi), c(level) ...".into(),
        "marksample" => {
            marksample(me, r)?;
            String::new()
        }
        "markout" => {
            let w: Vec<&str> = r.split(',').next().unwrap_or("").split_whitespace().collect();
            let touse = w.first().ok_or_else(|| rc_err(198, "markout touse varlist"))?.to_string();
            let vars = varlist::expand(&me.engine.dataset, &w[1..].join(" "))?;
            let ds = &mut me.engine.dataset;
            let mut mask: Vec<bool> = util::col(ds, &touse)?.iter().map(|v| *v != 0.0 && !is_missing(*v)).collect();
            util::markout(ds, &mut mask, &vars);
            let var = ds.get_var_mut(&touse).unwrap();
            var.data = mask.iter().map(|&m| Value::Byte(m as i8)).collect();
            var.var_type = VarType::Byte;
            String::new()
        }
        "tempvar" | "tempname" | "tempfile" => {
            for name in r.split_whitespace() {
                me.temp_counter += 1;
                let val = format!("__{:06}", me.temp_counter);
                let val = if verb == "tempfile" {
                    let p = std::path::PathBuf::from(vfs::temp_path(&format!("ferrostat_{}_{}", process_id(), val)));
                    me.tempfiles.push(p.clone());
                    p.display().to_string()
                } else {
                    if verb == "tempvar" {
                        if let Some(f) = me.tempvars.last_mut() {
                            f.push(val.clone());
                        }
                    }
                    val
                };
                me.engine.store.set_local(name, val);
            }
            String::new()
        }
        "unab" | "tsunab" | "fvunab" => {
            let colon = r.find(':').ok_or_else(|| rc_err(198, "unab lmacname : varlist"))?;
            let name = r[..colon].trim();
            let (list, _) = split_opts(&r[colon + 1..]);
            let vars = varlist::expand(&me.engine.dataset, &list)?;
            me.engine.store.set_local(name, vars.join(" "));
            String::new()
        }
        "numlist" => {
            let (text, opts) = split_opts(r);
            let vals = macros::numlist(&unquote(&text))?;
            let mut res = Results::new();
            res.set_str("numlist", vals.iter().map(|v| macro_number(*v)).collect::<Vec<_>>().join(" "));
            me.engine.store.r = res;
            let _ = opts;
            String::new()
        }
        "do" | "run" | "include" => {
            let (path, args) = path_and_rest(r);
            let out = me.run_do(&path, &args, verb == "run", verb == "include")?;
            out
        }
        "log" => log_cmd(me, r)?,
        "cmdlog" => String::new(),
        "cd" | "chdir" => {
            let p = unquote(r);
            if p.is_empty() {
                vfs::current_dir()
            } else {
                vfs::set_current_dir(&p).map_err(|_| rc_err(170, format!("unable to change to {p}")))?;
                vfs::current_dir()
            }
        }
        "pwd" => vfs::current_dir(),
        "dir" => {
            let pat = if r.is_empty() { "*".to_string() } else { unquote(r) };
            let mut names: Vec<String> = vfs::list(".").map_err(|e| rc_err(601, e))?.into_iter().map(|e| e.0).filter(|n| macros::glob_match(&pat, n)).collect();
            names.sort();
            names.join("\n")
        }
        "erase" => {
            let p = unquote(r);
            vfs::remove(&p).map_err(|_| rc_err(601, format!("file {p} not found")))?;
            String::new()
        }
        "mkdir" => {
            let p = unquote(r.split(',').next().unwrap_or(""));
            vfs::create_dir_all(&p).map_err(|e| rc_err(693, e))?;
            String::new()
        }
        "rmdir" => {
            let p = unquote(r);
            vfs::remove_dir(&p).map_err(|e| rc_err(693, e))?;
            String::new()
        }
        "type" => {
            let p = unquote(r.split(',').next().unwrap_or(""));
            vfs::read_to_string(&p).map_err(|_| rc_err(601, format!("file {p} not found")))?.trim_end().to_string()
        }
        "copy" => {
            let w = split_top_level_words(r.split(',').next().unwrap_or(""));
            if w.len() < 2 {
                return Err(rc_err(198, "copy source destination"));
            }
            let (a, b) = (unquote(&w[0]), unquote(&w[1]));
            if vfs::exists(&b) && !r.contains("replace") {
                return Err(rc_err(602, format!("file {b} already exists")));
            }
            vfs::copy(&a, &b).map_err(|_| rc_err(601, format!("file {a} not found")))?;
            String::new()
        }
        "shell" | "!" | "winexec" | "xshell" => {
            shell(r)?
        }
        "which" => {
            let name = r.split(',').next().unwrap_or("").trim();
            if me.engine.is_command(name) || is_interp_command(name) {
                format!("built-in command:  {name}")
            } else if me.engine.store.programs.contains_key(name) {
                format!("{name} (program in memory)")
            } else if NATIVE_PACKAGES.contains(&name) {
                format!("built-in command:  {name} (native implementation)")
            } else {
                return Err(rc_err(111, format!("command {name} not found as either built-in or ado-file")));
            }
        }
        "ssc" | "net" | "ado" => {
            let w: Vec<&str> = r.split_whitespace().collect();
            match w.first().copied() {
                Some("install") => {
                    let pkg = w.get(1).copied().unwrap_or("");
                    if NATIVE_PACKAGES.contains(&pkg) || me.engine.is_command(pkg) {
                        format!("{pkg} is provided natively by Ferrostat; nothing to install")
                    } else {
                        return Err(rc_err(631, format!("package {pkg} is not available; Ferrostat does not download community packages")));
                    }
                }
                _ => String::new(),
            }
        }
        "adopath" | "discard" | "more" | "pause" | "timer" | "set_defaults" | "graph_set" => String::new(),
        "sleep" => {
            let ms: u64 = r.parse().unwrap_or(0);
            // the browser has no blocking sleep
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(ms.min(60_000)));
            #[cfg(target_arch = "wasm32")]
            let _ = ms;
            String::new()
        }
        "help" | "search" | "findit" => format!("help for {r}: see the Ferrostat documentation"),
        "about" => format!("Ferrostat {}", env!("CARGO_PKG_VERSION")),
        "query" => String::new(),
        "error" => {
            let code: i32 = r.parse().map_err(|_| rc_err(198, "error #"))?;
            return Err(rc_err(code, error_text(code)));
        }
        "putexcel" => crate::putexcel::putexcel(me, r)?,
        "outreg2" | "outreg" => crate::outreg::outreg2(me, r)?,
        "mata" | "mata:" | "python" | "python:" => return Err(rc_err(3000, "mata/python are not supported by Ferrostat")),
        _ => {
            let _ = full;
            return Ok(None);
        }
    }))
}

/// Commands handled by the interpreter itself rather than the engine.
pub const INTERP_COMMANDS: &[&str] = &[
    "local", "global", "macro", "display", "scalar", "program", "syntax", "args", "tokenize", "gettoken", "confirm", "return", "ereturn", "sreturn", "marksample", "markout", "tempvar", "tempname", "tempfile", "unab", "numlist", "do", "run", "include", "log", "cd", "pwd", "dir", "erase", "mkdir", "type", "copy", "shell", "which", "foreach", "forvalues", "while", "if", "capture", "quietly", "noisily", "by", "bysort", "putexcel", "error", "exit",
];

fn is_interp_command(n: &str) -> bool {
    INTERP_COMMANDS.contains(&n)
}

fn error_text(code: i32) -> &'static str {
    match code {
        1 => "--Break--",
        4 => "no; dataset in memory has changed since last saved",
        9 => "assertion is false",
        100 => "varlist required",
        109 => "type mismatch",
        110 => "already defined",
        111 => "not found",
        198 => "invalid syntax",
        199 => "unrecognized command",
        459 => "something that should be true of your data is not",
        601 => "file not found",
        602 => "file already exists",
        2000 => "no observations",
        _ => "",
    }
}

/// Split `text, options` into text and parsed options.
fn split_opts(r: &str) -> (String, Vec<(String, Option<String>)>) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

fn path_and_rest(r: &str) -> (String, String) {
    let w = split_top_level_words(r.split(',').next().unwrap_or(""));
    let path = w.first().map(|x| unquote(x)).unwrap_or_default();
    let args = w[1.min(w.len())..].join(" ");
    (path, args)
}

fn define_macro(me: &mut AdoInterpreter, global: bool, r: &str) -> Result<(), String> {
    let r = r.trim();
    if let Some(n) = r.strip_prefix("++") {
        let st = &mut me.engine.store;
        let v = st.local(n.trim()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) + 1.0;
        st.set_local(n.trim(), macro_number(v));
        return Ok(());
    }
    if let Some(n) = r.strip_prefix("--") {
        let st = &mut me.engine.store;
        let v = st.local(n.trim()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) - 1.0;
        st.set_local(n.trim(), macro_number(v));
        return Ok(());
    }
    let end = r.find(|c: char| c.is_whitespace() || c == '=' || c == ':').unwrap_or(r.len());
    let name = r[..end].to_string();
    if name.is_empty() {
        return Err(rc_err(198, "macro name required"));
    }
    let rest = r[end..].trim_start();
    let value = if let Some(e) = rest.strip_prefix('=') {
        macros::eval_to_macro(&me.engine.dataset, &me.engine.store, e)?
    } else if let Some(f) = rest.strip_prefix(':') {
        let eng = &mut me.engine;
        macros::macro_function(f.trim(), &eng.dataset, &mut eng.store)?
    } else {
        let t = rest.trim();
        // strip one level of outer quotes when they enclose the whole text
        if t.starts_with("`\"") && t.ends_with("\"'") {
            t[2..t.len() - 2].to_string()
        } else if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') && split_top_level_words(t).len() == 1 {
            t[1..t.len() - 1].to_string()
        } else {
            t.to_string()
        }
    };
    let st = &mut me.engine.store;
    if global {
        if value.is_empty() {
            st.globals.remove(&name);
        } else {
            st.globals.insert(name, value);
        }
    } else {
        st.set_local(&name, value);
    }
    Ok(())
}

fn macro_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let mut w = r.split_whitespace();
    match w.next().unwrap_or("list") {
        "drop" => {
            for n in w {
                if n == "_all" {
                    me.engine.store.globals.clear();
                } else if let Some(l) = n.strip_prefix('_') {
                    me.engine.store.clear_local(l);
                } else {
                    me.engine.store.globals.remove(n);
                }
            }
            Ok(String::new())
        }
        _ => {
            let st = &me.engine.store;
            let mut out: Vec<String> = st.globals.iter().map(|(k, v)| format!("{:<13}{}", format!("{k}:"), v)).collect();
            if let Some(f) = st.frames.last() {
                out.extend(f.iter().map(|(k, v)| format!("{:<13}{}", format!("_{k}:"), v)));
            }
            Ok(out.join("\n"))
        }
    }
}

/// Strip simple SMCL directives used in display strings.
pub fn smcl_text(s: &str) -> String {
    let mut out = String::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '{' {
            if let Some(j) = c[i..].iter().position(|&x| x == '}') {
                let tag: String = c[i + 1..i + j].iter().collect();
                let t = tag.trim();
                let (name, arg) = t.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((t, ""));
                let known = matches!(name, "txt" | "text" | "res" | "result" | "err" | "error" | "inp" | "input" | "com" | "bf" | "it" | "sf" | "ul" | "hline" | "col" | "column" | "space" | "break" | "p" | "p_end" | "c" | "char" | "dup" | "help" | "browse" | "cmd" | "hi" | "title");
                if known {
                    match name {
                        "hline" => out.push_str(&"-".repeat(arg.parse().unwrap_or(78))),
                        "space" => out.push_str(&" ".repeat(arg.parse().unwrap_or(1))),
                        "col" | "column" => {
                            let col: usize = arg.parse().unwrap_or(1);
                            let cur = out.rsplit('\n').next().map_or(0, |l| l.chars().count());
                            if col > cur + 1 {
                                out.push_str(&" ".repeat(col - 1 - cur));
                            }
                        }
                        "break" => out.push('\n'),
                        "c" | "char" => out.push(match arg {
                            "|" => '|',
                            "-" => '-',
                            "+" => '+',
                            _ => ' ',
                        }),
                        "help" | "browse" | "cmd" | "hi" | "title" | "bf" | "it" | "ul" => {
                            if let Some((_, txt)) = arg.split_once(':') {
                                out.push_str(txt);
                            } else if let Some(t) = tag.split_once(':') {
                                out.push_str(t.1);
                            }
                        }
                        _ => {
                            if let Some((_, txt)) = tag.split_once(':') {
                                out.push_str(txt);
                            }
                        }
                    }
                    i += j + 1;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// `display` with strings, expressions, formats and directives.
pub fn display(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut s = r.trim().to_string();
    let mut fmt: Option<String> = None;
    let mut dup = 1usize;
    let mut newline_end = true;
    while !s.is_empty() {
        s = s.trim_start().to_string();
        if s.is_empty() {
            break;
        }
        let lower = s.to_lowercase();
        // compound or simple string
        if s.starts_with("`\"") {
            let end = s.find("\"'").ok_or_else(|| rc_err(198, "unmatched quote"))?;
            out.push_str(&smcl_text(&s[2..end]).repeat(dup));
            dup = 1;
            s = s[end + 2..].to_string();
            continue;
        }
        if s.starts_with('"') {
            let end = s[1..].find('"').map(|i| i + 1).ok_or_else(|| rc_err(198, "unmatched quote"))?;
            out.push_str(&smcl_text(&s[1..end]).repeat(dup));
            dup = 1;
            s = s[end + 1..].to_string();
            continue;
        }
        // `as style` and the old `in smcl` / `in red|green|yellow|white|blue`
        let old_in = lower.strip_prefix("in ").map(|x| x.trim_start()).map_or(false, |x| {
            ["smcl", "red", "green", "yellow", "white", "blue"].iter().any(|c| x.starts_with(c) && !x[c.len()..].starts_with(|ch: char| ch.is_alphanumeric() || ch == '_'))
        });
        if lower.starts_with("as ") || old_in {
            let mut parts = s.splitn(3, char::is_whitespace);
            parts.next();
            parts.next();
            s = parts.next().unwrap_or("").to_string();
            continue;
        }
        if s.starts_with('_') {
            let (word, rest) = match s.find(|c: char| c.is_whitespace()) {
                Some(i) => (s[..i].to_string(), s[i..].to_string()),
                None => (s.clone(), String::new()),
            };
            let (name, arg) = match word.find('(') {
                Some(i) => (word[..i].to_string(), word[i + 1..].trim_end_matches(')').to_string()),
                None => (word.clone(), String::new()),
            };
            let n: usize = arg.trim().parse().unwrap_or(1);
            let handled = match name.as_str() {
                "_n" | "_newline" => {
                    out.push_str(&"\n".repeat(n));
                    true
                }
                "_s" | "_skip" => {
                    out.push_str(&" ".repeat(n));
                    true
                }
                "_col" | "_column" => {
                    let cur = out.rsplit('\n').next().map_or(0, |l| l.chars().count());
                    if n > cur + 1 {
                        out.push_str(&" ".repeat(n - 1 - cur));
                    }
                    true
                }
                "_c" | "_continue" => {
                    newline_end = false;
                    true
                }
                "_dup" => {
                    dup = n;
                    true
                }
                "_char" => {
                    out.push(char::from_u32(n as u32).unwrap_or(' '));
                    true
                }
                "_request" => true,
                _ => false,
            };
            if handled {
                s = rest;
                continue;
            }
        }
        if s.starts_with('%') {
            let i = s.find(char::is_whitespace).unwrap_or(s.len());
            fmt = Some(s[..i].to_string());
            s = s[i..].to_string();
            continue;
        }
        // expression: longest prefix of words that parses
        let words = split_top_level_words(&s);
        let mut matched = None;
        for k in (1..=words.len()).rev() {
            let cand = words[..k].join(" ");
            if let Ok(e) = Parser::new(&cand).parse_full_expr() {
                matched = Some((k, e));
                break;
            }
        }
        let (k, e) = matched.ok_or_else(|| rc_err(198, format!("{s} invalid name")))?;
        let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?;
        let text = match (&v, &fmt) {
            (Val::Num(n), Some(f)) => format::fmt(*n, f),
            (Val::Num(n), None) => format::display_number(*n),
            (Val::Str(t), Some(f)) => format::fmt_str(t, f),
            (Val::Str(t), None) => t.clone(),
        };
        fmt = None;
        out.push_str(&text.repeat(dup));
        dup = 1;
        s = words[k..].join(" ");
    }
    let _ = newline_end;
    Ok(out)
}

fn scalar(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let r = r.strip_prefix("define ").unwrap_or(r).trim();
    let w: Vec<&str> = r.split_whitespace().collect();
    match w.first().copied() {
        Some("list") | Some("li") | Some("l") | Some("dir") => {
            let st = &me.engine.store;
            let names: Vec<String> = if w.len() > 1 && w[1] != "_all" { w[1..].iter().map(|s| s.to_string()).collect() } else { st.scalars.keys().chain(st.str_scalars.keys()).cloned().collect() };
            Ok(names
                .iter()
                .map(|n| match (st.scalars.get(n), st.str_scalars.get(n)) {
                    (Some(v), _) => format!("{:>12} = {}", n, format::fmt_g(*v, 10)),
                    (_, Some(s)) => format!("{:>12} = {}", n, s),
                    _ => format!("{n} not found"),
                })
                .collect::<Vec<_>>()
                .join("\n"))
        }
        Some("drop") => {
            for n in &w[1..] {
                if *n == "_all" {
                    me.engine.store.scalars.clear();
                    me.engine.store.str_scalars.clear();
                } else {
                    me.engine.store.scalars.remove(*n);
                    me.engine.store.str_scalars.remove(*n);
                }
            }
            Ok(String::new())
        }
        _ => {
            let eq = r.find('=').ok_or_else(|| rc_err(198, "scalar name = exp"))?;
            let name = r[..eq].trim().to_string();
            varlist::check_name(&name)?;
            let e = parse_expression(&r[eq + 1..]).map_err(|e| rc_err(198, e.message))?;
            let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?;
            match v {
                Val::Num(n) => {
                    me.engine.store.str_scalars.remove(&name);
                    me.engine.store.scalars.insert(name, n);
                }
                Val::Str(s) => {
                    me.engine.store.scalars.remove(&name);
                    me.engine.store.str_scalars.insert(name, s);
                }
            }
            Ok(String::new())
        }
    }
}

fn program_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    match w.first().copied() {
        Some("drop") => {
            for n in &w[1..] {
                if *n == "_all" || *n == "_allado" {
                    me.engine.store.programs.clear();
                } else if me.engine.store.programs.remove(*n).is_none() {
                    return Err(rc_err(111, format!("program {n} not found")));
                }
            }
            Ok(String::new())
        }
        Some("dir") => Ok(me.engine.store.programs.keys().cloned().collect::<Vec<_>>().join("\n")),
        Some("list") | Some("l") | Some("li") => {
            let names: Vec<String> = if w.len() > 1 { w[1..].iter().map(|s| s.to_string()).collect() } else { me.engine.store.programs.keys().cloned().collect() };
            let mut out = Vec::new();
            for n in names {
                if let Some(p) = me.engine.store.programs.get(&n) {
                    out.push(format!("{n}:"));
                    for l in &p.body {
                        out.push(format!("  {l}"));
                    }
                }
            }
            Ok(out.join("\n"))
        }
        _ => Err(rc_err(198, "program definitions must end with end")),
    }
}

/// Split a string into tokens on whitespace (and `parse` characters),
/// keeping quoted strings together.
pub fn tokenize(text: &str, parse: Option<&str>) -> Vec<String> {
    let parse_chars: Vec<char> = parse.map(|p| p.chars().filter(|c| !c.is_whitespace()).collect()).unwrap_or_default();
    let mut out = Vec::new();
    for w in split_top_level_words(text) {
        if parse_chars.is_empty() || w.starts_with('"') {
            out.push(unquote(&w));
            continue;
        }
        let mut cur = String::new();
        for c in w.chars() {
            if parse_chars.contains(&c) {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

fn gettoken(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let (spec, opts) = split_opts(r);
    let colon = spec.find(':').ok_or_else(|| rc_err(198, "gettoken emname : macname"))?;
    let targets: Vec<String> = spec[..colon].split_whitespace().map(String::from).collect();
    let src = spec[colon + 1..].trim().to_string();
    let quotes = opts.iter().any(|(n, _)| n.starts_with("qu"));
    let parse: Vec<char> = opts
        .iter()
        .find(|(n, _)| n.starts_with('p'))
        .and_then(|(_, a)| a.clone())
        .map(|a| unquote(&a).chars().collect())
        .unwrap_or_else(|| vec![' ']);
    let st = &mut me.engine.store;
    let text = st.local(&src).cloned().unwrap_or_default();
    let t = text.trim_start_matches(|c: char| c == ' ');
    // token: quoted string, or up to a parse character
    let (tok, rest) = if t.starts_with('"') || t.starts_with("`\"") {
        let w = split_top_level_words(t);
        let first = w.first().cloned().unwrap_or_default();
        let rest = t[first.len()..].to_string();
        (if quotes { first } else { unquote(&first) }, rest)
    } else {
        let mut depth = 0;
        let mut end = t.len();
        for (i, c) in t.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth == 0 && parse.contains(&c) {
                end = if i == 0 && c != ' ' { c.len_utf8() } else { i };
                break;
            }
        }
        (t[..end].to_string(), t[end..].to_string())
    };
    if let Some(t1) = targets.first() {
        st.set_local(t1, tok);
    }
    if let Some(t2) = targets.get(1) {
        st.set_local(t2, rest);
    } else {
        st.set_local(&src, rest);
    }
    Ok(())
}

fn confirm(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    let ds = &me.engine.dataset;
    let st = &me.engine.store;
    let kind = w.first().copied().unwrap_or("");
    let args = |n: usize| -> String { w[n.min(w.len())..].join(" ") };
    match kind {
        "variable" | "var" | "numeric" | "string" | "str" | "byte" | "int" | "long" | "float" | "double" => {
            let (want, text) = if kind.starts_with("var") { (None, args(1)) } else { (Some(kind), args(2)) };
            let vars = varlist::expand(ds, &text).map_err(|_| rc_err(111, format!("variable {} not found", text.trim())))?;
            if let Some(t) = want {
                for v in &vars {
                    let vt = ds.get_var(v).unwrap().var_type;
                    let ok = match t {
                        "numeric" => vt.is_numeric(),
                        "string" | "str" => vt.is_string(),
                        other => vt.type_str() == other,
                    };
                    if !ok {
                        return Err(rc_err(7, format!("'{v}' found where {t} expected")));
                    }
                }
            }
            Ok(())
        }
        "new" => match w.get(1).copied() {
            Some("variable") | Some("var") => {
                for v in &w[2..] {
                    varlist::check_name(v)?;
                    if ds.var_index.contains_key(*v) {
                        return Err(rc_err(110, format!("variable {v} already defined")));
                    }
                }
                Ok(())
            }
            Some("file") => {
                let p = unquote(&args(2));
                if vfs::exists(&p) {
                    return Err(rc_err(602, format!("file {p} already exists")));
                }
                Ok(())
            }
            _ => Err(rc_err(198, "confirm new variable|file")),
        },
        "file" => {
            let p = unquote(&args(1));
            if vfs::exists(&p) { Ok(()) } else { Err(rc_err(601, format!("file {p} not found"))) }
        }
        "existence" => if args(1).trim().is_empty() { Err(rc_err(6, "")) } else { Ok(()) },
        "integer" | "number" | "n" => {
            let t = if kind == "integer" { args(2) } else { args(1) };
            match t.trim().parse::<f64>() {
                Ok(v) if kind != "integer" || v.fract() == 0.0 => Ok(()),
                _ => Err(rc_err(7, format!("'{t}' found where {} expected", if kind == "integer" { "integer" } else { "number" }))),
            }
        }
        "matrix" => if st.matrices.contains_key(args(1).trim()) { Ok(()) } else { Err(rc_err(111, format!("matrix {} not found", args(1)))) },
        "scalar" => if st.scalars.contains_key(args(1).trim()) || st.str_scalars.contains_key(args(1).trim()) { Ok(()) } else { Err(rc_err(111, format!("scalar {} not found", args(1)))) },
        "name" | "names" => {
            for n in &w[1..] {
                varlist::check_name(n)?;
            }
            Ok(())
        }
        _ => Err(rc_err(198, format!("confirm {kind}: invalid"))),
    }
}

fn returns(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    let sub = w.first().copied().unwrap_or("list");
    let prefix = verb.chars().next().unwrap();
    if sub == "list" || sub == "li" {
        let st = &me.engine.store;
        let res = match prefix {
            'r' => &st.r,
            'e' => &st.e,
            _ => &st.s,
        };
        let mut clean = res.clone();
        clean.macros.retain(|k, _| !k.starts_with('_'));
        return Ok(clean.listing(prefix));
    }
    if sub == "clear" {
        match prefix {
            'r' => {
                if let Some(Some(p)) = me.pending_r.last_mut() {
                    *p = Results::new();
                } else {
                    me.engine.store.r = Results::new();
                }
            }
            'e' => me.engine.store.e = Results::new(),
            _ => me.engine.store.s = Results::new(),
        }
        return Ok(String::new());
    }
    let rest = r[sub.len()..].trim();
    if sub == "add" {
        if let Some(Some(p)) = me.pending_r.last_mut() {
            let cur = me.engine.store.r.clone();
            for (k, v) in cur.scalars {
                p.scalars.entry(k).or_insert(v);
            }
            for (k, v) in cur.macros {
                p.macros.entry(k).or_insert(v);
            }
            for (k, v) in cur.matrices {
                p.matrices.entry(k).or_insert(v);
            }
        }
        return Ok(String::new());
    }
    if prefix == 'e' && (sub == "post" || sub == "repost") {
        let (main, opts) = split_opts(rest);
        let names: Vec<&str> = main.split_whitespace().collect();
        let b = names.first().and_then(|n| me.engine.store.matrices.get(*n).cloned());
        let v = names.get(1).and_then(|n| me.engine.store.matrices.get(*n).cloned());
        let mut e = if sub == "repost" { me.engine.store.e.clone() } else { Results::new() };
        if let Some(b) = b {
            e.set_mat("b", b);
        }
        if let Some(v) = v {
            e.set_mat("V", v);
        }
        for (n, a) in opts {
            match n.as_str() {
                "obs" => e.set("N", a.unwrap_or_default().trim().parse().unwrap_or(0.0)),
                "depname" => e.set_str("depvar", a.unwrap_or_default().trim()),
                "esample" => {
                    let var = a.unwrap_or_default().trim().to_string();
                    let c = util::col(&me.engine.dataset, &var)?;
                    e.sample = Some(c.iter().map(|x| *x != 0.0 && !is_missing(*x)).collect());
                    let _ = me.engine.dataset.drop_variable(&var);
                }
                "dof" => e.set("df_r", a.unwrap_or_default().trim().parse().unwrap_or(0.0)),
                _ => {}
            }
        }
        me.engine.store.e = e;
        return Ok(String::new());
    }
    if prefix == 'e' && sub == "display" {
        let st = &me.engine.store;
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(b.cols, b.cols));
        let k = b.cols;
        let lines = ferrostat_cmd::est::coef_table(&ferrostat_cmd::est::TableSpec {
            depvar: st.e.macros.get("depvar").map(String::as_str).unwrap_or(""),
            names: &b.colnames,
            b: &b.data,
            v: &v,
            omitted: &vec![false; k],
            is_base: &vec![false; k],
            df: st.e.get("df_r"),
            level: st.level,
            eform: None,
            coef_title: "Coefficient",
        });
        return Ok(lines.join("\n"));
    }
    // scalar / local / matrix
    let eq = rest.find(|c: char| c == '=' || c.is_whitespace()).unwrap_or(rest.len());
    let name = rest[..eq].trim().to_string();
    let value_text = rest[eq..].trim_start();
    let target_is_pending = prefix == 'r' && matches!(me.pending_r.last(), Some(Some(_)));
    match sub {
        "scalar" | "sca" => {
            let e = parse_expression(value_text.trim_start_matches('=')).map_err(|e| rc_err(198, e.message))?;
            let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?.num()?;
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set(&name, v);
            } else {
                results_mut(me, prefix).set(&name, v);
            }
        }
        "local" | "loc" => {
            let v = if let Some(e) = value_text.strip_prefix('=') { macros::eval_to_macro(&me.engine.dataset, &me.engine.store, e)? } else { unquote(value_text) };
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set_str(&name, v);
            } else {
                results_mut(me, prefix).set_str(&name, v);
            }
        }
        "matrix" | "mat" => {
            let src = value_text.trim_start_matches('=').split(',').next().unwrap_or("").trim().to_string();
            let m = ferrostat_cmd::matcmd::eval_matrix(&me.engine.dataset, &me.engine.store, &src)?;
            if !r.contains(", copy") && !r.contains(",copy") {
                me.engine.store.matrices.remove(&src);
            }
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set_mat(&name, m);
            } else {
                results_mut(me, prefix).set_mat(&name, m);
            }
        }
        _ => return Err(rc_err(198, format!("{verb} {sub}: invalid"))),
    }
    Ok(String::new())
}

fn results_mut(me: &mut AdoInterpreter, prefix: char) -> &mut Results {
    match prefix {
        'r' => &mut me.engine.store.r,
        'e' => &mut me.engine.store.e,
        _ => &mut me.engine.store.s,
    }
}

/// `syntax` for user programs.
fn syntax(me: &mut AdoInterpreter, spec: &str) -> Result<(), String> {
    let zero = me.engine.store.local("0").cloned().unwrap_or_default();
    let cmd = parse_line(&format!("__syntax {zero}")).map_err(|e| rc_err(198, e.message))?;
    // the option list starts at the first comma outside parentheses
    // (it is usually inside `[ ]`)
    let comma = {
        let mut depth = 0i32;
        let mut found = None;
        for (i, c) in spec.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    found = Some(i);
                    break;
                }
                _ => {}
            }
        }
        found
    };
    let (spec_main, spec_opts) = match comma {
        Some(i) => (spec[..i].trim_end().trim_end_matches('[').to_string(), spec[i + 1..].to_string()),
        None => (spec.to_string(), String::new()),
    };
    let spec_main = spec_main.replace("[,", " ");
    let mut main_used = false;
    for item in split_top_level_words(&spec_main) {
        let optional = item.starts_with('[');
        let inner = item.trim_start_matches('[').trim_end_matches(']').to_string();
        let lower = inner.to_lowercase();
        let st = &mut me.engine.store;
        if lower.starts_with("varlist") || lower.starts_with("newvarlist") || lower.starts_with("varname") || lower.starts_with("newvarname") {
            main_used = true;
            let text = cmd.main.trim();
            if text.is_empty() {
                if optional {
                    let default_all = !lower.contains("default=none");
                    let v = if default_all && lower.starts_with("varlist") { me.engine.dataset.varlist().join(" ") } else { String::new() };
                    me.engine.store.set_local("varlist", v);
                    continue;
                }
                return Err(rc_err(100, "varlist required"));
            }
            let name = if lower.starts_with("newvar") { "varlist" } else { "varlist" };
            let v = if lower.starts_with("new") { text.to_string() } else { varlist::expand(&me.engine.dataset, text)?.join(" ") };
            if lower.contains("numeric") && !lower.starts_with("new") {
                for x in v.split_whitespace() {
                    if let Some(var) = me.engine.dataset.get_var(util::base_name(x)) {
                        if var.var_type.is_string() {
                            return Err(rc_err(109, "type mismatch"));
                        }
                    }
                }
            }
            if let Some(i) = lower.find("max=") {
                let m: usize = lower[i + 4..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(usize::MAX);
                if v.split_whitespace().count() > m {
                    return Err(rc_err(103, "too many variables specified"));
                }
            }
            if let Some(i) = lower.find("min=") {
                let m: usize = lower[i + 4..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
                if v.split_whitespace().count() < m {
                    return Err(rc_err(102, "too few variables specified"));
                }
            }
            me.engine.store.set_local(name, v);
        } else if lower.starts_with("anything") || lower.starts_with("namelist") || lower.starts_with("name") {
            main_used = true;
            let n = if lower.starts_with("anything") { "anything" } else { "namelist" };
            if cmd.main.trim().is_empty() && !optional {
                return Err(rc_err(100, format!("{n} required")));
            }
            st.set_local(n, cmd.main.trim());
        } else if lower == "if" {
            st.set_local("if", cmd.if_raw.as_ref().map(|i| format!("if {i}")).unwrap_or_default());
        } else if lower == "in" {
            let text = cmd.in_range.as_ref().map(|r| match r {
                ferrostat_parser::InRange::Single(a) => format!("in {a}"),
                ferrostat_parser::InRange::Range(a, b) => format!("in {a}/{b}"),
                ferrostat_parser::InRange::First(b) => format!("in f/{b}"),
                ferrostat_parser::InRange::Last(a) => format!("in -{a}/l"),
                ferrostat_parser::InRange::Rel(a, b) => format!("in {a}/{b}"),
            });
            st.set_local("in", text.unwrap_or_default());
        } else if lower.starts_with("using") {
            let u = cmd.using.clone().unwrap_or_default();
            if u.is_empty() && !optional {
                return Err(rc_err(100, "using required"));
            }
            st.set_local("using", u);
        } else if lower.contains("weight") {
            if let Some(w) = &cmd.weight {
                let raw = zero.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("");
                let exp = raw.split_once('=').map(|x| x.1).unwrap_or("");
                st.set_local("weight", w.weight_type.clone());
                st.set_local("exp", format!("= {}", exp.trim()));
            } else {
                st.set_local("weight", "");
                st.set_local("exp", "");
            }
        }
    }
    if !main_used && !cmd.main.trim().is_empty() {
        return Err(rc_err(198, format!("{} invalid", cmd.main.trim())));
    }
    // options
    let given = cmd.opts.clone();
    let mut used = vec![false; given.len()];
    for item in split_top_level_words(&spec_opts.replace(['[', ']'], " ")) {
        if item == "*" {
            continue;
        }
        let (decl, arg) = match item.find('(') {
            Some(i) => (item[..i].to_string(), Some(item[i + 1..item.len() - 1].to_string())),
            None => (item.clone(), None),
        };
        let neg = decl.starts_with("no") && decl.chars().nth(2).map_or(false, |c| c.is_uppercase());
        let full = decl.to_lowercase();
        let min = decl.chars().take_while(|c| c.is_uppercase() || (neg && decl.starts_with("no") && false)).count();
        let min = if neg { 2 + decl[2..].chars().take_while(|c| c.is_uppercase()).count() } else if min == 0 { full.len() } else { min };
        let local_name = if neg { full[2..].to_string() } else { full.clone() };
        let hit = given.iter().enumerate().find(|(_, (n, _))| {
            let n = n.to_lowercase();
            n.len() >= min && full.starts_with(&n)
        });
        let st = &mut me.engine.store;
        match (hit, arg) {
            (Some((i, (_, a))), Some(argspec)) => {
                used[i] = true;
                let a = a.clone().unwrap_or_default();
                let kind = argspec.split_whitespace().next().unwrap_or("string").to_lowercase();
                let val = match kind.as_str() {
                    "varlist" | "varname" => varlist::expand(&me.engine.dataset, &a)?.join(" "),
                    "numlist" => macros::numlist(&a)?.into_iter().map(macro_number).collect::<Vec<_>>().join(" "),
                    _ => a.trim().to_string(),
                };
                let st = &mut me.engine.store;
                if argspec.to_lowercase().contains("passthru") {
                    st.set_local(&local_name, format!("{}({})", local_name, a));
                } else {
                    st.set_local(&local_name, val);
                }
            }
            (Some((i, _)), None) => {
                used[i] = true;
                st.set_local(&local_name, if neg { format!("no{local_name}") } else { local_name.clone() });
            }
            (None, Some(argspec)) => {
                let mut parts = argspec.split_whitespace();
                let kind = parts.next().unwrap_or("").to_lowercase();
                let default = parts.next().unwrap_or("").to_string();
                let v = if matches!(kind.as_str(), "integer" | "real" | "cilevel") {
                    if kind == "cilevel" { macro_number(st.level) } else { default }
                } else {
                    String::new()
                };
                st.set_local(&local_name, v);
            }
            (None, None) => st.set_local(&local_name, ""),
        }
    }
    let star = spec_opts.contains('*');
    let leftover: Vec<String> = given.iter().zip(&used).filter(|(_, u)| !**u).map(|((n, a), _)| match a {
        Some(a) => format!("{n}({a})"),
        None => n.clone(),
    }).collect();
    if !leftover.is_empty() {
        if star {
            me.engine.store.set_local("options", leftover.join(" "));
        } else {
            return Err(rc_err(198, format!("option {} not allowed", leftover[0])));
        }
    } else if star {
        me.engine.store.set_local("options", "");
    }
    Ok(())
}

fn marksample(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let (main, opts) = split_opts(r);
    let name = main.trim().to_string();
    me.temp_counter += 1;
    let var = format!("__{:06}", me.temp_counter);
    if let Some(f) = me.tempvars.last_mut() {
        f.push(var.clone());
    }
    me.engine.store.set_local(&name, var.clone());
    let st = &me.engine.store;
    let text = format!("__m {} {}", st.local("if").cloned().unwrap_or_default(), st.local("in").cloned().unwrap_or_default());
    let cmd = parse_line(&text).map_err(|e| rc_err(198, e.message))?;
    let mut mask = util::touse(&me.engine.dataset, &me.engine.store, &cmd)?;
    if !opts.iter().any(|(n, _)| n.starts_with("nov")) {
        let vars: Vec<String> = st.local("varlist").cloned().unwrap_or_default().split_whitespace().map(String::from).collect();
        util::markout(&me.engine.dataset, &mut mask, &vars);
    }
    let mut v = Variable::new(&var, VarType::Byte);
    v.data = mask.iter().map(|&m| Value::Byte(m as i8)).collect();
    me.engine.dataset.add_variable(v)?;
    Ok(())
}

fn log_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_opts(r);
    let w: Vec<&str> = main.split_whitespace().collect();
    match w.first().copied() {
        Some("using") => {
            let path = unquote(&main["using".len()..]);
            let has = |n: &str| opts.iter().any(|(o, _)| o.starts_with(n));
            let name = opts.iter().find(|(o, _)| o == "name").and_then(|(_, a)| a.clone()).unwrap_or_else(|| "<unnamed>".into());
            let smcl = !has("text") && !path.ends_with(".log") && !path.ends_with(".txt");
            let path = if std::path::Path::new(&path).extension().is_none() { format!("{path}.{}", if smcl { "smcl" } else { "log" }) } else { path };
            if vfs::exists(&path) && !has("replace") && !has("append") {
                return Err(rc_err(602, format!("file {path} already exists")));
            }
            if me.logs.iter().any(|l| l.name == name) {
                return Err(rc_err(604, "log file already open"));
            }
            let header = if smcl { format!("{{smcl}}\n{{txt}}{{sf}}{{ul off}}{{.-}}\n      name:  {{res}}{name}\n {{txt}}log:  {{res}}{path}\n{{txt}}log type:  {{res}}smcl\n opened on:  {{res}}{}\n", chrono_now()) } else { format!("      name:  {name}\n       log:  {path}\n  log type:  text\n opened on:  {}\n", chrono_now()) };
            if has("append") {
                vfs::append(&path, header.as_bytes()).map_err(|e| rc_err(603, e))?;
            } else {
                vfs::write(&path, header.as_bytes()).map_err(|e| rc_err(603, e))?;
            }
            me.logs.push(crate::interpreter::LogFile { path: path.clone().into(), smcl, on: true, name: name.clone() });
            Ok(format!("      name:  {name}\n       log:  {path}\n  log type:  {}", if smcl { "smcl" } else { "text" }))
        }
        Some("close") => {
            if me.logs.is_empty() {
                return Err(rc_err(606, "no log file open"));
            }
            let which = w.get(1).copied();
            let mut kept = Vec::new();
            let mut closed = Vec::new();
            for l in me.logs.drain(..) {
                if which.map_or(true, |n| n == "_all" || n == l.name) {
                    closed.push(l);
                } else {
                    kept.push(l);
                }
            }
            me.logs = kept;
            let mut out = Vec::new();
            for l in closed {
                {
                    let _ = vfs::append(&l.path.to_string_lossy(), format!("{}      name:  {}\n       log:  {}\n closed on:  {}\n", if l.smcl { "{txt}" } else { "" }, l.name, l.path.display(), chrono_now()).as_bytes());
                }
                out.push(format!("      name:  {}\n       log:  {}", l.name, l.path.display()));
            }
            Ok(out.join("\n"))
        }
        Some("off") => {
            for l in me.logs.iter_mut() {
                l.on = false;
            }
            Ok(String::new())
        }
        Some("on") => {
            for l in me.logs.iter_mut() {
                l.on = true;
            }
            Ok(String::new())
        }
        _ => Ok(me.logs.iter().map(|l| format!("      name:  {}\n       log:  {}", l.name, l.path.display())).collect::<Vec<_>>().join("\n")),
    }
}

fn chrono_now() -> String {
    chrono::Local::now().format("%d %b %Y, %H:%M:%S").to_string()
}

/// `input` block: `input [type] var1 [type] var2 ...` followed by data lines.
pub fn input(me: &mut AdoInterpreter, header: &str, lines: &[String]) -> Result<String, String> {
    let spec = header.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut pending_type: Option<VarType> = None;
    for w in spec.split_whitespace() {
        if let Some(t) = VarType::from_type_str(w) {
            pending_type = Some(t);
            continue;
        }
        names.push(w.to_string());
        types.push(pending_type.take());
    }
    let ds = &mut me.engine.dataset;
    let existing = names.iter().all(|n| ds.var_index.contains_key(n));
    if !existing {
        for (n, t) in names.iter().zip(&types) {
            if ds.var_index.contains_key(n) {
                return Err(rc_err(110, format!("{n} already defined")));
            }
            let ty = t.unwrap_or(me.engine.store.default_type);
            let mut v = Variable::new(n, ty);
            if let VarType::Str(w) = ty {
                v.format = format!("%{w}s");
            }
            ds.add_variable(v)?;
        }
    }
    for line in lines {
        let vals = split_top_level_words(line);
        if vals.is_empty() {
            continue;
        }
        ds.add_observation();
        let i = ds.n_obs - 1;
        for (k, n) in names.iter().enumerate() {
            let raw = vals.get(k).map(|x| unquote(x)).unwrap_or_default();
            let var = ds.get_var_mut(n).unwrap();
            if var.var_type.is_string() {
                if let VarType::Str(w) = var.var_type {
                    if raw.len() > w {
                        var.var_type = VarType::Str(raw.len());
                    }
                }
                var.data[i] = Value::Str(raw);
            } else {
                let v = if raw == "." || raw.is_empty() { ferrostat_core::SYSMISS } else { raw.parse::<f64>().map_err(|_| rc_err(198, format!("cannot be read as a number: {raw}")))? };
                var.data[i] = ferrostat_cmd::data::store_num(var.var_type, v);
            }
        }
    }
    me.engine.store.data_changed = true;
    Ok(String::new())
}

#[allow(dead_code)]
fn unused(_: &str) -> String {
    err_message("")
}

#[cfg(not(target_arch = "wasm32"))]
fn process_id() -> u32 {
    std::process::id()
}

#[cfg(target_arch = "wasm32")]
fn process_id() -> u32 {
    1
}

#[cfg(not(target_arch = "wasm32"))]
fn shell(r: &str) -> Result<String, String> {
    let out = if cfg!(windows) {
        std::process::Command::new("cmd").arg("/C").arg(r).output()
    } else {
        std::process::Command::new("sh").arg("-c").arg(r).output()
    }
    .map_err(|e| rc_err(693, e))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
}

#[cfg(target_arch = "wasm32")]
fn shell(_r: &str) -> Result<String, String> {
    Err(rc_err(199, "shell commands are not available in the browser"))
}
