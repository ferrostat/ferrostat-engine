//! Macro expansion (`` `local' ``, `$global`, `` `=exp' ``, `` `:fcn' ``,
//! `` `++i' ``) and extended macro functions.

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::store::Store;
use ferrostat_cmd::{format, varlist};
use ferrostat_core::missing::{missing_code, missing_name};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_expression, split_top_level_words};

/// Number as stored in a macro: full precision.
pub fn macro_number(v: f64) -> String {
    if let Some(c) = missing_code(v) {
        return missing_name(c);
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = if v.abs() >= 1e15 || v.abs() < 1e-5 {
        let e = format!("{:e}", v);
        let (m, ex) = e.split_once('e').unwrap();
        let ex: i32 = ex.parse().unwrap();
        format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
    } else {
        format!("{v}")
    };
    if let Some(r) = s.strip_prefix("0.") {
        format!(".{r}")
    } else if let Some(r) = s.strip_prefix("-0.") {
        format!("-.{r}")
    } else {
        s
    }
}

/// Evaluate an expression and render it for a macro.
pub fn eval_to_macro(ds: &DataSet, st: &Store, text: &str) -> Result<String, String> {
    let e = parse_expression(text).map_err(|e| rc_err(198, format!("{}: {}", text.trim(), e.message)))?;
    Ok(match Env::with_store(ds, st).eval(&e, 0)? {
        Val::Num(n) => macro_number(n),
        Val::Str(s) => s,
    })
}

fn get_macro(st: &Store, name: &str) -> String {
    st.local(name).cloned().unwrap_or_default()
}

/// Expand all macros in `text`.
pub fn expand(text: &str, ds: &DataSet, st: &mut Store) -> Result<String, String> {
    // `\`` and `\$` stand for a literal backtick and dollar sign (they delay
    // expansion: `local exp "\`name'"` stores the reference itself)
    const TICK: char = '\u{E000}';
    const DOLLAR: char = '\u{E001}';
    let escaped = text.contains("\\`") || text.contains("\\$");
    let text = if escaped { text.replace("\\`", &TICK.to_string()).replace("\\$", &DOLLAR.to_string()) } else { text.to_string() };
    let s = expand_inner(&text, ds, st)?;
    Ok(if escaped { s.replace(TICK, "`").replace(DOLLAR, "$") } else { s })
}

fn expand_inner(text: &str, ds: &DataSet, st: &mut Store) -> Result<String, String> {
    let mut s = expand_globals(text, st);
    // innermost-first expansion of `...'
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 10000 {
            return Err(rc_err(198, "macro expansion too deep"));
        }
        let chars: Vec<char> = s.chars().collect();
        // find a closing quote ' with the nearest preceding backtick that is
        // not the start of a compound double quote
        let mut found = None;
        let mut last_tick: Option<usize> = None;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '`' {
                if chars.get(i + 1) == Some(&'"') {
                    i += 2;
                    continue;
                }
                last_tick = Some(i);
            } else if c == '\'' {
                if i > 0 && chars[i - 1] == '"' && last_tick.map_or(true, |t| t < i - 1) && is_compound_close(&chars, i) {
                    i += 1;
                    continue;
                }
                if let Some(t) = last_tick {
                    found = Some((t, i));
                    break;
                }
            }
            i += 1;
        }
        let Some((a, b)) = found else { break };
        let inner: String = chars[a + 1..b].iter().collect();
        let value = expand_one(&inner, ds, st)?;
        let before: String = chars[..a].iter().collect();
        let after: String = chars[b + 1..].iter().collect();
        s = format!("{before}{value}{after}");
        if s.contains('$') {
            s = expand_globals(&s, st);
        }
    }
    Ok(s)
}

/// Is the `'` at position i the end of a compound quote `"...'?
/// Byte index of the `"'` closing the compound quote that opens `s`,
/// honouring nested compound quotes.
pub fn compound_end(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] == b'`' && b[i + 1] == b'"' {
            depth += 1;
            i += 2;
            continue;
        }
        if b[i] == b'"' && b[i + 1] == b'\'' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(i);
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    None
}

fn is_compound_close(chars: &[char], i: usize) -> bool {
    // count compound opens before i
    let mut depth = 0;
    let mut k = 0;
    while k + 1 < i {
        if chars[k] == '`' && chars[k + 1] == '"' {
            depth += 1;
            k += 2;
            continue;
        }
        if chars[k] == '"' && chars[k + 1] == '\'' && depth > 0 {
            depth -= 1;
            k += 2;
            continue;
        }
        k += 1;
    }
    depth > 0
}

fn expand_one(inner: &str, ds: &DataSet, st: &mut Store) -> Result<String, String> {
    if let Some(e) = inner.strip_prefix('=') {
        return eval_to_macro(ds, st, e);
    }
    if let Some(f) = inner.strip_prefix(':') {
        return macro_function(f.trim(), ds, st);
    }
    if let Some(n) = inner.strip_prefix("++") {
        let v = get_macro(st, n).parse::<f64>().unwrap_or(0.0) + 1.0;
        st.set_local(n, macro_number(v));
        return Ok(macro_number(v));
    }
    if let Some(n) = inner.strip_prefix("--") {
        let v = get_macro(st, n).parse::<f64>().unwrap_or(0.0) - 1.0;
        st.set_local(n, macro_number(v));
        return Ok(macro_number(v));
    }
    if let Some(n) = inner.strip_suffix("++") {
        let old = get_macro(st, n);
        let v = old.parse::<f64>().unwrap_or(0.0) + 1.0;
        st.set_local(n, macro_number(v));
        return Ok(old);
    }
    if let Some(n) = inner.strip_suffix("--") {
        let old = get_macro(st, n);
        let v = old.parse::<f64>().unwrap_or(0.0) - 1.0;
        st.set_local(n, macro_number(v));
        return Ok(old);
    }
    // `r(name)', `e(name)', `s(name)', `c(name)'
    if let Some((k, name)) = inner.split_once('(').and_then(|(k, r)| Some((k, r.strip_suffix(')')?))) {
        let res = match k {
            "r" => Some(&st.r),
            "e" => Some(&st.e),
            "s" => Some(&st.s),
            _ => None,
        };
        if let Some(res) = res {
            if let Some(m) = res.macros.get(name) {
                return Ok(m.clone());
            }
            if let Some(v) = res.scalars.get(name) {
                return Ok(macro_number(*v));
            }
            return Ok(if res.matrices.contains_key(name) { "matrix".into() } else { String::new() });
        }
        if k == "c" && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return eval_to_macro(ds, st, inner);
        }
    }
    Ok(get_macro(st, inner))
}

fn expand_globals(text: &str, st: &Store) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' && (i == 0 || chars[i - 1] != '\\') {
            if chars.get(i + 1) == Some(&'{') {
                // matching `}` (names may nest: `${x${i}}`)
                let mut depth = 0usize;
                let close = chars[i + 2..].iter().position(|&c| {
                    match c {
                        '{' => depth += 1,
                        '}' if depth == 0 => return true,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    false
                });
                if let Some(close) = close {
                    // globals inside the braces are expanded first: `${x$i}`
                    let raw: String = chars[i + 2..i + 2 + close].iter().collect();
                    let name = if raw.contains('$') { expand_globals(&raw, st) } else { raw };
                    out.push_str(st.globals.get(name.trim()).map(String::as_str).unwrap_or(""));
                    i += close + 3;
                    continue;
                }
            }
            let mut j = i + 1;
            if j < chars.len() && (chars[j].is_alphabetic() || chars[j] == '_') {
                while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') && j - i <= 32 {
                    j += 1;
                }
                let name: String = chars[i + 1..j].iter().collect();
                out.push_str(st.globals.get(&name).map(String::as_str).unwrap_or(""));
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Strip one level of surrounding double or compound quotes.
pub fn unquote(s: &str) -> String {
    let t = s.trim();
    if let Some(x) = t.strip_prefix("`\"").and_then(|x| x.strip_suffix("\"'")) {
        return x.to_string();
    }
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') && !t[1..t.len() - 1].contains('"') {
        return t[1..t.len() - 1].to_string();
    }
    t.to_string()
}

/// Words of a list, honouring quotes (quotes kept unless `strip`).
pub fn words(s: &str) -> Vec<String> {
    split_top_level_words(s)
}

fn list_fn(spec: &str, st: &Store) -> Result<String, String> {
    let w: Vec<&str> = spec.split_whitespace().collect();
    let lm = |n: &str| -> Vec<String> { words(&get_macro(st, n)) };
    let join = |v: Vec<String>| v.join(" ");
    match w.as_slice() {
        ["uniq", m] => {
            let mut out: Vec<String> = Vec::new();
            for x in lm(m) {
                if !out.contains(&x) {
                    out.push(x);
                }
            }
            Ok(join(out))
        }
        ["dups", m] => {
            let mut seen: Vec<String> = Vec::new();
            let mut out = Vec::new();
            for x in lm(m) {
                if seen.contains(&x) {
                    out.push(x);
                } else {
                    seen.push(x);
                }
            }
            Ok(join(out))
        }
        ["sort", m] => {
            let mut v = lm(m);
            v.sort();
            Ok(join(v))
        }
        ["retokenize", m] | ["clean", m] => Ok(join(lm(m))),
        ["sizeof", m] => Ok(lm(m).len().to_string()),
        ["posof", elem, "in", m] => {
            let e = unquote(elem);
            Ok(lm(m).iter().position(|x| unquote(x) == e).map_or(0, |p| p + 1).to_string())
        }
        [a, "|", b] => {
            let mut v = lm(a);
            for x in lm(b) {
                if !v.contains(&x) {
                    v.push(x);
                }
            }
            Ok(join(v))
        }
        [a, "&", b] => {
            let bb = lm(b);
            Ok(join(lm(a).into_iter().filter(|x| bb.contains(x)).collect()))
        }
        [a, "-", b] => {
            let bb = lm(b);
            Ok(join(lm(a).into_iter().filter(|x| !bb.contains(x)).collect()))
        }
        [a, "in", b] => {
            let bb = lm(b);
            Ok(if lm(a).iter().all(|x| bb.contains(x)) { "1" } else { "0" }.into())
        }
        [a, "==", b] => {
            let mut x = lm(a);
            let mut y = lm(b);
            x.sort();
            y.sort();
            Ok(if x == y { "1" } else { "0" }.into())
        }
        [a, "===", b] => Ok(if lm(a) == lm(b) { "1" } else { "0" }.into()),
        _ => Err(rc_err(198, format!("list {spec}: invalid"))),
    }
}

/// Extended macro functions: `local x : word count ...`.
pub fn macro_function(spec: &str, ds: &DataSet, st: &mut Store) -> Result<String, String> {
    let w: Vec<String> = spec.split_whitespace().map(String::from).collect();
    let mut lw: Vec<String> = w.iter().map(|x| x.to_lowercase()).collect();
    // expand the documented abbreviations of the function names
    // (`var l`, `val la`, `ty`, `f`, `lab`, `char`...)
    let expand = |word: &str, table: &[(&str, usize)]| -> Option<String> {
        table.iter().find(|(full, min)| word.len() >= *min && full.starts_with(word)).map(|(f, _)| f.to_string())
    };
    if let Some(first) = lw.first().cloned() {
        if let Some(full) = expand(&first, &[("type", 2), ("format", 1), ("value", 3), ("variable", 3), ("label", 2), ("characteristic", 4)]) {
            lw[0] = if full == "characteristic" { "char".into() } else { full };
        }
        if matches!(lw[0].as_str(), "value" | "variable" | "data") && lw.len() > 1 {
            if let Some(full) = expand(&lw[1], &[("label", 1)]) {
                lw[1] = full;
            }
        }
    }
    let rest_after = |n: usize| -> String {
        let mut s = spec.trim();
        for _ in 0..n {
            s = s.trim_start();
            let end = s.find(char::is_whitespace).unwrap_or(s.len());
            s = &s[end..];
        }
        s.trim().to_string()
    };
    match lw.first().map(String::as_str) {
        Some("word") => {
            if lw.get(1).map(String::as_str) == Some("count") {
                return Ok(words(&rest_after(2)).len().to_string());
            }
            let k: usize = w.get(1).and_then(|x| x.parse().ok()).ok_or_else(|| rc_err(198, "word # of expected"))?;
            let ws = words(&rest_after(3));
            Ok(ws.get(k.wrapping_sub(1)).map(|x| unquote(x)).unwrap_or_default())
        }
        Some("piece") => {
            let k: usize = w.get(1).and_then(|x| x.parse().ok()).unwrap_or(1);
            let len: usize = w.get(2).and_then(|x| x.parse().ok()).unwrap_or(80);
            let text = unquote(&rest_after(4));
            let mut pieces = Vec::new();
            let mut cur = String::new();
            for word in text.split_whitespace() {
                if !cur.is_empty() && cur.len() + 1 + word.len() > len {
                    pieces.push(std::mem::take(&mut cur));
                }
                if !cur.is_empty() {
                    cur.push(' ');
                }
                cur.push_str(word);
            }
            if !cur.is_empty() {
                pieces.push(cur);
            }
            Ok(pieces.get(k - 1).cloned().unwrap_or_default())
        }
        Some("list") => list_fn(&rest_after(1), st),
        Some("variable") | Some("var") if lw.get(1).map(String::as_str) == Some("label") => {
            let v = varlist::resolve_name(ds, w.get(2).map(String::as_str).unwrap_or(""))?;
            Ok(ds.get_var(&v).unwrap().label.clone())
        }
        Some("value") if lw.get(1).map(String::as_str) == Some("label") => {
            let v = varlist::resolve_name(ds, w.get(2).map(String::as_str).unwrap_or(""))?;
            Ok(ds.get_var(&v).unwrap().value_label_name.clone())
        }
        Some("type") => {
            let v = varlist::resolve_name(ds, w.get(1).map(String::as_str).unwrap_or(""))?;
            Ok(ds.get_var(&v).unwrap().var_type.type_str())
        }
        Some("format") => {
            let v = varlist::resolve_name(ds, w.get(1).map(String::as_str).unwrap_or(""))?;
            Ok(ds.get_var(&v).unwrap().format.clone())
        }
        Some("label") => {
            // label lblname # [, strict] or label (varname) #
            let name = w.get(1).cloned().unwrap_or_default();
            let val = w.get(2).cloned().unwrap_or_default();
            let lname = if let Some(v) = name.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
                ds.get_var(v).map(|x| x.value_label_name.clone()).unwrap_or_default()
            } else {
                name
            };
            let v: f64 = val.parse().unwrap_or(f64::NAN);
            Ok(ds.value_labels.get(&lname).and_then(|vl| vl.lookup(v)).map(String::from).unwrap_or_else(|| if spec.contains("strict") { String::new() } else { val.clone() }))
        }
        Some("data") if lw.get(1).map(String::as_str) == Some("label") => Ok(ds.datalabel.clone()),
        Some("char") => {
            let arg = rest_after(1);
            let (var, name) = arg.trim_end_matches(']').split_once('[').unwrap_or((&arg, ""));
            Ok(ds.characteristics.get(var).and_then(|m| m.get(name)).cloned().unwrap_or_default())
        }
        Some("rownames") | Some("colnames") | Some("rowfullnames") | Some("colfullnames") | Some("roweq") | Some("coleq") => {
            let name = w.get(1).cloned().unwrap_or_default();
            let m = if let Some(k) = name.strip_prefix("e(").and_then(|x| x.strip_suffix(')')) {
                st.e.matrices.get(k).cloned()
            } else if let Some(k) = name.strip_prefix("r(").and_then(|x| x.strip_suffix(')')) {
                st.r.matrices.get(k).cloned()
            } else {
                st.matrices.get(&name).cloned()
            }
            .ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
            let names = if lw[0].starts_with("row") { m.rownames } else { m.colnames };
            let names: Vec<String> = if lw[0].ends_with("eq") {
                names.iter().map(|n| n.split_once(':').map(|x| x.0.to_string()).unwrap_or_else(|| "_".into())).collect()
            } else if lw[0].contains("full") {
                names
            } else {
                names.iter().map(|n| n.rsplit(':').next().unwrap().to_string()).collect()
            };
            Ok(names.join(" "))
        }
        Some("dir") => {
            // dir "path" files|dirs "pattern"
            let parts = words(&rest_after(1));
            let dir = unquote(parts.first().map(String::as_str).unwrap_or("."));
            let kind = parts.get(1).map(|s| s.to_lowercase()).unwrap_or_else(|| "files".into());
            let pat = unquote(parts.get(2).map(String::as_str).unwrap_or("*"));
            let mut names = Vec::new();
            if let Ok(rd) = ferrostat_core::vfs::list(&dir) {
                for (n, is_dir) in rd {
                    if (kind.starts_with("dir")) != is_dir {
                        continue;
                    }
                    if glob_match(&pat, &n) {
                        names.push(format!("\"{n}\""));
                    }
                }
            }
            names.sort();
            Ok(names.join(" "))
        }
        Some("subinstr") => {
            // subinstr local|global name "from" "to" [, all word count(local n)]
            let (body, opts) = match spec.find(',') {
                Some(i) if !spec[..i].ends_with('"') || true => (spec[..i].to_string(), spec[i + 1..].to_string()),
                _ => (spec.to_string(), String::new()),
            };
            let parts = words(&body);
            let kind = parts.get(1).cloned().unwrap_or_default();
            let name = parts.get(2).cloned().unwrap_or_default();
            let from = unquote(parts.get(3).map(String::as_str).unwrap_or(""));
            let to = unquote(parts.get(4).map(String::as_str).unwrap_or(""));
            let src = if kind.starts_with("glo") { st.globals.get(&name).cloned().unwrap_or_default() } else { get_macro(st, &name) };
            let all = opts.contains("all");
            let word = opts.contains("word");
            let mut count = 0;
            let result = if word {
                let mut out = Vec::new();
                for wd in src.split_whitespace() {
                    if wd == from && (all || count == 0) {
                        out.push(to.clone());
                        count += 1;
                    } else {
                        out.push(wd.to_string());
                    }
                }
                out.join(" ")
            } else if from.is_empty() {
                src.clone()
            } else if all {
                count = src.matches(&from).count();
                src.replace(&from, &to)
            } else {
                count = src.contains(&from) as usize;
                src.replacen(&from, &to, 1)
            };
            if let Some(i) = opts.find("count(") {
                let inner = &opts[i + 6..opts[i..].find(')').map_or(opts.len(), |j| i + j)];
                let nm = inner.split_whitespace().last().unwrap_or("");
                st.set_local(nm, count.to_string());
            }
            Ok(result)
        }
        Some("strlen") | Some("ustrlen") | Some("length") => {
            let kind = lw.get(1).cloned().unwrap_or_default();
            let name = w.get(2).cloned().unwrap_or_default();
            let v = if kind.starts_with("glo") { st.globals.get(&name).cloned().unwrap_or_default() } else { get_macro(st, &name) };
            Ok(if lw[0] == "ustrlen" { v.chars().count() } else { v.len() }.to_string())
        }
        Some("copy") => {
            let kind = lw.get(1).cloned().unwrap_or_default();
            let name = w.get(2).cloned().unwrap_or_default();
            Ok(if kind.starts_with("glo") { st.globals.get(&name).cloned().unwrap_or_default() } else { get_macro(st, &name) })
        }
        Some("display") | Some("di") => {
            let r = rest_after(1);
            let r = r.trim();
            if r.starts_with('%') {
                let (f, e) = r.split_once(char::is_whitespace).unwrap_or((r, ""));
                let e = parse_expression(e).map_err(|x| rc_err(198, x.message))?;
                let v = Env::with_store(ds, st).eval(&e, 0)?;
                return Ok(match v {
                    Val::Num(n) => format::fmt(n, f).trim().to_string(),
                    Val::Str(s) => format::fmt_str(&s, f),
                });
            }
            if r.starts_with('"') {
                return Ok(unquote(r));
            }
            eval_to_macro(ds, st, r)
        }
        Some("all") => {
            let kind = lw.get(1).cloned().unwrap_or_default();
            Ok(match kind.as_str() {
                "globals" => st.globals.keys().cloned().collect::<Vec<_>>().join(" "),
                "scalars" => st.scalars.keys().cloned().collect::<Vec<_>>().join(" "),
                "matrices" => st.matrices.keys().cloned().collect::<Vec<_>>().join(" "),
                "locals" => st.frames.last().map(|f| f.keys().cloned().collect::<Vec<_>>().join(" ")).unwrap_or_default(),
                "numeric" | "string" => {
                    let want_str = kind == "string";
                    ds.vars.iter().filter(|v| v.var_type.is_string() == want_str).map(|v| v.name.clone()).collect::<Vec<_>>().join(" ")
                }
                _ => String::new(),
            })
        }
        Some("permname") | Some("tempvar") => Ok(w.get(1).cloned().unwrap_or_default()),
        Some("sysdir") => {
            let code = w.get(1).map(|c| c.to_lowercase()).unwrap_or_default();
            let env = ferrostat_cmd::eval::Env::with_store(ds, st);
            Ok(match ferrostat_cmd::eval::c_value(&env, &format!("sysdir_{code}")) {
                ferrostat_cmd::eval::Val::Str(s) => s,
                _ => w.get(1).cloned().unwrap_or_default(),
            })
        }
        Some("properties") => Ok(crate::progtools::properties(st, w.get(1).map(String::as_str).unwrap_or(""))),
        Some("environment") | Some("env") => Ok(std::env::var(w.get(1).cloned().unwrap_or_default()).unwrap_or_default()),
        Some("results") => Ok(String::new()),
        Some("r(scalars)") | Some("r(macros)") | Some("e(scalars)") | Some("e(macros)") | Some("e(matrices)") | Some("r(matrices)") => {
            let res = if lw[0].starts_with('r') { &st.r } else { &st.e };
            Ok(if lw[0].contains("scalars") {
                res.scalars.keys().cloned().collect::<Vec<_>>().join(" ")
            } else if lw[0].contains("macros") {
                res.macros.keys().filter(|k| !k.starts_with('_')).cloned().collect::<Vec<_>>().join(" ")
            } else {
                res.matrices.keys().cloned().collect::<Vec<_>>().join(" ")
            })
        }
        _ => Err(rc_err(198, format!("{spec}: invalid macro extended function"))),
    }
}

pub fn glob_match(pat: &str, name: &str) -> bool {
    fn rec(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') => (0..=t.len()).any(|k| rec(&p[1..], &t[k..])),
            Some('?') => !t.is_empty() && rec(&p[1..], &t[1..]),
            Some(c) => t.first() == Some(c) && rec(&p[1..], &t[1..]),
        }
    }
    rec(&pat.chars().collect::<Vec<_>>(), &name.chars().collect::<Vec<_>>())
}

/// Expand a numlist: `1/5`, `1(2)9`, `10(-1)1`, `1 2 3`, `0(.5)2`.
pub fn numlist(text: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for tok in text.replace(',', " ").split_whitespace() {
        if let Some((a, b)) = tok.split_once('/') {
            let (a, b): (f64, f64) = (a.parse().map_err(|_| rc_err(121, "invalid numlist"))?, b.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
            let step = if b >= a { 1.0 } else { -1.0 };
            let mut x = a;
            while (step > 0.0 && x <= b + 1e-9) || (step < 0.0 && x >= b - 1e-9) {
                out.push(x);
                x += step;
            }
        } else if let Some(open) = tok.find(['(', '[']) {
            let close = tok.find([')', ']']).ok_or_else(|| rc_err(121, "invalid numlist"))?;
            let a: f64 = tok[..open].parse().map_err(|_| rc_err(121, "invalid numlist"))?;
            let step: f64 = tok[open + 1..close].parse().map_err(|_| rc_err(121, "invalid numlist"))?;
            let b: f64 = tok[close + 1..].parse().map_err(|_| rc_err(121, "invalid numlist"))?;
            if step == 0.0 {
                return Err(rc_err(121, "invalid numlist"));
            }
            let n = ((b - a) / step + 1e-9).floor() as i64;
            for k in 0..=n.max(-1) {
                out.push(a + step * k as f64);
            }
        } else {
            out.push(tok.parse::<f64>().map_err(|_| rc_err(121, format!("invalid numlist: {tok}")))?);
        }
    }
    Ok(out)
}

/// One bound of a numlist range check: `>0`, `>=1`, `<100`, `<=99`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeCheck {
    pub op: &'static str,
    pub bound: f64,
}

impl RangeCheck {
    pub fn parse(w: &str) -> Option<RangeCheck> {
        let (op, rest) = [">=", "<=", ">", "<"].iter().find_map(|op| w.strip_prefix(op).map(|r| (*op, r)))?;
        Some(RangeCheck { op, bound: rest.trim().parse().ok()? })
    }

    fn ok(&self, x: f64) -> bool {
        match self.op {
            ">" => x > self.bound,
            ">=" => x >= self.bound,
            "<" => x < self.bound,
            _ => x <= self.bound,
        }
    }
}

/// Constraints on a numlist (from `syntax ... numlist(...)` or `numlist, ...`).
#[derive(Debug, Clone, Default)]
pub struct NumlistRules {
    pub min: Option<usize>,
    pub max: Option<usize>,
    pub integer: bool,
    pub ascending: bool,
    pub descending: bool,
    pub missing_ok: bool,
    pub range: Vec<RangeCheck>,
}

/// Check an expanded numlist against `rules`, with the standard numlist
/// return codes 122-127.
pub fn check_numlist(vals: &[f64], rules: &NumlistRules) -> Result<(), String> {
    if rules.min.is_some_and(|m| vals.len() < m) {
        return Err(rc_err(122, "invalid numlist has too few elements"));
    }
    if rules.max.is_some_and(|m| vals.len() > m) {
        return Err(rc_err(123, "invalid numlist has too many elements"));
    }
    let ordered = |up: bool| vals.windows(2).all(|w| if up { w[0] <= w[1] } else { w[0] >= w[1] });
    if (rules.ascending && !ordered(true)) || (rules.descending && !ordered(false)) {
        return Err(rc_err(124, "invalid numlist has elements out of order"));
    }
    if !rules.missing_ok && vals.iter().any(|v| ferrostat_core::missing::is_missing(*v)) {
        return Err(rc_err(127, "invalid numlist has missing values"));
    }
    if vals.iter().any(|&v| !ferrostat_core::missing::is_missing(v) && !rules.range.iter().all(|c| c.ok(v))) {
        return Err(rc_err(125, "invalid numlist has elements outside of allowed range"));
    }
    if rules.integer && vals.iter().any(|v| v.fract() != 0.0) {
        return Err(rc_err(126, "invalid numlist has noninteger elements"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion() {
        let ds = DataSet::new();
        let mut st = Store::new();
        st.set_local("x", "hello");
        st.set_local("i", "2");
        st.set_local("v2", "two");
        st.globals.insert("g".into(), "world".into());
        assert_eq!(expand("di \"`x' $g ${g}\"", &ds, &mut st).unwrap(), "di \"hello world world\"");
        assert_eq!(expand("`v`i''", &ds, &mut st).unwrap(), "two");
        assert_eq!(expand("`=1+2'", &ds, &mut st).unwrap(), "3");
        assert_eq!(expand("`++i'", &ds, &mut st).unwrap(), "3");
        assert_eq!(expand("`:word count a b c'", &ds, &mut st).unwrap(), "3");
        assert_eq!(expand("`\"quoted\"'", &ds, &mut st).unwrap(), "`\"quoted\"'");
        assert_eq!(macro_number(1.0 / 3.0), ".3333333333333333");
    }

    #[test]
    fn numlists() {
        assert_eq!(numlist("1/3").unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(numlist("1(2)7").unwrap(), [1.0, 3.0, 5.0, 7.0]);
        assert_eq!(numlist("10(-5)0").unwrap(), [10.0, 5.0, 0.0]);
    }
}
