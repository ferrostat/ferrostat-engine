//! Text and binary file helpers: `outfile`, `filefilter`, `hexdump`,
//! `changeeol`, `zipfile`, `unzipfile`, `unicode`.

use ferrostat_core::missing::is_missing;
use ferrostat_core::{vfs, DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("outfile", Box::new(Outfile)),
        ("filefilter", Box::new(Filefilter)),
        ("hexdump", Box::new(Hexdump)),
        ("changeeol", Box::new(Changeeol)),
        ("zipfile", Box::new(Zipfile)),
        ("unzipfile", Box::new(Unzipfile)),
        ("unicode", Box::new(Unicode)),
    ]
}

fn read_file(path: &str) -> Result<Vec<u8>, String> {
    vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))
}

fn write_file(path: &str, data: &[u8]) -> Result<(), String> {
    vfs::write(path, data).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))
}

fn check_replace(path: &str, cmd: &Command) -> Result<(), String> {
    if vfs::exists(path) && !cmd.has_opt("replace", 1) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    Ok(())
}

// ---------------------------------------------------------------- outfile

struct Outfile;

fn shortest(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn cell(ds: &DataSet, v: &ferrostat_core::Variable, i: usize, labels: bool, quote: bool) -> String {
    match &v.data[i] {
        Value::Str(s) => {
            if quote {
                format!("\"{s}\"")
            } else {
                s.clone()
            }
        }
        Value::StrL(a, b) => {
            let s = ds.resolve_strl(*a, *b).unwrap_or("");
            if quote {
                format!("\"{s}\"")
            } else {
                s.to_string()
            }
        }
        Value::Float(f) if !v.data[i].is_missing() => {
            if labels && !v.value_label_name.is_empty() {
                if let Some(l) = lookup_label(ds, v, *f as f64) {
                    return if quote { format!("\"{l}\"") } else { l };
                }
            }
            format!("{f}")
        }
        x => {
            let n = x.num();
            if is_missing(n) {
                return ".".into();
            }
            if labels && !v.value_label_name.is_empty() {
                if let Some(l) = lookup_label(ds, v, n) {
                    return if quote { format!("\"{l}\"") } else { l };
                }
            }
            shortest(n)
        }
    }
}

fn lookup_label(ds: &DataSet, v: &ferrostat_core::Variable, x: f64) -> Option<String> {
    ds.value_labels.get(&v.value_label_name).and_then(|l| l.mapping.get(&ferrostat_core::value_label::FloatKey(x)).cloned())
}

impl CommandHandler for Outfile {
    fn name(&self) -> &'static str {
        "outfile"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let path = util::unquote(&path);
        let path = if std::path::Path::new(&path).extension().is_some() { path } else { format!("{path}.raw") };
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        check_replace(&path, cmd)?;
        let mask = util::touse(ds, st, cmd)?;
        let comma = cmd.has_opt("comma", 2);
        let wide = cmd.has_opt("wide", 1);
        let labels = !cmd.has_opt("nolabel", 3);
        let quote = !cmd.has_opt("noquote", 3);
        let mut out = String::new();
        if cmd.has_opt("dictionary", 4) {
            out.push_str("infile dictionary {\n");
            for n in &vars {
                let v = ds.get_var(n).unwrap();
                let fmt = if v.var_type.is_string() { format!("%{}s", v.var_type.storage_width().max(1)) } else { "%9.0g".to_string() };
                let ty = v.var_type.type_str();
                let label = if v.label.is_empty() { String::new() } else { format!(" \"{}\"", v.label) };
                out.push_str(&format!("\t{ty:<8} {:<12} {fmt}{label}\n", v.name));
            }
            out.push_str("}\n");
        }
        let mut count = 0;
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            count += 1;
            let cells: Vec<String> = vars.iter().map(|n| cell(ds, ds.get_var(n).unwrap(), i, labels, quote)).collect();
            if comma {
                out.push_str(&cells.join(","));
                out.push('\n');
            } else if wide {
                out.push_str(&cells.join(" "));
                out.push('\n');
            } else {
                let mut line = String::new();
                for c in cells {
                    if !line.is_empty() && line.chars().count() + c.chars().count() + 1 > 80 {
                        out.push_str(&line);
                        out.push('\n');
                        line.clear();
                    }
                    if !line.is_empty() {
                        line.push(' ');
                    }
                    line.push_str(&c);
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        write_file(&path, out.as_bytes())?;
        st.begin_r().set("N", count as f64);
        Ok(format!("file {path} saved"))
    }
}

// ---------------------------------------------------------------- filefilter

/// Interpret the escapes accepted by `filefilter`.
fn unescape(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 1 < b.len() {
            let rest = &s[i + 1..];
            let named: [(&str, &[u8]); 8] = [("Windows", b"\r\n"), ("Mac", b"\r"), ("Unix", b"\n"), ("LQ", b"`"), ("RQ", b"'"), ("BS", b"\\"), ("Q", b"\""), ("Dollar", b"$")];
            if let Some((n, v)) = named.iter().find(|(n, _)| rest.starts_with(n)) {
                out.extend_from_slice(v);
                i += 1 + n.len();
                continue;
            }
            match b[i + 1] {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\\' => out.push(b'\\'),
                c => {
                    out.push(b'\\');
                    out.push(c);
                }
            }
            i += 2;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

fn replace_bytes(data: &[u8], from: &[u8], to: &[u8]) -> (Vec<u8>, usize) {
    if from.is_empty() {
        return (data.to_vec(), 0);
    }
    let mut out = Vec::with_capacity(data.len());
    let mut n = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
            n += 1;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    (out, n)
}

struct Filefilter;

impl CommandHandler for Filefilter {
    fn name(&self) -> &'static str {
        "filefilter"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let files = super::vars::split_words(&cmd.main);
        if files.len() != 2 {
            return Err(rc_err(198, "filefilter oldfile newfile, from() to()"));
        }
        let from = unescape(&util::unquote(cmd.opt("from", 1).ok_or_else(|| rc_err(198, "option from() required"))?));
        let to = unescape(&util::unquote(cmd.opt("to", 1).ok_or_else(|| rc_err(198, "option to() required"))?));
        check_replace(&files[1], cmd)?;
        let data = read_file(&files[0])?;
        let (out, n) = replace_bytes(&data, &from, &to);
        write_file(&files[1], &out)?;
        let r = st.begin_r();
        r.set("occurrences", n as f64);
        r.set("bytes_from", data.len() as f64);
        r.set("bytes_to", out.len() as f64);
        Ok(format!("  {n} replacement{} made", if n == 1 { "" } else { "s" }))
    }
}

// ---------------------------------------------------------------- hexdump

struct Hexdump;

impl CommandHandler for Hexdump {
    fn name(&self) -> &'static str {
        "hexdump"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = util::unquote(cmd.main.trim());
        let data = read_file(&path)?;
        let from = util::opt_num(cmd, "from", 1)?.unwrap_or(0.0) as usize;
        let to = util::opt_num(cmd, "to", 1)?.map(|x| x as usize).unwrap_or(data.len()).min(data.len());
        let mut out = Vec::new();
        let (mut lf, mut cr, mut crlf, mut high, mut nul, mut ctl) = (0, 0, 0, 0, 0, 0);
        let mut i = 0;
        while i < data.len() {
            let b = data[i];
            match b {
                b'\n' => lf += 1,
                b'\r' if data.get(i + 1) == Some(&b'\n') => {
                    crlf += 1;
                    i += 1;
                    lf += 0;
                }
                b'\r' => cr += 1,
                0 => nul += 1,
                128..=255 => high += 1,
                0..=31 if b != b'\t' => ctl += 1,
                _ => {}
            }
            i += 1;
        }
        let r = st.begin_r();
        r.set("filelen", data.len() as f64);
        r.set("eol_unix", lf as f64);
        r.set("eol_mac", cr as f64);
        r.set("eol_windows", crlf as f64);
        r.set("nonascii", high as f64);
        r.set("nul", nul as f64);
        if cmd.has_opt("analyze", 1) {
            out.push(format!("  File {path}: {} bytes", data.len()));
            out.push(format!("      Unix-style line endings (\\n):     {lf}"));
            out.push(format!("      Windows-style line endings (\\r\\n): {crlf}"));
            out.push(format!("      Mac-style line endings (\\r):      {cr}"));
            out.push(format!("      Bytes above 127:                 {high}"));
            out.push(format!("      Null bytes:                      {nul}"));
            out.push(format!("      Other control characters:        {ctl}"));
            return Ok(out.join("\n"));
        }
        let mut p = from;
        while p < to {
            let end = (p + 16).min(to);
            let chunk = &data[p..end];
            let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
            let asc: String = chunk.iter().map(|&b| if (32..127).contains(&b) { b as char } else { '.' }).collect();
            out.push(format!("{p:>8}  {:<47}  {asc}", hex.join(" ")));
            p = end;
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------- changeeol

struct Changeeol;

impl CommandHandler for Changeeol {
    fn name(&self) -> &'static str {
        "changeeol"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let files = super::vars::split_words(&cmd.main);
        if files.len() != 2 {
            return Err(rc_err(198, "changeeol oldfile newfile, eol(...)"));
        }
        let eol: &[u8] = match cmd.opt("eol", 3).map(|s| s.trim().to_lowercase()).as_deref() {
            Some("windows") | Some("dos") => b"\r\n",
            Some("unix") | Some("linux") => b"\n",
            Some("mac") => b"\r",
            Some(o) => return Err(rc_err(198, format!("eol({o}) invalid"))),
            None => return Err(rc_err(198, "option eol() required")),
        };
        check_replace(&files[1], cmd)?;
        let data = read_file(&files[0])?;
        let mut out = Vec::with_capacity(data.len());
        let mut n = 0;
        let mut i = 0;
        while i < data.len() {
            match data[i] {
                b'\r' if data.get(i + 1) == Some(&b'\n') => {
                    out.extend_from_slice(eol);
                    i += 2;
                    n += 1;
                }
                b'\r' | b'\n' => {
                    out.extend_from_slice(eol);
                    i += 1;
                    n += 1;
                }
                b => {
                    out.push(b);
                    i += 1;
                }
            }
        }
        write_file(&files[1], &out)?;
        st.begin_r().set("n_lines", n as f64);
        Ok(format!("  {n} line ending{} converted", if n == 1 { "" } else { "s" }))
    }
}

// ---------------------------------------------------------------- zip

struct Zipfile;

impl CommandHandler for Zipfile {
    fn name(&self) -> &'static str {
        "zipfile"
    }
    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let target = cmd.opt("saving", 6).ok_or_else(|| rc_err(198, "option saving() required"))?;
        let (zname, replace) = match target.split_once(',') {
            Some((a, b)) => (util::unquote(a.trim()), b.trim().starts_with("replace")),
            None => (util::unquote(target.trim()), false),
        };
        let zname = if std::path::Path::new(&zname).extension().is_some() { zname } else { format!("{zname}.zip") };
        if vfs::exists(&zname) && !replace && !cmd.has_opt("replace", 7) {
            return Err(rc_err(602, format!("file {zname} already exists")));
        }
        let names = super::vars::split_words(&cmd.main);
        if names.is_empty() {
            return Err(rc_err(100, "filename required"));
        }
        let mut entries = Vec::new();
        let mut out = Vec::new();
        for n in &names {
            let data = read_file(n)?;
            out.push(format!("  file {n} added successfully"));
            entries.push((n.trim_start_matches("./").to_string(), data));
        }
        write_file(&zname, &crate::xlsx::zip(&entries))?;
        out.push(format!("  file {zname} created"));
        Ok(out.join("\n"))
    }
}

struct Unzipfile;

impl CommandHandler for Unzipfile {
    fn name(&self) -> &'static str {
        "unzipfile"
    }
    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let path = util::unquote(cmd.main.trim());
        let data = read_file(&path)?;
        let files = crate::xlsx::unzip(&data).map_err(|e| rc_err(610, e))?;
        let mut out = Vec::new();
        for (name, bytes) in files {
            if name.ends_with('/') || name.contains("..") {
                continue;
            }
            if vfs::exists(&name) && !cmd.has_opt("replace", 1) {
                return Err(rc_err(602, format!("file {name} already exists")));
            }
            if let Some((dir, _)) = name.rsplit_once('/') {
                let _ = vfs::create_dir_all(dir);
            }
            write_file(&name, &bytes)?;
            out.push(format!("  inflating: {name}"));
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------- unicode

/// Single-byte code pages understood by `unicode`.
pub(super) fn decode_single(bytes: &[u8], cp1252: bool) -> String {
    const CP1252: [char; 32] = [
        '\u{20AC}', '\u{81}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{8D}', '\u{017D}', '\u{8F}', '\u{90}', '\u{2018}', '\u{2019}',
        '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{9D}', '\u{017E}', '\u{0178}',
    ];
    bytes
        .iter()
        .map(|&b| if cp1252 && (0x80..0xA0).contains(&b) { CP1252[(b - 0x80) as usize] } else { b as char })
        .collect()
}

fn encode_single(s: &str, cp1252: bool) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for c in s.chars() {
        let cu = c as u32;
        if cu < 0x80 || (cu < 0x100 && !(cp1252 && (0x80..0xA0).contains(&cu))) {
            out.push(cu as u8);
        } else if cp1252 {
            let probe = decode_single(&(0x80u8..0xA0).collect::<Vec<_>>(), true);
            match probe.chars().position(|p| p == c) {
                Some(k) => out.push(0x80 + k as u8),
                None => return Err(rc_err(198, format!("character {c} cannot be represented in the target encoding"))),
            }
        } else {
            return Err(rc_err(198, format!("character {c} cannot be represented in the target encoding")));
        }
    }
    Ok(out)
}

fn to_utf8(bytes: &[u8], enc: &str) -> Result<String, String> {
    match enc {
        "utf8" | "utf-8" => String::from_utf8(bytes.to_vec()).map_err(|_| rc_err(198, "input is not valid UTF-8")),
        "ascii" | "us-ascii" => {
            if bytes.is_ascii() {
                Ok(String::from_utf8_lossy(bytes).to_string())
            } else {
                Err(rc_err(198, "input contains non-ASCII bytes"))
            }
        }
        "latin1" | "iso-8859-1" | "iso8859-1" | "l1" => Ok(decode_single(bytes, false)),
        "cp1252" | "windows-1252" | "latin-1" => Ok(decode_single(bytes, true)),
        "utf16" | "utf-16" | "utf16le" | "utf-16le" | "utf16be" | "utf-16be" => {
            let (be, skip) = if bytes.starts_with(&[0xFE, 0xFF]) {
                (true, 2)
            } else if bytes.starts_with(&[0xFF, 0xFE]) {
                (false, 2)
            } else {
                (enc.ends_with("be"), 0)
            };
            let units: Vec<u16> = bytes[skip..].chunks_exact(2).map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) }).collect();
            String::from_utf16(&units).map_err(|_| rc_err(198, "invalid UTF-16 input"))
        }
        o => Err(rc_err(198, format!("encoding {o} not supported"))),
    }
}

fn from_utf8(s: &str, enc: &str) -> Result<Vec<u8>, String> {
    match enc {
        "utf8" | "utf-8" => Ok(s.as_bytes().to_vec()),
        "ascii" | "us-ascii" => {
            if s.is_ascii() {
                Ok(s.as_bytes().to_vec())
            } else {
                Err(rc_err(198, "text contains non-ASCII characters"))
            }
        }
        "latin1" | "iso-8859-1" | "iso8859-1" | "l1" => encode_single(s, false),
        "cp1252" | "windows-1252" | "latin-1" => encode_single(s, true),
        "utf16le" | "utf-16le" => Ok(s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect()),
        "utf16be" | "utf-16be" => Ok(s.encode_utf16().flat_map(|u| u.to_be_bytes()).collect()),
        o => Err(rc_err(198, format!("encoding {o} not supported"))),
    }
}

struct Unicode;

impl CommandHandler for Unicode {
    fn name(&self) -> &'static str {
        "unicode"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "analyze" => {
                if rest.is_empty() {
                    st.begin_r().set("N_needs_translation", 0.0);
                    return Ok(format!("  File in memory: {} variables, {} observations: all strings are valid UTF-8 (no translation necessary)", ds.n_vars(), ds.n_obs));
                }
                let mut out = Vec::new();
                let mut bad = 0;
                for f in super::vars::split_words(rest) {
                    let data = read_file(&f)?;
                    let verdict = if data.is_ascii() {
                        "ASCII only; no translation necessary"
                    } else if std::str::from_utf8(&data).is_ok() {
                        "valid UTF-8; no translation necessary"
                    } else {
                        bad += 1;
                        "contains invalid UTF-8; translation necessary"
                    };
                    out.push(format!("  {f}: {verdict}"));
                }
                st.begin_r().set("N_needs_translation", bad as f64);
                Ok(out.join("\n"))
            }
            // translate again from the backup with another encoding / put the backup back
            "retranslate" | "restore" => {
                let src = cmd.opt("encoding", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "latin1".into());
                let mut out = Vec::new();
                for f in super::vars::split_words(rest) {
                    let bak = format!("{f}.bak");
                    let data = read_file(&bak).map_err(|_| rc_err(601, format!("{f}: no backup {bak} found; nothing to {sub}")))?;
                    if sub == "restore" {
                        write_file(&f, &data)?;
                        out.push(format!("  {f}: restored from {bak}"));
                    } else {
                        let text = to_utf8(&data, &src)?;
                        write_file(&f, text.as_bytes())?;
                        out.push(format!("  {f}: retranslated from {src} to UTF-8 using {bak}"));
                    }
                }
                Ok(out.join("\n"))
            }
            "erasebackups" => {
                if !cmd.has_opt("badidea", 7) {
                    return Err(rc_err(198, "option badidea required"));
                }
                let mut n = 0;
                if let Ok(list) = ferrostat_core::vfs::list(".") {
                    for (f, is_dir) in list {
                        if !is_dir && f.ends_with(".bak") {
                            let _ = ferrostat_core::vfs::remove(&f);
                            n += 1;
                        }
                    }
                }
                Ok(format!("  ({n} backup file{} erased)", if n == 1 { "" } else { "s" }))
            }
            "translate" => {
                let src = cmd.opt("encoding", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "latin1".into());
                let mut out = Vec::new();
                for f in super::vars::split_words(rest) {
                    let data = read_file(&f)?;
                    if std::str::from_utf8(&data).is_ok() {
                        out.push(format!("  {f}: already UTF-8; unchanged"));
                        continue;
                    }
                    let text = to_utf8(&data, &src)?;
                    write_file(&format!("{f}.bak"), &data)?;
                    write_file(&f, text.as_bytes())?;
                    out.push(format!("  {f}: translated from {src} to UTF-8 (original saved as {f}.bak)"));
                }
                Ok(out.join("\n"))
            }
            "convertfile" => {
                let files = super::vars::split_words(rest);
                if files.len() != 2 {
                    return Err(rc_err(198, "unicode convertfile oldfile newfile, dstencoding() [srcencoding()]"));
                }
                check_replace(&files[1], cmd)?;
                let src = cmd.opt("srcencoding", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "latin1".into());
                let dst = cmd.opt("dstencoding", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "utf8".into());
                let data = read_file(&files[0])?;
                let text = to_utf8(&data, &src)?;
                write_file(&files[1], &from_utf8(&text, &dst)?)?;
                Ok(format!("  file {} written ({src} -> {dst})", files[1]))
            }
            "collator" | "locale" | "encoding" => {
                let (what, _) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                if what != "list" && what != "" && !(sub == "encoding" && (what == "set" || what == "alias")) {
                    return Err(rc_err(198, format!("unicode {sub} {what}: invalid subcommand")));
                }
                if what == "set" {
                    return Ok(String::new());
                }
                if what == "alias" {
                    // canonical name of an encoding
                    let name = rest[5..].trim().to_lowercase().replace(['-', '_'], "");
                    let canon = match name.as_str() {
                        "utf8" => "utf-8",
                        "ascii" | "usascii" => "us-ascii",
                        "latin1" | "iso88591" | "l1" => "iso-8859-1",
                        "cp1252" | "windows1252" => "windows-1252",
                        "utf16le" => "utf-16le",
                        "utf16be" => "utf-16be",
                        _ => return Err(rc_err(198, format!("encoding {} not known", rest[5..].trim()))),
                    };
                    st.begin_r().set_str("encoding", canon);
                    return Ok(format!("  {} is an alias of {canon}", rest[5..].trim()));
                }
                let items: &[&str] = match sub {
                    "collator" => &["root"],
                    "locale" => &["en", "en_US", "de", "es", "fr", "it", "ja", "pt", "ru", "zh"],
                    _ => &["utf8", "ascii", "latin1", "cp1252", "utf16le", "utf16be"],
                };
                Ok(items.iter().map(|i| format!("  {i}")).collect::<Vec<_>>().join("\n"))
            }
            "" => Err(rc_err(198, "unicode: subcommand required")),
            o => Err(rc_err(198, format!("unicode {o}: invalid subcommand"))),
        }
    }
}
