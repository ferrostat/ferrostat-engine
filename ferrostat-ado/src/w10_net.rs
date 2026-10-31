//! Network commands: `net`, `ssc`, `ado`, `adoupdate`, `update`, the
//! `netio` settings, `help` for installed help files, `findit` with results
//! from the SSC archive, and autoloading of installed ado-files.
//!
//! Packages follow the published plain-text conventions: a site lists its
//! contents in a table-of-contents file (`v`, `d`, `t`, `l`, `p` lines) and
//! each package is described by a `.pkg` file (`d`, `f`, `F`, `g`, `G`, `h`
//! lines). Installed files go to the PLUS directory, in letter
//! subdirectories (`f/fre.ado`), and every installation is recorded in a
//! tracking file there so that `ado dir`, `ado uninstall` and `adoupdate`
//! work.

use std::cell::RefCell;
use std::collections::BTreeSet;

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::store::Results;
use ferrostat_core::{net, vfs};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;
use crate::progtools::{has_opt, opt_arg, split_opts};

/// The SSC archive.
pub const SSC_BASE: &str = "http://fmwww.bc.edu/repec/bocode/";
/// Latest Ferrostat release.
const RELEASES_API: &str = "https://api.github.com/repos/ferrostat/ferrostat-engine/releases/latest";
const RELEASES_PAGE: &str = "https://github.com/ferrostat/ferrostat-engine/releases/latest";
/// Download statistics of the SSC archive (for `ssc hot`).
const SSC_HOT: &str = "http://logec.repec.org/scripts/seritemstat.pf?h=repec:boc:bocode;topnum=";
/// The SSC archive's listing, newest first (for `ssc new`).
const SSC_NEW: &str = "https://ideas.repec.org/s/boc/bocode.html";
/// Installation log in each ado directory.
pub const TRK_FILE: &str = "ferrostat.trk";
/// Commands installed with `nonative`: used instead of the native version.
const NONATIVE_FILE: &str = "ferrostat.nonative";
/// Stem of the conventional table-of-contents file name.
const TOC_STEM: [u8; 5] = [0x73, 0x74, 0x61, 0x74, 0x61];
const RULE: usize = 79;

/// Name of the table-of-contents file a package site provides.
pub fn toc_file_name() -> String {
    format!("{}.toc", String::from_utf8_lossy(&TOC_STEM))
}

/// File types `net install` copies into the ado directory; other `f` files
/// are ancillary (`net get`).
const INSTALL_EXT: &[&str] = &["ado", "hlp", "sthlp", "ihlp", "mo", "mlib", "mata", "class", "dlg", "idlg", "scheme", "style", "plugin", "py", "jar", "lst"];

fn rule() -> String {
    "-".repeat(RULE)
}

// ---------------------------------------------------------------------------
// fetching
// ---------------------------------------------------------------------------

fn is_abs_local(s: &str) -> bool {
    s.starts_with('/') || s.starts_with('\\') || s.starts_with('~') || s.as_bytes().get(1) == Some(&b':')
}

/// `rel` relative to the location `loc` (a URL or a directory).
pub fn join_loc(loc: &str, rel: &str) -> String {
    let rel = rel.trim();
    if net::is_url(rel) || is_abs_local(rel) || loc.is_empty() {
        return rel.to_string();
    }
    let (prefix, path) = if net::is_url(loc) {
        let after = loc.find("://").map(|i| i + 3).unwrap_or(0);
        let slash = loc[after..].find('/').map(|i| after + i).unwrap_or(loc.len());
        (loc[..slash].to_string(), loc[slash..].to_string())
    } else {
        (String::new(), loc.to_string())
    };
    let absolute = path.starts_with('/');
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty() && *s != ".").collect();
    for seg in rel.split(['/', '\\']) {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    let joined = parts.join("/");
    if prefix.is_empty() {
        if absolute { format!("/{joined}") } else { joined }
    } else {
        format!("{prefix}/{joined}")
    }
}

fn decode(b: Vec<u8>) -> String {
    String::from_utf8(b).unwrap_or_else(|e| e.into_bytes().iter().map(|&c| c as char).collect())
}

fn net_err(e: &std::io::Error, what: &str) -> String {
    match e.kind() {
        std::io::ErrorKind::Unsupported => rc_err(631, net::NO_NETWORK),
        std::io::ErrorKind::NotFound => rc_err(601, format!("file {what} not found")),
        _ => rc_err(631, format!("could not reach {what}\n{e}")),
    }
}

/// Read a file from a URL or a local path.
fn fetch_bytes(path: &str) -> Result<Vec<u8>, String> {
    if net::is_url(path) {
        vfs::fetch_url(path).map_err(|e| net_err(&e, path))
    } else {
        vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))
    }
}

fn fetch_text(path: &str) -> Result<String, String> {
    fetch_bytes(path).map(decode)
}

// ---------------------------------------------------------------------------
// table of contents and package files
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Toc {
    desc: Vec<String>,
    dirs: Vec<(String, String)>,
    links: Vec<(String, String, String)>,
    pkgs: Vec<(String, String)>,
}

fn split_tag(line: &str) -> Option<(char, &str)> {
    let mut ch = line.chars();
    let tag = ch.next()?;
    let rest = ch.as_str();
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    Some((tag, rest.strip_prefix(' ').unwrap_or(rest).trim_end_matches('\r')))
}

fn first_word(s: &str) -> (String, String) {
    let s = s.trim_start();
    match s.split_once(char::is_whitespace) {
        Some((a, b)) => (a.to_string(), b.trim().to_string()),
        None => (s.to_string(), String::new()),
    }
}

fn parse_toc(text: &str) -> Toc {
    let mut t = Toc::default();
    for line in text.lines() {
        let Some((tag, rest)) = split_tag(line) else { continue };
        match tag {
            'd' => t.desc.push(rest.replace("@@", "@")),
            't' => t.dirs.push(first_word(rest)),
            'l' => {
                let (name, r) = first_word(rest);
                let (url, d) = first_word(&r);
                t.links.push((name, url, d));
            }
            'p' => t.pkgs.push(first_word(rest)),
            _ => {}
        }
    }
    t
}

#[derive(Debug, Clone)]
struct PkgFile {
    tag: char,
    src: String,
    target: Option<String>,
}

#[derive(Debug, Default)]
struct Pkg {
    desc: Vec<String>,
    files: Vec<PkgFile>,
}

fn parse_pkg(text: &str) -> Pkg {
    let mut p = Pkg::default();
    for line in text.lines() {
        let Some((tag, rest)) = split_tag(line) else { continue };
        match tag {
            'd' => p.desc.push(rest.replace("@@", "@")),
            'f' | 'F' | 'h' => {
                let src = unquote(rest.trim());
                if !src.is_empty() {
                    p.files.push(PkgFile { tag, src, target: None });
                }
            }
            'g' | 'G' => {
                let w: Vec<String> = rest.split_whitespace().map(unquote).collect();
                if w.len() >= 2 {
                    p.files.push(PkgFile { tag, src: w[1].clone(), target: w.get(2).cloned() });
                }
            }
            'e' => break,
            _ => {}
        }
    }
    p
}

fn base_name(p: &str) -> String {
    p.rsplit(['/', '\\']).next().unwrap_or(p).to_string()
}

fn ext(p: &str) -> String {
    let b = base_name(p);
    b.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default()
}

/// Letter subdirectory for an installed file.
fn letter(file: &str) -> char {
    match file.chars().next() {
        Some(c) if c.is_ascii_alphabetic() => c.to_ascii_lowercase(),
        _ => '_',
    }
}

/// SSC subdirectory of a package name.
fn ssc_letter(pkg: &str) -> char {
    match pkg.chars().next() {
        Some(c) if c.is_ascii_alphabetic() => c.to_ascii_lowercase(),
        Some(c) if c.is_ascii_digit() => '0',
        _ => '_',
    }
}

impl Pkg {
    fn title(&self) -> String {
        self.desc.first().cloned().unwrap_or_default()
    }
    fn install_files(&self) -> Vec<&PkgFile> {
        self.files.iter().filter(|f| f.tag == 'F' || (f.tag == 'f' && INSTALL_EXT.contains(&ext(&f.src).as_str()))).collect()
    }
    fn ancillary(&self) -> Vec<&PkgFile> {
        self.files.iter().filter(|f| f.tag == 'f' && !INSTALL_EXT.contains(&ext(&f.src).as_str())).collect()
    }
    fn distribution_date(&self) -> Option<String> {
        dist_date(&self.desc)
    }
}

fn dist_date(desc: &[String]) -> Option<String> {
    desc.iter().find_map(|l| l.trim().strip_prefix("Distribution-Date:").map(|d| d.trim().to_string()))
}

// ---------------------------------------------------------------------------
// tracking file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct TrkEntry {
    src: String,
    name: String,
    date: String,
    id: u32,
    desc: Vec<String>,
    files: Vec<String>,
}

impl TrkEntry {
    fn pkg(&self) -> String {
        self.name.trim_end_matches(".pkg").to_string()
    }
    fn title(&self) -> String {
        self.desc.first().cloned().unwrap_or_default()
    }
}

fn with_slash(p: &str) -> String {
    if p.is_empty() || p.ends_with('/') || p.ends_with('\\') { p.to_string() } else { format!("{p}/") }
}

fn read_trk(dir: &str) -> Vec<TrkEntry> {
    let Ok(text) = vfs::read_to_string(&format!("{dir}{TRK_FILE}")) else { return Vec::new() };
    let mut out = Vec::new();
    let mut cur: Option<TrkEntry> = None;
    for line in text.lines() {
        let Some((tag, rest)) = split_tag(line) else { continue };
        match tag {
            'S' => {
                if let Some(e) = cur.take() {
                    out.push(e);
                }
                cur = Some(TrkEntry { src: rest.trim().to_string(), ..Default::default() });
            }
            'N' => cur.iter_mut().for_each(|e| e.name = rest.trim().to_string()),
            'D' => cur.iter_mut().for_each(|e| e.date = rest.trim().to_string()),
            'U' => cur.iter_mut().for_each(|e| e.id = rest.trim().parse().unwrap_or(0)),
            'd' => cur.iter_mut().for_each(|e| e.desc.push(rest.to_string())),
            'f' => cur.iter_mut().for_each(|e| e.files.push(rest.trim().to_string())),
            'e' => {
                if let Some(e) = cur.take() {
                    out.push(e);
                }
            }
            _ => {}
        }
    }
    if let Some(e) = cur.take() {
        out.push(e);
    }
    out
}

fn write_trk(dir: &str, entries: &[TrkEntry]) -> Result<(), String> {
    let next = entries.iter().map(|e| e.id).max().unwrap_or(0);
    let mut s = format!("* {next:08}\n*! package tracking file of ferrostat net/ssc; do not edit\n");
    for e in entries {
        s.push_str(&format!("S {}\nN {}\nD {}\nU {}\n", e.src, e.name, e.date, e.id));
        for d in &e.desc {
            s.push_str(&format!("d {d}\n"));
        }
        for f in &e.files {
            s.push_str(&format!("f {f}\n"));
        }
        s.push_str("e\n");
    }
    vfs::create_dir_all(dir.trim_end_matches('/')).map_err(|e| rc_err(603, format!("cannot create {dir}: {e}")))?;
    vfs::write(&format!("{dir}{TRK_FILE}"), s.as_bytes()).map_err(|e| rc_err(603, format!("cannot write {dir}{TRK_FILE}: {e}")))?;
    invalidate();
    Ok(())
}

fn read_nonative(dir: &str) -> BTreeSet<String> {
    vfs::read_to_string(&format!("{dir}{NONATIVE_FILE}")).map(|t| t.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()).unwrap_or_default()
}

fn write_nonative(dir: &str, set: &BTreeSet<String>) {
    let path = format!("{dir}{NONATIVE_FILE}");
    if set.is_empty() {
        let _ = vfs::remove(&path);
    } else {
        let _ = vfs::write(&path, set.iter().map(|s| format!("{s}\n")).collect::<String>().as_bytes());
    }
    invalidate();
}

// ---------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------

fn setting(me: &AdoInterpreter, key: &str) -> Option<String> {
    me.engine.store.settings.get(key).cloned().filter(|s| !s.is_empty())
}

/// Directory packages are installed into (`net set ado`), with a trailing
/// separator.
fn ado_dir(me: &AdoInterpreter) -> Result<String, String> {
    let code = setting(me, "net_ado").unwrap_or_else(|| "PLUS".into());
    let dir = crate::progtools::sysdir_path(me, &code).unwrap_or(code);
    if dir.is_empty() {
        return Err(rc_err(198, "no ado directory for installing packages; use sysdir set PLUS dir or net set ado dir"));
    }
    Ok(with_slash(&dir))
}

fn other_dir(me: &AdoInterpreter) -> String {
    match setting(me, "net_other") {
        Some(code) => with_slash(&crate::progtools::sysdir_path(me, &code).unwrap_or(code)),
        None => String::new(),
    }
}

fn dir_option(me: &AdoInterpreter, opts: &[(String, Option<String>)]) -> Result<String, String> {
    match opt_arg(opts, "from", 4).or_else(|| opt_arg(opts, "dir", 3)) {
        Some(d) => {
            let d = unquote(&d);
            Ok(with_slash(&crate::progtools::sysdir_path(me, &d).unwrap_or(d)))
        }
        None => ado_dir(me),
    }
}

const NET_SETTINGS: &[&str] = &["httpproxy", "httpproxyhost", "httpproxyport", "httpproxyauth", "httpproxyuser", "httpproxypw", "timeout1", "timeout2", "fredkey", "ado_native"];

/// `set` of a network setting.
pub fn is_set_net(r: &str) -> bool {
    let w = r.split_whitespace().next().unwrap_or("").to_lowercase();
    NET_SETTINGS.contains(&w.as_str())
}

/// `set httpproxy on|off`, `set httpproxyhost ...`, `set timeout1 #`, ...
pub fn set_net(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, _opts) = split_opts(r);
    let (key, val) = first_word(&body);
    let key = key.to_lowercase();
    let val = unquote(val.trim());
    match key.as_str() {
        "httpproxy" | "httpproxyauth" | "ado_native" => {
            let v = val.to_lowercase();
            if !matches!(v.as_str(), "on" | "off") {
                return Err(rc_err(198, format!("set {key} on|off")));
            }
            me.engine.store.settings.insert(key.clone(), v);
        }
        "httpproxyport" | "timeout1" | "timeout2" => {
            let n: i64 = val.parse().map_err(|_| rc_err(198, format!("set {key} #")))?;
            let (lo, hi) = if key == "httpproxyport" { (1, 65535) } else { (1, 32000) };
            if n < lo || n > hi {
                return Err(rc_err(198, format!("{key} must be between {lo} and {hi}")));
            }
            me.engine.store.settings.insert(key.clone(), n.to_string());
        }
        _ => {
            me.engine.store.settings.insert(key.clone(), val);
        }
    }
    if key != "fredkey" {
        apply_config(me);
    }
    if key == "ado_native" {
        invalidate();
    }
    Ok(String::new())
}

/// Push the `netio` settings to the HTTP layer.
fn apply_config(me: &AdoInterpreter) {
    let s = &me.engine.store.settings;
    let get = |k: &str| s.get(k).cloned().unwrap_or_default();
    let num = |k: &str, d: u64| s.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    net::configure(net::NetConfig {
        proxy_on: get("httpproxy") == "on",
        proxy_host: get("httpproxyhost"),
        proxy_port: num("httpproxyport", 80) as u16,
        proxy_auth: get("httpproxyauth") == "on",
        proxy_user: get("httpproxyuser"),
        proxy_pw: get("httpproxypw"),
        timeout1: num("timeout1", 30),
        timeout2: num("timeout2", 180),
    });
}

// ---------------------------------------------------------------------------
// autoloading
// ---------------------------------------------------------------------------

thread_local! {
    /// (key, commands to take from ado-files rather than native code)
    static OVERRIDES: RefCell<Option<(String, BTreeSet<String>)>> = const { RefCell::new(None) };
}

fn invalidate() {
    OVERRIDES.with(|o| *o.borrow_mut() = None);
}

fn overrides(me: &AdoInterpreter) -> BTreeSet<String> {
    let s = &me.engine.store.settings;
    let key = format!("{:?}|{:?}|{:?}", s.get("sysdir_plus"), s.get("net_ado"), s.get("ado_native"));
    if let Some(set) = OVERRIDES.with(|o| o.borrow().as_ref().filter(|(k, _)| *k == key).map(|(_, v)| v.clone())) {
        return set;
    }
    let mut set = BTreeSet::new();
    if let Ok(dir) = ado_dir(me) {
        set = read_nonative(&dir);
        if s.get("ado_native").map(String::as_str) == Some("off") {
            for e in read_trk(&dir) {
                for f in &e.files {
                    if ext(f) == "ado" {
                        set.insert(base_name(f).trim_end_matches(".ado").to_string());
                    }
                }
            }
        }
    }
    OVERRIDES.with(|o| *o.borrow_mut() = Some((key, set.clone())));
    set
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 32 && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !n.starts_with(|c: char| c.is_ascii_digit())
}

/// Before normal dispatch: commands installed with `nonative` (or every
/// installed command under `set ado_native off`) run from their ado-files;
/// the browser build rejects URLs with a clear message.
pub fn pre_dispatch(me: &mut AdoInterpreter, verb: &str, verb_raw: &str, rest: &str) -> Result<Option<String>, String> {
    if !net::available() && matches!(verb, "use" | "copy" | "import" | "insheet" | "infile" | "infix" | "merge" | "append" | "joinby" | "cross" | "do" | "run" | "include" | "type" | "checksum")
        && (rest.contains("http://") || rest.contains("https://"))
    {
        return Err(rc_err(631, net::NO_NETWORK));
    }
    if !valid_name(verb_raw) {
        return Ok(None);
    }
    let ov = overrides(me);
    if ov.is_empty() || !ov.contains(verb_raw) {
        return Ok(None);
    }
    if !me.engine.store.programs.contains_key(verb_raw) && !autoload(me, verb_raw)? {
        return Ok(None);
    }
    me.run_program(verb_raw, rest.trim()).map(Some)
}

/// Load `name.ado` from the ado-path; true when it defined program `name`.
pub fn autoload(me: &mut AdoInterpreter, name: &str) -> Result<bool, String> {
    if !valid_name(name) {
        return Ok(false);
    }
    let found = crate::progtools::find_on_path(me, &format!("{name}.ado"), None, true, false);
    let Some(path) = found.first().cloned() else { return Ok(false) };
    me.run_do(&path, "", true, false).map_err(|e| {
        let msg = crate::interpreter::err_message(&e);
        rc_err(crate::interpreter::rc_of(&e).max(198), format!("error loading {path}:\n{msg}"))
    })?;
    Ok(me.engine.store.programs.contains_key(name))
}

/// `egen ... = fcn(...)`: load `_gfcn.ado` when it is installed.
pub fn autoload_egen(me: &mut AdoInterpreter, rest: &str) {
    let Some((_, rhs)) = rest.split_once('=') else { return };
    let rhs = rhs.trim_start();
    let Some(open) = rhs.find('(') else { return };
    let prog = format!("_g{}", rhs[..open].trim());
    if !me.engine.store.programs.contains_key(&prog) {
        let _ = autoload(me, &prog);
    }
}

/// `which name` for an installed ado-file: its path and `*!` lines.
pub fn which_ado(me: &AdoInterpreter, name: &str) -> Option<String> {
    if !valid_name(name) {
        return None;
    }
    let found = crate::progtools::find_on_path(me, &format!("{name}.ado"), None, true, false);
    let path = found.first()?;
    let mut out = vec![path.clone()];
    if let Ok(t) = vfs::read_to_string(path) {
        out.extend(t.lines().filter(|l| l.starts_with("*!")).map(|l| l.trim_end().to_string()));
    }
    Some(out.join("\n"))
}

// ---------------------------------------------------------------------------
// help, search
// ---------------------------------------------------------------------------

/// `help`/`whelp`/`search`/`findit`.
pub fn help_search(me: &AdoInterpreter, verb: &str, r: &str) -> String {
    let (body, opts) = split_opts(r);
    let words: Vec<String> = body.split_whitespace().map(|w| unquote(w)).collect();
    if matches!(verb, "help" | "whelp") {
        if let Some(w) = words.first() {
            let name = w.to_lowercase();
            if valid_name(&name) {
                for e in ["sthlp", "hlp"] {
                    let found = crate::progtools::find_on_path(me, &format!("{name}.{e}"), None, true, false);
                    if let Some(text) = found.first().and_then(|p| vfs::read_to_string(p).ok()) {
                        return crate::w10_smcl::render_help(&text);
                    }
                }
            }
        }
        return crate::w8_prog::search(me, "help", r);
    }
    let local = crate::w8_prog::search(me, "search", r);
    let want_net = verb == "findit" || has_opt(&opts, "net", 3) || has_opt(&opts, "all", 3);
    if !want_net || words.is_empty() || !net::available() {
        return local;
    }
    let local = local.replace(" (offline; no network search)", "");
    match ssc_search(&words) {
        Ok(hits) => format!("{local}\n\n{}", format_ssc_hits(&words, &hits)),
        Err(e) => format!("{local}\n\n(SSC archive not searched: {})", crate::interpreter::err_message(&e).lines().next().unwrap_or("")),
    }
}

thread_local! {
    static TOC_CACHE: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
}

/// A table of contents of the SSC archive, cached for the session.
fn ssc_toc(sub: &str) -> Result<String, String> {
    let url = if sub.is_empty() { format!("{SSC_BASE}{}", toc_file_name()) } else { format!("{SSC_BASE}{sub}/{}", toc_file_name()) };
    if let Some(t) = TOC_CACHE.with(|c| c.borrow().iter().find(|(u, _)| *u == url).map(|(_, t)| t.clone())) {
        return Ok(t);
    }
    let t = fetch_text(&url)?;
    TOC_CACHE.with(|c| c.borrow_mut().push((url, t.clone())));
    Ok(t)
}

/// Packages of the SSC archive whose name or description contains every word.
fn ssc_search(words: &[String]) -> Result<Vec<(String, String)>, String> {
    let root = parse_toc(&ssc_toc("")?);
    let words: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let mut hits = Vec::new();
    for (d, _) in &root.dirs {
        // some letter directories have no listing
        let toc = match ssc_toc(d) {
            Ok(t) => parse_toc(&t),
            Err(e) if crate::interpreter::rc_of(&e) == 601 => continue,
            Err(e) => return Err(e),
        };
        for (name, desc) in toc.pkgs {
            let hay = format!("{} {}", name, desc).to_lowercase();
            if words.iter().all(|w| hay.contains(w.as_str())) {
                hits.push((name, desc));
            }
        }
    }
    Ok(hits)
}

fn format_ssc_hits(words: &[String], hits: &[(String, String)]) -> String {
    let mut out = vec![format!("Search of the SSC archive for \"{}\"", words.join(" ")), rule()];
    if hits.is_empty() {
        out.push("no packages found".into());
    } else {
        for (n, d) in hits {
            out.push(format!("{n:<16} {d}"));
            out.push(format!("{:16} (type ssc describe {n})", ""));
        }
        out.push(format!("({} package{} found)", hits.len(), if hits.len() == 1 { "" } else { "s" }));
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// net
// ---------------------------------------------------------------------------

/// Entry point: `ssc`, `net`, `ado`, `adoupdate`, `update`.
pub fn command(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    match verb {
        "net" => net_cmd(me, r),
        "ssc" => ssc_cmd(me, r),
        "ado" => ado_cmd(me, r),
        "adoupdate" => adoupdate(me, r),
        "update" => update(me, r),
        _ => Err(rc_err(199, format!("command {verb} is unrecognized"))),
    }
}

fn current_loc(me: &AdoInterpreter) -> Result<String, String> {
    setting(me, "net_from").ok_or_else(|| rc_err(198, "no current location; type net from <url or directory> first"))
}

fn show_toc(loc: &str, toc: &Toc) -> String {
    let mut out = vec![rule(), loc.to_string()];
    out.extend(toc.desc.iter().cloned());
    out.push(rule());
    if !toc.dirs.is_empty() {
        out.push(String::new());
        out.push("DIRECTORIES you could -net cd- to:".into());
        for (d, desc) in &toc.dirs {
            out.push(format!("    {d:<14} {desc}").trim_end().to_string());
        }
    }
    if !toc.links.is_empty() {
        out.push(String::new());
        out.push("PLACES you could -net link- to:".into());
        for (n, _, desc) in &toc.links {
            out.push(format!("    {n:<14} {desc}").trim_end().to_string());
        }
    }
    if !toc.pkgs.is_empty() {
        out.push(String::new());
        out.push("PACKAGES you could -net describe-:".into());
        for (n, desc) in &toc.pkgs {
            out.push(format!("    {n:<14} {desc}").trim_end().to_string());
        }
    }
    out.push(rule());
    out.join("\n")
}

/// Set the current location and list its contents.
fn goto_loc(me: &mut AdoInterpreter, loc: &str) -> Result<String, String> {
    let loc = loc.trim_end_matches('/').to_string();
    let loc = if net::is_url(&loc) || is_abs_local(&loc) { loc } else { join_loc(&vfs::current_dir(), &loc) };
    let text = fetch_text(&join_loc(&loc, &toc_file_name())).map_err(|e| {
        if crate::interpreter::rc_of(&e) == 601 { rc_err(601, format!("{loc} has no table of contents ({} not found)", toc_file_name())) } else { e }
    })?;
    me.engine.store.settings.insert("net_from".into(), loc.clone());
    Ok(show_toc(&loc, &parse_toc(&text)))
}

fn net_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let (sub, arg) = first_word(&body);
    let sub = sub.to_lowercase();
    let arg = unquote(arg.trim());
    match sub.as_str() {
        "from" => {
            if arg.is_empty() {
                return Err(rc_err(198, "net from <url or directory>"));
            }
            goto_loc(me, &arg)
        }
        "cd" => {
            let cur = current_loc(me)?;
            if arg.is_empty() {
                return Err(rc_err(198, "net cd path"));
            }
            goto_loc(me, &join_loc(&cur, &arg))
        }
        "link" => {
            let cur = current_loc(me)?;
            let toc = parse_toc(&fetch_text(&join_loc(&cur, &toc_file_name()))?);
            let (_, url, _) = toc.links.iter().find(|(n, _, _)| n.eq_ignore_ascii_case(&arg)).ok_or_else(|| rc_err(601, format!("link {arg} not found at {cur}")))?;
            let target = join_loc(&cur, url);
            goto_loc(me, &target)
        }
        "query" | "q" | "" => {
            let cur = setting(me, "net_from").unwrap_or_else(|| "(none; type net from <url or directory>)".into());
            let ado = ado_dir(me).unwrap_or_default();
            let other = other_dir(me);
            Ok(format!("current location:  {cur}\n   ado directory:  {ado}\n other directory:  {}", if other.is_empty() { vfs::current_dir() } else { other }))
        }
        "set" => {
            let (what, dir) = first_word(&arg);
            let dir = unquote(dir.trim());
            if dir.is_empty() {
                return Err(rc_err(198, "net set ado|other dirname"));
            }
            let key = match what.to_lowercase().as_str() {
                "ado" => "net_ado",
                "other" => "net_other",
                w => return Err(rc_err(198, format!("net set {w}: invalid; use net set ado or net set other"))),
            };
            let code = if crate::progtools::sysdir_path(me, &dir).is_some() { dir.to_uppercase() } else { dir };
            me.engine.store.settings.insert(key.into(), code);
            invalidate();
            Ok(String::new())
        }
        "describe" | "d" | "de" | "des" | "desc" | "descr" | "descri" | "describ" => {
            let (loc, pkg) = pkg_location(me, &arg, opt_arg(&opts, "from", 4))?;
            let p = parse_pkg(&fetch_text(&join_loc(&loc, &format!("{pkg}.pkg")))?);
            Ok(describe_pkg(&loc, &pkg, &p, "net install", "net get", None))
        }
        "install" | "inst" | "insta" | "instal" | "get" => {
            let (loc, pkg) = pkg_location(me, &arg, opt_arg(&opts, "from", 4))?;
            if let Some(msg) = native_message(me, &pkg, &opts) {
                return Ok(msg);
            }
            let o = InstallOpts::from(&opts);
            if sub == "get" { get_ancillary(me, &loc, &pkg, o.replace) } else { install(me, &loc, &pkg, o) }
        }
        "search" | "sea" | "sear" | "searc" => {
            let words: Vec<String> = arg.split_whitespace().map(String::from).collect();
            if words.is_empty() {
                return Err(rc_err(198, "net search word [word ...]"));
            }
            let hits = ssc_search(&words)?;
            Ok(format_ssc_hits(&words, &hits))
        }
        "sj" | "stb" => Err(rc_err(198, format!("net {sub}: journal archives have no default location; use net from <url>"))),
        other => Err(rc_err(198, format!("net {other}: invalid subcommand"))),
    }
}

/// (location, package name) for `net describe|install|get pkg [, from()]`.
fn pkg_location(me: &AdoInterpreter, arg: &str, from: Option<String>) -> Result<(String, String), String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err(rc_err(198, "package name required"));
    }
    let arg = arg.strip_suffix(".pkg").unwrap_or(arg);
    let base = match from {
        Some(f) => {
            let f = unquote(&f);
            Some(if net::is_url(&f) || is_abs_local(&f) { f } else { join_loc(&vfs::current_dir(), &f) })
        }
        None => setting(me, "net_from"),
    };
    let full = if net::is_url(arg) || is_abs_local(arg) {
        arg.to_string()
    } else {
        let b = base.ok_or_else(|| rc_err(198, "no current location; type net from <url or directory> first, or use option from()"))?;
        join_loc(&b, arg)
    };
    match full.rfind(['/', '\\']) {
        Some(i) => Ok((full[..i].to_string(), full[i + 1..].to_string())),
        None => Ok((String::new(), full)),
    }
}

fn describe_pkg(loc: &str, pkg: &str, p: &Pkg, install_cmd: &str, get_cmd: &str, num: Option<usize>) -> String {
    let mut out = vec![rule(), format!("{}package {pkg} from {loc}", num.map(|n| format!("[{n}] ")).unwrap_or_default()), rule(), String::new()];
    out.push("TITLE".into());
    out.push(format!("      {}", p.title()));
    if p.desc.len() > 1 {
        out.push(String::new());
        out.push("DESCRIPTION/AUTHOR(S)".into());
        for d in &p.desc[1..] {
            out.push(format!("      {d}").trim_end().to_string());
        }
    }
    let inst = p.install_files();
    if !inst.is_empty() {
        out.push(String::new());
        out.push(format!("{:<48} (type {install_cmd} {pkg})", "INSTALLATION FILES"));
        for f in inst {
            out.push(format!("      {}", f.src));
        }
    }
    let plat: Vec<&PkgFile> = p.files.iter().filter(|f| matches!(f.tag, 'g' | 'G')).collect();
    if !plat.is_empty() {
        out.push(String::new());
        out.push("PLATFORM-SPECIFIC FILES (not installed by Ferrostat)".into());
        for f in plat {
            out.push(format!("      {}{}", f.src, f.target.as_ref().map(|t| format!(" -> {t}")).unwrap_or_default()));
        }
    }
    let anc = p.ancillary();
    if !anc.is_empty() {
        out.push(String::new());
        out.push(format!("{:<48} (type {get_cmd} {pkg})", "ANCILLARY FILES"));
        for f in anc {
            out.push(format!("      {}", f.src));
        }
    }
    out.push(rule());
    out.join("\n")
}

/// Packages Ferrostat provides natively are not installed unless asked for
/// with `nonative` (or `set ado_native off`).
fn native_message(me: &AdoInterpreter, pkg: &str, opts: &[(String, Option<String>)]) -> Option<String> {
    let force = has_opt(opts, "nonative", 8) || setting(me, "ado_native").as_deref() == Some("off");
    if force {
        return None;
    }
    match crate::progcmd::package_commands(me, &pkg.to_lowercase()) {
        Some(cmds) if cmds.is_empty() => Some(format!("package {pkg}: built-in features cover it; nothing to install")),
        Some(cmds) => Some(format!("package {pkg}: provided natively ({}); nothing to install\n(to install the community version anyway, add option nonative)", cmds.join(", "))),
        None => None,
    }
}

#[derive(Clone, Copy, Default)]
struct InstallOpts {
    replace: bool,
    force: bool,
    all: bool,
    nonative: bool,
}

impl InstallOpts {
    fn from(opts: &[(String, Option<String>)]) -> Self {
        InstallOpts { replace: has_opt(opts, "replace", 7), force: has_opt(opts, "force", 5), all: has_opt(opts, "all", 3), nonative: has_opt(opts, "nonative", 8) }
    }
}

fn today() -> String {
    chrono::Local::now().format("%e %b %Y").to_string().trim().to_string()
}

/// `net install`: copy the package's files into the ado directory.
fn install(me: &mut AdoInterpreter, loc: &str, pkg: &str, o: InstallOpts) -> Result<String, String> {
    let pkg_path = join_loc(loc, &format!("{pkg}.pkg"));
    let p = parse_pkg(&fetch_text(&pkg_path)?);
    let dir = ado_dir(me)?;
    let mut entries = read_trk(&dir);
    let mut out = vec![format!("checking {pkg} consistency and verifying not already installed...")];
    let existing = entries.iter().position(|e| e.pkg().eq_ignore_ascii_case(pkg));
    let reinstall = existing.is_some() && !o.replace && !o.force;
    let owned: Vec<String> = existing.map(|i| entries[i].files.clone()).unwrap_or_default();
    // files to install: (source, destination relative to the ado dir)
    let mut plan: Vec<(String, String)> = Vec::new();
    for f in p.install_files() {
        let b = base_name(&f.src);
        plan.push((join_loc(loc, &f.src), format!("{}/{b}", letter(&b))));
    }
    if plan.is_empty() && p.files.iter().all(|f| matches!(f.tag, 'g' | 'G')) && !p.files.is_empty() {
        return Err(rc_err(198, format!("package {pkg} has only platform-specific (compiled) files, which Ferrostat cannot use")));
    }
    if reinstall {
        // already installed: fine when the files are unchanged
        for (src, dest) in &plan {
            if vfs::read(&format!("{dir}{dest}")).ok() != Some(fetch_bytes(src)?) {
                return Err(rc_err(602, format!("package {pkg} is already installed and its files differ from the ones at {loc}\nno files installed or copied; use option replace to reinstall it")));
            }
        }
        out.push("all files already exist and are identical with the installed files".into());
        return Ok(out.join("\n"));
    }
    let conflicts: Vec<&String> = plan.iter().map(|(_, d)| d).filter(|d| vfs::exists(&format!("{dir}{d}")) && !owned.contains(d)).collect();
    if !conflicts.is_empty() && !o.replace && !o.force {
        let list = conflicts.iter().map(|c| format!("    {dir}{c}")).collect::<Vec<_>>().join("\n");
        return Err(rc_err(602, format!("the following files already exist:\n{list}\nno files installed or copied (use option replace)")));
    }
    // download everything before writing anything
    let mut data = Vec::new();
    for (src, dest) in &plan {
        data.push((dest.clone(), fetch_bytes(src)?));
    }
    let mut anc = Vec::new();
    if o.all {
        for f in p.ancillary() {
            anc.push((base_name(&f.src), fetch_bytes(&join_loc(loc, &f.src))?));
        }
    }
    if let Some(i) = existing {
        let old = entries.remove(i);
        for f in old.files.iter().filter(|f| !plan.iter().any(|(_, d)| d == *f)) {
            let _ = vfs::remove(&format!("{dir}{f}"));
        }
    }
    out.push(format!("installing into {dir}..."));
    for (dest, bytes) in &data {
        let full = format!("{dir}{dest}");
        if let Some(parent) = full.rfind('/').map(|i| &full[..i]) {
            vfs::create_dir_all(parent).map_err(|e| rc_err(603, format!("cannot create {parent}: {e}")))?;
        }
        vfs::write(&full, bytes).map_err(|e| rc_err(603, format!("cannot write {full}: {e}")))?;
    }
    let id = entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;
    entries.push(TrkEntry { src: loc.to_string(), name: format!("{pkg}.pkg"), date: today(), id, desc: p.desc.clone(), files: plan.iter().map(|(_, d)| d.clone()).collect() });
    write_trk(&dir, &entries)?;
    // reload changed programs on next use
    let stems: Vec<String> = plan.iter().filter(|(_, d)| ext(d) == "ado").map(|(_, d)| base_name(d).trim_end_matches(".ado").to_string()).collect();
    for s in &stems {
        me.engine.store.programs.remove(s);
    }
    let mut nn = read_nonative(&dir);
    let before = nn.clone();
    for s in &stems {
        if o.nonative || nn.contains(s) && existing.is_some() {
            nn.insert(s.clone());
        }
    }
    if nn != before {
        write_nonative(&dir, &nn);
    }
    out.push("installation complete.".into());
    let skipped: Vec<&PkgFile> = p.files.iter().filter(|f| matches!(f.tag, 'g' | 'G')).collect();
    if !skipped.is_empty() {
        out.push(format!("(note: {} platform-specific file(s) skipped; compiled plugins are not supported)", skipped.len()));
    }
    if !anc.is_empty() {
        let od = other_dir(me);
        for (name, bytes) in &anc {
            let path = format!("{od}{name}");
            if vfs::exists(&path) && !o.replace {
                out.push(format!("(file {path} already exists; not copied)"));
                continue;
            }
            vfs::write(&path, bytes).map_err(|e| rc_err(603, format!("cannot write {path}: {e}")))?;
            out.push(format!("copying into {}: {name}", if od.is_empty() { "current directory" } else { &od }));
        }
    } else if !p.ancillary().is_empty() {
        out.push(format!("(ancillary files not copied; type net get {pkg} to copy them)"));
    }
    Ok(out.join("\n"))
}

/// `net get`: copy the ancillary files into the "other" directory.
fn get_ancillary(me: &mut AdoInterpreter, loc: &str, pkg: &str, replace: bool) -> Result<String, String> {
    let p = parse_pkg(&fetch_text(&join_loc(loc, &format!("{pkg}.pkg")))?);
    let od = other_dir(me);
    let mut out = vec![format!("checking {pkg} consistency and verifying not already installed...")];
    let files = p.ancillary();
    if files.is_empty() {
        out.push("(no ancillary files)".into());
        return Ok(out.join("\n"));
    }
    for f in files {
        let name = base_name(&f.src);
        let path = format!("{od}{name}");
        if vfs::exists(&path) && !replace {
            return Err(rc_err(602, format!("file {path} already exists; use option replace")));
        }
        let bytes = fetch_bytes(&join_loc(loc, &f.src))?;
        vfs::write(&path, &bytes).map_err(|e| rc_err(603, format!("cannot write {path}: {e}")))?;
        out.push(format!("copying into {}: {name}", if od.is_empty() { "current directory" } else { &od }));
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// ssc
// ---------------------------------------------------------------------------

fn ssc_loc(pkg: &str) -> String {
    format!("{SSC_BASE}{}", ssc_letter(pkg))
}

fn ssc_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let (sub, arg) = first_word(&body);
    let sub = sub.to_lowercase();
    let arg = unquote(arg.trim());
    let pkg = arg.split_whitespace().next().unwrap_or("").to_lowercase();
    let need = |p: &str| if p.is_empty() { Err(rc_err(198, "package name required")) } else { Ok(()) };
    match sub.as_str() {
        "install" | "inst" | "insta" | "instal" => {
            need(&pkg)?;
            if let Some(msg) = native_message(me, &pkg, &opts) {
                return Ok(msg);
            }
            let o = InstallOpts::from(&opts);
            install(me, &ssc_loc(&pkg), &pkg, o).map_err(|e| {
                if crate::interpreter::rc_of(&e) == 601 && e.contains(&format!("{pkg}.pkg")) { rc_err(601, format!("ssc install: \"{pkg}\" not found at SSC, type search {pkg}")) } else { e }
            })
        }
        "describe" | "d" | "de" | "des" | "desc" | "descr" | "descri" | "describ" => {
            need(&pkg)?;
            if pkg.chars().count() == 1 {
                let sub = ssc_letter(&pkg).to_string();
                let toc = parse_toc(&ssc_toc(&sub)?);
                let mut out = vec![format!("Packages at SSC starting with \"{pkg}\""), rule()];
                for (n, d) in &toc.pkgs {
                    out.push(format!("{n:<16} {d}"));
                }
                out.push(rule());
                out.push(format!("({} packages; type ssc describe pkgname for details)", toc.pkgs.len()));
                return Ok(out.join("\n"));
            }
            let loc = ssc_loc(&pkg);
            let text = match fetch_text(&join_loc(&loc, &format!("{pkg}.pkg"))) {
                Ok(t) => t,
                Err(e) => {
                    if let Some(msg) = native_message(me, &pkg, &[]) {
                        return Ok(msg);
                    }
                    return Err(if crate::interpreter::rc_of(&e) == 601 { rc_err(601, format!("ssc describe: \"{pkg}\" not found at SSC, type search {pkg}")) } else { e });
                }
            };
            let p = parse_pkg(&text);
            let mut out = describe_pkg(&loc, &pkg, &p, "ssc install", "net get", None);
            if let Some(cmds) = crate::progcmd::package_commands(me, &pkg) {
                if !cmds.is_empty() {
                    out.push_str(&format!("\n(note: Ferrostat provides {} natively; ssc install {pkg} keeps the native version unless option nonative is given)", cmds.join(", ")));
                }
            }
            let res = me.engine.store.begin_r();
            res.set_str("fn", format!("{pkg}.pkg"));
            Ok(out)
        }
        "uninstall" => {
            need(&pkg)?;
            let dir = dir_option(me, &opts)?;
            uninstall(me, &dir, &pkg)
        }
        "type" | "copy" | "cp" => {
            let file = arg.trim().to_string();
            need(&file)?;
            let url = format!("{SSC_BASE}{}/{file}", ssc_letter(&file));
            let bytes = fetch_bytes(&url).map_err(|e| if crate::interpreter::rc_of(&e) == 601 { rc_err(601, format!("file {file} not found at SSC")) } else { e })?;
            if sub == "type" {
                let t = decode(bytes);
                return Ok(if ext(&file) == "sthlp" || ext(&file) == "hlp" && t.contains("{smcl}") { crate::w10_smcl::render_help(&t) } else { t.trim_end().to_string() });
            }
            let target = if has_opt(&opts, "plus", 4) {
                let d = ado_dir(me)?;
                let rel = format!("{}/{file}", letter(&file));
                let full = format!("{d}{rel}");
                vfs::create_dir_all(full.rsplit_once('/').map(|x| x.0).unwrap_or(".")).map_err(|e| rc_err(603, e))?;
                full
            } else if has_opt(&opts, "personal", 8) {
                format!("{}{file}", crate::progtools::sysdir_path(me, "PERSONAL").unwrap_or_default())
            } else {
                file.clone()
            };
            if vfs::exists(&target) && !has_opt(&opts, "replace", 7) {
                return Err(rc_err(602, format!("file {target} already exists; use option replace")));
            }
            vfs::write(&target, &bytes).map_err(|e| rc_err(603, format!("cannot write {target}: {e}")))?;
            Ok(format!("(file {file} copied to {target})"))
        }
        "new" => ssc_new(&opts),
        "hot" => ssc_hot(&opts),
        "update" | "upd" => adoupdate(me, &format!("{arg}, ssconly{}", if has_opt(&opts, "update", 3) { " update" } else { "" })),
        "" => Err(rc_err(198, "ssc install|describe|uninstall|type|copy|new|hot pkgname")),
        other => Err(rc_err(198, format!("ssc {other}: invalid subcommand"))),
    }
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for c in s.chars() {
        match c {
            '<' => inside = true,
            '>' => inside = false,
            _ if !inside => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&").replace("&nbsp;", " ").replace("&quot;", "\"").replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `ssc new`: the most recent additions to the archive.
fn ssc_new(opts: &[(String, Option<String>)]) -> Result<String, String> {
    let n: usize = opt_arg(opts, "n", 1).and_then(|v| v.trim().parse().ok()).unwrap_or(25);
    let html = fetch_text(SSC_NEW)?;
    let mut out = vec!["New packages at the SSC archive (most recent first)".to_string(), rule()];
    let mut k = 0;
    for item in html.split("<LI").skip(1) {
        let Some(a) = item.find("<A HREF=") else { continue };
        let Some(t0) = item[a..].find('>') else { continue };
        let Some(t1) = item[a + t0..].find("</A>") else { continue };
        let title = strip_tags(&item[a + t0 + 1..a + t0 + t1]);
        let by = item.find("<I>by</I>").map(|i| strip_tags(item[i + 9..].split("</LI>").next().unwrap_or("").split("<LI").next().unwrap_or(""))).unwrap_or_default();
        let (name, desc) = title.split_once(':').map(|(a, b)| (a.trim().to_lowercase(), b.trim().to_string())).unwrap_or((title.clone(), String::new()));
        out.push(format!("{name:<16} {desc}"));
        if !by.is_empty() {
            out.push(format!("{:16} by {by}", ""));
        }
        k += 1;
        if k >= n {
            break;
        }
    }
    if k == 0 {
        return Err(rc_err(631, "could not read the list of new SSC packages"));
    }
    out.push(rule());
    out.push("(type ssc describe pkgname for details)".into());
    Ok(out.join("\n"))
}

/// `ssc hot [, n(#) author(name)]`: most downloaded packages last month.
fn ssc_hot(opts: &[(String, Option<String>)]) -> Result<String, String> {
    let n: usize = opt_arg(opts, "n", 1).and_then(|v| v.trim().parse().ok()).unwrap_or(10).clamp(1, 500);
    let author = opt_arg(opts, "author", 6).map(|a| unquote(&a).to_lowercase());
    let fetch_n = if author.is_some() { 500 } else { n };
    let html = fetch_text(&format!("{SSC_HOT}{fetch_n}"))?;
    let mut rows = Vec::new();
    for tr in html.split("<tr").skip(1) {
        if !tr.contains("boc:bocode:") {
            continue;
        }
        let cells: Vec<String> = tr.split("<td").skip(1).map(|c| strip_tags(&format!("<td{c}"))).collect();
        if cells.len() < 3 {
            continue;
        }
        let rank = cells[0].clone();
        let Some(a) = tr.find("boc:bocode:") else { continue };
        let Some(t0) = tr[a..].find('>') else { continue };
        let Some(t1) = tr[a + t0..].find("</a>") else { continue };
        let title = strip_tags(&tr[a + t0 + 1..a + t0 + t1]);
        let authors = strip_tags(tr[a + t0 + t1..].split("</td>").next().unwrap_or(""));
        let name = title.split(':').next().unwrap_or("").trim().to_lowercase();
        let hits = cells[2].clone();
        if let Some(au) = &author {
            if !authors.to_lowercase().contains(au.as_str()) {
                continue;
            }
        }
        rows.push((rank, name, hits, authors));
        if rows.len() >= n {
            break;
        }
    }
    if rows.is_empty() {
        return Ok("(no packages found)".into());
    }
    let mut out = vec!["Top SSC packages by file downloads last month".to_string(), rule(), format!("{:>4}  {:>6}  {:<16} {}", "Rank", "Hits", "Package", "Author(s)"), rule()];
    for (r, name, h, au) in rows {
        out.push(format!("{r:>4}  {h:>6}  {name:<16} {au}"));
    }
    out.push(rule());
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// ado
// ---------------------------------------------------------------------------

/// Entries selected by `pkgid` (`name`, `[#]`, `#`) or all entries.
fn select(entries: &[TrkEntry], ids: &str) -> Result<Vec<usize>, String> {
    let ids = ids.trim();
    if ids.is_empty() {
        return Ok((0..entries.len()).collect());
    }
    let mut out = Vec::new();
    for id in ids.split_whitespace() {
        let n = id.trim_start_matches('[').trim_end_matches(']');
        let hit: Vec<usize> = if let Ok(k) = n.parse::<u32>() {
            entries.iter().enumerate().filter(|(_, e)| e.id == k).map(|(i, _)| i).collect()
        } else {
            let pat = n.to_lowercase();
            entries.iter().enumerate().filter(|(_, e)| crate::macros::glob_match(&pat, &e.pkg().to_lowercase())).map(|(i, _)| i).collect()
        };
        if hit.is_empty() {
            return Err(rc_err(111, format!("package {n} not installed")));
        }
        out.extend(hit);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn ado_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let (sub, arg) = first_word(&body);
    let sub = sub.to_lowercase();
    let dir = dir_option(me, &opts)?;
    match sub.as_str() {
        "" | "dir" => {
            let entries = read_trk(&dir);
            let idx = select(&entries, &arg)?;
            if idx.is_empty() {
                return Ok("(no packages installed)".into());
            }
            let mut out = Vec::new();
            for i in idx {
                let e = &entries[i];
                out.push(format!("[{}] package {} from {}", e.id, e.pkg(), e.src));
                out.push(format!("      {}", e.title()));
            }
            Ok(out.join("\n"))
        }
        "describe" | "d" | "de" | "des" | "desc" => {
            let entries = read_trk(&dir);
            let idx = select(&entries, &arg)?;
            if idx.is_empty() {
                return Ok("(no packages installed)".into());
            }
            let mut out = Vec::new();
            for i in idx {
                let e = &entries[i];
                out.push(rule());
                out.push(format!("[{}] package {} from {}", e.id, e.pkg(), e.src));
                out.push(rule());
                out.push(String::new());
                out.push("TITLE".into());
                out.push(format!("      {}", e.title()));
                if e.desc.len() > 1 {
                    out.push(String::new());
                    out.push("DESCRIPTION/AUTHOR(S)".into());
                    out.extend(e.desc[1..].iter().map(|d| format!("      {d}").trim_end().to_string()));
                }
                out.push(String::new());
                out.push("INSTALLATION FILES".into());
                out.extend(e.files.iter().map(|f| format!("      {f}")));
                out.push(String::new());
                out.push("INSTALLED ON".into());
                out.push(format!("      {}", e.date));
            }
            out.push(rule());
            Ok(out.join("\n"))
        }
        "uninstall" | "uninst" => {
            if arg.trim().is_empty() {
                return Err(rc_err(198, "ado uninstall pkgname"));
            }
            uninstall(me, &dir, &arg)
        }
        "update" => adoupdate(me, r.trim_start()[sub.len()..].trim_start()),
        other => Err(rc_err(198, format!("ado {other}: invalid subcommand"))),
    }
}

fn uninstall(me: &mut AdoInterpreter, dir: &str, ids: &str) -> Result<String, String> {
    let mut entries = read_trk(dir);
    let idx = select(&entries, ids).map_err(|e| if crate::interpreter::rc_of(&e) == 111 { rc_err(111, format!("criterion matches no packages\n{}", crate::interpreter::err_message(&e))) } else { e })?;
    if idx.len() > 1 && ids.split_whitespace().count() == 1 && !ids.trim_start_matches('[').starts_with(|c: char| c.is_ascii_digit()) && ids.contains(['*', '?']) {
        // a pattern may uninstall several packages; that is allowed
    }
    let mut out = Vec::new();
    let mut nn = read_nonative(dir);
    for &i in idx.iter().rev() {
        let e = entries.remove(i);
        for f in &e.files {
            let _ = vfs::remove(&format!("{dir}{f}"));
            if ext(f) == "ado" {
                let stem = base_name(f).trim_end_matches(".ado").to_string();
                me.engine.store.programs.remove(&stem);
                nn.remove(&stem);
            }
        }
        out.push(format!("package {} from {}\n      {}", e.pkg(), e.src, e.title()));
    }
    write_trk(dir, &entries)?;
    write_nonative(dir, &nn);
    out.push("(package uninstalled)".into());
    Ok(out.join("\n"))
}

/// `adoupdate [pkglist] [, update all ssconly dir()]`.
fn adoupdate(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let dir = dir_option(me, &opts)?;
    let do_update = has_opt(&opts, "update", 3);
    let all = has_opt(&opts, "all", 3);
    let ssconly = has_opt(&opts, "ssconly", 7);
    let entries = read_trk(&dir);
    let idx = select(&entries, &body)?;
    let mut out = Vec::new();
    let mut stale: Vec<TrkEntry> = Vec::new();
    for i in idx {
        let e = &entries[i];
        let ssc_host = SSC_BASE.trim_start_matches("http://").trim_end_matches('/');
        if ssconly && !e.src.contains(ssc_host) {
            continue;
        }
        let remote = match fetch_text(&join_loc(&e.src, &e.name)) {
            Ok(t) => parse_pkg(&t),
            Err(err) => {
                out.push(format!("[{}] {}: could not check ({})", e.id, e.pkg(), crate::interpreter::err_message(&err).lines().next().unwrap_or("")));
                continue;
            }
        };
        let newer = match (dist_date(&e.desc), remote.distribution_date()) {
            (Some(a), Some(b)) => b > a,
            _ => remote.desc != e.desc,
        };
        if newer || all {
            out.push(format!("[{}] {} at {}:\n    {}", e.id, e.pkg(), e.src, if newer { "installed package is not up to date" } else { "will be reinstalled (option all)" }));
            stale.push(e.clone());
        } else {
            out.push(format!("[{}] {} at {}:\n    installed package is up to date", e.id, e.pkg(), e.src));
        }
    }
    let names: Vec<String> = stale.iter().map(|e| e.pkg()).collect();
    if stale.is_empty() {
        out.push("(no packages require updating)".into());
    } else if do_update {
        for e in &stale {
            let o = InstallOpts { replace: true, ..Default::default() };
            match install(me, &e.src, &e.pkg(), o) {
                Ok(_) => out.push(format!("(package {} updated)", e.pkg())),
                Err(err) => out.push(format!("(package {} not updated: {})", e.pkg(), crate::interpreter::err_message(&err).lines().next().unwrap_or(""))),
            }
        }
    } else {
        out.push("packages to be updated are".into());
        for e in &stale {
            out.push(format!("    [{}] {:<16} -- {}", e.id, e.pkg(), e.title()));
        }
        out.push("(type adoupdate, update to update them)".into());
    }
    let mut res = Results::new();
    res.set_str("pkglist", names.join(" "));
    me.engine.store.r = res;
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

fn version_tuple(v: &str) -> Vec<u64> {
    v.trim().trim_start_matches(['v', 'V']).split(['.', '-']).map(|p| p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)).collect()
}

/// `update [query|all|from ...]`: compare with the latest published release.
fn update(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let sub = r.split([',', ' ']).next().unwrap_or("").trim().to_lowercase();
    if !matches!(sub.as_str(), "" | "query" | "q" | "all" | "ado" | "executable" | "utilities" | "swap" | "from") {
        return Err(rc_err(198, format!("update {sub}: invalid subcommand")));
    }
    let current = env!("CARGO_PKG_VERSION");
    let mut out = vec![format!("Ferrostat version installed:  {current}")];
    let json = fetch_text(RELEASES_API)?;
    let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| rc_err(631, format!("unexpected reply from the release server: {e}")))?;
    let tag = v.get("tag_name").and_then(|t| t.as_str()).unwrap_or("").to_string();
    let date = v.get("published_at").and_then(|t| t.as_str()).unwrap_or("").get(..10).unwrap_or("").to_string();
    let page = v.get("html_url").and_then(|t| t.as_str()).unwrap_or(RELEASES_PAGE).to_string();
    let latest = tag.trim_start_matches(['v', 'V']).to_string();
    out.push(format!("Latest release:               {latest}{}", if date.is_empty() { String::new() } else { format!(" ({date})") }));
    let newer = version_tuple(&latest) > version_tuple(current);
    if newer {
        out.push(String::new());
        out.push(format!("A newer version is available. To upgrade, download the build for your platform from\n    {page}\nand replace your installation (or rebuild from source). Ferrostat does not modify its own executable."));
    } else {
        out.push("Ferrostat is up to date.".into());
    }
    let mut res = Results::new();
    res.set_str("current", current.to_string());
    res.set_str("latest", latest);
    res.set("newer", if newer { 1.0 } else { 0.0 });
    me.engine.store.r = res;
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_join() {
        assert_eq!(join_loc("http://a.org/x/y", "../z/p.pkg"), "http://a.org/x/z/p.pkg");
        assert_eq!(join_loc("/tmp/site", "sub"), "/tmp/site/sub");
        assert_eq!(join_loc("/tmp/site", "https://b.org/q"), "https://b.org/q");
        assert_eq!(toc_file_name().len(), 9);
    }

    #[test]
    fn pkg_parse() {
        let p = parse_pkg("v 3\nd 'X': a module\nd Distribution-Date: 20200101\nf x.ado\nf x.sthlp\nf x.dta\ng LINUX64 x.plugin\ne\nf ignored.ado\n");
        assert_eq!(p.install_files().len(), 2);
        assert_eq!(p.ancillary().len(), 1);
        assert_eq!(p.distribution_date().as_deref(), Some("20200101"));
        assert_eq!(letter("_gx.ado"), '_');
    }
}

/// `c_local name value`: define a local macro in the caller's scope (used
/// by community ado-files to return results in a local named by an option).
pub fn c_local(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let name = r.split(|c: char| c.is_whitespace() || c == '=' || c == ':').next().unwrap_or("").to_string();
    if name.is_empty() {
        return Err(rc_err(198, "c_local name value"));
    }
    let st = &mut me.engine.store;
    let saved = st.local(&name).cloned();
    crate::progcmd::interp_command(me, "local", r, &format!("local {r}"))?;
    let st = &mut me.engine.store;
    let value = st.local(&name).cloned().unwrap_or_default();
    match saved {
        Some(v) => st.set_local(&name, v),
        None => st.clear_local(&name),
    }
    let n = st.frames.len();
    if n >= 2 {
        if value.is_empty() {
            st.frames[n - 2].remove(&name);
        } else {
            st.frames[n - 2].insert(name, value);
        }
    }
    Ok(String::new())
}
