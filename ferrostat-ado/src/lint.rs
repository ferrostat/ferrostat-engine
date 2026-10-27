//! Static checks of do-files without running them (`ferrostat --check` and
//! the `lint` command): block structure, strings and comments, unknown
//! commands and local macros used but never defined (or defined but never
//! used). Diagnostics are reported like a compiler's:
//! `file.do:12:5: error: unbalanced "{" opened here`.

use std::collections::{HashMap, HashSet};

use ferrostat_core::vfs;

use crate::interpreter::{split_if_command, split_verb};
use crate::script::{is_end, is_program_header, opens_block, strip_line_comment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diag {
    pub line: usize,
    pub col: usize,
    pub severity: Severity,
    pub message: String,
}

impl Diag {
    pub fn format(&self, file: &str) -> String {
        format!("{file}:{}:{}: {}: {}", self.line, self.col, self.severity.as_str(), self.message)
    }
}

/// Counts of (errors, warnings, notes).
pub fn counts(diags: &[Diag]) -> (usize, usize, usize) {
    let n = |s| diags.iter().filter(|d| d.severity == s).count();
    (n(Severity::Error), n(Severity::Warning), n(Severity::Note))
}

pub fn summary(errors: usize, warnings: usize, notes: usize) -> String {
    let pl = |n: usize, w: &str| format!("{n} {w}{}", if n == 1 { "" } else { "s" });
    format!("{}, {}, {}", pl(errors, "error"), pl(warnings, "warning"), pl(notes, "note"))
}

// ---------------------------------------------------------------------------
// statements with positions

/// A logical statement and where its pieces came from: (offset in `text`,
/// line, column), both 1-based, for each physical line it spans.
#[derive(Debug, Clone)]
struct Stmt {
    text: String,
    segs: Vec<(usize, usize, usize)>,
}

impl Stmt {
    fn pos(&self, off: usize) -> (usize, usize) {
        let s = self.segs.iter().rev().find(|s| s.0 <= off).or(self.segs.first()).copied().unwrap_or((0, 1, 1));
        (s.1, s.2 + off.saturating_sub(s.0))
    }

    /// The statement from byte `from` on, trimmed.
    fn sub(&self, from: usize) -> Stmt {
        let rest = &self.text[from..];
        let lead = rest.len() - rest.trim_start().len();
        let start = from + lead;
        let (line, col) = self.pos(start);
        let mut segs = vec![(0, line, col)];
        segs.extend(self.segs.iter().filter(|s| s.0 > start).map(|s| (s.0 - start, s.1, s.2)));
        Stmt { text: self.text[start..].trim_end().to_string(), segs }
    }
}

/// Builder of the current statement.
#[derive(Default)]
struct Cur {
    text: String,
    segs: Vec<(usize, usize, usize)>,
}

impl Cur {
    fn take(&mut self) -> Option<Stmt> {
        let st = Stmt { text: std::mem::take(&mut self.text), segs: std::mem::take(&mut self.segs) };
        if st.text.trim().is_empty() {
            return None;
        }
        Some(st.sub(0))
    }
}

/// Replace `/* */` comments (outside strings, not inside `//` or `*`
/// comment lines) with blanks, keeping line and column positions.
fn blank_comments(text: &str, diags: &mut Vec<Diag>) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let (mut line, mut col) = (1usize, 1usize);
    let mut depth = 0;
    let mut open = (0, 0);
    let mut in_str = false;
    let mut compound = 0;
    let mut line_comment = false;
    let mut at_start = true;
    let mut delimit = false;
    let mut prev_cont = false;
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch == '\n' {
            // continuation: a line ending in `///` (outside comments)
            let line_text: String = c[i.saturating_sub(col - 1)..i].iter().collect();
            prev_cont = depth == 0 && strip_line_comment(&line_text).1;
            out.push('\n');
            line += 1;
            col = 1;
            in_str = false;
            compound = 0;
            line_comment = false;
            at_start = true;
            i += 1;
            continue;
        }
        if depth > 0 {
            if ch == '*' && c.get(i + 1) == Some(&'/') {
                depth -= 1;
                out.push_str("  ");
                i += 2;
                col += 2;
                continue;
            }
            if ch == '/' && c.get(i + 1) == Some(&'*') {
                depth += 1;
                out.push_str("  ");
                i += 2;
                col += 2;
                continue;
            }
            out.push(if ch == '\t' { '\t' } else { ' ' });
            i += 1;
            col += 1;
            continue;
        }
        if at_start && !ch.is_whitespace() {
            at_start = false;
            let rest: String = c[i..c.len().min(i + 9)].iter().collect::<String>().to_lowercase();
            if rest.starts_with("#d") {
                let l: String = c[i..].iter().take_while(|&&x| x != '\n').collect::<String>().to_lowercase();
                let arg = l.split_whitespace().nth(1).unwrap_or(";");
                delimit = (arg.starts_with(';') || l.trim_end().ends_with(';')) && !l.trim_end().ends_with("cr");
            }
            if ch == '*' && !delimit && !prev_cont {
                line_comment = true;
            }
        }
        if !line_comment {
            if compound == 0 && ch == '"' {
                in_str = !in_str;
            } else if ch == '`' && c.get(i + 1) == Some(&'"') {
                compound += 1;
            } else if ch == '"' && c.get(i + 1) == Some(&'\'') && compound > 0 {
                compound -= 1;
            }
            if !in_str && compound == 0 {
                if ch == '/' && c.get(i + 1) == Some(&'/') && (i == 0 || c[i - 1].is_whitespace()) {
                    line_comment = true;
                } else if ch == '/' && c.get(i + 1) == Some(&'*') {
                    depth = 1;
                    open = (line, col);
                    out.push_str("  ");
                    i += 2;
                    col += 2;
                    continue;
                }
            }
        }
        out.push(ch);
        i += 1;
        col += 1;
    }
    if depth > 0 {
        // the rest of the file is ignored, which runs; most likely a mistake
        diags.push(Diag { line: open.0, col: open.1, severity: Severity::Warning, message: "unterminated \"/*\" comment opened here (the rest of the file is ignored)".into() });
    }
    out
}

fn find_semicolon(s: &str) -> Option<usize> {
    let mut in_str = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            in_str = !in_str;
        }
        if c == ';' && !in_str {
            return Some(i);
        }
    }
    None
}

/// Logical statements with positions: `//` and `*` comments, `///`
/// continuation, `#delimit ;`, and `} else {` split at the brace.
fn statements(text: &str, diags: &mut Vec<Diag>) -> Vec<Stmt> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let blank = blank_comments(&text, diags);
    let mut out = Vec::new();
    let mut cur = Cur::default();
    let mut delimit = false;
    let mut last_cont = 0;
    let emit = |st: Stmt, out: &mut Vec<Stmt>| {
        let mut st = st;
        loop {
            if st.text.starts_with('}') && st.text.len() > 1 {
                out.push(Stmt { text: "}".into(), segs: vec![st.segs[0]] });
                st = st.sub(1);
                continue;
            }
            out.push(st);
            break;
        }
    };
    for (k, raw) in blank.split('\n').enumerate() {
        let ln = k + 1;
        let trimmed = raw.trim();
        let lower = trimmed.to_lowercase();
        if cur.text.trim().is_empty() && !delimit && trimmed.starts_with('*') {
            continue;
        }
        if cur.text.trim().is_empty() && (lower.starts_with("#delimit") || lower.starts_with("#d ") || lower == "#d;" || lower == "#d") {
            let arg = trimmed.split_whitespace().nth(1).unwrap_or(";");
            delimit = (arg.starts_with(';') || trimmed.ends_with(';')) && !lower.ends_with("cr");
            continue;
        }
        let (code, cont) = strip_line_comment(raw);
        let lead = code.len() - code.trim_start().len();
        let push = |cur: &mut Cur, code: &str| {
            if !code.trim().is_empty() {
                cur.segs.push((cur.text.len(), ln, lead + 1));
                cur.text.push_str(code.trim_start());
            }
        };
        if delimit {
            push(&mut cur, &code);
            cur.text.push(' ');
            while let Some(pos) = find_semicolon(&cur.text) {
                let st = Stmt { text: cur.text[..pos].to_string(), segs: cur.segs.clone() };
                let rest = cur.text[pos + 1..].to_string();
                let rest_pos = st.pos(pos + 1);
                cur = Cur::default();
                if !rest.trim().is_empty() {
                    let l = rest.len() - rest.trim_start().len();
                    let p = Stmt { text: rest.clone(), segs: vec![(0, rest_pos.0, rest_pos.1)] }.pos(l);
                    cur.segs.push((0, p.0, p.1));
                    cur.text.push_str(rest.trim_start());
                }
                if st.text.trim().is_empty() || st.text.trim().starts_with('*') {
                    continue;
                }
                if let Some(s) = (Cur { text: st.text, segs: st.segs }).take() {
                    emit(s, &mut out);
                }
            }
            continue;
        }
        if cont {
            push(&mut cur, &code);
            cur.text.push(' ');
            last_cont = ln;
            continue;
        }
        push(&mut cur, &code);
        if let Some(st) = cur.take() {
            emit(st, &mut out);
        }
    }
    if let Some(st) = cur.take() {
        if !delimit && last_cont > 0 {
            diags.push(Diag { line: last_cont, col: 1, severity: Severity::Warning, message: "file ends with a \"///\" continuation".into() });
        }
        if delimit {
            let (l, c) = st.pos(0);
            diags.push(Diag { line: l, col: c, severity: Severity::Warning, message: "statement not terminated by \";\" (#delimit ; is in effect)".into() });
        }
        emit(st, &mut out);
    }
    out
}

// ---------------------------------------------------------------------------
// command words

/// Commands the interpreter handles itself (with their usual abbreviations).
const INTERP_WORDS: &[&str] = &[
    "about", "ado", "adopath", "args", "cd", "cf", "chdir", "cls", "cmdlog", "collect", "confirm", "copy", "creturn", "dir", "discard", "display", "do", "dtable", "dyndoc", "dyntext", "erase", "ereturn", "error", "estadd", "estout", "estpost", "eststo", "esttab", "etable", "findit", "frmttable", "fvunab", "gettoken", "global", "help", "html2docx", "include", "local", "log", "macro", "markdown", "markout", "marksample", "mata", "mkdir", "more", "net", "numlist", "outreg", "outreg2", "pause", "program", "putdocx", "putexcel", "putpdf", "pwd", "python", "query", "return", "rmdir", "run", "scalar", "search", "set", "shell", "sleep", "sreturn", "ssc", "syntax", "tempfile", "tempname", "tempvar", "timer", "tokenize", "tsunab", "type", "unab", "which", "winexec", "xshell", "lint", "foreach", "forvalues", "while", "if", "else", "capture", "quietly", "noisily", "by", "bysort", "exit", "quit", "continue", "break", "version", "end", "clear", "preserve", "restore", "sysuse", "assert", "c_local", "bootstrap", "jackknife", "simulate", "permute", "frame", "frames", "xi", "input", "mata:", "python:", "class", "file", "window", "plugin", "_return", "_estimates",
];

/// Two-letter abbreviations accepted for interpreter commands.
const SHORT_WORDS: &[&str] = &["di", "gl", "ma", "pr", "bs", "sw", "cd", "do", "if", "by"];

/// Prefix commands followed by `:` and another command.
const PREFIXES: &[&str] = &[
    "by", "bys", "byso", "bysor", "bysort", "xi", "svy", "mi", "bootstrap", "bs", "jackknife", "jknife", "jack", "simulate", "sim", "simul", "permute", "statsby", "rolling", "nestreg", "stepwise", "sw", "eststo", "estpost", "version", "vers", "frame", "frames", "timer", "fp", "mfp", "collect", "fmm", "parallel", "rhsbsample",
];

fn is_capture(v: &str) -> bool {
    v.len() >= 3 && "capture".starts_with(v)
}

fn is_quiet_noisy(v: &str) -> bool {
    v.len() >= 3 && "quietly".starts_with(v) || !v.is_empty() && "noisily".starts_with(v)
}

fn is_local_cmd(v: &str) -> bool {
    v.len() >= 3 && "local".starts_with(v)
}

/// Words accepted as commands: the engine's (with abbreviations), the
/// interpreter's, and the natively provided packages.
pub struct Commands {
    words: HashSet<String>,
    full: Vec<String>,
    /// programs defined in the files being checked (and the session)
    pub programs: HashSet<String>,
}

impl Commands {
    pub fn new(engine_words: Vec<String>) -> Commands {
        let mut words: HashSet<String> = engine_words.iter().cloned().collect();
        for w in INTERP_WORDS.iter().chain(crate::progcmd::INTERP_COMMANDS) {
            for len in 3.min(w.len())..=w.len() {
                if w.is_char_boundary(len) {
                    words.insert(w[..len].to_string());
                }
            }
        }
        words.extend(SHORT_WORDS.iter().map(|s| s.to_string()));
        words.extend(PREFIXES.iter().map(|s| s.to_string()));
        words.extend(crate::progcmd::NATIVE_PACKAGES.iter().map(|s| s.to_string()));
        let mut full: Vec<String> = engine_words;
        full.extend(INTERP_WORDS.iter().map(|s| s.to_string()));
        full.sort();
        full.dedup();
        Commands { words, full, programs: HashSet::new() }
    }

    fn known(&self, v: &str) -> bool {
        self.words.contains(v) || self.programs.contains(v)
    }
}

/// `program define name` headers in `text`.
pub fn defined_programs(text: &str) -> Vec<String> {
    let mut diags = Vec::new();
    statements(text, &mut diags)
        .iter()
        .filter(|s| is_program_header(&s.text))
        .filter_map(|s| {
            let w: Vec<&str> = s.text.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).collect();
            let name = if w.get(1).is_some_and(|x| { let x = x.to_lowercase(); x.len() >= 3 && "define".starts_with(&x) }) { w.get(2) } else { w.get(1) };
            name.map(|n| n.to_string())
        })
        .collect()
}

/// Files run by `do`/`run`/`include` in `text` (paths as written).
fn called_files(text: &str) -> Vec<String> {
    let mut diags = Vec::new();
    statements(text, &mut diags)
        .iter()
        .filter_map(|s| {
            let (v, rest) = command_of(&s.text).0;
            let v = v.to_lowercase();
            if !matches!(v.as_str(), "do" | "run" | "include") {
                return None;
            }
            let r = rest.trim();
            let p = if let Some(q) = r.strip_prefix('"') { q.split('"').next().unwrap_or("") } else { r.split_whitespace().next().unwrap_or("") };
            (!p.is_empty() && !p.contains('`') && !p.contains('$')).then(|| p.to_string())
        })
        .collect()
}

fn dir_of(path: &str) -> String {
    match path.rfind(['/', '\\']) {
        Some(i) => path[..i].to_string(),
        None => String::new(),
    }
}

fn join(dir: &str, p: &str) -> String {
    if dir.is_empty() || p.starts_with('/') || p.contains(':') {
        p.to_string()
    } else {
        format!("{dir}/{p}")
    }
}

/// Programs available to `path`: defined in it or in the files it runs.
fn programs_for(path: &str, text: &str) -> Vec<String> {
    let mut out = defined_programs(text);
    let dir = dir_of(path);
    for f in called_files(text) {
        let mut p = join(&dir, &f);
        if !p.to_lowercase().ends_with(".do") && !p.to_lowercase().ends_with(".ado") && !vfs::exists(&p) {
            p.push_str(".do");
        }
        if let Ok(t) = vfs::read_to_string(&p) {
            out.extend(defined_programs(&t));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// statement analysis

/// Position of the first top-level `:` (outside quotes, parentheses, macros).
fn top_colon(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut mac = 0;
    for (i, c) in s.char_indices() {
        match c {
            '"' if mac == 0 => in_str = !in_str,
            '`' if !in_str => mac += 1,
            '\'' if !in_str && mac > 0 => mac -= 1,
            '(' | '[' if !in_str => depth += 1,
            ')' | ']' if !in_str => depth -= 1,
            ':' if !in_str && depth == 0 && mac == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// The command of a statement after `capture`/`quietly`/`by ...:` and
/// other prefixes: ((verb, rest), offset of the verb, under capture).
fn command_of(text: &str) -> ((String, String), usize, bool) {
    let mut off = 0;
    let mut captured = false;
    loop {
        let t = &text[off..];
        let lead = t.len() - t.trim_start().len();
        let t = t.trim_start();
        let (v, rest) = split_verb(t);
        let lv = v.to_lowercase();
        let after_verb = off + lead + v.len();
        if is_capture(&lv) || is_quiet_noisy(&lv) {
            captured |= is_capture(&lv);
            let r = &text[after_verb..];
            let skip = r.len() - r.trim_start().trim_start_matches(':').len();
            if r.trim().is_empty() {
                return ((v, rest), off + lead, captured);
            }
            off = after_verb + skip;
            continue;
        }
        if PREFIXES.contains(&lv.as_str()) {
            if let Some(c) = top_colon(&text[after_verb..]) {
                off = after_verb + c + 1;
                continue;
            }
        }
        return ((v, rest), off + lead, captured);
    }
}

/// Macro references `` `name' `` in `s`: (byte offset, name). Compound
/// quotes are skipped; nested references are returned with their inner
/// ones (an outer name containing a macro is returned as written).
fn macro_refs(s: &str) -> Vec<(usize, String)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'`' if b.get(i + 1) == Some(&b'"') => i += 2,
            b'"' if b.get(i + 1) == Some(&b'\'') => i += 2,
            b'`' => {
                stack.push(i);
                i += 1;
            }
            b'\'' => {
                if let Some(start) = stack.pop() {
                    out.push((start, s[start + 1..i].to_string()));
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_ascii_alphabetic() || f == '_') && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

/// A glob pattern (`*` for unknown parts) from a name built with macros.
fn dyn_pattern(name: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in name.chars() {
        match c {
            '`' => {
                if depth == 0 {
                    out.push('*');
                }
                depth += 1;
            }
            '\'' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            '$' => out.push('*'),
            c => out.push(c),
        }
    }
    out
}

fn glob(pat: &str, s: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return pat == s;
    }
    let mut rest = s;
    for (k, p) in parts.iter().enumerate() {
        if k == 0 {
            match rest.strip_prefix(p) {
                Some(r) => rest = r,
                None => return false,
            }
        } else if k == parts.len() - 1 {
            return rest.ends_with(p);
        } else {
            match rest.find(p) {
                Some(i) => rest = &rest[i + p.len()..],
                None => return false,
            }
        }
    }
    true
}

#[derive(Default)]
struct Scope {
    /// locals defined by `local` (unused ones are reported): first definition position
    locals: HashMap<String, (usize, usize)>,
    /// every other definition (loop variables, args, syntax, tempvar, ...)
    defs: HashSet<String>,
    dyn_defs: Vec<String>,
    /// uses: first position of each name
    uses: Vec<(String, usize, usize)>,
    used: HashSet<String>,
    dyn_uses: Vec<String>,
    /// `include` (or `syntax` with unknowable locals) makes undefined uses unknowable
    opaque: bool,
    program: bool,
}

impl Scope {
    fn def(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        if name.contains('`') || name.contains('$') {
            self.dyn_defs.push(dyn_pattern(name));
        } else {
            self.defs.insert(name.to_string());
        }
    }

    fn def_local(&mut self, name: &str, pos: (usize, usize)) {
        let name = name.trim();
        if is_ident(name) {
            self.locals.entry(name.to_string()).or_insert(pos);
        } else {
            self.def(name);
        }
    }

    fn use_name(&mut self, name: &str, pos: (usize, usize)) {
        if name.contains('`') || name.contains('$') {
            self.dyn_uses.push(dyn_pattern(name));
            return;
        }
        if !is_ident(name) {
            return;
        }
        if self.used.insert(name.to_string()) {
            self.uses.push((name.to_string(), pos.0, pos.1));
        }
    }

    fn defined(&self, name: &str, global: &HashSet<String>) -> bool {
        self.locals.contains_key(name) || self.defs.contains(name) || global.contains(name) || self.dyn_defs.iter().any(|p| glob(p, name))
    }

    fn finish(self, global: &HashSet<String>, diags: &mut Vec<Diag>) {
        if !self.opaque {
            for (name, line, col) in &self.uses {
                if !self.defined(name, global) {
                    let what = if self.program { "in this program" } else { "in this file" };
                    diags.push(Diag { line: *line, col: *col, severity: Severity::Warning, message: format!("local macro `{name}' is used but never defined {what}") });
                }
            }
        }
        for (name, (line, col)) in &self.locals {
            if !self.used.contains(name) && !self.dyn_uses.iter().any(|p| glob(p, name)) {
                diags.push(Diag { line: *line, col: *col, severity: Severity::Note, message: format!("local macro `{name}' is defined but never used") });
            }
        }
    }
}

/// Locals created by `syntax`: the standard ones plus each option's name.
fn syntax_locals(spec: &str, scope: &mut Scope) {
    for n in ["varlist", "typlist", "namelist", "anything", "if", "in", "using", "weight", "exp", "options", "0"] {
        scope.def(n);
    }
    let spec = spec.replace(['[', ']'], " ");
    let spec = spec.as_str();
    let (main, opts) = match top_level_comma(spec) {
        Some(i) => (&spec[..i], &spec[i + 1..]),
        None => (spec, ""),
    };
    // `anything(name=eqlist)` and the like
    let mut rest = main;
    while let Some(i) = rest.find("name=") {
        let n: String = rest[i + 5..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        scope.def(&n);
        rest = &rest[i + 5..];
    }
    for w in ferrostat_parser::parser::split_top_level_words(&opts.replace(['[', ']'], " ")) {
        let name: String = w.split('(').next().unwrap_or("").to_string();
        if name == "*" {
            continue;
        }
        let lower = name.to_lowercase();
        scope.def(&lower);
        // noCONStant: the local is `constant`
        if name.starts_with("no") && name[2..].starts_with(|c: char| c.is_ascii_uppercase()) {
            scope.def(&lower[2..]);
        }
    }
}

fn top_level_comma(s: &str) -> Option<usize> {
    ferrostat_parser::parser::find_top_level(s, ',')
}

/// Record the locals a statement defines and uses.
fn scan_macros(st: &Stmt, verb: &str, rest: &str, rest_off: usize, scope: &mut Scope, global: &mut HashSet<String>) {
    let text = &st.text;
    for (off, name) in macro_refs(text) {
        let name = name.trim_start_matches("++").trim_start_matches("--").trim_end_matches("++").trim_end_matches("--");
        if name.starts_with('=') || name.starts_with(':') || name.contains('(') && !name.starts_with("macval(") || name.contains('[') {
            continue;
        }
        let name = name.strip_prefix("macval(").map(|n| n.trim_end_matches(')')).unwrap_or(name);
        if name.chars().all(|c| c.is_ascii_digit()) || name == "*" || name.is_empty() {
            continue;
        }
        scope.use_name(name, st.pos(off));
    }
    // options that store results in a local: levelsof x, local(name)
    let lower = text.to_lowercase();
    for key in ["local(", "loc(", "_request(", "_request2(", "st_local(\""] {
        let mut from = 0;
        while let Some(i) = lower[from..].find(key) {
            let s = from + i + key.len();
            let n: String = text[s..].trim_start().chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '`' || *c == '\'').collect();
            if key.starts_with("st_local") {
                // st_local("name") reads; st_local("name", value) sets
                let after = text[s + n.len()..].trim_start_matches('"').trim_start();
                if after.starts_with(',') {
                    scope.def(&n);
                } else {
                    scope.use_name(&n, st.pos(s));
                }
            } else {
                scope.def(&n);
            }
            from = s;
        }
    }
    // names come before any options
    let words: Vec<&str> = rest.split(',').next().unwrap_or("").split_whitespace().collect();
    let pos = st.pos(rest_off);
    match verb {
        v if is_local_cmd(v) => {
            let r = rest.trim_start();
            let name: String = r.chars().take_while(|c| !c.is_whitespace() && *c != '=' && *c != ':').collect();
            let plain = name.trim_start_matches("++").trim_start_matches("--");
            let name_off = rest_off + (rest.len() - r.len());
            if name.starts_with("++") || name.starts_with("--") {
                scope.use_name(plain, st.pos(name_off));
                scope.def(plain);
            } else {
                scope.def_local(plain, st.pos(name_off));
            }
            // extended functions taking macro names: `: list a | b`, `: list sizeof a`
            let after = r[name.len()..].trim_start();
            if let Some(f) = after.strip_prefix(':') {
                let f = f.trim_start();
                if f.starts_with("list ") || f.starts_with("word count") || f.starts_with("list") {
                    for w in f.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                        if is_ident(w) {
                            scope.used.insert(w.to_string());
                        }
                    }
                }
            }
        }
        "tempvar" | "tempname" | "tempfile" => words.iter().for_each(|w| scope.def(w)),
        "args" => words.iter().for_each(|w| scope.def(w)),
        "foreach" => {
            if let Some(w) = words.first() {
                scope.def(w);
            }
            if words.get(1) == Some(&"of") && words.get(2).is_some_and(|w| is_local_cmd(w)) {
                if let Some(n) = words.get(3) {
                    scope.use_name(n.trim_end_matches('{'), pos);
                }
            }
        }
        v if v.len() >= 4 && "forvalues".starts_with(v) => {
            let name: String = rest.trim_start().chars().take_while(|c| !c.is_whitespace() && *c != '=').collect();
            scope.def(&name);
        }
        "syntax" | "syntax:" => syntax_locals(rest, scope),
        "marksample" => {
            if let Some(w) = words.first() {
                scope.def(w);
            }
        }
        "unab" | "tsunab" | "fvunab" => {
            if let Some(c) = rest.find(':') {
                scope.def(rest[..c].trim());
            }
        }
        "gettoken" => {
            if let Some(c) = rest.find(':') {
                let mut global_next = false;
                for w in rest[..c].split_whitespace() {
                    match w {
                        "(local)" => global_next = false,
                        "(global)" => global_next = true,
                        _ if global_next => global_next = false,
                        _ => scope.def(w),
                    }
                }
                let src = rest[c + 1..].split(',').next().unwrap_or("").trim();
                if let Some(n) = src.strip_prefix("(local)") {
                    scope.use_name(n.trim(), pos);
                } else if !src.starts_with("(global)") {
                    scope.use_name(src, pos);
                }
            }
        }
        "c_local" => {
            if let Some(w) = words.first() {
                global.insert(w.to_string());
            }
        }
        "file" => {
            // file read handle name
            if words.first() == Some(&"read") {
                if let Some(n) = words.get(2) {
                    scope.def(n);
                }
            }
        }
        "include" => scope.opaque = true,
        "macro" | "ma" | "mac" => {
            // macro shift renumbers `1', `2', ...
        }
        _ => {}
    }
}

/// Unterminated strings and compound quotes in a statement.
fn check_strings(st: &Stmt, diags: &mut Vec<Diag>) {
    let b = st.text.as_bytes();
    let mut in_str: Option<usize> = None;
    let mut compound: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if let Some(_s) = in_str {
            if b[i] == b'"' {
                in_str = None;
            }
            i += 1;
            continue;
        }
        if b[i] == b'`' && b.get(i + 1) == Some(&b'"') {
            compound.push(i);
            i += 2;
            continue;
        }
        if b[i] == b'"' && b.get(i + 1) == Some(&b'\'') && !compound.is_empty() {
            compound.pop();
            i += 2;
            continue;
        }
        if b[i] == b'"' && compound.is_empty() {
            in_str = Some(i);
        }
        i += 1;
    }
    if let Some(s) = in_str {
        let (l, c) = st.pos(s);
        diags.push(Diag { line: l, col: c, severity: Severity::Error, message: "unterminated string".into() });
    } else if let Some(&s) = compound.first() {
        let (l, c) = st.pos(s);
        diags.push(Diag { line: l, col: c, severity: Severity::Error, message: "unterminated compound quote `\"".into() });
    }
}

// ---------------------------------------------------------------------------
// the checker

enum Frame {
    Brace { line: usize, col: usize, last_if: bool },
    Program { line: usize, col: usize, name: String, last_if: bool },
}

/// Did the previous statement of the innermost block (or program, or file) start an `if`?
fn last_if_mut<'a>(frames: &'a mut [Frame], top: &'a mut bool) -> &'a mut bool {
    match frames.last_mut() {
        Some(Frame::Brace { last_if, .. }) | Some(Frame::Program { last_if, .. }) => last_if,
        None => top,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Raw {
    Mata,
    Python,
    Input,
    /// inline `infix dictionary ... { ... }`: closed by `}`
    Dict,
}

/// Check one do-file's text. `path` locates files it runs (for the
/// programs they define) and `.ado` files next to it.
pub fn check_text(path: &str, text: &str, cmds: &Commands) -> Vec<Diag> {
    let mut diags = Vec::new();
    let stmts = statements(text, &mut diags);
    let mut programs: HashSet<String> = programs_for(path, text).into_iter().map(|p| p.to_lowercase()).collect();
    programs.extend(cmds.programs.iter().cloned());
    let dir = dir_of(path);
    let mut frames: Vec<Frame> = Vec::new();
    let mut top_last_if = false;
    let mut raw: Option<(Raw, usize, usize)> = None;
    let mut global: HashSet<String> = HashSet::new();
    let mut scopes: Vec<Scope> = vec![Scope::default()];
    let mut finished: Vec<Scope> = Vec::new();
    let mut exit_line = None;

    for st in &stmts {
        let t = st.text.trim();
        let (line, col) = st.pos(0);
        if let Some((kind, _, _)) = raw {
            if kind == Raw::Dict {
                if t == "}" {
                    raw = None;
                }
                continue;
            }
            if is_end(t) {
                raw = None;
            } else if kind == Raw::Mata {
                // macros and st_local() in Mata code
                let scope = scopes.last_mut().unwrap();
                scan_macros(st, "", "", 0, scope, &mut global);
            }
            continue;
        }
        if t == "}" {
            match frames.last() {
                Some(Frame::Brace { .. }) => {
                    frames.pop();
                }
                _ => diags.push(Diag { line, col, severity: Severity::Error, message: "unexpected \"}\": no block is open".into() }),
            }
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        if is_program_header(t) {
            let name = defined_programs(t).into_iter().next().unwrap_or_default();
            if let Some(Frame::Program { line: pl, name: pn, .. }) = frames.iter().find(|f| matches!(f, Frame::Program { .. })) {
                diags.push(Diag { line, col, severity: Severity::Error, message: format!("program {name} defined inside program {pn} (line {pl}); is an \"end\" missing?") });
            }
            frames.push(Frame::Program { line, col, name, last_if: false });
            scopes.push(Scope { program: true, ..Default::default() });
            continue;
        }
        if is_end(t) {
            match frames.iter().rposition(|f| matches!(f, Frame::Program { .. })) {
                Some(p) => {
                    for f in frames.drain(p + 1..) {
                        if let Frame::Brace { line: bl, col: bc, .. } = f {
                            diags.push(Diag { line: bl, col: bc, severity: Severity::Error, message: format!("unbalanced \"{{\" opened here (the program ends at line {line})") });
                        }
                    }
                    frames.pop();
                    if let Some(s) = scopes.pop() {
                        finished.push(s);
                    }
                }
                None => diags.push(Diag { line, col, severity: Severity::Error, message: "\"end\" without program, mata or python".into() }),
            }
            continue;
        }
        let words: Vec<String> = t.split_whitespace().take(2).map(|w| w.to_lowercase()).collect();
        let first = words.first().map(String::as_str).unwrap_or("");
        let alone = words.len() == 1 || words.get(1).is_some_and(|w| w == "{");
        let raw_kind = match first {
            "mata" | "mata:" if alone => Some(Raw::Mata),
            "python" | "python:" if words.len() == 1 => Some(Raw::Python),
            "input" | "inp" | "inpu" if !t.contains(',') && !frames.iter().any(|f| matches!(f, Frame::Program { .. })) => Some(Raw::Input),
            _ => None,
        };
        let raw_kind = raw_kind.or_else(|| (opens_block(t) && ferrostat_cmd::infile::is_dictionary_header(t)).then_some(Raw::Dict));
        if let Some(k) = raw_kind {
            raw = Some((k, line, col));
            if k == Raw::Input {
                scan_macros(st, "input", "", 0, scopes.last_mut().unwrap(), &mut global);
            }
            *last_if_mut(&mut frames, &mut top_last_if) = false;
            continue;
        }
        check_strings(st, &mut diags);

        // if / else chains
        let li = last_if_mut(&mut frames, &mut top_last_if);
        let mut body = st.clone();
        let mut check_cmd = true;
        if first == "else" {
            if !*li {
                diags.push(Diag { line, col, severity: Severity::Error, message: "\"else\" without \"if\"".into() });
            }
            let rest = st.sub(4);
            let rl = rest.text.to_lowercase();
            if rl.starts_with("if ") || rl == "if" || rl.starts_with("if(") {
                *li = true;
                body = rest;
            } else {
                *li = false;
                if rest.text.is_empty() || rest.text == "{" {
                    check_cmd = false;
                } else {
                    body = rest;
                }
            }
        } else {
            *li = first == "if" || first.starts_with("if(");
        }
        let bt = body.text.trim().to_string();
        let block = opens_block(&bt);
        let ((verb, rest), voff, captured) = command_of(&bt);
        let lverb = verb.to_lowercase();
        // one-line `if cond command`
        let mut cmd_stmt: Option<Stmt> = None;
        if lverb == "if" || lverb.starts_with("if(") {
            if !block {
                let cond_cmd = body.sub(voff + 2);
                let plain: String = {
                    // macros as 1 so the condition parses
                    let mut s = cond_cmd.text.clone();
                    for (_, n) in macro_refs(&cond_cmd.text).into_iter().rev() {
                        s = s.replacen(&format!("`{n}'"), "1", 1);
                    }
                    s
                };
                if let Some((cond, _)) = split_if_command(&plain) {
                    // find the command start: the words after the condition
                    let nwords = ferrostat_parser::parser::split_top_level_words(&cond).len();
                    let mut off = 0;
                    let mut seen = 0;
                    let ct = &cond_cmd.text;
                    let mut in_word = false;
                    for (i, c) in ct.char_indices() {
                        if c.is_whitespace() {
                            in_word = false;
                        } else if !in_word {
                            in_word = true;
                            if seen == nwords {
                                off = i;
                                break;
                            }
                            seen += 1;
                        }
                    }
                    if off > 0 {
                        cmd_stmt = Some(cond_cmd.sub(off));
                    }
                }
            }
            check_cmd = false;
        }
        let scope = scopes.last_mut().unwrap();
        let rest_off = voff + verb.len();
        scan_macros(&body, &lverb, &rest, rest_off, scope, &mut global);
        let mut to_check: Vec<(Stmt, String, usize, bool)> = Vec::new();
        if check_cmd {
            to_check.push((body.clone(), verb.clone(), voff, captured));
        }
        if let Some(cs) = cmd_stmt {
            let ((v, r), o, c) = command_of(&cs.text);
            scan_macros(&cs, &v.to_lowercase(), &r, o + v.len(), scopes.last_mut().unwrap(), &mut global);
            to_check.push((cs, v, o, c));
        }
        for (s, v, o, cap) in to_check {
            let lv = v.to_lowercase();
            let lv = lv.trim_end_matches(':');
            if lv.is_empty() || lv == "{" || lv.starts_with(['`', '$', '.', '(', '"', '_', '#']) || lv.starts_with(|c: char| c.is_ascii_digit()) || lv.contains(['`', '$']) {
                continue;
            }
            if cmds.known(lv) || programs.contains(lv) {
                continue;
            }
            let ado = join(&dir, &format!("{lv}.ado"));
            if vfs::exists(&ado) || vfs::exists(&format!("{lv}.ado")) {
                continue;
            }
            let (l, c) = s.pos(o);
            let hint = ferrostat_engine::suggest::closest(lv, cmds.full.iter().map(String::as_str)).map(|h| format!(" (did you mean {h}?)")).unwrap_or_default();
            let (sev, extra) = if cap { (Severity::Warning, " (under capture)") } else { (Severity::Error, "") };
            diags.push(Diag { line: l, col: c, severity: sev, message: format!("unknown command \"{v}\"{extra}{hint}") });
        }
        if block {
            let (bl, bc) = body.pos(bt.len() - 1);
            frames.push(Frame::Brace { line: bl, col: bc, last_if: false });
        }
        // a top-level `exit` ends the do-file
        if lverb == "exit" && frames.is_empty() && !captured && t.split_whitespace().nth(1).is_none_or(|a| a.starts_with(',')) {
            exit_line = Some(line);
            break;
        }
    }
    if exit_line.is_none() {
        if let Some((kind, l, c)) = raw {
            let (sev, what) = match kind {
                Raw::Mata => (Severity::Error, "mata block"),
                Raw::Python => (Severity::Error, "python block"),
                Raw::Input => (Severity::Warning, "input block"),
                Raw::Dict => (Severity::Error, "dictionary block"),
            };
            let closer = if kind == Raw::Dict { "}" } else { "end" };
            diags.push(Diag { line: l, col: c, severity: sev, message: format!("{what} opened here is not closed with \"{closer}\"") });
        }
        for f in &frames {
            match f {
                Frame::Brace { line, col, .. } => diags.push(Diag { line: *line, col: *col, severity: Severity::Error, message: "unbalanced \"{\" opened here".into() }),
                Frame::Program { line, col, name, .. } => diags.push(Diag { line: *line, col: *col, severity: Severity::Error, message: format!("program {name} is not closed with \"end\"") }),
            }
        }
    } else if let Some(el) = exit_line {
        // the rest of the file is never read
        diags.retain(|d| d.line <= el);
    }
    finished.extend(scopes);
    for s in finished {
        s.finish(&global, &mut diags);
    }
    if let Some(el) = exit_line {
        diags.retain(|d| d.line <= el);
    }
    diags.sort_by(|a, b| (a.line, a.col, a.severity).cmp(&(b.line, b.col, b.severity)));
    diags.dedup();
    diags
}

/// Check files; returns the printed report and (errors, warnings, notes).
pub fn check_files(paths: &[String], engine_words: Vec<String>, extra_programs: &[String]) -> (String, (usize, usize, usize)) {
    let mut cmds = Commands::new(engine_words);
    cmds.programs.extend(extra_programs.iter().map(|p| p.to_lowercase()));
    let texts: Vec<Option<String>> = paths.iter().map(|p| vfs::read_to_string(p).ok()).collect();
    // programs defined in any of the files are available to all of them
    for t in texts.iter().flatten() {
        cmds.programs.extend(defined_programs(t).into_iter().map(|p| p.to_lowercase()));
    }
    let mut out = Vec::new();
    let mut tot = (0, 0, 0);
    for (p, t) in paths.iter().zip(&texts) {
        let Some(t) = t else {
            out.push(format!("{p}: error: file not found"));
            tot.0 += 1;
            continue;
        };
        let d = check_text(p, t, &cmds);
        out.extend(d.iter().map(|x| x.format(p)));
        let (e, w, n) = counts(&d);
        tot = (tot.0 + e, tot.1 + w, tot.2 + n);
    }
    let files = if paths.len() == 1 { "1 file".to_string() } else { format!("{} files", paths.len()) };
    out.push(format!("{files} checked: {}", summary(tot.0, tot.1, tot.2)));
    (out.join("\n"), tot)
}

/// `lint [using] file.do [file2.do ...]`: check do-files without running
/// them. Stores r(errors), r(warnings) and r(notes).
pub fn lint_command(me: &mut crate::interpreter::AdoInterpreter, args: &str) -> Result<String, String> {
    let body = args.split(',').next().unwrap_or("").trim();
    let body = body.strip_prefix("using").map(str::trim_start).unwrap_or(body);
    let paths: Vec<String> = ferrostat_parser::parser::split_top_level_words(body).iter().map(|w| crate::macros::unquote(w)).filter(|w| !w.is_empty()).collect();
    if paths.is_empty() {
        return Err(ferrostat_cmd::eval::rc_err(198, "lint using filename.do"));
    }
    let paths: Vec<String> = paths.into_iter().map(|p| if !vfs::exists(&p) && !p.contains('.') { format!("{p}.do") } else { p }).collect();
    if let Some(p) = paths.iter().find(|p| !vfs::exists(p)) {
        return Err(ferrostat_cmd::eval::rc_err(601, format!("file {p} not found")));
    }
    let mut words = me.engine.command_words();
    words.extend(crate::progcmd::INTERP_COMMANDS.iter().map(|s| s.to_string()));
    let programs: Vec<String> = me.engine.store.programs.keys().cloned().collect();
    let (report, (e, w, n)) = check_files(&paths, words, &programs);
    let mut res = ferrostat_cmd::store::Results::new();
    res.set("errors", e as f64);
    res.set("warnings", w as f64);
    res.set("notes", n as f64);
    me.engine.store.r = res;
    Ok(report)
}
