//! Reading raw text data: `infile` (free format, or fixed format through a
//! dictionary) and `infix` (fixed columns, on the command line or in a
//! dictionary).

use ferrostat_core::missing::{self, SYSMISS};
use ferrostat_core::value_label::ValueLabel;
use ferrostat_core::vfs;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn with_ext(path: &str, ext: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    if base.contains('.') { path.to_string() } else { format!("{path}.{ext}") }
}

fn read_text(path: &str) -> Result<String, String> {
    let bytes = vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    Ok(match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => e.into_bytes().iter().map(|&b| b as char).collect(),
    })
}

fn split_lines(text: &str) -> Vec<String> {
    let t = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut v: Vec<String> = t.split('\n').map(str::to_string).collect();
    while v.last().is_some_and(|l| l.trim().is_empty()) {
        v.pop();
    }
    v
}

fn default_format(ty: VarType) -> String {
    match ty {
        VarType::Byte | VarType::Int => "%8.0g".into(),
        VarType::Long => "%12.0g".into(),
        VarType::Float => "%9.0g".into(),
        VarType::Double => "%10.0g".into(),
        VarType::Str(w) => format!("%{w}s"),
        VarType::StrL => "%9s".into(),
    }
}

/// Parse a number as written in raw data (`.`, `.a`..`.z` are missing).
fn parse_num(tok: &str) -> Option<f64> {
    let t = tok.trim();
    if t == "." {
        return Some(SYSMISS);
    }
    if t.len() == 2 && t.starts_with('.') {
        if let Some(m) = missing::missing_letter(t.as_bytes()[1] as char) {
            return Some(m);
        }
    }
    if t.is_empty() || !t.bytes().all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-')) {
        return None;
    }
    if !t.bytes().any(|c| c.is_ascii_digit()) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Store a number in a variable of the given numeric type.
fn num_value(ty: VarType, v: f64) -> Value {
    if missing::is_missing(v) {
        return Value::Double(v);
    }
    match ty {
        VarType::Byte => {
            let t = v.trunc();
            if (-127.0..=missing::MAX_BYTE as f64).contains(&t) { Value::Byte(t as i8) } else { Value::Double(SYSMISS) }
        }
        VarType::Int => {
            let t = v.trunc();
            if (-32767.0..=missing::MAX_INT as f64).contains(&t) { Value::Int(t as i16) } else { Value::Double(SYSMISS) }
        }
        VarType::Long => {
            let t = v.trunc();
            if (-2147483647.0..=missing::MAX_LONG as f64).contains(&t) { Value::Long(t as i32) } else { Value::Double(SYSMISS) }
        }
        VarType::Float => Value::Float(v as f32),
        _ => Value::Double(v),
    }
}

fn str_value(ty: VarType, s: &str) -> Value {
    match ty {
        VarType::Str(w) => Value::Str(truncate(s, w)),
        _ => Value::Str(s.to_string()),
    }
}

fn truncate(s: &str, w: usize) -> String {
    if s.len() <= w {
        return s.to_string();
    }
    let mut end = w;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// One variable being read.
#[derive(Clone, Debug)]
struct VarSpec {
    name: String,
    ty: VarType,
    label: String,
    vlabel: String,
}

fn new_dataset(specs: &[VarSpec]) -> Result<DataSet, String> {
    let mut ds = DataSet::new();
    for s in specs {
        let mut v = Variable::new(s.name.clone(), s.ty);
        v.format = default_format(s.ty);
        v.label = s.label.clone();
        v.value_label_name = s.vlabel.clone();
        ds.add_variable(v).map_err(|_| rc_err(110, format!("{} already defined", s.name)))?;
    }
    Ok(ds)
}

/// Refuse to replace data in memory unless `clear` was given.
fn check_clear(ds: &DataSet, st: &Store, cmd: &Command) -> Result<(), String> {
    if cmd.has_opt("clear", 5) || ds.n_vars() == 0 {
        return Ok(());
    }
    if st.data_changed {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
    }
    Err(rc_err(18, "you must start with an empty dataset"))
}

/// Apply `if`/`in`, install the dataset and report the count.
fn finish(mut new: DataSet, ds: &mut DataSet, st: &mut Store, cmd: &Command, echo: String) -> Result<String, String> {
    if cmd.if_expr.is_some() || cmd.in_range.is_some() {
        let mut sub = cmd.clone();
        sub.main.clear();
        let tmp = Store::new();
        let mask = util::touse(&new, &tmp, &sub)?;
        new.keep_if(&mask)?;
    }
    let n = new.n_obs;
    *ds = new;
    st.data_changed = false;
    st.filename.clear();
    st.tsset = None;
    let msg = format!("({} observation{} read)", n, if n == 1 { "" } else { "s" });
    Ok(if echo.is_empty() { msg } else { format!("{echo}\n{msg}") })
}

fn push_row(ds: &mut DataSet, row: Vec<Value>) {
    for (v, x) in ds.vars.iter_mut().zip(row) {
        v.data.push(x);
    }
    ds.n_obs += 1;
}

/// Expand `x1-x3` into x1 x2 x3 (a name that is not a range is returned as is).
fn expand_names(tok: &str) -> Vec<String> {
    if let Some((a, b)) = tok.split_once('-') {
        let split = |s: &str| {
            let d = s.len() - s.bytes().rev().take_while(u8::is_ascii_digit).count();
            (s[..d].to_string(), s[d..].parse::<usize>().ok())
        };
        let (sa, na) = split(a);
        let (sb, nb) = split(b);
        if let (Some(x), Some(y)) = (na, nb) {
            if !sa.is_empty() && (sb.is_empty() || sb == sa) && x <= y {
                return (x..=y).map(|i| format!("{sa}{i}")).collect();
            }
        }
    }
    vec![tok.to_string()]
}

fn valid_name(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_') && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_') && s.len() <= 32
}

fn paren_num(tok: &str, name: &str, min: usize) -> Option<Option<usize>> {
    let lower = tok.to_lowercase();
    let (head, arg) = match lower.find('(') {
        Some(i) => (&lower[..i], Some(lower[i + 1..].trim_end_matches(')').trim().to_string())),
        None => (lower.as_str(), None),
    };
    if head.len() < min || !name.starts_with(head) {
        return None;
    }
    Some(arg.and_then(|a| a.parse().ok()))
}

// ---------------------------------------------------------------------------
// Free-format tokens
// ---------------------------------------------------------------------------

/// Split free-format data into values: blanks and commas separate values,
/// two commas in a row mean a missing value, quotes group text.
fn free_tokens(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut comma_pending = false;
    while i < c.len() {
        let ch = c[i];
        if ch == ',' {
            if comma_pending {
                out.push(String::new());
            }
            comma_pending = true;
            i += 1;
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        let mut tok = String::new();
        if ch == '"' || ch == '\'' {
            i += 1;
            while i < c.len() && c[i] != ch && c[i] != '\n' {
                tok.push(c[i]);
                i += 1;
            }
            i += 1;
        } else {
            while i < c.len() && !c[i].is_whitespace() && c[i] != ',' {
                tok.push(c[i]);
                i += 1;
            }
        }
        out.push(tok);
        comma_pending = false;
    }
    out
}

/// Read one free-format line field starting at `pos`: returns the text and
/// the position after it (and after one trailing comma).
fn free_field(line: &[char], mut pos: usize) -> (Option<String>, usize) {
    while pos < line.len() && line[pos].is_whitespace() {
        pos += 1;
    }
    if pos >= line.len() {
        return (None, pos);
    }
    let mut tok = String::new();
    let ch = line[pos];
    if ch == ',' {
        return (Some(String::new()), pos + 1);
    }
    if ch == '"' || ch == '\'' {
        pos += 1;
        while pos < line.len() && line[pos] != ch {
            tok.push(line[pos]);
            pos += 1;
        }
        pos += 1;
    } else {
        while pos < line.len() && !line[pos].is_whitespace() && line[pos] != ',' {
            tok.push(line[pos]);
            pos += 1;
        }
    }
    let mut p = pos;
    while p < line.len() && line[p].is_whitespace() {
        p += 1;
    }
    if p < line.len() && line[p] == ',' {
        pos = p + 1;
    }
    (Some(tok), pos)
}

/// Converts tokens to values, collecting the diagnostics printed for
/// unreadable numbers and building `automatic` value labels.
struct Converter {
    automatic: bool,
    labels: Vec<(String, ValueLabel)>,
    notes: Vec<String>,
}

impl Converter {
    fn new(automatic: bool) -> Self {
        Self { automatic, labels: Vec::new(), notes: Vec::new() }
    }

    fn value(&mut self, spec: &VarSpec, tok: Option<&str>, obs: usize) -> Value {
        let Some(tok) = tok else {
            return if spec.ty.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) };
        };
        if spec.ty.is_string() {
            return str_value(spec.ty, tok);
        }
        if tok.trim().is_empty() {
            return Value::Double(SYSMISS);
        }
        if let Some(v) = parse_num(tok) {
            return num_value(spec.ty, v);
        }
        if self.automatic {
            let k = match self.labels.iter().position(|(n, _)| *n == spec.name) {
                Some(k) => k,
                None => {
                    self.labels.push((spec.name.clone(), ValueLabel::new(spec.name.clone())));
                    self.labels.len() - 1
                }
            };
            let vl = &mut self.labels[k].1;
            let code = match vl.value_of(tok) {
                Some(c) => c,
                None => {
                    let c = vl.len() as f64 + 1.0;
                    vl.add(c, tok);
                    c
                }
            };
            return num_value(spec.ty, code);
        }
        self.notes.push(format!("'{tok}' cannot be read as a number for {}[{}]", spec.name, obs + 1));
        Value::Double(SYSMISS)
    }

    fn install(self, ds: &mut DataSet) {
        for (name, vl) in self.labels {
            if let Some(v) = ds.get_var_mut(&name) {
                v.value_label_name = name.clone();
            }
            ds.value_labels.insert(name, vl);
        }
    }
}

// ---------------------------------------------------------------------------
// infile
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum FreeItem {
    Var(VarSpec),
    Skip(usize),
}

fn parse_free_varlist(text: &str, default: VarType) -> Result<Vec<FreeItem>, String> {
    let mut items = Vec::new();
    let mut pending: Option<VarType> = None;
    for tok in text.split_whitespace() {
        if let Some(n) = paren_num(tok, "_skip", 5) {
            items.push(FreeItem::Skip(n.unwrap_or(1)));
            continue;
        }
        let lower = tok.to_lowercase();
        if let Some(t) = VarType::from_type_str(&lower) {
            if t == VarType::StrL {
                return Err(rc_err(198, "strL not allowed"));
            }
            pending = Some(t);
            continue;
        }
        if lower == "str" {
            return Err(rc_err(198, "str must be followed by a width, e.g. str10"));
        }
        let ty = pending.take().unwrap_or(default);
        for name in expand_names(tok) {
            if !valid_name(&name) {
                return Err(rc_err(198, format!("{name} invalid name")));
            }
            items.push(FreeItem::Var(VarSpec { name, ty, label: String::new(), vlabel: String::new() }));
        }
    }
    if pending.is_some() {
        return Err(rc_err(198, "varlist required"));
    }
    Ok(items)
}

fn infile_free(st: &Store, cmd: &Command, path: &str) -> Result<(DataSet, String), String> {
    let items = parse_free_varlist(&cmd.main, st.default_type)?;
    let specs: Vec<VarSpec> = items
        .iter()
        .filter_map(|i| match i {
            FreeItem::Var(v) => Some(v.clone()),
            _ => None,
        })
        .collect();
    if specs.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let mut new = new_dataset(&specs)?;
    let text = read_text(path)?;
    let toks = free_tokens(&text);
    let per_obs: usize = items
        .iter()
        .map(|i| match i {
            FreeItem::Var(_) => 1,
            FreeItem::Skip(n) => *n,
        })
        .sum();
    let mut conv = Converter::new(cmd.has_opt("automatic", 4));
    if let Some(b) = cmd.opt("byvariable", 2) {
        let nobs: usize = b.trim().parse().map_err(|_| rc_err(198, "byvariable() invalid"))?;
        let mut cols: Vec<Vec<Value>> = Vec::new();
        let mut pos = 0;
        for it in &items {
            match it {
                FreeItem::Skip(k) => pos += k * nobs,
                FreeItem::Var(s) => {
                    let col = (0..nobs).map(|i| conv.value(s, toks.get(pos + i).map(String::as_str), i)).collect();
                    cols.push(col);
                    pos += nobs;
                }
            }
        }
        for (v, col) in new.vars.iter_mut().zip(cols) {
            v.data = col;
        }
        new.n_obs = nobs;
    } else {
        let mut pos = 0;
        let mut obs = 0;
        while pos < toks.len() {
            let mut row = Vec::with_capacity(specs.len());
            for it in &items {
                match it {
                    FreeItem::Skip(k) => pos += k,
                    FreeItem::Var(s) => {
                        row.push(conv.value(s, toks.get(pos).map(String::as_str), obs));
                        pos += 1;
                    }
                }
            }
            push_row(&mut new, row);
            obs += 1;
        }
        if per_obs > 0 && toks.len() % per_obs != 0 {
            conv.notes.push("(eof not at end of obs)".to_string());
        }
    }
    let notes = conv.notes.join("\n");
    conv.install(&mut new);
    Ok((new, notes))
}

pub struct Infile;

impl CommandHandler for Infile {
    fn name(&self) -> &'static str {
        "infile"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let using = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        check_clear(ds, st, cmd)?;
        let main = cmd.main.trim();
        let (new, echo) = if main.is_empty() {
            let path = with_ext(&using, "dct");
            let text = read_text(&path)?;
            let d = Dictionary::parse(&text, DictKind::Infile)?;
            let echo = d.echo.join("\n");
            (d.read(st, cmd, Some(&text), &path)?, echo)
        } else if main.eq_ignore_ascii_case("dictionary") {
            return Err(rc_err(198, "dictionary must be followed by { on the same line"));
        } else {
            infile_free(st, cmd, &with_ext(&using, "raw"))?
        };
        finish(new, ds, st, cmd, echo)
    }
}

// ---------------------------------------------------------------------------
// Dictionaries (infile and infix)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DictKind {
    Infile,
    Infix,
}

/// Input format `%[w][.d]{f,g,e,s,S}`.
#[derive(Clone, Copy, Debug)]
struct InFmt {
    width: Option<usize>,
    dec: Option<usize>,
    keep_blanks: bool,
}

fn parse_infmt(tok: &str) -> Option<InFmt> {
    let t = tok.strip_prefix('%')?;
    let t = t.trim_start_matches('-');
    let kind = t.chars().last()?;
    let body = &t[..t.len() - kind.len_utf8()];
    let (w, d) = match body.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (body, None),
    };
    let width = if w.is_empty() { None } else { Some(w.parse().ok()?) };
    let dec = match d {
        Some(x) => Some(x.parse().ok()?),
        None => None,
    };
    match kind {
        'f' | 'g' | 'e' => Some(InFmt { width, dec, keep_blanks: false }),
        's' => Some(InFmt { width, dec: None, keep_blanks: false }),
        'S' => Some(InFmt { width, dec: None, keep_blanks: true }),
        _ => None,
    }
}

#[derive(Clone, Debug)]
enum DictItem {
    /// infile: a variable read at the current position.
    Var { spec: VarSpec, fmt: Option<InFmt> },
    /// infix: a variable read from fixed columns of a line of the observation.
    Cols { spec: VarSpec, line: usize, from: usize, to: usize },
    Column(usize),
    Skip(usize),
    Line(usize),
    Newline(usize),
}

#[derive(Debug)]
pub struct Dictionary {
    using: Option<String>,
    lines: usize,
    first_line: Option<usize>,
    items: Vec<DictItem>,
    /// Line of the source text where inline data begin.
    data_start: usize,
    echo: Vec<String>,
}

/// Remove `/* */` and `//` comments and `*` comment lines, keeping line
/// structure.
fn strip_comments(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    for l in lines {
        let c: Vec<char> = l.chars().collect();
        let mut s = String::new();
        let mut i = 0;
        let mut in_str = false;
        while i < c.len() {
            if depth > 0 {
                if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            if c[i] == '"' {
                in_str = !in_str;
            }
            if !in_str && c[i] == '/' && c.get(i + 1) == Some(&'*') {
                depth += 1;
                i += 2;
                s.push(' ');
                continue;
            }
            if !in_str && c[i] == '/' && c.get(i + 1) == Some(&'/') {
                break;
            }
            s.push(c[i]);
            i += 1;
        }
        if s.trim_start().starts_with('*') {
            s.clear();
        }
        out.push(s);
    }
    out
}

/// Words of dictionary text, with quoted strings kept together.
fn dict_words(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_whitespace() {
            i += 1;
            continue;
        }
        let mut w = String::new();
        if c[i] == '"' {
            w.push('"');
            i += 1;
            while i < c.len() && c[i] != '"' {
                w.push(c[i]);
                i += 1;
            }
            w.push('"');
            i += 1;
        } else if c[i] == '/' {
            w.push('/');
            i += 1;
        } else {
            while i < c.len() && !c[i].is_whitespace() && !(c[i] == '"' && !w.is_empty()) {
                w.push(c[i]);
                i += 1;
            }
        }
        out.push(w);
    }
    out
}

/// Parse a column range `[l:]a[-b]`.
fn parse_cols(tok: &str) -> Option<(Option<usize>, usize, usize)> {
    let (line, rest) = match tok.split_once(':') {
        Some((l, r)) => (Some(l.trim().parse().ok()?), r),
        None => (None, tok),
    };
    let (a, b) = match rest.split_once('-') {
        Some((a, b)) => (a.trim().parse().ok()?, b.trim().parse().ok()?),
        None => {
            let a: usize = rest.trim().parse().ok()?;
            (a, a)
        }
    };
    (a >= 1 && b >= a).then_some((line, a, b))
}

impl Dictionary {
    /// Parse a dictionary file (the dictionary text possibly followed by data).
    pub fn parse(text: &str, kind: DictKind) -> Result<Self, String> {
        let raw = split_lines(text);
        let clean = strip_comments(&raw);
        let start = clean
            .iter()
            .position(|l| l.split_whitespace().any(|w| w.eq_ignore_ascii_case("dictionary") || w.eq_ignore_ascii_case("dict")))
            .ok_or_else(|| rc_err(610, "file is not a dictionary"))?;
        let open = (start..clean.len()).find(|&i| clean[i].contains('{')).ok_or_else(|| rc_err(198, "dictionary invalid: { not found"))?;
        let mut header = String::new();
        for (i, l) in clean.iter().enumerate().take(open + 1).skip(start) {
            let l = if i == open { &l[..l.find('{').unwrap()] } else { l.as_str() };
            header.push_str(l);
            header.push(' ');
        }
        let mut body = Vec::new();
        let mut close = None;
        let after_open = clean[open][clean[open].find('{').unwrap() + 1..].to_string();
        if !after_open.trim().is_empty() {
            body.push(after_open);
        }
        for (i, l) in clean.iter().enumerate().skip(open + 1) {
            if let Some(p) = l.find('}') {
                body.push(l[..p].to_string());
                close = Some(i);
                break;
            }
            body.push(l.clone());
        }
        let close = close.ok_or_else(|| rc_err(198, "dictionary invalid: } not found"))?;
        let mut d = Self::from_parts(&header, &body, kind)?;
        d.data_start = close + 1;
        d.echo = raw[start..=close].iter().map(|l| l.trim_end().to_string()).collect();
        Ok(d)
    }

    /// Build from the header (text before `{`) and the body lines.
    pub fn from_parts(header: &str, body: &[String], kind: DictKind) -> Result<Self, String> {
        let hw = dict_words(header);
        let mut using = None;
        if let Some(p) = hw.iter().position(|w| w.eq_ignore_ascii_case("using")) {
            let f = hw.get(p + 1).ok_or_else(|| rc_err(198, "using requires a filename"))?;
            using = Some(util::unquote(f.trim_end_matches(',')));
        }
        let mut d = Dictionary { using, lines: 1, first_line: None, items: Vec::new(), data_start: 0, echo: Vec::new() };
        let clean = strip_comments(body);
        let words: Vec<String> = clean.iter().flat_map(|l| dict_words(l)).collect();
        match kind {
            DictKind::Infile => d.parse_infile_items(&words)?,
            DictKind::Infix => d.items = d.parse_infix_items(&words, 1)?,
        }
        Ok(d)
    }

    /// Common `_lines(#)` / `_firstlineoffile(#)` directives.
    fn directive(&mut self, w: &str) -> Result<bool, String> {
        if let Some(n) = paren_num(w, "_lines", 6) {
            self.lines = n.ok_or_else(|| rc_err(198, "_lines() invalid"))?.max(1);
            return Ok(true);
        }
        if let Some(n) = paren_num(w, "_firstlineoffile", 6) {
            self.first_line = Some(n.ok_or_else(|| rc_err(198, "_firstlineoffile() invalid"))?.max(1));
            return Ok(true);
        }
        Ok(false)
    }

    fn parse_infile_items(&mut self, words: &[String]) -> Result<(), String> {
        let mut i = 0;
        while i < words.len() {
            let w = &words[i];
            i += 1;
            if self.directive(w)? {
                continue;
            }
            if let Some(n) = paren_num(w, "_column", 4) {
                self.items.push(DictItem::Column(n.ok_or_else(|| rc_err(198, "_column() invalid"))?));
                continue;
            }
            if let Some(n) = paren_num(w, "_skip", 5) {
                self.items.push(DictItem::Skip(n.unwrap_or(1)));
                continue;
            }
            if let Some(n) = paren_num(w, "_newline", 4) {
                self.items.push(DictItem::Newline(n.unwrap_or(1)));
                continue;
            }
            if let Some(n) = paren_num(w, "_line", 5) {
                self.items.push(DictItem::Line(n.ok_or_else(|| rc_err(198, "_line() invalid"))?.max(1)));
                continue;
            }
            let lower = w.to_lowercase();
            let mut ty = VarType::Float;
            let mut name = w.clone();
            if let Some(t) = VarType::from_type_str(&lower) {
                ty = t;
                name = words.get(i).cloned().ok_or_else(|| rc_err(198, "variable name expected"))?;
                i += 1;
            }
            let mut vlabel = String::new();
            if let Some((n, l)) = name.clone().split_once(':') {
                name = n.to_string();
                vlabel = l.to_string();
            }
            if !valid_name(&name) {
                return Err(rc_err(198, format!("{name} invalid name")));
            }
            if vlabel.is_empty() {
                if let Some(l) = words.get(i).and_then(|x| x.strip_prefix(':')) {
                    vlabel = if l.is_empty() {
                        i += 1;
                        words.get(i).cloned().unwrap_or_default()
                    } else {
                        l.to_string()
                    };
                    i += 1;
                }
            }
            let mut fmt = None;
            if let Some(f) = words.get(i).and_then(|x| parse_infmt(x)) {
                fmt = Some(f);
                i += 1;
            }
            let mut label = String::new();
            if let Some(l) = words.get(i).filter(|x| x.starts_with('"')) {
                label = util::unquote(l);
                i += 1;
            }
            self.items.push(DictItem::Var { spec: VarSpec { name, ty, label, vlabel }, fmt });
        }
        Ok(())
    }

    /// infix specifications: `[#lines] [#:] [type] name [#:]a[-b] ... [/]`.
    fn parse_infix_items(&mut self, words: &[String], default_line: usize) -> Result<Vec<DictItem>, String> {
        let mut items = Vec::new();
        let mut line = default_line;
        let mut i = 0;
        let mut pending: Option<String> = None;
        while i < words.len() {
            let w = words[i].clone();
            i += 1;
            if self.directive(&w)? {
                continue;
            }
            if w == "/" {
                line += 1;
                continue;
            }
            if let Some(n) = w.strip_suffix(':').and_then(|x| x.parse::<usize>().ok()) {
                line = n.max(1);
                continue;
            }
            if let Some(n) = w.strip_suffix("lines").and_then(|x| x.parse::<usize>().ok()) {
                self.lines = n.max(1);
                continue;
            }
            if let Ok(n) = w.parse::<usize>() {
                if words.get(i).is_some_and(|x| x.eq_ignore_ascii_case("lines") || x.eq_ignore_ascii_case("line")) {
                    self.lines = n.max(1);
                    i += 1;
                    continue;
                }
            }
            let lower = w.to_lowercase();
            if lower == "str" || VarType::from_type_str(&lower).is_some_and(|t| t != VarType::StrL) {
                pending = Some(lower);
                continue;
            }
            if !valid_name(&w) {
                return Err(rc_err(198, format!("{w} invalid name")));
            }
            let c = words.get(i).ok_or_else(|| rc_err(198, format!("column specification for {w} required")))?;
            i += 1;
            let (l, from, to) = parse_cols(c).ok_or_else(|| rc_err(198, format!("{c} invalid column specification")))?;
            let ty = match pending.take() {
                Some(t) if t == "str" => VarType::Str((to - from + 1).min(2045)),
                Some(t) => VarType::from_type_str(&t).unwrap(),
                None => VarType::Float,
            };
            items.push(DictItem::Cols {
                spec: VarSpec { name: w, ty, label: String::new(), vlabel: String::new() },
                line: l.unwrap_or(line),
                from,
                to,
            });
        }
        Ok(items)
    }

    fn specs(&self, default: VarType) -> Vec<VarSpec> {
        self.items
            .iter()
            .filter_map(|it| match it {
                DictItem::Var { spec, .. } | DictItem::Cols { spec, .. } => {
                    let mut s = spec.clone();
                    // an unspecified numeric type follows `set type`
                    if s.ty == VarType::Float {
                        s.ty = default;
                    }
                    Some(s)
                }
                _ => None,
            })
            .collect()
    }

    /// Read the data described by the dictionary. `own_text` is the text of
    /// the file holding the dictionary (used when data follow it).
    pub fn read(&self, st: &Store, cmd: &Command, own_text: Option<&str>, own_path: &str) -> Result<DataSet, String> {
        let specs = self.specs(st.default_type);
        if specs.is_empty() {
            return Err(rc_err(198, "no variables defined"));
        }
        let mut new = new_dataset(&specs)?;
        let data_file = cmd.opt("using", 5).map(util::unquote).or_else(|| self.using.clone());
        let (lines, mut start) = match data_file {
            Some(f) => {
                let p = with_ext(&f, "raw");
                (split_lines(&read_text(&p)?), 0)
            }
            None => match own_text {
                Some(t) => (split_lines(t), self.data_start),
                None => return Err(rc_err(198, format!("no data file specified for dictionary {own_path}"))),
            },
        };
        if let Some(f) = self.first_line {
            start = f - 1;
        }
        let chars: Vec<Vec<char>> = lines.iter().map(|l| l.chars().collect()).collect();
        let mut conv = Converter::new(cmd.has_opt("automatic", 4));
        let mut obs = 0;
        let mut l0 = start;
        let types: Vec<VarType> = specs.iter().map(|s| s.ty).collect();
        while l0 < chars.len() {
            let mut row = Vec::with_capacity(specs.len());
            let mut used = self.lines;
            let mut cur = l0;
            let mut pos = 0usize;
            let mut k = 0;
            for it in &self.items {
                match it {
                    DictItem::Column(c) => pos = c.saturating_sub(1),
                    DictItem::Skip(n) => pos += n,
                    DictItem::Line(n) => {
                        cur = l0 + n - 1;
                        pos = 0;
                    }
                    DictItem::Newline(n) => {
                        cur += n;
                        pos = 0;
                    }
                    DictItem::Cols { spec, line, from, to } => {
                        used = used.max(*line);
                        let text = chars.get(l0 + line - 1).map(|l| {
                            let a = (from - 1).min(l.len());
                            let b = (*to).min(l.len());
                            l[a..b].iter().collect::<String>()
                        });
                        let spec = VarSpec { ty: types[k], ..spec.clone() };
                        let t = text.as_deref().map(str::trim);
                        let v = if spec.ty.is_string() { str_value(spec.ty, t.unwrap_or("")) } else { fixed_num(&spec, t, None) };
                        row.push(v);
                        k += 1;
                    }
                    DictItem::Var { spec, fmt } => {
                        used = used.max(cur - l0 + 1);
                        let line: &[char] = chars.get(cur).map(Vec::as_slice).unwrap_or(&[]);
                        let spec = VarSpec { ty: types[k], ..spec.clone() };
                        let width = fmt.and_then(|f| f.width);
                        let tok = match width {
                            Some(w) => {
                                let a = pos.min(line.len());
                                let b = (pos + w).min(line.len());
                                pos += w;
                                let s: String = line[a..b].iter().collect();
                                let keep = fmt.is_some_and(|f| f.keep_blanks);
                                Some(if keep { s } else { s.trim().to_string() })
                            }
                            None => {
                                let (t, p) = free_field(line, pos);
                                pos = p;
                                t
                            }
                        };
                        let v = if spec.ty.is_string() || width.is_none() {
                            if !spec.ty.is_string() && fmt.and_then(|f| f.dec).is_some() {
                                fixed_num(&spec, tok.as_deref(), fmt.and_then(|f| f.dec))
                            } else {
                                conv.value(&spec, tok.as_deref(), obs)
                            }
                        } else {
                            match tok.as_deref().map(str::trim) {
                                Some(t) if !t.is_empty() && parse_num(t).is_none() => conv.value(&spec, Some(t), obs),
                                t => fixed_num(&spec, t, fmt.and_then(|f| f.dec)),
                            }
                        };
                        row.push(v);
                        k += 1;
                    }
                }
            }
            push_row(&mut new, row);
            obs += 1;
            l0 += used.max(1);
        }
        conv.install(&mut new);
        Ok(new)
    }
}

/// A number read from fixed columns; blank or unreadable text is missing.
/// With `dec` digits and no decimal point in the text, the decimal point
/// is implied.
fn fixed_num(spec: &VarSpec, text: Option<&str>, dec: Option<usize>) -> Value {
    let Some(t) = text.map(str::trim).filter(|t| !t.is_empty()) else {
        return Value::Double(SYSMISS);
    };
    match parse_num(t) {
        Some(mut v) => {
            if let Some(d) = dec {
                if !t.contains('.') && !missing::is_missing(v) && !t.contains(['e', 'E']) {
                    v /= 10f64.powi(d as i32);
                }
            }
            num_value(spec.ty, v)
        }
        None => Value::Double(SYSMISS),
    }
}

// ---------------------------------------------------------------------------
// infix
// ---------------------------------------------------------------------------

pub struct Infix;

impl CommandHandler for Infix {
    fn name(&self) -> &'static str {
        "infix"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let using = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        check_clear(ds, st, cmd)?;
        let main = cmd.main.trim();
        let (new, echo) = if main.is_empty() {
            let path = with_ext(&using, "dct");
            let text = read_text(&path)?;
            let d = Dictionary::parse(&text, DictKind::Infix)?;
            let echo = d.echo.join("\n");
            (d.read(st, cmd, Some(&text), &path)?, echo)
        } else {
            let mut d = Dictionary {
                using: Some(using),
                lines: 1,
                first_line: None,
                items: Vec::new(),
                data_start: 0,
                echo: Vec::new(),
            };
            let words = dict_words(main);
            d.items = d.parse_infix_items(&words, 1)?;
            (d.read(st, cmd, None, "")?, String::new())
        };
        finish(new, ds, st, cmd, echo)
    }
}

/// Run an inline dictionary from a do-file:
/// `infix dictionary [using file] [, options] { ... }` (or `infile`).
/// `cmd` is the parsed header line and `body` the lines between the braces.
pub fn inline_dictionary(ds: &mut DataSet, st: &mut Store, cmd: &Command, body: &[String]) -> Result<String, String> {
    let kind = if cmd.verb.starts_with("infix") { DictKind::Infix } else { DictKind::Infile };
    check_clear(ds, st, cmd)?;
    let mut d = Dictionary::from_parts("", body, kind)?;
    d.using = cmd.using.clone();
    let new = d.read(st, cmd, None, "")?;
    finish(new, ds, st, cmd, String::new())
}

/// Is this do-file line the header of an inline dictionary block?
pub fn is_dictionary_header(line: &str) -> bool {
    let mut w = line.split_whitespace();
    let first = w.next().unwrap_or("").to_lowercase();
    let second = w.next().unwrap_or("").to_lowercase();
    (first == "infix" || (first.len() >= 3 && "infile".starts_with(&first))) && (second == "dictionary" || second == "dict")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens() {
        assert_eq!(free_tokens("1 2,3,,4 \"a b\"\n5"), ["1", "2", "3", "", "4", "a b", "5"]);
        assert_eq!(expand_names("x1-x3"), ["x1", "x2", "x3"]);
        assert_eq!(parse_cols("2:4-17"), Some((Some(2), 4, 17)));
        assert!(parse_num("abc").is_none());
        assert_eq!(parse_num("-1.5e2"), Some(-150.0));
    }
}
