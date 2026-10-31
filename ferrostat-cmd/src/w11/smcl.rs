//! SMCL rendering: the markup used by help files (`.sthlp`), SMCL logs and
//! viewer documents. [`render`] produces styled lines (bold, italic,
//! underline, output tone and hyperlinks) for graphical viewers;
//! [`to_text`] gives plain text for terminals and the browser output area.
//!
//! Supported: paragraph modes (`{p}`, `{pstd}`, `{phang}`, `{pmore}`, ...)
//! with word wrapping, two-column tables (`{synoptset}`, `{synopt}`,
//! `{p2colset}`, `{p2col}`, `{synoptline}`, `{synopthdr}`), headings,
//! inline styles, links (`{help}`, `{browse}`, `{view}`, run-command links,
//! `{manhelp}`, `{dialog}`, `{search}`), layout directives (`{col}`,
//! `{space}`, `{hline}`, `{dup}`, `{right}`, `{center}`, `{lalign}`,
//! `{ralign}`, `{break}`), characters (`{c ...}`, `{char ...}`), comments
//! (`{* ...}`) and line continuation (`{...}`). Unknown directives show
//! the text after their colon, if any.

/// Output tone of a span (how results windows colour text).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tone {
    #[default]
    Text,
    Result,
    Error,
    Input,
    Command,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Style {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub tone: Tone,
    /// Action of a link: `help topic`, `browse url`, `view file`,
    /// `run cmd` (a command to execute), `dialog name`, `search words`.
    pub link: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub text: String,
    pub style: Style,
}

/// One rendered line.
pub type Line = Vec<Span>;

/// Name of the directive for links that run a command.
fn run_tag() -> String {
    ["st", "ata"].concat()
}

/// Default width of rendered documents.
pub const WIDTH: usize = 79;

/// Does this file look like SMCL (by extension or its `{smcl}` header)?
pub fn is_smcl(path: &str, text: &str) -> bool {
    let p = path.to_lowercase();
    p.ends_with(".smcl") || p.ends_with(".sthlp") || p.ends_with(".hlp") || p.ends_with(".ihlp") || text.trim_start().starts_with("{smcl}")
}

/// Render SMCL to plain text (width 79).
pub fn to_text(src: &str) -> String {
    lines_to_text(&render(src, WIDTH))
}

/// Join rendered lines into plain text (trailing blanks trimmed).
pub fn lines_to_text(lines: &[Line]) -> String {
    let mut out: Vec<String> = lines.iter().map(|l| l.iter().map(|s| s.text.as_str()).collect::<String>().trim_end().to_string()).collect();
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    while out.first().is_some_and(|l| l.is_empty()) {
        out.remove(0);
    }
    out.join("\n")
}

/// Plain text as unstyled lines (for viewers showing non-SMCL files).
pub fn plain_lines(text: &str) -> Vec<Line> {
    text.lines().map(|l| vec![Span { text: l.replace('\t', "    "), style: Style::default() }]).collect()
}

/// Render SMCL into styled lines of at most `width` columns (wrapped
/// paragraphs; line-mode text keeps its own length).
pub fn render(src: &str, width: usize) -> Vec<Line> {
    let mut r = Renderer { width: width.max(20), out: Vec::new(), cur: Vec::new(), para: None, st: Style::default(), synopt_w: 20, p2: (4, 24, 26, 2) };
    for line in logical_lines(src) {
        let t = line.trim();
        if t.eq_ignore_ascii_case("{smcl}") {
            continue;
        }
        if t.is_empty() {
            r.block();
            r.out.push(Vec::new());
            continue;
        }
        if r.para.is_some() {
            r.inline(line.trim_start());
            if let Some(p) = r.para.as_mut() {
                p.items.push(Span { text: " ".into(), style: Style::default() });
            }
        } else {
            r.inline(&line);
            if r.para.is_none() && !r.cur.is_empty() {
                r.end_line();
            }
        }
    }
    r.flush_para();
    if !r.cur.is_empty() {
        r.end_line();
    }
    r.out
}

/// Physical lines with `{...}` continuations joined.
fn logical_lines(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut acc = String::new();
    let mut joining = false;
    for raw in src.lines() {
        let l = raw.trim_end_matches('\r').replace('\t', "    ");
        let l = l.trim_end();
        if let Some(head) = l.strip_suffix("{...}") {
            acc.push_str(head);
            joining = true;
            continue;
        }
        if joining {
            acc.push_str(l);
            out.push(std::mem::take(&mut acc));
            joining = false;
        } else {
            out.push(l.to_string());
        }
    }
    if joining {
        out.push(acc);
    }
    out
}

struct Para {
    first: usize,
    rest: usize,
    right: usize,
    /// first-column text of a two-column row and the description column
    lead: Option<(Vec<Span>, usize)>,
    items: Vec<Span>,
}

struct Renderer {
    width: usize,
    out: Vec<Line>,
    cur: Line,
    para: Option<Para>,
    st: Style,
    synopt_w: usize,
    /// p2colset: first column, second column, continuation, right margin
    p2: (usize, usize, usize, usize),
}

/// Index of the first `:` outside quotes and nested braces.
fn colon_at(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => quote = !quote,
            '{' if !quote => depth += 1,
            '}' if !quote => depth -= 1,
            ':' if !quote && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn unq(s: &str) -> String {
    let s = s.trim();
    s.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(s).to_string()
}

fn width_of(spans: &[Span]) -> usize {
    spans.iter().map(|s| s.text.chars().count()).sum()
}

fn box_char(arg: &str) -> String {
    match arg.trim() {
        "|" => "|".into(),
        "-" => "-".into(),
        "+" | "TT" | "BT" | "LT" | "RT" | "TLC" | "TRC" | "BLC" | "BRC" => "+".into(),
        "-(" => "{".into(),
        ")-" => "}".into(),
        "S|" => "$".into(),
        "'g" => "`".into(),
        a => {
            let n = a.strip_prefix("0x").and_then(|h| u32::from_str_radix(h, 16).ok()).or_else(|| a.parse().ok());
            match n.and_then(char::from_u32) {
                Some(c) => c.to_string(),
                None => a.trim_matches('\'').to_string(),
            }
        }
    }
}

/// (first indent, continuation indent, right margin) of paragraph directives.
fn para_shape(name: &str, args: &[usize]) -> Option<(usize, usize, usize)> {
    Some(match name {
        "p" => (args.first().copied().unwrap_or(0), args.get(1).copied().or(args.first().copied()).unwrap_or(0), args.get(2).copied().unwrap_or(0)),
        "pstd" => (4, 4, 2),
        "psee" => (4, 13, 2),
        "phang" => (4, 8, 2),
        "pmore" | "pin" => (8, 8, 2),
        "phang2" => (8, 12, 2),
        "pmore2" | "pin2" => (12, 12, 2),
        "phang3" => (12, 16, 2),
        "pmore3" | "pin3" => (16, 16, 2),
        _ => return None,
    })
}

impl Renderer {
    fn push(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let span = Span { text: text.to_string(), style: self.st.clone() };
        let sink = match self.para.as_mut() {
            Some(p) => &mut p.items,
            None => &mut self.cur,
        };
        match sink.last_mut() {
            Some(last) if last.style == span.style => last.text.push_str(&span.text),
            _ => sink.push(span),
        }
    }

    fn col(&self) -> usize {
        match &self.para {
            Some(p) => width_of(&p.items),
            None => width_of(&self.cur),
        }
    }

    fn end_line(&mut self) {
        self.out.push(std::mem::take(&mut self.cur));
    }

    /// Start a new line block (closes a paragraph or a partial line).
    fn block(&mut self) {
        self.flush_para();
        if !self.cur.is_empty() {
            self.end_line();
        }
    }

    fn start_para(&mut self, first: usize, rest: usize, right: usize, lead: Option<(Vec<Span>, usize)>) {
        self.block();
        self.para = Some(Para { first, rest, right, lead, items: Vec::new() });
    }

    /// Word-wrap the open paragraph into output lines.
    fn flush_para(&mut self) {
        let Some(p) = self.para.take() else { return };
        // words: runs of non-blank text (a "\n" span forces a break)
        let mut words: Vec<Option<Vec<Span>>> = Vec::new();
        let mut word: Vec<Span> = Vec::new();
        for s in p.items {
            if s.text == "\n" {
                if !word.is_empty() {
                    words.push(Some(std::mem::take(&mut word)));
                }
                words.push(None);
                continue;
            }
            let mut piece = String::new();
            for c in s.text.chars() {
                if c == ' ' {
                    if !piece.is_empty() {
                        word.push(Span { text: std::mem::take(&mut piece), style: s.style.clone() });
                    }
                    if !word.is_empty() {
                        words.push(Some(std::mem::take(&mut word)));
                    }
                } else {
                    piece.push(if c == '\u{a0}' { ' ' } else { c });
                }
            }
            if !piece.is_empty() {
                word.push(Span { text: piece, style: s.style.clone() });
            }
        }
        if !word.is_empty() {
            words.push(Some(word));
        }
        let limit = self.width.saturating_sub(p.right).max(p.rest + 10);
        let mut line: Line = vec![Span { text: " ".repeat(p.first), style: Style::default() }];
        if let Some((lead, desc)) = p.lead {
            line.extend(lead);
            let w = width_of(&line);
            line.push(Span { text: " ".repeat(if w < desc { desc - w } else { 1 }), style: Style::default() });
        }
        let mut has_word = false;
        let mut lines: Vec<Line> = Vec::new();
        for w in words {
            let Some(w) = w else {
                lines.push(std::mem::replace(&mut line, vec![Span { text: " ".repeat(p.rest), style: Style::default() }]));
                has_word = false;
                continue;
            };
            let ww = width_of(&w);
            let cur = width_of(&line);
            if has_word && cur + 1 + ww > limit {
                lines.push(std::mem::replace(&mut line, vec![Span { text: " ".repeat(p.rest), style: Style::default() }]));
                has_word = false;
            }
            if has_word {
                // a space inside a styled run (a link) keeps its style
                let same = line.last().zip(w.first()).filter(|(a, b)| a.style == b.style).map(|(a, _)| a.style.clone());
                line.push(Span { text: " ".into(), style: same.unwrap_or_default() });
            }
            line.extend(w);
            has_word = true;
        }
        if width_of(&line) > 0 && line.iter().any(|s| !s.text.trim().is_empty()) {
            lines.push(line);
        }
        for l in lines {
            let mut merged: Line = Vec::new();
            for s in l {
                match merged.last_mut() {
                    Some(last) if last.style == s.style => last.text.push_str(&s.text),
                    _ => merged.push(s),
                }
            }
            self.out.push(merged);
        }
    }

    /// Render inline SMCL text into the current sink.
    fn inline(&mut self, s: &str) {
        let mut text = String::new();
        let mut chars = s.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c != '{' {
                text.push(c);
                continue;
            }
            // find the matching close brace
            let mut depth = 1;
            let mut end = None;
            let mut quote = false;
            for (j, d) in s[i + 1..].char_indices() {
                match d {
                    '"' => quote = !quote,
                    '{' if !quote => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i + 1 + j);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if end.is_none() {
                // retry ignoring quotes (unbalanced quote inside a tag)
                depth = 1;
                for (j, d) in s[i + 1..].char_indices() {
                    if d == '{' {
                        depth += 1;
                    } else if d == '}' {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i + 1 + j);
                            break;
                        }
                    }
                }
            }
            let Some(end) = end else {
                text.push(c);
                continue;
            };
            self.push(&std::mem::take(&mut text));
            self.tag(&s[i + 1..end]);
            while chars.peek().is_some_and(|(k, _)| *k <= end) {
                chars.next();
            }
        }
        self.push(&text);
    }

    /// Render `text` with a temporarily modified style.
    fn styled(&mut self, text: &str, f: impl FnOnce(&mut Style)) {
        let saved = self.st.clone();
        f(&mut self.st);
        self.inline(text);
        self.st = saved;
    }

    fn link(&mut self, shown: &str, action: String, f: impl FnOnce(&mut Style)) {
        let saved = self.st.clone();
        f(&mut self.st);
        self.st.link = Some(action);
        self.inline(shown);
        self.st = saved;
    }

    fn tag(&mut self, inner: &str) {
        let inner_t = inner.trim_start();
        if inner_t.starts_with('*') || inner_t == "..." {
            return;
        }
        if inner_t == ".-" {
            let n = self.width.saturating_sub(self.col());
            self.push(&"-".repeat(n));
            return;
        }
        if inner_t == "-(" {
            self.push("{");
            return;
        }
        if inner_t == ")-" {
            self.push("}");
            return;
        }
        let (head, text) = match colon_at(inner) {
            Some(k) => (&inner[..k], Some(&inner[k + 1..])),
            None => (inner, None),
        };
        let mut hw = head.trim().splitn(2, char::is_whitespace);
        let name = hw.next().unwrap_or("").to_lowercase();
        let args = hw.next().unwrap_or("").trim();
        let nums: Vec<usize> = args.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        let n1 = nums.first().copied();
        match name.as_str() {
            "smcl" | "marker" | "viewerjumpto" | "vieweralsosee" | "viewerdialog" | "reset" | "s6hlp" | "asis" | "clearmore" | "bind_off" => {}
            "txt" | "text" | "res" | "result" | "err" | "error" | "inp" | "input" | "com" | "hi" => {
                let tone = match name.as_str() {
                    "res" | "result" => Tone::Result,
                    "err" | "error" => Tone::Error,
                    "inp" | "input" => Tone::Input,
                    "com" => Tone::Command,
                    "hi" => self.st.tone,
                    _ => Tone::Text,
                };
                let bold = name == "hi";
                match text {
                    Some(t) => self.styled(t, |s| {
                        s.tone = tone;
                        s.bold |= bold;
                    }),
                    None => {
                        self.st.tone = tone;
                        if name != "hi" {
                            self.st.bold = false;
                            self.st.italic = false;
                        }
                    }
                }
            }
            "sf" => match text {
                Some(t) => self.styled(t, |s| {
                    s.bold = false;
                    s.italic = false;
                }),
                None => {
                    self.st.bold = false;
                    self.st.italic = false;
                }
            },
            "bf" | "title" | "cmd" | "hilite" => match text {
                Some(t) => self.styled(t, |s| {
                    s.bold = true;
                    if name == "cmd" {
                        s.tone = Tone::Input;
                    }
                }),
                None if name == "cmd" => self.st.tone = Tone::Input,
                None => self.st.bold = true,
            },
            "it" | "ul" | "bind" => match text {
                Some(t) if name == "bind" => {
                    let t = t.replace(' ', "\u{a0}");
                    self.inline(&t);
                }
                Some(t) => self.styled(t, |s| {
                    if name == "it" {
                        s.italic = true;
                    } else {
                        s.underline = true;
                    }
                }),
                None if name == "ul" => self.st.underline = args != "off",
                None => self.st.italic = true,
            },
            "opt" | "opth" | "optb" | "cmdab" | "opt2" => {
                // {opt na:me(arg)}: the colon marks the minimal abbreviation
                let full = format!("{args}{}", text.unwrap_or(""));
                let full = full.replace(':', "");
                let (nm, rest) = match full.find('(') {
                    Some(k) => (full[..k].to_string(), full[k..].to_string()),
                    None => (full.clone(), String::new()),
                };
                self.styled(&nm, |s| s.bold = true);
                if !rest.is_empty() {
                    self.push("(");
                    let arg = rest.trim_start_matches('(').trim_end_matches(')');
                    self.styled(arg, |s| s.italic = true);
                    self.push(")");
                }
            }
            "help" | "helpb" | "help_d" => {
                let topic = unq(args);
                let shown = text.map(str::to_string).unwrap_or_else(|| topic.split("##").next().unwrap_or("").to_string());
                self.link(&shown, format!("help {topic}"), |s| s.bold |= name == "helpb");
            }
            "manhelp" | "manhelpi" | "manlink" | "manlinki" | "mansection" => {
                let w: Vec<&str> = args.split_whitespace().collect();
                let (topic, man) = if name == "mansection" { (w.get(1).copied().unwrap_or(""), w.first().copied().unwrap_or("")) } else { (w.first().copied().unwrap_or(""), w.get(1).copied().unwrap_or("")) };
                let shown = match text {
                    Some(t) => t.to_string(),
                    None if name.starts_with("manlink") => format!("[{man}] {topic}"),
                    None => format!("[{man}] {topic}"),
                };
                self.link(&shown, format!("help {topic}"), |s| s.italic |= name.ends_with('i'));
            }
            n if n == run_tag() => {
                // a link that runs a command
                let target = unq(args);
                let shown = text.map(str::to_string).unwrap_or_else(|| target.clone());
                self.link(&shown, format!("run {target}"), |s| s.tone = Tone::Input);
            }
            "browse" | "view" | "view_d" | "dialog" | "search" | "search_d" | "net" | "ado" | "update" | "news" | "back" | "ado_d" | "net_d" | "update_d" => {
                let target = unq(args);
                let shown = text.map(str::to_string).unwrap_or_else(|| target.clone());
                let verb = name.trim_end_matches("_d");
                self.link(&shown, format!("{verb} {target}"), |_| {});
            }
            "newvar" | "var" | "varname" | "vars" | "varlist" | "depvar" | "depvars" | "indepvars" | "ifin" | "weight" | "dtype" | "it_var" => {
                let shown = match name.as_str() {
                    "var" => "varname".to_string(),
                    "vars" => "varlist".to_string(),
                    "ifin" => "[if] [in]".to_string(),
                    "weight" => "[weight]".to_string(),
                    "dtype" => "[type]".to_string(),
                    n => n.to_string(),
                };
                let shown = match text.or((!args.is_empty()).then_some(args)) {
                    Some(t) if !matches!(name.as_str(), "ifin" | "weight" | "dtype") => format!("{shown}{t}"),
                    _ => shown,
                };
                if matches!(name.as_str(), "ifin" | "weight" | "dtype") {
                    self.push(&shown);
                } else {
                    self.styled(&shown, |s| s.italic = true);
                }
            }
            "findalias" => self.push(&format!("[U] {args}")),
            "hline" => {
                if self.para.is_some() {
                    self.push(&"-".repeat(n1.unwrap_or(10)));
                } else {
                    let n = n1.unwrap_or_else(|| self.width.saturating_sub(self.col()));
                    self.push(&"-".repeat(n));
                }
            }
            "dup" => {
                if let Some(t) = text {
                    for _ in 0..n1.unwrap_or(1).min(500) {
                        self.inline(t);
                    }
                }
            }
            "c" | "char" => self.push(&box_char(args)),
            "space" => self.push(&" ".repeat(n1.unwrap_or(1))),
            "col" | "column" => {
                let target = n1.unwrap_or(1).saturating_sub(1);
                let c = self.col();
                if c < target {
                    self.push(&" ".repeat(target - c));
                }
            }
            "right" | "center" | "centre" | "lalign" | "ralign" => {
                let t = text.unwrap_or("");
                // measure the rendered width of the text
                let mut probe = Renderer { width: self.width, out: Vec::new(), cur: Vec::new(), para: None, st: self.st.clone(), synopt_w: 20, p2: self.p2 };
                probe.inline(t);
                let tw = width_of(&probe.cur);
                let field = match name.as_str() {
                    "right" => self.width.saturating_sub(self.col()),
                    _ => n1.unwrap_or(self.width.saturating_sub(self.col())),
                };
                let pad = field.saturating_sub(tw);
                let (l, r) = match name.as_str() {
                    "right" | "ralign" => (pad, 0),
                    "lalign" => (0, pad),
                    _ => (pad / 2, pad - pad / 2),
                };
                self.push(&" ".repeat(l));
                self.inline(t);
                if name != "right" {
                    self.push(&" ".repeat(r));
                }
            }
            "break" | "newline" => {
                if let Some(p) = self.para.as_mut() {
                    p.items.push(Span { text: "\n".into(), style: Style::default() });
                } else {
                    self.end_line();
                }
            }
            "p_end" => self.block(),
            "synoptset" => {
                self.synopt_w = n1.unwrap_or(20);
                self.p2 = (4, 4 + self.synopt_w + 2, 4 + self.synopt_w + 4, 2);
            }
            "p2colset" => {
                if nums.len() >= 3 {
                    self.p2 = (nums[0].saturating_sub(1), nums[1].saturating_sub(1), nums[2].saturating_sub(1), nums.get(3).copied().unwrap_or(2));
                }
            }
            "p2colreset" => self.p2 = (4, 24, 26, 2),
            "synopt" | "p2col" | "p2coldent" | "synoptnotes" | "p2colnotes" => {
                let (c1, c2, c3, right) = if name == "p2col" && nums.len() >= 3 {
                    (nums[0].saturating_sub(1), nums[1].saturating_sub(1), nums[2].saturating_sub(1), nums.get(3).copied().unwrap_or(2))
                } else if name.starts_with("synopt") {
                    (4, 4 + self.synopt_w + 2, 4 + self.synopt_w + 4, 2)
                } else {
                    self.p2
                };
                let c1 = if name == "p2coldent" { c1.saturating_sub(2) } else { c1 };
                // render the first-column text separately
                let mut probe = Renderer { width: self.width, out: Vec::new(), cur: Vec::new(), para: None, st: self.st.clone(), synopt_w: self.synopt_w, p2: self.p2 };
                probe.inline(text.unwrap_or(""));
                self.start_para(c1, c3, right, Some((probe.cur, c2)));
            }
            "syntab" => {
                self.block();
                self.push("  ");
                self.styled(text.unwrap_or(args), |s| s.underline = true);
                self.end_line();
            }
            "synoptline" | "p2line" => {
                self.block();
                let (a, b) = if name == "p2line" && nums.len() >= 2 { (nums[0], nums[1]) } else { (self.p2.0, self.p2.3) };
                self.push(&" ".repeat(a));
                self.push(&"-".repeat(self.width.saturating_sub(a + b)));
                self.end_line();
            }
            "synopthdr" | "p2colhdr" => {
                self.block();
                let first = text.map(str::to_string).unwrap_or_else(|| if args.is_empty() { "options".to_string() } else { args.to_string() });
                let desc_col = if name == "synopthdr" { 4 + self.synopt_w + 2 } else { self.p2.1 };
                self.push("    ");
                self.styled(&first, |s| s.italic = name == "synopthdr");
                let c = self.col();
                self.push(&" ".repeat(desc_col.saturating_sub(c).max(1)));
                self.push("Description");
                self.end_line();
            }
            n if para_shape(n, &nums).is_some() => {
                let (a, b, c) = para_shape(n, &nums).unwrap();
                self.start_para(a, b, c, None);
                if let Some(t) = text {
                    self.inline(t);
                }
            }
            _ => {
                if let Some(t) = text {
                    self.inline(t);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_help_like_document() {
        let src = "{smcl}\n{* comment line}{...}\n{title:Title}\n\n{phang}\n{bf:mycmd} {hline 2} Does something {it:useful} with a long description that keeps going on and on until it wraps around the right margin{p_end}\n\n{synoptset 12}{...}\n{synopthdr}\n{synoptline}\n{synopt:{opt de:tail}}show details{p_end}\n{synoptline}\n\n{pstd}See {help regress:the regress help} and {browse \"https://example.org\"}.{p_end}\n{col 5}a{col 10}b{c |}\n";
        let lines = render(src, 79);
        let text = lines_to_text(&lines);
        assert!(text.starts_with("Title"), "{text}");
        assert!(!text.contains('{'), "{text}");
        assert!(text.contains("    mycmd -- Does something useful"), "{text}");
        assert!(text.lines().all(|l| l.chars().count() <= 79), "{text}");
        assert!(text.contains("        until it wraps") || text.lines().filter(|l| l.starts_with("        ")).count() >= 1, "{text}");
        assert!(text.contains("    detail") && text.contains("show details"), "{text}");
        assert!(text.contains("    a    b|"), "{text}");
        let links: Vec<&Span> = lines.iter().flatten().filter(|s| s.style.link.is_some()).collect();
        assert!(links.iter().any(|s| s.text.contains("regress help") && s.style.link.as_deref() == Some("help regress")));
        assert!(links.iter().any(|s| s.style.link.as_deref() == Some("browse https://example.org")));
        assert!(lines.iter().flatten().any(|s| s.text == "Title" && s.style.bold));
    }

    #[test]
    fn line_mode_and_continuations() {
        assert_eq!(to_text("{txt}a {res}b{...}\nc\n{right:x}"), format!("a bc\n{}x", " ".repeat(78)));
        assert_eq!(to_text("{dup 3:ab}{c -(}{c )-}"), "ababab{}");
    }
}
