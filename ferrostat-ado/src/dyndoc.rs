//! Dynamic documents: `dyndoc` (Markdown with dynamic tags to HTML or .docx),
//! `dyntext` (dynamic tags in plain text), `markdown` and `html2docx`.
//!
//! Dynamic tags: `<<dd_do[: quietly nocommands nooutput noprompt]>> … <</dd_do>>`,
//! `<<dd_display: [%fmt] exp>>`, `<<dd_graph[: options]>>`, `<<dd_ignore>>`,
//! `<<dd_remove>>`, `<<dd_skip_if: exp>>/<<dd_skip_else>>/<<dd_skip_end>>`,
//! `<<dd_include: file>>` and `<<dd_version: #>>`. The code runs through the
//! interpreter of the current session; its output is captured into the document.

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_core::vfs;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::docx::{self, Block, Docx, Inline, Paragraph, Run};
use crate::interpreter::AdoInterpreter;
use crate::macros::{self, unquote};
use crate::script;

type Opts = Vec<(String, Option<String>)>;

fn split_main(r: &str) -> (String, Opts) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

fn opt_arg(opts: &Opts, name: &str) -> Option<String> {
    opts.iter().find(|(n, _)| n.to_lowercase() == name).and_then(|(_, a)| a.clone()).map(|a| unquote(a.trim()))
}

fn has(opts: &Opts, name: &str) -> bool {
    opts.iter().any(|(n, _)| n.to_lowercase() == name)
}

/// File name without directory and extension.
fn stem(path: &str) -> &str {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match base.rfind('.') {
        Some(i) if i > 0 => &base[..i],
        _ => base,
    }
}

/// Replace the extension of `path` (or add one).
fn with_ext(path: &str, ext: &str) -> String {
    let base_start = path.rfind(['/', '\\']).map(|i| i + 1).unwrap_or(0);
    match path[base_start..].rfind('.') {
        Some(i) if i > 0 => format!("{}.{ext}", &path[..base_start + i]),
        _ => format!("{path}.{ext}"),
    }
}

/// Run code like a do-file and capture what it prints: commands (with a `. `
/// prompt) and their output.
pub fn run_code(me: &mut AdoInterpreter, code: &str, commands: bool, output: bool, quietly: bool) -> Result<String, String> {
    let lines = script::logical_lines(code);
    let stmts = script::parse_block(&lines)?;
    let saved_out = std::mem::take(&mut me.out);
    let (sq, sn) = (me.quiet, me.noisy);
    me.quiet = if quietly { 1 } else { 0 };
    me.noisy = 0;
    me.depth += 1;
    let mut text: Vec<String> = Vec::new();
    let mut res = Ok(());
    for s in &stmts {
        if commands && !quietly {
            text.push(format!(". {}", s.text.trim()));
        }
        let r = me.exec_block(std::slice::from_ref(s));
        let produced = std::mem::take(&mut me.out);
        if output && !quietly {
            text.extend(produced.into_iter().filter(|l| !l.is_empty()));
        }
        if let Err(e) = r {
            res = Err(e);
            break;
        }
    }
    me.depth -= 1;
    me.quiet = sq;
    me.noisy = sn;
    me.out = saved_out;
    res.map(|_| text.join("\n"))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Markdown,
    Text,
}

struct Ctx {
    target: Target,
    docx: bool,
    base: String,
    graphs: usize,
}

/// Does `s` (the text since the last newline) contain only blanks?
fn blank_tail(out: &str) -> bool {
    out.rsplit('\n').next().unwrap_or("").chars().all(|c| c == ' ' || c == '\t')
}

/// Length of a blank rest-of-line (including the newline) at the start of `s`.
fn blank_head(s: &str) -> Option<usize> {
    let end = s.find('\n').map(|i| i + 1).unwrap_or(s.len());
    if s[..end].trim().is_empty() { Some(end) } else { None }
}

/// Process the dynamic tags of a document.
fn process(me: &mut AdoInterpreter, src: &str, ctx: &mut Ctx) -> Result<String, String> {
    let mut buf = src.to_string();
    let mut pos = 0usize;
    let mut out = String::new();
    while let Some(off) = buf[pos..].find("<<dd_") {
        let start = pos + off;
        out.push_str(&buf[pos..start]);
        let tag_end = buf[start..].find(">>").map(|i| start + i).ok_or_else(|| rc_err(198, "unterminated dynamic tag"))?;
        let tag = buf[start + 2..tag_end].to_string();
        let (name, attrs) = match tag.find(':') {
            Some(i) => (tag[..i].trim().to_lowercase(), tag[i + 1..].trim().to_string()),
            None => (tag.trim().to_lowercase(), String::new()),
        };
        let standalone_before = blank_tail(&out);
        let mut after = tag_end + 2;
        // block tags: the content up to the closing tag
        let block = |buf: &str, after: usize, name: &str| -> Result<(String, usize), String> {
            let close = format!("<</{name}>>");
            let j = buf[after..].find(&close).ok_or_else(|| rc_err(198, format!("<<{name}>> without <</{name}>>")))?;
            let mut content = &buf[after..after + j];
            if let Some(n) = blank_head(content).filter(|_| content.starts_with(['\n', '\r', ' ', '\t'])) {
                content = &content[n..];
            }
            Ok((content.to_string(), after + j + close.len()))
        };
        let replacement: String;
        let mut is_block_output = false;
        match name.as_str() {
            "dd_do" => {
                let (code, end) = block(&buf, after, "dd_do")?;
                after = end;
                let a = attrs.to_lowercase();
                let quietly = a.split_whitespace().any(|w| w.starts_with("qui"));
                let commands = !a.contains("nocommand");
                let output = !a.contains("nooutput");
                let text = run_code(me, &code, commands, output, quietly)?;
                let text = if a.contains("noprompt") { text.lines().map(|l| l.strip_prefix(". ").unwrap_or(l)).collect::<Vec<_>>().join("\n") } else { text };
                replacement = if text.is_empty() {
                    String::new()
                } else if ctx.target == Target::Markdown {
                    format!("~~~~\n{text}\n~~~~\n")
                } else {
                    format!("{text}\n")
                };
                is_block_output = true;
            }
            "dd_display" => {
                let e = {
                    let eng = &mut me.engine;
                    macros::expand(&attrs, &eng.dataset, &mut eng.store)?
                };
                let saved = me.engine.store.rc;
                let t = crate::progcmd::display(me, &e)?;
                me.engine.store.rc = saved;
                replacement = t.trim_matches(['\n', ' ']).to_string();
            }
            "dd_ignore" => {
                let (content, end) = block(&buf, after, "dd_ignore")?;
                after = end;
                replacement = content;
                is_block_output = true;
            }
            "dd_remove" => {
                let (_, end) = block(&buf, after, "dd_remove")?;
                after = end;
                replacement = String::new();
                is_block_output = standalone_before;
            }
            "dd_version" => {
                replacement = String::new();
                is_block_output = true;
            }
            "dd_include" => {
                let path = {
                    let eng = &mut me.engine;
                    unquote(&macros::expand(&attrs, &eng.dataset, &mut eng.store)?)
                };
                let text = vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
                replacement = if text.ends_with('\n') { text } else { format!("{text}\n") };
                is_block_output = true;
            }
            "dd_skip_if" => {
                let cond = {
                    let eng = &mut me.engine;
                    macros::expand(&attrs, &eng.dataset, &mut eng.store)?
                };
                let e = parse_expression(&cond).map_err(|e| rc_err(198, format!("dd_skip_if: {}", e.message)))?;
                let skip = match Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)? {
                    Val::Num(x) => x != 0.0,
                    Val::Str(_) => return Err(rc_err(109, "type mismatch")),
                };
                let end_tag = "<<dd_skip_end>>";
                let j = buf[after..].find(end_tag).map(|j| after + j).ok_or_else(|| rc_err(198, "<<dd_skip_if>> without <<dd_skip_end>>"))?;
                let inner = buf[after..j].to_string();
                let (first, second) = match inner.find("<<dd_skip_else>>") {
                    Some(k) => (inner[..k].to_string(), inner[k + "<<dd_skip_else>>".len()..].to_string()),
                    None => (inner, String::new()),
                };
                let mut kept = if skip { second } else { first };
                if let Some(n) = blank_head(&kept) {
                    kept = kept[n..].to_string();
                }
                if blank_tail(&kept) {
                    let cut = kept.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    kept.truncate(cut);
                }
                let mut tail = j + end_tag.len();
                if standalone_before {
                    if let Some(n) = blank_head(&buf[tail..]) {
                        tail += n;
                    }
                    let cut = out.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    out.truncate(cut);
                }
                // the kept part is processed like the rest of the document
                buf = format!("{}{}{}", &buf[..start], kept, &buf[tail..]);
                pos = start;
                continue;
            }
            "dd_graph" => {
                replacement = graph(me, &attrs, ctx)?;
            }
            _ => return Err(rc_err(198, format!("<<{tag}>>: unknown dynamic tag"))),
        }
        if is_block_output && standalone_before {
            if let Some(n) = blank_head(&buf[after..]) {
                after += n;
                let cut = out.rfind('\n').map(|i| i + 1).unwrap_or(0);
                out.truncate(cut);
                out.push_str(&replacement);
                if !replacement.is_empty() && !replacement.ends_with('\n') {
                    out.push('\n');
                }
                pos = after;
                continue;
            }
        }
        out.push_str(replacement.trim_end_matches('\n'));
        pos = after;
    }
    out.push_str(&buf[pos..]);
    Ok(out)
}

/// `<<dd_graph>>`: export the current graph and link it.
fn graph(me: &mut AdoInterpreter, attrs: &str, ctx: &mut Ctx) -> Result<String, String> {
    let attrs = {
        let eng = &mut me.engine;
        macros::expand(attrs, &eng.dataset, &mut eng.store)?
    };
    let opts = split_options(&attrs);
    ctx.graphs += 1;
    let fmt = ["svg", "png", "pdf", "eps"].into_iter().find(|f| has(&opts, f)).unwrap_or(if ctx.docx { "png" } else { "svg" });
    let file = opt_arg(&opts, "saving").or_else(|| opt_arg(&opts, "sav")).unwrap_or_else(|| format!("{}_{}.{fmt}", ctx.base, ctx.graphs));
    let file = if file.contains('.') { file } else { format!("{file}.{fmt}") };
    if vfs::exists(&file) && !has(&opts, "replace") {
        return Err(rc_err(602, format!("file {file} already exists")));
    }
    let name = opt_arg(&opts, "graphname").map(|n| format!(" name({n})")).unwrap_or_default();
    let mut cmd = format!("graph export \"{file}\", replace{name}");
    for o in ["width", "height"] {
        if let Some(v) = opt_arg(&opts, o) {
            if v.chars().all(|c| c.is_ascii_digit()) {
                cmd.push_str(&format!(" {o}({v})"));
            }
        }
    }
    if run_code(me, &cmd, false, false, true).is_err() {
        // the export size options are optional; retry with defaults
        run_code(me, &format!("graph export \"{file}\", replace{name}"), false, false, true)?;
    }
    let alt = opt_arg(&opts, "alt").unwrap_or_default();
    if ctx.target == Target::Text {
        return Ok(file);
    }
    if has(&opts, "markdown") {
        return Ok(format!("![{alt}]({file})"));
    }
    let mut img = format!("<img src=\"{}\"", html_escape(&file));
    for o in ["height", "width"] {
        if let Some(v) = opt_arg(&opts, o) {
            img.push_str(&format!(" {o}=\"{}\"", html_escape(&v)));
        }
    }
    img.push_str(&format!(" alt=\"{}\" >", html_escape(&alt)));
    Ok(img)
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Markdown to HTML
// ---------------------------------------------------------------------------

/// Does `inner` (the text between `<` and `>`) look like an HTML tag?
fn is_tag_text(inner: &str) -> bool {
    let t = inner.strip_prefix('/').unwrap_or(inner);
    if t.starts_with('!') {
        return true;
    }
    let n = t.chars().take_while(|c| c.is_ascii_alphanumeric()).count();
    n > 0 && t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) && t[n..].chars().next().is_none_or(|c| c.is_whitespace() || c == '/')
}

/// Inline Markdown (emphasis, code, links, images) to HTML.
pub fn inline_html(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let find = |from: usize, pat: &str| -> Option<usize> {
        let p: Vec<char> = pat.chars().collect();
        (from..c.len()).find(|&k| c[k..].starts_with(&p))
    };
    while i < c.len() {
        let ch = c[i];
        // raw HTML tags pass through
        if ch == '<' {
            if let Some(j) = find(i, ">") {
                let inner: String = c[i + 1..j].iter().collect();
                if is_tag_text(&inner) {
                    out.push_str(&c[i..=j].iter().collect::<String>());
                    i = j + 1;
                    continue;
                }
            }
            out.push_str("&lt;");
            i += 1;
            continue;
        }
        if ch == '\\' && i + 1 < c.len() && "\\`*_{}[]()#+-.!<>".contains(c[i + 1]) {
            out.push_str(&html_escape(&c[i + 1].to_string()));
            i += 2;
            continue;
        }
        if ch == '`' {
            if let Some(j) = find(i + 1, "`") {
                out.push_str(&format!("<code>{}</code>", html_escape(&c[i + 1..j].iter().collect::<String>())));
                i = j + 1;
                continue;
            }
        }
        if (ch == '*' || ch == '_') && i + 1 < c.len() && c[i + 1] == ch {
            let pat: String = [ch, ch].iter().collect();
            if let Some(j) = find(i + 2, &pat).filter(|&j| j > i + 2) {
                out.push_str(&format!("<strong>{}</strong>", inline_html(&c[i + 2..j].iter().collect::<String>())));
                i = j + 2;
                continue;
            }
        }
        if ch == '*' || (ch == '_' && (i == 0 || !c[i - 1].is_alphanumeric())) {
            if let Some(j) = find(i + 1, &ch.to_string()).filter(|&j| j > i + 1 && !c[i + 1].is_whitespace()) {
                out.push_str(&format!("<em>{}</em>", inline_html(&c[i + 1..j].iter().collect::<String>())));
                i = j + 1;
                continue;
            }
        }
        let image = ch == '!' && c.get(i + 1) == Some(&'[');
        if ch == '[' || image {
            let open = if image { i + 1 } else { i };
            if let Some(cb) = find(open, "]") {
                if c.get(cb + 1) == Some(&'(') {
                    if let Some(cp) = find(cb + 2, ")") {
                        let text: String = c[open + 1..cb].iter().collect();
                        let target: String = c[cb + 2..cp].iter().collect();
                        let url = target.split_whitespace().next().unwrap_or("");
                        if image {
                            out.push_str(&format!("<img src=\"{}\" alt=\"{}\" >", html_escape(url), html_escape(&text)));
                        } else {
                            out.push_str(&format!("<a href=\"{}\">{}</a>", html_escape(url), inline_html(&text)));
                        }
                        i = cp + 1;
                        continue;
                    }
                }
            }
        }
        match ch {
            '&' => {
                // keep entities such as &amp; as written
                let ent = find(i, ";").filter(|&j| j - i <= 8 && c[i + 1..j].iter().all(|x| x.is_ascii_alphanumeric() || *x == '#'));
                if ent.is_some() { out.push('&') } else { out.push_str("&amp;") }
            }
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
        i += 1;
    }
    out
}

fn heading_level(line: &str) -> Option<(usize, &str)> {
    let t = line.trim_start();
    let n = t.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&n) && (t.len() == n || t[n..].starts_with(' ')) {
        Some((n, t[n..].trim().trim_end_matches('#').trim()))
    } else {
        None
    }
}

fn list_item(line: &str) -> Option<(bool, &str)> {
    let t = line.trim_start();
    for p in ["- ", "* ", "+ "] {
        if let Some(r) = t.strip_prefix(p) {
            return Some((false, r));
        }
    }
    let d = t.chars().take_while(|c| c.is_ascii_digit()).count();
    if d > 0 && (t[d..].starts_with(". ") || t[d..].starts_with(") ")) {
        return Some((true, &t[d + 2..]));
    }
    None
}

fn is_html_block(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('<') && t[1..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '/' || c == '!')
}

/// Convert Markdown to an HTML fragment.
pub fn markdown_to_html(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut para: Vec<String> = Vec::new();
    let flush = |para: &mut Vec<String>, out: &mut String| {
        if !para.is_empty() {
            out.push_str(&format!("<p>{}</p>\n", inline_html(&para.join("\n"))));
            para.clear();
        }
    };
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        // fenced code
        if t.starts_with("~~~") || t.starts_with("```") {
            flush(&mut para, &mut out);
            let fence = &t[..3];
            let mut code = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with(fence) {
                code.push(lines[i]);
                i += 1;
            }
            out.push_str(&format!("<pre><code>{}</code></pre>\n", html_escape(&code.join("\n"))));
            i += 1;
            continue;
        }
        if t.is_empty() {
            flush(&mut para, &mut out);
            i += 1;
            continue;
        }
        // setext headings
        if para.is_empty() && i + 1 < lines.len() {
            let next = lines[i + 1].trim();
            if !next.is_empty() && (next.chars().all(|c| c == '=') || (next.chars().all(|c| c == '-') && next.len() >= 2)) && !is_html_block(line) && list_item(line).is_none() {
                let lvl = if next.starts_with('=') { 1 } else { 2 };
                out.push_str(&format!("<h{lvl}>{}</h{lvl}>\n", inline_html(t)));
                i += 2;
                continue;
            }
        }
        if let Some((lvl, text)) = heading_level(line) {
            flush(&mut para, &mut out);
            out.push_str(&format!("<h{lvl}>{}</h{lvl}>\n", inline_html(text)));
            i += 1;
            continue;
        }
        if para.is_empty() && (t == "---" || t == "***" || t == "___") {
            out.push_str("<hr>\n");
            i += 1;
            continue;
        }
        // indented code (only outside paragraphs)
        if para.is_empty() && (line.starts_with("    ") || line.starts_with('\t')) {
            let mut code = Vec::new();
            while i < lines.len() && (lines[i].starts_with("    ") || lines[i].starts_with('\t') || lines[i].trim().is_empty()) {
                let l = lines[i];
                code.push(l.strip_prefix("    ").or_else(|| l.strip_prefix('\t')).unwrap_or(""));
                i += 1;
            }
            while code.last().is_some_and(|l| l.trim().is_empty()) {
                code.pop();
            }
            if !code.is_empty() {
                out.push_str(&format!("<pre><code>{}</code></pre>\n", html_escape(&code.join("\n"))));
            }
            continue;
        }
        if para.is_empty() && is_html_block(line) {
            // raw HTML up to the next blank line
            while i < lines.len() && !lines[i].trim().is_empty() {
                out.push_str(lines[i]);
                out.push('\n');
                i += 1;
            }
            continue;
        }
        if let Some((ordered, _)) = list_item(line) {
            flush(&mut para, &mut out);
            let tag = if ordered { "ol" } else { "ul" };
            out.push_str(&format!("<{tag}>\n"));
            while i < lines.len() {
                match list_item(lines[i]) {
                    Some((o, text)) if o == ordered => {
                        let mut item = text.to_string();
                        i += 1;
                        while i < lines.len() && !lines[i].trim().is_empty() && list_item(lines[i]).is_none() && lines[i].starts_with(' ') {
                            item.push(' ');
                            item.push_str(lines[i].trim());
                            i += 1;
                        }
                        out.push_str(&format!("<li>{}</li>\n", inline_html(&item)));
                    }
                    _ => break,
                }
            }
            out.push_str(&format!("</{tag}>\n"));
            continue;
        }
        if let Some(q) = t.strip_prefix('>') {
            flush(&mut para, &mut out);
            out.push_str(&format!("<blockquote><p>{}</p></blockquote>\n", inline_html(q.trim())));
            i += 1;
            continue;
        }
        // two trailing spaces force a line break
        if line.ends_with("  ") {
            para.push(format!("{}<br>", t));
        } else {
            para.push(t.to_string());
        }
        i += 1;
    }
    flush(&mut para, &mut out);
    out
}

/// A complete HTML page around a fragment.
pub fn html_page(title: &str, body: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{}</title>\n<style>\nbody {{ font-family: sans-serif; max-width: 50em; margin: 2em auto; padding: 0 1em; line-height: 1.45; }}\npre {{ background: #f5f5f5; padding: 0.6em; overflow-x: auto; }}\ntable {{ border-collapse: collapse; }} td, th {{ border: 1px solid #999; padding: 0.2em 0.5em; }}\n</style>\n</head>\n<body>\n{}</body>\n</html>\n",
        html_escape(title),
        body
    )
}

// ---------------------------------------------------------------------------
// HTML to .docx
// ---------------------------------------------------------------------------

fn unescape_html(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'").replace("&nbsp;", "\u{a0}").replace("&amp;", "&")
}

/// Attributes of a start tag (`name="v"`, `name='v'`, `name=v`).
fn tag_attrs(tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let c: Vec<char> = tag.chars().collect();
    let mut i = c.iter().position(|ch| ch.is_whitespace()).unwrap_or(c.len());
    while i < c.len() {
        while i < c.len() && (c[i].is_whitespace() || c[i] == '/') {
            i += 1;
        }
        let ns = i;
        while i < c.len() && !c[i].is_whitespace() && c[i] != '=' && c[i] != '/' {
            i += 1;
        }
        let name: String = c[ns..i].iter().collect::<String>().to_lowercase();
        let mut val = String::new();
        if i < c.len() && c[i] == '=' {
            i += 1;
            if i < c.len() && (c[i] == '"' || c[i] == '\'') {
                let q = c[i];
                i += 1;
                let vs = i;
                while i < c.len() && c[i] != q {
                    i += 1;
                }
                val = c[vs..i].iter().collect();
                i += 1;
            } else {
                let vs = i;
                while i < c.len() && !c[i].is_whitespace() {
                    i += 1;
                }
                val = c[vs..i].iter().collect();
            }
        }
        if !name.is_empty() {
            out.push((name, unescape_html(&val)));
        }
    }
    out
}

/// Build a document from (simple) HTML: headings, paragraphs, preformatted
/// text, lists, line breaks, emphasis, tables and images.
pub fn html_to_docx(html: &str) -> Docx {
    let mut d = Docx::new();
    let (mut bold, mut italic, mut underline, mut code) = (0usize, 0usize, 0usize, 0usize);
    let mut pre = 0usize;
    let mut style: Option<String> = None;
    let mut list: Vec<(bool, usize)> = Vec::new();
    let mut table: Option<(Vec<Vec<docx::Cell>>, bool)> = None;
    let mut cell: Option<String> = None;
    let mut open = false;
    let mut skip = 0usize;
    let body = match (html.find("<body"), html.rfind("</body>")) {
        (Some(a), Some(b)) if a < b => html[a..b].find('>').map(|k| &html[a + k + 1..b]).unwrap_or(html),
        _ => html,
    };
    let ensure_para = |d: &mut Docx, open: &mut bool, style: &Option<String>| {
        if !*open {
            d.blocks.push(Block::Para(Paragraph { style: style.clone(), halign: None, items: Vec::new() }));
            *open = true;
        }
    };
    let mut rest = body;
    while !rest.is_empty() {
        if let Some(stripped) = rest.strip_prefix("<!--") {
            rest = stripped.find("-->").map(|k| &stripped[k + 3..]).unwrap_or("");
            continue;
        }
        if rest.starts_with('<') && rest[1..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '/' || c == '!') {
            let end = rest.find('>').unwrap_or(rest.len() - 1);
            let tag = &rest[1..end];
            rest = &rest[end + 1..];
            let closing = tag.starts_with('/');
            let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
            match (name.as_str(), closing) {
                ("script" | "style" | "head" | "title", false) => skip += 1,
                ("script" | "style" | "head" | "title", true) => skip = skip.saturating_sub(1),
                ("h1" | "h2" | "h3" | "h4" | "h5" | "h6", false) => {
                    style = Some(format!("Heading{}", &name[1..]));
                    open = false;
                }
                ("h1" | "h2" | "h3" | "h4" | "h5" | "h6", true) => {
                    style = None;
                    open = false;
                }
                ("p" | "div" | "blockquote", _) => open = false,
                ("pre", false) => {
                    pre += 1;
                    style = Some("Code".into());
                    open = false;
                }
                ("pre", true) => {
                    pre = pre.saturating_sub(1);
                    style = None;
                    open = false;
                }
                ("br", _) => {
                    if let Some(c) = cell.as_mut() {
                        c.push('\n');
                    } else {
                        ensure_para(&mut d, &mut open, &style);
                        d.current_para().unwrap().items.push(Inline::Break);
                    }
                }
                ("hr", _) => open = false,
                ("b" | "strong", c) => bold = if c { bold.saturating_sub(1) } else { bold + 1 },
                ("i" | "em", c) => italic = if c { italic.saturating_sub(1) } else { italic + 1 },
                ("u", c) => underline = if c { underline.saturating_sub(1) } else { underline + 1 },
                ("code" | "tt" | "kbd", c) => code = if c { code.saturating_sub(1) } else { code + 1 },
                ("ul" | "ol", false) => {
                    list.push((name == "ol", 0));
                    open = false;
                }
                ("ul" | "ol", true) => {
                    list.pop();
                    open = false;
                }
                ("li", false) => {
                    let marker = match list.last_mut() {
                        Some((true, n)) => {
                            *n += 1;
                            format!("{n}. ")
                        }
                        _ => "\u{2022} ".to_string(),
                    };
                    d.blocks.push(Block::Para(Paragraph { style: Some("ListParagraph".into()), halign: None, items: vec![Inline::Run(Run { text: marker, ..Default::default() })] }));
                    open = true;
                }
                ("li", true) => open = false,
                ("table", false) => {
                    table = Some((Vec::new(), true));
                    open = false;
                }
                ("table", true) => {
                    if let Some((cells, border)) = table.take() {
                        let n = d.blocks.iter().filter(|b| matches!(b, Block::Table(_))).count();
                        d.blocks.push(Block::Table(docx::Table { name: format!("t{}", n + 1), cells, border, title: None, notes: Vec::new() }));
                    }
                    open = false;
                }
                ("tr", false) => {
                    if let Some((cells, _)) = table.as_mut() {
                        cells.push(Vec::new());
                    }
                }
                ("td" | "th", false) => cell = Some(String::new()),
                ("td" | "th", true) => {
                    if let (Some(text), Some((cells, _))) = (cell.take(), table.as_mut()) {
                        if cells.is_empty() {
                            cells.push(Vec::new());
                        }
                        cells.last_mut().unwrap().push(docx::Cell { text: text.trim().to_string(), bold: name == "th", ..Default::default() });
                    }
                }
                ("img", false) => {
                    let attrs = tag_attrs(tag);
                    let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
                    if let Some(src) = get("src") {
                        if let Ok(data) = vfs::read(&src) {
                            let px = |v: Option<String>| v.and_then(|v| docx::parse_length(&if v.chars().all(|c| c.is_ascii_digit() || c == '.') { format!("{v}px") } else { v }));
                            let (w, h) = (px(get("width")), px(get("height")));
                            let img = d.add_image(&src, data, w, h);
                            ensure_para(&mut d, &mut open, &style);
                            d.current_para().unwrap().items.push(img);
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        let next = rest[1..].find('<').map(|k| k + 1).unwrap_or(rest.len());
        let raw = &rest[..next];
        rest = &rest[next..];
        if skip > 0 {
            continue;
        }
        let text = if pre > 0 {
            unescape_html(raw.strip_prefix('\n').unwrap_or(raw))
        } else {
            let collapsed: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
            let lead = raw.starts_with(char::is_whitespace) && !collapsed.is_empty();
            let trail = raw.ends_with(char::is_whitespace) && !collapsed.is_empty();
            unescape_html(&format!("{}{}{}", if lead { " " } else { "" }, collapsed, if trail { " " } else { "" }))
        };
        if let Some(c) = cell.as_mut() {
            c.push_str(&text);
            continue;
        }
        if text.trim().is_empty() && (!open || pre == 0) {
            if open && !text.is_empty() {
                if let Some(Inline::Run(r)) = d.current_para().and_then(|p| p.items.last_mut()) {
                    if !r.text.ends_with(' ') {
                        r.text.push(' ');
                    }
                }
            }
            continue;
        }
        let was_open = open;
        ensure_para(&mut d, &mut open, &style);
        let text = if was_open { text } else { text.trim_start().to_string() };
        let text = if pre > 0 { text.trim_end_matches('\n').to_string() } else { text };
        d.current_para().unwrap().items.push(Inline::Run(Run { text, bold: bold > 0, italic: italic > 0, underline: underline > 0, code: code > 0 || pre > 0, ..Default::default() }));
    }
    d
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Read the source file and set up `1`, `2`, … from extra arguments.
fn source(me: &mut AdoInterpreter, main: &str) -> Result<(String, String), String> {
    let words = split_top_level_words(main);
    let path = unquote(words.first().ok_or_else(|| rc_err(198, "source file required"))?);
    let text = vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let st = &mut me.engine.store;
    for (k, w) in words.iter().skip(1).enumerate() {
        st.set_local(&(k + 1).to_string(), unquote(w));
    }
    Ok((path, text))
}

fn check_target(path: &str, opts: &Opts) -> Result<(), String> {
    if vfs::exists(path) && !has(opts, "replace") {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    Ok(())
}

fn write(path: &str, data: &[u8]) -> Result<String, String> {
    vfs::write(path, data).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("successfully created \"{}\"", crate::putdocx::full_path(path)))
}

/// `dyndoc srcfile [args] [, saving(file) replace docx html]`.
pub fn dyndoc(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_main(r);
    me.engine.store.push_frame();
    let res = (|| {
        let (path, text) = source(me, &main)?;
        let to_docx = has(&opts, "docx");
        let target = opt_arg(&opts, "saving").map(|s| if s.contains('.') { s } else { with_ext(&s, if to_docx { "docx" } else { "html" }) }).unwrap_or_else(|| with_ext(&path, if to_docx { "docx" } else { "html" }));
        check_target(&target, &opts)?;
        let mut ctx = Ctx { target: Target::Markdown, docx: to_docx, base: stem(&path).to_string(), graphs: 0 };
        let md = process(me, &text, &mut ctx)?;
        let body = markdown_to_html(&md);
        if to_docx {
            write(&target, &crate::rpt_w6::html_to_docx_set(me, &body).to_bytes())
        } else {
            write(&target, html_page(stem(&path), &body).as_bytes())
        }
    })();
    me.engine.store.pop_frame();
    res
}

/// `dyntext srcfile [args], saving(file) [replace]`.
pub fn dyntext(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_main(r);
    let target = opt_arg(&opts, "saving").ok_or_else(|| rc_err(198, "option saving() required"))?;
    me.engine.store.push_frame();
    let res = (|| {
        let (path, text) = source(me, &main)?;
        check_target(&target, &opts)?;
        let mut ctx = Ctx { target: Target::Text, docx: false, base: stem(&path).to_string(), graphs: 0 };
        let out = process(me, &text, &mut ctx)?;
        write(&target, out.as_bytes())
    })();
    me.engine.store.pop_frame();
    res
}

/// `markdown srcfile, saving(file) [replace docx]`.
pub fn markdown(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_main(r);
    let path = unquote(&main);
    let text = vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let to_docx = has(&opts, "docx");
    let target = opt_arg(&opts, "saving").ok_or_else(|| rc_err(198, "option saving() required"))?;
    check_target(&target, &opts)?;
    let body = markdown_to_html(&text);
    if to_docx {
        write(&target, &crate::rpt_w6::html_to_docx_set(me, &body).to_bytes())
    } else {
        write(&target, html_page(stem(&path), &body).as_bytes())
    }
}

/// `html2docx srcfile [, saving(file) replace]`.
pub fn html2docx(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_main(r);
    let path = unquote(&main);
    let text = vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let target = opt_arg(&opts, "saving").map(|s| if s.contains('.') { s } else { with_ext(&s, "docx") }).unwrap_or_else(|| with_ext(&path, "docx"));
    check_target(&target, &opts)?;
    write(&target, &crate::rpt_w6::html_to_docx_set(me, &text).to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_basics() {
        let h = markdown_to_html("Title\n=====\n\n## Sub *x*\n\nSome **bold** and `code`.\n\n- a\n- b\n\n~~~~\n. di 1\n1\n~~~~\n");
        assert!(h.contains("<h1>Title</h1>"));
        assert!(h.contains("<h2>Sub <em>x</em></h2>"));
        assert!(h.contains("<strong>bold</strong>"));
        assert!(h.contains("<li>a</li>"));
        assert!(h.contains("<pre><code>. di 1\n1</code></pre>"));
    }

    #[test]
    fn html_to_docx_blocks() {
        let d = html_to_docx("<h1>A</h1>\n<p>x <b>y</b></p><pre><code>. di 1\n1</code></pre>");
        let x = d.document_xml();
        assert!(x.contains("Heading1"));
        assert!(x.contains("<w:b/>"));
        assert!(x.contains("Courier New"));
    }

    #[test]
    fn names() {
        assert_eq!(with_ext("dir/Dyndoc.txt", "html"), "dir/Dyndoc.html");
        assert_eq!(stem("dir/Dyndoc.txt"), "Dyndoc");
    }
}
