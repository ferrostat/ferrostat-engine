//! Reporting, wave 6: document conversion (`docx2pdf`, `docx2html`,
//! `docx2txt`, `docx2markdown`, `docx2epub`), text extraction from PDF
//! files (`pdf2txt`), `putdocx collect` / `putpdf collect`, the
//! `docx_hardbreak` / `docx_paramode` settings, formulas and images for
//! `putexcel`, and `ml` with survey designs.

pub mod doccollect;
pub mod docconv;
pub mod docread;
pub mod mlsvy;
pub mod pdftext;
pub mod xlsx_extra;

use ferrostat_cmd::eval::rc_err;
use ferrostat_parser::parser::{find_top_level, split_options};

use crate::docx::{Block, Docx, Inline, Paragraph};
use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

pub(crate) type Opts = Vec<(String, Option<String>)>;

pub(crate) fn split_main(r: &str) -> (String, Opts) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

/// Option `name` (abbreviable to `min` characters) given?
pub(crate) fn has(o: &Opts, name: &str, min: usize) -> bool {
    o.iter().any(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= min && name.starts_with(n.as_str())
    })
}

pub(crate) fn opt_arg(o: &Opts, name: &str, min: usize) -> Option<String> {
    o.iter()
        .find(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min && name.starts_with(n.as_str())
        })
        .and_then(|(_, a)| a.clone())
        .map(|a| a.trim().to_string())
}

/// Replace the extension of `path` (or add one).
pub(crate) fn with_ext(path: &str, ext: &str) -> String {
    let base_start = path.rfind(['/', '\\']).map(|i| i + 1).unwrap_or(0);
    match path[base_start..].rfind('.') {
        Some(i) if i > 0 => format!("{}.{ext}", &path[..base_start + i]),
        _ => format!("{path}.{ext}"),
    }
}

/// File name without directory and extension.
pub(crate) fn stem(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match base.rfind('.') {
        Some(i) if i > 0 => base[..i].to_string(),
        _ => base.to_string(),
    }
}

/// `docx2pdf`, `docx2html`, `docx2txt`, `docx2markdown`, `docx2epub`, `pdf2txt`.
pub fn command(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    let _ = me;
    let (main, opts) = split_main(r);
    let src = unquote(main.trim());
    if src.is_empty() {
        return Err(rc_err(198, "source file required"));
    }
    if verb == "pdf2txt" {
        return pdftext::command(&src, &opts);
    }
    docconv::command(verb, &src, &opts)
}

// ---------------------------------------------------------------------------
// putdocx collect / putpdf collect
// ---------------------------------------------------------------------------

/// Is this `putdocx collect ...` / `putpdf collect ...`?
pub fn is_collect(r: &str) -> bool {
    let w = r.trim_start().split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").to_lowercase();
    w == "collect"
}

pub fn put_collect(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    doccollect::put_collect(me, verb == "putpdf", r)
}

// ---------------------------------------------------------------------------
// set docx_hardbreak / docx_paramode
// ---------------------------------------------------------------------------

/// Is this `set docx_hardbreak ...` or `set docx_paramode ...`?
pub fn is_set_docx(r: &str) -> bool {
    let w = r.trim_start().split_whitespace().next().unwrap_or("").to_lowercase();
    w == "docx_hardbreak" || w == "docx_paramode"
}

pub fn set_docx(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let mut it = r.split(',').next().unwrap_or("").split_whitespace();
    let what = it.next().unwrap_or("").to_lowercase();
    let v = it.next().unwrap_or("").to_lowercase();
    let v = match v.as_str() {
        "on" | "1" => "on",
        "off" | "0" => "off",
        "" => return Err(rc_err(198, format!("set {what}: on or off required"))),
        x => return Err(rc_err(198, format!("set {what} {x} not allowed"))),
    };
    me.engine.store.settings.insert(what, v.to_string());
    Ok(String::new())
}

fn setting_on(me: &AdoInterpreter, name: &str) -> bool {
    me.engine.store.settings.get(name).is_some_and(|v| v == "on")
}

/// HTML to a Word document honouring `docx_hardbreak` (line breaks inside
/// text become hard breaks instead of spaces) and `docx_paramode` (every
/// hard break starts a new paragraph).
pub fn html_to_docx_set(me: &AdoInterpreter, html: &str) -> Docx {
    let src = if setting_on(me, "docx_hardbreak") { hard_breaks(html) } else { html.to_string() };
    let mut d = crate::dyndoc::html_to_docx(&src);
    if setting_on(me, "docx_paramode") {
        para_mode(&mut d);
    }
    d
}

/// Turn newlines inside running text (outside tags and `<pre>`) into `<br>`.
fn hard_breaks(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut pre = 0usize;
    let mut rest = html;
    while !rest.is_empty() {
        if rest.starts_with('<') {
            let end = rest.find('>').map_or(rest.len(), |k| k + 1);
            let tag = rest[..end].to_lowercase();
            if tag.starts_with("<pre") {
                pre += 1;
            } else if tag.starts_with("</pre") {
                pre = pre.saturating_sub(1);
            }
            out.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        let end = rest.find('<').unwrap_or(rest.len());
        let text = &rest[..end];
        if pre > 0 || text.trim().is_empty() {
            out.push_str(text);
        } else {
            // only breaks between pieces of text count
            let lines: Vec<&str> = text.split('\n').collect();
            let first = lines.iter().position(|l| !l.trim().is_empty()).unwrap_or(0);
            let last = lines.iter().rposition(|l| !l.trim().is_empty()).unwrap_or(0);
            for (k, l) in lines.iter().enumerate() {
                if k > 0 {
                    out.push_str(if k > first && k <= last { "<br>" } else { "\n" });
                }
                out.push_str(l);
            }
        }
        rest = &rest[end..];
    }
    out
}

/// Split paragraphs at hard breaks.
fn para_mode(d: &mut Docx) {
    let mut blocks = Vec::with_capacity(d.blocks.len());
    for b in std::mem::take(&mut d.blocks) {
        match b {
            Block::Para(p) if p.items.iter().any(|i| matches!(i, Inline::Break)) => {
                let mut cur = Paragraph { style: p.style.clone(), halign: p.halign.clone(), items: Vec::new() };
                for it in p.items {
                    if matches!(it, Inline::Break) {
                        let next = Paragraph { style: p.style.clone(), halign: p.halign.clone(), items: Vec::new() };
                        blocks.push(Block::Para(std::mem::replace(&mut cur, next)));
                    } else {
                        cur.items.push(it);
                    }
                }
                blocks.push(Block::Para(cur));
            }
            b => blocks.push(b),
        }
    }
    d.blocks = blocks;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_breaks_only_inside_text() {
        let h = hard_breaks("<p>a\nb</p>\n<pre>x\ny</pre>");
        assert_eq!(h, "<p>a<br>b</p>\n<pre>x\ny</pre>");
    }
}
