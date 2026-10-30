//! `docx2pdf`, `docx2html`, `docx2txt`, `docx2markdown`, `docx2epub`:
//! convert a Word document into other formats with our own writers.
//!
//! Syntax: `docx2xxx srcfile [, saving(destfile) replace nomsg]`; without
//! `saving()` the target has the source name with the new extension.

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::xlsx::crc32;
use ferrostat_core::vfs;

use super::docread::{heading_level, para_text, read_docx};
use super::{has, opt_arg, stem, with_ext, Opts};
use crate::docx::{Block, Docx, Inline, Paragraph, Run, Table, EMU_PER_INCH};
use crate::macros::unquote;

pub fn command(verb: &str, src: &str, opts: &Opts) -> Result<String, String> {
    let ext = match verb {
        "docx2pdf" => "pdf",
        "docx2html" => "html",
        "docx2txt" => "txt",
        "docx2markdown" => "md",
        "docx2epub" => "epub",
        v => return Err(rc_err(199, format!("unrecognized command: {v}"))),
    };
    let src = if src.to_lowercase().ends_with(".docx") || vfs::exists(src) { src.to_string() } else { format!("{src}.docx") };
    let data = vfs::read(&src).map_err(|_| rc_err(601, format!("file {src} not found")))?;
    let doc = read_docx(&data).map_err(|e| rc_err(610, format!("{src}: {e}")))?;
    let target = match opt_arg(opts, "saving", 3).map(|s| unquote(&s)) {
        Some(s) if !s.is_empty() => {
            let s = unquote(s.split(',').next().unwrap_or("").trim());
            if s.rsplit(['/', '\\']).next().unwrap_or(&s).contains('.') { s } else { format!("{s}.{ext}") }
        }
        _ => with_ext(&src, ext),
    };
    let replace = has(opts, "replace", 3) || opt_arg(opts, "saving", 3).is_some_and(|s| s.contains("replace"));
    if vfs::exists(&target) && !replace {
        return Err(rc_err(602, format!("file {target} already exists")));
    }
    let title = stem(&src);
    let bytes: Vec<u8> = match ext {
        "pdf" => crate::pdf::render(&doc),
        "html" => html_page(&title, &to_html(&doc, false, &|m| data_uri(&doc, m))).into_bytes(),
        "txt" => to_text(&doc).into_bytes(),
        "md" => {
            // images are written next to the Markdown file
            let base = with_ext(&target, "");
            let base = base.trim_end_matches('.').to_string();
            let mut names = Vec::new();
            for (k, (n, bytes)) in doc.media.iter().enumerate() {
                let e = n.rsplit('.').next().unwrap_or("png");
                let path = format!("{base}_img{}.{e}", k + 1);
                vfs::write(&path, bytes).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
                names.push(path.rsplit(['/', '\\']).next().unwrap_or(&path).to_string());
            }
            to_markdown(&doc, &names).into_bytes()
        }
        _ => to_epub(&doc, &title),
    };
    vfs::write(&target, &bytes).map_err(|e| rc_err(603, format!("could not write {target}: {e}")))?;
    if has(opts, "nomsg", 5) {
        return Ok(String::new());
    }
    Ok(format!("successfully converted \"{}\"", crate::putdocx::full_path(&target)))
}

// ---------------------------------------------------------------------------
// HTML / XHTML
// ---------------------------------------------------------------------------

pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn mime(name: &str) -> &'static str {
    match name.rsplit('.').next().unwrap_or("").to_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        _ => "image/png",
    }
}

pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for ch in data.chunks(3) {
        let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if ch.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if ch.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn data_uri(doc: &Docx, m: usize) -> String {
    let (n, b) = &doc.media[m];
    format!("data:{};base64,{}", mime(n), base64(b))
}

fn run_html(r: &Run, xhtml: bool) -> String {
    let br = if xhtml { "<br/>" } else { "<br>" };
    let mut s = r.text.split('\n').map(|l| esc(l).replace('\t', "&#9;")).collect::<Vec<_>>().join(br);
    if r.code {
        s = format!("<code>{s}</code>");
    }
    if r.underline {
        s = format!("<u>{s}</u>");
    }
    if r.italic {
        s = format!("<em>{s}</em>");
    }
    if r.bold {
        s = format!("<strong>{s}</strong>");
    }
    let mut css = Vec::new();
    if let Some(c) = &r.color {
        css.push(format!("color:#{c}"));
    }
    if let Some(z) = r.size {
        css.push(format!("font-size:{z}pt"));
    }
    if let Some(f) = r.font.as_ref().filter(|_| !r.code) {
        css.push(format!("font-family:'{}'", esc(f)));
    }
    if !css.is_empty() {
        s = format!("<span style=\"{}\">{s}</span>", css.join(";"));
    }
    if r.linebreak {
        s.push_str(br);
    }
    s
}

fn para_inner_html(p: &Paragraph, xhtml: bool, img: &dyn Fn(usize) -> String) -> String {
    let br = if xhtml { "<br/>" } else { "<br>" };
    let mut s = String::new();
    for it in &p.items {
        match it {
            Inline::Run(r) => s.push_str(&run_html(r, xhtml)),
            Inline::Break => s.push_str(br),
            Inline::Image { media, cx, cy } => {
                let px = |e: u64| (e as f64 / EMU_PER_INCH * 96.0).round() as i64;
                let close = if xhtml { "/>" } else { ">" };
                s.push_str(&format!("<img src=\"{}\" width=\"{}\" height=\"{}\" alt=\"image {}\"{close}", esc(&img(*media)), px(*cx), px(*cy), media + 1));
            }
        }
    }
    s
}

fn table_html(t: &Table, xhtml: bool) -> String {
    let br = if xhtml { "<br/>" } else { "<br>" };
    let mut s = String::from("<table>\n");
    if let Some(ti) = &t.title {
        s.push_str(&format!("<caption>{}</caption>\n", esc(ti)));
    }
    for row in &t.cells {
        s.push_str("<tr>");
        for c in row {
            let tag = if c.bold { "th" } else { "td" };
            let al = c.halign.as_deref().filter(|h| *h != "left").map(|h| format!(" style=\"text-align:{}\"", if h == "both" { "justify" } else { h })).unwrap_or_default();
            let mut txt = c.text.split('\n').map(esc).collect::<Vec<_>>().join(br);
            if c.italic {
                txt = format!("<em>{txt}</em>");
            }
            s.push_str(&format!("<{tag}{al}>{txt}</{tag}>"));
        }
        s.push_str("</tr>\n");
    }
    s.push_str("</table>\n");
    for n in &t.notes {
        s.push_str(&format!("<p class=\"note\">{}</p>\n", esc(n)));
    }
    s
}

/// Body HTML of a document; `img` gives the source of image `m`.
pub fn to_html(doc: &Docx, xhtml: bool, img: &dyn Fn(usize) -> String) -> String {
    let mut s = String::new();
    for b in &doc.blocks {
        match b {
            Block::Para(p) => {
                let inner = para_inner_html(p, xhtml, img);
                let code = p.style.as_deref().is_some_and(|x| x.eq_ignore_ascii_case("code"));
                if let Some(h) = heading_level(p.style.as_deref()) {
                    s.push_str(&format!("<h{h}>{inner}</h{h}>\n"));
                } else if code {
                    s.push_str(&format!("<pre>{}</pre>\n", esc(&para_text(p))));
                } else if inner.is_empty() {
                    continue;
                } else {
                    let al = p.halign.as_deref().filter(|h| *h != "left").map(|h| format!(" style=\"text-align:{}\"", if h == "both" { "justify" } else { h })).unwrap_or_default();
                    s.push_str(&format!("<p{al}>{inner}</p>\n"));
                }
            }
            Block::Table(t) => s.push_str(&table_html(t, xhtml)),
            Block::PageBreak => s.push_str(if xhtml { "<hr class=\"pagebreak\"/>\n" } else { "<hr class=\"pagebreak\">\n" }),
            Block::Raw(_) => {}
        }
    }
    s
}

fn html_page(title: &str, body: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{}</title>\n<style>\nbody {{ font-family: sans-serif; max-width: 50em; margin: 2em auto; padding: 0 1em; line-height: 1.45; }}\ntable {{ border-collapse: collapse; margin: 0.5em 0; }} td, th {{ border: 1px solid #999; padding: 0.2em 0.5em; }}\n.note {{ font-size: 0.85em; }}\nhr.pagebreak {{ border: 0; border-top: 1px dashed #bbb; }}\n</style>\n</head>\n<body>\n{}</body>\n</html>\n",
        esc(title),
        body
    )
}

// ---------------------------------------------------------------------------
// Plain text
// ---------------------------------------------------------------------------

fn table_text(t: &Table) -> String {
    let ncols = t.cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut width = vec![0usize; ncols];
    for row in &t.cells {
        for (j, c) in row.iter().enumerate() {
            for l in c.text.split('\n') {
                width[j] = width[j].max(l.chars().count());
            }
        }
    }
    let mut out = String::new();
    if let Some(ti) = &t.title {
        out.push_str(ti);
        out.push('\n');
    }
    for row in &t.cells {
        let mut cols: Vec<String> = Vec::new();
        for j in 0..ncols {
            let c = row.get(j).map(|c| c.text.replace('\n', " ")).unwrap_or_default();
            let pad = width[j].saturating_sub(c.chars().count());
            let right = row.get(j).and_then(|c| c.halign.as_deref()) == Some("right") || (j > 0 && c.trim().parse::<f64>().is_ok());
            cols.push(if right { format!("{}{c}", " ".repeat(pad)) } else { format!("{c}{}", " ".repeat(pad)) });
        }
        out.push_str(cols.join("  ").trim_end());
        out.push('\n');
    }
    for n in &t.notes {
        out.push_str(n);
        out.push('\n');
    }
    out
}

pub fn to_text(doc: &Docx) -> String {
    let mut out = String::new();
    for b in &doc.blocks {
        match b {
            Block::Para(p) => {
                out.push_str(&para_text(p));
                out.push('\n');
            }
            Block::Table(t) => out.push_str(&table_text(t)),
            Block::PageBreak => out.push('\u{c}'),
            Block::Raw(_) => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------

fn md_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '\\' | '*' | '_' | '`' | '[' | ']' | '|') {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

fn run_md(r: &Run) -> String {
    let lines: Vec<String> = r
        .text
        .split('\n')
        .map(|l| {
            if l.trim().is_empty() {
                return l.to_string();
            }
            // emphasis markers must hug the text
            let lead = &l[..l.len() - l.trim_start().len()];
            let trail = &l[l.trim_end().len()..];
            let mut core = if r.code { format!("`{}`", l.trim()) } else { md_escape(l.trim()) };
            if r.italic {
                core = format!("*{core}*");
            }
            if r.bold {
                core = format!("**{core}**");
            }
            format!("{lead}{core}{trail}")
        })
        .collect();
    let mut s = lines.join("  \n");
    if r.linebreak {
        s.push_str("  \n");
    }
    s
}

pub fn to_markdown(doc: &Docx, images: &[String]) -> String {
    let mut out = String::new();
    for b in &doc.blocks {
        match b {
            Block::Para(p) => {
                let code = p.style.as_deref().is_some_and(|x| x.eq_ignore_ascii_case("code"));
                if code {
                    out.push_str(&format!("```\n{}\n```\n\n", para_text(p)));
                    continue;
                }
                let mut s = String::new();
                for it in &p.items {
                    match it {
                        Inline::Run(r) => s.push_str(&run_md(r)),
                        Inline::Break => s.push_str("  \n"),
                        Inline::Image { media, .. } => s.push_str(&format!("![image {}]({})", media + 1, images.get(*media).cloned().unwrap_or_default())),
                    }
                }
                if s.trim().is_empty() {
                    continue;
                }
                if let Some(h) = heading_level(p.style.as_deref()) {
                    s = format!("{} {}", "#".repeat(h), s.replace("  \n", " "));
                }
                out.push_str(&s);
                out.push_str("\n\n");
            }
            Block::Table(t) => {
                if let Some(ti) = &t.title {
                    out.push_str(&format!("**{}**\n\n", md_escape(ti)));
                }
                let ncols = t.cells.iter().map(Vec::len).max().unwrap_or(0).max(1);
                for (i, row) in t.cells.iter().enumerate() {
                    let cells: Vec<String> = (0..ncols).map(|j| row.get(j).map(|c| md_escape(&c.text.replace('\n', " "))).unwrap_or_default()).collect();
                    out.push_str(&format!("| {} |\n", cells.join(" | ")));
                    if i == 0 {
                        let al: Vec<&str> = (0..ncols)
                            .map(|j| match t.cells.get(1).and_then(|r| r.get(j)).and_then(|c| c.halign.as_deref()) {
                                Some("right") => "---:",
                                Some("center") => ":---:",
                                _ => "---",
                            })
                            .collect();
                        out.push_str(&format!("| {} |\n", al.join(" | ")));
                    }
                }
                out.push('\n');
                for n in &t.notes {
                    out.push_str(&format!("{}\n\n", md_escape(n)));
                }
            }
            Block::PageBreak => out.push_str("---\n\n"),
            Block::Raw(_) => {}
        }
    }
    out.trim_end().to_string() + "\n"
}

// ---------------------------------------------------------------------------
// EPUB 3
// ---------------------------------------------------------------------------

/// A ZIP archive with stored (uncompressed) entries, as EPUB requires for
/// its first entry, `mimetype`.
pub fn zip_stored(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let n = name.as_bytes();
        out.extend_from_slice(&0x04034b50u32.to_le_bytes());
        out.extend_from_slice(&[10, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(n);
        out.extend_from_slice(data);
        central.extend_from_slice(&0x02014b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(n.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(n);
    }
    let cd_off = out.len() as u32;
    let cd_len = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x06054b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

/// A content-derived identifier, stable for the same document.
fn uid(doc: &str) -> String {
    let h = crc32(doc.as_bytes());
    let g = crc32(&doc.as_bytes().iter().rev().copied().collect::<Vec<u8>>());
    format!("urn:uuid:{h:08x}-{:04x}-4{:03x}-8{:03x}-{g:08x}{:04x}", h >> 16, g & 0xfff, (h >> 4) & 0xfff, g >> 16)
}

pub fn to_epub(doc: &Docx, title: &str) -> Vec<u8> {
    let img_name = |m: usize| {
        let e = doc.media[m].0.rsplit('.').next().unwrap_or("png").to_lowercase();
        format!("images/image{}.{e}", m + 1)
    };
    let body = to_html(doc, true, &img_name);
    let content = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE html>\n<html xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:epub=\"http://www.idpf.org/2007/ops\" xml:lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n<title>{}</title>\n<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\"/>\n</head>\n<body>\n{}</body>\n</html>\n",
        esc(title),
        body
    );
    let mut toc = String::new();
    let mut n_head = 0;
    let mut body_ids = content.clone();
    for (k, b) in doc.blocks.iter().enumerate() {
        if let Block::Para(p) = b {
            if heading_level(p.style.as_deref()).is_some() {
                n_head += 1;
                let _ = k;
                toc.push_str(&format!("<li><a href=\"content.xhtml#h{n_head}\">{}</a></li>\n", esc(&para_text(p))));
            }
        }
    }
    // anchor the headings for the navigation document
    let mut out_body = String::new();
    let mut k = 0;
    let mut rest = body_ids.as_str();
    while let Some(i) = rest.find("<h") {
        let is_h = rest[i + 2..].chars().next().is_some_and(|c| c.is_ascii_digit()) && rest[i + 3..].starts_with('>');
        out_body.push_str(&rest[..i + 3]);
        if is_h {
            k += 1;
            out_body.push_str(&format!(" id=\"h{k}\""));
        }
        rest = &rest[i + 3..];
    }
    out_body.push_str(rest);
    body_ids = out_body;
    if toc.is_empty() {
        toc = "<li><a href=\"content.xhtml\">Document</a></li>\n".to_string();
    }
    let nav = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE html>\n<html xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:epub=\"http://www.idpf.org/2007/ops\">\n<head><meta charset=\"utf-8\"/><title>{}</title></head>\n<body>\n<nav epub:type=\"toc\" id=\"toc\"><h1>Contents</h1>\n<ol>\n{toc}</ol>\n</nav>\n</body>\n</html>\n",
        esc(title)
    );
    let mut manifest = String::from("<item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" properties=\"nav\"/>\n<item id=\"content\" href=\"content.xhtml\" media-type=\"application/xhtml+xml\"/>\n<item id=\"css\" href=\"style.css\" media-type=\"text/css\"/>\n");
    for m in 0..doc.media.len() {
        manifest.push_str(&format!("<item id=\"img{}\" href=\"{}\" media-type=\"{}\"/>\n", m + 1, img_name(m), mime(&doc.media[m].0)));
    }
    let opf = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<package xmlns=\"http://www.idpf.org/2007/opf\" version=\"3.0\" unique-identifier=\"uid\">\n<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n<dc:identifier id=\"uid\">{}</dc:identifier>\n<dc:title>{}</dc:title>\n<dc:language>en</dc:language>\n<dc:creator>Ferrostat</dc:creator>\n<meta property=\"dcterms:modified\">2000-01-01T00:00:00Z</meta>\n</metadata>\n<manifest>\n{manifest}</manifest>\n<spine>\n<itemref idref=\"content\"/>\n</spine>\n</package>\n",
        uid(&body_ids),
        esc(title)
    );
    let container = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<container version=\"1.0\" xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\">\n<rootfiles>\n<rootfile full-path=\"OEBPS/content.opf\" media-type=\"application/oebps-package+xml\"/>\n</rootfiles>\n</container>\n";
    let css = "body { font-family: serif; line-height: 1.4; }\ntable { border-collapse: collapse; }\ntd, th { border: 1px solid #999; padding: 0.15em 0.4em; }\n.note { font-size: 0.85em; }\n";
    let mut files: Vec<(String, Vec<u8>)> = vec![
        ("mimetype".into(), b"application/epub+zip".to_vec()),
        ("META-INF/container.xml".into(), container.as_bytes().to_vec()),
        ("OEBPS/content.opf".into(), opf.into_bytes()),
        ("OEBPS/nav.xhtml".into(), nav.into_bytes()),
        ("OEBPS/content.xhtml".into(), body_ids.into_bytes()),
        ("OEBPS/style.css".into(), css.as_bytes().to_vec()),
    ];
    for (m, (_, b)) in doc.media.iter().enumerate() {
        files.push((format!("OEBPS/{}", img_name(m)), b.clone()));
    }
    zip_stored(&files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    #[test]
    fn stored_zip_reads_back() {
        let z = zip_stored(&[("mimetype".into(), b"application/epub+zip".to_vec()), ("a/b.txt".into(), b"hello".to_vec())]);
        assert_eq!(&z[30..38], b"mimetype");
        let m = ferrostat_cmd::xlsx::unzip(&z).unwrap();
        assert_eq!(m["a/b.txt"], b"hello");
    }
}
