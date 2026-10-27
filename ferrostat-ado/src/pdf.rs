//! Render the document model of `crate::docx` as a simple PDF: the standard
//! Helvetica/Courier fonts, word-wrapped paragraphs, grid tables and
//! RGB/grey PNG or JPEG images.

use crate::docx::{Block, Docx, Inline, EMU_PER_INCH};

const MARGIN: f64 = 72.0;

/// Approximate Helvetica advance width of a character (in em).
fn char_width(c: char, bold: bool, mono: bool) -> f64 {
    if mono {
        return 0.6;
    }
    let w = match c {
        ' ' | '.' | ',' | ':' | ';' | '!' | '\'' | '|' | 'i' | 'j' | 'l' | 'I' => 0.278,
        'f' | 't' | 'r' | '(' | ')' | '[' | ']' | '-' | '/' => 0.333,
        'm' | 'M' | 'W' | '%' | '@' => 0.833,
        'w' => 0.722,
        'A'..='Z' => 0.667,
        _ => 0.556,
    };
    if bold { w * 1.06 } else { w }
}

fn text_width(s: &str, size: f64, bold: bool, mono: bool) -> f64 {
    s.chars().map(|c| char_width(c, bold, mono)).sum::<f64>() * size
}

/// A PDF string literal in WinAnsi encoding.
fn pdf_str(s: &str) -> String {
    let mut out = String::from("(");
    for c in s.chars() {
        match c {
            '(' | ')' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 32 => out.push(' '),
            c if (c as u32) < 128 => out.push(c),
            c if (c as u32) < 256 => out.push_str(&format!("\\{:03o}", c as u32)),
            _ => out.push('?'),
        }
    }
    out.push(')');
    out
}

struct Image {
    w: u32,
    h: u32,
    /// Image dictionary entries (without /Length) and the stream bytes.
    dict: String,
    data: Vec<u8>,
}

/// Prepare an image for embedding; PNG data is passed through with the PNG predictor.
fn load_image(data: &[u8]) -> Option<Image> {
    if data.starts_with(b"\x89PNG") {
        let mut i = 8;
        let (mut w, mut h, mut depth, mut ctype, mut interlace) = (0, 0, 0, 0, 0);
        let mut idat = Vec::new();
        while i + 8 <= data.len() {
            let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
            let kind = &data[i + 4..i + 8];
            let body = data.get(i + 8..i + 8 + len)?;
            match kind {
                b"IHDR" => {
                    w = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    h = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                    depth = body[8];
                    ctype = body[9];
                    interlace = body[12];
                }
                b"IDAT" => idat.extend_from_slice(body),
                _ => {}
            }
            i += 12 + len;
        }
        let colors = match ctype {
            0 => 1,
            2 => 3,
            _ => return None,
        };
        if depth != 8 || interlace != 0 {
            return None;
        }
        let cs = if colors == 1 { "/DeviceGray" } else { "/DeviceRGB" };
        return Some(Image {
            w,
            h,
            dict: format!("/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {cs} /BitsPerComponent 8 /Filter /FlateDecode /DecodeParms << /Predictor 15 /Colors {colors} /BitsPerComponent 8 /Columns {w} >>"),
            data: idat,
        });
    }
    if data.starts_with(&[0xFF, 0xD8]) {
        let (w, h) = crate::docx::image_size(data)?;
        let mut comps = 3;
        let mut i = 2;
        while i + 9 < data.len() {
            if data[i] == 0xFF && (0xC0..=0xC2).contains(&data[i + 1]) {
                comps = data[i + 9];
                break;
            }
            i += 1;
        }
        let cs = match comps {
            1 => "/DeviceGray",
            4 => "/DeviceCMYK",
            _ => "/DeviceRGB",
        };
        return Some(Image { w, h, dict: format!("/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {cs} /BitsPerComponent 8 /Filter /DCTDecode"), data: data.to_vec() });
    }
    None
}

struct Writer {
    pages: Vec<String>,
    cur: String,
    y: f64,
    pw: f64,
    ph: f64,
    /// Images used: (media index, object name).
    images: Vec<usize>,
}

impl Writer {
    fn new_page(&mut self) {
        let done = std::mem::take(&mut self.cur);
        self.pages.push(done);
        self.y = self.ph - MARGIN;
    }

    fn need(&mut self, h: f64) {
        if self.y - h < MARGIN && self.y < self.ph - MARGIN - 1.0 {
            self.new_page();
        }
    }
}

/// Font resource name for a style.
fn font_name(bold: bool, italic: bool, mono: bool) -> &'static str {
    match (mono, bold, italic) {
        (true, _, _) => "F5",
        (false, false, false) => "F1",
        (false, true, false) => "F2",
        (false, false, true) => "F3",
        (false, true, true) => "F4",
    }
}

#[derive(Clone)]
struct Word {
    text: String,
    size: f64,
    bold: bool,
    italic: bool,
    mono: bool,
    color: (f64, f64, f64),
    space_before: bool,
}

fn hex_rgb(h: &str) -> (f64, f64, f64) {
    let p = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f64 / 255.0;
    (p(0), p(2), p(4))
}

/// Render a document as PDF bytes.
pub fn render(doc: &Docx) -> Vec<u8> {
    let (pw, ph) = if doc.landscape { (doc.page.1 as f64 / 20.0, doc.page.0 as f64 / 20.0) } else { (doc.page.0 as f64 / 20.0, doc.page.1 as f64 / 20.0) };
    let base = doc.font_size.unwrap_or(11.0);
    let tw = pw - 2.0 * MARGIN;
    let mut wr = Writer { pages: Vec::new(), cur: String::new(), y: ph - MARGIN, pw, ph, images: Vec::new() };
    let mut loaded: Vec<Option<Image>> = doc.media.iter().map(|(_, d)| load_image(d)).collect();
    for b in &doc.blocks {
        match b {
            Block::PageBreak => wr.new_page(),
            Block::Raw(_) => {}
            Block::Table(t) => {
                let size = base * 0.9;
                let lh = size * 1.6;
                if let Some(title) = &t.title {
                    wr.need(lh);
                    wr.y -= lh;
                    wr.cur.push_str(&format!("BT /F2 {size} Tf {MARGIN} {:.2} Td {} Tj ET\n", wr.y + size * 0.4, pdf_str(title)));
                }
                let ncols = t.cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
                let cw = tw / ncols as f64;
                for row in &t.cells {
                    wr.need(lh);
                    let top = wr.y;
                    wr.y -= lh;
                    for (c, cell) in row.iter().enumerate() {
                        let x0 = MARGIN + c as f64 * cw;
                        let mut text = cell.text.clone();
                        while !text.is_empty() && text_width(&text, size, cell.bold, false) > cw - 6.0 {
                            text.pop();
                        }
                        let w = text_width(&text, size, cell.bold, false);
                        let x = match cell.halign.as_deref() {
                            Some("center") => x0 + (cw - w) / 2.0,
                            Some("right") => x0 + cw - 3.0 - w,
                            _ => x0 + 3.0,
                        };
                        wr.cur.push_str(&format!("BT /{} {size} Tf {x:.2} {:.2} Td {} Tj ET\n", font_name(cell.bold, cell.italic, false), wr.y + lh * 0.3, pdf_str(&text)));
                        if t.border {
                            wr.cur.push_str(&format!("0.5 w {x0:.2} {:.2} {cw:.2} {lh:.2} re S\n", wr.y));
                        }
                    }
                    let _ = top;
                }
                wr.y -= base * 0.8;
            }
            Block::Para(p) => {
                let (size, bold, before) = match p.style.as_deref().map(str::to_lowercase).as_deref() {
                    Some("title") => (base * 2.4, false, base),
                    Some("subtitle") => (base * 1.5, false, base * 0.5),
                    Some("heading1") => (base * 1.5, true, base * 1.2),
                    Some("heading2") => (base * 1.25, true, base),
                    Some(s) if s.starts_with("heading") => (base * 1.1, true, base * 0.8),
                    Some("code") => (base * 0.85, false, 0.0),
                    _ => (base, false, 0.0),
                };
                let mono_para = p.style.as_deref() == Some("Code");
                wr.y -= before;
                // lines of words; images are placed on their own lines
                let mut words: Vec<Word> = Vec::new();
                let mut lines: Vec<Vec<Word>> = Vec::new();
                let flush = |words: &mut Vec<Word>, lines: &mut Vec<Vec<Word>>, wr: &mut Writer, align: Option<&str>| {
                    if words.is_empty() {
                        return;
                    }
                    let mut line: Vec<Word> = Vec::new();
                    let mut width = 0.0;
                    let mut all: Vec<(Vec<Word>, f64)> = Vec::new();
                    for w in words.drain(..) {
                        let ww = text_width(&w.text, w.size, w.bold, w.mono);
                        let sp = if w.space_before && !line.is_empty() { text_width(" ", w.size, false, w.mono) } else { 0.0 };
                        if !line.is_empty() && width + sp + ww > tw && w.space_before {
                            all.push((std::mem::take(&mut line), width));
                            width = 0.0;
                        }
                        width += if line.is_empty() { 0.0 } else { sp } + ww;
                        line.push(w);
                    }
                    if !line.is_empty() {
                        all.push((line, width));
                    }
                    for (line, width) in all {
                        let lsize = line.iter().map(|w| w.size).fold(0.0, f64::max);
                        let lh = lsize * if mono_para { 1.2 } else { 1.35 };
                        wr.need(lh);
                        wr.y -= lh;
                        let mut x = match align {
                            Some("center") => MARGIN + (tw - width) / 2.0,
                            Some("right") => MARGIN + tw - width,
                            _ => MARGIN,
                        };
                        for (k, w) in line.iter().enumerate() {
                            if k > 0 && w.space_before {
                                x += text_width(" ", w.size, false, w.mono);
                            }
                            let (r, g, b) = w.color;
                            wr.cur.push_str(&format!("BT /{} {:.2} Tf {r:.3} {g:.3} {b:.3} rg {x:.2} {:.2} Td {} Tj ET\n", font_name(w.bold, w.italic, w.mono), w.size, wr.y + lh * 0.25, pdf_str(&w.text)));
                            x += text_width(&w.text, w.size, w.bold, w.mono);
                        }
                        lines.push(line);
                    }
                };
                let mut pending_space = false;
                for it in &p.items {
                    match it {
                        Inline::Run(r) => {
                            let size = r.size.unwrap_or(size);
                            let color = r.color.as_deref().map(hex_rgb).unwrap_or((0.0, 0.0, 0.0));
                            let mono = r.code || mono_para;
                            for (k, seg) in r.text.split('\n').enumerate() {
                                if k > 0 {
                                    flush(&mut words, &mut lines, &mut wr, p.halign.as_deref());
                                    pending_space = false;
                                }
                                let mut first = true;
                                let parts: Vec<&str> = if mono { vec![seg] } else { seg.split(' ').collect() };
                                for part in parts {
                                    if part.is_empty() {
                                        pending_space = true;
                                        continue;
                                    }
                                    let space_before = pending_space || (!first && !mono);
                                    words.push(Word { text: part.to_string(), size, bold: r.bold || bold, italic: r.italic, mono, color, space_before });
                                    pending_space = false;
                                    first = false;
                                }
                                if seg.ends_with(' ') {
                                    pending_space = true;
                                }
                            }
                            if r.linebreak {
                                flush(&mut words, &mut lines, &mut wr, p.halign.as_deref());
                            }
                        }
                        Inline::Break => {
                            flush(&mut words, &mut lines, &mut wr, p.halign.as_deref());
                        }
                        Inline::Image { media, cx, cy } => {
                            flush(&mut words, &mut lines, &mut wr, p.halign.as_deref());
                            let (mut w, mut h) = (*cx as f64 / EMU_PER_INCH * 72.0, *cy as f64 / EMU_PER_INCH * 72.0);
                            if w > tw {
                                h *= tw / w;
                                w = tw;
                            }
                            let maxh = ph - 2.0 * MARGIN;
                            if h > maxh {
                                w *= maxh / h;
                                h = maxh;
                            }
                            wr.need(h);
                            wr.y -= h;
                            let x = match p.halign.as_deref() {
                                Some("center") => MARGIN + (tw - w) / 2.0,
                                Some("right") => MARGIN + tw - w,
                                _ => MARGIN,
                            };
                            if loaded.get(*media).is_some_and(Option::is_some) {
                                if !wr.images.contains(media) {
                                    wr.images.push(*media);
                                }
                                wr.cur.push_str(&format!("q {w:.2} 0 0 {h:.2} {x:.2} {:.2} cm /Im{media} Do Q\n", wr.y));
                            } else {
                                wr.cur.push_str(&format!("0.5 w {x:.2} {:.2} {w:.2} {h:.2} re S\n", wr.y));
                            }
                        }
                    }
                }
                flush(&mut words, &mut lines, &mut wr, p.halign.as_deref());
                wr.y -= base * 0.6;
            }
        }
    }
    wr.pages.push(std::mem::take(&mut wr.cur));
    // objects: 1 catalog, 2 pages, 3-7 fonts, then images, then page/content pairs
    let mut objs: Vec<Vec<u8>> = vec![Vec::new(), Vec::new()];
    for f in ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"] {
        objs.push(format!("<< /Type /Font /Subtype /Type1 /BaseFont /{f} /Encoding /WinAnsiEncoding >>").into_bytes());
    }
    let mut xobjects = String::new();
    for &m in &wr.images {
        if let Some(img) = loaded.get_mut(m).and_then(Option::take) {
            let _ = (img.w, img.h);
            let mut o = format!("<< {} /Length {} >>\nstream\n", img.dict, img.data.len()).into_bytes();
            o.extend_from_slice(&img.data);
            o.extend_from_slice(b"\nendstream");
            objs.push(o);
            xobjects.push_str(&format!("/Im{m} {} 0 R ", objs.len()));
        }
    }
    let resources = format!("<< /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R /F5 7 0 R >> /XObject << {xobjects}>> >>");
    let mut kids = Vec::new();
    for content in &wr.pages {
        let page_id = objs.len() + 1;
        kids.push(format!("{page_id} 0 R"));
        objs.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.0} {:.0}] /Resources {resources} /Contents {} 0 R >>", wr.pw, wr.ph, page_id + 1).into_bytes());
        objs.push(format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content).into_bytes());
    }
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    objs[1] = format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), kids.len()).into_bytes();
    let mut out = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(o);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in offsets {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", objs.len() + 1, xref).as_bytes());
    out
}
