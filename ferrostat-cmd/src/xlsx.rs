//! Office Open XML spreadsheets (.xlsx): ZIP container, a small XML reader,
//! workbook reading for `import excel` and writing for `export excel` /
//! `putexcel`. Implemented from the published ECMA-376 format.

use std::collections::BTreeMap;
use std::io::Read;

use crate::eval::rc_err;

// ---------------------------------------------------------------------------
// ZIP
// ---------------------------------------------------------------------------

fn u16le(b: &[u8], p: usize) -> usize {
    u16::from_le_bytes([b[p], b[p + 1]]) as usize
}
fn u32le(b: &[u8], p: usize) -> usize {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as usize
}

/// Read all entries of a ZIP archive (stored or deflated).
pub fn unzip(data: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let bad = || rc_err(610, "file is not a valid .xlsx (ZIP) workbook");
    // end of central directory: search backwards for the signature
    let eocd = (0..data.len().saturating_sub(21)).rev().find(|&p| data[p..p + 4] == [0x50, 0x4b, 0x05, 0x06]).ok_or_else(bad)?;
    let count = u16le(data, eocd + 10);
    let mut p = u32le(data, eocd + 16);
    let mut out = BTreeMap::new();
    for _ in 0..count {
        if p + 46 > data.len() || data[p..p + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(bad());
        }
        let method = u16le(data, p + 10);
        let csize = u32le(data, p + 20);
        let usize_ = u32le(data, p + 24);
        let nlen = u16le(data, p + 28);
        let elen = u16le(data, p + 30);
        let clen = u16le(data, p + 32);
        let local = u32le(data, p + 42);
        let name = String::from_utf8_lossy(&data[p + 46..p + 46 + nlen]).to_string();
        p += 46 + nlen + elen + clen;
        if local + 30 > data.len() {
            return Err(bad());
        }
        let lstart = local + 30 + u16le(data, local + 26) + u16le(data, local + 28);
        let raw = data.get(lstart..lstart + csize).ok_or_else(bad)?;
        let content = match method {
            0 => raw.to_vec(),
            8 => {
                let mut dec = flate2::read::DeflateDecoder::new(raw);
                let mut v = Vec::with_capacity(usize_);
                dec.read_to_end(&mut v).map_err(|_| bad())?;
                v
            }
            m => return Err(rc_err(610, format!("unsupported ZIP compression method {m}"))),
        };
        out.insert(name, content);
    }
    Ok(out)
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Write a ZIP archive with deflated entries.
pub fn zip(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    use std::io::Write;
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        let comp = enc.finish().unwrap();
        let offset = out.len() as u32;
        let crc = crc32(data);
        let n = name.as_bytes();
        out.extend_from_slice(&0x04034b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 8, 0, 0, 0, 0x21, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(n);
        out.extend_from_slice(&comp);
        central.extend_from_slice(&0x02014b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0x21, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(comp.len() as u32).to_le_bytes());
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

// ---------------------------------------------------------------------------
// Minimal XML reader (elements, attributes, text)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum XmlEvent {
    Start(String, Vec<(String, String)>, bool), // name, attributes, self-closing
    End(String),
    Text(String),
}

pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let end = tail.find(';').unwrap_or(0);
        let ent = &tail[1..end.max(1)];
        let rep = match ent {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            e if e.starts_with("#x") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
            e if e.starts_with('#') => e[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match rep {
            Some(c) if end > 0 => {
                out.push(c);
                rest = &tail[end + 1..];
            }
            _ => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Strip a namespace prefix: `x:row` -> `row`.
fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

pub fn xml_events(src: &str) -> Vec<XmlEvent> {
    let mut ev = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'<' {
            if src[i..].starts_with("<?") {
                i = src[i..].find("?>").map_or(b.len(), |j| i + j + 2);
                continue;
            }
            if src[i..].starts_with("<!--") {
                i = src[i..].find("-->").map_or(b.len(), |j| i + j + 3);
                continue;
            }
            if src[i..].starts_with("<![CDATA[") {
                let end = src[i..].find("]]>").map_or(b.len(), |j| i + j);
                ev.push(XmlEvent::Text(src[i + 9..end].to_string()));
                i = end + 3;
                continue;
            }
            let end = match src[i..].find('>') {
                Some(j) => i + j,
                None => break,
            };
            let inner = &src[i + 1..end];
            if let Some(n) = inner.strip_prefix('/') {
                ev.push(XmlEvent::End(local(n.trim()).to_string()));
            } else {
                let self_closing = inner.ends_with('/');
                let inner = inner.trim_end_matches('/');
                let name_end = inner.find(|c: char| c.is_whitespace()).unwrap_or(inner.len());
                let name = local(&inner[..name_end]).to_string();
                let mut attrs = Vec::new();
                let mut rest = &inner[name_end..];
                while let Some(eq) = rest.find('=') {
                    let key = rest[..eq].trim().to_string();
                    let after = rest[eq + 1..].trim_start();
                    let q = after.chars().next().unwrap_or('"');
                    let close = after[1..].find(q).map_or(after.len(), |j| j + 1);
                    let val = unescape(&after[1..close]);
                    attrs.push((local(&key).to_string(), val));
                    rest = &after[(close + 1).min(after.len())..];
                }
                ev.push(XmlEvent::Start(name, attrs, self_closing));
            }
            i = end + 1;
        } else {
            let end = src[i..].find('<').map_or(b.len(), |j| i + j);
            let text = &src[i..end];
            if !text.is_empty() {
                ev.push(XmlEvent::Text(unescape(text)));
            }
            i = end;
        }
    }
    ev
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

// ---------------------------------------------------------------------------
// Workbook model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Num(f64),
    Str(String),
    Bool(bool),
    /// A number with a date/time number format (Excel serial value).
    Date(f64),
}

#[derive(Debug, Clone, Default)]
pub struct Workbook {
    /// (sheet name, cells keyed by (row, col), 1-based)
    pub sheets: Vec<(String, BTreeMap<(u32, u32), Cell>)>,
}

/// "B12" -> (12, 2)
pub fn parse_ref(r: &str) -> Option<(u32, u32)> {
    let r = r.trim().replace('$', "");
    let letters: String = r.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    let digits = &r[letters.len()..];
    if letters.is_empty() || digits.is_empty() {
        return None;
    }
    let col = letters.to_uppercase().bytes().fold(0u32, |a, b| a * 26 + (b - b'A' + 1) as u32);
    Some((digits.parse().ok()?, col))
}

pub fn col_letters(mut c: u32) -> String {
    let mut s = Vec::new();
    while c > 0 {
        s.push((b'A' + ((c - 1) % 26) as u8) as char);
        c = (c - 1) / 26;
    }
    s.iter().rev().collect()
}

/// Built-in number formats that display dates or times.
fn is_date_format(id: u32, code: Option<&str>) -> bool {
    if (14..=22).contains(&id) || (45..=47).contains(&id) || (27..=36).contains(&id) || (50..=58).contains(&id) {
        return true;
    }
    match code {
        Some(c) => {
            // remove quoted text and bracketed sections, then look for date tokens
            let mut s = String::new();
            let mut in_q = false;
            let mut in_b = false;
            for ch in c.chars() {
                match ch {
                    '"' => in_q = !in_q,
                    '[' if !in_q => in_b = true,
                    ']' if !in_q => in_b = false,
                    _ if !in_q && !in_b => s.push(ch.to_ascii_lowercase()),
                    _ => {}
                }
            }
            s.contains('y') || s.contains('d') || (s.contains('m') && (s.contains('h') || s.contains('s') || s.contains("mm")))
        }
        None => false,
    }
}

/// Excel serial day number (1900 system) to days since 01jan1960.
pub fn excel_to_td(serial: f64) -> f64 {
    serial - 21916.0
}

impl Workbook {
    pub fn read(data: &[u8]) -> Result<Workbook, String> {
        let files = unzip(data)?;
        let text = |name: &str| -> Option<String> { files.get(name).map(|b| String::from_utf8_lossy(b).to_string()) };
        let wb = text("xl/workbook.xml").ok_or_else(|| rc_err(610, "xl/workbook.xml not found in workbook"))?;
        // relationships: id -> target
        let mut rels = BTreeMap::new();
        if let Some(r) = text("xl/_rels/workbook.xml.rels") {
            for e in xml_events(&r) {
                if let XmlEvent::Start(n, a, _) = e {
                    if n == "Relationship" {
                        if let (Some(id), Some(t)) = (attr(&a, "Id"), attr(&a, "Target")) {
                            let t = t.trim_start_matches('/');
                            let path = if t.starts_with("xl/") { t.to_string() } else { format!("xl/{t}") };
                            rels.insert(id.to_string(), path);
                        }
                    }
                }
            }
        }
        // shared strings
        let mut shared: Vec<String> = Vec::new();
        if let Some(s) = text("xl/sharedStrings.xml") {
            let mut cur: Option<String> = None;
            let mut in_t = false;
            let mut in_rph = false;
            for e in xml_events(&s) {
                match e {
                    XmlEvent::Start(n, _, sc) if n == "si" => {
                        cur = Some(String::new());
                        if sc {
                            shared.push(String::new());
                            cur = None;
                        }
                    }
                    XmlEvent::Start(n, _, sc) if n == "t" && !sc => in_t = true,
                    XmlEvent::Start(n, _, sc) if n == "rPh" && !sc => in_rph = true,
                    XmlEvent::End(n) if n == "rPh" => in_rph = false,
                    XmlEvent::End(n) if n == "t" => in_t = false,
                    XmlEvent::Text(t) if in_t && !in_rph => {
                        if let Some(c) = cur.as_mut() {
                            c.push_str(&t);
                        }
                    }
                    XmlEvent::End(n) if n == "si" => shared.push(cur.take().unwrap_or_default()),
                    _ => {}
                }
            }
        }
        // styles: which cellXfs indexes are dates
        let mut date_styles: Vec<bool> = Vec::new();
        if let Some(s) = text("xl/styles.xml") {
            let mut codes: BTreeMap<u32, String> = BTreeMap::new();
            let mut in_xfs = false;
            for e in xml_events(&s) {
                match e {
                    XmlEvent::Start(n, a, _) if n == "numFmt" => {
                        if let (Some(id), Some(c)) = (attr(&a, "numFmtId"), attr(&a, "formatCode")) {
                            codes.insert(id.parse().unwrap_or(0), c.to_string());
                        }
                    }
                    XmlEvent::Start(n, _, sc) if n == "cellXfs" && !sc => in_xfs = true,
                    XmlEvent::End(n) if n == "cellXfs" => in_xfs = false,
                    XmlEvent::Start(n, a, _) if n == "xf" && in_xfs => {
                        let id: u32 = attr(&a, "numFmtId").and_then(|v| v.parse().ok()).unwrap_or(0);
                        date_styles.push(is_date_format(id, codes.get(&id).map(String::as_str)));
                    }
                    _ => {}
                }
            }
        }
        let mut book = Workbook::default();
        for e in xml_events(&wb) {
            if let XmlEvent::Start(n, a, _) = e {
                if n == "sheet" {
                    let name = attr(&a, "name").unwrap_or("Sheet").to_string();
                    let path = attr(&a, "id").and_then(|id| rels.get(id)).cloned().unwrap_or_else(|| format!("xl/worksheets/sheet{}.xml", book.sheets.len() + 1));
                    let cells = match text(&path) {
                        Some(x) => parse_sheet(&x, &shared, &date_styles),
                        None => BTreeMap::new(),
                    };
                    book.sheets.push((name, cells));
                }
            }
        }
        Ok(book)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut files = Vec::new();
        let mut ct = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>"#);
        for i in 0..self.sheets.len() {
            ct.push_str(&format!(r#"<Override PartName="/xl/worksheets/sheet{}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#, i + 1));
        }
        ct.push_str("</Types>");
        files.push(("[Content_Types].xml".to_string(), ct.into_bytes()));
        files.push(("_rels/.rels".into(), r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#.as_bytes().to_vec()));
        // style 0 = general, style 1 = date (numFmtId 14)
        files.push(("xl/styles.xml".into(), r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="14" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs></styleSheet>"#.as_bytes().to_vec()));
        let mut wb = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>"#);
        let mut rels = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#);
        for (i, (name, cells)) in self.sheets.iter().enumerate() {
            wb.push_str(&format!(r#"<sheet name="{}" sheetId="{}" r:id="rId{}"/>"#, escape(name), i + 1, i + 1));
            rels.push_str(&format!(r#"<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{}.xml"/>"#, i + 1, i + 1));
            let mut sx = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>"#);
            let mut cur_row = 0;
            for ((r, c), v) in cells {
                if *r != cur_row {
                    if cur_row != 0 {
                        sx.push_str("</row>");
                    }
                    sx.push_str(&format!(r#"<row r="{r}">"#));
                    cur_row = *r;
                }
                let refn = format!("{}{}", col_letters(*c), r);
                match v {
                    Cell::Num(x) => sx.push_str(&format!(r#"<c r="{refn}"><v>{}</v></c>"#, x)),
                    Cell::Date(x) => sx.push_str(&format!(r#"<c r="{refn}" s="1"><v>{}</v></c>"#, x)),
                    Cell::Bool(b) => sx.push_str(&format!(r#"<c r="{refn}" t="b"><v>{}</v></c>"#, *b as u8)),
                    Cell::Str(s) => sx.push_str(&format!(r#"<c r="{refn}" t="inlineStr"><is><t xml:space="preserve">{}</t></is></c>"#, escape(s))),
                }
            }
            if cur_row != 0 {
                sx.push_str("</row>");
            }
            sx.push_str("</sheetData></worksheet>");
            files.push((format!("xl/worksheets/sheet{}.xml", i + 1), sx.into_bytes()));
        }
        wb.push_str("</sheets></workbook>");
        rels.push_str(r#"<Relationship Id="rIdS" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#);
        files.push(("xl/workbook.xml".into(), wb.into_bytes()));
        files.push(("xl/_rels/workbook.xml.rels".into(), rels.into_bytes()));
        zip(&files)
    }

    pub fn sheet_mut(&mut self, name: &str) -> &mut BTreeMap<(u32, u32), Cell> {
        if let Some(i) = self.sheets.iter().position(|(n, _)| n == name) {
            return &mut self.sheets[i].1;
        }
        self.sheets.push((name.to_string(), BTreeMap::new()));
        &mut self.sheets.last_mut().unwrap().1
    }
}

fn parse_sheet(xml: &str, shared: &[String], date_styles: &[bool]) -> BTreeMap<(u32, u32), Cell> {
    let mut cells = BTreeMap::new();
    let mut cur: Option<((u32, u32), String, bool)> = None; // pos, type, is_date
    let mut value = String::new();
    let mut in_v = false;
    let mut in_t = false;
    let mut row_no = 0u32;
    let mut col_no = 0u32;
    for e in xml_events(xml) {
        match e {
            XmlEvent::Start(n, a, _) if n == "row" => {
                row_no = attr(&a, "r").and_then(|r| r.parse().ok()).unwrap_or(row_no + 1);
                col_no = 0;
            }
            XmlEvent::Start(n, a, sc) if n == "c" => {
                let pos = attr(&a, "r").and_then(parse_ref).unwrap_or((row_no, col_no + 1));
                col_no = pos.1;
                let ty = attr(&a, "t").unwrap_or("n").to_string();
                let style: usize = attr(&a, "s").and_then(|s| s.parse().ok()).unwrap_or(0);
                let is_date = date_styles.get(style).copied().unwrap_or(false);
                value.clear();
                cur = if sc { None } else { Some((pos, ty, is_date)) };
            }
            XmlEvent::Start(n, _, sc) if n == "v" && !sc => in_v = true,
            XmlEvent::Start(n, _, sc) if n == "t" && !sc => in_t = true,
            XmlEvent::End(n) if n == "v" => in_v = false,
            XmlEvent::End(n) if n == "t" => in_t = false,
            XmlEvent::Text(t) if in_v || in_t => value.push_str(&t),
            XmlEvent::End(n) if n == "c" => {
                if let Some((pos, ty, is_date)) = cur.take() {
                    let v = value.trim();
                    let cell = match ty.as_str() {
                        "s" => v.parse::<usize>().ok().and_then(|i| shared.get(i)).map(|s| Cell::Str(s.clone())),
                        "str" | "inlineStr" => Some(Cell::Str(value.clone())),
                        "b" => Some(Cell::Bool(v == "1")),
                        "e" => None,
                        _ => v.parse::<f64>().ok().map(|x| if is_date { Cell::Date(x) } else { Cell::Num(x) }),
                    };
                    if let Some(c) = cell {
                        cells.insert(pos, c);
                    }
                }
            }
            _ => {}
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_dates_and_compression() {
        let mut b = Workbook::default();
        let s = b.sheet_mut("Data");
        s.insert((1, 1), Cell::Str("name & <id>".into()));
        s.insert((2, 1), Cell::Num(1.5));
        s.insert((2, 2), Cell::Date(36526.0));
        s.insert((3, 3), Cell::Bool(true));
        let back = Workbook::read(&b.to_bytes()).unwrap();
        let c = &back.sheets[0].1;
        assert_eq!(back.sheets[0].0, "Data");
        assert_eq!(c[&(1, 1)], Cell::Str("name & <id>".into()));
        assert_eq!(c[&(2, 1)], Cell::Num(1.5));
        assert_eq!(c[&(2, 2)], Cell::Date(36526.0));
        assert_eq!(c[&(3, 3)], Cell::Bool(true));
        assert_eq!(excel_to_td(36526.0), 14610.0); // 01jan2000
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn date_format_detection() {
        assert!(is_date_format(14, None));
        assert!(is_date_format(164, Some("dd/mm/yyyy")));
        assert!(!is_date_format(164, Some("0.00%")));
        assert!(!is_date_format(164, Some("\"days\" 0")));
    }
}
