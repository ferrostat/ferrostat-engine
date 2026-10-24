//! `putexcel`: writing cells, expressions and matrices to .xlsx workbooks.
//! The workbook is written as an uncompressed (stored) ZIP of SpreadsheetML
//! parts; files previously written this way can be reopened with `modify`.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matcmd::eval_matrix;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Num(f64),
    Str(String),
}

#[derive(Debug, Clone, Default)]
pub struct Workbook {
    pub sheets: Vec<(String, BTreeMap<(u32, u32), Cell>)>,
}

#[derive(Debug, Clone, Default)]
pub struct PutExcel {
    pub path: Option<String>,
    pub sheet: String,
    pub books: BTreeMap<String, Workbook>,
}

/// "B12" -> (row 12, col 2), 1-based.
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
        let r = ((c - 1) % 26) as u8;
        s.push((b'A' + r) as char);
        c = (c - 1) / 26;
    }
    s.iter().rev().collect()
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn zip_stored(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let offset = out.len() as u32;
        let crc = crc32(data);
        let n = name.as_bytes();
        out.extend_from_slice(&0x04034b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(n);
        out.extend_from_slice(data);
        central.extend_from_slice(&0x02014b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
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

fn unzip_stored(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut p = 0;
    while p + 30 <= data.len() && data[p..p + 4] == 0x04034b50u32.to_le_bytes() {
        let method = u16::from_le_bytes([data[p + 8], data[p + 9]]);
        let size = u32::from_le_bytes(data[p + 18..p + 22].try_into().unwrap()) as usize;
        let nlen = u16::from_le_bytes([data[p + 26], data[p + 27]]) as usize;
        let elen = u16::from_le_bytes([data[p + 28], data[p + 29]]) as usize;
        let name = String::from_utf8_lossy(&data[p + 30..p + 30 + nlen]).to_string();
        let start = p + 30 + nlen + elen;
        if method != 0 || start + size > data.len() {
            break;
        }
        out.push((name, data[start..start + size].to_vec()));
        p = start + size;
    }
    out
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&amp;", "&")
}

impl Workbook {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut files = Vec::new();
        let mut ct = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>"#);
        for i in 0..self.sheets.len() {
            ct.push_str(&format!(r#"<Override PartName="/xl/worksheets/sheet{}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#, i + 1));
        }
        ct.push_str("</Types>");
        files.push(("[Content_Types].xml".to_string(), ct.into_bytes()));
        files.push(("_rels/.rels".into(), r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#.as_bytes().to_vec()));
        let mut wb = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>"#);
        let mut rels = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#);
        for (i, (name, cells)) in self.sheets.iter().enumerate() {
            wb.push_str(&format!(r#"<sheet name="{}" sheetId="{}" r:id="rId{}"/>"#, xml_escape(name), i + 1, i + 1));
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
                    Cell::Str(s) => sx.push_str(&format!(r#"<c r="{refn}" t="inlineStr"><is><t xml:space="preserve">{}</t></is></c>"#, xml_escape(s))),
                }
            }
            if cur_row != 0 {
                sx.push_str("</row>");
            }
            sx.push_str("</sheetData></worksheet>");
            files.push((format!("xl/worksheets/sheet{}.xml", i + 1), sx.into_bytes()));
        }
        wb.push_str("</sheets></workbook>");
        rels.push_str("</Relationships>");
        files.push(("xl/workbook.xml".into(), wb.into_bytes()));
        files.push(("xl/_rels/workbook.xml.rels".into(), rels.into_bytes()));
        zip_stored(&files)
    }

    /// Read a workbook previously written by Ferrostat.
    pub fn from_bytes(data: &[u8]) -> Option<Workbook> {
        let files = unzip_stored(data);
        let wb = files.iter().find(|(n, _)| n == "xl/workbook.xml")?;
        let wbs = String::from_utf8_lossy(&wb.1).to_string();
        let mut book = Workbook::default();
        for (i, part) in wbs.split("<sheet ").skip(1).enumerate() {
            let name = part.split("name=\"").nth(1)?.split('"').next()?.to_string();
            let sheet = files.iter().find(|(n, _)| *n == format!("xl/worksheets/sheet{}.xml", i + 1))?;
            let sx = String::from_utf8_lossy(&sheet.1).to_string();
            let mut cells = BTreeMap::new();
            for c in sx.split("<c r=\"").skip(1) {
                let refn = c.split('"').next()?;
                let pos = parse_ref(refn)?;
                if c.contains("inlineStr") {
                    let t = c.split("<t xml:space=\"preserve\">").nth(1).or_else(|| c.split("<t>").nth(1))?.split("</t>").next()?;
                    cells.insert(pos, Cell::Str(xml_unescape(t)));
                } else if let Some(v) = c.split("<v>").nth(1) {
                    let v = v.split("</v>").next()?;
                    cells.insert(pos, Cell::Num(v.parse().ok()?));
                }
            }
            book.sheets.push((xml_unescape(&name), cells));
        }
        Some(book)
    }
}

fn opt_arg(opts: &[(String, Option<String>)], name: &str) -> Option<String> {
    opts.iter().find(|(n, _)| n.to_lowercase().starts_with(name)).and_then(|(_, a)| a.clone())
}

fn has_opt(opts: &[(String, Option<String>)], name: &str) -> bool {
    opts.iter().any(|(n, _)| n.to_lowercase() == name)
}

pub fn putexcel(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    };
    let first = main.split_whitespace().next().unwrap_or("").to_lowercase();
    if first == "set" {
        let path = unquote(main[3..].trim());
        let path = if path.ends_with(".xlsx") || path.ends_with(".xls") { path } else { format!("{path}.xlsx") };
        let replace = has_opt(&opts, "replace");
        let mut book = if replace {
            Workbook::default()
        } else if let Some(b) = me.putexcel.books.get(&path) {
            b.clone()
        } else if let Ok(bytes) = std::fs::read(&path) {
            Workbook::from_bytes(&bytes).ok_or_else(|| rc_err(603, format!("{path} could not be read; use replace")))?
        } else {
            Workbook::default()
        };
        let (sheet, sheet_replace) = match opt_arg(&opts, "sheet") {
            Some(s) => {
                let parts: Vec<String> = s.split(',').map(|x| x.trim().to_string()).collect();
                (unquote(&parts[0]), parts.get(1).map_or(false, |x| x.starts_with("replace")))
            }
            None => ("Sheet1".to_string(), false),
        };
        match book.sheets.iter_mut().find(|(n, _)| *n == sheet) {
            Some((_, cells)) => {
                if sheet_replace {
                    cells.clear();
                }
            }
            None => book.sheets.push((sheet.clone(), BTreeMap::new())),
        }
        me.putexcel.path = Some(path.clone());
        me.putexcel.sheet = sheet;
        me.putexcel.books.insert(path, book);
        return Ok(String::new());
    }
    if first == "clear" {
        me.putexcel.path = None;
        return Ok(String::new());
    }
    if first == "describe" {
        return Ok(format!("{}", me.putexcel.path.clone().unwrap_or_default()));
    }
    let path = me.putexcel.path.clone().ok_or_else(|| rc_err(198, "putexcel set required first"))?;
    // assignments: CELL = item ...
    let words = split_top_level_words(&main);
    let mut i = 0;
    let mut writes: Vec<((u32, u32), Cell)> = Vec::new();
    while i < words.len() {
        let target = words[i].clone();
        let (cellref, after_eq) = if let Some((a, b)) = target.split_once('=') {
            (a.to_string(), b.to_string())
        } else {
            if words.get(i + 1).map(String::as_str) != Some("=") {
                return Err(rc_err(198, format!("{target}: = expected")));
            }
            i += 1;
            (target, String::new())
        };
        let cell = cellref.split(':').next().unwrap();
        let (row, col) = parse_ref(cell).ok_or_else(|| rc_err(198, format!("{cell} invalid cell")))?;
        // the value is the next word (or the text after '=')
        let value = if !after_eq.is_empty() {
            after_eq
        } else {
            i += 1;
            words.get(i).cloned().ok_or_else(|| rc_err(198, "value expected"))?
        };
        i += 1;
        // swallow further words that belong to an expression until the next CELL=
        let mut value = value;
        while i < words.len() && !(parse_ref(words[i].split('=').next().unwrap()).is_some() && (words[i].contains('=') || words.get(i + 1).map(String::as_str) == Some("="))) {
            value.push(' ');
            value.push_str(&words[i]);
            i += 1;
        }
        let v = value.trim();
        if let Some(inner) = v.strip_prefix("matrix(").and_then(|x| x.strip_suffix(')')) {
            let m = eval_matrix(&me.engine.dataset, &me.engine.store, inner)?;
            let names = has_opt(&opts, "names");
            let rown = names || has_opt(&opts, "rownames");
            let coln = names || has_opt(&opts, "colnames");
            let (r0, c0) = (row + coln as u32, col + rown as u32);
            if coln {
                for (j, cn) in m.colnames.iter().enumerate() {
                    writes.push(((row, c0 + j as u32), Cell::Str(cn.clone())));
                }
            }
            if rown {
                for (k, rn) in m.rownames.iter().enumerate() {
                    writes.push(((r0 + k as u32, col), Cell::Str(rn.clone())));
                }
            }
            for a in 0..m.rows {
                for b in 0..m.cols {
                    let x = m[(a, b)];
                    writes.push(((r0 + a as u32, c0 + b as u32), if ferrostat_core::is_missing(x) { Cell::Str(String::new()) } else { Cell::Num(x) }));
                }
            }
        } else if v.starts_with('"') || v.starts_with("`\"") {
            writes.push(((row, col), Cell::Str(unquote(v))));
        } else if v.starts_with("image(") || v.starts_with("picture(") {
            // images are not embedded; nothing to write
        } else {
            let e = parse_expression(v).map_err(|e| rc_err(198, e.message))?;
            match Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)? {
                Val::Num(x) => writes.push(((row, col), Cell::Num(x))),
                Val::Str(s) => writes.push(((row, col), Cell::Str(s))),
            }
        }
    }
    let sheet = me.putexcel.sheet.clone();
    let book = me.putexcel.books.entry(path.clone()).or_default();
    let cells = match book.sheets.iter_mut().find(|(n, _)| *n == sheet) {
        Some((_, c)) => c,
        None => {
            book.sheets.push((sheet.clone(), BTreeMap::new()));
            &mut book.sheets.last_mut().unwrap().1
        }
    };
    for (pos, c) in writes {
        cells.insert(pos, c);
    }
    std::fs::write(&path, book.to_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("file {path} saved"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_and_roundtrip() {
        assert_eq!(parse_ref("A1"), Some((1, 1)));
        assert_eq!(parse_ref("AB12"), Some((12, 28)));
        assert_eq!(col_letters(28), "AB");
        let mut b = Workbook::default();
        let mut cells = BTreeMap::new();
        cells.insert((1, 1), Cell::Str("a <b>".into()));
        cells.insert((2, 3), Cell::Num(1.5));
        b.sheets.push(("Table 1".into(), cells));
        let back = Workbook::from_bytes(&b.to_bytes()).unwrap();
        assert_eq!(back.sheets[0].0, "Table 1");
        assert_eq!(back.sheets[0].1.get(&(2, 3)), Some(&Cell::Num(1.5)));
        assert_eq!(back.sheets[0].1.get(&(1, 1)), Some(&Cell::Str("a <b>".into())));
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }
}
