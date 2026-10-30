//! `putexcel` extras: `cell = formula(...)` (stored with its computed value
//! when the formula uses the common arithmetic and SUM/AVERAGE/MIN/MAX/
//! COUNT/ROUND/ABS/SQRT/EXP/LN functions), `cell = image(file)` (PNG/JPEG
//! pictures anchored at the cell) and the `open` / `save` / `close` mode of
//! `putexcel set`.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::xlsx::{col_letters, escape, parse_ref, unzip, zip, Cell, Workbook};
use ferrostat_core::vfs;

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

/// Formulas and pictures of one workbook, by sheet name.
#[derive(Debug, Clone, Default)]
pub struct Extras {
    pub formulas: BTreeMap<String, Vec<((u32, u32), String)>>,
    pub images: BTreeMap<String, Vec<Picture>>,
}

#[derive(Debug, Clone)]
pub struct Picture {
    pub at: (u32, u32),
    pub ext: String,
    pub data: Vec<u8>,
    /// size in pixels
    pub size: (u32, u32),
}

/// Handle `formula(...)` and `image(...)` values; `None` for other values.
pub fn special(me: &mut AdoInterpreter, path: &str, sheet: &str, at: (u32, u32), v: &str) -> Result<Option<Vec<((u32, u32), Cell)>>, String> {
    let lower = v.to_lowercase();
    if let Some(inner) = (lower.starts_with("formula(") && v.ends_with(')')).then(|| &v[8..v.len() - 1]) {
        let f = unquote(inner.trim());
        let f = f.strip_prefix('=').unwrap_or(&f).trim().to_string();
        if f.is_empty() {
            return Err(rc_err(198, "formula() requires a formula"));
        }
        let ex = me.putexcel.extras.entry(path.to_string()).or_default();
        let list = ex.formulas.entry(sheet.to_string()).or_default();
        list.retain(|(p, _)| *p != at);
        list.push((at, f));
        return Ok(Some(vec![(at, Cell::Num(0.0))]));
    }
    let img = ["image(", "picture("].iter().find(|p| lower.starts_with(**p)).map(|p| p.len());
    if let Some(n) = img.filter(|_| v.ends_with(')')) {
        let file = unquote(v[n..v.len() - 1].trim());
        let data = vfs::read(&file).map_err(|_| rc_err(601, format!("file {file} not found")))?;
        let ext = if data.starts_with(&[0x89, b'P', b'N', b'G']) {
            "png"
        } else if data.starts_with(&[0xFF, 0xD8]) {
            "jpeg"
        } else {
            return Err(rc_err(198, format!("{file}: only PNG and JPEG images can be placed")));
        };
        let size = crate::docx::image_size(&data).unwrap_or((400, 300));
        let ex = me.putexcel.extras.entry(path.to_string()).or_default();
        let list = ex.images.entry(sheet.to_string()).or_default();
        list.retain(|p| p.at != at);
        list.push(Picture { at, ext: ext.into(), data, size });
        return Ok(Some(Vec::new()));
    }
    Ok(None)
}

/// A cell that is overwritten by a plain value loses its formula.
pub fn forget(me: &mut AdoInterpreter, path: &str, sheet: &str, at: (u32, u32)) {
    if let Some(l) = me.putexcel.extras.get_mut(path).and_then(|e| e.formulas.get_mut(sheet)) {
        l.retain(|(p, _)| *p != at);
    }
}

/// Serialise the workbook with formulas and pictures.
pub fn to_bytes(book: &mut Workbook, ex: Option<&Extras>) -> Vec<u8> {
    let Some(ex) = ex.filter(|e| !e.formulas.is_empty() || !e.images.is_empty()) else { return book.to_bytes() };
    // computed values (twice, so formulas may use other formulas)
    for _ in 0..2 {
        for (sheet, list) in &ex.formulas {
            for (at, f) in list {
                let cells = book.sheet_mut(sheet);
                let v = eval_formula(f, cells);
                cells.insert(*at, match v {
                    Some(Val::N(x)) => Cell::Num(x),
                    Some(Val::S(s)) => Cell::Str(s),
                    None => Cell::Num(0.0),
                });
            }
        }
    }
    let base = book.to_bytes();
    let Ok(parts) = unzip(&base) else { return base };
    let mut parts: Vec<(String, Vec<u8>)> = parts.into_iter().collect();
    parts.sort_by_key(|(n, _)| n != "[Content_Types].xml");
    let mut ct_extra = String::new();
    let mut new_parts: Vec<(String, Vec<u8>)> = Vec::new();
    let mut n_drawing = 0;
    let mut n_media = 0;
    let mut exts: Vec<String> = Vec::new();
    for (i, (sheet, _)) in book.sheets.iter().enumerate() {
        let part = format!("xl/worksheets/sheet{}.xml", i + 1);
        let Some(pos) = parts.iter().position(|(n, _)| *n == part) else { continue };
        let mut xml = String::from_utf8_lossy(&parts[pos].1).to_string();
        if let Some(list) = ex.formulas.get(sheet) {
            for (at, f) in list {
                xml = put_formula(&xml, *at, f);
            }
        }
        if let Some(pics) = ex.images.get(sheet).filter(|p| !p.is_empty()) {
            n_drawing += 1;
            let mut anchors = String::new();
            let mut rels = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#);
            for (k, p) in pics.iter().enumerate() {
                n_media += 1;
                let media = format!("image{n_media}.{}", p.ext);
                if !exts.contains(&p.ext) {
                    exts.push(p.ext.clone());
                }
                new_parts.push((format!("xl/media/{media}"), p.data.clone()));
                rels.push_str(&format!(r#"<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media}"/>"#, k + 1));
                let (cx, cy) = (p.size.0 as u64 * 9525, p.size.1 as u64 * 9525);
                anchors.push_str(&format!(
                    r#"<xdr:oneCellAnchor><xdr:from><xdr:col>{}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from><xdr:ext cx="{cx}" cy="{cy}"/><xdr:pic><xdr:nvPicPr><xdr:cNvPr id="{}" name="Picture {}"/><xdr:cNvPicPr><a:picLocks noChangeAspect="1"/></xdr:cNvPicPr></xdr:nvPicPr><xdr:blipFill><a:blip r:embed="rId{}"/><a:stretch><a:fillRect/></a:stretch></xdr:blipFill><xdr:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></xdr:spPr></xdr:pic><xdr:clientData/></xdr:oneCellAnchor>"#,
                    p.at.1 - 1,
                    p.at.0 - 1,
                    k + 2,
                    k + 1,
                    k + 1
                ));
            }
            rels.push_str("</Relationships>");
            let drawing = format!(
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">{anchors}</xdr:wsDr>"#
            );
            new_parts.push((format!("xl/drawings/drawing{n_drawing}.xml"), drawing.into_bytes()));
            new_parts.push((format!("xl/drawings/_rels/drawing{n_drawing}.xml.rels"), rels.into_bytes()));
            new_parts.push((
                format!("xl/worksheets/_rels/sheet{}.xml.rels", i + 1),
                format!(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdDr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing{n_drawing}.xml"/></Relationships>"#).into_bytes(),
            ));
            ct_extra.push_str(&format!(r#"<Override PartName="/xl/drawings/drawing{n_drawing}.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>"#));
            if !xml.contains("xmlns:r=") {
                xml = xml.replacen("<worksheet ", r#"<worksheet xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" "#, 1);
            }
            xml = xml.replacen("</worksheet>", r#"<drawing r:id="rIdDr1"/></worksheet>"#, 1);
        }
        parts[pos].1 = xml.into_bytes();
    }
    for e in &exts {
        let mime = if e == "png" { "image/png" } else { "image/jpeg" };
        ct_extra.push_str(&format!(r#"<Default Extension="{e}" ContentType="{mime}"/>"#));
    }
    if !ct_extra.is_empty() {
        if let Some((_, ct)) = parts.iter_mut().find(|(n, _)| n == "[Content_Types].xml") {
            let s = String::from_utf8_lossy(ct).replacen("</Types>", &format!("{ct_extra}</Types>"), 1);
            *ct = s.into_bytes();
        }
    }
    parts.extend(new_parts);
    zip(&parts)
}

/// Put `<f>` into the cell element of `at`.
fn put_formula(xml: &str, at: (u32, u32), f: &str) -> String {
    let r = format!("{}{}", col_letters(at.1), at.0);
    let key = format!("<c r=\"{r}\"");
    let Some(start) = xml.find(&key) else { return xml.to_string() };
    let after = &xml[start..];
    let tag_end = after.find('>').unwrap_or(after.len());
    let self_closing = after[..tag_end].ends_with('/');
    let end = if self_closing { start + tag_end + 1 } else { start + after.find("</c>").map_or(tag_end + 1, |k| k + 4) };
    let elem = &xml[start..end];
    let open = &elem[..tag_end.min(elem.len())].trim_end_matches('/');
    let s_attr = open.find(" s=\"").map(|k| {
        let rest = &open[k + 4..];
        format!(" s=\"{}\"", &rest[..rest.find('"').unwrap_or(0)])
    });
    let value = elem.find("<v>").and_then(|a| elem[a + 3..].find("</v>").map(|b| elem[a + 3..a + 3 + b].to_string()));
    let text = elem.find("<t xml:space=\"preserve\">").and_then(|a| elem[a + 24..].find("</t>").map(|b| elem[a + 24..a + 24 + b].to_string()));
    let new = match (value, text) {
        (_, Some(t)) => format!("<c r=\"{r}\"{} t=\"str\"><f>{}</f><v>{t}</v></c>", s_attr.unwrap_or_default(), escape(f)),
        (Some(v), None) => format!("<c r=\"{r}\"{}><f>{}</f><v>{v}</v></c>", s_attr.unwrap_or_default(), escape(f)),
        (None, None) => format!("<c r=\"{r}\"{}><f>{}</f></c>", s_attr.unwrap_or_default(), escape(f)),
    };
    format!("{}{}{}", &xml[..start], new, &xml[end..])
}

// ---------------------------------------------------------------------------
// A small formula evaluator for cached values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    N(f64),
    S(String),
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    cells: &'a BTreeMap<(u32, u32), Cell>,
}

fn cell_num(c: Option<&Cell>) -> Option<f64> {
    match c {
        Some(Cell::Num(x)) | Some(Cell::Date(x)) => Some(*x),
        Some(Cell::Bool(b)) => Some(*b as u8 as f64),
        _ => None,
    }
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i] == b' ' {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn expr(&mut self) -> Option<Val> {
        let mut a = self.term()?;
        while let Some(c) = self.peek() {
            match c {
                b'+' | b'-' => {
                    self.i += 1;
                    let b = num(&self.term()?)?;
                    let x = num(&a)?;
                    a = Val::N(if c == b'+' { x + b } else { x - b });
                }
                b'&' => {
                    self.i += 1;
                    let b = self.term()?;
                    a = Val::S(format!("{}{}", txt(&a), txt(&b)));
                }
                _ => break,
            }
        }
        Some(a)
    }
    fn term(&mut self) -> Option<Val> {
        let mut a = self.power()?;
        while let Some(c @ (b'*' | b'/')) = self.peek() {
            self.i += 1;
            let b = num(&self.power()?)?;
            let x = num(&a)?;
            a = Val::N(if c == b'*' { x * b } else if b == 0.0 { return None } else { x / b });
        }
        Some(a)
    }
    fn power(&mut self) -> Option<Val> {
        let a = self.unary()?;
        if self.peek() == Some(b'^') {
            self.i += 1;
            let b = num(&self.power()?)?;
            return Some(Val::N(num(&a)?.powf(b)));
        }
        Some(a)
    }
    fn unary(&mut self) -> Option<Val> {
        match self.peek()? {
            b'-' => {
                self.i += 1;
                Some(Val::N(-num(&self.unary()?)?))
            }
            b'+' => {
                self.i += 1;
                self.unary()
            }
            _ => self.primary(),
        }
    }
    /// Values of an argument: a range gives several.
    fn arg(&mut self) -> Option<Vec<f64>> {
        self.ws();
        let save = self.i;
        if let Some(((r0, c0), (r1, c1))) = self.range() {
            if (r0, c0) != (r1, c1) {
                let mut v = Vec::new();
                for r in r0.min(r1)..=r0.max(r1) {
                    for c in c0.min(c1)..=c0.max(c1) {
                        if let Some(x) = cell_num(self.cells.get(&(r, c))) {
                            v.push(x);
                        }
                    }
                }
                return Some(v);
            }
        }
        self.i = save;
        Some(vec![num(&self.expr()?)?])
    }
    fn reference(&mut self) -> Option<(u32, u32)> {
        let st = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'$') {
            self.i += 1;
        }
        let w = std::str::from_utf8(&self.s[st..self.i]).ok()?;
        match parse_ref(w) {
            Some(p) if w.chars().any(|c| c.is_ascii_digit()) => Some(p),
            _ => {
                self.i = st;
                None
            }
        }
    }
    fn range(&mut self) -> Option<((u32, u32), (u32, u32))> {
        let a = self.reference()?;
        if self.s.get(self.i) == Some(&b':') {
            self.i += 1;
            let b = self.reference()?;
            return Some((a, b));
        }
        Some((a, a))
    }
    fn primary(&mut self) -> Option<Val> {
        let c = self.peek()?;
        if c == b'(' {
            self.i += 1;
            let v = self.expr()?;
            if self.peek() != Some(b')') {
                return None;
            }
            self.i += 1;
            return Some(v);
        }
        if c == b'"' {
            self.i += 1;
            let st = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' {
                self.i += 1;
            }
            let t = String::from_utf8_lossy(&self.s[st..self.i]).to_string();
            self.i += 1;
            return Some(Val::S(t));
        }
        if c.is_ascii_digit() || c == b'.' {
            let st = self.i;
            while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.' || ((self.s[self.i] == b'e' || self.s[self.i] == b'E') && self.i > st)) {
                self.i += 1;
            }
            return std::str::from_utf8(&self.s[st..self.i]).ok()?.parse().ok().map(Val::N);
        }
        // function call or reference
        let st = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'$' || self.s[self.i] == b'.') {
            self.i += 1;
        }
        let w = std::str::from_utf8(&self.s[st..self.i]).ok()?.to_uppercase();
        if self.peek() == Some(b'(') {
            self.i += 1;
            let mut vals: Vec<Vec<f64>> = Vec::new();
            if self.peek() != Some(b')') {
                loop {
                    vals.push(self.arg()?);
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b')') => break,
                        _ => return None,
                    }
                }
            }
            self.i += 1;
            let all: Vec<f64> = vals.iter().flatten().copied().collect();
            let one = |k: usize| vals.get(k).and_then(|v| v.first()).copied();
            return Some(Val::N(match w.as_str() {
                "SUM" => all.iter().sum(),
                "AVERAGE" if !all.is_empty() => all.iter().sum::<f64>() / all.len() as f64,
                "MIN" if all.is_empty() => 0.0,
                "MAX" if all.is_empty() => 0.0,
                "MIN" => all.iter().copied().fold(f64::INFINITY, f64::min),
                "MAX" => all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "COUNT" => all.len() as f64,
                "ABS" => one(0)?.abs(),
                "SQRT" if one(0)? >= 0.0 => one(0)?.sqrt(),
                "EXP" => one(0)?.exp(),
                "LN" if one(0)? > 0.0 => one(0)?.ln(),
                "ROUND" => {
                    let d = one(1).unwrap_or(0.0);
                    let m = 10f64.powf(d);
                    (one(0)? * m).round() / m
                }
                _ => return None,
            }));
        }
        let p = parse_ref(&w).filter(|_| w.chars().any(|c| c.is_ascii_digit()))?;
        match self.cells.get(&p) {
            Some(Cell::Str(s)) => Some(Val::S(s.clone())),
            c => Some(Val::N(cell_num(c).unwrap_or(0.0))),
        }
    }
}

fn num(v: &Val) -> Option<f64> {
    match v {
        Val::N(x) => Some(*x),
        Val::S(s) => s.trim().parse().ok(),
    }
}

fn txt(v: &Val) -> String {
    match v {
        Val::N(x) => ferrostat_cmd::format::display_number(*x).trim().to_string(),
        Val::S(s) => s.clone(),
    }
}

/// Value of a formula over the cells of its sheet; `None` when it uses
/// something the evaluator does not know.
pub fn eval_formula(f: &str, cells: &BTreeMap<(u32, u32), Cell>) -> Option<Val> {
    let mut p = P { s: f.as_bytes(), i: 0, cells };
    let v = p.expr()?;
    p.ws();
    (p.i == p.s.len()).then_some(v).filter(|v| !matches!(v, Val::N(x) if !x.is_finite()))
}

// ---------------------------------------------------------------------------
// open / save / close
// ---------------------------------------------------------------------------

/// Write the workbook of `path` (with its formulas and pictures).
pub fn write(me: &mut AdoInterpreter, path: &str) -> Result<(), String> {
    let ex = me.putexcel.extras.get(path).cloned();
    let book = me.putexcel.books.entry(path.to_string()).or_default();
    let bytes = to_bytes(book, ex.as_ref());
    vfs::write(path, &bytes).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))
}

/// `putexcel save` and `putexcel close`.
pub fn save_close(me: &mut AdoInterpreter, save: bool) -> Result<String, String> {
    let Some(path) = me.putexcel.path.clone() else {
        return if save { Err(rc_err(198, "putexcel set required first")) } else { Ok(String::new()) };
    };
    if save {
        write(me, &path)?;
    }
    me.putexcel.open = false;
    if !save {
        // unsaved changes are dropped
        me.putexcel.books.remove(&path);
        me.putexcel.extras.remove(&path);
        me.putexcel.path = None;
        return Ok(String::new());
    }
    Ok(format!("file {path} saved"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formulas() {
        let mut c = BTreeMap::new();
        c.insert((1, 2), Cell::Num(2.0));
        c.insert((2, 2), Cell::Num(3.0));
        c.insert((3, 2), Cell::Str("x".into()));
        assert_eq!(eval_formula("SUM(B1:B3)", &c), Some(Val::N(5.0)));
        assert_eq!(eval_formula("B1*B2+1", &c), Some(Val::N(7.0)));
        assert_eq!(eval_formula("AVERAGE(B1:B2)", &c), Some(Val::N(2.5)));
        assert_eq!(eval_formula("ROUND(B2/B1, 1)", &c), Some(Val::N(1.5)));
        assert_eq!(eval_formula("B3&\"y\"", &c), Some(Val::S("xy".into())));
        assert_eq!(eval_formula("VLOOKUP(1,A1:B2,2)", &c), None);
        let _ = col_letters(1);
    }
}
