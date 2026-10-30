//! Reading a .docx file (WordprocessingML) into the document model of
//! `crate::docx`: paragraphs with styled runs, line and page breaks, tabs,
//! tables, inline images and the page size.

use std::collections::BTreeMap;

use ferrostat_cmd::xlsx::{unzip, xml_events, XmlEvent};

use crate::docx::{Block, Cell, Docx, Inline, Paragraph, Run, Table};

type Attrs = Vec<(String, String)>;

fn attr<'a>(a: &'a Attrs, k: &str) -> Option<&'a str> {
    a.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

/// On/off property such as `<w:b/>` or `<w:b w:val="0"/>`.
fn flag(a: &Attrs) -> bool {
    !matches!(attr(a, "val"), Some("0" | "false" | "off" | "none"))
}

/// Parse the bytes of a .docx file.
pub fn read_docx(data: &[u8]) -> Result<Docx, String> {
    let parts = unzip(data).map_err(|_| "not a valid .docx (ZIP) file".to_string())?;
    let doc = parts.get("word/document.xml").ok_or("not a .docx file: word/document.xml missing")?;
    let doc = String::from_utf8_lossy(doc).to_string();
    // relationship id -> media target
    let mut rels: BTreeMap<String, String> = BTreeMap::new();
    if let Some(r) = parts.get("word/_rels/document.xml.rels") {
        for ev in xml_events(&String::from_utf8_lossy(r)) {
            if let XmlEvent::Start(n, a, _) = ev {
                if n == "Relationship" {
                    if let (Some(id), Some(t)) = (attr(&a, "Id"), attr(&a, "Target")) {
                        rels.insert(id.to_string(), t.to_string());
                    }
                }
            }
        }
    }
    let mut d = Docx::new();
    let mut media_index: BTreeMap<String, usize> = BTreeMap::new();
    let mut st = State::default();
    for ev in xml_events(&doc) {
        match ev {
            XmlEvent::Start(name, a, empty) => {
                st.start(&name, &a, empty, &mut d, &rels, &parts, &mut media_index);
                if empty {
                    st.end(&name, &mut d);
                }
            }
            XmlEvent::End(name) => st.end(&name, &mut d),
            XmlEvent::Text(t) => {
                if st.in_t {
                    st.run.text.push_str(&t);
                }
            }
        }
    }
    Ok(d)
}

#[derive(Default)]
struct State {
    /// Paragraph being read.
    para: Option<Paragraph>,
    run: Run,
    in_run: bool,
    in_rpr: bool,
    in_ppr: bool,
    in_t: bool,
    /// Tables being read (nested tables are flattened into their cell).
    tables: Vec<TableState>,
    /// Image extent of the drawing being read.
    extent: (u64, u64),
    in_sect: bool,
}

#[derive(Default)]
struct TableState {
    rows: Vec<Vec<Cell>>,
    row: Vec<Cell>,
    cell: Option<(Vec<String>, bool, bool, Option<String>, usize)>,
}

impl State {
    #[allow(clippy::too_many_arguments)]
    fn start(&mut self, name: &str, a: &Attrs, _empty: bool, d: &mut Docx, rels: &BTreeMap<String, String>, parts: &BTreeMap<String, Vec<u8>>, media_index: &mut BTreeMap<String, usize>) {
        match name {
            "p" => {
                self.para = Some(Paragraph::default());
            }
            "pPr" => self.in_ppr = true,
            "pStyle" if self.in_ppr => {
                if let Some(p) = &mut self.para {
                    p.style = attr(a, "val").map(String::from);
                }
            }
            "jc" if self.in_ppr => {
                if let Some(p) = &mut self.para {
                    p.halign = attr(a, "val").map(|v| match v {
                        "center" => "center".to_string(),
                        "right" | "end" => "right".to_string(),
                        "both" | "distribute" => "both".to_string(),
                        _ => "left".to_string(),
                    });
                }
            }
            "r" => {
                self.in_run = true;
                self.run = Run::default();
            }
            "rPr" if self.in_run => self.in_rpr = true,
            "b" if self.in_rpr => self.run.bold = flag(a),
            "i" if self.in_rpr => self.run.italic = flag(a),
            "u" if self.in_rpr => self.run.underline = flag(a),
            "color" if self.in_rpr => self.run.color = attr(a, "val").filter(|v| v.len() == 6).map(String::from),
            "sz" if self.in_rpr => self.run.size = attr(a, "val").and_then(|v| v.parse::<f64>().ok()).map(|x| x / 2.0),
            "rFonts" if self.in_rpr => {
                let f = attr(a, "ascii").or_else(|| attr(a, "hAnsi")).map(String::from);
                if f.as_deref().is_some_and(|f| f.to_lowercase().contains("courier")) {
                    self.run.code = true;
                }
                self.run.font = f;
            }
            "t" if self.in_run => self.in_t = true,
            "tab" if self.in_run && !self.in_rpr => self.run.text.push('\t'),
            "br" if self.in_run => {
                if attr(a, "type") == Some("page") {
                    self.flush_run();
                    self.flush_para(d);
                    if self.tables.is_empty() {
                        d.blocks.push(Block::PageBreak);
                    }
                    self.para = Some(Paragraph::default());
                } else {
                    self.run.text.push('\n');
                }
            }
            "cr" if self.in_run => self.run.text.push('\n'),
            "extent" => {
                let g = |k: &str| attr(a, k).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
                self.extent = (g("cx"), g("cy"));
            }
            "blip" => {
                let Some(id) = attr(a, "embed") else { return };
                let Some(target) = rels.get(id) else { return };
                let key = if let Some(t) = target.strip_prefix('/') { t.to_string() } else { format!("word/{target}") };
                let Some(bytes) = parts.get(&key) else { return };
                let idx = *media_index.entry(key.clone()).or_insert_with(|| {
                    d.media.push((key.rsplit('/').next().unwrap_or("image.png").to_string(), bytes.clone()));
                    d.media.len() - 1
                });
                self.flush_run();
                if self.tables.is_empty() {
                    let (cx, cy) = self.extent;
                    let img = Inline::Image { media: idx, cx: cx.max(1), cy: cy.max(1) };
                    self.para.get_or_insert_with(Paragraph::default).items.push(img);
                }
            }
            "tbl" => {
                self.flush_run();
                if self.tables.is_empty() {
                    // text before the table in an enclosing paragraph (rare)
                    self.flush_para(d);
                }
                self.tables.push(TableState::default());
            }
            "tr" => {
                if let Some(t) = self.tables.last_mut() {
                    t.row.clear();
                }
            }
            "tc" => {
                if let Some(t) = self.tables.last_mut() {
                    t.cell = Some((Vec::new(), true, true, None, 1));
                }
            }
            "gridSpan" => {
                if let Some(Some(c)) = self.tables.last_mut().map(|t| &mut t.cell) {
                    c.4 = attr(a, "val").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
                }
            }
            "pgSz" if self.in_sect => {
                let g = |k: &str| attr(a, k).and_then(|v| v.parse::<u32>().ok());
                if let (Some(w), Some(h)) = (g("w"), g("h")) {
                    if attr(a, "orient") == Some("landscape") || w > h {
                        d.landscape = true;
                        d.page = (h, w);
                    } else {
                        d.page = (w, h);
                    }
                }
            }
            "sectPr" => self.in_sect = true,
            _ => {}
        }
    }

    fn flush_run(&mut self) {
        if !self.in_run && self.run.text.is_empty() {
            return;
        }
        let run = std::mem::take(&mut self.run);
        if run.text.is_empty() {
            return;
        }
        if let Some(p) = &mut self.para {
            p.items.push(Inline::Run(run));
        } else {
            self.para = Some(Paragraph { items: vec![Inline::Run(run)], ..Default::default() });
        }
    }

    fn flush_para(&mut self, d: &mut Docx) {
        let Some(p) = self.para.take() else { return };
        if let Some(t) = self.tables.last_mut() {
            // paragraph inside a table cell: keep its text
            if let Some(c) = &mut t.cell {
                let text: String = p.items.iter().filter_map(|i| if let Inline::Run(r) = i { Some(r.text.as_str()) } else { None }).collect();
                let runs: Vec<&Run> = p.items.iter().filter_map(|i| if let Inline::Run(r) = i { Some(r) } else { None }).filter(|r| !r.text.trim().is_empty()).collect();
                if !runs.is_empty() {
                    c.1 &= runs.iter().all(|r| r.bold);
                    c.2 &= runs.iter().all(|r| r.italic);
                }
                if c.3.is_none() {
                    c.3 = p.halign.clone();
                }
                c.0.push(text);
            }
            return;
        }
        d.blocks.push(Block::Para(p));
    }

    fn end(&mut self, name: &str, d: &mut Docx) {
        match name {
            "pPr" => self.in_ppr = false,
            "rPr" => self.in_rpr = false,
            "t" => self.in_t = false,
            "r" => {
                self.flush_run();
                self.in_run = false;
            }
            "p" => {
                self.flush_run();
                self.flush_para(d);
            }
            "tc" => {
                if let Some(t) = self.tables.last_mut() {
                    if let Some((texts, bold, italic, halign, span)) = t.cell.take() {
                        let text = texts.join("\n").trim_end_matches('\n').to_string();
                        let any = !text.trim().is_empty();
                        t.row.push(Cell { text, bold: bold && any, italic: italic && any, halign });
                        for _ in 1..span {
                            t.row.push(Cell::default());
                        }
                    }
                }
            }
            "tr" => {
                if let Some(t) = self.tables.last_mut() {
                    let row = std::mem::take(&mut t.row);
                    t.rows.push(row);
                }
            }
            "tbl" => {
                let Some(t) = self.tables.pop() else { return };
                if let Some(outer) = self.tables.last_mut() {
                    // nested table: its text goes into the enclosing cell
                    if let Some(c) = &mut outer.cell {
                        for r in &t.rows {
                            c.0.push(r.iter().map(|x| x.text.as_str()).collect::<Vec<_>>().join(" "));
                        }
                    }
                    return;
                }
                let k = d.blocks.iter().filter(|b| matches!(b, Block::Table(_))).count() + 1;
                d.blocks.push(Block::Table(Table { name: format!("table{k}"), cells: t.rows, border: true, title: None, notes: Vec::new() }));
            }
            "sectPr" => self.in_sect = false,
            _ => {}
        }
    }
}

/// Plain text of a paragraph.
pub fn para_text(p: &Paragraph) -> String {
    let mut s = String::new();
    for it in &p.items {
        match it {
            Inline::Run(r) => {
                s.push_str(&r.text);
                if r.linebreak {
                    s.push('\n');
                }
            }
            Inline::Break => s.push('\n'),
            Inline::Image { .. } => {}
        }
    }
    s
}

/// Heading level of a paragraph style: Title 1, Subtitle/Heading1 2, ...
pub fn heading_level(style: Option<&str>) -> Option<usize> {
    let s = style?.to_lowercase().replace(' ', "");
    if s == "title" {
        return Some(1);
    }
    if s == "subtitle" {
        return Some(2);
    }
    let n = s.strip_prefix("heading")?.parse::<usize>().ok()?;
    Some((n + 1).min(6))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_model() {
        let mut d = Docx::new();
        d.blocks.push(Block::Para(Paragraph { style: Some("Heading1".into()), halign: None, items: vec![Inline::Run(Run { text: "Head & more".into(), bold: true, ..Default::default() })] }));
        d.blocks.push(Block::Table(Table { name: "t".into(), cells: vec![vec![Cell { text: "a".into(), ..Default::default() }, Cell { text: "1".into(), ..Default::default() }]], border: true, title: None, notes: vec![] }));
        let back = read_docx(&d.to_bytes()).unwrap();
        match &back.blocks[0] {
            Block::Para(p) => {
                assert_eq!(p.style.as_deref(), Some("Heading1"));
                assert_eq!(para_text(p), "Head & more");
            }
            b => panic!("{b:?}"),
        }
        assert!(back.blocks.iter().any(|b| matches!(b, Block::Table(t) if t.cells[0][1].text == "1")));
    }
}
