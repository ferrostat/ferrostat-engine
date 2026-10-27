//! A small Office Open XML (.docx) writer: paragraphs with styled runs,
//! PNG/JPEG images, tables and page breaks. Used by `putdocx` and by
//! `dyndoc, docx`.

use ferrostat_cmd::xlsx::{escape, zip};

/// A run of text with character formatting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Run {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub font: Option<String>,
    /// Font size in points.
    pub size: Option<f64>,
    /// Colour as six hex digits.
    pub color: Option<String>,
    /// Monospaced code text.
    pub code: bool,
    /// Line break after the text.
    pub linebreak: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Inline {
    Run(Run),
    /// Index into `Docx::media`, size in EMU.
    Image { media: usize, cx: u64, cy: u64 },
    Break,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Paragraph {
    pub style: Option<String>,
    pub halign: Option<String>,
    pub items: Vec<Inline>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cell {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub halign: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    pub name: String,
    pub cells: Vec<Vec<Cell>>,
    pub border: bool,
    pub title: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Para(Paragraph),
    Table(Table),
    PageBreak,
    /// Body XML copied from an existing document (`save, append`).
    Raw(String),
}

#[derive(Debug, Clone, Default)]
pub struct Docx {
    pub blocks: Vec<Block>,
    /// (file name, bytes) of embedded images.
    pub media: Vec<(String, Vec<u8>)>,
    pub landscape: bool,
    /// Page size in twentieths of a point (width, height), portrait.
    pub page: (u32, u32),
    pub font: Option<String>,
    pub font_size: Option<f64>,
    /// Extra relationships for copied content: (id, target).
    pub extra_rels: Vec<(String, String)>,
}

pub const EMU_PER_INCH: f64 = 914_400.0;

/// Page size by name (letter, legal, A3, A4, A5, B4, B5) in twips.
pub fn page_size(name: &str) -> Option<(u32, u32)> {
    Some(match name.to_lowercase().as_str() {
        "letter" => (12240, 15840),
        "legal" => (12240, 20160),
        "a3" => (16838, 23811),
        "a4" => (11906, 16838),
        "a5" => (8391, 11906),
        "b4" => (14173, 20013),
        "b5" => (9978, 14173),
        _ => return None,
    })
}

/// Pixel size of a PNG, JPEG or GIF image.
pub fn image_size(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() > 24 && data.starts_with(b"\x89PNG") {
        let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        return Some((w, h));
    }
    if data.len() > 10 && data.starts_with(b"GIF") {
        return Some((u16::from_le_bytes([data[6], data[7]]) as u32, u16::from_le_bytes([data[8], data[9]]) as u32));
    }
    if data.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2;
        while i + 9 < data.len() {
            if data[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = data[i + 1];
            let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            if (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC {
                let h = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
                let w = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
                return Some((w, h));
            }
            i += 2 + len;
        }
    }
    None
}

/// Parse a length such as `10 cm`, `2in`, `72pt`, `300px` or `3` (inches) into EMU.
pub fn parse_length(s: &str) -> Option<u64> {
    let t = s.trim().to_lowercase();
    let end = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let v: f64 = t[..end].parse().ok()?;
    let inches = match t[end..].trim() {
        "" | "in" | "inch" | "inches" => v,
        "cm" => v / 2.54,
        "mm" => v / 25.4,
        "pt" => v / 72.0,
        "px" => v / 96.0,
        "twips" | "tw" => v / 1440.0,
        _ => return None,
    };
    Some((inches * EMU_PER_INCH).round() as u64)
}

/// Six-digit hex colour for a colour name or hex string.
pub fn color_hex(name: &str) -> Option<String> {
    let n = name.trim().trim_start_matches('#').to_lowercase();
    if n.len() == 6 && n.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(n.to_uppercase());
    }
    Some(
        match n.as_str() {
            "black" => "000000",
            "white" => "FFFFFF",
            "red" => "FF0000",
            "green" => "008000",
            "blue" => "0000FF",
            "yellow" => "FFFF00",
            "orange" => "FFA500",
            "purple" => "800080",
            "gray" | "grey" => "808080",
            "lightgray" | "lightgrey" => "D3D3D3",
            "darkgray" | "darkgrey" => "A9A9A9",
            "navy" => "000080",
            "maroon" => "800000",
            "brown" => "A52A2A",
            "pink" => "FFC0CB",
            "cyan" => "00FFFF",
            "magenta" => "FF00FF",
            "teal" => "008080",
            "olive" => "808000",
            "lime" => "00FF00",
            "darkblue" => "00008B",
            "darkred" => "8B0000",
            "darkgreen" => "006400",
            _ => return None,
        }
        .to_string(),
    )
}

impl Docx {
    pub fn new() -> Self {
        Docx { page: (12240, 15840), ..Default::default() }
    }

    /// Add an image file; returns the inline element sized to `width`/`height`
    /// (EMU) or, by default, the pixel size at 96 dpi limited to the text width.
    pub fn add_image(&mut self, name: &str, data: Vec<u8>, width: Option<u64>, height: Option<u64>) -> Inline {
        let (pw, ph) = image_size(&data).unwrap_or((640, 480));
        let (pw, ph) = (pw.max(1) as f64, ph.max(1) as f64);
        let ratio = ph / pw;
        let (cx, cy) = match (width, height) {
            (Some(w), Some(h)) => (w, h),
            (Some(w), None) => (w, (w as f64 * ratio).round() as u64),
            (None, Some(h)) => ((h as f64 / ratio).round() as u64, h),
            (None, None) => {
                let max = self.text_width();
                let w = (pw / 96.0 * EMU_PER_INCH).min(max as f64);
                (w.round() as u64, (w * ratio).round() as u64)
            }
        };
        let ext = name.rsplit('.').next().unwrap_or("png").to_lowercase();
        let ext = if ext == "jpg" { "jpeg".to_string() } else { ext };
        let idx = self.media.len();
        self.media.push((format!("image{}.{}", idx + 1, ext), data));
        Inline::Image { media: idx, cx, cy }
    }

    /// Text width in EMU (page width minus 1-inch margins).
    pub fn text_width(&self) -> u64 {
        let w = if self.landscape { self.page.1 } else { self.page.0 };
        ((w.saturating_sub(2880)) as u64) * 635
    }

    /// The paragraph currently being written, if any.
    pub fn current_para(&mut self) -> Option<&mut Paragraph> {
        match self.blocks.last_mut() {
            Some(Block::Para(p)) => Some(p),
            _ => None,
        }
    }

    pub fn table_mut(&mut self, name: &str) -> Option<&mut Table> {
        self.blocks.iter_mut().find_map(|b| match b {
            Block::Table(t) if t.name == name => Some(t),
            _ => None,
        })
    }

    fn run_xml(r: &Run) -> String {
        let mut props = String::new();
        if r.code {
            props.push_str(r#"<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>"#);
        } else if let Some(f) = &r.font {
            let f = escape(f);
            props.push_str(&format!(r#"<w:rFonts w:ascii="{f}" w:hAnsi="{f}" w:cs="{f}"/>"#));
        }
        if r.bold {
            props.push_str("<w:b/>");
        }
        if r.italic {
            props.push_str("<w:i/>");
        }
        if let Some(c) = &r.color {
            props.push_str(&format!(r#"<w:color w:val="{c}"/>"#));
        }
        if let Some(s) = r.size {
            props.push_str(&format!(r#"<w:sz w:val="{}"/>"#, (s * 2.0).round() as i64));
        }
        if r.underline {
            props.push_str(r#"<w:u w:val="single"/>"#);
        }
        let mut s = String::from("<w:r>");
        if !props.is_empty() {
            s.push_str(&format!("<w:rPr>{props}</w:rPr>"));
        }
        for (k, line) in r.text.split('\n').enumerate() {
            if k > 0 {
                s.push_str("<w:br/>");
            }
            if !line.is_empty() {
                s.push_str(&format!(r#"<w:t xml:space="preserve">{}</w:t>"#, escape(line)));
            }
        }
        if r.linebreak {
            s.push_str("<w:br/>");
        }
        s.push_str("</w:r>");
        s
    }

    fn image_xml(&self, media: usize, cx: u64, cy: u64, id: usize) -> String {
        let name = escape(&self.media[media].0);
        format!(
            r#"<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{id}" name="Picture {id}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdImg{}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"#,
            media + 1
        )
    }

    fn para_xml(&self, p: &Paragraph, next_id: &mut usize) -> String {
        let mut s = String::from("<w:p>");
        let mut props = String::new();
        if let Some(st) = &p.style {
            props.push_str(&format!(r#"<w:pStyle w:val="{}"/>"#, escape(st)));
        }
        if let Some(h) = &p.halign {
            let v = match h.as_str() {
                "center" => "center",
                "right" => "right",
                "both" | "distribute" => "both",
                _ => "left",
            };
            props.push_str(&format!(r#"<w:jc w:val="{v}"/>"#));
        }
        if !props.is_empty() {
            s.push_str(&format!("<w:pPr>{props}</w:pPr>"));
        }
        for it in &p.items {
            match it {
                Inline::Run(r) => s.push_str(&Self::run_xml(r)),
                Inline::Image { media, cx, cy } => {
                    *next_id += 1;
                    s.push_str(&self.image_xml(*media, *cx, *cy, *next_id));
                }
                Inline::Break => s.push_str("<w:r><w:br/></w:r>"),
            }
        }
        s.push_str("</w:p>");
        s
    }

    fn table_xml(t: &Table) -> String {
        let mut s = String::new();
        if let Some(title) = &t.title {
            s.push_str(&format!(r#"<w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr>{}</w:p>"#, Self::run_xml(&Run { text: title.clone(), ..Default::default() })));
        }
        let ncols = t.cells.iter().map(Vec::len).max().unwrap_or(0).max(1);
        s.push_str(r#"<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>"#);
        if t.border {
            s.push_str("<w:tblBorders>");
            for side in ["top", "left", "bottom", "right", "insideH", "insideV"] {
                s.push_str(&format!(r#"<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>"#));
            }
            s.push_str("</w:tblBorders>");
        } else {
            s.push_str("<w:tblBorders>");
            for side in ["top", "left", "bottom", "right", "insideH", "insideV"] {
                s.push_str(&format!(r#"<w:{side} w:val="nil"/>"#));
            }
            s.push_str("</w:tblBorders>");
        }
        s.push_str("</w:tblPr><w:tblGrid>");
        for _ in 0..ncols {
            s.push_str(r#"<w:gridCol/>"#);
        }
        s.push_str("</w:tblGrid>");
        for row in &t.cells {
            s.push_str("<w:tr>");
            for c in 0..ncols {
                let cell = row.get(c).cloned().unwrap_or_default();
                s.push_str("<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr><w:p>");
                if let Some(h) = &cell.halign {
                    s.push_str(&format!(r#"<w:pPr><w:jc w:val="{}"/></w:pPr>"#, escape(h)));
                }
                if !cell.text.is_empty() {
                    s.push_str(&Self::run_xml(&Run { text: cell.text.clone(), bold: cell.bold, italic: cell.italic, ..Default::default() }));
                }
                s.push_str("</w:p></w:tc>");
            }
            s.push_str("</w:tr>");
        }
        s.push_str("</w:tbl>");
        for n in &t.notes {
            s.push_str(&format!("<w:p>{}</w:p>", Self::run_xml(&Run { text: n.clone(), size: Some(9.0), ..Default::default() })));
        }
        // an empty paragraph keeps consecutive tables apart
        s.push_str("<w:p/>");
        s
    }

    /// The `word/document.xml` part.
    pub fn document_xml(&self) -> String {
        let mut body = String::new();
        let mut next_id = 0usize;
        for b in &self.blocks {
            match b {
                Block::Para(p) => body.push_str(&self.para_xml(p, &mut next_id)),
                Block::Table(t) => body.push_str(&Self::table_xml(t)),
                Block::PageBreak => body.push_str(r#"<w:p><w:r><w:br w:type="page"/></w:r></w:p>"#),
                Block::Raw(x) => body.push_str(x),
            }
        }
        let (w, h) = if self.landscape { (self.page.1, self.page.0) } else { self.page };
        let orient = if self.landscape { r#" w:orient="landscape""# } else { "" };
        format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{body}<w:sectPr><w:pgSz w:w="{w}" w:h="{h}"{orient}/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>"#
        )
    }

    fn styles_xml(&self) -> String {
        let font = escape(self.font.as_deref().unwrap_or("Calibri"));
        let size = (self.font_size.unwrap_or(11.0) * 2.0).round() as i64;
        let heading = |id: &str, name: &str, sz: i64, color: &str, level: Option<u8>| {
            let outline = level.map(|l| format!(r#"<w:outlineLvl w:val="{l}"/>"#)).unwrap_or_default();
            format!(
                r#"<w:style w:type="paragraph" w:styleId="{id}"><w:name w:val="{name}"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/>{outline}</w:pPr><w:rPr><w:sz w:val="{sz}"/><w:color w:val="{color}"/></w:rPr></w:style>"#
            )
        };
        let mut s = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" w:eastAsia="{font}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="259" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>"#
        );
        s.push_str(&heading("Title", "Title", 56, "000000", None));
        s.push_str(&heading("Subtitle", "Subtitle", 30, "5A5A5A", None));
        let sizes = [32, 26, 24, 22, 22, 22, 22, 22, 22];
        for (k, sz) in sizes.iter().enumerate() {
            s.push_str(&heading(&format!("Heading{}", k + 1), &format!("heading {}", k + 1), *sz, "2F5496", Some(k as u8)));
        }
        s.push_str(r#"<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="80"/></w:pPr><w:rPr><w:b/></w:rPr></w:style>"#);
        s.push_str(r#"<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>"#);
        s.push_str(r#"<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:style>"#);
        s.push_str(r#"<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>"#);
        s.push_str(r#"<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:style>"#);
        s.push_str("</w:styles>");
        s
    }

    /// The complete .docx file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut exts: Vec<String> = self.media.iter().map(|(n, _)| n.rsplit('.').next().unwrap_or("png").to_string()).collect();
        for (_, t) in &self.extra_rels {
            if let Some(e) = t.rsplit('.').next() {
                exts.push(e.to_lowercase());
            }
        }
        exts.sort();
        exts.dedup();
        let mut ct = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>"#,
        );
        for e in &exts {
            if e == "rels" || e == "xml" {
                continue;
            }
            let mime = match e.as_str() {
                "jpeg" | "jpg" => "image/jpeg",
                "gif" => "image/gif",
                "svg" => "image/svg+xml",
                _ => "image/png",
            };
            ct.push_str(&format!(r#"<Default Extension="{e}" ContentType="{mime}"/>"#));
        }
        ct.push_str(r#"<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>"#);
        let rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>"#;
        let mut doc_rels = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>"#,
        );
        let img_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image";
        for (k, (name, _)) in self.media.iter().enumerate() {
            doc_rels.push_str(&format!(r#"<Relationship Id="rIdImg{}" Type="{img_type}" Target="media/{}"/>"#, k + 1, escape(name)));
        }
        for (id, target) in &self.extra_rels {
            doc_rels.push_str(&format!(r#"<Relationship Id="{}" Type="{img_type}" Target="{}"/>"#, escape(id), escape(target)));
        }
        doc_rels.push_str("</Relationships>");
        let core = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>Ferrostat</dc:creator></cp:coreProperties>"#;
        let app = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Ferrostat</Application></Properties>"#;
        let mut files: Vec<(String, Vec<u8>)> = vec![
            ("[Content_Types].xml".into(), ct.into_bytes()),
            ("_rels/.rels".into(), rels.as_bytes().to_vec()),
            ("docProps/core.xml".into(), core.as_bytes().to_vec()),
            ("docProps/app.xml".into(), app.as_bytes().to_vec()),
            ("word/document.xml".into(), self.document_xml().into_bytes()),
            ("word/styles.xml".into(), self.styles_xml().into_bytes()),
            ("word/_rels/document.xml.rels".into(), doc_rels.into_bytes()),
        ];
        for (name, data) in &self.media {
            files.push((format!("word/media/{name}"), data.clone()));
        }
        zip(&files)
    }

    /// Load the body of an existing .docx so new content can be appended to it.
    /// Images of the old document are kept under new relationship ids.
    pub fn prepend_existing(&mut self, data: &[u8]) -> Result<(Vec<(String, Vec<u8>)>, Vec<String>), String> {
        let parts = ferrostat_cmd::xlsx::unzip(data)?;
        let doc = parts.get("word/document.xml").ok_or("not a .docx file")?;
        let doc = String::from_utf8_lossy(doc).to_string();
        let start = doc.find("<w:body>").ok_or("not a .docx file")? + "<w:body>".len();
        let end = doc.rfind("<w:sectPr").or_else(|| doc.rfind("</w:body>")).ok_or("not a .docx file")?;
        let mut body = doc[start..end.max(start)].to_string();
        let mut kept = Vec::new();
        let mut names = Vec::new();
        if let Some(rels) = parts.get("word/_rels/document.xml.rels") {
            let rels = String::from_utf8_lossy(rels).to_string();
            for ev in ferrostat_cmd::xlsx::xml_events(&rels) {
                if let ferrostat_cmd::xlsx::XmlEvent::Start(name, attrs, _) = ev {
                    if !name.ends_with("Relationship") {
                        continue;
                    }
                    let get = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone()).unwrap_or_default();
                    let (id, target, ty) = (get("Id"), get("Target"), get("Type"));
                    if !ty.ends_with("/image") {
                        continue;
                    }
                    let new_id = format!("rIdOld{}", kept.len() + 1);
                    let fname = format!("old{}_{}", kept.len() + 1, target.rsplit('/').next().unwrap_or("image.png"));
                    if let Some(bytes) = parts.get(&format!("word/{target}")) {
                        kept.push((fname.clone(), bytes.clone()));
                        body = body.replace(&format!("r:embed=\"{id}\""), &format!("r:embed=\"{new_id}\""));
                        self.extra_rels.push((new_id, format!("media/{fname}")));
                        names.push(fname);
                    }
                }
            }
        }
        self.blocks.insert(0, Block::Raw(body));
        Ok((kept, names))
    }

    /// Serialise, including media copied by `prepend_existing`.
    pub fn to_bytes_with(&self, extra_media: &[(String, Vec<u8>)]) -> Vec<u8> {
        if extra_media.is_empty() {
            return self.to_bytes();
        }
        let base = self.to_bytes();
        let mut parts: Vec<(String, Vec<u8>)> = ferrostat_cmd::xlsx::unzip(&base).map(|m| m.into_iter().collect()).unwrap_or_default();
        // keep [Content_Types].xml first
        parts.sort_by_key(|(n, _)| n != "[Content_Types].xml");
        for (n, d) in extra_media {
            parts.push((format!("word/media/{n}"), d.clone()));
        }
        zip(&parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        assert_eq!(parse_length("1in"), Some(914_400));
        assert_eq!(parse_length("2.54 cm"), Some(914_400));
        assert_eq!(parse_length("72pt"), Some(914_400));
        assert_eq!(color_hex("red").as_deref(), Some("FF0000"));
    }

    #[test]
    fn document_has_parts() {
        let mut d = Docx::new();
        d.blocks.push(Block::Para(Paragraph { style: Some("Title".into()), halign: None, items: vec![Inline::Run(Run { text: "A & B".into(), ..Default::default() })] }));
        let x = d.document_xml();
        assert!(x.contains("A &amp; B"));
        let parts = ferrostat_cmd::xlsx::unzip(&d.to_bytes()).unwrap();
        assert!(parts.contains_key("word/styles.xml"));
        assert!(parts.contains_key("[Content_Types].xml"));
    }
}
