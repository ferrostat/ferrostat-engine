//! Cell formats for .xlsx workbooks: number formats, fonts, fills, borders,
//! alignment and merged ranges. Builds `xl/styles.xml` from the formats used
//! by a workbook and reads them back from existing files.

use std::collections::{BTreeMap, HashMap};

use crate::xlsx::{escape, parse_ref, xml_events, XmlEvent};

/// Format of one cell. Colors are ARGB hex (`FF000000`); number formats are
/// format codes (`#,##0`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct CellStyle {
    pub num_fmt: Option<String>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    /// "superscript" or "subscript"
    pub script: Option<String>,
    pub font_name: Option<String>,
    pub font_size: Option<String>,
    pub font_color: Option<String>,
    pub halign: Option<String>,
    pub valign: Option<String>,
    pub wrap: bool,
    pub shrink: bool,
    pub indent: Option<u32>,
    pub rotate: Option<u32>,
    pub fill_pattern: Option<String>,
    pub fill_fg: Option<String>,
    pub fill_bg: Option<String>,
    /// left, right, top, bottom: (line style, color)
    pub borders: [Option<(String, String)>; 4],
}

impl CellStyle {
    pub fn is_default(&self) -> bool {
        *self == CellStyle::default()
    }

    fn has_font(&self) -> bool {
        self.bold || self.italic || self.underline || self.strike || self.script.is_some() || self.font_name.is_some() || self.font_size.is_some() || self.font_color.is_some()
    }

    fn has_align(&self) -> bool {
        self.halign.is_some() || self.valign.is_some() || self.wrap || self.shrink || self.indent.is_some() || self.rotate.is_some()
    }

    fn font_xml(&self) -> String {
        let mut s = String::from("<font>");
        if self.bold {
            s.push_str("<b/>");
        }
        if self.italic {
            s.push_str("<i/>");
        }
        if self.strike {
            s.push_str("<strike/>");
        }
        if self.underline {
            s.push_str("<u/>");
        }
        if let Some(v) = &self.script {
            s.push_str(&format!(r#"<vertAlign val="{v}"/>"#));
        }
        s.push_str(&format!(r#"<sz val="{}"/>"#, self.font_size.as_deref().unwrap_or("11")));
        if let Some(c) = &self.font_color {
            s.push_str(&format!(r#"<color rgb="{c}"/>"#));
        }
        s.push_str(&format!(r#"<name val="{}"/></font>"#, escape(self.font_name.as_deref().unwrap_or("Calibri"))));
        s
    }

    fn fill_xml(&self) -> String {
        let p = self.fill_pattern.as_deref().unwrap_or("none");
        let mut s = format!(r#"<fill><patternFill patternType="{p}">"#);
        if let Some(c) = &self.fill_fg {
            s.push_str(&format!(r#"<fgColor rgb="{c}"/>"#));
        }
        if let Some(c) = &self.fill_bg {
            s.push_str(&format!(r#"<bgColor rgb="{c}"/>"#));
        }
        s.push_str("</patternFill></fill>");
        s
    }

    fn border_xml(&self) -> String {
        let mut s = String::from("<border>");
        for (k, side) in ["left", "right", "top", "bottom"].iter().enumerate() {
            match &self.borders[k] {
                Some((st, c)) => s.push_str(&format!(r#"<{side} style="{st}"><color rgb="{c}"/></{side}>"#)),
                None => s.push_str(&format!("<{side}/>")),
            }
        }
        s.push_str("<diagonal/></border>");
        s
    }

    fn align_xml(&self) -> String {
        let mut s = String::from("<alignment");
        if let Some(h) = &self.halign {
            s.push_str(&format!(r#" horizontal="{h}""#));
        }
        if let Some(v) = &self.valign {
            s.push_str(&format!(r#" vertical="{v}""#));
        }
        if self.wrap {
            s.push_str(r#" wrapText="1""#);
        }
        if self.shrink {
            s.push_str(r#" shrinkToFit="1""#);
        }
        if let Some(i) = self.indent {
            s.push_str(&format!(r#" indent="{i}""#));
        }
        if let Some(r) = self.rotate {
            s.push_str(&format!(r#" textRotation="{r}""#));
        }
        s.push_str("/>");
        s
    }
}

/// Built-in number formats (id, code).
const BUILTIN: &[(u32, &str)] = &[
    (0, "General"),
    (1, "0"),
    (2, "0.00"),
    (3, "#,##0"),
    (4, "#,##0.00"),
    (9, "0%"),
    (10, "0.00%"),
    (11, "0.00E+00"),
    (12, "# ?/?"),
    (13, "# ??/??"),
    (14, "m/d/yyyy"),
    (15, "d-mmm-yy"),
    (16, "d-mmm"),
    (17, "mmm-yy"),
    (18, "h:mm AM/PM"),
    (19, "h:mm:ss AM/PM"),
    (20, "h:mm"),
    (21, "h:mm:ss"),
    (22, "m/d/yyyy h:mm"),
    (37, "#,##0 ;(#,##0)"),
    (38, "#,##0 ;[Red](#,##0)"),
    (39, "#,##0.00;(#,##0.00)"),
    (40, "#,##0.00;[Red](#,##0.00)"),
    (45, "mm:ss"),
    (46, "[h]:mm:ss"),
    (47, "mmss.0"),
    (48, "##0.0E+0"),
    (49, "@"),
];

/// Format code of a named number format (`number_sep`, `percent_d2`, ...).
pub fn named_num_format(name: &str) -> Option<&'static str> {
    let id = match name.to_lowercase().as_str() {
        "number" => 1,
        "number_d2" => 2,
        "number_sep" => 3,
        "number_sep_d2" => 4,
        "number_sep_negbra" => 37,
        "number_sep_negbrared" => 38,
        "number_d2_sep_negbra" => 39,
        "number_d2_sep_negbrared" => 40,
        "percent" => 9,
        "percent_d2" => 10,
        "scientific_d2" => 11,
        "fraction_onedig" => 12,
        "fraction_twodig" => 13,
        "date" => 14,
        "date_d_mon_yy" => 15,
        "date_d_mon" => 16,
        "date_mon_yy" => 17,
        "time_hmm_am" => 18,
        "time_hmmss_am" => 19,
        "time_hmm" => 20,
        "time_hmmss" => 21,
        "datetime" | "date_time" => 22,
        "time_mmss" => 45,
        "time_h_mmss" => 46,
        "time_mmss0" => 47,
        "scientific_d1" => 48,
        "text" => 49,
        _ => return None,
    };
    BUILTIN.iter().find(|(i, _)| *i == id).map(|(_, c)| *c)
}

/// Color as ARGB hex from a name, `RRGGBB`/`#RRGGBB` or `"r g b"`.
pub fn color(spec: &str) -> Option<String> {
    let s = spec.trim().trim_matches('"').trim();
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() == 3 {
        let v: Vec<u8> = parts.iter().filter_map(|p| p.parse().ok()).collect();
        if v.len() == 3 {
            return Some(format!("FF{:02X}{:02X}{:02X}", v[0], v[1], v[2]));
        }
    }
    let hex = s.trim_start_matches('#');
    if hex.len() == 6 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(format!("FF{}", hex.to_uppercase()));
    }
    let rgb = match s.to_lowercase().replace(' ', "").as_str() {
        "black" => "000000",
        "white" => "FFFFFF",
        "red" => "FF0000",
        "green" => "008000",
        "lime" => "00FF00",
        "blue" => "0000FF",
        "navy" => "000080",
        "yellow" => "FFFF00",
        "orange" => "FFA500",
        "purple" => "800080",
        "magenta" | "fuchsia" => "FF00FF",
        "cyan" | "aqua" => "00FFFF",
        "teal" => "008080",
        "maroon" => "800000",
        "olive" => "808000",
        "brown" => "A52A2A",
        "pink" => "FFC0CB",
        "gray" | "grey" => "808080",
        "silver" => "C0C0C0",
        "lightgray" | "lightgrey" => "D3D3D3",
        "darkgray" | "darkgrey" => "A9A9A9",
        "lightblue" => "ADD8E6",
        "darkblue" => "00008B",
        "lightgreen" => "90EE90",
        "darkgreen" => "006400",
        "darkred" => "8B0000",
        "lightyellow" => "FFFFE0",
        "gold" => "FFD700",
        "beige" => "F5F5DC",
        "bisque" => "FFE4C4",
        "khaki" => "F0E68C",
        "lavender" => "E6E6FA",
        "violet" => "EE82EE",
        "indigo" => "4B0082",
        "tan" => "D2B48C",
        "salmon" => "FA8072",
        "coral" => "FF7F50",
        "tomato" => "FF6347",
        "skyblue" => "87CEEB",
        "steelblue" => "4682B4",
        "royalblue" => "4169E1",
        "slategray" | "slategrey" => "708090",
        "whitesmoke" => "F5F5F5",
        "ivory" => "FFFFF0",
        "turquoise" => "40E0D0",
        "chocolate" => "D2691E",
        "crimson" => "DC143C",
        _ => return None,
    };
    Some(format!("FF{rgb}"))
}

/// `styles.xml` for a set of styles. Returns the XML and the `s` index of
/// each style; index 0 is the default style and index 1 the plain date style.
pub fn build_styles<'a>(styles: impl Iterator<Item = &'a CellStyle>) -> (String, HashMap<CellStyle, usize>) {
    let date = CellStyle { num_fmt: Some("m/d/yyyy".into()), ..Default::default() };
    let mut list: Vec<CellStyle> = vec![CellStyle::default(), date];
    let mut index: HashMap<CellStyle, usize> = HashMap::new();
    index.insert(list[0].clone(), 0);
    index.insert(list[1].clone(), 1);
    for s in styles {
        if !index.contains_key(s) {
            index.insert(s.clone(), list.len());
            list.push(s.clone());
        }
    }
    let mut fonts = vec![CellStyle::default().font_xml()];
    let mut fills = vec![r#"<fill><patternFill patternType="none"/></fill>"#.to_string(), r#"<fill><patternFill patternType="gray125"/></fill>"#.to_string()];
    let mut borders = vec![CellStyle::default().border_xml()];
    let mut numfmts: Vec<(u32, String)> = Vec::new();
    let mut xfs = Vec::new();
    let pos = |v: &mut Vec<String>, x: String| match v.iter().position(|y| *y == x) {
        Some(i) => i,
        None => {
            v.push(x);
            v.len() - 1
        }
    };
    for st in &list {
        let font = if st.has_font() { pos(&mut fonts, st.font_xml()) } else { 0 };
        let fill = if st.fill_pattern.is_some() { pos(&mut fills, st.fill_xml()) } else { 0 };
        let border = if st.borders.iter().any(Option::is_some) { pos(&mut borders, st.border_xml()) } else { 0 };
        let fmt = match &st.num_fmt {
            None => 0,
            Some(code) => match BUILTIN.iter().find(|(_, c)| c == code) {
                Some((id, _)) => *id,
                None => match numfmts.iter().find(|(_, c)| c == code) {
                    Some((id, _)) => *id,
                    None => {
                        let id = 164 + numfmts.len() as u32;
                        numfmts.push((id, code.clone()));
                        id
                    }
                },
            },
        };
        let mut x = format!(r#"<xf numFmtId="{fmt}" fontId="{font}" fillId="{fill}" borderId="{border}" xfId="0""#);
        if fmt != 0 {
            x.push_str(r#" applyNumberFormat="1""#);
        }
        if font != 0 {
            x.push_str(r#" applyFont="1""#);
        }
        if fill != 0 {
            x.push_str(r#" applyFill="1""#);
        }
        if border != 0 {
            x.push_str(r#" applyBorder="1""#);
        }
        if st.has_align() {
            x.push_str(&format!(r#" applyAlignment="1">{}</xf>"#, st.align_xml()));
        } else {
            x.push_str("/>");
        }
        xfs.push(x);
    }
    let mut s = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#);
    if !numfmts.is_empty() {
        s.push_str(&format!(r#"<numFmts count="{}">"#, numfmts.len()));
        for (id, c) in &numfmts {
            s.push_str(&format!(r#"<numFmt numFmtId="{id}" formatCode="{}"/>"#, escape(c)));
        }
        s.push_str("</numFmts>");
    }
    s.push_str(&format!(r#"<fonts count="{}">{}</fonts>"#, fonts.len(), fonts.concat()));
    s.push_str(&format!(r#"<fills count="{}">{}</fills>"#, fills.len(), fills.concat()));
    s.push_str(&format!(r#"<borders count="{}">{}</borders>"#, borders.len(), borders.concat()));
    s.push_str(r#"<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>"#);
    s.push_str(&format!(r#"<cellXfs count="{}">{}</cellXfs>"#, xfs.len(), xfs.concat()));
    s.push_str(r#"<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>"#);
    (s, index)
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The cell formats (`cellXfs`) defined in a `styles.xml`.
pub fn parse_styles(xml: &str) -> Vec<CellStyle> {
    let mut codes: BTreeMap<u32, String> = BUILTIN.iter().map(|(i, c)| (*i, c.to_string())).collect();
    let mut fonts: Vec<CellStyle> = Vec::new();
    let mut fills: Vec<CellStyle> = Vec::new();
    let mut borders: Vec<CellStyle> = Vec::new();
    let mut xfs = Vec::new();
    let mut section = String::new();
    let mut side: Option<usize> = None;
    let ev = xml_events(xml);
    let mut cur_xf: Option<CellStyle> = None;
    for e in ev {
        match e {
            XmlEvent::Start(n, a, sc) => match n.as_str() {
                "numFmt" => {
                    if let (Some(id), Some(c)) = (attr(&a, "numFmtId"), attr(&a, "formatCode")) {
                        codes.insert(id.parse().unwrap_or(0), c.to_string());
                    }
                }
                "fonts" | "fills" | "borders" | "cellXfs" | "cellStyleXfs" | "dxfs" if !sc => section = n.clone(),
                "font" if section == "fonts" => fonts.push(CellStyle::default()),
                "fill" if section == "fills" => fills.push(CellStyle::default()),
                "border" if section == "borders" => borders.push(CellStyle::default()),
                "b" | "i" | "u" | "strike" | "sz" | "name" | "vertAlign" | "color" if section == "fonts" => {
                    if let Some(f) = fonts.last_mut() {
                        let val = attr(&a, "val");
                        let on = val.map_or(true, |v| v != "0" && v != "false" && v != "none");
                        match n.as_str() {
                            "b" => f.bold = on,
                            "i" => f.italic = on,
                            "u" => f.underline = on,
                            "strike" => f.strike = on,
                            "sz" => f.font_size = val.map(String::from).filter(|v| v != "11"),
                            "name" => f.font_name = val.map(String::from).filter(|v| v != "Calibri"),
                            "vertAlign" => f.script = val.map(String::from).filter(|v| v != "baseline"),
                            _ => f.font_color = attr(&a, "rgb").map(String::from),
                        }
                    }
                }
                "patternFill" if section == "fills" => {
                    if let Some(f) = fills.last_mut() {
                        f.fill_pattern = attr(&a, "patternType").map(String::from).filter(|p| p != "none" && p != "gray125");
                    }
                }
                "fgColor" | "bgColor" if section == "fills" => {
                    if let Some(f) = fills.last_mut() {
                        let c = attr(&a, "rgb").map(String::from);
                        if n == "fgColor" {
                            f.fill_fg = c;
                        } else {
                            f.fill_bg = c;
                        }
                    }
                }
                "left" | "right" | "top" | "bottom" if section == "borders" => {
                    let k = ["left", "right", "top", "bottom"].iter().position(|s| *s == n).unwrap();
                    if let (Some(b), Some(st)) = (borders.last_mut(), attr(&a, "style")) {
                        b.borders[k] = Some((st.to_string(), "FF000000".into()));
                        side = (!sc).then_some(k);
                    }
                }
                "color" if section == "borders" => {
                    if let (Some(b), Some(k), Some(c)) = (borders.last_mut(), side, attr(&a, "rgb")) {
                        if let Some(x) = b.borders[k].as_mut() {
                            x.1 = c.to_string();
                        }
                    }
                }
                "xf" if section == "cellXfs" => {
                    let get = |k: &str| attr(&a, k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                    let mut st = fonts.get(get("fontId")).cloned().unwrap_or_default();
                    let fill = fills.get(get("fillId")).cloned().unwrap_or_default();
                    st.fill_pattern = fill.fill_pattern;
                    st.fill_fg = fill.fill_fg;
                    st.fill_bg = fill.fill_bg;
                    st.borders = borders.get(get("borderId")).cloned().unwrap_or_default().borders;
                    let id = get("numFmtId") as u32;
                    st.num_fmt = if id == 0 { None } else { codes.get(&id).cloned() };
                    if sc {
                        xfs.push(st);
                    } else {
                        cur_xf = Some(st);
                    }
                }
                "alignment" => {
                    if let Some(st) = cur_xf.as_mut() {
                        st.halign = attr(&a, "horizontal").map(String::from).filter(|h| h != "general");
                        st.valign = attr(&a, "vertical").map(String::from).filter(|v| v != "bottom");
                        st.wrap = attr(&a, "wrapText").map_or(false, |v| v == "1" || v == "true");
                        st.shrink = attr(&a, "shrinkToFit").map_or(false, |v| v == "1" || v == "true");
                        st.indent = attr(&a, "indent").and_then(|v| v.parse().ok()).filter(|&v| v > 0);
                        st.rotate = attr(&a, "textRotation").and_then(|v| v.parse().ok()).filter(|&v| v > 0);
                    }
                }
                _ => {}
            },
            XmlEvent::End(n) => match n.as_str() {
                "fonts" | "fills" | "borders" | "cellXfs" | "cellStyleXfs" | "dxfs" => section.clear(),
                "left" | "right" | "top" | "bottom" => side = None,
                "xf" => {
                    if let Some(st) = cur_xf.take() {
                        xfs.push(st);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    xfs
}

/// Formats of the cells of a worksheet and its merged ranges.
pub fn parse_sheet_styles(xml: &str, xfs: &[CellStyle]) -> (BTreeMap<(u32, u32), CellStyle>, Vec<String>) {
    let mut styles = BTreeMap::new();
    let mut merges = Vec::new();
    for e in xml_events(xml) {
        if let XmlEvent::Start(n, a, _) = e {
            if n == "c" {
                if let (Some(pos), Some(s)) = (attr(&a, "r").and_then(parse_ref), attr(&a, "s").and_then(|s| s.parse::<usize>().ok())) {
                    if let Some(st) = xfs.get(s).filter(|st| !st.is_default()) {
                        styles.insert(pos, st.clone());
                    }
                }
            } else if n == "mergeCell" {
                if let Some(r) = attr(&a, "ref") {
                    merges.push(r.to_string());
                }
            }
        }
    }
    (styles, merges)
}
