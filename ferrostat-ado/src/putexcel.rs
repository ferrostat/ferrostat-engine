//! `putexcel`: writing cells, expressions and matrices to .xlsx workbooks.
//! Workbooks are read and written by `ferrostat_cmd::xlsx`.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matcmd::eval_matrix;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

pub use ferrostat_cmd::xlsx::{col_letters, parse_ref, Cell, Workbook};
use ferrostat_cmd::xlsx_style::CellStyle;

#[derive(Debug, Clone, Default)]
pub struct PutExcel {
    pub path: Option<String>,
    pub sheet: String,
    pub books: BTreeMap<String, Workbook>,
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
        } else if let Ok(bytes) = ferrostat_core::vfs::read(&path) {
            Workbook::read(&bytes).map_err(|_| rc_err(603, format!("{path} could not be read; use replace")))?
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
                    book.styles.remove(&sheet);
                    book.merges.remove(&sheet);
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
    // items: `CELL = value` assignments and bare `CELL[:CELL]` format targets
    let words = split_top_level_words(&main);
    let mut i = 0;
    let mut writes: Vec<((u32, u32), Cell)> = Vec::new();
    // ranges (top-left, bottom-right) that receive the format options
    let mut targets: Vec<((u32, u32), (u32, u32))> = Vec::new();
    while i < words.len() {
        let target = words[i].clone();
        let is_assign = target.contains('=') || words.get(i + 1).map(String::as_str) == Some("=");
        if !is_assign {
            targets.push(parse_range(&target)?);
            i += 1;
            continue;
        }
        let (cellref, after_eq) = if let Some((a, b)) = target.split_once('=') {
            (a.to_string(), b.to_string())
        } else {
            i += 1;
            (target, String::new())
        };
        let ((row, col), range_end) = parse_range(&cellref)?;
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
        while i < words.len() && !(parse_ref(words[i].split('=').next().unwrap().split(':').next().unwrap()).is_some() && (words[i].contains('=') || words.get(i + 1).map(String::as_str) == Some("="))) {
            value.push(' ');
            value.push_str(&words[i]);
            i += 1;
        }
        let v = value.trim();
        let before = writes.len();
        let mat = v.strip_prefix("matrix(").or_else(|| v.strip_prefix("mat(")).and_then(|x| x.strip_suffix(')'));
        if let Some(inner) = mat {
            // options may also be given inside the parentheses: matrix(A, names)
            let (name, inner_opts) = match inner.split_once(',') {
                Some((a, b)) => (a.trim(), split_options(b)),
                None => (inner.trim(), Vec::new()),
            };
            let m = eval_matrix(&me.engine.dataset, &me.engine.store, name)?;
            let has = |n: &str| has_opt(&opts, n) || has_opt(&inner_opts, n);
            let names = has("names");
            let rown = names || has("rownames");
            let coln = names || has("colnames");
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
        } else if v == "etable" || v.starts_with("etable(") {
            writes.extend(etable(me, row, col)?);
        } else if v.starts_with('"') || v.starts_with("`\"") {
            writes.push(((row, col), Cell::Str(unquote(v))));
        } else if v.starts_with("image(") || v.starts_with("picture(") || v.starts_with("formula(") {
            let what = &v[..v.find('(').unwrap_or(v.len())];
            return Err(rc_err(198, format!("putexcel: {what}() is not supported")));
        } else {
            let e = parse_expression(v).map_err(|e| rc_err(198, e.message))?;
            match Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)? {
                Val::Num(x) if has_opt(&opts, "asdate") || has_opt(&opts, "asdatetime") => {
                    // daily dates count days, datetimes milliseconds, from 01jan1960
                    let days = if has_opt(&opts, "asdatetime") { x / 86_400_000.0 } else { x };
                    writes.push(((row, col), Cell::Date(days + 21916.0)));
                }
                Val::Num(x) => writes.push(((row, col), Cell::Num(x))),
                Val::Str(s) => writes.push(((row, col), Cell::Str(s))),
            }
        }
        // format the given range, or the block that was written
        if cellref.contains(':') {
            targets.push(((row, col), range_end));
        } else if writes.len() > before {
            let block = &writes[before..];
            let r1 = block.iter().map(|((r, _), _)| *r).max().unwrap();
            let c1 = block.iter().map(|((_, c), _)| *c).max().unwrap();
            targets.push(((row, col), (r1, c1)));
        }
    }
    let fmt = FormatSpec::parse(&opts)?;
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
    if !fmt.is_empty() {
        let styles = book.styles.entry(sheet.clone()).or_default();
        for &((r0, c0), (r1, c1)) in &targets {
            for r in r0.min(r1)..=r0.max(r1) {
                for c in c0.min(c1)..=c0.max(c1) {
                    let st = styles.entry((r, c)).or_default();
                    if fmt.overwrite {
                        *st = CellStyle::default();
                    }
                    fmt.apply(st, (r, c), ((r0.min(r1), c0.min(c1)), (r0.max(r1), c0.max(c1))));
                    if st.is_default() {
                        styles.remove(&(r, c));
                    }
                }
            }
        }
        let merges = book.merges.entry(sheet.clone()).or_default();
        for &((r0, c0), (r1, c1)) in &targets {
            let name = format!("{}{}:{}{}", col_letters(c0.min(c1)), r0.min(r1), col_letters(c0.max(c1)), r0.max(r1));
            if fmt.merge && !merges.contains(&name) && (r0, c0) != (r1, c1) {
                merges.push(name);
            } else if fmt.unmerge {
                merges.retain(|m| *m != name);
            }
        }
    }
    ferrostat_core::vfs::write(&path, &book.to_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("file {path} saved"))
}

/// "B2" or "B2:D5" -> ((row, col), (row, col)).
fn parse_range(s: &str) -> Result<((u32, u32), (u32, u32)), String> {
    let mut it = s.split(':');
    let a = it.next().unwrap_or("");
    let p = parse_ref(a).ok_or_else(|| rc_err(198, format!("{a} invalid cell")))?;
    let q = match it.next() {
        Some(b) => parse_ref(b).ok_or_else(|| rc_err(198, format!("{b} invalid cell")))?,
        None => p,
    };
    Ok((p, q))
}

/// Coefficient table of the last estimation: names, coefficients, standard
/// errors, test statistics, p-values and confidence intervals.
fn etable(me: &AdoInterpreter, row: u32, col: u32) -> Result<Vec<((u32, u32), Cell)>, String> {
    let e = &me.engine.store.e;
    let (Some(b), Some(v)) = (e.matrices.get("b"), e.matrices.get("V")) else { return Err(rc_err(301, "last estimates not found")) };
    let df = e.get("df_r");
    let level = 95.0;
    let q = match df {
        Some(d) => ferrostat_cmd::dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        None => ferrostat_cmd::dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    let stat = if df.is_some() { "t" } else { "z" };
    let mut out = Vec::new();
    let head = [e.macros.get("depvar").cloned().unwrap_or_default(), "Coefficient".into(), "Std. err.".into(), stat.into(), format!("P>|{stat}|"), "[95% conf.".into(), "interval]".into()];
    for (j, h) in head.iter().enumerate() {
        out.push(((row, col + j as u32), Cell::Str(h.clone())));
    }
    for (k, name) in b.colnames.iter().enumerate() {
        let r = row + 1 + k as u32;
        let (bk, se) = (b.get(0, k), v.get(k, k).max(0.0).sqrt());
        let t = bk / se;
        let p = match df {
            Some(d) => 2.0 * ferrostat_cmd::dist::ttail(d, t.abs()),
            None => 2.0 * (1.0 - ferrostat_cmd::dist::normal(t.abs())),
        };
        out.push(((r, col), Cell::Str(name.clone())));
        for (j, x) in [bk, se, t, p, bk - q * se, bk + q * se].into_iter().enumerate() {
            out.push(((r, col + 1 + j as u32), if x.is_finite() { Cell::Num(x) } else { Cell::Str(String::new()) }));
        }
    }
    Ok(out)
}

/// Cell format options of one `putexcel` call.
#[derive(Debug, Default)]
struct FormatSpec {
    nformat: Option<String>,
    bold: Option<bool>,
    italic: Option<bool>,
    underline: Option<bool>,
    strike: Option<bool>,
    script: Option<Option<String>>,
    font: Option<(String, Option<String>, Option<String>)>,
    halign: Option<String>,
    valign: Option<String>,
    wrap: Option<bool>,
    shrink: Option<bool>,
    indent: Option<u32>,
    rotate: Option<u32>,
    fpattern: Option<(String, Option<String>, Option<String>)>,
    /// (sides, line style, color); sides: 0 left, 1 right, 2 top, 3 bottom, 4 all
    borders: Vec<(usize, Option<String>, String)>,
    merge: bool,
    unmerge: bool,
    overwrite: bool,
}

/// Comma-separated arguments of an option, unquoted.
fn args(a: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    for c in a.chars() {
        match c {
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur).trim().to_string()),
            c => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn color_arg(s: &str) -> Result<String, String> {
    ferrostat_cmd::xlsx_style::color(s).ok_or_else(|| rc_err(198, format!("color {s} invalid")))
}

impl FormatSpec {
    fn parse(opts: &[(String, Option<String>)]) -> Result<FormatSpec, String> {
        let mut f = FormatSpec::default();
        for (name, arg) in opts {
            let n = name.to_lowercase();
            let a = arg.clone().unwrap_or_default();
            match n.as_str() {
                "nformat" => {
                    let code = unquote(a.trim());
                    let code = ferrostat_cmd::xlsx_style::named_num_format(&code).map(String::from).unwrap_or(code);
                    f.nformat = Some(code);
                }
                "bold" => f.bold = Some(arg.as_deref().map_or(true, |x| x.trim() != "off")),
                "nobold" => f.bold = Some(false),
                "italic" => f.italic = Some(arg.as_deref().map_or(true, |x| x.trim() != "off")),
                "noitalic" => f.italic = Some(false),
                "underline" => f.underline = Some(arg.as_deref().map_or(true, |x| x.trim() != "off")),
                "nounderline" => f.underline = Some(false),
                "strikeout" => f.strike = Some(arg.as_deref().map_or(true, |x| x.trim() != "off")),
                "nostrikeout" => f.strike = Some(false),
                "script" => {
                    f.script = Some(match a.trim() {
                        "sub" => Some("subscript".into()),
                        "super" => Some("superscript".into()),
                        _ => None,
                    })
                }
                "font" => {
                    let p = args(&a);
                    let size = p.get(1).filter(|x| !x.is_empty()).cloned();
                    let color = match p.get(2).filter(|x| !x.is_empty()) {
                        Some(c) => Some(color_arg(c)?),
                        None => None,
                    };
                    f.font = Some((p[0].clone(), size, color));
                }
                "hcenter" => f.halign = Some("center".into()),
                "vcenter" => f.valign = Some("center".into()),
                "halign" => {
                    f.halign = Some(match a.trim().to_lowercase().as_str() {
                        "left" => "left",
                        "right" => "right",
                        "center" => "center",
                        "fill" => "fill",
                        "justify" => "justify",
                        "merge" | "centercont" | "centercontinuous" => "centerContinuous",
                        "distributed" => "distributed",
                        x => return Err(rc_err(198, format!("halign({x}) invalid"))),
                    }
                    .into())
                }
                "valign" => {
                    f.valign = Some(match a.trim().to_lowercase().as_str() {
                        "top" => "top",
                        "bottom" => "bottom",
                        "center" => "center",
                        "justify" => "justify",
                        "distributed" => "distributed",
                        x => return Err(rc_err(198, format!("valign({x}) invalid"))),
                    }
                    .into())
                }
                "txtwrap" => f.wrap = Some(true),
                "notxtwrap" => f.wrap = Some(false),
                "shrinkfit" => f.shrink = Some(true),
                "noshrinkfit" => f.shrink = Some(false),
                "txtindent" => f.indent = Some(a.trim().parse().map_err(|_| rc_err(198, "txtindent() invalid"))?),
                "txtrotate" => f.rotate = Some(a.trim().parse().map_err(|_| rc_err(198, "txtrotate() invalid"))?),
                "fpattern" => {
                    let p = args(&a);
                    let fg = match p.get(1).filter(|x| !x.is_empty()) {
                        Some(c) => Some(color_arg(c)?),
                        None => None,
                    };
                    let bg = match p.get(2).filter(|x| !x.is_empty()) {
                        Some(c) => Some(color_arg(c)?),
                        None => None,
                    };
                    let pat = match p[0].to_lowercase().as_str() {
                        "solid" | "" => "solid".to_string(),
                        x => x.to_string(),
                    };
                    f.fpattern = Some((pat, fg, bg));
                }
                "border" | "dborder" => {
                    let p = args(&a);
                    let sides: &[usize] = match p[0].to_lowercase().as_str() {
                        "all" => &[0, 1, 2, 3],
                        "left" => &[0],
                        "right" => &[1],
                        "top" => &[2],
                        "bottom" => &[3],
                        x => return Err(rc_err(198, format!("border({x}) invalid"))),
                    };
                    let style = match p.get(1).map(|x| x.to_lowercase()).filter(|x| !x.is_empty()).as_deref() {
                        None => Some("thin".to_string()),
                        Some("none") => None,
                        Some(x) => Some(match x {
                            "mediumdashed" => "mediumDashed".to_string(),
                            "dashdot" => "dashDot".to_string(),
                            "mediumdashdot" => "mediumDashDot".to_string(),
                            "dashdotdot" => "dashDotDot".to_string(),
                            "mediumdashdotdot" => "mediumDashDotDot".to_string(),
                            "slantdashdot" => "slantDashDot".to_string(),
                            x => x.to_string(),
                        }),
                    };
                    let color = match p.get(2).filter(|x| !x.is_empty()) {
                        Some(c) => color_arg(c)?,
                        None => "FF000000".to_string(),
                    };
                    for &s in sides {
                        f.borders.push((s, style.clone(), color.clone()));
                    }
                }
                "merge" => f.merge = true,
                "unmerge" => f.unmerge = true,
                "overwritefmt" => f.overwrite = true,
                _ => {}
            }
        }
        Ok(f)
    }

    fn is_empty(&self) -> bool {
        self.nformat.is_none()
            && self.bold.is_none()
            && self.italic.is_none()
            && self.underline.is_none()
            && self.strike.is_none()
            && self.script.is_none()
            && self.font.is_none()
            && self.halign.is_none()
            && self.valign.is_none()
            && self.wrap.is_none()
            && self.shrink.is_none()
            && self.indent.is_none()
            && self.rotate.is_none()
            && self.fpattern.is_none()
            && self.borders.is_empty()
            && !self.merge
            && !self.unmerge
            && !self.overwrite
    }

    /// Apply to the cell at `pos` inside range `(top-left, bottom-right)`;
    /// borders go on the outer edges of the range.
    fn apply(&self, st: &mut CellStyle, pos: (u32, u32), range: ((u32, u32), (u32, u32))) {
        if let Some(n) = &self.nformat {
            st.num_fmt = Some(n.clone());
        }
        if let Some(b) = self.bold {
            st.bold = b;
        }
        if let Some(b) = self.italic {
            st.italic = b;
        }
        if let Some(b) = self.underline {
            st.underline = b;
        }
        if let Some(b) = self.strike {
            st.strike = b;
        }
        if let Some(s) = &self.script {
            st.script = s.clone();
        }
        if let Some((name, size, color)) = &self.font {
            st.font_name = Some(name.clone()).filter(|n| !n.is_empty());
            if size.is_some() {
                st.font_size = size.clone();
            }
            if color.is_some() {
                st.font_color = color.clone();
            }
        }
        if let Some(h) = &self.halign {
            st.halign = Some(h.clone());
        }
        if let Some(v) = &self.valign {
            st.valign = Some(v.clone());
        }
        if let Some(w) = self.wrap {
            st.wrap = w;
        }
        if let Some(w) = self.shrink {
            st.shrink = w;
        }
        if let Some(i) = self.indent {
            st.indent = Some(i).filter(|&i| i > 0);
        }
        if let Some(r) = self.rotate {
            st.rotate = Some(r).filter(|&r| r > 0);
        }
        if let Some((p, fg, bg)) = &self.fpattern {
            if p == "none" {
                (st.fill_pattern, st.fill_fg, st.fill_bg) = (None, None, None);
            } else {
                st.fill_pattern = Some(p.clone());
                st.fill_fg = fg.clone();
                st.fill_bg = bg.clone();
            }
        }
        let ((r0, c0), (r1, c1)) = range;
        for (side, style, color) in &self.borders {
            let edge = match side {
                0 => pos.1 == c0,
                1 => pos.1 == c1,
                2 => pos.0 == r0,
                _ => pos.0 == r1,
            };
            if edge {
                st.borders[*side] = style.clone().map(|s| (s, color.clone()));
            }
        }
    }
}
