//! `frmttable`-compatible formatted tables (Ferrostat's own implementation):
//! a matrix of statistics becomes a table of formatted cells with row and
//! column titles. Tables are kept in memory so later calls can `merge`
//! (add columns) or `append` (add rows), and are written to RTF (Word) or
//! LaTeX files with `using`.

use std::collections::HashMap;

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::matcmd::eval_matrix;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_parser::parser::{find_top_level, split_options};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

/// A formatted table: header rows and body rows of text cells. When
/// `has_rtitles` is set, column 0 holds the row titles.
#[derive(Debug, Clone, Default)]
pub struct FrmtTable {
    pub title: Vec<String>,
    pub notes: Vec<String>,
    pub has_rtitles: bool,
    pub head: Vec<Vec<String>>,
    pub body: Vec<Vec<String>>,
    /// body rows that hold sub-statistics (they follow their main row)
    pub sub: Vec<bool>,
}

impl FrmtTable {
    fn ncols(&self) -> usize {
        self.head.iter().chain(self.body.iter()).map(Vec::len).max().unwrap_or(0)
    }

    /// Pad all rows to the full width.
    fn square(&mut self) {
        let n = self.ncols();
        for r in self.head.iter_mut().chain(self.body.iter_mut()) {
            r.resize(n, String::new());
        }
    }

    /// Plain-text rendering for the Results window.
    pub fn render(&self) -> String {
        let n = self.ncols();
        let mut w = vec![1usize; n];
        for r in self.head.iter().chain(self.body.iter()) {
            for (j, c) in r.iter().enumerate() {
                w[j] = w[j].max(c.chars().count());
            }
        }
        let total: usize = w.iter().map(|x| x + 2).sum::<usize>();
        let cell = |j: usize, s: &str, left: bool| {
            let len = s.chars().count();
            if left {
                format!(" {}{} ", s, " ".repeat(w[j] - len))
            } else {
                let pad = w[j] - len;
                format!(" {}{}{} ", " ".repeat(pad / 2), s, " ".repeat(pad - pad / 2))
            }
        };
        let mut out = Vec::new();
        for t in &self.title {
            let pad = total.saturating_sub(t.chars().count()) / 2;
            out.push(format!("{}{}", " ".repeat(pad), t));
        }
        let rule = format!(" {}", "-".repeat(total));
        out.push(rule.clone());
        for r in &self.head {
            out.push(format!(" {}", r.iter().enumerate().map(|(j, c)| cell(j, c, false)).collect::<String>().trim_end()));
        }
        if !self.head.is_empty() {
            out.push(rule.clone());
        }
        for r in &self.body {
            out.push(format!(" {}", r.iter().enumerate().map(|(j, c)| cell(j, c, j == 0 && self.has_rtitles)).collect::<String>().trim_end()));
        }
        out.push(rule);
        for t in &self.notes {
            out.push(format!(" {t}"));
        }
        out.join("\n")
    }

    /// Rich Text Format document holding the table (opens in Word).
    pub fn to_rtf(&self) -> String {
        let n = self.ncols().max(1);
        let width = 9000 / n as u32;
        let mut s = String::from("{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\froman Times New Roman;}}\\fs24\n");
        for t in &self.title {
            s.push_str(&format!("\\pard\\qc {}\\par\n", rtf_escape(t)));
        }
        let rows: Vec<(&Vec<String>, bool, bool)> = self.head.iter().enumerate().map(|(i, r)| (r, i == 0, i + 1 == self.head.len())).chain(self.body.iter().enumerate().map(|(i, r)| (r, i == 0 && self.head.is_empty(), i + 1 == self.body.len()))).collect();
        for (r, top, bottom) in rows {
            s.push_str("\\trowd\\trgaph108");
            for j in 0..n {
                if top {
                    s.push_str("\\clbrdrt\\brdrs\\brdrw10");
                }
                if bottom {
                    s.push_str("\\clbrdrb\\brdrs\\brdrw10");
                }
                s.push_str(&format!("\\cellx{}", width * (j as u32 + 1)));
            }
            s.push('\n');
            for j in 0..n {
                let c = r.get(j).map(String::as_str).unwrap_or("");
                let align = if j == 0 && self.has_rtitles { "\\ql" } else { "\\qc" };
                s.push_str(&format!("\\pard\\intbl{align} {}\\cell\n", rtf_escape(c)));
            }
            s.push_str("\\row\n");
        }
        for t in &self.notes {
            s.push_str(&format!("\\pard\\ql {}\\par\n", rtf_escape(t)));
        }
        s.push('}');
        s
    }

    /// LaTeX `tabular` holding the table.
    pub fn to_tex(&self) -> String {
        let n = self.ncols().max(1);
        let spec: String = (0..n).map(|j| if j == 0 && self.has_rtitles { 'l' } else { 'c' }).collect();
        let mut s = String::from("\\begin{table}[htbp]\\centering\n");
        for t in &self.title {
            s.push_str(&format!("\\caption{{{}}}\n", tex_escape(t)));
        }
        s.push_str(&format!("\\begin{{tabular}}{{{spec}}}\n\\hline\n"));
        let line = |r: &Vec<String>| (0..n).map(|j| tex_escape(r.get(j).map(String::as_str).unwrap_or(""))).collect::<Vec<_>>().join(" & ") + " \\\\\n";
        for r in &self.head {
            s.push_str(&line(r));
        }
        if !self.head.is_empty() {
            s.push_str("\\hline\n");
        }
        for r in &self.body {
            s.push_str(&line(r));
        }
        s.push_str("\\hline\n\\end{tabular}\n");
        for t in &self.notes {
            s.push_str(&format!("\\\\ {}\n", tex_escape(t)));
        }
        s.push_str("\\end{table}\n");
        s
    }

    /// Add the columns of `new` to the right; rows are matched by row title
    /// when both tables have titles, otherwise by position.
    fn merge(&mut self, mut new: FrmtTable) {
        self.square();
        new.square();
        let old_w = self.ncols();
        let (new_cols, skip) = if new.has_rtitles { (new.ncols().saturating_sub(1), 1) } else { (new.ncols(), 0) };
        // header rows: align from the bottom
        let hh = self.head.len().max(new.head.len());
        while self.head.len() < hh {
            self.head.insert(0, vec![String::new(); old_w]);
        }
        let off = hh - new.head.len();
        for (i, r) in self.head.iter_mut().enumerate() {
            let src = if i >= off { new.head.get(i - off) } else { None };
            for j in 0..new_cols {
                r.push(src.and_then(|x| x.get(j + skip)).cloned().unwrap_or_default());
            }
        }
        let by_title = new.has_rtitles && self.has_rtitles;
        let mut target: Vec<usize> = Vec::new();
        let mut last = None::<usize>;
        for (k, r) in new.body.iter().enumerate() {
            let pos = if by_title {
                if new.sub[k] {
                    // a sub-statistic row goes below its main row
                    last.map(|p| p + 1).filter(|&p| p < self.body.len() && self.sub[p] && !target.contains(&p))
                } else {
                    self.body.iter().enumerate().position(|(i, b)| !self.sub[i] && b[0] == r[0] && !r[0].is_empty() && !target.contains(&i))
                }
            } else {
                (k < self.body.len()).then_some(k)
            };
            let pos = match pos {
                Some(p) => p,
                None => {
                    let mut row = vec![String::new(); old_w];
                    if self.has_rtitles && new.has_rtitles {
                        row[0] = r[0].clone();
                    }
                    let at = if new.sub[k] { last.map(|p| p + 1).unwrap_or(self.body.len()) } else { self.body.len() };
                    self.body.insert(at, row);
                    self.sub.insert(at, new.sub[k]);
                    for t in target.iter_mut() {
                        if *t >= at {
                            *t += 1;
                        }
                    }
                    at
                }
            };
            target.push(pos);
            last = Some(pos);
        }
        for (i, r) in self.body.iter_mut().enumerate() {
            let src = target.iter().position(|&t| t == i).map(|k| &new.body[k]);
            for j in 0..new_cols {
                r.push(src.and_then(|x| x.get(j + skip)).cloned().unwrap_or_default());
            }
        }
        if !self.has_rtitles && new.has_rtitles && self.body.is_empty() {
            self.has_rtitles = true;
        }
        self.notes.extend(new.notes);
        if self.title.is_empty() {
            self.title = new.title;
        }
    }

    /// Add the rows of `new` below.
    fn append(&mut self, new: FrmtTable) {
        if self.head.is_empty() {
            self.head = new.head;
        }
        self.body.extend(new.body);
        self.sub.extend(new.sub);
        self.notes.extend(new.notes);
        self.square();
    }
}

fn rtf_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '\\' | '{' | '}' => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) > 127 => o.push_str(&format!("\\u{}?", c as u32 as i16)),
            c => o.push(c),
        }
    }
    o
}

fn tex_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '&' | '%' | '$' | '#' | '_' | '{' | '}' => {
                o.push('\\');
                o.push(c);
            }
            c => o.push(c),
        }
    }
    o
}

/// Parse a text grid: `"a","b" \ "c","d"` (rows split by `\`, columns by `,`).
pub fn text_grid(s: &str) -> Vec<Vec<String>> {
    let mut rows = vec![Vec::new()];
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let mut quoted = false;
    let push = |rows: &mut Vec<Vec<String>>, cur: &mut String, quoted: &mut bool| {
        let t = if *quoted { cur.clone() } else { cur.trim().to_string() };
        rows.last_mut().unwrap().push(t);
        cur.clear();
        *quoted = false;
    };
    while let Some(c) = chars.next() {
        match c {
            '`' if chars.peek() == Some(&'"') => {
                chars.next();
                let mut depth = 1;
                let mut prev = ' ';
                while let Some(d) = chars.next() {
                    if prev == '`' && d == '"' {
                        depth += 1;
                    } else if prev == '"' && d == '\'' {
                        depth -= 1;
                        if depth == 0 {
                            cur.pop();
                            break;
                        }
                    }
                    cur.push(d);
                    prev = d;
                }
                quoted = true;
            }
            '"' => {
                for d in chars.by_ref() {
                    if d == '"' {
                        break;
                    }
                    cur.push(d);
                }
                quoted = true;
            }
            ',' => push(&mut rows, &mut cur, &mut quoted),
            '\\' => {
                push(&mut rows, &mut cur, &mut quoted);
                rows.push(Vec::new());
            }
            c => {
                if !quoted {
                    cur.push(c);
                }
            }
        }
    }
    push(&mut rows, &mut cur, &mut quoted);
    if rows.len() == 1 && rows[0].len() == 1 && rows[0][0].is_empty() {
        return Vec::new();
    }
    rows
}

/// Value of a grid option at (i, j); the last row/column repeats.
fn grid_at<T: Clone>(g: &[Vec<T>], i: usize, j: usize, default: T) -> T {
    let Some(row) = g.get(i.min(g.len().saturating_sub(1))) else { return default };
    row.get(j.min(row.len().saturating_sub(1))).cloned().unwrap_or(default)
}

fn fmt_num(v: f64, dec: usize, f: &str) -> String {
    if ferrostat_core::is_missing(v) {
        return String::new();
    }
    match f {
        "fc" => ferrostat_cmd::format::fmt_trim(v, &format!("%30.{dec}fc")),
        "g" => ferrostat_cmd::format::fmt_g(v, dec.max(1) + 6).trim().to_string(),
        "gc" => ferrostat_cmd::format::fmt_trim(v, &format!("%30.{}gc", dec.max(1))),
        "e" => ferrostat_cmd::format::fmt_e(v, dec),
        _ => format!("{:.*}", dec, v),
    }
}

fn is_default_name(n: &str, prefix: char) -> bool {
    n.is_empty() || n.strip_prefix(prefix).map_or(false, |d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

/// Build a table from a statistics matrix and the formatting options.
fn build(m: &Matrix, opt: &dyn Fn(&str) -> Option<String>, has: &dyn Fn(&str) -> bool, annot: Option<&Matrix>) -> Result<FrmtTable, String> {
    let nsub: usize = opt("substat").map(|s| s.trim().parse().map_err(|_| rc_err(198, "substat() invalid"))).transpose()?.unwrap_or(0);
    if m.cols % (nsub + 1) != 0 {
        return Err(rc_err(198, "number of statmat columns must be a multiple of substat()+1"));
    }
    let ncol = m.cols / (nsub + 1);
    let sdec: Vec<Vec<usize>> = opt("sdec").map(|s| text_grid(&s).into_iter().map(|r| r.iter().map(|x| x.parse().unwrap_or(3)).collect()).collect()).unwrap_or_default();
    let sfmt: Vec<Vec<String>> = opt("sfmt").map(|s| text_grid(&s).into_iter().map(|r| r.into_iter().map(|x| x.to_lowercase()).collect()).collect()).unwrap_or_default();
    let asym: Vec<String> = opt("asymbol").map(|s| text_grid(&s).into_iter().flatten().collect()).unwrap_or_default();
    let brackets: Vec<(String, String)> = match opt("brackets") {
        Some(b) => text_grid(&b).into_iter().map(|r| (r.first().cloned().unwrap_or_default(), r.get(1).cloned().unwrap_or_default())).collect(),
        None => vec![("(".into(), ")".into()), ("[".into(), "]".into())],
    };
    let nobrk = has("nobrket") || has("nobrackets");
    let mut t = FrmtTable::default();
    // row titles: rtitles() or non-default matrix row names
    let rt: Vec<String> = match opt("rtitles") {
        Some(r) => text_grid(&r).into_iter().map(|x| x.join(" ")).collect(),
        None if m.rownames.iter().any(|n| !is_default_name(n, 'r')) => m.rownames.iter().map(|n| n.rsplit(':').next().unwrap_or(n).to_string()).collect(),
        None => Vec::new(),
    };
    t.has_rtitles = !rt.is_empty();
    for i in 0..m.rows {
        for s in 0..=nsub {
            let mut row = Vec::new();
            if t.has_rtitles {
                row.push(if s == 0 { rt.get(i).cloned().unwrap_or_default() } else { String::new() });
            }
            for j in 0..ncol {
                let c = j * (nsub + 1) + s;
                let v = m.get(i, c);
                let mut txt = fmt_num(v, grid_at(&sdec, i, c, 3), &grid_at(&sfmt, i, c, "f".to_string()));
                if s > 0 && !txt.is_empty() && !nobrk {
                    if let Some((a, b)) = brackets.get(s - 1) {
                        txt = format!("{a}{txt}{b}");
                    }
                }
                if let Some(a) = annot {
                    let k = if i < a.rows && c < a.cols { a.get(i, c) } else { 0.0 };
                    if !ferrostat_core::is_missing(k) && k >= 1.0 {
                        if let Some(sym) = asym.get(k as usize - 1) {
                            txt.push_str(sym);
                        }
                    }
                }
                row.push(txt);
            }
            t.body.push(row);
            t.sub.push(s > 0);
        }
    }
    // column titles: ctitles() or non-default matrix column names
    match opt("ctitles") {
        Some(c) => t.head = text_grid(&c),
        None if m.colnames.iter().any(|n| !is_default_name(n, 'c')) => {
            let mut h: Vec<String> = if t.has_rtitles { vec![String::new()] } else { Vec::new() };
            h.extend((0..ncol).map(|j| m.colnames[j * (nsub + 1)].clone()));
            t.head = vec![h];
        }
        None => {}
    }
    if let Some(x) = opt("title") {
        t.title = text_grid(&x).into_iter().flatten().collect();
    }
    if let Some(x) = opt("note") {
        t.notes = text_grid(&x).into_iter().flatten().collect();
    }
    t.square();
    Ok(t)
}

pub fn frmttable(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    };
    let find = |n: &str, min: usize| opts.iter().find(|(o, _)| {
        let o = o.to_lowercase();
        o.len() >= min.min(n.len()) && n.starts_with(o.as_str())
    });
    let opt = |n: &str| find(n, n.len()).or_else(|| match n {
        "statmat" => find("statmat", 1),
        "ctitles" => find("ctitles", 2),
        "rtitles" => find("rtitles", 2),
        "substat" => find("substat", 4),
        "annotate" => find("annotate", 2),
        "asymbol" => find("asymbol", 3),
        "title" => find("title", 5),
        "note" => find("note", 4),
        _ => None,
    }).map(|(_, a)| a.clone().unwrap_or_default());
    let has = |n: &str| opts.iter().any(|(o, _)| o.to_lowercase() == n);
    let named = |n: &str| opts.iter().find(|(o, _)| o.to_lowercase() == n).map(|(_, a)| a.clone().map(|x| x.trim().to_string()).unwrap_or_default());
    let store = opt("store").map(|s| s.trim().to_string()).unwrap_or_default();
    if has("clear") || named("clear").is_some() {
        let name = named("clear").filter(|x| !x.is_empty()).unwrap_or(store.clone());
        me.frmt_tables.remove(&name);
        if opt("statmat").is_none() {
            return Ok(String::new());
        }
    }
    let mut table = match opt("statmat") {
        Some(name) => {
            let m = eval_matrix(&me.engine.dataset, &me.engine.store, name.trim())?;
            let annot = match opt("annotate") {
                Some(a) => Some(eval_matrix(&me.engine.dataset, &me.engine.store, a.trim())?),
                None => None,
            };
            let new = build(&m, &opt, &has, annot.as_ref())?;
            if let Some(target) = named("merge") {
                let key = if target.is_empty() { store.clone() } else { target };
                match me.frmt_tables.get(&key).cloned() {
                    Some(mut old) => {
                        old.merge(new);
                        old
                    }
                    None => new,
                }
            } else if let Some(target) = named("append") {
                let key = if target.is_empty() { store.clone() } else { target };
                match me.frmt_tables.get(&key).cloned() {
                    Some(mut old) => {
                        old.append(new);
                        old
                    }
                    None => new,
                }
            } else {
                new
            }
        }
        None => {
            // reuse the table in memory, possibly with new titles
            let mut t = me.frmt_tables.get(&store).cloned().ok_or_else(|| rc_err(198, "statmat() required: no table in memory"))?;
            if let Some(x) = opt("title") {
                t.title = text_grid(&x).into_iter().flatten().collect();
            }
            if let Some(x) = opt("note") {
                t.notes = text_grid(&x).into_iter().flatten().collect();
            }
            t
        }
    };
    table.square();
    me.frmt_tables.insert(store, table.clone());
    let mut out = String::new();
    if !has("nodisplay") {
        out = table.render();
    }
    if let Some(p) = main.trim().strip_prefix("using") {
        let tex = has("tex");
        let mut path = unquote(p.trim());
        if std::path::Path::new(&path).extension().is_none() {
            path.push_str(if tex { ".tex" } else { ".doc" });
        }
        let addtable = has("addtable");
        if ferrostat_core::vfs::exists(&path) && !has("replace") && !addtable {
            return Err(rc_err(602, format!("file {path} already exists")));
        }
        let text = if tex {
            table.to_tex()
        } else if addtable {
            // add the table to the end of an existing RTF document
            let new = table.to_rtf();
            match ferrostat_core::vfs::read_to_string(&path) {
                Ok(old) if old.trim_end().ends_with('}') => {
                    let body = new.find("\\fs24\n").map(|i| &new[i + 6..]).unwrap_or(&new);
                    format!("{}\\par\n{}", &old.trim_end()[..old.trim_end().len() - 1], body)
                }
                _ => new,
            }
        } else {
            table.to_rtf()
        };
        ferrostat_core::vfs::write(&path, text.as_bytes()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("(output written to {path})"));
    }
    Ok(out)
}

pub type Tables = HashMap<String, FrmtTable>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_parsing() {
        assert_eq!(text_grid(r#""Foreign","N""#), vec![vec!["Foreign".to_string(), "N".to_string()]]);
        assert_eq!(text_grid("0,2 \\ 1,1"), vec![vec!["0".to_string(), "2".to_string()], vec!["1".to_string(), "1".to_string()]]);
        assert_eq!(text_grid("*,**,***")[0].len(), 3);
    }
}
