//! `putdocx` and `putpdf`: building Word and PDF documents from commands.
//! Both build the same document model (`crate::docx`); `putdocx save`
//! writes .docx, `putpdf save` renders it with `crate::pdf`.

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::format;
use ferrostat_cmd::matcmd::eval_matrix;
use ferrostat_cmd::util;
use ferrostat_core::missing::is_missing;
use ferrostat_core::vfs;
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options};

use crate::docx::{self, Block, Cell, Docx, Inline, Paragraph, Run, Table};
use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

/// Documents being built: one for `putdocx`, one for `putpdf`.
#[derive(Debug, Clone, Default)]
pub struct DocState {
    pub docx: Option<Docx>,
    pub pdf: Option<Docx>,
}

type Opts = Vec<(String, Option<String>)>;

fn opt<'a>(opts: &'a Opts, names: &[&str]) -> Option<&'a (String, Option<String>)> {
    opts.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        names.iter().any(|m| n == *m)
    })
}

fn opt_arg(opts: &Opts, names: &[&str]) -> Option<String> {
    opt(opts, names).and_then(|(_, a)| a.clone()).map(|a| a.trim().to_string())
}

fn has(opts: &Opts, names: &[&str]) -> bool {
    opt(opts, names).is_some()
}

fn split_main(r: &str) -> (String, Opts) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

/// Apply `font(name, size, color)` to a run.
fn apply_font(run: &mut Run, spec: &str) {
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    if let Some(f) = parts.first().filter(|f| !f.is_empty()) {
        run.font = Some(unquote(f));
    }
    if let Some(s) = parts.get(1).and_then(|s| s.parse::<f64>().ok()) {
        run.size = Some(s);
    }
    if let Some(c) = parts.get(2).filter(|c| !c.is_empty()) {
        run.color = docx::color_hex(&unquote(c));
    }
}

/// Format a number for a document: `nformat()` if given, else the default display format.
fn num_text(v: f64, nformat: Option<&str>) -> String {
    match nformat {
        Some(f) => format::fmt(v, f).trim().to_string(),
        None => format::display_number(v).trim().to_string(),
    }
}

/// Evaluate the `(exp)` of `text (exp)` or a table cell assignment.
fn eval_text(me: &AdoInterpreter, inner: &str, nformat: Option<&str>) -> Result<String, String> {
    let t = inner.trim();
    if t.is_empty() {
        return Ok(String::new());
    }
    let e = parse_expression(t).map_err(|e| rc_err(198, format!("{t}: {}", e.message)))?;
    Ok(match Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)? {
        Val::Num(x) => num_text(x, nformat),
        Val::Str(s) => s,
    })
}

/// Text inside the outer parentheses of `(…)`.
fn paren_inner(s: &str) -> Option<&str> {
    let s = s.trim();
    if s.starts_with('(') && s.ends_with(')') { Some(&s[1..s.len() - 1]) } else { None }
}

pub fn putdocx(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    putdoc(me, r, false)
}

pub fn putpdf(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    putdoc(me, r, true)
}

fn putdoc(me: &mut AdoInterpreter, r: &str, pdf: bool) -> Result<String, String> {
    let cmd = if pdf { "putpdf" } else { "putdocx" };
    let (main, opts) = split_main(r);
    let (sub, rest) = match main.find(|c: char| c.is_whitespace() || c == '(') {
        Some(i) => (main[..i].to_lowercase(), main[i..].trim().to_string()),
        None => (main.to_lowercase(), String::new()),
    };
    fn slot(me: &mut AdoInterpreter, pdf: bool) -> Option<&mut Docx> {
        if pdf { me.docs.pdf.as_mut() } else { me.docs.docx.as_mut() }
    }
    let no_doc = || rc_err(198, format!("document not created; use {cmd} begin"));
    match sub.as_str() {
        "begin" => {
            if slot(me, pdf).is_some() {
                return Err(rc_err(198, format!("{cmd} document already open in memory; use {cmd} clear or {cmd} save")));
            }
            let mut d = Docx::new();
            if let Some(p) = opt_arg(&opts, &["pagesize"]) {
                d.page = docx::page_size(&p).ok_or_else(|| rc_err(198, format!("pagesize({p}) invalid")))?;
            }
            d.landscape = has(&opts, &["landscape"]);
            if let Some(f) = opt_arg(&opts, &["font"]) {
                let mut run = Run::default();
                apply_font(&mut run, &f);
                d.font = run.font;
                d.font_size = run.size;
            }
            if pdf {
                me.docs.pdf = Some(d);
            } else {
                me.docs.docx = Some(d);
            }
            Ok(String::new())
        }
        "clear" => {
            if pdf {
                me.docs.pdf = None;
            } else {
                me.docs.docx = None;
            }
            Ok(String::new())
        }
        "describe" => {
            let d = slot(me, pdf).ok_or_else(no_doc)?;
            let paras = d.blocks.iter().filter(|b| matches!(b, Block::Para(_))).count();
            let tables: Vec<String> = d.blocks.iter().filter_map(|b| if let Block::Table(t) = b { Some(format!("{} ({} x {})", t.name, t.cells.len(), t.cells.first().map_or(0, Vec::len))) } else { None }).collect();
            let mut out = format!("Document in memory\n  paragraphs: {paras}\n  tables:     {}\n  images:     {}", tables.len(), d.media.len());
            for t in tables {
                out.push_str(&format!("\n    {t}"));
            }
            Ok(out)
        }
        "paragraph" => {
            let d = slot(me, pdf).ok_or_else(no_doc)?;
            let style = opt_arg(&opts, &["style"]).map(|s| unquote(&s));
            let halign = opt_arg(&opts, &["halign"]).map(|s| s.to_lowercase());
            let mut p = Paragraph { style, halign, items: Vec::new() };
            // a paragraph-wide font becomes the default for its runs
            if let Some(f) = opt_arg(&opts, &["font"]) {
                let mut run = Run::default();
                apply_font(&mut run, &f);
                p.items.push(Inline::Run(run));
            }
            d.blocks.push(Block::Para(p));
            Ok(String::new())
        }
        "text" => {
            let inner = paren_inner(&rest).ok_or_else(|| rc_err(198, format!("{cmd} text: (exp) required")))?.to_string();
            let nformat = opt_arg(&opts, &["nformat"]);
            let text = eval_text(me, &inner, nformat.as_deref())?;
            let d = slot(me, pdf).ok_or_else(no_doc)?;
            let p = d.current_para().ok_or_else(|| rc_err(198, "no active paragraph; use paragraph first"))?;
            // inherit the paragraph font set by `paragraph, font()`
            let mut run = match p.items.first() {
                Some(Inline::Run(r)) if r.text.is_empty() => Run { text: String::new(), linebreak: false, ..r.clone() },
                _ => Run::default(),
            };
            run.text = text;
            run.bold = has(&opts, &["bold"]);
            run.italic = has(&opts, &["italic"]);
            run.underline = has(&opts, &["underline"]) || opt_arg(&opts, &["underline"]).is_some();
            run.linebreak = has(&opts, &["linebreak"]);
            if let Some(f) = opt_arg(&opts, &["font"]) {
                apply_font(&mut run, &f);
            }
            p.items.push(Inline::Run(run));
            Ok(String::new())
        }
        "image" => {
            let path = unquote(&rest);
            let data = vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
            let width = opt_arg(&opts, &["width"]).map(|w| docx::parse_length(&w).ok_or_else(|| rc_err(198, format!("width({w}) invalid")))).transpose()?;
            let height = opt_arg(&opts, &["height"]).map(|w| docx::parse_length(&w).ok_or_else(|| rc_err(198, format!("height({w}) invalid")))).transpose()?;
            let linebreak = has(&opts, &["linebreak"]);
            let d = slot(me, pdf).ok_or_else(no_doc)?;
            let img = d.add_image(&path, data, width, height);
            if d.current_para().is_none() {
                d.blocks.push(Block::Para(Paragraph::default()));
            }
            let p = d.current_para().unwrap();
            p.items.push(img);
            if linebreak {
                p.items.push(Inline::Break);
            }
            Ok(String::new())
        }
        "pagebreak" | "sectionbreak" => {
            let d = slot(me, pdf).ok_or_else(no_doc)?;
            d.blocks.push(Block::PageBreak);
            Ok(String::new())
        }
        "table" => table(me, &rest, &opts, pdf),
        "save" => {
            let mut path = unquote(&rest);
            let ext = if pdf { ".pdf" } else { ".docx" };
            if !path.to_lowercase().ends_with(ext) {
                path.push_str(ext);
            }
            let replace = has(&opts, &["replace"]);
            let append = has(&opts, &["append"]);
            if vfs::exists(&path) && !replace && !append {
                return Err(rc_err(602, format!("file {path} already exists")));
            }
            let mut d = slot(me, pdf).ok_or_else(no_doc)?.clone();
            let bytes = if pdf {
                crate::pdf::render(&d)
            } else if append && vfs::exists(&path) {
                let old = vfs::read(&path).map_err(|e| rc_err(603, e))?;
                let (media, _) = d.prepend_existing(&old).map_err(|e| rc_err(610, format!("{path}: {e}")))?;
                d.to_bytes_with(&media)
            } else {
                d.to_bytes()
            };
            vfs::write(&path, &bytes).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
            if pdf {
                me.docs.pdf = None;
            } else {
                me.docs.docx = None;
            }
            Ok(format!("successfully created \"{}\"", full_path(&path)))
        }
        _ => Err(rc_err(198, format!("{cmd} {sub}: unknown subcommand"))),
    }
}

/// `table name = …`, `table name(i,j) = (…)` and `table name(i,j), options`.
fn table(me: &mut AdoInterpreter, rest: &str, opts: &Opts, pdf: bool) -> Result<String, String> {
    let (lhs, rhs) = match find_top_level(rest, '=') {
        Some(i) => (rest[..i].trim().to_string(), Some(rest[i + 1..].trim().to_string())),
        None => (rest.trim().to_string(), None),
    };
    let nformat = opt_arg(opts, &["nformat"]);
    let bold = has(opts, &["bold"]);
    let italic = has(opts, &["italic"]);
    let halign = opt_arg(opts, &["halign"]).map(|s| s.to_lowercase());
    // cell reference: name(i,j)
    if let Some(p) = lhs.find('(') {
        let name = lhs[..p].trim().to_string();
        let idx = lhs[p + 1..].trim_end_matches(')');
        let (i, j) = idx.split_once(',').ok_or_else(|| rc_err(198, "table: (i,j) expected"))?;
        let parse_idx = |s: &str| -> Result<Option<usize>, String> {
            let s = s.trim();
            if s == "." || s.is_empty() {
                return Ok(None);
            }
            s.parse::<usize>().map(Some).map_err(|_| rc_err(198, format!("table: invalid index {s}")))
        };
        let (i, j) = (parse_idx(i)?, parse_idx(j)?);
        let value = match &rhs {
            Some(v) => {
                let inner = paren_inner(v).unwrap_or(v);
                Some(eval_text(me, inner, nformat.as_deref())?)
            }
            None => None,
        };
        let d = (if pdf { me.docs.pdf.as_mut() } else { me.docs.docx.as_mut() }).ok_or_else(|| rc_err(198, "document not created"))?;
        let t = d.table_mut(&name).ok_or_else(|| rc_err(111, format!("table {name} not found")))?;
        let nrows = t.cells.len();
        let ncols = t.cells.iter().map(Vec::len).max().unwrap_or(0);
        if i.is_some_and(|i| i == 0 || i > nrows) || j.is_some_and(|j| j == 0 || j > ncols) {
            return Err(rc_err(198, format!("table {name}: cell out of range")));
        }
        for r in 0..nrows {
            if i.is_some_and(|i| i - 1 != r) {
                continue;
            }
            for c in 0..ncols {
                if j.is_some_and(|j| j - 1 != c) {
                    continue;
                }
                while t.cells[r].len() <= c {
                    t.cells[r].push(Cell::default());
                }
                let cell = &mut t.cells[r][c];
                if let Some(v) = &value {
                    cell.text = v.clone();
                } else if let Some(f) = &nformat {
                    if let Ok(x) = cell.text.trim().parse::<f64>() {
                        cell.text = num_text(x, Some(f));
                    }
                }
                cell.bold |= bold;
                cell.italic |= italic;
                if halign.is_some() {
                    cell.halign = halign.clone();
                }
            }
        }
        return Ok(String::new());
    }
    let name = lhs;
    let rhs = rhs.ok_or_else(|| rc_err(198, "table: = expected"))?;
    let rl = rhs.to_lowercase();
    let mut cells: Vec<Vec<Cell>> = Vec::new();
    let text_cell = |s: String| Cell { text: s, ..Default::default() };
    if let Some(dims) = paren_inner(&rhs) {
        let (a, b) = dims.split_once(',').ok_or_else(|| rc_err(198, "table: (rows, cols) expected"))?;
        let rows: usize = a.trim().parse().map_err(|_| rc_err(198, "table: invalid rows"))?;
        let cols: usize = b.trim().parse().map_err(|_| rc_err(198, "table: invalid columns"))?;
        cells = vec![vec![Cell::default(); cols]; rows];
    } else if rl.starts_with("data(") || rl.starts_with("data ") {
        let close = rhs.find(')').ok_or_else(|| rc_err(198, "data(varlist) expected"))?;
        let vars_text = rhs[rhs.find('(').unwrap() + 1..close].to_string();
        let ifin = rhs[close + 1..].to_string();
        let c = parse_line(&format!("list {vars_text} {ifin}")).map_err(|e| rc_err(198, e.message))?;
        let ds = &me.engine.dataset;
        let vars = ferrostat_cmd::varlist::expand(ds, &vars_text)?;
        let mask = util::touse(ds, &me.engine.store, &c)?;
        let obsno = has(opts, &["obsno"]);
        if has(opts, &["varnames"]) {
            let mut row: Vec<Cell> = Vec::new();
            if obsno {
                row.push(text_cell(String::new()));
            }
            row.extend(vars.iter().map(|v| text_cell(v.clone())));
            cells.push(row);
        }
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            let mut row: Vec<Cell> = Vec::new();
            if obsno {
                row.push(text_cell((i + 1).to_string()));
            }
            for v in &vars {
                let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
                let s = match (&nformat, var.data[i].is_string()) {
                    (Some(f), false) => num_text(var.data[i].num(), Some(f)),
                    _ => util::cell_text(ds, var, i, true),
                };
                row.push(text_cell(s));
            }
            cells.push(row);
        }
    } else if rl.starts_with("matrix(") {
        let inner = paren_inner(&rhs[6..]).ok_or_else(|| rc_err(198, "matrix(name) expected"))?;
        let m = eval_matrix(&me.engine.dataset, &me.engine.store, inner)?;
        let rown = has(opts, &["rownames"]);
        let coln = has(opts, &["colnames"]);
        if coln {
            let mut row: Vec<Cell> = Vec::new();
            if rown {
                row.push(text_cell(String::new()));
            }
            row.extend(m.colnames.iter().map(|c| text_cell(c.clone())));
            cells.push(row);
        }
        for a in 0..m.rows {
            let mut row: Vec<Cell> = Vec::new();
            if rown {
                row.push(text_cell(m.rownames.get(a).cloned().unwrap_or_default()));
            }
            for b in 0..m.cols {
                let x = m[(a, b)];
                row.push(text_cell(if is_missing(x) { ".".into() } else { num_text(x, nformat.as_deref()) }));
            }
            cells.push(row);
        }
    } else if rl.starts_with("etable") {
        cells = etable(me, nformat.as_deref())?;
    } else if rl.starts_with("returnset") || rl.starts_with("mata(") {
        return Err(rc_err(198, format!("table: {rhs} not supported")));
    } else {
        return Err(rc_err(198, format!("table: {rhs} invalid")));
    }
    let border = match opt_arg(opts, &["border"]) {
        Some(b) => !b.to_lowercase().contains("nil"),
        None => true,
    };
    let title = opt_arg(opts, &["title"]).map(|t| unquote(&t));
    let notes: Vec<String> = opt_arg(opts, &["note", "notes"]).map(|n| vec![unquote(&n)]).unwrap_or_default();
    let d = (if pdf { me.docs.pdf.as_mut() } else { me.docs.docx.as_mut() }).ok_or_else(|| rc_err(198, "document not created"))?;
    if let Some(t) = d.table_mut(&name) {
        *t = Table { name, cells, border, title, notes };
    } else {
        d.blocks.push(Block::Table(Table { name, cells, border, title, notes }));
    }
    Ok(String::new())
}

/// Coefficient table of the last estimation: coefficient, standard error,
/// test statistic, p-value and confidence interval.
fn etable(me: &AdoInterpreter, nformat: Option<&str>) -> Result<Vec<Vec<Cell>>, String> {
    let e = &me.engine.store.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V");
    let df = e.scalars.get("df_r").copied().filter(|d| !is_missing(*d) && *d > 0.0);
    let level = me.engine.store.level;
    let crit = match df {
        Some(d) => ferrostat_cmd::dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        None => ferrostat_cmd::dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    let f = |x: f64, d: &str| if is_missing(x) || !x.is_finite() { String::new() } else { format::fmt(x, nformat.unwrap_or(d)).trim().to_string() };
    let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
    let stat = if df.is_some() { "t" } else { "z" };
    let mut rows = vec![[depvar, "Coef.".into(), "Std. err.".into(), stat.into(), format!("P>|{stat}|"), format!("[{}% conf. interval]", level), String::new()].map(|s| Cell { text: s, ..Default::default() }).to_vec()];
    for k in 0..b.cols {
        let coef = b[(0, k)];
        let se = v.map(|v| v[(k, k)].max(0.0).sqrt()).unwrap_or(f64::NAN);
        let ts = coef / se;
        let p = match df {
            Some(d) => 2.0 * ferrostat_cmd::dist::ttail(d, ts.abs()),
            None => 2.0 * (1.0 - ferrostat_cmd::dist::normal(ts.abs())),
        };
        let name = b.colnames.get(k).cloned().unwrap_or_default();
        let name = name.rsplit(':').next().unwrap_or(&name).to_string();
        let cells = [name, f(coef, "%9.0g"), f(se, "%9.0g"), f(ts, "%5.2f"), f(p, "%5.3f"), f(coef - crit * se, "%9.0g"), f(coef + crit * se, "%9.0g")];
        rows.push(cells.map(|s| Cell { text: s, ..Default::default() }).to_vec());
    }
    Ok(rows)
}

/// Absolute form of a path, for messages.
pub(crate) fn full_path(path: &str) -> String {
    if path.starts_with('/') || path.contains(':') {
        path.to_string()
    } else {
        format!("{}/{}", vfs::current_dir().trim_end_matches(['/', '\\']), path)
    }
}
