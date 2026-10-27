//! coefplot: point estimates and confidence intervals from stored estimation
//! results (or matrices), drawn with the twoway graph engine.
//!
//! `coefplot plot [plot ...] [|| plot ... ] [, options]`: a plot is a stored
//! estimates name, `.` (the active results), `*` (every stored result),
//! `matrix(name[r, c])`, or `(model [\ model] [, plot options])`. `||`
//! separates subgraphs, drawn side by side on a common scale. Options are
//! looked up in the plot, then the subgraph, then the global options.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, parse_line, split_top_level_words};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_trim;
use crate::graph::{finish, palette, Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub struct Coefplot;

/// One coefficient of one plot.
#[derive(Debug, Clone)]
struct Coef {
    eq: String,
    name: String,
    b: f64,
    /// confidence intervals, one per level
    cis: Vec<(f64, f64)>,
    /// marker label from mlabels()
    mlbl: String,
}

struct Plot {
    opts: Command,
    label: String,
    coefs: Vec<Coef>,
}

struct Subgraph {
    opts: Command,
    plots: Vec<Plot>,
}

/// Options looked up from the most specific level outwards.
struct Opts<'a>(Vec<&'a Command>);

impl<'a> Opts<'a> {
    fn get(&self, name: &str, min: usize) -> Option<&'a str> {
        self.0.iter().find(|c| c.has_opt(name, min)).and_then(|c| c.opt(name, min))
    }
    fn has(&self, name: &str, min: usize) -> bool {
        self.0.iter().any(|c| c.has_opt(name, min))
    }
    /// An option spelled exactly `name` (for names that prefix others, like mlabels).
    fn exact(&self, name: &str) -> Option<&'a str> {
        self.0.iter().find_map(|c| c.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, a)| a.as_deref().unwrap_or("")))
    }
    fn num(&self, name: &str, min: usize) -> Result<Option<f64>, String> {
        match self.0.iter().find(|c| c.has_opt(name, min)) {
            Some(c) => util::opt_num(c, name, min),
            None => Ok(None),
        }
    }
}

fn opts_of(text: &str) -> Result<Command, String> {
    parse_line(&format!("o , {text}")).map_err(|e| rc_err(198, e))
}

/// Split at top-level `||`.
fn split_bars(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    loop {
        let mut cut = None;
        let mut from = 0;
        while let Some(i) = find_top_level(&rest[from..], '|') {
            let i = from + i;
            if rest[i + 1..].starts_with('|') {
                cut = Some(i);
                break;
            }
            from = i + 1;
        }
        match cut {
            Some(i) => {
                out.push(rest[..i].to_string());
                rest = &rest[i + 2..];
            }
            None => {
                out.push(rest.to_string());
                return out;
            }
        }
    }
}

/// Text before and after the first top-level comma.
fn split_comma(s: &str) -> (String, String) {
    match find_top_level(s, ',') {
        Some(i) => (s[..i].trim().to_string(), s[i + 1..].trim().to_string()),
        None => (s.trim().to_string(), String::new()),
    }
}

/// Glob match with `*` and `?`.
fn glob(pat: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, None, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// Does a coefficient pattern match? Patterns with `:` also name the equation.
fn coef_match(pat: &str, eq: &str, name: &str) -> bool {
    match pat.split_once(':') {
        Some((pe, pn)) => (pe.is_empty() || glob(pe, eq)) && (pn.is_empty() || glob(pn, name)),
        None => glob(pat, name),
    }
}

/// `pattern = value ...` lists (coeflabels, rename, groups): the text before
/// a top-level comma, values unquoted.
fn assign_list(s: &str) -> Vec<(String, String)> {
    let (list, _) = split_comma(s);
    let mut out: Vec<(String, String)> = Vec::new();
    let mut want_value = false;
    for w in split_top_level_words(&list) {
        let quote = w.find('"').unwrap_or(w.len());
        if want_value {
            if let Some(last) = out.last_mut() {
                last.1 = util::unquote(&w);
            }
            want_value = false;
        } else if let Some(i) = w[..quote].find('=') {
            let (k, v) = (w[..i].trim(), w[i + 1..].trim());
            if !k.is_empty() {
                out.push((k.to_string(), String::new()));
            }
            if v.is_empty() {
                want_value = true;
            } else if let Some(last) = out.last_mut() {
                last.1 = util::unquote(v);
            }
        } else {
            out.push((w.clone(), String::new()));
        }
    }
    out
}

/// Words of a list option, unquoted (eqlabels, plotlabels, bylabels).
fn word_list(s: &str) -> Vec<String> {
    split_top_level_words(&split_comma(s).0).iter().map(|w| util::unquote(w)).collect()
}

/// Omitted and base-level coefficients (`0b.treat`, `o.x`, `1.a#0b.b`).
fn is_omitted(name: &str) -> bool {
    name.split('#').any(|part| {
        part.split_once('.').is_some_and(|(op, _)| {
            let op = op.to_lowercase();
            (op.ends_with('b') || op.ends_with('o')) && (op.starts_with(|c: char| c.is_ascii_digit()) || op == "o")
        })
    })
}

/// Default text of a coefficient: value labels for factor levels.
fn default_label(ds: &DataSet, name: &str, varlabels: bool) -> String {
    let part_label = |part: &str| -> Option<String> {
        if let Some((op, var)) = part.split_once('.') {
            let level: f64 = op.trim_end_matches(|c: char| c.is_ascii_alphabetic()).parse().ok()?;
            let v = ds.get_var(var)?;
            let vl = ds.value_labels.get(&v.value_label_name)?;
            return vl.lookup(level).map(String::from);
        }
        if part == "_cons" {
            return Some("Constant".into());
        }
        if varlabels {
            return ds.get_var(part).map(|v| v.label.clone()).filter(|l| !l.is_empty());
        }
        None
    };
    let parts: Vec<&str> = name.split('#').collect();
    let labels: Vec<Option<String>> = parts.iter().map(|p| part_label(p)).collect();
    if labels.iter().all(|l| l.is_some()) {
        labels.into_iter().map(|l| l.unwrap()).collect::<Vec<_>>().join(" # ")
    } else {
        name.to_string()
    }
}

/// Critical values for the confidence levels (t when the model has residual df).
fn crit(levels: &[f64], df: Option<f64>) -> Vec<f64> {
    levels
        .iter()
        .map(|l| {
            let a = (1.0 - l / 100.0) / 2.0;
            match df {
                Some(d) if d > 0.0 && !is_missing(d) => dist::invttail(d, a),
                _ => dist::invnormal(1.0 - a),
            }
        })
        .collect()
}

/// Coefficients of a stored or active estimation result.
fn estimate_coefs(e: &Results, levels: &[f64]) -> Result<Vec<Coef>, String> {
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V");
    let z = crit(levels, e.scalars.get("df_r").copied());
    let mut out = Vec::new();
    for j in 0..b.cols {
        let full = b.colnames.get(j).cloned().unwrap_or_else(|| format!("c{}", j + 1));
        let (eq, name) = match full.split_once(':') {
            Some((q, n)) => (q.to_string(), n.to_string()),
            None => ("_".to_string(), full.clone()),
        };
        let bj = b.get(0, j);
        let se = v.filter(|m| m.rows > j && m.cols > j).map(|m| m.get(j, j).max(0.0).sqrt());
        if is_missing(bj) || is_omitted(&name) || (bj == 0.0 && se == Some(0.0) && name != "_cons") {
            continue;
        }
        let cis = match se {
            Some(s) if !is_missing(s) => z.iter().map(|z| (bj - z * s, bj + z * s)).collect(),
            _ => Vec::new(),
        };
        out.push(Coef { eq, name, b: bj, cis, mlbl: String::new() });
    }
    Ok(out)
}

/// Row or column index of a matrix from a number (1-based) or a name.
fn mat_index(spec: &str, names: &[String], n: usize) -> Result<usize, String> {
    let s = spec.trim();
    let k = match s.parse::<usize>() {
        Ok(k) => k.checked_sub(1),
        Err(_) => names.iter().position(|x| x == s),
    };
    k.filter(|k| *k < n).ok_or_else(|| rc_err(503, format!("{s}: invalid matrix subscript")))
}

/// `matrix(name[rows, cols])`: estimates from a row or column of a matrix,
/// confidence limits from two other rows/columns given by ci((lo hi)).
fn matrix_coefs(st: &Store, spec: &str, o: &Opts) -> Result<(String, Vec<Coef>), String> {
    let (name, sub) = match spec.split_once('[') {
        Some((n, s)) => (n.trim(), Some(s.trim_end_matches(']'))),
        None => (spec.trim(), None),
    };
    let m: &Matrix = st.matrices.get(name).ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
    // coefficients run along the columns of a selected row, or down a selected column
    let (mut along_cols, mut pick) = (m.rows == 1 || m.cols != 1, 0usize);
    if let Some(s) = sub {
        let (r, c) = s.split_once(',').map_or((s, "."), |(r, c)| (r, c));
        if r.trim() == "." || r.trim().is_empty() {
            along_cols = false;
            pick = if c.trim() == "." { 0 } else { mat_index(c, &m.colnames, m.cols)? };
        } else {
            along_cols = true;
            pick = mat_index(r, &m.rownames, m.rows)?;
        }
    }
    let (other_names, other_n) = if along_cols { (&m.rownames, m.rows) } else { (&m.colnames, m.cols) };
    if let Some(bspec) = o.get("b", 1) {
        pick = mat_index(bspec, other_names, other_n)?;
    }
    let at = |k: usize, other: usize| if along_cols { m.get(other, k) } else { m.get(k, other) };
    let ci = match o.get("ci", 2) {
        Some(c) => {
            let inner = c.trim().trim_start_matches('(').trim_end_matches(')');
            let w: Vec<&str> = inner.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "ci(): two rows or columns required"));
            }
            Some((mat_index(w[0], other_names, other_n)?, mat_index(w[1], other_names, other_n)?))
        }
        None => None,
    };
    let (n, names) = if along_cols { (m.cols, &m.colnames) } else { (m.rows, &m.rownames) };
    let mut out = Vec::new();
    for k in 0..n {
        let b = at(k, pick);
        if is_missing(b) {
            continue;
        }
        let cis = ci.map(|(lo, hi)| (at(k, lo), at(k, hi))).filter(|(l, h)| !is_missing(*l) && !is_missing(*h)).into_iter().collect();
        let full = names.get(k).cloned().unwrap_or_else(|| format!("r{}", k + 1));
        let (eq, nm) = full.split_once(':').map_or(("_".to_string(), full.clone()), |(q, x)| (q.to_string(), x.to_string()));
        out.push(Coef { eq, name: nm, b, cis, mlbl: String::new() });
    }
    Ok((name.to_string(), out))
}

/// Names of stored estimates in the order they were stored.
fn stored_names(st: &Store) -> Vec<String> {
    let mut names: Vec<String> = st.estimates_order.iter().filter(|n| st.estimates.contains_key(*n)).cloned().collect();
    let rest: Vec<String> = st.estimates.keys().filter(|k| !names.contains(k)).cloned().collect();
    names.extend(rest);
    names
}

/// Expand a model word: `*` and wildcards to every matching stored result.
fn expand_model(st: &Store, w: &str) -> Result<Vec<String>, String> {
    if w.contains('*') || w.contains('?') {
        let names: Vec<String> = stored_names(st).into_iter().filter(|n| glob(w, n)).collect();
        if names.is_empty() {
            return Err(rc_err(111, format!("estimation result {w} not found")));
        }
        return Ok(names);
    }
    Ok(vec![w.to_string()])
}

fn is_matrix_model(w: &str) -> bool {
    let l = w.to_lowercase();
    l.split_once('(').is_some_and(|(k, _)| k.len() >= 3 && "matrix".starts_with(k)) && l.ends_with(')')
}

/// Coefficients of one model word, and its default plot label.
fn model_coefs(st: &Store, w: &str, o: &Opts, levels: &[f64]) -> Result<(String, Vec<Coef>), String> {
    if is_matrix_model(w) {
        let inner = &w[w.find('(').unwrap() + 1..w.len() - 1];
        return matrix_coefs(st, inner, o);
    }
    let e = if w == "." {
        if st.e.matrices.get("b").is_none() {
            return Err(rc_err(301, "last estimates not found"));
        }
        &st.e
    } else {
        st.estimates.get(w).ok_or_else(|| rc_err(111, format!("estimation result {w} not found")))?
    };
    let label = if w == "." { e.macros.get("depvar").cloned().filter(|d| !d.is_empty()).unwrap_or_else(|| "(active)".into()) } else { w.to_string() };
    Ok((label, estimate_coefs(e, levels)?))
}

/// Parse one subgraph's plot list into plots (coefficients not yet filtered).
fn parse_plots(st: &Store, list: &str, sub: &Command, global: &Command, levels_of: &dyn Fn(&Opts) -> Result<Vec<f64>, String>) -> Result<Vec<Plot>, String> {
    let mut plots = Vec::new();
    let words = split_top_level_words(list);
    let mut models: Vec<(Vec<String>, String)> = Vec::new();
    for w in &words {
        if w.starts_with('(') && w.ends_with(')') {
            let (m, opts) = split_comma(&w[1..w.len() - 1]);
            let names: Vec<String> = split_top_level_words(&m.replace('\\', " ")).into_iter().flat_map(|x| expand_model(st, &x).unwrap_or_else(|_| vec![x])).collect();
            models.push((if names.is_empty() { vec![".".to_string()] } else { names }, opts));
        } else {
            for n in expand_model(st, w)? {
                models.push((vec![n], String::new()));
            }
        }
    }
    if models.is_empty() {
        models.push((vec![".".to_string()], String::new()));
    }
    for (names, opts) in models {
        let pc = opts_of(&opts)?;
        let mut coefs = Vec::new();
        let mut label = String::new();
        {
            let o = Opts(vec![&pc, sub, global]);
            let levels = levels_of(&o)?;
            for (k, n) in names.iter().enumerate() {
                let (l, mut c) = model_coefs(st, n, &o, &levels)?;
                if k == 0 {
                    label = l;
                }
                // aseq: the model name becomes the equation of all its coefficients
                if let Some(a) = o.get("asequation", 4) {
                    c.iter_mut().for_each(|x| x.eq = util::unquote(a));
                } else if o.has("asequation", 4) {
                    c.iter_mut().for_each(|x| x.eq = n.clone());
                }
                coefs.extend(c);
            }
        }
        plots.push(Plot { opts: pc, label, coefs });
    }
    Ok(plots)
}

#[derive(Debug, Clone, PartialEq)]
enum Row {
    Heading(String),
    Coef(String, String),
}

/// Graph values of a plot: category position and coefficient.
struct Placed {
    at: f64,
    coef: Coef,
}

fn intensity(rgb: (u8, u8, u8), f: f64) -> (u8, u8, u8) {
    if f >= 1.0 {
        return rgb;
    }
    let mix = |c: u8| (c as f64 * f + 255.0 * (1.0 - f)).round() as u8;
    (mix(rgb.0), mix(rgb.1), mix(rgb.2))
}

/// A colour option: a colour name, or `*#` for an intensity of `base`.
fn colour_opt(spec: &str, base: (u8, u8, u8)) -> Result<(u8, u8, u8), String> {
    let s = util::unquote(split_comma(spec).0.trim());
    if let Some(f) = s.strip_prefix('*') {
        let f: f64 = f.trim().parse().map_err(|_| rc_err(198, format!("{s}: invalid color")))?;
        return Ok(intensity(base, f));
    }
    crate::colors::parse(&s).ok_or_else(|| rc_err(198, format!("{s}: invalid color")))
}

/// Numbers mentioned by an axis option: `r(0 7500)`, `0(1500)7500`, `0 5 10`.
fn axis_numbers(s: &str) -> Vec<f64> {
    let s = split_comma(s).0;
    let s = s.trim();
    let s = s.strip_prefix("range(").or_else(|| s.strip_prefix("r(")).map(|x| x.trim_end_matches(')')).unwrap_or(s);
    s.split(|c: char| c.is_whitespace() || c == '(' || c == ')').filter_map(|w| w.parse::<f64>().ok()).collect()
}

/// legend(): (shown?, order with texts); `label(# "text")` entries fill the texts.
fn parse_legend(l: &str) -> (Option<bool>, Option<Vec<(usize, String)>>, Vec<(usize, String)>) {
    let c = match opts_of(l) {
        Ok(c) => c,
        Err(_) => return (None, None, Vec::new()),
    };
    let shown = if c.has_opt("off", 3) {
        Some(false)
    } else if c.has_opt("on", 2) {
        Some(true)
    } else {
        None
    };
    let order = c.opt("order", 3).map(|o| {
        let mut v: Vec<(usize, String)> = Vec::new();
        for w in split_top_level_words(o) {
            if let Ok(k) = w.parse::<usize>() {
                v.push((k, String::new()));
            } else if let Some(last) = v.last_mut() {
                let t = util::unquote(&w);
                last.1 = if last.1.is_empty() { t } else { format!("{} {t}", last.1) };
            }
        }
        v
    });
    let labels = c
        .opts
        .iter()
        .filter(|(n, _)| n.len() >= 3 && "label".starts_with(n.as_str()))
        .filter_map(|(_, a)| {
            let a = a.as_deref()?;
            let w = split_top_level_words(a);
            let k = w.first()?.trim_end_matches(',').parse::<usize>().ok()?;
            Some((k, w[1..].iter().map(|x| util::unquote(x)).collect::<Vec<_>>().join(" ")))
        })
        .collect();
    (shown, order, labels)
}

/// Value of an internal variable (`@b`, `@ll`, `@ul`, `@at`) for addplot().
fn internal(p: &Placed, var: &str) -> Option<f64> {
    match var {
        "@b" => Some(p.coef.b),
        "@ll" | "@ll1" => p.coef.cis.first().map(|c| c.0),
        "@ul" | "@ul1" => p.coef.cis.first().map(|c| c.1),
        "@at" => Some(p.at),
        _ => None,
    }
}

/// addplot(): twoway plots of the internal variables of all plotted coefficients.
fn addplot_layers(text: &str, placed: &[Placed]) -> Result<Vec<Layer>, String> {
    let mut out = Vec::new();
    let parts: Vec<String> = if text.trim_start().starts_with('(') {
        split_top_level_words(text).into_iter().map(|p| p.trim().trim_start_matches('(').trim_end_matches(')').to_string()).collect()
    } else {
        split_bars(text)
    };
    for p in parts.iter().filter(|p| !p.trim().is_empty()) {
        let c = parse_line(p.trim()).map_err(|e| rc_err(198, e))?;
        let vars: Vec<String> = c.main.split_whitespace().map(String::from).collect();
        if vars.len() < 2 || vars.iter().any(|v| !v.starts_with('@')) {
            return Err(rc_err(198, "addplot(): only the internal variables @b @ll @ul @at are available"));
        }
        let col = |v: &str| -> Vec<Option<f64>> { placed.iter().map(|q| internal(q, v)).collect() };
        let cols: Vec<Vec<Option<f64>>> = vars.iter().map(|v| col(v)).collect();
        let rows: Vec<usize> = (0..placed.len()).filter(|&i| cols.iter().all(|c| c[i].is_some())).collect();
        let val = |k: usize, i: usize| cols[k][i].unwrap();
        let mut l = Layer::default();
        match c.verb.as_str() {
            "scatter" | "sc" | "line" | "connected" => {
                l.kind = match c.verb.as_str() {
                    "line" => LayerKind::Line,
                    "connected" => LayerKind::Connected,
                    _ => LayerKind::Scatter,
                };
                l.points = rows.iter().map(|&i| (val(1, i), val(0, i))).collect();
                if let Some(m) = c.opt("mlabel", 4) {
                    let m = m.trim();
                    l.mlabels = rows
                        .iter()
                        .map(|&i| if m == "@mlbl" { placed[i].coef.mlbl.clone() } else { internal(&placed[i], m).map(|v| fmt_trim(v, c.opt("format", 3).unwrap_or("%9.0g").trim())).unwrap_or_else(|| util::unquote(m)) })
                        .collect();
                }
            }
            verb => {
                let style = match verb {
                    "rcap" => RangeStyle::Cap,
                    "rspike" => RangeStyle::Spike,
                    "rbar" => RangeStyle::Bar,
                    _ => return Err(rc_err(198, format!("addplot(): {verb} not supported"))),
                };
                if vars.len() != 3 {
                    return Err(rc_err(198, format!("{verb}: y1var y2var xvar required")));
                }
                l.kind = LayerKind::Range(style);
                l.ranges = rows.iter().map(|&i| (val(2, i), val(0, i), val(1, i))).collect();
                l.horizontal = c.has_opt("horizontal", 3);
                l.width = util::opt_num(&c, "barwidth", 4)?.unwrap_or(0.5);
            }
        }
        if let Some(s) = c.opt("msymbol", 2) {
            l.symbol = s.trim().to_string();
        }
        for name in ["mlabcolor", "mcolor", "lcolor", "color"] {
            if let Some(s) = c.opt(name, if name == "color" { 5 } else { 4 }) {
                if let Some(rgb) = crate::colors::parse(s) {
                    l.color = Some(rgb);
                    break;
                }
            }
        }
        out.push(l);
    }
    Ok(out)
}

impl CommandHandler for Coefplot {
    fn name(&self) -> &'static str {
        "coefplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // the text after the command name (which may be followed directly by a comma)
        let raw = cmd.raw.trim();
        let body = raw[raw.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(raw.len())..].trim().to_string();
        let segments = split_bars(&body);
        // the options of the last segment are the global options
        let (_, gtext) = split_comma(segments.last().map(String::as_str).unwrap_or(""));
        let global = opts_of(&gtext)?;
        let default_level = st.level;
        let levels_of = |o: &Opts| -> Result<Vec<f64>, String> {
            if let Some(l) = o.get("levels", 6) {
                let v: Vec<f64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if !v.is_empty() {
                    return Ok(v);
                }
            }
            Ok(vec![o.num("level", 5)?.unwrap_or(default_level)])
        };
        let mut subs: Vec<Subgraph> = Vec::new();
        let nseg = segments.len();
        for (k, seg) in segments.iter().enumerate() {
            let (list, stext) = split_comma(seg);
            if list.is_empty() && k + 1 == nseg && nseg > 1 {
                // `|| , options`: global options only
                continue;
            }
            let sc = if k + 1 == nseg { opts_of("")? } else { opts_of(&stext)? };
            let plots = parse_plots(st, &list, &sc, &global, &levels_of)?;
            subs.push(Subgraph { opts: sc, plots });
        }
        let g = Opts(vec![&global]);
        let vertical = g.has("vertical", 4) && !g.has("horizontal", 3);
        let eqstrict = g.has("eqstrict", 4);
        // variable labels name the coefficients unless nolabels is given
        let varlabels = !g.has("nolabels", 5);

        // filter and rename the coefficients of every plot
        for s in subs.iter_mut() {
            let sc = &s.opts;
            for p in s.plots.iter_mut() {
                let o = Opts(vec![&p.opts, sc, &global]);
                let neq = {
                    let mut eqs: Vec<&str> = p.coefs.iter().map(|c| c.eq.as_str()).collect();
                    eqs.dedup();
                    eqs.len()
                };
                if !eqstrict && neq <= 1 && !o.has("asequation", 4) {
                    p.coefs.iter_mut().for_each(|c| c.eq = "_".into());
                }
                if let Some(r) = o.get("rename", 6) {
                    let pairs = assign_list(r);
                    for c in p.coefs.iter_mut() {
                        if let Some((_, new)) = pairs.iter().find(|(old, _)| coef_match(old, &c.eq, &c.name)) {
                            c.name = new.clone();
                        }
                    }
                }
                if let Some(k) = o.get("keep", 4) {
                    let pats = split_top_level_words(k);
                    p.coefs.retain(|c| pats.iter().any(|pt| coef_match(pt, &c.eq, &c.name)));
                }
                if let Some(d) = o.get("drop", 4) {
                    let pats = split_top_level_words(d);
                    p.coefs.retain(|c| !pats.iter().any(|pt| coef_match(pt, &c.eq, &c.name)));
                }
                if o.has("noci", 4) {
                    p.coefs.iter_mut().for_each(|c| c.cis.clear());
                }
                // mlabels(coef = position "text" ...)
                if let Some(m) = o.exact("mlabels") {
                    let words = split_top_level_words(m);
                    let mut i = 0;
                    while i < words.len() {
                        let w = &words[i];
                        let pat = w.trim_end_matches('=').to_string();
                        i += 1;
                        if i < words.len() && words[i] == "=" {
                            i += 1;
                        }
                        if i < words.len() && words[i].parse::<f64>().is_ok() {
                            i += 1;
                        }
                        let text = words.get(i).map(|t| util::unquote(t)).unwrap_or_default();
                        i += 1;
                        for c in p.coefs.iter_mut().filter(|c| coef_match(&pat, &c.eq, &c.name)) {
                            c.mlbl = text.clone();
                        }
                    }
                }
            }
        }
        if subs.iter().all(|s| s.plots.iter().all(|p| p.coefs.is_empty())) {
            return Err(rc_err(2000, "no coefficients to plot"));
        }

        // category rows: equations in order of appearance, coefficients within them
        let mut keys: Vec<(String, String)> = Vec::new();
        for s in &subs {
            for p in &s.plots {
                for c in &p.coefs {
                    let k = (c.eq.clone(), c.name.clone());
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
            }
        }
        let mut eqs: Vec<String> = Vec::new();
        for (e, _) in &keys {
            if !eqs.contains(e) {
                eqs.push(e.clone());
            }
        }
        let mut ordered: Vec<(String, String)> = eqs.iter().flat_map(|e| keys.iter().filter(move |k| &k.0 == e).cloned()).collect();
        if let Some(o) = g.get("order", 5) {
            let mut front = Vec::new();
            for pat in split_top_level_words(o) {
                for k in &ordered {
                    if coef_match(&pat, &k.0, &k.1) && !front.contains(k) {
                        front.push(k.clone());
                    }
                }
            }
            let rest: Vec<(String, String)> = ordered.iter().filter(|k| !front.contains(k)).cloned().collect();
            front.extend(rest);
            ordered = front;
        }
        let eq_headings = eqs.len() > 1 && !g.has("noeqlabels", 6);
        let eqlabels = g.get("eqlabels", 4).map(|s| {
            let pairs = assign_list(s);
            if pairs.iter().any(|(_, v)| !v.is_empty()) {
                (Vec::new(), pairs)
            } else {
                (word_list(s), Vec::new())
            }
        });
        let eq_text = |e: &str| -> String {
            if let Some((list, pairs)) = &eqlabels {
                if let Some((_, v)) = pairs.iter().find(|(p, _)| glob(p, e)) {
                    return v.clone();
                }
                if let Some(t) = eqs.iter().position(|x| x == e).and_then(|i| list.get(i)) {
                    return t.clone();
                }
            }
            if g.get("eqlabels", 4).is_some_and(|s| split_comma(s).1.split_whitespace().any(|w| w == "label" || w == "lab")) {
                if let Some(v) = ds.get_var(e).filter(|v| !v.label.is_empty()) {
                    return v.label.clone();
                }
            }
            e.to_string()
        };
        let groups: Vec<(String, String)> = g.get("groups", 5).or_else(|| g.get("headings", 4)).map(assign_list).unwrap_or_default();
        let mut rows: Vec<Row> = Vec::new();
        let (mut cur_eq, mut cur_group): (Option<String>, Option<usize>) = (None, None);
        for (e, n) in &ordered {
            if eq_headings && cur_eq.as_deref() != Some(e.as_str()) {
                rows.push(Row::Heading(eq_text(e)));
                cur_group = None;
            }
            cur_eq = Some(e.clone());
            let gi = groups.iter().position(|(p, _)| coef_match(p, e, n));
            if gi.is_some() && gi != cur_group {
                rows.push(Row::Heading(groups[gi.unwrap()].1.clone()));
            }
            cur_group = gi;
            rows.push(Row::Coef(e.clone(), n.clone()));
        }
        let nrows = rows.len();
        // category axis coordinate of row position `at` (1-based; the first row on top when horizontal)
        let coord = |at: f64| if vertical { at } else { nrows as f64 + 1.0 - at };
        let coeflabels = g.get("coeflabels", 5).map(assign_list).unwrap_or_default();
        let blank_cats = g.get(if vertical { "xlabel" } else { "ylabel" }, 4).is_some_and(|s| s.trim() == "\"\"");
        let cat_labels: Vec<(f64, String)> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let text = match r {
                    Row::Heading(h) => h.clone(),
                    Row::Coef(e, n) => coeflabels.iter().find(|(p, _)| coef_match(p, e, n)).map(|(_, l)| l.clone()).unwrap_or_else(|| default_label(ds, n, varlabels)),
                };
                (coord(i as f64 + 1.0), if blank_cats { String::new() } else { text })
            })
            .collect();
        let row_of = |e: &str, n: &str| rows.iter().position(|r| matches!(r, Row::Coef(re, rn) if re == e && rn == n));

        let scheme = match global.opt("scheme", 3).map(|s| s.trim().to_lowercase()) {
            Some(s) if crate::graph::is_scheme(&s) => s,
            _ => st.scheme.clone(),
        };
        let pal = palette(&scheme);
        let plotlabels = g.get("plotlabels", 5).map(word_list).unwrap_or_default();
        let bylabels = g.get("bylabels", 5).map(word_list).unwrap_or_default();
        let (legend_shown, legend_order, legend_labels) = g.get("legend", 3).map(parse_legend).unwrap_or((None, None, Vec::new()));
        // values named by the value axis options (yscale(r()), ylabel(), ...)
        let value_axis = if vertical { 'y' } else { 'x' };
        let axis_vals: Vec<f64> = [format!("{value_axis}scale"), format!("{value_axis}label")].iter().filter_map(|n| global.opt(n, 4)).flat_map(axis_numbers).collect();
        // extent of all estimates and intervals, for the base of bars
        let (all_lo, all_hi) = subs.iter().flat_map(|s| s.plots.iter()).flat_map(|p| p.coefs.iter()).flat_map(|c| std::iter::once(c.b).chain(c.cis.iter().flat_map(|x| [x.0, x.1]))).chain(axis_vals.iter().copied()).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        let mut graphs: Vec<Graph> = Vec::new();
        let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for (si, s) in subs.iter().enumerate() {
            let n = s.plots.len();
            let mut layers: Vec<Layer> = Vec::new();
            let mut draw: Vec<usize> = Vec::new();
            let mut placed_all: Vec<Placed> = Vec::new();
            for (pi, p) in s.plots.iter().enumerate() {
                let o = Opts(vec![&p.opts, &s.opts, &global]);
                let offset = match o.num("offset", 6)? {
                    Some(v) => v,
                    // the first plot on top (horizontal) or at the left (vertical)
                    None if n > 1 && !o.has("nooffset", 5) => (pi as f64 - (n as f64 - 1.0) / 2.0) / (n as f64 + 1.0) * if vertical { 1.0 } else { -1.0 },
                    None => 0.0,
                };
                let cycle = o.num("pcycle", 6)?.map_or(usize::MAX, |c| c.max(1.0) as usize);
                let base = pal[(pi % cycle) % pal.len()];
                let base = (base.0, base.1, base.2);
                let mut colour = base;
                for (name, min) in [("color", 5), ("mcolor", 2)] {
                    if let Some(c) = o.get(name, min) {
                        colour = colour_opt(c, base)?;
                        break;
                    }
                }
                let placed: Vec<Placed> = p.coefs.iter().filter_map(|c| row_of(&c.eq, &c.name).map(|r| Placed { at: coord(r as f64 + 1.0) + offset, coef: c.clone() })).collect();
                let label = plotlabels.get(pi + if si > 0 { subs[..si].iter().map(|x| x.plots.len()).sum::<usize>() } else { 0 }).cloned().or_else(|| o.get("label", 3).map(util::unquote)).unwrap_or_else(|| p.label.clone());
                let recast = o.get("recast", 3).map(|r| r.trim().to_lowercase()).unwrap_or_else(|| "scatter".into());
                // the estimates
                let mut est = Layer { label, color: Some(colour), ..Default::default() };
                let show_labels = (o.has("mlabel", 4) || p.coefs.iter().any(|c| !c.mlbl.is_empty())) && o.get("mlabcolor", 6).map(|c| c.trim()) != Some("none");
                let fmt = o.get("format", 3).unwrap_or("%9.0g").trim().to_string();
                let mlabels: Vec<String> = placed.iter().map(|q| if !q.coef.mlbl.is_empty() || !o.has("mlabel", 4) { q.coef.mlbl.clone() } else { fmt_trim(q.coef.b, &fmt) }).collect();
                let pt = |at: f64, v: f64| if vertical { (at, v) } else { (v, at) };
                match recast.as_str() {
                    "bar" | "hbar" | "rbar" => {
                        let fill = match o.get("fcolor", 2).or_else(|| o.get("fintensity", 4)) {
                            Some(f) => colour_opt(f, colour)?,
                            None => colour,
                        };
                        est.kind = LayerKind::Range(RangeStyle::Bar);
                        est.color = Some(fill);
                        est.horizontal = !vertical;
                        let step = if n > 1 && !o.has("nooffset", 5) { 1.0 / (n as f64 + 1.0) } else { 1.0 };
                        est.width = o.num("barwidth", 4)?.unwrap_or(step * 0.9);
                        // bars start at zero when it is near the estimates, otherwise at the axis edge
                        let (lo, hi) = (all_lo, all_hi);
                        let zero_near = lo <= 0.0 || lo <= 0.5 * (hi - lo).max(hi.abs() * 1e-9);
                        let base_v = o.num("base", 4)?.unwrap_or(if zero_near || !lo.is_finite() { 0.0 } else { lo });
                        est.ranges = placed.iter().map(|q| (q.at, base_v, q.coef.b)).collect();
                        if show_labels {
                            est.mlabels = mlabels.clone();
                        }
                    }
                    other => {
                        est.kind = match other {
                            "line" => LayerKind::Line,
                            "connected" => LayerKind::Connected,
                            _ => LayerKind::Scatter,
                        };
                        let mut pts: Vec<(f64, f64)> = placed.iter().map(|q| pt(q.at, q.coef.b)).collect();
                        if est.kind != LayerKind::Scatter {
                            pts.sort_by(|a, b| if vertical { a.0.total_cmp(&b.0) } else { a.1.total_cmp(&b.1) });
                        }
                        est.points = pts;
                        if let Some(sym) = o.get("msymbol", 2) {
                            est.symbol = sym.trim().to_string();
                            est.hollow = crate::graph::symbol_shape(&est.symbol).1;
                        }
                        if show_labels {
                            est.mlabels = mlabels.clone();
                        }
                    }
                }
                for q in &placed {
                    for v in std::iter::once(q.coef.b).chain(q.coef.cis.iter().flat_map(|c| [c.0, c.1])) {
                        vmin = vmin.min(v);
                        vmax = vmax.max(v);
                    }
                    if let Some((_, b0, _)) = est.ranges.first() {
                        vmin = vmin.min(*b0);
                        vmax = vmax.max(*b0);
                    }
                }
                let first = layers.len();
                layers.push(est);
                // the confidence intervals: one layer per level, the widest lightest
                let cio = o.get("ciopts", 6).map(opts_of).transpose()?;
                let nlev = placed.iter().map(|q| q.coef.cis.len()).max().unwrap_or(0);
                for li in 0..nlev {
                    let style = match cio.as_ref().and_then(|c| c.opt("recast", 3)).map(|r| r.trim().to_lowercase()).as_deref() {
                        Some("rcap") => RangeStyle::Cap,
                        Some("rbar") => RangeStyle::Bar,
                        Some("rline") => RangeStyle::Line,
                        Some("rarea") => RangeStyle::Area,
                        _ => RangeStyle::Spike,
                    };
                    let mut ci_col = colour;
                    if let Some(c) = cio.as_ref().and_then(|c| c.opt("lcolor", 2).or_else(|| c.opt("color", 5))) {
                        ci_col = colour_opt(c, colour)?;
                    }
                    if cio.as_ref().is_none_or(|c| !c.has_opt("lcolor", 2) && !c.has_opt("color", 5)) && matches!(recast.as_str(), "bar" | "hbar" | "rbar") {
                        // intervals over bars of the same colour: a darker shade
                        ci_col = ((ci_col.0 as f64 * 0.6) as u8, (ci_col.1 as f64 * 0.6) as u8, (ci_col.2 as f64 * 0.6) as u8);
                    }
                    if nlev > 1 {
                        ci_col = intensity(ci_col, 0.45 + 0.55 * (li as f64 + 1.0) / nlev as f64);
                    }
                    let mut l = Layer { kind: LayerKind::Range(style), horizontal: !vertical, color: Some(ci_col), width: 0.15, ..Default::default() };
                    l.ranges = placed.iter().filter_map(|q| q.coef.cis.get(li).map(|c| (q.at, c.0, c.1))).collect();
                    l.ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
                    if style == RangeStyle::Area || style == RangeStyle::Bar {
                        l.alpha = 0.35;
                    }
                    layers.push(l);
                }
                // citop: intervals over the estimates; otherwise the estimates on top
                let ids: Vec<usize> = (first..layers.len()).collect();
                if o.has("citop", 5) {
                    draw.extend(ids);
                } else {
                    draw.extend(ids[1..].iter().copied());
                    draw.push(first);
                }
                placed_all.extend(placed);
            }
            if let Some(a) = g.get("addplot", 7) {
                for l in addplot_layers(a, &placed_all)? {
                    draw.push(layers.len());
                    layers.push(l);
                }
            }
            // layers in drawing order; legend numbers refer to the order above
            let pos_of = |k: usize| draw.iter().position(|d| *d == k);
            let mut graph = Graph { ..Default::default() };
            graph.layers = draw.iter().map(|&k| layers[k].clone()).collect();
            let nplots = s.plots.len();
            graph.legend = legend_shown.unwrap_or(nplots > 1 || legend_order.is_some()) && !g.has("nokey", 5);
            let label_of = |k: usize| legend_labels.iter().rev().find(|(i, _)| *i == k).map(|(_, t)| t.clone());
            if let Some(order) = &legend_order {
                graph.legend_order = order.iter().filter_map(|(k, t)| pos_of(k - 1).map(|p| (p + 1, if t.is_empty() { label_of(*k).unwrap_or_default() } else { t.clone() }))).collect();
            } else if !legend_labels.is_empty() {
                for (k, t) in &legend_labels {
                    if let Some(p) = pos_of(k - 1) {
                        graph.layers[p].label = t.clone();
                    }
                }
            }
            if vertical {
                graph.xlabels = cat_labels.clone();
            } else {
                graph.ylabels = cat_labels.clone();
            }
            graph.title = bylabels.get(si).cloned().or_else(|| if subs.len() > 1 { s.plots.first().map(|p| p.label.clone()) } else { None }).unwrap_or_default();
            graphs.push(graph);
        }

        // value axis: range options, and a common scale across subgraphs
        for v in &axis_vals {
            vmin = vmin.min(*v);
            vmax = vmax.max(*v);
        }
        let lines = |name: &str| -> Vec<f64> { global.opt(name, 5).map(|s| split_comma(s).0.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default() };
        for gr in graphs.iter_mut() {
            if vmin.is_finite() && vmax.is_finite() {
                let at = if nrows > 0 { 1.0 } else { 0.0 };
                let pts = vec![if vertical { (at, vmin) } else { (vmin, at) }, if vertical { (at, vmax) } else { (vmax, at) }];
                gr.layers.push(Layer { kind: LayerKind::Scatter, points: pts, symbol: "i".into(), ..Default::default() });
            }
            gr.xlines = lines("xline");
            gr.ylines = lines("yline");
        }
        // finish() handles titles, lines, name and saving; the legend is set here
        let mut gcmd = global.clone();
        gcmd.opts.retain(|(n, _)| !(n.len() >= 3 && "legend".starts_with(n.to_lowercase().as_str())));
        gcmd.raw = cmd.raw.clone();
        let mut graph = if graphs.len() == 1 {
            graphs.pop().unwrap()
        } else {
            let by = g.get("byopts", 6).map(opts_of).transpose()?;
            let n = graphs.len();
            let cols = match by.as_ref() {
                Some(b) if b.has_opt("rows", 3) || b.has_opt("row", 3) => {
                    let r = util::opt_num(b, "rows", 3)?.unwrap_or(1.0).max(1.0) as usize;
                    n.div_ceil(r)
                }
                Some(b) if b.has_opt("cols", 3) || b.has_opt("col", 3) => util::opt_num(b, "cols", 3)?.unwrap_or(1.0).max(1.0) as usize,
                _ => (n as f64).sqrt().ceil() as usize,
            };
            Graph { children: graphs, cols: cols.max(1), ..Default::default() }
        };
        finish(&mut graph, &gcmd, st)
    }
}
