//! table (current syntax and the legacy contents() syntax) and total.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::{fmt, fmt_g};
use crate::matrix::Matrix;
use crate::restructure::take_rows;
use crate::store::Store;
use crate::summarize::{abbrev, summarize_values, Summary};
use crate::util::{self, WeightKind, Weights};
use crate::varlist;

/// One categorical dimension of a table: levels in sorted order and the
/// level index of every observation (usize::MAX when not selected).
struct Dim {
    var: String,
    label: String,
    levels: Vec<String>,
    reps: Vec<usize>,
    ids: Vec<usize>,
}

fn make_dim(ds: &DataSet, var: &str, mask: &[bool]) -> Dim {
    let (ids, reps) = util::group_ids(ds, &[var.to_string()], mask);
    let v = ds.get_var(var).unwrap();
    let levels = reps.iter().map(|&r| util::cell_text(ds, v, r, true)).collect();
    let label = if v.label.is_empty() { var.to_string() } else { v.label.clone() };
    Dim { var: var.to_string(), label, levels, reps, ids }
}

/// A statistic: canonical name and its variable (none for frequency/percent).
#[derive(Clone, Debug)]
pub(crate) struct Stat {
    pub(crate) name: String,
    pub(crate) var: Option<String>,
}

impl Stat {
    fn is_count(&self) -> bool {
        matches!(self.name.as_str(), "frequency" | "count" | "rawcount")
    }
}

pub(crate) fn canonical_stat(s: &str) -> Result<String, String> {
    let s = s.to_lowercase();
    Ok(match s.as_str() {
        "freq" | "frequency" | "fr" | "fre" | "frequ" | "frequen" => "frequency".into(),
        "percent" | "perc" | "per" => "percent".into(),
        "n" | "count" => "count".into(),
        "med" | "median" => "median".into(),
        "variance" | "var" => "variance".into(),
        "skew" | "skewness" => "skewness".into(),
        "kurt" | "kurtosis" => "kurtosis".into(),
        "sem" | "semean" => "semean".into(),
        "mean" | "sd" | "sum" | "rawsum" | "sumw" | "min" | "max" | "iqr" | "range" | "cv" | "q1" | "q2" | "q3" | "sebinomial" | "sepoisson" => s,
        p if p.starts_with('p') && p[1..].parse::<f64>().map_or(false, |x| x > 0.0 && x < 100.0) => s,
        other => return Err(rc_err(198, format!("{other} invalid statistic"))),
    })
}

/// Heading of a statistic in the current layout.
pub(crate) fn stat_title(s: &str) -> String {
    match s {
        "frequency" => "Frequency".into(),
        "percent" => "Percent".into(),
        "mean" => "Mean".into(),
        "sd" => "Standard deviation".into(),
        "semean" => "Standard error".into(),
        "count" => "Number of nonmissing values".into(),
        "median" | "q2" => "Median".into(),
        "q1" => "25th percentile".into(),
        "q3" => "75th percentile".into(),
        "min" => "Minimum value".into(),
        "max" => "Maximum value".into(),
        "sum" => "Total".into(),
        "rawsum" => "Unweighted total".into(),
        "sumw" => "Sum of weights".into(),
        "variance" => "Variance".into(),
        "cv" => "Coefficient of variation".into(),
        "skewness" => "Skewness".into(),
        "kurtosis" => "Kurtosis".into(),
        "iqr" => "Interquartile range".into(),
        "range" => "Range".into(),
        p if p.starts_with('p') => {
            let n = &p[1..];
            let suf = match n.parse::<i64>().unwrap_or(0) % 100 {
                11..=13 => "th",
                x if x % 10 == 1 => "st",
                x if x % 10 == 2 => "nd",
                x if x % 10 == 3 => "rd",
                _ => "th",
            };
            format!("{n}{suf} percentile")
        }
        other => other.to_string(),
    }
}

/// Heading of a statistic in the legacy layout, e.g. `mean(pop)`.
fn legacy_title(st: &Stat) -> String {
    let v = st.var.clone().unwrap_or_default();
    match st.name.as_str() {
        "frequency" => "Freq.".into(),
        "count" => format!("N({v})"),
        "median" => format!("med({v})"),
        "semean" => format!("sem({v})"),
        n => format!("{n}({v})"),
    }
}

/// Evaluate one statistic over the observations in `rows`.
pub(crate) fn eval_stat(ds: &DataSet, stat: &Stat, rows: &[usize], w: Option<&Weights>, grand: f64) -> f64 {
    let wt = |i: usize| w.map_or(1.0, |w| w.w[i]);
    let freq = || -> f64 {
        match w.map(|w| w.kind) {
            Some(WeightKind::F) | Some(WeightKind::I) => rows.iter().map(|&i| wt(i)).sum(),
            _ => rows.len() as f64,
        }
    };
    match stat.name.as_str() {
        "frequency" => return freq(),
        "percent" => return if grand > 0.0 { 100.0 * freq() / grand } else { SYSMISS },
        _ => {}
    }
    let Some(var) = stat.var.as_ref().and_then(|v| ds.get_var(v)) else { return SYSMISS };
    let x: Vec<f64> = rows.iter().map(|&i| var.num(i)).collect();
    let ww: Vec<f64> = rows.iter().map(|&i| wt(i)).collect();
    if stat.name == "rawsum" {
        return x.iter().filter(|v| !is_missing(**v)).sum();
    }
    let s: Summary = summarize_values(&x, w.map(|_| ww.as_slice()), w.map(|w| w.kind));
    if s.x.is_empty() && stat.name != "count" && stat.name != "sum" && stat.name != "sumw" {
        return SYSMISS;
    }
    match stat.name.as_str() {
        "mean" => s.mean,
        "count" => s.n,
        "sum" => s.sum,
        "sumw" => s.sum_w,
        "min" => s.min,
        "max" => s.max,
        "range" => s.max - s.min,
        "sd" => s.sd(),
        "variance" => s.var,
        "cv" => s.sd() / s.mean,
        "semean" => {
            if is_missing(s.var) { SYSMISS } else { s.sd() / s.n.sqrt() }
        }
        "sebinomial" => (s.mean * (1.0 - s.mean) / s.n).sqrt(),
        "sepoisson" => (s.mean / s.n).sqrt(),
        "skewness" => s.moments().0,
        "kurtosis" => s.moments().1,
        "median" | "q2" => s.pctile(50.0),
        "q1" => s.pctile(25.0),
        "q3" => s.pctile(75.0),
        "iqr" => s.pctile(75.0) - s.pctile(25.0),
        p if p.starts_with('p') => s.pctile(p[1..].parse().unwrap_or(50.0)),
        _ => SYSMISS,
    }
}

/// Observations of the selection matching `key` (None = all levels).
fn select(mask: &[bool], dims: &[&Dim], key: &[Option<usize>]) -> Vec<usize> {
    (0..mask.len()).filter(|&i| mask[i] && dims.iter().zip(key).all(|(d, k)| k.map_or(true, |l| d.ids[i] == l))).collect()
}

/// Display text of a value with an optional numeric format.
fn cell_text(v: f64, count: bool, f: Option<&str>) -> String {
    if let Some(f) = f {
        return fmt(v, f).trim().to_string();
    }
    if count && !is_missing(v) && v.fract() == 0.0 {
        return format!("{}", v as i64);
    }
    fmt_g(v, 8)
}

/// Lay out a text table: stub column plus data columns whose headings are
/// stacked parts (upper parts shared by neighbouring columns are centred
/// over them).  Rows are (stub text, cells, rule-before).
struct Grid {
    stub_head: Vec<String>,
    col_heads: Vec<Vec<String>>,
    rows: Vec<(String, Vec<String>, bool)>,
    min_width: usize,
    sep: usize,
    stub_min: usize,
}

impl Grid {
    fn render(&self) -> String {
        let nc = self.col_heads.len();
        let depth = self.col_heads.iter().map(|h| h.len()).max().unwrap_or(0);
        let stub_w = self
            .rows
            .iter()
            .map(|r| r.0.chars().count())
            .chain(self.stub_head.iter().map(|s| s.chars().count()))
            .max()
            .unwrap_or(0)
            .max(self.stub_min);
        let mut widths: Vec<usize> = (0..nc)
            .map(|j| {
                let last = self.col_heads[j].last().map_or(0, |s| s.chars().count());
                self.rows.iter().map(|r| r.1.get(j).map_or(0, |c| c.chars().count())).max().unwrap_or(0).max(last).max(self.min_width)
            })
            .collect();
        // padded heading parts: index by depth from the bottom
        let part = |j: usize, h: usize| -> String {
            let hs = &self.col_heads[j];
            let off = depth - hs.len();
            if h < off { String::new() } else { hs[h - off].clone() }
        };
        // spans for upper heading rows; widen the last column of a span if needed
        let mut spans: Vec<Vec<(usize, usize, String)>> = Vec::new();
        for h in 0..depth.saturating_sub(1) {
            let mut row = Vec::new();
            let mut j = 0;
            while j < nc {
                let t = part(j, h);
                let mut k = j + 1;
                while k < nc && part(k, h) == t && (0..h).all(|u| part(k, u) == part(j, u)) {
                    k += 1;
                }
                let span_w: usize = widths[j..k].iter().sum::<usize>() + self.sep * (k - j - 1);
                let tl = t.chars().count();
                if tl > span_w {
                    widths[k - 1] += tl - span_w;
                }
                row.push((j, k, t));
                j = k;
            }
            spans.push(row);
        }
        let body_w: usize = widths.iter().map(|w| w + self.sep).sum();
        let total_w = stub_w + 2 + body_w;
        let mut out = vec!["-".repeat(total_w)];
        let nhead = depth.max(self.stub_head.len());
        for h in 0..nhead {
            let sh = h as isize - (nhead - self.stub_head.len()) as isize;
            let stub = if sh >= 0 { self.stub_head[sh as usize].clone() } else { String::new() };
            let dh = h as isize - (nhead - depth) as isize;
            let mut line = format!("{:<stub_w$} |", stub);
            if dh >= 0 {
                let dh = dh as usize;
                if dh + 1 < depth {
                    for (a, b, t) in &spans[dh] {
                        let span_w: usize = widths[*a..*b].iter().sum::<usize>() + self.sep * (b - a - 1);
                        line.push_str(&" ".repeat(self.sep));
                        line.push_str(&format!("{:^span_w$}", t));
                    }
                } else {
                    for j in 0..nc {
                        line.push_str(&" ".repeat(self.sep));
                        line.push_str(&format!("{:>w$}", part(j, dh), w = widths[j]));
                    }
                }
            }
            out.push(line.trim_end().to_string());
        }
        let rule = format!("{}-+{}", "-".repeat(stub_w), "-".repeat(body_w));
        out.push(rule.clone());
        for (stub, cells, rule_before) in &self.rows {
            if *rule_before {
                out.push(rule.clone());
            }
            let mut line = format!("{:<stub_w$} |", stub);
            for j in 0..nc {
                line.push_str(&" ".repeat(self.sep));
                line.push_str(&format!("{:>w$}", cells.get(j).cloned().unwrap_or_default(), w = widths[j]));
            }
            out.push(line.trim_end().to_string());
        }
        out.push("-".repeat(total_w));
        out.join("\n")
    }
}

/// Split the variable specification of `table` into dimension groups:
/// `a b c`, `(a b) (c)`, `(a b)(c) ()`.
pub(crate) fn split_specs(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '(' => {
                if depth == 0 && !cur.trim().is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(c);
                }
            }
            c if c.is_whitespace() && depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Options belonging to the legacy syntax.
fn is_legacy(cmd: &Command) -> bool {
    cmd.opts.iter().any(|(n, _)| {
        let n = n.to_lowercase();
        n == "c" || (n.len() >= 4 && "contents".starts_with(&n)) || n == "by" || n == "replace" || n == "row" || n == "col" || n == "column" || n == "scol"
            || (n.len() >= 4 && "cellwidth".starts_with(&n)) || (n.len() >= 4 && "stubwidth".starts_with(&n)) || (n.len() >= 4 && "csepwidth".starts_with(&n))
            || n == "concise" || n == "center" || n == "left" || "format".starts_with(&n)
    })
}

fn opt_matches(n: &str, full: &str, min: usize) -> bool {
    let n = n.to_lowercase();
    n.len() >= min && full.starts_with(&n)
}

pub struct Table;

impl CommandHandler for Table {
    fn name(&self) -> &'static str {
        "table"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if is_legacy(cmd) {
            legacy(ds, st, cmd)
        } else {
            current(ds, st, cmd)
        }
    }
}

// ---- current syntax: see coltable.rs -------------------------------------------

fn current(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    crate::coltable::run(ds, st, cmd)
}

// ---- legacy syntax --------------------------------------------------------------

fn legacy(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let vars = varlist::expand(ds, &cmd.main)?;
    if vars.is_empty() || vars.len() > 3 {
        return Err(rc_err(198, "table requires one to three variables"));
    }
    let byv: Vec<String> = match cmd.opt("by", 2) {
        Some(b) => varlist::expand(ds, b)?,
        None => Vec::new(),
    };
    if byv.len() > 4 {
        return Err(rc_err(103, "too many by() variables"));
    }
    let contents = cmd.opts.iter().find(|(n, _)| { let n = n.to_lowercase(); n == "c" || opt_matches(&n, "contents", 4) }).and_then(|x| x.1.clone()).unwrap_or_else(|| "freq".into());
    let mut stats: Vec<Stat> = Vec::new();
    let mut toks = contents.split_whitespace().peekable();
    while let Some(t) = toks.next() {
        let name = canonical_stat(t)?;
        if name == "frequency" {
            stats.push(Stat { name, var: None });
        } else {
            let v = toks.next().ok_or_else(|| rc_err(198, format!("{t} requires a variable name")))?;
            stats.push(Stat { name, var: Some(varlist::resolve_name(ds, v)?) });
        }
    }
    if stats.len() > 5 {
        return Err(rc_err(198, "contents() may specify at most 5 statistics"));
    }
    let fmtspec = cmd.opts.iter().find(|(n, _)| opt_matches(n, "format", 1)).and_then(|x| x.1.clone()).map(|s| s.trim().to_string());
    let cellwidth = util::opt_num(cmd, "cellwidth", 4)?.map(|x| x as usize);
    let stubwidth = util::opt_num(cmd, "stubwidth", 4)?.map(|x| x as usize).unwrap_or(10);
    let csep = util::opt_num(cmd, "csepwidth", 4)?.map(|x| x as usize).unwrap_or(2);
    let row_tot = cmd.has_opt("row", 3);
    let col_tot = cmd.has_opt("col", 3) || cmd.has_opt("column", 3);
    let scol_tot = cmd.has_opt("scol", 4);
    let concise = cmd.has_opt("concise", 4);
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "faip")?;
    // dimensions: by..., row, col, supercol
    let mut all: Vec<String> = byv.clone();
    all.extend(vars.iter().cloned());
    if !cmd.has_opt("missing", 4) {
        util::markout(ds, &mut mask, &all);
    }
    let dims: Vec<Dim> = all.iter().map(|v| make_dim(ds, v, &mask)).collect();
    let dref: Vec<&Dim> = dims.iter().collect();
    let nb = byv.len();
    let ri = nb;
    let ci = if vars.len() > 1 { Some(nb + 1) } else { None };
    let si = if vars.len() > 2 { Some(nb + 2) } else { None };
    let grand_rows = select(&mask, &[], &[]);
    let grand = grand_rows.len() as f64;
    let cell_of = |key: &[Option<usize>], s: &Stat| -> Option<f64> {
        let rows = select(&mask, &dref, key);
        if rows.is_empty() { None } else { Some(eval_stat(ds, s, &rows, w.as_ref(), grand)) }
    };
    let show = |v: Option<f64>, s: &Stat| -> String {
        match v {
            None => String::new(),
            Some(v) => cell_text(v, s.is_count(), if s.name == "frequency" { None } else { fmtspec.as_deref() }),
        }
    };
    // columns: (col level, supercol level)
    let mut cols: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut heads: Vec<Vec<String>> = Vec::new();
    match (ci, si) {
        (None, _) => {
            cols.push((None, None));
        }
        (Some(c), None) => {
            for l in 0..dims[c].levels.len() {
                cols.push((Some(l), None));
                heads.push(vec![dims[c].label.clone(), dims[c].levels[l].clone()]);
            }
            if col_tot {
                cols.push((None, None));
                heads.push(vec![dims[c].label.clone(), "Total".into()]);
            }
        }
        (Some(c), Some(s)) => {
            let top = format!("{} and {}", dims[s].label, dims[c].label);
            for sl in 0..dims[s].levels.len() {
                for l in 0..dims[c].levels.len() {
                    cols.push((Some(l), Some(sl)));
                    heads.push(vec![top.clone(), dims[s].levels[sl].clone(), dims[c].levels[l].clone()]);
                }
                if col_tot {
                    cols.push((None, Some(sl)));
                    heads.push(vec![top.clone(), dims[s].levels[sl].clone(), "Total".into()]);
                }
            }
            if scol_tot {
                for l in 0..dims[c].levels.len() {
                    cols.push((Some(l), None));
                    heads.push(vec![top.clone(), "Total".into(), dims[c].levels[l].clone()]);
                }
                if col_tot {
                    cols.push((None, None));
                    heads.push(vec![top.clone(), "Total".into(), "Total".into()]);
                }
            }
        }
    }
    let one_way = ci.is_none();
    if one_way {
        heads = stats.iter().map(|s| vec![legacy_title(s)]).collect();
    }
    let key_for = |by: &[Option<usize>], r: Option<usize>, col: &(Option<usize>, Option<usize>)| -> Vec<Option<usize>> {
        let mut k: Vec<Option<usize>> = by.to_vec();
        k.push(r);
        if ci.is_some() {
            k.push(col.0);
        }
        if si.is_some() {
            k.push(col.1);
        }
        k
    };
    // by-groups (observed combinations)
    let mut by_keys: Vec<Vec<Option<usize>>> = Vec::new();
    if nb > 0 {
        let (bids, breps) = util::group_ids(ds, &byv, &mask);
        let _ = bids;
        for &rep in &breps {
            by_keys.push((0..nb).map(|d| Some(dims[d].ids[rep])).collect());
        }
    } else {
        by_keys.push(Vec::new());
    }
    let cw = cellwidth.unwrap_or_else(|| heads.iter().filter_map(|h| h.last()).map(|s| s.chars().count()).max().unwrap_or(0).max(9));
    let mut labels: Vec<String> = byv.iter().map(|v| dims[all.iter().position(|x| x == v).unwrap()].label.clone()).collect();
    labels.push(dims[ri].label.clone());
    let stub_text = match labels.len() {
        1 => labels[0].clone(),
        n => format!("{} and {}", labels[..n - 1].join(", "), labels[n - 1]),
    };
    let stub_w = stubwidth.max(dims[ri].levels.iter().map(|s| s.chars().count()).max().unwrap_or(0));
    let stub_head = wrap(&stub_text, stub_w);
    let mut grid = Grid { stub_head, col_heads: heads, rows: Vec::new(), min_width: cw, sep: csep, stub_min: stub_w };
    // replace: cells collected as (key, values)
    let mut cells_out: Vec<(Vec<Option<usize>>, Vec<f64>)> = Vec::new();
    for (bi, bk) in by_keys.iter().enumerate() {
        if nb > 0 {
            let txt: Vec<String> = bk.iter().enumerate().map(|(d, l)| dims[d].levels[l.unwrap()].clone()).collect();
            grid.rows.push((txt.join(", "), Vec::new(), bi > 0));
        }
        let mut rlevels: Vec<Option<usize>> = (0..dims[ri].levels.len()).map(Some).collect();
        if row_tot {
            rlevels.push(None);
        }
        for r in rlevels {
            let mut probe = bk.clone();
            probe.push(r);
            probe.resize(all.len(), None);
            if select(&mask, &dref, &probe).is_empty() {
                continue;
            }
            let rlabel = r.map_or("Total".to_string(), |l| dims[ri].levels[l].clone());
            let stub = format!("{:>w$}", rlabel, w = stub_w);
            if one_way {
                let key = key_for(bk, r, &cols[0]);
                let vals: Vec<Option<f64>> = stats.iter().map(|s| cell_of(&key, s)).collect();
                if concise && vals.iter().all(|v| v.map_or(true, is_missing)) {
                    continue;
                }
                cells_out.push((key, vals.iter().map(|v| v.unwrap_or(SYSMISS)).collect()));
                grid.rows.push((stub, stats.iter().zip(&vals).map(|(s, v)| show(*v, s)).collect(), false));
            } else {
                let mut lines: Vec<Vec<String>> = vec![Vec::new(); stats.len()];
                for col in &cols {
                    let key = key_for(bk, r, col);
                    let vals: Vec<Option<f64>> = stats.iter().map(|s| cell_of(&key, s)).collect();
                    if vals[0].is_some() {
                        cells_out.push((key, vals.iter().map(|v| v.unwrap_or(SYSMISS)).collect()));
                    }
                    for (k, s) in stats.iter().enumerate() {
                        lines[k].push(show(vals[k], s));
                    }
                }
                for (k, l) in lines.into_iter().enumerate() {
                    grid.rows.push((if k == 0 { stub.clone() } else { String::new() }, l, false));
                }
            }
        }
    }
    let text = grid.render();
    if cmd.has_opt("replace", 7) {
        let reps: Vec<usize> = cells_out.iter().map(|(k, _)| {
            // a representative observation of the cell's levels
            dims.iter().zip(k).find_map(|(d, l)| l.map(|l| d.reps[l])).unwrap_or(0)
        }).collect();
        let mut nds = take_rows(ds, &reps);
        nds.keep_variables(&all);
        // level variables: copy each level's value, missing for totals
        for (d, dim) in dims.iter().enumerate() {
            let src = ds.get_var(&dim.var).unwrap().clone();
            let var = nds.get_var_mut(&dim.var).unwrap();
            for (o, (k, _)) in cells_out.iter().enumerate() {
                match k[d] {
                    Some(l) => var.data[o] = src.data[dim.reps[l]].clone(),
                    None => {
                        var.data[o] = if src.var_type.is_string() { ferrostat_core::Value::Str(String::new()) } else { util::typed_value(src.var_type, SYSMISS) };
                    }
                }
            }
        }
        for (k, s) in stats.iter().enumerate() {
            let mut v = util::new_double(&format!("table{}", k + 1), cells_out.iter().map(|c| c.1[k]).collect());
            v.label = legacy_title(s);
            nds.add_variable(v)?;
        }
        nds.n_obs = cells_out.len();
        *ds = nds;
        st.data_changed = true;
    }
    Ok(text)
}

/// Wrap words into lines of at most `w` characters.
fn wrap(s: &str, w: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > w {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

// ---- total ------------------------------------------------------------------------

pub struct Total;

impl CommandHandler for Total {
    fn name(&self) -> &'static str {
        "total"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fpi")?;
        util::markout(ds, &mut mask, &vars);
        let over: Vec<String> = match cmd.opt("over", 2) {
            Some(o) => varlist::expand(ds, o.split(',').next().unwrap_or(""))?,
            None => Vec::new(),
        };
        util::markout(ds, &mut mask, &over);
        let level = util::level(cmd, st)?;
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let kind = w.as_ref().map(|w| w.kind);
        let ww: Vec<f64> = rows.iter().map(|&i| w.as_ref().map_or(1.0, |w| w.w[i])).collect();
        // over groups
        let (gids, greps) = if over.is_empty() { (vec![0; ds.n_obs], vec![rows[0]]) } else { util::group_ids(ds, &over, &mask) };
        let ng = greps.len();
        let glabel = |g: usize| -> String {
            over.iter().map(|v| util::cell_text(ds, ds.get_var(v).unwrap(), greps[g], true)).collect::<Vec<_>>().join(", ")
        };
        let gname = |g: usize| -> String {
            over.iter()
                .map(|v| {
                    let x = ds.get_var(v).unwrap().num(greps[g]);
                    format!("{}.{v}", fmt_g(x, 9))
                })
                .collect::<Vec<_>>()
                .join("#")
        };
        // z[j][t]: contribution of observation t to the j-th total (unweighted value)
        let k = vars.len() * ng;
        let mut names = Vec::new();
        let mut z: Vec<Vec<f64>> = Vec::new();
        for v in &vars {
            let col = ds.get_var(v).unwrap();
            for g in 0..ng {
                names.push(if over.is_empty() { v.clone() } else { format!("c.{v}@{}", gname(g)) });
                z.push(rows.iter().map(|&i| if gids[i] == g { col.num(i) } else { 0.0 }).collect());
            }
        }
        let n_obs = rows.len() as f64;
        let n = match kind {
            Some(WeightKind::F) => ww.iter().sum(),
            _ => n_obs,
        };
        let b: Vec<f64> = z.iter().map(|zj| zj.iter().zip(&ww).map(|(x, w)| x * w).sum()).collect();
        let mut vm = Matrix::zeros(k, k);
        for a in 0..k {
            for c in 0..k {
                vm[(a, c)] = match kind {
                    Some(WeightKind::F) | Some(WeightKind::I) => {
                        // replicate each observation w times
                        let sw: f64 = ww.iter().sum();
                        let ma = b[a] / sw;
                        let mc = b[c] / sw;
                        let s: f64 = (0..rows.len()).map(|t| ww[t] * (z[a][t] - ma) * (z[c][t] - mc)).sum();
                        sw / (sw - 1.0) * s
                    }
                    _ => {
                        let ma = b[a] / n_obs;
                        let mc = b[c] / n_obs;
                        let s: f64 = (0..rows.len()).map(|t| (ww[t] * z[a][t] - ma) * (ww[t] * z[c][t] - mc)).sum();
                        n_obs / (n_obs - 1.0) * s
                    }
                };
            }
        }
        let df = n - 1.0;
        let tcrit = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
        let mut out = vec![format!("Total estimation{:>39}Number of obs = {:>7}", "", fmt_g(n, 9)), String::new()];
        let rule = "--------------------------------------------------------------".to_string();
        out.push(rule.clone());
        out.push(format!("             |      Total   Std. err.     [{}% conf. interval]", fmt_g(level, 4)));
        out.push("-------------+------------------------------------------------".to_string());
        for (vi, v) in vars.iter().enumerate() {
            if !over.is_empty() {
                out.push(format!("{:<12} |", abbrev(&format!("c.{v}@{}", over.join("#")), 12)));
            }
            for g in 0..ng {
                let j = vi * ng + g;
                let se = vm[(j, j)].sqrt();
                let lab = if over.is_empty() { abbrev(v, 12) } else { abbrev(&glabel(g), 12) };
                out.push(format!("{:>12} | {:>10} {:>10}  {:>10}  {:>10}", lab, fmt_g(b[j], 8), fmt_g(se, 8), fmt_g(b[j] - tcrit * se, 8), fmt_g(b[j] + tcrit * se, 8)));
            }
        }
        out.push(rule);
        let e = st.begin_e();
        e.set("N", n);
        e.set("df_r", df);
        e.set("k_eq", 1.0);
        e.set("level", level);
        e.set_str("cmd", "total");
        e.set_str("title", "Total estimation");
        e.set_str("estat_cmd", "estat_vce_only");
        e.set_str("vce", "analytic");
        e.set_str("varlist", vars.join(" "));
        if let Some(w) = &w {
            e.set_str("wtype", match w.kind { WeightKind::F => "fweight", WeightKind::P => "pweight", WeightKind::I => "iweight", WeightKind::A => "aweight" });
        }
        if !over.is_empty() {
            e.set_str("over", over.join(" "));
            e.set_str("over_namelist", (0..ng).map(|g| gname(g).split('.').next().unwrap_or("").to_string()).collect::<Vec<_>>().join(" "));
            e.set_str("over_labels", (0..ng).map(|g| format!("\"{}\"", glabel(g))).collect::<Vec<_>>().join(" "));
            e.set("N_over", ng as f64);
            let counts: Vec<f64> = (0..ng).map(|g| (0..rows.len()).filter(|&t| gids[rows[t]] == g).map(|t| if kind == Some(WeightKind::F) { ww[t] } else { 1.0 }).sum()).collect();
            e.set_mat("_N", Matrix::row_vector(&counts.iter().cycle().take(k).cloned().collect::<Vec<_>>()).with_names(vec!["r1".into()], names.clone()));
        } else {
            e.set_mat("_N", Matrix::row_vector(&vec![n; k]).with_names(vec!["r1".into()], names.clone()));
        }
        e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", vm.with_names(names.clone(), names));
        e.sample = Some(mask);
        Ok(out.join("\n"))
    }
}
