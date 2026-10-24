//! `outreg2`-compatible regression tables (Ferrostat's own implementation):
//! each call adds one column per model to a tab-delimited table written to
//! the `using` file (`.xls` files are tab-delimited text, as Excel opens them).

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::{rc_err, Env};
use ferrostat_cmd::store::Results;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;

#[derive(Debug, Clone, Default)]
pub struct Table {
    pub title: String,
    /// row labels (first column)
    pub rows: Vec<String>,
    /// one column per model: header + cells by row label
    pub cols: Vec<(String, std::collections::HashMap<String, String>)>,
}

impl Table {
    fn to_tsv(&self) -> String {
        let mut out = String::new();
        if !self.title.is_empty() {
            out.push_str(&self.title);
            out.push('\n');
        }
        let k = self.cols.len();
        out.push('\t');
        out.push_str(&(1..=k).map(|i| format!("({i})")).collect::<Vec<_>>().join("\t"));
        out.push('\n');
        out.push_str("VARIABLES\t");
        out.push_str(&self.cols.iter().map(|c| c.0.clone()).collect::<Vec<_>>().join("\t"));
        out.push('\n');
        for r in &self.rows {
            let label = if r.starts_with('\u{1}') { "" } else { r.as_str() };
            out.push_str(label);
            for c in &self.cols {
                out.push('\t');
                out.push_str(c.1.get(r).map(String::as_str).unwrap_or(""));
            }
            out.push('\n');
        }
        out.push_str("Standard errors in parentheses\n*** p<0.01, ** p<0.05, * p<0.1\n");
        out
    }

    fn from_tsv(text: &str) -> Table {
        let mut t = Table::default();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        if let Some(l) = lines.first() {
            if !l.starts_with('\t') {
                t.title = l.to_string();
                i = 1;
            }
        }
        i += 1; // (1) (2) ...
        let headers: Vec<String> = lines.get(i).map(|l| l.split('\t').skip(1).map(String::from).collect()).unwrap_or_default();
        i += 1;
        for h in &headers {
            t.cols.push((h.clone(), Default::default()));
        }
        let mut last = String::new();
        let mut blank = 0;
        for l in &lines[i.min(lines.len())..] {
            if l.starts_with("Standard errors") || l.starts_with("***") {
                break;
            }
            let f: Vec<&str> = l.split('\t').collect();
            let key = if f[0].is_empty() {
                blank += 1;
                format!("\u{1}{last}\u{1}{blank}")
            } else {
                last = f[0].to_string();
                blank = 0;
                f[0].to_string()
            };
            t.rows.push(key.clone());
            for (c, v) in f.iter().skip(1).enumerate() {
                if let Some(col) = t.cols.get_mut(c) {
                    col.1.insert(key.clone(), v.to_string());
                }
            }
        }
        t
    }
}

fn stars(p: f64) -> &'static str {
    if p < 0.01 {
        "***"
    } else if p < 0.05 {
        "**"
    } else if p < 0.1 {
        "*"
    } else {
        ""
    }
}

fn fmt3(v: f64) -> String {
    if v.abs() >= 1e-3 || v == 0.0 { format!("{:.3}", v) } else { ferrostat_cmd::format::fmt_g(v, 6) }
}

/// Build one column from a set of e() results.
fn column(e: &Results, stats: &[String], addstat: &[(String, f64)]) -> Vec<(String, String)> {
    let mut cells = Vec::new();
    let (Some(b), Some(v)) = (e.matrices.get("b"), e.matrices.get("V")) else { return cells };
    let df = e.get("df_r");
    for (j, name) in b.colnames.iter().enumerate() {
        let n = name.rsplit(':').next().unwrap_or(name);
        if n.contains("b.") || n.starts_with("o.") || n.contains("o.") && n.split('.').next().map_or(false, |x| x.ends_with('o')) {
            continue;
        }
        let bj = b.get(0, j);
        let se = v.get(j, j).max(0.0).sqrt();
        if se == 0.0 && bj == 0.0 {
            continue;
        }
        let t = bj / se;
        let p = match df {
            Some(d) => 2.0 * dist::ttail(d, t.abs()),
            None => 2.0 * (1.0 - dist::normal(t.abs())),
        };
        let label = display_label(n);
        let mut k = 0;
        for s in stats {
            let key = if k == 0 { label.clone() } else { format!("\u{1}{label}\u{1}{k}") };
            let cell = match s.as_str() {
                "coef" => format!("{}{}", fmt3(bj), stars(p)),
                "se" => format!("({})", fmt3(se)),
                "pval" => fmt3(p),
                "tstat" => fmt3(t),
                "ci" => format!("[{}, {}]", fmt3(bj - 1.96 * se), fmt3(bj + 1.96 * se)),
                _ => continue,
            };
            cells.push((key, cell));
            k += 1;
        }
    }
    if let Some(n) = e.get("N") {
        cells.push(("Observations".into(), format!("{}", n as i64)));
    }
    if let Some(r2) = e.get("r2").or_else(|| e.get("r2_w")) {
        cells.push(("R-squared".into(), fmt3(r2)));
    }
    if let Some(g) = e.get("N_g") {
        cells.push(("Number of groups".into(), format!("{}", g as i64)));
    }
    for (name, val) in addstat {
        cells.push((name.clone(), fmt3(*val)));
    }
    cells
}

/// Stata-like display label: `L.x` -> `L.x`, `2.g` -> `2.g`.
fn display_label(n: &str) -> String {
    n.to_string()
}

pub fn outreg2(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    };
    let opt = |n: &str| opts.iter().find(|(o, _)| o.to_lowercase().starts_with(n)).map(|(_, a)| a.clone().unwrap_or_default());
    let has = |n: &str| opts.iter().any(|(o, _)| o.to_lowercase() == n);
    // [model list] using file
    let (models, rest) = if main.starts_with('[') {
        let close = main.find(']').ok_or_else(|| rc_err(198, "] expected"))?;
        (main[1..close].split_whitespace().map(String::from).collect::<Vec<_>>(), main[close + 1..].trim().to_string())
    } else {
        (Vec::new(), main.clone())
    };
    let path = rest.trim().strip_prefix("using").map(|p| unquote(p.trim())).ok_or_else(|| rc_err(100, "using required"))?;
    let path = if std::path::Path::new(&path).extension().is_some() { path } else { format!("{path}.txt") };
    let stats: Vec<String> = opt("stats").map(|s| s.split_whitespace().map(|x| x.to_lowercase()).collect()).unwrap_or_else(|| vec!["coef".into(), "se".into()]);
    let mut addstat = Vec::new();
    if let Some(a) = opt("addstat") {
        let parts: Vec<String> = a.split(',').map(|x| x.trim().to_string()).collect();
        for pair in parts.chunks(2) {
            if pair.len() == 2 {
                let e = parse_expression(&pair[1]).map_err(|e| rc_err(198, e.message))?;
                let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?.num()?;
                addstat.push((unquote(&pair[0]), v));
            }
        }
    }
    let ctitles: Vec<String> = opt("ctitle").map(|c| split_top_level_words(&c).iter().map(|x| unquote(x)).collect()).unwrap_or_default();
    let replace = has("replace") || !std::path::Path::new(&path).exists() && !has("append");
    let mut table = if replace {
        Table::default()
    } else {
        me.outreg_tables.get(&path).cloned().or_else(|| std::fs::read_to_string(&path).ok().map(|t| Table::from_tsv(&t))).unwrap_or_default()
    };
    if let Some(t) = opt("title") {
        table.title = unquote(&t);
    }
    let results: Vec<(String, Results)> = if models.is_empty() {
        vec![(me.engine.store.e.macros.get("depvar").cloned().unwrap_or_default(), me.engine.store.e.clone())]
    } else {
        models
            .iter()
            .map(|m| me.engine.store.estimates.get(m).cloned().map(|e| (m.clone(), e)).ok_or_else(|| rc_err(111, format!("estimation result {m} not found"))))
            .collect::<Result<_, _>>()?
    };
    if results.iter().any(|(_, e)| !e.matrices.contains_key("b")) {
        return Err(rc_err(301, "last estimates not found"));
    }
    for (k, (name, e)) in results.iter().enumerate() {
        let header = ctitles.get(k).cloned().unwrap_or_else(|| if models.is_empty() { name.clone() } else { name.clone() });
        let cells = column(e, &stats, &addstat);
        let mut map = std::collections::HashMap::new();
        for (key, v) in cells {
            if !table.rows.contains(&key) {
                // keep statistics rows at the bottom
                let pos = table.rows.iter().position(|r| r == "Observations").filter(|_| key != "Observations" && !matches!(key.as_str(), "R-squared" | "Number of groups") && !addstat.iter().any(|(n, _)| *n == key));
                match pos {
                    Some(p) => {
                        table.rows.insert(p, key.clone());
                    }
                    None => table.rows.push(key.clone()),
                }
            }
            map.insert(key, v);
        }
        table.cols.push((header, map));
    }
    std::fs::write(&path, table.to_tsv()).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    me.outreg_tables.insert(path.clone(), table);
    Ok(format!("{path}\ndir : seeout"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsv_roundtrip() {
        let mut t = Table { title: "T".into(), ..Default::default() };
        t.rows = vec!["x".into(), "\u{1}x\u{1}1".into(), "Observations".into()];
        let mut m = std::collections::HashMap::new();
        m.insert("x".to_string(), "1.000***".to_string());
        m.insert("\u{1}x\u{1}1".to_string(), "(0.100)".to_string());
        m.insert("Observations".to_string(), "10".to_string());
        t.cols.push(("OLS".into(), m));
        let back = Table::from_tsv(&t.to_tsv());
        assert_eq!(back.title, "T");
        assert_eq!(back.cols[0].0, "OLS");
        assert_eq!(back.cols[0].1.get("Observations").map(String::as_str), Some("10"));
        assert_eq!(back.rows.len(), 3);
    }
}
