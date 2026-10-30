//! Displays, tables, graphs and stored results of `power` and `ciwidth`.

use ferrostat_parser::Command;

use super::{g, has, Def, Row, SolveFor};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{finish, palette, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Solve {
    Power,
    N,
    Effect,
    Width,
    ProbWidth,
}

/// Everything needed to display one analysis.
pub struct Report {
    /// "power" or "ciwidth"
    pub kind: String,
    pub method: String,
    pub title: String,
    pub test: String,
    pub hyp: String,
    pub solve: Solve,
    pub target: Option<String>,
    pub tword: String,
    pub two: bool,
    pub cluster: bool,
    /// method parameters in display order
    pub show: Vec<String>,
    /// extra result columns (events, ...)
    pub extra: Vec<String>,
    pub rows: Vec<Row>,
}

impl Report {
    pub(crate) fn from_def(def: &Def, s: SolveFor, rows: Vec<Row>, cluster: bool) -> Report {
        let mut extra = Vec::new();
        if let Some(r) = rows.first() {
            for k in ["E", "Pr_E", "E1", "E2"] {
                if r.contains_key(k) {
                    extra.push(k.to_string());
                }
            }
        }
        Report {
            kind: "power".into(),
            method: def.method.clone(),
            title: def.title.clone(),
            test: def.test.clone(),
            hyp: def.hyp.clone(),
            solve: match s {
                SolveFor::Power => Solve::Power,
                SolveFor::N => Solve::N,
                SolveFor::Effect => Solve::Effect,
            },
            target: def.target.map(String::from),
            tword: def.tword.into(),
            two: def.two,
            cluster,
            show: def.show.clone(),
            extra,
            rows,
        }
    }

    fn size_cols(&self) -> Vec<&'static str> {
        let mut v = vec!["N"];
        if self.two {
            v.extend(["N1", "N2", "nratio"]);
        }
        if self.cluster {
            if self.two {
                v.extend(["K1", "K2", "M1", "M2", "rho"]);
            } else {
                v.extend(["K", "M", "rho"]);
            }
        }
        v
    }

    /// All columns, in order, that hold a value in some row.
    fn columns(&self) -> Vec<String> {
        let mut c: Vec<String> = vec!["alpha".into()];
        if self.kind == "power" {
            c.push("power".into());
        } else {
            c.push("Pr_width".into());
        }
        c.extend(self.size_cols().iter().map(|s| s.to_string()));
        if self.kind == "power" {
            c.push("delta".into());
        } else {
            c.push("width".into());
        }
        for s in self.show.iter().chain(self.extra.iter()) {
            if !c.contains(s) {
                c.push(s.clone());
            }
        }
        c.retain(|k| self.rows.iter().any(|r| has(r, k)));
        c
    }

    /// Keys reported as the solution (not as study parameters).
    fn solved_keys(&self) -> Vec<String> {
        match self.solve {
            Solve::Power => vec![if self.kind == "power" { "power".into() } else { "Pr_width".into() }],
            Solve::ProbWidth => vec!["Pr_width".into()],
            Solve::Width => vec!["width".into()],
            Solve::N => {
                let r = &self.rows[0];
                if self.cluster {
                    let k = if self.two { ["K1", "K2"] } else { ["K", "K"] };
                    let m = if self.two { ["M1", "M2"] } else { ["M", "M"] };
                    // what was solved: the one missing from the inputs
                    let solved_k = r.get("_solvedK").is_some_and(|v| *v == 1.0);
                    let mut v: Vec<String> = if solved_k { k.iter().map(|s| s.to_string()).collect() } else { m.iter().map(|s| s.to_string()).collect() };
                    v.dedup();
                    v.push("N".into());
                    if self.two {
                        v.extend(["N1".into(), "N2".into()]);
                    }
                    v
                } else if self.two {
                    vec!["N".into(), "N1".into(), "N2".into()]
                } else {
                    vec!["N".into()]
                }
            }
            Solve::Effect => {
                let mut v = vec![];
                if let Some(t) = &self.target {
                    v.push(t.clone());
                }
                v.push("delta".into());
                v
            }
        }
    }

    fn heading(&self) -> String {
        let what = match self.solve {
            Solve::Power => "power".to_string(),
            Solve::N => {
                if self.cluster {
                    if self.rows[0].get("_solvedK").is_some_and(|v| *v == 1.0) {
                        "numbers of clusters".into()
                    } else {
                        "cluster sizes".into()
                    }
                } else if self.two {
                    "sample sizes".into()
                } else {
                    "sample size".into()
                }
            }
            Solve::Effect => format!("target {}", self.tword),
            Solve::Width => "CI width".into(),
            Solve::ProbWidth => "probability of CI width".into(),
        };
        let what = if self.cluster && self.solve == Solve::N && !self.two {
            what.replace("numbers", "number").replace("sizes", "size")
        } else {
            what
        };
        format!("Estimated {what} for {}", self.title)
    }
}

fn label(k: &str) -> String {
    match k {
        "N" => "N".into(),
        "nratio" => "N2/N1".into(),
        "Pr_width" => "Pr_width".into(),
        "power" => "power".into(),
        other => other.into(),
    }
}

fn is_int_key(k: &str) -> bool {
    matches!(k, "N" | "N1" | "N2" | "K" | "K1" | "K2" | "M" | "M1" | "M2" | "E" | "E1" | "E2" | "ngroups" | "ncontrol" | "ntested" | "nlevels" | "nrows" | "ncols")
}

fn num(v: f64, k: &str) -> String {
    if v.is_nan() {
        return ".".into();
    }
    if (is_int_key(k) || v.fract() == 0.0) && v.abs() < 1e12 && v.fract() == 0.0 {
        return format!("{}", v as i64);
    }
    if is_int_key(k) {
        return format!("{:.2}", v);
    }
    let s = format!("{:.4}", v);
    if v.abs() >= 1e6 || (v != 0.0 && v.abs() < 1e-4) {
        fmt_g(v, 9)
    } else {
        s
    }
}

/// Render the analysis, store r() results and draw the graph if asked.
pub fn report(st: &mut Store, cmd: &Command, rep: Report) -> Result<String, String> {
    if rep.rows.is_empty() {
        return Err(rc_err(198, "nothing to compute"));
    }
    let cols = rep.columns();
    let graph_opt = cmd.opts.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= 2 && "graph".starts_with(&n)
    });
    let table_opt = cmd.opts.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= 3 && "table".starts_with(&n)
    });
    let notable = cmd.has_opt("notable", 5);
    let multi = rep.rows.len() > 1;
    let mut out: Vec<String> = Vec::new();
    let show_table = !notable && (table_opt.is_some() || (multi && graph_opt.is_none()));
    if show_table {
        out.extend(table(&rep, &cols, table_opt.and_then(|t| t.1.as_deref()))?);
    } else if !multi && graph_opt.is_none() && !notable {
        out.extend(detail(&rep));
    }
    // stored results
    let mut r = Results::new();
    let last = rep.rows.last().unwrap();
    for (k, v) in last {
        if k.starts_with('_') || v.is_nan() {
            continue;
        }
        r.set(k, *v);
    }
    if !last.contains_key("beta") && last.contains_key("power") {
        r.set("beta", 1.0 - g(last, "power"));
    }
    r.set("onesided", if super::flag(last, "_onesided") { 1.0 } else { 0.0 });
    if rep.kind == "power" {
        r.set("nfractional", if super::flag(last, "_nfrac") { 1.0 } else { 0.0 });
    }
    if multi || table_opt.is_some() {
        let mut m = Matrix::zeros(rep.rows.len(), cols.len());
        for (i, row) in rep.rows.iter().enumerate() {
            for (j, c) in cols.iter().enumerate() {
                let v = g(row, c);
                m[(i, j)] = if v.is_nan() { ferrostat_core::missing::SYSMISS } else { v };
            }
        }
        m.colnames = cols.clone();
        m.rownames = (1..=rep.rows.len()).map(|i| format!("r{i}")).collect();
        r.set_mat("pss_table", m);
        r.set("separator", 0.0);
    }
    r.set_str("type", if rep.kind == "power" { "test" } else { "ci" });
    r.set_str("method", rep.method.clone());
    r.set_str("columns", cols.join(" "));
    r.set_str(
        "solvefor",
        match rep.solve {
            Solve::Power => "power",
            Solve::N => {
                if rep.cluster {
                    if last.get("_solvedK").is_some_and(|v| *v == 1.0) {
                        "K"
                    } else {
                        "M"
                    }
                } else {
                    "N"
                }
            }
            Solve::Effect => "esize",
            Solve::Width => "width",
            Solve::ProbWidth => "probwidth",
        },
    );
    if !rep.test.is_empty() {
        r.set_str("test", rep.test.clone());
    }
    if rep.cluster {
        r.set_str("design", "CRD");
    }
    st.r = r;
    if let Some((_, gargs)) = graph_opt {
        let msg = draw(st, cmd, &rep, gargs.as_deref().unwrap_or(""))?;
        if !msg.is_empty() && out.is_empty() {
            return Ok(msg);
        }
    }
    Ok(out.join("\n"))
}

fn detail(rep: &Report) -> Vec<String> {
    let r = &rep.rows[0];
    let mut out = vec![String::new(), rep.heading()];
    if !rep.test.is_empty() {
        out.push(rep.test.clone());
    }
    if !rep.hyp.is_empty() {
        out.push(rep.hyp.clone());
    }
    out.push(String::new());
    out.push("Study parameters:".into());
    out.push(String::new());
    let solved = rep.solved_keys();
    let line = |k: &str, v: f64| format!("{:>13} = {:>10}", label(k), num(v, k));
    let mut keys: Vec<String> = vec!["alpha".into()];
    if rep.kind == "power" {
        keys.push("power".into());
    } else {
        keys.push("Pr_width".into());
    }
    keys.extend(rep.size_cols().iter().map(|s| s.to_string()));
    keys.push(if rep.kind == "power" { "delta".into() } else { "width".into() });
    keys.extend(rep.show.iter().cloned());
    let mut seen = Vec::new();
    for k in &keys {
        if seen.contains(k) || solved.contains(k) || !has(r, k) {
            continue;
        }
        if rep.kind == "ciwidth" && k == "Pr_width" && !r.contains_key("_probgiven") {
            continue;
        }
        seen.push(k.clone());
        let v = if k == "power" && has(r, "_ptarget") { g(r, "_ptarget") } else { g(r, k) };
        out.push(line(k, v));
    }
    out.push(String::new());
    let head = rep.heading();
    let head = head.split(" for ").next().unwrap_or("").to_string();
    out.push(format!("{head}:"));
    out.push(String::new());
    for k in &solved {
        if has(r, k) {
            out.push(line(k, g(r, k)));
        }
    }
    for k in &rep.extra {
        if has(r, k) && !solved.contains(k) {
            out.push(line(k, g(r, k)));
        }
    }
    out
}

fn table(rep: &Report, cols: &[String], spec: Option<&str>) -> Result<Vec<String>, String> {
    let mut use_cols: Vec<String> = cols.to_vec();
    let mut labels: Vec<String> = use_cols.iter().map(|c| label(c)).collect();
    if let Some(s) = spec {
        let body = s.split(',').next().unwrap_or("").trim();
        if !body.is_empty() && !body.starts_with("_") {
            use_cols.clear();
            labels.clear();
            // col[:"label"] ...
            let mut rest = body;
            while !rest.trim().is_empty() {
                let t = rest.trim_start();
                let end = t.find(|c: char| c.is_whitespace() || c == ':').unwrap_or(t.len());
                let name = &t[..end];
                let mut after = &t[end..];
                let mut lab = name.to_string();
                if let Some(a) = after.strip_prefix(':') {
                    let a = a.trim_start();
                    if let Some(q) = a.strip_prefix('"') {
                        let close = q.find('"').unwrap_or(q.len());
                        lab = q[..close].to_string();
                        after = &q[(close + 1).min(q.len())..];
                    }
                }
                let key = cols.iter().find(|c| c.eq_ignore_ascii_case(name) || label(c).eq_ignore_ascii_case(name)).cloned();
                match key {
                    Some(k) => {
                        use_cols.push(k);
                        labels.push(lab);
                    }
                    None => {
                        if !name.starts_with('_') {
                            return Err(rc_err(198, format!("table(): column {name} not found")));
                        }
                    }
                }
                rest = after;
            }
        }
    }
    let w = 10usize;
    let mut out = vec![String::new(), rep.heading()];
    if !rep.test.is_empty() {
        out.push(rep.test.clone());
    }
    if !rep.hyp.is_empty() {
        out.push(rep.hyp.clone());
    }
    out.push(String::new());
    let width = use_cols.len() * (w + 1) + 1;
    out.push("-".repeat(width));
    out.push(labels.iter().map(|l| format!("{:>w$}", l)).collect::<Vec<_>>().join(" "));
    out.push("-".repeat(width));
    for r in &rep.rows {
        out.push(use_cols.iter().map(|c| format!("{:>w$}", num(g(r, c), c))).collect::<Vec<_>>().join(" "));
    }
    out.push("-".repeat(width));
    Ok(out)
}

/// The y quantity of a graph: what was solved for.
fn y_key(rep: &Report) -> String {
    match rep.solve {
        Solve::Power => if rep.kind == "power" { "power" } else { "Pr_width" }.into(),
        Solve::N => {
            if rep.cluster {
                if rep.rows[0].get("_solvedK").is_some_and(|v| *v == 1.0) {
                    if rep.two { "K1" } else { "K" }.into()
                } else {
                    if rep.two { "M1" } else { "M" }.into()
                }
            } else {
                "N".into()
            }
        }
        Solve::Effect => rep.target.clone().unwrap_or_else(|| "delta".into()),
        Solve::Width => "width".into(),
        Solve::ProbWidth => "Pr_width".into(),
    }
}

fn axis_title(k: &str) -> String {
    match k {
        "power" => "Power (1-{&beta})".replace("{&beta}", "β"),
        "N" => "Sample size (N)".into(),
        "alpha" => "Significance level (α)".into(),
        "delta" => "Effect size (δ)".into(),
        "width" => "CI width".into(),
        "Pr_width" => "Probability of CI width".into(),
        other => other.into(),
    }
}

fn draw(st: &mut Store, outer: &Command, rep: &Report, gargs: &str) -> Result<String, String> {
    let gcmd = ferrostat_parser::parser::parse_line(&format!("graph_pss , {gargs}")).map_err(|e| rc_err(198, e.message))?;
    let y = gcmd.opt("ydimension", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| y_key(rep));
    let y = rep.rows[0].keys().find(|k| k.eq_ignore_ascii_case(&y)).cloned().ok_or_else(|| rc_err(198, format!("graph: {y} not found")))?;
    // varying inputs (besides y)
    let mut varying: Vec<String> = Vec::new();
    let mut cand: Vec<String> = vec!["N".into(), "N1".into(), "N2".into(), "K".into(), "M".into(), "K1".into(), "M1".into()];
    if let Some(t) = &rep.target {
        cand.push(t.clone());
    }
    cand.extend(["alpha".into(), "power".into(), "width".into(), "Pr_width".into(), "nratio".into(), "rho".into()]);
    cand.extend(rep.show.iter().cloned());
    for k in cand {
        if k == y || varying.contains(&k) || matches!(y.as_str(), "N" | "K" | "M" | "K1" | "M1") && matches!(k.as_str(), "N" | "N1" | "N2" | "K" | "K1" | "K2" | "M" | "M1" | "M2") {
            continue;
        }
        if rep.solve == Solve::Power && k == "power" || rep.solve == Solve::Effect && k == "delta" {
            continue;
        }
        let mut vals: Vec<f64> = rep.rows.iter().map(|r| g(r, &k)).filter(|v| !v.is_nan()).collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        if vals.len() > 1 {
            varying.push(k);
        }
    }
    let x = match gcmd.opt("xdimension", 2).map(|s| s.trim().to_string()) {
        Some(x) => rep.rows[0].keys().find(|k| k.eq_ignore_ascii_case(&x)).cloned().ok_or_else(|| rc_err(198, format!("graph: {x} not found")))?,
        None => varying.first().cloned().unwrap_or_else(|| if y == "power" { "N".into() } else { "power".into() }),
    };
    let by: Vec<String> = varying.iter().filter(|k| **k != x).cloned().collect();
    let mut series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    for r in &rep.rows {
        let key = by.iter().map(|k| format!("{}={}", label(k), num(g(r, k), k))).collect::<Vec<_>>().join(", ");
        let p = (g(r, &x), g(r, &y));
        match series.iter_mut().find(|s| s.0 == key) {
            Some(s) => s.1.push(p),
            None => series.push((key, vec![p])),
        }
    }
    let pal = palette(&st.scheme.clone());
    let mut graph = Graph::default();
    for (i, (key, mut pts)) in series.into_iter().enumerate() {
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let c = pal[i % pal.len()];
        graph.layers.push(Layer { kind: LayerKind::Connected, points: pts, label: key, color: Some((c.0, c.1, c.2)), ..Default::default() });
    }
    graph.legend = graph.layers.len() > 1;
    graph.title = rep.heading();
    graph.ytitle = axis_title(&y);
    graph.xtitle = axis_title(&x);
    graph.command = outer.raw.clone();
    finish(&mut graph, &gcmd, st)
}
