//! Graphics: twoway (scatter, line, connected, lfit, area, bar), histogram,
//! graph bar/hbar/box, kdensity plots. Commands build the graph in memory;
//! `graph export` renders it to PNG or SVG.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_line, split_top_level_words};
use ferrostat_parser::Command;
use plotters::prelude::*;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Scatter,
    Line,
    Connected,
    Area,
    Bar,
    /// box plot: (q1, median, q3, lower whisker, upper whisker)
    Box,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub kind: LayerKind,
    pub points: Vec<(f64, f64)>,
    pub label: String,
    /// for bars: bar width; for box: five-number summaries
    pub width: f64,
    pub boxes: Vec<[f64; 5]>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub title: String,
    pub xtitle: String,
    pub ytitle: String,
    pub layers: Vec<Layer>,
    pub xlabels: Vec<(f64, String)>,
    pub legend: bool,
    pub name: String,
}

fn opt_text(cmd: &Command, name: &str, min: usize) -> Option<String> {
    cmd.opt(name, min).map(|s| util::unquote(s.split(',').next().unwrap_or(s)))
}

fn finish(graph: &mut Graph, cmd: &Command, st: &mut Store) -> Result<String, String> {
    if let Some(t) = opt_text(cmd, "title", 1) {
        graph.title = t;
    }
    if let Some(t) = opt_text(cmd, "xtitle", 2) {
        graph.xtitle = t;
    }
    if let Some(t) = opt_text(cmd, "ytitle", 2) {
        graph.ytitle = t;
    }
    if let Some(l) = cmd.opt("legend", 3) {
        graph.legend = !l.contains("off");
    }
    graph.name = cmd.opt("name", 4).map(|s| s.split(',').next().unwrap().trim().to_string()).unwrap_or_else(|| "Graph".into());
    let mut msg = String::new();
    if let Some(s) = cmd.opt("saving", 3) {
        let file = util::unquote(s.split(',').next().unwrap());
        render(graph, &file, 800, 600)?;
        msg = format!("file {file} saved");
    }
    st.graphs.insert(graph.name.clone(), graph.clone());
    st.graph = Some(graph.clone());
    Ok(msg)
}

fn xy_points(ds: &DataSet, mask: &[bool], y: &str, x: &str) -> Result<Vec<(f64, f64)>, String> {
    let yc = util::col(ds, y)?;
    let xc = util::col(ds, x)?;
    Ok((0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(xc[i])).map(|i| (xc[i], yc[i])).collect())
}

fn layer(kind: LayerKind, points: Vec<(f64, f64)>, label: &str) -> Layer {
    Layer { kind, points, label: label.to_string(), width: 0.0, boxes: Vec::new() }
}

/// Build the layers for one plot specification such as `scatter y x`.
fn plot_layers(ds: &DataSet, st: &Store, spec: &Command) -> Result<Vec<Layer>, String> {
    let mask = util::touse(ds, st, spec)?;
    let vars = varlist::expand(ds, &spec.main)?;
    if vars.len() < 2 && !matches!(spec.verb.as_str(), "function") {
        return Err(rc_err(102, "too few variables specified"));
    }
    let x = vars.last().unwrap().clone();
    let ys = &vars[..vars.len() - 1];
    let mut out = Vec::new();
    for y in ys {
        let mut pts = xy_points(ds, &mask, y, &x)?;
        let kind = match spec.verb.as_str() {
            "scatter" | "sc" => LayerKind::Scatter,
            "line" => {
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                LayerKind::Line
            }
            "connected" => {
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                LayerKind::Connected
            }
            "area" => {
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                LayerKind::Area
            }
            "bar" => LayerKind::Bar,
            "lfit" | "qfit" => {
                let n = pts.len() as f64;
                let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
                let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
                let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
                let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
                let b = sxy / sxx;
                let lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                pts = (0..=50).map(|k| {
                    let xv = lo + (hi - lo) * k as f64 / 50.0;
                    (xv, my + b * (xv - mx))
                }).collect();
                LayerKind::Line
            }
            other => return Err(rc_err(198, format!("{other} is not a twoway plot type"))),
        };
        let label = if spec.verb == "lfit" { "Fitted values".to_string() } else { ds.get_var(y).map(|v| if v.label.is_empty() { y.clone() } else { v.label.clone() }).unwrap_or_default() };
        out.push(layer(kind, pts, &label));
    }
    Ok(out)
}

pub struct Twoway;

impl CommandHandler for Twoway {
    fn name(&self) -> &'static str {
        "twoway"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut graph = Graph { legend: true, ..Default::default() };
        let specs: Vec<String> = if cmd.verb == "twoway" || cmd.verb == "tw" {
            let main = cmd.raw.trim();
            let body = main.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("");
            // (plot) (plot) or plot || plot
            if body.trim_start().starts_with('(') {
                let (plots, _) = match ferrostat_parser::parser::find_top_level(body, ',') {
                    Some(i) => (&body[..i], &body[i..]),
                    None => (body, ""),
                };
                split_top_level_words(plots).into_iter().map(|p| p.trim().trim_start_matches('(').trim_end_matches(')').to_string()).collect()
            } else {
                body.split("||").map(|s| s.trim().to_string()).collect()
            }
        } else {
            vec![cmd.raw.clone()]
        };
        for spec in specs {
            let c = parse_line(&spec).map_err(|e| rc_err(198, e))?;
            if c.verb.is_empty() {
                continue;
            }
            graph.layers.extend(plot_layers(ds, st, &c)?);
            if graph.xtitle.is_empty() {
                if let Some(x) = varlist::expand(ds, &c.main)?.last() {
                    graph.xtitle = ds.get_var(x).map(|v| if v.label.is_empty() { x.clone() } else { v.label.clone() }).unwrap_or_default();
                }
            }
        }
        if graph.layers.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        graph.legend = graph.layers.len() > 1;
        if graph.layers.len() == 1 {
            graph.ytitle = graph.layers[0].label.clone();
        }
        finish(&mut graph, cmd, st)
    }
}

pub struct Histogram;

impl CommandHandler for Histogram {
    fn name(&self) -> &'static str {
        "histogram"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "f")?;
        let x = util::col(ds, &v)?;
        let pts: Vec<(f64, f64)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(x[i])).map(|i| (x[i], w.as_ref().map_or(1.0, |w| w.w[i]))).collect();
        if pts.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let n: f64 = pts.iter().map(|p| p.1).sum();
        let lo = util::opt_num(cmd, "start", 2)?.unwrap_or_else(|| pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min));
        let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let discrete = cmd.has_opt("discrete", 1);
        let (bins, width) = if discrete {
            let mut vals: Vec<f64> = pts.iter().map(|p| p.0).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            let wd = vals.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
            let wd = if wd.is_finite() { wd } else { 1.0 };
            (((hi - lo) / wd).round() as usize + 1, wd)
        } else if let Some(wd) = util::opt_num(cmd, "width", 1)? {
            (((hi - lo) / wd).ceil().max(1.0) as usize, wd)
        } else {
            // Stata default: min(sqrt(N), 10 ln(N)/ln(10))
            let k = util::opt_num(cmd, "bin", 3)?.map(|b| b as usize).unwrap_or_else(|| (n.sqrt().min(10.0 * n.ln() / 10f64.ln())).floor().max(1.0) as usize);
            (k, (hi - lo) / k as f64)
        };
        let start = if discrete { lo - width / 2.0 } else { lo };
        let mut counts = vec![0.0; bins];
        for (xv, wv) in &pts {
            let mut b = ((xv - start) / width).floor() as isize;
            if b as usize >= bins {
                b = bins as isize - 1;
            }
            counts[b.max(0) as usize] += wv;
        }
        let scale = if cmd.has_opt("frequency", 4) {
            1.0
        } else if cmd.has_opt("percent", 4) {
            100.0 / n
        } else if cmd.has_opt("fraction", 4) {
            1.0 / n
        } else {
            1.0 / (n * width)
        };
        let bars: Vec<(f64, f64)> = counts.iter().enumerate().map(|(k, c)| (start + (k as f64 + 0.5) * width, c * scale)).collect();
        let mut graph = Graph { xtitle: v.clone(), ytitle: if cmd.has_opt("frequency", 4) { "Frequency".into() } else if cmd.has_opt("percent", 4) { "Percent".into() } else { "Density".into() }, ..Default::default() };
        let mut l = layer(LayerKind::Bar, bars.clone(), &v);
        l.width = width;
        graph.layers.push(l);
        if cmd.has_opt("normal", 4) {
            let m = pts.iter().map(|p| p.0 * p.1).sum::<f64>() / n;
            let sd = (pts.iter().map(|p| p.1 * (p.0 - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            let curve: Vec<(f64, f64)> = (0..=100).map(|k| {
                let xv = start + (bins as f64 * width) * k as f64 / 100.0;
                (xv, crate::dist::normalden((xv - m) / sd) / sd / scale * scale * if cmd.has_opt("frequency", 4) { n * width } else if cmd.has_opt("percent", 4) { 100.0 * width } else { 1.0 })
            }).collect();
            graph.layers.push(layer(LayerKind::Line, curve, "Normal density"));
        }
        let mut msg = format!("(bin={}, start={}, width={})", bins, crate::format::fmt_g(start, 9), crate::format::fmt_g(width, 9));
        let r = st.begin_r();
        r.set("N", n);
        r.set("bin", bins as f64);
        r.set("start", start);
        r.set("width", width);
        r.set("max", counts.iter().cloned().fold(0.0, f64::max) * scale);
        let extra = finish(&mut graph, cmd, st)?;
        if !extra.is_empty() {
            msg = format!("{msg}\n{extra}");
        }
        Ok(msg)
    }
}

pub struct GraphCmd;

impl CommandHandler for GraphCmd {
    fn name(&self) -> &'static str {
        "graph"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let body = cmd.raw.trim().split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim().to_string();
        let sub = body.split_whitespace().next().unwrap_or("").to_lowercase();
        let rest = body[sub.len()..].trim().to_string();
        match sub.as_str() {
            "export" => {
                let c = parse_line(&format!("export {rest}")).map_err(|e| rc_err(198, e))?;
                let file = util::unquote(c.main.trim());
                let g = match c.opt("name", 2) {
                    Some(n) => st.graphs.get(n.trim()).cloned(),
                    None => st.graph.clone(),
                }
                .ok_or_else(|| rc_err(693, "no graph in memory"))?;
                if std::path::Path::new(&file).exists() && !c.has_opt("replace", 3) {
                    return Err(rc_err(602, format!("file {file} already exists")));
                }
                let width = util::opt_num(&c, "width", 2)?.unwrap_or(800.0) as u32;
                let height = util::opt_num(&c, "height", 2)?.unwrap_or(width as f64 * 0.75) as u32;
                render(&g, &file, width, height)?;
                Ok(format!("file {file} written in {} format", if file.ends_with(".svg") { "SVG" } else { "PNG" }))
            }
            "twoway" | "tw" => {
                let c = parse_line(&format!("twoway {rest}")).map_err(|e| rc_err(198, e))?;
                Twoway.execute(ds, st, &c)
            }
            "bar" | "hbar" | "dot" => graph_bar(ds, st, &rest),
            "box" | "hbox" => graph_box(ds, st, &rest),
            "drop" => {
                if rest.trim() == "_all" {
                    st.graphs.clear();
                    st.graph = None;
                } else {
                    for n in rest.split_whitespace() {
                        st.graphs.remove(n);
                    }
                }
                Ok(String::new())
            }
            "dir" => Ok(st.graphs.keys().cloned().collect::<Vec<_>>().join("\n")),
            "display" => Ok(String::new()),
            "combine" | "save" | "use" => Err(rc_err(198, format!("graph {sub} not supported"))),
            _ => Err(rc_err(198, format!("graph {sub}: invalid subcommand"))),
        }
    }
}

fn graph_bar(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let c = parse_line(&format!("bar {rest}")).map_err(|e| rc_err(198, e))?;
    let mask = util::touse(ds, st, &c)?;
    let mut stat = "mean".to_string();
    let mut vars = Vec::new();
    for w in split_top_level_words(&c.main) {
        if w.starts_with('(') {
            stat = w.trim_matches(|ch| ch == '(' || ch == ')').to_string();
        } else {
            vars.extend(varlist::expand(ds, &w)?);
        }
    }
    let over = c.opt("over", 2).map(|o| varlist::resolve_name(ds, o.split(',').next().unwrap().trim())).transpose()?;
    let mut graph = Graph::default();
    let groups: Vec<(String, Vec<bool>)> = match &over {
        Some(o) => {
            let (ids, reps) = util::group_ids(ds, &[o.clone()], &mask);
            let var = ds.get_var(o).unwrap();
            reps.iter().enumerate().map(|(g, &r)| (util::cell_text(ds, var, r, true), (0..ds.n_obs).map(|i| ids[i] == g).collect())).collect()
        }
        None => vec![(String::new(), mask.clone())],
    };
    let nv = vars.len().max(1);
    for (vi, v) in vars.iter().enumerate() {
        let mut pts = Vec::new();
        for (gi, (_, m)) in groups.iter().enumerate() {
            let x: Vec<f64> = util::rows(m).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
            let val = match stat.as_str() {
                "sum" => x.iter().sum(),
                "count" => x.len() as f64,
                "max" => x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "min" => x.iter().cloned().fold(f64::INFINITY, f64::min),
                "median" => {
                    let mut s = x.clone();
                    s.sort_by(f64::total_cmp);
                    util::pctile_sorted(&s, 50.0)
                }
                _ => util::mean(&x),
            };
            pts.push((gi as f64 + (vi as f64 + 0.5) / nv as f64, val));
        }
        let mut l = layer(LayerKind::Bar, pts, &format!("{stat} of {v}"));
        l.width = 0.8 / nv as f64;
        graph.layers.push(l);
    }
    graph.xlabels = groups.iter().enumerate().map(|(i, (n, _))| (i as f64 + 0.5, n.clone())).collect();
    graph.legend = vars.len() > 1;
    finish(&mut graph, &c, st)
}

fn graph_box(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let c = parse_line(&format!("box {rest}")).map_err(|e| rc_err(198, e))?;
    let mask = util::touse(ds, st, &c)?;
    let vars = varlist::expand(ds, &c.main)?;
    let mut graph = Graph::default();
    let mut l = layer(LayerKind::Box, Vec::new(), "");
    for (k, v) in vars.iter().enumerate() {
        let mut x: Vec<f64> = util::rows(&mask).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
        x.sort_by(f64::total_cmp);
        let q1 = util::pctile_sorted(&x, 25.0);
        let q3 = util::pctile_sorted(&x, 75.0);
        let iqr = q3 - q1;
        let lo = x.iter().cloned().find(|&v| v >= q1 - 1.5 * iqr).unwrap_or(q1);
        let hi = x.iter().rev().cloned().find(|&v| v <= q3 + 1.5 * iqr).unwrap_or(q3);
        l.boxes.push([q1, util::pctile_sorted(&x, 50.0), q3, lo, hi]);
        graph.xlabels.push((k as f64 + 0.5, v.clone()));
    }
    graph.layers.push(l);
    finish(&mut graph, &c, st)
}

const COLORS: [RGBColor; 6] = [RGBColor(26, 71, 111), RGBColor(144, 53, 59), RGBColor(85, 117, 47), RGBColor(227, 126, 0), RGBColor(110, 142, 132), RGBColor(193, 5, 52)];

/// Render a graph to a file (PNG by default, SVG for .svg).
pub fn render(g: &Graph, path: &str, w: u32, h: u32) -> Result<(), String> {
    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (w, h)).into_drawing_area();
        draw(g, &root)
    } else {
        let root = BitMapBackend::new(path, (w, h)).into_drawing_area();
        draw(g, &root)
    }
}

fn draw<DB: DrawingBackend>(g: &Graph, root: &DrawingArea<DB, plotters::coord::Shift>) -> Result<(), String>
where
    DB::ErrorType: 'static,
{
    let e = |x: DrawingAreaErrorKind<DB::ErrorType>| rc_err(603, format!("could not render graph: {x:?}"));
    root.fill(&WHITE).map_err(e)?;
    let mut xs: Vec<f64> = g.layers.iter().flat_map(|l| l.points.iter().map(|p| p.0)).collect();
    let mut ys: Vec<f64> = g.layers.iter().flat_map(|l| l.points.iter().map(|p| p.1)).collect();
    for l in &g.layers {
        for b in &l.boxes {
            ys.extend(b.iter());
        }
        if l.kind == LayerKind::Bar {
            ys.push(0.0);
            xs.extend(l.points.iter().flat_map(|p| [p.0 - l.width / 2.0, p.0 + l.width / 2.0]));
        }
    }
    for (x, _) in &g.xlabels {
        xs.push(x - 0.5);
        xs.push(x + 0.5);
    }
    if xs.is_empty() {
        xs = vec![0.0, 1.0];
    }
    if ys.is_empty() {
        ys = vec![0.0, 1.0];
    }
    let (x0, x1) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (y0, y1) = (ys.iter().cloned().fold(f64::INFINITY, f64::min), ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let pad = |a: f64, b: f64| if b > a { (b - a) * 0.05 } else { 1.0 };
    let (px, py) = (pad(x0, x1), pad(y0, y1));
    let mut chart = ChartBuilder::on(root)
        .caption(&g.title, ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d((x0 - px)..(x1 + px), (y0 - py)..(y1 + py))
        .map_err(e)?;
    let labels = g.xlabels.clone();
    let mut mesh = chart.configure_mesh();
    mesh.x_desc(g.xtitle.clone()).y_desc(g.ytitle.clone()).light_line_style(WHITE);
    let fmt_x = move |x: &f64| {
        if labels.is_empty() {
            crate::format::fmt_g(*x, 6)
        } else {
            labels.iter().find(|(p, _)| (p - x).abs() < 0.26).map(|(_, l)| l.clone()).unwrap_or_default()
        }
    };
    mesh.x_label_formatter(&fmt_x);
    mesh.draw().map_err(e)?;
    for (k, l) in g.layers.iter().enumerate() {
        let col = COLORS[k % COLORS.len()];
        let label = l.label.clone();
        match l.kind {
            LayerKind::Scatter => {
                chart.draw_series(l.points.iter().map(|p| Circle::new(*p, 3, col.filled()))).map_err(e)?.label(label).legend(move |(x, y)| Circle::new((x + 10, y), 3, col.filled()));
            }
            LayerKind::Line => {
                chart.draw_series(LineSeries::new(l.points.clone(), col.stroke_width(2))).map_err(e)?.label(label).legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], col));
            }
            LayerKind::Connected => {
                chart.draw_series(LineSeries::new(l.points.clone(), col.stroke_width(2))).map_err(e)?;
                chart.draw_series(l.points.iter().map(|p| Circle::new(*p, 3, col.filled()))).map_err(e)?.label(label).legend(move |(x, y)| Circle::new((x + 10, y), 3, col.filled()));
            }
            LayerKind::Area => {
                chart.draw_series(AreaSeries::new(l.points.clone(), 0.0, col.mix(0.4)).border_style(col)).map_err(e)?.label(label).legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.filled()));
            }
            LayerKind::Bar => {
                let hw = l.width / 2.0 * 0.95;
                chart.draw_series(l.points.iter().map(|p| Rectangle::new([(p.0 - hw, 0.0), (p.0 + hw, p.1)], col.mix(0.8).filled()))).map_err(e)?.label(label).legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.filled()));
            }
            LayerKind::Box => {
                for (i, b) in l.boxes.iter().enumerate() {
                    let c = i as f64 + 0.5;
                    chart.draw_series(std::iter::once(Rectangle::new([(c - 0.25, b[0]), (c + 0.25, b[2])], col.mix(0.6).filled()))).map_err(e)?;
                    chart.draw_series(std::iter::once(PathElement::new(vec![(c - 0.25, b[1]), (c + 0.25, b[1])], WHITE.stroke_width(2)))).map_err(e)?;
                    chart.draw_series(std::iter::once(PathElement::new(vec![(c, b[2]), (c, b[4])], col))).map_err(e)?;
                    chart.draw_series(std::iter::once(PathElement::new(vec![(c, b[0]), (c, b[3])], col))).map_err(e)?;
                }
            }
        }
    }
    if g.legend {
        chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw().map_err(e)?;
    }
    root.present().map_err(e)?;
    Ok(())
}
