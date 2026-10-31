//! Graphics: twoway (scatter, line, connected, lfit, area, bar), histogram,
//! graph bar/hbar/box, kdensity plots. Commands build the graph in memory;
//! `graph export` renders it to PNG or SVG.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_line, split_top_level_words};
use ferrostat_parser::Command;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::commands::CommandHandler;
use crate::dist;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum LayerKind {
    #[default]
    Scatter,
    Line,
    Connected,
    Area,
    Bar,
    /// box plot: (q1, median, q3, lower whisker, upper whisker)
    Box,
    /// range plots (rbar, rarea, rspike, ...): `ranges` holds (x, y1, y2)
    Range(RangeStyle),
    /// sunflowers at `points`, with `sizes` petals each
    Sunflower,
    /// twoway dot: markers with dotted guide lines across the plot
    Dot,
    /// paired-coordinate plots: `segments` hold (y1, x1, y2, x2)
    Pc(PcStyle),
    /// filled cells (contour, heatmap): `cells` (x0, y0, x1, y1) with `cell_colors`
    Surface,
    /// contour lines: `segments` with `seg_colors`
    ContourLines,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcStyle {
    Spike,
    CapSym,
    Arrow,
    BArrow,
    Scatter,
}

impl PcStyle {
    fn from_verb(v: &str) -> Option<PcStyle> {
        Some(match v {
            "pcspike" | "pci" => PcStyle::Spike,
            "pccapsym" => PcStyle::CapSym,
            "pcarrow" | "pcarrowi" => PcStyle::Arrow,
            "pcbarrow" | "pcbarrowi" => PcStyle::BArrow,
            "pcscatter" => PcStyle::Scatter,
            _ => return None,
        })
    }
    fn from_name(s: &str) -> PcStyle {
        match s {
            "CapSym" => PcStyle::CapSym,
            "Arrow" => PcStyle::Arrow,
            "BArrow" => PcStyle::BArrow,
            "Scatter" => PcStyle::Scatter,
            _ => PcStyle::Spike,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RangeStyle {
    Bar,
    Area,
    Spike,
    Cap,
    Line,
    Connected,
    Scatter,
    /// spike from y1 to y2 with a marker at y2 (dropline)
    Drop,
    /// range with a marker at a third value (`points`): rpcap, rpspike
    PointCap,
    PointSpike,
}

impl RangeStyle {
    fn from_verb(v: &str) -> Option<RangeStyle> {
        Some(match v {
            "rbar" => RangeStyle::Bar,
            "rarea" => RangeStyle::Area,
            "rspike" => RangeStyle::Spike,
            "rcap" | "rcapsym" => RangeStyle::Cap,
            "rline" => RangeStyle::Line,
            "rconnected" => RangeStyle::Connected,
            "rscatter" => RangeStyle::Scatter,
            "rpcap" => RangeStyle::PointCap,
            "rpspike" => RangeStyle::PointSpike,
            _ => return None,
        })
    }
    fn from_name(s: &str) -> Option<RangeStyle> {
        Some(match s {
            "Bar" => RangeStyle::Bar,
            "Area" => RangeStyle::Area,
            "Spike" => RangeStyle::Spike,
            "Cap" => RangeStyle::Cap,
            "Line" => RangeStyle::Line,
            "Connected" => RangeStyle::Connected,
            "Scatter" => RangeStyle::Scatter,
            "Drop" => RangeStyle::Drop,
            "PointCap" => RangeStyle::PointCap,
            "PointSpike" => RangeStyle::PointSpike,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layer {
    pub kind: LayerKind,
    pub points: Vec<(f64, f64)>,
    pub label: String,
    /// for bars: bar width; for box: five-number summaries
    pub width: f64,
    pub boxes: Vec<[f64; 5]>,
    /// range plots: (x, y1, y2), and whether x runs along the vertical axis
    pub ranges: Vec<(f64, f64, f64)>,
    pub horizontal: bool,
    /// scatter: marker size multipliers ([w=] weights), marker labels, hollow symbols
    pub sizes: Vec<f64>,
    pub mlabels: Vec<String>,
    pub hollow: bool,
    /// fill opacity (confidence bands are translucent); 0 means opaque
    pub alpha: f64,
    /// bars and areas drawn as outlines only (fcolor(none))
    pub nofill: bool,
    /// box plots: draw the median as a marker (medtype(marker))
    pub median_marker: bool,
    /// msymbol() and lpattern() names, and an explicit colour (mcolor/lcolor/fcolor/color)
    pub symbol: String,
    pub pattern: String,
    pub color: Option<(u8, u8, u8)>,
    /// paired-coordinate segments (y1, x1, y2, x2)
    pub segments: Vec<[f64; 4]>,
    /// surfaces: rectangles (x0, y0, x1, y1) and their colours; colours of segments
    pub cells: Vec<[f64; 4]>,
    pub cell_colors: Vec<(u8, u8, u8)>,
    pub seg_colors: Vec<(u8, u8, u8)>,
    /// colour key of a surface layer: (range label, colour)
    pub key: Vec<(String, (u8, u8, u8))>,
    /// observation numbers (1-based) of `points`, when they are observations (hover readouts)
    pub obs: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub title: String,
    pub xtitle: String,
    /// display formats of the axis variables (date formats label the ticks)
    pub xformat: String,
    pub yformat: String,
    /// colour scheme name (empty = default, Okabe-Ito)
    pub scheme: String,
    pub ytitle: String,
    pub layers: Vec<Layer>,
    /// category labels at fixed positions (bar, box and dot charts)
    pub xlabels: Vec<(f64, String)>,
    pub ylabels: Vec<(f64, String)>,
    /// dotted guide lines across the plot at these y positions (dot charts)
    pub guides: Vec<f64>,
    /// legend(order(# "text" ...)): layer numbers (1-based) and labels shown
    pub legend_order: Vec<(usize, String)>,
    /// pie charts: slices in drawing order, and the angle of the first edge (degrees)
    pub pie: Vec<PieSlice>,
    pub pie_angle0: Option<f64>,
    /// colour key of surfaces: (range label, colour)
    pub color_legend: Vec<(String, (u8, u8, u8))>,
    /// text drawn in the middle of an otherwise empty plot (graph matrix diagonal)
    pub center_text: String,
    /// the command that drew the graph, and when
    pub command: String,
    pub created: String,
    pub legend: bool,
    pub name: String,
    /// Reference lines from yline()/xline().
    pub ylines: Vec<f64>,
    pub xlines: Vec<f64>,
    /// Choropleth polygons (spmap) and their legend.
    pub polygons: Vec<MapPolygon>,
    pub map_legend: Vec<(String, (u8, u8, u8))>,
    /// Heading above the map legend (spmap `legtitle()`).
    pub map_legend_title: String,
    /// Sub-graphs laid out in a grid (graph combine).
    pub children: Vec<Graph>,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PieSlice {
    pub label: String,
    pub value: f64,
    pub explode: bool,
    /// text drawn inside the slice (plabel)
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapPolygon {
    pub points: Vec<(f64, f64)>,
    pub fill: (u8, u8, u8),
}

fn opt_text(cmd: &Command, name: &str, min: usize) -> Option<String> {
    cmd.opt(name, min).map(|s| util::unquote(s.split(',').next().unwrap_or(s)))
}

pub(crate) fn finish(graph: &mut Graph, cmd: &Command, st: &mut Store) -> Result<String, String> {
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
        graph.legend = !l.contains("off") || l.contains("on");
        if let Some(i) = l.find("order(") {
            // order(1 "label" 2 3 "label" ...)
            let inner = &l[i + 6..];
            let mut depth = 1;
            let end = inner.char_indices().find(|(_, c)| {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                depth == 0
            }).map_or(inner.len(), |(j, _)| j);
            let mut order: Vec<(usize, String)> = Vec::new();
            for w in split_top_level_words(&inner[..end]) {
                if let Ok(k) = w.parse::<usize>() {
                    order.push((k, String::new()));
                } else if let Some(last) = order.last_mut() {
                    let text = util::unquote(&w);
                    last.1 = if last.1.is_empty() { text } else { format!("{} {}", last.1, text) };
                }
            }
            graph.legend_order = order;
            graph.legend = true;
        }
        // label(# "text" ...): rename the legend key of plot #
        for (name, arg) in ferrostat_parser::parser::split_options(l) {
            let (Some(arg), true) = (arg, name.len() >= 3 && "label".starts_with(name.to_lowercase().as_str())) else { continue };
            let words = split_top_level_words(&arg);
            let Some(k) = words.first().and_then(|w| w.parse::<usize>().ok()) else { continue };
            let text = words[1..].iter().map(|w| util::unquote(w)).collect::<Vec<_>>().join(" ");
            let mut relabel = |g: &mut Graph| {
                if let Some(layer) = g.layers.get_mut(k.wrapping_sub(1)) {
                    layer.label = text.clone();
                }
                if let Some(o) = g.legend_order.iter_mut().find(|o| o.0 == k) {
                    o.1 = text.clone();
                }
            };
            relabel(graph);
            graph.children.iter_mut().for_each(&mut relabel);
        }
    }
    let nums = |s: &str| -> Vec<f64> { s.split(',').next().unwrap_or("").split_whitespace().filter_map(|x| x.parse().ok()).collect() };
    if let Some(y) = cmd.opt("yline", 5) {
        graph.ylines = nums(y);
    }
    if let Some(x) = cmd.opt("xline", 5) {
        graph.xlines = nums(x);
    }
    graph.name = cmd.opt("name", 4).map(|s| s.split(',').next().unwrap().trim().to_string()).unwrap_or_else(|| "Graph".into());
    if graph.command.is_empty() {
        graph.command = cmd.raw.trim().to_string();
        graph.created = chrono::Local::now().format("%d %b %Y %H:%M").to_string();
    }
    st.graph_window_closed = false;
    graph.scheme = match cmd.opt("scheme", 3).map(|s| s.trim().to_lowercase()) {
        Some(s) if is_scheme(&s) => s,
        _ => st.scheme.clone(),
    };
    for child in graph.children.iter_mut() {
        if child.scheme.is_empty() {
            child.scheme = graph.scheme.clone();
        }
    }
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

/// Does a plot variable carry a time-series operator (`d.gnp96`, `L2.x`, `DS12.air`)?
fn is_ts_op(v: &str) -> bool {
    v.split_once('.').is_some_and(|(op, _)| !op.is_empty() && op.chars().all(|c| c.is_ascii_digit() || "lLdDfFsS".contains(c)))
}

/// Plot variables: a varlist whose items may carry time-series operators.
fn plot_varlist(ds: &DataSet, main: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for w in main.split_whitespace() {
        if is_ts_op(w) {
            let (op, v) = w.split_once('.').unwrap();
            out.push(format!("{}.{}", op.to_uppercase(), varlist::resolve_name(ds, v)?));
        } else {
            out.extend(varlist::expand(ds, w)?);
        }
    }
    Ok(out)
}

/// Values of a plot variable (operators evaluated on the tsset data).
pub(crate) fn plot_col(ds: &DataSet, st: &Store, v: &str) -> Result<Vec<f64>, String> {
    if is_ts_op(v) {
        let e = ferrostat_parser::parser::parse_expression(v).map_err(|e| rc_err(198, e.message))?;
        util::eval_col(ds, st, &e, &vec![true; ds.n_obs])
    } else {
        util::col(ds, v)
    }
}

fn xy_points(ds: &DataSet, st: &Store, mask: &[bool], y: &str, x: &str) -> Result<Vec<(f64, f64)>, String> {
    let yc = plot_col(ds, st, y)?;
    let xc = plot_col(ds, st, x)?;
    Ok((0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(xc[i])).map(|i| (xc[i], yc[i])).collect())
}

/// Points with their observation numbers (1-based).
fn xy_points_obs(ds: &DataSet, st: &Store, mask: &[bool], y: &str, x: &str) -> Result<(Vec<(f64, f64)>, Vec<u32>), String> {
    let yc = plot_col(ds, st, y)?;
    let xc = plot_col(ds, st, x)?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(xc[i])).collect();
    Ok((rows.iter().map(|&i| (xc[i], yc[i])).collect(), rows.iter().map(|&i| i as u32 + 1).collect()))
}

/// Sort points by x (stable), keeping their observation numbers alongside.
fn sort_by_x(pts: &mut Vec<(f64, f64)>, obs: &mut Vec<u32>) {
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    idx.sort_by(|&a, &b| pts[a].0.total_cmp(&pts[b].0));
    *pts = idx.iter().map(|&i| pts[i]).collect();
    *obs = idx.iter().map(|&i| obs[i]).collect();
}

fn layer(kind: LayerKind, points: Vec<(f64, f64)>, label: &str) -> Layer {
    Layer { kind, points, label: label.to_string(), ..Default::default() }
}

/// Build the layers for one plot specification, then apply its style
/// options: msymbol(), lpattern() and colours (one entry per layer).
fn plot_layers(ds: &DataSet, st: &Store, spec: &Command, within: Option<&[bool]>) -> Result<Vec<Layer>, String> {
    let mut layers = plot_layers_raw(ds, st, spec, within)?;
    let list = |name: &str, min: usize| -> Vec<String> { spec.opt(name, min).map(|s| split_top_level_words(&s).into_iter().map(|w| util::unquote(&w)).collect()).unwrap_or_default() };
    let symbols = list("msymbol", 2);
    let patterns = list("lpattern", 3);
    // "255 0 0" arrives as three words: regroup RGB triplets
    let grouped = |colours: Vec<String>| -> Vec<String> {
        let mut grouped: Vec<String> = Vec::new();
        let mut i = 0;
        while i < colours.len() {
            if i + 2 < colours.len() && colours[i..i + 3].iter().all(|w| w.parse::<u8>().is_ok()) {
                grouped.push(colours[i..i + 3].join(" "));
                i += 3;
            } else {
                grouped.push(colours[i].clone());
                i += 1;
            }
        }
        grouped
    };
    // colour(), then the marker, line and fill colours; `none` fills nothing
    let lists: Vec<Vec<String>> = ["color", "mcolor", "lcolor", "fcolor"].iter().map(|n| grouped(list(n, 3))).collect();
    let pick = |v: &Vec<String>, k: usize| v.get(k).or(v.last()).cloned();
    for (k, l) in layers.iter_mut().enumerate() {
        let filled = matches!(l.kind, LayerKind::Area | LayerKind::Bar | LayerKind::Range(RangeStyle::Bar) | LayerKind::Range(RangeStyle::Area));
        l.nofill = filled && [&lists[0], &lists[3]].iter().any(|v| pick(v, k).is_some_and(|c| c.eq_ignore_ascii_case("none")));
        if let Some(c) = lists.iter().filter_map(|v| pick(v, k)).find(|c| !c.eq_ignore_ascii_case("none")) {
            // fills take `colour%NN` as opacity; other plots get a lighter colour
            match c.split_once('%').filter(|_| filled) {
                Some((base, pct)) => {
                    l.color = Some(crate::colors::parse(base).ok_or_else(|| rc_err(198, format!("{c}: invalid color")))?);
                    let a: f64 = pct.trim().parse().map_err(|_| rc_err(198, format!("{c}: invalid color")))?;
                    l.alpha = (a / 100.0).clamp(0.01, 1.0);
                }
                None => l.color = Some(crate::colors::parse(&c).ok_or_else(|| rc_err(198, format!("{c}: invalid color")))?),
            }
        }
    }
    for (k, l) in layers.iter_mut().enumerate() {
        if let Some(s) = symbols.get(k).or(symbols.last()) {
            l.symbol = s.clone();
            l.hollow = symbol_shape(s).1;
        }
        if let Some(p) = patterns.get(k).or(patterns.last()) {
            l.pattern = p.clone();
        }
    }
    Ok(layers)
}

/// Build the layers for one plot specification such as `scatter y x`;
/// `within` restricts the sample further (one panel of `by()`).
fn plot_layers_raw(ds: &DataSet, st: &Store, spec: &Command, within: Option<&[bool]>) -> Result<Vec<Layer>, String> {
    let mut mask = util::touse(ds, st, spec)?;
    // marker sizes: `[weight=]` means analytic weights, which need not be integers
    let wts = util::weights(ds, st, spec, &mut mask, "af")?;
    if let Some(w) = within {
        for (m, k) in mask.iter_mut().zip(w) {
            *m = *m && *k;
        }
    }
    match spec.verb.as_str() {
        "scatteri" | "pci" | "pcarrowi" | "pcbarrowi" => return immediate_layers(spec),
        "function" => return function_layers(spec),
        "tsline" | "tsrline" => return plot_layers_raw(ds, st, &ts_as_line(st, spec)?, within),
        "histogram" | "kdensity" => {
            let v = plot_varlist(ds, &spec.main)?.into_iter().next().ok_or_else(|| rc_err(102, "too few variables specified"))?;
            let c = plot_col(ds, st, &v)?;
            let mut xs: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect();
            if xs.len() < 2 {
                return Err(rc_err(2000, "no observations"));
            }
            xs.sort_by(f64::total_cmp);
            return Ok(vec![if spec.verb == "histogram" { twoway_histogram(spec, &xs)? } else { twoway_kdensity(spec, &xs)? }]);
        }
        _ => {}
    }
    let vars = plot_varlist(ds, &spec.main)?;
    if vars.len() < 2 && !matches!(spec.verb.as_str(), "function") {
        return Err(rc_err(102, "too few variables specified"));
    }
    let x = vars.last().unwrap().clone();
    let ys = &vars[..vars.len() - 1];
    let lab = |v: &str| ds.get_var(v).map(|var| if var.label.is_empty() { v.to_string() } else { var.label.clone() }).unwrap_or_default();
    let horizontal = spec.has_opt("horizontal", 3);
    let base = util::opt_num(spec, "base", 4)?.unwrap_or(0.0);
    // bar, spike, dropline and horizontal area: ranges from base() to y
    let based = match spec.verb.as_str() {
        "bar" => Some(RangeStyle::Bar),
        "spike" => Some(RangeStyle::Spike),
        "dropline" => Some(RangeStyle::Drop),
        "area" if horizontal => Some(RangeStyle::Area),
        _ => None,
    };
    if let Some(style) = based {
        let mut out = Vec::new();
        for y in ys {
            let pts = xy_points(ds, st, &mask, y, &x)?;
            let mut l = layer(LayerKind::Range(style), Vec::new(), &lab(y));
            l.ranges = pts.iter().map(|&(px, py)| (px, base, py)).collect();
            l.ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
            l.width = util::opt_num(spec, "barwidth", 4)?.unwrap_or(1.0);
            l.horizontal = horizontal;
            out.push(l);
        }
        return Ok(out);
    }
    // surfaces: z y x
    if matches!(spec.verb.as_str(), "contour" | "contourline" | "heatmap") {
        if vars.len() != 3 {
            return Err(rc_err(if vars.len() < 3 { 102 } else { 103 }, format!("{}: zvar yvar xvar required", spec.verb)));
        }
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| plot_col(ds, st, v)).collect::<Result<_, _>>()?;
        let pts: Vec<(f64, f64, f64)> = (0..ds.n_obs).filter(|&i| mask[i] && cols.iter().all(|c| !is_missing(c[i]))).map(|i| (cols[2][i], cols[1][i], cols[0][i])).collect();
        if pts.len() < 3 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        return surface_layers(spec, &pts, &lab(&vars[0]));
    }
 // paired coordinates: y1 x1 y2 x2
    if let Some(style) = PcStyle::from_verb(spec.verb.as_str()) {
        if vars.len() != 4 {
            return Err(rc_err(if vars.len() < 4 { 102 } else { 103 }, format!("{}: y1var x1var y2var x2var required", spec.verb)));
        }
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| plot_col(ds, st, v)).collect::<Result<_, _>>()?;
        let mut l = layer(LayerKind::Pc(style), Vec::new(), &format!("{}/{}", lab(&vars[0]), lab(&vars[2])));
        l.segments = (0..ds.n_obs).filter(|&i| mask[i] && cols.iter().all(|c| !is_missing(c[i]))).map(|i| [cols[0][i], cols[1][i], cols[2][i], cols[3][i]]).collect();
        return Ok(vec![l]);
    }
    // range with a point: y1 y2 y3 x
    if matches!(spec.verb.as_str(), "rpcap" | "rpspike") {
        if vars.len() != 4 {
            return Err(rc_err(if vars.len() < 4 { 102 } else { 103 }, format!("{}: y1var y2var y3var xvar required", spec.verb)));
        }
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| plot_col(ds, st, v)).collect::<Result<_, _>>()?;
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && cols.iter().all(|c| !is_missing(c[i]))).collect();
        let style = if spec.verb == "rpcap" { RangeStyle::PointCap } else { RangeStyle::PointSpike };
        let mut l = layer(LayerKind::Range(style), rows.iter().map(|&i| (cols[3][i], cols[2][i])).collect(), &format!("{}/{}", lab(&vars[0]), lab(&vars[1])));
        l.ranges = rows.iter().map(|&i| (cols[3][i], cols[0][i], cols[1][i])).collect();
        l.ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
        l.horizontal = horizontal;
        return Ok(vec![l]);
    }
    // scatterplot smoothers
    if matches!(spec.verb.as_str(), "lowess" | "lpoly" | "lpolyci" | "mband" | "mspline" | "dot") {
        let y = &ys[0];
        let pts = xy_points(ds, st, &mask, y, &x)?;
        if pts.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let label = lab(y);
        return Ok(match spec.verb.as_str() {
            "dot" => {
                let mut l = layer(LayerKind::Dot, pts, &label);
                l.horizontal = horizontal;
                vec![l]
            }
            "lowess" => {
                let bw = util::opt_num(spec, "bwidth", 2)?.unwrap_or(0.8);
                vec![layer(LayerKind::Line, crate::smooth::lowess(&pts, bw, spec.has_opt("mean", 4), spec.has_opt("noweight", 3)), &format!("lowess {y} {x}"))]
            }
            "lpoly" | "lpolyci" => {
                let n = util::opt_num(spec, "n", 1)?.map_or(50, |v| v as usize).max(2);
                let grid: Vec<f64> = (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect();
                let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
                let h = util::opt_num(spec, "bwidth", 2)?.unwrap_or_else(|| crate::smooth::rot_bandwidth(&xs));
                let degree = util::opt_num(spec, "degree", 3)?.unwrap_or(0.0) as usize;
                let kname = crate::kdensity::canonical_kernel(&spec.opt("kernel", 1).unwrap_or("epanechnikov").trim().to_lowercase());
                let fit = crate::smooth::lpoly(&pts, &grid, kname, degree, h);
                let mut out = Vec::new();
                if spec.verb == "lpolyci" {
                    let level = util::level(spec, st)?;
                    let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
                    let mut band = layer(LayerKind::Range(RangeStyle::Area), Vec::new(), &format!("{}% CI", fmt_g(level, 4)));
                    band.ranges = fit.iter().map(|&(gx, m, se)| (gx, m - z * se, m + z * se)).collect();
                    band.alpha = 0.35;
                    out.push(band);
                }
                out.push(layer(LayerKind::Line, fit.iter().map(|&(gx, m, _)| (gx, m)).collect(), "lpoly smooth"));
                out
            }
            _ => {
                let bands = util::opt_num(spec, "bands", 2)?.map_or(200, |b| b as usize).max(1);
                let med = crate::smooth::median_bands(&pts, bands);
                let line = if spec.verb == "mspline" { crate::smooth::natural_spline(&med, util::opt_num(spec, "n", 1)?.map_or(10, |v| v as usize).max(1)) } else { med };
                vec![layer(LayerKind::Line, line, &format!("{} {y} {x}", spec.verb))]
            }
        });
    }
    // prediction plots (with confidence bands for the *ci variants)
    if let Some(model) = spec.verb.strip_suffix("ci").or(Some(spec.verb.as_str())).filter(|v| matches!(*v, "lfit" | "qfit" | "fpfit")) {
        let with_ci = spec.verb.ends_with("ci");
        let y = &ys[0];
        let pts = xy_points(ds, st, &mask, y, &x)?;
        let fit = fit_curve(&pts, model)?;
        let mut out = Vec::new();
        if with_ci {
            let level = util::level(spec, st)?;
            let mut band = layer(LayerKind::Range(RangeStyle::Area), Vec::new(), &format!("{}% CI", fmt_g(level, 4)));
            let tq = dist::invttail(fit.df, (1.0 - level / 100.0) / 2.0);
            band.ranges = fit.grid.iter().map(|&(gx, gy, se)| (gx, gy - tq * se, gy + tq * se)).collect();
            band.alpha = 0.35;
            out.push(band);
        }
        out.push(layer(LayerKind::Line, fit.grid.iter().map(|&(gx, gy, _)| (gx, gy)).collect(), "Fitted values"));
        return Ok(out);
    }
    if let Some(style) = RangeStyle::from_verb(spec.verb.as_str()) {
        if vars.len() != 3 {
            return Err(rc_err(if vars.len() < 3 { 102 } else { 103 }, format!("{}: y1var y2var xvar required", spec.verb)));
        }
        let (c1, c2, cx) = (plot_col(ds, st, &vars[0])?, plot_col(ds, st, &vars[1])?, plot_col(ds, st, &x)?);
        let mut ranges: Vec<(f64, f64, f64)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c1[i]) && !is_missing(c2[i]) && !is_missing(cx[i])).map(|i| (cx[i], c1[i], c2[i])).collect();
        ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut l = layer(LayerKind::Range(style), Vec::new(), &format!("{}/{}", lab(&vars[0]), lab(&vars[1])));
        l.width = util::opt_num(spec, "barwidth", 4)?.unwrap_or(1.0);
        l.horizontal = spec.has_opt("horizontal", 3);
        l.ranges = ranges;
        return Ok(vec![l]);
    }
    let mut out = Vec::new();
    for y in ys {
        let (mut pts, mut obs) = xy_points_obs(ds, st, &mask, y, &x)?;
        let kind = match spec.verb.as_str() {
            "scatter" | "sc" => LayerKind::Scatter,
            "line" => {
                sort_by_x(&mut pts, &mut obs);
                LayerKind::Line
            }
            "connected" => {
                sort_by_x(&mut pts, &mut obs);
                LayerKind::Connected
            }
            "area" => {
                sort_by_x(&mut pts, &mut obs);
                LayerKind::Area
            }
            "bar" => LayerKind::Bar,
            other => return Err(rc_err(198, format!("{other} is not a twoway plot type"))),
        };
        let label = lab(y);
        let mut l = layer(kind, pts, &label);
        l.obs = obs;
        if l.kind == LayerKind::Scatter {
            scatter_extras(ds, st, spec, &mask, y, &x, wts.as_ref(), &mut l)?;
        }
        out.push(l);
    }
    Ok(out)
}

/// Fitted curve on a 100-point grid: (x, prediction, standard error of the mean prediction).
struct Fit {
    grid: Vec<(f64, f64, f64)>,
    df: f64,
}

/// Least-squares fit of y on a basis of x: linear (`lfit`), quadratic
/// (`qfit`) or a degree-2 fractional polynomial (`fpfit`; Royston & Altman
/// 1994: powers from {-2, -1, -0.5, 0, 0.5, 1, 2, 3}, 0 meaning ln x, a
/// repeated power p giving x^p and x^p ln x; the pair with the smallest
/// residual sum of squares is used).
fn fit_curve(pts: &[(f64, f64)], model: &str) -> Result<Fit, String> {
    if pts.len() < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    // fractional polynomials need x > 0
    let shift = if model == "fpfit" && lo <= 0.0 { 1.0 - lo } else { 0.0 };
    let fp = |p: f64, x: f64| if p == 0.0 { x.ln() } else { x.powf(p) };
    let bases: Vec<Vec<Box<dyn Fn(f64) -> f64>>> = match model {
        "lfit" => vec![vec![Box::new(|x| x)]],
        "qfit" => vec![vec![Box::new(|x| x), Box::new(|x| x * x)]],
        _ => {
            const POWERS: [f64; 8] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
            let mut v: Vec<Vec<Box<dyn Fn(f64) -> f64>>> = Vec::new();
            for (i, &p1) in POWERS.iter().enumerate() {
                for &p2 in &POWERS[i..] {
                    let f1: Box<dyn Fn(f64) -> f64> = Box::new(move |x| fp(p1, x + shift));
                    let f2: Box<dyn Fn(f64) -> f64> = if p1 == p2 { Box::new(move |x| fp(p1, x + shift) * (x + shift).ln()) } else { Box::new(move |x| fp(p2, x + shift)) };
                    v.push(vec![f1, f2]);
                }
            }
            v
        }
    };
    let n = pts.len();
    let mut best: Option<(f64, Vec<f64>, Matrix, usize)> = None;
    for (bi, basis) in bases.iter().enumerate() {
        let k = basis.len() + 1;
        let row = |x: f64| -> Vec<f64> { std::iter::once(1.0).chain(basis.iter().map(|f| f(x))).collect() };
        let mut xtx = Matrix::zeros(k, k);
        let mut xty = vec![0.0; k];
        for &(x, y) in pts {
            let r = row(x);
            if r.iter().any(|v| !v.is_finite()) {
                continue;
            }
            for a in 0..k {
                xty[a] += r[a] * y;
                for c in 0..k {
                    xtx[(a, c)] += r[a] * r[c];
                }
            }
        }
        let (xi, dropped) = xtx.sym_inverse(1e-12);
        if !dropped.is_empty() {
            continue;
        }
        let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| xi[(a, c)] * xty[c]).sum()).collect();
        let rss: f64 = pts.iter().map(|&(x, y)| (y - row(x).iter().zip(&b).map(|(r, b)| r * b).sum::<f64>()).powi(2)).sum();
        if best.as_ref().map_or(true, |(r, ..)| rss < *r) {
            best = Some((rss, b, xi, bi));
        }
    }
    let (rss, b, xi, bi) = best.ok_or_else(|| rc_err(430, "could not fit the prediction curve"))?;
    let basis = &bases[bi];
    let k = basis.len() + 1;
    let s2 = rss / (n - k) as f64;
    let grid = (0..100)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / 99.0;
            let r: Vec<f64> = std::iter::once(1.0).chain(basis.iter().map(|f| f(x))).collect();
            let yhat: f64 = r.iter().zip(&b).map(|(r, b)| r * b).sum();
            let var: f64 = (0..k).map(|a| (0..k).map(|c| r[a] * xi[(a, c)] * r[c]).sum::<f64>()).sum::<f64>() * s2;
            (x, yhat, var.max(0.0).sqrt())
        })
        .collect();
    Ok(Fit { grid, df: (n - k) as f64 })
}

/// Marker sizes from [w=] weights, marker labels from mlabel(), hollow symbols.
#[allow(clippy::too_many_arguments)]
fn scatter_extras(ds: &DataSet, st: &Store, spec: &Command, mask: &[bool], y: &str, x: &str, wts: Option<&util::Weights>, l: &mut Layer) -> Result<(), String> {
    let yc = plot_col(ds, st, y)?;
    let xc = plot_col(ds, st, x)?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(xc[i])).collect();
    if let Some(w) = wts {
        let mean = rows.iter().map(|&i| w.w[i]).sum::<f64>() / rows.len().max(1) as f64;
        // marker area proportional to the weight
        l.sizes = rows.iter().map(|&i| (w.w[i] / mean).max(0.0).sqrt()).collect();
    }
    if let Some(v) = spec.opt("mlabel", 4) {
        let name = varlist::resolve_name(ds, v.split(',').next().unwrap_or("").trim())?;
        let var = ds.get_var(&name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
        l.mlabels = rows.iter().map(|&i| util::cell_text(ds, var, i, true)).collect();
    }
    if let Some(m) = spec.opt("msymbol", 2) {
        let m = m.trim().to_lowercase();
        l.hollow = m.ends_with("hollow") || m.ends_with('h') && m.len() <= 2;
    }
    Ok(())
}

/// contour, contourline and heatmap layers from (x, y, z) points, with their colour key.
fn surface_layers(spec: &Command, pts: &[(f64, f64, f64)], zlabel: &str) -> Result<Vec<Layer>, String> {
    let zmin = pts.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
    let zmax = pts.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);
    let levels = util::opt_num(spec, "levels", 3)?.map_or(if spec.verb == "heatmap" { 8 } else { 5 }, |v| v as usize).max(2);
    let cuts: Vec<f64> = match spec.opt("ccuts", 2) {
        Some(c) => {
            let mut v: Vec<f64> = c.split(|ch: char| ch.is_whitespace() || ch == ',').filter_map(|x| x.trim().parse().ok()).collect();
            v.sort_by(f64::total_cmp);
            v
        }
        None => crate::surface::cuts(zmin, zmax, levels),
    };
    let colours = crate::surface::band_colors(cuts.len() + 1);
    let mut l = layer(LayerKind::Surface, Vec::new(), zlabel);
    if spec.verb == "heatmap" {
        // one cell per (x, y), as wide as the spacing of the distinct values
        let spacing = |v: Vec<f64>| -> f64 {
            let mut v = v;
            v.sort_by(f64::total_cmp);
            v.dedup();
            v.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min)
        };
        let dx = spacing(pts.iter().map(|p| p.0).collect());
        let dy = spacing(pts.iter().map(|p| p.1).collect());
        let (dx, dy) = (if dx.is_finite() { dx } else { 1.0 }, if dy.is_finite() { dy } else { 1.0 });
        for &(x, y, z) in pts {
            l.cells.push([x - dx / 2.0, y - dy / 2.0, x + dx / 2.0, y + dy / 2.0]);
            l.cell_colors.push(colours[crate::surface::band(z, &cuts)]);
        }
    } else {
        let method = spec.opt("interp", 3).map(|m| m.trim().to_lowercase()).unwrap_or_else(|| "thinplatespline".into());
        // a complete grid is used as it is; scattered data are interpolated
        let grid = match crate::surface::as_grid(pts) {
            Some(g) => g,
            None if method == "none" => return Err(rc_err(198, "interp(none) requires data on a regular grid")),
            None => crate::surface::interpolate(pts, 60, &method),
        };
        if spec.verb == "contour" {
            for j in 0..grid.ys.len() - 1 {
                for i in 0..grid.xs.len() - 1 {
                    let zc = (grid.z[j][i] + grid.z[j][i + 1] + grid.z[j + 1][i] + grid.z[j + 1][i + 1]) / 4.0;
                    l.cells.push([grid.xs[i], grid.ys[j], grid.xs[i + 1], grid.ys[j + 1]]);
                    l.cell_colors.push(colours[crate::surface::band(zc, &cuts)]);
                }
            }
        } else {
            l.kind = LayerKind::ContourLines;
            let line_colours = crate::surface::band_colors(cuts.len());
            for (k, &c) in cuts.iter().enumerate() {
                for ((x1, y1), (x2, y2)) in crate::surface::contour_segments(&grid, c) {
                    l.segments.push([y1, x1, y2, x2]);
                    l.seg_colors.push(line_colours[k]);
                }
            }
        }
    }
    // colour key: band ranges (or the levels of contour lines)
    let key: Vec<(String, (u8, u8, u8))> = if spec.verb == "contourline" {
        cuts.iter().zip(crate::surface::band_colors(cuts.len())).map(|(c, col)| (fmt_g(*c, 6), col)).collect()
    } else {
        let mut edges = vec![zmin];
        edges.extend(cuts.iter().copied());
        edges.push(zmax);
        edges.windows(2).zip(&colours).map(|(w, col)| (format!("{} - {}", fmt_g(w[0], 6), fmt_g(w[1], 6)), *col)).collect()
    };
    l.key = key;
    Ok(vec![l])
}

/// Immediate plots: `scatteri y x [(#) "label"] ...`, `pci y1 x1 y2 x2 ...`,
/// `pcarrowi ...`: the numbers are typed on the command line.
fn immediate_layers(spec: &Command) -> Result<Vec<Layer>, String> {
    let mut nums: Vec<f64> = Vec::new();
    let mut labels: Vec<(usize, String)> = Vec::new();
    for w in split_top_level_words(&spec.main) {
        if w.starts_with('"') || w.starts_with("`\"") {
            labels.push((nums.len(), util::unquote(&w)));
        } else if w.starts_with('(') {
            // marker-label clock position: accepted, labels are drawn to the right
        } else {
            nums.push(w.parse().map_err(|_| rc_err(198, format!("{w}: number expected")))?);
        }
    }
    let group = if spec.verb == "scatteri" { 2 } else { 4 };
    if nums.is_empty() || nums.len() % group != 0 {
        return Err(rc_err(198, format!("{}: numbers must come in groups of {group}", spec.verb)));
    }
    if spec.verb == "scatteri" {
        let pts: Vec<(f64, f64)> = nums.chunks(2).map(|c| (c[1], c[0])).collect();
        let mut l = layer(LayerKind::Scatter, pts.clone(), "");
        if !labels.is_empty() {
            l.mlabels = vec![String::new(); pts.len()];
            for (after, text) in labels {
                if after >= 2 {
                    l.mlabels[after / 2 - 1] = text;
                }
            }
        }
        return Ok(vec![l]);
    }
    let style = PcStyle::from_verb(spec.verb.as_str()).unwrap_or(PcStyle::Spike);
    let mut l = layer(LayerKind::Pc(style), Vec::new(), "");
    l.segments = nums.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    Ok(vec![l])
}

/// tsline/tsrline as twoway plots: line/rline against the tsset time variable.
fn ts_as_line(st: &Store, spec: &Command) -> Result<Command, String> {
    let (_, t) = st.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset"))?;
    let mut c = spec.clone();
    c.verb = if spec.verb == "tsrline" { "rline".into() } else { "line".into() };
    if let Some(r) = spec.opt("recast", 3) {
        c.verb = r.trim().to_string();
    }
    c.main = format!("{} {t}", spec.main.trim());
    Ok(c)
}

/// twoway function [y =] f(x) [, range(# #|varname) n(#) horizontal droplines(numlist)]
fn function_layers(spec: &Command) -> Result<Vec<Layer>, String> {
    let main = spec.main.trim();
    let text = match main.split_once('=') {
        Some((lhs, rhs)) if !lhs.trim().is_empty() && lhs.trim().chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !rhs.starts_with('=') => rhs.trim(),
        _ => main,
    };
    let expr = ferrostat_parser::parser::parse_expression(text).map_err(|e| rc_err(198, e.message))?;
    let (lo, hi) = match spec.opt("range", 1) {
        Some(r) => {
            let v: Vec<f64> = r.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() != 2 {
                return Err(rc_err(198, "range(# #) required"));
            }
            (v[0], v[1])
        }
        None => (0.0, 1.0),
    };
    let n = util::opt_num(spec, "n", 1)?.map_or(300, |v| v as usize).max(2);
    let xs: Vec<f64> = (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect();
    let eval = |at: &[f64]| -> Result<Vec<f64>, String> {
        let mut tmp = DataSet::new();
        tmp.add_variable(util::new_double("x", at.to_vec()))?;
        tmp.n_obs = at.len();
        util::eval_col(&tmp, &Store::default(), &expr, &vec![true; at.len()])
    };
    let ys = eval(&xs)?;
    let horizontal = spec.has_opt("horizontal", 3);
    let pt = |x: f64, y: f64| if horizontal { (y, x) } else { (x, y) };
    let pts: Vec<(f64, f64)> = xs.iter().zip(&ys).filter(|(_, y)| !is_missing(**y) && y.is_finite()).map(|(&x, &y)| pt(x, y)).collect();
    let mut out = vec![layer(LayerKind::Line, pts, text)];
    if let Some(d) = spec.opt("droplines", 4) {
        let at: Vec<f64> = d.split(|c: char| c.is_whitespace() || c == ',').filter_map(|x| x.trim().parse().ok()).collect();
        let fy = eval(&at)?;
        let mut l = layer(LayerKind::Range(RangeStyle::Spike), Vec::new(), "");
        l.ranges = at.iter().zip(&fy).map(|(&x, &y)| (x, 0.0, y)).collect();
        l.horizontal = horizontal;
        out.push(l);
    }
    Ok(out)
}

/// twoway histogram: bars of a (sorted) sample; bin(#), width(#), start(#),
/// discrete; density (default), fraction, frequency or percent; horizontal; gap(#).
fn twoway_histogram(spec: &Command, xs: &[f64]) -> Result<Layer, String> {
    let n = xs.len() as f64;
    let (lo, hi) = (util::opt_num(spec, "start", 2)?.unwrap_or(xs[0]), xs[xs.len() - 1]);
    let discrete = spec.has_opt("discrete", 1);
    // (bin width, number of bins when fixed by bin())
    let (width, fixed) = if discrete {
        let mut d: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
        d.sort_by(f64::total_cmp);
        (d.first().copied().unwrap_or(1.0), None)
    } else if let Some(w) = util::opt_num(spec, "width", 1)? {
        (w, None)
    } else {
        let bins = util::opt_num(spec, "bin", 3)?.unwrap_or_else(|| n.sqrt().min(10.0 * n.ln() / 10f64.ln()).floor().max(1.0));
        (((hi - lo) / bins).max(1e-12), Some(bins as usize))
    };
    let start = if discrete { lo - width / 2.0 } else { lo };
    // the maximum belongs to the last bin, not to a new one on its upper edge
    let nb = match fixed {
        Some(b) => b.max(1),
        None if discrete => ((hi - start) / width).floor() as usize + 1,
        None => (((hi - start) / width).ceil() as usize).max(1),
    };
    let mut counts = vec![0.0; nb];
    for &x in xs {
        let k = (((x - start) / width).floor().max(0.0) as usize).min(nb - 1);
        counts[k] += 1.0;
    }
    let scale = if spec.has_opt("frequency", 4) { 1.0 } else if spec.has_opt("fraction", 4) { 1.0 / n } else if spec.has_opt("percent", 4) { 100.0 / n } else { 1.0 / (n * width) };
    let gap = util::opt_num(spec, "gap", 3)?.unwrap_or(0.0).clamp(0.0, 99.0);
    let mut l = layer(LayerKind::Range(RangeStyle::Bar), Vec::new(), if spec.has_opt("frequency", 4) { "Frequency" } else if spec.has_opt("fraction", 4) { "Fraction" } else if spec.has_opt("percent", 4) { "Percent" } else { "Density" });
    l.ranges = counts.iter().enumerate().map(|(k, c)| (start + (k as f64 + 0.5) * width, 0.0, c * scale)).collect();
    l.width = width * (1.0 - gap / 100.0);
    l.horizontal = spec.has_opt("horizontal", 3);
    Ok(l)
}

/// twoway kdensity: kernel density line; kernel(), bwidth(), n(), range(# #).
fn twoway_kdensity(spec: &Command, xs: &[f64]) -> Result<Layer, String> {
    let kname = crate::kdensity::canonical_kernel(&spec.opt("kernel", 1).unwrap_or("epanechnikov").trim().to_lowercase());
    let h = util::opt_num(spec, "bwidth", 2)?.unwrap_or_else(|| crate::smooth::rot_bandwidth(xs));
    let n = util::opt_num(spec, "n", 1)?.map_or(300, |v| v as usize).max(2);
    let (lo, hi) = match spec.opt("range", 1).map(|r| r.split_whitespace().filter_map(|x| x.parse::<f64>().ok()).collect::<Vec<_>>()) {
        Some(v) if v.len() == 2 => (v[0], v[1]),
        _ => (xs[0], xs[xs.len() - 1]),
    };
    let m = xs.len() as f64;
    let pts = (0..n)
        .map(|k| {
            let g = lo + (hi - lo) * k as f64 / (n - 1) as f64;
            (g, xs.iter().map(|x| crate::kdensity::kernel(kname, (g - x) / h)).sum::<f64>() / (m * h))
        })
        .collect();
    let mut l = layer(LayerKind::Line, pts, "Density");
    l.horizontal = spec.has_opt("horizontal", 3);
    Ok(l)
}

pub struct Twoway;

impl CommandHandler for Twoway {
    fn name(&self) -> &'static str {
        "twoway"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // if/in written after all the plots apply to every plot
        let mut common = String::new();
        let specs: Vec<String> = if cmd.verb == "twoway" || cmd.verb == "tw" {
            let main = cmd.raw.trim();
            let body = main.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("");
            // (plot) (plot) or plot || plot
            if body.trim_start().starts_with('(') {
                let (plots, _) = match ferrostat_parser::parser::find_top_level(body, ',') {
                    Some(i) => (&body[..i], &body[i..]),
                    None => (body, ""),
                };
                let (list, rest) = split_paren_plots(plots);
                common = rest;
                list
            } else {
                // a final `|| , options` segment holds graph-wide options only,
                // a final `|| if exp in range` segment restricts every plot
                let mut list: Vec<String> = body.split("||").map(|s| s.trim().to_string()).filter(|s| !s.is_empty() && !s.starts_with(',')).collect();
                if list.last().is_some_and(|l| l.starts_with("if ") || l.starts_with("in ")) {
                    let last = list.pop().unwrap();
                    common = match ferrostat_parser::parser::find_top_level(&last, ',') {
                        Some(i) => last[..i].to_string(),
                        None => last,
                    };
                }
                list
            }
        } else {
            vec![cmd.raw.clone()]
        };
        let mut specs: Vec<Command> = specs.iter().map(|s| parse_line(s).map_err(|e| rc_err(198, e))).collect::<Result<Vec<_>, _>>()?.into_iter().filter(|c| !c.verb.is_empty()).collect();
        for c in specs.iter_mut() {
            if let Some(v) = plot_type(&c.verb) {
                c.verb = v.to_string();
            }
        }
        let common_mask = if common.trim().is_empty() {
            None
        } else {
            let c = parse_line(&format!("twoway {}", common.trim())).map_err(|e| rc_err(198, e))?;
            if !c.main.trim().is_empty() {
                return Err(rc_err(198, format!("{}: invalid plot specification", c.main.trim())));
            }
            Some(util::touse(ds, st, &c)?)
        };
        let both = |w: &[bool]| -> Vec<bool> {
            match &common_mask {
                Some(m) => m.iter().zip(w).map(|(a, b)| *a && *b).collect(),
                None => w.to_vec(),
            }
        };
        // graph-wide options may also be written inside a plot: legend()
        // contents are merged, the others apply unless given after the comma
        let mut merged = cmd.clone();
        for spec in specs.iter().filter(|_| cmd.verb == "twoway" || cmd.verb == "tw") {
            for (name, arg) in &spec.opts {
                let n = name.to_lowercase();
                let is = |full: &str, min: usize| n.len() >= min && full.starts_with(n.as_str());
                if is("legend", 3) {
                    let extra = arg.clone().unwrap_or_default();
                    match merged.opts.iter_mut().find(|(m, _)| { let m = m.to_lowercase(); m.len() >= 3 && "legend".starts_with(m.as_str()) }) {
                        Some((_, Some(a))) => *a = format!("{a} {extra}"),
                        Some(_) => {}
                        None => merged.opts.push((name.clone(), Some(extra))),
                    }
                } else if (is("title", 1) && n.len() >= 2 || is("xtitle", 2) || is("ytitle", 2) || is("yline", 5) || is("xline", 5) || is("name", 4) || is("scheme", 3)) && !merged.opts.iter().any(|(m, _)| m.eq_ignore_ascii_case(name)) {
                    merged.opts.push((name.clone(), arg.clone()));
                }
            }
        }
        let cmd = &merged;
        let Some(by) = cmd.opt("by", 2) else {
            let mut graph = twoway_graph(ds, st, &specs, common_mask.as_deref())?;
            return finish(&mut graph, cmd, st);
        };
        // by(var [, total rows(#) cols(#)]): one panel per group
        let (var, opts) = match by.split_once(',') {
            Some((v, o)) => (v.trim().to_string(), o.to_string()),
            None => (by.trim().to_string(), String::new()),
        };
        let var = varlist::resolve_name(ds, &var)?;
        let byopt = parse_line(&format!("by ,{opts}")).map_err(|e| rc_err(198, e))?;
        let col = util::col(ds, &var)?;
        let v = ds.get_var(&var).unwrap();
        let mut levels: Vec<(f64, usize)> = Vec::new();
        for i in 0..ds.n_obs {
            if !is_missing(col[i]) && !levels.iter().any(|(x, _)| *x == col[i]) {
                levels.push((col[i], i));
            }
        }
        levels.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut children = Vec::new();
        for (val, row) in &levels {
            let within: Vec<bool> = both(&col.iter().map(|c| c == val).collect::<Vec<_>>());
            // groups left without observations by if/in get no panel
            if common_mask.is_some() && !within.iter().any(|b| *b) {
                continue;
            }
            let mut g = twoway_graph(ds, st, &specs, Some(&within))?;
            g.title = util::cell_text(ds, v, *row, true);
            children.push(g);
        }
        if byopt.has_opt("total", 3) {
            let mut g = twoway_graph(ds, st, &specs, common_mask.as_deref())?;
            g.title = "Total".into();
            children.push(g);
        }
        let n = children.len();
        let cols = if let Some(r) = util::opt_num(&byopt, "rows", 3)? {
            n.div_ceil(r.max(1.0) as usize)
        } else if let Some(c) = util::opt_num(&byopt, "cols", 3)? {
            c.max(1.0) as usize
        } else {
            (n as f64).sqrt().ceil() as usize
        };
        let mut graph = Graph { children, cols: cols.max(1), ..Default::default() };
        finish(&mut graph, cmd, st)
    }
}

/// Split `(plot) (plot) ... [if exp] [in range]` into the plots and the
/// trailing text after the last parenthesized plot.
fn split_paren_plots(s: &str) -> (Vec<String>, String) {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut plots = Vec::new();
    let mut k = 0;
    while k < chars.len() {
        let (i, c) = chars[k];
        if c.is_whitespace() {
            k += 1;
            continue;
        }
        if c != '(' {
            return (plots, s[i..].to_string());
        }
        let (mut depth, mut quote) = (0usize, false);
        let mut end = s.len();
        while k < chars.len() {
            let (j, ch) = chars[k];
            k += 1;
            match ch {
                '"' => quote = !quote,
                '(' if !quote => depth += 1,
                ')' if !quote => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        plots.push(s[i + 1..end].trim().to_string());
    }
    (plots, String::new())
}

/// Full plot-type name for an abbreviation used inside twoway
/// (`sc`, `scat`, `li`, `conn`, `hist`, `kdens`, ...).
fn plot_type(v: &str) -> Option<&'static str> {
    const TYPES: [(&str, usize); 6] = [("scatter", 2), ("line", 2), ("connected", 4), ("histogram", 4), ("kdensity", 5), ("lowess", 4)];
    TYPES.iter().find(|(n, min)| v.len() >= *min && n.starts_with(v)).map(|(n, _)| *n)
}

/// One twoway graph from its plot specifications.
fn twoway_graph(ds: &DataSet, st: &Store, specs: &[Command], within: Option<&[bool]>) -> Result<Graph, String> {
    let mut graph = Graph { legend: true, ..Default::default() };
    let mut yformats: Vec<String> = Vec::new();
    for c in specs {
        let ts;
        let c = if matches!(c.verb.as_str(), "tsline" | "tsrline") {
            ts = ts_as_line(st, c)?;
            &ts
        } else {
            c
        };
        graph.layers.extend(plot_layers(ds, st, c, within)?);
        if matches!(c.verb.as_str(), "function" | "scatteri" | "pci" | "pcarrowi" | "pcbarrowi") {
            if graph.xtitle.is_empty() && c.verb == "function" {
                graph.xtitle = "x".into();
            }
            continue;
        }
        let vars = plot_varlist(ds, &c.main)?;
        if matches!(c.verb.as_str(), "histogram" | "kdensity") {
            if let Some(v) = vars.first() {
                if graph.xtitle.is_empty() {
                    graph.xtitle = ds.get_var(v).map(|x| if x.label.is_empty() { v.clone() } else { x.label.clone() }).unwrap_or_else(|| v.clone());
                    graph.xformat = ds.get_var(v).map(|x| x.format.clone()).unwrap_or_default();
                }
            }
            continue;
        }
        if graph.xtitle.is_empty() {
            // paired coordinates: y1 x1 y2 x2 -> the x axis is x1
            let xvar = if PcStyle::from_verb(c.verb.as_str()).is_some() { vars.get(1) } else { vars.last() };
            if let Some(x) = xvar {
                graph.xtitle = ds.get_var(x).map(|v| if v.label.is_empty() { x.clone() } else { v.label.clone() }).unwrap_or_default();
                graph.xformat = ds.get_var(x).map(|v| v.format.clone()).unwrap_or_default();
            }
        }
        if vars.len() >= 2 {
            yformats.extend(vars[..vars.len() - 1].iter().map(|y| ds.get_var(y).map(|v| v.format.clone()).unwrap_or_default()));
        }
    }
    if graph.layers.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    if let Some(l) = graph.layers.iter().find(|l| !l.key.is_empty()) {
        graph.color_legend = l.key.clone();
    }
    // the y axis uses the y variables' format only when they all share it
    if let Some(first) = yformats.first() {
        if yformats.iter().all(|f| f == first) {
            graph.yformat = first.clone();
        }
    }
    graph.legend = graph.layers.len() > 1;
    if graph.layers.len() == 1 {
        graph.ytitle = graph.layers[0].label.clone();
    }
    // surfaces (z y x): the y axis is the y variable, not z
    if let Some(c) = specs.iter().find(|c| matches!(c.verb.as_str(), "contour" | "contourline" | "heatmap")) {
        if let Some(y) = plot_varlist(ds, &c.main)?.get(1) {
            graph.ytitle = ds.get_var(y).map(|v| if v.label.is_empty() { y.clone() } else { v.label.clone() }).unwrap_or_else(|| y.clone());
        }
    }
    if graph.layers.iter().all(|l| l.horizontal) {
        std::mem::swap(&mut graph.xtitle, &mut graph.ytitle);
        std::mem::swap(&mut graph.xformat, &mut graph.yformat);
    }
    Ok(graph)
}

/// `tsline y1 [y2 ...]` and `tsrline y1 y2`: line and range-line plots
/// against the time variable set by tsset; `recast()` changes the plot type.
pub struct TsLine;

impl CommandHandler for TsLine {
    fn name(&self) -> &'static str {
        "tsline"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (_, t) = st.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset"))?;
        let ranged = cmd.verb.starts_with("tsr");
        let plot = cmd.opt("recast", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| if ranged { "rline".into() } else { "line".into() });
        let raw = cmd.raw.trim();
        let rest = raw.split_once(char::is_whitespace).map_or("", |x| x.1);
        // insert the time variable after the y variables (before if/in/options)
        let cut = [" if ", " in ", ","].iter().filter_map(|k| rest.find(k)).min().unwrap_or(rest.len());
        let line = format!("{plot} {} {t} {}", &rest[..cut], &rest[cut..]);
        let c = parse_line(&line).map_err(|e| rc_err(198, e))?;
        Twoway.execute(ds, st, &c)
    }
}

/// Print a graph through the system spooler: `lp` or `lpr`, or the command
/// set with `graph set print command "..."` (the PDF file is appended).
#[cfg(not(target_arch = "wasm32"))]
fn print_graph(g: &Graph, printcolor: &str, command: Option<&String>) -> Result<String, String> {
    let file = ferrostat_core::vfs::temp_path(&format!("ferrostat_print_{}.pdf", std::process::id()));
    render_with(g, &file, 1100, 850, printcolor)?;
    let candidates: Vec<String> = match command {
        Some(c) => vec![c.clone()],
        None => vec!["lp".into(), "lpr".into()],
    };
    for c in candidates {
        let mut parts = c.split_whitespace();
        let Some(prog) = parts.next() else { continue };
        if let Ok(status) = std::process::Command::new(prog).args(parts).arg(&file).status() {
            let _ = std::fs::remove_file(&file);
            return if status.success() { Ok(format!("(graph {} sent to the printer)", g.name)) } else { Err(rc_err(691, "printer error")) };
        }
    }
    let _ = std::fs::remove_file(&file);
    Err(rc_err(691, "no printer available (lp or lpr not found)"))
}

#[cfg(target_arch = "wasm32")]
fn print_graph(_g: &Graph, _printcolor: &str, _command: Option<&String>) -> Result<String, String> {
    Err(rc_err(691, "printing is not available in the browser; use graph export"))
}

/// `palette color c1 [c2]`, `palette linepalette`, `palette symbolpalette`:
/// draw samples of colours, line patterns or marker symbols.
pub struct Palette;

impl CommandHandler for Palette {
    fn name(&self) -> &'static str {
        "palette"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words = split_top_level_words(&cmd.main);
        let sub = words.first().map(|s| s.to_lowercase()).unwrap_or_default();
        let mut g = Graph::default();
        let mut text = Vec::new();
        match sub.as_str() {
            "color" | "colors" => {
                let specs: Vec<String> = words[1..].iter().map(|w| util::unquote(w)).collect();
                if specs.is_empty() || specs.len() > 2 {
                    return Err(rc_err(198, "palette color colorstyle [colorstyle]"));
                }
                for (k, s) in specs.iter().enumerate() {
                    let rgb = crate::colors::parse(s).ok_or_else(|| rc_err(198, format!("{s}: invalid color")))?;
                    let mut l = layer(LayerKind::Range(RangeStyle::Bar), Vec::new(), s);
                    l.ranges = vec![(k as f64 + 0.5, 0.0, 1.0)];
                    l.width = 0.8;
                    l.color = Some(rgb);
                    g.layers.push(l);
                    g.xlabels.push((k as f64 + 0.5, format!("{s} = {} {} {}", rgb.0, rgb.1, rgb.2)));
                    g.ylabels = vec![(0.5, String::new())];
                    text.push(format!("{s}: {} {} {}", rgb.0, rgb.1, rgb.2));
                }
                g.title = "Color palette".into();
            }
            "linepalette" => {
                let pats = ["solid", "dash", "dot", "shortdash", "longdash", "dash_dot", "blank"];
                for (k, p) in pats.iter().enumerate() {
                    let y = (pats.len() - k) as f64;
                    let mut l = layer(LayerKind::Line, vec![(0.0, y), (1.0, y)], p);
                    l.pattern = p.to_string();
                    l.color = Some((0, 0, 0));
                    g.layers.push(l);
                    g.ylabels.push((y, p.to_string()));
                }
                g.title = "Line patterns".into();
            }
            "symbolpalette" => {
                let syms = ["circle", "diamond", "triangle", "square", "plus", "x", "smcircle", "smdiamond", "smsquare", "smtriangle", "circle_hollow", "diamond_hollow", "triangle_hollow", "square_hollow", "point"];
                for (k, s) in syms.iter().enumerate() {
                    let (x, y) = ((k % 5) as f64, (k / 5) as f64 * -1.0);
                    let mut l = layer(LayerKind::Scatter, vec![(x, y)], s);
                    l.symbol = s.to_string();
                    l.color = Some((0, 0, 0));
                    l.mlabels = vec![format!("  {s}")];
                    l.sizes = vec![2.0];
                    g.layers.push(l);
                }
                g.title = "Marker symbols".into();
                // no numeric axes: blank category labels on both
                g.xlabels = (0..5).map(|k| (k as f64, String::new())).collect();
                g.ylabels = (0..3).map(|k| (-(k as f64), String::new())).collect();
            }
            _ => return Err(rc_err(198, "palette color|linepalette|symbolpalette")),
        }
        g.legend = false;
        let extra = finish(&mut g, cmd, st)?;
        Ok([text.join("\n"), extra].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"))
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
            // default number of bins: min(sqrt(N), 10 ln(N)/ln(10))
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
        if cmd.has_opt("kdensity", 4) {
            // Epanechnikov kernel, the same default bandwidth rule as kdensity
            let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
            let m = pts.iter().map(|p| p.0 * p.1).sum::<f64>() / n;
            let sd = (pts.iter().map(|p| p.1 * (p.0 - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            let mut sorted = xs.clone();
            sorted.sort_by(f64::total_cmp);
            let iqr = util::pctile_sorted(&sorted, 75.0) - util::pctile_sorted(&sorted, 25.0);
            let h = 0.9 * if iqr > 0.0 { sd.min(iqr / 1.349) } else { sd } * n.powf(-0.2);
            let unit = if cmd.has_opt("frequency", 4) { n * width } else if cmd.has_opt("percent", 4) { 100.0 * width } else if cmd.has_opt("fraction", 4) { width } else { 1.0 };
            let curve: Vec<(f64, f64)> = (0..=100).map(|k| {
                let xv = start + (bins as f64 * width) * k as f64 / 100.0;
                let d = pts.iter().map(|(xi, wi)| wi * crate::kdensity::kernel("epanechnikov", (xv - xi) / h)).sum::<f64>() / (n * h);
                (xv, d * unit)
            }).collect();
            graph.layers.push(layer(LayerKind::Line, curve, "Kernel density estimate"));
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
        // the subcommand ends at the first non-letter (`graph query, schemes`)
        let sub_len = body.find(|c: char| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(body.len());
        let sub = body[..sub_len].to_lowercase();
        let rest = body[sub_len..].trim().to_string();
        match sub.as_str() {
            "export" => {
                let c = parse_line(&format!("export {rest}")).map_err(|e| rc_err(198, e))?;
                let file = util::unquote(c.main.trim());
                let g = match c.opt("name", 2) {
                    Some(n) => st.graphs.get(n.trim()).cloned(),
                    None => st.graph.clone(),
                }
                .ok_or_else(|| rc_err(693, "no graph in memory"))?;
                if ferrostat_core::vfs::exists(&file) && !c.has_opt("replace", 3) {
                    return Err(rc_err(602, format!("file {file} already exists")));
                }
                let width = util::opt_num(&c, "width", 2)?.unwrap_or(800.0) as u32;
                let height = util::opt_num(&c, "height", 2)?.unwrap_or(width as f64 * 0.75) as u32;
                render_with(&g, &file, width, height, &st.printcolor)?;
                if file.to_lowercase().ends_with(".svg") {
                    if let Some(face) = st.graph_settings.get(&("svg".to_string(), "fontface".to_string())) {
                        let svg = ferrostat_core::vfs::read_to_string(&file).map_err(|e| rc_err(603, e))?;
                        ferrostat_core::vfs::write(&file, svg.replace("font-family=\"sans-serif\"", &format!("font-family=\"{face}\"")).as_bytes()).map_err(|e| rc_err(603, e))?;
                    }
                }
                let lower = file.to_lowercase();
                let kind = if lower.ends_with(".svg") { "SVG" } else if lower.ends_with(".pdf") { "PDF" } else if lower.ends_with(".gph") { "gph" } else if (lower.ends_with(".eps") || lower.ends_with(".ps")) && cfg!(not(target_arch = "wasm32")) { "EPS" } else { "PNG" };
                Ok(format!("file {file} written in {kind} format"))
            }
            "twoway" | "tw" => {
                let c = parse_line(&format!("twoway {rest}")).map_err(|e| rc_err(198, e))?;
                Twoway.execute(ds, st, &c)
            }
            "matrix" => graph_matrix(ds, st, &rest),
            "bar" | "hbar" | "dot" => graph_bar(ds, st, &sub, &rest),
            "pie" => graph_pie(ds, st, &rest),
            "box" | "hbox" => graph_box(ds, st, &if sub == "hbox" { format!("{rest} horizontal") } else { rest.clone() }),
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
            "display" | "replay" => {
                // graph display [name] [, scheme()]; graph replay name|file ...
                let c = parse_line(&format!("display {rest}")).map_err(|e| rc_err(198, e))?;
                let mut names: Vec<String> = c.main.split_whitespace().map(util::unquote).collect();
                if names.iter().any(|n| n == "_all") {
                    names = st.graphs.keys().cloned().collect();
                }
                if names.is_empty() {
                    names.push(st.graph.as_ref().map(|g| g.name.clone()).ok_or_else(|| rc_err(693, "no graph in memory"))?);
                }
                let mut shown = Vec::new();
                for n in names {
                    let mut g = match st.graphs.get(&n) {
                        Some(g) => g.clone(),
                        None if sub == "replay" || n.ends_with(".gph") => load_gph(&if n.contains('.') { n.clone() } else { format!("{n}.gph") })?,
                        None => return Err(rc_err(111, format!("graph {n} not found"))),
                    };
                    if let Some(s) = c.opt("scheme", 3).map(|s| s.trim().to_lowercase()).filter(|s| is_scheme(s)) {
                        g.scheme = s;
                    }
                    shown.push(g.name.clone());
                    st.graph = Some(g);
                }
                st.graph_window_closed = false;
                Ok(if sub == "replay" { shown.iter().map(|n| format!("(graph {n} displayed)")).collect::<Vec<_>>().join("\n") } else { String::new() })
            }
            "close" => {
                st.graph_window_closed = true;
                Ok(String::new())
            }
            "copy" | "rename" => {
                let c = parse_line(&format!("{sub} {rest}")).map_err(|e| rc_err(198, e))?;
                let w: Vec<String> = c.main.split_whitespace().map(String::from).collect();
                let (from, to) = match (sub.as_str(), w.as_slice()) {
                    ("copy", [to]) => (st.graph.as_ref().map(|g| g.name.clone()).ok_or_else(|| rc_err(693, "no graph in memory"))?, to.clone()),
                    (_, [from, to]) => (from.clone(), to.clone()),
                    _ => return Err(rc_err(198, format!("graph {sub} oldname newname"))),
                };
                if st.graphs.contains_key(&to) && !c.has_opt("replace", 3) {
                    return Err(rc_err(110, format!("graph {to} already exists")));
                }
                let mut g = st.graphs.get(&from).cloned().or_else(|| st.graph.clone().filter(|g| g.name == from)).ok_or_else(|| rc_err(111, format!("graph {from} not found")))?;
                if sub == "rename" {
                    st.graphs.remove(&from);
                }
                g.name = to.clone();
                st.graphs.insert(to.clone(), g.clone());
                if sub == "rename" && st.graph.as_ref().is_some_and(|cur| cur.name == from) {
                    st.graph = Some(g);
                }
                Ok(String::new())
            }
            "describe" => {
                let target = util::unquote(rest.split(',').next().unwrap_or("").trim());
                let (g, source) = if target.is_empty() {
                    (st.graph.clone().ok_or_else(|| rc_err(693, "no graph in memory"))?, "memory".to_string())
                } else if let Some(g) = st.graphs.get(&target) {
                    (g.clone(), "memory".to_string())
                } else {
                    let f = if target.contains('.') { target.clone() } else { format!("{target}.gph") };
                    (load_gph(&f)?, f)
                };
                let kinds: Vec<String> = g.layers.iter().map(|l| format!("{:?}", l.kind).to_lowercase()).collect();
                Ok([
                    format!("{}:", if source == "memory" { format!("{} stored in memory", g.name) } else { source.clone() }),
                    String::new(),
                    format!("        name:  {}", g.name),
                    format!("     command:  {}", g.command),
                    format!("     created:  {}", g.created),
                    format!("      scheme:  {}", if g.scheme.is_empty() { "okabeito" } else { &g.scheme }),
                    format!("       title:  {}", g.title),
                    format!("      layers:  {}", if g.children.is_empty() { kinds.join(" ") } else { format!("{} combined graphs", g.children.len()) }),
                ]
                .join("\n"))
            }
            "query" => {
                let what = rest.trim_start_matches(',').trim().to_lowercase();
                let out = match what.as_str() {
                    "schemes" | "scheme" | "schemestyle" => "okabeito (default; also okabe_ito, colorblind)\nclassic\nmono (also gray, grey)".to_string(),
                    "colorstyle" | "color" | "colors" => crate::colors::NAMED.iter().map(|(n, (r, g, b))| format!("{n:<16} {r:>3} {g:>3} {b:>3}")).chain(std::iter::once("gs0 ... gs16    gray scale".to_string())).collect::<Vec<_>>().join("\n"),
                    "symbolstyle" | "symbol" => "circle O, diamond D, triangle T, square S, plus +, x X, point p, none i;\nsmall versions: smcircle o, smdiamond d, smtriangle t, smsquare s, smplus, smx x;\nhollow versions: circle_hollow Oh, diamond_hollow Dh, triangle_hollow Th, square_hollow Sh".to_string(),
                    "linepatternstyle" | "linepattern" => "solid, dash, dot, shortdash, longdash, dash_dot, blank".to_string(),
                    _ => "Styles that can be queried (graph query <style>):\n  schemes  colorstyle  symbolstyle  linepatternstyle".to_string(),
                };
                Ok(out)
            }
            "set" => {
                // graph set [group [setting [value]]]
                let w = split_top_level_words(&rest);
                match w.as_slice() {
                    [group, setting, value, ..] => {
                        st.graph_settings.insert((group.to_lowercase(), setting.to_lowercase()), util::unquote(value));
                        Ok(String::new())
                    }
                    _ => {
                        let group = w.first().map(|s| s.to_lowercase());
                        let rows: Vec<String> = st.graph_settings.iter().filter(|((gname, _), _)| group.as_ref().is_none_or(|x| x == gname)).map(|((gname, s), v)| format!("  {gname:<8} {s:<16} {v}")).collect();
                        Ok(if rows.is_empty() { "(no graph settings changed from their defaults)".to_string() } else { rows.join("\n") })
                    }
                }
            }
            "print" => {
                let c = parse_line(&format!("print {rest}")).map_err(|e| rc_err(198, e))?;
                let g = match c.opt("name", 2) {
                    Some(n) => st.graphs.get(n.trim()).cloned(),
                    None => st.graph.clone(),
                }
                .ok_or_else(|| rc_err(693, "no graph in memory"))?;
                print_graph(&g, &st.printcolor, st.graph_settings.get(&("print".to_string(), "command".to_string())))
            }
            "play" => {
                let file = util::unquote(rest.split(',').next().unwrap_or("").trim());
                let file = if file.contains('.') { file } else { format!("{file}.grec") };
                let text = ferrostat_core::vfs::read_to_string(&file).map_err(|_| rc_err(601, format!("file {file} not found")))?;
                let mut g = st.graph.clone().ok_or_else(|| rc_err(693, "no graph in memory"))?;
                // a recording holds one graph option per line, e.g. title("Sales") or legend(off)
                let opts: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('*') && !l.starts_with("//")).map(String::from).collect();
                let c = parse_line(&format!("play , {}", opts.join(" "))).map_err(|e| rc_err(198, e))?;
                let name = g.name.clone();
                let msg = finish(&mut g, &c, st)?;
                if let Some(cur) = st.graph.as_mut() {
                    cur.name = name.clone();
                }
                if let Some(saved) = st.graphs.remove("Graph").filter(|_| name != "Graph") {
                    st.graphs.insert(name.clone(), saved);
                }
                Ok(msg)
            }
            "save" => {
                let c = parse_line(&format!("save {rest}")).map_err(|e| rc_err(198, e))?;
                let words = split_top_level_words(c.main.trim());
                let (name, file) = match words.as_slice() {
                    [f] => (None, util::unquote(f)),
                    [n, f] => (Some(n.clone()), util::unquote(f)),
                    _ => return Err(rc_err(198, "graph save [graphname] filename")),
                };
                let file = if file.contains('.') { file } else { format!("{file}.gph") };
                if ferrostat_core::vfs::exists(&file) && !c.has_opt("replace", 3) {
                    return Err(rc_err(602, format!("file {file} already exists")));
                }
                let g = match name {
                    Some(n) => st.graphs.get(&n).cloned(),
                    None => st.graph.clone(),
                }
                .ok_or_else(|| rc_err(693, "no graph in memory"))?;
                save_gph(&g, &file)?;
                Ok(format!("file {file} saved"))
            }
            "use" => {
                let file = util::unquote(rest.split(',').next().unwrap_or(""));
                let g = load_gph(&file)?;
                st.graphs.insert(g.name.clone(), g.clone());
                st.graph = Some(g);
                Ok(String::new())
            }
            "combine" => {
                let c = parse_line(&format!("combine {rest}")).map_err(|e| rc_err(198, e))?;
                let mut children = Vec::new();
                for n in c.main.split_whitespace() {
                    let g = st.graphs.get(n).cloned().or_else(|| load_gph(&util::unquote(n)).ok()).ok_or_else(|| rc_err(111, format!("graph {n} not found")))?;
                    children.push(g);
                }
                if children.is_empty() {
                    return Err(rc_err(198, "graph combine requires graph names"));
                }
                let cols = util::opt_num(&c, "cols", 3)?.map(|v| v as usize).unwrap_or_else(|| (children.len() as f64).sqrt().ceil() as usize).max(1);
                let mut g = Graph { children, cols, ..Default::default() };
                finish(&mut g, &c, st)
            }
            _ => Err(rc_err(198, format!("graph {sub}: invalid subcommand"))),
        }
    }
}

/// Groups of `over(var)` (label, membership mask), or one unnamed group.
/// A second over() nests the first within its groups (labels "inner\nouter").
fn over_groups(ds: &DataSet, c: &Command, mask: &[bool]) -> Result<Vec<(String, Vec<bool>)>, String> {
    let mut overs: Vec<String> = Vec::new();
    for (name, arg) in &c.opts {
        let n = name.to_lowercase();
        if n.len() >= 2 && "over".starts_with(n.as_str()) {
            if let Some(a) = arg {
                overs.push(varlist::resolve_name(ds, a.split(',').next().unwrap().trim())?);
            }
        }
    }
    if overs.is_empty() {
        return Ok(vec![(String::new(), mask.to_vec())]);
    }
    // outermost first for grouping
    overs.reverse();
    let (ids, reps) = util::group_ids(ds, &overs, mask);
    let vars: Vec<_> = overs.iter().map(|o| ds.get_var(o).unwrap()).collect();
    Ok(reps
        .iter()
        .enumerate()
        .map(|(g, &r)| {
            let label = vars.iter().rev().map(|v| util::cell_text(ds, v, r, true)).collect::<Vec<_>>().join("\n");
            (label, (0..ds.n_obs).map(|i| ids[i] == g).collect())
        })
        .collect())
}

/// graph bar, graph hbar and graph dot: a statistic (default mean) of each
/// variable within the groups of over().
fn graph_bar(ds: &DataSet, st: &mut Store, kind: &str, rest: &str) -> Result<String, String> {
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
    let groups = over_groups(ds, &c, &mask)?;
    let value = |v: &str, m: &[bool]| -> f64 {
        let x: Vec<f64> = util::rows(m).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
        match stat.as_str() {
            "sum" => x.iter().sum(),
            "count" => x.len() as f64,
            "max" => x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "min" => x.iter().cloned().fold(f64::INFINITY, f64::min),
            "median" | "p50" => {
                let mut s = x.clone();
                s.sort_by(f64::total_cmp);
                util::pctile_sorted(&s, 50.0)
            }
            _ => util::mean(&x),
        }
    };
    let nv = vars.len().max(1);
    let ng = groups.len();
    let stack = c.has_opt("stack", 3);
    let blabel = c.opt("blabel", 2).map(|b| b.split(',').next().unwrap_or("").trim().to_lowercase()).filter(|b| b != "none");
    let mut graph = Graph::default();
    // category labels: groups, or the variables themselves when there is no over()
    // the category axis has no numeric ticks, even with a single unnamed group
    let labels: Vec<(f64, String)> = groups.iter().enumerate().map(|(g, (n, _))| (g as f64 + 0.5, n.clone())).collect();
    if kind == "dot" {
        // categories down the vertical axis (first at the top), one marker per variable
        let pos = |g: usize| (ng - g) as f64 - 0.5;
        for v in &vars {
            let pts: Vec<(f64, f64)> = groups.iter().enumerate().map(|(g, (_, m))| (value(v, m), pos(g))).collect();
            graph.layers.push(layer(LayerKind::Scatter, pts, &format!("{stat} of {v}")));
        }
        graph.ylabels = groups.iter().enumerate().map(|(g, (n, _))| (pos(g), n.replace('\n', ", "))).collect();
        graph.guides = (0..ng).map(pos).collect();
    } else {
        let horizontal = kind == "hbar";
        let mut cum = vec![0.0; ng];
        for (vi, v) in vars.iter().enumerate() {
            let mut l = layer(LayerKind::Range(RangeStyle::Bar), Vec::new(), &format!("{stat} of {v}"));
            l.horizontal = horizontal;
            l.width = if stack { 0.7 } else { 0.8 / nv as f64 };
            for (g, (_, m)) in groups.iter().enumerate() {
                let val = value(v, m);
                let (x, y1, y2) = if stack {
                    let r = (g as f64 + 0.5, cum[g], cum[g] + val);
                    cum[g] += val;
                    r
                } else {
                    (g as f64 + (vi as f64 + 0.5) / nv as f64 * 0.8 + 0.1, 0.0, val)
                };
                // hbar: the first category at the top
                let x = if horizontal { ng as f64 - x } else { x };
                l.ranges.push((x, y1, y2));
                if let Some(b) = &blabel {
                    l.mlabels.push(match b.as_str() {
                        "name" => v.clone(),
                        _ => fmt_g(val, 9),
                    });
                }
            }
            graph.layers.push(l);
        }
        if horizontal {
            graph.ylabels = labels.iter().map(|(p, n)| (ng as f64 - p, n.replace('\n', ", "))).collect();
        } else {
            // nested over(): inner labels under the bars, each outer label once below its block
            let mut outer: Vec<(f64, f64, String)> = Vec::new();
            for (p, n) in &labels {
                let Some((inner, o)) = n.split_once('\n') else {
                    graph.xlabels.push((*p, n.clone()));
                    continue;
                };
                graph.xlabels.push((*p, inner.to_string()));
                match outer.last_mut() {
                    Some(last) if last.2 == o => last.1 = *p,
                    _ => outer.push((*p, *p, o.to_string())),
                }
            }
            graph.xlabels.extend(outer.into_iter().map(|(a, b, o)| ((a + b) / 2.0, format!("\n{o}"))));
        }
    }
    graph.legend = vars.len() > 1;
    finish(&mut graph, &c, st)
}

/// graph pie: slices are the sums of the variables, or of one variable
/// within the groups of over(). pie(#, explode) pulls a slice out;
/// plabel(#|_all percent|name|sum) writes inside slices.
fn graph_pie(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let c = parse_line(&format!("pie {rest}")).map_err(|e| rc_err(198, e))?;
    let mask = util::touse(ds, st, &c)?;
    let vars = varlist::expand(ds, &c.main)?;
    let sum = |v: &str, m: &[bool]| -> f64 { util::rows(m).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).sum() };
    let mut slices: Vec<PieSlice> = if c.opt("over", 2).is_some() {
        // no variable: slices count the observations in each group
        over_groups(ds, &c, &mask)?
            .into_iter()
            .map(|(label, m)| {
                let label = label.replace('\n', ", ");
                let value = match vars.first() {
                    Some(v) => sum(v, &m),
                    None => m.iter().filter(|b| **b).count() as f64,
                };
                PieSlice { label, value, explode: false, text: String::new() }
            })
            .collect()
    } else {
        if vars.is_empty() {
            return Err(rc_err(102, "too few variables specified"));
        }
        let lab = |v: &str| ds.get_var(v).map(|x| if x.label.is_empty() { v.to_string() } else { x.label.clone() }).unwrap_or_else(|| v.to_string());
        vars.iter().map(|v| PieSlice { label: lab(v), value: sum(v, &mask), explode: false, text: String::new() }).collect()
    };
    // sort: slices from smallest to largest (descending reverses)
    if c.has_opt("sort", 4) {
        slices.sort_by(|a, b| a.value.total_cmp(&b.value));
        if c.has_opt("descending", 4) {
            slices.reverse();
        }
    }
    let total: f64 = slices.iter().map(|s| s.value).sum();
    for (name, arg) in &c.opts {
        let arg = arg.clone().unwrap_or_default();
        let (which, what) = match arg.split_once([',', ' ']) {
            Some((a, b)) => (a.trim().to_string(), b.trim().to_lowercase()),
            None => (arg.trim().to_string(), String::new()),
        };
        let targets: Vec<usize> = if which == "_all" { (0..slices.len()).collect() } else { which.parse::<usize>().ok().filter(|k| *k >= 1 && *k <= slices.len()).map(|k| vec![k - 1]).unwrap_or_default() };
        let name = name.to_lowercase();
        for k in targets {
            if name == "pie" && what.starts_with("expl") {
                slices[k].explode = true;
            } else if name.starts_with("plabel") || name.starts_with("plab") {
                let s = &slices[k];
                slices[k].text = match what.split_whitespace().next().unwrap_or("percent") {
                    "name" => s.label.clone(),
                    "sum" => fmt_g(s.value, 9),
                    _ => format!("{}%", fmt_g((s.value / total * 1000.0).round() / 10.0, 9)),
                };
            }
        }
    }
    let angle0 = util::opt_num(&c, "angle0", 6)?;
    let mut graph = Graph { pie: std::mem::take(&mut slices), legend: true, pie_angle0: angle0, ..Default::default() };
    finish(&mut graph, &c, st)
}

fn draw_pie<DB: DrawingBackend>(g: &Graph, root: &DrawingArea<DB, plotters::coord::Shift>) -> Result<(), String>
where
    DB::ErrorType: 'static,
{
    let e = |x: DrawingAreaErrorKind<DB::ErrorType>| rc_err(603, format!("could not render pie chart: {x:?}"));
    let area = if g.title.is_empty() { root.clone() } else { root.titled(&g.title, ("sans-serif", 20)).map_err(e)? };
    let (w, h) = area.dim_in_pixel();
    let legend_w = 180.min(w as i32 / 3);
    let r = ((w as i32 - legend_w).min(h as i32) as f64 / 2.0 - 30.0).max(10.0);
    let (cx, cy) = ((w as i32 - legend_w) as f64 / 2.0, h as f64 / 2.0);
    let total: f64 = g.pie.iter().map(|s| s.value.max(0.0)).sum();
    let pal = palette(&g.scheme);
    // radians clockwise from 12 o'clock; angle0() is counter-clockwise from 3 o'clock
    let mut a0 = (90.0 - g.pie_angle0.unwrap_or(90.0)).to_radians();
    for (k, s) in g.pie.iter().enumerate() {
        let frac = if total > 0.0 { s.value.max(0.0) / total } else { 0.0 };
        let a1 = a0 + frac * std::f64::consts::TAU;
        let mid = (a0 + a1) / 2.0;
        let off = if s.explode { r * 0.1 } else { 0.0 };
        let (ox, oy) = (cx + off * mid.sin(), cy - off * mid.cos());
        let steps = ((a1 - a0) * 60.0).ceil().max(2.0) as usize;
        let mut poly = vec![(ox as i32, oy as i32)];
        poly.extend((0..=steps).map(|j| {
            let a = a0 + (a1 - a0) * j as f64 / steps as f64;
            ((ox + r * a.sin()) as i32, (oy - r * a.cos()) as i32)
        }));
        let col = pal[k % pal.len()];
        area.draw(&Polygon::new(poly.clone(), col.filled())).map_err(e)?;
        area.draw(&PathElement::new([poly.clone(), vec![poly[0]]].concat(), WHITE.stroke_width(2))).map_err(e)?;
        if !s.text.is_empty() {
            let (tx, ty) = (ox + 0.65 * r * mid.sin(), oy - 0.65 * r * mid.cos());
            let style = TextStyle::from(("sans-serif", 13).into_font()).color(&WHITE).pos(Pos::new(HPos::Center, VPos::Center));
            area.draw(&Text::new(s.text.clone(), (tx as i32, ty as i32), style)).map_err(e)?;
        }
        if g.legend {
            let y = 40 + 22 * k as i32;
            let x = w as i32 - legend_w + 10;
            area.draw(&Rectangle::new([(x, y), (x + 14, y + 14)], col.filled())).map_err(e)?;
            area.draw(&Text::new(s.label.clone(), (x + 20, y + 1), ("sans-serif", 13))).map_err(e)?;
        }
        a0 = a1;
    }
    root.present().map_err(e)?;
    Ok(())
}

/// graph matrix varlist [, half]: a scatterplot of every pair of variables,
/// variable names on the diagonal; `half` keeps the lower triangle.
fn graph_matrix(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let c = parse_line(&format!("matrix {rest}")).map_err(|e| rc_err(198, e))?;
    let mask = util::touse(ds, st, &c)?;
    let vars = varlist::expand(ds, &c.main)?;
    if vars.len() < 2 {
        return Err(rc_err(102, "too few variables specified"));
    }
    let half = c.has_opt("half", 4);
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    // the same observations in every panel
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && cols.iter().all(|c| !is_missing(c[i]))).collect();
    let label = |v: &str| ds.get_var(v).map(|x| if x.label.is_empty() { v.to_string() } else { x.label.clone() }).unwrap_or_default();
    let k = vars.len();
    let mut children = Vec::new();
    for r in 0..k {
        for j in 0..k {
            let g = if r == j {
                Graph { center_text: label(&vars[r]), ..Default::default() }
            } else if half && j > r {
                Graph::default()
            } else {
                let pts = rows.iter().map(|&i| (cols[j][i], cols[r][i])).collect();
                Graph { layers: vec![layer(LayerKind::Scatter, pts, &vars[r])], ..Default::default() }
            };
            children.push(g);
        }
    }
    let mut graph = Graph { children, cols: k, ..Default::default() };
    finish(&mut graph, &c, st)
}

/// graph box: five-number boxes (whiskers to the most extreme values within
/// 1.5 IQR of the box, points beyond them drawn individually).
fn graph_box(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let c = parse_line(&format!("box {rest}")).map_err(|e| rc_err(198, e))?;
    let mask = util::touse(ds, st, &c)?;
    let vars = varlist::expand(ds, &c.main)?;
    let groups = over_groups(ds, &c, &mask)?;
    let horizontal = c.has_opt("horizontal", 3) || c.verb == "hbox";
    let marker = c.opt("medtype", 4).is_some_and(|m| m.trim().starts_with("mark"));
    let nv = vars.len().max(1);
    let ng = groups.len();
    let mut graph = Graph::default();
    for (vi, v) in vars.iter().enumerate() {
        let mut l = layer(LayerKind::Box, Vec::new(), v);
        l.horizontal = horizontal;
        l.median_marker = marker;
        l.width = 0.8 / nv as f64;
        for (g, (_, m)) in groups.iter().enumerate() {
            let mut x: Vec<f64> = util::rows(m).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
            if x.is_empty() {
                continue;
            }
            x.sort_by(f64::total_cmp);
            let q1 = util::pctile_sorted(&x, 25.0);
            let q3 = util::pctile_sorted(&x, 75.0);
            let iqr = q3 - q1;
            let lo = x.iter().cloned().find(|&v| v >= q1 - 1.5 * iqr).unwrap_or(q1);
            let hi = x.iter().rev().cloned().find(|&v| v <= q3 + 1.5 * iqr).unwrap_or(q3);
            let pos = g as f64 + (vi as f64 + 0.5) / nv as f64 * 0.8 + 0.1;
            let pos = if horizontal { ng as f64 - pos } else { pos };
            l.points.push((pos, 0.0));
            l.boxes.push([q1, util::pctile_sorted(&x, 50.0), q3, lo, hi]);
            // outside values, stored as degenerate ranges (position, value, value)
            l.ranges.extend(x.iter().filter(|&&v| v < lo || v > hi).map(|&v| (pos, v, v)));
        }
        graph.layers.push(l);
    }
    let labels: Vec<(f64, String)> = groups.iter().enumerate().map(|(g, (n, _))| (if horizontal { ng as f64 - g as f64 - 0.5 } else { g as f64 + 0.5 }, n.replace('\n', ", "))).collect();
    if horizontal {
        graph.ylabels = labels;
    } else {
        graph.xlabels = labels;
    }
    graph.legend = vars.len() > 1;
    finish(&mut graph, &c, st)
}

/// Okabe & Ito (2008), "Color Universal Design": eight colours that stay
/// distinguishable with the common forms of colour-vision deficiency. Ordered
/// so the first series get the most distinct colours; yellow, weak on white,
/// comes last.
const OKABE_ITO: [RGBColor; 8] = [
    RGBColor(0, 114, 178),   // blue
    RGBColor(213, 94, 0),    // vermillion
    RGBColor(0, 158, 115),   // bluish green
    RGBColor(204, 121, 167), // reddish purple
    RGBColor(230, 159, 0),   // orange
    RGBColor(86, 180, 233),  // sky blue
    RGBColor(0, 0, 0),       // black
    RGBColor(240, 228, 66),  // yellow
];
/// The earlier default palette.
const CLASSIC: [RGBColor; 6] = [RGBColor(26, 71, 111), RGBColor(144, 53, 59), RGBColor(85, 117, 47), RGBColor(227, 126, 0), RGBColor(110, 142, 132), RGBColor(193, 5, 52)];
/// Grays for print.
const MONO: [RGBColor; 5] = [RGBColor(40, 40, 40), RGBColor(110, 110, 110), RGBColor(160, 160, 160), RGBColor(75, 75, 75), RGBColor(200, 200, 200)];

/// Scheme names accepted by `set scheme` and `scheme()`.
pub fn is_scheme(name: &str) -> bool {
    matches!(name, "okabeito" | "okabe_ito" | "colorblind" | "classic" | "mono" | "gray" | "grey")
}

/// Colours of a scheme, in series order.
pub fn palette(scheme: &str) -> &'static [RGBColor] {
    match scheme {
        "classic" => &CLASSIC,
        "mono" | "gray" | "grey" => &MONO,
        _ => &OKABE_ITO,
    }
}

/// Render a graph to a file (PNG by default, SVG for .svg).
pub fn render(g: &Graph, path: &str, w: u32, h: u32) -> Result<(), String> {
    render_with(g, path, w, h, "")
}

/// Render; `printcolor` (gs1/gs2/gs3) converts PDF output to gray or black and white.
pub fn render_with(g: &Graph, path: &str, w: u32, h: u32, printcolor: &str) -> Result<(), String> {
    let lower = path.to_lowercase();
    if lower.ends_with(".svg") {
        let svg = render_svg(g, w, h)?;
        ferrostat_core::vfs::write(path, svg.as_bytes()).map_err(|e| rc_err(603, e))
    } else if lower.ends_with(".gph") {
        save_gph(g, path)
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        if lower.ends_with(".eps") || lower.ends_with(".ps") {
            let mut rgb = render_rgb(g, w, h)?;
            apply_printcolor(&mut rgb, printcolor);
            return ferrostat_core::vfs::write(path, &eps_with_image(&rgb, w, h)).map_err(|e| rc_err(603, e));
        }
        render_raster(g, path, w, h, lower.ends_with(".pdf"), printcolor)
    }
}

/// Encapsulated PostScript holding an RGB image (hex-encoded, 0.6 pt per pixel).
pub fn eps_with_image(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (pw, ph) = ((w as f64 * 0.6).round(), (h as f64 * 0.6).round());
    let mut s = format!(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 {pw} {ph}\n%%Pages: 1\n%%EndComments\ngsave\n{pw} {ph} scale\n/DeviceRGB setcolorspace\n<< /ImageType 1 /Width {w} /Height {h} /BitsPerComponent 8 /Decode [0 1 0 1 0 1] /ImageMatrix [{w} 0 0 -{h} 0 {h}] /DataSource currentfile /ASCIIHexDecode filter >> image\n"
    );
    s.reserve(rgb.len() * 2 + rgb.len() / 39 + 64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for chunk in rgb.chunks(39) {
        for b in chunk {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 15) as usize] as char);
        }
        s.push('\n');
    }
    s.push_str(">\ngrestore\nshowpage\n%%EOF\n");
    s.into_bytes()
}

/// The graph as an SVG document.
pub fn render_svg(g: &Graph, w: u32, h: u32) -> Result<String, String> {
    let mut s = String::new();
    {
        let root = SVGBackend::with_string(&mut s, (w, h)).into_drawing_area();
        draw(g, &root)?;
        root.present().map_err(|e| rc_err(693, e))?;
    }
    Ok(s)
}

/// The graph as packed RGB pixels (w * h * 3 bytes).
#[cfg(not(target_arch = "wasm32"))]
pub fn render_rgb(g: &Graph, w: u32, h: u32) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buf, (w, h)).into_drawing_area();
        draw(g, &root)?;
        root.present().map_err(|e| rc_err(693, e))?;
    }
    Ok(buf)
}

#[cfg(not(target_arch = "wasm32"))]
fn render_raster(g: &Graph, path: &str, w: u32, h: u32, pdf: bool, printcolor: &str) -> Result<(), String> {
    let mut rgb = render_rgb(g, w, h)?;
    if pdf {
        apply_printcolor(&mut rgb, printcolor);
    }
    let bytes = if pdf { pdf_with_image(&rgb, w, h) } else { png_encode(&rgb, w, h) };
    ferrostat_core::vfs::write(path, &bytes).map_err(|e| rc_err(603, e))
}

/// In the browser there are no system fonts to rasterise text with: the
/// graph is queued as SVG and the host page converts it (see `take_raster_jobs`).
#[cfg(target_arch = "wasm32")]
fn render_raster(g: &Graph, path: &str, w: u32, h: u32, pdf: bool, _printcolor: &str) -> Result<(), String> {
    let svg = render_svg(g, w, h)?;
    RASTER_JOBS.with(|j| j.borrow_mut().push(RasterJob { path: ferrostat_core::vfs::norm(path), svg, width: w, height: h, pdf }));
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub struct RasterJob {
    pub path: String,
    pub svg: String,
    pub width: u32,
    pub height: u32,
    pub pdf: bool,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static RASTER_JOBS: std::cell::RefCell<Vec<RasterJob>> = std::cell::RefCell::new(Vec::new());
}

/// PNG/PDF exports waiting to be rasterised by the host page.
#[cfg(target_arch = "wasm32")]
pub fn take_raster_jobs() -> Vec<RasterJob> {
    RASTER_JOBS.with(|j| std::mem::take(&mut *j.borrow_mut()))
}

/// Encode packed RGB pixels as a PNG file.
pub fn png_encode(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    use std::io::Write;
    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xffff_ffffu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }
    let mut raw = Vec::with_capacity(rgb.len() + h as usize);
    for row in rgb.chunks(w as usize * 3) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = z.write_all(&raw);
    let idat = z.finish().unwrap_or_default();
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &idat);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Minimal PDF 1.4 document with one page showing an RGB image.
pub fn pdf_with_image(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    // page size in points: fit the longer side to 720pt
    let scale = 720.0 / (w.max(h) as f64);
    let (pw, ph) = ((w as f64 * scale).round(), (h as f64 * scale).round());
    let content = format!("q {pw} 0 0 {ph} 0 0 cm /Im0 Do Q");
    let mut objs: Vec<Vec<u8>> = Vec::new();
    objs.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
    objs.push(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec());
    objs.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw} {ph}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").into_bytes());
    let mut img = format!("<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>\nstream\n", rgb.len()).into_bytes();
    img.extend_from_slice(rgb);
    img.extend_from_slice(b"\nendstream");
    objs.push(img);
    objs.push(format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content).into_bytes());
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(o);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in offsets {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", objs.len() + 1, xref).as_bytes());
    out
}

/// Save a graph in Ferrostat's own text-based .gph format.
pub fn save_gph(g: &Graph, path: &str) -> Result<(), String> {
    let mut s = String::from("ferrostat-graph 1\n");
    s.push_str(&format!("title\t{}\nxtitle\t{}\nytitle\t{}\nlegend\t{}\nxformat\t{}\nyformat\t{}\n", g.title, g.xtitle, g.ytitle, g.legend, g.xformat, g.yformat));
    s.push_str(&format!("ylines\t{}\nxlines\t{}\nscheme\t{}\nname\t{}\ncommand\t{}\ncreated\t{}\n", join(&g.ylines), join(&g.xlines), g.scheme, g.name, g.command.replace(['\t', '\n'], " "), g.created));
    for (x, l) in &g.xlabels {
        // second-row labels start with a newline, kept as an escape
        s.push_str(&format!("xlabel\t{x}\t{}\n", l.replace('\n', "\\n")));
    }
    for (y, l) in &g.ylabels {
        s.push_str(&format!("ylabel\t{y}\t{l}\n"));
    }
    if !g.guides.is_empty() {
        s.push_str(&format!("guides\t{}\n", join(&g.guides)));
    }
    for (k, l) in &g.legend_order {
        s.push_str(&format!("order\t{k}\t{l}\n"));
    }
    for (l, c) in &g.color_legend {
        s.push_str(&format!("ckey\t{l}\t{} {} {}\n", c.0, c.1, c.2));
    }
    if !g.center_text.is_empty() {
        s.push_str(&format!("center\t{}\n", g.center_text));
    }
    for p in &g.pie {
        s.push_str(&format!("pie\t{}\t{}\t{}\t{}\n", p.value, p.explode, p.label, p.text));
    }
    for l in &g.layers {
        let colour = l.color.map(|(r, g, b)| format!("{r} {g} {b}")).unwrap_or_default();
        s.push_str(&format!("layer\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", l.kind, l.width, l.label, l.horizontal, l.hollow, l.alpha, l.median_marker, l.symbol, l.pattern, colour, l.nofill));
        if !l.sizes.is_empty() {
            s.push_str(&format!("s\t{}\n", join(&l.sizes)));
        }
        for m in &l.mlabels {
            s.push_str(&format!("m\t{}\n", m.replace(['\t', '\n'], " ")));
        }
        for (x, y) in &l.points {
            s.push_str(&format!("p\t{x}\t{y}\n"));
        }
        for b in &l.boxes {
            s.push_str(&format!("b\t{}\n", join(b)));
        }
        for (x, y1, y2) in &l.ranges {
            s.push_str(&format!("r\t{x}\t{y1}\t{y2}\n"));
        }
        for (k, g) in l.segments.iter().enumerate() {
            let c = l.seg_colors.get(k).map(|c| format!("\t{} {} {}", c.0, c.1, c.2)).unwrap_or_default();
            s.push_str(&format!("g\t{}{c}\n", join(g)));
        }
        for (c, col) in l.cells.iter().zip(&l.cell_colors) {
            s.push_str(&format!("cell\t{}\t{} {} {}\n", join(c), col.0, col.1, col.2));
        }
    }
    ferrostat_core::vfs::write(path, s.as_bytes()).map_err(|e| rc_err(603, e))
}

fn join(v: &[f64]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

pub fn load_gph(path: &str) -> Result<Graph, String> {
    let text = ferrostat_core::vfs::read_to_string(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    if !text.starts_with("ferrostat-graph") {
        return Err(rc_err(610, format!("{path} is not a Ferrostat graph file")));
    }
    let mut g = Graph::default();
    let nums = |s: &str| -> Vec<f64> { s.split_whitespace().filter_map(|x| x.parse().ok()).collect() };
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "title" => g.title = f.get(1).unwrap_or(&"").to_string(),
            "xtitle" => g.xtitle = f.get(1).unwrap_or(&"").to_string(),
            "xformat" => g.xformat = f.get(1).unwrap_or(&"").to_string(),
            "yformat" => g.yformat = f.get(1).unwrap_or(&"").to_string(),
            "scheme" => g.scheme = f.get(1).unwrap_or(&"").to_string(),
            "name" => g.name = f.get(1).unwrap_or(&"").to_string(),
            "command" => g.command = f.get(1).unwrap_or(&"").to_string(),
            "created" => g.created = f.get(1).unwrap_or(&"").to_string(),
            "ytitle" => g.ytitle = f.get(1).unwrap_or(&"").to_string(),
            "legend" => g.legend = f.get(1) == Some(&"true"),
            "ylines" => g.ylines = nums(f.get(1).unwrap_or(&"")),
            "xlines" => g.xlines = nums(f.get(1).unwrap_or(&"")),
            "xlabel" => g.xlabels.push((f[1].parse().unwrap_or(0.0), f.get(2).unwrap_or(&"").replace("\\n", "\n"))),
            "ylabel" => g.ylabels.push((f[1].parse().unwrap_or(0.0), f.get(2).unwrap_or(&"").to_string())),
            "guides" => g.guides = nums(f.get(1).unwrap_or(&"")),
            "ckey" => g.color_legend.push((f.get(1).unwrap_or(&"").to_string(), f.get(2).and_then(|c| crate::colors::parse(c)).unwrap_or((0, 0, 0)))),
            "center" => g.center_text = f.get(1).unwrap_or(&"").to_string(),
            "order" => g.legend_order.push((f[1].parse().unwrap_or(0), f.get(2).unwrap_or(&"").to_string())),
            "pie" => g.pie.push(PieSlice {
                value: f[1].parse().unwrap_or(0.0),
                explode: f.get(2) == Some(&"true"),
                label: f.get(3).unwrap_or(&"").to_string(),
                text: f.get(4).unwrap_or(&"").to_string(),
            }),
            "layer" => {
                let kind = match f[1] {
                    "Scatter" => LayerKind::Scatter,
                    "Line" => LayerKind::Line,
                    "Connected" => LayerKind::Connected,
                    "Area" => LayerKind::Area,
                    "Bar" => LayerKind::Bar,
                    "Sunflower" => LayerKind::Sunflower,
                    "Dot" => LayerKind::Dot,
                    "Surface" => LayerKind::Surface,
                    "ContourLines" => LayerKind::ContourLines,
                    s if s.starts_with("Pc(") => LayerKind::Pc(PcStyle::from_name(s.trim_start_matches("Pc(").trim_end_matches(')'))),
                    s if s.starts_with("Range(") => LayerKind::Range(RangeStyle::from_name(s.trim_start_matches("Range(").trim_end_matches(')')).unwrap_or(RangeStyle::Bar)),
                    _ => LayerKind::Box,
                };
                g.layers.push(Layer {
                    kind,
                    points: Vec::new(),
                    label: f.get(3).unwrap_or(&"").to_string(),
                    width: f[2].parse().unwrap_or(0.0),
                    boxes: Vec::new(),
                    horizontal: f.get(4) == Some(&"true"),
                    hollow: f.get(5) == Some(&"true"),
                    alpha: f.get(6).and_then(|a| a.parse().ok()).unwrap_or(0.0),
                    median_marker: f.get(7) == Some(&"true"),
                    symbol: f.get(8).unwrap_or(&"").to_string(),
                    pattern: f.get(9).unwrap_or(&"").to_string(),
                    color: f.get(10).and_then(|c| crate::colors::parse(c)),
                    nofill: f.get(11) == Some(&"true"),
                    ..Default::default()
                });
            }
            "p" => {
                if let Some(l) = g.layers.last_mut() {
                    l.points.push((f[1].parse().unwrap_or(0.0), f[2].parse().unwrap_or(0.0)));
                }
            }
            "s" => {
                if let Some(l) = g.layers.last_mut() {
                    l.sizes = nums(f.get(1).unwrap_or(&""));
                }
            }
            "g" => {
                if let Some(l) = g.layers.last_mut() {
                    let v = nums(f.get(1).unwrap_or(&""));
                    if v.len() == 4 {
                        l.segments.push([v[0], v[1], v[2], v[3]]);
                        if let Some(c) = f.get(2).and_then(|c| crate::colors::parse(c)) {
                            l.seg_colors.push(c);
                        }
                    }
                }
            }
            "cell" => {
                if let Some(l) = g.layers.last_mut() {
                    let v = nums(f.get(1).unwrap_or(&""));
                    if let (4, Some(c)) = (v.len(), f.get(2).and_then(|c| crate::colors::parse(c))) {
                        l.cells.push([v[0], v[1], v[2], v[3]]);
                        l.cell_colors.push(c);
                    }
                }
            }
            "m" => {
                if let Some(l) = g.layers.last_mut() {
                    l.mlabels.push(f.get(1).unwrap_or(&"").to_string());
                }
            }
            "r" => {
                if let Some(l) = g.layers.last_mut() {
                    let v: Vec<f64> = f[1..].iter().filter_map(|x| x.parse().ok()).collect();
                    if v.len() == 3 {
                        l.ranges.push((v[0], v[1], v[2]));
                    }
                }
            }
            "b" => {
                if let Some(l) = g.layers.last_mut() {
                    let v = nums(f[1]);
                    if v.len() == 5 {
                        l.boxes.push([v[0], v[1], v[2], v[3], v[4]]);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(g)
}

/// Marker shape names: (shape, hollow, small). Accepts the long names
/// (circle, diamond, triangle, square, plus, x, point, none, sm…, …_hollow)
/// and the one-letter forms (O D T S + X p i; lower case = small; h = hollow).
pub fn symbol_shape(s: &str) -> (&'static str, bool, bool) {
    let s = s.trim();
    let long = s.to_lowercase();
    let (base, hollow) = match long.strip_suffix("_hollow") {
        Some(b) => (b.to_string(), true),
        None => (long.clone(), false),
    };
    let (base, small) = match base.strip_prefix("sm") {
        Some(b) if b != "" => (b.to_string(), true),
        _ => (base, false),
    };
    let named = match base.as_str() {
        "circle" => Some("circle"),
        "diamond" => Some("diamond"),
        "triangle" => Some("triangle"),
        "square" => Some("square"),
        "plus" => Some("plus"),
        "x" if s.len() > 1 => Some("x"),
        "point" => Some("point"),
        "none" => Some("none"),
        _ => None,
    };
    if let Some(n) = named {
        return (n, hollow, small);
    }
    // one-letter codes, optionally followed by h (hollow)
    let mut ch = s.chars();
    let c = ch.next().unwrap_or('O');
    let hollow = ch.next() == Some('h');
    let shape = match c.to_ascii_uppercase() {
        'D' => "diamond",
        'T' => "triangle",
        'S' => "square",
        '+' => "plus",
        'X' => "x",
        'P' => "point",
        'I' => "none",
        _ => "circle",
    };
    (shape, hollow, c.is_ascii_lowercase() && c != 'p' && c != 'i')
}

enum MarkerShape {
    Circle(i32, ShapeStyle),
    Poly(Vec<(i32, i32)>, ShapeStyle),
    Path(Vec<(i32, i32)>, ShapeStyle),
    Cross(Vec<(i32, i32)>, Vec<(i32, i32)>, ShapeStyle),
    None,
}

fn marker_shape(sym: &str, r: i32, col: RGBColor) -> MarkerShape {
    let (shape, hollow, small) = symbol_shape(if sym.is_empty() { "O" } else { sym });
    let r = if small { ((r as f64) * 0.7).round().max(1.0) as i32 } else { r };
    let fill = col.filled();
    let line = col.stroke_width(1);
    let outline = |pts: Vec<(i32, i32)>| {
        if hollow {
            let mut closed = pts.clone();
            closed.push(pts[0]);
            MarkerShape::Path(closed, line)
        } else {
            MarkerShape::Poly(pts, fill)
        }
    };
    match shape {
        "diamond" => outline(vec![(0, -r - 1), (r + 1, 0), (0, r + 1), (-r - 1, 0)]),
        "triangle" => outline(vec![(0, -r - 1), (r + 1, r), (-r - 1, r)]),
        "square" => outline(vec![(-r, -r), (r, -r), (r, r), (-r, r)]),
        "plus" => MarkerShape::Cross(vec![(-r - 1, 0), (r + 1, 0)], vec![(0, -r - 1), (0, r + 1)], col.stroke_width(2)),
        "x" => MarkerShape::Cross(vec![(-r, -r), (r, r)], vec![(-r, r), (r, -r)], col.stroke_width(2)),
        "point" => MarkerShape::Circle(1, fill),
        "none" => MarkerShape::None,
        _ => MarkerShape::Circle(r, if hollow { line } else { fill }),
    }
}

type Chart2d<'a, DB> = ChartContext<'a, DB, Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>>;

/// Draw markers of one symbol at `pts` (radius per point), with a legend entry.
fn draw_markers<'a, DB: DrawingBackend + 'a>(chart: &mut Chart2d<'a, DB>, pts: &[(f64, f64)], radius: &dyn Fn(usize) -> i32, sym: &str, col: RGBColor, label: &Option<String>) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (shape, _, _) = symbol_shape(if sym.is_empty() { "O" } else { sym });
    let sym = sym.to_string();
    // Markers land on whole pixels and are opaque, so a marker at the same
    // pixel with the same size as an earlier one changes nothing: skip it.
    // This bounds the output for scatters of millions of points.
    let mut seen = std::collections::HashSet::new();
    let keep: Vec<usize> = (0..pts.len()).filter(|&i| seen.insert((chart.backend_coord(&pts[i]), radius(i)))).collect();
    let pts: Vec<(f64, f64)> = keep.iter().map(|&i| pts[i]).collect();
    let radius = |i: usize| radius(keep[i]);
    let pts = &pts[..];
    match shape {
        "circle" | "point" => {
            let st = move |r: i32| match marker_shape(&sym, r, col) { MarkerShape::Circle(rr, s) => (rr, s), _ => (r, col.filled()) };
            let st2 = st.clone();
            let anno = chart.draw_series(pts.iter().enumerate().map(|(i, p)| { let (rr, s) = st(radius(i)); Circle::new(*p, rr, s) }))?;
            if let Some(l) = label {
                anno.label(l.clone()).legend(move |(x, y)| { let (rr, s) = st2(3); Circle::new((x + 10, y), rr, s) });
            }
        }
        "none" => {
            let anno = chart.draw_series(std::iter::empty::<Circle<(f64, f64), i32>>())?;
            if let Some(l) = label {
                anno.label(l.clone()).legend(move |(x, y)| Circle::new((x + 10, y), 0, col.filled()));
            }
        }
        "plus" | "x" => {
            let lines = move |r: i32| match marker_shape(&sym, r, col) { MarkerShape::Cross(a, b, s) => (a, b, s), _ => (vec![], vec![], col.filled()) };
            let l2 = lines.clone();
            let anno = chart.draw_series(pts.iter().enumerate().map(|(i, p)| { let (a, b, s) = lines(radius(i)); EmptyElement::at(*p) + PathElement::new(a, s) + PathElement::new(b, s) }))?;
            if let Some(l) = label {
                anno.label(l.clone()).legend(move |(x, y)| { let (a, b, s) = l2(3); EmptyElement::at((x + 10, y)) + PathElement::new(a, s) + PathElement::new(b, s) });
            }
        }
        _ => {
            // polygons: filled, or outlines for hollow symbols
            let poly = move |r: i32| match marker_shape(&sym, r, col) { MarkerShape::Poly(p, s) => (p, s, false), MarkerShape::Path(p, s) => (p, s, true), _ => (vec![], col.filled(), false) };
            let p2 = poly.clone();
            let anno = chart.draw_series(pts.iter().enumerate().map(|(i, p)| {
                let (v, s, hollow) = poly(radius(i));
                // a closed outline drawn as a polygon with an invisible fill keeps one element type
                if hollow { EmptyElement::at(*p) + Polygon::new(v.clone(), WHITE.mix(0.0).filled()) + PathElement::new(v, s) } else { EmptyElement::at(*p) + Polygon::new(v.clone(), s) + PathElement::new(v, s) }
            }))?;
            if let Some(l) = label {
                anno.label(l.clone()).legend(move |(x, y)| {
                    let (v, s, hollow) = p2(3);
                    if hollow { EmptyElement::at((x + 10, y)) + Polygon::new(v.clone(), WHITE.mix(0.0).filled()) + PathElement::new(v, s) } else { EmptyElement::at((x + 10, y)) + Polygon::new(v.clone(), s) + PathElement::new(v, s) }
                });
            }
        }
    }
    Ok(())
}

/// Line pattern names: (dash, gap) in pixels, or None for solid.
pub fn line_pattern(p: &str) -> Option<(u32, u32)> {
    match p.trim().to_lowercase().as_str() {
        "" | "solid" | "l" => None,
        "dash" | "-" => Some((8, 5)),
        "dot" | "." | "shortdot" => Some((2, 4)),
        "shortdash" | "_" => Some((4, 4)),
        "longdash" | "__" => Some((14, 6)),
        "dash_dot" | "-." | "shortdash_dot" | "longdash_dot" | "longdash_shortdash" => Some((10, 4)),
        "blank" | "none" => Some((0, 1000)),
        _ => None,
    }
}

/// A line through `pts` in the given pattern.
fn draw_line<'a, 'b, DB: DrawingBackend + 'a>(
    chart: &'b mut Chart2d<'a, DB>,
    pts: &[(f64, f64)],
    col: RGBColor,
    pattern: &str,
) -> Result<&'b mut plotters::chart::SeriesAnno<'a, DB>, DrawingAreaErrorKind<DB::ErrorType>> {
    match line_pattern(pattern) {
        None => chart.draw_series(LineSeries::new(pts.to_vec(), col.stroke_width(2))),
        Some((dash, gap)) => chart.draw_series(DashedLineSeries::new(pts.to_vec(), dash, gap, col.stroke_width(2))),
    }
}

/// `set printcolor`: gs1 and gs2 print in grays (gs2 with more contrast), gs3 in black and white.
pub fn apply_printcolor(rgb: &mut [u8], mode: &str) {
    if !matches!(mode, "gs1" | "gs2" | "gs3") {
        return;
    }
    for px in rgb.chunks_mut(3) {
        let y = crate::colors::luma(px[0], px[1], px[2]) as f64;
        let v = match mode {
            "gs1" => y,
            "gs2" => (((y / 255.0 - 0.5) * 1.6 + 0.5).clamp(0.0, 1.0) * 255.0).round(),
            _ => if y >= 250.0 { 255.0 } else { 0.0 },
        } as u8;
        px.fill(v);
    }
}

#[cfg(test)]
mod style_tests {
    use super::*;

    #[test]
    fn symbols_and_print_colours() {
        assert_eq!(symbol_shape("D"), ("diamond", false, false));
        assert_eq!(symbol_shape("Oh"), ("circle", true, false));
        assert_eq!(symbol_shape("smtriangle_hollow"), ("triangle", true, true));
        assert_eq!(symbol_shape("x"), ("x", false, true));
        assert_eq!(line_pattern("solid"), None);
        assert!(line_pattern("dash").is_some());
        let mut px = vec![0, 114, 178, 255, 255, 255, 230, 159, 0];
        apply_printcolor(&mut px, "gs3");
        assert_eq!(px, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]);
        let mut px = vec![0, 114, 178];
        apply_printcolor(&mut px, "gs1");
        assert!(px[0] == px[1] && px[1] == px[2]);
        let mut px = vec![1, 2, 3];
        apply_printcolor(&mut px, "asis");
        assert_eq!(px, vec![1, 2, 3]);
    }
}

/// Legend text of layer `k`, or None when legend(order()) leaves it out.
fn legend_label(g: &Graph, k: usize) -> Option<String> {
    if matches!(g.layers[k].kind, LayerKind::Surface | LayerKind::ContourLines) {
        return None;
    }
    if g.legend_order.is_empty() {
        // unlabelled layers (immediate plots, guides) have no legend entry
        return Some(g.layers[k].label.clone()).filter(|l| !l.is_empty());
    }
    g.legend_order.iter().find(|(n, _)| *n == k + 1).map(|(_, s)| if s.is_empty() { g.layers[k].label.clone() } else { s.clone() })
}

/// Attach a legend entry only for layers the legend shows.
trait LegendIf<'a, DB: DrawingBackend> {
    fn legend_if<E: IntoDynElement<'a, DB, (i32, i32)>, T: Fn((i32, i32)) -> E + 'a>(&mut self, label: &Option<String>, f: T);
}

impl<'a, DB: DrawingBackend + 'a> LegendIf<'a, DB> for plotters::chart::SeriesAnno<'a, DB> {
    fn legend_if<E: IntoDynElement<'a, DB, (i32, i32)>, T: Fn((i32, i32)) -> E + 'a>(&mut self, label: &Option<String>, f: T) {
        if let Some(l) = label {
            self.label(l.clone()).legend(f);
        }
    }
}

/// Plot coordinates of a range-plot point (x along the vertical axis when horizontal).
fn range_pt(l: &Layer, x: f64, y: f64) -> (f64, f64) {
    if l.horizontal { (y, x) } else { (x, y) }
}

/// Half-width, in x units, of range bars and of the caps of `rcap`.
fn range_half_width(l: &Layer, style: RangeStyle) -> f64 {
    match style {
        RangeStyle::Bar => l.width / 2.0,
        RangeStyle::Cap | RangeStyle::PointCap => {
            // a quarter of the smallest gap between x values; a width below 1 caps it (coefplot)
            let gap = l.ranges.windows(2).map(|w| w[1].0 - w[0].0).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
            let hw = if gap.is_finite() { gap / 4.0 } else { 0.25 };
            if l.width > 0.0 && l.width < 1.0 { hw.min(l.width / 2.0) } else { hw }
        }
        _ => 0.0,
    }
}

/// Date and time display formats (`%td`, `%d`, `%tc`, `%tm`, `%tq`, ...).
fn is_date_format(f: &str) -> bool {
    let f = f.trim_start_matches('%').trim_start_matches('-');
    f.starts_with('t') || f.starts_with('d')
}

/// Axis tick label: dates in the variable's display format, numbers in
/// general format (no spurious trailing `.0`).
fn tick_label(v: f64, format: &str) -> String {
    if is_date_format(format) {
        // ticks fall between dates: label the date they belong to
        crate::format::fmt(v.round(), format).trim().to_string()
    } else {
        let r = (v * 1e9).round() / 1e9;
        crate::format::fmt_g(r, 6)
    }
}

/// Register the embedded font used for all graph text (native builds; the
/// browser build measures and draws text with the page's fonts).
#[cfg(not(target_arch = "wasm32"))]
fn register_fonts() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        static FONT: &[u8] = include_bytes!("../../assets/fonts/DejaVuSans.ttf");
        for family in ["sans-serif", "serif", "monospace"] {
            for style in [FontStyle::Normal, FontStyle::Bold, FontStyle::Italic, FontStyle::Oblique] {
                let _ = plotters::style::register_font(family, style, FONT);
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn register_fonts() {}

fn draw<DB: DrawingBackend>(g: &Graph, root: &DrawingArea<DB, plotters::coord::Shift>) -> Result<(), String>
where
    DB::ErrorType: 'static,
{
    let e = |x: DrawingAreaErrorKind<DB::ErrorType>| rc_err(603, format!("could not render graph: {x:?}"));
    register_fonts();
    root.fill(&WHITE).map_err(e)?;
    if !g.children.is_empty() {
        let cols = g.cols.max(1);
        let rows = (g.children.len() + cols - 1) / cols;
        let area = if g.title.is_empty() { root.clone() } else { root.titled(&g.title, ("sans-serif", 20)).map_err(e)? };
        let cells = area.split_evenly((rows, cols));
        for (child, cell) in g.children.iter().zip(cells.iter()) {
            draw(child, cell)?;
        }
        root.present().map_err(e)?;
        return Ok(());
    }
    if !g.polygons.is_empty() {
        return draw_map(g, root);
    }
    if !g.pie.is_empty() {
        return draw_pie(g, root);
    }
    if g.layers.is_empty() {
        // an empty panel (graph matrix): only its centred text, if any
        if !g.center_text.is_empty() {
            let (w, h) = root.dim_in_pixel();
            let style = TextStyle::from(("sans-serif", 16).into_font()).pos(Pos::new(HPos::Center, VPos::Center));
            root.draw(&Text::new(g.center_text.clone(), (w as i32 / 2, h as i32 / 2), style)).map_err(e)?;
        }
        return Ok(());
    }
    // range layers keep their marker points in data orientation: added below
    let plain = |l: &&Layer| !matches!(l.kind, LayerKind::Range(_));
    let mut xs: Vec<f64> = g.layers.iter().filter(plain).flat_map(|l| l.points.iter().map(|p| p.0)).collect();
    let mut ys: Vec<f64> = g.layers.iter().filter(plain).flat_map(|l| l.points.iter().map(|p| p.1)).collect();
    for l in &g.layers {
        if l.kind == LayerKind::Box {
            // box layers keep positions in `points` (not plot coordinates)
            xs.clear();
            ys.clear();
        }
    }
    for l in &g.layers {
        if l.kind == LayerKind::Box {
            let hw = l.width / 2.0;
            for (i, b) in l.boxes.iter().enumerate() {
                let pos = l.points.get(i).map_or(i as f64 + 0.5, |p| p.0);
                for v in b.iter().chain(l.ranges.iter().filter(|r| r.0 == pos).map(|r| &r.1)) {
                    for q in [range_pt(l, pos - hw, *v), range_pt(l, pos + hw, *v)] {
                        xs.push(q.0);
                        ys.push(q.1);
                    }
                }
            }
            continue;
        }
        if l.kind == LayerKind::Bar {
            ys.push(0.0);
            xs.extend(l.points.iter().flat_map(|p| [p.0 - l.width / 2.0, p.0 + l.width / 2.0]));
        }
        for s in &l.segments {
            xs.extend([s[1], s[3]]);
            ys.extend([s[0], s[2]]);
        }
        for c in &l.cells {
            xs.extend([c[0], c[2]]);
            ys.extend([c[1], c[3]]);
        }
        if let LayerKind::Range(style) = l.kind {
            for &(x, y) in &l.points {
                let q = range_pt(l, x, y);
                xs.push(q.0);
                ys.push(q.1);
            }
            let hw = range_half_width(l, style);
            for &(x, y1, y2) in &l.ranges {
                for (px, py) in [range_pt(l, x - hw, y1), range_pt(l, x + hw, y2)] {
                    xs.push(px);
                    ys.push(py);
                }
            }
        }
    }
    ys.extend(g.ylines.iter().copied());
    xs.extend(g.xlines.iter().copied());
    for (x, _) in &g.xlabels {
        xs.push(x - 0.5);
        xs.push(x + 0.5);
    }
    for (y, _) in &g.ylabels {
        ys.push(y - 0.5);
        ys.push(y + 0.5);
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
    let key_w: u32 = if g.color_legend.is_empty() { 0 } else { 30 + 8 * g.color_legend.iter().map(|(s, _)| s.chars().count() as u32).max().unwrap_or(0) };
    let mut chart = ChartBuilder::on(root)
        .caption(&g.title, ("sans-serif", 24))
        .margin(15)
        .margin_right(15 + key_w)
        .x_label_area_size(45)
        .y_label_area_size(g.ylabels.iter().map(|(_, s)| 18 + 7 * s.chars().count() as u32).max().unwrap_or(0).max(60))
        .build_cartesian_2d((x0 - px)..(x1 + px), (y0 - py)..(y1 + py))
        .map_err(e)?;
    let mut mesh = chart.configure_mesh();
    mesh.x_desc(g.xtitle.clone()).y_desc(g.ytitle.clone()).light_line_style(WHITE);
    let (xfmt, yfmt) = (g.xformat.clone(), g.yformat.clone());
    let (xcat, ycat) = (!g.xlabels.is_empty(), !g.ylabels.is_empty());
    let fmt_x = move |x: &f64| if xcat { String::new() } else { tick_label(*x, &xfmt) };
    let fmt_y = move |y: &f64| if ycat { String::new() } else { tick_label(*y, &yfmt) };
    mesh.x_label_formatter(&fmt_x).y_label_formatter(&fmt_y);
    if xcat {
        mesh.disable_x_mesh();
    }
    if ycat {
        mesh.disable_y_mesh();
    }
    if is_date_format(&g.xformat) {
        // date labels are wide: fewer of them
        mesh.x_labels(6);
    }
    mesh.draw().map_err(e)?;
    // category labels at their exact positions, outside the plotting area
    {
        let (xr, yr) = (chart.x_range(), chart.y_range());
        let font = TextStyle::from(("sans-serif", 12).into_font());
        // backend coordinates are absolute; this area may be one panel of a combined graph
        let (bx, by) = root.get_base_pixel();
        for (x, s) in &g.xlabels {
            let (px, py) = chart.backend_coord(&(*x, yr.start));
            // a leading newline puts the label on a second row (nested categories)
            let (s, dy) = match s.strip_prefix('\n') {
                Some(t) => (t.to_string(), 26),
                None => (s.clone(), 8),
            };
            root.draw(&Text::new(s, (px - bx, py - by + dy), font.clone().pos(Pos::new(HPos::Center, VPos::Top)))).map_err(e)?;
        }
        for (y, s) in &g.ylabels {
            let (px, py) = chart.backend_coord(&(xr.start, *y));
            root.draw(&Text::new(s.clone(), (px - bx - 6, py - by), font.clone().pos(Pos::new(HPos::Right, VPos::Center)))).map_err(e)?;
        }
        // dot charts: dotted guide lines at each category
        for &y in &g.guides {
            let n = 80;
            chart.draw_series((0..n).map(|i| {
                let x = xr.start + (xr.end - xr.start) * (i as f64 + 0.5) / n as f64;
                Circle::new((x, y), 1, RGBColor(150, 150, 150).filled())
            })).map_err(e)?;
        }
    }
    for (k, l) in g.layers.iter().enumerate() {
        let pal = palette(&g.scheme);
        let col = l.color.map(|(r, gg, b)| RGBColor(r, gg, b)).unwrap_or(pal[k % pal.len()]);
        let label = legend_label(g, k);
        crate::hover::record(&chart, g, k, l);
        match l.kind {
            LayerKind::Scatter => {
                let sym = if l.symbol.is_empty() && l.hollow { "Oh".to_string() } else { l.symbol.clone() };
                let radius = |i: usize| (3.0 * l.sizes.get(i).copied().unwrap_or(1.0)).round().max(1.0) as i32;
                draw_markers(&mut chart, &l.points, &radius, &sym, col, &label).map_err(e)?;
                if !l.mlabels.is_empty() {
                    chart.draw_series(l.points.iter().zip(&l.mlabels).map(|(p, s)| Text::new(s.clone(), *p, ("sans-serif", 11).into_font().color(&col)))).map_err(e)?;
                }
            }
            LayerKind::Line => {
                draw_line(&mut chart, &l.points, col, &l.pattern).map_err(e)?.legend_if(&label, move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], col.stroke_width(2)));
            }
            LayerKind::Connected => {
                draw_line(&mut chart, &l.points, col, &l.pattern).map_err(e)?;
                draw_markers(&mut chart, &l.points, &|_| 3, &l.symbol, col, &label).map_err(e)?;
            }
            LayerKind::Area => {
                // opaque fills drawn in order: later series cover earlier ones
                let fill = if l.nofill { col.mix(0.0).filled() } else if l.alpha > 0.0 { col.mix(l.alpha).filled() } else { col.filled() };
                chart.draw_series(AreaSeries::new(l.points.clone(), 0.0, fill).border_style(col)).map_err(e)?.legend_if(&label, move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.filled()));
            }
            LayerKind::Bar => {
                let hw = l.width / 2.0 * 0.95;
                chart.draw_series(l.points.iter().map(|p| Rectangle::new([(p.0 - hw, 0.0), (p.0 + hw, p.1)], col.mix(0.8).filled()))).map_err(e)?.legend_if(&label, move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.filled()));
            }
            LayerKind::Range(style) => {
                let hw = range_half_width(l, style);
                let p = |x: f64, y: f64| range_pt(l, x, y);
                let fill = if l.alpha > 0.0 { col.mix(l.alpha).filled() } else { col.filled() };
                let rect_legend = move |(x, y): (i32, i32)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], fill);
                match style {
                    RangeStyle::Bar if l.nofill => {
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| Rectangle::new([p(x - hw, y1), p(x + hw, y2)], col.stroke_width(1)))).map_err(e)?.legend_if(&label, move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.stroke_width(1)));
                    }
                    RangeStyle::Bar => {
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| Rectangle::new([p(x - hw, y1), p(x + hw, y2)], fill))).map_err(e)?.legend_if(&label, rect_legend);
                        // thin outline keeps adjacent bars apart
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| Rectangle::new([p(x - hw, y1), p(x + hw, y2)], WHITE.stroke_width(1)))).map_err(e)?;
                        // blabel(): the value at the end of each bar
                        let anchor = if l.horizontal { Pos::new(HPos::Left, VPos::Center) } else { Pos::new(HPos::Center, VPos::Bottom) };
                        chart.draw_series(l.ranges.iter().zip(&l.mlabels).map(|(&(x, _, y2), s)| Text::new(s.clone(), p(x, y2), ("sans-serif", 11).into_font().color(&BLACK).pos(anchor)))).map_err(e)?;
                    }
                    RangeStyle::Area => {
                        let mut poly: Vec<(f64, f64)> = l.ranges.iter().map(|&(x, _, y2)| p(x, y2)).collect();
                        poly.extend(l.ranges.iter().rev().map(|&(x, y1, _)| p(x, y1)));
                        chart.draw_series(std::iter::once(Polygon::new(poly, fill))).map_err(e)?.legend_if(&label, rect_legend);
                    }
                    RangeStyle::Drop => {
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| PathElement::new(vec![p(x, y1), p(x, y2)], col.stroke_width(1)))).map_err(e)?;
                        chart.draw_series(l.ranges.iter().map(|&(x, _, y2)| Circle::new(p(x, y2), 3, col.filled()))).map_err(e)?.legend_if(&label, move |(x, y)| Circle::new((x + 10, y), 3, col.filled()));
                    }
                    RangeStyle::PointCap | RangeStyle::PointSpike => {
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| PathElement::new(vec![p(x, y1), p(x, y2)], col.stroke_width(2)))).map_err(e)?;
                        if style == RangeStyle::PointCap {
                            chart.draw_series(l.ranges.iter().flat_map(|&(x, y1, y2)| [y1, y2].map(|yy| PathElement::new(vec![p(x - hw, yy), p(x + hw, yy)], col.stroke_width(2))))).map_err(e)?;
                        }
                        let pts: Vec<(f64, f64)> = l.points.iter().map(|&(x, y)| p(x, y)).collect();
                        draw_markers(&mut chart, &pts, &|_| 3, &l.symbol, col, &label).map_err(e)?;
                    }
                    RangeStyle::Spike | RangeStyle::Cap => {
                        chart.draw_series(l.ranges.iter().map(|&(x, y1, y2)| PathElement::new(vec![p(x, y1), p(x, y2)], col.stroke_width(2)))).map_err(e)?.legend_if(&label, move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], col.stroke_width(2)));
                        if style == RangeStyle::Cap {
                            chart.draw_series(l.ranges.iter().flat_map(|&(x, y1, y2)| [y1, y2].map(|yy| PathElement::new(vec![p(x - hw, yy), p(x + hw, yy)], col.stroke_width(2))))).map_err(e)?;
                        }
                    }
                    RangeStyle::Line | RangeStyle::Connected | RangeStyle::Scatter => {
                        let lo: Vec<(f64, f64)> = l.ranges.iter().map(|&(x, y1, _)| p(x, y1)).collect();
                        let hi: Vec<(f64, f64)> = l.ranges.iter().map(|&(x, _, y2)| p(x, y2)).collect();
                        if style != RangeStyle::Scatter {
                            chart.draw_series(LineSeries::new(lo.clone(), col.stroke_width(2))).map_err(e)?;
                            chart.draw_series(LineSeries::new(hi.clone(), col.stroke_width(2))).map_err(e)?;
                        }
                        if style == RangeStyle::Line {
                            chart.draw_series(std::iter::empty::<PathElement<(f64, f64)>>()).map_err(e)?.legend_if(&label, move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], col.stroke_width(2)));
                        } else {
                            chart.draw_series(lo.iter().chain(hi.iter()).map(|q| Circle::new(*q, 3, col.filled()))).map_err(e)?.legend_if(&label, move |(x, y)| Circle::new((x + 10, y), 3, col.filled()));
                        }
                    }
                }
            }
            LayerKind::Surface => {
                chart.draw_series(l.cells.iter().zip(&l.cell_colors).map(|(c, rgb)| Rectangle::new([(c[0], c[1]), (c[2], c[3])], RGBColor(rgb.0, rgb.1, rgb.2).filled()))).map_err(e)?;
            }
            LayerKind::ContourLines => {
                chart.draw_series(l.segments.iter().zip(&l.seg_colors).map(|(s, rgb)| PathElement::new(vec![(s[1], s[0]), (s[3], s[2])], RGBColor(rgb.0, rgb.1, rgb.2).stroke_width(2)))).map_err(e)?;
            }
            LayerKind::Pc(style) => {
                let segs: Vec<((f64, f64), (f64, f64))> = l.segments.iter().map(|s| ((s[1], s[0]), (s[3], s[2]))).collect();
                if style != PcStyle::Scatter {
                    let anno = chart.draw_series(segs.iter().map(|&(a, b)| PathElement::new(vec![a, b], col.stroke_width(2)))).map_err(e)?;
                    anno.legend_if(&label, move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], col.stroke_width(2)));
                }
                if matches!(style, PcStyle::CapSym | PcStyle::Scatter) {
                    let ends: Vec<(f64, f64)> = segs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    let no_label = if style == PcStyle::Scatter { label.clone() } else { None };
                    draw_markers(&mut chart, &ends, &|_| 3, &l.symbol, col, &no_label).map_err(e)?;
                }
                if matches!(style, PcStyle::Arrow | PcStyle::BArrow) {
                    // arrowheads in pixel space, so they keep their shape on any axis scale
                    let (bx, by) = root.get_base_pixel();
                    for &(a, b) in &segs {
                        let (pa, pb) = (chart.backend_coord(&a), chart.backend_coord(&b));
                        let mut heads = vec![(pb, pa)];
                        if style == PcStyle::BArrow {
                            heads.push((pa, pb));
                        }
                        for (tip, from) in heads {
                            let (dx, dy) = ((tip.0 - from.0) as f64, (tip.1 - from.1) as f64);
                            let len = (dx * dx + dy * dy).sqrt();
                            if len < 1.0 {
                                continue;
                            }
                            let (ux, uy) = (dx / len, dy / len);
                            let wing = |s: f64| {
                                let (c, sn) = ((0.45f64).cos(), s * (0.45f64).sin());
                                let (rx, ry) = (ux * c - uy * sn, ux * sn + uy * c);
                                ((tip.0 as f64 - 10.0 * rx) as i32 - bx, (tip.1 as f64 - 10.0 * ry) as i32 - by)
                            };
                            let t = (tip.0 - bx, tip.1 - by);
                            root.draw(&Polygon::new(vec![t, wing(1.0), wing(-1.0)], col.filled())).map_err(e)?;
                        }
                    }
                }
            }
            LayerKind::Dot => {
                // dotted guide across the plot at each x (each y when horizontal), then the markers
                let (xr, yr) = (chart.x_range(), chart.y_range());
                let n = 60;
                let grey = RGBColor(160, 160, 160).filled();
                let guides: Vec<(f64, f64)> = l.points.iter().flat_map(|&(px, py)| {
                    (0..n).map(move |i| {
                        let t = (i as f64 + 0.5) / n as f64;
                        if l.horizontal { (xr.start + (xr.end - xr.start) * t, py) } else { (px, yr.start + (yr.end - yr.start) * t) }
                    })
                }).collect();
                chart.draw_series(guides.into_iter().map(|q| Circle::new(q, 1, grey))).map_err(e)?;
                draw_markers(&mut chart, &l.points, &|_| 3, &l.symbol, col, &label).map_err(e)?;
            }
            LayerKind::Sunflower => {
                // a dot with one petal per observation, petals evenly spaced
                let petals = |k: f64| -> Vec<PathElement<(i32, i32)>> {
                    let n = k as usize;
                    (0..n).map(|j| {
                        let a = std::f64::consts::TAU * j as f64 / n as f64;
                        PathElement::new(vec![(0, 0), ((7.0 * a.cos()) as i32, (-7.0 * a.sin()) as i32)], col.stroke_width(1))
                    }).collect()
                };
                for (q, k) in l.points.iter().zip(&l.sizes) {
                    chart.draw_series(std::iter::once(Circle::new(*q, 2, col.filled()))).map_err(e)?;
                    chart.draw_series(petals(*k).into_iter().map(|petal| EmptyElement::at(*q) + petal)).map_err(e)?;
                }
                chart.draw_series(std::iter::empty::<Circle<(f64, f64), i32>>()).map_err(e)?.legend_if(&label, move |(x, y)| Circle::new((x + 10, y), 3, col.filled()));
            }
            LayerKind::Box => {
                let hw = l.width / 2.0;
                let p = |x: f64, y: f64| range_pt(l, x, y);
                for (i, b) in l.boxes.iter().enumerate() {
                    let c = l.points.get(i).map_or(i as f64 + 0.5, |q| q.0);
                    let rect = chart.draw_series(std::iter::once(Rectangle::new([p(c - hw, b[0]), p(c + hw, b[2])], col.mix(0.75).filled()))).map_err(e)?;
                    if i == 0 {
                        rect.legend_if(&label, move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], col.filled()));
                    }
                    if l.median_marker {
                        chart.draw_series(std::iter::once(Circle::new(p(c, b[1]), 4, WHITE.filled()))).map_err(e)?;
                    } else {
                        chart.draw_series(std::iter::once(PathElement::new(vec![p(c - hw, b[1]), p(c + hw, b[1])], WHITE.stroke_width(2)))).map_err(e)?;
                    }
                    chart.draw_series([(b[2], b[4]), (b[0], b[3])].map(|(from, to)| PathElement::new(vec![p(c, from), p(c, to)], col))).map_err(e)?;
                    chart.draw_series([b[3], b[4]].map(|w| PathElement::new(vec![p(c - hw / 2.0, w), p(c + hw / 2.0, w)], col))).map_err(e)?;
                }
                // outside values
                chart.draw_series(l.ranges.iter().map(|&(c, v, _)| Circle::new(p(c, v), 2, col.stroke_width(1)))).map_err(e)?;
            }
        }
    }
    for &y in &g.ylines {
        chart.draw_series(std::iter::once(PathElement::new(vec![(x0 - px, y), (x1 + px, y)], RGBColor(144, 53, 59).stroke_width(2)))).map_err(e)?;
    }
    for &x in &g.xlines {
        chart.draw_series(std::iter::once(PathElement::new(vec![(x, y0 - py), (x, y1 + py)], RGBColor(144, 53, 59).stroke_width(2)))).map_err(e)?;
    }
    if !g.color_legend.is_empty() {
        // colour key at the right of the plot, highest band on top
        let (w, _) = root.dim_in_pixel();
        let x0 = w as i32 - key_w as i32;
        for (k, (s, c)) in g.color_legend.iter().rev().enumerate() {
            let y = 40 + 20 * k as i32;
            root.draw(&Rectangle::new([(x0, y), (x0 + 14, y + 14)], RGBColor(c.0, c.1, c.2).filled())).map_err(e)?;
            root.draw(&Text::new(s.clone(), (x0 + 20, y + 1), ("sans-serif", 12))).map_err(e)?;
        }
    }
    if g.legend && (0..g.layers.len()).any(|k| legend_label(g, k).is_some()) {
        chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK).draw().map_err(e)?;
    }
    root.present().map_err(e)?;
    Ok(())
}

fn draw_map<DB: DrawingBackend>(g: &Graph, root: &DrawingArea<DB, plotters::coord::Shift>) -> Result<(), String>
where
    DB::ErrorType: 'static,
{
    let e = |x: DrawingAreaErrorKind<DB::ErrorType>| rc_err(603, format!("could not render map: {x:?}"));
    let xs = g.polygons.iter().flat_map(|p| p.points.iter().map(|q| q.0));
    let ys = g.polygons.iter().flat_map(|p| p.points.iter().map(|q| q.1));
    let (x0, x1) = xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
    let (y0, y1) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
    // keep the aspect ratio of the map
    let (w, h) = root.dim_in_pixel();
    let (sx, sy) = ((x1 - x0).max(1e-12), (y1 - y0).max(1e-12));
    let scale = ((w as f64 * 0.9) / sx).min((h as f64 * 0.85) / sy);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let half_w = w as f64 / scale / 2.0;
    let half_h = h as f64 / scale / 2.0;
    let mut chart = ChartBuilder::on(root)
        .caption(&g.title, ("sans-serif", 18))
        .margin(5)
        .build_cartesian_2d((cx - half_w)..(cx + half_w), (cy - half_h)..(cy + half_h))
        .map_err(e)?;
    for p in &g.polygons {
        let col = RGBColor(p.fill.0, p.fill.1, p.fill.2);
        chart.draw_series(std::iter::once(Polygon::new(p.points.clone(), col.filled()))).map_err(e)?;
        chart.draw_series(std::iter::once(PathElement::new(p.points.clone(), BLACK.stroke_width(1)))).map_err(e)?;
    }
    // legend in the bottom-left corner, under its title
    if !g.map_legend_title.is_empty() && !g.map_legend.is_empty() {
        let y = (h as i32) - 20 - 18 * (g.map_legend.len() + 1) as i32;
        root.draw(&Text::new(g.map_legend_title.clone(), (10, y), ("sans-serif", 12))).map_err(e)?;
    }
    for (k, (label, c)) in g.map_legend.iter().enumerate() {
        let y = (h as i32) - 20 - 18 * (g.map_legend.len() - k) as i32;
        let col = RGBColor(c.0, c.1, c.2);
        root.draw(&Rectangle::new([(10, y), (24, y + 12)], col.filled())).map_err(e)?;
        root.draw(&Rectangle::new([(10, y), (24, y + 12)], BLACK.stroke_width(1))).map_err(e)?;
        root.draw(&Text::new(label.clone(), (30, y), ("sans-serif", 12))).map_err(e)?;
    }
    root.present().map_err(e)?;
    Ok(())
}
