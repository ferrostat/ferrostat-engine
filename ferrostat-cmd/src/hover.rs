//! Hover readouts for on-screen graphs: while a graph is drawn for a window
//! (not for `graph export`), the renderer records the screen position and
//! values of markers, line vertices and bars. Front ends hit-test the
//! pointer against this side channel instead of the drawing itself, so the
//! SVG stays plain and big scatters stay cheap (one entry per pixel).

use std::cell::RefCell;
use std::collections::HashSet;

use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind, RangeStyle};

/// A marker or line vertex: pixel position, data values, observation
/// number (1-based; 0 when the point is not an observation) and marker label.
#[derive(Debug, Clone, PartialEq)]
pub struct HoverPoint {
    pub px: i32,
    pub py: i32,
    pub x: f64,
    pub y: f64,
    pub obs: u32,
    pub label: String,
    pub series: u32,
}

/// A bar: its pixel rectangle (x0, y0, x1, y1), value and category.
#[derive(Debug, Clone, PartialEq)]
pub struct HoverBar {
    pub rect: [i32; 4],
    pub value: f64,
    pub x: f64,
    pub category: String,
    pub series: u32,
}

/// One drawn series: its name, the plot panel it is in and the axis title of x.
#[derive(Debug, Clone, PartialEq)]
pub struct HoverSeries {
    pub name: String,
    pub xname: String,
    pub panel: u32,
    /// line plots: their vertices, sorted by pixel x
    pub line: Vec<HoverPoint>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HoverMap {
    pub width: u32,
    pub height: u32,
    /// plotting areas (x0, y0, x1, y1) in pixels
    pub panels: Vec<[i32; 4]>,
    pub series: Vec<HoverSeries>,
    pub points: Vec<HoverPoint>,
    pub bars: Vec<HoverBar>,
    grid: Option<Grid>,
}

const CELL: i32 = 16;
/// Markers within this distance (pixels) of the pointer are hits.
pub const RADIUS: i32 = 8;

/// Spatial grid over the points: CELL x CELL pixel buckets of point indices.
#[derive(Debug, Clone, PartialEq)]
struct Grid {
    cols: i32,
    rows: i32,
    cells: Vec<Vec<u32>>,
}

struct Recorder {
    map: HoverMap,
    seen: HashSet<(i32, i32)>,
    panel: Option<usize>,
}

thread_local! {
    static REC: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

fn recording() -> bool {
    REC.with(|r| r.borrow().is_some())
}

/// Draw `g` with recording on: the drawing's result and the hover map.
fn with_recording<T>(w: u32, h: u32, f: impl FnOnce() -> Result<T, String>) -> Result<(T, HoverMap), String> {
    REC.with(|r| *r.borrow_mut() = Some(Recorder { map: HoverMap { width: w, height: h, ..Default::default() }, seen: HashSet::new(), panel: None }));
    let res = f();
    let rec = REC.with(|r| r.borrow_mut().take());
    let mut map = rec.map(|r| r.map).unwrap_or_default();
    map.build_grid();
    res.map(|t| (t, map))
}

/// The graph as SVG plus its hover map (for display; exports use `render_svg`).
pub fn render_svg_hover(g: &Graph, w: u32, h: u32) -> Result<(String, HoverMap), String> {
    with_recording(w, h, || crate::graph::render_svg(g, w, h))
}

/// The graph as RGB pixels plus its hover map (desktop graph window).
#[cfg(not(target_arch = "wasm32"))]
pub fn render_rgb_hover(g: &Graph, w: u32, h: u32) -> Result<(Vec<u8>, HoverMap), String> {
    with_recording(w, h, || crate::graph::render_rgb(g, w, h))
}

type Chart<'a, DB> = ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>;

/// Called by the renderer for each layer it draws (a no-op unless recording).
pub(crate) fn record<'a, DB: DrawingBackend + 'a>(chart: &Chart<'a, DB>, g: &Graph, k: usize, l: &Layer) {
    if !recording() {
        return;
    }
    REC.with(|r| {
        let mut guard = r.borrow_mut();
        let Some(rec) = guard.as_mut() else { return };
        if k == 0 || rec.panel.is_none() {
            let (xr, yr) = chart.plotting_area().get_pixel_range();
            rec.map.panels.push([xr.start, yr.start, xr.end, yr.end]);
            rec.panel = Some(rec.map.panels.len() - 1);
        }
        let series = rec.map.series.len() as u32;
        let mut s = HoverSeries { name: l.label.clone(), xname: g.xtitle.clone(), panel: rec.panel.unwrap_or(0) as u32, line: Vec::new() };
        let to_px = |p: (f64, f64)| chart.backend_coord(&p);
        let swap = |l: &Layer, x: f64, y: f64| if l.horizontal { (y, x) } else { (x, y) };
        let obs = |i: usize| l.obs.get(i).copied().unwrap_or(0);
        match l.kind {
            LayerKind::Scatter | LayerKind::Connected | LayerKind::Dot => {
                for (i, &(x, y)) in l.points.iter().enumerate() {
                    let (px, py) = to_px((x, y));
                    if rec.seen.insert((px, py)) {
                        rec.map.points.push(HoverPoint { px, py, x, y, obs: obs(i), label: l.mlabels.get(i).cloned().unwrap_or_default(), series });
                    }
                }
            }
            LayerKind::Line | LayerKind::Area => {
                let mut seen = HashSet::new();
                for (i, &(x, y)) in l.points.iter().enumerate() {
                    let (px, py) = to_px((x, y));
                    if seen.insert((px, py)) {
                        s.line.push(HoverPoint { px, py, x, y, obs: obs(i), label: String::new(), series });
                    }
                }
                s.line.sort_by_key(|p| p.px);
            }
            LayerKind::Bar => {
                let hw = l.width / 2.0 * 0.95;
                for (i, &(x, y)) in l.points.iter().enumerate() {
                    let (a, b) = (to_px((x - hw, 0.0)), to_px((x + hw, y)));
                    rec.map.bars.push(HoverBar { rect: rect(a, b), value: y, x, category: category(g, x), series });
                    let _ = i;
                }
            }
            LayerKind::Range(RangeStyle::Bar) | LayerKind::Range(RangeStyle::Spike) | LayerKind::Range(RangeStyle::Drop) => {
                let hw = if l.kind == LayerKind::Range(RangeStyle::Bar) { l.width / 2.0 } else { 0.0 };
                let mut seen = HashSet::new();
                for &(x, y1, y2) in &l.ranges {
                    let (a, b) = (to_px(swap(l, x - hw, y1)), to_px(swap(l, x + hw, y2)));
                    let mut r = rect(a, b);
                    // thin spikes get a few pixels of slack
                    if r[2] - r[0] < 4 {
                        r[0] -= 2;
                        r[2] += 2;
                    }
                    if r[3] - r[1] < 4 {
                        r[1] -= 2;
                        r[3] += 2;
                    }
                    if seen.insert(r) {
                        rec.map.bars.push(HoverBar { rect: r, value: y2 - y1, x, category: category(g, x), series });
                    }
                }
            }
            _ => return,
        }
        rec.map.series.push(s);
    });
}

fn rect(a: (i32, i32), b: (i32, i32)) -> [i32; 4] {
    [a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)]
}

/// The category label nearest to position `x` (bar charts), if any.
fn category(g: &Graph, x: f64) -> String {
    let labels = if g.xlabels.is_empty() { &g.ylabels } else { &g.xlabels };
    labels
        .iter()
        .filter(|(p, _)| (p - x).abs() < 0.5 + 1e-9)
        .min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs()))
        .map(|(_, s)| s.trim_start_matches('\n').to_string())
        .unwrap_or_default()
}

/// Numbers in the side channel: 7 significant digits, shortest form.
fn num(v: f64) -> String {
    if !v.is_finite() {
        return "null".into();
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let r: f64 = format!("{v:.6e}").parse().unwrap_or(v);
    format!("{r}")
}

fn jstr(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

impl HoverMap {
    fn build_grid(&mut self) {
        let cols = (self.width as i32 + CELL - 1) / CELL + 1;
        let rows = (self.height as i32 + CELL - 1) / CELL + 1;
        let mut cells = vec![Vec::new(); (cols * rows) as usize];
        for (i, p) in self.points.iter().enumerate() {
            let (cx, cy) = (p.px.div_euclid(CELL), p.py.div_euclid(CELL));
            if (0..cols).contains(&cx) && (0..rows).contains(&cy) {
                cells[(cy * cols + cx) as usize].push(i as u32);
            }
        }
        self.grid = Some(Grid { cols, rows, cells });
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty() && self.bars.is_empty() && self.series.iter().all(|s| s.line.is_empty())
    }

    /// The marker nearest to pixel (x, y) within `RADIUS`.
    pub fn nearest_point(&self, x: i32, y: i32) -> Option<&HoverPoint> {
        let grid = self.grid.as_ref()?;
        let mut best: Option<(i32, u32)> = None;
        let (c0, c1) = ((x - RADIUS).div_euclid(CELL), (x + RADIUS).div_euclid(CELL));
        let (r0, r1) = ((y - RADIUS).div_euclid(CELL), (y + RADIUS).div_euclid(CELL));
        for cy in r0.max(0)..=r1.min(grid.rows - 1) {
            for cx in c0.max(0)..=c1.min(grid.cols - 1) {
                for &i in &grid.cells[(cy * grid.cols + cx) as usize] {
                    let p = &self.points[i as usize];
                    let d = (p.px - x).pow(2) + (p.py - y).pow(2);
                    // the first point recorded wins ties (it was drawn first)
                    if d <= RADIUS * RADIUS && best.is_none_or(|(bd, bi)| d < bd || d == bd && i < bi) {
                        best = Some((d, i));
                    }
                }
            }
        }
        best.map(|(_, i)| &self.points[i as usize])
    }

    /// The vertex of a line plot nearest in x to the pointer (inside the
    /// line's panel); among several lines, the one closest vertically.
    pub fn nearest_line_point(&self, x: i32, y: i32) -> Option<&HoverPoint> {
        let mut best: Option<(i32, &HoverPoint)> = None;
        for s in &self.series {
            if s.line.is_empty() {
                continue;
            }
            let Some(p) = self.panels.get(s.panel as usize) else { continue };
            if x < p[0] || x > p[2] || y < p[1] || y > p[3] {
                continue;
            }
            let i = s.line.partition_point(|q| q.px < x);
            let cands = [i.checked_sub(1).and_then(|j| s.line.get(j)), s.line.get(i)];
            let Some(q) = cands.into_iter().flatten().min_by_key(|q| (q.px - x).abs()) else { continue };
            // same column: take the vertex nearest vertically among those at that px
            let col: Vec<&HoverPoint> = s.line.iter().skip(s.line.partition_point(|r| r.px < q.px)).take_while(|r| r.px == q.px).collect();
            let q = col.into_iter().min_by_key(|r| (r.py - y).abs()).unwrap_or(q);
            let d = (q.py - y).abs() + (q.px - x).abs();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, q));
            }
        }
        best.map(|(_, q)| q)
    }

    pub fn bar_at(&self, x: i32, y: i32) -> Option<&HoverBar> {
        self.bars.iter().rev().find(|b| x >= b.rect[0] && x <= b.rect[2] && y >= b.rect[1] && y <= b.rect[3])
    }

    /// Readout text for the pointer at pixel (x, y), if it is over something.
    pub fn hit(&self, x: i32, y: i32) -> Option<String> {
        if let Some(p) = self.nearest_point(x, y) {
            return Some(self.point_text(p));
        }
        if let Some(b) = self.bar_at(x, y) {
            let s = &self.series[b.series as usize];
            let mut t = String::new();
            if !b.category.is_empty() {
                t.push_str(&b.category);
                t.push_str(": ");
            }
            if !s.name.is_empty() {
                t.push_str(&s.name);
                t.push_str(" = ");
            }
            t.push_str(&fmt_g(b.value, 9));
            return Some(t);
        }
        self.nearest_line_point(x, y).map(|p| self.point_text(p))
    }

    fn point_text(&self, p: &HoverPoint) -> String {
        let s = &self.series[p.series as usize];
        let yname = if s.name.is_empty() { "y" } else { s.name.as_str() };
        let xname = if s.xname.is_empty() { "x" } else { s.xname.as_str() };
        let mut t = format!("{yname} = {}, {xname} = {}", fmt_g(p.y, 9), fmt_g(p.x, 9));
        if p.obs > 0 {
            t.push_str(&format!(" (obs {})", p.obs));
        }
        if !p.label.is_empty() {
            t.push_str(&format!(" [{}]", p.label));
        }
        t
    }

    /// Compact JSON for the browser:
    /// `{"w","h","panels":[[x0,y0,x1,y1]],"series":[{"name","x","panel","line":[[px,py,x,y,obs]]}],
    ///   "pts":[[px,py,x,y,obs,series,label?]],"bars":[[x0,y0,x1,y1,value,series,category]]}`.
    pub fn to_json(&self) -> String {
        let mut o = String::with_capacity(64 + self.points.len() * 40);
        o.push_str(&format!("{{\"w\":{},\"h\":{},\"panels\":[", self.width, self.height));
        o.push_str(&self.panels.iter().map(|p| format!("[{},{},{},{}]", p[0], p[1], p[2], p[3])).collect::<Vec<_>>().join(","));
        o.push_str("],\"series\":[");
        for (k, s) in self.series.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            o.push_str(&format!("{{\"name\":{},\"x\":{},\"panel\":{},\"line\":[", jstr(&s.name), jstr(&s.xname), s.panel));
            for (i, p) in s.line.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&format!("[{},{},{},{},{}]", p.px, p.py, num(p.x), num(p.y), p.obs));
            }
            o.push_str("]}");
        }
        o.push_str("],\"pts\":[");
        for (i, p) in self.points.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("[{},{},{},{},{},{}", p.px, p.py, num(p.x), num(p.y), p.obs, p.series));
            if !p.label.is_empty() {
                o.push(',');
                o.push_str(&jstr(&p.label));
            }
            o.push(']');
        }
        o.push_str("],\"bars\":[");
        for (i, b) in self.bars.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("[{},{},{},{},{},{},{}]", b.rect[0], b.rect[1], b.rect[2], b.rect[3], num(b.value), b.series, jstr(&b.category)));
        }
        o.push_str("]}");
        o
    }
}
