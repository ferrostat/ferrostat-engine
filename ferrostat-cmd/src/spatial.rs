//! Spatial data: ESRI shapefile import (`shp2dta`), contiguity weights and
//! spatial lags (`spmat`), and choropleth maps (`spmap`).
//!
//! Implemented from the published ESRI Shapefile and dBase III formats.

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::graph::{Graph, MapPolygon};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

/// One shape: list of rings (parts), each a list of (x, y).
pub type Shape = Vec<Vec<(f64, f64)>>;

fn le_i32(b: &[u8]) -> i32 {
    i32::from_le_bytes(b[..4].try_into().unwrap())
}
fn be_i32(b: &[u8]) -> i32 {
    i32::from_be_bytes(b[..4].try_into().unwrap())
}
fn le_f64(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

/// Read polygons/polylines/points from a .shp file.
pub fn read_shp(path: &str) -> Result<Vec<Shape>, String> {
    let data = ferrostat_core::vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    if data.len() < 100 || be_i32(&data[0..]) != 9994 {
        return Err(rc_err(610, format!("{path} is not a shapefile")));
    }
    let mut pos = 100;
    let mut shapes = Vec::new();
    while pos + 8 <= data.len() {
        let len = be_i32(&data[pos + 4..]) as usize * 2;
        let c = &data[pos + 8..(pos + 8 + len).min(data.len())];
        pos += 8 + len;
        if c.len() < 4 {
            break;
        }
        let ty = le_i32(c);
        match ty {
            0 => shapes.push(Vec::new()),
            1 | 11 | 21 => shapes.push(vec![vec![(le_f64(&c[4..]), le_f64(&c[12..]))]]),
            3 | 5 | 13 | 15 | 23 | 25 => {
                let nparts = le_i32(&c[36..]) as usize;
                let npoints = le_i32(&c[40..]) as usize;
                let parts: Vec<usize> = (0..nparts).map(|k| le_i32(&c[44 + 4 * k..]) as usize).collect();
                let pts_at = 44 + 4 * nparts;
                let pt = |k: usize| (le_f64(&c[pts_at + 16 * k..]), le_f64(&c[pts_at + 16 * k + 8..]));
                let mut rings = Vec::new();
                for (k, &s) in parts.iter().enumerate() {
                    let e = parts.get(k + 1).copied().unwrap_or(npoints);
                    rings.push((s..e).map(pt).collect());
                }
                shapes.push(rings);
            }
            8 | 18 | 28 => {
                let npoints = le_i32(&c[36..]) as usize;
                shapes.push((0..npoints).map(|k| vec![(le_f64(&c[40 + 16 * k..]), le_f64(&c[48 + 16 * k..]))]).collect());
            }
            other => return Err(rc_err(610, format!("unsupported shape type {other}"))),
        }
    }
    Ok(shapes)
}

/// Read a dBase table: (field names, rows of text values, numeric flags).
pub fn read_dbf(path: &str) -> Result<(Vec<String>, Vec<bool>, Vec<Vec<String>>), String> {
    let data = ferrostat_core::vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    if data.len() < 32 {
        return Err(rc_err(610, format!("{path} is not a dBase file")));
    }
    let nrec = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let hlen = u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize;
    let rlen = u16::from_le_bytes(data[10..12].try_into().unwrap()) as usize;
    let mut fields: Vec<(String, u8, usize)> = Vec::new();
    let mut p = 32;
    while p + 32 <= data.len() && data[p] != 0x0D {
        let name: String = data[p..p + 11].iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
        fields.push((name, data[p + 11], data[p + 16] as usize));
        p += 32;
    }
    let mut rows = Vec::with_capacity(nrec);
    for r in 0..nrec {
        let start = hlen + r * rlen + 1;
        if start + rlen - 1 > data.len() {
            break;
        }
        let mut off = start;
        let mut row = Vec::new();
        for (_, _, w) in &fields {
            let raw = &data[off..off + w];
            // dBase files are commonly Latin-1; fall back when not UTF-8
            let s = match std::str::from_utf8(raw) {
                Ok(s) => s.to_string(),
                Err(_) => raw.iter().map(|&b| b as char).collect(),
            };
            row.push(s.trim().to_string());
            off += w;
        }
        rows.push(row);
    }
    let numeric = fields.iter().map(|f| matches!(f.1, b'N' | b'F' | b'O' | b'I' | b'B')).collect();
    Ok((fields.into_iter().map(|f| f.0).collect(), numeric, rows))
}

fn with_dta(p: &str) -> String {
    if p.ends_with(".dta") { p.to_string() } else { format!("{p}.dta") }
}

fn write_dta(ds: &DataSet, path: &str, replace: bool) -> Result<(), String> {
    if ferrostat_core::vfs::exists(path) && !replace {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut buf = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut buf, ds).map_err(|e| rc_err(693, e))?;
    ferrostat_core::vfs::write(path, &buf).map_err(|e| rc_err(603, e))
}

/// Area-weighted centroid of the largest ring.
fn centroid(shape: &Shape) -> (f64, f64) {
    let mut best = (0.0f64, (SYSMISS, SYSMISS));
    for ring in shape {
        let n = ring.len();
        if n == 0 {
            continue;
        }
        let mut a = 0.0;
        let (mut cx, mut cy) = (0.0, 0.0);
        for i in 0..n {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % n];
            let cr = x0 * y1 - x1 * y0;
            a += cr;
            cx += (x0 + x1) * cr;
            cy += (y0 + y1) * cr;
        }
        let c = if a.abs() > 1e-300 {
            (cx / (3.0 * a), cy / (3.0 * a))
        } else {
            (ring.iter().map(|p| p.0).sum::<f64>() / n as f64, ring.iter().map(|p| p.1).sum::<f64>() / n as f64)
        };
        if a.abs() >= best.0 {
            best = (a.abs(), c);
        }
    }
    best.1
}

pub struct Shp2dta;

impl CommandHandler for Shp2dta {
    fn name(&self) -> &'static str {
        "shp2dta"
    }

    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let shp = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let base = shp.strip_suffix(".shp").unwrap_or(&shp).to_string();
        let shp_path = format!("{base}.shp");
        let shapes = read_shp(&shp_path)?;
        let (fields, numeric, rows) = read_dbf(&format!("{base}.dbf"))?;
        let db = with_dta(&util::unquote(cmd.opt("database", 1).ok_or_else(|| rc_err(198, "option database() required"))?));
        let coord = with_dta(&util::unquote(cmd.opt("coordinates", 1).ok_or_else(|| rc_err(198, "option coordinates() required"))?));
        let replace = cmd.has_opt("replace", 3);
        let genid = cmd.opt("genid", 4).map(|s| s.trim().to_string());
        // database
        let mut d = DataSet::new();
        for (k, name) in fields.iter().enumerate() {
            let cells: Vec<String> = rows.iter().map(|r| r[k].clone()).collect();
            let clean = crate::varlist::check_name(name).map(|_| name.clone()).unwrap_or_else(|_| format!("v{}", k + 1));
            let var = if numeric[k] && cells.iter().all(|c| c.is_empty() || c.parse::<f64>().is_ok()) {
                let vals: Vec<f64> = cells.iter().map(|c| c.parse().unwrap_or(SYSMISS)).collect();
                let ty = util::best_numeric_type(&vals);
                let mut v = Variable::new(&clean, ty);
                v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
                v
            } else {
                let w = cells.iter().map(|c| c.len()).max().unwrap_or(1).max(1);
                let mut v = Variable::new(&clean, util::str_type_for(w));
                v.data = cells.into_iter().map(Value::Str).collect();
                v
            };
            d.add_variable(var)?;
        }
        d.n_obs = rows.len();
        if let Some(g) = &genid {
            let mut v = Variable::new(g, VarType::Int);
            v.data = (1..=d.n_obs as i16).map(Value::Int).collect();
            d.add_variable(v)?;
        }
        if let Some(stub) = cmd.opt("gencentroids", 4) {
            let cs: Vec<(f64, f64)> = shapes.iter().map(centroid).collect();
            d.add_variable(util::new_double(&format!("x_{}", stub.trim()), cs.iter().map(|c| c.0).collect()))?;
            d.add_variable(util::new_double(&format!("y_{}", stub.trim()), cs.iter().map(|c| c.1).collect()))?;
        }
        write_dta(&d, &db, replace)?;
        // coordinates: _ID _X _Y, each ring preceded by a separator row
        let (mut id, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new());
        for (k, s) in shapes.iter().enumerate() {
            for ring in s {
                id.push((k + 1) as f64);
                xs.push(SYSMISS);
                ys.push(SYSMISS);
                for &(x, y) in ring {
                    id.push((k + 1) as f64);
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        let mut c = DataSet::new();
        c.add_variable(util::new_double("_ID", id))?;
        c.add_variable(util::new_double("_X", xs))?;
        c.add_variable(util::new_double("_Y", ys))?;
        c.n_obs = c.vars[0].data.len();
        write_dta(&c, &coord, replace)?;
        Ok(format!("type: {}\n   file {} created\n   file {} created", if shapes.iter().any(|s| s.len() > 1 || s.first().map_or(false, |r| r.len() > 1)) { "polygon" } else { "point" }, db, coord))
    }
}

/// Load polygons by id from a coordinates dataset (`_ID _X _Y`).
pub fn load_coordinates(path: &str) -> Result<BTreeMap<i64, Shape>, String> {
    let c = ferrostat_core::dta::DtaReader::open(with_dta(path)).map_err(|e| rc_err(601, format!("{path}: {e}")))?;
    let id = util::col(&c, "_ID")?;
    let x = util::col(&c, "_X")?;
    let y = util::col(&c, "_Y")?;
    let mut out: BTreeMap<i64, Shape> = BTreeMap::new();
    for i in 0..c.n_obs {
        let s = out.entry(id[i] as i64).or_default();
        if is_missing(x[i]) || is_missing(y[i]) || s.is_empty() {
            s.push(Vec::new());
        }
        if !is_missing(x[i]) && !is_missing(y[i]) {
            s.last_mut().unwrap().push((x[i], y[i]));
        }
    }
    for s in out.values_mut() {
        s.retain(|r| !r.is_empty());
    }
    Ok(out)
}

/// Queen (shared vertex) or rook (shared edge) contiguity.
pub fn contiguity(shapes: &[&Shape], rook: bool) -> Matrix {
    let n = shapes.len();
    let key = |p: (f64, f64)| ((p.0 * 1e6).round() as i64, (p.1 * 1e6).round() as i64);
    let mut by_vertex: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut by_edge: HashMap<((i64, i64), (i64, i64)), Vec<usize>> = HashMap::new();
    for (i, s) in shapes.iter().enumerate() {
        for ring in s.iter() {
            for (k, &p) in ring.iter().enumerate() {
                by_vertex.entry(key(p)).or_default().push(i);
                let q = ring[(k + 1) % ring.len()];
                let (a, b) = (key(p), key(q));
                let e = if a <= b { (a, b) } else { (b, a) };
                if a != b {
                    by_edge.entry(e).or_default().push(i);
                }
            }
        }
    }
    let mut w = Matrix::zeros(n, n);
    let mut link = |ids: &Vec<usize>| {
        for &a in ids {
            for &b in ids {
                if a != b {
                    w[(a, b)] = 1.0;
                }
            }
        }
    };
    if rook {
        by_edge.values().for_each(&mut link);
    } else {
        by_vertex.values().for_each(&mut link);
    }
    w
}

pub struct Spmat;

impl CommandHandler for Spmat {
    fn name(&self) -> &'static str {
        "spmat"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let w: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        let sub = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
        match sub.as_str() {
            "contiguity" => {
                let name = w.get(1).ok_or_else(|| rc_err(198, "spmat contiguity name using coordfile, id(var)"))?.clone();
                if st.matrices.contains_key(&format!("__spmat_{name}")) && !cmd.has_opt("replace", 3) {
                    return Err(rc_err(110, format!("spmat object {name} already exists")));
                }
                let coord = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
                let idv = crate::varlist::resolve_name(ds, cmd.opt("id", 2).ok_or_else(|| rc_err(198, "option id() required"))?.trim())?;
                let ids = util::col(ds, &idv)?;
                let shapes = load_coordinates(&coord)?;
                let empty = Shape::new();
                let list: Vec<&Shape> = ids.iter().map(|i| shapes.get(&(*i as i64)).unwrap_or(&empty)).collect();
                let mut m = contiguity(&list, cmd.has_opt("rook", 4));
                let norm = cmd.opt("normalize", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "none".into());
                match norm.as_str() {
                    "row" => {
                        for i in 0..m.rows {
                            let s: f64 = m.row(i).iter().sum();
                            if s > 0.0 {
                                for j in 0..m.cols {
                                    m[(i, j)] /= s;
                                }
                            }
                        }
                    }
                    "minmax" => {
                        let rmax = (0..m.rows).map(|i| m.row(i).iter().sum::<f64>()).fold(0.0, f64::max);
                        let cmax = (0..m.cols).map(|j| (0..m.rows).map(|i| m[(i, j)]).sum::<f64>()).fold(0.0, f64::max);
                        let d = rmax.min(cmax);
                        if d > 0.0 {
                            m = m.scale(1.0 / d);
                        }
                    }
                    "spectral" => {
                        let (vals, _) = m.sym_eigen();
                        let l = vals.iter().cloned().fold(0.0, |a: f64, b| a.max(b.abs()));
                        if l > 0.0 {
                            m = m.scale(1.0 / l);
                        }
                    }
                    _ => {}
                }
                let names: Vec<String> = ids.iter().map(|i| format!("{}", *i as i64)).collect();
                let islands = (0..m.rows).filter(|&i| m.row(i).iter().all(|v| *v == 0.0)).count();
                st.matrices.insert(format!("__spmat_{name}"), m.with_names(names.clone(), names));
                Ok(if islands > 0 { format!("{islands} island(s) without neighbours") } else { String::new() })
            }
            "lag" => {
                // spmat lag [type] newvar W var
                let mut it = w[1..].iter();
                let first = it.next().ok_or_else(|| rc_err(198, "spmat lag newvar W varname"))?;
                let (ty, newv) = match VarType::from_type_str(first) {
                    Some(t) => (t, it.next().ok_or_else(|| rc_err(198, "new variable name required"))?.clone()),
                    None => (VarType::Double, first.clone()),
                };
                let wname = it.next().ok_or_else(|| rc_err(198, "spmat object required"))?.clone();
                let var = crate::varlist::resolve_name(ds, it.next().ok_or_else(|| rc_err(198, "varname required"))?)?;
                let m = st.matrices.get(&format!("__spmat_{wname}")).ok_or_else(|| rc_err(111, format!("spmat object {wname} not found")))?;
                if m.rows != ds.n_obs {
                    return Err(rc_err(198, format!("spmat object {wname} is {}x{} but the data have {} observations", m.rows, m.cols, ds.n_obs)));
                }
                let x = util::col(ds, &var)?;
                if x.iter().any(|v| is_missing(*v)) {
                    return Err(rc_err(416, format!("missing values encountered in {var}")));
                }
                let lag: Vec<f64> = (0..m.rows).map(|i| m.row(i).iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
                let mut v = Variable::new(&newv, ty);
                v.data = lag.iter().map(|&x| crate::data::store_num(ty, x)).collect();
                ds.add_variable(v)?;
                Ok(String::new())
            }
            "summarize" | "sum" => {
                let name = w.get(1).cloned().unwrap_or_default();
                let m = st.matrices.get(&format!("__spmat_{name}")).ok_or_else(|| rc_err(111, format!("spmat object {name} not found")))?;
                let links: Vec<f64> = (0..m.rows).map(|i| m.row(i).iter().filter(|v| **v != 0.0).count() as f64).collect();
                let (rows, cols) = (m.rows, m.cols);
                let r = st.begin_r();
                r.set("n", rows as f64);
                r.set("l_min", links.iter().cloned().fold(f64::INFINITY, f64::min));
                r.set("l_max", links.iter().cloned().fold(0.0, f64::max));
                r.set("l_mean", util::mean(&links));
                Ok(format!("Weighting matrix {name}: {} x {}, links per unit: min {} mean {:.3} max {}", rows, cols, links.iter().cloned().fold(f64::INFINITY, f64::min), util::mean(&links), links.iter().cloned().fold(0.0, f64::max)))
            }
            "drop" => {
                for n in &w[1..] {
                    st.matrices.remove(&format!("__spmat_{n}"));
                }
                Ok(String::new())
            }
            "export" | "getmatrix" => {
                let name = w.get(1).cloned().unwrap_or_default();
                let target = w.get(2).cloned().unwrap_or_else(|| name.clone());
                let m = st.matrices.get(&format!("__spmat_{name}")).cloned().ok_or_else(|| rc_err(111, format!("spmat object {name} not found")))?;
                st.matrices.insert(target, m);
                Ok(String::new())
            }
            _ => Err(rc_err(198, format!("spmat {sub} not supported"))),
        }
    }
}

/// Stata gray scale gs0 (black) .. gs16 (white) and a few named colours.
pub fn color_rgb(name: &str) -> (u8, u8, u8) {
    let n = name.trim().to_lowercase();
    if let Some(g) = n.strip_prefix("gs") {
        if let Ok(k) = g.parse::<f64>() {
            let v = (k.clamp(0.0, 16.0) / 16.0 * 255.0).round() as u8;
            return (v, v, v);
        }
    }
    match n.as_str() {
        "white" => (255, 255, 255),
        "black" => (0, 0, 0),
        "navy" => (26, 71, 111),
        "maroon" => (144, 53, 59),
        "cranberry" => (144, 53, 59),
        "red" => (220, 30, 30),
        "blue" => (30, 60, 200),
        "green" => (40, 140, 60),
        "orange" => (227, 126, 0),
        "yellow" => (240, 220, 40),
        "gray" | "grey" => (128, 128, 128),
        _ => (160, 160, 160),
    }
}

pub struct Spmap;

impl CommandHandler for Spmap {
    fn name(&self) -> &'static str {
        "spmap"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let var = crate::varlist::resolve_name(ds, cmd.main.trim())?;
        let coord = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
        let idv = crate::varlist::resolve_name(ds, cmd.opt("id", 2).ok_or_else(|| rc_err(198, "option id() required"))?.trim())?;
        let mask = util::touse(ds, st, cmd)?;
        let ids = util::col(ds, &idv)?;
        let vals = util::col(ds, &var)?;
        let shapes = load_coordinates(&coord)?;
        let k = cmd.opt("clnumber", 3).and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(4).max(1);
        let method = cmd.opt("clmethod", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "quantile".into());
        let mut obs: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(vals[i])).map(|i| vals[i]).collect();
        obs.sort_by(f64::total_cmp);
        if obs.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let (lo, hi) = (obs[0], *obs.last().unwrap());
        let breaks: Vec<f64> = match method.as_str() {
            "eqint" => (0..=k).map(|j| lo + (hi - lo) * j as f64 / k as f64).collect(),
            "custom" => cmd.opt("clbreaks", 3).unwrap_or("").split_whitespace().filter_map(|x| x.parse().ok()).collect(),
            "unique" => {
                let mut u = obs.clone();
                u.dedup();
                u
            }
            _ => {
                let mut b = vec![lo];
                for j in 1..k {
                    b.push(util::pctile_sorted(&obs, 100.0 * j as f64 / k as f64));
                }
                b.push(hi);
                b
            }
        };
        let nclass = if method == "unique" { breaks.len() } else { breaks.len().saturating_sub(1).max(1) };
        let colors: Vec<(u8, u8, u8)> = match cmd.opt("fcolor", 2) {
            Some(c) => c.split_whitespace().map(color_rgb).collect(),
            None => (0..nclass).map(|j| {
                let v = (230.0 - 200.0 * j as f64 / (nclass.max(2) - 1) as f64) as u8;
                (v, v, v)
            }).collect(),
        };
        let class_of = |v: f64| -> Option<usize> {
            if is_missing(v) {
                return None;
            }
            if method == "unique" {
                return breaks.iter().position(|b| *b == v);
            }
            for j in 0..nclass {
                if v <= breaks[j + 1] + 1e-12 * breaks[j + 1].abs().max(1.0) {
                    return Some(j);
                }
            }
            Some(nclass - 1)
        };
        let mut polys = Vec::new();
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            if let Some(s) = shapes.get(&(ids[i] as i64)) {
                let col = class_of(vals[i]).map(|c| colors.get(c).copied().unwrap_or((128, 128, 128))).unwrap_or((255, 255, 255));
                for ring in s {
                    polys.push(MapPolygon { points: ring.clone(), fill: col });
                }
            }
        }
        let legend: Vec<(String, (u8, u8, u8))> = (0..nclass)
            .map(|j| {
                let label = if method == "unique" {
                    crate::format::fmt_g(breaks[j], 8)
                } else {
                    format!("({},{}]", crate::format::fmt_g(breaks[j], 6), crate::format::fmt_g(breaks[j + 1], 6))
                };
                (label, colors.get(j).copied().unwrap_or((128, 128, 128)))
            })
            .collect();
        let mut g = Graph { title: cmd.opt("title", 2).map(|t| util::unquote(t.split(',').next().unwrap_or(t))).unwrap_or_default(), polygons: polys, map_legend: legend, legend: true, ..Default::default() };
        g.name = cmd.opt("name", 4).map(|s| s.split(',').next().unwrap().trim().to_string()).unwrap_or_else(|| "Graph".into());
        st.graphs.insert(g.name.clone(), g.clone());
        st.graph = Some(g);
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64) -> Shape {
        vec![vec![(x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0), (x, y)]]
    }

    #[test]
    fn queen_and_rook() {
        // 2x2 grid: diagonal squares touch only at a corner
        let s = [square(0.0, 0.0), square(1.0, 0.0), square(0.0, 1.0), square(1.0, 1.0)];
        let refs: Vec<&Shape> = s.iter().collect();
        let q = contiguity(&refs, false);
        let r = contiguity(&refs, true);
        assert_eq!(q[(0, 3)], 1.0);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(0, 1)], 1.0);
        assert_eq!(centroid(&s[0]), (0.5, 0.5));
    }
}
