//! Spatial autoregressive data tools: `spset`, `spshape2dta`, `spbalance`,
//! `spcompress`, `spdistance`, `spgenerate`, `spmatrix` and `grmap`.
//!
//! Spatial settings are kept as dataset characteristics (`_dta[sp__ID]`,
//! `sp__ID_var`, `sp__CX`, `sp__CY`, `sp__coord_sys`, `sp__shp_dta`,
//! `sp__ver`), so they travel with saved datasets.
//!
//! Weighting matrices live in the session store: the dense matrix is kept in
//! the matrix table under `__spmatrix_<name>` (row and column names are the
//! spatial-unit IDs, sorted ascending) and its metadata in the settings table
//! under the same key as `type<TAB>normalization<TAB>note`.
//!
//! On-disk formats (all file I/O through the virtual file system):
//!
//! * `spmatrix save` / `spmatrix use` (default extension `.stswm`): a UTF-8
//!   text file.  Line 1 is `ferrostat-spmatrix 1`; lines 2-4 hold the type,
//!   the normalization and the note (empty line when there is none); line 5
//!   holds n; each of the next n lines holds the spatial-unit ID followed by
//!   the n weights of that row, separated by single spaces and written in
//!   shortest round-trip form, so a save/use cycle is exact.
//! * `spmatrix export` / `spmatrix import`: the plain interchange text
//!   format: line 1 holds n, each following line holds the ID followed by the
//!   n weights of its row.

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::spatial::{load_coordinates, read_dbf, read_shp, Shape};
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// spset settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpSet {
    pub id: String,
    pub id_var: String,
    pub cx: Option<String>,
    pub cy: Option<String>,
    /// "planar" or "latlong, miles" / "latlong, kilometers"
    pub coordsys: String,
    pub shp: Option<String>,
}

fn dta_char<'a>(ds: &'a DataSet, key: &str) -> Option<&'a String> {
    ds.characteristics.get("_dta").and_then(|m| m.get(key)).filter(|s| !s.is_empty())
}

/// Current spatial settings, if the data are spset.
pub fn spset_info(ds: &DataSet) -> Option<SpSet> {
    let id = dta_char(ds, "sp__ID")?.clone();
    Some(SpSet {
        id_var: dta_char(ds, "sp__ID_var").cloned().unwrap_or_else(|| id.clone()),
        id,
        cx: dta_char(ds, "sp__CX").cloned(),
        cy: dta_char(ds, "sp__CY").cloned(),
        coordsys: dta_char(ds, "sp__coord_sys").cloned().unwrap_or_else(|| "planar".into()),
        shp: dta_char(ds, "sp__shp_dta").cloned(),
    })
}

pub fn need_spset(ds: &DataSet) -> Result<SpSet, String> {
    spset_info(ds).ok_or_else(|| rc_err(459, "data not spset\n    Type spset to set the spatial settings."))
}

fn write_spset(ds: &mut DataSet, s: &SpSet) {
    let m = ds.characteristics.entry("_dta".into()).or_default();
    m.insert("sp__ver".into(), "1".into());
    m.insert("sp__ID".into(), s.id.clone());
    m.insert("sp__ID_var".into(), s.id_var.clone());
    m.insert("sp__CX".into(), s.cx.clone().unwrap_or_default());
    m.insert("sp__CY".into(), s.cy.clone().unwrap_or_default());
    m.insert("sp__coord_sys".into(), s.coordsys.clone());
    m.insert("sp__shp_dta".into(), s.shp.clone().unwrap_or_default());
}

fn clear_spset(ds: &mut DataSet) {
    if let Some(m) = ds.characteristics.get_mut("_dta") {
        m.retain(|k, _| !k.starts_with("sp__"));
    }
}

/// Panel variable and time variable when the data are xtset.
pub fn panel_vars(st: &Store) -> Option<(String, String)> {
    match &st.tsset {
        Some((Some(p), t)) => Some((p.clone(), t.clone())),
        _ => None,
    }
}

fn with_dta(p: &str) -> String {
    let p = util::unquote(p.trim());
    if p.ends_with(".dta") { p } else { format!("{p}.dta") }
}

fn write_dta(ds: &DataSet, path: &str, replace: bool) -> Result<(), String> {
    if ferrostat_core::vfs::exists(path) && !replace {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut buf = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut buf, ds).map_err(|e| rc_err(693, e))?;
    ferrostat_core::vfs::write(path, &buf).map_err(|e| rc_err(603, e))
}

fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// Area-weighted centroid over all rings of a shape (holes, with opposite
/// orientation, subtract); falls back to the vertex mean for degenerate shapes.
pub fn shape_centroid(shape: &Shape) -> (f64, f64) {
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    let (mut sx, mut sy, mut np) = (0.0, 0.0, 0usize);
    for ring in shape {
        let n = ring.len();
        for i in 0..n {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % n];
            let cr = x0 * y1 - x1 * y0;
            a += cr;
            cx += (x0 + x1) * cr;
            cy += (y0 + y1) * cr;
            sx += x0;
            sy += y0;
            np += 1;
        }
    }
    if a.abs() > 1e-300 {
        (cx / (3.0 * a), cy / (3.0 * a))
    } else if np > 0 {
        (sx / np as f64, sy / np as f64)
    } else {
        (SYSMISS, SYSMISS)
    }
}

fn spset_display(ds: &DataSet, st: &Store, s: &SpSet) -> String {
    let fname = if st.filename.is_empty() { "<unnamed>".to_string() } else { base_name(&st.filename) };
    let data = if panel_vars(st).is_some() { "Panel" } else { "Cross sectional" };
    let id = if s.id_var != s.id { format!("{} (equal to {})", s.id, s.id_var) } else { s.id.clone() };
    let coords = match (&s.cx, &s.cy) {
        (Some(x), Some(y)) => {
            let sys = if s.coordsys.starts_with("latlong") {
                format!("latitude-and-longitude, {}", s.coordsys.split(',').nth(1).map(|u| u.trim().to_string()).unwrap_or_else(|| "miles".into()))
            } else {
                "planar".into()
            };
            format!("{x}, {y} ({sys})")
        }
        _ => "none".into(),
    };
    let mut out = vec![
        format!("      Sp dataset: {fname}"),
        format!("Linked shapefile: {}", s.shp.clone().unwrap_or_else(|| "none".into())),
        format!("            Data: {data}"),
        format!(" Spatial-unit ID: {id}"),
    ];
    if let Some((_, t)) = panel_vars(st) {
        out.push(format!("         Time ID: {t} (see xtset)"));
    }
    out.push(format!("     Coordinates: {coords}"));
    let _ = ds;
    out.join("\n")
}

fn spset_r(st: &mut Store, s: &SpSet) {
    let r = st.begin_r();
    r.set_str("sp_ver", "1");
    r.set_str("sp_id", s.id.clone());
    r.set_str("sp_id_var", s.id_var.clone());
    r.set_str("sp_cx", s.cx.clone().unwrap_or_default());
    r.set_str("sp_cy", s.cy.clone().unwrap_or_default());
    r.set_str("sp_coord_sys", s.coordsys.clone());
    r.set_str("sp_shp_dta", s.shp.clone().unwrap_or_default());
}

/// Check that `ids` identify units (within time periods for panel data).
fn check_unique_ids(ds: &DataSet, st: &Store, ids: &[f64], var: &str) -> Result<(), String> {
    let time = panel_vars(st).and_then(|(_, t)| util::col(ds, &t).ok());
    let mut seen = std::collections::HashSet::new();
    for i in 0..ids.len() {
        if is_missing(ids[i]) {
            return Err(rc_err(459, format!("variable {var} has missing values")));
        }
        let t = time.as_ref().map(|t| t[i].to_bits()).unwrap_or(0);
        if !seen.insert((ids[i].to_bits(), t)) {
            return Err(rc_err(459, format!("variable {var} does not uniquely identify the observations")));
        }
    }
    Ok(())
}

pub struct Spset;

impl CommandHandler for Spset {
    fn name(&self) -> &'static str {
        "spset"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.has_opt("clear", 5) {
            clear_spset(ds);
            return Ok(String::new());
        }
        let main = cmd.main.trim();
        let modify = cmd.has_opt("modify", 3);
        let mut s = match spset_info(ds) {
            Some(s) => s,
            None => {
                if main.is_empty() {
                    if modify || !cmd.opts.is_empty() {
                        // `spset, modify ...` on data that already carry an _ID variable
                        if ds.var_index.contains_key("_ID") {
                            SpSet { id: "_ID".into(), id_var: "_ID".into(), cx: None, cy: None, coordsys: "planar".into(), shp: None }
                        } else {
                            return Err(rc_err(459, "data not spset"));
                        }
                    } else {
                        return Err(rc_err(459, "data not spset\n    Type spset idvar to set the spatial-unit ID."));
                    }
                } else {
                    SpSet { id: "_ID".into(), id_var: "_ID".into(), cx: None, cy: None, coordsys: "planar".into(), shp: None }
                }
            }
        };
        if !main.is_empty() {
            let var = crate::varlist::resolve_name(ds, main)?;
            let ids = util::col(ds, &var)?;
            if ids.iter().any(|v| !is_missing(*v) && v.fract() != 0.0) {
                return Err(rc_err(459, format!("variable {var} is not integer valued")));
            }
            check_unique_ids(ds, st, &ids, &var)?;
            if var != "_ID" {
                if ds.var_index.contains_key("_ID") {
                    ds.drop_variable("_ID")?;
                }
                let ty = util::best_numeric_type(&ids);
                let mut v = Variable::new("_ID", ty);
                v.data = ids.iter().map(|&x| util::typed_value(ty, x)).collect();
                ds.add_variable(v)?;
            }
            s.id = "_ID".into();
            s.id_var = var;
        }
        if let Some(c) = cmd.opt("coord", 5) {
            let names = crate::varlist::expand(ds, c)?;
            if names.len() != 2 {
                return Err(rc_err(198, "option coord() requires two variables"));
            }
            s.cx = Some(names[0].clone());
            s.cy = Some(names[1].clone());
        }
        if let Some(c) = cmd.opt("coordsys", 6) {
            let c = c.to_lowercase();
            let mut parts = c.split(',').map(|p| p.trim());
            s.coordsys = match parts.next().unwrap_or("") {
                "planar" => "planar".into(),
                l if l.starts_with("latlong") => {
                    let unit = match parts.next() {
                        Some(u) if u.starts_with("kilo") || u == "km" => "kilometers",
                        _ => "miles",
                    };
                    format!("latlong, {unit}")
                }
                other => return Err(rc_err(198, format!("coordsys({other}) invalid"))),
            };
        }
        if let Some(f) = cmd.opt("shpfile", 3) {
            let path = with_dta(f);
            let shapes = load_coordinates(&path)?;
            let ids = util::col(ds, &s.id)?;
            let missing = ids.iter().filter(|i| !shapes.contains_key(&(**i as i64))).count();
            if missing > 0 {
                return Err(rc_err(459, format!("{missing} spatial units in the data not found in {path}")));
            }
            if s.cx.is_none() {
                let mut cx = Vec::with_capacity(ds.n_obs);
                let mut cy = Vec::with_capacity(ds.n_obs);
                for i in ids {
                    let c = shape_centroid(&shapes[&(i as i64)]);
                    cx.push(c.0);
                    cy.push(c.1);
                }
                for (n, v) in [("_CX", cx), ("_CY", cy)] {
                    if ds.var_index.contains_key(n) {
                        ds.drop_variable(n)?;
                    }
                    ds.add_variable(util::new_double(n, v))?;
                }
                s.cx = Some("_CX".into());
                s.cy = Some("_CY".into());
            }
            s.shp = Some(base_name(&path));
        }
        if cmd.has_opt("noshpfile", 5) {
            s.shp = None;
        }
        if s.cx.is_none() && ds.var_index.contains_key("_CX") && ds.var_index.contains_key("_CY") {
            s.cx = Some("_CX".into());
            s.cy = Some("_CY".into());
        }
        write_spset(ds, &s);
        let out = spset_display(ds, st, &s);
        spset_r(st, &s);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// spshape2dta
// ---------------------------------------------------------------------------

pub struct Spshape2dta;

impl CommandHandler for Spshape2dta {
    fn name(&self) -> &'static str {
        "spshape2dta"
    }

    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = util::unquote(cmd.main.trim());
        if name.is_empty() {
            return Err(rc_err(198, "shapefile name required"));
        }
        let base = name.strip_suffix(".shp").or_else(|| name.strip_suffix(".zip")).unwrap_or(&name).to_string();
        let replace = cmd.has_opt("replace", 3);
        let out_base = cmd.opt("saving", 3).map(|s| util::unquote(s.split(',').next().unwrap_or("").trim())).unwrap_or_else(|| base.clone());
        let out_base = out_base.strip_suffix(".dta").unwrap_or(&out_base).to_string();
        let shapes = read_shp(&format!("{base}.shp"))?;
        let (fields, numeric, rows) = read_dbf(&format!("{base}.dbf"))?;
        let mut lines = vec!["  (importing .shp file)".to_string(), "  (importing .dbf file)".into(), "  (creating _ID spatial-unit id)".into(), "  (creating _CX coordinate)".into(), "  (creating _CY coordinate)".into(), String::new()];
        // attribute data
        let mut d = DataSet::new();
        let n = rows.len();
        let cs: Vec<(f64, f64)> = (0..n).map(|k| shapes.get(k).map(shape_centroid).unwrap_or((SYSMISS, SYSMISS))).collect();
        let mut idv = Variable::new("_ID", VarType::Long);
        idv.data = (1..=n as i32).map(Value::Long).collect();
        d.add_variable(idv)?;
        d.add_variable(util::new_double("_CX", cs.iter().map(|c| c.0).collect()))?;
        d.add_variable(util::new_double("_CY", cs.iter().map(|c| c.1).collect()))?;
        for (k, fname) in fields.iter().enumerate() {
            let cells: Vec<String> = rows.iter().map(|r| r[k].clone()).collect();
            let mut clean = fname.clone();
            if crate::varlist::check_name(&clean).is_err() || d.var_index.contains_key(&clean) {
                clean = format!("v{}", k + 1);
            }
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
        d.n_obs = n;
        let shp_file = format!("{out_base}_shp.dta");
        let dta_file = format!("{out_base}.dta");
        write_spset(&mut d, &SpSet { id: "_ID".into(), id_var: "_ID".into(), cx: Some("_CX".into()), cy: Some("_CY".into()), coordsys: "planar".into(), shp: Some(base_name(&shp_file)) });
        d.sort_by = vec!["_ID".into()];
        // coordinates: one separator row (missing _X/_Y) before each ring
        let (mut id, mut xs, mut ys, mut ord) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (k, s) in shapes.iter().enumerate().take(n.max(shapes.len())) {
            let mut o = 0.0;
            for ring in s {
                o += 1.0;
                id.push((k + 1) as f64);
                xs.push(SYSMISS);
                ys.push(SYSMISS);
                ord.push(o);
                for &(x, y) in ring {
                    o += 1.0;
                    id.push((k + 1) as f64);
                    xs.push(x);
                    ys.push(y);
                    ord.push(o);
                }
            }
        }
        let mut c = DataSet::new();
        let mut v = Variable::new("_ID", VarType::Long);
        v.data = id.iter().map(|&x| Value::Long(x as i32)).collect();
        c.add_variable(v)?;
        c.add_variable(util::new_double("_X", xs))?;
        c.add_variable(util::new_double("_Y", ys))?;
        let mut v = Variable::new("shape_order", VarType::Long);
        v.data = ord.iter().map(|&x| Value::Long(x as i32)).collect();
        c.add_variable(v)?;
        c.n_obs = id.len();
        c.sort_by = vec!["_ID".into()];
        write_dta(&c, &shp_file, replace)?;
        write_dta(&d, &dta_file, replace)?;
        lines.push(format!("  file {shp_file} created"));
        lines.push(format!("  file {dta_file} created"));
        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// spbalance, spcompress, spdistance
// ---------------------------------------------------------------------------

fn keep_rows(ds: &mut DataSet, keep: &[bool]) {
    for v in ds.vars.iter_mut() {
        let old = std::mem::take(&mut v.data);
        v.data = old.into_iter().zip(keep).filter(|(_, k)| **k).map(|(x, _)| x).collect();
    }
    ds.n_obs = keep.iter().filter(|k| **k).count();
}

pub struct Spbalance;

impl CommandHandler for Spbalance {
    fn name(&self) -> &'static str {
        "spbalance"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (p, t) = panel_vars(st).ok_or_else(|| rc_err(459, "data not xtset"))?;
        let pid = util::col(ds, &p)?;
        let tv = util::col(ds, &t)?;
        let times: std::collections::BTreeSet<u64> = tv.iter().filter(|v| !is_missing(**v)).map(|v| v.to_bits()).collect();
        let mut count: HashMap<u64, usize> = HashMap::new();
        for i in 0..ds.n_obs {
            if !is_missing(tv[i]) {
                *count.entry(pid[i].to_bits()).or_default() += 1;
            }
        }
        let full = |i: usize| count.get(&pid[i].to_bits()).copied().unwrap_or(0) == times.len() && !is_missing(tv[i]);
        let balanced = (0..ds.n_obs).all(full);
        let r = st.begin_r();
        r.set("balanced", if balanced { 1.0 } else { 0.0 });
        if balanced {
            return Ok("  (data strongly balanced)".into());
        }
        if !cmd.has_opt("balance", 3) {
            return Ok("  (data not strongly balanced)".into());
        }
        let keep: Vec<bool> = (0..ds.n_obs).map(full).collect();
        let dropped = keep.iter().filter(|k| !**k).count();
        let units: std::collections::HashSet<u64> = (0..ds.n_obs).filter(|&i| !keep[i]).map(|i| pid[i].to_bits()).collect();
        keep_rows(ds, &keep);
        st.data_changed = true;
        Ok(format!("  ({} observations, {} panels dropped)\n  (data now strongly balanced)", dropped, units.len()))
    }
}

pub struct Spcompress;

impl CommandHandler for Spcompress {
    fn name(&self) -> &'static str {
        "spcompress"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut s = need_spset(ds)?;
        let shp = s.shp.clone().ok_or_else(|| rc_err(459, "no shapefile linked"))?;
        let c = ferrostat_core::dta::DtaReader::open(&shp).map_err(|e| rc_err(601, format!("{shp}: {e}")))?;
        let ids: std::collections::HashSet<i64> = util::col(ds, &s.id)?.iter().map(|v| *v as i64).collect();
        let cid = util::col(&c, "_ID")?;
        let keep: Vec<bool> = cid.iter().map(|v| ids.contains(&(*v as i64))).collect();
        let mut c2 = c.clone();
        keep_rows(&mut c2, &keep);
        let stem = if st.filename.is_empty() { "spdata".to_string() } else { st.filename.trim_end_matches(".dta").to_string() };
        let target = format!("{stem}_shp.dta");
        let _ = cmd;
        write_dta(&c2, &target, true)?;
        s.shp = Some(base_name(&target));
        write_spset(ds, &s);
        Ok(format!("  (shapefile {} saved)\n  (data linked to {})", base_name(&target), base_name(&target)))
    }
}

/// Great-circle distance between (lon, lat) points in degrees.
pub fn great_circle(lon1: f64, lat1: f64, lon2: f64, lat2: f64, km: bool) -> f64 {
    let r = if km { 6371.0088 } else { 3958.7613 };
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().min(1.0).asin()
}

/// Distance function for the current coordinate system.
pub fn distance_fn(s: &SpSet) -> impl Fn((f64, f64), (f64, f64)) -> f64 {
    let latlong = s.coordsys.starts_with("latlong");
    let km = s.coordsys.contains("kilo");
    move |a: (f64, f64), b: (f64, f64)| {
        if latlong {
            great_circle(a.0, a.1, b.0, b.1, km)
        } else {
            ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
        }
    }
}

pub struct Spdistance;

impl CommandHandler for Spdistance {
    fn name(&self) -> &'static str {
        "spdistance"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let s = need_spset(ds)?;
        let (cxn, cyn) = match (&s.cx, &s.cy) {
            (Some(x), Some(y)) => (x.clone(), y.clone()),
            _ => return Err(rc_err(459, "no coordinates set; see spset, modify coord()")),
        };
        let w: Vec<f64> = cmd.main.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| rc_err(198, "spdistance #1 #2"))).collect::<Result<_, _>>()?;
        if w.len() != 2 {
            return Err(rc_err(198, "spdistance #1 #2"));
        }
        let ids = util::col(ds, &s.id)?;
        let cx = util::col(ds, &cxn)?;
        let cy = util::col(ds, &cyn)?;
        let find = |v: f64| ids.iter().position(|i| *i == v).ok_or_else(|| rc_err(459, format!("{} = {} not found", s.id, fmt_g(v, 9))));
        let (a, b) = (find(w[0])?, find(w[1])?);
        let d = distance_fn(&s)((cx[a], cy[a]), (cx[b], cy[b]));
        let unit = if s.coordsys.starts_with("latlong") { if s.coordsys.contains("kilo") { " kilometers" } else { " miles" } } else { "" };
        let mut out = Vec::new();
        for (lab, k) in [("(A)", a), ("(B)", b)] {
            out.push(format!("{lab} {} = {:>10}", s.id, fmt_g(ids[k], 9)));
            out.push(format!("    {:>4} = {:>10}", cxn, fmt_g(cx[k], 9)));
            out.push(format!("    {:>4} = {:>10}", cyn, fmt_g(cy[k], 9)));
        }
        out.push(format!("distance = {:>10}{unit}", fmt_g(d, 9)));
        let r = st.begin_r();
        r.set("distance", d);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Weighting-matrix objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpMat {
    pub ids: Vec<f64>,
    pub w: Matrix,
    pub kind: String,
    pub norm: String,
    pub note: String,
}

fn key(name: &str) -> String {
    format!("__spmatrix_{name}")
}

pub fn get_spmat(st: &Store, name: &str) -> Result<SpMat, String> {
    let m = st.matrices.get(&key(name)).ok_or_else(|| rc_err(498, format!("weighting matrix {name} not found")))?;
    let meta = st.settings.get(&key(name)).cloned().unwrap_or_default();
    let mut it = meta.splitn(3, '\t');
    let kind = it.next().unwrap_or("custom").to_string();
    let norm = it.next().unwrap_or("none").to_string();
    let note = it.next().unwrap_or("").to_string();
    let ids = m.rownames.iter().map(|s| s.parse::<f64>().unwrap_or(SYSMISS)).collect();
    Ok(SpMat { ids, w: m.clone(), kind, norm, note })
}

pub fn put_spmat(st: &mut Store, name: &str, m: SpMat) {
    let names: Vec<String> = m.ids.iter().map(|i| fmt_id(*i)).collect();
    st.settings.insert(key(name), format!("{}\t{}\t{}", m.kind, m.norm, m.note));
    st.matrices.insert(key(name), m.w.with_names(names.clone(), names));
}

fn fmt_id(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{v}") }
}

fn spmat_names(st: &Store) -> Vec<String> {
    st.matrices.keys().filter_map(|k| k.strip_prefix("__spmatrix_").map(String::from)).collect()
}

fn check_new(st: &Store, name: &str, cmd: &Command) -> Result<(), String> {
    crate::varlist::check_name(name)?;
    if st.matrices.contains_key(&key(name)) && !cmd.has_opt("replace", 3) {
        return Err(rc_err(110, format!("weighting matrix {name} already exists")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Eigenvalues
// ---------------------------------------------------------------------------

/// Eigenvalues of a real symmetric matrix (Householder tridiagonalization
/// followed by implicit QL), unsorted.
pub fn sym_eigenvalues(m: &Matrix) -> Vec<f64> {
    let n = m.rows;
    if n == 0 {
        return Vec::new();
    }
    let mut a: Vec<f64> = (0..n * n).map(|k| m[(k / n, k % n)]).collect();
    let at = |i: usize, j: usize| i * n + j;
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| a[at(i, k)].abs()).sum();
            if scale == 0.0 {
                e[i] = a[at(i, l)];
            } else {
                for k in 0..=l {
                    a[at(i, k)] /= scale;
                    h += a[at(i, k)] * a[at(i, k)];
                }
                let f = a[at(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[at(i, l)] = f - g;
                let mut f = 0.0;
                for j in 0..=l {
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += a[at(j, k)] * a[at(i, k)];
                    }
                    for k in j + 1..=l {
                        g += a[at(k, j)] * a[at(i, k)];
                    }
                    e[j] = g / h;
                    f += e[j] * a[at(i, j)];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    let f = a[at(i, j)];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[at(j, k)] -= f * e[k] + g * a[at(i, k)];
                    }
                }
            }
        } else {
            e[i] = a[at(i, l)];
        }
        d[i] = h;
    }
    for i in 0..n {
        d[i] = a[at(i, i)];
    }
    // implicit QL
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m as isize - 1;
            let mut early = false;
            while i >= l as isize {
                let iu = i as usize;
                let f = s * e[iu];
                let b = c * e[iu];
                r = f.hypot(g);
                e[iu + 1] = r;
                if r == 0.0 {
                    d[iu + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[iu + 1] - p;
                r = (d[iu] - g) * s + 2.0 * c * b;
                p = s * r;
                d[iu + 1] = g + p;
                g = c * r - b;
                i -= 1;
            }
            if early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d
}

fn is_symmetric(m: &Matrix) -> bool {
    let n = m.rows;
    let scale = m.data.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-300);
    for i in 0..n {
        for j in i + 1..n {
            if (m[(i, j)] - m[(j, i)]).abs() > 1e-12 * scale {
                return false;
            }
        }
    }
    true
}

/// D^{1/2} W D^{-1/2} when D W is symmetric for some positive diagonal D.
fn symmetrize(m: &Matrix) -> Option<Matrix> {
    let n = m.rows;
    let mut d = vec![0.0f64; n];
    for root in 0..n {
        if d[root] != 0.0 {
            continue;
        }
        d[root] = 1.0;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            for j in 0..n {
                let (a, b) = (m[(i, j)], m[(j, i)]);
                if a == 0.0 && b == 0.0 {
                    continue;
                }
                if a == 0.0 || b == 0.0 || a / b <= 0.0 {
                    return None;
                }
                let dj = d[i] * a / b;
                if d[j] == 0.0 {
                    d[j] = dj;
                    stack.push(j);
                } else if ((d[j] - dj) / dj).abs() > 1e-9 {
                    return None;
                }
            }
        }
    }
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if m[(i, j)] != 0.0 {
                s[(i, j)] = m[(i, j)] * (d[i] / d[j]).sqrt();
            }
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let v = 0.5 * (s[(i, j)] + s[(j, i)]);
            s[(i, j)] = v;
            s[(j, i)] = v;
        }
    }
    Some(s)
}

/// Eigenvalues (real, imaginary) of a weighting matrix.
pub fn eigenvalues(m: &Matrix) -> Vec<(f64, f64)> {
    if is_symmetric(m) {
        return sym_eigenvalues(m).into_iter().map(|v| (v, 0.0)).collect();
    }
    if let Some(s) = symmetrize(m) {
        return sym_eigenvalues(&s).into_iter().map(|v| (v, 0.0)).collect();
    }
    crate::tsla::eig_general(m)
}

/// ln |I - a W| from the eigenvalues of W.
pub fn logdet(eig: &[(f64, f64)], a: f64) -> f64 {
    eig.iter().map(|&(re, im)| ((1.0 - a * re).powi(2) + (a * im).powi(2)).sqrt().ln()).sum()
}

/// d/da ln|I - aW| = -sum w/(1 - a w) (real part).
pub fn dlogdet(eig: &[(f64, f64)], a: f64) -> f64 {
    eig.iter()
        .map(|&(re, im)| {
            // w / (1 - a w) for complex w
            let (dr, di) = (1.0 - a * re, -a * im);
            let den = dr * dr + di * di;
            -((re * dr + im * di) / den)
        })
        .sum()
}

/// Admissible range (1/w_min, 1/w_max) of the autoregressive parameter.
pub fn param_bounds(eig: &[(f64, f64)]) -> (f64, f64) {
    let real: Vec<f64> = eig.iter().filter(|e| e.1.abs() < 1e-10).map(|e| e.0).collect();
    let max = real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = real.iter().cloned().fold(f64::INFINITY, f64::min);
    let modmax = eig.iter().map(|e| e.0.hypot(e.1)).fold(0.0, f64::max).max(1e-300);
    let hi = if max > 0.0 { 1.0 / max } else { 1.0 / modmax };
    let lo = if min < 0.0 { 1.0 / min } else { -1.0 / modmax };
    (lo, hi)
}

fn spectral_radius(m: &Matrix) -> f64 {
    eigenvalues(m).iter().map(|e| e.0.hypot(e.1)).fold(0.0, f64::max)
}

pub fn normalize(m: &mut Matrix, how: &str) -> Result<(), String> {
    match how {
        "row" => {
            for i in 0..m.rows {
                let s: f64 = m.row(i).iter().sum();
                if s != 0.0 {
                    for j in 0..m.cols {
                        m[(i, j)] /= s;
                    }
                }
            }
        }
        "minmax" => {
            let rmax = (0..m.rows).map(|i| m.row(i).iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
            let cmax = (0..m.cols).map(|j| (0..m.rows).map(|i| m[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max);
            let d = rmax.min(cmax);
            if d > 0.0 {
                *m = m.scale(1.0 / d);
            }
        }
        "spectral" => {
            let l = spectral_radius(m);
            if l > 0.0 {
                *m = m.scale(1.0 / l);
            }
        }
        "none" => {}
        other => return Err(rc_err(198, format!("normalize({other}) invalid"))),
    }
    Ok(())
}

fn norm_opt(cmd: &Command) -> Result<String, String> {
    let n = cmd.opt("normalize", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "spectral".into());
    let n = match n.as_str() {
        "r" | "ro" | "row" => "row",
        "s" | "sp" | "spe" | "spec" | "spectral" => "spectral",
        "m" | "mi" | "min" | "minmax" => "minmax",
        "n" | "no" | "none" => "none",
        other => return Err(rc_err(198, format!("normalize({other}) invalid"))),
    };
    Ok(n.into())
}

/// Spatial-unit IDs of the observations in `mask`, sorted, with checks.
fn sample_ids(ds: &DataSet, st: &Store, s: &SpSet, mask: &[bool]) -> Result<Vec<(f64, usize)>, String> {
    let ids = util::col(ds, &s.id)?;
    let mut v: Vec<(f64, usize)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let panel = panel_vars(st).is_some();
    for i in 0..ds.n_obs {
        if !mask[i] || is_missing(ids[i]) {
            continue;
        }
        if seen.insert(ids[i].to_bits()) {
            v.push((ids[i], i));
        } else if !panel {
            return Err(rc_err(459, format!("{} does not uniquely identify the observations", s.id)));
        }
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(v)
}

/// Inverse-distance weights 1/d (0 on the diagonal and for coincident points).
pub fn idistance_matrix(pts: &[(f64, f64)], dist: &dyn Fn((f64, f64), (f64, f64)) -> f64, vtrunc: Option<f64>) -> Matrix {
    let n = pts.len();
    let mut w = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i + 1..n {
            let d = dist(pts[i], pts[j]);
            let mut v = if d > 0.0 { 1.0 / d } else { 0.0 };
            if let Some(t) = vtrunc {
                if v <= t {
                    v = 0.0;
                }
            }
            w[(i, j)] = v;
            w[(j, i)] = v;
        }
    }
    w
}

fn parse_num_opt(cmd: &Command, name: &str, min: usize) -> Result<Option<f64>, String> {
    match cmd.opt(name, min) {
        Some(s) => s.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("option {name}() invalid"))),
        None => Ok(None),
    }
}

fn read_text_matrix(text: &str, header_lines: usize) -> Result<(Vec<f64>, Matrix), String> {
    let mut lines = text.lines().skip(header_lines).filter(|l| !l.trim().is_empty());
    let n: usize = lines.next().and_then(|l| l.trim().parse().ok()).ok_or_else(|| rc_err(610, "invalid weighting-matrix file: n expected"))?;
    let mut ids = Vec::with_capacity(n);
    let mut w = Matrix::zeros(n, n);
    let nums: Vec<f64> = lines.flat_map(|l| l.split_whitespace().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect::<Vec<_>>()).collect();
    if nums.len() < n * (n + 1) || nums.iter().any(|v| v.is_nan()) {
        return Err(rc_err(610, "invalid weighting-matrix file"));
    }
    for i in 0..n {
        let row = &nums[i * (n + 1)..(i + 1) * (n + 1)];
        ids.push(row[0]);
        for j in 0..n {
            w[(i, j)] = row[j + 1];
        }
    }
    Ok((ids, w))
}

fn write_text_matrix(m: &SpMat) -> String {
    let n = m.ids.len();
    let mut s = format!("{n}\n");
    for i in 0..n {
        s.push_str(&fmt_id(m.ids[i]));
        for j in 0..n {
            s.push(' ');
            s.push_str(&format!("{}", m.w[(i, j)]));
        }
        s.push('\n');
    }
    s
}

fn add_ext(f: &str, ext: &str) -> String {
    let f = util::unquote(f.trim());
    let last = f.rsplit(['/', '\\']).next().unwrap_or(&f);
    if last.contains('.') { f } else { format!("{f}{ext}") }
}

pub struct Spmatrix;

impl CommandHandler for Spmatrix {
    fn name(&self) -> &'static str {
        "spmatrix"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let w: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        let sub = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
        let arg = |k: usize| w.get(k).cloned().ok_or_else(|| rc_err(198, format!("spmatrix {sub}: name required")));
        match sub.as_str() {
            "create" => {
                let kind = w.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
                let name = arg(2)?;
                check_new(st, &name, cmd)?;
                let s = need_spset(ds)?;
                let mask = util::touse(ds, st, cmd)?;
                let units = sample_ids(ds, st, &s, &mask)?;
                let ids: Vec<f64> = units.iter().map(|u| u.0).collect();
                let (mut m, kind) = if kind.starts_with("cont") {
                    let shp = s.shp.clone().ok_or_else(|| rc_err(459, "no shapefile linked; contiguity requires a shapefile\n    see spset, modify shpfile()"))?;
                    let shapes = load_coordinates(&shp)?;
                    let empty = Shape::new();
                    let missing = ids.iter().filter(|i| !shapes.contains_key(&(**i as i64))).count();
                    if missing > 0 {
                        return Err(rc_err(459, format!("{missing} spatial units not found in shapefile {shp}")));
                    }
                    let list: Vec<&Shape> = ids.iter().map(|i| shapes.get(&(*i as i64)).unwrap_or(&empty)).collect();
                    let c1 = crate::spatial::contiguity(&list, cmd.has_opt("rook", 4));
                    let first = cmd.has_opt("first", 5);
                    let second = cmd.find_second();
                    let mut m = Matrix::zeros(ids.len(), ids.len());
                    let use_first = first || second.is_none();
                    if use_first {
                        m = c1.clone();
                    }
                    if let Some(sw) = second {
                        let n = ids.len();
                        let nb: Vec<Vec<usize>> = (0..n).map(|i| (0..n).filter(|&j| c1[(i, j)] != 0.0).collect()).collect();
                        for i in 0..n {
                            for &k in &nb[i] {
                                for &j in &nb[k] {
                                    if j != i && c1[(i, j)] == 0.0 {
                                        m[(i, j)] = sw;
                                    }
                                }
                            }
                        }
                    }
                    (m, "contiguity")
                } else if kind.starts_with("idist") {
                    let (cx, cy) = match (&s.cx, &s.cy) {
                        (Some(x), Some(y)) => (util::col(ds, x)?, util::col(ds, y)?),
                        _ => return Err(rc_err(459, "no coordinates set; see spset, modify coord()")),
                    };
                    let pts: Vec<(f64, f64)> = units.iter().map(|u| (cx[u.1], cy[u.1])).collect();
                    if pts.iter().any(|p| is_missing(p.0) || is_missing(p.1)) {
                        return Err(rc_err(459, "missing coordinates"));
                    }
                    let df = distance_fn(&s);
                    (idistance_matrix(&pts, &df, parse_num_opt(cmd, "vtruncate", 2)?), "idistance")
                } else {
                    return Err(rc_err(198, format!("spmatrix create {kind} not allowed; use contiguity or idistance")));
                };
                let norm = norm_opt(cmd)?;
                normalize(&mut m, &norm)?;
                put_spmat(st, &name, SpMat { ids, w: m, kind: kind.into(), norm, note: String::new() });
                Ok(String::new())
            }
            "fromdata" => {
                // spmatrix fromdata W = v1 v2 ... [, idistance normalize() replace]
                let rest = cmd.main.trim()[sub.len()..].trim().to_string();
                let (name, vars) = rest.split_once('=').ok_or_else(|| rc_err(198, "spmatrix fromdata Wname = varlist"))?;
                let name = name.trim().to_string();
                check_new(st, &name, cmd)?;
                let s = need_spset(ds)?;
                let vars = crate::varlist::expand(ds, vars)?;
                let mask = util::touse(ds, st, cmd)?;
                let units = sample_ids(ds, st, &s, &mask)?;
                let n = units.len();
                if vars.len() != n {
                    return Err(rc_err(503, format!("{} variables specified but {n} observations (places)", vars.len())));
                }
                let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
                // rows follow the data order of the units; columns follow the variables
                let mut order: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
                order.truncate(n);
                let ids_d = util::col(ds, &s.id)?;
                let mut m = Matrix::zeros(n, n);
                let idist = cmd.has_opt("idistance", 5);
                let rank: Vec<usize> = order.iter().map(|&i| units.iter().position(|u| u.0 == ids_d[i]).unwrap()).collect();
                for (a, &i) in order.iter().enumerate() {
                    for (b, c) in cols.iter().enumerate() {
                        let mut v = c[i];
                        if is_missing(v) {
                            return Err(rc_err(416, "missing values not allowed"));
                        }
                        if idist {
                            v = if v > 0.0 { 1.0 / v } else { 0.0 };
                        }
                        m[(rank[a], rank[b])] = v;
                    }
                }
                for i in 0..n {
                    m[(i, i)] = 0.0;
                }
                let norm = norm_opt(cmd)?;
                normalize(&mut m, &norm)?;
                put_spmat(st, &name, SpMat { ids: units.iter().map(|u| u.0).collect(), w: m, kind: if idist { "idistance".into() } else { "custom".into() }, norm, note: String::new() });
                Ok(String::new())
            }
            "import" | "use" => {
                let name = arg(1)?;
                check_new(st, &name, cmd)?;
                let file = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
                let path = if sub == "use" { add_ext(&file, ".stswm") } else { util::unquote(file.trim()) };
                let bytes = ferrostat_core::vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
                let text = String::from_utf8_lossy(&bytes).to_string();
                let m = if text.starts_with("ferrostat-spmatrix") {
                    let lines: Vec<&str> = text.lines().collect();
                    let (ids, w) = read_text_matrix(&text, 4)?;
                    SpMat { ids, w, kind: lines.get(1).unwrap_or(&"custom").to_string(), norm: lines.get(2).unwrap_or(&"none").to_string(), note: lines.get(3).unwrap_or(&"").to_string() }
                } else if sub == "use" {
                    return Err(rc_err(610, format!("{path} is not a weighting-matrix file")));
                } else {
                    let (ids, w) = read_text_matrix(&text, 0)?;
                    SpMat { ids, w, kind: "custom".into(), norm: "none".into(), note: String::new() }
                };
                // keep IDs sorted
                let mut ord: Vec<usize> = (0..m.ids.len()).collect();
                ord.sort_by(|&a, &b| m.ids[a].total_cmp(&m.ids[b]));
                let n = ord.len();
                let mut w2 = Matrix::zeros(n, n);
                for (a, &i) in ord.iter().enumerate() {
                    for (b, &j) in ord.iter().enumerate() {
                        w2[(a, b)] = m.w[(i, j)];
                    }
                }
                let ids = ord.iter().map(|&i| m.ids[i]).collect();
                put_spmat(st, &name, SpMat { ids, w: w2, ..m });
                Ok(String::new())
            }
            "export" | "save" => {
                let name = arg(1)?;
                let m = get_spmat(st, &name)?;
                let file = cmd.using.clone().ok_or_else(|| rc_err(100, "using required"))?;
                let path = if sub == "save" { add_ext(&file, ".stswm") } else { util::unquote(file.trim()) };
                if ferrostat_core::vfs::exists(&path) && !cmd.has_opt("replace", 3) {
                    return Err(rc_err(602, format!("file {path} already exists")));
                }
                let text = if sub == "save" {
                    format!("ferrostat-spmatrix 1\n{}\n{}\n{}\n{}", m.kind, m.norm, m.note.replace('\n', " "), write_text_matrix(&m))
                } else {
                    write_text_matrix(&m)
                };
                ferrostat_core::vfs::write(&path, text.as_bytes()).map_err(|e| rc_err(603, e))?;
                Ok(if sub == "save" { String::new() } else { format!("  (file {path} saved)") })
            }
            "copy" => {
                let (a, b) = (arg(1)?, arg(2)?);
                let m = get_spmat(st, &a)?;
                check_new(st, &b, cmd)?;
                put_spmat(st, &b, m);
                Ok(String::new())
            }
            "drop" => {
                let name = arg(1)?;
                if !st.matrices.contains_key(&key(&name)) {
                    return Err(rc_err(498, format!("weighting matrix {name} not found")));
                }
                st.matrices.remove(&key(&name));
                st.settings.remove(&key(&name));
                Ok(String::new())
            }
            "clear" => {
                for n in spmat_names(st) {
                    st.matrices.remove(&key(&n));
                    st.settings.remove(&key(&n));
                }
                Ok(String::new())
            }
            "dir" => {
                let names = spmat_names(st);
                let mut out = vec![
                    "  ---------------------------------------------------------".to_string(),
                    "    Weighting matrix name          N   Type        Normalization".into(),
                    "  ---------------------------------------------------------".into(),
                ];
                for n in &names {
                    let m = get_spmat(st, n)?;
                    out.push(format!("    {:<24} {:>7}   {:<11} {}", n, m.ids.len(), m.kind, m.norm));
                }
                out.push("  ---------------------------------------------------------".into());
                let r = st.begin_r();
                r.set_str("names", names.join(" "));
                Ok(if names.is_empty() { "  (no weighting matrices defined)".into() } else { out.join("\n") })
            }
            "note" => {
                let rest = cmd.main.trim()[sub.len()..].trim().to_string();
                let (name, text) = match rest.split_once(':') {
                    Some((n, t)) => (n.trim().to_string(), Some(t.trim().to_string())),
                    None => (rest.trim().to_string(), None),
                };
                let mut m = get_spmat(st, &name)?;
                match text {
                    Some(t) => {
                        m.note = t;
                        put_spmat(st, &name, m);
                        Ok(String::new())
                    }
                    None => {
                        let r = st.begin_r();
                        r.set_str("note", m.note.clone());
                        Ok(if m.note.is_empty() { format!("  (weighting matrix {name} has no note)") } else { format!("  {name}: {}", m.note) })
                    }
                }
            }
            "summarize" | "sum" | "summ" | "summa" | "summar" | "summari" | "summariz" => {
                let name = arg(1)?;
                let m = get_spmat(st, &name)?;
                let n = m.ids.len();
                let mut mn = f64::INFINITY;
                let mut mn0 = f64::INFINITY;
                let mut mx = f64::NEG_INFINITY;
                for v in &m.w.data {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                    if *v > 0.0 {
                        mn0 = mn0.min(*v);
                    }
                }
                if !mn0.is_finite() {
                    mn0 = 0.0;
                }
                let nb: Vec<f64> = (0..n).map(|i| m.w.row(i).iter().filter(|v| **v != 0.0).count() as f64).collect();
                let nmin = nb.iter().cloned().fold(f64::INFINITY, f64::min);
                let nmax = nb.iter().cloned().fold(0.0, f64::max);
                let nmean = util::mean(&nb);
                let g = |v: f64| fmt_g(v, 9);
                let out = [
                    format!("Weighting matrix {name}"),
                    "---------------------------------------".into(),
                    format!("                  Type  | {:>12}", m.kind),
                    format!("         Normalization  | {:>12}", m.norm),
                    format!("             Dimension  | {:>12}", format!("{n} x {n}")),
                    "------------------------+--------------".into(),
                    "Elements                |".into(),
                    format!("  minimum               | {:>12}", g(mn)),
                    format!("  minimum > 0           | {:>12}", g(mn0)),
                    format!("  maximum               | {:>12}", g(mx)),
                    "Neighbors               |".into(),
                    format!("  minimum               | {:>12}", g(nmin)),
                    format!("  mean                  | {:>12}", g(nmean)),
                    format!("  maximum               | {:>12}", g(nmax)),
                    "---------------------------------------".into(),
                ]
                .join("\n");
                if let Some(gv) = cmd.opt("generate", 3) {
                    let gv = gv.trim().to_string();
                    crate::varlist::check_name(&gv)?;
                    let s = need_spset(ds)?;
                    let ids = util::col(ds, &s.id)?;
                    let pos: HashMap<u64, usize> = m.ids.iter().enumerate().map(|(k, v)| (v.to_bits(), k)).collect();
                    let vals: Vec<f64> = ids.iter().map(|i| pos.get(&i.to_bits()).map(|&k| nb[k]).unwrap_or(SYSMISS)).collect();
                    ds.add_variable(util::new_double(&gv, vals))?;
                }
                let r = st.begin_r();
                r.set("n", n as f64);
                r.set("min", mn);
                r.set("min0", mn0);
                r.set("max", mx);
                r.set("n_min", nmin);
                r.set("n_mean", nmean);
                r.set("n_max", nmax);
                Ok(out)
            }
            "normalize" => {
                let name = arg(1)?;
                let mut m = get_spmat(st, &name)?;
                let norm = norm_opt(cmd)?;
                normalize(&mut m.w, &norm)?;
                m.norm = norm;
                put_spmat(st, &name, m);
                Ok(String::new())
            }
            "matafromsp" => {
                // spmatrix matafromsp M id = W  (copies into session matrices M and id)
                let rest = cmd.main.trim()[sub.len()..].trim().to_string();
                let (lhs, wname) = rest.split_once('=').ok_or_else(|| rc_err(198, "spmatrix matafromsp matname idname = Wname"))?;
                let lhs: Vec<&str> = lhs.split_whitespace().collect();
                if lhs.len() != 2 {
                    return Err(rc_err(198, "spmatrix matafromsp matname idname = Wname"));
                }
                let m = get_spmat(st, wname.trim())?;
                let names: Vec<String> = m.ids.iter().map(|i| fmt_id(*i)).collect();
                st.matrices.insert(lhs[0].to_string(), m.w.clone().with_names(names.clone(), names.clone()));
                st.matrices.insert(lhs[1].to_string(), Matrix::col_vector(&m.ids).with_names(names, vec!["c1".into()]));
                Ok(String::new())
            }
            "spfrommata" => {
                // spmatrix spfrommata W = M id [, normalize() replace]
                let rest = cmd.main.trim()[sub.len()..].trim().to_string();
                let (name, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "spmatrix spfrommata Wname = matname idname"))?;
                let name = name.trim().to_string();
                check_new(st, &name, cmd)?;
                let rhs: Vec<&str> = rhs.split_whitespace().collect();
                if rhs.len() != 2 {
                    return Err(rc_err(198, "spmatrix spfrommata Wname = matname idname"));
                }
                let w = st.matrices.get(rhs[0]).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", rhs[0])))?;
                let idm = st.matrices.get(rhs[1]).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", rhs[1])))?;
                if w.rows != w.cols || idm.data.len() != w.rows {
                    return Err(rc_err(503, "conformability error"));
                }
                let mut m = SpMat { ids: idm.data.clone(), w: Matrix::zeros(w.rows, w.cols), kind: "custom".into(), norm: "none".into(), note: String::new() };
                let mut ord: Vec<usize> = (0..w.rows).collect();
                ord.sort_by(|&a, &b| m.ids[a].total_cmp(&m.ids[b]));
                for (a, &i) in ord.iter().enumerate() {
                    for (b, &j) in ord.iter().enumerate() {
                        m.w[(a, b)] = w[(i, j)];
                    }
                }
                m.ids = ord.iter().map(|&i| idm.data[i]).collect();
                let norm = norm_opt(cmd)?;
                normalize(&mut m.w, &norm)?;
                m.norm = norm;
                put_spmat(st, &name, m);
                Ok(String::new())
            }
            "userdefined" => Err(rc_err(198, "spmatrix userdefined requires a user-written weighting function; build the matrix with spmatrix fromdata or spmatrix spfrommata")),
            "" => Err(rc_err(198, "spmatrix subcommand required")),
            other => Err(rc_err(198, format!("spmatrix {other} is not a valid subcommand"))),
        }
    }
}

trait SecondOpt {
    fn find_second(&self) -> Option<f64>;
}

impl SecondOpt for Command {
    /// `second` (weight 1) or `second(#)`.
    fn find_second(&self) -> Option<f64> {
        self.opts.iter().find(|(n, _)| n.to_lowercase().starts_with("sec")).map(|(_, a)| a.as_deref().and_then(|s| s.trim().parse().ok()).unwrap_or(1.0))
    }
}

/// Align a weighting matrix with the spatial units `ids` (in that order).
/// Fails when a unit is missing; `force` allows W to contain extra units.
pub fn align(m: &SpMat, ids: &[f64], name: &str, force: bool) -> Result<Matrix, String> {
    let pos: HashMap<u64, usize> = m.ids.iter().enumerate().map(|(k, v)| (v.to_bits(), k)).collect();
    let idx: Vec<usize> = ids
        .iter()
        .map(|i| pos.get(&i.to_bits()).copied().ok_or_else(|| rc_err(459, format!("spatial unit {} not found in weighting matrix {name}", fmt_id(*i)))))
        .collect::<Result<_, _>>()?;
    if m.ids.len() > ids.len() && !force {
        return Err(rc_err(459, format!("weighting matrix {name} defines {} places but the estimation sample has {}\n    Specify option force to use the subset of W.", m.ids.len(), ids.len())));
    }
    let n = idx.len();
    let mut w = Matrix::zeros(n, n);
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            w[(a, b)] = m.w[(i, j)];
        }
    }
    Ok(w)
}

pub fn matvec(w: &Matrix, x: &[f64]) -> Vec<f64> {
    (0..w.rows).map(|i| w.row(i).iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

// ---------------------------------------------------------------------------
// spgenerate
// ---------------------------------------------------------------------------

pub struct Spgenerate;

impl CommandHandler for Spgenerate {
    fn name(&self) -> &'static str {
        "spgenerate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (lhs, rhs) = cmd.main.split_once('=').ok_or_else(|| rc_err(198, "spgenerate newvar = W*varname"))?;
        let lw: Vec<&str> = lhs.split_whitespace().collect();
        let (ty, newv) = match lw.as_slice() {
            [t, v] => (VarType::from_type_str(t).ok_or_else(|| rc_err(198, format!("{t} invalid type")))?, v.to_string()),
            [v] => (st.default_type, v.to_string()),
            _ => return Err(rc_err(198, "spgenerate newvar = W*varname")),
        };
        crate::varlist::check_name(&newv)?;
        if ds.var_index.contains_key(&newv) {
            return Err(rc_err(110, format!("variable {newv} already defined")));
        }
        let (wname, var) = rhs.split_once('*').ok_or_else(|| rc_err(198, "spgenerate newvar = W*varname"))?;
        let m = get_spmat(st, wname.trim())?;
        let var = crate::varlist::resolve_name(ds, var.trim())?;
        let s = need_spset(ds)?;
        let mask = util::touse(ds, st, cmd)?;
        let ids = util::col(ds, &s.id)?;
        let x = util::col(ds, &var)?;
        let pos: HashMap<u64, usize> = m.ids.iter().enumerate().map(|(k, v)| (v.to_bits(), k)).collect();
        // group by time period for panel data
        let time = panel_vars(st).map(|(_, t)| util::col(ds, &t)).transpose()?;
        let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for i in 0..ds.n_obs {
            if mask[i] {
                groups.entry(time.as_ref().map(|t| t[i].to_bits()).unwrap_or(0)).or_default().push(i);
            }
        }
        let mut out = vec![SYSMISS; ds.n_obs];
        for rows in groups.values() {
            let mut xv = vec![0.0; m.ids.len()];
            for &i in rows {
                let k = *pos.get(&ids[i].to_bits()).ok_or_else(|| rc_err(459, format!("spatial unit {} not found in weighting matrix {}", fmt_id(ids[i]), wname.trim())))?;
                if is_missing(x[i]) {
                    return Err(rc_err(416, format!("missing values encountered in {var}")));
                }
                xv[k] = x[i];
            }
            for &i in rows {
                let k = pos[&ids[i].to_bits()];
                out[i] = m.w.row(k).iter().zip(&xv).map(|(a, b)| a * b).sum();
            }
        }
        let mut v = Variable::new(&newv, ty);
        v.data = out.iter().map(|&x| crate::data::store_num(ty, x)).collect();
        ds.add_variable(v)?;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// grmap
// ---------------------------------------------------------------------------

pub struct Grmap;

impl CommandHandler for Grmap {
    fn name(&self) -> &'static str {
        "grmap"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let s = need_spset(ds)?;
        let shp = s.shp.clone().ok_or_else(|| rc_err(459, "no shapefile linked; see spset, modify shpfile()"))?;
        let mut c = cmd.clone();
        c.verb = "spmap".into();
        c.using = Some(shp);
        if c.opt("id", 2).is_none() {
            c.opts.push(("id".into(), Some(s.id.clone())));
        }
        if panel_vars(st).is_some() && c.if_raw.is_none() && cmd.opt("t", 1).is_none() {
            return Err(rc_err(459, "panel data: specify if to select one time period, or option t(#)"));
        }
        if let Some(t) = cmd.opt("t", 1) {
            let (_, tv) = panel_vars(st).ok_or_else(|| rc_err(459, "option t() requires xtset data"))?;
            let cond = format!("{tv} == {}", t.trim());
            c.if_raw = Some(cond.clone());
            c.if_expr = Some(ferrostat_parser::parser::parse_expression(&cond).map_err(|e| rc_err(198, e.message))?);
        }
        crate::spatial::Spmap.execute(ds, st, &c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_of_path_graph() {
        // path graph on 4 nodes: eigenvalues 2 cos(k pi / 5)
        let mut m = Matrix::zeros(4, 4);
        for i in 0..3 {
            m[(i, i + 1)] = 1.0;
            m[(i + 1, i)] = 1.0;
        }
        let mut e = sym_eigenvalues(&m);
        e.sort_by(f64::total_cmp);
        let mut want: Vec<f64> = (1..=4).map(|k| 2.0 * (k as f64 * std::f64::consts::PI / 5.0).cos()).collect();
        want.sort_by(f64::total_cmp);
        for (a, b) in e.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12, "{a} {b}");
        }
        // row-normalized version is symmetrizable
        let mut r = m.clone();
        normalize(&mut r, "row").unwrap();
        let ev = eigenvalues(&r);
        let ldet = logdet(&ev, 0.5);
        let mut a = Matrix::identity(4);
        for i in 0..4 {
            for j in 0..4 {
                a[(i, j)] -= 0.5 * r[(i, j)];
            }
        }
        assert!((ldet - a.det().ln()).abs() < 1e-12);
    }
}
