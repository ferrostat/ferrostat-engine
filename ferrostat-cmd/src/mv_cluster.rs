//! Cluster analysis: `cluster` (hierarchical linkage methods, kmeans,
//! kmedians, dendrogram, generate, stop and the utilities list/dir/drop/
//! rename/renamevar/use/notes), `clustermat`, and `matrix dissimilarity`
//! with the similarity/dissimilarity measures shared with mds and discrim.
//!
//! Cluster analyses are kept as dataset characteristics (`_dta[_cl_*]`)
//! together with generated variables: `<stub>_id`, `<stub>_ord` (dendrogram
//! position), `<stub>_hgt` (fusion height between this leaf and the next in
//! dendrogram order) and, for centroid/median linkage, `<stub>_pht`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{finish, Graph, Layer, LayerKind, PcStyle};
use crate::matrix::Matrix;
use crate::mvutil;
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// measures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bin {
    Matching,
    Jaccard,
    Russell,
    Hamann,
    Dice,
    AntiDice,
    Sneath,
    Rogers,
    Ochiai,
    Yule,
    Anderberg,
    Kulczynski,
    Pearson,
    Gower2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    L2,
    L2sq,
    L1,
    Linf,
    Lp(f64),
    Lpow(f64),
    Canberra,
    Corr,
    Angular,
    Bin(Bin),
    Gower,
}

impl Measure {
    pub fn parse(s: &str) -> Result<Measure, String> {
        let t = s.trim().to_lowercase().replace(' ', "");
        let num = |pre: &str| -> Option<f64> { t.strip_prefix(pre).and_then(|r| r.strip_prefix('(')).and_then(|r| r.strip_suffix(')')).and_then(|r| r.parse().ok()) };
        if let Some(p) = num("lpower") {
            return Ok(if p == 2.0 { Measure::L2sq } else if p == 1.0 { Measure::L1 } else { Measure::Lpow(p) });
        }
        if let Some(p) = num("l") {
            return Ok(if p == 2.0 { Measure::L2 } else if p == 1.0 { Measure::L1 } else { Measure::Lp(p) });
        }
        Ok(match t.as_str() {
            "l2" | "euclidean" | "euclid" => Measure::L2,
            "l2squared" | "l2sq" => Measure::L2sq,
            "l1" | "absolute" | "abs" | "cityblock" | "manhattan" => Measure::L1,
            "linfinity" | "linf" | "maximum" | "max" => Measure::Linf,
            "canberra" => Measure::Canberra,
            "correlation" | "corr" => Measure::Corr,
            "angular" | "angle" => Measure::Angular,
            "matching" | "match" => Measure::Bin(Bin::Matching),
            "jaccard" => Measure::Bin(Bin::Jaccard),
            "russell" => Measure::Bin(Bin::Russell),
            "hamann" => Measure::Bin(Bin::Hamann),
            "dice" => Measure::Bin(Bin::Dice),
            "antidice" => Measure::Bin(Bin::AntiDice),
            "sneath" => Measure::Bin(Bin::Sneath),
            "rogers" => Measure::Bin(Bin::Rogers),
            "ochiai" => Measure::Bin(Bin::Ochiai),
            "yule" => Measure::Bin(Bin::Yule),
            "anderberg" => Measure::Bin(Bin::Anderberg),
            "kulczynski" => Measure::Bin(Bin::Kulczynski),
            "pearson" => Measure::Bin(Bin::Pearson),
            "gower2" => Measure::Bin(Bin::Gower2),
            "gower" => Measure::Gower,
            _ => return Err(rc_err(198, format!("unknown similarity or dissimilarity measure {s}"))),
        })
    }

    /// The `measure()` option, or a measure given as a bare option name.
    pub fn from_cmd(cmd: &Command, default: Measure) -> Result<Measure, String> {
        if let Some(m) = cmd.opt("measure", 3) {
            return Measure::parse(m);
        }
        for (n, a) in &cmd.opts {
            let full = match a {
                Some(x) => format!("{n}({x})"),
                None => n.clone(),
            };
            if let Ok(m) = Measure::parse(&full) {
                return Ok(m);
            }
        }
        Ok(default)
    }

    pub fn is_sim(&self) -> bool {
        matches!(self, Measure::Corr | Measure::Angular | Measure::Bin(_))
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Measure::Bin(_))
    }

    pub fn name(&self) -> String {
        match self {
            Measure::L2 => "L2".into(),
            Measure::L2sq => "L2squared".into(),
            Measure::L1 => "L1".into(),
            Measure::Linf => "Linfinity".into(),
            Measure::Lp(p) => format!("L({})", fmt_g(*p, 8)),
            Measure::Lpow(p) => format!("Lpower({})", fmt_g(*p, 8)),
            Measure::Canberra => "Canberra".into(),
            Measure::Corr => "correlation".into(),
            Measure::Angular => "angular".into(),
            Measure::Gower => "Gower".into(),
            Measure::Bin(b) => match b {
                Bin::Matching => "matching",
                Bin::Jaccard => "Jaccard",
                Bin::Russell => "Russell",
                Bin::Hamann => "Hamann",
                Bin::Dice => "Dice",
                Bin::AntiDice => "antiDice",
                Bin::Sneath => "Sneath",
                Bin::Rogers => "Rogers",
                Bin::Ochiai => "Ochiai",
                Bin::Yule => "Yule",
                Bin::Anderberg => "Anderberg",
                Bin::Kulczynski => "Kulczynski",
                Bin::Pearson => "Pearson",
                Bin::Gower2 => "Gower2",
            }
            .into(),
        }
    }

    /// Range shown by `cluster list`: most similar, least similar.
    pub fn range(&self) -> &'static str {
        match self {
            Measure::Corr | Measure::Angular => "1 -1",
            Measure::Bin(Bin::Hamann) | Measure::Bin(Bin::Yule) | Measure::Bin(Bin::Pearson) => "1 -1",
            Measure::Bin(_) => "1 0",
            _ => "0 .",
        }
    }

    /// Similarity or dissimilarity between two vectors. `ranges` are the
    /// variable ranges (Gower) and `binvar` flags binary variables (Gower).
    pub fn eval(&self, x: &[f64], y: &[f64], ranges: &[f64], binvar: &[bool]) -> f64 {
        let p = x.len();
        match self {
            Measure::L2 => x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt(),
            Measure::L2sq => x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum(),
            Measure::L1 => x.iter().zip(y).map(|(a, b)| (a - b).abs()).sum(),
            Measure::Linf => x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
            Measure::Lp(q) => x.iter().zip(y).map(|(a, b)| (a - b).abs().powf(*q)).sum::<f64>().powf(1.0 / q),
            Measure::Lpow(q) => x.iter().zip(y).map(|(a, b)| (a - b).abs().powf(*q)).sum(),
            Measure::Canberra => x
                .iter()
                .zip(y)
                .map(|(a, b)| {
                    let d = a.abs() + b.abs();
                    if d == 0.0 { 0.0 } else { (a - b).abs() / d }
                })
                .sum(),
            Measure::Corr => {
                let mx = x.iter().sum::<f64>() / p as f64;
                let my = y.iter().sum::<f64>() / p as f64;
                let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
                let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
                let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
                sxy / (sxx * syy).sqrt()
            }
            Measure::Angular => {
                let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
                let sxx: f64 = x.iter().map(|a| a * a).sum();
                let syy: f64 = y.iter().map(|b| b * b).sum();
                sxy / (sxx * syy).sqrt()
            }
            Measure::Gower => {
                let (mut s, mut n) = (0.0, 0.0);
                for v in 0..p {
                    if is_missing(x[v]) || is_missing(y[v]) {
                        continue;
                    }
                    n += 1.0;
                    if binvar.get(v).copied().unwrap_or(false) {
                        s += if x[v] == y[v] { 0.0 } else { 1.0 };
                    } else {
                        let r = ranges.get(v).copied().unwrap_or(0.0);
                        s += if r > 0.0 { (x[v] - y[v]).abs() / r } else { 0.0 };
                    }
                }
                if n == 0.0 { SYSMISS } else { s / n }
            }
            Measure::Bin(b) => {
                let (mut a, mut bb, mut c, mut d) = (0.0, 0.0, 0.0, 0.0);
                for v in 0..p {
                    let xi = if x[v] != 0.0 { x[v].min(1.0) } else { 0.0 };
                    let yi = if y[v] != 0.0 { y[v].min(1.0) } else { 0.0 };
                    // x as weights (proportions allowed on either side)
                    a += xi * yi;
                    bb += xi * (1.0 - yi);
                    c += (1.0 - xi) * yi;
                    d += (1.0 - xi) * (1.0 - yi);
                }
                bin_sim(*b, a, bb, c, d)
            }
        }
    }
}

fn bin_sim(m: Bin, a: f64, b: f64, c: f64, d: f64) -> f64 {
    let n = a + b + c + d;
    let eps = 1e-12;
    match m {
        Bin::Matching => (a + d) / n,
        Bin::Jaccard => if a + b + c < eps { 1.0 } else { a / (a + b + c) },
        Bin::Russell => a / n,
        Bin::Hamann => ((a + d) - (b + c)) / n,
        Bin::Dice => if a + b + c < eps { 1.0 } else { 2.0 * a / (2.0 * a + b + c) },
        Bin::AntiDice => if a + b + c < eps { 1.0 } else { a / (a + 2.0 * (b + c)) },
        Bin::Sneath => 2.0 * (a + d) / (2.0 * (a + d) + b + c),
        Bin::Rogers => (a + d) / ((a + d) + 2.0 * (b + c)),
        Bin::Ochiai => {
            if a + b < eps && a + c < eps {
                1.0
            } else if a + b < eps || a + c < eps {
                0.0
            } else {
                a / ((a + b) * (a + c)).sqrt()
            }
        }
        Bin::Yule | Bin::Pearson => {
            if b + c < eps {
                1.0
            } else if a + d < eps {
                -1.0
            } else if (a * d - b * c).abs() < eps {
                0.0
            } else if m == Bin::Yule {
                (a * d - b * c) / (a * d + b * c)
            } else {
                (a * d - b * c) / ((a + b) * (a + c) * (d + b) * (d + c)).sqrt()
            }
        }
        Bin::Anderberg => {
            if (a + b + c < eps) || (b + c + d < eps) {
                1.0
            } else if a + b < eps || a + c < eps || c + d < eps || b + d < eps {
                0.0
            } else {
                (a / (a + b) + a / (a + c) + d / (c + d) + d / (b + d)) / 4.0
            }
        }
        Bin::Kulczynski => {
            if a + b < eps && a + c < eps {
                1.0
            } else if a + b < eps || a + c < eps {
                0.0
            } else {
                (a / (a + b) + a / (a + c)) / 2.0
            }
        }
        Bin::Gower2 => {
            if (a + b + c < eps) || (b + c + d < eps) {
                1.0
            } else if a * d < eps {
                0.0
            } else {
                a * d / ((a + b) * (a + c) * (d + b) * (d + c)).sqrt()
            }
        }
    }
}

/// Ranges and binary flags of the columns (for Gower).
fn gower_info(cols: &[Vec<f64>]) -> (Vec<f64>, Vec<bool>) {
    let ranges = cols
        .iter()
        .map(|c| {
            let (mn, mx) = c.iter().filter(|v| !is_missing(**v)).fold((f64::INFINITY, f64::NEG_INFINITY), |a, &v| (a.0.min(v), a.1.max(v)));
            if mx > mn { mx - mn } else { 0.0 }
        })
        .collect();
    let bins = cols.iter().map(|c| c.iter().all(|&v| is_missing(v) || v == 0.0 || v == 1.0)).collect();
    (ranges, bins)
}

/// Pairwise (dis)similarities between the rows of `rows` (each a vector over variables).
pub fn pairwise(rows: &[Vec<f64>], m: Measure) -> Vec<Vec<f64>> {
    let n = rows.len();
    let p = rows.first().map(|r| r.len()).unwrap_or(0);
    let cols: Vec<Vec<f64>> = (0..p).map(|v| rows.iter().map(|r| r[v]).collect()).collect();
    let (ranges, bins) = gower_info(&cols);
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            let v = m.eval(&rows[i], &rows[j], &ranges, &bins);
            d[i][j] = v;
            d[j][i] = v;
        }
        d[i][i] = if m.is_sim() { m.eval(&rows[i], &rows[i], &ranges, &bins) } else { 0.0 };
    }
    d
}

// ---------------------------------------------------------------------------
// matrix dissimilarity
// ---------------------------------------------------------------------------

/// `matrix dissimilarity matname = varlist [if] [in] [, measure observations
/// variables names() allbinary proportions dissim(oneminus|standard)]`
pub fn matrix_dissimilarity(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    // cmd.main: "dissimilarity name = varlist"
    let body = cmd.main.trim();
    let rest = body.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("");
    let (name, vl) = rest.split_once('=').ok_or_else(|| rc_err(198, "= expected"))?;
    let name = name.trim().to_string();
    let vars = crate::varlist::expand(ds, vl.trim())?;
    let m = Measure::from_cmd(cmd, Measure::L2)?;
    let by_var = cmd.has_opt("variables", 3);
    let mut mask = util::touse(ds, st, cmd)?;
    if m != Measure::Gower {
        util::markout(ds, &mut mask, &vars);
    }
    let rows_idx = util::rows(&mask);
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows_idx.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
    if cmd.has_opt("allbinary", 4) && cols.iter().flatten().any(|&v| !is_missing(v) && v != 0.0 && v != 1.0) {
        return Err(rc_err(198, "nonbinary data encountered"));
    }
    let items: Vec<Vec<f64>> = if by_var { cols.clone() } else { (0..rows_idx.len()).map(|t| cols.iter().map(|c| c[t]).collect()).collect() };
    let mut d = pairwise(&items, m);
    if m.is_sim() {
        if let Some(how) = cmd.opt("dissim", 4) {
            let k = d.len();
            let diag: Vec<f64> = (0..k).map(|i| d[i][i]).collect();
            for i in 0..k {
                for j in 0..k {
                    d[i][j] = if how.trim().starts_with('o') { 1.0 - d[i][j] } else { (diag[i] + diag[j] - 2.0 * d[i][j]).max(0.0).sqrt() };
                }
            }
        }
    }
    let names: Vec<String> = if by_var {
        vars.clone()
    } else if let Some(nv) = cmd.opt("names", 2) {
        let var = ds.get_var(nv.trim()).ok_or_else(|| rc_err(111, format!("variable {} not found", nv.trim())))?;
        rows_idx.iter().map(|&i| crate::util::cell_text(ds, var, i, false)).collect()
    } else {
        rows_idx.iter().map(|&i| format!("obs{}", i + 1)).collect()
    };
    let k = d.len();
    let mut mat = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            mat[(i, j)] = d[i][j];
        }
    }
    st.matrices.insert(name, mat.with_names(names.clone(), names));
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// cluster objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ClObj {
    pub name: String,
    pub typ: String,
    pub method: String,
    pub measure: String,
    pub sim: bool,
    /// (variable, role) with roles id, order, height, pseudo_height, group
    pub vars: Vec<(String, String)>,
    pub other: Vec<(String, String)>,
    pub notes: Vec<String>,
}

impl ClObj {
    fn var(&self, role: &str) -> Option<String> {
        self.vars.iter().find(|(_, r)| r == role).map(|(v, _)| v.clone())
    }
    fn other(&self, key: &str) -> Option<String> {
        self.other.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }
    fn encode(&self) -> String {
        let mut lines = vec![format!("type\t{}", self.typ), format!("method\t{}", self.method), format!("measure\t{}", self.measure), format!("sim\t{}", if self.sim { 1 } else { 0 })];
        for (v, r) in &self.vars {
            lines.push(format!("var\t{v}\t{r}"));
        }
        for (k, v) in &self.other {
            lines.push(format!("other\t{k}\t{v}"));
        }
        for n in &self.notes {
            lines.push(format!("note\t{n}"));
        }
        lines.join("\n")
    }
    fn decode(name: &str, s: &str) -> ClObj {
        let mut o = ClObj { name: name.to_string(), ..Default::default() };
        for line in s.lines() {
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            match parts.as_slice() {
                ["type", v] => o.typ = v.to_string(),
                ["method", v] => o.method = v.to_string(),
                ["measure", v] => o.measure = v.to_string(),
                ["sim", v] => o.sim = *v == "1",
                ["var", v, r] => o.vars.push((v.to_string(), r.to_string())),
                ["other", k, v] => o.other.push((k.to_string(), v.to_string())),
                ["note", v] => o.notes.push(v.to_string()),
                ["note", v, w] => o.notes.push(format!("{v}\t{w}")),
                _ => {}
            }
        }
        o
    }
}

pub(crate) fn load_objs(ds: &DataSet) -> Vec<ClObj> {
    let Some(ch) = ds.characteristics.get("_dta") else { return vec![] };
    let names = ch.get("_cl_list").cloned().unwrap_or_default();
    names.split_whitespace().filter_map(|n| ch.get(&format!("_cl_{n}")).map(|s| ClObj::decode(n, s))).collect()
}

pub(crate) fn save_objs(ds: &mut DataSet, objs: &[ClObj]) {
    let ch = ds.characteristics.entry("_dta".into()).or_default();
    ch.retain(|k, _| !k.starts_with("_cl_"));
    if objs.is_empty() {
        return;
    }
    ch.insert("_cl_list".into(), objs.iter().map(|o| o.name.clone()).collect::<Vec<_>>().join(" "));
    for o in objs {
        ch.insert(format!("_cl_{}", o.name), o.encode());
    }
}

/// Add (or replace) an object and make it the most recent one.
fn push_obj(ds: &mut DataSet, o: ClObj) {
    let mut objs = load_objs(ds);
    objs.retain(|x| x.name != o.name);
    objs.insert(0, o);
    save_objs(ds, &objs);
}

fn find_obj(ds: &DataSet, name: Option<&str>) -> Result<ClObj, String> {
    let objs = load_objs(ds);
    match name {
        Some(n) => {
            let n = n.trim();
            objs.into_iter().find(|o| o.name == n || glob(n, &o.name)).ok_or_else(|| rc_err(198, format!("{n} not found")))
        }
        None => objs.into_iter().next().ok_or_else(|| rc_err(198, "no cluster solutions defined")),
    }
}

fn glob(pat: &str, s: &str) -> bool {
    if !pat.contains('*') && !pat.contains('?') {
        return pat == s;
    }
    fn rec(p: &[char], s: &[char]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some('*'), _) => rec(&p[1..], s) || (!s.is_empty() && rec(p, &s[1..])),
            (Some('?'), Some(_)) => rec(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => rec(&p[1..], &s[1..]),
            _ => false,
        }
    }
    rec(&pat.chars().collect::<Vec<_>>(), &s.chars().collect::<Vec<_>>())
}

fn default_name(ds: &DataSet) -> String {
    let objs = load_objs(ds);
    let mut k = 1;
    loop {
        let n = format!("_clus_{k}");
        if !objs.iter().any(|o| o.name == n) && ds.get_var(&format!("{n}_id")).is_none() && ds.get_var(&n).is_none() {
            return n;
        }
        k += 1;
    }
}

fn set_var(ds: &mut DataSet, name: &str, vt: VarType, vals: &[f64]) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    let mut v = Variable::new(name, vt);
    v.data = vals.iter().map(|&x| crate::data::store_num(vt, x)).collect();
    ds.add_variable(v)
}

fn int_type(maxv: f64) -> VarType {
    if maxv <= 100.0 {
        VarType::Byte
    } else if maxv <= 32740.0 {
        VarType::Int
    } else {
        VarType::Long
    }
}

// ---------------------------------------------------------------------------
// hierarchical clustering
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Weighted,
    Centroid,
    Median,
    Ward,
}

impl Linkage {
    fn from_sub(s: &str) -> Option<Linkage> {
        let s = s.to_lowercase();
        let pre = |full: &str, min: usize| s.len() >= min && full.starts_with(s.as_str());
        Some(if pre("singlelinkage", 1) {
            Linkage::Single
        } else if pre("completelinkage", 1) {
            Linkage::Complete
        } else if pre("averagelinkage", 1) {
            Linkage::Average
        } else if pre("waveragelinkage", 2) {
            Linkage::Weighted
        } else if pre("centroidlinkage", 4) {
            Linkage::Centroid
        } else if pre("medianlinkage", 3) {
            Linkage::Median
        } else if pre("wardslinkage", 4) || s == "ward" {
            Linkage::Ward
        } else {
            return None;
        })
    }
    fn method(self) -> &'static str {
        match self {
            Linkage::Single => "single",
            Linkage::Complete => "complete",
            Linkage::Average => "average",
            Linkage::Weighted => "waverage",
            Linkage::Centroid => "centroid",
            Linkage::Median => "median",
            Linkage::Ward => "wards",
        }
    }
    fn full(self) -> &'static str {
        match self {
            Linkage::Single => "singlelinkage",
            Linkage::Complete => "completelinkage",
            Linkage::Average => "averagelinkage",
            Linkage::Weighted => "waveragelinkage",
            Linkage::Centroid => "centroidlinkage",
            Linkage::Median => "medianlinkage",
            Linkage::Ward => "wardslinkage",
        }
    }
}

/// Result of a hierarchical clustering: dendrogram order (leaf indices),
/// heights between adjacent leaves (dissimilarity orientation) and the merge
/// step at each boundary.
pub struct Tree {
    pub order: Vec<usize>,
    pub hgt: Vec<f64>,
    pub step: Vec<f64>,
    pub reversal: bool,
}

/// Lance-Williams agglomeration on a dissimilarity matrix.
pub fn hclust(d0: &[Vec<f64>], link: Linkage) -> Tree {
    let n = d0.len();
    let mut d: Vec<Vec<f64>> = d0.to_vec();
    let mut size = vec![1.0f64; n];
    let mut active = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // boundary heights keyed by left leaf
    let mut bh = vec![SYSMISS; n];
    let mut bs = vec![SYSMISS; n];
    let mut reversal = false;
    let mut last_h = f64::NEG_INFINITY;
    // nearest neighbour cache (only j > i)
    let nn_of = |d: &Vec<Vec<f64>>, active: &Vec<bool>, i: usize| -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        for j in 0..n {
            if j != i && active[j] && d[i][j] < best.1 {
                best = (j, d[i][j]);
            }
        }
        best
    };
    let mut nn: Vec<(usize, f64)> = (0..n).map(|i| nn_of(&d, &active, i)).collect();
    for stepno in 1..n {
        // closest pair (smallest i on ties)
        let mut bi = usize::MAX;
        let mut bd = f64::INFINITY;
        for i in 0..n {
            if active[i] && nn[i].0 != usize::MAX && nn[i].1 < bd {
                bd = nn[i].1;
                bi = i;
            }
        }
        if bi == usize::MAX {
            break;
        }
        let bj = nn[bi].0;
        let (i, j) = if bi < bj { (bi, bj) } else { (bj, bi) };
        let h = d[i][j];
        if h < last_h - 1e-12 {
            reversal = true;
        }
        last_h = last_h.max(h);
        let (ni, nj) = (size[i], size[j]);
        // left part: the cluster holding the smaller observation index
        let (l, r) = if members[i][0].min(*members[i].iter().min().unwrap()) <= *members[j].iter().min().unwrap() { (i, j) } else { (j, i) };
        let last_left = *members[l].last().unwrap();
        bh[last_left] = h;
        bs[last_left] = stepno as f64;
        let mut merged = members[l].clone();
        merged.extend(members[r].iter().copied());
        for k in 0..n {
            if !active[k] || k == i || k == j {
                continue;
            }
            let nk = size[k];
            let (dik, djk) = (d[i][k], d[j][k]);
            let v = match link {
                Linkage::Single => 0.5 * dik + 0.5 * djk - 0.5 * (dik - djk).abs(),
                Linkage::Complete => 0.5 * dik + 0.5 * djk + 0.5 * (dik - djk).abs(),
                Linkage::Average => (ni * dik + nj * djk) / (ni + nj),
                Linkage::Weighted => 0.5 * (dik + djk),
                Linkage::Centroid => (ni * dik + nj * djk) / (ni + nj) - ni * nj * h / ((ni + nj) * (ni + nj)),
                Linkage::Median => 0.5 * dik + 0.5 * djk - 0.25 * h,
                Linkage::Ward => ((ni + nk) * dik + (nj + nk) * djk - nk * h) / (ni + nj + nk),
            };
            d[i][k] = v;
            d[k][i] = v;
        }
        active[j] = false;
        size[i] = ni + nj;
        members[i] = merged;
        members[j].clear();
        // refresh nearest neighbours
        for k in 0..n {
            if !active[k] {
                continue;
            }
            if k == i || nn[k].0 == i || nn[k].0 == j {
                nn[k] = nn_of(&d, &active, k);
            } else if d[k][i] < nn[k].1 {
                nn[k] = (i, d[k][i]);
            }
        }
    }
    let root = (0..n).find(|&k| active[k]).unwrap_or(0);
    let order = members[root].clone();
    let hgt: Vec<f64> = order.iter().map(|&o| bh[o]).collect();
    let step: Vec<f64> = order.iter().map(|&o| bs[o]).collect();
    Tree { order, hgt, step, reversal }
}

/// Store a tree as cluster variables and object.
#[allow(clippy::too_many_arguments)]
fn store_tree(ds: &mut DataSet, rows: &[usize], tree: &Tree, link: Linkage, m: &str, sim: bool, name: &str, stub: &str, other: Vec<(String, String)>) -> Result<(), String> {
    let n = ds.n_obs;
    let mut id = vec![SYSMISS; n];
    let mut ord = vec![SYSMISS; n];
    let mut hgt = vec![SYSMISS; n];
    let mut pht = vec![SYSMISS; n];
    for (pos, &leaf) in tree.order.iter().enumerate() {
        let obs = rows[leaf];
        id[obs] = (obs + 1) as f64;
        ord[obs] = (pos + 1) as f64;
        let h = tree.hgt[pos];
        hgt[obs] = if is_missing(h) { SYSMISS } else if sim { -h } else { h };
        pht[obs] = tree.step[pos];
    }
    let it = int_type(n as f64);
    set_var(ds, &format!("{stub}_id"), it, &id)?;
    set_var(ds, &format!("{stub}_ord"), it, &ord)?;
    set_var(ds, &format!("{stub}_hgt"), VarType::Double, &hgt)?;
    let mut vars = vec![(format!("{stub}_id"), "id".to_string()), (format!("{stub}_ord"), "order".to_string()), (format!("{stub}_hgt"), if tree.reversal || matches!(link, Linkage::Centroid | Linkage::Median) { "real_height".to_string() } else { "height".to_string() })];
    if matches!(link, Linkage::Centroid | Linkage::Median) && tree.reversal {
        set_var(ds, &format!("{stub}_pht"), VarType::Double, &pht)?;
        vars.push((format!("{stub}_pht"), "pseudo_height".to_string()));
    }
    push_obj(ds, ClObj { name: name.to_string(), typ: "hierarchical".into(), method: link.method().into(), measure: m.to_string(), sim, vars, other, notes: vec![] });
    Ok(())
}

/// Dendrogram data: (rows in order, adjacency heights used for cutting in
/// dissimilarity orientation, user-scale heights).
fn tree_of(ds: &DataSet, o: &ClObj) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>), String> {
    let ordv = o.var("order").ok_or_else(|| rc_err(198, format!("{} is not a hierarchical cluster analysis", o.name)))?;
    let hv = o.vars.iter().find(|(_, r)| r.ends_with("height") && r != "pseudo_height").map(|(v, _)| v.clone()).unwrap_or_default();
    let ordc = util::col(ds, &ordv)?;
    let hc = util::col(ds, &hv)?;
    let phc = match o.var("pseudo_height") {
        Some(v) => Some(util::col(ds, &v)?),
        None => None,
    };
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(ordc[i])).collect();
    rows.sort_by(|&a, &b| ordc[a].total_cmp(&ordc[b]));
    let user: Vec<f64> = rows.iter().map(|&i| hc[i]).collect();
    let cut: Vec<f64> = match &phc {
        Some(p) => rows.iter().map(|&i| p[i]).collect(),
        None => user.iter().map(|&h| if is_missing(h) { h } else if o.sim { -h } else { h }).collect(),
    };
    Ok((rows, cut, user))
}

/// Group numbers (1-based, left to right) for `k` groups; None on ties.
fn groups_k(cut: &[f64], k: usize) -> Option<Vec<usize>> {
    let n = cut.len();
    if k == 0 || k > n {
        return None;
    }
    let mut hs: Vec<f64> = cut[..n.saturating_sub(1)].to_vec();
    hs.sort_by(|a, b| b.total_cmp(a));
    let thr_idx = k - 1;
    let splits: Vec<bool> = if k == 1 {
        vec![false; n]
    } else {
        let t = hs[thr_idx - 1];
        if thr_idx < hs.len() && (hs[thr_idx] - t).abs() <= 1e-12 * t.abs().max(1.0) {
            return None;
        }
        cut.iter().enumerate().map(|(i, &h)| i + 1 < n && h >= t).collect()
    };
    let mut g = vec![0usize; n];
    let mut cur = 1;
    for i in 0..n {
        g[i] = cur;
        if splits[i] {
            cur += 1;
        }
    }
    Some(g)
}

// ---------------------------------------------------------------------------
// kmeans / kmedians
// ---------------------------------------------------------------------------

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return SYSMISS;
    }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn centers_of(x: &[Vec<f64>], g: &[usize], k: usize, med: bool) -> Vec<Vec<f64>> {
    let p = x.first().map(|r| r.len()).unwrap_or(0);
    (0..k)
        .map(|c| {
            let members: Vec<&Vec<f64>> = x.iter().zip(g).filter(|(_, gg)| **gg == c).map(|(r, _)| r).collect();
            (0..p)
                .map(|v| {
                    let mut vals: Vec<f64> = members.iter().map(|r| r[v]).collect();
                    if vals.is_empty() {
                        SYSMISS
                    } else if med {
                        median(&mut vals)
                    } else {
                        vals.iter().sum::<f64>() / vals.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}

fn kpartition(x: &[Vec<f64>], mut centers: Vec<Vec<f64>>, m: Measure, med: bool, maxit: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = x.len();
    let k = centers.len();
    let cols: Vec<Vec<f64>> = (0..x.first().map(|r| r.len()).unwrap_or(0)).map(|v| x.iter().map(|r| r[v]).collect()).collect();
    let (ranges, bins) = gower_info(&cols);
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        let v = m.eval(a, b, &ranges, &bins);
        if m.is_sim() { -v } else { v }
    };
    let mut g = vec![usize::MAX; n];
    for _ in 0..maxit.max(1) {
        let mut changed = false;
        for i in 0..n {
            let ds: Vec<f64> = centers.iter().map(|c| if c.iter().any(|v| is_missing(*v)) { f64::INFINITY } else { dist(&x[i], c) }).collect();
            let best = ds.iter().cloned().fold(f64::INFINITY, f64::min);
            let keep = g[i] != usize::MAX && (ds[g[i]] - best).abs() <= 1e-12 * best.abs().max(1e-300);
            if !keep {
                let ng = ds.iter().position(|&v| v == best).unwrap_or(0);
                if g[i] != ng {
                    g[i] = ng;
                    changed = true;
                }
            }
        }
        // orphaned centers: use the observations farthest from their centers
        let counts: Vec<usize> = (0..k).map(|c| g.iter().filter(|&&gg| gg == c).count()).collect();
        if counts.iter().any(|&c| c == 0) {
            let mut far: Vec<(f64, usize)> = (0..n).map(|i| (dist(&x[i], &centers[g[i]]), i)).collect();
            far.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut fi = 0;
            for c in 0..k {
                if counts[c] == 0 && fi < far.len() {
                    centers[c] = x[far[fi].1].clone();
                    g[far[fi].1] = c;
                    fi += 1;
                }
            }
            changed = true;
        }
        centers = centers_of(x, &g, k, med);
        if !changed {
            break;
        }
    }
    (g, centers)
}

// ---------------------------------------------------------------------------
// the cluster command
// ---------------------------------------------------------------------------

pub struct Cluster {
    pub mat: bool,
}

impl CommandHandler for Cluster {
    fn name(&self) -> &'static str {
        if self.mat { "clustermat" } else { "cluster" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim().to_string();
        let (sub, rest) = match main.split_once(char::is_whitespace) {
            Some((a, b)) => (a.to_string(), b.trim().to_string()),
            None => (main.clone(), String::new()),
        };
        let subl = sub.to_lowercase();
        let pre = |full: &str, min: usize| subl.len() >= min && full.starts_with(subl.as_str());
        if pre("kmedians", 4) {
            return self.kmeans(ds, st, cmd, &rest, true);
        }
        if pre("kmeans", 1) {
            return self.kmeans(ds, st, cmd, &rest, false);
        }
        if pre("dendrogram", 4) || subl == "tree" {
            return dendrogram(ds, st, cmd, &rest);
        }
        if pre("generate", 3) {
            return generate(ds, cmd, &rest);
        }
        if pre("stop", 4) {
            return stop(ds, st, cmd, &rest);
        }
        if subl == "dir" {
            return Ok(load_objs(ds).iter().map(|o| o.name.clone()).collect::<Vec<_>>().join("\n"));
        }
        if pre("list", 1) {
            return list(ds, cmd, &rest);
        }
        if subl == "drop" {
            let mut objs = load_objs(ds);
            if rest.trim() == "_all" {
                objs.clear();
            } else {
                for n in rest.split_whitespace() {
                    let before = objs.len();
                    objs.retain(|o| !glob(n, &o.name));
                    if objs.len() == before {
                        return Err(rc_err(198, format!("{n} not found")));
                    }
                }
            }
            save_objs(ds, &objs);
            return Ok(String::new());
        }
        if subl == "use" {
            let o = find_obj(ds, Some(&rest))?;
            push_obj(ds, o);
            return Ok(String::new());
        }
        if subl == "rename" {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "cluster rename oldname newname"));
            }
            let mut objs = load_objs(ds);
            if objs.iter().any(|o| o.name == w[1]) {
                return Err(rc_err(110, format!("{} already defined", w[1])));
            }
            let o = objs.iter_mut().find(|o| glob(w[0], &o.name)).ok_or_else(|| rc_err(198, format!("{} not found", w[0])))?;
            o.name = w[1].to_string();
            save_objs(ds, &objs);
            return Ok(String::new());
        }
        if subl == "renamevar" {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "cluster renamevar oldvar newvar"));
            }
            let mut objs = load_objs(ds);
            let idx = match cmd.opt("name", 1) {
                Some(n) => objs.iter().position(|o| glob(n.trim(), &o.name)).ok_or_else(|| rc_err(198, format!("{} not found", n.trim())))?,
                None => 0,
            };
            let prefix = cmd.has_opt("prefix", 3);
            let o = objs.get_mut(idx).ok_or_else(|| rc_err(198, "no cluster solutions defined"))?;
            let mut found = false;
            let mut renames = Vec::new();
            for (v, _) in o.vars.iter_mut() {
                let newn = if prefix && v.starts_with(w[0]) { Some(format!("{}{}", w[1], &v[w[0].len()..])) } else if !prefix && v == w[0] { Some(w[1].to_string()) } else { None };
                if let Some(nn) = newn {
                    renames.push((v.clone(), nn.clone()));
                    *v = nn;
                    found = true;
                }
            }
            if !found {
                return Err(rc_err(198, format!("variable {} not found in {}", w[0], o.name)));
            }
            for (a, b) in renames {
                if let Some(i) = ds.var_index.get(&a).copied() {
                    ds.vars[i].name = b.clone();
                    ds.var_index.remove(&a);
                    ds.var_index.insert(b, i);
                }
            }
            save_objs(ds, &objs);
            return Ok(String::new());
        }
        if subl == "notes" || subl == "note" {
            return notes(ds, cmd, &rest);
        }
        if subl == "query" {
            let objs = load_objs(ds);
            let r = st.begin_r();
            r.set_str("names", objs.iter().map(|o| o.name.clone()).collect::<Vec<_>>().join(" "));
            if let Some(n) = rest.split_whitespace().next() {
                let o = objs.iter().find(|o| o.name == n).ok_or_else(|| rc_err(198, format!("{n} not found")))?;
                r.set_str("name", o.name.clone());
                r.set_str("type", o.typ.clone());
                r.set_str("method", o.method.clone());
                r.set_str(if o.sim { "similarity" } else { "dissimilarity" }, o.measure.clone());
                for (i, (v, role)) in o.vars.iter().enumerate() {
                    r.set_str(&format!("v{}_name", i + 1), v.clone());
                    r.set_str(&format!("v{}_tag", i + 1), role.clone());
                }
                for (i, (k, v)) in o.other.iter().enumerate() {
                    r.set_str(&format!("o{}_tag", i + 1), k.clone());
                    r.set_str(&format!("o{}_val", i + 1), v.clone());
                }
            }
            return Ok(String::new());
        }
        if let Some(link) = Linkage::from_sub(&subl) {
            return self.linkage(ds, st, cmd, link, &rest);
        }
        if let Some(r) = crate::mv_w5::cluster_more(ds, st, cmd, &subl, &rest) { return r; }
        Err(rc_err(199, format!("unknown cluster subcommand {sub}")))
    }
}

impl Cluster {
    fn linkage(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command, link: Linkage, rest: &str) -> Result<String, String> {
        let default_m = if matches!(link, Linkage::Centroid | Linkage::Median | Linkage::Ward) { Measure::L2sq } else { Measure::L2 };
        let given_name = cmd.opt("name", 1).map(|s| s.trim().to_string());
        let mut out = Vec::new();
        let (d, rows, m_name, sim, varlist): (Vec<Vec<f64>>, Vec<usize>, String, bool, String) = if self.mat {
            let mname = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "matrix name required"))?;
            let mm = st.matrices.get(mname).cloned().ok_or_else(|| rc_err(111, format!("matrix {mname} not found")))?;
            let full = shape_matrix(&mm, cmd.opt("shape", 2).unwrap_or("full"))?;
            let k = full.rows;
            let mut dd = vec![vec![0.0; k]; k];
            for i in 0..k {
                for j in 0..k {
                    dd[i][j] = if cmd.has_opt("force", 5) { 0.5 * (full[(i, j)] + full[(j, i)]) } else { full[(i, j)] };
                }
                if cmd.has_opt("force", 5) {
                    dd[i][i] = 0.0;
                }
            }
            if !cmd.has_opt("force", 5) {
                for i in 0..k {
                    for j in 0..i {
                        if (dd[i][j] - dd[j][i]).abs() > 1e-8 * (1.0 + dd[i][j].abs()) {
                            return Err(rc_err(505, format!("matrix {mname} not symmetric")));
                        }
                    }
                }
            }
            let rows: Vec<usize> = if cmd.has_opt("clear", 5) || ds.n_vars() == 0 {
                *ds = DataSet::new();
                for _ in 0..k {
                    ds.add_observation();
                }
                out.push(format!("Number of observations (_N) was 0, now {k}."));
                (0..k).collect()
            } else if cmd.has_opt("add", 3) {
                let mask = util::touse(ds, st, cmd)?;
                let r = util::rows(&mask);
                if r.len() != k {
                    return Err(rc_err(198, format!("matrix {mname} has {k} rows but there are {} observations", r.len())));
                }
                r
            } else {
                return Err(rc_err(4, "no; data in memory would be lost (specify add or clear)"));
            };
            if let Some(lv) = cmd.opt("labelvar", 3) {
                let names = if full.rownames.len() == k { full.rownames.clone() } else { mvutil::names("r", k) };
                let w = names.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                let mut v = Variable::new(lv.trim(), VarType::Str(w.min(2045)));
                let mut data = vec![Value::Str(String::new()); ds.n_obs];
                for (t, &i) in rows.iter().enumerate() {
                    data[i] = Value::Str(names[t].clone());
                }
                v.data = data;
                if ds.get_var(lv.trim()).is_some() {
                    ds.drop_variable(lv.trim())?;
                }
                ds.add_variable(v)?;
            }
            (dd, rows, "user matrix ".to_string() + mname, false, String::new())
        } else {
            let vars = crate::varlist::expand(ds, if rest.is_empty() { "_all" } else { rest })?;
            let m = Measure::from_cmd(cmd, default_m)?;
            let mut mask = util::touse(ds, st, cmd)?;
            util::markout(ds, &mut mask, &vars);
            let rows = util::rows(&mask);
            let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
            let items: Vec<Vec<f64>> = rows.iter().map(|&i| cols.iter().map(|c| c[i]).collect()).collect();
            let s = pairwise(&items, m);
            let dd: Vec<Vec<f64>> = if m.is_sim() { s.iter().map(|r| r.iter().map(|v| -v).collect()).collect() } else { s };
            (dd, rows, m.name(), m.is_sim(), vars.join(" "))
        };
        if rows.len() < 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let name = match &given_name {
            Some(n) => n.clone(),
            None => {
                let n = default_name(ds);
                out.push(format!("cluster name: {n}"));
                n
            }
        };
        let stub = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| name.clone());
        let tree = hclust(&d, link);
        let mut other = vec![("cmd".to_string(), format!("{} {}", if self.mat { "clustermat" } else { "cluster" }, cmd.raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim().replacen(cmd.main.split_whitespace().next().unwrap_or(""), link.full(), 1)))];
        if !self.mat {
            other.push(("varlist".into(), varlist));
        }
        let meas = if self.mat { Measure::L2 } else { Measure::from_cmd(cmd, default_m)? };
        other.push(("range".into(), if self.mat { "0 .".into() } else { meas.range().to_string() }));
        let mname = if self.mat { "user matrix".to_string() } else { m_name };
        store_tree(ds, &rows, &tree, link, &mname, sim, &name, &stub, other)?;
        st.data_changed = true;
        let _ = st;
        Ok(out.join("\n"))
    }

    fn kmeans(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str, med: bool) -> Result<String, String> {
        let k = mvutil::opt_usize(cmd, "k", 1)?.ok_or_else(|| rc_err(198, "option k() required"))?;
        if k < 1 {
            return Err(rc_err(198, "k() must be positive"));
        }
        let vars = crate::varlist::expand(ds, if rest.is_empty() { "_all" } else { rest })?;
        let m = Measure::from_cmd(cmd, Measure::L2)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let mut rows = util::rows(&mask);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let row_of = |i: usize| -> Vec<f64> { cols.iter().map(|c| c[i]).collect() };
        let start = cmd.opt("start", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| "krandom".into());
        let sl = start.to_lowercase();
        let (head, arg) = match sl.split_once('(') {
            Some((h, a)) => (h.trim().to_string(), Some(a.trim_end_matches(')').trim().to_string())),
            None => (sl.split(',').next().unwrap_or("").trim().to_string(), None),
        };
        let exclude = sl.contains("exclude") || sl.contains("excl");
        if rows.len() < k {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let seed = |a: &Option<String>| {
            if let Some(s) = a.as_ref().and_then(|s| s.parse::<f64>().ok()) {
                st.rng.borrow_mut().reseed(s as u64);
            }
        };
        let nr = rows.len();
        let centers: Vec<Vec<f64>> = if head.starts_with("kr") {
            seed(&arg);
            let mut idx: Vec<usize> = (0..nr).collect();
            let mut picked = Vec::new();
            for _ in 0..k {
                let u = st.rng.borrow_mut().uniform();
                let j = ((u * idx.len() as f64) as usize).min(idx.len() - 1);
                picked.push(idx.remove(j));
            }
            picked.iter().map(|&t| row_of(rows[t])).collect()
        } else if head.starts_with("f") {
            let c: Vec<Vec<f64>> = rows[..k].iter().map(|&i| row_of(i)).collect();
            if exclude {
                rows.drain(..k);
            }
            c
        } else if head.starts_with("la") {
            let c: Vec<Vec<f64>> = rows[nr - k..].iter().map(|&i| row_of(i)).collect();
            if exclude {
                rows.truncate(nr - k);
            }
            c
        } else if head.starts_with("r") {
            seed(&arg);
            let p = vars.len();
            let (mn, mx): (Vec<f64>, Vec<f64>) = (0..p).map(|v| rows.iter().map(|&i| cols[v][i]).fold((f64::INFINITY, f64::NEG_INFINITY), |a, x| (a.0.min(x), a.1.max(x)))).unzip();
            (0..k).map(|_| (0..p).map(|v| mn[v] + st.rng.borrow_mut().uniform() * (mx[v] - mn[v])).collect()).collect()
        } else if head.starts_with("p") || head.starts_with("ev") || head.starts_with("seg") || head.starts_with("g") {
            let g: Vec<usize> = if head.starts_with("p") {
                seed(&arg);
                (0..nr).map(|_| ((st.rng.borrow_mut().uniform() * k as f64) as usize).min(k - 1)).collect()
            } else if head.starts_with("ev") {
                (0..nr).map(|t| t % k).collect()
            } else if head.starts_with("seg") {
                (0..nr).map(|t| (t * k / nr).min(k - 1)).collect()
            } else {
                let gv = arg.clone().ok_or_else(|| rc_err(198, "group() requires a variable"))?;
                let gc = util::col(ds, &gv)?;
                let mut levels: Vec<f64> = rows.iter().map(|&i| gc[i]).filter(|v| !is_missing(*v)).collect();
                levels.sort_by(|a, b| a.total_cmp(b));
                levels.dedup();
                rows.iter().map(|&i| levels.iter().position(|&l| l == gc[i]).unwrap_or(0)).collect()
            };
            let x: Vec<Vec<f64>> = rows.iter().map(|&i| row_of(i)).collect();
            centers_of(&x, &g, k, med)
        } else {
            return Err(rc_err(198, format!("invalid start() option {start}")));
        };
        let x: Vec<Vec<f64>> = rows.iter().map(|&i| row_of(i)).collect();
        let maxit = mvutil::opt_usize(cmd, "iterate", 4)?.unwrap_or(10000);
        let (g, centers) = kpartition(&x, centers, m, med, maxit);
        let mut out = Vec::new();
        let name = match cmd.opt("name", 1) {
            Some(n) => n.trim().to_string(),
            None => {
                let n = default_name(ds);
                out.push(format!("cluster name: {n}"));
                n
            }
        };
        let genv = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| name.clone());
        let mut gv = vec![SYSMISS; ds.n_obs];
        for (t, &i) in rows.iter().enumerate() {
            gv[i] = (g[t] + 1) as f64;
        }
        set_var(ds, &genv, int_type(k as f64), &gv)?;
        if cmd.has_opt("keepcenters", 5) {
            let base = ds.n_obs;
            for _ in 0..k {
                ds.add_observation();
            }
            for (v, var) in vars.iter().enumerate() {
                let idx = ds.var_index[var];
                let vt = ds.vars[idx].var_type.clone();
                for c in 0..k {
                    ds.vars[idx].data[base + c] = crate::data::store_num(vt.clone(), centers[c][v]);
                }
            }
        }
        let other = vec![
            ("cmd".to_string(), format!("cluster {} {}", if med { "kmedians" } else { "kmeans" }, cmd.raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim().split_once(char::is_whitespace).map(|x| x.1).unwrap_or(""))),
            ("varlist".into(), vars.join(" ")),
            ("k".into(), k.to_string()),
            ("start".into(), start.clone()),
            ("range".into(), m.range().to_string()),
        ];
        push_obj(ds, ClObj { name, typ: "partition".into(), method: if med { "kmedians".into() } else { "kmeans".into() }, measure: m.name(), sim: m.is_sim(), vars: vec![(genv, "group".into())], other, notes: vec![] });
        st.data_changed = true;
        Ok(out.join("\n"))
    }
}

fn shape_matrix(m: &Matrix, shape: &str) -> Result<Matrix, String> {
    let s = shape.trim().to_lowercase();
    if s.starts_with('f') {
        if m.rows != m.cols {
            return Err(rc_err(505, "matrix not square"));
        }
        return Ok(m.clone());
    }
    let v = &m.data;
    let strict = s == "llower" || s == "uupper";
    let disc = if strict { 1.0 + 8.0 * v.len() as f64 } else { 1.0 + 8.0 * v.len() as f64 };
    let k = if strict { ((1.0 + disc.sqrt()) / 2.0).round() as usize } else { ((disc.sqrt() - 1.0) / 2.0).round() as usize };
    let expect = if strict { k * (k - 1) / 2 } else { k * (k + 1) / 2 };
    if expect != v.len() {
        return Err(rc_err(503, "matrix has the wrong number of elements"));
    }
    let mut f = Matrix::zeros(k, k);
    let mut idx = 0;
    for i in 0..k {
        let js: Vec<usize> = if s.starts_with('l') { if strict { (0..i).collect() } else { (0..=i).collect() } } else if strict { (i + 1..k).collect() } else { (i..k).collect() };
        for j in js {
            f[(i, j)] = v[idx];
            f[(j, i)] = v[idx];
            idx += 1;
        }
    }
    Ok(f)
}

// ---------------------------------------------------------------------------
// generate / stop / list / notes / dendrogram
// ---------------------------------------------------------------------------

fn generate(ds: &mut DataSet, cmd: &Command, rest: &str) -> Result<String, String> {
    let (lhs, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "= expected"))?;
    let lhs = lhs.trim().split_whitespace().last().unwrap_or("").to_string();
    let rhs = rhs.trim().to_string();
    let o = find_obj(ds, cmd.opt("name", 1))?;
    let (rows, cut, user) = tree_of(ds, &o)?;
    let n = rows.len();
    let ties = cmd.opt("ties", 3).unwrap_or("error").trim().to_lowercase();
    let make = |ds: &mut DataSet, name: &str, g: &[usize]| -> Result<(), String> {
        let mut v = vec![SYSMISS; ds.n_obs];
        for (t, &i) in rows.iter().enumerate() {
            v[i] = g[t] as f64;
        }
        let maxg = g.iter().copied().max().unwrap_or(1) as f64;
        set_var(ds, name, int_type(maxg), &v)
    };
    if let Some(a) = rhs.strip_prefix("cut(").or_else(|| rhs.strip_prefix("cut (")) {
        let val: f64 = a.trim_end_matches(')').trim().parse().map_err(|_| rc_err(198, "cut() requires a number"))?;
        let mut g = vec![0usize; n];
        let mut cur = 1;
        for t in 0..n {
            g[t] = cur;
            if t + 1 < n {
                let h = user[t];
                let split = if o.sim { h < val } else { h > val };
                if split {
                    cur += 1;
                }
            }
        }
        make(ds, &lhs, &g)?;
        return Ok(String::new());
    }
    let inner = rhs.split_once('(').map(|x| x.1.trim_end_matches(')').to_string()).ok_or_else(|| rc_err(198, "groups() or cut() required"))?;
    if !(rhs.starts_with("group")) {
        return Err(rc_err(198, "groups() or cut() required"));
    }
    let ks: Vec<usize> = crate::margins::numlist(&inner)?.into_iter().map(|v| v as usize).collect();
    let multi = ks.len() > 1;
    for &k in &ks {
        let g = match groups_k(&cut, k) {
            Some(g) => g,
            None => match ties.as_str() {
                "skip" => continue,
                "fewer" => (1..k).rev().find_map(|kk| groups_k(&cut, kk)).ok_or_else(|| rc_err(198, "cannot create groups"))?,
                "more" => (k + 1..=n).find_map(|kk| groups_k(&cut, kk)).ok_or_else(|| rc_err(198, "cannot create groups"))?,
                _ => return Err(rc_err(198, format!("cannot create {k} groups because of ties"))),
            },
        };
        let nm = if multi { format!("{lhs}{k}") } else { lhs.clone() };
        make(ds, &nm, &g)?;
    }
    Ok(String::new())
}

fn stop(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let o = find_obj(ds, rest.split_whitespace().next())?;
    let rule = cmd.opt("rule", 1).unwrap_or("calinski").trim().to_lowercase();
    let duda = rule.starts_with('d');
    let vars: Vec<String> = match cmd.opt("variables", 1) {
        Some(v) => crate::varlist::expand(ds, v)?,
        None => o.other("varlist").map(|s| s.split_whitespace().map(String::from).collect()).filter(|v: &Vec<String>| !v.is_empty()).ok_or_else(|| rc_err(198, "variables() required"))?,
    };
    let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    // groupings to evaluate
    let mut sols: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new(); // (k, rows, groups)
    if o.typ == "partition" {
        if duda {
            return Err(rc_err(198, "rule(duda) requires a hierarchical cluster analysis"));
        }
        let gv = o.var("group").unwrap_or_default();
        let gc = util::col(ds, &gv)?;
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(gc[i]) && cols.iter().all(|c| !is_missing(c[i]))).collect();
        let g: Vec<usize> = rows.iter().map(|&i| gc[i] as usize).collect();
        let k = g.iter().copied().max().unwrap_or(1);
        sols.push((k, rows, g));
    } else {
        let (rows, cut, _) = tree_of(ds, &o)?;
        let ks: Vec<usize> = match cmd.opt("groups", 1) {
            Some(s) => crate::margins::numlist(s)?.into_iter().map(|v| v as usize).collect(),
            None => {
                if duda {
                    (1..=15).collect()
                } else {
                    (2..=15).collect()
                }
            }
        };
        for k in ks {
            if k > rows.len() || (!duda && k < 2) {
                continue;
            }
            if let Some(g) = groups_k(&cut, k) {
                if duda {
                    if let Some(g2) = groups_k(&cut, k + 1) {
                        sols.push((k, rows.clone(), g.iter().zip(&g2).map(|(a, b)| a * 100000 + b).collect()));
                    }
                } else {
                    sols.push((k, rows.clone(), g));
                }
            }
        }
    }
    let sse = |rows: &[usize], members: &[usize]| -> f64 {
        let mut s = 0.0;
        for c in &cols {
            let vals: Vec<f64> = members.iter().map(|&t| c[rows[t]]).collect();
            let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
            s += vals.iter().map(|v| (v - m).powi(2)).sum::<f64>();
        }
        s
    };
    let mut out = vec![String::new()];
    let r = st.begin_r();
    let mut mat_rows = Vec::new();
    if !duda {
        out.push("+---------------------------+".into());
        out.push("|             |  Calinski/  |".into());
        out.push("|  Number of  |  Harabasz   |".into());
        out.push("|  clusters   |  pseudo-F   |".into());
        out.push("|-------------+-------------|".into());
        for (k, rows, g) in &sols {
            let n = rows.len();
            let all: Vec<usize> = (0..n).collect();
            let t = sse(rows, &all);
            let mut w = 0.0;
            for c in 1..=*k {
                let mem: Vec<usize> = (0..n).filter(|&t| g[t] == c).collect();
                if !mem.is_empty() {
                    w += sse(rows, &mem);
                }
            }
            let ch = ((t - w) / (*k as f64 - 1.0)) / (w / (n as f64 - *k as f64));
            out.push(format!("|  {:^9}  |  {:>9.2}  |", k, ch));
            r.set(&format!("calinski_{k}"), ch);
            mat_rows.push(vec![*k as f64, ch]);
        }
        out.push("+---------------------------+".into());
        r.set_str("rule", "calinski");
        r.set_str("label", "C-H pseudo-F");
        r.set_str("longlabel", "Calinski & Harabasz pseudo-F");
    } else {
        out.push("+-----------------------------------------+".into());
        out.push("|             |         Duda/Hart         |".into());
        out.push("|  Number of  |             |  pseudo     |".into());
        out.push("|  clusters   | Je(2)/Je(1) |  T-squared  |".into());
        out.push("|-------------+-------------+-------------|".into());
        for (k, rows, code) in &sols {
            let n = rows.len();
            // the group of the k-solution that splits in the k+1-solution
            let mut found = None;
            for c in 1..=*k {
                let mem: Vec<usize> = (0..n).filter(|&t| code[t] / 100000 == c).collect();
                let mut subs: Vec<usize> = mem.iter().map(|&t| code[t] % 100000).collect();
                subs.sort();
                subs.dedup();
                if subs.len() == 2 {
                    let a: Vec<usize> = mem.iter().copied().filter(|&t| code[t] % 100000 == subs[0]).collect();
                    let b: Vec<usize> = mem.iter().copied().filter(|&t| code[t] % 100000 == subs[1]).collect();
                    found = Some((mem, a, b));
                    break;
                }
            }
            let Some((mem, a, b)) = found else { continue };
            let je1 = sse(rows, &mem);
            let je2 = sse(rows, &a) + sse(rows, &b);
            let ratio = je2 / je1;
            let t2 = if je2 > 0.0 { (je1 / je2 - 1.0) * ((a.len() + b.len()) as f64 - 2.0) } else { SYSMISS };
            out.push(format!("|  {:^9}  |  {:>9.4}  |  {:>9}  |", k, ratio, if is_missing(t2) { ".".to_string() } else { format!("{t2:.2}") }));
            r.set(&format!("duda_{k}"), ratio);
            r.set(&format!("dudat2_{k}"), t2);
            mat_rows.push(vec![*k as f64, ratio, t2]);
        }
        out.push("+-----------------------------------------+".into());
        r.set_str("rule", "duda");
        r.set_str("label", "D-H Je(2)/Je(1)");
        r.set_str("longlabel", "Duda & Hart Je(2)/Je(1)");
        r.set_str("label2", "D-H pseudo-T-squared");
        r.set_str("longlabel2", "Duda & Hart pseudo-T-squared");
    }
    if let Some(mn) = cmd.opt("matrix", 3) {
        let cn: Vec<String> = if duda { vec!["groups".into(), "Je2_Je1".into(), "T2".into()] } else { vec!["groups".into(), "calinski".into()] };
        let m = Matrix::from_rows(&mat_rows).with_names(mvutil::names("r", mat_rows.len()), cn);
        st.matrices.insert(mn.trim().to_string(), m);
    }
    Ok(out.join("\n"))
}

fn list(ds: &DataSet, cmd: &Command, rest: &str) -> Result<String, String> {
    let objs = load_objs(ds);
    let pats: Vec<&str> = rest.split_whitespace().collect();
    let sel: Vec<&ClObj> = if pats.is_empty() { objs.iter().collect() } else { objs.iter().filter(|o| pats.iter().any(|p| glob(p, &o.name))).collect() };
    let any_opt = ["notes", "type", "method", "dissimilarity", "similarity", "vars", "chars", "other"].iter().any(|o| cmd.has_opt(o, 3));
    let want = |o: &str| !any_opt || cmd.has_opt("all", 3) || cmd.has_opt(o, 3);
    let mut out = Vec::new();
    for o in sel {
        let mut head = Vec::new();
        if want("type") {
            head.push(format!("type: {}", o.typ));
        }
        if want("method") {
            head.push(format!("method: {}", o.method));
        }
        if (o.sim && want("similarity")) || (!o.sim && want("dissimilarity")) {
            head.push(format!("{}: {}", if o.sim { "similarity" } else { "dissimilarity" }, o.measure));
        }
        out.push(if head.is_empty() { o.name.clone() } else { format!("{}  ({})", o.name, head.join(",  ")) });
        if want("notes") {
            for (i, n) in o.notes.iter().enumerate() {
                out.push(format!("{:>10} {}", if i == 0 { "notes:" } else { "" }, n));
            }
        }
        if want("vars") {
            for (i, (v, r)) in o.vars.iter().enumerate() {
                out.push(format!("{:>10} {} ({} variable)", if i == 0 { "vars:" } else { "" }, v, r));
            }
        }
        if want("other") {
            for (i, (k, v)) in o.other.iter().enumerate() {
                out.push(format!("{:>10} {}: {}", if i == 0 { "other:" } else { "" }, k, v));
            }
        }
    }
    Ok(out.join("\n"))
}

fn notes(ds: &mut DataSet, cmd: &Command, rest: &str) -> Result<String, String> {
    let mut objs = load_objs(ds);
    let rest = rest.trim();
    // cluster notes clname : text   |  cluster notes [clname]  |  cluster notes drop clname [in numlist]
    if let Some(r) = rest.strip_prefix("drop") {
        let r = r.trim();
        let (nm, inl) = match r.split_once(" in ") {
            Some((a, b)) => (a.trim(), Some(b.trim())),
            None => (r, None),
        };
        let o = objs.iter_mut().find(|o| glob(nm, &o.name)).ok_or_else(|| rc_err(198, format!("{nm} not found")))?;
        match inl {
            Some(l) => {
                let del: Vec<usize> = crate::margins::numlist(l)?.into_iter().map(|v| v as usize).collect();
                let kept: Vec<String> = o.notes.iter().enumerate().filter(|(i, _)| !del.contains(&(i + 1))).map(|(_, n)| n.clone()).collect();
                o.notes = kept;
            }
            None => o.notes.clear(),
        }
        save_objs(ds, &objs);
        return Ok(String::new());
    }
    if let Some((nm, text)) = rest.split_once(':') {
        let nm = nm.trim();
        let o = objs.iter_mut().find(|o| glob(nm, &o.name)).ok_or_else(|| rc_err(198, format!("{nm} not found")))?;
        o.notes.push(text.trim().to_string());
        save_objs(ds, &objs);
        return Ok(String::new());
    }
    let _ = cmd;
    let mut out = Vec::new();
    for o in objs.iter().filter(|o| rest.is_empty() || rest.split_whitespace().any(|p| glob(p, &o.name))) {
        if o.notes.is_empty() {
            continue;
        }
        out.push(String::new());
        out.push(o.name.clone());
        for (i, n) in o.notes.iter().enumerate() {
            out.push(format!("{:>4}.  {}", i + 1, n));
        }
    }
    Ok(out.join("\n"))
}

fn dendrogram(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let o = find_obj(ds, rest.split_whitespace().next())?;
    let (rows, cut, user) = tree_of(ds, &o)?;
    if o.vars.iter().any(|(_, r)| r == "pseudo_height") {
        return Err(rc_err(198, "dendrogram not available because of reversals in the hierarchy"));
    }
    let n = rows.len();
    // rebuild merges from adjacency heights: repeatedly merge the adjacent pair with the lowest height
    #[derive(Clone)]
    struct Node {
        x: f64,
        h: f64,
        size: usize,
    }
    let mut nodes: Vec<Node> = (0..n).map(|i| Node { x: (i + 1) as f64, h: if o.sim { user.iter().filter(|v| !is_missing(**v)).cloned().fold(f64::NEG_INFINITY, f64::max) } else { 0.0 }, size: 1 }).collect();
    if o.sim {
        // leaves at the top of the similarity scale
        let top = user.iter().filter(|v| !is_missing(**v)).cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
        for nd in nodes.iter_mut() {
            nd.h = top;
        }
    }
    let mut bounds: Vec<f64> = cut[..n.saturating_sub(1)].to_vec();
    let mut ubounds: Vec<f64> = user[..n.saturating_sub(1)].to_vec();
    let mut segs: Vec<[f64; 4]> = Vec::new();
    let cutnum = mvutil::opt_usize(cmd, "cutnumber", 4)?;
    let cutval = util::opt_num(cmd, "cutvalue", 4)?;
    let mut leaves_shown = n;
    while nodes.len() > 1 {
        if let Some(cn) = cutnum {
            if nodes.len() <= cn {
                break;
            }
        }
        let (mut bi, mut bv) = (0, f64::INFINITY);
        for (i, &b) in bounds.iter().enumerate() {
            if b < bv {
                bv = b;
                bi = i;
            }
        }
        let h = ubounds[bi];
        if let Some(cv) = cutval {
            if (!o.sim && h > cv) || (o.sim && h < cv) {
                break;
            }
        }
        let (a, b) = (nodes[bi].clone(), nodes[bi + 1].clone());
        segs.push([a.h, a.x, h, a.x]);
        segs.push([b.h, b.x, h, b.x]);
        segs.push([h, a.x, h, b.x]);
        let nx = (a.x * 1.0 + b.x * 1.0) / 2.0;
        nodes[bi] = Node { x: nx, h, size: a.size + b.size };
        nodes.remove(bi + 1);
        bounds.remove(bi);
        ubounds.remove(bi);
        leaves_shown = nodes.len();
    }
    let _ = leaves_shown;
    let mut g = Graph { title: String::new(), ytitle: format!("{} {} {}", o.measure, if o.sim { "similarity" } else { "dissimilarity" }, "measure"), ..Default::default() };
    if cutnum.is_some() || cutval.is_some() {
        // redraw on compressed positions: one slot per remaining node
        let mut map: Vec<(f64, f64)> = nodes.iter().enumerate().map(|(i, nd)| (nd.x, (i + 1) as f64)).collect();
        map.sort_by(|a, b| a.0.total_cmp(&b.0));
        let _ = map;
        for (i, nd) in nodes.iter().enumerate() {
            g.xlabels.push((nd.x, format!("G{}{}", i + 1, if cmd.has_opt("showcount", 4) { format!("\nn={}", nd.size) } else { String::new() })));
        }
        segs.retain(|s| s[0] != s[2] || true);
    } else {
        let labels: Vec<String> = match cmd.opt("labels", 3) {
            Some(lv) => {
                let var = ds.get_var(lv.trim()).ok_or_else(|| rc_err(111, format!("variable {} not found", lv.trim())))?;
                rows.iter().map(|&i| crate::util::cell_text(ds, var, i, true)).collect()
            }
            None => rows.iter().map(|&i| (i + 1).to_string()).collect(),
        };
        for (i, l) in labels.into_iter().enumerate() {
            g.xlabels.push(((i + 1) as f64, l));
        }
    }
    g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Spike), segments: segs, label: String::new(), ..Default::default() });
    g.title = format!("Dendrogram for {} cluster analysis", o.name);
    g.command = cmd.raw.clone();
    finish(&mut g, cmd, st)
}
