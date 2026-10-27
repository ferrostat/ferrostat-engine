//! Interpreter state shared by all commands: stored results, scalars,
//! matrices, macros, random-number generator, preserved data.

use std::collections::BTreeMap;

use ferrostat_core::DataSet;

use crate::matrix::Matrix;
use crate::rng::Rng;

/// One class of stored results (`r()`, `e()` or `s()`).
#[derive(Debug, Clone, Default)]
pub struct Results {
    pub scalars: BTreeMap<String, f64>,
    pub macros: BTreeMap<String, String>,
    pub matrices: BTreeMap<String, Matrix>,
    /// e(sample): estimation-sample marker, one entry per observation.
    pub sample: Option<Vec<bool>>,
}

impl Results {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.scalars.is_empty() && self.macros.is_empty() && self.matrices.is_empty()
    }

    pub fn set(&mut self, name: &str, v: f64) {
        self.scalars.insert(name.to_string(), v);
    }

    pub fn set_str(&mut self, name: &str, v: impl Into<String>) {
        self.macros.insert(name.to_string(), v.into());
    }

    pub fn set_mat(&mut self, name: &str, m: Matrix) {
        self.matrices.insert(name.to_string(), m);
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.scalars.get(name).copied()
    }

    /// `return list` / `ereturn list` style listing.
    pub fn listing(&self, prefix: char) -> String {
        let mut out = Vec::new();
        if !self.scalars.is_empty() {
            out.push("scalars:".to_string());
            for (k, v) in scalar_order(&self.scalars) {
                out.push(format!("{:>20} =  {}", format!("{prefix}({k})"), crate::format::fmt_g(*v, 17)));
            }
        }
        if !self.macros.is_empty() {
            out.push("macros:".to_string());
            for (k, v) in &self.macros {
                out.push(format!("{:>20} : \"{}\"", format!("{prefix}({k})"), v));
            }
        }
        if !self.matrices.is_empty() {
            out.push("matrices:".to_string());
            for (k, m) in &self.matrices {
                out.push(format!("{:>20} :  {} x {}", format!("{prefix}({k})"), m.rows, m.cols));
            }
        }
        if self.sample.is_some() {
            out.push("functions:".to_string());
            out.push(format!("{:>20}", format!("{prefix}(sample)")));
        }
        out.join("\n")
    }
}

/// Scalars in listing order: alphabetical, except for the summary-statistics
/// set, which is listed in the order it is computed (N, sum_w, mean, ...).
fn scalar_order(sc: &BTreeMap<String, f64>) -> Vec<(&String, &f64)> {
    const PLAIN: [&str; 8] = ["N", "sum_w", "mean", "Var", "sd", "min", "max", "sum"];
    const DETAIL: [&str; 19] = ["N", "sum_w", "mean", "Var", "sd", "skewness", "kurtosis", "sum", "min", "max", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];
    let mut v: Vec<(&String, &f64)> = sc.iter().collect();
    for order in [&PLAIN[..], &DETAIL[..]] {
        if sc.contains_key("sum_w") && sc.keys().all(|k| order.contains(&k.as_str())) {
            v.sort_by_key(|(k, _)| order.iter().position(|o| o == k));
            break;
        }
    }
    v
}

/// A user-defined program.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub name: String,
    pub body: Vec<String>,
    /// rclass / eclass / sclass / nclass
    pub class: String,
    pub byable: bool,
}

#[derive(Debug, Clone)]
pub struct Store {
    pub r: Results,
    pub e: Results,
    pub s: Results,
    pub scalars: BTreeMap<String, f64>,
    pub str_scalars: BTreeMap<String, String>,
    pub matrices: BTreeMap<String, Matrix>,
    pub globals: BTreeMap<String, String>,
    /// Local-macro frames; the last one is current.
    pub frames: Vec<BTreeMap<String, String>>,
    pub programs: BTreeMap<String, Program>,
    pub rng: std::cell::RefCell<Rng>,
    pub preserved: Vec<DataSet>,
    pub estimates: BTreeMap<String, Results>,
    /// Names of stored estimates in the order they were stored.
    pub estimates_order: Vec<String>,
    /// Confidence level for estimation output (`set level`).
    pub level: f64,
    pub varabbrev: bool,
    /// `set rmsg on`: report elapsed time after each command.
    pub rmsg: bool,
    /// Return code of the last command (`_rc`).
    pub rc: i32,
    /// Panel and time variables set by tsset/xtset.
    pub tsset: Option<(Option<String>, String)>,
    pub data_changed: bool,
    /// Set while a `by` prefix is executing: observations of the current group.
    pub by_mask: Option<Vec<bool>>,
    /// Set while a `by` prefix is executing for generate/replace/egen.
    pub by_groups: Option<crate::eval::Groups>,
    /// Default storage type for generate (`set type`).
    pub default_type: ferrostat_core::VarType,
    pub filename: String,
    /// Colour scheme for new graphs (`set scheme`); empty = default (Okabe-Ito).
    pub scheme: String,
    /// `set graphics off`: graphs are drawn but not shown in a window.
    pub graphics_off: bool,
    /// `graph close` closed the graph window (reopened by the next graph).
    pub graph_window_closed: bool,
    /// `set printcolor automatic|asis|gs1|gs2|gs3`.
    pub printcolor: String,
    /// `graph set <group> <setting> <value>`: (group, setting) -> value.
    pub graph_settings: BTreeMap<(String, String), String>,
    /// Current graph and named graphs in memory.
    pub graph: Option<crate::graph::Graph>,
    pub graphs: BTreeMap<String, crate::graph::Graph>,
    /// Working directory set by `cd` (None = process directory).
    pub cwd: Option<std::path::PathBuf>,
    /// Data frames other than the current one, by name.
    pub data_frames: BTreeMap<String, crate::frames::Frame>,
    /// Name of the current data frame.
    pub cur_frame: String,
    /// Values of `set` options kept only for reporting through `c()`.
    pub settings: BTreeMap<String, String>,
    /// Result collections (`collect`, `table`, `dtable`, `etable`).
    pub collect: crate::collect::Collections,
}

impl Default for Store {
    fn default() -> Self {
        Self {
            r: Results::new(),
            e: Results::new(),
            s: Results::new(),
            scalars: BTreeMap::new(),
            str_scalars: BTreeMap::new(),
            matrices: BTreeMap::new(),
            globals: default_globals(),
            frames: vec![BTreeMap::new()],
            programs: BTreeMap::new(),
            rng: std::cell::RefCell::new(Rng::new(123456789)),
            preserved: Vec::new(),
            estimates: BTreeMap::new(),
            estimates_order: Vec::new(),
            level: 95.0,
            varabbrev: true,
            rmsg: false,
            rc: 0,
            tsset: None,
            data_changed: false,
            by_mask: None,
            by_groups: None,
            default_type: ferrostat_core::VarType::Float,
            filename: String::new(),
            scheme: String::new(),
            graphics_off: false,
            graph_window_closed: false,
            printcolor: String::new(),
            graph_settings: BTreeMap::new(),
            graph: None,
            graphs: BTreeMap::new(),
            cwd: None,
            data_frames: BTreeMap::new(),
            cur_frame: crate::frames::DEFAULT_FRAME.to_string(),
            settings: BTreeMap::new(),
            collect: Default::default(),
        }
    }
}

fn default_globals() -> BTreeMap<String, String> {
    let mut g = BTreeMap::new();
    g.insert("S_level".into(), "95".into());
    g
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn local(&self, name: &str) -> Option<&String> {
        self.frames.last().and_then(|f| f.get(name))
    }

    pub fn set_local(&mut self, name: &str, v: impl Into<String>) {
        if let Some(f) = self.frames.last_mut() {
            f.insert(name.to_string(), v.into());
        }
    }

    pub fn clear_local(&mut self, name: &str) {
        if let Some(f) = self.frames.last_mut() {
            f.remove(name);
        }
    }

    pub fn push_frame(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    pub fn pop_frame(&mut self) {
        if self.frames.len() > 1 {
            self.frames.pop();
        }
    }

    /// Start a fresh set of r() results (r-class commands do this).
    pub fn begin_r(&mut self) -> &mut Results {
        self.r = Results::new();
        &mut self.r
    }

    /// Start a fresh set of e() results (estimation commands do this).
    pub fn begin_e(&mut self) -> &mut Results {
        self.e = Results::new();
        &mut self.e
    }
}
