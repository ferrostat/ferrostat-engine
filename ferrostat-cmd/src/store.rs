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
            for (k, v) in &self.scalars {
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
    /// Current graph and named graphs in memory.
    pub graph: Option<crate::graph::Graph>,
    pub graphs: BTreeMap<String, crate::graph::Graph>,
    /// Working directory set by `cd` (None = process directory).
    pub cwd: Option<std::path::PathBuf>,
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
            graph: None,
            graphs: BTreeMap::new(),
            cwd: None,
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
