//! Test harness for the Ferrostat behavioural suite.
//!
//! Tests drive the interpreter exactly like a user would (one command line at
//! a time) and check either the resulting dataset or the numbers printed in
//! the output.
//!
//! Conventions used across `tests/`:
//! - `#[ignore = "missing: ..."]`: feature not implemented yet.
//! - `#[ignore = "bug: ..."]`: feature exists but returns a wrong result.
//!
//! `cargo test -p ferrostat-tests` runs what currently works;
//! `cargo test -p ferrostat-tests -- --ignored` shows the remaining gap.

pub mod reference;

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use ferrostat_ado::interpreter::AdoInterpreter;
use ferrostat_core::{DataSet, Value, ValueLabel, VarType, Variable};

/// Estimation results are kept in process-wide state, so sessions must not
/// run concurrently.
static SESSION_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// Sessions currently open on this thread (the lock is re-entrant per thread).
    static HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub struct Session {
    interp: AdoInterpreter,
    _guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for Session {
    fn drop(&mut self) {
        HELD.with(|h| h.set(h.get() - 1));
    }
}

impl Session {
    /// An empty session (no data in memory).
    pub fn new() -> Self {
        let guard = if HELD.with(|h| h.get()) == 0 { Some(SESSION_LOCK.lock().unwrap_or_else(|e| e.into_inner())) } else { None };
        HELD.with(|h| h.set(h.get() + 1));
        Self { interp: AdoInterpreter::new(), _guard: guard }
    }

    /// A session with the standard 12-observation fixture in memory.
    pub fn with_fixture() -> Self {
        let mut s = Self::new();
        s.interp.engine_mut().dataset = fixture();
        s
    }

    /// Same as `with_fixture`, but `y` is missing in observation 12 and
    /// `z` is missing in observation 1.
    pub fn with_missing_fixture() -> Self {
        let mut s = Self::new();
        s.interp.engine_mut().dataset = fixture_with_missing();
        s
    }

    /// Replace the data in memory.
    pub fn load(&mut self, ds: DataSet) {
        self.interp.engine_mut().dataset = ds;
    }

    /// Run one command; panics (failing the test) if the command errors.
    pub fn run(&mut self, line: &str) -> String {
        match self.interp.run_line(line) {
            Ok(out) => out,
            Err(e) => panic!("command failed: `{line}`\n  error: {e}"),
        }
    }

    pub fn try_run(&mut self, line: &str) -> Result<String, String> {
        self.interp.run_line(line)
    }

    /// Run a multi-line script (do-file body); returns the joined output.
    pub fn script(&mut self, text: &str) -> String {
        let mut out = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let o = self.run(line);
            if !o.is_empty() {
                out.push(o);
            }
        }
        out.join("\n")
    }

    /// Assert that a command is rejected.
    pub fn fails(&mut self, line: &str) {
        if let Ok(out) = self.interp.run_line(line) {
            panic!("expected `{line}` to fail, but it succeeded with:\n{out}");
        }
    }

    /// Evaluate a scalar expression through `display` (e.g. `r(mean)`,
    /// `_b[x]`, `e(N)`) and parse the printed number.
    pub fn scalar(&mut self, expr: &str) -> f64 {
        let out = self.run(&format!("display %21.0g {expr}"));
        out.trim().parse::<f64>().unwrap_or_else(|_| {
            panic!("`display {expr}` did not print a number, got: {out:?}")
        })
    }

    pub fn ds(&self) -> &DataSet {
        &self.interp.engine().dataset
    }

    pub fn n(&self) -> usize {
        self.ds().n_obs
    }

    pub fn varnames(&self) -> Vec<String> {
        self.ds().varlist()
    }

    pub fn has_var(&self, name: &str) -> bool {
        self.ds().get_var(name).is_some()
    }

    /// Numeric column; missing values come back as NaN.
    pub fn nums(&self, var: &str) -> Vec<f64> {
        let v = self.ds().get_var(var).unwrap_or_else(|| panic!("no variable {var}"));
        v.data
            .iter()
            .map(|x| {
                let n = x.num();
                if ferrostat_core::is_missing(n) { f64::NAN } else { n }
            })
            .collect()
    }

    /// String column.
    pub fn strs(&self, var: &str) -> Vec<String> {
        let v = self.ds().get_var(var).unwrap_or_else(|| panic!("no variable {var}"));
        v.data.iter().map(|x| x.as_string().unwrap_or("").to_string()).collect()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Find `label` in the output and return the first number after it,
/// e.g. `stat(out, "R-squared")` on "R-squared     = 0.9997".
pub fn stat(output: &str, label: &str) -> Option<f64> {
    let pos = output.find(label)?;
    numbers(&output[pos + label.len()..]).into_iter().next()
}

/// Numbers on the table row whose first column (left of `|`) is `name`.
pub fn row(output: &str, name: &str) -> Option<Vec<f64>> {
    output.lines().find_map(|l| {
        let (head, rest) = l.split_once('|')?;
        (head.trim() == name).then(|| numbers(rest))
    })
}

/// All numbers appearing in a string, in order.
pub fn numbers(s: &str) -> Vec<f64> {
    s.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == '+'))
        .filter_map(|t| t.trim_matches(|c| c == 'e' || c == '+').parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

/// Relative/absolute closeness check with a readable failure message.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    assert!(
        diff <= tol * scale,
        "expected {expected}, got {actual} (diff {diff:e}, tol {tol:e})"
    );
}

#[track_caller]
pub fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert_close(*a, *e, tol);
    }
}

/// A fresh, empty temporary directory for one test.
pub fn temp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferrostat-tests-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

pub const NAMES: [&str; 12] = [
    "ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "ibis", "jay", "koi", "lynx",
];
pub const GROUP: [f64; 12] = [1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.];
pub const X: [f64; 12] = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
pub const Y: [f64; 12] = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 19.9, 22.2, 23.8];
pub const Z: [f64; 12] = [3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.];
pub const D: [f64; 12] = [0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1.];
pub const CNT: [f64; 12] = [0., 1., 1., 2., 1., 3., 2., 4., 3., 5., 4., 6.];

/// The standard synthetic fixture. Must stay in sync with reference.py.
pub fn fixture() -> DataSet {
    let mut ds = DataSet::new();
    let mut add = |name: &str, ty: VarType, data: Vec<Value>| {
        let mut v = Variable::new(name, ty);
        v.data = data;
        ds.add_variable(v).unwrap();
    };
    add("id", VarType::Int, (1..=12).map(Value::Int).collect());
    add("name", VarType::Str(4), NAMES.iter().map(|s| Value::Str(s.to_string())).collect());
    add("group", VarType::Byte, GROUP.iter().map(|&v| Value::Byte(v as i8)).collect());
    add("x", VarType::Double, X.iter().map(|&v| Value::Double(v)).collect());
    add("y", VarType::Double, Y.iter().map(|&v| Value::Double(v)).collect());
    add("z", VarType::Double, Z.iter().map(|&v| Value::Double(v)).collect());
    add("d", VarType::Byte, D.iter().map(|&v| Value::Byte(v as i8)).collect());
    add("cnt", VarType::Int, CNT.iter().map(|&v| Value::Int(v as i16)).collect());
    ds.n_obs = 12;

    let mut yesno = ValueLabel::new("yesno");
    yesno.add(0.0, "no");
    yesno.add(1.0, "yes");
    ds.value_labels.insert("yesno".into(), yesno);
    let d = ds.get_var_mut("d").unwrap();
    d.value_label_name = "yesno".into();
    d.label = "Binary outcome".into();
    ds.datalabel = "Ferrostat test fixture".into();
    ds
}

pub fn fixture_with_missing() -> DataSet {
    let mut ds = fixture();
    ds.get_var_mut("y").unwrap().data[11] = Value::Double(f64::NAN);
    ds.get_var_mut("z").unwrap().data[0] = Value::Double(f64::NAN);
    ds
}
