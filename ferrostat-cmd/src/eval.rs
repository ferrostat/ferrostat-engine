//! Expression evaluator.
//!
//! Semantics of the do-file expression language:
//! - missing values are larger than every number, so `. > 5` is true;
//! - arithmetic involving a missing value gives `.`; so do domain errors
//!   such as `ln(0)` or `1/0`;
//! - in logical context any nonzero value (including missing) is true;
//! - `+` concatenates strings and `*` repeats them; mixing strings and
//!   numbers in an operator is a type mismatch (r(109));
//! - `_n`/`_N` and explicit subscripts `x[_n-1]` respect `by` groups.

use std::cell::RefCell;
use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::{BinOp, Expr, UnaryOp};

use crate::store::Store;
use crate::{dates, dist};

/// Value of an evaluated expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Num(f64),
    Str(String),
}

impl Val {
    pub fn num(&self) -> Result<f64, String> {
        match self {
            Val::Num(n) => Ok(*n),
            Val::Str(_) => Err(type_mismatch()),
        }
    }

    pub fn str(&self) -> Result<&str, String> {
        match self {
            Val::Str(s) => Ok(s),
            Val::Num(_) => Err(type_mismatch()),
        }
    }

    pub fn is_true(&self) -> bool {
        match self {
            Val::Num(n) => *n != 0.0,
            Val::Str(s) => !s.is_empty(),
        }
    }

    pub fn is_missing(&self) -> bool {
        match self {
            Val::Num(n) => is_missing(*n),
            Val::Str(s) => s.is_empty(),
        }
    }

    /// Text as `display` would show it.
    pub fn display(&self) -> String {
        match self {
            Val::Num(n) => crate::format::display_number(*n),
            Val::Str(s) => s.clone(),
        }
    }
}

pub fn rc_err(code: i32, msg: impl std::fmt::Display) -> String {
    format!("{msg}\nr({code});")
}

pub fn type_mismatch() -> String {
    rc_err(109, "type mismatch")
}

/// Observation ranges of `by` groups: for each observation, the first and
/// one-past-last index of its group.
#[derive(Debug, Clone, Default)]
pub struct Groups {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
}

impl Groups {
    pub fn whole(n: usize) -> Self {
        Groups { start: vec![0; n], end: vec![n; n] }
    }
}

/// Evaluation environment.
pub struct Env<'a> {
    pub ds: &'a DataSet,
    pub st: Option<&'a Store>,
    pub groups: Option<&'a Groups>,
    /// Running-sum state for `sum()`, keyed by expression address.
    running: RefCell<HashMap<usize, (usize, f64)>>,
    /// Captures of the last `regexm()` match, for `regexs()`.
    regex_caps: RefCell<Vec<String>>,
    rng: Option<&'a RefCell<crate::rng::Rng>>,
    /// (panel, time) -> observation index, built on first time-series lookup.
    ts_index: RefCell<Option<HashMap<(u64, u64), usize>>>,
}

impl<'a> Env<'a> {
    pub fn new(ds: &'a DataSet) -> Self {
        Env { ds, st: None, groups: None, running: Default::default(), regex_caps: Default::default(), rng: None, ts_index: Default::default() }
    }

    pub fn with_store(ds: &'a DataSet, st: &'a Store) -> Self {
        Env {
            ds,
            st: Some(st),
            groups: None,
            running: Default::default(),
            regex_caps: Default::default(),
            rng: Some(&st.rng),
            ts_index: Default::default(),
        }
    }

    pub fn groups(mut self, g: Option<&'a Groups>) -> Self {
        self.groups = g;
        self
    }

    fn group_bounds(&self, row: usize) -> (usize, usize) {
        match self.groups {
            Some(g) if row < g.start.len() => (g.start[row], g.end[row]),
            _ => (0, self.ds.n_obs),
        }
    }

    /// Resolve a variable name (exact, then unique abbreviation).
    pub fn var_index(&self, name: &str) -> Option<usize> {
        if let Some(&i) = self.ds.var_index.get(name) {
            return Some(i);
        }
        if self.st.map_or(true, |s| s.varabbrev) {
            let mut hit = None;
            for (i, v) in self.ds.vars.iter().enumerate() {
                if v.name.starts_with(name) {
                    if hit.is_some() {
                        return None;
                    }
                    hit = Some(i);
                }
            }
            return hit;
        }
        None
    }

    fn cell(&self, vi: usize, row: usize) -> Val {
        let v = &self.ds.vars[vi];
        match v.data.get(row) {
            Some(Value::Str(s)) => Val::Str(s.clone()),
            Some(Value::StrL(a, b)) => Val::Str(self.ds.resolve_strl(*a, *b).unwrap_or("").to_string()),
            Some(x) if v.var_type.is_string() => Val::Str(x.as_string().unwrap_or("").to_string()),
            Some(x) => Val::Num(x.num()),
            None => {
                if v.var_type.is_string() { Val::Str(String::new()) } else { Val::Num(SYSMISS) }
            }
        }
    }

    pub fn eval(&self, e: &Expr, row: usize) -> Result<Val, String> {
        match e {
            Expr::Number(n) => Ok(Val::Num(*n)),
            Expr::String(s) => Ok(Val::Str(s.clone())),
            Expr::Missing => Ok(Val::Num(SYSMISS)),
            Expr::MissingCode(c) => Ok(Val::Num(missing(*c))),
            Expr::Paren(x) => self.eval(x, row),
            Expr::Var(name) => self.eval_name(name, row),
            Expr::Unary(op, x) => {
                let v = self.eval(x, row)?;
                match op {
                    UnaryOp::Neg => {
                        let n = v.num()?;
                        Ok(Val::Num(if is_missing(n) { SYSMISS } else { -n }))
                    }
                    UnaryOp::Not => Ok(Val::Num(if v.num()? != 0.0 { 0.0 } else { 1.0 })),
                }
            }
            Expr::Binary(l, op, r) => self.eval_binary(l, op, r, row),
            Expr::Index(base, idx) => self.eval_index(base, idx, row),
            Expr::Call(name, args) => self.eval_call(e, name, args, row),
        }
    }

    pub fn eval_num(&self, e: &Expr, row: usize) -> Result<f64, String> {
        self.eval(e, row)?.num()
    }

    fn eval_name(&self, name: &str, row: usize) -> Result<Val, String> {
        match name {
            "_n" => {
                let (s, _) = self.group_bounds(row);
                return Ok(Val::Num((row - s + 1) as f64));
            }
            "_N" => {
                let (s, e) = self.group_bounds(row);
                return Ok(Val::Num((e - s) as f64));
            }
            "_pi" => return Ok(Val::Num(std::f64::consts::PI)),
            "_rc" => return Ok(Val::Num(self.st.map_or(0, |s| s.rc) as f64)),
            _ => {}
        }
        if let Some(vi) = self.var_index(name) {
            return Ok(self.cell(vi, row));
        }
        if let Some(st) = self.st {
            if let Some(v) = st.scalars.get(name) {
                return Ok(Val::Num(*v));
            }
            if let Some(v) = st.str_scalars.get(name) {
                return Ok(Val::Str(v.clone()));
            }
        }
        if name.contains('.') {
            return self.eval_factor_ts(name, row);
        }
        Err(rc_err(111, format!("{name} not found")))
    }

    /// Time-series operators (L., F., D., S.) and factor-variable indicators
    /// (3.group) used directly in expressions.
    fn eval_factor_ts(&self, name: &str, row: usize) -> Result<Val, String> {
        let (op, var) = name.split_once('.').unwrap();
        let lower = op.to_lowercase();
        if let Ok(level) = op.parse::<f64>() {
            let v = self.eval_name(var, row)?.num()?;
            return Ok(Val::Num(if is_missing(v) { SYSMISS } else if v == level { 1.0 } else { 0.0 }));
        }
        let (kind, k) = match lower.chars().next() {
            Some(c @ ('l' | 'f' | 'd' | 's')) => {
                let n = lower[1..].parse::<i64>().unwrap_or(1);
                (c, n)
            }
            _ => return Err(rc_err(111, format!("{name} not found"))),
        };
        let lag = |r: usize, k: i64| -> Result<f64, String> { self.ts_value(var, r, k) };
        let v = match kind {
            'l' => lag(row, k)?,
            'f' => lag(row, -k)?,
            'd' => {
                let mut cur: Vec<f64> = (0..=k).map(|j| lag(row, j)).collect::<Result<_, _>>()?;
                for _ in 0..k {
                    cur = cur.windows(2).map(|w| if is_missing(w[0]) || is_missing(w[1]) { SYSMISS } else { w[0] - w[1] }).collect();
                }
                cur[0]
            }
            _ => {
                let a = lag(row, 0)?;
                let b = lag(row, k)?;
                if is_missing(a) || is_missing(b) { SYSMISS } else { a - b }
            }
        };
        Ok(Val::Num(v))
    }

    /// Value of `var` k periods before observation `row` using the tsset
    /// time (and panel) variables; falls back to the previous observation.
    fn ts_value(&self, var: &str, row: usize, k: i64) -> Result<f64, String> {
        let vi = self.var_index(var).ok_or_else(|| rc_err(111, format!("{var} not found")))?;
        if k == 0 {
            return self.cell(vi, row).num();
        }
        if let Some((panel, time)) = self.st.and_then(|s| s.tsset.clone()) {
            let ti = self.var_index(&time).ok_or_else(|| rc_err(111, format!("{time} not found")))?;
            let t0 = self.ds.vars[ti].num(row);
            if is_missing(t0) {
                return Ok(SYSMISS);
            }
            let target = t0 - k as f64;
            let pi = panel.as_ref().and_then(|p| self.var_index(p));
            let p0 = pi.map(|p| self.ds.vars[p].num(row));
            // search near the current observation first (data usually sorted)
            let n = self.ds.n_obs;
            let check = |j: usize| -> bool {
                self.ds.vars[ti].num(j) == target && pi.map_or(true, |p| Some(self.ds.vars[p].num(j)) == p0)
            };
            let guess = row as i64 - k;
            if guess >= 0 && (guess as usize) < n && check(guess as usize) {
                return self.cell(vi, guess as usize).num();
            }
            let mut idx = self.ts_index.borrow_mut();
            let map = idx.get_or_insert_with(|| {
                let mut m = HashMap::with_capacity(n);
                for j in 0..n {
                    let tj = self.ds.vars[ti].num(j);
                    let pj = pi.map_or(0.0, |p| self.ds.vars[p].num(j));
                    m.entry((pj.to_bits(), tj.to_bits())).or_insert(j);
                }
                m
            });
            let key = (p0.unwrap_or(0.0).to_bits(), target.to_bits());
            return match map.get(&key) {
                Some(&j) => self.cell(vi, j).num(),
                None => Ok(SYSMISS),
            };
        }
        let (s, e) = self.group_bounds(row);
        let j = row as i64 - k;
        if j < s as i64 || j >= e as i64 {
            return Ok(SYSMISS);
        }
        self.cell(vi, j as usize).num()
    }

    fn eval_binary(&self, l: &Expr, op: &BinOp, r: &Expr, row: usize) -> Result<Val, String> {
        // no short-circuit evaluation: evaluate both sides
        let a = self.eval(l, row)?;
        let b = self.eval(r, row)?;
        use BinOp::*;
        match (&a, &b) {
            (Val::Str(x), Val::Str(y)) => match op {
                Add => Ok(Val::Str(format!("{x}{y}"))),
                Eq => Ok(bool_val(x == y)),
                Ne => Ok(bool_val(x != y)),
                Lt => Ok(bool_val(x < y)),
                Le => Ok(bool_val(x <= y)),
                Gt => Ok(bool_val(x > y)),
                Ge => Ok(bool_val(x >= y)),
                _ => Err(type_mismatch()),
            },
            (Val::Str(s), Val::Num(n)) | (Val::Num(n), Val::Str(s)) if *op == Mul => {
                if is_missing(*n) || *n < 1.0 {
                    Ok(Val::Str(String::new()))
                } else {
                    Ok(Val::Str(s.repeat(*n as usize)))
                }
            }
            (Val::Num(x), Val::Num(y)) => {
                let (x, y) = (*x, *y);
                let arith = |v: f64| Val::Num(if v.is_finite() { v } else { SYSMISS });
                let anymiss = is_missing(x) || is_missing(y);
                match op {
                    Add => Ok(if anymiss { Val::Num(SYSMISS) } else { arith(x + y) }),
                    Sub => Ok(if anymiss { Val::Num(SYSMISS) } else { arith(x - y) }),
                    Mul => Ok(if anymiss { Val::Num(SYSMISS) } else { arith(x * y) }),
                    Div => Ok(if anymiss || y == 0.0 { Val::Num(SYSMISS) } else { arith(x / y) }),
                    Pow => Ok(if anymiss { Val::Num(SYSMISS) } else { arith(pow(x, y)) }),
                    Eq => Ok(bool_val(x == y)),
                    Ne => Ok(bool_val(x != y)),
                    Lt => Ok(bool_val(x < y)),
                    Le => Ok(bool_val(x <= y)),
                    Gt => Ok(bool_val(x > y)),
                    Ge => Ok(bool_val(x >= y)),
                    And => Ok(bool_val(x != 0.0 && y != 0.0)),
                    Or => Ok(bool_val(x != 0.0 || y != 0.0)),
                }
            }
            _ => Err(type_mismatch()),
        }
    }

    fn eval_index(&self, base: &Expr, idx: &Expr, row: usize) -> Result<Val, String> {
        let name = match base {
            Expr::Var(n) => n.as_str(),
            _ => return Err(rc_err(198, "invalid subscript")),
        };
        let st = self.st;
        if name == "_b" || name == "_coef" || name == "_se" {
            let coef = coef_name(idx);
            let st = st.ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let j = find_coef(b, &coef).ok_or_else(|| rc_err(111, format!("[{coef}] not found")))?;
            if name == "_se" {
                let v = st.e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
                let var = v.get(j, j);
                return Ok(Val::Num(if var > 0.0 { var.sqrt() } else { 0.0 }));
            }
            return Ok(Val::Num(b.get(0, j)));
        }
        if let Some(vi) = self.var_index(name) {
            let k = self.eval(idx, row)?.num()?;
            let (s, e) = self.group_bounds(row);
            if is_missing(k) || k < 1.0 {
                return Ok(self.missing_like(vi));
            }
            let j = s + k as usize - 1;
            if j >= e {
                return Ok(self.missing_like(vi));
            }
            return Ok(self.cell(vi, j));
        }
        if let Some(m) = st.and_then(|s| s.matrices.get(name)) {
            let (i, j) = match idx {
                Expr::String(s) => {
                    let (a, b) = s.split_once(',').unwrap_or((s, "1"));
                    let ev = |t: &str| -> Result<f64, String> {
                        let e = ferrostat_parser::parser::parse_expression(t).map_err(|e| e.to_string())?;
                        self.eval(&e, row)?.num()
                    };
                    (ev(a)?, ev(b)?)
                }
                other => (self.eval(other, row)?.num()?, 1.0),
            };
            return Ok(Val::Num(m.el(i as usize - 1, j as usize - 1)));
        }
        Err(rc_err(111, format!("{name} not found")))
    }

    fn missing_like(&self, vi: usize) -> Val {
        if self.ds.vars[vi].var_type.is_string() { Val::Str(String::new()) } else { Val::Num(SYSMISS) }
    }

    fn rng(&self) -> Result<std::cell::RefMut<'_, crate::rng::Rng>, String> {
        self.rng.map(|r| r.borrow_mut()).ok_or_else(|| rc_err(198, "random numbers not available here"))
    }

    fn eval_call(&self, whole: &Expr, name: &str, args: &[Expr], row: usize) -> Result<Val, String> {
        let lname = name.to_lowercase();
        // functions taking names rather than values
        match lname.as_str() {
            "r" | "e" | "s" | "c" => return self.stored(&lname, args, row),
            "scalar" => {
                let n = arg_name(args.first());
                let st = self.st.ok_or_else(|| rc_err(111, "scalar not found"))?;
                if let Some(v) = st.scalars.get(&n) {
                    return Ok(Val::Num(*v));
                }
                if let Some(v) = st.str_scalars.get(&n) {
                    return Ok(Val::Str(v.clone()));
                }
                return Err(rc_err(111, format!("scalar {n} not found")));
            }
            "rowsof" | "colsof" | "det" | "trace" | "issymmetric" | "matmissing" | "el" | "colnumb" | "rownumb" => {
                return self.matrix_fn(&lname, args, row);
            }
            _ => {}
        }
        let vals: Vec<Val> = args.iter().map(|a| self.eval(a, row)).collect::<Result<_, _>>()?;
        if lname == "sum" {
            let x = vals.first().map(|v| v.num()).transpose()?.unwrap_or(0.0);
            let x = if is_missing(x) { 0.0 } else { x };
            let key = whole as *const Expr as usize;
            let (s, _) = self.group_bounds(row);
            let mut run = self.running.borrow_mut();
            let acc = match run.get(&key) {
                Some(&(last, acc)) if row > s && last + 1 == row => acc + x,
                _ if row == s => x,
                _ => {
                    // recompute from the group start
                    drop(run);
                    let mut acc = 0.0;
                    for j in s..=row {
                        let v = self.eval(&args[0], j)?.num()?;
                        if !is_missing(v) {
                            acc += v;
                        }
                    }
                    run = self.running.borrow_mut();
                    acc
                }
            };
            run.insert(key, (row, acc));
            return Ok(Val::Num(acc));
        }
        call_function(self, &lname, &vals)
    }

    fn stored(&self, class: &str, args: &[Expr], row: usize) -> Result<Val, String> {
        let key = arg_name(args.first());
        let st = match self.st {
            Some(s) => s,
            None => return Ok(Val::Num(SYSMISS)),
        };
        if class == "c" {
            return Ok(c_value(self, &key));
        }
        let res = match class {
            "r" => &st.r,
            "e" => &st.e,
            _ => &st.s,
        };
        if key == "sample" && class == "e" {
            return Ok(Val::Num(match &res.sample {
                Some(s) => {
                    if s.get(row).copied().unwrap_or(false) { 1.0 } else { 0.0 }
                }
                None => 0.0,
            }));
        }
        if let Some(v) = res.scalars.get(&key) {
            return Ok(Val::Num(*v));
        }
        if let Some(v) = res.macros.get(&key) {
            return Ok(Val::Str(v.clone()));
        }
        if let Some(m) = res.matrices.get(&key) {
            return Ok(Val::Num(m.el(0, 0)));
        }
        Ok(Val::Num(SYSMISS))
    }

    fn matrix_fn(&self, name: &str, args: &[Expr], row: usize) -> Result<Val, String> {
        let mname = arg_name(args.first());
        let st = self.st.ok_or_else(|| rc_err(111, "matrix not found"))?;
        let m = st
            .matrices
            .get(&mname)
            .or_else(|| {
                // r(name) / e(name)
                if let Some(Expr::Call(c, a)) = args.first() {
                    let k = arg_name(a.first());
                    match c.as_str() {
                        "r" => st.r.matrices.get(&k),
                        "e" => st.e.matrices.get(&k),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .ok_or_else(|| rc_err(111, format!("matrix {mname} not found")))?;
        let num = |i: usize| -> Result<f64, String> { self.eval(&args[i], row)?.num() };
        Ok(match name {
            "rowsof" => Val::Num(m.rows as f64),
            "colsof" => Val::Num(m.cols as f64),
            "det" => Val::Num(m.det()),
            "trace" => Val::Num(m.trace()),
            "issymmetric" => Val::Num(bool_f(m.rows == m.cols && (0..m.rows).all(|i| (0..m.cols).all(|j| m.get(i, j) == m.get(j, i))))),
            "matmissing" => Val::Num(bool_f(m.has_missing())),
            "el" => Val::Num(m.el(num(1)? as usize - 1, num(2)? as usize - 1)),
            "colnumb" | "rownumb" => {
                let s = self.eval(&args[1], row)?;
                let s = s.str()?;
                let idx = if name == "colnumb" { m.col_index(s) } else { m.row_index(s) };
                Val::Num(idx.map_or(SYSMISS, |i| (i + 1) as f64))
            }
            _ => unreachable!(),
        })
    }
}

fn bool_f(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

fn bool_val(b: bool) -> Val {
    Val::Num(bool_f(b))
}

fn pow(x: f64, y: f64) -> f64 {
    if x < 0.0 && y.fract() != 0.0 {
        return SYSMISS;
    }
    if x == 0.0 && y < 0.0 {
        return SYSMISS;
    }
    x.powf(y)
}

/// Name argument of r()/e()/scalar(): `r(mean)` parses `mean` as a variable.
fn arg_name(e: Option<&Expr>) -> String {
    match e {
        Some(Expr::Var(n)) => n.clone(),
        Some(Expr::String(s)) => s.clone(),
        Some(Expr::Number(n)) => format!("{n}"),
        Some(Expr::Call(n, _)) => n.clone(),
        _ => String::new(),
    }
}

fn coef_name(e: &Expr) -> String {
    match e {
        Expr::Var(n) | Expr::String(n) => n.clone(),
        Expr::Number(n) => format!("{n}"),
        other => format!("{other:?}"),
    }
}

/// Find a coefficient column by name; accepts `eq:name` and bare names.
pub fn find_coef(b: &crate::matrix::Matrix, name: &str) -> Option<usize> {
    if let Some(j) = b.col_index(name) {
        return Some(j);
    }
    let bare = name.rsplit(':').next().unwrap_or(name);
    if let Some(j) = b.colnames.iter().position(|c| c.rsplit(':').next() == Some(bare)) {
        return Some(j);
    }
    // 1.group vs 1b.group / 1o.group
    b.colnames.iter().position(|c| {
        let c = c.rsplit(':').next().unwrap_or(c);
        if let (Some((l1, v1)), Some((l2, v2))) = (c.split_once('.'), bare.split_once('.')) {
            v1 == v2 && l1.trim_end_matches(['b', 'o', 'n']) == l2
        } else {
            false
        }
    })
}

fn c_value(env: &Env, key: &str) -> Val {
    let now = chrono::Local::now();
    match key {
        "N" => Val::Num(env.ds.n_obs as f64),
        "k" => Val::Num(env.ds.n_vars() as f64),
        "pi" => Val::Num(std::f64::consts::PI),
        "e" => Val::Num(std::f64::consts::E),
        "level" => Val::Num(env.st.map_or(95.0, |s| s.level)),
        "rc" => Val::Num(env.st.map_or(0, |s| s.rc) as f64),
        "maxdouble" => Val::Num(8.988465674311579e307),
        "mindouble" => Val::Num(-8.988465674311579e307),
        "epsdouble" => Val::Num(f64::EPSILON),
        "smallestdouble" => Val::Num(f64::MIN_POSITIVE),
        "maxbyte" => Val::Num(100.0),
        "minbyte" => Val::Num(-127.0),
        "maxint" => Val::Num(32740.0),
        "minint" => Val::Num(-32767.0),
        "maxlong" => Val::Num(2147483620.0),
        "minlong" => Val::Num(-2147483647.0),
        "maxfloat" => Val::Num(1.70141173319e38),
        "maxstrvarlen" => Val::Num(2045.0),
        "maxvar" => Val::Num(120000.0),
        // the first name is kept for existing do-files that query it
        "stata_version" | "version" => Val::Num(18.0),
        "current_date" => Val::Str(now.format("%e %b %Y").to_string().trim().to_string()),
        "current_time" => Val::Str(now.format("%H:%M:%S").to_string()),
        "os" => Val::Str(if cfg!(windows) { "Windows" } else if cfg!(target_os = "macos") { "MacOSX" } else { "Unix" }.into()),
        "dirsep" => Val::Str(std::path::MAIN_SEPARATOR.to_string()),
        "pwd" => Val::Str(ferrostat_core::vfs::current_dir()),
        "filename" => Val::Str(env.st.map(|s| s.filename.clone()).unwrap_or_default()),
        "changed" => Val::Num(bool_f(env.st.map_or(false, |s| s.data_changed))),
        "seed" => Val::Num(env.st.map_or(0, |s| s.rng.borrow().seed) as f64),
        "rngstate" => Val::Str(env.st.map(|s| s.rng.borrow().state()).unwrap_or_default()),
        "alpha" => Val::Str("a b c d e f g h i j k l m n o p q r s t u v w x y z".into()),
        "ALPHA" => Val::Str("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".into()),
        "Mons" => Val::Str("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".into()),
        "Months" => Val::Str("January February March April May June July August September October November December".into()),
        "Wdays" => Val::Str("Sun Mon Tue Wed Thu Fri Sat".into()),
        _ => Val::Num(SYSMISS),
    }
}

// ---------------------------------------------------------------------------
// Function library
// ---------------------------------------------------------------------------

fn n(v: &[Val], i: usize) -> Result<f64, String> {
    v.get(i).ok_or_else(|| rc_err(198, "too few arguments"))?.num()
}

fn s(v: &[Val], i: usize) -> Result<&str, String> {
    v.get(i).ok_or_else(|| rc_err(198, "too few arguments"))?.str()
}

fn num1(v: &[Val], f: impl Fn(f64) -> f64) -> Result<Val, String> {
    let x = n(v, 0)?;
    Ok(Val::Num(if is_missing(x) { SYSMISS } else { finite(f(x)) }))
}

fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { SYSMISS }
}

fn nums_ok(xs: &[f64]) -> bool {
    xs.iter().all(|x| !is_missing(*x))
}

fn num2(v: &[Val], f: impl Fn(f64, f64) -> f64) -> Result<Val, String> {
    let (a, b) = (n(v, 0)?, n(v, 1)?);
    Ok(Val::Num(if nums_ok(&[a, b]) { finite(f(a, b)) } else { SYSMISS }))
}

fn num3(v: &[Val], f: impl Fn(f64, f64, f64) -> f64) -> Result<Val, String> {
    let (a, b, c) = (n(v, 0)?, n(v, 1)?, n(v, 2)?);
    Ok(Val::Num(if nums_ok(&[a, b, c]) { finite(f(a, b, c)) } else { SYSMISS }))
}

fn round_to(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return x;
    }
    let q = x / y;
    // round half away from zero, guarding against representation error
    let r = (q + 1e-12 * q.signum()).round();
    let v = r * y;
    // clean up e.g. 3.1400000000000001
    let digits = (-y.abs().log10()).ceil().max(0.0) as i32;
    if digits < 15 {
        let p = 10f64.powi(digits);
        (v * p).round() / p
    } else {
        v
    }
}

fn substr(st: &str, start: f64, len: f64) -> String {
    let chars: Vec<char> = st.chars().collect();
    let n = chars.len() as i64;
    if is_missing(start) || start == 0.0 {
        return String::new();
    }
    let mut b = start as i64;
    if b < 0 {
        b += n + 1;
    }
    if b < 1 || b > n {
        return String::new();
    }
    let l = if is_missing(len) { n } else { len as i64 };
    if l <= 0 {
        return String::new();
    }
    let e = (b - 1 + l).min(n);
    chars[(b - 1) as usize..e as usize].iter().collect()
}

fn wildcard_match(pat: &str, text: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) && p[pi] != '*' {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Convert do-file regular expressions to Rust regex syntax (mostly compatible).
fn regex(pat: &str) -> Result<regex::Regex, String> {
    thread_local! {
        static CACHE: RefCell<HashMap<String, regex::Regex>> = RefCell::new(HashMap::new());
    }
    if let Some(r) = CACHE.with(|c| c.borrow().get(pat).cloned()) {
        return Ok(r);
    }
    let r = regex::Regex::new(pat).map_err(|e| rc_err(198, format!("invalid regular expression: {e}")))?;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 256 {
            c.clear();
        }
        c.insert(pat.to_string(), r.clone());
    });
    Ok(r)
}

fn real(s: &str) -> f64 {
    let t = s.trim();
    if t == "." {
        return SYSMISS;
    }
    if let Some(c) = t.strip_prefix('.') {
        if c.len() == 1 && c.chars().all(|c| c.is_ascii_lowercase()) {
            return missing(c.as_bytes()[0] - b'a' + 1);
        }
    }
    t.parse::<f64>().ok().filter(|v| v.is_finite()).unwrap_or(SYSMISS)
}

pub fn string_of(x: f64, fmt: Option<&str>) -> String {
    match fmt {
        Some(f) => crate::format::fmt(x, f).trim().to_string(),
        None => crate::format::fmt_g(x, 8),
    }
}

fn proper(s: &str) -> String {
    let mut out = String::new();
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha { out.extend(c.to_lowercase()) } else { out.extend(c.to_uppercase()) }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

fn strtoname(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if out.chars().next().map_or(true, |c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out.chars().take(32).collect()
}

fn soundex(s: &str) -> String {
    let code = |c: char| match c.to_ascii_lowercase() {
        'b' | 'f' | 'p' | 'v' => '1',
        'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => '2',
        'd' | 't' => '3',
        'l' => '4',
        'm' | 'n' => '5',
        'r' => '6',
        _ => '0',
    };
    let letters: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return String::new();
    }
    let mut out = letters[0].to_ascii_uppercase().to_string();
    let mut last = code(letters[0]);
    for &c in &letters[1..] {
        let d = code(c);
        if d != '0' && d != last {
            out.push(d);
        }
        if !matches!(c.to_ascii_lowercase(), 'h' | 'w') {
            last = d;
        }
        if out.len() == 4 {
            break;
        }
    }
    while out.len() < 4 {
        out.push('0');
    }
    out
}

fn call_function(env: &Env, name: &str, v: &[Val]) -> Result<Val, String> {
    use std::f64::consts;
    let st = |x: String| Ok(Val::Str(x));
    let num = |x: f64| Ok(Val::Num(x));
    match name {
        // ---- math
        "abs" => num1(v, f64::abs),
        "ceil" | "ceiling" => num1(v, f64::ceil),
        "floor" => num1(v, f64::floor),
        "int" | "trunc" => num1(v, f64::trunc),
        "round" => {
            if v.len() >= 2 {
                num2(v, round_to)
            } else {
                num1(v, |x| round_to(x, 1.0))
            }
        }
        "exp" => num1(v, f64::exp),
        "ln" | "log" => num1(v, |x| if x > 0.0 { x.ln() } else { SYSMISS }),
        "log10" => num1(v, |x| if x > 0.0 { x.log10() } else { SYSMISS }),
        "log2" => num1(v, |x| if x > 0.0 { x.log2() } else { SYSMISS }),
        "ln1p" | "log1p" => num1(v, |x| if x > -1.0 { x.ln_1p() } else { SYSMISS }),
        "expm1" => num1(v, f64::exp_m1),
        "sqrt" => num1(v, |x| if x >= 0.0 { x.sqrt() } else { SYSMISS }),
        "sin" => num1(v, f64::sin),
        "cos" => num1(v, f64::cos),
        "tan" => num1(v, f64::tan),
        "asin" => num1(v, |x| if x.abs() <= 1.0 { x.asin() } else { SYSMISS }),
        "acos" => num1(v, |x| if x.abs() <= 1.0 { x.acos() } else { SYSMISS }),
        "atan" => num1(v, f64::atan),
        "atan2" => num2(v, f64::atan2),
        "sinh" => num1(v, f64::sinh),
        "cosh" => num1(v, f64::cosh),
        "tanh" => num1(v, f64::tanh),
        "asinh" => num1(v, f64::asinh),
        "acosh" => num1(v, |x| if x >= 1.0 { x.acosh() } else { SYSMISS }),
        "atanh" => num1(v, |x| if x.abs() < 1.0 { x.atanh() } else { SYSMISS }),
        "sign" => num1(v, |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        "mod" => num2(v, |x, y| if y == 0.0 { SYSMISS } else { x - y * (x / y).floor() }),
        "float" => num1(v, |x| x as f32 as f64),
        "reldif" => num2(v, |x, y| (x - y).abs() / (y.abs() + 1.0)),
        "logit" => num1(v, dist::logit),
        "invlogit" => num1(v, dist::invlogit),
        "cloglog" => num1(v, |p| if p > 0.0 && p < 1.0 { (-(1.0 - p).ln()).ln() } else { SYSMISS }),
        "invcloglog" => num1(v, |x| 1.0 - (-x.exp()).exp()),
        "lngamma" => num1(v, dist::lngamma),
        "gamma" | "tgamma" => num1(v, |x| dist::lngamma(x).exp()),
        "digamma" => num1(v, dist::digamma),
        "trigamma" => num1(v, dist::trigamma),
        "lnfactorial" => num1(v, dist::lnfactorial),
        "factorial" => num1(v, |x| dist::lnfactorial(x).exp().round()),
        "comb" => num2(v, dist::comb),
        "clip" => num3(v, |x, a, b| x.max(a).min(b)),
        "min" | "max" => {
            let xs: Vec<f64> = v.iter().map(|x| x.num()).collect::<Result<_, _>>()?;
            let xs: Vec<f64> = xs.into_iter().filter(|x| !is_missing(*x)).collect();
            if xs.is_empty() {
                return num(SYSMISS);
            }
            num(if name == "min" { xs.iter().cloned().fold(f64::INFINITY, f64::min) } else { xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) })
        }
        // ---- distributions
        "normal" | "norm" => num1(v, dist::normal),
        "normalden" | "normd" => match v.len() {
            1 => num1(v, dist::normalden),
            2 => num2(v, |x, s| if s > 0.0 { dist::normalden(x / s) / s } else { SYSMISS }),
            _ => num3(v, |x, m, s| if s > 0.0 { dist::normalden((x - m) / s) / s } else { SYSMISS }),
        },
        "lnnormal" => num1(v, dist::lnnormal),
        "lnnormalden" => match v.len() {
            1 => num1(v, dist::lnnormalden),
            2 => num2(v, |x, s| dist::lnnormalden(x / s) - s.ln()),
            _ => num3(v, |x, m, s| dist::lnnormalden((x - m) / s) - s.ln()),
        },
        "invnormal" | "invnorm" => num1(v, dist::invnormal),
        "binormal" => num3(v, dist::binormal),
        "t" => num2(v, dist::t),
        "ttail" => num2(v, dist::ttail),
        "tden" => num2(v, dist::tden),
        "invt" => num2(v, dist::invt),
        "invttail" => num2(v, dist::invttail),
        "chi2" => num2(v, dist::chi2),
        "chi2tail" => num2(v, dist::chi2tail),
        "chi2den" => num2(v, dist::chi2den),
        "invchi2" => num2(v, dist::invchi2),
        "invchi2tail" => num2(v, dist::invchi2tail),
        "f" => num3(v, dist::f_cdf),
        "ftail" => num3(v, dist::ftail),
        "fden" => num3(v, dist::fden),
        "invf" => num3(v, dist::invf),
        "invftail" => num3(v, dist::invftail),
        "gammap" => num2(v, dist::gammap),
        "gammaptail" => num2(v, dist::gammaptail),
        "invgammap" => num2(v, dist::invgammap),
        "invgammaptail" => num2(v, dist::invgammaptail),
        "gammaden" => {
            let xs: Vec<f64> = (0..4).map(|i| n(v, i)).collect::<Result<_, _>>()?;
            num(if nums_ok(&xs) { dist::gammaden(xs[0], xs[1], xs[2], xs[3]) } else { SYSMISS })
        }
        "ibeta" => num3(v, dist::ibeta),
        "ibetatail" => num3(v, dist::ibetatail),
        "invibeta" => num3(v, dist::invibeta),
        "betaden" => num3(v, dist::betaden),
        "binomial" => num3(v, dist::binomial),
        "binomialp" => num3(v, dist::binomialp),
        "binomialtail" => num3(v, dist::binomialtail),
        "poisson" => num2(v, dist::poisson),
        "poissonp" => num2(v, dist::poissonp),
        "poissontail" => num2(v, dist::poissontail),
        // ---- random numbers
        "runiform" | "uniform" => {
            let mut r = env.rng()?;
            if v.len() >= 2 {
                let (a, b) = (n(v, 0)?, n(v, 1)?);
                num(r.uniform_ab(a, b))
            } else {
                num(r.uniform())
            }
        }
        "runiformint" => {
            let (a, b) = (n(v, 0)?, n(v, 1)?);
            num(env.rng()?.uniform_int(a, b))
        }
        "rnormal" | "invnorm_uniform" => {
            let (m, sd) = (if v.is_empty() { 0.0 } else { n(v, 0)? }, if v.len() < 2 { 1.0 } else { n(v, 1)? });
            let z = env.rng()?.normal();
            num(m + sd * z)
        }
        "rbeta" => {
            let (a, b) = (n(v, 0)?, n(v, 1)?);
            num(env.rng()?.beta(a, b))
        }
        "rbinomial" => {
            let (a, b) = (n(v, 0)?, n(v, 1)?);
            num(env.rng()?.binomial(a, b))
        }
        "rchi2" => {
            let a = n(v, 0)?;
            num(env.rng()?.chi2(a))
        }
        "rexponential" => {
            let a = n(v, 0)?;
            num(env.rng()?.exponential(a))
        }
        "rgamma" => {
            let (a, b) = (n(v, 0)?, n(v, 1)?);
            num(env.rng()?.gamma(a) * b)
        }
        "rpoisson" => {
            let a = n(v, 0)?;
            num(env.rng()?.poisson(a))
        }
        "rt" => {
            let a = n(v, 0)?;
            num(env.rng()?.t(a))
        }
        "rlogistic" => {
            let (m, sc) = (if v.is_empty() { 0.0 } else { n(v, 0)? }, if v.len() < 2 { 1.0 } else { n(v, 1)? });
            let u = env.rng()?.uniform();
            num(m + sc * (u / (1.0 - u)).ln())
        }
        // ---- programming
        "missing" | "mi" => num(bool_f(v.iter().any(|x| x.is_missing()))),
        "inlist" => {
            let first = v.first().ok_or_else(|| rc_err(198, "too few arguments"))?;
            for x in &v[1..] {
                match (first, x) {
                    (Val::Num(a), Val::Num(b)) if a == b => return num(1.0),
                    (Val::Str(a), Val::Str(b)) if a == b => return num(1.0),
                    (Val::Num(_), Val::Num(_)) | (Val::Str(_), Val::Str(_)) => {}
                    _ => return Err(type_mismatch()),
                }
            }
            num(0.0)
        }
        "inrange" => match (&v[0], &v[1], &v[2]) {
            (Val::Num(x), Val::Num(a), Val::Num(b)) => {
                // missing bounds are treated as -inf / +inf
                let lo = if is_missing(*a) { f64::NEG_INFINITY } else { *a };
                let hi = if is_missing(*b) { f64::INFINITY } else { *b };
                num(bool_f(*x >= lo && *x <= hi && (!is_missing(*x) || is_missing(*b))))
            }
            (Val::Str(x), Val::Str(a), Val::Str(b)) => num(bool_f(x >= a && x <= b)),
            _ => Err(type_mismatch()),
        },
        "cond" => {
            let c = n(v, 0)?;
            if v.len() >= 4 && is_missing(c) {
                return Ok(v[3].clone());
            }
            Ok(if c != 0.0 { v[1].clone() } else { v[2].clone() })
        }
        "autocode" => {
            let (x, k, lo, hi) = (n(v, 0)?, n(v, 1)?, n(v, 2)?, n(v, 3)?);
            if is_missing(x) {
                return num(SYSMISS);
            }
            let w = (hi - lo) / k;
            let mut b = lo + w;
            while x > b && b < hi {
                b += w;
            }
            num(b.min(hi))
        }
        "recode" => {
            let x = n(v, 0)?;
            if is_missing(x) {
                return num(SYSMISS);
            }
            for c in &v[1..] {
                let c = c.num()?;
                if x <= c {
                    return num(c);
                }
            }
            num(v.last().unwrap().num()?)
        }
        "irecode" => {
            let x = n(v, 0)?;
            if is_missing(x) {
                return num(SYSMISS);
            }
            let mut k = 0.0;
            for c in &v[1..] {
                if x <= c.num()? {
                    return num(k);
                }
                k += 1.0;
            }
            num(k)
        }
        "fileexists" => num(bool_f(ferrostat_core::vfs::exists(s(v, 0)?))),
        // ---- strings
        "strlen" | "length" | "ustrlen" | "udstrlen" => {
            let x = s(v, 0)?;
            num(if name == "strlen" || name == "length" { x.len() } else { x.chars().count() } as f64)
        }
        "substr" | "usubstr" | "substrb" => {
            let x = s(v, 0)?;
            st(substr(x, n(v, 1)?, n(v, 2)?))
        }
        "strupper" | "upper" | "ustrupper" => st(s(v, 0)?.to_uppercase()),
        "strlower" | "lower" | "ustrlower" => st(s(v, 0)?.to_lowercase()),
        "strproper" | "proper" | "ustrtitle" => st(proper(s(v, 0)?)),
        "strtrim" | "trim" | "ustrtrim" => st(s(v, 0)?.trim().to_string()),
        "strltrim" | "ltrim" | "ustrltrim" => st(s(v, 0)?.trim_start().to_string()),
        "strrtrim" | "rtrim" | "ustrrtrim" => st(s(v, 0)?.trim_end().to_string()),
        "stritrim" | "itrim" => st(s(v, 0)?.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")),
        "strreverse" | "reverse" | "ustrreverse" => st(s(v, 0)?.chars().rev().collect()),
        "strpos" | "index" | "ustrpos" => {
            let (h, nd) = (s(v, 0)?, s(v, 1)?);
            num(h.find(nd).map_or(0.0, |i| (h[..i].chars().count() + 1) as f64))
        }
        "strrpos" | "ustrrpos" => {
            let (h, nd) = (s(v, 0)?, s(v, 1)?);
            num(h.rfind(nd).map_or(0.0, |i| (h[..i].chars().count() + 1) as f64))
        }
        "subinstr" | "usubinstr" => {
            let (x, from, to) = (s(v, 0)?, s(v, 1)?, s(v, 2)?);
            let k = n(v, 3)?;
            if from.is_empty() {
                return st(x.to_string());
            }
            st(if is_missing(k) { x.replace(from, to) } else { x.replacen(from, to, k.max(0.0) as usize) })
        }
        "subinword" => {
            let (x, from, to) = (s(v, 0)?, s(v, 1)?, s(v, 2)?);
            let k = n(v, 3)?;
            let mut count = 0usize;
            let words: Vec<String> = x
                .split(' ')
                .map(|w| {
                    if w == from && (is_missing(k) || (count as f64) < k) {
                        count += 1;
                        to.to_string()
                    } else {
                        w.to_string()
                    }
                })
                .collect();
            st(words.join(" "))
        }
        "word" | "ustrword" => {
            let x = s(v, 0)?;
            let k = n(v, 1)?;
            let w: Vec<&str> = x.split_whitespace().collect();
            let idx = if k < 0.0 { w.len() as i64 + k as i64 } else { k as i64 - 1 };
            st(if idx >= 0 && (idx as usize) < w.len() { w[idx as usize].to_string() } else { String::new() })
        }
        "wordcount" | "ustrwordcount" => num(s(v, 0)?.split_whitespace().count() as f64),
        "real" => num(real(s(v, 0)?)),
        "string" | "strofreal" => {
            let x = n(v, 0)?;
            let f = if v.len() > 1 { Some(s(v, 1)?.to_string()) } else { None };
            st(string_of(x, f.as_deref()))
        }
        "char" | "uchar" => {
            let x = n(v, 0)?;
            st(char::from_u32(x as u32).map(|c| c.to_string()).unwrap_or_default())
        }
        "ascii" => num(s(v, 0)?.chars().next().map_or(SYSMISS, |c| c as u32 as f64)),
        "abbrev" => {
            let x = s(v, 0)?;
            let k = n(v, 1)? as usize;
            st(if x.chars().count() <= k { x.to_string() } else { format!("{}~", x.chars().take(k.saturating_sub(1)).collect::<String>()) })
        }
        "strtoname" | "ustrtoname" => st(strtoname(s(v, 0)?)),
        "strmatch" => num(bool_f(wildcard_match(s(v, 1)?, s(v, 0)?))),
        "strdup" => {
            let (x, k) = (s(v, 0)?, n(v, 1)?);
            st(if is_missing(k) || k < 1.0 { String::new() } else { x.repeat(k as usize) })
        }
        "soundex" => st(soundex(s(v, 0)?)),
        "indexnot" => {
            let (x, set) = (s(v, 0)?, s(v, 1)?);
            num(x.chars().position(|c| !set.contains(c)).map_or(0.0, |i| (i + 1) as f64))
        }
        "regexm" | "ustrregexm" => {
            let (x, p) = (s(v, 0)?, s(v, 1)?);
            let re = regex(p)?;
            let mut caps = env.regex_caps.borrow_mut();
            caps.clear();
            if let Some(c) = re.captures(x) {
                for g in c.iter() {
                    caps.push(g.map(|m| m.as_str().to_string()).unwrap_or_default());
                }
                num(1.0)
            } else {
                num(0.0)
            }
        }
        "regexs" | "ustrregexs" => {
            let k = n(v, 0)? as usize;
            st(env.regex_caps.borrow().get(k).cloned().unwrap_or_default())
        }
        "regexr" | "ustrregexrf" => {
            let (x, p, to) = (s(v, 0)?, s(v, 1)?, s(v, 2)?);
            st(regex(p)?.replace(x, to).into_owned())
        }
        "ustrregexra" | "regexra" => {
            let (x, p, to) = (s(v, 0)?, s(v, 1)?, s(v, 2)?);
            st(regex(p)?.replace_all(x, to).into_owned())
        }
        "plural" => {
            let k = n(v, 0)?;
            let w = s(v, 1)?;
            st(if k == 1.0 { w.to_string() } else { format!("{w}s") })
        }
        "uisdigit" => num(bool_f(s(v, 0)?.chars().all(|c| c.is_ascii_digit()))),
        "uisletter" => num(bool_f(s(v, 0)?.chars().all(|c| c.is_alphabetic()))),
        "hash1" => num(s(v, 0)?.bytes().fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) as f64),
        // ---- dates
        "mdy" => num3(v, dates::mdy),
        "mdyhms" => {
            let xs: Vec<f64> = (0..6).map(|i| n(v, i)).collect::<Result<_, _>>()?;
            num(if nums_ok(&xs) { dates::mdyhms(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5]) } else { SYSMISS })
        }
        "dhms" => {
            let xs: Vec<f64> = (0..4).map(|i| n(v, i)).collect::<Result<_, _>>()?;
            num(if nums_ok(&xs) { xs[0] * dates::MS_PER_DAY + ((xs[1] * 60.0 + xs[2]) * 60.0 + xs[3]) * 1000.0 } else { SYSMISS })
        }
        "hms" => num3(v, |h, m, sec| ((h * 60.0 + m) * 60.0 + sec) * 1000.0),
        "date" | "daily" => {
            let top = if v.len() > 2 { Some(n(v, 2)?) } else { None };
            num(dates::date(s(v, 0)?, s(v, 1)?, top))
        }
        "clock" | "clock2" | "Clock" => num(dates::clock(s(v, 0)?, s(v, 1)?)),
        "monthly" => {
            let x = s(v, 0)?;
            let td = dates::date(&format!("1 {x}"), &format!("D{}", s(v, 1)?), None);
            num(dates::mofd(td))
        }
        "quarterly" => {
            let x = s(v, 0)?.to_lowercase();
            let parts: Vec<&str> = x.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).collect();
            if parts.len() == 2 {
                let (a, b): (f64, f64) = (parts[0].parse().unwrap(), parts[1].parse().unwrap());
                let (y, q) = if s(v, 1)?.starts_with('Y') { (a, b) } else { (b, a) };
                num(dates::yq(y, q))
            } else {
                num(SYSMISS)
            }
        }
        "yearly" => num(real(s(v, 0)?)),
        "year" => num1(v, dates::year),
        "month" => num1(v, dates::month),
        "day" => num1(v, dates::day),
        "dow" => num1(v, dates::dow),
        "doy" => num1(v, dates::doy),
        "week" => num1(v, dates::week),
        "quarter" => num1(v, dates::quarter),
        "halfyear" => num1(v, dates::halfyear),
        "mofd" => num1(v, dates::mofd),
        "qofd" => num1(v, dates::qofd),
        "yofd" => num1(v, dates::yofd),
        "wofd" => num1(v, dates::wofd),
        "hofd" => num1(v, dates::hofd),
        "dofm" => num1(v, dates::dofm),
        "dofq" => num1(v, dates::dofq),
        "dofy" => num1(v, dates::dofy),
        "dofw" => num1(v, dates::dofw),
        "dofh" => num1(v, dates::dofh),
        "dofc" | "dofC" => num1(v, dates::dofc),
        "cofd" | "Cofd" => num1(v, dates::cofd),
        "ym" => num2(v, dates::ym),
        "yq" => num2(v, dates::yq),
        "yh" => num2(v, dates::yh),
        "yw" => num2(v, dates::yw),
        "hh" => num1(v, dates::hh),
        "mm" => num1(v, dates::mm),
        "ss" => num1(v, dates::ss),
        "today" => {
            let now = chrono::Local::now();
            use chrono::Datelike;
            num(dates::mdy(now.month() as f64, now.day() as f64, now.year() as f64))
        }
        "pi" => num(consts::PI),
        _ => Err(rc_err(133, format!("unknown function {name}()"))),
    }
}

// ---------------------------------------------------------------------------
// Backwards-compatible helpers
// ---------------------------------------------------------------------------

/// Result of evaluating an expression (legacy API).
#[derive(Debug, Clone, PartialEq)]
pub enum EvalResult {
    Number(f64),
    String(String),
    Missing,
}

impl EvalResult {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            EvalResult::Number(n) => Some(*n),
            EvalResult::String(s) => s.parse::<f64>().ok(),
            EvalResult::Missing => None,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            EvalResult::Number(n) => *n != 0.0,
            EvalResult::String(s) => !s.is_empty(),
            EvalResult::Missing => true,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, EvalResult::Missing)
    }
}

/// Evaluate an expression for one observation without interpreter state.
pub fn eval_expr(expr: &Expr, ds: &DataSet, row: usize) -> Result<EvalResult, String> {
    let env = Env::new(ds);
    Ok(match env.eval(expr, row)? {
        Val::Num(n) if is_missing(n) => EvalResult::Missing,
        Val::Num(n) => EvalResult::Number(n),
        Val::Str(s) => EvalResult::String(s),
    })
}

pub fn expr_to_string(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(s) => Some(s.clone()),
        Expr::String(s) => Some(s.clone()),
        Expr::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Number(n) => Some(*n),
        Expr::Unary(UnaryOp::Neg, x) => expr_to_f64(x).map(|v| -v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(substr("hello", 2.0, 3.0), "ell");
        assert_eq!(substr("hello", -3.0, SYSMISS), "llo");
        assert!(wildcard_match("a*c?", "abbbcd"));
        assert!(!wildcard_match("a*c", "abd"));
        assert_eq!(round_to(3.14159, 0.01), 3.14);
        assert_eq!(round_to(17.0, 5.0), 15.0);
        assert_eq!(round_to(-2.5, 1.0), -3.0);
        assert_eq!(soundex("Robert"), "R163");
    }
}
