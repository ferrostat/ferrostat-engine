//! Helpers shared by commands: sample selection, weights, groups, columns.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::{Command, Expr};

use crate::eval::{rc_err, Env, Groups, Val};
use crate::store::Store;

/// Observations selected by `if`/`in` (and by the current `by` group).
pub fn touse(ds: &DataSet, st: &Store, cmd: &Command) -> Result<Vec<bool>, String> {
    let n = ds.n_obs;
    let mut mask = vec![true; n];
    if let Some(m) = &st_by_mask(st) {
        for i in 0..n {
            mask[i] = m.get(i).copied().unwrap_or(false);
        }
    }
    if let Some(r) = &cmd.in_range {
        let (a, b) = r.resolve(n).map_err(|e| rc_err(198, e))?;
        for (i, m) in mask.iter_mut().enumerate() {
            if i < a || i >= b {
                *m = false;
            }
        }
    }
    if let Some(e) = &cmd.if_expr {
        let groups = st.by_groups.as_ref();
        let env = Env::with_store(ds, st).groups(groups);
        for i in 0..n {
            if mask[i] {
                mask[i] = env.eval(e, i)?.num()? != 0.0;
            }
        }
    }
    Ok(mask)
}

fn st_by_mask(st: &Store) -> Option<Vec<bool>> {
    st.by_mask.clone()
}

/// Exclude observations where any listed variable is missing.
pub fn markout(ds: &DataSet, mask: &mut [bool], vars: &[String]) {
    for v in vars {
        let name = base_name(v);
        if let Some(var) = ds.get_var(name) {
            for (i, m) in mask.iter_mut().enumerate() {
                if *m {
                    let miss = match &var.data[i] {
                        Value::Str(s) => s.is_empty(),
                        x => is_missing(x.num()),
                    };
                    if miss {
                        *m = false;
                    }
                }
            }
        }
    }
}

/// Strip factor/time-series operators: `i.group` -> `group`.
pub fn base_name(v: &str) -> &str {
    match crate::varlist::split_operator(v) {
        Some((_, rest)) => rest,
        None => v,
    }
}

pub fn rows(mask: &[bool]) -> Vec<usize> {
    mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i).collect()
}

/// Numeric column (missing kept as missing codes).
pub fn col(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let v = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if v.var_type.is_string() {
        return Err(rc_err(109, "type mismatch"));
    }
    Ok(v.data.iter().map(|x| x.num()).collect())
}

/// Evaluate an expression for every observation in `mask`.
pub fn eval_col(ds: &DataSet, st: &Store, e: &Expr, mask: &[bool]) -> Result<Vec<f64>, String> {
    let env = Env::with_store(ds, st);
    let mut out = vec![SYSMISS; ds.n_obs];
    for i in 0..ds.n_obs {
        if mask[i] {
            out[i] = env.eval(e, i)?.num()?;
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightKind {
    F,
    A,
    P,
    I,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub kind: WeightKind,
    pub w: Vec<f64>,
    pub expr: String,
}

/// Evaluate the weight expression; drops observations with missing or
/// non-positive weights from `mask`. `allowed` lists accepted kinds.
pub fn weights(ds: &DataSet, st: &Store, cmd: &Command, mask: &mut [bool], allowed: &str) -> Result<Option<Weights>, String> {
    let Some(ws) = &cmd.weight else { return Ok(None) };
    let kind = match ws.weight_type.to_lowercase().as_str() {
        "fw" | "fweight" | "frequency" => WeightKind::F,
        "aw" | "aweight" | "cellsize" => WeightKind::A,
        "pw" | "pweight" => WeightKind::P,
        "iw" | "iweight" => WeightKind::I,
        "weight" => {
            if allowed.contains('f') { WeightKind::F } else { WeightKind::A }
        }
        w => return Err(rc_err(198, format!("{w} not allowed"))),
    };
    let code = match kind {
        WeightKind::F => 'f',
        WeightKind::A => 'a',
        WeightKind::P => 'p',
        WeightKind::I => 'i',
    };
    if !allowed.contains(code) {
        return Err(rc_err(101, format!("{}weights not allowed", code)));
    }
    let w = eval_col(ds, st, &ws.expr, mask)?;
    for i in 0..ds.n_obs {
        if mask[i] {
            if is_missing(w[i]) {
                mask[i] = false;
            } else if w[i] < 0.0 && kind != WeightKind::I {
                return Err(rc_err(402, "negative weights encountered"));
            } else if w[i] == 0.0 && kind != WeightKind::I {
                mask[i] = false;
            } else if kind == WeightKind::F && w[i].fract() != 0.0 {
                return Err(rc_err(401, "may not use noninteger frequency weights"));
            }
        }
    }
    Ok(Some(Weights { kind, w, expr: format!("{:?}", ws.expr) }))
}

/// Compute `by` groups for data sorted by `vars`; `mask` restricts which
/// observations count (others get their own singleton ranges).
pub fn groups_by(ds: &DataSet, vars: &[String]) -> Groups {
    let n = ds.n_obs;
    let mut g = Groups { start: vec![0; n], end: vec![n; n] };
    if n == 0 {
        return g;
    }
    let cols: Vec<&Variable> = vars.iter().filter_map(|v| ds.get_var(v)).collect();
    let same = |a: usize, b: usize| cols.iter().all(|v| v.data[a].partial_cmp(&v.data[b]) == Some(std::cmp::Ordering::Equal) || (v.data[a].is_missing() && v.data[b].is_missing() && v.data[a].num() == v.data[b].num()));
    let mut s = 0;
    for i in 1..=n {
        if i == n || !same(i - 1, i) {
            for j in s..i {
                g.start[j] = s;
                g.end[j] = i;
            }
            s = i;
        }
    }
    g
}

/// Is the dataset sorted by these variables?
pub fn is_sorted_by(ds: &DataSet, vars: &[String]) -> bool {
    if ds.sort_by.len() >= vars.len() && ds.sort_by[..vars.len()] == *vars {
        return true;
    }
    let cols: Vec<&Variable> = vars.iter().filter_map(|v| ds.get_var(v)).collect();
    (1..ds.n_obs).all(|i| {
        for v in &cols {
            match v.data[i - 1].partial_cmp(&v.data[i]) {
                Some(std::cmp::Ordering::Less) => return true,
                Some(std::cmp::Ordering::Greater) => return false,
                _ => {}
            }
        }
        true
    })
}

/// Distinct groups (in sorted order of key values) for observations in mask.
/// Returns (group id per obs or usize::MAX, representative obs per group).
pub fn group_ids(ds: &DataSet, vars: &[String], mask: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let cols: Vec<&Variable> = vars.iter().filter_map(|v| ds.get_var(v)).collect();
    let mut idx: Vec<usize> = rows(mask);
    let cmp = |a: &usize, b: &usize| {
        for v in &cols {
            let o = v.data[*a].partial_cmp(&v.data[*b]).unwrap_or(std::cmp::Ordering::Equal);
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    };
    idx.sort_by(|a, b| cmp(a, b).then(a.cmp(b)));
    let mut ids = vec![usize::MAX; ds.n_obs];
    let mut reps = Vec::new();
    for (k, &i) in idx.iter().enumerate() {
        if k == 0 || cmp(&idx[k - 1], &i) != std::cmp::Ordering::Equal {
            reps.push(i);
        }
        ids[i] = reps.len() - 1;
    }
    (ids, reps)
}

/// Text of a cell for display, using value labels when `labels` is true.
pub fn cell_text(ds: &DataSet, var: &Variable, i: usize, labels: bool) -> String {
    match &var.data[i] {
        Value::Str(s) => s.clone(),
        Value::StrL(a, b) => ds.resolve_strl(*a, *b).unwrap_or("").to_string(),
        x => {
            let v = x.num();
            if labels && !var.value_label_name.is_empty() {
                if let Some(l) = ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(v)) {
                    return l.to_string();
                }
            }
            fmt_cell(v, &var.format)
        }
    }
}

/// Number formatted with a variable's display format, trimmed.
pub fn fmt_cell(v: f64, f: &str) -> String {
    if f.is_empty() || f == "%9.0g" || f == "%8.0g" || f == "%10.0g" {
        crate::format::fmt_g(v, 9)
    } else {
        crate::format::fmt(v, f).trim().to_string()
    }
}

/// Make a new numeric variable from values (stored as double).
pub fn new_double(name: &str, vals: Vec<f64>) -> Variable {
    let mut v = Variable::new(name, VarType::Double);
    v.data = vals.into_iter().map(Value::Double).collect();
    v
}

/// Choose the smallest storage type that holds all values exactly.
pub fn best_numeric_type(vals: &[f64]) -> VarType {
    let mut ty = VarType::Byte;
    for &v in vals {
        if is_missing(v) {
            continue;
        }
        if v.fract() != 0.0 {
            return if (v as f32) as f64 == v { promote(ty, VarType::Float) } else { VarType::Double };
        }
        let t = if (-127.0..=100.0).contains(&v) {
            VarType::Byte
        } else if (-32767.0..=32740.0).contains(&v) {
            VarType::Int
        } else if (-2147483647.0..=2147483620.0).contains(&v) {
            VarType::Long
        } else {
            VarType::Double
        };
        ty = promote(ty, t);
    }
    ty
}

/// The wider of two numeric types.
pub fn promote(a: VarType, b: VarType) -> VarType {
    let rank = |t: VarType| match t {
        VarType::Byte => 0,
        VarType::Int => 1,
        VarType::Long => 2,
        VarType::Float => 3,
        _ => 4,
    };
    // long and float together need double to stay exact
    if (a == VarType::Long && b == VarType::Float) || (a == VarType::Float && b == VarType::Long) {
        return VarType::Double;
    }
    if rank(a) >= rank(b) { a } else { b }
}

/// Store a numeric value into a typed cell.
pub fn typed_value(ty: VarType, v: f64) -> Value {
    if is_missing(v) {
        return Value::Double(ferrostat_core::missing::normalize(v));
    }
    match ty {
        VarType::Byte => Value::Byte(v as i8),
        VarType::Int => Value::Int(v as i16),
        VarType::Long => Value::Long(v as i32),
        VarType::Float => Value::Float(v as f32),
        _ => Value::Double(v),
    }
}

/// Does a value fit a numeric type without loss?
pub fn fits(ty: VarType, v: f64) -> bool {
    if is_missing(v) {
        return true;
    }
    match ty {
        VarType::Byte => v.fract() == 0.0 && (-127.0..=100.0).contains(&v),
        VarType::Int => v.fract() == 0.0 && (-32767.0..=32740.0).contains(&v),
        VarType::Long => v.fract() == 0.0 && (-2147483647.0..=2147483620.0).contains(&v),
        VarType::Float => (v as f32) as f64 == v,
        _ => true,
    }
}

pub fn str_type_for(len: usize) -> VarType {
    if len > 2045 { VarType::StrL } else { VarType::Str(len.max(1)) }
}

/// Weighted mean and variance helpers.
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Percentile as defined for `summarize, detail` and `centile`-free uses:
/// with weights all 1, P = (n p / 100); if integer, average of x[P], x[P+1].
pub fn pctile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return SYSMISS;
    }
    let np = n as f64 * p / 100.0;
    let i = np.floor() as usize;
    if i == 0 {
        return sorted[0];
    }
    if i >= n {
        return sorted[n - 1];
    }
    if (np - i as f64).abs() < 1e-12 { (sorted[i - 1] + sorted[i]) / 2.0 } else { sorted[i] }
}

/// Weighted percentile (frequency/analytic weights).
pub fn pctile_weighted(pairs: &mut [(f64, f64)], p: f64) -> f64 {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|x| x.1).sum();
    let target = total * p / 100.0;
    let mut cum = 0.0;
    for (k, (x, w)) in pairs.iter().enumerate() {
        cum += w;
        if (cum - target).abs() < 1e-9 * total.max(1.0) {
            let next = pairs.get(k + 1).map_or(*x, |n| n.0);
            return (x + next) / 2.0;
        }
        if cum > target {
            return *x;
        }
    }
    pairs.last().map_or(SYSMISS, |x| x.0)
}

/// Value of an option given as a number (e.g. `level(90)`).
pub fn opt_num(cmd: &Command, name: &str, min: usize) -> Result<Option<f64>, String> {
    match cmd.opt(name, min) {
        Some(t) => {
            let e = ferrostat_parser::parser::parse_expression(t).map_err(|e| rc_err(198, e))?;
            match crate::eval::Env::new(&DataSet::new()).eval(&e, 0)? {
                Val::Num(v) => Ok(Some(v)),
                _ => Err(rc_err(109, "type mismatch")),
            }
        }
        None => Ok(None),
    }
}

/// Unquote `"text"` or compound-quoted text.
pub fn unquote(s: &str) -> String {
    let t = s.trim();
    if let Some(x) = t.strip_prefix("`\"").and_then(|x| x.strip_suffix("\"'")) {
        return x.to_string();
    }
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        return t[1..t.len() - 1].to_string();
    }
    t.to_string()
}

/// Confidence level from `level()` option or the store default.
pub fn level(cmd: &Command, st: &Store) -> Result<f64, String> {
    let l = opt_num(cmd, "level", 1)?.unwrap_or(st.level);
    if !(10.0..=99.99).contains(&l) {
        return Err(rc_err(198, "level() must be between 10 and 99.99 inclusive"));
    }
    Ok(l)
}

/// Draw a horizontal rule of `n` dashes with `+` at `at` positions.
pub fn rule(widths: &[usize]) -> String {
    widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("+")
}
