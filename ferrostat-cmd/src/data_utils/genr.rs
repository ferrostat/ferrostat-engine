//! Generating and transforming data: `range`, `ipolate`, `stack`,
//! `expandcl`, `pctile`, `_pctile`, `drawnorm`, `corr2data`, `splitsample`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util::{self, typed_value};
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("range", Box::new(Range)),
        ("ipolate", Box::new(Ipolate)),
        ("stack", Box::new(Stack)),
        ("expandcl", Box::new(Expandcl)),
        ("pctile", Box::new(Pctile)),
        ("_pctile", Box::new(PctileR)),
        ("drawnorm", Box::new(Drawnorm)),
        ("corr2data", Box::new(Drawnorm)),
        ("splitsample", Box::new(Splitsample)),
    ]
}

fn eval_num(ds: &DataSet, st: &Store, text: &str) -> Result<f64, String> {
    let e = ferrostat_parser::parser::parse_expression(text.trim()).map_err(|e| rc_err(198, e.message))?;
    Env::with_store(ds, st).eval(&e, 0)?.num()
}

fn add_num_var(ds: &mut DataSet, name: &str, ty: VarType, vals: &[f64]) -> Result<(), String> {
    let mut v = Variable::new(name, ty);
    v.data = vals.iter().map(|&x| typed_value(ty, x)).collect();
    ds.add_variable(v)
}

fn ensure_obs(ds: &mut DataSet, n: usize) {
    while ds.n_obs < n {
        ds.add_observation();
    }
}

// ---------------------------------------------------------------- range

struct Range;

impl CommandHandler for Range {
    fn name(&self) -> &'static str {
        "range"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let toks = super::vars::split_words(&cmd.main);
        if toks.len() < 3 || toks.len() > 4 {
            return Err(rc_err(198, "range newvar # # [#]"));
        }
        let name = toks[0].clone();
        varlist::check_name(&name)?;
        let a = eval_num(ds, st, &toks[1])?;
        let b = eval_num(ds, st, &toks[2])?;
        let n = match toks.get(3) {
            Some(t) => eval_num(ds, st, t)? as usize,
            None => ds.n_obs,
        };
        if n == 0 {
            return Err(rc_err(198, "no observations"));
        }
        ensure_obs(ds, n);
        let vals: Vec<f64> = (0..ds.n_obs)
            .map(|i| {
                if i >= n {
                    SYSMISS
                } else if n == 1 {
                    a
                } else {
                    a + (b - a) * i as f64 / (n - 1) as f64
                }
            })
            .collect();
        add_num_var(ds, &name, VarType::Double, &vals)?;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------- ipolate

struct Ipolate;

impl CommandHandler for Ipolate {
    fn name(&self) -> &'static str {
        "ipolate"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(102, "ipolate requires two variables: y x"));
        }
        let newname = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        varlist::check_name(&newname)?;
        let epolate = cmd.has_opt("epolate", 1);
        let y = util::col(ds, &vars[0])?;
        let x = util::col(ds, &vars[1])?;
        let mask = util::touse(ds, st, cmd)?;
        let by = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let (gid, _) = if by.is_empty() { (vec![0; ds.n_obs], vec![]) } else { util::group_ids(ds, &by, &vec![true; ds.n_obs]) };
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..ds.n_obs {
            groups.entry(gid[i]).or_default().push(i);
        }
        let mut out = vec![SYSMISS; ds.n_obs];
        for (_, rows) in groups {
            let mut pts: Vec<(f64, f64)> = rows.iter().filter(|&&i| mask[i] && !is_missing(y[i]) && !is_missing(x[i])).map(|&i| (x[i], y[i])).collect();
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &i in &rows {
                if !mask[i] {
                    continue;
                }
                if !is_missing(y[i]) {
                    out[i] = y[i];
                    continue;
                }
                if is_missing(x[i]) || pts.len() < 2 {
                    continue;
                }
                let xi = x[i];
                let k = pts.partition_point(|p| p.0 < xi);
                let (lo, hi) = if k == 0 {
                    if !epolate {
                        continue;
                    }
                    (0, 1)
                } else if k >= pts.len() {
                    if !epolate {
                        continue;
                    }
                    (pts.len() - 2, pts.len() - 1)
                } else {
                    (k - 1, k)
                };
                let (x0, y0) = pts[lo];
                let (x1, y1) = pts[hi];
                if x1 == x0 {
                    continue;
                }
                out[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0);
            }
        }
        add_num_var(ds, &newname, VarType::Float, &out)?;
        let missing = out.iter().zip(&mask).filter(|(v, m)| **m && is_missing(**v)).count();
        st.data_changed = true;
        Ok(if missing > 0 { format!("({missing} missing value{} generated)", if missing == 1 { "" } else { "s" }) } else { String::new() })
    }
}

// ---------------------------------------------------------------- stack

struct Stack;

impl CommandHandler for Stack {
    fn name(&self) -> &'static str {
        "stack"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let into: Vec<String> = cmd.opt("into", 3).ok_or_else(|| rc_err(198, "option into() required"))?.split_whitespace().map(String::from).collect();
        if into.is_empty() {
            return Err(rc_err(198, "into() requires new variable names"));
        }
        let wide = cmd.has_opt("wide", 1);
        let k = into.len();
        // varlist: names, `.` for a missing column, groups separated by `\` or consecutive
        let mut names: Vec<Option<String>> = Vec::new();
        for tok in cmd.main.replace('\\', " ").split_whitespace() {
            if tok == "." {
                names.push(None);
            } else {
                for v in varlist::expand(ds, tok)? {
                    names.push(Some(v));
                }
            }
        }
        if names.is_empty() || names.len() % k != 0 {
            return Err(rc_err(198, format!("number of variables must be a multiple of {k}")));
        }
        let mask = util::touse(ds, st, cmd)?;
        let rows = util::rows(&mask);
        let blocks = names.len() / k;
        let mut out = DataSet::new();
        let total = rows.len() * blocks;
        let mut stackv = Variable::new("_stack", VarType::Byte);
        for b in 0..blocks {
            for _ in &rows {
                stackv.data.push(Value::Byte((b + 1) as i8));
            }
        }
        if wide {
            for v in &ds.vars {
                let mut nv = Variable::new(v.name.clone(), v.var_type);
                nv.format = v.format.clone();
                nv.label = v.label.clone();
                nv.value_label_name = v.value_label_name.clone();
                for _ in 0..blocks {
                    for &i in &rows {
                        nv.data.push(v.data[i].clone());
                    }
                }
                out.n_obs = total;
                out.add_variable(nv)?;
            }
            out.value_labels = ds.value_labels.clone();
        }
        out.n_obs = total;
        out.add_variable(stackv)?;
        for (j, newname) in into.iter().enumerate() {
            varlist::check_name(newname)?;
            let srcs: Vec<Option<&Variable>> = (0..blocks).map(|b| names[b * k + j].as_ref().and_then(|n| ds.get_var(n))).collect();
            let all_str = srcs.iter().flatten().all(|v| v.var_type.is_string()) && srcs.iter().flatten().count() > 0;
            let mut ty = if all_str {
                VarType::Str(srcs.iter().flatten().map(|v| v.var_type.storage_width()).max().unwrap_or(1))
            } else {
                VarType::Byte
            };
            if !all_str {
                for v in srcs.iter().flatten() {
                    if v.var_type.is_string() {
                        return Err(rc_err(109, "type mismatch: cannot stack numeric and string variables"));
                    }
                    ty = util::promote(ty, v.var_type);
                }
            }
            let mut nv = Variable::new(newname.clone(), ty);
            if let Some(Some(v)) = srcs.first() {
                nv.format = v.format.clone();
                nv.label = v.label.clone();
                nv.value_label_name = v.value_label_name.clone();
            }
            for s in &srcs {
                for &i in &rows {
                    nv.data.push(match s {
                        Some(v) if all_str => Value::Str(v.data[i].as_string().unwrap_or("").to_string()),
                        Some(v) => typed_value(ty, v.num(i)),
                        None if all_str => Value::Str(String::new()),
                        None => typed_value(ty, SYSMISS),
                    });
                }
            }
            if wide && out.var_index.contains_key(newname) {
                out.drop_variable(newname)?;
            }
            out.add_variable(nv)?;
            if let Some(Some(v)) = srcs.first() {
                if !v.value_label_name.is_empty() {
                    if let Some(l) = ds.value_labels.get(&v.value_label_name) {
                        out.value_labels.insert(v.value_label_name.clone(), l.clone());
                    }
                }
            }
        }
        out.datalabel = ds.datalabel.clone();
        *ds = out;
        st.data_changed = true;
        st.tsset = None;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------- expandcl

struct Expandcl;

impl CommandHandler for Expandcl {
    fn name(&self) -> &'static str {
        "expandcl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let text = cmd.main.trim().trim_start_matches('=').trim();
        let e = ferrostat_parser::parser::parse_expression(text).map_err(|e| rc_err(198, e.message))?;
        let cl = varlist::expand(ds, cmd.opt("cluster", 2).ok_or_else(|| rc_err(198, "option cluster() required"))?)?;
        let newname = cmd.opt("generate", 3).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        varlist::check_name(&newname)?;
        let mask = util::touse(ds, st, cmd)?;
        let (gid, _) = util::group_ids(ds, &cl, &mask);
        let env = Env::with_store(ds, st);
        let n = ds.n_obs;
        let mut copies: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..n {
            if mask[i] {
                let k = env.eval(&e, i)?.num()?;
                if is_missing(k) || k < 1.0 {
                    return Err(rc_err(198, "expansion factor must be at least 1"));
                }
                let k = k.floor() as usize;
                match copies.get(&gid[i]) {
                    Some(&old) if old != k => return Err(rc_err(459, "expression not constant within cluster")),
                    _ => {
                        copies.insert(gid[i], k);
                    }
                }
            }
        }
        let nclust = copies.len();
        let rank: BTreeMap<usize, usize> = copies.keys().enumerate().map(|(r, g)| (*g, r + 1)).collect();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut ids: Vec<f64> = (0..n).map(|i| if mask[i] { rank[&gid[i]] as f64 } else { SYSMISS }).collect();
        let maxk = copies.values().copied().max().unwrap_or(1);
        for j in 1..maxk {
            for i in 0..n {
                if mask[i] && copies[&gid[i]] > j {
                    idx.push(i);
                    ids.push((nclust * j + rank[&gid[i]]) as f64);
                }
            }
        }
        let added = idx.len() - n;
        *ds = crate::restructure::take_rows(ds, &idx);
        add_num_var(ds, &newname, VarType::Long, &ids)?;
        st.data_changed = true;
        Ok(format!("({added} observations created)"))
    }
}

// ---------------------------------------------------------------- pctile

fn quantile_points(cmd: &Command) -> Result<Vec<f64>, String> {
    if let Some(p) = cmd.opt("percentiles", 1) {
        let mut v = Vec::new();
        for t in p.split_whitespace() {
            if let Some((a, b)) = t.split_once('/') {
                let (a, b): (f64, f64) = (a.parse().map_err(|_| rc_err(198, "invalid percentiles()"))?, b.parse().map_err(|_| rc_err(198, "invalid percentiles()"))?);
                let mut x = a;
                while x <= b + 1e-9 {
                    v.push(x);
                    x += 1.0;
                }
            } else {
                v.push(t.parse().map_err(|_| rc_err(198, "invalid percentiles()"))?);
            }
        }
        return Ok(v);
    }
    let nq = util::opt_num(cmd, "nquantiles", 2)?.unwrap_or(2.0) as usize;
    if nq < 2 {
        return Err(rc_err(198, "nquantiles() must be at least 2"));
    }
    Ok((1..nq).map(|k| 100.0 * k as f64 / nq as f64).collect())
}

fn percentiles_of(vals: &[f64], pts: &[f64], altdef: bool) -> Vec<f64> {
    let mut s: Vec<f64> = vals.iter().copied().filter(|x| !is_missing(*x)).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    pts.iter()
        .map(|&p| {
            if altdef && !s.is_empty() {
                let h = (s.len() as f64 + 1.0) * p / 100.0;
                let k = h.floor() as usize;
                if k < 1 {
                    s[0]
                } else if k >= s.len() {
                    s[s.len() - 1]
                } else {
                    s[k - 1] + (h - k as f64) * (s[k] - s[k - 1])
                }
            } else {
                util::pctile_sorted(&s, p)
            }
        })
        .collect()
}

struct Pctile;

impl CommandHandler for Pctile {
    fn name(&self) -> &'static str {
        "pctile"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (lhs, rhs) = cmd.main.split_once('=').ok_or_else(|| rc_err(198, "pctile newvar = exp"))?;
        let toks: Vec<&str> = lhs.split_whitespace().collect();
        let (ty, name) = match toks.as_slice() {
            [n] => (None, *n),
            [t, n] => (VarType::from_type_str(t), *n),
            _ => return Err(rc_err(198, "pctile newvar = exp")),
        };
        varlist::check_name(name)?;
        let e = ferrostat_parser::parser::parse_expression(rhs.trim()).map_err(|e| rc_err(198, e.message))?;
        let mask = util::touse(ds, st, cmd)?;
        let x = util::eval_col(ds, st, &e, &mask)?;
        let vals: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| x[i]).collect();
        let pts = quantile_points(cmd)?;
        let q = percentiles_of(&vals, &pts, cmd.has_opt("altdef", 3));
        ensure_obs(ds, q.len());
        let mut col = vec![SYSMISS; ds.n_obs];
        let mut pcol = vec![SYSMISS; ds.n_obs];
        for (k, v) in q.iter().enumerate() {
            col[k] = *v;
            pcol[k] = pts[k];
        }
        let genp = cmd.opt("genp", 4).map(|s| s.trim().to_string());
        if let Some(g) = &genp {
            varlist::check_name(g)?;
        }
        add_num_var(ds, name, ty.unwrap_or(VarType::Float), &col)?;
        if let Some(g) = genp {
            add_num_var(ds, &g, VarType::Float, &pcol)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

struct PctileR;

impl CommandHandler for PctileR {
    fn name(&self) -> &'static str {
        "_pctile"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = ferrostat_parser::parser::parse_expression(cmd.main.trim()).map_err(|e| rc_err(198, e.message))?;
        let mask = util::touse(ds, st, cmd)?;
        let x = util::eval_col(ds, st, &e, &mask)?;
        let vals: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| x[i]).collect();
        let pts = if cmd.opt("percentiles", 1).is_none() && cmd.opt("nquantiles", 2).is_none() { vec![50.0] } else { quantile_points(cmd)? };
        let q = percentiles_of(&vals, &pts, cmd.has_opt("altdef", 3));
        let r = st.begin_r();
        for (k, v) in q.iter().enumerate() {
            r.set(&format!("r{}", k + 1), *v);
        }
        Ok(String::new())
    }
}

// ---------------------------------------------------------------- drawnorm / corr2data

fn matrix_arg(st: &Store, name: &str) -> Result<Matrix, String> {
    st.matrices.get(name.trim()).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", name.trim())))
}

/// Expand a matrix given in `cstorage()` form into a full symmetric one.
fn full_matrix(m: Matrix, k: usize, storage: &str) -> Result<Matrix, String> {
    match storage {
        "full" => {
            if m.rows != k || m.cols != k {
                return Err(rc_err(503, "covariance/correlation matrix has wrong dimensions"));
            }
            Ok(m)
        }
        "lower" | "upper" => {
            if m.data.len() != k * (k + 1) / 2 {
                return Err(rc_err(503, "matrix has wrong number of elements for cstorage()"));
            }
            let mut out = Matrix::zeros(k, k);
            let mut p = 0;
            for i in 0..k {
                if storage == "lower" {
                    for j in 0..=i {
                        out[(i, j)] = m.data[p];
                        out[(j, i)] = m.data[p];
                        p += 1;
                    }
                } else {
                    for j in i..k {
                        out[(i, j)] = m.data[p];
                        out[(j, i)] = m.data[p];
                        p += 1;
                    }
                }
            }
            Ok(out)
        }
        o => Err(rc_err(198, format!("cstorage({o}) invalid"))),
    }
}

struct Drawnorm;

impl CommandHandler for Drawnorm {
    fn name(&self) -> &'static str {
        "drawnorm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let exact = cmd.verb == "corr2data";
        let names = varlist::new_names(ds, &cmd.main)?;
        let k = names.len();
        if k == 0 {
            return Err(rc_err(100, "newvarlist required"));
        }
        if let Some(s) = util::opt_num(cmd, "seed", 4)? {
            st.rng.borrow_mut().reseed(s as u64);
        }
        let n = match util::opt_num(cmd, "n", 1)? {
            Some(n) => n as usize,
            None => ds.n_obs,
        };
        if n == 0 {
            return Err(rc_err(198, "n() required when there are no observations in memory"));
        }
        if n != ds.n_obs {
            if ds.n_vars() == 0 || cmd.has_opt("clear", 5) {
                *ds = DataSet::new();
                ensure_obs(ds, n);
            } else if n > ds.n_obs {
                ensure_obs(ds, n);
            } else {
                return Err(rc_err(198, "n() is smaller than the number of observations in memory; use clear"));
            }
        }
        let storage = cmd.opt("cstorage", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "full".into());
        let means: Vec<f64> = match cmd.opt("means", 1) {
            Some(m) => {
                let m = matrix_arg(st, m)?;
                if m.data.len() != k {
                    return Err(rc_err(503, "means() must have one element per variable"));
                }
                m.data
            }
            None => vec![0.0; k],
        };
        let sds: Option<Vec<f64>> = match cmd.opt("sds", 2) {
            Some(m) => {
                let m = matrix_arg(st, m)?;
                if m.data.len() != k {
                    return Err(rc_err(503, "sds() must have one element per variable"));
                }
                Some(m.data)
            }
            None => None,
        };
        let cov = match (cmd.opt("cov", 2), cmd.opt("corr", 2)) {
            (Some(c), _) => full_matrix(matrix_arg(st, c)?, k, &storage)?,
            (None, Some(c)) => {
                let r = full_matrix(matrix_arg(st, c)?, k, &storage)?;
                let s = sds.clone().unwrap_or_else(|| vec![1.0; k]);
                let mut c = Matrix::zeros(k, k);
                for i in 0..k {
                    for j in 0..k {
                        c[(i, j)] = r[(i, j)] * s[i] * s[j];
                    }
                }
                c
            }
            (None, None) => {
                let s = sds.clone().unwrap_or_else(|| vec![1.0; k]);
                Matrix::from_diag(&s.iter().map(|x| x * x).collect::<Vec<_>>())
            }
        };
        let chol = cov.cholesky().map_err(|_| rc_err(506, "covariance matrix is not positive definite"))?;
        // independent standard normal draws, n x k
        let mut z: Vec<Vec<f64>> = (0..k).map(|_| (0..n).map(|_| st.rng.borrow_mut().normal()).collect()).collect();
        if exact {
            if n <= k {
                return Err(rc_err(198, "n() must exceed the number of variables"));
            }
            for c in z.iter_mut() {
                let m = c.iter().sum::<f64>() / n as f64;
                c.iter_mut().for_each(|x| *x -= m);
            }
            // sample covariance of z and its Cholesky factor
            let mut s = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    s[(i, j)] = z[i].iter().zip(&z[j]).map(|(a, b)| a * b).sum::<f64>() / (n as f64 - 1.0);
                }
            }
            let ls = s.cholesky().map_err(|_| rc_err(506, "draws are collinear"))?;
            let inv = ls.inverse()?;
            // whitened = z * inv'
            let mut w = vec![vec![0.0; n]; k];
            for i in 0..k {
                for j in 0..=i {
                    let f = inv[(i, j)];
                    for t in 0..n {
                        w[i][t] += f * z[j][t];
                    }
                }
            }
            z = w;
        }
        let ty = if cmd.has_opt("double", 1) { VarType::Double } else { VarType::Float };
        for (i, name) in names.iter().enumerate() {
            let col: Vec<f64> = (0..n).map(|t| means[i] + (0..=i).map(|j| chol[(i, j)] * z[j][t]).sum::<f64>()).collect();
            add_num_var(ds, name, ty, &col)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------- splitsample

struct Splitsample;

impl CommandHandler for Splitsample {
    fn name(&self) -> &'static str {
        "splitsample"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let newname = cmd.opt("generate", 1).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        varlist::check_name(&newname)?;
        let fr: Vec<f64> = if let Some(s) = cmd.opt("split", 2) {
            s.split_whitespace().map(|t| t.parse::<f64>().map_err(|_| rc_err(198, "invalid split()"))).collect::<Result<_, _>>()?
        } else if let Some(k) = util::opt_num(cmd, "nsplit", 2)? {
            vec![1.0 / k; k as usize]
        } else {
            vec![0.5, 0.5]
        };
        if fr.len() < 2 || fr.iter().any(|f| *f <= 0.0) {
            return Err(rc_err(198, "split() requires at least two positive fractions"));
        }
        let tot: f64 = fr.iter().sum();
        let fr: Vec<f64> = fr.iter().map(|f| f / tot).collect();
        let mut mask = util::touse(ds, st, cmd)?;
        if !cmd.main.trim().is_empty() {
            let vars = varlist::expand(ds, &cmd.main)?;
            util::markout(ds, &mut mask, &vars);
        }
        let rround = cmd.has_opt("rround", 2);
        // units: clusters or single observations
        let units: Vec<Vec<usize>> = match cmd.opt("cluster", 2) {
            Some(c) => {
                let cv = varlist::expand(ds, c)?;
                let (gid, _) = util::group_ids(ds, &cv, &mask);
                let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for i in 0..ds.n_obs {
                    if mask[i] {
                        m.entry(gid[i]).or_default().push(i);
                    }
                }
                m.into_values().collect()
            }
            None => (0..ds.n_obs).filter(|&i| mask[i]).map(|i| vec![i]).collect(),
        };
        // strata
        let strata: Vec<Vec<usize>> = match cmd.opt("balance", 3) {
            Some(b) => {
                let bv = varlist::expand(ds, b)?;
                let (gid, _) = util::group_ids(ds, &bv, &mask);
                let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (u, rows) in units.iter().enumerate() {
                    m.entry(gid[rows[0]]).or_default().push(u);
                }
                m.into_values().collect()
            }
            None => vec![(0..units.len()).collect()],
        };
        let mut out = vec![SYSMISS; ds.n_obs];
        for mut stratum in strata {
            // random permutation
            let keys: Vec<f64> = stratum.iter().map(|_| st.rng.borrow_mut().uniform()).collect();
            let mut order: Vec<usize> = (0..stratum.len()).collect();
            order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
            stratum = order.iter().map(|&o| stratum[o]).collect();
            let m = stratum.len() as f64;
            let mut bounds = Vec::new();
            let mut cum = 0.0;
            for f in &fr {
                cum += f;
                let x = m * cum.min(1.0);
                let b = if rround {
                    let fl = x.floor();
                    if st.rng.borrow_mut().uniform() < x - fl { fl + 1.0 } else { fl }
                } else {
                    (x + 1e-9).round()
                };
                bounds.push(b as usize);
            }
            *bounds.last_mut().unwrap() = stratum.len();
            let mut grp = 0;
            for (pos, &u) in stratum.iter().enumerate() {
                while pos >= bounds[grp] && grp + 1 < bounds.len() {
                    grp += 1;
                }
                for &i in &units[u] {
                    out[i] = (grp + 1) as f64;
                }
            }
        }
        add_num_var(ds, &newname, VarType::Byte, &out)?;
        st.data_changed = true;
        Ok(String::new())
    }
}
