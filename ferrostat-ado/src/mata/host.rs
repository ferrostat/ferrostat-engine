//! The interface to the host session: data, macros, scalars, matrices,
//! commands, and the `xl()` spreadsheet class.

use std::cell::RefCell;
use std::rc::Rc;

use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::xlsx::{Cell, Workbook};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{Value, VarType, Variable};

use super::builtins::{real, sc, ss, strm};
use super::eval::{conform, err, Ev, View, R};
use super::parser::Expr;
use super::value::{Mat, Obj, Val};

fn hosterr(e: String) -> super::eval::MErr {
    err(crate::interpreter::rc_of(&e), crate::interpreter::err_message(&e))
}

/// Variable indices (0-based) from a real index vector or names.
fn var_list(ev: &Ev, v: &Val) -> R<Vec<usize>> {
    let ds = &ev.me.engine.dataset;
    match v {
        Val::R(m) => {
            if m.is_scalar() && is_missing(m.d[0]) {
                return Ok((0..ds.vars.len()).collect());
            }
            m.d.iter()
                .map(|x| {
                    let k = *x as usize;
                    if k < 1 || k > ds.vars.len() {
                        Err(err(3300, "argument out of range"))
                    } else {
                        Ok(k - 1)
                    }
                })
                .collect()
        }
        Val::S(m) => {
            let names = ferrostat_cmd::varlist::expand(ds, &m.d.join(" ")).map_err(|e| err(3500, crate::interpreter::err_message(&e)))?;
            Ok(names.iter().filter_map(|n| ds.var_index.get(n).copied()).collect())
        }
        _ => Err(err(3253, "nonreal found where real required")),
    }
}

/// Observation indices (0-based): `.` for all, a vector, or a two-column range matrix.
fn obs_list(ev: &Ev, v: &Val) -> R<Vec<usize>> {
    let n = ev.me.engine.dataset.n_obs;
    let m = real(v)?;
    if m.is_scalar() && is_missing(m.d[0]) || m.d.is_empty() {
        return Ok((0..n).collect());
    }
    let chk = |x: f64| -> R<usize> {
        if x < 1.0 || x as usize > n {
            Err(err(3300, "argument out of range"))
        } else {
            Ok(x as usize - 1)
        }
    };
    if m.c == 2 && m.r > 1 {
        let mut out = Vec::new();
        for i in 0..m.r {
            let a = chk(*m.at(i, 0))?;
            let b = if is_missing(*m.at(i, 1)) { n - 1 } else { chk(*m.at(i, 1))? };
            out.extend(a..=b);
        }
        return Ok(out);
    }
    m.d.iter().map(|x| chk(*x)).collect()
}

/// Apply a selection variable: a name keeps observations where it is not zero,
/// a real zero drops observations with missing values in the selected variables.
fn select_obs(ev: &Ev, obs: Vec<usize>, vars: &[usize], sel: Option<&Val>) -> R<Vec<usize>> {
    let ds = &ev.me.engine.dataset;
    match sel {
        None => Ok(obs),
        Some(Val::S(m)) if m.is_scalar() && !m.d[0].is_empty() => {
            let name = ferrostat_cmd::varlist::resolve_name(ds, m.d[0].trim()).map_err(|e| err(3500, crate::interpreter::err_message(&e)))?;
            let var = ds.get_var(&name).ok_or_else(|| err(3500, format!("variable {name} not found")))?;
            Ok(obs.into_iter().filter(|&i| var.num(i) != 0.0).collect())
        }
        Some(Val::R(m)) if m.is_scalar() && m.d[0] == 0.0 => Ok(obs
            .into_iter()
            .filter(|&i| vars.iter().all(|&j| {
                let v = &ds.vars[j];
                if v.var_type.is_string() { true } else { !is_missing(v.num(i)) }
            }))
            .collect()),
        Some(Val::R(m)) if m.is_scalar() && m.d[0] != 0.0 && !is_missing(m.d[0]) => {
            // a variable index
            let k = m.d[0] as usize;
            let var = ds.vars.get(k.wrapping_sub(1)).ok_or_else(|| err(3300, "argument out of range"))?;
            Ok(obs.into_iter().filter(|&i| var.num(i) != 0.0).collect())
        }
        _ => Ok(obs),
    }
}

fn fetch(ev: &Ev, obs: &[usize], vars: &[usize], string: bool) -> Val {
    let ds = &ev.me.engine.dataset;
    if string {
        let mut d = Vec::with_capacity(obs.len() * vars.len());
        for &i in obs {
            for &j in vars {
                let v = &ds.vars[j];
                d.push(match v.data.get(i) {
                    Some(Value::Str(s)) => s.clone(),
                    Some(Value::StrL(a, b)) => ds.resolve_strl(*a, *b).unwrap_or("").to_string(),
                    _ => String::new(),
                });
            }
        }
        Val::S(Mat::new(obs.len(), vars.len(), d))
    } else {
        let mut d = Vec::with_capacity(obs.len() * vars.len());
        for &i in obs {
            for &j in vars {
                let v = &ds.vars[j];
                d.push(if v.var_type.is_string() { SYSMISS } else { v.num(i) });
            }
        }
        Val::R(Mat::new(obs.len(), vars.len(), d))
    }
}

fn data_args(ev: &Ev, a: &[Val]) -> R<(Vec<usize>, Vec<usize>)> {
    if a.len() < 2 {
        return Err(err(3001, "too few arguments"));
    }
    let vars = var_list(ev, &a[1])?;
    let obs = obs_list(ev, &a[0])?;
    let obs = select_obs(ev, obs, &vars, a.get(2))?;
    Ok((obs, vars))
}

pub fn store_num(ev: &mut Ev, obs: usize, var: &str, x: f64) -> R<()> {
    let ds = &mut ev.me.engine.dataset;
    let v = ds.get_var_mut(var).ok_or_else(|| err(3500, format!("variable {var} not found")))?;
    if v.var_type.is_string() {
        return Err(err(3250, "type mismatch"));
    }
    let ty = v.var_type;
    v.set(obs, ferrostat_cmd::util::typed_value(ty, x));
    ev.me.engine.store.data_changed = true;
    Ok(())
}

pub fn store_str(ev: &mut Ev, obs: usize, var: &str, s: &str) -> R<()> {
    let ds = &mut ev.me.engine.dataset;
    let v = ds.get_var_mut(var).ok_or_else(|| err(3500, format!("variable {var} not found")))?;
    match v.var_type {
        VarType::Str(w) => {
            if s.len() > w {
                v.var_type = ferrostat_cmd::util::str_type_for(s.len());
            }
        }
        VarType::StrL => {}
        _ => return Err(err(3250, "type mismatch")),
    }
    v.set(obs, Value::Str(s.to_string()));
    ev.me.engine.store.data_changed = true;
    Ok(())
}

fn store(ev: &mut Ev, a: &[Val], string: bool) -> R<()> {
    if a.len() < 3 {
        return Err(err(3001, "too few arguments"));
    }
    let x = a.last().unwrap();
    let vars = var_list(ev, &a[1])?;
    let obs = obs_list(ev, &a[0])?;
    let obs = select_obs(ev, obs, &vars, if a.len() == 4 { Some(&a[2]) } else { None })?;
    let (r, c) = x.dims();
    if r != obs.len() || c != vars.len() {
        return Err(conform());
    }
    let names: Vec<String> = vars.iter().map(|&j| ev.me.engine.dataset.vars[j].name.clone()).collect();
    for (k, &i) in obs.iter().enumerate() {
        for (l, n) in names.iter().enumerate() {
            match (x, string) {
                (Val::R(m), false) => store_num(ev, i, n, *m.at(k, l))?,
                (Val::S(m), true) => store_str(ev, i, n, m.at(k, l))?,
                _ => return Err(err(3250, "type mismatch")),
            }
        }
    }
    Ok(())
}

fn to_host_matrix(m: &Mat<f64>) -> Matrix {
    let mut x = Matrix::zeros(m.r, m.c);
    x.data = m.d.clone();
    x
}

fn stored_name(name: &str) -> Option<(char, String)> {
    let n = name.trim();
    for p in ['r', 'e', 's'] {
        if let Some(rest) = n.strip_prefix(&format!("{p}(")).and_then(|x| x.strip_suffix(')')) {
            return Some((p, rest.to_string()));
        }
    }
    None
}

fn results<'s>(st: &'s mut ferrostat_cmd::store::Store, p: char) -> &'s mut ferrostat_cmd::store::Results {
    match p {
        'r' => &mut st.r,
        'e' => &mut st.e,
        _ => &mut st.s,
    }
}

fn get_matrix(ev: &mut Ev, name: &str) -> Option<Matrix> {
    let st = &mut ev.me.engine.store;
    match stored_name(name) {
        Some((p, n)) => results(st, p).matrices.get(&n).cloned(),
        None => st.matrices.get(name.trim()).cloned(),
    }
}

fn stripe(names: &[String]) -> Val {
    let mut d = Vec::new();
    for n in names {
        let (eq, nm) = match n.rsplit_once(':') {
            Some((e, x)) => (e.to_string(), x.to_string()),
            None => (String::new(), n.clone()),
        };
        d.push(eq);
        d.push(nm);
    }
    Val::S(Mat::new(names.len(), 2, d))
}

fn unstripe(v: &Val) -> R<Vec<String>> {
    let m = strm(v)?;
    if m.c != 2 {
        return Err(conform());
    }
    Ok((0..m.r).map(|i| if m.at(i, 0).is_empty() { m.at(i, 1).clone() } else { format!("{}:{}", m.at(i, 0), m.at(i, 1)) }).collect())
}

/// Host-interface functions; `None` when the name is not one of them.
pub fn call(ev: &mut Ev, name: &str, a: &[Val], omitted: &[bool]) -> R<Option<Option<Val>>> {
    let _ = omitted;
    let n = a.len();
    let out: Option<Val> = match name {
        "st_nobs" => Some(Val::num(ev.me.engine.dataset.n_obs as f64)),
        "st_nvar" => Some(Val::num(ev.me.engine.dataset.vars.len() as f64)),
        "st_data" | "st_sdata" => {
            let (obs, vars) = data_args(ev, a)?;
            Some(fetch(ev, &obs, &vars, name == "st_sdata"))
        }
        "st_store" | "st_sstore" => {
            store(ev, a, name == "st_sstore")?;
            None
        }
        "st_addobs" => {
            let k = sc(&a[0])?;
            if k < 0.0 || is_missing(k) {
                return Err(err(3300, "argument out of range"));
            }
            for _ in 0..k as usize {
                ev.me.engine.dataset.add_observation();
            }
            ev.me.engine.store.data_changed = true;
            None
        }
        "st_addvar" => {
            if n < 2 {
                return Err(err(3001, "too few arguments"));
            }
            let types = strm(&a[0])?.d.clone();
            let names = strm(&a[1])?.d.clone();
            let mut idx = Vec::new();
            for (k, nm) in names.iter().enumerate() {
                let t = types.get(k).or(types.first()).cloned().unwrap_or_else(|| "float".into());
                let ty = VarType::from_type_str(t.trim()).ok_or_else(|| err(3300, format!("invalid type {t}")))?;
                ferrostat_cmd::varlist::check_name(nm).map_err(|e| err(3300, crate::interpreter::err_message(&e)))?;
                let ds = &mut ev.me.engine.dataset;
                if ds.var_index.contains_key(nm.as_str()) {
                    return Err(err(110, format!("variable {nm} already defined")));
                }
                let mut v = Variable::new(nm.clone(), ty);
                if ty.is_string() {
                    v.format = format!("%{}s", match ty {
                        VarType::Str(w) => w,
                        _ => 9,
                    });
                    while v.len() < ds.n_obs {
                        v.push(Value::Str(String::new()));
                    }
                } else if ty == VarType::Double {
                    v.format = "%10.0g".into();
                } else if matches!(ty, VarType::Byte | VarType::Int) {
                    v.format = "%8.0g".into();
                } else if ty == VarType::Long {
                    v.format = "%12.0g".into();
                }
                ds.add_variable(v).map_err(|e| err(110, e))?;
                idx.push(ds.vars.len() as f64);
            }
            ev.me.engine.store.data_changed = true;
            Some(Val::R(Mat::row(idx)))
        }
        "st_dropvar" | "st_keepvar" => {
            let vars = var_list(ev, &a[0])?;
            let ds = &mut ev.me.engine.dataset;
            let names: Vec<String> = ds.vars.iter().enumerate().filter(|(j, _)| vars.contains(j) == (name == "st_dropvar")).map(|(_, v)| v.name.clone()).collect();
            for nm in names {
                let _ = ds.drop_variable(&nm);
            }
            None
        }
        "st_dropobsif" | "st_keepobsif" => {
            let m = real(&a[0])?;
            let ds = &mut ev.me.engine.dataset;
            if m.d.len() != ds.n_obs {
                return Err(conform());
            }
            let mask: Vec<bool> = m.d.iter().map(|x| (*x != 0.0) == (name == "st_keepobsif")).collect();
            ds.keep_if(&mask).map_err(|e| err(3300, e))?;
            None
        }
        "st_varindex" => {
            let ds = &ev.me.engine.dataset;
            let m = strm(&a[0])?;
            let names: Vec<String> = m.d.iter().flat_map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>()).collect();
            let idx: Vec<f64> = names
                .iter()
                .map(|nm| {
                    let r = ferrostat_cmd::varlist::resolve_name(ds, nm).ok();
                    r.and_then(|r| ds.var_index.get(&r).map(|i| (*i + 1) as f64)).unwrap_or(SYSMISS)
                })
                .collect();
            Some(Val::R(Mat::row(idx)))
        }
        "st_varname" | "st_vartype" | "st_varformat" | "st_varlabel" | "st_varvaluelabel" | "st_isnumvar" | "st_isstrvar" => {
            let vars = var_list(ev, &a[0])?;
            if n == 2 && matches!(name, "st_varlabel" | "st_varformat" | "st_varvaluelabel") {
                let s = ss(&a[1])?;
                let ds = &mut ev.me.engine.dataset;
                for &j in &vars {
                    match name {
                        "st_varlabel" => ds.vars[j].label = s.clone(),
                        "st_varformat" => ds.vars[j].format = s.clone(),
                        _ => ds.vars[j].value_label_name = s.clone(),
                    }
                }
                None
            } else {
                let ds = &ev.me.engine.dataset;
                let vs: Vec<&Variable> = vars.iter().map(|&j| &ds.vars[j]).collect();
                match name {
                    "st_isnumvar" => Some(Val::R(Mat::row(vs.iter().map(|v| v.var_type.is_numeric() as i32 as f64).collect()))),
                    "st_isstrvar" => Some(Val::R(Mat::row(vs.iter().map(|v| v.var_type.is_string() as i32 as f64).collect()))),
                    _ => Some(Val::S(Mat::row(
                        vs.iter()
                            .map(|v| match name {
                                "st_varname" => v.name.clone(),
                                "st_vartype" => v.var_type.type_str(),
                                "st_varformat" => v.format.clone(),
                                "st_varlabel" => v.label.clone(),
                                _ => v.value_label_name.clone(),
                            })
                            .collect(),
                    ))),
                }
            }
        }
        "st_local" | "st_global" => {
            let nm = ss(&a[0])?;
            let st = &mut ev.me.engine.store;
            if n >= 2 {
                let v = ss(&a[1])?;
                if name == "st_local" {
                    st.set_local(&nm, v);
                } else if let Some((p, k)) = stored_name(&nm) {
                    results(st, p).set_str(&k, v);
                } else {
                    st.globals.insert(nm, v);
                }
                None
            } else {
                let v = if name == "st_local" {
                    st.local(&nm).cloned()
                } else if let Some((p, k)) = stored_name(&nm) {
                    results(st, p).macros.get(&k).cloned()
                } else {
                    st.globals.get(&nm).cloned()
                };
                Some(Val::str(v.unwrap_or_default()))
            }
        }
        "st_numscalar" | "st_strscalar" => {
            let nm = ss(&a[0])?;
            let st = &mut ev.me.engine.store;
            if n >= 2 {
                match stored_name(&nm) {
                    Some((p, k)) => {
                        if name == "st_numscalar" {
                            results(st, p).set(&k, sc(&a[1])?);
                        } else {
                            results(st, p).set_str(&k, ss(&a[1])?);
                        }
                    }
                    None => {
                        if name == "st_numscalar" {
                            st.str_scalars.remove(&nm);
                            st.scalars.insert(nm, sc(&a[1])?);
                        } else {
                            st.scalars.remove(&nm);
                            st.str_scalars.insert(nm, ss(&a[1])?);
                        }
                    }
                }
                None
            } else if name == "st_numscalar" {
                let v = match stored_name(&nm) {
                    Some((p, k)) => results(st, p).get(&k),
                    None => st.scalars.get(&nm).copied(),
                };
                Some(match v {
                    Some(x) => Val::num(x),
                    None => Val::empty(),
                })
            } else {
                let v = match stored_name(&nm) {
                    Some((p, k)) => results(st, p).macros.get(&k).cloned(),
                    None => st.str_scalars.get(&nm).cloned(),
                };
                Some(match v {
                    Some(x) => Val::str(x),
                    None => Val::S(Mat::default()),
                })
            }
        }
        "st_matrix" => {
            let nm = ss(&a[0])?;
            if n >= 2 {
                let m = to_host_matrix(real(&a[1])?);
                let st = &mut ev.me.engine.store;
                match stored_name(&nm) {
                    Some((p, k)) => results(st, p).set_mat(&k, m),
                    None => {
                        st.matrices.insert(nm, m);
                    }
                }
                None
            } else {
                Some(match get_matrix(ev, &nm) {
                    Some(m) => Val::R(Mat::new(m.rows, m.cols, m.data.clone())),
                    None => Val::empty(),
                })
            }
        }
        "st_matrixrowstripe" | "st_matrixcolstripe" => {
            let nm = ss(&a[0])?;
            let row = name == "st_matrixrowstripe";
            if n >= 2 {
                let names = unstripe(&a[1])?;
                let st = &mut ev.me.engine.store;
                let m = match stored_name(&nm) {
                    Some((p, k)) => results(st, p).matrices.get_mut(&k),
                    None => st.matrices.get_mut(&nm),
                };
                let m = m.ok_or_else(|| err(3499, format!("{nm} not found")))?;
                if names.len() != if row { m.rows } else { m.cols } {
                    return Err(conform());
                }
                if row {
                    m.rownames = names;
                } else {
                    m.colnames = names;
                }
                None
            } else {
                Some(match get_matrix(ev, &nm) {
                    Some(m) => stripe(if row { &m.rownames } else { &m.colnames }),
                    None => Val::S(Mat::default()),
                })
            }
        }
        "st_rclear" => {
            ev.me.engine.store.r = Default::default();
            None
        }
        "st_eclear" => {
            ev.me.engine.store.e = Default::default();
            None
        }
        "st_tempname" | "st_tempfilename" => {
            let k = if n >= 1 { sc(&a[0])? as usize } else { 1 };
            let mut v = Vec::new();
            for _ in 0..k.max(1) {
                ev.me.temp_counter += 1;
                let base = format!("__{:06}", ev.me.temp_counter);
                v.push(if name == "st_tempfilename" { ferrostat_core::vfs::temp_path(&format!("St{base}.tmp")) } else { base });
            }
            Some(Val::S(Mat::row(v)))
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Functions whose first argument is a variable that receives the result.
pub fn lvalue_call(ev: &mut Ev, name: &str, args: &[Expr]) -> R<Option<Val>> {
    let target = match args.first() {
        Some(Expr::Var(n)) => n.clone(),
        Some(e @ Expr::Assign(l, _, _)) => {
            ev.eval(e)?;
            match &**l {
                Expr::Var(n) => n.clone(),
                _ => return Err(err(3000, "invalid lval")),
            }
        }
        _ => return Err(err(3000, format!("{name}(): first argument must be a variable"))),
    };
    let mut rest = Vec::new();
    for a in &args[1..] {
        rest.push(ev.eval(a)?);
    }
    let cur = || -> R<Val> { ev_lookup(ev, &target) };
    match name {
        "st_view" | "st_sview" | "st_subview" => {
            if name == "st_subview" {
                let src = rest.first().cloned().ok_or_else(|| err(3001, "too few arguments"))?;
                let i = rest.get(1).cloned().unwrap_or(Val::num(SYSMISS));
                let j = rest.get(2).cloned().unwrap_or(Val::num(SYSMISS));
                let ix = |v: &Val| -> R<Option<Vec<usize>>> {
                    let m = real(v)?;
                    if m.is_scalar() && is_missing(m.d[0]) { Ok(None) } else { Ok(Some(m.d.iter().map(|x| *x as usize - 1).collect())) }
                };
                let v = super::ops::subscript(&src, &[ix(&i)?, ix(&j)?])?;
                ev.set_var(&target, v);
                return Ok(None);
            }
            let (obs, vars) = data_args(ev, &rest)?;
            let v = fetch(ev, &obs, &vars, name == "st_sview");
            let names: Vec<String> = vars.iter().map(|&j| ev.me.engine.dataset.vars[j].name.clone()).collect();
            ev.set_var(&target, v);
            let slot = ev.lookup(&target).unwrap();
            ev.set_view(&slot, View { obs, vars: names });
            Ok(None)
        }
        "swap" => {
            let Some(Expr::Var(other)) = args.get(1) else { return Err(err(3000, "swap(): variables required")) };
            let a = cur()?;
            let b = ev_lookup(ev, other)?;
            ev.set_var(&target, b);
            ev.set_var(other, a);
            Ok(None)
        }
        "_sort" => {
            let keys = rest.first().cloned().ok_or_else(|| err(3001, "too few arguments"))?;
            let v = super::builtins::sort_val(&cur()?, &keys)?;
            ev.set_var(&target, v);
            Ok(None)
        }
        "_transpose" => {
            let v = cur()?.transpose();
            ev.set_var(&target, v);
            Ok(None)
        }
        _ => {
            // `_f(X, ...)` replaces X with `f(X, ...)`
            let base = name.trim_start_matches('_');
            let mut all = vec![cur()?];
            all.extend(rest);
            let omitted = vec![false; all.len()];
            if let Some(v) = super::builtins::call(ev, base, all, &omitted)? {
                ev.set_var(&target, v);
            }
            Ok(None)
        }
    }
}

fn ev_lookup(ev: &Ev, n: &str) -> R<Val> {
    ev.lookup(n).map(|s| s.borrow().clone()).ok_or_else(|| err(3499, format!("{n} not found")))
}

// ---- xl() ----

fn find_file(path: &str) -> String {
    let vfs = ferrostat_core::vfs::exists;
    if vfs(path) {
        return path.to_string();
    }
    // tolerate differences in letter case of the file name
    let (dir, file) = match path.rfind(['/', '\\']) {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (".", path),
    };
    if let Ok(list) = ferrostat_core::vfs::list(if dir.is_empty() { "/" } else { dir }) {
        for (n, is_dir) in list {
            let base = n.rsplit(['/', '\\']).next().unwrap_or(&n).to_string();
            if !is_dir && base.eq_ignore_ascii_case(file) {
                return if dir == "." { base } else { format!("{dir}/{base}") };
            }
        }
    }
    path.to_string()
}

fn with_ext(p: &str) -> String {
    let file = p.rsplit(['/', '\\']).next().unwrap_or(p);
    if file.contains('.') { p.to_string() } else { format!("{p}.xlsx") }
}

/// Row or column spec: scalar or (first, last); columns may be letters.
fn span(v: &Val, col: bool) -> R<(u32, u32)> {
    let conv = |s: &str| -> R<u32> {
        if col {
            ferrostat_cmd::xlsx::parse_ref(&format!("{}1", s.trim())).map(|x| x.1).ok_or_else(|| err(3300, "invalid column"))
        } else {
            s.trim().parse().map_err(|_| err(3300, "invalid row"))
        }
    };
    let list: Vec<u32> = match v {
        Val::R(m) => m.d.iter().map(|x| *x as u32).collect(),
        Val::S(m) => m.d.iter().map(|s| conv(s)).collect::<R<_>>()?,
        _ => return Err(err(3250, "type mismatch")),
    };
    match list.len() {
        1 => Ok((list[0], list[0])),
        2 => Ok((list[0], list[1])),
        _ => Err(conform()),
    }
}

pub fn method(ev: &mut Ev, obj: &Rc<RefCell<Obj>>, name: &str, a: Vec<Val>) -> R<Option<Val>> {
    let mut o = obj.borrow_mut();
    let Obj::Xl(x) = &mut *o;
    let need_book = |x: &super::value::XlBook| -> R<()> {
        if x.book.is_none() {
            return Err(err(3598, "no workbook loaded"));
        }
        Ok(())
    };
    match name {
        "load_book" => {
            let p = find_file(&with_ext(&ss(a.first().ok_or_else(|| err(3001, "too few arguments"))?)?));
            let data = ferrostat_core::vfs::read(&p).map_err(|_| err(601, format!("file {p} could not be loaded")))?;
            let book = Workbook::read(&data).map_err(hosterr)?;
            x.path = p;
            x.book = Some(book);
            x.sheet = 0;
            Ok(None)
        }
        "create_book" => {
            let p = with_ext(&ss(a.first().ok_or_else(|| err(3001, "too few arguments"))?)?);
            let sheet = match a.get(1) {
                Some(v) => ss(v)?,
                None => "Sheet1".into(),
            };
            let mut book = Workbook::default();
            book.sheet_mut(&sheet);
            x.path = p;
            x.book = Some(book);
            x.sheet = 0;
            x.dirty = true;
            Ok(None)
        }
        "get_sheets" => {
            need_book(x)?;
            let names: Vec<String> = x.book.as_ref().unwrap().sheets.iter().map(|s| s.0.clone()).collect();
            Ok(Some(Val::S(Mat::col(names))))
        }
        "set_sheet" | "add_sheet" => {
            need_book(x)?;
            let s = ss(a.first().ok_or_else(|| err(3001, "too few arguments"))?)?;
            let book = x.book.as_mut().unwrap();
            match book.sheets.iter().position(|(n, _)| *n == s) {
                Some(i) => x.sheet = i,
                None if name == "add_sheet" => {
                    book.sheet_mut(&s);
                    x.sheet = book.sheets.len() - 1;
                    x.dirty = true;
                }
                None => return Err(err(3598, format!("worksheet {s} not found"))),
            }
            Ok(None)
        }
        "get_string" | "get_number" | "get_cell_type" => {
            need_book(x)?;
            if a.len() < 2 {
                return Err(err(3001, "too few arguments"));
            }
            let (r0, r1) = span(&a[0], false)?;
            let (c0, c1) = span(&a[1], true)?;
            let cells = &x.book.as_ref().unwrap().sheets.get(x.sheet).ok_or_else(|| err(3598, "no worksheet"))?.1;
            let nr = (r1.max(r0) - r0 + 1) as usize;
            let nc = (c1.max(c0) - c0 + 1) as usize;
            let mut nums = Vec::new();
            let mut strs = Vec::new();
            for r in r0..=r1.max(r0) {
                for c in c0..=c1.max(c0) {
                    let cell = cells.get(&(r, c));
                    match name {
                        "get_number" => nums.push(match cell {
                            Some(Cell::Num(v)) | Some(Cell::Date(v)) => *v,
                            Some(Cell::Bool(b)) => *b as i32 as f64,
                            _ => SYSMISS,
                        }),
                        "get_string" => strs.push(match cell {
                            Some(Cell::Str(s)) => s.clone(),
                            Some(Cell::Num(v)) | Some(Cell::Date(v)) => ferrostat_cmd::format::fmt_g(*v, 16),
                            Some(Cell::Bool(b)) => if *b { "TRUE".into() } else { "FALSE".into() },
                            None => String::new(),
                        }),
                        _ => strs.push(
                            match cell {
                                Some(Cell::Str(_)) => "string",
                                Some(Cell::Num(_)) | Some(Cell::Bool(_)) => "numeric",
                                Some(Cell::Date(_)) => "date",
                                None => "blank",
                            }
                            .into(),
                        ),
                    }
                }
            }
            Ok(Some(if name == "get_number" { Val::R(Mat::new(nr, nc, nums)) } else { Val::S(Mat::new(nr, nc, strs)) }))
        }
        "put_string" | "put_number" => {
            need_book(x)?;
            if a.len() < 3 {
                return Err(err(3001, "too few arguments"));
            }
            let (r0, _) = span(&a[0], false)?;
            let (c0, _) = span(&a[1], true)?;
            let sheet = x.sheet;
            let cells = &mut x.book.as_mut().unwrap().sheets.get_mut(sheet).ok_or_else(|| err(3598, "no worksheet"))?.1;
            let (r, c) = a[2].dims();
            for i in 0..r {
                for j in 0..c {
                    let key = (r0 + i as u32, c0 + j as u32);
                    match &a[2] {
                        Val::S(m) => {
                            cells.insert(key, Cell::Str(m.at(i, j).clone()));
                        }
                        Val::R(m) => {
                            let v = *m.at(i, j);
                            if is_missing(v) {
                                cells.remove(&key);
                            } else {
                                cells.insert(key, Cell::Num(v));
                            }
                        }
                        _ => return Err(err(3250, "type mismatch")),
                    }
                }
            }
            x.dirty = true;
            Ok(None)
        }
        "close_book" => {
            if x.dirty {
                if let Some(b) = &x.book {
                    ferrostat_core::vfs::write(&x.path, &b.to_bytes()).map_err(|e| err(603, format!("file {} could not be written: {e}", x.path)))?;
                }
            }
            x.book = None;
            x.dirty = false;
            Ok(None)
        }
        "query" | "set_mode" | "set_missing" | "set_keep_cell_format" | "set_error_mode" | "clear_book" | "set_column_width" => {
            let _ = ev;
            Ok(None)
        }
        _ => Err(err(3000, format!("xl::{name}() not supported"))),
    }
}
