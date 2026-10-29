//! Further `matrix` subcommands: `eigenvalues`, `svd`, `score`,
//! `rowjoinbyname`/`coljoinbyname`, and the `get()` matrix function.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Variable, VarType};
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options, split_top_level_words};
use ferrostat_parser::Command;

use crate::eval::{rc_err, Env};
use crate::matcmd::eval_matrix;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::varlist;

/// Handle a subcommand of `matrix` not covered by the main parser.
pub fn subcommand(ds: &mut DataSet, st: &mut Store, sub: &str, rest: &str, _cmd: &Command) -> Option<Result<String, String>> {
    Some(match sub {
        "eigenvalues" | "eigenvalue" | "eigenval" | "eigen" | "eig" => eigenvalues(ds, st, rest),
        "svd" => svd(ds, st, rest),
        "score" | "scor" | "sco" => score(ds, st, rest),
        "rowjoinbyname" | "coljoinbyname" => joinbyname(ds, st, rest, sub.starts_with("row")),
        _ => return None,
    })
}

/// `get(_b)`, `get(_se)`, `get(VCE)`, `get(Cns)`, `get(Rr)` in matrix expressions.
pub fn get(st: &Store, what: &str) -> Result<Matrix, String> {
    let what = what.trim();
    let e = &st.e;
    let need = |k: &str| -> Result<Matrix, String> { e.matrices.get(k).cloned().ok_or_else(|| rc_err(301, "last estimates not found")) };
    match what {
        "_b" => need("b"),
        "VCE" => need("V"),
        "_se" => {
            let b = need("b")?;
            let v = need("V")?;
            let mut m = b.clone();
            for j in 0..m.cols {
                let d = if j < v.rows { v[(j, j)] } else { SYSMISS };
                m[(0, j)] = if d.is_finite() && d >= 0.0 { d.sqrt() } else { SYSMISS };
            }
            Ok(m)
        }
        "Cns" => e.matrices.get("Cns").cloned().ok_or_else(|| rc_err(111, "matrix get(Cns) not found")),
        "Rr" => st.r.matrices.get("Rr").or_else(|| st.matrices.get("__Rr")).cloned().ok_or_else(|| rc_err(111, "matrix get(Rr) not found")),
        other => Err(rc_err(198, format!("get({other}) invalid"))),
    }
}

fn lhs_rhs(rest: &str) -> Result<(Vec<String>, String), String> {
    let eq = find_top_level(rest, '=').ok_or_else(|| rc_err(198, "= expected"))?;
    Ok((rest[..eq].split_whitespace().map(String::from).collect(), rest[eq + 1..].trim().to_string()))
}

fn numbered(prefix: &str, n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("{prefix}{i}")).collect()
}

/// `matrix eigenvalues r c = A`: real and imaginary parts of the eigenvalues
/// of a square (possibly nonsymmetric) matrix, in order of decreasing modulus.
fn eigenvalues(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (names, rhs) = lhs_rhs(rest)?;
    if names.len() != 2 {
        return Err(rc_err(198, "matrix eigenvalues r c = A"));
    }
    let a = eval_matrix(ds, st, &rhs)?;
    if a.rows != a.cols {
        return Err(rc_err(503, "conformability error"));
    }
    if a.has_missing() {
        return Err(rc_err(504, "matrix has missing values"));
    }
    let mut ev = crate::tsla::eig_general(&a);
    ev.sort_by(|x, y| (y.0.hypot(y.1)).total_cmp(&x.0.hypot(x.1)).then(y.0.total_cmp(&x.0)).then(y.1.total_cmp(&x.1)));
    let cn = numbered("c", ev.len());
    let re: Vec<f64> = ev.iter().map(|v| clean(v.0)).collect();
    let im: Vec<f64> = ev.iter().map(|v| clean(v.1)).collect();
    for n in &names {
        varlist::check_name(n)?;
    }
    st.matrices.insert(names[0].clone(), Matrix::row_vector(&re).with_names(vec!["real".into()], cn.clone()));
    st.matrices.insert(names[1].clone(), Matrix::row_vector(&im).with_names(vec!["complex".into()], cn));
    Ok(String::new())
}

fn clean(x: f64) -> f64 {
    if x.abs() < 1e-14 { 0.0 } else { x }
}

/// Singular value decomposition A = U diag(w) V' by one-sided Jacobi
/// rotations; singular values in decreasing order.
pub fn svd_decompose(a: &Matrix) -> (Matrix, Vec<f64>, Matrix) {
    let (m, n) = (a.rows, a.cols);
    let mut u = a.clone();
    let mut v = Matrix::identity(n);
    for _sweep in 0..80 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    alpha += u[(i, p)] * u[(i, p)];
                    beta += u[(i, q)] * u[(i, q)];
                    gamma += u[(i, p)] * u[(i, q)];
                }
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (u[(i, p)], u[(i, q)]);
                    u[(i, p)] = c * x - s * y;
                    u[(i, q)] = s * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[(i, p)], v[(i, q)]);
                    v[(i, p)] = c * x - s * y;
                    v[(i, q)] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut w: Vec<f64> = (0..n).map(|j| (0..m).map(|i| u[(i, j)] * u[(i, j)]).sum::<f64>().sqrt()).collect();
    for j in 0..n {
        if w[j] > 1e-300 {
            for i in 0..m {
                u[(i, j)] /= w[j];
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w[j].total_cmp(&w[i]));
    let (mut us, mut vs) = (Matrix::zeros(m, n), Matrix::zeros(n, n));
    for (nj, &oj) in idx.iter().enumerate() {
        for i in 0..m {
            us[(i, nj)] = u[(i, oj)];
        }
        for i in 0..n {
            vs[(i, nj)] = v[(i, oj)];
        }
    }
    w = idx.iter().map(|&j| w[j]).collect();
    // sign convention: the largest-magnitude element of each V column is positive
    for j in 0..n {
        let mut best = 0.0f64;
        for i in 0..n {
            if vs[(i, j)].abs() > best.abs() + 1e-12 {
                best = vs[(i, j)];
            }
        }
        if best < 0.0 {
            for i in 0..n {
                vs[(i, j)] = -vs[(i, j)];
            }
            for i in 0..m {
                us[(i, j)] = -us[(i, j)];
            }
        }
    }
    (us, w, vs)
}

/// `matrix svd U w V = A`
fn svd(ds: &DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (names, rhs) = lhs_rhs(rest)?;
    if names.len() != 3 {
        return Err(rc_err(198, "matrix svd U w V = A"));
    }
    let a = eval_matrix(ds, st, &rhs)?;
    if a.has_missing() {
        return Err(rc_err(504, "matrix has missing values"));
    }
    if a.rows < a.cols {
        return Err(rc_err(503, "conformability error"));
    }
    for n in &names {
        varlist::check_name(n)?;
    }
    let (u, w, v) = svd_decompose(&a);
    st.matrices.insert(names[0].clone(), u.with_names(a.rownames.clone(), a.colnames.clone()));
    st.matrices.insert(names[1].clone(), Matrix::row_vector(&w).with_names(vec!["r1".into()], a.colnames.clone()));
    st.matrices.insert(names[2].clone(), v.with_names(a.colnames.clone(), a.colnames.clone()));
    Ok(String::new())
}

/// Values of a coefficient name such as `x`, `L.x`, `2.g`, `1b.g#c.x` over all observations.
pub fn column_values(ds: &DataSet, st: &Store, name: &str, forcezero: bool) -> Result<Vec<f64>, String> {
    let n = ds.n_obs;
    let mut acc = vec![1.0; n];
    for part in name.split('#') {
        let part = part.strip_prefix("c.").unwrap_or(part);
        let part = part.strip_prefix("co.").unwrap_or(part);
        let part = part.strip_prefix("o.").unwrap_or(part);
        // factor level: digits followed by optional b/o/n flags, a dot and the variable
        let level = part.split_once('.').and_then(|(l, v)| {
            let digits: String = l.chars().take_while(|c| c.is_ascii_digit()).collect();
            let flags = &l[digits.len()..];
            (!digits.is_empty() && flags.chars().all(|c| matches!(c, 'b' | 'o' | 'n'))).then(|| (digits.parse::<f64>().unwrap_or(0.0), v.to_string()))
        });
        let vals: Vec<f64> = match level {
            Some((lv, var)) => match ds.get_var(&var) {
                Some(v) => (0..n).map(|i| {
                    let x = v.num(i);
                    if is_missing(x) { SYSMISS } else if x == lv { 1.0 } else { 0.0 }
                }).collect(),
                None if forcezero => vec![0.0; n],
                None => return Err(rc_err(111, format!("variable {var} not found"))),
            },
            None => {
                if let Some(v) = ds.get_var(part) {
                    if v.var_type.is_string() {
                        return Err(rc_err(109, "type mismatch"));
                    }
                    (0..n).map(|i| v.num(i)).collect()
                } else {
                    match parse_expression(part).ok().map(|e| {
                        let env = Env::with_store(ds, st);
                        (0..n).map(|i| env.eval(&e, i).and_then(|x| x.num())).collect::<Result<Vec<f64>, String>>()
                    }) {
                        Some(Ok(v)) => v,
                        _ if forcezero => vec![0.0; n],
                        _ => return Err(rc_err(111, format!("variable {part} not found"))),
                    }
                }
            }
        };
        for i in 0..n {
            acc[i] = if is_missing(acc[i]) || is_missing(vals[i]) { SYSMISS } else { acc[i] * vals[i] };
        }
    }
    Ok(acc)
}

/// `matrix score [type] newvar = b [if] [in] [, equation(#n|name) replace forcezero missval(#)]`
fn score(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (lhs, rhs) = lhs_rhs(rest)?;
    let (ty, newvar) = match lhs.as_slice() {
        [n] => (st.default_type, n.clone()),
        [t, n] => (VarType::from_type_str(t).ok_or_else(|| rc_err(198, format!("{t} invalid type")))?, n.clone()),
        _ => return Err(rc_err(198, "matrix score newvar = b")),
    };
    let c: Command = parse_line(&format!("score {rhs}")).map_err(|e| rc_err(198, e.message))?;
    let bname = c.main.trim().to_string();
    let b = eval_matrix(ds, st, &bname)?;
    if b.rows != 1 {
        return Err(rc_err(503, "conformability error"));
    }
    let replace = c.has_opt("replace", 7);
    let forcezero = c.has_opt("forcezero", 9);
    let missval = c.opt("missval", 7).map(|v| v.trim().parse::<f64>().map_err(|_| rc_err(198, "missval() invalid"))).transpose()?;
    // equations in order of appearance
    let mut eqs: Vec<String> = Vec::new();
    for n in &b.colnames {
        let e = n.rsplit_once(':').map(|x| x.0.to_string()).unwrap_or_default();
        if !eqs.contains(&e) {
            eqs.push(e);
        }
    }
    let want = match c.opt("equation", 2) {
        Some(e) => {
            let e = e.trim();
            if let Some(k) = e.strip_prefix('#') {
                let k: usize = k.trim().parse().map_err(|_| rc_err(198, "equation() invalid"))?;
                eqs.get(k.wrapping_sub(1)).cloned().ok_or_else(|| rc_err(303, format!("equation {e} not found")))?
            } else if eqs.iter().any(|x| x == e) {
                e.to_string()
            } else {
                return Err(rc_err(303, format!("equation {e} not found")));
            }
        }
        None => eqs.first().cloned().unwrap_or_default(),
    };
    if ds.get_var(&newvar).is_some() && !replace {
        return Err(rc_err(110, format!("variable {newvar} already defined")));
    }
    if !replace {
        varlist::check_name(&newvar)?;
    }
    let mask = util::touse(ds, st, &c)?;
    let n = ds.n_obs;
    let mut out = vec![0.0; n];
    for (j, full) in b.colnames.iter().enumerate() {
        let (e, name) = full.rsplit_once(':').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((String::new(), full.clone()));
        if e != want {
            continue;
        }
        let coef = b[(0, j)];
        if name == "_cons" {
            out.iter_mut().for_each(|x| if !is_missing(*x) { *x += coef });
            continue;
        }
        let vals = column_values(ds, st, &name, forcezero)?;
        for i in 0..n {
            let v = match (is_missing(vals[i]), missval) {
                (true, Some(m)) => m,
                (true, None) => {
                    out[i] = SYSMISS;
                    continue;
                }
                (false, _) => vals[i],
            };
            if !is_missing(out[i]) {
                out[i] += coef * v;
            }
        }
    }
    let data: Vec<_> = (0..n).map(|i| util::typed_value(ty, if mask[i] { out[i] } else { SYSMISS })).collect();
    match ds.get_var_mut(&newvar) {
        Some(v) if replace => {
            if v.var_type.is_string() {
                return Err(rc_err(109, "type mismatch"));
            }
            for i in 0..n {
                if mask[i] {
                    v.data[i] = util::typed_value(v.var_type, out[i]);
                }
            }
        }
        _ => {
            let mut v = Variable::new(&newvar, ty);
            v.data = data;
            ds.add_variable(v)?;
        }
    }
    st.data_changed = true;
    Ok(String::new())
}

/// `matrix rowjoinbyname C = A B ... [, missing(#) consolidate ignore(names)]`
/// (and `coljoinbyname`): join matrices aligning the other dimension by name.
fn joinbyname(ds: &DataSet, st: &mut Store, rest: &str, by_row: bool) -> Result<String, String> {
    let (lhs, rhs) = lhs_rhs(rest)?;
    let target = lhs.first().cloned().ok_or_else(|| rc_err(198, "matrix name required"))?;
    let (list, opts) = match find_top_level(&rhs, ',') {
        Some(i) => (rhs[..i].to_string(), split_options(&rhs[i + 1..])),
        None => (rhs.clone(), Vec::new()),
    };
    let fill = opts.iter().find(|(n, _)| n.starts_with("mis")).and_then(|(_, a)| a.as_ref()).map(|a| a.trim().parse::<f64>().unwrap_or(SYSMISS)).unwrap_or(SYSMISS);
    let ignore: Vec<String> = opts.iter().find(|(n, _)| n.starts_with("ign")).and_then(|(_, a)| a.clone()).map(|a| a.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let consolidate = opts.iter().any(|(n, _)| n.starts_with("cons"));
    let mats: Vec<Matrix> = split_top_level_words(&list).iter().map(|n| eval_matrix(ds, st, n)).collect::<Result<_, _>>()?;
    if mats.is_empty() {
        return Err(rc_err(198, "matrices required"));
    }
    // work on row-joins; column-joins are row-joins of transposes
    let mats: Vec<Matrix> = if by_row { mats } else { mats.iter().map(|m| m.transpose()).collect() };
    let mut cols: Vec<String> = Vec::new();
    for m in &mats {
        for c in &m.colnames {
            if !cols.contains(c) && !ignore.contains(c) {
                cols.push(c.clone());
            }
        }
    }
    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    for m in &mats {
        for i in 0..m.rows {
            let mut r = vec![fill; cols.len()];
            for (j, c) in m.colnames.iter().enumerate() {
                if let Some(k) = cols.iter().position(|x| x == c) {
                    r[k] = m[(i, j)];
                }
            }
            let name = m.rownames.get(i).cloned().unwrap_or_else(|| format!("r{}", i + 1));
            if consolidate {
                if let Some(existing) = rows.iter_mut().find(|(n, _)| *n == name) {
                    for (k, v) in r.iter().enumerate() {
                        if existing.1[k] == fill || is_missing(existing.1[k]) {
                            existing.1[k] = *v;
                        }
                    }
                    continue;
                }
            }
            rows.push((name, r));
        }
    }
    let data: Vec<Vec<f64>> = rows.iter().map(|r| r.1.clone()).collect();
    let mut out = if data.is_empty() { Matrix::zeros(0, cols.len()) } else { Matrix::from_rows(&data) };
    out = out.with_names(rows.iter().map(|r| r.0.clone()).collect(), cols);
    if !by_row {
        out = out.transpose();
    }
    varlist::check_name(&target)?;
    st.matrices.insert(target, out);
    Ok(String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs() {
        let a = Matrix::from_rows(&[vec![2.0, 1.0], vec![1.0, 3.0], vec![0.0, 1.0]]);
        let (u, w, v) = svd_decompose(&a);
        for i in 0..3 {
            for j in 0..2 {
                let x: f64 = (0..2).map(|k| u[(i, k)] * w[k] * v[(j, k)]).sum();
                assert!((x - a[(i, j)]).abs() < 1e-10);
            }
        }
        assert!(w[0] >= w[1]);
    }
}
