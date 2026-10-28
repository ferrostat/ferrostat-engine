//! Time-series operators: lag polynomials for combined operators such as
//! `L2D.x` or `DS12.x`, expansion of operator lists like `L(1/3).x`, and
//! helpers shared by the time-series commands (time-ordered samples, gaps,
//! date labels).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;

use crate::eval::{rc_err, Env};
use crate::store::Store;

/// Lag polynomial of an operator string (`L`, `L2`, `LD`, `D2S12`, `L.D`,
/// case-insensitive): pairs (lag, coefficient) such that the operated
/// series is sum(coef * x[t - lag]). `F` gives negative lags.
pub fn lag_poly(op: &str) -> Option<Vec<(i64, f64)>> {
    let chars: Vec<char> = op.chars().filter(|c| *c != '.').collect();
    if chars.is_empty() {
        return None;
    }
    let mut poly: Vec<(i64, f64)> = vec![(0, 1.0)];
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].to_ascii_lowercase();
        if !"lfds".contains(c) {
            return None;
        }
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        let k: i64 = if i > start { chars[start..i].iter().collect::<String>().parse().ok()? } else { 1 };
        let factor: Vec<(i64, f64)> = match c {
            'l' => vec![(k, 1.0)],
            'f' => vec![(-k, 1.0)],
            's' => vec![(0, 1.0), (k, -1.0)],
            _ => {
                // (1 - L)^k
                let mut f = vec![(0i64, 1.0)];
                for _ in 0..k {
                    f = mul(&f, &[(0, 1.0), (1, -1.0)]);
                }
                f
            }
        };
        poly = mul(&poly, &factor);
    }
    Some(poly)
}

fn mul(a: &[(i64, f64)], b: &[(i64, f64)]) -> Vec<(i64, f64)> {
    let mut m: std::collections::BTreeMap<i64, f64> = Default::default();
    for &(la, ca) in a {
        for &(lb, cb) in b {
            *m.entry(la + lb).or_insert(0.0) += ca * cb;
        }
    }
    m.into_iter().filter(|(_, c)| *c != 0.0).collect()
}

/// Canonical operator text: counts of 1 dropped, `L1` -> `L`.
fn op_text(parts: &[(char, i64)]) -> String {
    parts.iter().filter(|(_, k)| *k != 0).map(|(c, k)| if *k == 1 { c.to_string() } else { format!("{c}{k}") }).collect()
}

fn numlist(s: &str) -> Option<Vec<i64>> {
    let mut out = Vec::new();
    for w in s.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()) {
        if let Some((a, b)) = w.split_once('/') {
            let (a, b): (i64, i64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
            if a <= b {
                out.extend(a..=b);
            } else {
                out.extend((b..=a).rev());
            }
        } else {
            out.push(w.parse().ok()?);
        }
    }
    Some(out)
}

/// Expand one token with operator lists: `L(1/2).x` -> `L.x L2.x`,
/// `L(0/1).(x y)` -> `x L.x y L.y`, `LD.x` unchanged.
pub fn expand_token(tok: &str) -> Vec<String> {
    let Some(dot) = top_dot(tok) else { return vec![tok.to_string()] };
    let (op, rest) = (&tok[..dot], &tok[dot + 1..]);
    if !op.contains('(') && !rest.starts_with('(') {
        return vec![tok.to_string()];
    }
    // parse the operator into (letter, choices)
    let oc: Vec<char> = op.chars().collect();
    let mut groups: Vec<(char, Vec<i64>)> = Vec::new();
    let mut i = 0;
    while i < oc.len() {
        let c = oc[i].to_ascii_uppercase();
        if !"LFDS".contains(c) {
            return vec![tok.to_string()];
        }
        i += 1;
        if i < oc.len() && oc[i] == '(' {
            let Some(close) = oc[i..].iter().position(|&x| x == ')') else { return vec![tok.to_string()] };
            let inner: String = oc[i + 1..i + close].iter().collect();
            let Some(nl) = numlist(&inner) else { return vec![tok.to_string()] };
            groups.push((c, nl));
            i += close + 1;
        } else {
            let s = i;
            while i < oc.len() && oc[i].is_ascii_digit() {
                i += 1;
            }
            let k = if i > s { oc[s..i].iter().collect::<String>().parse().unwrap_or(1) } else { 1 };
            groups.push((c, vec![k]));
        }
    }
    let vars: Vec<String> = match rest.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        Some(inner) => inner.split_whitespace().map(String::from).collect(),
        None => vec![rest.to_string()],
    };
    // cartesian product of the operator choices
    let mut combos: Vec<Vec<(char, i64)>> = vec![vec![]];
    for (c, ks) in &groups {
        combos = combos.iter().flat_map(|p| ks.iter().map(move |k| { let mut q = p.clone(); q.push((*c, *k)); q })).collect();
    }
    let mut out = Vec::new();
    for v in &vars {
        for c in &combos {
            let o = op_text(c);
            out.push(if o.is_empty() { v.clone() } else { format!("{o}.{v}") });
        }
    }
    out
}

fn top_dot(tok: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in tok.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            '.' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Expand operator lists in a whole varlist text.
pub fn expand_text(text: &str) -> String {
    if !text.contains('(') {
        return text.to_string();
    }
    ferrostat_parser::parser::split_top_level_words(text).iter().flat_map(|t| expand_token(t)).collect::<Vec<_>>().join(" ")
}

/// The tsset time variable (error 111 when not set).
pub fn time_var(st: &Store) -> Result<String, String> {
    match &st.tsset {
        Some((_, t)) if !t.is_empty() => Ok(t.clone()),
        _ => Err(rc_err(111, "time variable not set, use tsset varname ...")),
    }
}

/// Evaluate a (possibly operator-prefixed) variable or expression for every observation.
pub fn values(ds: &DataSet, st: &Store, name: &str) -> Result<Vec<f64>, String> {
    if let Some(v) = ds.get_var(name) {
        if v.var_type.is_string() {
            return Err(rc_err(109, "type mismatch"));
        }
        return Ok(v.data.iter().map(|x| x.num()).collect());
    }
    let e = ferrostat_parser::parser::parse_expression(name).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st);
    (0..ds.n_obs).map(|i| env.eval(&e, i)?.num()).collect()
}

/// Resolve the variable part of `op.var` (abbreviations allowed).
pub fn resolve(ds: &DataSet, tok: &str) -> Result<String, String> {
    match tok.rsplit_once('.') {
        Some((op, v)) if lag_poly(op).is_some() => Ok(format!("{}.{}", op.to_uppercase(), crate::varlist::resolve_name(ds, v)?)),
        _ => crate::varlist::resolve_name(ds, tok),
    }
}

/// Text of a time value in the time variable's display format.
pub fn time_label(ds: &DataSet, tvar: &str, v: f64) -> String {
    let fmt = ds.get_var(tvar).map(|v| v.format.clone()).unwrap_or_default();
    if fmt.trim_start_matches('%').trim_start_matches('-').starts_with('t') {
        crate::dates::format_date(v, &fmt)
    } else {
        crate::format::fmt_g(v, 9)
    }
}

/// Periods per year implied by the time variable's format (quarterly 4,
/// monthly 12, ...); None when the format is not a date format.
pub fn periodicity(ds: &DataSet, tvar: &str) -> Option<f64> {
    let fmt = ds.get_var(tvar)?.format.to_lowercase();
    let f = fmt.trim_start_matches('%').trim_start_matches('-');
    let f = f.strip_prefix('t')?;
    Some(match f.chars().next()? {
        'q' => 4.0,
        'm' => 12.0,
        'y' => 1.0,
        'h' => 2.0,
        'w' => 52.0,
        'd' => 365.0,
        _ => return None,
    })
}

/// Sample observations with nonmissing `cols`, in time order, with their times.
pub fn ordered_rows(ds: &DataSet, st: &Store, mask: &[bool], cols: &[&Vec<f64>]) -> Result<(Vec<usize>, Vec<f64>), String> {
    let t = time_var(st)?;
    let tc = crate::util::col(ds, &t)?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(tc[i]) && cols.iter().all(|c| !is_missing(c[i]))).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    let times = rows.iter().map(|&i| tc[i]).collect();
    Ok((rows, times))
}

/// Number of gaps in a sorted sequence of times (steps larger than one period).
pub fn gaps(times: &[f64]) -> usize {
    times.windows(2).filter(|w| w[1] - w[0] > 1.0 + 1e-9).count()
}

/// Missing value helper.
pub fn miss() -> f64 {
    SYSMISS
}

/// Add a double variable (error 110 if it exists); values outside `rows` missing.
pub fn add_var(ds: &mut DataSet, name: &str, vals: Vec<f64>) -> Result<(), String> {
    crate::varlist::check_name(name)?;
    if ds.var_index.contains_key(name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let v = crate::util::new_double(name, vals);
    ds.add_variable(v).map_err(|e| rc_err(198, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polys() {
        assert_eq!(lag_poly("LD").unwrap(), vec![(1, 1.0), (2, -1.0)]);
        assert_eq!(lag_poly("D2").unwrap(), vec![(0, 1.0), (1, -2.0), (2, 1.0)]);
        assert_eq!(lag_poly("S4").unwrap(), vec![(0, 1.0), (4, -1.0)]);
        assert_eq!(lag_poly("F").unwrap(), vec![(-1, 1.0)]);
        assert!(lag_poly("i").is_none());
        assert_eq!(expand_text("L(1/2).x z"), "L.x L2.x z");
        assert_eq!(expand_text("L(0/1).(a b)"), "a L.a b L.b");
        assert_eq!(expand_text("L(1/2)D.y"), "LD.y L2D.y");
    }
}
