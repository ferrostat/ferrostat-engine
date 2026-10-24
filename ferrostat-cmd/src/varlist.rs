//! Variable-list parsing: abbreviations, wildcards (`*`, `?`, `~`), ranges
//! (`a-c`), `_all`, and factor-variable / time-series notation.

use ferrostat_core::DataSet;

use crate::eval::rc_err;

fn wild(pat: &str, name: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = name.chars().collect();
    fn rec(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') | Some('~') => (0..=t.len()).any(|k| rec(&p[1..], &t[k..])),
            Some('?') => !t.is_empty() && rec(&p[1..], &t[1..]),
            Some(c) => t.first() == Some(c) && rec(&p[1..], &t[1..]),
        }
    }
    rec(&p, &t)
}

/// Resolve one name (exact or unique abbreviation).
pub fn resolve_name(ds: &DataSet, name: &str) -> Result<String, String> {
    if ds.var_index.contains_key(name) {
        return Ok(name.to_string());
    }
    let hits: Vec<&String> = ds.vars.iter().map(|v| &v.name).filter(|n| n.starts_with(name)).collect();
    match hits.len() {
        1 => Ok(hits[0].clone()),
        0 => Err(rc_err(111, format!("variable {name} not found"))),
        _ => Err(rc_err(111, format!("{name} ambiguous abbreviation"))),
    }
}

/// Split a varlist string into tokens, keeping `a - b` ranges together.
fn tokens(text: &str) -> Vec<String> {
    let raw: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let t = raw[i];
        if t == "-" && !out.is_empty() && i + 1 < raw.len() {
            let prev = out.pop().unwrap();
            out.push(format!("{prev}-{}", raw[i + 1]));
            i += 2;
            continue;
        }
        if t.ends_with('-') && t.len() > 1 && i + 1 < raw.len() {
            out.push(format!("{t}{}", raw[i + 1]));
            i += 2;
            continue;
        }
        if t.starts_with('-') && t.len() > 1 && !out.is_empty() {
            let prev = out.pop().unwrap();
            out.push(format!("{prev}{t}"));
            i += 1;
            continue;
        }
        out.push(t.to_string());
        i += 1;
    }
    out
}

/// Expand a list of existing variables.
pub fn expand(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for tok in tokens(text) {
        for v in expand_token(ds, &tok)? {
            out.push(v);
        }
    }
    Ok(out)
}

/// Expand, removing duplicates while keeping first occurrence.
pub fn expand_unique(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for v in expand(ds, text)? {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    Ok(out)
}

fn expand_token(ds: &DataSet, tok: &str) -> Result<Vec<String>, String> {
    if tok == "_all" || tok == "*" {
        return Ok(ds.varlist());
    }
    // factor variables / time-series operators: keep the operator prefix
    if let Some((op, rest)) = split_operator(tok) {
        let inner = expand_token(ds, rest)?;
        return Ok(inner.into_iter().map(|v| format!("{op}.{v}")).collect());
    }
    if tok.contains('#') {
        // interaction: expand each side, keep as written
        let parts: Vec<&str> = tok.split('#').filter(|p| !p.is_empty()).collect();
        let mut resolved = Vec::new();
        for p in &parts {
            let e = expand_token(ds, p)?;
            resolved.push(e.join(" "));
        }
        let sep = if tok.contains("##") { "##" } else { "#" };
        return Ok(vec![resolved.join(sep)]);
    }
    if let Some((a, b)) = tok.split_once('-') {
        if !a.is_empty() && !b.is_empty() {
            let a = resolve_name(ds, a)?;
            let b = resolve_name(ds, b)?;
            let ia = ds.var_index[&a];
            let ib = ds.var_index[&b];
            if ia > ib {
                return Err(rc_err(198, format!("{tok}: variables out of order")));
            }
            return Ok(ds.vars[ia..=ib].iter().map(|v| v.name.clone()).collect());
        }
    }
    if tok.contains(['*', '?', '~']) {
        let hits: Vec<String> = ds.vars.iter().map(|v| v.name.clone()).filter(|n| wild(tok, n)).collect();
        if hits.is_empty() {
            return Err(rc_err(111, format!("variable {tok} not found")));
        }
        return Ok(hits);
    }
    Ok(vec![resolve_name(ds, tok)?])
}

/// `i.x`, `c.x`, `ib2.x`, `L.x`, `L2D.x`, `2.x` ... -> (operator, rest)
pub fn split_operator(tok: &str) -> Option<(&str, &str)> {
    let (op, rest) = tok.split_once('.')?;
    if op.is_empty() || rest.is_empty() || tok.contains('#') {
        return None;
    }
    let lo = op.to_lowercase();
    let ok = lo.starts_with('i')
        || lo.starts_with('c')
        || lo.starts_with('o')
        || lo.chars().all(|c| c.is_ascii_digit() || c == 'b' || c == 'n' || c == 'o')
        || lo.chars().all(|c| "lfds0123456789".contains(c));
    ok.then_some((op, rest))
}

/// Validate a list of new variable names.
pub fn new_names(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for t in text.split_whitespace() {
        check_name(t)?;
        if ds.var_index.contains_key(t) {
            return Err(rc_err(110, format!("variable {t} already defined")));
        }
        out.push(t.to_string());
    }
    Ok(out)
}

pub fn check_name(t: &str) -> Result<(), String> {
    let ok = !t.is_empty()
        && t.chars().count() <= 32
        && t.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_')
        && t.chars().all(|c| c.is_alphanumeric() || c == '_');
    if ok { Ok(()) } else { Err(rc_err(198, format!("{t} invalid name"))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrostat_core::{VarType, Variable};

    fn ds() -> DataSet {
        let mut d = DataSet::new();
        for n in ["make", "price", "mpg", "rep78", "weight", "w2"] {
            d.add_variable(Variable::new(n, VarType::Double)).unwrap();
        }
        d
    }

    #[test]
    fn forms() {
        let d = ds();
        assert_eq!(expand(&d, "pr").unwrap(), ["price"]);
        assert_eq!(expand(&d, "price-rep78").unwrap(), ["price", "mpg", "rep78"]);
        assert_eq!(expand(&d, "price - mpg").unwrap(), ["price", "mpg"]);
        assert_eq!(expand(&d, "w*").unwrap(), ["weight", "w2"]);
        assert_eq!(expand(&d, "m?g").unwrap(), ["mpg"]);
        assert_eq!(expand(&d, "i.rep78 L.price").unwrap(), ["i.rep78", "L.price"]);
        assert!(expand(&d, "w").is_err()); // ambiguous
        assert!(expand(&d, "nosuch").is_err());
    }
}
