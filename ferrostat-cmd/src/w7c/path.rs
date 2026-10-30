//! Path notation shared by `sem` and `gsem`.
//!
//!   (y <- x1 x2) (L -> x1 x2@c1 x3@c1) (y2 <- L _cons@0, family(...))
//!
//! Names starting with a capital letter are latent variables (unless
//! `nocapslatent`, in which case only those named in `latent()` are);
//! `M[id]` is a latent variable at the level of `id`. `@#` fixes a
//! coefficient, `@name` gives it a label shared by every path with the same
//! label. A leading `#:` restricts a group to one latent class or group.

use std::collections::BTreeSet;

use ferrostat_core::DataSet;

use crate::eval::rc_err;
use crate::varlist;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coef {
    pub fix: Option<f64>,
    pub label: Option<String>,
}

impl Coef {
    pub fn parse(s: &str) -> Coef {
        let s = s.trim();
        if s.is_empty() {
            return Coef::default();
        }
        match s.parse::<f64>() {
            Ok(v) => Coef { fix: Some(v), label: None },
            Err(_) => Coef { fix: None, label: Some(s.to_string()) },
        }
    }
    pub fn is_free(&self) -> bool {
        self.fix.is_none() && self.label.is_none()
    }
}

/// Split `name@spec` into (name, constraint).
pub fn split_at(t: &str) -> (String, Coef) {
    match t.split_once('@') {
        Some((a, b)) => (a.trim().to_string(), Coef::parse(b)),
        None => (t.trim().to_string(), Coef::default()),
    }
}

#[derive(Clone, Debug)]
pub struct PathSpec {
    pub from: String,
    pub to: String,
    pub c: Coef,
    /// latent class / group restriction (`2: y <- x`)
    pub class: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub paths: Vec<PathSpec>,
    /// intercept constraints: (variable, constraint, class)
    pub cons: Vec<(String, Coef, Option<usize>)>,
    /// equations without a constant
    pub nocons: BTreeSet<String>,
    /// per-outcome option text (family/link/...)
    pub eqopts: Vec<(String, String)>,
    /// all variable names in order of first appearance
    pub order: Vec<String>,
    pub latent: BTreeSet<String>,
    /// multilevel latents: name -> level variable
    pub levels: Vec<(String, String)>,
}

impl Spec {
    pub fn is_latent(&self, n: &str) -> bool {
        self.latent.contains(n)
    }
    pub fn add_name(&mut self, n: &str) {
        if !self.order.iter().any(|x| x == n) {
            self.order.push(n.to_string());
        }
    }
    pub fn eqopt(&self, y: &str) -> String {
        self.eqopts.iter().filter(|(v, _)| v == y).map(|(_, o)| o.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// Top-level parenthesized groups of `text`; text outside parentheses forms
/// one group too.
pub fn groups(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut outside = String::new();
    for ch in text.chars() {
        match ch {
            '(' => {
                if depth > 0 {
                    cur.push(ch);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            _ => {
                if depth > 0 {
                    cur.push(ch);
                } else {
                    outside.push(ch);
                }
            }
        }
    }
    if !outside.trim().is_empty() {
        out.push(outside.trim().to_string());
    }
    out
}

/// Split at the first top-level comma.
pub fn split_comma(s: &str) -> (String, String) {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => return (s[..i].to_string(), s[i + 1..].to_string()),
            _ => {}
        }
    }
    (s.to_string(), String::new())
}

/// Words of `s`, keeping bracketed parts together.
pub fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '(' | '[' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' => {
                depth -= 1;
                cur.push(c);
            }
            c if c.is_whitespace() && depth == 0 => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn is_caps(n: &str) -> bool {
    n.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// Parse the model part of a sem/gsem command.
pub fn parse(ds: &DataSet, text: &str, latent_opt: &[String], caps: bool) -> Result<Spec, String> {
    let mut sp = Spec::default();
    for l in latent_opt {
        sp.latent.insert(l.clone());
    }
    for g in groups(text) {
        let (body, opts) = split_comma(&g);
        let mut body = body.trim().to_string();
        // class / group prefix "2:"
        let mut class = None;
        if let Some((a, b)) = body.split_once(':') {
            if let Ok(c) = a.trim().parse::<usize>() {
                class = Some(c);
                body = b.trim().to_string();
            }
        }
        let norm = body.replace("@ ", "@").replace(" @", "@").replace("->", " -> ").replace("<-", " <- ");
        let toks = words(&norm);
        // segments separated by arrows
        let mut segs: Vec<Vec<String>> = vec![Vec::new()];
        let mut ops: Vec<bool> = Vec::new(); // true: ->
        for t in toks {
            if t == "->" || t == "<-" {
                ops.push(t == "->");
                segs.push(Vec::new());
            } else {
                segs.last_mut().unwrap().push(t);
            }
        }
        if ops.is_empty() {
            // bare variable list: (y1 y2, family) declares outcomes with no predictors
            let mut targets = Vec::new();
            for t in &segs[0] {
                for (n, _) in expand_term(ds, &mut sp, t, caps)? {
                    sp.add_name(&n);
                    targets.push(n);
                }
            }
            for t in targets {
                if !opts.trim().is_empty() {
                    sp.eqopts.push((t.clone(), opts.trim().to_string()));
                }
                if has_nocons(&opts) {
                    sp.nocons.insert(t);
                }
            }
            continue;
        }
        let mut exp: Vec<Vec<(String, Coef)>> = Vec::new();
        for s in &segs {
            let mut v = Vec::new();
            for t in s {
                v.extend(expand_term(ds, &mut sp, t, caps)?);
            }
            if v.is_empty() {
                return Err(rc_err(198, format!("invalid path specification: {g}")));
            }
            exp.push(v);
        }
        let mut targets: Vec<String> = Vec::new();
        for (i, &fwd) in ops.iter().enumerate() {
            let (src, dst) = if fwd { (&exp[i], &exp[i + 1]) } else { (&exp[i + 1], &exp[i]) };
            for (d, dc) in dst {
                if d == "_cons" {
                    continue;
                }
                sp.add_name(d);
                if !targets.contains(d) {
                    targets.push(d.clone());
                }
                for (s, sc) in src {
                    if s == "_cons" {
                        sp.cons.push((d.clone(), sc.clone(), class));
                        continue;
                    }
                    sp.add_name(s);
                    // the constraint is written on the far end of the arrow
                    let c = if fwd { if dc.is_free() { sc.clone() } else { dc.clone() } } else if sc.is_free() { dc.clone() } else { sc.clone() };
                    sp.paths.push(PathSpec { from: s.clone(), to: d.clone(), c, class });
                }
            }
        }
        for t in targets {
            if !opts.trim().is_empty() {
                sp.eqopts.push((t.clone(), opts.trim().to_string()));
            }
            if has_nocons(&opts) {
                sp.nocons.insert(t);
            }
        }
    }
    Ok(sp)
}

fn has_nocons(opts: &str) -> bool {
    words(opts).iter().any(|w| {
        let w = w.to_lowercase();
        w.len() >= 6 && "noconstant".starts_with(&w)
    })
}

fn expand_term(ds: &DataSet, sp: &mut Spec, t: &str, caps: bool) -> Result<Vec<(String, Coef)>, String> {
    let (name, c) = split_at(t);
    if name.is_empty() {
        return Ok(vec![]);
    }
    if name == "_cons" {
        return Ok(vec![(name, c)]);
    }
    if let Some((l, rest)) = name.split_once('[') {
        let lev = rest.trim_end_matches(']').trim().to_string();
        let lev = varlist::resolve_name(ds, &lev)?;
        sp.latent.insert(l.to_string());
        if !sp.levels.iter().any(|(a, _)| a == l) {
            sp.levels.push((l.to_string(), lev));
        }
        return Ok(vec![(l.to_string(), c)]);
    }
    if sp.latent.contains(&name) || (caps && is_caps(&name) && !name.contains(['.', '#', '*', '-', '?'])) {
        sp.latent.insert(name.clone());
        return Ok(vec![(name, c)]);
    }
    let names = varlist::expand(ds, &name)?;
    Ok(names.into_iter().map(|n| (n, c.clone())).collect())
}

/// Parse `a*b@v c*d` style covariance lists (cov(), var(), means()).
/// Returns (first, second, constraint); `second` equals `first` for var().
pub fn cov_list(text: &str) -> Vec<(String, String, Coef)> {
    let mut out = Vec::new();
    for w in words(&text.replace("@ ", "@").replace(" @", "@")) {
        let (pair, c) = split_at(&w);
        match pair.split_once('*') {
            Some((a, b)) => out.push((a.trim().to_string(), b.trim().to_string(), c)),
            None => out.push((pair.clone(), pair, c)),
        }
    }
    out
}

/// All option texts named `name` (abbreviation `min`) in an option string.
pub fn opt_all(opts: &[(String, Option<String>)], name: &str, min: usize) -> Vec<String> {
    opts.iter()
        .filter(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min && name.starts_with(&n)
        })
        .filter_map(|(_, a)| a.clone())
        .collect()
}
