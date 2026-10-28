//! margins: predictive margins, adjusted predictions and marginal effects
//! with at() specifications (values, numlists, statistics), margins over
//! factor levels, subgroups (over()), predict() options; and marginsplot,
//! which draws the last margins results.
//!
//! The model is evaluated on the "element" level: every coefficient name is
//! a product of elements (a continuous variable, a factor-level indicator or
//! an expression). at() settings fix variables to values or to statistics of
//! their elements over the estimation sample; a margin is the average of the
//! prediction over the sample. Standard errors use the delta method with a
//! numerical Jacobian in the coefficients.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_expression, parse_line};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::graph::{finish, palette, Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

// ---------------------------------------------------------------------------
// numlists and at() specifications
// ---------------------------------------------------------------------------

/// Expand a numlist: `1 2 3`, `0/1`, `200(100)800`, `10(-2)0`, `1[2]9`.
pub fn numlist(text: &str) -> Result<Vec<f64>, String> {
    let bad = || rc_err(121, format!("invalid numlist: {}", text.trim()));
    // glue `30(10) 70` and `0 / 1` into single items
    let mut s = String::new();
    for c in text.replace(',', " ").chars() {
        if c.is_whitespace() && s.ends_with(['(', ')', '/', '[', ']']) {
            continue;
        }
        if matches!(c, '(' | ')' | '/' | '[' | ']') {
            while s.ends_with(' ') {
                s.pop();
            }
        }
        s.push(if c.is_whitespace() { ' ' } else { c });
    }
    let mut out = Vec::new();
    for item in s.split_whitespace() {
        let num = |t: &str| t.trim().parse::<f64>().map_err(|_| bad());
        if let Some((a, b)) = item.split_once('/') {
            let (a, b) = (num(a)?, num(b)?);
            let step = if b >= a { 1.0 } else { -1.0 };
            let n = ((b - a) / step + 1e-9).floor() as i64;
            out.extend((0..=n).map(|k| a + k as f64 * step));
        } else if let Some(i) = item.find(['(', '[']) {
            let close = if &item[i..i + 1] == "(" { ')' } else { ']' };
            let j = item.find(close).ok_or_else(bad)?;
            let (a, d, b) = (num(&item[..i])?, num(&item[i + 1..j])?, num(&item[j + 1..])?);
            if d == 0.0 || (b - a) / d < 0.0 {
                return Err(bad());
            }
            let n = ((b - a) / d + 1e-9).floor() as i64;
            out.extend((0..=n).map(|k| {
                let v = a + k as f64 * d;
                // tidy accumulated rounding (0.1 steps)
                let r = (v * 1e10).round() / 1e10;
                if (r - v).abs() < 1e-12 * v.abs().max(1.0) { r } else { v }
            }));
        } else {
            out.push(num(item)?);
        }
    }
    if out.is_empty() {
        return Err(bad());
    }
    Ok(out)
}

/// Statistic an at() setting fixes a variable at.
#[derive(Clone, Debug, PartialEq)]
enum Stat {
    Mean,
    Median,
    Min,
    Max,
    Zero,
    Base,
    Pctile(f64),
}

impl Stat {
    fn parse(s: &str) -> Option<Stat> {
        let s = s.trim().to_lowercase();
        Some(match s.as_str() {
            "mean" => Stat::Mean,
            "median" | "p50" => Stat::Median,
            "min" => Stat::Min,
            "max" => Stat::Max,
            "zero" => Stat::Zero,
            "base" => Stat::Base,
            _ => {
                let p: f64 = s.strip_prefix('p')?.parse().ok()?;
                if !(1.0..=99.0).contains(&p) {
                    return None;
                }
                Stat::Pctile(p)
            }
        })
    }
    fn name(&self) -> String {
        match self {
            Stat::Mean => "mean".into(),
            Stat::Median => "median".into(),
            Stat::Min => "min".into(),
            Stat::Max => "max".into(),
            Stat::Zero => "zero".into(),
            Stat::Base => "base".into(),
            Stat::Pctile(p) => format!("p{}", fmt_g(*p, 4)),
        }
    }
    fn of(&self, data: &[f64]) -> f64 {
        let mut v: Vec<f64> = data.iter().copied().filter(|x| !is_missing(*x)).collect();
        if v.is_empty() {
            return SYSMISS;
        }
        match self {
            Stat::Mean => v.iter().sum::<f64>() / v.len() as f64,
            Stat::Zero => 0.0,
            Stat::Min | Stat::Base => v.iter().cloned().fold(f64::INFINITY, f64::min),
            Stat::Max => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            Stat::Median | Stat::Pctile(_) => {
                v.sort_by(f64::total_cmp);
                let p = if let Stat::Pctile(p) = self { *p } else { 50.0 };
                util::pctile_sorted(&v, p)
            }
        }
    }
}

/// How one variable is set in an at() scenario.
#[derive(Clone, Debug, PartialEq)]
enum Setting {
    Value(f64),
    Stat(Stat),
    Asobserved,
}

/// One at() option: per variable, a list of settings (values expand into
/// the cartesian product, the last variable varying fastest).
type AtSpec = Vec<(String, Vec<Setting>)>;

fn read_balanced(chars: &[char], pos: &mut usize) -> String {
    // chars[*pos] == '('
    let mut depth = 0;
    let start = *pos + 1;
    while *pos < chars.len() {
        match chars[*pos] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let s: String = chars[start..*pos].iter().collect();
                    *pos += 1;
                    return s;
                }
            }
            _ => {}
        }
        *pos += 1;
    }
    chars[start.min(chars.len())..].iter().collect()
}

/// Parse the text of one at() option.
fn parse_at(text: &str, model: &Model, ds: &DataSet) -> Result<AtSpec, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pos = 0;
    let mut spec: AtSpec = Vec::new();
    let set = |spec: &mut AtSpec, var: String, s: Vec<Setting>| {
        if let Some(e) = spec.iter_mut().find(|(v, _)| *v == var) {
            e.1 = s;
        } else {
            spec.push((var, s));
        }
    };
    let skip_ws = |pos: &mut usize| {
        while *pos < chars.len() && chars[*pos].is_whitespace() {
            *pos += 1;
        }
    };
    let read_word = |pos: &mut usize| -> String {
        let st = *pos;
        while *pos < chars.len() && !chars[*pos].is_whitespace() && chars[*pos] != '=' && chars[*pos] != '(' {
            *pos += 1;
        }
        chars[st..*pos].iter().collect()
    };
    loop {
        skip_ws(&mut pos);
        if pos >= chars.len() {
            break;
        }
        if chars[pos] == '(' {
            let stat_text = read_balanced(&chars, &mut pos);
            let setting = if stat_text.trim().eq_ignore_ascii_case("asobserved") || stat_text.trim().eq_ignore_ascii_case("asobs") {
                Setting::Asobserved
            } else if stat_text.trim().eq_ignore_ascii_case("asbalanced") {
                return Err(rc_err(198, "at(): asbalanced not supported"));
            } else {
                Setting::Stat(Stat::parse(&stat_text).ok_or_else(|| rc_err(198, format!("at(): invalid statistic ({})", stat_text.trim())))?)
            };
            // the varlist that follows, up to the next (stat) or var=
            loop {
                skip_ws(&mut pos);
                if pos >= chars.len() || chars[pos] == '(' {
                    break;
                }
                let save = pos;
                let w = read_word(&mut pos);
                let mut look = pos;
                while look < chars.len() && chars[look].is_whitespace() {
                    look += 1;
                }
                if look < chars.len() && chars[look] == '=' {
                    pos = save;
                    break;
                }
                if w.is_empty() {
                    return Err(rc_err(198, "invalid at()"));
                }
                for v in model.expand_vars(ds, &w)? {
                    set(&mut spec, v, vec![setting.clone()]);
                }
            }
            continue;
        }
        let w = read_word(&mut pos);
        skip_ws(&mut pos);
        if w.is_empty() || pos >= chars.len() || chars[pos] != '=' {
            return Err(rc_err(198, "invalid at()"));
        }
        pos += 1;
        skip_ws(&mut pos);
        let value_text = if pos < chars.len() && chars[pos] == '(' {
            read_balanced(&chars, &mut pos)
        } else {
            let st = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() {
                if chars[pos] == '(' {
                    read_balanced(&chars, &mut pos);
                } else {
                    pos += 1;
                }
            }
            chars[st..pos].iter().collect()
        };
        let var = model.resolve_var(ds, &w)?;
        let vt = value_text.trim();
        let vt_inner = vt.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(vt);
        let settings = if let Some(s) = Stat::parse(vt_inner) {
            vec![Setting::Stat(s)]
        } else if vt_inner.eq_ignore_ascii_case("asobserved") {
            vec![Setting::Asobserved]
        } else {
            numlist(vt).map_err(|_| rc_err(198, "invalid at()"))?.into_iter().map(Setting::Value).collect()
        };
        set(&mut spec, var, settings);
    }
    Ok(spec)
}

/// Cartesian product of an at() specification.
fn expand_at(spec: &AtSpec) -> Vec<BTreeMap<String, Setting>> {
    let mut out: Vec<BTreeMap<String, Setting>> = vec![BTreeMap::new()];
    for (var, settings) in spec {
        let mut next = Vec::new();
        for m in &out {
            for s in settings {
                let mut m2 = m.clone();
                m2.insert(var.clone(), s.clone());
                next.push(m2);
            }
        }
        out = next;
    }
    out
}

// ---------------------------------------------------------------------------
// the model on the element level
// ---------------------------------------------------------------------------

struct Elem {
    /// variable name (empty for expressions)
    var: String,
    /// factor level of an indicator element
    level: Option<f64>,
    /// values over the estimation sample rows
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
enum Pred {
    Xb,
    Logit,
    Probit,
    Cloglog,
    /// exp(xb) (+ offset when `offset`)
    Exp { offset: bool },
    Glm(String),
    /// ordered outcome k (0-based) of `ncat`
    Ordered { probit: bool, k: usize },
    /// multinomial outcome k (0-based, in outcome order)
    Mlogit { k: usize },
}

struct Model {
    /// equation index of each coefficient (None for ancillary parameters)
    eq_of: Vec<Option<usize>>,
    term_of: Vec<usize>,
    terms: Vec<Vec<usize>>,
    elems: Vec<Elem>,
    neq: usize,
    /// model variables in order of first appearance
    vars: Vec<String>,
    /// sample rows (dataset observation indices)
    rows: Vec<usize>,
    offset: Vec<f64>,
    /// indices of the cut points (ordered models), in order
    cuts: Vec<usize>,
    /// mlogit: equation index of each outcome (None = base outcome, eta 0)
    out_eq: Vec<Option<usize>>,
}

impl Model {
    fn build(ds: &DataSet, st: &Store, b: &Matrix, rows: Vec<usize>) -> Result<Model, String> {
        let env = Env::with_store(ds, st);
        let mut eqs: Vec<String> = Vec::new();
        let mut eq_of = Vec::new();
        let mut term_of = Vec::new();
        let mut term_names: Vec<String> = Vec::new();
        let mut terms: Vec<Vec<usize>> = Vec::new();
        let mut elems: Vec<Elem> = Vec::new();
        let mut keys: Vec<String> = Vec::new();
        let mut vars: Vec<String> = Vec::new();
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        for (j, name) in b.colnames.iter().enumerate() {
            let (eq, nm) = match name.split_once(':') {
                Some((e, n)) => (e.to_string(), n.to_string()),
                None => (String::new(), name.clone()),
            };
            if eq == "/" || nm.starts_with('/') || name.starts_with('/') {
                eq_of.push(None);
                term_of.push(0);
                let bare = nm.trim_start_matches('/');
                if let Some(k) = bare.strip_prefix("cut").and_then(|k| k.parse::<usize>().ok()) {
                    cuts.push((k, j));
                }
                continue;
            }
            let qi = match eqs.iter().position(|e| *e == eq) {
                Some(i) => i,
                None => {
                    eqs.push(eq.clone());
                    eqs.len() - 1
                }
            };
            eq_of.push(Some(qi));
            let ti = match term_names.iter().position(|t| *t == nm) {
                Some(i) => i,
                None => {
                    let mut parts = Vec::new();
                    if nm != "_cons" {
                        for part in nm.split('#') {
                            let mut p = part;
                            for pre in ["co.", "c.", "o."] {
                                p = p.strip_prefix(pre).unwrap_or(p);
                            }
                            let (var, level, key) = match p.split_once('.') {
                                Some((l, v)) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => {
                                    let lv: f64 = l.trim_end_matches(['b', 'o', 'n']).parse().unwrap();
                                    (v.to_string(), Some(lv), format!("{}.{v}", fmt_g(lv, 12)))
                                }
                                _ if ds.var_index.contains_key(p) => (p.to_string(), None, p.to_string()),
                                _ => (String::new(), None, format!("expr:{p}")),
                            };
                            let ei = match keys.iter().position(|k| *k == key) {
                                Some(i) => i,
                                None => {
                                    let data: Vec<f64> = if !var.is_empty() {
                                        let col = util::col(ds, &var)?;
                                        rows.iter().map(|&i| match level {
                                            Some(l) => {
                                                if is_missing(col[i]) { SYSMISS } else if col[i] == l { 1.0 } else { 0.0 }
                                            }
                                            None => col[i],
                                        }).collect()
                                    } else {
                                        let e = parse_expression(p).map_err(|e| rc_err(198, e.message))?;
                                        rows.iter().map(|&i| env.eval(&e, i).and_then(|v| v.num())).collect::<Result<_, _>>()?
                                    };
                                    if !var.is_empty() && !vars.contains(&var) {
                                        vars.push(var.clone());
                                    }
                                    keys.push(key);
                                    elems.push(Elem { var, level, data });
                                    elems.len() - 1
                                }
                            };
                            parts.push(ei);
                        }
                    }
                    term_names.push(nm.clone());
                    terms.push(parts);
                    terms.len() - 1
                }
            };
            term_of.push(ti);
        }
        cuts.sort();
        let offset = match st.e.macros.get("offset") {
            Some(o) if !o.trim().is_empty() => {
                let e = parse_expression(o).map_err(|e| rc_err(198, e.message))?;
                rows.iter().map(|&i| env.eval(&e, i).and_then(|v| v.num())).collect::<Result<_, _>>()?
            }
            _ => vec![0.0; rows.len()],
        };
        let mut out_eq = Vec::new();
        if st.e.macros.get("cmd").is_some_and(|c| c == "mlogit") {
            let outs: Vec<f64> = st.e.matrices.get("out").map(|m| m.data.clone()).unwrap_or_default();
            if eqs.len() == outs.len() {
                out_eq = (0..outs.len()).map(Some).collect();
            } else {
                let base = st.e.get("baseout").unwrap_or(SYSMISS);
                let mut q = 0;
                for o in &outs {
                    if *o == base {
                        out_eq.push(None);
                    } else {
                        out_eq.push(Some(q));
                        q += 1;
                    }
                }
            }
        }
        Ok(Model { eq_of, term_of, terms, elems, neq: eqs.len().max(1), vars, rows, offset, cuts: cuts.into_iter().map(|c| c.1).collect(), out_eq })
    }

    fn is_factor(&self, var: &str) -> bool {
        self.elems.iter().any(|e| e.var == var && e.level.is_some())
    }

    fn levels(&self, var: &str) -> Vec<f64> {
        let mut l: Vec<f64> = self.elems.iter().filter(|e| e.var == var).filter_map(|e| e.level).collect();
        l.sort_by(f64::total_cmp);
        l.dedup();
        l
    }

    fn resolve_var(&self, ds: &DataSet, w: &str) -> Result<String, String> {
        let w = w.trim();
        let w = w.strip_prefix("i.").or_else(|| w.strip_prefix("c.")).unwrap_or(w);
        let v = crate::varlist::resolve_name(ds, w)?;
        if !self.vars.contains(&v) {
            return Err(rc_err(322, format!("'{v}' not found in list of covariates")));
        }
        Ok(v)
    }

    /// Variables named by a varlist word of an at() statistic.
    fn expand_vars(&self, ds: &DataSet, w: &str) -> Result<Vec<String>, String> {
        match w {
            "_all" => Ok(self.vars.clone()),
            "_factor" => Ok(self.vars.iter().filter(|v| self.is_factor(v)).cloned().collect()),
            "_continuous" => Ok(self.vars.iter().filter(|v| !self.is_factor(v)).cloned().collect()),
            _ => {
                let w = w.strip_prefix("i.").or_else(|| w.strip_prefix("c.")).unwrap_or(w);
                let mut out = Vec::new();
                for v in crate::varlist::expand(ds, w)? {
                    if !self.vars.contains(&v) {
                        return Err(rc_err(322, format!("'{v}' not found in list of covariates")));
                    }
                    out.push(v);
                }
                Ok(out)
            }
        }
    }
}

/// A fully specified evaluation: fixed element values (None = as observed),
/// the rows averaged over, and an optional derivative variable.
struct Scenario {
    fixed: Vec<Option<f64>>,
    rows: Vec<bool>,
    deriv: Option<String>,
}

impl Model {
    fn scenario(&self, settings: &BTreeMap<String, Setting>, rows: &[bool]) -> Scenario {
        let sub = |e: &Elem| -> Vec<f64> { e.data.iter().zip(rows).filter(|(_, r)| **r).map(|(v, _)| *v).collect() };
        let fixed = self
            .elems
            .iter()
            .map(|e| match settings.get(&e.var) {
                Some(Setting::Value(v)) => Some(match e.level {
                    Some(l) => {
                        if *v == l { 1.0 } else { 0.0 }
                    }
                    None => *v,
                }),
                Some(Setting::Stat(Stat::Base)) if e.level.is_some() => {
                    let base = self.levels(&e.var).first().copied().unwrap_or(0.0);
                    Some(if e.level == Some(base) { 1.0 } else { 0.0 })
                }
                Some(Setting::Stat(s)) => Some(s.of(&sub(e))),
                _ => None,
            })
            .collect();
        Scenario { fixed, rows: rows.to_vec(), deriv: None }
    }

    fn elem_val(&self, sc: &Scenario, e: usize, i: usize) -> f64 {
        match sc.fixed[e] {
            Some(v) => v,
            None => self.elems[e].data[i],
        }
    }

    /// Linear predictions of every equation for sample row `i`, and their
    /// derivatives with respect to the derivative variable.
    fn etas(&self, sc: &Scenario, b: &[f64], i: usize, eta: &mut [f64], deta: &mut [f64]) -> bool {
        eta.iter_mut().for_each(|x| *x = 0.0);
        deta.iter_mut().for_each(|x| *x = 0.0);
        for (j, eq) in self.eq_of.iter().enumerate() {
            let Some(q) = eq else { continue };
            if b[j] == 0.0 {
                continue;
            }
            let parts = &self.terms[self.term_of[j]];
            let mut prod = 1.0;
            for &p in parts {
                let v = self.elem_val(sc, p, i);
                if is_missing(v) {
                    return false;
                }
                prod *= v;
            }
            eta[*q] += b[j] * prod;
            if let Some(dv) = &sc.deriv {
                let mut d = 0.0;
                for (k, &p) in parts.iter().enumerate() {
                    let e = &self.elems[p];
                    if e.var == *dv && e.level.is_none() {
                        let mut o = 1.0;
                        for (m, &p2) in parts.iter().enumerate() {
                            if m != k {
                                o *= self.elem_val(sc, p2, i);
                            }
                        }
                        d += o;
                    }
                }
                deta[*q] += b[j] * d;
            }
        }
        true
    }

    /// Prediction and its derivatives with respect to each equation's xb.
    fn predict(&self, pk: &Pred, b: &[f64], eta: &[f64], i: usize, dp: &mut [f64]) -> f64 {
        dp.iter_mut().for_each(|x| *x = 0.0);
        let x = eta[0];
        match pk {
            Pred::Xb => {
                dp[0] = 1.0;
                x
            }
            Pred::Logit => {
                let p = dist::invlogit(x);
                dp[0] = p * (1.0 - p);
                p
            }
            Pred::Probit => {
                dp[0] = dist::normalden(x);
                dist::normal(x)
            }
            Pred::Cloglog => {
                let e = x.exp();
                dp[0] = e * (-e).exp();
                1.0 - (-e).exp()
            }
            Pred::Exp { offset } => {
                let v = (x + if *offset { self.offset[i] } else { 0.0 }).exp();
                dp[0] = v;
                v
            }
            Pred::Glm(link) => {
                let mu = |e: f64| glm_mu(link, e);
                let h = 1e-6 * (1.0 + x.abs());
                dp[0] = (mu(x + h) - mu(x - h)) / (2.0 * h);
                mu(x)
            }
            Pred::Ordered { probit, k } => {
                let (f, d): (fn(f64) -> f64, fn(f64) -> f64) = if *probit {
                    (dist::normal, dist::normalden)
                } else {
                    (dist::invlogit, |z| {
                        let p = dist::invlogit(z);
                        p * (1.0 - p)
                    })
                };
                let ncut = self.cuts.len();
                let (hi, dhi) = if *k < ncut { let c = b[self.cuts[*k]] - x; (f(c), d(c)) } else { (1.0, 0.0) };
                let (lo, dlo) = if *k > 0 { let c = b[self.cuts[*k - 1]] - x; (f(c), d(c)) } else { (0.0, 0.0) };
                dp[0] = -dhi + dlo;
                hi - lo
            }
            Pred::Mlogit { k } => {
                let e_of = |o: &Option<usize>| o.map_or(0.0, |q| eta[q]);
                let mx = self.out_eq.iter().map(e_of).fold(f64::NEG_INFINITY, f64::max);
                let den: f64 = self.out_eq.iter().map(|o| (e_of(o) - mx).exp()).sum();
                let p: Vec<f64> = self.out_eq.iter().map(|o| (e_of(o) - mx).exp() / den).collect();
                for (o, eq) in self.out_eq.iter().enumerate() {
                    if let Some(q) = eq {
                        dp[*q] = p[*k] * (if o == *k { 1.0 } else { 0.0 } - p[o]);
                    }
                }
                p[*k]
            }
        }
    }

    /// Average prediction (or derivative) over the scenario's rows.
    fn margin(&self, sc: &Scenario, pk: &Pred, b: &[f64]) -> f64 {
        let mut eta = vec![0.0; self.neq];
        let mut deta = vec![0.0; self.neq];
        let mut dp = vec![0.0; self.neq];
        let (mut s, mut n) = (0.0, 0.0);
        for i in 0..self.rows.len() {
            if !sc.rows[i] || !self.etas(sc, b, i, &mut eta, &mut deta) {
                continue;
            }
            let p = self.predict(pk, b, &eta, i, &mut dp);
            let v = if sc.deriv.is_some() { dp.iter().zip(&deta).map(|(a, c)| a * c).sum() } else { p };
            if !is_missing(v) && v.is_finite() {
                s += v;
                n += 1.0;
            }
        }
        if n == 0.0 { SYSMISS } else { s / n }
    }
}

fn glm_mu(link: &str, eta: f64) -> f64 {
    match link {
        "log" => eta.exp(),
        "logit" => dist::invlogit(eta),
        "probit" => dist::normal(eta),
        "cloglog" => 1.0 - (-eta.exp()).exp(),
        l if l.starts_with("power") => {
            let p: f64 = l[5..].trim().parse().unwrap_or(1.0);
            if p == 0.0 { eta.exp() } else { eta.powf(1.0 / p) }
        }
        _ => eta,
    }
}

// ---------------------------------------------------------------------------
// margins
// ---------------------------------------------------------------------------

pub struct Margins;

/// One estimated margin: a function of the coefficients.
struct Item {
    /// (dimension, level) pairs naming the row, e.g. [("_at", 2), ("rank", 3)]
    dims: Vec<(String, f64)>,
    /// dydx variable term (e.g. "gre" or "2.rank") for marginal effects
    eq: Option<String>,
    pred: Pred,
    /// scenarios: value = margin(sc[0]) - margin(sc[1]) when two are given
    sc: Vec<Scenario>,
}

fn outcome_label(ds: &DataSet, dep: &str, v: f64) -> String {
    if let Some(var) = ds.get_var(dep) {
        if !var.value_label_name.is_empty() {
            if let Some(l) = ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(v)) {
                return l.to_string();
            }
        }
    }
    fmt_g(v, 12)
}

fn dim_name(d: &(String, f64)) -> String {
    format!("{}.{}", fmt_g(d.1, 12), d.0)
}

impl CommandHandler for Margins {
    fn name(&self) -> &'static str {
        "margins"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.has_opt("pwcompare", 2) {
            return crate::pwcompare::margins_pw(ds, st, cmd);
        }
        let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default();
        let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
        let mut mask = util::touse(ds, st, cmd)?;
        for (i, m) in mask.iter_mut().enumerate() {
            *m = *m && sample.get(i).copied().unwrap_or(false);
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let level = util::level(cmd, st)?;
        let model = Model::build(ds, st, &b, rows.clone())?;
        let nrows = rows.len();
        let all_rows = vec![true; nrows];

        // ---- prediction ---------------------------------------------------
        let popt = cmd.opt("predict", 3).map(|s| s.trim().to_string()).unwrap_or_default();
        let pcmd = parse_line(&format!("p , {popt}")).map_err(|e| rc_err(198, e))?;
        let pstat = ["xb", "pr", "n", "ir", "mu", "p"].into_iter().find(|s| pcmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case(s)));
        let ordered = matches!(ecmd.as_str(), "ologit" | "oprobit");
        let mlogit = ecmd == "mlogit";
        // (Pred, expression label, predict() text)
        let mut preds: Vec<(Pred, String, String)> = Vec::new();
        let ptext = |extra: &str| -> String { format!("predict({extra})") };
        if (ordered || mlogit) && pstat != Some("xb") {
            let outs: Vec<f64> = st.e.matrices.get(if mlogit { "out" } else { "cat" }).map(|m| m.data.clone()).unwrap_or_default();
            let ncat = if mlogit { outs.len() } else { model.cuts.len() + 1 };
            let outs: Vec<f64> = if outs.len() == ncat { outs } else { (0..ncat).map(|k| k as f64 + 1.0).collect() };
            let want: Vec<usize> = match pcmd.opt("outcome", 3) {
                Some(o) => {
                    let o = o.trim();
                    let k = if let Some(ix) = o.strip_prefix('#') {
                        ix.parse::<usize>().ok().filter(|k| *k >= 1 && *k <= ncat).map(|k| k - 1)
                    } else if let Ok(val) = o.parse::<f64>() {
                        outs.iter().position(|x| *x == val)
                    } else {
                        outs.iter().position(|x| outcome_label(ds, &depvar, *x) == o)
                    };
                    vec![k.ok_or_else(|| rc_err(303, format!("outcome {o} not found")))?]
                }
                None => (0..ncat).collect(),
            };
            for k in want {
                let p = if mlogit { Pred::Mlogit { k } } else { Pred::Ordered { probit: ecmd == "oprobit", k } };
                let o = &popt;
                let pt = if o.is_empty() { ptext(&format!("pr outcome({})", fmt_g(outs[k], 12))) } else { ptext(o) };
                preds.push((p, format!("Pr({depvar}=={})", outcome_label(ds, &depvar, outs[k])), pt));
            }
        } else {
            let (p, label) = match (ecmd.as_str(), pstat) {
                (_, Some("xb")) => (Pred::Xb, "Linear prediction".to_string()),
                ("logit" | "logistic", _) => (Pred::Logit, format!("Pr({depvar})")),
                ("probit", _) => (Pred::Probit, format!("Pr({depvar})")),
                ("cloglog", _) => (Pred::Cloglog, format!("Pr({depvar})")),
                ("poisson" | "nbreg", Some("ir")) => (Pred::Exp { offset: false }, "Predicted incidence rate".to_string()),
                ("poisson" | "nbreg", _) => (Pred::Exp { offset: true }, "Predicted number of events".to_string()),
                ("glm", _) => (Pred::Glm(st.e.macros.get("link").cloned().unwrap_or_else(|| "identity".into())), format!("Predicted mean {depvar}")),
                ("regress", _) => (Pred::Xb, "Linear prediction".to_string()),
                _ => (Pred::Xb, "Linear prediction".to_string()),
            };
            preds.push((p, label, ptext(&popt)));
        }
        let multi_predict = preds.len() > 1;

        // ---- at() scenarios -----------------------------------------------
        let atmeans = cmd.has_opt("atmeans", 3);
        let default = if atmeans { Setting::Stat(Stat::Mean) } else { Setting::Asobserved };
        let mut at_sets: Vec<BTreeMap<String, Setting>> = Vec::new();
        let at_opts: Vec<&str> = cmd.opts.iter().filter(|(n, _)| n.eq_ignore_ascii_case("at")).filter_map(|(_, a)| a.as_deref()).collect();
        let at_given = !at_opts.is_empty();
        let mut atvars: Vec<String> = Vec::new();
        for text in &at_opts {
            let spec = parse_at(text, &model, ds)?;
            for (v, _) in &spec {
                if !atvars.contains(v) {
                    atvars.push(v.clone());
                }
            }
            for m in expand_at(&spec) {
                let mut full: BTreeMap<String, Setting> = model.vars.iter().map(|v| (v.clone(), default.clone())).collect();
                full.extend(m);
                at_sets.push(full);
            }
        }
        if at_sets.is_empty() {
            at_sets.push(model.vars.iter().map(|v| (v.clone(), default.clone())).collect());
        }
        let multi_at = at_sets.len() > 1;

        // ---- margins over factor levels and over() subgroups ----------------
        let mut over_terms: Vec<Vec<String>> = Vec::new();
        for w in cmd.main.split_whitespace() {
            let mut term = Vec::new();
            for part in w.split('#') {
                let part = part.strip_prefix("i.").unwrap_or(part);
                term.push(model.resolve_var(ds, part)?);
            }
            over_terms.push(term);
        }
        let mut subgroups: Vec<(Vec<(String, f64)>, Vec<bool>)> = vec![(Vec::new(), all_rows.clone())];
        let mut over_vars: Vec<String> = Vec::new();
        if let Some(o) = cmd.opt("over", 4) {
            over_vars = crate::varlist::expand(ds, o.trim())?;
            for ov in &over_vars {
                let col = util::col(ds, ov)?;
                let vals: Vec<f64> = rows.iter().map(|&i| col[i]).collect();
                let mut lv = vals.clone();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                let mut next = Vec::new();
                for (dims, m) in &subgroups {
                    for l in &lv {
                        let m2: Vec<bool> = m.iter().zip(&vals).map(|(a, x)| *a && *x == *l).collect();
                        if m2.iter().any(|x| *x) {
                            let mut d = dims.clone();
                            d.push((ov.clone(), *l));
                            next.push((d, m2));
                        }
                    }
                }
                subgroups = next;
            }
        }

        // ---- dydx ----------------------------------------------------------
        let dydx: Vec<String> = match cmd.opt("dydx", 2) {
            Some(t) if t.trim() == "*" || t.trim() == "_all" => model.vars.clone(),
            Some(t) => t.split_whitespace().map(|x| model.resolve_var(ds, x)).collect::<Result<_, _>>()?,
            None => Vec::new(),
        };

        // ---- items ---------------------------------------------------------
        // combinations of over-term levels: one block per term
        let level_combos = |term: &[String]| -> Vec<Vec<(String, f64)>> {
            let mut out: Vec<Vec<(String, f64)>> = vec![Vec::new()];
            for var in term {
                let mut lv = model.levels(var);
                if lv.is_empty() {
                    let col = util::col(ds, var).unwrap_or_default();
                    lv = rows.iter().map(|&i| col[i]).filter(|x| !is_missing(*x)).collect();
                    lv.sort_by(f64::total_cmp);
                    lv.dedup();
                }
                let mut next = Vec::new();
                for c in &out {
                    for l in &lv {
                        let mut c2 = c.clone();
                        c2.push((var.clone(), *l));
                        next.push(c2);
                    }
                }
                out = next;
            }
            out
        };
        let blocks: Vec<Vec<Vec<(String, f64)>>> = if over_terms.is_empty() { vec![vec![Vec::new()]] } else { over_terms.iter().map(|t| level_combos(t)).collect() };
        let mut items: Vec<Item> = Vec::new();
        let mut block_of: Vec<usize> = Vec::new();
        // dydx targets: (label, var, Option<(level, base)>)
        let mut targets: Vec<(String, String, Option<(f64, f64)>)> = Vec::new();
        for x in &dydx {
            if model.is_factor(x) {
                let lv = model.levels(x);
                let base_name = b.colnames.iter().find_map(|c| {
                    let nm = c.rsplit(':').next().unwrap_or(c);
                    let (l, var) = nm.split_once('.')?;
                    (var == x && l.ends_with('b') && !nm.contains('#')).then(|| l.trim_end_matches('b').parse::<f64>().ok()).flatten()
                });
                let base = base_name.unwrap_or(lv[0]);
                for l in lv.iter().filter(|l| **l != base) {
                    targets.push((format!("{}.{x}", fmt_g(*l, 12)), x.clone(), Some((*l, base))));
                }
            } else {
                targets.push((x.clone(), x.clone(), None));
            }
        }
        let n_targets = targets.len().max(1);
        for ti in 0..n_targets {
            for (bi, block) in blocks.iter().enumerate() {
                for (pi, (pk, _, _)) in preds.iter().enumerate() {
                    for (ai, at) in at_sets.iter().enumerate() {
                        for combo in block {
                            for (sdims, srows) in &subgroups {
                                let mut dims: Vec<(String, f64)> = Vec::new();
                                if multi_predict {
                                    dims.push(("_predict".into(), pi as f64 + 1.0));
                                }
                                if multi_at {
                                    dims.push(("_at".into(), ai as f64 + 1.0));
                                }
                                dims.extend(sdims.iter().cloned());
                                dims.extend(combo.iter().cloned());
                                let mut settings = at.clone();
                                for (var, l) in combo {
                                    settings.insert(var.clone(), Setting::Value(*l));
                                }
                                let (eq, sc) = match targets.get(ti) {
                                    None => (None, vec![model.scenario(&settings, srows)]),
                                    Some((label, var, None)) => {
                                        let mut s = model.scenario(&settings, srows);
                                        s.deriv = Some(var.clone());
                                        (Some(label.clone()), vec![s])
                                    }
                                    Some((label, var, Some((l, base)))) => {
                                        let mut s1 = settings.clone();
                                        s1.insert(var.clone(), Setting::Value(*l));
                                        let mut s0 = settings.clone();
                                        s0.insert(var.clone(), Setting::Value(*base));
                                        (Some(label.clone()), vec![model.scenario(&s1, srows), model.scenario(&s0, srows)])
                                    }
                                };
                                items.push(Item { dims, eq, pred: pk.clone(), sc });
                                block_of.push(bi);
                            }
                        }
                    }
                }
            }
        }

        // ---- estimates and delta-method variances ---------------------------
        let k = b.cols;
        let bv: Vec<f64> = b.data[..k].to_vec();
        let eval = |it: &Item, bb: &[f64]| -> f64 {
            let m0 = model.margin(&it.sc[0], &it.pred, bb);
            if it.sc.len() > 1 { m0 - model.margin(&it.sc[1], &it.pred, bb) } else { m0 }
        };
        let m = items.len();
        let mut est = Vec::with_capacity(m);
        let mut jac = Matrix::zeros(m, k);
        for (r, it) in items.iter().enumerate() {
            est.push(eval(it, &bv));
            for j in 0..k {
                if v[(j, j)] == 0.0 {
                    continue;
                }
                let h = 1e-6 * (1.0 + bv[j].abs());
                let mut bp = bv.clone();
                bp[j] += h;
                let mut bm = bv.clone();
                bm[j] -= h;
                jac[(r, j)] = (eval(it, &bp) - eval(it, &bm)) / (2.0 * h);
            }
        }
        let vm = jac.mul(&v)?.mul(&jac.transpose())?;

        // ---- output ---------------------------------------------------------
        let df = st.e.get("df_r").filter(|_| matches!(ecmd.as_str(), "regress" | "anova" | "reghdfe"));
        let crit = match df {
            Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
            None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
        };
        let stat = if df.is_some() { "t" } else { "z" };
        // every covariate fixed (by at() or the margins varlist): conditional
        let over_set: Vec<&String> = over_terms.iter().flatten().collect();
        let fixed_all = at_sets.iter().all(|a| model.vars.iter().all(|vn| over_set.contains(&vn) || !matches!(a.get(vn), Some(Setting::Asobserved) | None)));
        let title = if !dydx.is_empty() {
            if fixed_all { "Conditional marginal effects" } else { "Average marginal effects" }
        } else if fixed_all {
            "Adjusted predictions"
        } else {
            "Predictive margins"
        };
        let vce = st.e.macros.get("vcetype").cloned().unwrap_or_else(|| if ecmd == "regress" { "OLS".into() } else { "OIM".into() });
        let right = format!("Number of obs = {}", util::commas(nrows));
        let mut out = vec![String::new(), format!("{title}{:>w$}", right, w = 78usize.saturating_sub(title.len()).max(right.len() + 1)), format!("Model VCE: {vce}"), String::new()];
        if multi_predict {
            let pw = format!("{}._predict:", preds.len()).len() + 1;
            for (i, (_, l, p)) in preds.iter().enumerate() {
                out.push(format!("{:<pw$}{l}, {p}", format!("{}._predict:", i + 1)));
            }
        } else {
            out.push(format!("Expression: {}, {}", preds[0].1, preds[0].2));
        }
        if !targets.is_empty() {
            out.push(format!("dy/dx wrt:  {}", targets.iter().map(|t| t.0.clone()).collect::<Vec<_>>().join(" ")));
        }
        if !over_vars.is_empty() {
            out.push(format!("Over:       {}", over_vars.join(" ")));
        }
        // at legend
        let vsquish = cmd.has_opt("vsquish", 3);
        let mut atstats: Vec<String> = Vec::new();
        let mut at_cols: Vec<String> = Vec::new();
        for (ei, e) in model.elems.iter().enumerate() {
            if e.var.is_empty() || model.elems[..ei].iter().any(|x| x.var == e.var && x.level == e.level) {
                continue;
            }
            at_cols.push(match e.level {
                Some(l) => format!("{}.{}", fmt_g(l, 12), e.var),
                None => e.var.clone(),
            });
        }
        let mut at_mat = Matrix::zeros(at_sets.len(), at_cols.len());
        let mut legend: Vec<Vec<(String, String, String)>> = Vec::new();
        for (ai, a) in at_sets.iter().enumerate() {
            let sc = model.scenario(a, &all_rows);
            let mut lines = Vec::new();
            let mut stats = Vec::new();
            for (ci, cname) in at_cols.iter().enumerate() {
                let (var, lvl) = match cname.split_once('.') {
                    Some((l, v)) if l.parse::<f64>().is_ok() && model.elems.iter().any(|e| e.var == v && e.level.is_some()) => (v.to_string(), l.parse::<f64>().ok()),
                    _ => (cname.clone(), None),
                };
                let ei = model.elems.iter().position(|e| e.var == var && e.level == lvl).unwrap();
                let setting = a.get(&var).cloned().unwrap_or(Setting::Asobserved);
                at_mat[(ai, ci)] = sc.fixed[ei].unwrap_or(SYSMISS);
                stats.push(match &setting {
                    Setting::Value(_) => "value".to_string(),
                    Setting::Stat(s) => s.name(),
                    Setting::Asobserved => "asobserved".to_string(),
                });
            }
            // legend lines in model-variable order
            for var in &model.vars {
                match a.get(var) {
                    Some(Setting::Value(x)) => lines.push((var.clone(), g9(*x), String::new())),
                    Some(Setting::Stat(s)) => {
                        let si = format!(" ({})", s.name());
                        if model.is_factor(var) {
                            for l in model.levels(var) {
                                let ei = model.elems.iter().position(|e| e.var == *var && e.level == Some(l)).unwrap();
                                lines.push((format!("{}.{var}", fmt_g(l, 12)), g9(sc.fixed[ei].unwrap_or(SYSMISS)), si.clone()));
                            }
                        }
                        if let Some(ei) = model.elems.iter().position(|e| e.var == *var && e.level.is_none()) {
                            lines.push((var.clone(), g9(sc.fixed[ei].unwrap_or(SYSMISS)), si.clone()));
                        }
                    }
                    _ => {}
                }
            }
            atstats.push(stats.join(" "));
            legend.push(lines);
        }
        at_mat = at_mat.with_names((1..=at_sets.len()).map(|i| format!("{i}._at")).collect(), at_cols.clone());
        if !cmd.has_opt("noatlegend", 5) && legend.iter().any(|l| !l.is_empty()) {
            let nw = legend.iter().flatten().map(|l| l.0.chars().count()).max().unwrap_or(0);
            let vw = legend.iter().flatten().map(|l| l.1.chars().count()).max().unwrap_or(0);
            let prefix = |i: usize| if multi_at { format!("{}._at:", i + 1) } else { "At:".to_string() };
            let pw = (0..legend.len()).map(|i| prefix(i).len()).max().unwrap_or(3) + 1;
            for (i, lines) in legend.iter().enumerate() {
                if i > 0 && !vsquish && legend.iter().any(|l| l.len() > 1) {
                    out.push(String::new());
                }
                if lines.is_empty() {
                    out.push(format!("{:<pw$}(asobserved)", prefix(i)));
                }
                for (li, (n, val, s)) in lines.iter().enumerate() {
                    let p = if li == 0 { prefix(i) } else { String::new() };
                    out.push(format!("{p:<pw$}{n:<nw$} = {val:>vw$}{s}"));
                }
            }
        }
        out.push(String::new());
        out.push("-".repeat(78));
        out.push(format!("{:>12} |{:>12}Delta-method", "", ""));
        out.push(format!("{:>12} | {:>10}   std. err. {:>6}    P>|{}|     [{}% conf. interval]", "", if dydx.is_empty() { "Margin" } else { "dy/dx" }, stat, stat, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        let mut names: Vec<String> = Vec::new();
        let mut last_eq: Option<String> = None;
        let mut last_header: Option<String> = None;
        let mut last_block: Option<usize> = None;
        let mut any_factor_dydx = false;
        let mut table = Matrix::zeros(9, m);
        for (r, it) in items.iter().enumerate() {
            let se = vm[(r, r)].max(0.0).sqrt();
            let t = est[r] / se;
            let p = match df {
                Some(d) => 2.0 * dist::ttail(d, t.abs()),
                None => 2.0 * dist::normal(-t.abs()),
            };
            let (lo, hi) = (est[r] - crit * se, est[r] + crit * se);
            for (row, x) in [est[r], se, t, p, lo, hi, df.unwrap_or(SYSMISS), crit, 0.0].iter().enumerate() {
                table[(row, r)] = *x;
            }
            let dimtxt: Vec<String> = it.dims.iter().map(dim_name).collect();
            let base_name = dimtxt.join("#");
            names.push(match (&it.eq, base_name.is_empty()) {
                (Some(e), false) => format!("{e}:{base_name}"),
                (Some(e), true) => e.clone(),
                (None, false) => base_name.clone(),
                (None, true) => "_cons".into(),
            });
            // row label and headers
            let header = if it.dims.is_empty() { None } else { Some(it.dims.iter().map(|d| d.0.clone()).collect::<Vec<_>>().join("#")) };
            let mut label = it.dims.iter().map(|d| fmt_g(d.1, 12)).collect::<Vec<_>>().join(" ") + " ";
            if let Some(e) = &it.eq {
                if header.is_some() {
                    if last_eq.as_ref() != Some(e) {
                        if last_eq.is_some() {
                            out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
                        }
                        out.push(format!("{:<12} |", abbrev(e, 12)));
                        last_header = None;
                    }
                } else {
                    // dy/dx rows without dimensions: factor levels under a heading
                    let (lbl, h) = crate::est::display_name(e);
                    let nlev = targets.iter().filter(|t| t.2.is_some() && Some(&t.1) == h.as_ref()).count();
                    any_factor_dydx |= h.is_some();
                    match h {
                        Some(h) if nlev > 1 => {
                            if last_header.as_ref() != Some(&h) {
                                if !out.last().is_some_and(|l| l.ends_with(&"-".repeat(10))) {
                                    out.push(format!("{:>12} |", ""));
                                }
                                out.push(format!("{:>12} |", abbrev(&h, 12)));
                                last_header = Some(h);
                            }
                            label = format!("{lbl} ");
                        }
                        _ => {
                            if last_header.is_some() {
                                out.push(format!("{:>12} |", ""));
                                last_header = None;
                            }
                            label = e.clone();
                        }
                    }
                }
                last_eq = Some(e.clone());
            }
            if let Some(h) = &header {
                if last_header.as_ref() != Some(h) || last_block != Some(block_of[r]) {
                    if last_header.is_some() {
                        out.push(format!("{:>12} |", ""));
                    }
                    out.push(format!("{:>12} |", abbrev(h, 12)));
                    last_header = Some(h.clone());
                }
            } else if it.eq.is_none() {
                label = "_cons".into();
            }
            last_block = Some(block_of[r]);
            out.push(format!("{:>12} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", abbrev(&label, 12), g9(est[r]), g9(se), t, p, g9(lo), g9(hi)));
        }
        out.push("-".repeat(78));
        if any_factor_dydx {
            out.push("Note: dy/dx for factor levels is the discrete change from the base level.".into());
        }

        // ---- r() results ----------------------------------------------------
        let mut res = Results::new();
        res.set("N", nrows as f64);
        res.set("level", level);
        res.set("k_at", at_sets.len() as f64);
        res.set("k_predict", preds.len() as f64);
        res.set("k_margins", over_terms.len() as f64);
        res.set_str("cmd", "margins");
        res.set_str("est_cmd", ecmd.clone());
        res.set_str("title", title);
        res.set_str("model_vce", vce.to_lowercase());
        res.set_str("expression", preds[0].2.clone());
        for (i, (_, l, p)) in preds.iter().enumerate() {
            res.set_str(&format!("predict{}_label", i + 1), l.clone());
            res.set_str(&format!("predict{}_opts", i + 1), p.clone());
        }
        res.set_str("predict_label", preds[0].1.clone());
        if !over_terms.is_empty() {
            res.set_str("margins", over_terms.iter().map(|t| t.iter().map(|v| format!("i.{v}")).collect::<Vec<_>>().join("#")).collect::<Vec<_>>().join(" "));
        }
        if !over_vars.is_empty() {
            res.set_str("over", over_vars.join(" "));
        }
        if !dydx.is_empty() {
            res.set_str("xvars", dydx.join(" "));
            res.set_str("derivatives", "dy/dx");
        }
        if at_given || atmeans {
            res.set_mat("at", at_mat);
            if !atvars.is_empty() {
                res.set_str("atvars", atvars.join(" "));
            }
            for (i, s) in atstats.iter().enumerate() {
                res.set_str(&format!("atstats{}", i + 1), s.clone());
            }
        }
        let rown = vec!["b".to_string(), "se".into(), stat.into(), "pvalue".into(), "ll".into(), "ul".into(), "df".into(), "crit".into(), "eform".into()];
        res.set_mat("table", table.with_names(rown, names.clone()));
        let bm = Matrix::row_vector(&est).with_names(vec!["y1".into()], names.clone());
        let vmn = vm.with_names(names.clone(), names);
        if cmd.has_opt("post", 4) {
            let e = st.begin_e();
            e.set("N", nrows as f64);
            e.set_str("cmd", "margins");
            e.set_str("est_cmd", ecmd);
            e.set_str("title", title);
            e.set_str("properties", "b V");
            e.set_mat("b", bm.clone());
            e.set_mat("V", vmn.clone());
            e.sample = Some(sample);
        }
        res.set_mat("b", bm);
        res.set_mat("V", vmn);
        st.r = res;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// marginsplot
// ---------------------------------------------------------------------------

pub struct Marginsplot;

/// One plotted margin: its dimensions (with at() variables expanded), the
/// series it belongs to, and estimate with interval.
struct Pt {
    dims: Vec<(String, f64)>,
    eq: Option<String>,
    b: f64,
    lo: f64,
    hi: f64,
}

impl CommandHandler for Marginsplot {
    fn name(&self) -> &'static str {
        "marginsplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let r = &st.r;
        let src = if r.macros.get("cmd").is_some_and(|c| c == "margins") {
            r.clone()
        } else if st.e.macros.get("cmd").is_some_and(|c| c == "margins") {
            st.e.clone()
        } else {
            return Err(rc_err(301, "previous command was not margins"));
        };
        let b = src.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "margins results not found"))?;
        let table = src.matrices.get("table").cloned();
        let vmat = src.matrices.get("V").cloned();
        let level = src.get("level").unwrap_or(st.level);
        let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let atm = src.matrices.get("at").cloned();
        // the at() variables that vary across _at, with their value per _at
        let mut at_vals: Vec<(String, Vec<f64>)> = Vec::new();
        if let Some(a) = &atm {
            let mut vars: Vec<String> = Vec::new();
            for c in &a.colnames {
                let v = match c.split_once('.') {
                    Some((l, v)) if l.parse::<f64>().is_ok() => v.to_string(),
                    _ => c.clone(),
                };
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
            for var in vars {
                let mut vals = Vec::new();
                for i in 0..a.rows {
                    let stats: Vec<String> = src.macros.get(&format!("atstats{}", i + 1)).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
                    let mut val = SYSMISS;
                    for (j, c) in a.colnames.iter().enumerate() {
                        if stats.get(j).map(String::as_str) != Some("value") {
                            continue;
                        }
                        if *c == var {
                            val = a.get(i, j);
                        } else if let Some((l, v)) = c.split_once('.') {
                            if v == var && a.get(i, j) == 1.0 {
                                val = l.parse().unwrap_or(SYSMISS);
                            }
                        }
                    }
                    vals.push(val);
                }
                let first = vals[0];
                if vals.iter().all(|x| !is_missing(*x)) && vals.iter().any(|x| *x != first) {
                    at_vals.push((var, vals));
                }
            }
            // the order the at() options named the variables in
            let order: Vec<String> = src.macros.get("atvars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
            at_vals.sort_by_key(|(v, _)| order.iter().position(|o| o == v).unwrap_or(usize::MAX));
        }
        let mut pts: Vec<Pt> = Vec::new();
        for (j, name) in b.colnames.iter().enumerate() {
            let est = b.get(0, j);
            let (lo, hi) = match &table {
                Some(t) if t.rows >= 6 => (t.get(4, j), t.get(5, j)),
                _ => {
                    let se = vmat.as_ref().map_or(0.0, |v| v.get(j, j).max(0.0).sqrt());
                    (est - crit * se, est + crit * se)
                }
            };
            let (eq, rest) = match name.split_once(':') {
                Some((e, r)) => (Some(e.to_string()), r.to_string()),
                None => (None, name.clone()),
            };
            let mut dims = Vec::new();
            let mut eq = eq;
            if rest == "_cons" {
            } else {
                let mut parsed = Vec::new();
                let mut ok = true;
                for part in rest.split('#') {
                    match part.split_once('.') {
                        Some((l, v)) if l.trim_end_matches(['b', 'o']).parse::<f64>().is_ok() => parsed.push((v.to_string(), l.trim_end_matches(['b', 'o']).parse::<f64>().unwrap())),
                        _ => ok = false,
                    }
                }
                if ok && (eq.is_some() || src.macros.get("xvars").is_none()) {
                    for (d, l) in parsed {
                        if d == "_at" && !at_vals.is_empty() {
                            let ai = l as usize - 1;
                            for (var, vals) in &at_vals {
                                dims.push((var.clone(), vals.get(ai).copied().unwrap_or(SYSMISS)));
                            }
                        } else {
                            dims.push((d, l));
                        }
                    }
                } else if eq.is_none() {
                    // dy/dx term without dimensions
                    eq = Some(rest.clone());
                }
            }
            pts.push(Pt { dims, eq, b: est, lo, hi });
        }
        if pts.is_empty() {
            return Err(rc_err(2000, "nothing to plot"));
        }
        // dimension on the x axis
        let mut dim_names: Vec<String> = Vec::new();
        for p in &pts {
            for (d, _) in &p.dims {
                if !dim_names.contains(d) {
                    dim_names.push(d.clone());
                }
            }
        }
        let xopt = cmd.opt("xdimension", 2).or_else(|| cmd.opt("x", 1)).map(|s| s.trim().trim_start_matches("i.").to_string());
        let xdim: Option<String> = match xopt {
            Some(x) => {
                let x = if dim_names.contains(&x) { x } else { crate::varlist::resolve_name(ds, &x).unwrap_or(x) };
                if !dim_names.contains(&x) {
                    return Err(rc_err(198, format!("dimension {x} not found in margins results")));
                }
                Some(x)
            }
            None => {
                let at_first = at_vals.first().map(|a| a.0.clone()).filter(|a| dim_names.contains(a));
                at_first.or_else(|| dim_names.iter().find(|d| *d != "_predict").cloned()).or_else(|| dim_names.first().cloned())
            }
        };
        // x positions and series keys
        let eqs: Vec<String> = {
            let mut v: Vec<String> = Vec::new();
            for p in &pts {
                if let Some(e) = &p.eq {
                    if !v.contains(e) {
                        v.push(e.clone());
                    }
                }
            }
            v
        };
        let label_of = |dim: &str, val: f64| -> String {
            if dim == "_predict" {
                if let Some(l) = src.macros.get(&format!("predict{}_label", val as usize)) {
                    return l.clone();
                }
            }
            if let Some(var) = ds.get_var(dim) {
                if !var.value_label_name.is_empty() {
                    if let Some(l) = ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(val)) {
                        return l.to_string();
                    }
                }
            }
            fmt_g(val, 12)
        };
        let mut series: Vec<(String, Vec<(f64, f64, f64, f64)>)> = Vec::new();
        let mut x_cats: Vec<(f64, String)> = Vec::new();
        for (k, p) in pts.iter().enumerate() {
            let (x, rest): (f64, Vec<&(String, f64)>) = match &xdim {
                Some(xd) => {
                    let xv = p.dims.iter().find(|d| d.0 == *xd).map(|d| d.1).unwrap_or(k as f64 + 1.0);
                    (xv, p.dims.iter().filter(|d| d.0 != *xd).collect())
                }
                None => {
                    // no dimensions: one category per dy/dx term (or margin)
                    let pos = p.eq.as_ref().and_then(|e| eqs.iter().position(|x| x == e)).unwrap_or(k) as f64 + 1.0;
                    (pos, p.dims.iter().collect())
                }
            };
            if xdim.is_none() {
                let text = p.eq.clone().unwrap_or_else(|| if b.colnames[k] == "_cons" { String::new() } else { b.colnames[k].clone() });
                if !x_cats.iter().any(|c| c.0 == x) {
                    x_cats.push((x, text));
                }
            }
            let mut key: Vec<String> = Vec::new();
            if xdim.is_some() {
                if let Some(e) = &p.eq {
                    key.push(e.clone());
                }
            }
            for (d, v) in rest {
                key.push(if d == "_predict" { label_of(d, *v) } else { format!("{d}={}", label_of(d, *v)) });
            }
            let key = key.join(", ");
            match series.iter_mut().find(|s| s.0 == key) {
                Some(s) => s.1.push((x, p.b, p.lo, p.hi)),
                None => series.push((key, vec![(x, p.b, p.lo, p.hi)])),
            }
        }
        let x_is_cat = match &xdim {
            None => true,
            Some(xd) => !at_vals.iter().any(|a| a.0 == *xd),
        };
        if let Some(xd) = &xdim {
            let mut vals: Vec<f64> = pts.iter().filter_map(|p| p.dims.iter().find(|d| d.0 == *xd).map(|d| d.1)).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            if x_is_cat || vals.len() <= 25 {
                x_cats = vals.iter().map(|v| (*v, label_of(xd, *v))).collect();
            }
        }
        // drawing
        let scheme = match cmd.opt("scheme", 3).map(|s| s.trim().to_lowercase()) {
            Some(s) if crate::graph::is_scheme(&s) => s,
            _ => st.scheme.clone(),
        };
        let pal = palette(&scheme);
        let noci = cmd.has_opt("noci", 4);
        let recast = cmd.opt("recast", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "connected".into());
        let recastci = cmd.opt("recastci", 7).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "rcap".into());
        let ci_style = match recastci.as_str() {
            "rarea" => RangeStyle::Area,
            "rline" => RangeStyle::Line,
            "rspike" => RangeStyle::Spike,
            "rbar" => RangeStyle::Bar,
            "rcap" => RangeStyle::Cap,
            other => return Err(rc_err(198, format!("recastci({other}) not allowed"))),
        };
        let mut xs: Vec<f64> = series.iter().flat_map(|s| s.1.iter().map(|p| p.0)).collect();
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        let step = xs.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
        let step = if step.is_finite() && step > 0.0 { step } else { 1.0 };
        let mut graph = Graph::default();
        let mut ci_layers = Vec::new();
        let mut est_layers = Vec::new();
        let nser = series.len();
        let plotopts = cmd.opt("plotopts", 5).map(|s| parse_line(&format!("o , {s}"))).transpose().map_err(|e| rc_err(198, e))?;
        let ciopts = cmd.opt("ciopts", 3).map(|s| parse_line(&format!("o , {s}"))).transpose().map_err(|e| rc_err(198, e))?;
        for (si, (key, mut ps)) in series.into_iter().enumerate() {
            ps.sort_by(|a, b| a.0.total_cmp(&b.0));
            // series side by side when bars are drawn
            let shift = if recast == "bar" && nser > 1 { (si as f64 - (nser as f64 - 1.0) / 2.0) * step * 0.8 / nser as f64 } else { 0.0 };
            let c = pal[si % pal.len()];
            let mut colour = (c.0, c.1, c.2);
            if let Some(cc) = plotopts.as_ref().and_then(|p| p.opt("color", 5).or_else(|| p.opt("mcolor", 2)).or_else(|| p.opt("lcolor", 2))) {
                if let Some(rgb) = crate::colors::parse(cc.trim()) {
                    colour = rgb;
                }
            }
            let mut est = Layer { label: key.clone(), color: Some(colour), ..Default::default() };
            match recast.as_str() {
                "bar" => {
                    est.kind = LayerKind::Range(RangeStyle::Bar);
                    est.width = step * 0.8 / nser as f64;
                    est.ranges = ps.iter().map(|p| (p.0 + shift, 0.0, p.1)).collect();
                }
                r => {
                    est.kind = match r {
                        "line" => LayerKind::Line,
                        "scatter" => LayerKind::Scatter,
                        "connected" => LayerKind::Connected,
                        other => return Err(rc_err(198, format!("recast({other}) not allowed"))),
                    };
                    est.points = ps.iter().map(|p| (p.0, p.1)).collect();
                }
            }
            if let Some(sym) = plotopts.as_ref().and_then(|p| p.opt("msymbol", 2)) {
                est.symbol = sym.trim().to_string();
            }
            if let Some(pat) = plotopts.as_ref().and_then(|p| p.opt("lpattern", 3)) {
                est.pattern = pat.trim().to_string();
            }
            if !noci {
                let mut ci_col = colour;
                if recast == "bar" {
                    ci_col = ((colour.0 as f64 * 0.55) as u8, (colour.1 as f64 * 0.55) as u8, (colour.2 as f64 * 0.55) as u8);
                }
                if let Some(cc) = ciopts.as_ref().and_then(|p| p.opt("color", 5).or_else(|| p.opt("lcolor", 2))) {
                    if let Some(rgb) = crate::colors::parse(cc.trim()) {
                        ci_col = rgb;
                    }
                }
                let mut l = Layer { kind: LayerKind::Range(ci_style), color: Some(ci_col), width: step * 0.25 / nser.max(1) as f64, ..Default::default() };
                l.ranges = ps.iter().map(|p| (p.0 + shift, p.2, p.3)).collect();
                if matches!(ci_style, RangeStyle::Area | RangeStyle::Bar) {
                    l.alpha = 0.35;
                }
                ci_layers.push(l);
            }
            est_layers.push(est);
        }
        // intervals first, the estimates on top
        if recast == "bar" {
            graph.layers.extend(est_layers);
            graph.layers.extend(ci_layers);
        } else {
            graph.layers.extend(ci_layers);
            graph.layers.extend(est_layers);
        }
        graph.legend = nser > 1;
        if !x_cats.is_empty() {
            graph.xlabels = x_cats;
        }
        // default titles
        let derivs = src.macros.get("xvars").is_some();
        let plabel = src.macros.get("predict1_label").cloned().unwrap_or_else(|| "Linear prediction".into());
        let kind = src.macros.get("title").cloned().unwrap_or_else(|| "Predictive margins".into());
        let over_name = match &xdim {
            Some(x) if x_is_cat && x != "_at" && x != "_predict" => Some(x.clone()),
            _ => None,
        };
        let of = if derivs {
            Some(src.macros.get("xvars").cloned().unwrap_or_default())
        } else {
            over_name.clone()
        };
        graph.title = match of {
            Some(o) if !o.is_empty() => format!("{kind} of {o} with {}% CIs", fmt_g(level, 4)),
            _ => format!("{kind} with {}% CIs", fmt_g(level, 4)),
        };
        if noci {
            graph.title = graph.title.replace(&format!(" with {}% CIs", fmt_g(level, 4)), "");
        }
        graph.ytitle = if derivs { format!("Effects on {plabel}") } else { plabel };
        graph.xtitle = match &xdim {
            Some(x) if x != "_at" && x != "_predict" => ds.get_var(x).map(|v| if v.label.is_empty() { x.clone() } else { v.label.clone() }).unwrap_or_else(|| x.clone()),
            Some(x) => x.clone(),
            None => String::new(),
        };
        // explicit graph options (title(), ytitle(), name(), legend(), ...)
        finish(&mut graph, cmd, st)
    }
}

#[cfg(test)]
mod tests {
    use super::numlist;

    #[test]
    fn numlists() {
        assert_eq!(numlist("2000(500)4000").unwrap(), vec![2000.0, 2500.0, 3000.0, 3500.0, 4000.0]);
        assert_eq!(numlist("0/1").unwrap(), vec![0.0, 1.0]);
        assert_eq!(numlist("30(10) 70").unwrap(), vec![30.0, 40.0, 50.0, 60.0, 70.0]);
        assert_eq!(numlist("1 2 3").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(numlist("(1 2)").is_err(), true);
    }
}
