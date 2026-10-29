//! `forecast`: multi-equation forecast models. Equations come from stored
//! estimation results (`forecast estimates`) or identities; the model is
//! solved period by period (Gauss-Seidel iterations within each period)
//! over the forecast horizon, dynamically by default.
//!
//! The model is kept as text in the session settings: each equation is
//! `endogenous = expression` (estimation results are turned into linear
//! expressions in their regressors when added).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::{Command, Expr};

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsops;
use crate::util;

const K_NAME: &str = "_fc.name";
const K_EQS: &str = "_fc.eqs";
const K_EXOG: &str = "_fc.exog";
const K_ADJ: &str = "_fc.adjust";
const K_VARS: &str = "_fc.created";
const RS: char = '\u{1e}';
const US: char = '\u{1f}';

#[derive(Clone, Debug)]
struct Eqn {
    kind: String, // estimates | identity
    endog: String,
    expr: String,
    source: String,
}

fn eqs(st: &Store) -> Vec<Eqn> {
    st.settings
        .get(K_EQS)
        .map(|s| {
            s.split(RS)
                .filter(|l| !l.is_empty())
                .filter_map(|l| {
                    let p: Vec<&str> = l.split(US).collect();
                    (p.len() == 4).then(|| Eqn { kind: p[0].into(), endog: p[1].into(), expr: p[2].into(), source: p[3].into() })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn save_eqs(st: &mut Store, v: &[Eqn]) {
    let s = v.iter().map(|e| format!("{}{US}{}{US}{}{US}{}", e.kind, e.endog, e.expr, e.source)).collect::<Vec<_>>().join(&RS.to_string());
    st.settings.insert(K_EQS.into(), s);
}

fn list_of(st: &Store, key: &str) -> Vec<String> {
    st.settings.get(key).map(|s| s.split(RS).filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default()
}

fn save_list(st: &mut Store, key: &str, v: &[String]) {
    st.settings.insert(key.into(), v.join(&RS.to_string()));
}

fn add_eqs(st: &mut Store, new: Vec<(String, String)>, kind: &str, source: &str, par: &Par, head: &str) -> Result<String, String> {
    let mut v = eqs(st);
    for (d, _) in &new {
        if v.iter().any(|e| e.endog == *d) {
            return Err(rc_err(110, format!("an equation for {d} is already in the model")));
        }
    }
    if new.is_empty() {
        return Err(rc_err(198, "no equations found"));
    }
    v.extend(new.into_iter().map(|(d, x)| Eqn { kind: kind.into(), endog: d, expr: x, source: source.to_string() }));
    save_eqs(st, &v);
    save_par(st, source, par);
    Ok(format!("{head}\n  Forecast model {} now contains {} endogenous variable{}.", need_model(st)?, v.len(), if v.len() == 1 { "" } else { "s" }))
}

fn need_model(st: &Store) -> Result<String, String> {
    st.settings.get(K_NAME).cloned().ok_or_else(|| rc_err(198, "no forecast model has been created; use forecast create"))
}

/// Expression for one coefficient term (`L.x`, `1.g`, `c.x#c.z`, `_cons`).
fn term_expr(t: &str) -> Option<String> {
    if t == "_cons" {
        return Some("1".into());
    }
    let parts: Vec<String> = t
        .split('#')
        .map(|p| {
            let (op, v) = p.rsplit_once('.').unwrap_or(("", p));
            let opl = op.to_lowercase();
            if opl.is_empty() || opl == "c" {
                Some(format!("({v})"))
            } else if opl.ends_with('b') || opl.starts_with('o') {
                None
            } else if let Ok(level) = opl.trim_end_matches('b').parse::<f64>() {
                Some(format!("({v}=={level})"))
            } else if tsops::lag_poly(op).is_some() {
                Some(format!("({op}.{v})"))
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(parts.join("*"))
}

/// Linear equations (endogenous, expression) from estimation results.
fn eqs_from_results(e: &Results) -> Result<Vec<(String, String)>, String> {
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "estimation results not found"))?;
    let dep_list: Vec<String> = e.macros.get("depvar").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    if !matches!(cmd.as_str(), "regress" | "ivregress" | "newey" | "prais" | "var" | "reg3" | "sureg" | "areg" | "xtreg" | "arima" | "") && !cmd.starts_with("reg") {
        return Err(rc_err(322, format!("forecast estimates not supported after {cmd}")));
    }
    Ok(eqs_from_b(&b.colnames, &b.data, &dep_list, &cmd))
}

/// Linear equations from coefficient names (`eq:term`) and values.
fn eqs_from_b(cols: &[String], bv: &[f64], dep_list: &[String], cmd: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for (j, c) in cols.iter().enumerate() {
        let (eq, term) = match c.split_once(':') {
            Some((e, t)) => (e.to_string(), t.to_string()),
            None => (String::new(), c.clone()),
        };
        let endog = if eq.is_empty() || dep_list.len() == 1 && !dep_list.iter().any(|d| d == &eq) { dep_list.first().cloned().unwrap_or_default() } else { eq.clone() };
        if endog.is_empty() || dep_list.len() > 1 && !dep_list.contains(&endog) {
            continue;
        }
        if cmd == "arima" && eq != dep_list[0].rsplit('.').next().unwrap_or("") {
            continue;
        }
        let bj = bv[j];
        if bj == 0.0 {
            continue;
        }
        let Some(tx) = term_expr(&term) else { continue };
        let piece = format!("({})*{tx}", fmt_g(bj, 17));
        match out.iter_mut().find(|x| x.0 == endog) {
            Some(x) => x.1.push(piece),
            None => out.push((endog, vec![piece])),
        }
    }
    out.into_iter().map(|(d, p)| (d, p.join(" + "))).collect()
}

/// Coefficients behind an `estimates` or `coefvector` equation set, kept for
/// `solve, simulate()`: names, values, variance (row-major), dependent
/// variables, error standard deviations and the estimation command.
#[derive(Clone, Debug, Default)]
struct Par {
    cols: Vec<String>,
    b: Vec<f64>,
    v: Vec<f64>,
    deps: Vec<String>,
    sig: Vec<f64>,
    cmd: String,
}

const K_PAR: &str = "_fc.par.";

fn nums(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(&US.to_string())
}

fn parse_nums(s: &str) -> Vec<f64> {
    s.split(US).filter(|x| !x.is_empty()).filter_map(|x| x.parse().ok()).collect()
}

fn save_par(st: &mut Store, source: &str, p: &Par) {
    let j = |v: &[String]| v.join(&US.to_string());
    let s = [j(&p.cols), nums(&p.b), nums(&p.v), j(&p.deps), nums(&p.sig), p.cmd.clone()].join(&RS.to_string());
    st.settings.insert(format!("{K_PAR}{source}"), s);
}

fn load_par(st: &Store, source: &str) -> Option<Par> {
    let s = st.settings.get(&format!("{K_PAR}{source}"))?;
    let f: Vec<&str> = s.split(RS).collect();
    if f.len() != 6 {
        return None;
    }
    let words = |x: &str| x.split(US).filter(|w| !w.is_empty()).map(String::from).collect::<Vec<_>>();
    Some(Par { cols: words(f[0]), b: parse_nums(f[1]), v: parse_nums(f[2]), deps: words(f[3]), sig: parse_nums(f[4]), cmd: f[5].to_string() })
}

/// Parameters of stored estimation results, with dependent variables renamed by `names()`.
fn par_from_results(e: &Results, names: &[String]) -> Par {
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let mut deps: Vec<String> = e.macros.get("depvar").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    let v = e.matrices.get("V").filter(|v| v.rows == b.cols && v.cols == b.cols).map(|v| v.data.clone()).unwrap_or_default();
    let sig: Vec<f64> = if let Some(sg) = e.matrices.get("Sigma").filter(|m| m.rows == deps.len()) {
        (0..deps.len()).map(|i| sg.get(i, i).max(0.0).sqrt()).collect()
    } else if let Some(r) = e.get("rmse") {
        vec![r; deps.len()]
    } else if let Some(r) = e.get("sigma") {
        vec![r; deps.len()]
    } else {
        vec![0.0; deps.len()]
    };
    let mut cols = b.colnames.clone();
    if !names.is_empty() {
        for (i, nw) in names.iter().enumerate().take(deps.len()) {
            let old = deps[i].clone();
            for c in cols.iter_mut() {
                if let Some((eq, t)) = c.split_once(':') {
                    if eq == old {
                        *c = format!("{nw}:{t}");
                    }
                }
            }
            deps[i] = nw.clone();
        }
        if deps.len() == 1 {
            // single equation: name coefficients explicitly
            for c in cols.iter_mut() {
                if !c.contains(':') {
                    *c = format!("{}:{c}", deps[0]);
                }
            }
        }
    }
    Par { cols, b: b.data.clone(), v, deps, sig, cmd }
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Replace references to endogenous variables by their forecast variables.
/// In static mode references with a lag operator keep the actual values.
fn substitute(expr: &str, endog: &[String], fname: &dyn Fn(&str) -> String, static_: bool) -> String {
    let chars: Vec<char> = expr.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if is_ident_char(c) && (i == 0 || !is_ident_char(chars[i - 1])) {
            let mut j = i;
            while j < chars.len() && is_ident_char(chars[j]) {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            let next_is_dot = j < chars.len() && chars[j] == '.';
            let prev_dot = i > 0 && chars[i - 1] == '.';
            let next_paren = j < chars.len() && chars[j] == '(';
            if !next_is_dot && !next_paren && endog.contains(&word) && !(static_ && prev_dot) && !word.chars().next().unwrap().is_ascii_digit() {
                out.push_str(&fname(&word));
            } else {
                out.push_str(&word);
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

pub struct Forecast;

impl CommandHandler for Forecast {
    fn name(&self) -> &'static str {
        "forecast"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "create" | "cre" => {
                if st.settings.contains_key(K_NAME) && !cmd.has_opt("replace", 7) {
                    return Err(rc_err(110, "a forecast model already exists; specify replace"));
                }
                for k in [K_NAME, K_EQS, K_EXOG, K_ADJ] {
                    st.settings.remove(k);
                }
                st.settings.retain(|k, _| !k.starts_with(K_PAR));
                let name = if rest.is_empty() { "model".to_string() } else { rest.to_string() };
                st.settings.insert(K_NAME.into(), name.clone());
                Ok(format!("  Forecast model {name} started."))
            }
            "estimates" | "est" => {
                need_model(st)?;
                let name = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "estimation-results name required"))?;
                let res = if name == "." { st.e.clone() } else { st.estimates.get(name).cloned().ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))? };
                eqs_from_results(&res)?;
                let names: Vec<String> = cmd.opt("names", 1).map(|s| s.split(',').next().unwrap_or("").split_whitespace().map(String::from).collect()).unwrap_or_default();
                let par = par_from_results(&res, &names);
                let new = eqs_from_b(&par.cols, &par.b, &par.deps, &par.cmd);
                let source = if name == "." { "_est_e".to_string() } else { name.to_string() };
                add_eqs(st, new, "estimates", &source, &par, &format!("  Added estimation results from {}.", res.macros.get("cmd").cloned().unwrap_or_default()))
            }
            "coefvector" | "coefv" => {
                need_model(st)?;
                let name = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "coefficient vector name required"))?;
                let m = st.matrices.get(name).cloned().ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
                if m.rows != 1 && m.cols != 1 {
                    return Err(rc_err(503, format!("{name} must be a vector")));
                }
                let (cols, bv) = if m.rows == 1 { (m.colnames.clone(), m.data.clone()) } else { (m.rownames.clone(), m.data.clone()) };
                let k = bv.len();
                let names: Vec<String> = cmd.opt("names", 1).map(|s| s.split(',').next().unwrap_or("").split_whitespace().map(String::from).collect()).unwrap_or_default();
                let mut deps: Vec<String> = Vec::new();
                for c in &cols {
                    if let Some((eq, _)) = c.split_once(':') {
                        if !deps.iter().any(|d| d == eq) {
                            deps.push(eq.to_string());
                        }
                    }
                }
                if deps.is_empty() {
                    return Err(rc_err(198, format!("the columns of {name} must have equation names identifying the endogenous variables")));
                }
                let mut cols = cols;
                if !names.is_empty() {
                    if names.len() != deps.len() {
                        return Err(rc_err(198, "names() must list one name per equation"));
                    }
                    for c in cols.iter_mut() {
                        if let Some((eq, t)) = c.clone().split_once(':') {
                            let i = deps.iter().position(|d| d == eq).unwrap();
                            *c = format!("{}:{t}", names[i]);
                        }
                    }
                    deps = names;
                }
                let v = match cmd.opt("variance", 1) {
                    Some(vn) => {
                        let vm = st.matrices.get(vn.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", vn.trim())))?;
                        if vm.rows != k || vm.cols != k {
                            return Err(rc_err(503, "variance() matrix is not conformable with the coefficient vector"));
                        }
                        vm.data.clone()
                    }
                    None => Vec::new(),
                };
                let sig = match cmd.opt("errorvariance", 1) {
                    Some(en) => {
                        let em = st.matrices.get(en.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", en.trim())))?;
                        if em.rows != deps.len() || em.cols != deps.len() {
                            return Err(rc_err(503, "errorvariance() matrix must be square with one row per equation"));
                        }
                        (0..deps.len()).map(|i| em.get(i, i).max(0.0).sqrt()).collect()
                    }
                    None => vec![0.0; deps.len()],
                };
                let par = Par { cols, b: bv, v, deps, sig, cmd: "coefvector".into() };
                let new = eqs_from_b(&par.cols, &par.b, &par.deps, "");
                add_eqs(st, new, "coefvector", name, &par, &format!("  Added coefficient vector {name}."))
            }
            "identity" | "ide" => {
                need_model(st)?;
                let (lhs, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "identity must be of the form varname = exp"))?;
                let lhs = lhs.trim().to_string();
                let rhs = rhs.trim().to_string();
                if cmd.has_opt("generate", 3) {
                    if ds.get_var(&lhs).is_some() {
                        return Err(rc_err(110, format!("variable {lhs} already defined")));
                    }
                    let ex = ferrostat_parser::parser::parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
                    let env = crate::eval::Env::with_store(ds, st);
                    let vals: Vec<f64> = (0..ds.n_obs).map(|i| env.eval(&ex, i).and_then(|v| v.num()).unwrap_or(SYSMISS)).collect();
                    tsops::add_var(ds, &lhs, vals)?;
                } else if ds.get_var(&lhs).is_none() {
                    return Err(rc_err(111, format!("variable {lhs} not found; specify generate to create it")));
                }
                let mut v = eqs(st);
                if v.iter().any(|e| e.endog == lhs) {
                    return Err(rc_err(110, format!("an equation for {lhs} is already in the model")));
                }
                v.push(Eqn { kind: "identity".into(), endog: lhs, expr: rhs, source: String::new() });
                save_eqs(st, &v);
                Ok(format!("  Forecast model {} now contains {} endogenous variable{}.", need_model(st)?, v.len(), if v.len() == 1 { "" } else { "s" }))
            }
            "exogenous" | "exo" => {
                need_model(st)?;
                let mut v = list_of(st, K_EXOG);
                for w in tsops::expand_text(rest).split_whitespace() {
                    let w = tsops::resolve(ds, w)?;
                    if !v.contains(&w) {
                        v.push(w);
                    }
                }
                save_list(st, K_EXOG, &v);
                Ok(format!("  Forecast model {} now contains {} declared exogenous variable{}.", need_model(st)?, v.len(), if v.len() == 1 { "" } else { "s" }))
            }
            "adjust" | "adj" => {
                need_model(st)?;
                let (lhs, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "adjustment must be of the form varname = exp"))?;
                let lhs = lhs.trim().to_string();
                if !eqs(st).iter().any(|e| e.endog == lhs) {
                    return Err(rc_err(111, format!("{lhs} is not an endogenous variable of the model")));
                }
                let mut v = list_of(st, K_ADJ);
                v.push(format!("{lhs}{US}{}", rhs.trim()));
                save_list(st, K_ADJ, &v);
                Ok(format!("  Endogenous variable {lhs} now has {} adjustment{}.", v.iter().filter(|a| a.starts_with(&format!("{lhs}{US}"))).count(), if v.len() == 1 { "" } else { "s" }))
            }
            "describe" | "des" | "d" => {
                let name = need_model(st)?;
                let v = eqs(st);
                let ex = list_of(st, K_EXOG);
                let mut out = vec![String::new(), format!("  Forecast model {name} contains:"), format!("    {} endogenous variable{}", v.len(), if v.len() == 1 { "" } else { "s" })];
                for e in &v {
                    out.push(format!("      {:<12} ({}{})", e.endog, e.kind, if e.source.is_empty() { String::new() } else { format!(": {}", e.source) }));
                }
                out.push(format!("    {} declared exogenous variable{}", ex.len(), if ex.len() == 1 { "" } else { "s" }));
                for x in &ex {
                    out.push(format!("      {x}"));
                }
                let mut r = Results::new();
                r.set("n_endog", v.len() as f64);
                r.set("n_exog", ex.len() as f64);
                r.set_str("endog", v.iter().map(|e| e.endog.clone()).collect::<Vec<_>>().join(" "));
                r.set_str("exog", ex.join(" "));
                r.set_str("name", name);
                st.r = r;
                Ok(out.join("\n"))
            }
            "list" => {
                let name = need_model(st)?;
                let mut out = vec![String::new(), format!("    forecast create {name}")];
                let mut seen = Vec::new();
                for e in eqs(st) {
                    if e.kind == "estimates" || e.kind == "coefvector" {
                        if !seen.contains(&e.source) {
                            out.push(format!("    forecast {} {}", e.kind, if e.source == "_est_e" { "." } else { e.source.as_str() }));
                            seen.push(e.source.clone());
                        }
                    } else {
                        out.push(format!("    forecast identity {} = {}", e.endog, e.expr));
                    }
                }
                let ex = list_of(st, K_EXOG);
                if !ex.is_empty() {
                    out.push(format!("    forecast exogenous {}", ex.join(" ")));
                }
                for a in list_of(st, K_ADJ) {
                    let (l, r) = a.split_once(US).unwrap_or((&a, ""));
                    out.push(format!("    forecast adjust {l} = {r}"));
                }
                Ok(out.join("\n"))
            }
            "query" | "q" => Ok(match st.settings.get(K_NAME) {
                Some(n) => format!("  Forecast model {n} has been started."),
                None => "  No forecast model has been started.".into(),
            }),
            "clear" => {
                for k in [K_NAME, K_EQS, K_EXOG, K_ADJ] {
                    st.settings.remove(k);
                }
                st.settings.retain(|k, _| !k.starts_with(K_PAR));
                Ok(String::new())
            }
            "drop" => {
                let vars = list_of(st, K_VARS);
                let mut n = 0;
                for v in &vars {
                    if ds.get_var(v).is_some() {
                        ds.drop_variable(v).map_err(|e| rc_err(198, e))?;
                        n += 1;
                    }
                }
                st.settings.remove(K_VARS);
                Ok(format!("  ({n} forecast variable{} dropped)", if n == 1 { "" } else { "s" }))
            }
            "solve" => solve(ds, st, cmd),
            "" => Err(rc_err(198, "subcommand required")),
            o => Err(rc_err(198, format!("unknown forecast subcommand {o}"))),
        }
    }
}

fn set_val(ds: &mut DataSet, var: &str, i: usize, v: f64) {
    if let Some(x) = ds.get_var_mut(var) {
        x.data[i] = if is_missing(v) || !v.is_finite() { Value::Missing } else { Value::Double(v) };
    }
}

/// Solve the model period by period (Gauss-Seidel within a period), writing
/// into `targets`; `shock(k, row)` is added to equation k. Returns the rows
/// where the iterations did not converge.
#[allow(clippy::too_many_arguments)]
fn run_periods(ds: &mut DataSet, st: &mut Store, exprs: &[Expr], adj: &[(usize, Expr)], targets: &[String], periods: &[usize], actuals: bool, ycols: &[Vec<f64>], shock: &mut dyn FnMut(usize, usize) -> f64) -> Result<Vec<usize>, String> {
    let mut bad = Vec::new();
    for &i in periods {
        let shocks: Vec<f64> = (0..exprs.len()).map(|k| shock(k, i)).collect();
        let mut prev = vec![SYSMISS; exprs.len()];
        let mut converged = false;
        for _ in 0..500 {
            let mut change = 0.0f64;
            for (k, ex) in exprs.iter().enumerate() {
                if actuals && !is_missing(ycols[k][i]) {
                    continue;
                }
                let mut v = crate::eval::Env::with_store(ds, st).eval(ex, i)?.num()?;
                if !is_missing(v) {
                    v += shocks[k];
                }
                set_val(ds, &targets[k], i, v);
                for (ak, aex) in adj {
                    if *ak == k {
                        v = crate::eval::Env::with_store(ds, st).eval(aex, i)?.num()?;
                        set_val(ds, &targets[k], i, v);
                    }
                }
                let old = prev[k];
                let d = if is_missing(v) && is_missing(old) { 0.0 } else if is_missing(v) || is_missing(old) { 1.0 } else { (v - old).abs() / (old.abs() + 1.0) };
                change = change.max(d);
                prev[k] = v;
            }
            if change < 1e-10 {
                converged = true;
                break;
            }
        }
        if !converged {
            bad.push(i);
        }
    }
    Ok(bad)
}

/// Inside-parentheses contents of every `name(...)` in `s` (name abbreviable to `min`).
fn sub_opts(s: &str, name: &str, min: usize) -> Vec<String> {
    let mut out = Vec::new();
    let ch: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < ch.len() {
        if ch[i].is_ascii_alphabetic() && (i == 0 || !is_ident_char(ch[i - 1])) {
            let st0 = i;
            while i < ch.len() && is_ident_char(ch[i]) {
                i += 1;
            }
            let w: String = ch[st0..i].iter().collect::<String>().to_lowercase();
            if i < ch.len() && ch[i] == '(' {
                let mut depth = 0;
                let start = i + 1;
                while i < ch.len() {
                    if ch[i] == '(' {
                        depth += 1;
                    } else if ch[i] == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    i += 1;
                }
                if w.len() >= min && name.starts_with(&w) {
                    out.push(ch[start..i.min(ch.len())].iter().collect());
                }
            }
        }
        i += 1;
    }
    out
}

/// `solve, simulate(betas errors residuals, statistic(mean|stddev, prefix()|suffix()) reps(#))`.
#[allow(clippy::too_many_arguments)]
fn simulate(ds: &mut DataSet, st: &mut Store, spec: &str, model: &[Eqn], endog: &[String], exprs: &[Expr], adj: &[(usize, Expr)], periods: &[usize], actuals: bool, ycols: &[Vec<f64>], static_: bool, begin: f64, tc: &[f64], order: &[usize]) -> Result<Vec<String>, String> {
    let (tech, opts) = spec.split_once(',').unwrap_or((spec, ""));
    let tech: Vec<String> = tech.split_whitespace().map(|w| w.to_lowercase()).collect();
    let betas = tech.iter().any(|w| "betas".starts_with(w.as_str()) && w.len() >= 4);
    let errors = tech.iter().any(|w| "errors".starts_with(w.as_str()) && w.len() >= 3);
    let resid = tech.iter().any(|w| "residuals".starts_with(w.as_str()) && w.len() >= 3);
    if errors && resid {
        return Err(rc_err(198, "errors and residuals may not be combined"));
    }
    if !betas && !errors && !resid {
        return Err(rc_err(198, "simulate() requires betas, errors or residuals"));
    }
    let reps = sub_opts(opts, "reps", 3).first().map(|r| r.trim().parse::<usize>().map_err(|_| rc_err(198, "reps() invalid"))).transpose()?.unwrap_or(50).max(2);
    let mut stats: Vec<(bool, String, String)> = Vec::new(); // (is_mean, prefix, suffix)
    for sopt in sub_opts(opts, "statistic", 3) {
        let (kind, rest) = sopt.split_once(',').unwrap_or((&sopt, ""));
        let kind = kind.trim().to_lowercase();
        let is_mean = match kind.as_str() {
            "mean" => true,
            "stddev" | "sd" => false,
            o => return Err(rc_err(198, format!("statistic {o} invalid; use mean or stddev"))),
        };
        let pre = sub_opts(rest, "prefix", 1).first().cloned().unwrap_or_default().trim().to_string();
        let suf = sub_opts(rest, "suffix", 1).first().cloned().unwrap_or_default().trim().to_string();
        if pre.is_empty() && suf.is_empty() {
            return Err(rc_err(198, "statistic() requires prefix() or suffix()"));
        }
        stats.push((is_mean, pre, suf));
    }
    if stats.is_empty() {
        return Err(rc_err(198, "simulate() requires statistic()"));
    }
    for (_, p, sfx) in &stats {
        for v in endog {
            if ds.get_var(&format!("{p}{v}{sfx}")).is_some() {
                return Err(rc_err(110, format!("variable {p}{v}{sfx} already defined")));
            }
        }
    }
    let ne = endog.len();
    // per-equation parameter source and error scale
    let sources: Vec<Option<Par>> = model.iter().map(|e| if e.kind == "identity" { None } else { load_par(st, &e.source) }).collect();
    let sig: Vec<f64> = model
        .iter()
        .zip(&sources)
        .map(|(e, p)| p.as_ref().and_then(|p| p.deps.iter().position(|d| *d == e.endog).map(|j| p.sig.get(j).copied().unwrap_or(0.0))).unwrap_or(0.0))
        .collect();
    // in-sample residuals of stochastic equations (for `residuals`)
    let mut res_pool: Vec<Vec<f64>> = vec![Vec::new(); ne];
    if resid {
        let ident = |v: &str| v.to_string();
        for (k, e) in model.iter().enumerate() {
            if sources[k].is_none() {
                continue;
            }
            let ex = ferrostat_parser::parser::parse_expression(&substitute(&e.expr, endog, &ident, false)).map_err(|er| rc_err(198, er.message))?;
            for &i in order {
                if tc[i] >= begin || is_missing(ycols[k][i]) {
                    continue;
                }
                let f = crate::eval::Env::with_store(ds, st).eval(&ex, i).and_then(|v| v.num()).unwrap_or(SYSMISS);
                if !is_missing(f) {
                    res_pool[k].push(ycols[k][i] - f);
                }
            }
            if res_pool[k].is_empty() {
                return Err(rc_err(2000, format!("no in-sample residuals for {}", e.endog)));
            }
        }
    }
    // Cholesky factors of the coefficient variances, by source
    let mut chol: std::collections::BTreeMap<String, Matrix> = std::collections::BTreeMap::new();
    if betas {
        for (e, p) in model.iter().zip(&sources) {
            if let Some(p) = p {
                if chol.contains_key(&e.source) || p.v.is_empty() {
                    continue;
                }
                let k = p.b.len();
                let mut vm = Matrix::zeros(k, k);
                vm.data = p.v.clone();
                // keep only coefficients with positive variance
                let keep: Vec<usize> = (0..k).filter(|&j| vm.get(j, j) > 0.0).collect();
                let mut sub = Matrix::zeros(keep.len(), keep.len());
                for (a, &ja) in keep.iter().enumerate() {
                    for (b2, &jb) in keep.iter().enumerate() {
                        sub.data[a * keep.len() + b2] = vm.get(ja, jb);
                    }
                }
                let l = sub.cholesky().unwrap_or_else(|_| Matrix::from_diag(&sub.diag().iter().map(|x| x.max(0.0).sqrt()).collect::<Vec<_>>()));
                let mut full = Matrix::zeros(k, keep.len());
                for (a, &ja) in keep.iter().enumerate() {
                    for c in 0..keep.len() {
                        full.data[ja * keep.len() + c] = l.get(a, c);
                    }
                }
                chol.insert(e.source.clone(), full);
            }
        }
    }
    let np = periods.len();
    let mut sum = vec![vec![0.0; np]; ne];
    let mut sum2 = vec![vec![0.0; np]; ne];
    let mut cnt = vec![vec![0usize; np]; ne];
    let tmp: Vec<String> = (0..ne).map(|k| format!("__fcsim{k}")).collect();
    let tmpf = |v: &str| endog.iter().position(|e| e == v).map(|k| format!("__fcsim{k}")).unwrap_or_else(|| v.to_string());
    for (k, v) in endog.iter().enumerate() {
        let col = (0..ds.n_obs).map(|i| if tc[i] >= begin && !(actuals && !is_missing(ycols[k][i])) { SYSMISS } else { ycols[k][i] }).collect();
        let _ = v;
        ds.add_variable(util::new_double(&tmp[k], col)).map_err(|e| rc_err(198, e))?;
    }
    let row_pos: std::collections::HashMap<usize, usize> = periods.iter().enumerate().map(|(p, &r)| (r, p)).collect();
    let result = (|| -> Result<(), String> {
        for _ in 0..reps {
            // draw coefficients
            let mut drawn: std::collections::BTreeMap<String, Vec<f64>> = std::collections::BTreeMap::new();
            for (src, l) in &chol {
                let p = sources.iter().flatten().zip(model).find(|(_, e)| &e.source == src).map(|(p, _)| p.clone()).unwrap();
                let z: Vec<f64> = {
                    let mut rng = st.rng.borrow_mut();
                    (0..l.cols).map(|_| rng.normal()).collect()
                };
                let b: Vec<f64> = (0..p.b.len()).map(|j| p.b[j] + (0..l.cols).map(|c| l.get(j, c) * z[c]).sum::<f64>()).collect();
                drawn.insert(src.clone(), b);
            }
            let rep_exprs: Vec<Expr> = model
                .iter()
                .enumerate()
                .map(|(k, e)| match (&sources[k], drawn.get(&e.source)) {
                    (Some(p), Some(b)) => {
                        let eqs = eqs_from_b(&p.cols, b, &p.deps, &p.cmd);
                        let text = eqs.into_iter().find(|(d, _)| *d == e.endog).map(|x| x.1).unwrap_or_else(|| e.expr.clone());
                        ferrostat_parser::parser::parse_expression(&substitute(&text, endog, &tmpf, static_)).map_err(|er| rc_err(198, er.message))
                    }
                    _ => ferrostat_parser::parser::parse_expression(&substitute(&e.expr, endog, &tmpf, static_)).map_err(|er| rc_err(198, er.message)),
                })
                .collect::<Result<_, _>>()?;
            let _ = exprs;
            let rep_adj: Vec<(usize, Expr)> = adj.to_vec();
            let rng = &st.rng;
            let draws: Vec<Vec<f64>> = (0..ne)
                .map(|k| {
                    (0..np)
                        .map(|_| {
                            let mut r = rng.borrow_mut();
                            if errors {
                                sig[k] * r.normal()
                            } else if resid && !res_pool[k].is_empty() {
                                res_pool[k][((r.uniform() * res_pool[k].len() as f64) as usize).min(res_pool[k].len() - 1)]
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let mut shock = |k: usize, i: usize| row_pos.get(&i).map(|&p| draws[k][p]).unwrap_or(0.0);
            run_periods(ds, st, &rep_exprs, &rep_adj, &tmp, periods, actuals, ycols, &mut shock)?;
            for k in 0..ne {
                let col = util::col(ds, &tmp[k])?;
                for (p, &i) in periods.iter().enumerate() {
                    if !is_missing(col[i]) {
                        sum[k][p] += col[i];
                        sum2[k][p] += col[i] * col[i];
                        cnt[k][p] += 1;
                    }
                }
            }
        }
        Ok(())
    })();
    for t in &tmp {
        let _ = ds.drop_variable(t);
    }
    result?;
    let mut made = Vec::new();
    for (is_mean, pre, suf) in &stats {
        for (k, v) in endog.iter().enumerate() {
            let mut col = vec![SYSMISS; ds.n_obs];
            for p in 0..np {
                let n = cnt[k][p] as f64;
                if n < 1.0 {
                    continue;
                }
                let m = sum[k][p] / n;
                col[periods[p]] = if *is_mean { m } else if n > 1.0 { ((sum2[k][p] - n * m * m) / (n - 1.0)).max(0.0).sqrt() } else { SYSMISS };
            }
            let nm = format!("{pre}{v}{suf}");
            ds.add_variable(util::new_double(&nm, col)).map_err(|e| rc_err(198, e))?;
            made.push(nm);
        }
    }
    Ok(made)
}

fn solve(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let name = need_model(st)?;
    let model = eqs(st);
    if model.is_empty() {
        return Err(rc_err(198, "the forecast model has no equations"));
    }
    let tvar = tsops::time_var(st)?;
    let (prefix, suffix) = match (cmd.opt("prefix", 3), cmd.opt("suffix", 3)) {
        (None, None) => ("f_".to_string(), String::new()),
        (p, s) => (p.unwrap_or("").trim().to_string(), s.unwrap_or("").trim().to_string()),
    };
    let static_ = cmd.has_opt("static", 4);
    let actuals = cmd.has_opt("actuals", 3);
    let endog: Vec<String> = model.iter().map(|e| e.endog.clone()).collect();
    let fname = |v: &str| format!("{prefix}{v}{suffix}");
    for v in &endog {
        if ds.get_var(&fname(v)).is_some() {
            return Err(rc_err(110, format!("variable {} already defined", fname(v))));
        }
    }
    let tc = util::col(ds, &tvar)?;
    let mut order: Vec<usize> = (0..ds.n_obs).filter(|&i| !is_missing(tc[i])).collect();
    order.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    let ycols: Vec<Vec<f64>> = endog.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let eval_time = |s: &str, ds: &DataSet, st: &Store| -> Result<f64, String> {
        match s.trim().parse::<f64>() {
            Ok(v) => Ok(v),
            Err(_) => {
                let ex = ferrostat_parser::parser::parse_expression(s).map_err(|e| rc_err(198, e.message))?;
                crate::eval::Env::with_store(ds, st).eval(&ex, 0)?.num()
            }
        }
    };
    let begin = match cmd.opt("begin", 3) {
        Some(b) => eval_time(b, ds, st)?,
        None => {
            let last_full = order.iter().filter(|&&i| ycols.iter().all(|c| !is_missing(c[i]))).map(|&i| tc[i]).fold(f64::NEG_INFINITY, f64::max);
            if !last_full.is_finite() {
                return Err(rc_err(2000, "no observations with all endogenous variables"));
            }
            last_full + 1.0
        }
    };
    let tmax = order.last().map(|&i| tc[i]).unwrap_or(SYSMISS);
    let end = match (cmd.opt("end", 3), util::opt_num(cmd, "periods", 3)?) {
        (Some(e), _) => eval_time(e, ds, st)?,
        (None, Some(p)) => begin + p - 1.0,
        _ => tmax,
    };
    if begin > end || begin > tmax {
        return Err(rc_err(198, "no periods to forecast; use tsappend to add observations"));
    }
    // forecast variables start as the actual values
    for (k, v) in endog.iter().enumerate() {
        let mut col = ycols[k].clone();
        for &i in &order {
            if tc[i] >= begin && !(actuals && !is_missing(col[i])) {
                col[i] = SYSMISS;
            }
        }
        let mut var = util::new_double(&fname(v), col);
        var.format = ds.get_var(v).map(|x| x.format.clone()).unwrap_or_else(|| "%9.0g".into());
        ds.add_variable(var).map_err(|e| rc_err(198, e))?;
    }
    let adjust = list_of(st, K_ADJ);
    let parse = |s: &str| ferrostat_parser::parser::parse_expression(s).map_err(|e| rc_err(198, format!("{}: {s}", e.message)));
    let exprs: Vec<_> = model.iter().map(|e| parse(&substitute(&e.expr, &endog, &fname, static_))).collect::<Result<_, _>>()?;
    let adj: Vec<(usize, _)> = adjust
        .iter()
        .map(|a| {
            let (l, r) = a.split_once(US).unwrap_or((a, ""));
            let k = endog.iter().position(|e| e == l).unwrap_or(0);
            parse(&substitute(r, &endog, &fname, false)).map(|x| (k, x))
        })
        .collect::<Result<_, _>>()?;
    let periods: Vec<usize> = order.iter().copied().filter(|&i| tc[i] >= begin && tc[i] <= end).collect();
    let targets: Vec<String> = endog.iter().map(|v| fname(v)).collect();
    let mut log = Vec::new();
    for i in run_periods(ds, st, &exprs, &adj, &targets, &periods, actuals, &ycols, &mut |_, _| 0.0)? {
        log.push(format!("{}: model did not converge", tsops::time_label(ds, &tvar, tc[i])));
    }
    let mut sim_vars: Vec<String> = Vec::new();
    if let Some(sim) = cmd.opt("simulate", 3) {
        sim_vars = simulate(ds, st, sim, &model, &endog, &exprs, &adj, &periods, actuals, &ycols, static_, begin, &tc, &order)?;
    }
    let mut created: Vec<String> = endog.iter().map(|v| fname(v)).collect();
    created.extend(sim_vars);
    save_list(st, K_VARS, &created);
    let mut out = vec![String::new()];
    let head = format!("Computing {} forecasts for model {name}.", if static_ { "static" } else { "dynamic" });
    out.push(head.clone());
    out.push("-".repeat(head.len()));
    out.push(String::new());
    out.push(format!("Starting period:  {}", tsops::time_label(ds, &tvar, begin)));
    out.push(format!("Ending period:    {}", tsops::time_label(ds, &tvar, end.min(tmax))));
    if !prefix.is_empty() {
        out.push(format!("Forecast prefix:  {prefix}"));
    } else {
        out.push(format!("Forecast suffix:  {suffix}"));
    }
    out.push(String::new());
    for &i in &periods {
        out.push(format!("{}: {}", tsops::time_label(ds, &tvar, tc[i]), ".".repeat(endog.len().min(50))));
    }
    out.extend(log);
    out.push(String::new());
    out.push(format!("Forecast {} variable{} spanning {} period{}.", endog.len(), if endog.len() == 1 { "" } else { "s" }, periods.len(), if periods.len() == 1 { "" } else { "s" }));
    out.push("-".repeat(head.len()));
    st.data_changed = true;
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitution_respects_static() {
        let en = vec!["y".to_string(), "c".to_string()];
        let f = |v: &str| format!("f_{v}");
        assert_eq!(substitute("c + L.y + exp(yy)", &en, &f, false), "f_c + L.f_y + exp(yy)");
        assert_eq!(substitute("c + L.y", &en, &f, true), "f_c + L.y");
        assert_eq!(term_expr("L2.x").unwrap(), "(L2.x)");
        assert_eq!(term_expr("_cons").unwrap(), "1");
        assert!(term_expr("1b.g").is_none());
    }
}
