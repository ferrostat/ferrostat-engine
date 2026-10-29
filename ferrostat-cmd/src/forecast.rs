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
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
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
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for (j, c) in b.colnames.iter().enumerate() {
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
        let bj = b.data[j];
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
    Ok(out.into_iter().map(|(d, p)| (d, p.join(" + "))).collect())
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
                let name = if rest.is_empty() { "model".to_string() } else { rest.to_string() };
                st.settings.insert(K_NAME.into(), name.clone());
                Ok(format!("  Forecast model {name} started."))
            }
            "estimates" | "est" => {
                need_model(st)?;
                let name = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "estimation-results name required"))?;
                let res = if name == "." { st.e.clone() } else { st.estimates.get(name).cloned().ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))? };
                let new = eqs_from_results(&res)?;
                let mut v = eqs(st);
                for (d, _) in &new {
                    if v.iter().any(|e| e.endog == *d) {
                        return Err(rc_err(110, format!("an equation for {d} is already in the model")));
                    }
                }
                let n = new.len();
                let names: Vec<String> = new.iter().map(|x| x.0.clone()).collect();
                v.extend(new.into_iter().map(|(d, x)| Eqn { kind: "estimates".into(), endog: d, expr: x, source: name.to_string() }));
                save_eqs(st, &v);
                Ok(format!("  Added estimation results from {}.\n  Forecast model {} now contains {} endogenous variable{}.", res.macros.get("cmd").cloned().unwrap_or_default(), need_model(st)?, v.len(), if v.len() == 1 { "" } else { "s" })).map(|s| {
                    let _ = (n, names);
                    s
                })
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
                    if e.kind == "estimates" {
                        if !seen.contains(&e.source) {
                            out.push(format!("    forecast estimates {}", e.source));
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
    let mut log = Vec::new();
    for &i in &periods {
        let mut converged = false;
        for _ in 0..500 {
            let mut change = 0.0f64;
            for (k, ex) in exprs.iter().enumerate() {
                if actuals && !is_missing(ycols[k][i]) {
                    continue;
                }
                let mut v = crate::eval::Env::with_store(ds, st).eval(ex, i)?.num()?;
                set_val(ds, &fname(&endog[k]), i, v);
                for (ak, aex) in &adj {
                    if *ak == k {
                        v = crate::eval::Env::with_store(ds, st).eval(aex, i)?.num()?;
                        set_val(ds, &fname(&endog[k]), i, v);
                    }
                }
                let old = st.scalars.get(&format!("__fc_prev{k}")).copied().unwrap_or(SYSMISS);
                let d = if is_missing(v) && is_missing(old) { 0.0 } else if is_missing(v) || is_missing(old) { 1.0 } else { (v - old).abs() / (old.abs() + 1.0) };
                change = change.max(d);
                st.scalars.insert(format!("__fc_prev{k}"), v);
            }
            if change < 1e-10 {
                converged = true;
                break;
            }
        }
        for k in 0..endog.len() {
            st.scalars.remove(&format!("__fc_prev{k}"));
        }
        if !converged {
            log.push(format!("{}: model did not converge", tsops::time_label(ds, &tvar, tc[i])));
        }
    }
    let created: Vec<String> = endog.iter().map(|v| fname(v)).collect();
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
