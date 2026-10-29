//! Post-estimation: predict, test, testparm, lincom, estimates, margins, estat.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

fn need_e(st: &Store) -> Result<(&Matrix, &Matrix, String), String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let cmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    Ok((b, v, cmd))
}

/// Value of the regressor named by a coefficient for one observation.
pub fn coef_value(env: &Env, name: &str, row: usize) -> Result<f64, String> {
    let name = name.rsplit(':').next().unwrap_or(name);
    if name == "_cons" {
        return Ok(1.0);
    }
    if name.starts_with('/') {
        return Ok(0.0);
    }
    let mut prod = 1.0;
    for part in name.split('#') {
        let part = part.strip_prefix("c.").unwrap_or(part);
        let part = part.strip_prefix("o.").unwrap_or(part);
        let v = match part.split_once('.') {
            Some((lvl, var)) if lvl.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => {
                let l: f64 = lvl.trim_end_matches(['b', 'o', 'n']).parse().unwrap();
                let x = env.eval(&ferrostat_parser::Expr::Var(var.to_string()), row)?.num()?;
                if is_missing(x) { SYSMISS } else if x == l { 1.0 } else { 0.0 }
            }
            _ => {
                let e = parse_expression(part).map_err(|e| rc_err(198, e.message))?;
                env.eval(&e, row)?.num()?
            }
        };
        if is_missing(v) {
            return Ok(SYSMISS);
        }
        prod *= v;
    }
    Ok(prod)
}

/// Linear prediction for equation `eq` (None = first/only equation).
pub fn xb_row(env: &Env, b: &Matrix, eq: Option<&str>, row: usize) -> Result<f64, String> {
    let mut s = 0.0;
    let first_eq = b.colnames.first().and_then(|c| c.split_once(':').map(|x| x.0.to_string()));
    for (j, name) in b.colnames.iter().enumerate() {
        let coef = b.get(0, j);
        let (e, _) = name.split_once(':').map(|(a, c)| (Some(a), c)).unwrap_or((None, name));
        let want = eq.map(String::from).or(first_eq.clone());
        if e.map(String::from) != want && !(e.is_none() && want.is_none()) {
            continue;
        }
        if name.starts_with("/:") || e == Some("/") {
            continue;
        }
        if coef == 0.0 {
            continue;
        }
        let x = coef_value(env, name, row)?;
        if is_missing(x) {
            return Ok(SYSMISS);
        }
        s += coef * x;
    }
    Ok(s)
}

/// x' V x for one observation (variance of the linear prediction).
fn stdp_row(env: &Env, b: &Matrix, v: &Matrix, row: usize) -> Result<f64, String> {
    let x: Vec<f64> = b.colnames.iter().map(|n| if n.starts_with('/') { Ok(0.0) } else { coef_value(env, n, row) }).collect::<Result<_, _>>()?;
    if x.iter().any(|v| is_missing(*v)) {
        return Ok(SYSMISS);
    }
    let k = x.len();
    let mut s = 0.0;
    for i in 0..k {
        for j in 0..k {
            s += x[i] * v[(i, j)] * x[j];
        }
    }
    Ok(s.max(0.0).sqrt())
}

/// Default prediction of the current model (used by predict and margins).
pub fn default_prediction(st: &Store, env: &Env, b: &Matrix, row: usize) -> Result<f64, String> {
    let cmd = st.e.macros.get("cmd").map(String::as_str).unwrap_or("");
    let xb = xb_row(env, b, None, row)?;
    if is_missing(xb) {
        return Ok(SYSMISS);
    }
    let off = offset_value(st, env, row)?;
    Ok(match cmd {
        "logit" | "logistic" => dist::invlogit(xb),
        "probit" => dist::normal(xb),
        "cloglog" => 1.0 - (-(xb.exp())).exp(),
        "poisson" | "nbreg" => (xb + off).exp(),
        "glm" => glm_mu(st, xb),
        _ => xb,
    })
}

fn offset_value(st: &Store, env: &Env, row: usize) -> Result<f64, String> {
    match st.e.macros.get("offset") {
        Some(o) => {
            let e = parse_expression(o).map_err(|e| rc_err(198, e.message))?;
            env.eval(&e, row)?.num()
        }
        None => Ok(0.0),
    }
}

fn glm_mu(st: &Store, eta: f64) -> f64 {
    match st.e.macros.get("link").map(String::as_str).unwrap_or("identity") {
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

pub struct Predict;

impl CommandHandler for Predict {
    fn name(&self) -> &'static str {
        "predict"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if let Some(out) = crate::mv_post::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if let Some(out) = crate::sp_reg::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if let Some(r) = crate::ts_more::predict(ds, st, cmd) {
            return r;
        }
        let (b, v, ecmd) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        if ecmd == "arima" {
            return crate::arima::predict(ds, st, cmd);
        }
        if let Some(out) = crate::choice_mixed::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if let Some(out) = crate::xtnl::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if let Some(out) = crate::stcox::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if let Some(out) = crate::streg::predict(ds, st, cmd)? {
            return Ok(out);
        }
        if ecmd == "reghdfe" {
            if let Some(out) = crate::reghdfe::predict(ds, st, cmd)? {
                return Ok(out);
            }
        }
        let words: Vec<&str> = cmd.main.split_whitespace().collect();
        let (ty, names): (Option<VarType>, Vec<String>) = match words.first().and_then(|w| VarType::from_type_str(w)) {
            Some(t) => (Some(t), words[1..].iter().map(|s| s.to_string()).collect()),
            None => (None, words.iter().map(|s| s.to_string()).collect()),
        };
        if names.is_empty() {
            return Err(rc_err(100, "new variable name required"));
        }
        let stub = names.len() == 1 && names[0].ends_with('*');
        for n in &names {
            if !stub {
                crate::varlist::check_name(n)?;
                if ds.var_index.contains_key(n) {
                    return Err(rc_err(110, format!("variable {n} already defined")));
                }
            }
        }
        let mask = util::touse(ds, st, cmd)?;
        let opt_names = [
            "xb", "residuals", "stdp", "stdf", "stdr", "rstandard", "rstudent", "leverage", "hat", "cooksd", "pr", "p", "n", "ir", "mu", "eta", "u", "e", "ue", "xbu", "score", "deviance", "pearson",
        ];
        let stat = cmd
            .opts
            .iter()
            .map(|(n, _)| n.to_lowercase())
            .find(|n| opt_names.iter().any(|o| o.starts_with(n.as_str()) && n.len() >= 1))
            .unwrap_or_default();
        let stat = match stat.as_str() {
            "" => String::new(),
            s => opt_names.iter().find(|o| o.starts_with(s)).unwrap().to_string(),
        };
        let mut msg = String::new();
        let n = ds.n_obs;
        let env = Env::with_store(ds, st);
        let mut results: Vec<(String, Vec<f64>)> = Vec::new();
        let is_binary = matches!(ecmd.as_str(), "logit" | "logistic" | "probit" | "cloglog");
        let is_count = matches!(ecmd.as_str(), "poisson" | "nbreg");
        let ordered = matches!(ecmd.as_str(), "ologit" | "oprobit");
        let mlogit = ecmd == "mlogit";
        if matches!(ecmd.as_str(), "zip" | "zinb") {
            if stat.is_empty() {
                msg = "(option n assumed; predicted number of events)".into();
            }
            let vals: Vec<f64> = (0..n).map(|i| if mask[i] { crate::zip::zi_prediction(st, &env, &stat, i) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?;
            results.push((names[0].clone(), vals));
        } else if ordered || mlogit {
            let cats: Vec<f64> = st.e.matrices.get(if mlogit { "out" } else { "cat" }).map(|m| m.data.clone()).unwrap_or_default();
            if stat == "xb" {
                let vals: Vec<f64> = (0..n).map(|i| if mask[i] { xb_row(&env, &b, if mlogit { None } else { None }, i) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?;
                results.push((names[0].clone(), vals));
            } else {
                let outcome = util::opt_num(cmd, "outcome", 3)?;
                let want: Vec<usize> = match outcome {
                    Some(o) => vec![cats.iter().position(|&c| c == o).ok_or_else(|| rc_err(198, "outcome() not found"))?],
                    None => (0..cats.len()).collect(),
                };
                let newnames: Vec<String> = if stub {
                    let s = names[0].trim_end_matches('*');
                    (1..=want.len()).map(|i| format!("{s}{i}")).collect()
                } else {
                    names.clone()
                };
                if newnames.len() != want.len() {
                    if names.len() == 1 && want.len() > 1 && outcome.is_none() {
                        return Err(rc_err(198, format!("{} new variables required (one per outcome), or specify outcome()", want.len())));
                    }
                }
                let probit = ecmd == "oprobit";
                for (k, &c) in want.iter().enumerate() {
                    let mut vals = vec![SYSMISS; n];
                    for i in 0..n {
                        if !mask[i] {
                            continue;
                        }
                        vals[i] = if mlogit {
                            let mut eta = Vec::new();
                            for cv in &cats {
                                eta.push(xb_row(&env, &b, Some(&fmt_g(*cv, 12)), i)?);
                            }
                            if eta.iter().any(|x| is_missing(*x)) {
                                SYSMISS
                            } else {
                                let mx = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let den: f64 = eta.iter().map(|x| (x - mx).exp()).sum();
                                (eta[c] - mx).exp() / den
                            }
                        } else {
                            let xb = xb_row(&env, &b, None, i)?;
                            if is_missing(xb) {
                                SYSMISS
                            } else {
                                let cut = |j: usize| b.get(0, b.col_index(&format!("/:cut{}", j + 1)).unwrap());
                                let f = |z: f64| if probit { dist::normal(z) } else { dist::invlogit(z) };
                                let hi = if c < cats.len() - 1 { f(cut(c) - xb) } else { 1.0 };
                                let lo = if c > 0 { f(cut(c - 1) - xb) } else { 0.0 };
                                hi - lo
                            }
                        };
                    }
                    results.push((newnames.get(k).cloned().unwrap_or_else(|| format!("{}{}", names[0], k + 1)), vals));
                }
            }
        } else {
            let stat = if stat.is_empty() {
                if is_binary {
                    msg = "(option pr assumed; Pr(".to_string() + st.e.macros.get("depvar").map(String::as_str).unwrap_or("") + "))";
                    "pr".to_string()
                } else if is_count {
                    msg = "(option n assumed; predicted number of events)".into();
                    "n".to_string()
                } else if ecmd == "glm" {
                    msg = "(option mu assumed; predicted mean)".into();
                    "mu".into()
                } else {
                    msg = "(option xb assumed; fitted values)".into();
                    "xb".to_string()
                }
            } else {
                stat
            };
            let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
            let rmse = st.e.get("rmse").unwrap_or(SYSMISS);
            let df_r = st.e.get("df_r").unwrap_or(SYSMISS);
            let xtxinv = st.e.matrices.get("XtXinv").cloned();
            let mut vals = vec![SYSMISS; n];
            for i in 0..n {
                if !mask[i] {
                    continue;
                }
                let xb = xb_row(&env, &b, None, i)?;
                if is_missing(xb) {
                    continue;
                }
                let yv = if dep.is_empty() { SYSMISS } else { env.eval(&parse_expression(&dep).map_err(|e| rc_err(198, e.message))?, i)?.num()? };
                let off = offset_value(st, &env, i)?;
                let lev = || -> Result<f64, String> {
                    let x: Vec<f64> = b.colnames.iter().map(|nm| coef_value(&env, nm, i)).collect::<Result<_, _>>()?;
                    let m = xtxinv.as_ref().ok_or_else(|| rc_err(198, "leverage not available"))?;
                    let k = x.len();
                    Ok((0..k).map(|a| (0..k).map(|c| x[a] * m[(a, c)] * x[c]).sum::<f64>()).sum())
                };
                vals[i] = match stat.as_str() {
                    "xb" => xb + off,
                    "stdp" => stdp_row(&env, &b, &v, i)?,
                    "stdf" => {
                        let s = stdp_row(&env, &b, &v, i)?;
                        (s * s + rmse * rmse).sqrt()
                    }
                    "residuals" | "e" => {
                        if is_missing(yv) {
                            SYSMISS
                        } else if ecmd == "xtreg" && stat == "e" {
                            yv - xb
                        } else {
                            yv - default_prediction(st, &env, &b, i)?
                        }
                    }
                    "pr" | "p" => match ecmd.as_str() {
                        "logit" | "logistic" => dist::invlogit(xb),
                        "probit" => dist::normal(xb),
                        "cloglog" => 1.0 - (-(xb.exp())).exp(),
                        _ => return Err(rc_err(198, "option pr not allowed")),
                    },
                    "n" => (xb + off).exp(),
                    "ir" => xb.exp(),
                    "mu" => glm_mu(st, xb),
                    "eta" => xb,
                    "leverage" | "hat" => lev()?,
                    "stdr" => rmse * (1.0 - lev()?).max(0.0).sqrt(),
                    "rstandard" => (yv - xb) / (rmse * (1.0 - lev()?).sqrt()),
                    "rstudent" => {
                        let h = lev()?;
                        let r = yv - xb;
                        let s2 = rmse * rmse;
                        let s2i = (df_r * s2 - r * r / (1.0 - h)) / (df_r - 1.0);
                        r / (s2i * (1.0 - h)).sqrt()
                    }
                    "cooksd" => {
                        let h = lev()?;
                        let r = yv - xb;
                        let k = st.e.get("rank").unwrap_or(b.cols as f64);
                        r * r * h / (k * rmse * rmse * (1.0 - h).powi(2))
                    }
                    "deviance" | "pearson" | "score" => {
                        if is_missing(yv) {
                            SYSMISS
                        } else {
                            let p = default_prediction(st, &env, &b, i)?;
                            match (stat.as_str(), is_binary) {
                                ("pearson", true) => (yv - p) / (p * (1.0 - p)).sqrt(),
                                ("deviance", true) => {
                                    let yb = if yv != 0.0 { 1.0 } else { 0.0 };
                                    let d = if yb == 1.0 { -2.0 * p.ln() } else { -2.0 * (1.0 - p).ln() };
                                    (yb - p).signum() * d.sqrt()
                                }
                                ("score", true) => (if yv != 0.0 { 1.0 } else { 0.0 }) - p,
                                ("pearson", _) => (yv - p) / p.sqrt(),
                                ("score", _) => yv - p,
                                _ => {
                                    let d = if yv > 0.0 { 2.0 * (yv * (yv / p).ln() - (yv - p)) } else { 2.0 * p };
                                    (yv - p).signum() * d.sqrt()
                                }
                            }
                        }
                    }
                    "u" | "ue" | "xbu" => {
                        // xtreg: u_i = group mean residual
                        let ivar = st.e.macros.get("ivar").cloned().ok_or_else(|| rc_err(198, "option not allowed"))?;
                        let g = util::col(ds, &ivar)?;
                        let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; n]);
                        let mut sum = 0.0;
                        let mut cnt = 0.0;
                        let yc = util::col(ds, &dep)?;
                        for j in 0..n {
                            if sample[j] && g[j] == g[i] {
                                let xbj = xb_row(&env, &b, None, j)?;
                                sum += yc[j] - xbj;
                                cnt += 1.0;
                            }
                        }
                        let u = sum / cnt;
                        match stat.as_str() {
                            "u" => u,
                            "ue" => yv - xb,
                            _ => xb + u,
                        }
                    }
                    other => return Err(rc_err(198, format!("option {other} not allowed"))),
                };
            }
            results.push((names[0].clone(), vals));
        }
        drop(env);
        let mut nmiss = 0;
        for (name, vals) in results {
            let t = ty.unwrap_or(st.default_type);
            let mut var = Variable::new(&name, t);
            nmiss = nmiss.max(vals.iter().filter(|x| is_missing(**x)).count());
            var.data = vals.iter().map(|&x| crate::data::store_num(t, x)).collect();
            var.label = format!("{} prediction", ecmd);
            ds.add_variable(var)?;
        }
        st.data_changed = true;
        let mut out = Vec::new();
        if !msg.is_empty() {
            out.push(msg);
        }
        if nmiss > 0 {
            out.push(format!("({} missing value{} generated)", crate::util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Linear constraints (test / lincom)
// ---------------------------------------------------------------------------

/// `[eq]name` is shorthand for `_b[eq:name]`.
fn eq_shorthand(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let ident = |ch: char| ch.is_alphanumeric() || ch == '_' || ch == '.' || ch == '#';
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        let after_ident = i > 0 && (ident(c[i - 1]) || c[i - 1] == ']');
        if c[i] == '[' && !after_ident {
            if let Some(j) = c[i..].iter().position(|&x| x == ']').map(|j| i + j) {
                let k = (j + 1..c.len()).find(|&k| !ident(c[k])).unwrap_or(c.len());
                if k > j + 1 {
                    let eq: String = c[i + 1..j].iter().collect();
                    let name: String = c[j + 1..k].iter().collect();
                    out.push_str(&format!("_b[{}:{name}]", eq.trim()));
                    i = k;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// Wrap coefficient names in `_b[ ]` so the constraint can be evaluated.
pub(crate) fn wrap_coefs(text: &str, b: &Matrix) -> String {
    let text = &eq_shorthand(text);
    let mut out = String::new();
    let mut tok = String::new();
    let mut in_brack = 0;
    let flush = |tok: &mut String, out: &mut String| {
        if !tok.is_empty() {
            let is_num = tok.parse::<f64>().is_ok();
            if !is_num && crate::eval::find_coef(b, tok).is_some() {
                out.push_str(&format!("_b[{tok}]"));
            } else {
                out.push_str(tok);
            }
            tok.clear();
        }
    };
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '[' {
            in_brack += 1;
        }
        if c == ']' {
            in_brack -= 1;
        }
        if in_brack == 0 && (c.is_alphanumeric() || c == '_' || c == '.' || c == '#' || c == ':' || c == '@') {
            tok.push(c);
        } else if in_brack > 0 {
            if !tok.is_empty() && c == '[' {
                // `_b[` prefix: keep as is
                out.push_str(&tok);
                tok.clear();
            }
            out.push(c);
        } else {
            flush(&mut tok, &mut out);
            out.push(c);
        }
        i += 1;
    }
    flush(&mut tok, &mut out);
    out
}

/// Evaluate `expr(b)` for a linear expression in the coefficients.
fn linear_form(text: &str, b: &Matrix) -> Result<(Vec<f64>, f64), String> {
    let wrapped = wrap_coefs(text, b);
    let e = parse_expression(&wrapped).map_err(|e| rc_err(198, format!("{text}: {}", e.message)))?;
    let ds = DataSet::new();
    let eval_at = |bv: &[f64]| -> Result<f64, String> {
        let mut st = Store::new();
        st.e.set_mat("b", Matrix::row_vector(bv).with_names(vec!["y1".into()], b.colnames.clone()));
        st.e.set_mat("V", Matrix::zeros(bv.len(), bv.len()));
        let env = Env::with_store(&ds, &st);
        env.eval(&e, 0)?.num()
    };
    let k = b.cols;
    let zero = vec![0.0; k];
    let c0 = eval_at(&zero)?;
    let mut r = vec![0.0; k];
    for j in 0..k {
        let mut ej = zero.clone();
        ej[j] = 1.0;
        r[j] = eval_at(&ej)? - c0;
    }
    Ok((r, c0))
}

/// Parse `test` arguments into constraint texts "lhs - (rhs)".
fn constraints(main: &str, b: &Matrix) -> Result<Vec<(String, String)>, String> {
    let main = main.trim();
    let mut out = Vec::new();
    let groups: Vec<String> = if main.starts_with('(') {
        ferrostat_parser::parser::split_top_level_words(main).into_iter().map(|g| g.trim().trim_start_matches('(').trim_end_matches(')').to_string()).collect()
    } else {
        vec![main.to_string()]
    };
    for g in groups {
        if let Some(i) = ferrostat_parser::parser::find_top_level(&g, '=') {
            let (lhs, rhs) = (&g[..i], g[i + 1..].trim_start_matches('='));
            // `test a = b = c`
            let parts: Vec<&str> = std::iter::once(lhs).chain(rhs.split('=')).collect();
            for w in parts.windows(2) {
                out.push((format!("{} = {}", w[0].trim(), w[1].trim()), format!("({}) - ({})", w[0], w[1])));
            }
        } else {
            for tok in g.split_whitespace() {
                // factor-variable terms: all matching coefficients
                if let Some((op, var)) = tok.split_once('.') {
                    if op == "i" {
                        for c in &b.colnames {
                            if c.ends_with(&format!(".{var}")) && !c.contains("b.") && !c.contains("o.") {
                                out.push((format!("{c} = 0"), c.clone()));
                            }
                        }
                        continue;
                    }
                }
                if crate::eval::find_coef(b, tok).is_none() {
                    return Err(rc_err(111, format!("{tok} not found")));
                }
                out.push((format!("{tok} = 0"), tok.to_string()));
            }
        }
    }
    Ok(out)
}

pub struct Test;

impl CommandHandler for Test {
    fn name(&self) -> &'static str {
        "test"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v, _) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        let cons = constraints(&cmd.main, &b)?;
        wald_report(st, &b, &v, &cons)
    }
}

pub struct Testparm;

impl CommandHandler for Testparm {
    fn name(&self) -> &'static str {
        "testparm"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v, _) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        let mut cons = Vec::new();
        for tok in cmd.main.split_whitespace() {
            let var = tok.rsplit('.').next().unwrap_or(tok);
            for c in &b.colnames {
                let base = c.rsplit(':').next().unwrap_or(c);
                let matches = base == var || base.ends_with(&format!(".{var}")) && !base.contains("b.") && !base.contains("o.");
                if matches && c != "_cons" {
                    cons.push((format!("{c} = 0"), c.clone()));
                }
            }
        }
        if cons.is_empty() {
            return Err(rc_err(111, "no such variables"));
        }
        wald_report(st, &b, &v, &cons)
    }
}

fn wald_report(st: &mut Store, b: &Matrix, v: &Matrix, cons: &[(String, String)]) -> Result<String, String> {
    let q = cons.len();
    let k = b.cols;
    let mut r = Matrix::zeros(q, k);
    let mut rhs = vec![0.0; q];
    let mut lines = Vec::new();
    for (i, (label, text)) in cons.iter().enumerate() {
        let (row, c0) = linear_form(text, b)?;
        for j in 0..k {
            r[(i, j)] = row[j];
        }
        rhs[i] = -c0;
        lines.push(format!(" ({:>2})  {}", i + 1, label));
    }
    let bvec = Matrix::col_vector(&b.data);
    let rb = r.mul(&bvec)?;
    let diff: Vec<f64> = (0..q).map(|i| rb.get(i, 0) - rhs[i]).collect();
    let rvr = r.mul(v)?.mul(&r.transpose())?;
    let (inv, dropped) = rvr.sym_inverse(1e-12);
    let w: f64 = (0..q).map(|a| (0..q).map(|c| diff[a] * inv[(a, c)] * diff[c]).sum::<f64>()).sum();
    let df = (q - dropped.len()) as f64;
    let mut res = Results::new();
    lines.push(String::new());
    if !dropped.is_empty() {
        for d in &dropped {
            lines.push(format!("       Constraint {} dropped", d + 1));
        }
    }
    if st.e.macros.get("prefix").map(String::as_str) == Some("svy") && st.e.get("df_r").is_some() {
        // survey estimates: adjusted Wald test with the design df
        let d = st.e.get("df_r").unwrap();
        let df2 = d - df + 1.0;
        let f = (d - df + 1.0) / (d * df) * w;
        let p = dist::ftail(df, df2, f);
        lines.insert(0, "Adjusted Wald test\n".into());
        lines.push(format!("       F({:>3}, {:>5}) = {:>8.2}", df, df2, f));
        lines.push(format!("            Prob > F = {:>10.4}", p));
        res.set("F", f);
        res.set("df_r", df2);
        res.set("p", p);
    } else if let Some(df_r) = st.e.get("df_r").filter(|_| !matches!(st.e.macros.get("cmd").map(String::as_str), Some("logit" | "probit" | "poisson"))) {
        let f = w / df;
        let p = dist::ftail(df, df_r, f);
        lines.push(format!("       F({:>3}, {:>5}) = {:>8.2}", df, df_r, f));
        lines.push(format!("            Prob > F = {:>10.4}", p));
        res.set("F", f);
        res.set("df_r", df_r);
        res.set("p", p);
    } else {
        let p = dist::chi2tail(df, w);
        lines.push(format!("           chi2({:>3}) = {:>8.2}", df, w));
        lines.push(format!("         Prob > chi2 = {:>10.4}", p));
        res.set("chi2", w);
        res.set("p", p);
    }
    res.set("df", df);
    res.set("drop", if dropped.is_empty() { 0.0 } else { 1.0 });
    st.r = res;
    Ok(lines.join("\n"))
}

pub struct Lincom;

impl CommandHandler for Lincom {
    fn name(&self) -> &'static str {
        "lincom"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v, ecmd) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        let (row, c0) = linear_form(&cmd.main, &b)?;
        let est: f64 = row.iter().zip(&b.data).map(|(r, x)| r * x).sum::<f64>() + c0;
        let k = row.len();
        let var: f64 = (0..k).map(|a| (0..k).map(|c| row[a] * v[(a, c)] * row[c]).sum::<f64>()).sum();
        let se = var.max(0.0).sqrt();
        let level = util::level(cmd, st)?;
        let df = st.e.get("df_r").filter(|_| matches!(ecmd.as_str(), "regress" | "anova" | "areg" | "reghdfe" | "qreg" | "xtreg" | "mean" | "total" | "proportion" | "contrast") && st.e.macros.get("model").map_or(true, |m| m != "re"));
        let t = est / se;
        let (p, crit) = match df {
            Some(d) => (2.0 * dist::ttail(d, t.abs()), dist::invttail(d, (1.0 - level / 100.0) / 2.0)),
            None => (2.0 * (1.0 - dist::normal(t.abs())), dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0)),
        };
        let eform = ["or", "irr", "rrr", "eform", "hr"].iter().find(|o| cmd.has_opt(o, o.len())).copied();
        let (shown, sse, lo, hi) = if eform.is_some() {
            (est.exp(), est.exp() * se, (est - crit * se).exp(), (est + crit * se).exp())
        } else {
            (est, se, est - crit * se, est + crit * se)
        };
        let stat = if df.is_some() { "t" } else { "z" };
        let mut res = Results::new();
        res.set("estimate", est);
        res.set("se", se);
        res.set(stat, t);
        res.set("p", p);
        res.set("lb", est - crit * se);
        res.set("ub", est + crit * se);
        if let Some(d) = df {
            res.set("df", d);
        }
        res.set("level", level);
        st.r = res;
        let title = match eform {
            Some("or") => "Odds ratio",
            Some("irr") => "IRR",
            Some(_) => "exp(b)",
            None => "Coefficient",
        };
        Ok(format!(
            " ( 1)  {} = 0\n\n------------------------------------------------------------------------------\n{:>12} | {:>11}  Std. err.  {:>6}    P>|{}|     [{}% conf. interval]\n-------------+----------------------------------------------------------------\n{:>12} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>10}  {:>10}\n------------------------------------------------------------------------------",
            cmd.main.trim(), st.e.macros.get("depvar").cloned().unwrap_or_default(), title, stat, stat, fmt_g(level, 4), "(1)", g9(shown), g9(sse), t, p, g9(lo), g9(hi)
        ))
    }
}

// ---------------------------------------------------------------------------
// estimates
// ---------------------------------------------------------------------------

pub struct Estimates;

impl CommandHandler for Estimates {
    fn name(&self) -> &'static str {
        "estimates"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut w = cmd.main.split_whitespace();
        let sub = w.next().unwrap_or("").to_lowercase();
        let rest: Vec<String> = w.map(String::from).collect();
        match sub.as_str() {
            "store" | "sto" => {
                let name = rest.first().ok_or_else(|| rc_err(198, "name required"))?;
                if st.e.is_empty() {
                    return Err(rc_err(301, "last estimates not found"));
                }
                st.estimates.insert(name.clone(), st.e.clone());
                st.estimates_order.retain(|n| n != name);
                st.estimates_order.push(name.clone());
                Ok(String::new())
            }
            "restore" | "res" => {
                let name = rest.first().ok_or_else(|| rc_err(198, "name required"))?;
                st.e = st.estimates.get(name).cloned().ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))?;
                Ok(String::new())
            }
            "dir" => crate::postest2::estimates_dir(st, &rest),
            "drop" => {
                if rest.iter().any(|r| r == "_all") {
                    st.estimates.clear();
                } else {
                    for r in &rest {
                        st.estimates.remove(r);
                    }
                }
                Ok(String::new())
            }
            "clear" => {
                st.estimates.clear();
                Ok(String::new())
            }
            "replay" | "rep" => {
                let e = match rest.first() {
                    Some(n) => st.estimates.get(n).ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?,
                    None => &st.e,
                };
                crate::est::stash_table_from_e(e, st.level);
                Ok(e.macros.get("_output").cloned().unwrap_or_default())
            }
            "table" | "tab" => {
                let names: Vec<String> = if rest.is_empty() { vec![".".into()] } else { rest.clone() };
                let models: Vec<(String, Results)> = names
                    .iter()
                    .map(|n| if n == "." { Ok(("active".to_string(), st.e.clone())) } else { st.estimates.get(n).cloned().map(|e| (n.clone(), e)).ok_or_else(|| rc_err(111, format!("estimation result {n} not found"))) })
                    .collect::<Result<_, _>>()?;
                estimates_table(&models, cmd)
            }
            "stats" | "sta" => {
                let names: Vec<String> = if rest.is_empty() { vec![".".into()] } else { rest.clone() };
                let mut out = vec![
                    "-----------------------------------------------------------------------------".to_string(),
                    "       Model |          N   ll(null)  ll(model)      df        AIC        BIC".to_string(),
                    "-------------+---------------------------------------------------------------".to_string(),
                ];
                for n in &names {
                    let e = if n == "." { st.e.clone() } else { st.estimates.get(n).cloned().ok_or_else(|| rc_err(111, format!("{n} not found")))? };
                    let (nn, ll, ll0, k) = (e.get("N").unwrap_or(SYSMISS), e.get("ll").unwrap_or(SYSMISS), e.get("ll_0").unwrap_or(SYSMISS), e.get("rank").unwrap_or(SYSMISS));
                    out.push(format!("{:>12} | {:>10} {:>10} {:>10} {:>7} {:>10} {:>10}", if n == "." { "." } else { n }, g9(nn), g9(ll0), g9(ll), k, g9(-2.0 * ll + 2.0 * k), g9(-2.0 * ll + k * nn.ln())));
                }
                out.push("-----------------------------------------------------------------------------".to_string());
                Ok(out.join("\n"))
            }
            _ => crate::postest2::estimates_more(ds, st, cmd, sub.trim_end_matches(':'), &rest),
        }
    }
}

fn estimates_table(models: &[(String, Results)], cmd: &Command) -> Result<String, String> {
    let show_se = cmd.has_opt("se", 2);
    let show_t = cmd.has_opt("t", 1);
    let show_p = cmd.has_opt("p", 1);
    let star = cmd.has_opt("star", 4);
    let stats: Vec<String> = cmd.opt("stats", 2).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let fmtb = cmd.opt("b", 1).map(|f| f.trim().to_string()).filter(|f| !f.is_empty());
    let baselevels = cmd.has_opt("baselevels", 4) || cmd.has_opt("allbaselevels", 4);
    let is_cons = |c: &str| c == "_cons" || c.ends_with(":_cons");
    let is_base = |c: &str| {
        let c = c.rsplit(':').next().unwrap_or(c);
        c.split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))
    };
    // union of the coefficient names; terms that only appear in later models
    // go before the constant, which stays last
    let mut rownames: Vec<String> = Vec::new();
    for (_, e) in models {
        if let Some(b) = e.matrices.get("b") {
            for c in &b.colnames {
                if rownames.contains(c) || (!baselevels && is_base(c)) {
                    continue;
                }
                match rownames.iter().position(|r| is_cons(r)).filter(|_| !is_cons(c)) {
                    Some(at) => rownames.insert(at, c.clone()),
                    None => rownames.push(c.clone()),
                }
            }
        }
    }
    let cw = 12;
    let mut out = Vec::new();
    let bar = format!("{}+{}", "-".repeat(13), "-".repeat(models.len() * (cw + 1)));
    out.push("-".repeat(14 + models.len() * (cw + 1)));
    out.push(format!("{:>12} |{}", "Variable", models.iter().map(|(n, _)| format!(" {:>w$}", crate::summarize::abbrev(n, cw), w = cw)).collect::<String>()));
    out.push(bar.clone());
    let fmtv = |v: f64| match &fmtb {
        Some(f) => crate::format::fmt(v, f).trim().to_string(),
        None => g9(v),
    };
    let mut last_header = String::new();
    for rn in &rownames {
        let mut cells = Vec::new();
        let mut ses = Vec::new();
        let mut ts = Vec::new();
        let mut ps = Vec::new();
        for (_, e) in models {
            let b = e.matrices.get("b");
            let v = e.matrices.get("V");
            match (b, v, b.and_then(|b| b.col_index(rn))) {
                (Some(b), Some(v), Some(j)) => {
                    let bj = b.get(0, j);
                    let se = v.get(j, j).max(0.0).sqrt();
                    if se == 0.0 {
                        // base or omitted level
                        cells.push(fmtv(bj));
                        for col in [&mut ses, &mut ts, &mut ps] {
                            col.push(".".into());
                        }
                        continue;
                    }
                    let t = bj / se;
                    let p = match e.get("df_r") {
                        Some(d) => 2.0 * dist::ttail(d, t.abs()),
                        None => 2.0 * (1.0 - dist::normal(t.abs())),
                    };
                    let stars = if star { if p < 0.001 { "***" } else if p < 0.01 { "**" } else if p < 0.05 { "*" } else { "" } } else { "" };
                    cells.push(format!("{}{}", fmtv(bj), stars));
                    ses.push(fmtv(se));
                    ts.push(format!("{:.2}", t));
                    ps.push(format!("{:.4}", p));
                }
                _ => {
                    cells.push(String::new());
                    ses.push(String::new());
                    ts.push(String::new());
                    ps.push(String::new());
                }
            }
        }
        // factor levels are grouped under their variable, as in coefficient tables
        let (label, header) = if rn.contains(':') { (rn.clone(), None) } else { crate::est::display_name(rn) };
        match header {
            Some(h) if h != last_header => {
                if !out.last().is_some_and(|l| l.starts_with("-------")) {
                    out.push(format!("{:>12} |", ""));
                }
                out.push(format!("{:>12} |", crate::summarize::abbrev(&h, 12)));
                last_header = h;
            }
            Some(_) => {}
            None if !last_header.is_empty() => {
                out.push(format!("{:>12} |", ""));
                last_header.clear();
            }
            None => {}
        }
        out.push(format!("{:>12} |{}", crate::summarize::abbrev(&label, 12), cells.iter().map(|c| format!(" {:>w$}", c, w = cw)).collect::<String>()));
        if show_se {
            out.push(format!("{:>12} |{}", "", ses.iter().map(|c| format!(" {:>w$}", c, w = cw)).collect::<String>()));
        }
        if show_t {
            out.push(format!("{:>12} |{}", "", ts.iter().map(|c| format!(" {:>w$}", c, w = cw)).collect::<String>()));
        }
        if show_p {
            out.push(format!("{:>12} |{}", "", ps.iter().map(|c| format!(" {:>w$}", c, w = cw)).collect::<String>()));
        }
    }
    if !stats.is_empty() {
        out.push(bar.clone());
        for s in &stats {
            let cells: Vec<String> = models
                .iter()
                .map(|(_, e)| {
                    let v = match s.as_str() {
                        "aic" => e.get("ll").map(|ll| -2.0 * ll + 2.0 * e.get("rank").unwrap_or(0.0)),
                        "bic" => e.get("ll").map(|ll| -2.0 * ll + e.get("rank").unwrap_or(0.0) * e.get("N").unwrap_or(1.0).ln()),
                        k => e.get(k),
                    };
                    v.map(|x| g9(x)).unwrap_or_default()
                })
                .collect();
            out.push(format!("{:>12} |{}", s, cells.iter().map(|c| format!(" {:>w$}", c, w = cw)).collect::<String>()));
        }
    }
    out.push("-".repeat(14 + models.len() * (cw + 1)));
    if star {
        out.push(format!("{:>40}", "Legend: * p<0.05; ** p<0.01; *** p<0.001"));
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// margins
// ---------------------------------------------------------------------------

pub struct Margins;

/// Average of the default prediction over the estimation sample, with
/// variables optionally overridden, evaluated at coefficient vector `bv`.
fn avg_pred(ds: &DataSet, st: &Store, rows: &[usize], bv: &Matrix) -> Result<f64, String> {
    let env = Env::with_store(ds, st);
    let mut s = 0.0;
    let mut n = 0.0;
    for &i in rows {
        let p = default_prediction(st, &env, bv, i)?;
        if !is_missing(p) {
            s += p;
            n += 1.0;
        }
    }
    Ok(s / n)
}

fn set_var(ds: &mut DataSet, var: &str, rows: &[usize], value: f64) -> Result<(), String> {
    let v = ds.get_var_mut(var).ok_or_else(|| rc_err(111, format!("variable {var} not found")))?;
    v.var_type = VarType::Double;
    for &i in rows {
        v.data[i] = ferrostat_core::Value::Double(value);
    }
    Ok(())
}

impl CommandHandler for Margins {
    fn name(&self) -> &'static str {
        "margins"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v, ecmd) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
        let mut mask = util::touse(ds, st, cmd)?;
        for i in 0..ds.n_obs {
            mask[i] = mask[i] && sample.get(i).copied().unwrap_or(false);
        }
        let rows = util::rows(&mask);
        let level = util::level(cmd, st)?;
        let dydx: Vec<String> = match cmd.opt("dydx", 2) {
            Some(t) if t.trim() == "*" || t.trim() == "_all" => {
                let mut vs = Vec::new();
                for c in &b.colnames {
                    if c == "_cons" || c.starts_with('/') {
                        continue;
                    }
                    let base = c.rsplit('.').next().unwrap_or(c).to_string();
                    if !c.contains('#') && !vs.contains(&base) {
                        vs.push(base);
                    }
                }
                vs
            }
            Some(t) => t.split_whitespace().map(|x| x.rsplit('.').next().unwrap().to_string()).collect(),
            None => Vec::new(),
        };
        // margins over factor levels
        let over: Vec<String> = cmd.main.split_whitespace().map(|x| x.trim_start_matches("i.").to_string()).collect();
        let atmeans = cmd.has_opt("atmeans", 3);
        let mut work = ds.clone();
        if atmeans {
            // set every continuous regressor to its sample mean
            for c in &b.colnames {
                if c == "_cons" || c.contains('.') || c.contains('#') {
                    continue;
                }
                if let Some(var) = ds.get_var(c) {
                    let m = rows.iter().map(|&i| var.num(i)).sum::<f64>() / rows.len() as f64;
                    set_var(&mut work, c, &rows, m)?;
                }
            }
        }
        if let Some(at) = cmd.opt("at", 2) {
            for part in ferrostat_parser::parser::split_top_level_words(at) {
                if let Some((var, val)) = part.split_once('=') {
                    let val: f64 = val.trim().trim_start_matches('(').trim_end_matches(')').split_whitespace().next().unwrap_or("0").parse().map_err(|_| rc_err(198, "invalid at()"))?;
                    set_var(&mut work, var.trim(), &rows, val)?;
                }
            }
        }
        // list of (label, function computing margin given b)
        let mut labels: Vec<String> = Vec::new();
        type MFn<'a> = Box<dyn Fn(&Matrix) -> Result<f64, String> + 'a>;
        let mut funcs: Vec<MFn> = Vec::new();
        let st_ref: &Store = st;
        if !dydx.is_empty() {
            for x in &dydx {
                // factor variable: discrete change from base level
                let fac_levels: Vec<(String, f64, bool)> = b
                    .colnames
                    .iter()
                    .filter_map(|c| {
                        let (l, var) = c.split_once('.')?;
                        if var != x || c.contains('#') {
                            return None;
                        }
                        let isb = l.ends_with('b');
                        Some((c.clone(), l.trim_end_matches(['b', 'o', 'n']).parse().ok()?, isb))
                    })
                    .collect();
                if !fac_levels.is_empty() {
                    let base = fac_levels.iter().find(|f| f.2).map(|f| f.1).unwrap_or(fac_levels[0].1);
                    for (_, lvl, isb) in fac_levels.clone() {
                        if isb {
                            continue;
                        }
                        let w2 = work.clone();
                        let rows2 = rows.clone();
                        let xn = x.clone();
                        labels.push(format!("{}.{}", fmt_g(lvl, 12), x));
                        funcs.push(Box::new(move |bv: &Matrix| {
                            let mut a = w2.clone();
                            set_var(&mut a, &xn, &rows2, lvl)?;
                            let mut c = w2.clone();
                            set_var(&mut c, &xn, &rows2, base)?;
                            Ok(avg_pred(&a, st_ref, &rows2, bv)? - avg_pred(&c, st_ref, &rows2, bv)?)
                        }));
                    }
                } else {
                    let w2 = work.clone();
                    let rows2 = rows.clone();
                    let xn = x.clone();
                    labels.push(x.clone());
                    funcs.push(Box::new(move |bv: &Matrix| {
                        // average derivative by central differences, observation-specific step
                        let var = w2.get_var(&xn).ok_or_else(|| rc_err(111, format!("{xn} not found")))?;
                        let vals: Vec<f64> = rows2.iter().map(|&i| var.num(i)).collect();
                        let sd = (util::variance(&vals)).sqrt().max(1e-8);
                        let h = 1e-5 * sd;
                        let mut a = w2.clone();
                        let mut c = w2.clone();
                        {
                            let va = a.get_var_mut(&xn).unwrap();
                            va.var_type = VarType::Double;
                            for (k, &i) in rows2.iter().enumerate() {
                                va.data[i] = ferrostat_core::Value::Double(vals[k] + h);
                            }
                            let vc = c.get_var_mut(&xn).unwrap();
                            vc.var_type = VarType::Double;
                            for (k, &i) in rows2.iter().enumerate() {
                                vc.data[i] = ferrostat_core::Value::Double(vals[k] - h);
                            }
                        }
                        Ok((avg_pred(&a, st_ref, &rows2, bv)? - avg_pred(&c, st_ref, &rows2, bv)?) / (2.0 * h))
                    }));
                }
            }
        } else if !over.is_empty() {
            for var in &over {
                let col = util::col(ds, var)?;
                let mut lv: Vec<f64> = rows.iter().map(|&i| col[i]).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                for l in lv {
                    let w2 = work.clone();
                    let rows2 = rows.clone();
                    let vn = var.clone();
                    labels.push(format!("{}.{}", fmt_g(l, 12), var));
                    funcs.push(Box::new(move |bv: &Matrix| {
                        let mut a = w2.clone();
                        set_var(&mut a, &vn, &rows2, l)?;
                        avg_pred(&a, st_ref, &rows2, bv)
                    }));
                }
            }
        } else {
            let w2 = work.clone();
            let rows2 = rows.clone();
            labels.push("_cons".into());
            funcs.push(Box::new(move |bv: &Matrix| avg_pred(&w2, st_ref, &rows2, bv)));
        }
        // estimates and delta-method variances
        let k = b.cols;
        let mut est = Vec::new();
        let mut jac = Matrix::zeros(funcs.len(), k);
        for (m, f) in funcs.iter().enumerate() {
            let e0 = f(&b)?;
            est.push(e0);
            for j in 0..k {
                if v[(j, j)] == 0.0 {
                    continue;
                }
                let h = 1e-6 * (1.0 + b.get(0, j).abs());
                let mut bp = b.clone();
                bp[(0, j)] += h;
                let mut bm = b.clone();
                bm[(0, j)] -= h;
                jac[(m, j)] = (f(&bp)? - f(&bm)?) / (2.0 * h);
            }
        }
        let vm = jac.mul(&v)?.mul(&jac.transpose())?;
        drop(funcs);
        let df = st.e.get("df_r").filter(|_| matches!(ecmd.as_str(), "regress" | "anova"));
        let crit = match df {
            Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
            None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
        };
        let stat = if df.is_some() { "t" } else { "z" };
        let title = if !dydx.is_empty() { "Average marginal effects" } else if atmeans { "Adjusted predictions" } else { "Predictive margins" };
        let expr = match ecmd.as_str() {
            "logit" | "logistic" | "probit" | "cloglog" => format!("Pr({}), predict()", st.e.macros.get("depvar").cloned().unwrap_or_default()),
            "poisson" | "nbreg" => "Predicted number of events, predict()".to_string(),
            _ => "Linear prediction, predict()".to_string(),
        };
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", title, "Number of obs", rows.len()),
            format!("Model VCE: {}", st.e.macros.get("vcetype").cloned().unwrap_or_else(|| if df.is_some() { "OLS".into() } else { "OIM".into() })),
            String::new(),
            format!("Expression: {expr}"),
        ];
        if !dydx.is_empty() {
            out.push(format!("dy/dx wrt:  {}", dydx.join(" ")));
        }
        out.push(String::new());
        out.push("-".repeat(78));
        out.push(format!("{:>12} |{:>12}Delta-method", "", ""));
        out.push(format!("{:>12} | {:>10}   std. err.  {:>6}    P>|{}|     [{}% conf. interval]", "", if dydx.is_empty() { "Margin" } else { "dy/dx" }, stat, stat, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        for (m, l) in labels.iter().enumerate() {
            let se = vm[(m, m)].max(0.0).sqrt();
            let t = est[m] / se;
            let p = match df {
                Some(d) => 2.0 * dist::ttail(d, t.abs()),
                None => 2.0 * (1.0 - dist::normal(t.abs())),
            };
            out.push(format!("{:>12} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>10}  {:>10}", crate::summarize::abbrev(l, 12), g9(est[m]), g9(se), t, p, g9(est[m] - crit * se), g9(est[m] + crit * se)));
        }
        out.push("-".repeat(78));
        let mut res = Results::new();
        res.set("N", rows.len() as f64);
        res.set_mat("b", Matrix::row_vector(&est).with_names(vec!["y1".into()], labels.clone()));
        res.set_mat("V", vm.with_names(labels.clone(), labels));
        res.set("level", level);
        st.r = res;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// estat
// ---------------------------------------------------------------------------

pub struct Estat;

impl CommandHandler for Estat {
    fn name(&self) -> &'static str {
        "estat"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let sub = cmd.main.split_whitespace().next().unwrap_or("").to_lowercase();
        if let Some(r) = crate::mv_post::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::ts_more::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        let (b, v, ecmd) = need_e(st)?;
        let (b, v) = (b.clone(), v.clone());
        if let Some(r) = crate::sp_reg::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::choice_mixed::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::xtdpd::estat(st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::stcox::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::did::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if let Some(r) = crate::binpost::estat(ds, st, sub.as_str(), cmd) {
            return r;
        }
        if ecmd == "arima" && sub == "aroots" {
            return crate::arima::aroots(st);
        }
        match sub.as_str() {
            "ic" => {
                let n = st.e.get("N").unwrap_or(SYSMISS);
                let ll = st.e.get("ll").ok_or_else(|| rc_err(321, "requested action not valid after this command"))?;
                let ll0 = st.e.get("ll_0").unwrap_or(SYSMISS);
                let k = st.e.get("rank").unwrap_or(b.cols as f64) + if ecmd == "regress" { 0.0 } else { 0.0 };
                let aic = -2.0 * ll + 2.0 * k;
                let bic = -2.0 * ll + k * n.ln();
                let mut m = Matrix::zeros(1, 6).with_names(vec![".".into()], vec!["N".into(), "ll0".into(), "ll".into(), "df".into(), "AIC".into(), "BIC".into()]);
                m.data = vec![n, ll0, ll, k, aic, bic];
                let mut r = Results::new();
                r.set_mat("S", m);
                st.r = r;
                Ok(format!(
                    "-----------------------------------------------------------------------------\n       Model |          N   ll(null)  ll(model)      df        AIC        BIC\n-------------+---------------------------------------------------------------\n           . | {:>10} {:>10} {:>10} {:>7} {:>10} {:>10}\n-----------------------------------------------------------------------------",
                    g9(n), g9(ll0), g9(ll), k, g9(aic), g9(bic)
                ))
            }
            "gof" if ecmd == "poisson" => crate::gof::estat_gof(ds, st),
            "vce" => {
                let mut out = vec!["Covariance matrix of coefficients".to_string(), String::new()];
                out.push(format!("{:>12} |{}", "e(V)", b.colnames.iter().map(|c| format!(" {:>10}", crate::summarize::abbrev(c, 10))).collect::<String>()));
                out.push(format!("-------------+{}", "-".repeat(11 * b.cols)));
                for i in 0..b.cols {
                    out.push(format!("{:>12} |{}", crate::summarize::abbrev(&b.colnames[i], 12), (0..=i).map(|j| format!(" {:>10}", g9(v[(i, j)]))).collect::<String>()));
                }
                Ok(out.join("\n"))
            }
            "hettest" => {
                // Breusch-Pagan / Cook-Weisberg using fitted values
                let sample = st.e.sample.clone().unwrap_or_default();
                let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
                let y = util::col(ds, &dep)?;
                let env = Env::with_store(ds, st);
                let rows = util::rows(&sample);
                let mut fit = Vec::new();
                let mut e2 = Vec::new();
                for &i in &rows {
                    let xb = xb_row(&env, &b, None, i)?;
                    fit.push(xb);
                    e2.push((y[i] - xb).powi(2));
                }
                let n = rows.len() as f64;
                let s2 = e2.iter().sum::<f64>() / n;
                let z: Vec<f64> = e2.iter().map(|e| e / s2).collect();
                // regress z on fitted
                let mf = util::mean(&fit);
                let mz = util::mean(&z);
                let sxx: f64 = fit.iter().map(|f| (f - mf).powi(2)).sum();
                let sxy: f64 = fit.iter().zip(&z).map(|(f, zz)| (f - mf) * (zz - mz)).sum();
                let mss = sxy * sxy / sxx;
                let chi2 = mss / 2.0;
                let p = dist::chi2tail(1.0, chi2);
                let mut r = Results::new();
                r.set("chi2", chi2);
                r.set("df", 1.0);
                r.set("p", p);
                st.r = r;
                Ok(format!("Breusch–Pagan/Cook–Weisberg test for heteroskedasticity\nAssumption: Normal error terms\nVariable: Fitted values of {dep}\n\nH0: Constant variance\n\n    chi2(1) = {:>8.2}\nProb > chi2 = {:>8.4}", chi2, p))
            }
            "ovtest" => {
                let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
                let sample = st.e.sample.clone().unwrap_or_default();
                let env = Env::with_store(ds, st);
                let rows = util::rows(&sample);
                let fit: Vec<f64> = rows.iter().map(|&i| xb_row(&env, &b, None, i)).collect::<Result<_, _>>()?;
                drop(env);
                let mf = util::mean(&fit);
                let sf = util::variance(&fit).sqrt();
                let mut work = ds.clone();
                let mut extra = Vec::new();
                for pw in 2..=4 {
                    let name = format!("__yhat{pw}");
                    let mut vals = vec![SYSMISS; ds.n_obs];
                    for (k, &i) in rows.iter().enumerate() {
                        vals[i] = ((fit[k] - mf) / sf).powi(pw);
                    }
                    work.add_variable(util::new_double(&name, vals))?;
                    extra.push(name);
                }
                let xnames: Vec<String> = b.colnames.iter().filter(|c| *c != "_cons" && !c.contains("b.") && !c.starts_with("o.")).cloned().collect();
                let mut st2 = st.clone();
                st2.by_mask = Some(sample.clone());
                let base = ferrostat_parser::parser::parse_line(&format!("regress {} {}", dep, xnames.join(" "))).map_err(|e| rc_err(198, e))?;
                let full = ferrostat_parser::parser::parse_line(&format!("regress {} {} {}", dep, xnames.join(" "), extra.join(" "))).map_err(|e| rc_err(198, e))?;
                let d0 = crate::est::build(&work, &st2, &base, crate::est::DesignSpec { depvar: Some(&dep), indep: &xnames.join(" "), noconstant: false, weights: "", extra: vec![] })?;
                let d1 = crate::est::build(&work, &st2, &full, crate::est::DesignSpec { depvar: Some(&dep), indep: &format!("{} {}", xnames.join(" "), extra.join(" ")), noconstant: false, weights: "", extra: vec![] })?;
                let f0 = crate::est::ols(&d0, &crate::est::Vce::Ols, None)?;
                let f1 = crate::est::ols(&d1, &crate::est::Vce::Ols, None)?;
                let q = f1.df_m - f0.df_m;
                let f = ((f0.rss - f1.rss) / q) / (f1.rss / f1.df_r);
                let p = dist::ftail(q, f1.df_r, f);
                let mut r = Results::new();
                r.set("F", f);
                r.set("df", q);
                r.set("df_r", f1.df_r);
                r.set("p", p);
                st.r = r;
                Ok(format!("Ramsey RESET test for omitted variables\nOmitted: Powers of fitted values of {dep}\n\nH0: Model has no omitted variables\n\nF({}, {}) = {:>8.2}\nProb > F = {:>8.4}", q, f1.df_r, f, p))
            }
            "vif" => {
                let sample = st.e.sample.clone().unwrap_or_default();
                let xs: Vec<String> = b.colnames.iter().filter(|c| *c != "_cons" && !c.contains("b.") && !c.starts_with("o.")).cloned().collect();
                let rows = util::rows(&sample);
                let env = Env::with_store(ds, st);
                let cols: Vec<Vec<f64>> = xs.iter().map(|x| rows.iter().map(|&i| coef_value(&env, x, i)).collect::<Result<Vec<_>, _>>()).collect::<Result<_, _>>()?;
                let mut out = vec!["    Variable |       VIF       1/VIF".to_string(), "-------------+----------------------".to_string()];
                let mut vifs = Vec::new();
                for j in 0..xs.len() {
                    // R2 of x_j on the others (with constant)
                    let y = &cols[j];
                    let others: Vec<Vec<f64>> = (0..xs.len()).filter(|&k| k != j).map(|k| cols[k].clone()).collect();
                    let r2 = r2_of(y, &others);
                    let vif = 1.0 / (1.0 - r2);
                    vifs.push((xs[j].clone(), vif));
                }
                vifs.sort_by(|a, c| c.1.total_cmp(&a.1));
                for (n, vf) in &vifs {
                    out.push(format!("{:>12} | {:>9.2}    {:.6}", crate::summarize::abbrev(n, 12), vf, 1.0 / vf));
                }
                let mean = vifs.iter().map(|x| x.1).sum::<f64>() / vifs.len() as f64;
                out.push("-------------+----------------------".to_string());
                out.push(format!("    Mean VIF | {:>9.2}", mean));
                Ok(out.join("\n"))
            }
            "firststage" | "first" => {
                let fs = st.e.matrices.get("_firststage").cloned().ok_or_else(|| rc_err(321, "estat firststage is only valid after ivregress"))?;
                let robust = st.e.macros.get("vce").map_or(false, |v| v != "unadjusted" && v != "ols");
                let mut out = vec![
                    "  First-stage regression summary statistics".to_string(),
                    "  ----------------------------------------------------------------------".to_string(),
                    format!("               |            Adjusted      Partial     {}", if robust { "Robust" } else { "" }),
                    format!("      Variable |   R-sq.       R-sq.        R-sq.     F({:.0},{:.0})   Prob > F", fs.get(0, 4), fs.get(0, 5)),
                    "  -------------+--------------------------------------------------------".to_string(),
                ];
                for i in 0..fs.rows {
                    out.push(format!(
                        "  {:>12} | {:>7.4} {:>11.4} {:>12.4} {:>11.6} {:>9.4}",
                        crate::summarize::abbrev(&fs.rownames[i], 12),
                        fs.get(i, 0),
                        fs.get(i, 1),
                        fs.get(i, 2),
                        fs.get(i, 3),
                        fs.get(i, 6)
                    ));
                }
                out.push("  ----------------------------------------------------------------------".to_string());
                let mut r = Results::new();
                r.set_mat("singleresults", fs);
                st.r = r;
                Ok(out.join("\n"))
            }
            "overid" => {
                let est_name = st.e.macros.get("cmd").cloned().unwrap_or_default();
                if est_name != "ivregress" {
                    return Err(rc_err(321, "estat overid is only valid after ivregress"));
                }
                let text = st.e.macros.get("_overid").cloned().ok_or_else(|| rc_err(498, "no overidentifying restrictions"))?;
                let m = st.e.matrices.get("_overid").cloned().unwrap();
                let mut r = Results::new();
                for (j, name) in m.colnames.iter().enumerate() {
                    r.set(name, m.get(0, j));
                }
                st.r = r;
                Ok(text)
            }
            "summarize" | "sum" => {
                let sample = st.e.sample.clone().unwrap_or_default();
                let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
                let mut vars = vec![dep];
                vars.extend(b.colnames.iter().filter(|c| *c != "_cons" && !c.contains('.') && !c.contains('#')).cloned());
                let mut out = vec!["  Estimation sample".to_string(), "    Variable |         Mean      Std. dev.          Min          Max".to_string(), "-------------+----------------------------------------------------".to_string()];
                for vname in vars {
                    let s = crate::summarize::quick_summary(ds, &vname, &sample);
                    out.push(format!("{:>12} | {:>12} {:>12} {:>12} {:>12}", vname, g9(s.mean), g9(s.sd()), g9(s.min), g9(s.max)));
                }
                Ok(out.join("\n"))
            }
            _ => Err(rc_err(321, format!("estat {sub} not valid"))),
        }
    }
}

fn r2_of(y: &[f64], xs: &[Vec<f64>]) -> f64 {
    let n = y.len();
    let k = xs.len() + 1;
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    let row = |t: usize| -> Vec<f64> {
        let mut r: Vec<f64> = xs.iter().map(|c| c[t]).collect();
        r.push(1.0);
        r
    };
    for t in 0..n {
        let r = row(t);
        for a in 0..k {
            xy[a] += r[a] * y[t];
            for c in 0..k {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    let (inv, _) = m.sym_inverse(1e-14);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xy[c]).sum()).collect();
    let my = y.iter().sum::<f64>() / n as f64;
    let mut rss = 0.0;
    let mut tss = 0.0;
    for t in 0..n {
        let r = row(t);
        let f: f64 = r.iter().zip(&b).map(|(x, bb)| x * bb).sum();
        rss += (y[t] - f).powi(2);
        tss += (y[t] - my).powi(2);
    }
    1.0 - rss / tss
}
