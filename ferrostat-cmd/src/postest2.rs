//! More post-estimation commands: likelihood-ratio tests (`lrtest`),
//! nonlinear combinations and tests (`nlcom`, `testnl`, `predictnl`) and
//! the remaining `estimates` subcommands (`save`, `use`, `describe`,
//! `notes`, `title`, `dir`, `query`).
//!
//! Nonlinear functions of the coefficients are differentiated numerically
//! (central differences); standard errors follow from the delta method.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType};
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_top_level_words};
use ferrostat_parser::{Command, Expr};

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("lrtest", Box::new(Lrtest)),
        ("nlcom", Box::new(Nlcom)),
        ("testnl", Box::new(Testnl)),
        ("predictnl", Box::new(Predictnl)),
    ]
}

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

/// Whether results of this estimator are reported with t statistics.
pub(crate) fn uses_t(e: &Results) -> Option<f64> {
    let cmd = e.macros.get("cmd").map(String::as_str).unwrap_or("");
    if matches!(
        cmd,
        "logit" | "logistic" | "probit" | "cloglog" | "poisson" | "nbreg" | "ologit" | "oprobit" | "mlogit" | "tobit" | "glm" | "ivregress" | "xtivreg" | "nlcom" | "mlexp" | "ml"
    ) {
        return None;
    }
    if e.macros.get("model").is_some_and(|m| m == "re") {
        return None;
    }
    e.get("df_r").filter(|d| *d > 0.0 && !is_missing(*d))
}

// ---------------------------------------------------------------------------
// Functions of the coefficients
// ---------------------------------------------------------------------------

/// An expression in the coefficients (`_b[x]`, `[eq]x`, `_se[x]` ...),
/// evaluated at arbitrary coefficient vectors.
pub(crate) struct CoefFn {
    expr: Expr,
    tmp: Store,
    empty: DataSet,
}

impl CoefFn {
    pub(crate) fn new(text: &str, st: &Store) -> Result<Self, String> {
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let wrapped = crate::postest::wrap_coefs(text, b);
        let expr = parse_expression(&wrapped).map_err(|e| rc_err(198, format!("{text}: {}", e.message)))?;
        let mut tmp = Store::new();
        tmp.scalars = st.scalars.clone();
        tmp.matrices = st.matrices.clone();
        tmp.e = st.e.clone();
        tmp.e.sample = None;
        Ok(CoefFn { expr, tmp, empty: DataSet::new() })
    }

    pub(crate) fn eval(&mut self, bv: &[f64]) -> Result<f64, String> {
        if let Some(b) = self.tmp.e.matrices.get_mut("b") {
            b.data[..bv.len()].copy_from_slice(bv);
        }
        let env = Env::with_store(&self.empty, &self.tmp);
        env.eval(&self.expr, 0)?.num()
    }
}

/// Step for numerical derivatives with respect to a coefficient.
fn step(b: f64) -> f64 {
    1e-5 * (b.abs() + 1e-3)
}

/// Values and Jacobian (rows: functions) of functions of `b`.
pub(crate) fn jacobian(fs: &mut [CoefFn], b: &[f64]) -> Result<(Vec<f64>, Matrix), String> {
    let k = b.len();
    let q = fs.len();
    let mut val = vec![0.0; q];
    let mut g = Matrix::zeros(q, k);
    for (i, f) in fs.iter_mut().enumerate() {
        val[i] = f.eval(b)?;
        if is_missing(val[i]) {
            return Err(rc_err(498, "expression evaluates to missing"));
        }
        let mut bb = b.to_vec();
        for j in 0..k {
            let h = step(b[j]);
            bb[j] = b[j] + h;
            let up = f.eval(&bb)?;
            bb[j] = b[j] - h;
            let dn = f.eval(&bb)?;
            bb[j] = b[j];
            g[(i, j)] = if is_missing(up) || is_missing(dn) { 0.0 } else { (up - dn) / (2.0 * h) };
        }
    }
    Ok((val, g))
}

fn need_bv(st: &Store) -> Result<(Matrix, Matrix), String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let v = st.e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    Ok((b, v))
}

/// `(name: exp) (exp) ...` or a single `exp` / `name: exp`.
fn named_exps(text: &str) -> Result<Vec<(Option<String>, String)>, String> {
    let t = text.trim();
    let groups: Vec<String> = if t.starts_with('(') && split_top_level_words(t).iter().all(|w| w.starts_with('(') && w.ends_with(')')) {
        split_top_level_words(t).into_iter().map(|g| g[1..g.len() - 1].trim().to_string()).collect()
    } else {
        vec![t.to_string()]
    };
    let mut out = Vec::new();
    for g in groups {
        if g.is_empty() {
            return Err(rc_err(198, "expression required"));
        }
        // `name:` prefix (but not `_b[eq:x]`)
        match find_top_level(&g, ':') {
            Some(i) if g[..i].trim().chars().all(|c| c.is_alphanumeric() || c == '_') && !g[..i].trim().is_empty() => {
                out.push((Some(g[..i].trim().to_string()), g[i + 1..].trim().to_string()));
            }
            _ => out.push((None, g)),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// lrtest
// ---------------------------------------------------------------------------

pub struct Lrtest;

fn model_of(st: &Store, name: &str) -> Result<Results, String> {
    if name == "." {
        if st.e.is_empty() {
            return Err(rc_err(301, "last estimates not found"));
        }
        return Ok(st.e.clone());
    }
    st.estimates.get(name).cloned().ok_or_else(|| rc_err(111, format!("estimation result {name} not found")))
}

fn model_df(e: &Results) -> f64 {
    if let Some(r) = e.get("rank") {
        return r;
    }
    e.matrices.get("b").map_or(0.0, |b| b.cols as f64)
}

impl CommandHandler for Lrtest {
    fn name(&self) -> &'static str {
        "lrtest"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let names: Vec<String> = cmd.main.split_whitespace().map(|s| s.trim_matches(|c| c == '(' || c == ')').to_string()).collect();
        if names.is_empty() || names.len() > 2 {
            return Err(rc_err(198, "lrtest modelspec1 [modelspec2]"));
        }
        let n1 = names[0].clone();
        let n2 = names.get(1).cloned().unwrap_or_else(|| ".".into());
        let a = model_of(st, &n1)?;
        let b = model_of(st, &n2)?;
        let force = cmd.has_opt("force", 5);
        for (n, e) in [(&n1, &a), (&n2, &b)] {
            if e.get("ll").is_none() {
                return Err(rc_err(498, format!("{n} does not contain a log likelihood (e(ll) not found)")));
            }
            if !force && e.macros.get("vce").is_some_and(|v| v == "robust" || v == "cluster") {
                return Err(rc_err(498, "LR test likely invalid for models with robust vce\n(use option force to perform the test anyway)"));
            }
        }
        let (ca, cb) = (a.macros.get("cmd").cloned().unwrap_or_default(), b.macros.get("cmd").cloned().unwrap_or_default());
        if !force && ca != cb {
            return Err(rc_err(402, format!("estimation commands differ: {ca} vs. {cb}\n(use option force to perform the test anyway)")));
        }
        let (na, nb) = (a.get("N").unwrap_or(SYSMISS), b.get("N").unwrap_or(SYSMISS));
        if !force && na != nb {
            return Err(rc_err(498, format!("observations differ: {} vs. {}", g9(na), g9(nb))));
        }
        let (dfa, dfb) = (model_df(&a), model_df(&b));
        let (lla, llb) = (a.get("ll").unwrap(), b.get("ll").unwrap());
        // the unrestricted model has more parameters (or the larger ll on ties)
        let a_full = if dfa != dfb { dfa > dfb } else { lla >= llb };
        let (full, rest, llf, llr, dff, dfr) = if a_full { (&n1, &n2, lla, llb, dfa, dfb) } else { (&n2, &n1, llb, lla, dfb, dfa) };
        let df = match util::opt_num(cmd, "df", 2)? {
            Some(d) => d,
            None => dff - dfr,
        };
        if df <= 0.0 && util::opt_num(cmd, "df", 2)?.is_none() {
            return Err(rc_err(498, format!("df(unrestricted) = df(restricted) = {}", dff)));
        }
        let chi2 = 2.0 * (llf - llr);
        let p = dist::chi2tail(df, chi2.max(0.0));
        let mut out = vec![String::new(), "Likelihood-ratio test".to_string(), format!("Assumption: {rest} nested within {full}"), String::new()];
        let label = format!("LR chi2({})", fmt_g(df, 8));
        let w = label.len().max(11);
        out.push(format!("{:>w$} = {:>6.2}", label, chi2));
        out.push(format!("{:>w$} = {:>6.4}", "Prob > chi2", p));
        if chi2 < 0.0 {
            out.push("Note: ll(restricted) > ll(unrestricted); the models may not be nested.".into());
        }
        let mut r = Results::new();
        r.set("chi2", chi2);
        r.set("df", df);
        r.set("p", p);
        if cmd.has_opt("stats", 2) {
            out.push(String::new());
            out.push("Akaike's information criterion and Bayesian information criterion".into());
            out.push(String::new());
            out.push("-".repeat(77));
            out.push("       Model |          N   ll(null)  ll(model)      df        AIC        BIC".into());
            out.push(format!("{}+{}", "-".repeat(13), "-".repeat(63)));
            let mut s = Matrix::zeros(2, 6);
            for (i, (n, e)) in [(&n1, &a), (&n2, &b)].iter().enumerate() {
                let (nn, ll, ll0, k) = (e.get("N").unwrap_or(SYSMISS), e.get("ll").unwrap_or(SYSMISS), e.get("ll_0").unwrap_or(SYSMISS), model_df(e));
                let aic = -2.0 * ll + 2.0 * k;
                let bic = -2.0 * ll + k * nn.ln();
                out.push(format!("{:>12} | {:>10} {:>10} {:>10} {:>7} {:>10} {:>10}", crate::summarize::abbrev(n, 12), g9(nn), if is_missing(ll0) { ".".into() } else { g9(ll0) }, g9(ll), k, g9(aic), g9(bic)));
                for (j, x) in [nn, ll0, ll, k, aic, bic].iter().enumerate() {
                    s[(i, j)] = *x;
                }
            }
            out.push("-".repeat(77));
            r.set_mat("stats", s.with_names(vec![n1.clone(), n2.clone()], ["N", "ll0", "ll", "df", "AIC", "BIC"].iter().map(|x| x.to_string()).collect()));
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// nlcom
// ---------------------------------------------------------------------------

pub struct Nlcom;

impl CommandHandler for Nlcom {
    fn name(&self) -> &'static str {
        "nlcom"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v) = need_bv(st)?;
        let exps = named_exps(&cmd.main)?;
        let mut fs = Vec::new();
        let mut names = Vec::new();
        for (k, (n, t)) in exps.iter().enumerate() {
            fs.push(CoefFn::new(t, st)?);
            names.push(n.clone().unwrap_or_else(|| format!("_nl_{}", k + 1)));
        }
        let (val, g) = jacobian(&mut fs, &b.data[..b.cols])?;
        let vm = g.mul(&v)?.mul(&g.transpose())?;
        let level = util::level(cmd, st)?;
        let df = util::opt_num(cmd, "df", 2)?;
        let q = names.len();
        let mut out = vec![String::new()];
        for (k, (_, t)) in exps.iter().enumerate() {
            out.push(format!("{:>12}: {}", crate::summarize::abbrev(&names[k], 12), t));
        }
        out.push(String::new());
        let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default();
        let depvar = depvar.split_whitespace().next().unwrap_or("").to_string();
        let no = vec![false; q];
        let spec = crate::est::TableSpec { depvar: &depvar, names: &names, b: &val, v: &vm, omitted: &no, is_base: &no, df, level, eform: None, coef_title: "Coefficient" };
        out.extend(crate::est::coef_table(&spec));
        let bm = Matrix::row_vector(&val).with_names(vec!["y1".into()], names.clone());
        let vmn = vm.with_names(names.clone(), names.clone());
        let n = st.e.get("N");
        let mut r = Results::new();
        // keep r(b) and r(V) next to the r(table) of the displayed table
        if let Some((m, lvl)) = crate::est::take_table() {
            r.set_mat("table", m);
            r.set("level", lvl);
        }
        r.set_mat("b", bm.clone());
        r.set_mat("V", vmn.clone());
        if let Some(n) = n {
            r.set("N", n);
        }
        if let Some(d) = df {
            r.set("df_r", d);
        }
        if cmd.has_opt("post", 4) {
            let sample = st.e.sample.clone();
            let e = st.begin_e();
            e.set_str("cmd", "nlcom");
            e.set_str("depvar", depvar);
            e.set_str("properties", "b V");
            if let Some(n) = n {
                e.set("N", n);
            }
            if let Some(d) = df {
                e.set("df_r", d);
            }
            e.set_mat("b", bm);
            e.set_mat("V", vmn);
            e.sample = sample;
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// testnl
// ---------------------------------------------------------------------------

pub struct Testnl;

impl CommandHandler for Testnl {
    fn name(&self) -> &'static str {
        "testnl"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v) = need_bv(st)?;
        let t = cmd.main.trim();
        let groups: Vec<String> = if t.starts_with('(') && split_top_level_words(t).iter().all(|w| w.starts_with('(') && w.ends_with(')')) {
            split_top_level_words(t).into_iter().map(|g| g[1..g.len() - 1].trim().to_string()).collect()
        } else {
            vec![t.to_string()]
        };
        let mut labels = Vec::new();
        let mut fs = Vec::new();
        for g in &groups {
            let parts = split_equals(g);
            if parts.len() < 2 {
                return Err(rc_err(198, format!("{g}: equal sign required")));
            }
            for w in parts.windows(2) {
                labels.push(format!("{} = {}", w[0].trim(), w[1].trim()));
                fs.push(CoefFn::new(&format!("({}) - ({})", w[0], w[1]), st)?);
            }
        }
        let (val, g) = jacobian(&mut fs, &b.data[..b.cols])?;
        let q = val.len();
        let gvg = g.mul(&v)?.mul(&g.transpose())?;
        let (inv, dropped) = gvg.sym_inverse(1e-12);
        let w: f64 = (0..q).map(|a| (0..q).map(|c| val[a] * inv[(a, c)] * val[c]).sum::<f64>()).sum();
        let df = (q - dropped.len()) as f64;
        let mut out = vec![String::new()];
        for (i, l) in labels.iter().enumerate() {
            out.push(format!("  ({}) {} {}", i + 1, if i + 1 < 10 { " " } else { "" }, l));
        }
        out.push(String::new());
        let mut r = Results::new();
        let dfr = cmd.opt("df", 2).map(|s| s.trim().parse::<f64>()).transpose().map_err(|_| rc_err(198, "df() invalid"))?.or_else(|| uses_t(&st.e));
        if let Some(d) = dfr {
            let f = w / df;
            let p = dist::ftail(df, d, f);
            out.push(format!("{:>23} = {:>11.2}", format!("F({}, {})", df, fmt_g(d, 8)), f));
            out.push(format!("{:>23} = {:>11.4}", "Prob > F", p));
            r.set("F", f);
            r.set("df_r", d);
            r.set("p", p);
        } else {
            let p = dist::chi2tail(df, w);
            out.push(format!("{:>23} = {:>11.2}", format!("chi2({})", df), w));
            out.push(format!("{:>23} = {:>11.4}", "Prob > chi2", p));
            r.set("chi2", w);
            r.set("p", p);
        }
        r.set("df", df);
        r.set_mat("G", g.with_names((1..=q).map(|i| format!("r{i}")).collect(), b.colnames.clone()));
        r.set_mat("R", Matrix::col_vector(&val).with_names((1..=q).map(|i| format!("r{i}")).collect(), vec!["c1".into()]));
        st.r = r;
        Ok(out.join("\n"))
    }
}

/// Split at top-level `=` signs that are not part of `==`, `<=`, `>=`, `!=`.
fn split_equals(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let mut quote = false;
    for i in 0..b.len() {
        match b[i] {
            b'"' => quote = !quote,
            b'(' | b'[' if !quote => depth += 1,
            b')' | b']' if !quote => depth -= 1,
            b'=' if !quote && depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { b' ' };
                let next = b.get(i + 1).copied().unwrap_or(b' ');
                if next != b'=' && !matches!(prev, b'=' | b'<' | b'>' | b'!' | b'~') {
                    parts.push(s[start..i].to_string());
                    start = i + 1;
                }
            }
            _ => {}
        }
    }
    parts.push(s[start..].to_string());
    parts
}

// ---------------------------------------------------------------------------
// predictnl
// ---------------------------------------------------------------------------

pub struct Predictnl;

/// Replace `predict(opts)` and `xb(eq)` calls by temporary variable names;
/// returns the rewritten text and the (variable, predict options) pairs.
fn predict_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut out = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    loop {
        let lower = rest.to_lowercase();
        let pi = lower.find("predict(");
        let xi = lower.find("xb(").filter(|&i| i == 0 || !rest.as_bytes()[i - 1].is_ascii_alphanumeric() && rest.as_bytes()[i - 1] != b'_');
        let (i, len, is_xb) = match (pi, xi) {
            (Some(p), Some(x)) if x < p => (x, 3, true),
            (Some(p), _) => (p, 8, false),
            (None, Some(x)) => (x, 3, true),
            (None, None) => break,
        };
        out.push_str(&rest[..i]);
        let after = &rest[i + len..];
        let mut depth = 1;
        let mut end = after.len();
        for (j, c) in after.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = after[..end].trim().to_string();
        let name = format!("__pnl{}", calls.len() + 1);
        let opts = if is_xb {
            if inner.is_empty() { "xb".to_string() } else { format!("xb equation({inner})") }
        } else {
            inner
        };
        calls.push((name.clone(), opts));
        out.push_str(&name);
        rest = &after[(end + 1).min(after.len())..];
    }
    out.push_str(rest);
    (out, calls)
}

/// Evaluate the predictnl expression for all observations at coefficients `bv`.
fn pnl_values(ds: &mut DataSet, st: &mut Store, b0: &Matrix, bv: &[f64], expr: &Expr, calls: &[(String, String)], mask: &[bool]) -> Result<Vec<f64>, String> {
    let saved = st.e.matrices.get("b").cloned();
    let mut bm = b0.clone();
    bm.data[..bv.len()].copy_from_slice(bv);
    st.e.matrices.insert("b".into(), bm);
    let res = (|| -> Result<Vec<f64>, String> {
        for (name, opts) in calls {
            let _ = ds.drop_variable(name);
            let line = if opts.is_empty() { format!("predict double {name}") } else { format!("predict double {name}, {opts}") };
            let c = parse_line(&line).map_err(|e| rc_err(198, e.message))?;
            crate::postest::Predict.execute(ds, st, &c)?;
        }
        let env = Env::with_store(ds, st);
        (0..ds.n_obs).map(|i| if mask[i] { env.eval(expr, i)?.num() } else { Ok(SYSMISS) }).collect()
    })();
    for (name, _) in calls {
        let _ = ds.drop_variable(name);
    }
    if let Some(s) = saved {
        st.e.matrices.insert("b".into(), s);
    }
    res
}

impl CommandHandler for Predictnl {
    fn name(&self) -> &'static str {
        "predictnl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, v) = need_bv(st)?;
        let main = cmd.main.trim();
        let eq = find_top_level(main, '=').ok_or_else(|| rc_err(198, "predictnl newvar = expression"))?;
        let lhs: Vec<&str> = main[..eq].split_whitespace().collect();
        let (ty, newvar) = match lhs.as_slice() {
            [t, n] => (VarType::from_type_str(t).ok_or_else(|| rc_err(198, format!("{t} invalid type")))?, n.to_string()),
            [n] => (VarType::Float, n.to_string()),
            _ => return Err(rc_err(198, "predictnl newvar = expression")),
        };
        crate::varlist::check_name(&newvar)?;
        if ds.var_index.contains_key(&newvar) {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        let text = main[eq + 1..].trim();
        let (rewritten, calls) = predict_calls(text);
        let wrapped = crate::postest::wrap_coefs(&rewritten, &b);
        let expr = parse_expression(&wrapped).map_err(|e| rc_err(198, format!("{text}: {}", e.message)))?;
        let mask = util::touse(ds, st, cmd)?;
        let k = b.cols;
        let b0: Vec<f64> = b.data[..k].to_vec();
        let f0 = pnl_values(ds, st, &b, &b0, &expr, &calls, &mask)?;
        let n = ds.n_obs;
        let want_se = ["se", "variance", "wald", "p", "ci"].iter().any(|o| cmd.has_opt(o, o.len()));
        let mut var = vec![0.0; n];
        if want_se {
            // gradient of every observation's value
            let mut grads: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
            for j in 0..k {
                if v[(j, j)] == 0.0 {
                    continue;
                }
                let h = step(b0[j]);
                let mut bb = b0.clone();
                bb[j] = b0[j] + h;
                let up = pnl_values(ds, st, &b, &bb, &expr, &calls, &mask)?;
                bb[j] = b0[j] - h;
                let dn = pnl_values(ds, st, &b, &bb, &expr, &calls, &mask)?;
                for i in 0..n {
                    grads[i][j] = if is_missing(up[i]) || is_missing(dn[i]) { SYSMISS } else { (up[i] - dn[i]) / (2.0 * h) };
                }
            }
            for i in 0..n {
                if is_missing(f0[i]) || grads[i].iter().any(|g| is_missing(*g)) {
                    var[i] = SYSMISS;
                    continue;
                }
                let g = &grads[i];
                var[i] = (0..k).map(|a| (0..k).map(|c| g[a] * v[(a, c)] * g[c]).sum::<f64>()).sum::<f64>().max(0.0);
            }
        }
        let n_miss = (0..n).filter(|&i| mask[i] && is_missing(f0[i])).count();
        let put = |ds: &mut DataSet, name: &str, ty: VarType, vals: &[f64]| -> Result<(), String> {
            crate::varlist::check_name(name)?;
            if ds.var_index.contains_key(name) {
                return Err(rc_err(110, format!("variable {name} already defined")));
            }
            let mut var = ferrostat_core::Variable::new(name.to_string(), ty);
            var.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
            ds.add_variable(var)
        };
        put(ds, &newvar, ty, &f0)?;
        let level = util::level(cmd, st)?;
        let df = uses_t(&st.e);
        let se: Vec<f64> = var.iter().map(|x| if is_missing(*x) { SYSMISS } else { x.sqrt() }).collect();
        if let Some(name) = cmd.opt("se", 2) {
            put(ds, name.trim(), VarType::Float, &se)?;
        }
        if let Some(name) = cmd.opt("variance", 3) {
            put(ds, name.trim(), VarType::Float, &var)?;
        }
        if let Some(name) = cmd.opt("wald", 4) {
            let w: Vec<f64> = (0..n).map(|i| if is_missing(f0[i]) || is_missing(var[i]) || var[i] == 0.0 { SYSMISS } else { f0[i] * f0[i] / var[i] }).collect();
            put(ds, name.trim(), VarType::Float, &w)?;
        }
        if let Some(name) = cmd.opt("p", 1) {
            let p: Vec<f64> = (0..n)
                .map(|i| {
                    if is_missing(f0[i]) || is_missing(var[i]) || var[i] == 0.0 {
                        return SYSMISS;
                    }
                    let w = f0[i] * f0[i] / var[i];
                    match df {
                        Some(d) => dist::ftail(1.0, d, w),
                        None => dist::chi2tail(1.0, w),
                    }
                })
                .collect();
            put(ds, name.trim(), VarType::Float, &p)?;
        }
        if let Some(names) = cmd.opt("ci", 2) {
            let nm: Vec<&str> = names.split_whitespace().collect();
            if nm.len() != 2 {
                return Err(rc_err(198, "ci() requires two new variable names"));
            }
            let crit = match df {
                Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
                None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
            };
            let lo: Vec<f64> = (0..n).map(|i| if is_missing(f0[i]) || is_missing(se[i]) { SYSMISS } else { f0[i] - crit * se[i] }).collect();
            let hi: Vec<f64> = (0..n).map(|i| if is_missing(f0[i]) || is_missing(se[i]) { SYSMISS } else { f0[i] + crit * se[i] }).collect();
            put(ds, nm[0], VarType::Float, &lo)?;
            put(ds, nm[1], VarType::Float, &hi)?;
        }
        st.data_changed = true;
        Ok(if n_miss > 0 { format!("({} missing value{} generated)", util::commas(n_miss), if n_miss == 1 { "" } else { "s" }) } else { String::new() })
    }
}

// ---------------------------------------------------------------------------
// estimates save | use | describe | notes | title | dir | query
// ---------------------------------------------------------------------------

fn results_json(e: &Results) -> serde_json::Value {
    use serde_json::json;
    let mats: serde_json::Map<String, serde_json::Value> = e
        .matrices
        .iter()
        .map(|(k, m)| {
            (
                k.clone(),
                json!({"rows": m.rows, "cols": m.cols, "data": m.data.iter().map(|x| if is_missing(*x) { serde_json::Value::Null } else { json!(x) }).collect::<Vec<_>>(), "rownames": m.rownames, "colnames": m.colnames}),
            )
        })
        .collect();
    let sc: serde_json::Map<String, serde_json::Value> = e.scalars.iter().map(|(k, v)| (k.clone(), if is_missing(*v) { serde_json::Value::Null } else { json!(v) })).collect();
    json!({"scalars": sc, "macros": e.macros, "matrices": mats})
}

fn results_from_json(v: &serde_json::Value) -> Result<Results, String> {
    let bad = || rc_err(610, "file not in estimates format");
    let mut e = Results::new();
    for (k, x) in v["scalars"].as_object().ok_or_else(bad)? {
        e.set(k, x.as_f64().unwrap_or(SYSMISS));
    }
    for (k, x) in v["macros"].as_object().ok_or_else(bad)? {
        e.set_str(k, x.as_str().unwrap_or(""));
    }
    for (k, m) in v["matrices"].as_object().ok_or_else(bad)? {
        let rows = m["rows"].as_u64().ok_or_else(bad)? as usize;
        let cols = m["cols"].as_u64().ok_or_else(bad)? as usize;
        let mut mat = Matrix::zeros(rows, cols);
        mat.data = m["data"].as_array().ok_or_else(bad)?.iter().map(|x| x.as_f64().unwrap_or(SYSMISS)).collect();
        if mat.data.len() != rows * cols {
            return Err(bad());
        }
        let names = |key: &str| -> Vec<String> { m[key].as_array().map(|a| a.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default() };
        mat = mat.with_names(names("rownames"), names("colnames"));
        e.set_mat(k, mat);
    }
    Ok(e)
}

const STER_MAGIC: &str = "ferrostat-estimates";

fn ster_path(file: &str) -> String {
    let f = util::unquote(file.trim());
    if std::path::Path::new(&f).extension().is_some() { f } else { format!("{f}.ster") }
}

fn with_cwd(st: &Store, f: &str) -> String {
    match &st.cwd {
        Some(d) if !std::path::Path::new(f).is_absolute() => d.join(f).display().to_string(),
        _ => f.to_string(),
    }
}

/// Subcommands of `estimates` not handled by `postest::Estimates`.
pub fn estimates_more(ds: &mut DataSet, st: &mut Store, cmd: &Command, sub: &str, rest: &[String]) -> Result<String, String> {
    let _ = ds;
    let after_sub = || -> String {
        let m = cmd.main.trim();
        let i = m.find(|c: char| c.is_whitespace() || c == ':').unwrap_or(m.len());
        m[i..].trim().trim_start_matches(':').trim().to_string()
    };
    match sub {
        "save" | "sav" => {
            let file = rest.first().ok_or_else(|| rc_err(198, "filename required"))?;
            let path = with_cwd(st, &ster_path(file));
            if st.e.is_empty() {
                return Err(rc_err(301, "last estimates not found"));
            }
            let exists = ferrostat_core::vfs::exists(&path);
            let append = cmd.has_opt("append", 6);
            if exists && !cmd.has_opt("replace", 7) && !append {
                return Err(rc_err(602, format!("file {path} already exists")));
            }
            let mut sets: Vec<serde_json::Value> = Vec::new();
            if append && exists {
                let text = ferrostat_core::vfs::read_to_string(&path).map_err(|e| rc_err(603, e))?;
                let v: serde_json::Value = serde_json::from_str(&text).map_err(|_| rc_err(610, "file not in estimates format"))?;
                sets = v["sets"].as_array().cloned().unwrap_or_default();
            }
            sets.push(results_json(&st.e));
            let doc = serde_json::json!({"format": STER_MAGIC, "version": 1, "sets": sets});
            ferrostat_core::vfs::write(&path, doc.to_string().as_bytes()).map_err(|e| rc_err(603, e))?;
            Ok(format!("file {} saved", ster_path(file)))
        }
        "use" | "us" => {
            let file = rest.first().ok_or_else(|| rc_err(198, "filename required"))?;
            let path = with_cwd(st, &ster_path(file));
            let text = ferrostat_core::vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
            let v: serde_json::Value = serde_json::from_str(&text).map_err(|_| rc_err(610, "file not in estimates format"))?;
            if v["format"].as_str() != Some(STER_MAGIC) {
                return Err(rc_err(610, "file not in estimates format"));
            }
            let sets = v["sets"].as_array().ok_or_else(|| rc_err(610, "file not in estimates format"))?;
            let k = util::opt_num(cmd, "number", 3)?.unwrap_or(1.0) as usize;
            let set = sets.get(k.wrapping_sub(1)).ok_or_else(|| rc_err(198, format!("estimation set {k} not found")))?;
            st.e = results_from_json(set)?;
            st.e.sample = None;
            Ok(String::new())
        }
        "describe" | "des" | "desc" | "describ" => {
            let (name, e) = match rest.first().map(String::as_str) {
                Some("using") => {
                    let file = rest.get(1).ok_or_else(|| rc_err(198, "filename required"))?;
                    let path = with_cwd(st, &ster_path(file));
                    let text = ferrostat_core::vfs::read_to_string(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
                    let v: serde_json::Value = serde_json::from_str(&text).map_err(|_| rc_err(610, "file not in estimates format"))?;
                    let sets = v["sets"].as_array().cloned().unwrap_or_default();
                    let k = util::opt_num(cmd, "number", 3)?.unwrap_or(1.0) as usize;
                    (None, results_from_json(sets.get(k.wrapping_sub(1)).ok_or_else(|| rc_err(198, "estimation set not found"))?)?)
                }
                Some(n) => (Some(n.to_string()), st.estimates.get(n).cloned().ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?),
                None => {
                    if st.e.is_empty() {
                        return Err(rc_err(301, "last estimates not found"));
                    }
                    (None, st.e.clone())
                }
            };
            let mut out = vec![String::new()];
            if let Some(n) = &name {
                out.push(format!("Estimation results \"{n}\":"));
                out.push(String::new());
            } else {
                out.push("Estimation results produced by".to_string());
                out.push(String::new());
            }
            let line = e.macros.get("cmdline").cloned().unwrap_or_else(|| {
                let c = e.macros.get("cmd").cloned().unwrap_or_default();
                let d = e.macros.get("depvar").cloned().unwrap_or_default();
                let x: Vec<String> = e.matrices.get("b").map(|b| b.colnames.iter().filter(|c| *c != "_cons" && !c.contains(':') && !c.contains('.')).cloned().collect()).unwrap_or_default();
                format!("{c} {d} {}", x.join(" ")).trim().to_string()
            });
            if name.is_some() {
                out.push("Estimation results produced by".to_string());
                out.push(String::new());
            }
            out.push(format!("    . {line}"));
            if let Some(t) = e.macros.get("_estimates_title") {
                out.push(String::new());
                out.push(format!("Title: {t}"));
            }
            let mut r = Results::new();
            r.set_str("cmdline", line);
            st.r = r;
            Ok(out.join("\n"))
        }
        "title" | "tit" => {
            let t = after_sub();
            if st.e.is_empty() {
                return Err(rc_err(301, "last estimates not found"));
            }
            if t.is_empty() {
                return Ok(st.e.macros.get("_estimates_title").map(|t| format!("\n{t}")).unwrap_or_default());
            }
            st.e.set_str("_estimates_title", util::unquote(&t));
            Ok(String::new())
        }
        "notes" | "note" => {
            let m = cmd.main.trim();
            let after = m.split_once(char::is_whitespace).map_or("", |x| x.1).trim();
            let after_colon = m.find(':').map(|i| m[i + 1..].trim().to_string());
            let count = st.e.macros.keys().filter(|k| k.starts_with("_estnote_")).count();
            if let Some(text) = after_colon {
                if st.e.is_empty() {
                    return Err(rc_err(301, "last estimates not found"));
                }
                st.e.set_str(&format!("_estnote_{}", count + 1), text);
                return Ok(String::new());
            }
            let words: Vec<&str> = after.split_whitespace().collect();
            match words.first().copied() {
                Some("drop") => {
                    // `estimates notes drop in #[/#]`
                    let range = words.get(2).copied().unwrap_or("");
                    let (lo, hi) = match range.split_once('/') {
                        Some((a, b)) => (a.parse::<usize>().unwrap_or(1), b.parse::<usize>().unwrap_or(count)),
                        None => {
                            let x = range.parse::<usize>().map_err(|_| rc_err(198, "estimates notes drop in #[/#]"))?;
                            (x, x)
                        }
                    };
                    let notes: Vec<String> = (1..=count).filter(|i| *i < lo || *i > hi).filter_map(|i| st.e.macros.get(&format!("_estnote_{i}")).cloned()).collect();
                    for i in 1..=count {
                        st.e.macros.remove(&format!("_estnote_{i}"));
                    }
                    for (i, n) in notes.iter().enumerate() {
                        st.e.set_str(&format!("_estnote_{}", i + 1), n.clone());
                    }
                    let dropped = count - notes.len();
                    Ok(format!("({} note{} dropped)", dropped, if dropped == 1 { "" } else { "s" }))
                }
                _ => {
                    if count == 0 {
                        return Ok(String::new());
                    }
                    let mut out = vec![String::new()];
                    for i in 1..=count {
                        out.push(format!("{:>3}. {}", i, st.e.macros.get(&format!("_estnote_{i}")).cloned().unwrap_or_default()));
                    }
                    Ok(out.join("\n"))
                }
            }
        }
        "query" | "q" => {
            let active = st.estimates.iter().find(|(_, e)| e.macros == st.e.macros && e.scalars == st.e.scalars).map(|(n, _)| n.clone());
            Ok(match active {
                Some(n) => format!("(active results are from estimates {n})"),
                None => "(active results produced by\n    . last estimation command; not yet stored)".to_string(),
            })
        }
        _ => Err(rc_err(198, format!("estimates {sub}: invalid subcommand"))),
    }
}

/// `estimates dir`: table of stored estimates.
pub fn estimates_dir(st: &mut Store, names: &[String]) -> Result<String, String> {
    let list: Vec<String> = if names.is_empty() || names.iter().any(|n| n == "_all" || n == "*") {
        let mut v: Vec<String> = st.estimates_order.iter().filter(|n| st.estimates.contains_key(*n)).cloned().collect();
        for k in st.estimates.keys() {
            if !v.contains(k) {
                v.push(k.clone());
            }
        }
        v
    } else {
        names.to_vec()
    };
    let mut out = vec![String::new(), "-".repeat(55), "        Name | Command      Depvar       npar  Title".to_string(), format!("{}+{}", "-".repeat(13), "-".repeat(41))];
    for n in &list {
        let e = st.estimates.get(n).ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?;
        let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
        let dep = e.macros.get("depvar").cloned().unwrap_or_default();
        let npar = e.matrices.get("b").map_or(0, |b| b.cols);
        let title = e.macros.get("_estimates_title").cloned().unwrap_or_default();
        out.push(format!(
            "{:>12} | {:<12} {:<12} {:>4}  {}",
            crate::summarize::abbrev(n, 12),
            crate::summarize::abbrev(&cmd, 12),
            crate::summarize::abbrev(&dep, 12),
            npar,
            title
        ));
    }
    out.push("-".repeat(55));
    let mut r = Results::new();
    r.set_str("names", list.join(" "));
    st.r = r;
    Ok(out.join("\n"))
}
