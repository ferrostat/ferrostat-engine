//! Estimators defined by substitutable expressions: nonlinear least squares
//! (`nl`), systems of nonlinear equations (`nlsur`) and maximum likelihood
//! from an expression for the log-likelihood contribution (`mlexp`).
//!
//! Parameters are written in braces: `{b0}`, `{b1=0.5}` (initial value)
//! and linear combinations `{xb: x1 x2 _cons}` whose coefficients are
//! named `xb:x1`, ...; `{xb:}` reuses a combination defined earlier.
//! Derivatives are computed numerically.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_top_level_words};
use ferrostat_parser::{Command, Expr};

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("nl", Box::new(Nl)), ("nlsur", Box::new(Nlsur)), ("mlexp", Box::new(Mlexp))]
}

/// Parameters shared by all expressions of a model.
#[derive(Default, Clone)]
struct Params {
    /// coefficient names: `b0` for scalars (shown as `/b0`), `xb:x1` for combinations
    names: Vec<String>,
    init: Vec<f64>,
    /// linear combinations already defined: name -> expression text
    combos: BTreeMap<String, String>,
}

fn pvar(k: usize) -> String {
    format!("__nlp{k}")
}

impl Params {
    fn index(&mut self, name: &str, init: f64) -> usize {
        match self.names.iter().position(|n| n == name) {
            Some(i) => i,
            None => {
                self.names.push(name.to_string());
                self.init.push(init);
                self.names.len() - 1
            }
        }
    }

    /// Replace the `{...}` parameters of `text` by placeholder names.
    fn rewrite(&mut self, ds: &DataSet, text: &str) -> Result<String, String> {
        let mut out = String::new();
        let mut rest = text;
        while let Some(i) = rest.find('{') {
            out.push_str(&rest[..i]);
            let j = rest[i..].find('}').ok_or_else(|| rc_err(198, "unmatched {"))? + i;
            let inner = rest[i + 1..j].trim();
            rest = &rest[j + 1..];
            if let Some((eq, vars)) = inner.split_once(':') {
                let eq = eq.trim().to_string();
                let vars = vars.trim();
                if vars.is_empty() {
                    let t = self.combos.get(&eq).cloned().ok_or_else(|| rc_err(198, format!("linear combination {eq} not defined")))?;
                    out.push_str(&t);
                    continue;
                }
                let mut terms = Vec::new();
                for w in vars.split_whitespace() {
                    let list = if w == "_cons" || w.contains('.') { vec![w.to_string()] } else { crate::varlist::expand(ds, w)? };
                    for v in list {
                        let k = self.index(&format!("{eq}:{v}"), 0.0);
                        terms.push(if v == "_cons" { pvar(k) } else { format!("{}*{v}", pvar(k)) });
                    }
                }
                let t = format!("({})", terms.join(" + "));
                self.combos.insert(eq, t.clone());
                out.push_str(&t);
            } else {
                let (name, init) = match inner.split_once('=') {
                    Some((n, v)) => (n.trim(), Some(v.trim().parse::<f64>().map_err(|_| rc_err(198, format!("{{{inner}}}: invalid initial value")))?)),
                    None => (inner, None),
                };
                if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Err(rc_err(198, format!("{{{inner}}}: invalid parameter name")));
                }
                let k = self.index(name, init.unwrap_or(0.0));
                if let Some(v) = init {
                    self.init[k] = v;
                }
                out.push_str(&pvar(k));
            }
        }
        out.push_str(rest);
        Ok(out)
    }

    /// Display names: `/b0` for scalar parameters.
    fn display(&self) -> Vec<String> {
        self.names.iter().map(|n| if n.contains(':') { n.clone() } else { format!("/{n}") }).collect()
    }

    /// Names in e(b): `b0:_cons` for scalar parameters.
    fn stored(&self) -> Vec<String> {
        self.names.iter().map(|n| if n.contains(':') { n.clone() } else { format!("{n}:_cons") }).collect()
    }
}

/// Substitute parameter values into a parsed expression.
fn subst(e: &Expr, b: &[f64]) -> Expr {
    match e {
        Expr::Var(n) => match n.strip_prefix("__nlp").and_then(|k| k.parse::<usize>().ok()) {
            Some(k) if k < b.len() => Expr::Number(b[k]),
            _ => e.clone(),
        },
        Expr::Binary(l, op, r) => Expr::Binary(Box::new(subst(l, b)), op.clone(), Box::new(subst(r, b))),
        Expr::Unary(op, x) => Expr::Unary(op.clone(), Box::new(subst(x, b))),
        Expr::Call(f, args) => Expr::Call(f.clone(), args.iter().map(|a| subst(a, b)).collect()),
        Expr::Paren(x) => Expr::Paren(Box::new(subst(x, b))),
        Expr::Index(x, i) => Expr::Index(Box::new(subst(x, b)), Box::new(subst(i, b))),
        _ => e.clone(),
    }
}

/// Values of `e` at parameters `b` for the given rows.
fn values(ds: &DataSet, st: &Store, e: &Expr, b: &[f64], rows: &[usize]) -> Result<Vec<f64>, String> {
    let s = subst(e, b);
    let env = Env::with_store(ds, st);
    rows.iter().map(|&r| env.eval(&s, r).and_then(|v| v.num())).collect()
}

fn step(b: f64) -> f64 {
    1e-6 * (b.abs() + 1e-2).max(1e-4)
}

/// Jacobian columns d values / d b_k (central differences).
fn jacobian(ds: &DataSet, st: &Store, e: &Expr, b: &[f64], rows: &[usize]) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::new();
    for k in 0..b.len() {
        let h = step(b[k]);
        let mut bp = b.to_vec();
        bp[k] += h;
        let up = values(ds, st, e, &bp, rows)?;
        bp[k] = b[k] - h;
        let dn = values(ds, st, e, &bp, rows)?;
        out.push(up.iter().zip(&dn).map(|(u, d)| (u - d) / (2.0 * h)).collect());
    }
    Ok(out)
}

/// Estimation sample: if/in, weights, nonmissing variables of the expressions.
fn sample(ds: &DataSet, st: &Store, cmd: &Command, exprs: &[&Expr], init: &[f64]) -> Result<(Vec<bool>, Option<util::Weights>), String> {
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
    let env = Env::with_store(ds, st);
    let subs: Vec<Expr> = exprs.iter().map(|e| subst(e, init)).collect();
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        // observations where the expression cannot be evaluated drop out only
        // when a variable is missing (checked by evaluating at the start values)
        for e in &subs {
            match env.eval(e, i).and_then(|v| v.num()) {
                Ok(v) if is_missing(v) => {
                    if vars_missing(ds, e, i) {
                        mask[i] = false;
                    }
                }
                Ok(_) => {}
                Err(er) => return Err(er),
            }
        }
    }
    Ok((mask, w))
}

fn vars_missing(ds: &DataSet, e: &Expr, i: usize) -> bool {
    match e {
        Expr::Var(n) => ds.get_var(n).is_some_and(|v| !v.var_type.is_string() && is_missing(v.num(i))),
        Expr::Binary(l, _, r) => vars_missing(ds, l, i) || vars_missing(ds, r, i),
        Expr::Unary(_, x) | Expr::Paren(x) => vars_missing(ds, x, i),
        Expr::Call(_, a) => a.iter().any(|x| vars_missing(ds, x, i)),
        Expr::Index(x, y) => vars_missing(ds, x, i) || vars_missing(ds, y, i),
        _ => false,
    }
}

fn mat_of(cols: &[Vec<f64>], w: &[f64]) -> Matrix {
    let k = cols.len();
    let mut m = Matrix::zeros(k, k);
    for a in 0..k {
        for c in a..k {
            let s: f64 = cols[a].iter().zip(&cols[c]).zip(w).map(|((x, y), w)| x * y * w).sum();
            m[(a, c)] = s;
            m[(c, a)] = s;
        }
    }
    m
}

fn table(depvar: &str, names: &[String], b: &[f64], v: &Matrix, df: Option<f64>, level: f64) -> Vec<String> {
    let no = vec![false; names.len()];
    crate::est::coef_table(&crate::est::TableSpec { depvar, names, b, v, omitted: &no, is_base: &no, df, level, eform: None, coef_title: "Coefficient" })
}

// ---------------------------------------------------------------------------
// nl
// ---------------------------------------------------------------------------

pub struct Nl;

/// `(y = expression)` -> (depvar, expression text).
fn split_eq(text: &str) -> Result<(String, String), String> {
    let t = text.trim();
    let t = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(t);
    let i = find_top_level(t, '=').ok_or_else(|| rc_err(198, "equation y = expression required"))?;
    Ok((t[..i].trim().to_string(), t[i + 1..].trim().to_string()))
}

/// Least squares on stacked residuals r(b) (Levenberg–Marquardt).
fn gauss_newton(ds: &DataSet, st: &Store, eqs: &[(Vec<f64>, Expr)], wts: &[Vec<f64>], b0: Vec<f64>, rows: &[usize], iterate: usize, log: &mut Vec<f64>) -> Result<(Vec<f64>, bool), String> {
    let k = b0.len();
    let ssr = |b: &[f64]| -> Result<Option<f64>, String> {
        let mut s = 0.0;
        for (q, (y, e)) in eqs.iter().enumerate() {
            let f = values(ds, st, e, b, rows)?;
            for t in 0..rows.len() {
                if is_missing(f[t]) || !f[t].is_finite() {
                    return Ok(None);
                }
                s += wts[q][t] * (y[t] - f[t]).powi(2);
            }
        }
        Ok(Some(s))
    };
    let mut b = b0;
    let mut s = ssr(&b)?.ok_or_else(|| rc_err(480, "starting values invalid or some RHS variables have missing values"))?;
    log.push(s);
    let mut lambda = 1e-6;
    let mut converged = false;
    for _ in 0..iterate {
        let mut jtj = Matrix::zeros(k, k);
        let mut jtr = vec![0.0; k];
        for (q, (y, e)) in eqs.iter().enumerate() {
            let f = values(ds, st, e, &b, rows)?;
            let j = jacobian(ds, st, e, &b, rows)?;
            let m = mat_of(&j, &wts[q]);
            for a in 0..k {
                for c in 0..k {
                    jtj[(a, c)] += m[(a, c)];
                }
                jtr[a] += j[a].iter().zip(y).zip(&f).zip(&wts[q]).map(|(((d, yy), ff), w)| d * (yy - ff) * w).sum::<f64>();
            }
        }
        let mut accepted = false;
        for _ in 0..30 {
            let mut a = jtj.clone();
            for i in 0..k {
                a[(i, i)] += lambda * jtj[(i, i)].max(1e-12);
            }
            let (inv, _) = a.sym_inverse(1e-14);
            let d: Vec<f64> = (0..k).map(|i| (0..k).map(|c| inv[(i, c)] * jtr[c]).sum()).collect();
            let nb: Vec<f64> = b.iter().zip(&d).map(|(x, y)| x + y).collect();
            if let Some(ns) = ssr(&nb)? {
                if ns <= s {
                    let rel = (s - ns) / (s.abs() + 1e-30);
                    let db = b.iter().zip(&nb).map(|(x, y)| (x - y).abs() / (x.abs() + 1e-6)).fold(0.0, f64::max);
                    b = nb;
                    s = ns;
                    lambda = (lambda / 10.0).max(1e-12);
                    accepted = true;
                    log.push(s);
                    if rel < 1e-12 || db < 1e-9 {
                        converged = true;
                    }
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !accepted {
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }
    Ok((b, converged))
}

impl CommandHandler for Nl {
    fn name(&self) -> &'static str {
        "nl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("nl") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (depvar, rhs) = split_eq(&cmd.main)?;
        let mut p = Params::default();
        let text = p.rewrite(ds, &rhs)?;
        let e = parse_expression(&text).map_err(|er| rc_err(198, format!("{rhs}: {}", er.message)))?;
        if let Some(init) = cmd.opt("initial", 2) {
            let w: Vec<&str> = init.split_whitespace().collect();
            for pair in w.chunks(2) {
                if let [n, v] = pair {
                    let k = p.names.iter().position(|x| x == n).ok_or_else(|| rc_err(111, format!("parameter {n} not found")))?;
                    p.init[k] = v.parse().map_err(|_| rc_err(198, "initial() invalid"))?;
                }
            }
        }
        let ye = parse_expression(&depvar).map_err(|er| rc_err(198, er.message))?;
        let (mut mask, wts) = sample(ds, st, cmd, &[&e], &p.init)?;
        let yall = util::eval_col(ds, st, &ye, &mask)?;
        for i in 0..ds.n_obs {
            if mask[i] && is_missing(yall[i]) {
                mask[i] = false;
            }
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let n = rows.len();
        let y: Vec<f64> = rows.iter().map(|&r| yall[r]).collect();
        let w: Vec<f64> = match &wts {
            Some(w) => {
                let v: Vec<f64> = rows.iter().map(|&r| w.w[r]).collect();
                if w.kind == util::WeightKind::A {
                    let m = v.iter().sum::<f64>() / n as f64;
                    v.iter().map(|x| x / m).collect()
                } else {
                    v
                }
            }
            None => vec![1.0; n],
        };
        let n_eff = if wts.as_ref().is_some_and(|w| w.kind == util::WeightKind::F) { w.iter().sum::<f64>() } else { n as f64 };
        let iterate = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let mut log = Vec::new();
        let eqs = vec![(y.clone(), e.clone())];
        let (b, converged) = gauss_newton(ds, st, &eqs, &[w.clone()], p.init.clone(), &rows, iterate, &mut log)?;
        let k = b.len();
        let f = values(ds, st, &e, &b, &rows)?;
        let j = jacobian(ds, st, &e, &b, &rows)?;
        // a parameter acting as a constant term (derivative 1 everywhere)
        let cons = j.iter().position(|c| c.iter().all(|d| (d - 1.0).abs() < 1e-6));
        let rss: f64 = y.iter().zip(&f).zip(&w).map(|((a, c), w)| w * (a - c).powi(2)).sum();
        let ybar = y.iter().zip(&w).map(|(a, w)| a * w).sum::<f64>() / w.iter().sum::<f64>();
        let tss = if cons.is_some() { y.iter().zip(&w).map(|(a, w)| w * (a - ybar).powi(2)).sum::<f64>() } else { y.iter().zip(&w).map(|(a, w)| w * a * a).sum() };
        let mss = tss - rss;
        let df_r = n_eff - k as f64;
        let df_m = if cons.is_some() { k as f64 - 1.0 } else { k as f64 };
        let df_t = if cons.is_some() { n_eff - 1.0 } else { n_eff };
        let s2 = rss / df_r;
        let (inv, _) = mat_of(&j, &w).sym_inverse(1e-14);
        let vce_robust = cmd.opt("vce", 3).is_some_and(|v| v.trim().starts_with('r')) || cmd.has_opt("robust", 1);
        let v = if vce_robust {
            let mut meat = Matrix::zeros(k, k);
            for t in 0..n {
                let r = y[t] - f[t];
                for a in 0..k {
                    for c in 0..k {
                        meat[(a, c)] += w[t] * w[t] * r * r * j[a][t] * j[c][t];
                    }
                }
            }
            inv.mul(&meat)?.mul(&inv)?.scale(n_eff / df_r)
        } else {
            inv.scale(s2)
        };
        let r2 = mss / tss;
        let r2a = 1.0 - (1.0 - r2) * df_t / df_r;
        let rmse = s2.sqrt();
        let dev = n_eff * (1.0 + (2.0 * std::f64::consts::PI * rss / n_eff).ln());
        let level = util::level(cmd, st)?;
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 4) {
            for (i, s) in log.iter().enumerate() {
                out.push(format!("Iteration {i}:{}Residual SS = {:>12}", if i < 10 { "  " } else { " " }, fmt_g(*s, 10)));
            }
            out.push(String::new());
        }
        let f8 = |x: f64| fmt_g(x, 9);
        out.push("      Source |      SS            df       MS".into());
        out.push(format!("-------------+----------------------------------    Number of obs = {:>10}", util::commas(n_eff as usize)));
        out.push(format!("       Model | {:>10} {:>10} {:>11}    R-squared     = {:>10.4}", f8(mss), df_m, f8(mss / df_m), r2));
        out.push(format!("    Residual | {:>10} {:>10} {:>11}    Adj R-squared = {:>10.4}", f8(rss), df_r, f8(s2), r2a));
        out.push(format!("-------------+----------------------------------    Root MSE      = {:>10}", fmt_g(rmse, 9)));
        out.push(format!("       Total | {:>10} {:>10} {:>11}    Res. dev.     = {:>10}", f8(tss), df_t, f8(tss / df_t), fmt_g(dev, 9)));
        out.push(String::new());
        out.extend(table(&depvar, &p.display(), &b, &v, Some(df_r), level));
        if let Some(c) = cons {
            out.push(format!("Note: Parameter {} is used as a constant term during estimation.", p.names[c]));
        }
        if !converged {
            out.push("convergence not achieved".into());
        }
        let names = p.stored();
        let mut res = Results::new();
        res.set("N", n_eff);
        res.set("k", k as f64);
        res.set("df_m", df_m);
        res.set("df_r", df_r);
        res.set("df_t", df_t);
        res.set("rss", rss);
        res.set("mss", mss);
        res.set("tss", tss);
        res.set("r2", r2);
        res.set("r2_a", r2a);
        res.set("rmse", rmse);
        res.set("dev", dev);
        res.set("ll", -dev / 2.0);
        res.set("ic", (log.len() - 1) as f64);
        res.set("converged", if converged { 1.0 } else { 0.0 });
        res.set("rank", k as f64);
        res.set_str("cmd", "nl");
        res.set_str("depvar", depvar.clone());
        res.set_str("title", "");
        res.set_str("type", "1");
        res.set_str("sexp", rhs.clone());
        res.set_str("params", p.names.join(" "));
        res.set_str("properties", "b V");
        res.set_str("predict", "nl_p");
        res.set_str("vce", if vce_robust { "robust" } else { "gnr" });
        if vce_robust {
            res.set_str("vcetype", "Robust");
        }
        res.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        res.set_mat("V", v.with_names(names.clone(), names));
        res.sample = Some(mask);
        let text = out.join("\n");
        res.set_str("_output", text.clone());
        st.e = res;
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// nlsur
// ---------------------------------------------------------------------------

pub struct Nlsur;

impl CommandHandler for Nlsur {
    fn name(&self) -> &'static str {
        "nlsur"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let groups = split_top_level_words(cmd.main.trim());
        if groups.is_empty() || !groups.iter().all(|g| g.starts_with('(')) {
            return Err(rc_err(198, "nlsur (y1 = exp1) (y2 = exp2) ..."));
        }
        let mut p = Params::default();
        let mut deps = Vec::new();
        let mut exprs = Vec::new();
        let mut texts = Vec::new();
        for g in &groups {
            let (d, r) = split_eq(g)?;
            let t = p.rewrite(ds, &r)?;
            exprs.push(parse_expression(&t).map_err(|er| rc_err(198, format!("{r}: {}", er.message)))?);
            deps.push(d);
            texts.push(r);
        }
        let refs: Vec<&Expr> = exprs.iter().collect();
        let (mut mask, _) = sample(ds, st, cmd, &refs, &p.init)?;
        let mut ycols = Vec::new();
        for d in &deps {
            let ye = parse_expression(d).map_err(|er| rc_err(198, er.message))?;
            let y = util::eval_col(ds, st, &ye, &mask)?;
            for i in 0..ds.n_obs {
                if mask[i] && is_missing(y[i]) {
                    mask[i] = false;
                }
            }
            ycols.push(y);
        }
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let n = rows.len();
        let q = deps.len();
        let ys: Vec<Vec<f64>> = ycols.iter().map(|y| rows.iter().map(|&r| y[r]).collect()).collect();
        let iterate = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let ifgnls = cmd.has_opt("ifgnls", 3);
        let nls = cmd.has_opt("nls", 3);
        // step 1: equal weights
        let ones = vec![vec![1.0; n]; q];
        let eqs: Vec<(Vec<f64>, Expr)> = ys.iter().cloned().zip(exprs.iter().cloned()).collect();
        let mut log = Vec::new();
        let (mut b, mut converged) = gauss_newton(ds, st, &eqs, &ones, p.init.clone(), &rows, iterate, &mut log)?;
        let resid = |b: &[f64]| -> Result<Vec<Vec<f64>>, String> {
            exprs.iter().zip(&ys).map(|(e, y)| Ok(values(ds, st, e, b, &rows)?.iter().zip(y).map(|(f, yy)| yy - f).collect())).collect()
        };
        let sigma_of = |r: &[Vec<f64>]| -> Matrix {
            let mut s = Matrix::zeros(q, q);
            for a in 0..q {
                for c in 0..q {
                    s[(a, c)] = r[a].iter().zip(&r[c]).map(|(x, y)| x * y).sum::<f64>() / n as f64;
                }
            }
            s
        };
        let mut sigma = if nls { Matrix::identity(q) } else { sigma_of(&resid(&b)?) };
        if !nls {
            for _ in 0..if ifgnls { 100 } else { 1 } {
                // whiten: minimise sum_t r_t' S^-1 r_t via the Cholesky factor of S^-1
                let sinv = sigma.inverse()?;
                let l = sinv.cholesky()?;
                // transformed system: z_a = sum_c L'[a,c] r_c
                let (nb, conv) = gls_step(ds, st, &exprs, &ys, &l, b.clone(), &rows, iterate)?;
                converged = conv;
                let change = b.iter().zip(&nb).map(|(x, y)| (x - y).abs() / (x.abs() + 1e-6)).fold(0.0, f64::max);
                b = nb;
                let ns = sigma_of(&resid(&b)?);
                let sch = ns.data.iter().zip(&sigma.data).map(|(x, y)| (x - y).abs() / (y.abs() + 1e-12)).fold(0.0, f64::max);
                sigma = ns;
                if !ifgnls || (change < 1e-9 && sch < 1e-9) {
                    break;
                }
            }
        }
        // VCE: (sum_t J_t' S^-1 J_t)^-1
        let k = b.len();
        let js: Vec<Vec<Vec<f64>>> = exprs.iter().map(|e| jacobian(ds, st, e, &b, &rows)).collect::<Result<_, _>>()?;
        let sinv = sigma.inverse()?;
        let mut info = Matrix::zeros(k, k);
        for t in 0..n {
            for a in 0..q {
                for c in 0..q {
                    let s = sinv[(a, c)];
                    if s == 0.0 {
                        continue;
                    }
                    for i in 0..k {
                        let ji = js[a][i][t];
                        if ji == 0.0 {
                            continue;
                        }
                        for jj in 0..k {
                            info[(i, jj)] += ji * s * js[c][jj][t];
                        }
                    }
                }
            }
        }
        let (mut v, _) = info.sym_inverse(1e-14);
        if nls {
            // equal weights: sandwich with the residual covariance
            let r = resid(&b)?;
            let s = sigma_of(&r);
            let mut meat = Matrix::zeros(k, k);
            for t in 0..n {
                for a in 0..q {
                    for c in 0..q {
                        for i in 0..k {
                            for jj in 0..k {
                                meat[(i, jj)] += js[a][i][t] * s[(a, c)] * js[c][jj][t];
                            }
                        }
                    }
                }
            }
            v = v.mul(&meat)?.mul(&v)?;
        }
        let r = resid(&b)?;
        let level = util::level(cmd, st)?;
        let title = if nls { "NLS regression" } else if ifgnls { "FGNLS regression (iterated)" } else { "FGNLS regression" };
        let mut out = vec![String::new(), title.to_string()];
        out.push("-".repeat(70));
        out.push("       Equation |       Obs  Parms       RMSE      R-sq     Constant".into());
        out.push(format!("{}+{}", "-".repeat(16), "-".repeat(53)));
        let mut res = Results::new();
        let mut any_unc = false;
        for a in 0..q {
            let used: Vec<usize> = (0..k).filter(|&i| js[a][i].iter().any(|d| *d != 0.0)).collect();
            let cons = used.iter().copied().find(|&i| js[a][i].iter().all(|d| (d - 1.0).abs() < 1e-6));
            let rss: f64 = r[a].iter().map(|x| x * x).sum();
            let ybar = util::mean(&ys[a]);
            let tss: f64 = if cons.is_some() { ys[a].iter().map(|y| (y - ybar).powi(2)).sum() } else { ys[a].iter().map(|y| y * y).sum() };
            let r2 = 1.0 - rss / tss;
            let rmse = (rss / n as f64).sqrt();
            any_unc |= cons.is_none();
            out.push(format!(
                " {:<2}{:>12} | {:>9} {:>6} {:>10} {:>9} {:>12}",
                a + 1,
                crate::summarize::abbrev(&deps[a], 12),
                util::commas(n),
                used.len(),
                fmt_g(rmse, 8),
                format!("{:.4}{}", r2, if cons.is_none() { "*" } else { "" }),
                cons.map_or("(none)".to_string(), |c| p.names[c].clone())
            ));
            res.set(&format!("rss_{}", a + 1), rss);
            res.set(&format!("rmse_{}", a + 1), rmse);
            res.set(&format!("r2_{}", a + 1), r2);
            res.set(&format!("k_{}", a + 1), used.len() as f64);
            res.set_str(&format!("sexp_{}", a + 1), texts[a].clone());
        }
        out.push("-".repeat(70));
        if any_unc {
            out.push("* Uncentered R-sq".into());
        }
        out.push(String::new());
        out.extend(table("", &p.display(), &b, &v, None, level));
        let names = p.stored();
        res.set("N", n as f64);
        res.set("k", k as f64);
        res.set("n_eq", q as f64);
        res.set("converged", if converged { 1.0 } else { 0.0 });
        let ldet = sigma.det().ln();
        res.set("ll", -0.5 * n as f64 * (q as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()) + ldet));
        res.set_str("cmd", "nlsur");
        res.set_str("depvar", deps.join(" "));
        res.set_str("method", if nls { "nls" } else if ifgnls { "ifgnls" } else { "fgnls" });
        res.set_str("title", title);
        res.set_str("properties", "b V");
        res.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        res.set_mat("V", v.with_names(names.clone(), names));
        res.set_mat("Sigma", sigma.with_names(deps.clone(), deps.clone()));
        res.sample = Some(mask);
        st.e = res;
        Ok(out.join("\n"))
    }
}

/// One FGNLS estimation with fixed weighting matrix: residuals of all
/// equations whitened by the Cholesky factor `l` of S^-1 (S^-1 = L L').
#[allow(clippy::too_many_arguments)]
fn gls_step(ds: &DataSet, st: &Store, exprs: &[Expr], ys: &[Vec<f64>], l: &Matrix, b0: Vec<f64>, rows: &[usize], iterate: usize) -> Result<(Vec<f64>, bool), String> {
    let q = exprs.len();
    let n = rows.len();
    let k = b0.len();
    let obj = |b: &[f64]| -> Result<Option<(f64, Vec<Vec<f64>>)>, String> {
        let mut r = Vec::new();
        for (e, y) in exprs.iter().zip(ys) {
            let f = values(ds, st, e, b, rows)?;
            if f.iter().any(|x| is_missing(*x) || !x.is_finite()) {
                return Ok(None);
            }
            r.push(f.iter().zip(y).map(|(ff, yy)| yy - ff).collect::<Vec<f64>>());
        }
        // z = L' r
        let mut s = 0.0;
        let mut z = vec![vec![0.0; n]; q];
        for a in 0..q {
            for c in 0..q {
                let lca = l[(c, a)];
                if lca != 0.0 {
                    for t in 0..n {
                        z[a][t] += lca * r[c][t];
                    }
                }
            }
            s += z[a].iter().map(|x| x * x).sum::<f64>();
        }
        Ok(Some((s, z)))
    };
    let mut b = b0;
    let (mut s, mut z) = obj(&b)?.ok_or_else(|| rc_err(480, "starting values invalid"))?;
    let mut lambda = 1e-6;
    let mut converged = false;
    for _ in 0..iterate {
        let js: Vec<Vec<Vec<f64>>> = exprs.iter().map(|e| jacobian(ds, st, e, &b, rows)).collect::<Result<_, _>>()?;
        // whitened Jacobian: Jz_a = sum_c L'[a,c] J_c
        let mut jz = vec![vec![vec![0.0; n]; k]; q];
        for a in 0..q {
            for c in 0..q {
                let lca = l[(c, a)];
                if lca != 0.0 {
                    for i in 0..k {
                        for t in 0..n {
                            jz[a][i][t] += lca * js[c][i][t];
                        }
                    }
                }
            }
        }
        let mut jtj = Matrix::zeros(k, k);
        let mut jtr = vec![0.0; k];
        for a in 0..q {
            for i in 0..k {
                jtr[i] += jz[a][i].iter().zip(&z[a]).map(|(x, y)| x * y).sum::<f64>();
                for jj in i..k {
                    let v: f64 = jz[a][i].iter().zip(&jz[a][jj]).map(|(x, y)| x * y).sum();
                    jtj[(i, jj)] += v;
                    if jj != i {
                        jtj[(jj, i)] += v;
                    }
                }
            }
        }
        let mut accepted = false;
        for _ in 0..30 {
            let mut a = jtj.clone();
            for i in 0..k {
                a[(i, i)] += lambda * jtj[(i, i)].max(1e-12);
            }
            let (inv, _) = a.sym_inverse(1e-14);
            let d: Vec<f64> = (0..k).map(|i| (0..k).map(|c| inv[(i, c)] * jtr[c]).sum()).collect();
            let nb: Vec<f64> = b.iter().zip(&d).map(|(x, y)| x + y).collect();
            if let Some((ns, nz)) = obj(&nb)? {
                if ns <= s {
                    let rel = (s - ns) / (s.abs() + 1e-30);
                    let db = b.iter().zip(&nb).map(|(x, y)| (x - y).abs() / (x.abs() + 1e-6)).fold(0.0, f64::max);
                    b = nb;
                    s = ns;
                    z = nz;
                    lambda = (lambda / 10.0).max(1e-12);
                    accepted = true;
                    if rel < 1e-12 || db < 1e-9 {
                        converged = true;
                    }
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !accepted || converged {
            converged = true;
            break;
        }
    }
    Ok((b, converged))
}

// ---------------------------------------------------------------------------
// mlexp
// ---------------------------------------------------------------------------

pub struct Mlexp;

impl CommandHandler for Mlexp {
    fn name(&self) -> &'static str {
        "mlexp"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let inner = main.strip_prefix('(').and_then(|x| x.strip_suffix(')')).ok_or_else(|| rc_err(198, "mlexp (expression)"))?;
        let mut p = Params::default();
        let text = p.rewrite(ds, inner)?;
        let e = parse_expression(&text).map_err(|er| rc_err(198, format!("{inner}: {}", er.message)))?;
        if let Some(from) = cmd.opt("from", 4) {
            for item in from.split_whitespace() {
                let (n, v) = item.split_once('=').ok_or_else(|| rc_err(198, "from() invalid"))?;
                let k = p.names.iter().position(|x| x == n.trim_start_matches('/')).ok_or_else(|| rc_err(111, format!("parameter {n} not found")))?;
                p.init[k] = v.parse().map_err(|_| rc_err(198, "from() invalid"))?;
            }
        }
        let (mask, wts) = sample(ds, st, cmd, &[&e], &p.init)?;
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let n = rows.len();
        let w: Vec<f64> = match &wts {
            Some(w) => rows.iter().map(|&r| w.w[r]).collect(),
            None => vec![1.0; n],
        };
        let n_eff = if wts.as_ref().is_some_and(|w| w.kind == util::WeightKind::F) { w.iter().sum::<f64>() } else { n as f64 };
        let k = p.names.len();
        let ll_at = |b: &[f64]| -> Result<Option<f64>, String> {
            let v = values(ds, st, &e, b, &rows)?;
            if v.iter().any(|x| is_missing(*x) || !x.is_finite()) {
                return Ok(None);
            }
            Ok(Some(v.iter().zip(&w).map(|(a, w)| a * w).sum()))
        };
        let mut out = Vec::new();
        let mut b = p.init.clone();
        let mut ll = match ll_at(&b)? {
            Some(v) => {
                out.push(format!("initial:       Log likelihood = {}", fmt_g(v, 10)));
                v
            }
            None => {
                out.push("initial:       Log likelihood =     -<inf>  (could not be evaluated)".to_string());
                let mut found = None;
                for g in [0.1, 1.0, 0.5, 2.0, -0.1, -1.0, 5.0, 0.01] {
                    let bb = vec![g; k];
                    if let Some(v) = ll_at(&bb)? {
                        found = Some((bb, v));
                        break;
                    }
                }
                let (bb, v) = found.ok_or_else(|| rc_err(1400, "could not find feasible values"))?;
                b = bb;
                out.push(format!("feasible:      Log likelihood = {}", fmt_g(v, 10)));
                v
            }
        };
        out.push(String::new());
        let derivs = |b: &[f64], f0: f64| -> Result<(Vec<f64>, Matrix, Vec<Vec<f64>>), String> {
            // per-observation scores and the Hessian by finite differences
            let h: Vec<f64> = b.iter().map(|x| 1e-4 * (x.abs() + 1e-2).max(1e-3)).collect();
            let mut sc = vec![vec![0.0; k]; n];
            let mut g = vec![0.0; k];
            let mut hm = Matrix::zeros(k, k);
            let mut up = Vec::new();
            let mut dn = Vec::new();
            for j in 0..k {
                let mut bp = b.to_vec();
                bp[j] += h[j];
                let u = values(ds, st, &e, &bp, &rows)?;
                bp[j] = b[j] - h[j];
                let d = values(ds, st, &e, &bp, &rows)?;
                if u.iter().chain(&d).any(|x| is_missing(*x) || !x.is_finite()) {
                    return Err(rc_err(1400, "could not calculate numerical derivatives"));
                }
                for t in 0..n {
                    sc[t][j] = (u[t] - d[t]) / (2.0 * h[j]);
                    g[j] += w[t] * sc[t][j];
                }
                let su: f64 = u.iter().zip(&w).map(|(a, w)| a * w).sum();
                let sd: f64 = d.iter().zip(&w).map(|(a, w)| a * w).sum();
                hm[(j, j)] = -(su - 2.0 * f0 + sd) / (h[j] * h[j]);
                up.push(su);
                dn.push(sd);
            }
            for j in 0..k {
                for l in j + 1..k {
                    let f = |sj: f64, sl: f64| -> Result<f64, String> {
                        let mut bp = b.to_vec();
                        bp[j] += sj * h[j];
                        bp[l] += sl * h[l];
                        Ok(values(ds, st, &e, &bp, &rows)?.iter().zip(&w).map(|(a, w)| a * w).sum())
                    };
                    let v = -(f(1.0, 1.0)? - f(1.0, -1.0)? - f(-1.0, 1.0)? + f(-1.0, -1.0)?) / (4.0 * h[j] * h[l]);
                    hm[(j, l)] = v;
                    hm[(l, j)] = v;
                }
            }
            Ok((g, hm, sc))
        };
        let iterate = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let mut converged = false;
        let mut it = 0;
        let mut ilog = Vec::new();
        loop {
            out.push(format!("Iteration {it}:{}Log likelihood = {}", if it < 10 { "  " } else { " " }, fmt_g(ll, 10)));
            ilog.push(ll);
            if it >= iterate {
                break;
            }
            let (g, h, _) = derivs(&b, ll)?;
            let concave = h.cholesky().is_ok();
            let mut hh = h.clone();
            if !concave {
                if let Some(l) = out.last_mut() {
                    l.push_str("  (not concave)");
                }
                for lam in [1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3] {
                    let mut t = h.clone();
                    for i in 0..k {
                        t[(i, i)] += lam * h[(i, i)].abs().max(1e-8);
                    }
                    if t.cholesky().is_ok() {
                        hh = t;
                        break;
                    }
                }
            }
            let (inv, _) = hh.sym_inverse(1e-14);
            let d: Vec<f64> = (0..k).map(|i| (0..k).map(|c| inv[(i, c)] * g[c]).sum()).collect();
            let gnorm: f64 = d.iter().zip(&g).map(|(a, c)| a * c).sum::<f64>().abs();
            let mut lam = 1.0;
            let mut acc = None;
            for _ in 0..60 {
                let nb: Vec<f64> = b.iter().zip(&d).map(|(x, y)| x + lam * y).collect();
                if let Some(v) = ll_at(&nb)? {
                    if v >= ll - 1e-12 * ll.abs().max(1.0) {
                        acc = Some((nb, v));
                        break;
                    }
                }
                lam /= 2.0;
            }
            let Some((nb, v)) = acc else { break };
            let dll = (v - ll).abs();
            b = nb;
            ll = v;
            it += 1;
            if concave && (gnorm < 1e-9 || (gnorm < 1e-5 && dll < 1e-15 * ll.abs().max(1.0))) {
                out.push(format!("Iteration {it}:{}Log likelihood = {}", if it < 10 { "  " } else { " " }, fmt_g(ll, 10)));
                ilog.push(ll);
                converged = true;
                break;
            }
        }
        let (_, h, sc) = derivs(&b, ll)?;
        let (vinv, _) = h.sym_inverse(1e-14);
        let robust = cmd.opt("vce", 3).is_some_and(|v| v.trim().starts_with('r'));
        let v = if robust {
            let mut meat = Matrix::zeros(k, k);
            for t in 0..n {
                for a in 0..k {
                    for c in 0..k {
                        meat[(a, c)] += w[t] * w[t] * sc[t][a] * sc[t][c];
                    }
                }
            }
            vinv.mul(&meat)?.mul(&vinv)?.scale(n_eff / (n_eff - 1.0))
        } else {
            vinv
        };
        if cmd.has_opt("nolog", 4) {
            out.clear();
        }
        let level = util::level(cmd, st)?;
        out.push(String::new());
        let title = cmd.opt("title", 2).map(util::unquote).unwrap_or_else(|| "Maximum likelihood estimation".into());
        out.push(title.clone());
        out.push(String::new());
        out.push(format!("{:<51}{:<13} = {:>10}", format!("Log likelihood = {}", fmt_g(ll, 10)), "Number of obs", util::commas(n_eff as usize)));
        out.push(String::new());
        if robust {
            out.push(format!("{:>13}{:>23}", "|", "Robust"));
        }
        out.extend(table("", &p.display(), &b, &v, None, level));
        if !converged {
            out.push("convergence not achieved".into());
        }
        let names = p.stored();
        let mut res = Results::new();
        res.set("N", n_eff);
        res.set("ll", ll);
        res.set("k", k as f64);
        res.set("rank", k as f64);
        res.set("ic", it as f64);
        res.set("converged", if converged { 1.0 } else { 0.0 });
        res.set_str("cmd", "mlexp");
        res.set_str("lexp", inner.to_string());
        res.set_str("title", title);
        res.set_str("properties", "b V");
        res.set_str("vce", if robust { "robust" } else { "oim" });
        if robust {
            res.set_str("vcetype", "Robust");
        }
        res.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        res.set_mat("V", v.with_names(names.clone(), names));
        res.set_mat("ilog", Matrix::row_vector(&ilog));
        res.sample = Some(mask);
        st.e = res;
        Ok(out.join("\n"))
    }
}
