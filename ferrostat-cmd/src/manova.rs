//! Multivariate linear models: `manova`, `manovatest` and `mvreg`.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

/// Fitted multivariate regression: coefficients for every design column
/// (zeros for base/omitted), generalized inverse of X'X and the residual
/// SSCP matrix.
struct MvFit {
    d: Design,
    deps: Vec<String>,
    /// k x p
    b: Matrix,
    xtxinv: Matrix,
    e: Matrix,
    df_r: f64,
    /// centered total SSCP
    t: Matrix,
    ys: Vec<Vec<f64>>,
}

fn fit(ds: &DataSet, st: &Store, cmd: &Command, deps: &[String], rhs: &str, noconstant: bool) -> Result<MvFit, String> {
    let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&deps[0]), indep: rhs, noconstant, weights: "", extra: deps.to_vec() })?;
    let ys: Vec<Vec<f64>> = deps.iter().map(|v| util::col(ds, v).map(|c| d.rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
    let k = d.k();
    let p = deps.len();
    let n = d.n();
    let act = d.active();
    let xtx = est::cross(&d, &act);
    let (inv_a, _) = xtx.sym_inverse(1e-13);
    let mut xtxinv = Matrix::zeros(k, k);
    for (a, &i) in act.iter().enumerate() {
        for (c, &j) in act.iter().enumerate() {
            xtxinv[(i, j)] = inv_a[(a, c)];
        }
    }
    let mut b = Matrix::zeros(k, p);
    let mut resid = vec![vec![0.0; n]; p];
    for (m, y) in ys.iter().enumerate() {
        let xty: Vec<f64> = (0..k).map(|j| if d.omitted[j] { 0.0 } else { (0..n).map(|t| d.cols[j][t] * y[t]).sum() }).collect();
        for i in 0..k {
            b[(i, m)] = (0..k).map(|j| xtxinv[(i, j)] * xty[j]).sum();
        }
        for t in 0..n {
            let fit: f64 = (0..k).map(|j| d.cols[j][t] * b[(j, m)]).sum();
            resid[m][t] = y[t] - fit;
        }
    }
    let mut e = Matrix::zeros(p, p);
    let mut tm = Matrix::zeros(p, p);
    let means: Vec<f64> = ys.iter().map(|y| y.iter().sum::<f64>() / n as f64).collect();
    for i in 0..p {
        for j in 0..p {
            e[(i, j)] = (0..n).map(|t| resid[i][t] * resid[j][t]).sum();
            tm[(i, j)] = (0..n).map(|t| (ys[i][t] - means[i]) * (ys[j][t] - means[j])).sum();
        }
    }
    let df_r = (n - act.len()) as f64;
    Ok(MvFit { d, deps: deps.to_vec(), b, xtxinv, e, df_r, t: tm, ys })
}

/// Hypothesis SSCP for L B = 0 (L: q x k); returns (H, rank of L).
fn hyp_sscp(f: &MvFit, l: &Matrix) -> (Matrix, usize) {
    let lb = l.mul(&f.b).unwrap();
    let lxl = l.mul(&f.xtxinv).unwrap().mul(&l.transpose()).unwrap();
    let (inv, dropped) = lxl.sym_inverse(1e-12);
    let h = lb.transpose().mul(&inv).unwrap().mul(&lb).unwrap();
    (h, l.rows - dropped.len())
}

/// One test statistic row: (letter, value, F, df1, df2, p, flag).
type StatRow = (char, f64, f64, f64, f64, f64, char);

/// Wilks, Pillai, Lawley-Hotelling and Roy statistics for H vs E.
fn mv_stats(h: &Matrix, e: &Matrix, q: f64, nu: f64) -> Vec<StatRow> {
    let p = e.rows as f64;
    // eigenvalues of E^-1 H via the Cholesky factor of E
    let lam: Vec<f64> = match e.cholesky().and_then(|l| l.inverse()) {
        Ok(li) => {
            let m = li.mul(h).unwrap().mul(&li.transpose()).unwrap();
            let mut s = m.clone();
            for i in 0..s.rows {
                for j in 0..i {
                    let a = 0.5 * (m[(i, j)] + m[(j, i)]);
                    s[(i, j)] = a;
                    s[(j, i)] = a;
                }
            }
            s.sym_eigen().0.into_iter().map(|x| x.max(0.0)).collect()
        }
        Err(_) => vec![0.0; e.rows],
    };
    let s = p.min(q);
    let m = ((p - q).abs() - 1.0) / 2.0;
    let nn = (nu - p - 1.0) / 2.0;
    let mut out = Vec::new();
    // Wilks' lambda
    let w: f64 = lam.iter().map(|l| 1.0 / (1.0 + l)).product();
    let t = if p * p + q * q - 5.0 > 0.0 { ((p * p * q * q - 4.0) / (p * p + q * q - 5.0)).sqrt() } else { 1.0 };
    let df1 = p * q;
    let ww = nu + q - (p + q + 1.0) / 2.0;
    let df2 = ww * t - (p * q - 2.0) / 2.0;
    let wt = w.powf(1.0 / t);
    let f = (1.0 - wt) / wt * df2 / df1;
    out.push(('W', w, f, df1, df2, dist::ftail(df1, df2, f), if p.min(q) <= 2.0 { 'e' } else { 'a' }));
    // Pillai's trace
    let v: f64 = lam.iter().map(|l| l / (1.0 + l)).sum();
    let (d1, d2) = (s * (2.0 * m + s + 1.0), s * (2.0 * nn + s + 1.0));
    let f = (2.0 * nn + s + 1.0) / (2.0 * m + s + 1.0) * v / (s - v);
    out.push(('P', v, f, d1, d2, dist::ftail(d1, d2, f), if s == 1.0 { 'e' } else { 'a' }));
    // Lawley-Hotelling trace
    let u: f64 = lam.iter().sum();
    let d2l = 2.0 * (s * nn + 1.0);
    let f = d2l * u / (s * s * (2.0 * m + s + 1.0));
    out.push(('L', u, f, d1, d2l, dist::ftail(d1, d2l, f), if s == 1.0 { 'e' } else { 'a' }));
    // Roy's largest root
    let th = lam.iter().cloned().fold(0.0, f64::max);
    let r = p.max(q);
    let (r1, r2) = (r, nu - r + q);
    let f = th * r2 / r1;
    out.push(('R', th, f, r1, r2, dist::ftail(r1, r2, f), if s == 1.0 { 'e' } else { 'u' }));
    out
}

const RULE_IN: &str = "|--------------------------------------------------";

fn legend() -> Vec<String> {
    vec![
        format!("{:27}W = Wilks' lambda      L = Lawley-Hotelling trace", ""),
        format!("{:27}P = Pillai's trace     R = Roy's largest root", ""),
        String::new(),
        format!("{:>24} |  Statistic     df   F(df1,    df2) =    F   Prob>F", "Source"),
        format!("{:14}{}+{}", "", "-".repeat(11), "-".repeat(50)),
    ]
}

fn stat_block(name: &str, rows: &[StatRow], q: f64) -> Vec<String> {
    let mut out = Vec::new();
    for (i, (c, v, f, d1, d2, p, flag)) in rows.iter().enumerate() {
        let src = if i == 0 { crate::summarize::abbrev(name, 24) } else { String::new() };
        let df = if i == 0 { fmt_g(q, 7) } else { String::new() };
        out.push(format!("{:>24} | {} {:>8.4} {:>6} {:>7.1} {:>8.1} {:>8.2} {:>6.4} {}", src, c, v, df, d1, d2, f, p, flag));
    }
    out.push(format!("{:25}{}", "", RULE_IN));
    out
}

fn split_eq(main: &str) -> Result<(String, String), String> {
    let (l, r) = main.split_once('=').ok_or_else(|| rc_err(198, "equal sign required"))?;
    Ok((l.trim().to_string(), r.trim().to_string()))
}

/// In manova, right-hand-side variables are categorical unless `c.`.
fn categorical_terms(rhs: &str) -> Vec<String> {
    rhs.split_whitespace()
        .map(|t| {
            let sep = if t.contains("##") { "##" } else { "#" };
            t.split(sep).map(|p| if p.contains('.') || p.is_empty() { p.to_string() } else { format!("i.{p}") }).collect::<Vec<_>>().join(sep)
        })
        .collect()
}

/// Columns of the design that belong to a term (by name shape).
fn term_columns(d: &Design, term: &str) -> Vec<usize> {
    let want: Vec<String> = term.split('#').map(|p| util::base_name(p).to_string()).collect();
    (0..d.k())
        .filter(|&j| {
            if d.omitted[j] || d.names[j] == "_cons" {
                return false;
            }
            let parts: Vec<String> = d.names[j].split('#').map(|p| util::base_name(p).to_string()).collect();
            parts == want
        })
        .collect()
}

fn select_rows(k: usize, cols: &[usize]) -> Matrix {
    let mut l = Matrix::zeros(cols.len(), k);
    for (r, &j) in cols.iter().enumerate() {
        l[(r, j)] = 1.0;
    }
    l
}

fn post_mv(st: &mut Store, verb: &str, f: &MvFit, extra: &dyn Fn(&mut Results)) {
    let k = f.d.k();
    let p = f.deps.len();
    let mut names = Vec::new();
    let mut b = Vec::new();
    for m in 0..p {
        for j in 0..k {
            names.push(format!("{}:{}", f.deps[m], f.d.names[j]));
            b.push(f.b[(j, m)]);
        }
    }
    let sig = f.e.scale(1.0 / f.df_r);
    let mut v = Matrix::zeros(k * p, k * p);
    for a in 0..p {
        for c in 0..p {
            for i in 0..k {
                for j in 0..k {
                    v[(a * k + i, c * k + j)] = sig[(a, c)] * f.xtxinv[(i, j)];
                }
            }
        }
    }
    let mut d = f.d.clone();
    d.names = names;
    d.depvar = f.deps.join(" ");
    est::post(st, verb, &d, &b, &v, &|e: &mut Results| {
        e.set("df_r", f.df_r);
        e.set("k_eq", p as f64);
        e.set_str("eqnames", f.deps.join(" "));
        let dn = f.deps.clone();
        e.set_mat("E", f.e.clone().with_names(dn.clone(), dn.clone()));
        e.set_mat("Sigma", sig.clone().with_names(dn.clone(), dn));
        e.set_mat("XtXinv", f.xtxinv.clone().with_names(f.d.names.clone(), f.d.names.clone()));
        extra(e);
    });
}

// ---------------------------------------------------------------------------
// manova
// ---------------------------------------------------------------------------

pub struct Manova;

impl CommandHandler for Manova {
    fn name(&self) -> &'static str {
        "manova"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("manova") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (lhs, rhs) = split_eq(&cmd.main)?;
        let deps = crate::varlist::expand(ds, &lhs)?;
        if deps.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let terms = categorical_terms(&rhs);
        let f = fit(ds, st, cmd, &deps, &terms.join(" "), cmd.has_opt("noconstant", 3))?;
        let k = f.d.k();
        let n = f.d.n();
        let mut out = vec![String::new(), format!("{:27}Number of obs = {:>10}", "", n), String::new()];
        out.extend(legend());
        let mut stats = Vec::new();
        // model
        let model_cols: Vec<usize> = (0..k).filter(|&j| !f.d.omitted[j] && f.d.names[j] != "_cons").collect();
        let (hm, qm) = hyp_sscp(&f, &select_rows(k, &model_cols));
        let model_rows = mv_stats(&hm, &f.e, qm as f64, f.df_r);
        let resid = format!("{:>24} | {:>17}", "Residual", fmt_g(f.df_r, 9));
        let mid_rule = format!("{:14}{}+{}", "", "-".repeat(11), "-".repeat(50));
        if terms.len() > 1 {
            out.extend(stat_block("Model", &model_rows, qm as f64));
            out.push(resid.clone());
            out.push(mid_rule.clone());
        }
        for term in &terms {
            let expanded: Vec<String> = if term.contains("##") {
                // a##b: a, b and a#b
                let parts: Vec<&str> = term.split("##").collect();
                let mut v: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
                v.push(parts.join("#"));
                v
            } else {
                vec![term.clone()]
            };
            for t in expanded {
                let cols = term_columns(&f.d, &t);
                let (h, q) = hyp_sscp(&f, &select_rows(k, &cols));
                let rows = mv_stats(&h, &f.e, q as f64, f.df_r);
                let label = t.split('#').map(|p| util::base_name(p).to_string()).collect::<Vec<_>>().join("#");
                out.extend(stat_block(&label, &rows, q as f64));
                stats.push((label, q, rows));
            }
        }
        out.push(resid);
        out.push(mid_rule);
        out.push(format!("{:>24} | {:>17}", "Total", n - 1));
        out.push(format!("{:14}{}", "", "-".repeat(62)));
        out.push(format!("{:27}e = exact, a = approximate, u = upper bound on F", ""));
        let text = out.join("\n");
        let terms_c = terms.clone();
        post_mv(st, "manova", &f, &|e: &mut Results| {
            e.set_str("title", "MANOVA");
            e.set_str("depvar", deps.join(" "));
            e.set_str("indepvars", terms_c.join(" "));
            e.set("df_m", qm as f64);
            e.set_mat("stat_m", Matrix::from_rows(&[model_rows.iter().map(|r| r.1).collect(), model_rows.iter().map(|r| r.2).collect(), model_rows.iter().map(|r| r.3).collect(), model_rows.iter().map(|r| r.4).collect()]));
            for (i, (label, q, rows)) in stats.iter().enumerate() {
                e.set(&format!("df_{}", i + 1), *q as f64);
                e.set_str(&format!("term_{}", i + 1), label.clone());
                e.set_mat(&format!("stat_{}", i + 1), Matrix::from_rows(&[rows.iter().map(|r| r.1).collect(), rows.iter().map(|r| r.2).collect(), rows.iter().map(|r| r.3).collect(), rows.iter().map(|r| r.4).collect()]));
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// manovatest
// ---------------------------------------------------------------------------

pub struct Manovatest;

/// Rebuild the multivariate fit from the stored e() results.
fn stored(st: &Store) -> Result<(Matrix, Matrix, Matrix, f64, Vec<String>), String> {
    let e = &st.e;
    if e.macros.get("cmd").map(String::as_str) != Some("manova") {
        return Err(rc_err(301, "last estimates not found"));
    }
    let xtxinv = e.matrices.get("XtXinv").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let em = e.matrices.get("E").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let b = e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let k = xtxinv.rows;
    let p = em.rows;
    let mut bm = Matrix::zeros(k, p);
    for m in 0..p {
        for j in 0..k {
            bm[(j, m)] = b.get(0, m * k + j);
        }
    }
    let names = xtxinv.colnames.clone();
    Ok((bm, xtxinv, em, e.get("df_r").unwrap_or(0.0), names))
}

fn order_label(name: &str) -> String {
    if name == "_cons" {
        return name.into();
    }
    name.split('#')
        .map(|p| match p.split_once('.') {
            Some((l, v)) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => format!("({}=={})", v, l.trim_end_matches(['b', 'o', 'n'])),
            _ => p.to_string(),
        })
        .collect::<Vec<_>>()
        .join("*")
}

impl CommandHandler for Manovatest {
    fn name(&self) -> &'static str {
        "manovatest"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (b, xtxinv, e, df_r, names) = stored(st)?;
        let k = names.len();
        let mut out = Vec::new();
        if cmd.has_opt("showorder", 5) {
            out.push(String::new());
            out.push(" Order of columns in the design matrix".to_string());
            for (j, nm) in names.iter().enumerate() {
                out.push(format!("{:>7}: {}", j + 1, order_label(nm)));
            }
        }
        let fake = |cols: &Vec<usize>| select_rows(k, cols);
        let (l, label): (Matrix, String) = if let Some(mname) = cmd.opt("test", 1) {
            let m = st.matrices.get(mname.trim()).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", mname.trim())))?;
            if m.cols != k {
                return Err(rc_err(503, "conformability error"));
            }
            out.push(String::new());
            out.push(format!(" Test constraint{}", if m.rows > 1 { "s" } else { "" }));
            for r in 0..m.rows {
                let mut terms = Vec::new();
                for j in 0..k {
                    let c = m[(r, j)];
                    if c == 0.0 {
                        continue;
                    }
                    let nm = strip_base(&names[j]);
                    let mag = c.abs();
                    let coef = if mag == 1.0 { String::new() } else { format!("{}*", fmt_g(mag, 8)) };
                    let sign = if c < 0.0 { "-" } else { "+" };
                    if terms.is_empty() {
                        terms.push(format!("{}{coef}{nm}", if c < 0.0 { "-" } else { "" }));
                    } else {
                        terms.push(format!("{sign} {coef}{nm}"));
                    }
                }
                out.push(format!(" ({}) {:>4}{} = 0", r + 1, "", terms.join(" ")));
            }
            (m, "manovatest".to_string())
        } else if !cmd.main.trim().is_empty() {
            let term = cmd.main.trim().to_string();
            let mut d = Design {
                names: names.clone(),
                cols: vec![Vec::new(); k],
                omitted: names.iter().map(|n| n.contains("b.") || n.starts_with("o.")).collect(),
                is_base: vec![false; k],
                rows: vec![],
                y: vec![],
                w: vec![],
                wkind: None,
                has_cons: true,
                depvar: String::new(),
                mask: vec![],
                notes: vec![],
                cluster: None,
            };
            d.is_base = d.omitted.clone();
            let cols = term_columns(&d, &term.split('#').map(|p| if p.contains('.') { p.to_string() } else { format!("i.{p}") }).collect::<Vec<_>>().join("#"));
            if cols.is_empty() {
                return Err(rc_err(111, format!("{term} not found in the model")));
            }
            (fake(&cols), term)
        } else if cmd.has_opt("showorder", 5) {
            return Ok(out.join("\n"));
        } else {
            return Err(rc_err(198, "term or test() required"));
        };
        let f = MvFit {
            d: Design { names: names.clone(), cols: vec![], omitted: vec![], is_base: vec![], rows: vec![], y: vec![], w: vec![], wkind: None, has_cons: true, depvar: String::new(), mask: vec![], notes: vec![], cluster: None },
            deps: vec![],
            b,
            xtxinv,
            e: e.clone(),
            df_r,
            t: Matrix::zeros(0, 0),
            ys: vec![],
        };
        let (h, q) = hyp_sscp(&f, &l);
        let rows = mv_stats(&h, &e, q as f64, df_r);
        out.push(String::new());
        out.extend(legend());
        out.extend(stat_block(&label, &rows, q as f64));
        out.push(format!("{:>24} | {:>17}", "Residual", fmt_g(df_r, 9)));
        out.push(format!("{:14}{}", "", "-".repeat(62)));
        out.push(format!("{:27}e = exact, a = approximate, u = upper bound on F", ""));
        let r = st.begin_r();
        r.set("df", q as f64);
        r.set("df_r", df_r);
        r.set_mat("stat", Matrix::from_rows(&[rows.iter().map(|r| r.1).collect(), rows.iter().map(|r| r.2).collect(), rows.iter().map(|r| r.3).collect(), rows.iter().map(|r| r.4).collect()]));
        r.set_mat("H", h);
        r.set_mat("E", e);
        Ok(out.join("\n"))
    }
}

fn strip_base(n: &str) -> String {
    n.split('#')
        .map(|p| match p.split_once('.') {
            Some((l, v)) if l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok() => format!("{}.{v}", l.trim_end_matches(['b', 'o', 'n'])),
            _ => p.to_string(),
        })
        .collect::<Vec<_>>()
        .join("#")
}

// ---------------------------------------------------------------------------
// mvreg
// ---------------------------------------------------------------------------

pub struct Mvreg;

impl CommandHandler for Mvreg {
    fn name(&self) -> &'static str {
        "mvreg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("mvreg") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (lhs, rhs) = split_eq(&cmd.main)?;
        let deps = crate::varlist::expand(ds, &lhs)?;
        let f = fit(ds, st, cmd, &deps, &rhs, cmd.has_opt("noconstant", 3))?;
        let k = f.d.k();
        let n = f.d.n();
        let p = deps.len();
        let nparm = f.d.active().len();
        let level = util::level(cmd, st)?;
        let mut out = vec![
            String::new(),
            format!("{:<12}{:>9}{:>8}{:>12}{:>10}{:>11}{:>9}", "Equation", "Obs", "Parms", "RMSE", "\"R-sq\"", "F", "P>F"),
            "-".repeat(71),
        ];
        let df_m = (nparm - if f.d.has_cons { 1 } else { 0 }) as f64;
        let mut eqstats = Vec::new();
        for m in 0..p {
            let rss = f.e[(m, m)];
            let tss = if f.d.has_cons { f.t[(m, m)] } else { f.ys[m].iter().map(|y| y * y).sum() };
            let r2 = 1.0 - rss / tss;
            let rmse = (rss / f.df_r).sqrt();
            let fs = ((tss - rss) / df_m) / (rss / f.df_r);
            let pf = dist::ftail(df_m, f.df_r, fs);
            out.push(format!("{:<12}{:>9}{:>8}{:>12}{:>10.4}{:>11}{:>9.4}", crate::summarize::abbrev(&deps[m], 12), n, nparm, fmt_g(rmse, 8), r2, fmt_g(fs, 8), pf));
            eqstats.push((rmse, r2, fs, pf));
        }
        out.push(String::new());
        // coefficient table
        let sig = f.e.scale(1.0 / f.df_r);
        let mut names = Vec::new();
        let mut b = Vec::new();
        let mut omitted = Vec::new();
        let mut is_base = Vec::new();
        for m in 0..p {
            for j in 0..k {
                names.push(format!("{}:{}", deps[m], f.d.names[j]));
                b.push(f.b[(j, m)]);
                omitted.push(f.d.omitted[j]);
                is_base.push(f.d.is_base[j]);
            }
        }
        let mut v = Matrix::zeros(k * p, k * p);
        for a in 0..p {
            for c in 0..p {
                for i in 0..k {
                    for j in 0..k {
                        v[(a * k + i, c * k + j)] = sig[(a, c)] * f.xtxinv[(i, j)];
                    }
                }
            }
        }
        out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: Some(f.df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        post_mv(st, "mvreg", &f, &|e: &mut Results| {
            e.set_str("title", "Multivariate regression");
            e.set("df_m", df_m);
            for (m, (rmse, r2, fs, pf)) in eqstats.iter().enumerate() {
                e.set(&format!("rmse_{}", m + 1), *rmse);
                e.set(&format!("r2_{}", m + 1), *r2);
                e.set(&format!("F_{}", m + 1), *fs);
                e.set(&format!("p_F_{}", m + 1), *pf);
            }
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
