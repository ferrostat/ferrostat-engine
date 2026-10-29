//! Remaining base-reference commands: the command list and the `predict`
//! hook of the family. Estimators live in `r_trunc` (truncated samples),
//! `r_ord` (zero-inflated ordered and stereotype models), `r_more`
//! (vwls, ivqregress, suest, tetrachoric, ...) and `r_tools` (tests,
//! descriptive tools, settings and informational commands).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.extend(crate::r_trunc::handlers());
    v.extend(crate::r_ord::handlers());
    v.extend(crate::r_tools::handlers());
    v.extend(crate::r_more::handlers());
    v
}

/// `predict` after the commands of this family (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if let Some(r) = crate::r_trunc::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    if let Some(r) = crate::r_ord::predict(ds, st, cmd, &ecmd) {
        return r.map(Some);
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// estat dwatson / durbinalt / bgodfrey / archlm after regress
// ---------------------------------------------------------------------------

use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Results;
use ferrostat_core::missing::is_missing;

/// Residuals of the last regression ordered by time: (time, residual,
/// regressor row) for the estimation sample.
fn ts_resid(ds: &DataSet, st: &Store) -> Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>), String> {
    let (panel, tvar) = st.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset varname ..."))?;
    if panel.is_some() {
        return Err(rc_err(459, "not appropriate with panel data"));
    }
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
    let y = crate::util::col(ds, &dep)?;
    let t = crate::util::col(ds, &tvar)?;
    let env = Env::with_store(ds, st);
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    rows.sort_by(|&a, &c| t[a].total_cmp(&t[c]));
    let mut tt = Vec::new();
    let mut e = Vec::new();
    let mut xs = Vec::new();
    for &i in &rows {
        let x: Vec<f64> = b.colnames.iter().map(|c| crate::postest::coef_value(&env, c, i)).collect::<Result<_, _>>()?;
        if x.iter().any(|v| is_missing(*v)) {
            continue;
        }
        let xb: f64 = x.iter().enumerate().map(|(j, v)| v * b.get(0, j)).sum();
        tt.push(t[i]);
        e.push(y[i] - xb);
        xs.push(x);
    }
    Ok((tt, e, xs))
}

/// OLS of y on columns X (rows); returns (RSS, R2 (centred), coef, V).
fn ols_rows(y: &[f64], x: &[Vec<f64>]) -> (f64, f64, Vec<f64>, Matrix) {
    let n = y.len();
    let k = x.first().map_or(0, |r| r.len());
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for t in 0..n {
        for a in 0..k {
            xty[a] += x[t][a] * y[t];
            for c in 0..k {
                xtx[(a, c)] += x[t][a] * x[t][c];
            }
        }
    }
    let (inv, dropped) = xtx.sym_inverse(1e-12);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
    let rss: f64 = (0..n).map(|t| (y[t] - (0..k).map(|a| x[t][a] * b[a]).sum::<f64>()).powi(2)).sum();
    let m = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|v| (v - m).powi(2)).sum();
    let df = n as f64 - (k - dropped.len()) as f64;
    (rss, 1.0 - rss / tss, b, inv.scale(rss / df))
}

fn lag_of(tt: &[f64], v: &[f64], j: usize, t: usize) -> Option<f64> {
    let target = tt[t] - j as f64;
    tt.iter().position(|x| (*x - target).abs() < 1e-9).map(|p| v[p])
}

fn lag_list(cmd: &Command) -> Result<Vec<usize>, String> {
    match cmd.opt("lags", 1) {
        None => Ok(vec![1]),
        Some(s) => {
            let mut out = Vec::new();
            for tok in s.split(|c: char| c.is_whitespace() || c == ',').filter(|x| !x.is_empty()) {
                match tok.split_once('/') {
                    Some((a, b)) => {
                        let (a, b): (usize, usize) = (a.parse().map_err(|_| rc_err(198, "lags() invalid"))?, b.parse().map_err(|_| rc_err(198, "lags() invalid"))?);
                        out.extend(a..=b);
                    }
                    None => out.push(tok.parse().map_err(|_| rc_err(198, "lags() invalid"))?),
                }
            }
            Ok(out)
        }
    }
}

fn lag_table(title: &str, rows: &[(usize, f64, f64, f64, f64)], small: bool, foot: &str) -> String {
    let mut out = vec![title.to_string(), "---------------------------------------------------------------------------".to_string()];
    out.push(format!("    lags(p)  |  {:>12}            {:>6}               {}", if small { "F" } else { "chi2" }, "df", if small { "  Prob > F" } else { "Prob > chi2" }));
    out.push("-------------+-------------------------------------------------------------".to_string());
    for (p, stat, df1, df2, pv) in rows {
        let df = if small { format!("( {:>2}, {:>4} )", fmt_g(*df1, 4), fmt_g(*df2, 6)) } else { format!("{:>5}", fmt_g(*df1, 4)) };
        out.push(format!("{:>8}     |  {:>12.3}         {:^16}           {:.4}", p, stat, df, pv));
    }
    out.push("---------------------------------------------------------------------------".to_string());
    out.push(foot.to_string());
    out.join("\n")
}

pub fn estat(ds: &DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("regress") {
        return None;
    }
    let which = match sub {
        "dwatson" | "dwa" | "dwat" | "dwats" | "dwatso" => "dw",
        s if s.len() >= 3 && "durbinalt".starts_with(s) => "da",
        s if s.len() >= 3 && "bgodfrey".starts_with(s) => "bg",
        "archlm" => "arch",
        _ => return None,
    };
    Some((|| -> Result<String, String> {
        let (tt, e, xs) = ts_resid(ds, st)?;
        let n = e.len();
        let k = st.e.get("rank").unwrap_or(xs.first().map_or(0, |x| x.len()) as f64);
        let mut r = Results::new();
        if which == "dw" {
            let mut num = 0.0;
            for t in 1..n {
                if let Some(l) = lag_of(&tt, &e, 1, t) {
                    num += (e[t] - l).powi(2);
                }
            }
            let den: f64 = e.iter().map(|v| v * v).sum();
            let d = num / den;
            r.set("N", n as f64);
            r.set("k", k);
            r.set("dw", d);
            st.r = r;
            return Ok(format!("\nDurbin--Watson d-statistic({:>3},{:>6}) = {:>9}", k, n, fmt_g(d, 9)));
        }
        let small = cmd.has_opt("small", 1);
        let lags = lag_list(cmd)?;
        let mut rows = Vec::new();
        let mut chi = Vec::new();
        for &p in &lags {
            if which == "arch" {
                let e2: Vec<f64> = e.iter().map(|v| v * v).collect();
                let mut yy = Vec::new();
                let mut xx = Vec::new();
                for t in 0..n {
                    let mut row = vec![1.0];
                    let mut ok = true;
                    for j in 1..=p {
                        match lag_of(&tt, &e2, j, t) {
                            Some(v) => row.push(v),
                            None => ok = false,
                        }
                    }
                    if ok {
                        yy.push(e2[t]);
                        xx.push(row);
                    }
                }
                let (_, r2, _, _) = ols_rows(&yy, &xx);
                let lm = yy.len() as f64 * r2;
                rows.push((p, lm, p as f64, 0.0, crate::dist::chi2tail(p as f64, lm)));
                chi.push(lm);
                continue;
            }
            let nomiss0 = cmd.has_opt("nomiss0", 3);
            let mut yy = Vec::new();
            let mut xx = Vec::new();
            for t in 0..n {
                let mut row = xs[t].clone();
                let mut ok = true;
                for j in 1..=p {
                    match lag_of(&tt, &e, j, t) {
                        Some(v) => row.push(v),
                        None if nomiss0 => ok = false,
                        None => row.push(0.0),
                    }
                }
                if ok {
                    yy.push(e[t]);
                    xx.push(row);
                }
            }
            let nn = yy.len() as f64;
            let (_, r2, _, _) = ols_rows(&yy, &xx);
            let dfr = nn - k - p as f64;
            let (stat, pv) = if which == "bg" {
                let lm = nn * r2;
                if small { (lm / p as f64, crate::dist::ftail(p as f64, dfr, lm / p as f64)) } else { (lm, crate::dist::chi2tail(p as f64, lm)) }
            } else {
                let f = dfr / p as f64 * r2 / (1.0 - r2);
                if small { (f, crate::dist::ftail(p as f64, dfr, f)) } else { (f * p as f64, crate::dist::chi2tail(p as f64, f * p as f64)) }
            };
            rows.push((p, stat, p as f64, dfr, pv));
            chi.push(stat);
        }
        let title = match which {
            "bg" => "Breusch--Godfrey LM test for autocorrelation",
            "da" => "Durbin's alternative test for autocorrelation",
            _ => "LM test for autoregressive conditional heteroskedasticity (ARCH)",
        };
        let foot = if which == "arch" { "                 H0: no ARCH effects   vs.  H1: ARCH(p) disturbance" } else { "                        H0: no serial correlation" };
        let small = small && which != "arch";
        let text = lag_table(title, &rows, small, foot);
        r.set("N", n as f64);
        r.set("k", k);
        let lm = Matrix::row_vector(&lags.iter().map(|x| *x as f64).collect::<Vec<_>>());
        r.set_mat("lags", lm);
        r.set_mat(if small { "F" } else { "chi2" }, Matrix::row_vector(&chi));
        r.set_mat("p", Matrix::row_vector(&rows.iter().map(|x| x.4).collect::<Vec<_>>()));
        r.set_mat("df", Matrix::row_vector(&rows.iter().map(|x| x.2).collect::<Vec<_>>()));
        if small {
            r.set_mat("df_r", Matrix::row_vector(&rows.iter().map(|x| x.3).collect::<Vec<_>>()));
        }
        st.r = r;
        Ok(text)
    })())
}
