//! Smaller multivariate commands: `canon`, `alpha`, `hotelling`, `mvtest`
//! (means, covariances, correlations, normality) and `procrustes`.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::mvutil::{self, sig};
use crate::store::Store;
use crate::util;

fn cols_of(ds: &DataSet, vars: &[String], rows: &[usize]) -> Result<Vec<Vec<f64>>, String> {
    vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect()
}

/// Mean vector and covariance (divisor `n - ddof`).
fn mean_cov(x: &[Vec<f64>], ddof: f64) -> (Vec<f64>, Matrix) {
    let p = x.len();
    let n = x.first().map(|c| c.len()).unwrap_or(0);
    let m: Vec<f64> = x.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
    let mut s = Matrix::zeros(p, p);
    for a in 0..p {
        for b in 0..=a {
            let v = (0..n).map(|t| (x[a][t] - m[a]) * (x[b][t] - m[b])).sum::<f64>() / (n as f64 - ddof);
            s[(a, b)] = v;
            s[(b, a)] = v;
        }
    }
    (m, s)
}

/// Wilks, Pillai, Lawley-Hotelling, Roy for eigenvalues of E^-1 H.
/// Returns rows (name, stat, df1, df2, F, p, flag).
fn mv_stats(lam: &[f64], p: f64, q: f64, nu: f64) -> Vec<(&'static str, f64, f64, f64, f64, f64, char)> {
    let s = p.min(q);
    let m = ((p - q).abs() - 1.0) / 2.0;
    let nn = (nu - p - 1.0) / 2.0;
    let mut out = Vec::new();
    let w: f64 = lam.iter().map(|l| 1.0 / (1.0 + l)).product();
    let t = if p * p + q * q - 5.0 > 0.0 { ((p * p * q * q - 4.0) / (p * p + q * q - 5.0)).sqrt() } else { 1.0 };
    let df1 = p * q;
    let ww = nu + q - (p + q + 1.0) / 2.0;
    let df2 = ww * t - (p * q - 2.0) / 2.0;
    let wt = w.powf(1.0 / t);
    let f = (1.0 - wt) / wt * df2 / df1;
    out.push(("Wilks' lambda", w, df1, df2, f, dist::ftail(df1, df2, f), if s <= 2.0 { 'e' } else { 'a' }));
    let v: f64 = lam.iter().map(|l| l / (1.0 + l)).sum();
    let (d1, d2) = (s * (2.0 * m + s + 1.0), s * (2.0 * nn + s + 1.0));
    let f = (2.0 * nn + s + 1.0) / (2.0 * m + s + 1.0) * v / (s - v);
    out.push(("Pillai's trace", v, d1, d2, f, dist::ftail(d1, d2, f), if s == 1.0 { 'e' } else { 'a' }));
    let u: f64 = lam.iter().sum();
    let d2l = 2.0 * (s * nn + 1.0);
    let f = d2l * u / (s * s * (2.0 * m + s + 1.0));
    out.push(("Lawley-Hotelling trace", u, d1, d2l, f, dist::ftail(d1, d2l, f), if s == 1.0 { 'e' } else { 'a' }));
    let th = lam.iter().cloned().fold(0.0, f64::max);
    let r = p.max(q);
    let (r1, r2) = (r, nu - r + q);
    let f = th * r2 / r1;
    out.push(("Roy's largest root", th, r1, r2, f, dist::ftail(r1, r2, f), if s == 1.0 { 'e' } else { 'u' }));
    out
}

fn stats_table(rows: &[(&'static str, f64, f64, f64, f64, f64, char)]) -> Vec<String> {
    let mut out = vec![format!("    {}", "-".repeat(76))];
    out.push(format!("    {:>22} | {:>10} {:>8} {:>10} {:>11} {:>8}", "", "Statistic", "df1", "df2", "F", "Prob>F"));
    out.push(format!("    {}+{}", "-".repeat(23), "-".repeat(52)));
    for (n, s, d1, d2, f, p, fl) in rows {
        out.push(format!("    {:>22} | {:>10} {:>8} {:>10} {:>11.4} {:>8.4} {}", n, sig(*s, 6), crate::format::fmt_g(*d1, 8), crate::format::fmt_g(*d2, 8), f, p, fl));
    }
    out.push(format!("    {}", "-".repeat(76)));
    out.push(format!("    {:>50}", "e = exact, a = approximate, u = upper bound on F"));
    out
}

// ---------------------------------------------------------------------------
// canon
// ---------------------------------------------------------------------------

pub struct Canon;

fn paren_groups(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                if depth == 1 {
                    cur.clear();
                    continue;
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(cur.trim().to_string());
                    continue;
                }
            }
            _ => {}
        }
        if depth > 0 {
            cur.push(ch);
        }
    }
    out
}

impl CommandHandler for Canon {
    fn name(&self) -> &'static str {
        "canon"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("canon") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return canon_display(st, cmd);
        }
        let g = paren_groups(&cmd.main);
        if g.len() != 2 {
            return Err(rc_err(198, "canon (varlist1) (varlist2)"));
        }
        let v1 = crate::varlist::expand(ds, &g[0])?;
        let v2 = crate::varlist::expand(ds, &g[1])?;
        let mut all = v1.clone();
        all.extend(v2.iter().cloned());
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &all);
        let rows = util::rows(&mask);
        let n = rows.len();
        let x = cols_of(ds, &all, &rows)?;
        let (_, s) = mean_cov(&x, 1.0);
        let (p, q) = (v1.len(), v2.len());
        let sub = |r0: usize, r1: usize, c0: usize, c1: usize| -> Matrix {
            let mut m = Matrix::zeros(r1 - r0, c1 - c0);
            for i in r0..r1 {
                for j in c0..c1 {
                    m[(i - r0, j - c0)] = s[(i, j)];
                }
            }
            m
        };
        let s11 = sub(0, p, 0, p);
        let s22 = sub(p, p + q, p, p + q);
        let s12 = sub(0, p, p, p + q);
        let l1 = s11.cholesky()?;
        let l1i = l1.inverse()?;
        let s22i = mvutil::sym_inv(&s22);
        // M = L1^-1 S12 S22^-1 S21 L1^-T
        let m = l1i.mul(&s12).unwrap().mul(&s22i).unwrap().mul(&s12.transpose()).unwrap().mul(&l1i.transpose()).unwrap();
        let (vals, vecs) = mvutil::sym_eig(&m);
        let k = p.min(q);
        let r: Vec<f64> = vals[..k].iter().map(|v| v.max(0.0).sqrt()).collect();
        let mut a = l1i.transpose().mul(&vecs).unwrap(); // p x p, unit variance columns
        let mut b = Matrix::zeros(q, k);
        for c in 0..k {
            // b = S22^-1 S21 a / r
            let sa: Vec<f64> = (0..q).map(|j| (0..p).map(|i| s12[(i, j)] * a[(i, c)]).sum()).collect();
            for j in 0..q {
                b[(j, c)] = (0..q).map(|l| s22i[(j, l)] * sa[l]).sum::<f64>() / r[c].max(1e-300);
            }
            // orientation: largest standardized coefficient of the second set positive
            let big = (0..q).max_by(|&x, &y| (b[(x, c)] * s22[(x, x)].sqrt()).abs().total_cmp(&(b[(y, c)] * s22[(y, y)].sqrt()).abs())).unwrap_or(0);
            if b[(big, c)] < 0.0 {
                for j in 0..q {
                    b[(j, c)] = -b[(j, c)];
                }
                for i in 0..p {
                    a[(i, c)] = -a[(i, c)];
                }
            }
        }
        let a = mvutil::left_cols(&a, k);
        let lam: Vec<f64> = r.iter().map(|x| x * x / (1.0 - x * x)).collect();
        let tests = mv_stats(&lam, p as f64, q as f64, n as f64 - 1.0 - q as f64);
        let cn: Vec<String> = (1..=k).map(|i| i.to_string()).collect();
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("df", (p * q) as f64);
        e.set_str("cmd", "canon");
        e.set_str("properties", "b V");
        e.set_mat("rawcoef_var1", a.with_names(v1.clone(), cn.clone()));
        e.set_mat("rawcoef_var2", b.with_names(v2.clone(), cn.clone()));
        e.set_mat("ccorr", Matrix::row_vector(&r).with_names(vec!["r1".into()], cn.clone()));
        e.set_mat("stdcoef_var1", {
            let mut m = e.matrices["rawcoef_var1"].clone();
            for i in 0..p {
                for c in 0..k {
                    m[(i, c)] *= s11[(i, i)].sqrt();
                }
            }
            m
        });
        e.set_mat("stdcoef_var2", {
            let mut m = e.matrices["rawcoef_var2"].clone();
            for j in 0..q {
                for c in 0..k {
                    m[(j, c)] *= s22[(j, j)].sqrt();
                }
            }
            m
        });
        e.set("chi2_wilks", tests[0].1);
        for (i, t) in tests.iter().enumerate() {
            let key = ["wilks", "pillai", "lawley", "roy"][i];
            e.set(&format!("stat_{key}"), t.1);
            e.set(&format!("F_{key}"), t.4);
            e.set(&format!("p_{key}"), t.5);
        }
        e.set_mat("stat_m", Matrix::from_rows(&[tests.iter().map(|t| t.1).collect(), tests.iter().map(|t| t.4).collect(), tests.iter().map(|t| t.2).collect(), tests.iter().map(|t| t.3).collect(), tests.iter().map(|t| t.5).collect()]));
        e.sample = Some(mask);
        canon_display(st, cmd)
    }
}

fn canon_display(st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let std_ = cmd.has_opt("stdcoef", 3);
    let (m1, m2) = if std_ { ("stdcoef_var1", "stdcoef_var2") } else { ("rawcoef_var1", "rawcoef_var2") };
    let a = e.matrices.get(m1).cloned().unwrap_or_default();
    let b = e.matrices.get(m2).cloned().unwrap_or_default();
    let r = e.matrices.get("ccorr").map(|m| m.data.clone()).unwrap_or_default();
    let mut out = vec![String::new(), mvutil::hdr("Canonical correlation analysis", "Number of obs", &util::commas(e.get("N").unwrap_or(0.0) as usize))];
    let what = if std_ { "Standardized" } else { "Raw" };
    out.push(String::new());
    out.push(format!("{what} coefficients for the first variable set"));
    out.push(String::new());
    out.extend(mvutil::matlist(&a, "", "%9.4f", 9, false, 4));
    out.push(String::new());
    out.push(format!("{what} coefficients for the second variable set"));
    out.push(String::new());
    out.extend(mvutil::matlist(&b, "", "%9.4f", 9, false, 4));
    out.push(String::new());
    out.push(format!("{}", "-".repeat(78)));
    out.push("Canonical correlations:".into());
    out.push(format!("  {}", r.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join("  ")));
    out.push(format!("{}", "-".repeat(78)));
    if !cmd.has_opt("notests", 4) {
        if let Some(sm) = e.matrices.get("stat_m") {
            let names = ["Wilks' lambda", "Pillai's trace", "Lawley-Hotelling trace", "Roy's largest root"];
            let flags = ['a', 'a', 'a', 'u'];
            let rows: Vec<_> = (0..4).map(|i| (names[i], sm[(0, i)], sm[(2, i)], sm[(3, i)], sm[(1, i)], sm[(4, i)], flags[i])).collect();
            out.push(String::new());
            out.push("Tests of significance of all canonical correlations".into());
            out.push(String::new());
            out.extend(stats_table(&rows));
        }
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// alpha
// ---------------------------------------------------------------------------

pub struct Alpha;

impl CommandHandler for Alpha {
    fn name(&self) -> &'static str {
        "alpha"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let k = vars.len();
        if k < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let std_ = cmd.has_opt("std", 3);
        let casewise = cmd.has_opt("casewise", 4);
        let mut mask = util::touse(ds, st, cmd)?;
        if casewise {
            util::markout(ds, &mut mask, &vars);
        }
        let rows = util::rows(&mask);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
        let miss = |v: f64| ferrostat_core::missing::is_missing(v);
        // pairwise covariances
        let mut cov = Matrix::zeros(k, k);
        let mut nobs = vec![0usize; k];
        for a in 0..k {
            nobs[a] = cols[a].iter().filter(|v| !miss(**v)).count();
            for b in 0..=a {
                let idx: Vec<usize> = (0..rows.len()).filter(|&t| !miss(cols[a][t]) && !miss(cols[b][t])).collect();
                let ma = idx.iter().map(|&t| cols[a][t]).sum::<f64>() / idx.len() as f64;
                let mb = idx.iter().map(|&t| cols[b][t]).sum::<f64>() / idx.len() as f64;
                let c = idx.iter().map(|&t| (cols[a][t] - ma) * (cols[b][t] - mb)).sum::<f64>() / (idx.len() as f64 - 1.0);
                cov[(a, b)] = c;
                cov[(b, a)] = c;
            }
        }
        let mut corr = cov.clone();
        let mut sdv = vec![0.0; k];
        for a in 0..k {
            let idx: Vec<f64> = cols[a].iter().copied().filter(|v| !miss(*v)).collect();
            sdv[a] = util::variance(&idx).sqrt();
        }
        for a in 0..k {
            for b in 0..k {
                // pairwise correlation
                let idx: Vec<usize> = (0..rows.len()).filter(|&t| !miss(cols[a][t]) && !miss(cols[b][t])).collect();
                let xa: Vec<f64> = idx.iter().map(|&t| cols[a][t]).collect();
                let xb: Vec<f64> = idx.iter().map(|&t| cols[b][t]).collect();
                corr[(a, b)] = if a == b { 1.0 } else { cov[(a, b)] / (util::variance(&xa).sqrt() * util::variance(&xb).sqrt()) };
            }
        }
        // signs: from the first principal component unless asis
        let mut sign = vec![1.0; k];
        if !cmd.has_opt("asis", 4) {
            let (_, v) = mvutil::sym_eig(&corr);
            let first = v[(0, 0)].signum();
            for a in 0..k {
                if v[(a, 0)] * first < 0.0 {
                    sign[a] = -1.0;
                }
            }
            if let Some(rev) = cmd.opt("reverse", 3) {
                let rv = crate::varlist::expand(ds, rev)?;
                for (a, v) in vars.iter().enumerate() {
                    sign[a] = if rv.contains(v) { -1.0 } else { 1.0 };
                }
            }
        }
        let m = if std_ { &corr } else { &cov };
        let sm = |a: usize, b: usize| sign[a] * sign[b] * m[(a, b)];
        let stats_for = |items: &[usize]| -> (f64, f64) {
            let kk = items.len() as f64;
            let (mut off, mut np) = (0.0f64, 0.0f64);
            let mut diag = 0.0;
            for &a in items {
                diag += m[(a, a)];
                for &b in items {
                    if a < b {
                        off += sm(a, b);
                        np += 1.0;
                    }
                }
            }
            let avg = off / np.max(1.0);
            let alpha = if std_ { kk * avg / (1.0 + (kk - 1.0) * avg) } else { kk * avg / (diag / kk + (kk - 1.0) * avg) };
            (avg, alpha)
        };
        let all: Vec<usize> = (0..k).collect();
        let (avg, alpha) = stats_for(&all);
        let rev: Vec<&String> = vars.iter().zip(&sign).filter(|(_, s)| **s < 0.0).map(|(v, _)| v).collect();
        let mut out = vec![String::new()];
        out.push(format!("Test scale = mean({}standardized items)", if std_ { "" } else { "un" }));
        if !rev.is_empty() && !cmd.has_opt("item", 3) {
            out.push(format!("Reversed item{}:  {}", if rev.len() > 1 { "s" } else { "" }, rev.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")));
        }
        let mut items_tbl = Vec::new();
        if cmd.has_opt("item", 3) {
            out.push(String::new());
            out.push(format!("{:>48}{:>15}", "", "average"));
            out.push(format!("{:>34}{:>14}{:>15}", "item-test", "item-rest", "interitem"));
            out.push(format!("{:<13} {:>5}  {:>4}  {:>11}  {:>11}  {:>13}  {:>9}", "Item", "Obs", "Sign", "correlation", "correlation", if std_ { "correlation" } else { "covariance" }, "alpha"));
            out.push("-".repeat(79));
            for a in 0..k {
                // item-test / item-rest correlations from the per-observation scale
                // (mean of the available signed, possibly standardized, items)
                let rest: Vec<usize> = all.iter().copied().filter(|&b| b != a).collect();
                let it = corr_scale(&cols, &sign, &sdv, std_, a, &all);
                let ir = corr_scale(&cols, &sign, &sdv, std_, a, &rest);
                let (av, al) = stats_for(&rest);
                out.push(format!("{:<13} {:>5}  {:>4}  {:>11.4}  {:>11.4}  {:>13}  {:>9.4}", crate::summarize::abbrev(&vars[a], 13), nobs[a], if sign[a] > 0.0 { "+" } else { "-" }, it, ir, if std_ { format!("{av:.4}") } else { sig(av, 7) }, al));
                items_tbl.push(vec![it, ir, av, al]);
            }
            out.push("-".repeat(79));
            out.push(format!("{:<13} {:>44}  {:>13}  {:>9.4}", "Test scale", "", if std_ { format!("{avg:.4}") } else { sig(avg, 7) }, alpha));
            out.push("-".repeat(79));
        } else {
            out.push(String::new());
            out.push(format!("Average interitem {}:{:>13}", if std_ { "correlation" } else { "covariance: " }, if std_ { format!("{avg:.4}") } else { sig(avg, 7) }));
            out.push(format!("Number of items in the scale:{:>16}", k));
            out.push(format!("Scale reliability coefficient:{:>15.4}", alpha));
        }
        if let Some(g) = cmd.opt("generate", 3) {
            let n = ds.n_obs;
            let full: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
            let means: Vec<f64> = (0..k).map(|a| util::mean(&cols[a].iter().copied().filter(|v| !miss(*v)).collect::<Vec<_>>())).collect();
            let vals: Vec<f64> = rows
                .iter()
                .map(|&i| {
                    let items: Vec<f64> = (0..k)
                        .filter(|&a| !miss(full[a][i]))
                        .map(|a| {
                            let x = if std_ { (full[a][i] - means[a]) / sdv[a] } else { full[a][i] };
                            if sign[a] < 0.0 { if std_ { -x } else { -x } } else { x }
                        })
                        .collect();
                    if items.is_empty() { SYSMISS } else { items.iter().sum::<f64>() / items.len() as f64 }
                })
                .collect();
            mvutil::put_var(ds, st, g.trim(), None, n, &rows, &vals, &format!("mean({}standardized items)", if std_ { "" } else { "un" }))?;
        }
        let r = st.begin_r();
        r.set("alpha", alpha);
        r.set("k", k as f64);
        if std_ {
            r.set("rho", avg);
        } else {
            r.set("cov", avg);
        }
        if !items_tbl.is_empty() {
            r.set_mat("ItemTestCorr", Matrix::row_vector(&items_tbl.iter().map(|v| v[0]).collect::<Vec<_>>()).with_names(vec!["r1".into()], vars.clone()));
            r.set_mat("ItemRestCorr", Matrix::row_vector(&items_tbl.iter().map(|v| v[1]).collect::<Vec<_>>()).with_names(vec!["r1".into()], vars.clone()));
            r.set_mat("Alpha", Matrix::row_vector(&items_tbl.iter().map(|v| v[3]).collect::<Vec<_>>()).with_names(vec!["r1".into()], vars.clone()));
        }
        Ok(out.join("\n"))
    }
}

/// Correlation of item `a` with the mean of the available items `items`.
fn corr_scale(cols: &[Vec<f64>], sign: &[f64], sdv: &[f64], std_: bool, a: usize, items: &[usize]) -> f64 {
    let miss = ferrostat_core::missing::is_missing;
    let k = cols.len();
    let means: Vec<f64> = (0..k).map(|j| util::mean(&cols[j].iter().copied().filter(|v| !miss(*v)).collect::<Vec<_>>())).collect();
    let val = |j: usize, t: usize| -> f64 {
        let x = cols[j][t];
        let z = if std_ { (x - means[j]) / sdv[j] } else { x };
        sign[j] * z
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..cols[a].len() {
        if miss(cols[a][t]) {
            continue;
        }
        let av: Vec<f64> = items.iter().filter(|&&j| !miss(cols[j][t])).map(|&j| val(j, t)).collect();
        if av.is_empty() {
            continue;
        }
        xs.push(cols[a][t]);
        ys.push(av.iter().sum::<f64>() / av.len() as f64);
    }
    let (mx, my) = (util::mean(&xs), util::mean(&ys));
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let r = sxy / (xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>() * ys.iter().map(|y| (y - my).powi(2)).sum::<f64>()).sqrt();
    r * sign[a]
}

// ---------------------------------------------------------------------------
// hotelling
// ---------------------------------------------------------------------------

pub struct Hotelling;

fn summary_lines(vars: &[String], x: &[Vec<f64>]) -> Vec<String> {
    let mut out = vec![format!("{:>12} | {:>8} {:>10} {:>10} {:>10} {:>10}", "Variable", "Obs", "Mean", "Std. dev.", "Min", "Max"), format!("{}+{}", "-".repeat(13), "-".repeat(52))];
    for (v, c) in vars.iter().zip(x) {
        let (mn, mx) = c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &v| (a.0.min(v), a.1.max(v)));
        out.push(format!("{:>12} | {:>8} {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(v, 12), c.len(), mvutil::g(util::mean(c), 9), mvutil::g(util::variance(c).sqrt(), 9), mvutil::g(mn, 9), mvutil::g(mx, 9)));
    }
    out
}

fn quad(s: &Matrix, d: &[f64]) -> f64 {
    let si = mvutil::sym_inv(s);
    let p = d.len();
    (0..p).map(|a| (0..p).map(|b| d[a] * si[(a, b)] * d[b]).sum::<f64>()).sum()
}

impl CommandHandler for Hotelling {
    fn name(&self) -> &'static str {
        "hotelling"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let p = vars.len() as f64;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let byv = cmd.opt("by", 2).map(|s| s.trim().to_string());
        if let Some(b) = &byv {
            util::markout(ds, &mut mask, std::slice::from_ref(b));
        }
        let rows = util::rows(&mask);
        let mut out = vec![String::new()];
        let notable = cmd.has_opt("notable", 5);
        let (t2, f, df1, df2) = match &byv {
            None => {
                let x = cols_of(ds, &vars, &rows)?;
                let n = rows.len() as f64;
                let (m, s) = mean_cov(&x, 1.0);
                if !notable {
                    out.extend(summary_lines(&vars, &x));
                    out.push(String::new());
                }
                let t2 = n * quad(&s, &m);
                let f = (n - p) / ((n - 1.0) * p) * t2;
                out.push(format!("1-group Hotelling's T-squared = {}", sig(t2, 8)));
                out.push(format!("F test statistic: (({n}-{p})/({n}-1)({p})) x {} = {}", sig(t2, 8), sig(f, 8)));
                out.push(String::new());
                out.push("H0: Vector of means is equal to a vector of zeros".into());
                (t2, f, p, n - p)
            }
            Some(b) => {
                let bc = util::col(ds, b)?;
                let bvar = ds.get_var(b).unwrap();
                let mut levels: Vec<f64> = rows.iter().map(|&i| bc[i]).collect();
                levels.sort_by(|a, c| a.total_cmp(c));
                levels.dedup();
                if levels.len() != 2 {
                    return Err(rc_err(420, format!("{b} must take on exactly 2 values")));
                }
                let mut parts = Vec::new();
                for &lv in &levels {
                    let r: Vec<usize> = rows.iter().copied().filter(|&i| bc[i] == lv).collect();
                    let x = cols_of(ds, &vars, &r)?;
                    if !notable {
                        out.push(format!("-> {} = {}", b, util::cell_text(ds, bvar, r[0], true)));
                        out.extend(summary_lines(&vars, &x));
                        out.push(String::new());
                    }
                    parts.push((r.len() as f64, mean_cov(&x, 1.0)));
                }
                let (n1, (m1, s1)) = &parts[0];
                let (n2, (m2, s2)) = &parts[1];
                let sp = s1.scale((n1 - 1.0) / (n1 + n2 - 2.0)).add(&s2.scale((n2 - 1.0) / (n1 + n2 - 2.0))).unwrap();
                let d: Vec<f64> = m1.iter().zip(m2).map(|(a, c)| a - c).collect();
                let t2 = n1 * n2 / (n1 + n2) * quad(&sp, &d);
                let n = n1 + n2;
                let f = (n - p - 1.0) / ((n - 2.0) * p) * t2;
                out.push(format!("2-group Hotelling's T-squared = {}", sig(t2, 8)));
                out.push(format!("F test statistic: (({n}-{p}-1)/({n}-2)({p})) x {} = {}", sig(t2, 8), sig(f, 8)));
                out.push(String::new());
                out.push("H0: Vectors of means are equal for the two groups".into());
                (t2, f, p, n - p - 1.0)
            }
        };
        let pv = dist::ftail(df1, df2, f);
        out.push(format!("{:>20} = {:>9.4}", format!("F({df1},{df2})"), f));
        out.push(format!("{:>20} = {:>9.4}", format!("Prob > F({df1},{df2})"), pv));
        let r = st.begin_r();
        r.set("N", rows.len() as f64);
        r.set("T2", t2);
        r.set("k", p);
        r.set("F", f);
        r.set("df1", df1);
        r.set("df2", df2);
        r.set("p", pv);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// mvtest
// ---------------------------------------------------------------------------

pub struct Mvtest;

fn groups(ds: &DataSet, byv: &str, rows: &[usize]) -> Result<Vec<Vec<usize>>, String> {
    let bc = util::col(ds, byv)?;
    let mut levels: Vec<f64> = rows.iter().map(|&i| bc[i]).collect();
    levels.sort_by(|a, c| a.total_cmp(c));
    levels.dedup();
    Ok(levels.iter().map(|&l| rows.iter().copied().filter(|&i| bc[i] == l).collect()).collect())
}

fn chi_line(label: &str, df: f64, chi2: f64) -> Vec<String> {
    vec![format!("{:>27} = {:>9.2}", format!("{label}({})", crate::format::fmt_g(df, 8)), chi2), format!("{:>27} = {:>9.4}", "Prob > chi2", dist::chi2tail(df, chi2))]
}

impl CommandHandler for Mvtest {
    fn name(&self) -> &'static str {
        "mvtest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub = sub.to_lowercase();
        let vars = crate::varlist::expand(ds, rest)?;
        let k = vars.len();
        let kf = k as f64;
        let byv = cmd.opt("by", 2).map(|s| s.trim().to_string());
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        if let Some(b) = &byv {
            util::markout(ds, &mut mask, std::slice::from_ref(b));
        }
        let rows = util::rows(&mask);
        let n = rows.len() as f64;
        let x = cols_of(ds, &vars, &rows)?;
        let mut out = vec![String::new()];
        let r = &mut crate::store::Results::new();
        if "means".starts_with(&sub) && sub.len() >= 1 {
            if let Some(b) = &byv {
                let gs = groups(ds, b, &rows)?;
                let m = gs.len();
                let mut e = Matrix::zeros(k, k);
                let mut h = Matrix::zeros(k, k);
                let (gm, _) = mean_cov(&x, 1.0);
                for g in &gs {
                    let xg = cols_of(ds, &vars, g)?;
                    let (mg, sg) = mean_cov(&xg, 1.0);
                    let ng = g.len() as f64;
                    e = e.add(&sg.scale(ng - 1.0)).unwrap();
                    for a in 0..k {
                        for c in 0..k {
                            h[(a, c)] += ng * (mg[a] - gm[a]) * (mg[c] - gm[c]);
                        }
                    }
                }
                let l = e.cholesky()?.inverse()?;
                let mm = l.mul(&h).unwrap().mul(&l.transpose()).unwrap();
                let (lam, _) = mvutil::sym_eig(&mm);
                let lam: Vec<f64> = lam.into_iter().map(|v| v.max(0.0)).collect();
                let rowsx = mv_stats(&lam, kf, (m - 1) as f64, n - m as f64);
                out.push(format!("Test for equality of {m} group means, assuming homogeneity"));
                out.push(String::new());
                out.extend(stats_table(&rowsx));
                r.set("F", rowsx[0].4);
                r.set("df_m", rowsx[0].2);
                r.set("df_r", rowsx[0].3);
                r.set("p_F", rowsx[0].5);
                r.set_mat("stat_m", Matrix::from_rows(&[rowsx.iter().map(|t| t.1).collect(), rowsx.iter().map(|t| t.4).collect(), rowsx.iter().map(|t| t.2).collect(), rowsx.iter().map(|t| t.3).collect(), rowsx.iter().map(|t| t.5).collect()]));
            } else {
                let (m, s) = mean_cov(&x, 1.0);
                // contrast matrix C and target vector
                let (c, target, title) = if cmd.has_opt("equal", 3) {
                    let mut c = Matrix::zeros(k - 1, k);
                    for i in 0..k - 1 {
                        c[(i, i)] = 1.0;
                        c[(i, i + 1)] = -1.0;
                    }
                    (c, vec![0.0; k - 1], "Test that all means are the same".to_string())
                } else if let Some(mn) = cmd.opt("equals", 6) {
                    let v = st.matrices.get(mn.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", mn.trim())))?;
                    (Matrix::identity(k), v.data.clone(), format!("Test that means equal vector {}", mn.trim()))
                } else if let Some(mn) = cmd.opt("linear", 3) {
                    let v = st.matrices.get(mn.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", mn.trim())))?.clone();
                    let mut c = Matrix::zeros(v.rows, k);
                    let mut t = vec![0.0; v.rows];
                    for i in 0..v.rows {
                        for j in 0..k {
                            c[(i, j)] = v[(i, j)];
                        }
                        if v.cols > k {
                            t[i] = v[(i, k)];
                        }
                    }
                    (c, t, format!("Test that mean vector satisfies linear hypothesis {}", mn.trim()))
                } else {
                    (Matrix::identity(k), vec![0.0; k], "Test that all means are 0".to_string())
                };
                let q = c.rows as f64;
                let cm: Vec<f64> = (0..c.rows).map(|i| (0..k).map(|j| c[(i, j)] * m[j]).sum::<f64>() - target[i]).collect();
                let cs = c.mul(&s).unwrap().mul(&c.transpose()).unwrap();
                let t2 = n * quad(&cs, &cm);
                let f = (n - q) / ((n - 1.0) * q) * t2;
                let pv = dist::ftail(q, n - q, f);
                out.push(title);
                out.push(String::new());
                out.push(format!("{:>27} = {:>9.2}", "Hotelling T2", t2));
                out.push(format!("{:>27} = {:>9.2}", format!("Hotelling F({},{})", q, n - q), f));
                out.push(format!("{:>27} = {:>9.4}", "Prob > F", pv));
                r.set("T2", t2);
                r.set("F", f);
                r.set("df_m", q);
                r.set("df_r", n - q);
                r.set("p_F", pv);
            }
        } else if sub.len() >= 3 && "covariances".starts_with(&sub) {
            if let Some(b) = &byv {
                let gs = groups(ds, b, &rows)?;
                let m = gs.len() as f64;
                let mut sp = Matrix::zeros(k, k);
                let mut sum_ld = 0.0;
                let (mut inv1, mut inv2) = (0.0, 0.0);
                for g in &gs {
                    let ng = g.len() as f64;
                    let (_, sg) = mean_cov(&cols_of(ds, &vars, g)?, 1.0);
                    sp = sp.add(&sg.scale(ng - 1.0)).unwrap();
                    sum_ld += (ng - 1.0) * mvutil::ln_det(&sg);
                    inv1 += 1.0 / (ng - 1.0);
                    inv2 += 1.0 / (ng - 1.0).powi(2);
                }
                let sp = sp.scale(1.0 / (n - m));
                let m2 = (n - m) * mvutil::ln_det(&sp) - sum_ld;
                let c1 = (inv1 - 1.0 / (n - m)) * (2.0 * kf * kf + 3.0 * kf - 1.0) / (6.0 * (kf + 1.0) * (m - 1.0));
                let c2 = (inv2 - 1.0 / (n - m).powi(2)) * (kf - 1.0) * (kf + 2.0) / (6.0 * (m - 1.0));
                let a1 = (m - 1.0) * kf * (kf + 1.0) / 2.0;
                let a2 = (a1 + 2.0) / (c2 - c1 * c1).abs();
                let b1 = (1.0 - c1 - a1 / a2) / a1;
                let b2 = (1.0 - c1 + 2.0 / a2) / a2;
                let fstat = if c2 > c1 * c1 { b1 * m2 } else { a2 * b2 * m2 / (a1 * (1.0 - 2.0 * b2 * m2 / 2.0)) };
                let chi2 = (1.0 - c1) * m2;
                out.push(format!("Test of equality of covariance matrices across {} samples", gs.len()));
                out.push(String::new());
                out.push(format!("{:>27} = {:>12.4}", "Modified LR chi2", m2));
                out.push(format!("{:>27} = {:>9.2}    Prob > F = {:.4}", format!("Box F({},{})", a1, sig(a2, 6)), fstat, dist::ftail(a1, a2, fstat)));
                out.push(format!("{:>27} = {:>9.2}    Prob > chi2 = {:.4}", format!("Box chi2({})", a1), chi2, dist::chi2tail(a1, chi2)));
                r.set("chi2", chi2);
                r.set("df", a1);
                r.set("p_chi2", dist::chi2tail(a1, chi2));
                r.set("F_Box", fstat);
                r.set("df_m_Box", a1);
                r.set("df_r_Box", a2);
                r.set("p_F_Box", dist::ftail(a1, a2, fstat));
            } else {
                let (_, s1) = mean_cov(&x, 1.0);
                let s = s1.scale((n - 1.0) / n);
                let (title, chi2, df) = if cmd.has_opt("spherical", 3) {
                    let c = (n - 1.0) - (2.0 * kf * kf + kf + 2.0) / (6.0 * kf);
                    ("Test that covariance matrix is spherical", c * (kf * s.trace().ln() - mvutil::ln_det(&s) - kf * kf.ln()), kf * (kf + 1.0) / 2.0 - 1.0)
                } else if cmd.has_opt("compound", 3) {
                    let s2 = s.trace() / kf;
                    let off: f64 = (0..k).flat_map(|a| (0..k).map(move |b| (a, b))).filter(|(a, b)| a != b).map(|(a, b)| s[(a, b)]).sum();
                    let rr = off / (kf * (kf - 1.0) * s2);
                    let c = n - 1.0 - kf * (kf + 1.0).powi(2) * (2.0 * kf - 3.0) / (6.0 * (kf - 1.0) * (kf * kf + kf - 4.0));
                    ("Test that covariance matrix is compound symmetric", c * (kf * s2.ln() + (kf - 1.0) * (1.0 - rr).ln() + (1.0 + (kf - 1.0) * rr).ln() - mvutil::ln_det(&s)), kf * (kf + 1.0) / 2.0 - 2.0)
                } else if let Some(bl) = cmd.opt("block", 3) {
                    let blocks: Vec<Vec<usize>> = bl.split("||").map(|b| crate::varlist::expand(ds, b).map(|v| v.iter().filter_map(|x| vars.iter().position(|y| y == x)).collect())).collect::<Result<_, _>>()?;
                    let a2: f64 = kf * kf - blocks.iter().map(|b| (b.len() as f64).powi(2)).sum::<f64>();
                    let a3: f64 = kf.powi(3) - blocks.iter().map(|b| (b.len() as f64).powi(3)).sum::<f64>();
                    let mut sl = 0.0;
                    for b in &blocks {
                        let mut sb = Matrix::zeros(b.len(), b.len());
                        for (i, &x1) in b.iter().enumerate() {
                            for (j, &x2) in b.iter().enumerate() {
                                sb[(i, j)] = s[(x1, x2)];
                            }
                        }
                        sl += mvutil::ln_det(&sb);
                    }
                    ("Test that covariance matrix is block diagonal", (n - 1.0 - (2.0 * a3 + 3.0 * a2) / (6.0 * a2)) * (sl - mvutil::ln_det(&s)), a2 / 2.0)
                } else if let Some(mn) = cmd.opt("equals", 6) {
                    let s0 = st.matrices.get(mn.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", mn.trim())))?.clone();
                    let su = s.scale(n / (n - 1.0));
                    let c = (n - 1.0) * (1.0 - 1.0 / (6.0 * (n - 1.0) - 1.0) * (2.0 * kf + 1.0 - 2.0 / (kf + 1.0)));
                    let tr = su.mul(&mvutil::sym_inv(&s0)).unwrap().trace();
                    ("Test that covariance matrix equals matrix", c * (mvutil::ln_det(&s0) - mvutil::ln_det(&su) + tr - kf), kf * (kf + 1.0) / 2.0)
                } else {
                    // diagonal (default)
                    let rr = mvutil::cov_to_corr(&s);
                    ("Test that covariance matrix is diagonal", -(n - 1.0 - (2.0 * kf + 5.0) / 6.0) * mvutil::ln_det(&rr), kf * (kf - 1.0) / 2.0)
                };
                out.push(if title.ends_with("matrix") { format!("{title} {}", cmd.opt("equals", 6).unwrap_or("").trim()) } else { title.to_string() });
                out.push(String::new());
                out.extend(chi_line("Adjusted LR chi2", df, chi2));
                r.set("chi2", chi2);
                r.set("df", df);
                r.set("p_chi2", dist::chi2tail(df, chi2));
            }
        } else if sub.len() >= 3 && "correlations".starts_with(&sub) {
            let jennrich = |rr: &Matrix, r0: &Matrix, nn: f64| -> f64 {
                let r0i = mvutil::sym_inv(r0);
                let z = r0i.mul(&rr.sub(r0).unwrap()).unwrap().scale(nn.sqrt());
                let zz = z.mul(&z).unwrap();
                let dz = z.diag();
                let mut t = Matrix::identity(k);
                for a in 0..k {
                    for b in 0..k {
                        t[(a, b)] += r0[(a, b)] * r0i[(a, b)];
                    }
                }
                0.5 * zz.trace() - quad(&t, &dz)
            };
            let (title, label, chi2, df) = if let Some(b) = &byv {
                let gs = groups(ds, b, &rows)?;
                let mut rbar = Matrix::zeros(k, k);
                let mut rs = Vec::new();
                for g in &gs {
                    let (_, sg) = mean_cov(&cols_of(ds, &vars, g)?, 1.0);
                    let rg = mvutil::cov_to_corr(&sg);
                    rbar = rbar.add(&rg.scale(g.len() as f64 / n)).unwrap();
                    rs.push((g.len() as f64, rg));
                }
                let chi: f64 = rs.iter().map(|(ng, rg)| jennrich(rg, &rbar, *ng)).sum();
                ("Test of equality of correlation matrices across samples".to_string(), "Jennrich chi2", chi, (gs.len() as f64 - 1.0) * kf * (kf - 1.0) / 2.0)
            } else {
                let (_, s) = mean_cov(&x, 1.0);
                let rr = mvutil::cov_to_corr(&s);
                if let Some(mn) = cmd.opt("equals", 6) {
                    let r0 = st.matrices.get(mn.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", mn.trim())))?.clone();
                    (format!("Test that correlation matrix equals specified pattern {}", mn.trim()), "Jennrich chi2", jennrich(&rr, &r0, n), kf * (kf - 1.0) / 2.0)
                } else {
                    let mut rbar = 0.0;
                    for a in 1..k {
                        for b in 0..a {
                            rbar += rr[(a, b)];
                        }
                    }
                    rbar *= 2.0 / (kf * (kf - 1.0));
                    let rh: Vec<f64> = (0..k).map(|h| (0..k).filter(|&i| i != h).map(|i| rr[(i, h)]).sum::<f64>() / (kf - 1.0)).collect();
                    let u = (kf - 1.0).powi(2) * (1.0 - (1.0 - rbar).powi(2)) / (kf - (kf - 2.0) * (1.0 - rbar).powi(2));
                    let mut s1 = 0.0;
                    for a in 1..k {
                        for b in 0..a {
                            s1 += (rr[(a, b)] - rbar).powi(2);
                        }
                    }
                    let s2: f64 = rh.iter().map(|v| (v - rbar).powi(2)).sum();
                    ("Test that correlation matrix is compound symmetric (all correlations equal)".to_string(), "Lawley chi2", (n - 1.0) / (1.0 - rbar).powi(2) * (s1 - u * s2), (kf - 2.0) * (kf + 1.0) / 2.0)
                }
            };
            out.push(title);
            out.push(String::new());
            out.extend(chi_line(label, df, chi2));
            r.set("chi2", chi2);
            r.set("df", df);
            r.set("p_chi2", dist::chi2tail(df, chi2));
        } else if sub.len() >= 4 && "normality".starts_with(&sub) {
            let (m, s1) = mean_cov(&x, 1.0);
            let s = s1.scale((n - 1.0) / n);
            let si = mvutil::sym_inv(&s);
            let nn = rows.len();
            let dev: Vec<Vec<f64>> = (0..nn).map(|t| (0..k).map(|j| x[j][t] - m[j]).collect()).collect();
            let gij = |i: usize, j: usize| -> f64 { (0..k).map(|a| (0..k).map(|b| dev[i][a] * si[(a, b)] * dev[j][b]).sum::<f64>()).sum() };
            let mut g = vec![vec![0.0; nn]; nn];
            for i in 0..nn {
                for j in 0..=i {
                    let v = gij(i, j);
                    g[i][j] = v;
                    g[j][i] = v;
                }
            }
            let b1: f64 = g.iter().flatten().map(|v| v.powi(3)).sum::<f64>() / (n * n);
            let b2: f64 = (0..nn).map(|i| g[i][i].powi(2)).sum::<f64>() / n;
            let z1 = (kf + 1.0) * (n + 1.0) * (n + 3.0) / (6.0 * ((n + 1.0) * (kf + 1.0) - 6.0)) * b1;
            let df1 = kf * (kf + 1.0) * (kf + 2.0) / 6.0;
            let z2 = (b2 - kf * (kf + 2.0)) / (8.0 * kf * (kf + 2.0) / n).sqrt();
            // Henze-Zirkler
            let beta = (n * (2.0 * kf + 1.0) / 4.0).powf(1.0 / (kf + 4.0)) / 2f64.sqrt();
            let b2_ = beta * beta;
            let mut t1 = 0.0;
            for i in 0..nn {
                for j in 0..nn {
                    let dij = g[i][i] + g[j][j] - 2.0 * g[i][j];
                    t1 += (-b2_ / 2.0 * dij).exp();
                }
            }
            let t2: f64 = (0..nn).map(|i| (-b2_ / (2.0 * (1.0 + b2_)) * g[i][i]).exp()).sum();
            let hz = t1 / n - 2.0 * (1.0 + b2_).powf(-kf / 2.0) * t2 + n * (1.0 + 2.0 * b2_).powf(-kf / 2.0);
            let et = 1.0 - (1.0 + 2.0 * b2_).powf(-kf / 2.0) * (1.0 + kf * b2_ / (1.0 + 2.0 * b2_) + kf * (kf + 2.0) * b2_ * b2_ / (2.0 * (1.0 + 2.0 * b2_).powi(2)));
            let w = (1.0 + b2_) * (1.0 + 3.0 * b2_);
            let vt = 2.0 * (1.0 + 4.0 * b2_).powf(-kf / 2.0) + 2.0 * (1.0 + 2.0 * b2_).powf(-kf) * (1.0 + 2.0 * kf * b2_.powi(2) / (1.0 + 2.0 * b2_).powi(2) + 3.0 * kf * (kf + 2.0) * b2_.powi(4) / (4.0 * (1.0 + 2.0 * b2_).powi(4))) - 4.0 * w.powf(-kf / 2.0) * (1.0 + 3.0 * kf * b2_.powi(2) / (2.0 * w) + kf * (kf + 2.0) * b2_.powi(4) / (2.0 * w * w));
            let vz = (1.0 + vt / (et * et)).ln();
            let ez = et.ln() - vz / 2.0;
            let zhz = (hz.ln() - ez) / vz.sqrt();
            // Doornik-Hansen
            let rr = mvutil::cov_to_corr(&s);
            let (lam, hm) = rr.sym_eigen();
            let sd: Vec<f64> = (0..k).map(|j| s[(j, j)].sqrt()).collect();
            let mut tr = Matrix::zeros(k, k);
            for a in 0..k {
                for b in 0..k {
                    tr[(a, b)] = (0..k).map(|c| hm[(a, c)] * hm[(b, c)] / lam[c].max(1e-300).sqrt()).sum();
                }
            }
            let mut dh = 0.0;
            for c in 0..k {
                let z: Vec<f64> = (0..nn).map(|t| (0..k).map(|a| dev[t][a] / sd[a] * tr[(a, c)]).sum()).collect();
                let (zz1, zz2) = dh_z(&z);
                dh += zz1 * zz1 + zz2 * zz2;
            }
            let stats = cmd.opt("stats", 2).map(|s| s.to_lowercase()).unwrap_or_else(|| "dhansen".into());
            let all = stats.contains("all");
            out.push("Test for multivariate normality".into());
            out.push(String::new());
            if all || stats.contains("mskew") {
                out.push(format!("    Mardia mSkewness = {:>10}    chi2({}) = {:>9.3}    Prob>chi2 = {:.4}", sig(b1, 7), df1, z1, dist::chi2tail(df1, z1)));
            }
            if all || stats.contains("mkurt") {
                out.push(format!("    Mardia mKurtosis = {:>10}    chi2(1) = {:>9.3}    Prob>chi2 = {:.4}", sig(b2, 7), z2 * z2, dist::chi2tail(1.0, z2 * z2)));
            }
            if all || stats.contains("hz") {
                out.push(format!("    Henze-Zirkler    = {:>10}    chi2(1) = {:>9.3}    Prob>chi2 = {:.4}", sig(hz, 7), zhz * zhz, dist::chi2tail(1.0, zhz * zhz)));
            }
            if all || stats.contains("dh") {
                out.push(format!("    Doornik-Hansen                  chi2({}) = {:>9.3}    Prob>chi2 = {:.4}", 2 * k, dh, dist::chi2tail(2.0 * kf, dh)));
            }
            r.set("mskew", b1);
            r.set("chi2_mskew", z1);
            r.set("df_mskew", df1);
            r.set("p_mskew", dist::chi2tail(df1, z1));
            r.set("mkurt", b2);
            r.set("chi2_mkurt", z2 * z2);
            r.set("p_mkurt", dist::chi2tail(1.0, z2 * z2));
            r.set("hz", hz);
            r.set("chi2_hz", zhz * zhz);
            r.set("p_hz", dist::chi2tail(1.0, zhz * zhz));
            r.set("chi2_dh", dh);
            r.set("df_dh", 2.0 * kf);
            r.set("p_dh", dist::chi2tail(2.0 * kf, dh));
        } else {
            return Err(rc_err(198, format!("unknown mvtest subcommand {sub}")));
        }
        st.r = r.clone();
        Ok(out.join("\n"))
    }
}

/// Doornik-Hansen transformed skewness and kurtosis (z1, z2) of a standardized series.
fn dh_z(z: &[f64]) -> (f64, f64) {
    let n = z.len() as f64;
    let m = z.iter().sum::<f64>() / n;
    let m2 = z.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = z.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    let m4 = z.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    let rb1 = m3 / m2.powf(1.5);
    let b2 = m4 / (m2 * m2);
    let b1 = rb1 * rb1;
    let beta = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta - 1.0)).sqrt();
    let delta = 1.0 / (w2.sqrt().ln()).sqrt();
    let y = rb1 * ((w2 - 1.0) / 2.0 * (n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let z1 = delta * (y + (1.0 + y * y).sqrt()).ln();
    let dd = (n - 3.0) * (n + 1.0) * (n * n + 15.0 * n - 4.0);
    let a = (n - 2.0) * (n + 5.0) * (n + 7.0) * (n * n + 27.0 * n - 70.0) / (6.0 * dd);
    let c = (n - 7.0) * (n + 5.0) * (n + 7.0) * (n * n + 2.0 * n - 5.0) / (6.0 * dd);
    let f = (n + 5.0) * (n + 7.0) * (n.powi(3) + 37.0 * n * n + 11.0 * n - 313.0) / (12.0 * dd);
    let alpha = a + b1 * c;
    let chi = 2.0 * f * (b2 - 1.0 - b1);
    let z2 = ((chi / (2.0 * alpha)).cbrt() - 1.0 + 1.0 / (9.0 * alpha)) * (9.0 * alpha).sqrt();
    (z1, z2)
}

// ---------------------------------------------------------------------------
// procrustes
// ---------------------------------------------------------------------------

pub struct Procrustes;

impl CommandHandler for Procrustes {
    fn name(&self) -> &'static str {
        "procrustes"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let g = paren_groups(&cmd.main);
        if g.len() != 2 {
            return Err(rc_err(198, "procrustes (varlist_y) (varlist_x)"));
        }
        let yv = crate::varlist::expand(ds, &g[0])?;
        let xv = crate::varlist::expand(ds, &g[1])?;
        let mut all = yv.clone();
        all.extend(xv.iter().cloned());
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &all);
        let rows = util::rows(&mask);
        let n = rows.len();
        let (ky, kx) = (yv.len(), xv.len());
        let trans = cmd.opt("transform", 5).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "orthogonal".into());
        if trans.starts_with('o') && trans != "oblique" && ky != kx {
            return Err(rc_err(198, "orthogonal transformation requires the same number of variables"));
        }
        let ycols = cols_of(ds, &yv, &rows)?;
        let xcols = cols_of(ds, &xv, &rows)?;
        let my: Vec<f64> = ycols.iter().map(|c| util::mean(c)).collect();
        let mx: Vec<f64> = xcols.iter().map(|c| util::mean(c)).collect();
        let mut yc = Matrix::zeros(n, ky);
        let mut xc = Matrix::zeros(n, kx);
        for t in 0..n {
            for j in 0..ky {
                yc[(t, j)] = ycols[j][t] - my[j];
            }
            for j in 0..kx {
                xc[(t, j)] = xcols[j][t] - mx[j];
            }
        }
        let xty = xc.transpose().mul(&yc).unwrap();
        let xtx = xc.transpose().mul(&xc).unwrap();
        let norho = cmd.has_opt("norho", 4);
        let (a, rho, title, dfa) = if trans.starts_with("un") {
            (mvutil::sym_inv(&xtx).mul(&xty).unwrap(), 1.0, "unrestricted", (kx * ky) as f64)
        } else if trans.starts_with("obl") {
            // oblique: columns of A with unit length, solved column by column
            let mut a = Matrix::zeros(kx, ky);
            for j in 0..ky {
                // minimize ||y_j - rho X a||, |a|=1: a ~ (X'X - mu I)^-1 X'y; use the unconstrained direction normalized
                let col: Vec<f64> = (0..kx).map(|i| xty[(i, j)]).collect();
                let xi = mvutil::sym_inv(&xtx);
                let v: Vec<f64> = (0..kx).map(|i| (0..kx).map(|l| xi[(i, l)] * col[l]).sum()).collect();
                let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                for i in 0..kx {
                    a[(i, j)] = v[i] / nrm;
                }
            }
            let xa = xc.mul(&a).unwrap();
            let rho = if norho { 1.0 } else { xa.data.iter().zip(&yc.data).map(|(p, q)| p * q).sum::<f64>() / xa.data.iter().map(|p| p * p).sum::<f64>() };
            (a, rho, "oblique", (kx * ky) as f64 - ky as f64 + 1.0)
        } else {
            // orthogonal: SVD of X'Y = U S V', A = U V'
            let (u, _) = mvutil::polar(&xty);
            let a = u;
            let s_tr = xty.transpose().mul(&a).unwrap().trace();
            let rho = if norho { 1.0 } else { s_tr / xtx.trace() };
            (a, rho, "orthogonal", (kx * (kx - 1) / 2) as f64 + 1.0)
        };
        let dfa = if norho && !trans.starts_with("un") { dfa - 1.0 } else { dfa };
        let fit = xc.mul(&a).unwrap().scale(rho);
        // translation c = my - rho mx A
        let c: Vec<f64> = (0..ky).map(|j| my[j] - rho * (0..kx).map(|i| mx[i] * a[(i, j)]).sum::<f64>()).collect();
        let df_m = dfa + ky as f64;
        let df_r = (n * ky) as f64 - df_m;
        let mut ss = vec![0.0; ky];
        let mut rss = vec![0.0; ky];
        let mut corr = vec![0.0; ky];
        for j in 0..ky {
            for t in 0..n {
                ss[j] += yc[(t, j)].powi(2);
                rss[j] += (yc[(t, j)] - fit[(t, j)]).powi(2);
            }
            let yy: Vec<f64> = (0..n).map(|t| yc[(t, j)]).collect();
            let ff: Vec<f64> = (0..n).map(|t| fit[(t, j)]).collect();
            let (m1, m2) = (util::mean(&yy), util::mean(&ff));
            let cv: f64 = yy.iter().zip(&ff).map(|(p, q)| (p - m1) * (q - m2)).sum();
            corr[j] = cv / (yy.iter().map(|p| (p - m1).powi(2)).sum::<f64>() * ff.iter().map(|q| (q - m2).powi(2)).sum::<f64>()).sqrt();
        }
        let sst: f64 = ss.iter().sum();
        let rsst: f64 = rss.iter().sum();
        let rmse = (rsst / df_r).sqrt();
        let mut out = vec![String::new()];
        out.push(mvutil::hdr(&format!("Procrustes analysis ({title})"), "Number of observations", ""));
        out.pop();
        out.push(format!("{:<46}Number of observations  = {:>10}", format!("Procrustes analysis ({title})"), n));
        out.push(format!("{:<46}Model df (df_m)         = {:>10}", "", df_m));
        out.push(format!("{:<46}Residual df (df_r)      = {:>10}", "", df_r));
        out.push(format!("{:<46}SS(target)              = {:>10}", "", sig(sst, 7)));
        out.push(format!("{:<46}RSS(target)             = {:>10}", "", sig(rsst, 7)));
        out.push(format!("{:<46}RMSE = root(RSS/df_r)   = {:>10}", "", sig(rmse, 7)));
        out.push(format!("{:<46}Procrustes = RSS/SS     = {:>10.4}", "", rsst / sst));
        let cm = Matrix::row_vector(&c).with_names(vec!["_cons".into()], yv.clone());
        let am = a.clone().with_names(xv.clone(), yv.clone());
        out.push(String::new());
        out.push("Translation c".into());
        out.push(String::new());
        out.extend(mvutil::matlist(&cm, "", "%10.0g", 11, false, 4));
        out.push(String::new());
        out.push(format!("{} matrix A ({title})", if title == "orthogonal" { "Rotation and reflection" } else { "Transformation" }));
        out.push(String::new());
        out.extend(mvutil::matlist(&am, "", "%10.0g", 11, false, 4));
        if !trans.starts_with("un") {
            out.push(String::new());
            out.push("Dilation factor".into());
            out.push(String::new());
            out.push(format!("{:>16} = {:>10.4}", "rho", rho));
        }
        let fs = Matrix::from_rows(&[ss.clone(), rss.clone(), (0..ky).map(|j| (rss[j] / (df_r / ky as f64)).sqrt()).collect(), (0..ky).map(|j| rss[j] / ss[j]).collect(), corr.clone()]).with_names(vec!["SS".into(), "RSS".into(), "RMSE".into(), "Procrustes".into(), "Corr_y_yhat".into()], yv.clone());
        out.push(String::new());
        out.push("Fit statistics by target variable".into());
        out.push(String::new());
        out.extend(mvutil::matlist(&fs, "Statistics", "%10.0g", 11, false, 4));
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("rho", rho);
        e.set("P", rsst / sst);
        e.set("ss", sst);
        e.set("rss", rsst);
        e.set("rmse", rmse);
        e.set("df_m", df_m);
        e.set("df_r", df_r);
        e.set_str("cmd", "procrustes");
        e.set_str("transform", title);
        e.set_str("ylist", yv.join(" "));
        e.set_str("xlist", xv.join(" "));
        e.set_mat("c", cm);
        e.set_mat("A", am);
        e.set_mat("ystats", fs);
        e.sample = Some(mask);
        Ok(out.join("\n"))
    }
}
