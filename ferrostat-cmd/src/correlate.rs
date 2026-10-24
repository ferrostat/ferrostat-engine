//! correlate, pwcorr, spearman.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util::{self, WeightKind};
use crate::varlist;

fn lower_triangle(names: &[String], m: &Matrix, fmtf: &dyn Fn(usize, usize) -> String) -> Vec<String> {
    let k = names.len();
    let mut out = Vec::new();
    for start in (0..k).step_by(8) {
        let end = (start + 8).min(k);
        out.push(format!("{:>12} |{}", "", names[start..end].iter().map(|n| format!(" {:>8}", abbrev(n, 8))).collect::<String>()));
        out.push(format!("-------------+{}", "-".repeat((end - start) * 9)));
        for i in start..k {
            let cells: String = (start..end.min(i + 1)).map(|j| format!(" {:>8}", fmtf(i, j))).collect();
            out.push(format!("{:>12} |{}", abbrev(&names[i], 12), cells));
        }
        let _ = m;
        out.push(String::new());
    }
    out
}

/// Weighted covariance matrix of complete cases.
fn cov_matrix(cols: &[Vec<f64>], w: &[f64], kind: Option<WeightKind>) -> Matrix {
    let k = cols.len();
    let n = w.len() as f64;
    let sw: f64 = w.iter().sum();
    let means: Vec<f64> = cols.iter().map(|c| c.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / sw).collect();
    let denom = match kind {
        Some(WeightKind::F) => sw - 1.0,
        Some(_) => sw * (n - 1.0) / n,
        None => n - 1.0,
    };
    let mut m = Matrix::zeros(k, k);
    for a in 0..k {
        for b in 0..=a {
            let s: f64 = (0..w.len()).map(|t| w[t] * (cols[a][t] - means[a]) * (cols[b][t] - means[b])).sum();
            m[(a, b)] = s / denom;
            m[(b, a)] = s / denom;
        }
    }
    m
}

pub struct Correlate;

impl CommandHandler for Correlate {
    fn name(&self) -> &'static str {
        "correlate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars: Vec<String> = if cmd.main.trim().is_empty() {
            ds.vars.iter().filter(|v| v.var_type.is_numeric()).map(|v| v.name.clone()).collect()
        } else {
            varlist::expand(ds, &cmd.main)?
        };
        if vars.is_empty() {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| rows.iter().map(|&i| ds.get_var(v).unwrap().num(i)).collect()).collect();
        let ww: Vec<f64> = rows.iter().map(|&i| w.as_ref().map_or(1.0, |w| w.w[i])).collect();
        let n = match w.as_ref().map(|w| w.kind) {
            Some(WeightKind::F) => ww.iter().sum(),
            _ => rows.len() as f64,
        };
        let cov = cov_matrix(&cols, &ww, w.as_ref().map(|w| w.kind));
        let k = vars.len();
        let mut corr = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                corr[(a, b)] = cov[(a, b)] / (cov[(a, a)] * cov[(b, b)]).sqrt();
            }
        }
        let covariance = cmd.has_opt("covariance", 1);
        let shown = if covariance { cov.clone() } else { corr.clone() };
        let mut out = vec![format!("(obs={})", crate::format::fmt_g(n, 12)), String::new()];
        let fmtf = |i: usize, j: usize| {
            if covariance { crate::format::fmt_g(shown[(i, j)], 8) } else { format!("{:.4}", shown[(i, j)]) }
        };
        out.extend(lower_triangle(&vars, &shown, &fmtf));
        let mut r = Results::new();
        r.set("N", n);
        if k == 2 {
            r.set("rho", corr[(0, 1)]);
            r.set("cov_12", cov[(0, 1)]);
            r.set("Var_1", cov[(0, 0)]);
            r.set("Var_2", cov[(1, 1)]);
        }
        r.set_mat("C", shown.with_names(vars.clone(), vars.clone()));
        st.r = r;
        Ok(out.join("\n").trim_end().to_string())
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

pub struct Pwcorr;

impl CommandHandler for Pwcorr {
    fn name(&self) -> &'static str {
        "pwcorr"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mask = util::touse(ds, st, cmd)?;
        let sig = cmd.has_opt("sig", 3);
        let obs = cmd.has_opt("obs", 3);
        let star = util::opt_num(cmd, "star", 2)?;
        let k = vars.len();
        let mut rho = Matrix::zeros(k, k);
        let mut pm = Matrix::zeros(k, k);
        let mut nm = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..=a {
                let ca = util::col(ds, &vars[a])?;
                let cb = util::col(ds, &vars[b])?;
                let (x, y): (Vec<f64>, Vec<f64>) = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(ca[i]) && !is_missing(cb[i])).map(|i| (ca[i], cb[i])).unzip();
                let r = pearson(&x, &y);
                let n = x.len() as f64;
                let t = r * ((n - 2.0) / (1.0 - r * r)).sqrt();
                let p = if a == b { 0.0 } else { 2.0 * dist::ttail(n - 2.0, t.abs()) };
                for (i, j) in [(a, b), (b, a)] {
                    rho[(i, j)] = r;
                    pm[(i, j)] = p;
                    nm[(i, j)] = n;
                }
            }
        }
        let mut out = Vec::new();
        out.push(format!("{:>12} |{}", "", vars.iter().map(|n| format!(" {:>8}", abbrev(n, 8))).collect::<String>()));
        out.push(format!("-------------+{}", "-".repeat(k * 9)));
        for i in 0..k {
            let cells: String = (0..=i)
                .map(|j| {
                    let s = format!("{:.4}", rho[(i, j)]);
                    let s = if star.map_or(false, |lv| i != j && pm[(i, j)] <= lv) { format!("{s}*") } else { s };
                    format!(" {:>8}", s)
                })
                .collect();
            out.push(format!("{:>12} |{}", abbrev(&vars[i], 12), cells));
            if sig {
                out.push(format!("{:>12} |{}", "", (0..=i).map(|j| if i == j { format!(" {:>8}", "") } else { format!(" {:>8.4}", pm[(i, j)]) }).collect::<String>()));
            }
            if obs {
                out.push(format!("{:>12} |{}", "", (0..=i).map(|j| format!(" {:>8}", nm[(i, j)])).collect::<String>()));
            }
            if sig || obs {
                out.push(format!("{:>12} |", ""));
            }
        }
        let mut r = Results::new();
        r.set_mat("C", rho.clone().with_names(vars.clone(), vars.clone()));
        r.set_mat("sig", pm);
        r.set_mat("Nobs", nm.clone());
        if k == 2 {
            r.set("rho", rho[(0, 1)]);
            r.set("N", nm[(0, 1)]);
        } else {
            r.set("N", nm[(k - 1, k - 1)]);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

/// Average ranks (ties share the mean rank).
pub fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        for k in i..=j {
            r[idx[k]] = (i + j) as f64 / 2.0 + 1.0;
        }
        i = j + 1;
    }
    r
}

pub struct Spearman;

impl CommandHandler for Spearman {
    fn name(&self) -> &'static str {
        "spearman"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let ranked: Vec<Vec<f64>> = vars.iter().map(|v| ranks(&rows.iter().map(|&i| ds.get_var(v).unwrap().num(i)).collect::<Vec<_>>())).collect();
        let n = rows.len() as f64;
        let mut r = Results::new();
        r.set("N", n);
        if vars.len() == 2 {
            let rho = pearson(&ranked[0], &ranked[1]);
            let t = rho * ((n - 2.0) / (1.0 - rho * rho)).sqrt();
            let p = 2.0 * dist::ttail(n - 2.0, t.abs());
            r.set("rho", rho);
            r.set("p", p);
            st.r = r;
            return Ok(format!(
                " Number of obs = {:>8}\nSpearman's rho = {:>12.4}\n\nTest of Ho: {} and {} are independent\n    Prob > |t| = {:>12.4}",
                n, rho, vars[0], vars[1], p
            ));
        }
        let k = vars.len();
        let mut m = Matrix::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                m[(a, b)] = pearson(&ranked[a], &ranked[b]);
            }
        }
        let mut out = vec![format!("(obs={n})"), String::new()];
        out.extend(lower_triangle(&vars, &m, &|i, j| format!("{:.4}", m[(i, j)])));
        r.set_mat("Rho", m.with_names(vars.clone(), vars.clone()));
        st.r = r;
        Ok(out.join("\n"))
    }
}
