//! Panel ARDL in error-correction form: mean group (MG), pooled mean group
//! (PMG) and dynamic fixed effects (DFE) estimators, following Pesaran, Shin
//! & Smith (1999, JASA), and Pesaran's (2004) cross-section dependence test.
//!
//! Model for panel i:  D.y_it = phi_i (y_i,t-1 - theta' x_it) + delta_i' D.x_it + mu_i + e_it
//! `xtpmg D.y D.x..., lr(L.y x...) [pmg|mg|dfe] [ec(name) replace]`.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

struct Panel {
    dy: Vec<f64>,
    ylag: Vec<f64>,
    /// long-run regressors (levels), column-major: lr[k][t]
    lr: Vec<Vec<f64>>,
    /// short-run regressors (differences)
    sr: Vec<Vec<f64>>,
    times: Vec<f64>,
    rows: Vec<usize>,
}

/// OLS returning coefficients and residual sum of squares.
fn ols(cols: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64, Vec<f64>) {
    let k = cols.len();
    let n = y.len();
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for t in 0..n {
        for a in 0..k {
            xty[a] += cols[a][t] * y[t];
            for b in 0..k {
                xtx[(a, b)] += cols[a][t] * cols[b][t];
            }
        }
    }
    let (inv, _) = xtx.sym_inverse(1e-14);
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv[(a, b)] * xty[b]).sum()).collect();
    let resid: Vec<f64> = (0..n).map(|t| y[t] - (0..k).map(|a| beta[a] * cols[a][t]).sum::<f64>()).collect();
    let rss = resid.iter().map(|e| e * e).sum();
    (beta, rss, resid)
}

impl Panel {
    fn t(&self) -> usize {
        self.dy.len()
    }

    /// Short-run regression given theta: D.y on [ec, D.x..., 1].
    fn short_run(&self, theta: &[f64]) -> (Vec<f64>, f64, Vec<f64>) {
        let ec: Vec<f64> = (0..self.t()).map(|t| self.ylag[t] - theta.iter().zip(&self.lr).map(|(th, x)| th * x[t]).sum::<f64>()).collect();
        let mut cols = vec![ec];
        cols.extend(self.sr.iter().cloned());
        cols.push(vec![1.0; self.t()]);
        ols(&cols, &self.dy)
    }
}

fn profile_ll(panels: &[Panel], theta: &[f64]) -> f64 {
    panels
        .iter()
        .map(|p| {
            let (_, rss, _) = p.short_run(theta);
            let t = p.t() as f64;
            let s2 = rss / t;
            -0.5 * t * ((2.0 * std::f64::consts::PI * s2).ln() + 1.0)
        })
        .sum()
}

fn num_hessian(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> (Vec<f64>, Matrix) {
    let k = x.len();
    let h: Vec<f64> = x.iter().map(|v| 1e-4 * (1.0 + v.abs())).collect();
    let mut g = vec![0.0; k];
    let mut hm = Matrix::zeros(k, k);
    let f0 = f(x);
    for i in 0..k {
        let mut xp = x.to_vec();
        xp[i] += h[i];
        let mut xm = x.to_vec();
        xm[i] -= h[i];
        let (fp, fm) = (f(&xp), f(&xm));
        g[i] = (fp - fm) / (2.0 * h[i]);
        hm[(i, i)] = (fp - 2.0 * f0 + fm) / (h[i] * h[i]);
        for j in 0..i {
            let mut a = x.to_vec();
            a[i] += h[i];
            a[j] += h[j];
            let mut b = x.to_vec();
            b[i] += h[i];
            b[j] -= h[j];
            let mut c = x.to_vec();
            c[i] -= h[i];
            c[j] += h[j];
            let mut d = x.to_vec();
            d[i] -= h[i];
            d[j] -= h[j];
            let v = (f(&a) - f(&b) - f(&c) + f(&d)) / (4.0 * h[i] * h[j]);
            hm[(i, j)] = v;
            hm[(j, i)] = v;
        }
    }
    (g, hm)
}

/// Pesaran CD statistic, its p-value and the mean absolute correlation.
pub fn cd_test(resid: &[(Vec<f64>, Vec<f64>)]) -> (f64, f64, f64) {
    // resid: (times, residuals) per panel
    let n = resid.len();
    let mut sum = 0.0;
    let mut sum_abs = 0.0;
    let mut pairs = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            let (ti, ei) = &resid[i];
            let (tj, ej) = &resid[j];
            let mut a = Vec::new();
            let mut b = Vec::new();
            for (k, t) in ti.iter().enumerate() {
                if let Some(l) = tj.iter().position(|x| x == t) {
                    a.push(ei[k]);
                    b.push(ej[l]);
                }
            }
            let tij = a.len() as f64;
            if tij < 3.0 {
                continue;
            }
            let (ma, mb) = (util::mean(&a), util::mean(&b));
            let sab: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
            let sbb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
            let rho = sab / (saa * sbb).sqrt();
            sum += tij.sqrt() * rho;
            sum_abs += rho.abs();
            pairs += 1.0;
        }
    }
    let nf = n as f64;
    let cd = (2.0 / (nf * (nf - 1.0))).sqrt() * sum;
    (cd, 2.0 * (1.0 - dist::normal(cd.abs())), sum_abs / pairs)
}

fn mean_se(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let m = util::mean(vals);
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

pub struct Xtpmg;

impl CommandHandler for Xtpmg {
    fn name(&self) -> &'static str {
        "xtpmg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (panel, time) = st.tsset.clone().filter(|(p, t)| p.is_some() && !t.is_empty()).map(|(p, t)| (p.unwrap(), t)).ok_or_else(|| rc_err(459, "panel and time variables not set; use xtset"))?;
        let words: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        if words.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let dep = words[0].clone();
        let sr_names: Vec<String> = words[1..].to_vec();
        let lr_text = cmd.opt("lr", 2).ok_or_else(|| rc_err(198, "option lr() required"))?;
        let lr_words: Vec<String> = lr_text.split_whitespace().map(String::from).collect();
        if lr_words.len() < 2 {
            return Err(rc_err(198, "lr() needs the lagged dependent variable and at least one regressor"));
        }
        let ylag_name = lr_words[0].clone();
        let lr_names: Vec<String> = lr_words[1..].to_vec();
        let model = if cmd.has_opt("mg", 2) { "mg" } else if cmd.has_opt("dfe", 3) { "dfe" } else { "pmg" };
        let mask = util::touse(ds, st, cmd)?;
        let env = Env::with_store(ds, st);
        let ev = |name: &str| -> Result<Vec<f64>, String> {
            let e = ferrostat_parser::parser::parse_expression(name).map_err(|e| rc_err(198, e.message))?;
            (0..ds.n_obs).map(|i| env.eval(&e, i)?.num()).collect()
        };
        let dy = ev(&dep)?;
        let yl = ev(&ylag_name)?;
        let lrv: Vec<Vec<f64>> = lr_names.iter().map(|n| ev(n)).collect::<Result<_, _>>()?;
        let srv: Vec<Vec<f64>> = sr_names.iter().map(|n| ev(n)).collect::<Result<_, _>>()?;
        let pv = util::col(ds, &panel)?;
        let tv = util::col(ds, &time)?;
        drop(env);
        // build panels
        let mut keys: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| pv[i]).collect();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let mut panels: Vec<Panel> = Vec::new();
        for &k in &keys {
            let rows: Vec<usize> = (0..ds.n_obs)
                .filter(|&i| {
                    mask[i] && pv[i] == k && !is_missing(dy[i]) && !is_missing(yl[i]) && lrv.iter().all(|c| !is_missing(c[i])) && srv.iter().all(|c| !is_missing(c[i]))
                })
                .collect();
            if rows.len() <= lr_names.len() + sr_names.len() + 2 {
                continue;
            }
            panels.push(Panel {
                dy: rows.iter().map(|&i| dy[i]).collect(),
                ylag: rows.iter().map(|&i| yl[i]).collect(),
                lr: lrv.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(),
                sr: srv.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect(),
                times: rows.iter().map(|&i| tv[i]).collect(),
                rows,
            });
        }
        let ng = panels.len();
        if ng < 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let nobs: usize = panels.iter().map(|p| p.t()).sum();
        let klr = lr_names.len();
        let ksr = sr_names.len();
        // per-panel unrestricted fits (MG)
        let mut mg_theta: Vec<Vec<f64>> = Vec::new();
        let mut mg_sr: Vec<Vec<f64>> = Vec::new();
        for p in &panels {
            let mut cols = vec![p.ylag.clone()];
            cols.extend(p.lr.iter().cloned());
            cols.extend(p.sr.iter().cloned());
            cols.push(vec![1.0; p.t()]);
            let (b, _, _) = ols(&cols, &p.dy);
            let phi = b[0];
            mg_theta.push((0..klr).map(|j| -b[1 + j] / phi).collect());
            let mut s = vec![phi];
            s.extend(b[1 + klr..].iter().cloned());
            mg_sr.push(s);
        }
        let mut names: Vec<String> = Vec::new();
        let mut est: Vec<f64> = Vec::new();
        let mut v: Matrix;
        let theta: Vec<f64>;
        let mut ll = f64::NAN;
        let mut iters: Vec<f64> = Vec::new();
        let ecn = |n: &str| format!("ec:{n}");
        let srn = |n: &str| {
            let short = if n.to_lowercase().starts_with("d.") { format!("D.{}", &n[2..]) } else { n.to_string() };
            format!("SR:{short}")
        };
        match model {
            "mg" => {
                theta = (0..klr).map(|j| util::mean(&mg_theta.iter().map(|t| t[j]).collect::<Vec<_>>())).collect();
                let mut ses = Vec::new();
                for j in 0..klr {
                    let (m, s) = mean_se(&mg_theta.iter().map(|t| t[j]).collect::<Vec<_>>());
                    names.push(ecn(&lr_names[j]));
                    est.push(m);
                    ses.push(s);
                }
                for j in 0..ksr + 2 {
                    let (m, s) = mean_se(&mg_sr.iter().map(|t| t[j]).collect::<Vec<_>>());
                    names.push(match j {
                        0 => "SR:ec".to_string(),
                        j if j == ksr + 1 => "SR:_cons".to_string(),
                        j => srn(&sr_names[j - 1]),
                    });
                    est.push(m);
                    ses.push(s);
                }
                v = Matrix::from_diag(&ses.iter().map(|s| s * s).collect::<Vec<_>>());
            }
            _ => {
                // PMG: maximise the concentrated likelihood in theta
                let f = |th: &[f64]| profile_ll(&panels, th);
                let mut th: Vec<f64> = (0..klr).map(|j| util::mean(&mg_theta.iter().map(|t| t[j]).collect::<Vec<_>>())).collect();
                // robust start: median of panel estimates
                for j in 0..klr {
                    let mut c: Vec<f64> = mg_theta.iter().map(|t| t[j]).collect();
                    c.sort_by(f64::total_cmp);
                    th[j] = c[c.len() / 2];
                }
                let mut cur = f(&th);
                iters.push(cur);
                for _ in 0..200 {
                    let (g, h) = num_hessian(&f, &th);
                    let neg = h.scale(-1.0);
                    let (inv, _) = neg.sym_inverse(1e-14);
                    let concave = neg.cholesky().is_ok();
                    let step: Vec<f64> = if concave {
                        (0..klr).map(|i| (0..klr).map(|j| inv[(i, j)] * g[j]).sum()).collect()
                    } else {
                        // gradient ascent scaled by the diagonal
                        (0..klr).map(|i| g[i] / h[(i, i)].abs().max(1e-8)).collect()
                    };
                    let mut lambda = 1.0;
                    let mut improved = false;
                    for _ in 0..50 {
                        let cand: Vec<f64> = th.iter().zip(&step).map(|(a, s)| a + lambda * s).collect();
                        let val = f(&cand);
                        if val.is_finite() && val >= cur - 1e-12 {
                            let change = (val - cur).abs();
                            th = cand;
                            cur = val;
                            improved = change > 1e-10;
                            break;
                        }
                        lambda /= 2.0;
                    }
                    iters.push(cur);
                    if !improved {
                        break;
                    }
                }
                ll = cur;
                theta = th.clone();
                // information for theta: sum_i phi_i^2/s_i^2 X_i' M_i X_i (Pesaran, Shin & Smith 1999)
                let mut info = Matrix::zeros(klr, klr);
                for p in &panels {
                    let (b, rss, _) = p.short_run(&theta);
                    let phi = b[0];
                    let s2 = rss / p.t() as f64;
                    // the theta block of the inverse information partials out the
                    // other mean parameters of the panel: ec term, differences, constant
                    let ec: Vec<f64> = (0..p.t()).map(|t| p.ylag[t] - theta.iter().zip(&p.lr).map(|(th, x)| th * x[t]).sum::<f64>()).collect();
                    let mut w: Vec<Vec<f64>> = vec![ec];
                    w.extend(p.sr.iter().cloned());
                    w.push(vec![1.0; p.t()]);
                    // residualise each long-run regressor on W
                    let xr: Vec<Vec<f64>> = p.lr.iter().map(|x| ols(&w, x).2).collect();
                    for a in 0..klr {
                        for c in 0..klr {
                            let s: f64 = (0..p.t()).map(|t| xr[a][t] * xr[c][t]).sum();
                            info[(a, c)] += phi * phi / s2 * s;
                        }
                    }
                }
                let (vth, _) = info.sym_inverse(1e-14);
                let mut ses: Vec<f64> = (0..klr).map(|j| vth[(j, j)].max(0.0).sqrt()).collect();
                for j in 0..klr {
                    names.push(ecn(&lr_names[j]));
                    est.push(theta[j]);
                }
                // short-run: group means of the panel coefficients given theta
                let srs: Vec<Vec<f64>> = panels.iter().map(|p| p.short_run(&theta).0).collect();
                for j in 0..ksr + 2 {
                    let (m, s) = mean_se(&srs.iter().map(|t| t[j]).collect::<Vec<_>>());
                    names.push(match j {
                        0 => "SR:ec".to_string(),
                        j if j == ksr + 1 => "SR:_cons".to_string(),
                        j => srn(&sr_names[j - 1]),
                    });
                    est.push(m);
                    ses.push(s);
                }
                v = Matrix::from_diag(&ses.iter().map(|s| s * s).collect::<Vec<_>>());
                for a in 0..klr {
                    for b in 0..klr {
                        v[(a, b)] = vth[(a, b)];
                    }
                }
            }
        }
        // residuals for the CD test and the ec variable
        let resid: Vec<(Vec<f64>, Vec<f64>)> = panels
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let th = if model == "mg" { &mg_theta[i] } else { &theta };
                (p.times.clone(), p.short_run(th).2)
            })
            .collect();
        let (cd, cdp, mabs) = cd_test(&resid);
        if let Some(ecname) = cmd.opt("ec", 2).map(|s| s.trim().to_string()) {
            if ds.var_index.contains_key(&ecname) {
                if !cmd.has_opt("replace", 3) {
                    return Err(rc_err(110, format!("variable {ecname} already defined")));
                }
                ds.drop_variable(&ecname)?;
            }
            let mut vals = vec![ferrostat_core::SYSMISS; ds.n_obs];
            for (i, p) in panels.iter().enumerate() {
                let th = if model == "mg" { &mg_theta[i] } else { &theta };
                for (t, &r) in p.rows.iter().enumerate() {
                    vals[r] = p.ylag[t] - th.iter().zip(&p.lr).map(|(a, x)| a * x[t]).sum::<f64>();
                }
            }
            ds.add_variable(util::new_double(&ecname, vals))?;
        }
        // output
        let level = util::level(cmd, st)?;
        let lags: Vec<String> = std::iter::repeat("1").take(1 + klr).map(String::from).collect();
        let tcount: Vec<f64> = panels.iter().map(|p| p.t() as f64).collect();
        let mut out = Vec::new();
        if model == "pmg" {
            out.extend(crate::est::iteration_log(&iters, "Log likelihood"));
            out.push(String::new());
        }
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Panel Variable (i): {panel}"), "Number of obs", nobs));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Time Variable (t): {time}"), "Number of groups", ng));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Obs per group: min", fmt_g(tcount.iter().cloned().fold(f64::INFINITY, f64::min), 9)));
        out.push(format!("{:<52}{:<15} = {:>10.1}", "", "avg", nobs as f64 / ng as f64));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "max", fmt_g(tcount.iter().cloned().fold(0.0, f64::max), 9)));
        if model == "pmg" {
            out.push(format!("{:<52}{:<15} = {:>10.3}", "", "Log Likelihood", ll));
        }
        out.push(format!("Model order = ARDL({})", lags.join(",")));
        out.push(String::new());
        let k = names.len();
        let depdisp = if dep.to_lowercase().starts_with("d.") { format!("D.{}", &dep[2..]) } else { dep.clone() };
        out.extend(crate::est::coef_table(&crate::est::TableSpec {
            depvar: &depdisp,
            names: &names,
            b: &est,
            v: &v,
            omitted: &vec![false; k],
            is_base: &vec![false; k],
            df: None,
            level,
            eform: None,
            coef_title: "Coefficient",
        }));
        out.push(String::new());
        out.push("  Pesaran (2004, 2015) CD test   H0: cross-sectional independence".into());
        out.push(format!("  CD statistic = {:>9.4}  {}", cd, if cdp < 0.01 { "***" } else if cdp < 0.05 { "**" } else if cdp < 0.1 { "*" } else { "" }));
        out.push(format!("  p-value      = {:>9.4}", cdp));
        out.push(format!("  Mean |pairwise correlation| = {:>7.4}", mabs));
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("N", nobs as f64);
        e.set("n_g", ng as f64);
        e.set("N_g", ng as f64);
        if model == "pmg" {
            e.set("ll", ll);
        }
        e.set("cd", cd);
        e.set("cd_p", cdp);
        e.set_str("cmd", "xtpmg");
        e.set_str("model", model);
        e.set_str("depvar", depdisp.clone());
        e.set_str("ivar", panel.clone());
        e.set_str("tvar", time.clone());
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&est).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        let mut sample = vec![false; ds.n_obs];
        for p in &panels {
            for &r in &p.rows {
                sample[r] = true;
            }
        }
        e.sample = Some(sample);
        Ok(text)
    }
}

/// `hausman consistent efficient [, sigmamore sigmaless constant]`.
pub struct Hausman;

impl CommandHandler for Hausman {
    fn name(&self) -> &'static str {
        "hausman"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let w: Vec<&str> = cmd.main.split_whitespace().collect();
        if w.len() != 2 {
            return Err(rc_err(198, "hausman name-consistent name-efficient"));
        }
        let get = |n: &str| -> Result<Results, String> {
            if n == "." {
                Ok(st.e.clone())
            } else {
                st.estimates.get(n).cloned().ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))
            }
        };
        let (ec, ee) = (get(w[0])?, get(w[1])?);
        let (bc, vc) = (ec.matrices.get("b").ok_or_else(|| rc_err(301, "b not found"))?, ec.matrices.get("V").ok_or_else(|| rc_err(301, "V not found"))?);
        let (be, ve) = (ee.matrices.get("b").ok_or_else(|| rc_err(301, "b not found"))?, ee.matrices.get("V").ok_or_else(|| rc_err(301, "V not found"))?);
        let with_cons = cmd.has_opt("constant", 4);
        // common coefficients of the first equation
        let first_eq = |m: &Matrix| m.colnames.first().and_then(|c| c.split_once(':').map(|x| x.0.to_string()));
        let eqc = first_eq(bc);
        let mut common: Vec<(usize, usize, String)> = Vec::new();
        for (i, c) in bc.colnames.iter().enumerate() {
            if c.split_once(':').map(|x| x.0.to_string()) != eqc {
                continue;
            }
            let bare = c.rsplit(':').next().unwrap();
            if (bare == "_cons" && !with_cons) || c.contains("b.") || c.starts_with("o.") {
                continue;
            }
            if let Some(j) = be.colnames.iter().position(|x| x == c) {
                if vc.get(i, i) > 0.0 && ve.get(j, j) > 0.0 {
                    common.push((i, j, bare.to_string()));
                }
            }
        }
        if common.is_empty() {
            return Err(rc_err(498, "no coefficients in common"));
        }
        let k = common.len();
        let mut s2_scale = (1.0, 1.0);
        if cmd.has_opt("sigmamore", 6) || cmd.has_opt("sigmaless", 6) {
            let sig = |e: &Results| e.get("rmse").or_else(|| e.get("sigma_e")).or_else(|| e.get("sigma"));
            if let (Some(sc), Some(se)) = (sig(&ec), sig(&ee)) {
                if cmd.has_opt("sigmamore", 6) {
                    s2_scale = ((se / sc).powi(2), 1.0);
                } else {
                    s2_scale = (1.0, (sc / se).powi(2));
                }
            }
        }
        let d: Vec<f64> = common.iter().map(|(i, j, _)| bc.get(0, *i) - be.get(0, *j)).collect();
        let mut vd = Matrix::zeros(k, k);
        for (a, (ia, ja, _)) in common.iter().enumerate() {
            for (c, (ic, jc, _)) in common.iter().enumerate() {
                vd[(a, c)] = vc.get(*ia, *ic) * s2_scale.0 - ve.get(*ja, *jc) * s2_scale.1;
            }
        }
        let (inv, dropped) = vd.sym_inverse(1e-12);
        // generalised inverse via eigen-decomposition when not positive definite
        let chi2 = if dropped.is_empty() && vd.cholesky().is_ok() {
            (0..k).map(|a| (0..k).map(|c| d[a] * inv[(a, c)] * d[c]).sum::<f64>()).sum::<f64>()
        } else {
            let (vals, vecs) = vd.sym_eigen();
            let mut s = 0.0;
            for (m, &l) in vals.iter().enumerate() {
                if l.abs() > 1e-12 * vals[0].abs().max(1e-300) {
                    let proj: f64 = (0..k).map(|a| vecs[(a, m)] * d[a]).sum();
                    s += proj * proj / l;
                }
            }
            s
        };
        let df = (k - dropped.len()) as f64;
        let mut out = vec![
            format!("{:>30}---- Coefficients ----", ""),
            format!("{:>12} |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))", ""),
            format!("{:>12} | {:>12} {:>12} {:>12} {:>14}", "", crate::summarize::abbrev(w[0], 12), crate::summarize::abbrev(w[1], 12), "Difference", "Std. err."),
            "-------------+----------------------------------------------------------------".to_string(),
        ];
        for (a, (_, _, n)) in common.iter().enumerate() {
            let bvc = bc.get(0, common[a].0);
            let bve = be.get(0, common[a].1);
            let sd = if vd[(a, a)] > 0.0 { fmt_g(vd[(a, a)].sqrt(), 8) } else { ".".into() };
            out.push(format!("{:>12} | {:>12} {:>12} {:>12} {:>14}", crate::summarize::abbrev(n, 12), fmt_g(bvc, 8), fmt_g(bve, 8), fmt_g(d[a], 8), sd));
        }
        out.push("------------------------------------------------------------------------------".to_string());
        out.push(format!("{:>30}b = Consistent under H0 and Ha; obtained from {}.", "", w[0]));
        out.push(format!("{:>30}B = Inconsistent under Ha, efficient under H0; obtained from {}.", "", w[1]));
        out.push(String::new());
        out.push("Test of H0: Difference in coefficients not systematic".to_string());
        out.push(String::new());
        let mut r = Results::new();
        r.set("df", df);
        if chi2 >= 0.0 {
            let p = dist::chi2tail(df, chi2);
            out.push(format!("    chi2({}) = (b-B)'[(V_b-V_B)^(-1)](b-B)", df));
            out.push(format!("            = {:>8.2}", chi2));
            out.push(format!("Prob > chi2 = {:>8.4}", p));
            r.set("chi2", chi2);
            r.set("p", p);
        } else {
            out.push(format!("    chi2({}) = (b-B)'[(V_b-V_B)^(-1)](b-B)", df));
            out.push(format!("            = {:>8.2}    chi2<0 ==> model fitted on these", chi2));
            out.push("    data fails to meet the asymptotic assumptions of the Hausman test".to_string());
            r.set("chi2", chi2);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}
