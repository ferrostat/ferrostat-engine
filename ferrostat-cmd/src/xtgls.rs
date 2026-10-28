//! `xtgls` (feasible GLS for panels with heteroskedastic, cross-sectionally
//! correlated and AR(1) disturbances) and `xtpcse` (OLS / Prais-Winsten
//! with panel-corrected standard errors).

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xtnl::{hl, hv, prepare, time_var};

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("xtgls", Box::new(Xtgls)), ("xtpcse", Box::new(Xtpcse))]
}

#[derive(Clone, Copy, PartialEq)]
enum Corr {
    Indep,
    Ar1,
    Psar1,
}

fn parse_corr(cmd: &Command) -> Result<Corr, String> {
    match cmd.opt("corr", 1).or_else(|| cmd.opt("correlation", 1)).map(|s| s.trim().to_lowercase()) {
        None => Ok(Corr::Indep),
        Some(s) if !s.is_empty() && "independent".starts_with(s.as_str()) => Ok(Corr::Indep),
        Some(s) if s == "ar1" => Ok(Corr::Ar1),
        Some(s) if s == "psar1" => Ok(Corr::Psar1),
        Some(s) => Err(rc_err(198, format!("corr({s}) invalid"))),
    }
}

/// Panel structure ordered by time: rows of each panel and their times.
struct Panels {
    rows: Vec<Vec<usize>>,
    times: Vec<Vec<f64>>,
    all_times: Vec<f64>,
}

fn panels(ds: &DataSet, d: &Design, groups: &[Vec<usize>], tvar: Option<&str>) -> Result<Panels, String> {
    let tc = match tvar {
        Some(t) => Some(util::col(ds, t)?),
        None => None,
    };
    let mut rows = Vec::new();
    let mut times = Vec::new();
    let mut all = Vec::new();
    for g in groups {
        let mut r = g.clone();
        let tm: Vec<f64> = match &tc {
            Some(c) => {
                r.sort_by(|a, b| c[d.rows[*a]].total_cmp(&c[d.rows[*b]]));
                r.iter().map(|&t| c[d.rows[t]]).collect()
            }
            None => (0..r.len()).map(|i| i as f64).collect(),
        };
        all.extend(tm.iter().cloned());
        rows.push(r);
        times.push(tm);
    }
    all.sort_by(f64::total_cmp);
    all.dedup();
    Ok(Panels { rows, times, all_times: all })
}

/// Residuals y - Xb on the active columns.
fn resid(x: &[Vec<f64>], y: &[f64], b: &[f64]) -> Vec<f64> {
    (0..y.len()).map(|t| y[t] - x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>()).collect()
}

fn ols_b(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> (Vec<f64>, Matrix) {
    let k = x.len();
    let mut a = Matrix::zeros(k, k);
    let mut r = vec![0.0; k];
    for t in 0..y.len() {
        for i in 0..k {
            let xi = x[i][t] * w[t];
            r[i] += xi * y[t];
            for j in 0..=i {
                a[(i, j)] += xi * x[j][t];
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            a[(j, i)] = a[(i, j)];
        }
    }
    let (ai, _) = a.sym_inverse(1e-14);
    let b = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * r[j]).sum()).collect();
    (b, ai)
}

/// Panel-specific AR(1) coefficients of residuals (regression of e_t on e_t-1).
fn panel_rhos(p: &Panels, e: &[f64]) -> Vec<f64> {
    p.rows
        .iter()
        .map(|r| {
            let (mut num, mut den) = (0.0, 0.0);
            for s in 1..r.len() {
                num += e[r[s]] * e[r[s - 1]];
                den += e[r[s - 1]] * e[r[s - 1]];
            }
            if den > 0.0 { (num / den).clamp(-0.999999, 0.999999) } else { 0.0 }
        })
        .collect()
}

/// Prais-Winsten transform of the columns and y within panels.
fn prais(p: &Panels, x: &[Vec<f64>], y: &[f64], rho: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    for (gi, r) in p.rows.iter().enumerate() {
        let rh = rho[gi];
        let f = (1.0 - rh * rh).sqrt();
        for (s, &t) in r.iter().enumerate() {
            if s == 0 {
                ys[t] = f * y[t];
                for j in 0..x.len() {
                    xs[j][t] = f * x[j][t];
                }
            } else {
                let tp = r[s - 1];
                ys[t] = y[t] - rh * y[tp];
                for j in 0..x.len() {
                    xs[j][t] = x[j][t] - rh * x[j][tp];
                }
            }
        }
    }
    (xs, ys)
}

/// Cross-panel covariance of residuals over common periods.
fn cross_cov(p: &Panels, e: &[f64], diag_only: bool) -> Matrix {
    let m = p.rows.len();
    let mut s = Matrix::zeros(m, m);
    for i in 0..m {
        for j in 0..=i {
            if diag_only && i != j {
                continue;
            }
            let (mut sum, mut cnt) = (0.0, 0usize);
            let (mut a, mut b) = (0usize, 0usize);
            while a < p.rows[i].len() && b < p.rows[j].len() {
                let (ta, tb) = (p.times[i][a], p.times[j][b]);
                if ta == tb {
                    sum += e[p.rows[i][a]] * e[p.rows[j][b]];
                    cnt += 1;
                    a += 1;
                    b += 1;
                } else if ta < tb {
                    a += 1;
                } else {
                    b += 1;
                }
            }
            let v = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
            s[(i, j)] = v;
            s[(j, i)] = v;
        }
    }
    s
}

/// sum_t sum_ij W_ij a_it b_jt' over common periods (a, b: columns).
fn kron_cross(p: &Panels, w: &Matrix, a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let ka = a.len();
    let kb = b.len();
    let m = p.rows.len();
    let mut out = Matrix::zeros(ka, kb);
    // index: for each time, the row of each panel present
    for &tm in &p.all_times {
        let present: Vec<(usize, usize)> = (0..m).filter_map(|i| p.times[i].binary_search_by(|x| x.total_cmp(&tm)).ok().map(|s| (i, p.rows[i][s]))).collect();
        for &(i, ti) in &present {
            for &(j, tj) in &present {
                let wij = w[(i, j)];
                if wij == 0.0 {
                    continue;
                }
                for r in 0..ka {
                    let ar = a[r][ti] * wij;
                    if ar == 0.0 {
                        continue;
                    }
                    for c in 0..kb {
                        out[(r, c)] += ar * b[c][tj];
                    }
                }
            }
        }
    }
    out
}

fn balanced(p: &Panels) -> bool {
    p.rows.iter().all(|r| r.len() == p.all_times.len())
}

pub struct Xtgls;

impl CommandHandler for Xtgls {
    fn name(&self) -> &'static str {
        "xtgls"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "xtgls");
        }
        let pan = match cmd.opt("panels", 1).map(|s| s.trim().to_lowercase()) {
            None => "iid",
            Some(s) if s == "iid" => "iid",
            Some(s) if s.len() >= 3 && "heteroskedastic".starts_with(s.as_str()) => "het",
            Some(s) if s.len() >= 3 && "hetero".starts_with(s.as_str()) => "het",
            Some(s) if s.len() >= 1 && "correlated".starts_with(s.as_str()) => "corr",
            Some(s) => return Err(rc_err(198, format!("panels({s}) invalid"))),
        };
        let corr = parse_corr(cmd)?;
        let igls = cmd.has_opt("igls", 4);
        let nmk = cmd.has_opt("nmk", 3);
        let pr = prepare(ds, st, cmd, cmd.has_opt("noconstant", 3))?;
        let tvar = time_var(st).ok();
        if corr != Corr::Indep && tvar.is_none() {
            return Err(rc_err(459, "time variable not set; use xtset panelvar timevar"));
        }
        let d = &pr.d;
        let p = panels(ds, d, &pr.groups, tvar.as_deref())?;
        if pan == "corr" && !balanced(&p) && !cmd.has_opt("force", 5) {
            return Err(rc_err(459, "panels(correlated) requires balanced data"));
        }
        let act = d.active();
        let k = d.k();
        let x: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
        let n = d.n();
        let m = p.rows.len();
        let ones = vec![1.0; n];
        let (mut b, _) = ols_b(&x, &d.y, &ones);
        let mut rho = vec![0.0; m];
        let mut sigma = Matrix::identity(m);
        let mut vact = Matrix::zeros(act.len(), act.len());
        let mut ll = SYSMISS;
        let mut iters = 0;
        let ka = act.len();
        for it in 0..if igls { 1000 } else { 1 } {
            iters = it + 1;
            let e = resid(&x, &d.y, &b);
            if corr != Corr::Indep {
                let r = panel_rhos(&p, &e);
                rho = if corr == Corr::Ar1 { vec![r.iter().sum::<f64>() / m as f64; m] } else { r };
            }
            let (xs, ys) = prais(&p, &x, &d.y, &rho);
            let bs = if it == 0 { ols_b(&xs, &ys, &ones).0 } else { b.clone() };
            let es = resid(&xs, &ys, &bs);
            let div = if nmk { (n - ka) as f64 / n as f64 } else { 1.0 };
            sigma = match pan {
                "iid" => {
                    let s2 = es.iter().map(|v| v * v).sum::<f64>() / n as f64 / div;
                    let mut s = Matrix::zeros(m, m);
                    for i in 0..m {
                        s[(i, i)] = s2;
                    }
                    s
                }
                "het" => cross_cov(&p, &es, true).scale(1.0 / div),
                _ => cross_cov(&p, &es, false).scale(1.0 / div),
            };
            let (si, _) = sigma.sym_inverse(1e-14);
            let mut yy = vec![ys.clone()];
            let a = kron_cross(&p, &si, &xs, &xs);
            let r = kron_cross(&p, &si, &xs, &yy);
            let (ai, _) = a.sym_inverse(1e-14);
            let nb: Vec<f64> = (0..ka).map(|i| (0..ka).map(|j| ai[(i, j)] * r[(j, 0)]).sum()).collect();
            let change = nb.iter().zip(&b).map(|(a, c)| (a - c).abs() / (c.abs() + 1.0)).fold(0.0, f64::max);
            b = nb;
            vact = ai;
            // log likelihood of the transformed model
            let eg = resid(&xs, &ys, &b);
            yy[0] = eg.clone();
            let q = kron_cross(&p, &si, &yy, &yy)[(0, 0)];
            let ldet = sigma.det().ln();
            ll = -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + p.all_times.len() as f64 * ldet + q);
            if pan != "corr" {
                // panel-specific variances: each panel contributes T_i ln s_i^2
                let lsum: f64 = (0..m).map(|i| p.rows[i].len() as f64 * sigma[(i, i)].ln()).sum();
                ll = -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + lsum + q);
            }
            if change < 1e-9 && it > 0 {
                break;
            }
        }
        let mut bfull = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            bfull[j] = b[a];
        }
        let v = est::expand_v(&vact, &act, k);
        let (chi2, df_m) = est::wald_all(d, &bfull, &v);
        let n_cov = match pan {
            "iid" => 1,
            "het" => m,
            _ => m * (m + 1) / 2,
        };
        let n_ar = match corr {
            Corr::Indep => 0,
            Corr::Ar1 => 1,
            Corr::Psar1 => m,
        };
        let panels_txt = match pan {
            "iid" => "homoskedastic",
            "het" => "heteroskedastic",
            _ => "heteroskedastic with cross-sectional correlation",
        };
        let corr_txt = match corr {
            Corr::Indep => "no autocorrelation".to_string(),
            Corr::Ar1 => format!("common AR(1) coefficient for all panels  ({:.4})", rho[0]),
            Corr::Psar1 => "panel-specific AR(1)".to_string(),
        };
        let nt = p.all_times.len();
        let show_ll = corr == Corr::Indep;
        let mut out = vec![String::new()];
        if igls {
            out.push(format!("Iteration {iters}: tolerance = converged"));
            out.push(String::new());
        }
        out.push("Cross-sectional time-series FGLS regression".into());
        out.push(String::new());
        out.push("Coefficients:  generalized least squares".into());
        out.push(format!("Panels:        {panels_txt}"));
        out.push(format!("Correlation:   {corr_txt}"));
        out.push(String::new());
        let wlab = format!("Wald chi2({})", fmt_g(df_m, 9));
        let rows = vec![
            ("Estimated covariances", n_cov.to_string(), "Number of obs", util::commas(n)),
            ("Estimated autocorrelations", n_ar.to_string(), "Number of groups", util::commas(m)),
            ("Estimated coefficients", ka.to_string(), "Time periods", util::commas(nt)),
            ("", String::new(), wlab.as_str(), format!("{:.2}", chi2)),
            (if show_ll { "Log likelihood" } else { "" }, if show_ll { fmt_g(ll, 9) } else { String::new() }, "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))),
        ];
        let wr = rows.iter().map(|r| r.3.len()).max().unwrap_or(0);
        for (a, av, bl, bv) in &rows {
            let left = if a.is_empty() { String::new() } else { format!("{:<27}= {:>10}", a, av) };
            out.push(format!("{:<49}{:<16} = {:>wr$}", left, bl, bv));
        }
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &pr.dep, names: &d.names, b: &bfull, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level: pr.level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let ivar = pr.ivar.clone();
        est::post(st, "xtgls", d, &bfull, &v, &|e: &mut Results| {
            e.set("N_g", m as f64);
            e.set("N_t", nt as f64);
            e.set("n_cf", ka as f64);
            e.set("n_cv", n_cov as f64);
            e.set("n_cr", n_ar as f64);
            e.set("df_pear", ka as f64 + n_cov as f64 + n_ar as f64);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            if show_ll {
                e.set("ll", ll);
            }
            if corr == Corr::Ar1 {
                e.set("rho", rho[0]);
            }
            e.set_mat("Sigma", sigma.clone());
            e.set_str("ivar", ivar.clone());
            if let Some(t) = &tvar {
                e.set_str("tvar", t.clone());
            }
            e.set_str("coefftype", "gls");
            e.set_str("vt", match pan {
                "iid" => "homoskedastic",
                "het" => "heteroskedastic",
                _ => "correlated",
            });
            e.set_str("corr", match corr {
                Corr::Indep => "independent",
                Corr::Ar1 => "ar1",
                Corr::Psar1 => "psar1",
            });
            e.set_str("title", "Cross-sectional time-series FGLS regression");
            e.set_str("chi2type", "Wald");
            e.set_str("predict", "xtgls_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

fn replay(st: &Store, verb: &str) -> Result<String, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
        return Err(rc_err(301, "last estimates not found"));
    }
    est::stash_table_from_e(&st.e, st.level);
    Ok(st.e.macros.get("_output").cloned().unwrap_or_default())
}

pub struct Xtpcse;

impl CommandHandler for Xtpcse {
    fn name(&self) -> &'static str {
        "xtpcse"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "xtpcse");
        }
        let corr = parse_corr(cmd)?;
        let hetonly = cmd.has_opt("hetonly", 3);
        let indep = cmd.has_opt("independent", 3);
        let pr = prepare(ds, st, cmd, cmd.has_opt("noconstant", 3))?;
        let tvar = time_var(st)?;
        let d = &pr.d;
        let p = panels(ds, d, &pr.groups, Some(&tvar))?;
        let act = d.active();
        let k = d.k();
        let ka = act.len();
        let x: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
        let n = d.n();
        let m = p.rows.len();
        let ones = vec![1.0; n];
        let (b0, _) = ols_b(&x, &d.y, &ones);
        let mut rho = vec![0.0; m];
        if corr != Corr::Indep {
            let r = panel_rhos(&p, &resid(&x, &d.y, &b0));
            rho = if corr == Corr::Ar1 { vec![r.iter().sum::<f64>() / m as f64; m] } else { r };
        }
        let (xs, ys) = prais(&p, &x, &d.y, &rho);
        let (b, xtxi) = ols_b(&xs, &ys, &ones);
        let e = resid(&xs, &ys, &b);
        let sigma = if indep {
            let s2 = e.iter().map(|v| v * v).sum::<f64>() / n as f64;
            Matrix::identity(m).scale(s2)
        } else {
            cross_cov(&p, &e, hetonly)
        };
        let meat = kron_cross(&p, &sigma, &xs, &xs);
        let vact = xtxi.mul(&meat).unwrap().mul(&xtxi).unwrap();
        let mut bfull = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            bfull[j] = b[a];
        }
        let v = est::expand_v(&vact, &act, k);
        let (chi2, df_m) = est::wald_all(d, &bfull, &v);
        let rss: f64 = e.iter().map(|v| v * v).sum();
        let ym = ys.iter().sum::<f64>() / n as f64;
        let tss: f64 = if d.has_cons { ys.iter().map(|y| (y - ym).powi(2)).sum() } else { ys.iter().map(|y| y * y).sum() };
        let r2 = 1.0 - rss / tss;
        let (mn, av, mx) = (p.rows.iter().map(|r| r.len()).min().unwrap_or(0), n as f64 / m as f64, p.rows.iter().map(|r| r.len()).max().unwrap_or(0));
        let bal = balanced(&p);
        let pan_txt = if indep {
            "homoskedastic"
        } else if hetonly {
            "heteroskedastic"
        } else {
            "correlated"
        };
        let pan_txt = format!("{pan_txt} ({})", if bal { "balanced" } else { "unbalanced" });
        let corr_txt = match corr {
            Corr::Indep => "no autocorrelation".to_string(),
            Corr::Ar1 => "common AR(1)".to_string(),
            Corr::Psar1 => "panel-specific AR(1)".to_string(),
        };
        let n_cov = if indep {
            1
        } else if hetonly {
            m
        } else {
            m * (m + 1) / 2
        };
        let n_ar = match corr {
            Corr::Indep => 0,
            Corr::Ar1 => 1,
            Corr::Psar1 => m,
        };
        let title = match (indep, hetonly) {
            (true, _) => "Linear regression, independent panels corrected standard errors (PCSEs)",
            (_, true) => "Linear regression, heteroskedastic panels corrected standard errors",
            _ => "Linear regression, correlated panels corrected standard errors (PCSEs)",
        };
        let title = if corr == Corr::Indep { title.to_string() } else { title.replace("Linear regression", "Prais-Winsten regression") };
        let mut h = vec![hl(""), hl(&title), hl("")];
        h.push(hv(&format!("Group variable:   {}", pr.ivar), "Number of obs", util::commas(n)));
        h.push(hv(&format!("Time variable:    {tvar}"), "Number of groups", util::commas(m)));
        h.push(hv(&format!("Panels:           {pan_txt}"), "Obs per group:", String::new()));
        h.push(hv(&format!("Autocorrelation:  {corr_txt}"), &format!("{:>16}", "min"), util::commas(mn)));
        h.push(hv("", &format!("{:>16}", "avg"), fmt_g(av, 9)));
        h.push(hv("", &format!("{:>16}", "max"), util::commas(mx)));
        h.push(hv(&format!("{:<27}= {:>10}", "Estimated covariances", n_cov), "R-squared", format!("{:.4}", r2)));
        h.push(hv(&format!("{:<27}= {:>10}", "Estimated autocorrelations", n_ar), &format!("Wald chi2({})", fmt_g(df_m, 9)), format!("{:.2}", chi2)));
        h.push(hv(&format!("{:<27}= {:>10}", "Estimated coefficients", ka), "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))));
        let mut out = crate::xtnl::header_block(&h);
        out.push(String::new());
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &pr.dep, names: &d.names, b: &bfull, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level: pr.level, eform: None, coef_title: "Coefficient" });
        let lw = tbl.last().map_or(12, |l| l.len().saturating_sub(66));
        tbl.insert(1, format!("{:>lw$} |{:>26}", "", "Panel-corrected"));
        out.extend(tbl);
        if corr != Corr::Indep {
            out.push(format!("{:>lw$} | {:>10}", "rho", fmt_g(rho.iter().sum::<f64>() / m as f64, 9), lw = lw));
            out.push("-".repeat(lw + 66));
        }
        let text = out.join("\n");
        let ivar = pr.ivar.clone();
        est::post(st, "xtpcse", d, &bfull, &v, &|e: &mut Results| {
            e.set("N_g", m as f64);
            e.set("N_gaps", 0.0);
            e.set("g_min", mn as f64);
            e.set("g_avg", av);
            e.set("g_max", mx as f64);
            e.set("r2", r2);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("n_cf", ka as f64);
            e.set("n_cv", n_cov as f64);
            e.set("n_cr", n_ar as f64);
            if corr != Corr::Indep {
                e.set("rho", rho.iter().sum::<f64>() / m as f64);
            }
            e.set_mat("Sigma", sigma.clone());
            e.set_str("ivar", ivar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("panels", if indep { "independent" } else if hetonly { "heteroskedastic" } else { "correlated" });
            e.set_str("corr", match corr {
                Corr::Indep => "independent",
                Corr::Ar1 => "ar1",
                Corr::Psar1 => "psar1",
            });
            e.set_str("vcetype", "Panel-corrected");
            e.set_str("title", title.clone());
            e.set_str("chi2type", "Wald");
            e.set_str("predict", "xtpcse_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
