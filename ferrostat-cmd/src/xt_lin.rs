//! Linear panel estimators: `xtregar` (fixed and random effects with AR(1)
//! disturbances, Baltagi and Wu), `xthtaylor` (Hausman-Taylor) and `xtrc`
//! (Swamy random coefficients).

use std::collections::BTreeMap;

use ferrostat_core::missing::SYSMISS;
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

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    vec![("xtregar", Box::new(XtRegar)), ("xthtaylor", Box::new(XtHtaylor)), ("xtrc", Box::new(XtRc))]
}

/// Least squares on columns: (b, (X'X)^-1, residuals). Collinear columns get 0.
pub fn ls(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Matrix, Vec<f64>) {
    let k = x.len();
    let n = y.len();
    let mut a = Matrix::zeros(k, k);
    let mut u = vec![0.0; k];
    for i in 0..k {
        u[i] = (0..n).map(|t| x[i][t] * y[t]).sum();
        for j in 0..=i {
            let s: f64 = (0..n).map(|t| x[i][t] * x[j][t]).sum();
            a[(i, j)] = s;
            a[(j, i)] = s;
        }
    }
    let (ai, _) = a.sym_inverse(1e-12);
    let b: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * u[j]).sum()).collect();
    let e: Vec<f64> = (0..n).map(|t| y[t] - (0..k).map(|j| b[j] * x[j][t]).sum::<f64>()).collect();
    (b, ai, e)
}

/// Two-stage least squares: (b, (X'PzX)^-1, residuals).
fn tsls(x: &[Vec<f64>], z: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Matrix, Vec<f64>) {
    // fitted values of each x on z
    let xh: Vec<Vec<f64>> = x
        .iter()
        .map(|c| {
            let (_, _, e) = ls(z, c);
            c.iter().zip(&e).map(|(a, b)| a - b).collect()
        })
        .collect();
    let (b, ai, _) = ls(&xh, y);
    let n = y.len();
    let e: Vec<f64> = (0..n).map(|t| y[t] - (0..x.len()).map(|j| b[j] * x[j][t]).sum::<f64>()).collect();
    (b, ai, e)
}

struct Panel {
    d: Design,
    /// panels: estimation rows (indices into the design) sorted by time
    groups: Vec<Vec<usize>>,
    time: Vec<f64>,
    ivar: String,
    tvar: String,
    dep: String,
}

fn panel_design(ds: &DataSet, st: &Store, cmd: &Command, need_t: bool) -> Result<Panel, String> {
    let main = cmd.main.trim();
    let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
    let dep = crate::varlist::resolve_name(ds, dep)?;
    let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
    let tvar = st.tsset.as_ref().map(|(_, t)| t.clone()).unwrap_or_default();
    if need_t && tvar.is_empty() {
        return Err(rc_err(459, "time variable not set; use xtset panelvar timevar"));
    }
    let mut extra = vec![ivar.clone()];
    if !tvar.is_empty() {
        extra.push(tvar.clone());
    }
    let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: false, weights: "", extra })?;
    let tc = if tvar.is_empty() { vec![0.0; ds.n_obs] } else { util::col(ds, &tvar)? };
    let time: Vec<f64> = d.rows.iter().map(|&i| tc[i]).collect();
    let (_, mut groups) = crate::xtnl::panel_groups(ds, &ivar, &d.rows)?;
    for g in groups.iter_mut() {
        g.sort_by(|a, b| time[*a].total_cmp(&time[*b]));
    }
    Ok(Panel { d, groups, time, ivar, tvar, dep })
}

fn opg(groups: &[Vec<usize>]) -> (f64, f64, f64) {
    let mn = groups.iter().map(|g| g.len()).min().unwrap_or(0) as f64;
    let mx = groups.iter().map(|g| g.len()).max().unwrap_or(0) as f64;
    let n: usize = groups.iter().map(|g| g.len()).sum();
    (mn, n as f64 / groups.len().max(1) as f64, mx)
}

fn hrow(l: &str, k: &str, v: String) -> String {
    format!("{:<52}{:<15} = {:>10}", l, k, v)
}

// ---------------------------------------------------------------------------
// xtregar
// ---------------------------------------------------------------------------

pub struct XtRegar;

impl CommandHandler for XtRegar {
    fn name(&self) -> &'static str {
        "xtregar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return st.e.macros.get("_output").cloned().filter(|_| st.e.macros.get("cmd").map(String::as_str) == Some("xtregar")).ok_or_else(|| rc_err(301, "last estimates not found"));
        }
        let fe = cmd.has_opt("fe", 2);
        let p = panel_design(ds, st, cmd, true)?;
        let d = &p.d;
        let act: Vec<usize> = d.active().into_iter().filter(|&j| d.names[j] != "_cons").collect();
        let kx = act.len();
        let n = d.n();
        let ng = p.groups.len();
        let level = util::level(cmd, st)?;
        let y = &d.y;
        let xc: Vec<&Vec<f64>> = act.iter().map(|&j| &d.cols[j]).collect();
        // within regression on the original data
        let demean = |v: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0; n];
            for g in &p.groups {
                let m = g.iter().map(|&t| v[t]).sum::<f64>() / g.len() as f64;
                for &t in g {
                    o[t] = v[t] - m;
                }
            }
            o
        };
        let wy = demean(y);
        let wx: Vec<Vec<f64>> = xc.iter().map(|c| demean(c)).collect();
        let (bw, _, ew) = ls(&wx, &wy);
        let _ = bw;
        // rho
        let dw = |e: &[f64]| -> f64 {
            let (mut num, mut den) = (0.0, 0.0);
            for g in &p.groups {
                for (a, &t) in g.iter().enumerate() {
                    den += e[t] * e[t];
                    if a > 0 {
                        let s = g[a - 1];
                        if p.time[t] - p.time[s] == 1.0 {
                            num += (e[t] - e[s]).powi(2);
                        }
                    }
                }
            }
            num / den
        };
        let d_bhargava = dw(&ew);
        let rho = match util::opt_num(cmd, "rhof", 4)? {
            Some(r) => r,
            None => {
                let rt = cmd.opt("rhotype", 4).unwrap_or("dw").trim().to_lowercase();
                match rt.as_str() {
                    "dw" => 1.0 - d_bhargava / 2.0,
                    "regress" | "reg" => {
                        let (mut a, mut b) = (0.0, 0.0);
                        for g in &p.groups {
                            for w in g.windows(2) {
                                if p.time[w[1]] - p.time[w[0]] == 1.0 {
                                    a += ew[w[1]] * ew[w[0]];
                                    b += ew[w[0]] * ew[w[0]];
                                }
                            }
                        }
                        a / b
                    }
                    _ => {
                        // tscorr / freg / onestep fall back to the lag-1 autocorrelation
                        let (mut a, mut b) = (0.0, 0.0);
                        for g in &p.groups {
                            for w in g.windows(2) {
                                if p.time[w[1]] - p.time[w[0]] == 1.0 {
                                    a += ew[w[1]] * ew[w[0]];
                                }
                            }
                            for &t in g {
                                b += ew[t] * ew[t];
                            }
                        }
                        a / b
                    }
                }
            }
        };
        if rho.abs() >= 1.0 {
            return Err(rc_err(498, "estimated rho is outside (-1, 1)"));
        }
        // Baltagi-Wu transformation (gaps allowed)
        let r2 = (1.0 - rho * rho).sqrt();
        let trans = |v: &[f64], keep_first: bool| -> Vec<Option<f64>> {
            let mut o = vec![None; n];
            for g in &p.groups {
                for (a, &t) in g.iter().enumerate() {
                    if a == 0 {
                        if keep_first {
                            o[t] = Some(r2 * v[t]);
                        }
                    } else {
                        let s = g[a - 1];
                        let m = (p.time[t] - p.time[s]).round();
                        let rm = rho.powf(m);
                        let f = r2 / (1.0 - rho.powf(2.0 * m)).sqrt();
                        o[t] = Some(if rho == 0.0 { v[t] } else { f * (v[t] - rm * v[s]) });
                    }
                }
            }
            o
        };
        let ones = vec![1.0; n];
        let keep_first = !fe;
        let ty = trans(y, keep_first);
        let tx: Vec<Vec<Option<f64>>> = xc.iter().map(|c| trans(c, keep_first)).collect();
        let tc = trans(&ones, keep_first);
        // transformed panels
        let tg: Vec<Vec<usize>> = p.groups.iter().map(|g| g.iter().copied().filter(|&t| ty[t].is_some()).collect::<Vec<_>>()).filter(|g: &Vec<usize>| !g.is_empty()).collect();
        let nt: usize = tg.iter().map(|g| g.len()).sum();
        let get = |o: &Vec<Option<f64>>, t: usize| o[t].unwrap();
        let mut out = Vec::new();
        let (b, v, stats): (Vec<f64>, Matrix, Vec<(&str, f64)>);
        let (mn, av, mx) = opg(&tg);
        if fe {
            // within on the transformed data
            let flat: Vec<usize> = tg.iter().flatten().copied().collect();
            let pos: BTreeMap<usize, usize> = flat.iter().enumerate().map(|(a, &t)| (t, a)).collect();
            let dm = |o: &Vec<Option<f64>>| -> (Vec<f64>, f64) {
                let mut r = vec![0.0; nt];
                let all = flat.iter().map(|&t| get(o, t)).sum::<f64>() / nt as f64;
                for g in &tg {
                    let m = g.iter().map(|&t| get(o, t)).sum::<f64>() / g.len() as f64;
                    for &t in g {
                        r[pos[&t]] = get(o, t) - m;
                    }
                }
                (r, all)
            };
            let (yy, ybar) = dm(&ty);
            let xx: Vec<(Vec<f64>, f64)> = tx.iter().map(|o| dm(o)).collect();
            let cbar = flat.iter().map(|&t| get(&tc, t)).sum::<f64>() / nt as f64;
            let xcols: Vec<Vec<f64>> = xx.iter().map(|x| x.0.clone()).collect();
            let (bb, ai, e) = ls(&xcols, &yy);
            let df_r = nt as f64 - ng as f64 - kx as f64;
            let rss: f64 = e.iter().map(|v| v * v).sum();
            let s2 = rss / df_r;
            let cons_t = ybar - (0..kx).map(|j| bb[j] * xx[j].1).sum::<f64>();
            let cons = cons_t / cbar;
            // variance of the constant (delta through the means)
            let mut full = bb.clone();
            full.push(cons);
            let k1 = kx + 1;
            let mut vv = Matrix::zeros(k1, k1);
            let xb: Vec<f64> = xx.iter().map(|x| x.1 / cbar).collect();
            for i in 0..kx {
                for j in 0..kx {
                    vv[(i, j)] = s2 * ai[(i, j)];
                }
            }
            for i in 0..kx {
                let c: f64 = -(0..kx).map(|j| xb[j] * vv[(i, j)]).sum::<f64>();
                vv[(i, kx)] = c;
                vv[(kx, i)] = c;
            }
            let vc: f64 = s2 / (nt as f64 * cbar * cbar) + (0..kx).map(|i| (0..kx).map(|j| xb[i] * vv[(i, j)] * xb[j]).sum::<f64>()).sum::<f64>();
            vv[(kx, kx)] = vc;
            // u_i from the untransformed data
            let ui: Vec<f64> = p.groups.iter().map(|g| g.iter().map(|&t| y[t] - (0..kx).map(|j| bb[j] * xc[j][t]).sum::<f64>()).sum::<f64>() / g.len() as f64 - cons).collect();
            let mu = util::mean(&ui);
            let su = (ui.iter().map(|u| (u - mu).powi(2)).sum::<f64>() / (ng as f64 - 1.0)).sqrt();
            let se = s2.sqrt();
            let rho_fov = su * su / (su * su + se * se);
            // F tests
            let (fw, _) = {
                let mut vs = Matrix::zeros(kx, kx);
                for i in 0..kx {
                    for j in 0..kx {
                        vs[(i, j)] = vv[(i, j)];
                    }
                }
                let (vi, _) = vs.sym_inverse(1e-14);
                let w: f64 = (0..kx).map(|i| (0..kx).map(|j| bb[i] * vi[(i, j)] * bb[j]).sum::<f64>()).sum();
                (w / kx as f64, 0)
            };
            // pooled regression of transformed y on transformed x and c for the u_i test
            let mut pcols: Vec<Vec<f64>> = tx.iter().map(|o| flat.iter().map(|&t| get(o, t)).collect()).collect();
            pcols.push(flat.iter().map(|&t| get(&tc, t)).collect());
            let py: Vec<f64> = flat.iter().map(|&t| get(&ty, t)).collect();
            let (_, _, pe) = ls(&pcols, &py);
            let prss: f64 = pe.iter().map(|v| v * v).sum();
            let fu = ((prss - rss) / (ng as f64 - 1.0)) / s2;
            out.push(hrow("FE (within) regression with AR(1) disturbances", "Number of obs", util::commas(nt)));
            out.push(hrow(&format!("Group variable: {}", p.ivar), "Number of groups", util::commas(tg.len())));
            out.push(String::new());
            out.push(format!("{:<52}{}", "R-squared:", "Obs per group:"));
            let corr2 = |a: &[f64], b: &[f64]| {
                let (ma, mb) = (util::mean(a), util::mean(b));
                let sab: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
                let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
                let sbb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
                sab * sab / (saa * sbb)
            };
            let fit_w: Vec<f64> = (0..nt).map(|a| (0..kx).map(|j| bb[j] * xcols[j][a]).sum()).collect();
            let r2w = corr2(&yy, &fit_w);
            let fit_o: Vec<f64> = (0..n).map(|t| (0..kx).map(|j| bb[j] * xc[j][t]).sum()).collect();
            let r2o = corr2(y, &fit_o);
            let ym: Vec<f64> = p.groups.iter().map(|g| g.iter().map(|&t| y[t]).sum::<f64>() / g.len() as f64).collect();
            let fm: Vec<f64> = p.groups.iter().map(|g| g.iter().map(|&t| fit_o[t]).sum::<f64>() / g.len() as f64).collect();
            let r2b = corr2(&ym, &fm);
            out.push(format!("     Within  = {:<37.4}{:>15} = {:>10}", r2w, "min", fmt_g(mn, 9)));
            out.push(format!("     Between = {:<37.4}{:>15} = {:>10.1}", r2b, "avg", av));
            out.push(format!("     Overall = {:<37.4}{:>15} = {:>10}", r2o, "max", fmt_g(mx, 9)));
            out.push(String::new());
            out.push(hrow("", &format!("F({kx},{})", fmt_g(df_r, 9)), format!("{:.2}", fw)));
            let ufit: Vec<f64> = (0..n).map(|t| ui[p.groups.iter().position(|g| g.contains(&t)).unwrap_or(0)]).collect();
            let corr_u = corr2(&ufit, &fit_o).sqrt() * if {
                let (ma, mb) = (util::mean(&ufit), util::mean(&fit_o));
                ufit.iter().zip(&fit_o).map(|(a, b)| (a - ma) * (b - mb)).sum::<f64>()
            } < 0.0 { -1.0 } else { 1.0 };
            out.push(hrow(&format!("corr(u_i, Xb) = {:.4}", corr_u), "Prob > F", format!("{:.4}", dist::ftail(kx as f64, df_r, fw))));
            b = full;
            v = vv;
            stats = vec![("rho_ar", rho), ("sigma_u", su), ("sigma_e", se), ("rho_fov", rho_fov), ("df_r", df_r), ("F", fw), ("F_f", fu), ("df_a", ng as f64 - 1.0), ("df_b", df_r), ("r2_w", r2w), ("r2_b", r2b), ("r2_o", r2o), ("corr", corr_u), ("d1", d_bhargava)];
        } else {
            // random effects: Baltagi-Wu GLS
            let flat: Vec<usize> = tg.iter().flatten().copied().collect();
            let mut pcols: Vec<Vec<f64>> = tx.iter().map(|o| flat.iter().map(|&t| get(o, t)).collect()).collect();
            pcols.push(flat.iter().map(|&t| get(&tc, t)).collect());
            let py: Vec<f64> = flat.iter().map(|&t| get(&ty, t)).collect();
            let pos: BTreeMap<usize, usize> = flat.iter().enumerate().map(|(a, &t)| (t, a)).collect();
            // within (orthogonal to c_i) regression for sigma_e
            let proj = |v: &[f64]| -> Vec<f64> {
                let mut r = v.to_vec();
                for g in &tg {
                    let cc: f64 = g.iter().map(|&t| get(&tc, t).powi(2)).sum();
                    let cv: f64 = g.iter().map(|&t| get(&tc, t) * v[pos[&t]]).sum();
                    for &t in g {
                        r[pos[&t]] -= get(&tc, t) * cv / cc;
                    }
                }
                r
            };
            let wy = proj(&py);
            let wx: Vec<Vec<f64>> = pcols[..kx].iter().map(|c| proj(c)).collect();
            let (_, _, we) = ls(&wx, &wy);
            let s2e = we.iter().map(|v| v * v).sum::<f64>() / (nt as f64 - ng as f64 - kx as f64);
            // between: c-weighted panel means of the pooled residuals
            let (_, _, pe) = ls(&pcols, &py);
            let mut om = 0.0;
            let mut gsum = 0.0;
            for g in &tg {
                let cc: f64 = g.iter().map(|&t| get(&tc, t).powi(2)).sum();
                let ce: f64 = g.iter().map(|&t| get(&tc, t) * pe[pos[&t]]).sum();
                om += ce * ce / cc;
                gsum += cc;
            }
            let om2 = om / (ng as f64 - kx as f64 - 1.0);
            let s2u = ((om2 - s2e) / (gsum / ng as f64)).max(0.0);
            let mut gy = py.clone();
            let mut gx = pcols.clone();
            let mut thetas = Vec::new();
            for g in &tg {
                let cc: f64 = g.iter().map(|&t| get(&tc, t).powi(2)).sum();
                let th = 1.0 - (s2e / (s2e + s2u * cc)).sqrt();
                thetas.push(th);
                let adj = |v: &[f64], out: &mut Vec<f64>| {
                    let cv: f64 = g.iter().map(|&t| get(&tc, t) * v[pos[&t]]).sum();
                    for &t in g {
                        out[pos[&t]] = v[pos[&t]] - th * get(&tc, t) * cv / cc;
                    }
                };
                let yy = gy.clone();
                adj(&py, &mut gy);
                let _ = yy;
                for j in 0..=kx {
                    let src = pcols[j].clone();
                    adj(&src, &mut gx[j]);
                }
            }
            let (bb, ai, e) = ls(&gx, &gy);
            let s2 = e.iter().map(|v| v * v).sum::<f64>() / (nt as f64 - kx as f64 - 1.0);
            let _ = s2;
            let vv = ai.scale(s2e);
            let mut vs = Matrix::zeros(kx, kx);
            for i in 0..kx {
                for j in 0..kx {
                    vs[(i, j)] = vv[(i, j)];
                }
            }
            let (vi, _) = vs.sym_inverse(1e-14);
            let wald: f64 = (0..kx).map(|i| (0..kx).map(|j| bb[i] * vi[(i, j)] * bb[j]).sum::<f64>()).sum();
            out.push(hrow("RE GLS regression with AR(1) disturbances", "Number of obs", util::commas(nt)));
            out.push(hrow(&format!("Group variable: {}", p.ivar), "Number of groups", util::commas(tg.len())));
            out.push(String::new());
            out.push(format!("{:<52}{}", "", "Obs per group:"));
            out.push(format!("{:<52}{:>15} = {:>10}", "", "min", fmt_g(mn, 9)));
            out.push(format!("{:<52}{:>15} = {:>10.1}", "", "avg", av));
            out.push(format!("{:<52}{:>15} = {:>10}", "", "max", fmt_g(mx, 9)));
            out.push(String::new());
            out.push(hrow("", &format!("Wald chi2({kx})"), format!("{:.2}", wald)));
            out.push(hrow("corr(u_i, Xb) = 0 (assumed)", "Prob > chi2", format!("{:.4}", dist::chi2tail(kx as f64, wald))));
            let su = s2u.sqrt();
            let se = s2e.sqrt();
            let mut ts = thetas.clone();
            ts.sort_by(f64::total_cmp);
            stats = vec![("rho_ar", rho), ("sigma_u", su), ("sigma_e", se), ("rho_fov", s2u / (s2u + s2e)), ("chi2", wald), ("df_m", kx as f64), ("d1", d_bhargava), ("thta_min", ts[0]), ("thta_max", ts[ts.len() - 1])];
            b = bb;
            v = vv;
            if ts[0] == ts[ts.len() - 1] {
                out.push(String::new());
                out.push(format!("theta = {:.8}", ts[0]));
            }
        }
        out.push(String::new());
        let mut names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        names.push("_cons".into());
        let kk = names.len();
        let fdf = if fe { stats.iter().find(|s| s.0 == "df_r").map(|s| s.1) } else { None };
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &p.dep, names: &names, b: &b, v: &v, omitted: &vec![false; kk], is_base: &vec![false; kk], df: fdf, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let sv = |n: &str| stats.iter().find(|s| s.0 == n).map(|s| s.1).unwrap_or(SYSMISS);
        tbl.push(format!("{:>lw$} | {:>10}", "rho_ar", fmt_g(rho, 9)));
        tbl.push(format!("{:>lw$} | {:>10}", "sigma_u", fmt_g(sv("sigma_u"), 9)));
        tbl.push(format!("{:>lw$} | {:>10}", "sigma_e", fmt_g(sv("sigma_e"), 9)));
        tbl.push(format!("{:>lw$} | {:>10}   (fraction of variance because of u_i)", "rho_fov", fmt_g(sv("rho_fov"), 9)));
        if !fe {
            if sv("thta_min") != sv("thta_max") {
                tbl.push(format!("{:>lw$} |", "theta"));
                tbl.push(format!("{:>lw$} | {:>10}", "min", fmt_g(sv("thta_min"), 9)));
                tbl.push(format!("{:>lw$} | {:>10}", "max", fmt_g(sv("thta_max"), 9)));
            }
        }
        tbl.push(bar);
        out.extend(tbl);
        if fe {
            let df_r = sv("df_r");
            out.push(format!("F test that all u_i=0: F({},{}) = {:.2}{:>22}", ng - 1, fmt_g(df_r, 9), sv("F_f"), format!("Prob > F = {:.4}", dist::ftail(ng as f64 - 1.0, df_r, sv("F_f")))));
        }
        let mut lbi = SYSMISS;
        if cmd.has_opt("lbi", 3) {
            // Baltagi-Wu LBI: Bhargava d plus end-point correction
            let ss: f64 = ew.iter().map(|e| e * e).sum();
            let ends: f64 = p.groups.iter().map(|g| ew[g[0]].powi(2) + ew[*g.last().unwrap()].powi(2)).sum();
            lbi = d_bhargava + ends / ss;
            out.push(format!("modified Bhargava et al. Durbin-Watson = {}", fmt_g(d_bhargava, 9)));
            out.push(format!("Baltagi-Wu LBI = {}", fmt_g(lbi, 9)));
        }
        let text = out.join("\n");
        let mut dp = d.clone();
        dp.names = names.clone();
        dp.omitted = vec![false; kk];
        dp.is_base = vec![false; kk];
        dp.cols = vec![vec![]; kk];
        let model = if fe { "fe" } else { "re" };
        let ivar = p.ivar.clone();
        let tvar = p.tvar.clone();
        est::post(st, "xtregar", &dp, &b, &v, &|e: &mut Results| {
            e.set("N", nt as f64);
            e.set("N_g", tg.len() as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            for (k, val) in &stats {
                e.set(k, *val);
            }
            if lbi != SYSMISS {
                e.set("LBI", lbi);
            }
            e.set_str("model", model);
            e.set_str("ivar", ivar.clone());
            e.set_str("tvar", tvar.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// xthtaylor
// ---------------------------------------------------------------------------

pub struct XtHtaylor;

impl CommandHandler for XtHtaylor {
    fn name(&self) -> &'static str {
        "xthtaylor"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let endog: Vec<String> = match cmd.opt("endog", 3) {
            Some(s) => crate::varlist::expand(ds, s.trim())?,
            None => return Err(rc_err(198, "option endog() required")),
        };
        let p = panel_design(ds, st, cmd, false)?;
        let d = &p.d;
        let n = d.n();
        let ng = p.groups.len();
        let act: Vec<usize> = d.active().into_iter().filter(|&j| d.names[j] != "_cons").collect();
        // classify time-varying / invariant
        let varies = |c: &[f64]| p.groups.iter().any(|g| g.iter().any(|&t| (c[t] - c[g[0]]).abs() > 1e-12 * (1.0 + c[g[0]].abs())));
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut z1 = Vec::new();
        let mut z2 = Vec::new();
        for &j in &act {
            let en = endog.iter().any(|e| e == &d.names[j]);
            match (varies(&d.cols[j]), en) {
                (true, false) => x1.push(j),
                (true, true) => x2.push(j),
                (false, false) => z1.push(j),
                (false, true) => z2.push(j),
            }
        }
        if x1.len() < z2.len() {
            return Err(rc_err(481, "the order condition fails: fewer exogenous time-varying variables than endogenous time-invariant ones"));
        }
        let gmean = |v: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0; n];
            for g in &p.groups {
                let m = g.iter().map(|&t| v[t]).sum::<f64>() / g.len() as f64;
                for &t in g {
                    o[t] = m;
                }
            }
            o
        };
        let y = &d.y;
        let tv: Vec<usize> = x1.iter().chain(&x2).copied().collect();
        let ym = gmean(y);
        let wy: Vec<f64> = (0..n).map(|t| y[t] - ym[t]).collect();
        let xm: Vec<Vec<f64>> = tv.iter().map(|&j| gmean(&d.cols[j])).collect();
        let wx: Vec<Vec<f64>> = tv.iter().enumerate().map(|(a, &j)| (0..n).map(|t| d.cols[j][t] - xm[a][t]).collect()).collect();
        let (bw, _, ew) = ls(&wx, &wy);
        let s2e = ew.iter().map(|e| e * e).sum::<f64>() / (n as f64 - ng as f64);
        // second step: d_i on z with instruments x1, z1
        let di: Vec<f64> = (0..n).map(|t| ym[t] - (0..tv.len()).map(|a| bw[a] * xm[a][t]).sum::<f64>()).collect();
        let ones = vec![1.0; n];
        let mut zcols: Vec<Vec<f64>> = z1.iter().chain(&z2).map(|&j| d.cols[j].clone()).collect();
        zcols.push(ones.clone());
        let mut inst: Vec<Vec<f64>> = x1.iter().map(|&j| gmean(&d.cols[j])).collect();
        inst.extend(z1.iter().map(|&j| d.cols[j].clone()));
        inst.push(ones.clone());
        let (_, _, e2) = tsls(&zcols, &inst, &di);
        let s2 = e2.iter().map(|e| e * e).sum::<f64>() / n as f64;
        let tbar = ng as f64 / p.groups.iter().map(|g| 1.0 / g.len() as f64).sum::<f64>();
        let s2u = (s2 - s2e / tbar).max(0.0);
        // GLS-IV
        let mut theta = vec![0.0; n];
        for g in &p.groups {
            let th = 1.0 - (s2e / (s2e + g.len() as f64 * s2u)).sqrt();
            for &t in g {
                theta[t] = th;
            }
        }
        let star = |v: &[f64]| -> Vec<f64> {
            let m = gmean(v);
            (0..n).map(|t| v[t] - theta[t] * m[t]).collect()
        };
        let order: Vec<usize> = x1.iter().chain(&x2).chain(&z1).chain(&z2).copied().collect();
        let mut xs: Vec<Vec<f64>> = order.iter().map(|&j| star(&d.cols[j])).collect();
        xs.push(star(&ones));
        let ys = star(y);
        let mut zz: Vec<Vec<f64>> = tv.iter().enumerate().map(|(a, &j)| (0..n).map(|t| d.cols[j][t] - xm[a][t]).collect()).collect();
        zz.extend(x1.iter().map(|&j| gmean(&d.cols[j])));
        zz.extend(z1.iter().map(|&j| d.cols[j].clone()));
        zz.push(star(&ones));
        let (b, ai, _) = tsls(&xs, &zz, &ys);
        let v = ai.scale(s2e);
        let kk = b.len();
        let names: Vec<String> = order.iter().map(|&j| d.names[j].clone()).chain(std::iter::once("_cons".to_string())).collect();
        let mut vs = Matrix::zeros(kk - 1, kk - 1);
        for i in 0..kk - 1 {
            for j in 0..kk - 1 {
                vs[(i, j)] = v[(i, j)];
            }
        }
        let (vi, _) = vs.sym_inverse(1e-14);
        let wald: f64 = (0..kk - 1).map(|i| (0..kk - 1).map(|j| b[i] * vi[(i, j)] * b[j]).sum::<f64>()).sum();
        let (mn, av, mx) = opg(&p.groups);
        let level = util::level(cmd, st)?;
        let mut out = vec![String::new()];
        out.push(hrow("Hausman-Taylor estimation", "Number of obs", util::commas(n)));
        out.push(hrow(&format!("Group variable: {}", p.ivar), "Number of groups", util::commas(ng)));
        out.push(String::new());
        out.push(format!("{:<52}{}", "", "Obs per group:"));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "min", fmt_g(mn, 9)));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "avg", fmt_g(av, 7)));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "max", fmt_g(mx, 9)));
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "Random effects u_i ~ i.i.d.", format!("Wald chi2({})", kk - 1), format!("{:.2}", wald)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > chi2", format!("{:.4}", dist::chi2tail((kk - 1) as f64, wald))));
        out.push(String::new());
        // table with group headers
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &p.dep, names: &names, b: &b, v: &v, omitted: &vec![false; kk], is_base: &vec![false; kk], df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        let mut rows = tbl.split_off(3);
        let mut idx = 0;
        for (lab, grp) in [("TVexogenous", &x1), ("TVendogenous", &x2), ("TIexogenous", &z1), ("TIendogenous", &z2)] {
            if grp.is_empty() {
                continue;
            }
            tbl.push(format!("{:<lw$} |", lab));
            for _ in 0..grp.len() {
                tbl.push(rows[idx].clone());
                idx += 1;
            }
        }
        tbl.push(format!("{:>lw$} |", ""));
        tbl.extend(rows.drain(idx..));
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let (su, se) = (s2u.sqrt(), s2e.sqrt());
        tbl.push(format!("{:>lw$} | {:>10}", "sigma_u", fmt_g(su, 9)));
        tbl.push(format!("{:>lw$} | {:>10}", "sigma_e", fmt_g(se, 9)));
        tbl.push(format!("{:>lw$} | {:>10}   (fraction of variance due to u_i)", "rho", fmt_g(s2u / (s2u + s2e), 9)));
        tbl.push(bar);
        out.extend(tbl);
        out.push("Note: TV refers to time varying; TI refers to time invariant.".into());
        let text = out.join("\n");
        let mut dp = d.clone();
        dp.names = names.clone();
        dp.omitted = vec![false; kk];
        dp.is_base = vec![false; kk];
        dp.cols = vec![vec![]; kk];
        let ivar = p.ivar.clone();
        let nm = |v: &[usize]| v.iter().map(|&j| d.names[j].clone()).collect::<Vec<_>>().join(" ");
        let (sx1, sx2, sz1, sz2) = (nm(&x1), nm(&x2), nm(&z1), nm(&z2));
        est::post(st, "xthtaylor", &dp, &b, &v, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            e.set("chi2", wald);
            e.set("df_m", (kk - 1) as f64);
            e.set("sigma_u", su);
            e.set("sigma_e", se);
            e.set("rho", s2u / (s2u + s2e));
            e.set_str("ivar", ivar.clone());
            e.set_str("TVexogenous", sx1.clone());
            e.set_str("TVendogenous", sx2.clone());
            e.set_str("TIexogenous", sz1.clone());
            e.set_str("TIendogenous", sz2.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// xtrc
// ---------------------------------------------------------------------------

pub struct XtRc;

impl CommandHandler for XtRc {
    fn name(&self) -> &'static str {
        "xtrc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let p = panel_design(ds, st, cmd, false)?;
        let d = &p.d;
        let act = d.active();
        let k = act.len();
        let ng = p.groups.len();
        let mut bs = Vec::new();
        let mut vs = Vec::new();
        for g in &p.groups {
            if g.len() <= k {
                return Err(rc_err(2001, "insufficient observations in some panels"));
            }
            let x: Vec<Vec<f64>> = act.iter().map(|&j| g.iter().map(|&t| d.cols[j][t]).collect()).collect();
            let y: Vec<f64> = g.iter().map(|&t| d.y[t]).collect();
            let (b, ai, e) = ls(&x, &y);
            let s2 = e.iter().map(|v| v * v).sum::<f64>() / (g.len() - k) as f64;
            bs.push(b);
            vs.push(ai.scale(s2));
        }
        let ngf = ng as f64;
        let bbar: Vec<f64> = (0..k).map(|j| bs.iter().map(|b| b[j]).sum::<f64>() / ngf).collect();
        let mut s1 = Matrix::zeros(k, k);
        for b in &bs {
            for i in 0..k {
                for j in 0..k {
                    s1[(i, j)] += (b[i] - bbar[i]) * (b[j] - bbar[j]) / (ngf - 1.0);
                }
            }
        }
        let mut sig = s1.clone();
        for v in &vs {
            for i in 0..k {
                for j in 0..k {
                    sig[(i, j)] -= v[(i, j)] / ngf;
                }
            }
        }
        if sig.cholesky().is_err() {
            sig = s1;
        }
        let mut wsum = Matrix::zeros(k, k);
        let mut wb = vec![0.0; k];
        let mut ws = Vec::new();
        for (b, v) in bs.iter().zip(&vs) {
            let w = sig.add(v).unwrap().sym_inverse(1e-14).0;
            for i in 0..k {
                for j in 0..k {
                    wsum[(i, j)] += w[(i, j)];
                    wb[i] += w[(i, j)] * b[j];
                }
            }
            ws.push(w);
        }
        let vb = wsum.sym_inverse(1e-14).0;
        let beta: Vec<f64> = (0..k).map(|i| (0..k).map(|j| vb[(i, j)] * wb[j]).sum()).collect();
        // test of parameter constancy
        let mut vsum = Matrix::zeros(k, k);
        let mut vbsum = vec![0.0; k];
        let vinv: Vec<Matrix> = vs.iter().map(|v| v.sym_inverse(1e-14).0).collect();
        for (b, vi) in bs.iter().zip(&vinv) {
            for i in 0..k {
                for j in 0..k {
                    vsum[(i, j)] += vi[(i, j)];
                    vbsum[i] += vi[(i, j)] * b[j];
                }
            }
        }
        let vsi = vsum.sym_inverse(1e-14).0;
        let bstar: Vec<f64> = (0..k).map(|i| (0..k).map(|j| vsi[(i, j)] * vbsum[j]).sum()).collect();
        let mut chi = 0.0;
        for (b, vi) in bs.iter().zip(&vinv) {
            for i in 0..k {
                for j in 0..k {
                    chi += (b[i] - bstar[i]) * vi[(i, j)] * (b[j] - bstar[j]);
                }
            }
        }
        let dfc = (k * (ng - 1)) as f64;
        let mut bfull = vec![0.0; d.k()];
        for (a, &j) in act.iter().enumerate() {
            bfull[j] = beta[a];
        }
        let v = est::expand_v(&vb, &act, d.k());
        let (wald, df_m) = est::wald_all(d, &bfull, &v);
        let (mn, av, mx) = opg(&p.groups);
        let level = util::level(cmd, st)?;
        let mut out = vec![String::new()];
        out.push(hrow("Random-coefficients regression", "Number of obs", util::commas(d.n())));
        out.push(hrow(&format!("Group variable: {}", p.ivar), "Number of groups", util::commas(ng)));
        out.push(String::new());
        out.push(format!("{:<52}{}", "", "Obs per group:"));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "min", fmt_g(mn, 9)));
        out.push(format!("{:<52}{:>15} = {:>10.1}", "", "avg", av));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "max", fmt_g(mx, 9)));
        out.push(String::new());
        out.push(hrow("", &format!("Wald chi2({})", fmt_g(df_m, 9)), format!("{:.2}", wald)));
        out.push(hrow("", "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, wald))));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &p.dep, names: &d.names, b: &bfull, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
        out.push(format!("Test of parameter constancy: chi2({}) = {:.2}{:>25}", fmt_g(dfc, 9), chi, format!("Prob > chi2 = {:.4}", dist::chi2tail(dfc, chi))));
        let text = out.join("\n");
        let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        let sigm = sig.clone().with_names(names.clone(), names.clone());
        let mut beta_m = Matrix::zeros(ng, k);
        for (g, b) in bs.iter().enumerate() {
            for j in 0..k {
                beta_m[(g, j)] = b[j];
            }
        }
        let ivar = p.ivar.clone();
        est::post(st, "xtrc", d, &bfull, &v, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn);
            e.set("g_avg", av);
            e.set("g_max", mx);
            e.set("chi2", wald);
            e.set("df_m", df_m);
            e.set("chi2_c", chi);
            e.set("df_chi2c", dfc);
            e.set_mat("Sigma", sigm.clone());
            e.set_mat("beta_ps", beta_m.clone());
            e.set_str("ivar", ivar.clone());
            e.set_str("title", "Random-coefficients regression");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
