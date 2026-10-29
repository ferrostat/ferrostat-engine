//! `lpirf` and `ivlpirf`: impulse-response functions by local projections
//! (Jorda 2005). For step s = 1..steps, the response of y_i to y_j is the
//! coefficient on L.y_j in the regression of F(s-1).y_i on lags 1..p of
//! all the dependent variables (plus exogenous variables); `ivlpirf` uses
//! the contemporaneous value of an endogenous impulse instrumented by
//! two-stage least squares, with responses at steps 0..steps.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_ml::{self, Post};
use crate::tsops;
use crate::tsreg::hline;
use crate::util;

/// Value of a series at time t + shift (None if missing / outside).
fn at(col: &[f64], pos: &std::collections::HashMap<i64, usize>, t: f64, shift: i64) -> Option<f64> {
    pos.get(&(t.round() as i64 + shift)).map(|&i| col[i]).filter(|v| !is_missing(*v))
}

/// OLS (or 2SLS when `z` is given) with robust (default) or ols variance:
/// returns (b, V) for rows x.
fn fit(x: &[Vec<f64>], y: &[f64], z: Option<&[Vec<f64>]>, robust: bool) -> Option<(Vec<f64>, Matrix)> {
    let n = x.len();
    let k = x.first()?.len();
    if n <= k {
        return None;
    }
    let xr: Vec<Vec<f64>> = match z {
        Some(z) => {
            // first stage: project x on z
            let kz = z[0].len();
            let mut zz = Matrix::zeros(kz, kz);
            for r in z {
                for a in 0..kz {
                    for c in 0..kz {
                        zz[(a, c)] += r[a] * r[c];
                    }
                }
            }
            let zi = zz.sym_inverse(1e-12).0;
            let mut out = vec![vec![0.0; k]; n];
            for j in 0..k {
                let zy: Vec<f64> = (0..kz).map(|a| (0..n).map(|t| z[t][a] * x[t][j]).sum()).collect();
                let g = ts_ml::mv(&zi, &zy);
                for t in 0..n {
                    out[t][j] = ts_ml::dot(&z[t], &g);
                }
            }
            out
        }
        None => x.to_vec(),
    };
    let mut xx = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for t in 0..n {
        for a in 0..k {
            xy[a] += xr[t][a] * y[t];
            for c in 0..k {
                xx[(a, c)] += xr[t][a] * x[t][c];
            }
        }
    }
    let xi = xx.inverse().ok()?;
    let b = ts_ml::mv(&xi, &xy);
    let e: Vec<f64> = (0..n).map(|t| y[t] - ts_ml::dot(&x[t], &b)).collect();
    let v = if robust {
        let mut meat = Matrix::zeros(k, k);
        for t in 0..n {
            for a in 0..k {
                for c in 0..k {
                    meat[(a, c)] += xr[t][a] * xr[t][c] * e[t] * e[t];
                }
            }
        }
        let s = n as f64 / (n - k) as f64;
        xi.mul(&meat).ok()?.mul(&xi.transpose()).ok()?.scale(s)
    } else {
        let s2 = e.iter().map(|v| v * v).sum::<f64>() / (n - k) as f64;
        let mut xxr = Matrix::zeros(k, k);
        for t in 0..n {
            for a in 0..k {
                for c in 0..k {
                    xxr[(a, c)] += xr[t][a] * xr[t][c];
                }
            }
        }
        xxr.sym_inverse(1e-12).0.scale(s2)
    };
    Some((b, v))
}

pub struct Lpirf {
    pub iv: bool,
}

impl CommandHandler for Lpirf {
    fn name(&self) -> &'static str {
        if self.iv { "ivlpirf" } else { "lpirf" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let deps: Vec<String> = tsops::expand_text(cmd.main.trim()).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        if deps.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let steps = util::opt_num(cmd, "steps", 4)?.unwrap_or(8.0) as i64;
        let lags = cmd.opt("lags", 3).map_or(Ok(vec![1, 2]), ts_ml::parse_lags)?;
        let exog: Vec<String> = cmd.opt("exog", 3).map_or(Ok(vec![]), |s| tsops::expand_text(s).split_whitespace().map(|v| tsops::resolve(ds, v)).collect())?;
        let (endo, insts): (Option<String>, Vec<String>) = if self.iv {
            let s = cmd.opt("endogenous", 4).ok_or_else(|| rc_err(198, "option endogenous() required"))?;
            let (l, r) = s.split_once('=').ok_or_else(|| rc_err(198, "endogenous(varname = instruments) required"))?;
            (Some(tsops::resolve(ds, l.trim())?), tsops::expand_text(r).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?)
        } else {
            (None, vec![])
        };
        let robust = !cmd.opt("vce", 3).is_some_and(|v| v.trim().starts_with("ols"));
        let tvar = tsops::time_var(st)?;
        let tc = util::col(ds, &tvar)?;
        let mask = util::touse(ds, st, cmd)?;
        let pos: std::collections::HashMap<i64, usize> = (0..ds.n_obs).filter(|&i| !is_missing(tc[i])).map(|i| (tc[i].round() as i64, i)).collect();
        let col = |v: &str| tsops::values(ds, st, v);
        let dcols: Vec<Vec<f64>> = deps.iter().map(|v| col(v)).collect::<Result<_, _>>()?;
        let xcols: Vec<Vec<f64>> = exog.iter().map(|v| col(v)).collect::<Result<_, _>>()?;
        let ecol: Option<Vec<f64>> = endo.as_ref().map(|v| col(v)).transpose()?;
        let icols: Vec<Vec<f64>> = insts.iter().map(|v| col(v)).collect::<Result<_, _>>()?;
        let mut times: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(tc[i])).map(|i| tc[i]).collect();
        times.sort_by(f64::total_cmp);
        let s0 = if self.iv { 0 } else { 1 };
        // regressors at time t: [impulse (iv)], lags of deps (and of the endogenous impulse), exog, const
        let regs = |t: f64| -> Option<Vec<f64>> {
            let mut r = Vec::new();
            if let Some(e) = &ecol {
                r.push(at(e, &pos, t, 0)?);
            }
            for &l in &lags {
                for c in &dcols {
                    r.push(at(c, &pos, t, -(l as i64))?);
                }
                if let Some(e) = &ecol {
                    r.push(at(e, &pos, t, -(l as i64))?);
                }
            }
            for c in &xcols {
                r.push(at(c, &pos, t, 0)?);
            }
            r.push(1.0);
            Some(r)
        };
        let inst_row = |t: f64, base: &[f64]| -> Option<Vec<f64>> {
            let mut r: Vec<f64> = icols.iter().map(|c| at(c, &pos, t, 0)).collect::<Option<Vec<_>>>()?;
            r.extend(base[1..].iter().copied());
            Some(r)
        };
        let mut names = Vec::new();
        let mut b = Vec::new();
        let mut vblocks: Vec<f64> = Vec::new();
        let mut nmax = 0usize;
        let impulses: Vec<String> = if let Some(e) = &endo { vec![e.clone()] } else { deps.clone() };
        for (ri, rv) in deps.iter().enumerate() {
            for (ii, iv) in impulses.iter().enumerate() {
                for s in s0..=steps {
                    let shift = if self.iv { s } else { s - 1 };
                    let mut xs = Vec::new();
                    let mut ys = Vec::new();
                    let mut zs = Vec::new();
                    for &t in &times {
                        let Some(y) = at(&dcols[ri], &pos, t, shift) else { continue };
                        let Some(x) = regs(t) else { continue };
                        if self.iv {
                            let Some(z) = inst_row(t, &x) else { continue };
                            zs.push(z);
                        }
                        xs.push(x);
                        ys.push(y);
                    }
                    nmax = nmax.max(xs.len());
                    let (bb, vv) = fit(&xs, &ys, if self.iv { Some(&zs) } else { None }, robust).ok_or_else(|| rc_err(2001, "insufficient observations"))?;
                    // coefficient on L.impulse (lpirf) or on the impulse (ivlpirf)
                    let j = if self.iv { 0 } else { ii };
                    names.push(format!("{rv}:F{}.{iv}", s));
                    b.push(bb[j]);
                    vblocks.push(vv[(j, j)]);
                }
            }
        }
        let k = b.len();
        let mut v = Matrix::zeros(k, k);
        for j in 0..k {
            v[(j, j)] = vblocks[j];
        }
        let lagtxt = if lags.len() > 1 && lags.windows(2).all(|w| w[1] == w[0] + 1) { format!("{}/{}", lags[0], lags[lags.len() - 1]) } else { lags.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" ") };
        let header = vec![hline(&format!("Sample: {} thru {}", tsops::time_label(ds, &tvar, times[0]), tsops::time_label(ds, &tvar, *times.last().unwrap())), "Number of obs", &fmt_g(nmax as f64, 10)), format!("Lags: {lagtxt}")];
        let title = if self.iv { "Local-projection IV impulse-response function" } else { "Local-projection impulse-response function" };
        let text = ts_ml::post(
            st,
            cmd,
            Post { title, cmd: if self.iv { "ivlpirf" } else { "lpirf" }, depvar: "", names, b, v, ll: f64::NAN, n: nmax, header, vcetype: if robust { "Robust" } else { "" }, vce: if robust { "robust" } else { "ols" }, log: vec![], notes: vec![], converged: true, iters: 0, df: None },
            None,
        )?;
        let e = &mut st.e;
        e.scalars.remove("ll");
        e.set("steps", steps as f64);
        e.set("lags", lags.iter().max().copied().unwrap_or(0) as f64);
        e.set_str("depvar", deps.join(" "));
        e.set_str("exog", exog.join(" "));
        e.set_str("impulse", impulses.join(" "));
        e.set_str("response", deps.join(" "));
        Ok(text)
    }
}
