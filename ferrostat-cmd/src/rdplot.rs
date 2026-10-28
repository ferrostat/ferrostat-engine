//! `rdplot`: regression-discontinuity plots with data-driven bins
//! (Calonico, Cattaneo & Titiunik 2015, JASA). Bin means of the outcome
//! within evenly spaced or quantile-spaced bins on each side of the cutoff,
//! with the number of bins chosen to minimise the integrated MSE or to mimic
//! the variability of the raw data, overlaid with a global polynomial fit.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::rdrobust::{cross, ff, hline, inv_sym, kweight, powers, ralign, Kernel, Ln};
use crate::store::Store;
use crate::util;

pub struct Rdplot;

/// Least-squares polynomial of order k in x (fitted on a standardised
/// scale); returns fitted values and first derivatives at `at`.
fn poly_fit(x: &[f64], y: &[f64], k: usize, at: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = util::mean(x);
    let s = util::variance(x).sqrt().max(1e-300);
    let t: Vec<f64> = x.iter().map(|v| (v - m) / s).collect();
    let r = powers(&t, 0.0, k);
    let ones = vec![1.0; x.len()];
    let g = inv_sym(&cross(&r, &ones, &r));
    let ry: Vec<f64> = (0..=k).map(|j| r[j].iter().zip(y).map(|(a, b)| a * b).sum()).collect();
    let beta: Vec<f64> = (0..=k).map(|a| (0..=k).map(|b| g[(a, b)] * ry[b]).sum()).collect();
    let mut fit = Vec::with_capacity(at.len());
    let mut der = Vec::with_capacity(at.len());
    for &xv in at {
        let tv = (xv - m) / s;
        let mut f = 0.0;
        let mut dv = 0.0;
        for j in 0..=k {
            f += beta[j] * tv.powi(j as i32);
            if j >= 1 {
                dv += j as f64 * beta[j] * tv.powi(j as i32 - 1);
            }
        }
        fit.push(f);
        der.push(dv / s);
    }
    (fit, der)
}

/// Equally spaced grid of `n` points from a to b.
fn rangen(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

/// Largest j (1-based) with at[j] <= x.
fn group_id(x: f64, at: &[f64]) -> Option<usize> {
    at.iter().rposition(|&a| a <= x).map(|j| j + 1)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 0 {
        return SYSMISS;
    }
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// Bin selectors of one side: (IMSE spacings, IMSE polynomial, MV spacings,
/// MV polynomial) for evenly (es) and quantile (qs) spaced bins.
struct BinChoice {
    es_hat_dw: f64,
    es_chk_dw: f64,
    es_hat_mv: f64,
    es_chk_mv: f64,
    qs_hat_dw: f64,
    qs_chk_dw: f64,
    qs_hat_mv: f64,
    qs_chk_mv: f64,
}

fn bin_choice(x: &[f64], y: &[f64], range: f64, n_total: f64) -> BinChoice {
    let ns = x.len();
    let nf = ns as f64;
    let k = 4.min(ns.saturating_sub(1));
    let y2: Vec<f64> = y.iter().map(|v| v * v).collect();
    let xbar: Vec<f64> = x.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    let (mu0, mu1) = poly_fit(x, y, k, x);
    let (mu2, _) = poly_fit(x, &y2, k, x);
    let (mu0i, mu1i) = poly_fit(x, y, k, &xbar);
    let (mu2i, _) = poly_fit(x, &y2, k, &xbar);
    let var_y = util::variance(y);
    let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let fix = |v: f64| if v < 0.0 { var_y } else { v };
    let s2_bar: Vec<f64> = mu2i.iter().zip(&mu0i).map(|(a, b)| fix(a - b * b)).collect();
    let s2: Vec<f64> = mu2.iter().zip(&mu0).map(|(a, b)| fix(a - b * b)).collect();
    let n = n_total;
    let b_es = (range * range / (12.0 * n)) * mu1.iter().map(|v| v * v).sum::<f64>();
    let v_es_hat = (0.5 / range) * dx.iter().zip(&dy).map(|(a, b)| a * b * b).sum::<f64>();
    let v_es_chk = (1.0 / range) * dx.iter().zip(&s2_bar).map(|(a, b)| a * b).sum::<f64>();
    let b_qs = (nf * nf / (24.0 * n)) * dx.iter().zip(&mu1i).map(|(a, b)| a * a * b * b).sum::<f64>();
    let v_qs_hat = (1.0 / (2.0 * nf)) * dy.iter().map(|v| v * v).sum::<f64>();
    let v_qs_chk = (1.0 / nf) * s2.iter().sum::<f64>();
    let dw = |b: f64, v: f64| ((2.0 * b / v) * n).powf(1.0 / 3.0).ceil();
    let mv = |v: f64| ((var_y / v) * (n / n.ln().powi(2))).ceil();
    BinChoice {
        es_hat_dw: dw(b_es, v_es_hat),
        es_chk_dw: dw(b_es, v_es_chk),
        es_hat_mv: mv(v_es_hat),
        es_chk_mv: mv(v_es_chk),
        qs_hat_dw: dw(b_qs, v_qs_hat),
        qs_chk_dw: dw(b_qs, v_qs_chk),
        qs_hat_mv: mv(v_qs_hat),
        qs_chk_mv: mv(v_qs_chk),
    }
}

fn pair(cmd: &Command, name: &str, min: usize) -> Result<Option<(f64, f64)>, String> {
    let Some(t) = cmd.opt(name, min) else { return Ok(None) };
    let v: Vec<f64> = t.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, format!("{name}() invalid")))).collect::<Result<_, _>>()?;
    match v.len() {
        0 => Ok(None),
        1 => Ok(Some((v[0], v[0]))),
        2 => Ok(Some((v[0], v[1]))),
        _ => Err(rc_err(125, format!("{name}() accepts at most two inputs"))),
    }
}

/// Per-bin summary.
#[derive(Clone, Debug)]
struct Bin {
    id: i64,
    n: usize,
    mean_x: f64,
    mean_y: f64,
    sd_y: f64,
    lo: f64,
    hi: f64,
}

fn collapse(x: &[f64], y: &[f64], ids: &[usize], edges: &[f64], sign: i64, offset: i64) -> Vec<Bin> {
    let nb = edges.len() - 1;
    let mut out = Vec::new();
    for j in 1..=nb {
        let idx: Vec<usize> = (0..x.len()).filter(|&i| ids[i] == j).collect();
        if idx.is_empty() {
            continue;
        }
        let xs: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let sd = if ys.len() > 1 { util::variance(&ys).sqrt() } else { SYSMISS };
        out.push(Bin { id: sign * (j as i64) + offset, n: idx.len(), mean_x: util::mean(&xs), mean_y: util::mean(&ys), sd_y: sd, lo: edges[j - 1], hi: edges[j] });
    }
    out
}

impl CommandHandler for Rdplot {
    fn name(&self) -> &'static str {
        "rdplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(if vars.len() < 2 { 102 } else { 103 }, if vars.len() < 2 { "too few variables specified" } else { "too many variables specified" }));
        }
        if cmd.opt("covs", 4).is_some() {
            return Err(rc_err(198, "covs() is not supported by rdplot here; residualize the outcome first"));
        }
        let (yname, xname) = (vars[0].clone(), vars[1].clone());
        let c = util::opt_num(cmd, "c", 1)?.unwrap_or(0.0);
        let p = util::opt_num(cmd, "p", 1)?.unwrap_or(4.0);
        if p < 0.0 || p.fract() != 0.0 {
            return Err(rc_err(411, "p() and nbins() should be a positive integers"));
        }
        let p = p as usize;
        let kernel = Kernel::parse(cmd.opt("kernel", 3).unwrap_or(""), Kernel::Uniform)?;
        let mut binselect = cmd.opt("binselect", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "esmv".into());
        if binselect.is_empty() {
            binselect = "esmv".into();
        }
        if !["es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr"].contains(&binselect.as_str()) {
            return Err(rc_err(7, "binselect() incorrectly specified"));
        }
        let masspoints = cmd.opt("masspoints", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "adjust".into());
        if !["adjust", "check", "off"].contains(&masspoints.as_str()) {
            return Err(rc_err(125, "masspoints() must be one of check, adjust, off"));
        }
        let ci = util::opt_num(cmd, "ci", 2)?.unwrap_or(0.0);
        let shade = cmd.has_opt("shade", 5);
        let hide = cmd.has_opt("hide", 4);
        let genvars = cmd.has_opt("genvars", 6);
        let nbins = pair(cmd, "nbins", 5)?;
        let scale = pair(cmd, "scale", 5)?.unwrap_or((1.0, 1.0));
        let support = pair(cmd, "support", 7)?;
        let wname = cmd.opt("weights", 7).map(|s| crate::varlist::resolve_name(ds, s.trim())).transpose()?;
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut mv = vec![yname.clone(), xname.clone()];
        mv.extend(wname.iter().cloned());
        util::markout(ds, &mut mask, &mv);
        let yv = util::col(ds, &yname)?;
        let xv = util::col(ds, &xname)?;
        let wv = wname.as_ref().map(|w| util::col(ds, w)).transpose()?;
        if let Some(w) = &wv {
            for i in 0..ds.n_obs {
                if mask[i] && w[i] <= 0.0 {
                    mask[i] = false;
                }
            }
        }
        let mut rows = util::rows(&mask);
        rows.sort_by(|&a, &b| xv[a].total_cmp(&xv[b]));
        let n = rows.len();
        if n < 20 {
            return Err(rc_err(2001, "Not enough observations to perform bin calculations"));
        }
        let xs: Vec<f64> = rows.iter().map(|&i| xv[i]).collect();
        let ys: Vec<f64> = rows.iter().map(|&i| yv[i]).collect();
        let mut x_min = xs[0];
        let mut x_max = xs[n - 1];
        if let Some((sl, sr)) = support {
            x_min = x_min.min(sl);
            x_max = x_max.max(sr);
        }
        if c <= x_min || c >= x_max {
            return Err(rc_err(125, format!("c() should be set within the range of {xname}")));
        }
        let split = xs.iter().position(|&x| x >= c).unwrap_or(n);
        let (x_l, x_r) = (&xs[..split], &xs[split..]);
        let (y_l, y_r) = (&ys[..split], &ys[split..]);
        let (n_l, n_r) = (x_l.len(), x_r.len());
        if n_l < 2 || n_r < 2 {
            return Err(rc_err(2001, "Not enough observations on each side of the cutoff"));
        }
        let range_l = (c - x_min).abs();
        let range_r = (x_max - c).abs();
        let (h_l, h_r) = pair(cmd, "h", 1)?.unwrap_or((range_l, range_r));
        let n_h_l = x_l.iter().filter(|&&x| x >= c - h_l).count();
        let n_h_r = x_r.iter().filter(|&&x| x <= c + h_r).count();
        let sd_l = util::variance(y_l).sqrt();
        let sd_r = util::variance(y_r).sqrt();
        let mut notes = Vec::new();
        // mass points
        let mut masspoints_found = false;
        if masspoints != "off" {
            let ml = {
                let mut v = x_l.to_vec();
                v.dedup();
                v.len()
            };
            let mr = {
                let mut v = x_r.to_vec();
                v.dedup();
                v.len()
            };
            if 1.0 - ml as f64 / n_l as f64 >= 0.2 || 1.0 - mr as f64 / n_r as f64 >= 0.2 {
                masspoints_found = true;
                notes.push("Mass points detected in the running variable.".to_string());
                if masspoints == "adjust" {
                    binselect = match binselect.as_str() {
                        "es" => "espr".into(),
                        "esmv" => "esmvpr".into(),
                        "qs" => "qspr".into(),
                        "qsmv" => "qsmvpr".into(),
                        b => b.into(),
                    };
                } else {
                    notes.push("Try using option masspoints(adjust).".into());
                }
            }
        }
        // polynomial fit on each side
        let fit_side = |x: &[f64], y: &[f64], w: Option<Vec<f64>>, h: f64| -> Vec<f64> {
            let mut kw = kweight(x, c, h + 1e-8, kernel);
            if let Some(w) = w {
                for (a, b) in kw.iter_mut().zip(w) {
                    *a *= b;
                }
            }
            let r = powers(x, c, p);
            let g = inv_sym(&cross(&r, &kw, &r));
            let ry: Vec<f64> = (0..=p).map(|j| (0..x.len()).map(|i| r[j][i] * kw[i] * y[i]).sum()).collect();
            (0..=p).map(|a| (0..=p).map(|b| g[(a, b)] * ry[b]).sum()).collect()
        };
        let wl = wv.as_ref().map(|w| rows[..split].iter().map(|&i| w[i]).collect::<Vec<f64>>());
        let wr = wv.as_ref().map(|w| rows[split..].iter().map(|&i| w[i]).collect::<Vec<f64>>());
        let coef_l = fit_side(x_l, y_l, wl, h_l);
        let coef_r = fit_side(x_r, y_r, wr, h_r);
        let poly = |coef: &[f64], x: f64| -> f64 { coef.iter().enumerate().map(|(j, b)| b * (x - c).powi(j as i32)).sum() };
        // number of bins
        let bl = bin_choice(x_l, y_l, c - x_min, n as f64);
        let br = bin_choice(x_r, y_r, x_max - c, n as f64);
        let pick = |b: &BinChoice| -> (f64, f64, f64) {
            match binselect.as_str() {
                "es" => (b.es_hat_dw, b.es_hat_dw, b.es_hat_mv),
                "espr" => (b.es_chk_dw, b.es_chk_dw, b.es_chk_mv),
                "esmv" => (b.es_hat_mv, b.es_hat_dw, b.es_hat_mv),
                "esmvpr" => (b.es_chk_mv, b.es_chk_dw, b.es_chk_mv),
                "qs" => (b.qs_hat_dw, b.qs_hat_dw, b.qs_hat_mv),
                "qspr" => (b.qs_chk_dw, b.qs_chk_dw, b.qs_chk_mv),
                "qsmv" => (b.qs_hat_mv, b.qs_hat_dw, b.qs_hat_mv),
                _ => (b.qs_chk_mv, b.qs_chk_dw, b.qs_chk_mv),
            }
        };
        let (mut jl_orig, jl_imse, jl_mv) = pick(&bl);
        let (mut jr_orig, jr_imse, jr_mv) = pick(&br);
        let manual = matches!(nbins, Some((a, b)) if a != 0.0 && b != 0.0);
        if let Some((a, b)) = nbins {
            if a < 0.0 || b < 0.0 {
                return Err(rc_err(411, "p() and nbins() should be a positive integers"));
            }
            if manual {
                jl_orig = a;
                jr_orig = b;
            }
        }
        if sd_l == 0.0 {
            jl_orig = 1.0;
            notes.push("Warning: not enough variability in the outcome variable below the threshold".into());
        }
        if sd_r == 0.0 {
            jr_orig = 1.0;
            notes.push("Warning: not enough variability in the outcome variable above the threshold".into());
        }
        let j_l = ((scale.0 * jl_orig).round().max(1.0)) as usize;
        let j_r = ((scale.1 * jr_orig).round().max(1.0)) as usize;
        if !jl_orig.is_finite() || !jr_orig.is_finite() {
            return Err(rc_err(498, "could not determine the number of bins"));
        }
        // bin edges
        let quantile_bins = binselect.starts_with("qs");
        let (edges_l, edges_r) = if quantile_bins {
            let cuts = |v: &[f64], j: usize| -> Vec<f64> { (1..j).map(|k| util::pctile_sorted(v, 100.0 * k as f64 / j as f64)).collect() };
            let mut el = vec![x_min - 1e-8];
            el.extend(cuts(x_l, j_l));
            el.push(c);
            let mut er = vec![c];
            er.extend(cuts(x_r, j_r));
            er.push(x_max + 1e-8);
            (el, er)
        } else {
            (rangen(x_min - 1e-8, c, j_l + 1), rangen(c, x_max + 1e-8, j_r + 1))
        };
        let ids_l: Vec<usize> = x_l.iter().map(|&x| group_id(x, &edges_l).unwrap_or(1)).collect();
        let ids_r: Vec<usize> = x_r.iter().map(|&x| group_id(x, &edges_r).unwrap_or(1)).collect();
        let bins_l = collapse(x_l, y_l, &ids_l, &edges_l, 1, -(j_l as i64) - 1);
        let bins_r = collapse(x_r, y_r, &ids_r, &edges_r, 1, 0);
        let len_l: Vec<f64> = edges_l.windows(2).map(|w| w[1] - w[0]).collect();
        let len_r: Vec<f64> = edges_r.windows(2).map(|w| w[1] - w[0]).collect();
        let lvl = if ci > 0.0 { ci } else { 95.0 };
        let bins: Vec<Bin> = bins_l.iter().chain(&bins_r).cloned().collect();
        let ci_of = |b: &Bin| -> (f64, f64, f64) {
            let se = if is_missing(b.sd_y) { SYSMISS } else { b.sd_y / (b.n as f64).sqrt() };
            if is_missing(se) {
                return (SYSMISS, SYSMISS, SYSMISS);
            }
            let dfv = ((b.n as f64) - 1.0).max(1.0);
            let q = -dist::invt(dfv, ((1.0 - lvl / 100.0) / 2.0).abs());
            (se, b.mean_y - q * se, b.mean_y + q * se)
        };
        // ---- output ---------------------------------------------------------
        let scale_l = j_l as f64 / jl_imse;
        let scale_r = j_r as f64 / jr_imse;
        let bst = match binselect.as_str() {
            "es" => "evenly spaced number of bins using spacings estimators.",
            "espr" => "evenly spaced number of bins using polynomial regression.",
            "esmv" => "evenly spaced mimicking variance number of bins using spacings estimators.",
            "esmvpr" => "evenly spaced mimicking variance number of bins using polynomial regression.",
            "qs" => "quantile spaced number of bins using spacings estimators.",
            "qspr" => "quantile spaced number of bins using polynomial regression.",
            "qsmv" => "quantile spaced mimicking variance quantile spaced using spacings estimators.",
            _ => "quantile spaced mimicking variance number of bins using polynomial regression.",
        };
        let bst = if manual { "RD plot with manually set number of bins.".to_string() } else { bst.to_string() };
        let mut out: Vec<String> = notes;
        out.push(String::new());
        out.push(format!("RD Plot with {bst}"));
        out.push(String::new());
        let cut = format!(" Cutoff c = {}", crate::format::fmt_g(c, 9).trim());
        let two = |label: &str, a: String, b: String| -> Ln { Ln::new().t(&ralign(label, 21)).col(22).t(" | ").col(23).t(&a).col(37).t(&b) };
        out.push(Ln::new().t(&ralign(&cut, 21)).col(22).t(" | ").col(23).t("Left of c").col(36).t("Right of c").col(54).t("Number of obs  = ").t(&ff(n as f64, 10, 0)).s());
        out.push(Ln::new().t(&hline(22)).t("+").t(&hline(22)).col(54).t("Kernel         = ").t(&ralign(kernel.title(), 10)).s());
        out.push(two("Number of obs", ff(n_l as f64, 9, 0), ff(n_r as f64, 9, 0)).s());
        out.push(two("Eff. Number of obs", ff(n_h_l as f64, 9, 0), ff(n_h_r as f64, 9, 0)).s());
        out.push(two("Order poly. fit (p)", ff(p as f64, 9, 0), ff(p as f64, 9, 0)).s());
        out.push(two("BW poly. fit (h)", ff(h_l, 9, 3), ff(h_r, 9, 3)).s());
        out.push(two("Number of bins scale", ff(scale.0, 9, 3), ff(scale.1, 9, 3)).s());
        out.push(String::new());
        out.push(format!("Outcome: {yname}. Running variable: {xname}."));
        out.push(format!("{}+{}", hline(22), hline(22)));
        out.push(Ln::new().col(22).t(" | ").col(23).t("Left of c").col(36).t("Right of c").s());
        out.push(format!("{}+{}", hline(22), hline(22)));
        out.push(two("Bins selected", ff(j_l as f64, 9, 0), ff(j_r as f64, 9, 0)).s());
        out.push(two("Average bin length", ff(util::mean(&len_l), 9, 3), ff(util::mean(&len_r), 9, 3)).s());
        out.push(two("Median bin length", ff(median(&len_l), 9, 3), ff(median(&len_r), 9, 3)).s());
        out.push(format!("{}+{}", hline(22), hline(22)));
        out.push(two("IMSE-optimal bins", ff(jl_imse, 9, 0), ff(jr_imse, 9, 0)).s());
        out.push(two("Mimicking Var. bins", ff(jl_mv, 9, 0), ff(jr_mv, 9, 0)).s());
        out.push(format!("{}+{}", hline(22), hline(22)));
        out.push(Ln::new().t("Rel. to IMSE-optimal:").col(22).t(" |").s());
        out.push(two("Implied scale", ff(scale_l, 9, 3), ff(scale_r, 9, 3)).s());
        out.push(two("WIMSE var. weight", ff(1.0 / (1.0 + scale_l.powi(3)), 9, 3), ff(1.0 / (1.0 + scale_r.powi(3)), 9, 3)).s());
        out.push(two("WIMSE bias weight", ff(scale_l.powi(3) / (1.0 + scale_l.powi(3)), 9, 3), ff(scale_r.powi(3) / (1.0 + scale_r.powi(3)), 9, 3)).s());
        out.push(format!("{}+{}", hline(22), hline(22)));
        if masspoints_found && masspoints == "adjust" {
            out.push("Bins adjusted for mass points in the running variable.".into());
        }
        let text = out.join("\n");
        // ---- generated variables ------------------------------------------
        if genvars {
            let names = ["rdplot_id", "rdplot_N", "rdplot_min_bin", "rdplot_max_bin", "rdplot_mean_bin", "rdplot_mean_x", "rdplot_mean_y", "rdplot_se_y", "rdplot_ci_l", "rdplot_ci_r", "rdplot_hat_y"];
            for nm in names {
                if ds.var_index.contains_key(nm) {
                    return Err(rc_err(110, format!("variable {nm} already defined")));
                }
            }
            let mut cols: Vec<Vec<f64>> = vec![vec![SYSMISS; ds.n_obs]; names.len()];
            for (k, &obs) in rows.iter().enumerate() {
                let (bid, left) = if k < split { (ids_l[k], true) } else { (ids_r[k - split], false) };
                let id = if left { bid as i64 - j_l as i64 - 1 } else { bid as i64 };
                let Some(b) = bins.iter().find(|b| b.id == id) else { continue };
                let (se, lo, hi) = ci_of(b);
                let vals = [id as f64, b.n as f64, b.lo, b.hi, (b.lo + b.hi) / 2.0, b.mean_x, b.mean_y, se, lo, hi, poly(if left { &coef_l } else { &coef_r }, xs[k])];
                for (cv, v) in cols.iter_mut().zip(vals) {
                    cv[obs] = v;
                }
            }
            for (nm, cv) in names.iter().zip(cols) {
                ds.add_variable(util::new_double(nm, cv))?;
            }
            st.data_changed = true;
        }
        // ---- graph ----------------------------------------------------------
        if !hide {
            let gray = Some((161u8, 161u8, 161u8));
            let black = Some((0u8, 0u8, 0u8));
            let mut layers = Vec::new();
            let pts: Vec<(f64, f64)> = bins.iter().map(|b| ((b.lo + b.hi) / 2.0, b.mean_y)).collect();
            let gl = rangen(c - h_l, c, 500);
            let gr = rangen(c, c + h_r, 500);
            let line_l: Vec<(f64, f64)> = gl.iter().map(|&x| (x, poly(&coef_l, x))).collect();
            let line_r: Vec<(f64, f64)> = gr.iter().map(|&x| (x, poly(&coef_r, x))).collect();
            let mut first = 1;
            if ci > 0.0 {
                let ranges: Vec<(f64, f64, f64)> = bins.iter().filter_map(|b| {
                    let (_, lo, hi) = ci_of(b);
                    (!is_missing(lo)).then_some(((b.lo + b.hi) / 2.0, lo, hi))
                }).collect();
                if shade {
                    let (rl, rr): (Vec<_>, Vec<_>) = ranges.into_iter().partition(|r| r.0 < c);
                    for r in [rl, rr] {
                        layers.push(Layer { kind: LayerKind::Range(RangeStyle::Area), ranges: r, color: Some((200, 200, 200)), alpha: 0.5, ..Default::default() });
                        first += 1;
                    }
                } else {
                    layers.push(Layer { kind: LayerKind::Range(RangeStyle::Cap), ranges, color: Some((180, 180, 180)), ..Default::default() });
                    first += 1;
                }
            }
            layers.push(Layer { kind: LayerKind::Scatter, points: pts, color: gray, label: "Sample average within bin".into(), ..Default::default() });
            layers.push(Layer { kind: LayerKind::Line, points: line_l, color: black, label: format!("Polynomial fit of order {p}"), ..Default::default() });
            layers.push(Layer { kind: LayerKind::Line, points: line_r, color: black, ..Default::default() });
            let ylab = ds.get_var(&yname).map(|v| if v.label.is_empty() { yname.clone() } else { v.label.clone() }).unwrap_or(yname.clone());
            let xlab = ds.get_var(&xname).map(|v| if v.label.is_empty() { xname.clone() } else { v.label.clone() }).unwrap_or(xname.clone());
            let mut g = Graph {
                title: "Regression function fit".into(),
                xtitle: xlab,
                ytitle: ylab,
                layers,
                xlines: vec![c],
                legend: true,
                legend_order: vec![(first, "Sample average within bin".into()), (first + 1, format!("Polynomial fit of order {p}"))],
                ..Default::default()
            };
            let gcmd = match cmd.opt("graph_options", 7) {
                Some(go) => parse_line(&format!("rdplot , {go}")).map_err(|e| rc_err(198, e.message))?,
                None => cmd.clone(),
            };
            let mut gcmd = gcmd;
            gcmd.raw = cmd.raw.clone();
            graph::finish(&mut g, &gcmd, st)?;
            if g.xlines != vec![c] {
                let mut xl = vec![c];
                xl.extend(g.xlines.iter().copied().filter(|v| *v != c));
                g.xlines = xl;
                st.graphs.insert(g.name.clone(), g.clone());
                st.graph = Some(g);
            }
        }
        // ---- e() ------------------------------------------------------------
        let rn: Vec<String> = (0..=p).map(|i| format!("r{}", i + 1)).collect();
        let eq = |coef: &[f64]| -> String {
            coef.iter().enumerate().map(|(j, b)| format!("{}*(x-{})^{}", crate::format::fmt_g(*b, 9).trim(), crate::format::fmt_g(c, 9).trim(), j)).collect::<Vec<_>>().join(" + ")
        };
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("N_l", n_l as f64);
        e.set("N_r", n_r as f64);
        e.set("N_h_l", n_h_l as f64);
        e.set("N_h_r", n_h_r as f64);
        e.set("c", c);
        e.set("p", p as f64);
        e.set("J_star_l", j_l as f64);
        e.set("J_star_r", j_r as f64);
        e.set_mat("coef_l", Matrix::col_vector(&coef_l).with_names(rn.clone(), vec!["c1".into()]));
        e.set_mat("coef_r", Matrix::col_vector(&coef_r).with_names(rn, vec!["c1".into()]));
        e.set_str("binselect", binselect.clone());
        e.set_str("depvar", yname.clone());
        e.set_str("runningvar", xname.clone());
        e.set_str("cmdline", crate::rdrobust::cmdline("rdplot", cmd));
        e.set_str("title", "RD plot");
        e.set_str("cmd", "rdplot");
        e.set_str("eq_l", format!("y = {}", eq(&coef_l)));
        e.set_str("eq_r", format!("y = {}", eq(&coef_r)));
        Ok(text)
    }
}
