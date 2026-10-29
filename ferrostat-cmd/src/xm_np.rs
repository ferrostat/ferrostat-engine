//! Nonparametric regression: `npregress kernel` (local-linear or
//! local-constant with cross-validated bandwidths, bootstrap standard
//! errors) and `npregress series` (cubic B-spline, spline or polynomial
//! bases with the number of terms chosen by cross-validation).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("npregress", Box::new(Npregress))]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    if ecmd != "npregress" {
        return None;
    }
    Some(np_predict(ds, st, cmd))
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

struct NpData {
    dep: String,
    y: Vec<f64>,
    /// continuous covariates (columns over the sample)
    xc: Vec<Vec<f64>>,
    cnames: Vec<String>,
    /// discrete covariates
    xd: Vec<Vec<f64>>,
    dnames: Vec<String>,
    rows: Vec<usize>,
    mask: Vec<bool>,
}

fn np_data(ds: &DataSet, st: &Store, cmd: &Command, text: &str) -> Result<NpData, String> {
    let (dep, rest) = xm::first_word(text);
    let dep = xm::var_name(ds, &dep)?;
    let mut cnames = Vec::new();
    let mut dnames = Vec::new();
    for tok in rest.split_whitespace() {
        if let Some(v) = tok.strip_prefix("i.") {
            dnames.extend(crate::varlist::expand(ds, v)?);
        } else {
            cnames.extend(crate::varlist::expand(ds, tok.strip_prefix("c.").unwrap_or(tok))?);
        }
    }
    if cnames.is_empty() && dnames.is_empty() {
        return Err(rc_err(102, "too few variables specified"));
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let mut all = vec![dep.clone()];
    all.extend(cnames.iter().cloned());
    all.extend(dnames.iter().cloned());
    util::markout(ds, &mut mask, &all);
    let rows = util::rows(&mask);
    let pick = |v: &str| -> Result<Vec<f64>, String> {
        let c = util::col(ds, v)?;
        Ok(rows.iter().map(|&i| c[i]).collect())
    };
    let y = pick(&dep)?;
    let xc = cnames.iter().map(|v| pick(v)).collect::<Result<Vec<_>, _>>()?;
    let xd = dnames.iter().map(|v| pick(v)).collect::<Result<Vec<_>, _>>()?;
    Ok(NpData { dep, y, xc, cnames, xd, dnames, rows, mask })
}

// ---------------------------------------------------------------------------
// kernel regression
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct KSpec {
    kernel: String,
    linear: bool,
    h: Vec<f64>,
    lam: Vec<f64>,
}

/// Local fit at point (pc, pd) using observations `idx` (skipping `skip`):
/// returns (mean, gradient over continuous covariates, effective count).
fn local_fit(xc: &[Vec<f64>], xd: &[Vec<f64>], y: &[f64], idx: &[usize], pc: &[f64], pd: &[f64], spec: &KSpec, skip: Option<usize>) -> Option<(f64, Vec<f64>, usize)> {
    let kc = xc.len();
    let p = if spec.linear { kc + 1 } else { 1 };
    let mut a = vec![0.0; p * p];
    let mut r = vec![0.0; p];
    let mut cnt = 0;
    let mut z = vec![0.0; p];
    for &i in idx {
        if Some(i) == skip {
            continue;
        }
        let mut w = 1.0;
        for j in 0..kc {
            let u = (xc[j][i] - pc[j]) / spec.h[j];
            w *= crate::kdensity::kernel(&spec.kernel, u);
            if w == 0.0 {
                break;
            }
        }
        if w == 0.0 {
            continue;
        }
        for j in 0..xd.len() {
            if xd[j][i] != pd[j] {
                w *= spec.lam[j];
            }
        }
        if w == 0.0 {
            continue;
        }
        cnt += 1;
        z[0] = 1.0;
        if spec.linear {
            for j in 0..kc {
                z[j + 1] = xc[j][i] - pc[j];
            }
        }
        for s in 0..p {
            r[s] += w * z[s] * y[i];
            for t in 0..p {
                a[s * p + t] += w * z[s] * z[t];
            }
        }
    }
    if cnt == 0 {
        return None;
    }
    let m = Matrix::from_rows(&(0..p).map(|s| a[s * p..(s + 1) * p].to_vec()).collect::<Vec<_>>());
    let (inv, dropped) = m.sym_inverse(1e-12);
    if !dropped.is_empty() {
        return None;
    }
    let g: Vec<f64> = (0..p).map(|s| (0..p).map(|t| inv[(s, t)] * r[t]).sum()).collect();
    Some((g[0], g[1..].to_vec(), cnt))
}

fn cv_crit(d: &NpData, idx: &[usize], spec: &KSpec) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for &i in idx {
        let pc: Vec<f64> = d.xc.iter().map(|c| c[i]).collect();
        let pd: Vec<f64> = d.xd.iter().map(|c| c[i]).collect();
        match local_fit(&d.xc, &d.xd, &d.y, idx, &pc, &pd, spec, Some(i)) {
            Some((m, _, _)) => {
                s += (d.y[i] - m).powi(2);
                n += 1.0;
            }
            None => return f64::INFINITY,
        }
    }
    if n == 0.0 { f64::INFINITY } else { s / n }
}

/// Nelder-Mead minimisation.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, iters: usize, log: &mut Vec<f64>) -> Vec<f64> {
    let n = x0.len();
    let mut pts: Vec<(Vec<f64>, f64)> = Vec::new();
    pts.push((x0.to_vec(), f(x0)));
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        let v = f(&x);
        pts.push((x, v));
    }
    for _ in 0..iters {
        pts.sort_by(|a, b| a.1.total_cmp(&b.1));
        log.push(pts[0].1);
        let spread = (pts[n].1 - pts[0].1).abs();
        if spread < 1e-10 * (1.0 + pts[0].1.abs()) && iters > 0 {
            let size = pts.iter().map(|p| p.0.iter().zip(&pts[0].0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            if size < 1e-6 {
                break;
            }
        }
        let cen: Vec<f64> = (0..n).map(|j| pts[..n].iter().map(|p| p.0[j]).sum::<f64>() / n as f64).collect();
        let worst = pts[n].clone();
        let refl: Vec<f64> = (0..n).map(|j| cen[j] + (cen[j] - worst.0[j])).collect();
        let fr = f(&refl);
        if fr < pts[0].1 {
            let exp: Vec<f64> = (0..n).map(|j| cen[j] + 2.0 * (cen[j] - worst.0[j])).collect();
            let fe = f(&exp);
            pts[n] = if fe < fr { (exp, fe) } else { (refl, fr) };
        } else if fr < pts[n - 1].1 {
            pts[n] = (refl, fr);
        } else {
            let con: Vec<f64> = (0..n).map(|j| cen[j] + 0.5 * (worst.0[j] - cen[j])).collect();
            let fc = f(&con);
            if fc < worst.1 {
                pts[n] = (con, fc);
            } else {
                let best = pts[0].0.clone();
                for p in pts.iter_mut().skip(1) {
                    let x: Vec<f64> = (0..n).map(|j| best[j] + 0.5 * (p.0[j] - best[j])).collect();
                    let v = f(&x);
                    *p = (x, v);
                }
            }
        }
    }
    pts.sort_by(|a, b| a.1.total_cmp(&b.1));
    pts[0].0.clone()
}

/// Golden-section search on [a, b].
fn golden(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, log: &mut Vec<f64>) -> f64 {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        log.push(fc.min(fd));
        if (b - a).abs() < 1e-7 * (1.0 + a.abs()) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// Fitted means and derivatives at every sample observation.
fn fit_all(d: &NpData, idx: &[usize], spec: &KSpec, dspec: &KSpec) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n = d.y.len();
    let mut m = vec![SYSMISS; n];
    let mut g = vec![vec![SYSMISS; n]; d.xc.len()];
    let mut cnt = 0.0;
    for &i in idx {
        let pc: Vec<f64> = d.xc.iter().map(|c| c[i]).collect();
        let pd: Vec<f64> = d.xd.iter().map(|c| c[i]).collect();
        if let Some((mm, _, c)) = local_fit(&d.xc, &d.xd, &d.y, idx, &pc, &pd, spec, None) {
            m[i] = mm;
            cnt += c as f64;
        }
        if spec.linear && !d.xc.is_empty() {
            if let Some((_, gg, _)) = local_fit(&d.xc, &d.xd, &d.y, idx, &pc, &pd, dspec, None) {
                for j in 0..gg.len() {
                    g[j][i] = gg[j];
                }
            }
        }
    }
    (m, g, cnt / idx.len().max(1) as f64)
}

/// Averages: mean prediction, average derivatives (continuous) and
/// average contrasts against the base level (discrete).
fn averages(d: &NpData, idx: &[usize], spec: &KSpec, dspec: &KSpec, m: &[f64], g: &[Vec<f64>]) -> Vec<f64> {
    let nn = idx.len() as f64;
    let mut out = vec![idx.iter().map(|&i| m[i]).filter(|v| !is_missing(*v)).sum::<f64>() / nn];
    for gj in g {
        let v: Vec<f64> = idx.iter().map(|&i| gj[i]).filter(|v| !is_missing(*v)).collect();
        out.push(v.iter().sum::<f64>() / v.len().max(1) as f64);
    }
    for (j, xdj) in d.xd.iter().enumerate() {
        let mut lv: Vec<f64> = idx.iter().map(|&i| xdj[i]).collect();
        lv.sort_by(f64::total_cmp);
        lv.dedup();
        let at = |level: f64| -> f64 {
            let mut s = 0.0;
            let mut c = 0.0;
            for &i in idx {
                let pc: Vec<f64> = d.xc.iter().map(|c| c[i]).collect();
                let mut pd: Vec<f64> = d.xd.iter().map(|c| c[i]).collect();
                pd[j] = level;
                if let Some((mm, _, _)) = local_fit(&d.xc, &d.xd, &d.y, idx, &pc, &pd, spec, None) {
                    s += mm;
                    c += 1.0;
                }
            }
            s / c
        };
        let base = at(lv[0]);
        for &l in &lv[1..] {
            out.push(at(l) - base);
        }
        let _ = dspec;
    }
    out
}

fn kernel_name(cmd: &Command) -> String {
    crate::kdensity::canonical_kernel(&cmd.opt("kernel", 3).unwrap_or("epanechnikov").trim().to_lowercase()).to_string()
}

fn kernel_reg(ds: &mut DataSet, st: &mut Store, cmd: &Command, text: &str) -> Result<String, String> {
    let d = np_data(ds, st, cmd, text)?;
    let n = d.y.len();
    let idx: Vec<usize> = (0..n).collect();
    let linear = !cmd.opt("estimator", 3).is_some_and(|e| e.trim().to_lowercase().starts_with("c"));
    let kernel = kernel_name(cmd);
    let kc = d.xc.len();
    let kd = d.xd.len();
    let mut out = vec!["Computing mean function".to_string(), String::new()];
    // bandwidths: given or cross-validated
    let given: Vec<f64> = cmd.opt("bwidth", 2).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let sd: Vec<f64> = d.xc.iter().map(|c| util::variance(c).sqrt()).collect();
    let h0: Vec<f64> = sd.iter().map(|s| 1.06 * s * (n as f64).powf(-0.2)).collect();
    let mk = |th: &[f64]| -> KSpec {
        KSpec { kernel: kernel.clone(), linear, h: (0..kc).map(|j| th[j].exp()).collect(), lam: (0..kd).map(|j| dist::invlogit(th[kc + j])).collect() }
    };
    let mut log = Vec::new();
    let spec = if given.len() >= kc + kd && !given.is_empty() {
        KSpec { kernel: kernel.clone(), linear, h: given[..kc].to_vec(), lam: given[kc..kc + kd].to_vec() }
    } else {
        let f = |th: &[f64]| cv_crit(&d, &idx, &mk(th));
        let th = if kc == 1 && kd == 0 {
            let fl = |t: f64| f(&[t]);
            vec![golden(&fl, h0[0].ln() - 3.0, h0[0].ln() + 3.0, &mut log)]
        } else {
            let mut x0: Vec<f64> = h0.iter().map(|h| h.ln()).collect();
            x0.extend(vec![-1.0; kd]);
            nelder_mead(&f, &x0, 0.5, 400, &mut log)
        };
        mk(&th)
    };
    log.dedup();
    for (i, v) in log.iter().take(30).enumerate() {
        out.push(format!("Iteration {i}:   Cross-validation criterion = {}", fmt_g(*v, 9)));
    }
    out.push(String::new());
    let dspec = if given.len() >= 2 * (kc + kd) && kc > 0 {
        KSpec { kernel: kernel.clone(), linear, h: given[kc + kd..2 * kc + kd].to_vec(), lam: given[2 * kc + kd..].to_vec() }
    } else {
        spec.clone()
    };
    let (m, g, ekobs) = fit_all(&d, &idx, &spec, &dspec);
    let est = averages(&d, &idx, &spec, &dspec, &m, &g);
    let ybar = util::mean(&d.y);
    let rss: f64 = (0..n).filter(|&i| !is_missing(m[i])).map(|i| (d.y[i] - m[i]).powi(2)).sum();
    let tss: f64 = d.y.iter().map(|y| (y - ybar).powi(2)).sum();
    let r2 = 1.0 - rss / tss;
    // names
    let mut names = vec![format!("Mean:{}", d.dep)];
    for c in &d.cnames {
        names.push(format!("Effect:{c}"));
    }
    for (j, dn) in d.dnames.iter().enumerate() {
        let mut lv: Vec<f64> = d.xd[j].clone();
        lv.sort_by(f64::total_cmp);
        lv.dedup();
        for l in &lv[1..] {
            names.push(format!("Effect:{}.{dn}", fmt_g(*l, 12)));
        }
    }
    if !linear {
        // local-constant: derivatives are not reported for continuous covariates
        names.retain(|nm| !(nm.starts_with("Effect:") && d.cnames.iter().any(|c| nm == &format!("Effect:{c}"))));
    }
    let est: Vec<f64> = if linear { est } else { std::iter::once(est[0]).chain(est[1 + kc..].iter().cloned()).collect() };
    let k = est.len();
    // bootstrap
    let reps = util::opt_num(cmd, "reps", 3)?.map(|r| r as usize).or_else(|| {
        cmd.opt("vce", 3).and_then(|v| if v.to_lowercase().starts_with("boot") { Some(v.split(|c: char| !c.is_ascii_digit()).filter_map(|t| t.parse().ok()).next().unwrap_or(50)) } else { None })
    });
    let level = util::level(cmd, st)?;
    let mut v = Matrix::zeros(k, k);
    let mut ci: Option<Vec<(f64, f64)>> = None;
    if let Some(r) = reps {
        if let Some(s) = util::opt_num(cmd, "seed", 4)? {
            st.rng.borrow_mut().reseed(s as u64);
        }
        out.push("(running npregress on estimation sample)".into());
        out.push(String::new());
        let mut draws: Vec<Vec<f64>> = Vec::new();
        for _ in 0..r {
            let bidx: Vec<usize> = {
                let mut rng = st.rng.borrow_mut();
                (0..n).map(|_| ((rng.uniform() * n as f64) as usize).min(n - 1)).collect()
            };
            let (mb, gb, _) = fit_all(&d, &bidx, &spec, &dspec);
            let mut e = averages(&d, &bidx, &spec, &dspec, &mb, &gb);
            if !linear {
                e = std::iter::once(e[0]).chain(e[1 + kc..].iter().cloned()).collect();
            }
            draws.push(e);
        }
        out.push(format!("Bootstrap replications ({r}): done"));
        out.push(String::new());
        let means: Vec<f64> = (0..k).map(|j| draws.iter().map(|e| e[j]).sum::<f64>() / r as f64).collect();
        for a in 0..k {
            for c in 0..k {
                v[(a, c)] = draws.iter().map(|e| (e[a] - means[a]) * (e[c] - means[c])).sum::<f64>() / (r as f64 - 1.0);
            }
        }
        let alpha = (100.0 - level) / 200.0;
        ci = Some(
            (0..k)
                .map(|j| {
                    let mut s: Vec<f64> = draws.iter().map(|e| e[j]).collect();
                    s.sort_by(f64::total_cmp);
                    (util::pctile_sorted(&s, alpha * 100.0), util::pctile_sorted(&s, (1.0 - alpha) * 100.0))
                })
                .collect(),
        );
    }
    // output
    out.push(format!("{:>12}  {:>10} {:>10}", "", "Bandwidth", ""));
    out.push(format!("{:>12}  {:>10} {:>10}", "", "Mean", if linear { "Effect" } else { "" }));
    for (j, c) in d.cnames.iter().enumerate() {
        out.push(format!("{:>12}  {:>10} {:>10}", crate::summarize::abbrev(c, 12), fmt_g(spec.h[j], 7), if linear { fmt_g(dspec.h[j], 7) } else { String::new() }));
    }
    for (j, c) in d.dnames.iter().enumerate() {
        out.push(format!("{:>12}  {:>10} {:>10}", crate::summarize::abbrev(c, 12), fmt_g(spec.lam[j], 7), if linear { fmt_g(dspec.lam[j], 7) } else { String::new() }));
    }
    out.push(String::new());
    let title = if linear { "Local-linear regression" } else { "Local-constant regression" };
    out.push(crate::choice_mixed::hrow(title, "Number of obs", util::commas(n)));
    out.push(crate::choice_mixed::hrow(&format!("Kernel   : {kernel}"), "E(Kernel obs)", util::commas(ekobs.round() as usize)));
    out.push(crate::choice_mixed::hrow(&format!("Bandwidth: {}", if given.is_empty() { "cross-validation" } else { "user specified" }), "R-squared", xm::f4(r2)));
    out.push(String::new());
    let lw = 12;
    let bar = "-".repeat(lw + 66);
    out.push(bar.clone());
    if let Some(ci) = &ci {
        out.push(format!("{:>lw$} |   Observed   Bootstrap                         Percentile", ""));
        out.push(format!("{:>lw$} |   estimate   std. err.      z    P>|z|     [{}% conf. interval]", crate::summarize::abbrev(&d.dep, lw), fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let mut last = String::new();
        for (j, nm) in names.iter().enumerate() {
            let (eq, var) = nm.split_once(':').unwrap();
            if eq != last {
                if !last.is_empty() {
                    out.push(format!("{:>lw$} |", ""));
                }
                out.push(format!("{:<lw$} |", eq));
                last = eq.to_string();
            }
            let se = v[(j, j)].max(0.0).sqrt();
            let z = est[j] / se;
            out.push(format!("{:>lw$} |  {:>9}  {:>9}  {:>7.2}   {:.3}     {:>9}   {:>9}", crate::summarize::abbrev(var, lw), fmt_g(est[j], 7), fmt_g(se, 7), z, 2.0 * dist::normal(-z.abs()), fmt_g(ci[j].0, 7), fmt_g(ci[j].1, 7)));
        }
    } else {
        out.push(format!("{:>lw$} |   Estimate", crate::summarize::abbrev(&d.dep, lw)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let mut last = String::new();
        for (j, nm) in names.iter().enumerate() {
            let (eq, var) = nm.split_once(':').unwrap();
            if eq != last {
                if !last.is_empty() {
                    out.push(format!("{:>lw$} |", ""));
                }
                out.push(format!("{:<lw$} |", eq));
                last = eq.to_string();
            }
            out.push(format!("{:>lw$} |  {:>9}", crate::summarize::abbrev(var, lw), fmt_g(est[j], 7)));
        }
    }
    out.push(bar);
    if linear && kc > 0 {
        out.push("Note: Effect estimates are averages of derivatives.".into());
    }
    if ci.is_none() {
        out.push("Note: You may compute standard errors using vce(bootstrap) or reps().".into());
    }
    // predict(mean deriv) at estimation
    if let Some(p) = cmd.opt("predict", 4) {
        let p = p.split(',').next().unwrap_or("").to_string();
        let words: Vec<String> = p.split_whitespace().map(String::from).collect();
        let mut newvars: Vec<(String, Vec<f64>)> = Vec::new();
        let full = |vals: &[f64]| -> Vec<f64> {
            let mut o = vec![SYSMISS; ds.n_obs];
            for (t, &r) in d.rows.iter().enumerate() {
                o[r] = vals[t];
            }
            o
        };
        if words.len() == 1 && words[0].ends_with('*') {
            let s = words[0].trim_end_matches('*');
            newvars.push((format!("{s}1"), full(&m)));
            for (j, gj) in g.iter().enumerate() {
                newvars.push((format!("{s}{}", j + 2), full(gj)));
            }
        } else {
            if let Some(w) = words.first() {
                newvars.push((w.clone(), full(&m)));
            }
            for (j, w) in words.iter().skip(1).enumerate() {
                if j < g.len() {
                    newvars.push((w.clone(), full(&g[j])));
                }
            }
        }
        for (nm, vals) in newvars {
            crate::varlist::check_name(&nm)?;
            ds.add_variable(util::new_double(&nm, vals)).map_err(|e| rc_err(110, e))?;
        }
    }
    let text = out.join("\n");
    let mut hm = Matrix::zeros(1, kc + kd);
    for j in 0..kc {
        hm[(0, j)] = spec.h[j];
    }
    for j in 0..kd {
        hm[(0, kc + j)] = spec.lam[j];
    }
    let ynames = names.clone();
    let mut bw_names = d.cnames.clone();
    bw_names.extend(d.dnames.iter().cloned());
    let e = st.begin_e();
    e.set("N", n as f64);
    e.set("r2", r2);
    e.set("mean", est[0]);
    if reps.is_some() {
        e.set("N_reps", reps.unwrap() as f64);
    }
    e.set_str("cmd", "npregress");
    e.set_str("subcmd", "kernel");
    e.set_str("depvar", d.dep.clone());
    e.set_str("kname", kernel.clone());
    e.set_str("estimator", if linear { "linear" } else { "constant" });
    e.set_str("cvars", d.cnames.join(" "));
    e.set_str("dvars", d.dnames.join(" "));
    e.set_str("title", title);
    e.set_str("properties", "b V");
    e.set_mat("b", Matrix::row_vector(&est).with_names(vec!["y1".into()], ynames.clone()));
    e.set_mat("V", v.with_names(ynames.clone(), ynames));
    e.set_mat("bwidth", hm.with_names(vec!["Mean".into()], bw_names));
    e.sample = Some(d.mask.clone());
    e.set_str("_output", text.clone());
    Ok(text)
}

// ---------------------------------------------------------------------------
// series regression
// ---------------------------------------------------------------------------

/// Cubic (degree 3) B-spline basis on [0,1] with equally spaced interior knots.
fn bspline(x: f64, nk: usize, deg: usize) -> Vec<f64> {
    let mut t = vec![0.0; deg + 1];
    for i in 1..=nk {
        t.push(i as f64 / (nk + 1) as f64);
    }
    t.extend(vec![1.0; deg + 1]);
    let nb = t.len() - deg - 1;
    let x = x.clamp(0.0, 1.0);
    // degree 0
    let mut b: Vec<f64> = (0..t.len() - 1).map(|i| if (t[i] <= x && x < t[i + 1]) || (x == 1.0 && t[i] < 1.0 && t[i + 1] == 1.0) { 1.0 } else { 0.0 }).collect();
    for p in 1..=deg {
        let mut nbv = vec![0.0; t.len() - 1 - p];
        for i in 0..nbv.len() {
            let a = if t[i + p] > t[i] { (x - t[i]) / (t[i + p] - t[i]) * b[i] } else { 0.0 };
            let c = if t[i + p + 1] > t[i + 1] { (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]) * b[i + 1] } else { 0.0 };
            nbv[i] = a + c;
        }
        b = nbv;
    }
    b.truncate(nb);
    b
}

/// Basis for one rescaled continuous covariate.
fn basis(kind: &str, x: f64, nk: usize) -> Vec<f64> {
    match kind {
        "polynomial" => (1..=nk).map(|p| x.powi(p as i32)).collect(),
        "spline" => {
            let mut v = vec![x, x * x, x * x * x];
            for i in 1..=nk {
                let k = i as f64 / (nk + 1) as f64;
                v.push((x - k).max(0.0).powi(3));
            }
            v
        }
        _ => bspline(x, nk, 3),
    }
}

/// Design rows: basis of each continuous covariate (with interactions
/// with the discrete cells) plus cell indicators.
fn series_row(kind: &str, xs: &[f64], cells: usize, cell: usize, nk: usize) -> Vec<f64> {
    let mut b: Vec<f64> = Vec::new();
    let bspl = kind == "bspline";
    for (j, &x) in xs.iter().enumerate() {
        let mut bj = basis(kind, x, nk);
        if bspl && j > 0 {
            bj.remove(0);
        }
        b.extend(bj);
    }
    // full interaction of the basis with the discrete cells
    let mut row = Vec::with_capacity(b.len() * cells + cells);
    for c in 0..cells {
        let on = if c == cell { 1.0 } else { 0.0 };
        if !bspl {
            row.push(on);
        }
        for v in &b {
            row.push(on * v);
        }
    }
    row
}

fn ols_fit(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, Matrix, Vec<f64>)> {
    let n = y.len();
    let k = x[0].len();
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
    let (xi, _) = xtx.sym_inverse(1e-12);
    let b: Vec<f64> = (0..k).map(|a| (0..k).map(|c| xi[(a, c)] * xty[c]).sum()).collect();
    let e: Vec<f64> = (0..n).map(|t| y[t] - (0..k).map(|a| x[t][a] * b[a]).sum::<f64>()).collect();
    Some((b, xi, e))
}

fn series_reg(ds: &mut DataSet, st: &mut Store, cmd: &Command, text: &str) -> Result<String, String> {
    let d = np_data(ds, st, cmd, text)?;
    let n = d.y.len();
    let kind = if cmd.has_opt("polynomial", 4) {
        "polynomial"
    } else if cmd.has_opt("spline", 3) {
        "spline"
    } else {
        "bspline"
    };
    let lo: Vec<f64> = d.xc.iter().map(|c| c.iter().cloned().fold(f64::INFINITY, f64::min)).collect();
    let hi: Vec<f64> = d.xc.iter().map(|c| c.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect();
    let xs: Vec<Vec<f64>> = (0..n).map(|t| (0..d.xc.len()).map(|j| (d.xc[j][t] - lo[j]) / (hi[j] - lo[j])).collect()).collect();
    // discrete cells
    let mut keys: Vec<Vec<u64>> = (0..n).map(|t| d.xd.iter().map(|c| c[t].to_bits()).collect()).collect();
    let mut uk = keys.clone();
    uk.sort();
    uk.dedup();
    let cells = uk.len();
    let cell_of = |key: &Vec<u64>| uk.iter().position(|k| k == key).unwrap_or(0);
    let cell: Vec<usize> = keys.iter().map(&cell_of).collect();
    keys.clear();
    let build = |nk: usize| -> Vec<Vec<f64>> { (0..n).map(|t| series_row(kind, &xs[t], cells, cell[t], nk)).collect() };
    let cv = |nk: usize| -> f64 {
        let x = build(nk);
        let Some((_, xi, e)) = ols_fit(&x, &d.y) else { return f64::INFINITY };
        let k = x[0].len();
        (0..n)
            .map(|t| {
                let h: f64 = (0..k).map(|a| (0..k).map(|c| x[t][a] * xi[(a, c)] * x[t][c]).sum::<f64>()).sum();
                (e[t] / (1.0 - h)).powi(2)
            })
            .sum::<f64>()
            / n as f64
    };
    let mut out = vec!["Computing approximating function".to_string(), String::new(), "Minimizing cross-validation criterion".to_string(), String::new()];
    let nk = match util::opt_num(cmd, "knots", 2)?.or(util::opt_num(cmd, "polynomial", 4)?).or(util::opt_num(cmd, "spline", 3)?).or(util::opt_num(cmd, "bspline", 3)?) {
        Some(k) => k as usize,
        None => {
            let start = if kind == "polynomial" { 1 } else { 1 };
            let mut best = (start, cv(start));
            out.push(format!("Iteration 0:   Cross-validation criterion = {}", fmt_g(best.1, 9)));
            let mut it = 1;
            for k in start + 1..start + 20 {
                if xs.is_empty() || k * cells + 4 >= n / 2 {
                    break;
                }
                let c = cv(k);
                out.push(format!("Iteration {it}:   Cross-validation criterion = {}", fmt_g(c, 9)));
                it += 1;
                if c < best.1 {
                    best = (k, c);
                } else {
                    break;
                }
            }
            best.0
        }
    };
    out.push(String::new());
    out.push("Computing average derivatives".into());
    out.push(String::new());
    let x = build(nk);
    let (b, xi, e) = ols_fit(&x, &d.y).ok_or_else(|| rc_err(430, "could not fit series"))?;
    let k = x[0].len();
    // robust (HC1) variance of the series coefficients
    let mut meat = Matrix::zeros(k, k);
    for t in 0..n {
        for a in 0..k {
            for c in 0..k {
                meat[(a, c)] += x[t][a] * x[t][c] * e[t] * e[t];
            }
        }
    }
    let vb = xi.mul(&meat)?.mul(&xi)?.scale(n as f64 / (n - k) as f64);
    // effects as linear functions a'b of the coefficients
    let mut names = Vec::new();
    let mut avec: Vec<Vec<f64>> = Vec::new();
    for j in 0..d.xc.len() {
        let mut a = vec![0.0; k];
        for t in 0..n {
            let h = 1e-6;
            let mut up = xs[t].clone();
            up[j] += h;
            let mut dn = xs[t].clone();
            dn[j] -= h;
            let ru = series_row(kind, &up, cells, cell[t], nk);
            let rd = series_row(kind, &dn, cells, cell[t], nk);
            for c in 0..k {
                a[c] += (ru[c] - rd[c]) / (2.0 * h) / (hi[j] - lo[j]) / n as f64;
            }
        }
        names.push(d.cnames[j].clone());
        avec.push(a);
    }
    for (j, dn) in d.dnames.iter().enumerate() {
        let mut lv: Vec<f64> = d.xd[j].clone();
        lv.sort_by(f64::total_cmp);
        lv.dedup();
        let row_at = |t: usize, level: f64| -> Vec<f64> {
            let mut key: Vec<u64> = d.xd.iter().map(|c| c[t].to_bits()).collect();
            key[j] = level.to_bits();
            let c = uk.iter().position(|k| *k == key);
            match c {
                Some(c) => series_row(kind, &xs[t], cells, c, nk),
                None => vec![f64::NAN; k],
            }
        };
        for &l in &lv[1..] {
            let mut a = vec![0.0; k];
            let mut cnt = 0.0;
            for t in 0..n {
                let r1 = row_at(t, l);
                let r0 = row_at(t, lv[0]);
                if r1.iter().chain(&r0).any(|v| v.is_nan()) {
                    continue;
                }
                cnt += 1.0;
                for c in 0..k {
                    a[c] += r1[c] - r0[c];
                }
            }
            names.push(format!("{}.{dn}", fmt_g(l, 12)));
            avec.push(a.iter().map(|v| v / cnt).collect());
        }
    }
    let ne = names.len();
    let est: Vec<f64> = avec.iter().map(|a| a.iter().zip(&b).map(|(x, y)| x * y).sum()).collect();
    let mut v = Matrix::zeros(ne, ne);
    for p in 0..ne {
        for q in 0..ne {
            v[(p, q)] = (0..k).map(|a| (0..k).map(|c| avec[p][a] * vb[(a, c)] * avec[q][c]).sum::<f64>()).sum();
        }
    }
    let title = match kind {
        "polynomial" => "Polynomial estimation",
        "spline" => "Cubic spline estimation",
        _ => "Cubic B-spline estimation",
    };
    out.push(crate::choice_mixed::hrow(title, "Number of obs", util::commas(n)));
    out.push(crate::choice_mixed::hrow("Criterion: cross-validation", if kind == "polynomial" { "Polynomial order" } else { "Number of knots" }, nk.to_string()));
    out.push(String::new());
    let level = util::level(cmd, st)?;
    let no = vec![false; ne];
    let mut tbl = crate::est::coef_table(&crate::est::TableSpec { depvar: &d.dep, names: &names, b: &est, v: &v, omitted: &no, is_base: &no, df: None, level, eform: None, coef_title: "Effect" });
    if tbl.len() > 1 {
        let w = tbl[0].len().saturating_sub(66);
        tbl.insert(1, format!("{:>w$} |               Robust", ""));
    }
    out.extend(tbl);
    out.push("Note: Effect estimates are averages of derivatives.".into());
    let text = out.join("\n");
    let ynames = names.clone();
    let e = st.begin_e();
    e.set("N", n as f64);
    e.set("n_knots", nk as f64);
    e.set_str("cmd", "npregress");
    e.set_str("subcmd", "series");
    e.set_str("basis", kind);
    e.set_str("depvar", d.dep.clone());
    e.set_str("title", title);
    e.set_str("vcetype", "Robust");
    e.set_str("properties", "b V");
    e.set_mat("b", Matrix::row_vector(&est).with_names(vec!["y1".into()], ynames.clone()));
    e.set_mat("V", v.with_names(ynames.clone(), ynames));
    e.sample = Some(d.mask.clone());
    e.set_str("_output", text.clone());
    Ok(text)
}

pub struct Npregress;

impl CommandHandler for Npregress {
    fn name(&self) -> &'static str {
        "npregress"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (sub, rest) = xm::first_word(&cmd.main);
        match sub.to_lowercase().as_str() {
            "kernel" => kernel_reg(ds, st, cmd, &rest),
            "series" => series_reg(ds, st, cmd, &rest),
            _ => Err(rc_err(198, "npregress: kernel or series required")),
        }
    }
}

fn np_predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    if st.e.macros.get("subcmd").map(String::as_str) != Some("kernel") {
        return Err(rc_err(198, "predict after npregress series is not available"));
    }
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = words.last().ok_or_else(|| rc_err(198, "new variable name required"))?.to_string();
    crate::varlist::check_name(&name)?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let cvars = st.e.macros.get("cvars").cloned().unwrap_or_default();
    let dvars = st.e.macros.get("dvars").cloned().unwrap_or_default();
    let linear = st.e.macros.get("estimator").is_some_and(|e| e == "linear");
    let kernel = st.e.macros.get("kname").cloned().unwrap_or_else(|| "epanechnikov".into());
    let bw = st.e.matrices.get("bwidth").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let cn: Vec<&str> = cvars.split_whitespace().collect();
    let dn: Vec<&str> = dvars.split_whitespace().collect();
    let spec = KSpec { kernel, linear, h: (0..cn.len()).map(|j| bw.get(0, j)).collect(), lam: (0..dn.len()).map(|j| bw.get(0, cn.len() + j)).collect() };
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    let colv = |v: &str| util::col(ds, v);
    let ycol = colv(&dep)?;
    let ccols: Vec<Vec<f64>> = cn.iter().map(|v| colv(v)).collect::<Result<_, _>>()?;
    let dcols: Vec<Vec<f64>> = dn.iter().map(|v| colv(v)).collect::<Result<_, _>>()?;
    let xc: Vec<Vec<f64>> = ccols.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect();
    let xd: Vec<Vec<f64>> = dcols.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect();
    let y: Vec<f64> = rows.iter().map(|&i| ycol[i]).collect();
    let idx: Vec<usize> = (0..rows.len()).collect();
    let mask = util::touse(ds, st, cmd)?;
    let mut vals = vec![SYSMISS; ds.n_obs];
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let pc: Vec<f64> = ccols.iter().map(|c| c[i]).collect();
        let pd: Vec<f64> = dcols.iter().map(|c| c[i]).collect();
        if pc.iter().chain(&pd).any(|v| is_missing(*v)) {
            continue;
        }
        if let Some((m, _, _)) = local_fit(&xc, &xd, &y, &idx, &pc, &pd, &spec, None) {
            vals[i] = m;
        }
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    Ok(if nmiss > 0 { format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }) } else { String::new() })
}

#[allow(dead_code)]
fn _unused(_: &Results) {}
