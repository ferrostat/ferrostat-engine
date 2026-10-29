//! Shared maximum-likelihood machinery for the time-series estimators
//! (arch, mgarch, arfima, sspace, dfactor, ucm, mswitch): numerical scores
//! and Hessians of a per-observation log-likelihood, a BHHH/BFGS optimizer
//! with the usual iteration log, variance estimators, the time grid of an
//! estimation sample and a small helper to print and post the results.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::dist;
use crate::est;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::tsops;
use crate::tsreg::{f2, f4, hline, with_vcetype};
use crate::util;

pub const LN2PI: f64 = 1.837_877_066_409_345_5;

/// Per-observation log-likelihood (None when infeasible).
pub type LlFn<'a> = dyn Fn(&[f64]) -> Option<Vec<f64>> + 'a;

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn mv(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

pub fn ll_sum(f: &LlFn, th: &[f64]) -> f64 {
    if th.iter().any(|v| !v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    match f(th) {
        Some(v) => {
            let s: f64 = v.iter().sum();
            if s.is_finite() { s } else { f64::NEG_INFINITY }
        }
        None => f64::NEG_INFINITY,
    }
}

fn step(th: &[f64], j: usize) -> f64 {
    1e-6 * th[j].abs().max(1e-2)
}

/// Per-observation scores by central differences.
pub fn scores(f: &LlFn, th: &[f64]) -> Option<Vec<Vec<f64>>> {
    let k = th.len();
    let base = f(th)?;
    let mut s = vec![vec![0.0; k]; base.len()];
    for j in 0..k {
        let h = step(th, j);
        let mut tp = th.to_vec();
        tp[j] += h;
        let mut tm = th.to_vec();
        tm[j] -= h;
        let (lp, lm, denom) = match (f(&tp).filter(|v| v.len() == base.len()), f(&tm).filter(|v| v.len() == base.len())) {
            (Some(a), Some(b)) => (a, b, 2.0 * h),
            (Some(a), None) => (a, base.clone(), h),
            (None, Some(b)) => (base.clone(), b, h),
            _ => return None,
        };
        for t in 0..base.len() {
            s[t][j] = (lp[t] - lm[t]) / denom;
        }
    }
    Some(s)
}

pub fn gradient(f: &LlFn, th: &[f64]) -> Option<Vec<f64>> {
    let s = scores(f, th)?;
    Some((0..th.len()).map(|j| s.iter().map(|r| r[j]).sum()).collect())
}

/// Negative Hessian of the summed log-likelihood (second differences).
pub fn neg_hessian(f: &LlFn, th: &[f64]) -> Option<Matrix> {
    let k = th.len();
    let f0 = ll_sum(f, th);
    if !f0.is_finite() {
        return None;
    }
    let hs: Vec<f64> = (0..k).map(|j| 1e-4 * th[j].abs().max(1e-2)).collect();
    let mut h = Matrix::zeros(k, k);
    let ev = |d: &[(usize, f64)]| -> Option<f64> {
        let mut t = th.to_vec();
        for &(j, v) in d {
            t[j] += v;
        }
        let v = ll_sum(f, &t);
        if v.is_finite() { Some(v) } else { None }
    };
    for i in 0..k {
        let fp = ev(&[(i, hs[i])])?;
        let fm = ev(&[(i, -hs[i])])?;
        h[(i, i)] = -(fp - 2.0 * f0 + fm) / (hs[i] * hs[i]);
        for j in 0..i {
            let a = ev(&[(i, hs[i]), (j, hs[j])])?;
            let b = ev(&[(i, hs[i]), (j, -hs[j])])?;
            let c = ev(&[(i, -hs[i]), (j, hs[j])])?;
            let d = ev(&[(i, -hs[i]), (j, -hs[j])])?;
            let v = -(a - b - c + d) / (4.0 * hs[i] * hs[j]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    Some(h)
}

pub fn opg(s: &[Vec<f64>], k: usize) -> Matrix {
    let mut m = Matrix::zeros(k, k);
    for r in s {
        for a in 0..k {
            for b in 0..k {
                m[(a, b)] += r[a] * r[b];
            }
        }
    }
    m
}

pub struct Optimum {
    pub th: Vec<f64>,
    pub ll: f64,
    pub log: Vec<String>,
    pub iters: usize,
    pub converged: bool,
}

pub fn iter_line(i: usize, ll: f64, label: &str) -> String {
    format!("Iteration {i}:{}{label} = {}", if i < 10 { "  " } else { " " }, fmt_g(ll, 9))
}

#[derive(Clone, Copy, PartialEq)]
pub enum Tech {
    /// BHHH 5 iterations then BFGS 10, alternating (time-series default)
    BhhhBfgs,
    /// BFGS throughout
    Bfgs,
    /// Newton-Raphson with a numerical Hessian
    Nr,
}

pub fn tech_from(cmd: &Command, default: Tech) -> Tech {
    match cmd.opt("technique", 4).map(|s| s.trim().to_lowercase()) {
        Some(s) if s.starts_with("nr") => Tech::Nr,
        Some(s) if s.starts_with("bfgs") || s.starts_with("dfp") => Tech::Bfgs,
        Some(s) if s.starts_with("bhhh") => Tech::BhhhBfgs,
        _ => default,
    }
}

/// Maximize sum f(th). `refine` adds a final Newton polish.
pub fn optimize(f: &LlFn, th0: Vec<f64>, maxiter: usize, tech: Tech, label: &str, refine: bool) -> Result<Optimum, String> {
    let k = th0.len();
    let mut th = th0;
    let mut ll = ll_sum(f, &th);
    if !ll.is_finite() {
        return Err(rc_err(1400, "initial values not feasible"));
    }
    let mut log = Vec::new();
    match tech {
        Tech::BhhhBfgs => log.push("(setting optimization to BHHH)".to_string()),
        Tech::Bfgs => log.push("(setting optimization to BFGS)".to_string()),
        Tech::Nr => {}
    }
    let mut hinv = Matrix::identity(k);
    let mut prev: Option<(f64, f64)> = None;
    let mut converged = false;
    let mut it = 0;
    loop {
        let s = scores(f, &th).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"))?;
        let g: Vec<f64> = (0..k).map(|j| s.iter().map(|r| r[j]).sum()).collect();
        let o = opg(&s, k);
        let (oinv, _) = o.sym_inverse(1e-14);
        let phase_bhhh = tech == Tech::BhhhBfgs && (it % 15) < 5;
        if tech == Tech::BhhhBfgs {
            if it > 0 && it % 15 == 5 {
                log.push("(switching optimization to BFGS)".into());
                hinv = oinv.clone();
            } else if it > 0 && it % 15 == 0 {
                log.push("(switching optimization to BHHH)".into());
            }
        } else if tech == Tech::Bfgs && it == 0 {
            hinv = oinv.clone();
        }
        log.push(iter_line(it, ll, label));
        let nhi = if tech == Tech::Nr { neg_hessian(f, &th).map(|h| h.sym_inverse(1e-14).0) } else { None };
        let metric = nhi.as_ref().unwrap_or(&oinv);
        let nr = dot(&g, &mv(metric, &g)).abs();
        if let Some((dl, db)) = prev {
            if (dl < 1e-7 || db < 1e-6) && nr < 1e-5 {
                converged = true;
                break;
            }
            if dl == 0.0 && db == 0.0 {
                break;
            }
        }
        if it >= maxiter {
            break;
        }
        let dir = if let Some(h) = &nhi {
            mv(h, &g)
        } else if phase_bhhh {
            mv(&oinv, &g)
        } else {
            mv(&hinv, &g)
        };
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let cand: Vec<f64> = th.iter().zip(&dir).map(|(t, d)| t + lambda * d).collect();
            let lc = ll_sum(f, &cand);
            if lc.is_finite() && lc >= ll - 1e-12 * ll.abs() {
                accepted = Some((cand, lc));
                break;
            }
            lambda *= 0.5;
        }
        let Some((nth, nll)) = accepted else {
            if !phase_bhhh && tech != Tech::Nr {
                // reset the quasi-Newton metric and try again
                hinv = oinv.clone();
                if prev.is_none_or(|p| p.0 > 0.0) {
                    prev = Some((0.0, 1.0));
                    it += 1;
                    continue;
                }
            }
            it += 1;
            log.push(iter_line(it, ll, label));
            converged = nr < 1e-3;
            break;
        };
        if !phase_bhhh && tech != Tech::Nr {
            if let Some(gn) = gradient(f, &nth) {
                let sv: Vec<f64> = nth.iter().zip(&th).map(|(a, b)| a - b).collect();
                let yv: Vec<f64> = g.iter().zip(&gn).map(|(a, b)| a - b).collect();
                let sy = dot(&sv, &yv);
                if sy > 1e-16 {
                    let hy = mv(&hinv, &yv);
                    let yhy = dot(&yv, &hy);
                    let mut nh = hinv.clone();
                    for i in 0..k {
                        for j in 0..k {
                            nh[(i, j)] += (sy + yhy) * sv[i] * sv[j] / (sy * sy) - (hy[i] * sv[j] + sv[i] * hy[j]) / sy;
                        }
                    }
                    hinv = nh;
                }
            }
        }
        let db = th.iter().zip(&nth).map(|(a, b)| (a - b).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        let dl = (nll - ll).abs() / (ll.abs() + 1.0);
        prev = Some((dl, db));
        th = nth;
        ll = nll;
        it += 1;
    }
    if refine && maxiter > 0 {
        let mut rll = ll;
        for _ in 1..=20 {
            let (Some(g), Some(h)) = (gradient(f, &th), neg_hessian(f, &th)) else { break };
            let (hi, _) = h.sym_inverse(1e-14);
            let dir = mv(&hi, &g);
            if dot(&g, &dir) <= 0.0 {
                break;
            }
            let mut lambda = 1.0;
            let mut moved = false;
            for _ in 0..30 {
                let cand: Vec<f64> = th.iter().zip(&dir).map(|(t, d)| t + lambda * d).collect();
                let lc = ll_sum(f, &cand);
                if lc.is_finite() && lc >= rll {
                    th = cand;
                    rll = lc;
                    moved = true;
                    break;
                }
                lambda *= 0.5;
            }
            let small = dir.iter().zip(&th).all(|(d, t)| (lambda * d).abs() <= 1e-9 * (t.abs() + 1.0));
            if !moved || small {
                break;
            }
        }
        if rll > ll + 1e-7 * ll.abs().max(1.0) {
            converged = true;
        }
        ll = rll;
    }
    Ok(Optimum { th, ll, log, iters: it, converged })
}

/// Variance matrix: opg (default for most time-series commands), oim or robust.
pub fn vce(f: &LlFn, th: &[f64], cmd: &Command, default: &str) -> Result<(Matrix, &'static str, &'static str), String> {
    let k = th.len();
    let kind = cmd.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| default.to_string());
    let s = || scores(f, th).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"));
    let h = || neg_hessian(f, th).ok_or_else(|| rc_err(430, "Hessian could not be computed"));
    Ok(match kind.as_str() {
        "opg" => (opg(&s()?, k).sym_inverse(1e-14).0, "OPG", "opg"),
        // at a boundary the Hessian may not be computable: fall back on OPG
        "oim" => match h() {
            Ok(h) => (h.sym_inverse(1e-14).0, "", "oim"),
            Err(_) => (opg(&s()?, k).sym_inverse(1e-14).0, "OPG", "opg"),
        },
        r if !r.is_empty() && "robust".starts_with(r) => {
            let (hi, _) = h()?.sym_inverse(1e-14);
            (hi.mul(&opg(&s()?, k))?.mul(&hi)?, "Robust", "robust")
        }
        o => return Err(rc_err(198, format!("vce({o}) not allowed"))),
    })
}

/// Wald chi2 that the coefficients `idx` are jointly zero.
pub fn wald(b: &[f64], v: &Matrix, idx: &[usize]) -> (f64, f64) {
    let k = idx.len();
    if k == 0 {
        return (SYSMISS, 0.0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-14);
    let bs: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
    (dot(&bs, &mv(&vi, &bs)), (k - dropped.len()) as f64)
}

/// Least squares of y on the rows (no constant added).
pub fn lsq(rows: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = rows.first().map_or(0, |r| r.len());
    if k == 0 || rows.len() < k {
        return vec![0.0; k];
    }
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for (r, yv) in rows.iter().zip(y) {
        for a in 0..k {
            xy[a] += r[a] * yv;
            for c in 0..k {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    mv(&m.sym_inverse(1e-12).0, &xy)
}

// ---------------------------------------------------------------------------
// Estimation sample on the time grid
// ---------------------------------------------------------------------------

/// A regression design placed in time order.
pub struct TsSample {
    pub tvar: String,
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    pub xnames: Vec<String>,
    pub times: Vec<f64>,
    pub rows: Vec<usize>,
    pub mask: Vec<bool>,
    pub depvar: String,
    pub n_gaps: usize,
}

impl TsSample {
    pub fn n(&self) -> usize {
        self.y.len()
    }
    pub fn kx(&self) -> usize {
        self.xnames.len()
    }
    pub fn xb(&self, t: usize, b: &[f64]) -> f64 {
        dot(&self.x[t], b)
    }
    pub fn tmin(&self) -> f64 {
        self.times.first().copied().unwrap_or(SYSMISS)
    }
    pub fn tmax(&self) -> f64 {
        self.times.last().copied().unwrap_or(SYSMISS)
    }
}

/// Build `depvar indepvars` (time-series operators allowed) on the tsset time order.
pub fn ts_sample(ds: &DataSet, st: &Store, cmd: &Command, main: &str, noconstant: bool, extra: Vec<String>) -> Result<TsSample, String> {
    let tvar = tsops::time_var(st)?;
    let (dep, indep) = crate::tsreg::split_dep(ds, main)?;
    let indep = tsops::expand_text(&indep);
    let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant, weights: "", extra })?;
    let act = d.active();
    let tc = util::col(ds, &tvar)?;
    let mut pos: Vec<usize> = (0..d.n()).collect();
    pos.sort_by(|&a, &b| tc[d.rows[a]].total_cmp(&tc[d.rows[b]]));
    let times: Vec<f64> = pos.iter().map(|&p| tc[d.rows[p]]).collect();
    Ok(TsSample {
        n_gaps: tsops::gaps(&times),
        tvar,
        y: pos.iter().map(|&p| d.y[p]).collect(),
        x: pos.iter().map(|&p| act.iter().map(|&j| d.cols[j][p]).collect()).collect(),
        xnames: act.iter().map(|&j| d.names[j].clone()).collect(),
        rows: pos.iter().map(|&p| d.rows[p]).collect(),
        times,
        mask: d.mask.clone(),
        depvar: dep,
    })
}

/// Sample line: `Sample: 1960q2 thru 1990q4` (with gaps note).
pub fn sample_text(ds: &DataSet, s: &TsSample) -> (String, String, String) {
    let a = tsops::time_label(ds, &s.tvar, s.tmin());
    let b = tsops::time_label(ds, &s.tvar, s.tmax());
    (format!("Sample: {a} thru {b}{}", if s.n_gaps > 0 { ", but with gaps" } else { "" }), a, b)
}

// ---------------------------------------------------------------------------
// Display and posting
// ---------------------------------------------------------------------------

pub struct Post<'a> {
    pub title: &'a str,
    pub cmd: &'a str,
    pub depvar: &'a str,
    /// "eq:name" (names starting with "/" are ancillary parameters)
    pub names: Vec<String>,
    pub b: Vec<f64>,
    pub v: Matrix,
    pub ll: f64,
    pub n: usize,
    /// header lines between the title and the table (already formatted)
    pub header: Vec<String>,
    pub vcetype: &'a str,
    pub vce: &'a str,
    pub log: Vec<String>,
    pub notes: Vec<String>,
    pub converged: bool,
    pub iters: usize,
    pub df: Option<f64>,
}

/// Build the output text and post b, V and the common scalars to e().
pub fn post(st: &mut Store, cmd: &Command, p: Post, sample: Option<&[bool]>) -> Result<String, String> {
    let level = util::level(cmd, st)?;
    let mut out = Vec::new();
    if !p.log.is_empty() {
        out.push(String::new());
        out.extend(p.log.iter().cloned());
    }
    out.push(String::new());
    out.push(p.title.to_string());
    out.push(String::new());
    out.extend(p.header.iter().cloned());
    out.push(String::new());
    let k = p.b.len();
    let zeros = vec![false; k];
    let table = est::coef_table(&est::TableSpec { depvar: p.depvar, names: &p.names, b: &p.b, v: &p.v, omitted: &zeros, is_base: &zeros, df: p.df, level, eform: None, coef_title: "Coefficient" });
    out.extend(with_vcetype(table, p.vcetype));
    out.extend(p.notes.iter().cloned());
    if !p.converged {
        out.push("convergence not achieved".into());
    }
    let text = out.join("\n");
    let mut eqnames: Vec<String> = Vec::new();
    for n in &p.names {
        let e = n.split_once(':').map_or(n.trim_start_matches('/').to_string(), |x| x.0.to_string());
        if !eqnames.contains(&e) {
            eqnames.push(e);
        }
    }
    let names: Vec<String> = p
        .names
        .iter()
        .map(|n| if let Some(r) = n.strip_prefix('/') { format!("/:{r}") } else if n.contains(':') { n.clone() } else { format!("_:{n}") })
        .collect();
    let e = st.begin_e();
    e.set("N", p.n as f64);
    e.set("k", k as f64);
    e.set("k_eq", eqnames.len() as f64);
    e.set("ll", p.ll);
    e.set("rank", k as f64);
    e.set("ic", p.iters as f64);
    e.set("rc", 0.0);
    e.set("converged", if p.converged { 1.0 } else { 0.0 });
    e.set_str("cmd", p.cmd);
    e.set_str("cmdline", format!("{} {}", p.cmd, cmd.raw.trim_start_matches(cmd.verb.as_str()).trim()));
    e.set_str("depvar", p.depvar);
    e.set_str("title", p.title);
    e.set_str("eqnames", eqnames.join(" "));
    e.set_str("vce", p.vce);
    if !p.vcetype.is_empty() {
        e.set_str("vcetype", p.vcetype);
    }
    e.set_str("opt", "moptimize");
    e.set_str("properties", "b V");
    e.set_str("_output", text.clone());
    e.set_mat("b", Matrix::row_vector(&p.b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", p.v.with_names(names.clone(), names));
    if let Some(m) = sample {
        e.sample = Some(m.to_vec());
    }
    Ok(text)
}

/// The standard right-hand header block: Number of obs / Wald chi2 / Prob > chi2.
pub fn std_header(sample_line: &str, n: usize, chi2: f64, df: f64, ll: f64, ll_label: &str) -> Vec<String> {
    let pchi = if df > 0.0 && chi2 < SYSMISS { dist::chi2tail(df, chi2) } else { SYSMISS };
    vec![
        hline(sample_line, "Number of obs", &fmt_g(n as f64, 10)),
        hline("", &format!("Wald chi2({})", df), &f2(chi2)),
        hline(&format!("{ll_label} = {}", fmt_g(ll, 8)), "Prob > chi2", &f4(pchi)),
    ]
}

/// Parse a `from()` vector if given.
pub fn from_opt(cmd: &Command, n: usize) -> Result<Option<Vec<f64>>, String> {
    let Some(f) = cmd.opt("from", 4) else { return Ok(None) };
    let v: Vec<f64> = f
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty() && !w.contains('='))
        .map(|w| w.parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| rc_err(198, "invalid from()"))?;
    if v.len() != n {
        return Err(rc_err(503, format!("from() must contain {n} values")));
    }
    Ok(Some(v))
}

/// Parse a numlist of positive lags (`1 2`, `1/3`).
pub fn parse_lags(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for w in s.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()) {
        if let Some((a, b)) = w.split_once('/') {
            let (a, b): (usize, usize) = (a.trim().parse().map_err(|_| rc_err(121, "invalid numlist"))?, b.trim().parse().map_err(|_| rc_err(121, "invalid numlist"))?);
            out.extend(a..=b);
        } else {
            out.push(w.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
        }
    }
    if out.contains(&0) {
        return Err(rc_err(125, "lags must be positive"));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// New variable name from `predict [type] newvar`.
pub fn predict_name(cmd: &Command) -> Result<String, String> {
    let names: Vec<&str> = cmd.main.split_whitespace().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).collect();
    if names.len() != 1 {
        return Err(rc_err(if names.is_empty() { 100 } else { 103 }, "one new variable name required"));
    }
    Ok(names[0].to_string())
}

/// Add a prediction (missing outside `mask`) and the usual message.
pub fn add_prediction(ds: &mut DataSet, name: &str, mut vals: Vec<f64>, mask: &[bool], note: &str) -> Result<String, String> {
    for i in 0..ds.n_obs {
        if !mask[i] {
            vals[i] = SYSMISS;
        }
    }
    let nmiss = vals.iter().filter(|v| ferrostat_core::missing::is_missing(**v)).count();
    tsops::add_var(ds, name, vals)?;
    let mut lines = Vec::new();
    if !note.is_empty() {
        lines.push(note.to_string());
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", nmiss, if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}
