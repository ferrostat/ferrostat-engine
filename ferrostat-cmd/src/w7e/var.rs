//! After `bayes: var`: impulse responses (`bayesirf`), forecasts
//! (`bayesfcast`) and the stability check (`bayesvarstable`), each computed
//! for every retained posterior draw and summarized by posterior means,
//! medians and credible intervals.
//!
//! ```text
//! bayesirf create name [, step(#) set(file[, replace])]
//! bayesirf table|graph stat [, irf(name) impulse() response()]
//! bayesirf ctable|cgraph|ograph (name stat [impulse response]) ...
//! bayesfcast compute prefix [, step(#)]
//! bayesfcast graph varlist
//! bayesvarstable
//! ```
//! Statistics: irf, oirf, cirf, coirf, fevd.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

use super::post::draws_of;
use super::run::num9;
use super::stats;
use super::text;

/// VAR coefficients of one draw.
struct VarDraw {
    /// A[j] is the lag-(j+1) matrix (K x K): row = equation, col = variable
    a: Vec<Matrix>,
    c: Vec<f64>,
    /// residual covariance of this draw, when sampled
    sigma: Option<Matrix>,
}

struct VarSpec {
    vars: Vec<String>,
    p: usize,
    draws: Vec<VarDraw>,
    sigma: Matrix,
}

fn parse_lag(name: &str) -> Option<(usize, String)> {
    let (op, v) = name.split_once('.')?;
    let op = op.to_uppercase();
    let l = op.strip_prefix('L')?;
    let k = if l.is_empty() { 1 } else { l.parse().ok()? };
    Some((k, v.to_string()))
}

fn var_spec(st: &Store, maxdraws: usize) -> Result<VarSpec, String> {
    if st.e.macros.get("cmdname").map(String::as_str) != Some("var") {
        return Err(rc_err(301, "last estimation results not bayes: var"));
    }
    let d = draws_of(&st.e)?;
    let mut vars: Vec<String> = Vec::new();
    let mut p = 0;
    for n in &d.names {
        if let Some((eq, c)) = n.split_once(':') {
            if !vars.contains(&eq.to_string()) {
                vars.push(eq.to_string());
            }
            if let Some((k, _)) = parse_lag(c) {
                p = p.max(k);
            }
        }
    }
    let k = vars.len();
    if k == 0 || p == 0 {
        return Err(rc_err(498, "no VAR lag coefficients found"));
    }
    let n = d.ll.len();
    let step = (n / maxdraws.max(1)).max(1);
    let mut draws = Vec::new();
    for r in (0..n).step_by(step) {
        let mut a = vec![Matrix::zeros(k, k); p];
        let mut c = vec![0.0; k];
        for (j, nm) in d.names.iter().enumerate() {
            let Some((eq, cn)) = nm.split_once(':') else { continue };
            let Some(i) = vars.iter().position(|v| v == eq) else { continue };
            if cn == "_cons" {
                c[i] = d.cols[j][r];
            } else if let Some((l, v)) = parse_lag(cn) {
                if let Some(vj) = vars.iter().position(|x| *x == v) {
                    a[l - 1][(i, vj)] = d.cols[j][r];
                }
            }
        }
        let mut sg = Matrix::zeros(k, k);
        let mut have = false;
        for i in 0..k {
            for j in 0..=i {
                if let Some(c2) = d.names.iter().position(|n| *n == format!("Sigma_{}_{}", i + 1, j + 1)) {
                    sg[(i, j)] = d.cols[c2][r];
                    sg[(j, i)] = d.cols[c2][r];
                    have = true;
                }
            }
        }
        draws.push(VarDraw { a, c, sigma: if have { Some(sg) } else { None } });
    }
    let sigma = st.e.matrices.get("Sigma").cloned().unwrap_or_else(|| Matrix::identity(k));
    Ok(VarSpec { vars, p, draws, sigma })
}

/// Impulse responses Phi_0..Phi_h.
fn phis(dr: &VarDraw, k: usize, h: usize) -> Vec<Matrix> {
    let mut out = vec![Matrix::identity(k)];
    for s in 1..=h {
        let mut m = Matrix::zeros(k, k);
        for j in 1..=dr.a.len().min(s) {
            let t = dr.a[j - 1].mul(&out[s - j]).unwrap();
            m = m.add(&t).unwrap();
        }
        out.push(m);
    }
    out
}

const STATS: [&str; 5] = ["irf", "oirf", "cirf", "coirf", "fevd"];

/// stat x impulse x response x step values for one draw.
fn irf_values(dr: &VarDraw, chol: &Matrix, k: usize, h: usize) -> Vec<Vec<Vec<Vec<f64>>>> {
    let ph = phis(dr, k, h);
    let th: Vec<Matrix> = ph.iter().map(|m| m.mul(chol).unwrap()).collect();
    let mut out = vec![vec![vec![vec![0.0; h + 1]; k]; k]; 5];
    for imp in 0..k {
        for res in 0..k {
            let (mut ci, mut co) = (0.0, 0.0);
            for s in 0..=h {
                let a = ph[s][(res, imp)];
                let o = th[s][(res, imp)];
                ci += a;
                co += o;
                out[0][imp][res][s] = a;
                out[1][imp][res][s] = o;
                out[2][imp][res][s] = ci;
                out[3][imp][res][s] = co;
                // fevd at horizon s: share of the forecast error of `res` due to `imp`
                if s > 0 {
                    let num: f64 = (0..s).map(|t| th[t][(res, imp)].powi(2)).sum();
                    let den: f64 = (0..s).map(|t| (0..k).map(|j| th[t][(res, j)].powi(2)).sum::<f64>()).sum();
                    out[4][imp][res][s] = if den > 0.0 { num / den } else { 0.0 };
                }
            }
        }
    }
    out
}

fn spectral_radius(a: &Matrix) -> f64 {
    // ||A^m||^(1/m) by repeated squaring with renormalization
    let mut m = a.clone();
    let mut logscale = 0.0;
    let mut pow = 1.0;
    for _ in 0..9 {
        let nrm = (0..m.rows).flat_map(|i| (0..m.cols).map(move |j| (i, j))).map(|(i, j)| m[(i, j)].abs()).fold(0.0, f64::max);
        if nrm == 0.0 || !nrm.is_finite() {
            return if nrm == 0.0 { 0.0 } else { f64::INFINITY };
        }
        m = m.scale(1.0 / nrm);
        logscale = 2.0 * (logscale + nrm.ln());
        m = m.mul(&m).unwrap();
        pow *= 2.0;
    }
    let nrm = (0..m.rows).flat_map(|i| (0..m.cols).map(move |j| (i, j))).map(|(i, j)| m[(i, j)].abs()).fold(0.0, f64::max);
    ((logscale + nrm.max(1e-300).ln()) / pow).exp()
}

fn companion(dr: &VarDraw, k: usize, p: usize) -> Matrix {
    let mut c = Matrix::zeros(k * p, k * p);
    for j in 0..p {
        for r in 0..k {
            for s in 0..k {
                c[(r, j * k + s)] = dr.a[j][(r, s)];
            }
        }
    }
    for i in k..k * p {
        c[(i, i - k)] = 1.0;
    }
    c
}

fn split(rest: &str) -> (String, Vec<(String, Option<String>)>) {
    match text::find_top(rest, ',') {
        Some(i) => (rest[..i].trim().to_string(), text::options(&rest[i + 1..])),
        None => (rest.trim().to_string(), Vec::new()),
    }
}

fn optv(o: &[(String, Option<String>)], full: &str, min: usize) -> Option<String> {
    o.iter().rev().find(|(n, _)| text::abbrev(n, full, min)).and_then(|x| x.1.clone())
}

fn level(st: &Store, o: &[(String, Option<String>)]) -> f64 {
    optv(o, "clevel", 2).and_then(|s| s.trim().parse().ok()).or_else(|| st.settings.get("clevel").and_then(|v| v.parse().ok())).or_else(|| st.e.get("clevel")).unwrap_or(st.level)
}

pub fn dispatch(verb: &str, ds: &mut DataSet, st: &mut Store, rest: &str, cmd: &Command) -> Result<String, String> {
    let r = rest.trim();
    let end = r.find(|c: char| c.is_whitespace() || c == ',').unwrap_or(r.len());
    let (sub, rr) = (r[..end].to_lowercase(), r[end..].trim().to_string());
    match verb {
        "bayesvarstable" => varstable(st, rest),
        "bayesirf" => match sub.as_str() {
            s if text::abbrev(s, "create", 2) => irf_create(st, &rr),
            "table" => irf_table(st, &rr, false),
            "ctable" => irf_table(st, &rr, true),
            "graph" => irf_graph(st, &rr, cmd, "graph"),
            "cgraph" => irf_graph(st, &rr, cmd, "cgraph"),
            "ograph" => irf_graph(st, &rr, cmd, "ograph"),
            o => Err(rc_err(198, format!("bayesirf {o} is unrecognized"))),
        },
        _ => match sub.as_str() {
            s if text::abbrev(s, "compute", 4) => fcast_compute(ds, st, &rr),
            "graph" => fcast_graph(ds, st, &rr, cmd),
            o => Err(rc_err(198, format!("bayesfcast {o} is unrecognized"))),
        },
    }
}

fn varstable(st: &mut Store, rest: &str) -> Result<String, String> {
    let (_, o) = split(rest);
    let vs = var_spec(st, 1000)?;
    let k = vs.vars.len();
    let rad: Vec<f64> = vs.draws.iter().map(|d| spectral_radius(&companion(d, k, vs.p))).collect();
    let lv = level(st, &o);
    let s = stats::summarize(&rad, lv, false, 0);
    let pstable = rad.iter().filter(|r| **r < 1.0).count() as f64 / rad.len() as f64;
    let mut out = vec![String::new(), "Eigenvalue stability condition".to_string(), String::new()];
    out.push("-".repeat(78));
    out.push(format!("{:>12} |      Mean   Std. dev.     MCSE     Median  [{}% cred. interval]", "", crate::format::fmt_g(lv, 4)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    out.push(format!("{:>12} | {:>9}  {:>9}  {:>8}  {:>9}  {:>9}  {:>9}", "Max modulus", num9(s.mean), num9(s.sd), num9(s.mcse), num9(s.median), num9(s.lo), num9(s.hi)));
    out.push("-".repeat(78));
    out.push(format!("Pr(eigenvalues lie inside the unit circle) = {pstable:.4}"));
    if pstable > 0.95 {
        out.push("The VAR is stable with high posterior probability.".into());
    } else {
        out.push("The VAR may not be stable.".into());
    }
    let r = st.begin_r();
    r.set("stable_prob", pstable);
    r.set("modulus_mean", s.mean);
    r.set_mat("modulus", Matrix::row_vector(&[s.mean, s.sd, s.median, s.lo, s.hi]));
    Ok(out.join("\n"))
}

fn irf_key(name: &str) -> String {
    format!("__bayesirf_{name}")
}

fn irf_create(st: &mut Store, rest: &str) -> Result<String, String> {
    let (name, o) = split(rest);
    let name = name.split_whitespace().next().ok_or_else(|| rc_err(198, "IRF name required"))?.to_string();
    let h = optv(&o, "step", 4).and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(8);
    let vs = var_spec(st, 1000)?;
    let k = vs.vars.len();
    let chol = vs.sigma.cholesky().unwrap_or_else(|_| Matrix::identity(k));
    let lv = level(st, &o);
    let all: Vec<Vec<Vec<Vec<Vec<f64>>>>> = vs.draws.iter().map(|d| {
        let c = d.sigma.as_ref().and_then(|s| s.cholesky().ok()).unwrap_or_else(|| chol.clone());
        irf_values(d, &c, k, h)
    }).collect();
    let mut rows = Vec::new();
    let mut rn = Vec::new();
    let a = (1.0 - lv / 100.0) / 2.0;
    for s in 0..5 {
        for imp in 0..k {
            for res in 0..k {
                for t in 0..=h {
                    let mut v: Vec<f64> = all.iter().map(|x| x[s][imp][res][t]).collect();
                    let mean = stats::mean(&v);
                    v.sort_by(f64::total_cmp);
                    rows.push(vec![s as f64, imp as f64, res as f64, t as f64, mean, stats::quantile(&v, 0.5), stats::quantile(&v, a), stats::quantile(&v, 1.0 - a)]);
                    rn.push(format!("{}:{}>{}", STATS[s], vs.vars[imp], vs.vars[res]));
                }
            }
        }
    }
    let m = Matrix::from_rows(&rows).with_names(rn, ["stat", "impulse", "response", "step", "mean", "median", "lower", "upper"].iter().map(|s| s.to_string()).collect());
    st.matrices.insert(irf_key(&name), m.clone());
    st.globals.insert("__bayesirf_vars".into(), vs.vars.join(" "));
    st.globals.insert("__bayesirf_last".into(), name.clone());
    st.globals.insert(format!("__bayesirf_level_{name}"), lv.to_string());
    if let Some(f) = optv(&o, "set", 3) {
        let parts = text::split_top(&f, ',');
        let file = util::unquote(parts[0].trim());
        let file = if file.contains('.') { file } else { format!("{file}.irf") };
        let replace = parts.iter().skip(1).any(|p| p.trim() == "replace");
        if ferrostat_core::vfs::exists(&file) && !replace {
            return Err(rc_err(602, format!("file {file} already exists")));
        }
        let mut ds = DataSet::new();
        let cols = ["stat", "impulse", "response", "step", "mean", "median", "lower", "upper"];
        for (j, c) in cols.iter().enumerate() {
            ds.add_variable(util::new_double(c, rows.iter().map(|r| r[j]).collect())).map_err(|e| rc_err(110, e))?;
        }
        ds.n_obs = rows.len();
        let mut buf = Vec::new();
        ferrostat_core::dta::DtaWriter::write(&mut buf, &ds).map_err(|e| rc_err(693, e))?;
        ferrostat_core::vfs::write(&file, &buf).map_err(|e| rc_err(603, e))?;
    }
    Ok(format!("(bayesirf {name} created: {} draws, {h} steps)", vs.draws.len()))
}

/// Rows of an IRF result for (stat, impulse, response).
fn irf_rows(st: &Store, name: &str, stat: &str, imp: &str, res: &str) -> Result<Vec<Vec<f64>>, String> {
    let m = st.matrices.get(&irf_key(name)).ok_or_else(|| rc_err(111, format!("bayesirf result {name} not found; use bayesirf create")))?;
    let vars: Vec<String> = st.globals.get("__bayesirf_vars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let s = STATS.iter().position(|x| *x == stat).ok_or_else(|| rc_err(198, format!("statistic {stat} not available; use irf, oirf, cirf, coirf or fevd")))? as f64;
    let i = vars.iter().position(|v| v == imp).ok_or_else(|| rc_err(111, format!("impulse {imp} not found")))? as f64;
    let r = vars.iter().position(|v| v == res).ok_or_else(|| rc_err(111, format!("response {res} not found")))? as f64;
    Ok((0..m.rows).filter(|&k| m.get(k, 0) == s && m.get(k, 1) == i && m.get(k, 2) == r).map(|k| m.row(k).to_vec()).collect())
}

/// (name stat [impulse response]) specifications, or `stat` with options.
fn irf_specs(st: &Store, rest: &str, combined: bool) -> Result<Vec<(String, String, String, String)>, String> {
    let (main, o) = split(rest);
    let vars: Vec<String> = st.globals.get("__bayesirf_vars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let last = st.globals.get("__bayesirf_last").cloned().unwrap_or_default();
    let imps = optv(&o, "impulse", 1).map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>()).unwrap_or_else(|| vars.clone());
    let ress = optv(&o, "response", 1).map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>()).unwrap_or_else(|| vars.clone());
    let name = optv(&o, "irf", 3).unwrap_or(last);
    let mut out = Vec::new();
    if combined || main.starts_with('(') {
        for w in text::words(&main) {
            let inner = w.trim_start_matches('(').trim_end_matches(')');
            let ws: Vec<&str> = inner.split_whitespace().collect();
            if ws.len() < 2 {
                return Err(rc_err(198, format!("({inner}) invalid; specify (name stat [impulse response])")));
            }
            let (ii, rr) = if ws.len() >= 4 { (vec![ws[2].to_string()], vec![ws[3].to_string()]) } else { (imps.clone(), ress.clone()) };
            for i in &ii {
                for r in &rr {
                    out.push((ws[0].to_string(), ws[1].to_lowercase(), i.clone(), r.clone()));
                }
            }
        }
    } else {
        let stats_: Vec<String> = if main.is_empty() { vec!["oirf".into()] } else { main.split_whitespace().map(|s| s.to_lowercase()).collect() };
        for s in &stats_ {
            for i in &imps {
                for r in &ress {
                    out.push((name.clone(), s.clone(), i.clone(), r.clone()));
                }
            }
        }
    }
    Ok(out)
}

fn irf_table(st: &mut Store, rest: &str, combined: bool) -> Result<String, String> {
    let specs = irf_specs(st, rest, combined)?;
    let mut out = Vec::new();
    for (name, stat, i, r) in &specs {
        let rows = irf_rows(st, name, stat, i, r)?;
        let lv = st.globals.get(&format!("__bayesirf_level_{name}")).cloned().unwrap_or_else(|| "95".into());
        out.push(String::new());
        out.push(format!("Results from {name}: {stat}, impulse {i}, response {r}"));
        out.push("-".repeat(56));
        out.push(format!("{:>6} | {:>10} {:>10} {:>12} {:>12}", "Step", "Mean", "Median", format!("[{lv}% cred."), "interval]"));
        out.push(format!("{}+{}", "-".repeat(7), "-".repeat(48)));
        for row in &rows {
            out.push(format!("{:>6} | {:>10} {:>10} {:>12} {:>12}", row[3] as usize, num9(row[4]), num9(row[5]), num9(row[6]), num9(row[7])));
        }
        out.push("-".repeat(56));
    }
    Ok(out.join("\n"))
}

fn irf_graph(st: &mut Store, rest: &str, cmd: &Command, kind: &str) -> Result<String, String> {
    let specs = irf_specs(st, rest, kind != "graph")?;
    let mut g = Graph::default();
    for (name, stat, i, r) in &specs {
        let rows = irf_rows(st, name, stat, i, r)?;
        let mean: Vec<(f64, f64)> = rows.iter().map(|x| (x[3], x[4])).collect();
        let band: Vec<(f64, f64, f64)> = rows.iter().map(|x| (x[3], x[6], x[7])).collect();
        let lab = format!("{name}: {stat} {i} -> {r}");
        if kind == "ograph" {
            g.layers.push(Layer { kind: LayerKind::Line, points: mean, label: lab, ..Default::default() });
            g.legend = true;
        } else {
            let mut c = Graph::default();
            c.layers.push(Layer { kind: LayerKind::Range(RangeStyle::Area), ranges: band, label: "credible interval".into(), alpha: 0.3, ..Default::default() });
            c.layers.push(Layer { kind: LayerKind::Line, points: mean, label: "posterior mean".into(), ..Default::default() });
            c.title = lab;
            c.xtitle = "Step".into();
            g.children.push(c);
        }
    }
    if g.children.len() == 1 {
        g = g.children.remove(0);
    } else if !g.children.is_empty() {
        g.cols = (g.children.len() as f64).sqrt().ceil() as usize;
    } else {
        g.xtitle = "Step".into();
    }
    g.command = format!("bayesirf {kind} {rest}");
    crate::graph::finish(&mut g, cmd, st)
}

fn fcast_compute(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (pre, o) = split(rest);
    let pre = pre.trim().to_string();
    if pre.is_empty() {
        return Err(rc_err(198, "prefix for the new variables required"));
    }
    let h = optv(&o, "step", 4).and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(1);
    let vs = var_spec(st, 1000)?;
    let k = vs.vars.len();
    let lv = level(st, &o);
    let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let last = (0..ds.n_obs).rev().find(|&i| sample.get(i).copied().unwrap_or(false)).ok_or_else(|| rc_err(2000, "no observations"))?;
    if last + 1 < vs.p {
        return Err(rc_err(2000, "insufficient observations"));
    }
    let cols: Vec<Vec<f64>> = vs.vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let chol = vs.sigma.cholesky().unwrap_or_else(|_| Matrix::identity(k));
    // simulate forecast paths
    let mut paths: Vec<Vec<Vec<f64>>> = Vec::new(); // draw x step x var
    {
        let mut rng = st.rng.borrow_mut();
        for d in &vs.draws {
            let mut hist: Vec<Vec<f64>> = (0..vs.p).map(|j| (0..k).map(|v| cols[v][last - j]).collect()).collect(); // hist[0] = most recent
            let mut path = Vec::new();
            for _ in 0..h {
                let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
                let mut y = d.c.clone();
                for (j, a) in d.a.iter().enumerate() {
                    for r in 0..k {
                        for s in 0..k {
                            y[r] += a[(r, s)] * hist[j][s];
                        }
                    }
                }
                let cd = d.sigma.as_ref().and_then(|s| s.cholesky().ok()).unwrap_or_else(|| chol.clone());
                for r in 0..k {
                    y[r] += (0..=r).map(|s| cd[(r, s)] * z[s]).sum::<f64>();
                }
                hist.insert(0, y.clone());
                hist.truncate(vs.p);
                path.push(y);
            }
            paths.push(path);
        }
    }
    // extend the data when the forecast horizon runs past the end
    let need = last + 1 + h;
    if need > ds.n_obs {
        let add = need - ds.n_obs;
        let old = ds.n_obs;
        for v in ds.vars.iter_mut() {
            for _ in 0..add {
                v.push_missing();
            }
        }
        ds.n_obs = need;
        if let Some((_, tv)) = st.tsset.clone() {
            if let Some(var) = ds.vars.iter_mut().find(|v| v.name == tv) {
                let prev = var.data[old - 1].num();
                let delta = if old >= 2 { prev - var.data[old - 2].num() } else { 1.0 };
                for t in old..need {
                    let val = prev + delta * (t - old + 1) as f64;
                    var.data[t] = crate::util::typed_value(var.var_type, val);
                }
            }
        }
    }
    let a = (1.0 - lv / 100.0) / 2.0;
    for (vi, v) in vs.vars.iter().enumerate() {
        let mut mean = vec![SYSMISS; ds.n_obs];
        let mut lo = vec![SYSMISS; ds.n_obs];
        let mut hi = vec![SYSMISS; ds.n_obs];
        let mut sd = vec![SYSMISS; ds.n_obs];
        // the last observed value anchors the forecast line
        if !is_missing(cols[vi][last]) {
            mean[last] = cols[vi][last];
        }
        for s in 0..h {
            let mut x: Vec<f64> = paths.iter().map(|p| p[s][vi]).collect();
            mean[last + 1 + s] = stats::mean(&x);
            sd[last + 1 + s] = stats::var(&x).sqrt();
            x.sort_by(f64::total_cmp);
            lo[last + 1 + s] = stats::quantile(&x, a);
            hi[last + 1 + s] = stats::quantile(&x, 1.0 - a);
        }
        for (suffix, vals) in [("", mean), ("_lb", lo), ("_ub", hi), ("_sd", sd)] {
            let nm = format!("{pre}{v}{suffix}");
            if ds.var_index.contains_key(&nm) {
                if o.iter().any(|(n, _)| n == "replace") {
                    let _ = ds.drop_variable(&nm);
                } else {
                    return Err(rc_err(110, format!("variable {nm} already defined")));
                }
            }
            ds.add_variable(util::new_double(&nm, vals)).map_err(|e| rc_err(110, e))?;
        }
    }
    st.globals.insert("__bayesfcast_vars".into(), vs.vars.iter().map(|v| format!("{pre}{v}")).collect::<Vec<_>>().join(" "));
    Ok(format!("(forecasts computed for {} steps from {} draws)", h, vs.draws.len()))
}

fn fcast_graph(ds: &mut DataSet, st: &mut Store, rest: &str, cmd: &Command) -> Result<String, String> {
    let (main, _o) = split(rest);
    let names: Vec<String> = if main.is_empty() { st.globals.get("__bayesfcast_vars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default() } else { main.split_whitespace().map(String::from).collect() };
    if names.is_empty() {
        return Err(rc_err(198, "no forecasts found; use bayesfcast compute"));
    }
    let tvar = st.tsset.as_ref().map(|t| t.1.clone());
    let t: Vec<f64> = match &tvar {
        Some(tv) => util::col(ds, tv)?,
        None => (1..=ds.n_obs).map(|i| i as f64).collect(),
    };
    let mut g = Graph::default();
    for n in &names {
        let m = util::col(ds, n)?;
        let lo = util::col(ds, &format!("{n}_lb"))?;
        let hi = util::col(ds, &format!("{n}_ub"))?;
        let mut c = Graph::default();
        c.layers.push(Layer { kind: LayerKind::Range(RangeStyle::Area), ranges: (0..ds.n_obs).filter(|&i| !is_missing(lo[i])).map(|i| (t[i], lo[i], hi[i])).collect(), label: "credible interval".into(), alpha: 0.3, ..Default::default() });
        c.layers.push(Layer { kind: LayerKind::Line, points: (0..ds.n_obs).filter(|&i| !is_missing(m[i])).map(|i| (t[i], m[i])).collect(), label: "posterior mean forecast".into(), ..Default::default() });
        c.title = format!("Forecast for {}", n);
        g.children.push(c);
    }
    if g.children.len() == 1 {
        g = g.children.remove(0);
    } else {
        g.cols = 1;
    }
    g.command = format!("bayesfcast graph {rest}");
    crate::graph::finish(&mut g, cmd, st)
}
