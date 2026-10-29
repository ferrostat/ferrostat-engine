//! Generalized method of moments: a numerical core (moment functions,
//! weight matrices, Gauss-Newton minimisation, sandwich variance) shared by
//! the moment-based estimators, and the interactive `gmm` command with
//! substitutable expressions (`{b}` parameters, `{xb: x1 x2}` combinations).

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::nlest::{self, Params};
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("gmm", Box::new(Gmm))]
}

pub fn predict(_ds: &mut DataSet, _st: &mut Store, _cmd: &Command, _ecmd: &str) -> Option<Result<String, String>> {
    None
}

// ---------------------------------------------------------------------------
// numerical core
// ---------------------------------------------------------------------------

/// Per-observation moment contributions (n rows of q moments) at b.
pub type MomFn<'a> = dyn Fn(&[f64]) -> Result<Vec<Vec<f64>>, String> + 'a;

/// How the moment covariance S is estimated.
#[derive(Clone, Debug, PartialEq)]
pub enum SKind {
    Robust,
    Cluster(Vec<f64>),
    /// Newey-West (Bartlett) with this many lags
    Hac(usize),
}

/// Weighted mean of the moment rows.
pub fn gbar(m: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
    let q = m.first().map_or(0, |r| r.len());
    let sw: f64 = w.iter().sum();
    let mut g = vec![0.0; q];
    for (r, wt) in m.iter().zip(w) {
        for j in 0..q {
            g[j] += wt * r[j];
        }
    }
    g.iter().map(|x| x / sw).collect()
}

/// S = (1/N) sum of outer products (by observation, cluster or with HAC lags).
pub fn s_matrix(m: &[Vec<f64>], w: &[f64], kind: &SKind) -> Matrix {
    let q = m.first().map_or(0, |r| r.len());
    let sw: f64 = w.iter().sum();
    let mut s = Matrix::zeros(q, q);
    match kind {
        SKind::Robust => {
            for (r, wt) in m.iter().zip(w) {
                for a in 0..q {
                    for c in 0..q {
                        s[(a, c)] += wt * r[a] * r[c];
                    }
                }
            }
        }
        SKind::Cluster(cl) => {
            let mut groups: std::collections::BTreeMap<u64, Vec<f64>> = std::collections::BTreeMap::new();
            for ((r, wt), g) in m.iter().zip(w).zip(cl) {
                let e = groups.entry(g.to_bits()).or_insert_with(|| vec![0.0; q]);
                for a in 0..q {
                    e[a] += wt * r[a];
                }
            }
            for v in groups.values() {
                for a in 0..q {
                    for c in 0..q {
                        s[(a, c)] += v[a] * v[c];
                    }
                }
            }
        }
        SKind::Hac(l) => {
            let n = m.len();
            for lag in 0..=*l {
                let k = 1.0 - lag as f64 / (*l as f64 + 1.0);
                for t in lag..n {
                    for a in 0..q {
                        for c in 0..q {
                            let v = w[t] * m[t][a] * m[t - lag][c];
                            if lag == 0 {
                                s[(a, c)] += v;
                            } else {
                                s[(a, c)] += k * v;
                                s[(c, a)] += k * v;
                            }
                        }
                    }
                }
            }
        }
    }
    s.scale(1.0 / sw)
}

/// Jacobian of gbar with respect to b (q x k, central differences).
pub fn jac(mom: &MomFn, b: &[f64], w: &[f64]) -> Result<Matrix, String> {
    let k = b.len();
    let mut cols = Vec::with_capacity(k);
    for j in 0..k {
        let h = 1e-6 * (b[j].abs() + 1e-2).max(1e-4);
        let mut bp = b.to_vec();
        bp[j] += h;
        let gp = gbar(&mom(&bp)?, w);
        bp[j] = b[j] - h;
        let gm = gbar(&mom(&bp)?, w);
        cols.push(gp.iter().zip(&gm).map(|(a, c)| (a - c) / (2.0 * h)).collect::<Vec<f64>>());
    }
    let q = cols.first().map_or(0, |c| c.len());
    let mut g = Matrix::zeros(q, k);
    for j in 0..k {
        for i in 0..q {
            g[(i, j)] = cols[j][i];
        }
    }
    Ok(g)
}

fn quad(g: &[f64], w: &Matrix) -> f64 {
    let q = g.len();
    (0..q).map(|a| (0..q).map(|c| g[a] * w[(a, c)] * g[c]).sum::<f64>()).sum()
}

/// Minimise Q(b) = gbar' W gbar by Gauss-Newton with step halving.
/// Returns (b, Q, iteration log of Q, converged).
pub fn minimize(mom: &MomFn, b0: Vec<f64>, wm: &Matrix, w: &[f64], max_iter: usize) -> Result<(Vec<f64>, f64, Vec<f64>, bool), String> {
    let k = b0.len();
    let mut b = b0;
    let crit = |b: &[f64]| -> Result<f64, String> {
        let g = gbar(&mom(b)?, w);
        if g.iter().any(|x| !x.is_finite() || is_missing(*x)) {
            return Ok(f64::INFINITY);
        }
        Ok(quad(&g, wm))
    };
    let mut q = crit(&b)?;
    if !q.is_finite() {
        return Err(rc_err(480, "could not evaluate the moments at the initial values"));
    }
    let mut log = vec![q];
    let mut converged = false;
    for _ in 0..max_iter {
        let g = gbar(&mom(&b)?, w);
        let gm = jac(mom, &b, w)?;
        let gw = gm.transpose().mul(wm)?;
        let a = gw.mul(&gm)?;
        let rhs: Vec<f64> = (0..k).map(|i| (0..g.len()).map(|j| gw[(i, j)] * g[j]).sum()).collect();
        let ai = eq_inverse(&a);
        let step: Vec<f64> = (0..k).map(|i| -(0..k).map(|j| ai[(i, j)] * rhs[j]).sum::<f64>()).collect();
        let mut lambda = 1.0;
        let mut acc = None;
        for _ in 0..40 {
            let nb: Vec<f64> = b.iter().zip(&step).map(|(x, s)| x + lambda * s).collect();
            let nq = crit(&nb)?;
            if nq <= q {
                acc = Some((nb, nq));
                break;
            }
            lambda /= 2.0;
        }
        let Some((nb, nq)) = acc else {
            converged = true;
            break;
        };
        let db = b.iter().zip(&nb).map(|(x, y)| (x - y).abs() / (x.abs() + 1.0)).fold(0.0, f64::max);
        b = nb;
        let dq = q - nq;
        q = nq;
        log.push(q);
        if db < 1e-9 || dq < 1e-16 * (1.0 + q) || q < 1e-28 {
            converged = true;
            break;
        }
    }
    Ok((b, q, log, converged))
}

/// Inverse of a symmetric positive (semi)definite matrix after scaling it
/// to unit diagonal (robust to regressors on very different scales).
pub fn eq_inverse(a: &Matrix) -> Matrix {
    let k = a.rows;
    let d: Vec<f64> = (0..k).map(|i| if a[(i, i)] > 0.0 { 1.0 / a[(i, i)].sqrt() } else { 1.0 }).collect();
    let mut b = a.clone();
    for i in 0..k {
        for j in 0..k {
            b[(i, j)] *= d[i] * d[j];
        }
    }
    let (mut bi, _) = b.sym_inverse(1e-14);
    for i in 0..k {
        for j in 0..k {
            bi[(i, j)] *= d[i] * d[j];
        }
    }
    bi
}

/// Sandwich variance (G'WG)^-1 G'W S W G (G'WG)^-1 / N.
pub fn gmm_v(gm: &Matrix, wm: &Matrix, s: &Matrix, n: f64) -> Result<Matrix, String> {
    if gm.rows == gm.cols {
        // exactly identified: G^-1 S G^-1' / N with row and column scaling
        let k = gm.rows;
        let rs: Vec<f64> = (0..k).map(|i| 1.0 / (0..k).map(|j| gm[(i, j)].abs()).fold(1e-300, f64::max)).collect();
        let mut g2 = gm.clone();
        for i in 0..k {
            for j in 0..k {
                g2[(i, j)] *= rs[i];
            }
        }
        let cs: Vec<f64> = (0..k).map(|j| 1.0 / (0..k).map(|i| g2[(i, j)].abs()).fold(1e-300, f64::max)).collect();
        for i in 0..k {
            for j in 0..k {
                g2[(i, j)] *= cs[j];
            }
        }
        if let Ok(inv) = g2.inverse() {
            let mut gi = inv;
            for i in 0..k {
                for j in 0..k {
                    gi[(i, j)] *= cs[i] * rs[j];
                }
            }
            return Ok(gi.mul(s)?.mul(&gi.transpose())?.scale(1.0 / n));
        }
    }
    let gw = gm.transpose().mul(wm)?;
    let ai = eq_inverse(&gw.mul(gm)?);
    let mid = gw.mul(s)?.mul(&gw.transpose())?;
    Ok(ai.mul(&mid)?.mul(&ai)?.scale(1.0 / n))
}

/// Inverse of S as a weight matrix.
pub fn weight_of(s: &Matrix) -> Matrix {
    eq_inverse(s)
}

/// Exactly identified stacked estimator variance G^-1 S G^-1' / N.
pub fn exact_v(mom: &MomFn, b: &[f64], w: &[f64], kind: &SKind) -> Result<Matrix, String> {
    let gm = jac(mom, b, w)?;
    let s = s_matrix(&mom(b)?, w, kind);
    let n: f64 = w.iter().sum();
    let wm = Matrix::identity(gm.rows);
    gmm_v(&gm, &wm, &s, n)
}

// ---------------------------------------------------------------------------
// gmm command
// ---------------------------------------------------------------------------

pub struct Gmm;

struct GEq {
    name: String,
    expr: ferrostat_parser::Expr,
    inst: Vec<String>,
    cons: bool,
}

/// Split the moment-equation list: `(name: expr) (expr)` or a single expression.
fn equations(main: &str) -> Vec<(Option<String>, String)> {
    let t = main.trim();
    let groups = if t.starts_with('(') { xm::paren_groups(t) } else { vec![t.to_string()] };
    groups
        .into_iter()
        .map(|g| {
            let brace = g.find('{').unwrap_or(g.len());
            match g[..brace].find(':') {
                Some(i) if g[..i].trim().chars().all(|c| c.is_alphanumeric() || c == '_') && !g[..i].trim().is_empty() => (Some(g[..i].trim().to_string()), g[i + 1..].trim().to_string()),
                _ => (None, g.trim().to_string()),
            }
        })
        .collect()
}

fn skind_of(ds: &DataSet, rows: &[usize], spec: &str) -> Result<Option<SKind>, String> {
    let w: Vec<&str> = spec.split_whitespace().collect();
    Ok(match w.first().map(|s| s.to_lowercase()).as_deref() {
        Some("robust") | Some("r") => Some(SKind::Robust),
        Some("cluster") | Some("cl") => {
            let v = xm::var_name(ds, w.get(1).ok_or_else(|| rc_err(198, "cluster variable required"))?)?;
            let c = util::col(ds, &v)?;
            Some(SKind::Cluster(rows.iter().map(|&i| c[i]).collect()))
        }
        Some("hac") => {
            let lags: usize = w.iter().skip(1).find_map(|s| s.parse().ok()).unwrap_or(1);
            Some(SKind::Hac(lags))
        }
        Some("unadjusted") | Some("un") => None,
        Some(o) => return Err(rc_err(198, format!("{o} not allowed"))),
        None => Some(SKind::Robust),
    })
}

impl CommandHandler for Gmm {
    fn name(&self) -> &'static str {
        "gmm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let eqtext = equations(&cmd.main);
        if eqtext.is_empty() {
            return Err(rc_err(198, "moment equation required"));
        }
        let mut params = Params::default();
        let mut eqs: Vec<GEq> = Vec::new();
        for (i, (nm, t)) in eqtext.iter().enumerate() {
            let rw = params.rewrite(ds, t)?;
            let e = parse_expression(&rw).map_err(|e| rc_err(198, e.message))?;
            eqs.push(GEq { name: nm.clone().unwrap_or_else(|| (i + 1).to_string()), expr: e, inst: Vec::new(), cons: true });
        }
        if let Some(fr) = cmd.opt("from", 4) {
            let toks: Vec<&str> = fr.split_whitespace().collect();
            let mut i = 0;
            while i + 1 < toks.len() {
                let (n, v) = (toks[i].trim_end_matches('='), toks[i + 1]);
                if let (Some(j), Ok(x)) = (params.names.iter().position(|p| p == n || p == n.trim_start_matches('/')), v.parse::<f64>()) {
                    params.init[j] = x;
                }
                i += 2;
            }
        }
        // instruments
        for (name, arg) in &cmd.opts {
            let lname = name.to_lowercase();
            if !(lname.len() >= 4 && "instruments".starts_with(lname.as_str())) {
                continue;
            }
            let arg = arg.clone().unwrap_or_default();
            let (body, nocons) = match arg.split_once(',') {
                Some((a, o)) => (a.to_string(), o.to_lowercase().contains("nocons")),
                None => (arg.clone(), false),
            };
            let (targets, list): (Vec<usize>, String) = match body.split_once(':') {
                Some((l, r)) => {
                    let mut ts = Vec::new();
                    for tok in l.split_whitespace() {
                        let k = eqs.iter().position(|e| e.name == tok).or_else(|| tok.parse::<usize>().ok().map(|n| n - 1)).ok_or_else(|| rc_err(198, format!("equation {tok} not found")))?;
                        ts.push(k);
                    }
                    (ts, r.to_string())
                }
                None => ((0..eqs.len()).collect(), body.clone()),
            };
            let vars = if list.trim().is_empty() { Vec::new() } else { crate::varlist::expand(ds, list.trim())? };
            for k in targets {
                eqs[k].inst.extend(vars.iter().cloned());
                if nocons {
                    eqs[k].cons = false;
                }
            }
        }
        let exprs: Vec<&ferrostat_parser::Expr> = eqs.iter().map(|e| &e.expr).collect();
        let (mut mask, wts) = nlest::sample(ds, st, cmd, &exprs, &params.init)?;
        let allinst: Vec<String> = eqs.iter().flat_map(|e| e.inst.clone()).collect();
        util::markout(ds, &mut mask, &allinst);
        let rows = util::rows(&mask);
        let n = rows.len();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let w: Vec<f64> = match &wts {
            Some(wt) => rows.iter().map(|&i| wt.w[i]).collect(),
            None => vec![1.0; n],
        };
        // instrument columns per equation
        let mut zs: Vec<Vec<Vec<f64>>> = Vec::new();
        for e in &eqs {
            let mut z: Vec<Vec<f64>> = Vec::new();
            for v in &e.inst {
                let c = util::col(ds, v)?;
                z.push(rows.iter().map(|&i| c[i]).collect());
            }
            if e.cons {
                z.push(vec![1.0; n]);
            }
            zs.push(z);
        }
        let q: usize = zs.iter().map(|z| z.len()).sum();
        let k = params.names.len();
        if q < k {
            return Err(rc_err(481, format!("model not identified: {q} moments for {k} parameters")));
        }
        let dsr = &*ds;
        let stt = &*st;
        let resid = |b: &[f64]| -> Result<Vec<Vec<f64>>, String> { eqs.iter().map(|e| nlest::values(dsr, stt, &e.expr, b, &rows)).collect() };
        let mom = |b: &[f64]| -> Result<Vec<Vec<f64>>, String> {
            let r = resid(b)?;
            Ok((0..n)
                .map(|t| {
                    let mut row = Vec::with_capacity(q);
                    for (e, z) in zs.iter().enumerate() {
                        for c in z {
                            row.push(r[e][t] * c[t]);
                        }
                    }
                    row
                })
                .collect())
        };
        // homoskedastic S: Sigma (x) Z'Z / N
        let s_unadj = |b: &[f64]| -> Result<Matrix, String> {
            let r = resid(b)?;
            let sw: f64 = w.iter().sum();
            let neq = eqs.len();
            let mut sig = Matrix::zeros(neq, neq);
            for a in 0..neq {
                for c in 0..neq {
                    sig[(a, c)] = (0..n).map(|t| w[t] * r[a][t] * r[c][t]).sum::<f64>() / sw;
                }
            }
            let mut s = Matrix::zeros(q, q);
            let mut ra = 0;
            for a in 0..neq {
                let mut rc = 0;
                for c in 0..neq {
                    for (i, za) in zs[a].iter().enumerate() {
                        for (j, zc) in zs[c].iter().enumerate() {
                            s[(ra + i, rc + j)] = sig[(a, c)] * (0..n).map(|t| w[t] * za[t] * zc[t]).sum::<f64>() / sw;
                        }
                    }
                    rc += zs[c].len();
                }
                ra += zs[a].len();
            }
            Ok(s)
        };
        let zz = || -> Matrix {
            let sw: f64 = w.iter().sum();
            let mut s = Matrix::zeros(q, q);
            let mut ra = 0;
            for a in 0..eqs.len() {
                for (i, za) in zs[a].iter().enumerate() {
                    for (j, zc) in zs[a].iter().enumerate() {
                        s[(ra + i, ra + j)] = (0..n).map(|t| w[t] * za[t] * zc[t]).sum::<f64>() / sw;
                    }
                }
                ra += zs[a].len();
            }
            s
        };
        let wspec = cmd.opt("wmatrix", 2).unwrap_or("robust").to_string();
        let wkind = skind_of(ds, &rows, &wspec)?;
        let winit = cmd.opt("winitial", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "unadjusted".into());
        let mut wm = if winit.starts_with("i") { Matrix::identity(q) } else { weight_of(&zz()) };
        let steps = if cmd.has_opt("onestep", 3) {
            1
        } else if cmd.has_opt("igmm", 4) {
            usize::MAX
        } else {
            2
        };
        let mut b = params.init.clone();
        let mut out = Vec::new();
        let mut step = 1;
        let mut qv;
        let mut conv;
        loop {
            out.push(format!("Step {step}"));
            let (nb, qq, log, c) = minimize(&mom, b.clone(), &wm, &w, 300)?;
            for (i, v) in log.iter().enumerate() {
                out.push(format!("Iteration {i}:   GMM criterion Q(b) = {}", fmt_g(*v, 9)));
            }
            out.push(String::new());
            let change = nb.iter().zip(&b).map(|(x, y)| (x - y).abs() / (y.abs() + 1.0)).fold(0.0, f64::max);
            b = nb;
            qv = qq;
            conv = c;
            if step >= steps || (steps == usize::MAX && step > 1 && change < 1e-8) || step > 200 {
                break;
            }
            let s = match &wkind {
                Some(kd) => s_matrix(&mom(&b)?, &w, kd),
                None => s_unadj(&b)?,
            };
            wm = weight_of(&s);
            step += 1;
        }
        // variance
        let vspec = cmd.opt("vce", 3).map(|s| s.to_string()).unwrap_or_else(|| if wspec.trim().is_empty() { "robust".into() } else { wspec.clone() });
        let vkind = skind_of(ds, &rows, &vspec)?;
        let gm = jac(&mom, &b, &w)?;
        let s = match &vkind {
            Some(kd) => s_matrix(&mom(&b)?, &w, kd),
            None => s_unadj(&b)?,
        };
        let sw: f64 = w.iter().sum();
        let v = gmm_v(&gm, &wm, &s, sw)?;
        let jstat = sw * qv;
        let level = util::level(cmd, st)?;
        let wlabel = |s: &str| -> String {
            let w = s.split_whitespace().next().unwrap_or("robust").to_lowercase();
            match w.as_str() {
                "unadjusted" | "un" => "Unadjusted".into(),
                "cluster" | "cl" => "Cluster".into(),
                "hac" => "HAC Bartlett".into(),
                _ => "Robust".into(),
            }
        };
        out.push("GMM estimation ".into());
        out.push(String::new());
        out.push(format!("Number of parameters = {:>3}", k));
        out.push(format!("Number of moments    = {:>3}", q));
        out.push(format!("{:<48}{:<17} = {:>10}", format!("Initial weight matrix: {}", if winit.starts_with('i') { "Identity" } else { "Unadjusted" }), "Number of obs", util::commas(n)));
        if steps > 1 {
            out.push(format!("GMM weight matrix:     {}", wlabel(&wspec)));
        }
        out.push(String::new());
        let disp = params.display();
        let no = vec![false; k];
        out.extend(crate::est::coef_table(&crate::est::TableSpec { depvar: "", names: &disp, b: &b, v: &v, omitted: &no, is_base: &no, df: None, level, eform: None, coef_title: "Coefficient" }));
        for (i, e) in eqs.iter().enumerate() {
            let mut list = e.inst.join(" ");
            if e.cons {
                list = format!("{list} _cons").trim().to_string();
            }
            if eqs.len() == 1 {
                out.push(format!("Instruments for equation 1: {list}"));
            } else {
                out.push(format!("Instruments for equation {}: {list}", i + 1));
            }
        }
        if !conv {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let stored = params.stored();
        let names = stored.clone();
        let mut sample = vec![false; ds.n_obs];
        for &r in &rows {
            sample[r] = true;
        }
        let estimator = if steps == 1 { "onestep" } else if steps == 2 { "twostep" } else { "igmm" };
        let vce_l = wlabel(&vspec).to_lowercase();
        let e = st.begin_e();
        e.set("N", sw);
        e.set("k", k as f64);
        e.set("n_moments", q as f64);
        e.set("n_eq", eqs.len() as f64);
        e.set("Q", qv);
        e.set("J", jstat);
        e.set("J_df", (q - k) as f64);
        e.set("rank", k as f64);
        e.set("converged", if conv { 1.0 } else { 0.0 });
        e.set_str("cmd", "gmm");
        e.set_str("estimator", estimator);
        e.set_str("wmatrix", wspec.trim());
        e.set_str("winit", if winit.starts_with('i') { "Identity" } else { "Unadjusted" });
        e.set_str("vce", vce_l.clone());
        e.set_str("vcetype", wlabel(&vspec));
        e.set_str("inst_1", eqs[0].inst.join(" "));
        e.set_str("properties", "b V");
        e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.set_mat("W", wm);
        e.sample = Some(sample);
        e.set_str("_output", text.clone());
        let _ = dist::normal;
        Ok(text)
    }
}

/// Unused helper to keep `Results` imported for future stored results.
#[allow(dead_code)]
fn _r(_: &Results) {}
