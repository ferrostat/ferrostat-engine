//! `mgarch ccc|dcc|vcc|dvech`: multivariate GARCH models by maximum
//! likelihood (Gaussian or Student t errors).
//!
//! Each series has a conditional mean x_t b_i and, for ccc/dcc/vcc, a
//! univariate GARCH variance h_it = c_i + sum a_l e2_{i,t-l} + sum g_l h_{i,t-l};
//! the conditional covariance is H_t = D_t R_t D_t with
//!   ccc: R_t = R
//!   dcc: Q_t = (1 - l1 - l2) R + l1 u_{t-1} u_{t-1}' + l2 Q_{t-1},  R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
//!   vcc: R_t = (1 - l1 - l2) R + l1 Psi_{t-1} + l2 R_{t-1}, Psi the correlation of the last K standardized residuals
//! dvech: H_t = S + sum A_l o e_{t-l} e_{t-l}' + sum B_l o H_{t-l} (elementwise).
//! Presample squared residuals and variances are the sample covariance of
//! the residuals at the current parameters.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_ml::{self, parse_lags, Post, LN2PI};
use crate::tsops;
use crate::util;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ccc,
    Dcc,
    Vcc,
    Dvech,
}

struct Model {
    kind: Kind,
    k: usize,
    /// regressor counts per series
    kx: Vec<usize>,
    arch: Vec<usize>,
    garch: Vec<usize>,
    t_dist: bool,
}

impl Model {
    fn n_mean(&self) -> usize {
        self.kx.iter().sum()
    }
    fn n_uni(&self) -> usize {
        self.arch.len() + self.garch.len() + 1
    }
    fn n_corr(&self) -> usize {
        self.k * (self.k - 1) / 2
    }
    fn n_pairs(&self) -> usize {
        self.k * (self.k + 1) / 2
    }
    fn npar(&self) -> usize {
        let base = self.n_mean()
            + match self.kind {
                Kind::Dvech => self.n_pairs() * (1 + self.arch.len() + self.garch.len()),
                _ => self.k * self.n_uni() + self.n_corr() + if self.kind == Kind::Ccc { 0 } else { 2 },
            };
        base + usize::from(self.t_dist)
    }
}

/// log density of e ~ N(0, H) or multivariate t with df nu (scaled to covariance H).
fn mvlden(e: &[f64], h: &Matrix, nu: Option<f64>) -> Option<f64> {
    let k = e.len();
    let l = h.cholesky().ok()?;
    let mut logdet = 0.0;
    for i in 0..k {
        if !(l[(i, i)] > 0.0) {
            return None;
        }
        logdet += 2.0 * l[(i, i)].ln();
    }
    // solve L z = e
    let mut z = vec![0.0; k];
    for i in 0..k {
        let s: f64 = (0..i).map(|j| l[(i, j)] * z[j]).sum();
        z[i] = (e[i] - s) / l[(i, i)];
    }
    let q: f64 = z.iter().map(|v| v * v).sum();
    let kf = k as f64;
    Some(match nu {
        None => -0.5 * (kf * LN2PI + logdet + q),
        Some(n) => dist::lngamma((n + kf) / 2.0) - dist::lngamma(n / 2.0) - 0.5 * kf * ((n - 2.0) * std::f64::consts::PI).ln() - 0.5 * logdet - (n + kf) / 2.0 * (1.0 + q / (n - 2.0)).ln(),
    })
}

fn corr_matrix(k: usize, v: &[f64]) -> Matrix {
    let mut r = Matrix::identity(k);
    let mut c = 0;
    for j in 0..k {
        for i in j + 1..k {
            r[(i, j)] = v[c];
            r[(j, i)] = v[c];
            c += 1;
        }
    }
    r
}

/// Per-period log likelihood and conditional covariances.
fn evaluate(m: &Model, th: &[f64], y: &[Vec<f64>], x: &[Vec<Vec<f64>>]) -> Option<(Vec<f64>, Vec<Matrix>)> {
    let k = m.k;
    let n = y.len();
    // residuals
    let mut e = vec![vec![0.0; k]; n];
    let mut off = 0;
    for i in 0..k {
        for t in 0..n {
            e[t][i] = y[t][i] - ts_ml::dot(&x[i][t], &th[off..off + m.kx[i]]);
        }
        off += m.kx[i];
    }
    let mut s0 = Matrix::zeros(k, k);
    for t in 0..n {
        for i in 0..k {
            for j in 0..k {
                s0[(i, j)] += e[t][i] * e[t][j] / n as f64;
            }
        }
    }
    let nu = if m.t_dist {
        let v = th[th.len() - 1];
        if !(v > 2.0) {
            return None;
        }
        Some(v)
    } else {
        None
    };
    let mut ll = Vec::with_capacity(n);
    let mut hs: Vec<Matrix> = Vec::with_capacity(n);
    if m.kind == Kind::Dvech {
        let np = m.n_pairs();
        let pairs: Vec<(usize, usize)> = (0..k).flat_map(|j| (j..k).map(move |i| (i, j))).collect();
        let base = off;
        for t in 0..n {
            let mut h = Matrix::zeros(k, k);
            for (p, &(i, j)) in pairs.iter().enumerate() {
                let mut v = th[base + p];
                for (a, &l) in m.arch.iter().enumerate() {
                    v += th[base + np * (1 + a) + p] * if t >= l { e[t - l][i] * e[t - l][j] } else { s0[(i, j)] };
                }
                for (g, &l) in m.garch.iter().enumerate() {
                    v += th[base + np * (1 + m.arch.len() + g) + p] * if t >= l { hs[t - l][(i, j)] } else { s0[(i, j)] };
                }
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
            ll.push(mvlden(&e[t], &h, nu)?);
            hs.push(h);
        }
        return Some((ll, hs));
    }
    // univariate variances
    let mut hv = vec![vec![0.0; k]; n];
    for i in 0..k {
        let p = off + i * m.n_uni();
        for t in 0..n {
            let mut v = th[p + m.n_uni() - 1];
            for (a, &l) in m.arch.iter().enumerate() {
                v += th[p + a] * if t >= l { e[t - l][i].powi(2) } else { s0[(i, i)] };
            }
            for (g, &l) in m.garch.iter().enumerate() {
                v += th[p + m.arch.len() + g] * if t >= l { hv[t - l][i] } else { s0[(i, i)] };
            }
            if !(v > 0.0) || !v.is_finite() {
                return None;
            }
            hv[t][i] = v;
        }
    }
    let cbase = off + k * m.n_uni();
    let r = corr_matrix(k, &th[cbase..cbase + m.n_corr()]);
    let (l1, l2) = if m.kind == Kind::Ccc { (0.0, 0.0) } else { (th[cbase + m.n_corr()], th[cbase + m.n_corr() + 1]) };
    if m.kind != Kind::Ccc && (l1 <= -1.0 || l2 <= -1.0 || l1 + l2 >= 1.0) {
        return None;
    }
    let u: Vec<Vec<f64>> = (0..n).map(|t| (0..k).map(|i| e[t][i] / hv[t][i].sqrt()).collect()).collect();
    let mut q = r.clone();
    let mut rt_prev = r.clone();
    for t in 0..n {
        let rt = match m.kind {
            Kind::Ccc => r.clone(),
            Kind::Dcc => {
                if t > 0 {
                    let mut nq = Matrix::zeros(k, k);
                    for i in 0..k {
                        for j in 0..k {
                            nq[(i, j)] = (1.0 - l1 - l2) * r[(i, j)] + l1 * u[t - 1][i] * u[t - 1][j] + l2 * q[(i, j)];
                        }
                    }
                    q = nq;
                }
                let mut rr = Matrix::zeros(k, k);
                for i in 0..k {
                    for j in 0..k {
                        rr[(i, j)] = q[(i, j)] / (q[(i, i)] * q[(j, j)]).sqrt();
                    }
                }
                rr
            }
            _ => {
                // vcc
                if t > 0 {
                    let w = k.min(t);
                    let mut psi = Matrix::identity(k);
                    for i in 0..k {
                        for j in 0..i {
                            let (mut sij, mut sii, mut sjj) = (0.0, 0.0, 0.0);
                            for s in t - w..t {
                                sij += u[s][i] * u[s][j];
                                sii += u[s][i] * u[s][i];
                                sjj += u[s][j] * u[s][j];
                            }
                            let c = if sii > 0.0 && sjj > 0.0 { sij / (sii * sjj).sqrt() } else { 0.0 };
                            psi[(i, j)] = c;
                            psi[(j, i)] = c;
                        }
                    }
                    let mut nr = Matrix::zeros(k, k);
                    for i in 0..k {
                        for j in 0..k {
                            nr[(i, j)] = (1.0 - l1 - l2) * r[(i, j)] + l1 * psi[(i, j)] + l2 * rt_prev[(i, j)];
                        }
                    }
                    nr
                } else {
                    r.clone()
                }
            }
        };
        let mut h = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                h[(i, j)] = rt[(i, j)] * (hv[t][i] * hv[t][j]).sqrt();
            }
        }
        ll.push(mvlden(&e[t], &h, nu)?);
        hs.push(h);
        rt_prev = rt;
    }
    Some((ll, hs))
}

pub struct Mgarch;

struct Spec {
    deps: Vec<String>,
    xnames: Vec<Vec<String>>,
}

fn parse_eqs(ds: &DataSet, groups: &[String]) -> Result<Spec, String> {
    let mut deps = Vec::new();
    let mut xnames = Vec::new();
    for g in groups {
        let (body, opts) = g.split_once(',').unwrap_or((g.as_str(), ""));
        let (l, r) = body.split_once('=').unwrap_or((body, ""));
        let nocons = opts.to_lowercase().contains("nocons");
        let xs: Vec<String> = tsops::expand_text(r.trim()).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        for d in tsops::expand_text(l.trim()).split_whitespace() {
            deps.push(tsops::resolve(ds, d)?);
            let mut x = xs.clone();
            if !nocons {
                x.push("_cons".into());
            }
            xnames.push(x);
        }
    }
    if deps.len() < 2 {
        return Err(rc_err(198, "at least two dependent variables required"));
    }
    Ok(Spec { deps, xnames })
}

impl CommandHandler for Mgarch {
    fn name(&self) -> &'static str {
        "mgarch"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (model, rest) = main.split_once(|c: char| c.is_whitespace() || c == '(').map(|(a, _)| (a, &main[a.len()..])).unwrap_or((main, ""));
        if model.is_empty() {
            return crate::models::replay(st, "mgarch");
        }
        let kind = match model {
            "ccc" => Kind::Ccc,
            "dcc" => Kind::Dcc,
            "vcc" => Kind::Vcc,
            "dvech" => Kind::Dvech,
            o => return Err(rc_err(198, format!("unknown mgarch model {o}"))),
        };
        let groups = crate::sspace::paren_groups(rest.trim())?;
        let spec = parse_eqs(ds, &groups)?;
        let k = spec.deps.len();
        let arch = cmd.opt("arch", 4).map_or(Ok(vec![]), parse_lags)?;
        let garch = cmd.opt("garch", 5).map_or(Ok(vec![]), parse_lags)?;
        let t_dist = cmd.opt("distribution", 4).is_some_and(|d| d.trim().starts_with('t'));
        let ycols: Vec<Vec<f64>> = spec.deps.iter().map(|d| tsops::values(ds, st, d)).collect::<Result<_, _>>()?;
        let mut xcols: Vec<Vec<Vec<f64>>> = Vec::new();
        for xs in &spec.xnames {
            xcols.push(xs.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?);
        }
        let refs: Vec<&Vec<f64>> = ycols.iter().chain(xcols.iter().flatten()).collect();
        let sd = crate::sspace::ss_rows(ds, st, cmd, &refs)?;
        let n = sd.rows.len();
        let y: Vec<Vec<f64>> = sd.rows.iter().map(|&r| ycols.iter().map(|c| c[r]).collect()).collect();
        let x: Vec<Vec<Vec<f64>>> = xcols.iter().map(|cs| sd.rows.iter().map(|&r| cs.iter().map(|c| c[r]).collect()).collect()).collect();
        let m = Model { kind, k, kx: spec.xnames.iter().map(|v| v.len()).collect(), arch: arch.clone(), garch: garch.clone(), t_dist };
        if n <= m.npar() {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let llf = |th: &[f64]| evaluate(&m, th, &y, &x).map(|o| o.0);
        // starting values
        let mut th0 = Vec::new();
        let mut resid = vec![vec![0.0; k]; n];
        for i in 0..k {
            let b = ts_ml::lsq(&x[i], &y.iter().map(|r| r[i]).collect::<Vec<_>>());
            for t in 0..n {
                resid[t][i] = y[t][i] - ts_ml::dot(&x[i][t], &b);
            }
            th0.extend(b);
        }
        let var: Vec<f64> = (0..k).map(|i| resid.iter().map(|r| r[i] * r[i]).sum::<f64>() / n as f64).collect();
        let cov = |i: usize, j: usize| resid.iter().map(|r| r[i] * r[j]).sum::<f64>() / n as f64;
        let (a0, g0) = (if arch.is_empty() { 0.0 } else { 0.1 / arch.len() as f64 }, if garch.is_empty() { 0.0 } else { 0.8 / garch.len() as f64 });
        let persist = a0 * arch.len() as f64 + g0 * garch.len() as f64;
        if kind == Kind::Dvech {
            for j in 0..k {
                for i in j..k {
                    th0.push(cov(i, j) * (1.0 - persist));
                }
            }
            for _ in &arch {
                th0.extend(std::iter::repeat(a0).take(m.n_pairs()));
            }
            for _ in &garch {
                th0.extend(std::iter::repeat(g0).take(m.n_pairs()));
            }
        } else {
            for i in 0..k {
                th0.extend(arch.iter().map(|_| a0));
                th0.extend(garch.iter().map(|_| g0));
                th0.push(var[i] * (1.0 - persist));
            }
            for j in 0..k {
                for i in j + 1..k {
                    th0.push(cov(i, j) / (var[i] * var[j]).sqrt());
                }
            }
            if kind != Kind::Ccc {
                th0.push(0.05);
                th0.push(0.85);
            }
        }
        if t_dist {
            th0.push(10.0);
        }
        if let Some(f) = ts_ml::from_opt(cmd, m.npar())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let opt = ts_ml::optimize(&llf, th0, maxiter, ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs), "log likelihood", true)?;
        let th = opt.th.clone();
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "oim")?;
        // names
        let lag = |l: usize, w: &str| if l == 1 { format!("L.{w}") } else { format!("L{l}.{w}") };
        let mut names = Vec::new();
        for i in 0..k {
            names.extend(spec.xnames[i].iter().map(|nm| format!("{}:{nm}", spec.deps[i])));
        }
        if kind == Kind::Dvech {
            let pairs: Vec<(usize, usize)> = (0..k).flat_map(|j| (j..k).map(move |i| (i, j))).collect();
            names.extend(pairs.iter().map(|(i, j)| format!("/Sigma0:{}_{}", i + 1, j + 1)));
            for &l in &arch {
                names.extend(pairs.iter().map(|(i, j)| format!("{}:{}_{}", lag(l, "ARCH"), i + 1, j + 1)));
            }
            for &l in &garch {
                names.extend(pairs.iter().map(|(i, j)| format!("{}:{}_{}", lag(l, "GARCH"), i + 1, j + 1)));
            }
        } else {
            for i in 0..k {
                let eq = format!("ARCH_{}", spec.deps[i].rsplit('.').next().unwrap_or(&spec.deps[i]));
                names.extend(arch.iter().map(|&l| format!("{eq}:{}", lag(l, "arch"))));
                names.extend(garch.iter().map(|&l| format!("{eq}:{}", lag(l, "garch"))));
                names.push(format!("{eq}:_cons"));
            }
            for j in 0..k {
                for i in j + 1..k {
                    names.push(format!("/Correlation:corr({},{})", spec.deps[j], spec.deps[i]));
                }
            }
            if kind != Kind::Ccc {
                names.push("/Adjustment:lambda1".into());
                names.push("/Adjustment:lambda2".into());
            }
        }
        if t_dist {
            names.push("/:df".into());
        }
        let mean_idx: Vec<usize> = (0..m.n_mean()).filter(|&j| !names[j].ends_with(":_cons")).collect();
        let (chi2, df) = ts_ml::wald(&th, &v, &mean_idx);
        let samp = ts_ml::TsSample { tvar: sd.tvar.clone(), y: vec![], x: vec![], xnames: vec![], times: sd.times.clone(), rows: sd.rows.clone(), mask: sd.mask.clone(), depvar: String::new(), n_gaps: tsops::gaps(&sd.times) };
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let title = match kind {
            Kind::Ccc => "Constant conditional correlation MGARCH model",
            Kind::Dcc => "Dynamic conditional correlation MGARCH model",
            Kind::Vcc => "Varying conditional correlation MGARCH model",
            Kind::Dvech => "Diagonal vech MGARCH model",
        };
        let mut header = vec![crate::tsreg::hline(&sline, "Number of obs", &crate::format::fmt_g(n as f64, 10))];
        header.push(crate::tsreg::hline(&format!("Distribution: {}", if t_dist { "t" } else { "Gaussian" }), &format!("Wald chi2({})", if df > 0.0 { df.to_string() } else { ".".into() }), &crate::tsreg::f2(chi2)));
        header.push(crate::tsreg::hline(&format!("Log likelihood = {}", crate::format::fmt_g(opt.ll, 8)), "Prob > chi2", &crate::tsreg::f4(if df > 0.0 { dist::chi2tail(df, chi2) } else { SYSMISS })));
        let text = ts_ml::post(
            st,
            cmd,
            Post { title, cmd: "mgarch", depvar: "", names, b: th.clone(), v, ll: opt.ll, n, header, vcetype, vce: vce_name, log: opt.log, notes: vec![], converged: opt.converged, iters: opt.iters, df: None },
            Some(&sd.mask),
        )?;
        let e = &mut st.e;
        e.set_str("model", model);
        e.set_str("depvar", spec.deps.join(" "));
        e.set("k_dv", k as f64);
        e.set("chi2", chi2);
        e.set("df_m", df);
        e.set("tmin", sd.times[0]);
        e.set("tmax", *sd.times.last().unwrap());
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("dist", if t_dist { "t" } else { "gaussian" });
        e.set_str("arch", arch.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
        e.set_str("garch", garch.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
        e.set_str("predict", "mgarch_p");
        e.set_str("_mg", groups.join("\u{1}"));
        Ok(text)
    }
}

/// `predict` after mgarch: xb (default, equation()), variance (conditional variances/covariances, stub*).
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let model = e.macros.get("model").cloned().unwrap_or_default();
    let kind = match model.as_str() {
        "ccc" => Kind::Ccc,
        "dcc" => Kind::Dcc,
        "vcc" => Kind::Vcc,
        _ => Kind::Dvech,
    };
    let groups: Vec<String> = e.macros.get("_mg").map(|s| s.split('\u{1}').map(String::from).collect()).unwrap_or_default();
    let spec = parse_eqs(ds, &groups)?;
    let k = spec.deps.len();
    let lags = |s: &str| e.macros.get(s).map(|v| v.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let m = Model { kind, k, kx: spec.xnames.iter().map(|v| v.len()).collect(), arch: lags("arch"), garch: lags("garch"), t_dist: e.macros.get("dist").is_some_and(|d| d == "t") };
    let ycols: Vec<Vec<f64>> = spec.deps.iter().map(|d| tsops::values(ds, st, d)).collect::<Result<_, _>>()?;
    let mut xcols: Vec<Vec<Vec<f64>>> = Vec::new();
    for xs in &spec.xnames {
        xcols.push(xs.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?);
    }
    let refs: Vec<&Vec<f64>> = ycols.iter().chain(xcols.iter().flatten()).collect();
    let sample = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let (rows, _) = tsops::ordered_rows(ds, st, &sample, &refs)?;
    let y: Vec<Vec<f64>> = rows.iter().map(|&r| ycols.iter().map(|c| c[r]).collect()).collect();
    let x: Vec<Vec<Vec<f64>>> = xcols.iter().map(|cs| rows.iter().map(|&r| cs.iter().map(|c| c[r]).collect()).collect()).collect();
    let (_, hs) = evaluate(&m, &b.data, &y, &x).ok_or_else(|| rc_err(498, "model could not be evaluated"))?;
    let words: Vec<String> = cmd.main.split_whitespace().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).map(String::from).collect();
    let mask = util::touse(ds, st, cmd)?;
    if cmd.has_opt("variance", 3) {
        let stub = words.first().ok_or_else(|| rc_err(100, "new variable name required"))?.trim_end_matches('*').to_string();
        let mut msgs = Vec::new();
        for j in 0..k {
            for i in j..k {
                let nm = format!("{stub}_{}_{}", spec.deps[i].replace('.', "_"), spec.deps[j].replace('.', "_"));
                let mut v = vec![SYSMISS; ds.n_obs];
                for (t, &r) in rows.iter().enumerate() {
                    v[r] = hs[t][(i, j)];
                }
                msgs.push(ts_ml::add_prediction(ds, &nm, v, &mask, "")?);
            }
        }
        return Ok(msgs.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"));
    }
    let eqn = cmd.opt("equation", 2).map(|s| s.trim().to_string());
    let i = match &eqn {
        Some(q) => spec.deps.iter().position(|d| d == q).or_else(|| q.trim_start_matches('#').parse::<usize>().ok().map(|v| v.saturating_sub(1))).unwrap_or(0),
        None => 0,
    };
    let off: usize = spec.xnames[..i].iter().map(|v| v.len()).sum();
    let name = words.first().ok_or_else(|| rc_err(100, "new variable name required"))?.clone();
    let mut v = vec![SYSMISS; ds.n_obs];
    for r in 0..ds.n_obs {
        let vals: Vec<f64> = xcols[i].iter().map(|c| c[r]).collect();
        if vals.iter().all(|v| !ferrostat_core::missing::is_missing(*v)) {
            v[r] = ts_ml::dot(&vals, &b.data[off..off + spec.xnames[i].len()]);
        }
    }
    let resid = cmd.has_opt("residuals", 1);
    if resid {
        for r in 0..ds.n_obs {
            v[r] = if ferrostat_core::missing::is_missing(v[r]) || ferrostat_core::missing::is_missing(ycols[i][r]) { SYSMISS } else { ycols[i][r] - v[r] };
        }
    }
    ts_ml::add_prediction(ds, &name, v, &mask, if resid || cmd.has_opt("xb", 2) { "" } else { "(option xb assumed; fitted values)" })
}
