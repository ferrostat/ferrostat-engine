//! Seemingly unrelated estimation (`suest`): combines stored estimation
//! results into one parameter vector with a joint sandwich variance
//! V = D (S'S) D, where D is the block-diagonal matrix of the model-based
//! variances and S stacks each model's observation-level scores.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est;
use crate::eval::{rc_err, Env};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("suest", Box::new(Suest)), ("ivqregress", Box::new(Ivqregress))]
}

pub struct Suest;

/// One model's parameters, bread and observation scores.
struct Block {
    names: Vec<String>,
    b: Vec<f64>,
    d: Matrix,
    scores: Vec<Vec<f64>>,
}

fn block_of(ds: &DataSet, st: &Store, e: &Results, label: &str) -> Result<Block, String> {
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, format!("estimation results {label} not found")))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, format!("estimation results {label} not found")))?;
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let sample = e.sample.clone().ok_or_else(|| rc_err(322, format!("{label}: e(sample) not found")))?;
    if sample.len() != ds.n_obs {
        return Err(rc_err(322, format!("{label}: estimation sample does not match the data in memory")));
    }
    if !matches!(cmd.as_str(), "regress" | "logit" | "logistic" | "probit" | "poisson" | "cloglog") {
        return Err(rc_err(322, format!("{label}: suest is not supported after {cmd}")));
    }
    let env = Env::with_store(ds, st);
    let k = b.cols;
    let y = util::col(ds, &dep)?;
    let n = ds.n_obs;
    let mut xs: Vec<Option<Vec<f64>>> = vec![None; n];
    for i in 0..n {
        if sample[i] {
            let x: Vec<f64> = b.colnames.iter().map(|c| crate::postest::coef_value(&env, c, i)).collect::<Result<_, _>>()?;
            if x.iter().all(|v| !is_missing(*v)) {
                xs[i] = Some(x);
            }
        }
    }
    let xb = |x: &[f64]| -> f64 { x.iter().enumerate().map(|(j, v)| v * b.get(0, j)).sum() };
    // omitted columns have zero variance and are left out of the bread
    let mut d = v.clone();
    let mut names: Vec<String> = b.colnames.iter().map(|c| format!("{label}:{}", c.rsplit(':').next().unwrap_or(c))).collect();
    let mut bv: Vec<f64> = (0..k).map(|j| b.get(0, j)).collect();
    let mut scores = vec![Vec::new(); n];
    if cmd == "regress" {
        let nn = e.get("N").unwrap_or(0.0);
        let rss = e.get("rss").unwrap_or(0.0);
        let s2 = rss / nn;
        let df_r = e.get("df_r").unwrap_or(nn - k as f64);
        // model-based ML variance of the mean equation, then ln(sigma^2)
        let mut dd = Matrix::zeros(k + 1, k + 1);
        for a in 0..k {
            for c in 0..k {
                dd[(a, c)] = v[(a, c)] * df_r / nn;
            }
        }
        dd[(k, k)] = 2.0 / nn;
        d = dd;
        names = b.colnames.iter().map(|c| format!("{label}_mean:{}", c.rsplit(':').next().unwrap_or(c))).collect();
        names.push(format!("{label}_lnvar:_cons"));
        bv.push(s2.ln());
        for i in 0..n {
            if let Some(x) = &xs[i] {
                let r = y[i] - xb(x);
                let mut s: Vec<f64> = x.iter().map(|v| v * r / s2).collect();
                s.push(0.5 * (r * r / s2 - 1.0));
                scores[i] = s;
            }
        }
    } else {
        for i in 0..n {
            if let Some(x) = &xs[i] {
                let eta = xb(x);
                let yi = y[i];
                let g = match cmd.as_str() {
                    "logit" | "logistic" => (if yi != 0.0 { 1.0 } else { 0.0 }) - dist::invlogit(eta),
                    "probit" => {
                        if yi != 0.0 {
                            dist::normalden(eta) / xm::ncdf(eta).max(1e-300)
                        } else {
                            -dist::normalden(eta) / xm::ncdf(-eta).max(1e-300)
                        }
                    }
                    "cloglog" => {
                        let p = 1.0 - (-eta.exp()).exp();
                        let dp = eta.exp() * (-eta.exp()).exp();
                        if yi != 0.0 { dp / p.max(1e-300) } else { -eta.exp() }
                    }
                    _ => yi - eta.exp(),
                };
                scores[i] = x.iter().map(|v| v * g).collect();
            }
        }
    }
    Ok(Block { names, b: bv, d, scores })
}

impl CommandHandler for Suest {
    fn name(&self) -> &'static str {
        "suest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let names: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        if names.is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) == Some("suest") {
                return crate::r_trunc::replay(st, "suest");
            }
            return Err(rc_err(198, "estimation results required"));
        }
        let mut blocks = Vec::new();
        let mut labels = Vec::new();
        for nm in &names {
            let (e, label) = if nm == "." {
                (st.e.clone(), "_LAST".to_string())
            } else {
                let key = st.estimates.keys().find(|k| k.eq_ignore_ascii_case(nm)).cloned().ok_or_else(|| rc_err(111, format!("estimation result {nm} not found")))?;
                (st.estimates[&key].clone(), key)
            };
            blocks.push(block_of(ds, st, &e, &label)?);
            labels.push(label);
        }
        let n = ds.n_obs;
        let ktot: usize = blocks.iter().map(|b| b.b.len()).sum();
        let mut d = Matrix::zeros(ktot, ktot);
        let mut bvec = Vec::with_capacity(ktot);
        let mut allnames = Vec::with_capacity(ktot);
        let mut off = 0;
        for bl in &blocks {
            let k = bl.b.len();
            for a in 0..k {
                for c in 0..k {
                    d[(off + a, off + c)] = bl.d[(a, c)];
                }
            }
            bvec.extend_from_slice(&bl.b);
            allnames.extend(bl.names.iter().cloned());
            off += k;
        }
        // stacked scores; observations used by at least one model
        let used: Vec<bool> = (0..n).map(|i| blocks.iter().any(|bl| !bl.scores[i].is_empty())).collect();
        let stacked: Vec<Vec<f64>> = (0..n)
            .filter(|&i| used[i])
            .map(|i| {
                let mut s = Vec::with_capacity(ktot);
                for bl in &blocks {
                    if bl.scores[i].is_empty() {
                        s.extend(std::iter::repeat_n(0.0, bl.b.len()));
                    } else {
                        s.extend_from_slice(&bl.scores[i]);
                    }
                }
                s
            })
            .collect();
        let nused = stacked.len() as f64;
        let cl = match cmd.opt("vce", 1).map(|s| s.trim().to_string()) {
            Some(v) if v.to_lowercase().starts_with("cl") => Some(v.split_whitespace().nth(1).map(String::from).ok_or_else(|| rc_err(198, "vce(cluster clustvar) required"))?),
            _ => cmd.opt("cluster", 2).map(|s| s.trim().to_string()),
        };
        let (v, nclust) = match &cl {
            Some(c) => {
                let c = xm::var_name(ds, c)?;
                let col = util::col(ds, &c)?;
                let cv: Vec<f64> = (0..n).filter(|&i| used[i]).map(|i| col[i]).collect();
                let (m, g) = est::sandwich(&d, &stacked, Some(&cv), 1.0);
                let gf = g as f64;
                (m.scale(gf / (gf - 1.0)), Some((g, c)))
            }
            None => (est::sandwich(&d, &stacked, None, nused / (nused - 1.0)).0, None),
        };
        let level = util::level(cmd, st)?;
        let mut out = vec![String::new()];
        out.push(crate::choice_mixed::hrow(&format!("Simultaneous results for {}", labels.join(", ")), "Number of obs", util::commas(nused as usize)));
        if let Some((g, c)) = &nclust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(*g))));
        }
        out.push(String::new());
        let k = allnames.len();
        let omitted: Vec<bool> = (0..k).map(|j| v[(j, j)] == 0.0 && bvec[j] == 0.0).collect();
        let is_base: Vec<bool> = allnames.iter().map(|c| c.rsplit(':').next().unwrap_or(c).split('#').any(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
        let keep: Vec<usize> = (0..k).filter(|&j| !(cmd.has_opt("noomitted", 5) && omitted[j] && !is_base[j])).collect();
        let kn: Vec<String> = keep.iter().map(|&j| allnames[j].clone()).collect();
        let kb: Vec<f64> = keep.iter().map(|&j| bvec[j]).collect();
        let mut kv = Matrix::zeros(keep.len(), keep.len());
        for (a, &i) in keep.iter().enumerate() {
            for (c, &j) in keep.iter().enumerate() {
                kv[(a, c)] = v[(i, j)];
            }
        }
        let ko: Vec<bool> = keep.iter().map(|&j| omitted[j]).collect();
        let kbase: Vec<bool> = keep.iter().map(|&j| is_base[j]).collect();
        out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &kn, b: &kb, v: &kv, omitted: &ko, is_base: &kbase, df: None, level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("N", nused);
        e.set("rank", keep.len() as f64);
        e.set_str("cmd", "suest");
        e.set_str("names", labels.join(" "));
        e.set_str("title", "Simultaneous results");
        e.set_str("vce", if nclust.is_some() { "cluster" } else { "robust" });
        e.set_str("vcetype", "Robust");
        e.set_str("properties", "b V");
        if let Some((g, c)) = &nclust {
            e.set("N_clust", *g as f64);
            e.set_str("clustvar", c.clone());
        }
        e.set_mat("b", Matrix::row_vector(&kb).with_names(vec!["y1".into()], kn.clone()));
        e.set_mat("V", kv.with_names(kn.clone(), kn));
        e.sample = Some(used);
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// ivqregress
// ---------------------------------------------------------------------------

/// Quantile regression of y on the rows of X by iteratively reweighted
/// least squares.
pub fn rq(y: &[f64], x: &[Vec<f64>], tau: f64) -> Vec<f64> {
    let n = y.len();
    let k = x.first().map_or(0, |r| r.len());
    let solve = |w: &[f64]| -> Vec<f64> {
        let mut xtx = Matrix::zeros(k, k);
        let mut xty = vec![0.0; k];
        for t in 0..n {
            for a in 0..k {
                xty[a] += w[t] * x[t][a] * y[t];
                for c in 0..k {
                    xtx[(a, c)] += w[t] * x[t][a] * x[t][c];
                }
            }
        }
        let (inv, _) = xtx.sym_inverse(1e-14);
        (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * xty[c]).sum()).collect()
    };
    let mut b = solve(&vec![1.0; n]);
    let mut eps = 1e-2 * (y.iter().map(|v| v.abs()).sum::<f64>() / n.max(1) as f64).max(1e-8);
    for _ in 0..400 {
        let w: Vec<f64> = (0..n)
            .map(|t| {
                let r = y[t] - (0..k).map(|a| x[t][a] * b[a]).sum::<f64>();
                (if r >= 0.0 { tau } else { 1.0 - tau }) / r.abs().max(eps)
            })
            .collect();
        let nb = solve(&w);
        let ch = nb.iter().zip(&b).map(|(a, c)| (a - c).abs() / (1.0 + c.abs())).fold(0.0, f64::max);
        b = nb;
        if ch < 1e-10 {
            eps *= 0.1;
            if eps < 1e-12 {
                break;
            }
        }
    }
    b
}

/// Inverse of a square matrix by Gauss-Jordan elimination with pivoting.
pub fn ginv(m: &Matrix) -> Result<Matrix, String> {
    let k = m.rows;
    let mut a = m.clone();
    let mut inv = Matrix::identity(k);
    for c in 0..k {
        let p = (c..k).max_by(|&i, &j| a[(i, c)].abs().total_cmp(&a[(j, c)].abs())).unwrap();
        if a[(p, c)].abs() < 1e-300 {
            return Err(rc_err(506, "matrix not invertible"));
        }
        for j in 0..k {
            let (x, y) = (a[(c, j)], a[(p, j)]);
            a[(c, j)] = y;
            a[(p, j)] = x;
            let (x, y) = (inv[(c, j)], inv[(p, j)]);
            inv[(c, j)] = y;
            inv[(p, j)] = x;
        }
        let d = a[(c, c)];
        for j in 0..k {
            a[(c, j)] /= d;
            inv[(c, j)] /= d;
        }
        for i in 0..k {
            if i != c {
                let f = a[(i, c)];
                if f != 0.0 {
                    for j in 0..k {
                        a[(i, j)] -= f * a[(c, j)];
                        inv[(i, j)] -= f * inv[(c, j)];
                    }
                }
            }
        }
    }
    Ok(inv)
}

pub struct Ivqregress;

impl CommandHandler for Ivqregress {
    fn name(&self) -> &'static str {
        "ivqregress"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::r_trunc::replay(st, "ivqregress");
        }
        let (estimator, rest) = xm::first_word(&cmd.main);
        let estimator = estimator.to_lowercase();
        if !matches!(estimator.as_str(), "iqr" | "smooth") {
            return Err(rc_err(198, format!("estimator {estimator} not allowed; iqr or smooth required")));
        }
        let (outside, endog, inst) = xm::iv_parts(&rest)?;
        let (dep, exog) = xm::first_word(&outside);
        let dep = xm::var_name(ds, &dep)?;
        let tau = cmd.opt("quantile", 1).map(|s| s.trim().parse::<f64>().map_err(|_| rc_err(198, "quantile() invalid"))).transpose()?.unwrap_or(0.5);
        let tau = if tau >= 1.0 { tau / 100.0 } else { tau };
        let eqs = xm::build_eqs(ds, st, cmd, &[(Some(dep.as_str()), endog.as_str(), true), (None, exog.as_str(), false), (None, inst.as_str(), true)], "", &[])?;
        let (dd, dx, dz) = (&eqs[0], &eqs[1], &eqs[2]);
        let (ad, ax, az) = (dd.active(), dx.active(), dz.active());
        if ad.len() != az.len() {
            return Err(rc_err(481, "the number of excluded instruments must equal the number of endogenous regressors"));
        }
        if estimator == "iqr" && ad.len() != 1 {
            return Err(rc_err(198, "iqr allows one endogenous regressor"));
        }
        let n = dd.n();
        let y = dd.y.clone();
        let w_rows: Vec<Vec<f64>> = (0..n).map(|t| ad.iter().map(|&j| dd.cols[j][t]).chain(ax.iter().map(|&j| dx.cols[j][t])).collect()).collect();
        let z_rows: Vec<Vec<f64>> = (0..n).map(|t| az.iter().map(|&j| dz.cols[j][t]).chain(ax.iter().map(|&j| dx.cols[j][t])).collect()).collect();
        let k = w_rows.first().map_or(0, |r| r.len());
        let nd = ad.len();
        // 2SLS starting values
        let iv = |zr: &[Vec<f64>]| -> Vec<f64> {
            let mut zz = Matrix::zeros(k, k);
            let mut zw = Matrix::zeros(k, k);
            let mut zy = vec![0.0; k];
            for t in 0..n {
                for a in 0..k {
                    zy[a] += zr[t][a] * y[t];
                    for c in 0..k {
                        zz[(a, c)] += zr[t][a] * zr[t][c];
                        zw[(a, c)] += zr[t][a] * w_rows[t][c];
                    }
                }
            }
            let _ = zz;
            // just identified: (Z'W)^-1 Z'y
            let inv = ginv(&zw).unwrap_or_else(|_| Matrix::identity(k));
            (0..k).map(|a| (0..k).map(|c| inv[(a, c)] * zy[c]).sum()).collect()
        };
        let b2sls = iv(&z_rows);
        let resid = |b: &[f64]| -> Vec<f64> { (0..n).map(|t| y[t] - (0..k).map(|a| w_rows[t][a] * b[a]).sum::<f64>()).collect() };
        let sd = {
            let r = resid(&b2sls);
            let m = r.iter().sum::<f64>() / n as f64;
            (r.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
        };
        let h = 1.06 * sd * (n as f64).powf(-0.2);
        let mut out = vec![String::new()];
        let b: Vec<f64> = if estimator == "iqr" {
            // inverse quantile regression: the coefficient of the endogenous
            // variable makes the instrument's quantile-regression coefficient 0
            let gamma = |a: f64| -> (f64, Vec<f64>) {
                let yy: Vec<f64> = (0..n).map(|t| y[t] - a * w_rows[t][0]).collect();
                let c = rq(&yy, &z_rows, tau);
                (c[0], c)
            };
            let se0 = {
                // rough scale for the grid from the 2SLS fit
                let mut s = 0.0;
                let dm = w_rows.iter().map(|r| r[0]).sum::<f64>() / n as f64;
                for r in &w_rows {
                    s += (r[0] - dm).powi(2);
                }
                (sd / s.sqrt()).max(1e-8) * 4.0
            };
            let center = b2sls[0];
            let grid: Vec<f64> = (0..=30).map(|i| center - 5.0 * se0 + 10.0 * se0 * i as f64 / 30.0).collect();
            out.push(format!("Initial grid:\nQuantile = {tau:.2}: .........10.........20.........30 done"));
            let vals: Vec<f64> = grid.iter().map(|&a| gamma(a).0.abs()).collect();
            let mut best = 0;
            for i in 1..vals.len() {
                if vals[i] < vals[best] {
                    best = i;
                }
            }
            let (mut lo, mut hi) = (grid[best.saturating_sub(1)], grid[(best + 1).min(grid.len() - 1)]);
            out.push(format!("Adaptive grid:\nQuantile = {tau:.2}: .........10.........20.........30 done"));
            // golden-section refinement of |gamma(a)|
            let gr = 0.618_033_988_749_895;
            for _ in 0..80 {
                let c = hi - gr * (hi - lo);
                let d = lo + gr * (hi - lo);
                if gamma(c).0.abs() < gamma(d).0.abs() {
                    hi = d;
                } else {
                    lo = c;
                }
            }
            let a = 0.5 * (lo + hi);
            let (_, c) = gamma(a);
            let mut b = vec![a];
            b.extend_from_slice(&c[nd..]);
            b
        } else {
            // smoothed estimating equations, Gaussian integrated kernel
            let mut b = b2sls.clone();
            for _ in 0..100 {
                let r = resid(&b);
                let mut g = vec![0.0; k];
                let mut jm = Matrix::zeros(k, k);
                for t in 0..n {
                    let u = -r[t] / h;
                    let gg = tau - dist::normal(u);
                    let dens = dist::normalden(u) / h;
                    for a in 0..k {
                        g[a] += z_rows[t][a] * gg;
                        for c in 0..k {
                            jm[(a, c)] += z_rows[t][a] * w_rows[t][c] * dens;
                        }
                    }
                }
                let Ok(ji) = ginv(&jm) else { break };
                let step: Vec<f64> = (0..k).map(|a| (0..k).map(|c| ji[(a, c)] * g[c]).sum()).collect();
                for a in 0..k {
                    b[a] -= -step[a];
                }
                if step.iter().map(|s| s.abs()).fold(0.0, f64::max) < 1e-10 {
                    break;
                }
            }
            b
        };
        // robust variance: J^-1 S J^-1'
        let r = resid(&b);
        let mut jm = Matrix::zeros(k, k);
        let mut sm = Matrix::zeros(k, k);
        for t in 0..n {
            let dens = dist::normalden(r[t] / h) / h;
            for a in 0..k {
                for c in 0..k {
                    jm[(a, c)] += z_rows[t][a] * w_rows[t][c] * dens;
                    sm[(a, c)] += tau * (1.0 - tau) * z_rows[t][a] * z_rows[t][c];
                }
            }
        }
        let ji = ginv(&jm)?;
        let vact = ji.mul(&sm)?.mul(&ji.transpose())?;
        // full vectors over the design columns
        let mut names: Vec<String> = dd.names.clone();
        names.extend(dx.names.iter().cloned());
        let mut omitted: Vec<bool> = dd.omitted.clone();
        omitted.extend(dx.omitted.iter().cloned());
        let mut is_base: Vec<bool> = dd.is_base.clone();
        is_base.extend(dx.is_base.iter().cloned());
        let pos: Vec<usize> = ad.iter().cloned().chain(ax.iter().map(|&j| j + dd.k())).collect();
        let kf = names.len();
        let mut bf = vec![0.0; kf];
        for (a, &j) in pos.iter().enumerate() {
            bf[j] = b[a];
        }
        let v = est::expand_v(&vact, &pos, kf);
        let widx: Vec<usize> = pos.iter().cloned().filter(|&j| names[j] != "_cons").collect();
        let (chi2, df) = crate::heckman::wald(&bf, &v, &widx);
        let level = util::level(cmd, st)?;
        let title = if (tau - 0.5).abs() < 1e-12 { "IV median regression".to_string() } else { format!("IV quantile regression (quantile = {})", crate::format::fmt_g(tau, 8)) };
        let est_name = if estimator == "iqr" { "Inverse quantile regression" } else { "Smoothed estimating equations" };
        let left = vec![title.clone(), format!("Estimator: {est_name}"), String::new()];
        let right = vec![("Number of obs".to_string(), util::commas(n)), (format!("Wald chi2({})", crate::format::fmt_g(df, 9)), xm::f2(chi2)), ("Prob > chi2".to_string(), xm::f4(dist::chi2tail(df, chi2)))];
        out.push(String::new());
        out.extend(crate::r_trunc::header_new(&left, &right));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &bf, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
        let endn: Vec<String> = ad.iter().map(|&j| dd.names[j].clone()).collect();
        let exn: Vec<String> = ax.iter().filter(|&&j| dx.names[j] != "_cons").map(|&j| dx.names[j].clone()).chain(az.iter().map(|&j| dz.names[j].clone())).collect();
        out.push(format!("Endogenous: {}", endn.join(" ")));
        out.push(format!("Exogenous:  {}", exn.join(" ")));
        let text = out.join("\n");
        let mut dpost = dd.clone();
        dpost.names = names;
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        dpost.cols = vec![Vec::new(); kf];
        est::post(st, "ivqregress", &dpost, &bf, &v, &|e: &mut Results| {
            e.set("q", tau);
            e.set("chi2", chi2);
            e.set("df_m", df);
            e.set("p", dist::chi2tail(df, chi2));
            e.set("bwidth", h);
            e.set_str("estimator", estimator.clone());
            e.set_str("endog", endn.join(" "));
            e.set_str("exog", exn.join(" "));
            e.set_str("vce", "robust");
            e.set_str("vcetype", "Robust");
            e.set_str("title", title.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
