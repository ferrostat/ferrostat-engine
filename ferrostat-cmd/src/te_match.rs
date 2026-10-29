//! Matching estimators of treatment effects and their diagnostics:
//! `teffects nnmatch` (nearest-neighbor matching on covariates with exact
//! matching, bias adjustment and Abadie–Imbens robust standard errors),
//! `teffects psmatch` (matching on the estimated propensity score with the
//! Abadie–Imbens (2016) adjustment for the estimated score), `teffects
//! overlap`/`teoverlap` and `tebalance summarize|density|box|overid`.
//!
//! References: Abadie & Imbens (2006, Econometrica 74:235-267; 2011, JBES
//! 29:1-11; 2016, Econometrica 84:781-807); Imai & Ratkovic (2014, JRSS-B
//! 76:243-263) for the overidentification test.

use std::cell::RefCell;

use ferrostat_core::DataSet;
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::teffects::{self as te, cdf, design_rows, dot, effects_table, fit_binary, invert, level_label, model, pdf, score, sub_design, wrap12, Block, Link, Stat};
use crate::{util, varlist};

/// What the diagnostics need from the last treatment-effects estimation.
#[derive(Clone)]
pub(crate) struct TeState {
    /// data rows of the estimation sample
    pub rows: Vec<usize>,
    /// treatment level index of each row (0 = control)
    pub lev: Vec<usize>,
    pub lvals: Vec<f64>,
    /// estimated probabilities of each level (empty for nnmatch)
    pub ps: Vec<Vec<f64>>,
    /// balancing weights (IPW or matching frequency weights)
    pub w: Vec<f64>,
    /// variables of the treatment model (or matching covariates)
    pub tvars: Vec<String>,
    pub tx_terms: Vec<String>,
    pub link: Option<Link>,
    /// treatment-model design rows (with the constant, if any)
    pub xt: Vec<Vec<f64>>,
    pub atet: bool,
    pub trl: usize,
}

thread_local! {
    static STATE: RefCell<Option<TeState>> = const { RefCell::new(None) };
}

pub(crate) fn set_state(s: TeState) {
    STATE.with(|c| *c.borrow_mut() = Some(s));
}

pub(crate) fn clear_state() {
    STATE.with(|c| *c.borrow_mut() = None);
}

fn get_state(st: &Store) -> Result<TeState, String> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("teffects") {
        return Err(rc_err(301, "last estimates not found"));
    }
    STATE.with(|c| c.borrow().clone()).ok_or_else(|| rc_err(321, "requested action not valid after this estimator"))
}

/// Indices of the `m` nearest candidates (ties at the m-th distance included).
fn nearest(cands: &[usize], m: usize, d: &dyn Fn(usize) -> f64, caliper: Option<f64>) -> Vec<usize> {
    let mut ds: Vec<(f64, usize)> = cands.iter().map(|&j| (d(j), j)).collect();
    if let Some(c) = caliper {
        ds.retain(|x| x.0 <= c);
    }
    if ds.is_empty() {
        return Vec::new();
    }
    let m = m.min(ds.len());
    ds.select_nth_unstable_by(m - 1, |a, b| a.0.total_cmp(&b.0));
    let thr = ds[m - 1].0;
    let tol = thr.abs() * 1e-9 + 1e-13;
    let mut out: Vec<usize> = ds.iter().filter(|x| x.0 <= thr + tol).map(|x| x.1).collect();
    out.sort_unstable();
    out
}

/// `vce(iid)`: homoskedastic conditional variance.
fn vce_iid(cmd: &Command) -> bool {
    cmd.opt("vce", 3).is_some_and(|v| v.trim().starts_with("iid"))
}

/// Parse `vce(robust [, nn(#)])`: number of neighbours for the conditional variances.
fn vce_nn(cmd: &Command) -> Result<usize, String> {
    let Some(v) = cmd.opt("vce", 3) else { return Ok(2) };
    if let Some(i) = v.find("nn(") {
        let rest = &v[i + 3..];
        let num = rest.split(')').next().unwrap_or("").trim();
        return num.parse::<usize>().ok().filter(|&k| k >= 1).ok_or_else(|| rc_err(198, "vce(robust, nn(#)) requires a positive integer"));
    }
    if !v.trim().starts_with('r') && !v.trim().starts_with("iid") {
        return Err(rc_err(198, "only vce(robust) and vce(iid) are allowed"));
    }
    Ok(2)
}

/// Lower-triangular Cholesky factor.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 1e-14 * a[i][i].abs().max(1e-300) {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Transform covariates so that the matching metric becomes Euclidean.
fn whiten(x: &[Vec<f64>], metric: &str) -> Result<Vec<Vec<f64>>, String> {
    let n = x.len();
    let p = x.first().map_or(0, |r| r.len());
    if p == 0 {
        return Ok(x.to_vec());
    }
    let mean: Vec<f64> = (0..p).map(|a| x.iter().map(|r| r[a]).sum::<f64>() / n as f64).collect();
    let cov: Vec<Vec<f64>> = (0..p).map(|a| (0..p).map(|b| x.iter().map(|r| (r[a] - mean[a]) * (r[b] - mean[b])).sum::<f64>() / (n as f64 - 1.0)).collect()).collect();
    if metric.starts_with("maha") {
        let l = cholesky(&cov).ok_or_else(|| rc_err(506, "covariance matrix of the matching variables is not positive definite"))?;
        Ok(x.iter()
            .map(|r| {
                let mut zz = vec![0.0; p];
                for i in 0..p {
                    zz[i] = (r[i] - (0..i).map(|k| l[i][k] * zz[k]).sum::<f64>()) / l[i][i];
                }
                zz
            })
            .collect())
    } else if metric.starts_with("iva") {
        Ok(x.iter().map(|r| (0..p).map(|a| r[a] / cov[a][a].sqrt().max(1e-300)).collect()).collect())
    } else if metric.starts_with("euc") {
        Ok(x.to_vec())
    } else {
        Err(rc_err(198, format!("metric({metric}) not allowed")))
    }
}

/// Results of matching each unit to the opposite group.
struct Matched {
    /// matches in the opposite group (empty for controls under the ATET)
    sets: Vec<Vec<usize>>,
    /// within-group neighbours (excluding the unit itself)
    own: Vec<Vec<usize>>,
    /// imputed potential outcomes (index 0 = control, 1 = treated)
    yhat: [Vec<f64>; 2],
    k: Vec<f64>,
    kp: Vec<f64>,
    sig2: Vec<f64>,
    min_m: usize,
    max_m: usize,
}

fn sample_var(v: &[f64]) -> f64 {
    let h = v.len() as f64;
    if h < 2.0 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / h;
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (h - 1.0)
}

/// Matching core shared by nnmatch and psmatch: matched sets, the counts
/// K and K', conditional variances from within-group neighbours and the
/// imputed potential outcomes (bias-adjusted by regressions on `adj`
/// weighted by K, when given).
#[allow(clippy::too_many_arguments)]
fn do_match(y: &[f64], w: &[usize], keys: &[Vec<u64>], dist2: &dyn Fn(usize, usize) -> f64, m: usize, nnv: usize, caliper: Option<f64>, atet: bool, adj: Option<&[Vec<f64>]>) -> Result<Matched, String> {
    let n = y.len();
    let mut cells: std::collections::HashMap<(Vec<u64>, usize), Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        cells.entry((keys[i].clone(), w[i])).or_default().push(i);
    }
    let empty = Vec::new();
    let mut sets = vec![Vec::new(); n];
    let mut own = vec![Vec::new(); n];
    let mut k = vec![0.0; n];
    let mut kp = vec![0.0; n];
    let mut sig2 = vec![0.0; n];
    let (mut min_m, mut max_m) = (usize::MAX, 0);
    for i in 0..n {
        let oth = 1 - w[i];
        if !(atet && w[i] == 0) {
            let pool = cells.get(&(keys[i].clone(), oth)).unwrap_or(&empty);
            let js = nearest(pool, m, &|j| dist2(i, j), caliper);
            if js.is_empty() {
                return Err(rc_err(459, format!("no {} found for observation {}{}", if caliper.is_some() { "matches within caliper" } else { "exact matches" }, i + 1, if caliper.is_some() { "; caliper() is too small" } else { "" })));
            }
            min_m = min_m.min(js.len());
            max_m = max_m.max(js.len());
            let nj = js.len() as f64;
            for &j in &js {
                k[j] += 1.0 / nj;
                kp[j] += 1.0 / (nj * nj);
            }
            sets[i] = js;
        }
        // conditional variance: the unit and its nearest neighbours in its own group
        let pool = cells.get(&(keys[i].clone(), w[i])).unwrap_or(&empty);
        let others: Vec<usize> = pool.iter().copied().filter(|&j| j != i).collect();
        let hs = nearest(&others, nnv, &|j| dist2(i, j), None);
        let vals: Vec<f64> = std::iter::once(y[i]).chain(hs.iter().map(|&j| y[j])).collect();
        sig2[i] = sample_var(&vals);
        own[i] = hs;
    }
    if max_m == 0 {
        min_m = 0;
    }
    // bias adjustment: K-weighted regressions within each treatment group
    let beta: Option<[Vec<f64>; 2]> = match adj {
        None => None,
        Some(b) => {
            let fit = |g: usize| -> Result<Vec<f64>, String> {
                let kk = b[0].len() + 1;
                let mut xx = vec![vec![0.0; kk]; kk];
                let mut xy = vec![0.0; kk];
                for i in (0..n).filter(|&i| w[i] == g && k[i] > 0.0) {
                    let row: Vec<f64> = b[i].iter().copied().chain(std::iter::once(1.0)).collect();
                    for a in 0..kk {
                        xy[a] += k[i] * row[a] * y[i];
                        for c in 0..kk {
                            xx[a][c] += k[i] * row[a] * row[c];
                        }
                    }
                }
                let s = te::solve(&xx, &xy).ok_or_else(|| rc_err(459, "bias-adjustment regression is collinear"))?;
                Ok(s[..kk - 1].to_vec())
            };
            Some([fit(0)?, fit(1)?])
        }
    };
    let mut yhat = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        yhat[w[i]][i] = y[i];
        if sets[i].is_empty() {
            continue;
        }
        let oth = 1 - w[i];
        let mut s = 0.0;
        for &j in &sets[i] {
            let mut v = y[j];
            if let (Some(b), Some(beta)) = (adj, &beta) {
                v += (0..b[i].len()).map(|a| (b[i][a] - b[j][a]) * beta[oth][a]).sum::<f64>();
            }
            s += v;
        }
        yhat[oth][i] = s / sets[i].len() as f64;
    }
    Ok(Matched { sets, own, yhat, k, kp, sig2, min_m, max_m })
}

/// Abadie–Imbens variance of the ATE / ATET (point estimate first); with
/// `iid` the conditional variance is a single pooled estimate.
fn ai_variance(mt: &Matched, w: &[usize], y: &[f64], atet: bool, iid: bool) -> (f64, f64) {
    let n = y.len();
    let wt = |i: usize| -> bool { !atet || w[i] == 1 };
    let nn = (0..n).filter(|&i| wt(i)).count() as f64;
    let tau = (0..n).filter(|&i| wt(i)).map(|i| mt.yhat[1][i] - mt.yhat[0][i]).sum::<f64>() / nn;
    let s2: Vec<f64> = if iid {
        let mut acc = 0.0;
        for i in (0..n).filter(|&i| wt(i)) {
            let sg = if w[i] == 1 { 1.0 } else { -1.0 };
            let js = &mt.sets[i];
            acc += js.iter().map(|&j| (sg * (y[i] - y[j]) - tau).powi(2)).sum::<f64>() / js.len() as f64;
        }
        vec![acc / (2.0 * nn); n]
    } else {
        mt.sig2.clone()
    };
    let mut v = 0.0;
    for i in 0..n {
        if wt(i) {
            v += (mt.yhat[1][i] - mt.yhat[0][i] - tau).powi(2);
        }
        v += if atet { mt.k[i] * mt.k[i] - mt.kp[i] } else { mt.k[i] * mt.k[i] + 2.0 * mt.k[i] - mt.kp[i] } * s2[i];
    }
    (tau, v / (nn * nn))
}

fn float_key(v: f64) -> u64 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

/// Binary 0/1 treatment from a column (control = lower value).
fn binary_levels(t: &[f64], tvar: &str) -> Result<(Vec<usize>, [f64; 2]), String> {
    let mut vals: Vec<f64> = t.to_vec();
    vals.sort_by(f64::total_cmp);
    vals.dedup();
    if vals.len() != 2 {
        return Err(rc_err(459, format!("{tvar} must be a binary treatment variable")));
    }
    Ok((t.iter().map(|&v| usize::from(v == vals[1])).collect(), [vals[0], vals[1]]))
}

#[allow(clippy::too_many_arguments)]
fn report(ds: &DataSet, st: &mut Store, cmd: &Command, sub: &str, est_name: &str, extra_head: &[String], ovar: &str, tvar: &str, vals: [f64; 2], atet: bool, tau: f64, var: f64, n: usize, rows: &[usize], m: usize, min_m: usize, max_m: usize) -> Result<String, String> {
    let level = util::level(cmd, st)?;
    let eff = if atet { "ATET" } else { "ATE" };
    let mut out = Vec::new();
    out.push(format!("{:<48}Number of obs     = {:>10}", "Treatment-effects estimation", util::commas(n)));
    out.push(format!("{:<48}Matches: requested = {:>9}", format!("Estimator      : {est_name}"), m));
    out.push(format!("{:<48}{:>18} = {:>9}", "Outcome model  : matching", "min", min_m));
    let h3 = extra_head.first().cloned().unwrap_or_default();
    out.push(format!("{:<48}{:>18} = {:>9}", h3, "max", max_m));
    out.extend(extra_head.iter().skip(1).cloned());
    let blocks = [Block { title: eff.into(), rows: vec![(wrap12(&format!("({} vs {})", level_label(ds, tvar, vals[1]), level_label(ds, tvar, vals[0]))), tau, var)] }];
    effects_table(&mut out, ovar, tvar, "AI robust", level, &blocks);
    let fv = |v: f64| fmt_g(v, 9).trim().to_string();
    let names = vec![format!("{eff}:r{}vs{}.{tvar}", fv(vals[1]), fv(vals[0]))];
    let bm = Matrix::row_vector(&[tau]).with_names(vec!["y1".into()], names.clone());
    let mut vm = Matrix::zeros(1, 1);
    vm[(0, 0)] = var;
    let vm = vm.with_names(names.clone(), names);
    let mut sample = vec![false; ds.n_obs];
    for &r in rows {
        sample[r] = true;
    }
    let e = st.begin_e();
    e.set("N", n as f64);
    e.set("level", level);
    e.set("k_nneighbor", m as f64);
    e.set("k_nnmin", min_m as f64);
    e.set("k_nnmax", max_m as f64);
    e.set("control", vals[0]);
    e.set("treated", vals[1]);
    e.set_str("cmd", "teffects");
    e.set_str("subcmd", sub);
    e.set_str("stat", if atet { "atet" } else { "ate" });
    e.set_str("depvar", ovar);
    e.set_str("tvar", tvar);
    e.set_str("title", "Treatment-effects estimation");
    e.set_str("vce", "robust");
    e.set_str("vcetype", "AI robust");
    e.set_mat("b", bm);
    e.set_mat("V", vm);
    e.sample = Some(sample);
    Ok(out.join("\n"))
}

/// Matching frequency weights for balance diagnostics.
fn match_weights(mt: &Matched, w: &[usize], atet: bool) -> Vec<f64> {
    (0..w.len()).map(|i| if atet { if w[i] == 1 { 1.0 } else { mt.k[i] } } else { 1.0 + mt.k[i] }).collect()
}

/// `generate(stub)`-free option checks shared by the matching estimators.
fn match_opts(cmd: &Command) -> Result<(usize, Option<f64>, bool, usize), String> {
    let m = util::opt_num(cmd, "nneighbor", 2)?.unwrap_or(1.0);
    if m < 1.0 || m.fract() != 0.0 {
        return Err(rc_err(198, "nneighbor() must be a positive integer"));
    }
    let cal = util::opt_num(cmd, "caliper", 3)?;
    let stat = te::parse_stat(cmd);
    if stat == Stat::Pomeans {
        return Err(rc_err(198, "pomeans is not allowed with matching estimators"));
    }
    Ok((m as usize, cal, stat == Stat::Atet, vce_nn(cmd)?))
}

/// `teffects nnmatch (ovar omvarlist) (tvar) [, nneighbor() metric() biasadj() ematch() caliper() atet]`
pub(crate) fn nnmatch(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let groups: Vec<String> = split_top_level_words(rest).into_iter().filter(|w| w.starts_with('(')).collect();
    if groups.len() != 2 {
        return Err(rc_err(198, "teffects nnmatch requires an outcome model and a treatment model in parentheses"));
    }
    let (om, _) = model(&groups[0]);
    let (tm, _) = model(&groups[1]);
    let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
    let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
    if tm.len() > 1 {
        return Err(rc_err(198, "nnmatch: the treatment model takes no covariates"));
    }
    let (m, cal, atet, nnv) = match_opts(cmd)?;
    let ematch = match cmd.opt("ematch", 2) {
        Some(v) => varlist::expand(ds, v)?,
        None => Vec::new(),
    };
    let badj = match cmd.opt("biasadj", 4) {
        Some(v) => varlist::expand(ds, v)?,
        None => Vec::new(),
    };
    // distance covariates exclude the exactly matched variables
    let xs: Vec<String> = om[1..].iter().filter(|t| !ematch.iter().any(|e| e == *t)).cloned().collect();
    let mut extra = vec![tvar.clone()];
    extra.extend(ematch.iter().cloned());
    extra.extend(badj.iter().cloned());
    extra.extend(te::base_vars(ds, &om[1..])?);
    let d = sub_design(ds, st, cmd, &ovar, &xs, true, extra)?;
    let n = d.rows.len();
    if n < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let x = if xs.is_empty() { vec![Vec::new(); n] } else { design_rows(&d) };

    let tcol = util::col(ds, &tvar)?;
    let (w, vals) = binary_levels(&d.rows.iter().map(|&r| tcol[r]).collect::<Vec<_>>(), &tvar)?;
    let y = d.y.clone();
    let metric = cmd.opt("metric", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "mahalanobis".into());
    let z = whiten(&x, &metric)?;
    let ecols: Vec<Vec<f64>> = ematch.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let keys: Vec<Vec<u64>> = d.rows.iter().map(|&r| ecols.iter().map(|c| float_key(c[r])).collect()).collect();
    let dist2 = |i: usize, j: usize| -> f64 { z[i].iter().zip(&z[j]).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt() };
    let adj: Option<Vec<Vec<f64>>> = if badj.is_empty() {
        None
    } else {
        let bcols: Vec<Vec<f64>> = badj.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        Some(d.rows.iter().map(|&r| bcols.iter().map(|c| c[r]).collect()).collect())
    };
    let mt = do_match(&y, &w, &keys, &dist2, m, nnv, cal, atet, adj.as_deref())?;
    let (tau, var) = ai_variance(&mt, &w, &y, atet, vce_iid(cmd));
    let mname = if metric.starts_with("maha") {
        "Mahalanobis"
    } else if metric.starts_with("iva") {
        "inverse variance"
    } else {
        "Euclidean"
    };
    let head = vec![format!("Distance metric: {mname}")];
    // balance state: matching covariates
    let tvars = te::base_vars(ds, &om[1..])?;
    set_state(TeState {
        rows: d.rows.clone(),
        lev: w.clone(),
        lvals: vals.to_vec(),
        ps: Vec::new(),
        w: match_weights(&mt, &w, atet),
        tvars,
        tx_terms: om[1..].to_vec(),
        link: None,
        xt: Vec::new(),
        atet,
        trl: 1,
    });
    let rows = d.rows.clone();
    let out = report(ds, st, cmd, "nnmatch", "nearest-neighbor matching", &head, &ovar, &tvar, vals, atet, tau, var, n, &rows, m, mt.min_m, mt.max_m)?;
    if !badj.is_empty() {
        st.e.set_str("biasadj", badj.join(" "));
    }
    if !ematch.is_empty() {
        st.e.set_str("emvarlist", ematch.join(" "));
    }
    st.e.set_str("metric", metric);
    generate_matches(ds, cmd, &d.rows, &mt)?;
    Ok(out)
}

/// `generate(stub)`: not supported beyond a clear error.
fn generate_matches(_ds: &mut DataSet, cmd: &Command, _rows: &[usize], _mt: &Matched) -> Result<(), String> {
    if cmd.has_opt("generate", 3) {
        return Err(rc_err(198, "generate() is not supported; use psmatch2 to create match identifiers"));
    }
    Ok(())
}

/// `teffects psmatch (ovar) (tvar tmvarlist [, logit|probit]) [, nneighbor() caliper() atet]`
pub(crate) fn psmatch(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let groups: Vec<String> = split_top_level_words(rest).into_iter().filter(|w| w.starts_with('(')).collect();
    if groups.len() != 2 {
        return Err(rc_err(198, "teffects psmatch requires an outcome model and a treatment model in parentheses"));
    }
    let (om, _) = model(&groups[0]);
    let (tm, topts) = model(&groups[1]);
    let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
    let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
    if om.len() > 1 {
        return Err(rc_err(198, "psmatch: the outcome model takes no covariates"));
    }
    let tx: Vec<String> = tm[1..].to_vec();
    let link = if topts.contains("probit") { Link::Probit } else { Link::Logit };
    let (m, cal, atet, nnv) = match_opts(cmd)?;
    let mut extra = vec![ovar.clone()];
    extra.extend(te::base_vars(ds, &tx)?);
    let d = sub_design(ds, st, cmd, &tvar, &tx, topts.contains("nocons"), extra)?;
    let n = d.rows.len();
    if n < 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let (w, vals) = binary_levels(&d.y, &tvar)?;
    let x = design_rows(&d);
    let kx = x[0].len();
    let ycol = util::col(ds, &ovar)?;
    let y: Vec<f64> = d.rows.iter().map(|&r| ycol[r]).collect();
    let t01: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let g = fit_binary(&x, &t01, link)?;
    let idx: Vec<f64> = x.iter().map(|r| dot(r, &g)).collect();
    let p: Vec<f64> = idx.iter().map(|&z| cdf(link, z)).collect();
    let keys = vec![Vec::new(); n];
    let dist2 = |i: usize, j: usize| -> f64 { (p[i] - p[j]).abs() };
    let mt = do_match(&y, &w, &keys, &dist2, m, nnv, cal, atet, None)?;
    let iid = vce_iid(cmd);
    let (tau, v_ai) = ai_variance(&mt, &w, &y, atet, iid);
    // adjustment for the estimated propensity score (Abadie & Imbens 2016)
    let nf = n as f64;
    let f: Vec<f64> = idx.iter().map(|&z| pdf(link, z)).collect();
    let mut info = vec![vec![0.0; kx]; kx];
    for i in 0..n {
        let pi = p[i].clamp(1e-12, 1.0 - 1e-12);
        let wgt = f[i] * f[i] / (pi * (1.0 - pi));
        for a in 0..kx {
            for b in 0..kx {
                info[a][b] += wgt * x[i][a] * x[i][b];
            }
        }
    }
    // variance of the treatment-model coefficients
    let vg = invert(&info).ok_or_else(|| rc_err(459, "treatment-model information matrix is singular"))?;
    let groups_idx: [Vec<usize>; 2] = [(0..n).filter(|&i| w[i] == 0).collect(), (0..n).filter(|&i| w[i] == 1).collect()];
    let opp: Vec<Vec<usize>> = (0..n).map(|i| nearest(&groups_idx[1 - w[i]], nnv, &|j| (p[i] - p[j]).abs(), None)).collect();
    let cov_of = |hs: &[usize]| -> Vec<f64> {
        let h = hs.len() as f64;
        if h < 2.0 {
            return vec![0.0; kx];
        }
        let my = hs.iter().map(|&j| y[j]).sum::<f64>() / h;
        (0..kx)
            .map(|a| {
                let mx = hs.iter().map(|&j| x[j][a]).sum::<f64>() / h;
                hs.iter().map(|&j| (x[j][a] - mx) * (y[j] - my)).sum::<f64>() / (h - 1.0)
            })
            .collect()
    };
    // conditional covariances of the covariates and the outcome of each group
    let covs = |i: usize| -> [Vec<f64>; 2] {
        let own: Vec<usize> = std::iter::once(i).chain(mt.own[i].iter().copied()).collect();
        let co = cov_of(&own);
        let cp = cov_of(&opp[i]);
        if w[i] == 1 { [cp, co] } else { [co, cp] }
    };
    let quad = |c: &[f64]| -> f64 { (0..kx).map(|a| (0..kx).map(|b| c[a] * vg[a][b] * c[b]).sum::<f64>()).sum() };
    let var = if iid {
        v_ai
    } else if atet {
        let n1 = groups_idx[1].len() as f64;
        let mut c = vec![0.0; kx];
        let mut dt = vec![0.0; kx];
        // matches on the covariates for the derivative term
        let xz: Vec<Vec<f64>> = x.iter().map(|r| r.iter().enumerate().filter(|(a, _)| (0..n).any(|i| x[i][*a] != x[0][*a])).map(|(_, v)| *v).collect()).collect();
        let zz = whiten(&xz, "mahalanobis")?;
        let dz = |i: usize, j: usize| -> f64 { zz[i].iter().zip(&zz[j]).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() };
        let mean_y = |hs: &[usize]| -> f64 { if hs.is_empty() { 0.0 } else { hs.iter().map(|&j| y[j]).sum::<f64>() / hs.len() as f64 } };
        for i in 0..n {
            let [c0, c1] = covs(i);
            let pi = p[i].clamp(1e-12, 1.0 - 1e-12);
            let ytil_own = mean_y(&mt.own[i]);
            let ytil_opp = mean_y(&opp[i]);
            let (yt1, yt0) = if w[i] == 1 { (ytil_own, ytil_opp) } else { (ytil_opp, ytil_own) };
            let zm = nearest(&groups_idx[1 - w[i]], m, &|j| dz(i, j), None);
            let sg = if w[i] == 1 { 1.0 } else { -1.0 };
            let dterm = sg * (y[i] - mean_y(&zm)) - tau;
            for a in 0..kx {
                c[a] += f[i] * (x[i][a] * (yt1 - yt0 - tau) + c1[a] + pi / (1.0 - pi) * c0[a]) / n1;
                dt[a] += x[i][a] * f[i] * dterm / n1;
            }
        }
        v_ai - quad(&c) + quad(&dt)
    } else {
        let mut c = vec![0.0; kx];
        for i in 0..n {
            let [c0, c1] = covs(i);
            let pi = p[i].clamp(1e-12, 1.0 - 1e-12);
            for a in 0..kx {
                c[a] += f[i] * (c1[a] / pi + c0[a] / (1.0 - pi)) / nf;
            }
        }
        v_ai + quad(&c)
    };
    let tname = if link == Link::Logit { "logit" } else { "probit" };
    let head = vec![format!("Treatment model: {tname}")];
    set_state(TeState {
        rows: d.rows.clone(),
        lev: w.clone(),
        lvals: vals.to_vec(),
        ps: p.iter().map(|&v| vec![1.0 - v, v]).collect(),
        w: match_weights(&mt, &w, atet),
        tvars: te::base_vars(ds, &tx)?,
        tx_terms: tx.clone(),
        link: Some(link),
        xt: x.clone(),
        atet,
        trl: 1,
    });
    let rows = d.rows.clone();
    let out = report(ds, st, cmd, "psmatch", "propensity-score matching", &head, &ovar, &tvar, vals, atet, tau, var.max(0.0), n, &rows, m, mt.min_m, mt.max_m)?;
    st.e.set_str("tmodel", tname);
    st.e.set("V_unadj", v_ai);
    Ok(out)
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

fn wmean_var(x: &[f64], w: &[f64]) -> (f64, f64) {
    let sw: f64 = w.iter().sum();
    let m = x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let n_pos = w.iter().filter(|&&v| v > 0.0).count() as f64;
    let ss = x.iter().zip(w).map(|(a, b)| b * (a - m).powi(2)).sum::<f64>() / sw;
    // frequency-style correction as for aweights
    let v = if n_pos > 1.0 { ss * n_pos / (n_pos - 1.0) } else { 0.0 };
    (m, v)
}

/// Weighted kernel density (Epanechnikov, Silverman bandwidth) on a grid.
pub(crate) fn kde(x: &[f64], w: &[f64], lo: f64, hi: f64, npts: usize) -> Vec<(f64, f64)> {
    let sw: f64 = w.iter().sum();
    let (_, v) = wmean_var(x, w);
    let mut s: Vec<(f64, f64)> = x.iter().copied().zip(w.iter().copied()).collect();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    let q = |p: f64| -> f64 { util::pctile_weighted(&mut s.clone(), p) };
    let iqr = q(75.0) - q(25.0);
    let n_eff = x.len() as f64;
    let sd = v.sqrt();
    let spread = if iqr > 0.0 { sd.min(iqr / 1.349) } else { sd };
    let h = (0.9 * spread * n_eff.powf(-0.2)).max(1e-12) * 1.1;
    (0..npts)
        .map(|g| {
            let at = lo + (hi - lo) * g as f64 / (npts - 1).max(1) as f64;
            let dens = x.iter().zip(w).map(|(xi, wi)| wi * crate::kdensity::kernel("epanechnikov", (at - xi) / h)).sum::<f64>() / (sw * h);
            (at, dens)
        })
        .collect()
}

fn store_graph(mut g: Graph, cmd: &Command, st: &mut Store) -> Result<String, String> {
    graph::finish(&mut g, cmd, st)
}

/// `teffects overlap` / `teoverlap`: densities of the estimated probability
/// of one treatment level, by treatment group.
pub(crate) fn teoverlap(_ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = get_state(st)?;
    if s.ps.is_empty() {
        return Err(rc_err(321, "teoverlap requires an estimator with a treatment model"));
    }
    let tvar = st.e.macros.get("tvar").cloned().unwrap_or_default();
    let ptl = match util::opt_num(cmd, "ptlevel", 3)? {
        Some(v) => s.lvals.iter().position(|&l| l == v).ok_or_else(|| rc_err(198, "ptlevel() is not a treatment level"))?,
        None => s.trl,
    };
    let mut layers = Vec::new();
    let (lo, hi) = (0.0, 1.0);
    for l in 0..s.lvals.len() {
        let x: Vec<f64> = (0..s.rows.len()).filter(|&i| s.lev[i] == l).map(|i| s.ps[i][ptl]).collect();
        let w = vec![1.0; x.len()];
        let pts = kde(&x, &w, lo, hi, 200);
        layers.push(Layer { kind: LayerKind::Line, points: pts, label: format!("{tvar}={}", fmt_g(s.lvals[l], 9).trim()), ..Default::default() });
    }
    let g = Graph {
        title: String::new(),
        xtitle: format!("Propensity score, {tvar}={}", fmt_g(s.lvals[ptl], 9).trim()),
        ytitle: "Density".into(),
        layers,
        legend: true,
        ..Default::default()
    };
    store_graph(g, cmd, st)
}

pub struct Teoverlap;

impl CommandHandler for Teoverlap {
    fn name(&self) -> &'static str {
        "teoverlap"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        teoverlap(ds, st, cmd)
    }
}

pub struct Tebalance;

impl CommandHandler for Tebalance {
    fn name(&self) -> &'static str {
        "tebalance"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let s = get_state(st)?;
        if s.lvals.len() != 2 {
            return Err(rc_err(198, "tebalance requires a binary treatment"));
        }
        // covariate columns on the estimation sample
        let names: Vec<String> = if rest.trim().is_empty() { s.tvars.clone() } else { varlist::expand(ds, rest)? };
        let mut cols = Vec::new();
        for v in &names {
            let c = util::col(ds, v)?;
            cols.push(s.rows.iter().map(|&r| c[r]).collect::<Vec<f64>>());
        }
        // c.x#c.x style interaction terms of the treatment model
        let mut labels = names.clone();
        if rest.trim().is_empty() {
            for t in &s.tx_terms {
                if t.contains("##") || t.contains('#') {
                    let parts: Vec<&str> = t.split('#').filter(|p| !p.is_empty()).collect();
                    let vars: Vec<String> = parts.iter().map(|p| p.rsplit('.').next().unwrap_or(p).to_string()).collect();
                    if parts.iter().all(|p| p.starts_with("c.")) && vars.len() == 2 {
                        let a = util::col(ds, &vars[0])?;
                        let b = util::col(ds, &vars[1])?;
                        cols.push(s.rows.iter().map(|&r| a[r] * b[r]).collect());
                        labels.push(format!("c.{}#c.{}", vars[0], vars[1]));
                    }
                }
            }
        }
        let n = s.rows.len();
        let one = vec![1.0; n];
        let grp = |l: usize, wts: &[f64]| -> Vec<f64> { (0..n).map(|i| if s.lev[i] == l { wts[i] } else { 0.0 }).collect() };
        match sub {
            "summarize" | "sum" | "summ" | "summa" | "summar" | "summari" | "summariz" => {
                let mut out = Vec::new();
                out.push("  Covariate balance summary".to_string());
                out.push(String::new());
                let lab = if s.ps.is_empty() || st.e.macros.get("subcmd").is_some_and(|c| c == "psmatch") { "Matched" } else { "Weighted" };
                out.push(format!("{:>34}{:>12}", "Raw", lab));
                out.push("-".repeat(46));
                let cnt = |l: usize| (0..n).filter(|&i| s.lev[i] == l).count();
                let wsum = |l: usize| (0..n).filter(|&i| s.lev[i] == l).map(|i| s.w[i]).sum::<f64>();
                out.push(format!("Number of obs = {:>18}{:>12}", util::commas(n), format!("{:.1}", wsum(0) + wsum(1))));
                out.push(format!("Treated obs   = {:>18}{:>12}", util::commas(cnt(1)), format!("{:.1}", wsum(1))));
                out.push(format!("Control obs   = {:>18}{:>12}", util::commas(cnt(0)), format!("{:.1}", wsum(0))));
                out.push("-".repeat(46));
                out.push(String::new());
                out.push("-".repeat(65));
                out.push(format!("{:>16}|{:>25}{:>23}", "", "Standardized differences", "Variance ratio"));
                out.push(format!("{:>16}|{:>11}{:>12}{:>14}{:>12}", "", "Raw", lab, "Raw", lab));
                out.push(format!("{}+{}", "-".repeat(16), "-".repeat(48)));
                let mut table = Matrix::zeros(labels.len(), 4);
                for (k, (c, lab)) in cols.iter().zip(&labels).enumerate() {
                    let stats = |wts: &[f64]| -> (f64, f64) {
                        let (m1, v1) = wmean_var(c, &grp(1, wts));
                        let (m0, v0) = wmean_var(c, &grp(0, wts));
                        ((m1 - m0) / ((v1 + v0) / 2.0).sqrt(), v1 / v0)
                    };
                    let (sd_r, vr_r) = stats(&one);
                    let (sd_w, vr_w) = stats(&s.w);
                    table[(k, 0)] = sd_r;
                    table[(k, 1)] = sd_w;
                    table[(k, 2)] = vr_r;
                    table[(k, 3)] = vr_w;
                    out.push(format!("{:>15} |{:>11}{:>12}{:>14}{:>12}", crate::summarize::abbrev(lab, 15), fmt_g(sd_r, 7), fmt_g(sd_w, 7), fmt_g(vr_r, 7), fmt_g(vr_w, 7)));
                }
                out.push("-".repeat(65));
                let table = table.with_names(labels.clone(), vec!["std_diff:raw".into(), format!("std_diff:{}", lab.to_lowercase()), "ratio:raw".into(), format!("ratio:{}", lab.to_lowercase())]);
                let r = st.begin_r();
                r.set_mat("table", table);
                Ok(out.join("\n"))
            }
            "density" | "dens" | "box" => {
                if names.len() != 1 {
                    return Err(rc_err(198, format!("tebalance {sub} requires one variable")));
                }
                let c = &cols[0];
                let tvar = st.e.macros.get("tvar").cloned().unwrap_or_default();
                let mut children = Vec::new();
                for (title, wts) in [("Raw", &one), (if s.ps.is_empty() { "Matched" } else { "Weighted" }, &s.w)] {
                    let mut layers = Vec::new();
                    if sub == "box" {
                        let mut lay = Layer { kind: LayerKind::Box, width: 0.6, ..Default::default() };
                        for l in 0..2 {
                            let mut pairs: Vec<(f64, f64)> = (0..n).filter(|&i| s.lev[i] == l && wts[i] > 0.0).map(|i| (c[i], wts[i])).collect();
                            let q = |p: f64, v: &mut Vec<(f64, f64)>| util::pctile_weighted(v, p);
                            let q1 = q(25.0, &mut pairs);
                            let md = q(50.0, &mut pairs);
                            let q3 = q(75.0, &mut pairs);
                            let iqr = q3 - q1;
                            let lo = pairs.iter().map(|p| p.0).filter(|&v| v >= q1 - 1.5 * iqr).fold(f64::INFINITY, f64::min);
                            let hi = pairs.iter().map(|p| p.0).filter(|&v| v <= q3 + 1.5 * iqr).fold(f64::NEG_INFINITY, f64::max);
                            lay.boxes.push([q1, md, q3, lo, hi]);
                            lay.points.push((l as f64 + 0.5, md));
                        }
                        layers.push(lay);
                        children.push(Graph { title: title.into(), ytitle: names[0].clone(), xlabels: vec![(0.5, format!("{tvar}={}", fmt_g(s.lvals[0], 9).trim())), (1.5, format!("{tvar}={}", fmt_g(s.lvals[1], 9).trim()))], layers, ..Default::default() });
                    } else {
                        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        for l in 0..2 {
                            let idx: Vec<usize> = (0..n).filter(|&i| s.lev[i] == l && wts[i] > 0.0).collect();
                            let xs: Vec<f64> = idx.iter().map(|&i| c[i]).collect();
                            let ws: Vec<f64> = idx.iter().map(|&i| wts[i]).collect();
                            layers.push(Layer { kind: LayerKind::Line, points: kde(&xs, &ws, lo, hi, 200), label: if l == 1 { "treated".into() } else { "control".into() }, ..Default::default() });
                        }
                        children.push(Graph { title: title.into(), xtitle: names[0].clone(), ytitle: "Density".into(), layers, legend: true, ..Default::default() });
                    }
                }
                let g = Graph { title: "Balance plot".into(), children, cols: 2, ..Default::default() };
                store_graph(g, cmd, st)
            }
            "overid" => overid(st, &s),
            _ => Err(rc_err(198, format!("tebalance {sub}: unknown subcommand"))),
        }
    }
}

/// Imai–Ratkovic overidentification test: the score equations of the
/// treatment model plus covariate-balance conditions, by two-step GMM.
fn overid(st: &mut Store, s: &TeState) -> Result<String, String> {
    let link = s.link.ok_or_else(|| rc_err(321, "tebalance overid requires a binary treatment model"))?;
    let x = &s.xt;
    let n = x.len();
    let k = x[0].len();
    let t: Vec<f64> = s.lev.iter().map(|&l| l as f64).collect();
    let atet = s.atet;
    let gbar = |g: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut mean = vec![0.0; 2 * k];
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let z = dot(&x[i], g);
            let p = cdf(link, z).clamp(1e-10, 1.0 - 1e-10);
            let sc = score(link, t[i], z);
            let bal = if atet { t[i] - (1.0 - t[i]) * p / (1.0 - p) } else { t[i] / p - (1.0 - t[i]) / (1.0 - p) };
            let r: Vec<f64> = (0..k).map(|a| x[i][a] * sc).chain((0..k).map(|a| x[i][a] * bal)).collect();
            for a in 0..2 * k {
                mean[a] += r[a] / n as f64;
            }
            rows.push(r);
        }
        (mean, rows)
    };
    let mut g = fit_binary(x, &t, link)?;
    let mut wmat: Vec<Vec<f64>>;
    let obj = |g: &[f64], w: &[Vec<f64>]| -> f64 {
        let (m, _) = gbar(g);
        (0..2 * k).map(|a| (0..2 * k).map(|b| m[a] * w[a][b] * m[b]).sum::<f64>()).sum::<f64>()
    };
    // two steps: weight from the first-step estimate, then Gauss–Newton
    for _step in 0..2 {
        let (m, rows) = gbar(&g);
        let mut sm = vec![vec![0.0; 2 * k]; 2 * k];
        for r in &rows {
            for a in 0..2 * k {
                for b in 0..2 * k {
                    sm[a][b] += (r[a] - m[a]) * (r[b] - m[b]) / n as f64;
                }
            }
        }
        wmat = invert(&sm).ok_or_else(|| rc_err(459, "moment covariance is singular"))?;
        for _ in 0..50 {
            let (m, _) = gbar(&g);
            let mut jac = vec![vec![0.0; k]; 2 * k];
            for c in 0..k {
                let h = 1e-6 * g[c].abs().max(1e-3);
                let mut gp = g.clone();
                gp[c] += h;
                let (mp, _) = gbar(&gp);
                for a in 0..2 * k {
                    jac[a][c] = (mp[a] - m[a]) / h;
                }
            }
            let jw: Vec<Vec<f64>> = (0..k).map(|c| (0..2 * k).map(|b| (0..2 * k).map(|a| jac[a][c] * wmat[a][b]).sum()).collect()).collect();
            let h: Vec<Vec<f64>> = (0..k).map(|c| (0..k).map(|d| (0..2 * k).map(|b| jw[c][b] * jac[b][d]).sum()).collect()).collect();
            let gr: Vec<f64> = (0..k).map(|c| (0..2 * k).map(|b| jw[c][b] * m[b]).sum()).collect();
            let Some(step) = te::solve(&h, &gr) else { break };
            let before = obj(&g, &wmat);
            let mut lam = 1.0;
            let mut improved = false;
            for _ in 0..30 {
                let gn: Vec<f64> = g.iter().zip(&step).map(|(a, b)| a - lam * b).collect();
                if obj(&gn, &wmat) <= before {
                    g = gn;
                    improved = true;
                    break;
                }
                lam /= 2.0;
            }
            if !improved || step.iter().map(|v| v.abs()).fold(0.0, f64::max) * lam < 1e-9 {
                break;
            }
        }
        if _step == 1 {
            let j = n as f64 * obj(&g, &wmat);
            let df = k as f64;
            let p = dist::chi2tail(df, j);
            let mut out = Vec::new();
            out.push("Overidentification test for covariate balance".to_string());
            out.push(format!("         H0: Covariates are balanced:"));
            out.push(String::new());
            out.push(format!("         chi2({}) = {:>9.4}", k, j));
            out.push(format!("         Prob > chi2 = {:>9.4}", p));
            let r = st.begin_r();
            r.set("chi2", j);
            r.set("df", df);
            r.set("p", p);
            return Ok(out.join("\n"));
        }
    }
    Err(rc_err(430, "overidentification test: convergence not achieved"))
}
