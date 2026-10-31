//! Posterior predictive simulation: `bayespredict`, `bayesreps` and
//! `bayesstats ppvalues`.
//!
//! The model is rebuilt from the stored specification on the estimation
//! sample, and outcomes are simulated for each retained draw.
//!
//! ```text
//! bayespredict newvar [, mean median std]          posterior predictive summary per observation
//! bayespredict lower upper, cri [clevel(#)]        credible interval of the predictions
//! bayespredict {_ysim} [{_mu}], saving(file[, replace])
//! bayespredict (name: @func({_ysim})) ..., saving(file[, replace])
//! bayesreps stub, nreps(#)                         replicated outcomes as new variables
//! bayesstats ppvalues (name: @func({_ysim})) ...   posterior predictive p-values
//! ```
//! Functions: mean, median, sd, var, min, max, sum, skewness, kurtosis.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;

use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::rng::Rng;
use crate::store::Store;
use crate::util;

use super::model::Model;
use super::post::draws_of;
use super::run::num9;
use super::stats;
use super::text;

/// The rebuilt model and the draws (full parameter vectors).
fn setup(ds: &DataSet, st: &Store) -> Result<(Model, Vec<Vec<f64>>), String> {
    let spec = st.e.macros.get("_spec").cloned().ok_or_else(|| rc_err(301, "last estimation results not bayes"))?;
    if spec.contains("_quad(") {
        return Err(rc_err(498, "posterior predictions are not available for models estimated with an approximate likelihood"));
    }
    let mut st2 = Store::default();
    st2.scalars = st.scalars.clone();
    st2.matrices = st.matrices.clone();
    st2.level = st.level;
    let built = super::bayesmh::build(ds, &st2, &spec)?;
    let m = built.model;
    let d = draws_of(&st.e)?;
    if d.names.len() != m.params.len() || d.names.iter().zip(&m.params).any(|(a, p)| *a != p.name) {
        return Err(rc_err(459, "data have changed since estimation; cannot rebuild the model"));
    }
    if m.liks.is_empty() || m.eqs.is_empty() || m.eqs[m.liks[0].eqs.first().copied().unwrap_or(0)].y.is_empty() {
        return Err(rc_err(198, "model has no outcome to predict"));
    }
    let n = d.ll.len();
    let rows: Vec<Vec<f64>> = (0..n).map(|r| d.cols.iter().map(|c| c[r]).collect()).collect();
    Ok((m, rows))
}

fn seed(st: &Store, o: &[(String, Option<String>)]) {
    if let Some(Some(s)) = o.iter().find(|(n, _)| text::abbrev(n, "rseed", 5)).map(|x| &x.1) {
        if let Ok(v) = s.trim().parse::<f64>() {
            st.rng.borrow_mut().reseed(v as u64);
        }
    }
}

fn split(rest: &str) -> (String, Vec<(String, Option<String>)>) {
    match text::find_top(rest, ',') {
        Some(i) => (rest[..i].trim().to_string(), text::options(&rest[i + 1..])),
        None => (rest.trim().to_string(), Vec::new()),
    }
}

fn has(o: &[(String, Option<String>)], full: &str, min: usize) -> bool {
    o.iter().any(|(n, _)| text::abbrev(n, full, min))
}

/// Simulated outcomes: draws x sample rows.
fn simulate_all(m: &Model, draws: &[Vec<f64>], rng: &mut Rng) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(draws.len());
    for th in draws {
        let mut row = Vec::with_capacity(m.n);
        for r in 0..m.n {
            row.push(m.simulate(0, th, r, rng).ok_or_else(|| rc_err(498, "predictions are not available for this likelihood"))?);
        }
        out.push(row);
    }
    Ok(out)
}

fn apply(f: &str, x: &[f64]) -> Result<f64, String> {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let m = stats::mean(x);
    let sd = stats::var(x).sqrt();
    Ok(match f {
        "mean" => m,
        "median" => stats::quantile(&s, 0.5),
        "sd" | "std" => sd,
        "var" | "variance" => sd * sd,
        "min" => s[0],
        "max" => s[s.len() - 1],
        "sum" => x.iter().sum(),
        "skewness" => x.iter().map(|v| ((v - m) / sd).powi(3)).sum::<f64>() / x.len() as f64,
        "kurtosis" => x.iter().map(|v| ((v - m) / sd).powi(4)).sum::<f64>() / x.len() as f64,
        o => return Err(rc_err(198, format!("@{o}() is not a supported statistic"))),
    })
}

/// `(label: @func({_ysim}))` items.
fn stat_items(spec: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for w in text::words(spec) {
        let inner = w.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(&w).trim().to_string();
        let (label, ex) = match text::find_top(&inner, ':') {
            Some(i) if !inner[..i].contains('{') && !inner[..i].contains('@') => (inner[..i].trim().to_string(), inner[i + 1..].trim().to_string()),
            _ => (String::new(), inner.clone()),
        };
        let ex = ex.trim();
        let f = ex.strip_prefix('@').ok_or_else(|| rc_err(198, format!("{ex}: expected @function({{_ysim}})")))?;
        let (fname, arg) = text::call(f);
        if !arg.unwrap_or_default().contains("_ysim") {
            return Err(rc_err(198, "only functions of {_ysim} are supported"));
        }
        let label = if label.is_empty() { fname.clone() } else { label };
        out.push((label, fname));
    }
    Ok(out)
}

fn observed(m: &Model) -> Vec<f64> {
    m.eqs[m.liks[0].eqs[0]].y.clone()
}

fn save(ds: DataSet, file: &str, replace: bool) -> Result<(), String> {
    if ferrostat_core::vfs::exists(file) && !replace {
        return Err(rc_err(602, format!("file {file} already exists")));
    }
    let mut buf = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut buf, &ds).map_err(|e| rc_err(693, e))?;
    ferrostat_core::vfs::write(file, &buf).map_err(|e| rc_err(603, e))
}

fn saving_of(o: &[(String, Option<String>)]) -> Option<(String, bool)> {
    let a = o.iter().find(|(n, _)| text::abbrev(n, "saving", 3))?.1.clone()?;
    let parts = text::split_top(&a, ',');
    let f = util::unquote(parts[0].trim());
    let f = if f.contains('.') { f } else { format!("{f}.dta") };
    Some((f, parts.iter().skip(1).any(|p| p.trim() == "replace")))
}

pub fn bayespredict(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (main, o) = split(rest);
    let (m, draws) = setup(ds, st)?;
    seed(st, &o);
    let main = main.trim();
    if main.starts_with('{') || main.starts_with('(') {
        let (file, replace) = saving_of(&o).ok_or_else(|| rc_err(198, "option saving() required"))?;
        let sims = {
            let mut rng = st.rng.borrow_mut();
            simulate_all(&m, &draws, &mut rng)?
        };
        let mut out = DataSet::new();
        let nd = draws.len();
        out.add_variable(util::new_double("_index", (1..=nd).map(|i| i as f64).collect())).map_err(|e| rc_err(110, e))?;
        if main.starts_with('{') {
            let want_mu = main.contains("_mu");
            let want_y = main.contains("_ysim") || !want_mu;
            if want_y {
                for r in 0..m.n {
                    out.add_variable(util::new_double(&format!("_ysim1_{}", r + 1), sims.iter().map(|s| s[r]).collect())).map_err(|e| rc_err(110, e))?;
                }
            }
            if want_mu {
                for r in 0..m.n {
                    out.add_variable(util::new_double(&format!("_mu1_{}", r + 1), draws.iter().map(|th| m.mu(0, th, r).unwrap_or(SYSMISS)).collect())).map_err(|e| rc_err(110, e))?;
                }
            }
        } else {
            for (label, f) in stat_items(main)? {
                let v = sims.iter().map(|s| apply(&f, s)).collect::<Result<Vec<_>, _>>()?;
                out.add_variable(util::new_double(&label, v)).map_err(|e| rc_err(110, e))?;
            }
        }
        out.n_obs = nd;
        save(out, &file, replace)?;
        st.e.set_str("predfile", &file);
        return Ok(format!("file {file} saved"));
    }
    // new variables
    let names: Vec<String> = main.split_whitespace().filter(|w| !matches!(*w, "double" | "float")).map(String::from).collect();
    if names.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    for n in &names {
        if ds.var_index.contains_key(n) {
            return Err(rc_err(110, format!("variable {n} already defined")));
        }
    }
    let sims = {
        let mut rng = st.rng.borrow_mut();
        simulate_all(&m, &draws, &mut rng)?
    };
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| m.mask.get(i).copied().unwrap_or(false)).collect();
    let level = o.iter().find(|(n, _)| text::abbrev(n, "clevel", 2)).and_then(|x| x.1.as_ref()).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(st.level);
    let percol = |f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
        let mut v = vec![SYSMISS; ds.n_obs];
        for (r, &i) in rows.iter().enumerate() {
            let x: Vec<f64> = sims.iter().map(|s| s[r]).collect();
            v[i] = f(&x);
        }
        v
    };
    if has(&o, "cri", 3) {
        if names.len() != 2 {
            return Err(rc_err(198, "cri requires two new variable names"));
        }
        let a = (1.0 - level / 100.0) / 2.0;
        let lo = percol(&|x| {
            let mut s = x.to_vec();
            s.sort_by(f64::total_cmp);
            stats::quantile(&s, a)
        });
        let hi = percol(&|x| {
            let mut s = x.to_vec();
            s.sort_by(f64::total_cmp);
            stats::quantile(&s, 1.0 - a)
        });
        ds.add_variable(util::new_double(&names[0], lo)).map_err(|e| rc_err(110, e))?;
        ds.add_variable(util::new_double(&names[1], hi)).map_err(|e| rc_err(110, e))?;
        return Ok(String::new());
    }
    let f = if has(&o, "median", 3) { "median" } else if has(&o, "std", 3) { "sd" } else { "mean" };
    let v = percol(&|x| apply(f, x).unwrap_or(SYSMISS));
    ds.add_variable(util::new_double(&names[0], v)).map_err(|e| rc_err(110, e))?;
    Ok(String::new())
}

pub fn bayesreps(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (main, o) = split(rest);
    let (m, draws) = setup(ds, st)?;
    seed(st, &o);
    let nreps = o.iter().find(|(n, _)| text::abbrev(n, "nreps", 4)).and_then(|x| x.1.as_ref()).and_then(|s| s.trim().parse::<usize>().ok()).ok_or_else(|| rc_err(198, "option nreps() required"))?;
    let stub = main.trim().trim_end_matches('*').to_string();
    if stub.is_empty() {
        return Err(rc_err(100, "new variable stub required"));
    }
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| m.mask.get(i).copied().unwrap_or(false)).collect();
    let mut rng = st.rng.borrow().clone();
    for k in 0..nreps {
        let d = ((rng.uniform() * draws.len() as f64) as usize).min(draws.len() - 1);
        let mut v = vec![SYSMISS; ds.n_obs];
        for (r, &i) in rows.iter().enumerate() {
            v[i] = m.simulate(0, &draws[d], r, &mut rng).unwrap_or(SYSMISS);
        }
        let name = format!("{stub}{}", k + 1);
        if ds.var_index.contains_key(&name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        ds.add_variable(util::new_double(&name, v)).map_err(|e| rc_err(110, e))?;
    }
    *st.rng.borrow_mut() = rng;
    Ok(String::new())
}

pub fn ppvalues(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let (main, o) = split(rest);
    // `using file` is accepted: the replicates are simulated again from the draws
    let main = match main.find(" using ") {
        Some(i) => main[..i].to_string(),
        None => main,
    };
    let (m, draws) = setup(ds, st)?;
    seed(st, &o);
    let items = stat_items(&main)?;
    if items.is_empty() {
        return Err(rc_err(198, "test quantities required, e.g. (mean: @mean({_ysim}))"));
    }
    let sims = {
        let mut rng = st.rng.borrow_mut();
        simulate_all(&m, &draws, &mut rng)?
    };
    let yobs = observed(&m);
    let mut out = vec![String::new(), "Posterior predictive summary".to_string(), format!("{:>47}{:<17}= {:>10}", "", "MCMC sample size", util::commas(draws.len())), String::new()];
    out.push("-".repeat(65));
    out.push(format!("{:>12} |      Mean   Std. dev.     E(T_obs)    P(T>=T_obs)", "T"));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(51)));
    let mut mat = Matrix::zeros(items.len(), 4);
    let mut labels = Vec::new();
    for (k, (label, f)) in items.iter().enumerate() {
        let t: Vec<f64> = sims.iter().map(|s| apply(f, s)).collect::<Result<_, _>>()?;
        let tobs = apply(f, &yobs)?;
        let p = t.iter().filter(|v| **v >= tobs).count() as f64 / t.len() as f64;
        let (mn, sd) = (stats::mean(&t), stats::var(&t).sqrt());
        out.push(format!("{:>12} | {:>9}  {:>9}   {:>10}     {:>10}", label, num9(mn), num9(sd), num9(tobs), format!("{p:.4}")));
        for (c, v) in [mn, sd, tobs, p].into_iter().enumerate() {
            mat[(k, c)] = v;
        }
        labels.push(label.clone());
    }
    out.push("-".repeat(65));
    out.push("Note: P(T>=T_obs) close to 0 or 1 indicates lack of fit.".into());
    let r = st.begin_r();
    r.set_mat("ppvalues", mat.with_names(labels, vec!["mean".into(), "std".into(), "E(T_obs)".into(), "P(T>=T_obs)".into()]));
    r.set("mcmcsize", draws.len() as f64);
    Ok(out.join("\n"))
}
