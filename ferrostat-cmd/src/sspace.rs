//! `sspace`: linear state-space models in covariance form estimated by
//! maximum likelihood through the Kalman filter, `constraint` (fixed-value
//! linear constraints used by the time-series estimators) and the shared
//! helpers that post state-space results (also used by dfactor and ucm).
//!
//!   (state L.states indepvars, state [noerror noconstant])
//!   (depvar states indepvars [, noerror noconstant])

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_kalman::{kfilter, Ssm};
use crate::ts_ml::{self, Post};
use crate::tsops;
use crate::util;

// ---------------------------------------------------------------------------
// constraint
// ---------------------------------------------------------------------------

/// `constraint [define] # spec`, `constraint drop #|_all`, `constraint list`.
pub struct Constraint;

impl CommandHandler for Constraint {
    fn name(&self) -> &'static str {
        "constraint"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let rest = raw.split_once(char::is_whitespace).map_or("", |x| x.1).trim();
        let (sub, tail) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let key = |n: &str| format!("_constraint.{n}");
        match sub {
            "drop" => {
                for w in tail.split_whitespace() {
                    if w == "_all" {
                        st.settings.retain(|k, _| !k.starts_with("_constraint."));
                    } else {
                        st.settings.remove(&key(w));
                    }
                }
                Ok(String::new())
            }
            "list" | "dir" | "" => {
                let mut v: Vec<(usize, String)> = st.settings.iter().filter_map(|(k, s)| k.strip_prefix("_constraint.").and_then(|n| n.parse().ok()).map(|n| (n, s.clone()))).collect();
                v.sort();
                Ok(v.iter().map(|(n, s)| format!("     {n}: {s}")).collect::<Vec<_>>().join("\n"))
            }
            "free" | "get" => Ok(String::new()),
            _ => {
                let (num, spec) = if sub == "define" || sub == "def" { tail.split_once(char::is_whitespace).unwrap_or((tail, "")) } else { (sub, tail) };
                let n: usize = num.parse().map_err(|_| rc_err(198, "constraint number required"))?;
                if spec.trim().is_empty() {
                    return Err(rc_err(198, "constraint specification required"));
                }
                st.settings.insert(key(&n.to_string()), spec.trim().to_string());
                Ok(String::new())
            }
        }
    }
}

/// Fixed-value constraints from `constraints(numlist)`: (equation, name, value).
pub fn fixed_constraints(st: &Store, cmd: &Command) -> Result<Vec<(String, String, f64)>, String> {
    let Some(list) = cmd.opt("constraints", 2) else { return Ok(vec![]) };
    let mut out = Vec::new();
    for n in ts_ml::parse_lags(list)? {
        let spec = st.settings.get(&format!("_constraint.{n}")).ok_or_else(|| rc_err(111, format!("constraint {n} not found")))?;
        let (lhs, rhs) = spec.split_once('=').ok_or_else(|| rc_err(198, format!("constraint {n}: only constraints of the form [eq]coef = # are supported")))?;
        let val: f64 = rhs.trim().parse().map_err(|_| rc_err(198, format!("constraint {n}: only constraints of the form [eq]coef = # are supported")))?;
        let lhs = lhs.trim().trim_start_matches("_b");
        let (eq, name) = if let Some(r) = lhs.strip_prefix('[') {
            let (e, nm) = r.split_once(']').ok_or_else(|| rc_err(198, "invalid constraint"))?;
            (e.trim().to_string(), nm.trim().to_string())
        } else {
            ("".to_string(), lhs.to_string())
        };
        out.push((eq, name, val));
    }
    Ok(out)
}

fn lag_eq(a: &str, b: &str) -> bool {
    let norm = |s: &str| -> String {
        let s = s.trim();
        match s.split_once('.') {
            Some((op, v)) => {
                let op = op.to_uppercase();
                let op = if op == "L1" { "L".to_string() } else { op };
                format!("{op}.{v}")
            }
            None => s.to_string(),
        }
    };
    norm(a) == norm(b)
}

/// Index of the parameter `eq:name` among `names`.
pub fn find_param(names: &[String], eq: &str, name: &str) -> Option<usize> {
    names.iter().position(|n| {
        let (e, nm) = n.split_once(':').unwrap_or(("", n));
        (eq.is_empty() || e == eq || e.trim_start_matches('/') == eq.trim_start_matches('/')) && (lag_eq(nm, name))
    })
}

/// Replace the rows of constrained coefficients with `(constrained)`.
pub fn mark_constrained(text: String, b: &[f64], v: &Matrix, fixed: &[bool]) -> String {
    if !fixed.iter().any(|f| *f) {
        return text;
    }
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let mut j = 0usize;
    let start = lines.iter().position(|l| l.contains("[95% conf. interval]") || l.contains("conf. interval]")).unwrap_or(0);
    let k = b.len();
    for l in lines.iter_mut().skip(start + 2) {
        if j >= k {
            break;
        }
        // coefficient rows have numbers after the bar
        let Some(bar) = l.find(" | ") else { continue };
        let after = l[bar + 3..].trim();
        if after.is_empty() {
            continue;
        }
        if fixed[j] && v[(j, j)] == 0.0 {
            *l = format!("{} | {:>10}  (constrained)", &l[..bar], fmt_g(b[j], 8));
        }
        j += 1;
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// sspace
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Eq {
    lhs: String,
    state: bool,
    error: bool,
    cons: bool,
    /// (term as written, kind): kind 0 lagged state (index), 1 state (index), 2 data variable
    terms: Vec<(String, usize, usize)>,
}

/// Split `(a b, opts) (c d)` into the parenthesized groups.
pub fn paren_groups(s: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' => {
                if depth > 0 {
                    cur.push(ch);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            _ if depth > 0 => cur.push(ch),
            c if c.is_whitespace() => {}
            _ => return Err(rc_err(198, "equations must be enclosed in parentheses")),
        }
    }
    if depth != 0 {
        return Err(rc_err(198, "unbalanced parentheses"));
    }
    Ok(out)
}

/// Data for a state-space estimation: observed series and covariates on
/// the estimation rows (time order).
pub struct SsData {
    pub rows: Vec<usize>,
    pub times: Vec<f64>,
    pub mask: Vec<bool>,
    pub tvar: String,
}

/// Rows with all `vars` nonmissing (and if/in), time ordered.
pub fn ss_rows(ds: &DataSet, st: &Store, cmd: &Command, cols: &[&Vec<f64>]) -> Result<SsData, String> {
    let mask0 = util::touse(ds, st, cmd)?;
    let (rows, times) = tsops::ordered_rows(ds, st, &mask0, cols)?;
    let mut mask = vec![false; ds.n_obs];
    for &r in &rows {
        mask[r] = true;
    }
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    Ok(SsData { rows, times, mask, tvar: tsops::time_var(st)? })
}

pub struct Sspace;

struct Model {
    eqs: Vec<Eq>,
    states: Vec<usize>,
    obs: Vec<usize>,
    names: Vec<String>,
    /// parameter index -> (eq index, term index or usize::MAX for _cons)
    coef_of: Vec<(usize, usize)>,
    n_coef: usize,
    /// covariance parameters: (state?, i, j) indices into states / obs
    cov: Vec<(bool, usize, usize)>,
}

fn build_model(ds: &DataSet, groups: &[String], cov_state: &str, cov_obs: &str) -> Result<Model, String> {
    let mut eqs = Vec::new();
    for g in groups {
        let (body, opts) = g.split_once(',').unwrap_or((g.as_str(), ""));
        let words: Vec<String> = tsops::expand_text(body.trim()).split_whitespace().map(String::from).collect();
        if words.is_empty() {
            return Err(rc_err(198, "empty equation"));
        }
        let o: Vec<String> = opts.split_whitespace().map(|w| w.to_lowercase()).collect();
        let has = |n: &str, min: usize| o.iter().any(|w| w.len() >= min && n.starts_with(w.as_str()));
        eqs.push(Eq { lhs: words[0].clone(), state: has("state", 5), error: !has("noerror", 5), cons: !has("noconstant", 4), terms: words[1..].iter().map(|w| (w.clone(), 2, 0)).collect() });
    }
    let states: Vec<usize> = (0..eqs.len()).filter(|&i| eqs[i].state).collect();
    let obs: Vec<usize> = (0..eqs.len()).filter(|&i| !eqs[i].state).collect();
    if states.is_empty() || obs.is_empty() {
        return Err(rc_err(198, "at least one state equation and one observation equation required"));
    }
    let sname: Vec<String> = states.iter().map(|&i| eqs[i].lhs.clone()).collect();
    for e in eqs.iter_mut() {
        for t in e.terms.iter_mut() {
            if let Some((op, v)) = t.0.split_once('.') {
                if let Some(si) = sname.iter().position(|s| s == v) {
                    let opu = op.to_uppercase();
                    if opu != "L" && opu != "L1" {
                        return Err(rc_err(198, "only first lags of states are allowed"));
                    }
                    if !e.state {
                        return Err(rc_err(198, "lagged states are not allowed in observation equations"));
                    }
                    *t = (format!("L.{v}"), 0, si);
                    continue;
                }
            }
            if let Some(si) = sname.iter().position(|s| *s == t.0) {
                if e.state {
                    return Err(rc_err(198, "contemporaneous states are not allowed in state equations"));
                }
                *t = (t.0.clone(), 1, si);
                continue;
            }
            t.0 = tsops::resolve(ds, &t.0)?;
        }
        if !e.state {
            e.lhs = tsops::resolve(ds, &e.lhs)?;
        }
    }
    let mut names = Vec::new();
    let mut coef_of = Vec::new();
    for (ei, e) in eqs.iter().enumerate() {
        for (ti, t) in e.terms.iter().enumerate() {
            names.push(format!("{}:{}", e.lhs, t.0));
            coef_of.push((ei, ti));
        }
        if e.cons {
            names.push(format!("{}:_cons", e.lhs));
            coef_of.push((ei, usize::MAX));
        }
    }
    let n_coef = names.len();
    let mut cov = Vec::new();
    let unstr = |s: &str| s.starts_with("un");
    let ident = |s: &str| s.starts_with("id");
    let err_s: Vec<usize> = (0..states.len()).filter(|&i| eqs[states[i]].error).collect();
    let err_o: Vec<usize> = (0..obs.len()).filter(|&i| eqs[obs[i]].error).collect();
    for (blk, errs, kind, set) in [(true, &err_s, cov_state, &states), (false, &err_o, cov_obs, &obs)] {
        let tag = if blk { "/state" } else { "/observable" };
        if ident(kind) {
            continue;
        }
        if unstr(kind) {
            for (a, &i) in errs.iter().enumerate() {
                for &j in errs.iter().take(a + 1) {
                    if i == j {
                        names.push(format!("{tag}:var({})", eqs[set[i]].lhs));
                    } else {
                        names.push(format!("{tag}:cov({},{})", eqs[set[j]].lhs, eqs[set[i]].lhs));
                    }
                    cov.push((blk, i, j));
                }
            }
        } else if kind.starts_with("ds") {
            // one common variance
            if !errs.is_empty() {
                names.push(format!("{tag}:var(e)"));
                cov.push((blk, usize::MAX, usize::MAX));
            }
        } else {
            for &i in errs.iter() {
                names.push(format!("{tag}:var({})", eqs[set[i]].lhs));
                cov.push((blk, i, i));
            }
        }
    }
    let _ = (&err_s, &err_o);
    Ok(Model { eqs, states, obs, names, coef_of, n_coef, cov })
}

impl Model {
    /// Measurement matrices for parameters th; xv[eq][term] are the data
    /// columns on the estimation rows.
    fn ssm(&self, th: &[f64], xv: &[Vec<Vec<f64>>], n: usize, cov_state: &str, cov_obs: &str, diffuse: bool) -> Option<Ssm> {
        let m = self.states.len();
        let p = self.obs.len();
        let mut tm = Matrix::zeros(m, m);
        let mut z = Matrix::zeros(p, m);
        let mut c = vec![vec![0.0; m]; n];
        let mut d = vec![vec![0.0; p]; n];
        for (j, &(ei, ti)) in self.coef_of.iter().enumerate() {
            let e = &self.eqs[ei];
            let row_s = self.states.iter().position(|&s| s == ei);
            let row_o = self.obs.iter().position(|&s| s == ei);
            if ti == usize::MAX {
                for t in 0..n {
                    if let Some(r) = row_s {
                        c[t][r] += th[j];
                    } else if let Some(r) = row_o {
                        d[t][r] += th[j];
                    }
                }
                continue;
            }
            let (_, kind, si) = &e.terms[ti];
            match kind {
                0 => tm[(row_s?, *si)] += th[j],
                1 => z[(row_o?, *si)] += th[j],
                _ => {
                    for t in 0..n {
                        let v = xv[ei][ti][t] * th[j];
                        if let Some(r) = row_s {
                            c[t][r] += v;
                        } else if let Some(r) = row_o {
                            d[t][r] += v;
                        }
                    }
                }
            }
        }
        let mut q = Matrix::zeros(m, m);
        let mut h = Matrix::zeros(p, p);
        if cov_state.starts_with("id") {
            for i in 0..m {
                if self.eqs[self.states[i]].error {
                    q[(i, i)] = 1.0;
                }
            }
        }
        if cov_obs.starts_with("id") {
            for i in 0..p {
                if self.eqs[self.obs[i]].error {
                    h[(i, i)] = 1.0;
                }
            }
        }
        for (k, &(blk, i, j)) in self.cov.iter().enumerate() {
            let v = th[self.n_coef + k];
            let mat = if blk { &mut q } else { &mut h };
            if i == usize::MAX {
                let set = if blk { &self.states } else { &self.obs };
                for a in 0..set.len() {
                    if self.eqs[set[a]].error {
                        mat[(a, a)] = v;
                    }
                }
            } else {
                mat[(i, j)] = v;
                mat[(j, i)] = v;
            }
        }
        // covariance matrices must be positive semidefinite
        for mat in [&q, &h] {
            let (ev, _) = mat.sym_eigen();
            if ev.iter().any(|v| *v < -1e-12) {
                return None;
            }
        }
        Some(Ssm { m, p, t: tm, rqr: q, z, h, c, d, diffuse: vec![diffuse; m] })
    }
}

impl CommandHandler for Sspace {
    fn name(&self) -> &'static str {
        "sspace"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "sspace");
        }
        let groups = paren_groups(&cmd.main)?;
        let cov_state = cmd.opt("covstate", 4).unwrap_or("diagonal").trim().to_lowercase();
        let cov_obs = cmd.opt("covobserved", 4).unwrap_or("diagonal").trim().to_lowercase();
        let model = build_model(ds, &groups, &cov_state, &cov_obs)?;
        // data columns
        let mut all_cols: Vec<Vec<f64>> = Vec::new();
        let mut col_ids: Vec<Vec<Option<usize>>> = Vec::new();
        for e in &model.eqs {
            let mut ids = Vec::new();
            for t in &e.terms {
                if t.1 == 2 {
                    all_cols.push(tsops::values(ds, st, &t.0)?);
                    ids.push(Some(all_cols.len() - 1));
                } else {
                    ids.push(None);
                }
            }
            col_ids.push(ids);
        }
        let ycols: Vec<Vec<f64>> = model.obs.iter().map(|&i| tsops::values(ds, st, &model.eqs[i].lhs)).collect::<Result<_, _>>()?;
        let refs: Vec<&Vec<f64>> = all_cols.iter().chain(ycols.iter()).collect();
        let sd = ss_rows(ds, st, cmd, &refs)?;
        let n = sd.rows.len();
        let xv: Vec<Vec<Vec<f64>>> = col_ids.iter().map(|ids| ids.iter().map(|id| id.map_or(vec![], |c| sd.rows.iter().map(|&r| all_cols[c][r]).collect())).collect()).collect();
        let y: Vec<Vec<Option<f64>>> = sd.rows.iter().map(|&r| ycols.iter().map(|c| Some(c[r])).collect()).collect();
        let diffuse = cmd.has_opt("diffuse", 3);
        let k = model.names.len();
        // constraints
        let cons = fixed_constraints(st, cmd)?;
        let mut fixed: Vec<Option<f64>> = vec![None; k];
        for (eq, nm, v) in &cons {
            let j = find_param(&model.names, eq, nm).ok_or_else(|| rc_err(111, format!("[{eq}]{nm} not found")))?;
            fixed[j] = Some(*v);
        }
        let free: Vec<usize> = (0..k).filter(|&j| fixed[j].is_none()).collect();
        let expand = |f: &[f64]| -> Vec<f64> {
            let mut th: Vec<f64> = fixed.iter().map(|v| v.unwrap_or(0.0)).collect();
            for (a, &j) in free.iter().enumerate() {
                th[j] = f[a];
            }
            th
        };
        let llf = |f: &[f64]| -> Option<Vec<f64>> {
            let th = expand(f);
            let s = model.ssm(&th, &xv, n, &cov_state, &cov_obs, diffuse)?;
            Some(kfilter(&s, &y)?.ll)
        };
        // starting values: lagged-state coefficients .5, state loadings 1, regressions by OLS-like means
        let yvar: Vec<f64> = (0..model.obs.len()).map(|i| util::variance(&y.iter().map(|r| r[i].unwrap()).collect::<Vec<_>>())).collect();
        let ymean: Vec<f64> = (0..model.obs.len()).map(|i| util::mean(&y.iter().map(|r| r[i].unwrap()).collect::<Vec<_>>())).collect();
        let mk = |rho: f64, share: f64| -> Vec<f64> {
            let mut th = vec![0.0; k];
            for (j, &(ei, ti)) in model.coef_of.iter().enumerate() {
                let e = &model.eqs[ei];
                th[j] = if ti == usize::MAX {
                    model.obs.iter().position(|&o| o == ei).map_or(0.0, |r| ymean[r])
                } else {
                    match e.terms[ti].1 {
                        0 => rho,
                        1 => 1.0,
                        _ => 0.0,
                    }
                };
            }
            let vbar = yvar.iter().sum::<f64>() / yvar.len() as f64;
            for (c, &(blk, i, jj)) in model.cov.iter().enumerate() {
                th[model.n_coef + c] = if i != jj { 0.0 } else if blk { vbar * share * (1.0 - rho * rho) } else { vbar * (1.0 - share) };
            }
            for j in 0..k {
                if let Some(v) = fixed[j] {
                    th[j] = v;
                }
            }
            free.iter().map(|&j| th[j]).collect()
        };
        let mut th0 = mk(0.5, 0.5);
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (r, s) in [(0.2, 0.5), (0.8, 0.5), (0.5, 0.9), (0.9, 0.9), (0.2, 0.9), (-0.3, 0.5)] {
            let c = mk(r, s);
            let l = ts_ml::ll_sum(&llf, &c);
            if l > best {
                best = l;
                th0 = c;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, free.len())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let tech = ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs);
        let opt = ts_ml::optimize(&llf, th0, maxiter, tech, "log likelihood", true)?;
        let (vf, vcetype, vce_name) = ts_ml::vce(&llf, &opt.th, cmd, "oim")?;
        let th = expand(&opt.th);
        let mut v = Matrix::zeros(k, k);
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                v[(i, j)] = vf[(a, b)];
            }
        }
        let depvars: Vec<String> = model.obs.iter().map(|&i| model.eqs[i].lhs.clone()).collect();
        let nobs_ll = n;
        let test_idx: Vec<usize> = (0..model.n_coef).filter(|&j| fixed[j].is_none() && !model.names[j].ends_with(":_cons")).collect();
        let (chi2, df) = ts_ml::wald(&th, &v, &test_idx);
        let samp = ts_ml::TsSample { tvar: sd.tvar.clone(), y: vec![], x: vec![], xnames: vec![], times: sd.times.clone(), rows: sd.rows.clone(), mask: sd.mask.clone(), depvar: String::new(), n_gaps: tsops::gaps(&sd.times) };
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let header = ts_ml::std_header(&sline, nobs_ll, chi2, df, opt.ll, "Log likelihood");
        let title = "State-space model";
        let fixedb: Vec<bool> = fixed.iter().map(|f| f.is_some()).collect();
        let mut notes = Vec::new();
        notes.push("Note: Tests of variances against zero are one sided, and the two-sided".into());
        notes.push("      confidence intervals are truncated at zero.".into());
        let text = ts_ml::post(
            st,
            cmd,
            Post { title, cmd: "sspace", depvar: "", names: model.names.clone(), b: th.clone(), v: v.clone(), ll: opt.ll, n: nobs_ll, header, vcetype, vce: vce_name, log: opt.log, notes, converged: opt.converged, iters: opt.iters, df: None },
            Some(&sd.mask),
        )?;
        let text = mark_constrained(text, &th, &v, &fixedb);
        let e = &mut st.e;
        e.set_str("_output", text.clone());
        e.set("k_aux", model.cov.len() as f64);
        e.set("k_eq", (model.eqs.len() + usize::from(model.cov.iter().any(|c| c.0)) + usize::from(model.cov.iter().any(|c| !c.0))) as f64);
        e.set("k_dv", depvars.len() as f64);
        e.set("k_state", model.states.len() as f64);
        e.set("k_obser", model.obs.len() as f64);
        e.set("df_m", df);
        e.set("chi2", chi2);
        e.set("p", if df > 0.0 { crate::dist::chi2tail(df, chi2) } else { SYSMISS });
        e.set("tmin", sd.times[0]);
        e.set("tmax", *sd.times.last().unwrap());
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("depvar", depvars.join(" "));
        e.set_str("obser_deps", depvars.join(" "));
        e.set_str("state_deps", model.states.iter().map(|&i| model.eqs[i].lhs.clone()).collect::<Vec<_>>().join(" "));
        e.set_str("covariates", "");
        e.set_str("method", "ml");
        e.set_str("initial_values", if diffuse { "diffuse" } else { "stationary" });
        e.set_str("chi2type", "Wald");
        e.set_str("predict", "sspace_p");
        e.set_str("_ss_cov", format!("{cov_state}|{cov_obs}"));
        e.set_str("_ss_groups", groups.join("\u{1}"));
        // the constrained coefficients have zero variance
        let _ = is_missing(0.0);
        Ok(text)
    }
}

/// `predict` after `sspace`: xb (one-step predictions of the observed
/// variables, default), residuals, states (filtered), mse.
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let groups: Vec<String> = e.macros.get("_ss_groups").map(|s| s.split('\u{1}').map(String::from).collect()).unwrap_or_default();
    let (cs, co) = e.macros.get("_ss_cov").and_then(|s| s.split_once('|')).map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or_default();
    let model = build_model(ds, &groups, &cs, &co)?;
    let th = b.data.clone();
    let diffuse = e.macros.get("initial_values").is_some_and(|s| s == "diffuse");
    let newvar = ts_ml::predict_name(cmd)?;
    let eqopt = cmd.opt("equation", 2).map(|s| s.trim().trim_start_matches('#').to_string());
    let stat = if cmd.has_opt("states", 5) {
        "states"
    } else if cmd.has_opt("residuals", 1) {
        "residuals"
    } else if cmd.has_opt("mse", 3) {
        "mse"
    } else {
        "xb"
    };
    let mut all_cols: Vec<Vec<f64>> = Vec::new();
    let mut col_ids: Vec<Vec<Option<usize>>> = Vec::new();
    for eq in &model.eqs {
        let mut ids = Vec::new();
        for t in &eq.terms {
            if t.1 == 2 {
                all_cols.push(tsops::values(ds, st, &t.0)?);
                ids.push(Some(all_cols.len() - 1));
            } else {
                ids.push(None);
            }
        }
        col_ids.push(ids);
    }
    let ycols: Vec<Vec<f64>> = model.obs.iter().map(|&i| tsops::values(ds, st, &model.eqs[i].lhs)).collect::<Result<_, _>>()?;
    let mask = util::touse(ds, st, cmd)?;
    let refs: Vec<&Vec<f64>> = all_cols.iter().collect();
    let tmin = e.get("tmin").unwrap_or(f64::NEG_INFINITY);
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let all = vec![true; ds.n_obs];
    let (rows0, _) = tsops::ordered_rows(ds, st, &all, &refs)?;
    let rows: Vec<usize> = rows0.into_iter().filter(|&r| tc[r] >= tmin).collect();
    let n = rows.len();
    let xv: Vec<Vec<Vec<f64>>> = col_ids.iter().map(|ids| ids.iter().map(|id| id.map_or(vec![], |c| rows.iter().map(|&r| all_cols[c][r]).collect())).collect()).collect();
    let y: Vec<Vec<Option<f64>>> = rows.iter().map(|&r| ycols.iter().map(|c| if is_missing(c[r]) { None } else { Some(c[r]) }).collect()).collect();
    let s = model.ssm(&th, &xv, n, &cs, &co, diffuse).ok_or_else(|| rc_err(498, "model could not be evaluated"))?;
    let kf = kfilter(&s, &y).ok_or_else(|| rc_err(498, "Kalman filter failed"))?;
    let pick = |names: Vec<String>| -> usize { eqopt.as_ref().and_then(|q| names.iter().position(|n| n == q).or_else(|| q.parse::<usize>().ok().map(|v| v.saturating_sub(1)))).unwrap_or(0) };
    let mut out = vec![SYSMISS; ds.n_obs];
    if stat == "states" {
        let j = pick(model.states.iter().map(|&i| model.eqs[i].lhs.clone()).collect());
        for (t, &r) in rows.iter().enumerate() {
            out[r] = kf.filt[t][j];
        }
    } else {
        let j = pick(model.obs.iter().map(|&i| model.eqs[i].lhs.clone()).collect());
        for (t, &r) in rows.iter().enumerate() {
            out[r] = match stat {
                "residuals" => y[t][j].map_or(SYSMISS, |v| v - kf.yhat[t][j]),
                "mse" => kf.fvar[t][j],
                _ => kf.yhat[t][j],
            };
        }
    }
    let _ = fmt_g;
    ts_ml::add_prediction(ds, &newvar, out, &mask, "")
}
