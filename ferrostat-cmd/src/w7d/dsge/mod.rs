//! Dynamic stochastic general-equilibrium models.
//!
//! `dsge` takes linear equations written in the model variables:
//!
//! ```text
//! dsge (p = {beta}*E(F.p) + {kappa}*y) (y = E(F.y) - (r - E(F.p) - z), unobserved)
//!      (r = 1/{beta}*p + u) (F.z = {rhoz}*z, state) (F.u = {rhou}*u, state)
//! ```
//!
//! An equation whose left-hand side is a plain variable defines a control;
//! `F.s = ..., state` defines a state, which receives a shock `e.s` unless
//! `noshock` is given. Controls are observed unless marked `unobserved`.
//!
//! `dsgenl` takes nonlinear equations `lhs = rhs` in levels and declares the
//! variables through `observed()`, `unobserved()`, `exostate()` (states
//! with shocks) and `endostate()`; the model is solved to first order
//! around its numerically computed steady state (`steadyinit()` sets the
//! starting values, 1 by default), and the observed series are treated as
//! deviations from the steady state.
//!
//! Both linearize to `A E x(t+1) = B x(t)` with x = (states, controls), are
//! solved for the stable saddle path (`cx::klein`): controls = G states,
//! states(t+1) = H states(t) + shocks, and are estimated by maximum
//! likelihood through the Kalman filter. Structural parameters start at
//! `{p=value}`, `from()` or 0.5; `constraints()` fixes parameters, and
//! `solve` evaluates the model at the starting values without estimating.

pub mod cx;
pub mod expr;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_kalman::{kfilter, stationary_cov, Ssm};
use crate::ts_ml;
use crate::util;
use expr::Ex;

#[derive(Clone, Debug)]
pub struct Model {
    pub nonlinear: bool,
    /// residual equations lhs - rhs
    pub eqs: Vec<(Ex, Ex)>,
    pub states: Vec<String>,
    pub controls: Vec<String>,
    /// states that receive a shock
    pub shocked: Vec<String>,
    pub observed: Vec<String>,
    pub params: Vec<String>,
    pub inits: Vec<(String, f64)>,
    pub steady_init: Vec<(String, f64)>,
}

impl Model {
    pub fn vars(&self) -> Vec<String> {
        self.states.iter().chain(&self.controls).cloned().collect()
    }
    pub fn ns(&self) -> usize {
        self.states.len()
    }
    pub fn n(&self) -> usize {
        self.states.len() + self.controls.len()
    }
    /// Names of the estimated parameters: structural, then shock sds.
    pub fn theta_names(&self) -> Vec<String> {
        let mut v = self.params.clone();
        v.extend(self.shocked.iter().map(|s| format!("/sd(e.{s})")));
        v
    }
}

fn split_eq(text: &str) -> Result<(Ex, Ex, Vec<(String, f64)>), String> {
    let (l, r) = text.split_once('=').ok_or_else(|| rc_err(198, format!("equation '{text}' has no '='")))?;
    // `{p=v}` contains '=': split at the first '=' outside braces
    let mut depth = 0;
    let mut pos = None;
    for (i, ch) in text.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            '=' if depth == 0 => {
                pos = Some(i);
                break;
            }
            _ => {}
        }
    }
    let (l, r) = match pos {
        Some(p) => (&text[..p], &text[p + 1..]),
        None => (l, r),
    };
    let (le, mut li) = expr::parse(l)?;
    let (re, ri) = expr::parse(r)?;
    li.extend(ri);
    Ok((le, re, li))
}

fn names_opt(cmd: &Command, name: &str, min: usize) -> Vec<String> {
    cmd.opt(name, min).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default()
}

/// Parse the equations of `dsge` / `dsgenl`.
pub fn parse_model(main: &str, cmd: &Command, nonlinear: bool) -> Result<Model, String> {
    let groups = crate::sspace::paren_groups(main)?;
    if groups.is_empty() {
        return Err(rc_err(198, "equations required"));
    }
    let mut m = Model { nonlinear, eqs: Vec::new(), states: Vec::new(), controls: Vec::new(), shocked: Vec::new(), observed: Vec::new(), params: Vec::new(), inits: Vec::new(), steady_init: Vec::new() };
    let mut allvars: Vec<String> = Vec::new();
    for g in &groups {
        let (body, opts) = match g.rsplit_once(',') {
            Some((b, o)) if !o.contains('=') && !o.contains('{') => (b.trim(), o.trim().to_lowercase()),
            _ => (g.trim(), String::new()),
        };
        let words: Vec<&str> = opts.split_whitespace().collect();
        let (lhs, rhs, inits) = split_eq(body)?;
        m.inits.extend(inits);
        let mut vs = Vec::new();
        let mut ps = Vec::new();
        lhs.collect(&mut vs, &mut ps);
        rhs.collect(&mut vs, &mut ps);
        for p in ps {
            if !m.params.contains(&p) {
                m.params.push(p);
            }
        }
        for v in vs {
            if !allvars.contains(&v) {
                allvars.push(v);
            }
        }
        if !nonlinear {
            let is_state = words.iter().any(|w| *w == "state");
            match (&lhs, is_state) {
                (Ex::Var(s, true), true) => {
                    if m.states.contains(s) || m.controls.contains(s) {
                        return Err(rc_err(198, format!("variable {s} defined by more than one equation")));
                    }
                    m.states.push(s.clone());
                    if !words.iter().any(|w| *w == "noshock") {
                        m.shocked.push(s.clone());
                    }
                }
                (Ex::Var(c, false), false) => {
                    if m.states.contains(c) || m.controls.contains(c) {
                        return Err(rc_err(198, format!("variable {c} defined by more than one equation")));
                    }
                    m.controls.push(c.clone());
                    if !words.iter().any(|w| w.starts_with("unobs")) {
                        m.observed.push(c.clone());
                    }
                }
                (_, true) => return Err(rc_err(198, format!("state equation '{body}' must have F.statevar on the left")) ),
                _ => return Err(rc_err(198, format!("equation '{body}' must have a control variable on the left")) ),
            }
        }
        m.eqs.push((lhs, rhs));
    }
    if nonlinear {
        m.observed = names_opt(cmd, "observed", 3);
        let unobs = names_opt(cmd, "unobserved", 5);
        let exo = names_opt(cmd, "exostate", 4);
        let endo = names_opt(cmd, "endostate", 5);
        m.controls = m.observed.iter().chain(&unobs).cloned().collect();
        m.states = exo.iter().chain(&endo).cloned().collect();
        m.shocked = exo;
        if let Some(s) = cmd.opt("steadyinit", 6) {
            let w: Vec<&str> = s.split_whitespace().collect();
            for p in w.chunks(2) {
                if p.len() == 2 {
                    m.steady_init.push((p[0].to_string(), p[1].parse().map_err(|_| rc_err(198, "steadyinit() requires pairs of variable and value"))?));
                }
            }
        }
    }
    let vars = m.vars();
    for v in &allvars {
        if !vars.contains(v) {
            return Err(rc_err(198, format!("variable {v} is not declared as a control or a state")));
        }
    }
    for v in &vars {
        if vars.iter().filter(|x| *x == v).count() > 1 {
            return Err(rc_err(198, format!("variable {v} declared more than once")));
        }
    }
    if m.eqs.len() != vars.len() {
        return Err(rc_err(198, format!("model has {} equations but {} variables", m.eqs.len(), vars.len())));
    }
    if m.shocked.is_empty() {
        return Err(rc_err(198, "model has no shocks"));
    }
    if m.observed.is_empty() {
        return Err(rc_err(198, "model has no observed control variables"));
    }
    Ok(m)
}

/// The model solved at a parameter vector.
#[derive(Clone, Debug)]
pub struct Solved {
    pub sol: cx::Solution,
    pub steady: Vec<f64>,
}

fn residuals(m: &Model, pars: &dyn Fn(&str) -> f64, lead: &[f64], cur: &[f64]) -> Vec<f64> {
    let vars = m.vars();
    let idx = |n: &str| vars.iter().position(|v| v == n).unwrap_or(0);
    let var = |n: &str, l: bool| if l { lead[idx(n)] } else { cur[idx(n)] };
    m.eqs.iter().map(|(l, r)| l.eval(&var, pars) - r.eval(&var, pars)).collect()
}

/// Steady state of a nonlinear model by damped Newton.
fn steady_state(m: &Model, pars: &dyn Fn(&str) -> f64) -> Result<Vec<f64>, String> {
    let vars = m.vars();
    let n = vars.len();
    if !m.nonlinear {
        return Ok(vec![0.0; n]);
    }
    let mut x: Vec<f64> = vars.iter().map(|v| m.steady_init.iter().find(|(n, _)| n == v).map_or(1.0, |p| p.1)).collect();
    let f = |x: &[f64]| residuals(m, pars, x, x);
    let norm = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut r = f(&x);
    for _ in 0..200 {
        let nr = norm(&r);
        if !nr.is_finite() {
            break;
        }
        if nr < 1e-12 {
            return Ok(x);
        }
        let j = crate::tsla::jacobian(&f, &x);
        let jm = Matrix::from_rows(&(0..n).map(|i| (0..n).map(|k| j[(i, k)]).collect()).collect::<Vec<_>>());
        let Ok(ji) = jm.inverse() else { break };
        let step: Vec<f64> = (0..n).map(|i| (0..n).map(|k| ji[(i, k)] * r[k]).sum()).collect();
        let mut lam = 1.0;
        let mut ok = false;
        for _ in 0..40 {
            let nx: Vec<f64> = x.iter().zip(&step).map(|(a, s)| a - lam * s).collect();
            let nr2 = f(&nx);
            if nr2.iter().all(|v| v.is_finite()) && norm(&nr2) < nr {
                x = nx;
                r = nr2;
                ok = true;
                break;
            }
            lam /= 2.0;
        }
        if !ok {
            break;
        }
    }
    if norm(&r) < 1e-8 {
        return Ok(x);
    }
    Err(rc_err(498, "steady state could not be found; use steadyinit() to give starting values"))
}

/// Linearize and solve the model at structural parameters `par`.
pub fn solve(m: &Model, par: &[f64]) -> Result<Solved, String> {
    let pf = |n: &str| m.params.iter().position(|p| p == n).map_or(f64::NAN, |i| par[i]);
    let steady = steady_state(m, &pf)?;
    let n = m.n();
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![vec![0.0; n]; n];
    for j in 0..n {
        let h = if m.nonlinear { 1e-6 * steady[j].abs().max(1e-3) } else { 1.0 };
        let mut lp = steady.clone();
        let mut lm = steady.clone();
        lp[j] += h;
        lm[j] -= h;
        let rp = residuals(m, &pf, &lp, &steady);
        let rm = residuals(m, &pf, &lm, &steady);
        for i in 0..n {
            a[i][j] = (rp[i] - rm[i]) / (2.0 * h);
        }
        let rp = residuals(m, &pf, &steady, &lp);
        let rm = residuals(m, &pf, &steady, &lm);
        for i in 0..n {
            b[i][j] = -(rp[i] - rm[i]) / (2.0 * h);
        }
    }
    if a.iter().chain(&b).flatten().any(|v| !v.is_finite()) {
        return Err(rc_err(498, "model could not be linearized at these parameter values"));
    }
    match cx::klein(&a, &b, m.ns()) {
        Ok(sol) => Ok(Solved { sol, steady }),
        Err(k) if k == usize::MAX => Err(rc_err(498, "model could not be solved: the linearized system is singular")),
        Err(k) => Err(rc_err(
            498,
            format!("model is not saddle-path stable: {k} stable eigenvalue{} for {} state variable{}", if k == 1 { "" } else { "s" }, m.ns(), if m.ns() == 1 { "" } else { "s" }),
        )),
    }
}

/// State-space form: transition H, shock loading (states x shocks) and the
/// observation matrix of the observed variables.
fn ssm(m: &Model, s: &Solved, sds: &[f64]) -> Ssm {
    let ns = m.ns();
    let t = Matrix::from_rows(&s.sol.h);
    let mut rqr = Matrix::zeros(ns, ns);
    for (k, sh) in m.shocked.iter().enumerate() {
        let i = m.states.iter().position(|x| x == sh).unwrap();
        rqr[(i, i)] = sds[k] * sds[k];
    }
    let p = m.observed.len();
    let mut z = Matrix::zeros(p, ns);
    for (r, o) in m.observed.iter().enumerate() {
        if let Some(i) = m.states.iter().position(|x| x == o) {
            z[(r, i)] = 1.0;
        } else if let Some(c) = m.controls.iter().position(|x| x == o) {
            for j in 0..ns {
                z[(r, j)] = s.sol.g[c][j];
            }
        }
    }
    Ssm { m: ns, p, t, rqr, z, h: Matrix::zeros(p, p), c: vec![], d: vec![], diffuse: vec![false; ns] }
}

/// Observed data in time order over the estimation sample.
struct Data {
    y: Vec<Vec<Option<f64>>>,
    mask: Vec<bool>,
    tvar: String,
    times: Vec<f64>,
}

fn data(ds: &DataSet, st: &Store, cmd: &Command, m: &Model) -> Result<Data, String> {
    let tvar = crate::tsops::time_var(st)?;
    let mut mask = util::touse(ds, st, cmd)?;
    for o in &m.observed {
        if !ds.var_index.contains_key(o) {
            return Err(rc_err(111, format!("variable {o} not found")));
        }
    }
    util::markout(ds, &mut mask, &m.observed);
    let tc = util::col(ds, &tvar)?;
    let cols: Vec<Vec<f64>> = m.observed.iter().map(|o| util::col(ds, o)).collect::<Result<_, _>>()?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(tc[i])).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    if rows.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    // fill gaps in time with missing observations
    let mut y = Vec::new();
    let mut times = Vec::new();
    let mut prev: Option<f64> = None;
    for &r in &rows {
        if let Some(p) = prev {
            let mut t = p + 1.0;
            while t < tc[r] - 0.5 {
                y.push(vec![None; m.observed.len()]);
                times.push(t);
                t += 1.0;
            }
        }
        y.push(cols.iter().map(|c| Some(c[r])).collect());
        times.push(tc[r]);
        prev = Some(tc[r]);
    }
    Ok(Data { y, mask, tvar, times })
}

/// Model, solution and parameters rebuilt from e().
pub struct Fitted {
    pub model: Model,
    pub theta: Vec<f64>,
    pub v: Matrix,
    pub free: Vec<bool>,
}

impl Fitted {
    pub fn par(&self) -> &[f64] {
        &self.theta[..self.model.params.len()]
    }
    pub fn sds(&self) -> &[f64] {
        &self.theta[self.model.params.len()..]
    }
}

pub fn fitted(st: &Store) -> Result<Fitted, String> {
    let e = &st.e;
    let ecmd = e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd != "dsge" && ecmd != "dsgenl" {
        return Err(rc_err(301, "last estimates not found"));
    }
    let main = e.macros.get("_dsge_main").cloned().unwrap_or_default();
    let opts = e.macros.get("_dsge_opts").cloned().unwrap_or_default();
    let line = format!("{ecmd} {main}{}", if opts.is_empty() { String::new() } else { format!(", {opts}") });
    let cmd = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
    let model = parse_model(&cmd.main, &cmd, ecmd == "dsgenl")?;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let v = e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
    let theta = b.data.clone();
    let free = (0..theta.len()).map(|j| v[(j, j)] > 0.0).collect();
    Ok(Fitted { model, theta, v, free })
}

pub struct Dsge {
    pub nonlinear: bool,
}

impl CommandHandler for Dsge {
    fn name(&self) -> &'static str {
        if self.nonlinear { "dsgenl" } else { "dsge" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        let model = parse_model(&cmd.main, cmd, self.nonlinear)?;
        let names = model.theta_names();
        let np = model.params.len();
        let k = names.len();
        let d = data(ds, st, cmd, &model)?;
        // starting values
        let mut th: Vec<f64> = model.params.iter().map(|p| model.inits.iter().rev().find(|(n, _)| n == p).map_or(0.5, |x| x.1)).collect();
        for o in &model.shocked {
            let _ = o;
            th.push(1.0);
        }
        if let Some(f) = cmd.opt("from", 2) {
            let toks: Vec<&str> = f.split_whitespace().collect();
            for t in toks {
                if let Some((n, v)) = t.split_once('=') {
                    let n = n.trim().trim_start_matches("_b[").trim_end_matches(']');
                    let val: f64 = v.parse().map_err(|_| rc_err(198, "invalid from()"))?;
                    match names.iter().position(|x| x == n || x.trim_start_matches('/') == n) {
                        Some(j) => th[j] = val,
                        None => return Err(rc_err(198, format!("from(): parameter {n} not in model"))),
                    }
                }
            }
        }
        // initial shock sds: sample sd of the first observed series
        if cmd.opt("from", 2).is_none() {
            let v: Vec<f64> = d.y.iter().filter_map(|r| r[0]).collect();
            let sd = util::variance(&v).sqrt();
            if sd.is_finite() && sd > 0.0 {
                for t in th.iter_mut().skip(np) {
                    *t = sd;
                }
            }
        }
        let mut fixed = vec![false; k];
        for (eqn, n, val) in crate::sspace::fixed_constraints(st, cmd)? {
            let n = if n.is_empty() { eqn } else if eqn.is_empty() || eqn == "_" { n } else { format!("/{eqn}") };
            let n2 = n.trim_start_matches('/');
            let j = names.iter().position(|x| *x == n || x.trim_start_matches('/') == n2).ok_or_else(|| rc_err(198, format!("constraint on {n}: parameter not in model")))?;
            th[j] = val;
            fixed[j] = true;
        }
        let free: Vec<usize> = (0..k).filter(|&j| !fixed[j]).collect();
        let full = |x: &[f64]| -> Vec<f64> {
            let mut t = th.clone();
            for (a, &j) in free.iter().enumerate() {
                t[j] = x[a];
            }
            t
        };
        let llf = |x: &[f64]| -> Option<Vec<f64>> {
            let t = full(x);
            if t[np..].iter().any(|s| *s <= 0.0) {
                return None;
            }
            let s = solve(&model, &t[..np]).ok()?;
            let ss = ssm(&model, &s, &t[np..]);
            let o = kfilter(&ss, &d.y)?;
            Some(o.ll)
        };
        // the model must solve at the starting values
        solve(&model, &th[..np])?;
        let solve_only = cmd.has_opt("solve", 5);
        let x0: Vec<f64> = free.iter().map(|&j| th[j]).collect();
        let (xh, ll, log, iters, conv) = if solve_only || free.is_empty() {
            let ll = ts_ml::ll_sum(&llf, &x0);
            (x0.clone(), ll, Vec::new(), 0, true)
        } else {
            let tech = ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs);
            let o = ts_ml::optimize(&llf, x0, util::opt_num(cmd, "iterate", 4)?.map_or(300, |v| v as usize), tech, "log likelihood", true)?;
            (o.th, o.ll, o.log, o.iters, o.converged)
        };
        let tfull = full(&xh);
        let mut v = Matrix::zeros(k, k);
        if !solve_only && !free.is_empty() {
            let (vf, _, _) = ts_ml::vce(&llf, &xh, cmd, "oim")?;
            for (a, &i) in free.iter().enumerate() {
                for (b, &j) in free.iter().enumerate() {
                    v[(i, j)] = vf[(a, b)];
                }
            }
        }
        let nobs = d.y.len();
        let (tvar, t0, t1) = (d.tvar.clone(), d.times[0], *d.times.last().unwrap());
        let sample_line = format!("Sample: {} thru {}", crate::tsops::time_label(ds, &tvar, t0), crate::tsops::time_label(ds, &tvar, t1));
        let title = if self.nonlinear { "First-order DSGE model" } else { "DSGE model" };
        let header = vec![crate::tsreg::hline(&sample_line, "Number of obs", &fmt_g(nobs as f64, 10)), format!("Log likelihood = {}", fmt_g(ll, 9))];
        let mut notes = Vec::new();
        if fixed.iter().any(|f| *f) {
            notes.push("Note: constrained parameters are fixed at their constrained values.".to_string());
        }
        let opts: Vec<String> = cmd
            .opts
            .iter()
            .filter(|(n, _)| ["observed", "unobserved", "exostate", "endostate", "steadyinit"].iter().any(|o| o.starts_with(n.to_lowercase().as_str()) && n.len() >= 3))
            .map(|(n, a)| match a {
                Some(a) => format!("{n}({a})"),
                None => n.clone(),
            })
            .collect();
        let text = ts_ml::post(
            st,
            cmd,
            ts_ml::Post {
                title,
                cmd: verb,
                depvar: &model.observed.join(" "),
                names: names.clone(),
                b: tfull.clone(),
                v: v.clone(),
                ll,
                n: nobs,
                header,
                vcetype: "",
                vce: "oim",
                log,
                notes,
                converged: conv,
                iters,
                df: None,
            },
            Some(&d.mask),
        )?;
        let mark: Vec<bool> = if solve_only { vec![true; k] } else { fixed.clone() };
        let text = crate::sspace::mark_constrained(text, &tfull, &v, &mark);
        let solved = solve(&model, &tfull[..np])?;
        let e = &mut st.e;
        e.set_str("_dsge_main", cmd.main.trim());
        e.set_str("_dsge_opts", opts.join(" "));
        e.set_str("state", model.states.join(" "));
        e.set_str("control", model.controls.join(" "));
        e.set_str("observed", model.observed.join(" "));
        e.set_str("shocks", model.shocked.iter().map(|s| format!("e.{s}")).collect::<Vec<_>>().join(" "));
        e.set_str("_output", text.clone());
        e.set("k_state", model.ns() as f64);
        e.set("k_control", model.controls.len() as f64);
        e.set("k_shock", model.shocked.len() as f64);
        e.set("k_observed", model.observed.len() as f64);
        e.set("k_stable", solved.sol.n_stable as f64);
        e.set("tmin", t0);
        e.set("tmax", t1);
        e.set_str("tvar", tvar);
        e.set_mat("policy", named(&solved.sol.g, &model.controls, &model.states));
        e.set_mat("transition", named(&solved.sol.h, &model.states, &model.states));
        if self.nonlinear {
            e.set_mat("steady", Matrix::row_vector(&solved.steady).with_names(vec!["steady".into()], model.vars()));
        }
        e.set_str("estat_cmd", "dsge_estat");
        Ok(text)
    }
}

fn named(m: &[Vec<f64>], rows: &[String], cols: &[String]) -> Matrix {
    Matrix::from_rows(m).with_names(rows.to_vec(), cols.to_vec())
}

// ---------------------------------------------------------------------------
// postestimation
// ---------------------------------------------------------------------------

/// Table of transformed quantities with delta-method standard errors.
fn delta_table(st: &mut Store, cmd: &Command, title: &str, labels: &[String], f: &dyn Fn(&[f64]) -> Vec<f64>, fit: &Fitted) -> Result<String, String> {
    let level = util::level(cmd, st)?;
    let val = f(&fit.theta);
    let j = crate::tsla::jacobian(f, &fit.theta);
    let var = crate::tsla::delta_var(&j, &fit.v);
    let crit = crate::dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut out = vec![String::new(), title.to_string(), String::new()];
    let lw = labels.iter().map(|l| l.len()).max().unwrap_or(8).clamp(12, 24);
    let bar = format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64));
    out.push("-".repeat(lw + 66));
    out.push(format!("{:>w$} | {:>10}  {:>9}  {:>7}  {:>5}     [{}% conf. interval]", "", "Coefficient", "Std. err.", "z", "P>|z|", fmt_g(level, 4), w = lw));
    out.push(bar);
    let mut tab = Matrix::zeros(6, val.len());
    for (i, l) in labels.iter().enumerate() {
        let se = var[i].max(0.0).sqrt();
        let b = val[i];
        if se > 1e-12 * b.abs().max(1.0) {
            let z = b / se;
            let p = 2.0 * crate::dist::normal(-z.abs());
            out.push(format!("{:>w$} | {:>10}  {:>9}  {:>7.2}  {:>5.3}    {:>10}  {:>10}", l, fmt_g(b, 9), fmt_g(se, 9), z, p, fmt_g(b - crit * se, 9), fmt_g(b + crit * se, 9), w = lw));
            tab[(0, i)] = b;
            tab[(1, i)] = se;
            tab[(2, i)] = z;
            tab[(3, i)] = p;
            tab[(4, i)] = b - crit * se;
            tab[(5, i)] = b + crit * se;
        } else {
            out.push(format!("{:>w$} | {:>10}  (constrained)", l, fmt_g(b, 9), w = lw));
            tab[(0, i)] = b;
            for r in 1..6 {
                tab[(r, i)] = SYSMISS;
            }
        }
    }
    out.push("-".repeat(lw + 66));
    let res = st.begin_r();
    res.set_mat("b", Matrix::row_vector(&val).with_names(vec!["y1".into()], labels.to_vec()));
    let mut vm = Matrix::zeros(val.len(), val.len());
    let full = crate::tsla::delta_var(&j, &fit.v);
    let _ = full;
    let jv = j.mul(&fit.v)?.mul(&j.transpose())?;
    for a in 0..val.len() {
        for b in 0..val.len() {
            vm[(a, b)] = jv[(a, b)];
        }
    }
    res.set_mat("V", vm.with_names(labels.to_vec(), labels.to_vec()));
    res.set_mat("table", tab.with_names(["b", "se", "z", "pvalue", "ll", "ul"].iter().map(|s| s.to_string()).collect(), labels.to_vec()));
    Ok(out.join("\n"))
}

pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    // other estimators (mswitch, ...) have their own estat transition etc.
    if ecmd != "dsge" && ecmd != "dsgenl" {
        return None;
    }
    Some(estat_dsge(st, sub, cmd))
}

fn estat_dsge(st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let fit = fitted(st)?;
    let m = fit.model.clone();
    let np = m.params.len();
    match sub {
        "policy" | "pol" | "poli" | "polic" => {
            let labels: Vec<String> = m.controls.iter().flat_map(|c| m.states.iter().map(move |s| format!("{c}:{s}"))).collect();
            let f = |t: &[f64]| -> Vec<f64> { solve(&m, &t[..np]).map(|s| s.sol.g.concat()).unwrap_or_else(|_| vec![f64::NAN; m.controls.len() * m.ns()]) };
            let text = delta_table(st, cmd, "Policy matrix", &labels, &f, &fit)?;
            let s = solve(&m, fit.par())?;
            st.r.set_mat("policy", named(&s.sol.g, &m.controls, &m.states));
            Ok(text)
        }
        "transition" | "tr" | "tra" | "tran" | "trans" | "transi" | "transit" | "transiti" | "transitio" => {
            let labels: Vec<String> = m.states.iter().flat_map(|a| m.states.iter().map(move |s| format!("F.{a}:{s}"))).collect();
            let f = |t: &[f64]| -> Vec<f64> { solve(&m, &t[..np]).map(|s| s.sol.h.concat()).unwrap_or_else(|_| vec![f64::NAN; m.ns() * m.ns()]) };
            let text = delta_table(st, cmd, "Transition matrix of state variables", &labels, &f, &fit)?;
            let s = solve(&m, fit.par())?;
            st.r.set_mat("transition", named(&s.sol.h, &m.states, &m.states));
            Ok(text)
        }
        "steady" | "ste" | "stea" => {
            if !m.nonlinear {
                return Err(rc_err(321, "estat steady is available only after dsgenl"));
            }
            let labels = m.vars();
            let f = |t: &[f64]| -> Vec<f64> { solve(&m, &t[..np]).map(|s| s.steady).unwrap_or_else(|_| vec![f64::NAN; m.n()]) };
            let text = delta_table(st, cmd, "Location of model steady state", &labels, &f, &fit)?;
            let s = solve(&m, fit.par())?;
            st.r.set_mat("steady", Matrix::row_vector(&s.steady).with_names(vec!["steady".into()], labels));
            Ok(text)
        }
        "stable" | "stab" | "stabl" => {
            let s = solve(&m, fit.par())?;
            let mut out = vec![String::new(), "Stability results".to_string(), String::new(), "-".repeat(37), format!("{:>24} |  Modulus", "Eigenvalues"), format!("{}+{}", "-".repeat(25), "-".repeat(11))];
            let mut moduli = Vec::new();
            for z in &s.sol.eig {
                let md = z.abs();
                if md.is_infinite() {
                    out.push(format!("{:>24} |  {:>7}", "(infinite)", "."));
                    moduli.push(SYSMISS);
                    continue;
                }
                let txt = if z.im.abs() < 1e-10 { format!("{:.6}", z.re) } else { format!("{:.6} {} {:.6}i", z.re, if z.im < 0.0 { "-" } else { "+" }, z.im.abs()) };
                out.push(format!("{:>24} |  {:>7.4}", txt, md));
                moduli.push(md);
            }
            out.push("-".repeat(37));
            out.push(format!("Stable eigenvalues: {} (state variables: {})", s.sol.n_stable, m.ns()));
            out.push("The model is saddle-path stable.".into());
            let r = st.begin_r();
            r.set("stable", 1.0);
            r.set("n_stable", s.sol.n_stable as f64);
            r.set_mat("eigenvalues", Matrix::row_vector(&moduli));
            Ok(out.join("\n"))
        }
        "covariance" | "cov" | "cova" | "covar" | "covari" | "covaria" | "covarian" | "covarianc" => {
            // model-implied covariance of the observed (or all) variables
            let names: Vec<String> = if cmd.has_opt("addcovariance", 3) || cmd.main.split_whitespace().nth(1).is_some_and(|w| w == "_all") {
                m.vars()
            } else {
                let rest: Vec<String> = cmd.main.split_whitespace().skip(1).map(String::from).collect();
                if rest.is_empty() { m.observed.clone() } else { rest }
            };
            let vars = m.vars();
            for n in &names {
                if !vars.contains(n) {
                    return Err(rc_err(198, format!("{n} is not a model variable")));
                }
            }
            let nn = names.len();
            let f = |t: &[f64]| -> Vec<f64> {
                match implied_cov(&m, &t[..np], &t[np..], &names) {
                    Some(c) => {
                        let mut v = Vec::new();
                        for i in 0..nn {
                            for j in 0..=i {
                                v.push(c[(i, j)]);
                            }
                        }
                        v
                    }
                    None => vec![f64::NAN; nn * (nn + 1) / 2],
                }
            };
            let mut labels = Vec::new();
            for i in 0..nn {
                for j in 0..=i {
                    labels.push(format!("{}:{}", names[i], names[j]));
                }
            }
            let text = delta_table(st, cmd, "Estimated covariances of model variables", &labels, &f, &fit)?;
            let c = implied_cov(&m, fit.par(), fit.sds(), &names).ok_or_else(|| rc_err(498, "covariance could not be computed"))?;
            st.r.set_mat("covariance", c.with_names(names.clone(), names));
            Ok(text)
        }
        _ => Err(rc_err(321, format!("estat {sub} not valid after {}", if m.nonlinear { "dsgenl" } else { "dsge" }))),
    }
}

/// Unconditional covariance of selected model variables.
pub fn implied_cov(m: &Model, par: &[f64], sds: &[f64], names: &[String]) -> Option<Matrix> {
    let s = solve(m, par).ok()?;
    let ss = ssm(m, &s, sds);
    let p = stationary_cov(&ss.t, &ss.rqr)?;
    let ns = m.ns();
    // loadings of each variable on the states
    let rows: Vec<Vec<f64>> = names
        .iter()
        .map(|n| {
            if let Some(i) = m.states.iter().position(|x| x == n) {
                (0..ns).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
            } else {
                let c = m.controls.iter().position(|x| x == n).unwrap();
                s.sol.g[c].clone()
            }
        })
        .collect();
    let l = Matrix::from_rows(&rows);
    l.mul(&p).ok()?.mul(&l.transpose()).ok()
}

/// Impulse responses after dsge for `irf create`: variables respond to a
/// one-standard-deviation shock to each shocked state; impulses that are
/// not shocks are missing.
pub fn irf_result(st: &Store, name: &str, step: usize) -> Result<crate::tsirf::IrfResult, String> {
    let fit = fitted(st)?;
    let m = &fit.model;
    let s = solve(m, fit.par())?;
    let vars = m.vars();
    let k = vars.len();
    let ns = m.ns();
    let compute = |t: &[f64]| -> Vec<f64> {
        let np = m.params.len();
        let Ok(s) = solve(m, &t[..np]) else { return vec![f64::NAN; k * k * (step + 1)] };
        let mut out = vec![SYSMISS; k * k * (step + 1)];
        for (si, sh) in m.shocked.iter().enumerate() {
            let i = vars.iter().position(|v| v == sh).unwrap();
            let mut x = vec![0.0; ns];
            x[m.states.iter().position(|v| v == sh).unwrap()] = t[np + si];
            for h in 0..=step {
                for (j, _) in vars.iter().enumerate() {
                    let val = if j < ns { x[j] } else { (0..ns).map(|a| s.sol.g[j - ns][a] * x[a]).sum() };
                    out[(i * k + j) * (step + 1) + h] = val;
                }
                x = (0..ns).map(|a| (0..ns).map(|b| s.sol.h[a][b] * x[b]).sum()).collect();
            }
        }
        out
    };
    let _ = s;
    let point = compute(&fit.theta);
    let j = crate::tsla::jacobian(&|t: &[f64]| compute(t).iter().map(|v| if is_missing(*v) { 0.0 } else { *v }).collect(), &fit.theta);
    let se: Vec<f64> = crate::tsla::delta_var(&j, &fit.v).iter().zip(&point).map(|(v, p)| if is_missing(*p) { SYSMISS } else { v.max(0.0).sqrt() }).collect();
    let mut r = crate::tsirf::IrfResult { name: name.to_string(), endog: vars, exog: Vec::new(), step, ..Default::default() };
    r.vals.insert("irf".into(), point);
    r.vals.insert("stdirf".into(), se);
    r.meta = vec![("model".into(), st.e.macros.get("cmd").cloned().unwrap_or_default()), ("stderror".into(), "asymptotic".into()), ("step".into(), step.to_string()), ("order".into(), String::new()), ("exog".into(), String::new()), ("endog".into(), r.endog.join(" ")), ("version".into(), "1.1".into())];
    Ok(r)
}

/// Write the shocks' responses into an irf file (hook of `irf create`).
pub fn irf_create(st: &mut Store, cmd: &Command, name: &str, path: &str) -> Result<String, String> {
    let step = util::opt_num(cmd, "step", 3)?.unwrap_or(8.0).max(1.0) as usize;
    let r = irf_result(st, name, step)?;
    let mut file = crate::tsirf::IrfFile::load(path)?;
    if file.get(name).is_some() && !cmd.has_opt("replace", 3) {
        return Err(rc_err(110, format!("irfname {name} already exists in {path}")));
    }
    file.results.retain(|q| q.name != name);
    file.results.push(r);
    file.save(path)?;
    Ok(format!("(file {path} updated)"))
}
