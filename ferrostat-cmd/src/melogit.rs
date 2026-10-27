//! `melogit` / `meqrlogit`: multilevel mixed-effects logistic regression
//! with random intercepts and (independent) random slopes, fitted by
//! mean-variance adaptive Gauss-Hermite quadrature.

use std::cell::RefCell;
use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec, Likelihood};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

/// Gauss-Hermite nodes and weights for the weight function exp(-x^2)
/// (Golub-Welsch).
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = Matrix::zeros(n, n);
    for i in 1..n {
        let b = (i as f64 / 2.0).sqrt();
        j[(i, i - 1)] = b;
        j[(i - 1, i)] = b;
    }
    let (vals, vecs) = j.sym_eigen();
    let mut pairs: Vec<(f64, f64)> = (0..n).map(|k| (vals[k], std::f64::consts::PI.sqrt() * vecs[(0, k)] * vecs[(0, k)])).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1).collect())
}

/// A random-effects level: design columns of the random effects (over the
/// estimation sample) and the position of their log standard deviations.
struct Level {
    var: String,
    z: Vec<Vec<f64>>,
    znames: Vec<String>,
    psi: usize,
}

/// A group at some level: its observations and, above the lowest level,
/// the nested groups.
struct Unit {
    obs: Vec<usize>,
    children: Vec<Unit>,
}

struct MeModel<'a> {
    x: Vec<&'a [f64]>,
    y: &'a [f64],
    levels: Vec<Level>,
    units: Vec<Unit>,
    a: Vec<f64>,
    w: Vec<f64>,
    /// adaptive parameters per group (and higher-level node path)
    cache: RefCell<HashMap<Vec<usize>, (Vec<f64>, Matrix)>>,
}

struct NodeEval {
    ll: f64,
    g: Vec<f64>,
    h: Option<Matrix>,
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl<'a> MeModel<'a> {
    fn nparm(&self) -> usize {
        self.x.len() + self.levels.iter().map(|l| l.z.len()).sum::<usize>()
    }

    /// All multi-dimensional quadrature points: (index per dimension).
    fn grid(&self, q: usize) -> Vec<Vec<usize>> {
        let m = self.a.len();
        let mut out = vec![vec![]];
        for _ in 0..q {
            let mut next = Vec::new();
            for p in &out {
                for i in 0..m {
                    let mut v = p.clone();
                    v.push(i);
                    next.push(v);
                }
            }
            out = next;
        }
        out
    }

    fn sds(&self, l: usize, th: &[f64]) -> Vec<f64> {
        let lv = &self.levels[l];
        (0..lv.z.len()).map(|r| th[lv.psi + r].exp()).collect()
    }

    /// Log of the integrand at random-effect value `u` for `unit` at level `l`
    /// (with higher-level contributions `off` to the linear predictor).
    fn node(&self, l: usize, unit: &Unit, path: &[usize], u: &[f64], off: &HashMap<usize, f64>, xb: &[f64], th: &[f64], derivs: bool) -> NodeEval {
        let lv = &self.levels[l];
        let q = lv.z.len();
        let p = self.nparm();
        let k = self.x.len();
        let sd = self.sds(l, th);
        let mut ll = 0.0;
        let mut g = vec![0.0; if derivs { p } else { 0 }];
        let mut h = if derivs { Some(Matrix::zeros(p, p)) } else { None };
        // prior density of u
        for r in 0..q {
            let z = u[r] / sd[r];
            ll += -0.5 * (2.0 * std::f64::consts::PI).ln() - th[lv.psi + r] - 0.5 * z * z;
            if derivs {
                g[lv.psi + r] += -1.0 + z * z;
                h.as_mut().unwrap()[(lv.psi + r, lv.psi + r)] += -2.0 * z * z;
            }
        }
        if unit.children.is_empty() {
            for &i in &unit.obs {
                let mut eta = xb[i] + off.get(&i).copied().unwrap_or(0.0);
                for r in 0..q {
                    eta += lv.z[r][i] * u[r];
                }
                let yi = self.y[i];
                let lp = -(1.0 + (-eta).exp()).ln();
                let lq = -(1.0 + eta.exp()).ln();
                let lp = if lp.is_finite() { lp } else { eta };
                let lq = if lq.is_finite() { lq } else { -eta };
                ll += if yi != 0.0 { lp } else { lq };
                if derivs {
                    let pr = dist::invlogit(eta);
                    let r = (if yi != 0.0 { 1.0 } else { 0.0 }) - pr;
                    let wv = pr * (1.0 - pr);
                    let hm = h.as_mut().unwrap();
                    for a in 0..k {
                        let xa = self.x[a][i];
                        g[a] += r * xa;
                        if wv != 0.0 && xa != 0.0 {
                            for c in 0..=a {
                                hm[(a, c)] -= wv * xa * self.x[c][i];
                            }
                        }
                    }
                }
            }
            if let Some(hm) = h.as_mut() {
                for a in 0..k {
                    for c in 0..a {
                        hm[(c, a)] = hm[(a, c)];
                    }
                }
            }
        } else {
            let mut off2 = off.clone();
            for &i in &unit.obs {
                let mut e = 0.0;
                for r in 0..q {
                    e += lv.z[r][i] * u[r];
                }
                *off2.entry(i).or_insert(0.0) += e;
            }
            for (ci, child) in unit.children.iter().enumerate() {
                let mut cp = path.to_vec();
                cp.push(ci);
                let r = self.integrate(l + 1, child, &cp, &off2, xb, th, derivs);
                ll += r.ll;
                if derivs {
                    for a in 0..p {
                        g[a] += r.g[a];
                    }
                    let hm = h.as_mut().unwrap();
                    let rh = r.h.unwrap();
                    for a in 0..p {
                        for c in 0..p {
                            hm[(a, c)] += rh[(a, c)];
                        }
                    }
                }
            }
        }
        NodeEval { ll, g, h }
    }

    /// Adaptive quadrature over the random effects of `unit` at level `l`.
    fn integrate(&self, l: usize, unit: &Unit, path: &[usize], off: &HashMap<usize, f64>, xb: &[f64], th: &[f64], derivs: bool) -> NodeEval {
        let q = self.levels[l].z.len();
        let p = self.nparm();
        let grid = self.grid(q);
        let sqrt2 = std::f64::consts::SQRT_2;
        let sd = self.sds(l, th);
        let cached = self.cache.borrow().get(path).cloned();
        let (mut mu, mut c) = match cached {
            Some(v) => v,
            None => {
                let mut c = Matrix::zeros(q, q);
                for r in 0..q {
                    c[(r, r)] = sd[r];
                }
                let mut mu = vec![0.0; q];
                // lowest level: start at the posterior mode and curvature
                if self.levels.len() == l + 1 {
                    if let Some((m, cc)) = self.mode(l, unit, off, xb, &sd) {
                        mu = m;
                        c = cc;
                    }
                }
                (mu, c)
            }
        };
        let points = |mu: &[f64], c: &Matrix| -> Vec<(Vec<f64>, f64)> {
            let logdet: f64 = (0..q).map(|r| c[(r, r)].abs().ln()).sum();
            grid.iter()
                .map(|ix| {
                    let a: Vec<f64> = ix.iter().map(|&i| self.a[i]).collect();
                    let u: Vec<f64> = (0..q).map(|r| mu[r] + sqrt2 * (0..=r).map(|s| c[(r, s)] * a[s]).sum::<f64>()).collect();
                    let lw: f64 = ix.iter().map(|&i| self.w[i].ln()).sum::<f64>() + a.iter().map(|x| x * x).sum::<f64>() + 0.5 * q as f64 * 2f64.ln() + logdet;
                    (u, lw)
                })
                .collect()
        };
        // mean-variance adaptation
        for _ in 0..60 {
            let pts = points(&mu, &c);
            let lv: Vec<f64> = pts.iter().enumerate().map(|(k, (u, lw))| lw + self.node(l, unit, &child_path(path, k), u, off, xb, th, false).ll).collect();
            let tot = lse(&lv);
            let pi: Vec<f64> = lv.iter().map(|v| (v - tot).exp()).collect();
            let nmu: Vec<f64> = (0..q).map(|r| pts.iter().zip(&pi).map(|((u, _), w)| w * u[r]).sum()).collect();
            let mut cov = Matrix::zeros(q, q);
            for ((u, _), w) in pts.iter().zip(&pi) {
                for r in 0..q {
                    for s in 0..q {
                        cov[(r, s)] += w * (u[r] - nmu[r]) * (u[s] - nmu[s]);
                    }
                }
            }
            let nc = match cov.cholesky() {
                Ok(m) => m,
                Err(_) => break,
            };
            let delta = nmu.iter().zip(&mu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max).max((0..q).map(|r| (0..q).map(|s| (nc[(r, s)] - c[(r, s)]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max));
            mu = nmu;
            c = nc;
            if delta < 1e-9 {
                break;
            }
        }
        self.cache.borrow_mut().insert(path.to_vec(), (mu.clone(), c.clone()));
        let pts = points(&mu, &c);
        let evals: Vec<NodeEval> = pts.iter().enumerate().map(|(k, (u, _))| self.node(l, unit, &child_path(path, k), u, off, xb, th, derivs)).collect();
        let lv: Vec<f64> = pts.iter().zip(&evals).map(|((_, lw), e)| lw + e.ll).collect();
        let tot = lse(&lv);
        if !derivs {
            return NodeEval { ll: tot, g: vec![], h: None };
        }
        let pi: Vec<f64> = lv.iter().map(|v| (v - tot).exp()).collect();
        let mut g = vec![0.0; p];
        let mut h = Matrix::zeros(p, p);
        for (e, w) in evals.iter().zip(&pi) {
            let eh = e.h.as_ref().unwrap();
            for a in 0..p {
                g[a] += w * e.g[a];
                for b in 0..p {
                    h[(a, b)] += w * (eh[(a, b)] + e.g[a] * e.g[b]);
                }
            }
        }
        for a in 0..p {
            for b in 0..p {
                h[(a, b)] -= g[a] * g[b];
            }
        }
        NodeEval { ll: tot, g, h: Some(h) }
    }

    /// Posterior mode and Cholesky factor of the inverse curvature for a
    /// lowest-level group.
    fn mode(&self, l: usize, unit: &Unit, off: &HashMap<usize, f64>, xb: &[f64], sd: &[f64]) -> Option<(Vec<f64>, Matrix)> {
        let lv = &self.levels[l];
        let q = lv.z.len();
        let mut u = vec![0.0; q];
        let mut hinv = Matrix::zeros(q, q);
        for _ in 0..50 {
            let mut g: Vec<f64> = (0..q).map(|r| -u[r] / (sd[r] * sd[r])).collect();
            let mut h = Matrix::zeros(q, q);
            for r in 0..q {
                h[(r, r)] = 1.0 / (sd[r] * sd[r]);
            }
            for &i in &unit.obs {
                let mut eta = xb[i] + off.get(&i).copied().unwrap_or(0.0);
                for r in 0..q {
                    eta += lv.z[r][i] * u[r];
                }
                let pr = dist::invlogit(eta);
                let res = (if self.y[i] != 0.0 { 1.0 } else { 0.0 }) - pr;
                for r in 0..q {
                    g[r] += res * lv.z[r][i];
                    for s in 0..q {
                        h[(r, s)] += pr * (1.0 - pr) * lv.z[r][i] * lv.z[s][i];
                    }
                }
            }
            hinv = h.inverse().ok()?;
            let step: Vec<f64> = (0..q).map(|r| (0..q).map(|s| hinv[(r, s)] * g[s]).sum()).collect();
            for r in 0..q {
                u[r] += step[r].clamp(-5.0, 5.0);
            }
            if step.iter().all(|s| s.abs() < 1e-10) {
                break;
            }
        }
        let c = hinv.cholesky().ok()?;
        Some((u, c))
    }
}

fn child_path(path: &[usize], k: usize) -> Vec<usize> {
    // node index marks the conditioning point of nested groups
    let mut v = path.to_vec();
    v.push(usize::MAX - k);
    v
}

impl<'a> Likelihood for MeModel<'a> {
    fn k(&self) -> usize {
        self.nparm()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let p = self.nparm();
        let n = self.y.len();
        let xb: Vec<f64> = (0..n).map(|i| self.x.iter().zip(b).map(|(c, bb)| c[i] * bb).sum()).collect();
        let mut ll = 0.0;
        let mut g = vec![0.0; p];
        let mut h = Matrix::zeros(p, p);
        let mut scores = Vec::new();
        let off = HashMap::new();
        for (ui, unit) in self.units.iter().enumerate() {
            let r = self.integrate(0, unit, &[ui], &off, &xb, b, true);
            if !r.ll.is_finite() {
                return (f64::NEG_INFINITY, g, h, scores);
            }
            ll += r.ll;
            let rh = r.h.unwrap();
            for a in 0..p {
                g[a] += r.g[a];
                for c in 0..p {
                    h[(a, c)] -= rh[(a, c)];
                }
            }
            if need_scores {
                scores.push(r.g.clone());
            }
        }
        (ll, g, h, scores)
    }
}

// ---------------------------------------------------------------------------
// command
// ---------------------------------------------------------------------------

pub struct Melogit {
    pub verb: &'static str,
}

struct LevelSpec {
    var: String,
    revars: Vec<String>,
    cons: bool,
}

fn group_codes(ds: &DataSet, var: &str, rows: &[usize]) -> Result<Vec<String>, String> {
    let v = ds.get_var(var).ok_or_else(|| rc_err(111, format!("variable {var} not found")))?;
    Ok(rows
        .iter()
        .map(|&i| match &v.data[i] {
            Value::Str(s) => s.clone(),
            x => fmt_g(x.num(), 17),
        })
        .collect())
}

/// Build nested units: level `l` groups within the observations `obs`.
fn build_units(codes: &[Vec<String>], l: usize, obs: &[usize]) -> Vec<Unit> {
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for &i in obs {
        let key = codes[l][i].clone();
        if !map.contains_key(&key) {
            order.push(key.clone());
        }
        map.entry(key).or_default().push(i);
    }
    order
        .into_iter()
        .map(|key| {
            let o = map.remove(&key).unwrap();
            let children = if l + 1 < codes.len() { build_units(codes, l + 1, &o) } else { vec![] };
            Unit { obs: o, children }
        })
        .collect()
}

fn count_units(units: &[Unit], depth: usize, out: &mut Vec<Vec<usize>>) {
    for u in units {
        out[depth].push(u.obs.len());
        count_units(&u.children, depth + 1, out);
    }
}

impl CommandHandler for Melogit {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let after_verb = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        if !raw.contains("||") {
            // replay
            let e = &st.e;
            if !matches!(e.macros.get("cmd").map(String::as_str), Some("melogit" | "meqrlogit")) || !cmd.main.trim().is_empty() {
                return Err(if cmd.main.trim().is_empty() { rc_err(301, "last estimates not found") } else { rc_err(198, "random-effects equation required (||)") });
            }
            let eform = cmd.has_opt("or", 2);
            let text = render(&st.e, eform, util::level(cmd, st)?);
            return Ok(text);
        }
        let mut parts = after_verb.split("||");
        let fixed = parts.next().unwrap_or("").trim().to_string();
        let mut global_opts: Vec<String> = Vec::new();
        let mut specs = Vec::new();
        for part in parts {
            let (body, opts) = match part.split_once(',') {
                Some((a, b)) => (a.trim(), b.trim()),
                None => (part.trim(), ""),
            };
            let (gv, rest) = body.split_once(':').ok_or_else(|| rc_err(198, "invalid random-effects specification"))?;
            let gv = gv.trim();
            if gv == "_all" {
                return Err(rc_err(198, "_all: random effects not supported"));
            }
            let var = crate::varlist::resolve_name(ds, gv)?;
            let revars = if rest.trim().is_empty() { vec![] } else { crate::varlist::expand(ds, rest.trim())? };
            let lopts = ferrostat_parser::parser::parse_line(&format!("x , {opts}")).map_err(|e| rc_err(198, e.message))?;
            let mut cons = true;
            for (n, a) in &lopts.opts {
                let nl = n.to_lowercase();
                if "noconstant".starts_with(nl.as_str()) && nl.len() >= 5 {
                    cons = false;
                } else if "covariance".starts_with(nl.as_str()) && nl.len() >= 3 {
                    let c = a.clone().unwrap_or_default().to_lowercase();
                    if !c.is_empty() && !"independent".starts_with(c.trim()) {
                        return Err(rc_err(198, format!("covariance({}) not supported", c.trim())));
                    }
                } else {
                    global_opts.push(match a {
                        Some(a) => format!("{n}({a})"),
                        None => n.clone(),
                    });
                }
            }
            if revars.is_empty() && !cons {
                return Err(rc_err(198, "no random effects specified"));
            }
            specs.push(LevelSpec { var, revars, cons });
        }
        let fixed_line = if global_opts.is_empty() {
            format!("{} {fixed}", self.verb)
        } else if fixed.contains(',') {
            format!("{} {fixed} {}", self.verb, global_opts.join(" "))
        } else {
            format!("{} {fixed}, {}", self.verb, global_opts.join(" "))
        };
        let fcmd = ferrostat_parser::parser::parse_line(&fixed_line).map_err(|e| rc_err(198, e.message))?;
        let main = fcmd.main.trim().to_string();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main.as_str(), ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        if fcmd.weight.is_some() {
            return Err(rc_err(101, "weights not allowed"));
        }
        let mut extra: Vec<String> = specs.iter().map(|s| s.var.clone()).collect();
        for s in &specs {
            extra.extend(s.revars.iter().cloned());
        }
        let mut d = est::build(ds, st, &fcmd, DesignSpec { depvar: Some(&dep), indep, noconstant: fcmd.has_opt("noconstant", 3), weights: "", extra })?;
        for y in d.y.iter_mut() {
            *y = if *y != 0.0 { 1.0 } else { 0.0 };
        }
        let nq = util::opt_num(&fcmd, "intpoints", 3)?.unwrap_or(7.0) as usize;
        if nq < 1 || nq > 128 {
            return Err(rc_err(198, "intpoints() must be between 1 and 128"));
        }
        let n = d.n();
        let act = d.active();
        let k = act.len();
        // random-effects designs
        let mut levels = Vec::new();
        let mut psi = k;
        for s in &specs {
            let mut z = Vec::new();
            let mut znames = Vec::new();
            for v in &s.revars {
                let c = util::col(ds, v)?;
                z.push(d.rows.iter().map(|&i| c[i]).collect::<Vec<f64>>());
                znames.push(v.clone());
            }
            if s.cons {
                z.push(vec![1.0; n]);
                znames.push("_cons".into());
            }
            let q = z.len();
            levels.push(Level { var: s.var.clone(), z, znames, psi });
            psi += q;
        }
        let codes: Vec<Vec<String>> = specs.iter().map(|s| group_codes(ds, &s.var, &d.rows)).collect::<Result<_, _>>()?;
        // nested groups: identify lower-level groups within higher ones
        let mut nested = codes.clone();
        for l in 1..nested.len() {
            for i in 0..n {
                nested[l][i] = format!("{}\u{1}{}", nested[l - 1][i], codes[l][i]);
            }
        }
        let all: Vec<usize> = (0..n).collect();
        let units = build_units(&nested, 0, &all);
        let mut counts = vec![Vec::new(); specs.len()];
        count_units(&units, 0, &mut counts);
        let (a, w) = gauss_hermite(nq);
        let ll_label = "Log likelihood";
        let mut out = Vec::new();
        // fixed-effects logit for starting values and the LR test
        let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let fe = crate::zip::MultiIndex {
            x: vec![cols.clone()],
            naux: 0,
            w: &d.w,
            fweight: false,
            f: &|t: usize, idx: &[f64]| {
                let p = dist::invlogit(idx[0]);
                let eta = idx[0];
                let lp = -(1.0 + (-eta).exp()).ln();
                let lq = -(1.0 + eta.exp()).ln();
                let lp = if lp.is_finite() { lp } else { eta };
                let lq = if lq.is_finite() { lq } else { -eta };
                (if d.y[t] != 0.0 { lp } else { lq }, vec![d.y[t] - p], vec![vec![-p * (1.0 - p)]])
            },
        };
        let mut b0 = vec![0.0; k];
        if let Some(cp) = act.iter().position(|&j| d.names[j] == "_cons") {
            let my = d.y.iter().sum::<f64>() / n as f64;
            b0[cp] = dist::logit(my.clamp(1e-6, 1.0 - 1e-6));
        }
        let fres = est::maximize(&fe, b0, 300)?;
        out.push(String::new());
        out.push("Fitting fixed-effects model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&fres.iterations, ll_label));
        let model = MeModel { x: cols.clone(), y: &d.y, levels, units, a, w, cache: RefCell::new(HashMap::new()) };
        let p = model.nparm();
        let mut th0 = fres.b.clone();
        th0.resize(p, 0.0);
        // refine the starting variance on a small grid
        out.push(String::new());
        out.push("Refining starting values:".into());
        out.push(String::new());
        let mut best = (f64::NEG_INFINITY, th0.clone());
        for (gi, lsd) in [0.0f64, -0.5, 0.5].iter().enumerate() {
            let mut th = th0.clone();
            for v in th.iter_mut().skip(k) {
                *v = *lsd;
            }
            model.cache.borrow_mut().clear();
            let (ll, ..) = model.eval(&th, false);
            if gi == 0 {
                out.push(format!("Grid node {gi}:  {ll_label} = {}", fmt_g(ll, 9)));
            }
            if ll > best.0 {
                best = (ll, th);
            }
        }
        model.cache.borrow_mut().clear();
        let res = est::maximize(&model, best.1, 200)?;
        out.push(String::new());
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, ll_label));
        // parameters in display metric: variances
        let (vth, _) = res.info.sym_inverse(1e-14);
        let kt = d.k();
        let nre = p - k;
        let total = kt + nre;
        let mut names: Vec<String> = d.names.iter().map(|nm| format!("{dep}:{nm}")).collect();
        for lv in &model.levels {
            for zn in &lv.znames {
                names.push(format!("/:var({zn}[{}])", lv.var));
            }
        }
        let mut b = vec![0.0; total];
        let mut idx: Vec<usize> = act.clone();
        for r in 0..nre {
            idx.push(kt + r);
        }
        // Jacobian from (beta, ln sd) to (beta, var)
        let mut jac = vec![1.0; p];
        for (a, &j) in idx.iter().enumerate() {
            if a < k {
                b[j] = res.b[a];
            } else {
                let var = (2.0 * res.b[a]).exp();
                b[j] = var;
                jac[a] = 2.0 * var;
            }
        }
        let mut vt = Matrix::zeros(p, p);
        for a in 0..p {
            for c in 0..p {
                vt[(a, c)] = jac[a] * vth[(a, c)] * jac[c];
            }
        }
        let v = est::expand_v(&vt, &idx, total);
        let mut omitted = d.omitted.clone();
        omitted.extend(vec![false; nre]);
        let mut is_base = d.is_base.clone();
        is_base.extend(vec![false; nre]);
        // Wald test of the fixed effects
        let (chi2, df_m) = est::wald_all(&d, &b[..kt], &est::expand_v(&vth, &act, kt).clone());
        let pval = dist::chi2tail(df_m, chi2);
        let lr = (2.0 * (res.ll - fres.ll)).max(0.0);
        let text_head = out.join("\n");
        let mut dpost = d.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        let level = util::level(&fcmd, st)?;
        let ivars: Vec<String> = specs.iter().map(|s| s.var.clone()).collect();
        let verb = self.verb;
        est::post(st, verb, &dpost, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_c", fres.ll);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", pval);
            e.set("chi2_c", lr);
            e.set("df_c", nre as f64);
            e.set("p_c", if nre == 1 { 0.5 * dist::chi2tail(1.0, lr) } else { dist::chi2tail(nre as f64, lr) });
            e.set("n_quad", nq as f64);
            e.set("k_r", nre as f64);
            e.set("k_f", kt as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            e.set("ic", (res.iterations.len() - 1) as f64);
            let mut ng = Matrix::zeros(1, counts.len());
            let mut mn = Matrix::zeros(1, counts.len());
            let mut av = Matrix::zeros(1, counts.len());
            let mut mx = Matrix::zeros(1, counts.len());
            for (l, c) in counts.iter().enumerate() {
                ng[(0, l)] = c.len() as f64;
                mn[(0, l)] = *c.iter().min().unwrap_or(&0) as f64;
                av[(0, l)] = c.iter().sum::<usize>() as f64 / c.len().max(1) as f64;
                mx[(0, l)] = *c.iter().max().unwrap_or(&0) as f64;
            }
            e.set_mat("N_g", ng);
            e.set_mat("g_min", mn);
            e.set_mat("g_avg", av);
            e.set_mat("g_max", mx);
            e.set_str("ivars", ivars.join(" "));
            e.set_str("title", "Mixed-effects logistic regression");
            e.set_str("model", "logistic");
            e.set_str("intmethod", if nq == 1 { "laplace" } else { "mvaghermite" });
            e.set_str("chi2type", "Wald");
            e.set_str("vce", "oim");
            e.set_str("predict", "melogit_p");
            e.set_str("_head", text_head.clone());
        });
        let text = format!("{}\n{}", text_head, render(&st.e, fcmd.has_opt("or", 2), level));
        if !res.converged {
            return Ok(format!("{text}\nconvergence not achieved"));
        }
        Ok(text)
    }
}

/// Header, coefficient table and random-effects table from e().
fn render(e: &Results, eform: bool, level: f64) -> String {
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let mat = |s: &str| e.matrices.get(s).map(|m| m.data.clone()).unwrap_or_default();
    let (ng, mn, av, mx) = (mat("N_g"), mat("g_min"), mat("g_avg"), mat("g_max"));
    let row = |l: &str, k: &str, v: String| format!("{:<48}{:<17} = {:>10}", l, k, v);
    let mut out = vec![String::new()];
    out.push(row("Mixed-effects logistic regression", "Number of obs", util::commas(get("N") as usize)));
    if ivars.len() == 1 {
        out.push(row(&format!("Group variable: {}", ivars[0]), "Number of groups", util::commas(ng.first().copied().unwrap_or(0.0) as usize)));
        out.push(String::new());
        out.push(format!("{:<48}Obs per group:", ""));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "min", util::commas(mn[0] as usize)));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "avg", format!("{:.1}", av[0])));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "max", util::commas(mx[0] as usize)));
    } else {
        out.push(String::new());
        out.push("-".repeat(61));
        out.push("                |     No. of       Observations per group".into());
        out.push(" Group variable |     groups    Minimum    Average    Maximum".into());
        out.push(format!("{}+{}", "-".repeat(16), "-".repeat(44)));
        for (l, iv) in ivars.iter().enumerate() {
            out.push(format!("{:>15} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(iv, 15), util::commas(ng[l] as usize), util::commas(mn[l] as usize), format!("{:.1}", av[l]), util::commas(mx[l] as usize)));
        }
        out.push("-".repeat(61));
    }
    out.push(String::new());
    let nq = get("n_quad");
    out.push(row(&format!("Integration method: {}", e.macros.get("intmethod").cloned().unwrap_or_default()), "Integration pts.", fmt_g(nq, 9)));
    out.push(String::new());
    out.push(row("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), format!("{:.2}", get("chi2"))));
    out.push(row(&format!("Log likelihood = {}", fmt_g(get("ll"), 9)), "Prob > chi2", format!("{:.4}", get("p"))));
    // fixed part
    let names = b.colnames.clone();
    let kf = names.iter().filter(|n| !n.starts_with("/:")).count();
    let fnames: Vec<String> = names[..kf].iter().map(|n| n.split_once(':').map(|x| x.1.to_string()).unwrap_or_else(|| n.clone())).collect();
    let fb: Vec<f64> = b.data[..kf].to_vec();
    let mut fv = Matrix::zeros(kf, kf);
    for i in 0..kf {
        for j in 0..kf {
            fv[(i, j)] = v[(i, j)];
        }
    }
    let is_base: Vec<bool> = fnames.iter().map(|c| c.split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = fnames.iter().zip(&is_base).map(|(c, ib)| *ib || c.starts_with("o.")).collect();
    let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &fnames, b: &fb, v: &fv, omitted: &omitted, is_base: &is_base, df: None, level, eform: if eform { Some("Odds ratio") } else { None }, coef_title: "Coefficient" });
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut last = String::new();
    for j in kf..names.len() {
        // "/:var(zn[level])"
        let inner = names[j].trim_start_matches("/:").trim_start_matches("var(").trim_end_matches(')');
        let (zn, lvar) = inner.split_once('[').map(|(a, c)| (a.to_string(), c.trim_end_matches(']').to_string())).unwrap_or((inner.to_string(), String::new()));
        if lvar != last {
            tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            tbl.push(format!("{:<lw$} |", crate::summarize::abbrev(&lvar, lw)));
            last = lvar;
        }
        let var = b.data[j];
        let se = v[(j, j)].max(0.0).sqrt();
        let sl = se / (2.0 * var);
        let lsd = 0.5 * var.ln();
        let lo = (2.0 * (lsd - crit * sl)).exp();
        let hi = (2.0 * (lsd + crit * sl)).exp();
        let label = format!("var({})", crate::summarize::abbrev(&zn, lw.saturating_sub(4)));
        tbl.push(format!("{:>w$}| {:>10}  {:>9}{:21}{:>9}   {:>9}", label, fmt_g(var, 8), fmt_g(se, 8), "", fmt_g(lo, 8), fmt_g(hi, 8), w = lw + 1));
    }
    tbl.push(bar);
    out.extend(tbl);
    if eform {
        out.push("Note: Estimates are transformed only in the first equation to odds ratios.".into());
        out.push("Note: _cons estimates baseline odds (conditional on zero random effects).".into());
    }
    let chi2c = get("chi2_c");
    let dfc = get("df_c");
    if dfc == 1.0 {
        out.push(format!("LR test vs. logistic model: chibar2(01) = {:.2}     Prob >= chibar2 = {:.4}", chi2c, get("p_c")));
    } else if !is_missing(dfc) {
        out.push(format!("LR test vs. logistic model: chi2({}) = {:.2}{:>27}", fmt_g(dfc, 9), chi2c, format!("Prob > chi2 = {:.4}", get("p_c"))));
        out.push(String::new());
        out.push("Note: LR test is conservative and provided only for reference.".into());
    }
    out.join("\n")
}
