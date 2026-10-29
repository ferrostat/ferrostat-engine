//! `xtmlogit` (panel multinomial logit: conditional fixed effects or
//! correlated random intercepts per outcome) and `xtvar` (panel vector
//! autoregression by GMM on forward-orthogonal deviations).
//!
//! The fixed-effects estimator conditions on the multiset of outcomes
//! observed in each panel; its denominator sums over all orderings of that
//! multiset, computed by dynamic programming over the counts used so far.
//! The random-effects estimator integrates normal random intercepts (one
//! per non-base outcome) by adaptive quadrature, sharing the panel
//! likelihood machinery of the extended regression models.
//!
//! `xtvar` removes the panel effects by forward orthogonal deviations
//! (Helmert transformation) and instruments the transformed lags with
//! untransformed lagged levels, equation by equation (two-stage least
//! squares; with the same instruments in every equation this equals system
//! GMM with a block-diagonal weight), with standard errors clustered on the
//! panel.

use std::collections::HashMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::erm::{gh_normal, naturalize, PanelLik, PanelObs};
use crate::est::{self, Design, Likelihood, MlResult, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("xtmlogit", Box::new(Xtmlogit)), ("xtvar", Box::new(Xtvar))]
}

fn fmt_cat(v: f64) -> String {
    fmt_g(v, 9).trim().to_string()
}

// ---------------------------------------------------------------------------
// xtmlogit
// ---------------------------------------------------------------------------

struct MlogitObs {
    /// category index of each observation (into the non-base equations; None = base)
    eqk: Vec<Option<usize>>,
}

impl PanelObs for MlogitObs {
    type Prep = ();
    fn n_obs(&self) -> usize {
        self.eqk.len()
    }
    fn prep(&self, _aux: &[f64]) -> Option<()> {
        Some(())
    }
    fn ll(&self, t: usize, eta: &[f64], _aux: &[f64], _p: &()) -> f64 {
        let mx = eta.iter().cloned().fold(0.0f64, f64::max);
        let den = (-mx).exp() + eta.iter().map(|e| (e - mx).exp()).sum::<f64>();
        let num = self.eqk[t].map_or(0.0, |k| eta[k]);
        num - mx - den.ln()
    }
}

/// Conditional (fixed-effects) multinomial logit likelihood.
struct FeLik<'a> {
    x: Vec<&'a [f64]>,
    eqk: Vec<Option<usize>>,
    neq: usize,
    panels: Vec<Vec<usize>>,
}

impl FeLik<'_> {
    fn panel_ll(&self, b: &[f64], g: usize) -> f64 {
        let p = self.x.len();
        let rows = &self.panels[g];
        let ncat = self.neq + 1;
        // scores s[t][c] for c = 0 (base) .. neq
        let s: Vec<Vec<f64>> = rows
            .iter()
            .map(|&t| {
                let mut v = vec![0.0; ncat];
                for k in 0..self.neq {
                    v[k + 1] = (0..p).map(|j| self.x[j][t] * b[k * p + j]).sum();
                }
                v
            })
            .collect();
        let cat = |t: usize| self.eqk[t].map_or(0, |k| k + 1);
        let mut counts = vec![0usize; ncat];
        for &t in rows {
            counts[cat(t)] += 1;
        }
        let num: f64 = rows.iter().enumerate().map(|(i, &t)| s[i][cat(t)]).sum();
        // DP over count vectors (mixed radix)
        let radix: Vec<usize> = counts.iter().map(|c| c + 1).collect();
        let mut dp: HashMap<usize, f64> = HashMap::new();
        dp.insert(0, 0.0);
        for si in s.iter() {
            let mut nd: HashMap<usize, f64> = HashMap::new();
            for (&key, &lv) in &dp {
                let mut rem = key;
                let mut mult = 1;
                for c in 0..ncat {
                    let used = rem % radix[c];
                    rem /= radix[c];
                    if used < counts[c] {
                        let nk = key + mult;
                        let val = lv + si[c];
                        let e = nd.entry(nk).or_insert(f64::NEG_INFINITY);
                        let m = e.max(val);
                        *e = if m == f64::NEG_INFINITY { m } else { m + ((*e - m).exp() + (val - m).exp()).ln() };
                    }
                    mult *= radix[c];
                }
            }
            dp = nd;
        }
        let full: usize = {
            let mut k = 0;
            let mut mult = 1;
            for c in 0..ncat {
                k += counts[c] * mult;
                mult *= radix[c];
            }
            k
        };
        num - dp.get(&full).copied().unwrap_or(0.0)
    }

    fn lls(&self, b: &[f64]) -> Vec<f64> {
        (0..self.panels.len()).map(|g| self.panel_ll(b, g)).collect()
    }

    fn scores(&self, b: &[f64]) -> Vec<Vec<f64>> {
        let k = b.len();
        let g = self.panels.len();
        let mut sc = vec![vec![0.0; k]; g];
        for j in 0..k {
            let h = 1e-5 * b[j].abs().max(1.0);
            let mut bp = b.to_vec();
            let mut bm = b.to_vec();
            bp[j] += h;
            bm[j] -= h;
            let (lp, lm) = (self.lls(&bp), self.lls(&bm));
            for i in 0..g {
                sc[i][j] = (lp[i] - lm[i]) / (2.0 * h);
            }
        }
        sc
    }
}

impl Likelihood for FeLik<'_> {
    fn k(&self) -> usize {
        self.x.len() * self.neq
    }
    fn eval(&self, b: &[f64], _need: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = b.len();
        let ll: f64 = self.lls(b).iter().sum();
        let sc = self.scores(b);
        let grad: Vec<f64> = (0..k).map(|j| sc.iter().map(|s| s[j]).sum()).collect();
        let mut info = Matrix::zeros(k, k);
        for j in 0..k {
            let h = 1e-4 * b[j].abs().max(1.0);
            let mut bp = b.to_vec();
            let mut bm = b.to_vec();
            bp[j] += h;
            bm[j] -= h;
            let gp: Vec<f64> = { let s = self.scores(&bp); (0..k).map(|c| s.iter().map(|r| r[c]).sum()).collect() };
            let gm: Vec<f64> = { let s = self.scores(&bm); (0..k).map(|c| s.iter().map(|r| r[c]).sum()).collect() };
            for c in 0..k {
                info[(j, c)] = -(gp[c] - gm[c]) / (2.0 * h);
            }
        }
        for a in 0..k {
            for c in 0..a {
                let v = 0.5 * (info[(a, c)] + info[(c, a)]);
                info[(a, c)] = v;
                info[(c, a)] = v;
            }
        }
        (ll, grad, info, sc)
    }
}

struct Xtmlogit;

impl CommandHandler for Xtmlogit {
    fn name(&self) -> &'static str {
        "xtmlogit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("xtmlogit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let fe = cmd.has_opt("fe", 2);
        let pvar = crate::xtnl::panel_var(ds, st, cmd)?;
        let (dep, xs) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &xs, noconstant: fe || cmd.has_opt("noconstant", 3), weights: "", extra: vec![pvar.clone()] })?;
        let n = d.n();
        let pc = util::col(ds, &pvar)?;
        let pid: Vec<f64> = d.rows.iter().map(|&r| pc[r]).collect();
        let mut cats: Vec<f64> = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        if cats.len() < 2 {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        if cats.len() > 30 {
            return Err(rc_err(149, "too many outcomes"));
        }
        let base = match util::opt_num(cmd, "baseoutcome", 2)? {
            Some(b) => cats.iter().position(|c| *c == b).ok_or_else(|| rc_err(111, "baseoutcome() not an outcome"))?,
            None => {
                let freq: Vec<usize> = cats.iter().map(|c| d.y.iter().filter(|y| **y == *c).count()).collect();
                (0..cats.len()).max_by_key(|&i| (freq[i], usize::MAX - i)).unwrap()
            }
        };
        let neq = cats.len() - 1;
        let eq_cats: Vec<f64> = cats.iter().enumerate().filter(|(i, _)| *i != base).map(|(_, c)| *c).collect();
        let eqk_all: Vec<Option<usize>> = d.y.iter().map(|y| eq_cats.iter().position(|c| c == y)).collect();
        // panels
        let mut keys = pid.clone();
        keys.sort_by(f64::total_cmp);
        keys.dedup();
        let mut panels: Vec<Vec<usize>> = vec![Vec::new(); keys.len()];
        for (t, v) in pid.iter().enumerate() {
            panels[keys.binary_search_by(|k| k.total_cmp(v)).unwrap()].push(t);
        }
        let level = util::level(cmd, st)?;
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let robust = vce != Vce::Oim;
        let eq_names: Vec<String> = eq_cats.iter().map(|c| fmt_cat(*c)).collect();
        let (mut fit, ng, ndrop, nobs_used, title) = if fe {
            // drop panels without variation in the outcome
            let mut keep = vec![true; n];
            let mut ndrop = 0;
            let mut used: Vec<Vec<usize>> = Vec::new();
            for p in &panels {
                let first = d.y[p[0]];
                if p.iter().all(|&t| d.y[t] == first) {
                    ndrop += 1;
                    for &t in p {
                        keep[t] = false;
                    }
                } else {
                    used.push(p.clone());
                }
            }
            let act = d.active();
            let x: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
            if x.is_empty() {
                return Err(rc_err(102, "independent variables required"));
            }
            let nused = keep.iter().filter(|k| **k).count();
            let lik = FeLik { x, eqk: eqk_all.clone(), neq, panels: used.clone() };
            let res = xm::maximize(&lik, vec![0.0; act.len() * neq], 200)?;
            let eqs: Vec<(&str, &Design)> = eq_names.iter().map(|s| (s.as_str(), &d)).collect();
            let mut fit = xm::finish_fit(&eqs, &[], res, &Vce::Oim, None)?;
            if robust {
                let (hinv, _) = fit.res.info.sym_inverse(1e-14);
                let g = used.len() as f64;
                fit.vt = est::sandwich(&hinv, &fit.res.scores, None, g / (g - 1.0)).0;
                fit.v = est::expand_v(&fit.vt, &fit.pos, fit.names.len());
                fit.n_clust = Some(used.len());
            }
            (fit, used.len(), ndrop, nused, "Conditional fixed-effects multinomial logit")
        } else {
            let act = d.active();
            let model = MlogitObs { eqk: eqk_all.clone() };
            let r = neq;
            let q = util::opt_num(cmd, "intpoints", 4)?.map(|v| v as usize).unwrap_or(match r {
                1 => 12,
                2 => 7,
                _ => 4,
            });
            let (nodes, weights) = gh_normal(q);
            let x: Vec<Vec<&[f64]>> = (0..neq).map(|_| act.iter().map(|&j| d.cols[j].as_slice()).collect()).collect();
            let lik = PanelLik { model: &model, x, naux: 0, r, panels: panels.clone(), nodes, weights };
            let mut b0 = Vec::new();
            let nb = cats.iter().map(|c| d.y.iter().filter(|y| **y == *c).count()).collect::<Vec<_>>();
            for (k, c) in eq_cats.iter().enumerate() {
                let _ = k;
                let ci = cats.iter().position(|v| v == c).unwrap();
                for &j in &act {
                    b0.push(if d.names[j] == "_cons" { ((nb[ci] as f64 + 0.5) / (nb[base] as f64 + 0.5)).ln() } else { 0.0 });
                }
            }
            for _ in 0..r {
                b0.push(0.5f64.ln());
            }
            for _ in 0..r * (r - 1) / 2 {
                b0.push(0.0);
            }
            let res = xm::maximize(&lik, b0, 200)?;
            let mut aux: Vec<String> = eq_names.iter().map(|e| format!("relnsd:u{e}")).collect();
            for j in 1..r {
                for i in 0..j {
                    aux.push(format!("reathrho:u{}[panel],u{}[panel]", eq_names[j], eq_names[i]));
                }
            }
            let auxr: Vec<&str> = aux.iter().map(|s| s.as_str()).collect();
            let eqs: Vec<(&str, &Design)> = eq_names.iter().map(|s| (s.as_str(), &d)).collect();
            let mut fit = xm::finish_fit(&eqs, &auxr, res, &Vce::Oim, None)?;
            if robust {
                let (hinv, _) = fit.res.info.sym_inverse(1e-14);
                let g = panels.len() as f64;
                fit.vt = est::sandwich(&hinv, &fit.res.scores, None, g / (g - 1.0)).0;
                fit.v = est::expand_v(&fit.vt, &fit.pos, fit.names.len());
                fit.n_clust = Some(panels.len());
            }
            naturalize(&mut fit);
            for nm in fit.names.iter_mut() {
                *nm = nm.replace("[panel]", &format!("[{pvar}]"));
            }
            (fit, panels.len(), 0, n, "Random-effects multinomial logistic regression")
        };
        let rrr = cmd.has_opt("rrr", 3);
        // Wald test of all slope coefficients
        let idx: Vec<usize> = (0..fit.pos.len()).filter(|&a| {
            let nm = &fit.names[fit.pos[a]];
            !nm.starts_with("/:") && !nm.ends_with(":_cons")
        }).collect();
        let (chi2, _) = crate::heckman::wald(&fit.bt, &fit.vt, &idx);
        let df = idx.len() as f64;
        let mut out = est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" });
        out.push(String::new());
        if ndrop > 0 {
            out.insert(0, format!("note: {ndrop} group{} dropped because of no variation in the outcome.", if ndrop == 1 { "" } else { "s" }));
        }
        let ncl = if robust { Some((ng, pvar.clone())) } else { None };
        let lines = vec![(format!("Group variable: {pvar}"), "Number of groups".into(), util::commas(ng))];
        out.extend(xm::Header { title, n: nobs_used as f64, lines, test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: ncl }.render());
        let mut tbl = xm::table(&dep, &fit, level, if rrr { Some("RRR") } else { None }, &[]);
        let bar = tbl.pop().unwrap_or_default();
        tbl.push(bar);
        out.extend(tbl);
        out.push(format!("Note: {} is the base outcome.", fmt_cat(cats[base])));
        let text = out.join("\n");
        let catm = Matrix::row_vector(&cats);
        let model_name = if fe { "fe" } else { "re" };
        let vce2 = if robust { Vce::Cluster(pvar.clone()) } else { Vce::Oim };
        let basec = cats[base];
        xm::post_fit(st, "xtmlogit", &d, &fit, &vce2, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set_str("depvar", dep.clone());
            e.set_str("model", model_name);
            e.set_str("ivar", pvar.clone());
            e.set("N_g", ng as f64);
            e.set("N", nobs_used as f64);
            e.set("k_out", catm.cols as f64);
            e.set("baseout", basec);
            e.set_mat("out", catm.clone());
            e.set_str("title", title);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

#[allow(dead_code)]
fn unused(_: MlResult) {}

// ---------------------------------------------------------------------------
// xtvar
// ---------------------------------------------------------------------------

struct Xtvar;

/// Forward orthogonal deviation of a series (missing at the last period).
fn fod(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n.saturating_sub(1) {
        let rest = &x[t + 1..];
        if rest.iter().any(|v| v.is_nan()) || x[t].is_nan() {
            continue;
        }
        let m = rest.len() as f64;
        let mean = rest.iter().sum::<f64>() / m;
        out[t] = (m / (m + 1.0)).sqrt() * (x[t] - mean);
    }
    out
}

impl CommandHandler for Xtvar {
    fn name(&self) -> &'static str {
        "xtvar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("xtvar") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let (ivar, tvar) = match &st.tsset {
            Some((Some(p), t)) if !t.is_empty() => (p.clone(), t.clone()),
            _ => return Err(rc_err(459, "panel variable and time variable not set; use xtset panelvar timevar")),
        };
        let ys: Vec<String> = crate::varlist::expand(ds, cmd.main.trim())?;
        if ys.len() < 2 {
            return Err(rc_err(102, "at least two variables required"));
        }
        let p = util::opt_num(cmd, "lags", 1)?.unwrap_or(1.0) as usize;
        if p < 1 {
            return Err(rc_err(198, "lags() must be positive"));
        }
        let il = util::opt_num(cmd, "instlags", 5)?.map(|v| v as usize).unwrap_or(p).max(p);
        let exog: Vec<String> = match cmd.opt("exog", 2) {
            Some(e) => crate::varlist::expand(ds, e)?,
            None => vec![],
        };
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = ys.clone();
        all.extend(exog.iter().cloned());
        all.push(ivar.clone());
        all.push(tvar.clone());
        util::markout(ds, &mut mask, &all);
        let pc = util::col(ds, &ivar)?;
        let tc = util::col(ds, &tvar)?;
        let ycols: Vec<Vec<f64>> = ys.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let xcols: Vec<Vec<f64>> = exog.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mut by: std::collections::BTreeMap<i64, Vec<(f64, usize)>> = std::collections::BTreeMap::new();
        for i in 0..ds.n_obs {
            if mask[i] {
                by.entry((pc[i] * 1000.0).round() as i64).or_default().push((tc[i], i));
            }
        }
        let m = ys.len();
        let kx = exog.len();
        // stacked rows: (panel index, transformed y (m), transformed lags (m*p), exog (kx), instruments)
        let mut yrows: Vec<Vec<f64>> = Vec::new();
        let mut xrows: Vec<Vec<f64>> = Vec::new();
        let mut zrows: Vec<Vec<f64>> = Vec::new();
        let mut gid: Vec<usize> = Vec::new();
        let mut ng = 0;
        for (_, mut obs) in by {
            obs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let t0 = obs[0].0;
            let tmax = obs.last().unwrap().0;
            let len = (tmax - t0) as usize + 1;
            // full time grid with gaps as NaN
            let mut grid: Vec<Option<usize>> = vec![None; len];
            for &(t, i) in &obs {
                grid[(t - t0) as usize] = Some(i);
            }
            let ser = |c: &Vec<f64>| -> Vec<f64> { grid.iter().map(|o| o.map_or(f64::NAN, |i| c[i])).collect() };
            let yl: Vec<Vec<f64>> = ycols.iter().map(ser).collect();
            let xl: Vec<Vec<f64>> = xcols.iter().map(ser).collect();
            let yf: Vec<Vec<f64>> = yl.iter().map(|s| fod(s)).collect();
            // lagged series transformed over the same rows as the dependent variable
            let ylagf: Vec<Vec<Vec<f64>>> = (1..=p)
                .map(|l| {
                    yl.iter()
                        .map(|s| {
                            let lagged: Vec<f64> = (0..len).map(|t| if t >= l { s[t - l] } else { f64::NAN }).collect();
                            fod(&lagged)
                        })
                        .collect()
                })
                .collect();
            let xf: Vec<Vec<f64>> = xl.iter().map(|s| fod(s)).collect();
            let mut used = false;
            for t in il..len {
                // dependent: FOD of y_t; regressors: FOD of y_{t-l}
                let mut yr = Vec::with_capacity(m);
                let mut xr = Vec::with_capacity(m * p + kx);
                let mut zr = Vec::with_capacity(m * il + kx);
                for j in 0..m {
                    yr.push(yf[j][t]);
                }
                for l in 1..=p {
                    for j in 0..m {
                        xr.push(ylagf[l - 1][j][t]);
                    }
                }
                for j in 0..kx {
                    xr.push(xf[j][t]);
                }
                for l in 1..=il {
                    for j in 0..m {
                        zr.push(yl[j][t - l]);
                    }
                }
                for j in 0..kx {
                    zr.push(xf[j][t]);
                }
                if yr.iter().chain(&xr).chain(&zr).any(|v| v.is_nan()) {
                    continue;
                }
                yrows.push(yr);
                xrows.push(xr);
                zrows.push(zr);
                gid.push(ng);
                used = true;
            }
            if used {
                ng += 1;
            }
        }
        let n = yrows.len();
        let kreg = m * p + kx;
        let kz = m * il + kx;
        if n <= kz + 1 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // 2SLS (same instruments for every equation)
        let zz = {
            let mut a = Matrix::zeros(kz, kz);
            for z in &zrows {
                for i in 0..kz {
                    for j in 0..kz {
                        a[(i, j)] += z[i] * z[j];
                    }
                }
            }
            a
        };
        let (zzi, _) = zz.sym_inverse(1e-12);
        let mut zx = Matrix::zeros(kz, kreg);
        for (z, x) in zrows.iter().zip(&xrows) {
            for i in 0..kz {
                for j in 0..kreg {
                    zx[(i, j)] += z[i] * x[j];
                }
            }
        }
        let zxt = zx.transpose();
        let a = zxt.mul(&zzi).unwrap().mul(&zx).unwrap();
        let (ai, _) = a.sym_inverse(1e-12);
        let proj = ai.mul(&zxt).unwrap().mul(&zzi).unwrap(); // kreg x kz
        let mut names = Vec::new();
        let mut bfull = Vec::new();
        let mut eq_b: Vec<Vec<f64>> = Vec::new();
        let mut resid: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
        for e in 0..m {
            let mut zy = vec![0.0; kz];
            for (z, y) in zrows.iter().zip(&yrows) {
                for i in 0..kz {
                    zy[i] += z[i] * y[e];
                }
            }
            let b: Vec<f64> = (0..kreg).map(|i| (0..kz).map(|j| proj[(i, j)] * zy[j]).sum()).collect();
            for t in 0..n {
                resid[t][e] = yrows[t][e] - (0..kreg).map(|j| xrows[t][j] * b[j]).sum::<f64>();
            }
            for l in 1..=p {
                for y in &ys {
                    names.push(format!("{}:{}.{y}", ys[e], if l == 1 { "L".to_string() } else { format!("L{l}") }));
                }
            }
            for x in &exog {
                names.push(format!("{}:{x}", ys[e]));
            }
            bfull.extend(b.iter().cloned());
            eq_b.push(b);
        }
        // cluster-robust variance on the stacked system
        let k = m * kreg;
        let mut meat = Matrix::zeros(k, k);
        let mut sg: Vec<Vec<f64>> = vec![vec![0.0; k]; ng];
        for t in 0..n {
            // score contribution: (P z_t) * resid_e for each equation e
            let pz: Vec<f64> = (0..kreg).map(|i| (0..kz).map(|j| proj[(i, j)] * zrows[t][j]).sum()).collect();
            for e in 0..m {
                for i in 0..kreg {
                    sg[gid[t]][e * kreg + i] += pz[i] * resid[t][e];
                }
            }
        }
        for s in &sg {
            for i in 0..k {
                for j in 0..k {
                    meat[(i, j)] += s[i] * s[j];
                }
            }
        }
        let g = ng as f64;
        let v = meat.scale(g / (g - 1.0));
        // output
        let mut out = vec![String::new(), format!("{:<48}Number of obs     = {:>9}", "Panel vector autoregression", util::commas(n))];
        out.push(format!("{:<48}Number of panels  = {:>9}", "GMM estimation (forward orthogonal deviations)", util::commas(ng)));
        out.push(format!("{:<48}Number of instr.  = {:>9}", format!("Panel variable: {ivar}"), kz));
        out.push(format!("{:<48}Lags              = {:>9}", format!("Time variable: {tvar}"), p));
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {ivar})", util::commas(ng))));
        let omitted = vec![false; k];
        let is_base = vec![false; k];
        let tbl = est::coef_table(&est::TableSpec { depvar: "", names: &names, b: &bfull, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" });
        out.extend(tbl);
        out.push(format!("Instruments: lags 1 to {il} of {} (levels){}", ys.join(" "), if exog.is_empty() { String::new() } else { format!("; {}", exog.join(" ")) }));
        let text = out.join("\n");
        let mut e = Results::new();
        e.set_mat("b", Matrix::row_vector(&bfull).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names.clone()));
        e.set("N", n as f64);
        e.set("N_g", g);
        e.set("N_clust", g);
        e.set("lags", p as f64);
        e.set("k_eq", m as f64);
        e.set("n_inst", kz as f64);
        let mut sigma = Matrix::zeros(m, m);
        for r in &resid {
            for i in 0..m {
                for j in 0..m {
                    sigma[(i, j)] += r[i] * r[j] / n as f64;
                }
            }
        }
        e.set_mat("Sigma", sigma.with_names(ys.clone(), ys.clone()));
        e.set_str("cmd", "xtvar");
        e.set_str("depvar", ys.join(" "));
        e.set_str("endog", ys.join(" "));
        e.set_str("exog", exog.join(" "));
        e.set_str("ivar", ivar);
        e.set_str("tvar", tvar);
        e.set_str("vce", "cluster");
        e.set_str("vcetype", "Robust");
        e.set_str("title", "Panel vector autoregression");
        e.set_str("_output", text.clone());
        let mut smask = vec![false; ds.n_obs];
        for i in 0..ds.n_obs {
            smask[i] = mask[i];
        }
        e.sample = Some(smask);
        st.e = e;
        crate::est::stash_table_from_e(&st.e, level);
        let _ = (eq_b, is_missing(0.0));
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_orthogonal_deviations() {
        let f = fod(&[1.0, 2.0, 3.0]);
        // t=0: sqrt(2/3) * (1 - 2.5)
        assert!((f[0] - (2.0f64 / 3.0).sqrt() * -1.5).abs() < 1e-12);
        assert!((f[1] - (0.5f64).sqrt() * -1.0).abs() < 1e-12);
        assert!(f[2].is_nan());
    }
}
