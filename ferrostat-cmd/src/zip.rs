//! Zero-inflated count models: `zip` (Poisson) and `zinb` (negative
//! binomial, NB2), with a logit or probit inflation equation.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::ln_gamma;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};

// ---------------------------------------------------------------------------
// Generic multiple-index likelihood
// ---------------------------------------------------------------------------

/// Per-observation log likelihood of a few linear indexes (and auxiliary
/// parameters): returns (ll, d ll/d idx, d2 ll/d idx d idx).
pub type IndexFn<'a> = dyn Fn(usize, &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) + 'a;

/// A model whose observations depend on the parameters only through linear
/// indexes `eq_e = X_e b_e`, plus auxiliary scalar parameters.
pub struct MultiIndex<'a> {
    /// per equation: active columns over the estimation sample
    pub x: Vec<Vec<&'a [f64]>>,
    pub naux: usize,
    pub w: &'a [f64],
    pub fweight: bool,
    pub f: &'a IndexFn<'a>,
}

impl<'a> MultiIndex<'a> {
    fn layout(&self) -> Vec<(usize, Option<usize>)> {
        // parameter -> (index slot, column within equation | None for aux)
        let mut out = Vec::new();
        for (e, cols) in self.x.iter().enumerate() {
            for c in 0..cols.len() {
                out.push((e, Some(c)));
            }
        }
        for a in 0..self.naux {
            out.push((self.x.len() + a, None));
        }
        out
    }
    fn indexes(&self, b: &[f64], t: usize) -> Vec<f64> {
        let mut idx = Vec::with_capacity(self.x.len() + self.naux);
        let mut p = 0;
        for cols in &self.x {
            let mut s = 0.0;
            for c in cols {
                s += c[t] * b[p];
                p += 1;
            }
            idx.push(s);
        }
        idx.extend_from_slice(&b[p..p + self.naux]);
        idx
    }
}

impl<'a> Likelihood for MultiIndex<'a> {
    fn k(&self) -> usize {
        self.x.iter().map(|c| c.len()).sum::<usize>() + self.naux
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.k();
        let lay = self.layout();
        let n = self.w.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let mut scores = Vec::new();
        let xv = |p: usize, t: usize| -> f64 {
            match lay[p] {
                (e, Some(c)) => self.x[e][c][t],
                _ => 1.0,
            }
        };
        for t in 0..n {
            let w = self.w[t];
            let idx = self.indexes(b, t);
            let (l, d1, d2) = (self.f)(t, &idx);
            ll += w * l;
            let mut sc = vec![0.0; k];
            for p in 0..k {
                let xp = xv(p, t);
                sc[p] = d1[lay[p].0] * xp;
                g[p] += w * sc[p];
                for q in 0..=p {
                    h[(p, q)] -= w * d2[lay[p].0][lay[q].0] * xp * xv(q, t);
                }
            }
            if need_scores {
                let fw = if self.fweight { w.sqrt() } else { w };
                scores.push(sc.iter().map(|s| s * fw).collect());
            }
        }
        for p in 0..k {
            for q in 0..p {
                h[(q, p)] = h[(p, q)];
            }
        }
        (ll, g, h, scores)
    }
}

/// Numerical first and second derivatives of a small function (used for
/// the per-observation index derivatives of the negative binomial parts).
pub fn num_derivs(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let m = x.len();
    let f0 = f(x);
    let hs: Vec<f64> = x.iter().map(|v| 1e-4 * (1.0 + v.abs())).collect();
    let at = |d: &[(usize, f64)]| {
        let mut y = x.to_vec();
        for &(i, s) in d {
            y[i] += s;
        }
        f(&y)
    };
    let mut g = vec![0.0; m];
    let mut h = vec![vec![0.0; m]; m];
    for i in 0..m {
        let (fp, fm) = (at(&[(i, hs[i])]), at(&[(i, -hs[i])]));
        g[i] = (fp - fm) / (2.0 * hs[i]);
        h[i][i] = (fp - 2.0 * f0 + fm) / (hs[i] * hs[i]);
        for j in 0..i {
            let v = (at(&[(i, hs[i]), (j, hs[j])]) - at(&[(i, hs[i]), (j, -hs[j])]) - at(&[(i, -hs[i]), (j, hs[j])]) + at(&[(i, -hs[i]), (j, -hs[j])])) / (4.0 * hs[i] * hs[j]);
            h[i][j] = v;
            h[j][i] = v;
        }
    }
    (f0, g, h)
}

/// Variance matrix of an ML fit by VCE type.
pub fn ml_vce(res: &est::MlResult, vce: &Vce, cluster: Option<&[f64]>, n: f64) -> (Matrix, Option<usize>) {
    let (hinv, _) = res.info.sym_inverse(1e-14);
    match vce {
        Vce::Ols | Vce::Oim => (hinv, None),
        Vce::Robust => (est::sandwich(&hinv, &res.scores, None, n / (n - 1.0)).0, None),
        Vce::Cluster(_) => {
            let (v, g) = est::sandwich(&hinv, &res.scores, cluster, 1.0);
            let gf = g as f64;
            (v.scale(gf / (gf - 1.0)), Some(g))
        }
    }
}

/// Plain variable names used by a varlist (for marking out missing values).
pub fn base_vars(ds: &DataSet, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        for part in tok.split('#') {
            let b = util::base_name(part).trim_start_matches('(').trim_end_matches(')');
            if let Ok(v) = crate::varlist::expand(ds, b) {
                out.extend(v);
            }
        }
    }
    out
}

fn f2(v: f64) -> String {
    if is_missing(v) || v.is_nan() { ".".into() } else { format!("{v:.2}") }
}
fn f4(v: f64) -> String {
    if is_missing(v) || v.is_nan() { ".".into() } else { format!("{v:.4}") }
}

// ---------------------------------------------------------------------------
// zip / zinb
// ---------------------------------------------------------------------------

pub struct Zip {
    pub nb: bool,
}

fn log1pexp(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}

fn lse(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Log probabilities (ln p, ln(1-p)) of the inflation equation.
fn infl_logs(probit: bool, z: f64) -> (f64, f64) {
    if probit {
        (dist::normal(z).max(1e-300).ln(), dist::normal(-z).max(1e-300).ln())
    } else {
        (-log1pexp(-z), -log1pexp(z))
    }
}

/// ZIP observation: analytic derivatives (logit inflation).
fn zip_obs(y: f64, e1: f64, e2: f64) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let lam = e1.exp();
    let p = dist::invlogit(e2);
    if y > 0.0 {
        let ll = -log1pexp(e2) + y * e1 - lam - ln_gamma(y + 1.0);
        (ll, vec![y - lam, -p], vec![vec![-lam, 0.0], vec![0.0, -p * (1.0 - p)]])
    } else {
        let la = lse(e2, -lam);
        let ll = la - log1pexp(e2);
        let q = (-lam - la).exp();
        let r = (e2 - la).exp();
        let d11 = -lam * q + lam * lam * q * r;
        let d12 = lam * q * r;
        let d22 = r * q - p * (1.0 - p);
        (ll, vec![-lam * q, r - p], vec![vec![d11, d12], vec![d12, d22]])
    }
}

/// Count-part log density: Poisson or NB2 (idx[2] = ln alpha).
fn count_ll(nb: bool, y: f64, e1: f64, lnalpha: f64) -> f64 {
    if nb {
        let a = lnalpha.exp();
        let m = 1.0 / a;
        let mu = e1.exp();
        ln_gamma(y + m) - ln_gamma(m) - ln_gamma(y + 1.0) - (m + y) * (a * mu).ln_1p() + y * (a * mu).ln()
    } else {
        y * e1 - e1.exp() - ln_gamma(y + 1.0)
    }
}

fn zi_ll(nb: bool, probit: bool, y: f64, idx: &[f64]) -> f64 {
    let (lp, lq) = infl_logs(probit, idx[1]);
    let la = if nb { idx[2] } else { 0.0 };
    if y > 0.0 {
        lq + count_ll(nb, y, idx[0], la)
    } else {
        lse(lp, lq + count_ll(nb, 0.0, idx[0], la))
    }
}

fn zi_obs(nb: bool, probit: bool, y: f64, idx: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    if !nb && !probit {
        return zip_obs(y, idx[0], idx[1]);
    }
    num_derivs(&|v: &[f64]| zi_ll(nb, probit, y, v), idx)
}

impl CommandHandler for Zip {
    fn name(&self) -> &'static str {
        if self.nb { "zinb" } else { "zip" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let infl = cmd.opt("inflate", 3).map(str::trim).filter(|s| !s.is_empty()).ok_or_else(|| rc_err(198, "inflate() required"))?.to_string();
        let infl_cons_only = infl == "_cons";
        let infl_vars = if infl_cons_only { String::new() } else { infl.replace("_cons", " ") };
        let probit = cmd.has_opt("probit", 3);
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let mut extra = base_vars(ds, &infl_vars);
        if let Vce::Cluster(c) = &vce {
            extra.push(c.clone());
        }
        let d1 = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fip", extra })?;
        let mut extra2 = base_vars(ds, indep);
        if let Vce::Cluster(c) = &vce {
            extra2.push(c.clone());
        }
        let d2 = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &infl_vars, noconstant: false, weights: "fip", extra: extra2 })?;
        if d1.rows != d2.rows {
            return Err(rc_err(459, "estimation samples of the equations differ"));
        }
        if d1.y.iter().any(|&y| y < 0.0 || y.fract() != 0.0) {
            return Err(rc_err(459, format!("{dep} must be a nonnegative integer")));
        }
        let vce = if d1.wkind == Some(WeightKind::P) && vce == Vce::Oim { Vce::Robust } else { vce };
        let robust = vce != Vce::Oim;
        let cl = est::cluster_values(ds, &d1, &vce)?;
        let n = d1.n();
        let act1 = d1.active();
        let act2 = d2.active();
        let y = d1.y.clone();
        let naux = if self.nb { 1 } else { 0 };
        let nb = self.nb;
        let f = move |t: usize, idx: &[f64]| zi_obs(nb, probit, y[t], idx);
        let fw = d1.wkind == Some(WeightKind::F);
        let cols1: Vec<&[f64]> = act1.iter().map(|&j| d1.cols[j].as_slice()).collect();
        let cols2: Vec<&[f64]> = act2.iter().map(|&j| d2.cols[j].as_slice()).collect();
        let cons1 = act1.iter().position(|&j| d1.names[j] == "_cons");
        let wsum: f64 = d1.w.iter().sum();
        let ybar = d1.y.iter().zip(&d1.w).map(|(a, b)| a * b).sum::<f64>() / wsum;
        let mut out: Vec<String> = d1.notes.clone();
        // constant-only model (count equation), full inflation equation
        let mut ll0 = SYSMISS;
        let mut b_c = None;
        if let Some(c1) = cons1 {
            let x0: Vec<&[f64]> = vec![cols1[c1]];
            let m0 = MultiIndex { x: vec![x0, cols2.clone()], naux, w: &d1.w, fweight: fw, f: &f };
            let mut s0 = vec![0.0; m0.k()];
            s0[0] = ybar.max(1e-8).ln();
            let r0 = est::maximize(&m0, s0, 300)?;
            out.push(String::new());
            out.push("Fitting constant-only model:".into());
            out.push(String::new());
            out.extend(est::iteration_log(&r0.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
            ll0 = r0.ll;
            b_c = Some(r0.b);
        }
        // starting values: count equation from a Poisson fit, inflation and
        // dispersion from the constant-only model
        let mpois = MultiIndex {
            x: vec![cols1.clone()],
            naux: 0,
            w: &d1.w,
            fweight: fw,
            f: &|t: usize, idx: &[f64]| {
                let mu = idx[0].exp();
                (d1.y[t] * idx[0] - mu - ln_gamma(d1.y[t] + 1.0), vec![d1.y[t] - mu], vec![vec![-mu]])
            },
        };
        let mut sp = vec![0.0; cols1.len()];
        if let Some(c) = cons1 {
            sp[c] = ybar.max(1e-8).ln();
        }
        let pois = est::maximize(&mpois, sp, 300)?;
        let model = MultiIndex { x: vec![cols1.clone(), cols2.clone()], naux, w: &d1.w, fweight: fw, f: &f };
        let mut b0 = pois.b.clone();
        match &b_c {
            Some(bc) => b0.extend_from_slice(&bc[1..]),
            None => b0.extend(vec![0.0; cols2.len() + naux]),
        }
        let res = est::maximize(&model, b0, 300)?;
        out.push(String::new());
        out.push("Fitting full model:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
        let neff = d1.n_eff();
        let (vact, nclust) = ml_vce(&res, &vce, cl.as_deref(), neff);
        // full coefficient vector
        let eq1 = dep.clone();
        let (k1, k2) = (d1.k(), d2.k());
        let total = k1 + k2 + naux;
        let mut names: Vec<String> = d1.names.iter().map(|nm| format!("{eq1}:{nm}")).collect();
        names.extend(d2.names.iter().map(|nm| format!("inflate:{nm}")));
        if self.nb {
            names.push("/:lnalpha".into());
        }
        let mut idx: Vec<usize> = act1.clone();
        idx.extend(act2.iter().map(|j| j + k1));
        if self.nb {
            idx.push(k1 + k2);
        }
        let mut b = vec![0.0; total];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vact, &idx, total);
        let mut omitted = d1.omitted.clone();
        omitted.extend(d2.omitted.iter().cloned());
        let mut is_base = d1.is_base.clone();
        is_base.extend(d2.is_base.iter().cloned());
        if self.nb {
            omitted.push(false);
            is_base.push(false);
        }
        // model test on the count equation
        let cidx: Vec<usize> = act1.iter().filter(|&&j| d1.names[j] != "_cons").cloned().collect();
        let df_m = cidx.len() as f64;
        let chi2 = if robust || is_missing(ll0) {
            let kk = cidx.len();
            let mut vs = Matrix::zeros(kk, kk);
            for (a, &i) in cidx.iter().enumerate() {
                for (c, &j) in cidx.iter().enumerate() {
                    vs[(a, c)] = v[(i, j)];
                }
            }
            let (vi, _) = vs.sym_inverse(1e-14);
            (0..kk).map(|a| (0..kk).map(|c| b[cidx[a]] * vi[(a, c)] * b[cidx[c]]).sum::<f64>()).sum()
        } else {
            2.0 * (res.ll - ll0)
        };
        let p = dist::chi2tail(df_m, chi2);
        let nzero = d1.y.iter().zip(&d1.w).filter(|(y, _)| **y == 0.0).map(|(_, w)| if fw { *w } else { 1.0 }).sum::<f64>();
        let level = util::level(cmd, st)?;
        let title = if self.nb { "Zero-inflated negative binomial regression" } else { "Zero-inflated Poisson regression" };
        let row = |l: &str, k: &str, v: String| format!("{:<52}{:<15} = {:>10}", l, k, v);
        let g10 = |v: f64| fmt_g(v, 9);
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", title, "Number of obs", g10(neff)));
        out.push(row("", "Nonzero obs", g10(neff - nzero)));
        out.push(row("", "Zero obs", g10(nzero)));
        out.push(String::new());
        let chi_label = if robust || is_missing(ll0) { format!("Wald chi2({})", df_m) } else { format!("LR chi2({})", df_m) };
        out.push(row(&format!("Inflation model = {}", if probit { "probit" } else { "logit" }), &chi_label, f2(chi2)));
        let ll_label = if robust { "Log pseudolikelihood" } else { "Log likelihood " };
        out.push(row(&format!("{ll_label} = {}", fmt_g(res.ll, 9)), "Prob > chi2", f4(p)));
        if let Some(g) = nclust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", g, match &vce { Vce::Cluster(c) => c.as_str(), _ => "" })));
        }
        out.push(String::new());
        let eform = if cmd.has_opt("irr", 3) { Some("IRR") } else { None };
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform, coef_title: "Coefficient" });
        if self.nb {
            // alpha row after /lnalpha
            let la = b[total - 1];
            let se = v[(total - 1, total - 1)].max(0.0).sqrt();
            let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
            let bar = tbl.pop().unwrap_or_default();
            let lw = bar.len() - 66;
            tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            tbl.push(format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", "alpha", fmt_g(la.exp(), 8), fmt_g(la.exp() * se, 8), "", fmt_g((la - crit * se).exp(), 8), fmt_g((la + crit * se).exp(), 8)));
            tbl.push(bar);
        }
        out.extend(tbl);
        // Vuong test against the standard (non-inflated) model
        let mut vuong = None;
        if cmd.has_opt("vuong", 2) {
            let std_b: Vec<f64> = if self.nb {
                let mnb = MultiIndex {
                    x: vec![cols1.clone()],
                    naux: 1,
                    w: &d1.w,
                    fweight: fw,
                    f: &|t: usize, idx: &[f64]| {
                        let yy = d1.y[t];
                        num_derivs(&|v: &[f64]| count_ll(true, yy, v[0], v[1]), idx)
                    },
                };
                let mut s = pois.b.clone();
                s.push(0.0);
                est::maximize(&mnb, s, 300)?.b
            } else {
                pois.b.clone()
            };
            let mut m = Vec::with_capacity(n);
            for t in 0..n {
                let e1: f64 = cols1.iter().zip(&res.b).map(|(c, bb)| c[t] * bb).sum();
                let e2: f64 = cols2.iter().zip(&res.b[cols1.len()..]).map(|(c, bb)| c[t] * bb).sum();
                let la = if self.nb { res.b[res.b.len() - 1] } else { 0.0 };
                let lz = zi_ll(self.nb, probit, d1.y[t], &[e1, e2, la]);
                let s1: f64 = cols1.iter().zip(&std_b).map(|(c, bb)| c[t] * bb).sum();
                let ls = count_ll(self.nb, d1.y[t], s1, if self.nb { std_b[std_b.len() - 1] } else { 0.0 });
                m.push(lz - ls);
            }
            let mean = m.iter().sum::<f64>() / n as f64;
            let sd = (m.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
            let z = (n as f64).sqrt() * mean / sd;
            let pz = dist::normal(-z);
            let what = if self.nb { "Vuong test of zinb vs. standard negative binomial:" } else { "Vuong test of zip vs. standard Poisson:" };
            out.push(format!("{what:<51}z = {:>8.2}  Pr>z = {:.4}", z, pz));
            vuong = Some(z);
        }
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let (vcetype, vcelabel) = est::vce_label(&vce);
        let mut dpost: Design = d1.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        dpost.cols = vec![Vec::new(); total];
        est::post(st, verb, &dpost, &b, &v, &|e: &mut Results| {
            e.set("rank", idx.len() as f64);
            e.set("ll", res.ll);
            if !is_missing(ll0) {
                e.set("ll_0", ll0);
            }
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", p);
            e.set("N_zero", nzero);
            e.set("k_eq", if nb { 3.0 } else { 2.0 });
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if nb {
                e.set("alpha", b[total - 1].exp());
            }
            if let Some(z) = vuong {
                e.set("vuong", z);
            }
            if let Some(g) = nclust {
                e.set("N_clust", g as f64);
            }
            e.set_str("inflate", if probit { "probit" } else { "logit" });
            e.set_str("chi2type", if robust || is_missing(ll0) { "Wald" } else { "LR" });
            e.set_str("title", title);
            e.set_str("vce", vcetype);
            if !vcelabel.is_empty() {
                e.set_str("vcetype", vcelabel);
            }
            e.set_str("predict", if nb { "zinb_p" } else { "zip_p" });
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// Predictions after zip/zinb: n (default), pr (Pr(inflation zero)), xb.
pub fn zi_prediction(st: &Store, env: &crate::eval::Env, stat: &str, row: usize) -> Result<f64, String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let probit = st.e.macros.get("inflate").is_some_and(|s| s == "probit");
    let xb = crate::postest::xb_row(env, b, Some(&dep), row)?;
    let zb = crate::postest::xb_row(env, b, Some("inflate"), row)?;
    if is_missing(xb) || is_missing(zb) {
        return Ok(SYSMISS);
    }
    let p = if probit { dist::normal(zb) } else { dist::invlogit(zb) };
    Ok(match stat {
        "xb" => xb,
        "pr" | "p" => p,
        "ir" => xb.exp(),
        _ => (1.0 - p) * xb.exp(),
    })
}
