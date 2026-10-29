//! Ordered and multinomial variants: zero-inflated ordered logit/probit
//! (`ziologit`, `zioprobit`) and stereotype logistic regression (`slogit`).
//!
//! Zero-inflated ordered models (Harris and Zhao 2007): an inflation
//! equation gives the probability F(z'g) of being a potential participant;
//! nonparticipants always report the lowest category, participants follow
//! an ordered model with cutpoints. The stereotype model (Anderson 1984)
//! sets ln Pr(y=j)/Pr(y=base) = theta_j - sum_k phi_kj x'beta_k.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::choice_mixed::{row_ci, row_z, sep, zcrit};
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Likelihood, MlResult, Vce};

/// Inverse of a symmetric matrix after scaling it to unit diagonal (robust
/// to parameters of very different magnitudes).
pub fn inv_scaled(h: &Matrix) -> Matrix {
    let k = h.rows;
    let d: Vec<f64> = (0..k).map(|i| { let a = h[(i, i)].abs(); if a > 0.0 { 1.0 / a.sqrt() } else { 1.0 } }).collect();
    let mut s = h.clone();
    for i in 0..k {
        for j in 0..k {
            s[(i, j)] *= d[i] * d[j];
        }
    }
    let (si, _) = s.sym_inverse(1e-13);
    let mut out = si;
    for i in 0..k {
        for j in 0..k {
            out[(i, j)] *= d[i] * d[j];
        }
    }
    out
}

/// Variance of an ML fit by VCE type, using `inv_scaled` for the bread.
pub fn ml_vce_scaled(res: &MlResult, vce: &Vce, cluster: Option<&[f64]>, n: f64) -> (Matrix, Option<usize>) {
    let hinv = inv_scaled(&res.info);
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
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::r_trunc::header_new;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("ziologit", Box::new(Zio { probit: false })), ("zioprobit", Box::new(Zio { probit: true })), ("slogit", Box::new(Slogit))]
}

/// Base levels and omitted terms from coefficient names (for replays).
fn flags_from_names(names: &[String]) -> (Vec<bool>, Vec<bool>) {
    let is_base: Vec<bool> = names.iter().map(|c| c.rsplit(':').next().unwrap_or(c).split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = names.iter().zip(&is_base).map(|(c, b)| *b || c.rsplit(':').next().unwrap_or(c).split('#').any(|p| p.starts_with("o.") || p.split_once('.').is_some_and(|(l, _)| l.ends_with('o') && l.trim_end_matches(['o', 'b']).parse::<f64>().is_ok()))).collect();
    (omitted, is_base)
}

/// Coefficient table where the `/:name` parameters are shown with their
/// interval only (no test), as for cutpoints and dispersion parameters.
pub fn table_aux_ci(depvar: &str, names: &[String], b: &[f64], v: &Matrix, level: f64, eform: Option<&str>) -> Vec<String> {
    let ncoef = names.iter().filter(|n| !n.starts_with("/:")).count();
    let (omitted, is_base) = flags_from_names(names);
    let mut tbl = est::coef_table(&est::TableSpec { depvar, names: &names[..ncoef], b: &b[..ncoef], v, omitted: &omitted[..ncoef], is_base: &is_base[..ncoef], df: None, level, eform, coef_title: "Coefficient" });
    let full = est::table_matrix(&est::TableSpec { depvar, names, b, v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" });
    est::stash_table(full, level);
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    let crit = zcrit(level);
    if ncoef < names.len() {
        tbl.push(sep(lw));
    }
    for j in ncoef..names.len() {
        let se = v[(j, j)].max(0.0).sqrt();
        tbl.push(row_ci(lw, &format!("/{}", &names[j][2..]), b[j], se, b[j] - crit * se, b[j] + crit * se));
    }
    tbl.push(bar);
    tbl
}

fn cdf(probit: bool, z: f64) -> f64 {
    if probit {
        xm::ncdf(z)
    } else if z == f64::INFINITY {
        1.0
    } else if z == f64::NEG_INFINITY {
        0.0
    } else {
        dist::invlogit(z)
    }
}

fn inv_cdf(probit: bool, p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    if probit { dist::invnormal(p) } else { (p / (1.0 - p)).ln() }
}

/// Ordered-model probability of category k given the index and cutpoints.
fn ord_prob(probit: bool, k: usize, xb: f64, cuts: &[f64]) -> f64 {
    let ncat = cuts.len() + 1;
    let hi = if k + 1 < ncat { cuts[k] - xb } else { f64::INFINITY };
    let lo = if k > 0 { cuts[k - 1] - xb } else { f64::NEG_INFINITY };
    if lo > 0.0 { cdf(probit, -lo) - cdf(probit, -hi) } else { cdf(probit, hi) - cdf(probit, lo) }
}

/// Zero-inflated ordered probability of category k.
fn zio_prob(probit: bool, k: usize, xb: f64, zg: f64, cuts: &[f64]) -> f64 {
    let q = cdf(probit, zg);
    let p = ord_prob(probit, k, xb, cuts);
    if k == 0 { (1.0 - q) + q * p } else { q * p }
}

pub struct Zio {
    pub probit: bool,
}

fn zio_render(st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let probit = e.macros.get("cmd").is_some_and(|c| c == "zioprobit");
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let robust = e.macros.get("vcetype").is_some();
    let ll = e.get("ll").unwrap_or(SYSMISS);
    let (chi2, df) = (e.get("chi2").unwrap_or(SYSMISS), e.get("df_m").unwrap_or(0.0));
    let title = e.macros.get("title").cloned().unwrap_or_default();
    let left = vec![title, String::new(), format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9))];
    let right = vec![
        ("Number of obs".to_string(), util::commas(e.get("N").unwrap_or(0.0) as usize)),
        (format!("{} chi2({})", e.macros.get("chi2type").map_or("Wald", |s| s.as_str()), fmt_g(df, 9)), xm::f2(chi2)),
        ("Prob > chi2".to_string(), xm::f4(xm::chi2p(df, chi2))),
    ];
    let mut out = header_new(&left, &right);
    if let (Some(g), Some(c)) = (e.get("N_clust"), e.macros.get("clustvar")) {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g as usize))));
    }
    out.push(String::new());
    let level = util::level(cmd, st)?;
    let eform = if !probit && cmd.has_opt("or", 2) { Some("Odds ratio") } else { None };
    out.extend(table_aux_ci(&dep, &b.colnames, &b.data[..b.cols], v, level, eform));
    if eform.is_some() {
        out.push("Note: Estimates are transformed only in the first 2 equations to odds ratios.".into());
        out.push("Note: _cons estimates baseline odds.".into());
    }
    Ok(out.join("\n"))
}

impl CommandHandler for Zio {
    fn name(&self) -> &'static str {
        if self.probit { "zioprobit" } else { "ziologit" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            return zio_render(st, cmd);
        }
        let probit = self.probit;
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let (infl, sub) = xm::opt_with_sub(cmd, "inflate", 3).ok_or_else(|| rc_err(198, "option inflate() required"))?;
        let infl_nocons = sub.contains("nocons");
        let infl = if infl.trim() == "_cons" { String::new() } else { infl };
        let vce0 = est::parse_vce(cmd, Vce::Oim)?;
        let mut extra = Vec::new();
        if let Vce::Cluster(c) = &vce0 {
            extra.push(c.clone());
        }
        let eqs = xm::build_eqs(ds, st, cmd, &[(Some(&dep), &indep, true), (Some(&dep), &infl, infl_nocons)], "fpi", &extra)?;
        let (d1, d2) = (&eqs[0], &eqs[1]);
        let mut cats: Vec<f64> = d1.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let ncat = cats.len();
        if ncat < 2 {
            return Err(rc_err(148, "outcome does not vary"));
        }
        let yk: Vec<usize> = d1.y.iter().map(|y| cats.iter().position(|c| c == y).unwrap()).collect();
        let f = |t: usize, v: &[f64]| -> f64 { zio_prob(probit, yk[t], v[0], v[1], &v[2..]).max(1e-300).ln() };
        let g = xm::numeric(&f);
        let vce = xm::ml_vce_opt(cmd, d1)?;
        let cluster = est::cluster_values(ds, d1, &vce)?;
        let fw = d1.wkind == Some(WeightKind::F);
        // starting values: ordered model cutpoints from the marginal
        // frequencies of the participants, inflation probability near one
        let wsum: f64 = d1.w.iter().sum();
        let mut b0 = vec![0.0; d1.active().len()];
        let act2 = d2.active();
        let mut g0 = vec![0.0; act2.len()];
        if let Some(c) = act2.iter().position(|&j| d2.names[j] == "_cons") {
            g0[c] = inv_cdf(probit, 0.9);
        }
        b0.extend(g0);
        let mut cum = 0.0;
        for k in 0..ncat - 1 {
            cum += yk.iter().zip(&d1.w).filter(|(y, _)| **y == k).map(|(_, w)| w).sum::<f64>() / wsum;
            b0.push(inv_cdf(probit, cum));
        }
        let aux: Vec<String> = (1..ncat).map(|k| format!("cut{k}")).collect();
        let auxr: Vec<&str> = aux.iter().map(String::as_str).collect();
        let fit = xm::ml_fit(&[(&dep, d1), ("inflate", d2)], &auxr, &g, b0, &d1.w, fw, &vce, cluster.as_deref())?;
        let (chi2, df) = fit.wald_eq(0);
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 5) {
            out.extend(est::iteration_log(&fit.res.iterations, if vce != Vce::Oim { "Log pseudolikelihood" } else { "Log likelihood" }));
            out.push(String::new());
        }
        let nzero = yk.iter().zip(&d1.w).filter(|(y, _)| **y == 0).map(|(_, w)| if fw { *w } else { 1.0 }).sum::<f64>();
        let title = if probit { "Zero-inflated ordered probit regression" } else { "Zero-inflated ordered logit regression" };
        let catm = Matrix::row_vector(&cats);
        xm::post_fit(st, verb, d1, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("N_zero", nzero);
            e.set("k_cat", ncat as f64);
            e.set_mat("cat", catm.clone());
            e.set_str("title", title);
            e.set_str("predict", if probit { "zioprobit_p" } else { "ziologit_p" });
        });
        let text = zio_render(st, cmd)?;
        out.push(text);
        let text = out.join("\n");
        st.e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// slogit
// ---------------------------------------------------------------------------

/// A likelihood over b = T g + b0 (linear constraints).
pub struct Reduced<'a> {
    pub inner: &'a dyn Likelihood,
    pub t: Matrix,
    pub b0: Vec<f64>,
}

impl Reduced<'_> {
    pub fn full(&self, g: &[f64]) -> Vec<f64> {
        (0..self.t.rows).map(|i| self.b0[i] + (0..self.t.cols).map(|c| self.t[(i, c)] * g[c]).sum::<f64>()).collect()
    }
    /// Least-squares projection of a full vector on the free parameters.
    pub fn project(&self, b: &[f64]) -> Vec<f64> {
        let m = self.t.cols;
        let tt = self.t.transpose().mul(&self.t).unwrap_or_else(|_| Matrix::identity(m));
        let (inv, _) = tt.sym_inverse(1e-14);
        let r: Vec<f64> = (0..m).map(|c| (0..self.t.rows).map(|i| self.t[(i, c)] * (b[i] - self.b0[i])).sum()).collect();
        (0..m).map(|a| (0..m).map(|c| inv[(a, c)] * r[c]).sum()).collect()
    }
}

impl Likelihood for Reduced<'_> {
    fn k(&self) -> usize {
        self.t.cols
    }
    fn eval(&self, g: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let b = self.full(g);
        let (ll, gr, h, sc) = self.inner.eval(&b, need_scores);
        let m = self.t.cols;
        let k = self.t.rows;
        let gg: Vec<f64> = (0..m).map(|c| (0..k).map(|i| self.t[(i, c)] * gr[i]).sum()).collect();
        let ht = h.mul(&self.t).unwrap_or_else(|_| Matrix::zeros(k, m));
        let hh = self.t.transpose().mul(&ht).unwrap_or_else(|_| Matrix::zeros(m, m));
        let sc2 = sc.iter().map(|s| (0..m).map(|c| (0..k).map(|i| self.t[(i, c)] * s[i]).sum()).collect()).collect();
        (ll, gg, hh, sc2)
    }
}

pub struct Slogit;

struct SlogitModel<'a> {
    x: Vec<&'a [f64]>,
    yk: Vec<usize>,
    w: &'a [f64],
    fweight: bool,
    dim: usize,
    ncat: usize,
}

impl SlogitModel<'_> {
    /// Parameters: beta_1..beta_dim (kx each), phi[k][j] (dim x ncat), theta[j].
    fn obs(&self, b: &[f64], t: usize) -> (f64, Vec<f64>) {
        let kx = self.x.len();
        let (d, m) = (self.dim, self.ncat);
        let u: Vec<f64> = (0..d).map(|k| (0..kx).map(|c| self.x[c][t] * b[k * kx + c]).sum()).collect();
        let po = d * kx;
        let to = po + d * m;
        let eta: Vec<f64> = (0..m).map(|j| b[to + j] - (0..d).map(|k| b[po + k * m + j] * u[k]).sum::<f64>()).collect();
        let mx = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = eta.iter().map(|e| (e - mx).exp()).sum();
        let p: Vec<f64> = eta.iter().map(|e| (e - mx).exp() / s).collect();
        let y = self.yk[t];
        let ll = eta[y] - mx - s.ln();
        let r: Vec<f64> = (0..m).map(|j| (if j == y { 1.0 } else { 0.0 }) - p[j]).collect();
        let mut g = vec![0.0; b.len()];
        for k in 0..d {
            let sk: f64 = (0..m).map(|j| r[j] * b[po + k * m + j]).sum();
            for c in 0..kx {
                g[k * kx + c] = -sk * self.x[c][t];
            }
            for j in 0..m {
                g[po + k * m + j] = -r[j] * u[k];
            }
        }
        for j in 0..m {
            g[to + j] = r[j];
        }
        (ll, g)
    }
}

impl Likelihood for SlogitModel<'_> {
    fn k(&self) -> usize {
        self.dim * self.x.len() + self.dim * self.ncat + self.ncat
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.k();
        let n = self.w.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut scores = Vec::new();
        for t in 0..n {
            let (l, gi) = self.obs(b, t);
            ll += self.w[t] * l;
            for p in 0..k {
                g[p] += self.w[t] * gi[p];
            }
            if need_scores {
                let fw = if self.fweight { self.w[t].sqrt() } else { self.w[t] };
                scores.push(gi.iter().map(|s| s * fw).collect());
            }
        }
        // information by differencing the analytic gradient
        let grad = |bb: &[f64]| -> Vec<f64> {
            let mut gg = vec![0.0; k];
            for t in 0..n {
                let (_, gi) = self.obs(bb, t);
                for p in 0..k {
                    gg[p] += self.w[t] * gi[p];
                }
            }
            gg
        };
        let hv = crate::choice_mixed::num_hess(&grad, b);
        let mut h = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                h[(i, j)] = -(hv[i][j] + hv[j][i]) / 2.0;
            }
        }
        (ll, g, h, scores)
    }
}

fn slogit_render(st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let robust = e.macros.get("vcetype").is_some();
    let ll = e.get("ll").unwrap_or(SYSMISS);
    let (chi2, df) = (e.get("chi2").unwrap_or(SYSMISS), e.get("df_m").unwrap_or(0.0));
    let left = vec!["Stereotype logistic regression".to_string(), String::new(), format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9))];
    let right = vec![
        ("Number of obs".to_string(), util::commas(e.get("N").unwrap_or(0.0) as usize)),
        (format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)),
        ("Prob > chi2".to_string(), xm::f4(xm::chi2p(df, chi2))),
    ];
    let mut out = header_new(&left, &right);
    if let (Some(g), Some(c)) = (e.get("N_clust"), e.macros.get("clustvar")) {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g as usize))));
    }
    if let Some(c) = e.macros.get("_constraints") {
        out.extend(c.lines().map(String::from));
    }
    out.push(String::new());
    let level = util::level(cmd, st)?;
    let names = &b.colnames;
    let nb = names.iter().filter(|n| !n.starts_with("phi") && !n.starts_with("theta")).count();
    let (omitted, is_base) = flags_from_names(names);
    let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &names[..nb], b: &b.data[..nb], v, omitted: &omitted[..nb], is_base: &is_base[..nb], df: None, level, eform: None, coef_title: "Coefficient" });
    let full = est::table_matrix(&est::TableSpec { depvar: &dep, names, b: &b.data[..b.cols], v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" });
    est::stash_table(full, level);
    let bar = tbl.pop().unwrap_or_default();
    let lw = bar.len().saturating_sub(66);
    let crit = zcrit(level);
    tbl.push(sep(lw));
    let dim = e.get("k_dim").unwrap_or(1.0) as usize;
    let ncat = e.get("k_out").unwrap_or(0.0) as usize;
    let ibase = e.get("i_base").unwrap_or(ncat as f64) as usize;
    let mut fixed: Vec<(String, f64)> = Vec::new();
    if let Some(fx) = e.macros.get("_fixed") {
        for l in fx.split(';') {
            if let Some((n, val)) = l.split_once('=') {
                fixed.push((n.to_string(), val.parse().unwrap_or(0.0)));
            }
        }
    }
    let row = |label: &str, nm: &str, base: bool| -> String {
        if base {
            return format!("{:>lw$} | {:>10}  (base outcome)", label, 0);
        }
        let j = b.col_index(nm);
        match j {
            Some(j) => {
                let se = v[(j, j)].max(0.0).sqrt();
                if se < 1e-12 {
                    let val = b.get(0, j);
                    let tag = if fixed.iter().any(|(n, _)| n == nm) && val == 0.0 { "(omitted)" } else { "(constrained)" };
                    format!("{:>lw$} | {:>10}  {tag}", label, fmt_g(val, 8))
                } else {
                    row_z(lw, label, b.get(0, j), se, crit)
                }
            }
            None => format!("{:>lw$} | {:>10}  (omitted)", label, 0),
        }
    };
    for k in 1..=dim {
        for j in 1..=ncat {
            tbl.push(row(&format!("/phi{k}_{j}"), &format!("phi{k}_{j}:_cons"), j == ibase));
        }
    }
    for j in 1..=ncat {
        tbl.push(row(&format!("/theta{j}"), &format!("theta{j}:_cons"), j == ibase));
    }
    tbl.push(bar);
    out.extend(tbl);
    if let Some(bl) = e.macros.get("_baselabel") {
        out.push(format!("({dep}={bl} is the base outcome)"));
    }
    Ok(out.join("\n"))
}

impl CommandHandler for Slogit {
    fn name(&self) -> &'static str {
        "slogit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("slogit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return slogit_render(st, cmd);
        }
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let vce0 = est::parse_vce(cmd, Vce::Oim)?;
        let mut extra = Vec::new();
        if let Vce::Cluster(c) = &vce0 {
            extra.push(c.clone());
        }
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: true, weights: "fpi", extra })?;
        let mut cats: Vec<f64> = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let m = cats.len();
        if m < 3 {
            return Err(rc_err(148, format!("{dep} has {m} outcome{}; slogit requires at least 3", if m == 1 { "" } else { "s" })));
        }
        let dim: usize = cmd.opt("dimension", 3).map(|s| s.trim().parse().map_err(|_| rc_err(198, "dimension() invalid"))).transpose()?.unwrap_or(1);
        if dim < 1 || dim > m - 1 {
            return Err(rc_err(198, format!("dimension() must be between 1 and {}", m - 1)));
        }
        let ibase = match cmd.opt("baseoutcome", 4) {
            Some(s) => {
                let v: f64 = s.trim().parse().map_err(|_| rc_err(198, "baseoutcome() invalid"))?;
                cats.iter().position(|c| *c == v).ok_or_else(|| rc_err(198, format!("outcome {s} not found")))? + 1
            }
            None => m,
        };
        let yk: Vec<usize> = d.y.iter().map(|y| cats.iter().position(|c| c == y).unwrap()).collect();
        let act = d.active();
        let kx = act.len();
        let xcols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let fw = d.wkind == Some(WeightKind::F);
        let model = SlogitModel { x: xcols, yk: yk.clone(), w: &d.w, fweight: fw, dim, ncat: m };
        // full parameter names
        let xn: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        let mut names = Vec::new();
        for k in 1..=dim {
            for n in &xn {
                names.push(if dim == 1 { n.clone() } else { format!("dim{k}:{n}") });
            }
        }
        for k in 1..=dim {
            for j in 1..=m {
                names.push(format!("phi{k}_{j}:_cons"));
            }
        }
        for j in 1..=m {
            names.push(format!("theta{j}:_cons"));
        }
        let kf = names.len();
        let po = dim * kx;
        let to = po + dim * m;
        // identification: base outcome, corner constraints, user constraints
        let mut rrows: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut shown = Vec::new();
        let mut fixed = Vec::new();
        for k in 0..dim {
            let mut r = vec![0.0; kf];
            r[po + k * m + ibase - 1] = 1.0;
            rrows.push((r, 0.0));
        }
        let mut r = vec![0.0; kf];
        r[to + ibase - 1] = 1.0;
        rrows.push((r, 0.0));
        let corner: Vec<usize> = (1..=m).filter(|&j| j != ibase).take(dim).collect();
        for k in 0..dim {
            for (a, &j) in corner.iter().enumerate() {
                let mut r = vec![0.0; kf];
                r[po + k * m + j - 1] = 1.0;
                let val = if a == k { 1.0 } else { 0.0 };
                rrows.push((r.clone(), val));
                shown.push(xm::constraint_text(shown.len() + 1, &r, val, &names));
                fixed.push(format!("phi{}_{}:_cons={}", k + 1, j, val));
            }
        }
        for (_, spec) in xm::constraint_specs(st, cmd)? {
            let (r, c) = xm::parse_constraint(&spec, &names)?;
            shown.push(xm::constraint_text(shown.len() + 1, &r, c, &names));
            rrows.push((r, c));
        }
        let mut rm = Matrix::zeros(rrows.len(), kf);
        for (i, (r, _)) in rrows.iter().enumerate() {
            for j in 0..kf {
                rm[(i, j)] = r[j];
            }
        }
        let rv: Vec<f64> = rrows.iter().map(|x| x.1).collect();
        let (tm, b0c) = xm::constraint_basis(&rm, &rv)?;
        let red = Reduced { inner: &model, t: tm.clone(), b0: b0c };
        // starting values: zero slopes, scores spaced linearly, intercepts
        // from the marginal log odds against the base
        let mut bs = vec![0.0; kf];
        let wsum: f64 = d.w.iter().sum();
        let freq: Vec<f64> = (0..m).map(|j| (yk.iter().zip(&d.w).filter(|(y, _)| **y == j).map(|(_, w)| w).sum::<f64>() / wsum).max(1e-4)).collect();
        for j in 0..m {
            bs[to + j] = (freq[j] / freq[ibase - 1]).ln();
        }
        for k in 0..dim {
            for j in 0..m {
                let ck = corner[k];
                // 1 at the corner category, 0 at the base, interpolated by position
                let pos = |c: usize| -> f64 { (c as f64 - ibase as f64).abs() };
                bs[po + k * m + j] = if j + 1 == ibase { 0.0 } else { pos(j + 1) / pos(ck) };
            }
        }
        // stage 1: slopes and intercepts with the scores held at their
        // starting values (a concave problem), then all parameters
        {
            let mut rm1 = Matrix::zeros(rrows.len() + dim * m, kf);
            let mut rv1 = rv.clone();
            for i in 0..rrows.len() {
                for j in 0..kf {
                    rm1[(i, j)] = rm[(i, j)];
                }
            }
            let mut q = rrows.len();
            for k in 0..dim {
                for j in 0..m {
                    let p = po + k * m + j;
                    // skip scores already fixed by the identification rows
                    if rrows.iter().any(|(r, _)| r[p] != 0.0 && r.iter().filter(|x| **x != 0.0).count() == 1) {
                        continue;
                    }
                    rm1[(q, p)] = 1.0;
                    rv1.push(bs[p]);
                    q += 1;
                }
            }
            let mut rm1b = Matrix::zeros(q, kf);
            for i in 0..q {
                for j in 0..kf {
                    rm1b[(i, j)] = rm1[(i, j)];
                }
            }
            if let Ok((t1, b01)) = xm::constraint_basis(&rm1b, &rv1) {
                let red1 = Reduced { inner: &model, t: t1, b0: b01 };
                let g1 = red1.project(&bs);
                if let Ok(r1) = crate::zip::maximize_robust(&red1, g1, 100) {
                    bs = red1.full(&r1.b);
                }
            }
        }
        let g0 = red.project(&bs);
        let res = crate::choice_mixed::maximize_tight(&red, g0, 300)?;
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let (vg, nclust) = ml_vce_scaled(&res, &vce, cluster.as_deref(), d.n_eff());
        let bf = red.full(&res.b);
        let vf = tm.mul(&vg).and_then(|x| x.mul(&tm.transpose()))?;
        // e(b) excludes the base-outcome parameters
        let keep: Vec<usize> = (0..kf).filter(|&j| !(j >= po && j < to && (j - po) % m == ibase - 1) && j != to + ibase - 1).collect();
        let bnames: Vec<String> = keep.iter().map(|&j| names[j].clone()).collect();
        let bk: Vec<f64> = keep.iter().map(|&j| bf[j]).collect();
        let mut vk = Matrix::zeros(keep.len(), keep.len());
        for (a, &i) in keep.iter().enumerate() {
            for (c, &j) in keep.iter().enumerate() {
                vk[(a, c)] = vf[(i, j)];
            }
        }
        let bidx: Vec<usize> = (0..po).collect();
        let (chi2, df) = crate::heckman::wald(&bk, &vk, &bidx);
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 5) {
            out.extend(est::iteration_log(&res.iterations, if vce != Vce::Oim { "Log pseudolikelihood" } else { "Log likelihood" }));
            out.push(String::new());
        }
        let blabel = crate::teffects::level_label(ds, &dep, cats[ibase - 1]);
        let mut dpost = d.clone();
        dpost.names = bnames.clone();
        dpost.omitted = vec![false; bnames.len()];
        dpost.is_base = vec![false; bnames.len()];
        let ll = res.ll;
        let iters = res.iterations.len().saturating_sub(1) as f64;
        let conv = res.converged;
        let catm = Matrix::row_vector(&cats);
        est::post(st, "slogit", &dpost, &bk, &vk, &|e: &mut Results| {
            e.set("ll", ll);
            e.set("k_dim", dim as f64);
            e.set("k_out", m as f64);
            e.set("i_base", ibase as f64);
            e.set("k_indvars", kx as f64);
            e.set("ic", iters);
            e.set("converged", if conv { 1.0 } else { 0.0 });
            xm::set_test(e, "Wald", chi2, df);
            let (vv, vt) = est::vce_label(&vce);
            e.set_str("vce", vv);
            if !vt.is_empty() {
                e.set_str("vcetype", vt);
            }
            if let (Some(g), Vce::Cluster(c)) = (nclust, &vce) {
                e.set("N_clust", g as f64);
                e.set_str("clustvar", c.clone());
            }
            e.set_mat("outcomes", catm.clone());
            e.set_str("indvars", xn.join(" "));
            e.set_str("title", "Stereotype logistic regression");
            e.set_str("predict", "slogit_p");
            e.set_str("_constraints", shown.join("\n"));
            e.set_str("_fixed", fixed.join(";"));
            e.set_str("_baselabel", blabel.clone());
        });
        out.push(slogit_render(st, cmd)?);
        let text = out.join("\n");
        st.e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

fn outcome_index(st: &Store, cats: &[f64], spec: &str) -> Result<usize, String> {
    let s = spec.trim();
    if let Some(r) = s.strip_prefix('#') {
        let k: usize = r.parse().map_err(|_| rc_err(198, "outcome() invalid"))?;
        return if k >= 1 && k <= cats.len() { Ok(k - 1) } else { Err(rc_err(198, "outcome() invalid")) };
    }
    let _ = st;
    let v: f64 = s.parse().map_err(|_| rc_err(198, "outcome() invalid"))?;
    cats.iter().position(|c| *c == v).ok_or_else(|| rc_err(198, format!("outcome {s} not found")))
}

/// Probabilities of all outcomes for one observation.
fn probs(st: &Store, env: &Env, ecmd: &str, row: usize) -> Result<Option<Vec<f64>>, String> {
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let aux = |n: &str| b.col_index(n).map_or(SYSMISS, |j| b.get(0, j));
    if ecmd == "slogit" {
        let dim = st.e.get("k_dim").unwrap_or(1.0) as usize;
        let m = st.e.get("k_out").unwrap_or(0.0) as usize;
        let ibase = st.e.get("i_base").unwrap_or(m as f64) as usize;
        let mut u = Vec::new();
        for k in 1..=dim {
            let eq = if dim == 1 { None } else { Some(format!("dim{k}")) };
            let x = if dim == 1 {
                // plain names: the slopes are the columns without an equation
                let mut s = 0.0;
                for (j, n) in b.colnames.iter().enumerate() {
                    if n.contains(':') {
                        continue;
                    }
                    let v = crate::postest::coef_value(env, n, row)?;
                    if is_missing(v) {
                        return Ok(None);
                    }
                    s += v * b.get(0, j);
                }
                s
            } else {
                crate::postest::xb_row(env, b, eq.as_deref(), row)?
            };
            if is_missing(x) {
                return Ok(None);
            }
            u.push(x);
        }
        let eta: Vec<f64> = (1..=m)
            .map(|j| {
                if j == ibase {
                    return 0.0;
                }
                let th = aux(&format!("theta{j}:_cons"));
                th - (1..=dim).map(|k| aux(&format!("phi{k}_{j}:_cons")) * u[k - 1]).sum::<f64>()
            })
            .collect();
        let mx = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = eta.iter().map(|e| (e - mx).exp()).sum();
        return Ok(Some(eta.iter().map(|e| (e - mx).exp() / s).collect()));
    }
    let probit = ecmd == "zioprobit";
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let xb = crate::postest::xb_row(env, b, Some(&dep), row)?;
    let zg = crate::postest::xb_row(env, b, Some("inflate"), row)?;
    if is_missing(xb) || is_missing(zg) {
        return Ok(None);
    }
    let ncat = st.e.get("k_cat").unwrap_or(0.0) as usize;
    let cuts: Vec<f64> = (1..ncat).map(|k| aux(&format!("/:cut{k}"))).collect();
    Ok(Some((0..ncat).map(|k| zio_prob(probit, k, xb, zg, &cuts)).collect()))
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    if !matches!(ecmd, "ziologit" | "zioprobit" | "slogit") {
        return None;
    }
    Some(predict_inner(ds, st, cmd, ecmd))
}

fn predict_inner(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Result<String, String> {
    let words: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
    let words: Vec<String> = if words.first().is_some_and(|w| matches!(w.as_str(), "double" | "float")) { words[1..].to_vec() } else { words };
    if words.is_empty() {
        return Err(rc_err(100, "new variable name required"));
    }
    let cats: Vec<f64> = st.e.matrices.get(if ecmd == "slogit" { "outcomes" } else { "cat" }).map(|m| m.data.clone()).unwrap_or_default();
    let m = cats.len();
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let stat = if cmd.has_opt("xb", 2) {
        "xb"
    } else if cmd.has_opt("pmargin", 2) {
        "pmargin"
    } else if cmd.has_opt("pjoint1", 6) {
        "pjoint1"
    } else if cmd.has_opt("pr", 2) {
        "pr"
    } else {
        ""
    };
    let env = Env::with_store(ds, st);
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let mut outvars: Vec<(String, Vec<f64>)> = Vec::new();
    let mut msg = String::new();
    if stat == "xb" {
        let eq = cmd.opt("equation", 2).map(|s| s.trim().trim_start_matches('#').to_string());
        let eqname = match (&eq, ecmd) {
            (Some(e), _) if e.parse::<usize>().is_ok() => {
                let k: usize = e.parse().unwrap();
                xm::eq_names(&b).get(k - 1).cloned().flatten()
            }
            (Some(e), _) => Some(e.clone()),
            (None, "slogit") => None,
            (None, _) => st.e.macros.get("depvar").cloned(),
        };
        let vals: Vec<f64> = (0..n).map(|i| if mask[i] { crate::postest::xb_row(&env, &b, eqname.as_deref(), i) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?;
        outvars.push((words[0].clone(), vals));
    } else {
        let all: Vec<Option<Vec<f64>>> = (0..n).map(|i| if mask[i] { probs(st, &env, ecmd, i) } else { Ok(None) }).collect::<Result<_, _>>()?;
        let pick = |k: usize| -> Vec<f64> { all.iter().map(|p| p.as_ref().map_or(SYSMISS, |v| v[k])).collect() };
        if stat == "pmargin" || stat == "pjoint1" {
            // probability of participation / of a participant at the lowest category
            let probit = ecmd == "zioprobit";
            let vals: Vec<f64> = (0..n)
                .map(|i| -> Result<f64, String> {
                    if !mask[i] {
                        return Ok(SYSMISS);
                    }
                    let zg = crate::postest::xb_row(&env, &b, Some("inflate"), i)?;
                    if is_missing(zg) {
                        return Ok(SYSMISS);
                    }
                    Ok(cdf(probit, zg))
                })
                .collect::<Result<_, _>>()?;
            outvars.push((words[0].clone(), vals));
        } else if let Some(o) = cmd.opt("outcome", 2) {
            let k = outcome_index(st, &cats, o)?;
            outvars.push((words[0].clone(), pick(k)));
        } else if words.len() == 1 && words[0].ends_with('*') {
            let stub = words[0].trim_end_matches('*');
            for k in 0..m {
                outvars.push((format!("{stub}{}", k + 1), pick(k)));
            }
        } else if words.len() == m {
            for k in 0..m {
                outvars.push((words[k].clone(), pick(k)));
            }
        } else if words.len() == 1 {
            outvars.push((words[0].clone(), pick(0)));
            msg = format!("(option pr assumed; predicted probability of outcome {})", fmt_g(cats[0], 8));
        } else {
            return Err(rc_err(102, "too few or too many variables specified"));
        }
    }
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    } else if stat.is_empty() && outvars.len() > 1 {
        lines.push("(option pr assumed; predicted probabilities)".into());
    }
    for (name, vals) in outvars {
        crate::varlist::check_name(&name)?;
        if ds.var_index.contains_key(&name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
        ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
        if nmiss > 0 && lines.len() < 2 {
            let _ = nmiss;
        }
    }
    Ok(lines.join("\n"))
}
