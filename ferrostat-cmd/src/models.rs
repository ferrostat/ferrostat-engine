//! Estimation commands.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::{digamma, ln_gamma};

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};

fn split_dep(main: &str) -> Result<(String, String), String> {
    let main = main.trim();
    let (dep, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
    if dep.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    Ok((dep.to_string(), rest.trim().to_string()))
}

fn resolve_dep(ds: &DataSet, dep: &str) -> Result<String, String> {
    if dep.contains('.') {
        let (op, v) = dep.rsplit_once('.').unwrap();
        Ok(format!("{op}.{}", crate::varlist::resolve_name(ds, v)?))
    } else {
        crate::varlist::resolve_name(ds, dep)
    }
}

fn f2(v: f64) -> String {
    format!("{:.2}", v)
}
fn f4(v: f64) -> String {
    format!("{:.4}", v)
}
fn g10(v: f64) -> String {
    fmt_g(v, 9)
}

/// Re-display the last estimates (estimation command typed alone).
pub fn replay(st: &Store, cmd: &str) -> Result<String, String> {
    let e = &st.e;
    if e.macros.get("cmd").map(String::as_str) != Some(cmd) {
        return Err(rc_err(301, "last estimates not found"));
    }
    Ok(e.macros.get("_output").cloned().unwrap_or_default())
}

// ---------------------------------------------------------------------------
// regress
// ---------------------------------------------------------------------------

pub struct Regress;

impl CommandHandler for Regress {
    fn name(&self) -> &'static str {
        "regress"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "regress");
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let mut extra = Vec::new();
        if let Vce::Cluster(c) = &vce {
            extra.push(c.clone());
        }
        let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "faip", extra })?;
        let vce = if d.wkind == Some(WeightKind::P) && vce == Vce::Ols { Vce::Robust } else { vce };
        let cl = est::cluster_values(ds, &d, &vce)?;
        d.cluster = cl.clone();
        let fit = est::ols(&d, &vce, cl.as_deref())?;
        let level = util::level(cmd, st)?;
        let n = d.n_eff();
        let r2 = 1.0 - fit.rss / fit.tss;
        let adj = 1.0 - (1.0 - r2) * (n - if d.has_cons { 1.0 } else { 0.0 }) / fit.df_r;
        let rmse = (fit.rss / fit.df_r).sqrt();
        let mss = fit.tss - fit.rss;
        let robust = vce != Vce::Ols;
        let (f, df_m_f, df_r_f) = if robust {
            let (w, q) = est::wald_all(&d, &fit.b, &fit.v);
            let df_r = fit.n_clust.map_or(fit.df_r, |g| g as f64 - 1.0);
            (w / q, q, df_r)
        } else {
            ((mss / fit.df_m) / (fit.rss / fit.df_r), fit.df_m, fit.df_r)
        };
        let pf = dist::ftail(df_m_f, df_r_f, f);
        let tdf = fit.n_clust.map_or(fit.df_r, |g| g as f64 - 1.0);
        let mut out: Vec<String> = d.notes.clone();
        if robust {
            out.push(format!("{:<52}{:<15} = {:>10}", "Linear regression", "Number of obs", g10(n)));
            out.push(format!("{:<52}{:<15} = {:>10}", "", format!("F({}, {})", df_m_f, df_r_f), f2(f)));
            out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > F", f4(pf)));
            out.push(format!("{:<52}{:<15} = {:>10}", "", "R-squared", f4(r2)));
            out.push(format!("{:<52}{:<15} = {:>10}", "", "Root MSE", rmse4(rmse)));
            if let Some(g) = fit.n_clust {
                out.push(String::new());
                out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", g, match &vce { Vce::Cluster(c) => c.as_str(), _ => "" })));
            }
        } else {
            out.push(format!("      Source |       SS           df       MS      Number of obs   = {:>9}", g10(n)));
            out.push(format!("-------------+----------------------------------   F({}, {}){} = {:>9}", fit.df_m, fit.df_r, " ".repeat(8usize.saturating_sub(format!("{}{}", fit.df_m, fit.df_r).len())), f2(f)));
            out.push(format!("       Model | {:>11} {:>9} {:>11}   Prob > F        = {:>9}", g10(mss), fit.df_m, g10(mss / fit.df_m), f4(pf)));
            out.push(format!("    Residual | {:>11} {:>9} {:>11}   R-squared       = {:>9}", g10(fit.rss), fit.df_r, g10(fit.rss / fit.df_r), f4(r2)));
            out.push(format!("-------------+----------------------------------   Adj R-squared   = {:>9}", f4(adj)));
            out.push(format!("       Total | {:>11} {:>9} {:>11}   Root MSE        = {:>9}", g10(fit.tss), n - 1.0, g10(fit.tss / (n - 1.0)), g10(rmse)));
        }
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &d.names,
            b: &fit.b,
            v: &fit.v,
            omitted: &d.omitted,
            is_base: &d.is_base,
            df: Some(tdf),
            level,
            eform: None,
            coef_title: "Coefficient",
        }));
        let text = out.join("\n");
        let (vcetype, vcelabel) = est::vce_label(&vce);
        est::post(st, "regress", &d, &fit.b, &fit.v, &|e: &mut Results| {
            e.set("df_m", fit.df_m);
            e.set("df_r", tdf);
            e.set("F", f);
            e.set("r2", r2);
            e.set("r2_a", adj);
            e.set("rmse", rmse);
            e.set("mss", mss);
            e.set("rss", fit.rss);
            let sw: f64 = d.w.iter().sum();
            let ll = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (fit.rss / n).ln() + 1.0);
            let ll0 = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (fit.tss / n).ln() + 1.0);
            let _ = sw;
            e.set("ll", ll);
            e.set("ll_0", ll0);
            if let Some(g) = fit.n_clust {
                e.set("N_clust", g as f64);
            }
            e.set_str("model", "ols");
            e.set_str("title", "Linear regression");
            e.set_str("vce", if robust { vcetype } else { "ols" });
            if !vcelabel.is_empty() {
                e.set_str("vcetype", vcelabel);
            }
            if let Vce::Cluster(c) = &vce {
                e.set_str("clustvar", c.clone());
            }
            e.set_str("predict", "regres_p");
            e.set_str("estat_cmd", "regress_estat");
            e.set_str("_output", text.clone());
            if d.wkind.is_some() {
                e.set_str("wtype", match d.wkind { Some(WeightKind::F) => "fweight", Some(WeightKind::A) => "aweight", Some(WeightKind::P) => "pweight", _ => "iweight" });
            }
        });
        // keep (X'X)^-1 for leverage predictions
        st.e.set_mat("XtXinv", fit.xtx_inv.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// single-index ML models
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Link {
    Logit,
    Probit,
    Cloglog,
    Poisson,
}

/// (ll, d ll / d xb, d2 ll / d xb2)
fn index_ll(link: Link, y: f64, xb: f64) -> (f64, f64, f64) {
    match link {
        Link::Logit => {
            let p = dist::invlogit(xb);
            let lnp = -(1.0 + (-xb).exp()).ln();
            let lnq = -(1.0 + xb.exp()).ln();
            let lnp = if lnp.is_finite() { lnp } else { xb };
            let lnq = if lnq.is_finite() { lnq } else { -xb };
            let ll = if y != 0.0 { lnp } else { lnq };
            (ll, (if y != 0.0 { 1.0 } else { 0.0 }) - p, -p * (1.0 - p))
        }
        Link::Probit => {
            let q = if y != 0.0 { 1.0 } else { -1.0 };
            let z = q * xb;
            let cdf = dist::normal(z);
            let lncdf = if cdf > 1e-300 { cdf.ln() } else { -0.5 * z * z - (-z).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() };
            // inverse Mills ratio, stable for large negative z
            let lam = if cdf > 1e-12 { dist::normalden(z) / cdf } else { -z + 1.0 / (-z) };
            let d1 = q * lam;
            let d2 = -lam * (lam + z);
            (lncdf, d1, d2)
        }
        Link::Cloglog => {
            let mu = xb.exp();
            if y != 0.0 {
                let p = -(-mu).exp_m1();
                let a = mu * (-mu).exp() / p;
                (p.ln(), a, a * (1.0 - mu) - a * a)
            } else {
                (-mu, -mu, -mu)
            }
        }
        Link::Poisson => {
            let mu = xb.exp();
            (y * xb - mu - ln_gamma(y + 1.0), y - mu, -mu)
        }
    }
}

struct IndexModel<'a> {
    d: &'a Design,
    act: Vec<usize>,
    link: Link,
    offset: Vec<f64>,
}

impl<'a> Likelihood for IndexModel<'a> {
    fn k(&self) -> usize {
        self.act.len()
    }

    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.act.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let mut scores = Vec::new();
        for t in 0..self.d.n() {
            let w = self.d.w[t];
            let xb: f64 = self.act.iter().zip(b).map(|(&j, bj)| self.d.cols[j][t] * bj).sum::<f64>() + self.offset[t];
            let (l, d1, d2) = index_ll(self.link, self.d.y[t], xb);
            ll += w * l;
            for a in 0..k {
                let xa = self.d.cols[self.act[a]][t];
                g[a] += w * d1 * xa;
                for c in 0..=a {
                    h[(a, c)] -= w * d2 * xa * self.d.cols[self.act[c]][t];
                }
            }
            if need_scores {
                let fw = if self.d.wkind == Some(WeightKind::F) { w.sqrt() } else { w };
                scores.push(self.act.iter().map(|&j| fw * d1 * self.d.cols[j][t]).collect());
            }
        }
        for a in 0..k {
            for c in 0..a {
                h[(c, a)] = h[(a, c)];
            }
        }
        (ll, g, h, scores)
    }
}

/// Variance from ML results according to the VCE type.
fn ml_vce(res: &est::MlResult, vce: &Vce, cluster: Option<&[f64]>, n: f64) -> (Matrix, Option<usize>) {
    let (hinv, _) = res.info.sym_inverse(1e-14);
    match vce {
        Vce::Ols | Vce::Oim => (hinv, None),
        Vce::Robust => {
            let (v, _) = est::sandwich(&hinv, &res.scores, None, n / (n - 1.0));
            (v, None)
        }
        Vce::Cluster(_) => {
            let (v, g) = est::sandwich(&hinv, &res.scores, cluster, 1.0);
            let gf = g as f64;
            (v.scale(gf / (gf - 1.0)), Some(g))
        }
    }
}

/// Fit a constant-only model of the same family (for LR tests / pseudo R2).
fn null_ll(d: &Design, link: Link, offset: &[f64]) -> f64 {
    if !d.has_cons {
        return SYSMISS;
    }
    let mut d0 = d.clone();
    let c = d0.k() - 1;
    for j in 0..d0.k() {
        d0.omitted[j] = j != c;
    }
    let m = IndexModel { d: &d0, act: vec![c], link, offset: offset.to_vec() };
    est::maximize(&m, vec![0.0], 100).map(|r| r.ll).unwrap_or(SYSMISS)
}

fn offset_values(ds: &DataSet, d: &Design, cmd: &Command) -> Result<Vec<f64>, String> {
    let mut off = vec![0.0; d.n()];
    if let Some(o) = cmd.opt("offset", 3) {
        let c = util::col(ds, &crate::varlist::resolve_name(ds, o.trim())?)?;
        for (t, &i) in d.rows.iter().enumerate() {
            off[t] += c[i];
        }
    }
    if let Some(o) = cmd.opt("exposure", 1) {
        let c = util::col(ds, &crate::varlist::resolve_name(ds, o.trim())?)?;
        for (t, &i) in d.rows.iter().enumerate() {
            if c[i] <= 0.0 {
                return Err(rc_err(459, "exposure must be positive"));
            }
            off[t] += c[i].ln();
        }
    }
    Ok(off)
}

pub struct BinaryModel {
    pub verb: &'static str,
    pub link: Link,
}

impl CommandHandler for BinaryModel {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, self.verb);
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let mut extra = Vec::new();
        if let Vce::Cluster(c) = &vce {
            extra.push(c.clone());
        }
        let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fip", extra })?;
        let vce = if d.wkind == Some(WeightKind::P) && vce == Vce::Oim { Vce::Robust } else { vce };
        let is_count = self.link == Link::Poisson;
        if !is_count {
            for y in d.y.iter_mut() {
                *y = if *y != 0.0 { 1.0 } else { 0.0 };
            }
            let ones = d.y.iter().filter(|&&y| y == 1.0).count();
            if ones == 0 || ones == d.n() {
                return Err(rc_err(2000, "outcome does not vary"));
            }
        } else if d.y.iter().any(|&y| y < 0.0) {
            return Err(rc_err(459, format!("{dep} must be greater than or equal to zero")));
        }
        // perfect prediction by a dummy regressor: drop it, as Stata does
        let mut notes = d.notes.clone();
        if !is_count {
            for j in 0..d.k() {
                if d.omitted[j] || d.names[j] == "_cons" {
                    continue;
                }
                let col = &d.cols[j];
                let is_dummy = col.iter().all(|&x| x == 0.0 || x == 1.0);
                if !is_dummy {
                    continue;
                }
                for val in [0.0, 1.0] {
                    let ys: Vec<f64> = (0..d.n()).filter(|&t| col[t] != 0.0).map(|t| d.y[t]).collect();
                    if !ys.is_empty() && ys.iter().all(|&y| y == val) {
                        notes.push(format!("note: {} != 0 predicts {} perfectly;\n      {} omitted and {} obs not used.", d.names[j], if val == 1.0 { "success" } else { "failure" }, d.names[j], ys.len()));
                        let keep: Vec<usize> = (0..d.n()).filter(|&t| col[t] == 0.0).collect();
                        let rows: Vec<usize> = keep.iter().map(|&t| d.rows[t]).collect();
                        for c in d.cols.iter_mut() {
                            *c = keep.iter().map(|&t| c[t]).collect();
                        }
                        d.y = keep.iter().map(|&t| d.y[t]).collect();
                        d.w = keep.iter().map(|&t| d.w[t]).collect();
                        let mut mask = vec![false; ds.n_obs];
                        for &r in &rows {
                            mask[r] = true;
                        }
                        d.rows = rows;
                        d.mask = mask;
                        d.omitted[j] = true;
                        d.names[j] = format!("o.{}", d.names[j]);
                        break;
                    }
                }
            }
        }
        let cl = est::cluster_values(ds, &d, &vce)?;
        let offset = offset_values(ds, &d, cmd)?;
        let act = d.active();
        let model = IndexModel { d: &d, act: act.clone(), link: self.link, offset: offset.clone() };
        let mut b0 = vec![0.0; act.len()];
        if is_count && d.has_cons {
            let my = d.y.iter().zip(&d.w).map(|(y, w)| y * w).sum::<f64>() / d.w.iter().sum::<f64>();
            if let Some(pos) = act.iter().position(|&j| d.names[j] == "_cons") {
                b0[pos] = my.max(1e-8).ln();
            }
        }
        let res = est::maximize(&model, b0, 300)?;
        let n = d.n_eff();
        let (vact, nclust) = ml_vce(&res, &vce, cl.as_deref(), n);
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vact, &act, k);
        let ll0 = null_ll(&d, self.link, &offset);
        let robust = vce != Vce::Oim;
        let (chi2, df_m) = if robust {
            est::wald_all(&d, &b, &v)
        } else {
            (2.0 * (res.ll - ll0), (act.len() - if d.has_cons { 1 } else { 0 }) as f64)
        };
        let p = dist::chi2tail(df_m, chi2);
        let r2_p = 1.0 - res.ll / ll0;
        let level = util::level(cmd, st)?;
        let (title, eform) = match (self.verb, self.link) {
            ("logistic", _) => ("Logistic regression", Some("Odds ratio")),
            (_, Link::Logit) => ("Logistic regression", if cmd.has_opt("or", 2) { Some("Odds ratio") } else { None }),
            (_, Link::Probit) => ("Probit regression", None),
            (_, Link::Cloglog) => ("Complementary log-log regression", if cmd.has_opt("eform", 2) { Some("exp(b)") } else { None }),
            (_, Link::Poisson) => ("Poisson regression", if cmd.has_opt("irr", 3) { Some("IRR") } else { None }),
        };
        let mut out = notes;
        out.push(String::new());
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        let ll_label = if robust { "Log pseudolikelihood" } else { "Log likelihood" };
        let chi_label = if robust { format!("Wald chi2({})", df_m) } else { format!("LR chi2({})", df_m) };
        out.push(format!("{:<52}{:<15} = {:>10}", title, "Number of obs", g10(n)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", chi_label, f2(chi2)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > chi2", f4(p)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("{} = {}", ll_label, fmt_g(res.ll, 10)), "Pseudo R2", f4(r2_p)));
        if let Some(g) = nclust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", g, match &vce { Vce::Cluster(c) => c.as_str(), _ => "" })));
        }
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec {
            depvar: &dep,
            names: &d.names,
            b: &b,
            v: &v,
            omitted: &d.omitted,
            is_base: &d.is_base,
            df: None,
            level,
            eform,
            coef_title: "Coefficient",
        }));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let verb = self.verb;
        let (vcetype, vcelabel) = est::vce_label(&vce);
        est::post(st, verb, &d, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_0", ll0);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", p);
            e.set("r2_p", r2_p);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if let Some(g) = nclust {
                e.set("N_clust", g as f64);
            }
            e.set_str("chi2type", if robust { "Wald" } else { "LR" });
            e.set_str("title", title);
            e.set_str("vce", vcetype);
            if !vcelabel.is_empty() {
                e.set_str("vcetype", vcelabel);
            }
            e.set_str("_output", text.clone());
            if is_count {
                e.set_str("predict", "poisso_p");
            }
        });
        if !offset.iter().all(|&o| o == 0.0) {
            if let Some(o) = cmd.opt("offset", 3) {
                st.e.set_str("offset", o.trim());
            }
            if let Some(o) = cmd.opt("exposure", 1) {
                st.e.set_str("offset", format!("ln({})", o.trim()));
            }
        }
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// nbreg (NB2)
// ---------------------------------------------------------------------------

struct NbModel<'a> {
    d: &'a Design,
    act: Vec<usize>,
    offset: Vec<f64>,
}

impl<'a> NbModel<'a> {
    fn obs(&self, b: &[f64], t: usize) -> (f64, Vec<f64>) {
        let k = self.act.len();
        let la = b[k];
        let a = la.exp();
        let m = 1.0 / a;
        let y = self.d.y[t];
        let xb: f64 = self.act.iter().zip(b).map(|(&j, bj)| self.d.cols[j][t] * bj).sum::<f64>() + self.offset[t];
        let mu = xb.exp();
        let ll = ln_gamma(y + m) - ln_gamma(m) - ln_gamma(y + 1.0) - (m + y) * (a * mu).ln_1p() + y * (a * mu).ln();
        let d_xb = (y - mu) / (1.0 + a * mu);
        let d_a = -(digamma(y + m) - digamma(m)) / (a * a) + (a * mu).ln_1p() / (a * a) - (m + y) * mu / (1.0 + a * mu) + y / a;
        let mut g: Vec<f64> = self.act.iter().map(|&j| d_xb * self.d.cols[j][t]).collect();
        g.push(d_a * a);
        (ll, g)
    }
}

impl<'a> Likelihood for NbModel<'a> {
    fn k(&self) -> usize {
        self.act.len() + 1
    }

    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let kk = self.k();
        let grad = |b: &[f64]| -> (f64, Vec<f64>, Vec<Vec<f64>>) {
            let mut ll = 0.0;
            let mut g = vec![0.0; kk];
            let mut sc = Vec::new();
            for t in 0..self.d.n() {
                let w = self.d.w[t];
                let (l, gt) = self.obs(b, t);
                ll += w * l;
                for j in 0..kk {
                    g[j] += w * gt[j];
                }
                if need_scores {
                    sc.push(gt.iter().map(|x| x * w).collect());
                }
            }
            (ll, g, sc)
        };
        let (ll, g, sc) = grad(b);
        // numerical Hessian from analytic gradient
        let mut h = Matrix::zeros(kk, kk);
        for j in 0..kk {
            let eps = 1e-5 * (1.0 + b[j].abs());
            let mut bp = b.to_vec();
            bp[j] += eps;
            let mut bm = b.to_vec();
            bm[j] -= eps;
            let gp = grad(&bp).1;
            let gm = grad(&bm).1;
            for i in 0..kk {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        for i in 0..kk {
            for j in 0..i {
                let avg = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = avg;
                h[(j, i)] = avg;
            }
        }
        (ll, g, h, sc)
    }
}

pub struct Nbreg;

impl CommandHandler for Nbreg {
    fn name(&self) -> &'static str {
        "nbreg"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "nbreg");
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fip", extra: vec![] })?;
        let cl = est::cluster_values(ds, &d, &vce)?;
        let offset = offset_values(ds, &d, cmd)?;
        let act = d.active();
        // start from Poisson
        let pm = IndexModel { d: &d, act: act.clone(), link: Link::Poisson, offset: offset.clone() };
        let pres = est::maximize(&pm, vec![0.0; act.len()], 200)?;
        let mut b0 = pres.b.clone();
        b0.push(0.0);
        let model = NbModel { d: &d, act: act.clone(), offset: offset.clone() };
        let res = est::maximize(&model, b0, 300)?;
        let n = d.n_eff();
        let (vfull, _) = ml_vce(&res, &vce, cl.as_deref(), n);
        let k = d.k();
        let mut names = d.names.clone();
        names.push("/:lnalpha".into());
        let mut b = vec![0.0; k + 1];
        let mut idx: Vec<usize> = act.clone();
        idx.push(k);
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vfull, &idx, k + 1);
        // constant-only NB for LR test
        let mut d0 = d.clone();
        if d.has_cons {
            for j in 0..d0.k() {
                d0.omitted[j] = j != d0.k() - 1;
            }
        }
        let m0 = NbModel { d: &d0, act: vec![d0.k() - 1], offset: offset.clone() };
        let ll0 = est::maximize(&m0, vec![(d.y.iter().sum::<f64>() / d.n() as f64).max(1e-8).ln(), 0.0], 200).map(|r| r.ll).unwrap_or(SYSMISS);
        let chi2 = 2.0 * (res.ll - ll0);
        let df_m = (act.len() - 1) as f64;
        let p = dist::chi2tail(df_m, chi2);
        let alpha = b[k].exp();
        let level = util::level(cmd, st)?;
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "Negative binomial regression", "Number of obs", g10(n)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", format!("LR chi2({})", df_m), f2(chi2)));
        out.push(format!("{:<52}{:<15} = {:>10}", "Dispersion: mean", "Prob > chi2", f4(p)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Log likelihood = {}", fmt_g(res.ll, 10)), "Pseudo R2", f4(1.0 - res.ll / ll0)));
        out.push(String::new());
        let mut omitted = d.omitted.clone();
        omitted.push(false);
        let mut is_base = d.is_base.clone();
        is_base.push(false);
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: if cmd.has_opt("irr", 3) { Some("IRR") } else { None }, coef_title: "Coefficient" }));
        out.push(format!("{:>12} | {:>10}", "alpha", g10(alpha)));
        // LR test of alpha = 0 (vs Poisson)
        let chibar = (2.0 * (res.ll - pres.ll)).max(0.0);
        out.push(format!("LR test of alpha=0: chibar2(01) = {:.2}{:>20}Prob >= chibar2 = {:.3}", chibar, "", 0.5 * dist::chi2tail(1.0, chibar)));
        let text = out.join("\n");
        let mut d2 = d.clone();
        d2.names = names.clone();
        d2.omitted = omitted;
        d2.is_base = is_base;
        d2.cols.push(vec![0.0; d.n()]);
        est::post(st, "nbreg", &d2, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_0", ll0);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", p);
            e.set("alpha", alpha);
            e.set("r2_p", 1.0 - res.ll / ll0);
            e.set("chi2_c", chibar);
            e.set_str("title", "Negative binomial regression");
            e.set_str("_output", text.clone());
            e.set_str("predict", "nbreg_p");
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// Ordered logit / probit
// ---------------------------------------------------------------------------

struct OrderedModel<'a> {
    d: &'a Design,
    act: Vec<usize>,
    cats: Vec<f64>,
    yidx: Vec<usize>,
    probit: bool,
}

impl<'a> OrderedModel<'a> {
    fn cdf(&self, z: f64) -> f64 {
        if self.probit { dist::normal(z) } else { dist::invlogit(z) }
    }
    fn pdf(&self, z: f64) -> f64 {
        if self.probit {
            dist::normalden(z)
        } else {
            let p = dist::invlogit(z);
            p * (1.0 - p)
        }
    }
    fn obs(&self, b: &[f64], t: usize) -> (f64, Vec<f64>) {
        let k = self.act.len();
        let nc = self.cats.len();
        let xb: f64 = self.act.iter().zip(b).map(|(&j, bj)| self.d.cols[j][t] * bj).sum();
        let j = self.yidx[t];
        let cut = |c: usize| b[k + c];
        let (hi, lo) = (if j < nc - 1 { Some(cut(j)) } else { None }, if j > 0 { Some(cut(j - 1)) } else { None });
        let fh = hi.map_or(1.0, |c| self.cdf(c - xb));
        let fl = lo.map_or(0.0, |c| self.cdf(c - xb));
        let p = (fh - fl).max(1e-300);
        let dh = hi.map_or(0.0, |c| self.pdf(c - xb));
        let dl = lo.map_or(0.0, |c| self.pdf(c - xb));
        let mut g = vec![0.0; k + nc - 1];
        for (a, &col) in self.act.iter().enumerate() {
            g[a] = -(dh - dl) / p * self.d.cols[col][t];
        }
        if let Some(_) = hi {
            g[k + j] += dh / p;
        }
        if let Some(_) = lo {
            g[k + j - 1] -= dl / p;
        }
        (p.ln(), g)
    }
}

impl<'a> Likelihood for OrderedModel<'a> {
    fn k(&self) -> usize {
        self.act.len() + self.cats.len() - 1
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let kk = self.k();
        // enforce ordering of cutpoints
        let k = self.act.len();
        for c in 1..self.cats.len() - 1 {
            if b[k + c] <= b[k + c - 1] {
                return (f64::NEG_INFINITY, vec![0.0; kk], Matrix::identity(kk), Vec::new());
            }
        }
        let grad = |b: &[f64]| {
            let mut ll = 0.0;
            let mut g = vec![0.0; kk];
            let mut sc = Vec::new();
            for t in 0..self.d.n() {
                let w = self.d.w[t];
                let (l, gt) = self.obs(b, t);
                ll += w * l;
                for j in 0..kk {
                    g[j] += w * gt[j];
                }
                if need_scores {
                    sc.push(gt.iter().map(|x| x * w).collect::<Vec<f64>>());
                }
            }
            (ll, g, sc)
        };
        let (ll, g, sc) = grad(b);
        let mut h = Matrix::zeros(kk, kk);
        for j in 0..kk {
            let eps = 1e-6 * (1.0 + b[j].abs());
            let mut bp = b.to_vec();
            bp[j] += eps;
            let mut bm = b.to_vec();
            bm[j] -= eps;
            let gp = grad(&bp).1;
            let gm = grad(&bm).1;
            for i in 0..kk {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        for i in 0..kk {
            for j in 0..i {
                let a = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = a;
                h[(j, i)] = a;
            }
        }
        (ll, g, h, sc)
    }
}

pub struct Ordered {
    pub verb: &'static str,
    pub probit: bool,
}

impl CommandHandler for Ordered {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, self.verb);
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: true, weights: "fip", extra: vec![] })?;
        let cl = est::cluster_values(ds, &d, &vce)?;
        let mut cats = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        if cats.len() < 2 {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        let yidx: Vec<usize> = d.y.iter().map(|y| cats.iter().position(|c| c == y).unwrap()).collect();
        let act = d.active();
        let nc = cats.len();
        // starting cutpoints from cumulative proportions
        let n = d.n() as f64;
        let mut b0 = vec![0.0; act.len()];
        let mut cum = 0.0;
        for c in 0..nc - 1 {
            cum += yidx.iter().filter(|&&j| j == c).count() as f64 / n;
            b0.push(if self.probit { dist::invnormal(cum.clamp(1e-6, 1.0 - 1e-6)) } else { dist::logit(cum.clamp(1e-6, 1.0 - 1e-6)) });
        }
        let model = OrderedModel { d: &d, act: act.clone(), cats: cats.clone(), yidx: yidx.clone(), probit: self.probit };
        let res = est::maximize(&model, b0.clone(), 300)?;
        let m0 = OrderedModel { d: &d, act: vec![], cats: cats.clone(), yidx: yidx.clone(), probit: self.probit };
        let ll0 = est::maximize(&m0, b0[act.len()..].to_vec(), 200).map(|r| r.ll).unwrap_or(SYSMISS);
        let neff = d.n_eff();
        let (vfull, _) = ml_vce(&res, &vce, cl.as_deref(), neff);
        let k = d.k();
        let mut names = d.names.clone();
        let mut idx = act.clone();
        for c in 0..nc - 1 {
            names.push(format!("/:cut{}", c + 1));
            idx.push(k + c);
        }
        let total = k + nc - 1;
        let mut b = vec![0.0; total];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vfull, &idx, total);
        let chi2 = 2.0 * (res.ll - ll0);
        let df_m = act.len() as f64;
        let p = dist::chi2tail(df_m, chi2);
        let level = util::level(cmd, st)?;
        let title = if self.probit { "Ordered probit regression" } else { "Ordered logistic regression" };
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", title, "Number of obs", g10(neff)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", format!("LR chi2({})", df_m), f2(chi2)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > chi2", f4(p)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Log likelihood = {}", fmt_g(res.ll, 10)), "Pseudo R2", f4(1.0 - res.ll / ll0)));
        out.push(String::new());
        let mut omitted = d.omitted.clone();
        omitted.extend(vec![false; nc - 1]);
        let mut is_base = d.is_base.clone();
        is_base.extend(vec![false; nc - 1]);
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: if cmd.has_opt("or", 2) { Some("Odds ratio") } else { None }, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let mut d2 = d.clone();
        d2.names = names;
        d2.omitted = omitted;
        d2.is_base = is_base;
        let verb = self.verb;
        est::post(st, verb, &d2, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_0", ll0);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", p);
            e.set("r2_p", 1.0 - res.ll / ll0);
            e.set("k_cat", nc as f64);
            e.set_mat("cat", Matrix::row_vector(&cats));
            e.set_str("title", title);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// Multinomial logit
// ---------------------------------------------------------------------------

struct MlogitModel<'a> {
    d: &'a Design,
    act: Vec<usize>,
    yidx: Vec<usize>,
    ncat: usize,
    base: usize,
}

impl<'a> Likelihood for MlogitModel<'a> {
    fn k(&self) -> usize {
        self.act.len() * (self.ncat - 1)
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.act.len();
        let kk = self.k();
        let eqs: Vec<usize> = (0..self.ncat).filter(|&c| c != self.base).collect();
        let mut ll = 0.0;
        let mut g = vec![0.0; kk];
        let mut h = Matrix::zeros(kk, kk);
        let mut scores = Vec::new();
        for t in 0..self.d.n() {
            let w = self.d.w[t];
            let x: Vec<f64> = self.act.iter().map(|&j| self.d.cols[j][t]).collect();
            let mut eta = vec![0.0; self.ncat];
            for (e, &c) in eqs.iter().enumerate() {
                eta[c] = (0..k).map(|a| b[e * k + a] * x[a]).sum();
            }
            let mx = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = eta.iter().map(|v| (v - mx).exp()).sum();
            let p: Vec<f64> = eta.iter().map(|v| (v - mx).exp() / den).collect();
            ll += w * p[self.yidx[t]].ln();
            let mut sc = vec![0.0; kk];
            for (e, &c) in eqs.iter().enumerate() {
                let r = (if self.yidx[t] == c { 1.0 } else { 0.0 }) - p[c];
                for a in 0..k {
                    g[e * k + a] += w * r * x[a];
                    sc[e * k + a] = w * r * x[a];
                }
                for (f, &c2) in eqs.iter().enumerate() {
                    let wgt = p[c] * ((c == c2) as i32 as f64 - p[c2]);
                    for a in 0..k {
                        for a2 in 0..k {
                            h[(e * k + a, f * k + a2)] += w * wgt * x[a] * x[a2];
                        }
                    }
                }
            }
            if need_scores {
                scores.push(sc);
            }
        }
        (ll, g, h, scores)
    }
}

pub struct Mlogit;

impl CommandHandler for Mlogit {
    fn name(&self) -> &'static str {
        "mlogit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "mlogit");
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: false, weights: "fip", extra: vec![] })?;
        let cl = est::cluster_values(ds, &d, &vce)?;
        let mut cats = d.y.clone();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let ncat = cats.len();
        if ncat < 2 {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        let yidx: Vec<usize> = d.y.iter().map(|y| cats.iter().position(|c| c == y).unwrap()).collect();
        let base = match util::opt_num(cmd, "baseoutcome", 1)? {
            Some(v) => cats.iter().position(|&c| c == v).ok_or_else(|| rc_err(198, "baseoutcome() not found"))?,
            None => {
                // most frequent
                (0..ncat).max_by_key(|&c| (yidx.iter().filter(|&&j| j == c).count(), std::cmp::Reverse(c))).unwrap()
            }
        };
        let act = d.active();
        let model = MlogitModel { d: &d, act: act.clone(), yidx: yidx.clone(), ncat, base };
        let res = est::maximize(&model, vec![0.0; model.k()], 300)?;
        let neff = d.n_eff();
        let (vfull, _) = ml_vce(&res, &vce, cl.as_deref(), neff);
        // null model: constant only
        let freqs: Vec<f64> = (0..ncat).map(|c| yidx.iter().zip(&d.w).filter(|(j, _)| **j == c).map(|(_, w)| w).sum::<f64>()).collect();
        let tot: f64 = freqs.iter().sum();
        let ll0: f64 = freqs.iter().map(|f| if *f > 0.0 { f * (f / tot).ln() } else { 0.0 }).sum();
        let chi2 = 2.0 * (res.ll - ll0);
        let df_m = ((act.len() - 1) * (ncat - 1)) as f64;
        let p = dist::chi2tail(df_m, chi2);
        // assemble full coefficient vector with base equation of zeros
        let k = d.k();
        let mut names = Vec::new();
        let mut b = Vec::new();
        let mut omitted = Vec::new();
        let mut is_base = Vec::new();
        let mut map = Vec::new();
        let eqs: Vec<usize> = (0..ncat).filter(|&c| c != base).collect();
        for c in 0..ncat {
            let eq = fmt_g(cats[c], 12);
            for j in 0..k {
                names.push(format!("{eq}:{}", d.names[j]));
                if c == base {
                    b.push(0.0);
                    omitted.push(true);
                    is_base.push(true);
                } else {
                    let e = eqs.iter().position(|&x| x == c).unwrap();
                    match act.iter().position(|&a| a == j) {
                        Some(a) => {
                            b.push(res.b[e * act.len() + a]);
                            map.push(names.len() - 1);
                            omitted.push(false);
                        }
                        None => {
                            b.push(0.0);
                            omitted.push(true);
                        }
                    }
                    is_base.push(d.is_base[j]);
                }
            }
        }
        let v = est::expand_v(&vfull, &map, names.len());
        let level = util::level(cmd, st)?;
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "Multinomial logistic regression", "Number of obs", g10(neff)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", format!("LR chi2({})", df_m), f2(chi2)));
        out.push(format!("{:<52}{:<15} = {:>10}", "", "Prob > chi2", f4(p)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Log likelihood = {}", fmt_g(res.ll, 10)), "Pseudo R2", f4(1.0 - res.ll / ll0)));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: if cmd.has_opt("rrr", 2) { Some("RRR") } else { None }, coef_title: "Coefficient" }));
        out.push(format!("Note: {} is the base outcome.", fmt_g(cats[base], 12)));
        let text = out.join("\n");
        let mut d2 = d.clone();
        d2.names = names;
        d2.omitted = omitted;
        d2.is_base = is_base;
        est::post(st, "mlogit", &d2, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_0", ll0);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", p);
            e.set("r2_p", 1.0 - res.ll / ll0);
            e.set("k_out", ncat as f64);
            e.set("ibaseout", (base + 1) as f64);
            e.set("baseout", cats[base]);
            e.set_mat("out", Matrix::row_vector(&cats));
            e.set_str("title", "Multinomial logistic regression");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// tobit
// ---------------------------------------------------------------------------

struct TobitModel<'a> {
    d: &'a Design,
    act: Vec<usize>,
    ll_lim: Option<f64>,
    ul_lim: Option<f64>,
}

impl<'a> TobitModel<'a> {
    fn obs(&self, b: &[f64], t: usize) -> f64 {
        let k = self.act.len();
        let s = b[k].exp();
        let xb: f64 = self.act.iter().zip(b).map(|(&j, bj)| self.d.cols[j][t] * bj).sum();
        let y = self.d.y[t];
        if self.ll_lim.map_or(false, |l| y <= l) {
            dist::normal((self.ll_lim.unwrap() - xb) / s).max(1e-300).ln()
        } else if self.ul_lim.map_or(false, |u| y >= u) {
            dist::normal((xb - self.ul_lim.unwrap()) / s).max(1e-300).ln()
        } else {
            dist::lnnormalden((y - xb) / s) - s.ln()
        }
    }
}

impl<'a> Likelihood for TobitModel<'a> {
    fn k(&self) -> usize {
        self.act.len() + 1
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let kk = self.k();
        let total = |b: &[f64]| (0..self.d.n()).map(|t| self.d.w[t] * self.obs(b, t)).sum::<f64>();
        let ll = total(b);
        let grad_obs = |b: &[f64], t: usize| -> Vec<f64> {
            (0..kk)
                .map(|j| {
                    let eps = 1e-6 * (1.0 + b[j].abs());
                    let mut bp = b.to_vec();
                    bp[j] += eps;
                    let mut bm = b.to_vec();
                    bm[j] -= eps;
                    (self.obs(&bp, t) - self.obs(&bm, t)) / (2.0 * eps)
                })
                .collect()
        };
        let grad = |b: &[f64]| -> Vec<f64> {
            let mut g = vec![0.0; kk];
            for t in 0..self.d.n() {
                let gt = grad_obs(b, t);
                for j in 0..kk {
                    g[j] += self.d.w[t] * gt[j];
                }
            }
            g
        };
        let g = grad(b);
        let mut h = Matrix::zeros(kk, kk);
        for j in 0..kk {
            let eps = 1e-5 * (1.0 + b[j].abs());
            let mut bp = b.to_vec();
            bp[j] += eps;
            let mut bm = b.to_vec();
            bm[j] -= eps;
            let gp = grad(&bp);
            let gm = grad(&bm);
            for i in 0..kk {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        let scores = if need_scores { (0..self.d.n()).map(|t| grad_obs(b, t).iter().map(|x| x * self.d.w[t]).collect()).collect() } else { Vec::new() };
        (ll, g, h, scores)
    }
}

pub struct Tobit;

impl CommandHandler for Tobit {
    fn name(&self) -> &'static str {
        "tobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fa", extra: vec![] })?;
        let ymin = d.y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ymax = d.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lim = |name: &str, def: f64| -> Result<Option<f64>, String> {
            if cmd.has_opt(name, 2) {
                Ok(Some(util::opt_num(cmd, name, 2)?.unwrap_or(def)))
            } else {
                Ok(None)
            }
        };
        let ll_lim = lim("ll", ymin)?;
        let ul_lim = lim("ul", ymax)?;
        let act = d.active();
        let fit = est::ols(&d, &Vce::Ols, None)?;
        let mut b0: Vec<f64> = act.iter().map(|&j| fit.b[j]).collect();
        b0.push((fit.rss / d.n() as f64).sqrt().ln());
        let model = TobitModel { d: &d, act: act.clone(), ll_lim, ul_lim };
        let res = est::maximize(&model, b0, 300)?;
        let (vfull, _) = res.info.sym_inverse(1e-14);
        let k = d.k();
        let mut idx = act.clone();
        idx.push(k);
        let mut b = vec![0.0; k + 1];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let mut v = est::expand_v(&vfull, &idx, k + 1);
        // report sigma (delta method)
        let sigma = b[k].exp();
        let vs = v[(k, k)] * sigma * sigma;
        v[(k, k)] = vs;
        b[k] = sigma;
        let mut names = d.names.clone();
        names.push("/:sigma".into());
        let level = util::level(cmd, st)?;
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "Tobit regression", "Number of obs", g10(d.n_eff())));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Log likelihood = {}", fmt_g(res.ll, 10)), "", ""));
        out.push(String::new());
        let mut omitted = d.omitted.clone();
        omitted.push(false);
        let mut is_base = d.is_base.clone();
        is_base.push(false);
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: Some(d.n_eff() - act.len() as f64), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let mut d2 = d.clone();
        d2.names = names;
        d2.omitted = omitted;
        d2.is_base = is_base;
        est::post(st, "tobit", &d2, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("sigma", sigma);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// anova
// ---------------------------------------------------------------------------

pub struct Anova;

impl CommandHandler for Anova {
    fn name(&self) -> &'static str {
        "anova"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "anova");
        }
        let (dep, terms_text) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        // in anova, bare variables are categorical
        let terms: Vec<String> = terms_text
            .split_whitespace()
            .map(|t| {
                t.split('#')
                    .map(|p| if p.contains('.') || p.is_empty() { p.to_string() } else { format!("i.{p}") })
                    .collect::<Vec<_>>()
                    .join("#")
            })
            .collect();
        let full = terms.join(" ");
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &full, noconstant: false, weights: "fa", extra: vec![] })?;
        let fit = est::ols(&d, &Vce::Ols, None)?;
        let n = d.n_eff();
        let mss = fit.tss - fit.rss;
        let mse = fit.rss / fit.df_r;
        let fm = (mss / fit.df_m) / mse;
        let sequential = cmd.has_opt("sequential", 3);
        // partial (or sequential) SS per term: refit without the term
        let mut rows = Vec::new();
        let mut prev_rss = fit.tss;
        for (i, term) in terms.iter().enumerate() {
            let reduced: Vec<String> = if sequential { terms[..=i].to_vec() } else { terms.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, t)| t.clone()).collect() };
            let mut dr = d.clone();
            let keep_names = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &reduced.join(" "), noconstant: false, weights: "fa", extra: vec![] })?;
            let _ = &mut dr;
            let fr = est::ols(&keep_names, &Vce::Ols, None)?;
            let (ss, df) = if sequential {
                let s = prev_rss - fr.rss;
                let df = fr.df_m - rows.iter().map(|r: &(String, f64, f64)| r.2).sum::<f64>();
                prev_rss = fr.rss;
                (s, df)
            } else {
                (fr.rss - fit.rss, fit.df_m - fr.df_m)
            };
            rows.push((term.replace("i.", ""), ss, df));
        }
        let r2 = mss / fit.tss;
        let adj = 1.0 - (1.0 - r2) * (n - 1.0) / fit.df_r;
        let rmse = mse.sqrt();
        let mut out = vec![
            format!("{:>25}Number of obs = {:>10}    R-squared     = {:>8}", "", g10(n), f4(r2)),
            format!("{:>25}Root MSE      = {:>10}    Adj R-squared = {:>8}", "", g10(rmse), f4(adj)),
            String::new(),
            format!("{:>24} | {}    df         MS        F    Prob>F", "Source", if sequential { "    Seq. SS" } else { " Partial SS" }),
            format!("{:>24}-+----------------------------------------------------", "-".repeat(11)),
            format!("{:>24} | {:>11} {:>9} {:>11} {:>9} {:>8}", "Model", g10(mss), fit.df_m, g10(mss / fit.df_m), f2(fm), f4(dist::ftail(fit.df_m, fit.df_r, fm))),
            format!("{:>24} |", ""),
        ];
        let mut stats = Vec::new();
        for (name, ss, df) in &rows {
            let f = (ss / df) / mse;
            let p = dist::ftail(*df, fit.df_r, f);
            out.push(format!("{:>24} | {:>11} {:>9} {:>11} {:>9} {:>8}", name, g10(*ss), df, g10(ss / df), f2(f), f4(p)));
            out.push(format!("{:>24} |", ""));
            stats.push((name.clone(), *ss, *df, f));
        }
        out.push(format!("{:>24} | {:>11} {:>9} {:>11}", "Residual", g10(fit.rss), fit.df_r, g10(mse)));
        out.push(format!("{:>24}-+----------------------------------------------------", "-".repeat(11)));
        out.push(format!("{:>24} | {:>11} {:>9} {:>11}", "Total", g10(fit.tss), n - 1.0, g10(fit.tss / (n - 1.0))));
        let text = out.join("\n");
        est::post(st, "anova", &d, &fit.b, &fit.v, &|e: &mut Results| {
            e.set("df_m", fit.df_m);
            e.set("df_r", fit.df_r);
            e.set("F", fm);
            e.set("r2", r2);
            e.set("r2_a", adj);
            e.set("rmse", rmse);
            e.set("mss", mss);
            e.set("rss", fit.rss);
            for (i, (name, ss, df, f)) in stats.iter().enumerate() {
                e.set(&format!("ss_{}", i + 1), *ss);
                e.set(&format!("df_{}", i + 1), *df);
                e.set(&format!("F_{}", i + 1), *f);
                e.set_str(&format!("term_{}", i + 1), name.clone());
            }
            e.set_str("varnames", terms_text.clone());
            e.set_str("_output", text.clone());
            e.set_str("predict", "regres_p");
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// Panel data: xtreg fe/re/be, areg
// ---------------------------------------------------------------------------

fn group_index(ds: &DataSet, d: &Design, var: &str) -> Result<(Vec<usize>, usize), String> {
    let c = util::col(ds, var)?;
    let mut keys: Vec<f64> = d.rows.iter().map(|&i| c[i]).collect();
    keys.sort_by(f64::total_cmp);
    keys.dedup();
    let idx = d.rows.iter().map(|&i| keys.binary_search_by(|k| k.total_cmp(&c[i])).unwrap()).collect();
    Ok((idx, keys.len()))
}

/// Demean columns (and y) within groups. Returns group means of y and x.
fn within(d: &mut Design, g: &[usize], ng: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut cnt = vec![0.0; ng];
    for (t, &gi) in g.iter().enumerate() {
        cnt[gi] += d.w[t];
    }
    let mean_of = |v: &[f64], w: &[f64]| {
        let mut m = vec![0.0; ng];
        for (t, &gi) in g.iter().enumerate() {
            m[gi] += w[t] * v[t];
        }
        for (gi, mm) in m.iter_mut().enumerate() {
            *mm /= cnt[gi];
        }
        m
    };
    let ym = mean_of(&d.y, &d.w);
    let xm: Vec<Vec<f64>> = d.cols.iter().map(|c| mean_of(c, &d.w)).collect();
    (ym, xm)
}

pub struct Xtreg;

impl CommandHandler for Xtreg {
    fn name(&self) -> &'static str {
        "xtreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return replay(st, "xtreg");
        }
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let panel = match cmd.opt("i", 1) {
            Some(p) => crate::varlist::resolve_name(ds, p.trim())?,
            None => st.tsset.as_ref().and_then(|(p, _)| p.clone()).ok_or_else(|| rc_err(459, "panel variable not set; use xtset varname"))?,
        };
        let model = if cmd.has_opt("fe", 2) { "fe" } else if cmd.has_opt("be", 2) { "be" } else { "re" };
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: false, weights: "", extra: vec![panel.clone()] })?;
        let (g, ng) = group_index(ds, &d, &panel)?;
        let n = d.n() as f64;
        let level = util::level(cmd, st)?;
        let k = d.k();
        let cons = k - 1;
        let (ym, xm) = within(&mut d, &g, ng);
        let ybar = util::mean(&d.y);
        let xbar: Vec<f64> = d.cols.iter().map(|c| util::mean(c)).collect();
        let tcount: Vec<f64> = (0..ng).map(|gi| g.iter().filter(|&&x| x == gi).count() as f64).collect();
        let mut out = Vec::new();
        let (b, v, stats): (Vec<f64>, Matrix, Vec<(String, f64)>);
        match model {
            "fe" => {
                let mut dw = d.clone();
                for t in 0..d.n() {
                    dw.y[t] = d.y[t] - ym[g[t]] + ybar;
                    for j in 0..k {
                        if j != cons {
                            dw.cols[j][t] = d.cols[j][t] - xm[j][g[t]] + xbar[j];
                        }
                    }
                }
                est::drop_collinear(&mut dw);
                let cl = match &vce {
                    Vce::Robust => Some(g.iter().map(|&x| x as f64).collect::<Vec<_>>()),
                    Vce::Cluster(c) => {
                        let cc = util::col(ds, c)?;
                        Some(d.rows.iter().map(|&i| cc[i]).collect())
                    }
                    _ => None,
                };
                let vce2 = if cl.is_some() { Vce::Cluster(panel.clone()) } else { Vce::Ols };
                let fit = est::ols(&dw, &vce2, cl.as_deref())?;
                let kx = dw.active().len() as f64 - 1.0;
                let df_r = n - ng as f64 - kx;
                let scale = fit.df_r / df_r;
                // with clustered errors the reference distribution has G-1 df
                let df_t = match fit.n_clust {
                    Some(gc) if cl.is_some() => gc as f64 - 1.0,
                    _ => df_r,
                };
                let vv = if cl.is_some() {
                    // Stata's xtreg fe cluster uses (N-1)/(N-K) with K excluding the absorbed effects
                    fit.v.clone()
                } else {
                    fit.v.scale(scale)
                };
                let sigma_e = (fit.rss / df_r).sqrt();
                // u_i
                let ui: Vec<f64> = (0..ng).map(|gi| ym[gi] - (0..k).map(|j| fit.b[j] * if j == cons { 1.0 } else { xm[j][gi] }).sum::<f64>()).collect();
                let su = {
                    let m = util::mean(&ui);
                    (ui.iter().map(|u| (u - m).powi(2)).sum::<f64>() / (ng as f64 - 1.0)).sqrt()
                };
                let rho = su * su / (su * su + sigma_e * sigma_e);
                // R2 within/between/overall
                let fitted: Vec<f64> = (0..d.n()).map(|t| (0..k).filter(|&j| j != cons).map(|j| fit.b[j] * d.cols[j][t]).sum()).collect();
                let wy: Vec<f64> = (0..d.n()).map(|t| d.y[t] - ym[g[t]]).collect();
                let wf: Vec<f64> = (0..d.n()).map(|t| fitted[t] - (0..k).filter(|&j| j != cons).map(|j| fit.b[j] * xm[j][g[t]]).sum::<f64>()).collect();
                let corr2 = |a: &[f64], b: &[f64]| {
                    let ma = util::mean(a);
                    let mb = util::mean(b);
                    let sab: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
                    let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
                    let sbb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
                    sab * sab / (saa * sbb)
                };
                let r2_w = corr2(&wy, &wf);
                let gf: Vec<f64> = (0..ng).map(|gi| (0..k).filter(|&j| j != cons).map(|j| fit.b[j] * xm[j][gi]).sum()).collect();
                let r2_b = corr2(&ym, &gf);
                let r2_o = corr2(&d.y, &fitted);
                let (wald, q) = est::wald_all(&dw, &fit.b, &vv);
                let f = wald / q;
                // F test that all u_i = 0
                let mut dp = d.clone();
                dp.omitted = dw.omitted.clone();
                let pooled = est::ols(&dp, &Vce::Ols, None)?;
                let ui_obs: Vec<f64> = (0..d.n()).map(|t| ui[g[t]]).collect();
                let corr_u_xb = {
                    let (ma, mb) = (util::mean(&ui_obs), util::mean(&fitted));
                    let sab: f64 = ui_obs.iter().zip(&fitted).map(|(a, b)| (a - ma) * (b - mb)).sum();
                    let saa: f64 = ui_obs.iter().map(|a| (a - ma).powi(2)).sum();
                    let sbb: f64 = fitted.iter().map(|b| (b - mb).powi(2)).sum();
                    sab / (saa * sbb).sqrt()
                };
                let fu = ((pooled.rss - fit.rss) / (ng as f64 - 1.0)) / (fit.rss / df_r);
                stats = vec![
                    ("sigma_u".into(), su),
                    ("sigma_e".into(), sigma_e),
                    ("rho".into(), rho),
                    ("r2_w".into(), r2_w),
                    ("r2_b".into(), r2_b),
                    ("r2_o".into(), r2_o),
                    ("F".into(), f),
                    ("df_r".into(), df_t),
                    ("df_m".into(), q),
                    ("F_f".into(), fu),
                    ("corr".into(), corr_u_xb),
                    ("df_a".into(), ng as f64 - 1.0),
                ];
                out.extend(dw.notes.iter().cloned());
                if !dw.notes.is_empty() {
                    out.push(String::new());
                }
                out.push(format!("{:<52}{:<15} = {:>10}", "Fixed-effects (within) regression", "Number of obs", g10(n)));
                out.push(format!("{:<52}{:<15} = {:>10}", format!("Group variable: {panel}"), "Number of groups", ng));
                out.push(String::new());
                out.push(format!("{:<52}{}", "R-squared:", "Obs per group:"));
                out.push(format!("     Within  = {:<37.4}{:>15} = {:>10}", r2_w, "min", g10(tcount.iter().cloned().fold(f64::INFINITY, f64::min))));
                out.push(format!("     Between = {:<37.4}{:>15} = {:>10.1}", r2_b, "avg", n / ng as f64));
                out.push(format!("     Overall = {:<37.4}{:>15} = {:>10}", r2_o, "max", g10(tcount.iter().cloned().fold(0.0, f64::max))));
                out.push(String::new());
                out.push(format!("{:<52}{:<15} = {:>10}", "", format!("F({}, {})", q, df_t), f2(f)));
                out.push(format!("{:<52}{:<15} = {:>10}", format!("corr(u_i, Xb) = {:.4}", corr_u_xb), "Prob > F", f4(dist::ftail(q, df_t, f))));
                if let Some(gc) = fit.n_clust.filter(|_| cl.is_some()) {
                    out.push(String::new());
                    out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", gc, match &vce { Vce::Cluster(c) => c.as_str(), _ => panel.as_str() })));
                }
                out.push(String::new());
                b = fit.b.clone();
                v = vv;
                out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &dw.names, b: &b, v: &v, omitted: &dw.omitted, is_base: &dw.is_base, df: Some(df_t), level, eform: None, coef_title: "Coefficient" }));
                out.pop();
                out.push(format!("{}+{}", "-".repeat(tbl_lw(&out) + 1), "-".repeat(64)));
                out.push(format!("{:>lw$} | {:>10}", "sigma_u", g10(su), lw = tbl_lw(&out)));
                out.push(format!("{:>lw$} | {:>10}", "sigma_e", g10(sigma_e), lw = tbl_lw(&out)));
                out.push(format!("{:>lw$} | {:>10}   (fraction of variance due to u_i)", "rho", g10(rho), lw = tbl_lw(&out)));
                out.push("-".repeat(tbl_lw(&out) + 66));
                out.push(format!("F test that all u_i=0: F({}, {}) = {:.2}{:>18}Prob > F = {:.4}", ng - 1, df_r, fu, "", dist::ftail(ng as f64 - 1.0, df_r, fu)));
                d = dw;
            }
            "be" => {
                // regression on group means
                let mut dm = d.clone();
                dm.y = ym.clone();
                dm.cols = xm.clone();
                dm.w = vec![1.0; ng];
                dm.rows = (0..ng).collect();
                let fit = est::ols(&dm, &Vce::Ols, None)?;
                b = fit.b.clone();
                v = fit.v.clone();
                let r2 = 1.0 - fit.rss / fit.tss;
                stats = vec![("r2_b".into(), r2), ("df_r".into(), fit.df_r), ("df_m".into(), fit.df_m), ("sd_b".into(), (fit.rss / fit.df_r).sqrt())];
                out.push(format!("{:<52}{:<15} = {:>10}", "Between regression (regression on group means)", "Number of obs", g10(n)));
                out.push(format!("{:<52}{:<15} = {:>10}", format!("Group variable: {panel}"), "Number of groups", ng));
                out.push(String::new());
                out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: Some(fit.df_r), level, eform: None, coef_title: "Coefficient" }));
            }
            _ => {
                // Swamy-Arora random effects
                let mut dw = d.clone();
                for t in 0..d.n() {
                    dw.y[t] = d.y[t] - ym[g[t]];
                    for j in 0..k {
                        dw.cols[j][t] = if j == cons { 0.0 } else { d.cols[j][t] - xm[j][g[t]] };
                    }
                }
                dw.omitted[cons] = true;
                est::drop_collinear(&mut dw);
                let fe = est::ols(&dw, &Vce::Ols, None)?;
                let kx = (dw.active().len()) as f64;
                let se2 = fe.rss / (n - ng as f64 - kx);
                let mut dm = d.clone();
                dm.y = ym.clone();
                dm.cols = xm.clone();
                dm.w = vec![1.0; ng];
                dm.rows = (0..ng).collect();
                let be = est::ols(&dm, &Vce::Ols, None)?;
                let tbar = ng as f64 / tcount.iter().map(|t| 1.0 / t).sum::<f64>();
                let su2 = (be.rss / (ng as f64 - (kx + 1.0)) - se2 / tbar).max(0.0);
                let theta: Vec<f64> = tcount.iter().map(|&ti| 1.0 - (se2 / (ti * su2 + se2)).sqrt()).collect();
                let mut dq = d.clone();
                for t in 0..d.n() {
                    let th = theta[g[t]];
                    dq.y[t] = d.y[t] - th * ym[g[t]];
                    for j in 0..k {
                        dq.cols[j][t] = d.cols[j][t] - th * xm[j][g[t]];
                    }
                }
                dq.has_cons = false;
                let cl = match &vce {
                    Vce::Robust => Some(g.iter().map(|&x| x as f64).collect::<Vec<_>>()),
                    Vce::Cluster(c) => {
                        let cc = util::col(ds, c)?;
                        Some(d.rows.iter().map(|&i| cc[i]).collect())
                    }
                    _ => None,
                };
                let fit = est::ols(&dq, if cl.is_some() { &Vce::Robust } else { &Vce::Ols }, cl.as_deref())?;
                let vv = if cl.is_some() {
                    let gg = ng as f64;
                    fit.v.scale(gg / (gg - 1.0) * (fit.df_r) / (n - 1.0))
                } else {
                    fit.xtx_inv.scale(fit.rss / (n - k as f64))
                };
                let sigma_e = se2.sqrt();
                let sigma_u = su2.sqrt();
                let rho = su2 / (su2 + se2);
                let (wald, q) = est::wald_all(&d, &fit.b, &vv);
                stats = vec![
                    ("sigma_u".into(), sigma_u),
                    ("sigma_e".into(), sigma_e),
                    ("rho".into(), rho),
                    ("chi2".into(), wald),
                    ("df_m".into(), q),
                    ("theta".into(), theta[0]),
                ];
                out.push(format!("{:<52}{:<15} = {:>10}", "Random-effects GLS regression", "Number of obs", g10(n)));
                out.push(format!("{:<52}{:<15} = {:>10}", format!("Group variable: {panel}"), "Number of groups", ng));
                out.push(String::new());
                out.push(format!("{:<52}{:<15} = {:>10}", "", format!("Wald chi2({})", q), f2(wald)));
                out.push(format!("{:<52}{:<15} = {:>10}", "corr(u_i, X) = 0 (assumed)", "Prob > chi2", f4(dist::chi2tail(q, wald))));
                out.push(String::new());
                b = fit.b.clone();
                v = vv;
                out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
                out.pop();
                out.push(format!("{}+{}", "-".repeat(tbl_lw(&out) + 1), "-".repeat(64)));
                out.push(format!("{:>lw$} | {:>10}", "sigma_u", g10(sigma_u), lw = tbl_lw(&out)));
                out.push(format!("{:>lw$} | {:>10}", "sigma_e", g10(sigma_e), lw = tbl_lw(&out)));
                out.push(format!("{:>lw$} | {:>10}   (fraction of variance due to u_i)", "rho", g10(rho), lw = tbl_lw(&out)));
                out.push("-".repeat(tbl_lw(&out) + 66));
            }
        }
        let text = out.join("\n");
        est::post(st, "xtreg", &d, &b, &v, &|e: &mut Results| {
            for (k, val) in &stats {
                e.set(k, *val);
            }
            e.set("N_g", ng as f64);
            e.set_str("model", model);
            e.set_str("ivar", panel.clone());
            e.set_str("_output", text.clone());
            e.set_str("predict", "xtrefe_p");
        });
        Ok(text)
    }
}

pub struct Areg;

impl CommandHandler for Areg {
    fn name(&self) -> &'static str {
        "areg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let absorb = crate::varlist::resolve_name(ds, cmd.opt("absorb", 1).ok_or_else(|| rc_err(198, "option absorb() required"))?.trim())?;
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: false, weights: "fa", extra: vec![absorb.clone()] })?;
        let (g, ng) = group_index(ds, &d, &absorb)?;
        let mut d2 = d.clone();
        let (ym, xm) = within(&mut d2, &g, ng);
        let ybar = util::mean(&d.y);
        let k = d.k();
        let cons = k - 1;
        let xbar: Vec<f64> = d.cols.iter().map(|c| util::mean(c)).collect();
        for t in 0..d.n() {
            d2.y[t] = d.y[t] - ym[g[t]] + ybar;
            for j in 0..k {
                if j != cons {
                    d2.cols[j][t] = d.cols[j][t] - xm[j][g[t]] + xbar[j];
                }
            }
        }
        est::drop_collinear(&mut d2);
        let cl = est::cluster_values(ds, &d2, &vce)?;
        let fit = est::ols(&d2, &vce, cl.as_deref())?;
        let n = d.n_eff();
        let kx = d2.active().len() as f64;
        let df_r = n - kx - (ng as f64 - 1.0);
        let v = if vce == Vce::Ols { fit.v.scale(fit.df_r / df_r) } else { fit.v.scale((n - kx) / df_r) };
        let r2 = 1.0 - fit.rss / {
            let m = util::mean(&d.y);
            d.y.iter().map(|y| (y - m).powi(2)).sum::<f64>()
        };
        let level = util::level(cmd, st)?;
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", "Linear regression, absorbing indicators", "Number of obs", g10(n)),
            format!("{:<52}{:<15} = {:>10}", format!("Absorbed variable: {absorb}"), "No. of categories", ng),
            format!("{:<52}{:<15} = {:>10}", "", "R-squared", f4(r2)),
            format!("{:<52}{:<15} = {:>10}", "", "Root MSE", g10((fit.rss / df_r).sqrt())),
            String::new(),
        ];
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d2.names, b: &fit.b, v: &v, omitted: &d2.omitted, is_base: &d2.is_base, df: Some(df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        est::post(st, "areg", &d2, &fit.b, &v, &|e: &mut Results| {
            e.set("df_r", df_r);
            e.set("df_a", ng as f64 - 1.0);
            e.set("r2", r2);
            e.set("rss", fit.rss);
            e.set("rmse", (fit.rss / df_r).sqrt());
            e.set_str("absvar", absorb.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// ivregress 2sls
// ---------------------------------------------------------------------------

pub struct Ivregress;

impl CommandHandler for Ivregress {
    fn name(&self) -> &'static str {
        "ivregress"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (estimator, rest) = main.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        if !matches!(estimator, "2sls" | "liml" | "gmm") {
            return Err(rc_err(198, format!("{estimator}: estimator not allowed")));
        }
        let open = rest.find('(').ok_or_else(|| rc_err(198, "(varlist2 = instlist) required"))?;
        let close = rest.rfind(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
        let inside = &rest[open + 1..close];
        let outside = format!("{} {}", &rest[..open], &rest[close + 1..]);
        let (endog, inst) = inside.split_once('=').ok_or_else(|| rc_err(198, "= required inside parentheses"))?;
        let (dep, exog) = split_dep(&outside)?;
        let dep = resolve_dep(ds, &dep)?;
        let xlist = format!("{} {}", endog.trim(), exog);
        let zlist = format!("{} {}", inst.trim(), exog);
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        // sample: union of all variables
        let all = format!("{} {}", xlist, inst);
        let dall = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &all, noconstant: cmd.has_opt("noconstant", 3), weights: "fap", extra: vec![] })?;
        let mut sub = cmd.clone();
        sub.if_expr = None;
        sub.in_range = None;
        let mut st2 = st.clone();
        st2.by_mask = Some(dall.mask.clone());
        let dx = est::build(ds, &st2, &sub, DesignSpec { depvar: Some(&dep), indep: &xlist, noconstant: cmd.has_opt("noconstant", 3), weights: "fap", extra: vec![] })?;
        let dz = est::build(ds, &st2, &sub, DesignSpec { depvar: Some(&dep), indep: &zlist, noconstant: cmd.has_opt("noconstant", 3), weights: "fap", extra: vec![] })?;
        let ax = dx.active();
        let az = dz.active();
        let n = dx.n_eff();
        // first stage: project X on Z
        let zz = est::cross(&dz, &az);
        let (zzi, _) = zz.sym_inverse(1e-14);
        let mut xhat = dx.clone();
        for &j in &ax {
            let mut zx = vec![0.0; az.len()];
            for t in 0..dx.n() {
                for (a, &zj) in az.iter().enumerate() {
                    zx[a] += dz.w[t] * dz.cols[zj][t] * dx.cols[j][t];
                }
            }
            let coef: Vec<f64> = (0..az.len()).map(|a| (0..az.len()).map(|c| zzi[(a, c)] * zx[c]).sum()).collect();
            for t in 0..dx.n() {
                xhat.cols[j][t] = az.iter().zip(&coef).map(|(&zj, c)| dz.cols[zj][t] * c).sum();
            }
        }
        let fit2 = est::ols(&xhat, &Vce::Ols, None)?;
        // first-stage statistics for estat firststage:
        // rows = endogenous regressors; cols = rsq adjrsq partrsq F df1 df2 p
        let excluded: Vec<usize> = az.iter().copied().filter(|&j| !dx.names.contains(&dz.names[j])).collect();
        let endog_cols: Vec<usize> = ax.iter().copied().filter(|&j| !dz.names.contains(&dx.names[j])).collect();
        let mut fs = Matrix::zeros(endog_cols.len(), 7);
        let mut fs_names = Vec::new();
        let cl_fs = est::cluster_values(ds, &dz, &vce)?;
        for (r, &j) in endog_cols.iter().enumerate() {
            let mut d1 = dz.clone();
            d1.y = dx.cols[j].clone();
            let full = est::ols(&d1, &vce, cl_fs.as_deref())?;
            let mut d0 = d1.clone();
            for &e in &excluded {
                d0.omitted[e] = true;
            }
            let restr = est::ols(&d0, &Vce::Ols, None)?;
            let rsq = 1.0 - full.rss / full.tss;
            let nn = d1.n_eff();
            let kz = d1.active().len() as f64;
            let adj = 1.0 - (1.0 - rsq) * (nn - 1.0) / (nn - kz);
            let partial = 1.0 - full.rss / restr.rss;
            let q = excluded.len() as f64;
            let (f, df2) = if vce == Vce::Ols || vce == Vce::Oim {
                (((restr.rss - full.rss) / q) / (full.rss / (nn - kz)), nn - kz)
            } else {
                let mut vsub = Matrix::zeros(excluded.len(), excluded.len());
                for (a, &ea) in excluded.iter().enumerate() {
                    for (c, &ec) in excluded.iter().enumerate() {
                        vsub[(a, c)] = full.v[(ea, ec)];
                    }
                }
                let (vi, _) = vsub.sym_inverse(1e-14);
                let bsub: Vec<f64> = excluded.iter().map(|&e| full.b[e]).collect();
                let w: f64 = (0..bsub.len()).map(|a| (0..bsub.len()).map(|c| bsub[a] * vi[(a, c)] * bsub[c]).sum::<f64>()).sum();
                (w / q, full.n_clust.map_or(nn - kz, |g| g as f64 - 1.0))
            };
            let row = [rsq, adj, partial, f, q, df2, dist::ftail(q, df2, f)];
            for (c, v) in row.iter().enumerate() {
                fs[(r, c)] = *v;
            }
            fs_names.push(dx.names[j].clone());
        }
        let fs = fs.with_names(fs_names, ["rsq", "adjrsq", "partrsq", "F", "df1", "df2", "pval"].iter().map(|s| s.to_string()).collect());
        let b = fit2.b.clone();
        let resid: Vec<f64> = (0..dx.n()).map(|t| dx.y[t] - dx.xb(&b, t)).collect();
        let rss: f64 = (0..dx.n()).map(|t| dx.w[t] * resid[t] * resid[t]).sum();
        let small = cmd.has_opt("small", 2);
        let kk = ax.len() as f64;
        let s2 = if small { rss / (n - kk) } else { rss / n };
        let xtx = est::cross(&xhat, &ax);
        let (bread, _) = xtx.sym_inverse(1e-14);
        let vact = match &vce {
            Vce::Ols | Vce::Oim => bread.scale(s2),
            _ => {
                let scores: Vec<Vec<f64>> = (0..dx.n()).map(|t| ax.iter().map(|&j| xhat.cols[j][t] * resid[t] * dx.w[t]).collect()).collect();
                let cl = est::cluster_values(ds, &dx, &vce)?;
                let (v, g) = est::sandwich(&bread, &scores, cl.as_deref(), 1.0);
                match cl {
                    // without `small`, no finite-sample adjustment is applied
                    Some(_) => if small { v.scale(g as f64 / (g as f64 - 1.0) * (n - 1.0) / (n - kk)) } else { v },
                    None => v.scale(if small { n / (n - kk) } else { 1.0 }),
                }
            }
        };
        let v = est::expand_v(&vact, &ax, dx.k());
        let ybar = util::mean(&dx.y);
        let tss: f64 = dx.y.iter().map(|y| (y - ybar).powi(2)).sum();
        let r2 = 1.0 - rss / tss;
        let (wald, q) = est::wald_all(&dx, &b, &v);
        let level = util::level(cmd, st)?;
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", "Instrumental variables 2SLS regression", "Number of obs", g10(n)),
            format!("{:<52}{:<15} = {:>10}", "", if small { format!("F({}, {})", q, n - kk) } else { format!("Wald chi2({})", q) }, f2(if small { wald / q } else { wald })),
            format!("{:<52}{:<15} = {:>10}", "", if small { "Prob > F" } else { "Prob > chi2" }, f4(if small { dist::ftail(q, n - kk, wald / q) } else { dist::chi2tail(q, wald) })),
            format!("{:<52}{:<15} = {:>10}", "", "R-squared", if r2 < 0.0 { ".".to_string() } else { f4(r2) }),
            format!("{:<52}{:<15} = {:>10}", "", "Root MSE", rmse4(s2.sqrt())),
            String::new(),
        ];
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &dx.names, b: &b, v: &v, omitted: &dx.omitted, is_base: &dx.is_base, df: if small { Some(n - kk) } else { None }, level, eform: None, coef_title: "Coefficient" }));
        out.push(format!("Endogenous: {}", endog.trim()));
        out.push(format!("Exogenous:  {}", zlist.split_whitespace().collect::<Vec<_>>().join(" ")));
        let text = out.join("\n");
        est::post(st, "ivregress", &dx, &b, &v, &|e: &mut Results| {
            e.set("r2", r2);
            e.set("rss", rss);
            e.set("rmse", s2.sqrt());
            e.set("chi2", wald);
            e.set("df_m", q);
            if small {
                e.set("df_r", n - kk);
                e.set("F", wald / q);
            }
            e.set_str("instd", endog.trim());
            e.set_str("insts", zlist.trim());
            e.set_str("estimator", "2sls");
            let (vt, vl) = est::vce_label(&vce);
            e.set_str("vce", if vt == "ols" { "unadjusted" } else { vt });
            if !vl.is_empty() {
                e.set_str("vcetype", vl);
            }
            e.set_mat("_firststage", fs.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// glm (IRLS)
// ---------------------------------------------------------------------------

pub struct Glm;

impl CommandHandler for Glm {
    fn name(&self) -> &'static str {
        "glm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let fam_text = cmd.opt("family", 1).unwrap_or("gaussian").to_lowercase();
        let mut fw = fam_text.split_whitespace();
        let family = fw.next().unwrap_or("gaussian").to_string();
        let fam_arg = fw.next().map(String::from);
        let family = match family.as_str() {
            f if "gaussian".starts_with(f) => "gaussian",
            f if "binomial".starts_with(f) => "binomial",
            f if "poisson".starts_with(f) => "poisson",
            f if "gamma".starts_with(f) => "gamma",
            f if "igaussian".starts_with(f) => "igaussian",
            f if "nbinomial".starts_with(f) => "nbinomial",
            f => return Err(rc_err(198, format!("unknown family {f}"))),
        };
        let link = cmd.opt("link", 1).map(|l| l.trim().to_lowercase()).unwrap_or_else(|| match family {
            "gaussian" => "identity".into(),
            "binomial" => "logit".into(),
            "poisson" | "nbinomial" => "log".into(),
            "gamma" => "power -1".into(),
            _ => "power -2".into(),
        });
        let link = match link.as_str() {
            l if l.starts_with('i') => "identity".to_string(),
            l if l.starts_with("logi") => "logit".into(),
            l if l.starts_with("log") => "log".into(),
            l if l.starts_with('p') && l.contains(' ') => link.clone(),
            l if l.starts_with("pr") => "probit".into(),
            l if l.starts_with('c') => "cloglog".into(),
            l if l.starts_with("rec") || l == "power -1" => "power -1".into(),
            _ => link.clone(),
        };
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "faip", extra: vec![] })?;
        let act = d.active();
        let nbin: Vec<f64> = match (&family[..], fam_arg.as_deref()) {
            ("binomial", Some(a)) => match a.parse::<f64>() {
                Ok(v) => vec![v; d.n()],
                Err(_) => {
                    let c = util::col(ds, &crate::varlist::resolve_name(ds, a)?)?;
                    d.rows.iter().map(|&i| c[i]).collect()
                }
            },
            _ => vec![1.0; d.n()],
        };
        let nb_alpha: f64 = if family == "nbinomial" { fam_arg.as_deref().and_then(|a| a.parse().ok()).unwrap_or(1.0) } else { 1.0 };
        let power: f64 = link.strip_prefix("power ").and_then(|p| p.trim().parse().ok()).unwrap_or(1.0);
        let linkf = |mu: f64| -> f64 {
            match link.as_str() {
                "identity" => mu,
                "log" => mu.ln(),
                "logit" => dist::logit(mu),
                "probit" => dist::invnormal(mu),
                "cloglog" => (-(1.0 - mu).ln()).ln(),
                _ => if power == 0.0 { mu.ln() } else { mu.powf(power) },
            }
        };
        let inv = |eta: f64| -> f64 {
            match link.as_str() {
                "identity" => eta,
                "log" => eta.exp(),
                "logit" => dist::invlogit(eta),
                "probit" => dist::normal(eta),
                "cloglog" => 1.0 - (-eta.exp()).exp(),
                _ => if power == 0.0 { eta.exp() } else { eta.powf(1.0 / power) },
            }
        };
        let dmu = |eta: f64| -> f64 {
            match link.as_str() {
                "identity" => 1.0,
                "log" => eta.exp(),
                "logit" => {
                    let p = dist::invlogit(eta);
                    p * (1.0 - p)
                }
                "probit" => dist::normalden(eta),
                "cloglog" => eta.exp() * (-eta.exp()).exp(),
                _ => if power == 0.0 { eta.exp() } else { (1.0 / power) * eta.powf(1.0 / power - 1.0) },
            }
        };
        let var = |mu: f64, m: f64| -> f64 {
            match family {
                "gaussian" => 1.0,
                "binomial" => mu * (1.0 - mu / m),
                "poisson" => mu,
                "gamma" => mu * mu,
                "igaussian" => mu * mu * mu,
                _ => mu + nb_alpha * mu * mu,
            }
        };
        // binomial works with proportions times m
        let y: Vec<f64> = d.y.clone();
        let mut mu: Vec<f64> = (0..d.n()).map(|t| match family {
            "binomial" => (y[t] + 0.5) / (nbin[t] + 1.0) * nbin[t],
            "poisson" | "nbinomial" | "gamma" | "igaussian" => (y[t] + util::mean(&y)) / 2.0,
            _ => y[t],
        }).collect();
        let mu_to_eta = |m: f64, n: f64| if family == "binomial" { linkf(m / n) } else { linkf(m) };
        let mut eta: Vec<f64> = (0..d.n()).map(|t| mu_to_eta(mu[t], nbin[t])).collect();
        let mut b = vec![0.0; d.k()];
        let mut dev_old = f64::INFINITY;
        let mut iters = 0;
        let mut wfinal = vec![0.0; d.n()];
        for _ in 0..100 {
            iters += 1;
            let mut dw = d.clone();
            for t in 0..d.n() {
                let dm = dmu(eta[t]) * if family == "binomial" { nbin[t] } else { 1.0 };
                let v = var(mu[t], nbin[t]).max(1e-300);
                let z = eta[t] + (y[t] - mu[t]) / dm;
                dw.y[t] = z;
                dw.w[t] = d.w[t] * dm * dm / v;
                wfinal[t] = dw.w[t];
            }
            let fit = est::ols(&dw, &Vce::Ols, None)?;
            b = fit.b.clone();
            for t in 0..d.n() {
                eta[t] = d.xb(&b, t);
                mu[t] = inv(eta[t]) * if family == "binomial" { nbin[t] } else { 1.0 };
            }
            let dev = deviance(family, &y, &mu, &nbin, &d.w, nb_alpha);
            if (dev_old - dev).abs() < 1e-10 * (dev.abs() + 1.0) {
                break;
            }
            dev_old = dev;
        }
        let n = d.n_eff();
        let kk = act.len() as f64;
        let dev = deviance(family, &y, &mu, &nbin, &d.w, nb_alpha);
        let pearson: f64 = (0..d.n()).map(|t| d.w[t] * (y[t] - mu[t]).powi(2) / var(mu[t], nbin[t])).sum();
        let phi = match family {
            "gaussian" | "gamma" | "igaussian" => pearson / (n - kk),
            _ => 1.0,
        };
        let mut dw = d.clone();
        dw.w = wfinal.clone();
        let xtwx = est::cross(&dw, &act);
        let (inv_a, _) = xtwx.sym_inverse(1e-14);
        let vact = match &vce {
            Vce::Oim | Vce::Ols => inv_a.scale(phi),
            _ => {
                let scores: Vec<Vec<f64>> = (0..d.n())
                    .map(|t| {
                        let dm = dmu(eta[t]) * if family == "binomial" { nbin[t] } else { 1.0 };
                        let s = d.w[t] * (y[t] - mu[t]) * dm / var(mu[t], nbin[t]);
                        act.iter().map(|&j| s * d.cols[j][t]).collect()
                    })
                    .collect();
                let cl = est::cluster_values(ds, &d, &vce)?;
                let (v, g) = est::sandwich(&inv_a, &scores, cl.as_deref(), 1.0);
                match cl {
                    Some(_) => v.scale(g as f64 / (g as f64 - 1.0)),
                    None => v.scale(n / (n - 1.0)),
                }
            }
        };
        let v = est::expand_v(&vact, &act, d.k());
        let ll = glm_ll(family, &y, &mu, &nbin, &d.w, phi, nb_alpha);
        let level = util::level(cmd, st)?;
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", "Generalized linear models", "Number of obs", g10(n)),
            format!("{:<52}{:<15} = {:>10}", format!("Optimization     : ML (IRLS, {} iterations)", iters), "Residual df", g10(n - kk)),
            format!("{:<52}{:<15} = {:>10}", "", "Scale parameter", g10(phi)),
            format!("Deviance         = {:>15}{:>17}(1/df) Deviance = {:>10}", g10(dev), "", g10(dev / (n - kk))),
            format!("Pearson          = {:>15}{:>17}(1/df) Pearson  = {:>10}", g10(pearson), "", g10(pearson / (n - kk))),
            String::new(),
            format!("Variance function: {:<34}Link function   : {}", family, link),
            format!("{:<52}AIC             = {:>10}", format!("Log likelihood   = {}", fmt_g(ll, 10)), g10((-2.0 * ll + 2.0 * kk) / n)),
            String::new(),
        ];
        let eform = if cmd.has_opt("eform", 2) { Some("exp(b)") } else { None };
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let fam = family.to_string();
        est::post(st, "glm", &d, &b, &v, &|e: &mut Results| {
            e.set("deviance", dev);
            e.set("deviance_p", pearson);
            e.set("phi", phi);
            e.set("ll", ll);
            e.set("df", n - kk);
            e.set("aic", (-2.0 * ll + 2.0 * kk) / n);
            e.set_str("varfunc", fam.clone());
            e.set_str("link", link.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

fn deviance(family: &str, y: &[f64], mu: &[f64], m: &[f64], w: &[f64], alpha: f64) -> f64 {
    let xlogx = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
    (0..y.len())
        .map(|t| {
            let (yt, mt) = (y[t], mu[t]);
            w[t] * match family {
                "gaussian" => (yt - mt).powi(2),
                "binomial" => 2.0 * (xlogx(yt, mt) + xlogx(m[t] - yt, m[t] - mt)),
                "poisson" => 2.0 * (xlogx(yt, mt) - (yt - mt)),
                "gamma" => -2.0 * ((yt / mt).ln() - (yt - mt) / mt),
                "igaussian" => (yt - mt).powi(2) / (mt * mt * yt),
                _ => 2.0 * (xlogx(yt, mt) - (yt + 1.0 / alpha) * ((1.0 + alpha * yt) / (1.0 + alpha * mt)).ln()),
            }
        })
        .sum()
}

fn glm_ll(family: &str, y: &[f64], mu: &[f64], m: &[f64], w: &[f64], phi: f64, alpha: f64) -> f64 {
    (0..y.len())
        .map(|t| {
            let (yt, mt) = (y[t], mu[t]);
            w[t] * match family {
                "gaussian" => -0.5 * ((yt - mt).powi(2) / phi + (2.0 * std::f64::consts::PI * phi).ln()),
                "binomial" => {
                    let p = mt / m[t];
                    dist::lnfactorial(m[t]) - dist::lnfactorial(yt) - dist::lnfactorial(m[t] - yt) + if yt > 0.0 { yt * p.ln() } else { 0.0 } + if m[t] - yt > 0.0 { (m[t] - yt) * (1.0 - p).ln() } else { 0.0 }
                }
                "poisson" => yt * mt.ln() - mt - ln_gamma(yt + 1.0),
                "gamma" => {
                    let a = 1.0 / phi;
                    a * (a * yt / mt).ln() - a * yt / mt - yt.ln() - ln_gamma(a)
                }
                "igaussian" => -0.5 * ((yt - mt).powi(2) / (yt * mt * mt * phi) + (2.0 * std::f64::consts::PI * phi * yt.powi(3)).ln()),
                _ => {
                    let r = 1.0 / alpha;
                    ln_gamma(yt + r) - ln_gamma(r) - ln_gamma(yt + 1.0) + r * (r / (r + mt)).ln() + yt * (mt / (r + mt)).ln()
                }
            }
        })
        .sum()
}

// ---------------------------------------------------------------------------
// qreg
// ---------------------------------------------------------------------------

pub struct Qreg;

impl CommandHandler for Qreg {
    fn name(&self) -> &'static str {
        "qreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = split_dep(&cmd.main)?;
        let dep = resolve_dep(ds, &dep)?;
        let q = util::opt_num(cmd, "quantile", 1)?.unwrap_or(0.5);
        let q = if q >= 1.0 { q / 100.0 } else { q };
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: false, weights: "fia", extra: vec![] })?;
        let act = d.active();
        let n = d.n();
        // IRLS for the check function, then polish by exact interpolation
        let mut b = est::ols(&d, &Vce::Ols, None)?.b;
        let mut eps = 1e-3;
        for _ in 0..500 {
            let mut dw = d.clone();
            for t in 0..n {
                let r = d.y[t] - d.xb(&b, t);
                let wq = if r >= 0.0 { q } else { 1.0 - q };
                dw.w[t] = d.w[t] * wq / r.abs().max(eps);
            }
            let nb = est::ols(&dw, &Vce::Ols, None)?.b;
            let change: f64 = nb.iter().zip(&b).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
            b = nb;
            if change < 1e-10 {
                eps = (eps * 0.1).max(1e-12);
                if eps <= 1e-12 {
                    break;
                }
            }
        }
        let obj = |b: &[f64]| -> f64 {
            (0..n).map(|t| {
                let r = d.y[t] - d.xb(b, t);
                d.w[t] * if r >= 0.0 { q * r } else { (q - 1.0) * r }
            }).sum()
        };
        // polish: solve exactly through the k observations with smallest |residual|
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &c| (d.y[a] - d.xb(&b, a)).abs().total_cmp(&(d.y[c] - d.xb(&b, c)).abs()));
        let kk = act.len();
        if n >= kk {
            let mut sub = d.clone();
            let pick: Vec<usize> = idx[..kk].to_vec();
            sub.y = pick.iter().map(|&t| d.y[t]).collect();
            sub.cols = d.cols.iter().map(|c| pick.iter().map(|&t| c[t]).collect()).collect();
            sub.w = vec![1.0; kk];
            sub.rows = pick.clone();
            if let Ok(fit) = est::ols(&sub, &Vce::Ols, None) {
                if obj(&fit.b) <= obj(&b) + 1e-12 {
                    b = fit.b;
                }
            }
        }
        let sum_adev = obj(&b);
        // raw sum of deviations about the unconditional quantile
        let mut ys = d.y.clone();
        ys.sort_by(f64::total_cmp);
        let yq = util::pctile_sorted(&ys, q * 100.0);
        let sum_rdev: f64 = d.y.iter().map(|&y| { let r = y - yq; if r >= 0.0 { q * r } else { (q - 1.0) * r } }).sum();
        // iid VCE with sparsity estimated by the Hall-Sheather bandwidth
        let resid: Vec<f64> = (0..n).map(|t| d.y[t] - d.xb(&b, t)).collect();
        let nf = n as f64;
        let alpha = 0.05;
        let za = dist::invnormal(1.0 - alpha / 2.0);
        let zq = dist::invnormal(q);
        let h = nf.powf(-1.0 / 3.0) * za.powf(2.0 / 3.0) * ((1.5 * dist::normalden(zq).powi(2)) / (2.0 * zq * zq + 1.0)).powf(1.0 / 3.0);
        let mut rs = resid.clone();
        rs.sort_by(f64::total_cmp);
        let lo = util::pctile_sorted(&rs, ((q - h).max(0.0)) * 100.0);
        let hi = util::pctile_sorted(&rs, ((q + h).min(1.0)) * 100.0);
        let sparsity = (hi - lo) / (2.0 * h);
        let xtx = est::cross(&d, &act);
        let (inv_a, _) = xtx.sym_inverse(1e-14);
        let vact = inv_a.scale(q * (1.0 - q) * sparsity * sparsity);
        let v = est::expand_v(&vact, &act, d.k());
        let level = util::level(cmd, st)?;
        let df_r = nf - kk as f64;
        let title = if (q - 0.5).abs() < 1e-12 { "Median regression".to_string() } else { format!(".{} Quantile regression", (q * 100.0).round()) };
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", title, "Number of obs", g10(nf)),
            format!("  Raw sum of deviations {:>10} (about {})", g10(sum_rdev), g10(yq)),
            format!("  Min sum of deviations {:>10}{:>17}{:<15} = {:>10}", g10(sum_adev), "", "Pseudo R2", f4(1.0 - sum_adev / sum_rdev)),
            String::new(),
        ];
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: Some(df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        est::post(st, "qreg", &d, &b, &v, &|e: &mut Results| {
            e.set("q", q);
            e.set("sum_adev", sum_adev);
            e.set("sum_rdev", sum_rdev);
            e.set("df_r", df_r);
            e.set("f_r", 1.0 / sparsity);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// Width of the label column of the last coefficient table in `out`.
fn tbl_lw(out: &[String]) -> usize {
    out.iter().rev().find(|l| l.contains("+---")).and_then(|l| l.find('+')).map_or(12, |p| p.saturating_sub(1))
}

/// Root MSE as shown in estimation headers (`%10.5g`).
fn rmse4(v: f64) -> String {
    crate::format::fmt(v, "%10.5g").trim().to_string()
}

// ---------------------------------------------------------------------------
// xtivreg, fe
// ---------------------------------------------------------------------------

pub struct Xtivreg {
    /// `xtivreg2`: also report identification and overidentification tests
    pub ivreg2: bool,
}

/// Columns of a design restricted to active variables.
fn active_cols(d: &Design) -> Vec<Vec<f64>> {
    d.active().iter().map(|&j| d.cols[j].clone()).collect()
}

/// Two-step efficient GMM Hansen J statistic (cluster-robust when `cl` given).
/// Returns (J, df, S); with `s_fixed` the given moment covariance is used
/// instead of estimating one (difference-in-Hansen C statistic).
fn hansen_j(x: &[Vec<f64>], z: &[Vec<f64>], y: &[f64], cl: Option<&[f64]>, s_fixed: Option<&Matrix>) -> (f64, f64, Matrix) {
    let n = y.len();
    let (kx, kz) = (x.len(), z.len());
    if kz <= kx {
        return (0.0, 0.0, Matrix::zeros(kz, kz));
    }
    let mat = |cols: &[Vec<f64>]| {
        let mut m = Matrix::zeros(n, cols.len());
        for (j, c) in cols.iter().enumerate() {
            for t in 0..n {
                m[(t, j)] = c[t];
            }
        }
        m
    };
    let (xm, zm, ym) = (mat(x), mat(z), Matrix::col_vector(y));
    let zt = zm.transpose();
    let zz = zt.mul(&zm).unwrap();
    let (zzi, _) = zz.sym_inverse(1e-14);
    let zx = zt.mul(&xm).unwrap();
    let zy = zt.mul(&ym).unwrap();
    let gmm = |w: &Matrix| -> Vec<f64> {
        let a = zx.transpose().mul(w).unwrap();
        let (ai, _) = a.mul(&zx).unwrap().sym_inverse(1e-14);
        ai.mul(&a.mul(&zy).unwrap()).unwrap().data
    };
    let resid = |b: &[f64]| -> Vec<f64> { (0..n).map(|t| y[t] - (0..kx).map(|j| x[j][t] * b[j]).sum::<f64>()).collect() };
    let s_of = |u: &[f64]| -> Matrix {
        let mut s = Matrix::zeros(kz, kz);
        let mut add = |g: &[f64]| {
            for a in 0..kz {
                for c in 0..kz {
                    s[(a, c)] += g[a] * g[c];
                }
            }
        };
        match cl {
            Some(cl) => {
                let mut sums: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
                for t in 0..n {
                    let e = sums.entry(cl[t].to_bits()).or_insert_with(|| vec![0.0; kz]);
                    for a in 0..kz {
                        e[a] += z[a][t] * u[t];
                    }
                }
                sums.values().for_each(|g| add(g));
            }
            None => {
                for t in 0..n {
                    let g: Vec<f64> = (0..kz).map(|a| z[a][t] * u[t]).collect();
                    add(&g);
                }
            }
        }
        s
    };
    let s = match s_fixed {
        Some(s) => s.clone(),
        None => s_of(&resid(&gmm(&zzi))),
    };
    let (si, _) = s.sym_inverse(1e-14);
    let b2 = gmm(&si);
    let u2 = resid(&b2);
    let g: Vec<f64> = (0..kz).map(|a| (0..n).map(|t| z[a][t] * u2[t]).sum()).collect();
    let j: f64 = (0..kz).map(|a| (0..kz).map(|c| g[a] * si[(a, c)] * g[c]).sum::<f64>()).sum();
    (j, (kz - kx) as f64, s)
}

impl CommandHandler for Xtivreg {
    fn name(&self) -> &'static str {
        if self.ivreg2 { "xtivreg2" } else { "xtivreg" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !cmd.has_opt("fe", 2) {
            return Err(rc_err(198, format!("{}: only the fe estimator is supported", self.name())));
        }
        let rest = cmd.main.trim();
        let open = rest.find('(').ok_or_else(|| rc_err(198, "(varlist2 = instlist) required"))?;
        let close = rest.rfind(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
        let inside = &rest[open + 1..close];
        let outside = format!("{} {}", &rest[..open], &rest[close + 1..]);
        let (endog, inst) = inside.split_once('=').ok_or_else(|| rc_err(198, "= required inside parentheses"))?;
        let (dep, exog) = split_dep(&outside)?;
        let dep = resolve_dep(ds, &dep)?;
        let panel = match cmd.opt("i", 1) {
            Some(p) => crate::varlist::resolve_name(ds, p.trim())?,
            None => st.tsset.as_ref().and_then(|(p, _)| p.clone()).ok_or_else(|| rc_err(459, "panel variable not set; use xtset varname"))?,
        };
        let xlist = format!("{} {}", endog.trim(), exog);
        let zlist = format!("{} {}", inst.trim(), exog);
        let all = format!("{} {}", xlist, inst);
        let dall = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &all, noconstant: false, weights: "", extra: vec![panel.clone()] })?;
        let mut sub = cmd.clone();
        sub.if_expr = None;
        sub.in_range = None;
        let mut st2 = st.clone();
        st2.by_mask = Some(dall.mask.clone());
        let mut dx = est::build(ds, &st2, &sub, DesignSpec { depvar: Some(&dep), indep: &xlist, noconstant: false, weights: "", extra: vec![] })?;
        let mut dz = est::build(ds, &st2, &sub, DesignSpec { depvar: Some(&dep), indep: &zlist, noconstant: false, weights: "", extra: vec![] })?;
        let (g, ng) = group_index(ds, &dx, &panel)?;
        let n = dx.n() as f64;
        // within transformation (grand means added back, as Stata does)
        let demean = |d: &mut Design| {
            let (ym, xm) = within(d, &g, ng);
            let ybar = util::mean(&d.y);
            let k = d.k();
            let xbar: Vec<f64> = d.cols.iter().map(|c| util::mean(c)).collect();
            for t in 0..d.n() {
                d.y[t] = d.y[t] - ym[g[t]] + ybar;
                for j in 0..k {
                    if d.names[j] != "_cons" {
                        d.cols[j][t] = d.cols[j][t] - xm[j][g[t]] + xbar[j];
                    }
                }
            }
            (ym, xm)
        };
        let raw_y = dx.y.clone();
        let raw_x = dx.cols.clone();
        let (ym, xm) = demean(&mut dx);
        demean(&mut dz);
        est::drop_collinear(&mut dx);
        est::drop_collinear(&mut dz);
        let ax = dx.active();
        let az = dz.active();
        let zz = est::cross(&dz, &az);
        let (zzi, _) = zz.sym_inverse(1e-14);
        let mut xhat = dx.clone();
        for &j in &ax {
            let mut zx = vec![0.0; az.len()];
            for t in 0..dx.n() {
                for (a, &zj) in az.iter().enumerate() {
                    zx[a] += dz.cols[zj][t] * dx.cols[j][t];
                }
            }
            let coef: Vec<f64> = (0..az.len()).map(|a| (0..az.len()).map(|c| zzi[(a, c)] * zx[c]).sum()).collect();
            for t in 0..dx.n() {
                xhat.cols[j][t] = az.iter().zip(&coef).map(|(&zj, c)| dz.cols[zj][t] * c).sum();
            }
        }
        let b = est::ols(&xhat, &Vce::Ols, None)?.b;
        let resid: Vec<f64> = (0..dx.n()).map(|t| dx.y[t] - dx.xb(&b, t)).collect();
        let rss: f64 = resid.iter().map(|r| r * r).sum();
        let kx = ax.iter().filter(|&&j| dx.names[j] != "_cons").count() as f64;
        let df_r = n - ng as f64 - kx;
        let s2 = rss / df_r;
        let (bread, _) = est::cross(&xhat, &ax).sym_inverse(1e-14);
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        let small = cmd.has_opt("small", 2);
        let clv: Option<Vec<f64>> = match &vce {
            Vce::Cluster(c) => {
                let cc = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
                Some(dx.rows.iter().map(|&i| cc[i]).collect())
            }
            _ => None,
        };
        let mut n_clust = None;
        let vact = match &vce {
            Vce::Ols | Vce::Oim => bread.scale(s2),
            _ => {
                let scores: Vec<Vec<f64>> = (0..dx.n()).map(|t| ax.iter().map(|&j| xhat.cols[j][t] * resid[t]).collect()).collect();
                let (vv, gcount) = est::sandwich(&bread, &scores, clv.as_deref(), 1.0);
                let kk = ax.len() as f64;
                if clv.is_some() {
                    n_clust = Some(gcount);
                    let g = gcount as f64;
                    if small { vv.scale(g / (g - 1.0) * (n - 1.0) / (n - kk)) } else { vv }
                } else if small {
                    vv.scale(n / (n - kk))
                } else {
                    vv
                }
            }
        };
        let v = est::expand_v(&vact, &ax, dx.k());
        let cons = dx.names.iter().position(|nm| nm == "_cons").unwrap();
        // fixed effects, sigma_u, R-squared measures
        let slope = |x: &dyn Fn(usize) -> f64| -> f64 { (0..dx.k()).filter(|&j| j != cons).map(|j| b[j] * x(j)).sum() };
        let ui: Vec<f64> = (0..ng).map(|gi| ym[gi] - b[cons] - slope(&|j| xm[j][gi])).collect();
        let mu = util::mean(&ui);
        let su = (ui.iter().map(|u| (u - mu).powi(2)).sum::<f64>() / (ng as f64 - 1.0)).sqrt();
        let se = s2.sqrt();
        let rho = su * su / (su * su + se * se);
        let corr2 = |a: &[f64], c: &[f64]| {
            let (ma, mc) = (util::mean(a), util::mean(c));
            let sac: f64 = a.iter().zip(c).map(|(x, y)| (x - ma) * (y - mc)).sum();
            let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
            let scc: f64 = c.iter().map(|y| (y - mc).powi(2)).sum();
            (sac / (saa * scc).sqrt(), sac * sac / (saa * scc))
        };
        let xb_raw: Vec<f64> = (0..dx.n()).map(|t| slope(&|j| raw_x[j][t])).collect();
        let wy: Vec<f64> = (0..dx.n()).map(|t| raw_y[t] - ym[g[t]]).collect();
        let wf: Vec<f64> = (0..dx.n()).map(|t| xb_raw[t] - slope(&|j| xm[j][g[t]])).collect();
        let r2_w = {
            let r = 1.0 - wy.iter().zip(&wf).map(|(a, c)| (a - c).powi(2)).sum::<f64>() / wy.iter().map(|a| a * a).sum::<f64>();
            if r < 0.0 { SYSMISS } else { corr2(&wy, &wf).1 }
        };
        let gf: Vec<f64> = (0..ng).map(|gi| slope(&|j| xm[j][gi])).collect();
        let r2_b = corr2(&ym, &gf).1;
        let r2_o = corr2(&raw_y, &xb_raw).1;
        let ui_obs: Vec<f64> = (0..dx.n()).map(|t| ui[g[t]]).collect();
        let corr_u = corr2(&ui_obs, &xb_raw).0;
        let (_, q) = est::wald_all(&dx, &b, &v);
        // The reported model test covers every coefficient including the
        // constant (grand-mean parametrisation); its p-value uses that many
        // degrees of freedom while the label shows the number of slopes.
        let (wald, wald_df) = {
            let act: Vec<usize> = (0..dx.k()).filter(|&j| v[(j, j)] > 0.0).collect();
            let m = act.len();
            let mut vs = Matrix::zeros(m, m);
            for (a, &i) in act.iter().enumerate() {
                for (c, &k) in act.iter().enumerate() {
                    vs[(a, c)] = v[(i, k)];
                }
            }
            let (vi, _) = vs.sym_inverse(1e-14);
            ((0..m).map(|a| (0..m).map(|c| b[act[a]] * vi[(a, c)] * b[act[c]]).sum::<f64>()).sum::<f64>(), m as f64)
        };
        // F test that all u_i = 0: compare with pooled IV on the same regressors
        let tcount: Vec<f64> = (0..ng).map(|gi| g.iter().filter(|&&x| x == gi).count() as f64).collect();
        let level = util::level(cmd, st)?;
        let mut out = dx.notes.clone();
        if !out.is_empty() {
            out.push(String::new());
        }
        out.push(format!("{:<52}{:<15} = {:>10}", "Fixed-effects (within) IV regression", "Number of obs", g10(n)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("Group variable: {panel}"), "Number of groups", ng));
        out.push(String::new());
        out.push(format!("{:<52}{}", "R-squared:", "Obs per group:"));
        let r2s = |v: f64| if v >= SYSMISS { ".".to_string() } else { format!("{:.4}", v) };
        out.push(format!("     Within  = {:<37}{:>15} = {:>10}", r2s(r2_w), "min", g10(tcount.iter().cloned().fold(f64::INFINITY, f64::min))));
        out.push(format!("     Between = {:<37}{:>15} = {:>10.1}", r2s(r2_b), "avg", n / ng as f64));
        out.push(format!("     Overall = {:<37}{:>15} = {:>10}", r2s(r2_o), "max", g10(tcount.iter().cloned().fold(0.0, f64::max))));
        out.push(String::new());
        out.push(format!("{:<52}{:<15} = {:>10}", "", format!("Wald chi2({})", q), f2(wald)));
        out.push(format!("{:<52}{:<15} = {:>10}", format!("corr(u_i, Xb) = {:.4}", corr_u), "Prob > chi2", f4(dist::chi2tail(wald_df, wald))));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &dx.names, b: &b, v: &v, omitted: &dx.omitted, is_base: &dx.is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
        out.pop();
        out.push(format!("{}+{}", "-".repeat(tbl_lw(&out) + 1), "-".repeat(64)));
        out.push(format!("{:>lw$} | {:>10}", "sigma_u", g10(su), lw = tbl_lw(&out)));
        out.push(format!("{:>lw$} | {:>10}", "sigma_e", g10(se), lw = tbl_lw(&out)));
        out.push(format!("{:>lw$} | {:>10}   (fraction of variance due to u_i)", "rho", g10(rho), lw = tbl_lw(&out)));
        out.push("-".repeat(tbl_lw(&out) + 66));
        if let Some(g) = n_clust {
            out.push(format!("(Std. err. adjusted for {} clusters)", g));
        }
        let mut diag: Vec<(String, f64)> = Vec::new();
        if self.ivreg2 {
            let robust = clv.is_some() || !matches!(vce, Vce::Ols | Vce::Oim);
            let included: Vec<usize> = az.iter().copied().filter(|&j| dx.names.contains(&dz.names[j])).collect();
            let excluded: Vec<usize> = az.iter().copied().filter(|&j| !dx.names.contains(&dz.names[j])).collect();
            let endog_cols: Vec<usize> = ax.iter().copied().filter(|&j| !dz.names.contains(&dx.names[j])).collect();
            // partial the included exogenous regressors out of Y1 and Z1
            let mut d2 = dz.clone();
            let keep: Vec<usize> = (0..dz.k()).filter(|j| !included.contains(j)).collect();
            for &j in &keep {
                d2.omitted[j] = true;
            }
            let partial = |col: &[f64]| -> Result<Vec<f64>, String> {
                let mut d = d2.clone();
                d.y = col.to_vec();
                let fit = est::ols(&d, &Vce::Ols, None)?;
                Ok((0..d.n()).map(|t| d.y[t] - d.xb(&fit.b, t)).collect())
            };
            let y1: Vec<Vec<f64>> = endog_cols.iter().map(|&j| partial(&dx.cols[j])).collect::<Result<_, _>>()?;
            let z1: Vec<Vec<f64>> = excluded.iter().map(|&j| partial(&dz.cols[j])).collect::<Result<_, _>>()?;
            // fixed effects are absorbed: N net of the groups, L without the constant
            let n_eff = n - ng as f64;
            let l_inst = az.iter().filter(|&&j| dz.names[j] != "_cons").count() as f64;
            let dof = crate::ivdiag::IdDof { n: n_eff, l: l_inst, clusters: n_clust.map(|g| g as f64), robust };
            let rule = "-".repeat(78);
            out.push(String::new());
            out.push(rule.clone());
            if let Some(id) = crate::ivdiag::identification(&y1, &z1, clv.as_deref(), &dof) {
                let pid = dist::chi2tail(id.df, id.lm);
                let lm_name = if robust { "Kleibergen-Paap rk LM statistic" } else { "Anderson canon. corr. LM statistic" };
                out.push(format!("{:<66}{:>12.3}", format!("Underidentification test ({lm_name}):"), id.lm));
                out.push(format!("{:>66}{:>12.4}", format!("Chi-sq({}) P-val =", id.df), pid));
                out.push(rule.clone());
                out.push(format!("{:<66}{:>12.3}", "Weak identification test (Cragg-Donald Wald F statistic):", id.cd_f));
                if robust {
                    out.push(format!("{:<66}{:>12.3}", "                         (Kleibergen-Paap rk Wald F statistic):", id.kp_f));
                }
                out.push("Critical values for weak identification: see Stock and Yogo (2005);".to_string());
                if robust {
                    out.push("they are tabulated for the Cragg-Donald statistic under i.i.d. errors.".to_string());
                }
                diag.push(("idstat".into(), id.lm));
                diag.push(("iddf".into(), id.df));
                diag.push(("idp".into(), pid));
                diag.push(("cdf".into(), id.cd_f));
                diag.push(("widstat".into(), if robust { id.kp_f } else { id.cd_f }));
                if robust {
                    diag.push(("rkf".into(), id.kp_f));
                }
            }
            out.push(rule.clone());
            let xc = active_cols(&dx);
            let zc = active_cols(&dz);
            // overidentification: Hansen J (robust) or Sargan (i.i.d.); `scale` fixes
            // the moment covariance (S, or sigma² for Sargan) when given
            let overid = |zcols: &[Vec<f64>], scale: Option<&(Matrix, f64)>| -> (f64, f64, (Matrix, f64)) {
                if robust {
                    let (j, df, s) = hansen_j(&xc, zcols, &dx.y, clv.as_deref(), scale.map(|sc| &sc.0));
                    return (j, df, (s, 0.0));
                }
                let df = zcols.len() as f64 - xc.len() as f64;
                let zm = crate::ivdiag::columns(zcols);
                let (zzi, _) = zm.transpose().mul(&zm).unwrap().sym_inverse(1e-14);
                // 2SLS with these instruments
                let xm = crate::ivdiag::columns(&xc);
                let zx = zm.transpose().mul(&xm).unwrap();
                let zy = zm.transpose().mul(&Matrix::col_vector(&dx.y)).unwrap();
                let a = zx.transpose().mul(&zzi).unwrap();
                let (ai, _) = a.mul(&zx).unwrap().sym_inverse(1e-14);
                let bb = ai.mul(&a.mul(&zy).unwrap()).unwrap();
                let u: Vec<f64> = (0..dx.n()).map(|t| dx.y[t] - (0..xc.len()).map(|j| xc[j][t] * bb[(j, 0)]).sum::<f64>()).collect();
                let s2 = scale.map(|sc| sc.1).unwrap_or_else(|| u.iter().map(|e| e * e).sum::<f64>() / n_eff);
                if df <= 0.0 {
                    return (0.0, 0.0, (Matrix::zeros(0, 0), s2));
                }
                let zu = zm.transpose().mul(&Matrix::col_vector(&u)).unwrap();
                let upu = zu.transpose().mul(&zzi).unwrap().mul(&zu).unwrap()[(0, 0)];
                (upu / s2, df, (Matrix::zeros(0, 0), s2))
            };
            let jname = if robust { "Hansen J statistic" } else { "Sargan statistic" };
            let (j, jdf, _) = overid(&zc, None);
            if jdf > 0.0 {
                let jp = dist::chi2tail(jdf, j);
                out.push(format!("{:<66}{:>12.3}", format!("{jname} (overidentification test of all instruments):"), j));
                out.push(format!("{:>66}{:>12.4}", format!("Chi-sq({}) P-val =", jdf), jp));
                diag.push(("jp".into(), jp));
            } else {
                out.push(format!("{:<66}{:>12.3}", format!("{jname} (overidentification test of all instruments):"), 0.0));
                out.push(format!("{:>78}", "(equation exactly identified)"));
            }
            diag.push(("j".into(), j));
            diag.push(("jdf".into(), jdf));
            if let Some(evars) = cmd.opt("endog", 3) {
                // C statistic: J with the variables treated as exogenous minus J as specified
                let wanted: Vec<String> = evars.split_whitespace().map(|v| v.to_lowercase()).collect();
                let test: Vec<usize> = endog_cols.iter().copied().filter(|&jj| wanted.contains(&dx.names[jj].to_lowercase())).collect();
                let mut zc2 = zc.clone();
                for &jj in &test {
                    zc2.push(dx.cols[jj].clone());
                }
                // both statistics use the moment covariance of the larger instrument set
                let (j2, _, sc) = overid(&zc2, None);
                let sc_small = if robust {
                    let k = zc.len();
                    let mut s = Matrix::zeros(k, k);
                    for a in 0..k {
                        for c in 0..k {
                            s[(a, c)] = sc.0[(a, c)];
                        }
                    }
                    (s, 0.0)
                } else {
                    sc
                };
                let (j1, _, _) = overid(&zc, Some(&sc_small));
                let c = (j2 - j1).max(0.0);
                let cdf = test.len() as f64;
                let cp = dist::chi2tail(cdf, c);
                out.push(rule.clone());
                out.push("-endog- option:".to_string());
                out.push(format!("{:<66}{:>12.3}", "Endogeneity test of endogenous regressors:", c));
                out.push(format!("{:>66}{:>12.4}", format!("Chi-sq({}) P-val =", cdf), cp));
                out.push(format!("Regressors tested:    {}", test.iter().map(|&jj| dx.names[jj].clone()).collect::<Vec<_>>().join(" ")));
                diag.push(("estat".into(), c));
                diag.push(("estatdf".into(), cdf));
                diag.push(("estatp".into(), cp));
            }
            out.push(rule.clone());
            out.push(format!("Instrumented:         {}", endog_cols.iter().map(|&j| dx.names[j].clone()).collect::<Vec<_>>().join(" ")));
            out.push(format!("Included instruments: {}", included.iter().filter(|&&j| dz.names[j] != "_cons").map(|&j| dz.names[j].clone()).collect::<Vec<_>>().join(" ")));
            out.push(format!("Excluded instruments: {}", excluded.iter().map(|&j| dz.names[j].clone()).collect::<Vec<_>>().join(" ")));
            out.push(rule);
        }
        if !self.ivreg2 {
            out.push(format!("Endogenous: {}", endog.trim()));
            out.push(format!("Exogenous:  {}", zlist.split_whitespace().collect::<Vec<_>>().join(" ")));
        }
        let text = out.join("\n");
        let cmdname = self.name();
        est::post(st, cmdname, &dx, &b, &v, &|e: &mut Results| {
            for (k, val) in &diag {
                e.set(k, *val);
            }
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            e.set("sigma_u", su);
            e.set("sigma_e", se);
            e.set("rho", rho);
            e.set("chi2", wald);
            e.set("df_m", q);
            e.set("N_g", ng as f64);
            e.set("r2_w", r2_w);
            e.set("r2_b", r2_b);
            e.set("r2_o", r2_o);
            e.set("corr", corr_u);
            e.set_str("model", "fe");
            e.set_str("ivar", panel.clone());
            e.set_str("instd", endog.trim());
            e.set_str("insts", zlist.trim());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
