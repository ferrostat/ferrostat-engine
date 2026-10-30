//! `sem`: linear structural equation models fitted by maximum likelihood
//! (`method(ml)`), maximum likelihood with missing values (`method(mlmv)`,
//! full-information likelihood over missing-data patterns) or asymptotic
//! distribution free weighted least squares (`method(adf)`).
//!
//! The model is built from path notation by [`super::model`]; the
//! likelihood of each group uses the sufficient statistics (means and ML
//! covariances), the per-observation contributions only for robust,
//! cluster and OPG variance estimators.

use std::collections::BTreeSet;
use std::rc::Rc;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::la;
use super::model::{self, BuildOpts, Class, Kind, Model, Stats};
use super::opt::{self, Affine, Opts, Tech};
use super::path::{self, opt_all};
use super::state::{self, Fitted};
use super::tab::{self, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub const LN2PI: f64 = 1.837_877_066_409_345_5;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method {
    Ml,
    Mlmv,
    Adf,
}

impl Method {
    pub fn word(&self) -> &'static str {
        match self {
            Method::Ml => "ml",
            Method::Mlmv => "mlmv",
            Method::Adf => "adf",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub idx: Vec<usize>,
    pub n: f64,
    pub mean: Vec<f64>,
    pub cov: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct GroupData {
    pub stats: Stats,
    /// observed values (NaN when missing), one row per observation
    pub rows: Vec<Vec<f64>>,
    /// dataset observation numbers of `rows`
    pub obs: Vec<usize>,
    pub pats: Vec<Pattern>,
    /// ADF weight matrix (inverse asymptotic covariance of the moments)
    pub w: Vec<f64>,
    pub cluster: Vec<f64>,
}

pub struct Problem<'a> {
    pub md: &'a Model,
    pub gd: &'a [GroupData],
    pub method: Method,
}

/// (mean, vech cov) moment vector.
pub fn moments(mu: &[f64], sig: &[f64], p: usize) -> Vec<f64> {
    let mut s = mu.to_vec();
    for i in 0..p {
        for j in 0..=i {
            s.push(sig[i * p + j]);
        }
    }
    s
}

impl<'a> Problem<'a> {
    pub fn ll(&self, th: &[f64]) -> Option<f64> {
        let p = self.md.p;
        let mut total = 0.0;
        for (g, gd) in self.gd.iter().enumerate() {
            let (mu, sig) = self.md.implied(th, g)?;
            match self.method {
                Method::Ml => {
                    let l = la::chol(&sig, p)?;
                    let ld = la::chol_logdet(&l, p);
                    let si = la::chol_inv(&l, p);
                    let s = &gd.stats;
                    let tr: f64 = si.iter().zip(&s.cov).map(|(a, b)| a * b).sum();
                    let d: Vec<f64> = (0..p).map(|i| s.mean[i] - mu[i]).collect();
                    let q = la::chol_solve(&l, p, &d).iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
                    total += -0.5 * s.n * (p as f64 * LN2PI + ld + tr + q);
                }
                Method::Mlmv => {
                    for pt in &gd.pats {
                        let o = &pt.idx;
                        let po = o.len();
                        let so = la::sub(&sig, p, o, o);
                        let l = la::chol(&so, po)?;
                        let ld = la::chol_logdet(&l, po);
                        let si = la::chol_inv(&l, po);
                        let tr: f64 = si.iter().zip(&pt.cov).map(|(a, b)| a * b).sum();
                        let d: Vec<f64> = (0..po).map(|a| pt.mean[a] - mu[o[a]]).collect();
                        let q = la::chol_solve(&l, po, &d).iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
                        total += -0.5 * pt.n * (po as f64 * LN2PI + ld + tr + q);
                    }
                }
                Method::Adf => {
                    la::chol(&sig, p)?;
                    let s = moments(&gd.stats.mean, &gd.stats.cov, p);
                    let m = moments(&mu, &sig, p);
                    let q = s.len();
                    let d: Vec<f64> = (0..q).map(|i| s[i] - m[i]).collect();
                    let wd = la::mul(&gd.w, &d, q, q, 1);
                    total += -0.5 * gd.stats.n * d.iter().zip(&wd).map(|(a, b)| a * b).sum::<f64>();
                }
            }
        }
        Some(total)
    }

    /// Per-observation log likelihood (all groups, in group order).
    pub fn ll_obs(&self, th: &[f64]) -> Option<Vec<f64>> {
        let p = self.md.p;
        let mut out = Vec::new();
        for (g, gd) in self.gd.iter().enumerate() {
            let (mu, sig) = self.md.implied(th, g)?;
            let full = la::chol(&sig, p)?;
            for r in &gd.rows {
                let o: Vec<usize> = (0..p).filter(|&i| !r[i].is_nan()).collect();
                let po = o.len();
                let (l, d) = if po == p {
                    (full.clone(), (0..p).map(|i| r[i] - mu[i]).collect::<Vec<f64>>())
                } else {
                    let so = la::sub(&sig, p, &o, &o);
                    (la::chol(&so, po)?, o.iter().map(|&i| r[i] - mu[i]).collect())
                };
                let ld = la::chol_logdet(&l, po);
                let q = la::chol_solve(&l, po, &d).iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
                out.push(-0.5 * (po as f64 * LN2PI + ld + q));
            }
        }
        Some(out)
    }
}

pub struct SemFit {
    pub md: Model,
    pub th: Vec<f64>,
    /// variance of th (k x k)
    pub v: Matrix,
    pub gd: Vec<GroupData>,
    pub method: Method,
    pub vce: String,
    pub ll: f64,
    pub ll_sat: f64,
    pub ll_bs: f64,
    pub chi2_ms: f64,
    pub df_ms: f64,
    pub chi2_bs: f64,
    pub df_bs: f64,
    /// Satorra-Bentler scaled chi2 (model, baseline)
    pub sb: Option<(f64, f64)>,
    pub n: f64,
    pub aff: Affine,
    pub converged: bool,
    pub level: f64,
    pub kfree: usize,
    pub nm1: bool,
}

impl SemFit {
    pub fn problem(&self) -> Problem<'_> {
        Problem { md: &self.md, gd: &self.gd, method: self.method }
    }
}

/// Build per-group data for the observed variables.
pub fn group_data(cols: &[Vec<f64>], rows_g: &[Vec<usize>], method: Method, clus: Option<&[f64]>) -> Result<Vec<GroupData>, String> {
    let p = cols.len();
    let mut out = Vec::new();
    for rows in rows_g {
        let mut gd = GroupData::default();
        for &r in rows {
            let v: Vec<f64> = cols.iter().map(|c| if is_missing(c[r]) { f64::NAN } else { c[r] }).collect();
            gd.rows.push(v);
            gd.obs.push(r);
            if let Some(c) = clus {
                gd.cluster.push(c[r]);
            }
        }
        let n = gd.rows.len() as f64;
        if n < 2.0 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // available-case moments
        let mut mean = vec![0.0; p];
        let mut cnt = vec![0.0f64; p];
        for r in &gd.rows {
            for i in 0..p {
                if !r[i].is_nan() {
                    mean[i] += r[i];
                    cnt[i] += 1.0;
                }
            }
        }
        for i in 0..p {
            mean[i] /= cnt[i].max(1.0);
        }
        let mut cov = vec![0.0; p * p];
        let mut cc = vec![0.0f64; p * p];
        for r in &gd.rows {
            for i in 0..p {
                for j in 0..p {
                    if !r[i].is_nan() && !r[j].is_nan() {
                        cov[i * p + j] += (r[i] - mean[i]) * (r[j] - mean[j]);
                        cc[i * p + j] += 1.0;
                    }
                }
            }
        }
        for i in 0..p * p {
            cov[i] /= cc[i].max(1.0);
        }
        gd.stats = Stats { n, mean, cov };
        if method == Method::Mlmv {
            let mut pats: Vec<Pattern> = Vec::new();
            let mut members: Vec<Vec<usize>> = Vec::new();
            for (t, r) in gd.rows.iter().enumerate() {
                let idx: Vec<usize> = (0..p).filter(|&i| !r[i].is_nan()).collect();
                if idx.is_empty() {
                    continue;
                }
                match pats.iter().position(|x| x.idx == idx) {
                    Some(k) => members[k].push(t),
                    None => {
                        pats.push(Pattern { idx, n: 0.0, mean: vec![], cov: vec![] });
                        members.push(vec![t]);
                    }
                }
            }
            for (pt, mem) in pats.iter_mut().zip(&members) {
                let po = pt.idx.len();
                let nn = mem.len() as f64;
                let mut m = vec![0.0; po];
                for &t in mem {
                    for a in 0..po {
                        m[a] += gd.rows[t][pt.idx[a]] / nn;
                    }
                }
                let mut c = vec![0.0; po * po];
                for &t in mem {
                    for a in 0..po {
                        for b in 0..po {
                            c[a * po + b] += (gd.rows[t][pt.idx[a]] - m[a]) * (gd.rows[t][pt.idx[b]] - m[b]) / nn;
                        }
                    }
                }
                pt.n = nn;
                pt.mean = m;
                pt.cov = c;
            }
            gd.pats = pats;
        }
        if method == Method::Adf {
            let q = p + p * (p + 1) / 2;
            let s = &gd.stats;
            let ds: Vec<Vec<f64>> = gd
                .rows
                .iter()
                .map(|r| {
                    let mut d = r.clone();
                    for i in 0..p {
                        for j in 0..=i {
                            d.push((r[i] - s.mean[i]) * (r[j] - s.mean[j]));
                        }
                    }
                    d
                })
                .collect();
            let mut dm = vec![0.0; q];
            for d in &ds {
                for i in 0..q {
                    dm[i] += d[i] / n;
                }
            }
            let mut gam = vec![0.0; q * q];
            for d in &ds {
                for i in 0..q {
                    for j in 0..q {
                        gam[i * q + j] += (d[i] - dm[i]) * (d[j] - dm[j]) / n;
                    }
                }
            }
            gd.w = la::inv(&gam, q).ok_or_else(|| rc_err(506, "ADF weight matrix is singular"))?;
        }
        out.push(gd);
    }
    Ok(out)
}

/// Parsed estimation request shared by sem and its postestimation refits.
pub struct Request {
    pub spec: path::Spec,
    pub bo_covs: Vec<String>,
    pub bo_vars: Vec<String>,
    pub bo_means: Vec<String>,
    pub covstruct: Vec<String>,
    pub reliability: Vec<(String, f64)>,
    pub invariant: BTreeSet<Class>,
}

fn parse_invariant(cmd: &Command, ng: usize) -> Result<BTreeSet<Class>, String> {
    let mut inv: BTreeSet<Class> = [Class::MCoef, Class::MCons].into_iter().collect();
    if let Some(t) = cmd.opt("ginvariant", 2) {
        inv.clear();
        for w in t.split_whitespace() {
            let w = w.to_lowercase();
            match w.as_str() {
                "none" => inv.clear(),
                "all" => inv.extend(Class::all()),
                _ => {
                    inv.insert(Class::from_word(&w).ok_or_else(|| rc_err(198, format!("ginvariant({w}) invalid")))?);
                }
            }
        }
    }
    if ng <= 1 {
        inv.clear();
    }
    Ok(inv)
}

pub struct Sem;

impl CommandHandler for Sem {
    fn name(&self) -> &'static str {
        "sem"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            let f = state::sem(st).ok_or_else(|| rc_err(301, "last estimates not found"))?;
            return Ok(display(&f, st, cmd, cmd.has_opt("standardized", 3)).join("\n"));
        }
        let f = fit_cmd(ds, st, cmd)?;
        let f = Rc::new(f);
        post(st, &f, cmd)?;
        let mut out = Vec::new();
        if !cmd.has_opt("nodescribe", 5) && !cmd.has_opt("nodescribe", 5) {
            out.extend(describe(&f.md));
        }
        if !cmd.has_opt("nolog", 4) {
            out.push(String::new());
        }
        out.extend(display(&f, st, cmd, cmd.has_opt("standardized", 3)));
        state::put(&mut st.e, Fitted::Sem(f));
        Ok(out.join("\n"))
    }
}

/// Variables of the model in observed / latent, endogenous / exogenous groups.
pub fn describe(md: &Model) -> Vec<String> {
    let mut out = Vec::new();
    let cats: [(usize, &str, Box<dyn Fn(usize) -> bool>); 5] = [
        (0, "Measurement:", Box::new(|i: usize| md.meas[i])),
        (0, "Structural:", Box::new(|i: usize| i < md.p && md.endo[i] && !md.meas[i])),
        (0, "Latent:", Box::new(|i: usize| i >= md.p && md.endo[i])),
        (1, "Observed:", Box::new(|i: usize| i < md.p && !md.endo[i])),
        (1, "Latent:", Box::new(|i: usize| i >= md.p && !md.endo[i])),
    ];
    let mut shown = [false, false];
    for (k, t, f) in cats.iter() {
        let v: Vec<String> = (0..md.m()).filter(|&i| f(i)).map(|i| md.vars[i].clone()).collect();
        if v.is_empty() {
            continue;
        }
        if !shown[*k] {
            shown[*k] = true;
            out.push(String::new());
            out.push(if *k == 0 { "Endogenous variables".into() } else { "Exogenous variables".to_string() });
        }
        out.push(String::new());
        out.push(t.to_string());
        out.push(format!("  {}", v.join(" ")));
    }
    out
}

pub fn fit_cmd(ds: &DataSet, st: &Store, cmd: &Command) -> Result<SemFit, String> {
    let latent_opt: Vec<String> = opt_all(&cmd.opts, "latent", 3).iter().flat_map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>()).collect();
    let caps = !cmd.has_opt("nocapslatent", 6);
    let spec = path::parse(ds, &cmd.main, &latent_opt, caps)?;
    let method = match cmd.opt("method", 4).map(|s| s.trim().to_lowercase()) {
        None => Method::Ml,
        Some(m) if m == "ml" => Method::Ml,
        Some(m) if m == "mlmv" => Method::Mlmv,
        Some(m) if m == "adf" => Method::Adf,
        Some(m) => return Err(rc_err(198, format!("method({m}) not allowed"))),
    };
    let obs: Vec<String> = spec.order.iter().filter(|v| !spec.is_latent(v)).cloned().collect();
    if obs.is_empty() {
        return Err(rc_err(198, "model has no observed variables"));
    }
    if let Some(s) = super::ssd::stats(ds, &obs)? {
        return fit_ssd(st, cmd, spec, s);
    }
    let mut cols = Vec::new();
    for v in &obs {
        cols.push(util::col(ds, v)?);
    }
    let mut mask = util::touse(ds, st, cmd)?;
    if method == Method::Mlmv {
        for i in 0..ds.n_obs {
            if mask[i] && cols.iter().all(|c| is_missing(c[i])) {
                mask[i] = false;
            }
        }
    } else {
        util::markout(ds, &mut mask, &obs);
    }
    let (gvar, glevels, rows_g) = groups_of(ds, cmd, &mut mask)?;
    let vce = vce_word(cmd)?;
    let clus = if let Some(c) = vce.strip_prefix("cluster ") {
        let cv = crate::varlist::resolve_name(ds, c.trim())?;
        util::markout(ds, &mut mask, &[cv.clone()]);
        Some(util::col(ds, &cv)?)
    } else {
        None
    };
    let rows_g: Vec<Vec<usize>> = rows_g.into_iter().map(|r| r.into_iter().filter(|&i| mask[i]).collect()).collect();
    let gd = group_data(&cols, &rows_g, method, clus.as_deref())?;
    estimate(st, cmd, spec, gd, method, gvar, glevels, vce)
}

fn fit_ssd(st: &Store, cmd: &Command, spec: path::Spec, s: Vec<(String, Stats)>) -> Result<SemFit, String> {
    let method = Method::Ml;
    let mut gd = Vec::new();
    let mut labs = Vec::new();
    let sel: Option<Vec<String>> = cmd.opt("select", 3).map(|t| t.split_whitespace().map(String::from).collect());
    for (l, stt) in s {
        if sel.as_ref().is_some_and(|v| !v.contains(&l)) {
            continue;
        }
        labs.push(l);
        gd.push(GroupData { stats: stt, ..Default::default() });
    }
    if gd.len() > 1 && cmd.opt("group", 2).is_none() {
        // pool the groups: total moments
        let p = gd[0].stats.mean.len();
        let n: f64 = gd.iter().map(|g| g.stats.n).sum();
        let mean: Vec<f64> = (0..p).map(|i| gd.iter().map(|g| g.stats.n * g.stats.mean[i]).sum::<f64>() / n).collect();
        let cov: Vec<f64> = (0..p * p)
            .map(|t| {
                let (i, j) = (t / p, t % p);
                gd.iter().map(|g| g.stats.n * (g.stats.cov[t] + (g.stats.mean[i] - mean[i]) * (g.stats.mean[j] - mean[j]))).sum::<f64>() / n
            })
            .collect();
        gd = vec![GroupData { stats: Stats { n, mean, cov }, ..Default::default() }];
        labs = vec!["1".into()];
    }
    let gvar = if gd.len() > 1 { super::ssd::group_var(st).unwrap_or_else(|| "group".into()) } else { String::new() };
    estimate(st, cmd, spec, gd, method, gvar, labs, "oim".into())
}

fn vce_word(cmd: &Command) -> Result<String, String> {
    let v = cmd.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "oim".into());
    let w = v.split_whitespace().next().unwrap_or("oim").to_string();
    Ok(match w.as_str() {
        "oim" | "eim" => "oim".into(),
        "opg" => "opg".into(),
        "sbentler" | "sb" => "sbentler".into(),
        r if r.len() >= 1 && "robust".starts_with(r) => "robust".into(),
        c if c.len() >= 2 && "cluster".starts_with(c) => format!("cluster {}", v.split_whitespace().nth(1).ok_or_else(|| rc_err(198, "cluster variable required"))?),
        o => return Err(rc_err(198, format!("vce({o}) not allowed"))),
    })
}

/// Group variable and per-group rows from `group()`.
pub fn groups_of(ds: &DataSet, cmd: &Command, mask: &mut [bool]) -> Result<(String, Vec<String>, Vec<Vec<usize>>), String> {
    match cmd.opt("group", 2) {
        None => Ok((String::new(), vec!["1".into()], vec![(0..ds.n_obs).filter(|&i| mask[i]).collect()])),
        Some(g) => {
            let gv = crate::varlist::resolve_name(ds, g.trim())?;
            util::markout(ds, mask, &[gv.clone()]);
            let c = util::col(ds, &gv)?;
            let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| c[i]).collect();
            lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            lv.dedup();
            let rows = lv.iter().map(|l| (0..ds.n_obs).filter(|&i| mask[i] && c[i] == *l).collect()).collect();
            Ok((gv, lv.iter().map(|v| crate::format::fmt_g(*v, 12)).collect(), rows))
        }
    }
}

/// Linear constraints from `constraints()` mapped to the parameter vector.
pub fn constraint_rows(st: &Store, cmd: &Command, md: &Model) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let mut r = Vec::new();
    let mut c = Vec::new();
    let Some(list) = cmd.opt("constraints", 5) else { return Ok((r, c)) };
    let names: Vec<String> = (0..md.slots.len()).map(|i| md.bname(i)).collect();
    // also accept names without the group suffix
    let short: Vec<String> = names.iter().map(|n| n.split('#').next().unwrap_or(n).to_string()).collect();
    for w in list.split_whitespace() {
        let ids: Vec<usize> = if let Some((a, b)) = w.split_once('-') {
            let (a, b): (usize, usize) = (a.parse().map_err(|_| rc_err(198, "invalid numlist"))?, b.parse().map_err(|_| rc_err(198, "invalid numlist"))?);
            (a..=b).collect()
        } else {
            vec![w.parse().map_err(|_| rc_err(198, "invalid numlist"))?]
        };
        for id in ids {
            let spec = st.settings.get(&format!("_constraint.{id}")).ok_or_else(|| rc_err(111, format!("constraint {id} not found")))?;
            let spec = normalize_cns(spec);
            let (row, cons) = crate::xmodels::parse_constraint(&spec, &names).or_else(|_| crate::xmodels::parse_constraint(&spec, &short))?;
            let mut rt = vec![0.0; md.k];
            let mut ct = cons;
            for (j, s) in md.slots.iter().enumerate() {
                if row[j] == 0.0 {
                    continue;
                }
                match (s.fix, s.p) {
                    (Some(v), _) => ct -= row[j] * v,
                    (None, Some(p)) => rt[p] += row[j],
                    _ => {}
                }
            }
            r.push(rt);
            c.push(ct);
        }
    }
    Ok((r, c))
}

/// `[y]x` and `_b[y:x]` both become `y:x`; `[/]var(e.y)` stays usable.
fn normalize_cns(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('[') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let Some(j) = after.find(']') else { break };
        let eq = &after[..j];
        let tail = &after[j + 1..];
        let name_end = tail.find(|c: char| c.is_whitespace() || "+-=*".contains(c)).unwrap_or(tail.len());
        let nm = &tail[..name_end];
        if nm.is_empty() {
            out.push_str(eq);
        } else {
            out.push_str(&format!("{eq}:{nm}"));
        }
        rest = &tail[name_end..];
        if out.ends_with("_b") {
            out.truncate(out.len() - 2);
        }
    }
    out.push_str(rest);
    out
}

#[allow(clippy::too_many_arguments)]
pub fn estimate(st: &Store, cmd: &Command, spec: path::Spec, gd: Vec<GroupData>, method: Method, gvar: String, glabels: Vec<String>, vce: String) -> Result<SemFit, String> {
    let ng = gd.len();
    let stats: Vec<Stats> = gd.iter().map(|g| g.stats.clone()).collect();
    let rel: Vec<(String, f64)> = opt_all(&cmd.opts, "reliability", 3)
        .iter()
        .flat_map(|t| {
            let w: Vec<&str> = t.split_whitespace().collect();
            w.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].to_string(), c[1].parse::<f64>().unwrap_or(1.0))).collect::<Vec<_>>()
        })
        .collect();
    let bo = BuildOpts {
        covs: opt_all(&cmd.opts, "covariance", 3),
        vars: opt_all(&cmd.opts, "variance", 3),
        means: opt_all(&cmd.opts, "means", 5),
        covstruct: opt_all(&cmd.opts, "covstructure", 6),
        reliability: rel,
        invariant: parse_invariant(cmd, ng)?,
        stats: &stats,
        glabels,
        gvar,
    };
    let md = model::build(&spec, &bo)?;
    let (r, c) = constraint_rows(st, cmd, &md)?;
    let aff = Affine::new(&r, &c, md.k)?;
    let mut start = md.start.clone();
    if let Some(fm) = cmd.opt("from", 4) {
        apply_from(st, fm, &md, &mut start)?;
    }
    let prob = Problem { md: &md, gd: &gd, method };
    let f = |g: &[f64]| prob.ll(&aff.full(g));
    let maxit = util::opt_num(cmd, "iterate", 4)?.map(|v| v as usize).unwrap_or(300);
    let label = if method == Method::Adf { "discrepancy" } else { "log likelihood" };
    let mut g0 = aff.reduce(&start);
    if !f(&g0).is_some_and(|v| v.is_finite()) {
        // fall back on unit variances and zero paths
        let mut s2 = start.clone();
        for sl in &md.slots {
            if let Some(p) = sl.p {
                if sl.kind == Kind::Var {
                    s2[p] = s2[p].abs().max(0.1);
                } else if sl.kind == Kind::Cov && md.endo[sl.a] {
                    s2[p] = 0.0;
                }
            }
        }
        g0 = aff.reduce(&s2);
    }
    let fit = opt::maximize(&f, g0, &Opts { maxiter: maxit, tech: Tech::Newton, label, tol: 1e-10 })?;
    let th = aff.full(&fit.th);
    let kfree = aff.n.cols;
    // variance
    let hinv = || -> Result<Matrix, String> {
        let h = opt::neg_hessian(&f, &fit.th).ok_or_else(|| rc_err(430, "Hessian could not be computed"))?;
        Ok(h.sym_inverse(1e-12).0)
    };
    let scores = || -> Result<Vec<Vec<f64>>, String> {
        if gd.iter().all(|g| g.rows.is_empty()) {
            return Err(rc_err(198, "vce() requires raw data"));
        }
        let fo = |g: &[f64]| prob.ll_obs(&aff.full(g));
        let n0 = fo(&fit.th).ok_or_else(|| rc_err(430, "scores could not be computed"))?.len();
        let mut s = vec![vec![0.0; kfree]; n0];
        for j in 0..kfree {
            let h = 1e-6 * fit.th[j].abs().max(1.0);
            let mut a = fit.th.clone();
            a[j] += h;
            let mut b = fit.th.clone();
            b[j] -= h;
            let (la_, lb) = (fo(&a).ok_or_else(|| rc_err(430, "scores"))?, fo(&b).ok_or_else(|| rc_err(430, "scores"))?);
            for t in 0..n0 {
                s[t][j] = (la_[t] - lb[t]) / (2.0 * h);
            }
        }
        Ok(s)
    };
    let opg = |s: &[Vec<f64>]| {
        let mut m = Matrix::zeros(kfree, kfree);
        for r in s {
            for a in 0..kfree {
                for b in 0..kfree {
                    m[(a, b)] += r[a] * r[b];
                }
            }
        }
        m
    };
    let vg = match vce.as_str() {
        "oim" => hinv()?,
        "opg" => opg(&scores()?).sym_inverse(1e-12).0,
        "robust" | "sbentler" => {
            let hi = hinv()?;
            hi.mul(&opg(&scores()?))?.mul(&hi)?
        }
        v if v.starts_with("cluster") => {
            let hi = hinv()?;
            let s = scores()?;
            let cl: Vec<f64> = gd.iter().flat_map(|g| g.cluster.clone()).collect();
            let mut sums: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
            for (t, r) in s.iter().enumerate() {
                let e = sums.entry(cl[t].to_bits()).or_insert_with(|| vec![0.0; kfree]);
                for a in 0..kfree {
                    e[a] += r[a];
                }
            }
            let nc = sums.len() as f64;
            let b = opg(&sums.into_values().collect::<Vec<_>>()).scale(nc / (nc - 1.0).max(1.0));
            hi.mul(&b)?.mul(&hi)?
        }
        _ => hinv()?,
    };
    let v = aff.expand_v(&vg);
    let n: f64 = gd.iter().map(|g| g.stats.n).sum();
    let p = md.p;
    let moments_per = (p * (p + 3) / 2) as f64;
    let df_ms = moments_per * ng as f64 - kfree as f64;
    // saturated and baseline
    let (ll_sat, ll_bs, df_bs) = sat_base(&md, &gd, method)?;
    let ll = fit.ll;
    let (chi2_ms, chi2_bs) = match method {
        Method::Adf => (-2.0 * ll, -2.0 * ll_bs),
        _ => (2.0 * (ll_sat - ll), 2.0 * (ll_sat - ll_bs)),
    };
    let mut sf = SemFit {
        md,
        th,
        v,
        gd,
        method,
        vce,
        ll,
        ll_sat,
        ll_bs,
        chi2_ms: chi2_ms.max(0.0),
        df_ms,
        chi2_bs: chi2_bs.max(0.0),
        df_bs,
        sb: None,
        n,
        aff,
        converged: fit.converged,
        level: util::level(cmd, st)?,
        kfree,
        nm1: cmd.has_opt("nm1", 3),
    };
    if sf.vce == "sbentler" {
        if method != Method::Ml {
            return Err(rc_err(198, "vce(sbentler) requires method(ml)"));
        }
        let c = sb_scale(&sf, &sf.th, sf.df_ms);
        let cb = sb_base_scale(&sf);
        sf.sb = Some((sf.chi2_ms / c, sf.chi2_bs / cb));
    }
    if !cmd.has_opt("nolog", 4) && !cmd.has_opt("nolog", 4) {
        let mut log = vec![String::new(), "Fitting target model:".to_string(), String::new()];
        log.extend(fit.log);
        ITERLOG.with(|l| *l.borrow_mut() = log);
    } else {
        ITERLOG.with(|l| l.borrow_mut().clear());
    }
    Ok(sf)
}

thread_local! {
    pub static ITERLOG: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn apply_from(st: &Store, fm: &str, md: &Model, start: &mut [f64]) -> Result<(), String> {
    let name = fm.split(',').next().unwrap_or("").trim();
    let m = st.matrices.get(name).ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
    let vals: Vec<f64> = m.data.clone();
    let names: Vec<String> = (0..md.slots.len()).map(|i| md.bname(i)).collect();
    let by_name = m.colnames.iter().any(|c| names.contains(c));
    for (j, s) in md.slots.iter().enumerate() {
        let Some(p) = s.p else { continue };
        let v = if by_name {
            m.colnames.iter().position(|c| *c == names[j]).map(|c| vals[c])
        } else {
            vals.get(j).copied()
        };
        if let Some(v) = v {
            start[p] = v;
        }
    }
    Ok(())
}

/// Saturated and baseline log likelihoods and baseline df.
pub fn sat_base(md: &Model, gd: &[GroupData], method: Method) -> Result<(f64, f64, f64), String> {
    let p = md.p;
    let en: Vec<usize> = (0..p).filter(|&i| md.endo[i]).collect();
    let ex: Vec<usize> = (0..p).filter(|&i| !md.endo[i]).collect();
    let (pe, px) = (en.len() as f64, ex.len() as f64);
    let df_bs = gd.len() as f64 * (pe * (pe - 1.0) / 2.0 + pe * px);
    if method == Method::Ml {
        let mut sat = 0.0;
        let mut bs = 0.0;
        for g in gd {
            let s = &g.stats;
            let ldet = la::det(&s.cov, p).max(1e-300).ln();
            sat += -0.5 * s.n * (p as f64 * LN2PI + ldet + p as f64);
            let sx = la::sub(&s.cov, p, &ex, &ex);
            let lx = if ex.is_empty() { 0.0 } else { la::det(&sx, ex.len()).max(1e-300).ln() };
            let le: f64 = en.iter().map(|&i| s.cov[i * p + i].max(1e-300).ln()).sum();
            bs += -0.5 * s.n * (p as f64 * LN2PI + lx + le + p as f64);
        }
        return Ok((sat, bs, df_bs));
    }
    // generic fits of the saturated (all covariances free) and baseline models
    let fit_aux = |base: bool| -> Result<f64, String> {
        let mut aux = Model {
            vars: md.vars[..p].to_vec(),
            p,
            endo: vec![false; p],
            meas: vec![false; p],
            ng: gd.len(),
            slots: Vec::new(),
            k: 0,
            start: Vec::new(),
            glabels: md.glabels.clone(),
            gvar: md.gvar.clone(),
            labels: vec![],
        };
        let mut k = 0;
        let mut start = Vec::new();
        for (g, gdd) in gd.iter().enumerate() {
            let s = &gdd.stats;
            for i in 0..p {
                aux.slots.push(model::Slot { kind: Kind::Mean, a: i, b: i, g, fix: None, p: Some(k), class: Class::MeanEx, auto: false });
                start.push(s.mean[i]);
                k += 1;
                aux.slots.push(model::Slot { kind: Kind::Var, a: i, b: i, g, fix: None, p: Some(k), class: Class::CovEx, auto: false });
                start.push(s.cov[i * p + i]);
                k += 1;
            }
            for i in 0..p {
                for j in i + 1..p {
                    if base && (md.endo[i] || md.endo[j]) {
                        continue;
                    }
                    aux.slots.push(model::Slot { kind: Kind::Cov, a: i, b: j, g, fix: None, p: Some(k), class: Class::CovEx, auto: false });
                    start.push(if base { s.cov[i * p + j] } else { s.cov[i * p + j] * 0.9 });
                    k += 1;
                }
            }
        }
        aux.k = k;
        let pr = Problem { md: &aux, gd, method };
        let f = |t: &[f64]| pr.ll(t);
        let r = opt::maximize(&f, start, &Opts { maxiter: 200, tech: Tech::Newton, label: "ll", tol: 1e-10 })?;
        Ok(r.ll)
    };
    let sat = if method == Method::Adf { 0.0 } else { fit_aux(false)? };
    let bs = fit_aux(true)?;
    Ok((sat, bs, df_bs))
}

/// Satorra-Bentler scaling factor c = tr(U Gamma) / df for the model.
pub fn sb_scale(sf: &SemFit, th: &[f64], df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    let md = &sf.md;
    let p = md.p;
    let q = p + p * (p + 1) / 2;
    let mut tr = 0.0;
    let ntot = sf.n;
    // Delta: d moments / d gamma
    let mom = |g: usize, t: &[f64]| -> Vec<f64> {
        match md.implied(&sf.aff.full(t), g) {
            Some((mu, sig)) => moments(&mu, &sig, p),
            None => vec![f64::NAN; q],
        }
    };
    let gam0 = sf.aff.reduce(th);
    let ng = sf.gd.len();
    // stacked over groups with weights n_g / n
    let mut blocks = Vec::new();
    for g in 0..ng {
        let dl = opt::jacobian(&|t: &[f64]| mom(g, t), &gam0);
        let (mu, sig) = md.implied(th, g).unwrap();
        let _ = mu;
        let w = nt_weight(&sig, p);
        let gamma = adf_gamma(&sf.gd[g], p);
        blocks.push((dl, w, gamma, sf.gd[g].stats.n / ntot));
    }
    let k = gam0.len();
    // information = sum_g w_g Delta' W Delta
    let mut info = Matrix::zeros(k, k);
    for (dl, w, _, wg) in &blocks {
        let wd = mat(w, q, q).mul(dl).unwrap();
        let m = dl.transpose().mul(&wd).unwrap().scale(*wg);
        info = info.add(&m).unwrap();
    }
    let (ii, _) = info.sym_inverse(1e-12);
    // U_g = W_g - W_g D_g (info)^-1 D_h' W_h (block form); tr(U Gamma)
    for (g, (dl, w, gamma, wg)) in blocks.iter().enumerate() {
        let wm = mat(w, q, q);
        let gm = mat(gamma, q, q);
        // tr(W Gamma) * wg... with cross-group projection only diagonal blocks matter for the trace
        let t1 = wm.mul(&gm).unwrap().trace();
        let wd = wm.mul(dl).unwrap();
        let proj = wd.mul(&ii).unwrap().mul(&wd.transpose()).unwrap().scale(*wg);
        let t2 = proj.mul(&gm).unwrap().trace();
        tr += t1 - t2;
        let _ = g;
    }
    let c = tr / df;
    if c.is_finite() && c > 0.0 { c } else { 1.0 }
}

fn sb_base_scale(sf: &SemFit) -> f64 {
    // baseline: independence among endogenous variables; approximate with
    // the model-free trace of W Gamma restricted to the off-diagonal moments
    let p = sf.md.p;
    let mut tr = 0.0;
    let mut cnt = 0.0;
    for g in &sf.gd {
        let s = &g.stats;
        let w = nt_weight(&s.cov, p);
        let gamma = adf_gamma(g, p);
        let q = p + p * (p + 1) / 2;
        let mut idx = Vec::new();
        let mut k = p;
        for i in 0..p {
            for j in 0..=i {
                if i != j && (sf.md.endo[i] || sf.md.endo[j]) {
                    idx.push(k);
                }
                k += 1;
            }
        }
        let wm = mat(&w, q, q);
        let gm = mat(&gamma, q, q);
        let prod = wm.mul(&gm).unwrap();
        for &i in &idx {
            tr += prod[(i, i)];
            cnt += 1.0;
        }
    }
    if cnt > 0.0 && tr > 0.0 { tr / cnt } else { 1.0 }
}

fn mat(v: &[f64], r: usize, c: usize) -> Matrix {
    let mut m = Matrix::zeros(r, c);
    m.data.copy_from_slice(&v[..r * c]);
    m
}

/// Normal-theory weight matrix of (mean, vech cov).
pub fn nt_weight(sig: &[f64], p: usize) -> Vec<f64> {
    let q = p + p * (p + 1) / 2;
    let mut g = vec![0.0; q * q];
    for i in 0..p {
        for j in 0..p {
            g[i * q + j] = sig[i * p + j];
        }
    }
    let pairs: Vec<(usize, usize)> = (0..p).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    for (a, &(i, j)) in pairs.iter().enumerate() {
        for (b, &(k, l)) in pairs.iter().enumerate() {
            g[(p + a) * q + p + b] = sig[i * p + k] * sig[j * p + l] + sig[i * p + l] * sig[j * p + k];
        }
    }
    la::inv(&g, q).unwrap_or(vec![0.0; q * q])
}

/// Asymptotic covariance of the sample moments (fourth moments).
pub fn adf_gamma(gd: &GroupData, p: usize) -> Vec<f64> {
    let q = p + p * (p + 1) / 2;
    let s = &gd.stats;
    let n = gd.rows.len() as f64;
    let mut gam = vec![0.0; q * q];
    if n == 0.0 {
        return gam;
    }
    let mut dm = vec![0.0; q];
    let ds: Vec<Vec<f64>> = gd
        .rows
        .iter()
        .map(|r| {
            let mut d = r.clone();
            for i in 0..p {
                for j in 0..=i {
                    d.push((r[i] - s.mean[i]) * (r[j] - s.mean[j]));
                }
            }
            d
        })
        .collect();
    for d in &ds {
        for i in 0..q {
            dm[i] += d[i] / n;
        }
    }
    for d in &ds {
        for i in 0..q {
            for j in 0..q {
                gam[i * q + j] += (d[i] - dm[i]) * (d[j] - dm[j]) / n;
            }
        }
    }
    gam
}

/// Equations of the model: (dependent variable, slots of its paths and intercept), per group.
pub fn equations(md: &Model, g: usize) -> Vec<(usize, Vec<usize>)> {
    let mut eqs: Vec<(usize, Vec<usize>)> = Vec::new();
    for (j, s) in md.slots.iter().enumerate() {
        if s.g != g || !(s.kind == Kind::Path || s.kind == Kind::Cons) {
            continue;
        }
        match eqs.iter_mut().find(|(v, _)| *v == s.a) {
            Some((_, l)) => l.push(j),
            None => eqs.push((s.a, vec![j])),
        }
    }
    // intercept last
    for (_, l) in eqs.iter_mut() {
        l.sort_by_key(|&j| md.slots[j].kind == Kind::Cons);
    }
    eqs
}

/// Standardized slot values (all slots) at th.
pub fn std_values(md: &Model, th: &[f64]) -> Vec<f64> {
    let m = md.m();
    let mut out = Vec::with_capacity(md.slots.len());
    let imp: Vec<Option<(Vec<f64>, Vec<f64>)>> = (0..md.ng).map(|g| md.implied_all(th, g)).collect();
    for s in &md.slots {
        let Some((_, sig)) = &imp[s.g] else {
            out.push(f64::NAN);
            continue;
        };
        let sd = |i: usize| sig[i * m + i].max(0.0).sqrt();
        let v = md.val(s, th);
        let errvar = |i: usize| md.slots.iter().find(|x| x.g == s.g && x.kind == Kind::Var && x.a == i).map(|x| md.val(x, th)).unwrap_or(0.0);
        out.push(match s.kind {
            Kind::Path => v * sd(s.b) / sd(s.a),
            Kind::Cons | Kind::Mean => v / sd(s.a),
            Kind::Var => v / sig[s.a * m + s.a],
            Kind::Cov => {
                if md.endo[s.a] {
                    v / (errvar(s.a) * errvar(s.b)).sqrt()
                } else {
                    v / (sd(s.a) * sd(s.b))
                }
            }
        });
    }
    out
}

/// Standardized e(b) and its variance.
pub fn std_bv(f: &SemFit) -> (Vec<f64>, Matrix) {
    let b = std_values(&f.md, &f.th);
    let free: Vec<usize> = (0..f.md.k).collect();
    let _ = free;
    let j = opt::jacobian(&|t: &[f64]| std_values(&f.md, t), &f.th);
    let v = opt::sandwich_jv(&j, &f.v);
    (b, v)
}

pub fn display(f: &SemFit, st: &Store, cmd: &Command, standardized: bool) -> Vec<String> {
    let mut out = ITERLOG.with(|l| std::mem::take(&mut *l.borrow_mut()));
    let md = &f.md;
    let mut stats = vec![("Number of obs".to_string(), tab::n_str(f.n))];
    if md.ng > 1 {
        stats.push(("Number of groups".into(), md.ng.to_string()));
    }
    out.push(String::new());
    out.extend(tab::header("Structural equation model", &stats));
    if md.ng > 1 {
        out.push(format!("Grouping variable: {}", md.gvar));
    }
    out.push(format!("Estimation method: {}", f.method.word()));
    if f.method != Method::Adf {
        out.push(format!("Log likelihood = {}", crate::format::fmt_g(f.ll, 10)));
    } else {
        out.push(format!("Discrepancy    = {}", crate::format::fmt_g(-2.0 * f.ll / f.n, 10)));
    }
    if !cmd.has_opt("nocnsreport", 5) {
        let cns: Vec<String> = md
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == Kind::Path && s.fix.is_some())
            .map(|(i, s)| {
                let (eq, nm) = md.slot_name(s);
                let gs = if md.ng > 1 { format!("#{}.{}", md.glabels[s.g], md.gvar) } else { String::new() };
                let _ = i;
                format!("[{eq}]{nm}{gs} = {}", crate::format::fmt_g(s.fix.unwrap(), 8))
            })
            .collect();
        if !cns.is_empty() {
            out.push(String::new());
            for (i, c) in cns.iter().enumerate() {
                out.push(format!(" ({:>2})  {c}", i + 1));
            }
        }
    }
    out.push(String::new());
    let (b, v) = if standardized { std_bv(f) } else { (md.bvec(&f.th), md.bv(&f.v)) };
    let vt = match f.vce.as_str() {
        "oim" => "OIM",
        "opg" => "OPG",
        "robust" => "Robust",
        "sbentler" => "Satorra-Bentler",
        _ => "Robust",
    };
    let mut rows = Vec::new();
    for g in 0..md.ng {
        if md.ng > 1 {
            rows.push(Row::Head(format!("Group {}", md.glabels[g])));
        }
        let eqs = equations(md, g);
        for (sec, want_meas) in [("Structural", false), ("Measurement", true)] {
            let list: Vec<&(usize, Vec<usize>)> = eqs.iter().filter(|(v, _)| md.meas[*v] == want_meas).collect();
            if list.is_empty() {
                continue;
            }
            rows.push(Row::Head(sec.to_string()));
            for (y, sl) in list {
                rows.push(Row::Head(format!("  {} <-", md.vars[*y])));
                for &j in sl {
                    let s = &md.slots[j];
                    let lab = if s.kind == Kind::Cons { "_cons".to_string() } else { md.vars[s.b].clone() };
                    if s.kind == Kind::Cons && s.fix == Some(0.0) && (s.auto || sl.len() > 1) && md.is_latent(s.a) {
                        continue;
                    }
                    if s.kind == Kind::Cons && s.fix == Some(0.0) && !md.is_latent(s.a) {
                        continue;
                    }
                    rows.push(coef_row(lab, b[j], v[(j, j)], if standardized { v[(j, j)] <= 1e-18 } else { s.p.is_none() }, true));
                }
            }
            rows.push(Row::Rule);
        }
        for (j, s) in md.slots.iter().enumerate() {
            if s.g != g || matches!(s.kind, Kind::Path | Kind::Cons) {
                continue;
            }
            if s.kind == Kind::Mean && s.fix == Some(0.0) && s.auto {
                continue;
            }
            let (_, nm) = md.slot_name(s);
            let test = matches!(s.kind, Kind::Mean | Kind::Cov);
            rows.push(coef_row(nm, b[j], v[(j, j)], if standardized { v[(j, j)] <= 1e-18 } else { s.p.is_none() }, test));
        }
        if g + 1 < md.ng {
            rows.push(Row::Rule);
        }
    }
    if matches!(rows.last(), Some(Row::Rule)) {
        rows.pop();
    }
    let level = util::level(cmd, st).unwrap_or(f.level);
    out.extend(tab::table(vt, level, if standardized { "Standardized" } else { "Coefficient" }, &rows));
    if f.method != Method::Adf || true {
        if f.df_ms > 0.0 {
            let (c2, lab) = match f.sb {
                Some((c, _)) => (c, "Satorra-Bentler scaled test"),
                None => (f.chi2_ms, "LR test of model vs. saturated"),
            };
            out.push(format!("{lab}: chi2({}) = {:>8.2}, Prob > chi2 = {:.4}", f.df_ms, c2, dist::chi2tail(f.df_ms, c2)));
        } else if f.df_ms == 0.0 {
            out.push(format!("LR test of model vs. saturated: chi2(0) = {:>8.2}, Prob > chi2 =      .", 0.0));
        }
    }
    if !f.converged {
        out.push("Warning: convergence not achieved".into());
    }
    out
}

fn coef_row(label: String, b: f64, v: f64, cns: bool, test: bool) -> Row {
    if cns {
        Row::Coef(label, b, None, false)
    } else {
        Row::Coef(label, b, Some(v.max(0.0).sqrt()), test)
    }
}

pub fn post(st: &mut Store, f: &SemFit, _cmd: &Command) -> Result<(), String> {
    let md = &f.md;
    let names: Vec<String> = (0..md.slots.len()).map(|i| md.bname(i)).collect();
    let b = md.bvec(&f.th);
    let v = md.bv(&f.v);
    let mut sample = vec![false; 0];
    let nobs = f.gd.iter().flat_map(|g| g.obs.iter()).copied().max().map(|m| m + 1).unwrap_or(0);
    if nobs > 0 {
        sample = vec![false; nobs];
        for g in &f.gd {
            for &o in &g.obs {
                sample[o] = true;
            }
        }
    }
    let e: &mut Results = st.begin_e();
    e.set_str("cmd", "sem");
    e.set_str("cmdline", format!("sem {}", _cmd.raw.split_once(char::is_whitespace).map_or("", |x| x.1)).trim().to_string());
    e.set_str("title", "Structural equation model");
    e.set_str("method", f.method.word());
    e.set_str("vce", f.vce.split_whitespace().next().unwrap_or("oim"));
    e.set_str("vcetype", match f.vce.as_str() { "oim" => "OIM", "opg" => "OPG", "sbentler" => "Satorra-Bentler", _ => "Robust" });
    e.set_str("properties", "b V");
    e.set_str("estat_cmd", "sem_estat");
    e.set_str("predict", "sem_p");
    let list = |f: &dyn Fn(usize) -> bool| (0..md.m()).filter(|&i| f(i)).map(|i| md.vars[i].clone()).collect::<Vec<_>>().join(" ");
    e.set_str("oyvars", list(&|i| i < md.p && md.endo[i]));
    e.set_str("oxvars", list(&|i| i < md.p && !md.endo[i]));
    e.set_str("lyvars", list(&|i| i >= md.p && md.endo[i]));
    e.set_str("lxvars", list(&|i| i >= md.p && !md.endo[i]));
    e.set_str("depvar", list(&|i| i < md.p && md.endo[i]));
    if md.ng > 1 {
        e.set_str("groupvar", md.gvar.clone());
    }
    e.set("N", f.n);
    e.set("N_groups", md.ng as f64);
    if f.method != Method::Adf {
        e.set("ll", f.ll);
    }
    e.set("k", md.slots.len() as f64);
    e.set("rank", f.kfree as f64);
    e.set("df_m", f.kfree as f64);
    e.set("chi2_ms", f.chi2_ms);
    e.set("df_ms", f.df_ms);
    e.set("p_ms", if f.df_ms > 0.0 { dist::chi2tail(f.df_ms, f.chi2_ms) } else { ferrostat_core::missing::SYSMISS });
    e.set("chi2_bs", f.chi2_bs);
    e.set("df_bs", f.df_bs);
    e.set("p_bs", if f.df_bs > 0.0 { dist::chi2tail(f.df_bs, f.chi2_bs) } else { ferrostat_core::missing::SYSMISS });
    if let Some((a, bb)) = f.sb {
        e.set("chi2sb_ms", a);
        e.set("chi2sb_bs", bb);
    }
    e.set("converged", if f.converged { 1.0 } else { 0.0 });
    e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", v.with_names(names.clone(), names));
    if !sample.is_empty() {
        e.sample = Some(sample);
    }
    Ok(())
}
