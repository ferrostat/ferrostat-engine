//! Choice-model estimators over long-format choice data: conditional logit
//! (`cmclogit`), rank-ordered logit (`cmrologit`, `rologit`), mixed logit
//! with simulated likelihood over Halton draws (`cmmixlogit`, panel
//! `cmxtmixlogit`), and multinomial / rank-ordered probit with the GHK
//! simulator (`cmmprobit`, `cmroprobit`).
//!
//! Probit covariance: errors are differenced against the base alternative.
//! `correlation(independent)` with `stddev(homoskedastic)` (the default)
//! keeps unit variances; `stddev(heteroskedastic)` estimates log standard
//! deviations of all but the base and scale alternatives;
//! `correlation(unstructured)` estimates the Cholesky factor of the
//! differenced covariance with its first element fixed (scale).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::data::{self, CmData, Need, UnitLik};
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Likelihood, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Clogit,
    Rologit,
    Mixlogit,
    Mprobit,
    Roprobit,
}

pub struct CmEst {
    pub verb: &'static str,
    pub kind: Kind,
    pub panel: bool,
}

pub const VERBS: [&str; 10] = ["cmclogit", "cmrologit", "rologit", "cmmixlogit", "cmxtmixlogit", "cmmprobit", "cmroprobit", "asmprobit", "asroprobit", "asmixlogit"];

pub fn estimator(verb: &'static str) -> CmEst {
    let (kind, panel) = match verb {
        "cmclogit" => (Kind::Clogit, false),
        "cmrologit" | "rologit" => (Kind::Rologit, false),
        "cmmixlogit" | "asmixlogit" => (Kind::Mixlogit, false),
        "cmxtmixlogit" => (Kind::Mixlogit, true),
        "cmmprobit" | "asmprobit" => (Kind::Mprobit, false),
        _ => (Kind::Roprobit, false),
    };
    CmEst { verb, kind, panel }
}

// ---------------------------------------------------------------------------
// exploded logit stages (conditional and rank-ordered logit)
// ---------------------------------------------------------------------------

pub struct Stage {
    pub chosen: usize,
    pub risk: Vec<usize>,
    pub case: usize,
}

pub struct StageLogit<'a> {
    pub x: &'a [Vec<f64>],
    pub stages: &'a [Stage],
    pub ncase: usize,
}

impl Likelihood for StageLogit<'_> {
    fn k(&self) -> usize {
        self.x.len()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.x.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let mut sc = vec![vec![0.0; k]; if need_scores { self.ncase } else { 0 }];
        for s in self.stages {
            let eta: Vec<f64> = s.risk.iter().map(|&t| self.x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>()).collect();
            let mx = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = eta.iter().map(|e| (e - mx).exp()).sum();
            let p: Vec<f64> = eta.iter().map(|e| (e - mx).exp() / den).collect();
            let ech: f64 = self.x.iter().zip(b).map(|(c, bj)| c[s.chosen] * bj).sum();
            ll += ech - mx - den.ln();
            let xbar: Vec<f64> = (0..k).map(|a| s.risk.iter().zip(&p).map(|(&t, pp)| pp * self.x[a][t]).sum()).collect();
            for a in 0..k {
                let ga = self.x[a][s.chosen] - xbar[a];
                g[a] += ga;
                if need_scores {
                    sc[s.case][a] += ga;
                }
                for c in 0..=a {
                    let v: f64 = s.risk.iter().zip(&p).map(|(&t, pp)| pp * self.x[a][t] * self.x[c][t]).sum::<f64>() - xbar[a] * xbar[c];
                    h[(a, c)] += v;
                    if c != a {
                        h[(c, a)] += v;
                    }
                }
            }
        }
        if !ll.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; k], Matrix::identity(k), sc);
        }
        (ll, g, h, sc)
    }
}

/// Stages of the conditional (one chosen) or rank-ordered logit.
pub fn stages(data: &CmData, kind: Kind, higher: bool) -> Vec<Stage> {
    let mut out = Vec::new();
    for (c, g) in data.cases.iter().enumerate() {
        if kind == Kind::Clogit || kind == Kind::Mixlogit {
            let ch = *g.iter().find(|&&t| data.d.y[t] != 0.0).unwrap();
            out.push(Stage { chosen: ch, risk: g.clone(), case: c });
            continue;
        }
        let (ranked, bottom) = data.ranking(c, higher);
        let mut remaining: Vec<usize> = ranked.iter().chain(bottom.iter()).copied().collect();
        let mut i = 0;
        while i < ranked.len() {
            let v = data.d.y[ranked[i]];
            let mut j = i;
            while j < ranked.len() && data.d.y[ranked[j]] == v {
                j += 1;
            }
            for &t in &ranked[i..j] {
                out.push(Stage { chosen: t, risk: remaining.clone(), case: c });
            }
            remaining.retain(|t| !ranked[i..j].contains(t));
            i = j;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// model specification shared by estimation and prediction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Corr {
    Indep,
    Unstr,
}

pub struct Spec {
    kind: Kind,
    panel: bool,
    higher: bool,
    /// mixed logit: random columns and Cholesky entries (i, j) over them
    rand: Vec<usize>,
    entries: Vec<(usize, usize)>,
    /// probit covariance
    corr: Corr,
    hetero: bool,
    /// position of each alternative in the differenced space (None: base)
    dpos: Vec<Option<usize>>,
    /// alternatives with a free log standard deviation (heteroskedastic)
    free_sd: Vec<usize>,
    scale_alt: usize,
    ndraws: usize,
    draws: Vec<Vec<Vec<f64>>>,
    pub aux_names: Vec<String>,
}

fn all_opts(cmd: &Command, name: &str, min: usize) -> Vec<String> {
    cmd.opts
        .iter()
        .filter(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min.min(name.len()) && name.starts_with(n.as_str())
        })
        .filter_map(|(_, a)| a.clone())
        .collect()
}

fn word_opt(cmd: &Command, name: &str, min: usize) -> Option<String> {
    cmd.opt(name, min).map(|s| s.trim().to_lowercase())
}

impl Spec {
    pub fn new(ds: &DataSet, cmd: &Command, data: &CmData, kind: Kind, panel: bool, verb: &str) -> Result<Spec, String> {
        let higher = match kind {
            Kind::Rologit => !cmd.has_opt("reverse", 3),
            _ => cmd.has_opt("reverse", 3),
        };
        let mut s = Spec { kind, panel, higher, rand: vec![], entries: vec![], corr: Corr::Indep, hetero: false, dpos: vec![], free_sd: vec![], scale_alt: 0, ndraws: 0, draws: vec![], aux_names: vec![] };
        let alt = data.set.alt.clone().unwrap_or_default();
        match kind {
            Kind::Clogit | Kind::Rologit => {}
            Kind::Mixlogit => {
                let specs = all_opts(cmd, "random", 4);
                if specs.is_empty() {
                    return Err(rc_err(198, "option random() required"));
                }
                for sp in specs {
                    let (vl, o) = sp.split_once(',').map(|(a, b)| (a.trim().to_string(), b.trim().to_lowercase())).unwrap_or((sp.trim().to_string(), String::new()));
                    if o.starts_with("logn") || o.starts_with("tri") || o.starts_with("unif") {
                        return Err(rc_err(198, format!("random({sp}): only normal distributions are supported")));
                    }
                    let correlated = o.starts_with("corr");
                    let vars = crate::varlist::expand(ds, &vl)?;
                    let first = s.rand.len();
                    for v in &vars {
                        let want = if data.set.alt.is_some() { format!("{alt}:{v}") } else { v.clone() };
                        let c = data.names[..data.n_as].iter().position(|n| *n == want).ok_or_else(|| rc_err(198, format!("random variable {v} must also be an alternative-specific variable in the model")))?;
                        s.rand.push(c);
                    }
                    for i in first..s.rand.len() {
                        if correlated {
                            for j in first..=i {
                                s.entries.push((i, j));
                            }
                        } else {
                            s.entries.push((i, i));
                        }
                    }
                }
                for &(i, j) in &s.entries {
                    let vi = data.names[s.rand[i]].rsplit(':').next().unwrap().to_string();
                    let vj = data.names[s.rand[j]].rsplit(':').next().unwrap().to_string();
                    s.aux_names.push(if i == j && !s.entries.iter().any(|e| e.0 != e.1) { format!("sd({vi})") } else { format!("l({vi},{vj})") });
                }
                s.ndraws = data::n_draws(cmd, if verb == "cmxtmixlogit" { 200 } else { 100 })?;
                let units = if panel { data.npanel } else { data.cases.len() };
                s.draws = data::halton_draws(units, s.ndraws, s.rand.len(), 15);
            }
            Kind::Mprobit | Kind::Roprobit => {
                if data.set.alt.is_none() {
                    return Err(rc_err(198, "an alternatives variable is required"));
                }
                let j = data.alts.len();
                let mut p = 0;
                for k in 0..j {
                    if k == data.base {
                        s.dpos.push(None);
                    } else {
                        s.dpos.push(Some(p));
                        p += 1;
                    }
                }
                s.corr = match word_opt(cmd, "correlation", 4).as_deref() {
                    None | Some("independent") | Some("ind") | Some("indep") => Corr::Indep,
                    Some(x) if "unstructured".starts_with(x) && x.len() >= 3 => Corr::Unstr,
                    Some(x) => return Err(rc_err(198, format!("correlation({x}) not supported; use independent or unstructured"))),
                };
                s.hetero = match word_opt(cmd, "stddev", 3).as_deref() {
                    None | Some("homoskedastic") | Some("hom") => false,
                    Some(x) if "heteroskedastic".starts_with(x) && x.len() >= 3 => true,
                    Some(x) => return Err(rc_err(198, format!("stddev({x}) not supported"))),
                };
                s.scale_alt = match cmd.opt("scalealternative", 4) {
                    Some(v) => data::find_alt(ds, &data.set, &data.alts, &data.alt_labels, v.trim())?,
                    None => (0..j).find(|&k| k != data.base).unwrap_or(0),
                };
                if s.corr == Corr::Unstr {
                    let dd = j.saturating_sub(1);
                    for a in 0..dd {
                        for b in 0..=a {
                            if a == 0 && b == 0 {
                                continue;
                            }
                            s.aux_names.push(if a == b { format!("lnl{}_{}", a + 1, b + 1) } else { format!("l{}_{}", a + 1, b + 1) });
                        }
                    }
                } else if s.hetero {
                    for k in 0..j {
                        if k != data.base && k != s.scale_alt {
                            s.free_sd.push(k);
                            s.aux_names.push(format!("lnsigma{}", data.alt_labels[k]));
                        }
                    }
                }
                s.ndraws = data::n_draws(cmd, 100)?;
                let maxq = data.cases.iter().map(|g| g.len()).max().unwrap_or(1).saturating_sub(1).max(1);
                s.draws = data::halton_draws(data.cases.len(), s.ndraws, maxq, 15);
            }
        }
        Ok(s)
    }

    /// Differenced covariance (over non-base alternatives) from the
    /// covariance parameters.
    fn omega(&self, data: &CmData, aux: &[f64]) -> Vec<Vec<f64>> {
        let j = data.alts.len();
        let dd = j.saturating_sub(1);
        let mut om = vec![vec![0.0; dd]; dd];
        match self.corr {
            Corr::Indep => {
                for k in 0..j {
                    if let Some(p) = self.dpos[k] {
                        let sd = match self.free_sd.iter().position(|&f| f == k) {
                            Some(i) => aux[i].exp(),
                            None => 1.0,
                        };
                        om[p][p] += sd * sd;
                    }
                }
                for a in 0..dd {
                    for b in 0..dd {
                        om[a][b] += 1.0;
                    }
                }
            }
            Corr::Unstr => {
                let mut l = vec![vec![0.0; dd]; dd];
                let mut i = 0;
                for a in 0..dd {
                    for b in 0..=a {
                        if a == 0 && b == 0 {
                            l[0][0] = 2f64.sqrt();
                            continue;
                        }
                        l[a][b] = if a == b { aux[i].exp() } else { aux[i] };
                        i += 1;
                    }
                }
                for a in 0..dd {
                    for b in 0..dd {
                        om[a][b] = (0..dd).map(|c| l[a][c] * l[b][c]).sum();
                    }
                }
            }
        }
        om
    }

    /// GHK probability that each (winner, loser) pair holds in case c.
    fn pairs_prob(&self, data: &CmData, c: usize, pairs: &[(usize, usize)], beta: &[f64], om: &[Vec<f64>]) -> f64 {
        let q = pairs.len();
        let dd = om.len();
        let mut cm = vec![vec![0.0; dd]; q];
        let mut m = vec![0.0; q];
        for (r, &(w, l)) in pairs.iter().enumerate() {
            if let Some(p) = self.dpos[data.alt_of[l]] {
                cm[r][p] += 1.0;
            }
            if let Some(p) = self.dpos[data.alt_of[w]] {
                cm[r][p] -= 1.0;
            }
            m[r] = data.xb(beta, w) - data.xb(beta, l);
        }
        let mut sm = vec![vec![0.0; q]; q];
        for a in 0..q {
            for b in 0..=a {
                let mut v = 0.0;
                for i in 0..dd {
                    if cm[a][i] == 0.0 {
                        continue;
                    }
                    for j in 0..dd {
                        v += cm[a][i] * om[i][j] * cm[b][j];
                    }
                }
                sm[a][b] = v;
                sm[b][a] = v;
            }
        }
        match data::chol(&sm) {
            Some(l) => data::ghk(&m, &l, &self.draws[c]),
            None => 0.0,
        }
    }

    /// Winner/loser pairs of case c for the probit models.
    fn case_pairs(&self, data: &CmData, c: usize) -> Vec<(usize, usize)> {
        let g = &data.cases[c];
        if self.kind == Kind::Mprobit {
            let ch = *g.iter().find(|&&t| data.d.y[t] != 0.0).unwrap();
            return g.iter().filter(|&&t| t != ch).map(|&t| (ch, t)).collect();
        }
        let (ranked, bottom) = data.ranking(c, self.higher);
        let mut v = Vec::new();
        for w in ranked.windows(2) {
            v.push((w[0], w[1]));
        }
        if let Some(&last) = ranked.last() {
            for &t in &bottom {
                v.push((last, t));
            }
        }
        v
    }

    /// Mixed logit: log likelihood of unit u and its gradient.
    fn mix_unit(&self, data: &CmData, units: &[Vec<usize>], stg: &[Vec<Stage>], u: usize, th: &[f64]) -> (f64, Vec<f64>) {
        let k = data.cols.len();
        let ne = self.entries.len();
        let nr = self.rand.len();
        let np = k + ne;
        let mut lks = Vec::with_capacity(self.ndraws);
        let mut gks = Vec::with_capacity(self.ndraws);
        for dr in &self.draws[u] {
            let nu: Vec<f64> = dr.iter().map(|&h| dist::invnormal(h.clamp(1e-12, 1.0 - 1e-12))).collect();
            let mut delta = vec![0.0; nr];
            for (e, &(i, j)) in self.entries.iter().enumerate() {
                delta[i] += th[k + e] * nu[j];
            }
            let mut lk = 0.0;
            let mut gk = vec![0.0; np];
            for &c in &units[u] {
                for s in &stg[c] {
                    let eta = |t: usize| data.xb(&th[..k], t) + self.rand.iter().zip(&delta).map(|(&rc, dl)| data.cols[rc][t] * dl).sum::<f64>();
                    let e: Vec<f64> = s.risk.iter().map(|&t| eta(t)).collect();
                    let mx = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let den: f64 = e.iter().map(|v| (v - mx).exp()).sum();
                    let p: Vec<f64> = e.iter().map(|v| (v - mx).exp() / den).collect();
                    lk += eta(s.chosen) - mx - den.ln();
                    for a in 0..k {
                        gk[a] += data.cols[a][s.chosen] - s.risk.iter().zip(&p).map(|(&t, pp)| pp * data.cols[a][t]).sum::<f64>();
                    }
                    for (e2, &(i, j)) in self.entries.iter().enumerate() {
                        let rc = self.rand[i];
                        gk[k + e2] += nu[j] * (data.cols[rc][s.chosen] - s.risk.iter().zip(&p).map(|(&t, pp)| pp * data.cols[rc][t]).sum::<f64>());
                    }
                }
            }
            lks.push(lk);
            gks.push(gk);
        }
        let mx = lks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = lks.iter().map(|l| (l - mx).exp()).collect();
        let sw: f64 = w.iter().sum();
        let ll = mx + (sw / self.ndraws as f64).ln();
        let mut g = vec![0.0; np];
        for (wk, gk) in w.iter().zip(&gks) {
            for a in 0..np {
                g[a] += wk * gk[a] / sw;
            }
        }
        (ll, g)
    }

    fn units(&self, data: &CmData) -> Vec<Vec<usize>> {
        if self.panel {
            let mut u = vec![Vec::new(); data.npanel];
            for (c, &p) in data.case_panel.iter().enumerate() {
                u[p].push(c);
            }
            u
        } else {
            (0..data.cases.len()).map(|c| vec![c]).collect()
        }
    }

    /// Predicted probability that each row is chosen (first-ranked).
    pub fn probs(&self, data: &CmData, th: &[f64]) -> Vec<f64> {
        let k = data.cols.len();
        let mut out = vec![SYSMISS; data.n()];
        match self.kind {
            Kind::Clogit | Kind::Rologit => {
                for g in &data.cases {
                    let e: Vec<f64> = g.iter().map(|&t| data.xb(th, t)).collect();
                    let mx = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let den: f64 = e.iter().map(|v| (v - mx).exp()).sum();
                    for (i, &t) in g.iter().enumerate() {
                        out[t] = (e[i] - mx).exp() / den;
                    }
                }
            }
            Kind::Mixlogit => {
                let units = self.units(data);
                for (u, cs) in units.iter().enumerate() {
                    for &c in cs {
                        let g = &data.cases[c];
                        let mut acc = vec![0.0; g.len()];
                        for dr in &self.draws[u] {
                            let nu: Vec<f64> = dr.iter().map(|&h| dist::invnormal(h.clamp(1e-12, 1.0 - 1e-12))).collect();
                            let mut delta = vec![0.0; self.rand.len()];
                            for (e, &(i, j)) in self.entries.iter().enumerate() {
                                delta[i] += th[k + e] * nu[j];
                            }
                            let e: Vec<f64> = g.iter().map(|&t| data.xb(&th[..k], t) + self.rand.iter().zip(&delta).map(|(&rc, dl)| data.cols[rc][t] * dl).sum::<f64>()).collect();
                            let mx = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let den: f64 = e.iter().map(|v| (v - mx).exp()).sum();
                            for i in 0..g.len() {
                                acc[i] += (e[i] - mx).exp() / den;
                            }
                        }
                        for (i, &t) in g.iter().enumerate() {
                            out[t] = acc[i] / self.ndraws as f64;
                        }
                    }
                }
            }
            Kind::Mprobit | Kind::Roprobit => {
                let om = self.omega(data, &th[k..]);
                for (c, g) in data.cases.iter().enumerate() {
                    for &t in g {
                        let pairs: Vec<(usize, usize)> = g.iter().filter(|&&o| o != t).map(|&o| (t, o)).collect();
                        out[t] = self.pairs_prob(data, c, &pairs, &th[..k], &om);
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// estimation
// ---------------------------------------------------------------------------

fn title(kind: Kind, verb: &str) -> &'static str {
    match kind {
        Kind::Clogit => "Conditional logit choice model",
        Kind::Rologit => "Rank-ordered logit choice model",
        Kind::Mixlogit if verb == "cmxtmixlogit" => "Mixed logit choice model (panel data)",
        Kind::Mixlogit => "Mixed logit choice model",
        Kind::Mprobit => "Multinomial probit choice model",
        Kind::Roprobit => "Rank-ordered probit choice model",
    }
}

/// Conditional-logit fit of the model columns (starting values and the
/// comparison likelihood of nested models).
pub fn clogit_fit(data: &CmData) -> Result<est::MlResult, String> {
    let st = stages(data, Kind::Clogit, true);
    let m = StageLogit { x: &data.cols, stages: &st, ncase: data.cases.len() };
    xm::maximize(&m, vec![0.0; data.cols.len()], 300)
}

impl CommandHandler for CmEst {
    fn name(&self) -> &'static str {
        self.verb
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(self.verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        if self.panel {
            let s = data::resolve(ds, cmd)?;
            if s.panel.is_none() {
                return Err(rc_err(459, "panel data not cmset; use cmset panelvar timevar altvar"));
            }
        }
        let need = match self.kind {
            Kind::Rologit | Kind::Roprobit => Need::Rank,
            _ => Need::One,
        };
        let cmd2 = if self.kind == Kind::Mixlogit { with_random(ds, cmd)? } else { cmd.clone() };
        let cmd = &cmd2;
        let data = data::prepare(ds, st, cmd, need)?;
        if data.cols.is_empty() {
            return Err(rc_err(102, "no covariates or alternative-specific constants to estimate"));
        }
        let spec = Spec::new(ds, cmd, &data, self.kind, self.panel, self.verb)?;
        let level = util::level(cmd, st)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let k = data.cols.len();
        let mut names = data.names.clone();
        for a in &spec.aux_names {
            names.push(format!("/:{a}"));
        }
        let by_panel = self.panel;
        let cluster = data::case_cluster(ds, &data, &vce, by_panel)?;
        let neq = data::count_eqs(&data.names);
        let naux = spec.aux_names.len();
        let mut fit = match self.kind {
            Kind::Clogit | Kind::Rologit => {
                let stg = stages(&data, self.kind, spec.higher);
                let m = StageLogit { x: &data.cols, stages: &stg, ncase: data.cases.len() };
                let res = xm::maximize(&m, vec![0.0; k], 300)?;
                data::to_fit(res, &names, neq, naux, &vce, cluster.as_deref())?
            }
            Kind::Mixlogit => {
                let start = clogit_fit(&data)?;
                let units = spec.units(&data);
                let stg: Vec<Vec<Stage>> = {
                    let all = stages(&data, Kind::Clogit, true);
                    let mut v: Vec<Vec<Stage>> = (0..data.cases.len()).map(|_| Vec::new()).collect();
                    for s in all {
                        let c = s.case;
                        v[c].push(s);
                    }
                    v
                };
                let mut b0 = start.b.clone();
                for &(i, j) in &spec.entries {
                    b0.push(if i == j { 0.1 } else { 0.0 });
                }
                let gf = |u: usize, th: &[f64]| spec.mix_unit(&data, &units, &stg, u, th);
                let ff = |u: usize, th: &[f64]| gf(u, th).0;
                let lik = UnitLik { k: names.len(), units: units.len(), f: &ff, grad: Some(&gf) };
                data::fit_units(&lik, b0, &names, neq, naux, &vce, cluster.as_deref())?
            }
            Kind::Mprobit | Kind::Roprobit => {
                let start = clogit_fit(&data)?;
                let mut b0: Vec<f64> = start.b.iter().map(|b| b / 1.28).collect();
                b0.extend(vec![0.0; naux]);
                let pairs: Vec<Vec<(usize, usize)>> = (0..data.cases.len()).map(|c| spec.case_pairs(&data, c)).collect();
                let ff = |c: usize, th: &[f64]| {
                    let om = spec.omega(&data, &th[k..]);
                    let p = spec.pairs_prob(&data, c, &pairs[c], &th[..k], &om);
                    p.max(1e-300).ln()
                };
                let lik = UnitLik { k: names.len(), units: data.cases.len(), f: &ff, grad: None };
                data::fit_units(&lik, b0, &names, neq, naux, &vce, cluster.as_deref())?
            }
        };
        // standard deviations are identified up to sign
        if self.kind == Kind::Mixlogit {
            for (e, &(i, j)) in spec.entries.iter().enumerate() {
                if i == j && fit.b[k + e] < 0.0 {
                    let a = k + e;
                    fit.b[a] = -fit.b[a];
                    fit.bt[a] = -fit.bt[a];
                    let p = fit.names.len();
                    for o in 0..p {
                        if o != a {
                            fit.v[(a, o)] = -fit.v[(a, o)];
                            fit.v[(o, a)] = -fit.v[(o, a)];
                            fit.vt[(a, o)] = -fit.vt[(a, o)];
                            fit.vt[(o, a)] = -fit.vt[(o, a)];
                        }
                    }
                }
            }
        }
        let widx: Vec<usize> = (0..k).filter(|&j| !data.is_cons[j]).collect();
        let (chi2, df_m) = crate::heckman::wald(&fit.b, &fit.v, &widx);
        let ll = fit.ll();
        let robust = vce != Vce::Oim;
        let ttl = title(self.kind, self.verb);
        let mut out = vec![String::new()];
        out.extend(est::iteration_log(&fit.res.iterations, if robust { "Log pseudolikelihood" } else { "Log likelihood" }));
        out.push(String::new());
        let (amin, aavg, amax) = data.alt_sizes();
        let hr = crate::choice_mixed::hrow;
        out.push(hr(ttl, "Number of obs", util::commas(data.n())));
        out.push(hr(&format!("Case ID variable: {}", data.set.case), "Number of cases", util::commas(data.cases.len())));
        if self.panel {
            out.push(hr(&format!("Panel variable: {}", data.set.panel.clone().unwrap_or_default()), "Number of panels", util::commas(data.npanel)));
        }
        out.push(String::new());
        if let Some(a) = &data.set.alt {
            out.push(hr(&format!("Alternatives variable: {a}"), "Alts per case: min", util::commas(amin)));
        } else {
            out.push(hr("", "Alts per case: min", util::commas(amin)));
        }
        out.push(hr("", "               avg", format!("{aavg:.1}")));
        out.push(hr("", "               max", util::commas(amax)));
        if spec.ndraws > 0 {
            out.push(hr("Integration sequence: Halton", "Integration points", util::commas(spec.ndraws)));
        }
        out.push(String::new());
        out.push(hr("", &format!("Wald chi2({})", fmt_g(df_m, 9)), xm::f2(chi2)));
        out.push(hr(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9)), "Prob > chi2", xm::f4(xm::chi2p(df_m, chi2))));
        if let (Some(g), Vce::Cluster(c)) = (fit.n_clust, &vce) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g))));
        }
        out.push(String::new());
        out.extend(xm::table(&data.depvar, &fit, level, if cmd.has_opt("or", 2) { Some("Odds ratio") } else { None }, &[]));
        if let Some(_a) = &data.set.alt {
            if !data.alt_labels.is_empty() {
                out.push(format!("Base alternative: {}", data.alt_labels[data.base]));
            }
        }
        let text = out.join("\n");
        let verb = self.verb;
        let raw = cmd.raw.clone();
        let alts = data.alts.clone();
        let altl = data.alt_labels.clone();
        let base = data.base;
        let ncase = data.cases.len();
        let set = data.set.clone();
        let casevars = data.casevars.join(" ");
        let ndraws = spec.ndraws;
        let npanel = data.npanel;
        let depvar = data.depvar.clone();
        let panel = self.panel;
        xm::post_fit(st, verb, &data.d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df_m);
            e.set("ll", ll);
            e.set("N_case", ncase as f64);
            e.set("alt_min", amin as f64);
            e.set("alt_avg", aavg);
            e.set("alt_max", amax as f64);
            if !alts.is_empty() {
                e.set("k_alt", alts.len() as f64);
                e.set("i_base", (base + 1) as f64);
                e.set_mat("altvals", Matrix::row_vector(&alts));
                e.set_str("alteqs", altl.join(" "));
                e.set_str("altvar", set.alt.clone().unwrap_or_default());
            }
            if ndraws > 0 {
                e.set("n_sim", ndraws as f64);
                e.set_str("intmethod", "halton");
            }
            if panel {
                e.set("N_panel", npanel as f64);
                e.set_str("panelvar", set.panel.clone().unwrap_or_default());
            }
            e.set_str("caseid", set.case.clone());
            e.set_str("case", set.case.clone());
            e.set_str("casevars", casevars.clone());
            e.set_str("depvar", depvar.clone());
            e.set_str("title", ttl);
            e.set_str("predict", "cm_p");
            e.set_str("marginsok", "pr xb default");
            e.set_str("_cmdline", raw.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict / estat
// ---------------------------------------------------------------------------

/// Mixed logit: random-coefficient variables join the alternative-specific
/// regressors when not listed there.
fn with_random(ds: &DataSet, cmd: &Command) -> Result<Command, String> {
    let mut c = cmd.clone();
    let listed: Vec<String> = c.main.split_whitespace().skip(1).filter_map(|w| crate::varlist::expand(ds, w).ok()).flatten().collect();
    for sp in all_opts(cmd, "random", 4) {
        let vl = sp.split(',').next().unwrap_or("").to_string();
        for v in crate::varlist::expand(ds, vl.trim())? {
            if !listed.contains(&v) && !c.main.split_whitespace().skip(1).any(|w| w == v) {
                c.main = format!("{} {v}", c.main.trim_end());
            }
        }
    }
    Ok(c)
}

/// Re-create data and specification of the current estimates.
fn reprepare(ds: &DataSet, st: &Store) -> Result<(CmData, Spec, Vec<f64>), String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let raw = st.e.macros.get("_cmdline").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let c = ferrostat_parser::parser::parse_line(&raw).map_err(|e| rc_err(198, format!("{e:?}")))?;
    let verb = VERBS.iter().find(|v| **v == ecmd).copied().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let m = estimator(verb);
    let need = if matches!(m.kind, Kind::Rologit | Kind::Roprobit) { Need::Rank } else { Need::One };
    let c = if m.kind == Kind::Mixlogit { with_random(ds, &c)? } else { c };
    let data = data::prepare(ds, st, &c, need)?;
    let spec = Spec::new(ds, &c, &data, m.kind, m.panel, verb)?;
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let th: Vec<f64> = (0..b.cols).map(|j| b.get(0, j)).collect();
    Ok((data, spec, th))
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let xb = cmd.has_opt("xb", 2);
    let (data, spec, th) = reprepare(ds, st)?;
    let k = data.cols.len();
    let vals_s: Vec<f64> = if xb { (0..data.n()).map(|t| data.xb(&th[..k], t)).collect() } else { spec.probs(&data, &th) };
    let mask = util::touse(ds, st, cmd)?;
    let mut vals = vec![SYSMISS; ds.n_obs];
    for (t, &r) in data.d.rows.iter().enumerate() {
        if mask[r] {
            vals[r] = vals_s[t];
        }
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !xb && !cmd.has_opt("pr", 2) {
        lines.push("(option pr assumed; Pr(alternative))".to_string());
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

/// `estat alternatives` and, after the probit models, `estat covariance`.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, _cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "alternatives" | "alt" | "alte" | "alter" | "altern" => Some(estat_alternatives(ds, st)),
        "covariance" | "cov" | "cova" | "covar" | "correlation" | "cor" | "corr" => {
            let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
            if !matches!(ecmd.as_str(), "cmmprobit" | "cmroprobit" | "asmprobit" | "asroprobit") {
                return None;
            }
            Some(estat_cov(ds, st, sub.starts_with("cor")))
        }
        _ => None,
    }
}

fn estat_alternatives(ds: &DataSet, st: &mut Store) -> Result<String, String> {
    let (data, _, _) = reprepare(ds, st)?;
    if data.alts.is_empty() {
        return Err(rc_err(321, "no alternatives variable"));
    }
    let mut out = vec![String::new(), "Alternatives summary".to_string(), String::new()];
    out.push(format!("{:>10}  {:<16} {:>12} {:>10} {:>10}", "Index", "Alternative", "Cases", "Chosen", "Percent"));
    out.push("-".repeat(64));
    let mut cases = vec![0usize; data.alts.len()];
    let mut chosen = vec![0usize; data.alts.len()];
    for g in &data.cases {
        let best = g.iter().map(|&t| data.d.y[t]).fold(f64::NEG_INFINITY, f64::max);
        for &t in g {
            cases[data.alt_of[t]] += 1;
            if data.d.y[t] == best && best != 0.0 {
                chosen[data.alt_of[t]] += 1;
            }
        }
    }
    let tot: usize = chosen.iter().sum();
    for k in 0..data.alts.len() {
        let mark = if k == data.base { " (base)" } else { "" };
        out.push(format!("{:>10}  {:<16} {:>12} {:>10} {:>10.2}", k + 1, format!("{}{mark}", data.alt_labels[k]), util::commas(cases[k]), util::commas(chosen[k]), 100.0 * chosen[k] as f64 / tot.max(1) as f64));
    }
    out.push("-".repeat(64));
    Ok(out.join("\n"))
}

fn estat_cov(ds: &DataSet, st: &mut Store, corr: bool) -> Result<String, String> {
    let (data, spec, th) = reprepare(ds, st)?;
    let k = data.cols.len();
    let om = spec.omega(&data, &th[k..]);
    let labs: Vec<String> = (0..data.alts.len()).filter(|&a| a != data.base).map(|a| data.alt_labels[a].clone()).collect();
    let d = om.len();
    let mut m = Matrix::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            m[(i, j)] = if corr { om[i][j] / (om[i][i] * om[j][j]).sqrt() } else { om[i][j] };
        }
    }
    let m = m.with_names(labs.clone(), labs.clone());
    let mut out = vec![String::new(), format!("{} of the error differences (relative to base alternative {})", if corr { "Correlation" } else { "Covariance" }, data.alt_labels[data.base]), String::new()];
    out.push(format!("{:>14} {}", "", labs.iter().map(|l| format!("{:>11}", crate::summarize::abbrev(l, 11))).collect::<Vec<_>>().join("")));
    for i in 0..d {
        out.push(format!("{:>14} {}", crate::summarize::abbrev(&labs[i], 14), (0..=i).map(|j| format!("{:>11}", fmt_g(m[(i, j)], 8))).collect::<Vec<_>>().join("")));
    }
    st.r = Default::default();
    st.r.set_mat(if corr { "correlation" } else { "covariance" }, m);
    Ok(out.join("\n"))
}
