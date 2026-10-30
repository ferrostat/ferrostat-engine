//! Linear structural equation models in reticular-action form.
//!
//! Variables are ordered observed first (p of them), then latent (q).
//! For group g and parameters th:
//!   A[i][j]  coefficient of the path j -> i
//!   S        variances/covariances of exogenous variables and of the
//!            errors of endogenous variables
//!   m        means of exogenous variables, intercepts of endogenous ones
//! and the implied moments are  mu = (I-A)^-1 m,  Sigma = (I-A)^-1 S (I-A)^-T.

use std::collections::{BTreeSet, HashMap};

use super::la;
use super::path::{cov_list, Coef, Spec};
use crate::eval::rc_err;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Path,
    Cons,
    Mean,
    Var,
    Cov,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    MCoef,
    MCons,
    SCoef,
    SCons,
    MErrVar,
    SErrVar,
    MErrCov,
    SErrCov,
    SmErrCov,
    MeanEx,
    CovEx,
}

impl Class {
    pub fn from_word(w: &str) -> Option<Class> {
        Some(match w {
            "mcoef" => Class::MCoef,
            "mcons" => Class::MCons,
            "scoef" => Class::SCoef,
            "scons" => Class::SCons,
            "merrvar" => Class::MErrVar,
            "serrvar" => Class::SErrVar,
            "merrcov" => Class::MErrCov,
            "serrcov" => Class::SErrCov,
            "smerrcov" => Class::SmErrCov,
            "meanex" => Class::MeanEx,
            "covex" => Class::CovEx,
            _ => return None,
        })
    }
    pub fn all() -> Vec<Class> {
        use Class::*;
        vec![MCoef, MCons, SCoef, SCons, MErrVar, SErrVar, MErrCov, SErrCov, SmErrCov, MeanEx, CovEx]
    }
    pub fn word(&self) -> &'static str {
        match self {
            Class::MCoef => "mcoef",
            Class::MCons => "mcons",
            Class::SCoef => "scoef",
            Class::SCons => "scons",
            Class::MErrVar => "merrvar",
            Class::SErrVar => "serrvar",
            Class::MErrCov => "merrcov",
            Class::SErrCov => "serrcov",
            Class::SmErrCov => "smerrcov",
            Class::MeanEx => "meanex",
            Class::CovEx => "covex",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Slot {
    pub kind: Kind,
    /// Path: to; Cons/Mean/Var: variable; Cov: first
    pub a: usize,
    /// Path: from; Cov: second; otherwise = a
    pub b: usize,
    pub g: usize,
    pub fix: Option<f64>,
    pub p: Option<usize>,
    pub class: Class,
    /// fixed by the normalization rules rather than by the user
    pub auto: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub n: f64,
    pub mean: Vec<f64>,
    /// ML (divide by n) covariance, p x p
    pub cov: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub vars: Vec<String>,
    pub p: usize,
    pub endo: Vec<bool>,
    /// observed endogenous variables with a latent predictor
    pub meas: Vec<bool>,
    pub ng: usize,
    pub slots: Vec<Slot>,
    pub k: usize,
    pub start: Vec<f64>,
    /// group labels ("1", "2", ...) and the group variable
    pub glabels: Vec<String>,
    pub gvar: String,
    /// parameter labels (for estat scoretests): label -> parameter
    pub labels: Vec<(String, usize)>,
}

pub struct BuildOpts<'a> {
    pub covs: Vec<String>,
    pub vars: Vec<String>,
    pub means: Vec<String>,
    pub covstruct: Vec<String>,
    pub reliability: Vec<(String, f64)>,
    pub invariant: BTreeSet<Class>,
    pub stats: &'a [Stats],
    pub glabels: Vec<String>,
    pub gvar: String,
}

impl PartialOrd for Class {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Class {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*o as u8))
    }
}

impl Model {
    pub fn m(&self) -> usize {
        self.vars.len()
    }
    pub fn idx(&self, n: &str) -> Option<usize> {
        self.vars.iter().position(|v| v == n)
    }
    pub fn is_latent(&self, i: usize) -> bool {
        i >= self.p
    }
    pub fn val(&self, s: &Slot, th: &[f64]) -> f64 {
        match (s.fix, s.p) {
            (Some(v), _) => v,
            (None, Some(p)) => th[p],
            _ => 0.0,
        }
    }
    /// (A, S, m) of group g, each m x m (m vector).
    pub fn mats(&self, th: &[f64], g: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.m();
        let mut a = vec![0.0; m * m];
        let mut s = vec![0.0; m * m];
        let mut mv = vec![0.0; m];
        for sl in self.slots.iter().filter(|s| s.g == g) {
            let v = self.val(sl, th);
            match sl.kind {
                Kind::Path => a[sl.a * m + sl.b] = v,
                Kind::Cons | Kind::Mean => mv[sl.a] = v,
                Kind::Var => s[sl.a * m + sl.a] = v,
                Kind::Cov => {
                    s[sl.a * m + sl.b] = v;
                    s[sl.b * m + sl.a] = v;
                }
            }
        }
        (a, s, mv)
    }
    /// (I - A)^-1
    pub fn total(&self, a: &[f64]) -> Option<Vec<f64>> {
        let m = self.m();
        let mut ia = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                ia[i * m + j] = if i == j { 1.0 } else { 0.0 } - a[i * m + j];
            }
        }
        la::inv(&ia, m)
    }
    /// Implied means and covariances of all variables.
    pub fn implied_all(&self, th: &[f64], g: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        let m = self.m();
        let (a, s, mv) = self.mats(th, g);
        let b = self.total(&a)?;
        let mu = la::mul(&b, &mv, m, m, 1);
        let bs = la::mul(&b, &s, m, m, m);
        let sig = la::mul(&bs, &la::transpose(&b, m, m), m, m, m);
        Some((mu, sig))
    }
    /// Implied moments of the observed variables.
    pub fn implied(&self, th: &[f64], g: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        let (mu, sig) = self.implied_all(th, g)?;
        let m = self.m();
        let p = self.p;
        let idx: Vec<usize> = (0..p).collect();
        Some((mu[..p].to_vec(), la::sub(&sig, m, &idx, &idx)))
    }
    pub fn free(&self) -> Vec<usize> {
        (0..self.k).collect()
    }
    /// Display name of a slot (without group).
    pub fn slot_name(&self, s: &Slot) -> (String, String) {
        let v = |i: usize| self.vars[i].clone();
        let ev = |i: usize| if self.endo[i] { format!("e.{}", self.vars[i]) } else { self.vars[i].clone() };
        match s.kind {
            Kind::Path => (v(s.a), v(s.b)),
            Kind::Cons => (v(s.a), "_cons".into()),
            Kind::Mean => ("/".into(), format!("mean({})", v(s.a))),
            Kind::Var => ("/".into(), format!("var({})", ev(s.a))),
            Kind::Cov => ("/".into(), format!("cov({},{})", ev(s.a), ev(s.b))),
        }
    }
    /// e(b) column name of slot i.
    pub fn bname(&self, i: usize) -> String {
        let s = &self.slots[i];
        let (eq, nm) = self.slot_name(s);
        if self.ng > 1 {
            format!("{eq}:{nm}#{}.{}", self.glabels[s.g], self.gvar)
        } else {
            format!("{eq}:{nm}")
        }
    }
    /// Slots displayed in e(b): all of them.
    pub fn bvec(&self, th: &[f64]) -> Vec<f64> {
        self.slots.iter().map(|s| self.val(s, th)).collect()
    }
    /// Variance of e(b) from the variance of th.
    pub fn bv(&self, v: &crate::matrix::Matrix) -> crate::matrix::Matrix {
        let n = self.slots.len();
        let mut out = crate::matrix::Matrix::zeros(n, n);
        for (i, a) in self.slots.iter().enumerate() {
            for (j, b) in self.slots.iter().enumerate() {
                if let (None, Some(pa), None, Some(pb)) = (a.fix, a.p, b.fix, b.p) {
                    out[(i, j)] = v[(pa, pb)];
                }
            }
        }
        out
    }
}

fn err_name(n: &str) -> Option<&str> {
    n.strip_prefix("e.")
}

/// Build the model from the parsed paths.
pub fn build(sp: &Spec, o: &BuildOpts) -> Result<Model, String> {
    // observed first, latent after, in order of appearance
    let mut vars: Vec<String> = sp.order.iter().filter(|v| !sp.is_latent(v)).cloned().collect();
    let p = vars.len();
    vars.extend(sp.order.iter().filter(|v| sp.is_latent(v)).cloned());
    let m = vars.len();
    if p == 0 {
        return Err(rc_err(198, "no observed variables in the model"));
    }
    let idx = |n: &str| vars.iter().position(|v| v == n);
    let mut endo = vec![false; m];
    let mut haslat = vec![false; m];
    for ps in &sp.paths {
        let t = idx(&ps.to).unwrap();
        let f = idx(&ps.from).unwrap();
        if t == f {
            return Err(rc_err(198, format!("path from {} to itself", ps.to)));
        }
        endo[t] = true;
        if f >= p {
            haslat[t] = true;
        }
    }
    let meas: Vec<bool> = (0..m).map(|i| i < p && endo[i] && haslat[i]).collect();
    let ng = o.stats.len().max(1);
    // user variance/covariance/mean constraints
    let resolve = |n: &str| -> Result<(usize, bool), String> {
        let (base, err) = match err_name(n) {
            Some(b) => (b, true),
            None => (n, false),
        };
        let i = idx(base).ok_or_else(|| rc_err(198, format!("{n} is not a variable in the model")))?;
        Ok((i, err))
    };
    let mut var_c: HashMap<usize, Coef> = HashMap::new();
    for t in &o.vars {
        for (a, _, c) in cov_list(t) {
            let (i, _) = resolve(&a)?;
            var_c.insert(i, c);
        }
    }
    let mut mean_c: HashMap<usize, Coef> = HashMap::new();
    for t in &o.means {
        for (a, _, c) in cov_list(t) {
            let (i, _) = resolve(&a)?;
            if endo[i] {
                return Err(rc_err(198, format!("means() not allowed for endogenous variable {a}")));
            }
            mean_c.insert(i, c);
        }
    }
    let mut cov_c: HashMap<(usize, usize), Coef> = HashMap::new();
    for t in &o.covs {
        for (a, b, c) in cov_list(t) {
            let (i, _) = resolve(&a)?;
            let (j, _) = resolve(&b)?;
            if i == j {
                var_c.insert(i, c);
            } else {
                cov_c.insert((i.min(j), i.max(j)), c);
            }
        }
    }
    // covstructure(group, structure)
    let exo_obs: Vec<usize> = (0..p).filter(|&i| !endo[i]).collect();
    let exo_lat: Vec<usize> = (p..m).filter(|&i| !endo[i]).collect();
    let en_obs: Vec<usize> = (0..p).filter(|&i| endo[i]).collect();
    let en_lat: Vec<usize> = (p..m).filter(|&i| endo[i]).collect();
    let mut exch: Vec<(Vec<usize>, bool)> = Vec::new();
    for t in &o.covstruct {
        let (grp, stru) = super::path::split_comma(t);
        let grp = grp.trim();
        let members: Vec<usize> = match grp {
            "_Ex" | "_OEx" => exo_obs.clone(),
            "_LEx" => exo_lat.clone(),
            "e._En" | "e._OEn" => en_obs.clone(),
            "e._LEn" => en_lat.clone(),
            _ => {
                let mut v = Vec::new();
                for w in grp.split_whitespace() {
                    v.push(resolve(w)?.0);
                }
                v
            }
        };
        let stru = stru.trim().to_lowercase();
        for (x, &i) in members.iter().enumerate() {
            for &j in &members[x + 1..] {
                let key = (i.min(j), i.max(j));
                match stru.as_str() {
                    "diagonal" | "independent" | "identity" | "zero" => {
                        cov_c.insert(key, Coef { fix: Some(0.0), label: None });
                    }
                    "unstructured" | "exchangeable" => {
                        cov_c.entry(key).or_default();
                    }
                    s => return Err(rc_err(198, format!("covstructure {s} not supported"))),
                }
            }
            match stru.as_str() {
                "identity" => {
                    var_c.insert(i, Coef { fix: Some(1.0), label: None });
                }
                "zero" => {
                    var_c.insert(i, Coef { fix: Some(0.0), label: None });
                }
                _ => {}
            }
        }
        if stru == "exchangeable" {
            exch.push((members.clone(), true));
        }
    }
    for (name, r) in &o.reliability {
        let (i, _) = resolve(name)?;
        if i >= p {
            return Err(rc_err(198, "reliability() requires observed variables"));
        }
        let v = o.stats.first().map(|s| s.cov[i * p + i]).unwrap_or(1.0);
        var_c.insert(i, Coef { fix: Some((1.0 - r) * v), label: None });
    }
    // anchoring of latent variables
    let mut anchor: HashMap<usize, usize> = HashMap::new(); // path index fixed at 1
    for l in p..m {
        if var_c.get(&l).is_some_and(|c| c.fix.is_some()) {
            continue;
        }
        let out: Vec<usize> = (0..sp.paths.len()).filter(|&k| idx(&sp.paths[k].from) == Some(l)).collect();
        if out.iter().any(|&k| sp.paths[k].c.fix.is_some()) {
            continue;
        }
        let free: Vec<usize> = out.iter().copied().filter(|&k| sp.paths[k].c.is_free()).collect();
        let first = free.iter().copied().find(|&k| idx(&sp.paths[k].to).is_some_and(|t| t < p)).or(free.first().copied());
        if let Some(k) = first {
            anchor.insert(l, k);
        } else if !endo[l] {
            return Err(rc_err(198, format!("latent variable {} has no indicators", vars[l])));
        }
    }
    let mut slots: Vec<Slot> = Vec::new();
    let classify_path = |t: usize| if meas[t] { Class::MCoef } else { Class::SCoef };
    for g in 0..ng {
        for (k, ps) in sp.paths.iter().enumerate() {
            let t = idx(&ps.to).unwrap();
            let f = idx(&ps.from).unwrap();
            let mut c = ps.c.clone();
            let mut auto = false;
            if anchor.get(&f) == Some(&k) && c.is_free() {
                c.fix = Some(1.0);
                auto = true;
            }
            slots.push(Slot { kind: Kind::Path, a: t, b: f, g, fix: c.fix, p: None, class: classify_path(t), auto });
            if let Some(l) = c.label {
                slots.last_mut().unwrap().p = Some(usize::MAX - label_id(&l));
            }
        }
        for i in 0..m {
            if endo[i] {
                // intercept
                let user = sp.cons.iter().rev().find(|(v, _, _)| idx(v) == Some(i)).map(|x| x.1.clone());
                let lat = i >= p;
                let mut fix = None;
                let mut auto = false;
                if sp.nocons.contains(&vars[i]) {
                    fix = Some(0.0);
                }
                if lat && !(g > 0 && o.invariant.contains(&Class::MCons)) {
                    fix = Some(0.0);
                    auto = true;
                }
                let mut label = None;
                if let Some(c) = &user {
                    fix = c.fix.or(fix);
                    label = c.label.clone();
                    auto = false;
                }
                if !(lat && fix == Some(0.0) && auto) || true {
                    let cl = if meas[i] { Class::MCons } else { Class::SCons };
                    slots.push(Slot { kind: Kind::Cons, a: i, b: i, g, fix, p: label.map(|l| usize::MAX - label_id(&l)), class: cl, auto });
                }
                // error variance
                let c = var_c.get(&i).cloned().unwrap_or_default();
                let cl = if meas[i] { Class::MErrVar } else { Class::SErrVar };
                slots.push(Slot { kind: Kind::Var, a: i, b: i, g, fix: c.fix, p: c.label.map(|l| usize::MAX - label_id(&l)), class: cl, auto: false });
            } else {
                let lat = i >= p;
                let c = mean_c.get(&i).cloned();
                let (fix, label, auto) = match c {
                    Some(c) => (c.fix, c.label, false),
                    None if lat && !(g > 0 && o.invariant.contains(&Class::MCons)) => (Some(0.0), None, true),
                    None => (None, None, false),
                };
                slots.push(Slot { kind: Kind::Mean, a: i, b: i, g, fix, p: label.map(|l| usize::MAX - label_id(&l)), class: Class::MeanEx, auto });
                let c = var_c.get(&i).cloned().unwrap_or_default();
                slots.push(Slot { kind: Kind::Var, a: i, b: i, g, fix: c.fix, p: c.label.map(|l| usize::MAX - label_id(&l)), class: Class::CovEx, auto: false });
            }
        }
        // covariances
        for i in 0..m {
            for j in i + 1..m {
                let user = cov_c.get(&(i, j)).cloned();
                let default_free = !endo[i] && !endo[j];
                let c = match user {
                    Some(c) => c,
                    None if default_free => Coef::default(),
                    None => continue,
                };
                if c.fix == Some(0.0) && c.label.is_none() {
                    continue;
                }
                if endo[i] != endo[j] {
                    return Err(rc_err(198, format!("covariance between endogenous and exogenous variables {} and {} not allowed", vars[i], vars[j])));
                }
                let cl = if !endo[i] {
                    Class::CovEx
                } else if meas[i] && meas[j] {
                    Class::MErrCov
                } else if !meas[i] && !meas[j] {
                    Class::SErrCov
                } else {
                    Class::SmErrCov
                };
                slots.push(Slot { kind: Kind::Cov, a: i, b: j, g, fix: c.fix, p: c.label.map(|l| usize::MAX - label_id(&l)), class: cl, auto: false });
            }
        }
    }
    // exchangeable structures: tie variances and covariances
    let mut exch_label: HashMap<(usize, Kind, usize), String> = HashMap::new();
    for (e, (mem, _)) in exch.iter().enumerate() {
        for s in slots.iter_mut() {
            let inm = |x: usize| mem.contains(&x);
            if s.fix.is_none() && s.p.is_none() && ((s.kind == Kind::Var && inm(s.a)) || (s.kind == Kind::Cov && inm(s.a) && inm(s.b))) {
                let l = format!("__exch{e}_{:?}_{}", s.kind, s.g);
                exch_label.insert((s.a, s.kind, s.g), l.clone());
                s.p = Some(usize::MAX - label_id(&l));
            }
        }
    }
    // assign parameters
    let mut keys: HashMap<String, usize> = HashMap::new();
    let mut k = 0;
    let mut labels = Vec::new();
    for s in slots.iter_mut() {
        if s.fix.is_some() {
            s.p = None;
            continue;
        }
        let key = match s.p {
            Some(code) => {
                let l = label_name(usize::MAX - code);
                format!("L:{l}")
            }
            None => {
                let shared = ng > 1 && o.invariant.contains(&s.class) && !(s.g > 0 && (s.kind == Kind::Mean || s.kind == Kind::Cons) && s.a >= p);
                let g = if shared { usize::MAX } else { s.g };
                format!("{g}:{:?}:{}:{}", s.kind, s.a, s.b)
            }
        };
        let id = *keys.entry(key.clone()).or_insert_with(|| {
            k += 1;
            k - 1
        });
        if let Some(l) = key.strip_prefix("L:") {
            if !l.starts_with("__") && !labels.iter().any(|(a, _): &(String, usize)| a == l) {
                labels.push((l.to_string(), id));
            }
        }
        s.p = Some(id);
    }
    let mut model = Model { vars, p, endo, meas, ng, slots, k, start: vec![0.0; k], glabels: o.glabels.clone(), gvar: o.gvar.clone(), labels };
    model.start = starts(&model, sp, o.stats);
    Ok(model)
}

thread_local! {
    static LABELS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn label_id(l: &str) -> usize {
    LABELS.with(|v| {
        let mut v = v.borrow_mut();
        match v.iter().position(|x| x == l) {
            Some(i) => i,
            None => {
                v.push(l.to_string());
                v.len() - 1
            }
        }
    })
}

fn label_name(i: usize) -> String {
    LABELS.with(|v| v.borrow().get(i).cloned().unwrap_or_default())
}

fn starts(md: &Model, sp: &Spec, stats: &[Stats]) -> Vec<f64> {
    let p = md.p;
    let m = md.m();
    let mut th = vec![f64::NAN; md.k];
    let st = |g: usize| stats.get(g).or(stats.first());
    let var_of = |g: usize, i: usize| st(g).map(|s| s.cov[i * p + i]).filter(|v| *v > 0.0).unwrap_or(1.0);
    let cov_of = |g: usize, i: usize, j: usize| st(g).map(|s| s.cov[i * p + j]).unwrap_or(0.0);
    // first observed indicator of each latent (anchor)
    let mut ind: HashMap<usize, (usize, f64)> = HashMap::new();
    for s in &md.slots {
        if s.kind == Kind::Path && s.b >= p && s.a < p && !ind.contains_key(&s.b) {
            ind.insert(s.b, (s.a, s.fix.unwrap_or(1.0)));
        }
    }
    // latent variance starts
    let lat_var = |g: usize, l: usize| -> f64 {
        match ind.get(&l) {
            Some(&(a, lam)) => 0.5 * var_of(g, a) / (lam * lam).max(1e-8),
            None => 1.0,
        }
    };
    let latvar_fixed = |g: usize, l: usize| md.slots.iter().find(|s| s.g == g && s.kind == Kind::Var && s.a == l).and_then(|s| s.fix);
    let _ = sp;
    for s in &md.slots {
        let Some(pi) = s.p else { continue };
        if !th[pi].is_nan() {
            continue;
        }
        let g = s.g;
        th[pi] = match s.kind {
            Kind::Path => {
                if s.b >= p && s.a < p {
                    // loading: covariance with the anchor indicator
                    let phi = latvar_fixed(g, s.b).unwrap_or_else(|| lat_var(g, s.b));
                    match ind.get(&s.b) {
                        Some(&(a, lam)) if a != s.a => {
                            let c = cov_of(g, s.a, a) / (lam * phi).max(1e-8);
                            if c.is_finite() && c != 0.0 { c } else { 1.0 }
                        }
                        _ => (0.5 * var_of(g, s.a) / phi.max(1e-8)).sqrt(),
                    }
                } else {
                    0.0
                }
            }
            Kind::Cons | Kind::Mean => {
                if s.a < p {
                    st(g).map(|x| x.mean[s.a]).unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            Kind::Var => {
                if s.a < p {
                    if md.endo[s.a] { 0.5 * var_of(g, s.a) } else { var_of(g, s.a) }
                } else {
                    let v = lat_var(g, s.a);
                    if md.endo[s.a] { 0.5 * v } else { v }
                }
            }
            Kind::Cov => {
                if s.a < p && s.b < p && !md.endo[s.a] {
                    cov_of(g, s.a, s.b)
                } else {
                    0.0
                }
            }
        };
    }
    let _ = m;
    th.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect()
}
