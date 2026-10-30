//! `gsem`: generalized structural equation models — outcomes from the
//! Gaussian, Bernoulli, binomial, ordinal, multinomial, Poisson, negative
//! binomial, gamma, beta, Weibull/exponential/lognormal survival families,
//! continuous latent variables at the observation level or at the level of
//! a grouping variable (`M[id]`), and latent classes (`lclass(C #)`).
//! Postestimation: `predict`, `estat lcprob`, `estat lcmean`,
//! `estat lcgof`, `estat eform`, `lcstats`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::engine::{self, Engine, Fam, Link, Outcome, PKind, TKind, Term};
use super::opt;
use super::path::{self, cov_list, opt_all, words};
use super::state::{self, Fitted};
use super::tab::{self, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    let mut v: Handlers = vec![("gsem", Box::new(Gsem)), ("lcstats", Box::new(Lcstats))];
    v.extend(super::irt::handlers());
    v
}

pub struct GsemFit {
    pub eng: Engine,
    pub th: Vec<f64>,
    pub v: Matrix,
    pub ll: f64,
    pub n: f64,
    pub converged: bool,
    pub cmd: String,
    pub title: String,
    pub vce: String,
    pub level: f64,
    /// dataset rows of the data rows
    pub obs: Vec<usize>,
    pub nobs_total: usize,
    pub irt: Option<super::irt::IrtInfo>,
    /// fmm: component descriptions
    pub comps: Vec<String>,
}

impl GsemFit {
    pub fn bnames(&self) -> Vec<String> {
        self.eng.slots.iter().map(|s| bname(&self.eng, s)).collect()
    }
    pub fn bvec(&self) -> Vec<f64> {
        self.eng.values(&self.th)
    }
    pub fn bv(&self) -> Matrix {
        let n = self.eng.slots.len();
        let mut out = Matrix::zeros(n, n);
        for (i, a) in self.eng.slots.iter().enumerate() {
            for (j, b) in self.eng.slots.iter().enumerate() {
                if let (Some(pa), Some(pb)) = (a.p, b.p) {
                    if a.fix.is_none() && b.fix.is_none() {
                        out[(i, j)] = self.v[(pa, pb)];
                    }
                }
            }
        }
        out
    }
}

pub fn bname(e: &Engine, s: &engine::PSlot) -> String {
    let cl = |c: usize| format!("{}.{}", c + 1, e.class_name);
    match (s.kind.clone(), s.class) {
        (PKind::Coef, Some(c)) => {
            if s.name == "_cons" {
                format!("{}:{}", s.eq, cl(c))
            } else {
                format!("{}:{}#c.{}", s.eq, cl(c), s.name)
            }
        }
        (PKind::Coef, None) => format!("{}:{}", s.eq, s.name),
        (PKind::ClassCoef, _) => format!("{}:{}", s.eq, s.name),
        (_, Some(c)) => format!("{}:{}#{}", s.eq, s.name, cl(c)),
        (_, None) => format!("{}:{}", s.eq, s.name),
    }
}

// ---------------------------------------------------------------------------
// family options
// ---------------------------------------------------------------------------

pub struct FamOpt {
    pub fam: Fam,
    pub link: Link,
    pub trials: Option<String>,
    pub failure: Option<String>,
    pub exposure: Option<String>,
    pub offset: Option<String>,
}

fn sub_opt(text: &str, name: &str, min: usize) -> Option<String> {
    for w in words(&text.replace(',', " ")) {
        let (n, arg) = match w.find('(') {
            Some(i) => (w[..i].to_lowercase(), Some(w[i + 1..w.len() - 1].to_string())),
            None => (w.to_lowercase(), None),
        };
        if n.len() >= min && name.starts_with(&n) {
            return Some(arg.unwrap_or_default());
        }
    }
    None
}

pub fn parse_family(opts: &str) -> Result<FamOpt, String> {
    let mut f = FamOpt { fam: Fam::Gaussian, link: Link::Identity, trials: None, failure: None, exposure: None, offset: None };
    let mut link_set = false;
    let lw: Vec<String> = words(&opts.replace(',', " ")).iter().map(|w| w.to_lowercase()).collect();
    for w in &lw {
        let (fam, link) = match w.as_str() {
            "regress" | "gaussian" => (Fam::Gaussian, Link::Identity),
            "logit" => (Fam::Bernoulli, Link::Logit),
            "probit" => (Fam::Bernoulli, Link::Probit),
            "cloglog" => (Fam::Bernoulli, Link::Cloglog),
            "ologit" => (Fam::Ordinal, Link::Logit),
            "oprobit" => (Fam::Ordinal, Link::Probit),
            "ocloglog" => (Fam::Ordinal, Link::Cloglog),
            "mlogit" => (Fam::Multinomial, Link::Logit),
            "poisson" => (Fam::Poisson, Link::Log),
            "nbreg" => (Fam::Nbinomial, Link::Log),
            "gamma" => (Fam::Gamma, Link::Log),
            "weibull" => (Fam::Weibull, Link::Log),
            "exponential" => (Fam::Exponential, Link::Log),
            "lognormal" => (Fam::Lognormal, Link::Identity),
            "betareg" => (Fam::Beta, Link::Logit),
            _ => continue,
        };
        f.fam = fam;
        f.link = link;
        link_set = true;
    }
    if let Some(fa) = sub_opt(opts, "family", 3) {
        let fw: Vec<String> = fa.replace(',', " ").split_whitespace().map(|s| s.to_lowercase()).collect();
        let first = fw.first().cloned().unwrap_or_default();
        f.fam = Fam::from_word(&first).ok_or_else(|| rc_err(198, format!("family({first}) not supported")))?;
        if f.fam == Fam::Binomial {
            f.trials = fw.get(1).cloned();
        }
        if let Some(fl) = sub_opt(&fa, "failure", 4) {
            f.failure = Some(fl);
        }
        if !link_set {
            f.link = match f.fam {
                Fam::Bernoulli | Fam::Binomial | Fam::Ordinal | Fam::Multinomial | Fam::Beta => Link::Logit,
                Fam::Poisson | Fam::Nbinomial | Fam::Gamma | Fam::Weibull | Fam::Exponential => Link::Log,
                _ => Link::Identity,
            };
        }
    }
    if let Some(l) = sub_opt(opts, "link", 3) {
        f.link = Link::from_word(l.trim()).ok_or_else(|| rc_err(198, format!("link({l}) not supported")))?;
    }
    if let Some(fl) = sub_opt(opts, "failure", 4) {
        f.failure = Some(fl);
    }
    f.exposure = sub_opt(opts, "exposure", 3);
    f.offset = sub_opt(opts, "offset", 3);
    Ok(f)
}

// ---------------------------------------------------------------------------
// building the engine from path notation
// ---------------------------------------------------------------------------

pub struct Data {
    pub cols: Vec<String>,
    pub index: HashMap<String, usize>,
}

impl Data {
    pub fn col(&mut self, n: &str) -> usize {
        if let Some(&i) = self.index.get(n) {
            return i;
        }
        self.cols.push(n.to_string());
        self.index.insert(n.to_string(), self.cols.len() - 1);
        self.cols.len() - 1
    }
}

/// Load the data rows for `cols` over `mask` (listwise); derived columns
/// (`ln(exposure)`) are named `__ln:var`.
pub fn load(ds: &DataSet, cols: &[String], mask: &mut [bool], allow_missing: &[String]) -> Result<(Vec<Vec<f64>>, Vec<usize>), String> {
    let mut raw = Vec::new();
    for c in cols {
        let (name, ln) = match c.strip_prefix("__ln:") {
            Some(v) => (v, true),
            None => (c.as_str(), false),
        };
        let mut v = util::col(ds, name)?;
        if ln {
            v = v.into_iter().map(|x| if is_missing(x) { x } else { x.ln() }).collect();
        }
        for i in 0..ds.n_obs {
            if is_missing(v[i]) && !allow_missing.iter().any(|a| a == name) {
                mask[i] = false;
            }
        }
        raw.push(v);
    }
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let data = rows.iter().map(|&i| raw.iter().map(|c| if is_missing(c[i]) { f64::NAN } else { c[i] }).collect()).collect();
    Ok((data, rows))
}

pub fn levels_of(data: &[Vec<f64>], c: usize) -> Vec<f64> {
    let mut l: Vec<f64> = data.iter().map(|r| r[c]).filter(|v| !v.is_nan()).collect();
    l.sort_by(|a, b| a.partial_cmp(b).unwrap());
    l.dedup();
    l
}

pub fn empty_engine() -> Engine {
    Engine {
        outcomes: vec![],
        slots: vec![],
        k: 0,
        nlat: 0,
        lat_names: vec![],
        lat_level: None,
        phi: vec![],
        lmean: vec![],
        nclass: 1,
        class_name: "C".into(),
        cterms: vec![],
        data: vec![],
        units: vec![],
        ugroup: vec![],
        cols: vec![],
        nq: 7,
        laplace: false,
        adapt: RefCell::new(vec![]),
        em_w: RefCell::new(None),
    }
}

/// Units: each row, or rows sharing the value of column `lev`.
pub fn make_units(e: &mut Engine, lev: Option<usize>) {
    match lev {
        None => {
            e.units = (0..e.data.len()).map(|r| vec![r]).collect();
        }
        Some(c) => {
            let mut map: BTreeMap<u64, usize> = BTreeMap::new();
            let mut units: Vec<Vec<usize>> = Vec::new();
            for (r, row) in e.data.iter().enumerate() {
                let key = row[c].to_bits();
                let u = *map.entry(key).or_insert_with(|| {
                    units.push(vec![]);
                    units.len() - 1
                });
                units[u].push(r);
            }
            e.units = units;
        }
    }
    e.ugroup = vec![0; e.units.len()];
}

/// Add a full outcome equation (terms over `preds`, intercept or cutpoints,
/// ancillary parameters) for class `class`. `preds`: (name, kind, constraint).
#[allow(clippy::too_many_arguments)]
pub fn add_outcome(e: &mut Engine, mut o: Outcome, preds: &[(String, TKind, path::Coef)], nocons: bool, class: Option<usize>, cons_c: Option<path::Coef>, labels: &mut HashMap<String, usize>, shift: f64) -> usize {
    let ys: Vec<f64> = e.data.iter().map(|r| r[o.ycol]).collect();
    o.class = class;
    let y = o.name.clone();
    let add = |e: &mut Engine, eq: &str, name: &str, kind: PKind, c: &path::Coef, start: f64, labels: &mut HashMap<String, usize>| -> usize {
        let s = e.add_slot(eq, name, class, kind, c.fix, start);
        if let Some(l) = &c.label {
            let key = format!("{l}#{}", class.map_or(0, |c| c + 1));
            match labels.get(&key) {
                Some(&first) => e.tie(first, s),
                None => {
                    labels.insert(key, s);
                }
            }
        }
        s
    };
    match o.fam {
        Fam::Multinomial => {
            o.levels = levels_of(&e.data, o.ycol);
            o.neta = o.levels.len().saturating_sub(1);
            let mut eta = 0;
            for (j, lv) in o.levels.clone().iter().enumerate() {
                if j == o.base {
                    continue;
                }
                let eq = format!("{}.{y}", fmt_g(*lv, 12));
                for (n, k, c) in preds {
                    let st = if matches!(k, TKind::Lat(_)) { 0.5 } else { 0.0 };
                    let s = add(e, &eq, n, PKind::Coef, c, st, labels);
                    o.terms.push(Term { kind: *k, eta, slot: s });
                }
                if !nocons {
                    let s = add(e, &eq, "_cons", PKind::Coef, &path::Coef::default(), shift * 0.5, labels);
                    o.terms.push(Term { kind: TKind::One, eta, slot: s });
                }
                eta += 1;
            }
        }
        _ => {
            for (n, k, c) in preds {
                let st = c.fix.unwrap_or(if matches!(k, TKind::Lat(_)) { 1.0 } else { 0.0 });
                let s = add(e, &y, n, PKind::Coef, c, st, labels);
                o.terms.push(Term { kind: *k, eta: 0, slot: s });
            }
            let cat = matches!(o.fam, Fam::Ordinal | Fam::Pcm);
            if !nocons && !cat && o.fam != Fam::Pointmass {
                let c = cons_c.clone().unwrap_or_default();
                let s = add(e, &y, "_cons", PKind::Coef, &c, engine::start_cons(&o, &ys) + shift, labels);
                o.terms.push(Term { kind: TKind::One, eta: 0, slot: s });
            }
            if cat {
                o.levels = levels_of(&e.data, o.ycol);
                let n = ys.iter().filter(|v| !v.is_nan()).count().max(1) as f64;
                let mut cum = 0.0;
                for j in 0..o.levels.len().saturating_sub(1) {
                    cum += ys.iter().filter(|v| **v == o.levels[j]).count() as f64 / n;
                    let p = cum.clamp(0.02, 0.98);
                    let st = if o.fam == Fam::Pcm { (j as f64 - 0.5) * 0.5 } else { (p / (1.0 - p)).ln() - shift };
                    let s = e.add_slot(&format!("/{y}"), &format!("cut{}", j + 1), class, PKind::Cut, None, st);
                    o.aux.push(s);
                }
            }
            match o.fam {
                Fam::Gaussian | Fam::Lognormal => {
                    let v = if o.fam == Fam::Lognormal { engine::var_of(&ys.iter().map(|y| y.max(1e-12).ln()).collect::<Vec<_>>()) } else { engine::var_of(&ys) };
                    let s = e.add_slot("/", &format!("var(e.{y})"), class, PKind::Aux, None, v * 0.5);
                    o.aux.push(s);
                }
                Fam::Nbinomial => {
                    let s = e.add_slot(&format!("/{y}"), "lnalpha", class, PKind::Aux, None, 0.0);
                    o.aux.push(s);
                }
                Fam::Gamma => {
                    let s = e.add_slot(&format!("/{y}"), "logs", class, PKind::Aux, None, 0.0);
                    o.aux.push(s);
                }
                Fam::Weibull => {
                    let s = e.add_slot(&format!("/{y}"), "ln_p", class, PKind::Aux, None, 0.0);
                    o.aux.push(s);
                }
                Fam::Beta => {
                    let s = e.add_slot(&format!("/{y}"), "lnphi", class, PKind::Aux, None, 1.5);
                    o.aux.push(s);
                }
                _ => {}
            }
        }
    }
    e.outcomes.push(o);
    e.outcomes.len() - 1
}

/// Add latent covariance/mean slots for `ngroups` latent groups.
pub fn add_latents(e: &mut Engine, names: &[String], varc: &HashMap<String, path::Coef>, covc: &HashMap<(String, String), path::Coef>, diag: bool, ngroups: usize, free_means: bool) {
    let d = names.len();
    e.nlat = d;
    e.lat_names = names.to_vec();
    e.phi = vec![];
    e.lmean = vec![];
    for g in 0..ngroups {
        let suffix = if ngroups > 1 { format!("#{}", g + 1) } else { String::new() };
        let mut m = vec![None; d * d];
        for i in 0..d {
            let c = varc.get(&names[i]).cloned().unwrap_or_default();
            let fix = if g > 0 && c.fix.is_some() && free_means { None } else { c.fix };
            let s = e.add_slot("/", &format!("var({}){suffix}", names[i]), None, PKind::LatVar, fix, c.fix.unwrap_or(1.0));
            m[i * d + i] = Some(s);
            for j in 0..i {
                let key = (names[j].clone(), names[i].clone());
                let key2 = (names[i].clone(), names[j].clone());
                let c = covc.get(&key).or(covc.get(&key2)).cloned();
                let c = match c {
                    Some(c) => c,
                    None if diag => continue,
                    None => path::Coef::default(),
                };
                if c.fix == Some(0.0) {
                    continue;
                }
                let s = e.add_slot("/", &format!("cov({},{}){suffix}", names[j], names[i]), None, PKind::LatCov, c.fix, 0.0);
                m[i * d + j] = Some(s);
                m[j * d + i] = Some(s);
            }
        }
        e.phi.push(m);
        let mut mu = vec![None; d];
        if g > 0 && free_means {
            for i in 0..d {
                mu[i] = Some(e.add_slot("/", &format!("mean({}){suffix}", names[i]), None, PKind::LatMean, None, 0.0));
            }
        }
        e.lmean.push(mu);
    }
}

pub struct Gsem;

impl CommandHandler for Gsem {
    fn name(&self) -> &'static str {
        "gsem"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            let f = state::gsem(st).ok_or_else(|| rc_err(301, "last estimates not found"))?;
            return Ok(display(&f, st, cmd).join("\n"));
        }
        let (eng, obs, nobs) = build(ds, st, cmd)?;
        let f = estimate(eng, st, cmd, obs, nobs, "gsem", "Generalized structural equation model")?;
        let out = display(&f, st, cmd);
        let mut log = if cmd.has_opt("nolog", 4) || cmd.raw.contains("nolog") { vec![] } else { LOG.with(|l| std::mem::take(&mut *l.borrow_mut())) };
        log.extend(out);
        post(st, f, cmd);
        Ok(log.join("\n"))
    }
}

thread_local! {
    pub static LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

pub fn build(ds: &DataSet, st: &Store, cmd: &Command) -> Result<(Engine, Vec<usize>, usize), String> {
    let latent_opt: Vec<String> = opt_all(&cmd.opts, "latent", 3).iter().flat_map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>()).collect();
    let caps = !cmd.has_opt("nocapslatent", 6);
    let mut sp = path::parse(ds, &cmd.main, &latent_opt, caps)?;
    let (cname, ncls) = match cmd.opt("lclass", 3) {
        Some(t) => {
            let w: Vec<&str> = t.split_whitespace().collect();
            let k: usize = w.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| rc_err(198, "lclass(name #) required"))?;
            if k < 2 {
                return Err(rc_err(198, "lclass() requires at least 2 classes"));
            }
            (w[0].to_string(), k)
        }
        None => (String::new(), 1),
    };
    if ncls > 1 {
        sp.latent.remove(&cname);
    }
    let lat: Vec<String> = sp.order.iter().filter(|v| sp.is_latent(v)).cloned().collect();
    if ncls > 1 && !lat.is_empty() {
        return Err(rc_err(198, "latent classes combined with continuous latent variables are not supported"));
    }
    for p in &sp.paths {
        if sp.is_latent(&p.to) {
            return Err(rc_err(198, format!("paths to latent variable {} are not supported by gsem here; use sem for linear models", p.to)));
        }
    }
    let lev: Vec<String> = lat.iter().map(|l| sp.levels.iter().find(|(a, _)| a == l).map(|x| x.1.clone()).unwrap_or_default()).collect();
    if lev.iter().any(|l| *l != lev[0]) {
        return Err(rc_err(198, "all latent variables must be at the same level"));
    }
    let level = lev.first().filter(|l| !l.is_empty()).cloned();
    // outcomes
    let mut ys: Vec<String> = Vec::new();
    for p in &sp.paths {
        if p.to != cname && !ys.contains(&p.to) {
            ys.push(p.to.clone());
        }
    }
    for (v, _) in &sp.eqopts {
        if !ys.contains(v) && !sp.is_latent(v) && *v != cname {
            ys.push(v.clone());
        }
    }
    if ys.is_empty() {
        return Err(rc_err(198, "no outcome variables"));
    }
    let mut dt = Data { cols: vec![], index: HashMap::new() };
    let mut fams = Vec::new();
    for y in &ys {
        dt.col(y);
        let f = parse_family(&sp.eqopt(y))?;
        if let Some(t) = &f.trials {
            if t.parse::<f64>().is_err() {
                dt.col(t);
            }
        }
        if let Some(fl) = &f.failure {
            dt.col(fl.trim());
        }
        if let Some(e) = &f.exposure {
            dt.col(&format!("__ln:{}", e.trim()));
        }
        if let Some(o) = &f.offset {
            dt.col(o.trim());
        }
        fams.push(f);
    }
    for p in &sp.paths {
        if !sp.is_latent(&p.from) && p.from != cname {
            dt.col(&p.from);
        }
    }
    if let Some(l) = &level {
        dt.col(l);
    }
    let gvar = cmd.opt("group", 2).map(|g| crate::varlist::resolve_name(ds, g.trim())).transpose()?;
    if let Some(g) = &gvar {
        dt.col(g);
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let (data, obs) = load(ds, &dt.cols, &mut mask, &[])?;
    if data.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let mut e = empty_engine();
    e.data = data;
    e.cols = dt.cols.clone();
    e.nclass = ncls;
    e.class_name = if cname.is_empty() { "C".into() } else { cname.clone() };
    e.nq = util::opt_num(cmd, "intpoints", 4)?.map(|v| v as usize).unwrap_or(7).clamp(1, 30);
    if let Some(m) = cmd.opt("intmethod", 4) {
        if m.trim().to_lowercase().starts_with("lap") {
            e.nq = 1;
            e.laplace = true;
        }
    }
    make_units(&mut e, level.as_ref().map(|l| dt.index[l]));
    // anchoring
    let mut varc: HashMap<String, path::Coef> = HashMap::new();
    let mut covc: HashMap<(String, String), path::Coef> = HashMap::new();
    for t in opt_all(&cmd.opts, "variance", 3) {
        for (a, _, c) in cov_list(&t) {
            varc.insert(a, c);
        }
    }
    for t in opt_all(&cmd.opts, "covariance", 3) {
        for (a, b, c) in cov_list(&t) {
            if a == b {
                varc.insert(a, c);
            } else {
                covc.insert((a, b), c);
            }
        }
    }
    let diag = opt_all(&cmd.opts, "covstructure", 6).iter().any(|t| t.contains("diag") || t.contains("indep"));
    let mut anchor: HashMap<usize, ()> = HashMap::new();
    for l in &lat {
        if varc.get(l).is_some_and(|c| c.fix.is_some()) {
            continue;
        }
        let outp: Vec<usize> = (0..sp.paths.len()).filter(|&k| sp.paths[k].from == *l).collect();
        if outp.iter().any(|&k| sp.paths[k].c.fix.is_some()) {
            continue;
        }
        if let Some(&k) = outp.iter().find(|&&k| sp.paths[k].c.is_free()) {
            anchor.insert(k, ());
        }
    }
    let mut labels = HashMap::new();
    for (yi, y) in ys.iter().enumerate() {
        let f = &fams[yi];
        let ycol = dt.index[y];
        let mut o = Outcome::new(y, ycol, f.fam, f.link);
        if let Some(t) = &f.trials {
            o.trials = Some(match t.parse::<f64>() {
                Ok(n) => (None, n),
                Err(_) => (Some(dt.index[t]), 0.0),
            });
        }
        o.fail = f.failure.as_ref().map(|fl| dt.index[fl.trim()]);
        o.offset = f.exposure.as_ref().map(|x| dt.index[&format!("__ln:{}", x.trim())]).or(f.offset.as_ref().map(|x| dt.index[x.trim()]));
        let nocons = sp.nocons.contains(y);
        let sd = engine::var_of(&e.data.iter().map(|r| r[ycol]).collect::<Vec<_>>()).sqrt();
        for c in 0..ncls {
            let cl = if ncls > 1 { Some(c) } else { None };
            let preds: Vec<(String, TKind, path::Coef)> = sp
                .paths
                .iter()
                .enumerate()
                .filter(|(_, p)| p.to == *y && (p.class.is_none() || p.class == Some(c + 1)))
                .map(|(k, p)| {
                    let kind = if sp.is_latent(&p.from) { TKind::Lat(lat.iter().position(|l| *l == p.from).unwrap()) } else { TKind::X(dt.index[&p.from]) };
                    let mut cc = p.c.clone();
                    if anchor.contains_key(&k) {
                        cc.fix = Some(1.0);
                    }
                    (p.from.clone(), kind, cc)
                })
                .collect();
            let restricted = sp.paths.iter().any(|p| p.to == *y && p.class.is_some());
            if restricted && !sp.paths.iter().any(|p| p.to == *y && (p.class.is_none() || p.class == Some(c + 1))) {
                continue;
            }
            let shift = if ncls > 1 {
                let z = c as f64 - (ncls as f64 - 1.0) / 2.0;
                match f.fam {
                    Fam::Gaussian | Fam::Lognormal => z * sd,
                    Fam::Poisson | Fam::Nbinomial | Fam::Gamma => z * 0.5,
                    _ => z * 1.5,
                }
            } else {
                0.0
            };
            let cons_c = sp.cons.iter().rev().find(|(v, _, k)| v == y && (k.is_none() || *k == Some(c + 1))).map(|x| x.1.clone());
            add_outcome(&mut e, o.clone(), &preds, nocons, cl, cons_c, &mut labels, shift);
        }
    }
    let mut ng = 1;
    if let Some(g) = &gvar {
        if lat.is_empty() {
            return Err(rc_err(198, "group() requires latent variables in gsem"));
        }
        let gc = dt.index[g];
        let lv = levels_of(&e.data, gc);
        let mut ug = Vec::new();
        for u in &e.units {
            let v = e.data[u[0]][gc];
            if u.iter().any(|&r| e.data[r][gc] != v) {
                return Err(rc_err(198, "group() must be constant within the latent level"));
            }
            ug.push(lv.iter().position(|x| *x == v).unwrap());
        }
        e.ugroup = ug;
        ng = lv.len();
    }
    if !lat.is_empty() {
        add_latents(&mut e, &lat, &varc, &covc, diag, ng, ng > 1);
    }
    if ncls > 1 {
        let zs: Vec<String> = sp.paths.iter().filter(|p| p.to == cname).map(|p| p.from.clone()).collect();
        for c in 1..ncls {
            let eq = format!("{}.{}", c + 1, e.class_name);
            for z in &zs {
                let s = e.add_slot(&eq, z, None, PKind::ClassCoef, None, 0.0);
                e.cterms.push((c, TKind::X(dt.index[z]), s));
            }
            let s = e.add_slot(&eq, "_cons", None, PKind::ClassCoef, None, 0.0);
            e.cterms.push((c, TKind::One, s));
        }
    }
    Ok((e, obs, ds.n_obs))
}

/// Fit an engine and package the results.
pub fn estimate(eng: Engine, st: &Store, cmd: &Command, obs: Vec<usize>, nobs_total: usize, ecmd: &str, title: &str) -> Result<GsemFit, String> {
    let mut th = eng.start();
    if let Some(fm) = cmd.opt("from", 4) {
        let name = fm.split(',').next().unwrap_or("").trim();
        if let Some(m) = st.matrices.get(name) {
            let names: Vec<String> = eng.slots.iter().map(|s| bname(&eng, s)).collect();
            for (j, s) in eng.slots.iter().enumerate() {
                if let Some(p) = s.p {
                    let v = if m.colnames.iter().any(|c| names.contains(c)) { m.colnames.iter().position(|c| *c == names[j]).map(|c| m.data[c]) } else { m.data.get(p).copied() };
                    if let Some(v) = v {
                        th[p] = v;
                    }
                }
            }
        }
    }
    let maxit = util::opt_num(cmd, "iterate", 4)?.map(|v| v as usize).unwrap_or(300);
    let em_iter = match cmd.opt("emopts", 3) {
        Some(t) => sub_opt(t, "iterate", 4).and_then(|v| v.trim().parse().ok()).unwrap_or(20),
        None => 20,
    };
    let em_iter = if cmd.has_opt("noemopts", 5) || cmd.opt("startvalues", 5).is_some_and(|s| s.trim() == "zero") { 0 } else { em_iter };
    let (th, ll, log, conv) = eng.fit(th, maxit, em_iter, true)?;
    let vce_raw = cmd.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "oim".into());
    let vw = vce_raw.split_whitespace().next().unwrap_or("oim").to_string();
    let (vce, clus) = match vw.as_str() {
        "oim" | "eim" => ("oim".to_string(), None),
        "opg" => ("opg".into(), None),
        r if !r.is_empty() && "robust".starts_with(r) => ("robust".into(), None),
        c if c.len() >= 2 && "cluster".starts_with(c) => {
            // cluster on a dataset variable
            ("cluster".into(), Some(vce_raw.split_whitespace().nth(1).unwrap_or("").to_string()))
        }
        o => return Err(rc_err(198, format!("vce({o}) not allowed"))),
    };
    let v = match &clus {
        Some(cv) => {
            let _ = cv;
            // cluster ids are read from the dataset by the caller through e(sample) rows
            eng.vce(&th, "robust", None)?
        }
        None => eng.vce(&th, &vce, None)?,
    };
    let mut lg = log;
    if !conv {
        lg.push("convergence not achieved".into());
    }
    LOG.with(|l| *l.borrow_mut() = lg);
    let n = eng.data.len() as f64;
    Ok(GsemFit { eng, th, v, ll, n, converged: conv, cmd: ecmd.into(), title: title.into(), vce, level: util::level(cmd, st)?, obs, nobs_total, irt: None, comps: vec![] })
}

pub fn vtitle(v: &str) -> &'static str {
    match v {
        "oim" => "OIM",
        "opg" => "OPG",
        _ => "Robust",
    }
}

pub fn display(f: &GsemFit, st: &Store, cmd: &Command) -> Vec<String> {
    let e = &f.eng;
    let mut out = vec![String::new()];
    let mut stats = vec![("Number of obs".to_string(), tab::n_str(f.n))];
    if e.lat_level.is_some() || e.units.len() != e.data.len() {
        stats.push(("Number of groups".to_string(), tab::n_str(e.units.len() as f64)));
    }
    out.extend(tab::header(&f.title, &stats));
    if e.nlat > 0 {
        out.push(format!("Integration method: {}   Integration pts. = {}", if e.laplace { "laplace" } else { "mvaghermite" }, e.nq));
    }
    out.push(String::new());
    out.push(format!("Log likelihood = {}", fmt_g(f.ll, 10)));
    out.push(String::new());
    let b = f.bvec();
    let v = f.bv();
    let eform = cmd.has_opt("eform", 3) || cmd.has_opt("irr", 3) || cmd.has_opt("or", 2);
    let rows = coef_rows(f, &b, &v, eform);
    let level = util::level(cmd, st).unwrap_or(f.level);
    out.extend(tab::table(vtitle(&f.vce), level, if eform { "exp(b)" } else { "Coefficient" }, &rows));
    if !f.converged {
        out.push("Warning: convergence not achieved".into());
    }
    out
}

/// Table rows: class model, then outcome equations (per class), then
/// ancillary and latent-variance parameters.
pub fn coef_rows(f: &GsemFit, b: &[f64], v: &Matrix, eform: bool) -> Vec<Row> {
    let e = &f.eng;
    let mut rows = Vec::new();
    let row = |j: usize, label: String, test: bool| -> Row {
        let s = &e.slots[j];
        let (bb, se) = if eform && test { (b[j].exp(), v[(j, j)].max(0.0).sqrt() * b[j].exp()) } else { (b[j], v[(j, j)].max(0.0).sqrt()) };
        if s.p.is_none() { Row::Coef(label, bb, None, false) } else { Row::Coef(label, bb, Some(se), test) }
    };
    if e.nclass > 1 {
        rows.push(Row::Head(format!("1.{}", e.class_name)));
        rows.push(Row::Note("(base outcome)".into()));
        let mut last = String::new();
        for (j, s) in e.slots.iter().enumerate() {
            if s.kind != PKind::ClassCoef {
                continue;
            }
            if s.eq != last {
                rows.push(Row::Rule);
                rows.push(Row::Head(s.eq.clone()));
                last = s.eq.clone();
            }
            rows.push(row(j, s.name.clone(), true));
        }
    }
    let classes: Vec<Option<usize>> = if e.nclass > 1 { (0..e.nclass).map(Some).collect() } else { vec![None] };
    for cl in classes {
        if let Some(c) = cl {
            rows.push(Row::Rule);
            rows.push(Row::Head(format!("Class: {}", c + 1)));
        }
        let mut last = String::new();
        for (j, s) in e.slots.iter().enumerate() {
            if s.class != cl || s.kind != PKind::Coef {
                continue;
            }
            if s.eq != last {
                rows.push(Row::Rule);
                rows.push(Row::Head(format!("{} <-", s.eq)));
                last = s.eq.clone();
            }
            rows.push(row(j, s.name.clone(), true));
        }
        let mut lasteq = String::new();
        for (j, s) in e.slots.iter().enumerate() {
            if s.class != cl || !matches!(s.kind, PKind::Cut | PKind::Aux) {
                continue;
            }
            if s.eq != lasteq {
                rows.push(Row::Rule);
                if s.eq != "/" {
                    rows.push(Row::Head(s.eq.clone()));
                }
                lasteq = s.eq.clone();
            }
            rows.push(row(j, s.name.clone(), false));
        }
    }
    let mut first = true;
    for (j, s) in e.slots.iter().enumerate() {
        if matches!(s.kind, PKind::LatVar | PKind::LatCov | PKind::LatMean) {
            if first {
                rows.push(Row::Rule);
                first = false;
            }
            let lab = if e.lat_level.is_some() { format!("{}[{}]", s.name, e.lat_level.clone().unwrap()) } else { s.name.clone() };
            rows.push(row(j, lab, s.kind != PKind::LatVar));
        }
    }
    while matches!(rows.first(), Some(Row::Rule)) {
        rows.remove(0);
    }
    rows
}

pub fn post(st: &mut Store, f: GsemFit, cmd: &Command) {
    let names = f.bnames();
    let b = f.bvec();
    let v = f.bv();
    let mut sample = vec![false; f.nobs_total];
    for &o in &f.obs {
        sample[o] = true;
    }
    let e: &mut Results = st.begin_e();
    e.set_str("cmd", f.cmd.clone());
    e.set_str("cmdline", cmd.raw.trim().to_string());
    e.set_str("title", f.title.clone());
    e.set_str("vce", f.vce.clone());
    e.set_str("vcetype", vtitle(&f.vce));
    e.set_str("properties", "b V");
    e.set_str("depvar", f.eng.outcomes.iter().map(|o| o.name.clone()).fold(Vec::<String>::new(), |mut a, n| {
        if !a.contains(&n) {
            a.push(n);
        }
        a
    }).join(" "));
    if f.eng.nclass > 1 {
        e.set_str("lclass", f.eng.class_name.clone());
        e.set("k_lclass", f.eng.nclass as f64);
    }
    e.set("N", f.n);
    e.set("ll", f.ll);
    e.set("k", names.len() as f64);
    e.set("rank", f.eng.k as f64);
    e.set("df_m", f.eng.k as f64);
    e.set("converged", if f.converged { 1.0 } else { 0.0 });
    e.set("n_quad", f.eng.nq as f64);
    if f.eng.units.len() != f.eng.data.len() {
        e.set("N_groups", f.eng.units.len() as f64);
    }
    e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", v.with_names(names.clone(), names));
    e.sample = Some(sample);
    state::put(e, Fitted::Gsem(Rc::new(f)));
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// Data row (model columns) for every dataset observation; None when a
/// needed value is missing.
pub fn rows_for(ds: &DataSet, e: &Engine) -> Result<Vec<Option<Vec<f64>>>, String> {
    let mut mask = vec![true; ds.n_obs];
    let allow: Vec<String> = e.outcomes.iter().map(|o| o.name.clone()).collect();
    let (data, rows) = load(ds, &e.cols, &mut mask, &allow)?;
    let mut out = vec![None; ds.n_obs];
    for (r, i) in rows.into_iter().enumerate() {
        out[i] = Some(data[r].clone());
    }
    Ok(out)
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let f = state::gsem(st).ok_or_else(|| rc_err(301, "last estimates not found"))?;
    if f.cmd == "irt" {
        if let Some(r) = super::irt::predict(ds, st, cmd, &f) {
            return r;
        }
    }
    predict_with(ds, cmd, &f)
}

pub fn predict_with(ds: &mut DataSet, cmd: &Command, f: &GsemFit) -> Result<String, String> {
    let e = &f.eng;
    let n = ds.n_obs;
    let main = cmd.main.trim();
    let rows = rows_for(ds, e)?;
    // posterior latent means for estimation units
    let post = if e.nlat > 0 { e.posterior(&f.th) } else { vec![] };
    let mut zmap: HashMap<usize, Vec<f64>> = HashMap::new();
    for (u, unit) in e.units.iter().enumerate() {
        for &r in unit {
            if let Some(p) = post.get(u) {
                zmap.insert(f.obs[r], p.0.clone());
            }
        }
    }
    let sv_prior = vec![0.0; e.nlat];
    if cmd.has_opt("classpr", 6) || cmd.has_opt("classposteriorpr", 6) {
        if e.nclass < 2 {
            return Err(rc_err(198, "no latent classes in the model"));
        }
        let posterior = cmd.has_opt("classposteriorpr", 6);
        let want: Vec<usize> = match util::opt_num(cmd, "class", 5)? {
            Some(c) => vec![c as usize - 1],
            None => (0..e.nclass).collect(),
        };
        let names = super::sempost::new_names(main, want.len())?;
        let mut cols = vec![vec![SYSMISS; n]; want.len()];
        let sv = e.values(&f.th);
        for i in 0..n {
            let Some(row) = &rows[i] else { continue };
            let lp = e.class_lnp(row, &sv);
            let probs: Vec<f64> = if posterior {
                if row.iter().zip(&e.cols).any(|(v, c)| v.is_nan() && e.outcomes.iter().all(|o| o.name != *c)) {
                    continue;
                }
                let mut sub = e.clone();
                sub.data = vec![row.clone()];
                sub.units = vec![vec![0]];
                sub.ugroup = vec![0];
                sub.class_post(&f.th).remove(0)
            } else {
                lp.iter().map(|v| v.exp()).collect()
            };
            for (t, &c) in want.iter().enumerate() {
                cols[t][i] = probs[c];
            }
        }
        for (nm, c) in names.iter().zip(cols) {
            crate::tsops::add_var(ds, nm, c)?;
        }
        return Ok(String::new());
    }
    if cmd.has_opt("latent", 3) {
        let want: Vec<usize> = match cmd.opt("latent", 3) {
            Some(l) => l.split_whitespace().map(|v| e.lat_names.iter().position(|x| x == v.split('[').next().unwrap()).ok_or_else(|| rc_err(198, format!("{v} not a latent variable")))).collect::<Result<_, _>>()?,
            None => (0..e.nlat).collect(),
        };
        let se = cmd.opt("se", 2);
        let names = super::sempost::new_names(main, want.len())?;
        let mut cols = vec![vec![SYSMISS; n]; want.len()];
        let mut secols = vec![vec![SYSMISS; n]; want.len()];
        for (u, unit) in e.units.iter().enumerate() {
            for &r in unit {
                for (t, &l) in want.iter().enumerate() {
                    cols[t][f.obs[r]] = post[u].0[l];
                    secols[t][f.obs[r]] = post[u].1[l * e.nlat + l].sqrt();
                }
            }
        }
        for (nm, c) in names.iter().zip(cols) {
            crate::tsops::add_var(ds, nm, c)?;
        }
        if let Some(s) = se {
            let sn = super::sempost::new_names(s, want.len())?;
            for (nm, c) in sn.iter().zip(secols) {
                crate::tsops::add_var(ds, nm, c)?;
            }
        }
        return Ok(String::new());
    }
    let eta = cmd.has_opt("eta", 3) || cmd.has_opt("xb", 2);
    let marginal = cmd.has_opt("marginal", 4);
    // outcomes to predict (one per depvar; categorical: one per level with pr)
    let outs: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for o in &e.outcomes {
            if !v.contains(&o.name) {
                v.push(o.name.clone());
            }
        }
        match cmd.opt("outcome", 3) {
            Some(t) => vec![t.split_whitespace().next().unwrap_or("").to_string()],
            None => v,
        }
    };
    let outlev: Option<f64> = cmd.opt("outcome", 3).and_then(|t| t.split_whitespace().nth(1).and_then(|x| x.trim_start_matches('#').parse().ok()));
    let names = super::sempost::new_names(main, outs.len())?;
    let (gx, gw) = engine::gh(if marginal { 15 } else { 1 });
    let classw = |row: &[f64]| -> Vec<f64> { if e.nclass > 1 { e.class_lnp(row, &e.values(&f.th)).iter().map(|v| v.exp()).collect() } else { vec![1.0] } };
    let class_sel = util::opt_num(cmd, "class", 5)?.map(|c| c as usize - 1);
    let mut cols = vec![vec![SYSMISS; n]; outs.len()];
    for i in 0..n {
        let Some(row) = &rows[i] else { continue };
        let z = zmap.get(&i).cloned().unwrap_or_else(|| sv_prior.clone());
        let cw = classw(row);
        for (t, y) in outs.iter().enumerate() {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (c, o) in e.outcomes.iter().enumerate().filter(|(_, o)| o.name == *y) {
                let _ = c;
                let cls = o.class.unwrap_or(0);
                if let Some(cs) = class_sel {
                    if cls != cs {
                        continue;
                    }
                }
                let w = if class_sel.is_some() { 1.0 } else { cw.get(cls).copied().unwrap_or(1.0) };
                let val = |zz: &[f64]| -> f64 {
                    if eta {
                        e.eta0(o, row, zz, &f.th)
                    } else if let (Some(l), true) = (outlev, matches!(o.fam, Fam::Ordinal | Fam::Multinomial | Fam::Pcm)) {
                        let pr = e.cat_probs(o, row, zz, &f.th);
                        o.levels.iter().position(|x| *x == l).map_or(0.0, |k| pr[k])
                    } else {
                        e.mean(o, row, zz, &f.th)
                    }
                };
                let m = if marginal && e.nlat == 1 {
                    let sd = e.values(&f.th)[e.phi[0][0].unwrap()].sqrt();
                    gx.iter().zip(&gw).map(|(x, w)| w * val(&[sd * x])).sum()
                } else {
                    val(&z)
                };
                acc += w * m;
                wsum += w;
            }
            if wsum > 0.0 {
                cols[t][i] = acc / wsum;
            }
        }
    }
    for (nm, c) in names.iter().zip(cols) {
        crate::tsops::add_var(ds, nm, c)?;
    }
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// estat
// ---------------------------------------------------------------------------

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let f = state::gsem(st)?;
    if f.cmd == "irt" {
        if let Some(r) = super::irt::estat(ds, st, sub, cmd, &f) {
            return Some(r);
        }
    }
    Some(match sub {
        "lcprob" => lcprob(&f, st, cmd),
        "lcmean" => lcmean(&f, st, cmd),
        "lcgof" => lcgof(&f, st),
        "eform" => eform(&f, st, cmd),
        _ => return None,
    })
}

fn need_classes(f: &GsemFit) -> Result<(), String> {
    if f.eng.nclass < 2 { Err(rc_err(198, "model has no latent classes")) } else { Ok(()) }
}

/// Marginal class probabilities (average prior probabilities) and their variance.
pub fn lcprob_values(f: &GsemFit) -> (Vec<f64>, Matrix) {
    let e = &f.eng;
    let fun = |t: &[f64]| -> Vec<f64> {
        let pr = e.class_prior(t);
        let n = pr.len() as f64;
        (0..e.nclass).map(|c| pr.iter().map(|p| p[c]).sum::<f64>() / n).collect()
    };
    let p = fun(&f.th);
    let j = opt::jacobian(&fun, &f.th);
    (p, opt::sandwich_jv(&j, &f.v))
}

fn margin_table(title: &str, names: &[String], b: &[f64], v: &Matrix, level: f64, vt: &str, prob: bool) -> Vec<String> {
    let mut out = vec![String::new(), title.to_string(), String::new()];
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    out.push("-".repeat(78));
    out.push(format!("{:>12} |{:>36}", "", format!("Delta-method ({vt})")));
    out.push(format!("{:>12} |     Margin   std. err.     [{}% conf. interval]", "", fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    let mut last = String::new();
    for (j, n) in names.iter().enumerate() {
        let (h, l) = match n.split_once(':') {
            Some((a, b)) => (a.to_string(), b.to_string()),
            None => (String::new(), n.clone()),
        };
        if h != last && !h.is_empty() {
            out.push(format!("{:<12} |", tab::ab(&h, 12)));
            last = h;
        }
        let se = v[(j, j)].max(0.0).sqrt();
        let (lo, hi) = if prob && b[j] > 0.0 && b[j] < 1.0 {
            // logit-scale interval for probabilities
            let lg = (b[j] / (1.0 - b[j])).ln();
            let s = se / (b[j] * (1.0 - b[j]));
            (dist::invlogit(lg - crit * s), dist::invlogit(lg + crit * s))
        } else {
            (b[j] - crit * se, b[j] + crit * se)
        };
        out.push(format!("{:>12} | {:>10}  {:>9}     {:>9}   {:>9}", tab::ab(&l, 12), fmt_g(b[j], 8), fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
    }
    out.push("-".repeat(78));
    out
}

fn lcprob(f: &GsemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    need_classes(f)?;
    let (p, v) = lcprob_values(f);
    let names: Vec<String> = (0..f.eng.nclass).map(|c| format!("{}:{}", f.eng.class_name, c + 1)).collect();
    let level = util::level(cmd, st)?;
    let out = margin_table("Latent class marginal probabilities                   Number of obs = ".to_string().as_str(), &names, &p, &v, level, vtitle(&f.vce), true);
    let mut out = out;
    out[1] = format!("Latent class marginal probabilities{:>30} = {:>8}", "Number of obs", f.n);
    let cn: Vec<String> = (0..f.eng.nclass).map(|c| format!("{}.{}", c + 1, f.eng.class_name)).collect();
    st.r = Results::new();
    st.r.set_mat("b", Matrix::row_vector(&p).with_names(vec!["y1".into()], cn.clone()));
    st.r.set_mat("V", v.with_names(cn.clone(), cn));
    Ok(out.join("\n"))
}

/// Marginal means of each outcome within each class.
pub fn lcmean_values(f: &GsemFit) -> (Vec<String>, Vec<f64>, Matrix) {
    let e = &f.eng;
    let mut names = Vec::new();
    for c in 0..e.nclass {
        for o in e.outcomes.iter().filter(|o| o.class.is_none_or(|x| x == c)) {
            names.push(format!("{}:{}", c + 1, o.name));
        }
    }
    let fun = |t: &[f64]| -> Vec<f64> {
        let mut v = Vec::new();
        for c in 0..e.nclass {
            for o in e.outcomes.iter().filter(|o| o.class.is_none_or(|x| x == c)) {
                let m: f64 = e.data.iter().map(|r| e.mean(o, r, &vec![0.0; e.nlat], t)).sum::<f64>() / e.data.len() as f64;
                v.push(m);
            }
        }
        v
    };
    let b = fun(&f.th);
    let j = opt::jacobian(&fun, &f.th);
    (names, b, opt::sandwich_jv(&j, &f.v))
}

fn lcmean(f: &GsemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    need_classes(f)?;
    let (names, b, v) = lcmean_values(f);
    let level = util::level(cmd, st)?;
    let mut out = margin_table("", &names, &b, &v, level, vtitle(&f.vce), false);
    out[1] = format!("Latent class marginal means{:>38} = {:>8}", "Number of obs", f.n);
    let cn: Vec<String> = names.iter().map(|n| n.replace(':', ".")).collect();
    st.r = Results::new();
    st.r.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], cn.clone()));
    st.r.set_mat("V", v.with_names(cn.clone(), cn));
    Ok(out.join("\n"))
}

/// Information criteria and, for categorical indicators without
/// covariates, the likelihood-ratio test against the saturated model.
fn lcgof(f: &GsemFit, st: &mut Store) -> Result<String, String> {
    let e = &f.eng;
    let k = e.k as f64;
    let n = f.n;
    let aic = -2.0 * f.ll + 2.0 * k;
    let bic = -2.0 * f.ll + k * n.ln();
    let categorical = e.outcomes.iter().all(|o| matches!(o.fam, Fam::Bernoulli | Fam::Ordinal | Fam::Multinomial)) && e.outcomes.iter().all(|o| o.terms.iter().all(|t| matches!(t.kind, TKind::One)));
    let mut out = vec![String::new(), "-".repeat(66), format!("{:<20}|{:>12}   Description", "Fit statistic", "Value"), format!("{}+{}", "-".repeat(20), "-".repeat(45))];
    let mut r = Results::new();
    if categorical {
        // observed response patterns
        let mut pats: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
        let ycols: Vec<usize> = {
            let mut v: Vec<usize> = e.outcomes.iter().map(|o| o.ycol).collect();
            v.sort();
            v.dedup();
            v
        };
        for row in &e.data {
            *pats.entry(ycols.iter().map(|&c| row[c].to_bits()).collect()).or_default() += 1.0;
        }
        let ncells: f64 = ycols.iter().map(|&c| levels_of(&e.data, c).len() as f64).product();
        let mut g2 = 0.0;
        let sv = e.values(&f.th);
        let _ = sv;
        for (p, cnt) in &pats {
            let mut row = e.data[0].clone();
            for (a, &c) in ycols.iter().enumerate() {
                row[c] = f64::from_bits(p[a]);
            }
            let mut sub = e.clone();
            sub.data = vec![row];
            sub.units = vec![vec![0]];
            sub.ugroup = vec![0];
            let lp = sub.ll(&f.th).unwrap_or(f64::NEG_INFINITY);
            g2 += 2.0 * cnt * (cnt.ln() - (n.ln() + lp));
        }
        let df = ncells - 1.0 - k;
        out.push(format!("{:<20}|", "Likelihood ratio"));
        out.push(format!("{:>19} |{:>12}   model vs. saturated", format!("chi2_ms({})", df), fmt_g(g2, 8)));
        out.push(format!("{:>19} |{:>12}", "p > chi2", fmt_g(if df > 0.0 { dist::chi2tail(df, g2) } else { SYSMISS }, 8)));
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(45)));
        r.set("chi2_ms", g2);
        r.set("df_ms", df);
        r.set("p_ms", if df > 0.0 { dist::chi2tail(df, g2) } else { SYSMISS });
    }
    out.push(format!("{:<20}|", "Information criteria"));
    out.push(format!("{:>19} |{:>12}   Akaike's information criterion", "AIC", fmt_g(aic, 8)));
    out.push(format!("{:>19} |{:>12}   Bayesian information criterion", "BIC", fmt_g(bic, 8)));
    out.push("-".repeat(66));
    r.set("aic", aic);
    r.set("bic", bic);
    st.r = r;
    Ok(out.join("\n"))
}

fn eform(f: &GsemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let b = f.bvec();
    let v = f.bv();
    let e = &f.eng;
    let want: Option<Vec<String>> = if cmd.main.split_whitespace().count() > 1 { Some(cmd.main.split_whitespace().skip(1).map(String::from).collect()) } else { None };
    let mut rows = Vec::new();
    let mut last = String::new();
    for (j, s) in e.slots.iter().enumerate() {
        if s.kind != PKind::Coef {
            continue;
        }
        if let Some(w) = &want {
            if !w.iter().any(|x| s.eq.ends_with(x.as_str())) {
                continue;
            }
        }
        let head = match s.class {
            Some(c) => format!("{}: {}", c + 1, s.eq),
            None => s.eq.clone(),
        };
        if head != last {
            rows.push(Row::Head(head.clone()));
            last = head;
        }
        let se = v[(j, j)].max(0.0).sqrt() * b[j].exp();
        rows.push(if s.p.is_none() { Row::Coef(s.name.clone(), b[j].exp(), None, false) } else { Row::Coef(s.name.clone(), b[j].exp(), Some(se), true) });
    }
    let level = util::level(cmd, st)?;
    let mut out = vec![String::new()];
    out.extend(tab::table(vtitle(&f.vce), level, "exp(b)", &rows));
    Ok(out.join("\n"))
}

/// `lcstats`: model-selection statistics for latent class models.
pub struct Lcstats;

impl CommandHandler for Lcstats {
    fn name(&self) -> &'static str {
        "lcstats"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut sets: Vec<(String, Results)> = Vec::new();
        let names: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        if names.is_empty() || names.iter().any(|n| n == ".") {
            sets.push((".".into(), st.e.clone()));
        }
        for n in names.iter().filter(|n| *n != ".") {
            if n == "_all" || n == "*" {
                for (k, r) in &st.estimates {
                    sets.push((k.clone(), r.clone()));
                }
                continue;
            }
            let r = st.estimates.get(n).ok_or_else(|| rc_err(111, format!("estimation result {n} not found")))?;
            sets.push((n.clone(), r.clone()));
        }
        let mut out = vec![String::new(), "-".repeat(78), format!("{:>12} |{:>8}{:>10}{:>14}{:>5}{:>13}{:>13}{:>9}", "Model", "N", "Classes", "ll(model)", "df", "AIC", "BIC", "Entropy"), format!("{}+{}", "-".repeat(13), "-".repeat(64))];
        let mut m = Vec::new();
        for (name, e) in &sets {
            let id: Option<u64> = e.macros.get("_w7c_id").and_then(|s| s.parse().ok());
            let mut tmp = Store::new();
            tmp.e = e.clone();
            let _ = id;
            let f = state::gsem(&tmp);
            let (n, ll, k) = (e.get("N").unwrap_or(SYSMISS), e.get("ll").unwrap_or(SYSMISS), e.get("rank").unwrap_or(SYSMISS));
            let kc = e.get("k_lclass").unwrap_or(1.0);
            let ent = match &f {
                Some(f) if f.eng.nclass > 1 => {
                    let post = f.eng.class_post(&f.th);
                    let nn = post.len() as f64;
                    let h: f64 = post.iter().flat_map(|p| p.iter()).filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
                    1.0 - h / (nn * (f.eng.nclass as f64).ln())
                }
                _ => SYSMISS,
            };
            let aic = -2.0 * ll + 2.0 * k;
            let bic = -2.0 * ll + k * n.ln();
            out.push(format!("{:>12} |{:>8}{:>10}{:>14}{:>5}{:>13}{:>13}{:>9}", tab::ab(name, 12), n, kc, fmt_g(ll, 10), k, fmt_g(aic, 10), fmt_g(bic, 10), fmt_g(ent, 6)));
            m.push(vec![n, kc, ll, k, aic, bic, ent]);
        }
        out.push("-".repeat(78));
        let mut mm = Matrix::from_rows(&m);
        mm.rownames = sets.iter().map(|s| s.0.clone()).collect();
        mm.colnames = ["N", "classes", "ll", "df", "AIC", "BIC", "entropy"].iter().map(|s| s.to_string()).collect();
        st.r = Results::new();
        st.r.set_mat("S", mm);
        Ok(out.join("\n"))
    }
}
