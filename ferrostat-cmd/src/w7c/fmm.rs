//! `fmm`: finite mixture models.
//!
//!   fmm # [if] [in] [, lcprob(varlist) lcinvariant(...) vce() ...]: estimator depvar indepvars [, options]
//!   fmm [, options]: (estimator ...) (estimator ...) ...
//!
//! Components: regress, logit, probit, cloglog, ologit, oprobit, mlogit,
//! poisson, nbreg, glm, betareg, intreg, tobit, truncreg, streg, tpoisson
//! and pointmass. The mixture is a latent-class model of the generalized
//! engine: EM iterations for starting values, then quasi-Newton with a
//! Newton polish. Postestimation: estat lcprob, estat lcmean, estat eform,
//! predict mu / pr / classpr / classposteriorpr / eta.

use std::collections::HashMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::engine::{Fam, Link, Outcome, PKind, TKind};
use super::gsem::{self, Data, GsemFit};
use super::path::{self, words};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::xmodels::Handlers;

pub type FmmFit = GsemFit;

pub fn handlers() -> Handlers {
    vec![("fmm", Box::new(Fmm))]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    gsem::predict(ds, st, cmd)
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    gsem::estat(ds, st, sub, cmd)
}

/// Split at the first top-level colon.
fn split_colon(s: &str) -> Option<(String, String)> {
    let mut d = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => d += 1,
            ')' => d -= 1,
            ':' if d == 0 => return Some((s[..i].to_string(), s[i + 1..].to_string())),
            _ => {}
        }
    }
    None
}

struct Comp {
    est: String,
    fam: Fam,
    link: Link,
    y: String,
    y2: Option<String>,
    xs: Vec<String>,
    opts: String,
}

fn opt_in(opts: &str, name: &str, min: usize) -> Option<String> {
    for w in words(&opts.replace(',', " ")) {
        let (n, a) = match w.find('(') {
            Some(i) => (w[..i].to_lowercase(), w[i + 1..w.len() - 1].to_string()),
            None => (w.to_lowercase(), String::new()),
        };
        if n.len() >= min && name.starts_with(&n) {
            return Some(a);
        }
    }
    None
}

fn parse_comp(ds: &DataSet, text: &str) -> Result<Comp, String> {
    let (body, opts) = path::split_comma(text);
    let mut w = body.split_whitespace();
    let est = w.next().ok_or_else(|| rc_err(198, "fmm: estimator required"))?.to_lowercase();
    let rest: Vec<String> = crate::varlist::expand(ds, &w.collect::<Vec<_>>().join(" "))?;
    let (fam, link) = match est.as_str() {
        "regress" | "reg" | "intreg" | "tobit" | "truncreg" => (Fam::Gaussian, Link::Identity),
        "logit" => (Fam::Bernoulli, Link::Logit),
        "probit" => (Fam::Bernoulli, Link::Probit),
        "cloglog" => (Fam::Bernoulli, Link::Cloglog),
        "ologit" => (Fam::Ordinal, Link::Logit),
        "oprobit" => (Fam::Ordinal, Link::Probit),
        "mlogit" => (Fam::Multinomial, Link::Logit),
        "poisson" | "tpoisson" => (Fam::Poisson, Link::Log),
        "nbreg" => (Fam::Nbinomial, Link::Log),
        "betareg" => (Fam::Beta, Link::Logit),
        "pointmass" => (Fam::Pointmass, Link::Identity),
        "glm" => {
            let f = gsem::parse_family(&opts)?;
            (f.fam, f.link)
        }
        "streg" => {
            let d = opt_in(&opts, "distribution", 1).unwrap_or_else(|| "weibull".into()).to_lowercase();
            let fam = if d.starts_with("e") { Fam::Exponential } else if d.starts_with("lognor") { Fam::Lognormal } else if d.starts_with("w") { Fam::Weibull } else { return Err(rc_err(198, format!("fmm: streg distribution({d}) not supported"))) };
            (fam, Link::Log)
        }
        "ivregress" => return Err(rc_err(198, "fmm: ivregress components are not supported")),
        o => return Err(rc_err(199, format!("fmm: {o} is not a supported component"))),
    };
    let (y, y2, xs) = if est == "streg" {
        (String::from("_t"), None, rest)
    } else if est == "intreg" {
        if rest.len() < 2 {
            return Err(rc_err(198, "intreg requires two dependent variables"));
        }
        (rest[0].clone(), Some(rest[1].clone()), rest[2..].to_vec())
    } else {
        let y = rest.first().cloned().ok_or_else(|| rc_err(198, "depvar required"))?;
        (y, None, rest[1..].to_vec())
    };
    Ok(Comp { est, fam, link, y, y2, xs, opts })
}

pub struct Fmm;

impl CommandHandler for Fmm {
    fn name(&self) -> &'static str {
        "fmm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let rest = raw.split_once(char::is_whitespace).map_or("", |x| x.1);
        let Some((left, right)) = split_colon(rest).or_else(|| raw.strip_prefix("fmm:").map(|r| (String::new(), r.to_string()))) else {
            let f = super::state::gsem(st).filter(|f| f.cmd == "fmm").ok_or_else(|| rc_err(301, "last estimates not found"))?;
            return Ok(gsem::display(&f, st, cmd).join("\n"));
        };
        // an if/in qualifier written on the component moves to the prefix
        let mut right = right.trim().to_string();
        let mut left = left;
        if !right.starts_with('(') {
            if let Ok(ic) = ferrostat_parser::parser::parse_line(&right) {
                if let Some(ifr) = &ic.if_raw {
                    let (body, opts) = path::split_comma(&right);
                    if let Some(pos) = body.find(" if ") {
                        let (lb, lo) = path::split_comma(&left);
                        left = format!("{lb} if {}{}{lo}", ifr.trim(), if lo.trim().is_empty() { "" } else { ", " });
                        right = format!("{}{}{}", &body[..pos], if opts.trim().is_empty() { "" } else { ", " }, opts);
                    }
                }
            }
        }
        let fc = ferrostat_parser::parser::parse_line(&format!("fmm {left}")).map_err(|e| rc_err(198, e.message))?;
        let right = right.trim();
        let comps: Vec<Comp> = if right.starts_with('(') {
            path::groups(right).iter().map(|g| parse_comp(ds, g)).collect::<Result<_, _>>()?
        } else {
            let k: usize = fc.main.trim().parse().map_err(|_| rc_err(198, "fmm: number of components required"))?;
            (0..k).map(|_| parse_comp(ds, right)).collect::<Result<_, _>>()?
        };
        if !right.starts_with('(') && comps.is_empty() {
            return Err(rc_err(198, "fmm: at least one component required"));
        }
        let k = comps.len();
        let mut dt = Data { cols: vec![], index: HashMap::new() };
        for c in &comps {
            dt.col(&c.y);
            if let Some(y2) = &c.y2 {
                dt.col(y2);
            }
            for x in &c.xs {
                dt.col(x);
            }
            if c.est == "streg" {
                dt.col("_d");
            }
            if let Some(e) = opt_in(&c.opts, "exposure", 3) {
                dt.col(&format!("__ln:{}", e.trim()));
            }
            if let Some(o) = opt_in(&c.opts, "offset", 3) {
                dt.col(o.trim());
            }
        }
        let lcp: Vec<String> = match fc.opt("lcprob", 3) {
            Some(t) => crate::varlist::expand(ds, t)?,
            None => vec![],
        };
        for z in &lcp {
            dt.col(z);
        }
        let mut mask = util::touse(ds, st, &fc)?;
        let allow: Vec<String> = comps.iter().filter_map(|c| c.y2.clone()).chain(comps.iter().filter(|c| c.y2.is_some()).map(|c| c.y.clone())).collect();
        let (data, obs) = gsem::load(ds, &dt.cols, &mut mask, &allow)?;
        if data.len() < 3 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let mut e = gsem::empty_engine();
        e.data = data;
        e.cols = dt.cols.clone();
        e.nclass = k;
        e.class_name = "Class".into();
        gsem::make_units(&mut e, None);
        let mut labels = HashMap::new();
        let mut descr = Vec::new();
        for (c, comp) in comps.iter().enumerate() {
            let ycol = dt.index[&comp.y];
            let mut o = Outcome::new(&comp.y, ycol, comp.fam, comp.link);
            o.y2col = comp.y2.as_ref().map(|v| dt.index[v]);
            if comp.est == "streg" {
                o.fail = Some(dt.index["_d"]);
            }
            let num = |n: &str| opt_in(&comp.opts, n, 2).and_then(|v| v.trim().parse::<f64>().ok());
            match comp.est.as_str() {
                "tobit" => o.cens = (num("ll"), num("ul")),
                "truncreg" => o.trunc = (num("ll"), num("ul")),
                "tpoisson" => o.trunc = (Some(num("ll").unwrap_or(0.0)), None),
                "pointmass" => o.pm = num("lbvalue").unwrap_or(0.0),
                _ => {}
            }
            if comp.est == "tobit" && o.cens.0.is_none() && o.cens.1.is_none() {
                // tobit, ll without a value: censoring at the minimum
                if opt_in(&comp.opts, "ll", 2).is_some() {
                    o.cens.0 = e.data.iter().map(|r| r[ycol]).fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.min(v))));
                }
            }
            o.offset = opt_in(&comp.opts, "exposure", 3).map(|x| dt.index[&format!("__ln:{}", x.trim())]).or(opt_in(&comp.opts, "offset", 3).map(|x| dt.index[x.trim()]));
            let preds: Vec<(String, TKind, path::Coef)> = comp.xs.iter().map(|x| (x.clone(), TKind::X(dt.index[x]), path::Coef::default())).collect();
            let nocons = opt_in(&comp.opts, "noconstant", 6).is_some();
            let ys: Vec<f64> = e.data.iter().map(|r| r[ycol]).collect();
            let sd = super::engine::var_of(&ys).sqrt();
            let z = c as f64 - (k as f64 - 1.0) / 2.0;
            let shift = if k == 1 {
                0.0
            } else {
                match comp.fam {
                    Fam::Gaussian | Fam::Lognormal => z * sd,
                    Fam::Poisson | Fam::Nbinomial | Fam::Gamma | Fam::Weibull | Fam::Exponential => z * 0.7,
                    _ => z * 1.5,
                }
            };
            gsem::add_outcome(&mut e, o, &preds, nocons, Some(c), None, &mut labels, shift);
            descr.push(format!("{} {}", comp.est, comp.y));
        }
        if k == 1 {
            // a single class: no class model
            for s in e.slots.iter_mut() {
                s.class = None;
            }
            for o in e.outcomes.iter_mut() {
                o.class = None;
            }
            e.nclass = 1;
        } else {
            for c in 1..k {
                let eq = format!("{}.Class", c + 1);
                for z in &lcp {
                    let s = e.add_slot(&eq, z, None, PKind::ClassCoef, None, 0.0);
                    e.cterms.push((c, TKind::X(dt.index[z]), s));
                }
                let s = e.add_slot(&eq, "_cons", None, PKind::ClassCoef, None, 0.0);
                e.cterms.push((c, TKind::One, s));
            }
            // lcinvariant(): equal parameters across classes
            if let Some(li) = fc.opt("lcinvariant", 4) {
                let li = li.to_lowercase();
                let want = |s: &super::engine::PSlot| -> bool {
                    li.contains("all")
                        || (li.contains("coef") && s.kind == PKind::Coef && s.name != "_cons")
                        || (li.contains("cons") && s.kind == PKind::Coef && s.name == "_cons")
                        || (li.contains("errvar") && s.kind == PKind::Aux && s.name.starts_with("var("))
                        || (li.contains("scale") && s.kind == PKind::Aux && !s.name.starts_with("var("))
                        || (li.contains("cuts") && s.kind == PKind::Cut)
                };
                let n = e.slots.len();
                for a in 0..n {
                    if e.slots[a].class != Some(0) || !want(&e.slots[a]) {
                        continue;
                    }
                    for b in 0..n {
                        let (sa, sb) = (&e.slots[a], &e.slots[b]);
                        if sb.class.is_some_and(|c| c > 0) && sb.eq == sa.eq && sb.name == sa.name && sb.kind == sa.kind {
                            e.tie(a, b);
                        }
                    }
                }
            }
        }
        let mut f = gsem::estimate(e, st, &fc, obs, ds.n_obs, "fmm", &format!("Finite mixture model"))?;
        f.comps = descr;
        let mut out = if cmd.raw.contains("nolog") { vec![] } else { gsem::LOG.with(|l| std::mem::take(&mut *l.borrow_mut())) };
        out.extend(gsem::display(&f, st, &fc));
        gsem::post(st, f, cmd);
        st.e.set_str("cmd", "fmm");
        Ok(out.join("\n"))
    }
}
