//! Item response theory: `irt 1pl|2pl|3pl|grm|nrm|pcm|gpcm|rsm|hybrid`,
//! `group()` for multiple-group models, `estat report`, `estat greport`,
//! `irtgraph icc|iif|tif|tcc`, and the differential item functioning
//! tests `difmh` (Mantel-Haenszel) and `diflogistic`.
//!
//! The models are special cases of the generalized engine: one standard
//! normal latent trait, items as Bernoulli, ordinal, multinomial or
//! adjacent-category outcomes. Estimates are reported in the IRT metric
//! (discrimination and difficulty) with delta-method standard errors.

use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::engine::{self, Engine, Fam, Link, Outcome, PKind, TKind};
use super::gsem::{self, GsemFit};
use super::opt;
use super::path::{self, groups as pgroups};
use super::tab::{self, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    vec![("irt", Box::new(Irt)), ("irtgraph", Box::new(Irtgraph)), ("difmh", Box::new(Difmh)), ("diflogistic", Box::new(Diflogistic))]
}

#[derive(Clone, Debug)]
pub struct IrtInfo {
    /// (item, model)
    pub items: Vec<(String, String)>,
    pub gvar: Option<String>,
    pub glabels: Vec<String>,
}

fn sub_cns(opts: &str) -> Option<String> {
    let i = opts.find("cns(")?;
    Some(opts[i + 4..].split(')').next().unwrap_or("").to_string())
}

fn model_title(m: &str) -> &'static str {
    match m {
        "1pl" => "One-parameter logistic model",
        "2pl" => "Two-parameter logistic model",
        "3pl" => "Three-parameter logistic model",
        "grm" => "Graded response model",
        "nrm" => "Nominal response model",
        "pcm" => "Partial credit model",
        "gpcm" => "Generalized partial credit model",
        "rsm" => "Rating scale model",
        _ => "Hybrid IRT model",
    }
}

pub struct Irt;

impl CommandHandler for Irt {
    fn name(&self) -> &'static str {
        "irt"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            let f = super::state::gsem(st).filter(|f| f.cmd == "irt").ok_or_else(|| rc_err(301, "last estimates not found"))?;
            return Ok(report(&f, st, cmd)?.join("\n"));
        }
        let (model, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let model = model.to_lowercase();
        if model == "constraints" {
            return Err(rc_err(198, "use constraint define and irt ..., cns()"));
        }
        let mut blocks: Vec<(String, Vec<String>, String)> = Vec::new();
        if model == "hybrid" {
            for g in pgroups(rest) {
                let (body, opts) = path::split_comma(&g);
                let (m, vars) = body.trim().split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "irt hybrid: (model varlist) required"))?;
                blocks.push((m.to_lowercase(), crate::varlist::expand(ds, vars)?, opts));
            }
        } else {
            blocks.push((model.clone(), crate::varlist::expand(ds, rest)?, String::new()));
        }
        for (m, v, _) in &blocks {
            if !matches!(m.as_str(), "1pl" | "2pl" | "3pl" | "grm" | "nrm" | "pcm" | "gpcm" | "rsm") {
                return Err(rc_err(198, format!("irt {m} not a valid model")));
            }
            if v.is_empty() {
                return Err(rc_err(198, "varlist required"));
            }
        }
        let f = fit(ds, st, cmd, &blocks, if model == "hybrid" { "hybrid" } else { &model })?;
        let mut out = if cmd.raw.contains("nolog") { vec![] } else { gsem::LOG.with(|l| std::mem::take(&mut *l.borrow_mut())) };
        let rep = report(&f, st, cmd)?;
        out.extend(rep);
        gsem::post(st, f, cmd);
        st.e.set_str("cmd", "irt");
        st.e.set_str("model1", model.clone());
        st.e.set_str("irt_model", model);
        Ok(out.join("\n"))
    }
}

fn fit(ds: &DataSet, st: &Store, cmd: &Command, blocks: &[(String, Vec<String>, String)], title_model: &str) -> Result<GsemFit, String> {
    let mut items: Vec<(String, String, String)> = Vec::new();
    for (m, v, o) in blocks {
        for x in v {
            items.push((x.clone(), m.clone(), o.clone()));
        }
    }
    let mut cols: Vec<String> = items.iter().map(|i| i.0.clone()).collect();
    let gvar = cmd.opt("group", 2).map(|g| crate::varlist::resolve_name(ds, g.trim())).transpose()?;
    if let Some(g) = &gvar {
        cols.push(g.clone());
    }
    let mut mask = util::touse(ds, st, cmd)?;
    let listwise = cmd.has_opt("listwise", 4);
    let allow: Vec<String> = if listwise { vec![] } else { items.iter().map(|i| i.0.clone()).collect() };
    let (data, obs) = gsem::load(ds, &cols, &mut mask, &allow)?;
    // drop observations with all items missing
    let keep: Vec<usize> = (0..data.len()).filter(|&r| items.iter().enumerate().any(|(c, _)| !data[r][c].is_nan())).collect();
    let data: Vec<Vec<f64>> = keep.iter().map(|&r| data[r].clone()).collect();
    let obs: Vec<usize> = keep.iter().map(|&r| obs[r]).collect();
    if data.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let mut e = gsem::empty_engine();
    e.data = data;
    e.cols = cols.clone();
    e.class_name = "C".into();
    e.nq = util::opt_num(cmd, "intpoints", 4)?.map(|v| v as usize).unwrap_or(7).clamp(1, 30);
    gsem::make_units(&mut e, None);
    let mut glabels = vec![];
    let ng = if let Some(_g) = &gvar {
        let gc = cols.len() - 1;
        let lv = gsem::levels_of(&e.data, gc);
        glabels = lv.iter().map(|v| fmt_g(*v, 12)).collect();
        e.ugroup = e.units.iter().map(|u| lv.iter().position(|v| *v == e.data[u[0]][gc]).unwrap()).collect();
        lv.len()
    } else {
        1
    };
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut shared_a: HashMap<String, usize> = HashMap::new();
    let mut shared_c: Option<usize> = None;
    let mut rsm_cuts: Vec<usize> = Vec::new();
    let sepg = cmd.has_opt("sepguessing", 4);
    for (c, (item, m, _)) in items.iter().enumerate() {
        let (fam, link) = match m.as_str() {
            "1pl" | "2pl" | "3pl" => (Fam::Bernoulli, Link::Logit),
            "grm" => (Fam::Ordinal, Link::Logit),
            "nrm" => (Fam::Multinomial, Link::Logit),
            _ => (Fam::Pcm, Link::Logit),
        };
        let mut o = Outcome::new(item, c, fam, link);
        let ys: Vec<f64> = e.data.iter().map(|r| r[c]).collect();
        if fam == Fam::Bernoulli && ys.iter().any(|y| !y.is_nan() && *y != 0.0 && *y != 1.0) {
            return Err(rc_err(450, format!("{item} is not a 0/1 variable")));
        }
        o.levels = gsem::levels_of(&e.data, c);
        if o.levels.len() < 2 {
            return Err(rc_err(198, format!("{item} does not vary")));
        }
        let ncut = o.levels.len() - 1;
        let pre = |e: &mut Engine, nm: &str, st: f64| e.add_slot(item, nm, None, PKind::Coef, None, st);
        // constraints: block cns() and the global cns()
        let mut cns: HashMap<char, path::Coef> = HashMap::new();
        for src in [cmd.opt("cns", 3).map(String::from), sub_cns(&items[c].2)] {
            for w in src.unwrap_or_default().split_whitespace() {
                if let Some((k, v)) = w.split_once('@') {
                    if let Some(ch) = k.trim().chars().next() {
                        cns.insert(ch.to_ascii_lowercase(), path::Coef::parse(v));
                    }
                }
            }
        }
        match m.as_str() {
            "nrm" => {
                o.neta = ncut;
                for j in 1..o.levels.len() {
                    let eq = format!("{}.{item}", fmt_g(o.levels[j], 12));
                    let a = e.add_slot(&eq, "Theta", None, PKind::Coef, None, 0.5 * j as f64);
                    let b = e.add_slot(&eq, "_cons", None, PKind::Coef, None, 0.0);
                    o.terms.push(engine::Term { kind: TKind::Lat(0), eta: j - 1, slot: a });
                    o.terms.push(engine::Term { kind: TKind::One, eta: j - 1, slot: b });
                }
            }
            _ => {
                let ac = cns.get(&'a').cloned().unwrap_or_default();
                let a = e.add_slot(item, "Theta", None, PKind::Coef, ac.fix, ac.fix.unwrap_or(1.0));
                if let Some(l) = &ac.label {
                    match labels.get(l) {
                        Some(&f) => e.tie(f, a),
                        None => {
                            labels.insert(l.clone(), a);
                        }
                    }
                } else if ac.fix.is_none() && (m == "1pl" || m == "pcm" || m == "rsm") {
                    match shared_a.get(m.as_str()) {
                        Some(&f) => e.tie(f, a),
                        None => {
                            shared_a.insert(m.clone(), a);
                        }
                    }
                }
                o.terms.push(engine::Term { kind: TKind::Lat(0), eta: 0, slot: a });
                match fam {
                    Fam::Bernoulli => {
                        let bc = cns.get(&'b').and_then(|c| c.fix).filter(|v| *v == 0.0);
                        let s = e.add_slot(item, "_cons", None, PKind::Coef, bc, engine::start_cons(&o, &ys));
                        o.terms.push(engine::Term { kind: TKind::One, eta: 0, slot: s });
                        if m == "3pl" {
                            let cf = cns.get(&'c').and_then(|c| c.fix).map(|v| (v / (1.0 - v)).ln());
                            let g = e.add_slot(&format!("/{item}"), "logit_c", None, PKind::Aux, cf, cf.unwrap_or(-2.0));
                            if !sepg && cf.is_none() {
                                match shared_c {
                                    Some(f) => e.tie(f, g),
                                    None => shared_c = Some(g),
                                }
                            }
                            o.guess = Some(g);
                        }
                    }
                    Fam::Ordinal => {
                        let n = ys.iter().filter(|v| !v.is_nan()).count().max(1) as f64;
                        let mut cum = 0.0;
                        for j in 0..ncut {
                            cum += ys.iter().filter(|v| **v == o.levels[j]).count() as f64 / n;
                            let p = cum.clamp(0.02, 0.98);
                            o.aux.push(e.add_slot(&format!("/{item}"), &format!("cut{}", j + 1), None, PKind::Cut, None, (p / (1.0 - p)).ln()));
                        }
                    }
                    _ => {
                        if m == "rsm" {
                            let s = pre(&mut e, "_cons", 0.0);
                            o.terms.push(engine::Term { kind: TKind::One, eta: 0, slot: s });
                            if rsm_cuts.is_empty() {
                                for j in 0..ncut {
                                    rsm_cuts.push(e.add_slot("/rsm", &format!("cut{}", j + 1), None, PKind::Cut, if j == 0 { Some(0.0) } else { None }, j as f64 * 0.5));
                                }
                            }
                            if rsm_cuts.len() != ncut {
                                return Err(rc_err(198, "rsm items must have the same number of categories"));
                            }
                            o.aux = rsm_cuts.clone();
                        } else {
                            for j in 0..ncut {
                                o.aux.push(e.add_slot(&format!("/{item}"), &format!("cut{}", j + 1), None, PKind::Cut, None, (j as f64 - (ncut as f64 - 1.0) / 2.0) * 0.8));
                            }
                        }
                    }
                }
            }
        }
        e.outcomes.push(o);
    }
    let _ = &mut labels;
    let varc: HashMap<String, path::Coef> = [("Theta".to_string(), path::Coef { fix: Some(1.0), label: None })].into_iter().collect();
    gsem::add_latents(&mut e, &["Theta".into()], &varc, &HashMap::new(), true, ng, ng > 1);
    let obs_n = ds.n_obs;
    let mut f = gsem::estimate(e, st, cmd, obs, obs_n, "irt", model_title(title_model))?;
    f.irt = Some(IrtInfo { items: items.iter().map(|i| (i.0.clone(), i.1.clone())).collect(), gvar, glabels });
    Ok(f)
}

/// IRT-metric parameters: (item, label, value) as a function of theta.
fn irt_params(f: &GsemFit, th: &[f64]) -> Vec<(String, String, f64)> {
    let e = &f.eng;
    let sv = e.values(th);
    let info = f.irt.as_ref().unwrap();
    let mut out = Vec::new();
    for (o, (item, m)) in e.outcomes.iter().zip(&info.items) {
        let slot = |eta: usize, k: TKind| o.terms.iter().find(|t| t.eta == eta && t.kind == k).map(|t| sv[t.slot]);
        match o.fam {
            Fam::Multinomial => {
                for j in 1..o.levels.len() {
                    let a = slot(j - 1, TKind::Lat(0)).unwrap_or(0.0);
                    let l = format!("{} vs {}", fmt_g(o.levels[j], 8), fmt_g(o.levels[0], 8));
                    out.push((item.clone(), format!("Discrim {l}"), a));
                }
                for j in 1..o.levels.len() {
                    let a = slot(j - 1, TKind::Lat(0)).unwrap_or(1.0);
                    let c = slot(j - 1, TKind::One).unwrap_or(0.0);
                    let l = format!("{} vs {}", fmt_g(o.levels[j], 8), fmt_g(o.levels[0], 8));
                    out.push((item.clone(), format!("Diff {l}"), -c / a));
                }
            }
            _ => {
                let a = slot(0, TKind::Lat(0)).unwrap_or(1.0);
                out.push((item.clone(), "Discrim".into(), a));
                match o.fam {
                    Fam::Bernoulli => {
                        let c = slot(0, TKind::One).unwrap_or(0.0);
                        out.push((item.clone(), "Diff".into(), -c / a));
                        if let Some(g) = o.guess {
                            out.push((item.clone(), "Guess".into(), dist::invlogit(sv[g])));
                        }
                    }
                    Fam::Ordinal => {
                        for (j, &s) in o.aux.iter().enumerate() {
                            out.push((item.clone(), format!("Diff >={}", fmt_g(o.levels[j + 1], 8)), sv[s] / a));
                        }
                    }
                    _ => {
                        let loc = slot(0, TKind::One).unwrap_or(0.0);
                        for (j, &s) in o.aux.iter().enumerate() {
                            out.push((item.clone(), format!("Diff {} vs {}", fmt_g(o.levels[j + 1], 8), fmt_g(o.levels[j], 8)), (sv[s] - loc) / a));
                        }
                    }
                }
                let _ = m;
            }
        }
    }
    out
}

fn report(f: &GsemFit, st: &mut Store, cmd: &Command) -> Result<Vec<String>, String> {
    let ps = irt_params(f, &f.th);
    let j = opt::jacobian(&|t: &[f64]| irt_params(f, t).into_iter().map(|x| x.2).collect(), &f.th);
    let v = opt::sandwich_jv(&j, &f.v);
    let mut order: Vec<usize> = (0..ps.len()).collect();
    let items: Vec<String> = {
        let mut it: Vec<String> = Vec::new();
        for p in &ps {
            if !it.contains(&p.0) {
                it.push(p.0.clone());
            }
        }
        if let Some(s) = cmd.opt("sort", 2) {
            let key = s.trim().to_lowercase();
            let lab = if key.starts_with('a') { "Discrim" } else { "Diff" };
            let val = |i: &String| ps.iter().find(|p| p.0 == *i && p.1.starts_with(lab)).map_or(0.0, |p| p.2);
            it.sort_by(|a, b| val(a).partial_cmp(&val(b)).unwrap_or(std::cmp::Ordering::Equal));
        }
        it
    };
    if cmd.has_opt("byparm", 3) {
        order.sort_by_key(|&k| (ps[k].1.split_whitespace().next().unwrap_or("").to_string(), items.iter().position(|i| *i == ps[k].0)));
    } else {
        order.sort_by_key(|&k| items.iter().position(|i| *i == ps[k].0));
    }
    let mut rows = Vec::new();
    let mut last = String::new();
    let shared_cns = |lab: &str| lab == "Discrim" && f.irt.as_ref().is_some_and(|i| i.items.iter().all(|x| x.1 == "1pl" || x.1 == "pcm" || x.1 == "rsm"));
    let _ = shared_cns;
    for &k in &order {
        let (item, lab, val) = &ps[k];
        let head = if cmd.has_opt("byparm", 3) { lab.split_whitespace().next().unwrap_or("").to_string() } else { item.clone() };
        if head != last {
            if !last.is_empty() {
                rows.push(Row::Rule);
            }
            rows.push(Row::Head(head.clone()));
            last = head;
        }
        let label = if cmd.has_opt("byparm", 3) { item.clone() } else { lab.clone() };
        let se = v[(k, k)].max(0.0).sqrt();
        rows.push(Row::Coef(label, *val, if se > 0.0 { Some(se) } else { None }, true));
    }
    // latent distribution by group
    let e = &f.eng;
    let sv = e.values(&f.th);
    if e.phi.len() > 1 {
        let info = f.irt.as_ref().unwrap();
        rows.push(Row::Rule);
        for g in 1..e.phi.len() {
            rows.push(Row::Head(format!("{}.{}", info.glabels[g], info.gvar.clone().unwrap_or_default())));
            if let Some(s) = e.lmean[g][0] {
                let se = f.v[(e.slots[s].p.unwrap(), e.slots[s].p.unwrap())].sqrt();
                rows.push(Row::Coef("mean(Theta)".into(), sv[s], Some(se), true));
            }
            if let Some(s) = e.phi[g][0] {
                match e.slots[s].p {
                    Some(p) => rows.push(Row::Coef("var(Theta)".into(), sv[s], Some(f.v[(p, p)].sqrt()), false)),
                    None => rows.push(Row::Coef("var(Theta)".into(), sv[s], None, false)),
                }
            }
        }
    }
    let mut out = vec![String::new()];
    out.extend(tab::header(&f.title, &[("Number of obs".to_string(), tab::n_str(f.n))]));
    out.push(format!("Log likelihood = {}", fmt_g(f.ll, 10)));
    out.push(String::new());
    let level = util::level(cmd, st).unwrap_or(f.level);
    out.extend(tab::table(gsem::vtitle(&f.vce), level, "Coefficient", &rows));
    let names: Vec<String> = ps.iter().map(|p| format!("{}:{}", p.0, p.1.replace(' ', "_"))).collect();
    let b: Vec<f64> = ps.iter().map(|p| p.2).collect();
    st.r = Results::new();
    st.r.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    st.r.set_mat("V", v.with_names(names.clone(), names));
    tab::keep_table(&mut st.r);
    Ok(out)
}

pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command, f: &GsemFit) -> Option<Result<String, String>> {
    match sub {
        "report" => Some(report(f, st, cmd).map(|v| v.join("\n"))),
        "greport" => Some(greport(f, st, cmd)),
        _ => None,
    }
}

fn greport(f: &GsemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let e = &f.eng;
    let info = f.irt.as_ref().unwrap();
    if e.phi.len() < 2 {
        return Err(rc_err(198, "estat greport requires a model fitted with group()"));
    }
    let mut out = report(f, st, cmd)?;
    out.push(String::new());
    out.push(format!("Latent trait distribution by {}", info.gvar.clone().unwrap_or_default()));
    let sv = e.values(&f.th);
    for g in 0..e.phi.len() {
        let mean = e.lmean[g][0].map_or(0.0, |s| sv[s]);
        let var = e.phi[g][0].map_or(1.0, |s| sv[s]);
        out.push(format!("  {:>8}: mean(Theta) = {:>10}   var(Theta) = {:>10}", info.glabels[g], fmt_g(mean, 8), fmt_g(var, 8)));
    }
    Ok(out.join("\n"))
}

pub fn predict(_ds: &mut DataSet, _st: &mut Store, _cmd: &Command, _f: &GsemFit) -> Option<Result<String, String>> {
    None
}

// ---------------------------------------------------------------------------
// irtgraph
// ---------------------------------------------------------------------------

/// Category probabilities of item `o` at trait value t (group 1 metric).
fn probs_at(f: &GsemFit, o: &Outcome, t: f64) -> Vec<f64> {
    let row = f.eng.data[0].clone();
    if o.fam == Fam::Bernoulli {
        let p = f.eng.mean(o, &row, &[t], &f.th);
        return vec![1.0 - p, p];
    }
    f.eng.cat_probs(o, &row, &[t], &f.th)
}

fn item_info(f: &GsemFit, o: &Outcome, t: f64) -> f64 {
    let h = 1e-4;
    let p0 = probs_at(f, o, t);
    let p1 = probs_at(f, o, t + h);
    let pm = probs_at(f, o, t - h);
    p0.iter().enumerate().map(|(k, p)| {
        let d = (p1[k] - pm[k]) / (2.0 * h);
        if *p > 1e-12 { d * d / p } else { 0.0 }
    }).sum()
}

pub struct Irtgraph;

impl CommandHandler for Irtgraph {
    fn name(&self) -> &'static str {
        "irtgraph"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let f = super::state::gsem(st).filter(|f| f.cmd == "irt").ok_or_else(|| rc_err(301, "irtgraph requires irt estimates"))?;
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub = sub.to_lowercase();
        let e = &f.eng;
        let want: Vec<&Outcome> = if rest.trim().is_empty() {
            e.outcomes.iter().collect()
        } else {
            let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
            e.outcomes.iter().filter(|o| names.contains(&o.name)).collect()
        };
        let (lo, hi) = match cmd.opt("range", 1) {
            Some(r) => {
                let v: Vec<f64> = r.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                (v.first().copied().unwrap_or(-4.0), v.get(1).copied().unwrap_or(4.0))
            }
            None => (-4.0, 4.0),
        };
        let grid: Vec<f64> = (0..=100).map(|i| lo + (hi - lo) * i as f64 / 100.0).collect();
        let mut g = Graph { legend: true, ..Default::default() };
        g.xtitle = "Theta".into();
        match sub.as_str() {
            "icc" => {
                for o in &want {
                    let nc = if o.fam == Fam::Bernoulli { 2 } else { o.levels.len() };
                    for k in (if o.fam == Fam::Bernoulli { 1 } else { 0 })..nc {
                        let pts: Vec<(f64, f64)> = grid.iter().map(|t| (*t, probs_at(&f, o, *t)[k])).collect();
                        let lab = if o.fam == Fam::Bernoulli { format!("Pr({}=1)", o.name) } else { format!("Pr({}={})", o.name, fmt_g(o.levels[k], 8)) };
                        g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: lab, ..Default::default() });
                    }
                }
                g.title = "Item characteristic curves".into();
                g.ytitle = "Probability".into();
            }
            "iif" => {
                for o in &want {
                    let pts: Vec<(f64, f64)> = grid.iter().map(|t| (*t, item_info(&f, o, *t))).collect();
                    g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: o.name.clone(), ..Default::default() });
                }
                g.title = "Item information functions".into();
                g.ytitle = "Information".into();
            }
            "tif" => {
                let pts: Vec<(f64, f64)> = grid.iter().map(|t| (*t, want.iter().map(|o| item_info(&f, o, *t)).sum())).collect();
                if cmd.has_opt("se", 2) {
                    let se: Vec<(f64, f64)> = pts.iter().map(|(t, i)| (*t, 1.0 / i.max(1e-12).sqrt())).collect();
                    g.layers.push(Layer { kind: LayerKind::Line, points: se, label: "Standard error".into(), ..Default::default() });
                }
                g.layers.insert(0, Layer { kind: LayerKind::Line, points: pts, label: "Test information".into(), ..Default::default() });
                g.title = "Test information function".into();
                g.ytitle = "Information".into();
            }
            "tcc" => {
                let pts: Vec<(f64, f64)> = grid
                    .iter()
                    .map(|t| {
                        (*t, want.iter().map(|o| {
                            let p = probs_at(&f, o, *t);
                            if o.fam == Fam::Bernoulli { p[1] } else { o.levels.iter().zip(&p).map(|(l, q)| l * q).sum() }
                        }).sum())
                    })
                    .collect();
                g.layers.push(Layer { kind: LayerKind::Line, points: pts, label: "Expected score".into(), ..Default::default() });
                g.title = "Test characteristic curve".into();
                g.ytitle = "Expected score".into();
            }
            o => return Err(rc_err(198, format!("irtgraph {o} not recognized"))),
        }
        g.command = format!("irtgraph {main}");
        crate::graph::finish(&mut g, cmd, st)
    }
}

// ---------------------------------------------------------------------------
// differential item functioning
// ---------------------------------------------------------------------------

fn dif_data(ds: &DataSet, st: &Store, cmd: &Command) -> Result<(Vec<String>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64), String> {
    let items = crate::varlist::expand(ds, &cmd.main)?;
    let gv = crate::varlist::resolve_name(ds, cmd.opt("group", 2).ok_or_else(|| rc_err(198, "option group() required"))?.trim())?;
    let mut mask = util::touse(ds, st, cmd)?;
    let mut need = items.clone();
    need.push(gv.clone());
    let tv = cmd.opt("total", 2).map(|t| crate::varlist::resolve_name(ds, t.trim())).transpose()?;
    if let Some(t) = &tv {
        need.push(t.clone());
    }
    util::markout(ds, &mut mask, &need);
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let cols: Vec<Vec<f64>> = items.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
    for (v, c) in items.iter().zip(&cols) {
        if c.iter().any(|x| *x != 0.0 && *x != 1.0) {
            return Err(rc_err(450, format!("{v} is not a 0/1 variable")));
        }
    }
    let gc = util::col(ds, &gv)?;
    let g: Vec<f64> = rows.iter().map(|&i| gc[i]).collect();
    let mut lv = g.clone();
    lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lv.dedup();
    if lv.len() != 2 {
        return Err(rc_err(450, "group() must identify exactly two groups"));
    }
    let focal = util::opt_num(cmd, "focal", 3)?.unwrap_or(lv[1]);
    let total: Vec<f64> = match &tv {
        Some(t) => {
            let c = util::col(ds, t)?;
            rows.iter().map(|&i| c[i]).collect()
        }
        None => (0..rows.len()).map(|r| cols.iter().map(|c| c[r]).sum()).collect(),
    };
    let _ = is_missing;
    Ok((items, cols, g, total, focal))
}

pub struct Difmh;

impl CommandHandler for Difmh {
    fn name(&self) -> &'static str {
        "difmh"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (items, cols, g, total, focal) = dif_data(ds, st, cmd)?;
        let yates = !cmd.has_opt("noyates", 3);
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut strata: Vec<f64> = total.clone();
        strata.sort_by(|a, b| a.partial_cmp(b).unwrap());
        strata.dedup();
        let mut out = vec![String::new(), "Mantel-Haenszel DIF analysis".to_string(), String::new(), "-".repeat(76), format!("{:>12} |{:>10}{:>10}{:>14}{:>24}", "Item", "Chi2", "Prob.", "Odds Ratio", format!("[{}% Conf. Interval]", fmt_g(level, 4))), format!("{}+{}", "-".repeat(13), "-".repeat(62))];
        let mut res = Vec::new();
        for (it, c) in items.iter().zip(&cols) {
            let (mut sa, mut se, mut sv) = (0.0f64, 0.0f64, 0.0f64);
            let (mut num, mut den) = (0.0, 0.0);
            // Robins-Breslow-Greenland variance pieces
            let (mut pr, mut ps_, mut qs) = (0.0, 0.0, 0.0);
            for s in &strata {
                // reference group (non-focal) is the first row
                let (mut a, mut b, mut cc, mut d) = (0.0, 0.0, 0.0, 0.0);
                for r in 0..c.len() {
                    if total[r] != *s {
                        continue;
                    }
                    let foc = g[r] == focal;
                    match (foc, c[r] == 1.0) {
                        (false, true) => a += 1.0,
                        (false, false) => b += 1.0,
                        (true, true) => cc += 1.0,
                        (true, false) => d += 1.0,
                    }
                }
                let n = a + b + cc + d;
                if n < 2.0 || (a + b) == 0.0 || (cc + d) == 0.0 || (a + cc) == 0.0 || (b + d) == 0.0 {
                    continue;
                }
                sa += a;
                se += (a + b) * (a + cc) / n;
                sv += (a + b) * (cc + d) * (a + cc) * (b + d) / (n * n * (n - 1.0));
                let (r_, s_) = (a * d / n, b * cc / n);
                num += r_;
                den += s_;
                let (p, q) = ((a + d) / n, (b + cc) / n);
                pr += p * r_;
                ps_ += p * s_ + q * r_;
                qs += q * s_;
            }
            let dev = (sa - se).abs() - if yates { 0.5 } else { 0.0 };
            let chi2 = if sv > 0.0 { dev.max(0.0).powi(2) / sv } else { SYSMISS };
            let p = if chi2 == SYSMISS { SYSMISS } else { dist::chi2tail(1.0, chi2) };
            let or = if den > 0.0 { num / den } else { SYSMISS };
            let vl = pr / (2.0 * num * num) + ps_ / (2.0 * num * den) + qs / (2.0 * den * den);
            let (lo, hi) = if or != SYSMISS && or > 0.0 { ((or.ln() - z * vl.sqrt()).exp(), (or.ln() + z * vl.sqrt()).exp()) } else { (SYSMISS, SYSMISS) };
            out.push(format!("{:>12} |{:>10}{:>10.4}{:>14}{:>12}{:>12}", tab::ab(it, 12), fmt_g(chi2, 7), p, fmt_g(or, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
            res.push(vec![chi2, p, or, lo, hi]);
        }
        out.push("-".repeat(76));
        let mut m = Matrix::from_rows(&res);
        m.rownames = items;
        m.colnames = ["chi2", "p", "OR", "lower", "upper"].iter().map(|s| s.to_string()).collect();
        st.r = Results::new();
        st.r.set_mat("dif", m);
        Ok(out.join("\n"))
    }
}

/// Logistic regression by Newton-Raphson; returns (beta, log likelihood).
fn logit_fit(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    let k = x[0].len();
    let mut b = vec![0.0; k];
    let mut ll = 0.0;
    for _ in 0..50 {
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        ll = 0.0;
        for (r, xi) in x.iter().enumerate() {
            let e: f64 = xi.iter().zip(&b).map(|(a, c)| a * c).sum();
            let p = dist::invlogit(e);
            ll += if y[r] == 1.0 { p.max(1e-300).ln() } else { (1.0 - p).max(1e-300).ln() };
            for i in 0..k {
                g[i] += (y[r] - p) * xi[i];
                for j in 0..k {
                    h[(i, j)] += p * (1.0 - p) * xi[i] * xi[j];
                }
            }
        }
        let (hi, _) = h.sym_inverse(1e-12);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * g[j]).sum()).collect();
        for i in 0..k {
            b[i] += step[i];
        }
        if step.iter().all(|s| s.abs() < 1e-10) {
            break;
        }
    }
    (b, ll)
}

pub struct Diflogistic;

impl CommandHandler for Diflogistic {
    fn name(&self) -> &'static str {
        "diflogistic"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (items, cols, g, total, focal) = dif_data(ds, st, cmd)?;
        let gf: Vec<f64> = g.iter().map(|v| if *v == focal { 1.0 } else { 0.0 }).collect();
        let mut out = vec![String::new(), "Logistic regression DIF analysis".to_string(), String::new(), "-".repeat(66), format!("{:>12} |{:>25}{:>26}", "", "Nonuniform", "Uniform"), format!("{:>12} |{:>13}{:>12}{:>14}{:>12}", "Item", "Chi2", "Prob.", "Chi2", "Prob."), format!("{}+{}", "-".repeat(13), "-".repeat(52))];
        let mut res = Vec::new();
        for (it, c) in items.iter().zip(&cols) {
            let n = c.len();
            let x1: Vec<Vec<f64>> = (0..n).map(|r| vec![1.0, total[r]]).collect();
            let x2: Vec<Vec<f64>> = (0..n).map(|r| vec![1.0, total[r], gf[r]]).collect();
            let x3: Vec<Vec<f64>> = (0..n).map(|r| vec![1.0, total[r], gf[r], gf[r] * total[r]]).collect();
            let (_, l1) = logit_fit(&x1, c);
            let (_, l2) = logit_fit(&x2, c);
            let (_, l3) = logit_fit(&x3, c);
            let nu = (2.0 * (l3 - l2)).max(0.0);
            let un = (2.0 * (l2 - l1)).max(0.0);
            let (pn, pu) = (dist::chi2tail(1.0, nu), dist::chi2tail(1.0, un));
            out.push(format!("{:>12} |{:>13}{:>12.4}{:>14}{:>12.4}", tab::ab(it, 12), fmt_g(nu, 8), pn, fmt_g(un, 8), pu));
            res.push(vec![nu, pn, un, pu]);
        }
        out.push("-".repeat(66));
        let mut m = Matrix::from_rows(&res);
        m.rownames = items;
        m.colnames = ["chi2_nu", "p_nu", "chi2_u", "p_u"].iter().map(|s| s.to_string()).collect();
        st.r = Results::new();
        st.r.set_mat("dif", m);
        Ok(out.join("\n"))
    }
}
