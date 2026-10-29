//! Heterogeneity-robust difference-in-differences: `drdid` (doubly robust
//! 2x2 DiD, Sant'Anna and Zhao 2020) and `csdid` (group-time average
//! treatment effects with staggered adoption, Callaway and Sant'Anna 2021),
//! with the `estat` aggregations after `csdid` and `csdid_plot`.
//!
//! Inference uses influence functions: each 2x2 estimate comes with a
//! recentred influence function per unit (panel) or observation (repeated
//! cross sections). Aggregations are linear in the group-time effects and
//! in the cohort shares used as weights, so their influence functions (and
//! the delta-method variances used by `estat`) follow from the same
//! gradients. Wild-bootstrap standard errors use Mammen or Rademacher
//! multipliers drawn from the session random-number generator.

use std::collections::BTreeMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::did_core::{self, Method, Panel2, Rc2, Wboot};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn col(ds: &DataSet, name: &str) -> Result<Vec<f64>, String> {
    let v = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if v.var_type.is_string() {
        return Err(rc_err(109, format!("{name}: string variables not allowed")));
    }
    Ok(v.data.iter().map(|x| x.num()).collect())
}

fn opt_var(ds: &DataSet, cmd: &Command, name: &str, min: usize) -> Result<Option<String>, String> {
    match cmd.opt(name, min) {
        Some(v) if !v.trim().is_empty() => Ok(Some(crate::varlist::resolve_name(ds, v.trim())?)),
        _ => Ok(None),
    }
}

/// Group codes of a column over the given rows (missing values excluded
/// beforehand).
fn codes(v: &[f64]) -> Vec<usize> {
    let mut levels: Vec<f64> = v.to_vec();
    levels.sort_by(|a, b| a.total_cmp(b));
    levels.dedup();
    v.iter().map(|x| levels.binary_search_by(|l| l.total_cmp(x)).unwrap()).collect()
}

fn commas(v: f64) -> String {
    util::commas(v.max(0.0) as usize)
}

fn coef_lines(names: &[String], b: &[f64], v: &Matrix, level: f64) -> Vec<String> {
    let k = names.len();
    est_table(names, b, v, level, None, k)
}

fn est_table(names: &[String], b: &[f64], v: &Matrix, level: f64, df: Option<f64>, k: usize) -> Vec<String> {
    crate::est::coef_table(&crate::est::TableSpec {
        depvar: "",
        names,
        b,
        v,
        omitted: &vec![false; k],
        is_base: &vec![false; k],
        df,
        level,
        eform: None,
        coef_title: "Coefficient",
    })
}

/// Table of wild-bootstrap results: estimate, se, t and the (uniform or
/// pointwise) band.
fn wb_table(names: &[String], b: &[f64], cband: &[(f64, f64)], level: f64) -> Vec<String> {
    let mut out = Vec::new();
    out.push("-".repeat(78));
    out.push(format!("{:>12} | {:>11}  Std. err.      t    [{}% conf. interval]", "", "Coefficient", crate::format::fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    let mut last_eq = String::new();
    let g = |x: f64| crate::format::fmt_g(x, 8);
    for (j, name) in names.iter().enumerate() {
        let (eq, nm) = match name.split_once(':') {
            Some((e, n)) => (e.to_string(), n.to_string()),
            None => (String::new(), name.clone()),
        };
        if !eq.is_empty() && eq != last_eq {
            out.push(format!("{:<12} |", crate::summarize::abbrev(&eq, 12)));
            last_eq = eq;
        }
        let (se, c) = cband[j];
        out.push(format!("{:>12} | {:>10}  {:>9}  {:>7.2}    {:>9}   {:>9}", crate::summarize::abbrev(&nm, 12), g(b[j]), g(se), b[j] / se, g(b[j] - c * se), g(b[j] + c * se)));
    }
    out.push("-".repeat(78));
    out
}

fn named(m: Matrix, rows: &[String], cols: &[String]) -> Matrix {
    m.with_names(rows.to_vec(), cols.to_vec())
}

fn parse_wboot(cmd: &Command, st: &Store) -> Result<Option<Wboot>, String> {
    let has = cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("wboot"));
    if !has {
        return Ok(None);
    }
    let inner = cmd.opt("wboot", 5).unwrap_or("").to_string();
    let mut reps = 999usize;
    let mut rademacher = false;
    for (n, a) in ferrostat_parser::parser::split_options(&inner) {
        match n.to_lowercase().as_str() {
            "reps" => reps = a.as_deref().unwrap_or("999").trim().parse().map_err(|_| rc_err(198, "invalid reps()"))?,
            "wbtype" | "wtype" => {
                let t = a.unwrap_or_default().trim().to_lowercase();
                rademacher = match t.as_str() {
                    "mammen" | "1" => false,
                    "rademacher" | "2" => true,
                    _ => return Err(rc_err(198, "invalid wbtype()")),
                };
            }
            "rseed" => {
                if let Some(s) = a {
                    st.rng.borrow_mut().reseed(s.trim().parse::<f64>().map_err(|_| rc_err(198, "invalid rseed()"))? as u64);
                }
            }
            _ => {}
        }
    }
    if reps < 10 {
        return Err(rc_err(198, "wboot: reps() must be at least 10"));
    }
    Ok(Some(Wboot { reps, rademacher, pointwise: cmd.has_opt("pointwise", 5) }))
}

fn seed_option(cmd: &Command, st: &Store) -> Result<(), String> {
    if let Some(s) = cmd.opt("rseed", 5) {
        let v: f64 = s.trim().parse().map_err(|_| rc_err(198, "invalid rseed()"))?;
        st.rng.borrow_mut().reseed(v as u64);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// drdid
// ---------------------------------------------------------------------------

pub struct Drdid;

struct DrData {
    rows: Vec<usize>,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    w: Vec<f64>,
}

/// Build the 2x2 inputs. Returns (estimate inputs, rows whose stub value is
/// stored (panel: base period rows), unit cluster codes, N used).
enum Two {
    Panel(Panel2, Vec<usize>, Vec<usize>),
    Rc(Rc2, Vec<usize>),
}

/// Rows `sel` of the data, `pre` flags the earlier period, `d` treatment
/// indicator. With `ivar`, rows are paired by unit.
fn build_two(ds: &DataSet, dd: &DrData, sel: &[usize], post: &[bool], d: &[f64], ivar: Option<&[f64]>) -> Result<Two, String> {
    let k = dd.x.len();
    let xrow = |i: usize| -> Vec<f64> { dd.x.iter().map(|c| c[i]).collect() };
    let _ = ds;
    match ivar {
        Some(id) => {
            let mut map: BTreeMap<i64, (Option<usize>, Option<usize>)> = BTreeMap::new();
            for (p, &i) in sel.iter().enumerate() {
                let key = (id[i] * 1e6).round() as i64;
                let e = map.entry(key).or_insert((None, None));
                let slot = if post[p] { &mut e.1 } else { &mut e.0 };
                if slot.is_some() {
                    return Err(rc_err(451, "repeated time values within panel"));
                }
                *slot = Some(p);
            }
            let mut p2 = Panel2 { dy: Vec::new(), d: Vec::new(), w: Vec::new(), x: vec![Vec::new(); k + 1] };
            let mut base_rows = Vec::new();
            let mut both_rows = Vec::new();
            for (_, (a, b)) in map {
                let (Some(a), Some(b)) = (a, b) else { continue };
                let (ia, ib) = (sel[a], sel[b]);
                p2.dy.push(dd.y[ib] - dd.y[ia]);
                p2.d.push(d[a]);
                p2.w.push(dd.w[ia]);
                for (j, v) in xrow(ia).into_iter().enumerate() {
                    p2.x[j].push(v);
                }
                p2.x[k].push(1.0);
                base_rows.push(ia);
                both_rows.push(ia);
                both_rows.push(ib);
            }
            Ok(Two::Panel(p2, base_rows, both_rows))
        }
        None => {
            let mut r = Rc2 { y: Vec::new(), post: Vec::new(), d: Vec::new(), w: Vec::new(), x: vec![Vec::new(); k + 1] };
            for (p, &i) in sel.iter().enumerate() {
                r.y.push(dd.y[i]);
                r.post.push(if post[p] { 1.0 } else { 0.0 });
                r.d.push(d[p]);
                r.w.push(dd.w[i]);
                for (j, v) in xrow(i).into_iter().enumerate() {
                    r.x[j].push(v);
                }
                r.x[k].push(1.0);
            }
            Ok(Two::Rc(r, sel.to_vec()))
        }
    }
}

fn header_lines(title: &str, n: f64, method: Option<Method>) -> Vec<String> {
    let mut out = vec![String::new()];
    if title.len() <= 50 {
        out.extend(crate::est::header(title, &[("Number of obs".to_string(), commas(n))]));
    } else {
        out.push(title.to_string());
        out.push(String::new());
        out.extend(crate::est::header("", &[("Number of obs".to_string(), commas(n))]));
    }
    if let Some(m) = method {
        let (o, t) = m.models();
        out.push(format!("Outcome model  : {o}"));
        out.push(format!("Treatment model: {t}"));
    }
    out
}

/// Replace the placeholder row of the ATET table by the factor-style
/// "treatment / (1 vs 0)" rows.
fn atet_rows(table: Vec<String>, treat: &str, placeholders: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    for l in table {
        if let Some((ph, lab)) = placeholders.iter().find(|(p, _)| l.starts_with(p)) {
            let rest = &l[ph.len()..];
            if lab.is_empty() {
                out.push(format!("{:>12} |", crate::summarize::abbrev(treat, 12)));
                out.push(format!("{:<12}{}", "   (1 vs 0)", rest));
            } else {
                out.push(format!("{:>12}{}", lab, rest));
            }
        } else {
            out.push(l);
        }
    }
    out
}

impl CommandHandler for Drdid {
    fn name(&self) -> &'static str {
        "drdid"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "drdid");
        }
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let (yname, xnames) = vars.split_first().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        let ivar = opt_var(ds, cmd, "ivar", 1)?;
        let tvar = opt_var(ds, cmd, "time", 1)?.ok_or_else(|| rc_err(198, "option time() required"))?;
        let trvar = opt_var(ds, cmd, "treatment", 2)?.ok_or_else(|| rc_err(198, "option treatment() required"))?;
        let cluster = opt_var(ds, cmd, "cluster", 2)?;
        let level = util::level(cmd, st)?;
        let flags = ["drimp", "dripw", "reg", "stdipw", "ipw", "all"];
        let chosen: Vec<&str> = flags.iter().copied().filter(|f| cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case(f))).collect();
        if cmd.has_opt("ipwra", 5) {
            return Err(rc_err(198, "option ipwra is not available; use teffects ipwra"));
        }
        if cmd.has_opt("gmm", 3) {
            return Err(rc_err(198, "option gmm is not available; influence-function standard errors are used"));
        }
        let mut out: Vec<String> = Vec::new();
        if chosen.len() > 1 {
            out.push("Only one option allowed, more than 1 selected.".into());
            return Err(rc_err(1, out.join("\n")));
        }
        let all = chosen.first() == Some(&"all");
        let method = match chosen.first() {
            None => {
                out.push("No estimator selected. Using default drimp".into());
                Method::Drimp
            }
            Some(&"all") => Method::Dripw,
            Some(m) => Method::parse(m).unwrap(),
        };
        let rc1 = cmd.has_opt("rc1", 3);
        seed_option(cmd, st)?;
        let wb = parse_wboot(cmd, st)?;
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need: Vec<String> = vars.clone();
        need.extend([tvar.clone(), trvar.clone()]);
        if let Some(i) = &ivar {
            need.push(i.clone());
        }
        if let Some(c) = &cluster {
            need.push(c.clone());
        }
        util::markout(ds, &mut mask, &need);
        let wts = util::weights(ds, st, cmd, &mut mask, "iap")?;
        let tcol = col(ds, &tvar)?;
        let trcol = col(ds, &trvar)?;
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
        let mut tl: Vec<f64> = rows.iter().map(|&i| tcol[i]).collect();
        tl.sort_by(|a, b| a.total_cmp(b));
        tl.dedup();
        let mut gl: Vec<f64> = rows.iter().map(|&i| trcol[i]).collect();
        gl.sort_by(|a, b| a.total_cmp(b));
        gl.dedup();
        if tl.len() != 2 || gl.len() != 2 {
            return Err(rc_err(555, "You do not have a 2x2 design."));
        }
        let y = col(ds, yname)?;
        let x: Vec<Vec<f64>> = xnames.iter().map(|n| col(ds, n)).collect::<Result<_, _>>()?;
        let w = wts.as_ref().map_or(vec![1.0; ds.n_obs], |w| w.w.clone());
        let dd = DrData { rows: rows.clone(), y, x, w };
        let post: Vec<bool> = rows.iter().map(|&i| tcol[i] == tl[1]).collect();
        let d: Vec<f64> = rows.iter().map(|&i| if trcol[i] == gl[1] { 1.0 } else { 0.0 }).collect();
        let idcol = match &ivar {
            Some(i) => Some(col(ds, i)?),
            None => None,
        };
        if let Some(id) = &idcol {
            // units observed once are excluded
            let mut cnt: BTreeMap<i64, usize> = BTreeMap::new();
            for &i in &rows {
                *cnt.entry((id[i] * 1e6).round() as i64).or_default() += 1;
            }
            if cnt.values().any(|c| *c < 2) {
                out.push("Some panel units are observed only once.".into());
                out.push("Those observations will be excluded from the sample. If you want to keep them, do not use the panel id ivar(), which will request repeated crossection estimators".into());
            }
        }
        let two = build_two(ds, &dd, &dd.rows, &post, &d, idcol.as_deref())?;
        let clcol = match &cluster {
            Some(c) => Some(col(ds, c)?),
            None => None,
        };
        let methods: Vec<(Method, bool, &str)> = if all {
            let mut v = vec![(Method::Dripw, false, "dripw")];
            if ivar.is_none() {
                v.push((Method::Dripw, true, "dripw_rc1"));
            }
            v.push((Method::Drimp, false, "drimp"));
            if ivar.is_none() {
                v.push((Method::Drimp, true, "drimp_rc1"));
            }
            v.extend([(Method::Reg, false, "reg"), (Method::Ipw, false, "ipw"), (Method::Stdipw, false, "stdipw")]);
            v
        } else {
            vec![(method, rc1, "")]
        };
        if rc1 && ivar.is_some() {
            return Err(rc_err(198, "option rc1 is only available with repeated cross sections"));
        }
        let mut results: Vec<(String, did_core::Fit)> = Vec::new();
        for (m, r1, lab) in &methods {
            let fit = match &two {
                Two::Panel(p, ..) => did_core::panel(*m, p),
                Two::Rc(r, _) => did_core::rc(*m, *r1, r),
            }
            .map_err(|e| rc_err(430, e))?;
            results.push((lab.to_string(), fit));
        }
        let (stub_rows, used_rows, nunits) = match &two {
            Two::Panel(p, base, both) => (base.clone(), both.clone(), p.dy.len()),
            Two::Rc(r, sel) => (sel.clone(), sel.clone(), r.y.len()),
        };
        if nunits == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let ncodes: Option<Vec<usize>> = clcol.as_ref().map(|c| codes(&stub_rows.iter().map(|&i| c[i]).collect::<Vec<_>>()));
        let n_clust = ncodes.as_ref().map(|c| c.iter().copied().max().map_or(0, |m| m + 1));
        let nobs = used_rows.len() as f64;
        let mut esample = vec![false; ds.n_obs];
        for &i in &used_rows {
            esample[i] = true;
        }
        // estimates
        let k = results.len();
        let b: Vec<f64> = results.iter().map(|(_, f)| f.att).collect();
        let mut v = Matrix::zeros(k, k);
        for (j, (_, f)) in results.iter().enumerate() {
            v[(j, j)] = did_core::if_vce(&[f.rif.clone()], &[f.att], ncodes.as_deref())[(0, 0)];
        }
        let mut cband: Option<Vec<(f64, f64)>> = None;
        if let (Some(wbo), false) = (&wb, all) {
            let cb = did_core::wild_bootstrap(&[results[0].1.rif.clone()], &b, ncodes.as_deref(), wbo, level, st);
            v[(0, 0)] = cb[0].0 * cb[0].0;
            cband = Some(cb);
        }
        let names: Vec<String> = if all { results.iter().map(|(l, _)| format!("ATET:{l}")).collect() } else { vec![format!("ATET:r1vs0.{trvar}")] };
        // output
        let title = if all { "Doubly robust difference-in-differences estimator summary" } else { "Doubly robust difference-in-differences" };
        out.extend(header_lines(title, nobs, if all { None } else { Some(method) }));
        if let (Some(c), Some(g)) = (&cluster, n_clust) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g))));
        }
        let table = if let Some(cb) = &cband {
            let ph = vec!["ATET:__did_atet__".to_string()];
            atet_rows(wb_table(&ph, &b, cb, level), &trvar, &[("__did_atet__", "")])
        } else if all {
            coef_lines(&names, &b, &v, level)
        } else {
            let ph = vec!["ATET:__did_atet__".to_string()];
            let t = coef_lines(&ph, &b, &v, level);
            crate::est::stash_table(crate::est::table_matrix(&crate::est::TableSpec { depvar: "", names: &names, b: &b, v: &v, omitted: &[false], is_base: &[false], df: None, level, eform: None, coef_title: "Coefficient" }), level);
            atet_rows(t, &trvar, &[("__did_atet__", "")])
        };
        out.extend(table);
        if all {
            out.push("Note: This table is provided for comparison across estimations only. You cannot use it to compare estimates across different estimators".into());
            out.push("dripw :Doubly Robust IPW".into());
            out.push("drimp :Doubly Robust Improved estimator".into());
            out.push("reg   :Outcome regression or Regression augmented estimator".into());
            out.push("ipw   :Abadie(2005) IPW estimator".into());
            out.push("stdipw:Standardized IPW estimator".into());
        }
        let text = out.join("\n");
        // stub(): recentred influence function
        if let Some(stub) = cmd.opt("stub", 4) {
            let name = format!("{}att", stub.trim());
            if ds.get_var(&name).is_some() {
                if !cmd.has_opt("replace", 3) {
                    return Err(rc_err(110, format!("variable {name} already defined")));
                }
                ds.drop_variable(&name)?;
            }
            let mut vals = vec![ferrostat_core::missing::SYSMISS; ds.n_obs];
            for (p, &i) in stub_rows.iter().enumerate() {
                vals[i] = results[0].1.rif[p];
            }
            ds.add_variable(util::new_double(&name, vals))?;
            st.data_changed = true;
        }
        // e()
        let e = st.begin_e();
        e.set("N", nobs);
        e.set_str("cmd", "drdid");
        e.set_str("cmdline", cmd.raw.trim().to_string());
        e.set_str("depvar", yname.clone());
        e.set_str("properties", "b V");
        e.set_mat("b", named(Matrix::row_vector(&b), &["y1".into()], &names));
        e.set_mat("V", named(v.clone(), &names, &names));
        e.sample = Some(esample);
        e.set_str("method", if all { "all" } else { method.name() });
        e.set_str("policy", trvar.clone());
        e.set_str("datatype", if ivar.is_some() { "panel" } else { "rcs" });
        e.set_str("semethod", if wb.is_some() && !all { "wildboot" } else { "rif" });
        e.set_str("title", title);
        e.set_str("vce", if cluster.is_some() { "cluster" } else { "robust" });
        if !all {
            e.set("att1", b[0]);
            e.set("attvar1", v[(0, 0)]);
            for (nm, coefs, vm) in &results[0].1.mats {
                let mut cn: Vec<String> = xnames.to_vec();
                cn.push("_cons".into());
                e.set_mat(nm, named(Matrix::row_vector(coefs), &["y1".into()], &cn));
                if let Some(vm) = vm {
                    let vn = nm.replace('b', "V");
                    e.set_mat(&vn, named(vm.clone(), &cn, &cn));
                }
            }
        }
        if let Some(i) = &ivar {
            e.set_str("ivar", i.clone());
        }
        if let (Some(c), Some(g)) = (&cluster, n_clust) {
            e.set_str("clustvar", c.clone());
            e.set("N_clust", g as f64);
        }
        if let (Some(cb), Some(wbo)) = (&cband, &wb) {
            let (se, c) = cb[0];
            let mut m = Matrix::zeros(1, 5);
            m.data = vec![b[0] - c * se, b[0] + c * se, b[0], se, c];
            e.set_mat("ciband", named(m, &["r1".into()], &["ll".into(), "ul".into(), "b".into(), "se".into(), "crit".into()]));
            e.set_str("vcetype", "WBoot");
            e.set("reps", wbo.reps as f64);
            e.set_str("wbtype", if wbo.rademacher { "rademacher" } else { "mammen" });
        }
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// csdid
// ---------------------------------------------------------------------------

/// One estimated group-time effect.
#[derive(Clone, Debug)]
struct Cell {
    g: f64,
    /// period compared with the base period
    t: f64,
    base: f64,
}

impl Cell {
    fn e(&self) -> f64 {
        self.t - self.g
    }
    fn name(&self) -> String {
        format!("g{}:t_{}_{}", fmtn(self.g), fmtn(self.base.min(self.t)), fmtn(self.base.max(self.t)))
    }
}

fn fmtn(v: f64) -> String {
    crate::format::fmt_g(v, 12).trim().to_string()
}

/// An aggregated parameter: name, estimate and gradient with respect to
/// (theta_1..theta_K, p_1..p_K).
struct Agg {
    name: String,
    est: f64,
    grad: Vec<f64>,
}

/// Weighted average of cells `idx` with cohort-share weights p (estimated).
fn aggte(idx: &[usize], th: &[f64], p: &[f64], name: String) -> Option<Agg> {
    let k = th.len();
    let pt: f64 = idx.iter().map(|&j| p[j]).sum();
    if idx.is_empty() || pt <= 0.0 {
        return None;
    }
    let est = idx.iter().map(|&j| p[j] * th[j]).sum::<f64>() / pt;
    let mut grad = vec![0.0; 2 * k];
    for &j in idx {
        grad[j] = p[j] / pt;
        grad[k + j] = (th[j] - est) / pt;
    }
    Some(Agg { name, est, grad })
}

/// Fixed-weight average of aggregates.
fn fixed_avg(parts: &[&Agg], c: &[f64], name: String) -> Option<Agg> {
    let ct: f64 = c.iter().sum();
    if parts.is_empty() || ct == 0.0 {
        return None;
    }
    let k2 = parts[0].grad.len();
    let est = parts.iter().zip(c).map(|(a, w)| a.est * w).sum::<f64>() / ct;
    let mut grad = vec![0.0; k2];
    for (a, w) in parts.iter().zip(c) {
        for (g, ag) in grad.iter_mut().zip(&a.grad) {
            *g += ag * w / ct;
        }
    }
    Some(Agg { name, est, grad })
}

/// Aggregations of group-time effects. `bt` holds theta (K) then p (K).
fn aggregate(kind: &str, cells: &[Cell], bt: &[f64], from: f64, window: Option<(f64, f64)>) -> Result<Vec<Agg>, String> {
    let k = cells.len();
    let (th, p) = bt.split_at(k);
    let post: Vec<usize> = (0..k).filter(|&j| cells[j].e() >= from).collect();
    let mut out = Vec::new();
    match kind {
        "simple" => {
            if let Some(a) = aggte(&post, th, p, "ATT".into()) {
                out.push(a);
            }
        }
        "group" => {
            let mut gs: Vec<f64> = cells.iter().map(|c| c.g).collect();
            gs.sort_by(|a, b| a.total_cmp(b));
            gs.dedup();
            let mut parts = Vec::new();
            let mut cw = Vec::new();
            for g in gs {
                let idx: Vec<usize> = post.iter().copied().filter(|&j| cells[j].g == g).collect();
                if let Some(a) = aggte(&idx, th, p, format!("G{}", fmtn(g))) {
                    cw.push(idx.iter().map(|&j| p[j]).sum::<f64>() / idx.len() as f64);
                    parts.push(a);
                }
            }
            let refs: Vec<&Agg> = parts.iter().collect();
            if let Some(a) = fixed_avg(&refs, &cw, "GAverage".into()) {
                out.push(a);
            }
            out.extend(parts);
        }
        "calendar" => {
            let mut ts: Vec<f64> = post.iter().map(|&j| cells[j].t).collect();
            ts.sort_by(|a, b| a.total_cmp(b));
            ts.dedup();
            let mut parts = Vec::new();
            for t in ts {
                let idx: Vec<usize> = post.iter().copied().filter(|&j| cells[j].t == t).collect();
                if let Some(a) = aggte(&idx, th, p, format!("T{}", fmtn(t))) {
                    parts.push(a);
                }
            }
            let refs: Vec<&Agg> = parts.iter().collect();
            let ones = vec![1.0; refs.len()];
            if let Some(a) = fixed_avg(&refs, &ones, "CAverage".into()) {
                out.push(a);
            }
            out.extend(parts);
        }
        "event" => {
            let mut es: Vec<f64> = cells.iter().map(|c| c.e()).collect();
            es.sort_by(|a, b| a.total_cmp(b));
            es.dedup();
            if let Some((lo, hi)) = window {
                es.retain(|e| *e >= lo && *e <= hi);
            }
            let mut parts = Vec::new();
            for e in es {
                let idx: Vec<usize> = (0..k).filter(|&j| cells[j].e() == e).collect();
                let name = if e < 0.0 { format!("Tm{}", fmtn(-e)) } else { format!("Tp{}", fmtn(e)) };
                if let Some(a) = aggte(&idx, th, p, name) {
                    parts.push((e, a));
                }
            }
            let pre: Vec<&Agg> = parts.iter().filter(|(e, _)| *e < 0.0).map(|(_, a)| a).collect();
            let pos: Vec<&Agg> = parts.iter().filter(|(e, _)| *e >= 0.0).map(|(_, a)| a).collect();
            if let Some(a) = fixed_avg(&pre, &vec![1.0; pre.len()], "Pre_avg".into()) {
                out.push(a);
            }
            if let Some(a) = fixed_avg(&pos, &vec![1.0; pos.len()], "Post_avg".into()) {
                out.push(a);
            }
            out.extend(parts.into_iter().map(|(_, a)| a));
        }
        "attgt" => {
            for j in 0..k {
                let mut grad = vec![0.0; 2 * k];
                grad[j] = 1.0;
                out.push(Agg { name: cells[j].name(), est: th[j], grad });
            }
        }
        o => return Err(rc_err(198, format!("aggregation {o} not allowed"))),
    }
    if out.is_empty() {
        return Err(rc_err(2000, "no group-time effects available for this aggregation"));
    }
    Ok(out)
}

/// V of aggregates by the delta method.
fn delta_v(aggs: &[Agg], v: &Matrix) -> Matrix {
    let m = aggs.len();
    let k2 = v.rows;
    let mut out = Matrix::zeros(m, m);
    for a in 0..m {
        let gv: Vec<f64> = (0..k2).map(|j| (0..k2).map(|i| aggs[a].grad[i] * v[(i, j)]).sum()).collect();
        for c in 0..=a {
            let s: f64 = (0..k2).map(|j| gv[j] * aggs[c].grad[j]).sum();
            out[(a, c)] = s;
            out[(c, a)] = s;
        }
    }
    out
}

pub struct Csdid;

/// Pieces of csdid results kept for estat.
fn cells_from_e(e: &Results) -> Result<(Vec<Cell>, Vec<f64>, Matrix), String> {
    let b = e.matrices.get("b_attgt").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V_attgt").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let cm = e.matrices.get("gtt_cells").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let cells: Vec<Cell> = (0..cm.rows).map(|r| Cell { g: cm[(r, 0)], base: cm[(r, 1)], t: cm[(r, 2)] }).collect();
    Ok((cells, b.data.clone(), v.clone()))
}

impl CommandHandler for Csdid {
    fn name(&self) -> &'static str {
        "csdid"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "csdid");
        }
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let (yname, xnames) = vars.split_first().ok_or_else(|| rc_err(102, "too few variables specified"))?;
        let ivar = opt_var(ds, cmd, "ivar", 1)?;
        let tvar = opt_var(ds, cmd, "time", 1)?.ok_or_else(|| rc_err(198, "option time() required"))?;
        let gvar = opt_var(ds, cmd, "gvar", 4)?.ok_or_else(|| rc_err(198, "option gvar() required"))?;
        let cluster = opt_var(ds, cmd, "cluster", 2)?;
        let level = util::level(cmd, st)?;
        let method = match cmd.opt("method", 3) {
            Some(m) => Method::parse(m).ok_or_else(|| rc_err(198, format!("Method {} not allowed", m.trim())))?,
            None if xnames.is_empty() => Method::Reg,
            None => Method::Dripw,
        };
        let rc1 = cmd.has_opt("rc1", 3);
        let agg = cmd.opt("agg", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "attgt".into());
        if !["attgt", "simple", "group", "calendar", "event"].contains(&agg.as_str()) {
            return Err(rc_err(10, "Aggregation not Allowed"));
        }
        let long = cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("long"));
        let long2 = cmd.has_opt("long2", 5);
        let asinr = cmd.has_opt("asinr", 5);
        let mut notyet = cmd.has_opt("notyet", 6);
        seed_option(cmd, st)?;
        let wb = parse_wboot(cmd, st)?;
        if cmd.has_opt("saverif", 5) {
            return Err(rc_err(198, "option saverif() is not available; aggregations are available through estat"));
        }
        let mut out: Vec<String> = Vec::new();
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = vars.clone();
        need.extend([tvar.clone(), gvar.clone()]);
        if let Some(i) = &ivar {
            need.push(i.clone());
        }
        if let Some(c) = &cluster {
            need.push(c.clone());
        }
        util::markout(ds, &mut mask, &need);
        let wts = util::weights(ds, st, cmd, &mut mask, "iap")?;
        let tcol = col(ds, &tvar)?;
        let gcol = col(ds, &gvar)?;
        let levels = |mask: &[bool], c: &[f64]| -> Vec<f64> {
            let mut v: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| c[i]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup();
            v
        };
        let gl_all = levels(&mask, &gcol);
        if gl_all.iter().any(|g| *g < 0.0) {
            return Err(rc_err(198, "gvar() must be zero (never treated) or the first treatment period"));
        }
        if !notyet && !gl_all.contains(&0.0) {
            notyet = true;
            out.push("No never treated observations found. Using Not yet treated data".into());
        }
        let has_never = gl_all.contains(&0.0);
        let tmax_never = (0..ds.n_obs).filter(|&i| mask[i] && gcol[i] == 0.0).map(|i| tcol[i]).fold(f64::NEG_INFINITY, f64::max);
        if !notyet {
            for i in 0..ds.n_obs {
                if mask[i] && tcol[i] > tmax_never {
                    mask[i] = false;
                }
            }
        } else {
            let gmax = gl_all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for i in 0..ds.n_obs {
                if mask[i] && tcol[i] > gmax && (!has_never || tcol[i] > tmax_never) {
                    mask[i] = false;
                }
            }
        }
        let tl = levels(&mask, &tcol);
        if tl.len() < 2 {
            return Err(rc_err(2000, "at least two time periods are required"));
        }
        let delta = tl.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
        let time0 = tl[0];
        let mut gl: Vec<f64> = levels(&mask, &gcol).into_iter().filter(|g| *g > 0.0).collect();
        if gl.is_empty() {
            return Err(rc_err(2000, "no treated groups found"));
        }
        let maxg = gl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if maxg <= time0 {
            return Err(rc_err(190, format!("Gvar max value is {} and there are no periods available before that treatment\nVerify that Gvar is correctly defined", fmtn(maxg))));
        }
        if gl.iter().any(|g| *g <= time0) {
            out.push("Units always treated found. These will be ignored".into());
            for i in 0..ds.n_obs {
                if mask[i] && gcol[i] > 0.0 && gcol[i] <= time0 {
                    mask[i] = false;
                }
            }
            gl.retain(|g| *g > time0);
        }
        // time grid: regular steps from the first period
        let nsteps = ((tl[tl.len() - 1] - time0) / delta).round() as usize;
        let tgrid: Vec<f64> = (0..=nsteps).map(|s| time0 + delta * s as f64).collect();
        let tlev: Vec<f64> = tgrid[1..].to_vec();
        // data
        let y = col(ds, yname)?;
        let x: Vec<Vec<f64>> = xnames.iter().map(|n| col(ds, n)).collect::<Result<_, _>>()?;
        let w = wts.as_ref().map_or(vec![1.0; ds.n_obs], |w| w.w.clone());
        let idcol = match &ivar {
            Some(i) => Some(col(ds, i)?),
            None => None,
        };
        let clcol = match &cluster {
            Some(c) => Some(col(ds, c)?),
            None => None,
        };
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
        // units
        let (unit_of, nunits): (Vec<usize>, usize) = match &idcol {
            Some(id) => {
                let mut map: BTreeMap<i64, usize> = BTreeMap::new();
                let mut u = vec![usize::MAX; ds.n_obs];
                for &i in &rows {
                    let key = (id[i] * 1e6).round() as i64;
                    let n = map.len();
                    let c = *map.entry(key).or_insert(n);
                    u[i] = c;
                }
                // cohort must be constant within units
                let mut gu: Vec<Option<f64>> = vec![None; map.len()];
                for &i in &rows {
                    match gu[u[i]] {
                        Some(g) if g != gcol[i] => return Err(rc_err(198, "gvar() must be constant within panel units")),
                        _ => gu[u[i]] = Some(gcol[i]),
                    }
                }
                // balance check
                let mut cnt = vec![0usize; map.len()];
                for &i in &rows {
                    cnt[u[i]] += 1;
                }
                if cnt.iter().min() != cnt.iter().max() {
                    out.push("Panel is not balanced".into());
                    out.push("Will use observations with Pair balanced (observed at t0 and t1)".into());
                }
                let n = map.len();
                (u, n)
            }
            None => {
                let mut u = vec![usize::MAX; ds.n_obs];
                for (p, &i) in rows.iter().enumerate() {
                    u[i] = p;
                }
                (u, rows.len())
            }
        };
        let mut uw = vec![0.0; nunits];
        let mut ucnt = vec![0.0; nunits];
        let mut ug = vec![0.0; nunits];
        let mut ucl = vec![0.0; nunits];
        for &i in &rows {
            uw[unit_of[i]] += w[i];
            ucnt[unit_of[i]] += 1.0;
            ug[unit_of[i]] = gcol[i];
            if let Some(c) = &clcol {
                ucl[unit_of[i]] = c[i];
            }
        }
        for u in 0..nunits {
            uw[u] /= ucnt[u];
        }
        let dd = DrData { rows: rows.clone(), y, x, w };
        // estimate every cell
        let mut cells: Vec<Cell> = Vec::new();
        let mut rifs: Vec<Vec<f64>> = Vec::new(); // per cell, NaN for units outside
        let mut used = vec![false; ds.n_obs];
        let mut dots = String::new();
        let in_ctrl = |gv: f64, g: f64, j: f64| -> bool {
            if gv == 0.0 {
                return true;
            }
            if !notyet {
                return false;
            }
            if asinr && j < g {
                gv > j && gv != g
            } else {
                gv > g && gv > j
            }
        };
        let mut plan: Vec<(f64, f64, f64)> = Vec::new(); // (g, base, t)
        for &g in &gl {
            if long || long2 {
                let base = g - delta;
                for &j in &tgrid {
                    if (j - base).abs() > 1e-9 {
                        plan.push((g, base, j));
                    }
                }
            } else {
                let mut time1 = time0;
                for &j in &tlev {
                    plan.push((g, time1, j));
                    if j < g {
                        time1 = j;
                    }
                }
            }
        }
        for &(g, base, j) in &plan {
            let sel: Vec<usize> = rows.iter().copied().filter(|&i| (tcol[i] == base || tcol[i] == j) && (gcol[i] == g || in_ctrl(gcol[i], g, j))).collect();
            let lo = base.min(j);
            let post: Vec<bool> = sel.iter().map(|&i| tcol[i] != lo).collect();
            let d: Vec<f64> = sel.iter().map(|&i| if gcol[i] == g { 1.0 } else { 0.0 }).collect();
            let res = build_two(ds, &dd, &sel, &post, &d, idcol.as_deref()).and_then(|two| match two {
                Two::Panel(p, base_rows, both) => did_core::panel(method, &p).map(|f| (f, base_rows, both)),
                Two::Rc(r, s) => did_core::rc(method, rc1, &r).map(|f| (f, s.clone(), s)),
            });
            match res {
                Ok((fit, unit_rows, both)) => {
                    let sign = if long2 && j < base { -1.0 } else { 1.0 };
                    let mut col = vec![f64::NAN; nunits];
                    for (p, &i) in unit_rows.iter().enumerate() {
                        col[unit_of[i]] = sign * fit.rif[p];
                    }
                    for i in both {
                        used[i] = true;
                    }
                    cells.push(Cell { g, t: j, base });
                    rifs.push(col);
                    dots.push('.');
                }
                Err(_) => dots.push('x'),
            }
        }
        if cells.is_empty() {
            return Err(rc_err(2000, "no group-time average treatment effect could be estimated"));
        }
        // recentred influence functions on the full set of units, and
        // cohort-share weights
        let k = cells.len();
        let nf = nunits as f64;
        let mw = uw.iter().sum::<f64>() / nf;
        let mut rif_all: Vec<Vec<f64>> = Vec::with_capacity(2 * k);
        for c in &rifs {
            let nn = c.iter().filter(|v| !v.is_nan()).count() as f64;
            let m = c.iter().filter(|v| !v.is_nan()).sum::<f64>() / nn;
            rif_all.push(c.iter().map(|v| if v.is_nan() { m } else { m + (v - m) * nf / nn }).collect());
        }
        let mut wcols: Vec<Vec<f64>> = cells.iter().zip(&rifs).map(|(c, r)| (0..nunits).map(|u| if ug[u] == c.g && !r[u].is_nan() { uw[u] / mw } else { 0.0 }).collect()).collect();
        let s: f64 = wcols.iter().map(|c| c.iter().sum::<f64>() / nf).sum();
        for c in wcols.iter_mut() {
            for v in c.iter_mut() {
                *v /= s;
            }
        }
        rif_all.extend(wcols);
        let bt: Vec<f64> = rif_all.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
        let clcodes: Option<Vec<usize>> = clcol.as_ref().map(|_| codes(&ucl));
        let n_clust = clcodes.as_ref().map(|c| c.iter().copied().max().map_or(0, |m| m + 1));
        let vt = did_core::if_vce(&rif_all, &bt, clcodes.as_deref());
        // reported aggregation
        let aggs = aggregate(&agg, &cells, &bt, 0.0, None)?;
        let b: Vec<f64> = aggs.iter().map(|a| a.est).collect();
        let mut v = delta_v(&aggs, &vt);
        let names: Vec<String> = aggs.iter().map(|a| a.name.clone()).collect();
        let mut cband = None;
        if let Some(wbo) = &wb {
            // influence functions of the aggregates
            let ifs: Vec<Vec<f64>> = aggs.iter().map(|a| (0..nunits).map(|u| a.est + (0..2 * k).map(|j| a.grad[j] * (rif_all[j][u] - bt[j])).sum::<f64>()).collect()).collect();
            let cb = did_core::wild_bootstrap(&ifs, &b, clcodes.as_deref(), wbo, level, st);
            v = Matrix::zeros(b.len(), b.len());
            for (j, (se, _)) in cb.iter().enumerate() {
                v[(j, j)] = se * se;
            }
            cband = Some(cb);
        }
        let nobs = used.iter().filter(|u| **u).count() as f64;
        // output
        out.push(dots);
        out.extend(header_lines("Difference-in-difference with Multiple Time Periods", nobs, Some(method)));
        if let (Some(c), Some(g)) = (&cluster, n_clust) {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {c})", util::commas(g))));
        }
        match &cband {
            Some(cb) => out.extend(wb_table(&names, &b, cb, level)),
            None => out.extend(coef_lines(&names, &b, &v, level)),
        }
        let control = if notyet { "Not yet Treated" } else { "Never Treated" };
        out.push(format!("Control: {control}"));
        out.push(String::new());
        out.push("See Callaway and Sant'Anna (2021) for details".into());
        let text = out.join("\n");
        // e()
        let mut tnames: Vec<String> = cells.iter().map(|c| c.name()).collect();
        tnames.extend(cells.iter().map(|c| format!("wgt:w{}_{}", fmtn(c.g), fmtn(c.t))));
        let mut gtt = Matrix::zeros(k, 3);
        for (r, c) in cells.iter().enumerate() {
            gtt[(r, 0)] = c.g;
            gtt[(r, 1)] = c.base;
            gtt[(r, 2)] = c.t;
        }
        let e = st.begin_e();
        e.set("N", nobs);
        e.set_str("cmd", "csdid");
        e.set_str("cmdline", cmd.raw.trim().to_string());
        e.set_str("depvar", yname.clone());
        e.set_str("properties", "b V");
        e.set_str("estat_cmd", "csdid_estat");
        e.set_mat("b", named(Matrix::row_vector(&b), &["y1".into()], &names));
        e.set_mat("V", named(v.clone(), &names, &names));
        e.set_mat("b_attgt", named(Matrix::row_vector(&bt), &["y1".into()], &tnames));
        e.set_mat("V_attgt", named(vt, &tnames, &tnames));
        e.set_mat("gtt_cells", named(gtt, &cells.iter().map(|c| c.name()).collect::<Vec<_>>(), &["cohort".into(), "t0".into(), "t1".into()]));
        e.sample = Some(used);
        e.set_str("agg", agg.clone());
        e.set_str("glev", gl.iter().map(|g| fmtn(*g)).collect::<Vec<_>>().join(" "));
        e.set_str("tlev", tlev.iter().map(|g| fmtn(*g)).collect::<Vec<_>>().join(" "));
        e.set_str("time0", fmtn(time0));
        e.set_str("ggroup", gvar.clone());
        e.set_str("id", ivar.clone().unwrap_or_default());
        e.set_str("method", method.name());
        e.set_str("typebase", if long2 { "long2" } else if long { "long" } else { "default" });
        e.set_str("control_group", control);
        e.set_str("title", "Difference-in-difference with Multiple Time Periods");
        e.set("neqr", gl.len() as f64);
        if let (Some(c), Some(g)) = (&cluster, n_clust) {
            e.set_str("clustvar", c.clone());
            e.set("N_clust", g as f64);
        }
        if let (Some(cb), Some(wbo)) = (&cband, &wb) {
            let mut m = Matrix::zeros(b.len(), 5);
            for (j, (se, c)) in cb.iter().enumerate() {
                m[(j, 0)] = b[j];
                m[(j, 1)] = *se;
                m[(j, 2)] = b[j] / se;
                m[(j, 3)] = b[j] - c * se;
                m[(j, 4)] = b[j] + c * se;
            }
            e.set_mat("cband", named(m, &names, &["b".into(), "se".into(), "t".into(), "ll".into(), "ul".into()]));
            e.set_str("vcetype", "WBoot");
            e.set("reps", wbo.reps as f64);
            e.set_str("wbtype", if wbo.rademacher { "rademacher" } else { "mammen" });
            e.set_str("citype", if wbo.pointwise { "pointwise" } else { "uniform" });
        }
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// estat after csdid
// ---------------------------------------------------------------------------

fn parse_window(cmd: &Command) -> Result<Option<(f64, f64)>, String> {
    match cmd.opt("window", 3) {
        Some(w) => {
            let v: Vec<f64> = w.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() != 2 {
                return Err(rc_err(198, "window() requires two numbers"));
            }
            Ok(Some((v[0].min(v[1]), v[0].max(v[1]))))
        }
        None => Ok(None),
    }
}

/// `estat` subcommands after csdid; None when not applicable.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let _ = ds;
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd == "csdid" {
        if !["attgt", "simple", "pretrend", "group", "calendar", "event", "cevent", "all"].contains(&sub) {
            return Some(Err(rc_err(199, format!("Option {sub} not recognized"))));
        }
        return Some(csdid_estat(st, sub, cmd));
    }
    if ecmd == "didregress" || ecmd == "xtdidregress" {
        return crate::didregress::estat(ds, st, sub, cmd);
    }
    None
}

fn csdid_estat(st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let (cells, bt, vt) = cells_from_e(&st.e)?;
    let level = util::level(cmd, st)?;
    let from = util::opt_num(cmd, "from", 4)?.unwrap_or(0.0);
    let window = parse_window(cmd)?;
    let mut out = Vec::new();
    if st.e.macros.get("vcetype").map(String::as_str) == Some("WBoot") {
        out.push("Test will be based on asymptotic VCoV".to_string());
    }
    if sub == "all" {
        let mut parts = Vec::new();
        for s in ["pretrend", "simple", "group", "calendar", "event"] {
            parts.push(csdid_estat(st, s, cmd)?);
        }
        return Ok(parts.join("\n"));
    }
    if sub == "pretrend" {
        let k = cells.len();
        let pre: Vec<usize> = (0..k).filter(|&j| cells[j].t < cells[j].g && window.is_none_or(|(lo, hi)| cells[j].e() >= lo && cells[j].e() <= hi)).collect();
        if pre.is_empty() {
            return Err(rc_err(2000, "no pre-treatment effects"));
        }
        out.push(if window.is_some() { "Pretrend Test. H0 Pre-treatment within window are equal to 0" } else { "Pretrend Test. H0 All Pre-treatment are equal to 0" }.to_string());
        let m = pre.len();
        let mut vp = Matrix::zeros(m, m);
        for a in 0..m {
            for c in 0..m {
                vp[(a, c)] = vt[(pre[a], pre[c])];
            }
        }
        let (vi, dropped) = vp.sym_inverse(1e-12);
        let bp: Vec<f64> = pre.iter().map(|&j| bt[j]).collect();
        let chi2: f64 = (0..m).map(|a| (0..m).map(|c| bp[a] * vi[(a, c)] * bp[c]).sum::<f64>()).sum();
        let df = (m - dropped.len()) as f64;
        let p = crate::dist::chi2tail(df, chi2);
        out.push(format!("chi2({}) = {:>10.4}", df, chi2));
        out.push(format!("p-value  = {:>10.4}", p));
        let r = st.begin_r();
        r.set("chi2", chi2);
        r.set("pchi2", p);
        r.set("df", df);
        return Ok(out.join("\n"));
    }
    let kind = if sub == "cevent" { "event" } else { sub };
    let title = match sub {
        "attgt" => "",
        "simple" => "Average Treatment Effect on Treated",
        "group" => "ATT by group",
        "calendar" => "ATT by Calendar Period",
        "event" => "ATT by Periods Before and After treatment\nEvent Study:Dynamic effects",
        _ => "Event Study:Aggregate effects",
    };
    let mut aggs = aggregate(kind, &cells, &bt, from, window)?;
    if sub == "cevent" {
        let (lo, hi) = window.ok_or_else(|| rc_err(198, "option window() required"))?;
        let parts: Vec<&Agg> = aggs.iter().filter(|a| a.name.starts_with("T")).collect();
        let a = fixed_avg(&parts, &vec![1.0; parts.len()], "ATTC".into()).ok_or_else(|| rc_err(2000, "no events in window"))?;
        out.push(format!("ATT for events between {} {}", fmtn(lo), fmtn(hi)));
        aggs = vec![a];
    }
    if !title.is_empty() {
        out.push(title.to_string());
    }
    let b: Vec<f64> = aggs.iter().map(|a| a.est).collect();
    let v = delta_v(&aggs, &vt);
    let names: Vec<String> = aggs.iter().map(|a| a.name.clone()).collect();
    out.extend(coef_lines(&names, &b, &v, level));
    let table = crate::est::take_table();
    let r = st.begin_r();
    r.set_mat("b", named(Matrix::row_vector(&b), &["y1".into()], &names));
    r.set_mat("V", named(v.clone(), &names, &names));
    if let Some((t, _)) = table.clone() {
        r.set_mat("table", t);
    }
    r.set_str("agg", sub);
    r.set("level", level);
    let skip = match sub {
        "group" | "calendar" => 1,
        "event" => 2.min(b.len()),
        _ => 0,
    };
    if skip > 0 {
        let rest: Vec<String> = names[skip..].to_vec();
        let mut vv = Matrix::zeros(rest.len(), rest.len());
        for a in 0..rest.len() {
            for c in 0..rest.len() {
                vv[(a, c)] = v[(a + skip, c + skip)];
            }
        }
        r.set_mat("bb", named(Matrix::row_vector(&b[skip..]), &["y1".into()], &rest));
        r.set_mat("vv", named(vv, &rest, &rest));
    }
    let post = cmd.has_opt("post", 4);
    if post {
        let e = st.begin_e();
        e.set_mat("b", named(Matrix::row_vector(&b), &["y1".into()], &names));
        e.set_mat("V", named(v, &names, &names));
        e.set_str("cmd", "estat");
        e.set_str("cmdline", format!("estat {sub}"));
        e.set_str("agg", sub);
        e.set_str("properties", "b V");
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// csdid_plot and event_plot
// ---------------------------------------------------------------------------

/// Event-study style graph: estimates with confidence intervals; points
/// before treatment and after treatment in different colours.
pub(crate) fn plot_effects(st: &mut Store, cmd: &Command, pts: &[(f64, f64, f64, f64, bool)], xtitle: &str, title: &str, labels: Vec<(f64, String)>) -> Result<String, String> {
    use crate::graph::{Graph, Layer, LayerKind, RangeStyle};
    if pts.is_empty() {
        return Err(rc_err(2000, "nothing to plot"));
    }
    let mut g = Graph::default();
    let pre: Vec<&(f64, f64, f64, f64, bool)> = pts.iter().filter(|p| !p.4).collect();
    let post: Vec<&(f64, f64, f64, f64, bool)> = pts.iter().filter(|p| p.4).collect();
    let cols = [(213u8, 94u8, 0u8), (0u8, 114u8, 178u8)];
    let style = cmd.opt("style", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "rspike".into());
    for (set, colr, lab) in [(&pre, cols[0], "Pre-treatment"), (&post, cols[1], "Post-treatment")] {
        if set.is_empty() {
            continue;
        }
        let rs = match style.as_str() {
            "rarea" => RangeStyle::Area,
            "rcap" => RangeStyle::Cap,
            "rbar" => RangeStyle::Bar,
            _ => RangeStyle::Spike,
        };
        let mut l = Layer { kind: LayerKind::Range(rs), color: Some(colr), width: 0.3, ..Default::default() };
        l.ranges = set.iter().map(|p| (p.0, p.2, p.3)).collect();
        if rs == RangeStyle::Area {
            l.alpha = 0.35;
        }
        g.layers.push(l);
        let mut m = Layer { kind: LayerKind::Scatter, color: Some(colr), label: lab.into(), ..Default::default() };
        m.points = set.iter().map(|p| (p.0, p.1)).collect();
        g.layers.push(m);
    }
    g.legend = !pre.is_empty() && !post.is_empty();
    if g.legend {
        g.legend_order = vec![(2, "Pre-treatment".into()), (4, "Post-treatment".into())];
    }
    g.xtitle = xtitle.into();
    g.ytitle = "ATT".into();
    g.title = title.into();
    g.ylines = vec![0.0];
    if !labels.is_empty() {
        g.xlabels = labels;
    }
    crate::graph::finish(&mut g, cmd, st)
}

pub struct CsdidPlot;

impl CommandHandler for CsdidPlot {
    fn name(&self) -> &'static str {
        "csdid_plot"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = &st.e;
        let ecmd = e.macros.get("cmd").cloned().unwrap_or_default();
        let level = util::level(cmd, st)?;
        let crit = crate::dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let (names, b, se, kind): (Vec<String>, Vec<f64>, Vec<f64>, String) = if ecmd == "csdid" {
            let agg = e.macros.get("agg").cloned().unwrap_or_default();
            let (cells, bt, vt) = cells_from_e(e)?;
            let kind = cmd.opt("agg", 3).map(|s| s.trim().to_lowercase()).unwrap_or(if agg == "attgt" || agg == "simple" { if cmd.has_opt("group", 5) { "attgt".into() } else { "event".into() } } else { agg });
            if kind == "attgt" {
                let g: f64 = util::opt_num(cmd, "group", 5)?.ok_or_else(|| rc_err(198, "option group() required after csdid with agg(attgt)"))?;
                let aggs = aggregate("attgt", &cells, &bt, 0.0, None)?;
                let v = delta_v(&aggs, &vt);
                let mut names = Vec::new();
                let mut b = Vec::new();
                let mut se = Vec::new();
                for (j, c) in cells.iter().enumerate() {
                    if c.g == g {
                        names.push(format!("T{}", fmtn(c.t)));
                        b.push(aggs[j].est);
                        se.push(v[(j, j)].sqrt());
                    }
                }
                (names, b, se, "attgt".into())
            } else {
                let aggs = aggregate(&kind, &cells, &bt, 0.0, None)?;
                let v = delta_v(&aggs, &vt);
                let names = aggs.iter().map(|a| a.name.clone()).collect();
                let b = aggs.iter().map(|a| a.est).collect();
                let se = (0..aggs.len()).map(|j| v[(j, j)].sqrt()).collect();
                (names, b, se, kind)
            }
        } else if ecmd == "estat" {
            let bm = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let vm = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
            let kind = e.macros.get("agg").cloned().unwrap_or_default();
            (bm.colnames.clone(), bm.data.clone(), (0..bm.cols).map(|j| vm[(j, j)].sqrt()).collect(), kind)
        } else {
            return Err(rc_err(301, "csdid_plot requires results from csdid"));
        };
        let mut pts = Vec::new();
        let mut labels = Vec::new();
        let gnum = if kind == "attgt" { util::opt_num(cmd, "group", 5)?.unwrap_or(0.0) } else { 0.0 };
        for (j, n) in names.iter().enumerate() {
            let (x, post) = match kind.as_str() {
                "event" => match (n.strip_prefix("Tm"), n.strip_prefix("Tp")) {
                    (Some(v), _) => (-v.parse::<f64>().unwrap_or(0.0), false),
                    (_, Some(v)) => (v.parse::<f64>().unwrap_or(0.0), true),
                    _ => continue,
                },
                "group" => match n.strip_prefix('G') {
                    Some(v) => (v.parse::<f64>().unwrap_or(0.0), true),
                    None => continue,
                },
                "calendar" => match n.strip_prefix('T') {
                    Some(v) => (v.parse::<f64>().unwrap_or(0.0), true),
                    None => continue,
                },
                "attgt" => {
                    let t = n.trim_start_matches('T').parse::<f64>().unwrap_or(0.0);
                    (t, t >= gnum)
                }
                _ => continue,
            };
            pts.push((x, b[j], b[j] - crit * se[j], b[j] + crit * se[j], post));
            if kind == "group" {
                labels.push((x, n[1..].to_string()));
            }
        }
        let xt = match kind.as_str() {
            "event" => "Periods to Treatment",
            "group" => "Groups",
            "calendar" => "Periods",
            _ => "Periods",
        };
        let title = if kind == "attgt" { format!("Group {}", fmtn(gnum)) } else { String::new() };
        plot_effects(st, cmd, &pts, xt, &title, labels)
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = vec![("drdid", Box::new(Drdid)), ("csdid", Box::new(Csdid)), ("csdid_plot", Box::new(CsdidPlot))];
    v.extend(crate::did_imputation::handlers());
    v.extend(crate::didregress::handlers());
    v
}
