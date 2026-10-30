//! Treatment effects after the extended regression models: `estat
//! teffects` (average treatment effects, effects on the treated and
//! potential-outcome means, with delta-method standard errors conditional
//! on the covariates) and `predict` with treatment-specific statistics
//! (`te`, `tet`, `pomean`, `base()`, `fix()`, `mean`, `ystar`, and
//! category probabilities after `eoprobit`).
//!
//! The posted coefficient vector is read back by name: outcome-equation
//! terms that involve the treatment variable are re-evaluated with the
//! treatment set to each level; the treatment equation's index and the
//! correlation between the outcome and treatment errors give the
//! conditional means for the treated (truncated bivariate normal).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::util::{bvn, delta, jacobian, mills, table_matrix, ztable, zrow, ZRow};
use crate::dist::normal;
use crate::eval::{rc_err, Env};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

const ERM: [&str; 4] = ["eregress", "eprobit", "eoprobit", "eintreg"];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Linear,
    Probit,
    Oprobit,
}

struct Info {
    kind: Kind,
    tvar: Option<String>,
    endog: bool,
    b: Vec<f64>,
    v: Matrix,
    y_idx: Vec<usize>,
    t_idx: Vec<usize>,
    levels: Vec<f64>,
    var_idx: Option<usize>,
    corr_idx: Option<usize>,
    cut_idx: Vec<usize>,
    cats: Vec<f64>,
    /// observation rows covered
    rows: Vec<usize>,
    /// x[k][t][j]: outcome-equation term j for row t with treatment at level k
    /// (index levels.len(): observed treatment)
    x: Vec<Vec<Vec<f64>>>,
    z: Vec<Vec<f64>>,
    tval: Vec<f64>,
}

fn strip_level(part: &str) -> (Option<f64>, &str) {
    let p = part.strip_prefix("c.").unwrap_or(part);
    match p.split_once('.') {
        Some((lvl, var)) => {
            let l = lvl.trim_end_matches(['b', 'o', 'n']);
            match l.parse::<f64>() {
                Ok(v) => (Some(v), var),
                Err(_) => (None, p),
            }
        }
        None => (None, p),
    }
}

/// Value of coefficient term `name` at row `i` with the treatment set to `lvl`.
fn term(env: &Env, name: &str, i: usize, tvar: Option<&str>, lvl: Option<f64>) -> Result<f64, String> {
    if name == "_cons" {
        return Ok(1.0);
    }
    let mut prod = 1.0;
    for part in name.split('#') {
        let (l, var) = strip_level(part);
        let v = match (tvar, lvl) {
            (Some(t), Some(k)) if var == t => match l {
                Some(a) => {
                    if a == k {
                        1.0
                    } else {
                        0.0
                    }
                }
                None => k,
            },
            _ => crate::postest::coef_value(env, part, i)?,
        };
        if is_missing(v) {
            return Ok(SYSMISS);
        }
        prod *= v;
    }
    Ok(prod)
}

fn info(ds: &DataSet, st: &Store, rows: Vec<usize>) -> Result<Info, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let kind = match ecmd.as_str() {
        "eprobit" => Kind::Probit,
        "eoprobit" => Kind::Oprobit,
        _ => Kind::Linear,
    };
    let bm = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = bm.colnames.clone();
    let dep = st.e.macros.get("depvar").and_then(|d| d.split_whitespace().next().map(String::from)).unwrap_or_default();
    let tvar = st.e.macros.get("treatvar").cloned();
    let yeq = names.first().and_then(|n| n.split_once(':').map(|x| x.0.to_string())).unwrap_or(dep.clone());
    let y_idx: Vec<usize> = (0..names.len()).filter(|&j| names[j].split_once(':').map(|x| x.0) == Some(yeq.as_str())).collect();
    let t_idx: Vec<usize> = match &tvar {
        Some(t) => (0..names.len()).filter(|&j| names[j].split_once(':').map(|x| x.0) == Some(t.as_str())).collect(),
        None => Vec::new(),
    };
    let endog = !t_idx.is_empty();
    let mut levels: Vec<f64> = Vec::new();
    if let Some(t) = &tvar {
        for &j in &y_idx {
            let term = names[j].split_once(':').unwrap().1;
            for part in term.split('#') {
                let (l, var) = strip_level(part);
                if var == t {
                    if let Some(l) = l {
                        if !levels.contains(&l) {
                            levels.push(l);
                        }
                    }
                }
            }
        }
        if levels.is_empty() {
            levels = vec![0.0, 1.0];
        }
        levels.sort_by(f64::total_cmp);
    }
    let var_idx = names.iter().position(|n| n == &format!("/:var(e.{yeq})"));
    let corr_idx = tvar.as_ref().and_then(|t| names.iter().position(|n| n == &format!("/:corr(e.{t},e.{yeq})") || n == &format!("/:corr(e.{yeq},e.{t})")));
    let cut_idx: Vec<usize> = (0..names.len()).filter(|&j| names[j].starts_with("/:cut")).collect();
    let cats: Vec<f64> = st.e.matrices.get("cat").map(|m| m.data.clone()).unwrap_or_else(|| (1..=cut_idx.len() + 1).map(|v| v as f64).collect());
    let env = Env::with_store(ds, st);
    let tcol = match &tvar {
        Some(t) => util::col(ds, t)?,
        None => vec![0.0; ds.n_obs],
    };
    let nl = levels.len();
    let mut x = vec![Vec::with_capacity(rows.len()); nl + 1];
    let mut z = Vec::with_capacity(rows.len());
    for &i in &rows {
        for (k, xk) in x.iter_mut().enumerate() {
            let lvl = if k < nl { Some(levels[k]) } else { None };
            let r: Vec<f64> = y_idx.iter().map(|&j| term(&env, names[j].split_once(':').unwrap().1, i, tvar.as_deref(), lvl)).collect::<Result<_, _>>()?;
            xk.push(r);
        }
        let zr: Vec<f64> = t_idx.iter().map(|&j| term(&env, names[j].split_once(':').unwrap().1, i, None, None)).collect::<Result<_, _>>()?;
        z.push(zr);
    }
    let tval: Vec<f64> = rows.iter().map(|&i| tcol[i]).collect();
    Ok(Info { kind, tvar, endog, b: bm.data.clone(), v, y_idx, t_idx, levels, var_idx, corr_idx, cut_idx, cats, rows, x, z, tval })
}

impl Info {
    fn xb(&self, b: &[f64], k: usize, t: usize) -> f64 {
        let r = &self.x[k][t];
        if r.iter().any(|v| is_missing(*v)) {
            return SYSMISS;
        }
        r.iter().zip(&self.y_idx).map(|(a, &j)| a * b[j]).sum()
    }
    fn zg(&self, b: &[f64], t: usize) -> f64 {
        self.z[t].iter().zip(&self.t_idx).map(|(a, &j)| a * b[j]).sum()
    }
    fn rho(&self, b: &[f64]) -> f64 {
        self.corr_idx.map_or(0.0, |j| b[j].clamp(-0.999999, 0.999999))
    }
    fn sigma(&self, b: &[f64]) -> f64 {
        self.var_idx.map_or(1.0, |j| b[j].max(0.0).sqrt())
    }
    /// Category probability band (lo, hi) of the latent outcome minus xb.
    fn band(&self, b: &[f64], c: usize, xb: f64) -> (f64, f64) {
        let nc = self.cut_idx.len();
        let lo = if c == 0 { f64::NEG_INFINITY } else { b[self.cut_idx[c - 1]] - xb };
        let hi = if c >= nc { f64::INFINITY } else { b[self.cut_idx[c]] - xb };
        (lo, hi)
    }
    /// Potential-outcome mean at level k for row t: unconditional
    /// (`treated` false) or conditional on being treated.
    fn po(&self, b: &[f64], k: usize, t: usize, treated: bool, cat: usize) -> f64 {
        let xb = self.xb(b, k, t);
        if is_missing(xb) {
            return SYSMISS;
        }
        let cond = treated && self.endog;
        let zg = if cond { self.zg(b, t) } else { 0.0 };
        let r = self.rho(b);
        match self.kind {
            Kind::Linear => {
                if cond {
                    xb + self.sigma(b) * r * mills(zg)
                } else {
                    xb
                }
            }
            Kind::Probit => {
                if cond {
                    bvn(xb, zg, r) / normal(zg)
                } else {
                    normal(xb)
                }
            }
            Kind::Oprobit => {
                let (lo, hi) = self.band(b, cat, xb);
                if cond {
                    (bvn(hi, zg, -r) - bvn(lo, zg, -r)) / normal(zg)
                } else {
                    let a = if hi == f64::INFINITY { 1.0 } else { normal(hi) };
                    let c = if lo == f64::NEG_INFINITY { 0.0 } else { normal(lo) };
                    a - c
                }
            }
        }
    }
}

fn level_of(info: &Info, v: f64) -> Result<usize, String> {
    info.levels.iter().position(|l| *l == v).ok_or_else(|| rc_err(198, format!("treatment level {v} not found")))
}

fn cat_index(info: &Info, cmd: &Command) -> Result<usize, String> {
    match cmd.opt("outcome", 3) {
        Some(o) => {
            let o = o.trim().trim_start_matches('#');
            let v: f64 = o.parse().map_err(|_| rc_err(198, "invalid outcome()"))?;
            if cmd.opt("outcome", 3).unwrap().trim().starts_with('#') {
                let k = v as usize;
                if k == 0 || k > info.cats.len() {
                    return Err(rc_err(198, "outcome() out of range"));
                }
                return Ok(k - 1);
            }
            info.cats.iter().position(|c| *c == v).ok_or_else(|| rc_err(198, format!("outcome {v} not found")))
        }
        None => Ok(0),
    }
}

/// `estat teffects [, ate|atet|pomean tlevel(#) outcome(#) level(#)]`.
pub fn estat_teffects(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    let inf = info(ds, st, rows)?;
    let tvar = inf.tvar.clone().ok_or_else(|| rc_err(321, "estat teffects requires a model with entreat() or extreat()"))?;
    let level = util::level(cmd, st)?;
    let stat = if cmd.has_opt("atet", 4) {
        "atet"
    } else if cmd.has_opt("pomean", 2) {
        "pomean"
    } else {
        "ate"
    };
    let tl = match util::opt_num(cmd, "tlevel", 2)? {
        Some(v) => level_of(&inf, v)?,
        None => inf.levels.len() - 1,
    };
    let cat = if inf.kind == Kind::Oprobit { cat_index(&inf, cmd)? } else { 0 };
    let nl = inf.levels.len();
    let n = inf.rows.len();
    let treated: Vec<usize> = (0..n).filter(|&t| inf.tval[t] == inf.levels[tl]).collect();
    if stat == "atet" && treated.is_empty() {
        return Err(rc_err(2000, "no treated observations"));
    }
    let atet = stat == "atet";
    let set: Vec<usize> = if atet { treated.clone() } else { (0..n).collect() };
    let poms = |b: &[f64]| -> Vec<f64> {
        (0..nl)
            .map(|k| {
                let vals: Vec<f64> = set.iter().map(|&t| inf.po(b, k, t, atet, cat)).filter(|v| !is_missing(*v)).collect();
                vals.iter().sum::<f64>() / vals.len().max(1) as f64
            })
            .collect()
    };
    let f = |b: &[f64]| -> Vec<f64> {
        let p = poms(b);
        if stat == "pomean" {
            p
        } else {
            (1..nl).map(|k| p[k] - p[0]).collect()
        }
    };
    let est = f(&inf.b);
    let j = jacobian(&f, &inf.b);
    let vv = delta(&j, &inf.v);
    let lab = |k: usize| crate::format::fmt_g(inf.levels[k], 8);
    let (eqname, labels): (String, Vec<String>) = match stat {
        "pomean" => ("POmean".into(), (0..nl).map(|k| lab(k)).collect()),
        "atet" => ("ATET".into(), (1..nl).map(|k| format!("({} vs {})", lab(k), lab(0))).collect()),
        _ => ("ATE".into(), (1..nl).map(|k| format!("({} vs {})", lab(k), lab(0))).collect()),
    };
    let se: Vec<f64> = (0..est.len()).map(|a| vv[(a, a)].max(0.0).sqrt()).collect();
    let mut out = Vec::new();
    let title = match stat {
        "pomean" => "Potential-outcome means",
        "atet" => "Average treatment effect on the treated",
        _ => "Average treatment effect",
    };
    out.push(format!("{title}{:>w$}", format!("Number of obs = {}", util::commas(set.len())), w = 78usize.saturating_sub(title.len())));
    if inf.kind == Kind::Oprobit {
        out.push(format!("Outcome: Pr({} = {})", st.e.macros.get("depvar").and_then(|d| d.split_whitespace().next().map(String::from)).unwrap_or_default(), crate::format::fmt_g(inf.cats.get(cat).copied().unwrap_or(SYSMISS), 8)));
    }
    out.push(String::new());
    let mut rows: Vec<ZRow> = Vec::new();
    for (k, l) in labels.iter().enumerate() {
        rows.push(zrow(Some(&format!("{eqname}: {tvar}")), l, est[k], se[k]));
    }
    out.extend(ztable("", "", &rows, level));
    let names: Vec<String> = labels.iter().map(|l| format!("{eqname}:{l}")).collect();
    let tm = table_matrix(&names, &est, &se, level);
    let r = st.begin_r();
    r.set_mat("b", Matrix::row_vector(&est).with_names(vec!["y1".into()], names.clone()));
    r.set_mat("V", vv.with_names(names.clone(), names));
    r.set_mat("table", tm);
    r.set("N", set.len() as f64);
    r.set("level", level);
    r.set_str("stat", stat);
    Ok(out.join("\n"))
}

fn is_erm(st: &Store) -> bool {
    ERM.contains(&st.e.macros.get("cmd").map(String::as_str).unwrap_or(""))
}

/// `estat` hook: `teffects` after the extended regression models.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if sub.len() >= 3 && "teffects".starts_with(sub) && is_erm(st) {
        return Some(estat_teffects(ds, st, cmd));
    }
    None
}

/// `predict` hook after the extended regression models: only the options
/// the basic ERM `predict` does not handle.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    if !is_erm(st) {
        return None;
    }
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let names: Vec<String> = match words.first().and_then(|w| ferrostat_core::VarType::from_type_str(w)) {
        Some(_) => words[1..].iter().map(|s| s.to_string()).collect(),
        None => words.iter().map(|s| s.to_string()).collect(),
    };
    let ours = ["te", "tet", "pomean", "base", "fix", "target", "mean", "ystar", "outcome", "tlevel"];
    let has = |o: &str| cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case(o));
    let multi = names.len() > 1 || names.iter().any(|n| n.ends_with('*'));
    if !ours.iter().any(|o| has(o)) && !(ecmd == "eoprobit" && (multi || has("pr"))) {
        return None;
    }
    Some(do_predict(ds, st, cmd, &names))
}

fn do_predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, names: &[String]) -> Result<String, String> {
    let mask = util::touse(ds, st, cmd)?;
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
    let inf = info(ds, st, rows)?;
    let has = |o: &str| cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case(o));
    for (o, _) in &cmd.opts {
        let o = o.to_lowercase();
        if !["te", "tet", "pomean", "base", "fix", "target", "mean", "ystar", "outcome", "tlevel", "pr", "xb", "nooffset"].contains(&o.as_str()) {
            return Err(rc_err(198, format!("option {o} not allowed")));
        }
    }
    let nl = inf.levels.len();
    // treatment level fixed by base(tvar = #) or tlevel(#)
    let mut fixed: Option<usize> = None;
    if let Some(bs) = cmd.opt("base", 2) {
        let tv = inf.tvar.clone().ok_or_else(|| rc_err(198, "base() requires a treatment"))?;
        let spec = bs.replace(' ', "");
        if let Some((var, val)) = spec.split_once('=') {
            if var == tv {
                let v: f64 = val.parse().map_err(|_| rc_err(198, "invalid base()"))?;
                fixed = Some(level_of(&inf, v)?);
            }
        }
    }
    if let Some(v) = util::opt_num(cmd, "tlevel", 2)? {
        fixed = Some(level_of(&inf, v)?);
    }
    let te = has("te") || has("tet");
    let pom = has("pomean");
    if (te || pom) && inf.tvar.is_none() {
        return Err(rc_err(198, "te and pomean require a model with a treatment"));
    }
    let treated_cond = has("tet");
    // output variables
    let ncat = inf.cats.len();
    let oprobit_all = inf.kind == Kind::Oprobit && !te && !pom && cmd.opt("outcome", 3).is_none() && (names.len() > 1 || names.iter().any(|n| n.ends_with('*')));
    let out_names: Vec<String> = if oprobit_all {
        if names.len() == 1 && names[0].ends_with('*') {
            let stem = names[0].trim_end_matches('*');
            (1..=ncat).map(|k| format!("{stem}{k}")).collect()
        } else {
            if names.len() != ncat {
                return Err(rc_err(102, format!("{} new variables required", ncat)));
            }
            names.to_vec()
        }
    } else {
        if names.len() != 1 {
            return Err(rc_err(198, "invalid syntax"));
        }
        names.to_vec()
    };
    for nm in &out_names {
        crate::varlist::check_name(nm)?;
        if ds.var_index.contains_key(nm) {
            return Err(rc_err(110, format!("variable {nm} already defined")));
        }
    }
    let cat = if inf.kind == Kind::Oprobit { cat_index(&inf, cmd)? } else { 0 };
    let b = inf.b.clone();
    let obs_level = nl; // x[nl]: observed treatment
    let n = ds.n_obs;
    let mut msg = String::new();
    let mut results: Vec<Vec<f64>> = vec![vec![SYSMISS; n]; out_names.len()];
    let xb_default = has("xb") || (inf.kind == Kind::Linear && !has("pr"));
    for (t, &i) in inf.rows.iter().enumerate() {
        let treated = treated_cond && inf.tval[t] == inf.levels.last().copied().unwrap_or(1.0);
        if te {
            let k1 = fixed.unwrap_or(nl - 1);
            let a = inf.po(&b, k1, t, treated, cat);
            let c = inf.po(&b, 0, t, treated, cat);
            results[0][i] = if is_missing(a) || is_missing(c) { SYSMISS } else { a - c };
        } else if pom {
            let k = fixed.unwrap_or(nl - 1);
            results[0][i] = inf.po(&b, k, t, false, cat);
        } else if oprobit_all {
            let k = fixed.unwrap_or(obs_level);
            for (c, r) in results.iter_mut().enumerate() {
                r[i] = inf.po(&b, k, t, false, c);
            }
        } else {
            let k = fixed.unwrap_or(obs_level);
            let xb = inf.xb(&b, k, t);
            results[0][i] = if is_missing(xb) {
                SYSMISS
            } else if xb_default && inf.kind != Kind::Oprobit {
                xb
            } else {
                match inf.kind {
                    Kind::Linear => xb,
                    _ => inf.po(&b, k, t, false, cat),
                }
            };
        }
    }
    if te {
        msg = "(treatment effect)".into();
    }
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    for (nm, vals) in out_names.iter().zip(results) {
        let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
        ds.add_variable(util::new_double(nm, vals)).map_err(|e| rc_err(110, e))?;
        if nmiss > 0 {
            lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
        }
    }
    Ok(lines.join("\n"))
}
