//! `teffects ra|ipw|aipw|ipwra`: average treatment effects (ATE), effects
//! on the treated (ATET) and potential-outcome means for binary and
//! multivalued treatments, with robust standard errors from the stacked
//! estimating equations of the outcome models, the treatment model and the
//! effects (M-estimation). Outcome models may be linear, logit, probit or
//! poisson; the treatment model is logit/probit (binary) or multinomial
//! logit (multivalued). `nnmatch`, `psmatch` live in `te_match`.
//!
//!   teffects ra    (ovar omvarlist [, omodel noconstant]) (tvar) [if] [in] [, ate|atet|pomeans level(#) control(#) tlevel(#)]
//!   teffects ipw   (ovar) (tvar tmvarlist [, logit|probit noconstant]) [...]
//!   teffects aipw  (ovar omvarlist [, omodel]) (tvar tmvarlist [, logit|probit]) [...]
//!   teffects ipwra (ovar omvarlist [, omodel]) (tvar tmvarlist [, logit|probit]) [...]

use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_line, split_top_level_words};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::{util, varlist};

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Ra,
    Ipw,
    Aipw,
    Ipwra,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Stat {
    Ate,
    Atet,
    Pomeans,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Link {
    Logit,
    Probit,
}

/// Outcome-model families.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum OKind {
    Linear,
    Logit,
    Probit,
    Poisson,
}

impl OKind {
    fn mean(self, z: f64) -> f64 {
        match self {
            OKind::Linear => z,
            OKind::Logit => 1.0 / (1.0 + (-z).exp()),
            OKind::Probit => dist::normal(z),
            OKind::Poisson => z.min(700.0).exp(),
        }
    }
    /// d log L / d xb.
    fn score(self, y: f64, z: f64) -> f64 {
        match self {
            OKind::Linear => y - z,
            OKind::Logit => y - self.mean(z),
            OKind::Poisson => y - self.mean(z),
            OKind::Probit => score(Link::Probit, y, z),
        }
    }
    /// expected information weight (Fisher scoring).
    fn info(self, z: f64) -> f64 {
        match self {
            OKind::Linear => 1.0,
            OKind::Logit => {
                let p = self.mean(z);
                p * (1.0 - p)
            }
            OKind::Poisson => self.mean(z),
            OKind::Probit => {
                let p = dist::normal(z).clamp(1e-12, 1.0 - 1e-12);
                let d = dist::normalden(z);
                d * d / (p * (1.0 - p))
            }
        }
    }
    fn name(self) -> &'static str {
        match self {
            OKind::Linear => "linear",
            OKind::Logit => "logit",
            OKind::Probit => "probit",
            OKind::Poisson => "Poisson",
        }
    }
}

/// `(ovar xvars [, opts])`: the words and the options of one model.
pub(crate) fn model(text: &str) -> (Vec<String>, String) {
    let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
    let (words, opts) = match inner.split_once(',') {
        Some((w, o)) => (w, o.trim().to_lowercase()),
        None => (inner, String::new()),
    };
    (words.split_whitespace().map(str::to_string).collect(), opts)
}

/// Variable names used by a factor-variable list (for marking out).
pub(crate) fn base_vars(ds: &DataSet, terms: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for t in terms {
        for part in t.split('#') {
            let name = part.rsplit('.').next().unwrap_or(part).trim();
            if !name.is_empty() {
                out.extend(varlist::expand(ds, name)?);
            }
        }
    }
    Ok(out)
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
pub(crate) fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &v)| r.iter().copied().chain(std::iter::once(v)).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, p);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                if f != 0.0 {
                    for k in c..=n {
                        m[r][k] -= f * m[c][k];
                    }
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

pub(crate) fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let e: Vec<f64> = (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        cols.push(solve(a, &e)?);
    }
    Some((0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect())
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn cdf(link: Link, z: f64) -> f64 {
    match link {
        Link::Logit => 1.0 / (1.0 + (-z).exp()),
        Link::Probit => dist::normal(z),
    }
}

/// Density of the link at the index (d p / d xb).
pub(crate) fn pdf(link: Link, z: f64) -> f64 {
    match link {
        Link::Logit => {
            let p = cdf(link, z);
            p * (1.0 - p)
        }
        Link::Probit => dist::normalden(z),
    }
}

/// Score of the treatment model for one observation: d log L / d xb.
pub(crate) fn score(link: Link, t: f64, z: f64) -> f64 {
    let p = cdf(link, z).clamp(1e-12, 1.0 - 1e-12);
    match link {
        Link::Logit => t - p,
        Link::Probit => pdf(link, z) * (t - p) / (p * (1.0 - p)),
    }
}

/// Weighted GLM fit of `y` on the rows of `x` (Fisher scoring).
pub(crate) fn fit_glm(x: &[Vec<f64>], y: &[f64], w: &[f64], kind: OKind) -> Result<Vec<f64>, String> {
    let k = x[0].len();
    let mut b = vec![0.0; k];
    if kind == OKind::Poisson {
        let (sw, sy) = w.iter().zip(y).fold((0.0, 0.0), |a, (wi, yi)| (a.0 + wi, a.1 + wi * yi));
        if let Some(c) = (0..k).find(|&a| x.iter().zip(w).all(|(r, &wi)| wi == 0.0 || r[a] == 1.0)) {
            b[c] = (sy / sw).max(1e-8).ln();
        }
    }
    for it in 0..200 {
        let mut s = vec![0.0; k];
        let mut h = vec![vec![0.0; k]; k];
        for i in 0..y.len() {
            if w[i] == 0.0 {
                continue;
            }
            let z = dot(&x[i], &b);
            let sc = w[i] * kind.score(y[i], z);
            let iw = w[i] * kind.info(z);
            for a in 0..k {
                s[a] += x[i][a] * sc;
                for c in 0..k {
                    h[a][c] += iw * x[i][a] * x[i][c];
                }
            }
        }
        let step = solve(&h, &s).ok_or_else(|| rc_err(459, "outcome model: collinear covariates"))?;
        let mx = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let damp = if mx > 5.0 && kind != OKind::Linear { 5.0 / mx } else { 1.0 };
        for a in 0..k {
            b[a] += damp * step[a];
        }
        if (kind == OKind::Linear && damp == 1.0) || mx < 1e-11 || (kind == OKind::Linear && it > 50) {
            return Ok(b);
        }
    }
    Err(rc_err(430, "outcome model: convergence not achieved"))
}

/// Maximum-likelihood fit of a binary treatment model (Fisher scoring).
pub(crate) fn fit_binary(x: &[Vec<f64>], t: &[f64], link: Link) -> Result<Vec<f64>, String> {
    let kind = if link == Link::Logit { OKind::Logit } else { OKind::Probit };
    fit_glm(x, t, &vec![1.0; t.len()], kind).map_err(|_| rc_err(430, "treatment model: convergence not achieved"))
}

/// Multinomial logit with base level 0: coefficients of levels 1..L-1.
fn fit_mlogit(x: &[Vec<f64>], lev: &[usize], nl: usize) -> Result<Vec<f64>, String> {
    let k = x[0].len();
    let np = (nl - 1) * k;
    let mut g = vec![0.0; np];
    for _ in 0..200 {
        let mut s = vec![0.0; np];
        let mut h = vec![vec![0.0; np]; np];
        for i in 0..x.len() {
            let p = mlogit_probs(&x[i], &g, nl);
            for l in 1..nl {
                let r = if lev[i] == l { 1.0 } else { 0.0 } - p[l];
                for a in 0..k {
                    s[(l - 1) * k + a] += x[i][a] * r;
                }
                for m in 1..nl {
                    let wlm = p[l] * (if l == m { 1.0 } else { 0.0 } - p[m]);
                    for a in 0..k {
                        for c in 0..k {
                            h[(l - 1) * k + a][(m - 1) * k + c] += wlm * x[i][a] * x[i][c];
                        }
                    }
                }
            }
        }
        let step = solve(&h, &s).ok_or_else(|| rc_err(430, "treatment model: convergence not achieved"))?;
        let mx = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let damp = if mx > 5.0 { 5.0 / mx } else { 1.0 };
        for a in 0..np {
            g[a] += damp * step[a];
        }
        if mx < 1e-11 {
            return Ok(g);
        }
    }
    Err(rc_err(430, "treatment model: convergence not achieved"))
}

fn mlogit_probs(x: &[f64], g: &[f64], nl: usize) -> Vec<f64> {
    let k = x.len();
    let mut e = vec![0.0; nl];
    for l in 1..nl {
        e[l] = dot(x, &g[(l - 1) * k..l * k]);
    }
    let mx = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = e.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|v| v / s).collect()
}

/// Rows of the active columns of a design.
pub(crate) fn design_rows(d: &est::Design) -> Vec<Vec<f64>> {
    let act = d.active();
    (0..d.rows.len()).map(|i| act.iter().map(|&j| d.cols[j][i]).collect()).collect()
}

/// ` if ...` / ` in ...` of a command, to pass on to sub-commands.
pub(crate) fn if_in_tail(cmd: &Command) -> String {
    let mut s = String::new();
    if let Some(i) = &cmd.if_raw {
        s.push_str(&format!(" if {i}"));
    }
    if let Some(r) = &cmd.in_range {
        s.push_str(&format!(" in {}", match r {
            ferrostat_parser::InRange::Single(a) => a.to_string(),
            ferrostat_parser::InRange::Range(a, b) => format!("{a}/{b}"),
            ferrostat_parser::InRange::Rel(a, b) => format!("{a}/{b}"),
            ferrostat_parser::InRange::First(b) => format!("f/{b}"),
            ferrostat_parser::InRange::Last(a) => format!("-{a}/l"),
        }));
    }
    s
}

/// Design of `dep indep` on the sample of `cmd`, with `extra` variables
/// marked out too.
pub(crate) fn sub_design(ds: &DataSet, st: &Store, cmd: &Command, dep: &str, indep: &[String], nocons: bool, extra: Vec<String>) -> Result<est::Design, String> {
    let c = parse_line(&format!("regress {dep} {}{}", indep.join(" "), if_in_tail(cmd))).map_err(|e| rc_err(198, e.message))?;
    est::build(ds, st, &c, DesignSpec { depvar: Some(dep), indep: &indep.join(" "), noconstant: nocons, weights: "", extra })
}

/// Label of treatment level `v`.
pub(crate) fn level_label(ds: &DataSet, tvar: &str, v: f64) -> String {
    let var = ds.get_var(tvar).unwrap();
    ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(v)).map(str::to_string).unwrap_or_else(|| fmt_g(v, 9))
}

/// One block of the coefficient table: header and rows (label lines, estimate, variance).
pub(crate) struct Block {
    pub title: String,
    pub rows: Vec<(Vec<String>, f64, f64)>,
}

/// `(Treated vs Control)` wrapped into 12-character lines.
pub(crate) fn wrap12(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for w in text.split(' ') {
        if !cur.is_empty() && cur.len() + 1 + w.len() > 12 {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    lines.push(cur);
    lines
}

/// The coefficient table of the treatment-effects estimators.
pub(crate) fn effects_table(out: &mut Vec<String>, ovar: &str, tvar: &str, vcetype: &str, level: f64, blocks: &[Block]) {
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let rule = "-".repeat(78);
    out.push(rule.clone());
    out.push(format!("{:>12} |{:>17}", "", vcetype));
    out.push(format!("{:>12} | Coefficient  std. err.      z    P>|z|     [{}% conf. interval]", crate::summarize::abbrev(ovar, 12), fmt_g(level, 4)));
    let row = |lab: &str, est: f64, var: f64| -> String {
        let se = var.max(0.0).sqrt();
        let z = est / se;
        let p = 2.0 * (1.0 - dist::normal(z.abs()));
        format!("{:>12} |  {:>9}  {:>9}  {:>7.2}   {:>5.3}    {:>9}   {:>9}", lab, fmt_g(est, 7), fmt_g(se, 7), z, p, fmt_g(est - crit * se, 7), fmt_g(est + crit * se, 7))
    };
    for b in blocks {
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        out.push(format!("{:<12} |", b.title));
        out.push(format!("{:>12} |", crate::summarize::abbrev(tvar, 12)));
        for (lines, est, var) in &b.rows {
            for l in &lines[..lines.len() - 1] {
                out.push(format!("{:>12} |", l));
            }
            out.push(row(lines.last().unwrap(), *est, *var));
        }
    }
    out.push(rule);
}

/// Parse `stat` options common to the estimators.
pub(crate) fn parse_stat(cmd: &Command) -> Stat {
    if cmd.has_opt("atet", 4) {
        Stat::Atet
    } else if cmd.has_opt("pomeans", 3) {
        Stat::Pomeans
    } else if let Some(s) = cmd.opt("stat", 4) {
        match s.trim() {
            "atet" => Stat::Atet,
            "pomeans" => Stat::Pomeans,
            _ => Stat::Ate,
        }
    } else {
        Stat::Ate
    }
}

pub struct Teffects;

impl CommandHandler for Teffects {
    fn name(&self) -> &'static str {
        "teffects"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let method = match sub {
            "ra" => Method::Ra,
            "ipw" => Method::Ipw,
            "aipw" => Method::Aipw,
            "ipwra" | "wra" => Method::Ipwra,
            "nnmatch" | "nnm" => return crate::te_match::nnmatch(ds, st, cmd, rest),
            "psmatch" | "psm" => return crate::te_match::psmatch(ds, st, cmd, rest),
            "overlap" => return crate::te_match::teoverlap(ds, st, cmd),
            _ => return Err(rc_err(198, format!("teffects {sub}: unknown estimator"))),
        };
        let groups: Vec<String> = split_top_level_words(rest).into_iter().filter(|w| w.starts_with('(')).collect();
        if groups.len() != 2 {
            return Err(rc_err(198, "teffects requires an outcome model and a treatment model in parentheses"));
        }
        let (om, oopts) = model(&groups[0]);
        let (tm, topts) = model(&groups[1]);
        let ovar = varlist::resolve_name(ds, om.first().ok_or_else(|| rc_err(198, "outcome variable required"))?)?;
        let tvar = varlist::resolve_name(ds, tm.first().ok_or_else(|| rc_err(198, "treatment variable required"))?)?;
        let ox: Vec<String> = om[1..].to_vec();
        let tx: Vec<String> = tm[1..].to_vec();
        if method == Method::Ipw && !ox.is_empty() {
            return Err(rc_err(198, "ipw: the outcome model takes no covariates"));
        }
        if method == Method::Ra && !tx.is_empty() {
            return Err(rc_err(198, "ra: the treatment model takes no covariates"));
        }
        let link = if topts.contains("probit") { Link::Probit } else { Link::Logit };
        if topts.contains("hetprobit") {
            return Err(rc_err(198, "hetprobit treatment model not implemented"));
        }
        let okind = {
            let w: Vec<&str> = oopts.split_whitespace().collect();
            if w.iter().any(|o| matches!(*o, "flogit" | "fprobit" | "hetprobit")) {
                return Err(rc_err(198, "fractional and heteroskedastic outcome models are not implemented"));
            }
            if w.contains(&"logit") {
                OKind::Logit
            } else if w.contains(&"probit") {
                OKind::Probit
            } else if w.iter().any(|o| o.starts_with("pois")) {
                OKind::Poisson
            } else {
                OKind::Linear
            }
        };
        if method == Method::Ipw && okind != OKind::Linear {
            return Err(rc_err(198, "ipw takes no outcome-model type"));
        }
        let stat = parse_stat(cmd);
        if method == Method::Aipw && stat == Stat::Atet {
            return Err(rc_err(198, "atet is not allowed with aipw"));
        }
        let level = util::level(cmd, st)?;

        let mut o_extra = vec![tvar.clone()];
        o_extra.extend(base_vars(ds, &tx)?);
        let mut t_extra = vec![ovar.clone()];
        t_extra.extend(base_vars(ds, &ox)?);
        let od = sub_design(ds, st, cmd, &ovar, &ox, oopts.contains("nocons"), o_extra)?;
        let td = sub_design(ds, st, cmd, &tvar, &tx, topts.contains("nocons"), t_extra)?;
        if od.rows != td.rows {
            return Err(rc_err(459, "outcome and treatment models use different samples"));
        }
        let n = od.rows.len();
        if n < 3 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        // treatment levels
        let traw = td.y.clone();
        let vals: Vec<f64> = {
            let s: std::collections::BTreeSet<i64> = traw.iter().map(|&v| v as i64).collect();
            s.into_iter().map(|v| v as f64).collect()
        };
        if traw.iter().any(|v| v.fract() != 0.0 || *v < 0.0) || vals.len() < 2 || vals.len() > 20 {
            return Err(rc_err(459, format!("{tvar} must be a treatment variable with nonnegative integer levels")));
        }
        let nl = vals.len();
        if nl > 2 && link == Link::Probit && method != Method::Ra {
            return Err(rc_err(198, "probit treatment model requires a binary treatment"));
        }
        let ctrl_val = match util::opt_num(cmd, "control", 4)? {
            Some(v) => v,
            None => vals[0],
        };
        let ci = vals.iter().position(|&v| v == ctrl_val).ok_or_else(|| rc_err(459, "control() is not a treatment level"))?;
        let tr_val = match util::opt_num(cmd, "tlevel", 4)? {
            Some(v) => v,
            None => *vals.iter().enumerate().filter(|(i, _)| *i != ci).map(|(_, v)| v).next_back().unwrap(),
        };
        let tri = vals.iter().position(|&v| v == tr_val).ok_or_else(|| rc_err(459, "tlevel() is not a treatment level"))?;
        // level order: control first (base of the multinomial model)
        let mut order: Vec<usize> = vec![ci];
        order.extend((0..nl).filter(|&l| l != ci));
        let lev: Vec<usize> = traw.iter().map(|v| order.iter().position(|&o| vals[o] == *v).unwrap()).collect();
        let lvals: Vec<f64> = order.iter().map(|&o| vals[o]).collect();
        let trl = order.iter().position(|&o| o == tri).unwrap();
        let ind = |i: usize, l: usize| -> f64 { if lev[i] == l { 1.0 } else { 0.0 } };
        let y = od.y.clone();
        if okind != OKind::Linear && y.iter().any(|v| *v < 0.0) {
            return Err(rc_err(459, format!("{ovar} must be nonnegative for a {} outcome model", okind.name())));
        }
        let xo = design_rows(&od);
        let xt = design_rows(&td);
        let (ko, kt) = (xo[0].len(), xt[0].len());
        let has_o = method != Method::Ipw;
        let has_t = method != Method::Ra;
        let kg = if nl == 2 { kt } else { (nl - 1) * kt };
        let atet = stat == Stat::Atet;

        // treatment model
        let gamma = if has_t {
            if nl == 2 {
                let t01: Vec<f64> = lev.iter().map(|&l| l as f64).collect();
                fit_binary(&xt, &t01, link)?
            } else {
                fit_mlogit(&xt, &lev, nl)?
            }
        } else {
            Vec::new()
        };
        let probs = |g: &[f64], i: usize| -> Vec<f64> {
            if nl == 2 {
                let p = cdf(link, dot(&xt[i], g)).clamp(1e-12, 1.0 - 1e-12);
                vec![1.0 - p, p]
            } else {
                mlogit_probs(&xt[i], g, nl)
            }
        };
        // weights of the outcome-model fits (IPWRA)
        let ipw_w = |p: &[f64], i: usize| -> f64 {
            if method != Method::Ipwra {
                return 1.0;
            }
            let l = lev[i];
            if atet { p[trl] / p[l] } else { 1.0 / p[l] }
        };
        let mut bs: Vec<Vec<f64>> = Vec::new();
        if has_o {
            let w_all: Vec<f64> = (0..n).map(|i| if has_t { ipw_w(&probs(&gamma, i), i) } else { 1.0 }).collect();
            for l in 0..nl {
                let w: Vec<f64> = (0..n).map(|i| ind(i, l) * w_all[i]).collect();
                bs.push(fit_glm(&xo, &y, &w, okind)?);
            }
        }
        // parameter layout: b_0..b_{L-1} | gamma | pom_0..pom_{L-1}
        let og = if has_o { nl * ko } else { 0 };
        let op = og + if has_t { kg } else { 0 };
        let np = op + nl;
        let moments = |th: &[f64], i: usize, m: &mut [f64]| {
            let p = if has_t { probs(&th[og..og + kg], i) } else { vec![1.0; nl] };
            let mu: Vec<f64> = if has_o { (0..nl).map(|l| okind.mean(dot(&xo[i], &th[l * ko..(l + 1) * ko]))).collect() } else { vec![0.0; nl] };
            if has_o {
                let w = ipw_w(&p, i);
                for l in 0..nl {
                    let s = if lev[i] == l { w * okind.score(y[i], dot(&xo[i], &th[l * ko..(l + 1) * ko])) } else { 0.0 };
                    for a in 0..ko {
                        m[l * ko + a] = xo[i][a] * s;
                    }
                }
            }
            if has_t {
                if nl == 2 {
                    let sc = score(link, lev[i] as f64, dot(&xt[i], &th[og..og + kt]));
                    for a in 0..kt {
                        m[og + a] = xt[i][a] * sc;
                    }
                } else {
                    for l in 1..nl {
                        let r = ind(i, l) - p[l];
                        for a in 0..kt {
                            m[og + (l - 1) * kt + a] = xt[i][a] * r;
                        }
                    }
                }
            }
            for l in 0..nl {
                let pom = th[op + l];
                m[op + l] = match method {
                    Method::Ra | Method::Ipwra => {
                        let s = if atet { ind(i, trl) } else { 1.0 };
                        s * (mu[l] - pom)
                    }
                    Method::Ipw => {
                        if atet {
                            ind(i, l) * p[trl] / p[l] * (y[i] - pom)
                        } else {
                            ind(i, l) / p[l] * (y[i] - pom)
                        }
                    }
                    Method::Aipw => mu[l] + ind(i, l) * (y[i] - mu[l]) / p[l] - pom,
                };
            }
        };
        let mut theta = vec![0.0; np];
        for (l, b) in bs.iter().enumerate() {
            theta[l * ko..(l + 1) * ko].copy_from_slice(b);
        }
        theta[og..og + gamma.len()].copy_from_slice(&gamma);
        // the POMs solve their own moments, which are linear in each POM
        {
            let mut num = vec![0.0; nl];
            let mut den = vec![0.0; nl];
            let mut m = vec![0.0; np];
            for i in 0..n {
                let mut th = theta.clone();
                th[op..].iter_mut().for_each(|v| *v = 0.0);
                moments(&th, i, &mut m);
                let a0: Vec<f64> = m[op..].to_vec();
                th[op..].iter_mut().for_each(|v| *v = 1.0);
                moments(&th, i, &mut m);
                for g in 0..nl {
                    num[g] += a0[g];
                    den[g] += a0[g] - m[op + g];
                }
            }
            for g in 0..nl {
                theta[op + g] = num[g] / den[g];
            }
        }

        let vfull = sandwich_m(&theta, n, np, &moments)?;
        let pom: Vec<f64> = theta[op..].to_vec();
        let vp: Vec<Vec<f64>> = (0..nl).map(|r| (0..nl).map(|c| vfull[op + r][op + c]).collect()).collect();

        // reported parameters: linear combinations of the POMs
        let eff = if atet { "ATET" } else { "ATE" };
        let fmtv = |v: f64| fmt_g(v, 9).trim().to_string();
        let mut names = Vec::new();
        let mut cmat: Vec<Vec<f64>> = Vec::new();
        let mut blocks = Vec::new();
        match stat {
            Stat::Pomeans => {
                let mut b = Block { title: "POmeans".into(), rows: Vec::new() };
                for l in 0..nl {
                    names.push(format!("POmeans:{}.{tvar}", fmtv(lvals[l])));
                    let mut c = vec![0.0; nl];
                    c[l] = 1.0;
                    cmat.push(c);
                    b.rows.push((vec![crate::summarize::abbrev(&level_label(ds, &tvar, lvals[l]), 12)], 0.0, 0.0));
                }
                blocks.push(b);
            }
            _ => {
                let mut b = Block { title: eff.into(), rows: Vec::new() };
                for l in 1..nl {
                    names.push(format!("{eff}:r{}vs{}.{tvar}", fmtv(lvals[l]), fmtv(lvals[0])));
                    let mut c = vec![0.0; nl];
                    c[l] = 1.0;
                    c[0] = -1.0;
                    cmat.push(c);
                    b.rows.push((wrap12(&format!("({} vs {})", level_label(ds, &tvar, lvals[l]), level_label(ds, &tvar, lvals[0]))), 0.0, 0.0));
                }
                blocks.push(b);
                names.push(format!("POmean:{}.{tvar}", fmtv(lvals[0])));
                let mut c = vec![0.0; nl];
                c[0] = 1.0;
                cmat.push(c);
                blocks.push(Block { title: "POmean".into(), rows: vec![(vec![crate::summarize::abbrev(&level_label(ds, &tvar, lvals[0]), 12)], 0.0, 0.0)] });
            }
        }
        let k = cmat.len();
        let b: Vec<f64> = cmat.iter().map(|c| dot(c, &pom)).collect();
        let v: Vec<Vec<f64>> = (0..k).map(|r| (0..k).map(|c| (0..nl).map(|a| (0..nl).map(|bb| cmat[r][a] * vp[a][bb] * cmat[c][bb]).sum::<f64>()).sum()).collect()).collect();
        let mut idx = 0;
        for bl in blocks.iter_mut() {
            for r in bl.rows.iter_mut() {
                r.1 = b[idx];
                r.2 = v[idx][idx];
                idx += 1;
            }
        }

        let est_name = match method {
            Method::Ra => "regression adjustment",
            Method::Ipw => "inverse-probability weights",
            Method::Aipw => "augmented IPW",
            Method::Ipwra => "IPW regression adjustment",
        };
        let tm_name = match (method, link, nl) {
            (Method::Ra, _, _) => "none",
            (_, _, l) if l > 2 => "(multinomial) logit",
            (_, Link::Logit, _) => "logit",
            (_, Link::Probit, _) => "probit",
        };
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Treatment-effects estimation", util::commas(n)));
        out.push(format!("Estimator      : {est_name}"));
        out.push(format!("Outcome model  : {}", match method {
            Method::Ipw => "weighted mean".to_string(),
            Method::Aipw => format!("{} by ML", okind.name()),
            _ => okind.name().to_string(),
        }));
        out.push(format!("Treatment model: {tm_name}"));
        effects_table(&mut out, &ovar, &tvar, "Robust", level, &blocks);

        let bm = Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone());
        let mut vm = Matrix::zeros(k, k);
        for r in 0..k {
            for c in 0..k {
                vm[(r, c)] = v[r][c];
            }
        }
        let vm = vm.with_names(names.clone(), names);
        let mut sample = vec![false; ds.n_obs];
        for &r in &od.rows {
            sample[r] = true;
        }
        // state for tebalance / teoverlap
        if has_t {
            let ps: Vec<Vec<f64>> = (0..n).map(|i| probs(&gamma, i)).collect();
            let w: Vec<f64> = (0..n)
                .map(|i| {
                    let l = lev[i];
                    if atet { ps[i][trl] / ps[i][l] } else { 1.0 / ps[i][l] }
                })
                .collect();
            crate::te_match::set_state(crate::te_match::TeState {
                rows: od.rows.clone(),
                lev: lev.clone(),
                lvals: lvals.clone(),
                ps,
                w,
                tvars: base_vars(ds, &tx)?,
                tx_terms: tx.clone(),
                link: if nl == 2 { Some(link) } else { None },
                xt: xt.clone(),
                atet,
                trl,
            });
        } else {
            crate::te_match::clear_state();
        }
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("level", level);
        e.set("k_levels", nl as f64);
        e.set("control", lvals[0]);
        e.set("treated", lvals[trl]);
        e.set_str("cmd", "teffects");
        e.set_str("subcmd", sub);
        e.set_str("stat", match stat {
            Stat::Ate => "ate",
            Stat::Atet => "atet",
            Stat::Pomeans => "pomeans",
        });
        e.set_str("depvar", ovar.clone());
        e.set_str("tvar", tvar.clone());
        e.set_str("tmodel", tm_name);
        e.set_str("omodel", okind.name().to_lowercase());
        e.set_str("title", "Treatment-effects estimation");
        e.set_str("vce", "robust");
        e.set_str("vcetype", "Robust");
        e.set_mat("b", bm);
        e.set_mat("V", vm);
        e.sample = Some(sample);
        Ok(out.join("\n"))
    }
}

/// M-estimation sandwich A^-1 B A^-T / n with a numerical Jacobian.
fn sandwich_m(theta: &[f64], n: usize, np: usize, moments: &dyn Fn(&[f64], usize, &mut [f64])) -> Result<Vec<Vec<f64>>, String> {
    let mean_m = |th: &[f64]| -> Vec<f64> {
        let mut acc = vec![0.0; np];
        let mut m = vec![0.0; np];
        for i in 0..n {
            moments(th, i, &mut m);
            for a in 0..np {
                acc[a] += m[a];
            }
        }
        acc.iter().map(|v| v / n as f64).collect()
    };
    let mut a_mat = vec![vec![0.0; np]; np];
    for c in 0..np {
        let h = 1e-6 * theta[c].abs().max(1e-3);
        let mut tp = theta.to_vec();
        tp[c] += h;
        let up = mean_m(&tp);
        tp[c] -= 2.0 * h;
        let dn = mean_m(&tp);
        for r in 0..np {
            a_mat[r][c] = (up[r] - dn[r]) / (2.0 * h);
        }
    }
    let mut b_mat = vec![vec![0.0; np]; np];
    let mut m = vec![0.0; np];
    for i in 0..n {
        moments(theta, i, &mut m);
        for r in 0..np {
            for c in 0..np {
                b_mat[r][c] += m[r] * m[c] / n as f64;
            }
        }
    }
    let ai = invert(&a_mat).ok_or_else(|| rc_err(459, "estimating equations are singular"))?;
    let tmp: Vec<Vec<f64>> = (0..np).map(|r| (0..np).map(|l| (0..np).map(|k| ai[r][k] * b_mat[k][l]).sum()).collect()).collect();
    Ok((0..np).map(|r| (0..np).map(|c| (0..np).map(|l| tmp[r][l] * ai[c][l]).sum::<f64>() / n as f64).collect()).collect())
}
