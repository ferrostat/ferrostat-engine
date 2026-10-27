//! `teffects ra|ipw|aipw` for a binary treatment: average treatment effects
//! (ATE), effects on the treated (ATET) and potential-outcome means, with
//! robust standard errors from the stacked estimating equations of the
//! outcome models, the treatment model and the effects (M-estimation).
//!
//!   teffects ra   (ovar omvarlist [, noconstant]) (tvar) [if] [in] [, ate|atet|pomeans level(#) aequations]
//!   teffects ipw  (ovar) (tvar tmvarlist [, logit|probit noconstant]) [...]
//!   teffects aipw (ovar omvarlist [, noconstant]) (tvar tmvarlist [, logit|probit noconstant]) [...]

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
}

#[derive(Clone, Copy, PartialEq)]
enum Stat {
    Ate,
    Atet,
    Pomeans,
}

#[derive(Clone, Copy, PartialEq)]
enum Link {
    Logit,
    Probit,
}

/// `(ovar xvars [, opts])`: the words and the options of one model.
fn model(text: &str) -> (Vec<String>, String) {
    let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
    let (words, opts) = match inner.split_once(',') {
        Some((w, o)) => (w, o.trim().to_lowercase()),
        None => (inner, String::new()),
    };
    (words.split_whitespace().map(str::to_string).collect(), opts)
}

/// Variable names used by a factor-variable list (for marking out).
fn base_vars(ds: &DataSet, terms: &[String]) -> Result<Vec<String>, String> {
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
fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &v)| r.iter().copied().chain(std::iter::once(v)).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap())?;
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

fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let e: Vec<f64> = (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        cols.push(solve(a, &e)?);
    }
    Some((0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// OLS of `y` on the rows of `x` where `keep`.
fn ols(x: &[Vec<f64>], y: &[f64], keep: &dyn Fn(usize) -> bool) -> Option<Vec<f64>> {
    let k = x[0].len();
    let mut xx = vec![vec![0.0; k]; k];
    let mut xy = vec![0.0; k];
    for i in 0..y.len() {
        if keep(i) {
            for a in 0..k {
                xy[a] += x[i][a] * y[i];
                for b in 0..k {
                    xx[a][b] += x[i][a] * x[i][b];
                }
            }
        }
    }
    solve(&xx, &xy)
}

fn cdf(link: Link, z: f64) -> f64 {
    match link {
        Link::Logit => 1.0 / (1.0 + (-z).exp()),
        Link::Probit => dist::normal(z),
    }
}

/// Score of the treatment model for one observation: d log L / d xb.
fn score(link: Link, t: f64, z: f64) -> f64 {
    let p = cdf(link, z).clamp(1e-12, 1.0 - 1e-12);
    match link {
        Link::Logit => t - p,
        Link::Probit => {
            let d = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            d * (t - p) / (p * (1.0 - p))
        }
    }
}

/// Maximum-likelihood fit of the treatment model by Newton's method with a
/// numerical Hessian of the score (the models are small).
fn fit_binary(x: &[Vec<f64>], t: &[f64], link: Link) -> Result<Vec<f64>, String> {
    let k = x[0].len();
    let mut g = vec![0.0; k];
    let grad = |g: &[f64]| -> Vec<f64> {
        let mut s = vec![0.0; k];
        for (xi, &ti) in x.iter().zip(t) {
            let sc = score(link, ti, dot(xi, g));
            for a in 0..k {
                s[a] += xi[a] * sc;
            }
        }
        s
    };
    for _ in 0..100 {
        let s = grad(&g);
        let mut h = vec![vec![0.0; k]; k];
        for (xi, &ti) in x.iter().zip(t) {
            let z = dot(xi, &g);
            let w = match link {
                Link::Logit => {
                    let p = cdf(link, z);
                    p * (1.0 - p)
                }
                Link::Probit => {
                    let e = 1e-6;
                    -(score(link, ti, z + e) - score(link, ti, z - e)) / (2.0 * e)
                }
            };
            for a in 0..k {
                for b in 0..k {
                    h[a][b] += w * xi[a] * xi[b];
                }
            }
        }
        let step = solve(&h, &s).ok_or_else(|| rc_err(430, "treatment model: convergence not achieved"))?;
        for a in 0..k {
            g[a] += step[a];
        }
        if step.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-10 {
            return Ok(g);
        }
    }
    Err(rc_err(430, "treatment model: convergence not achieved"))
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
            "psmatch" | "nnmatch" | "ipwra" | "overlap" => return Err(rc_err(199, format!("teffects {sub} is not implemented yet"))),
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
        if oopts.split_whitespace().any(|o| matches!(o, "logit" | "probit" | "poisson" | "flogit" | "fprobit" | "hetprobit")) {
            return Err(rc_err(198, "only linear outcome models are implemented"));
        }
        let stat = if cmd.has_opt("atet", 4) {
            Stat::Atet
        } else if cmd.has_opt("pomeans", 3) {
            Stat::Pomeans
        } else {
            Stat::Ate
        };
        if method == Method::Aipw && stat == Stat::Atet {
            return Err(rc_err(198, "atet is not allowed with aipw"));
        }
        let level = util::level(cmd, st)?;

        // one estimation sample for both models
        let tail = {
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
        };
        let sub_cmd = |dep: &str, x: &[String]| -> Result<Command, String> { parse_line(&format!("regress {dep} {}{tail}", x.join(" "))).map_err(|e| rc_err(198, e.message)) };
        let mut o_extra = vec![tvar.clone()];
        o_extra.extend(base_vars(ds, &tx)?);
        let mut t_extra = vec![ovar.clone()];
        t_extra.extend(base_vars(ds, &ox)?);
        let ocmd = sub_cmd(&ovar, &ox)?;
        let tcmd = sub_cmd(&tvar, &tx)?;
        let od = est::build(ds, st, &ocmd, DesignSpec { depvar: Some(&ovar), indep: &ox.join(" "), noconstant: oopts.contains("nocons"), weights: "", extra: o_extra })?;
        let td = est::build(ds, st, &tcmd, DesignSpec { depvar: Some(&tvar), indep: &tx.join(" "), noconstant: topts.contains("nocons"), weights: "", extra: t_extra })?;
        if od.rows != td.rows {
            return Err(rc_err(459, "outcome and treatment models use different samples"));
        }
        let n = od.rows.len();
        if n < 3 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let t: Vec<f64> = td.y.clone();
        let tvals: std::collections::BTreeSet<i64> = t.iter().map(|&v| v as i64).collect();
        if t.iter().any(|v| v.fract() != 0.0) || tvals.len() != 2 || !tvals.contains(&0) {
            return Err(rc_err(459, format!("{tvar} must be a binary 0/1 treatment variable")));
        }
        let t: Vec<f64> = t.iter().map(|&v| if v != 0.0 { 1.0 } else { 0.0 }).collect();
        let y = od.y.clone();
        let rows_of = |d: &est::Design| -> Vec<Vec<f64>> {
            let act = d.active();
            (0..n).map(|i| act.iter().map(|&j| d.cols[j][i]).collect()).collect()
        };
        let xo = rows_of(&od);
        let xt = rows_of(&td);
        let (ko, kt) = (xo[0].len(), xt[0].len());

        // point estimates
        let fit_o = |grp: f64| ols(&xo, &y, &|i| t[i] == grp).ok_or_else(|| rc_err(459, "outcome model: collinear covariates"));
        let (b0, b1) = if method == Method::Ipw { (Vec::new(), Vec::new()) } else { (fit_o(0.0)?, fit_o(1.0)?) };
        let gamma = if method == Method::Ra { Vec::new() } else { fit_binary(&xt, &t, link)? };
        // parameter layout: b0 | b1 | gamma | pom0 | pom1
        let (o0, o1, og, op) = (0, ko * usize::from(!b0.is_empty()), 2 * ko * usize::from(!b0.is_empty()), 2 * ko * usize::from(!b0.is_empty()) + gamma.len());
        let np = op + 2;
        let atet = stat == Stat::Atet;
        // moments of observation i at theta
        let moments = |th: &[f64], i: usize, m: &mut [f64]| {
            let has_o = method != Method::Ipw;
            let has_t = method != Method::Ra;
            let mu0 = if has_o { dot(&xo[i], &th[o0..o0 + ko]) } else { 0.0 };
            let mu1 = if has_o { dot(&xo[i], &th[o1..o1 + ko]) } else { 0.0 };
            if has_o {
                for a in 0..ko {
                    m[o0 + a] = (1.0 - t[i]) * xo[i][a] * (y[i] - mu0);
                    m[o1 + a] = t[i] * xo[i][a] * (y[i] - mu1);
                }
            }
            let mut p = 0.5;
            if has_t {
                let z = dot(&xt[i], &th[og..og + kt]);
                let sc = score(link, t[i], z);
                for a in 0..kt {
                    m[og + a] = xt[i][a] * sc;
                }
                p = cdf(link, z).clamp(1e-12, 1.0 - 1e-12);
            }
            let (pom0, pom1) = (th[op], th[op + 1]);
            match method {
                Method::Ra => {
                    let s = if atet { t[i] } else { 1.0 };
                    m[op] = s * (mu0 - pom0);
                    m[op + 1] = s * (mu1 - pom1);
                }
                Method::Ipw => {
                    if atet {
                        m[op] = (1.0 - t[i]) * p / (1.0 - p) * (y[i] - pom0);
                        m[op + 1] = t[i] * (y[i] - pom1);
                    } else {
                        m[op] = (1.0 - t[i]) / (1.0 - p) * (y[i] - pom0);
                        m[op + 1] = t[i] / p * (y[i] - pom1);
                    }
                }
                Method::Aipw => {
                    m[op] = mu0 + (1.0 - t[i]) * (y[i] - mu0) / (1.0 - p) - pom0;
                    m[op + 1] = mu1 + t[i] * (y[i] - mu1) / p - pom1;
                }
            }
        };
        let mut theta = vec![0.0; np];
        theta[o0..o0 + b0.len()].copy_from_slice(&b0);
        theta[o1..o1 + b1.len()].copy_from_slice(&b1);
        theta[og..og + gamma.len()].copy_from_slice(&gamma);
        // the POMs solve their own moments given the other parameters
        {
            let mut num = [0.0; 2];
            let mut den = [0.0; 2];
            let mut m = vec![0.0; np];
            for i in 0..n {
                // the moments are linear in each POM with slope -(weight)
                let mut th = theta.clone();
                th[op] = 0.0;
                th[op + 1] = 0.0;
                moments(&th, i, &mut m);
                let (a0, a1) = (m[op], m[op + 1]);
                th[op] = 1.0;
                th[op + 1] = 1.0;
                moments(&th, i, &mut m);
                for (g, a) in [(0, a0), (1, a1)] {
                    num[g] += a;
                    den[g] += a - m[op + g];
                }
            }
            theta[op] = num[0] / den[0];
            theta[op + 1] = num[1] / den[1];
        }

        // sandwich: A = d mean(m)/d theta, B = mean(m m')
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
            let mut tp = theta.clone();
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
            moments(&theta, i, &mut m);
            for r in 0..np {
                for c in 0..np {
                    b_mat[r][c] += m[r] * m[c] / n as f64;
                }
            }
        }
        let ai = invert(&a_mat).ok_or_else(|| rc_err(459, "estimating equations are singular"))?;
        let vfull: Vec<Vec<f64>> = (0..np)
            .map(|r| (0..np).map(|c| (0..np).map(|k| (0..np).map(|l| ai[r][k] * b_mat[k][l] * ai[c][l]).sum::<f64>()).sum::<f64>() / n as f64).collect())
            .collect();
        let (v00, v11, v01) = (vfull[op][op], vfull[op + 1][op + 1], vfull[op][op + 1]);
        let (pom0, pom1) = (theta[op], theta[op + 1]);

        // reported parameters
        let tlab = |v: f64| -> String {
            let var = ds.get_var(&tvar).unwrap();
            ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(v)).map(str::to_string).unwrap_or_else(|| fmt_g(v, 9))
        };
        let eff = if atet { "ATET" } else { "ATE" };
        let (names, b, v): (Vec<String>, Vec<f64>, [[f64; 2]; 2]) = match stat {
            Stat::Pomeans => (vec![format!("POmeans:0.{tvar}"), format!("POmeans:1.{tvar}")], vec![pom0, pom1], [[v00, v01], [v01, v11]]),
            _ => {
                // (pom1 - pom0, pom0)
                let ve = v11 + v00 - 2.0 * v01;
                let c = v01 - v00;
                (vec![format!("{eff}:r1vs0.{tvar}"), format!("POmean:0.{tvar}")], vec![pom1 - pom0, pom0], [[ve, c], [c, v00]])
            }
        };

        let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let est_name = match method {
            Method::Ra => "regression adjustment",
            Method::Ipw => "inverse-probability weights",
            Method::Aipw => "augmented IPW",
        };
        let tm_name = match (method, link) {
            (Method::Ra, _) => "none",
            (_, Link::Logit) => "logit",
            (_, Link::Probit) => "probit",
        };
        let mut out = Vec::new();
        out.push(format!("{:<48}Number of obs     = {:>10}", "Treatment-effects estimation", util::commas(n)));
        out.push(format!("Estimator      : {est_name}"));
        out.push(format!("Outcome model  : {}", match method {
            Method::Ra => "linear",
            Method::Ipw => "weighted mean",
            Method::Aipw => "linear by ML",
        }));
        out.push(format!("Treatment model: {tm_name}"));
        let rule = "-".repeat(78);
        out.push(rule.clone());
        out.push(format!("{:>12} |{:>17}", "", "Robust"));
        out.push(format!("{:>12} | Coefficient  std. err.      z    P>|z|     [{}% conf. interval]", crate::summarize::abbrev(&ovar, 12), fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        let row = |lab: &str, est: f64, var: f64| -> String {
            let se = var.max(0.0).sqrt();
            let z = est / se;
            let p = 2.0 * (1.0 - dist::normal(z.abs()));
            format!("{:>12} |  {:>9}  {:>9}  {:>7.2}   {:>5.3}    {:>9}   {:>9}", lab, fmt_g(est, 7), fmt_g(se, 7), z, p, fmt_g(est - crit * se, 7), fmt_g(est + crit * se, 7))
        };
        // `(Treated vs Control)` wrapped into 12-character lines
        let wrap = |text: &str| -> Vec<String> {
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
        };
        match stat {
            Stat::Pomeans => {
                out.push(format!("{:<12} |", "POmeans"));
                out.push(format!("{:>12} |", crate::summarize::abbrev(&tvar, 12)));
                out.push(row(&crate::summarize::abbrev(&tlab(0.0), 12), b[0], v[0][0]));
                out.push(row(&crate::summarize::abbrev(&tlab(1.0), 12), b[1], v[1][1]));
            }
            _ => {
                out.push(format!("{:<12} |", eff));
                out.push(format!("{:>12} |", crate::summarize::abbrev(&tvar, 12)));
                let lines = wrap(&format!("({} vs {})", tlab(1.0), tlab(0.0)));
                for l in &lines[..lines.len() - 1] {
                    out.push(format!("{:>12} |", l));
                }
                out.push(row(lines.last().unwrap(), b[0], v[0][0]));
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
                out.push(format!("{:<12} |", "POmean"));
                out.push(format!("{:>12} |", crate::summarize::abbrev(&tvar, 12)));
                out.push(row(&crate::summarize::abbrev(&tlab(0.0), 12), b[1], v[1][1]));
            }
        }
        out.push(rule);

        let bm = Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone());
        let mut vm = Matrix::zeros(2, 2);
        for r in 0..2 {
            for c in 0..2 {
                vm[(r, c)] = v[r][c];
            }
        }
        let vm = vm.with_names(names.clone(), names);
        let mut sample = vec![false; ds.n_obs];
        for &r in &od.rows {
            sample[r] = true;
        }
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("level", level);
        e.set_str("cmd", "teffects");
        e.set_str("subcmd", sub);
        e.set_str("stat", match stat {
            Stat::Ate => "ate",
            Stat::Atet => "atet",
            Stat::Pomeans => "pomeans",
        });
        e.set_str("depvar", ovar.clone());
        e.set_str("tvar", tvar.clone());
        e.set_str("title", "Treatment-effects estimation");
        e.set_str("vce", "robust");
        e.set_str("vcetype", "Robust");
        e.set_mat("b", bm);
        e.set_mat("V", vm);
        e.sample = Some(sample);
        Ok(out.join("\n"))
    }
}
