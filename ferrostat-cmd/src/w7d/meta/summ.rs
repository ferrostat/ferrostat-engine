//! `meta summarize`: study table, overall effect, heterogeneity, and the
//! subgroup, cumulative and leave-one-out variants.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::pool::{self, Model, Pool};
use super::{settings, studies, Settings, Studies};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

/// Pooling options of a command over the declared defaults.
pub struct Opts {
    pub model: Model,
    pub method: String,
    pub level: f64,
    pub eform: bool,
    pub pred: Option<f64>,
}

pub fn opts(cmd: &Command, st: &Store, s: &Settings) -> Result<Opts, String> {
    let (model, method) = super::model_opts(cmd, (s.model, s.method.clone()), &s.es)?;
    let level = if cmd.has_opt("level", 1) { util::level(cmd, st)? } else { s.level };
    let pred = if cmd.has_opt("predinterval", 4) {
        Some(cmd.opt("predinterval", 4).and_then(|v| v.trim().parse().ok()).unwrap_or(95.0))
    } else {
        None
    };
    Ok(Opts { model, method, level, eform: cmd.has_opt("eform", 2) || cmd.has_opt("or", 2) || cmd.has_opt("rr", 2), pred })
}

pub fn run_pool(d: &Studies, idx: &[usize], s: &Settings, o: &Opts) -> Result<Pool, String> {
    let y: Vec<f64> = idx.iter().map(|&i| d.y[i]).collect();
    let v: Vec<f64> = idx.iter().map(|&i| d.se[i] * d.se[i]).collect();
    let cells: Option<Vec<[f64; 4]>> = d.cells.as_ref().map(|c| idx.iter().map(|&i| c[i]).collect());
    pool::pool(&y, &v, o.model, &o.method, o.level, cells.as_deref(), &s.es, o.pred)
}

fn tr(x: f64, eform: bool) -> f64 {
    if eform { x.exp() } else { x }
}

fn row(lab: &str, es: f64, lo: f64, hi: f64, w: Option<f64>) -> String {
    let wt = w.map_or(String::new(), |w| if w.is_sign_negative() { format!("{:>10.4}", -w) } else { format!("{:>10.2}", w) });
    format!("{:>16} | {:>10}  {:>10}  {:>10} {}", crate::summarize::abbrev(lab, 16), format!("{es:.3}"), format!("{lo:.3}"), format!("{hi:.3}"), wt)
}

fn header(s: &Settings, o: &Opts, k: usize, p: &Pool) -> Vec<String> {
    let mut out = vec![String::new()];
    out.push(format!("  Effect-size label: {}", if o.eform { super::eform_label(&s.es).to_string() } else { s.eslabel.clone() }));
    out.push("        Effect size: _meta_es".into());
    out.push("          Std. err.: _meta_se".into());
    if !s.studylabel.is_empty() {
        out.push(format!("        Study label: {}", s.studylabel));
    }
    out.push(String::new());
    out.push(format!("{:<46}Number of studies = {:>6}", "Meta-analysis summary", k));
    out.push(format!("{:<46}Heterogeneity:", o.model.title()));
    out.push(format!("{:<46}{:>17} = {:>8.4}", format!("Method: {}", pool::method_label(&o.method)), "tau2", p.tau2));
    out.push(format!("{:<46}{:>17} = {:>8.2}", "", "I2 (%)", p.i2));
    out.push(format!("{:<46}{:>17} = {:>8.2}", "", "H2", p.h2));
    out.push(String::new());
    out
}

fn table_top(o: &Opts, s: &Settings, last: &str) -> Vec<String> {
    let _ = s;
    vec![
        "-".repeat(78),
        format!("{:>16} | {:>10}  {:>10}  {:>10} {:>10}", "Study", if o.eform { "exp(ES)" } else { "Effect size" }, format!("[{}% conf.", fmt_g(o.level, 4)), "interval]", last),
        format!("{}+{}", "-".repeat(17), "-".repeat(60)),
    ]
}

fn tests(p: &Pool, o: &Opts) -> Vec<String> {
    let mut out = Vec::new();
    if let Some((a, b)) = p.pi {
        out.push(format!("Prediction interval = {:.3} {:.3}", tr(a, o.eform), tr(b, o.eform)));
    }
    out.push(format!("Test of theta = 0: z = {:.2}{:>34}Prob > |z| = {:.4}", p.z, "", p.p));
    out.push(format!("Test of homogeneity: Q = chi2({}) = {:.2}{:>21}Prob > Q = {:.4}", p.df as i64, p.q, "", p.pq));
    out
}

fn set_r(st: &mut Store, p: &Pool, o: &Opts, s: &Settings) {
    let r = st.begin_r();
    r.set("N", p.k as f64);
    r.set("K", p.k as f64);
    r.set("theta", p.theta);
    r.set("se", p.se);
    r.set("z", p.z);
    r.set("p", p.p);
    r.set("ci_lb", p.lo);
    r.set("ci_ub", p.hi);
    r.set("tau2", p.tau2);
    r.set("I2", p.i2);
    r.set("H2", p.h2);
    r.set("Q", p.q);
    r.set("df_Q", p.df);
    r.set("p_Q", p.pq);
    r.set("level", o.level);
    if let Some((a, b)) = p.pi {
        r.set("pi_lb", a);
        r.set("pi_ub", b);
    }
    r.set_str("model", o.model.name());
    r.set_str("method", o.method.clone());
    r.set_str("estype", s.es.clone());
}

pub fn summarize(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    let o = opts(cmd, st, &s)?;
    let sub = cmd.opt("subgroup", 3).map(|v| crate::varlist::expand(ds, v).map_err(|e| rc_err(111, e))).transpose()?;
    let cum = cmd.opt("cumulative", 3).map(|v| v.split(',').next().unwrap_or("").trim().to_string());
    let extra: Vec<String> = sub.clone().unwrap_or_default().into_iter().chain(cum.clone()).collect();
    let d = studies(ds, st, cmd, &s, &extra)?;
    let all: Vec<usize> = (0..d.k()).collect();
    let p = run_pool(&d, &all, &s, &o)?;
    let quiet = cmd.has_opt("nostudies", 5) || cmd.has_opt("nometashow", 6);
    let mut out = header(&s, &o, d.k(), &p);
    if let Some(gv) = sub {
        return subgroup(ds, st, cmd, &s, &o, &d, &p, &gv, out);
    }
    if let Some(cv) = cum {
        let c = util::col(ds, &cv)?;
        let mut order: Vec<usize> = all.clone();
        let desc = cmd.opt("cumulative", 3).is_some_and(|t| t.contains("desc"));
        order.sort_by(|&a, &b| c[d.rows[a]].total_cmp(&c[d.rows[b]]));
        if desc {
            order.reverse();
        }
        out.extend(table_top(&o, &s, "p-value"));
        let mut m = Matrix::zeros(order.len(), 6);
        for t in 0..order.len() {
            let q = run_pool(&d, &order[..=t], &s, &o)?;
            out.push(row(&d.labels[order[t]], tr(q.theta, o.eform), tr(q.lo, o.eform), tr(q.hi, o.eform), Some(-q.p)));
            for (j, v) in [q.theta, q.se, q.lo, q.hi, q.p, c[d.rows[order[t]]]].iter().enumerate() {
                m[(t, j)] = *v;
            }
        }
        out.push("-".repeat(78));
        set_r(st, &p, &o, &s);
        st.r.set_mat("cumul", m.with_names(order.iter().map(|&i| d.labels[i].clone()).collect(), ["es", "se", "ci_lb", "ci_ub", "pvalue", "order"].iter().map(|x| x.to_string()).collect()));
        return Ok(out.join("\n"));
    }
    if cmd.has_opt("leaveoneout", 5) {
        if d.k() < 2 {
            return Err(rc_err(2001, "leave-one-out analysis requires at least two studies"));
        }
        out.extend(table_top(&o, &s, "p-value"));
        let li = out.len() - 2;
        out[li] = out[li].replacen("           Study", "   Omitted study", 1);
        let mut m = Matrix::zeros(d.k(), 5);
        for i in 0..d.k() {
            let idx: Vec<usize> = all.iter().copied().filter(|&j| j != i).collect();
            let q = run_pool(&d, &idx, &s, &o)?;
            out.push(row(&d.labels[i], tr(q.theta, o.eform), tr(q.lo, o.eform), tr(q.hi, o.eform), Some(-q.p)));
            for (j, v) in [q.theta, q.se, q.lo, q.hi, q.p].iter().enumerate() {
                m[(i, j)] = *v;
            }
        }
        out.push(format!("{}+{}", "-".repeat(17), "-".repeat(60)));
        out.push(row("theta", tr(p.theta, o.eform), tr(p.lo, o.eform), tr(p.hi, o.eform), Some(-p.p)));
        out.push("-".repeat(78));
        set_r(st, &p, &o, &s);
        st.r.set_mat("leaveoneout", m.with_names(d.labels.clone(), ["es", "se", "ci_lb", "ci_ub", "pvalue"].iter().map(|x| x.to_string()).collect()));
        return Ok(out.join("\n"));
    }
    out.extend(table_top(&o, &s, "% weight"));
    if !quiet {
        let crit = crate::dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
        for i in 0..d.k() {
            let (y, se) = (d.y[i], d.se[i]);
            out.push(row(&d.labels[i], tr(y, o.eform), tr(y - crit * se, o.eform), tr(y + crit * se, o.eform), Some(p.wpct[i])));
        }
        out.push(format!("{}+{}", "-".repeat(17), "-".repeat(60)));
    }
    out.push(row("theta", tr(p.theta, o.eform), tr(p.lo, o.eform), tr(p.hi, o.eform), None));
    out.push("-".repeat(78));
    out.extend(tests(&p, &o));
    set_r(st, &p, &o, &s);
    st.r.set_mat("weights", Matrix::col_vector(&p.wpct));
    Ok(out.join("\n"))
}

#[allow(clippy::too_many_arguments)]
fn subgroup(ds: &DataSet, st: &mut Store, _cmd: &Command, s: &Settings, o: &Opts, d: &Studies, p: &Pool, gv: &[String], mut out: Vec<String>) -> Result<String, String> {
    out.extend(table_top(o, s, "% weight"));
    let crit = crate::dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut lines_q = Vec::new();
    for g in gv {
        let var = ds.get_var(g).ok_or_else(|| rc_err(111, format!("variable {g} not found")))?;
        let mut levels: Vec<(String, f64)> = Vec::new();
        let num = util::col(ds, g).ok();
        for &r in &d.rows {
            let lab = util::cell_text(ds, var, r, true).trim().to_string();
            let key = num.as_ref().map_or(0.0, |c| c[r]);
            if !levels.iter().any(|(l, _)| *l == lab) {
                levels.push((lab, key));
            }
        }
        levels.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let mut thetas = Vec::new();
        let mut qw = 0.0;
        for (lab, _) in &levels {
            let idx: Vec<usize> = (0..d.k()).filter(|&i| util::cell_text(ds, var, d.rows[i], true).trim() == lab).collect();
            let q = run_pool(d, &idx, s, o)?;
            out.push(format!("{g}: {lab}"));
            for &i in &idx {
                let (y, se) = (d.y[i], d.se[i]);
                out.push(row(&d.labels[i], tr(y, o.eform), tr(y - crit * se, o.eform), tr(y + crit * se, o.eform), Some(p.wpct[i])));
            }
            out.push(row("theta", tr(q.theta, o.eform), tr(q.lo, o.eform), tr(q.hi, o.eform), None));
            out.push(format!("{}+{}", "-".repeat(17), "-".repeat(60)));
            thetas.push((q.theta, q.se));
            qw += q.q;
            rows.push(vec![q.theta, q.se, q.lo, q.hi, q.p, q.tau2, q.i2, q.q, q.df, q.k as f64]);
            names.push(format!("{g}:{lab}"));
        }
        // between-group test from the subgroup estimates
        let w: Vec<f64> = thetas.iter().map(|(_, se)| 1.0 / (se * se)).collect();
        let tb = thetas.iter().zip(&w).map(|((t, _), w)| t * w).sum::<f64>() / w.iter().sum::<f64>();
        let qb: f64 = thetas.iter().zip(&w).map(|((t, _), w)| w * (t - tb) * (t - tb)).sum();
        let dfb = thetas.len() as f64 - 1.0;
        lines_q.push(format!("Test of group differences ({g}): Q_b = chi2({}) = {:.2}{:>8}Prob > Q_b = {:.4}", dfb as i64, qb, "", crate::dist::chi2tail(dfb.max(1.0), qb)));
        let _ = qw;
        st.r.set("Q_b", qb);
        st.r.set("df_Q_b", dfb);
        st.r.set("p_Q_b", crate::dist::chi2tail(dfb.max(1.0), qb));
    }
    out.push(row("Overall theta", tr(p.theta, o.eform), tr(p.lo, o.eform), tr(p.hi, o.eform), None));
    out.push("-".repeat(78));
    out.extend(tests(p, o));
    out.extend(lines_q);
    let (qb, dfb, pqb) = (st.r.get("Q_b").unwrap_or(SYSMISS), st.r.get("df_Q_b").unwrap_or(SYSMISS), st.r.get("p_Q_b").unwrap_or(SYSMISS));
    set_r(st, p, o, s);
    st.r.set("Q_b", qb);
    st.r.set("df_Q_b", dfb);
    st.r.set("p_Q_b", pqb);
    st.r.set_mat("esgroup", Matrix::from_rows(&rows).with_names(names, ["es", "se", "ci_lb", "ci_ub", "pvalue", "tau2", "I2", "Q", "df_Q", "K"].iter().map(|x| x.to_string()).collect()));
    Ok(out.join("\n"))
}
