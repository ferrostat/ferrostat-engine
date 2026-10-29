//! `wyoung`: Westfall–Young (1993) stepdown resampling-adjusted p-values
//! for a family of hypotheses, with the null imposed by recentring the
//! bootstrap t statistics (as in Jones, Molitor & Reif 2019, QJE, online
//! appendix). Also reports Bonferroni and Sidak adjustments.
//!
//!   wyoung varlist, cmd("regress OUTCOMEVAR treat x") familyp(treat) bootstraps(#) [seed(#) cluster(var) strata(var)]
//!   wyoung, cmd("regress y1 treat" "regress y2 treat") familyp(treat) bootstraps(#) [...]

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::varlist;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, split_options};

use crate::interpreter::{AdoInterpreter, Flow};
use crate::resample::{self, arg, num_arg};

pub(crate) fn intercept(me: &mut AdoInterpreter, v: &str, rest: &str) -> Result<Option<Flow>, String> {
    if v != "wyoung" {
        return Ok(None);
    }
    wyoung(me, rest)?;
    Ok(Some(Flow::Next))
}

/// Quoted strings of an option argument (or the whole text when unquoted).
fn quoted(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut inq = false;
    for ch in s.chars() {
        if ch == '"' {
            if inq {
                out.push(std::mem::take(&mut cur));
            }
            inq = !inq;
        } else if inq {
            cur.push(ch);
        }
    }
    if out.is_empty() {
        out.push(s.trim().to_string());
    }
    out.into_iter().map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect()
}

/// Coefficient, standard error and p-value of `var` in the current e().
fn coef(me: &AdoInterpreter, var: &str) -> Result<(f64, f64, f64), String> {
    let e = &me.engine.store.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = &b.colnames;
    let j = names.iter().position(|n| n == var || n.rsplit(':').next() == Some(var)).ok_or_else(|| rc_err(111, format!("{var} not found in e(b)")))?;
    let bb = b[(0, j)];
    let se = v[(j, j)].max(0.0).sqrt();
    let t = bb / se;
    let p = match e.get("df_r") {
        Some(df) if df > 0.0 => 2.0 * dist::ttail(df, t.abs()),
        _ => 2.0 * (1.0 - dist::normal(t.abs())),
    };
    Ok((bb, se, p))
}

fn pval_like(me: &AdoInterpreter, t: f64) -> f64 {
    match me.engine.store.e.get("df_r") {
        Some(df) if df > 0.0 => 2.0 * dist::ttail(df, t.abs()),
        _ => 2.0 * (1.0 - dist::normal(t.abs())),
    }
}

fn take_rows(base: &DataSet, rows: &[usize]) -> DataSet {
    let mut d = base.clone();
    for v in d.vars.iter_mut() {
        v.data = rows.iter().map(|&r| v.data[r].clone()).collect();
    }
    d.n_obs = rows.len();
    d.sort_by.clear();
    d
}

fn wyoung(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (vl, opt_text) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim(), &rest[i + 1..]),
        None => (rest.trim(), ""),
    };
    let opts = split_options(opt_text);
    let cmd_arg = arg(&opts, "cmd", 3).ok_or_else(|| rc_err(198, "option cmd() required"))?;
    let fam = arg(&opts, "familyp", 3).ok_or_else(|| rc_err(198, "option familyp() required"))?;
    let reps = num_arg(&opts, "bootstraps", 1)?.ok_or_else(|| rc_err(198, "option bootstraps() required"))?.max(1.0) as usize;
    resample::apply_seed(me, &opts)?;
    let templates = quoted(&cmd_arg);
    let mut cmds: Vec<(String, String)> = Vec::new(); // (command, outcome)
    if !vl.is_empty() {
        let vars = varlist::expand(&me.engine.dataset, vl)?;
        if templates.len() != 1 {
            return Err(rc_err(198, "with a varlist, cmd() takes one command containing OUTCOMEVAR"));
        }
        for v in vars {
            cmds.push((templates[0].replace("OUTCOMEVAR", &v), v));
        }
    } else {
        for t in templates {
            let outcome = t.split_whitespace().nth(1).unwrap_or("").to_string();
            cmds.push((t, outcome));
        }
    }
    let fams: Vec<String> = fam.split_whitespace().map(str::to_string).collect();
    let fam_of = |k: usize| -> String { if fams.len() == cmds.len() { fams[k].clone() } else { fams[0].clone() } };
    if fams.len() != 1 && fams.len() != cmds.len() {
        return Err(rc_err(198, "familyp() needs one variable or one per model"));
    }
    let cluster: Vec<String> = arg(&opts, "cluster", 2).map(|c| c.split_whitespace().map(str::to_string).collect()).unwrap_or_default();
    let strata: Vec<String> = arg(&opts, "strata", 3).map(|c| c.split_whitespace().map(str::to_string).collect()).unwrap_or_default();
    let orig = me.engine.dataset.clone();
    let old_e = me.engine.store.e.clone();
    let kk = cmds.len();
    let restore = |me: &mut AdoInterpreter, orig: &DataSet, old_e: &ferrostat_cmd::store::Results| {
        me.engine.dataset = orig.clone();
        me.engine.store.e = old_e.clone();
    };
    // original estimates
    let mut est = Vec::with_capacity(kk);
    for (k, (c, _)) in cmds.iter().enumerate() {
        if let Err(e) = resample::run_cmd(me, c, false) {
            restore(me, &orig, &old_e);
            return Err(e);
        }
        match coef(me, &fam_of(k)) {
            Ok(x) => est.push(x),
            Err(e) => {
                restore(me, &orig, &old_e);
                return Err(e);
            }
        }
    }
    // bootstrap: recentred t statistics give null p-values
    let units = resample::units(&orig, &cluster, &strata);
    let mut pstar = vec![vec![1.0; kk]; reps];
    for row in pstar.iter_mut() {
        let mut rows = Vec::new();
        {
            let mut rng = me.engine.store.rng.borrow_mut();
            for h in &units.strata {
                let m = h.len();
                for _ in 0..m {
                    let j = ((rng.uniform() * m as f64) as usize).min(m - 1);
                    rows.extend(units.rows[h[j]].iter().copied());
                }
            }
        }
        me.engine.dataset = take_rows(&orig, &rows);
        for (k, (c, _)) in cmds.iter().enumerate() {
            if resample::run_cmd(me, c, false).is_ok() {
                if let Ok((b, se, _)) = coef(me, &fam_of(k)) {
                    if se > 0.0 {
                        row[k] = pval_like(me, (b - est[k].0) / se);
                    }
                }
            }
        }
    }
    restore(me, &orig, &old_e);
    // stepdown: order by the original p-values
    let mut order: Vec<usize> = (0..kk).collect();
    order.sort_by(|&a, &b| est[a].2.total_cmp(&est[b].2));
    let mut padj = vec![0.0; kk];
    let mut counts = vec![0usize; kk];
    for row in &pstar {
        let mut qmin = f64::INFINITY;
        for pos in (0..kk).rev() {
            let h = order[pos];
            qmin = qmin.min(row[h]);
            if qmin <= est[h].2 {
                counts[pos] += 1;
            }
        }
    }
    let mut prev: f64 = 0.0;
    for pos in 0..kk {
        let h = order[pos];
        let p = (counts[pos] as f64 / reps as f64).max(prev);
        padj[h] = p.min(1.0);
        prev = p;
    }
    let mut out = Vec::new();
    out.push(format!("Westfall-Young stepdown adjusted p-values ({reps} bootstrap replications)"));
    out.push("-".repeat(96));
    out.push(format!("{:>5} {:>12} {:>12} {:>11} {:>11} {:>9} {:>9} {:>9} {:>9}", "model", "outcome", "familyp", "coef", "stderr", "p", "pwyoung", "pbonf", "psidak"));
    out.push("-".repeat(96));
    let mut tab = Matrix::zeros(kk, 6);
    for k in 0..kk {
        let (b, se, p) = est[k];
        let pb = (p * kk as f64).min(1.0);
        let ps = 1.0 - (1.0 - p).powi(kk as i32);
        for (j, v) in [b, se, p, padj[k], pb, ps].iter().enumerate() {
            tab[(k, j)] = *v;
        }
        out.push(format!(
            "{:>5} {:>12} {:>12} {:>11} {:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            k + 1,
            ferrostat_cmd::summarize::abbrev(&cmds[k].1, 12),
            ferrostat_cmd::summarize::abbrev(&fam_of(k), 12),
            fmt_g(b, 7),
            fmt_g(se, 7),
            p,
            padj[k],
            pb,
            ps
        ));
    }
    out.push("-".repeat(96));
    let tab = tab.with_names((1..=kk).map(|k| format!("model{k}")).collect(), ["coef", "stderr", "p", "pwyoung", "pbonf", "psidak"].iter().map(|s| s.to_string()).collect());
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set_mat("table", tab);
    r.set("reps", reps as f64);
    Ok(())
}
