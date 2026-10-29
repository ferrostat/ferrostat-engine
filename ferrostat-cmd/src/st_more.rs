//! More survival-analysis commands: `stcrreg` (Fine-Gray competing-risks
//! regression), `stgen`, `stvary`, `stfill`, `stir`, `stmh`, `snapspan`,
//! `ctset` and `cttost`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::survival::{require_st, st_char, st_id};
use crate::util;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    vec![
        ("stcrreg", Box::new(Stcrreg)),
        ("stgen", Box::new(Stgen)),
        ("stvary", Box::new(Stvary)),
        ("stfill", Box::new(Stfill)),
        ("stir", Box::new(Stir)),
        ("stmh", Box::new(Stmh)),
        ("snapspan", Box::new(Snapspan)),
        ("ctset", Box::new(Ctset)),
        ("cttost", Box::new(Cttost)),
    ]
}

fn put_var(ds: &mut DataSet, name: &str, vals: Vec<f64>) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    ds.add_variable(util::new_double(name, vals))
}

/// Records in the analysis (`_st == 1`) grouped by subject, sorted by `_t`.
fn subjects(ds: &DataSet, mask: &[bool]) -> Result<Vec<Vec<usize>>, String> {
    require_st(ds)?;
    let st_ = util::col(ds, "_st")?;
    let t = util::col(ds, "_t")?;
    let id = st_id(ds);
    let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if st_[i] != 1.0 || !mask[i] {
            continue;
        }
        let key = match &id {
            Some(v) => match &ds.get_var(v).unwrap().data[i] {
                Value::Str(s) => s.clone(),
                x => format!("{:020.6}", x.num() + 1e12),
            },
            None => format!("{i:012}"),
        };
        m.entry(key).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = m.into_values().collect();
    for g in out.iter_mut() {
        g.sort_by(|a, b| t[*a].total_cmp(&t[*b]));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// stcrreg
// ---------------------------------------------------------------------------

/// Fine-Gray subdistribution hazard fit.
pub struct FgFit {
    pub b: Vec<f64>,
    pub v: Matrix,
    pub ll: f64,
    pub iters: Vec<f64>,
}

/// `x`: covariate columns; `t`: times; `status`: 0 censored, 1 event of
/// interest, 2 competing event.
pub fn fine_gray(x: &[Vec<f64>], t: &[f64], status: &[u8]) -> Result<FgFit, String> {
    let n = t.len();
    let k = x.len();
    // censoring Kaplan-Meier, left limits G(t-)
    let mut times: Vec<f64> = t.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut gl = Vec::with_capacity(times.len());
    let mut g = 1.0;
    for &s in &times {
        gl.push(g);
        let at = t.iter().filter(|&&v| v >= s).count() as f64;
        let c = (0..n).filter(|&j| t[j] == s && status[j] == 0).count() as f64;
        if at > 0.0 {
            g *= 1.0 - c / at;
        }
    }
    let gminus = |s: f64| -> f64 {
        match times.binary_search_by(|v| v.total_cmp(&s)) {
            Ok(p) => gl[p],
            Err(p) => {
                // value after the last time <= s
                if p == 0 {
                    1.0
                } else {
                    let a = gl[p - 1];
                    let c = (0..n).filter(|&j| t[j] == times[p - 1] && status[j] == 0).count() as f64;
                    let at = t.iter().filter(|&&v| v >= times[p - 1]).count() as f64;
                    a * (1.0 - c / at)
                }
            }
        }
    };
    let mut ev: Vec<f64> = (0..n).filter(|&j| status[j] == 1).map(|j| t[j]).collect();
    ev.sort_by(f64::total_cmp);
    ev.dedup();
    if ev.is_empty() {
        return Err(rc_err(2000, "no failures of interest"));
    }
    // risk weights per event time: (subject, weight)
    let risk: Vec<Vec<(usize, f64)>> = ev
        .iter()
        .map(|&s| {
            let gs = gminus(s);
            (0..n)
                .filter_map(|j| {
                    if t[j] >= s {
                        Some((j, 1.0))
                    } else if status[j] == 2 {
                        let w = gs / gminus(t[j]);
                        if w > 0.0 {
                            Some((j, w))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();
    let dsets: Vec<Vec<usize>> = ev.iter().map(|&s| (0..n).filter(|&j| t[j] == s && status[j] == 1).collect()).collect();
    let eval = |b: &[f64]| -> (f64, Vec<f64>, Matrix, Vec<(f64, Vec<f64>)>) {
        let eta: Vec<f64> = (0..n).map(|j| (0..k).map(|c| b[c] * x[c][j]).sum()).collect();
        let mut ll = 0.0;
        let mut gr = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let mut per = Vec::new();
        for (e, rs) in risk.iter().enumerate() {
            let mut s0 = 0.0;
            let mut s1 = vec![0.0; k];
            let mut s2 = Matrix::zeros(k, k);
            for &(j, w) in rs {
                let r = w * eta[j].exp();
                s0 += r;
                for a in 0..k {
                    s1[a] += r * x[a][j];
                    for c in 0..k {
                        s2[(a, c)] += r * x[a][j] * x[c][j];
                    }
                }
            }
            let d = dsets[e].len() as f64;
            for &j in &dsets[e] {
                ll += eta[j];
                for a in 0..k {
                    gr[a] += x[a][j];
                }
            }
            ll -= d * s0.ln();
            let xbar: Vec<f64> = s1.iter().map(|v| v / s0).collect();
            for a in 0..k {
                gr[a] -= d * xbar[a];
                for c in 0..k {
                    h[(a, c)] += d * (s2[(a, c)] / s0 - xbar[a] * xbar[c]);
                }
            }
            per.push((s0, xbar));
        }
        (ll, gr, h, per)
    };
    let mut b = vec![0.0; k];
    let (mut ll, mut gr, mut h, _) = eval(&b);
    let mut iters = vec![ll];
    for _ in 0..100 {
        let (hi, _) = h.sym_inverse(1e-14);
        let step: Vec<f64> = (0..k).map(|a| (0..k).map(|c| hi[(a, c)] * gr[c]).sum()).collect();
        let mut lam = 1.0;
        let mut ok = false;
        for _ in 0..30 {
            let nb: Vec<f64> = b.iter().zip(&step).map(|(p, s)| p + lam * s).collect();
            let r = eval(&nb);
            if r.0.is_finite() && r.0 >= ll - 1e-12 * ll.abs() {
                let ch = (r.0 - ll).abs();
                b = nb;
                ll = r.0;
                gr = r.1;
                h = r.2;
                ok = ch > 1e-12 * ll.abs().max(1.0);
                iters.push(ll);
                break;
            }
            lam /= 2.0;
        }
        if !ok {
            break;
        }
    }
    let (_, _, h, per) = eval(&b);
    let (hi, _) = h.sym_inverse(1e-14);
    // score residuals
    let eta: Vec<f64> = (0..n).map(|j| (0..k).map(|c| b[c] * x[c][j]).sum()).collect();
    let mut u = vec![vec![0.0; k]; n];
    for (e, rs) in risk.iter().enumerate() {
        let (s0, xbar) = &per[e];
        let d = dsets[e].len() as f64;
        for &j in &dsets[e] {
            for a in 0..k {
                u[j][a] += x[a][j] - xbar[a];
            }
        }
        for &(j, w) in rs {
            let r = w * eta[j].exp() / s0;
            for a in 0..k {
                u[j][a] -= d * r * (x[a][j] - xbar[a]);
            }
        }
    }
    let nf = n as f64;
    let (v, _) = est::sandwich(&hi, &u, None, nf / (nf - 1.0));
    Ok(FgFit { b, v, ll, iters })
}

pub struct Stcrreg;

impl CommandHandler for Stcrreg {
    fn name(&self) -> &'static str {
        "stcrreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        if cmd.main.trim().is_empty() && cmd.opt("compete", 4).is_none() {
            return st.e.macros.get("_output").cloned().filter(|_| st.e.macros.get("cmd").map(String::as_str) == Some("stcrreg")).ok_or_else(|| rc_err(301, "last estimates not found"));
        }
        let comp = cmd.opt("compete", 4).ok_or_else(|| rc_err(198, "option compete() required"))?.trim().to_string();
        let (cvar, codes) = match comp.split_once("==") {
            Some((v, l)) => (v.trim().to_string(), Some(crate::margins::numlist(l.trim())?)),
            None => (comp.clone(), None),
        };
        let cvar = crate::varlist::resolve_name(ds, &cvar)?;
        let cc = util::col(ds, &cvar)?;
        let c2 = crate::stcox::with_st_if(cmd)?;
        let d = est::build(ds, st, &c2, est::DesignSpec { depvar: None, indep: cmd.main.trim(), noconstant: true, weights: "", extra: vec!["_t".into(), "_d".into()] })?;
        let tcol = util::col(ds, "_t")?;
        let t0 = util::col(ds, "_t0")?;
        let dcol = util::col(ds, "_d")?;
        if d.rows.iter().any(|&i| t0[i] > 0.0) {
            return Err(rc_err(198, "delayed entry or multiple records per subject not supported by stcrreg"));
        }
        let act = d.active();
        let x: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
        let t: Vec<f64> = d.rows.iter().map(|&i| tcol[i]).collect();
        let status: Vec<u8> = d
            .rows
            .iter()
            .map(|&i| {
                if dcol[i] != 0.0 {
                    1
                } else {
                    let v = cc[i];
                    let is_c = match &codes {
                        Some(l) => l.iter().any(|c| *c == v),
                        None => v != 0.0 && !is_missing(v),
                    };
                    if is_c {
                        2
                    } else {
                        0
                    }
                }
            })
            .collect();
        let fit = fine_gray(&x, &t, &status)?;
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = fit.b[a];
        }
        let v = est::expand_v(&fit.v, &act, k);
        let (chi2, df_m) = est::wald_all(&d, &b, &v);
        let n = t.len();
        let nf = status.iter().filter(|&&s| s == 1).count();
        let ncmp = status.iter().filter(|&&s| s == 2).count();
        let ncen = n - nf - ncmp;
        let level = util::level(cmd, st)?;
        let fail = match st_char(ds, "st_bd") {
            Some(v) => match st_char(ds, "st_ev") {
                Some(ev) => format!("{v} == {ev}"),
                None => v,
            },
            None => "_d".into(),
        };
        let row = |l: &str, k: &str, v: String| format!("{:<49}{:<16} = {:>10}", l, k, v);
        let mut out = vec![String::new()];
        out.push(row("Competing-risks regression", "No. of obs", util::commas(n)));
        out.push(row("", "No. of subjects", util::commas(n)));
        out.push(row(&format!("Failure event:   {fail}"), "No. failed", util::commas(nf)));
        out.push(row(&format!("Competing event: {comp}"), "No. competing", util::commas(ncmp)));
        out.push(row("", "No. censored", util::commas(ncen)));
        out.push(String::new());
        out.push(row("", &format!("Wald chi2({})", fmt_g(df_m, 9)), format!("{:.2}", chi2)));
        out.push(row(&format!("Log pseudolikelihood = {}", fmt_g(fit.ll, 9)), "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))));
        out.push(String::new());
        let eform = if cmd.has_opt("noshr", 5) { None } else { Some("SHR") };
        let mut tbl = est::coef_table(&est::TableSpec { depvar: "_t", names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform, coef_title: "Coefficient" });
        let lw = tbl.last().map(|l| l.len()).unwrap_or(78).saturating_sub(66);
        tbl.insert(1, format!("{:>lw$} |{:>23}", "", "Robust"));
        out.extend(tbl);
        let mut head = vec![String::new(), "Iteration log (pseudolikelihood):".to_string()];
        head.extend(est::iteration_log(&fit.iters, "Log pseudolikelihood"));
        let text = format!("{}\n{}", head.join("\n"), out.join("\n"));
        let mut dp = d.clone();
        dp.depvar = "_t".into();
        est::post(st, "stcrreg", &dp, &b, &v, &|e: &mut Results| {
            e.set("N", n as f64);
            e.set("N_sub", n as f64);
            e.set("N_fail", nf as f64);
            e.set("N_compete", ncmp as f64);
            e.set("N_censor", ncen as f64);
            e.set("ll", fit.ll);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set_str("compete", comp.clone());
            e.set_str("crittype", "log pseudolikelihood");
            e.set_str("vce", "robust");
            e.set_str("vcetype", "Robust");
            e.set_str("title", "Competing-risks regression");
            e.set_str("chi2type", "Wald");
            e.set_str("predict", "stcrre_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// stgen
// ---------------------------------------------------------------------------

pub struct Stgen;

impl CommandHandler for Stgen {
    fn name(&self) -> &'static str {
        "stgen"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let raw = cmd.raw.trim();
        let body = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let name = lhs.split_whitespace().last().ok_or_else(|| rc_err(198, "invalid syntax"))?.to_string();
        crate::varlist::check_name(&name)?;
        if ds.get_var(&name).is_some() {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        let rhs = rhs.trim();
        let open = rhs.find('(').ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let close = rhs.rfind(')').ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let fun = rhs[..open].trim().to_lowercase();
        let arg = rhs[open + 1..close].trim().to_string();
        let mask = vec![true; ds.n_obs];
        let subs = subjects(ds, &mask)?;
        let n = ds.n_obs;
        let ev = if arg.is_empty() {
            vec![SYSMISS; n]
        } else {
            let e = ferrostat_parser::parser::parse_expression(&arg).map_err(|e| rc_err(198, e.message))?;
            util::eval_col(ds, st, &e, &mask)?
        };
        let t = util::col(ds, "_t")?;
        let t0 = util::col(ds, "_t0")?;
        let d = util::col(ds, "_d")?;
        let truth = |v: f64| !is_missing(v) && v != 0.0;
        let mut out = vec![SYSMISS; n];
        for g in &subs {
            match fun.as_str() {
                "ever" | "never" | "always" => {
                    let any = g.iter().any(|&i| truth(ev[i]));
                    let all = g.iter().all(|&i| truth(ev[i]));
                    let v = match fun.as_str() {
                        "ever" => any,
                        "never" => !any,
                        _ => all,
                    };
                    for &i in g {
                        out[i] = if v { 1.0 } else { 0.0 };
                    }
                }
                "min" | "max" => {
                    let vals: Vec<f64> = g.iter().map(|&i| ev[i]).filter(|v| !is_missing(*v)).collect();
                    let v = if vals.is_empty() { SYSMISS } else if fun == "min" { vals.iter().cloned().fold(f64::INFINITY, f64::min) } else { vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
                    for &i in g {
                        out[i] = v;
                    }
                }
                "when" | "when0" => {
                    let v = g.iter().find(|&&i| truth(ev[i])).map(|&i| if fun == "when" { t[i] } else { t0[i] }).unwrap_or(SYSMISS);
                    for &i in g {
                        out[i] = v;
                    }
                }
                "count" | "count0" => {
                    let c = g.iter().filter(|&&i| truth(ev[i])).count() as f64;
                    let c0 = g.iter().filter(|&&i| !is_missing(ev[i]) && ev[i] == 0.0).count() as f64;
                    for &i in g {
                        out[i] = if fun == "count" { c } else { c0 };
                    }
                }
                "nfailures" => {
                    let mut c = 0.0;
                    for &i in g {
                        out[i] = c;
                        if d[i] != 0.0 {
                            c += 1.0;
                        }
                    }
                }
                "ngaps" | "gaplen" | "hasgap" => {
                    let mut c = 0.0;
                    let mut has = false;
                    for (a, &i) in g.iter().enumerate() {
                        let gap = if a == 0 { 0.0 } else { (t0[i] - t[g[a - 1]]).max(0.0) };
                        if gap > 0.0 {
                            c += 1.0;
                            has = true;
                        }
                        out[i] = if fun == "ngaps" { c } else { gap };
                    }
                    if fun == "hasgap" {
                        for &i in g {
                            out[i] = if has { 1.0 } else { 0.0 };
                        }
                    }
                }
                _ => return Err(rc_err(198, format!("unknown stgen function {fun}()"))),
            }
        }
        let nmiss = out.iter().filter(|v| is_missing(**v)).count();
        let vals = out;
        let vt = util::best_numeric_type(&vals);
        let mut var = ferrostat_core::Variable::new(&name, vt);
        var.data = vals.iter().map(|&x| if is_missing(x) { Value::Missing } else { util::typed_value(vt, x) }).collect();
        ds.add_variable(var)?;
        st.data_changed = true;
        Ok(if nmiss > 0 { format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }) } else { String::new() })
    }
}

// ---------------------------------------------------------------------------
// stvary / stfill
// ---------------------------------------------------------------------------

pub struct Stvary;

impl CommandHandler for Stvary {
    fn name(&self) -> &'static str {
        "stvary"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let skip = ["_st", "_d", "_t", "_t0"];
        let vars: Vec<String> = if cmd.main.trim().is_empty() {
            ds.varlist().into_iter().filter(|v| !skip.contains(&v.as_str()) && Some(v.clone()) != st_id(ds)).collect()
        } else {
            crate::varlist::expand(ds, cmd.main.trim())?
        };
        let mask = util::touse(ds, st, cmd)?;
        let subs = subjects(ds, &mask)?;
        let mut out = vec![String::new()];
        out.push(format!("{:>18}: {}", "Failure _d", crate::survival::st_header(ds).lines().next().unwrap_or("").split_once(": ").map(|x| x.1).unwrap_or("")));
        out.push(String::new());
        out.push("         |      subjects for whom the variable is".into());
        out.push("         |                                never    always  sometimes".into());
        out.push("variable |  constant   varying         missing   missing   missing".into());
        out.push("---------+---------------------------------------------------------".into());
        let mut last = [0usize; 5];
        for v in &vars {
            let var = ds.get_var(v).unwrap();
            let val = |i: usize| -> Option<String> {
                match &var.data[i] {
                    Value::Str(s) if s.is_empty() => None,
                    Value::Str(s) => Some(s.clone()),
                    x if is_missing(x.num()) => None,
                    x => Some(format!("{}", x.num())),
                }
            };
            let mut c = [0usize; 5];
            for g in &subs {
                let vals: Vec<Option<String>> = g.iter().map(|&i| val(i)).collect();
                let nm: Vec<&String> = vals.iter().flatten().collect();
                let varies = nm.windows(2).any(|w| w[0] != w[1]);
                if varies {
                    c[1] += 1;
                } else {
                    c[0] += 1;
                }
                let nmis = vals.iter().filter(|x| x.is_none()).count();
                if nmis == 0 {
                    c[2] += 1;
                } else if nmis == vals.len() {
                    c[3] += 1;
                } else {
                    c[4] += 1;
                }
            }
            out.push(format!("{:>8} | {:>9} {:>9}       {:>9} {:>9} {:>9}", crate::summarize::abbrev(v, 8), c[0], c[1], c[2], c[3], c[4]));
            last = c;
        }
        let r = st.begin_r();
        for (k, v) in ["cons", "varies", "never", "always", "miss"].iter().zip(last) {
            r.set(k, v as f64);
        }
        Ok(out.join("\n"))
    }
}

pub struct Stfill;

impl CommandHandler for Stfill {
    fn name(&self) -> &'static str {
        "stfill"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        let baseline = cmd.has_opt("baseline", 4);
        if !baseline && !cmd.has_opt("forward", 3) {
            return Err(rc_err(198, "must specify either baseline or forward"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let subs = subjects(ds, &mask)?;
        let mut lines = Vec::new();
        for v in &vars {
            let var = ds.get_var_mut(v).unwrap();
            let miss = |x: &Value| match x {
                Value::Str(s) => s.is_empty(),
                x => is_missing(x.num()),
            };
            let mut changed = 0usize;
            for g in &subs {
                if baseline {
                    let first = var.data[g[0]].clone();
                    for &i in &g[1..] {
                        if var.data[i] != first {
                            var.data[i] = first.clone();
                            changed += 1;
                        }
                    }
                } else {
                    for a in 1..g.len() {
                        if miss(&var.data[g[a]]) && !miss(&var.data[g[a - 1]]) {
                            var.data[g[a]] = var.data[g[a - 1]].clone();
                            changed += 1;
                        }
                    }
                }
            }
            lines.push(format!("{v}: {} real change{} made", changed, if changed == 1 { "" } else { "s" }));
        }
        st.data_changed = true;
        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// stir / stmh
// ---------------------------------------------------------------------------

/// Failures and time at risk of the selected records by exposure value.
fn cases_time(ds: &DataSet, st: &Store, cmd: &Command, var: &str, extra: &[String]) -> Result<Vec<(f64, f64, f64, Vec<f64>)>, String> {
    require_st(ds)?;
    let mut mask = util::touse(ds, st, cmd)?;
    let mut mv = vec![var.to_string()];
    mv.extend(extra.iter().cloned());
    util::markout(ds, &mut mask, &mv);
    let obs = crate::survival::st_obs_mask(ds, st, &mask)?;
    let x = util::col(ds, var)?;
    let ex: Vec<Vec<f64>> = extra.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    Ok(obs.iter().map(|o| (x[o.row], if o.d { o.w } else { 0.0 }, (o.t - o.t0) * o.w, ex.iter().map(|c| c[o.row]).collect())).collect())
}

pub struct Stir;

impl CommandHandler for Stir {
    fn name(&self) -> &'static str {
        "stir"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let var = crate::varlist::resolve_name(ds, cmd.main.trim())?;
        let rows = cases_time(ds, st, cmd, &var, &[])?;
        let mut vals: Vec<f64> = rows.iter().map(|r| r.0).collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        if vals.len() != 2 {
            return Err(rc_err(450, format!("{var} takes on {} values, not 2", vals.len())));
        }
        let (mut a, mut b, mut t1, mut t0) = (0.0, 0.0, 0.0, 0.0);
        for r in &rows {
            if r.0 == vals[1] {
                a += r.1;
                t1 += r.2;
            } else {
                b += r.1;
                t0 += r.2;
            }
        }
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let (i1, i0, it) = (a / t1, b / t0, (a + b) / (t1 + t0));
        let ird = i1 - i0;
        let se = (a / (t1 * t1) + b / (t0 * t0)).sqrt();
        let irr = i1 / i0;
        let m = a + b;
        let alpha = 1.0 - level / 100.0;
        let plo = if a > 0.0 { dist::invibeta(a, m - a + 1.0, alpha / 2.0) } else { 0.0 };
        let phi = if a < m { dist::invibeta(a + 1.0, m - a, 1.0 - alpha / 2.0) } else { 1.0 };
        let conv = |p: f64| p / (1.0 - p) * t0 / t1;
        let (rlo, rhi) = (conv(plo), if phi >= 1.0 { f64::INFINITY } else { conv(phi) });
        let afe = (irr - 1.0) / irr;
        let afp = afe * a / m;
        let p0 = t1 / (t1 + t0);
        let pk = dist::binomialtail(m, a, p0) - 0.5 * dist::binomialp(m, a, p0);
        let pl = dist::binomial(m, a, p0) - 0.5 * dist::binomialp(m, a, p0);
        let one = if irr >= 1.0 { pk } else { pl };
        let c = |v: f64| fmt_g(v, 9);
        let mut out = vec![String::new()];
        out.push(crate::survival::st_header(ds));
        out.push(String::new());
        out.push(format!("                 | {:<24}|", crate::summarize::abbrev(&var, 22)));
        out.push(format!("                 | {:>9}   {:>9}  | {:>9}", "Exposed", "Unexposed", "Total"));
        out.push("-----------------+------------------------+-----------".into());
        out.push(format!("         Failure | {:>9}   {:>9}  | {:>9}", c(a), c(b), c(m)));
        out.push(format!("            Time | {:>9}   {:>9}  | {:>9}", c(t1), c(t0), c(t1 + t0)));
        out.push("-----------------+------------------------+-----------".into());
        out.push("                 |                        |".into());
        out.push(format!("  Incidence rate | {:>9}   {:>9}  | {:>9}", c(i1), c(i0), c(it)));
        out.push("                 |                        |".into());
        out.push("                 |      Point estimate    |    [95% conf. interval]".replace("95", &fmt_g(level, 4)));
        out.push("                 |------------------------+------------------------".into());
        out.push(format!(" Inc. rate diff. | {:>14}         | {:>9}   {:>9} ", c(ird), c(ird - z * se), c(ird + z * se)));
        out.push(format!(" Inc. rate ratio | {:>14}         | {:>9}   {:>9} (exact)", c(irr), c(rlo), c(rhi)));
        if irr >= 1.0 {
            out.push(format!(" Attr. frac. ex. | {:>14}         | {:>9}   {:>9} (exact)", c(afe), c((rlo - 1.0) / rlo), c((rhi - 1.0) / rhi)));
            out.push(format!(" Attr. frac. pop | {:>14}         |", c(afp)));
        } else {
            out.push(format!(" Prev. frac. ex. | {:>14}         | {:>9}   {:>9} (exact)", c(1.0 - irr), c(1.0 - rhi), c(1.0 - rlo)));
            out.push(format!(" Prev. frac. pop | {:>14}         |", c((1.0 - irr) * b / m)));
        }
        out.push("                 +-------------------------------------------------".into());
        out.push(format!("                    (midp)   Pr(k{}{}) = {:>21.4} (exact)", if irr >= 1.0 { ">=" } else { "<=" }, c(a), one));
        out.push(format!("                    (midp) 2*Pr(k{}{}) = {:>19.4} (exact)", if irr >= 1.0 { ">=" } else { "<=" }, c(a), (2.0 * one).min(1.0)));
        let r = st.begin_r();
        r.set("ird", ird);
        r.set("lb_ird", ird - z * se);
        r.set("ub_ird", ird + z * se);
        r.set("irr", irr);
        r.set("lb_irr", rlo);
        r.set("ub_irr", rhi);
        r.set("afe", afe);
        r.set("afp", afp);
        r.set("p", (2.0 * one).min(1.0));
        Ok(out.join("\n"))
    }
}

pub struct Stmh;

impl CommandHandler for Stmh {
    fn name(&self) -> &'static str {
        "stmh"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        let (var, strata) = vars.split_first().ok_or_else(|| rc_err(100, "varlist required"))?;
        let rows = cases_time(ds, st, cmd, var, strata)?;
        let mut vals: Vec<f64> = rows.iter().map(|r| r.0).collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        let (hi, lo) = if let Some(cv) = cmd.opt("compare", 1) {
            let p: Vec<f64> = cv.split(',').map(|s| s.trim().parse::<f64>().map_err(|_| rc_err(198, "invalid compare()"))).collect::<Result<_, _>>()?;
            if p.len() != 2 {
                return Err(rc_err(198, "compare() requires two values"));
            }
            (p[0], p[1])
        } else {
            if vals.len() != 2 {
                return Err(rc_err(450, format!("{var} takes on {} values; use compare()", vals.len())));
            }
            (vals[1], vals[0])
        };
        let mut cells: BTreeMap<String, (Vec<f64>, [f64; 4])> = BTreeMap::new();
        for r in &rows {
            if r.0 != hi && r.0 != lo {
                continue;
            }
            let key = r.3.iter().map(|v| format!("{:020.6}", v + 1e12)).collect::<Vec<_>>().join("|");
            let e = cells.entry(key).or_insert((r.3.clone(), [0.0; 4]));
            if r.0 == hi {
                e.1[0] += r.1;
                e.1[2] += r.2;
            } else {
                e.1[1] += r.1;
                e.1[3] += r.2;
            }
        }
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let (mut num, mut den, mut vnum, mut oe, mut vv) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut out = vec![String::new(), crate::survival::st_header(ds), String::new()];
        out.push(format!("Maximum likelihood estimate of the rate ratio"));
        out.push(format!("    Comparing {var}=={} vs. {var}=={}", fmt_g(hi, 9), fmt_g(lo, 9)));
        if !strata.is_empty() {
            out.push(format!("    controlling for {}", strata.join(" ")));
        }
        out.push(String::new());
        out.push("RR estimate, and lower and upper 95% confidence limits".replace("95", &fmt_g(level, 4)));
        out.push(String::new());
        out.push(format!("{:>14} | {:>10}  {:>10}  {:>10}", strata.first().map(|s| crate::summarize::abbrev(s, 14)).unwrap_or_default(), "RR", "Lower", "Upper"));
        out.push(format!("{}+{}", "-".repeat(15), "-".repeat(38)));
        for (_, (sv, [a, b, t1, t0])) in &cells {
            let tt = t1 + t0;
            num += a * t0 / tt;
            den += b * t1 / tt;
            let m1 = a + b;
            vnum += m1 * t1 * t0 / (tt * tt);
            oe += a - m1 * t1 / tt;
            vv += m1 * t1 * t0 / (tt * tt);
            if !strata.is_empty() {
                let rr = (a / t1) / (b / t0);
                let s = (1.0 / a + 1.0 / b).sqrt();
                out.push(format!("{:>14} | {:>10}  {:>10}  {:>10}", sv.iter().map(|v| fmt_g(*v, 9)).collect::<Vec<_>>().join(" "), fmt_g(rr, 8), fmt_g((rr.ln() - z * s).exp(), 8), fmt_g((rr.ln() + z * s).exp(), 8)));
            }
        }
        let rr = num / den;
        let s = (vnum / (num * den)).sqrt();
        let chi = oe * oe / vv;
        if !strata.is_empty() {
            out.push(format!("{}+{}", "-".repeat(15), "-".repeat(38)));
        }
        out.push(format!("{:>14} | {:>10}  {:>10}  {:>10}", if strata.is_empty() { "" } else { "Overall" }, fmt_g(rr, 8), fmt_g((rr.ln() - z * s).exp(), 8), fmt_g((rr.ln() + z * s).exp(), 8)));
        out.push(String::new());
        out.push(format!("    chi2(1) = {:>8.2}  Pr>chi2 = {:.4}", chi, dist::chi2tail(1.0, chi)));
        let r = st.begin_r();
        r.set("RR", rr);
        r.set("lb", (rr.ln() - z * s).exp());
        r.set("ub", (rr.ln() + z * s).exp());
        r.set("chi2", chi);
        r.set("p", dist::chi2tail(1.0, chi));
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// snapspan, ctset, cttost
// ---------------------------------------------------------------------------

pub struct Snapspan;

impl CommandHandler for Snapspan {
    fn name(&self) -> &'static str {
        "snapspan"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words = crate::varlist::expand(ds, cmd.main.trim())?;
        if words.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let (id, time, events) = (words[0].clone(), words[1].clone(), words[2..].to_vec());
        let gname = cmd.opt("generate", 3).map(|s| s.trim().to_string()).ok_or_else(|| rc_err(198, "option generate() required"))?;
        if !cmd.has_opt("replace", 3) && st.data_changed {
            return Err(rc_err(4, "no; data in memory would be lost"));
        }
        ds.sort(&[id.clone(), time.clone()], true)?;
        let n = ds.n_obs;
        let idv: Vec<String> = (0..n).map(|i| util::cell_text(ds, ds.get_var(&id).unwrap(), i, false)).collect();
        let t = util::col(ds, &time)?;
        let mut t0 = vec![SYSMISS; n];
        let others: Vec<String> = ds.varlist().into_iter().filter(|v| v != &id && v != &time && !events.contains(v)).collect();
        // shift the non-event variables down by one record within id
        let mut shifted: Vec<(String, Vec<Value>)> = Vec::new();
        for v in &others {
            let col = ds.get_var(v).unwrap().data.clone();
            let mut nv = col.clone();
            for i in 1..n {
                if idv[i] == idv[i - 1] {
                    nv[i] = col[i - 1].clone();
                }
            }
            shifted.push((v.clone(), nv));
        }
        for i in 1..n {
            if idv[i] == idv[i - 1] {
                t0[i] = t[i - 1];
            }
        }
        for (v, nv) in shifted {
            ds.get_var_mut(&v).unwrap().data = nv;
        }
        put_var(ds, &gname, t0)?;
        st.data_changed = true;
        Ok(String::new())
    }
}

/// `ctset time nfail [ncensor [nenter]] [, by(varlist)]`: declare count-time data.
pub struct Ctset;

impl CommandHandler for Ctset {
    fn name(&self) -> &'static str {
        "ctset"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let _ = st;
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let ch = ds.characteristics.entry("_dta".into()).or_default();
        ch.insert("ct_t".into(), vars[0].clone());
        ch.insert("ct_f".into(), vars[1].clone());
        ch.insert("ct_c".into(), vars.get(2).cloned().unwrap_or_default());
        ch.insert("ct_e".into(), vars.get(3).cloned().unwrap_or_default());
        ch.insert("ct_by".into(), cmd.opt("by", 2).unwrap_or("").trim().to_string());
        let mut out = vec![String::new(), format!("{:>9}: {}", "time", vars[0]), format!("{:>9}: {}", "no. fail", vars[1])];
        if let Some(c) = vars.get(2) {
            out.push(format!("{:>9}: {}", "no. lost", c));
        }
        if let Some(c) = vars.get(3) {
            out.push(format!("{:>9}: {}", "no. enter", c));
        }
        if let Some(b) = cmd.opt("by", 2) {
            out.push(format!("{:>9}: {}", "by", b.trim()));
        }
        Ok(out.join("\n"))
    }
}

/// `cttost`: convert count-time data to survival-time data with frequency weights.
pub struct Cttost;

impl CommandHandler for Cttost {
    fn name(&self) -> &'static str {
        "cttost"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ch = ds.characteristics.get("_dta").cloned().unwrap_or_default();
        let get = |k: &str| ch.get(k).cloned().filter(|s| !s.is_empty());
        let tv = get("ct_t").ok_or_else(|| rc_err(119, "data not ct; use ctset"))?;
        let fv = get("ct_f").unwrap();
        let cv = get("ct_c");
        if get("ct_e").is_some() {
            return Err(rc_err(198, "cttost with entry counts is not supported"));
        }
        let wname = cmd.opt("wvar", 2).map(|s| s.trim().to_string()).unwrap_or_else(|| "w".into());
        let fname = cmd.opt("failure", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| "failure".into());
        let t = util::col(ds, &tv)?;
        let f = util::col(ds, &fv)?;
        let c = match &cv {
            Some(v) => util::col(ds, v)?,
            None => vec![0.0; ds.n_obs],
        };
        // build the new records: copy each row once per nonzero count
        let n = ds.n_obs;
        let mut src = Vec::new();
        let mut fail = Vec::new();
        let mut wt = Vec::new();
        for i in 0..n {
            if is_missing(t[i]) {
                continue;
            }
            if f[i] > 0.0 {
                src.push(i);
                fail.push(1.0);
                wt.push(f[i]);
            }
            if c[i] > 0.0 {
                src.push(i);
                fail.push(0.0);
                wt.push(c[i]);
            }
        }
        let mut nds = ds.clone();
        for v in nds.vars.iter_mut() {
            v.data = src.iter().map(|&i| v.data[i].clone()).collect();
        }
        nds.n_obs = src.len();
        for name in [&fv].into_iter().chain(cv.iter()) {
            let _ = nds.drop_variable(name);
        }
        put_var(&mut nds, &fname, fail)?;
        put_var(&mut nds, &wname, wt)?;
        if let Some(c) = nds.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("ct_"));
        }
        *ds = nds;
        let line = format!("stset {tv} [fweight={wname}], failure({fname})");
        let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
        let out = crate::survival::Stset.execute(ds, st, &c)?;
        st.data_changed = true;
        Ok(format!("     Failure event: {fname} != 0 & {fname} < .\n{out}"))
    }
}
