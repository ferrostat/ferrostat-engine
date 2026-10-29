//! Panel description and transformation commands: `xtdescribe`, `xtsum`,
//! `xttab`, `xttrans`, `xtdata` and `quadchk`.

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::Store;
use crate::util;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    vec![
        ("xtdescribe", Box::new(XtDescribe)),
        ("xtdes", Box::new(XtDescribe)),
        ("xtsum", Box::new(XtSum)),
        ("xttab", Box::new(XtTab)),
        ("xttrans", Box::new(XtTrans)),
        ("xtdata", Box::new(XtData)),
        ("quadchk", Box::new(QuadChk)),
        ("xtline", Box::new(XtLine)),
    ]
}

/// Panel and (optional) time variable.
fn panel_time(ds: &DataSet, st: &Store, cmd: &Command, need_t: bool) -> Result<(String, Option<String>), String> {
    let p = crate::xtnl::panel_var(ds, st, cmd)?;
    let t = match cmd.opt("t", 1) {
        Some(t) => Some(crate::varlist::resolve_name(ds, t.trim())?),
        None => st.tsset.as_ref().map(|(_, t)| t.clone()).filter(|t| !t.is_empty()),
    };
    if need_t && t.is_none() {
        return Err(rc_err(459, "time variable not set; use xtset panelvar timevar"));
    }
    Ok((p, t))
}

/// Key of a panel identifier value (numeric or string).
fn key(ds: &DataSet, var: &str, i: usize) -> String {
    match ds.get_var(var).map(|v| &v.data[i]) {
        Some(Value::Str(s)) => s.clone(),
        Some(v) => format!("{:020.6}", v.num() + 1e12),
        None => String::new(),
    }
}

/// Panels (sorted by identifier) of the rows selected by `mask`.
fn panels(ds: &DataSet, ivar: &str, mask: &[bool]) -> Vec<Vec<usize>> {
    let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..ds.n_obs {
        if mask[i] {
            m.entry(key(ds, ivar, i)).or_default().push(i);
        }
    }
    m.into_values().collect()
}

fn value_text(ds: &DataSet, var: &str, v: f64) -> String {
    let lab = ds.get_var(var).map(|x| x.value_label_name.clone()).unwrap_or_default();
    if !lab.is_empty() {
        if let Some(vl) = ds.value_labels.get(&lab) {
            if let Some(t) = vl.mapping.get(&ferrostat_core::value_label::FloatKey(v)) {
                return t.clone();
            }
        }
    }
    fmt_g(v, 9)
}

// ---------------------------------------------------------------------------
// xtdescribe
// ---------------------------------------------------------------------------

pub struct XtDescribe;

impl CommandHandler for XtDescribe {
    fn name(&self) -> &'static str {
        "xtdescribe"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (ivar, tvar) = panel_time(ds, st, cmd, true)?;
        let tvar = tvar.unwrap();
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[ivar.clone(), tvar.clone()]);
        let tcol = util::col(ds, &tvar)?;
        let groups = panels(ds, &ivar, &mask);
        let ng = groups.len();
        if ng == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let mut times: Vec<f64> = groups.iter().flatten().map(|&i| tcol[i]).collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let tmin = times[0];
        let tmax = *times.last().unwrap();
        // delta: gcd of the gaps between distinct times
        let mut delta = 0.0f64;
        for w in times.windows(2) {
            let mut a = w[1] - w[0];
            let mut b = delta;
            while b > 1e-9 {
                let r = a % b;
                a = b;
                b = r;
            }
            delta = a;
        }
        if delta <= 0.0 {
            delta = 1.0;
        }
        let span = ((tmax - tmin) / delta).round() as usize + 1;
        let mut unique = true;
        let mut ti: Vec<usize> = Vec::with_capacity(ng);
        let mut pats: HashMap<String, usize> = HashMap::new();
        let mut any = vec![false; span];
        for g in &groups {
            let mut pat = vec![b'.'; span];
            let mut seen = std::collections::HashSet::new();
            for &i in g {
                let pos = ((tcol[i] - tmin) / delta).round() as usize;
                if !seen.insert(pos) {
                    unique = false;
                }
                pat[pos] = b'1';
                any[pos] = true;
            }
            ti.push(seen.len());
            *pats.entry(String::from_utf8(pat).unwrap()).or_insert(0) += 1;
        }
        let first_last = |v: &str, vals: &[f64]| -> String {
            let txt: Vec<String> = vals.iter().map(|x| fmt_g(*x, 9)).collect();
            if txt.len() <= 3 {
                txt.join(", ")
            } else {
                format!("{}, {}, ..., {}", txt[0], txt[1], txt[txt.len() - 1]).replace("{v}", v)
            }
        };
        let ids: Vec<f64> = groups.iter().map(|g| ds.get_var(&ivar).map(|v| v.data[g[0]].num()).unwrap_or(SYSMISS)).collect();
        let w = ivar.len().max(tvar.len()).max(6);
        let mut out = vec![String::new()];
        out.push(format!("{:>w$}:  {:<45}n = {:>10}", ivar, first_last(&ivar, &ids), util::commas(ng)));
        out.push(format!("{:>w$}:  {:<45}T = {:>10}", tvar, first_last(&tvar, &times), util::commas(times.len())));
        out.push(format!("{:>w$}   Delta({tvar}) = {} unit{}", "", fmt_g(delta, 9), if delta == 1.0 { "" } else { "s" }));
        out.push(format!("{:>w$}   Span({tvar})  = {} periods", "", span));
        if unique {
            out.push(format!("{:>w$}   ({ivar}*{tvar} uniquely identifies each observation)", ""));
        } else {
            out.push(format!("{:>w$}   ({ivar}*{tvar} does not uniquely identify observations)", ""));
        }
        let mut sorted = ti.clone();
        sorted.sort();
        let sf: Vec<f64> = sorted.iter().map(|&x| x as f64).collect();
        let pc = |p: f64| crate::util::pctile_sorted(&sf, p);
        out.push(String::new());
        out.push("Distribution of T_i:   min      5%     25%       50%       75%     95%     max".into());
        out.push(format!(
            "{:>26}{:>8}{:>8}{:>10}{:>10}{:>8}{:>8}",
            fmt_g(sf[0], 9),
            fmt_g(pc(5.0), 9),
            fmt_g(pc(25.0), 9),
            fmt_g(pc(50.0), 9),
            fmt_g(pc(75.0), 9),
            fmt_g(pc(95.0), 9),
            fmt_g(sf[sf.len() - 1], 9)
        ));
        let npat = util::opt_num(cmd, "patterns", 1)?.unwrap_or(9.0).max(0.0) as usize;
        let mut plist: Vec<(String, usize)> = pats.into_iter().collect();
        plist.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let rule = format!(" {}+{}", "-".repeat(27), "-".repeat(span.max(21) + 2));
        out.push(String::new());
        out.push("     Freq.  Percent    Cum. |  Pattern".into());
        out.push(rule.clone());
        let mut cum = 0usize;
        let ngf = ng as f64;
        for (p, c) in plist.iter().take(npat) {
            cum += c;
            out.push(format!("{:>10}{:>9.2}{:>8.2} |  {}", util::commas(*c), 100.0 * *c as f64 / ngf, 100.0 * cum as f64 / ngf, p));
        }
        if plist.len() > npat {
            let rest: usize = plist.iter().skip(npat).map(|x| x.1).sum();
            cum += rest;
            out.push(format!("{:>10}{:>9.2}{:>8.2} | (other patterns)", util::commas(rest), 100.0 * rest as f64 / ngf, 100.0 * cum as f64 / ngf));
        }
        out.push(rule);
        let all: String = any.iter().map(|&a| if a { 'X' } else { '.' }).collect();
        out.push(format!("{:>10}{:>9.2}{:>8} |  {}", util::commas(ng), 100.0, "", all));
        let r = st.begin_r();
        r.set("n", ngf);
        r.set("N", ngf);
        r.set("T", times.len() as f64);
        r.set("min", sf[0]);
        r.set("max", sf[sf.len() - 1]);
        r.set("mean", sf.iter().sum::<f64>() / ngf);
        r.set("pct5", pc(5.0));
        r.set("pct25", pc(25.0));
        r.set("pct50", pc(50.0));
        r.set("pct75", pc(75.0));
        r.set("pct95", pc(95.0));
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// xtsum
// ---------------------------------------------------------------------------

pub struct XtSum;

impl CommandHandler for XtSum {
    fn name(&self) -> &'static str {
        "xtsum"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { crate::varlist::expand(ds, cmd.main.trim())? };
        let base = util::touse(ds, st, cmd)?;
        let mut out = vec![String::new()];
        out.push("Variable         |      Mean   Std. dev.       Min        Max |    Observations".into());
        out.push("-----------------+--------------------------------------------+----------------".into());
        let c9 = |v: f64| fmt_g(v, 8);
        let mut first = true;
        let mut last_r: Vec<(&str, f64)> = Vec::new();
        for v in &vars {
            if v == &ivar || ds.get_var(v).is_some_and(|x| x.var_type.is_string()) {
                continue;
            }
            let mut mask = base.clone();
            util::markout(ds, &mut mask, &[ivar.clone(), v.clone()]);
            let x = util::col(ds, v)?;
            let groups = panels(ds, &ivar, &mask);
            let n: usize = groups.iter().map(|g| g.len()).sum();
            if !first {
                out.push("                 |                                            |".into());
            }
            first = false;
            let name = crate::summarize::abbrev(v, 8);
            if n == 0 {
                out.push(format!("{:<8} overall |         .          .          .          . |     N =        0", name));
                out.push(format!("{:<8} between |                    .          .          . |     n =        0", ""));
                out.push(format!("{:<8} within  |                    .          .          . | T-bar =        .", ""));
                continue;
            }
            let all: Vec<f64> = groups.iter().flatten().map(|&i| x[i]).collect();
            let nf = n as f64;
            let mean = all.iter().sum::<f64>() / nf;
            let sd = (all.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
            let mn = all.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let gm: Vec<f64> = groups.iter().map(|g| g.iter().map(|&i| x[i]).sum::<f64>() / g.len() as f64).collect();
            let ng = gm.len() as f64;
            let bm = gm.iter().sum::<f64>() / ng;
            let sdb = (gm.iter().map(|a| (a - bm).powi(2)).sum::<f64>() / (ng - 1.0)).sqrt();
            let wv: Vec<f64> = groups.iter().zip(&gm).flat_map(|(g, m)| g.iter().map(move |&i| (i, *m))).map(|(i, m)| x[i] - m + mean).collect();
            let sdw = (wv.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
            let (bmin, bmax) = (gm.iter().cloned().fold(f64::INFINITY, f64::min), gm.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            let (wmin, wmax) = (wv.iter().cloned().fold(f64::INFINITY, f64::min), wv.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            let tbar = nf / ng;
            let sdbs = if ng > 1.0 { c9(sdb) } else { ".".into() };
            out.push(format!("{:<8} overall | {:>9}  {:>9}  {:>9}  {:>9} |     N = {:>7}", name, c9(mean), c9(sd), c9(mn), c9(mx), n));
            out.push(format!("{:<8} between | {:>9}  {:>9}  {:>9}  {:>9} |     n = {:>7}", "", "", sdbs, c9(bmin), c9(bmax), groups.len()));
            out.push(format!("{:<8} within  | {:>9}  {:>9}  {:>9}  {:>9} | T-bar = {:>7}", "", "", c9(sdw), c9(wmin), c9(wmax), fmt_g(tbar, 7)));
            last_r = vec![
                ("mean", mean),
                ("sd", sd),
                ("min", mn),
                ("max", mx),
                ("N", nf),
                ("sd_b", sdb),
                ("min_b", bmin),
                ("max_b", bmax),
                ("n", ng),
                ("sd_w", sdw),
                ("min_w", wmin),
                ("max_w", wmax),
                ("Tbar", tbar),
            ];
        }
        let r = st.begin_r();
        for (k, v) in last_r {
            r.set(k, v);
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// xttab / xttrans
// ---------------------------------------------------------------------------

pub struct XtTab;

impl CommandHandler for XtTab {
    fn name(&self) -> &'static str {
        "xttab"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
        let v = crate::varlist::resolve_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[ivar.clone(), v.clone()]);
        let x = util::col(ds, &v)?;
        let groups = panels(ds, &ivar, &mask);
        let mut overall: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
        let mut between: BTreeMap<i64, (usize, f64)> = BTreeMap::new();
        let kf = |v: f64| (v * 1e6).round() as i64;
        for g in &groups {
            let mut cnt: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
            for &i in g {
                cnt.entry(kf(x[i])).or_insert((x[i], 0)).1 += 1;
            }
            for (k, (val, c)) in cnt {
                overall.entry(k).or_insert((val, 0)).1 += c;
                let e = between.entry(k).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += c as f64 / g.len() as f64;
            }
        }
        let n: usize = groups.iter().map(|g| g.len()).sum();
        let ng = groups.len();
        let nf = n as f64;
        let w = v.len().max(9);
        let mut out = vec![String::new()];
        out.push(format!("{:w$}   {:>17}{:>20}{:>18}", "", "Overall", "Between", "Within"));
        out.push(format!("{:>w$} |    Freq.  Percent      Freq.  Percent        Percent", crate::summarize::abbrev(&v, w)));
        out.push(format!("{}+{}", "-".repeat(w + 1), "-".repeat(53)));
        let (mut bt, mut wsum) = (0usize, 0.0);
        for (k, (val, c)) in &overall {
            let (bf, ws) = between[k];
            bt += bf;
            wsum += ws;
            out.push(format!("{:>w$} | {:>8} {:>8.2} {:>10} {:>8.2} {:>14.2}", crate::summarize::abbrev(&value_text(ds, &v, *val), w), util::commas(*c), 100.0 * *c as f64 / nf, util::commas(bf), 100.0 * bf as f64 / ng as f64, 100.0 * ws / bf as f64));
        }
        out.push(format!("{}+{}", "-".repeat(w + 1), "-".repeat(53)));
        out.push(format!("{:>w$} | {:>8} {:>8.2} {:>10} {:>8.2} {:>14.2}", "Total", util::commas(n), 100.0, util::commas(bt), 100.0 * bt as f64 / ng as f64, 100.0 * wsum / bt as f64));
        out.push(format!("{:w$}   {:>39}", "", format!("(n = {})", util::commas(ng))));
        let r = st.begin_r();
        r.set("n", ng as f64);
        r.set("N", nf);
        Ok(out.join("\n"))
    }
}

pub struct XtTrans;

impl CommandHandler for XtTrans {
    fn name(&self) -> &'static str {
        "xttrans"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (ivar, tvar) = panel_time(ds, st, cmd, true)?;
        let tvar = tvar.unwrap();
        let v = crate::varlist::resolve_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[ivar.clone(), tvar.clone(), v.clone()]);
        let x = util::col(ds, &v)?;
        let tc = util::col(ds, &tvar)?;
        let groups = panels(ds, &ivar, &mask);
        let freq = cmd.has_opt("freq", 1);
        let mut vals: Vec<f64> = Vec::new();
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for g in &groups {
            let mut g = g.clone();
            g.sort_by(|a, b| tc[*a].total_cmp(&tc[*b]));
            for w in g.windows(2) {
                pairs.push((x[w[0]], x[w[1]]));
            }
        }
        for (a, b) in &pairs {
            vals.push(*a);
            vals.push(*b);
        }
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        let m = vals.len();
        let pos = |v: f64| vals.iter().position(|x| *x == v).unwrap();
        let mut tab = vec![vec![0usize; m]; m];
        for (a, b) in &pairs {
            tab[pos(*a)][pos(*b)] += 1;
        }
        let lw = 10usize;
        let labs: Vec<String> = vals.iter().map(|x| crate::summarize::abbrev(&value_text(ds, &v, *x), 10)).collect();
        let mut out = vec![String::new()];
        out.push(format!("{:>lw$} | {}", crate::summarize::abbrev(&v, lw), crate::summarize::abbrev(&v, lw)));
        out.push(format!("{:>lw$} | {} |     Total", crate::summarize::abbrev(&v, lw), labs.iter().map(|l| format!("{:>10}", l)).collect::<Vec<_>>().join(" ")));
        let rule = format!("{}+{}+{}", "-".repeat(lw + 1), "-".repeat(11 * m + 1), "-".repeat(10));
        out.push(rule.clone());
        let mut colt = vec![0usize; m];
        let push_rows = |out: &mut Vec<String>, label: &str, row: &[usize]| {
            let tot: usize = row.iter().sum();
            if freq {
                out.push(format!("{:>lw$} | {} | {:>9} ", label, row.iter().map(|c| format!("{:>10}", util::commas(*c))).collect::<Vec<_>>().join(" "), util::commas(tot)));
                out.push(format!("{:>lw$} | {} | {:>9.2} ", "", row.iter().map(|c| format!("{:>10.2}", 100.0 * *c as f64 / tot.max(1) as f64)).collect::<Vec<_>>().join(" "), 100.0));
            } else {
                out.push(format!("{:>lw$} | {} | {:>9.2} ", label, row.iter().map(|c| format!("{:>10.2}", 100.0 * *c as f64 / tot.max(1) as f64)).collect::<Vec<_>>().join(" "), 100.0));
            }
        };
        for i in 0..m {
            for j in 0..m {
                colt[j] += tab[i][j];
            }
            push_rows(&mut out, &labs[i], &tab[i]);
            if freq {
                out.push(rule.clone());
            }
        }
        if !freq {
            out.push(rule.clone());
        }
        push_rows(&mut out, "Total", &colt);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// xtdata
// ---------------------------------------------------------------------------

pub struct XtData;

impl CommandHandler for XtData {
    fn name(&self) -> &'static str {
        "xtdata"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !cmd.has_opt("clear", 5) && st.data_changed {
            return Err(rc_err(4, "no; data in memory would be lost"));
        }
        let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
        let vars: Vec<String> = if cmd.main.trim().is_empty() {
            ds.varlist().into_iter().filter(|v| v != &ivar && !v.starts_with('_') && ds.get_var(v).is_some_and(|x| !x.var_type.is_string())).collect()
        } else {
            crate::varlist::expand(ds, cmd.main.trim())?
        };
        let model = if cmd.has_opt("fe", 2) {
            "fe"
        } else if cmd.has_opt("be", 2) {
            "be"
        } else if cmd.has_opt("re", 2) {
            "re"
        } else {
            return Err(rc_err(198, "one of re, be, or fe must be specified"));
        };
        let mut mask = util::touse(ds, st, cmd)?;
        let mut mv = vars.clone();
        mv.push(ivar.clone());
        util::markout(ds, &mut mask, &mv);
        let groups = panels(ds, &ivar, &mask);
        let theta_opt = util::opt_num(cmd, "theta", 2)?;
        let mut theta = vec![0.0; groups.len()];
        if model == "re" {
            match theta_opt {
                Some(t) => theta.iter_mut().for_each(|x| *x = t),
                None => {
                    let mut st2 = st.clone();
                    let mut ds2 = ds.clone();
                    let line = format!("xtreg {}, re i({ivar})", vars.join(" "));
                    let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
                    crate::models::Xtreg.execute(&mut ds2, &mut st2, &c)?;
                    let su = st2.e.get("sigma_u").unwrap_or(0.0);
                    let se = st2.e.get("sigma_e").unwrap_or(1.0);
                    for (g, t) in groups.iter().zip(theta.iter_mut()) {
                        *t = 1.0 - (se * se / (g.len() as f64 * su * su + se * se)).sqrt();
                    }
                }
            }
        }
        let n = ds.n_obs;
        for v in &vars {
            let x = util::col(ds, v)?;
            let gm: Vec<f64> = groups.iter().map(|g| g.iter().map(|&i| x[i]).sum::<f64>() / g.len() as f64).collect();
            let nn: usize = groups.iter().map(|g| g.len()).sum();
            let mean = groups.iter().flatten().map(|&i| x[i]).sum::<f64>() / nn as f64;
            let mut nv = vec![SYSMISS; n];
            for (gi, g) in groups.iter().enumerate() {
                for &i in g {
                    nv[i] = match model {
                        "fe" => x[i] - gm[gi] + mean,
                        "be" => gm[gi],
                        _ => x[i] - theta[gi] * gm[gi],
                    };
                }
            }
            let var = ds.get_var_mut(v).unwrap();
            var.var_type = ferrostat_core::VarType::Double;
            var.data = nv.iter().map(|&a| if is_missing(a) { Value::Missing } else { Value::Double(a) }).collect();
        }
        let mut keep = mask.clone();
        if model == "be" {
            keep = vec![false; n];
            for g in &groups {
                keep[g[0]] = true;
            }
        }
        if model == "re" {
            let mut c = vec![SYSMISS; n];
            for (gi, g) in groups.iter().enumerate() {
                for &i in g {
                    c[i] = 1.0 - theta[gi];
                }
            }
            if ds.get_var("constant").is_some() {
                ds.drop_variable("constant")?;
            }
            ds.add_variable(util::new_double("constant", c))?;
        }
        ds.keep_if(&keep)?;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// quadchk
// ---------------------------------------------------------------------------

/// Refit the last quadrature-based panel model with more and fewer
/// integration points and compare the estimates.
pub struct QuadChk;

impl CommandHandler for QuadChk {
    fn name(&self) -> &'static str {
        "quadchk"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let nq = st.e.get("n_quad").ok_or_else(|| rc_err(301, "last estimates not found or not a quadrature-based model"))?;
        let ecmd = st.e.macros.get("cmdline").cloned().or_else(|| st.e.macros.get("_cmdline").cloned());
        let line = ecmd.ok_or_else(|| rc_err(301, "command line of the last estimates not available"))?;
        let words: Vec<f64> = cmd.main.split_whitespace().filter_map(|w| w.parse().ok()).collect();
        let n1 = words.first().copied().unwrap_or(nq - 4.0).max(1.0);
        let n2 = words.get(1).copied().unwrap_or(nq + 4.0).max(1.0);
        let saved = st.e.clone();
        let base_b = saved.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
        let base_ll = saved.get("ll").unwrap_or(SYSMISS);
        let strip = |l: &str| -> String {
            let mut s = l.to_string();
            if let Some(p) = s.find("intpoints(") {
                if let Some(q) = s[p..].find(')') {
                    s.replace_range(p..p + q + 1, "");
                }
            }
            s
        };
        let base_line = strip(&line);
        let mut fits = Vec::new();
        for q in [n1, n2] {
            let l = if base_line.contains(',') { format!("{base_line} intpoints({})", q as usize) } else { format!("{base_line}, intpoints({})", q as usize) };
            let c = ferrostat_parser::parser::parse_line(&l).map_err(|e| rc_err(198, e.message))?;
            let verb = c.verb.clone();
            let h = crate::xtnl::handlers().into_iter().chain(crate::xt_re::handlers()).find(|(n, _)| *n == verb).map(|x| x.1).ok_or_else(|| rc_err(198, format!("quadchk not available after {verb}")))?;
            h.execute(ds, st, &c)?;
            fits.push((q, st.e.get("ll").unwrap_or(SYSMISS), st.e.matrices.get("b").cloned().unwrap()));
        }
        st.e = saved;
        let mut out = vec![String::new(), "Refitting model intpoints() = ".to_string() + &fmt_g(n1, 9), "Refitting model intpoints() = ".to_string() + &fmt_g(n2, 9), String::new()];
        out.push("               Quadrature check".into());
        out.push(String::new());
        out.push(format!("{:>14} {:>13} {:>13} {:>13}", "", "Fitted", "Comparison", "Comparison"));
        out.push(format!("{:>14} {:>13} {:>13} {:>13}", "", "quadrature", "quadrature", "quadrature"));
        out.push(format!("{:>14} {:>13} {:>13} {:>13}", "", format!("{} points", nq), format!("{} points", n1), format!("{} points", n2)));
        out.push("-".repeat(56));
        let row = |lab: &str, a: f64, b: f64, c: f64| {
            vec![
                format!("{:>14} {:>13} {:>13} {:>13}", lab, fmt_g(a, 9), fmt_g(b, 9), fmt_g(c, 9)),
                format!("{:>14} {:>13} {:>13} {:>13}", "Difference", "", fmt_g(b - a, 9), fmt_g(c - a, 9)),
                format!("{:>14} {:>13} {:>13} {:>13}", "Rel. diff.", "", fmt_g((b - a) / a, 9), fmt_g((c - a) / a, 9)),
            ]
        };
        out.extend(row("Log likelihood", base_ll, fits[0].1, fits[1].1));
        out.push("-".repeat(56));
        for (j, name) in base_b.colnames.iter().enumerate() {
            let a = base_b.data[j];
            let b = fits[0].2.data.get(j).copied().unwrap_or(SYSMISS);
            let c = fits[1].2.data.get(j).copied().unwrap_or(SYSMISS);
            out.extend(row(&crate::summarize::abbrev(name.rsplit(':').next().unwrap_or(name), 14), a, b, c));
            out.push("-".repeat(56));
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// xtline
// ---------------------------------------------------------------------------

/// `xtline varlist [if] [in] [, overlay i() t() twoway_options]`: line plots
/// of panel series against time, one graph per panel or overlaid.
pub struct XtLine;

impl CommandHandler for XtLine {
    fn name(&self) -> &'static str {
        "xtline"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (ivar, tvar) = panel_time(ds, st, cmd, true)?;
        let tvar = tvar.unwrap();
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        let cond = cmd.if_raw.as_ref().map(|s| format!("({s})")).unwrap_or_default();
        let pass: Vec<String> = cmd.opts.iter().filter(|(n, _)| !matches!(n.as_str(), "overlay" | "i" | "t")).map(|(n, a)| a.as_ref().map(|a| format!("{n}({a})")).unwrap_or(n.clone())).collect();
        let line = if cmd.has_opt("overlay", 2) {
            if vars.len() != 1 {
                return Err(rc_err(103, "overlay allows only one variable"));
            }
            let mask = util::touse(ds, st, cmd)?;
            let groups = panels(ds, &ivar, &mask);
            let pv = util::col(ds, &ivar)?;
            let layers: Vec<String> = groups
                .iter()
                .map(|g| {
                    let v = fmt_g(pv[g[0]], 17);
                    let c = if cond.is_empty() { format!("{ivar}=={v}") } else { format!("{ivar}=={v} & {cond}") };
                    format!("(line {} {tvar} if {c}, sort)", vars[0])
                })
                .collect();
            format!("twoway {} , {}", layers.join(" "), pass.join(" "))
        } else {
            let c = if cond.is_empty() { String::new() } else { format!(" if {cond}") };
            format!("twoway line {} {tvar}{c}, sort by({ivar}) {}", vars.join(" "), pass.join(" "))
        };
        let c = ferrostat_parser::parser::parse_line(&line).map_err(|e| rc_err(198, e.message))?;
        crate::graph::Twoway.execute(ds, st, &c)
    }
}
