//! Choice-data utilities: `cmset`, `cmtab`, `cmsummarize`, `cmchoiceset`
//! and `cmsample`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, ValueLabel, VarType, Value, Variable};
use ferrostat_parser::Command;

use super::data::{self, CmSet};
use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::store::{Results, Store};
use crate::util;

pub struct Cmset;
pub struct Cmtab;
pub struct Cmsummarize;
pub struct Cmchoiceset;
pub struct Cmsample;

fn show_set(s: &CmSet) -> String {
    let mut out = Vec::new();
    if let Some(p) = &s.panel {
        out.push(format!("{:>25}: {p}", "Panel data: Panels"));
        out.push(format!("{:>25}: {}", "Time variable", s.time.clone().unwrap_or_default()));
    }
    out.push(format!("{:>25}: {}", "Case ID variable", s.case));
    out.push(format!("{:>25}: {}", "Alternatives variable", s.alt.clone().unwrap_or_else(|| "<none>".into())));
    out.join("\n")
}

fn set_r(st: &mut Store, s: &CmSet) {
    let mut r = Results::new();
    r.set_str("caseid", s.case.clone());
    if let Some(a) = &s.alt {
        r.set_str("altvar", a.clone());
    }
    if let Some(p) = &s.panel {
        r.set_str("panelvar", p.clone());
        r.set_str("timevar", s.time.clone().unwrap_or_default());
    }
    st.r = r;
}

impl CommandHandler for Cmset {
    fn name(&self) -> &'static str {
        "cmset"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.has_opt("clear", 5) {
            data::clear_set(ds);
            return Ok(String::new());
        }
        let words: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        if words.is_empty() {
            let s = data::get_set(ds).ok_or_else(|| rc_err(459, "data not cmset"))?;
            set_r(st, &s);
            return Ok(show_set(&s));
        }
        let noalt = cmd.has_opt("noalternatives", 5);
        let v: Vec<String> = words.iter().map(|w| crate::varlist::resolve_name(ds, w)).collect::<Result<_, _>>()?;
        let s = match (v.len(), noalt) {
            (1, true) => CmSet { case: v[0].clone(), alt: None, panel: None, time: None },
            (2, false) => CmSet { case: v[0].clone(), alt: Some(v[1].clone()), panel: None, time: None },
            (2, true) => {
                let cid = make_caseid(ds, &v[0], &v[1])?;
                CmSet { case: cid, alt: None, panel: Some(v[0].clone()), time: Some(v[1].clone()) }
            }
            (3, false) => {
                let cid = make_caseid(ds, &v[0], &v[1])?;
                CmSet { case: cid, alt: Some(v[2].clone()), panel: Some(v[0].clone()), time: Some(v[1].clone()) }
            }
            (1, false) => return Err(rc_err(198, "alternatives variable required; specify option noalternatives if there is none")),
            _ => return Err(rc_err(103, "too many variables specified")),
        };
        for name in [Some(&s.case), s.alt.as_ref()].into_iter().flatten() {
            let var = ds.get_var(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
            if var.var_type.is_string() {
                return Err(rc_err(109, format!("variable {name} must be numeric")));
            }
        }
        // alternatives must be unique within case
        if let Some(a) = &s.alt {
            let cc = util::col(ds, &s.case)?;
            let ac = util::col(ds, a)?;
            let mut seen = std::collections::HashSet::new();
            for i in 0..ds.n_obs {
                if is_missing(cc[i]) || is_missing(ac[i]) {
                    continue;
                }
                if !seen.insert((cc[i].to_bits(), ac[i].to_bits())) && !cmd.has_opt("force", 5) {
                    return Err(rc_err(459, format!("variable {a} has replicate levels for one or more cases")));
                }
            }
        }
        data::put_set(ds, &s);
        set_r(st, &s);
        Ok(show_set(&s))
    }
}

/// Case identifier `_caseid` for panel-form choice data: one value per
/// (panel, time) pair.
fn make_caseid(ds: &mut DataSet, p: &str, t: &str) -> Result<String, String> {
    let pc = util::col(ds, p)?;
    let tc = util::col(ds, t)?;
    let mut keys: Vec<(f64, f64)> = (0..ds.n_obs).filter(|&i| !is_missing(pc[i]) && !is_missing(tc[i])).map(|i| (pc[i], tc[i])).collect();
    keys.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    keys.dedup();
    let vals: Vec<f64> = (0..ds.n_obs)
        .map(|i| if is_missing(pc[i]) || is_missing(tc[i]) { SYSMISS } else { (keys.binary_search_by(|k| k.0.total_cmp(&pc[i]).then(k.1.total_cmp(&tc[i]))).unwrap() + 1) as f64 })
        .collect();
    let name = "_caseid".to_string();
    if ds.var_index.contains_key(&name) {
        ds.drop_variable(&name)?;
    }
    let mut v = util::new_double(&name, vals);
    v.var_type = VarType::Long;
    v.data = v.data.iter().map(|x| if let Value::Double(d) = x { if is_missing(*d) { Value::Missing } else { Value::Long(*d as i32) } } else { Value::Missing }).collect();
    ds.add_variable(v)?;
    Ok(name)
}

/// Sample rows (if/in, non-missing case and alternatives) grouped by case.
struct Cases {
    set: CmSet,
    groups: Vec<Vec<usize>>,
    alt: Vec<f64>,
}

fn cases(ds: &DataSet, st: &Store, cmd: &Command, extra: &[String]) -> Result<Cases, String> {
    let set = data::get_set(ds).ok_or_else(|| rc_err(459, "data not cmset; use cmset to declare the case and alternatives variables"))?;
    let mut mask = util::touse(ds, st, cmd)?;
    let mut vars = vec![set.case.clone()];
    if let Some(a) = &set.alt {
        vars.push(a.clone());
    }
    vars.extend(extra.iter().cloned());
    util::markout(ds, &mut mask, &vars);
    let rows = util::rows(&mask);
    let (_, groups) = crate::xtnl::panel_groups(ds, &set.case, &rows)?;
    let groups: Vec<Vec<usize>> = groups.into_iter().map(|g| g.into_iter().map(|t| rows[t]).collect()).collect();
    let alt = match &set.alt {
        Some(a) => util::col(ds, a)?,
        None => vec![SYSMISS; ds.n_obs],
    };
    Ok(Cases { set, groups, alt })
}

fn alt_text(ds: &DataSet, set: &CmSet, row: usize) -> String {
    match set.alt.as_ref().and_then(|a| ds.get_var(a)) {
        Some(v) => util::cell_text(ds, v, row, true),
        None => String::new(),
    }
}

fn choice_var(ds: &DataSet, cmd: &Command) -> Result<String, String> {
    let c = cmd.opt("choice", 3).ok_or_else(|| rc_err(198, "option choice() required"))?;
    crate::varlist::resolve_name(ds, c.trim())
}

impl CommandHandler for Cmtab {
    fn name(&self) -> &'static str {
        "cmtab"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ch = choice_var(ds, cmd)?;
        let other: Option<String> = match cmd.main.trim() {
            "" => None,
            w => Some(crate::varlist::resolve_name(ds, w)?),
        };
        let mut extra = vec![ch.clone()];
        if let Some(o) = &other {
            extra.push(o.clone());
        }
        let cs = cases(ds, st, cmd, &extra)?;
        if cs.set.alt.is_none() {
            return Err(rc_err(459, "cmtab requires an alternatives variable"));
        }
        let cv = util::col(ds, &ch)?;
        // chosen alternative per case
        let mut freq: BTreeMap<u64, (f64, usize, String)> = BTreeMap::new();
        let mut cross: BTreeMap<(u64, String), usize> = BTreeMap::new();
        let mut col_keys: Vec<String> = Vec::new();
        let mut nchosen = 0usize;
        let ov = match &other {
            Some(o) => Some(ds.get_var(o).unwrap().clone()),
            None => None,
        };
        let key = |a: f64| (a + 1e15).to_bits();
        for g in &cs.groups {
            for &r in g {
                if cv[r] != 0.0 {
                    let a = cs.alt[r];
                    let e = freq.entry(key(a)).or_insert((a, 0, alt_text(ds, &cs.set, r)));
                    e.1 += 1;
                    nchosen += 1;
                    if let Some(v) = &ov {
                        let t = util::cell_text(ds, v, r, true);
                        if !col_keys.contains(&t) {
                            col_keys.push(t.clone());
                        }
                        *cross.entry((key(a), t)).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut out = vec![String::new()];
        match &other {
            None => {
                out.push(format!("Tabulation of chosen alternatives ({ch} = 1)"));
                out.push(String::new());
                let a = cs.set.alt.clone().unwrap();
                out.push(format!("{:>16} |      Freq.     Percent        Cum.", crate::summarize::abbrev(&a, 16)));
                out.push(format!("{}-+-----------------------------------", "-".repeat(16)));
                let mut cum = 0.0;
                for (_, (_, f, lab)) in &freq {
                    let p = 100.0 * *f as f64 / nchosen.max(1) as f64;
                    cum += p;
                    out.push(format!("{:>16} | {:>10} {:>11.2} {:>11.2}", crate::summarize::abbrev(lab, 16), util::commas(*f), p, cum));
                }
                out.push(format!("{}-+-----------------------------------", "-".repeat(16)));
                out.push(format!("{:>16} | {:>10} {:>11.2}", "Total", util::commas(nchosen), 100.0));
            }
            Some(o) => {
                col_keys.sort();
                out.push(format!("Tabulation for chosen alternatives ({ch} = 1) by {o}"));
                out.push(String::new());
                out.push(format!("{:>16} |{} | {:>8}", "", col_keys.iter().map(|k| format!(" {:>8}", crate::summarize::abbrev(k, 8))).collect::<String>(), "Total"));
                out.push(format!("{}-+{}-+---------", "-".repeat(16), "-".repeat(9 * col_keys.len())));
                for (ak, (_, f, lab)) in &freq {
                    let cells: String = col_keys.iter().map(|k| format!(" {:>8}", util::commas(*cross.get(&(*ak, k.clone())).unwrap_or(&0)))).collect();
                    out.push(format!("{:>16} |{cells} | {:>8}", crate::summarize::abbrev(lab, 16), util::commas(*f)));
                }
            }
        }
        let mut r = Results::new();
        r.set("N", nchosen as f64);
        r.set("r", freq.len() as f64);
        st.r = r;
        Ok(out.join("\n"))
    }
}

impl CommandHandler for Cmsummarize {
    fn name(&self) -> &'static str {
        "cmsummarize"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ch = choice_var(ds, cmd)?;
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mut extra = vec![ch.clone()];
        extra.extend(vars.iter().cloned());
        let cs = cases(ds, st, cmd, &extra)?;
        if cs.set.alt.is_none() {
            return Err(rc_err(459, "cmsummarize requires an alternatives variable"));
        }
        let cv = util::col(ds, &ch)?;
        let stats: Vec<String> = match cmd.opt("statistics", 4) {
            Some(s) => s.split_whitespace().map(|w| w.to_lowercase()).collect(),
            None => ["mean", "sd", "min", "p25", "p50", "p75", "max"].iter().map(|s| s.to_string()).collect(),
        };
        // rows of cases grouped by the chosen alternative
        let mut by: BTreeMap<u64, (String, Vec<usize>)> = BTreeMap::new();
        for g in &cs.groups {
            if let Some(&r) = g.iter().find(|&&r| cv[r] != 0.0) {
                let e = by.entry((cs.alt[r] + 1e15).to_bits()).or_insert((alt_text(ds, &cs.set, r), Vec::new()));
                e.1.extend(g.iter().copied());
            }
        }
        let mut out = vec![String::new(), format!("Statistics by chosen alternatives ({ch} = 1)"), String::new()];
        let mut r = Results::new();
        for v in &vars {
            let x = util::col(ds, v)?;
            out.push(format!("Summary for variable: {v}"));
            out.push(format!("{:>16} |{}", "chosen alt.", stats.iter().map(|s| format!(" {:>9}", s)).collect::<String>()));
            out.push(format!("{}-+{}", "-".repeat(16), "-".repeat(10 * stats.len())));
            let mut mat = Vec::new();
            let mut all: Vec<f64> = Vec::new();
            let mut rows_out = Vec::new();
            for (_, (lab, rows)) in &by {
                let vals: Vec<f64> = rows.iter().map(|&i| x[i]).filter(|v| !is_missing(*v)).collect();
                all.extend(vals.iter().copied());
                let s: Vec<f64> = stats.iter().map(|s| stat(&vals, s)).collect();
                rows_out.push(format!("{:>16} |{}", crate::summarize::abbrev(lab, 16), s.iter().map(|v| format!(" {:>9}", fmt_g(*v, 9))).collect::<String>()));
                mat.push(s);
            }
            out.extend(rows_out);
            out.push(format!("{}-+{}", "-".repeat(16), "-".repeat(10 * stats.len())));
            out.push(format!("{:>16} |{}", "Total", stats.iter().map(|s| format!(" {:>9}", fmt_g(stat(&all, s), 9))).collect::<String>()));
            out.push(String::new());
            let mut m = crate::matrix::Matrix::zeros(mat.len(), stats.len());
            for (i, row) in mat.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    m[(i, j)] = *v;
                }
            }
            r.set_mat(&format!("stats_{v}"), m.with_names(by.values().map(|(l, _)| l.clone()).collect(), stats.clone()));
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn stat(v: &[f64], s: &str) -> f64 {
    if v.is_empty() {
        return if s == "n" || s == "count" { 0.0 } else { SYSMISS };
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(f64::total_cmp);
    match s {
        "n" | "count" => v.len() as f64,
        "mean" => util::mean(v),
        "sd" => {
            if v.len() < 2 {
                SYSMISS
            } else {
                util::variance(v).sqrt()
            }
        }
        "min" => sorted[0],
        "max" => *sorted.last().unwrap(),
        "sum" => v.iter().sum(),
        "median" | "p50" => util::pctile_sorted(&sorted, 50.0),
        _ if s.starts_with('p') => s[1..].parse::<f64>().map(|p| util::pctile_sorted(&sorted, p)).unwrap_or(SYSMISS),
        _ => SYSMISS,
    }
}

impl CommandHandler for Cmchoiceset {
    fn name(&self) -> &'static str {
        "cmchoiceset"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let by: Option<String> = match cmd.main.trim() {
            "" => None,
            w => Some(crate::varlist::resolve_name(ds, w)?),
        };
        let extra: Vec<String> = by.iter().cloned().collect();
        let cs = cases(ds, st, cmd, &extra)?;
        if cs.set.alt.is_none() {
            return Err(rc_err(459, "cmchoiceset requires an alternatives variable"));
        }
        let size = cmd.has_opt("size", 2);
        let mut sets: BTreeMap<String, usize> = BTreeMap::new();
        let mut row_set = vec![String::new(); ds.n_obs];
        let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
        for g in &cs.groups {
            let mut a: Vec<f64> = g.iter().map(|&r| cs.alt[r]).collect();
            a.sort_by(f64::total_cmp);
            let key = a.iter().map(|v| fmt_g(*v, 9)).collect::<Vec<_>>().join(" ");
            *sets.entry(key.clone()).or_insert(0) += 1;
            *sizes.entry(g.len()).or_insert(0) += 1;
            for &r in g {
                row_set[r] = key.clone();
            }
        }
        let n = cs.groups.len();
        let mut out = vec![String::new()];
        let title = if size { "Size of choice set" } else { "Choice set" };
        out.push(format!("{:>24} |      Freq.     Percent        Cum.", title));
        out.push(format!("{}-+-----------------------------------", "-".repeat(24)));
        let mut cum = 0.0;
        let rows: Vec<(String, usize)> = if size { sizes.iter().map(|(k, v)| (k.to_string(), *v)).collect() } else { sets.iter().map(|(k, v)| (k.clone(), *v)).collect() };
        for (k, f) in &rows {
            let p = 100.0 * *f as f64 / n.max(1) as f64;
            cum += p;
            out.push(format!("{:>24} | {:>10} {:>11.2} {:>11.2}", crate::summarize::abbrev(k, 24), util::commas(*f), p, cum));
        }
        out.push(format!("{}-+-----------------------------------", "-".repeat(24)));
        out.push(format!("{:>24} | {:>10} {:>11.2}", "Total", util::commas(n), 100.0));
        if let Some(gv) = cmd.opt("generate", 3) {
            let name = gv.split(',').next().unwrap_or("").trim().to_string();
            crate::varlist::check_name(&name)?;
            if ds.var_index.contains_key(&name) {
                if gv.contains("replace") {
                    ds.drop_variable(&name)?;
                } else {
                    return Err(rc_err(110, format!("variable {name} already defined")));
                }
            }
            let len = row_set.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let mut v = Variable::new(name, VarType::Str(len));
            v.data = row_set.into_iter().map(Value::Str).collect();
            ds.add_variable(v)?;
        }
        let mut r = Results::new();
        r.set("N", n as f64);
        r.set("r", rows.len() as f64);
        st.r = r;
        Ok(out.join("\n"))
    }
}

const REASONS: [&str; 9] = [
    "observations included",
    "case ID variable missing",
    "alternatives variable missing",
    "choice variable missing",
    "varlist missing",
    "casevars missing",
    "case has no chosen alternative",
    "case has more than one chosen alternative",
    "other alternatives in case excluded",
];

impl CommandHandler for Cmsample {
    fn name(&self) -> &'static str {
        "cmsample"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let set = data::get_set(ds).ok_or_else(|| rc_err(459, "data not cmset; use cmset to declare the case and alternatives variables"))?;
        let vars = if cmd.main.trim().is_empty() { vec![] } else { crate::varlist::expand(ds, cmd.main.trim())? };
        let casevars = data::opt_vars(ds, cmd, "casevars", 5)?;
        let ch = match cmd.opt("choice", 3) {
            Some(c) => Some(crate::varlist::resolve_name(ds, c.trim())?),
            None => None,
        };
        let ranks = cmd.has_opt("ranks", 4);
        let altwise = cmd.has_opt("altwise", 4);
        let touse = util::touse(ds, st, cmd)?;
        let n = ds.n_obs;
        let cc = util::col(ds, &set.case)?;
        let ac = match &set.alt {
            Some(a) => util::col(ds, a)?,
            None => vec![0.0; n],
        };
        let chc = match &ch {
            Some(c) => util::col(ds, c)?,
            None => vec![0.0; n],
        };
        let vcols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let ccols: Vec<Vec<f64>> = casevars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mut code = vec![SYSMISS; n];
        for i in 0..n {
            if !touse[i] {
                continue;
            }
            code[i] = if is_missing(cc[i]) {
                1.0
            } else if is_missing(ac[i]) {
                2.0
            } else if ch.is_some() && is_missing(chc[i]) {
                3.0
            } else if vcols.iter().any(|c| is_missing(c[i])) {
                4.0
            } else if ccols.iter().any(|c| is_missing(c[i])) {
                5.0
            } else {
                0.0
            };
        }
        // case-level checks
        let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if touse[i] && !is_missing(cc[i]) {
                groups.entry((cc[i] + 1e15).to_bits()).or_default().push(i);
            }
        }
        for g in groups.values() {
            if !altwise && g.iter().any(|&i| code[i] != 0.0) {
                for &i in g {
                    if code[i] == 0.0 {
                        code[i] = 8.0;
                    }
                }
                continue;
            }
            if ch.is_some() && !ranks {
                let ok: Vec<usize> = g.iter().copied().filter(|&i| code[i] == 0.0).collect();
                let pos = ok.iter().filter(|&&i| chc[i] != 0.0).count();
                let c = if pos == 0 { 6.0 } else if pos > 1 { 7.0 } else { 0.0 };
                if c != 0.0 {
                    for i in ok {
                        code[i] = c;
                    }
                }
            }
        }
        let mut cnt = [0usize; 9];
        for i in 0..n {
            if !is_missing(code[i]) {
                cnt[code[i] as usize] += 1;
            }
        }
        let tot: usize = cnt.iter().sum();
        let mut out = vec![String::new()];
        out.push(format!("{:>45} |      Freq.     Percent", "Reason for exclusion"));
        out.push(format!("{}-+------------------------", "-".repeat(45)));
        for (k, c) in cnt.iter().enumerate() {
            if *c > 0 || k == 0 {
                out.push(format!("{:>45} | {:>10} {:>11.2}", REASONS[k], util::commas(*c), 100.0 * *c as f64 / tot.max(1) as f64));
            }
        }
        out.push(format!("{}-+------------------------", "-".repeat(45)));
        out.push(format!("{:>45} | {:>10} {:>11.2}", "Total", util::commas(tot), 100.0));
        if let Some(gv) = cmd.opt("generate", 3) {
            let name = gv.split(',').next().unwrap_or("").trim().to_string();
            crate::varlist::check_name(&name)?;
            if ds.var_index.contains_key(&name) {
                if gv.contains("replace") {
                    ds.drop_variable(&name)?;
                } else {
                    return Err(rc_err(110, format!("variable {name} already defined")));
                }
            }
            let mut v = util::new_double(&name, code.clone());
            v.var_type = VarType::Byte;
            v.data = code.iter().map(|c| if is_missing(*c) { Value::Missing } else { Value::Byte(*c as i8) }).collect();
            let mut vl = ValueLabel::new(name.clone());
            for (k, l) in REASONS.iter().enumerate() {
                vl.add(k as f64, *l);
            }
            ds.value_labels.insert(name.clone(), vl);
            v.value_label_name = name.clone();
            ds.add_variable(v)?;
        }
        let mut r = Results::new();
        r.set("N", tot as f64);
        r.set("N_ex", (tot - cnt[0]) as f64);
        r.set("N_used", cnt[0] as f64);
        st.r = r;
        Ok(out.join("\n"))
    }
}
