//! User-defined methods for `power`, `ciwidth` and `gsdesign`.
//!
//! `power <name> ...` runs the r-class program `power_cmd_<name>` and
//! `ciwidth <name> ...` runs `ciwidth_cmd_<name>`. The program receives the
//! positional arguments followed by its own options plus `alpha(#)`, and
//! either `n(#)` (it must return `r(power)`, or `r(width)` for ciwidth) or,
//! when the sample size is to be computed, `power(#)` (or `width(#)`). A
//! program that returns `r(N)` in the latter case solves for the size
//! itself; otherwise the size is found by searching over `n()`.
//! `gsdesign <name>` uses `power_cmd_<name>` for its fixed-sample size.

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::pss::{self, Report, Row, Solve};
use ferrostat_cmd::store::Results;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::interpreter::{AdoInterpreter, Flow};

const POWER_OPTS: &[&str] = &["alpha", "power", "beta", "n", "n1", "n2", "nratio", "onesided", "table", "notable", "graph", "parallel", "nfractional", "compute", "direction"];
const GS_OPTS: &[&str] = &["nlooks", "information", "efficacy", "futility", "binding", "nonbinding"];
const CI_OPTS: &[&str] = &["width", "probwidth", "level"];

fn is_common(name: &str, verb: &str) -> bool {
    let n = name.to_lowercase();
    let full = |list: &[&str]| list.iter().any(|o| *o == n || (n.len() >= 3 && o.starts_with(n.as_str()) && !matches!(n.as_str(), "n1" | "n2")));
    full(POWER_OPTS) || (verb == "gsdesign" && full(GS_OPTS)) || (verb == "ciwidth" && full(CI_OPTS))
}

pub(crate) fn intercept(me: &mut AdoInterpreter, verb: &str, rest: &str, text: &str) -> Result<Option<Flow>, String> {
    let v = verb.to_lowercase();
    if !matches!(v.as_str(), "power" | "ciwidth" | "gsdesign") {
        return Ok(None);
    }
    let r = rest.trim_start();
    let end = r.find(|c: char| c.is_whitespace() || c == ',').unwrap_or(r.len());
    let mut name = r[..end].to_lowercase();
    let mut after = &r[end..];
    let explicit = name == "usermethod";
    if explicit {
        let a = after.trim_start();
        let e = a.find(|c: char| c.is_whitespace() || c == ',').unwrap_or(a.len());
        name = a[..e].to_lowercase();
        after = &a[e..];
    }
    if name.is_empty() {
        return Ok(None);
    }
    let builtin = match v.as_str() {
        "ciwidth" => ["onemean", "pairedmeans", "twomeans", "onevariance"].iter().any(|b| b.starts_with(name.as_str()) && name.len() >= 4),
        _ => pss::is_builtin(&name),
    };
    if builtin && !explicit {
        return Ok(None);
    }
    let prog = format!("{}_cmd_{name}", if v == "ciwidth" { "ciwidth" } else { "power" });
    if !me.engine.store.programs.contains_key(&prog) {
        if explicit {
            return Err(rc_err(199, format!("program {prog} not found")));
        }
        return Ok(None);
    }
    let _ = text;
    let cmd = parse_line(&format!("{v} {name} {after}")).map_err(|e| rc_err(198, e.message))?;
    let args = cmd.main.trim()[name.len()..].trim().to_string();
    // options passed through to the program
    let mut pass = Vec::new();
    for (n, a) in &cmd.opts {
        if is_common(n, &v) {
            continue;
        }
        pass.push(match a {
            Some(a) => format!("{n}({a})"),
            None => n.clone(),
        });
    }
    let pass = pass.join(" ");
    let out = match v.as_str() {
        "power" => user_power(me, &cmd, &prog, &name, &args, &pass)?,
        "ciwidth" => user_ciwidth(me, &cmd, &prog, &name, &args, &pass)?,
        _ => user_gs(me, &cmd, &prog, &name, &args, &pass)?,
    };
    me.emit(&out);
    Ok(Some(Flow::Next))
}

/// Run the program once and return its r() results.
fn call(me: &mut AdoInterpreter, prog: &str, args: &str, opts: &str) -> Result<Results, String> {
    me.engine.store.r = Results::new();
    me.run_program(prog, &format!("{args}, {opts}"))?;
    Ok(me.engine.store.r.clone())
}

fn fmt(v: f64) -> String {
    format!("{v:.12}")
}

fn base_opts(r: &Row, pass: &str) -> String {
    let mut o = format!("{pass} alpha({})", fmt(pss_g(r, "alpha")));
    if r.get("_onesided").is_some() {
        o.push_str(" onesided");
    }
    o
}

fn pss_g(r: &Row, k: &str) -> f64 {
    r.get(k).copied().unwrap_or(f64::NAN)
}

fn has(r: &Row, k: &str) -> bool {
    r.get(k).is_some_and(|v| !v.is_nan())
}

/// Smallest size m (integer unless `frac`) for which `ok(m)` holds.
fn search(mut ok: impl FnMut(f64) -> Result<bool, String>, frac: bool) -> Result<f64, String> {
    let mut lo = 1.0;
    if ok(lo)? && !frac {
        return Ok(lo);
    }
    let mut hi = 2.0;
    while !ok(hi)? {
        lo = hi;
        hi *= 2.0;
        if hi > 1e9 {
            return Err(rc_err(498, "sample size could not be computed"));
        }
    }
    if frac {
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if ok(mid)? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return Ok(hi);
    }
    while hi - lo > 1.0 {
        let mid = ((lo + hi) / 2.0).floor();
        if ok(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

const STD: &[&str] = &["power", "beta", "alpha", "N", "N1", "N2", "nratio", "onesided", "width", "Pr_width", "level"];

/// Add the program's own scalars to the row and to the displayed list.
fn absorb(row: &mut Row, res: &Results, show: &mut Vec<String>) {
    for (k, val) in &res.scalars {
        if !STD.contains(&k.as_str()) {
            row.insert(k.clone(), *val);
            if !show.contains(k) && k != "delta" {
                show.push(k.clone());
            }
        }
    }
    if let Some(d) = res.scalars.get("delta") {
        row.insert("delta".into(), *d);
    }
}

fn common_rows(cmd: &Command, two: bool) -> Result<Vec<Row>, String> {
    let mut l: Vec<(String, Vec<f64>)> = vec![("alpha".into(), pss::opt_list(cmd, "alpha", 1)?.unwrap_or(vec![0.05]))];
    if let Some(p) = pss::opt_list(cmd, "power", 1)? {
        l.push(("power".into(), p));
    } else if let Some(b) = pss::opt_list(cmd, "beta", 1)? {
        l.push(("power".into(), b.iter().map(|x| 1.0 - x).collect()));
    }
    for (k, o) in [("N", "n"), ("N1", "n1"), ("N2", "n2")] {
        if let Some(v) = pss::list_exact(cmd, o)? {
            l.push((k.into(), v));
        }
    }
    if two {
        if let Some(v) = pss::opt_list(cmd, "nratio", 6)? {
            l.push(("nratio".into(), v));
        }
    }
    if cmd.has_opt("onesided", 3) {
        l.push(("_onesided".into(), vec![1.0]));
    }
    if cmd.has_opt("nfractional", 5) {
        l.push(("_nfrac".into(), vec![1.0]));
    }
    pss::grid(&l, cmd.has_opt("parallel", 3))
}

fn user_power(me: &mut AdoInterpreter, cmd: &Command, prog: &str, name: &str, args: &str, pass: &str) -> Result<String, String> {
    let two = cmd.opts.iter().any(|(n, _)| matches!(n.to_lowercase().as_str(), "n1" | "n2" | "nratio"));
    let mut rows = common_rows(cmd, two)?;
    let mut show: Vec<String> = Vec::new();
    let mut solve = Solve::Power;
    for row in rows.iter_mut() {
        let base = base_opts(row, pass);
        let size_opts = |r: &Row, n: f64| -> String {
            if two {
                let ratio = if has(r, "nratio") { pss_g(r, "nratio") } else { 1.0 };
                let (n1, n2) = if has(r, "N1") && has(r, "N2") {
                    (pss_g(r, "N1"), pss_g(r, "N2"))
                } else {
                    let n1 = (n / (1.0 + ratio)).round();
                    (n1, n - n1)
                };
                format!(" n({}) n1({}) n2({}) nratio({})", fmt(n1 + n2), fmt(n1), fmt(n2), fmt(n2 / n1))
            } else {
                format!(" n({})", fmt(n))
            }
        };
        let given_n = if has(row, "N") {
            Some(pss_g(row, "N"))
        } else if has(row, "N1") && has(row, "N2") {
            Some(pss_g(row, "N1") + pss_g(row, "N2"))
        } else {
            None
        };
        let res = match given_n {
            Some(n) => {
                solve = Solve::Power;
                let res = call(me, prog, args, &format!("{base}{}", size_opts(row, n)))?;
                if res.scalars.get("power").is_none() {
                    return Err(rc_err(198, format!("{prog} did not return r(power)")));
                }
                row.insert("N".into(), n);
                res
            }
            None => {
                solve = Solve::N;
                let target = if has(row, "power") { pss_g(row, "power") } else { 0.8 };
                row.insert("_ptarget".into(), target);
                let first = call(me, prog, args, &format!("{base} power({})", fmt(target)));
                match first {
                    Ok(res) if res.scalars.contains_key("N") => {
                        row.insert("N".into(), res.scalars["N"]);
                        res
                    }
                    _ => {
                        let frac = row.contains_key("_nfrac");
                        let r0 = row.clone();
                        let n = search(
                            |n| {
                                let res = call(me, prog, args, &format!("{base}{}", size_opts(&r0, n)))?;
                                let p = res.scalars.get("power").copied().ok_or_else(|| rc_err(198, format!("{prog} did not return r(power)")))?;
                                Ok(p >= target)
                            },
                            frac,
                        )?;
                        row.insert("N".into(), n);
                        call(me, prog, args, &format!("{base}{}", size_opts(&r0, n)))?
                    }
                }
            }
        };
        let p = res.scalars.get("power").copied().unwrap_or_else(|| pss_g(row, "power"));
        row.insert("power".into(), p);
        row.insert("beta".into(), 1.0 - p);
        if two {
            for k in ["N1", "N2", "nratio"] {
                if let Some(v) = res.scalars.get(k) {
                    row.insert(k.into(), *v);
                }
            }
            if !has(row, "N1") {
                let n = pss_g(row, "N");
                let ratio = if has(row, "nratio") { pss_g(row, "nratio") } else { 1.0 };
                let n1 = (n / (1.0 + ratio)).round();
                row.insert("N1".into(), n1);
                row.insert("N2".into(), n - n1);
            }
        }
        absorb(row, &res, &mut show);
    }
    let rep = Report {
        kind: "power".into(),
        method: name.to_string(),
        title: format!("user-defined method {name}"),
        test: String::new(),
        hyp: String::new(),
        solve,
        target: None,
        tword: String::new(),
        two,
        cluster: false,
        show,
        extra: vec![],
        rows,
    };
    pss::report(&mut me.engine.store, cmd, rep)
}

fn user_ciwidth(me: &mut AdoInterpreter, cmd: &Command, prog: &str, name: &str, args: &str, pass: &str) -> Result<String, String> {
    let mut rows = common_rows(cmd, false)?;
    let widths = pss::opt_list(cmd, "width", 5)?;
    if let Some(w) = widths {
        let mut next = Vec::new();
        for r in &rows {
            for x in &w {
                let mut r2 = r.clone();
                r2.insert("width".into(), *x);
                next.push(r2);
            }
        }
        rows = next;
    }
    let mut show = Vec::new();
    let mut solve = Solve::Width;
    for row in rows.iter_mut() {
        let base = base_opts(row, pass);
        let res = if has(row, "N") {
            let mut o = format!("{base} n({})", fmt(pss_g(row, "N")));
            if has(row, "width") {
                o.push_str(&format!(" width({})", fmt(pss_g(row, "width"))));
                solve = Solve::ProbWidth;
            } else {
                solve = Solve::Width;
            }
            call(me, prog, args, &o)?
        } else {
            solve = Solve::N;
            let w = pss_g(row, "width");
            if w.is_nan() {
                return Err(rc_err(198, "width() required to compute the sample size"));
            }
            let frac = row.contains_key("_nfrac");
            let n = search(
                |n| {
                    let res = call(me, prog, args, &format!("{base} n({})", fmt(n)))?;
                    let got = res.scalars.get("width").copied().ok_or_else(|| rc_err(198, format!("{prog} did not return r(width)")))?;
                    Ok(got <= w)
                },
                frac,
            )?;
            row.insert("N".into(), n);
            call(me, prog, args, &format!("{base} n({})", fmt(n)))?
        };
        if solve == Solve::Width {
            let w = res.scalars.get("width").copied().ok_or_else(|| rc_err(198, format!("{prog} did not return r(width)")))?;
            row.insert("width".into(), w);
        }
        if let Some(p) = res.scalars.get("Pr_width") {
            row.insert("Pr_width".into(), *p);
        }
        absorb(row, &res, &mut show);
    }
    let rep = Report {
        kind: "ciwidth".into(),
        method: name.to_string(),
        title: format!("user-defined method {name}"),
        test: String::new(),
        hyp: String::new(),
        solve,
        target: None,
        tword: String::new(),
        two: false,
        cluster: false,
        show,
        extra: vec![],
        rows,
    };
    pss::report(&mut me.engine.store, cmd, rep)
}

fn user_gs(me: &mut AdoInterpreter, cmd: &Command, prog: &str, name: &str, args: &str, pass: &str) -> Result<String, String> {
    let spec = pss::gs::gs_spec(cmd)?;
    let mut base = format!("{pass} alpha({})", fmt(spec.alpha));
    if !spec.two {
        base.push_str(" onesided");
    }
    let target = 1.0 - spec.beta;
    let mut params = Vec::new();
    let nfix = match call(me, prog, args, &format!("{base} power({})", fmt(target))) {
        Ok(res) if res.scalars.contains_key("N") => {
            for (k, v) in &res.scalars {
                if !STD.contains(&k.as_str()) {
                    params.push((k.clone(), *v));
                }
            }
            res.scalars["N"]
        }
        _ => search(
            |n| {
                let res = call(me, prog, args, &format!("{base} n({})", fmt(n)))?;
                let p = res.scalars.get("power").copied().ok_or_else(|| rc_err(198, format!("{prog} did not return r(power)")))?;
                Ok(p >= target)
            },
            true,
        )?,
    };
    let fixed = pss::gs::Fixed { title: format!("user-defined method {name}"), test: String::new(), sizes: vec![("N".into(), nfix)], params };
    let out = pss::gs::run(&mut me.engine.store, cmd, &spec, Some(fixed))?;
    me.engine.store.r.set_str("cmd", "gsdesign");
    me.engine.store.r.set_str("method", name.to_string());
    Ok(out)
}
