//! Power, precision and sample-size analysis.
//!
//! * `power <method>`: power, sample size or effect size for many tests,
//!   with parameter lists producing tables and graphs;
//! * `ciwidth <method>`: precision (confidence-interval width) analysis;
//! * `gsdesign`, `gsbounds`: group-sequential designs whose boundaries come
//!   from recursive numerical integration of the joint distribution of the
//!   sequential test statistics.
//!
//! User-written methods (`power_cmd_<name>`, `ciwidth_cmd_<name>`) need the
//! program interpreter and live in the ado crate; they reuse [`report`].

pub mod ciwidth;
pub mod gs;
mod methods;
mod methods2;
pub mod ncdist;
mod report;

use std::collections::BTreeMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;

pub use report::{report, Report, Solve};

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("power", Box::new(Power)),
        ("ciwidth", Box::new(ciwidth::Ciwidth)),
        ("gsdesign", Box::new(gs::Gsdesign)),
        ("gsbounds", Box::new(gs::Gsbounds)),
    ]
}

/// One design point: parameter name → value. Names starting with `_` are
/// settings (test variant, flags) that are not displayed.
pub type Row = BTreeMap<String, f64>;

pub(crate) fn g(r: &Row, k: &str) -> f64 {
    r.get(k).copied().unwrap_or(f64::NAN)
}
pub(crate) fn has(r: &Row, k: &str) -> bool {
    r.get(k).is_some_and(|v| !v.is_nan())
}
pub(crate) fn flag(r: &Row, k: &str) -> bool {
    r.get(k).is_some_and(|v| *v != 0.0 && !v.is_nan())
}

/// Power function: (design point, effective size group 1, group 2).
pub(crate) type PowerFn = fn(&Row, f64, f64) -> f64;

/// A power method.
pub(crate) struct Def {
    pub method: String,
    /// "a one-sample mean test"
    pub title: String,
    pub test: String,
    pub hyp: String,
    /// two independent samples (N1, N2, nratio)
    pub two: bool,
    /// one-sample designs: N is a multiple of this (groups, cells)
    pub step: f64,
    /// cluster-randomized designs allowed
    pub cluster: bool,
    /// parameter solved for in effect-size mode, its null/reference
    /// parameter (for the direction) and a word for the heading
    pub target: Option<&'static str>,
    pub tword: &'static str,
    pub power: PowerFn,
    pub delta: fn(&Row) -> f64,
    /// (near, far) search range of the target: power increases from near to far
    pub erange: fn(&Row, bool) -> (f64, f64),
    /// exact (saw-toothed) power: sample size by linear search
    pub exact: bool,
    /// displayed parameters, in order
    pub show: Vec<String>,
    /// extra results computed from the solution (events, ...)
    pub extra: Option<fn(&Row, f64, f64) -> Vec<(String, f64)>>,
    /// smallest group size the method accepts
    pub nmin: f64,
    /// effect-size search goes below the null by default (hazard ratios)
    pub lower_default: bool,
}

impl Default for Def {
    fn default() -> Self {
        Def {
            method: String::new(),
            title: String::new(),
            test: String::new(),
            hyp: String::new(),
            two: false,
            step: 1.0,
            cluster: false,
            target: None,
            tword: "value",
            power: |_, _, _| f64::NAN,
            delta: |_| f64::NAN,
            erange: |_, _| (0.0, 1.0),
            exact: false,
            show: Vec::new(),
            extra: None,
            nmin: 2.0,
            lower_default: false,
        }
    }
}

// ---------------------------------------------------------------------------
// argument parsing
// ---------------------------------------------------------------------------

/// Positional arguments: rows separated by `\`, each a list of items; an
/// item is a number or a parenthesized numlist.
pub(crate) fn positional(main: &str) -> Result<Vec<Vec<Vec<f64>>>, String> {
    let mut rows = Vec::new();
    for part in main.split('\\') {
        let mut items = Vec::new();
        let mut rest = part.trim();
        while !rest.is_empty() {
            if let Some(r) = rest.strip_prefix('(') {
                let close = r.find(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
                items.push(crate::margins::numlist(&r[..close])?);
                rest = r[close + 1..].trim_start();
            } else {
                let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
                let w = &rest[..end];
                let v: f64 = w.parse().map_err(|_| rc_err(198, format!("'{w}' found where number expected")))?;
                items.push(vec![v]);
                rest = rest[end..].trim_start();
            }
        }
        rows.push(items);
    }
    while rows.last().is_some_and(|r: &Vec<Vec<f64>>| r.is_empty()) && rows.len() > 1 {
        rows.pop();
    }
    Ok(rows)
}

/// A numeric option that may hold a numlist.
pub fn opt_list(cmd: &Command, name: &str, min: usize) -> Result<Option<Vec<f64>>, String> {
    match cmd.opt(name, min) {
        Some(t) => {
            let t = t.trim();
            if let Ok(v) = crate::margins::numlist(t) {
                if !v.is_empty() {
                    return Ok(Some(v));
                }
            }
            // an expression such as 1/3 or ln(2)
            Ok(crate::util::opt_num(cmd, name, min)?.map(|v| vec![v]))
        }
        None => Ok(None),
    }
}

/// Option given with an exact name (no abbreviation), so that e.g. `m()`
/// does not match `m1()`.
pub(crate) fn opt_exact<'a>(cmd: &'a Command, name: &str) -> Option<&'a str> {
    cmd.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).and_then(|(_, a)| a.as_deref())
}

pub fn list_exact(cmd: &Command, name: &str) -> Result<Option<Vec<f64>>, String> {
    match opt_exact(cmd, name) {
        Some(t) => crate::margins::numlist(t.trim()).map(Some),
        None => Ok(None),
    }
}

pub(crate) fn word_opt(cmd: &Command, name: &str, min: usize) -> Option<String> {
    cmd.opt(name, min).map(|s| s.trim().to_lowercase())
}

/// Lists of parameter values, crossed (or paired with `parallel`) into rows.
pub fn grid(lists: &[(String, Vec<f64>)], parallel: bool) -> Result<Vec<Row>, String> {
    let mut rows: Vec<Row> = vec![Row::new()];
    if parallel {
        let len = lists.iter().map(|l| l.1.len()).max().unwrap_or(1);
        let mut out = Vec::new();
        for i in 0..len {
            let mut r = Row::new();
            for (k, v) in lists {
                r.insert(k.clone(), *v.get(i).or(v.last()).unwrap_or(&f64::NAN));
            }
            out.push(r);
        }
        return Ok(out);
    }
    for (k, vals) in lists {
        let mut next = Vec::with_capacity(rows.len() * vals.len());
        for r in &rows {
            for v in vals {
                let mut r2 = r.clone();
                r2.insert(k.clone(), *v);
                next.push(r2);
            }
        }
        rows = next;
        if rows.len() > 100_000 {
            return Err(rc_err(198, "too many combinations of parameter values"));
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// design sizes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum SolveFor {
    Power,
    N,
    Effect,
}

fn deff(m: f64, rho: f64) -> f64 {
    1.0 + (m - 1.0) * rho
}

/// Effective group sizes of a design point.
pub(crate) fn eff_sizes(def: &Def, r: &Row) -> (f64, f64) {
    let rho = if has(r, "rho") { g(r, "rho") } else { 0.0 };
    if flag(r, "_cluster") {
        if def.two {
            let (k1, k2, m1, m2) = (g(r, "K1"), g(r, "K2"), g(r, "M1"), g(r, "M2"));
            (k1 * m1 / deff(m1, rho), k2 * m2 / deff(m2, rho))
        } else {
            let (k, m) = (g(r, "K"), g(r, "M"));
            (k * m / deff(m, rho), 0.0)
        }
    } else if def.two {
        (g(r, "N1"), g(r, "N2"))
    } else {
        (g(r, "N"), 0.0)
    }
}

pub(crate) fn power_of(def: &Def, r: &Row) -> f64 {
    let (a, b) = eff_sizes(def, r);
    let p = (def.power)(r, a, b);
    if p.is_nan() {
        0.0
    } else {
        p.clamp(0.0, 1.0)
    }
}

/// Fill derived sizes (N, N1/N2, nratio) of a row from what was given.
pub(crate) fn complete_sizes(def: &Def, r: &mut Row) {
    if flag(r, "_cluster") {
        if def.two {
            for (a, b, rat) in [("K1", "K2", "kratio"), ("M1", "M2", "mratio")] {
                let ratio = if has(r, rat) { g(r, rat) } else { 1.0 };
                if has(r, a) && !has(r, b) {
                    let v = g(r, a) * ratio;
                    r.insert(b.into(), if flag(r, "_nfrac") { v } else { v.ceil() });
                } else if has(r, b) && !has(r, a) {
                    let v = g(r, b) / ratio;
                    r.insert(a.into(), if flag(r, "_nfrac") { v } else { v.ceil() });
                }
            }
            let (n1, n2) = (g(r, "K1") * g(r, "M1"), g(r, "K2") * g(r, "M2"));
            r.insert("N1".into(), n1);
            r.insert("N2".into(), n2);
            r.insert("N".into(), n1 + n2);
        } else {
            let n = g(r, "K") * g(r, "M");
            r.insert("N".into(), n);
        }
        return;
    }
    if def.two {
        let frac = flag(r, "_nfrac");
        let ratio = if has(r, "nratio") { g(r, "nratio") } else { 1.0 };
        let (n, n1, n2) = (g(r, "N"), g(r, "N1"), g(r, "N2"));
        if has(r, "N1") && has(r, "N2") {
            r.insert("N".into(), n1 + n2);
            r.insert("nratio".into(), n2 / n1);
        } else if has(r, "N1") {
            let v = n1 * ratio;
            r.insert("N2".into(), if frac { v } else { v.ceil() });
            r.insert("N".into(), n1 + g(r, "N2"));
        } else if has(r, "N2") {
            let v = n2 / ratio;
            r.insert("N1".into(), if frac { v } else { v.ceil() });
            r.insert("N".into(), n2 + g(r, "N1"));
        } else if has(r, "N") {
            let a = n / (1.0 + ratio);
            let (a, b) = if frac { (a, n - a) } else { (a.round(), n - a.round()) };
            r.insert("N1".into(), a);
            r.insert("N2".into(), b);
        }
        if !has(r, "nratio") && has(r, "N1") {
            r.insert("nratio".into(), g(r, "N2") / g(r, "N1"));
        }
    }
}

/// Smallest size meeting the target power; `set` writes candidate `m` into
/// the row. Integer search unless `frac`.
fn search_size(def: &Def, r: &Row, target: f64, lo0: f64, frac: bool, set: &dyn Fn(&mut Row, f64)) -> Result<f64, String> {
    let pw = |m: f64| {
        let mut rr = r.clone();
        set(&mut rr, m);
        power_of(def, &rr)
    };
    let mut lo = lo0;
    if pw(lo) >= target && !frac {
        return Ok(lo);
    }
    let mut hi = (lo * 2.0).max(lo + 1.0);
    while pw(hi) < target {
        lo = hi;
        hi *= 2.0;
        if hi > 1e10 {
            return Err(rc_err(498, "sample size could not be computed: target power not reached"));
        }
    }
    if def.exact && !frac && hi < 20_000.0 {
        // saw-toothed power: first size reaching the target
        let start = lo0.max(1.0);
        let mut m = start;
        while m <= hi {
            if pw(m) >= target {
                return Ok(m);
            }
            m += 1.0;
        }
        return Ok(hi);
    }
    if frac {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if pw(mid) >= target {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo < 1e-9 * hi {
                break;
            }
        }
        return Ok(hi);
    }
    while hi - lo > 1.0 {
        let mid = ((lo + hi) / 2.0).floor();
        if pw(mid) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(if pw(lo) >= target { lo } else { hi })
}

/// Solve a row for its sample size.
pub(crate) fn solve_n(def: &Def, r: &mut Row, target: f64, compute: Option<&str>) -> Result<(), String> {
    let frac = flag(r, "_nfrac");
    let nmin = def.nmin;
    if flag(r, "_cluster") {
        complete_sizes(def, r);
        let solved_k = if def.two { !(has(r, "K1") || has(r, "K2")) } else { !has(r, "K") };
        r.insert("_solvedK".into(), if solved_k { 1.0 } else { 0.0 });
        if def.two {
            let solve_k = !(has(r, "K1") || has(r, "K2"));
            let (a, b, rat) = if solve_k { ("K1", "K2", "kratio") } else { ("M1", "M2", "mratio") };
            if !solve_k && (has(r, "M1") || has(r, "M2")) {
                return Err(rc_err(198, "nothing to solve for: k1(), k2(), m1() and m2() all specified"));
            }
            let ratio = if has(r, rat) { g(r, rat) } else { 1.0 };
            let set = move |rr: &mut Row, m: f64| {
                rr.insert(a.into(), m);
                rr.insert(b.into(), if frac { m * ratio } else { (m * ratio).ceil() });
            };
            let m = search_size(def, r, target, if solve_k { 2.0 } else { 1.0 }, frac, &set)?;
            set(r, m);
        } else {
            let solve_k = !has(r, "K");
            if !solve_k && has(r, "M") {
                return Err(rc_err(198, "nothing to solve for: k() and m() both specified"));
            }
            let key = if solve_k { "K" } else { "M" };
            let set = move |rr: &mut Row, m: f64| {
                rr.insert(key.into(), m);
            };
            let m = search_size(def, r, target, if solve_k { 2.0 } else { 1.0 }, frac, &set)?;
            set(r, m);
        }
        complete_sizes(def, r);
        return Ok(());
    }
    if def.two {
        let ratio = if has(r, "nratio") { g(r, "nratio") } else { 1.0 };
        let comp = compute.map(|c| c.to_lowercase());
        if has(r, "N1") || comp.as_deref() == Some("n2") {
            if !has(r, "N1") {
                return Err(rc_err(198, "compute(N2) requires n1()"));
            }
            let set = |rr: &mut Row, m: f64| {
                rr.insert("N2".into(), m);
            };
            let m = search_size(def, r, target, nmin, frac, &set)?;
            set(r, m);
        } else if has(r, "N2") || comp.as_deref() == Some("n1") {
            if !has(r, "N2") {
                return Err(rc_err(198, "compute(N1) requires n2()"));
            }
            let set = |rr: &mut Row, m: f64| {
                rr.insert("N1".into(), m);
            };
            let m = search_size(def, r, target, nmin, frac, &set)?;
            set(r, m);
        } else {
            let set = move |rr: &mut Row, m: f64| {
                rr.insert("N1".into(), m);
                rr.insert("N2".into(), if frac { m * ratio } else { (m * ratio).ceil().max(1.0) });
            };
            let lo = if ratio < 1.0 { (nmin / ratio).ceil() } else { nmin };
            let m = search_size(def, r, target, lo, frac, &set)?;
            set(r, m);
        }
        let (n1, n2) = (g(r, "N1"), g(r, "N2"));
        r.insert("N".into(), n1 + n2);
        r.insert("nratio".into(), n2 / n1);
        return Ok(());
    }
    let step = def.step;
    let set = move |rr: &mut Row, m: f64| {
        rr.insert("N".into(), m * step);
    };
    let lo = (nmin / step).ceil().max(1.0);
    let m = search_size(def, r, target, lo, frac, &set)?;
    set(r, m);
    Ok(())
}

/// Solve a row for the target parameter.
pub(crate) fn solve_effect(def: &Def, r: &mut Row, target: f64, upper: bool) -> Result<(), String> {
    let t = def.target.ok_or_else(|| rc_err(198, format!("power {}: computing the effect size is not supported", def.method)))?;
    let (mut near, mut far) = (def.erange)(r, upper);
    let pw = |x: f64| {
        let mut rr = r.clone();
        rr.insert(t.into(), x);
        power_of(def, &rr)
    };
    if pw(far) < target {
        return Err(rc_err(498, format!("power {}: target power cannot be reached in the {} direction", def.method, if upper { "upper" } else { "lower" })));
    }
    for _ in 0..200 {
        let mid = 0.5 * (near + far);
        if pw(mid) >= target {
            far = mid;
        } else {
            near = mid;
        }
    }
    r.insert(t.into(), 0.5 * (near + far));
    Ok(())
}

// ---------------------------------------------------------------------------
// the power command
// ---------------------------------------------------------------------------

pub struct Power;

/// Built-in method names (full and short forms).
pub fn is_builtin(name: &str) -> bool {
    methods::canonical(name).is_some()
}

/// Common options shared by all methods, as row lists.
pub(crate) fn common_lists(cmd: &Command, def: &Def, lists: &mut Vec<(String, Vec<f64>)>, cluster: bool) -> Result<(), String> {
    let mut push = |k: &str, v: Option<Vec<f64>>| {
        if let Some(v) = v {
            lists.push((k.to_string(), v));
        }
    };
    push("alpha", Some(opt_list(cmd, "alpha", 1)?.unwrap_or(vec![0.05])));
    if let Some(p) = opt_list(cmd, "power", 1)? {
        push("power", Some(p));
    } else if let Some(b) = opt_list(cmd, "beta", 1)? {
        push("power", Some(b.iter().map(|x| 1.0 - x).collect()));
    }
    if cluster {
        if def.two {
            push("K1", list_exact(cmd, "k1")?);
            push("K2", list_exact(cmd, "k2")?);
            push("M1", list_exact(cmd, "m1")?);
            push("M2", list_exact(cmd, "m2")?);
            push("kratio", opt_list(cmd, "kratio", 6)?);
            push("mratio", opt_list(cmd, "mratio", 6)?);
            if let Some(k) = list_exact(cmd, "k")? {
                push("K1", Some(k.clone()));
                push("K2", Some(k));
            }
            if let Some(m) = list_exact(cmd, "m")? {
                push("M1", Some(m.clone()));
                push("M2", Some(m));
            }
        } else {
            push("K", list_exact(cmd, "k")?);
            push("M", list_exact(cmd, "m")?);
        }
        push("rho", Some(list_exact(cmd, "rho")?.unwrap_or(vec![0.0])));
        lists.push(("_cluster".into(), vec![1.0]));
    } else {
        push("N", list_exact(cmd, "n")?);
        if def.two {
            push("N1", list_exact(cmd, "n1")?);
            push("N2", list_exact(cmd, "n2")?);
            push("nratio", opt_list(cmd, "nratio", 6)?);
        }
    }
    if cmd.has_opt("onesided", 3) {
        lists.push(("_onesided".into(), vec![1.0]));
    }
    if cmd.has_opt("nfractional", 5) {
        lists.push(("_nfrac".into(), vec![1.0]));
    }
    Ok(())
}

/// Which quantity a row is solved for.
pub(crate) fn solve_for(def: &Def, r: &Row, cmd: &Command) -> SolveFor {
    let have_n = if flag(r, "_cluster") {
        if def.two {
            (has(r, "K1") || has(r, "K2")) && (has(r, "M1") || has(r, "M2"))
        } else {
            has(r, "K") && has(r, "M")
        }
    } else if def.two {
        has(r, "N") || (has(r, "N1") && has(r, "N2")) || ((has(r, "N1") || has(r, "N2")) && cmd.opt("compute", 4).is_none())
    } else {
        has(r, "N")
    };
    let have_target = def.target.map_or(true, |t| has(r, t) || target_implied(r, t));
    if !have_n {
        SolveFor::N
    } else if !have_target || (has(r, "power") && def.target.is_some()) {
        SolveFor::Effect
    } else {
        SolveFor::Power
    }
}

/// Alternatives given through diff()/ratio()/... count as the target.
fn target_implied(r: &Row, t: &str) -> bool {
    match t {
        "ma" | "m2" | "pa" | "p2" | "ma2" | "va" | "v2" | "h2" => ["diff", "ratio", "oratio", "rdiff", "rrisk", "altdiff", "hratio", "lhratio", "hdiff"].iter().any(|k| has(r, k)),
        _ => false,
    }
}

impl CommandHandler for Power {
    fn name(&self) -> &'static str {
        "power"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub_l = sub.to_lowercase();
        if sub_l.is_empty() {
            return Err(rc_err(198, "power: method required"));
        }
        let canon = methods::canonical(&sub_l).ok_or_else(|| rc_err(199, format!("power method {sub} not found")))?;
        let mut rest = rest.trim();
        let mut general = false;
        if canon == "logistic" {
            if let Some(r2) = rest.strip_prefix("general") {
                rest = r2.trim_start();
                general = true;
            }
        }
        let pos = positional(rest)?;
        let (mut def, mut lists) = methods::setup(canon, cmd, &pos)?;
        if general && !lists.iter().any(|l| l.0 == "_general") {
            lists.push(("_general".into(), vec![1.0]));
            def.test = "Wald test, information-based (general covariate distribution)".into();
        }
        let cluster = cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("cluster")) || (def.cluster && (opt_exact(cmd, "k").is_some() || opt_exact(cmd, "k1").is_some() || opt_exact(cmd, "k2").is_some() || opt_exact(cmd, "rho").is_some()));
        if cluster && !def.cluster {
            return Err(rc_err(198, format!("option cluster not allowed with power {canon}")));
        }
        common_lists(cmd, &def, &mut lists, cluster)?;
        if cluster {
            let t = if def.test == "t test" { "z test".to_string() } else { def.test.clone() };
            def.test = format!("{t}, CRD");
        }
        let parallel = cmd.has_opt("parallel", 3);
        let mut rows = grid(&lists, parallel)?;
        let upper = match word_opt(cmd, "direction", 3) {
            Some(d) => !d.starts_with('l'),
            None => !def.lower_default,
        };
        let mut solve = SolveFor::Power;
        for r in rows.iter_mut() {
            methods::prepare(&def, r)?;
            solve = solve_for(&def, r, cmd);
            if solve != SolveFor::N {
                complete_sizes(&def, r);
            }
            let target = if has(r, "power") { g(r, "power") } else { 0.8 };
            r.insert("_ptarget".into(), target);
            match solve {
                SolveFor::N => {
                    solve_n(&def, r, target, cmd.opt("compute", 4))?;
                }
                SolveFor::Effect => {
                    solve_effect(&def, r, target, upper)?;
                }
                SolveFor::Power => {}
            }
            complete_sizes(&def, r);
            let p = power_of(&def, r);
            r.insert("power".into(), p);
            r.insert("beta".into(), 1.0 - p);
            r.insert("delta".into(), (def.delta)(r));
            if let Some(f) = def.extra {
                let (a, b) = eff_sizes(&def, r);
                for (k, v) in f(r, a, b) {
                    r.insert(k, v);
                }
            }
        }
        let rep = Report::from_def(&def, solve, rows, cluster);
        report(st, cmd, rep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positional_parse() {
        let p = positional("1 (2 3) \\ 4 5").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0][1], vec![2.0, 3.0]);
        assert_eq!(p[1][1], vec![5.0]);
    }
}
