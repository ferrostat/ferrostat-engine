//! Resampling prefixes: `bootstrap`, `simulate` and `permute`, the
//! `vce(bootstrap)` option of estimation commands and `estat bootstrap`.
//!
//! Each prefix runs a command repeatedly through the interpreter (so user
//! programs, `capture`, loops and `by` all work inside it), collects the
//! statistics of an expression list after every run and summarises them.
//! Random draws use the session generator, so `set seed` is honoured.

use std::collections::HashMap;

use ferrostat_cmd::dist;
use ferrostat_cmd::est::{self, TableSpec};
use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};
use ferrostat_parser::Expr;

use crate::interpreter::{split_verb, AdoInterpreter, Flow};
use crate::macros::unquote;

type Opts = Vec<(String, Option<String>)>;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Handle the resampling prefixes and related syntax; None when `verb` is
/// something else.
pub(crate) fn intercept(me: &mut AdoInterpreter, verb_raw: &str, rest: &str, text: &str) -> Result<Option<Flow>, String> {
    let v = verb_raw.to_lowercase();
    if me.engine.store.programs.contains_key(verb_raw) {
        return Ok(None);
    }
    let is = |full: &str, min: usize| v.len() >= min && full.starts_with(v.as_str());
    if v == "bs" || is("bootstrap", 4) {
        bootstrap(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if is("simulate", 3) {
        simulate(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if is("permute", 4) {
        permute(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if v == "estat" {
        let sub = rest.trim().split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").to_lowercase();
        if sub.len() >= 4 && "bootstrap".starts_with(sub.as_str()) {
            let opts = rest.trim()[sub.len()..].trim().trim_start_matches(',');
            let out = estat_bootstrap(me, opts)?;
            me.emit(&out);
            return Ok(Some(Flow::Next));
        }
        return Ok(None);
    }
    if let Some(prefixed) = vce_bootstrap(me, &v, text) {
        bootstrap(me, &prefixed)?;
        return Ok(Some(Flow::Next));
    }
    Ok(None)
}

/// `cmd ..., vce(bootstrap[, opts]) ...` rewritten as
/// `_b, opts : cmd ...` (the arguments of the bootstrap prefix).
fn vce_bootstrap(me: &AdoInterpreter, verb: &str, text: &str) -> Option<String> {
    if !text.contains("vce(") || matches!(verb, "display" | "di" | "dis" | "local" | "global" | "estat") || !me.engine.is_command(&me.engine.resolve_verb(verb)) {
        return None;
    }
    let comma = find_top_level(text, ',')?;
    let opts = split_options(&text[comma + 1..]);
    let k = opts.iter().position(|(n, a)| {
        n.len() >= 3
            && "vce".starts_with(n.to_lowercase().as_str())
            && a.as_deref().is_some_and(|a| {
                let w = a.trim().split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").to_lowercase();
                w == "bs" || (w.len() >= 4 && "bootstrap".starts_with(w.as_str()))
            })
    })?;
    let arg = opts[k].1.clone().unwrap_or_default();
    let inner = arg.find(',').map(|i| arg[i + 1..].trim().to_string()).unwrap_or_default();
    let others: Vec<String> = opts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != k)
        .map(|(_, (n, a))| match a {
            Some(a) => format!("{n}({a})"),
            None => n.clone(),
        })
        .collect();
    let mut cmd = text[..comma].trim().to_string();
    if !others.is_empty() {
        cmd = format!("{cmd}, {}", others.join(" "));
    }
    Some(format!("_b, {inner} : {cmd}"))
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Kind {
    B,
    Se,
    Expr(String),
}

#[derive(Clone, Debug)]
struct ExpItem {
    name: Option<String>,
    kind: Kind,
}

struct Prefix {
    exps: Vec<ExpItem>,
    opts: Opts,
    command: String,
    /// Leading words before the expression list (`permute` variable).
    lead: Option<String>,
}

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| {
        let o = o.to_lowercase();
        o.len() >= min && name.starts_with(o.as_str())
    })
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opts.iter()
        .find(|(o, _)| {
            let o = o.to_lowercase();
            o.len() >= min && name.starts_with(o.as_str())
        })
        .and_then(|(_, a)| a.clone())
}

fn num_arg(opts: &Opts, name: &str, min: usize) -> Result<Option<f64>, String> {
    match arg(opts, name, min) {
        Some(a) => a.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("option {name}() incorrectly specified"))),
        None => Ok(None),
    }
}

fn parse_prefix(rest: &str, with_lead: bool) -> Result<Prefix, String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "a colon followed by a command is required"))?;
    let head = rest[..colon].trim();
    let command = rest[colon + 1..].trim().to_string();
    if command.is_empty() {
        return Err(rc_err(198, "command required after the colon"));
    }
    let (list, opts) = match find_top_level(head, ',') {
        Some(i) => (head[..i].trim(), split_options(&head[i + 1..])),
        None => (head, Vec::new()),
    };
    let mut list = list.to_string();
    let mut lead = None;
    if with_lead {
        let w = split_top_level_words(&list);
        let first = w.first().cloned().ok_or_else(|| rc_err(100, "permutation variable required"))?;
        list = list.trim_start()[first.len()..].to_string();
        lead = Some(first);
    }
    Ok(Prefix { exps: parse_exp_list(&list)?, opts, command, lead })
}

/// Position of a top-level `=` that is an assignment (not `==`, `<=`, ...).
fn assign_pos(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut quote = false;
    for i in 0..b.len() {
        match b[i] {
            b'"' => quote = !quote,
            b'(' | b'[' if !quote => depth += 1,
            b')' | b']' if !quote => depth -= 1,
            b'=' if !quote && depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { b' ' };
                let next = b.get(i + 1).copied().unwrap_or(b' ');
                if next != b'=' && !matches!(prev, b'=' | b'<' | b'>' | b'!' | b'~') {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn item(name: Option<String>, text: &str) -> Result<ExpItem, String> {
    let t = text.trim();
    let kind = match t {
        "_b" | "_coef" => Kind::B,
        "_se" => Kind::Se,
        _ => {
            parse_expression(t).map_err(|e| rc_err(198, format!("{t}: {}", e.message)))?;
            Kind::Expr(t.to_string())
        }
    };
    Ok(ExpItem { name, kind })
}

fn parse_exp_list(s: &str) -> Result<Vec<ExpItem>, String> {
    let words = split_top_level_words(s);
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let w = words[i].clone();
        // `(eqname: exp_list)` groups: the equation name is not kept
        if w.starts_with('(') && w.ends_with(')') {
            let inner = &w[1..w.len() - 1];
            if let Some(c) = find_top_level(inner, ':') {
                out.extend(parse_exp_list(&inner[c + 1..])?);
                i += 1;
                continue;
            }
        }
        if let Some(p) = assign_pos(&w) {
            let name = w[..p].trim().to_string();
            let mut e = w[p + 1..].to_string();
            i += 1;
            if e.is_empty() {
                e = words.get(i).cloned().ok_or_else(|| rc_err(198, "expression expected after ="))?;
                i += 1;
            }
            out.push(item(Some(name), &e)?);
            continue;
        }
        if let Some(next) = words.get(i + 1) {
            if next.starts_with('=') && !next.starts_with("==") {
                let mut e = next[1..].to_string();
                i += 2;
                if e.is_empty() {
                    e = words.get(i).cloned().ok_or_else(|| rc_err(198, "expression expected after ="))?;
                    i += 1;
                }
                out.push(item(Some(w), &e)?);
                continue;
            }
        }
        out.push(item(None, &w)?);
        i += 1;
    }
    for it in &out {
        if let Some(n) = &it.name {
            if n.is_empty() || !n.chars().all(|c| c.is_alphanumeric() || c == '_') || n.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Err(rc_err(198, format!("{n} invalid name")));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Statistics collected after each run
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Src {
    B(String),
    Se(String),
    Expr(usize),
}

#[derive(Clone, Debug)]
struct Stat {
    /// Name in the results table (and coefficient name in e(b)).
    name: String,
    /// Variable name in the dataset of replicates.
    var: String,
    /// Expression text (variable label / legend).
    label: String,
    src: Src,
}

/// Variable name for a coefficient statistic: `_b_x`, `eq_b_x`.
fn coef_var(prefix: &str, col: &str) -> String {
    let (eq, nm) = match col.split_once(':') {
        Some((e, n)) => (e.to_string(), n.to_string()),
        None => (String::new(), col.to_string()),
    };
    let raw = format!("{eq}{prefix}_{}", nm.strip_prefix('_').unwrap_or(&nm));
    let mut s: String = raw.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    s.truncate(32);
    s
}

/// Expand the expression list against the current e() results.
fn build_stats(me: &AdoInterpreter, exps: &[ExpItem], stem: &str) -> Result<(Vec<Stat>, Vec<Expr>), String> {
    let mut stats = Vec::new();
    let mut parsed = Vec::new();
    for (k, it) in exps.iter().enumerate() {
        match &it.kind {
            Kind::B | Kind::Se => {
                let b = me.engine.store.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
                for c in &b.colnames {
                    let se = matches!(it.kind, Kind::Se);
                    let pre = if se { "_se" } else { "_b" };
                    let nm = c.rsplit(':').next().unwrap_or(c);
                    stats.push(Stat {
                        name: if se { format!("_se:{nm}") } else { c.clone() },
                        var: coef_var(pre, c),
                        label: format!("{pre}[{c}]"),
                        src: if se { Src::Se(c.clone()) } else { Src::B(c.clone()) },
                    });
                }
            }
            Kind::Expr(t) => {
                let e = parse_expression(t).map_err(|e| rc_err(198, e.message))?;
                let name = it.name.clone().unwrap_or_else(|| format!("{stem}_{}", k + 1));
                stats.push(Stat { name: name.clone(), var: name, label: t.clone(), src: Src::Expr(parsed.len()) });
                parsed.push(e);
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    for s in &stats {
        if !seen.insert(s.var.clone()) {
            return Err(rc_err(110, format!("{} already defined", s.var)));
        }
    }
    Ok((stats, parsed))
}

fn eval_stats(me: &AdoInterpreter, stats: &[Stat], parsed: &[Expr]) -> Vec<f64> {
    let st = &me.engine.store;
    let ds = &me.engine.dataset;
    let coef = |name: &str, se: bool| -> f64 {
        let Some(b) = st.e.matrices.get("b") else { return SYSMISS };
        let Some(j) = b.colnames.iter().position(|c| c == name) else { return SYSMISS };
        if se {
            st.e.matrices.get("V").map_or(SYSMISS, |v| {
                let x = v.get(j, j);
                if x > 0.0 { x.sqrt() } else { SYSMISS }
            })
        } else {
            b.get(0, j)
        }
    };
    stats
        .iter()
        .map(|s| match &s.src {
            Src::B(c) => coef(c, false),
            Src::Se(c) => coef(c, true),
            Src::Expr(i) => match Env::with_store(ds, st).eval(&parsed[*i], 0) {
                Ok(Val::Num(x)) => x,
                _ => SYSMISS,
            },
        })
        .collect()
}

/// Run a command with its output suppressed (unless `noisily`).
fn run_cmd(me: &mut AdoInterpreter, cmd: &str, noisily: bool) -> Result<(), String> {
    let saved = me.noisy;
    if noisily {
        me.noisy += 1;
    } else {
        me.quiet += 1;
        me.noisy = 0;
    }
    let r = me.exec_text(cmd, None);
    if noisily {
        me.noisy -= 1;
    } else {
        me.quiet -= 1;
    }
    me.noisy = saved;
    r.map(|_| ())
}

/// The dots line: `Title (100): .........10.........20 ... done`.
fn dots_line(title: &str, reps: usize, marks: &[char]) -> String {
    let mut s = format!("{title} ({}): ", util::commas(reps));
    for (i, c) in marks.iter().enumerate() {
        if (i + 1) % 10 == 0 && *c == '.' {
            s.push_str(&util::commas(i + 1));
        } else {
            s.push(*c);
        }
    }
    s.push_str(" done");
    s
}

fn word_count(n: usize) -> String {
    match n {
        1 => "One".into(),
        _ => n.to_string(),
    }
}

/// Seed the generator from a `seed()` option.
fn apply_seed(me: &mut AdoInterpreter, opts: &Opts) -> Result<(), String> {
    if let Some(s) = arg(opts, "seed", 4) {
        let v: u64 = s.trim().parse().map_err(|_| rc_err(198, "seed() invalid"))?;
        me.engine.store.rng.borrow_mut().reseed(v);
    }
    Ok(())
}

fn uniform(me: &AdoInterpreter) -> f64 {
    me.engine.store.rng.borrow_mut().uniform()
}

/// A dataset with the variables of `shell` and the rows `rows` of `base`.
fn take_rows(shell: &DataSet, base: &DataSet, rows: &[usize]) -> DataSet {
    let mut d = shell.clone();
    for (v, src) in d.vars.iter_mut().zip(&base.vars) {
        v.data = rows.iter().map(|&r| src.data[r].clone()).collect();
    }
    d.n_obs = rows.len();
    d
}

fn empty_shell(base: &DataSet) -> DataSet {
    let mut shell = base.clone();
    for v in shell.vars.iter_mut() {
        v.data = Vec::new();
    }
    shell.n_obs = 0;
    shell.sort_by.clear();
    shell
}

/// Group key of each observation over `vars` (all equal when empty).
fn keys(ds: &DataSet, vars: &[String]) -> Vec<String> {
    (0..ds.n_obs)
        .map(|i| {
            vars.iter()
                .map(|v| {
                    let var = ds.get_var(v).unwrap();
                    if var.var_type.is_string() { var.str_at(i).to_string() } else { format!("{:?}", var.num(i)) }
                })
                .collect::<Vec<_>>()
                .join("\u{1}")
        })
        .collect()
}

/// Sampling units (observations or clusters) grouped by stratum.
struct Units {
    /// Rows of each unit.
    rows: Vec<Vec<usize>>,
    /// Units of each stratum.
    strata: Vec<Vec<usize>>,
}

fn units(ds: &DataSet, cluster: &[String], strata: &[String]) -> Units {
    let sk = keys(ds, strata);
    let ck = keys(ds, cluster);
    let mut strata_ix: HashMap<String, usize> = HashMap::new();
    let mut unit_ix: HashMap<(usize, String), usize> = HashMap::new();
    let mut u = Units { rows: Vec::new(), strata: Vec::new() };
    for i in 0..ds.n_obs {
        let h = *strata_ix.entry(sk[i].clone()).or_insert_with(|| {
            u.strata.push(Vec::new());
            u.strata.len() - 1
        });
        let key = if cluster.is_empty() { (h, i.to_string()) } else { (h, ck[i].clone()) };
        let id = match unit_ix.get(&key) {
            Some(&id) => id,
            None => {
                u.rows.push(Vec::new());
                let id = u.rows.len() - 1;
                unit_ix.insert(key, id);
                u.strata[h].push(id);
                id
            }
        };
        u.rows[id].push(i);
    }
    u
}

/// One bootstrap draw: rows of the sample and, per row, the id of the
/// drawn unit (1, 2, ... in draw order).
fn draw(me: &AdoInterpreter, u: &Units, size: Option<usize>) -> (Vec<usize>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut ids = Vec::new();
    let mut next = 0.0;
    for h in &u.strata {
        let m = h.len();
        let k = size.unwrap_or(m);
        for _ in 0..k {
            let j = ((uniform(me) * m as f64) as usize).min(m - 1);
            next += 1.0;
            for &r in &u.rows[h[j]] {
                rows.push(r);
                ids.push(next);
            }
        }
    }
    (rows, ids)
}

/// Results of the replications as a dataset (one observation per rep).
fn results_dataset(stats: &[Stat], reps: &[Vec<f64>], double: bool) -> Result<DataSet, String> {
    let mut ds = DataSet::new();
    let ty = if double { VarType::Double } else { VarType::Float };
    for (j, s) in stats.iter().enumerate() {
        let mut v = Variable::new(s.var.clone(), ty);
        v.label = s.label.clone();
        v.data = reps.iter().map(|r| util::typed_value(ty, r[j])).collect();
        ds.add_variable(v)?;
    }
    ds.n_obs = reps.len();
    Ok(ds)
}

/// `saving(filename[, replace double])`.
fn save_results(me: &mut AdoInterpreter, spec: &str, ds: DataSet) -> Result<(), String> {
    let (file, opts) = match find_top_level(spec, ',') {
        Some(i) => (spec[..i].trim(), split_options(&spec[i + 1..])),
        None => (spec.trim(), Vec::new()),
    };
    let file = unquote(file);
    let replace = has(&opts, "replace", 7);
    let held = std::mem::replace(&mut me.engine.dataset, ds);
    let r = run_cmd(me, &format!("save \"{file}\"{}", if replace { ", replace" } else { "" }), false);
    me.engine.dataset = held;
    r
}

fn saving_double(spec: &str) -> bool {
    find_top_level(spec, ',').is_some_and(|i| has(&split_options(&spec[i + 1..]), "double", 6))
}

// ---------------------------------------------------------------------------
// bootstrap
// ---------------------------------------------------------------------------

fn bootstrap(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    if find_top_level(rest, ':').is_none() {
        // replay
        let opts = rest.trim().trim_start_matches(',');
        let opts = split_options(opts);
        if me.engine.store.e.macros.get("prefix").map(String::as_str) != Some("bootstrap") {
            return Err(rc_err(301, "last estimates not found"));
        }
        let lvl = level_opt(me, &opts)?;
        let out = display(me, lvl, &opts)?;
        me.emit(&out);
        return Ok(());
    }
    let p = parse_prefix(rest, false)?;
    let opts = &p.opts;
    let reps = num_arg(opts, "reps", 1)?.unwrap_or(50.0);
    if reps < 1.0 {
        return Err(rc_err(198, "reps() must be a positive integer"));
    }
    let reps = reps as usize;
    let level = level_opt(me, opts)?;
    let nodots = has(opts, "nodots", 6);
    let noisily = has(opts, "noisily", 1);
    let mse = has(opts, "mse", 3);
    let want_bca = has(opts, "bca", 3);
    let size = num_arg(opts, "size", 4)?.map(|s| s as usize);
    let cluster = match arg(opts, "cluster", 2) {
        Some(c) => varlist::expand(&me.engine.dataset, &c)?,
        None => Vec::new(),
    };
    let strata = match arg(opts, "strata", 6) {
        Some(c) => varlist::expand(&me.engine.dataset, &c)?,
        None => Vec::new(),
    };
    let idcluster = arg(opts, "idcluster", 3).map(|s| s.trim().to_string());
    if idcluster.is_some() && cluster.is_empty() {
        return Err(rc_err(198, "idcluster() requires cluster()"));
    }
    apply_seed(me, opts)?;

    let cmd_verb = split_verb(&p.command).0;
    let cmd_name = me.engine.resolve_verb(&cmd_verb.to_lowercase());
    let orig = me.engine.dataset.clone();
    let orig_changed = me.engine.store.data_changed;
    let old_e = std::mem::take(&mut me.engine.store.e);

    // idcluster(): the observed run sees the cluster numbers
    let mut work = orig.clone();
    if let Some(id) = &idcluster {
        let u0 = units(&work, &cluster, &[]);
        let mut ids = vec![0.0; work.n_obs];
        for (k, rows) in u0.rows.iter().enumerate() {
            for &r in rows {
                ids[r] = (k + 1) as f64;
            }
        }
        let _ = work.drop_variable(id);
        let mut v = Variable::new(id.clone(), VarType::Long);
        v.data = ids.iter().map(|&x| util::typed_value(VarType::Long, x)).collect();
        work.add_variable(v)?;
        me.engine.dataset = work.clone();
    }

    // observed statistics
    if let Err(e) = run_cmd(me, &p.command, noisily) {
        me.engine.dataset = orig;
        me.engine.store.e = old_e;
        return Err(e);
    }
    let eclass = !me.engine.store.e.is_empty();
    let exps = if p.exps.is_empty() {
        if !eclass {
            me.engine.dataset = orig;
            me.engine.store.e = old_e;
            return Err(rc_err(198, "expression list required"));
        }
        vec![ExpItem { name: None, kind: Kind::B }]
    } else {
        p.exps.clone()
    };
    let default_b = eclass && exps.len() == 1 && matches!(exps[0].kind, Kind::B);
    let (stats, parsed) = match build_stats(me, &exps, "_bs") {
        Ok(x) => x,
        Err(e) => {
            me.engine.dataset = orig;
            me.engine.store.e = old_e;
            return Err(e);
        }
    };
    let observed = eval_stats(me, &stats, &parsed);
    let obs_e = me.engine.store.e.clone();

    // estimation sample
    let n0 = orig.n_obs;
    let sample: Vec<bool> = match &obs_e.sample {
        Some(s) if eclass && s.len() == n0 => s.clone(),
        _ => vec![true; n0],
    };
    let mut base = work;
    if sample.iter().any(|x| !x) {
        base.keep_if(&sample)?;
    }
    me.emit(&format!("(running {cmd_name} on estimation sample)"));
    let u = units(&base, &cluster, &strata);
    if size.is_some_and(|s| u.strata.iter().any(|h| s > h.len())) {
        me.engine.dataset = orig;
        me.engine.store.e = old_e;
        return Err(rc_err(498, "size() must not be greater than the number of sampling units"));
    }
    let shell = empty_shell(&base);
    let id_base = &base;

    let mut marks = Vec::with_capacity(reps);
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (rows, ids) = draw(me, &u, size);
        let mut d = take_rows(&shell, id_base, &rows);
        if let Some(id) = &idcluster {
            let var = d.get_var_mut(id).unwrap();
            var.var_type = VarType::Long;
            var.data = ids.iter().map(|&x| util::typed_value(VarType::Long, x)).collect();
        }
        me.engine.dataset = d;
        let vals = match run_cmd(me, &p.command, noisily) {
            Ok(()) => eval_stats(me, &stats, &parsed),
            Err(_) => vec![SYSMISS; stats.len()],
        };
        marks.push(if vals.iter().any(|x| is_missing(*x)) { 'x' } else { '.' });
        results.push(vals);
    }

    // jackknife for the acceleration of BCa intervals
    let accel = if want_bca {
        let mut jk: Vec<Vec<f64>> = Vec::new();
        for leave in 0..u.rows.len() {
            let rows: Vec<usize> = (0..u.rows.len()).filter(|&k| k != leave).flat_map(|k| u.rows[k].iter().copied()).collect();
            me.engine.dataset = take_rows(&shell, id_base, &rows);
            if let Ok(()) = run_cmd(me, &p.command, false) {
                jk.push(eval_stats(me, &stats, &parsed));
            }
        }
        Some(
            (0..stats.len())
                .map(|j| {
                    let v: Vec<f64> = jk.iter().map(|r| r[j]).filter(|x| !is_missing(*x)).collect();
                    let m = util::mean(&v);
                    let num: f64 = v.iter().map(|x| (m - x).powi(3)).sum();
                    let den: f64 = v.iter().map(|x| (m - x).powi(2)).sum::<f64>().powf(1.5);
                    if den > 0.0 { num / (6.0 * den) } else { SYSMISS }
                })
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    me.engine.dataset = orig;
    me.engine.store.data_changed = orig_changed;

    if !nodots {
        me.emit(&format!("\n{}", dots_line("Bootstrap replications", reps, &marks)));
    }

    if let Some(spec) = arg(opts, "saving", 3) {
        let ds = results_dataset(&stats, &results, saving_double(&spec))?;
        save_results(me, &spec, ds)?;
    }

    // summary statistics
    let k = stats.len();
    let complete: Vec<&Vec<f64>> = results.iter().filter(|r| r.iter().all(|x| !is_missing(*x))).collect();
    let nc = complete.len();
    let misreps = reps - nc;
    let mut v = Matrix::zeros(k, k);
    let mut mean = vec![SYSMISS; k];
    if nc > 0 {
        for j in 0..k {
            mean[j] = complete.iter().map(|r| r[j]).sum::<f64>() / nc as f64;
        }
    }
    if nc > 1 {
        for a in 0..k {
            for b in 0..k {
                let (ca, cb) = if mse { (observed[a], observed[b]) } else { (mean[a], mean[b]) };
                let s: f64 = complete.iter().map(|r| (r[a] - ca) * (r[b] - cb)).sum();
                v[(a, b)] = s / if mse { nc as f64 } else { nc as f64 - 1.0 };
            }
        }
    } else {
        for a in 0..k {
            v[(a, a)] = SYSMISS;
        }
    }
    let names: Vec<String> = stats.iter().map(|s| s.name.clone()).collect();
    let alpha = (1.0 - level / 100.0) / 2.0;
    let zc = dist::invnormal(1.0 - alpha);
    let mut ci_n = Matrix::zeros(2, k);
    let mut ci_p = Matrix::zeros(2, k);
    let mut ci_bc = Matrix::zeros(2, k);
    let mut ci_bca = Matrix::zeros(2, k);
    let mut z0v = vec![SYSMISS; k];
    let mut nreps = vec![0.0; k];
    for j in 0..k {
        let mut x: Vec<f64> = results.iter().map(|r| r[j]).filter(|x| !is_missing(*x)).collect();
        x.sort_by(|a, b| a.total_cmp(b));
        nreps[j] = x.len() as f64;
        let se = v[(j, j)].max(0.0).sqrt();
        ci_n[(0, j)] = observed[j] - zc * se;
        ci_n[(1, j)] = observed[j] + zc * se;
        if x.is_empty() {
            for m in [&mut ci_p, &mut ci_bc, &mut ci_bca] {
                m[(0, j)] = SYSMISS;
                m[(1, j)] = SYSMISS;
            }
            continue;
        }
        ci_p[(0, j)] = util::pctile_sorted(&x, 100.0 * alpha);
        ci_p[(1, j)] = util::pctile_sorted(&x, 100.0 * (1.0 - alpha));
        let below = x.iter().filter(|&&t| t <= observed[j]).count() as f64;
        let z0 = dist::invnormal(below / x.len() as f64);
        z0v[j] = if z0.is_finite() { z0 } else { SYSMISS };
        ci_bc[(0, j)] = util::pctile_sorted(&x, 100.0 * dist::normal(2.0 * z0 - zc));
        ci_bc[(1, j)] = util::pctile_sorted(&x, 100.0 * dist::normal(2.0 * z0 + zc));
        if let Some(acc) = &accel {
            let a = acc[j];
            let p1 = dist::normal(z0 + (z0 - zc) / (1.0 - a * (z0 - zc)));
            let p2 = dist::normal(z0 + (z0 + zc) / (1.0 - a * (z0 + zc)));
            ci_bca[(0, j)] = util::pctile_sorted(&x, 100.0 * p1);
            ci_bca[(1, j)] = util::pctile_sorted(&x, 100.0 * p2);
        }
    }

    // post e()
    let mut e = if default_b { obs_e.clone() } else { Results::new() };
    let rowv = |vals: &[f64]| Matrix::row_vector(vals).with_names(vec!["y1".into()], names.clone());
    let ci_names = vec!["ll".to_string(), "ul".to_string()];
    if default_b {
        e.scalars.remove("df_r");
        e.scalars.remove("F");
        e.set_str("cmdname", e.macros.get("cmd").cloned().unwrap_or_default());
    } else {
        e.set_str("cmd", "bootstrap");
        e.set_str("cmdname", cmd_name.clone());
        e.set_str("title", "Bootstrap results");
        e.set("N", if eclass { obs_e.get("N").unwrap_or(base.n_obs as f64) } else { base.n_obs as f64 });
        e.set("k_eq", 1.0);
        e.sample = Some(sample.clone());
    }
    e.set_str("prefix", "bootstrap");
    e.set_str("command", p.command.clone());
    e.set_str("vce", "bootstrap");
    e.set_str("vcetype", "Bootstrap");
    e.set_str("properties", "b V");
    e.set_mat("b", rowv(&observed));
    e.set_mat("V", v.clone().with_names(names.clone(), names.clone()));
    e.set_mat("b_bs", rowv(&mean));
    e.set_mat("bias", rowv(&mean.iter().zip(&observed).map(|(m, o)| if is_missing(*m) { SYSMISS } else { m - o }).collect::<Vec<_>>()));
    e.set_mat("se", rowv(&(0..k).map(|j| if is_missing(v[(j, j)]) { SYSMISS } else { v[(j, j)].max(0.0).sqrt() }).collect::<Vec<_>>()));
    e.set_mat("reps", rowv(&nreps));
    e.set_mat("z0", rowv(&z0v));
    e.set_mat("ci_normal", ci_n.with_names(ci_names.clone(), names.clone()));
    e.set_mat("ci_percentile", ci_p.with_names(ci_names.clone(), names.clone()));
    e.set_mat("ci_bc", ci_bc.with_names(ci_names.clone(), names.clone()));
    if let Some(acc) = &accel {
        e.set_mat("ci_bca", ci_bca.with_names(ci_names.clone(), names.clone()));
        e.set_mat("accel", rowv(acc));
    }
    e.set("N_reps", reps as f64);
    e.set("N_misreps", misreps as f64);
    e.set("k_exp", if default_b { 0.0 } else { k as f64 });
    e.set("k_eexp", if default_b { k as f64 } else { 0.0 });
    e.set("level", level);
    if mse {
        e.set_str("mse", "mse");
    }
    if !default_b {
        for (j, s) in stats.iter().enumerate() {
            e.set_str(&format!("exp{}", j + 1), s.label.clone());
        }
    }
    if !cluster.is_empty() {
        e.set("N_clust", u.rows.len() as f64);
        e.set_str("clustvar", cluster.join(" "));
    }
    if !strata.is_empty() {
        e.set("N_strata", u.strata.len() as f64);
        e.set_str("strata", strata.join(" "));
    }
    if size.is_some() {
        e.set("size", size.unwrap() as f64);
    }
    if default_b {
        if let Some((chi2, df)) = wald(&names, &observed, &v) {
            e.set("chi2", chi2);
            e.set("df_m", df as f64);
            e.set_str("chi2type", "Wald");
        }
    }
    me.engine.store.e = e;

    if !has(opts, "notable", 5) || !has(opts, "noheader", 5) {
        let out = display(me, level, opts)?;
        me.emit(&format!("\n{out}"));
    }
    Ok(())
}

fn level_opt(me: &AdoInterpreter, opts: &Opts) -> Result<f64, String> {
    let l = num_arg(opts, "level", 1)?.unwrap_or(me.engine.store.level);
    if !(10.0..=99.99).contains(&l) {
        return Err(rc_err(198, "level() must be between 10 and 99.99 inclusive"));
    }
    Ok(l)
}

fn is_base_name(c: &str) -> bool {
    c.rsplit(':').next().unwrap_or(c).split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))
}

fn is_omitted_name(c: &str) -> bool {
    c.rsplit(':').next().unwrap_or(c).split('#').any(|p| p.starts_with("o.") || p.split_once('.').is_some_and(|(l, _)| l.ends_with('o') && l.trim_end_matches(['o', 'b']).parse::<f64>().is_ok()))
}

/// Wald test that all coefficients except constants are zero.
fn wald(names: &[String], b: &[f64], v: &Matrix) -> Option<(f64, usize)> {
    if names.iter().any(|n| n.contains(':')) {
        return None;
    }
    let idx: Vec<usize> = (0..names.len()).filter(|&j| names[j] != "_cons" && !is_base_name(&names[j]) && !is_omitted_name(&names[j]) && v[(j, j)] > 0.0 && !is_missing(v[(j, j)])).collect();
    if idx.is_empty() {
        return None;
    }
    let q = idx.len();
    let mut sub = Matrix::zeros(q, q);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            sub[(a, c)] = v[(i, j)];
        }
    }
    let inv = sub.inverse().ok()?;
    let mut chi2 = 0.0;
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            chi2 += b[i] * inv[(a, c)] * b[j];
        }
    }
    Some((chi2, q))
}

fn title_for(e: &Results) -> String {
    let cmd = e.macros.get("cmd").map(String::as_str).unwrap_or("");
    match cmd {
        "regress" => "Linear regression".into(),
        "logit" | "logistic" => "Logistic regression".into(),
        "probit" => "Probit regression".into(),
        "poisson" => "Poisson regression".into(),
        "nbreg" => "Negative binomial regression".into(),
        "tobit" => "Tobit regression".into(),
        _ => e.macros.get("title").cloned().unwrap_or_else(|| "Bootstrap results".into()),
    }
}

/// Header with statistics right-aligned at column 78.
fn header_lines(title: &str, rows: &[(String, String, String)]) -> Vec<String> {
    let w = rows.iter().map(|r| r.2.len()).max().unwrap_or(0);
    let lab = rows.iter().map(|r| r.1.len()).max().unwrap_or(0).max(13);
    let start = 78usize.saturating_sub(lab + 3 + w);
    rows.iter()
        .enumerate()
        .map(|(i, (left, label, val))| {
            let left = if i == 0 { title } else { left.as_str() };
            let left = if left.len() >= start { format!("{left} ") } else { left.to_string() };
            format!("{:<start$}{:<lab$} = {:>w$}", left, label, val)
        })
        .collect()
}

fn eform_title(e: &Results) -> Option<&'static str> {
    let cmd = e.macros.get("cmd").map(String::as_str).unwrap_or("");
    let command = e.macros.get("command").cloned().unwrap_or_default();
    let opts = find_top_level(&command, ',').map(|i| split_options(&command[i + 1..])).unwrap_or_default();
    if cmd == "logistic" || (cmd == "logit" && has(&opts, "or", 2)) {
        return Some("Odds ratio");
    }
    if has(&opts, "irr", 3) {
        return Some("IRR");
    }
    None
}

fn header_rows(e: &Results) -> Vec<(String, String, String)> {
    let n = e.get("N").unwrap_or(SYSMISS);
    let mut rows = vec![
        (String::new(), "Number of obs".to_string(), if is_missing(n) { ".".into() } else { util::commas(n as usize) }),
        (String::new(), "Replications".to_string(), util::commas(e.get("N_reps").unwrap_or(0.0) as usize)),
    ];
    let cmd = e.macros.get("cmd").map(String::as_str).unwrap_or("");
    if cmd != "bootstrap" {
        if let (Some(c), Some(df)) = (e.get("chi2"), e.get("df_m")) {
            rows.push((String::new(), format!("Wald chi2({})", df as usize), format!("{:.2}", c)));
            rows.push((String::new(), "Prob > chi2".into(), format!("{:.4}", dist::chi2tail(df, c))));
        }
        if let Some(r2) = e.get("r2") {
            rows.push((String::new(), "R-squared".into(), format!("{:.4}", r2)));
        }
        if let Some(r2) = e.get("r2_a") {
            rows.push((String::new(), "Adj R-squared".into(), format!("{:.4}", r2)));
        }
        if let Some(r) = e.get("rmse") {
            rows.push((String::new(), "Root MSE".into(), ferrostat_cmd::format::fmt(r, "%10.5g").trim().to_string()));
        }
        if let Some(r2p) = e.get("r2_p") {
            let left = e.get("ll").map(|ll| format!("Log likelihood = {}", fmt_g(ll, 10))).unwrap_or_default();
            rows.push((left, "Pseudo R2".into(), format!("{:.4}", r2p)));
        }
    }
    rows
}

/// The bootstrap results table built from e().
fn display(me: &mut AdoInterpreter, level: f64, opts: &Opts) -> Result<String, String> {
    let e = me.engine.store.e.clone();
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = b.colnames.clone();
    let mut out = Vec::new();
    if !has(opts, "noheader", 5) {
        out.extend(header_lines(&title_for(&e), &header_rows(&e)));
        if let (Some(g), Some(cv)) = (e.get("N_clust"), e.macros.get("clustvar")) {
            out.push(String::new());
            out.push(format!("{:>78}", format!("(Replications based on {} clusters in {})", util::commas(g as usize), cv)));
        }
        if e.macros.get("cmd").map(String::as_str) == Some("bootstrap") && !has(opts, "nolegend", 5) {
            out.push(String::new());
            out.push(format!("{:>13}: {}", "Command", e.macros.get("command").cloned().unwrap_or_default()));
            for (j, n) in names.iter().enumerate() {
                if let Some(x) = e.macros.get(&format!("exp{}", j + 1)) {
                    out.push(format!("{:>13}: {}", summ_abbrev(n, 13), x));
                }
            }
        }
        out.push(String::new());
    }
    if !has(opts, "notable", 5) {
        let k = names.len();
        let is_base: Vec<bool> = names.iter().map(|c| is_base_name(c)).collect();
        let omitted: Vec<bool> = names.iter().map(|c| is_omitted_name(c)).collect();
        let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
        let eform = eform_title(&e);
        let spec = TableSpec { depvar: &depvar, names: &names, b: &b.data[..k], v, omitted: &omitted, is_base: &is_base, df: None, level, eform, coef_title: "coefficient" };
        let mut t = est::coef_table(&spec);
        if t.len() > 1 {
            let bar = t[1].find('|').unwrap_or(13);
            let pad = " ".repeat(bar);
            let first = "  Observed";
            let head = t[1].replacen("Std. err.", "std. err.", 1);
            let head = match eform {
                Some(_) => head,
                None => head.replacen("Coefficient", "coefficient", 1),
            };
            t[1] = head;
            t.insert(1, format!("{pad}| {first:>10}   Bootstrap                          Normal-based"));
        }
        out.extend(t);
        let mis = e.get("N_misreps").unwrap_or(0.0) as usize;
        if mis > 0 {
            out.push(format!("Note: One or more parameters could not be estimated in {} bootstrap replicate{};", word_count(mis).to_lowercase(), if mis == 1 { "" } else { "s" }));
            out.push("      standard-error estimates include only complete replications.".into());
        }
        if let Some((m, lvl)) = est::take_table() {
            let mut r = Results::new();
            r.set_mat("table", m);
            r.set("level", lvl);
            me.engine.store.r = r;
        }
    }
    Ok(out.join("\n"))
}

fn summ_abbrev(s: &str, n: usize) -> String {
    ferrostat_cmd::summarize::abbrev(s, n)
}

/// `estat bootstrap [, normal percentile bc bca all noheader nolegend]`.
fn estat_bootstrap(me: &mut AdoInterpreter, opts: &str) -> Result<String, String> {
    let e = me.engine.store.e.clone();
    if e.macros.get("prefix").map(String::as_str) != Some("bootstrap") {
        return Err(rc_err(321, "estat bootstrap requires bootstrapped results"));
    }
    let opts = split_options(opts);
    let all = has(&opts, "all", 3);
    let mut kinds: Vec<(&str, &str)> = Vec::new();
    let has_bca = e.matrices.contains_key("ci_bca");
    if all || has(&opts, "normal", 1) {
        kinds.push(("N", "ci_normal"));
    }
    if all || has(&opts, "percentile", 1) {
        kinds.push(("P", "ci_percentile"));
    }
    if all || has(&opts, "bc", 2) {
        kinds.push(("BC", "ci_bc"));
    }
    if (all && has_bca) || has(&opts, "bca", 3) {
        if !has_bca {
            return Err(rc_err(198, "bca intervals require the bca option of bootstrap"));
        }
        kinds.push(("BCa", "ci_bca"));
    }
    if kinds.is_empty() {
        kinds.push(("BC", "ci_bc"));
    }
    let level = e.get("level").unwrap_or(95.0);
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names = b.colnames.clone();
    let mut out = header_lines(&title_for(&e), &header_rows(&e)[..2]);
    if e.macros.get("cmd").map(String::as_str) == Some("bootstrap") && !has(&opts, "nolegend", 5) {
        out.push(String::new());
        out.push(format!("{:>13}: {}", "Command", e.macros.get("command").cloned().unwrap_or_default()));
        for (j, n) in names.iter().enumerate() {
            if let Some(x) = e.macros.get(&format!("exp{}", j + 1)) {
                out.push(format!("{:>13}: {}", summ_abbrev(n, 13), x));
            }
        }
    }
    out.push(String::new());
    let lw = 12usize;
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    out.push("-".repeat(78));
    out.push(format!("{:>lw$} | {:>11}{:>24}", "", "Observed", "Bootstrap"));
    out.push(format!("{:>lw$} | {:>11}{:>12}{:>12}  [{}% conf. interval]", summ_abbrev(&dep, lw), "coefficient", "Bias", "std. err.", fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
    let get = |m: &str, i: usize, j: usize| e.matrices.get(m).map_or(SYSMISS, |x| x.get(i, j));
    let cell = |x: f64, w: usize| if is_missing(x) { ".".to_string() } else { fmt_g(x, w) };
    for (j, n) in names.iter().enumerate() {
        if is_base_name(n) || is_omitted_name(n) {
            continue;
        }
        let label = summ_abbrev(n.rsplit(':').next().unwrap_or(n), lw);
        for (r, (key, m)) in kinds.iter().enumerate() {
            let ci = format!("{:>12}{:>11}{:>6}", cell(get(m, 0, j), 9), cell(get(m, 1, j), 9), format!("({key})"));
            if r == 0 {
                out.push(format!("{:>lw$} | {:>11}{:>12}{:>12}{}", label, cell(b.get(0, j), 9), cell(get("bias", 0, j), 9), cell(get("se", 0, j), 9), ci));
            } else {
                out.push(format!("{:>lw$} | {:>11}{:>12}{:>12}{}", "", "", "", "", ci));
            }
        }
    }
    out.push("-".repeat(78));
    let key_text = |k: &str| match k {
        "N" => "Normal",
        "P" => "Percentile",
        "BC" => "Bias-corrected",
        _ => "Bias-corrected and accelerated",
    };
    for (i, (k, _)) in kinds.iter().enumerate() {
        let line = format!("{:>7}: {}", k, key_text(k));
        out.push(if i == 0 { format!("Key:{}", &line[4..]) } else { line });
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// simulate
// ---------------------------------------------------------------------------

fn simulate(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let p = parse_prefix(rest, false)?;
    let opts = &p.opts;
    let reps = num_arg(opts, "reps", 1)?.ok_or_else(|| rc_err(198, "option reps() required"))?;
    if reps < 1.0 {
        return Err(rc_err(198, "reps() must be a positive integer"));
    }
    let reps = reps as usize;
    let nodots = has(opts, "nodots", 6);
    let noisily = has(opts, "noisily", 1);
    let double = has(opts, "double", 6);
    let saving = arg(opts, "saving", 3);
    apply_seed(me, opts)?;
    let mut legend = vec![format!("{:>13}: {}", "Command", p.command)];
    for (k, it) in p.exps.iter().enumerate() {
        let name = it.name.clone().unwrap_or_else(|| format!("_sim_{}", k + 1));
        let text = match &it.kind {
            Kind::B => "_b".to_string(),
            Kind::Se => "_se".to_string(),
            Kind::Expr(t) => t.clone(),
        };
        if matches!(it.kind, Kind::Expr(_)) {
            legend.push(format!("{:>13}: {}", summ_abbrev(&name, 13), text));
        }
    }
    let orig = me.engine.dataset.clone();
    let mut stats: Option<(Vec<Stat>, Vec<Expr>)> = None;
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(reps);
    let mut marks = Vec::with_capacity(reps);
    for rep in 0..reps {
        let r = run_cmd(me, &p.command, noisily);
        if let Err(e) = r {
            if rep == 0 {
                me.engine.dataset = orig;
                return Err(e);
            }
            marks.push('x');
            results.push(Vec::new());
            continue;
        }
        if stats.is_none() {
            let exps = if p.exps.is_empty() {
                if me.engine.store.e.matrices.get("b").is_none() {
                    me.engine.dataset = orig;
                    return Err(rc_err(198, "expression list required"));
                }
                vec![ExpItem { name: None, kind: Kind::B }]
            } else {
                p.exps.clone()
            };
            match build_stats(me, &exps, "_sim") {
                Ok(s) => stats = Some(s),
                Err(e) => {
                    me.engine.dataset = orig;
                    return Err(e);
                }
            }
        }
        let (s, parsed) = stats.as_ref().unwrap();
        let vals = eval_stats(me, s, parsed);
        marks.push(if vals.iter().any(|x| is_missing(*x)) { 'x' } else { '.' });
        results.push(vals);
    }
    let (s, _) = stats.unwrap();
    for r in results.iter_mut() {
        if r.is_empty() {
            *r = vec![SYSMISS; s.len()];
        }
    }
    me.emit(&format!("\n{}", legend.join("\n")));
    if !nodots {
        me.emit(&format!("\n{}", dots_line("Simulations", reps, &marks)));
    }
    let ds = results_dataset(&s, &results, double || saving.as_deref().is_some_and(saving_double))?;
    match saving {
        Some(spec) => {
            save_results(me, &spec, ds)?;
            me.engine.dataset = orig;
        }
        None => {
            me.engine.dataset = ds;
            me.engine.store.data_changed = true;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// permute
// ---------------------------------------------------------------------------

fn permute(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let p = parse_prefix(rest, true)?;
    let opts = &p.opts;
    let permvar = varlist::resolve_name(&me.engine.dataset, p.lead.as_deref().unwrap_or(""))?;
    let reps = num_arg(opts, "reps", 1)?.unwrap_or(100.0);
    if reps < 1.0 {
        return Err(rc_err(198, "reps() must be a positive integer"));
    }
    let reps = reps as usize;
    let level = level_opt(me, opts)?;
    let nodots = has(opts, "nodots", 6);
    let noisily = has(opts, "noisily", 1);
    let (left, right) = (has(opts, "left", 4), has(opts, "right", 5));
    let strata = match arg(opts, "strata", 6) {
        Some(c) => varlist::expand(&me.engine.dataset, &c)?,
        None => Vec::new(),
    };
    apply_seed(me, opts)?;
    let orig = me.engine.dataset.clone();
    let orig_changed = me.engine.store.data_changed;
    if let Err(e) = run_cmd(me, &p.command, noisily) {
        me.engine.dataset = orig;
        return Err(e);
    }
    let exps = if p.exps.is_empty() {
        if me.engine.store.e.matrices.get("b").is_none() {
            me.engine.dataset = orig;
            return Err(rc_err(198, "expression list required"));
        }
        vec![ExpItem { name: None, kind: Kind::B }]
    } else {
        p.exps.clone()
    };
    let (stats, parsed) = match build_stats(me, &exps, "_pm") {
        Ok(x) => x,
        Err(e) => {
            me.engine.dataset = orig;
            return Err(e);
        }
    };
    let observed = eval_stats(me, &stats, &parsed);
    let pverb = me.engine.resolve_verb(&split_verb(&p.command).0.to_lowercase());
    me.emit(&format!("(running {pverb} on estimation sample)"));
    let nobs = me.engine.store.e.get("N").filter(|_| matches!(exps[0].kind, Kind::B)).unwrap_or(orig.n_obs as f64);
    let u = units(&orig, &[], &strata);
    let col = orig.get_var(&permvar).unwrap().data.clone();
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(reps);
    let mut marks = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut perm: Vec<usize> = (0..orig.n_obs).collect();
        for h in &u.strata {
            // Fisher-Yates shuffle of the rows of this stratum
            let rows: Vec<usize> = h.iter().map(|&k| u.rows[k][0]).collect();
            let mut sh = rows.clone();
            for i in (1..sh.len()).rev() {
                let j = ((uniform(me) * (i + 1) as f64) as usize).min(i);
                sh.swap(i, j);
            }
            for (a, b) in rows.iter().zip(&sh) {
                perm[*a] = *b;
            }
        }
        let mut d = orig.clone();
        d.get_var_mut(&permvar).unwrap().data = perm.iter().map(|&i| col[i].clone()).collect();
        me.engine.dataset = d;
        let vals = match run_cmd(me, &p.command, noisily) {
            Ok(()) => eval_stats(me, &stats, &parsed),
            Err(_) => vec![SYSMISS; stats.len()],
        };
        marks.push(if vals.iter().any(|x| is_missing(*x)) { 'x' } else { '.' });
        results.push(vals);
    }
    me.engine.dataset = orig;
    me.engine.store.data_changed = orig_changed;
    if !nodots {
        me.emit(&format!("\n{}", dots_line("Permutation replications", reps, &marks)));
    }
    if let Some(spec) = arg(opts, "saving", 3) {
        let ds = results_dataset(&stats, &results, saving_double(&spec))?;
        save_results(me, &spec, ds)?;
    }
    let k = stats.len();
    let alpha = (1.0 - level / 100.0) / 2.0;
    let mut c = vec![0.0; k];
    let mut n = vec![0.0; k];
    let mut pv = vec![0.0; k];
    let mut se = vec![0.0; k];
    let mut ci = Matrix::zeros(2, k);
    for j in 0..k {
        let t0 = observed[j];
        for r in &results {
            let t = r[j];
            if is_missing(t) {
                continue;
            }
            n[j] += 1.0;
            let hit = if left {
                t <= t0
            } else if right {
                t >= t0
            } else {
                t.abs() >= t0.abs()
            };
            if hit {
                c[j] += 1.0;
            }
        }
        let (cc, nn) = (c[j], n[j]);
        pv[j] = if nn > 0.0 { cc / nn } else { SYSMISS };
        se[j] = if nn > 0.0 { (pv[j] * (1.0 - pv[j]) / nn).sqrt() } else { SYSMISS };
        ci[(0, j)] = if cc <= 0.0 { 0.0 } else { dist::invibeta(cc, nn - cc + 1.0, alpha) };
        ci[(1, j)] = if cc >= nn { 1.0 } else { dist::invibeta(cc + 1.0, nn - cc, 1.0 - alpha) };
    }
    let names: Vec<String> = stats.iter().map(|s| s.name.clone()).collect();
    let rowv = |vals: &[f64]| Matrix::row_vector(vals).with_names(vec!["y1".into()], names.clone());
    let mut r = Results::new();
    r.set("N", nobs);
    r.set("N_reps", reps as f64);
    r.set("level", level);
    r.set("k_exp", k as f64);
    r.set_str("command", p.command.clone());
    r.set_str("permvar", permvar.clone());
    r.set_str("cmd", "permute");
    r.set_mat("b", rowv(&observed));
    r.set_mat("c", rowv(&c));
    r.set_mat("reps", rowv(&n));
    r.set_mat("p", rowv(&pv));
    r.set_mat("se", rowv(&se));
    r.set_mat("ci", ci.clone().with_names(vec!["ll".into(), "ul".into()], names.clone()));
    me.engine.store.r = r;

    let mut out = vec![String::new()];
    let mut h = header_lines("Monte Carlo permutation results", &[(String::new(), "Number of obs".into(), util::commas(nobs as usize))]);
    out.append(&mut h);
    out.push(String::new());
    out.push(format!("{:>13}: {}", "Command", p.command));
    for s in &stats {
        out.push(format!("{:>13}: {}", summ_abbrev(&s.name, 13), s.label));
    }
    out.push(format!("{:>13}: {}", "Permute var", permvar));
    out.push(String::new());
    out.push("-".repeat(78));
    out.push(format!("{:<12} |     T(obs)       c       n   p=c/n   SE(p) [{}% conf. interval]", "T", fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    for j in 0..k {
        out.push(format!(
            "{:>12} | {:>10}{:>8}{:>8}{:>8}{:>8}{:>10}{:>11}",
            summ_abbrev(&names[j], 12),
            fmt_g(observed[j], 9),
            c[j] as usize,
            n[j] as usize,
            format!("{:.4}", pv[j]),
            format!("{:.4}", se[j]),
            fmt_g(ci[(0, j)], 9),
            fmt_g(ci[(1, j)], 9)
        ));
    }
    out.push("-".repeat(78));
    out.push("Note: Confidence interval is with respect to p=c/n.".into());
    out.push(
        if left {
            "Note: c = #{T <= T(obs)}"
        } else if right {
            "Note: c = #{T >= T(obs)}"
        } else {
            "Note: c = #{|T| >= |T(obs)|}"
        }
        .into(),
    );
    me.emit(&out.join("\n"));
    Ok(())
}
