//! Further mi commands, layered on the `MiData` model of `mi.rs`:
//!
//! * the flongsep style (m = 0 in memory, imputation k in `_k_name.dta`),
//!   `mi copy`, `mi erase`;
//! * commands run on every imputation and recombined: `mi xtset|tsset|
//!   stset|streset|st|svyset|fvset`, `mi stsplit|stjoin`, `mi expand`,
//!   `mi reshape`, `mi merge`, `mi append`;
//! * `mi add`, `mi replace0`, `mi select`;
//! * `mi import flong|flongsep|ice|nhanes1|wide`, `mi export ice|nhanes1`;
//! * `mi estimate, saving()` / `mi estimate using`, `mi predict`,
//!   `mi predictnl`;
//! * `mi impute intreg|nbreg|truncreg`, `mi ptrace describe|use`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use ferrostat_cmd::dist;
use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::rng::Rng;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options};

use crate::interpreter::AdoInterpreter;
use crate::mi::{col, parse_numlist_m, save, set_num_col, take_rows, MiData};
use crate::resample::run_cmd;

type Opts = Vec<(String, Option<String>)>;

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str()))
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opts.iter().find(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone())
}

/// `main , options`
fn split_args(args: &str) -> (String, Opts) {
    let a = args.trim();
    match find_top_level(a, ',') {
        Some(i) => (a[..i].trim().to_string(), split_options(&a[i + 1..])),
        None => (a.to_string(), Vec::new()),
    }
}

fn with_ext(f: &str, ext: &str) -> String {
    let f = util::unquote(f.trim());
    let last = f.rsplit(['/', '\\']).next().unwrap_or(&f);
    if last.contains('.') { f } else { format!("{f}{ext}") }
}

fn read_dta(path: &str) -> Result<DataSet, String> {
    ferrostat_core::dta::DtaReader::open(path).map_err(|e| rc_err(601, format!("file {path} not found or not readable ({e})")))
}

fn write_dta(ds: &DataSet, path: &str, replace: bool) -> Result<(), String> {
    if !replace && ferrostat_core::vfs::exists(path) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let mut buf = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut buf, ds).map_err(|e| rc_err(693, e.to_string()))?;
    ferrostat_core::vfs::write(path, &buf).map_err(|e| rc_err(603, e.to_string()))
}

fn strip_mi_chars(ds: &mut DataSet) {
    if let Some(c) = ds.characteristics.get_mut("_dta") {
        c.retain(|k, _| !k.starts_with("_mi_"));
    }
}

fn load(me: &AdoInterpreter) -> Result<MiData, String> {
    if !MiData::is_set(&me.engine.dataset) {
        return Err(rc_err(119, "no; data not mi set"));
    }
    MiData::load(&me.engine.dataset)
}

fn set_data(me: &mut AdoInterpreter, ds: DataSet) {
    me.engine.dataset = ds;
    me.engine.store.data_changed = true;
}

/// Is `opts` asking to discard the data in memory (or is there nothing to lose)?
fn check_clear(me: &AdoInterpreter, opts: &Opts) -> Result<(), String> {
    if has(opts, "clear", 5) || me.engine.dataset.n_vars() == 0 || !me.engine.store.data_changed {
        return Ok(());
    }
    Err(rc_err(4, "no; dataset in memory has changed since last saved\n    specify option clear"))
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/// The mi subcommands handled here; None for the others.
pub(crate) fn command(me: &mut AdoInterpreter, sub: &str, args: &str) -> Option<Result<(), String>> {
    let first = args.split(|c: char| c.is_whitespace() || c == ',').find(|s| !s.is_empty()).unwrap_or("").to_lowercase();
    Some(match sub {
        "set" if first == "flongsep" => set_flongsep(me, args, false),
        "convert" | "conv" | "conve" | "conver" if first == "flongsep" => set_flongsep(me, args, true),
        "streset" | "st" => mi_streset(me, sub, args),
        "xtset" | "tsset" | "stset" | "svyset" | "fvset" => {
            let c = format!("{sub} {}", args.trim());
            each_simple(me, &c, true, false)
        }
        "stsplit" | "stjoin" | "expand" => {
            let c = format!("{sub} {}", args.trim());
            each_simple(me, &c, true, false)
        }
        "reshape" => {
            let c = format!("reshape {}", args.trim());
            each_simple(me, &c, true, true)
        }
        "merge" => merge_append(me, args, true),
        "append" => merge_append(me, args, false),
        "add" => mi_add(me, args),
        "copy" => mi_copy(me, args),
        "erase" => mi_erase(args),
        "export" => mi_export(me, args),
        "import" => mi_import(me, args),
        "predict" => mi_predict(me, args, false),
        "predictnl" => mi_predict(me, args, true),
        "ptrace" => mi_ptrace(me, args),
        "select" => mi_select(me, args),
        "replace0" => mi_replace0(me, args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// flongsep
// ---------------------------------------------------------------------------

fn sep_file(k: usize, name: &str) -> String {
    format!("_{k}_{name}.dta")
}

/// Base data of flongsep data in memory; imputed and passive columns are
/// read from the `_k_name.dta` files and aligned on `_mi_id`.
pub(crate) fn flongsep_load(ds: &DataSet, ipv: &[String], m: usize, imp: &mut [BTreeMap<String, Vec<f64>>]) -> Result<DataSet, String> {
    let name = ds.characteristics.get("_dta").and_then(|c| c.get("_mi_name")).cloned().unwrap_or_default();
    let mut base = ds.clone();
    let ids = col(ds, "_mi_id");
    let _ = base.drop_variable("_mi_id");
    let _ = base.drop_variable("_mi_miss");
    let pos: HashMap<i64, usize> = ids.iter().enumerate().map(|(i, v)| (*v as i64, i)).collect();
    for k in 1..=m {
        let f = sep_file(k, &name);
        let d = read_dta(&f)?;
        let did = col(&d, "_mi_id");
        for v in ipv {
            let src = col(&d, v);
            let mut out = col(&base, v);
            for (r, idv) in did.iter().enumerate() {
                if let Some(&t) = pos.get(&(*idv as i64)) {
                    out[t] = src[r];
                }
            }
            imp[k - 1].insert(v.clone(), out);
        }
    }
    Ok(base)
}

/// The m = 0 data for memory; writes imputation files for m = 1..M.
pub(crate) fn flongsep_store(md: &MiData, miss: &[f64]) -> Result<DataSet, String> {
    if md.name.is_empty() {
        return Err(rc_err(198, "flongsep data require a name"));
    }
    let n = md.base.n_obs;
    let ids: Vec<f64> = (1..=n).map(|x| x as f64).collect();
    let add_sys = |ds: &mut DataSet, with_miss: bool| -> Result<(), String> {
        let _ = ds.drop_variable("_mi_id");
        let mut v = Variable::new("_mi_id", VarType::Long);
        v.label = "Observation ID".into();
        v.data = ids.iter().map(|&x| util::typed_value(VarType::Long, x)).collect();
        ds.add_variable(v)?;
        if with_miss {
            let _ = ds.drop_variable("_mi_miss");
            let mut v = Variable::new("_mi_miss", VarType::Byte);
            v.label = "Incomplete observation".into();
            v.data = miss.iter().map(|&x| util::typed_value(VarType::Byte, x)).collect();
            ds.add_variable(v)?;
        }
        Ok(())
    };
    for k in 1..=md.m {
        let mut d = md.extract(k);
        add_sys(&mut d, false)?;
        let c = d.characteristics.entry("_dta".into()).or_default();
        c.insert("_mi_style".into(), "flongsep_sub".into());
        c.insert("_mi_name".into(), md.name.clone());
        c.insert("_mi_m".into(), k.to_string());
        write_dta(&d, &sep_file(k, &md.name), true)?;
    }
    // remove files of imputations no longer present
    let mut k = md.m + 1;
    while ferrostat_core::vfs::exists(&sep_file(k, &md.name)) {
        let _ = ferrostat_core::vfs::remove(&sep_file(k, &md.name));
        k += 1;
    }
    let mut ds = md.base.clone();
    add_sys(&mut ds, true)?;
    Ok(ds)
}

fn set_flongsep(me: &mut AdoInterpreter, args: &str, convert: bool) -> Result<(), String> {
    let (main, _opts) = split_args(args);
    let w: Vec<&str> = main.split_whitespace().collect();
    let name = w.get(1).ok_or_else(|| rc_err(198, "flongsep name required"))?.to_string();
    let mut md = if convert {
        load(me)?
    } else {
        if MiData::is_set(&me.engine.dataset) {
            return Err(rc_err(119, "data already mi set"));
        }
        MiData::new(me.engine.dataset.clone(), "flongsep")
    };
    md.style = "flongsep".into();
    md.name = name;
    save(me, &md)
}

fn mi_copy(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (main, opts) = split_args(args);
    let name = main.split_whitespace().next().ok_or_else(|| rc_err(198, "new name required"))?.to_string();
    let mut md = load(me)?;
    let path = format!("{name}.dta");
    if ferrostat_core::vfs::exists(&path) && !has(&opts, "replace", 7) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    md.style = "flongsep".into();
    md.name = name;
    save(me, &md)?;
    write_dta(&me.engine.dataset, &path, true)?;
    me.engine.store.data_changed = false;
    Ok(())
}

fn mi_erase(args: &str) -> Result<(), String> {
    let (main, _) = split_args(args);
    let name = main.split_whitespace().next().ok_or_else(|| rc_err(198, "name required"))?.trim_end_matches(".dta").to_string();
    let mut removed = 0;
    for f in std::iter::once(format!("{name}.dta")).chain((1..).map(|k| sep_file(k, &name)).take_while(|f| ferrostat_core::vfs::exists(f))) {
        if ferrostat_core::vfs::exists(&f) {
            ferrostat_core::vfs::remove(&f).map_err(|e| rc_err(693, e.to_string()))?;
            removed += 1;
        }
    }
    if removed == 0 {
        return Err(rc_err(601, format!("file {name}.dta not found")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// running a command on every imputation
// ---------------------------------------------------------------------------

/// Run `cmd(k)` on the completed data of m = 0..M; returns the results.
fn run_each(me: &mut AdoInterpreter, md: &MiData, cmd: &dyn Fn(usize) -> String, show_first: bool) -> Result<Vec<DataSet>, String> {
    let saved = me.engine.dataset.clone();
    let saved_changed = me.engine.store.data_changed;
    let mut outs = Vec::new();
    for k in 0..=md.m {
        me.engine.dataset = md.extract(k);
        let c = cmd(k);
        if let Err(e) = run_cmd(me, &c, show_first && k == 0) {
            me.engine.dataset = saved;
            me.engine.store.data_changed = saved_changed;
            return Err(rc_err(crate::interpreter::rc_of(&e), format!("{}\nan error occurred when mi executed {} on m={k}", crate::interpreter::err_message(&e), c.split_whitespace().next().unwrap_or(""))));
        }
        outs.push(std::mem::take(&mut me.engine.dataset));
    }
    me.engine.dataset = saved;
    me.engine.store.data_changed = saved_changed;
    Ok(outs)
}

/// mi data from per-imputation results.  Registered variables keep their
/// registration; new variables that vary across imputations become passive
/// (imputed when `new_imputed`, as after reshape).
fn rebuild(md: &MiData, outs: Vec<DataSet>, new_imputed: bool, extra_ivars: &[String]) -> Result<MiData, String> {
    let n = outs[0].n_obs;
    if outs.iter().any(|d| d.n_obs != n) {
        return Err(rc_err(459, "the number of observations differs across imputations; imputed variables may not be used in this way"));
    }
    let mut base = outs[0].clone();
    strip_mi_chars(&mut base);
    let old: Vec<String> = md.base.vars.iter().map(|v| v.name.clone()).collect();
    let mut ivars = Vec::new();
    let mut pvars = Vec::new();
    let mut rvars = Vec::new();
    for v in base.vars.iter().filter(|v| !v.var_type.is_string()).map(|v| v.name.clone()) {
        let c0 = col(&base, &v);
        let varies = outs[1..].iter().any(|d| col(d, &v).iter().zip(&c0).any(|(a, b)| a.to_bits() != b.to_bits()));
        if md.ivars.contains(&v) || extra_ivars.contains(&v) {
            ivars.push(v);
        } else if md.pvars.contains(&v) {
            pvars.push(v);
        } else if md.rvars.contains(&v) && !varies {
            rvars.push(v);
        } else if !old.contains(&v) && new_imputed && (varies || md.ivars.iter().any(|iv| v.starts_with(iv.as_str()) || iv.starts_with(v.as_str()))) {
            ivars.push(v);
        } else if varies {
            pvars.push(v);
        }
    }
    let mut imp = Vec::new();
    for d in &outs[1..] {
        let mut slot = BTreeMap::new();
        for v in ivars.iter().chain(&pvars) {
            slot.insert(v.clone(), col(d, v));
        }
        imp.push(slot);
    }
    Ok(MiData { style: md.style.clone(), m: md.m, ivars, pvars, rvars, base, imp, name: md.name.clone() })
}

fn each_simple(me: &mut AdoInterpreter, c: &str, show: bool, new_imputed: bool) -> Result<(), String> {
    let md = load(me)?;
    let c = c.to_string();
    let outs = run_each(me, &md, &|_| c.clone(), show)?;
    let nd = rebuild(&md, outs, new_imputed, &[])?;
    save(me, &nd)
}

/// `mi streset [, options]` re-runs the stset command recorded in the data
/// with the given options replacing those of the same name; `mi st` shows it.
fn mi_streset(me: &mut AdoInterpreter, sub: &str, args: &str) -> Result<(), String> {
    let md = load(me)?;
    let prev = ferrostat_cmd::survival::st_char(&md.base, "st_cmd").ok_or_else(|| rc_err(119, "data not st; use mi stset"))?;
    if sub == "st" {
        me.emit(&format!("-> {prev}"));
        return Ok(());
    }
    let (pmain, popts) = split_args(prev.trim_start_matches("stset").trim_start_matches("streset"));
    let pmain = pmain.trim_start_matches("stset").trim().to_string();
    let (nmain, nopts) = split_args(args);
    let main = if nmain.trim().is_empty() { pmain } else { nmain };
    let mut opts: Vec<(String, Option<String>)> = popts.into_iter().filter(|(k, _)| !nopts.iter().any(|(n, _)| n.eq_ignore_ascii_case(k))).collect();
    opts.extend(nopts);
    let o: Vec<String> = opts.iter().map(|(k, v)| match v { Some(a) => format!("{k}({a})"), None => k.clone() }).collect();
    let c = format!("stset {main}{}", if o.is_empty() { String::new() } else { format!(", {}", o.join(" ")) });
    each_simple(me, &c, true, false)
}

/// `mi merge 1:1 varlist using f [, opts]` and `mi append using f [, opts]`.
/// A using file that is itself mi data contributes imputation k to m = k
/// (its last imputation to later ones); otherwise the same data are used
/// for every m.
fn merge_append(me: &mut AdoInterpreter, args: &str, merge: bool) -> Result<(), String> {
    let md = load(me)?;
    let a = args.trim();
    let ui = a.find(" using ").map(|i| i + 1).or_else(|| a.starts_with("using ").then_some(0)).ok_or_else(|| rc_err(100, "using required"))?;
    let head = a[..ui].trim().to_string();
    let rest = a[ui + 5..].trim();
    let (file, opts_txt) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim().to_string(), format!(", {}", rest[i + 1..].trim())),
        None => (rest.to_string(), String::new()),
    };
    let path = with_ext(&file, ".dta");
    let ud = read_dta(&path)?;
    let mut files = Vec::new();
    let mut extra = Vec::new();
    if MiData::is_set(&ud) {
        let um = MiData::load(&ud)?;
        extra = um.ivars.clone();
        for k in 0..=md.m {
            let f = format!("__mi_using_{k}.dta");
            write_dta(&um.extract(k.min(um.m)), &f, true)?;
            files.push(f);
        }
    } else {
        files.push(path);
    }
    let verb = if merge { "merge" } else { "append" };
    let r = run_each(me, &md, &|k| format!("{verb} {head} using \"{}\"{opts_txt}", files[k.min(files.len() - 1)]), true);
    for f in &files {
        if f.starts_with("__mi_using_") {
            let _ = ferrostat_core::vfs::remove(f);
        }
    }
    let nd = rebuild(&md, r?, false, &extra)?;
    save(me, &nd)
}

// ---------------------------------------------------------------------------
// mi add / replace0 / select
// ---------------------------------------------------------------------------

fn mi_add(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let mut md = load(me)?;
    let a = args.trim();
    let ui = a.find("using").ok_or_else(|| rc_err(100, "using required"))?;
    let vl = a[..ui].trim();
    let (file, opts) = split_args(&a[ui + 5..]);
    let ud = read_dta(&with_ext(&file, ".dta"))?;
    if !MiData::is_set(&ud) {
        return Err(rc_err(119, format!("{file} is not mi data")));
    }
    let um = MiData::load(&ud)?;
    if um.base.n_obs != md.base.n_obs {
        return Err(rc_err(459, "master and using have different numbers of observations"));
    }
    let vars: Vec<String> = if vl.is_empty() { md.ivars.iter().filter(|v| um.ivars.contains(v)).cloned().collect() } else { varlist::expand(&md.base, vl)? };
    for v in &vars {
        if !md.ivars.contains(v) {
            return Err(rc_err(198, format!("variable {v} not registered as imputed in master")));
        }
        if !um.ivars.contains(v) {
            return Err(rc_err(198, format!("variable {v} not registered as imputed in using")));
        }
    }
    let _ = has(&opts, "noupdate", 5);
    for k in 1..=um.m {
        let mut slot = BTreeMap::new();
        for v in md.ivars.iter().chain(&md.pvars) {
            slot.insert(v.clone(), col(&md.base, v));
        }
        for v in &vars {
            slot.insert(v.clone(), um.imp[k - 1].get(v).cloned().unwrap_or_else(|| col(&um.base, v)));
        }
        md.imp.push(slot);
    }
    md.m += um.m;
    save(me, &md)?;
    me.emit(&format!("({} imputation{} added; M = {})", um.m, if um.m == 1 { "" } else { "s" }, md.m));
    Ok(())
}

fn key_of(ds: &DataSet, vars: &[String], i: usize) -> String {
    vars.iter().map(|v| ds.get_var(v).map(|x| util::cell_text(ds, x, i, false)).unwrap_or_default()).collect::<Vec<_>>().join("\u{1f}")
}

fn mi_replace0(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let md = load(me)?;
    let a = args.trim();
    let rest = a.strip_prefix("using").ok_or_else(|| rc_err(100, "using required"))?;
    let (file, opts) = split_args(rest);
    let idv = arg(&opts, "id", 2).ok_or_else(|| rc_err(198, "option id() required"))?;
    let mut nb = read_dta(&with_ext(&file, ".dta"))?;
    strip_mi_chars(&mut nb);
    let ids = varlist::expand(&nb, &idv)?;
    let old: HashMap<String, usize> = (0..md.base.n_obs).map(|i| (key_of(&md.base, &ids, i), i)).collect();
    let map: Vec<Option<usize>> = (0..nb.n_obs).map(|i| old.get(&key_of(&nb, &ids, i)).copied()).collect();
    let keep = |l: &[String]| l.iter().filter(|v| nb.var_index.contains_key(*v)).cloned().collect::<Vec<_>>();
    let ivars = keep(&md.ivars);
    let pvars = keep(&md.pvars);
    let rvars = keep(&md.rvars);
    let mut imp = Vec::new();
    for slot in &md.imp {
        let mut ns = BTreeMap::new();
        for v in ivars.iter().chain(&pvars) {
            let newv = col(&nb, v);
            let oldc = slot.get(v).cloned().unwrap_or_default();
            let vals: Vec<f64> = (0..nb.n_obs)
                .map(|i| {
                    if !is_missing(newv[i]) || pvars.contains(v) && map[i].is_none() {
                        newv[i]
                    } else {
                        map[i].and_then(|o| oldc.get(o).copied()).unwrap_or(newv[i])
                    }
                })
                .collect();
            ns.insert(v.clone(), vals);
        }
        imp.push(ns);
    }
    let nd = MiData { style: md.style.clone(), m: md.m, ivars, pvars, rvars, base: nb, imp, name: md.name.clone() };
    save(me, &nd)
}

thread_local! {
    static SELECT: RefCell<Option<DataSet>> = const { RefCell::new(None) };
}

fn mi_select(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (main, _) = split_args(args);
    if main.trim().eq_ignore_ascii_case("init") {
        load(me)?;
        let ds = me.engine.dataset.clone();
        SELECT.with(|s| *s.borrow_mut() = Some(ds));
        let r = me.engine.store.begin_r();
        r.set_str("priorcmd", "");
        return Ok(());
    }
    let k: usize = main.trim().parse().map_err(|_| rc_err(198, "mi select init | mi select #"))?;
    let ds = SELECT.with(|s| s.borrow().clone()).ok_or_else(|| rc_err(198, "mi select init must be run first"))?;
    let md = MiData::load(&ds)?;
    if k > md.m {
        return Err(rc_err(198, format!("imputation {k} not found; M = {}", md.m)));
    }
    set_data(me, md.extract(k));
    Ok(())
}

// ---------------------------------------------------------------------------
// mi export / import
// ---------------------------------------------------------------------------

fn stacked(md: &MiData) -> Result<DataSet, String> {
    let n = md.base.n_obs;
    let idx: Vec<usize> = (0..=md.m).flat_map(|_| 0..n).collect();
    let mut ds = take_rows(&md.base, &idx);
    for v in md.ivars.iter().chain(&md.pvars) {
        let mut vals = col(&md.base, v);
        for k in 1..=md.m {
            vals.extend(md.imp[k - 1].get(v).cloned().unwrap_or_else(|| col(&md.base, v)));
        }
        set_num_col(&mut ds, v, &vals)?;
    }
    strip_mi_chars(&mut ds);
    Ok(ds)
}

fn add_int_var(ds: &mut DataSet, name: &str, vals: &[f64], ty: VarType) -> Result<(), String> {
    let _ = ds.drop_variable(name);
    let mut v = Variable::new(name, ty);
    v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
    ds.add_variable(v)
}

fn mi_export(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (main, opts) = split_args(args);
    let w: Vec<&str> = main.split_whitespace().collect();
    let fmt = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let md = load(me)?;
    let n = md.base.n_obs;
    match fmt.as_str() {
        "ice" => {
            let mut ds = stacked(&md)?;
            let mj: Vec<f64> = (0..=md.m).flat_map(|k| std::iter::repeat_n(k as f64, n)).collect();
            let mi: Vec<f64> = (0..=md.m).flat_map(|_| (1..=n).map(|i| i as f64)).collect();
            add_int_var(&mut ds, "_mj", &mj, VarType::Int)?;
            add_int_var(&mut ds, "_mi", &mi, VarType::Long)?;
            set_data(me, ds);
            Ok(())
        }
        "nhanes1" => {
            let base = w.get(1).ok_or_else(|| rc_err(198, "filename base required"))?.trim_end_matches(".dta").to_string();
            let replace = has(&opts, "replace", 7);
            if !md.pvars.is_empty() && !has(&opts, "passiveok", 7) {
                return Err(rc_err(198, "passive variables are present; specify passiveok to export them as imputed"));
            }
            let upper = has(&opts, "uppercase", 5);
            let suffix = |s: &str| if upper { s.to_uppercase() } else { s.to_string() };
            let id = if md.base.var_index.contains_key("seqn") { None } else { Some((1..=n).map(|x| x as f64).collect::<Vec<_>>()) };
            let ipv: Vec<String> = md.ivars.iter().chain(&md.pvars).cloned().collect();
            // nonimputed file: everything but the imputed variables
            let mut d0 = md.base.clone();
            strip_mi_chars(&mut d0);
            for v in &ipv {
                let _ = d0.drop_variable(v);
            }
            if let Some(idv) = &id {
                add_int_var(&mut d0, "seqn", idv, VarType::Long)?;
            }
            // imputation file: seqn, v1..vM and the flag vIF
            let mut d1 = DataSet::new();
            for _ in 0..n {
                d1.add_observation();
            }
            add_int_var(&mut d1, "seqn", &id.clone().unwrap_or_else(|| col(&md.base, "seqn")), VarType::Long)?;
            for v in &ipv {
                let c0 = col(&md.base, v);
                for k in 1..=md.m {
                    let vals = md.imp[k - 1].get(v).cloned().unwrap_or_else(|| c0.clone());
                    d1.add_variable(util::new_double(&format!("{v}{k}"), vals))?;
                }
                let flag: Vec<f64> = c0.iter().map(|x| if is_missing(*x) { 1.0 } else { 0.0 }).collect();
                add_int_var(&mut d1, &format!("{v}{}", suffix("if")), &flag, VarType::Byte)?;
            }
            write_dta(&d0, &format!("{base}.dta"), replace)?;
            write_dta(&d1, &format!("{base}_imp.dta"), replace)?;
            me.emit(&format!("(files {base}.dta and {base}_imp.dta saved)"));
            Ok(())
        }
        _ => Err(rc_err(198, format!("mi export {fmt}: format must be ice or nhanes1"))),
    }
}

/// Registration lists from imputed() / passive() / regular().
fn reg_lists(ds: &DataSet, opts: &Opts) -> Result<(Vec<String>, Vec<String>, Vec<String>), String> {
    let get = |k: &str| -> Result<Vec<String>, String> {
        match arg(opts, k, 3) {
            Some(l) if !l.trim().is_empty() => varlist::expand(ds, &l),
            _ => Ok(Vec::new()),
        }
    };
    Ok((get("imputed")?, get("passive")?, get("regular")?))
}

/// Variables that differ between m = 0 and some imputation.
fn varying(base: &DataSet, imp: &[BTreeMap<String, Vec<f64>>]) -> Vec<String> {
    base.vars
        .iter()
        .filter(|v| !v.var_type.is_string())
        .map(|v| v.name.clone())
        .filter(|v| {
            let c0 = col(base, v);
            imp.iter().any(|s| s.get(v).is_some_and(|c| c.iter().zip(&c0).any(|(a, b)| a.to_bits() != b.to_bits())))
        })
        .collect()
}

/// Build mi data from m = 0 data and per-imputation datasets matched on ids.
fn from_parts(base: DataSet, parts: &[DataSet], ids: &[String], ivars: Vec<String>, pvars: Vec<String>, rvars: Vec<String>, auto: bool, style: &str) -> Result<MiData, String> {
    let n = base.n_obs;
    let pos: HashMap<String, usize> = (0..n).map(|i| (key_of(&base, ids, i), i)).collect();
    if pos.len() != n {
        return Err(rc_err(459, "id() does not uniquely identify the observations in m = 0"));
    }
    let candidates: Vec<String> = if auto || ivars.is_empty() { base.vars.iter().filter(|v| !v.var_type.is_string() && !ids.contains(&v.name)).map(|v| v.name.clone()).collect() } else { ivars.iter().chain(&pvars).cloned().collect() };
    let mut imp = Vec::new();
    for d in parts {
        let mut slot = BTreeMap::new();
        for v in &candidates {
            let mut c = col(&base, v);
            let src = col(d, v);
            if d.var_index.contains_key(v) {
                for r in 0..d.n_obs {
                    if let Some(&t) = pos.get(&key_of(d, ids, r)) {
                        c[t] = src[r];
                    }
                }
            }
            slot.insert(v.clone(), c);
        }
        imp.push(slot);
    }
    let (ivars, pvars) = if auto || ivars.is_empty() && pvars.is_empty() {
        let v = varying(&base, &imp);
        (v, pvars)
    } else {
        (ivars, pvars)
    };
    for s in imp.iter_mut() {
        s.retain(|k, _| ivars.contains(k) || pvars.contains(k));
    }
    let mut md = MiData::new(base, style);
    md.m = parts.len();
    md.ivars = ivars;
    md.pvars = pvars;
    md.rvars = rvars;
    md.imp = imp;
    Ok(md)
}

fn mi_import(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (main, opts) = split_args(args);
    let w: Vec<&str> = main.split_whitespace().collect();
    let fmt = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    check_clear(me, &opts)?;
    let ds = me.engine.dataset.clone();
    if MiData::is_set(&ds) {
        return Err(rc_err(119, "data already mi set"));
    }
    let auto = has(&opts, "automatic", 4);
    let md = match fmt.as_str() {
        "flong" | "ice" => {
            let (mv, idv) = if fmt == "ice" {
                ("_mj".to_string(), "_mi".to_string())
            } else {
                (arg(&opts, "m", 1).ok_or_else(|| rc_err(198, "option m() required"))?.trim().to_string(), arg(&opts, "id", 2).ok_or_else(|| rc_err(198, "option id() required"))?)
            };
            let ids = varlist::expand(&ds, &idv)?;
            let mc = col(&ds, &mv);
            if !ds.var_index.contains_key(&mv) {
                return Err(rc_err(111, format!("variable {mv} not found")));
            }
            let mmax = mc.iter().filter(|x| !is_missing(**x)).fold(0.0f64, |a, b| a.max(*b)) as usize;
            let rows0: Vec<usize> = (0..ds.n_obs).filter(|&i| mc[i] == 0.0).collect();
            let mut base = take_rows(&ds, &rows0);
            let _ = base.drop_variable(&mv);
            let parts: Vec<DataSet> = (1..=mmax).map(|k| take_rows(&ds, &(0..ds.n_obs).filter(|&i| mc[i] == k as f64).collect::<Vec<_>>())).collect();
            let (iv, pv, rv) = reg_lists(&base, &opts)?;
            let mut md = from_parts(base, &parts, &ids, iv, pv, rv, auto || fmt == "ice" && arg(&opts, "imputed", 3).is_none(), "flong")?;
            if fmt == "ice" {
                for v in ["_mi", "_mj"] {
                    let _ = md.base.drop_variable(v);
                }
            }
            md
        }
        "flongsep" | "nhanes1" => {
            let name = w.get(1).ok_or_else(|| rc_err(198, "name required"))?.to_string();
            let files: Vec<String> = arg(&opts, "using", 2).ok_or_else(|| rc_err(198, "option using() required"))?.split_whitespace().map(|f| with_ext(f, ".dta")).collect();
            let idv = arg(&opts, "id", 2).unwrap_or_else(|| "seqn".into());
            let mut base = ds.clone();
            let ids = varlist::expand(&base, &idv)?;
            if fmt == "flongsep" {
                let parts: Vec<DataSet> = files.iter().map(|f| read_dta(f)).collect::<Result<_, _>>()?;
                let (iv, pv, rv) = reg_lists(&base, &opts)?;
                let mut md = from_parts(base, &parts, &ids, iv, pv, rv, auto, "flongsep")?;
                md.name = name;
                md
            } else {
                // nhanes1: the imputation file holds v1..vM and the flag vIF
                let d1 = read_dta(files.first().ok_or_else(|| rc_err(198, "using() required"))?)?;
                let pos: HashMap<String, usize> = (0..d1.n_obs).map(|i| (key_of(&d1, &ids, i), i)).collect();
                let names: Vec<String> = d1.vars.iter().map(|v| v.name.clone()).collect();
                let mut ivars = Vec::new();
                let mut m = 0;
                for f in &names {
                    let l = f.to_lowercase();
                    if let Some(stem) = l.strip_suffix("if").map(|s| f[..s.len()].to_string()) {
                        let k = (1..).take_while(|k| d1.var_index.contains_key(&format!("{stem}{k}"))).count();
                        if k > 0 {
                            ivars.push(stem);
                            m = if m == 0 { k } else { m.min(k) };
                        }
                    }
                }
                if ivars.is_empty() {
                    return Err(rc_err(459, "no imputed variables (v1, v2, ..., vIF) found in the imputation file"));
                }
                let n = base.n_obs;
                let rowmap: Vec<Option<usize>> = (0..n).map(|i| pos.get(&key_of(&base, &ids, i)).copied()).collect();
                let mut imp = vec![BTreeMap::new(); m];
                for v in &ivars {
                    let flag = col(&d1, &names.iter().find(|x| x.len() == v.len() + 2 && x.starts_with(v.as_str()) && x.to_lowercase().ends_with("if")).cloned().unwrap_or_default());
                    let first = col(&d1, &format!("{v}1"));
                    let vals0: Vec<f64> = rowmap.iter().map(|r| r.map_or(SYSMISS, |r| if flag[r] == 1.0 { SYSMISS } else { first[r] })).collect();
                    set_num_col(&mut base, v, &vals0)?;
                    for (k, slot) in imp.iter_mut().enumerate() {
                        let c = col(&d1, &format!("{v}{}", k + 1));
                        slot.insert(v.clone(), rowmap.iter().map(|r| r.map_or(SYSMISS, |r| c[r])).collect());
                    }
                }
                let mut md = MiData::new(base, "flongsep");
                md.m = m;
                md.ivars = ivars;
                md.imp = imp;
                md.name = name;
                md
            }
        }
        "wide" => {
            let spec = arg(&opts, "imputed", 3).ok_or_else(|| rc_err(198, "option imputed() required"))?;
            let mut base = ds.clone();
            let groups = parse_wide(&base, &spec)?;
            let pgroups = match arg(&opts, "passive", 3) {
                Some(p) => parse_wide(&base, &p)?,
                None => Vec::new(),
            };
            let m = groups.iter().chain(&pgroups).map(|g| g.1.len()).max().unwrap_or(0);
            if groups.iter().chain(&pgroups).any(|g| g.1.len() != m) {
                return Err(rc_err(198, "the same number of imputations must be given for every variable"));
            }
            let mut imp = vec![BTreeMap::new(); m];
            for (v, list) in groups.iter().chain(&pgroups) {
                for (k, src) in list.iter().enumerate() {
                    imp[k].insert(v.clone(), col(&base, src));
                }
            }
            if has(&opts, "drop", 4) {
                for (_, list) in groups.iter().chain(&pgroups) {
                    for s in list {
                        base.drop_variable(s)?;
                    }
                }
            }
            let mut md = MiData::new(base, "wide");
            md.m = m;
            md.ivars = groups.iter().map(|g| g.0.clone()).collect();
            md.pvars = pgroups.iter().map(|g| g.0.clone()).collect();
            md.imp = imp;
            md
        }
        other => return Err(rc_err(198, format!("mi import {other}: format must be flong, flongsep, ice, nhanes1 or wide"))),
    };
    me.engine.dataset = md.store(&md.style)?;
    me.engine.store.data_changed = true;
    Ok(())
}

/// `y = y1 y2 ...  z = z1 z2 ...`
fn parse_wide(ds: &DataSet, spec: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let toks: Vec<String> = spec.replace('=', " = ").split_whitespace().map(String::from).collect();
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks.get(i + 1).map(String::as_str) == Some("=") {
            let v = varlist::resolve_name(ds, &toks[i])?;
            out.push((v, Vec::new()));
            i += 2;
            continue;
        }
        let last = out.last_mut().ok_or_else(|| rc_err(198, "imputed(): varname = varlist expected"))?;
        last.1.extend(varlist::expand(ds, &toks[i])?);
        i += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// mi estimate, saving() / using; mi predict / predictnl
// ---------------------------------------------------------------------------

pub(crate) struct PreEst {
    pub opts: String,
    pub command: String,
    pub ests: Vec<Results>,
}

const MAGIC: &str = "ferrostat-mi-estimates";

pub(crate) fn save_estimates(spec: &str, list: &[usize], ests: &[Results], command: &str) -> Result<(), String> {
    let (f, o) = split_args(spec);
    let path = with_ext(&f, ".ster");
    if ferrostat_core::vfs::exists(&path) && !has(&o, "replace", 7) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let v = serde_json::json!({
        "magic": MAGIC,
        "command": command,
        "m": list,
        "ests": ests.iter().map(ferrostat_cmd::postest2::results_json).collect::<Vec<_>>(),
    });
    ferrostat_core::vfs::write(&path, v.to_string().as_bytes()).map_err(|e| rc_err(603, e.to_string()))
}

fn load_estimates(file: &str) -> Result<(String, Vec<Results>), String> {
    let path = with_ext(file, ".ster");
    let bytes = ferrostat_core::vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|_| rc_err(610, format!("file {path} not saved by mi estimate")))?;
    if v["magic"].as_str() != Some(MAGIC) {
        return Err(rc_err(610, format!("file {path} not saved by mi estimate")));
    }
    let ests = v["ests"].as_array().ok_or_else(|| rc_err(610, "invalid estimates file"))?.iter().map(ferrostat_cmd::postest2::results_from_json).collect::<Result<Vec<_>, _>>()?;
    Ok((v["command"].as_str().unwrap_or("").to_string(), ests))
}

/// `mi estimate [spec] using file [, options]` (no colon).
pub(crate) fn estimate_using(args: &str) -> Result<Option<PreEst>, String> {
    let a = args.trim();
    if find_top_level(a, ':').is_some() {
        return Ok(None);
    }
    let ui = if a.starts_with("using ") { 0 } else { match a.find(" using ") { Some(i) => i + 1, None => return Ok(None) } };
    let (file, opts) = match find_top_level(&a[ui + 5..], ',') {
        Some(i) => (a[ui + 5..ui + 5 + i].trim().to_string(), a[ui + 6 + i..].trim().to_string()),
        None => (a[ui + 5..].trim().to_string(), String::new()),
    };
    let (command, ests) = load_estimates(&file)?;
    if ests.is_empty() {
        return Err(rc_err(2000, "no estimates in file"));
    }
    Ok(Some(PreEst { opts, command, ests }))
}

fn mi_predict(me: &mut AdoInterpreter, args: &str, nl: bool) -> Result<(), String> {
    let a = args.trim();
    let ui = a.find(" using ").ok_or_else(|| rc_err(100, "using required"))?;
    let lhs = a[..ui].trim().to_string();
    let (file, opts) = match find_top_level(&a[ui + 7..], ',') {
        Some(i) => (a[ui + 7..ui + 7 + i].trim().to_string(), a[ui + 8 + i..].trim().to_string()),
        None => (a[ui + 7..].trim().to_string(), String::new()),
    };
    let (_, ests) = load_estimates(&file)?;
    let mut md = load(me)?;
    // combined estimates (Rubin's rules)
    let bs: Vec<Vec<f64>> = ests.iter().map(|e| e.matrices.get("b").map(|m| m.data.clone()).unwrap_or_default()).collect();
    let vs: Vec<Matrix> = ests.iter().map(|e| e.matrices.get("V").cloned().unwrap_or_default()).collect();
    if bs.is_empty() || bs[0].is_empty() {
        return Err(rc_err(301, "estimates file has no coefficients"));
    }
    let r = crate::mi::rubin(&bs, &vs, ests[0].get("df_r"));
    let mut e = ests[0].clone();
    let b0 = e.matrices.get("b").cloned().unwrap();
    let v0 = e.matrices.get("V").cloned().unwrap();
    let mut bm = b0.clone();
    bm.data = r.b.clone();
    let mut vm = v0.clone();
    vm.data = r.t.data.clone();
    e.set_mat("b", bm);
    e.set_mat("V", vm);
    let newvar = {
        let l = if nl { lhs.split('=').next().unwrap_or("").to_string() } else { lhs.clone() };
        let w: Vec<&str> = l.split_whitespace().collect();
        let w: Vec<&str> = w.into_iter().take_while(|t| *t != "if" && *t != "in").collect();
        w.last().ok_or_else(|| rc_err(198, "new variable name required"))?.to_string()
    };
    let verb = if nl { "predictnl" } else { "predict" };
    let c = format!("{verb} {lhs}{}", if opts.is_empty() { String::new() } else { format!(", {opts}") });
    let saved_e = std::mem::replace(&mut me.engine.store.e, e.clone());
    let res = run_each(me, &md, &|_| c.clone(), false);
    me.engine.store.e = saved_e;
    let outs = res?;
    let vals0 = col(&outs[0], &newvar);
    set_num_col(&mut md.base, &newvar, &vals0)?;
    if let Some(v) = outs[0].get_var(&newvar) {
        if let Some(t) = md.base.get_var_mut(&newvar) {
            t.label = v.label.clone();
        }
    }
    for k in 1..=md.m {
        md.imp[k - 1].insert(newvar.clone(), col(&outs[k], &newvar));
    }
    md.ivars.retain(|v| *v != newvar);
    md.rvars.retain(|v| *v != newvar);
    if !md.pvars.contains(&newvar) {
        md.pvars.push(newvar);
    }
    save(me, &md)
}

// ---------------------------------------------------------------------------
// mi ptrace
// ---------------------------------------------------------------------------

/// Write the parameter trace of `mi impute mvn, savetrace()`: one record
/// per iteration with the coefficients b_y_x and the covariances v_y1_y2.
pub(crate) fn write_trace(spec: &str, ys: &[String], xs: &[String], trace: &[(usize, usize, Vec<Vec<f64>>, Matrix)]) -> Result<(), String> {
    let (f, o) = split_args(spec);
    let path = with_ext(&f, ".stptrace");
    let mut ds = DataSet::new();
    for _ in 0..trace.len() {
        ds.add_observation();
    }
    add_int_var(&mut ds, "m", &trace.iter().map(|t| t.0 as f64).collect::<Vec<_>>(), VarType::Int)?;
    add_int_var(&mut ds, "iter", &trace.iter().map(|t| t.1 as f64).collect::<Vec<_>>(), VarType::Long)?;
    for (j, y) in ys.iter().enumerate() {
        for (i, x) in xs.iter().enumerate() {
            let nm = format!("b_{y}_{x}").replace('.', "_");
            let mut v = util::new_double(&nm, trace.iter().map(|t| t.2.get(i).and_then(|r| r.get(j)).copied().unwrap_or(SYSMISS)).collect());
            v.label = format!("[{y}]{x}");
            ds.add_variable(v)?;
        }
    }
    for a in 0..ys.len() {
        for b in a..ys.len() {
            let mut v = util::new_double(&format!("v_{}_{}", ys[a], ys[b]), trace.iter().map(|t| t.3[(a, b)]).collect());
            v.label = format!("cov({},{})", ys[a], ys[b]);
            ds.add_variable(v)?;
        }
    }
    let c = ds.characteristics.entry("_dta".into()).or_default();
    c.insert("ptrace_ynames".into(), ys.join(" "));
    c.insert("ptrace_xnames".into(), xs.join(" "));
    write_dta(&ds, &path, has(&o, "replace", 7))
}

fn mi_ptrace(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let (main, opts) = split_args(args);
    let w: Vec<&str> = main.split_whitespace().collect();
    let sub = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let file = w.get(1).ok_or_else(|| rc_err(198, "filename required"))?;
    let path = with_ext(file, ".stptrace");
    let mut ds = read_dta(&path)?;
    let mm = col(&ds, "m");
    let iters = col(&ds, "iter");
    let m = mm.iter().cloned().fold(0.0, f64::max);
    let ch = ds.characteristics.get("_dta").cloned().unwrap_or_default();
    let ys = ch.get("ptrace_ynames").cloned().unwrap_or_default();
    let xs = ch.get("ptrace_xnames").cloned().unwrap_or_default();
    match sub.as_str() {
        "describe" | "d" | "de" | "des" | "desc" => {
            let out = [
                String::new(),
                format!("  file {path} created by mi impute mvn"),
                String::new(),
                format!("  Dependent variables: ({}) {ys}", ys.split_whitespace().count()),
                format!("  Independent variables: ({}) {xs}", xs.split_whitespace().count()),
                format!("  Number of imputations: {}", m as usize),
                format!("  Number of iterations: {}", ds.n_obs),
                format!("  Parameters: {}", ds.vars.len().saturating_sub(2)),
            ];
            me.emit(&out.join("\n"));
            let r = me.engine.store.begin_r();
            r.set("m", m);
            r.set("niter", iters.iter().cloned().fold(0.0, f64::max));
            r.set_str("ynames", ys);
            r.set_str("xnames", xs);
            Ok(())
        }
        "use" | "u" => {
            check_clear(me, &opts)?;
            if let Some(sel) = arg(&opts, "select", 3) {
                let keep_m = parse_numlist_m(&sel.replace("m==", "").replace("m=", ""), m as usize)?;
                let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| keep_m.contains(&(mm[i] as usize))).collect();
                ds = take_rows(&ds, &rows);
            }
            set_data(me, ds);
            Ok(())
        }
        other => Err(rc_err(198, format!("mi ptrace {other}: use describe or use"))),
    }
}

// ---------------------------------------------------------------------------
// mi impute intreg | nbreg | truncreg
// ---------------------------------------------------------------------------

/// Maximize `ll` by Newton steps with numerical derivatives; returns the
/// estimates and the inverse of the negative Hessian.
fn ml_newton(ll: &dyn Fn(&[f64]) -> f64, start: Vec<f64>) -> Result<(Vec<f64>, Matrix), String> {
    let k = start.len();
    let mut th = start;
    let mut f0 = ll(&th);
    if !f0.is_finite() {
        return Err(rc_err(430, "could not evaluate the likelihood at the starting values"));
    }
    let hstep = |x: f64| 1e-4 * (1.0 + x.abs());
    let grad_hess = |th: &[f64]| -> (Vec<f64>, Matrix) {
        let mut g = vec![0.0; k];
        let mut h = Matrix::zeros(k, k);
        let f = ll(th);
        for i in 0..k {
            let hi = hstep(th[i]);
            let mut a = th.to_vec();
            a[i] += hi;
            let fp = ll(&a);
            a[i] -= 2.0 * hi;
            let fm = ll(&a);
            g[i] = (fp - fm) / (2.0 * hi);
            h[(i, i)] = (fp - 2.0 * f + fm) / (hi * hi);
            for j in 0..i {
                let hj = hstep(th[j]);
                let mut p = th.to_vec();
                p[i] += hi;
                p[j] += hj;
                let fpp = ll(&p);
                p[j] -= 2.0 * hj;
                let fpm = ll(&p);
                p[i] -= 2.0 * hi;
                let fmm = ll(&p);
                p[j] += 2.0 * hj;
                let fmp = ll(&p);
                let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        (g, h)
    };
    for _ in 0..200 {
        let (g, h) = grad_hess(&th);
        let (hi, _) = h.scale(-1.0).sym_inverse(1e-12);
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * g[j]).sum()).collect();
        let mut t = 1.0;
        let mut improved = false;
        for _ in 0..30 {
            let cand: Vec<f64> = th.iter().zip(&step).map(|(a, s)| a + t * s).collect();
            let f1 = ll(&cand);
            if f1.is_finite() && f1 >= f0 - 1e-12 {
                let diff = f1 - f0;
                th = cand;
                f0 = f1;
                improved = diff > 1e-10;
                break;
            }
            t *= 0.5;
        }
        if !improved {
            break;
        }
    }
    let (_, h) = grad_hess(&th);
    let (v, _) = h.scale(-1.0).sym_inverse(1e-12);
    Ok((th, v))
}

fn chol_draw(rng: &mut Rng, b: &[f64], v: &Matrix) -> Vec<f64> {
    let n = b.len();
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = v[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = if i == j { s.max(0.0).sqrt() } else if l[(j, j)] > 0.0 { s / l[(j, j)] } else { 0.0 };
        }
    }
    let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    (0..n).map(|i| b[i] + (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>()).collect()
}

/// Draw from N(mu, s^2) truncated to [lo, hi] (missing = unbounded).
fn trunc_normal(rng: &mut Rng, mu: f64, s: f64, lo: f64, hi: f64) -> f64 {
    if !is_missing(lo) && !is_missing(hi) && lo == hi {
        return lo;
    }
    let pa = if is_missing(lo) { 0.0 } else { dist::normal((lo - mu) / s) };
    let pb = if is_missing(hi) { 1.0 } else { dist::normal((hi - mu) / s) };
    if pb - pa < 1e-300 {
        // far in the tail: the nearest bound
        return if is_missing(lo) { hi } else { lo };
    }
    let u = pa + rng.uniform() * (pb - pa);
    let z = dist::invnormal(u.clamp(1e-300, 1.0 - 1e-16));
    let y = mu + s * z;
    let y = if !is_missing(lo) { y.max(lo) } else { y };
    if !is_missing(hi) { y.min(hi) } else { y }
}

fn ln_norm_cdf_diff(a: f64, b: f64) -> f64 {
    // ln(Phi(b) - Phi(a)) with infinite bounds allowed
    let pa = if a == f64::NEG_INFINITY { 0.0 } else { dist::normal(a) };
    let pb = if b == f64::INFINITY { 1.0 } else { dist::normal(b) };
    let d = pb - pa;
    if d > 1e-300 {
        d.ln()
    } else if a > 0.0 {
        // upper tail
        let qa = if a == f64::NEG_INFINITY { 1.0 } else { dist::normal(-a) };
        let qb = if b == f64::INFINITY { 0.0 } else { dist::normal(-b) };
        (qa - qb).max(1e-300).ln()
    } else {
        -690.0
    }
}

fn limit_col(base: &DataSet, s: Option<String>) -> Result<Vec<f64>, String> {
    let n = base.n_obs;
    match s {
        None => Ok(vec![SYSMISS; n]),
        Some(t) => {
            let t = t.trim().to_string();
            if let Ok(x) = t.parse::<f64>() {
                Ok(vec![x; n])
            } else {
                let v = varlist::resolve_name(base, &t)?;
                Ok(col(base, &v))
            }
        }
    }
}

pub(crate) fn impute_extra(me: &mut AdoInterpreter, meth: &str, rest: &str) -> Result<(), String> {
    let (main, opts) = split_args(rest);
    let mut md = load(me)?;
    let add = arg(&opts, "add", 3).map(|a| a.trim().parse::<usize>().map_err(|_| rc_err(198, "add() invalid"))).transpose()?;
    let replace = has(&opts, "replace", 7);
    if add.is_none() && !replace {
        return Err(rc_err(198, "either add() or replace is required"));
    }
    if let Some(s) = arg(&opts, "rseed", 5) {
        let seed: u64 = s.trim().parse().map_err(|_| rc_err(198, "rseed() invalid"))?;
        me.engine.store.rng.borrow_mut().reseed(seed);
    }
    let ims: Vec<usize> = if replace && add.is_none() { (1..=md.m).collect() } else { (md.m + 1..=md.m + add.unwrap_or(0)).collect() };
    if ims.is_empty() {
        return Err(rc_err(198, "no imputations to fill; specify add(#)"));
    }
    let new_m = md.m.max(*ims.last().unwrap());
    let base = md.base.clone();
    let n = base.n_obs;
    let (main, cond) = {
        let t = format!(" {main} ");
        match t.find(" if ") {
            Some(i) => (t[..i].trim().to_string(), Some(t[i + 4..].trim().to_string())),
            None => (main.trim().to_string(), None),
        }
    };
    let mut toks = main.split_whitespace();
    let y = varlist::resolve_name(&base, toks.next().ok_or_else(|| rc_err(100, "varlist required"))?)?;
    if !md.ivars.contains(&y) {
        return Err(rc_err(198, format!("variable {y} not registered as imputed; use mi register imputed")));
    }
    // predictors (i. factor notation expands to indicators)
    let mut pcols: Vec<(String, Option<f64>)> = Vec::new();
    for tok in toks {
        if let Some(v) = tok.strip_prefix("i.") {
            for name in varlist::expand(&base, v)? {
                let mut lv: Vec<f64> = col(&base, &name).into_iter().filter(|x| !is_missing(*x)).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                for l in lv.into_iter().skip(1) {
                    pcols.push((name.clone(), Some(l)));
                }
            }
        } else {
            for name in varlist::expand(&base, tok.trim_start_matches("c."))? {
                pcols.push((name, None));
            }
        }
    }
    let noconstant = has(&opts, "noconstant", 3);
    let mut sample = vec![true; n];
    if let Some(c) = &cond {
        let e = parse_expression(c).map_err(|e| rc_err(198, e.message))?;
        let vals = util::eval_col(&base, &me.engine.store, &e, &vec![true; n])?;
        for i in 0..n {
            sample[i] = !is_missing(vals[i]) && vals[i] != 0.0;
        }
    }
    for (v, _) in &pcols {
        if md.ivars.contains(v) {
            continue;
        }
        let c = col(&base, v);
        for i in 0..n {
            if is_missing(c[i]) {
                sample[i] = false;
            }
        }
    }
    let yc = col(&base, &y);
    let (ll, ul) = (limit_col(&base, arg(&opts, "ll", 2))?, limit_col(&base, arg(&opts, "ul", 2))?);
    if meth == "intreg" && arg(&opts, "ll", 2).is_none() && arg(&opts, "ul", 2).is_none() {
        return Err(rc_err(198, "options ll() and ul() required"));
    }
    // rows informative for the model and rows to fill
    let obs: Vec<usize> = (0..n)
        .filter(|&i| sample[i])
        .filter(|&i| if meth == "intreg" { !is_missing(ll[i]) || !is_missing(ul[i]) } else { !is_missing(yc[i]) })
        .collect();
    let fill: Vec<usize> = (0..n).filter(|&i| sample[i] && yc[i] == SYSMISS).collect();
    if obs.len() < 2 {
        return Err(rc_err(2001, "insufficient observations to fit the imputation model"));
    }
    if meth == "nbreg" && obs.iter().any(|&i| yc[i] < 0.0 || yc[i].fract() != 0.0) {
        return Err(rc_err(459, "nbreg imputation requires nonnegative integer values"));
    }
    while md.imp.len() < new_m {
        let mut slot = BTreeMap::new();
        for v in md.ivars.iter().chain(&md.pvars) {
            slot.insert(v.clone(), col(&base, v));
        }
        md.imp.push(slot);
    }
    md.m = new_m;
    let mut filled = usize::MAX;
    for &m in &ims {
        let mut x: Vec<Vec<f64>> = pcols
            .iter()
            .map(|(v, l)| {
                let c = if md.ivars.contains(v) { md.imp[m - 1].get(v).cloned().unwrap_or_else(|| col(&base, v)) } else { col(&base, v) };
                match l {
                    Some(l) => c.iter().map(|&z| if is_missing(z) { SYSMISS } else if z == *l { 1.0 } else { 0.0 }).collect(),
                    None => c,
                }
            })
            .collect();
        if !noconstant {
            x.push(vec![1.0; n]);
        }
        let k = x.len();
        let xb = |b: &[f64], t: usize| -> f64 { (0..k).map(|j| x[j][t] * b[j]).sum() };
        let ols = {
            let yy: Vec<f64> = obs.iter().map(|&t| if meth == "intreg" { if !is_missing(ll[t]) && !is_missing(ul[t]) { 0.5 * (ll[t] + ul[t]) } else if !is_missing(ll[t]) { ll[t] } else { ul[t] } } else { yc[t] }).collect();
            let mut a = Matrix::zeros(k, k);
            let mut c = vec![0.0; k];
            for (r, &t) in obs.iter().enumerate() {
                for i in 0..k {
                    c[i] += x[i][t] * if meth == "nbreg" { (yy[r] + 0.5).ln() } else { yy[r] };
                    for j in 0..k {
                        a[(i, j)] += x[i][t] * x[j][t];
                    }
                }
            }
            let (ai, _) = a.sym_inverse(1e-13);
            let b: Vec<f64> = (0..k).map(|i| (0..k).map(|j| ai[(i, j)] * c[j]).sum()).collect();
            let rss: f64 = obs.iter().enumerate().map(|(r, &t)| (if meth == "nbreg" { (yy[r] + 0.5).ln() } else { yy[r] } - xb(&b, t)).powi(2)).sum::<f64>();
            (b, (rss / obs.len() as f64).sqrt().max(1e-6))
        };
        let mut start = ols.0.clone();
        start.push(if meth == "nbreg" { 0.0 } else { ols.1.ln() });
        let lik: Box<dyn Fn(&[f64]) -> f64> = match meth {
            "intreg" => Box::new(|th: &[f64]| {
                let s = th[k].exp();
                obs.iter()
                    .map(|&t| {
                        let mu = xb(th, t);
                        let (a, b) = (ll[t], ul[t]);
                        if !is_missing(a) && !is_missing(b) && a == b {
                            -0.5 * ((a - mu) / s).powi(2) - s.ln() - 0.918_938_533_204_672_8
                        } else {
                            ln_norm_cdf_diff(if is_missing(a) { f64::NEG_INFINITY } else { (a - mu) / s }, if is_missing(b) { f64::INFINITY } else { (b - mu) / s })
                        }
                    })
                    .sum()
            }),
            "truncreg" => Box::new(|th: &[f64]| {
                let s = th[k].exp();
                obs.iter()
                    .map(|&t| {
                        let mu = xb(th, t);
                        let z = (yc[t] - mu) / s;
                        let lo = if is_missing(ll[t]) { f64::NEG_INFINITY } else { (ll[t] - mu) / s };
                        let hi = if is_missing(ul[t]) { f64::INFINITY } else { (ul[t] - mu) / s };
                        -0.5 * z * z - s.ln() - 0.918_938_533_204_672_8 - ln_norm_cdf_diff(lo, hi)
                    })
                    .sum()
            }),
            _ => Box::new(|th: &[f64]| {
                let a = th[k].exp();
                let r = 1.0 / a;
                obs.iter()
                    .map(|&t| {
                        let mu = xb(th, t).exp();
                        let y = yc[t];
                        dist::lngamma(y + r) - dist::lngamma(r) - dist::lngamma(y + 1.0) + r * (r / (r + mu)).ln() + y * (mu / (r + mu)).ln()
                    })
                    .sum()
            }),
        };
        let (th, v) = ml_newton(&*lik, start)?;
        let mut rng = me.engine.store.rng.borrow_mut();
        let rng: &mut Rng = &mut rng;
        let d = chol_draw(rng, &th, &v);
        let mut c = yc.clone();
        for &t in &fill {
            let mu = xb(&d, t);
            c[t] = match meth {
                "nbreg" => {
                    let a = d[k].exp();
                    let g = rng.gamma(1.0 / a) * a;
                    rng.poisson(mu.exp() * g)
                }
                _ => trunc_normal(rng, mu, d[k].exp(), ll[t], ul[t]),
            };
        }
        md.imp[m - 1].insert(y.clone(), c);
        filled = filled.min(fill.len());
    }
    save(me, &md)?;
    let title = match meth {
        "intreg" => "Interval regression",
        "truncreg" => "Truncated regression",
        _ => "Negative binomial regression",
    };
    let added = if replace && add.is_none() { 0 } else { ims.len() };
    let range = if ims.len() == 1 { format!("Imputed: m={}", ims[0]) } else { format!("Imputed: m={} through m={}", ims[0], ims[ims.len() - 1]) };
    let comp = (0..n).filter(|&i| sample[i] && !is_missing(yc[i])).count();
    let inc = (0..n).filter(|&i| sample[i] && is_missing(yc[i])).count();
    let out = [
        String::new(),
        format!("{:<44}{:>11} = {:>8}", "Univariate imputation", "Imputations", new_m),
        format!("{:<44}{:>11} = {:>8}", title, "added", added),
        format!("{:<44}{:>11} = {:>8}", range, "updated", ims.len() - added),
        String::new(),
        "-".repeat(66),
        format!("{:>19}|{:^46}", "", "Observations per m"),
        format!("{:>19}|{}", "", "-".repeat(46)),
        format!("{:>18} |   Complete   Incomplete   Imputed |     Total", "Variable"),
        format!("{}+{}+{}", "-".repeat(19), "-".repeat(35), "-".repeat(10)),
        format!("{:>18} | {:>10}   {:>10} {:>9} | {:>9}", ferrostat_cmd::summarize::abbrev(&y, 18), comp, inc, if filled == usize::MAX { 0 } else { filled }, comp + inc),
        "-".repeat(66),
        "(Complete + Incomplete = Total; Imputed is the minimum across m".into(),
        " of the number of filled-in observations.)".into(),
    ];
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set("M", new_m as f64);
    r.set("M_add", added as f64);
    r.set("M_update", (ims.len() - added) as f64);
    r.set("k_ivars", 1.0);
    r.set_str("method", meth);
    r.set_str("ivars", y);
    Ok(())
}
