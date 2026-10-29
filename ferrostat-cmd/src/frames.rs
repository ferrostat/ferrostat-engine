//! Data frames: several datasets in memory at once (`frame`, `cwf`, `pwf`),
//! links between frames (`frlink`), copying linked values (`frget`) and the
//! `frval()` expression function.
//!
//! The current frame's data is the dataset every command works on; the other
//! frames are parked in `Store::data_frames` together with the per-dataset
//! state (tsset, preserve stack, changed flag, file name). Changing frames
//! swaps them.

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env, Val};
use crate::store::Store;
use crate::util::{touse, typed_value};
use crate::varlist;

/// Name of the frame that exists at start-up and after `frame reset`.
pub const DEFAULT_FRAME: &str = "default";

/// A frame that is not the current one.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub data: DataSet,
    pub tsset: Option<(Option<String>, String)>,
    pub preserved: Vec<DataSet>,
    pub data_changed: bool,
    pub filename: String,
}

/// Characteristics stored on a link variable created by `frlink`.
const CH_FRAME: &str = "frlink_frame";
const CH_KIND: &str = "frlink_kind";
const CH_MKEYS: &str = "frlink_keys";
const CH_UKEYS: &str = "frlink_tkeys";

pub fn exists(st: &Store, name: &str) -> bool {
    name == st.cur_frame || st.data_frames.contains_key(name)
}

fn not_found(name: &str) -> String {
    rc_err(111, format!("frame {name} not found"))
}

/// Make `name` the current frame.
pub fn switch(ds: &mut DataSet, st: &mut Store, name: &str) -> Result<(), String> {
    if name == st.cur_frame {
        return Ok(());
    }
    let target = st.data_frames.remove(name).ok_or_else(|| not_found(name))?;
    let old = Frame {
        data: std::mem::replace(ds, target.data),
        tsset: std::mem::replace(&mut st.tsset, target.tsset),
        preserved: std::mem::replace(&mut st.preserved, target.preserved),
        data_changed: std::mem::replace(&mut st.data_changed, target.data_changed),
        filename: std::mem::replace(&mut st.filename, target.filename),
    };
    let old_name = std::mem::replace(&mut st.cur_frame, name.to_string());
    st.data_frames.insert(old_name, old);
    Ok(())
}

/// Drop every frame except the current one, which becomes `default`
/// (the caller clears its data).
pub fn reset(st: &mut Store) {
    st.data_frames.clear();
    st.cur_frame = DEFAULT_FRAME.to_string();
}

/// Data of frame `name` (the current dataset when it is the current frame).
fn frame_data<'a>(ds: &'a DataSet, st: &'a Store, name: &str) -> Result<&'a DataSet, String> {
    if name == st.cur_frame {
        Ok(ds)
    } else {
        st.data_frames.get(name).map(|f| &f.data).ok_or_else(|| not_found(name))
    }
}

fn check_new_frame(st: &Store, name: &str) -> Result<(), String> {
    varlist::check_name(name)?;
    if exists(st, name) {
        return Err(rc_err(110, format!("frame {name} already defined")));
    }
    Ok(())
}

fn new_frame(data: DataSet) -> Frame {
    Frame { data, ..Default::default() }
}

/// `frame`/`frames` with a subcommand. The `frame name: command` prefix and
/// `frame name { ... }` blocks are run by the interpreter via [`switch`].
pub struct FrameCmd;

impl CommandHandler for FrameCmd {
    fn name(&self) -> &'static str {
        "frame"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = match main.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (main, ""),
        };
        let words: Vec<&str> = rest.split_whitespace().collect();
        let sub = sub.to_lowercase();
        if let Some(r) = crate::dm_w3::frames_more::frame_sub(&sub, rest, ds, st, cmd) {
            return r;
        }
        match sub.as_str() {
            "" | "pwf" => Ok(pwf(st)),
            "create" | "cre" => {
                let name = *words.first().ok_or_else(|| rc_err(198, "frame name required"))?;
                check_new_frame(st, name)?;
                let data = new_vars(st, &words[1..])?;
                st.data_frames.insert(name.to_string(), new_frame(data));
                Ok(String::new())
            }
            "change" | "cha" => {
                let name = one_name(&words)?;
                switch(ds, st, name)?;
                Ok(String::new())
            }
            "dir" => Ok(dir(ds, st)),
            "drop" => {
                if words.is_empty() {
                    return Err(rc_err(198, "frame name required"));
                }
                for w in &words {
                    if *w == st.cur_frame {
                        return Err(rc_err(110, format!("frame {w} is the current frame; it cannot be dropped")));
                    }
                    if !st.data_frames.contains_key(*w) {
                        return Err(not_found(w));
                    }
                }
                for w in &words {
                    st.data_frames.remove(*w);
                }
                Ok(String::new())
            }
            "rename" | "ren" => {
                let (old, new) = match words.as_slice() {
                    [a, b] => (*a, *b),
                    _ => return Err(rc_err(198, "frame rename oldname newname")),
                };
                if !exists(st, old) {
                    return Err(not_found(old));
                }
                check_new_frame(st, new)?;
                if old == st.cur_frame {
                    st.cur_frame = new.to_string();
                } else {
                    let f = st.data_frames.remove(old).unwrap();
                    st.data_frames.insert(new.to_string(), f);
                }
                Ok(String::new())
            }
            "copy" => {
                let (from, to) = match words.as_slice() {
                    [a, b] => (*a, *b),
                    _ => return Err(rc_err(198, "frame copy frame_from frame_to")),
                };
                let data = frame_data(ds, st, from)?.clone();
                if exists(st, to) {
                    if !cmd.has_opt("replace", 3) {
                        return Err(rc_err(110, format!("frame {to} already defined")));
                    }
                } else {
                    varlist::check_name(to)?;
                }
                if to == st.cur_frame {
                    *ds = data;
                    st.data_changed = true;
                    st.tsset = None;
                } else {
                    st.data_frames.insert(to.to_string(), Frame { data, data_changed: true, ..Default::default() });
                }
                Ok(String::new())
            }
            "reset" => {
                reset(st);
                *ds = DataSet::new();
                st.data_changed = false;
                st.tsset = None;
                st.preserved.clear();
                st.filename.clear();
                Ok(String::new())
            }
            "put" => put(ds, st, cmd, rest),
            other => {
                // `frame name` alone shows nothing useful; treat it as an error
                if exists(st, other) {
                    return Err(rc_err(198, format!("frame {other}: colon or {{ expected")));
                }
                Err(rc_err(198, format!("frame {other}: invalid subcommand")))
            }
        }
    }
}

/// `cwf name`
pub struct Cwf;

impl CommandHandler for Cwf {
    fn name(&self) -> &'static str {
        "cwf"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words: Vec<&str> = cmd.main.split_whitespace().collect();
        switch(ds, st, one_name(&words)?)?;
        Ok(String::new())
    }
}

/// `pwf`
pub struct Pwf;

impl CommandHandler for Pwf {
    fn name(&self) -> &'static str {
        "pwf"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, _cmd: &Command) -> Result<String, String> {
        Ok(pwf(st))
    }
}

fn pwf(st: &Store) -> String {
    format!("  (current frame is {})", st.cur_frame)
}

fn one_name<'a>(words: &[&'a str]) -> Result<&'a str, String> {
    match words {
        [n] => Ok(n),
        [] => Err(rc_err(198, "frame name required")),
        _ => Err(rc_err(198, "too many frame names")),
    }
}

/// Empty dataset with the variables of `frame create name [type] var ...`.
fn new_vars(st: &Store, words: &[&str]) -> Result<DataSet, String> {
    let mut ds = DataSet::new();
    let mut ty = None;
    for w in words {
        if let Some(t) = VarType::from_type_str(w) {
            ty = Some(t);
            continue;
        }
        for name in varlist::new_names(&ds, w)? {
            let mut v = Variable::new(name, ty.unwrap_or(st.default_type));
            if v.var_type.is_string() {
                v.format = format!("%{}s", v.var_type.storage_width().max(9));
            }
            ds.add_variable(v)?;
        }
        ty = None;
    }
    Ok(ds)
}

fn dir(ds: &DataSet, st: &Store) -> String {
    let mut rows: Vec<(&str, &DataSet, bool)> = vec![(st.cur_frame.as_str(), ds, st.data_changed)];
    rows.extend(st.data_frames.iter().map(|(n, f)| (n.as_str(), &f.data, f.data_changed)));
    rows.sort_by(|a, b| a.0.cmp(b.0));
    let w = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let mut out: Vec<String> = rows
        .iter()
        .map(|(n, d, changed)| {
            let mut line = format!("{} {:<w$}  {} x {}", if *changed { "*" } else { " " }, n, d.n_obs, d.n_vars());
            if !d.datalabel.is_empty() {
                line.push_str(&format!("; {}", d.datalabel));
            }
            line
        })
        .collect();
    if rows.iter().any(|r| r.2) {
        out.push(String::new());
        out.push("Note: Frames marked with * contain unsaved data.".into());
    }
    out.join("\n")
}

/// Copy variable `v` restricted to `rows` (None = missing) from `src`.
fn copy_var(src: &DataSet, v: &Variable, name: &str, rows: &[Option<usize>]) -> Variable {
    let mut nv = Variable::new(name, v.var_type);
    nv.format = v.format.clone();
    nv.label = v.label.clone();
    nv.value_label_name = v.value_label_name.clone();
    let miss = if v.var_type.is_string() { Value::Str(String::new()) } else { typed_value(v.var_type, SYSMISS) };
    nv.data = rows
        .iter()
        .map(|r| match r.and_then(|i| v.data.get(i)) {
            Some(Value::StrL(a, b)) => Value::Str(src.resolve_strl(*a, *b).unwrap_or("").to_string()),
            Some(x) => x.clone(),
            None => miss.clone(),
        })
        .collect();
    nv
}

/// Bring the value label of `v` (if any) from `src` into `dst`.
fn copy_value_label(src: &DataSet, dst: &mut DataSet, v: &Variable) {
    if !v.value_label_name.is_empty() && !dst.value_labels.contains_key(&v.value_label_name) {
        if let Some(l) = src.value_labels.get(&v.value_label_name) {
            dst.value_labels.insert(v.value_label_name.clone(), l.clone());
        }
    }
}

/// `frame put varlist [if] [in], into(name)`
fn put(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let into = cmd.opt("into", 4).map(str::trim).ok_or_else(|| rc_err(198, "option into() required"))?.to_string();
    check_new_frame(st, &into)?;
    let vars = if rest.is_empty() { ds.varlist() } else { varlist::expand(ds, rest)? };
    let mask = touse(ds, st, cmd)?;
    let rows: Vec<Option<usize>> = (0..ds.n_obs).filter(|&i| mask[i]).map(Some).collect();
    let mut out = DataSet::new();
    for name in &vars {
        let v = ds.get_var(name).unwrap();
        out.add_variable(copy_var(ds, v, name, &rows))?;
        copy_value_label(ds, &mut out, v);
        if let Some(c) = ds.characteristics.get(name) {
            out.characteristics.insert(name.clone(), c.clone());
        }
    }
    out.n_obs = rows.len();
    st.data_frames.insert(into, Frame { data: out, data_changed: true, ..Default::default() });
    Ok(String::new())
}

fn key_of(d: &DataSet, keys: &[String], i: usize) -> Vec<String> {
    keys.iter()
        .map(|k| {
            let v = d.get_var(k).unwrap();
            match v.data.get(i) {
                Some(Value::Str(s)) => format!("s{s}"),
                Some(Value::StrL(a, b)) => format!("s{}", d.resolve_strl(*a, *b).unwrap_or("")),
                Some(x) => format!("n{:e}", x.num()),
                None => "n.".into(),
            }
        })
        .collect()
}

fn not_unique(keys: &[String], frame: &str) -> String {
    let s = keys.len() > 1;
    rc_err(
        459,
        format!("variable{} {} do{} not uniquely identify observations in frame {frame}", if s { "s" } else { "" }, keys.join(" "), if s { "" } else { "es" }),
    )
}

/// Observation numbers (1-based) in `target` matching each observation of
/// `ds`; checks the key types and uniqueness required by `kind`.
fn link_rows(ds: &DataSet, st: &Store, kind: &str, mkeys: &[String], target: &str, ukeys: &[String]) -> Result<Vec<Option<usize>>, String> {
    let tds = frame_data(ds, st, target)?;
    if mkeys.len() != ukeys.len() {
        return Err(rc_err(198, "different numbers of key variables in the two frames"));
    }
    for (m, u) in mkeys.iter().zip(ukeys) {
        let mv = ds.get_var(m).ok_or_else(|| rc_err(111, format!("variable {m} not found")))?;
        let uv = tds.get_var(u).ok_or_else(|| rc_err(111, format!("variable {u} not found in frame {target}")))?;
        if mv.var_type.is_string() != uv.var_type.is_string() {
            return Err(rc_err(106, format!("key variable {m} is str in one frame and numeric in the other")));
        }
    }
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    for j in 0..tds.n_obs {
        if index.insert(key_of(tds, ukeys, j), j).is_some() {
            return Err(not_unique(ukeys, target));
        }
    }
    if kind == "1:1" {
        let mut seen = std::collections::HashSet::new();
        for i in 0..ds.n_obs {
            if !seen.insert(key_of(ds, mkeys, i)) {
                return Err(not_unique(mkeys, &st.cur_frame));
            }
        }
    }
    Ok((0..ds.n_obs).map(|i| index.get(&key_of(ds, mkeys, i)).copied()).collect())
}

fn link_var(name: &str, rows: &[Option<usize>]) -> Variable {
    let mut v = Variable::new(name, VarType::Long);
    v.data = rows.iter().map(|r| typed_value(VarType::Long, r.map_or(SYSMISS, |j| (j + 1) as f64))).collect();
    v
}

fn match_report(rows: &[Option<usize>], frame: &str) -> String {
    let un = rows.iter().filter(|r| r.is_none()).count();
    match un {
        0 => format!("  (all observations in frame {frame} matched)"),
        1 => format!("  (1 observation in frame {frame} unmatched)"),
        n => format!("  ({n} observations in frame {frame} unmatched)"),
    }
}

/// `frlink 1:1|m:1 keys, frame(name [keys2]) [generate(var)]`,
/// `frlink rebuild linkvar`, `frlink dir`, `frlink describe linkvar`.
pub struct Frlink;

impl CommandHandler for Frlink {
    fn name(&self) -> &'static str {
        "frlink"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (first, rest) = match main.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (main, ""),
        };
        let kind = first.replace(' ', "").to_lowercase();
        match kind.as_str() {
            "dir" => {
                let links: Vec<String> = ds
                    .varlist()
                    .into_iter()
                    .filter_map(|v| ds.characteristics.get(&v).and_then(|c| c.get(CH_FRAME)).map(|f| format!("  {v} links to frame {f}")))
                    .collect();
                return Ok(if links.is_empty() { "  (no links found)".into() } else { links.join("\n") });
            }
            "describe" | "rebuild" => {
                let lv = varlist::resolve_name(ds, rest)?;
                let ch = ds.characteristics.get(&lv).filter(|c| c.contains_key(CH_FRAME)).cloned().ok_or_else(|| rc_err(111, format!("variable {lv} is not a link variable")))?;
                let target = ch[CH_FRAME].clone();
                let mkeys: Vec<String> = ch.get(CH_MKEYS).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
                let ukeys: Vec<String> = ch.get(CH_UKEYS).map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
                let lk = ch.get(CH_KIND).cloned().unwrap_or_else(|| "m:1".into());
                if kind == "describe" {
                    return Ok(format!(
                        "  {lv} created by frlink {lk} {}, frame({target} {})\n  Observations in frame {} link to frame {target}.",
                        mkeys.join(" "),
                        ukeys.join(" "),
                        st.cur_frame
                    ));
                }
                let rows = link_rows(ds, st, &lk, &mkeys, &target, &ukeys)?;
                let v = link_var(&lv, &rows);
                ds.get_var_mut(&lv).unwrap().data = v.data;
                st.data_changed = true;
                return Ok(match_report(&rows, &st.cur_frame));
            }
            "1:1" | "m:1" => {}
            _ => return Err(rc_err(198, format!("{first}: invalid link type; use 1:1 or m:1"))),
        }
        let mkeys = varlist::expand(ds, rest)?;
        if mkeys.is_empty() {
            return Err(rc_err(102, "too few variables specified"));
        }
        let spec = cmd.opt("frame", 2).ok_or_else(|| rc_err(198, "option frame() required"))?;
        let mut sw = spec.split_whitespace();
        let target = sw.next().ok_or_else(|| rc_err(198, "option frame() required"))?.to_string();
        if target == st.cur_frame {
            return Err(rc_err(498, format!("frame {target} is the current frame; it cannot be linked to itself")));
        }
        let tds = frame_data(ds, st, &target)?;
        let given: Vec<&str> = sw.collect();
        let ukeys = if given.is_empty() { mkeys.clone() } else { varlist::expand(tds, &given.join(" "))? };
        let lv = cmd.opt("generate", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| target.clone());
        varlist::new_names(ds, &lv)?;
        let rows = link_rows(ds, st, &kind, &mkeys, &target, &ukeys)?;
        let v = link_var(&lv, &rows);
        ds.add_variable(v)?;
        let ch = ds.characteristics.entry(lv).or_default();
        ch.insert(CH_FRAME.into(), target);
        ch.insert(CH_KIND.into(), kind);
        ch.insert(CH_MKEYS.into(), mkeys.join(" "));
        ch.insert(CH_UKEYS.into(), ukeys.join(" "));
        st.data_changed = true;
        Ok(match_report(&rows, &st.cur_frame))
    }
}

/// Target frame and observation numbers of link variable `lv`.
fn link_of(ds: &DataSet, lv: &str) -> Result<(String, Vec<Option<usize>>), String> {
    let lv = varlist::resolve_name(ds, lv)?;
    let target = ds
        .characteristics
        .get(&lv)
        .and_then(|c| c.get(CH_FRAME))
        .cloned()
        .ok_or_else(|| rc_err(111, format!("variable {lv} is not a link variable")))?;
    let v = ds.get_var(&lv).unwrap();
    let rows = (0..ds.n_obs).map(|i| {
        let x = v.num(i);
        (!ferrostat_core::missing::is_missing(x) && x >= 1.0).then(|| x as usize - 1)
    });
    Ok((target, rows.collect()))
}

/// `frget varlist, from(linkvar) [prefix() suffix() exclude()]` and
/// `frget newvar = var, from(linkvar)`.
pub struct Frget;

impl CommandHandler for Frget {
    fn name(&self) -> &'static str {
        "frget"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let from = cmd.opt("from", 4).map(str::trim).ok_or_else(|| rc_err(198, "option from() required"))?;
        let (target, rows) = link_of(ds, from)?;
        let tds = frame_data(ds, st, &target)?;
        let rows: Vec<Option<usize>> = rows.into_iter().map(|r| r.filter(|&j| j < tds.n_obs)).collect();
        let main = cmd.main.trim();
        // (new name, source name)
        let pairs: Vec<(String, String)> = if let Some((new, src)) = main.split_once('=') {
            let src = varlist::resolve_name(tds, src.trim())?;
            vec![(new.trim().to_string(), src)]
        } else {
            let mut src = varlist::expand(tds, main)?;
            if main == "*" || main == "_all" {
                // the key variables are already in the current frame
                let keys: Vec<String> = ds
                    .characteristics
                    .values()
                    .filter(|c| c.get(CH_FRAME) == Some(&target))
                    .flat_map(|c| c.get(CH_UKEYS).map(|k| k.split_whitespace().map(String::from).collect::<Vec<_>>()).unwrap_or_default())
                    .collect();
                src.retain(|v| !keys.contains(v));
            }
            if let Some(ex) = cmd.opt("exclude", 2) {
                let ex = varlist::expand(tds, ex)?;
                src.retain(|v| !ex.contains(v));
            }
            let pre = cmd.opt("prefix", 3).unwrap_or("").trim();
            let suf = cmd.opt("suffix", 3).unwrap_or("").trim();
            src.into_iter().map(|s| (format!("{pre}{s}{suf}"), s)).collect()
        };
        let newnames: Vec<String> = pairs.iter().map(|p| p.0.clone()).collect();
        varlist::new_names(ds, &newnames.join(" "))?;
        let mut vars = Vec::new();
        let mut out = Vec::new();
        for (new, src) in &pairs {
            let v = tds.get_var(src).unwrap();
            let nv = copy_var(tds, v, new, &rows);
            let miss = nv.data.iter().filter(|x| x.is_missing() || matches!(x, Value::Str(s) if s.is_empty())).count();
            if miss > 0 {
                out.push(format!("({} missing value{} generated)", miss, if miss == 1 { "" } else { "s" }));
            }
            vars.push((nv, v.clone()));
        }
        let mut labels = BTreeMap::new();
        for (_, v) in &vars {
            if !v.value_label_name.is_empty() && !ds.value_labels.contains_key(&v.value_label_name) {
                if let Some(l) = tds.value_labels.get(&v.value_label_name) {
                    labels.insert(v.value_label_name.clone(), l.clone());
                }
            }
        }
        let k = vars.len();
        for (nv, _) in vars {
            ds.add_variable(nv)?;
        }
        ds.value_labels.extend(labels);
        st.data_changed = true;
        out.push(format!("({k} variable{} copied from linked frame)", if k == 1 { "" } else { "s" }));
        Ok(out.join("\n"))
    }
}

/// `frval(linkvar, varname)`: value of `varname` in the linked frame.
pub fn frval(env: &Env, link: &str, var: &str, row: usize) -> Result<Val, String> {
    let st = env.st.ok_or_else(|| rc_err(133, "frval() not available here"))?;
    let lv = varlist::resolve_name(env.ds, link)?;
    let target = env
        .ds
        .characteristics
        .get(&lv)
        .and_then(|c| c.get(CH_FRAME))
        .ok_or_else(|| rc_err(111, format!("variable {lv} is not a link variable")))?;
    let tds = st.data_frames.get(target).map(|f| &f.data).ok_or_else(|| not_found(target))?;
    let src = varlist::resolve_name(tds, var)?;
    let v = tds.get_var(&src).unwrap();
    let x = env.ds.get_var(&lv).unwrap().num(row);
    let j = (!ferrostat_core::missing::is_missing(x) && x >= 1.0).then(|| x as usize - 1);
    Ok(match j.and_then(|j| v.data.get(j)) {
        Some(Value::Str(s)) => Val::Str(s.clone()),
        Some(Value::StrL(a, b)) => Val::Str(tds.resolve_strl(*a, *b).unwrap_or("").to_string()),
        Some(x) if !v.var_type.is_string() => Val::Num(x.num()),
        _ if v.var_type.is_string() => Val::Str(String::new()),
        _ => Val::Num(SYSMISS),
    })
}
