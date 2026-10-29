//! Frame sets on disk (`frames save`, `frames use`, `frames describe`,
//! `frames modify`), `frame put label`, and alias variables (`fralias`,
//! `frunalias`).
//!
//! A frame set file (`.dtas`) is a small container: a magic line, the number
//! of frames (u32 little endian), then for each frame the name length (u32),
//! the name and the length (u64) of a complete `.dta` image of the frame.
//!
//! Alias variables are materialized: `fralias add` copies the linked values
//! (like `frget`) and records the link in the variable's characteristics so
//! that `fralias describe` can report it; `frunalias` removes the record.

use ferrostat_core::dta::{DtaReader, DtaWriter};
use ferrostat_core::{vfs, DataSet};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::frames::Frame;
use crate::store::Store;
use crate::util;

const FRAME_MAGIC: &[u8] = b"FERROSTAT-FRAMES\x01\n";
const CH_ALIAS: &str = "fralias_of";

/// Subcommands of `frame`/`frames` handled here; `None` for the others.
pub fn frame_sub(sub: &str, rest: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "describe" | "des" | "desc" | "descr" | "descri" | "describ" => Some(describe(ds, st, cmd, rest)),
        "save" => Some(save(ds, st, cmd, rest)),
        "use" => Some(use_(ds, st, cmd, rest)),
        "modify" => Some(modify(ds, st, cmd, rest)),
        "put" if is_put_label(ds, rest) => Some(put_label(ds, st, cmd, rest)),
        _ => None,
    }
}

fn file_arg(cmd: &Command, rest: &str) -> String {
    let p = cmd.using.clone().unwrap_or_else(|| util::unquote(rest.split(',').next().unwrap_or("").trim()));
    let p = p.trim().to_string();
    if p.is_empty() || p.rsplit('/').next().unwrap_or("").contains('.') {
        p
    } else {
        format!("{p}.dtas")
    }
}

fn dta_bytes(d: &DataSet) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    DtaWriter::write_release(&mut buf, d, 118).map_err(|e| rc_err(610, format!("{e}")))?;
    Ok(buf)
}

fn encode(frames: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut buf = FRAME_MAGIC.to_vec();
    buf.extend((frames.len() as u32).to_le_bytes());
    for (n, b) in frames {
        buf.extend((n.len() as u32).to_le_bytes());
        buf.extend(n.as_bytes());
        buf.extend((b.len() as u64).to_le_bytes());
        buf.extend(b);
    }
    buf
}

/// Frames of a set file as (name, raw .dta bytes), in file order.
fn decode(path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let data = vfs::read(path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    if !data.starts_with(FRAME_MAGIC) {
        return Err(rc_err(610, format!("file {path} is not a frame set file")));
    }
    let bad = || rc_err(610, format!("file {path} is damaged"));
    let mut p = FRAME_MAGIC.len();
    let mut take = |n: usize| -> Result<&[u8], String> {
        let s = data.get(p..p + n).ok_or_else(bad)?;
        p += n;
        Ok(s)
    };
    let count = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
    let mut out = Vec::new();
    for _ in 0..count {
        let nl = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let name = String::from_utf8_lossy(take(nl)?).to_string();
        let dl = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        out.push((name, take(dl)?.to_vec()));
    }
    Ok(out)
}

fn names_opt(cmd: &Command) -> Option<Vec<String>> {
    cmd.opt("frames", 3).map(|l| l.split_whitespace().map(String::from).collect())
}

/// All frames in memory: the current one first, then the others by name.
fn in_memory<'a>(ds: &'a DataSet, st: &'a Store) -> Vec<(String, &'a DataSet)> {
    let mut v = vec![(st.cur_frame.clone(), ds)];
    v.extend(st.data_frames.iter().map(|(n, f)| (n.clone(), &f.data)));
    v
}

fn select<'a>(all: Vec<(String, &'a DataSet)>, want: &Option<Vec<String>>) -> Result<Vec<(String, &'a DataSet)>, String> {
    match want {
        None => Ok(all),
        Some(w) if w.len() == 1 && w[0] == "_all" => Ok(all),
        Some(w) => {
            let mut out = Vec::new();
            for n in w {
                let f = all.iter().find(|f| &f.0 == n).ok_or_else(|| rc_err(111, format!("frame {n} not found")))?;
                out.push((f.0.clone(), f.1));
            }
            Ok(out)
        }
    }
}

fn describe_one(out: &mut Vec<String>, name: &str, d: &DataSet, current: bool, cmd: &Command) {
    out.push(format!("  Frame {name}{}", if current { " (current frame)" } else { "" }));
    out.push(format!("    Observations: {:>10}", d.n_obs));
    out.push(format!("       Variables: {:>10}", d.n_vars()));
    if !d.datalabel.is_empty() {
        out.push(format!("      Data label: {}", d.datalabel));
    }
    if !cmd.has_opt("short", 1) && d.n_vars() > 0 {
        if cmd.has_opt("simple", 2) {
            out.push(format!("    {}", d.varlist().join(" ")));
        } else {
            for (k, v) in d.vars.iter().enumerate() {
                let num = if cmd.has_opt("numbers", 1) { format!("{:>4}. ", k + 1) } else { String::new() };
                let name = if cmd.has_opt("fullnames", 1) || v.name.chars().count() <= 15 { v.name.clone() } else { format!("{}~{}", &v.name[..13], v.name.chars().last().unwrap()) };
                out.push(format!("    {num}{:<16}{:<8}{:<10}{}", name, v.var_type.type_str(), v.format, v.label));
            }
        }
    }
    out.push(String::new());
}

fn describe(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let mut out = Vec::new();
    let mut names = Vec::new();
    if cmd.using.is_some() {
        let path = file_arg(cmd, "");
        let frames = decode(&path)?;
        let want = names_opt(cmd);
        out.push(format!("  Frame set {path}"));
        out.push(String::new());
        for (n, b) in frames {
            if want.as_ref().is_some_and(|w| !w.contains(&n)) {
                continue;
            }
            let d = DtaReader::from_bytes(b).map_err(|e| rc_err(610, format!("{e}")))?;
            describe_one(&mut out, &n, &d, false, cmd);
            names.push(n);
        }
    } else {
        let mut want = names_opt(cmd);
        let words: Vec<String> = rest.split(',').next().unwrap_or("").split_whitespace().map(String::from).collect();
        if !words.is_empty() {
            want = Some(words);
        }
        let mut all = in_memory(ds, st);
        all.sort_by(|a, b| a.0.cmp(&b.0));
        for (n, d) in select(all, &want)? {
            describe_one(&mut out, &n, d, n == st.cur_frame, cmd);
            names.push(n);
        }
    }
    st.begin_r().set_str("frames", names.join(" "));
    Ok(out.join("\n").trim_end().to_string())
}

fn save(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let path = file_arg(cmd, rest);
    if path.is_empty() {
        return Err(rc_err(100, "filename required"));
    }
    if vfs::exists(&path) && !cmd.has_opt("replace", 3) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let want = names_opt(cmd);
    let mut all = in_memory(ds, st);
    let cur = all.remove(0);
    all.sort_by(|a, b| a.0.cmp(&b.0));
    all.insert(0, cur);
    let chosen = select(all, &want)?;
    let mut frames = Vec::new();
    for (n, d) in &chosen {
        frames.push((n.clone(), dta_bytes(d)?));
    }
    vfs::write(&path, &encode(&frames)).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
    let list: Vec<String> = chosen.iter().map(|c| c.0.clone()).collect();
    let r = st.begin_r();
    r.set_str("frames", list.join(" "));
    r.set_str("fn", &path);
    Ok(format!("file {path} saved"))
}

fn use_(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let mut path = file_arg(cmd, rest);
    if !vfs::exists(&path) && vfs::exists(&format!("{path}.dtas")) {
        path = format!("{path}.dtas");
    }
    let mut loaded = decode(&path)?;
    if let Some(w) = names_opt(cmd) {
        for n in &w {
            if !loaded.iter().any(|l| &l.0 == n) {
                return Err(rc_err(111, format!("frame {n} not found in {path}")));
            }
        }
        loaded.retain(|l| w.contains(&l.0));
        loaded.sort_by_key(|l| w.iter().position(|x| x == &l.0));
    }
    let clear = cmd.has_opt("clear", 5);
    let replace = cmd.has_opt("replace", 3);
    let in_mem_changed = st.data_changed && ds.n_vars() > 0;
    if !clear {
        for (n, _) in &loaded {
            let exists = n == &st.cur_frame || st.data_frames.contains_key(n);
            let empty_default = n == &st.cur_frame && ds.n_vars() == 0 && ds.n_obs == 0;
            if exists && !empty_default && !replace {
                return Err(rc_err(110, format!("frame {n} already exists; specify replace or clear")));
            }
        }
        if in_mem_changed && loaded.iter().any(|l| l.0 == st.cur_frame) && !replace {
            return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
        }
    }
    let mut sets = Vec::new();
    for (n, b) in loaded {
        sets.push((n, DtaReader::from_bytes(b).map_err(|e| rc_err(610, format!("{e}")))?));
    }
    let names: Vec<String> = sets.iter().map(|s| s.0.clone()).collect();
    if clear {
        crate::frames::reset(st);
        *ds = DataSet::new();
        st.tsset = None;
        st.data_changed = false;
        st.preserved.clear();
        st.filename.clear();
        // the first frame loaded becomes the working frame
        if let Some(first) = names.first() {
            st.cur_frame = first.clone();
        }
    }
    for (n, d) in sets {
        if n == st.cur_frame {
            *ds = d;
            st.data_changed = false;
            st.tsset = None;
            st.filename = path.clone();
        } else {
            st.data_frames.insert(n, Frame { data: d, filename: path.clone(), ..Default::default() });
        }
    }
    let r = st.begin_r();
    r.set_str("frames", names.join(" "));
    r.set_str("fn", &path);
    Ok(String::new())
}

/// `frames modify filename, add(framelist|_all) [replace] | drop(framelist)`
fn modify(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let path = file_arg(cmd, rest);
    let mut frames = decode(&path)?;
    let add = cmd.opt("add", 3).map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>());
    let drop = cmd.opt("drop", 4).map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>());
    let (added, dropped);
    match (add, drop) {
        (Some(a), None) => {
            let chosen = select(in_memory(ds, st), &Some(a))?;
            let mut names = Vec::new();
            for (n, d) in chosen {
                if d.n_vars() == 0 && d.n_obs == 0 && !cmd.has_opt("emptyok", 6) {
                    return Err(rc_err(198, format!("frame {n} is empty; specify emptyok to add it")));
                }
                let mut d = d.clone();
                if cmd.has_opt("nolabel", 5) {
                    d.value_labels.clear();
                    for v in d.vars.iter_mut() {
                        v.value_label_name.clear();
                    }
                }
                let bytes = dta_bytes(&d)?;
                match frames.iter_mut().find(|f| f.0 == n) {
                    Some(f) if cmd.has_opt("replace", 3) => f.1 = bytes,
                    Some(_) => return Err(rc_err(110, format!("frame {n} already exists in {path}; specify replace"))),
                    None => frames.push((n.clone(), bytes)),
                }
                names.push(n);
            }
            added = names;
            dropped = Vec::new();
        }
        (None, Some(dl)) => {
            for n in &dl {
                if !frames.iter().any(|f| &f.0 == n) {
                    return Err(rc_err(111, format!("frame {n} not found in {path}")));
                }
            }
            frames.retain(|f| !dl.contains(&f.0));
            added = Vec::new();
            dropped = dl;
        }
        _ => return Err(rc_err(198, "one of add() or drop() required")),
    }
    vfs::write(&path, &encode(&frames)).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
    let all: Vec<String> = frames.iter().map(|f| f.0.clone()).collect();
    let r = st.begin_r();
    r.set_str("fn", &path);
    r.set_str("frames", all.join(" "));
    r.set_str("added", added.join(" "));
    r.set_str("dropped", dropped.join(" "));
    Ok(format!("file {path} modified ({} frame{} in set)", all.len(), if all.len() == 1 { "" } else { "s" }))
}

/// `frame put label` is taken as copying value labels unless a variable is
/// called `label`.
fn is_put_label(ds: &DataSet, rest: &str) -> bool {
    let first = rest.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("");
    first == "label" && ds.get_var("label").is_none()
}

/// `frame put label [lblnamelist], into(framename) [replace]`: copy value-label
/// definitions to another (existing or new) frame.
fn put_label(ds: &DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let into = cmd.opt("into", 4).map(str::trim).ok_or_else(|| rc_err(198, "option into() required"))?.to_string();
    let names: Vec<String> = rest["label".len()..].split(',').next().unwrap_or("").split_whitespace().map(String::from).collect();
    let names: Vec<String> = if names.is_empty() || names == ["_all"] { ds.value_labels.keys().cloned().collect() } else { names };
    for n in &names {
        if !ds.value_labels.contains_key(n) {
            return Err(rc_err(111, format!("value label {n} not found")));
        }
    }
    if into == st.cur_frame {
        return Err(rc_err(198, "into() must name another frame"));
    }
    if !st.data_frames.contains_key(&into) {
        crate::varlist::check_name(&into)?;
        st.data_frames.insert(into.clone(), Frame::default());
    }
    let target = &mut st.data_frames.get_mut(&into).unwrap().data;
    for n in &names {
        if target.value_labels.contains_key(n) && !cmd.has_opt("replace", 3) {
            return Err(rc_err(110, format!("value label {n} already defined in frame {into}; specify replace")));
        }
        target.value_labels.insert(n.clone(), ds.value_labels[n].clone());
    }
    Ok(format!("({} value label{} copied to frame {into})", names.len(), if names.len() == 1 { "" } else { "s" }))
}

/// `fralias add ..., from(linkvar)` and `fralias describe [varlist]`.
pub struct Fralias;

impl CommandHandler for Fralias {
    fn name(&self) -> &'static str {
        "fralias"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        match sub {
            "add" => {
                let link = cmd.opt("from", 4).map(str::trim).ok_or_else(|| rc_err(198, "option from() required"))?.to_string();
                let before: Vec<String> = ds.varlist();
                // copy the values through frget, then record the alias
                let mut c = cmd.clone();
                c.main = rest.trim().to_string();
                c.verb = "frget".into();
                let out = crate::frames::Frget.execute(ds, st, &c)?;
                let target = ds.characteristics.get(&link).and_then(|ch| ch.get("frlink_frame")).cloned().unwrap_or_default();
                // source names: `new = src` pairs or the same names with prefix/suffix
                let pairs: Vec<(String, String)> = if let Some((n, s)) = rest.split_once('=') {
                    vec![(n.trim().to_string(), s.trim().to_string())]
                } else {
                    let pre = cmd.opt("prefix", 3).unwrap_or("").trim().to_string();
                    let suf = cmd.opt("suffix", 3).unwrap_or("").trim().to_string();
                    ds.varlist()
                        .into_iter()
                        .filter(|v| !before.contains(v))
                        .map(|v| {
                            let src = v.strip_prefix(pre.as_str()).unwrap_or(&v);
                            let src = src.strip_suffix(suf.as_str()).unwrap_or(src).to_string();
                            (v, src)
                        })
                        .collect()
                };
                for (n, s) in pairs {
                    ds.characteristics.entry(n).or_default().insert(CH_ALIAS.into(), format!("{link} {target} {s}"));
                }
                Ok(out.replace("copied from linked frame", "aliased from linked frame"))
            }
            "describe" | "des" | "desc" => {
                let vars = if rest.trim().is_empty() { ds.varlist() } else { crate::varlist::expand(ds, rest)? };
                let mut out = vec![format!("  {:<16}{:<12}{:<12}{}", "Variable", "Link", "Frame", "Target variable"), format!("  {}", "-".repeat(56))];
                let mut n = 0;
                for v in vars {
                    if let Some(a) = ds.characteristics.get(&v).and_then(|c| c.get(CH_ALIAS)) {
                        let w: Vec<&str> = a.split_whitespace().collect();
                        out.push(format!("  {:<16}{:<12}{:<12}{}", v, w.first().unwrap_or(&""), w.get(1).unwrap_or(&""), w.get(2).unwrap_or(&"")));
                        n += 1;
                    }
                }
                if n == 0 {
                    return Ok("  (no alias variables)".into());
                }
                Ok(out.join("\n"))
            }
            "" => Err(rc_err(198, "fralias: subcommand required")),
            o => Err(rc_err(198, format!("fralias {o}: invalid subcommand"))),
        }
    }
}

/// `frunalias [varlist]`: turn alias variables into ordinary variables.
pub struct Frunalias;

impl CommandHandler for Frunalias {
    fn name(&self) -> &'static str {
        "frunalias"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { crate::varlist::expand(ds, &cmd.main)? };
        let mut n = 0;
        for v in vars {
            if let Some(c) = ds.characteristics.get_mut(&v) {
                if c.remove(CH_ALIAS).is_some() {
                    n += 1;
                }
            }
        }
        if n > 0 {
            st.data_changed = true;
        }
        Ok(format!("({n} alias variable{} converted)", if n == 1 { "" } else { "s" }))
    }
}
