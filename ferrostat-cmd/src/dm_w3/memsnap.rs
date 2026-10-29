//! `memory` (memory-usage report), `snapshot` (in-session copies of the data)
//! and `sysuse` (example datasets).

use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;

/// A saved copy of the data (`snapshot save`).
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub label: String,
    pub time: String,
    pub data: DataSet,
}

/// Bytes used by the observations of a dataset.
fn data_bytes(ds: &DataSet) -> (u64, u64) {
    let mut data = 0u64;
    let mut strl = 0u64;
    for v in &ds.vars {
        if v.var_type.is_string() {
            for x in &v.data {
                match x {
                    Value::Str(s) if v.var_type == ferrostat_core::VarType::StrL => strl += s.len() as u64 + 8,
                    _ => data += v.var_type.storage_width().max(1) as u64,
                }
            }
        } else {
            data += (v.var_type.storage_width() * ds.n_obs) as u64;
        }
    }
    strl += ds.strls.values().map(|s| s.len() as u64 + 8).sum::<u64>();
    (data, strl)
}

fn meta_bytes(ds: &DataSet) -> (u64, u64) {
    let names: u64 = ds.vars.iter().map(|v| (v.name.len() + v.label.len() + v.format.len() + v.value_label_name.len() + 16) as u64).sum();
    let labels: u64 = ds.value_labels.values().map(|l| l.mapping.values().map(|s| s.len() as u64 + 12).sum::<u64>() + 32).sum();
    (names, labels)
}

pub struct Memory;

impl CommandHandler for Memory {
    fn name(&self) -> &'static str {
        "memory"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, _cmd: &Command) -> Result<String, String> {
        let (data, strl) = data_bytes(ds);
        let (names, vlabels) = meta_bytes(ds);
        let other_frames: u64 = st.data_frames.values().map(|f| data_bytes(&f.data).0 + data_bytes(&f.data).1 + meta_bytes(&f.data).0 + meta_bytes(&f.data).1).sum();
        let preserved: u64 = st.preserved.iter().map(|d| data_bytes(d).0 + data_bytes(d).1).sum();
        let snaps: u64 = st.snapshots.iter().map(|s| data_bytes(&s.data).0 + data_bytes(&s.data).1).sum();
        let matrices: u64 = st.matrices.values().map(|m| (m.rows * m.cols * 8) as u64).sum::<u64>() + st.scalars.len() as u64 * 8;
        let macros: u64 = st.globals.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum();
        let rows: Vec<(&str, &str, u64)> = vec![
            ("data", "Data", data),
            ("strl", "strLs", strl),
            ("vnames", "Variable names, labels and formats", names),
            ("vlabels", "Value labels", vlabels),
            ("frames", "Other frames", other_frames),
            ("preserve", "Preserved copies of the data", preserved),
            ("snapshots", "Snapshots", snaps),
            ("matrices", "Matrices and scalars", matrices),
            ("macros", "Global macros", macros),
        ];
        let total: u64 = rows.iter().map(|r| r.2).sum();
        let mut out = vec![
            "Memory usage".to_string(),
            format!("{:<44}{:>18}", "", "Used (bytes)"),
            "-".repeat(62),
        ];
        for (_, label, b) in &rows {
            out.push(format!("{label:<44}{:>18}", util::commas(*b as usize)));
        }
        out.push("-".repeat(62));
        out.push(format!("{:<44}{:>18}", "Total", util::commas(total as usize)));
        out.push(String::new());
        out.push(format!("  {} observations, {} variables in the current frame", ds.n_obs, ds.n_vars()));
        let max_mem = st.settings.get("max_memory").cloned().unwrap_or_else(|| ".".into());
        let seg = st.settings.get("segmentsize").cloned().unwrap_or_else(|| "32m".into());
        out.push(format!("  max_memory: {max_mem}    segmentsize: {seg}"));
        let r = st.begin_r();
        for (k, _, b) in &rows {
            r.set(&format!("{k}_u"), *b as f64);
        }
        r.set("total_u", total as f64);
        Ok(out.join("\n"))
    }
}

pub struct SnapshotCmd;

fn now() -> String {
    chrono::Local::now().format("%e %b %Y %H:%M").to_string().trim().to_string()
}

fn snap_numbers(st: &Store, text: &str) -> Result<Vec<usize>, String> {
    let t = text.trim();
    if t.is_empty() || t == "_all" {
        return Ok((1..=st.snapshots.len()).filter(|&i| !st.snapshots[i - 1].time.is_empty()).collect());
    }
    let mut out = Vec::new();
    for w in t.split(|c: char| c.is_whitespace() || c == ',') .filter(|w| !w.is_empty()) {
        let (a, b) = match w.split_once('/') {
            Some((a, b)) => (a, b),
            None => (w, w),
        };
        let a: usize = a.parse().map_err(|_| rc_err(198, format!("{w}: invalid snapshot number")))?;
        let b: usize = b.parse().map_err(|_| rc_err(198, format!("{w}: invalid snapshot number")))?;
        for k in a..=b {
            if k == 0 || k > st.snapshots.len() || st.snapshots[k - 1].time.is_empty() {
                return Err(rc_err(198, format!("snapshot {k} not found")));
            }
            out.push(k);
        }
    }
    Ok(out)
}

impl CommandHandler for SnapshotCmd {
    fn name(&self) -> &'static str {
        "snapshot"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "save" => {
                let label = cmd.opt("label", 3).map(util::unquote).unwrap_or_default();
                st.snapshots.push(Snapshot { label, time: now(), data: ds.clone() });
                let n = st.snapshots.len();
                st.begin_r().set("snapshot", n as f64);
                Ok(format!("snapshot {n} created at {}", st.snapshots[n - 1].time))
            }
            "label" => {
                let (num, text) = rest.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "snapshot label # \"label\""))?;
                let k = snap_numbers(st, num)?;
                st.snapshots[k[0] - 1].label = util::unquote(text.trim());
                Ok(String::new())
            }
            "restore" => {
                let k = snap_numbers(st, rest)?;
                let [k] = k.as_slice() else {
                    return Err(rc_err(198, "one snapshot number required"));
                };
                *ds = st.snapshots[k - 1].data.clone();
                st.data_changed = true;
                Ok(String::new())
            }
            "list" => {
                let ks = snap_numbers(st, rest)?;
                if ks.is_empty() {
                    return Ok("  (no snapshots)".into());
                }
                Ok(ks
                    .iter()
                    .map(|&k| {
                        let s = &st.snapshots[k - 1];
                        format!("  snapshot {k} ({}){}", s.time, if s.label.is_empty() { String::new() } else { format!(": {}", s.label) })
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "erase" => {
                if rest.is_empty() {
                    return Err(rc_err(198, "snapshot numbers or _all required"));
                }
                for k in snap_numbers(st, rest)? {
                    let s = &mut st.snapshots[k - 1];
                    s.time.clear();
                    s.label.clear();
                    s.data = DataSet::new();
                }
                while st.snapshots.last().is_some_and(|s| s.time.is_empty()) {
                    st.snapshots.pop();
                }
                Ok(String::new())
            }
            "" => Err(rc_err(198, "snapshot: subcommand required")),
            o => Err(rc_err(198, format!("snapshot {o}: invalid subcommand"))),
        }
    }
}

/// `sysuse name [, clear]` and `sysuse dir [, all]`.
pub fn sysuse(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let name = util::unquote(cmd.main.trim()).trim_end_matches(".dta").to_string();
    let dirs = ["/usr/local/share/ferrostat", "."];
    if name == "dir" || name.is_empty() {
        let mut names: Vec<String> = super::sysdata::NAMES.iter().map(|(n, _)| n.to_string()).collect();
        for d in &dirs[..1] {
            if let Ok(list) = ferrostat_core::vfs::list(d) {
                for (f, is_dir) in list {
                    let f = f.rsplit('/').next().unwrap_or(&f).to_string();
                    if let Some(n) = f.strip_suffix(".dta") {
                        if !is_dir && !names.iter().any(|x| x == n) {
                            names.push(n.to_string());
                        }
                    }
                }
            }
        }
        names.sort();
        st.begin_r().set_str("files", names.iter().map(|n| format!("{n}.dta")).collect::<Vec<_>>().join(" "));
        let files: Vec<String> = names.iter().map(|n| format!("{n}.dta")).collect();
        return Ok(crate::data_utils::columns(&files));
    }
    if st.data_changed && ds.n_vars() > 0 && !cmd.has_opt("clear", 5) {
        return Err(rc_err(4, "no; dataset in memory has changed since last saved"));
    }
    let loaded = match super::sysdata::build(&name) {
        Some(d) => Some((d, format!("{name}.dta"))),
        None => dirs.iter().find_map(|d| {
            let p = format!("{d}/{name}.dta");
            ferrostat_core::dta::DtaReader::open(&p).ok().map(|ds| (ds, p))
        }),
    };
    let (d, path) = loaded.ok_or_else(|| rc_err(601, format!("file \"{name}.dta\" not found")))?;
    *ds = d;
    st.data_changed = false;
    st.tsset = None;
    st.filename = path;
    let label = if ds.datalabel.is_empty() { String::new() } else { format!(" ({})", ds.datalabel) };
    Ok(format!("(example data {name}{label})"))
}
