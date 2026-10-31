//! Commands implemented by the interpreter itself: macros, display, scalars,
//! programming commands, files, logs.

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{format, util, varlist};
use ferrostat_core::missing::is_missing;
use ferrostat_core::vfs;
use ferrostat_core::{Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, parse_line, split_options, split_top_level_words, Parser};

use crate::interpreter::{err_message, AdoInterpreter};
use crate::macros::{self, macro_number, unquote};

/// SSC packages whose functionality Ferrostat provides natively.
pub const NATIVE_PACKAGES: &[&str] = &["outreg2", "xtpmg", "spmap", "shp2dta", "sppack", "spmat", "xtivreg2", "ivreg2", "ranktest", "unique", "frmttable", "outreg", "reghdfe", "ivreghdfe", "ftools", "distinct", "mdesc", "winsor2", "gtools", "gcollapse", "gegen", "gquantiles", "fasterxtile", "gstats", "gtop", "gtoplevelsof", "gunique", "gdistinct", "glevelsof", "gisid", "gduplicates", "hashsort", "greshape", "fcollapse", "fegen", "fisid", "fsort", "join", "flevelsof", "fre", "sumup", "tabout", "egenmore", "rangestat", "carryforward", "labutil", "labmask", "labvalch", "labvalclone", "labvarch", "rdrobust", "rdbwselect", "rdplot", "rddensity", "rdbwdensity", "binscatter", "ppmlhdfe", "drdid", "csdid", "did_imputation", "event_plot", "eventstudyinteract", "psmatch2", "cem", "ebalance", "synth", "boottest", "wyoung"];

/// Community packages whose commands have a different name from the package,
/// as (package, commands it provides).
const PACKAGE_COMMANDS: &[(&str, &[&str])] = &[
    ("estout", &["esttab", "eststo", "estout", "estadd", "estpost"]),
    ("outreg", &["frmttable", "outreg"]),
    ("tabout", &["tabout"]),
    ("gtools", &["gcollapse", "gegen", "gquantiles", "fasterxtile", "gstats", "gtop", "gtoplevelsof", "gunique", "gdistinct", "glevelsof", "gisid", "gduplicates", "hashsort", "greshape"]),
    ("ftools", &["fcollapse", "fegen", "fisid", "fsort", "join", "flevelsof"]),
    ("labutil", &["labmask", "labvalch", "labvalclone", "labvarch"]),
    ("st0085_2", &["esttab", "eststo"]),
    ("rdrobust", &["rdrobust", "rdbwselect", "rdplot"]),
    ("rddensity", &["rddensity", "rdbwdensity"]),
    ("csdid", &["csdid", "csdid_plot"]),
    ("did_imputation", &["did_imputation", "event_plot"]),
    ("psmatch2", &["psmatch2", "pstest", "psgraph"]),
    ("cem", &["cem", "imb"]),
];

/// Packages accepted without providing commands (helpers whose features are
/// built in, or toolkits whose commands are not needed natively).
const NOOP_PACKAGES: &[&str] = &["ietoolkit", "iefieldkit", "moremata", "egenmore"];

/// Native commands providing `pkg`, or None when the package is unknown.
pub(crate) fn package_commands(me: &AdoInterpreter, pkg: &str) -> Option<Vec<String>> {
    if NOOP_PACKAGES.contains(&pkg) {
        return Some(Vec::new());
    }
    if let Some((_, cmds)) = PACKAGE_COMMANDS.iter().find(|(p, _)| *p == pkg) {
        let have: Vec<String> = cmds.iter().filter(|c| me.engine.is_command(c) || is_interp_command(c)).map(|c| c.to_string()).collect();
        return (!have.is_empty()).then_some(have);
    }
    (NATIVE_PACKAGES.contains(&pkg) || me.engine.is_command(pkg)).then(|| vec![pkg.to_string()])
}

pub fn interp_command(me: &mut AdoInterpreter, verb: &str, rest: &str, full: &str) -> Result<Option<String>, String> {
    let r = rest.trim();
    Ok(Some(match verb {
        "local" | "global" => {
            define_macro(me, verb == "global", r)?;
            String::new()
        }
        "macro" => macro_cmd(me, r)?,
        "lint" => crate::lint::lint_command(me, r)?,
        "display" => display(me, r)?,
        "scalar" => scalar(me, r)?,
        "program" => program_cmd(me, r)?,
        "syntax" => {
            syntax(me, r)?;
            String::new()
        }
        "args" => {
            let zero = me.engine.store.local("0").cloned().unwrap_or_default();
            let vals = macros::words(&zero);
            for (k, name) in r.split_whitespace().enumerate() {
                me.engine.store.set_local(name, vals.get(k).map(|v| unquote(v)).unwrap_or_default());
            }
            String::new()
        }
        "tokenize" => {
            let (text, opts) = split_opts(r);
            let text = unquote(&text);
            let parse = opts.iter().find(|(n, _)| n.starts_with("p")).and_then(|(_, a)| a.clone()).map(|a| unquote(&a));
            let toks = tokenize(&text, parse.as_deref());
            let st = &mut me.engine.store;
            for k in 1..100 {
                st.clear_local(&k.to_string());
            }
            for (k, t) in toks.iter().enumerate() {
                st.set_local(&(k + 1).to_string(), t.clone());
            }
            String::new()
        }
        "gettoken" => {
            gettoken(me, r)?;
            String::new()
        }
        "confirm" => {
            confirm(me, r)?;
            String::new()
        }
        "return" | "ereturn" | "sreturn" => returns(me, verb, r)?,
        "creturn" => creturn_list(me),
        "marksample" => {
            marksample(me, r)?;
            String::new()
        }
        "markout" => {
            let w: Vec<&str> = r.split(',').next().unwrap_or("").split_whitespace().collect();
            let touse = w.first().ok_or_else(|| rc_err(198, "markout touse varlist"))?.to_string();
            let vars = varlist::expand(&me.engine.dataset, &w[1..].join(" "))?;
            let ds = &mut me.engine.dataset;
            let mut mask: Vec<bool> = util::col(ds, &touse)?.iter().map(|v| *v != 0.0 && !is_missing(*v)).collect();
            util::markout(ds, &mut mask, &vars);
            let var = ds.get_var_mut(&touse).unwrap();
            var.data = mask.iter().map(|&m| Value::Byte(m as i8)).collect();
            var.var_type = VarType::Byte;
            String::new()
        }
        "tempvar" | "tempname" | "tempfile" => {
            for name in r.split_whitespace() {
                me.temp_counter += 1;
                let val = format!("__{:06}", me.temp_counter);
                let val = if verb == "tempfile" {
                    let p = std::path::PathBuf::from(vfs::temp_path(&format!("ferrostat_{}_{}.tmp", process_id(), val)));
                    me.tempfiles.push(p.clone());
                    p.display().to_string()
                } else {
                    if verb == "tempvar" {
                        if let Some(f) = me.tempvars.last_mut() {
                            f.push(val.clone());
                        }
                    }
                    val
                };
                me.engine.store.set_local(name, val);
            }
            String::new()
        }
        "numlist" => {
            let (text, opts) = split_opts(r);
            let vals = macros::numlist(&unquote(&text))?;
            let mut rules = macros::NumlistRules::default();
            for (o, a) in &opts {
                let a = a.as_deref().unwrap_or("").trim();
                match o.as_str() {
                    "min" => rules.min = a.parse().ok(),
                    "max" => rules.max = a.parse().ok(),
                    "int" | "integer" => rules.integer = true,
                    "a" | "ascending" => rules.ascending = true,
                    "d" | "descending" => rules.descending = true,
                    "missingokay" => rules.missing_ok = true,
                    "range" => rules.range.extend(a.split_whitespace().filter_map(macros::RangeCheck::parse)),
                    _ => {}
                }
            }
            macros::check_numlist(&vals, &rules)?;
            let mut vals = vals;
            if opts.iter().any(|(o, _)| o == "sort") {
                vals.sort_by(|a, b| a.total_cmp(b));
            }
            let mut res = Results::new();
            res.set_str("numlist", vals.iter().map(|v| macro_number(*v)).collect::<Vec<_>>().join(" "));
            me.engine.store.r = res;
            String::new()
        }
        "do" | "run" | "include" => {
            let (path, args) = path_and_rest(r);
            let out = me.run_do(&path, &args, verb == "run", verb == "include")?;
            out
        }
        "log" => log_cmd(me, r)?,
        "cmdlog" => String::new(),
        "cd" | "chdir" => {
            let p = unquote(r);
            if p.is_empty() {
                vfs::current_dir()
            } else {
                vfs::set_current_dir(&p).map_err(|_| rc_err(170, format!("unable to change to {p}")))?;
                vfs::current_dir()
            }
        }
        "pwd" => vfs::current_dir(),
        "notes" | "note" => notes_cmd(me, r)?,
        "char" => char_cmd(me, r)?,
        // the front ends clear their results window at this marker
        "cls" => crate::interpreter::CLS.to_string(),
        "dir" => dir_listing(r)?,
        "erase" => {
            let p = unquote(r);
            vfs::remove(&p).map_err(|_| rc_err(601, format!("file {p} not found")))?;
            String::new()
        }
        "mkdir" => {
            let p = unquote(r.split(',').next().unwrap_or(""));
            vfs::create_dir_all(&p).map_err(|e| rc_err(693, e))?;
            String::new()
        }
        "rmdir" => {
            let p = unquote(r);
            vfs::remove_dir(&p).map_err(|e| rc_err(693, e))?;
            String::new()
        }
        "type" => {
            let p = unquote(r.split(',').next().unwrap_or(""));
            vfs::read_to_string(&p).map_err(|_| rc_err(601, format!("file {p} not found")))?.trim_end().to_string()
        }
        "copy" => {
            let w = split_top_level_words(r.split(',').next().unwrap_or(""));
            if w.len() < 2 {
                return Err(rc_err(198, "copy source destination"));
            }
            let (a, b) = (unquote(&w[0]), unquote(&w[1]));
            if vfs::exists(&b) && !r.contains("replace") {
                return Err(rc_err(602, format!("file {b} already exists")));
            }
            vfs::copy(&a, &b).map_err(|_| rc_err(601, format!("file {a} not found")))?;
            String::new()
        }
        "shell" | "!" | "winexec" | "xshell" => {
            shell(r)?
        }
        "which" => {
            let name = r.split(',').next().unwrap_or("").trim();
            if me.engine.is_command(name) || is_interp_command(name) {
                format!("built-in command:  {name}")
            } else if let Some(p) = crate::w10_net::which_ado(me, name) {
                p
            } else if me.engine.store.programs.contains_key(name) {
                format!("{name} (program in memory)")
            } else if NATIVE_PACKAGES.contains(&name) {
                format!("built-in command:  {name} (native implementation)")
            } else {
                return Err(rc_err(111, format!("command {name} not found as either built-in or ado-file")));
            }
        }
        "ssc" | "net" | "ado" | "adoupdate" | "update" => crate::w10_net::command(me, verb, r)?,
        "c_local" => crate::w10_net::c_local(me, r)?,
        "cf" => cf(me, r)?,
        "more" | "set_defaults" | "graph_set" => String::new(),
        "timer" => timer(me, r)?,
        "sleep" => {
            let ms: u64 = r.parse().unwrap_or(0);
            // the browser has no blocking sleep
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(ms.min(60_000)));
            #[cfg(target_arch = "wasm32")]
            let _ = ms;
            String::new()
        }
        "help" | "whelp" | "search" | "findit" => crate::w10_net::help_search(me, verb, r),
        "set" if crate::w10_net::is_set_net(r) => crate::w10_net::set_net(me, r)?,
        "about" => crate::w8_prog::about(me),
        "error" => {
            let code: i32 = r.parse().map_err(|_| rc_err(198, "error #"))?;
            return Err(rc_err(code, error_text(code)));
        }
        "docx2pdf" | "docx2html" | "docx2txt" | "docx2markdown" | "docx2epub" | "pdf2txt" => crate::rpt_w6::command(me, verb, r)?,
        "putdocx" | "putpdf" if crate::rpt_w6::is_collect(r) => crate::rpt_w6::put_collect(me, verb, r)?,
        "set" if crate::rpt_w6::is_set_docx(r) => crate::rpt_w6::set_docx(me, r)?,
        "putexcel" => crate::putexcel::putexcel(me, r)?,
        "putdocx" => crate::putdocx::putdocx(me, r)?,
        "putpdf" => crate::putdocx::putpdf(me, r)?,
        "dyndoc" => crate::dyndoc::dyndoc(me, r)?,
        "dyntext" => crate::dyndoc::dyntext(me, r)?,
        "markdown" => crate::dyndoc::markdown(me, r)?,
        "html2docx" => crate::dyndoc::html2docx(me, r)?,
        "outreg2" | "outreg" => crate::outreg::outreg2(me, r)?,
        "eststo" => crate::estout::eststo(me, r)?,
        "esttab" => crate::estout::table_cmd(me, r, true)?,
        "estout" => crate::estout::table_cmd(me, r, false)?,
        "estpost" => crate::estout::estpost(me, r)?,
        "estadd" => crate::estout::estadd(me, r)?,
        "frmttable" => crate::frmttable::frmttable(me, r)?,
        "collect" => crate::collect::collect(me, full)?,
        "table" if crate::w8_table::has_command(r) => crate::w8_table::run(me, full)?,
        "estat" if crate::w8_prog::estat_handler(me).is_some() => {
            let h = crate::w8_prog::estat_handler(me).unwrap();
            crate::w8_prog::estat(me, &h, r)?
        }
        "estat_default" => crate::w8_prog::estat_default(me, r)?,
        "dtable" | "etable" => crate::collect::dtable_etable(me, verb, full)?,
        "mata" | "mata:" => crate::mata::run_line(me, r)?,
        "python" => crate::python::command(me, r)?,
        "putmata" => crate::dm_w3::putmata(me, r)?,
        "getmata" => crate::dm_w3::getmata(me, r)?,
        "set" if crate::python::is_set_python(r) => crate::python::set_option(me, r)?,
        _ => return crate::progtools::command(me, verb, rest, full),
    }))
}

/// `cf varlist using file [, all verbose]`: compare the data in memory with
/// a saved dataset; silent when everything matches, error 9 otherwise.
fn cf(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let has = |n: &str, min: usize| opts.iter().any(|(o, _)| o.len() >= min && n.starts_with(o.as_str()));
    let (all, verbose) = (has("all", 3), has("verbose", 1));
    let u = body.find(" using ").map(|i| i + 1).or_else(|| body.starts_with("using ").then_some(0)).ok_or_else(|| rc_err(100, "using required"))?;
    let list = body[..u].trim();
    let path = unquote(body[u + 5..].trim());
    let path = if std::path::Path::new(&path).extension().is_some() { path } else { format!("{path}.dta") };
    let other = ferrostat_core::dta::DtaReader::open(&path).map_err(|_| rc_err(601, format!("file {path} not found")))?;
    let ds = &me.engine.dataset;
    let vars = varlist::expand(ds, if list.is_empty() { "*" } else { list })?;
    if ds.n_obs != other.n_obs {
        return Err(rc_err(9, format!("master has {} obs., using {}", ds.n_obs, other.n_obs)));
    }
    let mut lines = Vec::new();
    let mut bad = false;
    for v in &vars {
        let a = ds.get_var(v).unwrap();
        let msg = match other.get_var(v) {
            None => Some("does not exist in using".to_string()),
            Some(b) if a.var_type.is_string() != b.var_type.is_string() => Some(format!("{} in master but {} in using", a.var_type.type_str(), b.var_type.type_str())),
            Some(b) => {
                let differs: Vec<usize> = (0..ds.n_obs)
                    .filter(|&i| {
                        if a.var_type.is_string() {
                            util::cell_text(ds, a, i, false) != util::cell_text(&other, b, i, false)
                        } else {
                            let (x, y) = (a.num(i), b.num(i));
                            x != y && !(x.is_nan() && y.is_nan())
                        }
                    })
                    .collect();
                if differs.is_empty() {
                    None
                } else {
                    let mut m = format!("{} mismatch{}", differs.len(), if differs.len() == 1 { "" } else { "es" });
                    if verbose {
                        for &i in &differs {
                            m.push_str(&format!("\n{:>18}obs {:>6}. {} in master; {} in using", "", i + 1, util::cell_text(ds, a, i, false), util::cell_text(&other, b, i, false)));
                        }
                    }
                    Some(m)
                }
            }
        };
        match msg {
            Some(m) => {
                bad = true;
                lines.push(format!("{v:>16}:  {m}"));
            }
            None if all => lines.push(format!("{v:>16}:  match")),
            None => {}
        }
    }
    if bad {
        Err(rc_err(9, lines.join("\n")))
    } else {
        Ok(lines.join("\n"))
    }
}

/// `creturn list`: the `c()` values, grouped.
fn creturn_list(me: &AdoInterpreter) -> String {
    use ferrostat_cmd::eval::{c_value, C_GROUPS};
    let env = ferrostat_cmd::eval::Env::with_store(&me.engine.dataset, &me.engine.store);
    let bar = "-".repeat(78);
    let mut out = Vec::new();
    for (title, keys) in C_GROUPS {
        out.push(bar.clone());
        out.push(format!("    {title}"));
        out.push(bar.clone());
        let w = keys.iter().map(|k| k.len()).max().unwrap_or(0) + 3;
        for k in keys.iter() {
            let v = match c_value(&env, k) {
                Val::Num(x) => macro_number(x),
                Val::Str(t) => format!("\"{t}\""),
            };
            out.push(format!("    {:>w$} = {v}", format!("c({k})")));
        }
    }
    out.join("\n")
}

/// `dir [path][pattern] [, wide]`: list files with size and modification time.
fn dir_listing(r: &str) -> Result<String, String> {
    let (spec, opts) = split_opts(r);
    let spec = unquote(&spec);
    let spec = if spec.is_empty() { "*".to_string() } else { spec.replace('\\', "/") };
    let (dir, pat) = if vfs::meta(&spec).map_or(false, |m| m.0) {
        (spec.trim_end_matches('/').to_string(), "*".to_string())
    } else {
        match spec.rfind('/') {
            Some(i) => (spec[..i.max(1)].to_string(), spec[i + 1..].to_string()),
            None => (".".to_string(), spec.clone()),
        }
    };
    let mut entries: Vec<(String, bool)> = vfs::list(&dir).map_err(|_| rc_err(601, format!("directory {dir} not found")))?.into_iter().filter(|e| macros::glob_match(&pat, &e.0)).collect();
    entries.sort();
    if opts.iter().any(|(n, _)| n.starts_with('w')) {
        return Ok(entries.iter().map(|(n, d)| if *d { format!("{n}/") } else { n.clone() }).collect::<Vec<_>>().join("  "));
    }
    let mut out = Vec::new();
    for (name, is_dir) in entries {
        let path = if dir == "." { name.clone() } else { format!("{dir}/{name}") };
        let (_, size, mtime) = vfs::meta(&path).unwrap_or((is_dir, 0, None));
        let size = if is_dir {
            "<dir>".to_string()
        } else if size >= 1 << 20 {
            format!("{:.1}M", size as f64 / 1048576.0)
        } else {
            format!("{:.1}k", size as f64 / 1024.0)
        };
        let when = mtime
            .and_then(|t| chrono::DateTime::from_timestamp(t as i64, 0))
            .map(|t| {
                use chrono::{Datelike, Timelike};
                let t = t.with_timezone(&chrono::Local);
                format!("{:>8} {:02}:{:02}", format!("{}/{}/{:02}", t.month(), t.day(), t.year() % 100), t.hour(), t.minute())
            })
            .unwrap_or_else(|| " ".repeat(14));
        out.push(format!("{size:>8}  {when}  {name}{}", if is_dir { "/" } else { "" }));
    }
    Ok(out.join("\n"))
}

/// Commands handled by the interpreter itself rather than the engine.
pub const INTERP_COMMANDS: &[&str] = &[
    "wyoung", "local", "global", "macro", "display", "cls", "notes", "note", "char", "scalar", "program", "syntax", "args", "tokenize", "gettoken", "confirm", "return", "ereturn", "sreturn", "marksample", "markout", "tempvar", "tempname", "tempfile", "unab", "numlist", "do", "run", "include", "log", "cd", "pwd", "dir", "erase", "mkdir", "type", "copy", "shell", "which", "cf", "foreach", "forvalues", "while", "if", "capture", "quietly", "noisily", "by", "bysort", "putexcel", "eststo", "esttab", "estout", "estpost", "estadd", "putdocx", "putpdf", "dyndoc", "dyntext", "markdown", "html2docx", "error", "exit", "collect", "dtable", "etable", "lint",
    // progtools
    "file", "postfile", "post", "postclose", "postutil", "serset", "sysdir", "personal", "findfile", "tabdisp", "mark", "markin", "fvexpand", "_rmcoll", "rmcoll", "_rmdcoll", "makecns", "matcproc",
    "putmata", "getmata", "dyngen",
    "docx2pdf", "docx2html", "docx2txt", "docx2markdown", "docx2epub", "pdf2txt",
    "signestimationsample", "checkestimationsample", "viewsource", "class", "classutil", "classexit", "unabcmd", "discard", "pause", "adopath", "tsunab", "fvunab",
    // w10_net
    "ssc", "net", "ado", "adoupdate", "update", "help", "whelp", "search", "findit", "c_local",
    // w11_ui, w11_plugin, w11_java
    "view", "doedit", "db", "plugin", "javacall", "window",
];

fn is_interp_command(n: &str) -> bool {
    INTERP_COMMANDS.contains(&n)
}

fn error_text(code: i32) -> &'static str {
    match code {
        1 => "--Break--",
        4 => "no; dataset in memory has changed since last saved",
        9 => "assertion is false",
        100 => "varlist required",
        109 => "type mismatch",
        110 => "already defined",
        111 => "not found",
        198 => "invalid syntax",
        199 => "unrecognized command",
        459 => "something that should be true of your data is not",
        601 => "file not found",
        602 => "file already exists",
        2000 => "no observations",
        _ => "",
    }
}

/// Split `text, options` into text and parsed options.
fn split_opts(r: &str) -> (String, Vec<(String, Option<String>)>) {
    match find_top_level(r, ',') {
        Some(i) => (r[..i].trim().to_string(), split_options(&r[i + 1..])),
        None => (r.trim().to_string(), Vec::new()),
    }
}

fn path_and_rest(r: &str) -> (String, String) {
    let w = split_top_level_words(r.split(',').next().unwrap_or(""));
    let path = w.first().map(|x| unquote(x)).unwrap_or_default();
    let args = w[1.min(w.len())..].join(" ");
    (path, args)
}

fn define_macro(me: &mut AdoInterpreter, global: bool, r: &str) -> Result<(), String> {
    let r = r.trim();
    if let Some(n) = r.strip_prefix("++") {
        let st = &mut me.engine.store;
        let v = st.local(n.trim()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) + 1.0;
        st.set_local(n.trim(), macro_number(v));
        return Ok(());
    }
    if let Some(n) = r.strip_prefix("--") {
        let st = &mut me.engine.store;
        let v = st.local(n.trim()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) - 1.0;
        st.set_local(n.trim(), macro_number(v));
        return Ok(());
    }
    let end = r.find(|c: char| c.is_whitespace() || c == '=' || c == ':').unwrap_or(r.len());
    let name = r[..end].to_string();
    if name.is_empty() {
        return Err(rc_err(198, "macro name required"));
    }
    let rest = r[end..].trim_start();
    let value = if let Some(e) = rest.strip_prefix('=') {
        macros::eval_to_macro(&me.engine.dataset, &me.engine.store, e)?
    } else if let Some(f) = rest.strip_prefix(':') {
        let eng = &mut me.engine;
        macros::macro_function(f.trim(), &eng.dataset, &mut eng.store)?
    } else {
        let t = rest.trim();
        // a leading and a trailing quote are removed, even when they do not
        // pair up: `local x "a" "b"` stores `a" "b`
        if t.len() >= 4 && t.starts_with("`\"") && t.ends_with("\"'") {
            t[2..t.len() - 2].to_string()
        } else if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
            t[1..t.len() - 1].to_string()
        } else {
            t.to_string()
        }
    };
    let st = &mut me.engine.store;
    if global {
        if value.is_empty() {
            st.globals.remove(&name);
        } else {
            st.globals.insert(name, value);
        }
    } else {
        st.set_local(&name, value);
    }
    Ok(())
}

fn macro_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let mut w = r.split_whitespace();
    match w.next().unwrap_or("list") {
        "drop" => {
            for n in w {
                if n == "_all" {
                    me.engine.store.globals.clear();
                } else if let Some(l) = n.strip_prefix('_') {
                    me.engine.store.clear_local(l);
                } else {
                    me.engine.store.globals.remove(n);
                }
            }
            Ok(String::new())
        }
        "shift" => {
            // `1' takes `2', `2' takes `3', ...; the last one is dropped
            let by: usize = w.next().map(|n| n.parse().map_err(|_| rc_err(198, &format!("{n} invalid"))) ).transpose()?.unwrap_or(1);
            let st = &mut me.engine.store;
            let mut i = 1;
            while st.local(&i.to_string()).is_some() {
                match st.local(&(i + by).to_string()).cloned() {
                    Some(v) => st.set_local(&i.to_string(), v),
                    None => st.clear_local(&i.to_string()),
                }
                i += 1;
            }
            Ok(String::new())
        }
        _ => {
            let st = &me.engine.store;
            let mut out: Vec<String> = st.globals.iter().map(|(k, v)| format!("{:<13}{}", format!("{k}:"), v)).collect();
            if let Some(f) = st.frames.last() {
                out.extend(f.iter().map(|(k, v)| format!("{:<13}{}", format!("_{k}:"), v)));
            }
            Ok(out.join("\n"))
        }
    }
}

/// Strip simple SMCL directives used in display strings.
pub fn smcl_text(s: &str) -> String {
    let mut out = String::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '{' {
            if let Some(j) = c[i..].iter().position(|&x| x == '}') {
                let tag: String = c[i + 1..i + j].iter().collect();
                let t = tag.trim();
                let (name, arg) = t.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((t, ""));
                let known = matches!(name, "txt" | "text" | "res" | "result" | "err" | "error" | "inp" | "input" | "com" | "bf" | "it" | "sf" | "ul" | "hline" | "col" | "column" | "space" | "break" | "p" | "p_end" | "c" | "char" | "dup" | "help" | "browse" | "cmd" | "hi" | "title");
                if known {
                    match name {
                        "hline" => out.push_str(&"-".repeat(arg.parse().unwrap_or(78))),
                        "space" => out.push_str(&" ".repeat(arg.parse().unwrap_or(1))),
                        "col" | "column" => {
                            let col: usize = arg.parse().unwrap_or(1);
                            let cur = out.rsplit('\n').next().map_or(0, |l| l.chars().count());
                            if col > cur + 1 {
                                out.push_str(&" ".repeat(col - 1 - cur));
                            }
                        }
                        "break" => out.push('\n'),
                        "c" | "char" => out.push(match arg {
                            "|" => '|',
                            "-" => '-',
                            "+" | "TT" | "BT" | "LT" | "RT" | "TLC" | "TRC" | "BLC" | "BRC" => '+',
                            _ => ' ',
                        }),
                        "help" | "browse" | "cmd" | "hi" | "title" | "bf" | "it" | "ul" => {
                            if let Some((_, txt)) = arg.split_once(':') {
                                out.push_str(txt);
                            } else if let Some(t) = tag.split_once(':') {
                                out.push_str(t.1);
                            }
                        }
                        _ => {
                            if let Some((_, txt)) = tag.split_once(':') {
                                out.push_str(txt);
                            }
                        }
                    }
                    i += j + 1;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// `display` with strings, expressions, formats and directives.
pub fn display(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut s = r.trim().to_string();
    let mut fmt: Option<String> = None;
    let mut dup = 1usize;
    let mut newline_end = true;
    // after `_asis`, strings are written without SMCL interpretation
    let mut asis = false;
    while !s.is_empty() {
        s = s.trim_start().to_string();
        if s.is_empty() {
            break;
        }
        let lower = s.to_lowercase();
        // compound or simple string
        if s.starts_with("`\"") {
            let end = macros::compound_end(&s).ok_or_else(|| rc_err(198, "unmatched quote"))?;
            let text = if asis { s[2..end].to_string() } else { smcl_text(&s[2..end]) };
            out.push_str(&text.repeat(dup));
            dup = 1;
            s = s[end + 2..].to_string();
            continue;
        }
        if s.starts_with('"') {
            let end = s[1..].find('"').map(|i| i + 1).ok_or_else(|| rc_err(198, "unmatched quote"))?;
            let text = if asis { s[1..end].to_string() } else { smcl_text(&s[1..end]) };
            out.push_str(&text.repeat(dup));
            dup = 1;
            s = s[end + 1..].to_string();
            continue;
        }
        // `as style` and the old `in smcl` / `in red|green|yellow|white|blue`
        let old_in = lower.strip_prefix("in ").map(|x| x.trim_start()).map_or(false, |x| {
            ["smcl", "red", "green", "yellow", "white", "blue"].iter().any(|c| x.starts_with(c) && !x[c.len()..].starts_with(|ch: char| ch.is_alphanumeric() || ch == '_'))
        });
        if lower.starts_with("as ") || old_in {
            let mut parts = s.splitn(3, char::is_whitespace);
            parts.next();
            parts.next();
            s = parts.next().unwrap_or("").to_string();
            continue;
        }
        // `,` puts one blank between two directives, `,,` none
        if let Some(rest) = s.strip_prefix(",,") {
            s = rest.to_string();
            continue;
        }
        if let Some(rest) = s.strip_prefix(',') {
            out.push(' ');
            s = rest.to_string();
            continue;
        }
        if s.starts_with('_') {
            let (word, rest) = match s.find(|c: char| c.is_whitespace()) {
                Some(i) => (s[..i].to_string(), s[i..].to_string()),
                None => (s.clone(), String::new()),
            };
            let (name, arg) = match word.find('(') {
                Some(i) => (word[..i].to_string(), word[i + 1..].trim_end_matches(')').to_string()),
                None => (word.clone(), String::new()),
            };
            let n: usize = arg.trim().parse().unwrap_or(1);
            let handled = match name.as_str() {
                "_n" | "_newline" => {
                    out.push_str(&"\n".repeat(n));
                    true
                }
                "_s" | "_skip" => {
                    out.push_str(&" ".repeat(n));
                    true
                }
                "_col" | "_column" => {
                    let cur = out.rsplit('\n').next().map_or(0, |l| l.chars().count());
                    if n > cur + 1 {
                        out.push_str(&" ".repeat(n - 1 - cur));
                    }
                    true
                }
                "_c" | "_continue" => {
                    newline_end = false;
                    true
                }
                "_dup" => {
                    dup = n;
                    true
                }
                "_char" => {
                    out.push(char::from_u32(n as u32).unwrap_or(' '));
                    true
                }
                "_request" => true,
                "_asis" => {
                    asis = true;
                    true
                }
                _ => false,
            };
            if handled {
                s = rest;
                continue;
            }
        }
        if s.starts_with('%') {
            let i = s.find(char::is_whitespace).unwrap_or(s.len());
            fmt = Some(s[..i].to_string());
            s = s[i..].to_string();
            continue;
        }
        // expression: longest prefix of words that parses, up to a top-level comma
        let cut = top_level_comma(&s).unwrap_or(s.len());
        let tail = s[cut..].to_string();
        let words = split_top_level_words(&s[..cut]);
        let mut matched = None;
        for k in (1..=words.len()).rev() {
            let cand = words[..k].join(" ");
            if let Ok(e) = Parser::new(&cand).parse_full_expr() {
                matched = Some((k, e));
                break;
            }
        }
        let (k, e) = matched.ok_or_else(|| rc_err(198, format!("{s} invalid name")))?;
        let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?;
        let text = match (&v, &fmt) {
            (Val::Num(n), Some(f)) => format::fmt(*n, f),
            (Val::Num(n), None) => format::display_number(*n),
            (Val::Str(t), Some(f)) => format::fmt_str(t, f),
            (Val::Str(t), None) => t.clone(),
        };
        fmt = None;
        out.push_str(&text.repeat(dup));
        dup = 1;
        s = words[k..].join(" ") + &tail;
    }
    let _ = newline_end;
    Ok(out)
}

/// `timer on|off|clear|list [#]`: stopwatches 1–100; `list` reports and
/// sets r(t#) and r(nt#).
fn timer(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let now = chrono::Utc::now().timestamp_micros();
    let mut w = r.split_whitespace();
    let sub = w.next().unwrap_or("list");
    let num = match w.next() {
        Some(n) => Some(n.parse::<u32>().ok().filter(|n| (1..=100).contains(n)).ok_or_else(|| rc_err(198, format!("timer {n} invalid; must be between 1 and 100")))?),
        None => None,
    };
    let t = &mut me.timers;
    match sub {
        "on" => {
            let n = num.ok_or_else(|| rc_err(198, "timer number required"))?;
            let e = t.entry(n).or_insert((0.0, 0, None));
            if e.2.is_some() {
                return Err(rc_err(198, format!("timer {n} already on")));
            }
            e.2 = Some(now);
        }
        "off" => {
            let n = num.ok_or_else(|| rc_err(198, "timer number required"))?;
            let e = t.entry(n).or_insert((0.0, 0, None));
            let start = e.2.take().ok_or_else(|| rc_err(198, format!("timer {n} not on")))?;
            e.0 += (now - start) as f64 / 1e6;
            e.1 += 1;
        }
        "clear" => match num {
            Some(n) => {
                t.remove(&n);
            }
            None => t.clear(),
        },
        "list" => {
            let mut out = Vec::new();
            let rr = me.engine.store.begin_r();
            for (n, (total, count, running)) in me.timers.iter().filter(|(n, _)| num.is_none_or(|k| k == **n)) {
                let total = total + running.map_or(0.0, |s| (now - s) as f64 / 1e6);
                rr.set(&format!("t{n}"), total);
                rr.set(&format!("nt{n}"), *count as f64);
                out.push(format!("{n:>4}: {total:>9.2} /{count:>8} = {:>10.4}", if *count > 0 { total / *count as f64 } else { total }));
            }
            return Ok(out.join("\n"));
        }
        other => return Err(rc_err(198, format!("timer {other}: invalid subcommand"))),
    }
    Ok(String::new())
}

/// `evarname`: `_dta` or a variable name (abbreviations allowed).
fn char_owner(me: &AdoInterpreter, name: &str) -> Result<String, String> {
    let name = name.trim();
    if name == "_dta" {
        return Ok(name.to_string());
    }
    varlist::resolve_name(&me.engine.dataset, name)
}

/// `notes [evarname]: text`, `notes [list] [evarlist]`, `notes drop evarlist
/// [in #[/#]]`, `notes replace evarname in #: text`, `notes renumber`.
/// Notes are characteristics `note1`, `note2`, … with the count in `note0`.
fn notes_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let first = r.split(|c: char| c.is_whitespace() || c == ':').next().unwrap_or("");
    let chars = |me: &AdoInterpreter, owner: &str| -> Vec<(usize, String)> {
        let Some(c) = me.engine.dataset.characteristics.get(owner) else { return Vec::new() };
        let n: usize = c.get("note0").and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        (1..=n).filter_map(|i| c.get(&format!("note{i}")).map(|t| (i, t.clone()))).collect()
    };
    // adding: everything up to the first top-level colon names the owner
    if let Some(colon) = r.find(':').filter(|_| !matches!(first, "list" | "drop" | "replace" | "renumber" | "search" | "_count" | "_fetch")) {
        let owner = if r[..colon].trim().is_empty() { "_dta".to_string() } else { char_owner(me, &r[..colon])? };
        let mut text = r[colon + 1..].trim().to_string();
        if text.contains("TS") {
            text = text.replace("TS", &chrono::Local::now().format("%e %b %Y %H:%M").to_string().trim().to_string());
        }
        let ds = &mut me.engine.dataset;
        let c = ds.characteristics.entry(owner).or_default();
        let n: usize = c.get("note0").and_then(|v| v.trim().parse().ok()).unwrap_or(0) + 1;
        c.insert("note0".into(), n.to_string());
        c.insert(format!("note{n}"), text);
        me.engine.store.data_changed = true;
        return Ok(String::new());
    }
    let (sub, rest) = match first {
        "list" | "drop" | "replace" | "renumber" | "search" => (first, r[first.len()..].trim()),
        _ => ("list", r),
    };
    let owners = |me: &AdoInterpreter, list: &str| -> Result<Vec<String>, String> {
        if list.trim().is_empty() {
            let mut v = vec!["_dta".to_string()];
            v.extend(me.engine.dataset.vars.iter().map(|x| x.name.clone()));
            return Ok(v);
        }
        let mut out = Vec::new();
        for w in list.split_whitespace() {
            if w == "_dta" {
                out.push(w.to_string());
            } else {
                out.extend(varlist::expand(&me.engine.dataset, w)?);
            }
        }
        Ok(out)
    };
    // `in #[/#]` range of note numbers
    let (list, range) = match rest.find(" in ").or_else(|| rest.starts_with("in ").then_some(0)) {
        Some(i) => {
            let spec = rest[i..].trim_start().trim_start_matches("in").trim();
            let spec = spec.split(':').next().unwrap_or("").trim();
            let (a, b) = match spec.split_once('/') {
                Some((a, b)) => (a.trim().parse().unwrap_or(1), if b.trim() == "l" { usize::MAX } else { b.trim().parse().unwrap_or(usize::MAX) }),
                None => {
                    let a = spec.parse().map_err(|_| rc_err(198, "invalid in range"))?;
                    (a, a)
                }
            };
            (&rest[..i], Some((a, b)))
        }
        None => (rest, None),
    };
    let in_range = |k: usize| range.is_none_or(|(a, b)| k >= a && k <= b);
    match sub {
        "list" => {
            let mut out = Vec::new();
            for o in owners(me, list)? {
                let ns: Vec<(usize, String)> = chars(me, &o).into_iter().filter(|(k, _)| in_range(*k)).collect();
                if ns.is_empty() {
                    continue;
                }
                out.push(String::new());
                out.push(format!("{o}:"));
                for (k, t) in ns {
                    out.push(format!("{k:>3}.  {t}"));
                }
            }
            Ok(out.join("\n"))
        }
        "search" => {
            let needle = rest.trim().trim_matches('"').to_lowercase();
            let mut out = Vec::new();
            for o in owners(me, "")? {
                for (k, t) in chars(me, &o) {
                    if t.to_lowercase().contains(&needle) {
                        out.push(format!("{o}:\n{k:>3}.  {t}"));
                    }
                }
            }
            Ok(out.join("\n"))
        }
        "drop" => {
            let mut dropped = 0;
            for o in owners(me, list)? {
                let keep: Vec<String> = chars(me, &o).into_iter().filter(|(k, _)| !in_range(*k)).map(|(_, t)| t).collect();
                let before = chars(me, &o).len();
                dropped += before - keep.len();
                if let Some(c) = me.engine.dataset.characteristics.get_mut(&o) {
                    c.retain(|k, _| !k.starts_with("note"));
                    for (i, t) in keep.iter().enumerate() {
                        c.insert(format!("note{}", i + 1), t.clone());
                    }
                    if !keep.is_empty() {
                        c.insert("note0".into(), keep.len().to_string());
                    }
                }
            }
            me.engine.store.data_changed = true;
            Ok(format!("({dropped} note{} dropped)", if dropped == 1 { "" } else { "s" }))
        }
        "replace" => {
            let colon = rest.find(':').ok_or_else(|| rc_err(198, "notes replace: text required"))?;
            let owner = char_owner(me, list)?;
            let (a, _) = range.ok_or_else(|| rc_err(198, "notes replace requires in #"))?;
            let c = me.engine.dataset.characteristics.entry(owner.clone()).or_default();
            if !c.contains_key(&format!("note{a}")) {
                return Err(rc_err(198, format!("note {a} of {owner} not found")));
            }
            c.insert(format!("note{a}"), rest[colon + 1..].trim().to_string());
            me.engine.store.data_changed = true;
            Ok(String::new())
        }
        _ => Ok(String::new()), // renumber: notes are always kept numbered 1..n
    }
}

/// `label language [name] [, new copy rename delete]`: several sets of
/// variable labels and value-label attachments, kept in characteristics
/// (`_dta[_lang_list]`, `_dta[_lang_c]`, `var[_lang_l_name]`,
/// `var[_lang_v_name]`) so they are saved with the data.
pub(crate) fn label_language(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (name, opts) = match r.split_once(',') {
        Some((n, o)) => (n.trim(), o.trim().to_lowercase()),
        None => (r.trim(), String::new()),
    };
    let ds = &mut me.engine.dataset;
    let dta = ds.characteristics.entry("_dta".into()).or_default();
    let current = dta.get("_lang_c").cloned().unwrap_or_else(|| "default".into());
    let mut list: Vec<String> = dta.get("_lang_list").map(|l| l.split_whitespace().map(str::to_string).collect()).unwrap_or_else(|| vec![current.clone()]);
    let write_list = |ds: &mut ferrostat_core::DataSet, list: &[String], cur: &str| {
        let dta = ds.characteristics.entry("_dta".into()).or_default();
        if list.len() <= 1 && cur == "default" {
            dta.remove("_lang_list");
            dta.remove("_lang_c");
        } else {
            dta.insert("_lang_list".into(), list.join(" "));
            dta.insert("_lang_c".into(), cur.to_string());
        }
    };
    // store the labels in effect under `lang`
    let save = |ds: &mut ferrostat_core::DataSet, lang: &str| {
        let dl = ds.datalabel.clone();
        for v in ds.vars.clone() {
            let c = ds.characteristics.entry(v.name.clone()).or_default();
            if v.label.is_empty() { c.remove(&format!("_lang_l_{lang}")) } else { c.insert(format!("_lang_l_{lang}"), v.label.clone()) };
            if v.value_label_name.is_empty() { c.remove(&format!("_lang_v_{lang}")) } else { c.insert(format!("_lang_v_{lang}"), v.value_label_name.clone()) };
        }
        let dta = ds.characteristics.entry("_dta".into()).or_default();
        if dl.is_empty() { dta.remove(&format!("_lang_l_{lang}")) } else { dta.insert(format!("_lang_l_{lang}"), dl) };
    };
    // put the labels stored under `lang` in effect (and drop the stored copy)
    let load = |ds: &mut ferrostat_core::DataSet, lang: &str| {
        for i in 0..ds.vars.len() {
            let name = ds.vars[i].name.clone();
            let c = ds.characteristics.entry(name.clone()).or_default();
            let l = c.remove(&format!("_lang_l_{lang}")).unwrap_or_default();
            let v = c.remove(&format!("_lang_v_{lang}")).unwrap_or_default();
            if c.is_empty() {
                ds.characteristics.remove(&name);
            }
            ds.vars[i].label = l;
            ds.vars[i].value_label_name = v;
        }
        let dta = ds.characteristics.entry("_dta".into()).or_default();
        ds.datalabel = dta.remove(&format!("_lang_l_{lang}")).unwrap_or_default();
    };
    if name.is_empty() {
        let mut out = vec![String::new(), "  Language for variable and value labels".into(), String::new()];
        if list.len() <= 1 {
            out.push(format!("  In this dataset, value and variable labels have been defined in only one language: {current}"));
        } else {
            out.push("  Available languages:".into());
            for l in &list {
                out.push(format!("          {l}"));
            }
            out.push(String::new());
            out.push(format!("  Currently active language: {current}"));
        }
        return Ok(out.join("\n"));
    }
    varlist::check_name(name)?;
    let opt = |o: &str| opts.split_whitespace().any(|w| w == o);
    if opt("rename") {
        if list.iter().any(|l| l == name) {
            return Err(rc_err(110, format!("language {name} already defined")));
        }
        for l in list.iter_mut().filter(|l| **l == current) {
            *l = name.to_string();
        }
        write_list(ds, &list, name);
        return Ok(format!("(language {current} renamed {name})"));
    }
    if opt("delete") {
        if !list.iter().any(|l| l == name) {
            return Err(rc_err(111, format!("language {name} not defined")));
        }
        if list.len() == 1 {
            return Err(rc_err(198, "cannot delete the only language"));
        }
        list.retain(|l| l != name);
        if name == current {
            let next = list[0].clone();
            load(ds, &next);
            write_list(ds, &list, &next);
            return Ok(format!("(language {name} deleted; language {next} now current)"));
        }
        // drop its stored labels
        for c in ds.characteristics.values_mut() {
            c.remove(&format!("_lang_l_{name}"));
            c.remove(&format!("_lang_v_{name}"));
        }
        write_list(ds, &list, &current);
        return Ok(format!("(language {name} deleted)"));
    }
    if opt("new") || opt("copy") {
        if list.iter().any(|l| l == name) {
            return Err(rc_err(110, format!("language {name} already defined")));
        }
        save(ds, &current);
        if opt("new") {
            ds.datalabel.clear();
            for v in ds.vars.iter_mut() {
                v.label.clear();
                v.value_label_name.clear();
            }
        }
        list.push(name.to_string());
        write_list(ds, &list, name);
        me.engine.store.data_changed = true;
        return Ok(format!("(language {name} now current language)"));
    }
    if name == current {
        return Ok(String::new());
    }
    if !list.iter().any(|l| l == name) {
        return Err(rc_err(111, format!("language {name} not defined")));
    }
    save(ds, &current);
    load(ds, name);
    write_list(ds, &list, name);
    me.engine.store.data_changed = true;
    Ok(String::new())
}

/// `char [define] evarname[charname] [text]`, `char list [evarname[charname]]`,
/// `char rename oldvar newvar`.
fn char_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let target = |s: &str| -> Option<(String, String, String)> {
        let open = s.find('[')?;
        let close = open + s[open..].find(']')?;
        Some((s[..open].trim().to_string(), s[open + 1..close].trim().to_string(), s[close + 1..].trim().to_string()))
    };
    if let Some(rest) = r.strip_prefix("list").filter(|x| x.is_empty() || x.starts_with(char::is_whitespace)) {
        let rest = rest.trim();
        let mut out = Vec::new();
        let (only_owner, only_name) = match target(rest) {
            Some((o, n, _)) => (Some(if o == "_dta" { o } else { char_owner(me, &o)? }), Some(n).filter(|n| !n.is_empty())),
            None if !rest.is_empty() => (Some(char_owner(me, rest)?), None),
            None => (None, None),
        };
        for (owner, c) in &me.engine.dataset.characteristics {
            if only_owner.as_ref().is_some_and(|o| o != owner) {
                continue;
            }
            for (name, v) in c {
                if only_name.as_ref().is_some_and(|n| n != name) {
                    continue;
                }
                out.push(format!("  {:<30}{}", format!("{owner}[{name}]:"), v));
            }
        }
        return Ok(out.join("\n"));
    }
    if let Some(rest) = r.strip_prefix("rename ") {
        let w: Vec<&str> = rest.split(|c: char| c.is_whitespace() || c == ',').filter(|x| !x.is_empty()).collect();
        if w.len() < 2 {
            return Err(rc_err(198, "char rename oldvar newvar"));
        }
        let (old, new) = (char_owner(me, w[0])?, char_owner(me, w[1])?);
        let moved = me.engine.dataset.characteristics.remove(&old).unwrap_or_default();
        me.engine.dataset.characteristics.entry(new).or_default().extend(moved);
        return Ok(String::new());
    }
    let r = r.strip_prefix("define ").unwrap_or(r).trim();
    let (owner, name, text) = target(r).ok_or_else(|| rc_err(198, "char evarname[charname] text"))?;
    let owner = char_owner(me, &owner)?;
    let text = crate::macros::unquote(&text);
    let c = me.engine.dataset.characteristics.entry(owner.clone()).or_default();
    if text.is_empty() {
        c.remove(&name);
        if c.is_empty() {
            me.engine.dataset.characteristics.remove(&owner);
        }
    } else {
        c.insert(name, text.to_string());
    }
    me.engine.store.data_changed = true;
    Ok(String::new())
}

/// Byte offset of the first comma outside quotes and brackets.
fn top_level_comma(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => quote = !quote,
            '(' | '[' if !quote => depth += 1,
            ')' | ']' if !quote => depth -= 1,
            ',' if !quote && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn scalar(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let r = r.strip_prefix("define ").unwrap_or(r).trim();
    let w: Vec<&str> = r.split_whitespace().collect();
    match w.first().copied() {
        Some("list") | Some("li") | Some("l") | Some("dir") => {
            let st = &me.engine.store;
            let names: Vec<String> = if w.len() > 1 && w[1] != "_all" { w[1..].iter().map(|s| s.to_string()).collect() } else { st.scalars.keys().chain(st.str_scalars.keys()).cloned().collect() };
            Ok(names
                .iter()
                .map(|n| match (st.scalars.get(n), st.str_scalars.get(n)) {
                    (Some(v), _) => format!("{:>12} = {}", n, format::fmt_g(*v, 10)),
                    (_, Some(s)) => format!("{:>12} = {}", n, s),
                    _ => format!("{n} not found"),
                })
                .collect::<Vec<_>>()
                .join("\n"))
        }
        Some("drop") => {
            for n in &w[1..] {
                if *n == "_all" {
                    me.engine.store.scalars.clear();
                    me.engine.store.str_scalars.clear();
                } else {
                    me.engine.store.scalars.remove(*n);
                    me.engine.store.str_scalars.remove(*n);
                }
            }
            Ok(String::new())
        }
        _ => {
            let eq = r.find('=').ok_or_else(|| rc_err(198, "scalar name = exp"))?;
            let name = r[..eq].trim().to_string();
            varlist::check_name(&name)?;
            let e = parse_expression(&r[eq + 1..]).map_err(|e| rc_err(198, e.message))?;
            let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?;
            match v {
                Val::Num(n) => {
                    me.engine.store.str_scalars.remove(&name);
                    me.engine.store.scalars.insert(name, n);
                }
                Val::Str(s) => {
                    me.engine.store.scalars.remove(&name);
                    me.engine.store.str_scalars.insert(name, s);
                }
            }
            Ok(String::new())
        }
    }
}

fn program_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    match w.first().copied() {
        Some("drop") => {
            for n in &w[1..] {
                if *n == "_all" || *n == "_allado" {
                    me.engine.store.programs.clear();
                } else if me.engine.store.programs.remove(*n).is_none() {
                    return Err(rc_err(111, format!("program {n} not found")));
                }
            }
            Ok(String::new())
        }
        Some("dir") => Ok(me.engine.store.programs.keys().cloned().collect::<Vec<_>>().join("\n")),
        Some("list") | Some("l") | Some("li") => {
            let names: Vec<String> = if w.len() > 1 { w[1..].iter().map(|s| s.to_string()).collect() } else { me.engine.store.programs.keys().cloned().collect() };
            let mut out = Vec::new();
            for n in names {
                if let Some(p) = me.engine.store.programs.get(&n) {
                    out.push(format!("{n}:"));
                    for l in &p.body {
                        out.push(format!("  {l}"));
                    }
                }
            }
            Ok(out.join("\n"))
        }
        _ if crate::w11_plugin::is_plugin_def(r) => crate::w11_plugin::define(me, r),
        _ => Err(rc_err(198, "program definitions must end with end")),
    }
}

/// Split a string into tokens on whitespace (and `parse` characters),
/// keeping quoted strings together.
pub fn tokenize(text: &str, parse: Option<&str>) -> Vec<String> {
    let parse_chars: Vec<char> = parse.map(|p| p.chars().filter(|c| !c.is_whitespace()).collect()).unwrap_or_default();
    let mut out = Vec::new();
    for w in split_top_level_words(text) {
        if parse_chars.is_empty() || w.starts_with('"') {
            out.push(unquote(&w));
            continue;
        }
        let mut cur = String::new();
        for c in w.chars() {
            if parse_chars.contains(&c) {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(c.to_string());
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

fn gettoken(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let (spec, opts) = split_opts(r);
    let colon = spec.find(':').ok_or_else(|| rc_err(198, "gettoken emname : macname"))?;
    // `(local)` / `(global)` designators may precede each macro name
    let mut targets: Vec<(bool, String)> = Vec::new();
    let mut global_next = false;
    for w in spec[..colon].split_whitespace() {
        match w {
            "(local)" => global_next = false,
            "(global)" => global_next = true,
            _ => {
                targets.push((global_next, w.to_string()));
                global_next = false;
            }
        }
    }
    let src_spec = spec[colon + 1..].trim();
    let (src_global, src) = if let Some(n) = src_spec.strip_prefix("(global)") {
        (true, n.trim().to_string())
    } else {
        (false, src_spec.strip_prefix("(local)").unwrap_or(src_spec).trim().to_string())
    };
    if targets.is_empty() || targets.len() > 2 || src.is_empty() {
        return Err(rc_err(198, "gettoken emname [restname] : macname"));
    }
    let quotes = opts.iter().any(|(n, _)| n.starts_with("qu"));
    let parse: Vec<char> = opts
        .iter()
        .find(|(n, _)| n.starts_with('p'))
        .and_then(|(_, a)| a.clone())
        .map(|a| unquote(&a).chars().collect())
        .unwrap_or_else(|| vec![' ']);
    let st = &mut me.engine.store;
    let text = if src_global { st.globals.get(&src).cloned() } else { st.local(&src).cloned() }.unwrap_or_default();
    let t = text.trim_start_matches(|c: char| c == ' ');
    // token: quoted string, or up to a parse character
    let (tok, rest) = if t.starts_with('"') || t.starts_with("`\"") {
        let w = split_top_level_words(t);
        let first = w.first().cloned().unwrap_or_default();
        let rest = t.get(first.len()..).unwrap_or("").to_string();
        (if quotes { first } else { unquote(&first) }, rest)
    } else {
        let mut depth = 0;
        let mut end = t.len();
        for (i, c) in t.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth == 0 && parse.contains(&c) {
                end = if i == 0 && c != ' ' { c.len_utf8() } else { i };
                break;
            }
        }
        (t[..end].to_string(), t[end..].to_string())
    };
    // only the two-target form assigns the remainder; the source macro is
    // never modified (unless it is itself named as a target)
    let mut set = |(global, name): &(bool, String), v: String| {
        if *global {
            st.globals.insert(name.clone(), v);
        } else {
            st.set_local(name, v);
        }
    };
    set(&targets[0], tok);
    if let Some(t2) = targets.get(1) {
        set(t2, rest);
    }
    Ok(())
}

fn confirm(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    let ds = &me.engine.dataset;
    let st = &me.engine.store;
    let kind = w.first().copied().unwrap_or("");
    let args = |n: usize| -> String { w[n.min(w.len())..].join(" ") };
    match kind {
        "variable" | "var" | "numeric" | "string" | "str" | "byte" | "int" | "long" | "float" | "double" => {
            let (want, text) = if kind.starts_with("var") { (None, args(1)) } else { (Some(kind), args(2)) };
            let vars = varlist::expand(ds, &text).map_err(|_| rc_err(111, format!("variable {} not found", text.trim())))?;
            if let Some(t) = want {
                for v in &vars {
                    let vt = ds.get_var(v).unwrap().var_type;
                    let ok = match t {
                        "numeric" => vt.is_numeric(),
                        "string" | "str" => vt.is_string(),
                        other => vt.type_str() == other,
                    };
                    if !ok {
                        return Err(rc_err(7, format!("'{v}' found where {t} expected")));
                    }
                }
            }
            Ok(())
        }
        "new" => match w.get(1).copied() {
            Some("variable") | Some("var") => {
                for v in &w[2..] {
                    varlist::check_name(v)?;
                    if ds.var_index.contains_key(*v) {
                        return Err(rc_err(110, format!("variable {v} already defined")));
                    }
                }
                Ok(())
            }
            Some("file") => {
                let p = unquote(&args(2));
                if vfs::exists(&p) {
                    return Err(rc_err(602, format!("file {p} already exists")));
                }
                Ok(())
            }
            _ => Err(rc_err(198, "confirm new variable|file")),
        },
        "file" => {
            let p = unquote(&args(1));
            if vfs::exists(&p) { Ok(()) } else { Err(rc_err(601, format!("file {p} not found"))) }
        }
        "existence" => if args(1).trim().is_empty() { Err(rc_err(6, "")) } else { Ok(()) },
        "integer" | "number" | "n" => {
            let t = if kind == "integer" { args(2) } else { args(1) };
            match t.trim().parse::<f64>() {
                Ok(v) if kind != "integer" || v.fract() == 0.0 => Ok(()),
                _ => Err(rc_err(7, format!("'{t}' found where {} expected", if kind == "integer" { "integer" } else { "number" }))),
            }
        }
        "matrix" => if st.matrices.contains_key(args(1).trim()) { Ok(()) } else { Err(rc_err(111, format!("matrix {} not found", args(1)))) },
        "scalar" => if st.scalars.contains_key(args(1).trim()) || st.str_scalars.contains_key(args(1).trim()) { Ok(()) } else { Err(rc_err(111, format!("scalar {} not found", args(1)))) },
        "name" | "names" => {
            for n in &w[1..] {
                varlist::check_name(n)?;
            }
            Ok(())
        }
        _ => Err(rc_err(198, format!("confirm {kind}: invalid"))),
    }
}

fn returns(me: &mut AdoInterpreter, verb: &str, r: &str) -> Result<String, String> {
    let w: Vec<&str> = r.split_whitespace().collect();
    let sub = w.first().copied().unwrap_or("list");
    let prefix = verb.chars().next().unwrap();
    if sub == "list" || sub == "li" {
        let st = &me.engine.store;
        let res = match prefix {
            'r' => &st.r,
            'e' => &st.e,
            _ => &st.s,
        };
        let mut clean = res.clone();
        clean.macros.retain(|k, _| !k.starts_with('_'));
        return Ok(clean.listing(prefix));
    }
    if sub == "clear" {
        match prefix {
            'r' => {
                if let Some(Some(p)) = me.pending_r.last_mut() {
                    *p = Results::new();
                } else {
                    me.engine.store.r = Results::new();
                }
            }
            'e' => me.engine.store.e = Results::new(),
            _ => me.engine.store.s = Results::new(),
        }
        return Ok(String::new());
    }
    let rest = r[sub.len()..].trim();
    if sub == "add" {
        if let Some(Some(p)) = me.pending_r.last_mut() {
            let cur = me.engine.store.r.clone();
            for (k, v) in cur.scalars {
                p.scalars.entry(k).or_insert(v);
            }
            for (k, v) in cur.macros {
                p.macros.entry(k).or_insert(v);
            }
            for (k, v) in cur.matrices {
                p.matrices.entry(k).or_insert(v);
            }
        }
        return Ok(String::new());
    }
    if prefix == 'e' && (sub == "post" || sub == "repost") {
        let (main, opts) = split_opts(rest);
        let names: Vec<&str> = main.split_whitespace().collect();
        let mut b = names.first().and_then(|n| me.engine.store.matrices.get(*n).cloned());
        let mut v = names.get(1).and_then(|n| me.engine.store.matrices.get(*n).cloned());
        // `ereturn repost b = name V = name`
        if main.contains('=') {
            let flat = main.replace('=', " = ");
            let w: Vec<&str> = flat.split_whitespace().collect();
            let find = |key: &str| w.windows(3).find(|t| t[0] == key && t[1] == "=").and_then(|t| me.engine.store.matrices.get(t[2]).cloned());
            b = find("b");
            v = find("V");
        }
        let mut e = if sub == "repost" { me.engine.store.e.clone() } else { Results::new() };
        if let Some(b) = b {
            e.set_mat("b", b);
        }
        if let Some(v) = v {
            e.set_mat("V", v);
        }
        for (n, a) in opts {
            match n.as_str() {
                "obs" => e.set("N", a.unwrap_or_default().trim().parse().unwrap_or(0.0)),
                "depname" => e.set_str("depvar", a.unwrap_or_default().trim()),
                "esample" => {
                    let var = a.unwrap_or_default().trim().to_string();
                    let c = util::col(&me.engine.dataset, &var)?;
                    e.sample = Some(c.iter().map(|x| *x != 0.0 && !is_missing(*x)).collect());
                    let _ = me.engine.dataset.drop_variable(&var);
                }
                "dof" => e.set("df_r", a.unwrap_or_default().trim().parse().unwrap_or(0.0)),
                _ => {}
            }
        }
        me.engine.store.e = e;
        return Ok(String::new());
    }
    if prefix == 'e' && sub == "display" {
        let st = &me.engine.store;
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(b.cols, b.cols));
        let k = b.cols;
        let lines = ferrostat_cmd::est::coef_table(&ferrostat_cmd::est::TableSpec {
            depvar: st.e.macros.get("depvar").map(String::as_str).unwrap_or(""),
            names: &b.colnames,
            b: &b.data,
            v: &v,
            omitted: &vec![false; k],
            is_base: &vec![false; k],
            df: st.e.get("df_r"),
            level: st.level,
            eform: None,
            coef_title: "Coefficient",
        });
        return Ok(lines.join("\n"));
    }
    // scalar / local / matrix
    let eq = rest.find(|c: char| c == '=' || c.is_whitespace()).unwrap_or(rest.len());
    let name = rest[..eq].trim().to_string();
    let value_text = rest[eq..].trim_start();
    let target_is_pending = prefix == 'r' && matches!(me.pending_r.last(), Some(Some(_)));
    match sub {
        "scalar" | "sca" => {
            let e = parse_expression(value_text.trim_start_matches('=')).map_err(|e| rc_err(198, e.message))?;
            let v = Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)?.num()?;
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set(&name, v);
            } else {
                results_mut(me, prefix).set(&name, v);
            }
        }
        "local" | "loc" => {
            // `return local name: extended_fcn` (e.g. `: copy local x`)
            let (name, ext) = match name.strip_suffix(':') {
                Some(n) => (n.to_string(), true),
                None => (name.clone(), false),
            };
            let v = if ext {
                let eng = &mut me.engine;
                macros::expand(&format!("`:{value_text}'"), &eng.dataset, &mut eng.store)?
            } else if let Some(e) = value_text.strip_prefix('=') {
                macros::eval_to_macro(&me.engine.dataset, &me.engine.store, e)?
            } else {
                unquote(value_text)
            };
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set_str(&name, v);
            } else {
                results_mut(me, prefix).set_str(&name, v);
            }
        }
        "matrix" | "mat" => {
            let src = value_text.trim_start_matches('=').split(',').next().unwrap_or("").trim().to_string();
            let m = ferrostat_cmd::matcmd::eval_matrix(&me.engine.dataset, &me.engine.store, &src)?;
            if !r.contains(", copy") && !r.contains(",copy") {
                me.engine.store.matrices.remove(&src);
            }
            if target_is_pending {
                me.pending_r.last_mut().unwrap().as_mut().unwrap().set_mat(&name, m);
            } else {
                results_mut(me, prefix).set_mat(&name, m);
            }
        }
        _ => return Err(rc_err(198, format!("{verb} {sub}: invalid"))),
    }
    Ok(String::new())
}

fn results_mut(me: &mut AdoInterpreter, prefix: char) -> &mut Results {
    match prefix {
        'r' => &mut me.engine.store.r,
        'e' => &mut me.engine.store.e,
        _ => &mut me.engine.store.s,
    }
}

/// The `name=lclname` part of a syntax element such as `anything(name=x)`.
fn syntax_name_opt(spec: &str) -> Option<String> {
    let open = spec.find('(')?;
    let args = spec[open + 1..].trim_end_matches(')');
    args.split([' ', ',']).find_map(|a| a.trim().strip_prefix("name=").map(|n| n.trim().to_string())).filter(|n| !n.is_empty())
}

/// `syntax` for user programs.
fn syntax(me: &mut AdoInterpreter, spec: &str) -> Result<(), String> {
    let zero = me.engine.store.local("0").cloned().unwrap_or_default();
    let cmd = parse_line(&format!("__syntax {zero}")).map_err(|e| rc_err(198, e.message))?;
    // the option list starts at the first comma outside parentheses
    // (it is usually inside `[ ]`)
    let comma = {
        let mut depth = 0i32;
        let mut found = None;
        for (i, c) in spec.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    found = Some(i);
                    break;
                }
                _ => {}
            }
        }
        found
    };
    // options declared outside `[ ]` are required
    let opts_depth = comma.map_or(0, |i| bracket_depth(&spec[..i]));
    let (spec_main, spec_opts) = match comma {
        Some(i) => (spec[..i].trim_end().trim_end_matches('[').to_string(), spec[i + 1..].to_string()),
        None => (spec.to_string(), String::new()),
    };
    let spec_main = spec_main.replace("[,", " ");
    // `=exp` / `=/exp`: the expression after the first top-level `=`
    let mut main_text = cmd.main.clone();
    // (only an `=exp` element, not the `=` of `varlist(default=none)`)
    if let Some(k) = spec_main.find("=exp").or_else(|| spec_main.find("=/exp")) {
        let optional = spec_main[..k].trim_end().ends_with('[');
        let raw = spec_main[k + 1..].starts_with('/');
        match find_top_level(&cmd.main, '=') {
            Some(i) => {
                let e = cmd.main[i + 1..].trim().to_string();
                main_text = cmd.main[..i].to_string();
                me.engine.store.set_local("exp", if raw { e } else { format!("= {e}") });
            }
            None if optional => me.engine.store.set_local("exp", ""),
            None => return Err(rc_err(100, "=exp required")),
        }
    }
    let mut main_used = false;
    for item in split_top_level_words(&spec_main) {
        let optional = item.starts_with('[');
        let inner = item.trim_start_matches('[').trim_end_matches(']').to_string();
        let lower = inner.to_lowercase();
        let st = &mut me.engine.store;
        if lower.starts_with("varlist") || lower.starts_with("newvarlist") || lower.starts_with("varname") || lower.starts_with("newvarname") {
            main_used = true;
            let mut text = main_text.trim();
            if lower.starts_with("new") {
                // an optional storage type in front of the new names
                let first = text.split_whitespace().next().unwrap_or("");
                let is_type = matches!(first, "byte" | "int" | "long" | "float" | "double" | "strL")
                    || first.strip_prefix("str").is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
                let typ = if is_type { first } else { "float" };
                if is_type {
                    text = text[first.len()..].trim_start();
                }
                let n = text.split_whitespace().count();
                me.engine.store.set_local("typlist", vec![typ; n].join(" "));
            }
            if text.is_empty() {
                if optional {
                    let default_all = !lower.contains("default=none");
                    let v = if default_all && lower.starts_with("varlist") { me.engine.dataset.varlist().join(" ") } else { String::new() };
                    me.engine.store.set_local("varlist", v);
                    continue;
                }
                return Err(rc_err(100, "varlist required"));
            }
            let custom = syntax_name_opt(&inner);
            let name = custom.as_deref().unwrap_or("varlist");
            let v = if lower.starts_with("new") { text.to_string() } else { varlist::expand(&me.engine.dataset, text)?.join(" ") };
            if lower.contains("numeric") && !lower.starts_with("new") {
                for x in v.split_whitespace() {
                    if let Some(var) = me.engine.dataset.get_var(util::base_name(x)) {
                        if var.var_type.is_string() {
                            return Err(rc_err(109, "type mismatch"));
                        }
                    }
                }
            }
            if let Some(i) = lower.find("max=") {
                let m: usize = lower[i + 4..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(usize::MAX);
                if v.split_whitespace().count() > m {
                    return Err(rc_err(103, "too many variables specified"));
                }
            }
            if let Some(i) = lower.find("min=") {
                let m: usize = lower[i + 4..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
                if v.split_whitespace().count() < m {
                    return Err(rc_err(102, "too few variables specified"));
                }
            }
            me.engine.store.set_local(name, v);
        } else if lower.starts_with("anything") || lower.starts_with("namelist") || lower.starts_with("name") {
            main_used = true;
            let n = if lower.starts_with("anything") { "anything" } else { "namelist" };
            if main_text.trim().is_empty() && !optional {
                return Err(rc_err(100, format!("{n} required")));
            }
            // `anything(name=values)` stores the text in local `values`
            let target = syntax_name_opt(&inner).unwrap_or_else(|| n.to_string());
            st.set_local(&target, main_text.trim());
        } else if lower == "if" || lower == "if/" {
            // `[if/]` stores the expression without the leading `if`
            let raw = lower == "if/";
            if cmd.if_raw.is_none() && !optional {
                return Err(rc_err(100, "if required"));
            }
            st.set_local("if", cmd.if_raw.as_ref().map(|i| if raw { i.trim().to_string() } else { format!("if {i}") }).unwrap_or_default());
        } else if lower == "in" {
            let text = cmd.in_range.as_ref().map(|r| match r {
                ferrostat_parser::InRange::Single(a) => format!("in {a}"),
                ferrostat_parser::InRange::Range(a, b) => format!("in {a}/{b}"),
                ferrostat_parser::InRange::First(b) => format!("in f/{b}"),
                ferrostat_parser::InRange::Last(a) => format!("in -{a}/l"),
                ferrostat_parser::InRange::Rel(a, b) => format!("in {a}/{b}"),
            });
            st.set_local("in", text.unwrap_or_default());
        } else if lower.starts_with("using") {
            let u = cmd.using.clone().unwrap_or_default();
            if u.is_empty() && !optional {
                return Err(rc_err(100, "using required"));
            }
            st.set_local("using", u);
        } else if lower.contains("weight") {
            if let Some(w) = &cmd.weight {
                let raw = zero.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("");
                let exp = raw.split_once('=').map(|x| x.1).unwrap_or("");
                st.set_local("weight", w.weight_type.clone());
                st.set_local("exp", format!("= {}", exp.trim()));
            } else {
                st.set_local("weight", "");
                st.set_local("exp", "");
            }
        }
    }
    if !main_used && !main_text.trim().is_empty() {
        return Err(rc_err(198, format!("{} invalid", main_text.trim())));
    }
    // options
    let given = cmd.opts.clone();
    let mut used = vec![false; given.len()];
    for (item, required) in option_items(&spec_opts, opts_depth) {
        if item == "*" {
            continue;
        }
        let (decl, arg) = match item.find('(') {
            Some(i) => (item[..i].to_string(), Some(item[i + 1..item.len() - 1].to_string())),
            None => (item.clone(), None),
        };
        let neg = decl.starts_with("no") && decl.chars().nth(2).map_or(false, |c| c.is_uppercase());
        let full = decl.to_lowercase();
        let min = decl.chars().take_while(|c| c.is_uppercase()).count();
        let min = if neg { 2 + decl[2..].chars().take_while(|c| c.is_uppercase()).count() } else if min == 0 { full.len() } else { min };
        let local_name = if neg { full[2..].to_string() } else { full.clone() };
        let hit = given.iter().enumerate().find(|(_, (n, _))| {
            let n = n.to_lowercase();
            n.len() >= min && full.starts_with(&n)
        });
        let st = &mut me.engine.store;
        match (hit, arg) {
            (Some((i, (_, a))), Some(argspec)) => {
                used[i] = true;
                let a = a.clone().unwrap_or_default();
                let val = syntax_option_value(me, &local_name, &argspec, &a)?;
                me.engine.store.set_local(&local_name, val);
            }
            (Some((i, _)), None) => {
                used[i] = true;
                st.set_local(&local_name, if neg { format!("no{local_name}") } else { local_name.clone() });
            }
            (None, Some(_)) if required => return Err(rc_err(198, format!("option {local_name}() required"))),
            (None, None) if required => return Err(rc_err(198, format!("option {full} required"))),
            (None, Some(argspec)) => {
                let mut parts = argspec.split_whitespace();
                let kind = parts.next().unwrap_or("").to_lowercase();
                let default = parts.next().unwrap_or("").to_string();
                let v = if matches!(kind.as_str(), "int" | "integer" | "real" | "cilevel") {
                    if kind == "cilevel" { macro_number(st.level) } else { default }
                } else {
                    String::new()
                };
                st.set_local(&local_name, v);
            }
            (None, None) => st.set_local(&local_name, ""),
        }
    }
    let star = spec_opts.contains('*');
    let leftover: Vec<String> = given.iter().zip(&used).filter(|(_, u)| !**u).map(|((n, a), _)| match a {
        Some(a) => format!("{n}({a})"),
        None => n.clone(),
    }).collect();
    if !leftover.is_empty() {
        if star {
            me.engine.store.set_local("options", leftover.join(" "));
        } else {
            let bad = given.iter().zip(&used).find(|(_, u)| !**u).map(|((n, _), _)| n.clone()).unwrap_or_default();
            // declared options as written: `Scale(real 1)` -> "scale()", `Detail` -> "detail"
            let declared: Vec<String> = split_top_level_words(&spec_opts.replace(['[', ']'], " "))
                .iter()
                .filter(|o| o.as_str() != "*")
                .map(|o| match o.split_once('(') {
                    Some((n, _)) => format!("{}()", n.trim().to_lowercase()),
                    None => o.trim().to_lowercase(),
                })
                .collect();
            let names: Vec<&str> = declared.iter().map(|d| d.trim_end_matches("()")).collect();
            let err = rc_err(198, format!("option {} not allowed", leftover[0]));
            return Err(match ferrostat_engine::suggest::closest(&bad, names.iter().copied()) {
                Some(s) => {
                    let shown = declared.iter().find(|d| d.trim_end_matches("()") == s).cloned().unwrap_or(s);
                    ferrostat_engine::suggest::with_hint(&err, &format!("(did you mean {shown}?)"))
                }
                None => err,
            });
        }
    } else if star {
        me.engine.store.set_local("options", "");
    }
    Ok(())
}

/// Number of `[` left open in `s` (outside parentheses and quotes).
fn bracket_depth(s: &str) -> i32 {
    let mut depth = 0i32;
    let mut paren = 0i32;
    let mut quoted = false;
    for c in s.chars() {
        match c {
            '"' => quoted = !quoted,
            '(' if !quoted => paren += 1,
            ')' if !quoted => paren -= 1,
            '[' if !quoted && paren == 0 => depth += 1,
            ']' if !quoted && paren == 0 => depth -= 1,
            _ => {}
        }
    }
    depth.max(0)
}

/// The option declarations of a `syntax` specification, each with whether it
/// is required (declared outside `[ ]`). `depth` is the bracket depth at the
/// comma that starts the option list.
fn option_items(spec_opts: &str, depth: i32) -> Vec<(String, bool)> {
    let mut depth = depth;
    let mut paren = 0i32;
    let mut quoted = false;
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_required = depth <= 0;
    let mut flush = |cur: &mut String, req: bool, out: &mut Vec<(String, bool)>| {
        if !cur.trim().is_empty() {
            out.push((cur.trim().to_string(), req));
        }
        cur.clear();
    };
    for c in spec_opts.chars() {
        if quoted {
            cur.push(c);
            if c == '"' {
                quoted = false;
            }
            continue;
        }
        match c {
            '"' => {
                quoted = true;
                cur.push(c);
            }
            '(' => {
                paren += 1;
                cur.push(c);
            }
            ')' => {
                paren -= 1;
                cur.push(c);
            }
            '[' | ']' if paren == 0 => {
                flush(&mut cur, cur_required, &mut out);
                depth += if c == '[' { 1 } else { -1 };
            }
            c if c.is_whitespace() && paren == 0 => flush(&mut cur, cur_required, &mut out),
            c => {
                if cur.is_empty() {
                    cur_required = depth <= 0;
                }
                cur.push(c);
            }
        }
    }
    flush(&mut cur, cur_required, &mut out);
    out
}

/// The value `syntax` stores for an option given as `name(arg)` and declared
/// with argument type `argspec` (`integer 5`, `numlist >0 max=1`, `string`, ...).
fn syntax_option_value(me: &mut AdoInterpreter, name: &str, argspec: &str, a: &str) -> Result<String, String> {
    let lspec = argspec.to_lowercase();
    let kind = lspec.split_whitespace().next().unwrap_or("string").to_string();
    let bad = || rc_err(198, format!("option {name}() incorrectly specified"));
    let modifier = |key: &str| -> Option<usize> {
        lspec.split_whitespace().find_map(|w| w.strip_prefix(key)).and_then(|v| v.parse().ok())
    };
    Ok(match kind.as_str() {
        "int" | "integer" => {
            let t = a.trim();
            let v: f64 = t.parse().map_err(|_| bad())?;
            if v.fract() != 0.0 || !v.is_finite() {
                return Err(bad());
            }
            macro_number(v)
        }
        "real" => {
            let t = a.trim();
            let v: f64 = t.parse().map_err(|_| bad())?;
            macro_number(v)
        }
        "numlist" => {
            let vals = macros::numlist(a)?;
            macros::check_numlist(&vals, &numlist_rules(&lspec))?;
            vals.into_iter().map(macro_number).collect::<Vec<_>>().join(" ")
        }
        "varlist" | "varname" => {
            let v = varlist::expand(&me.engine.dataset, a)?;
            let max = if kind == "varname" { Some(1) } else { modifier("max=") };
            if max.is_some_and(|m| v.len() > m) {
                return Err(rc_err(103, "too many variables specified"));
            }
            if modifier("min=").is_some_and(|m| v.len() < m) {
                return Err(rc_err(102, "too few variables specified"));
            }
            let want_num = lspec.split_whitespace().any(|w| w == "numeric");
            let want_str = lspec.split_whitespace().any(|w| w == "string");
            for x in &v {
                if let Some(var) = me.engine.dataset.get_var(util::base_name(x)) {
                    let is_str = var.var_type.is_string();
                    if (want_num && is_str) || (want_str && !is_str) {
                        return Err(rc_err(109, "type mismatch"));
                    }
                }
            }
            v.join(" ")
        }
        "name" | "namelist" => {
            let words: Vec<&str> = a.split_whitespace().collect();
            let max = if kind == "name" { Some(modifier("max=").unwrap_or(1)) } else { modifier("max=") };
            if max.is_some_and(|m| words.len() > m) || modifier("min=").is_some_and(|m| words.len() < m) {
                return Err(bad());
            }
            let valid = |w: &str| {
                let mut c = w.chars();
                c.next().is_some_and(|f| f.is_alphabetic() || f == '_') && c.all(|x| x.is_alphanumeric() || x == '_')
            };
            if !words.iter().all(|w| valid(w)) {
                return Err(rc_err(198, format!("{} invalid name", words.iter().find(|w| !valid(w)).unwrap_or(&""))));
            }
            words.join(" ")
        }
        _ if lspec.split_whitespace().any(|w| w == "passthru") => format!("{name}({a})"),
        // `string asis` keeps quotes and blanks as typed
        _ if lspec.split_whitespace().any(|w| w == "asis") => a.to_string(),
        _ => strip_outer_quotes(a.trim()).to_string(),
    })
}

/// `"text"` or `` `"text"' `` -> `text`; anything else unchanged.
fn strip_outer_quotes(s: &str) -> &str {
    if let Some(inner) = s.strip_prefix("`\"").and_then(|x| x.strip_suffix("\"'")) {
        return inner;
    }
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') && !s[1..s.len() - 1].contains('"') {
        return &s[1..s.len() - 1];
    }
    s
}

/// The checks a `numlist` element of `syntax` asks for (`min=3 >10 integer`).
fn numlist_rules(lspec: &str) -> macros::NumlistRules {
    let mut r = macros::NumlistRules::default();
    for w in lspec.split_whitespace().skip(1) {
        if let Some(v) = w.strip_prefix("min=") {
            r.min = v.parse().ok();
        } else if let Some(v) = w.strip_prefix("max=") {
            r.max = v.parse().ok();
        } else if w == "integer" {
            r.integer = true;
        } else if w == "ascending" {
            r.ascending = true;
        } else if w == "descending" {
            r.descending = true;
        } else if w == "missingokay" {
            r.missing_ok = true;
        } else if let Some(c) = macros::RangeCheck::parse(w) {
            r.range.push(c);
        }
    }
    r
}

fn marksample(me: &mut AdoInterpreter, r: &str) -> Result<(), String> {
    let (main, opts) = split_opts(r);
    let name = main.trim().to_string();
    me.temp_counter += 1;
    let var = format!("__{:06}", me.temp_counter);
    if let Some(f) = me.tempvars.last_mut() {
        f.push(var.clone());
    }
    me.engine.store.set_local(&name, var.clone());
    let st = &me.engine.store;
    // after `syntax ... [if/]` the local holds the bare expression
    let if_text = st.local("if").cloned().unwrap_or_default();
    let if_text = if if_text.trim().is_empty() || if_text.trim_start().starts_with("if ") { if_text } else { format!("if {if_text}") };
    let text = format!("__m {} {}", if_text, st.local("in").cloned().unwrap_or_default());
    let cmd = parse_line(&text).map_err(|e| rc_err(198, e.message))?;
    let mut mask = util::touse(&me.engine.dataset, &me.engine.store, &cmd)?;
    if !opts.iter().any(|(n, _)| n.starts_with("nov")) {
        let vars: Vec<String> = st.local("varlist").cloned().unwrap_or_default().split_whitespace().map(String::from).collect();
        util::markout(&me.engine.dataset, &mut mask, &vars);
    }
    let mut v = Variable::new(&var, VarType::Byte);
    v.data = mask.iter().map(|&m| Value::Byte(m as i8)).collect();
    me.engine.dataset.add_variable(v)?;
    Ok(())
}

fn log_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (main, opts) = split_opts(r);
    let w: Vec<&str> = main.split_whitespace().collect();
    match w.first().copied() {
        Some("using") => {
            let path = unquote(&main["using".len()..]);
            let has = |n: &str| opts.iter().any(|(o, _)| o.starts_with(n));
            let name = opts.iter().find(|(o, _)| o == "name").and_then(|(_, a)| a.clone()).unwrap_or_else(|| "<unnamed>".into());
            let smcl = !has("text") && !path.ends_with(".log") && !path.ends_with(".txt");
            let path = if std::path::Path::new(&path).extension().is_none() { format!("{path}.{}", if smcl { "smcl" } else { "log" }) } else { path };
            if vfs::exists(&path) && !has("replace") && !has("append") {
                return Err(rc_err(602, format!("file {path} already exists")));
            }
            if me.logs.iter().any(|l| l.name == name) {
                return Err(rc_err(604, "log file already open"));
            }
            let header = if smcl { format!("{{smcl}}\n{{txt}}{{sf}}{{ul off}}{{.-}}\n      name:  {{res}}{name}\n {{txt}}log:  {{res}}{path}\n{{txt}}log type:  {{res}}smcl\n opened on:  {{res}}{}\n", chrono_now()) } else { format!("      name:  {name}\n       log:  {path}\n  log type:  text\n opened on:  {}\n", chrono_now()) };
            if has("append") {
                vfs::append(&path, header.as_bytes()).map_err(|e| rc_err(603, e))?;
            } else {
                vfs::write(&path, header.as_bytes()).map_err(|e| rc_err(603, e))?;
            }
            me.logs.push(crate::interpreter::LogFile { path: path.clone().into(), smcl, on: true, name: name.clone() });
            Ok(format!("      name:  {name}\n       log:  {path}\n  log type:  {}", if smcl { "smcl" } else { "text" }))
        }
        Some("close") => {
            if me.logs.is_empty() {
                return Err(rc_err(606, "no log file open"));
            }
            let which = w.get(1).copied();
            let mut kept = Vec::new();
            let mut closed = Vec::new();
            for l in me.logs.drain(..) {
                if which.map_or(true, |n| n == "_all" || n == l.name) {
                    closed.push(l);
                } else {
                    kept.push(l);
                }
            }
            me.logs = kept;
            let mut out = Vec::new();
            for l in closed {
                {
                    let _ = vfs::append(&l.path.to_string_lossy(), format!("{}      name:  {}\n       log:  {}\n closed on:  {}\n", if l.smcl { "{txt}" } else { "" }, l.name, l.path.display(), chrono_now()).as_bytes());
                }
                out.push(format!("      name:  {}\n       log:  {}", l.name, l.path.display()));
            }
            Ok(out.join("\n"))
        }
        Some("off") => {
            for l in me.logs.iter_mut() {
                l.on = false;
            }
            Ok(String::new())
        }
        Some("on") => {
            for l in me.logs.iter_mut() {
                l.on = true;
            }
            Ok(String::new())
        }
        _ => Ok(me.logs.iter().map(|l| format!("      name:  {}\n       log:  {}", l.name, l.path.display())).collect::<Vec<_>>().join("\n")),
    }
}

fn chrono_now() -> String {
    chrono::Local::now().format("%d %b %Y, %H:%M:%S").to_string()
}

/// `input` block: `input [type] var1 [type] var2 ...` followed by data lines.
pub fn input(me: &mut AdoInterpreter, header: &str, lines: &[String]) -> Result<String, String> {
    let spec = header.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut pending_type: Option<VarType> = None;
    for w in spec.split_whitespace() {
        if let Some(t) = VarType::from_type_str(w) {
            pending_type = Some(t);
            continue;
        }
        names.push(w.to_string());
        types.push(pending_type.take());
    }
    let ds = &mut me.engine.dataset;
    let existing = names.iter().all(|n| ds.var_index.contains_key(n));
    if !existing {
        for (n, t) in names.iter().zip(&types) {
            if ds.var_index.contains_key(n) {
                return Err(rc_err(110, format!("{n} already defined")));
            }
            let ty = t.unwrap_or(me.engine.store.default_type);
            let mut v = Variable::new(n, ty);
            if let VarType::Str(w) = ty {
                v.format = format!("%{w}s");
            }
            ds.add_variable(v)?;
        }
    }
    for line in lines {
        let vals = split_top_level_words(line);
        if vals.is_empty() {
            continue;
        }
        ds.add_observation();
        let i = ds.n_obs - 1;
        for (k, n) in names.iter().enumerate() {
            let raw = vals.get(k).map(|x| unquote(x)).unwrap_or_default();
            let var = ds.get_var_mut(n).unwrap();
            if var.var_type.is_string() {
                if let VarType::Str(w) = var.var_type {
                    if raw.len() > w {
                        var.var_type = VarType::Str(raw.len());
                    }
                }
                var.data[i] = Value::Str(raw);
            } else {
                let v = if raw == "." || raw.is_empty() { ferrostat_core::SYSMISS } else { raw.parse::<f64>().map_err(|_| rc_err(198, format!("cannot be read as a number: {raw}")))? };
                var.data[i] = ferrostat_cmd::data::store_num(var.var_type, v);
            }
        }
    }
    me.engine.store.data_changed = true;
    Ok(String::new())
}

#[allow(dead_code)]
fn unused(_: &str) -> String {
    err_message("")
}

#[cfg(not(target_arch = "wasm32"))]
fn process_id() -> u32 {
    std::process::id()
}

#[cfg(target_arch = "wasm32")]
fn process_id() -> u32 {
    1
}

#[cfg(not(target_arch = "wasm32"))]
fn shell(r: &str) -> Result<String, String> {
    let out = if cfg!(windows) {
        std::process::Command::new("cmd").arg("/C").arg(r).output()
    } else {
        std::process::Command::new("sh").arg("-c").arg(r).output()
    }
    .map_err(|e| rc_err(693, e))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
}

#[cfg(target_arch = "wasm32")]
fn shell(_r: &str) -> Result<String, String> {
    Err(rc_err(199, "shell commands are not available in the browser"))
}
