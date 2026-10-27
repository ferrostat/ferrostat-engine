//! `python`: run Python code in an external interpreter process.
//!
//! One long-lived child process per session runs a small helper
//! (`python_driver.py`) so that Python objects persist between blocks. The
//! helper streams printed text back and calls into the host while code runs
//! (the `sfi` module: data, macros, scalars, matrices, commands).

// the browser build has no subprocesses: only the error paths are used there
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{Value, VarType, Variable};

use crate::interpreter::{err_message, rc_of, AdoInterpreter};
use crate::macros::unquote;
use crate::script;

const US: char = '\x1f';
const RS: char = '\x1e';

/// Return code for an error raised by Python code.
const RC_PYERR: i32 = 7102;

/// Python settings and the running helper process.
#[derive(Default)]
pub struct PyState {
    exec: Option<String>,
    userpath: Vec<String>,
    #[cfg(not(target_arch = "wasm32"))]
    proc: Option<native::PyProc>,
    busy: bool,
}

/// `set python_exec ...` / `set python_userpath ...`
pub fn is_set_python(r: &str) -> bool {
    matches!(r.split_whitespace().next().map(str::to_lowercase).as_deref(), Some("python_exec" | "python_userpath"))
}

pub fn set_option(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (name, rest) = r.trim().split_once(char::is_whitespace).unwrap_or((r.trim(), ""));
    set_cmd(me, if name.eq_ignore_ascii_case("python_exec") { "exec" } else { "userpath" }, rest)
}

fn set_cmd(me: &mut AdoInterpreter, what: &str, rest: &str) -> Result<String, String> {
    let rest = rest.split(',').next().unwrap_or("").trim();
    if what == "exec" {
        let p = unquote(rest);
        me.python.exec = if p.is_empty() { None } else { Some(p) };
        // the next use starts the newly chosen interpreter
        #[cfg(not(target_arch = "wasm32"))]
        {
            me.python.proc = None;
        }
    } else {
        me.python.userpath = rest.split(';').map(|p| unquote(p.trim())).filter(|p| !p.is_empty()).collect();
        #[cfg(not(target_arch = "wasm32"))]
        if me.python.proc.is_some() {
            let mut f = vec!["userpath".to_string(), "0".into()];
            f.extend(me.python.userpath.clone());
            run(me, "admin", f)?;
        }
    }
    Ok(String::new())
}

/// `python ...` written on one line: `python: code` or a subcommand.
pub fn command(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    if let Some(code) = r.strip_prefix(':') {
        let code = code.trim();
        if !code.is_empty() {
            exec_code(me, true, false, code)?;
        }
        return Ok(String::new());
    }
    let (sub, rest) = r.split_once(char::is_whitespace).unwrap_or((r, ""));
    let rest = rest.trim();
    let words = || rest.split_whitespace().map(String::from).collect::<Vec<_>>();
    match sub.to_lowercase().as_str() {
        "query" | "q" => query(me),
        "search" => search(me),
        "set" => {
            let (what, v) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
            match what {
                "exec" => set_cmd(me, "exec", v),
                "userpath" => set_cmd(me, "userpath", v),
                _ => Err(rc_err(198, format!("python set {what}: invalid setting"))),
            }
        }
        "which" => {
            let name = rest.trim();
            if name.is_empty() {
                return Err(rc_err(198, "python which modulename"));
            }
            admin(me, vec!["which".into(), name.into()])
        }
        "describe" | "des" | "d" => admin(me, [vec!["describe".to_string()], words()].concat()),
        "drop" => {
            if rest.is_empty() {
                return Err(rc_err(198, "python drop namelist"));
            }
            admin(me, [vec!["drop".to_string()], words()].concat())
        }
        "clear" => admin(me, vec!["clear".into()]),
        "script" => script_cmd(me, rest),
        "" => Ok(String::new()),
        _ => Err(rc_err(199, format!("python {sub}: unrecognized subcommand"))),
    }
}

/// A `python` ... `end` block. With the colon form the first error stops the
/// block and the command fails; without it every statement runs and errors
/// are only reported, as in an interactive session.
pub fn block(me: &mut AdoInterpreter, colon: bool, raw: &[String]) -> Result<(), String> {
    let code = raw.join("\n");
    let header = if colon { "python:" } else { "python" };
    if me.echo && me.depth == 0 {
        me.out.push(format!(". {header}"));
    }
    if me.depth == 0 {
        me.log_write(header, true);
    }
    let echo = !colon && me.echo && me.depth == 0;
    me.emit(&format!("{} python (type end to exit) {}", "-".repeat(45), "-".repeat(6)));
    let res = exec_code(me, colon, echo, &code);
    if echo {
        me.emit(">>> end");
    }
    me.emit(&"-".repeat(79));
    res
}

fn exec_code(me: &mut AdoInterpreter, stop: bool, echo: bool, code: &str) -> Result<(), String> {
    let ok = run(me, "run", vec![b(stop), b(echo), code.to_string()])?;
    if !ok && stop {
        return Err(rc_err(RC_PYERR, ""));
    }
    Ok(())
}

fn admin(me: &mut AdoInterpreter, fields: Vec<String>) -> Result<String, String> {
    if !run(me, "admin", fields)? {
        return Err(rc_err(RC_PYERR, ""));
    }
    Ok(String::new())
}

/// `python script file.py [, args(...) global userpath(...)]`
fn script_cmd(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (file, opts) = match ferrostat_parser::parser::find_top_level(r, ',') {
        Some(i) => (r[..i].trim(), ferrostat_parser::parser::split_options(&r[i + 1..])),
        None => (r.trim(), Vec::new()),
    };
    let file = unquote(file);
    if file.is_empty() {
        return Err(rc_err(198, "python script filename"));
    }
    if !ferrostat_core::vfs::exists(&file) {
        return Err(rc_err(601, format!("file {file} not found")));
    }
    let mut global = false;
    let mut args = Vec::new();
    for (n, a) in &opts {
        match n.as_str() {
            "global" => global = true,
            "args" => args = crate::macros::words(a.as_deref().unwrap_or("")).iter().map(|w| unquote(w)).collect(),
            "userpath" => {
                let mut f = vec!["userpath".to_string(), "1".into()];
                f.extend(a.as_deref().unwrap_or("").split(';').map(|p| unquote(p.trim())).filter(|p| !p.is_empty()));
                admin(me, f)?;
            }
            _ => return Err(rc_err(198, format!("option {n} not allowed"))),
        }
    }
    let mut f = vec![file, b(global)];
    f.extend(args);
    if !run(me, "script", f)? {
        return Err(rc_err(RC_PYERR, ""));
    }
    Ok(String::new())
}

fn b(v: bool) -> String {
    if v { "1".into() } else { "0".into() }
}

#[cfg(target_arch = "wasm32")]
fn run(_me: &mut AdoInterpreter, _mode: &str, _fields: Vec<String>) -> Result<bool, String> {
    Err(rc_err(199, "python is not available in the browser"))
}

#[cfg(target_arch = "wasm32")]
fn query(_me: &mut AdoInterpreter) -> Result<String, String> {
    Ok("Python is not available in the browser".into())
}

#[cfg(target_arch = "wasm32")]
fn search(_me: &mut AdoInterpreter) -> Result<String, String> {
    Ok("Python is not available in the browser".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn query(me: &mut AdoInterpreter) -> Result<String, String> {
    let started = ensure(me);
    let p = me.python.proc.as_ref();
    let exec = p.map(|p| p.executable.clone()).or_else(|| me.python.exec.clone()).unwrap_or_default();
    let mut out = vec![
        "-".repeat(79),
        "    Python Settings".to_string(),
        format!("      set python_exec      {exec}"),
        format!("      set python_userpath  {}", me.python.userpath.join(";")),
        String::new(),
        "    Python system information".to_string(),
    ];
    match (&started, p) {
        (Ok(()), Some(p)) => {
            out.push("      initialized          yes".into());
            out.push(format!("      version              {}", p.version));
            out.push(format!("      architecture         {}", p.arch));
            out.push(format!("      library path         {}", p.libpath));
        }
        _ => {
            out.push("      initialized          no".into());
            if let Err(e) = &started {
                out.push(format!("      {}", err_message(e).trim()));
            }
        }
    }
    Ok(out.join("\n"))
}

/// `python search`: Python installations on the search path.
#[cfg(not(target_arch = "wasm32"))]
fn search(_me: &mut AdoInterpreter) -> Result<String, String> {
    let names: Vec<String> = if cfg!(windows) {
        vec!["python.exe".into(), "python3.exe".into()]
    } else {
        let mut v = vec!["python3".to_string(), "python".to_string()];
        v.extend((6..=20).map(|k| format!("python3.{k}")));
        v
    };
    let mut found: Vec<String> = Vec::new();
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for n in &names {
                let p = dir.join(n).to_string_lossy().to_string();
                if ferrostat_core::vfs::exists(&p) && !found.contains(&p) {
                    found.push(p);
                }
            }
        }
    }
    if found.is_empty() {
        return Ok("no Python installation found".into());
    }
    let mut out = vec!["Python environments found:".to_string()];
    out.extend(found);
    Ok(out.join("\n"))
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure(me: &mut AdoInterpreter) -> Result<(), String> {
    if me.python.proc.as_mut().is_some_and(|p| p.alive()) {
        return Ok(());
    }
    me.python.proc = None;
    let candidates: Vec<String> = match &me.python.exec {
        Some(e) => vec![e.clone()],
        None if cfg!(windows) => vec!["python".into(), "python3".into(), "py".into()],
        None => vec!["python3".into(), "python".into()],
    };
    let mut last = String::new();
    for c in &candidates {
        match native::PyProc::spawn(c) {
            Ok(p) => {
                me.python.proc = Some(p);
                if !me.python.userpath.is_empty() {
                    let mut f = vec!["userpath".to_string(), "0".into()];
                    f.extend(me.python.userpath.clone());
                    run(me, "admin", f)?;
                }
                return Ok(());
            }
            Err(e) => last = format!("{c}: {e}"),
        }
    }
    Err(rc_err(7100, format!("Python installation not found ({last}); use set python_exec to choose one")))
}

/// Send one request to the helper and serve its callbacks until it is done.
/// Returns whether the Python code ran without error.
#[cfg(not(target_arch = "wasm32"))]
fn run(me: &mut AdoInterpreter, mode: &str, fields: Vec<String>) -> Result<bool, String> {
    if me.python.busy {
        return Err(rc_err(RC_PYERR, "python cannot be called from within python"));
    }
    ensure(me)?;
    let mut proc = me.python.proc.take().unwrap();
    let cwd = ferrostat_core::vfs::current_dir();
    let mut payload = vec![mode.to_string(), cwd];
    payload.extend(fields);
    me.python.busy = true;
    let res = serve(me, &mut proc, &payload.join(&US.to_string()));
    me.python.busy = false;
    match res {
        Ok(ok) => {
            me.python.proc = Some(proc);
            Ok(ok)
        }
        // the process died or the pipe broke: start afresh next time
        Err(e) => Err(rc_err(RC_PYERR, format!("python process terminated: {e}"))),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn serve(me: &mut AdoInterpreter, proc: &mut native::PyProc, payload: &str) -> std::io::Result<bool> {
    proc.send("EXEC", payload)?;
    let mut pending = String::new();
    loop {
        let (tag, body) = proc.recv()?;
        match tag.as_str() {
            "OUT" | "ERR" => pending.push_str(&body),
            "CALL" => {
                flush(me, &mut pending);
                match callback(me, &body) {
                    Ok(v) => proc.send("RET", &v)?,
                    Err(e) => proc.send("FAIL", &e)?,
                }
            }
            "DONE" => {
                flush(me, &mut pending);
                return Ok(body == "0");
            }
            _ => {}
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn flush(me: &mut AdoInterpreter, pending: &mut String) {
    if !pending.is_empty() {
        let t = std::mem::take(pending);
        me.emit(t.strip_suffix('\n').unwrap_or(&t));
    }
}

/// Serve a request made by Python code through the `sfi` module.
fn callback(me: &mut AdoInterpreter, body: &str) -> Result<String, String> {
    let f: Vec<&str> = body.split(US).collect();
    let arg = |k: usize| f.get(k).copied().unwrap_or("");
    let ds = &me.engine.dataset;
    let var_at = |i: &str| -> Result<String, String> {
        let k: usize = i.parse().map_err(|_| format!("invalid variable index {i}"))?;
        ds.vars.get(k).map(|v| v.name.clone()).ok_or_else(|| format!("variable index {k} out of range"))
    };
    let var = |n: &str| ds.get_var(n).ok_or_else(|| format!("variable {n} not found"));
    Ok(match f[0] {
        "obstotal" => ds.n_obs.to_string(),
        "varcount" => ds.vars.len().to_string(),
        "varname" => var_at(arg(1))?,
        "varindex" => {
            let name = varlist::expand(ds, arg(1)).map_err(|e| err_message(&e))?.into_iter().next().unwrap_or_default();
            ds.var_index.get(&name).map(|i| i.to_string()).ok_or_else(|| format!("variable {} not found", arg(1)))?
        }
        "varlist" => varlist::expand(ds, arg(1)).map_err(|e| err_message(&e))?.join(&US.to_string()),
        "vartype" => var(arg(1))?.var_type.type_str(),
        "varlabel" => var(arg(1))?.label.clone(),
        "varformat" => var(arg(1))?.format.clone(),
        "getcols" => {
            let labels = arg(1) == "1";
            let mut cols = Vec::new();
            for n in &f[2..] {
                let v = var(n)?;
                let vl = if labels { ds.value_labels.get(&v.value_label_name) } else { None };
                let cells: Vec<String> = (0..ds.n_obs)
                    .map(|i| match v.data.get(i) {
                        Some(Value::Str(s)) => format!("s{s}"),
                        Some(Value::StrL(a, o)) => format!("s{}", ds.resolve_strl(*a, *o).unwrap_or("")),
                        Some(x) => {
                            let x = x.num();
                            if let Some(l) = vl.and_then(|vl| vl.lookup(x)) {
                                format!("s{l}")
                            } else if is_missing(x) {
                                "m".into()
                            } else if matches!(v.var_type, VarType::Byte | VarType::Int | VarType::Long) {
                                format!("i{x}")
                            } else {
                                format!("n{x:?}")
                            }
                        }
                        None => if v.var_type.is_string() { "s".into() } else { "m".into() },
                    })
                    .collect();
                cols.push(cells.join(&US.to_string()));
            }
            cols.join(&RS.to_string())
        }
        "setcol" => {
            let obs: Vec<usize> = arg(2).split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(usize::MAX)).collect();
            set_column(me, arg(1), &obs, &f[3..])?;
            String::new()
        }
        "setvarlabel" | "setvarformat" => {
            let v = me.engine.dataset.get_var_mut(arg(1)).ok_or_else(|| format!("variable {} not found", arg(1)))?;
            if f[0] == "setvarlabel" { v.label = arg(2).to_string() } else { v.format = arg(2).to_string() }
            String::new()
        }
        "addvar" => {
            let ty = VarType::from_type_str(arg(2)).ok_or_else(|| format!("invalid type {}", arg(2)))?;
            let name = arg(1);
            if me.engine.dataset.get_var(name).is_some() {
                return Err(format!("variable {name} already defined"));
            }
            let mut v = Variable::new(name, ty);
            if ty.is_string() {
                v.format = format!("%{}s", if let VarType::Str(n) = ty { n } else { 9 });
            }
            for _ in 0..me.engine.dataset.n_obs {
                v.push_missing();
            }
            me.engine.dataset.add_variable(v)?;
            me.engine.store.data_changed = true;
            String::new()
        }
        "dropvar" => {
            for n in &f[1..] {
                me.engine.dataset.drop_variable(n)?;
            }
            me.engine.store.data_changed = true;
            String::new()
        }
        "renamevar" => {
            let (old, new) = (arg(1), arg(2));
            let cmd = format!("rename {old} {new}");
            run_commands(me, &cmd)?;
            String::new()
        }
        "setobs" => {
            let n: usize = arg(1).parse().map_err(|_| "invalid number of observations".to_string())?;
            let ds = &mut me.engine.dataset;
            if n < ds.n_obs {
                return Err("the number of observations cannot be reduced".into());
            }
            while ds.n_obs < n {
                ds.add_observation();
            }
            me.engine.store.data_changed = true;
            String::new()
        }
        "getlocal" => me.engine.store.local(arg(1)).cloned().unwrap_or_default(),
        "setlocal" => {
            me.engine.store.set_local(arg(1), arg(2));
            String::new()
        }
        "getglobal" => {
            let st = &me.engine.store;
            match result_name(arg(1)) {
                Some((c, n)) => results(st, c).macros.get(n).cloned().unwrap_or_default(),
                None => st.globals.get(arg(1)).cloned().unwrap_or_default(),
            }
        }
        "setglobal" => {
            let st = &mut me.engine.store;
            match result_name(arg(1)) {
                Some((c, n)) => {
                    results_mut(st, c).set_str(n, arg(2));
                }
                None => {
                    st.globals.insert(arg(1).to_string(), arg(2).to_string());
                }
            }
            String::new()
        }
        "getscalar" => {
            let st = &me.engine.store;
            let v = match result_name(arg(1)) {
                Some((c, n)) => results(st, c).get(n),
                None => st.scalars.get(arg(1)).copied(),
            };
            match v {
                Some(x) if is_missing(x) => "m".into(),
                Some(x) => format!("n{x:?}"),
                None => String::new(),
            }
        }
        "setscalar" => {
            let v: f64 = arg(2).parse().unwrap_or(SYSMISS);
            let v = if v.is_nan() || v >= SYSMISS { SYSMISS } else { v };
            let st = &mut me.engine.store;
            match result_name(arg(1)) {
                Some((c, n)) => results_mut(st, c).set(n, v),
                None => {
                    st.str_scalars.remove(arg(1));
                    st.scalars.insert(arg(1).to_string(), v);
                }
            }
            String::new()
        }
        "getstrscalar" => me.engine.store.str_scalars.get(arg(1)).cloned().unwrap_or_default(),
        "setstrscalar" => {
            let st = &mut me.engine.store;
            st.scalars.remove(arg(1));
            st.str_scalars.insert(arg(1).to_string(), arg(2).to_string());
            String::new()
        }
        "getmatrix" => {
            let m = matrix(me, arg(1))?;
            let mut cells = vec![m.rows.to_string(), m.cols.to_string()];
            cells.extend(m.data.iter().map(|x| if is_missing(*x) { "m".into() } else { format!("n{x:?}") }));
            cells.join(&US.to_string())
        }
        "matnames" => {
            let m = matrix(me, arg(1))?;
            if arg(2) == "row" { m.rownames.join(&US.to_string()) } else { m.colnames.join(&US.to_string()) }
        }
        "setmatrix" => {
            let (r, c): (usize, usize) = (arg(2).parse().unwrap_or(0), arg(3).parse().unwrap_or(0));
            let vals: Vec<f64> = f[4..].iter().map(|x| x.strip_prefix('n').and_then(|v| v.parse().ok()).unwrap_or(SYSMISS)).collect();
            if vals.len() != r * c {
                return Err("matrix dimensions do not match the values".into());
            }
            let mut m = Matrix::zeros(r, c);
            m.data = vals;
            let st = &mut me.engine.store;
            match result_name(arg(1)) {
                Some((k, n)) => results_mut(st, k).set_mat(n, m),
                None => {
                    st.matrices.insert(arg(1).to_string(), m);
                }
            }
            String::new()
        }
        "setmatnames" => {
            let names: Vec<String> = f[3..].iter().map(|s| s.to_string()).collect();
            let st = &mut me.engine.store;
            let m = match result_name(arg(1)) {
                Some((k, n)) => results_mut(st, k).matrices.get_mut(n),
                None => st.matrices.get_mut(arg(1)),
            }
            .ok_or_else(|| format!("matrix {} not found", arg(1)))?;
            let want = if arg(2) == "row" { m.rows } else { m.cols };
            if names.len() != want {
                return Err("number of names does not match the matrix".into());
            }
            if arg(2) == "row" { m.rownames = names } else { m.colnames = names }
            String::new()
        }
        "command" => {
            let cmd = arg(1).to_string();
            if arg(2) == "1" {
                me.emit(&format!(". {cmd}"));
            }
            run_commands(me, &cmd)?;
            String::new()
        }
        "expand" => {
            let eng = &mut me.engine;
            crate::macros::expand(arg(1), &eng.dataset, &mut eng.store).map_err(|e| err_message(&e))?
        }
        "format" => util::fmt_cell(arg(1).parse().unwrap_or(SYSMISS), arg(2)).trim().to_string(),
        other => return Err(format!("unknown request {other}")),
    })
}

/// Run command text on behalf of Python; errors carry their return code.
fn run_commands(me: &mut AdoInterpreter, cmd: &str) -> Result<(), String> {
    let lines = script::logical_lines(cmd);
    let stmts = script::parse_block(&lines).map_err(|e| format!("{}\nr({});", err_message(&e), rc_of(&e)))?;
    me.exec_block(&stmts).map(|_| ()).map_err(|e| {
        let msg = err_message(&e);
        if !msg.trim().is_empty() {
            me.emit(msg.trim_end());
        }
        format!("command failed with r({});", rc_of(&e))
    })
}

/// `r(name)` / `e(name)` / `s(name)`: the result class and the name.
fn result_name(n: &str) -> Option<(char, &str)> {
    let c = n.chars().next()?;
    if !matches!(c, 'r' | 'e' | 's') {
        return None;
    }
    n[1..].strip_prefix('(')?.strip_suffix(')').map(|x| (c, x))
}

fn results(st: &ferrostat_cmd::store::Store, c: char) -> &ferrostat_cmd::store::Results {
    match c {
        'r' => &st.r,
        'e' => &st.e,
        _ => &st.s,
    }
}

fn results_mut(st: &mut ferrostat_cmd::store::Store, c: char) -> &mut ferrostat_cmd::store::Results {
    match c {
        'r' => &mut st.r,
        'e' => &mut st.e,
        _ => &mut st.s,
    }
}

fn matrix(me: &AdoInterpreter, name: &str) -> Result<Matrix, String> {
    let st = &me.engine.store;
    match result_name(name) {
        Some((c, n)) => results(st, c).matrices.get(n),
        None => st.matrices.get(name),
    }
    .cloned()
    .ok_or_else(|| format!("matrix {name} not found"))
}

/// Store cells sent by Python into a variable, widening its type as needed.
fn set_column(me: &mut AdoInterpreter, name: &str, obs: &[usize], cells: &[&str]) -> Result<(), String> {
    let ds = &mut me.engine.dataset;
    let n = ds.n_obs;
    let v = ds.get_var_mut(name).ok_or_else(|| format!("variable {name} not found"))?;
    if obs.len() != cells.len() || obs.iter().any(|&o| o >= n) {
        return Err("observation out of range".into());
    }
    if v.var_type.is_string() {
        let mut width = match v.var_type {
            VarType::Str(w) => w,
            _ => 0,
        };
        for (&o, c) in obs.iter().zip(cells) {
            let s = match c.strip_prefix('s') {
                Some(s) => s,
                None if *c == "m" => "",
                None => return Err(format!("type mismatch: {name} is a string variable")),
            };
            width = width.max(s.len());
            v.set(o, Value::Str(s.to_string()));
        }
        if v.var_type != VarType::StrL {
            v.var_type = util::str_type_for(width);
        }
    } else {
        let mut vals = Vec::with_capacity(cells.len());
        for c in cells {
            vals.push(match c.strip_prefix('n') {
                Some(x) => x.parse::<f64>().map_err(|_| format!("invalid number {x}"))?,
                None if *c == "m" => SYSMISS,
                None => return Err(format!("type mismatch: {name} is a numeric variable")),
            });
        }
        let mut ty = v.var_type;
        for &x in &vals {
            if !util::fits(ty, x) {
                ty = util::promote(ty, util::best_numeric_type(&[x]));
            }
        }
        if ty != v.var_type {
            v.data = v.data.iter().map(|x| util::typed_value(ty, x.num())).collect();
            v.var_type = ty;
        }
        for (&o, &x) in obs.iter().zip(&vals) {
            v.set(o, util::typed_value(ty, x));
        }
    }
    me.engine.store.data_changed = true;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

    const DRIVER: &str = include_str!("python_driver.py");
    /// Reads the helper source from stdin, so its size is not limited by the command line.
    const BOOT: &str = "import sys;n=int(sys.stdin.buffer.readline());exec(compile(sys.stdin.buffer.read(n),'<ferrostat>','exec'))";

    pub struct PyProc {
        child: Child,
        tx: ChildStdin,
        rx: BufReader<ChildStdout>,
        pub version: String,
        pub executable: String,
        pub arch: String,
        pub libpath: String,
    }

    impl Drop for PyProc {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl PyProc {
        pub fn spawn(exe: &str) -> io::Result<PyProc> {
            let mut child = Command::new(exe)
                .args(["-u", "-c", BOOT])
                .env("PYTHONIOENCODING", "utf-8")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()?;
            let mut tx = child.stdin.take().unwrap();
            let rx = BufReader::new(child.stdout.take().unwrap());
            tx.write_all(format!("{}\n{DRIVER}", DRIVER.len()).as_bytes())?;
            tx.flush()?;
            let mut p = PyProc { child, tx, rx, version: String::new(), executable: exe.to_string(), arch: String::new(), libpath: String::new() };
            let (tag, body) = p.recv()?;
            if tag != "HELLO" {
                return Err(io::Error::other("unexpected reply from python"));
            }
            let f: Vec<&str> = body.split(super::US).collect();
            p.version = f.first().unwrap_or(&"").to_string();
            if let Some(e) = f.get(1).filter(|e| !e.is_empty()) {
                p.executable = e.to_string();
            }
            p.arch = f.get(2).unwrap_or(&"").to_string();
            p.libpath = f.get(3).unwrap_or(&"").to_string();
            Ok(p)
        }

        pub fn alive(&mut self) -> bool {
            matches!(self.child.try_wait(), Ok(None))
        }

        pub fn send(&mut self, tag: &str, payload: &str) -> io::Result<()> {
            self.tx.write_all(format!("{tag} {}\n", payload.len()).as_bytes())?;
            self.tx.write_all(payload.as_bytes())?;
            self.tx.flush()
        }

        pub fn recv(&mut self) -> io::Result<(String, String)> {
            let mut head = String::new();
            if self.rx.read_line(&mut head)? == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "python exited"));
            }
            let (tag, n) = head.trim_end().split_once(' ').ok_or_else(|| io::Error::other("bad frame from python"))?;
            let n: usize = n.parse().map_err(|_| io::Error::other("bad frame from python"))?;
            let mut buf = vec![0u8; n];
            self.rx.read_exact(&mut buf)?;
            Ok((tag.to_string(), String::from_utf8_lossy(&buf).into_owned()))
        }
    }
}
