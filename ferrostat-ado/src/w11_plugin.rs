//! Compiled C plugins: `program name, plugin [using(filename)]` loads a
//! shared library, and `plugin call name [varlist] [if] [in] [, args]`
//! runs it.
//!
//! The plugin exports an initialisation function (`pginit`), which
//! receives a pointer to a table of callbacks ([`CallbackTable`]) and
//! returns the interface version it was built for, and an entry function
//! called with the arguments as `(argc, argv)` returning a return code.
//! Through the callbacks the plugin reads and writes the variables of the
//! varlist, macros, scalars and matrices, tests the `if`/`in` restriction
//! and displays text. Indices of variables (within the varlist) and
//! observations are 1-based.
//!
//! `plugin header using file.h [, replace]` writes a C header describing
//! the table, the access macros and the entry point, for writing plugins.
//!
//! Plugins are native code: the browser build reports an error.

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::store::Program;
use ferrostat_core::vfs;
use ferrostat_parser::parser::find_top_level;

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;
use crate::progtools::{has_opt, opt_arg, split_opts};

/// Name of the plugin entry function (fixed by the plugin interface).
pub fn entry_symbol() -> String {
    ["st", "ata", "_call"].concat()
}

/// Interface version implemented (major * 1000 + minor).
pub const INTERFACE_VERSION: i32 = 3000;

/// Is this line a one-line plugin program definition
/// (`program [define] name, plugin [using(...)]`)?
pub fn is_plugin_header(line: &str) -> bool {
    let t = line.trim();
    let mut w = t.split_whitespace();
    let first = w.next().unwrap_or("").to_lowercase();
    if !matches!(first.as_str(), "program" | "pr" | "prog" | "progr" | "progra") {
        return false;
    }
    is_plugin_def(t[first.len()..].trim())
}

/// The arguments of `program` define a plugin.
pub fn is_plugin_def(rest: &str) -> bool {
    match find_top_level(rest, ',') {
        Some(i) => {
            let opts = ferrostat_parser::parser::split_options(&rest[i + 1..]);
            has_opt(&opts, "plugin", 6)
        }
        None => false,
    }
}

/// Loaded plugins, by program name.
#[derive(Default)]
pub struct Plugins {
    #[cfg(not(target_arch = "wasm32"))]
    loaded: std::collections::BTreeMap<String, native::Loaded>,
}

/// `program [define] name, plugin [using(filename)]`.
pub fn define(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let (head, opts) = split_opts(rest);
    let mut words: Vec<&str> = head.split_whitespace().collect();
    if words.first().is_some_and(|w| *w == "define" || *w == "def") {
        words.remove(0);
    }
    let name = match words.as_slice() {
        [n] => n.to_string(),
        _ => return Err(rc_err(198, "program name, plugin [using(filename)]")),
    };
    if me.engine.store.programs.contains_key(&name) {
        return Err(rc_err(110, format!("program {name} already defined")));
    }
    let file = opt_arg(&opts, "using", 5).map(|f| unquote(&f)).unwrap_or_else(|| format!("{name}.plugin"));
    let path = resolve(me, &file).ok_or_else(|| rc_err(601, format!("file {file} not found")))?;
    load(me, &name, &path)?;
    let mut o = format!("plugin using(\"{path}\")");
    if has_opt(&opts, "rclass", 6) {
        o.push_str(" rclass");
    }
    me.engine.store.programs.insert(name.clone(), Program { name, body: Vec::new(), class: "nclass".into(), byable: false, opts: o });
    Ok(String::new())
}

fn resolve(me: &AdoInterpreter, file: &str) -> Option<String> {
    let mut cands = vec![file.to_string()];
    if std::path::Path::new(file).extension().is_none() {
        cands.push(format!("{file}.plugin"));
    }
    for c in &cands {
        if vfs::exists(c) {
            let p = std::path::Path::new(c);
            return Some(if p.is_absolute() { c.clone() } else { std::path::Path::new(&vfs::current_dir()).join(p).display().to_string() });
        }
    }
    cands.iter().find_map(|c| crate::progtools::find_on_path(me, c, None, true, false).into_iter().next())
}

/// `plugin call name [varlist] [if] [in] [, arguments]`,
/// `plugin header using file.h [, replace]`.
pub fn plugin(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let r = r.trim();
    let (sub, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((r, ""));
    match sub {
        "call" => call(me, rest),
        "header" => header(rest),
        _ => Err(rc_err(198, "plugin call name [varlist] [if] [in] [, arguments]")),
    }
}

fn header(rest: &str) -> Result<String, String> {
    let (body, opts) = split_opts(rest);
    let file = unquote(body.trim().strip_prefix("using").map(str::trim).unwrap_or(body.trim()));
    if file.is_empty() {
        return Err(rc_err(198, "plugin header using filename [, replace]"));
    }
    if vfs::exists(&file) && !has_opt(&opts, "replace", 3) {
        return Err(rc_err(602, format!("file {file} already exists")));
    }
    vfs::write(&file, c_header().as_bytes()).map_err(|e| rc_err(603, format!("file {file} could not be written: {e}")))?;
    Ok(format!("(file {file} written)"))
}

/// The call arguments: the program name, the command text (varlist, if,
/// in) and the words after the comma.
#[cfg(not(target_arch = "wasm32"))]
fn split_call(rest: &str) -> (String, String, Vec<String>) {
    let (main, args) = match find_top_level(rest, ',') {
        Some(i) => (&rest[..i], rest[i + 1..].trim()),
        None => (rest, ""),
    };
    let main = main.trim();
    let (name, tail) = main.split_once(char::is_whitespace).map(|(a, b)| (a.to_string(), b.trim().to_string())).unwrap_or((main.to_string(), String::new()));
    let args = crate::macros::words(args).into_iter().map(|w| unquote(&w)).collect();
    (name, tail, args)
}

#[cfg(target_arch = "wasm32")]
fn load(_me: &mut AdoInterpreter, name: &str, _path: &str) -> Result<(), String> {
    Err(rc_err(199, format!("program {name}: compiled plugins need the desktop version of Ferrostat")))
}

#[cfg(target_arch = "wasm32")]
fn call(_me: &mut AdoInterpreter, _rest: &str) -> Result<String, String> {
    Err(rc_err(199, "plugin call: compiled plugins need the desktop version of Ferrostat"))
}

#[cfg(not(target_arch = "wasm32"))]
fn load(me: &mut AdoInterpreter, name: &str, path: &str) -> Result<(), String> {
    let l = native::Loaded::open(path)?;
    me.w11.plugins.loaded.insert(name.to_string(), l);
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn call(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    use ferrostat_parser::parser::parse_line;
    let (name, tail, args) = split_call(rest);
    if name.is_empty() {
        return Err(rc_err(198, "plugin call name [varlist] [if] [in] [, arguments]"));
    }
    let prog = me.engine.store.programs.get(&name).cloned();
    let is_plugin = prog.as_ref().is_some_and(|p| p.opts.starts_with("plugin"));
    if !is_plugin {
        return Err(rc_err(199, format!("plugin {name} not found")));
    }
    if !me.w11.plugins.loaded.contains_key(&name) {
        // dropped and redefined elsewhere: reload from the recorded path
        let opts = prog.map(|p| p.opts).unwrap_or_default();
        let path = opts.split_once("using(\"").and_then(|(_, b)| b.split_once("\")")).map(|(a, _)| a.to_string()).unwrap_or_default();
        load(me, &name, &path)?;
    }
    let cmd = parse_line(&format!("plugin {tail}")).map_err(|e| rc_err(198, e.message))?;
    let ds = &me.engine.dataset;
    let vars: Vec<usize> = if cmd.main.trim().is_empty() {
        Vec::new()
    } else {
        let names = ferrostat_cmd::varlist::expand(ds, cmd.main.trim())?;
        names.iter().map(|n| ds.vars.iter().position(|v| &v.name == n).ok_or_else(|| rc_err(111, format!("variable {n} not found")))).collect::<Result<_, _>>()?
    };
    let n = ds.n_obs;
    let ifmask: Vec<bool> = match &cmd.if_expr {
        Some(_) => {
            let mut c2 = cmd.clone();
            c2.in_range = None;
            ferrostat_cmd::util::touse(ds, &me.engine.store, &c2)?
        }
        None => vec![true; n],
    };
    let (in1, in2) = match &cmd.in_range {
        Some(_) => {
            let mut c2 = cmd.clone();
            c2.if_expr = None;
            c2.if_raw = None;
            let m = ferrostat_cmd::util::touse(ds, &me.engine.store, &c2)?;
            let first = m.iter().position(|&b| b).map(|i| i + 1).unwrap_or(1);
            let last = m.iter().rposition(|&b| b).map(|i| i + 1).unwrap_or(0);
            (first, last)
        }
        None => (1, n),
    };
    let entry = me.w11.plugins.loaded.get(&name).map(|l| l.entry).ok_or_else(|| rc_err(199, format!("plugin {name} not loaded")))?;
    let (rc, out) = native::run(me, entry, vars, ifmask, in1, in2, &args);
    let text = ferrostat_cmd::w11::smcl::to_text(&out);
    if rc != 0 {
        return Err(rc_err(rc, text));
    }
    Ok(text)
}

/// C header for writing plugins against this interface.
pub fn c_header() -> String {
    let entry = entry_symbol();
    let table = [
        ("int", "type", ""),
        ("int (*", "display", ")(const char *)"),
        ("int (*", "display_raw", ")(const char *)"),
        ("int (*", "flush", ")(void)"),
        ("int (*", "error", ")(const char *)"),
        ("int (*", "error_raw", ")(const char *)"),
        ("int (*", "poll", ")(void)"),
        ("int (*", "vdata", ")(int var, int obs, double *z)"),
        ("int (*", "vstore", ")(int var, int obs, double z)"),
        ("int (*", "sdata", ")(int var, int obs, char *buf)"),
        ("int (*", "sstore", ")(int var, int obs, const char *s)"),
        ("int (*", "nobs", ")(void)"),
        ("int (*", "nvar", ")(void)"),
        ("int (*", "mat_el", ")(const char *name, int i, int j, double *z)"),
        ("int (*", "mat_store", ")(const char *name, int i, int j, double z)"),
        ("int (*", "col", ")(const char *name)"),
        ("int (*", "row", ")(const char *name)"),
        ("int (*", "macro_save", ")(const char *name, const char *value)"),
        ("int (*", "macro_use", ")(const char *name, char *buf, int size)"),
        ("int (*", "scal_save", ")(const char *name, double z)"),
        ("int (*", "scal_use", ")(const char *name, double *z)"),
        ("int (*", "ifobs", ")(int obs)"),
        ("int (*", "in1", ")(void)"),
        ("int (*", "in2", ")(void)"),
        ("int (*", "nvars", ")(void)"),
        ("double", "missval", ""),
        ("int (*", "is_missing", ")(double z)"),
        ("int (*", "var_is_string", ")(int var)"),
        ("int (*", "var_is_strl", ")(int var)"),
        ("int (*", "var_is_binary", ")(int var, int obs)"),
        ("int (*", "sdatalen", ")(int var, int obs)"),
        ("int (*", "strldata", ")(int var, int obs, char *buf, int size)"),
    ];
    let mut s = String::from(
        "/* Ferrostat plugin interface (generated by `plugin header`).\n * Build:  cc -shared -fPIC -o name.plugin name.c\n * Variables (within the varlist) and observations are 1-based. */\n#ifndef FS_PLUGIN_H\n#define FS_PLUGIN_H\n\n#if defined(_WIN32)\n#define FS_EXPORT __declspec(dllexport)\n#else\n#define FS_EXPORT __attribute__((visibility(\"default\")))\n#endif\n\ntypedef int ST_int;\ntypedef int ST_retcode;\ntypedef double ST_double;\n\ntypedef struct {\n",
    );
    for (a, b, c) in table {
        if c.is_empty() {
            s.push_str(&format!("    {a} {b};\n"));
        } else {
            s.push_str(&format!("    {a}{b}{c};\n"));
        }
    }
    s.push_str("} fs_table;\n\nstatic fs_table *fs_tbl;\n\n");
    s.push_str(&format!("FS_EXPORT int pginit(fs_table *p) {{ fs_tbl = p; return {INTERFACE_VERSION}; }}\n"));
    s.push_str(&format!("#define FS_ENTRY FS_EXPORT ST_retcode {entry}\n\n"));
    for (m, call) in [
        ("SF_display(s)", "display(s)"),
        ("SF_error(s)", "error(s)"),
        ("SF_poll()", "poll()"),
        ("SF_vdata(i, j, z)", "vdata(i, j, z)"),
        ("SF_vstore(i, j, z)", "vstore(i, j, z)"),
        ("SF_sdata(i, j, s)", "sdata(i, j, s)"),
        ("SF_sstore(i, j, s)", "sstore(i, j, s)"),
        ("SF_nobs()", "nobs()"),
        ("SF_nvar()", "nvar()"),
        ("SF_nvars()", "nvars()"),
        ("SF_mat_el(m, i, j, z)", "mat_el(m, i, j, z)"),
        ("SF_mat_store(m, i, j, z)", "mat_store(m, i, j, z)"),
        ("SF_col(m)", "col(m)"),
        ("SF_row(m)", "row(m)"),
        ("SF_macro_save(n, v)", "macro_save(n, v)"),
        ("SF_macro_use(n, b, l)", "macro_use(n, b, l)"),
        ("SF_scal_save(n, z)", "scal_save(n, z)"),
        ("SF_scal_use(n, z)", "scal_use(n, z)"),
        ("SF_ifobs(j)", "ifobs(j)"),
        ("SF_in1()", "in1()"),
        ("SF_in2()", "in2()"),
        ("SF_is_missing(z)", "is_missing(z)"),
        ("SF_var_is_string(i)", "var_is_string(i)"),
        ("SF_var_is_strl(i)", "var_is_strl(i)"),
        ("SF_var_is_binary(i, j)", "var_is_binary(i, j)"),
        ("SF_sdatalen(i, j)", "sdatalen(i, j)"),
        ("SF_strldata(i, j, b, l)", "strldata(i, j, b, l)"),
    ] {
        s.push_str(&format!("#define {m} (fs_tbl->{call})\n"));
    }
    s.push_str("#define SV_missval (fs_tbl->missval)\n\n#endif\n");
    s
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use std::cell::Cell;
    use std::ffi::{c_char, CStr, CString};

    use ferrostat_core::missing::{is_missing, SYSMISS};
    use ferrostat_core::Value;

    type EntryFn = unsafe extern "C" fn(i32, *mut *mut c_char) -> i32;
    type InitFn = unsafe extern "C" fn(*mut CallbackTable) -> i32;

    /// The callback table handed to `pginit` (layout of the C header).
    #[repr(C)]
    pub struct CallbackTable {
        pub type_: i32,
        pub display: extern "C" fn(*const c_char) -> i32,
        pub display_raw: extern "C" fn(*const c_char) -> i32,
        pub flush: extern "C" fn() -> i32,
        pub error: extern "C" fn(*const c_char) -> i32,
        pub error_raw: extern "C" fn(*const c_char) -> i32,
        pub poll: extern "C" fn() -> i32,
        pub vdata: extern "C" fn(i32, i32, *mut f64) -> i32,
        pub vstore: extern "C" fn(i32, i32, f64) -> i32,
        pub sdata: extern "C" fn(i32, i32, *mut c_char) -> i32,
        pub sstore: extern "C" fn(i32, i32, *const c_char) -> i32,
        pub nobs: extern "C" fn() -> i32,
        pub nvar: extern "C" fn() -> i32,
        pub mat_el: extern "C" fn(*const c_char, i32, i32, *mut f64) -> i32,
        pub mat_store: extern "C" fn(*const c_char, i32, i32, f64) -> i32,
        pub col: extern "C" fn(*const c_char) -> i32,
        pub row: extern "C" fn(*const c_char) -> i32,
        pub macro_save: extern "C" fn(*const c_char, *const c_char) -> i32,
        pub macro_use: extern "C" fn(*const c_char, *mut c_char, i32) -> i32,
        pub scal_save: extern "C" fn(*const c_char, f64) -> i32,
        pub scal_use: extern "C" fn(*const c_char, *mut f64) -> i32,
        pub ifobs: extern "C" fn(i32) -> i32,
        pub in1: extern "C" fn() -> i32,
        pub in2: extern "C" fn() -> i32,
        pub nvars: extern "C" fn() -> i32,
        pub missval: f64,
        pub is_missing: extern "C" fn(f64) -> i32,
        pub var_is_string: extern "C" fn(i32) -> i32,
        pub var_is_strl: extern "C" fn(i32) -> i32,
        pub var_is_binary: extern "C" fn(i32, i32) -> i32,
        pub sdatalen: extern "C" fn(i32, i32) -> i32,
        pub strldata: extern "C" fn(i32, i32, *mut c_char, i32) -> i32,
    }

    pub struct Loaded {
        // kept alive while the entry point may be called
        _lib: libloading::Library,
        pub entry: EntryFn,
    }

    impl Loaded {
        pub fn open(path: &str) -> Result<Self, String> {
            // SAFETY: loading a shared library runs its initialisers; the
            // user asked for this library to be loaded.
            let lib = unsafe { libloading::Library::new(path) }.map_err(|e| rc_err(601, format!("plugin {path} could not be loaded: {e}")))?;
            let init: InitFn = unsafe { lib.get::<InitFn>(b"pginit\0") }.map(|s| *s).map_err(|_| rc_err(198, format!("{path} is not a plugin (no pginit)")))?;
            let mut name = entry_symbol().into_bytes();
            name.push(0);
            let entry: EntryFn = unsafe { lib.get::<EntryFn>(&name) }.map(|s| *s).map_err(|_| rc_err(198, format!("{path} is not a plugin (no entry point)")))?;
            // the table lives as long as the process: plugins keep the pointer
            let table: &'static mut CallbackTable = Box::leak(Box::new(table()));
            let ver = unsafe { init(table) };
            if ver < 2000 || ver / 1000 > INTERFACE_VERSION / 1000 {
                return Err(rc_err(198, format!("plugin {path}: interface version {ver} not supported")));
            }
            Ok(Loaded { _lib: lib, entry })
        }
    }

    fn table() -> CallbackTable {
        CallbackTable {
            type_: 1,
            display: cb_display,
            display_raw: cb_display_raw,
            flush: cb_zero,
            error: cb_error,
            error_raw: cb_error,
            poll: cb_zero,
            vdata: cb_vdata,
            vstore: cb_vstore,
            sdata: cb_sdata,
            sstore: cb_sstore,
            nobs: cb_nobs,
            nvar: cb_nvar_total,
            mat_el: cb_mat_el,
            mat_store: cb_mat_store,
            col: cb_col,
            row: cb_row,
            macro_save: cb_macro_save,
            macro_use: cb_macro_use,
            scal_save: cb_scal_save,
            scal_use: cb_scal_use,
            ifobs: cb_ifobs,
            in1: cb_in1,
            in2: cb_in2,
            nvars: cb_nvars_list,
            missval: SYSMISS,
            is_missing: cb_is_missing,
            var_is_string: cb_is_string,
            var_is_strl: cb_is_strl,
            var_is_binary: cb_is_binary,
            sdatalen: cb_sdatalen,
            strldata: cb_strldata,
        }
    }

    /// State of the running call, reached by the callbacks.
    struct Ctx {
        me: *mut AdoInterpreter,
        vars: Vec<usize>,
        ifmask: Vec<bool>,
        in1: usize,
        in2: usize,
        out: String,
        changed: bool,
    }

    thread_local! {
        static CTX: Cell<*mut Ctx> = const { Cell::new(std::ptr::null_mut()) };
    }

    /// Run `f` with the current call context (None outside a call).
    fn with<R>(default: R, f: impl FnOnce(&mut Ctx, &mut AdoInterpreter) -> R) -> R {
        let p = CTX.with(|c| c.get());
        if p.is_null() {
            return default;
        }
        // SAFETY: the pointer is set only for the duration of `run`, which
        // holds the only borrow of the interpreter and the context.
        unsafe {
            let ctx = &mut *p;
            let me = &mut *ctx.me;
            f(ctx, me)
        }
    }

    fn cstr(p: *const c_char) -> Option<String> {
        if p.is_null() {
            return None;
        }
        // SAFETY: the plugin passes NUL-terminated strings.
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// Copy `s` into a C buffer of `size` bytes (or unbounded when size < 0).
    fn copy_out(s: &str, buf: *mut c_char, size: i32) {
        if buf.is_null() {
            return;
        }
        let bytes = s.as_bytes();
        let n = if size < 0 { bytes.len() } else { bytes.len().min((size as usize).saturating_sub(1)) };
        if size == 0 {
            return;
        }
        // SAFETY: the plugin provides a buffer of at least `size` bytes
        // (or one large enough for the value when no size is given).
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
            *buf.add(n) = 0;
        }
    }

    /// Dataset column of varlist variable `i` (1-based) and row of obs `j`.
    fn cell(ctx: &Ctx, me: &AdoInterpreter, i: i32, j: i32) -> Option<(usize, usize)> {
        let v = *ctx.vars.get((i as usize).checked_sub(1)?)?;
        let row = (j as usize).checked_sub(1)?;
        (row < me.engine.dataset.n_obs && i >= 1).then_some((v, row))
    }

    extern "C" fn cb_display(s: *const c_char) -> i32 {
        let t = cstr(s).unwrap_or_default();
        with(0, |c, _| {
            c.out.push_str(&t);
            0
        })
    }

    extern "C" fn cb_display_raw(s: *const c_char) -> i32 {
        // escape braces so the text shows as typed
        let mut t = String::new();
        for ch in cstr(s).unwrap_or_default().chars() {
            match ch {
                '{' => t.push_str("{c -(}"),
                '}' => t.push_str("{c )-}"),
                c => t.push(c),
            }
        }
        with(0, |c, _| {
            c.out.push_str(&t);
            0
        })
    }

    extern "C" fn cb_error(s: *const c_char) -> i32 {
        let t = cstr(s).unwrap_or_default();
        with(0, |c, _| {
            c.out.push_str("{err}");
            c.out.push_str(&t);
            c.out.push_str("{txt}");
            0
        })
    }

    extern "C" fn cb_zero() -> i32 {
        0
    }

    extern "C" fn cb_vdata(i: i32, j: i32, z: *mut f64) -> i32 {
        with(198, |c, me| {
            let Some((v, row)) = cell(c, me, i, j) else { return 198 };
            let var = &me.engine.dataset.vars[v];
            if var.var_type.is_string() || z.is_null() {
                return 7;
            }
            // SAFETY: z points to a double owned by the plugin.
            unsafe { *z = var.data[row].num() };
            0
        })
    }

    extern "C" fn cb_vstore(i: i32, j: i32, z: f64) -> i32 {
        with(198, |c, me| {
            let Some((v, row)) = cell(c, me, i, j) else { return 198 };
            let var = &mut me.engine.dataset.vars[v];
            if var.var_type.is_string() {
                return 7;
            }
            let val = ferrostat_cmd::data::store_num(var.var_type, z);
            var.data[row] = val;
            c.changed = true;
            0
        })
    }

    fn str_cell(me: &AdoInterpreter, v: usize, row: usize) -> String {
        let ds = &me.engine.dataset;
        match &ds.vars[v].data[row] {
            Value::Str(s) => s.clone(),
            Value::StrL(a, b) => ds.resolve_strl(*a, *b).unwrap_or("").to_string(),
            _ => String::new(),
        }
    }

    extern "C" fn cb_sdata(i: i32, j: i32, buf: *mut c_char) -> i32 {
        with(198, |c, me| {
            let Some((v, row)) = cell(c, me, i, j) else { return 198 };
            if !me.engine.dataset.vars[v].var_type.is_string() {
                return 7;
            }
            copy_out(&str_cell(me, v, row), buf, -1);
            0
        })
    }

    extern "C" fn cb_strldata(i: i32, j: i32, buf: *mut c_char, size: i32) -> i32 {
        with(-1, |c, me| {
            let Some((v, row)) = cell(c, me, i, j) else { return -1 };
            let s = str_cell(me, v, row);
            copy_out(&s, buf, size.max(0));
            s.len().min(size.max(1) as usize - 1) as i32
        })
    }

    extern "C" fn cb_sdatalen(i: i32, j: i32) -> i32 {
        with(-1, |c, me| match cell(c, me, i, j) {
            Some((v, row)) => str_cell(me, v, row).len() as i32,
            None => -1,
        })
    }

    extern "C" fn cb_sstore(i: i32, j: i32, s: *const c_char) -> i32 {
        let t = cstr(s).unwrap_or_default();
        with(198, |c, me| {
            let Some((v, row)) = cell(c, me, i, j) else { return 198 };
            let var = &mut me.engine.dataset.vars[v];
            if !var.var_type.is_string() {
                return 7;
            }
            var.data[row] = Value::Str(t);
            c.changed = true;
            0
        })
    }

    extern "C" fn cb_is_string(i: i32) -> i32 {
        with(0, |c, me| match c.vars.get((i as usize).wrapping_sub(1)) {
            Some(&v) => me.engine.dataset.vars[v].var_type.is_string() as i32,
            None => 0,
        })
    }

    extern "C" fn cb_is_strl(i: i32) -> i32 {
        with(0, |c, me| match c.vars.get((i as usize).wrapping_sub(1)) {
            Some(&v) => (me.engine.dataset.vars[v].var_type.type_str() == "strL") as i32,
            None => 0,
        })
    }

    extern "C" fn cb_is_binary(_i: i32, _j: i32) -> i32 {
        0
    }

    extern "C" fn cb_nobs() -> i32 {
        with(0, |_, me| me.engine.dataset.n_obs as i32)
    }

    extern "C" fn cb_nvar_total() -> i32 {
        with(0, |_, me| me.engine.dataset.vars.len() as i32)
    }

    extern "C" fn cb_nvars_list() -> i32 {
        with(0, |c, _| c.vars.len() as i32)
    }

    extern "C" fn cb_ifobs(j: i32) -> i32 {
        with(0, |c, _| (j >= 1 && c.ifmask.get(j as usize - 1).copied().unwrap_or(false)) as i32)
    }

    extern "C" fn cb_in1() -> i32 {
        with(1, |c, _| c.in1 as i32)
    }

    extern "C" fn cb_in2() -> i32 {
        with(0, |c, _| c.in2 as i32)
    }

    extern "C" fn cb_is_missing(z: f64) -> i32 {
        is_missing(z) as i32
    }

    extern "C" fn cb_mat_el(name: *const c_char, i: i32, j: i32, z: *mut f64) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(111, |_, me| {
            let Some(m) = me.engine.store.matrices.get(&n) else { return 111 };
            if i < 1 || j < 1 || i as usize > m.rows || j as usize > m.cols || z.is_null() {
                return 503;
            }
            // SAFETY: z points to a double owned by the plugin.
            unsafe { *z = m.get(i as usize - 1, j as usize - 1) };
            0
        })
    }

    extern "C" fn cb_mat_store(name: *const c_char, i: i32, j: i32, z: f64) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(111, |_, me| {
            let Some(m) = me.engine.store.matrices.get_mut(&n) else { return 111 };
            if i < 1 || j < 1 || i as usize > m.rows || j as usize > m.cols {
                return 503;
            }
            let cols = m.cols;
            m.data[(i as usize - 1) * cols + j as usize - 1] = z;
            0
        })
    }

    extern "C" fn cb_col(name: *const c_char) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(0, |_, me| me.engine.store.matrices.get(&n).map(|m| m.cols as i32).unwrap_or(0))
    }

    extern "C" fn cb_row(name: *const c_char) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(0, |_, me| me.engine.store.matrices.get(&n).map(|m| m.rows as i32).unwrap_or(0))
    }

    /// Macro names starting with `_` are locals of the calling scope.
    extern "C" fn cb_macro_save(name: *const c_char, val: *const c_char) -> i32 {
        let (n, v) = (cstr(name).unwrap_or_default(), cstr(val).unwrap_or_default());
        if n.is_empty() {
            return 198;
        }
        with(198, |_, me| {
            match n.strip_prefix('_') {
                Some(local) => me.engine.store.set_local(local, v),
                None => {
                    me.engine.store.globals.insert(n, v);
                }
            }
            0
        })
    }

    extern "C" fn cb_macro_use(name: *const c_char, buf: *mut c_char, size: i32) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(198, |_, me| {
            let v = match n.strip_prefix('_') {
                Some(local) => me.engine.store.local(local).cloned(),
                None => me.engine.store.globals.get(&n).cloned(),
            };
            copy_out(&v.unwrap_or_default(), buf, size.max(0));
            0
        })
    }

    extern "C" fn cb_scal_save(name: *const c_char, z: f64) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(198, |_, me| {
            me.engine.store.scalars.insert(n, z);
            0
        })
    }

    extern "C" fn cb_scal_use(name: *const c_char, z: *mut f64) -> i32 {
        let n = cstr(name).unwrap_or_default();
        with(111, |_, me| match me.engine.store.scalars.get(&n) {
            Some(v) if !z.is_null() => {
                // SAFETY: z points to a double owned by the plugin.
                unsafe { *z = *v };
                0
            }
            _ => 111,
        })
    }

    /// Call the plugin; returns (rc, SMCL output).
    pub fn run(me: &mut AdoInterpreter, entry: EntryFn, vars: Vec<usize>, ifmask: Vec<bool>, in1: usize, in2: usize, args: &[String]) -> (i32, String) {
        let mut ctx = Ctx { me: me as *mut AdoInterpreter, vars, ifmask, in1, in2, out: String::new(), changed: false };
        let cargs: Vec<CString> = args.iter().map(|a| CString::new(a.replace('\0', "")).unwrap_or_default()).collect();
        let mut argv: Vec<*mut c_char> = cargs.iter().map(|c| c.as_ptr() as *mut c_char).collect();
        argv.push(std::ptr::null_mut());
        let prev = CTX.with(|c| c.replace(&mut ctx as *mut Ctx));
        // SAFETY: calling into the plugin with its documented signature;
        // the argument strings outlive the call.
        let rc = unsafe { entry(cargs.len() as i32, argv.as_mut_ptr()) };
        CTX.with(|c| c.set(prev));
        if ctx.changed {
            me.engine.store.data_changed = true;
        }
        (rc, ctx.out)
    }
}
