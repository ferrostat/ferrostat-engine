//! Viewer, editor and dialog commands, and the requests they send to the
//! front end: `view`, `doedit`, `db` and the `window` family.
//!
//! The interpreter itself never opens windows or launches programs. The
//! front end declares what it is ([`Frontend`]); commands then either
//! return their result as text (batch runs, the browser build, and the
//! terminal for documents) or queue a [`UiRequest`] that the front end
//! carries out (the desktop interface opens a viewer, the do-file editor
//! or a dialog; the terminal launches the editor or the web browser).

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::w11::smcl::{self, Line};
use ferrostat_core::vfs;
use ferrostat_parser::parser::split_top_level_words;

use crate::interpreter::AdoInterpreter;
use crate::macros::unquote;
use crate::progtools::{has_opt, split_opts};

/// What kind of front end runs the interpreter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Frontend {
    /// Batch runs, tests and the browser: results are text only.
    #[default]
    Text,
    /// The interactive terminal: documents are printed, the editor and the
    /// web browser are launched by the front end.
    Terminal,
    /// The desktop interface: viewers, the editor and dialogs are windows.
    Gui,
}

/// Kind of dialog generated for a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogKind {
    /// `cmd depvar [indepvars]` (estimation commands)
    Estimation,
    /// `cmd [varlist]`
    Varlist,
    /// free arguments
    Generic,
}

/// A generic dialog: the command and the shape of its syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialogSpec {
    pub command: String,
    pub kind: DialogKind,
    /// accepts `if` / `in`
    pub ifin: bool,
}

impl DialogSpec {
    /// Build the command line from the dialog fields.
    pub fn build(&self, depvar: &str, vars: &[String], args: &str, if_exp: &str, in_range: &str, options: &str) -> String {
        let mut s = self.command.clone();
        for part in [depvar.trim(), &vars.join(" "), args.trim()] {
            if !part.is_empty() {
                s.push(' ');
                s.push_str(part);
            }
        }
        if !if_exp.trim().is_empty() {
            s.push_str(" if ");
            s.push_str(if_exp.trim());
        }
        if !in_range.trim().is_empty() {
            s.push_str(" in ");
            s.push_str(in_range.trim());
        }
        if !options.trim().is_empty() {
            s.push_str(", ");
            s.push_str(options.trim());
        }
        s
    }
}

/// One entry of the user menu built by `window menu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuEntry {
    Submenu { parent: String, label: String },
    Item { parent: String, label: String, command: String },
    Separator { parent: String },
}

impl MenuEntry {
    pub fn parent(&self) -> &str {
        match self {
            MenuEntry::Submenu { parent, .. } | MenuEntry::Item { parent, .. } | MenuEntry::Separator { parent } => parent,
        }
    }
}

/// Requests for the front end, taken with [`AdoInterpreter::take_ui_requests`].
#[derive(Clone, Debug, PartialEq)]
pub enum UiRequest {
    /// Show a document in a viewer window.
    View { title: String, lines: Vec<Line> },
    /// Open a URL in the web browser.
    Browse(String),
    /// Open the do-file editor (with this file, if any).
    Edit(Option<String>),
    /// Open a dialog box.
    Dialog(DialogSpec),
    /// `window manage ...` words.
    Manage(Vec<String>),
    /// Add a line to the command history (`window push`).
    PushHistory(String),
    /// The user menu changed.
    MenuChanged,
}

/// A file-selection request from `window fopen` / `window fsave`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDialogRequest {
    pub save: bool,
    pub title: String,
    /// (description, extensions without dots; "*" for any)
    pub filters: Vec<(String, Vec<String>)>,
    pub default_ext: Option<String>,
}

/// Front-end callback answering file-selection requests (None = cancelled).
pub type FileDialogFn = Box<dyn FnMut(&FileDialogRequest) -> Option<String> + Send>;

/// Interpreter state for these commands.
#[derive(Default)]
pub struct W11State {
    pub(crate) frontend: Frontend,
    pub(crate) requests: Vec<UiRequest>,
    pub(crate) menu: Vec<MenuEntry>,
    pub(crate) file_dialog: Option<FileDialogFn>,
    /// Loaded compiled plugins.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) plugins: crate::w11_plugin::Plugins,
}

impl AdoInterpreter {
    /// Declare the front end (default: [`Frontend::Text`]).
    pub fn set_frontend(&mut self, f: Frontend) {
        self.w11.frontend = f;
    }

    pub fn frontend(&self) -> Frontend {
        self.w11.frontend
    }

    /// Take the queued front-end requests.
    pub fn take_ui_requests(&mut self) -> Vec<UiRequest> {
        std::mem::take(&mut self.w11.requests)
    }

    /// Install the file-selection callback used by `window fopen/fsave`.
    pub fn set_file_dialog(&mut self, f: Option<FileDialogFn>) {
        self.w11.file_dialog = f;
    }

    /// The user menu built with `window menu append ...` (after `refresh`).
    pub fn user_menu(&self) -> &[MenuEntry] {
        &self.w11.menu
    }

    /// The dialog spec of a command (None when the command does not exist).
    pub fn dialog_spec(&self, cmd: &str) -> Option<DialogSpec> {
        dialog_for(self, cmd)
    }

    fn queue(&mut self, r: UiRequest) {
        if self.w11.frontend != Frontend::Text {
            self.w11.requests.push(r);
        }
    }
}

/// Commands provided here.
pub const COMMANDS: &[&str] = &["view", "doedit", "db", "plugin", "javacall"];

/// Run another command and capture what it displays.
fn capture(me: &mut AdoInterpreter, line: &str) -> Result<String, String> {
    let saved = std::mem::take(&mut me.out);
    let r = me.exec_text(line, None);
    let produced = std::mem::replace(&mut me.out, saved);
    r.map(|_| produced.join("\n"))
}

/// Show a document: queued for a viewer window in the desktop interface,
/// returned as text elsewhere.
fn show(me: &mut AdoInterpreter, title: &str, lines: Vec<Line>) -> String {
    if me.w11.frontend == Frontend::Gui {
        me.queue(UiRequest::View { title: title.to_string(), lines });
        String::new()
    } else {
        smcl::lines_to_text(&lines)
    }
}

/// Locate a file for `view` (as given, with usual extensions, or on the
/// ado-path).
fn locate(me: &AdoInterpreter, file: &str, adopath: bool) -> Option<String> {
    let mut cands = vec![file.to_string()];
    if std::path::Path::new(file).extension().is_none() {
        for ext in ["smcl", "sthlp", "txt", "log", "do"] {
            cands.push(format!("{file}.{ext}"));
        }
    }
    if let Some(c) = cands.iter().find(|c| vfs::exists(c)) {
        return Some(c.clone());
    }
    if adopath || !file.contains('/') {
        for c in &cands {
            if let Some(f) = crate::progtools::find_on_path(me, c, None, true, false).into_iter().next() {
                return Some(f);
            }
        }
    }
    None
}

fn read_doc(me: &AdoInterpreter, file: &str, adopath: bool) -> Result<(String, String), String> {
    let path = locate(me, file, adopath).ok_or_else(|| rc_err(601, format!("file {file} not found")))?;
    let bytes = vfs::read(&path).map_err(|_| rc_err(601, format!("file {path} could not be read")))?;
    Ok((path, String::from_utf8_lossy(&bytes).into_owned()))
}

/// `view [file] "filename" [, asis adopath]`, `view browse url`,
/// `view help|search|net|ado|update|news [args]`, `view smcl file`,
/// `view do file`, and the `*_d` dialog forms.
pub fn view(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (body, opts) = split_opts(r);
    let body = body.trim();
    let (sub, rest) = match body.split_once(char::is_whitespace) {
        Some((a, b)) => (a.to_string(), b.trim().to_string()),
        None => (body.to_string(), String::new()),
    };
    let asis = has_opt(&opts, "asis", 4);
    let adopath = has_opt(&opts, "adopath", 3);
    match sub.to_lowercase().as_str() {
        "" => Err(rc_err(198, "view: filename or subcommand required")),
        "browse" => {
            let url = unquote(&rest);
            if url.is_empty() {
                return Err(rc_err(198, "view browse: URL required"));
            }
            match me.w11.frontend {
                Frontend::Text => Ok(format!("(browse {url}: no web browser in this interface; open the address yourself)")),
                _ => {
                    me.queue(UiRequest::Browse(url.clone()));
                    Ok(String::new())
                }
            }
        }
        "help" | "search" | "net" | "ado" | "update" | "news" => {
            let verb = sub.to_lowercase();
            let line = if verb == "news" { "news".to_string() } else { format!("{verb} {rest}") };
            let text = if verb == "news" {
                "News is not available offline; see the Ferrostat release notes.".to_string()
            } else {
                capture(me, line.trim())?
            };
            let title = format!("Viewer - {}", line.trim());
            Ok(show(me, &title, smcl::plain_lines(&text)))
        }
        "help_d" | "search_d" | "net_d" | "ado_d" | "update_d" | "view_d" => {
            let what = sub.trim_end_matches("_d").to_lowercase();
            let line = if what == "view" { "view".into() } else { format!("view {what}") };
            if me.w11.frontend == Frontend::Gui {
                me.queue(UiRequest::Dialog(DialogSpec { command: line, kind: DialogKind::Generic, ifin: false }));
                Ok(String::new())
            } else {
                Ok(format!("(type {line} followed by the arguments; dialogs are available in the desktop interface)"))
            }
        }
        "smcl" | "do" | "file" => {
            let file = unquote(&rest);
            if file.is_empty() {
                return Err(rc_err(198, format!("view {sub}: filename required")));
            }
            let (path, text) = read_doc(me, &file, adopath)?;
            let lines = match sub.as_str() {
                "smcl" => smcl::render(&text, smcl::WIDTH),
                "do" => smcl::plain_lines(&text),
                _ if !asis && smcl::is_smcl(&path, &text) => smcl::render(&text, smcl::WIDTH),
                _ => smcl::plain_lines(&text),
            };
            Ok(show(me, &format!("Viewer - {path}"), lines))
        }
        _ => {
            let file = unquote(body);
            let (path, text) = read_doc(me, &file, adopath)?;
            let lines = if !asis && smcl::is_smcl(&path, &text) { smcl::render(&text, smcl::WIDTH) } else { smcl::plain_lines(&text) };
            Ok(show(me, &format!("Viewer - {path}"), lines))
        }
    }
}

/// `doedit [filename]`.
pub fn doedit(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let file = unquote(r.trim());
    let path = if file.is_empty() {
        None
    } else if std::path::Path::new(&file).extension().is_none() {
        Some(format!("{file}.do"))
    } else {
        Some(file)
    };
    match me.w11.frontend {
        Frontend::Text => Ok(match (&path, cfg!(target_arch = "wasm32")) {
            (Some(p), true) => format!("(doedit: open {p} in the editor pane of this page)"),
            (None, true) => "(doedit: use the editor pane of this page)".into(),
            (Some(p), false) => format!("(doedit: no editor in batch mode; file {p})"),
            (None, false) => "(doedit: no editor in batch mode)".into(),
        }),
        _ => {
            me.queue(UiRequest::Edit(path));
            Ok(String::new())
        }
    }
}

const ESTIMATION: &[&str] = &[
    "regress", "logit", "probit", "logistic", "poisson", "nbreg", "ologit", "oprobit", "mlogit", "mprobit", "tobit", "cloglog", "glm", "qreg", "rreg", "areg", "xtreg", "xtlogit", "xtprobit", "xtpoisson",
    "xtnbreg", "mixed", "melogit", "mepoisson", "heckman", "truncreg", "intreg", "betareg", "fracreg", "zip", "zinb", "gnbreg", "binreg", "hetprobit", "scobit", "biprobit", "clogit", "cnsreg", "boxcox",
    "vwls", "eivreg", "prais", "newey", "arima", "arch", "xtgls", "xtpcse", "xtabond", "ivregress", "ivprobit", "ivtobit", "reghdfe", "ppmlhdfe", "sqreg", "iqreg", "bsqreg", "nl",
];

const VARLIST: &[&str] = &[
    "summarize", "describe", "list", "codebook", "inspect", "correlate", "pwcorr", "tabstat", "tabulate", "tab1", "tab2", "histogram", "scatter", "line", "graph", "misstable", "duplicates", "order",
    "keep", "drop", "sort", "gsort", "browse", "edit", "ci", "means", "ttest", "sktest", "swilk", "sfrancia", "alpha", "pca", "factor", "cluster", "egen", "mdesc", "fre", "kdensity", "qnorm", "pnorm",
    "dotplot", "lowess", "twoway", "matrix", "tsset", "xtset", "xtsum", "xttab", "stsum", "stdescribe", "sts", "bysort", "collapse", "contract", "fillin", "encode", "decode", "destring", "tostring",
];

fn dialog_for(me: &AdoInterpreter, cmd: &str) -> Option<DialogSpec> {
    let w = cmd.trim().to_lowercase();
    if w.is_empty() {
        return None;
    }
    let full = me.engine.resolve_verb(&w);
    let known = me.engine.is_command(&w) || crate::progcmd::INTERP_COMMANDS.contains(&full.as_str()) || COMMANDS.contains(&full.as_str()) || me.engine.store.programs.contains_key(&w);
    if !known {
        return None;
    }
    let kind = if ESTIMATION.contains(&full.as_str()) {
        DialogKind::Estimation
    } else if VARLIST.contains(&full.as_str()) {
        DialogKind::Varlist
    } else {
        DialogKind::Generic
    };
    Some(DialogSpec { command: full, kind, ifin: kind != DialogKind::Generic })
}

/// `db commandname`.
pub fn db(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let name = r.split(',').next().unwrap_or("").trim().to_string();
    if name.is_empty() {
        return Err(rc_err(198, "db: command name required"));
    }
    let spec = dialog_for(me, &name).ok_or_else(|| rc_err(601, format!("dialog {name} not found")))?;
    if me.w11.frontend == Frontend::Gui {
        me.queue(UiRequest::Dialog(spec));
        return Ok(String::new());
    }
    Ok(format!(
        "(db {}: dialog boxes are available in the desktop interface, ferrostat-gui;\n type help {} for the syntax of the command)",
        spec.command, spec.command
    ))
}

/// Parse a dialog filter "Desc|*.dta;*.csv|All|*.*".
fn parse_filter(f: &str) -> Vec<(String, Vec<String>)> {
    let parts: Vec<&str> = f.split('|').collect();
    let mut out = Vec::new();
    for pair in parts.chunks(2) {
        let (desc, pat) = match pair {
            [d, p] => (d.trim().to_string(), *p),
            [p] => (p.trim().to_string(), *p),
            _ => continue,
        };
        let exts: Vec<String> = pat.split([';', ',', ' ']).filter(|x| !x.trim().is_empty()).map(|x| x.trim().trim_start_matches('*').trim_start_matches('.').to_string()).map(|x| if x.is_empty() { "*".to_string() } else { x }).collect();
        if !exts.is_empty() {
            out.push((desc, exts));
        }
    }
    out
}

/// The remaining `window` subcommands: `fopen`, `fsave`, `menu`, `manage`,
/// `push`, `dialog`, `control`, `list`. Returns None for subcommands
/// handled elsewhere (`stopbox`).
pub fn window(me: &mut AdoInterpreter, r: &str) -> Option<Result<String, String>> {
    let w = split_top_level_words(r);
    let sub = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let args: Vec<String> = w.iter().skip(1).map(|s| unquote(s)).collect();
    Some(match sub.as_str() {
        "fopen" | "fsave" => (|| {
            let mac = args.first().ok_or_else(|| rc_err(198, format!("window {sub} macname \"title\" \"filter\" [extension]")))?.clone();
            let req = FileDialogRequest {
                save: sub == "fsave",
                title: args.get(1).cloned().unwrap_or_default(),
                filters: parse_filter(args.get(2).map(String::as_str).unwrap_or("")),
                default_ext: args.get(3).cloned().filter(|e| !e.is_empty()),
            };
            let chosen = match me.w11.file_dialog.as_mut() {
                Some(f) => f(&req),
                None => None,
            };
            let mut path = chosen.unwrap_or_default();
            if let (false, Some(ext)) = (path.is_empty(), &req.default_ext) {
                if std::path::Path::new(&path).extension().is_none() {
                    path = format!("{path}.{ext}");
                }
            }
            // the file name is returned in the global macro
            me.engine.store.globals.insert(mac.clone(), path.clone());
            me.engine.store.set_local(&mac, path);
            Ok(String::new())
        })(),
        "menu" => menu(me, &args),
        "manage" | "man" => {
            me.queue(UiRequest::Manage(args.iter().map(|s| s.to_string()).collect()));
            Ok(String::new())
        }
        "push" => {
            let line = r.trim_start()[w[0].len()..].trim().to_string();
            me.queue(UiRequest::PushHistory(line));
            Ok(String::new())
        }
        "dialog" | "control" | "list" => Ok(String::new()),
        _ => return None,
    })
}

/// `window menu clear | append submenu "parent" "label" | append item
/// "parent" "label" "command" | append separator "parent" | refresh |
/// add_recentfiles "parent"`.
fn menu(me: &mut AdoInterpreter, a: &[String]) -> Result<String, String> {
    let sub = a.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let arg = |i: usize| a.get(i).cloned().unwrap_or_default();
    match sub.as_str() {
        "clear" => {
            me.w11.menu.clear();
            me.queue(UiRequest::MenuChanged);
        }
        "refresh" => me.queue(UiRequest::MenuChanged),
        "add_recentfiles" => {}
        "append" => {
            let kind = arg(1).to_lowercase();
            let entry = match kind.as_str() {
                "submenu" => MenuEntry::Submenu { parent: arg(2), label: arg(3) },
                "item" => {
                    if a.len() < 5 {
                        return Err(rc_err(198, "window menu append item \"parent\" \"label\" \"command\""));
                    }
                    MenuEntry::Item { parent: arg(2), label: arg(3), command: arg(4) }
                }
                "separator" => MenuEntry::Separator { parent: arg(2) },
                _ => return Err(rc_err(198, "window menu append submenu|item|separator ...")),
            };
            if entry.parent().is_empty() {
                return Err(rc_err(198, "window menu: parent menu required"));
            }
            me.w11.menu.push(entry);
        }
        "" => return Err(rc_err(198, "window menu subcommand required")),
        other => return Err(rc_err(198, format!("window menu {other}: invalid subcommand"))),
    }
    Ok(String::new())
}

/// Is `parent` the root of the user menu (e.g. "stUser")?
pub fn is_menu_root(parent: &str) -> bool {
    parent.to_lowercase().ends_with("user")
}

// ---------------------------------------------------------------------------
// terminal front end
// ---------------------------------------------------------------------------

/// Set up an interpreter for the interactive terminal: documents print to
/// the terminal; `window fopen/fsave` ask for a file name on standard input.
#[cfg(not(target_arch = "wasm32"))]
pub fn terminal_attach(me: &mut AdoInterpreter) {
    use std::io::{BufRead, Write};
    me.set_frontend(Frontend::Terminal);
    me.set_file_dialog(Some(Box::new(|req: &FileDialogRequest| {
        let what = if req.save { "save as" } else { "open" };
        let title = if req.title.is_empty() { String::new() } else { format!("{} - ", req.title) };
        print!("{title}file to {what} (empty to cancel): ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).ok()?;
        let t = line.trim();
        (!t.is_empty()).then(|| t.to_string())
    })));
}

/// Carry out queued requests for the terminal: launch the editor
/// (`$VISUAL`, `$EDITOR`) and the web browser. Returns messages to print.
#[cfg(not(target_arch = "wasm32"))]
pub fn terminal_service(me: &mut AdoInterpreter) -> Vec<String> {
    let mut msgs = Vec::new();
    for r in me.take_ui_requests() {
        match r {
            UiRequest::Edit(path) => msgs.push(launch_editor(path.as_deref())),
            UiRequest::Browse(url) => msgs.push(open_browser(&url)),
            UiRequest::View { lines, .. } => msgs.push(smcl::lines_to_text(&lines)),
            UiRequest::Dialog(d) => msgs.push(format!("(dialog boxes are available in the desktop interface; type help {})", d.command)),
            _ => {}
        }
    }
    msgs
}

#[cfg(not(target_arch = "wasm32"))]
fn launch_editor(path: Option<&str>) -> String {
    let editor = ["VISUAL", "EDITOR"].iter().filter_map(|v| std::env::var(v).ok()).find(|e| !e.trim().is_empty());
    let Some(editor) = editor else {
        return match path {
            Some(p) => format!("(no editor configured: set VISUAL or EDITOR; file {p})"),
            None => "(no editor configured: set VISUAL or EDITOR)".into(),
        };
    };
    let mut parts = editor.split_whitespace();
    let prog = parts.next().unwrap_or_default().to_string();
    let mut cmd = std::process::Command::new(&prog);
    cmd.args(parts);
    if let Some(p) = path {
        cmd.arg(p);
    }
    match cmd.status() {
        Ok(_) => String::new(),
        Err(e) => format!("(could not start editor {prog}: {e}{})", path.map(|p| format!("; file {p}")).unwrap_or_default()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn open_browser(url: &str) -> String {
    if std::env::var_os("FERROSTAT_NO_BROWSER").is_some() {
        return format!("(browse {url})");
    }
    let mut cmd = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    cmd.arg(url).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    match cmd.spawn() {
        Ok(_) => format!("(opening {url} in the web browser)"),
        Err(_) => format!("(no web browser available; open {url} yourself)"),
    }
}
