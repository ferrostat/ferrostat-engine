//! The command-language interpreter: statements, blocks and control flow,
//! prefixes (quietly, noisily, capture, by), macros, programs, do-files.

use std::path::PathBuf;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_cmd::store::{Program, Results};
use ferrostat_cmd::util;
use ferrostat_cmd::varlist;
use ferrostat_engine::engine::Engine;
use ferrostat_parser::parser::{parse_expression, parse_line, Command};

use crate::macros::{self, macro_number, unquote};
use crate::script::{self, Stmt};

/// Control-flow outcome of executing statements.
#[derive(Debug, Clone, PartialEq)]
pub enum Flow {
    Next,
    Continue,
    Break,
    Exit,
}

/// Return code carried by an error message (`...\nr(111);`).
pub fn rc_of(err: &str) -> i32 {
    err.lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix("r(").and_then(|x| x.strip_suffix(");")).and_then(|x| x.parse().ok()))
        .unwrap_or(198)
}

/// Message text without the trailing `r(#);` line.
pub fn err_message(err: &str) -> String {
    let lines: Vec<&str> = err.lines().filter(|l| !(l.trim().starts_with("r(") && l.trim().ends_with(");"))).collect();
    lines.join("\n")
}

pub(crate) struct LogFile {
    pub path: PathBuf,
    pub smcl: bool,
    pub on: bool,
    pub name: String,
}

pub struct AdoInterpreter {
    pub(crate) engine: Engine,
    buffer: String,
    pub(crate) quiet: usize,
    pub(crate) noisy: usize,
    pub(crate) out: Vec<String>,
    pub(crate) logs: Vec<LogFile>,
    pub(crate) temp_counter: usize,
    pub(crate) tempvars: Vec<Vec<String>>,
    pub(crate) tempfiles: Vec<PathBuf>,
    /// r() values being built by `return` inside rclass programs.
    pub(crate) pending_r: Vec<Option<Results>>,
    pub(crate) depth: usize,
    pub(crate) echo: bool,
    pub(crate) putexcel: crate::putexcel::PutExcel,
    /// Documents being built by putdocx / putpdf.
    pub(crate) docs: crate::putdocx::DocState,
    /// Outcome of the last top-level `if`, so a later `else` line can see it.
    top_last_if: Option<bool>,
    /// Output produced before the error of the last failed `run_script`.
    pub partial_output: String,
    pub(crate) outreg_tables: std::collections::HashMap<String, crate::outreg::Table>,
    /// The external Python helper process and its settings.
    pub(crate) python: crate::python::PyState,
    pub(crate) frmt_tables: crate::frmttable::Tables,
    /// Matrix-language globals and functions.
    pub(crate) mata: crate::mata::MataState,
}

impl Default for AdoInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl AdoInterpreter {
    pub fn new() -> Self {
        Self {
            engine: Engine::new(),
            buffer: String::new(),
            quiet: 0,
            noisy: 0,
            out: Vec::new(),
            logs: Vec::new(),
            temp_counter: 0,
            tempvars: vec![Vec::new()],
            tempfiles: Vec::new(),
            pending_r: Vec::new(),
            depth: 0,
            echo: false,
            putexcel: Default::default(),
            docs: Default::default(),
            top_last_if: None,
            partial_output: String::new(),
            outreg_tables: Default::default(),
            python: Default::default(),
            frmt_tables: Default::default(),
            mata: Default::default(),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Echo commands in the output (batch mode).
    pub fn set_echo(&mut self, on: bool) {
        self.echo = on;
    }

    pub(crate) fn emit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.quiet > 0 && self.noisy == 0 {
            return;
        }
        self.out.push(text.to_string());
        self.log_write(text, false);
    }

    pub(crate) fn log_write(&mut self, text: &str, is_cmd: bool) {
        for lg in &self.logs {
            if !lg.on {
                continue;
            }
            {
                let t = if lg.smcl {
                    if is_cmd { format!("{{com}}. {}\n", smcl_escape(text)) } else { format!("{{res}}{}\n", smcl_escape(text)) }
                } else if is_cmd {
                    format!(". {text}\n")
                } else {
                    format!("{text}\n")
                };
                let _ = ferrostat_core::vfs::append(&lg.path.to_string_lossy(), t.as_bytes());
            }
        }
    }

    /// Feed one physical line (REPL / line-by-line do-file reading).
    /// Returns the output produced once a complete statement is available.
    pub fn run_line(&mut self, line: &str) -> Result<String, String> {
        self.buffer.push_str(line);
        self.buffer.push('\n');
        if !script::is_complete(&self.buffer) {
            return Ok(String::new());
        }
        let text = std::mem::take(&mut self.buffer);
        self.run_script(&text)
    }

    /// Is a multi-line construct waiting for more input?
    pub fn is_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Run a block of do-file text in the current scope.
    pub fn run_script(&mut self, text: &str) -> Result<String, String> {
        self.out.clear();
        let lines = script::logical_lines(text);
        let stmts = script::parse_block(&lines)?;
        let mut last_if = if self.depth == 0 { self.top_last_if.take() } else { None };
        let res = self.exec_block_with(&stmts, &mut last_if);
        if self.depth == 0 {
            self.top_last_if = last_if;
        }
        let out = std::mem::take(&mut self.out).join("\n");
        match res {
            Ok(Flow::Exit) if self.depth == 0 => Ok(if out.is_empty() { "__EXIT__".into() } else { format!("{out}\n__EXIT__") }),
            Ok(_) => Ok(out),
            Err(e) => {
                self.log_write(&err_message(&e), false);
                self.partial_output = out;
                Err(e)
            }
        }
    }

    pub(crate) fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        let mut last_if: Option<bool> = None;
        self.exec_block_with(stmts, &mut last_if)
    }

    fn exec_block_with(&mut self, stmts: &[Stmt], last_if: &mut Option<bool>) -> Result<Flow, String> {
        for s in stmts {
            let flow = self.exec_stmt(s, last_if)?;
            if flow != Flow::Next {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    fn expand(&mut self, text: &str) -> Result<String, String> {
        let eng = &mut self.engine;
        macros::expand(text, &eng.dataset, &mut eng.store)
    }

    fn eval_cond(&mut self, text: &str) -> Result<bool, String> {
        let e = parse_expression(text).map_err(|e| rc_err(198, format!("{}: {}", text.trim(), e.message)))?;
        let v = Env::with_store(&self.engine.dataset, &self.engine.store).eval(&e, 0)?;
        Ok(match v {
            Val::Num(n) => n != 0.0,
            Val::Str(_) => return Err(rc_err(109, "type mismatch")),
        })
    }

    fn exec_stmt(&mut self, s: &Stmt, last_if: &mut Option<bool>) -> Result<Flow, String> {
        if let Some(raw) = &s.raw {
            *last_if = None;
            return self.exec_raw_block(&s.text, raw);
        }
        let first = s.text.split_whitespace().next().unwrap_or("").to_lowercase();
        // `else` must see the preceding if
        if first == "else" {
            let taken = last_if.ok_or_else(|| rc_err(198, "else without if"))?;
            let rest = s.text.trim()[4..].trim().to_string();
            if taken {
                // skip, but keep chain state for `else if`
                return Ok(Flow::Next);
            }
            if rest.to_lowercase().starts_with("if ") || rest.to_lowercase() == "if" {
                let sub = Stmt { text: rest, body: s.body.clone(), raw: None };
                return self.exec_stmt(&sub, last_if);
            }
            *last_if = None;
            return match &s.body {
                Some(b) => self.exec_block(b),
                None => self.exec_text(&rest, None),
            };
        }
        if first == "if" {
            let header = self.expand(s.text.trim()[2..].trim())?;
            return match &s.body {
                Some(b) => {
                    let c = self.eval_cond(&header)?;
                    *last_if = Some(c);
                    if c { self.exec_block(b) } else { Ok(Flow::Next) }
                }
                None => {
                    let (cond, cmd) = split_if_command(&header).ok_or_else(|| rc_err(198, "if: command expected"))?;
                    let c = self.eval_cond(&cond)?;
                    *last_if = Some(c);
                    if c { self.exec_text(&cmd, None) } else { Ok(Flow::Next) }
                }
            };
        }
        *last_if = None;
        if first == "while" {
            let raw = s.text.trim()[5..].trim().to_string();
            let body = s.body.clone().ok_or_else(|| rc_err(198, "while requires {"))?;
            let mut guard = 0u64;
            loop {
                let cond = self.expand(&raw)?;
                if !self.eval_cond(&cond)? {
                    break;
                }
                guard += 1;
                if guard > 100_000_000 {
                    return Err(rc_err(198, "while loop limit reached"));
                }
                match self.exec_block(&body)? {
                    Flow::Break => break,
                    Flow::Exit => return Ok(Flow::Exit),
                    _ => {}
                }
            }
            return Ok(Flow::Next);
        }
        let text = self.expand(&s.text)?;
        if self.echo && self.depth == 0 {
            self.out.push(format!(". {}", s.text));
        }
        if self.depth == 0 {
            self.log_write(&s.text, true);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let flow = self.exec_text(&text, s.body.as_deref())?;
        if self.engine.store.rmsg && self.depth == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            let t = started.elapsed().as_secs_f64();
            #[cfg(target_arch = "wasm32")]
            let t = 0.0;
            self.emit(&format!("r; t={:.2}", t));
        }
        Ok(flow)
    }

    /// Execute already-expanded statement text (with optional block body).
    pub(crate) fn exec_text(&mut self, text: &str, body: Option<&[Stmt]>) -> Result<Flow, String> {
        let text = text.trim();
        if text.is_empty() {
            return match body {
                Some(b) => self.exec_block(b),
                None => Ok(Flow::Next),
            };
        }
        let (verb_raw, rest) = split_verb(text);
        let verb = self.engine.resolve_verb(&verb_raw.trim_end_matches(':').to_lowercase());
        let colon_prefix = verb_raw.ends_with(':');
        match verb.as_str() {
            "quietly" | "noisily" | "capture" => {
                let rest = rest.trim().trim_start_matches(':').trim().to_string();
                let _ = colon_prefix;
                return self.prefixed(&verb, &rest, body);
            }
            "version" => {
                // the version number selects legacy behaviours (e.g. coefficient names of `mean, over()`)
                let num = rest.split(':').next().unwrap_or("").split_whitespace().next().and_then(|v| v.parse::<f64>().ok());
                if let Some(i) = rest.find(':') {
                    let prev = self.engine.store.settings.get("version").cloned();
                    if let Some(n) = num {
                        self.engine.store.settings.insert("version".into(), n.to_string());
                    }
                    let res = self.exec_text(&rest[i + 1..], body);
                    match prev {
                        Some(p) => self.engine.store.settings.insert("version".into(), p),
                        None => self.engine.store.settings.remove("version"),
                    };
                    return res;
                }
                if let Some(n) = num {
                    self.engine.store.settings.insert("version".into(), n.to_string());
                }
                return Ok(Flow::Next);
            }
            "frame" | "frames" if frame_prefix(&rest, body.is_some()).is_some() => {
                let (name, command) = frame_prefix(&rest, body.is_some()).unwrap();
                return self.in_frame(&name, &command, body);
            }
            "xi" | "timer" if rest.contains(':') => {
                let i = rest.find(':').unwrap();
                let command = if verb == "xi" { xi_terms(&rest[i + 1..]) } else { rest[i + 1..].to_string() };
                return self.exec_text(&command, body);
            }
            "by" | "bysort" => return self.by_prefix(&verb, &rest),
            "foreach" => return self.foreach(&rest, body.ok_or_else(|| rc_err(198, "foreach requires {"))?),
            "forvalues" => return self.forvalues(&rest, body.ok_or_else(|| rc_err(198, "forvalues requires {"))?),
            "continue" => return Ok(if rest.contains("break") { Flow::Break } else { Flow::Continue }),
            "break" => return Ok(Flow::Break),
            "exit" | "quit" => {
                let code: i32 = rest.split(',').next().unwrap_or("").trim().parse().unwrap_or(0);
                if code != 0 {
                    return Err(rc_err(code, ""));
                }
                return Ok(Flow::Exit);
            }
            _ => {}
        }
        // bootstrap / simulate / permute prefixes, vce(bootstrap), estat bootstrap
        if body.is_none() {
            if let Some(f) = crate::resample::intercept(self, &verb_raw, &rest, text)? {
                return Ok(f);
            }
        }
        if let Some(b) = body {
            // a bare block such as `{ ... }` after other text
            let f = self.exec_text(text, None)?;
            if f != Flow::Next {
                return Ok(f);
            }
            return self.exec_block(b);
        }
        let out = self.dispatch(&verb, &verb_raw, &rest, text)?;
        if out == "__EXIT__" {
            return Ok(Flow::Exit);
        }
        self.emit(&out);
        Ok(Flow::Next)
    }

    fn prefixed(&mut self, verb: &str, rest: &str, body: Option<&[Stmt]>) -> Result<Flow, String> {
        let run = |me: &mut Self| -> Result<Flow, String> {
            if rest.is_empty() {
                match body {
                    Some(b) => me.exec_block(b),
                    None => Ok(Flow::Next),
                }
            } else {
                me.exec_text(rest, body)
            }
        };
        match verb {
            "quietly" => {
                self.quiet += 1;
                let saved = self.noisy;
                self.noisy = 0;
                let r = run(self);
                self.noisy = saved;
                self.quiet -= 1;
                r
            }
            "noisily" => {
                self.noisy += 1;
                let r = run(self);
                self.noisy -= 1;
                r
            }
            _ => {
                // capture: suppress output and errors, set _rc
                self.quiet += 1;
                let saved = self.noisy;
                let first = split_verb(rest).0;
                let noisily = self.engine.resolve_verb(first.trim_end_matches(':')) == "noisily";
                if !noisily {
                    self.noisy = 0;
                }
                let r = run(self);
                self.noisy = saved;
                self.quiet -= 1;
                match r {
                    Ok(f) => {
                        self.engine.store.rc = 0;
                        Ok(if f == Flow::Exit { Flow::Next } else { f })
                    }
                    Err(e) => {
                        let code = rc_of(&e);
                        if noisily {
                            self.out.push(err_message(&e));
                        }
                        self.engine.store.rc = code;
                        Ok(Flow::Next)
                    }
                }
            }
        }
    }

    /// `frame name: command` / `frame name { ... }`: run in another frame,
    /// then return to the current one (also after an error).
    fn in_frame(&mut self, name: &str, command: &str, body: Option<&[Stmt]>) -> Result<Flow, String> {
        let prev = self.engine.store.cur_frame.clone();
        let e = &mut self.engine;
        ferrostat_cmd::frames::switch(&mut e.dataset, &mut e.store, name)?;
        let r = if command.is_empty() {
            match body {
                Some(b) => self.exec_block(b),
                None => Ok(Flow::Next),
            }
        } else {
            self.exec_text(command, body)
        };
        let e = &mut self.engine;
        if ferrostat_cmd::frames::exists(&e.store, &prev) {
            ferrostat_cmd::frames::switch(&mut e.dataset, &mut e.store, &prev)?;
        }
        r
    }

    fn by_prefix(&mut self, verb: &str, rest: &str) -> Result<Flow, String> {
        let colon = ferrostat_parser::parser::find_top_level(rest, ':').ok_or_else(|| rc_err(198, "by: colon expected"))?;
        let spec = rest[..colon].trim();
        let command = rest[colon + 1..].trim().to_string();
        let (spec, opts) = match spec.find(',') {
            Some(i) => (&spec[..i], spec[i + 1..].to_string()),
            None => (spec, String::new()),
        };
        let mut sort = verb == "bysort" || opts.contains("sort");
        let (byvars_text, sortvars_text) = match spec.find('(') {
            Some(i) => {
                sort = true;
                (&spec[..i], spec[i + 1..].trim_end_matches(')').to_string())
            }
            None => (spec, String::new()),
        };
        let byvars = varlist::expand(&self.engine.dataset, byvars_text)?;
        let sortvars = if sortvars_text.trim().is_empty() { Vec::new() } else { varlist::expand(&self.engine.dataset, &sortvars_text)? };
        let groups = self.engine.by_groups(&byvars, &sortvars, sort)?;
        let (cverb, _) = split_verb(&command);
        let cverb = self.engine.resolve_verb(&cverb);
        let res = if self.engine.is_groupwise(&cverb) {
            self.engine.store.by_groups = Some(groups);
            let r = self.exec_text(&command, None);
            self.engine.store.by_groups = None;
            r
        } else {
            let n = self.engine.dataset.n_obs;
            let mut starts: Vec<usize> = Vec::new();
            for i in 0..n {
                if groups.start[i] == i {
                    starts.push(i);
                }
            }
            let mut result = Ok(Flow::Next);
            for s in starts {
                let e = groups.end[s];
                let header: Vec<String> = byvars
                    .iter()
                    .map(|v| {
                        let var = self.engine.dataset.get_var(v).unwrap();
                        format!("{} = {}", v, util::cell_text(&self.engine.dataset, var, s, true))
                    })
                    .collect();
                self.emit(&format!("\n{}\n-> {}\n", "-".repeat(78), header.join(", ")));
                let mask: Vec<bool> = (0..n).map(|i| i >= s && i < e).collect();
                self.engine.store.by_mask = Some(mask);
                self.engine.store.by_groups = Some(groups.clone());
                let r = self.exec_text(&command, None);
                self.engine.store.by_mask = None;
                self.engine.store.by_groups = None;
                if let Err(err) = r {
                    if !opts.contains("rc0") {
                        result = Err(err);
                        break;
                    }
                }
            }
            result
        };
        res
    }

    fn foreach(&mut self, rest: &str, body: &[Stmt]) -> Result<Flow, String> {
        let w: Vec<&str> = rest.split_whitespace().collect();
        if w.len() < 2 {
            return Err(rc_err(198, "foreach: invalid syntax"));
        }
        let name = w[0].to_string();
        let items: Vec<String> = match w[1] {
            "in" => macros::words(rest.trim_start()[name.len()..].trim_start()[2..].trim()).into_iter().map(|x| unquote(&x)).collect(),
            "of" => {
                let kind = w.get(2).copied().unwrap_or("");
                let list_text = w[3..].join(" ");
                match kind {
                    "local" | "loc" => macros::words(self.engine.store.local(list_text.trim()).map(String::as_str).unwrap_or("")).into_iter().map(|x| unquote(&x)).collect(),
                    "global" | "glo" => macros::words(self.engine.store.globals.get(list_text.trim()).map(String::as_str).unwrap_or("")).into_iter().map(|x| unquote(&x)).collect(),
                    "varlist" | "var" => varlist::expand(&self.engine.dataset, &list_text)?,
                    "newlist" | "new" => varlist::new_names(&self.engine.dataset, &list_text)?,
                    "numlist" | "num" => macros::numlist(&list_text)?.into_iter().map(macro_number).collect(),
                    _ => return Err(rc_err(198, format!("foreach: {kind} invalid"))),
                }
            }
            _ => return Err(rc_err(198, "foreach: in or of expected")),
        };
        for it in items {
            self.engine.store.set_local(&name, it);
            match self.exec_block(body)? {
                Flow::Break => break,
                Flow::Exit => return Ok(Flow::Exit),
                _ => {}
            }
        }
        Ok(Flow::Next)
    }

    fn forvalues(&mut self, rest: &str, body: &[Stmt]) -> Result<Flow, String> {
        let eq = rest.find('=').ok_or_else(|| rc_err(198, "forvalues: = expected"))?;
        let name = rest[..eq].trim().to_string();
        let range = rest[eq + 1..].trim();
        let vals = macros::numlist(&range.replace(" ", ""))?;
        for v in vals {
            self.engine.store.set_local(&name, macro_number(v));
            match self.exec_block(body)? {
                Flow::Break => break,
                Flow::Exit => return Ok(Flow::Exit),
                _ => {}
            }
        }
        Ok(Flow::Next)
    }

    fn exec_raw_block(&mut self, header: &str, raw: &[String]) -> Result<Flow, String> {
        let h = self.expand(header)?;
        let first = h.split_whitespace().next().unwrap_or("").to_lowercase();
        if script::is_program_header(&h) {
            let rest = h.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
            let rest = rest.strip_prefix("define").or_else(|| rest.strip_prefix("def")).map(str::trim).unwrap_or(rest);
            let (name, opts) = match rest.find(',') {
                Some(i) => (rest[..i].trim(), rest[i + 1..].to_lowercase()),
                None => (rest.trim(), String::new()),
            };
            if self.engine.store.programs.contains_key(name) {
                return Err(rc_err(110, format!("program {name} already defined")));
            }
            let class = ["rclass", "eclass", "sclass"].iter().find(|c| opts.contains(*c)).map(|c| c.to_string()).unwrap_or_else(|| "nclass".into());
            self.engine.store.programs.insert(
                name.to_string(),
                Program { name: name.to_string(), body: raw.to_vec(), class, byable: opts.contains("byable") },
            );
            return Ok(Flow::Next);
        }
        if first.starts_with("inp") {
            let out = crate::progcmd::input(self, &h, raw)?;
            self.emit(&out);
            return Ok(Flow::Next);
        }
        if first == "python" || first == "python:" {
            crate::python::block(self, first == "python:", raw)?;
            return Ok(Flow::Next);
        }
        if first == "mata" || first == "mata:" {
            crate::mata::run_block(self, &h, raw)?;
            return Ok(Flow::Next);
        }
        Err(rc_err(3000, format!("{first} blocks are not supported")))
    }

    /// Run a user-defined program.
    pub(crate) fn run_program(&mut self, name: &str, args: &str) -> Result<String, String> {
        let prog = self.engine.store.programs.get(name).cloned().ok_or_else(|| rc_err(199, format!("command {name} is unrecognized")))?;
        if self.depth > 200 {
            return Err(rc_err(1000, "system limit exceeded; program nesting too deep"));
        }
        self.engine.store.push_frame();
        self.engine.store.set_local("0", args.to_string());
        self.engine.store.set_local("*", args.trim().to_string());
        for (k, w) in macros::words(args).iter().enumerate() {
            self.engine.store.set_local(&(k + 1).to_string(), unquote(w));
        }
        self.tempvars.push(Vec::new());
        let rclass = prog.class == "rclass";
        self.pending_r.push(if rclass { Some(Results::new()) } else { None });
        self.depth += 1;
        let saved_out = std::mem::take(&mut self.out);
        let lines = script::logical_lines(&prog.body.join("\n"));
        let res = script::parse_block(&lines).and_then(|stmts| self.exec_block(&stmts));
        let produced = std::mem::replace(&mut self.out, saved_out);
        self.depth -= 1;
        let pending = self.pending_r.pop().flatten();
        let tv = self.tempvars.pop().unwrap_or_default();
        for v in tv {
            let _ = self.engine.dataset.drop_variable(&v);
        }
        self.engine.store.pop_frame();
        if let Some(p) = pending {
            self.engine.store.r = p;
        }
        let text = produced.join("\n");
        match res {
            Ok(_) => Ok(text),
            Err(e) => Err(if text.is_empty() { e } else { format!("{text}\n{e}") }),
        }
    }

    /// Run a do-file (own local scope, arguments in `0`, `1`, ...).
    pub fn run_do(&mut self, path: &str, args: &str, quiet: bool, same_scope: bool) -> Result<String, String> {
        let p = if ferrostat_core::vfs::exists(path) || path.ends_with(".do") || path.ends_with(".ado") { path.to_string() } else { format!("{path}.do") };
        let text = ferrostat_core::vfs::read_to_string(&p).map_err(|_| rc_err(601, format!("file {p} not found")))?;
        if !same_scope {
            self.engine.store.push_frame();
            self.engine.store.set_local("0", args.to_string());
            self.engine.store.set_local("*", args.trim().to_string());
            for (k, w) in macros::words(args).iter().enumerate() {
                self.engine.store.set_local(&(k + 1).to_string(), unquote(w));
            }
        }
        if quiet {
            self.quiet += 1;
        }
        self.depth += 1;
        let saved = std::mem::take(&mut self.out);
        let lines = script::logical_lines(&text);
        let res = script::parse_block(&lines).and_then(|s| self.exec_block(&s));
        let produced = std::mem::replace(&mut self.out, saved);
        self.depth -= 1;
        if quiet {
            self.quiet -= 1;
        }
        if !same_scope {
            self.engine.store.pop_frame();
        }
        let text = produced.join("\n");
        match res {
            Ok(_) => Ok(text),
            Err(e) => Err(if text.is_empty() { e } else { format!("{text}\n{e}") }),
        }
    }

    fn dispatch(&mut self, verb: &str, verb_raw: &str, rest: &str, full: &str) -> Result<String, String> {
        if let Some(out) = crate::progcmd::interp_command(self, verb, rest, full)? {
            return Ok(out);
        }
        if self.engine.store.programs.contains_key(verb_raw) {
            return self.run_program(verb_raw, rest.trim());
        }
        if verb == "egen" {
            if let Some((prog, args)) = user_egen(rest, |p| self.engine.store.programs.contains_key(p)) {
                return self.run_program(&prog, &args);
            }
        }
        if self.engine.is_command(verb) || matches!(verb, "clear" | "preserve" | "restore" | "set" | "sysuse") {
            if verb == "clear" && matches!(rest.trim(), "mata" | "all" | "*") {
                crate::mata::clear(self);
            }
            let cmd: Command = parse_line(full).map_err(|e| rc_err(198, e.message))?;
            return self.engine.execute(&cmd);
        }
        // a command with a trailing native ado: allow programs with abbreviations disabled
        Err(rc_err(199, format!("command {verb_raw} is unrecognized")))
    }
}

/// `xi:` interaction terms in factor-variable notation: `i.a*i.b` is
/// `i.a##i.b`, `i.a*b` is `i.a##c.b` and `i.a|b` is `i.a i.a#c.b`.
fn xi_terms(command: &str) -> String {
    command
        .split(' ')
        .map(|w| {
            let lower = w.to_lowercase();
            if !lower.starts_with("i.") {
                return w.to_string();
            }
            let cont = |v: &str| if v.to_lowercase().starts_with("i.") { v.to_string() } else { format!("c.{v}") };
            if let Some((a, b)) = w.split_once('*') {
                format!("{a}##{}", cont(b))
            } else if let Some((a, b)) = w.split_once('|') {
                format!("{a} {a}#c.{b}")
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `egen [type] newvar = fcn(args) [if] [in] [, options]` where a program
/// `_gfcn` is defined: that program and its arguments,
/// `type newvar = (args) [if] [in] [, options]`.
fn user_egen(rest: &str, defined: impl Fn(&str) -> bool) -> Option<(String, String)> {
    let (lhs, rhs) = rest.split_once('=')?;
    let rhs = rhs.trim_start();
    let open = rhs.find('(')?;
    let fcn = rhs[..open].trim();
    let prog = format!("_g{fcn}");
    if fcn.is_empty() || !defined(&prog) {
        return None;
    }
    let mut depth = 0;
    let close = rhs[open..].char_indices().find_map(|(i, c)| {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
        None
    })?;
    let lhs: Vec<&str> = lhs.split_whitespace().collect();
    let (ty, name) = match lhs.as_slice() {
        [name] => ("float", *name),
        [ty, name] => (*ty, *name),
        _ => return None,
    };
    Some((prog, format!("{ty} {name} = ({}){}", &rhs[open + 1..close], &rhs[close + 1..])))
}

/// First word (the verb) and the rest of a statement.
/// `name: command` (or `name` before a block) after `frame`: the frame name
/// and the command to run in it.
fn frame_prefix(rest: &str, block: bool) -> Option<(String, String)> {
    let rest = rest.trim();
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(rest.len());
    let (name, after) = rest.split_at(end);
    if name.is_empty() {
        return None;
    }
    let after = after.trim_start();
    if let Some(cmd) = after.strip_prefix(':') {
        return Some((name.to_string(), cmd.trim().to_string()));
    }
    (block && after.is_empty()).then(|| (name.to_string(), String::new()))
}

pub fn split_verb(text: &str) -> (String, String) {
    let t = text.trim();
    let end = t.char_indices().find(|&(_, c)| c.is_whitespace() || c == ',' || c == '{' || (c == ':' && false)).map(|(i, _)| i).unwrap_or(t.len());
    let mut verb = t[..end].to_string();
    let mut rest = t[end..].to_string();
    // `quietly:` / `capture:` forms
    if let Some(v) = verb.strip_suffix(':') {
        verb = v.to_string();
        rest = format!(": {rest}");
    }
    // `bysort g:` written without space
    if let Some(i) = verb.find(':') {
        rest = format!("{}{}", &verb[i..], rest);
        verb = verb[..i].to_string();
    }
    (verb, rest)
}

/// Split `cond command` for a one-line `if`: the longest prefix of words
/// that forms a valid expression is the condition.
pub fn split_if_command(text: &str) -> Option<(String, String)> {
    let words = ferrostat_parser::parser::split_top_level_words(text);
    for k in (1..words.len()).rev() {
        let cond = words[..k].join(" ");
        if parse_expression(&cond).is_ok() {
            return Some((cond, words[k..].join(" ")));
        }
    }
    None
}

fn smcl_escape(s: &str) -> String {
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbs() {
        assert_eq!(split_verb("bysort g: gen x = 1"), ("bysort".into(), " g: gen x = 1".into()));
        assert_eq!(split_verb("quietly: sum x").0, "quietly");
        assert_eq!(split_if_command("x > 1 di \"a\"").unwrap(), ("x > 1".into(), "di \"a\"".into()));
        assert_eq!(rc_of("variable x not found\nr(111);"), 111);
    }
}
