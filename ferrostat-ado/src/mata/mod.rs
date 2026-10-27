//! The matrix programming language: `mata` blocks, `mata:` one-liners and
//! the `mata describe/drop/clear` commands. Global variables and functions
//! persist in the session until `clear mata` or `clear all`.

mod builtins;
mod eval;
mod host;
mod lexer;
mod ops;
mod parser;
mod value;

use ferrostat_cmd::eval::rc_err;

use crate::interpreter::AdoInterpreter;
pub use eval::MataState;
use eval::{err, Ev, MErr, R};
use parser::{Parser, Stmt};
use value::Val;

/// Error text of a failed statement (without the `r(#);` line).
fn err_text(e: &MErr) -> String {
    if e.msg == "__exit__" {
        return String::new();
    }
    let mut lines = Vec::new();
    match e.trace.split_first() {
        // syntax errors carry no location
        None if e.rc == 3000 => lines.push(e.msg.clone()),
        None => lines.push(format!("{:>16}:  {:>4}  {}", "<istmt>", e.rc, e.msg)),
        Some((first, rest)) => {
            lines.push(format!("{:>16}:  {:>4}  {}", format!("{first}()"), e.rc, e.msg));
            for f in rest {
                lines.push(format!("{:>16}:     -  function returned error", format!("{f}()")));
            }
            lines.push(format!("{:>16}:     -  function returned error", "<istmt>"));
        }
    }
    lines.join("\n")
}

fn to_host_err(e: &MErr) -> String {
    rc_err(e.rc, err_text(e))
}

pub fn clear(me: &mut AdoInterpreter) {
    me.mata = MataState::default();
}

/// Execute source text; `stop_on_error` aborts at the first error, otherwise
/// errors are reported and execution continues with the next statement.
fn run(me: &mut AdoInterpreter, lines: &[String], stop_on_error: bool, echo: bool) -> Result<(), String> {
    let src = lines.join("\n");
    let toks = lexer::lex(&src).map_err(|m| rc_err(3000, m))?;
    let mut st = std::mem::take(&mut me.mata);
    let mut failure = None;
    {
        let mut ev = Ev::new(me, &mut st);
        let mut p = Parser::new(&toks);
        loop {
            p.skip_nl();
            if p.done() {
                break;
            }
            let start = p.p;
            let first_line = toks.get(start).map(|t| t.line);
            let parsed = p.stmt();
            if parsed.is_err() {
                p.recover();
            }
            if echo {
                if let Some(l0) = first_line {
                    let l1 = toks[..p.p.max(start + 1).min(toks.len())].last().map(|t| t.line).unwrap_or(l0);
                    for (k, l) in (l0..=l1.max(l0)).enumerate() {
                        let text = lines.get(l).map(|s| s.as_str()).unwrap_or("");
                        ev.me.out.push(format!("{}{}", if k == 0 { ": " } else { "> " }, text));
                    }
                }
            }
            let res = match parsed {
                Ok(Stmt::Nop) => continue,
                Ok(s) => ev.exec(&s).map(|_| ()),
                Err(m) => Err(err(3000, m)),
            };
            ev.flush();
            if let Err(e) = res {
                if e.msg == "__exit__" && e.rc == 0 {
                    continue;
                }
                if stop_on_error {
                    failure = Some(to_host_err(&e));
                    break;
                }
                let t = err_text(&e);
                if !t.is_empty() {
                    ev.me.emit(&t);
                }
                ev.me.emit(&format!("r({});", e.rc));
            }
            if echo {
                ev.me.out.push(String::new());
            }
        }
        ev.flush();
    }
    me.mata = st;
    match failure {
        Some(f) => Err(f),
        None => Ok(()),
    }
}

/// A `mata` / `mata:` block (lines up to `end`).
pub fn run_block(me: &mut AdoInterpreter, header: &str, raw: &[String]) -> Result<(), String> {
    let colon = header.trim().ends_with(':');
    let echo = me.echo && me.depth == 0;
    let lines: Vec<String> = raw
        .iter()
        .map(|l| {
            let eng = &mut me.engine;
            crate::macros::expand(l, &eng.dataset, &mut eng.store).unwrap_or_else(|_| l.clone())
        })
        .collect();
    if echo {
        me.out.push(format!(". {}", header.trim()));
        me.out.push(format!("{} mata (type end to exit) {}", "-".repeat(49), "-".repeat(6)));
    }
    let res = run(me, &lines, colon, echo);
    if echo {
        me.out.push(": end".into());
        me.out.push("-".repeat(80));
    }
    res
}

const COMMANDS: &[&str] = &["describe", "d", "des", "drop", "clear", "set", "mlib", "mosave", "which", "rename", "query", "memory", "help", "matsave", "matuse", "matdescribe"];

/// `mata: statement`, `mata statement` and `mata describe` etc. typed as commands.
pub fn run_line(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let text = rest.trim().trim_start_matches(':').trim();
    if text.is_empty() {
        return Ok(String::new());
    }
    let first = text.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("");
    let line = if COMMANDS.contains(&first) { format!("mata {text}") } else { text.to_string() };
    run(me, &[line], true, false)?;
    Ok(String::new())
}

fn type_text(v: &Val) -> String {
    format!("{} {}", v.eltype(), v.orgtype())
}

fn bytes(v: &Val) -> usize {
    match v {
        Val::R(m) => m.d.len() * 8,
        Val::S(m) => m.d.iter().map(|s| s.len()).sum(),
        Val::P(m) => m.d.len() * 8,
        Val::O(_) => 8,
    }
}

fn matches(pat: &str, name: &str) -> bool {
    fn rec(p: &[char], n: &[char]) -> bool {
        match p.first() {
            None => n.is_empty(),
            Some('*') => (0..=n.len()).any(|k| rec(&p[1..], &n[k..])),
            Some('?') => !n.is_empty() && rec(&p[1..], &n[1..]),
            Some(c) => n.first() == Some(c) && rec(&p[1..], &n[1..]),
        }
    }
    rec(&pat.chars().collect::<Vec<_>>(), &name.chars().collect::<Vec<_>>())
}

/// `mata describe`, `mata drop`, `mata clear`, ...
pub(crate) fn command(ev: &mut Ev, c: &str) -> R<()> {
    let c = c.trim();
    let (word, rest) = match c.find(|ch: char| ch.is_whitespace() || ch == ',') {
        Some(i) => (&c[..i], c[i..].trim()),
        None => (c, ""),
    };
    match word {
        "describe" | "d" | "des" => {
            ev.flush();
            let args = rest.split(',').next().unwrap_or("").trim();
            let pats: Vec<&str> = args.split_whitespace().collect();
            let want = |n: &str| pats.is_empty() || pats.iter().any(|p| matches(p.trim_end_matches("()"), n));
            let mut out = vec!["      # bytes   type                        name and extent".to_string(), "-".repeat(79)];
            for (n, f) in &ev.st.funcs {
                if want(n) {
                    let ty = match &f.ret {
                        Some(t) if t.el == parser::El::Void => "void".to_string(),
                        Some(t) => t.text(),
                        None => "transmorphic".to_string(),
                    };
                    out.push(format!("{:>13}   {:<28}{}()", 0, format!("{ty} function"), n));
                }
            }
            for (n, s) in &ev.st.globals {
                if want(n) {
                    let v = s.borrow();
                    let (r, c) = v.dims();
                    let ext = if (r, c) == (1, 1) { n.clone() } else { format!("{n}[{r},{c}]") };
                    out.push(format!("{:>13}   {:<28}{}", bytes(&v), type_text(&v), ext));
                }
            }
            out.push("-".repeat(79));
            ev.me.emit(&out.join("\n"));
            Ok(())
        }
        "drop" => {
            for pat in rest.split_whitespace() {
                if let Some(fp) = pat.strip_suffix("()") {
                    let names: Vec<String> = ev.st.funcs.keys().filter(|n| matches(fp, n)).cloned().collect();
                    if names.is_empty() && !fp.contains('*') {
                        return Err(err(3499, format!("{pat} not found")));
                    }
                    for n in names {
                        ev.st.funcs.remove(&n);
                    }
                } else {
                    let names: Vec<String> = ev.st.globals.keys().filter(|n| matches(pat, n)).cloned().collect();
                    if names.is_empty() && !pat.contains('*') {
                        return Err(err(3499, format!("{pat} not found")));
                    }
                    for n in names {
                        ev.st.globals.remove(&n);
                    }
                }
            }
            Ok(())
        }
        "clear" => {
            *ev.st = MataState::default();
            Ok(())
        }
        "which" => {
            ev.flush();
            let n = rest.trim_end_matches("()");
            let text = if ev.st.funcs.contains_key(n) { format!("  {n}():  user-defined") } else { format!("  {n}():  built-in") };
            ev.me.emit(&text);
            Ok(())
        }
        "rename" => {
            let w: Vec<&str> = rest.split_whitespace().collect();
            if w.len() != 2 {
                return Err(err(198, "invalid syntax"));
            }
            let s = ev.st.globals.remove(w[0]).ok_or_else(|| err(3499, format!("{} not found", w[0])))?;
            ev.st.globals.insert(w[1].to_string(), s);
            Ok(())
        }
        "end" => Ok(()),
        _ => Ok(()),
    }
}
