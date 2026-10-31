//! Programmer's conventions for user-written estimation commands.
//!
//! `estat`: when the current estimation results name a handler in
//! `e(estat_cmd)` and that program is defined, `estat ...` calls it with
//! the full argument text; the handler implements its own subcommands and
//! hands the standard ones (`ic`, `summarize`, `vce`, ...) to
//! `estat_default`, which runs the built-in `estat` regardless of
//! `e(estat_cmd)`.

use crate::interpreter::AdoInterpreter;

/// The user handler named by e(estat_cmd), if it is a defined program.
pub fn estat_handler(me: &AdoInterpreter) -> Option<String> {
    let h = me.engine.store.e.macros.get("estat_cmd")?.trim().to_string();
    (!h.is_empty() && h != "estat" && me.engine.store.programs.contains_key(&h)).then_some(h)
}

/// `estat ...` through the user handler.
pub fn estat(me: &mut AdoInterpreter, handler: &str, rest: &str) -> Result<String, String> {
    me.run_program(handler, rest)
}

/// `estat_default ...`: the built-in `estat`.
pub fn estat_default(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let saved = me.engine.store.e.macros.remove("estat_cmd");
    let r = me.exec_text(&format!("estat {rest}"), None).map(|_| String::new());
    if let Some(h) = saved {
        // the handler may have replaced e(); restore the marker only on the same results
        me.engine.store.e.macros.entry("estat_cmd".into()).or_insert(h);
    }
    r
}

/// Local marking a program called with the `nopreserve` option; its value
/// is the depth of the preserved-data stack at entry.
pub const NOPRESERVE_LOCAL: &str = "__nopreserve";

/// At program entry (in the program's frame): the preserved-data depth,
/// and the `nopreserve` marker when the caller passed that option.
///
/// Data preserved by a program are restored when it ends. A caller that
/// has already preserved the data (and will restore it) passes
/// `nopreserve`: `preserve` inside the program is then skipped, and so is
/// `restore` of data the program did not preserve, which saves the copy.
pub fn program_enter(me: &mut AdoInterpreter, args: &str) -> usize {
    let depth = me.engine.store.preserved.len();
    let opts = ferrostat_parser::parser::find_top_level(args, ',').map_or("", |c| &args[c + 1..]);
    let has = ferrostat_parser::parser::split_options(opts).iter().any(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= 6 && "nopreserve".starts_with(&n)
    });
    if has {
        me.engine.store.set_local(NOPRESERVE_LOCAL, depth.to_string());
    }
    depth
}

/// At program exit: restore data the program preserved and left preserved.
pub fn program_exit(me: &mut AdoInterpreter, depth: usize) {
    let st = &mut me.engine.store;
    if st.preserved.len() > depth {
        let ds = st.preserved.swap_remove(depth);
        st.preserved.truncate(depth);
        me.engine.dataset = ds;
    }
}

/// `about`: version and build information.
pub fn about(me: &AdoInterpreter) -> String {
    let n = me.engine.command_words().len();
    [
        format!("Ferrostat {}", env!("CARGO_PKG_VERSION")),
        format!("Statistics engine for .do and .dta files ({} {})", std::env::consts::OS, std::env::consts::ARCH),
        format!("License: {}", env!("CARGO_PKG_LICENSE")),
        format!("{n} command names and abbreviations recognised"),
    ]
    .join("\n")
}

/// `search`/`findit word ...`: offline keyword search of the command names
/// (built-in commands, interpreter commands and programs in memory).
pub fn search(me: &AdoInterpreter, verb: &str, r: &str) -> String {
    let words: Vec<String> = r.split(',').next().unwrap_or("").split_whitespace().map(|w| crate::macros::unquote(w).to_lowercase()).collect();
    if verb == "help" {
        let w = words.first().cloned().unwrap_or_default();
        if w.is_empty() {
            return "help: see the Ferrostat documentation; `search word` lists matching commands".into();
        }
        let full = me.engine.resolve_verb(&w);
        let kind = if me.engine.is_command(&w) {
            "built-in command"
        } else if crate::progcmd::INTERP_COMMANDS.contains(&w.as_str()) {
            "built-in programming command"
        } else if me.engine.store.programs.contains_key(&w) {
            "program in memory"
        } else {
            return format!("help for {w} not found; try `search {w}`");
        };
        let shown = if full != w { format!("{w} ({full})") } else { w };
        return format!("{shown}: {kind}; see the Ferrostat documentation");
    }
    let mut names: Vec<String> = me.engine.command_words();
    names.extend(crate::progcmd::INTERP_COMMANDS.iter().map(|s| s.to_string()));
    names.extend(me.engine.store.programs.keys().cloned());
    // full names only
    names.sort();
    names.dedup();
    let full: Vec<&String> = names.iter().filter(|n| !names.iter().any(|o| o.len() > n.len() && o.starts_with(n.as_str()) && me.engine.resolve_verb(n) == **o)).collect();
    let hits: Vec<&&String> = full.iter().filter(|n| words.iter().all(|w| n.contains(w.as_str()))).collect();
    let mut out = vec![format!("Search of command names for \"{}\" (offline; no network search)", words.join(" "))];
    if hits.is_empty() {
        out.push("no matching commands found".into());
    } else {
        for chunk in hits.chunks(6) {
            out.push(chunk.iter().map(|h| format!("{:<14}", h)).collect::<String>().trim_end().to_string());
        }
    }
    out.join("\n")
}
