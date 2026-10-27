//! Front-end neutral session: runs commands, keeps the results transcript
//! and exposes the state the graphical front ends display (variables, data,
//! the current graph).

use ferrostat_cmd::graph::Graph;
use ferrostat_cmd::util::cell_text;

use crate::interpreter::{err_message, rc_of, AdoInterpreter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Command,
    Output,
    Error,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub kind: EntryKind,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct VarInfo {
    pub name: String,
    pub vtype: String,
    pub format: String,
    pub value_label: String,
    pub label: String,
}

pub struct Session {
    pub interp: AdoInterpreter,
    pub transcript: Vec<Entry>,
    /// commands entered, oldest first
    pub history: Vec<String>,
    /// bumped every time the current graph changes
    pub graph_serial: u64,
    last_graph: Option<Graph>,
    pub exited: bool,
    /// bumped every time `cls` clears the transcript
    pub clears: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Session { interp: AdoInterpreter::new(), transcript: Vec::new(), history: Vec::new(), graph_serial: 0, last_graph: None, exited: false, clears: 0 }
    }

    fn push(&mut self, kind: EntryKind, text: &str) {
        let text = text.trim_end();
        if !text.is_empty() {
            self.transcript.push(Entry { kind, text: text.to_string() });
        }
    }

    /// Run one command line (as typed in the command window). Returns the rc.
    pub fn submit(&mut self, line: &str) -> i32 {
        let prompt = if self.interp.is_pending() { "> " } else { ". " };
        self.push(EntryKind::Command, &format!("{prompt}{line}"));
        if !line.trim().is_empty() {
            self.history.push(line.to_string());
        }
        let res = self.interp.run_line(line);
        self.finish(res)
    }

    /// Run a whole do-file text with command echo (the do-file editor).
    pub fn run_script(&mut self, text: &str) -> i32 {
        self.interp.set_echo(true);
        let res = self.interp.run_script(text);
        self.interp.set_echo(false);
        self.finish(res)
    }

    fn finish(&mut self, res: Result<String, String>) -> i32 {
        let rc = match res {
            Ok(out) => {
                if out.ends_with("__EXIT__") {
                    self.exited = true;
                }
                self.push_output(out.trim_end_matches("__EXIT__"));
                0
            }
            Err(e) => {
                let partial = std::mem::take(&mut self.interp.partial_output);
                self.push_output(&partial);
                self.push(EntryKind::Error, &format!("{}\nr({});", err_message(&e).trim_end(), rc_of(&e)));
                rc_of(&e)
            }
        };
        let g = &self.interp.engine().store.graph;
        if *g != self.last_graph {
            self.last_graph = g.clone();
            self.graph_serial += 1;
        }
        rc
    }

    /// Output text may contain echoed commands (". cmd" lines) from do-files,
    /// and `cls` markers, which clear the transcript.
    fn push_output(&mut self, out: &str) {
        let mut buf = String::new();
        for line in out.lines() {
            if line.contains(crate::interpreter::CLS) {
                buf.clear();
                self.transcript.clear();
                self.clears += 1;
                let rest = line.rsplit(crate::interpreter::CLS).next().unwrap_or("");
                if !rest.is_empty() {
                    buf.push_str(rest);
                    buf.push('\n');
                }
            } else if line.starts_with(". ") || line == "." {
                self.push(EntryKind::Output, &std::mem::take(&mut buf));
                self.push(EntryKind::Command, line);
            } else {
                buf.push_str(line);
                buf.push('\n');
            }
        }
        self.push(EntryKind::Output, &buf);
    }

    pub fn variables(&self) -> Vec<VarInfo> {
        self.interp
            .engine()
            .dataset
            .vars
            .iter()
            .map(|v| VarInfo { name: v.name.clone(), vtype: v.var_type.type_str(), format: v.format.clone(), value_label: v.value_label_name.clone(), label: v.label.clone() })
            .collect()
    }

    pub fn n_obs(&self) -> usize {
        self.interp.engine().dataset.n_obs
    }

    pub fn n_vars(&self) -> usize {
        self.interp.engine().dataset.vars.len()
    }

    /// Displayed text of one cell (value labels applied).
    pub fn cell(&self, row: usize, col: usize) -> String {
        let ds = &self.interp.engine().dataset;
        match ds.vars.get(col) {
            Some(v) if row < v.data.len() => cell_text(ds, v, row, true),
            _ => String::new(),
        }
    }

    pub fn is_string_var(&self, col: usize) -> bool {
        self.interp.engine().dataset.vars.get(col).is_some_and(|v| v.var_type.is_string())
    }

    pub fn graph(&self) -> Option<&Graph> {
        self.interp.engine().store.graph.as_ref()
    }

    /// Should a graph window show the current graph? (not after `graph close`
    /// or under `set graphics off`)
    pub fn graph_visible(&self) -> bool {
        let st = &self.interp.engine().store;
        st.graph.is_some() && !st.graphics_off && !st.graph_window_closed
    }

    pub fn graph_svg(&self, w: u32, h: u32) -> Option<String> {
        self.graph().and_then(|g| ferrostat_cmd::graph::render_svg(g, w, h).ok())
    }

    /// Words the language accepts as commands (with abbreviations and aliases).
    pub fn command_words(&self) -> Vec<String> {
        let mut w = self.interp.engine().command_words();
        w.extend(crate::progcmd::INTERP_COMMANDS.iter().map(|s| s.to_string()));
        w.extend(["else", "end", "set", "version", "clear", "pause", "continue", "break", "assert", "sleep", "help"].iter().map(|s| s.to_string()));
        w.sort();
        w.dedup();
        w
    }

    pub fn pending(&self) -> bool {
        self.interp.is_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_records_transcript_and_graph() {
        let mut s = Session::new();
        assert_eq!(s.submit("set obs 3"), 0);
        assert_eq!(s.submit("gen x = _n"), 0);
        assert_eq!(s.n_obs(), 3);
        assert_eq!(s.variables()[0].name, "x");
        assert_eq!(s.cell(2, 0), "3");
        assert_eq!(s.submit("nosuchcmd"), 199);
        assert_eq!(s.transcript.last().unwrap().kind, EntryKind::Error);
        assert_eq!(s.graph_serial, 0);
        s.submit("scatter x x");
        assert_eq!(s.graph_serial, 1);
        assert!(s.graph_svg(200, 150).unwrap().contains("<svg"));
        assert_eq!(s.run_script("di 1+1\ndi 2+2"), 0);
        assert!(s.transcript.iter().any(|e| e.kind == EntryKind::Command && e.text == ". di 2+2"));
        let words = s.command_words();
        for w in ["regress", "reg", "gen", "generate", "foreach", "local", "xtivreg", "su"] {
            assert!(words.iter().any(|x| x == w), "{w}");
        }
        assert!(!words.iter().any(|x| x == "x"));
    }
}
