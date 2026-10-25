//! Ferrostat for the browser: the engine compiled to WebAssembly, with an
//! in-memory file system the page fills by upload and empties by download.
//! Structured results are returned as JSON strings.

use ferrostat_ado::session::{EntryKind, Session};
use ferrostat_core::vfs;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(msg: &str);
}

/// Report Rust panics on the browser console (a panic stops the module).
#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::panic::set_hook(Box::new(|info| error(&format!("Ferrostat internal error: {info}")))));
}

#[wasm_bindgen]
pub struct Ferrostat {
    session: Session,
    /// transcript entries already handed to the page
    sent: usize,
}

impl Default for Ferrostat {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Ferrostat {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Ferrostat {
        #[cfg(target_arch = "wasm32")]
        install_panic_hook();
        Ferrostat { session: Session::new(), sent: 0 }
    }

    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Run one line from the command box; returns the return code.
    pub fn submit(&mut self, line: &str) -> i32 {
        self.session.submit(line)
    }

    /// Run a do-file text (with command echo); returns the return code.
    pub fn run_script(&mut self, text: &str) -> i32 {
        self.session.run_script(text)
    }

    /// Is a multi-line construct (program, loop, ...) waiting for input?
    pub fn pending(&self) -> bool {
        self.session.pending()
    }

    /// New transcript entries since the last call:
    /// `[{"kind": "command"|"output"|"error", "text": ...}]`.
    pub fn take_output(&mut self) -> String {
        let t = &self.session.transcript;
        let new: Vec<Value> = t[self.sent.min(t.len())..]
            .iter()
            .map(|e| {
                let kind = match e.kind {
                    EntryKind::Command => "command",
                    EntryKind::Output => "output",
                    EntryKind::Error => "error",
                };
                json!({"kind": kind, "text": e.text})
            })
            .collect();
        self.sent = t.len();
        Value::Array(new).to_string()
    }

    pub fn history(&self) -> String {
        json!(self.session.history).to_string()
    }

    /// `[{"name", "type", "format", "vallab", "label"}]`
    pub fn variables(&self) -> String {
        let v: Vec<Value> = self
            .session
            .variables()
            .into_iter()
            .map(|v| json!({"name": v.name, "type": v.vtype, "format": v.format, "vallab": v.value_label, "label": v.label}))
            .collect();
        Value::Array(v).to_string()
    }

    pub fn n_obs(&self) -> usize {
        self.session.n_obs()
    }

    pub fn n_vars(&self) -> usize {
        self.session.n_vars()
    }

    /// Displayed cells of observations `start..start+count`:
    /// `{"strings": [bool per variable], "rows": [[cell, ...], ...]}`.
    pub fn data_page(&self, start: usize, count: usize) -> String {
        let s = &self.session;
        let nv = s.n_vars();
        let end = (start + count).min(s.n_obs());
        let rows: Vec<Vec<String>> = (start.min(end)..end).map(|i| (0..nv).map(|c| s.cell(i, c)).collect()).collect();
        let strings: Vec<bool> = (0..nv).map(|c| s.is_string_var(c)).collect();
        json!({"strings": strings, "rows": rows}).to_string()
    }

    /// Changes whenever a new graph is drawn.
    pub fn graph_serial(&self) -> f64 {
        self.session.graph_serial as f64
    }

    pub fn graph_svg(&self, width: u32, height: u32) -> Option<String> {
        self.session.graph_svg(width, height)
    }

    // ---- in-memory file system ----

    pub fn write_file(&self, path: &str, data: &[u8]) -> Result<(), JsError> {
        vfs::write(path, data).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        vfs::read(path).ok()
    }

    pub fn remove_file(&self, path: &str) -> bool {
        vfs::remove(path).is_ok()
    }

    /// `[{"path", "size"}]` of every file.
    pub fn files(&self) -> String {
        let v: Vec<Value> = all_files().into_iter().map(|(p, n)| json!({"path": p, "size": n})).collect();
        Value::Array(v).to_string()
    }

    pub fn cwd(&self) -> String {
        vfs::current_dir()
    }

    /// PNG/PDF exports the page must rasterise from SVG:
    /// `[{"path", "svg", "width", "height", "pdf"}]`. When done, the page
    /// writes the PNG itself, or calls `pdf_from_rgb` for a PDF.
    pub fn take_raster_jobs(&self) -> String {
        let v: Vec<Value> = raster_jobs().into_iter().map(|(path, svg, w, h, pdf)| json!({"path": path, "svg": svg, "width": w, "height": h, "pdf": pdf})).collect();
        Value::Array(v).to_string()
    }

    /// Single-page PDF embedding packed RGB pixels.
    pub fn pdf_from_rgb(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
        ferrostat_cmd::graph::pdf_with_image(rgb, width, height)
    }
}

#[cfg(target_arch = "wasm32")]
fn all_files() -> Vec<(String, usize)> {
    vfs::all_files()
}

#[cfg(not(target_arch = "wasm32"))]
fn all_files() -> Vec<(String, usize)> {
    Vec::new()
}

#[cfg(target_arch = "wasm32")]
fn raster_jobs() -> Vec<(String, String, u32, u32, bool)> {
    ferrostat_cmd::graph::take_raster_jobs().into_iter().map(|j| (j.path, j.svg, j.width, j.height, j.pdf)).collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn raster_jobs() -> Vec<(String, String, u32, u32, bool)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_api() {
        let mut f = Ferrostat::new();
        assert_eq!(f.submit("set obs 2"), 0);
        f.submit("gen str3 s = \"a\" + string(_n)");
        let out: Value = serde_json::from_str(&f.take_output()).unwrap();
        assert_eq!(out[0]["kind"], "command");
        assert_eq!(f.take_output(), "[]");
        let page: Value = serde_json::from_str(&f.data_page(0, 10)).unwrap();
        assert_eq!(page["rows"][1][0], "a2");
        assert_eq!(page["strings"][0], true);
        let vars: Value = serde_json::from_str(&f.variables()).unwrap();
        assert_eq!(vars[0]["type"], "str3");
    }
}
