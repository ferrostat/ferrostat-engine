//! Ferrostat desktop interface: results window, command line, history,
//! variables, data browser, do-file editor and graph viewer.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui::{self, Color32, FontId, Key, RichText, TextStyle};
use egui_extras::{Column, TableBuilder};
use ferrostat_ado::session::{EntryKind, Session};

const GRAPH_W: u32 = 800;
const GRAPH_H: u32 = 600;

struct App {
    session: Session,
    command: String,
    /// position while browsing the history with the arrow keys
    hist_pos: Option<usize>,
    selected_var: Option<usize>,
    show_data: bool,
    show_editor: bool,
    show_graph: bool,
    editor_text: String,
    editor_path: Option<std::path::PathBuf>,
    graph_tex: Option<(u64, egui::TextureHandle)>,
    scroll_to_end: bool,
}

impl App {
    fn new() -> Self {
        let mut session = Session::new();
        session.transcript.push(ferrostat_ado::session::Entry {
            kind: EntryKind::Output,
            text: format!("Ferrostat {} — statistics engine\nType commands below, or use the File menu.", env!("CARGO_PKG_VERSION")),
        });
        App {
            session,
            command: String::new(),
            hist_pos: None,
            selected_var: None,
            show_data: false,
            show_editor: false,
            show_graph: false,
            editor_text: String::new(),
            editor_path: None,
            graph_tex: None,
            scroll_to_end: true,
        }
    }

    fn run(&mut self, cmd: &str) {
        let serial = self.session.graph_serial;
        self.session.submit(cmd);
        self.after_run(serial);
    }

    fn run_script(&mut self, text: &str) {
        let serial = self.session.graph_serial;
        self.session.run_script(text);
        self.after_run(serial);
    }

    fn after_run(&mut self, serial: u64) {
        if self.session.graph_serial != serial && self.session.graph().is_some() {
            self.show_graph = true;
        }
        self.scroll_to_end = true;
    }

    fn quoted(p: &std::path::Path) -> String {
        format!("\"{}\"", p.display())
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open dataset…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Datasets", &["dta"]).pick_file() {
                        self.run(&format!("use {}, clear", Self::quoted(&p)));
                    }
                }
                if ui.button("Save dataset as…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Datasets", &["dta"]).save_file() {
                        self.run(&format!("save {}, replace", Self::quoted(&p)));
                    }
                }
                ui.separator();
                if ui.button("Import delimited text…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Text", &["csv", "txt", "tsv"]).pick_file() {
                        self.run(&format!("import delimited using {}, clear", Self::quoted(&p)));
                    }
                }
                if ui.button("Import Excel…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Excel", &["xlsx"]).pick_file() {
                        self.run(&format!("import excel using {}, firstrow clear", Self::quoted(&p)));
                    }
                }
                if ui.button("Export Excel…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Excel", &["xlsx"]).save_file() {
                        self.run(&format!("export excel using {}, firstrow(variables) replace", Self::quoted(&p)));
                    }
                }
                ui.separator();
                if ui.button("Run do-file…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Do-files", &["do"]).pick_file() {
                        self.run(&format!("do {}", Self::quoted(&p)));
                    }
                }
                if ui.button("Change working directory…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.run(&format!("cd {}", Self::quoted(&p)));
                    }
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Window", |ui| {
                ui.checkbox(&mut self.show_data, "Data browser");
                ui.checkbox(&mut self.show_editor, "Do-file editor");
                ui.checkbox(&mut self.show_graph, "Graph");
            });
            ui.menu_button("Results", |ui| {
                if ui.button("Clear results").clicked() {
                    self.session.transcript.clear();
                }
                if ui.button("Copy results").clicked() {
                    let text: Vec<String> = self.session.transcript.iter().map(|e| e.text.clone()).collect();
                    ui.ctx().copy_text(text.join("\n"));
                }
            });
        });
    }

    fn history_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("History");
        ui.separator();
        let mut run = None;
        egui::ScrollArea::vertical().id_salt("history").stick_to_bottom(true).show(ui, |ui| {
            for (i, h) in self.session.history.iter().enumerate() {
                let r = ui.add(egui::Label::new(RichText::new(h).monospace()).truncate().sense(egui::Sense::click()));
                if r.double_clicked() {
                    run = Some(h.clone());
                } else if r.clicked() {
                    self.command = h.clone();
                    self.hist_pos = Some(i);
                }
            }
        });
        if let Some(c) = run {
            self.run(&c);
        }
    }

    fn variables_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Variables");
        ui.separator();
        let vars = self.session.variables();
        let mut insert = None;
        egui::ScrollArea::vertical().id_salt("vars").max_height(ui.available_height() * 0.6).show(ui, |ui| {
            egui::Grid::new("vargrid").striped(true).num_columns(2).show(ui, |ui| {
                for (i, v) in vars.iter().enumerate() {
                    let r = ui.selectable_label(self.selected_var == Some(i), RichText::new(&v.name).monospace());
                    if r.clicked() {
                        self.selected_var = Some(i);
                    }
                    if r.double_clicked() {
                        insert = Some(v.name.clone());
                    }
                    ui.label(&v.label);
                    ui.end_row();
                }
            });
        });
        if let Some(n) = insert {
            if !self.command.is_empty() && !self.command.ends_with(' ') {
                self.command.push(' ');
            }
            self.command.push_str(&n);
        }
        ui.separator();
        ui.heading("Properties");
        if let Some(v) = self.selected_var.and_then(|i| vars.get(i)) {
            egui::Grid::new("props").num_columns(2).show(ui, |ui| {
                for (k, val) in [("Name", &v.name), ("Label", &v.label), ("Type", &v.vtype), ("Format", &v.format), ("Value label", &v.value_label)] {
                    ui.label(k);
                    ui.monospace(val);
                    ui.end_row();
                }
            });
        }
        ui.separator();
        ui.label(format!("{} variables, {} observations", self.session.n_vars(), self.session.n_obs()));
    }

    fn command_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(if self.session.pending() { ">" } else { "." });
            let edit = egui::TextEdit::singleline(&mut self.command).font(TextStyle::Monospace).desired_width(f32::INFINITY).hint_text("Command");
            let r = ui.add(edit);
            if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                let c = std::mem::take(&mut self.command);
                self.hist_pos = None;
                self.run(&c);
                r.request_focus();
            }
            if r.has_focus() {
                let hist = &self.session.history;
                if ui.input(|i| i.key_pressed(Key::ArrowUp)) && !hist.is_empty() {
                    let p = self.hist_pos.map_or(hist.len() - 1, |p| p.saturating_sub(1));
                    self.hist_pos = Some(p);
                    self.command = hist[p].clone();
                }
                if ui.input(|i| i.key_pressed(Key::ArrowDown)) {
                    match self.hist_pos {
                        Some(p) if p + 1 < hist.len() => {
                            self.hist_pos = Some(p + 1);
                            self.command = hist[p + 1].clone();
                        }
                        _ => {
                            self.hist_pos = None;
                            self.command.clear();
                        }
                    }
                }
            }
        });
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        let font = FontId::monospace(13.0);
        let dark = ui.visuals().dark_mode;
        let (cmd_c, err_c) = if dark { (Color32::WHITE, Color32::from_rgb(255, 110, 110)) } else { (Color32::BLACK, Color32::from_rgb(190, 0, 0)) };
        let scroll = std::mem::take(&mut self.scroll_to_end);
        egui::ScrollArea::both().id_salt("results").auto_shrink(false).stick_to_bottom(true).show(ui, |ui| {
            for e in &self.session.transcript {
                let t = RichText::new(&e.text).font(font.clone());
                let t = match e.kind {
                    EntryKind::Command => t.color(cmd_c).strong(),
                    EntryKind::Error => t.color(err_c),
                    EntryKind::Output => t,
                };
                ui.add(egui::Label::new(t).extend());
            }
            if scroll {
                ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
            }
        });
    }

    fn data_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_data;
        egui::Window::new("Data browser").open(&mut open).default_size([700.0, 450.0]).show(ctx, |ui| {
            let s = &self.session;
            let nv = s.n_vars();
            let n = s.n_obs();
            if nv == 0 {
                ui.label("No data in memory.");
                return;
            }
            egui::ScrollArea::horizontal().show(ui, |ui| {
                TableBuilder::new(ui)
                    .striped(true)
                    .resizable(true)
                    .column(Column::auto().at_least(40.0))
                    .columns(Column::initial(90.0).at_least(30.0).clip(true), nv)
                    .header(20.0, |mut h| {
                        h.col(|ui| {
                            ui.strong("#");
                        });
                        for v in s.variables() {
                            h.col(|ui| {
                                ui.strong(&v.name).on_hover_text(&v.label);
                            });
                        }
                    })
                    .body(|body| {
                        body.rows(18.0, n, |mut row| {
                            let i = row.index();
                            row.col(|ui| {
                                ui.monospace((i + 1).to_string());
                            });
                            for c in 0..nv {
                                row.col(|ui| {
                                    let t = s.cell(i, c);
                                    if s.is_string_var(c) {
                                        ui.label(RichText::new(t).monospace().color(Color32::from_rgb(180, 30, 30)));
                                    } else {
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            ui.monospace(t);
                                        });
                                    }
                                });
                            }
                        });
                    });
            });
        });
        self.show_data = open;
    }

    fn editor_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_editor;
        let mut to_run = None;
        let title = match &self.editor_path {
            Some(p) => format!("Do-file editor — {}", p.display()),
            None => "Do-file editor".to_string(),
        };
        egui::Window::new(title).id(egui::Id::unique("editor")).open(&mut open).default_size([600.0, 450.0]).show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Do-files", &["do", "ado"]).pick_file() {
                        if let Ok(t) = std::fs::read_to_string(&p) {
                            self.editor_text = t;
                            self.editor_path = Some(p);
                        }
                    }
                }
                if ui.button("Save").clicked() {
                    let path = self.editor_path.clone().or_else(|| rfd::FileDialog::new().add_filter("Do-files", &["do"]).save_file());
                    if let Some(p) = path {
                        if std::fs::write(&p, &self.editor_text).is_ok() {
                            self.editor_path = Some(p);
                        }
                    }
                }
                if ui.button("▶ Run all").clicked() {
                    to_run = Some(self.editor_text.clone());
                }
                if ui.button("Run selection").clicked() {
                    to_run = Some(selected_text(ui.ctx(), &self.editor_text).unwrap_or_else(|| self.editor_text.clone()));
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::TextEdit::multiline(&mut self.editor_text).id(egui::Id::unique("dofile")).code_editor().desired_rows(24).desired_width(f32::INFINITY).lock_focus(true).show(ui);
            });
        });
        self.show_editor = open;
        if let Some(t) = to_run {
            self.run_script(&t);
        }
    }

    fn graph_window(&mut self, ctx: &egui::Context) {
        let serial = self.session.graph_serial;
        if self.graph_tex.as_ref().map(|t| t.0) != Some(serial) {
            self.graph_tex = self.session.graph().and_then(|g| ferrostat_cmd::graph::render_rgb(g, GRAPH_W, GRAPH_H).ok()).map(|rgb| {
                let img = egui::ColorImage::from_rgb([GRAPH_W as usize, GRAPH_H as usize], &rgb);
                (serial, ctx.load_texture("graph", img, egui::TextureOptions::LINEAR))
            });
        }
        let mut open = self.show_graph;
        let mut export = None;
        egui::Window::new("Graph").open(&mut open).default_size([GRAPH_W as f32 * 0.8, GRAPH_H as f32 * 0.8 + 30.0]).show(ctx, |ui| {
            match &self.graph_tex {
                Some((_, tex)) => {
                    ui.horizontal(|ui| {
                        if ui.button("Export…").clicked() {
                            export = rfd::FileDialog::new().add_filter("PNG", &["png"]).add_filter("SVG", &["svg"]).add_filter("PDF", &["pdf"]).set_file_name("graph.png").save_file();
                        }
                    });
                    let avail = ui.available_size();
                    let scale = (avail.x / GRAPH_W as f32).min(avail.y / GRAPH_H as f32).max(0.1);
                    ui.image((tex.id(), egui::vec2(GRAPH_W as f32 * scale, GRAPH_H as f32 * scale)));
                }
                None => {
                    ui.label("No graph in memory.");
                }
            }
        });
        self.show_graph = open;
        if let Some(p) = export {
            self.run(&format!("graph export {}, replace", Self::quoted(&p)));
        }
    }
}

/// The text currently selected in the do-file editor, if any.
fn selected_text(ctx: &egui::Context, text: &str) -> Option<String> {
    let state = egui::TextEdit::load_state(ctx, egui::Id::unique("dofile"))?;
    let range = state.cursor.char_range()?;
    let (a, b) = (range.primary.index.min(range.secondary.index), range.primary.index.max(range.secondary.index));
    if a == b {
        return None;
    }
    let (a, b): (usize, usize) = (a.into(), b.into());
    Some(text.chars().skip(a).take(b - a).collect())
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.session.exited {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        egui::Panel::top("menu").show(ui, |ui| self.menu(ui));
        egui::Panel::bottom("command").min_size(32.0).show(ui, |ui| {
            ui.add_space(4.0);
            self.command_panel(ui);
        });
        egui::Panel::left("history").resizable(true).default_size(200.0).show(ui, |ui| self.history_panel(ui));
        egui::Panel::right("variables").resizable(true).default_size(260.0).show(ui, |ui| self.variables_panel(ui));
        egui::CentralPanel::default_margins().show(ui, |ui| self.results(ui));
        let ctx = ui.ctx().clone();
        if self.show_data {
            self.data_window(&ctx);
        }
        if self.show_editor {
            self.editor_window(&ctx);
        }
        if self.show_graph {
            self.graph_window(&ctx);
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Ferrostat").with_inner_size([1200.0, 780.0]),
        ..Default::default()
    };
    eframe::run_native("Ferrostat", options, Box::new(|_cc| Ok(Box::new(App::new()))))
}
