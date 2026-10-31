//! Viewer windows, generic dialog boxes, the user menu and the other
//! requests the interpreter sends to the desktop interface (`view`,
//! `doedit`, `db`, `window manage/push/menu/fopen/fsave`).

use eframe::egui::{self, Color32, RichText};
use ferrostat_ado::session::Session;
use ferrostat_ado::w11_ui::{is_menu_root, DialogKind, DialogSpec, FileDialogRequest, Frontend, MenuEntry, UiRequest};
use ferrostat_cmd::w11::smcl::{Line, Tone};

use crate::App;

#[derive(Default)]
pub struct W11Gui {
    viewers: Vec<Viewer>,
    dialogs: Vec<Dialog>,
    next_id: u64,
    urls: Vec<String>,
    title: Option<String>,
}

struct Viewer {
    id: u64,
    title: String,
    lines: Vec<Line>,
    open: bool,
}

struct Dialog {
    id: u64,
    spec: DialogSpec,
    depvar: String,
    picked: Vec<String>,
    args: String,
    if_exp: String,
    in_range: String,
    options: String,
    open: bool,
}

/// What to do after drawing (links, dialog buttons, menu items).
enum Action {
    Run(String),
    Link(String),
}

/// Declare the desktop front end and install native file dialogs.
pub fn attach(session: &mut Session) {
    session.interp.set_frontend(Frontend::Gui);
    session.interp.set_file_dialog(Some(Box::new(|req: &FileDialogRequest| {
        let mut d = rfd::FileDialog::new();
        if !req.title.is_empty() {
            d = d.set_title(&req.title);
        }
        for (desc, exts) in &req.filters {
            if !exts.iter().any(|e| e == "*") {
                d = d.add_filter(desc, exts);
            }
        }
        let p = if req.save { d.save_file() } else { d.pick_file() };
        p.map(|p| p.display().to_string())
    })));
}

/// Take the interpreter's requests after a command ran.
pub fn drain(app: &mut App) {
    for r in app.session.interp.take_ui_requests() {
        match r {
            UiRequest::View { title, lines } => {
                app.w11.next_id += 1;
                let id = app.w11.next_id;
                app.w11.viewers.push(Viewer { id, title, lines, open: true });
            }
            UiRequest::Browse(url) => app.w11.urls.push(url),
            UiRequest::Edit(path) => {
                if let Some(p) = path {
                    app.editor_text = ferrostat_core::vfs::read_to_string(&p).unwrap_or_default();
                    app.editor_path = Some(std::path::PathBuf::from(p));
                }
                app.show_editor = true;
            }
            UiRequest::Dialog(spec) => {
                app.w11.next_id += 1;
                let id = app.w11.next_id;
                app.w11.dialogs.push(Dialog { id, spec, depvar: String::new(), picked: Vec::new(), args: String::new(), if_exp: String::new(), in_range: String::new(), options: String::new(), open: true });
            }
            UiRequest::Manage(words) => manage(app, &words),
            UiRequest::PushHistory(line) => {
                if !line.is_empty() {
                    app.session.history.push(line);
                }
            }
            UiRequest::MenuChanged => {}
        }
    }
}

/// `window manage forward|close <window>`, `window manage maintitle "t"`.
fn manage(app: &mut App, w: &[String]) {
    let a = w.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let b = w.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
    match (a.as_str(), b.as_str()) {
        ("forward", "graph") => app.show_graph = app.session.graph_visible(),
        ("forward", "doeditor" | "doedit" | "editor") => app.show_editor = true,
        ("forward", "databrowser" | "dataeditor" | "data") => app.show_data = true,
        ("close", "graph") => app.show_graph = false,
        ("close", "doeditor" | "doedit" | "editor") => app.show_editor = false,
        ("close", "databrowser" | "dataeditor" | "data") => app.show_data = false,
        ("close", "viewer") => app.w11.viewers.clear(),
        ("maintitle", "reset") => app.w11.title = Some("Ferrostat".into()),
        ("maintitle", _) => app.w11.title = w.get(1).cloned(),
        _ => {}
    }
}

fn tone_color(t: Tone, dark: bool) -> Option<Color32> {
    match (t, dark) {
        (Tone::Text, _) => None,
        (Tone::Result, false) => Some(Color32::BLACK),
        (Tone::Result, true) => Some(Color32::WHITE),
        (Tone::Error, false) => Some(Color32::from_rgb(190, 0, 0)),
        (Tone::Error, true) => Some(Color32::from_rgb(255, 110, 110)),
        (Tone::Input | Tone::Command, false) => Some(Color32::from_rgb(20, 20, 20)),
        (Tone::Input | Tone::Command, true) => Some(Color32::from_rgb(235, 235, 235)),
    }
}

fn line_ui(ui: &mut egui::Ui, line: &Line, actions: &mut Vec<Action>) {
    let dark = ui.visuals().dark_mode;
    let link_c = ui.visuals().hyperlink_color;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        if line.is_empty() {
            ui.label(RichText::new(" ").monospace());
        }
        for s in line {
            let mut t = RichText::new(&s.text).monospace();
            if s.style.bold {
                t = t.strong();
            }
            if s.style.italic {
                t = t.italics();
            }
            if s.style.underline {
                t = t.underline();
            }
            if let Some(c) = tone_color(s.style.tone, dark) {
                t = t.color(c);
            }
            match &s.style.link {
                Some(target) => {
                    let r = ui.add(egui::Label::new(t.color(link_c).underline()).sense(egui::Sense::click())).on_hover_text(target);
                    if r.clicked() {
                        actions.push(Action::Link(target.clone()));
                    }
                }
                None => {
                    ui.label(t);
                }
            }
        }
    });
}

fn dialog_ui(ui: &mut egui::Ui, d: &mut Dialog, vars: &[String], actions: &mut Vec<Action>, to_command: &mut Option<String>) {
    egui::Grid::new(("dlg-grid", d.id)).num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
        if d.spec.kind == DialogKind::Estimation {
            ui.label("Dependent variable");
            egui::ComboBox::from_id_salt(("dlg-dep", d.id)).selected_text(d.depvar.as_str()).show_ui(ui, |ui| {
                for v in vars {
                    ui.selectable_value(&mut d.depvar, v.clone(), v);
                }
            });
            ui.end_row();
        }
        if d.spec.kind != DialogKind::Generic {
            ui.label(if d.spec.kind == DialogKind::Estimation { "Independent variables" } else { "Variables" });
            egui::ScrollArea::vertical().id_salt(("dlg-vars", d.id)).max_height(150.0).show(ui, |ui| {
                if vars.is_empty() {
                    ui.weak("(no variables in memory)");
                }
                for v in vars {
                    let mut on = d.picked.contains(v);
                    if ui.checkbox(&mut on, v).changed() {
                        if on {
                            d.picked.push(v.clone());
                        } else {
                            d.picked.retain(|x| x != v);
                        }
                    }
                }
            });
            ui.end_row();
        }
        ui.label(if d.spec.kind == DialogKind::Generic { "Arguments" } else { "Other arguments" });
        ui.add(egui::TextEdit::singleline(&mut d.args).desired_width(320.0));
        ui.end_row();
        if d.spec.ifin {
            ui.label("if (expression)");
            ui.add(egui::TextEdit::singleline(&mut d.if_exp).desired_width(320.0));
            ui.end_row();
            ui.label("in (range)");
            ui.add(egui::TextEdit::singleline(&mut d.in_range).desired_width(320.0));
            ui.end_row();
        }
        ui.label("Options");
        ui.add(egui::TextEdit::singleline(&mut d.options).desired_width(320.0));
        ui.end_row();
    });
    // variables in dataset order
    let ordered: Vec<String> = vars.iter().filter(|v| d.picked.contains(v)).cloned().collect();
    let line = d.spec.build(&d.depvar, &ordered, &d.args, &d.if_exp, &d.in_range, &d.options);
    ui.separator();
    ui.monospace(&line);
    ui.horizontal(|ui| {
        if ui.button("OK").clicked() {
            actions.push(Action::Run(line.clone()));
            d.open = false;
        }
        if ui.button("Submit").clicked() {
            actions.push(Action::Run(line.clone()));
        }
        if ui.button("Copy to command").clicked() {
            *to_command = Some(line.clone());
        }
        if ui.button("Cancel").clicked() {
            d.open = false;
        }
        if ui.button("Help").clicked() {
            actions.push(Action::Run(format!("view help {}", d.spec.command)));
        }
    });
}

/// Draw viewers and dialogs; carry out URL and title requests.
pub fn show(app: &mut App, ctx: &egui::Context) {
    for url in std::mem::take(&mut app.w11.urls) {
        ctx.open_url(egui::OpenUrl::new_tab(url));
    }
    if let Some(t) = app.w11.title.take() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(t));
    }
    let mut actions = Vec::new();
    for v in &mut app.w11.viewers {
        let mut open = v.open;
        egui::Window::new(&v.title).id(egui::Id::unique(("viewer", v.id))).open(&mut open).default_size([660.0, 480.0]).vscroll(false).show(ctx, |ui| {
            egui::ScrollArea::both().id_salt(("viewer-scroll", v.id)).auto_shrink(false).show(ui, |ui| {
                for line in &v.lines {
                    line_ui(ui, line, &mut actions);
                }
            });
        });
        v.open = open;
    }
    app.w11.viewers.retain(|v| v.open);
    let vars: Vec<String> = app.session.variables().into_iter().map(|v| v.name).collect();
    let mut to_command = None;
    for d in &mut app.w11.dialogs {
        let mut open = d.open;
        egui::Window::new(format!("{} - dialog", d.spec.command)).id(egui::Id::unique(("dialog", d.id))).open(&mut open).resizable(true).show(ctx, |ui| {
            dialog_ui(ui, d, &vars, &mut actions, &mut to_command);
        });
        d.open = d.open && open;
    }
    app.w11.dialogs.retain(|d| d.open);
    if let Some(c) = to_command {
        app.command = c;
    }
    for a in actions {
        match a {
            Action::Run(c) => app.run(&c),
            Action::Link(target) => follow(app, &target),
        }
    }
}

/// Follow a link of a viewer document.
fn follow(app: &mut App, target: &str) {
    let (verb, rest) = target.split_once(' ').unwrap_or((target, ""));
    match verb {
        "browse" => app.w11.urls.push(rest.to_string()),
        "run" => app.run(rest),
        "dialog" => app.run(&format!("db {rest}")),
        "help" | "search" | "net" | "ado" | "update" | "news" => app.run(&format!("view {verb} {rest}")),
        "view" => app.run(&format!("view {rest}")),
        _ => app.run(target),
    }
}

/// The user menu built with `window menu` (shown when not empty).
pub fn user_menu(app: &mut App, ui: &mut egui::Ui) {
    let menu: Vec<MenuEntry> = app.session.interp.user_menu().to_vec();
    if menu.is_empty() {
        return;
    }
    let mut chosen = None;
    ui.menu_button("User", |ui| {
        for e in menu.iter().filter(|e| is_menu_root(e.parent())) {
            entry_ui(ui, &menu, e, &mut chosen, 0);
        }
    });
    if let Some(c) = chosen {
        app.run(&c);
    }
}

fn entry_ui(ui: &mut egui::Ui, menu: &[MenuEntry], e: &MenuEntry, chosen: &mut Option<String>, depth: usize) {
    match e {
        MenuEntry::Submenu { label, .. } => {
            ui.menu_button(label, |ui| {
                if depth < 8 {
                    for c in menu.iter().filter(|c| c.parent() == label) {
                        entry_ui(ui, menu, c, chosen, depth + 1);
                    }
                }
            });
        }
        MenuEntry::Item { label, command, .. } => {
            if ui.button(label).clicked() {
                *chosen = Some(command.clone());
                ui.close();
            }
        }
        MenuEntry::Separator { .. } => {
            ui.separator();
        }
    }
}
