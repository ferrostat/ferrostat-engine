//! `collect` subcommands (everything except the `collect:` prefix and Word
//! export, which the interpreter handles).

use ferrostat_core::vfs;
use ferrostat_parser::parser::{find_top_level, split_options};

use crate::collect::{self, parse_axes, parse_tags, parse_term, split_words, CellOpts, Collection, HMode, Layout, Stars, Style};
use crate::eval::rc_err;
use crate::store::Store;
use crate::util::unquote;

pub type Opts = Vec<(String, Option<String>)>;

/// Split `main, options`.
pub fn split_main(s: &str) -> (String, Opts) {
    match find_top_level(s, ',') {
        Some(i) => (s[..i].trim().to_string(), split_options(&s[i + 1..])),
        None => (s.trim().to_string(), Vec::new()),
    }
}

pub fn opt<'a>(o: &'a Opts, name: &str, min: usize) -> Option<&'a Option<String>> {
    o.iter().find(|(n, _)| {
        let n = n.to_lowercase();
        n.len() >= min && name.starts_with(n.as_str())
    }).map(|x| &x.1)
}

pub fn opt_arg(o: &Opts, name: &str, min: usize) -> Option<String> {
    opt(o, name, min).map(|a| a.clone().unwrap_or_default())
}

fn has(o: &Opts, name: &str, min: usize) -> bool {
    opt(o, name, min).is_some()
}

fn name_opt(o: &Opts) -> Option<String> {
    opt_arg(o, "name", 4).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Run `collect <rest>` (not a prefix).
pub fn run(st: &mut Store, rest: &str) -> Result<String, String> {
    let rest = rest.trim();
    let (sub, args) = match rest.find(|c: char| c.is_whitespace() || c == ',') {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    };
    let sub = sub.to_lowercase();
    let is = |full: &str, min: usize| sub.len() >= min && full.starts_with(sub.as_str());
    let cs = &mut st.collect;
    if sub.is_empty() {
        return Err(rc_err(198, "collect subcommand required"));
    }
    if is("clear", 5) {
        *cs = Default::default();
        return Ok(String::new());
    }
    if is("create", 6) {
        let (main, o) = split_main(args);
        let name = main.trim();
        if name.is_empty() {
            return Err(rc_err(198, "collection name required"));
        }
        if cs.map.contains_key(name) && !has(&o, "replace", 3) {
            return Err(rc_err(110, format!("collection {name} already exists")));
        }
        cs.fresh(name);
        return Ok(format!("(current collection is {name})"));
    }
    if sub == "set" {
        let name = args.trim();
        if !cs.map.contains_key(name) {
            return Err(rc_err(111, format!("collection {name} not found")));
        }
        cs.current = name.to_string();
        return Ok(String::new());
    }
    if is("drop", 4) {
        for n in split_words(args) {
            cs.map.remove(&n);
            if cs.current == n {
                cs.current = "default".into();
                cs.map.entry("default".into()).or_insert_with(|| Collection::new("default"));
            }
        }
        return Ok(String::new());
    }
    if sub == "dir" {
        let mut out = vec!["Collections in memory".to_string(), format!("Current: {}", cs.current), "-".repeat(37), format!("{:<26}{:>11}", "Name", "No. items"), "-".repeat(37)];
        for (n, c) in &cs.map {
            out.push(format!("{:<26}{:>11}", n, c.items.len()));
        }
        out.push("-".repeat(37));
        return Ok(out.join("\n"));
    }
    if is("copy", 4) || is("rename", 6) {
        let (main, o) = split_main(args);
        let w = split_words(&main);
        if w.len() != 2 {
            return Err(rc_err(198, "two collection names required"));
        }
        if cs.map.contains_key(&w[1]) && !has(&o, "replace", 3) {
            return Err(rc_err(110, format!("collection {} already exists", w[1])));
        }
        let mut c = cs.get(Some(&w[0]))?.clone();
        c.name = w[1].clone();
        cs.map.insert(w[1].clone(), c);
        if sub.starts_with("ren") {
            cs.map.remove(&w[0]);
            if cs.current == w[0] {
                cs.current = w[1].clone();
            }
        }
        return Ok(String::new());
    }
    if sub == "get" {
        return get(st, args);
    }
    if is("label", 3) {
        return label(st, args);
    }
    if is("style", 3) {
        return style(st, args);
    }
    if is("layout", 3) {
        let (main, o) = split_main(args);
        let axes = parse_axes(&main)?;
        let mut l = Layout::default();
        for (k, a) in axes.into_iter().enumerate() {
            match k {
                0 => l.rows = a,
                1 => l.cols = a,
                2 => l.tabs = a,
                _ => return Err(rc_err(198, "too many layout specifications")),
            }
        }
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        c.layout = Some(l);
        return c.layout_report();
    }
    if is("preview", 3) {
        let (_, o) = split_main(args);
        return st.collect.get(name_opt(&o).as_deref())?.preview();
    }
    if is("title", 5) {
        let (main, o) = split_main(args);
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        let t = unquote(&main);
        c.title = (!t.is_empty()).then_some(t);
        return Ok(String::new());
    }
    if is("notes", 5) {
        let (main, o) = split_main(args);
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        if has(&o, "clear", 5) || main.is_empty() {
            if main.is_empty() {
                c.notes.clear();
            }
        }
        if !main.is_empty() {
            let m = main.trim();
            // `#: "text"` replaces note #
            let (num, text) = match m.split_once(':') {
                Some((n, t)) if n.trim().parse::<usize>().is_ok() => (n.trim().parse::<usize>().ok(), t.trim()),
                _ => (None, m),
            };
            let words = split_words(text);
            let text = words.iter().map(|w| unquote(w)).collect::<Vec<_>>().join(" ");
            match num {
                Some(k) if k >= 1 && k <= c.notes.len() => c.notes[k - 1] = text,
                _ => c.notes.push(text),
            }
        }
        return Ok(String::new());
    }
    if is("composite", 4) {
        return composite(st, args);
    }
    if sub == "stars" {
        let (main, o) = split_main(args);
        let w = split_words(&main);
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        if w.is_empty() {
            if let Some(s) = c.style.stars.as_mut() {
                if has(&o, "shownote", 5) {
                    s.shownote = true;
                }
                if has(&o, "noshownote", 7) {
                    s.shownote = false;
                }
            }
            return Ok(String::new());
        }
        let mut rules = Vec::new();
        let mut i = 1;
        while i + 1 < w.len() {
            let t: f64 = w[i].parse().map_err(|_| rc_err(198, format!("invalid stars threshold {}", w[i])))?;
            rules.push((t, unquote(&w[i + 1])));
            i += 2;
        }
        let attach = opt_arg(&o, "attach", 3).map(|a| split_words(&a)).unwrap_or_else(|| vec!["_r_b".into()]);
        c.style.stars = Some(Stars { pvalue: w[0].clone(), rules, attach, shownote: has(&o, "shownote", 5) });
        return Ok(String::new());
    }
    if is("dims", 4) {
        let (_, o) = split_main(args);
        let c = st.collect.get(name_opt(&o).as_deref())?;
        let mut out = vec!["Collection dimensions".to_string(), format!("Collection: {}", c.name), "-".repeat(41), format!("{:>28}   No. levels", "Dimension"), "-".repeat(41)];
        let mut dims = c.dims.clone();
        dims.sort();
        for d in dims {
            out.push(format!("{:>28}   {}", d, c.levels.get(&d).map_or(0, Vec::len)));
        }
        out.push("-".repeat(41));
        return Ok(out.join("\n"));
    }
    if is("levelsof", 6) {
        let (main, o) = split_main(args);
        let c = st.collect.get(name_opt(&o).as_deref())?;
        let d = main.trim();
        let lv = c.levels.get(d).ok_or_else(|| rc_err(111, format!("dimension {d} not found")))?;
        let text = lv.join(" ");
        st.r = Default::default();
        st.r.set_str("levels", text.clone());
        return Ok(format!("Collection: {}\n Dimension: {}\n    Levels: {}", st.collect.cur().name, d, text));
    }
    if is("query", 1) || is("describe", 1) {
        let (_, o) = split_main(args);
        let c = st.collect.get(name_opt(&o).as_deref())?;
        return Ok(format!("Collection: {}\n     Items: {}\n Dimensions: {}", c.name, c.items.len(), c.dims.join(" ")));
    }
    if is("recode", 5) {
        let (main, o) = split_main(args);
        let w = split_words(&main);
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        let dim = w.first().cloned().ok_or_else(|| rc_err(198, "dimension required"))?;
        let mut pairs = Vec::new();
        for x in w[1..].iter().map(|x| x.as_str()).collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().windows(3) {
            if x[1] == "=" {
                pairs.push((unquote(x[0]), unquote(x[2])));
            }
        }
        for it in c.items.iter_mut() {
            for t in it.tags.iter_mut() {
                if t.0 == dim {
                    if let Some((_, n)) = pairs.iter().find(|(a, _)| *a == t.1) {
                        t.1 = n.clone();
                    }
                }
            }
        }
        c.levels.remove(&dim);
        c.reindex();
        return Ok(String::new());
    }
    if is("addtags", 4) {
        let (main, o) = split_main(args);
        let tags = parse_tags(&main)?;
        let only = opt_arg(&o, "fortags", 3).map(|f| parse_term(&f)).transpose()?;
        let c = st.collect.get_mut(name_opt(&o).as_deref())?;
        for it in c.items.iter_mut() {
            let ok = only.as_ref().map_or(true, |f| f.iter().all(|cd| it.tag(&cd.dim).is_some_and(|l| cd.levels.as_ref().map_or(true, |ls| ls.iter().any(|x| x == l)))));
            if ok {
                for t in &tags {
                    it.tags.retain(|x| x.0 != t.0);
                    it.tags.push(t.clone());
                }
            }
        }
        c.reindex();
        return Ok(String::new());
    }
    if is("export", 3) {
        return export(st, args);
    }
    if matches!(sub.as_str(), "save" | "use" | "combine" | "remap" | "unget") {
        return Ok(String::new());
    }
    Err(rc_err(199, format!("collect {sub} is unrecognized")))
}

fn get(st: &mut Store, args: &str) -> Result<String, String> {
    let (main, o) = split_main(args);
    let tags = match opt_arg(&o, "tags", 3) {
        Some(t) => parse_tags(&t)?,
        None => Vec::new(),
    };
    let name = name_opt(&o);
    if let Some(n) = &name {
        if !st.collect.map.contains_key(n) {
            st.collect.map.insert(n.clone(), Collection::new(n));
        }
    }
    let (r, e) = (st.r.clone(), st.e.clone());
    let c = st.collect.get_mut(name.as_deref())?;
    let mut tags = tags;
    if tags.is_empty() {
        c.cmdset += 1;
        tags.push(("cmdset".into(), c.cmdset.to_string()));
    }
    for w in split_words(&main) {
        let w = w.trim();
        let (class, inner) = if let Some(x) = w.strip_prefix("r(") {
            ('r', x.trim_end_matches(')'))
        } else if let Some(x) = w.strip_prefix("e(") {
            ('e', x.trim_end_matches(')'))
        } else {
            return Err(rc_err(198, format!("{w} invalid; r() or e() results expected")));
        };
        let res = if class == 'r' { &r } else { &e };
        let only: Option<Vec<String>> = (!inner.trim().is_empty()).then(|| inner.split_whitespace().map(String::from).collect());
        collect::collect_results(c, res, class == 'e', &tags, only.as_deref());
    }
    Ok(String::new())
}

fn label(st: &mut Store, args: &str) -> Result<String, String> {
    let (sub, rest) = match args.find(char::is_whitespace) {
        Some(i) => (&args[..i], args[i..].trim()),
        None => (args, ""),
    };
    let (main, o) = split_main(rest);
    let c = st.collect.get_mut(name_opt(&o).as_deref())?;
    let w = split_words(&main);
    match sub.to_lowercase().as_str() {
        "levels" | "level" | "lev" => {
            let dim = w.first().ok_or_else(|| rc_err(198, "dimension required"))?;
            let mut i = 1;
            while i + 1 < w.len() {
                c.set_level_label(dim, &unquote(&w[i]), &unquote(&w[i + 1]));
                i += 2;
            }
            Ok(String::new())
        }
        "dim" | "dimension" => {
            if w.len() >= 2 {
                c.dim_labels.insert(w[0].clone(), unquote(&w[1]));
            }
            Ok(String::new())
        }
        "list" => {
            let dim = w.first().ok_or_else(|| rc_err(198, "dimension required"))?;
            let mut out = vec![format!("  Collection: {}", c.name), format!("   Dimension: {dim}"), format!("       Label: {}", c.dim_label(dim)), "Level labels:".to_string()];
            for l in c.levels.get(dim).cloned().unwrap_or_default() {
                out.push(format!("{:>12}  {}", l, c.level_label(dim, &l)));
            }
            Ok(out.join("\n"))
        }
        "drop" | "save" | "use" => Ok(String::new()),
        other => Err(rc_err(198, format!("collect label {other} invalid"))),
    }
}

fn composite(st: &mut Store, args: &str) -> Result<String, String> {
    let (sub, rest) = match args.find(char::is_whitespace) {
        Some(i) => (&args[..i], args[i..].trim()),
        None => (args, ""),
    };
    let (main, o) = split_main(rest);
    let c = st.collect.get_mut(name_opt(&o).as_deref())?;
    match sub {
        "define" => {
            let (n, e) = main.split_once('=').ok_or_else(|| rc_err(198, "composite define name = elements"))?;
            let elems: Vec<String> = e.split_whitespace().map(String::from).collect();
            c.composites.insert(n.trim().to_string(), elems);
            if let Some(d) = opt_arg(&o, "delimiter", 3) {
                let _ = d;
            }
            Ok(String::new())
        }
        _ => Ok(String::new()),
    }
}

fn header_mode(s: &str) -> Option<HMode> {
    match s.trim().to_lowercase().as_str() {
        "hide" => Some(HMode::Hide),
        "label" => Some(HMode::Label),
        "name" | "value" => Some(HMode::Name),
        _ => None,
    }
}

fn style(st: &mut Store, args: &str) -> Result<String, String> {
    let (sub, rest) = match args.find(|c: char| c.is_whitespace() || c == ',') {
        Some(i) => (&args[..i], args[i..].trim()),
        None => (args, ""),
    };
    let (main, o) = split_main(rest);
    let c = st.collect.get_mut(name_opt(&o).as_deref())?;
    match sub.to_lowercase().as_str() {
        "cell" => {
            let specs: Vec<String> = split_words(&main);
            let nformat = opt_arg(&o, "nformat", 2).map(|s| unquote(&s));
            let sformat = opt_arg(&o, "sformat", 2).map(|s| unquote(&s));
            let minimum = opt_arg(&o, "minimum", 3).and_then(|s| s.trim().parse::<f64>().ok());
            if let Some(b) = opt_arg(&o, "border", 3) {
                let b = b.to_lowercase();
                let nil = b.contains("nil");
                if b.contains("right") || b.starts_with("all") || b.trim().is_empty() || b.starts_with(',') {
                    c.style.vbar = !nil;
                }
            }
            if nformat.is_some() || sformat.is_some() || minimum.is_some() {
                let e = CellOpts { nformat, sformat, minimum };
                if specs.is_empty() {
                    c.style.cell.push((Vec::new(), e));
                } else {
                    for s in specs {
                        c.style.cell.push((parse_term(&s)?, e.clone()));
                    }
                }
            }
            Ok(String::new())
        }
        "header" => {
            let specs: Vec<String> = split_words(&main);
            let title = opt_arg(&o, "title", 3).and_then(|s| header_mode(&s));
            let level = opt_arg(&o, "level", 3).and_then(|s| header_mode(&s));
            if specs.is_empty() {
                if let Some(t) = title {
                    c.style.title_default = t;
                    c.style.title_mode.clear();
                }
                if let Some(l) = level {
                    c.style.level_default = l;
                    c.style.level_mode.clear();
                }
            }
            for s in specs {
                for cd in parse_term(&s)? {
                    if let Some(t) = title {
                        c.style.title_mode.insert(cd.dim.clone(), t);
                    }
                    if let Some(l) = level {
                        match &cd.levels {
                            None => {
                                c.style.level_mode.insert(cd.dim.clone(), l);
                                c.style.level_hide.remove(&cd.dim);
                            }
                            Some(ls) => {
                                let h = c.style.level_hide.entry(cd.dim.clone()).or_default();
                                for x in ls {
                                    h.retain(|y| y != x);
                                    if l == HMode::Hide {
                                        h.push(x.clone());
                                    }
                                }
                                if l != HMode::Hide && c.style.level_mode.get(&cd.dim) == Some(&HMode::Hide) {
                                    // show only the listed levels of a hidden dimension
                                    let others: Vec<String> = c.levels.get(&cd.dim).cloned().unwrap_or_default().into_iter().filter(|x| !ls.contains(x)).collect();
                                    c.style.level_mode.remove(&cd.dim);
                                    c.style.level_hide.entry(cd.dim.clone()).or_default().extend(others);
                                }
                            }
                        }
                    }
                }
            }
            Ok(String::new())
        }
        "autolevels" => {
            let w = split_words(&main);
            if let Some(d) = w.first() {
                let levels: Vec<String> = w[1..].iter().map(|x| unquote(x)).collect();
                let e = c.style.autolevels.entry(d.clone()).or_default();
                if has(&o, "clear", 5) {
                    e.clear();
                }
                for l in levels {
                    if !e.contains(&l) {
                        e.push(l);
                    }
                }
            }
            Ok(String::new())
        }
        "row" => {
            if has(&o, "spacer", 5) {
                c.style.spacer = true;
            }
            if has(&o, "nospacer", 7) {
                c.style.spacer = false;
            }
            Ok(String::new())
        }
        "use" => {
            let w = split_words(&main);
            if let Some(s) = w.first().and_then(|n| Style::named(n)) {
                if has(&o, "override", 4) {
                    c.style.cell.extend(s.cell);
                } else {
                    c.style = s;
                }
            }
            Ok(String::new())
        }
        "clear" => {
            c.style = Style::default_style();
            Ok(String::new())
        }
        "showbase" | "showomit" | "showempty" | "column" | "title" | "notes" | "table" | "putdocx" | "putpdf" | "html" | "tex" | "smcl" | "save" | "composite" | "etable" | "dtable" | "_cons" | "tables" => Ok(String::new()),
        other => Err(rc_err(198, format!("collect style {other} invalid"))),
    }
}

/// Export format from `as()` or the file extension.
pub fn export_kind(path: &str, o: &Opts) -> String {
    if let Some(a) = opt_arg(o, "as", 2) {
        return a.trim().to_lowercase();
    }
    path.rsplit('.').next().unwrap_or("").to_lowercase()
}

/// `collect export filename [, as() replace name() tableonly]` for every
/// format but Word documents.
pub fn export(st: &mut Store, args: &str) -> Result<String, String> {
    let (main, o) = split_main(args);
    let path = unquote(&main);
    if path.is_empty() {
        return Err(rc_err(198, "filename required"));
    }
    let c = st.collect.get(name_opt(&o).as_deref())?;
    let r = c.render()?;
    let kind = export_kind(&path, &o);
    if vfs::exists(&path) && !has(&o, "replace", 3) && !has(&o, "modify", 3) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let bytes: Vec<u8> = match kind.as_str() {
        "txt" | "smcl" | "log" => (r.text() + "\n").into_bytes(),
        "md" | "markdown" => r.markdown().into_bytes(),
        "html" | "htm" => r.html().into_bytes(),
        "tex" => latex(&r).into_bytes(),
        "xlsx" | "xls" => {
            let mut book = crate::xlsx::Workbook::default();
            let sheet = opt_arg(&o, "sheet", 3).map(|s| unquote(s.split(',').next().unwrap_or(""))).filter(|s| !s.is_empty()).unwrap_or_else(|| "Sheet1".into());
            let mut cells = std::collections::BTreeMap::new();
            let mut row = 1u32;
            if let Some(t) = &r.title {
                cells.insert((row, 1), crate::xlsx::Cell::Str(t.clone()));
                row += 1;
            }
            for t in &r.tables {
                if let Some(h) = &t.heading {
                    cells.insert((row, 1), crate::xlsx::Cell::Str(h.clone()));
                    row += 1;
                }
                let (g, _) = t.grid(&r.style);
                for gr in g {
                    for (j, s) in gr.iter().enumerate() {
                        if s.is_empty() {
                            continue;
                        }
                        let num = s.replace(',', "").parse::<f64>().ok().filter(|_| !s.starts_with(' '));
                        cells.insert((row, j as u32 + 1), match num {
                            Some(v) if j > 0 => crate::xlsx::Cell::Num(v),
                            _ => crate::xlsx::Cell::Str(s.trim_end().to_string()),
                        });
                    }
                    row += 1;
                }
                row += 1;
            }
            for n in &r.notes {
                cells.insert((row, 1), crate::xlsx::Cell::Str(n.clone()));
                row += 1;
            }
            book.sheets.push((sheet, cells));
            book.to_bytes()
        }
        "docx" => return Err(rc_err(198, "Word export is handled by the interpreter")),
        other => return Err(rc_err(198, format!("export format {other} not supported"))),
    };
    vfs::write(&path, &bytes).map_err(|e| rc_err(603, format!("could not write {path}: {e}")))?;
    Ok(format!("({} table saved to file {path})", kind_name(&kind)))
}

pub fn kind_name(kind: &str) -> &'static str {
    match kind {
        "docx" => "collection",
        _ => "collection",
    }
}

fn latex(r: &collect::Rendered) -> String {
    let esc = |s: &str| s.replace('\\', "\\textbackslash{}").replace('%', "\\%").replace('&', "\\&").replace('_', "\\_").replace('#', "\\#").replace('$', "\\$");
    let mut out = vec!["\\documentclass{article}".to_string(), "\\begin{document}".to_string()];
    for t in &r.tables {
        let (g, nh) = t.grid(&r.style);
        let nc = g.first().map_or(1, Vec::len);
        out.push("\\begin{table}[!h]\\centering".into());
        if let Some(ti) = &r.title {
            out.push(format!("\\caption{{{}}}", esc(ti)));
        }
        out.push(format!("\\begin{{tabular}}{{l{}}}", "r".repeat(nc - 1)));
        out.push("\\hline".into());
        for (k, row) in g.iter().enumerate() {
            if k == nh {
                out.push("\\hline".into());
            }
            out.push(format!("{} \\\\", row.iter().map(|s| esc(s)).collect::<Vec<_>>().join(" & ")));
        }
        out.push("\\hline".into());
        out.push("\\end{tabular}".into());
        out.push("\\end{table}".into());
    }
    for n in &r.notes {
        out.push(esc(n));
        out.push(String::new());
    }
    out.push("\\end{document}".into());
    out.join("\n")
}
