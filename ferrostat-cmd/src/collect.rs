//! Collections: the tagged result stores behind `table`, `dtable`, `etable`
//! and `collect`.  Every result is an item tagged with `dimension[level]`
//! pairs; a layout picks the dimensions shown in rows, columns and separate
//! tables; styles control labels, headers, number formats and borders.

use std::collections::{BTreeMap, HashMap};

use ferrostat_core::missing::is_missing;

use crate::eval::rc_err;
use crate::format::{fmt, fmt_g};

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Num(f64),
    Str(String),
}

#[derive(Clone, Debug)]
pub struct Item {
    pub tags: Vec<(String, String)>,
    pub val: Val,
}

impl Item {
    pub fn tag(&self, dim: &str) -> Option<&str> {
        self.tags.iter().find(|(d, _)| d == dim).map(|(_, l)| l.as_str())
    }

    fn has_all(&self, tags: &[(String, String)]) -> bool {
        tags.iter().all(|(d, l)| self.tag(d) == Some(l.as_str()))
    }
}

/// How a header element (dimension title or level) is shown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HMode {
    Label,
    Name,
    Hide,
}

/// `dim` or `dim[lev lev ...]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cond {
    pub dim: String,
    pub levels: Option<Vec<String>>,
}

/// An interaction of conditions (`a#b[x]`).
pub type Term = Vec<Cond>;

#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub rows: Vec<Term>,
    pub cols: Vec<Term>,
    pub tabs: Vec<Term>,
}

#[derive(Clone, Debug, Default)]
pub struct CellOpts {
    pub nformat: Option<String>,
    pub sformat: Option<String>,
    pub minimum: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Stars {
    /// Result level holding the p-values.
    pub pvalue: String,
    /// (threshold, label), most significant first.
    pub rules: Vec<(f64, String)>,
    /// Result levels the stars are attached to.
    pub attach: Vec<String>,
    pub shownote: bool,
}

impl Stars {
    pub fn label(&self, p: f64) -> String {
        if is_missing(p) {
            return String::new();
        }
        let mut rules = self.rules.clone();
        rules.sort_by(|a, b| a.0.total_cmp(&b.0));
        rules.iter().find(|(t, _)| p < *t).map(|(_, l)| l.clone()).unwrap_or_default()
    }

    /// `** p<.01, * p<.05`
    pub fn note(&self) -> String {
        let mut rules = self.rules.clone();
        rules.sort_by(|a, b| a.0.total_cmp(&b.0));
        rules
            .iter()
            .map(|(t, l)| {
                let s = fmt_g(*t, 8);
                let s = s.strip_prefix('0').unwrap_or(&s).to_string();
                format!("{l} p<{s}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug)]
pub struct Style {
    pub name: String,
    /// Cell style edits; later edits override earlier ones.
    pub cell: Vec<(Vec<Cond>, CellOpts)>,
    pub title_default: HMode,
    pub level_default: HMode,
    pub title_mode: BTreeMap<String, HMode>,
    pub level_mode: BTreeMap<String, HMode>,
    /// Individual levels whose header is hidden.
    pub level_hide: BTreeMap<String, Vec<String>>,
    pub autolevels: BTreeMap<String, Vec<String>>,
    /// Vertical rule between the row headers and the cells.
    pub vbar: bool,
    /// Row headers right-aligned (indentation then goes on the right).
    pub row_right: bool,
    /// Blank row between groups of the outermost row dimension.
    pub spacer: bool,
    /// Single-column headers centred instead of right-aligned.
    pub col_center: bool,
    pub first_sep: usize,
    pub sep: usize,
    pub stars: Option<Stars>,
}

fn edit(spec: &str, nformat: Option<&str>, sformat: Option<&str>, minimum: Option<f64>) -> (Vec<Cond>, CellOpts) {
    (
        parse_term(spec).unwrap_or_default(),
        CellOpts { nformat: nformat.map(String::from), sformat: sformat.map(String::from), minimum },
    )
}

impl Style {
    pub fn default_style() -> Style {
        Style {
            name: "default".into(),
            cell: vec![
                edit("", Some("%9.0g"), None, None),
                edit("result[_r_z _r_z_abs]", Some("%8.2f"), None, None),
                edit("result[_r_p]", Some("%5.3f"), None, None),
            ],
            title_default: HMode::Hide,
            level_default: HMode::Label,
            title_mode: BTreeMap::new(),
            level_mode: BTreeMap::from([("stars".to_string(), HMode::Hide)]),
            level_hide: BTreeMap::new(),
            autolevels: BTreeMap::new(),
            vbar: true,
            row_right: false,
            spacer: false,
            col_center: false,
            first_sep: 1,
            sep: 1,
            stars: None,
        }
    }

    /// A named style (`collect style use`, `table, style()`).
    pub fn named(name: &str) -> Option<Style> {
        let mut s = Style::default_style();
        s.name = name.to_string();
        match name {
            "default" => {}
            "table" | "table-1" | "table-tab2" | "table-right" => {
                s.cell = vec![
                    edit("", Some("%9.0g"), None, None),
                    edit("result[_r_z _r_z_abs]", Some("%8.2f"), None, None),
                    edit("result[_r_p]", Some("%5.3f"), None, None),
                    edit("result[count frequency fvrawfrequency]", Some("%18.0fc"), None, None),
                    edit("result[fvpercent fvrawpercent percent rawpercent]", Some("%9.2f"), None, None),
                    edit("result[fvproportion fvrawproportion proportion rawproportion]", Some("%9.4g"), None, None),
                    edit("result[rawtotal sumw total fvfrequency]", Some("%18.6gc"), None, None),
                ];
                s.title_default = HMode::Label;
                for d in ["across", "coleq", "colname", "command", "result", "roweq", "rowname", "stars", "statcmd", "var"] {
                    s.title_mode.insert(d.into(), HMode::Hide);
                }
                s.first_sep = 2;
                s.sep = 3;
                if name != "table" {
                    s.row_right = true;
                }
                if name == "table-1" || name == "table-tab2" {
                    s.level_mode.insert("result".into(), HMode::Hide);
                    s.spacer = true;
                }
            }
            "dtable" => {
                s.cell = vec![
                    edit("", Some("%21.3fc"), None, None),
                    edit("result[N count sumw frequency fvfrequency fvrawfrequency]", Some("%21.0fc"), None, None),
                    edit("result[percent rawpercent fvpercent fvrawpercent]", Some("%9.1fc"), Some("(%s%%)"), None),
                    edit("result[sd]", None, Some("(%s)"), None),
                    edit("result[idi iqi rangei]", None, Some("[%s]"), None),
                    edit("result[regress poisson lnormal kwallis kwallis_ties pearson fisher lrchi2]", Some("%6.3f"), None, Some(0.001)),
                ];
                s.vbar = false;
                s.col_center = true;
            }
            "etable" => {
                s.cell = vec![
                    edit("", Some("%9.2f"), None, None),
                    edit("result[N df_m df_r N_clust N_g]", Some("%21.0f"), None, None),
                    edit("result[_r_b _r_se _r_lb _r_ub _r_ci]", Some("%9.3f"), None, None),
                    edit("result[_r_p]", Some("%9.2f"), None, None),
                    edit("result[_r_se]", None, Some("(%s)"), None),
                    edit("result[_r_ci]", None, Some("[%s]"), None),
                ];
                s.vbar = false;
                s.col_center = true;
                s.level_hide.insert("result".into(), ["_r_b", "_r_se", "_r_z", "_r_z_abs", "_r_p", "_r_ci", "_r_lb", "_r_ub", "_r_df"].iter().map(|x| x.to_string()).collect());
                s.stars = Some(Stars { pvalue: "_r_p".into(), rules: vec![(0.01, "**".into()), (0.05, "*".into())], attach: vec!["_r_b".into()], shownote: false });
            }
            _ => return None,
        }
        Some(s)
    }

    fn title_shown(&self, dim: &str) -> HMode {
        *self.title_mode.get(dim).unwrap_or(&self.title_default)
    }

    fn level_shown(&self, dim: &str, level: &str) -> HMode {
        if self.level_hide.get(dim).is_some_and(|l| l.iter().any(|x| x == level)) {
            return HMode::Hide;
        }
        *self.level_mode.get(dim).unwrap_or(&self.level_default)
    }
}

#[derive(Clone, Debug)]
pub struct Collection {
    pub name: String,
    pub items: Vec<Item>,
    index: HashMap<String, usize>,
    pub dims: Vec<String>,
    pub levels: BTreeMap<String, Vec<String>>,
    pub dim_labels: BTreeMap<String, String>,
    pub level_labels: BTreeMap<String, BTreeMap<String, String>>,
    /// Levels shown under a group header (factor-variable levels under the
    /// variable label): dim -> level -> group label.
    pub groups: BTreeMap<String, BTreeMap<String, String>>,
    pub layout: Option<Layout>,
    pub title: Option<String>,
    pub notes: Vec<String>,
    pub style: Style,
    /// Composite result levels: name -> elements.
    pub composites: BTreeMap<String, Vec<String>>,
    /// Last automatic command-set index.
    pub cmdset: usize,
}

fn tag_key(tags: &[(String, String)]) -> String {
    let mut t: Vec<String> = tags.iter().map(|(d, l)| format!("{d}[{l}]")).collect();
    t.sort();
    t.join("#")
}

impl Collection {
    pub fn new(name: &str) -> Self {
        Collection {
            name: name.to_string(),
            items: Vec::new(),
            index: HashMap::new(),
            dims: Vec::new(),
            levels: BTreeMap::new(),
            dim_labels: BTreeMap::new(),
            level_labels: BTreeMap::new(),
            groups: BTreeMap::new(),
            layout: None,
            title: None,
            notes: Vec::new(),
            style: Style::default_style(),
            composites: BTreeMap::new(),
            cmdset: 0,
        }
    }

    /// Add (or replace) the item with exactly these tags.
    pub fn add(&mut self, tags: Vec<(String, String)>, val: Val) {
        for (d, l) in &tags {
            if !self.dims.contains(d) {
                self.dims.push(d.clone());
            }
            let lv = self.levels.entry(d.clone()).or_default();
            if !lv.contains(l) {
                lv.push(l.clone());
            }
        }
        let key = tag_key(&tags);
        match self.index.get(&key) {
            Some(&i) => self.items[i].val = val,
            None => {
                self.index.insert(key, self.items.len());
                self.items.push(Item { tags, val });
            }
        }
    }

    /// Rebuild the tag index, dimensions and levels after items were
    /// edited in place (later duplicates win).
    pub fn reindex(&mut self) {
        let items = std::mem::take(&mut self.items);
        self.index.clear();
        let old_levels = std::mem::take(&mut self.levels);
        self.dims.clear();
        for it in items {
            self.add(it.tags, it.val);
        }
        // keep the previous level order where possible
        for (d, lv) in self.levels.iter_mut() {
            if let Some(old) = old_levels.get(d) {
                lv.sort_by_key(|l| old.iter().position(|o| o == l).unwrap_or(usize::MAX));
            }
        }
    }

    pub fn num(&mut self, tags: &[(&str, &str)], v: f64) {
        self.add(tags.iter().map(|(d, l)| (d.to_string(), l.to_string())).collect(), Val::Num(v));
    }

    pub fn set_level_label(&mut self, dim: &str, level: &str, label: &str) {
        self.level_labels.entry(dim.to_string()).or_default().insert(level.to_string(), label.to_string());
    }

    pub fn set_level_label_default(&mut self, dim: &str, level: &str, label: &str) {
        self.level_labels.entry(dim.to_string()).or_default().entry(level.to_string()).or_insert_with(|| label.to_string());
    }

    pub fn set_group(&mut self, dim: &str, level: &str, group: &str) {
        self.groups.entry(dim.to_string()).or_default().insert(level.to_string(), group.to_string());
    }

    pub fn dim_label(&self, dim: &str) -> String {
        self.dim_labels.get(dim).cloned().unwrap_or_else(|| dim.to_string())
    }

    pub fn level_label(&self, dim: &str, level: &str) -> String {
        if let Some(l) = self.level_labels.get(dim).and_then(|m| m.get(level)) {
            return l.clone();
        }
        if (dim == "colname" || dim == "rowname") && level == "_cons" {
            return "Intercept".into();
        }
        if dim == "result" {
            if let Some(l) = default_result_label(level) {
                return l.to_string();
            }
        }
        level.to_string()
    }

    fn group(&self, dim: &str, level: &str) -> Option<&String> {
        self.groups.get(dim).and_then(|m| m.get(level))
    }

    fn header_level(&self, dim: &str, level: &str) -> Option<String> {
        match self.style.level_shown(dim, level) {
            HMode::Hide => None,
            HMode::Name => Some(level.to_string()),
            HMode::Label => Some(self.level_label(dim, level)),
        }
    }

    fn header_title(&self, dim: &str) -> Option<String> {
        match self.style.title_shown(dim) {
            HMode::Hide => None,
            HMode::Name => Some(dim.to_string()),
            HMode::Label => Some(self.dim_label(dim)),
        }
    }

    fn exists(&self, tags: &[(String, String)]) -> bool {
        if let Some(pos) = tags.iter().position(|(d, l)| d == "result" && self.composites.contains_key(l)) {
            let mut t = tags.to_vec();
            return self.composites[&tags[pos].1].iter().any(|e| {
                t[pos].1 = e.clone();
                self.exists(&t)
            });
        }
        self.items.iter().any(|it| it.has_all(tags))
    }

    /// The item matching all `tags` with the fewest other tags.
    pub fn find(&self, tags: &[(String, String)]) -> Option<&Item> {
        let mut best: Option<&Item> = None;
        for it in &self.items {
            if it.has_all(tags) && best.map_or(true, |b| it.tags.len() <= b.tags.len()) {
                best = Some(it);
            }
        }
        best
    }

    fn cell_opts(&self, it: &Item) -> CellOpts {
        let mut o = CellOpts::default();
        for (conds, e) in &self.style.cell {
            let ok = conds.iter().all(|c| match c.dim.as_str() {
                "cell_type" => c.levels.as_ref().map_or(true, |l| l.iter().any(|x| x == "item")),
                "border_block" => false,
                d => it.tag(d).is_some_and(|l| c.levels.as_ref().map_or(true, |ls| ls.iter().any(|x| x == l))),
            });
            if ok {
                if e.nformat.is_some() {
                    o.nformat = e.nformat.clone();
                }
                if e.sformat.is_some() {
                    o.sformat = e.sformat.clone();
                }
                if e.minimum.is_some() {
                    o.minimum = e.minimum;
                }
            }
        }
        o
    }

    pub fn format_item(&self, it: &Item) -> String {
        let o = self.cell_opts(it);
        let t = match &it.val {
            Val::Str(s) => s.clone(),
            Val::Num(v) => {
                let nf = o.nformat.as_deref().unwrap_or("%9.0g");
                if is_missing(*v) {
                    fmt(*v, nf).trim().to_string()
                } else if let Some(m) = o.minimum.filter(|m| v.abs() < *m) {
                    format!("<{}", fmt(m, nf).trim())
                } else {
                    fmt(*v, nf).trim().to_string()
                }
            }
        };
        match &o.sformat {
            Some(sf) => apply_sformat(sf, &t),
            None => t,
        }
    }

    /// Text and stars of the cell at `tags` (None: no item).
    fn cell(&self, tags: &[(String, String)]) -> Option<(String, String)> {
        if let Some(pos) = tags.iter().position(|(d, l)| d == "result" && self.composites.contains_key(l)) {
            let mut t = tags.to_vec();
            let parts: Vec<String> = self.composites[&tags[pos].1]
                .iter()
                .filter_map(|e| {
                    t[pos].1 = e.clone();
                    self.cell(&t).map(|c| if c.1.is_empty() { c.0 } else { format!("{}{}", c.0, c.1) })
                })
                .collect();
            return (!parts.is_empty()).then(|| (parts.join(" "), String::new()));
        }
        let it = self.find(tags)?;
        let text = self.format_item(it);
        let mut stars = String::new();
        if let Some(s) = &self.style.stars {
            if let Some(r) = it.tag("result") {
                if s.attach.iter().any(|a| a == r) {
                    let mut pt = it.tags.clone();
                    for t in pt.iter_mut() {
                        if t.0 == "result" {
                            t.1 = s.pvalue.clone();
                        }
                    }
                    if let Some(Val::Num(p)) = self.find(&pt).map(|x| &x.val) {
                        stars = s.label(*p);
                    }
                }
            }
        }
        Some((text, stars))
    }

    fn term_levels(&self, c: &Cond) -> Vec<String> {
        if let Some(l) = &c.levels {
            return l.clone();
        }
        if let Some(l) = self.style.autolevels.get(&c.dim) {
            return l.clone();
        }
        let mut l = self.levels.get(&c.dim).cloned().unwrap_or_default();
        if c.dim == "result" {
            for k in self.composites.keys() {
                if !l.contains(k) {
                    l.push(k.clone());
                }
            }
        }
        l
    }

    fn leaves(&self, terms: &[Term]) -> Vec<Leaf> {
        let mut out = Vec::new();
        for (ti, term) in terms.iter().enumerate() {
            let levels: Vec<Vec<String>> = term.iter().map(|c| self.term_levels(c)).collect();
            let mut cur = Vec::new();
            self.rec_leaves(term, &levels, 0, &mut cur, ti, &mut out);
        }
        out
    }

    fn rec_leaves(&self, term: &Term, levels: &[Vec<String>], k: usize, cur: &mut Vec<(String, String)>, ti: usize, out: &mut Vec<Leaf>) {
        if k == term.len() {
            out.push(Leaf { tags: cur.clone(), term: ti });
            return;
        }
        for l in &levels[k] {
            cur.push((term[k].dim.clone(), l.clone()));
            if self.exists(cur) {
                self.rec_leaves(term, levels, k + 1, cur, ti, out);
            }
            cur.pop();
        }
    }

    /// The layout in effect: the explicit one, or a default.
    pub fn effective_layout(&self) -> Result<Layout, String> {
        if let Some(l) = &self.layout {
            return Ok(l.clone());
        }
        if self.items.is_empty() {
            return Err(rc_err(119, format!("collection {} is empty", self.name)));
        }
        let other: Vec<&String> = self.dims.iter().filter(|d| *d != "result").collect();
        let rows = match other.first() {
            Some(d) => vec![vec![Cond { dim: (*d).clone(), levels: None }]],
            None => Vec::new(),
        };
        let cols = if self.dims.iter().any(|d| d == "result") { vec![vec![Cond { dim: "result".into(), levels: None }]] } else { Vec::new() };
        Ok(Layout { rows, cols, tabs: Vec::new() })
    }

    /// Lay out the collection.
    pub fn render(&self) -> Result<Rendered, String> {
        let layout = self.effective_layout()?;
        let tab_leaves = if layout.tabs.is_empty() { vec![Leaf { tags: Vec::new(), term: 0 }] } else { self.leaves(&layout.tabs) };
        let row_leaves = if layout.rows.is_empty() { vec![Leaf { tags: Vec::new(), term: 0 }] } else { self.leaves(&layout.rows) };
        let col_leaves = if layout.cols.is_empty() { vec![Leaf { tags: Vec::new(), term: 0 }] } else { self.leaves(&layout.cols) };
        let mut tables = Vec::new();
        for tl in &tab_leaves {
            let heading = (!tl.tags.is_empty()).then(|| {
                tl.tags.iter().map(|(d, l)| format!("{} = {}", self.dim_label(d), self.level_label(d, l))).collect::<Vec<_>>().join(", ")
            });
            // columns that exist within this table
            let cols: Vec<&Leaf> = col_leaves
                .iter()
                .filter(|c| {
                    let mut t = tl.tags.clone();
                    t.extend(c.tags.iter().cloned());
                    self.exists(&t)
                })
                .collect();
            let col_parts: Vec<Vec<String>> = cols
                .iter()
                .map(|c| {
                    let mut p = Vec::new();
                    for (d, l) in &c.tags {
                        if let Some(t) = self.header_title(d) {
                            p.push(t);
                        }
                        if let Some(g) = self.group(d, l) {
                            p.push(g.clone());
                        }
                        if let Some(t) = self.header_level(d, l) {
                            p.push(t);
                        }
                    }
                    p
                })
                .collect();
            let rows_in: Vec<&Leaf> = row_leaves
                .iter()
                .filter(|r| {
                    let mut t = tl.tags.clone();
                    t.extend(r.tags.iter().cloned());
                    self.exists(&t)
                })
                .collect();
            let mut rows: Vec<RRow> = Vec::new();
            let mut prev: Option<&Leaf> = None;
            for leaf in rows_in {
                let n = leaf.tags.len();
                let same = |k: usize| prev.is_some_and(|p| p.term == leaf.term && p.tags.len() >= k && n >= k && p.tags[..k] == leaf.tags[..k]);
                if self.style.spacer && n >= 2 {
                    if let Some(p) = prev {
                        let unit = |t: &(String, String)| self.group(&t.0, &t.1).cloned().unwrap_or_else(|| t.1.clone());
                        if p.term == leaf.term && unit(&p.tags[0]) != unit(&leaf.tags[0]) {
                            rows.push(RRow { indent: 0, label: String::new(), cells: None, spacer: true });
                        }
                    }
                }
                let start = rows.len();
                let mut indent = 0;
                let mut done = false;
                for k in 0..n {
                    let (d, l) = &leaf.tags[k];
                    if let Some(t) = self.header_title(d) {
                        if !same(k) {
                            rows.push(RRow { indent, label: t, cells: None, spacer: false });
                        }
                        indent += 1;
                    }
                    if let Some(g) = self.group(d, l) {
                        let pg = if same(k) { prev.and_then(|p| self.group(&p.tags[k].0, &p.tags[k].1)) } else { None };
                        if pg != Some(g) {
                            rows.push(RRow { indent, label: g.clone(), cells: None, spacer: false });
                        }
                        indent += 1;
                    }
                    if let Some(t) = self.header_level(d, l) {
                        if k + 1 < n {
                            if !same(k + 1) {
                                rows.push(RRow { indent, label: t, cells: None, spacer: false });
                            }
                            indent += 1;
                        } else {
                            rows.push(RRow { indent, label: t, cells: Some(Vec::new()), spacer: false });
                            done = true;
                        }
                    }
                }
                let mut tags = tl.tags.clone();
                tags.extend(leaf.tags.iter().cloned());
                let cells: Vec<(String, String)> = cols
                    .iter()
                    .map(|c| {
                        let mut t = tags.clone();
                        t.extend(c.tags.iter().cloned());
                        self.cell(&t).unwrap_or_default()
                    })
                    .collect();
                if done {
                    rows.last_mut().unwrap().cells = Some(cells);
                } else if rows.len() > start && rows.last().is_some_and(|r| r.cells.is_none() && !r.spacer) {
                    rows.last_mut().unwrap().cells = Some(cells);
                } else {
                    rows.push(RRow { indent, label: String::new(), cells: Some(cells), spacer: false });
                }
                prev = Some(leaf);
            }
            let stars: Vec<bool> = (0..cols.len()).map(|j| self.style.stars.is_some() && rows.iter().any(|r| r.cells.as_ref().is_some_and(|c| !c[j].1.is_empty()))).collect();
            tables.push(RTable { heading, cols: col_parts, stars, rows });
        }
        let mut notes = Vec::new();
        if let Some(s) = &self.style.stars {
            if s.shownote {
                notes.push(s.note());
            }
        }
        notes.extend(self.notes.iter().cloned());
        Ok(Rendered { title: self.title.clone(), tables, notes, style: self.style.clone() })
    }

    /// Text of the table(s) as shown by `collect preview`.
    pub fn preview(&self) -> Result<String, String> {
        Ok(self.render()?.text())
    }

    /// `collect layout` report followed by the table.
    pub fn layout_report(&self) -> Result<String, String> {
        let l = self.effective_layout()?;
        let r = self.render()?;
        let mut out = vec![format!("Collection: {}", self.name)];
        let show = |ts: &[Term]| ts.iter().map(term_text).collect::<Vec<_>>().join(" ");
        if !l.rows.is_empty() {
            out.push(format!("      Rows: {}", show(&l.rows)));
        }
        if !l.cols.is_empty() {
            out.push(format!("   Columns: {}", show(&l.cols)));
        }
        if !l.tabs.is_empty() {
            out.push(format!("    Tables: {}", show(&l.tabs)));
        }
        for (k, t) in r.tables.iter().enumerate() {
            let nr = t.rows.iter().filter(|x| x.cells.is_some()).count();
            out.push(format!("{:>10}: {} x {}", format!("Table {}", k + 1), nr, t.cols.len().max(1)));
        }
        out.push(String::new());
        out.push(r.text());
        Ok(out.join("\n"))
    }
}

/// Built-in labels of common result names.
pub fn default_result_label(level: &str) -> Option<&'static str> {
    Some(match level {
        "N" => "Number of observations",
        "r2" => "R-squared",
        "r2_a" => "Adjusted R-squared",
        "r2_p" => "Pseudo R-squared",
        "F" => "F statistic",
        "rmse" => "RMSE",
        "ll" => "Log likelihood",
        "mss" => "Model sum of squares",
        "rss" => "Residual sum of squares",
        "df_m" => "Model DF",
        "df_r" => "Residual DF",
        "N_clust" => "Number of clusters",
        "chi2" => "χ²",
        "p" => "Model test p-value",
        "aic" => "AIC",
        "bic" => "BIC",
        "sum_w" => "Sum of the weights",
        "mean" => "Mean",
        "sd" => "Std. dev.",
        "Var" => "Variance",
        "min" => "Minimum",
        "max" => "Maximum",
        "sum" => "Sum of variable",
        "skewness" => "Skewness",
        "kurtosis" => "Kurtosis",
        "p1" => "1st percentile",
        "p5" => "5th percentile",
        "p10" => "10th percentile",
        "p25" => "25th percentile",
        "p50" => "50th percentile",
        "p75" => "75th percentile",
        "p90" => "90th percentile",
        "p95" => "95th percentile",
        "p99" => "99th percentile",
        "_r_b" => "Coefficient",
        "_r_se" => "Std. error",
        "_r_z" => "z",
        "_r_z_abs" => "|z|",
        "_r_p" => "p-value",
        "_r_ci" => "95% CI",
        "_r_lb" => "95% lower bound",
        "_r_ub" => "95% upper bound",
        "_r_df" => "df",
        _ => return None,
    })
}

pub fn apply_sformat(sf: &str, t: &str) -> String {
    let mut out = String::new();
    let mut rest = sf;
    let mut used = false;
    while let Some(i) = rest.find('%') {
        out.push_str(&rest[..i]);
        let r = &rest[i + 1..];
        if let Some(r2) = r.strip_prefix('s').filter(|_| !used) {
            out.push_str(t);
            used = true;
            rest = r2;
        } else if let Some(r2) = r.strip_prefix('%') {
            out.push('%');
            rest = r2;
        } else {
            out.push('%');
            rest = r;
        }
    }
    out.push_str(rest);
    out
}

pub struct Leaf {
    pub tags: Vec<(String, String)>,
    pub term: usize,
}

/// A rendered row: row-header text with its indentation and, for data
/// rows, the (text, stars) of every column.
#[derive(Clone, Debug)]
pub struct RRow {
    pub indent: usize,
    pub label: String,
    pub cells: Option<Vec<(String, String)>>,
    pub spacer: bool,
}

#[derive(Clone, Debug)]
pub struct RTable {
    pub heading: Option<String>,
    /// Header parts of each column, top to bottom.
    pub cols: Vec<Vec<String>>,
    pub stars: Vec<bool>,
    pub rows: Vec<RRow>,
}

#[derive(Clone, Debug)]
pub struct Rendered {
    pub title: Option<String>,
    pub tables: Vec<RTable>,
    pub notes: Vec<String>,
    pub style: Style,
}

fn clen(s: &str) -> usize {
    s.chars().count()
}

/// A header span: first and last+1 column and its text.
type Span = (usize, usize, String);

impl RTable {
    /// Header rows with merged spans (bottom-aligned parts, duplicates merged).
    pub fn header_spans(&self) -> Vec<Vec<Span>> {
        let depth = self.cols.iter().map(Vec::len).max().unwrap_or(0);
        let part = |j: usize, h: usize| -> Option<&str> {
            let p = &self.cols[j];
            let off = depth - p.len();
            (h >= off).then(|| p[h - off].as_str())
        };
        let mut out = Vec::new();
        for h in 0..depth {
            let mut row = Vec::new();
            let mut j = 0;
            while j < self.cols.len() {
                let t = part(j, h);
                let mut k = j + 1;
                while k < self.cols.len() && t.is_some() && part(k, h) == t && (0..h).all(|u| part(k, u) == part(j, u)) {
                    k += 1;
                }
                if let Some(t) = t {
                    row.push((j, k, t.to_string()));
                }
                j = k;
            }
            out.push(row);
        }
        out
    }

    fn stub_text(&self, r: &RRow, right: bool) -> String {
        let ind = "  ".repeat(r.indent);
        if right { format!("{}{}", r.label, ind) } else { format!("{}{}", ind, r.label) }
    }

    pub fn text(&self, st: &Style) -> String {
        let nc = self.cols.len();
        let stub_w = self.rows.iter().map(|r| clen(&self.stub_text(r, st.row_right))).max().unwrap_or(0);
        let mut w: Vec<usize> = (0..nc).map(|j| self.rows.iter().filter_map(|r| r.cells.as_ref()).map(|c| clen(&c[j].0)).max().unwrap_or(0)).collect();
        let sw: Vec<usize> = (0..nc).map(|j| if self.stars[j] { self.rows.iter().filter_map(|r| r.cells.as_ref()).map(|c| clen(&c[j].1)).max().unwrap_or(0) } else { 0 }).collect();
        let block = |w: &[usize], j: usize| w[j] + if self.stars[j] { 1 + sw[j] } else { 0 };
        let sep = |j: usize| if j == 0 { st.first_sep } else { st.sep };
        let spans = self.header_spans();
        // widen columns to fit their headers
        for row in &spans {
            for (a, b, t) in row {
                let span_w: usize = (*a..*b).map(|j| block(&w, j)).sum::<usize>() + (*a + 1..*b).map(sep).sum::<usize>();
                if clen(t) > span_w {
                    w[b - 1] += clen(t) - span_w;
                }
            }
        }
        let lead = stub_w + if st.vbar { 2 } else { 0 };
        let mut x = Vec::with_capacity(nc);
        let mut pos = lead;
        for j in 0..nc {
            pos += sep(j);
            x.push(pos);
            pos += block(&w, j);
        }
        let total = pos.max(lead);
        let line = |stub: &str, right: bool| -> Vec<char> {
            let mut v: Vec<char> = if right { format!("{:>stub_w$}", stub) } else { format!("{:<stub_w$}", stub) }.chars().collect();
            if st.vbar {
                v.push(' ');
                v.push('|');
            }
            v.resize(total, ' ');
            v
        };
        let put = |v: &mut Vec<char>, at: usize, s: &str| {
            for (i, c) in s.chars().enumerate() {
                if at + i < v.len() {
                    v[at + i] = c;
                } else {
                    v.push(c);
                }
            }
        };
        let fin = |v: Vec<char>| v.into_iter().collect::<String>().trim_end().to_string();
        let mut out = Vec::new();
        if let Some(h) = &self.heading {
            out.push(h.clone());
        }
        out.push("-".repeat(total));
        for row in &spans {
            let mut v = line("", false);
            for (a, b, t) in row {
                let start = x[*a];
                let end = x[b - 1] + block(&w, b - 1);
                let span_w = end - start;
                let tl = clen(t);
                let at = if b - a > 1 || st.col_center { start + (span_w.saturating_sub(tl) + 1) / 2 } else { end.saturating_sub(tl) };
                put(&mut v, at, t);
            }
            out.push(fin(v));
        }
        if !spans.is_empty() {
            if st.vbar {
                out.push(format!("{}+{}", "-".repeat(stub_w + 1), "-".repeat(total - stub_w - 2)));
            } else {
                out.push("-".repeat(total));
            }
        }
        for r in &self.rows {
            let mut v = line(&self.stub_text(r, st.row_right), st.row_right);
            if let Some(cells) = &r.cells {
                for j in 0..nc {
                    let (t, s) = &cells[j];
                    let tl = clen(t);
                    put(&mut v, x[j] + w[j] - tl.min(w[j]), t);
                    if self.stars[j] && !s.is_empty() {
                        put(&mut v, x[j] + w[j] + 1, s);
                    }
                }
            }
            out.push(fin(v));
        }
        out.push("-".repeat(total));
        out.join("\n")
    }

    /// The table as a grid of strings (header rows first), for export.
    pub fn grid(&self, st: &Style) -> (Vec<Vec<String>>, usize) {
        let nc = self.cols.len();
        let mut g = Vec::new();
        let spans = self.header_spans();
        for row in &spans {
            let mut r = vec![String::new(); nc + 1];
            for (a, _, t) in row {
                r[a + 1] = t.clone();
            }
            g.push(r);
        }
        let nh = g.len();
        for r in &self.rows {
            let mut v = vec![self.stub_text(r, false)];
            match &r.cells {
                Some(c) => v.extend(c.iter().map(|(t, s)| if s.is_empty() { t.clone() } else { format!("{t}{s}") })),
                None => v.extend(std::iter::repeat(String::new()).take(nc)),
            }
            g.push(v);
        }
        let _ = st;
        (g, nh)
    }
}

impl Rendered {
    pub fn text(&self) -> String {
        let mut out = Vec::new();
        if let Some(t) = &self.title {
            out.push(t.clone());
        }
        for (k, t) in self.tables.iter().enumerate() {
            if k > 0 {
                out.push(String::new());
            }
            out.push(t.text(&self.style));
        }
        out.extend(self.notes.iter().cloned());
        out.join("\n")
    }

    pub fn markdown(&self) -> String {
        let mut out = Vec::new();
        if let Some(t) = &self.title {
            out.push(format!("**{t}**"));
            out.push(String::new());
        }
        for t in &self.tables {
            if let Some(h) = &t.heading {
                out.push(h.clone());
                out.push(String::new());
            }
            let (g, nh) = t.grid(&self.style);
            let esc = |s: &str| s.replace('|', "\\|");
            let ncol = g.first().map_or(1, Vec::len);
            let head: Vec<String> = if nh > 0 {
                (0..ncol).map(|j| (0..nh).map(|h| g[h][j].clone()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")).collect()
            } else {
                vec![String::new(); ncol]
            };
            out.push(format!("| {} |", head.iter().map(|s| esc(s)).collect::<Vec<_>>().join(" | ")));
            out.push(format!("|{}|", (0..ncol).map(|j| if j == 0 { ":---" } else { "---:" }).collect::<Vec<_>>().join("|")));
            for r in &g[nh..] {
                out.push(format!("| {} |", r.iter().map(|s| esc(s).replace("  ", "&nbsp;&nbsp;")).collect::<Vec<_>>().join(" | ")));
            }
            out.push(String::new());
        }
        for n in &self.notes {
            out.push(n.clone());
            out.push(String::new());
        }
        out.join("\n")
    }

    pub fn html(&self) -> String {
        let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        let mut out = vec!["<!DOCTYPE html>".to_string(), "<html><head><meta charset=\"utf-8\"><style>table{border-collapse:collapse}td,th{padding:2px 8px}td{text-align:right}td:first-child{text-align:left}thead tr:first-child th{border-top:1px solid}thead tr:last-child th{border-bottom:1px solid}tbody tr:last-child td{border-bottom:1px solid}</style></head><body>".to_string()];
        if let Some(t) = &self.title {
            out.push(format!("<p><b>{}</b></p>", esc(t)));
        }
        for t in &self.tables {
            if let Some(h) = &t.heading {
                out.push(format!("<p>{}</p>", esc(h)));
            }
            out.push("<table>".into());
            out.push("<thead>".into());
            for row in t.header_spans() {
                let mut line = String::from("<tr><th></th>");
                let mut j = 0;
                for (a, b, txt) in row {
                    if a > j {
                        line.push_str(&format!("<th colspan=\"{}\"></th>", a - j));
                    }
                    line.push_str(&format!("<th colspan=\"{}\" style=\"text-align:center\">{}</th>", b - a, esc(&txt)));
                    j = b;
                }
                if j < t.cols.len() {
                    line.push_str(&format!("<th colspan=\"{}\"></th>", t.cols.len() - j));
                }
                line.push_str("</tr>");
                out.push(line);
            }
            out.push("</thead><tbody>".into());
            for r in &t.rows {
                let pad = "&nbsp;&nbsp;".repeat(r.indent);
                let mut line = format!("<tr><td>{}{}</td>", pad, esc(&r.label));
                for j in 0..t.cols.len() {
                    let c = r.cells.as_ref().map(|c| format!("{}{}", c[j].0, c[j].1)).unwrap_or_default();
                    line.push_str(&format!("<td>{}</td>", esc(&c)));
                }
                line.push_str("</tr>");
                out.push(line);
            }
            out.push("</tbody></table>".into());
        }
        for n in &self.notes {
            out.push(format!("<p>{}</p>", esc(n)));
        }
        out.push("</body></html>".into());
        out.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Specification parsing
// ---------------------------------------------------------------------------

pub fn term_text(t: &Term) -> String {
    t.iter()
        .map(|c| match &c.levels {
            Some(l) => format!("{}[{}]", c.dim, l.join(" ")),
            None => c.dim.clone(),
        })
        .collect::<Vec<_>>()
        .join("#")
}

/// Split at top-level whitespace, keeping `[...]`, `(...)` and quotes together.
pub fn split_words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut quote = false;
    for c in s.chars() {
        match c {
            '"' => {
                quote = !quote;
                cur.push(c);
            }
            '[' | '(' if !quote => {
                depth += 1;
                cur.push(c);
            }
            ']' | ')' if !quote => {
                depth -= 1;
                cur.push(c);
            }
            c if c.is_whitespace() && depth == 0 && !quote => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Parse one level word: `"label"` quotes removed.
fn unq(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("`\"").and_then(|x| x.strip_suffix("\"'")).unwrap_or(s);
    s.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(s).to_string()
}

/// `a#b[x y]#c` -> conditions.
pub fn parse_term(s: &str) -> Result<Term, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    let mut parts = Vec::new();
    for c in s.chars() {
        match c {
            '[' => {
                depth += 1;
                cur.push(c)
            }
            ']' => {
                depth -= 1;
                cur.push(c)
            }
            '#' if depth == 0 => parts.push(std::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    parts.push(cur);
    for p in parts {
        let p = p.trim();
        if p.is_empty() {
            return Err(rc_err(198, format!("invalid specification '{s}'")));
        }
        match p.find('[') {
            Some(i) => {
                let inner = p[i + 1..].trim_end_matches(']');
                out.push(Cond { dim: p[..i].trim().to_string(), levels: Some(split_words(inner).iter().map(|w| unq(w)).collect()) });
            }
            None => out.push(Cond { dim: p.to_string(), levels: None }),
        }
    }
    Ok(out)
}

/// `(a b#c) (d)` or `a b` -> list of axes, each a list of terms.
pub fn parse_axes(s: &str) -> Result<Vec<Vec<Term>>, String> {
    let s = s.trim();
    let mut axes = Vec::new();
    if !s.starts_with('(') {
        for w in split_words(s) {
            axes.push(vec![parse_term(&w)?]);
        }
        return Ok(axes);
    }
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let terms = split_words(&cur).iter().map(|w| parse_term(w)).collect::<Result<Vec<_>, _>>()?;
                    axes.push(terms);
                    cur.clear();
                } else {
                    cur.push(c);
                }
            }
            c if depth > 0 => cur.push(c),
            c if c.is_whitespace() => {}
            _ => return Err(rc_err(198, format!("invalid layout specification '{s}'"))),
        }
    }
    Ok(axes)
}

// ---------------------------------------------------------------------------
// The set of collections of a session
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Collections {
    pub map: BTreeMap<String, Collection>,
    pub current: String,
}

impl Default for Collections {
    fn default() -> Self {
        let mut map = BTreeMap::new();
        map.insert("default".to_string(), Collection::new("default"));
        Collections { map, current: "default".into() }
    }
}

impl Collections {
    pub fn cur(&self) -> &Collection {
        self.map.get(&self.current).expect("current collection exists")
    }

    pub fn cur_mut(&mut self) -> &mut Collection {
        let name = self.current.clone();
        self.map.entry(name.clone()).or_insert_with(|| Collection::new(&name))
    }

    pub fn get(&self, name: Option<&str>) -> Result<&Collection, String> {
        let n = name.unwrap_or(&self.current);
        self.map.get(n).ok_or_else(|| rc_err(111, format!("collection {n} not found")))
    }

    pub fn get_mut(&mut self, name: Option<&str>) -> Result<&mut Collection, String> {
        let n = name.map(String::from).unwrap_or_else(|| self.current.clone());
        self.map.get_mut(&n).ok_or_else(|| rc_err(111, format!("collection {n} not found")))
    }

    /// Replace (or create) collection `name` and make it current.
    pub fn fresh(&mut self, name: &str) -> &mut Collection {
        self.map.insert(name.to_string(), Collection::new(name));
        self.current = name.to_string();
        self.map.get_mut(name).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Collecting stored results
// ---------------------------------------------------------------------------

const SUM_ORDER: [&str; 19] = ["N", "sum_w", "mean", "Var", "sd", "skewness", "kurtosis", "sum", "min", "max", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// Add r() or e() results to a collection with extra `tags`.
pub fn collect_results(c: &mut Collection, res: &crate::store::Results, eclass: bool, tags: &[(String, String)], only: Option<&[String]>) {
    let want = |n: &str| only.map_or(true, |o| o.iter().any(|x| x == n));
    let mut names: Vec<&String> = res.scalars.keys().collect();
    names.sort_by_key(|k| SUM_ORDER.iter().position(|o| o == k).unwrap_or(usize::MAX));
    if eclass {
        if let Some(b) = res.matrices.get("b").filter(|_| want("b") || only.is_none()) {
            let v = res.matrices.get("V");
            let df = res.get("df_r");
            for (j, name) in b.colnames.iter().enumerate() {
                let (eq, coef) = match name.rsplit_once(':') {
                    Some((e, c)) => (Some(e.to_string()), c.to_string()),
                    None => (None, name.clone()),
                };
                let bv = b.get(0, j);
                let se = v.filter(|m| m.rows > j).map(|m| m.get(j, j).max(0.0).sqrt()).unwrap_or(ferrostat_core::missing::SYSMISS);
                let z = if se > 0.0 && !is_missing(se) { bv / se } else { ferrostat_core::missing::SYSMISS };
                let p = if is_missing(z) {
                    z
                } else {
                    match df {
                        Some(d) => 2.0 * crate::dist::ttail(d, z.abs()),
                        None => 2.0 * (1.0 - crate::dist::normal(z.abs())),
                    }
                };
                let crit = match df {
                    Some(d) => crate::dist::invttail(d, 0.025),
                    None => crate::dist::invnormal(0.975),
                };
                let mut base: Vec<(String, String)> = tags.to_vec();
                if let Some(e) = &eq {
                    base.push(("coleq".into(), e.clone()));
                }
                base.push(("colname".into(), coef.clone()));
                for (r, val) in [("_r_b", bv), ("_r_se", se), ("_r_z", z), ("_r_p", p), ("_r_lb", bv - crit * se), ("_r_ub", bv + crit * se)] {
                    let mut t = base.clone();
                    t.push(("result".into(), r.into()));
                    c.add(t, Val::Num(val));
                }
            }
        }
    }
    for k in names {
        if !want(k) {
            continue;
        }
        let mut t: Vec<(String, String)> = tags.to_vec();
        t.push(("result".into(), k.clone()));
        c.add(t, Val::Num(res.scalars[k]));
    }
    if let Some(o) = only {
        for k in o {
            if let Some(m) = res.macros.get(k) {
                let mut t: Vec<(String, String)> = tags.to_vec();
                t.push(("result".into(), k.clone()));
                c.add(t, Val::Str(m.clone()));
            }
        }
    }
}

/// Parse `dim[level] dim[level]` tag lists.
pub fn parse_tags(s: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for w in split_words(s) {
        for c in parse_term(&w)? {
            match c.levels {
                Some(l) if l.len() == 1 => out.push((c.dim, l[0].clone())),
                _ => return Err(rc_err(198, format!("invalid tag specification '{w}'"))),
            }
        }
    }
    Ok(out)
}
