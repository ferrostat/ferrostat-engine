//! More community data-management commands, written from their documented
//! behaviour: `fre`, `sumup`, `tabout`, `rangestat`, `carryforward` and the
//! label utilities `labmask`, `labvalch`, `labvalclone`, `labvarch`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::vfs;
use ferrostat_core::{DataSet, Value, ValueLabel, VarType, Variable};
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::community_dm::{gstat, put, strip, take_any, valid_stat, Delegate, Gcollapse, Gegen, Glevelsof, Greshape, Gquantiles, Gstats, Gtop, Gunique, Join};
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

/// All commands of this module, for registration in the engine.
pub fn handlers() -> Handlers {
    vec![
        ("gcollapse", Box::new(Gcollapse { name: "gcollapse" })),
        ("fcollapse", Box::new(Gcollapse { name: "fcollapse" })),
        ("gegen", Box::new(Gegen { name: "gegen" })),
        ("fegen", Box::new(Gegen { name: "fegen" })),
        ("glevelsof", Box::new(Glevelsof { name: "glevelsof" })),
        ("flevelsof", Box::new(Glevelsof { name: "flevelsof" })),
        ("gquantiles", Box::new(Gquantiles { name: "gquantiles" })),
        ("fasterxtile", Box::new(Gquantiles { name: "fasterxtile" })),
        ("gstats", Box::new(Gstats)),
        ("gtop", Box::new(Gtop)),
        ("gtoplevelsof", Box::new(Gtop)),
        ("gunique", Box::new(Gunique)),
        ("greshape", Box::new(Greshape)),
        ("join", Box::new(Join)),
        ("gisid", Box::new(Delegate { name: "gisid", target: "isid" })),
        ("fisid", Box::new(Delegate { name: "fisid", target: "isid" })),
        ("gduplicates", Box::new(Delegate { name: "gduplicates", target: "duplicates" })),
        ("gdistinct", Box::new(Delegate { name: "gdistinct", target: "distinct" })),
        ("hashsort", Box::new(Delegate { name: "hashsort", target: "gsort" })),
        ("fsort", Box::new(Delegate { name: "fsort", target: "sort" })),
        ("fre", Box::new(Fre)),
        ("sumup", Box::new(Sumup)),
        ("tabout", Box::new(Tabout)),
        ("rangestat", Box::new(Rangestat)),
        ("carryforward", Box::new(Carryforward)),
        ("labmask", Box::new(Labmask)),
        ("labvalch", Box::new(Labvalch)),
        ("labvalclone", Box::new(Labvalclone)),
        ("labvarch", Box::new(Labvarch)),
    ]
}

fn is_miss_at(var: &Variable, i: usize) -> bool {
    match &var.data[i] {
        Value::Str(s) => s.is_empty(),
        x => is_missing(x.num()),
    }
}

// ---- fre ----------------------------------------------------------------------------

pub struct Fre;

impl CommandHandler for Fre {
    fn name(&self) -> &'static str {
        "fre"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
        let nolabel = cmd.has_opt("nolabel", 3);
        let all = cmd.has_opt("all", 3);
        let mut out = Vec::new();
        for v in &vars {
            let var = ds.get_var(v).unwrap();
            let is_str = var.var_type.is_string();
            // (key, display text, freq)
            let mut valid: Vec<(f64, String, String, f64)> = Vec::new();
            let mut miss: Vec<(f64, String, f64)> = Vec::new();
            let lbl = ds.value_labels.get(&var.value_label_name);
            let mut vmap: std::collections::BTreeMap<u64, usize> = Default::default();
            let mut smap: std::collections::BTreeMap<String, usize> = Default::default();
            let mut mmap: std::collections::BTreeMap<u64, usize> = Default::default();
            let order = |x: f64| {
                let b = x.to_bits();
                if x.is_sign_negative() { !b } else { b | (1 << 63) }
            };
            for i in 0..ds.n_obs {
                if !mask[i] {
                    continue;
                }
                let wi = w.as_ref().map_or(1.0, |w| w.w[i]);
                if is_str {
                    let s = var.str_at(i).to_string();
                    if s.is_empty() {
                        let k = 0u64;
                        let p = *mmap.entry(k).or_insert_with(|| { miss.push((0.0, ".".into(), 0.0)); miss.len() - 1 });
                        miss[p].2 += wi;
                    } else {
                        let p = *smap.entry(s.clone()).or_insert_with(|| { valid.push((0.0, String::new(), s.clone(), 0.0)); valid.len() - 1 });
                        valid[p].3 += wi;
                    }
                } else {
                    let x = var.num(i);
                    if is_missing(x) {
                        let p = *mmap.entry(order(x)).or_insert_with(|| {
                            let t = ferrostat_core::missing::missing_code(x).map_or(".".to_string(), ferrostat_core::missing::missing_name);
                            miss.push((x, t, 0.0));
                            miss.len() - 1
                        });
                        miss[p].2 += wi;
                    } else {
                        let p = *vmap.entry(order(x)).or_insert_with(|| {
                            let l = if nolabel { None } else { lbl.and_then(|l| l.lookup(x)) };
                            valid.push((x, crate::format::fmt_g(x, 17), l.unwrap_or("").to_string(), 0.0));
                            valid.len() - 1
                        });
                        valid[p].3 += wi;
                    }
                }
            }
            if all && !is_str && !nolabel {
                if let Some(l) = lbl {
                    for (k, t) in &l.mapping {
                        if !vmap.contains_key(&order(k.0)) {
                            valid.push((k.0, crate::format::fmt_g(k.0, 17), t.clone(), 0.0));
                        }
                    }
                }
            }
            if is_str {
                valid.sort_by(|a, b| a.2.cmp(&b.2));
            } else {
                valid.sort_by(|a, b| a.0.total_cmp(&b.0));
                miss.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            let nv: f64 = valid.iter().map(|x| x.3).sum();
            let nm: f64 = miss.iter().map(|x| x.2).sum();
            let ntot = nv + nm;
            let entry = |a: &str, b: &str| if is_str { b.to_string() } else if b.is_empty() { a.to_string() } else { format!("{a} {b}") };
            let ents: Vec<String> = valid.iter().map(|x| entry(&x.1, &x.2)).collect();
            let ments: Vec<String> = miss.iter().map(|x| x.1.clone()).collect();
            let wl = ents.iter().chain(ments.iter()).map(|s| s.len() + 8).max().unwrap_or(0).max(13);
            let title = if var.label.is_empty() { v.clone() } else { format!("{v} -- {}", var.label) };
            let rule = format!("{}+{}", "-".repeat(wl + 0), "-".repeat(44));
            out.push(title);
            out.push(format!("{}", "-".repeat(wl + 45)));
            out.push(format!("{:wl$}|      Freq.    Percent      Valid       Cum.", ""));
            out.push(rule.clone());
            let num = |f: f64| format!("{:>11}", if f.fract() == 0.0 { util::commas(f as usize) } else { format!("{f:.2}") });
            let mut cum = 0.0;
            for (k, x) in valid.iter().enumerate() {
                cum += x.3;
                let left = format!("{}{}", if k == 0 { "Valid   " } else { "        " }, ents[k]);
                out.push(format!("{left:<wl$}|{}{:>11.2}{:>11.2}{:>11.2}", num(x.3), 100.0 * x.3 / ntot, 100.0 * x.3 / nv, 100.0 * cum / nv));
            }
            out.push(format!("{:<wl$}|{}{:>11.2}{:>11.2}", "        Total", num(nv), 100.0 * nv / ntot, 100.0));
            for (k, x) in miss.iter().enumerate() {
                let left = format!("{}{}", if k == 0 { "Missing " } else { "        " }, ments[k]);
                out.push(format!("{left:<wl$}|{}{:>11.2}", num(x.2), 100.0 * x.2 / ntot));
            }
            if !miss.is_empty() {
                out.push(format!("{:<wl$}|{}{:>11.2}", "        Total", num(nm), 100.0 * nm / ntot));
            }
            out.push(format!("{:<wl$}|{}{:>11.2}", "Total", num(ntot), 100.0));
            out.push("-".repeat(wl + 45));
            out.push(String::new());
            let r = st.begin_r();
            r.set("N", ntot);
            r.set("N_valid", nv);
            r.set("N_miss", nm);
            r.set("r", valid.len() as f64);
        }
        Ok(out.join("\n").trim_end().to_string())
    }
}

// ---- sumup --------------------------------------------------------------------------

pub struct Sumup;

impl CommandHandler for Sumup {
    fn name(&self) -> &'static str {
        "sumup"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut c = cmd.clone();
        c.verb = "tabstat".into();
        let stats = take_any(&mut c, &["statistics", "stat", "stats", "s"]).flatten().unwrap_or_else(|| "count mean sd min max".into());
        strip(&mut c, &["statistics", "stat", "statistics"]);
        put(&mut c, "statistics", Some(&stats));
        if !c.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case("columns")) {
            put(&mut c, "columns", Some("statistics"));
        }
        crate::summarize::Tabstat.execute(ds, st, &c)
    }
}

// ---- tabout -------------------------------------------------------------------------

pub struct Tabout;

fn fmt_count(x: f64, fmt: &str) -> String {
    let dec = fmt.chars().find(|c| c.is_ascii_digit()).and_then(|c| c.to_digit(10)).unwrap_or(0) as usize;
    let s = format!("{x:.dec$}");
    if fmt.contains('c') {
        let (ip, fp) = s.split_once('.').map(|(a, b)| (a.to_string(), format!(".{b}"))).unwrap_or((s.clone(), String::new()));
        let mut o = String::new();
        for (k, ch) in ip.chars().rev().enumerate() {
            if k > 0 && k % 3 == 0 && ch != '-' {
                o.push(',');
            }
            o.push(ch);
        }
        format!("{}{fp}", o.chars().rev().collect::<String>())
    } else {
        s
    }
}

impl CommandHandler for Tabout {
    fn name(&self) -> &'static str {
        "tabout"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() || vars.len() > 2 {
            return Err(rc_err(198, "tabout requires one or two variables"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
        util::markout(ds, &mut mask, &vars);
        let cells: Vec<String> = cmd.opt("cells", 1).or_else(|| cmd.opt("c", 1)).map(|s| s.split_whitespace().map(|x| x.to_lowercase()).collect()).unwrap_or_else(|| vec!["freq".into()]);
        for c in &cells {
            if !matches!(c.as_str(), "freq" | "col" | "row" | "cell") {
                return Err(rc_err(198, format!("cells({c}) not allowed")));
            }
        }
        let fmts: Vec<String> = cmd.opt("format", 1).or_else(|| cmd.opt("f", 1)).map(|s| s.split_whitespace().map(|x| x.to_string()).collect()).unwrap_or_default();
        let fmt_of = |k: usize| fmts.get(k).or(fmts.last()).cloned().unwrap_or_else(|| if cells[k] == "freq" { "0c".into() } else { "1".into() });
        let style = cmd.opt("style", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "tsv".into());
        let title = cmd.opt("title", 2).map(util::unquote);
        let a = ds.get_var(&vars[0]).unwrap();
        let (ida, ra) = util::group_ids(ds, &vars[..1], &mask);
        let (idb, rb) = if vars.len() == 2 { util::group_ids(ds, &vars[1..], &mask) } else { (vec![0; ds.n_obs], vec![0]) };
        let (na, nb) = (ra.len(), rb.len());
        let mut cnt = vec![vec![0.0; nb]; na];
        for i in 0..ds.n_obs {
            if mask[i] {
                cnt[ida[i]][idb[i]] += w.as_ref().map_or(1.0, |w| w.w[i]);
            }
        }
        let rowt: Vec<f64> = cnt.iter().map(|r| r.iter().sum()).collect();
        let colt: Vec<f64> = (0..nb).map(|j| cnt.iter().map(|r| r[j]).sum()).collect();
        let n: f64 = rowt.iter().sum();
        let lab = |var: &str, i: usize| util::cell_text(ds, ds.get_var(var).unwrap(), i, true);
        let _ = a;
        let cell_text = |k: usize, c: f64, r: f64, cl: f64| -> String {
            let v = match cells[k].as_str() {
                "freq" => c,
                "col" => 100.0 * c / cl,
                "row" => 100.0 * c / r,
                _ => 100.0 * c / n,
            };
            fmt_count(v, &fmt_of(k))
        };
        let nc = cells.len();
        let mut table: Vec<Vec<String>> = Vec::new();
        let mut head = vec![if vars.len() == 2 { format!("{} / {}", ds.get_var(&vars[0]).unwrap().label.clone().chars().next().map_or(vars[0].clone(), |_| ds.get_var(&vars[0]).unwrap().label.clone()), vars[1]) } else { vars[0].clone() }];
        if vars.len() == 2 {
            for (j, &rep) in rb.iter().enumerate() {
                head.push(lab(&vars[1], rep));
                head.extend(std::iter::repeat_n(String::new(), nc - 1));
                let _ = j;
            }
            head.push("Total".into());
            head.extend(std::iter::repeat_n(String::new(), nc - 1));
        } else {
            for c in &cells {
                head.push(match c.as_str() { "freq" => "Freq.".into(), "col" => "Col %".into(), "row" => "Row %".into(), _ => "Cell %".into() });
            }
        }
        table.push(head);
        if vars.len() == 2 && nc > 1 {
            let mut h2 = vec![String::new()];
            for _ in 0..=nb {
                h2.extend(cells.iter().map(|c| c.clone()));
            }
            table.push(h2);
        }
        for (i, &rep) in ra.iter().enumerate() {
            let mut row = vec![lab(&vars[0], rep)];
            if vars.len() == 2 {
                for j in 0..nb {
                    for k in 0..nc {
                        row.push(cell_text(k, cnt[i][j], rowt[i], colt[j]));
                    }
                }
            }
            for k in 0..nc {
                row.push(cell_text(k, rowt[i], rowt[i], n));
            }
            table.push(row);
        }
        let mut tot = vec!["Total".to_string()];
        if vars.len() == 2 {
            for j in 0..nb {
                for k in 0..nc {
                    tot.push(cell_text(k, colt[j], n, colt[j]));
                }
            }
        }
        for k in 0..nc {
            tot.push(cell_text(k, n, n, n));
        }
        table.push(tot);
        let mut text = String::new();
        let esc_tex = |s: &str| s.replace('%', "\\%").replace('&', "\\&").replace('_', "\\_");
        match style.as_str() {
            "csv" => {
                if let Some(t) = &title {
                    text.push_str(&format!("\"{t}\"\n"));
                }
                for r in &table {
                    text.push_str(&r.iter().map(|c| if c.contains([',', '"']) { format!("\"{}\"", c.replace('"', "\"\"")) } else { c.clone() }).collect::<Vec<_>>().join(","));
                    text.push('\n');
                }
            }
            "tex" | "latex" => {
                if let Some(t) = &title {
                    text.push_str(&format!("\\caption{{{}}}\n", esc_tex(t)));
                }
                text.push_str(&format!("\\begin{{tabular}}{{l{}}}\n\\hline\n", "r".repeat(table[0].len() - 1)));
                for (k, r) in table.iter().enumerate() {
                    text.push_str(&r.iter().map(|c| esc_tex(c)).collect::<Vec<_>>().join(" & "));
                    text.push_str(" \\\\\n");
                    if k == 0 || k + 2 == table.len() {
                        text.push_str("\\hline\n");
                    }
                }
                text.push_str("\\hline\n\\end{tabular}\n");
            }
            "htm" | "html" => {
                text.push_str("<table border=\"1\">\n");
                if let Some(t) = &title {
                    text.push_str(&format!("<caption>{t}</caption>\n"));
                }
                for (k, r) in table.iter().enumerate() {
                    let tag = if k == 0 { "th" } else { "td" };
                    text.push_str(&format!("<tr>{}</tr>\n", r.iter().map(|c| format!("<{tag}>{}</{tag}>", c.replace('&', "&amp;").replace('<', "&lt;"))).collect::<String>()));
                }
                text.push_str("</table>\n");
            }
            "tsv" | "tab" | "txt" => {
                if let Some(t) = &title {
                    text.push_str(&format!("{t}\n"));
                }
                for r in &table {
                    text.push_str(&r.join("\t"));
                    text.push('\n');
                }
            }
            s => return Err(rc_err(198, format!("style({s}) invalid"))),
        }
        let r = st.begin_r();
        r.set("N", n);
        match &cmd.using {
            Some(f) => {
                let f = util::unquote(f);
                if cmd.has_opt("append", 3) && vfs::exists(&f) {
                    vfs::append(&f, text.as_bytes())
                } else if vfs::exists(&f) && !cmd.has_opt("replace", 3) {
                    return Err(rc_err(602, format!("file {f} already exists")));
                } else {
                    vfs::write(&f, text.as_bytes())
                }
                .map_err(|e| rc_err(603, format!("file {f} could not be written: {e}")))?;
                Ok(format!("Table output written to: {f}"))
            }
            None => Ok(text.trim_end().to_string()),
        }
    }
}

// ---- rangestat ------------------------------------------------------------------------

pub struct Rangestat;

impl CommandHandler for Rangestat {
    fn name(&self) -> &'static str {
        "rangestat"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut targets: Vec<(String, String, String)> = Vec::new();
        let mut stat = "mean".to_string();
        for w in split_top_level_words(&cmd.main.replace(" = ", "=").replace("= ", "=").replace(" =", "=")) {
            if w.starts_with('(') && w.ends_with(')') {
                stat = w[1..w.len() - 1].trim().to_lowercase();
                if stat == "variance" || stat == "var" {
                    stat = "variance".into();
                }
                if !valid_stat(&stat) || matches!(stat.as_str(), "percent" | "rawsum") {
                    return Err(rc_err(198, format!("unknown statistic {stat}")));
                }
                continue;
            }
            if let Some((new, src)) = w.split_once('=') {
                targets.push((stat.clone(), new.to_string(), varlist::resolve_name(ds, src)?));
            } else {
                for v in varlist::expand(ds, &w)? {
                    targets.push((stat.clone(), format!("{v}_{stat}"), v));
                }
            }
        }
        if targets.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        for t in &targets {
            if ds.var_index.contains_key(&t.1) {
                return Err(rc_err(110, format!("variable {} already defined", t.1)));
            }
        }
        let n = ds.n_obs;
        let by = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let mut mask = util::touse(ds, st, cmd)?;
        if cmd.has_opt("casewise", 2) {
            let srcs: Vec<String> = targets.iter().map(|t| t.2.clone()).collect();
            util::markout(ds, &mut mask, &srcs);
        }
        let excl = cmd.has_opt("excludeself", 3);
        // interval(key low high)
        let (key, lo, hi): (Option<Vec<f64>>, Vec<f64>, Vec<f64>) = match cmd.opt("interval", 3) {
            Some(spec) => {
                let w: Vec<&str> = spec.split_whitespace().collect();
                if w.len() != 3 {
                    return Err(rc_err(198, "interval(keyvar low high) required"));
                }
                let kv = varlist::resolve_name(ds, w[0])?;
                let key = util::col(ds, &kv)?;
                let bound = |s: &str, unb: f64| -> Result<Vec<f64>, String> {
                    if s == "." {
                        Ok(vec![unb; n])
                    } else if let Ok(x) = s.parse::<f64>() {
                        Ok(vec![x; n])
                    } else {
                        util::col(ds, &varlist::resolve_name(ds, s)?)
                    }
                };
                (Some(key), bound(w[1], f64::NEG_INFINITY)?, bound(w[2], f64::INFINITY)?)
            }
            None => (None, vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
        };
        let (ids, ng) = if by.is_empty() { (mask.iter().map(|&m| if m { 0 } else { usize::MAX }).collect::<Vec<_>>(), 1) } else {
            let (i, r) = util::group_ids(ds, &by, &mask);
            (i, r.len())
        };
        // members of each group sorted by key
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for i in 0..n {
            if ids[i] != usize::MAX && key.as_ref().is_none_or(|k| !is_missing(k[i])) {
                members[ids[i]].push(i);
            }
        }
        if let Some(k) = &key {
            for m in members.iter_mut() {
                m.sort_by(|&a, &b| k[a].total_cmp(&k[b]).then(a.cmp(&b)));
            }
        }
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for (stat, _, src) in &targets {
            let x = util::col(ds, src)?;
            let mut res = vec![SYSMISS; n];
            for i in 0..n {
                if ids[i] == usize::MAX || key.as_ref().is_some_and(|k| is_missing(k[i])) {
                    continue;
                }
                let m = &members[ids[i]];
                let (a, b) = match &key {
                    Some(k) => {
                        let (l, h) = (k[i] + lo[i], k[i] + hi[i]);
                        (m.partition_point(|&j| k[j] < l), m.partition_point(|&j| k[j] <= h))
                    }
                    None => (0, m.len()),
                };
                let mut vals = Vec::new();
                let mut all = Vec::new();
                for &j in &m[a..b] {
                    if excl && j == i {
                        continue;
                    }
                    all.push(x[j]);
                    if !is_missing(x[j]) {
                        vals.push((x[j], 1.0));
                    }
                }
                res[i] = gstat(stat, &mut vals, &all, None);
            }
            cols.push(res);
        }
        for ((_, name, src), col) in targets.iter().zip(cols) {
            let mut v = util::new_double(name, col);
            v.label = ds.get_var(src).unwrap().label.clone();
            ds.add_variable(v)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- carryforward ----------------------------------------------------------------------

pub struct Carryforward;

impl CommandHandler for Carryforward {
    fn name(&self) -> &'static str {
        "carryforward"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let genv: Option<Vec<String>> = cmd.opt("generate", 3).map(|s| s.split_whitespace().map(|x| x.to_string()).collect());
        if let Some(g) = &genv {
            if g.len() != vars.len() {
                return Err(rc_err(198, "gen() must contain one new variable name per variable"));
            }
            for x in g {
                if ds.var_index.contains_key(x) {
                    return Err(rc_err(110, format!("variable {x} already defined")));
                }
            }
        }
        let back = cmd.has_opt("back", 4);
        let ind = cmd.opt("cfindicator", 2).map(|s| s.trim().to_string());
        let mask = util::touse(ds, st, cmd)?;
        let n = ds.n_obs;
        let gid: Vec<usize> = match &st.by_groups {
            Some(g) => g.start.clone(),
            None => vec![0; n],
        };
        let mut ranges: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            if mask[i] {
                if i > 0 && ranges.last().is_some() && gid[i] == gid[*ranges.last().unwrap().last().unwrap()] {
                    ranges.last_mut().unwrap().push(i);
                } else {
                    ranges.push(vec![i]);
                }
            }
        }
        for (k, v) in vars.iter().enumerate() {
            let mut var = ds.get_var(v).unwrap().clone();
            let mut flag = vec![0.0; n];
            for r in &ranges {
                let order: Vec<usize> = if back { r.iter().rev().copied().collect() } else { r.clone() };
                let mut last: Option<Value> = None;
                for &i in &order {
                    if is_miss_at(&var, i) {
                        if let Some(l) = &last {
                            var.data[i] = l.clone();
                            flag[i] = 1.0;
                        }
                    } else {
                        last = Some(var.data[i].clone());
                    }
                }
            }
            if let Some(name) = &ind {
                let nm = if vars.len() == 1 { name.clone() } else { format!("{name}{v}") };
                let mut iv = Variable::new(&nm, VarType::Byte);
                iv.data = flag.iter().map(|&f| Value::Byte(f as i8)).collect();
                ds.add_variable(iv)?;
            }
            match &genv {
                Some(g) => {
                    var.name = g[k].clone();
                    ds.add_variable(var)?;
                }
                None => {
                    let pos = ds.var_index[v];
                    ds.vars[pos].data = var.data;
                }
            }
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- labutil -----------------------------------------------------------------------------

pub struct Labmask;

impl CommandHandler for Labmask {
    fn name(&self) -> &'static str {
        "labmask"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let src = varlist::resolve_name(ds, cmd.opt("values", 3).ok_or_else(|| rc_err(198, "option values() required"))?.trim())?;
        if ds.get_var(&v).unwrap().var_type.is_string() {
            return Err(rc_err(109, "labmask requires a numeric variable"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let decode = cmd.has_opt("decode", 3);
        let name = cmd.opt("lblname", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| v.clone());
        let sv = ds.get_var(&src).unwrap();
        let mut lab = ValueLabel::new(&name);
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            let x = ds.get_var(&v).unwrap().num(i);
            if is_missing(x) || lab.lookup(x).is_some() {
                continue;
            }
            lab.add(x, util::cell_text(ds, sv, i, decode));
        }
        ds.value_labels.insert(name.clone(), lab);
        ds.get_var_mut(&v).unwrap().value_label_name = name;
        Ok(String::new())
    }
}

pub struct Labvalch;

fn numlist(s: &str) -> Result<Vec<f64>, String> {
    let mut o = Vec::new();
    for w in s.split_whitespace() {
        if let Some((a, b)) = w.split_once('/') {
            let (a, b): (i64, i64) = (a.parse().map_err(|_| rc_err(121, "invalid numlist"))?, b.parse().map_err(|_| rc_err(121, "invalid numlist"))?);
            o.extend((a..=b).map(|x| x as f64));
        } else {
            o.push(w.parse::<f64>().map_err(|_| rc_err(121, "invalid numlist"))?);
        }
    }
    Ok(o)
}

impl CommandHandler for Labvalch {
    fn name(&self) -> &'static str {
        "labvalch"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = cmd.main.trim().to_string();
        let lab = ds.value_labels.get_mut(&name).ok_or_else(|| rc_err(111, format!("value label {name} not found")))?;
        if let Some(d) = cmd.opt("delete", 3) {
            for x in numlist(d)? {
                lab.remove(x);
            }
        }
        if cmd.has_opt("swap", 2) {
            let old = lab.clone();
            lab.mapping.clear();
            for (k, t) in &old.mapping {
                lab.add(-k.0, t.clone());
            }
        }
        let from = cmd.opt("from", 1).or_else(|| cmd.opt("f", 1));
        let to = cmd.opt("to", 1).or_else(|| cmd.opt("t", 1));
        if let (Some(f), Some(t)) = (from, to) {
            let (f, t) = (numlist(f)?, numlist(t)?);
            if f.len() != t.len() {
                return Err(rc_err(198, "from() and to() must have the same number of values"));
            }
            let old = lab.clone();
            for (a, _) in f.iter().zip(&t) {
                lab.remove(*a);
            }
            for (a, b) in f.iter().zip(&t) {
                match old.lookup(*a) {
                    Some(txt) => lab.add(*b, txt.to_string()),
                    None => return Err(rc_err(198, format!("value {a} not labelled in {name}"))),
                }
            }
        } else if from.is_some() != to.is_some() {
            return Err(rc_err(198, "from() and to() must be specified together"));
        }
        Ok(String::new())
    }
}

pub struct Labvalclone;

impl CommandHandler for Labvalclone {
    fn name(&self) -> &'static str {
        "labvalclone"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let w: Vec<&str> = cmd.main.split_whitespace().collect();
        if w.len() != 2 {
            return Err(rc_err(198, "labvalclone oldlabel newlabel"));
        }
        let old = ds.value_labels.get(w[0]).cloned().ok_or_else(|| rc_err(111, format!("value label {} not found", w[0])))?;
        if ds.value_labels.contains_key(w[1]) && !cmd.has_opt("replace", 3) {
            return Err(rc_err(110, format!("value label {} already defined", w[1])));
        }
        let mut l = old;
        l.name = w[1].to_string();
        ds.value_labels.insert(w[1].to_string(), l);
        Ok(String::new())
    }
}

pub struct Labvarch;

impl CommandHandler for Labvarch {
    fn name(&self) -> &'static str {
        "labvarch"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let pre = cmd.opt("prefix", 3).map(util::unquote).unwrap_or_default();
        let suf = cmd.opt("suffix", 3).map(util::unquote).unwrap_or_default();
        let _ = &vars;
        for v in vars {
            let var = ds.get_var_mut(&v).unwrap();
            let base = if cmd.has_opt("usenames", 3) || var.label.is_empty() { v.clone() } else { var.label.clone() };
            var.label = format!("{pre}{base}{suf}");
        }
        Ok(String::new())
    }
}
