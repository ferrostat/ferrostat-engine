//! Business calendars: `.stbcal` files that list which days are business
//! days, the `%tb<calname>` display formats, the `bofd()` / `dofb()`
//! functions and the `bcal` command (`create`, `check`, `describe`, `load`,
//! `dir`).
//!
//! Calendar file directives (one per line, `*` and `//` start comments):
//! `version #`, `purpose "text"`, `dateformat ymd|mdy|dmy`,
//! `range date date`, `centerdate date`, `omit dayofweek (Sa Su)`,
//! `omit date date [and +#|-# [if dow(...)]]` (a `*` year means every year
//! in the range) and `omit dowinmonth +#|-# dow of month` (the #th, or #th
//! from last, weekday of a month; `*` month means every month).
//!
//! Business date 0 is the center date; the other business days are numbered
//! consecutively before and after it.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::{vfs, DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dates;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

#[derive(Debug, Clone)]
pub struct Calendar {
    pub name: String,
    pub purpose: String,
    pub dateformat: String,
    pub lo: i64,
    pub hi: i64,
    pub center: i64,
    /// Business days (%td), ascending.
    pub days: Vec<i64>,
    /// Index of the center date in `days`.
    pub center_idx: usize,
    pub path: String,
}

impl Calendar {
    pub fn bofd(&self, td: f64) -> f64 {
        if td >= SYSMISS || td.fract() != 0.0 {
            return SYSMISS;
        }
        match self.days.binary_search(&(td as i64)) {
            Ok(i) => i as f64 - self.center_idx as f64,
            Err(_) => SYSMISS,
        }
    }

    pub fn dofb(&self, b: f64) -> f64 {
        if b >= SYSMISS || b.fract() != 0.0 {
            return SYSMISS;
        }
        let i = b as i64 + self.center_idx as i64;
        if i < 0 || i as usize >= self.days.len() {
            return SYSMISS;
        }
        self.days[i as usize] as f64
    }
}

thread_local! {
    static CALENDARS: RefCell<HashMap<String, Calendar>> = RefCell::new(HashMap::new());
}

fn file_of(name: &str) -> String {
    if name.ends_with(".stbcal") {
        name.to_string()
    } else {
        format!("{name}.stbcal")
    }
}

fn find_file(name: &str) -> Option<String> {
    let f = file_of(name);
    for dir in ["", "bcal/"] {
        let p = format!("{dir}{f}");
        if vfs::exists(&p) {
            return Some(p);
        }
    }
    None
}

/// Calendar `name`, loading it from `name.stbcal` on first use.
pub fn get(name: &str) -> Result<Calendar, String> {
    if let Some(c) = CALENDARS.with(|m| m.borrow().get(name).cloned()) {
        return Ok(c);
    }
    load(name)
}

/// (Re)load calendar `name` from disk.
pub fn load(name: &str) -> Result<Calendar, String> {
    let path = find_file(name).ok_or_else(|| rc_err(601, format!("business calendar {name} not found; file {} not found", file_of(name))))?;
    let text = vfs::read_to_string(&path).map_err(|e| rc_err(603, format!("{path}: {e}")))?;
    let base = name.trim_end_matches(".stbcal").rsplit('/').next().unwrap_or(name).to_string();
    let mut cal = parse(&base, &text).map_err(|e| rc_err(198, format!("business calendar {base} ({path}): {e}")))?;
    cal.path = path;
    CALENDARS.with(|m| m.borrow_mut().insert(base, cal.clone()));
    Ok(cal)
}

const DOW: [&str; 7] = ["su", "mo", "tu", "we", "th", "fr", "sa"];
const MON: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

fn dow_of(w: &str) -> Option<i64> {
    let l = w.to_lowercase();
    DOW.iter().position(|d| l.len() >= 2 && l.starts_with(d)).map(|i| i as i64)
}

fn month_of(w: &str) -> Option<i64> {
    let l = w.to_lowercase();
    if let Ok(n) = l.parse::<i64>() {
        return (1..=12).contains(&n).then_some(n);
    }
    MON.iter().position(|m| l.len() >= 3 && l.starts_with(m)).map(|i| i as i64 + 1)
}

fn mask(fmt: &str) -> String {
    fmt.to_uppercase()
}

fn parse_date(s: &str, fmt: &str) -> Result<i64, String> {
    let d = dates::date(s, &mask(fmt), None);
    if d >= SYSMISS {
        return Err(format!("invalid date \"{s}\" (dateformat {fmt})"));
    }
    Ok(d as i64)
}

/// Dates matched by an `omit date` pattern within [lo, hi]; `*` stands for
/// every year.
fn pattern_dates(pat: &str, fmt: &str, lo: i64, hi: i64) -> Result<Vec<i64>, String> {
    if !pat.contains('*') {
        return Ok(vec![parse_date(pat, fmt)?]);
    }
    let (y0, y1) = (year_of(lo)?, year_of(hi)?);
    let mut out = Vec::new();
    for y in y0..=y1 {
        let d = dates::date(&pat.replace('*', &y.to_string()), &mask(fmt), None);
        if d < SYSMISS {
            out.push(d as i64);
        }
    }
    Ok(out)
}

fn strip_comment(l: &str) -> &str {
    let l = l.trim();
    if l.starts_with('*') {
        return "";
    }
    match l.find("//") {
        Some(i) => l[..i].trim(),
        None => l,
    }
}

/// Parse calendar text.
pub fn parse(name: &str, text: &str) -> Result<Calendar, String> {
    let mut purpose = String::new();
    let mut fmt = "ymd".to_string();
    let mut range = None;
    let mut center = None;
    let mut omit_dow: BTreeSet<i64> = BTreeSet::new();
    let mut omits: Vec<Vec<String>> = Vec::new();
    for raw in text.lines() {
        let l = strip_comment(raw);
        if l.is_empty() {
            continue;
        }
        let words: Vec<String> = l.split_whitespace().map(String::from).collect();
        match words[0].to_lowercase().as_str() {
            "version" => {}
            "purpose" => purpose = util::unquote(l[7..].trim()),
            "dateformat" => {
                let f = words.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
                if !matches!(f.as_str(), "ymd" | "mdy" | "dmy") {
                    return Err(format!("dateformat {f} invalid"));
                }
                fmt = f;
            }
            "range" => {
                if words.len() != 3 {
                    return Err("range requires two dates".into());
                }
                range = Some((words[1].clone(), words[2].clone()));
            }
            "centerdate" => center = Some(words.get(1).cloned().ok_or("centerdate requires a date")?),
            "omit" => {
                if words.get(1).is_some_and(|w| w.eq_ignore_ascii_case("dayofweek")) {
                    let rest = l.split_once(char::is_whitespace).unwrap().1.trim()[9..].trim();
                    for w in rest.trim_start_matches('(').trim_end_matches(')').split_whitespace() {
                        omit_dow.insert(dow_of(w).ok_or_else(|| format!("{w}: invalid day of week"))?);
                    }
                } else {
                    omits.push(words[1..].to_vec());
                }
            }
            other => return Err(format!("{other}: unknown directive")),
        }
    }
    let (a, b) = range.ok_or("range not specified")?;
    let (lo, hi) = (parse_date(&a, &fmt)?, parse_date(&b, &fmt)?);
    if hi < lo {
        return Err("range: end date precedes start date".into());
    }
    let mut omitted: BTreeSet<i64> = (lo..=hi).filter(|d| omit_dow.contains(&(dates::dow(*d as f64) as i64))).collect();
    for w in &omits {
        match w.first().map(|s| s.to_lowercase()).as_deref() {
            Some("date") => {
                let pat = w.get(1).ok_or("omit date requires a date")?;
                let base = pattern_dates(pat, &fmt, lo, hi)?;
                // optional `and +#|-# [if dow(...)]`
                let mut extra: Vec<(i64, Option<Vec<i64>>)> = Vec::new();
                let mut i = 2;
                while i < w.len() {
                    if w[i].eq_ignore_ascii_case("and") {
                        let off: i64 = w.get(i + 1).ok_or("and requires an offset")?.trim_matches(|c| c == '(' || c == ')').parse().map_err(|_| "invalid offset after and".to_string())?;
                        let mut cond = None;
                        if w.get(i + 2).is_some_and(|s| s.eq_ignore_ascii_case("if")) {
                            let spec = w[i + 3..].join(" ");
                            let inside = spec.trim().trim_start_matches("dow").trim().trim_start_matches('(');
                            let inside = inside.split(')').next().unwrap_or("");
                            cond = Some(inside.split_whitespace().map(|d| dow_of(d).ok_or_else(|| format!("{d}: invalid day of week"))).collect::<Result<Vec<_>, _>>()?);
                            i = w.len();
                        } else {
                            i += 2;
                        }
                        extra.push((off, cond));
                    } else {
                        return Err(format!("omit date: unexpected {}", w[i]));
                    }
                }
                for d in base {
                    omitted.insert(d);
                    for (off, cond) in &extra {
                        if cond.as_ref().is_none_or(|c| c.contains(&(dates::dow(d as f64) as i64))) {
                            omitted.insert(d + off);
                        }
                    }
                }
            }
            Some("dowinmonth") => {
                // omit dowinmonth +1 mo of jan
                if w.len() < 5 || !w[3].eq_ignore_ascii_case("of") {
                    return Err("omit dowinmonth #th dow of month".into());
                }
                let k: i64 = w[1].trim_start_matches('+').parse().map_err(|_| format!("{}: invalid number", w[1]))?;
                let dw = dow_of(&w[2]).ok_or_else(|| format!("{}: invalid day of week", w[2]))?;
                let months: Vec<i64> = if w[4] == "*" { (1..=12).collect() } else { vec![month_of(&w[4]).ok_or_else(|| format!("{}: invalid month", w[4]))?] };
                let (y0, y1) = (year_of(lo)?, year_of(hi)?);
                for y in y0..=y1 {
                    for &m in &months {
                        let first = dates::mdy(m as f64, 1.0, y as f64) as i64;
                        let last = if m == 12 { dates::mdy(1.0, 1.0, (y + 1) as f64) as i64 - 1 } else { dates::mdy((m + 1) as f64, 1.0, y as f64) as i64 - 1 };
                        let hits: Vec<i64> = (first..=last).filter(|d| dates::dow(*d as f64) as i64 == dw).collect();
                        let pick = if k > 0 { hits.get((k - 1) as usize) } else if k < 0 { hits.len().checked_sub((-k) as usize).and_then(|i| hits.get(i)) } else { None };
                        if let Some(d) = pick {
                            omitted.insert(*d);
                        }
                    }
                }
            }
            _ => return Err(format!("omit {}: invalid", w.join(" "))),
        }
    }
    let days: Vec<i64> = (lo..=hi).filter(|d| !omitted.contains(d)).collect();
    if days.is_empty() {
        return Err("calendar has no business days".into());
    }
    let center = match center {
        Some(c) => parse_date(&c, &fmt)?,
        None => days[0],
    };
    let center_idx = days.binary_search(&center).map_err(|_| "centerdate is not a business day in the range".to_string())?;
    Ok(Calendar { name: name.to_string(), purpose, dateformat: fmt, lo, hi, center, days, center_idx, path: String::new() })
}

/// Display a business date with `%tb<name>[:td-spec]`.
pub fn format_b(v: f64, spec: &str) -> String {
    let (name, td_spec) = match spec.split_once(':') {
        Some((a, b)) => (a, b),
        None => (spec, ""),
    };
    match get(name) {
        Ok(c) => {
            let td = c.dofb(v.floor());
            if td >= SYSMISS {
                ".".into()
            } else {
                dates::format_date(td, &format!("%td{td_spec}"))
            }
        }
        Err(_) => format!("{v}"),
    }
}

/// `bofd("cal", td)` and `dofb(b, "cal")`.
pub fn function(name: &str, cal: &str, x: f64) -> Result<f64, String> {
    let c = get(cal.trim())?;
    Ok(if name == "bofd" { c.bofd(x) } else { c.dofb(x) })
}

fn fmt_td(d: i64) -> String {
    dates::format_date(d as f64, "%td")
}

/// Calendar year of a `%td` date, or an error for an unrepresentable date.
fn year_of(d: i64) -> Result<i64, String> {
    dates::ymd(d as f64).map(|x| x.0).ok_or_else(|| format!("{d}: date out of range"))
}

fn date_text(d: i64, fmt: &str) -> String {
    let Some((y, m, dd)) = dates::ymd(d as f64) else { return d.to_string() };
    match fmt {
        "mdy" => format!("{}{:02}{y}", MON[(m - 1) as usize], dd),
        "dmy" => format!("{dd:02}{}{y}", MON[(m - 1) as usize]),
        _ => format!("{y}{}{dd:02}", MON[(m - 1) as usize]),
    }
}

pub struct Bcal;

impl CommandHandler for Bcal {
    fn name(&self) -> &'static str {
        "bcal"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let rest = rest.trim();
        match sub {
            "dir" => {
                let pat = if rest.is_empty() { "*" } else { rest };
                let mut names: BTreeSet<String> = BTreeSet::new();
                for dir in [".", "bcal"] {
                    if let Ok(list) = vfs::list(dir) {
                        for (f, is_dir) in list {
                            let f = f.rsplit('/').next().unwrap_or(&f).to_string();
                            if let Some(n) = f.strip_suffix(".stbcal") {
                                if !is_dir && crate::data_utils::glob(pat, n) {
                                    names.insert(n.to_string());
                                }
                            }
                        }
                    }
                }
                CALENDARS.with(|m| names.extend(m.borrow().keys().filter(|n| crate::data_utils::glob(pat, n)).cloned()));
                let list: Vec<String> = names.into_iter().collect();
                st.begin_r().set_str("calendars", list.join(" "));
                if list.is_empty() {
                    return Ok("  (no business calendars found)".into());
                }
                Ok(list.iter().map(|n| format!("  {n:<20} {}", find_file(n).unwrap_or_else(|| "(in memory)".into()))).collect::<Vec<_>>().join("\n"))
            }
            "load" => {
                let name = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "calendar name required"))?.trim_start_matches("%tb");
                let c = load(name)?;
                Ok(format!("  (calendar {} loaded from {})", c.name, c.path))
            }
            "check" => {
                let names: Vec<String> = if rest.is_empty() {
                    let mut v: Vec<String> = Vec::new();
                    for var in &ds.vars {
                        if let Some(n) = var.format.strip_prefix("%tb").or_else(|| var.format.strip_prefix("%-tb")) {
                            let n = n.split(':').next().unwrap_or(n).to_string();
                            if !v.contains(&n) {
                                v.push(n);
                            }
                        }
                    }
                    v
                } else {
                    rest.split_whitespace().map(|s| s.trim_start_matches("%tb").to_string()).collect()
                };
                let mut out = Vec::new();
                let mut bad = 0;
                for n in &names {
                    match load(n) {
                        Ok(c) => out.push(format!("  %tb{n}:  defined in {}, loaded ({} business days)", c.path, c.days.len())),
                        Err(e) => {
                            bad += 1;
                            out.push(format!("  %tb{n}:  {}", e.splitn(2, '|').last().unwrap_or(&e)));
                        }
                    }
                }
                if names.is_empty() {
                    out.push("  (no business calendars in use)".into());
                }
                if bad > 0 && !cmd.has_opt("rc0", 3) {
                    return Err(rc_err(459, out.join("\n")));
                }
                Ok(out.join("\n"))
            }
            "describe" | "des" => {
                let name = rest.split_whitespace().next().ok_or_else(|| rc_err(198, "calendar name required"))?.trim_start_matches("%tb");
                let c = get(name)?;
                let omitted = (c.hi - c.lo + 1) as usize - c.days.len();
                let r = st.begin_r();
                r.set_str("name", &c.name);
                r.set_str("purpose", &c.purpose);
                r.set_str("fn", &c.path);
                r.set("min_date_td", c.lo as f64);
                r.set("max_date_td", c.hi as f64);
                r.set("ctr_date_td", c.center as f64);
                r.set("min_date_tb", -(c.center_idx as f64));
                r.set("max_date_tb", (c.days.len() - 1 - c.center_idx) as f64);
                r.set("omitted", omitted as f64);
                r.set("included", c.days.len() as f64);
                Ok([
                    format!("  Business calendar {} (format %tb{}):", c.name, c.name),
                    format!("    purpose:  {}", c.purpose),
                    format!("      range:  {}  {}", fmt_td(c.lo), fmt_td(c.hi)),
                    format!("              {:>9}  {:>9}   in %td units", c.lo, c.hi),
                    format!("              {:>9}  {:>9}   in %tb units", -(c.center_idx as i64), c.days.len() as i64 - 1 - c.center_idx as i64),
                    format!("     center:  {}", fmt_td(c.center)),
                    format!("              {:>9}   in %td units", c.center),
                    format!("              {:>9}   in %tb units", 0),
                    format!("    omitted:  {:>9}   days", omitted),
                    format!("   included:  {:>9}   days", c.days.len()),
                ]
                .join("\n"))
            }
            "create" => create(ds, st, cmd, rest),
            "" => Err(rc_err(198, "bcal: subcommand required")),
            o => Err(rc_err(198, format!("bcal {o}: invalid subcommand"))),
        }
    }
}

/// `bcal create filename, from(varname) [generate(newvar) purpose() dateformat()
/// range(d1 d2) centerdate() excludemissing(varlist) replace maxgap(#)]`
fn create(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let fname = util::unquote(rest);
    if fname.is_empty() {
        return Err(rc_err(198, "filename required"));
    }
    let path = file_of(&fname);
    let name = path.trim_end_matches(".stbcal").rsplit('/').next().unwrap_or("").to_string();
    varlist::check_name(&name)?;
    if vfs::exists(&path) && !cmd.has_opt("replace", 3) {
        return Err(rc_err(602, format!("file {path} already exists")));
    }
    let from = varlist::resolve_name(ds, cmd.opt("from", 4).ok_or_else(|| rc_err(198, "option from() required"))?.trim())?;
    let mut mask = util::touse(ds, st, cmd)?;
    if let Some(ex) = cmd.opt("excludemissing", 3) {
        let (list, any) = match ex.split_once(',') {
            Some((a, b)) => (a, b.trim() == "any"),
            None => (ex, false),
        };
        let vars = varlist::expand(ds, list)?;
        for (i, m) in mask.iter_mut().enumerate() {
            let miss: Vec<bool> = vars.iter().map(|v| ds.get_var(v).unwrap().get(i).is_none_or(|x| x.is_missing())).collect();
            if if any { miss.iter().any(|&b| b) } else { miss.iter().all(|&b| b) } {
                *m = false;
            }
        }
    }
    let v = ds.get_var(&from).unwrap();
    let present: BTreeSet<i64> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| v.num(i)).filter(|x| *x < SYSMISS).map(|x| x.floor() as i64).collect();
    if present.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let fmt = cmd.opt("dateformat", 5).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "ymd".into());
    if !matches!(fmt.as_str(), "ymd" | "mdy" | "dmy") {
        return Err(rc_err(198, "dateformat() must be ymd, mdy or dmy"));
    }
    let (lo, hi) = match cmd.opt("range", 1) {
        Some(r) => {
            let w: Vec<&str> = r.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "range() requires two dates"));
            }
            let p = |s: &str| parse_date(s, &fmt).map_err(|e| rc_err(198, e));
            (p(w[0])?, p(w[1])?)
        }
        None => (*present.iter().next().unwrap(), *present.iter().last().unwrap()),
    };
    let maxgap = util::opt_num(cmd, "maxgap", 4)?.unwrap_or(10.0) as i64;
    let inside: Vec<i64> = present.iter().copied().filter(|d| (lo..=hi).contains(d)).collect();
    if let Some(g) = inside.windows(2).map(|w| w[1] - w[0] - 1).max() {
        if g > maxgap {
            return Err(rc_err(198, format!("gap of {g} days between business days exceeds maxgap({maxgap})")));
        }
    }
    let span = hi - lo + 1;
    let mut dow_seen = [false; 7];
    for d in &inside {
        dow_seen[dates::dow(*d as f64) as usize] = true;
    }
    let omit_dow: Vec<usize> = if span >= 14 { (0..7).filter(|&i| !dow_seen[i]).collect() } else { Vec::new() };
    let center = match cmd.opt("centerdate", 3) {
        Some(c) => parse_date(c.trim(), &fmt).map_err(|e| rc_err(198, e))?,
        None => inside[0],
    };
    let mut text = vec![
        format!("* Business calendar {name}, created by bcal create from variable {from}"),
        "version 17".to_string(),
        format!("purpose \"{}\"", cmd.opt("purpose", 1).map(util::unquote).unwrap_or_default()),
        format!("dateformat {fmt}"),
        String::new(),
        format!("range {} {}", date_text(lo, &fmt), date_text(hi, &fmt)),
        format!("centerdate {}", date_text(center, &fmt)),
        String::new(),
    ];
    if !omit_dow.is_empty() {
        let names: Vec<String> = omit_dow.iter().map(|&i| format!("{}{}", DOW[i][..1].to_uppercase(), &DOW[i][1..])).collect();
        text.push(format!("omit dayofweek ({})", names.join(" ")));
    }
    let set: BTreeSet<i64> = inside.iter().copied().collect();
    for d in lo..=hi {
        if !set.contains(&d) && !omit_dow.contains(&(dates::dow(d as f64) as usize)) {
            text.push(format!("omit date {}", date_text(d, &fmt)));
        }
    }
    let body = text.join("\n") + "\n";
    let cal = parse(&name, &body).map_err(|e| rc_err(198, e))?;
    vfs::write(&path, body.as_bytes()).map_err(|e| rc_err(603, format!("file {path} could not be opened ({e})")))?;
    CALENDARS.with(|m| m.borrow_mut().insert(name.clone(), Calendar { path: path.clone(), ..cal.clone() }));
    let mut out = vec![format!("  Business calendar {name} created; file {path} saved"), format!("  ({} business days, {} days omitted)", cal.days.len(), span as usize - cal.days.len())];
    if let Some(g) = cmd.opt("generate", 3) {
        let (gname, replace) = match g.split_once(',') {
            Some((a, b)) => (a.trim(), b.trim().starts_with("replace")),
            None => (g.trim(), false),
        };
        if replace && ds.get_var(gname).is_some() {
            ds.drop_variable(gname)?;
        }
        varlist::new_names(ds, gname)?;
        let src = ds.get_var(&from).unwrap();
        let mut nv = Variable::new(gname, VarType::Long);
        nv.format = format!("%tb{name}");
        for i in 0..ds.n_obs {
            let b = cal.bofd(src.num(i).floor());
            nv.push(if b >= SYSMISS { Value::Missing } else { Value::Long(b as i32) });
        }
        ds.add_variable(nv)?;
        st.data_changed = true;
        out.push(format!("  variable {gname} created"));
    }
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekday_calendar() {
        let text = "version 17\npurpose \"weekdays\"\ndateformat ymd\nrange 2020-01-01 2020-01-31\ncenterdate 2020-01-02\nomit dayofweek (Sa Su)\nomit date 2020-01-20\n";
        let c = parse("wk", text).unwrap();
        // January 2020: 23 weekdays, minus one holiday
        assert_eq!(c.days.len(), 22);
        let d2 = dates::mdy(1.0, 2.0, 2020.0);
        assert_eq!(c.bofd(d2), 0.0);
        assert_eq!(c.bofd(d2 - 1.0), -1.0);
        // Fri 3 Jan is 1, Mon 6 Jan is 2
        assert_eq!(c.bofd(d2 + 4.0), 2.0);
        assert_eq!(c.bofd(d2 + 2.0), SYSMISS);
        assert_eq!(c.dofb(2.0), d2 + 4.0);
        let c2 = parse("x", "range 2020-01-01 2020-12-31\nomit dowinmonth +1 mo of *\nomit dowinmonth -1 fr of dec\n").unwrap();
        assert_eq!(c2.days.len(), 366 - 12 - 1);
    }
}
