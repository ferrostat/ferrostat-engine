//! Display settings of coefficient tables: `set cformat`, `set pformat`,
//! `set sformat`, `set showbaselevels` and `set emptycells`.
//!
//! The engine copies the settings into this thread's state before every
//! command (`sync`), so table printers that do not see the store can use
//! them.

use std::cell::RefCell;

use crate::eval::rc_err;
use crate::format::{fmt, fmt_g, NumFormat};
use crate::store::Store;

#[derive(Clone, Default)]
struct Display {
    cformat: Option<String>,
    pformat: Option<String>,
    sformat: Option<String>,
    /// "", "on", "off" or "all"
    showbase: String,
    /// coefficient names of empty interaction cells of the last design
    empty: Vec<String>,
}

thread_local! {
    static DISPLAY: RefCell<Display> = RefCell::new(Display::default());
}

/// Copy the display settings of `st` (called before each command).
pub fn sync(st: &Store) {
    let get = |k: &str| st.settings.get(k).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    DISPLAY.with(|d| {
        let mut d = d.borrow_mut();
        d.cformat = get("cformat");
        d.pformat = get("pformat");
        d.sformat = get("sformat");
        d.showbase = get("showbaselevels").unwrap_or_default();
    });
}

/// Validate `set cformat|pformat|sformat fmt` (empty resets); the widths
/// are limited so the table columns keep their alignment.
pub fn check_format(what: &str, v: &str) -> Result<(), String> {
    let v = v.trim();
    if v.is_empty() {
        return Ok(());
    }
    let max = match what {
        "cformat" => 9,
        "pformat" => 5,
        _ => 8,
    };
    let f = NumFormat::parse(v).ok_or_else(|| rc_err(120, format!("invalid %format {v}")))?;
    let w = v.trim_start_matches('%').trim_start_matches('-').trim_start_matches('0');
    let width: usize = w.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
    let _ = f;
    if width > max {
        return Err(rc_err(198, format!("invalid {what}: width may not exceed {max}")));
    }
    if v.contains('s') || v.contains('t') {
        return Err(rc_err(120, format!("invalid numeric format {v}")));
    }
    Ok(())
}

/// Validate `set showbaselevels` / `set emptycells` values.
pub fn check_choice(what: &str, v: &str) -> Result<(), String> {
    let v = v.trim();
    let ok = match what {
        "showbaselevels" => matches!(v, "" | "on" | "off" | "all"),
        "emptycells" => matches!(v, "" | "keep" | "drop"),
        _ => true,
    };
    if ok { Ok(()) } else { Err(rc_err(198, format!("set {what} {v} not allowed"))) }
}

/// A coefficient, standard error or confidence limit (`set cformat`).
pub fn coef(v: f64) -> String {
    DISPLAY.with(|d| match &d.borrow().cformat {
        Some(f) => fmt(v, f).trim().to_string(),
        None => fmt_g(v, 8),
    })
}

/// A p-value, 5 columns (`set pformat`).
pub fn pvalue(p: f64) -> String {
    DISPLAY.with(|d| match &d.borrow().pformat {
        Some(f) => format!("{:>5}", fmt(p, f).trim()),
        None => format!("{p:.3}"),
    })
}

/// A test statistic, 7 columns (`set sformat`).
pub fn stat(t: f64) -> String {
    DISPLAY.with(|d| match &d.borrow().sformat {
        Some(f) => format!("{:>7}", fmt(t, f).trim()),
        None => format!("{t:>7.2}"),
    })
}

/// Are base levels shown in coefficient tables?
pub fn show_base() -> bool {
    DISPLAY.with(|d| matches!(d.borrow().showbase.as_str(), "on" | "all"))
}

/// Is `name` an empty interaction cell of the last design?
pub fn is_empty_cell(name: &str) -> bool {
    DISPLAY.with(|d| d.borrow().empty.iter().any(|n| n == name))
}

/// Interaction cells with no observations in the estimation sample: with
/// `set emptycells drop` they are removed from the design; otherwise
/// (`keep`, the default) they stay as omitted coefficients shown as
/// "(empty)".
pub fn empty_cells(st: &Store, d: &mut crate::est::Design) {
    let drop = st.settings.get("emptycells").is_some_and(|v| v.trim() == "drop");
    let mut empty = Vec::new();
    let mut j = 0;
    while j < d.names.len() {
        let is_cell = d.names[j].contains('#') && d.names[j].split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().is_ok()));
        if is_cell && !d.is_base[j] && d.cols[j].iter().all(|&x| x == 0.0) {
            if drop {
                d.names.remove(j);
                d.cols.remove(j);
                d.omitted.remove(j);
                d.is_base.remove(j);
                continue;
            }
            d.omitted[j] = true;
            let n = &d.names[j];
            // the omitted marker on the first level, as for collinear terms
            let renamed = match n.split_once('.') {
                Some((l, v)) if l.chars().all(|c| c.is_ascii_digit()) => format!("{l}o.{v}"),
                _ => n.clone(),
            };
            d.names[j] = renamed.clone();
            empty.push(renamed);
        }
        j += 1;
    }
    DISPLAY.with(|x| x.borrow_mut().empty = empty);
}
