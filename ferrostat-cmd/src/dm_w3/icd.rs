//! `icd9`, `icd9p`, `icd10`, `icd10cm`, `icd10pcs`: work with diagnosis and
//! procedure codes stored as strings.
//!
//! Everything here is based on the *structure* of the codes only: which
//! characters may appear where, where the period goes, how long a code can
//! be and how many leading characters form its category. No code-description
//! tables are distributed, so `check` validates the format of the codes,
//! `generate ..., description` produces empty strings and `lookup`/`search`
//! say that descriptions are unavailable.

use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::util;
use crate::varlist;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    /// ICD-9-CM diagnosis codes
    Icd9,
    /// ICD-9-CM procedure codes
    Icd9p,
    /// ICD-10 (international edition) diagnosis codes
    Icd10,
    /// ICD-10-CM diagnosis codes
    Icd10cm,
    /// ICD-10-PCS procedure codes
    Icd10pcs,
}

impl Kind {
    fn verb(self) -> &'static str {
        match self {
            Kind::Icd9 => "icd9",
            Kind::Icd9p => "icd9p",
            Kind::Icd10 => "icd10",
            Kind::Icd10cm => "icd10cm",
            Kind::Icd10pcs => "icd10pcs",
        }
    }

    /// Number of leading characters (undotted) after which the period goes,
    /// which is also the length of the category.
    fn head(self, code: &str) -> usize {
        match self {
            Kind::Icd9 if code.starts_with('E') => 4,
            Kind::Icd9 => 3,
            Kind::Icd9p => 2,
            Kind::Icd10 | Kind::Icd10cm | Kind::Icd10pcs => 3,
        }
    }

    fn max_len(self) -> usize {
        match self {
            Kind::Icd9 => 6,
            Kind::Icd9p => 5,
            Kind::Icd10 => 6,
            Kind::Icd10cm => 8,
            Kind::Icd10pcs => 7,
        }
    }
}

/// Reasons a code is invalid (the values of `check, generate()`).
const REASONS: [&str; 6] = [
    "",
    "invalid placement of period",
    "too many periods",
    "code too short",
    "code too long",
    "invalid character in code",
];

/// Validate one code; returns the undotted, upper-case code or the reason
/// number (1-5) why it is invalid.
pub fn parse(kind: Kind, raw: &str) -> Result<String, u8> {
    let s = raw.trim().to_uppercase();
    let dots = s.matches('.').count();
    if dots > 1 {
        return Err(2);
    }
    let code: String = s.chars().filter(|&c| c != '.').collect();
    if code.chars().any(|c| !c.is_ascii_alphanumeric()) {
        return Err(5);
    }
    if kind == Kind::Icd10pcs && dots > 0 {
        return Err(1);
    }
    if dots == 1 {
        let pos = s.find('.').unwrap();
        if pos != kind.head(&code) {
            return Err(1);
        }
    }
    let c: Vec<char> = code.chars().collect();
    let digit = |i: usize| c.get(i).is_some_and(|x| x.is_ascii_digit());
    let all_digits = |from: usize| c.iter().skip(from).all(|x| x.is_ascii_digit());
    let (min, max) = match kind {
        Kind::Icd9 => match c.first() {
            Some('E') => (4, 5),
            _ => (3, 5),
        },
        Kind::Icd9p => (2, 4),
        Kind::Icd10 => (3, 5),
        Kind::Icd10cm => (3, 7),
        Kind::Icd10pcs => (7, 7),
    };
    if c.len() < min {
        return Err(3);
    }
    if c.len() > max {
        return Err(4);
    }
    let ok = match kind {
        Kind::Icd9 => match c[0] {
            'E' | 'V' => all_digits(1),
            d if d.is_ascii_digit() => all_digits(0),
            _ => false,
        },
        Kind::Icd9p => all_digits(0),
        Kind::Icd10 => c[0].is_ascii_uppercase() && digit(1) && digit(2) && all_digits(3),
        // letter, digit, then letters or digits
        Kind::Icd10cm => c[0].is_ascii_uppercase() && digit(1),
        // 0-9 and A-Z without the letters O and I
        Kind::Icd10pcs => c.iter().all(|&x| x.is_ascii_digit() || x.is_ascii_uppercase() && x != 'O' && x != 'I'),
    };
    if ok {
        Ok(code)
    } else {
        Err(5)
    }
}

/// Code with the period inserted (undotted input, already validated).
pub fn dotted(kind: Kind, code: &str) -> String {
    let h = kind.head(code);
    if kind == Kind::Icd10pcs || code.len() <= h {
        code.to_string()
    } else {
        format!("{}.{}", &code[..h], &code[h..])
    }
}

/// Category of a valid code (undotted).
pub fn category(kind: Kind, code: &str) -> String {
    code.chars().take(kind.head(code)).collect()
}

/// One element of a code list: exact code, prefix (`code*`) or inclusive
/// range (`low/high`, where codes below `high` also match).
#[derive(Debug, Clone)]
enum Item {
    Exact(String),
    Prefix(String),
    Range(String, String),
}

fn norm(s: &str) -> String {
    s.trim().to_uppercase().chars().filter(|&c| c != '.').collect()
}

fn codelist(kind: Kind, text: &str) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    for tok in text.split_whitespace() {
        if let Some(p) = tok.strip_suffix('*') {
            items.push(Item::Prefix(norm(p)));
        } else if let Some((a, b)) = tok.split_once('/') {
            let (a, b) = (norm(a), norm(b));
            if a > b {
                return Err(rc_err(198, format!("{tok}: invalid range")));
            }
            items.push(Item::Range(a, b));
        } else {
            parse(kind, tok).map_err(|r| rc_err(198, format!("{tok}: invalid {} code ({})", kind.verb(), REASONS[r as usize])))?;
            items.push(Item::Exact(norm(tok)));
        }
    }
    if items.is_empty() {
        return Err(rc_err(198, "code list required"));
    }
    Ok(items)
}

fn in_list(items: &[Item], code: &str) -> bool {
    items.iter().any(|it| match it {
        Item::Exact(c) => code == c,
        Item::Prefix(p) => code.starts_with(p.as_str()),
        Item::Range(a, b) => code >= a.as_str() && (code <= b.as_str() || code.starts_with(b.as_str())),
    })
}

const NO_DESC: &str = "(code descriptions are not distributed with this program; only the structure of codes is used)";

pub struct Icd(pub Kind);

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("icd9", Box::new(Icd(Kind::Icd9))),
        ("icd9p", Box::new(Icd(Kind::Icd9p))),
        ("icd10", Box::new(Icd(Kind::Icd10))),
        ("icd10cm", Box::new(Icd(Kind::Icd10cm))),
        ("icd10pcs", Box::new(Icd(Kind::Icd10pcs))),
    ]
}

fn sub_is(given: &str, full: &str, min: usize) -> bool {
    given.len() >= min && full.starts_with(given)
}

/// The string variable named in `text` (exactly one).
fn code_var(ds: &DataSet, text: &str) -> Result<String, String> {
    let vars = varlist::expand(ds, text)?;
    let [v] = vars.as_slice() else {
        return Err(rc_err(103, "one variable required"));
    };
    if !ds.get_var(v).unwrap().var_type.is_string() {
        return Err(rc_err(109, format!("{v}: string variable required")));
    }
    Ok(v.clone())
}

fn str_var(name: &str, vals: Vec<String>) -> Variable {
    let w = vals.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let mut v = Variable::new(name, util::str_type_for(w));
    v.format = format!("%-{}s", w.max(9).min(2045));
    for s in vals {
        v.push(Value::Str(s));
    }
    v
}

fn byte_var(name: &str, vals: Vec<f64>) -> Variable {
    let mut v = Variable::new(name, VarType::Byte);
    v.format = "%8.0g".into();
    for x in vals {
        v.push(util::typed_value(VarType::Byte, x));
    }
    v
}

impl CommandHandler for Icd {
    fn name(&self) -> &'static str {
        self.0.verb()
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let kind = self.0;
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub = sub.to_lowercase();
        let rest = rest.trim();
        if sub_is(&sub, "check", 1) {
            check(kind, ds, st, cmd, rest)
        } else if sub_is(&sub, "clean", 2) {
            clean(kind, ds, st, cmd, rest)
        } else if sub_is(&sub, "generate", 3) {
            generate(kind, ds, st, cmd, rest)
        } else if sub_is(&sub, "lookup", 4) {
            let items = codelist(kind, rest)?;
            let mut out = Vec::new();
            for it in &items {
                let shown = match it {
                    Item::Exact(c) => dotted(kind, c),
                    Item::Prefix(p) => format!("{p}*"),
                    Item::Range(a, b) => format!("{a}/{b}"),
                };
                out.push(format!("  {shown:<10} (description not available)"));
            }
            out.push(NO_DESC.into());
            let r = st.begin_r();
            r.set("N_codes", items.len() as f64);
            Ok(out.join("\n"))
        } else if sub_is(&sub, "search", 3) {
            st.begin_r().set("N_codes", 0.0);
            Ok(format!("no codes found; {}", &NO_DESC[1..NO_DESC.len() - 1]))
        } else if sub_is(&sub, "query", 1) {
            Ok(format!("{}: format validation of codes (structure only); no code-description data available", kind.verb()))
        } else if sub.is_empty() {
            Err(rc_err(198, format!("{}: subcommand required", kind.verb())))
        } else {
            Err(rc_err(198, format!("{} {sub}: invalid subcommand", kind.verb())))
        }
    }
}

fn check(kind: Kind, ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let var = code_var(ds, rest)?;
    let mask = util::touse(ds, st, cmd)?;
    let v = ds.get_var(&var).unwrap();
    let mut counts = [0usize; 6];
    let mut miss = 0;
    let mut flags = vec![f64::NAN; ds.n_obs];
    let mut bad_rows = Vec::new();
    let mut freq: std::collections::BTreeMap<String, usize> = Default::default();
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let s = v.str_at(i);
        if s.trim().is_empty() {
            miss += 1;
            flags[i] = 0.0;
            continue;
        }
        match parse(kind, s) {
            Ok(_) => flags[i] = 0.0,
            Err(r) => {
                counts[r as usize] += 1;
                flags[i] = r as f64;
                bad_rows.push(i);
                *freq.entry(s.to_string()).or_default() += 1;
            }
        }
    }
    let esum: usize = counts.iter().sum();
    let mut out = Vec::new();
    if miss > 0 {
        out.push(format!("({var} contains {miss} missing value{})", if miss == 1 { "" } else { "s" }));
    }
    if esum == 0 {
        out.push(format!("{var} contains valid {} codes (format checked; no code list available)", kind.verb()));
    } else {
        out.push(format!("{var} contains invalid codes:"));
        out.push(String::new());
        for (k, n) in counts.iter().enumerate().skip(1) {
            out.push(format!("  {k:>2}.  {:<40}{n:>10}", REASONS[k]));
        }
        out.push(format!("      {:<40}{:>10}", "Total", esum));
    }
    if cmd.has_opt("summary", 3) && !freq.is_empty() {
        out.push(String::new());
        out.push(format!("  {:<12}{:>10}", "Invalid code", "Freq."));
        for (c, n) in &freq {
            out.push(format!("  {c:<12}{n:>10}"));
        }
    }
    if cmd.has_opt("list", 1) && !bad_rows.is_empty() {
        out.push(String::new());
        for i in &bad_rows {
            out.push(format!("  {:>6}.  {}", i + 1, v.str_at(*i)));
        }
    }
    if let Some(g) = cmd.opt("generate", 1) {
        let name = g.trim();
        varlist::new_names(ds, name)?;
        let vals = flags.iter().map(|&x| if x.is_nan() { ferrostat_core::missing::SYSMISS } else { x }).collect();
        ds.add_variable(byte_var(name, vals))?;
        st.data_changed = true;
    }
    let r = st.begin_r();
    for (k, n) in counts.iter().enumerate().skip(1) {
        r.set(&format!("e{k}"), *n as f64);
    }
    r.set("esum", esum as f64);
    r.set("miss", miss as f64);
    Ok(out.join("\n"))
}

fn clean(kind: Kind, ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let var = code_var(ds, rest)?;
    let mask = util::touse(ds, st, cmd)?;
    let gen_ = cmd.opt("generate", 3).map(|s| s.trim().to_string());
    let replace = cmd.has_opt("replace", 3);
    if gen_.is_some() && replace {
        return Err(rc_err(198, "options generate() and replace may not be combined"));
    }
    if matches!(kind, Kind::Icd10 | Kind::Icd10cm | Kind::Icd10pcs) && gen_.is_none() && !replace {
        return Err(rc_err(198, "option generate() or replace required"));
    }
    let nodots = cmd.has_opt("nodots", 5);
    let pad = cmd.has_opt("pad", 3);
    let v = ds.get_var(&var).unwrap();
    let mut vals = Vec::with_capacity(ds.n_obs);
    let mut bad = 0;
    for i in 0..ds.n_obs {
        let s = v.str_at(i);
        if !mask[i] {
            vals.push(if gen_.is_some() { String::new() } else { s.to_string() });
            continue;
        }
        if s.trim().is_empty() {
            vals.push(String::new());
            continue;
        }
        match parse(kind, s) {
            Ok(c) => vals.push(if nodots { c } else { dotted(kind, &c) }),
            Err(_) => {
                bad += 1;
                vals.push(s.to_string());
            }
        }
    }
    if bad > 0 {
        return Err(rc_err(459, format!("{var} contains {bad} invalid code{}; use {} check to list them", if bad == 1 { "" } else { "s" }, kind.verb())));
    }
    if pad {
        let w = vals.iter().map(|s| s.len()).max().unwrap_or(0).min(kind.max_len());
        for s in vals.iter_mut().filter(|s| !s.is_empty()) {
            *s = format!("{s:<w$}");
        }
    }
    let changed = (0..ds.n_obs).filter(|&i| mask[i] && v.str_at(i) != vals[i]).count();
    match gen_ {
        Some(name) => {
            varlist::new_names(ds, &name)?;
            ds.add_variable(str_var(&name, vals))?;
        }
        None => {
            let fmt = v.format.clone();
            let label = v.label.clone();
            let idx = ds.var_index[&var];
            let mut nv = str_var(&var, vals);
            nv.format = fmt;
            nv.label = label;
            ds.vars[idx] = nv;
        }
    }
    st.data_changed = true;
    st.begin_r().set("N", changed as f64);
    Ok(format!("({changed} change{} made)", if changed == 1 { "" } else { "s" }))
}

fn generate(kind: Kind, ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (new, src) = rest.split_once('=').ok_or_else(|| rc_err(198, format!("{} generate newvar = varname, ...", kind.verb())))?;
    let new = new.trim();
    varlist::new_names(ds, new)?;
    let var = code_var(ds, src.trim())?;
    let mask = util::touse(ds, st, cmd)?;
    if cmd.has_opt("check", 5) {
        let v = ds.get_var(&var).unwrap();
        let bad = (0..ds.n_obs).filter(|&i| mask[i] && !v.str_at(i).trim().is_empty() && parse(kind, v.str_at(i)).is_err()).count();
        if bad > 0 {
            return Err(rc_err(459, format!("{var} contains invalid codes")));
        }
    }
    let v = ds.get_var(&var).unwrap();
    let codes: Vec<Option<String>> = (0..ds.n_obs).map(|i| if mask[i] { parse(kind, v.str_at(i)).ok() } else { None }).collect();
    let mut note = None;
    let nv = if let Some(list) = cmd.opt("range", 1) {
        let items = codelist(kind, list)?;
        let vals = codes.iter().map(|c| c.as_ref().map_or(ferrostat_core::missing::SYSMISS, |c| if in_list(&items, c) { 1.0 } else { 0.0 })).collect();
        byte_var(new, vals)
    } else if cmd.has_opt("category", 3) {
        str_var(new, codes.iter().map(|c| c.as_ref().map(|c| category(kind, c)).unwrap_or_default()).collect())
    } else if cmd.has_opt("short", 2) {
        if !matches!(kind, Kind::Icd10) {
            return Err(rc_err(198, format!("option short not allowed with {}", kind.verb())));
        }
        str_var(new, codes.iter().map(|c| c.as_ref().map(|c| dotted(kind, &c.chars().take(4).collect::<String>())).unwrap_or_default()).collect())
    } else if cmd.has_opt("description", 1) {
        let add = cmd.opt("addcode", 4).map(|s| s.trim().to_lowercase());
        let nodots = cmd.has_opt("nodots", 5);
        note = Some(NO_DESC);
        str_var(
            new,
            codes
                .iter()
                .map(|c| match (c, &add) {
                    (Some(c), Some(_)) => {
                        if nodots {
                            c.clone()
                        } else {
                            dotted(kind, c)
                        }
                    }
                    _ => String::new(),
                })
                .collect(),
        )
    } else {
        return Err(rc_err(198, "one of category, short, description or range() required"));
    };
    ds.add_variable(nv)?;
    st.data_changed = true;
    Ok(note.map(String::from).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure() {
        assert_eq!(parse(Kind::Icd9, "250.01"), Ok("25001".into()));
        assert_eq!(parse(Kind::Icd9, "E812.0"), Ok("E8120".into()));
        assert_eq!(parse(Kind::Icd9, "25.001"), Err(1));
        assert_eq!(parse(Kind::Icd9p, "45.13"), Ok("4513".into()));
        assert_eq!(parse(Kind::Icd10, "a27.0"), Ok("A270".into()));
        assert_eq!(parse(Kind::Icd10, "A2"), Err(3));
        assert_eq!(parse(Kind::Icd10cm, "S52.521A"), Ok("S52521A".into()));
        assert_eq!(parse(Kind::Icd10pcs, "0DTJ4ZZ"), Ok("0DTJ4ZZ".into()));
        assert_eq!(parse(Kind::Icd10pcs, "0DTI4ZZ"), Err(5));
        assert_eq!(dotted(Kind::Icd10cm, "S52521A"), "S52.521A");
        let items = codelist(Kind::Icd10, "A27.0 G40* Y60/Y69.9").unwrap();
        assert!(in_list(&items, "A270") && in_list(&items, "G409") && in_list(&items, "Y691") && in_list(&items, "Y699"));
        assert!(!in_list(&items, "A271") && !in_list(&items, "Y70"));
    }
}
