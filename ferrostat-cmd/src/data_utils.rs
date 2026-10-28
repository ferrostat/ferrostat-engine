//! Data-management utilities: variable lists and lookup (`ds`, `lookfor`,
//! `vl`, `labelbook`, frame files), generation and transformation
//! (`range`, `ipolate`, `stack`, `pctile`, `drawnorm`, ...), integrity
//! checks (`compare`, `datasignature`, ...), text-file helpers (`outfile`,
//! `filefilter`, `hexdump`, `zipfile`, `unicode`) and the foreign-format
//! importers in [`formats`].

use crate::commands::CommandHandler;

pub mod check;
pub mod files;
pub mod formats;
pub mod genr;
pub mod vars;

/// Shell-style pattern match (`*` any run, `?` one character).
pub fn glob(pat: &str, s: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    let (mut i, mut j, mut star, mut mark) = (0usize, 0usize, usize::MAX, 0usize);
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = i;
            mark = j;
            i += 1;
        } else if star != usize::MAX {
            i = star + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}

/// Names laid out in columns of equal width, wrapped at 80 characters.
pub fn columns(names: &[String]) -> String {
    let w = names.iter().map(|n| n.chars().count()).max().unwrap_or(0) + 2;
    let per = (80 / w.max(1)).max(1);
    let mut out = Vec::new();
    for chunk in names.chunks(per) {
        let line: String = chunk.iter().map(|n| format!("{:<w$}", n)).collect();
        out.push(line.trim_end().to_string());
    }
    out.join("\n")
}

/// All commands of this module, for registration in the engine.
pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = Vec::new();
    v.extend(vars::handlers());
    v.extend(genr::handlers());
    v.extend(check::handlers());
    v.extend(files::handlers());
    v
}
