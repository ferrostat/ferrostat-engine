//! Do-file syntax highlighting: splits text into classified byte ranges.
//! (The browser editor has the same rules in `ferrostat-wasm/web/highlight.js`.)
//!
//! Commands are recognised only where a command can appear: at the start of
//! a statement, after a prefix (quietly, capture, by ...:), after `{`, or
//! after `;` under `#delimit ;`.

use std::collections::HashSet;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Comment,
    String,
    Macro,
    Command,
    Keyword,
    Function,
    Number,
}

const KEYWORDS: &[&str] = &[
    "foreach", "forvalues", "while", "if", "else", "in", "of", "local", "global", "program", "end", "capture", "quietly", "noisily", "by", "bysort", "continue", "break", "exit", "version", "do",
    "run", "include", "preserve", "restore", "return", "ereturn", "syntax", "args", "tempvar", "tempname", "tempfile",
];
/// Words after which the next word is again a command.
const PREFIXES: &[&str] = &["capture", "cap", "quietly", "qui", "noisily", "noi", "xi", "version", "nobreak", "timer"];

/// Classified ranges covering the whole text, in order.
pub fn tokenize(text: &str, commands: &HashSet<String>) -> Vec<(Range<usize>, Kind)> {
    let mut out: Vec<(Range<usize>, Kind)> = Vec::new();
    let mut push = |r: Range<usize>, k: Kind| {
        if r.is_empty() {
            return;
        }
        match out.last_mut() {
            Some((last, lk)) if *lk == k && last.end == r.start => last.end = r.end,
            _ => out.push((r, k)),
        }
    };
    let mut depth = 0usize; // /* */ nesting
    let mut delimit_semicolon = false;
    let mut continuation = false;
    let mut base = 0usize;
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let b = body.as_bytes();
        let n = b.len();
        let continued = continuation;
        continuation = false;
        let mut at_start = !continued;
        let mut i = 0usize;
        let rest_is = |i: usize, s: &str| body[i..].starts_with(s);
        let scan_block = |from: usize, depth: &mut usize| {
            let mut j = from;
            while j < n && *depth > 0 {
                if body[j..].starts_with("/*") {
                    *depth += 1;
                    j += 2;
                } else if body[j..].starts_with("*/") {
                    *depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            j
        };
        if depth > 0 {
            let j = scan_block(0, &mut depth);
            push(base..base + j, Kind::Comment);
            i = j;
        }
        let trimmed = body.trim_start();
        let lead = n - trimmed.len();
        let mut line_done = false;
        if depth == 0 && i == 0 {
            if trimmed.starts_with('*') && !continued {
                push(base..base + lead, Kind::Plain);
                push(base + lead..base + n, Kind::Comment);
                line_done = true;
            } else if let Some(r) = trimmed.strip_prefix("#delimit") {
                delimit_semicolon = r.trim() == ";";
                push(base..base + lead, Kind::Plain);
                push(base + lead..base + n, Kind::Keyword);
                line_done = true;
            }
        }
        while !line_done && i < n {
            let c = b[i];
            if rest_is(i, "/*") {
                depth = 1;
                let j = scan_block(i + 2, &mut depth);
                push(base + i..base + j, Kind::Comment);
                i = j;
                continue;
            }
            if rest_is(i, "//") && (i == 0 || b[i - 1].is_ascii_whitespace()) {
                if rest_is(i, "///") {
                    continuation = true;
                }
                push(base + i..base + n, Kind::Comment);
                break;
            }
            if rest_is(i, "`\"") {
                let end = body[i + 2..].find("\"'").map_or(n, |j| i + 2 + j + 2);
                string_with_macros(body, i, end, base, &mut push);
                i = end;
                at_start = false;
                continue;
            }
            if c == b'`' {
                let mut d = 0i32;
                let mut j = i;
                while j < n {
                    if b[j] == b'`' {
                        d += 1;
                    } else if b[j] == b'\'' {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    }
                    j += 1;
                }
                let end = (j + 1).min(n);
                push(base + i..base + end, Kind::Macro);
                i = end;
                at_start = false;
                continue;
            }
            if c == b'$' {
                if let Some(len) = global_len(&body[i..]) {
                    push(base + i..base + i + len, Kind::Macro);
                    i += len;
                    at_start = false;
                    continue;
                }
            }
            if c == b'"' {
                let end = body[i + 1..].find('"').map_or(n, |j| i + 1 + j + 1);
                string_with_macros(body, i, end, base, &mut push);
                i = end;
                at_start = false;
                continue;
            }
            if i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.') {
                if let Some(len) = number_len(&body[i..]) {
                    push(base + i..base + i + len, Kind::Number);
                    i += len;
                    at_start = false;
                    continue;
                }
            }
            if c.is_ascii_alphabetic() || c == b'_' {
                let mut j = i + 1;
                while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
                let w = &body[i..j];
                let is_kw = KEYWORDS.contains(&w);
                let kind = if at_start && is_kw {
                    Kind::Keyword
                } else if at_start && commands.contains(w) {
                    Kind::Command
                } else if is_kw && matches!(w, "in" | "of" | "if" | "else") {
                    Kind::Keyword
                } else if body[j..].starts_with('(') {
                    Kind::Function
                } else {
                    Kind::Plain
                };
                push(base + i..base + j, kind);
                at_start = at_start && PREFIXES.contains(&w);
                i = j;
                continue;
            }
            if c == b':' || c == b'{' || (c == b';' && delimit_semicolon) {
                at_start = true;
            } else if !c.is_ascii_whitespace() {
                at_start = false;
            }
            let len = body[i..].chars().next().map_or(1, |ch| ch.len_utf8());
            push(base + i..base + i + len, Kind::Plain);
            i += len;
        }
        // the newline (or, inside a block comment, keep it part of the comment)
        push(base + n..base + line.len(), if depth > 0 { Kind::Comment } else { Kind::Plain });
        base += line.len();
    }
    out
}

/// String token with any macros inside it marked (they expand at run time).
fn string_with_macros(body: &str, start: usize, end: usize, base: usize, push: &mut impl FnMut(Range<usize>, Kind)) {
    let s = &body[start..end];
    let b = s.as_bytes();
    let mut i = 0;
    let mut last = 0;
    // the opening `" of a compound string is part of the string
    if s.starts_with("`\"") {
        i = 2;
    }
    while i < b.len() {
        let len = if b[i] == b'`' {
            s[i + 1..].find(|c| c == '\'' || c == '`' || c == '"').filter(|&j| s.as_bytes()[i + 1 + j] == b'\'').map(|j| j + 2)
        } else if b[i] == b'$' {
            global_len(&s[i..])
        } else {
            None
        };
        match len {
            Some(l) => {
                push(base + start + last..base + start + i, Kind::String);
                push(base + start + i..base + start + i + l, Kind::Macro);
                i += l;
                last = i;
            }
            None => i += 1,
        }
    }
    push(base + start + last..base + end, Kind::String);
}

/// Length of `$name` or `${...}` at the start of `s`.
fn global_len(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'$') {
        return None;
    }
    if b.get(1) == Some(&b'{') {
        return s.find('}').map(|j| j + 1);
    }
    let mut j = 1;
    if !(b.get(1).is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')) {
        return None;
    }
    while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
        j += 1;
    }
    Some(j)
}

/// Length of a number (`12`, `1.5`, `.5`, `1e-3`) or missing value (`.`, `.a`).
fn number_len(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let digits = |mut j: usize| {
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let exponent = |j: usize| {
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                k += 1;
            }
            let e = digits(k);
            if e > k {
                return e;
            }
        }
        j
    };
    let ident = |j: usize| j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_');
    if b.first().is_some_and(|c| c.is_ascii_digit()) {
        let mut j = digits(0);
        if j < b.len() && b[j] == b'.' {
            j = digits(j + 1);
        }
        return Some(exponent(j));
    }
    if b.first() == Some(&b'.') {
        if b.get(1).is_some_and(|c| c.is_ascii_digit()) {
            return Some(exponent(digits(1)));
        }
        // missing value: `.` or `.a`-`.z` not followed by identifier characters
        if b.get(1).is_some_and(|c| c.is_ascii_lowercase()) && !ident(2) {
            return Some(2);
        }
        if !ident(1) {
            return Some(1);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes(text: &str) -> Vec<(String, Kind)> {
        let cmds: HashSet<String> = ["regress", "reg", "gen", "summarize", "di", "use", "label", "xtivreg", "cap", "noi"].iter().map(|s| s.to_string()).collect();
        let toks = tokenize(text, &cmds);
        // the ranges tile the text exactly
        let mut pos = 0;
        for (r, _) in &toks {
            assert_eq!(r.start, pos, "gap or overlap at {pos} in {toks:?}");
            pos = r.end;
        }
        assert_eq!(pos, text.len());
        toks.into_iter().filter(|(_, k)| *k != Kind::Plain).map(|(r, k)| (text[r].to_string(), k)).collect()
    }

    #[test]
    fn classifies_do_file_constructs() {
        use Kind::*;
        let t = classes("* header\nuse \"d.dta\", clear // note\ngen double y2 = log(y) * 2 if x > .5\nby id: summarize x\n");
        assert_eq!(
            t,
            vec![
                ("* header".into(), Comment),
                ("use".into(), Command),
                ("\"d.dta\"".into(), String),
                ("// note".into(), Comment),
                ("gen".into(), Command),
                ("log".into(), Function),
                ("2".into(), Number),
                ("if".into(), Keyword),
                (".5".into(), Number),
                ("by".into(), Keyword),
                ("summarize".into(), Command),
            ]
        );
    }

    #[test]
    fn macros_comments_and_continuations() {
        use Kind::*;
        let t = classes("foreach v of varlist a {\n    di \"`v' = \" `v'[1] + $g + ${h}\n}\n/* a\n /* b */ c\n*/ reg y x\nxtivreg y (x = z), fe ///\n    * 2\n");
        assert!(t.contains(&("foreach".into(), Keyword)));
        assert!(t.contains(&("of".into(), Keyword)));
        assert!(t.contains(&("\"".into(), String)) && t.contains(&(" = \"".into(), String)));
        assert_eq!(t.iter().filter(|(s, k)| s == "`v'" && *k == Macro).count(), 2);
        assert!(t.contains(&("$g".into(), Macro)) && t.contains(&("${h}".into(), Macro)));
        // a nested block comment over several lines is one comment token
        assert!(t.contains(&("/* a\n /* b */ c\n*/".into(), Comment)), "{t:?}");
        assert!(t.contains(&("reg".into(), Command)));
        assert!(t.contains(&("///".into(), Comment)));
        // a continued line starting with * is not a comment
        assert!(!t.iter().any(|(s, k)| s.contains("* 2") && *k == Comment));
    }

    #[test]
    fn prefixes_compound_strings_and_numbers() {
        use Kind::*;
        let t = classes("cap noi regress y x\nlabel variable x `\"say \"hi\"\"'\ndi 1e-3 .a 12.5 x.y\n");
        assert!(t.contains(&("cap".into(), Command)) && t.contains(&("noi".into(), Command)) && t.contains(&("regress".into(), Command)));
        assert!(t.contains(&("`\"say \"hi\"\"'".into(), String)));
        assert!(t.contains(&("1e-3".into(), Number)) && t.contains(&(".a".into(), Number)) && t.contains(&("12.5".into(), Number)));
        // words after the command are not commands even if they match
        let t2 = classes("regress reg\n");
        assert_eq!(t2, vec![("regress".into(), Command)]);
    }
}
