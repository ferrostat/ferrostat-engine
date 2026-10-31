//! Splitting helpers that respect (), [], {} and quotes.

/// Positions of characters at nesting depth zero.
fn depth_iter(s: &str) -> Vec<(usize, char, bool)> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut q = false;
    for (i, c) in s.char_indices() {
        let top = depth == 0 && !q;
        match c {
            '"' => q = !q,
            '(' | '[' | '{' if !q => depth += 1,
            ')' | ']' | '}' if !q => depth -= 1,
            _ => {}
        }
        out.push((i, c, top && !matches!(c, '"')));
    }
    out
}

/// First top-level occurrence of `ch`.
pub fn find_top(s: &str, ch: char) -> Option<usize> {
    depth_iter(s).into_iter().find(|&(_, c, t)| t && c == ch).map(|x| x.0)
}

/// Split at every top-level `ch`.
pub fn split_top(s: &str, ch: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for (i, c, t) in depth_iter(s) {
        if t && c == ch {
            out.push(s[last..i].trim().to_string());
            last = i + c.len_utf8();
        }
    }
    out.push(s[last..].trim().to_string());
    out
}

/// Split at top-level whitespace.
pub fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for (_, c, t) in depth_iter(s) {
        if t && c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Options text `a b(x) c(y, z)` into (name, arg).
pub fn options(s: &str) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    for w in words(s) {
        let w = w.trim_matches(',').trim().to_string();
        if w.is_empty() {
            continue;
        }
        // a word may hold several options separated by commas at top level
        for part in split_top(&w, ',') {
            if part.is_empty() {
                continue;
            }
            match part.find('(') {
                Some(i) if part.ends_with(')') => out.push((part[..i].trim().to_lowercase(), Some(part[i + 1..part.len() - 1].to_string()))),
                _ => out.push((part.to_lowercase(), None)),
            }
        }
    }
    out
}

/// Does option name `given` abbreviate `full` with at least `min` letters?
pub fn abbrev(given: &str, full: &str, min: usize) -> bool {
    given.len() >= min && full.starts_with(given)
}

/// `name(args)` into (name, args) ; plain `name` gives empty args.
pub fn call(s: &str) -> (String, Option<String>) {
    let s = s.trim();
    match s.find('(') {
        Some(i) if s.ends_with(')') => (s[..i].trim().to_lowercase(), Some(s[i + 1..s.len() - 1].to_string())),
        _ => (s.to_lowercase(), None),
    }
}

/// Brace groups `{a} {b:c d}` and the text outside them.
pub fn braces(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '{' => {
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur).trim().to_string());
                } else {
                    cur.push(c);
                }
            }
            _ if depth > 0 => cur.push(c),
            _ => {}
        }
    }
    out
}
