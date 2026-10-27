//! Turning do-file text into statements: comments, `///` continuation,
//! `#delimit ;`, and `{ }` blocks.

/// One statement; block statements carry their body.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub text: String,
    pub body: Option<Vec<Stmt>>,
    /// Raw lines for `program define` / `input` / `mata` blocks.
    pub raw: Option<Vec<String>>,
}

/// Remove `/* */` comments (which may span lines) outside of strings.
fn strip_block_comments(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut in_str = false;
    let mut depth = 0;
    while i < c.len() {
        if depth > 0 {
            if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                depth -= 1;
                i += 2;
                continue;
            }
            if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                depth += 1;
                i += 2;
                continue;
            }
            if c[i] == '\n' {
                out.push('\n');
            }
            i += 1;
            continue;
        }
        if c[i] == '\n' {
            in_str = false;
        }
        if c[i] == '"' {
            in_str = !in_str;
        }
        if !in_str && c[i] == '/' && c.get(i + 1) == Some(&'*') {
            depth = 1;
            i += 2;
            out.push(' ');
            continue;
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// Remove a trailing `//` comment (outside strings, preceded by blank).
pub fn strip_line_comment(line: &str) -> (String, bool) {
    let c: Vec<char> = line.chars().collect();
    let mut in_str = false;
    let mut compound = 0;
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if compound == 0 && ch == '"' {
            in_str = !in_str;
        } else if ch == '`' && c.get(i + 1) == Some(&'"') {
            compound += 1;
        } else if ch == '"' && c.get(i + 1) == Some(&'\'') && compound > 0 {
            compound -= 1;
        }
        if !in_str && compound == 0 && ch == '/' && c.get(i + 1) == Some(&'/') && (i == 0 || c[i - 1].is_whitespace()) {
            let cont = c.get(i + 2) == Some(&'/');
            return (c[..i].iter().collect::<String>().trim_end().to_string(), cont);
        }
        i += 1;
    }
    (line.to_string(), false)
}

/// Split text into logical lines (statements without block structure).
pub fn logical_lines(text: &str) -> Vec<String> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let text = strip_block_comments(&text);
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut delimit = false;
    // inside a python block lines are kept verbatim (indentation matters)
    let mut python = false;
    for raw in text.split('\n') {
        let trimmed = raw.trim();
        if python {
            python = !trimmed.eq_ignore_ascii_case("end");
            out.push(if python { raw.trim_end().to_string() } else { "end".to_string() });
            continue;
        }
        if cur.is_empty() && trimmed.starts_with('*') {
            continue;
        }
        if trimmed.to_lowercase().starts_with("#delimit") || trimmed.to_lowercase().starts_with("#d ") || trimmed.eq_ignore_ascii_case("#d;") {
            let arg = trimmed.split_whitespace().nth(1).unwrap_or(";");
            delimit = arg.starts_with(';') || trimmed.ends_with(';');
            if trimmed.ends_with("cr") {
                delimit = false;
            }
            continue;
        }
        let (code, cont) = strip_line_comment(raw);
        if delimit {
            cur.push_str(&code);
            cur.push(' ');
            while let Some(pos) = find_semicolon(&cur) {
                let stmt = cur[..pos].trim().to_string();
                if !stmt.is_empty() && !stmt.starts_with('*') {
                    out.push(stmt);
                }
                cur = cur[pos + 1..].to_string();
            }
            continue;
        }
        if cont {
            cur.push_str(&code);
            cur.push(' ');
            continue;
        }
        cur.push_str(&code);
        let line = cur.trim().to_string();
        cur.clear();
        if line.is_empty() {
            continue;
        }
        python = matches!(line.as_str(), "python" | "python:");
        out.extend(split_braces(&line));
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn find_semicolon(s: &str) -> Option<usize> {
    let mut in_str = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            in_str = !in_str;
        }
        if c == ';' && !in_str {
            return Some(i);
        }
    }
    None
}

/// `} else {` -> `}`, `else {`; `else if x {` stays.
fn split_braces(line: &str) -> Vec<String> {
    let t = line.trim();
    if t.starts_with('}') && t.len() > 1 {
        let rest = t[1..].trim();
        let mut v = vec!["}".to_string()];
        if !rest.is_empty() {
            v.extend(split_braces(rest));
        }
        return v;
    }
    vec![t.to_string()]
}

/// Does this line open a block (ends with an unquoted `{`)?
pub fn opens_block(line: &str) -> bool {
    let t = line.trim_end();
    if !t.ends_with('{') {
        return false;
    }
    // ignore `{` inside quotes
    let mut in_str = false;
    let mut last_open = None;
    for (i, c) in t.char_indices() {
        if c == '"' {
            in_str = !in_str;
        }
        if c == '{' && !in_str {
            last_open = Some(i);
        }
    }
    last_open == Some(t.len() - 1)
}

fn first_words(line: &str, n: usize) -> Vec<String> {
    line.split_whitespace().take(n).map(|w| w.to_lowercase()).collect()
}

/// Is this a `program define name` (or `program name`) header?
pub fn is_program_header(line: &str) -> bool {
    let w = first_words(line, 3);
    if w.is_empty() {
        return false;
    }
    let p = w[0].as_str();
    if !(p == "program" || p == "pr" || p == "prog" || p == "progr" || p == "progra") {
        return false;
    }
    match w.get(1).map(String::as_str) {
        None => false,
        Some("drop") | Some("dir") | Some("list") | Some("l") | Some("li") => false,
        Some(_) => true,
    }
}

pub fn is_end(line: &str) -> bool {
    line.trim().eq_ignore_ascii_case("end")
}

fn is_raw_block(line: &str) -> bool {
    let w = first_words(line, 2);
    let first = w.first().map(String::as_str).unwrap_or("");
    is_program_header(line)
        || first == "input" && !line.contains(',') || first == "inp"
        || first == "mata" && !line.trim_end().ends_with(':') && (w.len() == 1 || w[1] == "{")
        || first == "mata:" && w.len() == 1
        || first == "python" && w.len() == 1
        || first == "python:" && w.len() == 1
}

/// Build the statement tree from logical lines.
pub fn parse_block(lines: &[String]) -> Result<Vec<Stmt>, String> {
    let mut i = 0;
    parse_until(lines, &mut i, false)
}

fn parse_until(lines: &[String], i: &mut usize, inside: bool) -> Result<Vec<Stmt>, String> {
    let mut out = Vec::new();
    while *i < lines.len() {
        let line = lines[*i].clone();
        *i += 1;
        let t = line.trim();
        if t == "}" {
            if inside {
                return Ok(out);
            }
            return Err("unexpected }\nr(198);".into());
        }
        if is_raw_block(t) {
            let mut raw = Vec::new();
            let mut closed = false;
            while *i < lines.len() {
                let l = lines[*i].clone();
                *i += 1;
                if is_end(&l) {
                    closed = true;
                    break;
                }
                raw.push(l);
            }
            if !closed && !t.to_lowercase().starts_with("input") {
                return Err("program definition (or mata/input block) not closed with end\nr(198);".into());
            }
            out.push(Stmt { text: line, body: None, raw: Some(raw) });
            continue;
        }
        if opens_block(t) {
            let header = t[..t.len() - 1].trim_end().to_string();
            let body = parse_until(lines, i, true)?;
            out.push(Stmt { text: header, body: Some(body), raw: None });
            continue;
        }
        out.push(Stmt { text: line, body: None, raw: None });
    }
    if inside {
        return Err("unexpected end of file (missing })\nr(198);".into());
    }
    Ok(out)
}

/// Is the accumulated input complete (balanced braces, no open program)?
pub fn is_complete(text: &str) -> bool {
    let lines = logical_lines(text);
    let ends_with_cont = text.trim_end().ends_with("///");
    if ends_with_cont {
        return false;
    }
    let mut depth = 0i32;
    let mut raw = false;
    for l in &lines {
        let t = l.trim();
        if raw {
            if is_end(t) {
                raw = false;
            }
            continue;
        }
        if is_raw_block(t) {
            if t.to_lowercase().starts_with("input") || t.to_lowercase().starts_with("inp") {
                raw = true;
                continue;
            }
            raw = true;
            continue;
        }
        if t == "}" {
            depth -= 1;
        } else if opens_block(t) {
            depth += 1;
        }
    }
    depth <= 0 && !raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines() {
        let s = "* comment\ngen x = 1 // note\nreg y x, ///\n   robust\n/* block\ncomment */ list\n";
        assert_eq!(logical_lines(s), ["gen x = 1", "reg y x,    robust", "list"]);
    }

    #[test]
    fn delimit() {
        let s = "#delimit ;\nreg y\n x;\ndi 1;\n#delimit cr\ndi 2";
        assert_eq!(logical_lines(s), ["reg y  x", "di 1", "di 2"]);
    }

    #[test]
    fn blocks() {
        let s = "foreach v in a b {\n di \"`v'\"\n if 1 {\n  di 2\n }\n}\nelse {\n di 3\n}";
        let st = parse_block(&logical_lines(s)).unwrap();
        assert_eq!(st.len(), 2);
        assert_eq!(st[0].body.as_ref().unwrap().len(), 2);
        let s2 = "if 1 {\n di 1\n} else {\n di 2\n}";
        let st2 = parse_block(&logical_lines(s2)).unwrap();
        assert_eq!(st2.len(), 2);
        assert_eq!(st2[1].text, "else");
    }

    #[test]
    fn program_block() {
        let s = "program define hello\n di \"hi\"\nend\nhello";
        let st = parse_block(&logical_lines(s)).unwrap();
        assert_eq!(st.len(), 2);
        assert_eq!(st[0].raw.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn completeness() {
        assert!(!is_complete("foreach x in a {"));
        assert!(is_complete("foreach x in a {\n di 1\n}"));
        assert!(!is_complete("program define f"));
        assert!(!is_complete("reg y x ///"));
    }
}
