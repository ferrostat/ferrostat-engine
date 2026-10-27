//! Tokens of the matrix language.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    /// `.` (0) and `.a`..`.z` (1..26)
    Miss(u8),
    Str(String),
    Id(String),
    Op(&'static str),
    /// End of statement (newline or `;`).
    Nl,
    Semi,
    /// A whole `mata ...` command line, e.g. `mata describe`.
    Cmd(String),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    /// 0-based source line
    pub line: usize,
}

const OPS: &[&str] = &[
    ":==", ":!=", ":>=", ":<=", "::", "..", ":+", ":-", ":*", ":/", ":^", ":>", ":<", ":&", ":|", "==", "!=", ">=", "<=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "^=", "[|",
    "|]", "->", "+", "-", "*", "/", "^", "'", ",", "\\", "(", ")", "[", "]", "{", "}", "=", "<", ">", "!", "&", "|", "?", ":", "#", ".",
];

/// Operators after which a statement continues on the next line.
fn continues(t: &Tok) -> bool {
    match t {
        Tok::Op(o) => !matches!(*o, "'" | ")" | "]" | "|]" | "}" | "{" | "++" | "--"),
        _ => false,
    }
}

fn at_stmt_start(out: &[Token]) -> bool {
    match out.last().map(|t| &t.tok) {
        None | Some(Tok::Nl) | Some(Tok::Semi) => true,
        Some(Tok::Op(o)) => *o == "{" || *o == "}",
        _ => false,
    }
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let c: Vec<char> = src.chars().collect();
    let mut out: Vec<Token> = Vec::new();
    let mut i = 0;
    let mut line = 0;
    let mut depth = 0i32;
    while i < c.len() {
        let ch = c[i];
        if ch == '\n' {
            line += 1;
            i += 1;
            let cont = out.last().map(|t| continues(&t.tok)).unwrap_or(false);
            if depth <= 0 && !cont && !matches!(out.last().map(|t| &t.tok), Some(Tok::Nl) | None) {
                out.push(Token { tok: Tok::Nl, line: line - 1 });
            }
            continue;
        }
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        let push = |out: &mut Vec<Token>, tok: Tok| out.push(Token { tok, line });
        // `mata ...` commands and `version` lines at the start of a statement
        if ch.is_ascii_alphabetic() && at_stmt_start(&out) {
            let rest: String = c[i..].iter().take_while(|&&x| x != '\n').collect();
            let word: String = rest.chars().take_while(|x| x.is_alphanumeric() || *x == '_').collect();
            let after = rest[word.len()..].chars().next();
            if word == "mata" && after.map_or(true, |x| x == ' ' || x == '\t') {
                let body = rest[4..].trim().trim_end_matches(';').to_string();
                if !body.starts_with(':') {
                    push(&mut out, Tok::Cmd(body));
                    i += rest.chars().count();
                    continue;
                }
            }
            if word == "version" && after.map_or(true, |x| x == ' ' || x == '\t') {
                i += rest.chars().count();
                continue;
            }
        }
        if ch == ';' {
            if depth > 0 {
                push(&mut out, Tok::Semi);
            } else if !matches!(out.last().map(|t| &t.tok), Some(Tok::Nl) | None) {
                push(&mut out, Tok::Nl);
            }
            i += 1;
            continue;
        }
        if ch == '"' || (ch == '`' && c.get(i + 1) == Some(&'"')) {
            let compound = ch == '`';
            i += if compound { 2 } else { 1 };
            let mut s = String::new();
            let mut nest = 0;
            loop {
                if i >= c.len() || c[i] == '\n' {
                    return Err("unbalanced quotes".into());
                }
                if compound && c[i] == '`' && c.get(i + 1) == Some(&'"') {
                    nest += 1;
                    s.push_str("`\"");
                    i += 2;
                    continue;
                }
                if compound && c[i] == '"' && c.get(i + 1) == Some(&'\'') {
                    if nest == 0 {
                        i += 2;
                        break;
                    }
                    nest -= 1;
                    s.push_str("\"'");
                    i += 2;
                    continue;
                }
                if !compound && c[i] == '"' {
                    i += 1;
                    break;
                }
                s.push(c[i]);
                i += 1;
            }
            push(&mut out, Tok::Str(s));
            continue;
        }
        let after_value = matches!(out.last().map(|t| &t.tok), Some(Tok::Id(_)) | Some(Tok::Op(")")) | Some(Tok::Op("]")));
        if ch.is_ascii_digit() || (ch == '.' && c.get(i + 1).map_or(false, |x| x.is_ascii_digit())) {
            let st = i;
            while i < c.len() && c[i].is_ascii_digit() {
                i += 1;
            }
            if i < c.len() && c[i] == '.' && c.get(i + 1) != Some(&'.') {
                i += 1;
                while i < c.len() && c[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let mut k = i + 1;
                if k < c.len() && (c[k] == '+' || c[k] == '-') {
                    k += 1;
                }
                if k < c.len() && c[k].is_ascii_digit() {
                    i = k;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let s: String = c[st..i].iter().collect();
            let v: f64 = s.parse().map_err(|_| format!("invalid number {s}"))?;
            push(&mut out, Tok::Num(v));
            continue;
        }
        if ch == '.' && !after_value && c.get(i + 1) != Some(&'.') {
            // missing value literal
            if let Some(&l) = c.get(i + 1) {
                if l.is_ascii_lowercase() && !c.get(i + 2).map_or(false, |x| x.is_alphanumeric() || *x == '_') {
                    push(&mut out, Tok::Miss(l as u8 - b'a' + 1));
                    i += 2;
                    continue;
                }
            }
            push(&mut out, Tok::Miss(0));
            i += 1;
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            push(&mut out, Tok::Id(c[st..i].iter().collect()));
            continue;
        }
        let rest: String = c[i..(i + 3).min(c.len())].iter().collect();
        let op = OPS.iter().find(|o| rest.starts_with(**o)).ok_or_else(|| format!("unexpected character {ch}"))?;
        match *op {
            "(" | "[" | "[|" => depth += 1,
            ")" | "]" | "|]" => depth -= 1,
            _ => {}
        }
        i += op.len();
        push(&mut out, Tok::Op(op));
    }
    if !matches!(out.last().map(|t| &t.tok), Some(Tok::Nl) | None) {
        out.push(Token { tok: Tok::Nl, line });
    }
    Ok(out)
}
