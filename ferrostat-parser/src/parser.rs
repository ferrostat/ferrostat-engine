//! Recursive-descent parser for expressions and command lines.
//!
//! Operator precedence, from loosest to tightest:
//! `|`, `&`, relational (`== != ~= < <= > >=`), `+ -`, `* /`,
//! unary minus, `^` (right-associative), `!`/`~`, postfix (calls, `[ ]`).
//! So `-2^2` is `-(2^2)` and `2^-1` is `2^(-1)`.

use crate::lexer::{Lexer, Token};
use crate::token::{BinOp, UnaryOp};

// Re-export types for backwards compatibility
pub use crate::token::{Command, Expr, InRange, ParseError, WeightSpec};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    raw: String,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        let raw = input.to_string();
        let tokens: Vec<Token> = Lexer::new(input).collect();
        Self { tokens, pos: 0, raw }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    pub fn at_end(&self) -> bool {
        matches!(self.current(), Token::Eof)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::new(&format!("expected {}, found {}", expected, self.current())))
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    /// Parse an expression that must consume all input.
    pub fn parse_full_expr(&mut self) -> Result<Expr, ParseError> {
        let e = self.parse_or()?;
        if !self.at_end() {
            return Err(ParseError::new(&format!("invalid syntax near {}", self.current())));
        }
        Ok(e)
    }

    pub fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while matches!(self.current(), Token::Pipe) {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::Or, Box::new(rhs));
        }
        Ok(lhs)
    }

    pub fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_eq()?;
        while matches!(self.current(), Token::Amp) {
            self.advance();
            let rhs = self.parse_eq()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::And, Box::new(rhs));
        }
        Ok(lhs)
    }

    pub fn parse_eq(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.current() {
                Token::Eq => BinOp::Eq,
                Token::Ne => BinOp::Ne,
                Token::Lt => BinOp::Lt,
                Token::Le => BinOp::Le,
                Token::Gt => BinOp::Gt,
                Token::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_add()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    /// Kept for API compatibility; relational operators share one level.
    pub fn parse_rel(&mut self) -> Result<Expr, ParseError> {
        self.parse_add()
    }

    pub fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.current() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_mul()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    pub fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.current() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    pub fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.current() {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::Unary(UnaryOp::Neg, Box::new(operand)))
            }
            Token::Plus => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_pow(),
        }
    }

    pub fn parse_pow(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_not()?;
        if matches!(self.current(), Token::Caret) {
            self.advance();
            let rhs = self.parse_unary()?; // right associative, allows 2^-1
            return Ok(Expr::Binary(Box::new(lhs), BinOp::Pow, Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.current(), Token::Bang | Token::Tilde) {
            self.advance();
            // `!` applies to a whole comparison operand: !(x) == 1 is rare;
            // `!` has the highest precedence.
            let operand = self.parse_not()?;
            return Ok(Expr::Unary(UnaryOp::Not, Box::new(operand)));
        }
        if matches!(self.current(), Token::Minus) {
            // e.g. 2^-x
            self.advance();
            let operand = self.parse_not()?;
            return Ok(Expr::Unary(UnaryOp::Neg, Box::new(operand)));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_primary()?;
        while matches!(self.current(), Token::LBracket) {
            self.advance();
            let idx = self.parse_subscript()?;
            self.expect(&Token::RBracket)?;
            e = Expr::Index(Box::new(e), Box::new(idx));
        }
        Ok(e)
    }

    /// Subscript content: an expression, or a coefficient name such as
    /// `_b[2.group]`, `_b[eq:x]`, `_b[c.x#c.z]` (kept as a string).
    fn parse_subscript(&mut self) -> Result<Expr, ParseError> {
        let start = self.pos;
        if let Ok(e) = self.parse_expr() {
            if matches!(self.current(), Token::RBracket) {
                return Ok(e);
            }
        }
        self.pos = start;
        let mut s = String::new();
        while !matches!(self.current(), Token::RBracket | Token::Eof) {
            s.push_str(&self.current().to_string());
            self.advance();
        }
        Ok(Expr::String(s))
    }

    pub fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.current().clone() {
            Token::Num(n) => {
                self.advance();
                Ok(Expr::Number(n))
            }
            Token::Str(s) => {
                self.advance();
                Ok(Expr::String(s))
            }
            Token::Dot => {
                self.advance();
                Ok(Expr::Missing)
            }
            Token::ExtMissing(c) => {
                self.advance();
                Ok(Expr::MissingCode(c as u8 - b'a' + 1))
            }
            Token::Ident(name) => {
                self.advance();
                if matches!(self.current(), Token::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.current(), Token::RParen) {
                        args.push(self.parse_expr()?);
                        while matches!(self.current(), Token::Comma) {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Paren(Box::new(expr)))
            }
            t => Err(ParseError::new(&format!("unexpected token in expression: {}", t))),
        }
    }

    /// Parse a full command line.
    pub fn parse_command(&mut self) -> Result<Command, ParseError> {
        parse_command_text(&self.raw)
    }

    pub fn is_keyword(&self, kw: &str) -> bool {
        matches!(self.current(), Token::Ident(s) if s.eq_ignore_ascii_case(kw))
    }

    pub fn is_command_context(&self) -> bool {
        false
    }

    pub fn parse_in_range(&mut self) -> Result<InRange, ParseError> {
        let text: Vec<String> = self.tokens[self.pos..].iter().map(|t| t.to_string()).collect();
        parse_in_text(&text.join(""))
    }

    pub fn parse_weight(&mut self) -> Result<WeightSpec, ParseError> {
        self.expect(&Token::LBracket)?;
        let wt_type = if let Token::Ident(s) = self.current() {
            let s = s.clone();
            self.advance();
            s
        } else {
            return Err(ParseError::new("weight type expected (fw, aw, pw, iw)"));
        };
        self.expect(&Token::Equals)?;
        let expr = self.parse_expr()?;
        self.expect(&Token::RBracket)?;
        Ok(WeightSpec { weight_type: wt_type, expr })
    }

    pub fn parse_options(&mut self) -> Result<Vec<(String, Option<Expr>)>, ParseError> {
        let text: Vec<String> = self.tokens[self.pos..].iter().map(|t| t.to_string()).collect();
        Ok(compat_options(&split_options(&text.join(" "))))
    }
}

// ---------------------------------------------------------------------------
// Command-line splitting
// ---------------------------------------------------------------------------

/// Scanner state for finding top-level positions in raw text.
fn top_level_positions(s: &str) -> Vec<(usize, char, bool)> {
    // (byte index, char, is_top_level)
    let mut out = Vec::with_capacity(s.len());
    let mut depth_paren = 0i32;
    let mut depth_brack = 0i32;
    let mut in_str = false;
    let mut compound = 0i32;
    let b: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < b.len() {
        let (idx, c) = b[i];
        let next = b.get(i + 1).map(|x| x.1);
        let top = !in_str && compound == 0 && depth_paren == 0 && depth_brack == 0;
        out.push((idx, c, top));
        if compound > 0 {
            if c == '`' && next == Some('"') {
                compound += 1;
                out.push((b[i + 1].0, '"', false));
                i += 2;
                continue;
            }
            if c == '"' && next == Some('\'') {
                compound -= 1;
                out.push((b[i + 1].0, '\'', false));
                i += 2;
                continue;
            }
        } else if in_str {
            if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '`' if next == Some('"') => {
                    compound = 1;
                    out.push((b[i + 1].0, '"', false));
                    i += 2;
                    continue;
                }
                '"' => in_str = true,
                '(' => depth_paren += 1,
                ')' => depth_paren -= 1,
                '[' => depth_brack += 1,
                ']' => depth_brack -= 1,
                _ => {}
            }
        }
        i += 1;
    }
    out
}

/// Byte index of the first top-level occurrence of `ch`.
pub fn find_top_level(s: &str, ch: char) -> Option<usize> {
    top_level_positions(s).into_iter().find(|&(_, c, top)| top && c == ch).map(|(i, _, _)| i)
}

/// Split `s` at top-level whitespace (quotes and parentheses kept together).
pub fn split_top_level_words(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for (_, c, top) in top_level_positions(s) {
        if top && c.is_whitespace() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.' || c == '`' || c == '\''
}

/// Split options text `robust level(90), by(a b)` into (name, raw-arg).
pub fn split_options(s: &str) -> Vec<(String, Option<String>)> {
    let mut opts = Vec::new();
    let chars: Vec<(usize, char, bool)> = top_level_positions(s);
    let mut i = 0;
    while i < chars.len() {
        let (_, c, _) = chars[i];
        if c.is_whitespace() || c == ',' {
            i += 1;
            continue;
        }
        let mut name = String::new();
        while i < chars.len() && chars[i].2 && !chars[i].1.is_whitespace() && chars[i].1 != '(' && chars[i].1 != ',' {
            name.push(chars[i].1);
            i += 1;
        }
        let mut arg = None;
        if i < chars.len() && chars[i].1 == '(' {
            let start = chars[i].0 + 1;
            i += 1;
            while i < chars.len() && !(chars[i].1 == ')' && chars[i].2 == false && is_closing_top(&chars, i)) {
                i += 1;
            }
            let end = if i < chars.len() { chars[i].0 } else { s.len() };
            arg = Some(s[start..end].to_string());
            i += 1;
        }
        if name.is_empty() {
            i += 1;
            continue;
        }
        opts.push((name, arg));
    }
    opts
}

/// Is the `)` at position i the one closing a top-level `(`?
fn is_closing_top(chars: &[(usize, char, bool)], i: usize) -> bool {
    // The char after a top-level-closing paren is top level again.
    match chars.get(i + 1) {
        Some(&(_, _, top)) => top,
        None => true,
    }
}

fn compat_options(opts: &[(String, Option<String>)]) -> Vec<(String, Option<Expr>)> {
    opts.iter()
        .map(|(n, a)| {
            let e = a.as_ref().and_then(|a| {
                let mut p = Parser::new(a);
                p.parse_expr().ok()
            });
            (n.clone(), e)
        })
        .collect()
}

fn parse_in_bound(s: &str) -> Result<i64, ParseError> {
    let s = s.trim();
    match s {
        "f" | "F" => Ok(1),
        "l" | "L" => Ok(-1),
        _ => s.parse::<i64>().map_err(|_| ParseError::new("in range invalid")),
    }
}

/// Parse the text after `in`: `5`, `2/7`, `f/5`, `-3/l`, `f/l`.
pub fn parse_in_text(s: &str) -> Result<InRange, ParseError> {
    let s = s.trim();
    if let Some((a, b)) = s.split_once('/') {
        let (ta, tb) = (a.trim(), b.trim());
        let (x, y) = (parse_in_bound(ta)?, parse_in_bound(tb)?);
        if (ta == "f" || ta == "F") && y > 0 {
            return Ok(InRange::First(y as usize));
        }
        if x < 0 && y == -1 {
            return Ok(InRange::Last((-x) as usize));
        }
        if x > 0 && y > 0 {
            return Ok(InRange::Range(x as usize, y as usize));
        }
        return Ok(InRange::Rel(x, y));
    }
    let x = parse_in_bound(s)?;
    if x > 0 { Ok(InRange::Single(x as usize)) } else { Ok(InRange::Rel(x, x)) }
}

const WEIGHT_TYPES: [&str; 10] = ["fw", "fweight", "aw", "aweight", "pw", "pweight", "iw", "iweight", "weight", "w"];

/// Parse a full command line from text.
fn parse_command_text(raw: &str) -> Result<Command, ParseError> {
    let mut cmd = Command { raw: raw.to_string(), ..Default::default() };
    let line = strip_comments(raw);
    let line = line.trim();
    if line.is_empty() {
        return Ok(cmd);
    }
    // verb
    let vend = line
        .char_indices()
        .find(|&(_, c)| !(c.is_alphanumeric() || c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(line.len());
    if vend == 0 {
        return Err(ParseError::new("command must start with an identifier"));
    }
    cmd.verb = line[..vend].to_lowercase();
    let rest = &line[vend..];

    // options
    let (body, opt_text) = match find_top_level(rest, ',') {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest, None),
    };
    if let Some(o) = opt_text {
        cmd.opts = split_options(o);
        cmd.options = compat_options(&cmd.opts);
    }

    // clauses in body
    #[derive(Clone, Copy, PartialEq)]
    enum K { If, In, Using, Weight }
    let pos = top_level_positions(body);
    let mut marks: Vec<(usize, usize, K)> = Vec::new(); // (start of keyword, start of content, kind)
    let mut prev_top_ws = true;
    for (k, &(i, c, top)) in pos.iter().enumerate() {
        if top && prev_top_ws {
            for (kw, kind) in [("if", K::If), ("in", K::In), ("using", K::Using)] {
                if body[i..].len() >= kw.len()
                    && body[i..i + kw.len()].eq_ignore_ascii_case(kw)
                    && !body[i + kw.len()..].chars().next().map_or(false, is_word_char)
                {
                    marks.push((i, i + kw.len(), kind));
                }
            }
        }
        if c == '[' && top {
            let close = pos[k..].iter().find(|&&(_, cc, t)| cc == ']' && !t).map(|x| x.0);
            if let Some(close) = close {
                let inner = &body[i + 1..close];
                if let Some((w, _)) = inner.split_once('=') {
                    if WEIGHT_TYPES.contains(&w.trim().to_lowercase().as_str()) {
                        marks.push((i, i + 1, K::Weight));
                    }
                }
            }
        }
        prev_top_ws = top && (c.is_whitespace() || c == ']');
    }
    marks.sort_by_key(|m| m.0);
    let main_end = marks.first().map(|m| m.0).unwrap_or(body.len());
    cmd.main = body[..main_end].trim().to_string();
    for (j, &(_, cstart, kind)) in marks.iter().enumerate() {
        let cend = marks.get(j + 1).map(|m| m.0).unwrap_or(body.len());
        let content = body[cstart..cend].trim();
        match kind {
            K::If => {
                cmd.if_raw = Some(content.to_string());
                cmd.if_expr = Some(Parser::new(content).parse_full_expr()?);
            }
            K::In => cmd.in_range = Some(parse_in_text(content)?),
            K::Using => {
                let p = content.trim();
                let p = p.strip_prefix("`\"").and_then(|x| x.strip_suffix("\"'")).unwrap_or(p);
                cmd.using = Some(p.trim_matches('"').to_string());
            }
            K::Weight => {
                let inner = content.trim_end().trim_end_matches(']');
                let (w, e) = inner.split_once('=').unwrap();
                cmd.weight = Some(WeightSpec {
                    weight_type: w.trim().to_string(),
                    expr: Parser::new(e).parse_full_expr()?,
                });
            }
        }
    }

    cmd.raw_args = Lexer::new(&cmd.main).collect();
    cmd.args = tokens_to_strings(&cmd.raw_args);
    Ok(cmd)
}

/// Remove `//` and `/* */` comments outside of strings.
pub fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut in_str = false;
    let mut compound = 0;
    while i < chars.len() {
        let c = chars[i];
        let n = chars.get(i + 1).copied();
        if compound > 0 {
            if c == '`' && n == Some('"') {
                compound += 1;
            } else if c == '"' && n == Some('\'') {
                compound -= 1;
            }
            out.push(c);
            i += 1;
            continue;
        }
        if in_str {
            if c == '"' {
                in_str = false;
            }
            out.push(c);
            i += 1;
            continue;
        }
        match (c, n) {
            ('`', Some('"')) => {
                compound = 1;
                out.push(c);
            }
            ('"', _) => {
                in_str = true;
                out.push(c);
            }
            ('/', Some('/')) if i == 0 || chars[i - 1].is_whitespace() => break,
            ('/', Some('*')) => {
                i += 2;
                while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                    i += 1;
                }
                i += 2;
                out.push(' ');
                continue;
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

fn tokens_to_strings(tokens: &[Token]) -> Vec<String> {
    tokens
        .iter()
        .filter_map(|t| match t {
            Token::Ident(s) => Some(s.clone()),
            Token::Str(s) => Some(format!("\"{}\"", s)),
            Token::Num(n) => Some(format!("{}", n)),
            Token::Dot => Some(".".to_string()),
            Token::ExtMissing(c) => Some(format!(".{c}")),
            Token::Eq => Some("==".to_string()),
            Token::Ne => Some("!=".to_string()),
            Token::Le => Some("<=".to_string()),
            Token::Ge => Some(">=".to_string()),
            Token::Lt => Some("<".to_string()),
            Token::Gt => Some(">".to_string()),
            Token::Plus => Some("+".to_string()),
            Token::Minus => Some("-".to_string()),
            Token::Star => Some("*".to_string()),
            Token::Slash => Some("/".to_string()),
            Token::Caret => Some("^".to_string()),
            Token::LParen => Some("(".to_string()),
            Token::RParen => Some(")".to_string()),
            Token::Equals => Some("=".to_string()),
            Token::Comma => Some(",".to_string()),
            Token::Colon => Some(":".to_string()),
            Token::Pound => Some("#".to_string()),
            Token::DollarIdent(s) => Some(format!("${s}")),
            _ => None,
        })
        .collect()
}

/// Convenience function to parse a command from a string.
pub fn parse_line(input: &str) -> Result<Command, ParseError> {
    parse_command_text(input)
}

/// Parse a complete expression from text.
pub fn parse_expression(input: &str) -> Result<Expr, ParseError> {
    Parser::new(input).parse_full_expr()
}
