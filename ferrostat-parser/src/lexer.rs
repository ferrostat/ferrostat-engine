//! Ferrostat full-featured lexer.
//!
//! Handles quoted strings, numbers, operators, punctuation, and language-specific tokens.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),      // variable/command names
    Str(String),        // quoted string
    Num(f64),           // numeric literal
    LParen,             // (
    RParen,             // )
    LBracket,           // [
    RBracket,           // ]
    LBrace,             // {
    RBrace,             // }
    Comma,              // ,
    Colon,              // :
    Semicolon,          // ;
    Equals,             // =
    Plus,               // +
    Minus,              // -
    Star,               // *
    Slash,              // /
    Caret,              // ^
    Lt,                 // <
    Gt,                 // >
    Le,                 // <=
    Ge,                 // >=
    Eq,                 // ==
    Ne,                 // != or ~=
    Amp,                // &
    Pipe,               // |
    Bang,               // !
    Tilde,              // ~
    Dot,                // .
    At,                 // @
    Pound,              // #
    Dollar,             // $
    Backtick,           // `
    Apostrophe,         // ' (single quote after backtick for macros)
    Percent,            // %
    ExtMissing(char),   // .a .. .z
    DollarIdent(String), // $macname or ${macname}
    BacktickExpr(String),// `macname'
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "{}", s),
            Token::Str(s) => write!(f, "\"{}\"", s),
            Token::Num(n) => write!(f, "{}", n),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Equals => write!(f, "="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Le => write!(f, "<="),
            Token::Ge => write!(f, ">="),
            Token::Eq => write!(f, "=="),
            Token::Ne => write!(f, "!="),
            Token::Amp => write!(f, "&"),
            Token::Pipe => write!(f, "|"),
            Token::Bang => write!(f, "!"),
            Token::Tilde => write!(f, "~"),
            Token::Dot => write!(f, "."),
            Token::At => write!(f, "@"),
            Token::Pound => write!(f, "#"),
            Token::Dollar => write!(f, "$"),
            Token::Backtick => write!(f, "`"),
            Token::Apostrophe => write!(f, "'"),
            Token::Percent => write!(f, "%"),
            Token::ExtMissing(c) => write!(f, ".{}", c),
            Token::DollarIdent(s) => write!(f, "${}", s),
            Token::BacktickExpr(s) => write!(f, "`{}'", s),
            Token::Eof => write!(f, "<EOF>"),
        }
    }
}

pub struct Lexer {
    input: String,
    pos: usize,
    current_char: Option<char>,
    /// tokens already scanned, returned before reading further input
    pending: Vec<Token>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let input = input.to_string();
        let current_char = input.chars().next();
        Self {
            input,
            pos: 0,
            current_char,
            pending: Vec::new(),
        }
    }

    fn advance(&mut self) {
        self.pos += self.current_char.map_or(0, |c| c.len_utf8());
        self.current_char = self.input[self.pos..].chars().next();
    }

    fn peek_char(&self) -> Option<char> {
        if let Some(c) = self.current_char {
            let next_pos = self.pos + c.len_utf8();
            self.input[next_pos..].chars().next()
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() {
                self.advance();
            } else if c == '*' && self.pos == 0 {
                // Comment at start of line
                self.skip_comment();
            } else if c == '/' && self.peek_char() == Some('*') {
                self.advance(); // /
                self.advance(); // *
                self.skip_block_comment();
            } else if c == '/' && self.peek_char() == Some('/') {
                self.skip_comment();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(c) = self.current_char {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) {
        while let Some(c) = self.current_char {
            if c == '*' && self.peek_char() == Some('/') {
                self.advance();
                self.advance();
                break;
            }
            self.advance();
        }
    }

    fn read_string(&mut self, quote: char) -> Token {
        self.advance(); // consume opening quote
        let mut s = String::new();
        while let Some(c) = self.current_char {
            if c == quote {
                self.advance(); // closing quote
                return Token::Str(s);
            }
            s.push(c);
            self.advance();
        }
        Token::Str(s) // unterminated — return what we have
    }

    /// Compound double quotes: `"..."' (may nest).
    fn read_compound_string(&mut self) -> Token {
        self.advance(); // `
        self.advance(); // "
        let mut s = String::new();
        let mut depth = 1;
        while let Some(c) = self.current_char {
            if c == '`' && self.peek_char() == Some('"') {
                depth += 1;
                s.push_str("`\"");
                self.advance();
                self.advance();
                continue;
            }
            if c == '"' && self.peek_char() == Some('\'') {
                depth -= 1;
                self.advance();
                self.advance();
                if depth == 0 {
                    return Token::Str(s);
                }
                s.push_str("\"'");
                continue;
            }
            s.push(c);
            self.advance();
        }
        Token::Str(s)
    }

    fn read_number(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.current_char {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+' {
                // Only consume -/+ if part of exponent
                if (c == '-' || c == '+') && !s.ends_with('e') && !s.ends_with('E') {
                    break;
                }
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        let n = s.parse::<f64>().unwrap_or(0.0);
        Token::Num(n)
    }

    /// Factor-variable names such as `2.group`, `1b.group`, `3o.x`.
    fn try_factor_ident(&mut self) -> Option<Token> {
        let rest: Vec<char> = self.input[self.pos..].chars().collect();
        let mut i = 0;
        while i < rest.len() && rest[i].is_ascii_digit() {
            i += 1;
        }
        while i < rest.len() && matches!(rest[i], 'b' | 'o' | 'n') {
            i += 1;
        }
        if i < rest.len() && rest[i] == '.' && rest.get(i + 1).map_or(false, |c| c.is_alphabetic() || *c == '_') {
            let mut j = i + 1;
            while j < rest.len() && (rest[j].is_alphanumeric() || rest[j] == '_' || rest[j] == '.' || rest[j] == '#') {
                j += 1;
            }
            let s: String = rest[..j].iter().collect();
            for _ in 0..j {
                self.advance();
            }
            return Some(Token::Ident(s));
        }
        None
    }

    fn read_ident(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' || c == '.' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        // date literals: td(15jan2001), tm(1960m1), tq(1967q1), ... take their
        // argument as raw text, returned as a string
        if matches!(s.as_str(), "td" | "tc" | "tC" | "tw" | "tm" | "tq" | "th" | "ty") && self.current_char == Some('(') {
            if let Some(close) = self.input[self.pos..].find(')') {
                let raw = self.input[self.pos + 1..self.pos + close].trim().to_string();
                for _ in 0..self.input[self.pos..self.pos + close + 1].chars().count() {
                    self.advance();
                }
                // returned in reverse order by next_token
                self.pending = vec![Token::RParen, Token::Str(raw), Token::LParen];
            }
        }
        Token::Ident(s)
    }

    fn read_macro(&mut self) -> Token {
        self.advance(); // $
        if self.current_char == Some('{') {
            self.advance(); // {
            let mut s = String::new();
            while let Some(c) = self.current_char {
                if c == '}' {
                    self.advance();
                    return Token::DollarIdent(s);
                }
                s.push(c);
                self.advance();
            }
            Token::DollarIdent(s)
        } else {
            let mut s = String::new();
            while let Some(c) = self.current_char {
                if c.is_alphanumeric() || c == '_' {
                    s.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            Token::DollarIdent(s)
        }
    }

    fn read_backtick_macro(&mut self) -> Token {
        self.advance(); // `
        let mut s = String::new();
        while let Some(c) = self.current_char {
            if c == '\'' {
                self.advance();
                return Token::BacktickExpr(s);
            }
            s.push(c);
            self.advance();
        }
        Token::BacktickExpr(s)
    }

    pub fn next_token(&mut self) -> Token {
        if let Some(t) = self.pending.pop() {
            return t;
        }
        self.skip_whitespace();
        match self.current_char {
            None => Token::Eof,
            Some(c) => match c {
                '"' => self.read_string(c),
                '\'' => { self.advance(); Token::Apostrophe }
                '`' if self.peek_char() == Some('"') => self.read_compound_string(),
                '(' => { self.advance(); Token::LParen }
                ')' => { self.advance(); Token::RParen }
                '[' => { self.advance(); Token::LBracket }
                ']' => { self.advance(); Token::RBracket }
                '{' => { self.advance(); Token::LBrace }
                '}' => { self.advance(); Token::RBrace }
                ',' => { self.advance(); Token::Comma }
                ':' => { self.advance(); Token::Colon }
                ';' => { self.advance(); Token::Semicolon }
                '+' => { self.advance(); Token::Plus }
                '*' => { self.advance(); Token::Star }
                '^' => { self.advance(); Token::Caret }
                '@' => { self.advance(); Token::At }
                '%' => { self.advance(); Token::Percent }
                '`' => self.read_backtick_macro(),
                '$' => self.read_macro(),
                '-' => {
                    self.advance();
                    if self.current_char == Some('-') && self.peek_char().map_or(false, |c| c.is_ascii_digit()) {
                        // Negative number
                        self.read_number()
                    } else {
                        Token::Minus
                    }
                }
                '<' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Le
                    } else {
                        Token::Lt
                    }
                }
                '>' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Ge
                    } else {
                        Token::Gt
                    }
                }
                '=' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Eq
                    } else {
                        Token::Equals
                    }
                }
                '!' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Ne
                    } else {
                        Token::Bang
                    }
                }
                '~' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Ne
                    } else {
                        Token::Tilde
                    }
                }
                '&' => { self.advance(); Token::Amp }
                '|' => { self.advance(); Token::Pipe }
                '/' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        Token::Ne
                    } else {
                        Token::Slash
                    }
                }
                '.' => {
                    self.advance();
                    if self.current_char.map_or(false, |c| c.is_ascii_digit()) {
                        // .5 number
                        let mut s = String::from(".");
                        while let Some(c2) = self.current_char {
                            if c2.is_ascii_digit() || c2 == 'e' || c2 == 'E' {
                                s.push(c2);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        let n = s.parse::<f64>().unwrap_or(0.0);
                        Token::Num(n)
                    } else if self.current_char.map_or(false, |c| c.is_ascii_lowercase())
                        && !self.peek_char().map_or(false, |c| c.is_alphanumeric() || c == '_')
                    {
                        let c2 = self.current_char.unwrap();
                        self.advance();
                        Token::ExtMissing(c2)
                    } else if self.current_char == Some('_') || self.current_char.map_or(false, |c| c.is_numeric()) {
                        // Missing value: .a or .1
                        // Keep as special dot+ident
                        let mut s = String::from(".");
                        while let Some(c2) = self.current_char {
                            if c2.is_alphanumeric() || c2 == '_' {
                                s.push(c2);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        Token::Ident(s)
                    } else {
                        Token::Dot
                    }
                }
                '#' => { self.advance(); Token::Pound }
                c if c.is_ascii_digit() => {
                    if let Some(tok) = self.try_factor_ident() {
                        tok
                    } else {
                        self.read_number()
                    }
                }
                c if c.is_alphabetic() || c == '_' => self.read_ident(),
                _ => {
                    self.advance();
                    self.next_token()
                }
            }
        }
    }
}

impl Iterator for Lexer {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let tok = self.next_token();
        if tok == Token::Eof {
            None
        } else {
            Some(tok)
        }
    }
}
