//! Ferrostat full-featured lexer.
//!
//! Handles quoted strings, numbers, operators, punctuation, and Stata-specific tokens.

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
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let input = input.to_string();
        let current_char = input.chars().next();
        Self {
            input,
            pos: 0,
            current_char,
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
            } else if c == '\\' {
                self.advance();
                if let Some(next) = self.current_char {
                    match next {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        '\'' => s.push('\''),
                        _ => s.push(next),
                    }
                    self.advance();
                }
            } else {
                s.push(c);
                self.advance();
            }
        }
        Token::Str(s) // unterminated — return what we have
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
        self.skip_whitespace();
        match self.current_char {
            None => Token::Eof,
            Some(c) => match c {
                '"' | '\'' => self.read_string(c),
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
                    } else if self.current_char == Some('_') || self.current_char.map_or(false, |c| c.is_numeric()) {
                        // Stata missing value: .a or .1
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
                c if c.is_ascii_digit() => self.read_number(),
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
