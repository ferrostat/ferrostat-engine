//! Full recursive-descent parser for Ferrostat command syntax.

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

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::new(&format!(
                "expected {}, found {}",
                expected,
                self.current()
            )))
        }
    }

    // Expr → OrExpr
    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    // OrExpr → AndExpr (| AndExpr)*
    pub fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while matches!(self.current(), Token::Pipe) {
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::Or, Box::new(rhs));
        }
        Ok(lhs)
    }

    // AndExpr → EqExpr (& EqExpr)*
    pub fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_eq()?;
        while matches!(self.current(), Token::Amp) {
            self.advance();
            let rhs = self.parse_eq()?;
            lhs = Expr::Binary(Box::new(lhs), BinOp::And, Box::new(rhs));
        }
        Ok(lhs)
    }

    // EqExpr → RelExpr ((== | != | ~=) RelExpr)*
    pub fn parse_eq(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_rel()?;
        loop {
            let op = match self.current() {
                Token::Eq => { self.advance(); BinOp::Eq }
                Token::Ne => { self.advance(); BinOp::Ne }
                _ => break,
            };
            let rhs = self.parse_rel()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    // RelExpr → AddExpr ((< | <= | > | >=) AddExpr)*
    pub fn parse_rel(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.current() {
                Token::Lt => { self.advance(); BinOp::Lt }
                Token::Le => { self.advance(); BinOp::Le }
                Token::Gt => { self.advance(); BinOp::Gt }
                Token::Ge => { self.advance(); BinOp::Ge }
                _ => break,
            };
            let rhs = self.parse_add()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    // AddExpr → MulExpr ((+ | -) MulExpr)*
    pub fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.current() {
                Token::Plus => { self.advance(); BinOp::Add }
                Token::Minus => { self.advance(); BinOp::Sub }
                _ => break,
            };
            let rhs = self.parse_mul()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    // MulExpr → PowExpr ((* | /) PowExpr)*
    pub fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_pow()?;
        loop {
            let op = match self.current() {
                Token::Star => { self.advance(); BinOp::Mul }
                Token::Slash => { self.advance(); BinOp::Div }
                _ => break,
            };
            let rhs = self.parse_pow()?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    // PowExpr → Unary (^ PowExpr)?
    pub fn parse_pow(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        if matches!(self.current(), Token::Caret) {
            self.advance();
            let rhs = self.parse_pow()?; // right associative
            lhs = Expr::Binary(Box::new(lhs), BinOp::Pow, Box::new(rhs));
        }
        Ok(lhs)
    }

    // Unary → (- | ! | ~) Unary | Primary
    pub fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.current() {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::Unary(UnaryOp::Neg, Box::new(operand)))
            }
            Token::Bang | Token::Tilde => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::Unary(UnaryOp::Not, Box::new(operand)))
            }
            _ => self.parse_primary(),
        }
    }

    // Primary → number | string | ident | ident(args) | (Expr) | Expr[Expr] | . | missing
    pub fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.current() {
            Token::Num(n) => { let v = *n; self.advance(); Ok(Expr::Number(v)) }
            Token::Str(s) => { let v = s.clone(); self.advance(); Ok(Expr::String(v)) }
            Token::Dot => { self.advance(); Ok(Expr::Missing) }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Function call?
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
            _ => Err(ParseError::new(&format!("unexpected token in expression: {}", self.current()))),
        }
    }

    /// Parse a full command line.
    pub fn parse_command(&mut self) -> Result<Command, ParseError> {
        let mut cmd = Command::default();
        cmd.raw = self.raw.clone();

        // Empty line
        if matches!(self.current(), Token::Eof) {
            return Ok(cmd);
        }

        // Verb
        if let Token::Ident(v) = self.current() {
            cmd.verb = v.to_lowercase().clone();
            self.advance();
        } else {
            return Err(ParseError::new("command must start with an identifier"));
        }

        // Collect tokens until we hit if/in/weight/using/[,/
        // But we need to handle `using` specially
        while !matches!(self.current(), Token::Eof) {
            // Check for option start
            if matches!(self.current(), Token::Comma) {
                self.advance();
                cmd.options = self.parse_options()?;
                break;
            }

            // Check for if
            if self.is_keyword("if") {
                
                self.advance();
                cmd.if_expr = Some(self.parse_expr()?);
                continue;
            }

            // Check for in
            if self.is_keyword("in") && !self.is_command_context() {
                
                self.advance();
                cmd.in_range = Some(self.parse_in_range()?);
                continue;
            }

            // Check for using
            if self.is_keyword("using") {
                self.advance();
                // Collect path tokens: handles /tmp/file.dta or "path" or path
                let mut path_parts = Vec::new();
                while !matches!(self.current(), Token::Eof) {
                    let t = self.current();
                    // Stop at known command-level delimiters
                    if matches!(t, Token::Comma | Token::LBracket | Token::Semicolon)
                        || self.is_keyword("if")
                        || self.is_keyword("in")
                        || self.is_keyword("using")
                    {
                        break;
                    }
                    path_parts.push(t.to_string());
                    self.advance();
                }
                let path = path_parts.join("").trim().trim_matches('"').to_string();
                if path.is_empty() {
                    return Err(ParseError::new("using requires a filename"));
                }
                cmd.using = Some(path);
                continue;
            }

            // Check for weight [aw=exp], [fw=exp], etc.
            if matches!(self.current(), Token::LBracket) {
                
                let weight = self.parse_weight()?;
                cmd.weight = Some(weight);
                continue;
            }

            // Check for end of command (semicolon or comment start)
            if matches!(self.current(), Token::Semicolon) {
                self.advance();
                break;
            }

            // Otherwise collect as raw arg token
            cmd.raw_args.push(self.current().clone());
            self.advance();
        }

        // Convert raw_args to string args (for backwards compat with simple commands)
        cmd.args = tokens_to_strings(&cmd.raw_args);

        Ok(cmd)
    }

    pub fn is_keyword(&self, kw: &str) -> bool {
        matches!(self.current(), Token::Ident(s) if s.eq_ignore_ascii_case(kw))
    }

    // Distinguish "in" as keyword vs "in" as variable name (rough heuristic)
    pub fn is_command_context(&self) -> bool {
        // Simple heuristic: if previous token before "in" was also an ident, 
        // and we're after a verb, it's probably a keyword
        false
    }

    pub fn parse_in_range(&mut self) -> Result<InRange, ParseError> {
        let start_tok = self.current().clone();
        self.advance();

        if matches!(self.current(), Token::Slash) {
            self.advance();
            let end_tok = self.current().clone();
            self.advance();
            let start = token_to_usize(&start_tok)?;
            let end = token_to_usize(&end_tok)?;
            return Ok(InRange::Range(start, end));
        }

        // Single number in
        let n = token_to_usize(&start_tok)?;
        Ok(InRange::Single(n))
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

        Ok(WeightSpec {
            weight_type: wt_type,
            expr,
        })
    }

    pub fn parse_options(&mut self) -> Result<Vec<(String, Option<Expr>)>, ParseError> {
        let mut opts = Vec::new();

        while !matches!(self.current(), Token::Eof) {
            if let Token::Ident(name) = self.current() {
                let name = name.clone();
                self.advance();

                // Option with sub-option in parens: legend(pos(6))
                if matches!(self.current(), Token::LParen) {
                    self.advance();
                    let sub_expr = self.parse_expr()?;
                    self.expect(&Token::RParen)?;
                    opts.push((name, Some(sub_expr)));
                } else if matches!(self.current(), Token::Equals) {
                    self.advance();
                    let val = self.parse_expr()?;
                    opts.push((name, Some(val)));
                } else {
                    opts.push((name, None));
                }
            } else {
                break;
            }

            if matches!(self.current(), Token::Comma) {
                self.advance();
            }
        }

        Ok(opts)
    }
}

fn token_to_usize(tok: &Token) -> Result<usize, ParseError> {
    match tok {
        Token::Num(n) => Ok(*n as usize),
        Token::Ident(s) => s.parse::<usize>().map_err(|_| ParseError::new("expected integer in in-range")),
        _ => Err(ParseError::new("expected integer in in-range")),
    }
}

fn tokens_to_strings(tokens: &[Token]) -> Vec<String> {
    tokens.iter()
        .filter_map(|t| match t {
            Token::Ident(s) => Some(s.clone()),
            Token::Str(s) => Some(format!("\"{}\"", s)),
            Token::Num(n) => Some(format!("{}", n)),
            Token::Dot => Some(".".to_string()),
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
            _ => None,
        })
        .collect()
}

/// Convenience function to parse a command from a string.
pub fn parse_line(input: &str) -> Result<Command, ParseError> {
    Parser::new(input).parse_command()
}
