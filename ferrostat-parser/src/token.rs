//! Expression AST and command structures for the Ferrostat parser.

pub use crate::lexer::Token;

/// Parsed expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    String(String),
    Var(String),
    Missing,          // .
    MissingCode(u8),  // .a (1) .. .z (26)
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Call(String, Vec<Expr>), // function name, arguments
    Paren(Box<Expr>),
    Index(Box<Expr>, Box<Expr>), // expr[expr]
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not,
}

/// A single parsed command with all syntactic components decomposed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Command {
    pub verb: String,
    pub args: Vec<String>,          // positional arguments / varlist as strings
    pub raw_args: Vec<Token>,       // tokens before if/in/weight/using/options
    pub options: Vec<(String, Option<Expr>)>, // options with optional expressions
    pub if_expr: Option<Expr>,
    pub in_range: Option<InRange>,
    pub weight: Option<WeightSpec>,
    pub using: Option<String>,
    pub raw: String,
    /// Text between the verb and the first clause (varlist / arguments).
    pub main: String,
    /// Raw text of the `if` expression.
    pub if_raw: Option<String>,
    /// Options as written: name and the raw text inside its parentheses.
    pub opts: Vec<(String, Option<String>)>,
}

impl Command {
    /// Is option `name` present? `min` is the minimum abbreviation length.
    pub fn has_opt(&self, name: &str, min: usize) -> bool {
        self.find_opt(name, min).is_some()
    }

    /// Raw argument of option `name` (None if absent or given without parens).
    pub fn opt(&self, name: &str, min: usize) -> Option<&str> {
        self.find_opt(name, min).and_then(|(_, a)| a.as_deref())
    }

    fn find_opt(&self, name: &str, min: usize) -> Option<&(String, Option<String>)> {
        self.opts.iter().find(|(n, _)| {
            let n = n.to_lowercase();
            n.len() >= min.min(name.len()) && name.starts_with(n.as_str())
        })
    }
}

/// An `in` range specification.
#[derive(Debug, Clone, PartialEq)]
pub enum InRange {
    Single(usize),      // in 5
    Range(usize, usize), // in 5/10
    First(usize),       // in f/5 or in 1/5
    Last(usize),        // in -5/l (last five observations)
    /// General form; positive = from start (1-based), negative = from end (-1 = l).
    Rel(i64, i64),
}

impl InRange {
    /// Resolve to 0-based [start, end) for a dataset of `n` observations.
    pub fn resolve(&self, n: usize) -> Result<(usize, usize), String> {
        let pos = |v: i64| -> i64 { if v < 0 { n as i64 + v + 1 } else { v } };
        let (a, b) = match *self {
            InRange::Single(k) => (k as i64, k as i64),
            InRange::Range(a, b) => (a as i64, b as i64),
            InRange::First(k) => (1, k as i64),
            InRange::Last(k) => (n as i64 - k as i64 + 1, n as i64),
            InRange::Rel(a, b) => (pos(a), pos(b)),
        };
        if a < 1 || b > n as i64 || a > b {
            if n == 0 && a <= 1 {
                return Ok((0, 0));
            }
            return Err("Obs. nos. out of range".into());
        }
        Ok((a as usize - 1, b as usize))
    }
}

/// Weight specification.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightSpec {
    pub weight_type: String, // fw, aw, pw, iw
    pub expr: Expr,
}

/// Simple error type for parse failures.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    pub fn new(msg: &str) -> Self {
        Self { message: msg.to_string() }
    }
}
