//! Expression AST and command structures for the Ferrostat parser.

pub use crate::lexer::Token;

/// Parsed expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    String(String),
    Var(String),
    Missing,          // .
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
}

/// An `in` range specification.
#[derive(Debug, Clone, PartialEq)]
pub enum InRange {
    Single(usize),      // in 5
    Range(usize, usize), // in 5/10
    First(usize),       // in f/5 or in 1/5
    Last(usize),        // in 5/l or in 5/-1
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
