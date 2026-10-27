//! Statements and expressions of the matrix language.

use std::rc::Rc;

use super::lexer::{Tok, Token};

#[derive(Debug, Clone, PartialEq)]
pub enum El {
    Real,
    Str,
    Complex,
    Numeric,
    Pointer,
    Trans,
    Void,
    Struct(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Org {
    Scalar,
    Vector,
    Row,
    Col,
    Matrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ty {
    pub el: El,
    pub org: Org,
}

impl Ty {
    pub fn text(&self) -> String {
        let el = match &self.el {
            El::Real => "real".to_string(),
            El::Str => "string".into(),
            El::Complex => "complex".into(),
            El::Numeric => "numeric".into(),
            El::Pointer => "pointer".into(),
            El::Trans => "transmorphic".into(),
            El::Void => "void".into(),
            El::Struct(s) => format!("struct {s}"),
        };
        let org = match self.org {
            Org::Scalar => "scalar",
            Org::Vector => "vector",
            Org::Row => "rowvector",
            Org::Col => "colvector",
            Org::Matrix => "matrix",
        };
        format!("{el} {org}")
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    Miss(u8),
    Str(String),
    Var(String),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Deref(Box<Expr>),
    Addr(Box<Expr>),
    Trans(Box<Expr>),
    Call(String, Vec<Expr>),
    /// `(*p)(args)`
    CallPtr(Box<Expr>, Vec<Expr>),
    Method(Box<Expr>, String, Vec<Expr>),
    Sub(Box<Expr>, Vec<Expr>),
    RangeSub(Box<Expr>, Box<Expr>),
    /// lhs, compound operator (`+` for `+=`) and value
    Assign(Box<Expr>, Option<&'static str>, Box<Expr>),
    IncDec(Box<Expr>, f64, bool),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    /// an omitted argument
    Omitted,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Option<Ty>,
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: String,
    pub ret: Option<Ty>,
    pub params: Vec<Param>,
    /// number of required arguments
    pub nreq: usize,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Expr(Expr),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    For(Option<Expr>, Option<Expr>, Option<Expr>, Box<Stmt>),
    While(Expr, Box<Stmt>),
    DoWhile(Box<Stmt>, Expr),
    Block(Vec<Stmt>),
    Break,
    Continue,
    Return(Option<Expr>),
    Decl(Ty, Vec<String>, bool),
    Func(Rc<FuncDef>),
    Cmd(String),
    Nop,
}

pub struct Parser<'a> {
    pub t: &'a [Token],
    pub p: usize,
}

type PResult<T> = Result<T, String>;

const ELTYPES: &[&str] = &["real", "string", "complex", "numeric", "pointer", "transmorphic", "void", "struct", "class"];
const ORGTYPES: &[&str] = &["scalar", "vector", "rowvector", "colvector", "matrix"];

impl<'a> Parser<'a> {
    pub fn new(t: &'a [Token]) -> Self {
        Parser { t, p: 0 }
    }

    pub fn done(&self) -> bool {
        self.p >= self.t.len()
    }

    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.p).map(|t| &t.tok)
    }

    fn peek_at(&self, k: usize) -> Option<&Tok> {
        self.t.get(self.p + k).map(|t| &t.tok)
    }

    fn is_op(&self, o: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(x)) if *x == o)
    }

    fn eat_op(&mut self, o: &str) -> bool {
        if self.is_op(o) {
            self.p += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, o: &str) -> PResult<()> {
        if self.eat_op(o) {
            Ok(())
        } else {
            Err(format!("'{o}' expected"))
        }
    }

    fn is_id(&self, w: &str) -> bool {
        matches!(self.peek(), Some(Tok::Id(x)) if x == w)
    }

    pub fn skip_nl(&mut self) {
        while matches!(self.peek(), Some(Tok::Nl)) {
            self.p += 1;
        }
    }

    /// Skip the rest of a failed statement.
    pub fn recover(&mut self) {
        while let Some(t) = self.peek() {
            let nl = *t == Tok::Nl;
            self.p += 1;
            if nl {
                break;
            }
        }
    }

    fn end_stmt(&mut self) -> PResult<()> {
        match self.peek() {
            None => Ok(()),
            Some(Tok::Nl) => {
                self.p += 1;
                Ok(())
            }
            Some(Tok::Op("}")) => Ok(()),
            _ => Err("invalid expression".into()),
        }
    }

    pub fn stmt(&mut self) -> PResult<Stmt> {
        self.skip_nl();
        let Some(tok) = self.peek().cloned() else { return Ok(Stmt::Nop) };
        match tok {
            Tok::Cmd(c) => {
                self.p += 1;
                self.end_stmt()?;
                Ok(Stmt::Cmd(c))
            }
            Tok::Op("{") => {
                self.p += 1;
                let mut body = Vec::new();
                loop {
                    self.skip_nl();
                    if self.eat_op("}") {
                        break;
                    }
                    if self.done() {
                        return Err("unexpected end of code (missing })".into());
                    }
                    body.push(self.stmt()?);
                }
                Ok(Stmt::Block(body))
            }
            Tok::Id(w) => match w.as_str() {
                "if" => {
                    self.p += 1;
                    self.expect("(")?;
                    let c = self.expr(true)?;
                    self.expect(")")?;
                    let then = self.stmt()?;
                    // look past newlines for else
                    let save = self.p;
                    self.skip_nl();
                    if self.is_id("else") {
                        self.p += 1;
                        let e = self.stmt()?;
                        return Ok(Stmt::If(c, Box::new(then), Some(Box::new(e))));
                    }
                    self.p = save;
                    Ok(Stmt::If(c, Box::new(then), None))
                }
                "for" => {
                    self.p += 1;
                    self.expect("(")?;
                    let a = if matches!(self.peek(), Some(Tok::Semi)) { None } else { Some(self.expr(true)?) };
                    self.semi()?;
                    let b = if matches!(self.peek(), Some(Tok::Semi)) { None } else { Some(self.expr(true)?) };
                    self.semi()?;
                    let c = if self.is_op(")") { None } else { Some(self.expr(true)?) };
                    self.expect(")")?;
                    let body = self.loop_body()?;
                    Ok(Stmt::For(a, b, c, Box::new(body)))
                }
                "while" => {
                    self.p += 1;
                    self.expect("(")?;
                    let c = self.expr(true)?;
                    self.expect(")")?;
                    let body = self.loop_body()?;
                    Ok(Stmt::While(c, Box::new(body)))
                }
                "do" => {
                    self.p += 1;
                    let body = self.stmt()?;
                    self.skip_nl();
                    if !self.is_id("while") {
                        return Err("while expected after do".into());
                    }
                    self.p += 1;
                    self.expect("(")?;
                    let c = self.expr(true)?;
                    self.expect(")")?;
                    self.end_stmt()?;
                    Ok(Stmt::DoWhile(Box::new(body), c))
                }
                "break" => {
                    self.p += 1;
                    self.end_stmt()?;
                    Ok(Stmt::Break)
                }
                "continue" => {
                    self.p += 1;
                    self.end_stmt()?;
                    Ok(Stmt::Continue)
                }
                "return" => {
                    self.p += 1;
                    if matches!(self.peek(), None | Some(Tok::Nl) | Some(Tok::Op("}"))) {
                        self.end_stmt()?;
                        return Ok(Stmt::Return(None));
                    }
                    let e = self.expr(true)?;
                    self.end_stmt()?;
                    Ok(Stmt::Return(Some(e)))
                }
                "end" if matches!(self.peek_at(1), None | Some(Tok::Nl)) => {
                    self.p += 1;
                    self.end_stmt()?;
                    Ok(Stmt::Cmd("end".into()))
                }
                "external" => {
                    self.p += 1;
                    let ty = self.ty()?.unwrap_or(Ty { el: El::Trans, org: Org::Matrix });
                    let names = self.names()?;
                    Ok(Stmt::Decl(ty, names, true))
                }
                "function" => self.funcdef(None),
                _ if ELTYPES.contains(&w.as_str()) || ORGTYPES.contains(&w.as_str()) => {
                    let ty = self.ty()?.ok_or("invalid declaration")?;
                    if self.is_id("function") {
                        return self.funcdef(Some(ty));
                    }
                    if matches!(self.peek(), Some(Tok::Id(_))) && matches!(self.peek_at(1), Some(Tok::Op("("))) {
                        return self.funcdef(Some(ty));
                    }
                    let names = self.names()?;
                    Ok(Stmt::Decl(ty, names, false))
                }
                _ => self.expr_stmt(),
            },
            _ => self.expr_stmt(),
        }
    }

    fn semi(&mut self) -> PResult<()> {
        if matches!(self.peek(), Some(Tok::Semi)) {
            self.p += 1;
            Ok(())
        } else {
            Err("';' expected".into())
        }
    }

    fn loop_body(&mut self) -> PResult<Stmt> {
        // `while (x) ;` style empty bodies
        if matches!(self.peek(), Some(Tok::Nl)) && matches!(self.peek_at(1), Some(Tok::Op("{"))) {
            self.p += 1;
        }
        self.stmt()
    }

    fn expr_stmt(&mut self) -> PResult<Stmt> {
        let e = self.expr(true)?;
        self.end_stmt()?;
        Ok(Stmt::Expr(e))
    }

    fn names(&mut self) -> PResult<Vec<String>> {
        let mut out = Vec::new();
        loop {
            match self.peek().cloned() {
                Some(Tok::Id(n)) => {
                    self.p += 1;
                    out.push(n);
                }
                _ => return Err("invalid declaration".into()),
            }
            if !self.eat_op(",") {
                break;
            }
        }
        self.end_stmt()?;
        Ok(out)
    }

    /// Optional type: eltype, orgtype or both.
    fn ty(&mut self) -> PResult<Option<Ty>> {
        let mut el = None;
        if let Some(Tok::Id(w)) = self.peek().cloned() {
            if ELTYPES.contains(&w.as_str()) {
                self.p += 1;
                el = Some(match w.as_str() {
                    "real" => El::Real,
                    "string" => El::Str,
                    "complex" => El::Complex,
                    "numeric" => El::Numeric,
                    "pointer" => {
                        if self.is_op("(") {
                            let mut d = 0;
                            loop {
                                match self.peek() {
                                    Some(Tok::Op("(")) => d += 1,
                                    Some(Tok::Op(")")) => d -= 1,
                                    None => break,
                                    _ => {}
                                }
                                self.p += 1;
                                if d == 0 {
                                    break;
                                }
                            }
                        }
                        El::Pointer
                    }
                    "void" => El::Void,
                    "struct" | "class" => match self.peek().cloned() {
                        Some(Tok::Id(n)) => {
                            self.p += 1;
                            El::Struct(n)
                        }
                        _ => return Err("invalid declaration".into()),
                    },
                    _ => El::Trans,
                });
            }
        }
        let mut org = None;
        if let Some(Tok::Id(w)) = self.peek().cloned() {
            if ORGTYPES.contains(&w.as_str()) {
                self.p += 1;
                org = Some(match w.as_str() {
                    "scalar" => Org::Scalar,
                    "vector" => Org::Vector,
                    "rowvector" => Org::Row,
                    "colvector" => Org::Col,
                    _ => Org::Matrix,
                });
            }
        }
        if el.is_none() && org.is_none() {
            return Ok(None);
        }
        Ok(Some(Ty { el: el.unwrap_or(El::Trans), org: org.unwrap_or(Org::Matrix) }))
    }

    fn funcdef(&mut self, ret: Option<Ty>) -> PResult<Stmt> {
        if self.is_id("function") {
            self.p += 1;
        }
        let name = match self.peek().cloned() {
            Some(Tok::Id(n)) => n,
            _ => return Err("function name expected".into()),
        };
        self.p += 1;
        self.expect("(")?;
        let mut params = Vec::new();
        let mut nreq = None;
        if !self.eat_op(")") {
            loop {
                if self.eat_op("|") {
                    nreq.get_or_insert(params.len());
                }
                let ty = self.ty()?;
                let pname = match self.peek().cloned() {
                    Some(Tok::Id(n)) => n,
                    _ => return Err("invalid argument declaration".into()),
                };
                self.p += 1;
                params.push(Param { name: pname, ty });
                if self.eat_op(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.skip_nl();
        if !self.is_op("{") {
            return Err("'{' expected".into());
        }
        let body = match self.stmt()? {
            Stmt::Block(b) => b,
            s => vec![s],
        };
        let nreq = nreq.unwrap_or(params.len());
        let ret = ret.filter(|t| !(t.el == El::Trans && t.org == Org::Matrix));
        Ok(Stmt::Func(Rc::new(FuncDef { name, ret, params, nreq, body })))
    }

    // ---- expressions: precedence climbing ----

    /// `allow_comma`: whether `,` is the join operator (false in argument lists).
    pub fn expr(&mut self, allow_comma: bool) -> PResult<Expr> {
        self.assign(allow_comma)
    }

    fn assign(&mut self, ac: bool) -> PResult<Expr> {
        let lhs = self.cond(ac)?;
        let op = match self.peek() {
            Some(Tok::Op("=")) => None,
            Some(Tok::Op("+=")) => Some("+"),
            Some(Tok::Op("-=")) => Some("-"),
            Some(Tok::Op("*=")) => Some("*"),
            Some(Tok::Op("/=")) => Some("/"),
            Some(Tok::Op("^=")) => Some("^"),
            _ => return Ok(lhs),
        };
        self.p += 1;
        let rhs = self.assign(ac)?;
        Ok(Expr::Assign(Box::new(lhs), op, Box::new(rhs)))
    }

    fn cond(&mut self, ac: bool) -> PResult<Expr> {
        let c = self.bin(0, ac)?;
        if self.eat_op("?") {
            let a = self.cond(ac)?;
            self.expect(":")?;
            let b = self.cond(ac)?;
            return Ok(Expr::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn bin(&mut self, level: usize, ac: bool) -> PResult<Expr> {
        const LEVELS: &[&[&str]] = &[
            &["||", "|", ":|"],
            &["&&", "&", ":&"],
            &["==", "!=", ">=", "<=", "<", ">", ":==", ":!=", ":>=", ":<=", ":<", ":>"],
            &["\\"],
            &[","],
            &["::", ".."],
            &["+", "-", ":+", ":-"],
            &["*", "/", ":*", ":/", "#"],
        ];
        if level >= LEVELS.len() {
            return self.unary();
        }
        let mut lhs = self.bin(level + 1, ac)?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if LEVELS[level].contains(o) => *o,
                _ => break,
            };
            if op == "," && !ac {
                break;
            }
            self.p += 1;
            let rhs = self.bin(level + 1, ac)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        if self.eat_op("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat_op("+") {
            return self.unary();
        }
        if self.eat_op("!") {
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        if self.eat_op("*") {
            return Ok(Expr::Deref(Box::new(self.unary()?)));
        }
        if self.eat_op("&") {
            return Ok(Expr::Addr(Box::new(self.unary()?)));
        }
        if self.eat_op("++") {
            return Ok(Expr::IncDec(Box::new(self.unary()?), 1.0, true));
        }
        if self.eat_op("--") {
            return Ok(Expr::IncDec(Box::new(self.unary()?), -1.0, true));
        }
        self.power()
    }

    fn power(&mut self) -> PResult<Expr> {
        let base = self.postfix()?;
        if self.is_op("^") || self.is_op(":^") {
            let op = if self.eat_op("^") { "^" } else {
                self.p += 1;
                ":^"
            };
            // right operand may carry a sign
            let rhs = if self.eat_op("-") { Expr::Neg(Box::new(self.power()?)) } else { self.power()? };
            return Ok(Expr::Bin(op, Box::new(base), Box::new(rhs)));
        }
        Ok(base)
    }

    fn starts_primary(&self) -> bool {
        matches!(self.peek(), Some(Tok::Id(_)) | Some(Tok::Num(_)) | Some(Tok::Op("(")))
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            if self.eat_op("'") {
                e = Expr::Trans(Box::new(e));
                // X'Y means X'*Y
                if self.starts_primary() {
                    let rhs = self.postfix()?;
                    e = Expr::Bin("*", Box::new(e), Box::new(rhs));
                }
                continue;
            }
            if self.eat_op("[|") {
                let idx = self.expr(true)?;
                self.expect("|]")?;
                e = Expr::RangeSub(Box::new(e), Box::new(idx));
                continue;
            }
            if self.eat_op("[") {
                let mut idx = Vec::new();
                loop {
                    idx.push(self.expr(false)?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect("]")?;
                e = Expr::Sub(Box::new(e), idx);
                continue;
            }
            if self.is_op(".") || self.is_op("->") {
                self.p += 1;
                let name = match self.peek().cloned() {
                    Some(Tok::Id(n)) => n,
                    _ => return Err("member name expected".into()),
                };
                self.p += 1;
                let args = if self.eat_op("(") { self.args()? } else { Vec::new() };
                e = Expr::Method(Box::new(e), name, args);
                continue;
            }
            if self.is_op("(") && matches!(e, Expr::Deref(_)) {
                self.p += 1;
                let args = self.args()?;
                e = Expr::CallPtr(Box::new(e), args);
                continue;
            }
            if self.eat_op("++") {
                e = Expr::IncDec(Box::new(e), 1.0, false);
                continue;
            }
            if self.eat_op("--") {
                e = Expr::IncDec(Box::new(e), -1.0, false);
                continue;
            }
            break;
        }
        Ok(e)
    }

    fn args(&mut self) -> PResult<Vec<Expr>> {
        let mut args = Vec::new();
        if self.eat_op(")") {
            return Ok(args);
        }
        loop {
            if self.is_op(",") || self.is_op(")") {
                args.push(Expr::Omitted);
            } else {
                args.push(self.expr(false)?);
            }
            if self.eat_op(")") {
                break;
            }
            self.expect(",")?;
        }
        Ok(args)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let Some(t) = self.peek().cloned() else { return Err("invalid expression".into()) };
        self.p += 1;
        Ok(match t {
            Tok::Num(v) => {
                // imaginary unit suffix is not supported; `2i` is an error
                Expr::Num(v)
            }
            Tok::Miss(c) => Expr::Miss(c),
            Tok::Str(s) => Expr::Str(s),
            Tok::Id(n) => {
                if self.eat_op("(") {
                    let args = self.args()?;
                    Expr::Call(n, args)
                } else {
                    Expr::Var(n)
                }
            }
            Tok::Op("(") => {
                let e = self.expr(true)?;
                self.expect(")")?;
                // `(*f)(x)`
                if self.is_op("(") && matches!(e, Expr::Deref(_)) {
                    self.p += 1;
                    let args = self.args()?;
                    return Ok(Expr::CallPtr(Box::new(e), args));
                }
                e
            }
            _ => {
                self.p -= 1;
                return Err("invalid expression".into());
            }
        })
    }
}
