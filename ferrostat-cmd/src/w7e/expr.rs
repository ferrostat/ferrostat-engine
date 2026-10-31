//! Compiled scalar expressions for Bayesian models.
//!
//! Model expressions (substitutable likelihood expressions, `llf()`,
//! prior arguments, `density()`) mix data variables, model parameters
//! written in braces (`{b0}`, `{y:x}`) and linear forms of equations
//! (`{xb:}`). They are evaluated many thousands of times by the sampler,
//! so they are compiled once into a small tree that reads parameter
//! values and data columns directly.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_parser::{BinOp, Expr, UnaryOp};

use crate::dist;
use crate::eval::rc_err;

/// A compiled expression.
#[derive(Clone, Debug)]
pub enum CE {
    Num(f64),
    /// data column (index into the model's column store)
    Col(usize),
    /// scalar parameter
    Par(usize),
    /// linear form of equation k for the current observation
    Xb(usize),
    Neg(Box<CE>),
    Not(Box<CE>),
    Bin(Box<CE>, BinOp, Box<CE>),
    Call(Func, Vec<CE>),
}

#[derive(Clone, Copy, Debug)]
pub enum Func {
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
    InvLogit,
    Logit,
    Normal,
    LnNormal,
    NormalDen,
    LnNormalDen,
    LnGamma,
    Digamma,
    Sin,
    Cos,
    Tan,
    Atan,
    Tanh,
    Min,
    Max,
    Cond,
    Floor,
    Ceil,
    Round,
    Int,
    Missing,
    InvCloglog,
    Cloglog,
    LnFactorial,
    Comb,
    Ln1p,
    Ln1m,
    Expm1,
    Sign,
    Atanh,
    LnPoisson,
    LnBinomial,
    LnIGammaDen,
    LnGammaDen,
    LnBetaDen,
    LnTDen,
    LnMvNormalDen,
}

/// Values available while evaluating.
pub struct Ctx<'a> {
    pub cols: &'a [Vec<f64>],
    pub theta: &'a [f64],
    pub xb: &'a [f64],
    pub row: usize,
}

fn func_of(name: &str) -> Option<Func> {
    Some(match name {
        "exp" => Func::Exp,
        "ln" | "log" => Func::Ln,
        "log10" => Func::Log10,
        "sqrt" => Func::Sqrt,
        "abs" => Func::Abs,
        "invlogit" => Func::InvLogit,
        "logit" => Func::Logit,
        "normal" | "norm" => Func::Normal,
        "lnnormal" => Func::LnNormal,
        "normalden" => Func::NormalDen,
        "lnnormalden" => Func::LnNormalDen,
        "lngamma" => Func::LnGamma,
        "digamma" => Func::Digamma,
        "sin" => Func::Sin,
        "cos" => Func::Cos,
        "tan" => Func::Tan,
        "atan" => Func::Atan,
        "tanh" => Func::Tanh,
        "min" => Func::Min,
        "max" => Func::Max,
        "cond" => Func::Cond,
        "floor" => Func::Floor,
        "ceil" => Func::Ceil,
        "round" => Func::Round,
        "int" | "trunc" => Func::Int,
        "missing" | "mi" => Func::Missing,
        "invcloglog" => Func::InvCloglog,
        "cloglog" => Func::Cloglog,
        "lnfactorial" => Func::LnFactorial,
        "comb" => Func::Comb,
        "ln1p" | "log1p" => Func::Ln1p,
        "ln1m" | "log1m" => Func::Ln1m,
        "expm1" => Func::Expm1,
        "sign" => Func::Sign,
        "atanh" => Func::Atanh,
        "lnpoissonden" => Func::LnPoisson,
        "lnbinomialden" => Func::LnBinomial,
        "lnigammaden" => Func::LnIGammaDen,
        "lngammaden" => Func::LnGammaDen,
        "lnbetaden" => Func::LnBetaDen,
        "lntden" => Func::LnTDen,
        "lnmvnormalden" => Func::LnMvNormalDen,
        _ => return None,
    })
}

/// Resolves the names met while compiling.
pub trait Resolver {
    /// a `{...}` reference (text inside the braces)
    fn brace(&mut self, text: &str) -> Result<CE, String>;
    /// a bare name: data variable or scalar
    fn name(&mut self, name: &str) -> Result<CE, String>;
}

/// Replace `{...}` references by placeholder identifiers; returns the new
/// text and the brace contents in order.
pub fn extract_braces(s: &str) -> (String, Vec<String>) {
    let mut out = String::new();
    let mut refs = Vec::new();
    let mut chars = s.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if c == '"' {
            in_str = !in_str;
            out.push(c);
            continue;
        }
        if c == '{' && !in_str {
            let mut inner = String::new();
            let mut depth = 1;
            for d in chars.by_ref() {
                if d == '{' {
                    depth += 1;
                } else if d == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                inner.push(d);
            }
            out.push_str(&format!(" __bR{}_ ", refs.len()));
            refs.push(inner.trim().to_string());
            continue;
        }
        out.push(c);
    }
    (out, refs)
}

/// Compile expression text.
pub fn compile(text: &str, r: &mut dyn Resolver) -> Result<CE, String> {
    let (t, refs) = extract_braces(text);
    let e = ferrostat_parser::parser::parse_expression(t.trim()).map_err(|e| rc_err(198, format!("invalid expression {text}: {}", e.message)))?;
    let mut resolved: Vec<Option<CE>> = vec![None; refs.len()];
    for (k, rf) in refs.iter().enumerate() {
        resolved[k] = Some(r.brace(rf)?);
    }
    conv(&e, r, &resolved)
}

fn conv(e: &Expr, r: &mut dyn Resolver, refs: &[Option<CE>]) -> Result<CE, String> {
    Ok(match e {
        Expr::Number(n) => CE::Num(*n),
        Expr::Missing | Expr::MissingCode(_) => CE::Num(SYSMISS),
        Expr::String(_) => return Err(rc_err(109, "type mismatch: strings not allowed in model expressions")),
        Expr::Paren(x) => conv(x, r, refs)?,
        Expr::Var(name) => {
            if let Some(k) = name.strip_prefix("__bR").and_then(|s| s.strip_suffix('_')).and_then(|s| s.parse::<usize>().ok()) {
                refs[k].clone().unwrap()
            } else if name == "_pi" {
                CE::Num(std::f64::consts::PI)
            } else {
                r.name(name)?
            }
        }
        Expr::Unary(UnaryOp::Neg, x) => CE::Neg(Box::new(conv(x, r, refs)?)),
        Expr::Unary(UnaryOp::Not, x) => CE::Not(Box::new(conv(x, r, refs)?)),
        Expr::Binary(a, op, b) => CE::Bin(Box::new(conv(a, r, refs)?), op.clone(), Box::new(conv(b, r, refs)?)),
        Expr::Call(name, args) => {
            let f = func_of(&name.to_lowercase()).ok_or_else(|| rc_err(133, format!("unknown function {name}() in model expression")))?;
            let a = args.iter().map(|x| conv(x, r, refs)).collect::<Result<Vec<_>, _>>()?;
            CE::Call(f, a)
        }
        Expr::Index(..) => return Err(rc_err(198, "subscripts not allowed in model expressions")),
    })
}

impl CE {
    pub fn eval(&self, c: &Ctx) -> f64 {
        match self {
            CE::Num(v) => *v,
            CE::Col(j) => c.cols[*j][c.row],
            CE::Par(j) => c.theta[*j],
            CE::Xb(k) => c.xb[*k],
            CE::Neg(x) => {
                let v = x.eval(c);
                if is_missing(v) { SYSMISS } else { -v }
            }
            CE::Not(x) => {
                if x.eval(c) != 0.0 { 0.0 } else { 1.0 }
            }
            CE::Bin(a, op, b) => {
                let x = a.eval(c);
                let y = b.eval(c);
                bin(x, op, y)
            }
            CE::Call(f, a) => call(*f, a, c),
        }
    }

    /// Parameters this expression reads.
    pub fn params(&self, out: &mut Vec<usize>) {
        match self {
            CE::Par(j) => out.push(*j),
            CE::Neg(x) | CE::Not(x) => x.params(out),
            CE::Bin(a, _, b) => {
                a.params(out);
                b.params(out);
            }
            CE::Call(_, a) => a.iter().for_each(|x| x.params(out)),
            _ => {}
        }
    }

    /// Equations whose linear form this expression reads.
    pub fn xbs(&self, out: &mut Vec<usize>) {
        match self {
            CE::Xb(k) => out.push(*k),
            CE::Neg(x) | CE::Not(x) => x.xbs(out),
            CE::Bin(a, _, b) => {
                a.xbs(out);
                b.xbs(out);
            }
            CE::Call(_, a) => a.iter().for_each(|x| x.xbs(out)),
            _ => {}
        }
    }

    /// Does the expression read data (varies over observations)?
    pub fn uses_data(&self) -> bool {
        match self {
            CE::Col(_) | CE::Xb(_) => true,
            CE::Neg(x) | CE::Not(x) => x.uses_data(),
            CE::Bin(a, _, b) => a.uses_data() || b.uses_data(),
            CE::Call(_, a) => a.iter().any(|x| x.uses_data()),
            _ => false,
        }
    }

    /// Constant value, when the expression has no parameters or data.
    pub fn constant(&self) -> Option<f64> {
        let mut p = Vec::new();
        self.params(&mut p);
        if p.is_empty() && !self.uses_data() {
            Some(self.eval(&Ctx { cols: &[], theta: &[], xb: &[], row: 0 }))
        } else {
            None
        }
    }
}

fn bin(x: f64, op: &BinOp, y: f64) -> f64 {
    let m = is_missing(x) || is_missing(y);
    let b = |v: bool| if v { 1.0 } else { 0.0 };
    match op {
        BinOp::Add => if m { SYSMISS } else { x + y },
        BinOp::Sub => if m { SYSMISS } else { x - y },
        BinOp::Mul => if m { SYSMISS } else { x * y },
        BinOp::Div => if m || y == 0.0 { SYSMISS } else { x / y },
        BinOp::Pow => {
            if m {
                SYSMISS
            } else {
                let v = x.powf(y);
                if v.is_finite() { v } else { SYSMISS }
            }
        }
        BinOp::Eq => b(x == y),
        BinOp::Ne => b(x != y),
        BinOp::Lt => b(x < y),
        BinOp::Le => b(x <= y),
        BinOp::Gt => b(x > y),
        BinOp::Ge => b(x >= y),
        BinOp::And => b(x != 0.0 && y != 0.0),
        BinOp::Or => b(x != 0.0 || y != 0.0),
    }
}

fn fin(v: f64) -> f64 {
    if v.is_finite() { v } else { SYSMISS }
}

fn call(f: Func, a: &[CE], c: &Ctx) -> f64 {
    let v: Vec<f64> = a.iter().map(|x| x.eval(c)).collect();
    if !matches!(f, Func::Cond | Func::Missing | Func::Min | Func::Max) && v.iter().any(|x| is_missing(*x)) {
        return SYSMISS;
    }
    let x = v.first().copied().unwrap_or(SYSMISS);
    let arg = |k: usize| v.get(k).copied().unwrap_or(SYSMISS);
    match f {
        Func::Exp => fin(x.exp()),
        Func::Ln => if x > 0.0 { x.ln() } else { SYSMISS },
        Func::Log10 => if x > 0.0 { x.log10() } else { SYSMISS },
        Func::Sqrt => if x >= 0.0 { x.sqrt() } else { SYSMISS },
        Func::Abs => x.abs(),
        Func::InvLogit => dist::invlogit(x),
        Func::Logit => if x > 0.0 && x < 1.0 { (x / (1.0 - x)).ln() } else { SYSMISS },
        Func::Normal => dist::normal(x),
        Func::LnNormal => dist::lnnormal(x),
        Func::NormalDen => {
            if v.len() >= 3 {
                dist::normalden((x - arg(1)) / arg(2)) / arg(2)
            } else if v.len() == 2 {
                dist::normalden(x / arg(1)) / arg(1)
            } else {
                dist::normalden(x)
            }
        }
        Func::LnNormalDen => {
            if v.len() >= 3 {
                dist::lnnormalden((x - arg(1)) / arg(2)) - arg(2).ln()
            } else if v.len() == 2 {
                dist::lnnormalden(x / arg(1)) - arg(1).ln()
            } else {
                dist::lnnormalden(x)
            }
        }
        Func::LnGamma => if x > 0.0 { dist::lngamma(x) } else { SYSMISS },
        Func::Digamma => dist::digamma(x),
        Func::Sin => x.sin(),
        Func::Cos => x.cos(),
        Func::Tan => x.tan(),
        Func::Atan => x.atan(),
        Func::Tanh => x.tanh(),
        Func::Atanh => fin(x.atanh()),
        Func::Min => v.iter().copied().filter(|x| !is_missing(*x)).fold(SYSMISS, |a, b| if is_missing(a) || b < a { b } else { a }),
        Func::Max => v.iter().copied().filter(|x| !is_missing(*x)).fold(SYSMISS, |a, b| if is_missing(a) || b > a { b } else { a }),
        Func::Cond => {
            if is_missing(x) && v.len() > 3 {
                arg(3)
            } else if x != 0.0 {
                arg(1)
            } else {
                arg(2)
            }
        }
        Func::Floor => x.floor(),
        Func::Ceil => x.ceil(),
        Func::Round => {
            let u = if v.len() > 1 { arg(1) } else { 1.0 };
            (x / u).round() * u
        }
        Func::Int => x.trunc(),
        Func::Missing => if v.iter().any(|x| is_missing(*x)) { 1.0 } else { 0.0 },
        Func::InvCloglog => 1.0 - (-x.exp()).exp(),
        Func::Cloglog => if x > 0.0 && x < 1.0 { (-(1.0 - x).ln()).ln() } else { SYSMISS },
        Func::LnFactorial => dist::lnfactorial(x),
        Func::Comb => dist::comb(x, arg(1)),
        Func::Ln1p => fin(x.ln_1p()),
        Func::Ln1m => fin((-x).ln_1p()),
        Func::Expm1 => fin(x.exp_m1()),
        Func::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        // lnpoissonden(mu, k)
        Func::LnPoisson => if x > 0.0 { arg(1) * x.ln() - x - dist::lnfactorial(arg(1)) } else if arg(1) == 0.0 { 0.0 } else { SYSMISS },
        // lnbinomialden(n, k, p)
        Func::LnBinomial => {
            let (n, k, p) = (x, arg(1), arg(2));
            fin(dist::lnfactorial(n) - dist::lnfactorial(k) - dist::lnfactorial(n - k) + k * p.ln() + (n - k) * (1.0 - p).ln())
        }
        // lnigammaden(a, b, x), lngammaden(a, b, g, x), lnbetaden(a, b, x), lntden(df, t)
        Func::LnIGammaDen => fin(crate::w7e::prior::ln_igamma(arg(2), x, arg(1))),
        Func::LnGammaDen => fin(crate::w7e::prior::ln_gamma(arg(3) - arg(2), x, arg(1))),
        Func::LnBetaDen => fin(crate::w7e::prior::ln_beta(arg(2), x, arg(1))),
        Func::LnTDen => fin(dist::tden(x, arg(1)).ln()),
        Func::LnMvNormalDen => SYSMISS,
    }
}
