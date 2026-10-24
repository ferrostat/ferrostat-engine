//! Expression evaluator for Ferrostat AST.
//!
//! Evaluates parsed expressions against a DataSet row.

use ferrostat_core::{DataSet, Value};
use ferrostat_parser::{BinOp, Expr, UnaryOp};

/// Result of evaluating an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalResult {
    Number(f64),
    String(String),
    Missing,
}

impl EvalResult {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            EvalResult::Number(n) => Some(*n),
            EvalResult::String(s) => s.parse::<f64>().ok(),
            EvalResult::Missing => None,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            EvalResult::Number(n) => *n != 0.0 && !n.is_nan(),
            EvalResult::String(s) => !s.is_empty(),
            EvalResult::Missing => false,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, EvalResult::Missing)
    }
}

/// Evaluate an expression for a given observation index.
pub fn eval_expr(expr: &Expr, ds: &DataSet, row: usize) -> Result<EvalResult, String> {
    match expr {
        Expr::Number(n) => Ok(EvalResult::Number(*n)),
        Expr::String(s) => Ok(EvalResult::String(s.clone())),
        Expr::Missing => Ok(EvalResult::Missing),
        Expr::Var(name) => {
            if let Some(var) = ds.get_var(name) {
                Ok(value_to_eval(&var.data[row]))
            } else {
                Err(format!("variable {} not found", name))
            }
        }
        Expr::Binary(lhs, op, rhs) => {
            let l = eval_expr(lhs, ds, row)?;
            let r = eval_expr(rhs, ds, row)?;

            // Handle missing propagation
            if l.is_missing() || r.is_missing() {
                return Ok(EvalResult::Missing);
            }

            match op {
                BinOp::Add => Ok(EvalResult::Number(l.as_f64().unwrap_or(0.0) + r.as_f64().unwrap_or(0.0))),
                BinOp::Sub => Ok(EvalResult::Number(l.as_f64().unwrap_or(0.0) - r.as_f64().unwrap_or(0.0))),
                BinOp::Mul => Ok(EvalResult::Number(l.as_f64().unwrap_or(0.0) * r.as_f64().unwrap_or(0.0))),
                BinOp::Div => {
                    let rv = r.as_f64().unwrap_or(1.0);
                    if rv == 0.0 {
                        Ok(EvalResult::Missing)
                    } else {
                        Ok(EvalResult::Number(l.as_f64().unwrap_or(0.0) / rv))
                    }
                }
                BinOp::Pow => Ok(EvalResult::Number(l.as_f64().unwrap_or(0.0).powf(r.as_f64().unwrap_or(0.0)))),
                BinOp::Eq => Ok(EvalResult::Number(if eval_eq(&l, &r) { 1.0 } else { 0.0 })),
                BinOp::Ne => Ok(EvalResult::Number(if !eval_eq(&l, &r) { 1.0 } else { 0.0 })),
                BinOp::Lt => Ok(EvalResult::Number(lt(&l, &r) as i32 as f64)),
                BinOp::Le => Ok(EvalResult::Number((lt(&l, &r) || eval_eq(&l, &r)) as i32 as f64)),
                BinOp::Gt => Ok(EvalResult::Number((!lt(&l, &r) && !eval_eq(&l, &r)) as i32 as f64)),
                BinOp::Ge => Ok(EvalResult::Number((!lt(&l, &r)) as i32 as f64)),
                BinOp::And => Ok(EvalResult::Number(if l.as_bool() && r.as_bool() { 1.0 } else { 0.0 })),
                BinOp::Or => Ok(EvalResult::Number(if l.as_bool() || r.as_bool() { 1.0 } else { 0.0 })),
            }
        }
        Expr::Unary(op, expr) => {
            let v = eval_expr(expr, ds, row)?;
            if v.is_missing() {
                return Ok(EvalResult::Missing);
            }
            match op {
                UnaryOp::Neg => Ok(EvalResult::Number(-v.as_f64().unwrap_or(0.0))),
                UnaryOp::Not => Ok(EvalResult::Number(if v.as_bool() { 0.0 } else { 1.0 })),
            }
        }
        Expr::Call(name, args) => {
            let mut arg_vals = Vec::with_capacity(args.len());
            for arg in args {
                arg_vals.push(eval_expr(arg, ds, row)?);
            }
            eval_function(name, &arg_vals)
        }
        Expr::Paren(expr) => eval_expr(expr, ds, row),
        Expr::Index(_base, _idx) => {
            // Array indexing not supported yet
            Ok(EvalResult::Missing)
        }
    }
}

fn value_to_eval(v: &Value) -> EvalResult {
    match v {
        Value::Double(n) => if n.is_nan() { EvalResult::Missing } else { EvalResult::Number(*n) },
        Value::Float(n) => if n.is_nan() { EvalResult::Missing } else { EvalResult::Number(*n as f64) },
        Value::Int(n) => EvalResult::Number(*n as f64),
        Value::Long(n) => EvalResult::Number(*n as f64),
        Value::Byte(n) => EvalResult::Number(*n as f64),
        Value::Str(s) => EvalResult::String(s.clone()),
        Value::Missing => EvalResult::Missing,
        Value::StrL(_, _) => EvalResult::Missing,
    }
}

fn eval_eq(a: &EvalResult, b: &EvalResult) -> bool {
    match (a, b) {
        (EvalResult::Number(x), EvalResult::Number(y)) => x == y,
        (EvalResult::String(x), EvalResult::String(y)) => x == y,
        _ => false,
    }
}

fn lt(a: &EvalResult, b: &EvalResult) -> bool {
    match (a, b) {
        (EvalResult::Number(x), EvalResult::Number(y)) => x < y,
        (EvalResult::String(x), EvalResult::String(y)) => x < y,
        _ => false,
    }
}

fn eval_function(name: &str, args: &[EvalResult]) -> Result<EvalResult, String> {
    let name = name.to_lowercase();
    match name.as_str() {
        "abs" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.abs()))
        }
        "exp" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.exp()))
        }
        "ln" | "log" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(1.0);
            if x <= 0.0 { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(x.ln())) }
        }
        "log10" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(1.0);
            if x <= 0.0 { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(x.log10())) }
        }
        "sqrt" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            if x < 0.0 { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(x.sqrt())) }
        }
        "sin" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.sin()))
        }
        "cos" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.cos()))
        }
        "tan" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.tan()))
        }
        "round" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.round()))
        }
        "floor" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.floor()))
        }
        "ceil" | "ceiling" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.ceil()))
        }
        "min" => {
            let vals: Vec<f64> = args.iter().filter_map(|a| a.as_f64()).collect();
            if vals.is_empty() { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(vals.into_iter().fold(f64::INFINITY, f64::min))) }
        }
        "max" => {
            let vals: Vec<f64> = args.iter().filter_map(|a| a.as_f64()).collect();
            if vals.is_empty() { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(vals.into_iter().fold(f64::NEG_INFINITY, f64::max))) }
        }
        "mean" => {
            let vals: Vec<f64> = args.iter().filter_map(|a| a.as_f64()).collect();
            let n = vals.len() as f64;
            if n == 0.0 { Ok(EvalResult::Missing) } else { Ok(EvalResult::Number(vals.iter().sum::<f64>() / n)) }
        }
        "sum" => {
            let vals: Vec<f64> = args.iter().filter_map(|a| a.as_f64()).collect();
            Ok(EvalResult::Number(vals.iter().sum()))
        }
        "sign" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }))
        }
        "int" => {
            let x = args.get(0).and_then(|a| a.as_f64()).unwrap_or(0.0);
            Ok(EvalResult::Number(x.trunc()))
        }
        _ => Err(format!("unknown function: {}", name)),
    }
}

pub fn expr_to_string(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(s) => Some(s.clone()),
        Expr::String(s) => Some(s.clone()),
        Expr::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

pub fn expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Number(n) => Some(*n),
        _ => None,
    }
}

/// Extract a simple varlist from an expression (handles comma-separated via nested Call if any).
pub fn expr_to_varlist(expr: &Expr) -> Vec<String> {
    let mut result = Vec::new();
    match expr {
        Expr::Var(s) => result.push(s.clone()),
        Expr::Call(name, args) if name == "list" || name == "varlist" => {
            for arg in args {
                if let Expr::Var(s) = arg {
                    result.push(s.clone());
                }
            }
        }
        _ => {}
    }
    result
}
