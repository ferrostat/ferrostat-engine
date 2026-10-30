//! Expressions of DSGE equations: numbers, model variables (current or
//! led with `F.`; `E(...)` marks an expectation and is transparent),
//! parameters in braces (`{beta}` or `{beta=0.96}` with an initial value),
//! + - * / ^, parentheses and exp, ln/log, sqrt, abs.

use crate::eval::rc_err;

#[derive(Clone, Debug)]
pub enum Ex {
    Num(f64),
    /// variable name, led one period
    Var(String, bool),
    Param(String),
    Neg(Box<Ex>),
    Bin(char, Box<Ex>, Box<Ex>),
    Fun(String, Box<Ex>),
}

/// A parameter referenced as `{name=value}`.
pub type Init = (String, f64);

struct P<'a> {
    s: &'a [u8],
    i: usize,
    inits: Vec<Init>,
    lead: bool,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn err(&self, m: &str) -> String {
        rc_err(198, format!("{m} in equation near '{}'", String::from_utf8_lossy(&self.s[self.i.min(self.s.len())..])))
    }
    fn expr(&mut self) -> Result<Ex, String> {
        let mut a = self.term()?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.i += 1;
            let b = self.term()?;
            a = Ex::Bin(op as char, Box::new(a), Box::new(b));
        }
        Ok(a)
    }
    fn term(&mut self) -> Result<Ex, String> {
        let mut a = self.unary()?;
        while let Some(op @ (b'*' | b'/')) = self.peek() {
            self.i += 1;
            let b = self.unary()?;
            a = Ex::Bin(op as char, Box::new(a), Box::new(b));
        }
        Ok(a)
    }
    fn unary(&mut self) -> Result<Ex, String> {
        match self.peek() {
            Some(b'-') => {
                self.i += 1;
                Ok(Ex::Neg(Box::new(self.unary()?)))
            }
            Some(b'+') => {
                self.i += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }
    fn power(&mut self) -> Result<Ex, String> {
        let a = self.atom()?;
        if self.peek() == Some(b'^') {
            self.i += 1;
            let b = self.unary()?;
            return Ok(Ex::Bin('^', Box::new(a), Box::new(b)));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Ex, String> {
        match self.peek() {
            Some(b'(') => {
                self.i += 1;
                let e = self.expr()?;
                if self.peek() != Some(b')') {
                    return Err(self.err("missing )"));
                }
                self.i += 1;
                Ok(e)
            }
            Some(b'{') => {
                let st = self.i + 1;
                let end = self.s[st..].iter().position(|&ch| ch == b'}').ok_or_else(|| self.err("missing }"))? + st;
                let inner = String::from_utf8_lossy(&self.s[st..end]).to_string();
                self.i = end + 1;
                let (name, init) = match inner.split_once('=') {
                    Some((n, v)) => (n.trim().to_string(), Some(v.trim().parse::<f64>().map_err(|_| rc_err(198, format!("invalid initial value in {{{inner}}}")))?)),
                    None => (inner.trim().to_string(), None),
                };
                if name.is_empty() || !name.chars().all(|ch| ch.is_alphanumeric() || ch == '_') {
                    return Err(rc_err(198, format!("invalid parameter name {{{inner}}}")));
                }
                if let Some(v) = init {
                    self.inits.push((name.clone(), v));
                }
                Ok(Ex::Param(name))
            }
            Some(ch) if ch.is_ascii_digit() || ch == b'.' => {
                let st = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.') {
                    self.i += 1;
                }
                if self.i < self.s.len() && (self.s[self.i] == b'e' || self.s[self.i] == b'E') {
                    let save = self.i;
                    self.i += 1;
                    if self.i < self.s.len() && (self.s[self.i] == b'+' || self.s[self.i] == b'-') {
                        self.i += 1;
                    }
                    if self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                            self.i += 1;
                        }
                    } else {
                        self.i = save;
                    }
                }
                let t = String::from_utf8_lossy(&self.s[st..self.i]).to_string();
                t.parse::<f64>().map(Ex::Num).map_err(|_| self.err("invalid number"))
            }
            Some(ch) if ch.is_ascii_alphabetic() || ch == b'_' => {
                let st = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_') {
                    self.i += 1;
                }
                let w = String::from_utf8_lossy(&self.s[st..self.i]).to_string();
                // F.x
                if (w == "F" || w == "f" || w == "F1") && self.s.get(self.i) == Some(&b'.') {
                    self.i += 1;
                    let st2 = self.i;
                    while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_') {
                        self.i += 1;
                    }
                    let v = String::from_utf8_lossy(&self.s[st2..self.i]).to_string();
                    if v.is_empty() {
                        return Err(self.err("variable expected after F."));
                    }
                    return Ok(Ex::Var(v, true));
                }
                if self.peek() == Some(b'(') {
                    self.i += 1;
                    let lw = w.to_lowercase();
                    let was = self.lead;
                    let e = self.expr()?;
                    self.lead = was;
                    if self.peek() != Some(b')') {
                        return Err(self.err("missing )"));
                    }
                    self.i += 1;
                    return match lw.as_str() {
                        "e" => Ok(e),
                        "exp" | "ln" | "log" | "sqrt" | "abs" => Ok(Ex::Fun(lw, Box::new(e))),
                        _ => Err(rc_err(198, format!("unknown function {w}()"))),
                    };
                }
                Ok(Ex::Var(w, false))
            }
            _ => Err(self.err("unexpected input")),
        }
    }
}

/// Parse one side of an equation; returns the tree and `{p=v}` initial values.
pub fn parse(text: &str) -> Result<(Ex, Vec<Init>), String> {
    let mut p = P { s: text.as_bytes(), i: 0, inits: Vec::new(), lead: false };
    let e = p.expr()?;
    if p.peek().is_some() {
        return Err(p.err("unexpected input"));
    }
    Ok((e, p.inits))
}

impl Ex {
    /// Names of variables (current or led) and parameters, in order of appearance.
    pub fn collect(&self, vars: &mut Vec<String>, pars: &mut Vec<String>) {
        match self {
            Ex::Num(_) => {}
            Ex::Var(v, _) => {
                if !vars.contains(v) {
                    vars.push(v.clone());
                }
            }
            Ex::Param(p) => {
                if !pars.contains(p) {
                    pars.push(p.clone());
                }
            }
            Ex::Neg(a) | Ex::Fun(_, a) => a.collect(vars, pars),
            Ex::Bin(_, a, b) => {
                a.collect(vars, pars);
                b.collect(vars, pars);
            }
        }
    }

    /// Led variables.
    pub fn leads(&self, out: &mut Vec<String>) {
        match self {
            Ex::Var(v, true) => {
                if !out.contains(v) {
                    out.push(v.clone());
                }
            }
            Ex::Neg(a) | Ex::Fun(_, a) => a.leads(out),
            Ex::Bin(_, a, b) => {
                a.leads(out);
                b.leads(out);
            }
            _ => {}
        }
    }

    /// Evaluate with `var(name, lead)` and `par(name)`.
    pub fn eval(&self, var: &dyn Fn(&str, bool) -> f64, par: &dyn Fn(&str) -> f64) -> f64 {
        match self {
            Ex::Num(v) => *v,
            Ex::Var(n, l) => var(n, *l),
            Ex::Param(p) => par(p),
            Ex::Neg(a) => -a.eval(var, par),
            Ex::Bin(op, a, b) => {
                let (x, y) = (a.eval(var, par), b.eval(var, par));
                match op {
                    '+' => x + y,
                    '-' => x - y,
                    '*' => x * y,
                    '/' => x / y,
                    _ => x.powf(y),
                }
            }
            Ex::Fun(f, a) => {
                let x = a.eval(var, par);
                match f.as_str() {
                    "exp" => x.exp(),
                    "ln" | "log" => x.ln(),
                    "sqrt" => x.sqrt(),
                    _ => x.abs(),
                }
            }
        }
    }
}
