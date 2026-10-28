//! The `matrix` command family, `mkmat` and `svmat`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType};
use ferrostat_parser::parser::{find_top_level, parse_expression, split_top_level_words};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::varlist;

// ---------------------------------------------------------------------------
// Matrix expressions
// ---------------------------------------------------------------------------

struct MParser<'a> {
    s: Vec<char>,
    pos: usize,
    ds: &'a DataSet,
    st: &'a Store,
}

fn conform() -> String {
    rc_err(503, "conformability error")
}

impl<'a> MParser<'a> {
    fn peek(&self) -> Option<char> {
        let mut p = self.pos;
        while p < self.s.len() && self.s[p].is_whitespace() {
            p += 1;
        }
        self.s.get(p).copied()
    }

    fn skip_ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: char) -> bool {
        self.skip_ws();
        if self.s.get(self.pos) == Some(&c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn rowjoin(&mut self) -> Result<Matrix, String> {
        let mut m = self.coljoin()?;
        while self.peek() == Some('\\') {
            self.eat('\\');
            let b = self.coljoin()?;
            m = vjoin(&m, &b)?;
        }
        Ok(m)
    }

    fn coljoin(&mut self) -> Result<Matrix, String> {
        let mut m = self.add()?;
        while self.peek() == Some(',') {
            self.eat(',');
            let b = self.add()?;
            m = hjoin(&m, &b)?;
        }
        Ok(m)
    }

    fn add(&mut self) -> Result<Matrix, String> {
        let mut m = self.mul()?;
        loop {
            match self.peek() {
                Some('+') => {
                    self.eat('+');
                    let b = self.mul()?;
                    m = if m.rows == 1 && m.cols == 1 && (b.rows > 1 || b.cols > 1) { Err(conform())? } else { m.add(&b).map_err(|_| conform())? };
                }
                Some('-') => {
                    self.eat('-');
                    let b = self.mul()?;
                    m = m.sub(&b).map_err(|_| conform())?;
                }
                _ => return Ok(m),
            }
        }
    }

    fn mul(&mut self) -> Result<Matrix, String> {
        let mut m = self.unary()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.eat('*');
                    let b = self.unary()?;
                    m = if m.rows == 1 && m.cols == 1 {
                        b.scale(m.get(0, 0))
                    } else if b.rows == 1 && b.cols == 1 {
                        m.scale(b.get(0, 0))
                    } else {
                        m.mul(&b).map_err(|_| conform())?
                    };
                }
                Some('/') => {
                    self.eat('/');
                    let b = self.unary()?;
                    if b.rows != 1 || b.cols != 1 {
                        return Err(conform());
                    }
                    m = m.scale(1.0 / b.get(0, 0));
                }
                Some('#') => {
                    self.eat('#');
                    let b = self.unary()?;
                    m = kron(&m, &b);
                }
                _ => return Ok(m),
            }
        }
    }

    fn unary(&mut self) -> Result<Matrix, String> {
        if self.peek() == Some('-') {
            self.eat('-');
            return Ok(self.unary()?.scale(-1.0));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Matrix, String> {
        let mut m = self.primary()?;
        loop {
            self.skip_ws();
            match self.s.get(self.pos) {
                Some('\'') => {
                    self.pos += 1;
                    m = m.transpose();
                }
                Some('[') => {
                    self.pos += 1;
                    let start = self.pos;
                    let mut depth = 1;
                    while self.pos < self.s.len() && depth > 0 {
                        match self.s[self.pos] {
                            '[' => depth += 1,
                            ']' => depth -= 1,
                            _ => {}
                        }
                        self.pos += 1;
                    }
                    let inner: String = self.s[start..self.pos - 1].iter().collect();
                    m = self.subscript(&m, &inner)?;
                }
                _ => return Ok(m),
            }
        }
    }

    fn scalar(&self, text: &str) -> Result<f64, String> {
        let e = parse_expression(text).map_err(|e| rc_err(198, e.message))?;
        Env::with_store(self.ds, self.st).eval(&e, 0)?.num()
    }

    fn range(&self, text: &str, n: usize, names: &[String]) -> Result<(usize, usize), String> {
        let t = text.trim();
        if t.is_empty() {
            return Ok((0, n));
        }
        if let Some(stripped) = t.strip_prefix('"') {
            let name = stripped.trim_end_matches('"');
            let i = names.iter().position(|x| x == name).ok_or_else(|| rc_err(111, format!("{name} not found")))?;
            return Ok((i, i + 1));
        }
        if let Some((a, b)) = t.split_once("..") {
            let lo = if a.trim().is_empty() { 1.0 } else { self.scalar(a)? };
            let b = b.trim_start_matches('.');
            let hi = if b.trim().is_empty() { n as f64 } else { self.scalar(b)? };
            if lo < 1.0 || hi > n as f64 || lo > hi {
                return Err(rc_err(503, "subscript invalid"));
            }
            return Ok((lo as usize - 1, hi as usize));
        }
        let i = self.scalar(t)?;
        if i < 1.0 || i > n as f64 {
            return Err(rc_err(503, "subscript invalid"));
        }
        Ok((i as usize - 1, i as usize))
    }

    fn subscript(&self, m: &Matrix, inner: &str) -> Result<Matrix, String> {
        let (r, c) = match find_top_level(inner, ',') {
            Some(i) => (&inner[..i], &inner[i + 1..]),
            None => (inner, ""),
        };
        let (r0, r1) = self.range(r, m.rows, &m.rownames)?;
        let (c0, c1) = if c.trim().is_empty() && find_top_level(inner, ',').is_none() { (0, m.cols) } else { self.range(c, m.cols, &m.colnames)? };
        let mut out = Matrix::zeros(r1 - r0, c1 - c0);
        for i in r0..r1 {
            for j in c0..c1 {
                out[(i - r0, j - c0)] = m[(i, j)];
            }
        }
        out.rownames = m.rownames[r0..r1].to_vec();
        out.colnames = m.colnames[c0..c1].to_vec();
        Ok(out)
    }

    fn ident(&mut self) -> String {
        self.skip_ws();
        let mut s = String::new();
        while self.pos < self.s.len() && (self.s[self.pos].is_alphanumeric() || self.s[self.pos] == '_' || self.s[self.pos] == '.') {
            s.push(self.s[self.pos]);
            self.pos += 1;
        }
        s
    }

    /// Raw text of a parenthesised argument list, split at top-level commas.
    fn args(&mut self) -> Result<Vec<String>, String> {
        if !self.eat('(') {
            return Err(rc_err(198, "( expected"));
        }
        let start = self.pos;
        let mut depth = 1;
        let mut in_str = false;
        while self.pos < self.s.len() && depth > 0 {
            let c = self.s[self.pos];
            if c == '"' {
                in_str = !in_str;
            }
            if !in_str {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
            }
            self.pos += 1;
        }
        let inner: String = self.s[start..self.pos - 1].iter().collect();
        let mut out = Vec::new();
        let mut rest = inner.as_str();
        while let Some(i) = find_top_level(rest, ',') {
            out.push(rest[..i].to_string());
            rest = &rest[i + 1..];
        }
        if !rest.trim().is_empty() {
            out.push(rest.to_string());
        }
        Ok(out)
    }

    fn sub_matrix(&self, text: &str) -> Result<Matrix, String> {
        eval_matrix(self.ds, self.st, text)
    }

    fn primary(&mut self) -> Result<Matrix, String> {
        self.skip_ws();
        let c = self.s.get(self.pos).copied().ok_or_else(|| rc_err(198, "matrix expression expected"))?;
        if c == '(' {
            self.pos += 1;
            let m = self.rowjoin()?;
            if !self.eat(')') {
                return Err(rc_err(198, "unmatched parenthesis"));
            }
            return Ok(m);
        }
        if c.is_ascii_digit() || c == '.' {
            let start = self.pos;
            while self.pos < self.s.len() && (self.s[self.pos].is_ascii_digit() || matches!(self.s[self.pos], '.' | 'e' | 'E') || ((self.s[self.pos] == '-' || self.s[self.pos] == '+') && matches!(self.s[self.pos - 1], 'e' | 'E'))) {
                self.pos += 1;
            }
            let t: String = self.s[start..self.pos].iter().collect();
            let v = if t == "." { SYSMISS } else { t.parse::<f64>().map_err(|_| rc_err(198, format!("{t} invalid number")))? };
            return Ok(Matrix::filled(1, 1, v));
        }
        let name = self.ident();
        if name.is_empty() {
            return Err(rc_err(198, format!("unexpected character {c}")));
        }
        if self.peek() == Some('(') {
            let save = self.pos;
            let args = self.args()?;
            let lname = name.to_lowercase();
            let num = |p: &Self, i: usize| -> Result<f64, String> { p.scalar(args.get(i).ok_or_else(|| rc_err(198, "too few arguments"))?) };
            return Ok(match lname.as_str() {
                "j" => Matrix::filled(num(self, 0)? as usize, num(self, 1)? as usize, num(self, 2)?),
                "i" => Matrix::identity(num(self, 0)? as usize),
                "diag" => {
                    let v = self.sub_matrix(&args[0])?;
                    if v.rows == 1 || v.cols == 1 {
                        let names = if v.rows == 1 { v.colnames.clone() } else { v.rownames.clone() };
                        Matrix::from_diag(&v.data).with_names(names.clone(), names)
                    } else {
                        return Err(conform());
                    }
                }
                "vecdiag" => {
                    let m = self.sub_matrix(&args[0])?;
                    Matrix::row_vector(&m.diag()).with_names(vec!["r1".into()], m.colnames.clone())
                }
                "inv" => self.sub_matrix(&args[0])?.inverse()?,
                "invsym" | "syminv" => self.sub_matrix(&args[0])?.sym_inverse(1e-14).0,
                "cholesky" => self.sub_matrix(&args[0])?.cholesky()?,
                "hadamard" => {
                    let a = self.sub_matrix(&args[0])?;
                    let b = self.sub_matrix(&args[1])?;
                    if a.rows != b.rows || a.cols != b.cols {
                        return Err(conform());
                    }
                    let mut m = a.clone();
                    m.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x *= y);
                    m
                }
                "corr" => {
                    let a = self.sub_matrix(&args[0])?;
                    let mut m = a.clone();
                    for i in 0..a.rows {
                        for j in 0..a.cols {
                            m[(i, j)] = a[(i, j)] / (a[(i, i)] * a[(j, j)]).sqrt();
                        }
                    }
                    m
                }
                "nullmat" => {
                    let n = args[0].trim();
                    self.st.matrices.get(n).cloned().unwrap_or_else(|| Matrix::zeros(0, 0))
                }
                "r" | "e" => {
                    let key = args.first().map(|s| s.trim().to_string()).unwrap_or_default();
                    let res = if lname == "r" { &self.st.r } else { &self.st.e };
                    if let Some(m) = res.matrices.get(&key) {
                        m.clone()
                    } else if let Some(v) = res.scalars.get(&key) {
                        Matrix::filled(1, 1, *v)
                    } else {
                        return Err(rc_err(111, format!("{lname}({key}) not found")));
                    }
                }
                _ => {
                    // scalar function: evaluate the whole call as a scalar
                    self.pos = save;
                    let start = save - name.len();
                    let _ = self.args()?;
                    let text: String = self.s[start..self.pos].iter().collect();
                    Matrix::filled(1, 1, self.scalar(&text)?)
                }
            });
        }
        if let Some(m) = self.st.matrices.get(&name) {
            return Ok(m.clone());
        }
        if let Some(v) = self.st.scalars.get(&name) {
            return Ok(Matrix::filled(1, 1, *v));
        }
        Err(rc_err(111, format!("matrix {name} not found")))
    }
}

fn vjoin(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows == 0 {
        return Ok(b.clone());
    }
    if a.cols != b.cols {
        return Err(conform());
    }
    let mut m = Matrix::zeros(a.rows + b.rows, a.cols);
    m.data = a.data.iter().chain(&b.data).cloned().collect();
    m.colnames = a.colnames.clone();
    m.rownames = renumber(a.rownames.iter().chain(&b.rownames).cloned().collect(), "r");
    Ok(m)
}

/// Rows (columns) built from single elements, as in `(1,2\3,4)`, all carry
/// the name `r1` (`c1`) and are renumbered; joined names are otherwise kept.
fn renumber(names: Vec<String>, prefix: &str) -> Vec<String> {
    let first = format!("{prefix}1");
    if names.len() > 1 && names.iter().all(|n| *n == first) {
        return (1..=names.len()).map(|i| format!("{prefix}{i}")).collect();
    }
    names
}

fn hjoin(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols == 0 {
        return Ok(b.clone());
    }
    if a.rows != b.rows {
        return Err(conform());
    }
    let mut m = Matrix::zeros(a.rows, a.cols + b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            m[(i, j)] = a[(i, j)];
        }
        for j in 0..b.cols {
            m[(i, a.cols + j)] = b[(i, j)];
        }
    }
    m.rownames = a.rownames.clone();
    m.colnames = renumber(a.colnames.iter().chain(&b.colnames).cloned().collect(), "c");
    Ok(m)
}

fn kron(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = Matrix::zeros(a.rows * b.rows, a.cols * b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            for k in 0..b.rows {
                for l in 0..b.cols {
                    m[(i * b.rows + k, j * b.cols + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }
    m
}

/// Evaluate a matrix expression.
pub fn eval_matrix(ds: &DataSet, st: &Store, text: &str) -> Result<Matrix, String> {
    let mut p = MParser { s: text.chars().collect(), pos: 0, ds, st };
    let m = p.rowjoin()?;
    p.skip_ws();
    if p.pos != p.s.len() {
        return Err(rc_err(198, format!("invalid matrix expression near {}", p.s[p.pos..].iter().collect::<String>())));
    }
    Ok(m)
}

fn list_matrix(name: &str, m: &Matrix, fmt: Option<&str>) -> String {
    let mut out = vec![format!("{name}[{},{}]", m.rows, m.cols)];
    let cell = |v: f64| match fmt {
        Some(f) => crate::format::fmt(v, f).trim().to_string(),
        None => fmt_g(v, 9),
    };
    let rw = m.rownames.iter().map(|r| r.len()).max().unwrap_or(2).max(2);
    for start in (0..m.cols).step_by(6) {
        let end = (start + 6).min(m.cols);
        out.push(format!("{:rw$} {}", "", m.colnames[start..end].iter().map(|c| format!("{:>11}", crate::summarize::abbrev(c, 11))).collect::<String>(), rw = rw));
        for i in 0..m.rows {
            let symmetric_lower = m.rows == m.cols && (0..m.rows).all(|a| (0..m.cols).all(|b| m[(a, b)] == m[(b, a)])) && m.rows > 1;
            let cells: String = (start..end).map(|j| if symmetric_lower && j > i { format!("{:>11}", "") } else { format!("{:>11}", cell(m[(i, j)])) }).collect();
            out.push(format!("{:>rw$} {}", m.rownames[i], cells, rw = rw));
        }
        out.push(String::new());
    }
    out.join("\n").trim_end().to_string()
}

fn names_list(text: &str) -> Vec<String> {
    split_top_level_words(text).into_iter().map(|w| util::unquote(&w)).collect()
}

pub struct MatrixCmd;

impl CommandHandler for MatrixCmd {
    fn name(&self) -> &'static str {
        "matrix"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let body = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        let first = body.split_whitespace().next().unwrap_or("").to_string();
        let rest = body[first.len()..].trim();
        // `matrix L = ...` defines a matrix named like a subcommand abbreviation
        let lfirst = if rest.starts_with('=') { String::new() } else { first.to_lowercase() };
        match lfirst.as_str() {
            "list" | "li" | "l" => {
                let (name, opts) = match find_top_level(rest, ',') {
                    Some(i) => (rest[..i].trim(), Some(&rest[i + 1..])),
                    None => (rest, None),
                };
                let m = eval_matrix(ds, st, name)?;
                let fmt = opts.and_then(|o| ferrostat_parser::parser::split_options(o).into_iter().find(|(n, _)| n.starts_with("form")).and_then(|(_, a)| a));
                return Ok(list_matrix(name, &m, fmt.as_deref()));
            }
            "dir" => return Ok(st.matrices.iter().map(|(k, m)| format!("{:>12}[{},{}]", k, m.rows, m.cols)).collect::<Vec<_>>().join("\n")),
            "drop" => {
                if rest == "_all" {
                    st.matrices.clear();
                } else {
                    for n in rest.split_whitespace() {
                        st.matrices.remove(n);
                    }
                }
                return Ok(String::new());
            }
            "rownames" | "rowname" | "rown" | "colnames" | "colname" | "coln" | "roweq" | "coleq" => {
                let eq = find_top_level(rest, '=').ok_or_else(|| rc_err(198, "= expected"))?;
                let name = rest[..eq].trim();
                let names = names_list(&rest[eq + 1..]);
                let m = st.matrices.get_mut(name).ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
                let is_row = lfirst.starts_with("row");
                let n = if is_row { m.rows } else { m.cols };
                if names.len() != n && !lfirst.ends_with("eq") {
                    return Err(rc_err(503, "conformability error"));
                }
                let target = if is_row { &mut m.rownames } else { &mut m.colnames };
                if lfirst.ends_with("eq") {
                    for (k, t) in target.iter_mut().enumerate() {
                        let eqn = names.get(k).or(names.last()).cloned().unwrap_or_default();
                        let base = t.rsplit(':').next().unwrap().to_string();
                        *t = format!("{eqn}:{base}");
                    }
                } else {
                    *target = names;
                }
                return Ok(String::new());
            }
            "accum" | "ac" => {
                let eq = find_top_level(rest, '=').ok_or_else(|| rc_err(198, "= expected"))?;
                let name = rest[..eq].trim().to_string();
                let c = ferrostat_parser::parser::parse_line(&format!("accum {}", &rest[eq + 1..])).map_err(|e| rc_err(198, e))?;
                let vars = varlist::expand(ds, &c.main)?;
                let mut mask = util::touse(ds, st, &c)?;
                let w = util::weights(ds, st, &c, &mut mask, "fai")?;
                util::markout(ds, &mut mask, &vars);
                let rows = util::rows(&mask);
                let dev = c.has_opt("deviations", 3);
                let nocons = c.has_opt("noconstant", 3);
                let cols: Vec<Vec<f64>> = vars.iter().map(|v| rows.iter().map(|&i| ds.get_var(v).unwrap().num(i)).collect()).collect();
                let ww: Vec<f64> = rows.iter().map(|&i| w.as_ref().map_or(1.0, |w| w.w[i])).collect();
                let sw: f64 = ww.iter().sum();
                let mut cols = cols;
                let mut names = vars.clone();
                if dev {
                    for c in cols.iter_mut() {
                        let m = c.iter().zip(&ww).map(|(x, w)| x * w).sum::<f64>() / sw;
                        c.iter_mut().for_each(|x| *x -= m);
                    }
                }
                if !nocons {
                    cols.push(vec![if dev { 0.0 } else { 1.0 }; rows.len()]);
                    names.push("_cons".into());
                }
                let k = cols.len();
                let mut m = Matrix::zeros(k, k);
                for a in 0..k {
                    for b in 0..=a {
                        let s: f64 = (0..rows.len()).map(|t| ww[t] * cols[a][t] * cols[b][t]).sum();
                        m[(a, b)] = s;
                        m[(b, a)] = s;
                    }
                }
                st.matrices.insert(name, m.with_names(names.clone(), names));
                let r = st.begin_r();
                r.set("N", if w.as_ref().map_or(false, |w| w.kind == util::WeightKind::F) { sw } else { rows.len() as f64 });
                return Ok(format!("(obs={})", rows.len()));
            }
            l if l.len() >= 6 && "dissimilarity".starts_with(l) => return crate::mv_cluster::matrix_dissimilarity(ds, st, cmd),
            "symeigen" | "symeig" => {
                let eq = find_top_level(rest, '=').ok_or_else(|| rc_err(198, "= expected"))?;
                let names: Vec<&str> = rest[..eq].split_whitespace().collect();
                if names.len() != 2 {
                    return Err(rc_err(198, "matrix symeigen vectors values = A"));
                }
                let a = eval_matrix(ds, st, rest[eq + 1..].trim())?;
                if a.rows != a.cols {
                    return Err(rc_err(505, "matrix not symmetric"));
                }
                let (vals, mut vecs) = a.sym_eigen();
                // sign convention: make the largest-magnitude element of each vector positive
                for j in 0..vecs.cols {
                    let mut best = 0.0f64;
                    for i in 0..vecs.rows {
                        if vecs[(i, j)].abs() > best.abs() + 1e-12 {
                            best = vecs[(i, j)];
                        }
                    }
                    if best < 0.0 {
                        for i in 0..vecs.rows {
                            vecs[(i, j)] = -vecs[(i, j)];
                        }
                    }
                }
                let en: Vec<String> = (1..=vals.len()).map(|i| format!("e{i}")).collect();
                st.matrices.insert(names[0].to_string(), vecs.with_names(a.rownames.clone(), en.clone()));
                st.matrices.insert(names[1].to_string(), Matrix::row_vector(&vals).with_names(vec!["r1".into()], en));
                return Ok(String::new());
            }
            _ => {}
        }
        // definition or element assignment
        let body = if lfirst == "define" || lfirst == "def" || lfirst == "input" || lfirst == "in" { rest } else { body };
        let eq = find_top_level(body, '=').ok_or_else(|| rc_err(198, format!("matrix {first}: invalid subcommand")))?;
        let lhs = body[..eq].trim();
        let rhs = body[eq + 1..].trim();
        if let Some(open) = lhs.find('[') {
            let name = lhs[..open].trim();
            let inner = lhs[open + 1..lhs.rfind(']').ok_or_else(|| rc_err(198, "] expected"))?].to_string();
            let p = MParser { s: Vec::new(), pos: 0, ds, st };
            let m = st.matrices.get(name).ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?.clone();
            let (r, c) = match find_top_level(&inner, ',') {
                Some(i) => (inner[..i].to_string(), inner[i + 1..].to_string()),
                None => (inner.clone(), "1".to_string()),
            };
            let (r0, _) = p.range(&r, m.rows, &m.rownames)?;
            let (c0, _) = p.range(&c, m.cols, &m.colnames)?;
            // scalar expression first, then matrix expression
            let val = match parse_expression(rhs).ok().and_then(|e| Env::with_store(ds, st).eval(&e, 0).ok()) {
                Some(v) => v.num()?,
                None => {
                    let mv = eval_matrix(ds, st, rhs)?;
                    // assigning a block
                    let mut m2 = m.clone();
                    for i in 0..mv.rows {
                        for j in 0..mv.cols {
                            if r0 + i >= m2.rows || c0 + j >= m2.cols {
                                return Err(conform());
                            }
                            m2[(r0 + i, c0 + j)] = mv[(i, j)];
                        }
                    }
                    st.matrices.insert(name.to_string(), m2);
                    return Ok(String::new());
                }
            };
            let mm = st.matrices.get_mut(name).unwrap();
            mm[(r0, c0)] = val;
            return Ok(String::new());
        }
        let m = if lfirst == "input" || lfirst == "in" {
            eval_matrix(ds, st, rhs)?
        } else {
            eval_matrix(ds, st, rhs)?
        };
        crate::varlist::check_name(lhs)?;
        st.matrices.insert(lhs.to_string(), m);
        Ok(String::new())
    }
}

pub struct Mkmat;

impl CommandHandler for Mkmat {
    fn name(&self) -> &'static str {
        "mkmat"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        if cmd.has_opt("nomissing", 3) {
            util::markout(ds, &mut mask, &vars);
        }
        let rows = util::rows(&mask);
        let mut m = Matrix::zeros(rows.len(), vars.len());
        for (j, v) in vars.iter().enumerate() {
            let c = util::col(ds, v)?;
            for (i, &r) in rows.iter().enumerate() {
                m[(i, j)] = c[r];
            }
        }
        m.colnames = vars.clone();
        if let Some(rn) = cmd.opt("rownames", 4) {
            let v = ds.get_var(&varlist::resolve_name(ds, rn.trim())?).unwrap();
            m.rownames = rows.iter().map(|&r| util::cell_text(ds, v, r, false)).collect();
        }
        match cmd.opt("matrix", 3) {
            Some(name) => {
                st.matrices.insert(name.trim().to_string(), m);
            }
            None => {
                for (j, v) in vars.iter().enumerate() {
                    let mut col = Matrix::zeros(rows.len(), 1);
                    for i in 0..rows.len() {
                        col[(i, 0)] = m[(i, j)];
                    }
                    col.colnames = vec![v.clone()];
                    st.matrices.insert(v.clone(), col);
                }
            }
        }
        Ok(String::new())
    }
}

pub struct Svmat;

impl CommandHandler for Svmat {
    fn name(&self) -> &'static str {
        "svmat"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words: Vec<&str> = cmd.main.split_whitespace().collect();
        let (ty, name) = match words.as_slice() {
            [n] => (VarType::Float, *n),
            [t, n] => (crate::data::parse_type(t)?, *n),
            _ => return Err(rc_err(198, "invalid syntax")),
        };
        let m = st.matrices.get(name).ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?.clone();
        let names: Vec<String> = match cmd.opt("names", 1).map(|s| s.trim().to_string()) {
            Some(n) if n == "col" => m.colnames.clone(),
            Some(n) if n == "eqcol" => m.colnames.iter().map(|c| c.replace(':', "_")).collect(),
            Some(n) if n == "matcol" => m.colnames.iter().map(|c| format!("{name}{c}")).collect(),
            Some(stub) => (1..=m.cols).map(|j| format!("{stub}{j}")).collect(),
            None => (1..=m.cols).map(|j| format!("{name}{j}")).collect(),
        };
        for n in &names {
            if ds.var_index.contains_key(n) {
                return Err(rc_err(110, format!("{n} already defined")));
            }
        }
        while ds.n_obs < m.rows {
            ds.add_observation();
        }
        for (j, n) in names.iter().enumerate() {
            let vals: Vec<f64> = (0..ds.n_obs).map(|i| if i < m.rows { m[(i, j)] } else { SYSMISS }).collect();
            let mut v = ferrostat_core::Variable::new(n, ty);
            v.data = vals.iter().map(|&x| crate::data::store_num(ty, x)).collect();
            ds.add_variable(v)?;
        }
        st.data_changed = true;
        let _ = is_missing;
        Ok(String::new())
    }
}

/// `matlist` (simple form of matrix list).
pub struct Matlist;

impl CommandHandler for Matlist {
    fn name(&self) -> &'static str {
        "matlist"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let m = eval_matrix(ds, st, cmd.main.trim())?;
        Ok(list_matrix(cmd.main.trim(), &m, cmd.opt("format", 1)))
    }
}
