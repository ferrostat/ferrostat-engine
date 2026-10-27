//! Matrix values of the matrix language: real and string matrices,
//! pointers and class instances, plus the default result display.

use std::cell::RefCell;
use std::rc::Rc;

use ferrostat_core::missing::{missing_code, missing_name};

/// A dense row-major matrix.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mat<T> {
    pub r: usize,
    pub c: usize,
    pub d: Vec<T>,
}

impl<T: Clone> Mat<T> {
    pub fn new(r: usize, c: usize, d: Vec<T>) -> Self {
        debug_assert_eq!(r * c, d.len());
        Mat { r, c, d }
    }
    pub fn fill(r: usize, c: usize, v: T) -> Self {
        Mat { r, c, d: vec![v; r * c] }
    }
    pub fn scalar(v: T) -> Self {
        Mat { r: 1, c: 1, d: vec![v] }
    }
    pub fn row(v: Vec<T>) -> Self {
        Mat { r: 1, c: v.len(), d: v }
    }
    pub fn col(v: Vec<T>) -> Self {
        Mat { r: v.len(), c: 1, d: v }
    }
    pub fn at(&self, i: usize, j: usize) -> &T {
        &self.d[i * self.c + j]
    }
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        let c = self.c;
        self.d[i * c + j] = v;
    }
    pub fn t(&self) -> Self {
        let mut d = Vec::with_capacity(self.d.len());
        for j in 0..self.c {
            for i in 0..self.r {
                d.push(self.at(i, j).clone());
            }
        }
        Mat { r: self.c, c: self.r, d }
    }
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Mat<U> {
        Mat { r: self.r, c: self.c, d: self.d.iter().map(f).collect() }
    }
    pub fn is_scalar(&self) -> bool {
        self.r == 1 && self.c == 1
    }
}

/// Pointer targets.
#[derive(Debug, Clone)]
pub enum Ptr {
    Null,
    Var(Rc<RefCell<Val>>),
    Func(String),
}

/// Built-in class instances.
#[derive(Debug, Clone, Default)]
pub struct XlBook {
    pub path: String,
    pub book: Option<ferrostat_cmd::xlsx::Workbook>,
    pub sheet: usize,
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub enum Obj {
    Xl(XlBook),
}

#[derive(Debug, Clone)]
pub enum Val {
    R(Mat<f64>),
    S(Mat<String>),
    P(Mat<Ptr>),
    O(Rc<RefCell<Obj>>),
}

impl Default for Val {
    fn default() -> Self {
        Val::R(Mat::default())
    }
}

impl Val {
    pub fn num(v: f64) -> Val {
        Val::R(Mat::scalar(v))
    }
    pub fn str(s: impl Into<String>) -> Val {
        Val::S(Mat::scalar(s.into()))
    }
    pub fn empty() -> Val {
        Val::R(Mat::default())
    }
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Val::R(m) => (m.r, m.c),
            Val::S(m) => (m.r, m.c),
            Val::P(m) => (m.r, m.c),
            Val::O(_) => (1, 1),
        }
    }
    pub fn rows(&self) -> usize {
        self.dims().0
    }
    pub fn cols(&self) -> usize {
        self.dims().1
    }
    pub fn eltype(&self) -> &'static str {
        match self {
            Val::R(_) => "real",
            Val::S(_) => "string",
            Val::P(_) => "pointer",
            Val::O(_) => "class",
        }
    }
    pub fn orgtype(&self) -> &'static str {
        match self.dims() {
            (1, 1) => "scalar",
            (1, _) => "rowvector",
            (_, 1) => "colvector",
            _ => "matrix",
        }
    }
    pub fn transpose(&self) -> Val {
        match self {
            Val::R(m) => Val::R(m.t()),
            Val::S(m) => Val::S(m.t()),
            Val::P(m) => Val::P(m.t()),
            Val::O(_) => self.clone(),
        }
    }
}

/// Element formatting: up to ten significant digits, no leading zero.
pub fn fmt_num(v: f64) -> String {
    if let Some(code) = missing_code(v) {
        return missing_name(code);
    }
    if v == 0.0 {
        return "0".into();
    }
    if v.fract() == 0.0 && v.abs() < 1e10 {
        return format!("{}", v as i64);
    }
    let exp = v.abs().log10().floor() as i32;
    if (-5..10).contains(&exp) {
        let dec = (9 - exp).max(0) as usize;
        let mut s = format!("{:.*}", dec, v);
        if s.contains('.') {
            s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        }
        if let Some(r) = s.strip_prefix("0.") {
            return format!(".{r}");
        }
        if let Some(r) = s.strip_prefix("-0.") {
            return format!("-.{r}");
        }
        return s;
    }
    ferrostat_cmd::format::fmt_e(v, 5)
}

const LINESIZE: usize = 79;

/// Render a value the way an expression statement displays it.
pub fn display(v: &Val) -> String {
    match v {
        Val::R(m) => {
            if m.is_scalar() {
                return format!("  {}", fmt_num(m.d[0]));
            }
            let cells: Vec<String> = m.d.iter().map(|x| fmt_num(*x)).collect();
            boxed(m.r, m.c, &cells)
        }
        Val::S(m) => {
            if m.is_scalar() {
                return format!("  {}", m.d[0]);
            }
            boxed(m.r, m.c, &m.d)
        }
        Val::P(m) => {
            let cells: Vec<String> = m.d.iter().map(ptr_text).collect();
            if m.is_scalar() {
                return format!("  {}", cells[0]);
            }
            boxed(m.r, m.c, &cells)
        }
        Val::O(_) => "  <class instance>".to_string(),
    }
}

fn ptr_text(p: &Ptr) -> String {
    match p {
        Ptr::Null => "NULL".into(),
        Ptr::Var(rc) => format!("0x{:x}", Rc::as_ptr(rc) as usize & 0xffffffff),
        Ptr::Func(f) => format!("&{f}()"),
    }
}

/// Matrix display with column numbers on top and row numbers on the left,
/// split into column segments that fit the line width.
fn boxed(r: usize, c: usize, cells: &[String]) -> String {
    if r == 0 || c == 0 {
        return String::new();
    }
    let lw = r.to_string().len();
    let w = cells.iter().map(|s| s.chars().count()).max().unwrap_or(1).max(c.to_string().len());
    let pre = lw + 4;
    let per = (LINESIZE.saturating_sub(pre + 2) / (w + 3)).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < c {
        let end = (start + per).min(c);
        let first = start == 0;
        let last = end == c;
        let n = end - start;
        let dashes = "-".repeat(n * (w + 3) + if last { 1 } else { 0 });
        let mut head = " ".repeat(pre);
        for j in start..end {
            head.push_str(&format!("{:>width$}", j + 1, width = w + if j == start { 2 } else { 3 }));
        }
        out.push(head);
        let border = format!("{}{}{}{}", " ".repeat(pre - 1), if first { "+" } else { " " }, dashes, if last { "+" } else { "" });
        out.push(border.clone());
        for i in 0..r {
            let mut line = format!("  {:>lw$}{}", i + 1, if first { " |" } else { "  " });
            for j in start..end {
                let s = &cells[i * c + j];
                line.push_str(&format!("{:>width$}", s, width = w + if j == start { 2 } else { 3 }));
            }
            if last {
                line.push_str("  |");
            }
            out.push(line);
        }
        out.push(border);
        start = end;
    }
    out.join("\n")
}
