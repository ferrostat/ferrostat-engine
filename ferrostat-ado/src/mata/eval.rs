//! Statement execution and expression evaluation.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::{Rc, Weak};

use super::ops;
use super::parser::{El, Expr, FuncDef, Org, Stmt, Ty};
use super::value::{display, Mat, Ptr, Val};
use crate::interpreter::AdoInterpreter;

pub type Slot = Rc<RefCell<Val>>;

/// A view onto the data: observations and variables it maps.
#[derive(Debug, Clone)]
pub struct View {
    pub obs: Vec<usize>,
    pub vars: Vec<String>,
}

/// Everything that survives between blocks: global variables and functions.
#[derive(Debug, Default)]
pub struct MataState {
    pub globals: BTreeMap<String, Slot>,
    pub funcs: BTreeMap<String, Rc<FuncDef>>,
    pub views: Vec<(Weak<RefCell<Val>>, View)>,
}

/// A run-time error: return code, message and the function it arose in.
#[derive(Debug, Clone)]
pub struct MErr {
    pub rc: i32,
    pub msg: String,
    pub trace: Vec<String>,
}

pub fn err(rc: i32, msg: impl Into<String>) -> MErr {
    MErr { rc, msg: msg.into(), trace: Vec::new() }
}

pub type R<T> = Result<T, MErr>;

pub fn conform() -> MErr {
    err(3200, "conformability error")
}

pub fn type_mismatch() -> MErr {
    err(3250, "type mismatch")
}

pub enum Ctl {
    Next,
    Break,
    Continue,
    Return(Option<Val>),
}

struct Frame {
    vars: HashMap<String, Slot>,
    nargs: usize,
}

pub struct Ev<'a> {
    pub me: &'a mut AdoInterpreter,
    pub st: &'a mut MataState,
    frames: Vec<Frame>,
    /// printf output not yet terminated by a newline
    pub pending: String,
}

impl<'a> Ev<'a> {
    pub fn new(me: &'a mut AdoInterpreter, st: &'a mut MataState) -> Self {
        Ev { me, st, frames: Vec::new(), pending: String::new() }
    }

    // ---- output ----

    pub fn print(&mut self, text: &str) {
        self.pending.push_str(text);
        while let Some(i) = self.pending.find('\n') {
            let line: String = self.pending[..i].to_string();
            self.pending = self.pending[i + 1..].to_string();
            self.emit_line(&line);
        }
    }

    fn emit_line(&mut self, line: &str) {
        // emit() skips empty text; keep blank lines
        self.me.emit(if line.is_empty() { " " } else { line });
    }

    pub fn flush(&mut self) {
        if !self.pending.is_empty() {
            let p = std::mem::take(&mut self.pending);
            self.emit_line(&p);
        }
    }

    pub fn show(&mut self, v: &Val) {
        self.flush();
        let t = display(v);
        if !t.is_empty() {
            self.me.emit(&t);
        }
    }

    // ---- variables ----

    pub fn lookup(&self, name: &str) -> Option<Slot> {
        match self.frames.last() {
            Some(f) => f.vars.get(name).cloned(),
            None => self.st.globals.get(name).cloned(),
        }
    }

    fn slot_or_new(&mut self, name: &str) -> Slot {
        if let Some(s) = self.lookup(name) {
            return s;
        }
        let s: Slot = Rc::new(RefCell::new(Val::empty()));
        match self.frames.last_mut() {
            Some(f) => {
                f.vars.insert(name.to_string(), s.clone());
            }
            None => {
                self.st.globals.insert(name.to_string(), s.clone());
            }
        }
        s
    }

    pub fn set_var(&mut self, name: &str, v: Val) {
        let s = self.slot_or_new(name);
        self.drop_view(&s);
        *s.borrow_mut() = v;
    }

    fn drop_view(&mut self, s: &Slot) {
        if !self.st.views.is_empty() {
            let p = Rc::as_ptr(s);
            self.st.views.retain(|(w, _)| w.as_ptr() != p && w.strong_count() > 0);
        }
    }

    pub fn set_view(&mut self, s: &Slot, v: View) {
        self.drop_view(s);
        self.st.views.push((Rc::downgrade(s), v));
    }

    fn view_of(&self, s: &Slot) -> Option<View> {
        let p = Rc::as_ptr(s);
        self.st.views.iter().find(|(w, _)| w.as_ptr() == p).map(|(_, v)| v.clone())
    }

    pub fn nargs(&self) -> usize {
        self.frames.last().map(|f| f.nargs).unwrap_or(0)
    }

    // ---- statements ----

    pub fn exec(&mut self, s: &Stmt) -> R<Ctl> {
        match s {
            Stmt::Nop => Ok(Ctl::Next),
            Stmt::Expr(e) => {
                match e {
                    Expr::Assign(..) | Expr::IncDec(..) => {
                        self.eval(e)?;
                    }
                    Expr::Call(..) | Expr::Method(..) | Expr::CallPtr(..) => {
                        if let Some(v) = self.call_expr(e)? {
                            self.show(&v);
                        }
                    }
                    _ => {
                        let v = self.eval(e)?;
                        self.show(&v);
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::Block(b) => {
                for s in b {
                    match self.exec(s)? {
                        Ctl::Next => {}
                        c => return Ok(c),
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::If(c, a, b) => {
                if self.truth(c)? {
                    self.exec(a)
                } else if let Some(b) = b {
                    self.exec(b)
                } else {
                    Ok(Ctl::Next)
                }
            }
            Stmt::For(init, cond, step, body) => {
                if let Some(i) = init {
                    self.eval(i)?;
                }
                loop {
                    if let Some(c) = cond {
                        if !self.truth(c)? {
                            break;
                        }
                    }
                    match self.exec(body)? {
                        Ctl::Break => break,
                        Ctl::Return(v) => return Ok(Ctl::Return(v)),
                        _ => {}
                    }
                    if let Some(s) = step {
                        self.eval(s)?;
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::While(c, body) => {
                while self.truth(c)? {
                    match self.exec(body)? {
                        Ctl::Break => break,
                        Ctl::Return(v) => return Ok(Ctl::Return(v)),
                        _ => {}
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::DoWhile(body, c) => {
                loop {
                    match self.exec(body)? {
                        Ctl::Break => break,
                        Ctl::Return(v) => return Ok(Ctl::Return(v)),
                        _ => {}
                    }
                    if !self.truth(c)? {
                        break;
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::Break => Ok(Ctl::Break),
            Stmt::Continue => Ok(Ctl::Continue),
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => Some(self.eval(e)?),
                    None => None,
                };
                Ok(Ctl::Return(v))
            }
            Stmt::Decl(ty, names, external) => {
                for n in names {
                    if *external {
                        let g = match self.st.globals.get(n) {
                            Some(g) => g.clone(),
                            None => {
                                let g: Slot = Rc::new(RefCell::new(default_of(ty)));
                                self.st.globals.insert(n.clone(), g.clone());
                                g
                            }
                        };
                        if let Some(f) = self.frames.last_mut() {
                            f.vars.insert(n.clone(), g);
                        }
                    } else if self.lookup(n).is_none() || !self.frames.is_empty() {
                        if let Some(f) = self.frames.last_mut() {
                            // parameters keep their values
                            if f.vars.contains_key(n) {
                                continue;
                            }
                        }
                        self.set_var(n, default_of(ty));
                    }
                }
                Ok(Ctl::Next)
            }
            Stmt::Func(f) => {
                self.st.funcs.insert(f.name.clone(), f.clone());
                Ok(Ctl::Next)
            }
            Stmt::Cmd(c) => {
                super::command(self, c)?;
                Ok(Ctl::Next)
            }
        }
    }

    fn truth(&mut self, e: &Expr) -> R<bool> {
        let v = self.eval(e)?;
        match v {
            Val::R(m) if m.is_scalar() => Ok(m.d[0] != 0.0),
            Val::R(_) => Err(err(3204, "matrix found where scalar required")),
            _ => Err(err(3253, "nonreal found where real required")),
        }
    }

    // ---- expressions ----

    pub fn eval(&mut self, e: &Expr) -> R<Val> {
        match e {
            Expr::Num(v) => Ok(Val::num(*v)),
            Expr::Miss(c) => Ok(Val::num(ferrostat_core::missing::missing(*c))),
            Expr::Str(s) => Ok(Val::str(s.clone())),
            Expr::Omitted => Ok(Val::empty()),
            Expr::Var(n) => match self.lookup(n) {
                Some(s) => Ok(s.borrow().clone()),
                None if n == "NULL" => Ok(Val::P(Mat::scalar(Ptr::Null))),
                None => Err(err(3499, format!("{n} not found"))),
            },
            Expr::Bin(op, a, b) => {
                if *op == "&&" || *op == "||" {
                    let x = self.truth(a)?;
                    if (*op == "&&" && !x) || (*op == "||" && x) {
                        return Ok(Val::num(x as i32 as f64));
                    }
                    return Ok(Val::num(self.truth(b)? as i32 as f64));
                }
                let x = self.eval(a)?;
                let y = self.eval(b)?;
                ops::binop(op, x, y)
            }
            Expr::Neg(a) => ops::neg(self.eval(a)?),
            Expr::Not(a) => match self.eval(a)? {
                Val::R(m) => Ok(Val::R(m.map(|x| if *x == 0.0 { 1.0 } else { 0.0 }))),
                _ => Err(type_mismatch()),
            },
            Expr::Trans(a) => Ok(self.eval(a)?.transpose()),
            Expr::Cond(c, a, b) => {
                if self.truth(c)? {
                    self.eval(a)
                } else {
                    self.eval(b)
                }
            }
            Expr::Sub(base, idx) => {
                if let Expr::Var(n) = &**base {
                    let s = self.lookup(n).ok_or_else(|| err(3499, format!("{n} not found")))?;
                    let ix = self.indices(idx)?;
                    let v = s.borrow();
                    return ops::subscript(&v, &ix);
                }
                let v = self.eval(base)?;
                let ix = self.indices(idx)?;
                ops::subscript(&v, &ix)
            }
            Expr::RangeSub(base, idx) => {
                let v = self.eval(base)?;
                let i = self.eval(idx)?;
                let (rr, cc) = ops::range_spec(&v, &i)?;
                ops::subscript(&v, &[Some(rr), Some(cc)])
            }
            Expr::Assign(lhs, op, rhs) => {
                let mut v = self.eval(rhs)?;
                if let Some(op) = op {
                    let old = self.eval(lhs)?;
                    v = ops::binop(op, old, v)?;
                }
                self.assign(lhs, v.clone())?;
                Ok(v)
            }
            Expr::IncDec(lhs, d, pre) => {
                let old = self.eval(lhs)?;
                let new = ops::binop("+", old.clone(), Val::num(*d))?;
                self.assign(lhs, new.clone())?;
                Ok(if *pre { new } else { old })
            }
            Expr::Addr(a) => match &**a {
                Expr::Var(n) => {
                    let s = self.slot_or_new(n);
                    Ok(Val::P(Mat::scalar(Ptr::Var(s))))
                }
                Expr::Call(f, args) if args.is_empty() => Ok(Val::P(Mat::scalar(Ptr::Func(f.clone())))),
                other => {
                    let v = self.eval(other)?;
                    Ok(Val::P(Mat::scalar(Ptr::Var(Rc::new(RefCell::new(v))))))
                }
            },
            Expr::Deref(a) => match self.eval(a)? {
                Val::P(m) if m.is_scalar() => match &m.d[0] {
                    Ptr::Var(s) => Ok(s.borrow().clone()),
                    Ptr::Func(f) => Err(err(3000, format!("{f}() is a function"))),
                    Ptr::Null => Err(err(3120, "attempt to dereference NULL pointer")),
                },
                _ => Err(err(3253, "nonpointer found where pointer required")),
            },
            Expr::Call(..) | Expr::Method(..) | Expr::CallPtr(..) => match self.call_expr(e)? {
                Some(v) => Ok(v),
                None => Ok(Val::empty()),
            },
        }
    }

    /// Subscript lists: None means "all".
    fn indices(&mut self, idx: &[Expr]) -> R<Vec<Option<Vec<usize>>>> {
        let mut out = Vec::new();
        for e in idx {
            if matches!(e, Expr::Miss(0)) {
                out.push(None);
                continue;
            }
            let v = self.eval(e)?;
            match v {
                Val::R(m) => {
                    if m.is_scalar() && ferrostat_core::missing::is_missing(m.d[0]) {
                        out.push(None);
                        continue;
                    }
                    let mut list = Vec::with_capacity(m.d.len());
                    for x in &m.d {
                        if *x < 1.0 || ferrostat_core::missing::is_missing(*x) {
                            return Err(err(3301, "subscript invalid"));
                        }
                        list.push(*x as usize - 1);
                    }
                    out.push(Some(list));
                }
                _ => return Err(err(3253, "nonreal found where real required")),
            }
        }
        Ok(out)
    }

    pub fn assign(&mut self, lhs: &Expr, v: Val) -> R<()> {
        match lhs {
            Expr::Var(n) => {
                self.set_var(n, v);
                Ok(())
            }
            Expr::Sub(base, idx) => {
                let Expr::Var(n) = &**base else { return Err(err(3000, "invalid assignment")) };
                let s = self.lookup(n).ok_or_else(|| err(3499, format!("{n} not found")))?;
                let ix = self.indices(idx)?;
                let cells = {
                    let mut cur = s.borrow_mut();
                    ops::sub_assign(&mut cur, &ix, &v)?
                };
                self.write_through(&s, &cells)?;
                Ok(())
            }
            Expr::RangeSub(base, idx) => {
                let Expr::Var(n) = &**base else { return Err(err(3000, "invalid assignment")) };
                let s = self.lookup(n).ok_or_else(|| err(3499, format!("{n} not found")))?;
                let i = self.eval(idx)?;
                let (rr, cc) = ops::range_spec(&s.borrow(), &i)?;
                let cells = {
                    let mut cur = s.borrow_mut();
                    ops::sub_assign(&mut cur, &[Some(rr), Some(cc)], &v)?
                };
                self.write_through(&s, &cells)?;
                Ok(())
            }
            Expr::Deref(p) => match self.eval(p)? {
                Val::P(m) if m.is_scalar() => match &m.d[0] {
                    Ptr::Var(s) => {
                        *s.borrow_mut() = v;
                        Ok(())
                    }
                    _ => Err(err(3120, "attempt to dereference NULL pointer")),
                },
                _ => Err(err(3253, "nonpointer found where pointer required")),
            },
            _ => Err(err(3000, "invalid lval")),
        }
    }

    /// Changed cells of a view are written back to the data.
    fn write_through(&mut self, s: &Slot, cells: &[(usize, usize)]) -> R<()> {
        let Some(view) = self.view_of(s) else { return Ok(()) };
        let v = s.borrow().clone();
        for &(i, j) in cells {
            let (Some(&obs), Some(var)) = (view.obs.get(i), view.vars.get(j)) else { continue };
            match &v {
                Val::R(m) => super::host::store_num(self, obs, var, *m.at(i, j))?,
                Val::S(m) => super::host::store_str(self, obs, var, m.at(i, j))?,
                _ => {}
            }
        }
        Ok(())
    }

    // ---- calls ----

    pub fn call_expr(&mut self, e: &Expr) -> R<Option<Val>> {
        match e {
            Expr::Call(name, args) => self.call(name, args),
            Expr::CallPtr(p, args) => {
                let Expr::Deref(inner) = &**p else { return Err(err(3000, "invalid call")) };
                match self.eval(inner)? {
                    Val::P(m) if m.is_scalar() => match &m.d[0] {
                        Ptr::Func(f) => {
                            let f = f.clone();
                            self.call(&f, args)
                        }
                        _ => Err(err(3120, "attempt to dereference NULL pointer")),
                    },
                    _ => Err(err(3253, "nonpointer found where pointer required")),
                }
            }
            Expr::Method(obj, name, args) => {
                let o = self.eval(obj)?;
                let mut vals = Vec::new();
                for a in args {
                    vals.push(self.eval(a)?);
                }
                match o {
                    Val::O(rc) => super::host::method(self, &rc, name, vals),
                    _ => Err(err(3000, format!("{name}() is not a member function"))),
                }
            }
            _ => Ok(Some(self.eval(e)?)),
        }
    }

    pub fn call(&mut self, name: &str, args: &[Expr]) -> R<Option<Val>> {
        if let Some(f) = self.st.funcs.get(name).cloned() {
            return self.call_user(&f, args);
        }
        // functions whose first argument receives a result
        if matches!(name, "st_view" | "st_sview" | "st_subview" | "swap" | "_sort" | "_editmissing" | "_jumble" | "_transpose" | "_invsym" | "_luinv" | "_cholesky" | "_edittozero" | "_editvalue") {
            return super::host::lvalue_call(self, name, args);
        }
        // dimensions of a variable without copying it (common in loop conditions)
        if let ([Expr::Var(n)], "rows" | "cols" | "length") = (args, name) {
            if let Some(s) = self.lookup(n) {
                let (r, c) = s.borrow().dims();
                return Ok(Some(Val::num(match name {
                    "rows" => r,
                    "cols" => c,
                    _ => r * c,
                } as f64)));
            }
        }
        let mut vals = Vec::with_capacity(args.len());
        let mut omitted = Vec::with_capacity(args.len());
        for a in args {
            omitted.push(matches!(a, Expr::Omitted));
            vals.push(self.eval(a)?);
        }
        super::builtins::call(self, name, vals, &omitted)
    }

    fn call_user(&mut self, f: &Rc<FuncDef>, args: &[Expr]) -> R<Option<Val>> {
        if args.len() > f.params.len() || args.len() < f.nreq {
            let msg = if f.nreq == f.params.len() {
                format!("expected {} arguments but received {}", f.params.len(), args.len())
            } else {
                format!("expected {} to {} arguments but received {}", f.nreq, f.params.len(), args.len())
            };
            return Err(MErr { rc: 3001, msg, trace: vec![f.name.clone()] });
        }
        let mut vars = HashMap::new();
        for (p, a) in f.params.iter().zip(args) {
            // plain variables are passed by reference
            let slot = match a {
                Expr::Var(n) if self.lookup(n).is_some() => self.lookup(n).unwrap(),
                Expr::Assign(l, _, _) if matches!(&**l, Expr::Var(_)) => {
                    self.eval(a)?;
                    let Expr::Var(n) = &**l else { unreachable!() };
                    self.lookup(n).unwrap()
                }
                _ => Rc::new(RefCell::new(self.eval(a)?)),
            };
            if let Some(ty) = &p.ty {
                check_type(&slot.borrow(), ty).map_err(|mut e| {
                    e.trace.push(f.name.clone());
                    e
                })?;
            }
            vars.insert(p.name.clone(), slot);
        }
        if self.frames.len() > 500 {
            return Err(err(3000, "stack overflow"));
        }
        self.frames.push(Frame { vars, nargs: args.len() });
        let mut res = Ok(None);
        for s in &f.body {
            match self.exec(s) {
                Ok(Ctl::Next) | Ok(Ctl::Break) | Ok(Ctl::Continue) => {}
                Ok(Ctl::Return(v)) => {
                    res = Ok(v);
                    break;
                }
                Err(mut e) => {
                    if e.rc != 0 {
                        e.trace.push(f.name.clone());
                    }
                    res = Err(e);
                    break;
                }
            }
        }
        self.frames.pop();
        let v = res?;
        if let (Some(ty), Some(val)) = (&f.ret, &v) {
            if ty.el == El::Void {
                return Ok(None);
            }
            check_type(val, ty).map_err(|mut e| {
                e.trace.push(f.name.clone());
                e
            })?;
        }
        if matches!(&f.ret, Some(t) if t.el == El::Void) {
            return Ok(None);
        }
        Ok(v)
    }
}

fn default_of(ty: &Ty) -> Val {
    let (r, c) = match ty.org {
        Org::Scalar => (1, 1),
        Org::Row | Org::Vector => (1, 0),
        Org::Col => (0, 1),
        Org::Matrix => (0, 0),
    };
    match ty.el {
        El::Str => Val::S(Mat::fill(r, c, String::new())),
        El::Pointer => Val::P(Mat::fill(r, c, Ptr::Null)),
        _ if ty.org == Org::Scalar => Val::num(ferrostat_core::missing::SYSMISS),
        _ => Val::R(Mat::fill(r, c, ferrostat_core::missing::SYSMISS)),
    }
}

/// Check a value against a declared type.
pub fn check_type(v: &Val, ty: &Ty) -> R<()> {
    let ok_el = match (&ty.el, v) {
        (El::Real, Val::R(_)) | (El::Numeric, Val::R(_)) | (El::Str, Val::S(_)) | (El::Pointer, Val::P(_)) => true,
        (El::Trans, _) | (El::Void, _) | (El::Struct(_), _) => true,
        (El::Complex, Val::R(_)) => false,
        _ => false,
    };
    if !ok_el {
        let what = match ty.el {
            El::Real | El::Numeric => "real",
            El::Str => "string",
            El::Pointer => "pointer",
            _ => "complex",
        };
        let found = if v.eltype() == "real" { "real".to_string() } else { format!("non{what}") };
        let msg = if what == "real" { "nonreal found where real required".to_string() } else { format!("{found} found where {what} required") };
        return Err(err(3251, msg));
    }
    let (r, c) = v.dims();
    match ty.org {
        Org::Scalar if (r, c) != (1, 1) => Err(err(3204, format!("{} found where scalar required", v.orgtype()))),
        Org::Vector if r != 1 && c != 1 => Err(err(3201, "vector required")),
        Org::Row if r != 1 => Err(err(3202, "rowvector required")),
        Org::Col if c != 1 => Err(err(3203, "colvector required")),
        _ => Ok(()),
    }
}
