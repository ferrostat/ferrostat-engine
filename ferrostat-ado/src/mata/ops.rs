//! Operators, joins and subscripts.

use ferrostat_core::missing::{is_missing, SYSMISS};

use super::eval::{conform, err, type_mismatch, R};
use super::value::{Mat, Val};

fn arith(op: &str, a: f64, b: f64) -> f64 {
    if is_missing(a) || is_missing(b) {
        return SYSMISS;
    }
    let v = match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => {
            if b == 0.0 {
                return SYSMISS;
            }
            a / b
        }
        "^" => a.powf(b),
        _ => return SYSMISS,
    };
    if v.is_finite() { v } else { SYSMISS }
}

fn cmp_num(op: &str, a: f64, b: f64) -> bool {
    match op {
        "==" => a == b,
        "!=" => a != b,
        ">" => a > b,
        "<" => a < b,
        ">=" => a >= b,
        _ => a <= b,
    }
}

fn cmp_str(op: &str, a: &str, b: &str) -> bool {
    match op {
        "==" => a == b,
        "!=" => a != b,
        ">" => a > b,
        "<" => a < b,
        ">=" => a >= b,
        _ => a <= b,
    }
}

/// Result dimensions of elementwise (colon) operators.
fn bdims(a: (usize, usize), b: (usize, usize)) -> R<(usize, usize)> {
    let one = |x: usize, y: usize| -> R<usize> {
        if x == y {
            Ok(x)
        } else if x == 1 {
            Ok(y)
        } else if y == 1 {
            Ok(x)
        } else {
            Err(conform())
        }
    };
    Ok((one(a.0, b.0)?, one(a.1, b.1)?))
}

fn bcast<A: Clone, B: Clone, C: Clone>(a: &Mat<A>, b: &Mat<B>, f: impl Fn(&A, &B) -> C) -> R<Mat<C>> {
    let (r, c) = bdims((a.r, a.c), (b.r, b.c))?;
    let mut d = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            let x = a.at(if a.r == 1 { 0 } else { i }, if a.c == 1 { 0 } else { j });
            let y = b.at(if b.r == 1 { 0 } else { i }, if b.c == 1 { 0 } else { j });
            d.push(f(x, y));
        }
    }
    Ok(Mat::new(r, c, d))
}

pub fn neg(v: Val) -> R<Val> {
    match v {
        Val::R(m) => Ok(Val::R(m.map(|x| if is_missing(*x) { *x } else { -x }))),
        _ => Err(type_mismatch()),
    }
}

fn scalar_of(v: &Val) -> R<f64> {
    match v {
        Val::R(m) if m.is_scalar() => Ok(m.d[0]),
        Val::R(_) => Err(err(3204, "matrix found where scalar required")),
        _ => Err(type_mismatch()),
    }
}

fn truthy(v: &Val) -> R<bool> {
    Ok(scalar_of(v)? != 0.0)
}

pub fn join(row: bool, a: Val, b: Val) -> R<Val> {
    let empty = |v: &Val| {
        let (r, c) = v.dims();
        r == 0 || c == 0
    };
    if empty(&a) && !matches!(b, Val::O(_)) {
        let (ar, ac) = a.dims();
        let (br, bc) = b.dims();
        if (row && (ac == 0 || ac == bc)) || (!row && (ar == 0 || ar == br)) {
            return Ok(b);
        }
    }
    if empty(&b) && !matches!(a, Val::O(_)) {
        let (ar, ac) = a.dims();
        let (br, bc) = b.dims();
        if (row && (bc == 0 || ac == bc)) || (!row && (br == 0 || ar == br)) {
            return Ok(a);
        }
    }
    fn j<T: Clone>(row: bool, a: Mat<T>, b: Mat<T>) -> R<Mat<T>> {
        if row {
            if a.c != b.c {
                return Err(conform());
            }
            let mut d = a.d;
            d.extend(b.d);
            Ok(Mat::new(a.r + b.r, a.c, d))
        } else {
            if a.r != b.r {
                return Err(conform());
            }
            let mut d = Vec::with_capacity(a.d.len() + b.d.len());
            for i in 0..a.r {
                d.extend_from_slice(&a.d[i * a.c..(i + 1) * a.c]);
                d.extend_from_slice(&b.d[i * b.c..(i + 1) * b.c]);
            }
            Ok(Mat::new(a.r, a.c + b.c, d))
        }
    }
    match (a, b) {
        (Val::R(x), Val::R(y)) => Ok(Val::R(j(row, x, y)?)),
        (Val::S(x), Val::S(y)) => Ok(Val::S(j(row, x, y)?)),
        (Val::P(x), Val::P(y)) => Ok(Val::P(j(row, x, y)?)),
        _ => Err(type_mismatch()),
    }
}

fn range(a: f64, b: f64) -> Vec<f64> {
    let mut v = Vec::new();
    if is_missing(a) || is_missing(b) {
        return v;
    }
    if a <= b {
        let mut x = a;
        while x <= b + 1e-12 {
            v.push(x);
            x += 1.0;
        }
    } else {
        let mut x = a;
        while x >= b - 1e-12 {
            v.push(x);
            x -= 1.0;
        }
    }
    v
}

pub fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> R<Mat<f64>> {
    if a.c != b.r {
        return Err(conform());
    }
    let mut d = vec![0.0; a.r * b.c];
    for i in 0..a.r {
        for k in 0..a.c {
            let x = a.d[i * a.c + k];
            for j in 0..b.c {
                let y = b.d[k * b.c + j];
                let cell = &mut d[i * b.c + j];
                if is_missing(*cell) {
                    continue;
                }
                if is_missing(x) || is_missing(y) {
                    *cell = SYSMISS;
                } else {
                    *cell += x * y;
                }
            }
        }
    }
    Ok(Mat::new(a.r, b.c, d))
}

pub fn binop(op: &str, a: Val, b: Val) -> R<Val> {
    match op {
        "," => return join(false, a, b),
        "\\" => return join(true, a, b),
        "::" | ".." => {
            let v = range(scalar_of(&a)?, scalar_of(&b)?);
            return Ok(Val::R(if op == "::" { Mat::col(v) } else { Mat::row(v) }));
        }
        "&" | "&&" => return Ok(Val::num((truthy(&a)? && truthy(&b)?) as i32 as f64)),
        "|" | "||" => return Ok(Val::num((truthy(&a)? || truthy(&b)?) as i32 as f64)),
        _ => {}
    }
    // elementwise operators
    if let Some(eop) = op.strip_prefix(':') {
        return match (&a, &b) {
            (Val::R(x), Val::R(y)) => match eop {
                "+" | "-" | "*" | "/" | "^" => Ok(Val::R(bcast(x, y, |p, q| arith(eop, *p, *q))?)),
                "&" => Ok(Val::R(bcast(x, y, |p, q| (*p != 0.0 && *q != 0.0) as i32 as f64)?)),
                "|" => Ok(Val::R(bcast(x, y, |p, q| (*p != 0.0 || *q != 0.0) as i32 as f64)?)),
                _ => Ok(Val::R(bcast(x, y, |p, q| cmp_num(eop, *p, *q) as i32 as f64)?)),
            },
            (Val::S(x), Val::S(y)) => match eop {
                "+" => Ok(Val::S(bcast(x, y, |p, q| format!("{p}{q}"))?)),
                "==" | "!=" | ">" | "<" | ">=" | "<=" => Ok(Val::R(bcast(x, y, |p, q| cmp_str(eop, p, q) as i32 as f64)?)),
                _ => Err(type_mismatch()),
            },
            (Val::S(x), Val::R(y)) | (Val::R(y), Val::S(x)) if eop == "*" => Ok(Val::S(bcast(x, y, |p, q| if is_missing(*q) || *q < 1.0 { String::new() } else { p.repeat(*q as usize) })?)),
            (Val::S(_), Val::R(_)) | (Val::R(_), Val::S(_)) if eop == "==" || eop == "!=" => Err(type_mismatch()),
            _ => Err(type_mismatch()),
        };
    }
    match op {
        "==" | "!=" => {
            let eq = match (&a, &b) {
                (Val::R(x), Val::R(y)) => x == y,
                (Val::S(x), Val::S(y)) => x == y,
                (Val::P(x), Val::P(y)) => x.r == y.r && x.c == y.c && x.d.iter().zip(&y.d).all(|(p, q)| ptr_eq(p, q)),
                _ => return Err(type_mismatch()),
            };
            Ok(Val::num(((op == "==") == eq) as i32 as f64))
        }
        ">" | "<" | ">=" | "<=" => {
            let m = binop(&format!(":{op}"), a, b)?;
            match m {
                Val::R(m) => Ok(Val::num(m.d.iter().all(|x| *x != 0.0) as i32 as f64)),
                _ => Err(type_mismatch()),
            }
        }
        "+" | "-" => match (&a, &b) {
            (Val::R(x), Val::R(y)) => {
                if (x.r, x.c) != (y.r, y.c) {
                    return Err(conform());
                }
                Ok(Val::R(bcast(x, y, |p, q| arith(op, *p, *q))?))
            }
            (Val::S(x), Val::S(y)) if op == "+" => {
                if (x.r, x.c) != (y.r, y.c) && !x.is_scalar() && !y.is_scalar() {
                    return Err(conform());
                }
                Ok(Val::S(bcast(x, y, |p, q| format!("{p}{q}"))?))
            }
            _ => Err(type_mismatch()),
        },
        "*" => match (&a, &b) {
            (Val::R(x), Val::R(y)) => {
                if x.is_scalar() || y.is_scalar() {
                    Ok(Val::R(bcast(x, y, |p, q| arith("*", *p, *q))?))
                } else {
                    Ok(Val::R(matmul(x, y)?))
                }
            }
            (Val::S(_), Val::R(_)) | (Val::R(_), Val::S(_)) => binop(":*", a, b),
            _ => Err(type_mismatch()),
        },
        "/" => match (&a, &b) {
            (Val::R(x), Val::R(y)) => {
                if !y.is_scalar() {
                    return Err(conform());
                }
                Ok(Val::R(bcast(x, y, |p, q| arith("/", *p, *q))?))
            }
            _ => Err(type_mismatch()),
        },
        "^" => match (&a, &b) {
            (Val::R(x), Val::R(y)) => {
                if !x.is_scalar() || !y.is_scalar() {
                    return Err(conform());
                }
                Ok(Val::num(arith("^", x.d[0], y.d[0])))
            }
            _ => Err(type_mismatch()),
        },
        "#" => match (&a, &b) {
            (Val::R(x), Val::R(y)) => {
                let (r, c) = (x.r * y.r, x.c * y.c);
                let mut m = Mat::fill(r, c, 0.0);
                for i in 0..x.r {
                    for j in 0..x.c {
                        for k in 0..y.r {
                            for l in 0..y.c {
                                m.set(i * y.r + k, j * y.c + l, arith("*", *x.at(i, j), *y.at(k, l)));
                            }
                        }
                    }
                }
                Ok(Val::R(m))
            }
            _ => Err(type_mismatch()),
        },
        _ => Err(err(3000, format!("operator {op} not supported"))),
    }
}

fn ptr_eq(a: &super::value::Ptr, b: &super::value::Ptr) -> bool {
    use super::value::Ptr;
    match (a, b) {
        (Ptr::Null, Ptr::Null) => true,
        (Ptr::Var(x), Ptr::Var(y)) => std::rc::Rc::ptr_eq(x, y),
        (Ptr::Func(x), Ptr::Func(y)) => x == y,
        _ => false,
    }
}

fn pick<T: Clone>(m: &Mat<T>, rows: &[usize], cols: &[usize]) -> R<Mat<T>> {
    if rows.iter().any(|&i| i >= m.r) || cols.iter().any(|&j| j >= m.c) {
        return Err(err(3301, "subscript invalid"));
    }
    let mut d = Vec::with_capacity(rows.len() * cols.len());
    for &i in rows {
        for &j in cols {
            d.push(m.at(i, j).clone());
        }
    }
    Ok(Mat::new(rows.len(), cols.len(), d))
}

/// Translate one or two subscript lists into row and column lists.
fn resolve(dims: (usize, usize), ix: &[Option<Vec<usize>>]) -> R<(Vec<usize>, Vec<usize>, bool)> {
    let (r, c) = dims;
    let all = |n: usize, s: &Option<Vec<usize>>| s.clone().unwrap_or_else(|| (0..n).collect());
    match ix.len() {
        1 => {
            let list = &ix[0];
            if r == 1 {
                let cols = all(c, list);
                let col_shape = matches!(list, Some(l) if l.len() > 1) && c == 1;
                Ok((vec![0], cols, col_shape))
            } else if c == 1 {
                Ok((all(r, list), vec![0], false))
            } else {
                Err(err(3301, "subscript invalid"))
            }
        }
        2 => Ok((all(r, &ix[0]), all(c, &ix[1]), false)),
        _ => Err(err(3301, "subscript invalid")),
    }
}

pub fn subscript(v: &Val, ix: &[Option<Vec<usize>>]) -> R<Val> {
    let (rows, cols, flip) = resolve(v.dims(), ix)?;
    let out = match v {
        Val::R(m) => Val::R(pick(m, &rows, &cols)?),
        Val::S(m) => Val::S(pick(m, &rows, &cols)?),
        Val::P(m) => Val::P(pick(m, &rows, &cols)?),
        Val::O(_) => return Err(err(3000, "subscripts not allowed with class instance")),
    };
    Ok(if flip { out.transpose() } else { out })
}

/// `[| r1, c1 \ r2, c2 |]` (or `[| i1 \ i2 |]` for vectors) to row/column lists.
pub fn range_spec(v: &Val, spec: &Val) -> R<(Vec<usize>, Vec<usize>)> {
    let Val::R(s) = spec else { return Err(type_mismatch()) };
    let (r, c) = v.dims();
    let bound = |x: f64, def: usize| -> R<usize> {
        if is_missing(x) {
            Ok(def)
        } else if x < 1.0 {
            Err(err(3301, "subscript invalid"))
        } else {
            Ok(x as usize)
        }
    };
    let (r1, c1, r2, c2) = if s.d.len() == 4 && s.r == 2 {
        (bound(s.d[0], 1)?, bound(s.d[1], 1)?, bound(s.d[2], r)?, bound(s.d[3], c)?)
    } else if s.d.len() == 2 {
        if r == 1 {
            (1, bound(s.d[0], 1)?, 1, bound(s.d[1], c)?)
        } else {
            (bound(s.d[0], 1)?, 1, bound(s.d[1], r)?, 1)
        }
    } else {
        return Err(err(3301, "subscript invalid"));
    };
    if r2 > r || c2 > c {
        return Err(err(3301, "subscript invalid"));
    }
    Ok(((r1 - 1..r2).collect(), (c1 - 1..c2).collect()))
}

/// Assign `v` into the selected cells; returns the cells changed.
pub fn sub_assign(cur: &mut Val, ix: &[Option<Vec<usize>>], v: &Val) -> R<Vec<(usize, usize)>> {
    let (rows, cols, _) = resolve(cur.dims(), ix)?;
    let (cr, cc) = cur.dims();
    if rows.iter().any(|&i| i >= cr) || cols.iter().any(|&j| j >= cc) {
        return Err(err(3301, "subscript invalid"));
    }
    let (vr, vc) = v.dims();
    let n = rows.len() * cols.len();
    let scalar = (vr, vc) == (1, 1);
    // a vector source fills a vector target in order
    let linear = !scalar && vr * vc == n && (rows.len() == 1 || cols.len() == 1) && (vr == 1 || vc == 1);
    if !scalar && !linear && (vr, vc) != (rows.len(), cols.len()) {
        return Err(conform());
    }
    let src = |k: usize, a: usize, b: usize| if scalar { 0 } else if linear { k } else { a * vc + b };
    let mut cells = Vec::with_capacity(n);
    fn put<T: Clone>(m: &mut Mat<T>, s: &Mat<T>, rows: &[usize], cols: &[usize], src: &dyn Fn(usize, usize, usize) -> usize, cells: &mut Vec<(usize, usize)>) {
        let mut k = 0;
        for (a, &i) in rows.iter().enumerate() {
            for (b, &j) in cols.iter().enumerate() {
                m.set(i, j, s.d[src(k, a, b)].clone());
                cells.push((i, j));
                k += 1;
            }
        }
    }
    match (cur, v) {
        (Val::R(m), Val::R(s)) => put(m, s, &rows, &cols, &src, &mut cells),
        (Val::S(m), Val::S(s)) => put(m, s, &rows, &cols, &src, &mut cells),
        (Val::P(m), Val::P(s)) => put(m, s, &rows, &cols, &src, &mut cells),
        _ => return Err(type_mismatch()),
    }
    Ok(cells)
}
