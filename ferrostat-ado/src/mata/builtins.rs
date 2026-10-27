//! The function library.

use ferrostat_cmd::matrix::Matrix;
use ferrostat_core::missing::{is_missing, SYSMISS};

use super::eval::{conform, err, type_mismatch, Ev, R};
use super::ops;
use super::value::{fmt_num, Mat, Obj, Val, XlBook};

pub fn real(v: &Val) -> R<&Mat<f64>> {
    match v {
        Val::R(m) => Ok(m),
        _ => Err(err(3253, "nonreal found where real required")),
    }
}

pub fn strm(v: &Val) -> R<&Mat<String>> {
    match v {
        Val::S(m) => Ok(m),
        _ => Err(err(3255, "real found where string required")),
    }
}

pub fn sc(v: &Val) -> R<f64> {
    let m = real(v)?;
    if !m.is_scalar() {
        return Err(err(3204, "matrix found where scalar required"));
    }
    Ok(m.d[0])
}

pub fn ss(v: &Val) -> R<String> {
    let m = strm(v)?;
    if !m.is_scalar() {
        return Err(err(3204, "matrix found where scalar required"));
    }
    Ok(m.d[0].clone())
}

fn nargs(a: &[Val], lo: usize, hi: usize, name: &str) -> R<()> {
    if a.len() < lo || a.len() > hi {
        let msg = if lo == hi { format!("expected {lo} arguments but received {}", a.len()) } else { format!("expected {lo} to {hi} arguments but received {}", a.len()) };
        return Err(super::eval::MErr { rc: 3001, msg, trace: vec![name.to_string()] });
    }
    Ok(())
}

fn some(v: Val) -> R<Option<Val>> {
    Ok(Some(v))
}

fn rv(m: Mat<f64>) -> R<Option<Val>> {
    Ok(Some(Val::R(m)))
}

fn num(x: f64) -> R<Option<Val>> {
    Ok(Some(Val::num(x)))
}

fn to_matrix(m: &Mat<f64>) -> Matrix {
    let mut x = Matrix::zeros(m.r, m.c);
    x.data = m.d.clone();
    x
}

fn from_matrix(m: &Matrix) -> Mat<f64> {
    Mat::new(m.rows, m.cols, m.data.clone())
}

fn elementwise(v: &Val, f: impl Fn(f64) -> f64) -> R<Option<Val>> {
    let m = real(v)?;
    rv(m.map(|x| if is_missing(*x) { SYSMISS } else {
        let y = f(*x);
        if y.is_finite() { y } else { SYSMISS }
    }))
}

fn str_elementwise(v: &Val, f: impl Fn(&str) -> String) -> R<Option<Val>> {
    some(Val::S(strm(v)?.map(|s| f(s))))
}

fn str_to_num(v: &Val, f: impl Fn(&str) -> f64) -> R<Option<Val>> {
    rv(strm(v)?.map(|s| f(s)))
}

/// Sum ignoring missing values.
fn msum<'a>(it: impl Iterator<Item = &'a f64>) -> f64 {
    it.filter(|x| !is_missing(**x)).sum()
}

fn col(m: &Mat<f64>, j: usize) -> impl Iterator<Item = &f64> {
    (0..m.r).map(move |i| m.at(i, j))
}

fn row(m: &Mat<f64>, i: usize) -> impl Iterator<Item = &f64> {
    m.d[i * m.c..(i + 1) * m.c].iter()
}

fn opt_min(it: impl Iterator<Item = f64>, max: bool) -> f64 {
    let mut best: Option<f64> = None;
    for x in it.filter(|x| !is_missing(*x)) {
        best = Some(match best {
            None => x,
            Some(b) => {
                if (max && x > b) || (!max && x < b) {
                    x
                } else {
                    b
                }
            }
        });
    }
    best.unwrap_or(SYSMISS)
}

/// Rows without any missing value.
fn complete_rows(ms: &[&Mat<f64>]) -> Vec<usize> {
    let n = ms[0].r;
    (0..n).filter(|&i| ms.iter().all(|m| (0..m.c).all(|j| !is_missing(*m.at(i, j))))).collect()
}

fn cross(x: &Mat<f64>, xc: bool, w: Option<&Mat<f64>>, z: &Mat<f64>, zc: bool) -> R<Mat<f64>> {
    if x.r != z.r {
        return Err(conform());
    }
    let mut parts = vec![x, z];
    if let Some(w) = w {
        parts.push(w);
    }
    let keep = complete_rows(&parts);
    let ax = |i: usize, j: usize| if j < x.c { *x.at(i, j) } else { 1.0 };
    let az = |i: usize, j: usize| if j < z.c { *z.at(i, j) } else { 1.0 };
    let kx = x.c + xc as usize;
    let kz = z.c + zc as usize;
    let mut out = Mat::fill(kx, kz, 0.0);
    for &i in &keep {
        let wi = w.map(|w| *w.at(i, 0)).unwrap_or(1.0);
        for a in 0..kx {
            let xa = ax(i, a) * wi;
            for b in 0..kz {
                out.d[a * kz + b] += xa * az(i, b);
            }
        }
    }
    Ok(out)
}

fn sort_rows<T: Clone>(m: &Mat<T>, keys: &[(usize, bool)], cmp: impl Fn(&T, &T) -> std::cmp::Ordering) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..m.r).collect();
    idx.sort_by(|&a, &b| {
        for &(k, desc) in keys {
            let o = cmp(m.at(a, k), m.at(b, k));
            let o = if desc { o.reverse() } else { o };
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    idx
}

fn sort_keys(v: &Val, ncols: usize) -> R<Vec<(usize, bool)>> {
    let m = real(v)?;
    let mut keys = Vec::new();
    for x in &m.d {
        let k = x.abs() as usize;
        if k < 1 || k > ncols {
            return Err(err(3300, "argument out of range"));
        }
        keys.push((k - 1, *x < 0.0));
    }
    Ok(keys)
}

pub fn permute(v: &Val, idx: &[usize]) -> Val {
    let ix = [Some(idx.to_vec()), None];
    ops::subscript(v, &ix).unwrap_or_else(|_| v.clone())
}

pub fn sort_val(v: &Val, keys: &Val) -> R<Val> {
    let keys = sort_keys(keys, v.cols())?;
    let idx = match v {
        Val::R(m) => sort_rows(m, &keys, |a, b| a.total_cmp(b)),
        Val::S(m) => sort_rows(m, &keys, |a, b| a.cmp(b)),
        _ => return Err(type_mismatch()),
    };
    Ok(permute(v, &idx))
}

/// `sprintf`-style formatting with display formats (`%9.2f`, `%s`, `%g`).
pub fn format(fmt: &str, args: &[Val]) -> R<String> {
    let c: Vec<char> = fmt.chars().collect();
    let mut out = String::new();
    let mut k = 0;
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch == '\\' && i + 1 < c.len() {
            match c[i + 1] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                'r' => {}
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
            i += 2;
            continue;
        }
        if ch == '%' {
            if c.get(i + 1) == Some(&'%') {
                out.push('%');
                i += 2;
                continue;
            }
            let st = i;
            i += 1;
            while i < c.len() && "-~0123456789.,".contains(c[i]) {
                i += 1;
            }
            // type letter(s): f g e s d x, plus a trailing c for commas, or t.. dates
            if i < c.len() && c[i] == 't' {
                i += 1;
                while i < c.len() && c[i].is_ascii_alphabetic() {
                    i += 1;
                }
            } else if i < c.len() {
                i += 1;
                if i < c.len() && c[i] == 'c' && "fg".contains(c[i - 1]) {
                    i += 1;
                }
            }
            let spec: String = c[st..i].iter().collect();
            let arg = args.get(k).ok_or_else(|| err(3001, "too few arguments for format"))?;
            k += 1;
            out.push_str(&format_one(&spec, arg)?);
            continue;
        }
        out.push(ch);
        i += 1;
    }
    Ok(out)
}

fn format_one(spec: &str, arg: &Val) -> R<String> {
    let kind = spec.trim_end_matches('c').chars().last().unwrap_or('g');
    let body = &spec[1..];
    let left = body.starts_with('-');
    let width: usize = body.trim_start_matches(['-', '~', '0']).split(|c: char| !c.is_ascii_digit()).next().and_then(|w| w.parse().ok()).unwrap_or(0);
    let pad = |s: String| {
        let n = s.chars().count();
        if n >= width {
            s
        } else if left {
            format!("{s}{}", " ".repeat(width - n))
        } else {
            format!("{}{s}", " ".repeat(width - n))
        }
    };
    match arg {
        Val::S(m) => {
            let s = m.d.first().cloned().unwrap_or_default();
            Ok(pad(s))
        }
        Val::R(m) => {
            let x = m.d.first().copied().unwrap_or(SYSMISS);
            if kind == 's' {
                return Ok(pad(fmt_num(x)));
            }
            if kind == 'd' {
                return Ok(pad(if is_missing(x) { ".".into() } else { format!("{}", x.round() as i64) }));
            }
            let has_digits = body.chars().any(|c| c.is_ascii_digit());
            if !has_digits {
                return Ok(match kind {
                    'f' if !is_missing(x) => format!("{x:.6}"),
                    'e' if !is_missing(x) => ferrostat_cmd::format::fmt_e(x, 6),
                    _ => fmt_num(x),
                });
            }
            Ok(ferrostat_cmd::format::fmt(x, spec))
        }
        _ => Err(type_mismatch()),
    }
}

fn tokens(s: &str, parse: &str) -> Vec<String> {
    let mut out = Vec::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let is_sep = |ch: char| parse.contains(ch);
    while i < c.len() {
        let ch = c[i];
        if ch == ' ' && parse.contains(' ') {
            i += 1;
            continue;
        }
        if is_sep(ch) {
            out.push(ch.to_string());
            i += 1;
            continue;
        }
        if ch == '"' {
            let st = i;
            i += 1;
            while i < c.len() && c[i] != '"' {
                i += 1;
            }
            i = (i + 1).min(c.len());
            out.push(c[st..i].iter().collect());
            continue;
        }
        let st = i;
        while i < c.len() && !is_sep(c[i]) {
            i += 1;
        }
        out.push(c[st..i].iter().collect());
    }
    out
}

/// Evaluate a scalar function of the command language elementwise.
fn host_function(ev: &mut Ev, name: &str, a: &[Val]) -> R<Option<Val>> {
    use ferrostat_cmd::eval::{Env, Val as HVal};
    use ferrostat_parser::Expr as HExpr;
    let mut dims = (1, 1);
    for v in a {
        let d = v.dims();
        if d != (1, 1) {
            if dims != (1, 1) && dims != d {
                return Err(conform());
            }
            dims = d;
        }
        if matches!(v, Val::P(_) | Val::O(_)) {
            return Err(type_mismatch());
        }
    }
    let n = dims.0 * dims.1;
    let mut nums = Vec::new();
    let mut strs = Vec::new();
    for k in 0..n {
        let args: Vec<HExpr> = a
            .iter()
            .map(|v| match v {
                Val::R(m) => {
                    let x = if m.d.len() == 1 { m.d[0] } else { m.d.get(k).copied().unwrap_or(SYSMISS) };
                    if is_missing(x) { HExpr::Missing } else { HExpr::Number(x) }
                }
                Val::S(m) => HExpr::String(if m.d.len() == 1 { m.d[0].clone() } else { m.d.get(k).cloned().unwrap_or_default() }),
                _ => HExpr::Missing,
            })
            .collect();
        let e = HExpr::Call(name.to_string(), args);
        let eng = &ev.me.engine;
        match Env::with_store(&eng.dataset, &eng.store).eval(&e, 0) {
            Ok(HVal::Num(x)) => nums.push(x),
            Ok(HVal::Str(s)) => strs.push(s),
            Err(_) if k == 0 => return Err(err(3499, format!("{name}() not found"))),
            Err(e) => return Err(err(crate::interpreter::rc_of(&e), crate::interpreter::err_message(&e))),
        }
    }
    if !strs.is_empty() {
        return some(Val::S(Mat::new(dims.0, dims.1, strs)));
    }
    rv(Mat::new(dims.0, dims.1, nums))
}

pub fn call(ev: &mut Ev, name: &str, a: Vec<Val>, omitted: &[bool]) -> R<Option<Val>> {
    if let Some(r) = super::host::call(ev, name, &a, omitted)? {
        return Ok(r);
    }
    let a1 = || a.first().ok_or_else(|| err(3001, format!("{name}(): too few arguments")));
    let arg = |k: usize| a.get(k).filter(|_| !omitted.get(k).copied().unwrap_or(false));
    match name {
        // ---- dimensions and types ----
        "rows" => num(a1()?.rows() as f64),
        "cols" => num(a1()?.cols() as f64),
        "length" => {
            let (r, c) = a1()?.dims();
            num((r * c) as f64)
        }
        "eltype" => some(Val::str(a1()?.eltype())),
        "orgtype" => some(Val::str(a1()?.orgtype())),
        "isreal" => num(matches!(a1()?, Val::R(_)) as i32 as f64),
        "isstring" => num(matches!(a1()?, Val::S(_)) as i32 as f64),
        "ispointer" => num(matches!(a1()?, Val::P(_)) as i32 as f64),
        "iscomplex" => num(0.0),
        "isscalar" => num((a1()?.dims() == (1, 1)) as i32 as f64),
        "isvector" => {
            let (r, c) = a1()?.dims();
            num((r == 1 || c == 1) as i32 as f64)
        }
        "isrowvector" => num((a1()?.rows() == 1) as i32 as f64),
        "iscolvector" => num((a1()?.cols() == 1) as i32 as f64),
        "issquare" => {
            let (r, c) = a1()?.dims();
            num((r == c) as i32 as f64)
        }
        "issymmetric" => {
            let m = real(a1()?)?;
            num((m.r == m.c && (0..m.r).all(|i| (0..m.c).all(|j| m.at(i, j) == m.at(j, i)))) as i32 as f64)
        }
        "isview" => num(0.0),
        "args" => num(ev.nargs() as f64),
        // ---- creation ----
        "J" => {
            nargs(&a, 3, 3, name)?;
            let r = sc(&a[0])? as usize;
            let c = sc(&a[1])? as usize;
            let tile = |v: &Val| -> R<Val> {
                let (vr, vc) = v.dims();
                let mut out = Val::empty();
                if r == 0 || c == 0 {
                    return Ok(match v {
                        Val::S(_) => Val::S(Mat::new(r * vr, c * vc, Vec::new())),
                        _ => Val::R(Mat::new(r * vr, c * vc, Vec::new())),
                    });
                }
                for _ in 0..r {
                    let mut line = v.clone();
                    for _ in 1..c {
                        line = ops::join(false, line, v.clone())?;
                    }
                    out = if out.dims() == (0, 0) { line } else { ops::join(true, out, line)? };
                }
                Ok(out)
            };
            match &a[2] {
                Val::R(m) if m.is_scalar() => rv(Mat::fill(r, c, m.d[0])),
                Val::S(m) if m.is_scalar() => some(Val::S(Mat::fill(r, c, m.d[0].clone()))),
                v => some(tile(v)?),
            }
        }
        "I" => {
            let r = sc(a1()?)? as usize;
            let c = match arg(1) {
                Some(v) => sc(v)? as usize,
                None => r,
            };
            let mut m = Mat::fill(r, c, 0.0);
            for i in 0..r.min(c) {
                m.set(i, i, 1.0);
            }
            rv(m)
        }
        "e" => {
            nargs(&a, 2, 2, name)?;
            let i = sc(&a[0])? as usize;
            let n = sc(&a[1])? as usize;
            if i < 1 || i > n {
                return Err(err(3300, "argument out of range"));
            }
            let mut m = Mat::fill(1, n, 0.0);
            m.set(0, i - 1, 1.0);
            rv(m)
        }
        "range" => {
            nargs(&a, 3, 3, name)?;
            let (lo, hi, d) = (sc(&a[0])?, sc(&a[1])?, sc(&a[2])?.abs());
            let mut v = Vec::new();
            if d > 0.0 {
                let n = ((hi - lo).abs() / d + 1e-9).floor() as usize;
                for k in 0..=n {
                    v.push(if hi >= lo { lo + k as f64 * d } else { lo - k as f64 * d });
                }
            }
            rv(Mat::col(v))
        }
        "rangen" => {
            nargs(&a, 3, 3, name)?;
            let (lo, hi, n) = (sc(&a[0])?, sc(&a[1])?, sc(&a[2])? as usize);
            let v = (0..n).map(|k| if n == 1 { lo } else { lo + (hi - lo) * k as f64 / (n - 1) as f64 }).collect();
            rv(Mat::col(v))
        }
        "runiform" | "uniform" | "rnormal" | "runiformint" => {
            let r = sc(a1()?)? as usize;
            let c = sc(a.get(1).ok_or_else(|| err(3001, "too few arguments"))?)? as usize;
            let p1 = a.get(2).map(sc).transpose()?;
            let p2 = a.get(3).map(sc).transpose()?;
            let mut rng = ev.me.engine.store.rng.borrow_mut();
            let d = (0..r * c)
                .map(|_| match name {
                    "rnormal" => p1.unwrap_or(0.0) + p2.unwrap_or(1.0) * rng.normal(),
                    "runiformint" => rng.uniform_int(p1.unwrap_or(0.0), p2.unwrap_or(1.0)),
                    _ => match (p1, p2) {
                        (Some(lo), Some(hi)) => rng.uniform_ab(lo, hi),
                        _ => rng.uniform(),
                    },
                })
                .collect();
            rv(Mat::new(r, c, d))
        }
        "rseed" => {
            if let Some(v) = arg(0) {
                ev.me.engine.store.rng.borrow_mut().reseed(sc(v)? as u64);
                return Ok(None);
            }
            some(Val::str(ev.me.engine.store.rng.borrow().state()))
        }
        "diag" => {
            let m = real(a1()?)?;
            if m.r == 1 || m.c == 1 {
                let n = m.d.len();
                let mut out = Mat::fill(n, n, 0.0);
                for i in 0..n {
                    out.set(i, i, m.d[i]);
                }
                rv(out)
            } else {
                let mut out = Mat::fill(m.r, m.c, 0.0);
                for i in 0..m.r.min(m.c) {
                    out.set(i, i, *m.at(i, i));
                }
                rv(out)
            }
        }
        "diagonal" => {
            let m = real(a1()?)?;
            rv(Mat::col((0..m.r.min(m.c)).map(|i| *m.at(i, i)).collect()))
        }
        "vec" => match a1()? {
            Val::R(m) => rv(Mat::col(m.t().d)),
            Val::S(m) => some(Val::S(Mat::col(m.t().d))),
            _ => Err(type_mismatch()),
        },
        "vech" => {
            let m = real(a1()?)?;
            let mut v = Vec::new();
            for j in 0..m.c {
                for i in j..m.r {
                    v.push(*m.at(i, j));
                }
            }
            rv(Mat::col(v))
        }
        "rowshape" | "colshape" => {
            let v = a1()?;
            let k = sc(a.get(1).ok_or_else(|| err(3001, "too few arguments"))?)? as usize;
            let (r, c) = v.dims();
            let n = r * c;
            if k == 0 || n % k != 0 {
                return Err(conform());
            }
            let (nr, nc) = if name == "rowshape" { (k, n / k) } else { (n / k, k) };
            match v {
                Val::R(m) => rv(Mat::new(nr, nc, m.d.clone())),
                Val::S(m) => some(Val::S(Mat::new(nr, nc, m.d.clone()))),
                _ => Err(type_mismatch()),
            }
        }
        "makesymmetric" => {
            let mut m = real(a1()?)?.clone();
            for i in 0..m.r {
                for j in 0..i {
                    let v = *m.at(i, j);
                    m.set(j, i, v);
                }
            }
            rv(m)
        }
        "lowertriangle" | "uppertriangle" => {
            let mut m = real(a1()?)?.clone();
            for i in 0..m.r {
                for j in 0..m.c {
                    if (name == "lowertriangle" && j > i) || (name == "uppertriangle" && j < i) {
                        m.set(i, j, 0.0);
                    }
                }
            }
            rv(m)
        }
        // ---- sums and summaries ----
        "sum" | "quadsum" => num(msum(real(a1()?)?.d.iter())),
        "colsum" | "quadcolsum" => {
            let m = real(a1()?)?;
            rv(Mat::row((0..m.c).map(|j| msum(col(m, j))).collect()))
        }
        "rowsum" | "quadrowsum" => {
            let m = real(a1()?)?;
            rv(Mat::col((0..m.r).map(|i| msum(row(m, i))).collect()))
        }
        "runningsum" => {
            let m = real(a1()?)?;
            let mut s = 0.0;
            let d = m.d.iter().map(|x| {
                if !is_missing(*x) {
                    s += x;
                }
                s
            }).collect();
            rv(Mat::new(m.r, m.c, d))
        }
        "mean" => {
            let m = real(a1()?)?;
            let keep = complete_rows(&[m]);
            let n = keep.len() as f64;
            rv(Mat::row((0..m.c).map(|j| if n == 0.0 { SYSMISS } else { keep.iter().map(|&i| *m.at(i, j)).sum::<f64>() / n }).collect()))
        }
        "variance" | "quadvariance" | "correlation" | "quadcorrelation" => {
            let m = real(a1()?)?;
            let keep = complete_rows(&[m]);
            let n = keep.len() as f64;
            let mu: Vec<f64> = (0..m.c).map(|j| keep.iter().map(|&i| *m.at(i, j)).sum::<f64>() / n).collect();
            let mut v = Mat::fill(m.c, m.c, SYSMISS);
            if n > 1.0 {
                for a in 0..m.c {
                    for b in 0..m.c {
                        let s: f64 = keep.iter().map(|&i| (m.at(i, a) - mu[a]) * (m.at(i, b) - mu[b])).sum();
                        v.set(a, b, s / (n - 1.0));
                    }
                }
                if name.contains("correlation") {
                    let d: Vec<f64> = (0..m.c).map(|a| v.at(a, a).sqrt()).collect();
                    for a in 0..m.c {
                        for b in 0..m.c {
                            let x = v.at(a, b) / (d[a] * d[b]);
                            v.set(a, b, x);
                        }
                    }
                }
            }
            rv(v)
        }
        "min" | "max" => num(opt_min(real(a1()?)?.d.iter().copied(), name == "max")),
        "colmin" | "colmax" => {
            let m = real(a1()?)?;
            rv(Mat::row((0..m.c).map(|j| opt_min(col(m, j).copied(), name == "colmax")).collect()))
        }
        "rowmin" | "rowmax" => {
            let m = real(a1()?)?;
            rv(Mat::col((0..m.r).map(|i| opt_min(row(m, i).copied(), name == "rowmax")).collect()))
        }
        "minmax" => {
            let m = real(a1()?)?;
            rv(Mat::row(vec![opt_min(m.d.iter().copied(), false), opt_min(m.d.iter().copied(), true)]))
        }
        "colminmax" | "colmaxabs" => {
            let m = real(a1()?)?;
            if name == "colmaxabs" {
                return rv(Mat::row((0..m.c).map(|j| opt_min(col(m, j).map(|x| x.abs()), true)).collect()));
            }
            let mut d = Vec::new();
            d.extend((0..m.c).map(|j| opt_min(col(m, j).copied(), false)));
            d.extend((0..m.c).map(|j| opt_min(col(m, j).copied(), true)));
            rv(Mat::new(2, m.c, d))
        }
        "missing" => {
            let n = match a1()? {
                Val::R(m) => m.d.iter().filter(|x| is_missing(**x)).count(),
                Val::S(m) => m.d.iter().filter(|x| x.is_empty()).count(),
                _ => 0,
            };
            num(n as f64)
        }
        "nonmissing" => {
            let n = match a1()? {
                Val::R(m) => m.d.iter().filter(|x| !is_missing(**x)).count(),
                Val::S(m) => m.d.iter().filter(|x| !x.is_empty()).count(),
                _ => 0,
            };
            num(n as f64)
        }
        "hasmissing" => {
            let n = match a1()? {
                Val::R(m) => m.d.iter().any(|x| is_missing(*x)),
                Val::S(m) => m.d.iter().any(|x| x.is_empty()),
                _ => false,
            };
            num(n as i32 as f64)
        }
        "colmissing" | "colnonmissing" => {
            let m = real(a1()?)?;
            let want = name == "colmissing";
            rv(Mat::row((0..m.c).map(|j| col(m, j).filter(|x| is_missing(**x) == want).count() as f64).collect()))
        }
        "rowmissing" | "rownonmissing" => {
            let m = real(a1()?)?;
            let want = name == "rowmissing";
            rv(Mat::col((0..m.r).map(|i| row(m, i).filter(|x| is_missing(**x) == want).count() as f64).collect()))
        }
        "any" => num(real(a1()?)?.d.iter().any(|x| *x != 0.0) as i32 as f64),
        "all" => num(real(a1()?)?.d.iter().all(|x| *x != 0.0) as i32 as f64),
        "anyof" | "allof" => {
            nargs(&a, 2, 2, name)?;
            let eq: Vec<bool> = match (&a[0], &a[1]) {
                (Val::R(m), Val::R(v)) if v.is_scalar() => m.d.iter().map(|x| *x == v.d[0]).collect(),
                (Val::S(m), Val::S(v)) if v.is_scalar() => m.d.iter().map(|x| *x == v.d[0]).collect(),
                _ => return Err(type_mismatch()),
            };
            num(if name == "anyof" { eq.iter().any(|x| *x) } else { eq.iter().all(|x| *x) } as i32 as f64)
        }
        // ---- elementwise math ----
        "sqrt" => elementwise(a1()?, |x| if x < 0.0 { f64::NAN } else { x.sqrt() }),
        "exp" => elementwise(a1()?, f64::exp),
        "ln" | "log" => elementwise(a1()?, |x| if x <= 0.0 { f64::NAN } else { x.ln() }),
        "log10" => elementwise(a1()?, |x| if x <= 0.0 { f64::NAN } else { x.log10() }),
        "abs" => elementwise(a1()?, f64::abs),
        "sign" => elementwise(a1()?, |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        "floor" => elementwise(a1()?, f64::floor),
        "ceil" => elementwise(a1()?, f64::ceil),
        "trunc" => elementwise(a1()?, f64::trunc),
        "sin" => elementwise(a1()?, f64::sin),
        "cos" => elementwise(a1()?, f64::cos),
        "tan" => elementwise(a1()?, f64::tan),
        "asin" => elementwise(a1()?, f64::asin),
        "acos" => elementwise(a1()?, f64::acos),
        "atan" => elementwise(a1()?, f64::atan),
        "round" => {
            let u = match arg(1) {
                Some(v) => sc(v)?,
                None => 1.0,
            };
            elementwise(a1()?, |x| (x / u + 0.5).floor() * u)
        }
        "mod" => {
            nargs(&a, 2, 2, name)?;
            let r = ops::binop(":/", a[0].clone(), a[1].clone())?;
            let q = match r {
                Val::R(m) => m.map(|x| if is_missing(*x) { SYSMISS } else { x.floor() }),
                _ => return Err(type_mismatch()),
            };
            let p = ops::binop(":*", Val::R(q), a[1].clone())?;
            some(ops::binop(":-", a[0].clone(), p)?)
        }
        "pi" => num(std::f64::consts::PI),
        "editmissing" => {
            nargs(&a, 2, 2, name)?;
            let v = sc(&a[1])?;
            rv(real(&a[0])?.map(|x| if is_missing(*x) { v } else { *x }))
        }
        "editvalue" => {
            nargs(&a, 3, 3, name)?;
            match (&a[0], &a[1], &a[2]) {
                (Val::R(m), f, t) => {
                    let (f, t) = (sc(f)?, sc(t)?);
                    rv(m.map(|x| if *x == f { t } else { *x }))
                }
                (Val::S(m), f, t) => {
                    let (f, t) = (ss(f)?, ss(t)?);
                    some(Val::S(m.map(|x| if *x == f { t.clone() } else { x.clone() })))
                }
                _ => Err(type_mismatch()),
            }
        }
        // ---- linear algebra ----
        "invsym" => {
            let m = real(a1()?)?;
            if m.r != m.c {
                return Err(err(3205, "square matrix required"));
            }
            if m.d.iter().any(|x| is_missing(*x)) {
                return rv(Mat::fill(m.r, m.c, SYSMISS));
            }
            let (inv, _) = to_matrix(m).sym_inverse(1e-13);
            rv(from_matrix(&inv))
        }
        "luinv" | "inv" | "qrinv" | "cholinv" | "pinv" => {
            let m = real(a1()?)?;
            if m.r != m.c {
                return Err(err(3205, "square matrix required"));
            }
            match to_matrix(m).inverse() {
                Ok(inv) if !m.d.iter().any(|x| is_missing(*x)) => rv(from_matrix(&inv)),
                _ => rv(Mat::fill(m.r, m.c, SYSMISS)),
            }
        }
        "cholesky" => {
            let m = real(a1()?)?;
            if m.r != m.c {
                return Err(err(3205, "square matrix required"));
            }
            match to_matrix(m).cholesky() {
                Ok(l) => rv(from_matrix(&l)),
                Err(_) => rv(Mat::fill(m.r, m.c, SYSMISS)),
            }
        }
        "det" | "dettriangular" => {
            let m = real(a1()?)?;
            if m.r != m.c {
                return Err(err(3205, "square matrix required"));
            }
            num(to_matrix(m).det())
        }
        "trace" => {
            let m = real(a1()?)?;
            num((0..m.r.min(m.c)).map(|i| *m.at(i, i)).sum())
        }
        "rank" => {
            let m = real(a1()?)?;
            num(rank(m) as f64)
        }
        "lusolve" | "cholsolve" | "qrsolve" | "svsolve" => {
            nargs(&a, 2, 3, name)?;
            let am = real(&a[0])?;
            let b = real(&a[1])?;
            match to_matrix(am).inverse() {
                Ok(inv) if am.r == am.c => rv(ops::matmul(&from_matrix(&inv), b)?),
                _ => rv(Mat::fill(am.c, b.c, SYSMISS)),
            }
        }
        "symeigenvalues" | "eigenvalues" => {
            let m = real(a1()?)?;
            let (vals, _) = to_matrix(m).sym_eigen();
            rv(Mat::row(vals))
        }
        "cross" | "quadcross" => match a.len() {
            2 => rv(cross(real(&a[0])?, false, None, real(&a[1])?, false)?),
            3 => {
                let w = real(&a[1])?;
                let w = if w.is_scalar() { None } else { Some(w) };
                rv(cross(real(&a[0])?, false, w, real(&a[2])?, false)?)
            }
            4 => rv(cross(real(&a[0])?, sc(&a[1])? != 0.0, None, real(&a[2])?, sc(&a[3])? != 0.0)?),
            5 => {
                let w = real(&a[2])?;
                let w = if w.is_scalar() { None } else { Some(w) };
                rv(cross(real(&a[0])?, sc(&a[1])? != 0.0, w, real(&a[3])?, sc(&a[4])? != 0.0)?)
            }
            _ => Err(err(3001, "cross(): wrong number of arguments")),
        },
        "transposeonly" => some(a1()?.transpose()),
        // ---- selection and ordering ----
        "select" => {
            nargs(&a, 2, 2, name)?;
            let v = real(&a[1])?;
            let (r, c) = a[0].dims();
            let keep: Vec<usize> = (0..v.d.len()).filter(|&k| v.d[k] != 0.0).collect();
            if v.c == 1 && v.r == r {
                some(ops::subscript(&a[0], &[Some(keep), None])?)
            } else if v.r == 1 && v.c == c {
                some(ops::subscript(&a[0], &[None, Some(keep)])?)
            } else {
                Err(conform())
            }
        }
        "selectindex" => {
            let v = real(a1()?)?;
            let idx: Vec<f64> = (0..v.d.len()).filter(|&k| v.d[k] != 0.0).map(|k| (k + 1) as f64).collect();
            rv(if v.r == 1 && v.c != 1 { Mat::row(idx) } else { Mat::col(idx) })
        }
        "sort" => {
            nargs(&a, 2, 2, name)?;
            some(sort_val(&a[0], &a[1])?)
        }
        "order" => {
            nargs(&a, 2, 2, name)?;
            let keys = sort_keys(&a[1], a[0].cols())?;
            let idx = match &a[0] {
                Val::R(m) => sort_rows(m, &keys, |x, y| x.total_cmp(y)),
                Val::S(m) => sort_rows(m, &keys, |x, y| x.cmp(y)),
                _ => return Err(type_mismatch()),
            };
            rv(Mat::col(idx.iter().map(|i| (i + 1) as f64).collect()))
        }
        "revorder" | "invorder" => {
            let m = real(a1()?)?;
            let n = m.d.len();
            if name == "revorder" {
                rv(Mat::col((0..n).map(|k| m.d[n - 1 - k]).collect()))
            } else {
                let mut out = vec![0.0; n];
                for (k, x) in m.d.iter().enumerate() {
                    out[*x as usize - 1] = (k + 1) as f64;
                }
                rv(Mat::col(out))
            }
        }
        "jumble" => {
            let v = a1()?;
            let n = v.rows();
            let mut idx: Vec<usize> = (0..n).collect();
            {
                let mut rng = ev.me.engine.store.rng.borrow_mut();
                for i in (1..n).rev() {
                    let j = (rng.uniform() * (i + 1) as f64) as usize;
                    idx.swap(i, j.min(i));
                }
            }
            some(permute(v, &idx))
        }
        "uniqrows" => {
            let v = a1()?;
            let keys: Vec<f64> = (1..=v.cols()).map(|k| k as f64).collect();
            let s = sort_val(v, &Val::R(Mat::row(keys)))?;
            let n = s.rows();
            let mut keep = Vec::new();
            for i in 0..n {
                let same = i > 0
                    && match &s {
                        Val::R(m) => row(m, i).zip(row(m, i - 1)).all(|(x, y)| x == y),
                        Val::S(m) => (0..m.c).all(|j| m.at(i, j) == m.at(i - 1, j)),
                        _ => false,
                    };
                if !same {
                    keep.push(i);
                }
            }
            some(permute(&s, &keep))
        }
        // ---- strings ----
        "strlen" | "ustrlen" | "udstrlen" => str_to_num(a1()?, |s| if name == "strlen" { s.len() as f64 } else { s.chars().count() as f64 }),
        "strupper" | "ustrupper" => str_elementwise(a1()?, |s| s.to_uppercase()),
        "strlower" | "ustrlower" => str_elementwise(a1()?, |s| s.to_lowercase()),
        "strtrim" | "ustrtrim" => str_elementwise(a1()?, |s| s.trim().to_string()),
        "strltrim" | "ustrltrim" => str_elementwise(a1()?, |s| s.trim_start().to_string()),
        "strrtrim" | "ustrrtrim" => str_elementwise(a1()?, |s| s.trim_end().to_string()),
        "stritrim" => str_elementwise(a1()?, |s| s.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")),
        "strreverse" | "ustrreverse" => str_elementwise(a1()?, |s| s.chars().rev().collect()),
        "strproper" | "ustrproper" => host_function(ev, "proper", &a),
        "substr" | "usubstr" => {
            nargs(&a, 2, 3, name)?;
            let b = sc(&a[1])?;
            let l = match a.get(2) {
                Some(v) => sc(v)?,
                None => SYSMISS,
            };
            str_elementwise(&a[0], |s| {
                let c: Vec<char> = s.chars().collect();
                let n = c.len() as i64;
                let mut st = b as i64;
                if st < 0 {
                    st += n + 1;
                }
                if st < 1 || st > n {
                    return String::new();
                }
                let len = if is_missing(l) { n } else { l as i64 };
                let end = (st - 1 + len.max(0)).min(n);
                c[(st - 1) as usize..end as usize].iter().collect()
            })
        }
        "strpos" | "ustrpos" => {
            nargs(&a, 2, 2, name)?;
            let t = ss(&a[1])?;
            str_to_num(&a[0], |s| s.find(&t).map(|i| s[..i].chars().count() as f64 + 1.0).unwrap_or(0.0))
        }
        "strmatch" => {
            nargs(&a, 2, 2, name)?;
            host_function(ev, "strmatch", &a)
        }
        "subinstr" | "usubinstr" => {
            nargs(&a, 3, 4, name)?;
            let (f, t) = (ss(&a[1])?, ss(&a[2])?);
            let cnt = match a.get(3) {
                Some(v) => sc(v)?,
                None => SYSMISS,
            };
            str_elementwise(&a[0], |s| if is_missing(cnt) { s.replace(&f, &t) } else { s.replacen(&f, &t, cnt as usize) })
        }
        "subinword" => host_function(ev, name, &a),
        "strdup" => {
            nargs(&a, 2, 2, name)?;
            some(ops::binop(":*", a[0].clone(), a[1].clone())?)
        }
        "tokens" => {
            let s = ss(a1()?)?;
            let parse = match arg(1) {
                Some(v) => ss(v)?,
                None => " ".to_string(),
            };
            some(Val::S(Mat::row(tokens(&s, &parse))))
        }
        "invtokens" => {
            let m = strm(a1()?)?;
            let sep = match arg(1) {
                Some(v) => ss(v)?,
                None => " ".to_string(),
            };
            some(Val::str(m.d.join(&sep)))
        }
        "strofreal" => {
            let f = match arg(1) {
                Some(v) => Some(ss(v)?),
                None => None,
            };
            let m = real(a1()?)?;
            some(Val::S(m.map(|x| match &f {
                Some(f) => ferrostat_cmd::format::fmt(*x, f).trim().to_string(),
                None => ferrostat_cmd::format::fmt_g(*x, 8),
            })))
        }
        "strtoreal" => str_to_num(a1()?, |s| s.trim().parse::<f64>().unwrap_or(SYSMISS)),
        "char" => {
            let m = real(a1()?)?;
            some(Val::str(m.d.iter().filter_map(|x| char::from_u32(*x as u32)).collect::<String>()))
        }
        "ascii" => {
            let s = ss(a1()?)?;
            rv(Mat::row(s.bytes().map(|b| b as f64).collect()))
        }
        "sprintf" => {
            let f = ss(a1()?)?;
            some(Val::str(format(&f, &a[1..])?))
        }
        "printf" | "errprintf" => {
            let f = ss(a1()?)?;
            let s = format(&f, &a[1..])?;
            ev.print(&s);
            Ok(None)
        }
        "display" => {
            let v = a1()?;
            ev.flush();
            match v {
                Val::S(m) => {
                    for s in &m.d {
                        ev.print(&format!("{}\n", strip_smcl(s)));
                    }
                }
                Val::R(m) => {
                    for x in &m.d {
                        ev.print(&format!("{}\n", fmt_num(*x)));
                    }
                }
                _ => {}
            }
            Ok(None)
        }
        // ---- control ----
        "exit" => {
            let rc = match arg(0) {
                Some(v) => sc(v)? as i32,
                None => 0,
            };
            ev.flush();
            Err(err(rc, "__exit__"))
        }
        "error" | "_error" => {
            let rc = match arg(0) {
                Some(v) => sc(v)? as i32,
                None => 3498,
            };
            let msg = match arg(1) {
                Some(v) => ss(v)?,
                None => String::new(),
            };
            if name == "error" && msg.is_empty() {
                ev.flush();
                let text = crate::interpreter::err_message(&ferrostat_cmd::eval::rc_err(rc, ""));
                if !text.trim().is_empty() {
                    ev.print(&format!("{text}\n"));
                }
                return Err(err(rc, "__exit__"));
            }
            Err(err(rc, msg))
        }
        "xl" => some(Val::O(std::rc::Rc::new(std::cell::RefCell::new(Obj::Xl(XlBook::default()))))),
        "c" => {
            let n = ss(a1()?)?;
            let e = ferrostat_parser::Expr::Call("c".into(), vec![ferrostat_parser::Expr::Var(n.clone())]);
            let eng = &ev.me.engine;
            match ferrostat_cmd::eval::Env::with_store(&eng.dataset, &eng.store).eval(&e, 0) {
                Ok(ferrostat_cmd::eval::Val::Num(x)) => num(x),
                Ok(ferrostat_cmd::eval::Val::Str(s)) => some(Val::str(s)),
                Err(_) => num(SYSMISS),
            }
        }
        _ => host_function(ev, name, &a),
    }
}

fn strip_smcl(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '{' {
            // keep the text of `{bf:text}`-style directives
            let close = c[i..].iter().position(|&x| x == '}').map(|p| p + i);
            if let Some(cl) = close {
                let inner: String = c[i + 1..cl].iter().collect();
                if let Some((_, t)) = inner.split_once(':') {
                    out.push_str(t);
                }
                i = cl + 1;
                continue;
            }
            depth += 1;
        }
        out.push(c[i]);
        i += 1;
    }
    let _ = depth;
    out
}

fn rank(m: &Mat<f64>) -> usize {
    let mut a = m.clone();
    let (r, c) = (a.r, a.c);
    let scale = a.d.iter().fold(0.0f64, |s, x| s.max(x.abs()));
    let tol = 1e-12 * scale.max(1e-300) * (r.max(c) as f64);
    let mut rank = 0;
    let mut row_i = 0;
    for j in 0..c {
        if row_i >= r {
            break;
        }
        let p = (row_i..r).max_by(|&x, &y| a.at(x, j).abs().total_cmp(&a.at(y, j).abs())).unwrap();
        if a.at(p, j).abs() <= tol {
            continue;
        }
        for k in 0..c {
            a.d.swap(row_i * c + k, p * c + k);
        }
        for i in row_i + 1..r {
            let f = a.at(i, j) / a.at(row_i, j);
            for k in j..c {
                let v = a.at(i, k) - f * a.at(row_i, k);
                a.set(i, k, v);
            }
        }
        row_i += 1;
        rank += 1;
    }
    rank
}
