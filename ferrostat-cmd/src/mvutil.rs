//! Shared helpers for the multivariate commands (pca, factor, cluster, mds,
//! ca, discrim, canon, ...): data loading, moment matrices, eigen
//! decompositions with sign conventions, and boxed table rendering.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Variable};
use ferrostat_parser::Command;

use crate::eval::rc_err;
use crate::format::{fmt, fmt_g};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::summarize::abbrev;
use crate::util;

/// Estimation-sample data: one column per variable over the used rows.
pub struct DataMat {
    pub vars: Vec<String>,
    pub rows: Vec<usize>,
    pub mask: Vec<bool>,
    /// x[j][t]: variable j at used row t
    pub x: Vec<Vec<f64>>,
    /// observation weights (1 when unweighted; aweights rescaled to sum to N)
    pub w: Vec<f64>,
    /// number of observations (sum of frequency weights)
    pub n: f64,
    pub wtype: String,
    pub wexp: String,
}

impl DataMat {
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
}

/// Load `vars` over the `if`/`in` sample, dropping rows with missing values.
/// `allowed` lists the accepted weight kinds (as in `util::weights`).
pub fn load(ds: &DataSet, st: &Store, cmd: &Command, vars: &[String], allowed: &str) -> Result<DataMat, String> {
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, vars);
    let wts = util::weights(ds, st, cmd, &mut mask, allowed)?;
    let rows = util::rows(&mask);
    let mut x = Vec::with_capacity(vars.len());
    for v in vars {
        let c = util::col(ds, v)?;
        x.push(rows.iter().map(|&i| c[i]).collect::<Vec<f64>>());
    }
    let (w, n, wtype, wexp) = match &wts {
        Some(wt) => {
            let raw: Vec<f64> = rows.iter().map(|&i| wt.w[i]).collect();
            match wt.kind {
                util::WeightKind::F => {
                    let s: f64 = raw.iter().sum();
                    (raw, s, "fweight".to_string(), wt.expr.clone())
                }
                _ => {
                    let s: f64 = raw.iter().sum();
                    let k = rows.len() as f64;
                    (raw.iter().map(|v| v * k / s).collect(), k, "aweight".to_string(), wt.expr.clone())
                }
            }
        }
        None => (vec![1.0; rows.len()], rows.len() as f64, String::new(), String::new()),
    };
    Ok(DataMat { vars: vars.to_vec(), rows, mask, x, w, n, wtype, wexp })
}

/// Means and covariance matrix (divisor N-1).
pub fn moments(d: &DataMat) -> (Vec<f64>, Matrix) {
    let p = d.x.len();
    let sw: f64 = d.w.iter().sum();
    let means: Vec<f64> = d.x.iter().map(|c| c.iter().zip(&d.w).map(|(a, w)| a * w).sum::<f64>() / sw).collect();
    let mut cov = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..=i {
            let s: f64 = (0..d.nrows()).map(|t| d.w[t] * (d.x[i][t] - means[i]) * (d.x[j][t] - means[j])).sum();
            let v = s / (d.n - 1.0);
            cov[(i, j)] = v;
            cov[(j, i)] = v;
        }
    }
    (means, cov.with_names(d.vars.clone(), d.vars.clone()))
}

pub fn cov_to_corr(c: &Matrix) -> Matrix {
    let p = c.rows;
    let mut r = c.clone();
    for i in 0..p {
        for j in 0..p {
            r[(i, j)] = c[(i, j)] / (c[(i, i)] * c[(j, j)]).sqrt();
        }
    }
    r
}

/// Eigen decomposition of a symmetric matrix, eigenvalues descending, each
/// eigenvector oriented so that its elements have a positive sum.
pub fn sym_eig(m: &Matrix) -> (Vec<f64>, Matrix) {
    let mut s = m.clone();
    for i in 0..s.rows {
        for j in 0..i {
            let a = 0.5 * (m[(i, j)] + m[(j, i)]);
            s[(i, j)] = a;
            s[(j, i)] = a;
        }
    }
    let (vals, mut vecs) = s.sym_eigen();
    orient_columns(&mut vecs);
    (vals, vecs)
}

/// Flip columns so that each has a nonnegative sum.
pub fn orient_columns(m: &mut Matrix) {
    for j in 0..m.cols {
        let s: f64 = (0..m.rows).map(|i| m[(i, j)]).sum();
        if s < -1e-12 {
            for i in 0..m.rows {
                m[(i, j)] = -m[(i, j)];
            }
        }
    }
}

/// Inverse of a symmetric positive definite matrix (generalized inverse when singular).
pub fn sym_inv(m: &Matrix) -> Matrix {
    match m.cholesky().and_then(|l| l.inverse()) {
        Ok(li) => li.transpose().mul(&li).unwrap(),
        Err(_) => m.sym_inverse(1e-13).0,
    }
}

/// Natural log of the determinant of a symmetric positive definite matrix.
pub fn ln_det(m: &Matrix) -> f64 {
    match m.cholesky() {
        Ok(l) => 2.0 * (0..l.rows).map(|i| l[(i, i)].ln()).sum::<f64>(),
        Err(_) => m.det().ln(),
    }
}

pub fn names(prefix: &str, k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("{prefix}{i}")).collect()
}

/// Columns 0..k of a matrix.
pub fn left_cols(m: &Matrix, k: usize) -> Matrix {
    let mut r = Matrix::zeros(m.rows, k);
    for i in 0..m.rows {
        for j in 0..k {
            r[(i, j)] = m[(i, j)];
        }
    }
    r.rownames = m.rownames.clone();
    r.colnames = m.colnames.iter().take(k).cloned().collect();
    r
}

/// Read a square symmetric matrix argument of pcamat/factormat/... with
/// `shape(full|lower|upper)` and `names()`.
pub fn read_sym_matrix(st: &Store, cmd: &Command) -> Result<(Matrix, Vec<String>), String> {
    let name = cmd.main.split_whitespace().next().ok_or_else(|| rc_err(100, "matrix name required"))?;
    let m = st.matrices.get(name).cloned().ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
    let shape = cmd.opt("shape", 2).unwrap_or("full").trim().to_lowercase();
    let given: Option<Vec<String>> = cmd.opt("names", 2).map(|s| s.split_whitespace().map(|w| w.to_string()).collect());
    let full = if shape.starts_with('l') || shape.starts_with('u') {
        let v = m.data.clone();
        let k = (((8 * v.len() + 1) as f64).sqrt() as usize - 1) / 2;
        if k * (k + 1) / 2 != v.len() || (m.rows != 1 && m.cols != 1) {
            return Err(rc_err(503, format!("matrix {name} has the wrong number of elements")));
        }
        let mut f = Matrix::zeros(k, k);
        let mut idx = 0;
        for i in 0..k {
            if shape.starts_with('l') {
                for j in 0..=i {
                    f[(i, j)] = v[idx];
                    f[(j, i)] = v[idx];
                    idx += 1;
                }
            } else {
                for j in i..k {
                    f[(i, j)] = v[idx];
                    f[(j, i)] = v[idx];
                    idx += 1;
                }
            }
        }
        if given.is_none() {
            return Err(rc_err(198, "names() required with shape(lower) or shape(upper)"));
        }
        f
    } else {
        if m.rows != m.cols {
            return Err(rc_err(505, format!("matrix {name} not square")));
        }
        for i in 0..m.rows {
            for j in 0..i {
                if (m[(i, j)] - m[(j, i)]).abs() > 1e-8 * (1.0 + m[(i, j)].abs()) {
                    return Err(rc_err(505, format!("matrix {name} not symmetric")));
                }
            }
        }
        m.clone()
    };
    let nm = match given {
        Some(g) => {
            if g.len() != full.rows {
                return Err(rc_err(198, "names() has the wrong number of names"));
            }
            g
        }
        None => {
            if full.rownames.len() == full.rows && full.rownames != full.colnames {
                return Err(rc_err(198, format!("matrix {name}: row and column names differ; specify names()")));
            }
            if full.rownames.len() == full.rows { full.rownames.clone() } else { names("c", full.rows) }
        }
    };
    if full.has_missing() {
        return Err(rc_err(504, format!("matrix {name} has missing values")));
    }
    Ok((full.with_names(nm.clone(), nm.clone()), nm))
}

/// Optional 1 x k or k x 1 vector option (sds(), means()).
pub fn vec_opt(st: &Store, cmd: &Command, opt: &str, k: usize) -> Result<Option<Vec<f64>>, String> {
    match cmd.opt(opt, opt.len()) {
        Some(n) => {
            let m = st.matrices.get(n.trim()).ok_or_else(|| rc_err(111, format!("matrix {} not found", n.trim())))?;
            if m.data.len() != k {
                return Err(rc_err(503, "conformability error"));
            }
            Ok(Some(m.data.clone()))
        }
        None => Ok(None),
    }
}

/// Make a symmetric matrix positive semidefinite by zeroing negative eigenvalues.
pub fn force_psd(m: &Matrix) -> Matrix {
    let (vals, vecs) = m.sym_eigen();
    let p = m.rows;
    let mut r = Matrix::zeros(p, p);
    for k in 0..p {
        let l = vals[k].max(0.0);
        for i in 0..p {
            for j in 0..p {
                r[(i, j)] += l * vecs[(i, k)] * vecs[(j, k)];
            }
        }
    }
    r.with_names(m.rownames.clone(), m.colnames.clone())
}

/// Format a value with a display format (`%9.4f`, `%8.0g` ...), trimmed.
pub fn f(v: f64, f_: &str) -> String {
    if is_missing(v) {
        return ".".into();
    }
    fmt(v, f_).trim().to_string()
}

pub fn g(v: f64, w: usize) -> String {
    if is_missing(v) {
        return ".".into();
    }
    fmt_g(v, w)
}

/// A boxed table in the style of the multivariate output:
///
/// ```text
///     ---------------------------------------
///         Variable |    Comp1     Comp2 | Unexplained
///     -------------+--------------------+------------
///           lft500 |   0.4011   -0.3170 |      .2053
///     ---------------------------------------
/// ```
///
/// `groups` are column groups separated by `|`; each column has a header
/// and a width; `cells[r][c]` are preformatted strings (right-aligned).
pub struct Boxed {
    pub indent: usize,
    pub stub_title: String,
    pub stub_width: usize,
    pub groups: Vec<Vec<(String, usize)>>,
    pub rows: Vec<(String, Vec<String>)>,
    /// rows after which a separator line is drawn
    pub seps: Vec<usize>,
}

impl Boxed {
    pub fn new(stub_title: &str, groups: Vec<Vec<(String, usize)>>) -> Self {
        Boxed { indent: 4, stub_title: stub_title.into(), stub_width: 12, groups, rows: Vec::new(), seps: Vec::new() }
    }

    pub fn row(&mut self, name: &str, cells: Vec<String>) {
        self.rows.push((name.to_string(), cells));
    }

    fn group_width(g: &[(String, usize)]) -> usize {
        g.iter().map(|c| c.1).sum::<usize>() + 1
    }

    pub fn render(&self) -> Vec<String> {
        let ind = " ".repeat(self.indent);
        let total: usize = self.stub_width + 1 + self.groups.iter().map(|g| Self::group_width(g)).sum::<usize>() + self.groups.len().saturating_sub(1);
        let mut out = Vec::new();
        out.push(format!("{ind}{}", "-".repeat(total)));
        let mut head = format!("{ind}{:>w$} |", self.stub_title, w = self.stub_width);
        let mut mid = format!("{ind}{}+", "-".repeat(self.stub_width + 1));
        for (gi, g) in self.groups.iter().enumerate() {
            for (h, w) in g {
                head.push_str(&format!("{:>w$}", h, w = *w));
            }
            head.push(' ');
            mid.push_str(&"-".repeat(Self::group_width(g)));
            if gi + 1 < self.groups.len() {
                head.push('|');
                mid.push('+');
            }
        }
        out.push(head.trim_end().to_string());
        out.push(mid);
        for (ri, (name, cells)) in self.rows.iter().enumerate() {
            let mut line = format!("{ind}{:>w$} |", abbrev(name, self.stub_width), w = self.stub_width);
            let mut k = 0;
            for (gi, g) in self.groups.iter().enumerate() {
                for (_, w) in g {
                    let c = cells.get(k).cloned().unwrap_or_default();
                    line.push_str(&format!("{:>w$}", c, w = *w));
                    k += 1;
                }
                line.push(' ');
                if gi + 1 < self.groups.len() {
                    line.push('|');
                }
            }
            out.push(line.trim_end().to_string());
            if self.seps.contains(&ri) && ri + 1 < self.rows.len() {
                out.push(format!("{ind}{}+{}", "-".repeat(self.stub_width + 1), "-".repeat(total - self.stub_width - 2)));
            }
        }
        out.push(format!("{ind}{}", "-".repeat(total)));
        out
    }
}

/// Matrix display in the style of `matlist` with a border: a stub column of
/// row names and blocks of columns (wrapped to fit the line width). With
/// `lower`, only the lower triangle is shown.
pub fn matlist(m: &Matrix, stub_title: &str, fmt_: &str, colw: usize, lower: bool, indent: usize) -> Vec<String> {
    let stubw = m.rownames.iter().map(|s| s.chars().count()).max().unwrap_or(8).clamp(8, 12).max(stub_title.len().min(12));
    let per = ((78 - indent - stubw - 3) / (colw + 1)).max(1);
    let mut out = Vec::new();
    let ind = " ".repeat(indent);
    let rn = if m.rownames.len() == m.rows { m.rownames.clone() } else { names("r", m.rows) };
    let cn = if m.colnames.len() == m.cols { m.colnames.clone() } else { names("c", m.cols) };
    let mut c0 = 0;
    while c0 < m.cols {
        let c1 = (c0 + per).min(m.cols);
        if c0 > 0 {
            out.push(String::new());
        }
        let width = stubw + 2 + (c1 - c0) * (colw + 1) + 1;
        out.push(format!("{ind}{}", "-".repeat(width)));
        let mut head = format!("{ind}{:>w$} |", abbrev(stub_title, stubw), w = stubw);
        for j in c0..c1 {
            head.push_str(&format!(" {:>w$}", abbrev(&cn[j], colw), w = colw));
        }
        out.push(head);
        out.push(format!("{ind}{}+{}", "-".repeat(stubw + 1), "-".repeat(width - stubw - 2)));
        for i in 0..m.rows {
            if lower && i < c0 {
                continue;
            }
            let mut line = format!("{ind}{:>w$} |", abbrev(&rn[i], stubw), w = stubw);
            for j in c0..c1 {
                if lower && j > i {
                    break;
                }
                let v = m[(i, j)];
                let s = if is_missing(v) { ".".to_string() } else { fmt(v, fmt_).trim().to_string() };
                line.push_str(&format!(" {:>w$}", s, w = colw));
            }
            out.push(line.trim_end().to_string());
        }
        out.push(format!("{ind}{}", "-".repeat(width)));
        c0 = c1;
    }
    out
}

/// Add (or replace) a numeric variable over `rows`, missing elsewhere.
pub fn put_var(ds: &mut DataSet, st: &Store, name: &str, ty: Option<ferrostat_core::VarType>, n: usize, rows: &[usize], vals: &[f64], label: &str) -> Result<(), String> {
    let t = ty.unwrap_or(st.default_type);
    let mut full = vec![SYSMISS; n];
    for (k, &i) in rows.iter().enumerate() {
        full[i] = vals[k];
    }
    let mut var = Variable::new(name, t);
    var.data = full.iter().map(|&x| crate::data::store_num(t, x)).collect();
    var.label = label.to_string();
    ds.add_variable(var)
}

/// Parse `[type] newvarlist` (or `stub*` with `k` names) of predict-like commands.
pub fn new_var_names(ds: &DataSet, main: &str, k: usize) -> Result<(Option<ferrostat_core::VarType>, Vec<String>), String> {
    let words: Vec<&str> = main.split_whitespace().collect();
    let (ty, rest) = match words.first().and_then(|w| ferrostat_core::VarType::from_type_str(w)) {
        Some(t) => (Some(t), &words[1..]),
        None => (None, &words[..]),
    };
    if rest.is_empty() {
        return Err(rc_err(100, "new variable name required"));
    }
    let names: Vec<String> = if rest.len() == 1 && rest[0].ends_with('*') {
        let s = rest[0].trim_end_matches('*');
        (1..=k).map(|i| format!("{s}{i}")).collect()
    } else {
        let joined = rest.join(" ");
        crate::varlist::new_names(ds, &joined)?
    };
    for n in &names {
        crate::varlist::check_name(n)?;
        if ds.var_index.contains_key(n) {
            return Err(rc_err(110, format!("variable {n} already defined")));
        }
    }
    Ok((ty, names))
}

/// Integer option with a default.
pub fn opt_usize(cmd: &Command, name: &str, min: usize) -> Result<Option<usize>, String> {
    Ok(util::opt_num(cmd, name, min)?.map(|v| v.max(0.0) as usize))
}

/// Header line pair: left text padded to column 49 then `label = value`.
pub fn hdr(left: &str, label: &str, value: &str) -> String {
    if label.is_empty() {
        return left.to_string();
    }
    format!("{:<49}{:<17}= {:>10}", left, label, value)
}

/// `%9.#g`-like rendering with `d` significant digits (no leading zero).
pub fn sig(v: f64, d: usize) -> String {
    if is_missing(v) {
        return ".".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    let e = v.abs().log10().floor() as i32;
    if e >= d as i32 + 2 || e < -5 {
        let s = crate::format::fmt_e(v, d.saturating_sub(1));
        return s;
    }
    let dec = (d as i32 - 1 - e).max(0) as usize;
    let mut s = format!("{:.*}", dec, v);
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if let Some(r) = s.strip_prefix("0.") {
        s = format!(".{r}");
    } else if let Some(r) = s.strip_prefix("-0.") {
        s = format!("-.{r}");
    }
    if s == "-0" { "0".into() } else { s }
}

/// Polar factor U V' of a square matrix (closest orthogonal matrix) and the
/// sum of its singular values.
pub fn polar(x: &Matrix) -> (Matrix, f64) {
    let xtx = x.transpose().mul(x).unwrap();
    let (vals, vecs) = xtx.sym_eigen();
    let k = x.cols;
    let mut isq = Matrix::zeros(k, k);
    let mut sv = 0.0;
    for m in 0..k {
        let l = vals[m].max(1e-300);
        sv += l.sqrt();
        for i in 0..k {
            for j in 0..k {
                isq[(i, j)] += vecs[(i, m)] * vecs[(j, m)] / l.sqrt();
            }
        }
    }
    (x.mul(&isq).unwrap(), sv)
}
