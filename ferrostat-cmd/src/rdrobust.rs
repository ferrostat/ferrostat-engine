//! Regression-discontinuity estimation with local polynomials: `rdrobust`
//! (point estimates, bias-corrected estimates and robust confidence
//! intervals) and `rdbwselect` (MSE- and CER-optimal bandwidths).
//!
//! Methods: Calonico, Cattaneo & Titiunik (2014, Econometrica) for the
//! robust bias-corrected inference; Calonico, Cattaneo & Farrell (2018,
//! 2020) for the coverage-error-optimal bandwidths. On each side of the
//! cutoff a weighted polynomial of order p is fitted within bandwidth h; the
//! leading bias is estimated with a polynomial of order q within bandwidth
//! b, and the variance of the bias-corrected estimator accounts for the
//! estimated bias. Bandwidths come from a three-step plug-in procedure
//! (pilot, bias bandwidth, main bandwidth) whose variance and bias
//! constants are estimated with the nearest-neighbour or heteroskedasticity
//! robust residuals.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

// ---------------------------------------------------------------------------
// Small dense algebra on column-major designs
// ---------------------------------------------------------------------------

/// Columns of a design (each over the same observations).
pub(crate) type Cols = Vec<Vec<f64>>;

/// (x - c)^j for j = 0..=o.
pub(crate) fn powers(x: &[f64], c: f64, o: usize) -> Cols {
    let mut out: Cols = Vec::with_capacity(o + 1);
    out.push(vec![1.0; x.len()]);
    for j in 1..=o {
        let prev = &out[j - 1];
        let next: Vec<f64> = prev.iter().zip(x).map(|(p, xi)| p * (xi - c)).collect();
        out.push(next);
    }
    out
}

/// a' diag(w) b.
pub(crate) fn cross(a: &[Vec<f64>], w: &[f64], b: &[Vec<f64>]) -> Matrix {
    let mut m = Matrix::zeros(a.len(), b.len());
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            m[(i, j)] = ai.iter().zip(bj).zip(w).map(|((x, y), wi)| x * y * wi).sum();
        }
    }
    m
}

/// Moore-Penrose inverse of a symmetric matrix.
pub(crate) fn pinv_sym(a: &Matrix) -> Matrix {
    let n = a.rows;
    let (vals, vecs) = a.sym_eigen();
    let maxv = vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = maxv * n as f64 * f64::EPSILON;
    let mut out = Matrix::zeros(n, n);
    for (k, &l) in vals.iter().enumerate() {
        if l.abs() > tol {
            for i in 0..n {
                for j in 0..n {
                    out[(i, j)] += vecs[(i, k)] * vecs[(j, k)] / l;
                }
            }
        }
    }
    out
}

/// Inverse of a symmetric positive (semi)definite matrix: Cholesky after
/// equilibrating the diagonal, pseudo-inverse when that fails.
pub(crate) fn inv_sym(a: &Matrix) -> Matrix {
    let n = a.rows;
    let d: Vec<f64> = (0..n).map(|i| if a[(i, i)] > 0.0 { 1.0 / a[(i, i)].sqrt() } else { 1.0 }).collect();
    let mut b = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = a[(i, j)] * d[i] * d[j];
        }
    }
    let binv = match b.cholesky() {
        Ok(l) => {
            // inverse via L^{-1}
            let mut li = Matrix::zeros(n, n);
            let mut ok = true;
            for j in 0..n {
                for i in j..n {
                    let mut s = if i == j { 1.0 } else { 0.0 };
                    for k in j..i {
                        s -= l[(i, k)] * li[(k, j)];
                    }
                    let piv = l[(i, i)];
                    if piv.abs() < 1e-150 {
                        ok = false;
                    }
                    li[(i, j)] = s / piv;
                }
            }
            if ok {
                let mut r = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        r[(i, j)] = (i.max(j)..n).map(|k| li[(k, i)] * li[(k, j)]).sum();
                    }
                }
                // reject near-singular factorisations
                let cond_ok = (0..n).all(|i| l[(i, i)] > 1e-7);
                if cond_ok { r } else { pinv_sym(&b) }
            } else {
                pinv_sym(&b)
            }
        }
        Err(_) => pinv_sym(&b),
    };
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = binv[(i, j)] * d[i] * d[j];
        }
    }
    out
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    a.mul(b).expect("conformable")
}

/// Number of distinct values of a sorted vector.
fn n_unique_sorted(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }
    1 + x.windows(2).filter(|w| w[1] != w[0]).count()
}

fn n_unique(x: &[f64]) -> usize {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    n_unique_sorted(&v)
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

// ---------------------------------------------------------------------------
// Kernels and variance types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Kernel {
    Triangular,
    Epanechnikov,
    Uniform,
}

impl Kernel {
    pub(crate) fn parse(s: &str, default: Kernel) -> Result<Kernel, String> {
        Ok(match s.trim().to_lowercase().as_str() {
            "" => default,
            "tri" | "triangular" => Kernel::Triangular,
            "epa" | "epanechnikov" => Kernel::Epanechnikov,
            "uni" | "uniform" => Kernel::Uniform,
            _ => return Err(rc_err(7, "kernel() incorrectly specified")),
        })
    }
    pub(crate) fn title(self) -> &'static str {
        match self {
            Kernel::Triangular => "Triangular",
            Kernel::Epanechnikov => "Epanechnikov",
            Kernel::Uniform => "Uniform",
        }
    }
    /// Constant of the rule-of-thumb pilot bandwidth.
    fn pilot_const(self) -> f64 {
        match self {
            Kernel::Triangular => 2.576,
            Kernel::Epanechnikov => 2.34,
            Kernel::Uniform => 1.843,
        }
    }
}

/// Kernel weights K((x - c)/h)/h.
pub(crate) fn kweight(x: &[f64], c: f64, h: f64, k: Kernel) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let u = (xi - c) / h;
            if u.abs() > 1.0 {
                return 0.0;
            }
            match k {
                Kernel::Epanechnikov => 0.75 * (1.0 - u * u) / h,
                Kernel::Uniform => 0.5 / h,
                Kernel::Triangular => (1.0 - u.abs()) / h,
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ResKind {
    /// nearest-neighbour residuals with this many matches
    Nn(usize),
    /// heteroskedasticity-robust residuals hc0..hc3
    Hc(u8),
}

#[derive(Clone, Debug)]
struct VceSpec {
    res: ResKind,
    cluster: Option<String>,
    /// e(vce_select) as given
    raw: String,
    /// display label
    label: String,
}

fn parse_vce(text: Option<&str>) -> Result<VceSpec, String> {
    let words: Vec<String> = text.unwrap_or("").split_whitespace().map(str::to_string).collect();
    let bad = || rc_err(125, "vce() incorrectly specified");
    if words.len() > 3 {
        return Err(bad());
    }
    let kind = words.first().map(|w| w.to_lowercase()).unwrap_or_default();
    let mut nn = 3usize;
    let mut cluster = None;
    match words.len() {
        2 => {
            if kind == "nn" {
                nn = words[1].parse().map_err(|_| rc_err(125, "vce() incorrectly specified: nnmatch should be a positive integer"))?;
            } else if ["cluster", "nncluster", "cr1", "cr2", "cr3", "hc0", "hc1", "hc2", "hc3"].contains(&kind.as_str()) {
                cluster = Some(words[1].clone());
            } else {
                return Err(bad());
            }
        }
        3 => {
            if !["cluster", "nncluster", "cr1", "cr2", "cr3"].contains(&kind.as_str()) {
                return Err(bad());
            }
            cluster = Some(words[1].clone());
            nn = words[2].parse().map_err(|_| rc_err(125, "vce() incorrectly specified: nnmatch should be a positive integer"))?;
        }
        _ => {}
    }
    if nn == 0 {
        return Err(rc_err(411, "p(), q(), deriv(), nnmatch() should be positive"));
    }
    let raw = if kind.is_empty() { "nn".to_string() } else { kind.clone() };
    let mut sel = kind.clone();
    if sel == "nncluster" {
        sel = "cr1".into();
    }
    if cluster.is_some() {
        sel = match sel.as_str() {
            "hc0" | "hc1" | "cluster" => "cr1".into(),
            "hc2" => "cr2".into(),
            "hc3" => "cr3".into(),
            s => s.to_string(),
        };
    }
    if sel == "cr2" || sel == "cr3" {
        return Err(rc_err(198, format!("vce({sel}) is not available; use vce(cr1 clustvar)")));
    }
    let (res, label) = match sel.as_str() {
        "" | "nn" => (ResKind::Nn(nn), "NN"),
        "hc0" => (ResKind::Hc(0), "HC0"),
        "hc1" => (ResKind::Hc(1), "HC1"),
        "hc2" => (ResKind::Hc(2), "HC2"),
        "hc3" => (ResKind::Hc(3), "HC3"),
        "cr1" => {
            if cluster.is_none() {
                return Err(rc_err(125, "vce(cr1 clustervar) requires a cluster variable"));
            }
            (ResKind::Hc(0), "CR1")
        }
        _ => return Err(rc_err(7, "vce() incorrectly specified")),
    };
    Ok(VceSpec { res, cluster, raw, label: label.into() })
}

// ---------------------------------------------------------------------------
// One side of the cutoff
// ---------------------------------------------------------------------------

/// Observations on one side of the cutoff, sorted by the running variable.
#[derive(Clone, Debug, Default)]
struct Side {
    x: Vec<f64>,
    y: Vec<f64>,
    t: Option<Vec<f64>>,
    z: Cols,
    cl: Option<Vec<f64>>,
    fw: Option<Vec<f64>>,
    /// size of the tie group of x and position within it (1-based)
    dups: Vec<usize>,
    dupsid: Vec<usize>,
}

impl Side {
    fn sub(&self, idx: &[usize]) -> Side {
        let pick = |v: &[f64]| -> Vec<f64> { idx.iter().map(|&i| v[i]).collect() };
        Side {
            x: pick(&self.x),
            y: pick(&self.y),
            t: self.t.as_ref().map(|t| pick(t)),
            z: self.z.iter().map(|c| pick(c)).collect(),
            cl: self.cl.as_ref().map(|c| pick(c)),
            fw: self.fw.as_ref().map(|c| pick(c)),
            dups: idx.iter().map(|&i| self.dups[i]).collect(),
            dupsid: idx.iter().map(|&i| self.dupsid[i]).collect(),
        }
    }

    /// Outcome columns: y, then the treatment, then covariates.
    fn d_cols(&self) -> Vec<&[f64]> {
        let mut d: Vec<&[f64]> = vec![&self.y];
        if let Some(t) = &self.t {
            d.push(t);
        }
        for z in &self.z {
            d.push(z);
        }
        d
    }

    /// Kernel weights times frequency weights.
    fn weights(&self, c: f64, h: f64, k: Kernel) -> Vec<f64> {
        let mut w = kweight(&self.x, c, h, k);
        if let Some(fw) = &self.fw {
            for (a, b) in w.iter_mut().zip(fw) {
                *a *= b;
            }
        }
        w
    }
}

fn positive(w: &[f64]) -> Vec<usize> {
    w.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, _)| i).collect()
}

/// Residuals used in the variance estimators, one column per outcome.
#[allow(clippy::too_many_arguments)]
fn residuals(kind: ResKind, s: &Side, d: &[&[f64]], r: &Cols, w: &[f64], inv_g: &Matrix, beta: &Matrix, nparam: usize) -> Cols {
    let n = s.x.len();
    match kind {
        ResKind::Nn(matches) => {
            let mut out: Cols = vec![vec![0.0; n]; d.len()];
            let lim = matches.min(n.saturating_sub(1)) as isize;
            let ni = n as isize;
            for pos in 0..n {
                let p = pos as isize;
                let mut rpos = (s.dups[pos] - s.dupsid[pos]) as isize;
                let mut lpos = s.dupsid[pos] as isize - 1;
                while lpos + rpos < lim {
                    if p - lpos - 1 < 0 {
                        rpos += s.dups[(p + rpos + 1) as usize] as isize;
                    } else if p + rpos + 1 >= ni {
                        lpos += s.dups[(p - lpos - 1) as usize] as isize;
                    } else {
                        let dl = s.x[pos] - s.x[(p - lpos - 1) as usize];
                        let dr = s.x[(p + rpos + 1) as usize] - s.x[pos];
                        let tol = dl.max(dr) * f64::EPSILON.sqrt();
                        if dl - dr > tol {
                            rpos += s.dups[(p + rpos + 1) as usize] as isize;
                        } else if dr - dl > tol {
                            lpos += s.dups[(p - lpos - 1) as usize] as isize;
                        } else {
                            rpos += s.dups[(p + rpos + 1) as usize] as isize;
                            lpos += s.dups[(p - lpos - 1) as usize] as isize;
                        }
                    }
                }
                let lo = (p - lpos) as usize;
                let hi = (p + rpos) as usize;
                let ji = (hi - lo) as f64;
                let f = (ji / (ji + 1.0)).sqrt();
                for (l, col) in d.iter().enumerate() {
                    let sum: f64 = col[lo..=hi].iter().sum::<f64>() - col[pos];
                    out[l][pos] = f * (col[pos] - sum / ji);
                }
            }
            out
        }
        ResKind::Hc(k) => {
            let kk = r.len();
            let hii: Vec<f64> = if k >= 2 {
                (0..n)
                    .map(|i| {
                        let mut s2 = 0.0;
                        for a in 0..kk {
                            for b in 0..kk {
                                s2 += r[a][i] * inv_g[(a, b)] * r[b][i];
                            }
                        }
                        s2 * w[i]
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let nn = n as f64;
            (0..d.len())
                .map(|l| {
                    (0..n)
                        .map(|i| {
                            let pred: f64 = (0..kk).map(|a| r[a][i] * beta[(a, l)]).sum();
                            let f = match k {
                                0 => 1.0,
                                1 => (nn / (nn - nparam as f64)).sqrt(),
                                2 => (1.0 / (1.0 - hii[i])).sqrt(),
                                _ => 1.0 / (1.0 - hii[i]),
                            };
                            f * (d[l][i] - pred)
                        })
                        .collect()
                })
                .collect()
        }
    }
}

/// Middle of the sandwich: sum over observations (or clusters) of the
/// scores rx_i * (res_i . s).
fn meat(rx: &Cols, res: &Cols, s: &[f64], cl: Option<&[f64]>, k_override: usize) -> Matrix {
    let k = rx.len();
    let n = rx.first().map_or(0, |c| c.len());
    let comb: Vec<f64> = (0..n).map(|i| res.iter().zip(s).map(|(r, sl)| r[i] * sl).sum()).collect();
    let mut m = Matrix::zeros(k, k);
    match cl {
        None => {
            for i in 0..n {
                let e2 = comb[i] * comb[i];
                for a in 0..k {
                    for b in 0..k {
                        m[(a, b)] += rx[a][i] * rx[b][i] * e2;
                    }
                }
            }
        }
        Some(c) => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
            let mut g = 0usize;
            let mut i = 0;
            while i < n {
                let mut j = i;
                let mut score = vec![0.0; k];
                while j < n && c[order[j]] == c[order[i]] {
                    let o = order[j];
                    for a in 0..k {
                        score[a] += rx[a][o] * comb[o];
                    }
                    j += 1;
                }
                g += 1;
                for a in 0..k {
                    for b in 0..k {
                        m[(a, b)] += score[a] * score[b];
                    }
                }
                i = j;
            }
            let kk = if k_override > 0 { k_override } else { k } as f64;
            let nf = n as f64;
            let gf = g as f64;
            let w = ((nf - 1.0) / (nf - kk)) * (gf / (gf - 1.0));
            m = m.scale(w);
        }
    }
    m
}

fn scaled(r: &Cols, w: &[f64]) -> Cols {
    r.iter().map(|c| c.iter().zip(w).map(|(a, b)| a * b).collect()).collect()
}

/// beta = inv_g * R' W D, one column per outcome.
fn coefs(r: &Cols, w: &[f64], d: &[&[f64]], inv_g: &Matrix) -> Matrix {
    let k = r.len();
    let mut rwd = Matrix::zeros(k, d.len());
    for a in 0..k {
        for (l, col) in d.iter().enumerate() {
            rwd[(a, l)] = r[a].iter().zip(col.iter()).zip(w).map(|((x, y), wi)| x * y * wi).sum();
        }
    }
    matmul(inv_g, &rwd)
}

// ---------------------------------------------------------------------------
// Bandwidth constants
// ---------------------------------------------------------------------------

/// Variance, bias and regularisation constants and rate of one bandwidth
/// step (None when a window has too few distinct values).
#[allow(clippy::too_many_arguments)]
fn bw_consts(side: &Side, c: f64, o: usize, nu: usize, o_b: usize, h_v: f64, h_b: f64, scale: f64, vce: &VceSpec, k: Kernel) -> Option<(f64, f64, f64, f64)> {
    let rate = 1.0 / (2.0 * o as f64 + 3.0);
    // variance part at the pilot bandwidth
    let w = side.weights(c, h_v, k);
    let ind = positive(&w);
    let sv = side.sub(&ind);
    let ew: Vec<f64> = ind.iter().map(|&i| w[i]).collect();
    if n_unique_sorted(&sv.x) < o + 1 {
        return None;
    }
    let r = powers(&sv.x, c, o);
    let inv_g = inv_sym(&cross(&r, &ew, &r));
    let d = sv.d_cols();
    let dt = sv.t.is_some() as usize;
    let dz = sv.z.len();
    let beta = coefs(&r, &ew, &d, &inv_g);
    let mut s = vec![1.0];
    let mut gamma: Option<Matrix> = None;
    if dz > 0 {
        let g = covariate_gamma(&[(&r, &ew, &sv, &inv_g)], dt);
        s = std::iter::once(1.0).chain((0..dz).map(|j| -g[(j, 0)])).collect();
        gamma = Some(g);
    }
    let nf = factorial(nu);
    if dt == 1 {
        match &gamma {
            None => {
                let ty = nf * beta[(nu, 0)];
                let tt = nf * beta[(nu, 1)];
                s = vec![1.0 / tt, -(ty / (tt * tt))];
            }
            Some(g) => {
                let ty = nf * (beta[(nu, 0)] - (0..dz).map(|j| g[(j, 0)] * beta[(nu, 2 + j)]).sum::<f64>());
                let tt = nf * (beta[(nu, 1)] - (0..dz).map(|j| g[(j, 1)] * beta[(nu, 2 + j)]).sum::<f64>());
                s = vec![1.0 / tt, -(ty / (tt * tt))];
                for j in 0..dz {
                    s.push(-(1.0 / tt) * g[(j, 0)] + (ty / (tt * tt)) * g[(j, 1)]);
                }
            }
        }
    }
    let res = residuals(vce.res, &sv, &d, &r, &ew, &inv_g, &beta, o + 1);
    let rx = scaled(&r, &ew);
    let m = meat(&rx, &res, &s, sv.cl.as_deref(), 0);
    let vv = matmul(&matmul(&inv_g, &m), &inv_g)[(nu, nu)];
    let u_o: Vec<f64> = sv.x.iter().map(|xi| ((xi - c) / h_v).powi(o as i32 + 1)).collect();
    let v: Vec<f64> = (0..=o).map(|a| rx[a].iter().zip(&u_o).map(|(p, q)| p * q).sum()).collect();
    let igv: f64 = (0..=o).map(|b| inv_g[(nu, b)] * v[b]).sum();
    let bconst = h_v.powi(nu as i32) * igv;
    // bias part at bandwidth h_b
    if !h_b.is_finite() || h_b <= 0.0 || is_missing(h_b) {
        return None;
    }
    let w = side.weights(c, h_b, k);
    let ind = positive(&w);
    let sb = side.sub(&ind);
    let ew: Vec<f64> = ind.iter().map(|&i| w[i]).collect();
    if n_unique_sorted(&sb.x) < o_b + 1 {
        return None;
    }
    let r = powers(&sb.x, c, o_b);
    let inv_g = inv_sym(&cross(&r, &ew, &r));
    let d = sb.d_cols();
    let beta_b = coefs(&r, &ew, &d, &inv_g);
    let mut bw_reg = 0.0;
    if scale > 0.0 {
        let res = residuals(vce.res, &sb, &d, &r, &ew, &inv_g, &beta_b, o_b + 1);
        let rx = scaled(&r, &ew);
        let m = meat(&rx, &res, &s, sb.cl.as_deref(), 0);
        let vb = matmul(&matmul(&inv_g, &m), &inv_g)[(o + 1, o + 1)];
        bw_reg = 3.0 * bconst * bconst * vb;
    }
    let sb_dot: f64 = s.iter().enumerate().map(|(l, sl)| sl * beta_b[(o + 1, l)]).sum();
    let b = (2.0 * (o + 1 - nu) as f64).sqrt() * bconst * sb_dot;
    let vfin = (2.0 * nu as f64 + 1.0) * h_v.powi(2 * nu as i32 + 1) * vv;
    let rr = scale * 2.0 * (o + 1 - nu) as f64 * bw_reg;
    if !vfin.is_finite() || !b.is_finite() || !rr.is_finite() {
        return None;
    }
    Some((vfin, b, rr, rate))
}

/// Covariate adjustment coefficients: (Z'WZ - U'G^{-1}U)^{-1}(Z'WY - ...)
/// summed over the supplied sides. Columns: one per non-covariate outcome.
fn covariate_gamma(parts: &[(&Cols, &[f64], &Side, &Matrix)], dt: usize) -> Matrix {
    let dz = parts[0].2.z.len();
    let ny = 1 + dt;
    let mut zwz = Matrix::zeros(dz, dz);
    let mut zwy = Matrix::zeros(dz, ny);
    for (r, w, s, inv_g) in parts {
        let d = s.d_cols();
        let dd: Cols = d.iter().map(|c| c.to_vec()).collect();
        let u = cross(r, w, &dd); // k x (ny+dz)
        let gu = matmul(inv_g, &u);
        let zc: Cols = s.z.clone();
        let zwd = cross(&zc, w, &dd);
        for a in 0..dz {
            for b in 0..(ny + dz) {
                let uigu: f64 = (0..r.len()).map(|kk| u[(kk, ny + a)] * gu[(kk, b)]).sum();
                let v = zwd[(a, b)] - uigu;
                if b < ny {
                    zwy[(a, b)] += v;
                } else {
                    zwz[(a, b - ny)] += v;
                }
            }
        }
    }
    matmul(&pinv_sym(&zwz), &zwy)
}

// ---------------------------------------------------------------------------
// Data preparation
// ---------------------------------------------------------------------------

/// Sample of an RD command, sorted by the running variable.
struct RdData {
    yname: String,
    xname: String,
    fuzzy: Option<String>,
    sharpbw: bool,
    covs: Vec<String>,
    clustvar: Option<String>,
    left: Side,
    right: Side,
    /// original observation numbers of the sample (any order)
    rows: Vec<usize>,
    c: f64,
    n: usize,
    x_min: f64,
    x_max: f64,
    x_sd: f64,
    x_iq: f64,
    y_sd: f64,
    covs_dropped: bool,
}

fn opt_word(cmd: &Command, name: &str, min: usize) -> Option<String> {
    cmd.opt(name, min).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Drop covariates that are linear combinations of earlier ones.
fn noncollinear(cols: &[Vec<f64>]) -> Vec<usize> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut keep = Vec::new();
    let n = cols.first().map_or(0, |c| c.len()) as f64;
    let ones: Vec<f64> = vec![1.0 / n.sqrt(); n as usize];
    basis.push(ones);
    for (j, c) in cols.iter().enumerate() {
        let mut v = c.clone();
        let norm0: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for b in &basis {
            let p: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(b) {
                *x -= p * y;
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-9 * norm0.max(1e-300) {
            for x in v.iter_mut() {
                *x /= norm;
            }
            basis.push(v);
            keep.push(j);
        }
    }
    keep
}

fn prepare(ds: &DataSet, st: &Store, cmd: &Command, need_y: bool, vce: &VceSpec) -> Result<RdData, String> {
    let vars = crate::varlist::expand(ds, &cmd.main)?;
    let need = if need_y { 2 } else { 1 };
    if vars.len() < need {
        return Err(rc_err(102, "too few variables specified"));
    }
    if vars.len() > need {
        return Err(rc_err(103, "too many variables specified"));
    }
    let (yname, xname) = (vars[0].clone(), vars[need - 1].clone());
    let c = util::opt_num(cmd, "c", 1)?.unwrap_or(0.0);
    let mut fuzzy = None;
    let mut sharpbw = false;
    if let Some(f) = opt_word(cmd, "fuzzy", 5) {
        let w: Vec<&str> = f.split_whitespace().collect();
        if w.len() > 2 {
            return Err(rc_err(125, "fuzzy() only accepts two inputs"));
        }
        if w.len() == 2 {
            if w[1] != "sharpbw" {
                return Err(rc_err(125, "fuzzy() only accepts sharpbw as a second input"));
            }
            sharpbw = true;
        }
        fuzzy = Some(crate::varlist::resolve_name(ds, w[0])?);
    }
    let mut covs = match opt_word(cmd, "covs", 4) {
        Some(t) => crate::varlist::expand(ds, &t)?,
        None => Vec::new(),
    };
    let clustvar = match &vce.cluster {
        Some(v) => Some(crate::varlist::resolve_name(ds, v)?),
        None => None,
    };
    let wname = match opt_word(cmd, "weights", 7) {
        Some(w) => Some(crate::varlist::resolve_name(ds, &w)?),
        None => None,
    };
    let mut mask = util::touse(ds, st, cmd)?;
    let mut mv = vec![yname.clone(), xname.clone()];
    mv.extend(fuzzy.iter().cloned());
    mv.extend(clustvar.iter().cloned());
    mv.extend(covs.iter().cloned());
    mv.extend(wname.iter().cloned());
    util::markout(ds, &mut mask, &mv);
    let num = |name: &str| -> Result<Vec<f64>, String> { util::col(ds, name) };
    let yv = num(&yname)?;
    let xv = num(&xname)?;
    let wv = match &wname {
        Some(w) => Some(num(w)?),
        None => None,
    };
    if let Some(w) = &wv {
        for i in 0..ds.n_obs {
            if mask[i] && w[i] <= 0.0 {
                mask[i] = false;
            }
        }
    }
    let mut rows = util::rows(&mask);
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    rows.sort_by(|&a, &b| xv[a].total_cmp(&xv[b]));
    let tv = match &fuzzy {
        Some(f) => Some(num(f)?),
        None => None,
    };
    let clv = match &clustvar {
        Some(cv) => {
            let var = ds.get_var(cv).unwrap();
            if var.var_type.is_string() {
                let keys: Vec<&str> = (0..ds.n_obs).map(|i| var.str_at(i)).collect();
                let (codes, _) = crate::hdfe::codes_str(&keys);
                Some(codes.into_iter().map(|c| c as f64).collect::<Vec<f64>>())
            } else {
                Some(num(cv)?)
            }
        }
        None => None,
    };
    let mut zcols: Cols = covs.iter().map(|z| num(z).map(|v| rows.iter().map(|&i| v[i]).collect())).collect::<Result<_, _>>()?;
    let mut covs_dropped = false;
    if !zcols.is_empty() {
        let keep = noncollinear(&zcols);
        if keep.len() < zcols.len() {
            covs_dropped = true;
            covs = keep.iter().map(|&j| covs[j].clone()).collect();
            zcols = keep.iter().map(|&j| zcols[j].clone()).collect();
        }
    }
    let n = rows.len();
    let xs: Vec<f64> = rows.iter().map(|&i| xv[i]).collect();
    let ys: Vec<f64> = rows.iter().map(|&i| yv[i]).collect();
    // tie groups
    let mut dups = vec![0usize; n];
    let mut dupsid = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && xs[j] == xs[i] {
            j += 1;
        }
        for k in i..j {
            dups[k] = j - i;
            dupsid[k] = k - i + 1;
        }
        i = j;
    }
    let split = xs.iter().position(|&x| x >= c).unwrap_or(n);
    let mk = |lo: usize, hi: usize| -> Side {
        Side {
            x: xs[lo..hi].to_vec(),
            y: ys[lo..hi].to_vec(),
            t: tv.as_ref().map(|t| rows[lo..hi].iter().map(|&i| t[i]).collect()),
            z: zcols.iter().map(|z| z[lo..hi].to_vec()).collect(),
            cl: clv.as_ref().map(|t| rows[lo..hi].iter().map(|&i| t[i]).collect()),
            fw: wv.as_ref().map(|t| rows[lo..hi].iter().map(|&i| t[i]).collect()),
            dups: dups[lo..hi].to_vec(),
            dupsid: dupsid[lo..hi].to_vec(),
        }
    };
    let left = mk(0, split);
    let right = mk(split, n);
    let x_sd = util::variance(&xs).sqrt();
    let y_sd = util::variance(&ys).sqrt();
    let x_iq = util::pctile_sorted(&xs, 75.0) - util::pctile_sorted(&xs, 25.0);
    Ok(RdData {
        yname,
        xname,
        fuzzy,
        sharpbw,
        covs,
        clustvar,
        left,
        right,
        rows,
        c,
        n,
        x_min: xs[0],
        x_max: xs[n - 1],
        x_sd,
        x_iq,
        y_sd,
        covs_dropped,
    })
}

// ---------------------------------------------------------------------------
// Bandwidth selection
// ---------------------------------------------------------------------------

const SELECTORS: [&str; 10] = ["mserd", "msetwo", "msesum", "msecomb1", "msecomb2", "cerrd", "certwo", "cersum", "cercomb1", "cercomb2"];

#[derive(Clone, Debug)]
struct BwOpts {
    p: usize,
    q: usize,
    deriv: usize,
    kernel: Kernel,
    scaleregul: f64,
    masspoints: String,
    bwcheck: usize,
    bwrestrict: bool,
    stdvars: bool,
}

/// Selected bandwidths, (h_l, h_r, b_l, b_r) per selector.
#[derive(Default, Debug)]
struct BwOut {
    sel: Vec<(&'static str, [f64; 4])>,
    masspoints_found: bool,
    m_l: usize,
    m_r: usize,
    g_l: usize,
    g_r: usize,
}

impl BwOut {
    fn get(&self, name: &str) -> Option<[f64; 4]> {
        self.sel.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }
}

fn n_clusters(s: &Side) -> usize {
    s.cl.as_ref().map_or(0, |c| n_unique(c))
}

fn select_bandwidths(d: &RdData, o: &BwOpts, vce: &VceSpec, wanted: &[&str], announce: &mut Vec<String>) -> Result<BwOut, String> {
    let eps = f64::EPSILON.sqrt();
    let (x_sd, y_sd, mut c, bwp) = if o.stdvars {
        (d.x_sd, d.y_sd, d.c / d.x_sd, (1.0f64).min((d.x_iq / d.x_sd) / 1.349))
    } else {
        (1.0, 1.0, d.c, d.x_sd.min(d.x_iq / 1.349))
    };
    let stdz = |s: &Side| -> Side {
        let mut s = s.clone();
        if o.stdvars {
            s.x.iter_mut().for_each(|v| *v /= x_sd);
            s.y.iter_mut().for_each(|v| *v /= y_sd);
        }
        // sharp bandwidths for fuzzy designs with perfect compliance on a side
        s
    };
    let mut l = stdz(&d.left);
    let mut r = stdz(&d.right);
    if d.fuzzy.is_some() {
        let tl = l.t.clone().unwrap_or_default();
        let tr = r.t.clone().unwrap_or_default();
        let perf = util::variance(&tl) == 0.0 || util::variance(&tr) == 0.0 || tl.len() < 2 || tr.len() < 2;
        if perf || d.sharpbw {
            l.t = None;
            r.t = None;
            if d.sharpbw {
                announce.push("WARNING: bandwidths automatically computed for sharp RD estimation.".into());
            } else {
                announce.push("WARNING: bandwidths automatically computed for sharp RD estimation because perfect compliance was detected on at least one side of the threshold.".into());
            }
        }
    }
    if !o.stdvars {
        c = d.c;
    }
    let range_l = c - l.x[0];
    let range_r = r.x[r.x.len() - 1] - c;
    let m_l = n_unique_sorted(&l.x);
    let m_r = n_unique_sorted(&r.x);
    let mut out = BwOut { m_l, m_r, g_l: n_clusters(&d.left), g_r: n_clusters(&d.right), ..Default::default() };
    let mut bwcheck = o.bwcheck;
    if o.masspoints == "check" || o.masspoints == "adjust" {
        let mass_l = 1.0 - m_l as f64 / l.x.len() as f64;
        let mass_r = 1.0 - m_r as f64 / r.x.len() as f64;
        if mass_l >= 0.2 || mass_r >= 0.2 {
            out.masspoints_found = true;
            announce.push("Mass points detected in the running variable.".into());
            if o.masspoints == "adjust" && bwcheck == 0 {
                bwcheck = 10;
            }
            if o.masspoints == "check" {
                announce.push("Try using option masspoints(adjust).".into());
            }
        }
    }
    let n = d.n as f64;
    let mut c_bw = o.kernel.pilot_const() * bwp * n.powf(-0.2);
    if o.masspoints == "adjust" {
        c_bw = o.kernel.pilot_const() * bwp * ((m_l + m_r) as f64).powf(-0.2);
    }
    let bw_max = range_l.max(range_r);
    if o.bwrestrict {
        c_bw = c_bw.min(bw_max);
    }
    let (mut bw_min_l, mut bw_min_r) = (0.0, 0.0);
    if bwcheck > 0 {
        let mut ul: Vec<f64> = l.x.iter().map(|v| (v - c).abs()).collect();
        ul.dedup();
        ul.sort_by(|a, b| a.total_cmp(b));
        ul.dedup();
        let mut ur: Vec<f64> = r.x.iter().map(|v| (v - c).abs()).collect();
        ur.dedup();
        ur.sort_by(|a, b| a.total_cmp(b));
        ur.dedup();
        bw_min_l = ul[bwcheck.min(ul.len()) - 1] * (1.0 + eps);
        bw_min_r = ur[bwcheck.min(ur.len()) - 1] * (1.0 + eps);
        c_bw = c_bw.max(bw_min_l).max(bw_min_r);
    }
    let (p, q, deriv, sr, k) = (o.p, o.q, o.deriv, o.scaleregul, o.kernel);
    let mut undef = false;
    let mut call = |s: &Side, oo: usize, nu: usize, ob: usize, hb: f64, scale: f64| -> (f64, f64, f64, f64) {
        match bw_consts(s, c, oo, nu, ob, c_bw, hb, scale, vce, k) {
            Some(v) => v,
            None => {
                undef = true;
                (f64::NAN, f64::NAN, f64::NAN, 1.0 / (2.0 * oo as f64 + 3.0))
            }
        }
    };
    let cd_l = call(&l, q + 1, q + 1, q + 2, range_l * (1.0 + eps), 0.0);
    let cd_r = call(&r, q + 1, q + 1, q + 2, range_r * (1.0 + eps), 0.0);
    let want = |names: &[&str]| names.iter().any(|n| wanted.contains(n));
    let mut two: Option<(f64, f64, f64, f64)> = None;
    if want(&["msetwo", "certwo", "msecomb2", "cercomb2"]) {
        let mut d_l = (cd_l.0 / (cd_l.1 * cd_l.1)).powf(cd_l.3);
        let mut d_r = (cd_r.0 / (cd_r.1 * cd_r.1)).powf(cd_r.3);
        if o.bwrestrict {
            d_l = d_l.min(range_l);
            d_r = d_r.min(range_r);
        }
        if bwcheck > 0 {
            d_l = d_l.max(bw_min_l);
            d_r = d_r.max(bw_min_r);
        }
        let cb_l = call(&l, q, p + 1, q + 1, d_l, sr);
        let mut b_l = (cb_l.0 / (cb_l.1 * cb_l.1 + sr * cb_l.2)).powf(cb_l.3);
        let cb_r = call(&r, q, p + 1, q + 1, d_r, sr);
        let mut b_r = (cb_r.0 / (cb_r.1 * cb_r.1 + sr * cb_r.2)).powf(cb_r.3);
        if o.bwrestrict {
            b_l = b_l.min(range_l);
            b_r = b_r.min(range_r);
        }
        let ch_l = call(&l, p, deriv, q, b_l, sr);
        let mut h_l = (ch_l.0 / (ch_l.1 * ch_l.1 + sr * ch_l.2)).powf(ch_l.3);
        let ch_r = call(&r, p, deriv, q, b_r, sr);
        let mut h_r = (ch_r.0 / (ch_r.1 * ch_r.1 + sr * ch_r.2)).powf(ch_r.3);
        if o.bwrestrict {
            h_l = h_l.min(range_l);
            h_r = h_r.min(range_r);
        }
        two = Some((h_l, h_r, b_l, b_r));
    }
    let mut sum: Option<(f64, f64)> = None;
    if want(&["msesum", "cersum", "msecomb1", "msecomb2", "cercomb1", "cercomb2"]) {
        let mut d_s = ((cd_l.0 + cd_r.0) / (cd_r.1 + cd_l.1).powi(2)).powf(cd_l.3);
        if o.bwrestrict {
            d_s = d_s.min(bw_max);
        }
        if bwcheck > 0 {
            d_s = d_s.max(bw_min_l).max(bw_min_r);
        }
        let cb_l = call(&l, q, p + 1, q + 1, d_s, sr);
        let cb_r = call(&r, q, p + 1, q + 1, d_s, sr);
        let mut b_s = ((cb_l.0 + cb_r.0) / ((cb_r.1 + cb_l.1).powi(2) + sr * (cb_r.2 + cb_l.2))).powf(cb_l.3);
        if o.bwrestrict {
            b_s = b_s.min(bw_max);
        }
        let ch_l = call(&l, p, deriv, q, b_s, sr);
        let ch_r = call(&r, p, deriv, q, b_s, sr);
        let mut h_s = ((ch_l.0 + ch_r.0) / ((ch_r.1 + ch_l.1).powi(2) + sr * (ch_r.2 + ch_l.2))).powf(ch_l.3);
        if o.bwrestrict {
            h_s = h_s.min(bw_max);
        }
        sum = Some((h_s, b_s));
    }
    let mut rd: Option<(f64, f64)> = None;
    if want(&["mserd", "cerrd", "msecomb1", "msecomb2", "cercomb1", "cercomb2"]) {
        let mut d_d = ((cd_l.0 + cd_r.0) / (cd_r.1 - cd_l.1).powi(2)).powf(cd_l.3);
        if o.bwrestrict {
            d_d = d_d.min(bw_max);
        }
        if bwcheck > 0 {
            d_d = d_d.max(bw_min_l).max(bw_min_r);
        }
        let cb_l = call(&l, q, p + 1, q + 1, d_d, sr);
        let cb_r = call(&r, q, p + 1, q + 1, d_d, sr);
        let mut b_d = ((cb_l.0 + cb_r.0) / ((cb_r.1 - cb_l.1).powi(2) + sr * (cb_r.2 + cb_l.2))).powf(cb_l.3);
        if o.bwrestrict {
            b_d = b_d.min(bw_max);
        }
        let ch_l = call(&l, p, deriv, q, b_d, sr);
        let ch_r = call(&r, p, deriv, q, b_d, sr);
        let mut h_d = ((ch_l.0 + ch_r.0) / ((ch_r.1 - ch_l.1).powi(2) + sr * (ch_r.2 + ch_l.2))).powf(ch_l.3);
        if o.bwrestrict {
            h_d = h_d.min(bw_max);
        }
        rd = Some((h_d, b_d));
    }
    if undef {
        return Err(rc_err(2001, "Not enough variability in the running variable to compute the bandwidth. Check for mass points with option masspoints(check); if the running variable is discrete, an RD design may not be identified at this sample size."));
    }
    let cer_n = if d.clustvar.is_some() { (out.g_l + out.g_r) as f64 } else { n };
    let pf = p as f64;
    let cer_h = cer_n.powf(-(pf / ((3.0 + pf) * (3.0 + 2.0 * pf))));
    let mid = |a: f64, b: f64, c: f64| -> f64 {
        let mut v = [a, b, c];
        v.sort_by(|x, y| x.total_cmp(y));
        v[1]
    };
    let mserd = rd.map(|(h, b)| [x_sd * h, x_sd * h, x_sd * b, x_sd * b]);
    let msesum = sum.map(|(h, b)| [x_sd * h, x_sd * h, x_sd * b, x_sd * b]);
    let msetwo = two.map(|(hl, hr, bl, br)| [x_sd * hl, x_sd * hr, x_sd * bl, x_sd * br]);
    let comb1 = match (mserd, msesum) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[0].min(b[0]), a[2].min(b[2]), a[2].min(b[2])]),
        _ => None,
    };
    let comb2 = match (mserd, msesum, msetwo) {
        (Some(a), Some(b), Some(t)) => Some([mid(a[0], b[0], t[0]), mid(a[0], b[0], t[1]), mid(a[2], b[2], t[2]), mid(a[2], b[2], t[3])]),
        _ => None,
    };
    let cer = |v: Option<[f64; 4]>| v.map(|a| [a[0] * cer_h, a[1] * cer_h, a[2], a[3]]);
    for (name, v) in [
        ("mserd", mserd),
        ("msetwo", msetwo),
        ("msesum", msesum),
        ("msecomb1", comb1),
        ("msecomb2", comb2),
        ("cerrd", cer(mserd)),
        ("certwo", cer(msetwo)),
        ("cersum", cer(msesum)),
        ("cercomb1", cer(comb1)),
        ("cercomb2", cer(comb2)),
    ] {
        if let Some(v) = v {
            if wanted.contains(&name) {
                out.sel.push((name, v));
            }
        }
    }
    Ok(out)
}

/// Selectors needed to compute `name` (or all of them).
fn wanted_for(name: &str, all: bool) -> Vec<&'static str> {
    if all {
        return SELECTORS.to_vec();
    }
    SELECTORS.iter().copied().filter(|s| *s == name).collect()
}

fn parse_bw_opts(cmd: &Command) -> Result<(BwOpts, String, bool), String> {
    let deriv = util::opt_num(cmd, "deriv", 5)?.unwrap_or(0.0);
    let p_given = util::opt_num(cmd, "p", 1)?;
    let q_given = util::opt_num(cmd, "q", 1)?.unwrap_or(0.0);
    let p = match p_given {
        Some(p) => p,
        None if deriv > 0.0 && q_given == 0.0 => deriv + 1.0,
        None => 1.0,
    };
    let q = if q_given == 0.0 { p + 1.0 } else { q_given };
    if p < 0.0 || q <= 0.0 || deriv < 0.0 {
        return Err(rc_err(411, "p(), q(), deriv(), nnmatch() should be positive"));
    }
    if p.fract() != 0.0 || q.fract() != 0.0 || deriv.fract() != 0.0 {
        return Err(rc_err(126, "p(), q(), deriv() and nnmatch() should be integers"));
    }
    if p >= q {
        return Err(rc_err(125, "q() should be higher than p()"));
    }
    if deriv > p && deriv > 0.0 {
        return Err(rc_err(125, "deriv() can not be higher than p()"));
    }
    let kernel = Kernel::parse(cmd.opt("kernel", 3).unwrap_or(""), Kernel::Triangular)?;
    let bwselect = opt_word(cmd, "bwselect", 3).unwrap_or_default().to_lowercase();
    if ["cct", "ik", "cv"].contains(&bwselect.as_str()) {
        return Err(rc_err(7, "bwselect() options IK, CCT and CV have been deprecated. Please see help for new options"));
    }
    if !bwselect.is_empty() && !SELECTORS.contains(&bwselect.as_str()) {
        return Err(rc_err(7, "bwselect() incorrectly specified"));
    }
    let onoff = |name: &str, min: usize| -> Result<bool, String> {
        match opt_word(cmd, name, min).map(|s| s.to_lowercase()) {
            None => Ok(true),
            Some(s) if s == "on" => Ok(true),
            Some(s) if s == "off" => Ok(false),
            Some(s) => Err(rc_err(198, format!("{name}() incorrectly specified (received '{s}'); allowed: on, off."))),
        }
    };
    let masspoints = opt_word(cmd, "masspoints", 4).unwrap_or_else(|| "adjust".into()).to_lowercase();
    if !["adjust", "check", "off"].contains(&masspoints.as_str()) {
        return Err(rc_err(198, format!("masspoints() incorrectly specified (received '{masspoints}'); allowed: adjust, check, off.")));
    }
    let bwcheck = util::opt_num(cmd, "bwcheck", 7)?.unwrap_or(0.0);
    if bwcheck < 0.0 || bwcheck.fract() != 0.0 {
        return Err(rc_err(125, "bwcheck() must be a non-negative integer (0 = unset)"));
    }
    let o = BwOpts {
        p: p as usize,
        q: q as usize,
        deriv: deriv as usize,
        kernel,
        scaleregul: util::opt_num(cmd, "scaleregul", 6)?.unwrap_or(1.0),
        masspoints,
        bwcheck: bwcheck as usize,
        bwrestrict: onoff("bwrestrict", 10)?,
        stdvars: onoff("stdvars", 7)?,
    };
    Ok((o, bwselect, cmd.has_opt("all", 3)))
}

fn check_sample(d: &RdData, o: &BwOpts) -> Result<(), String> {
    if d.c <= d.x_min || d.c >= d.x_max {
        return Err(rc_err(125, format!("c() should be set within the range of {}", d.xname)));
    }
    let (ml, mr) = (n_unique_sorted(&d.left.x), n_unique_sorted(&d.right.x));
    if ml.min(mr) < o.q + 1 {
        let side = if ml < o.q + 1 { "left" } else { "right" };
        return Err(rc_err(2001, format!("Not enough distinct running-variable values on the {side} side of the cutoff ({}) to fit a polynomial of order q = {}.", ml.min(mr), o.q)));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// A line built with column stops, like `_col(#)` in display directives.
pub(crate) struct Ln(pub String);

impl Ln {
    pub(crate) fn new() -> Ln {
        Ln(String::new())
    }
    /// Move to column `c` (1-based); never moves backwards.
    pub(crate) fn col(mut self, c: usize) -> Ln {
        let cur = self.0.chars().count();
        if cur + 1 < c {
            self.0.push_str(&" ".repeat(c - 1 - cur));
        }
        self
    }
    pub(crate) fn t(mut self, s: &str) -> Ln {
        self.0.push_str(s);
        self
    }
    pub(crate) fn s(self) -> String {
        self.0.trim_end().to_string()
    }
}

pub(crate) fn ralign(s: &str, w: usize) -> String {
    format!("{s:>w$}")
}

pub(crate) fn ff(v: f64, w: usize, d: usize) -> String {
    if is_missing(v) || !v.is_finite() {
        return format!("{:>w$}", ".");
    }
    format!("{v:>w$.d$}")
}

pub(crate) fn fg(v: f64, w: usize) -> String {
    if !v.is_finite() {
        return format!("{:>w$}", ".");
    }
    crate::format::fmt(v, &format!("%{w}.0g"))
}

pub(crate) fn hline(n: usize) -> String {
    "-".repeat(n)
}

fn num_text(v: f64) -> String {
    crate::format::fmt_g(v, 9).trim().to_string()
}

// ---------------------------------------------------------------------------
// rdrobust
// ---------------------------------------------------------------------------

struct EstInput<'a> {
    c: f64,
    p: usize,
    q: usize,
    deriv: usize,
    h: f64,
    b: f64,
    kernel: Kernel,
    vce: &'a VceSpec,
}

/// Fitted pieces before the variance (which needs the s vector).
struct SidePrep {
    e: Side,
    r_p: Cols,
    r_q: Cols,
    w_h: Vec<f64>,
    w_b: Vec<f64>,
    inv_gp: Matrix,
    inv_gq: Matrix,
    q_q: Cols,
    beta_p: Matrix,
    beta_q: Matrix,
    beta_bc: Matrix,
    n_h: usize,
    n_b: usize,
}

fn side_prep(s: &Side, a: &EstInput) -> Result<SidePrep, String> {
    let w_h_all = s.weights(a.c, a.h, a.kernel);
    let w_b_all = s.weights(a.c, a.b, a.kernel);
    let ind_h = positive(&w_h_all);
    let ind_b = positive(&w_b_all);
    let uniq = |ind: &[usize]| n_unique_sorted(&ind.iter().map(|&i| s.x[i]).collect::<Vec<_>>());
    if uniq(&ind_h) < a.p + 1 || uniq(&ind_b) < a.q + 1 {
        return Err(rc_err(2001, "Not enough distinct running-variable values with positive weight to fit the requested polynomials on each side of the cutoff."));
    }
    let ind = if a.h > a.b { ind_h.clone() } else { ind_b.clone() };
    let e = s.sub(&ind);
    let w_h: Vec<f64> = ind.iter().map(|&i| w_h_all[i]).collect();
    let w_b: Vec<f64> = ind.iter().map(|&i| w_b_all[i]).collect();
    let r_q = powers(&e.x, a.c, a.q);
    let r_p: Cols = r_q[..=a.p].to_vec();
    let inv_gq = inv_sym(&cross(&r_q, &w_b, &r_q));
    let inv_gp = inv_sym(&cross(&r_p, &w_h, &r_p));
    let n = e.x.len();
    let up1: Vec<f64> = e.x.iter().map(|xi| ((xi - a.c) / a.h).powi(a.p as i32 + 1)).collect();
    let l: Vec<f64> = (0..=a.p).map(|j| (0..n).map(|i| r_p[j][i] * w_h[i] * up1[i]).sum()).collect();
    // row p+1 of inv_gq * R_q', times W_b
    let rowq: Vec<f64> = (0..n).map(|i| (0..=a.q).map(|j| inv_gq[(a.p + 1, j)] * r_q[j][i]).sum::<f64>() * w_b[i]).collect();
    let hp = a.h.powi(a.p as i32 + 1);
    let q_q: Cols = (0..=a.p).map(|j| (0..n).map(|i| r_p[j][i] * w_h[i] - hp * l[j] * rowq[i]).collect()).collect();
    let d = e.d_cols();
    let beta_p = coefs(&r_p, &w_h, &d, &inv_gp);
    let beta_q = coefs(&r_q, &w_b, &d, &inv_gq);
    let ones = vec![1.0; n];
    let beta_bc = coefs(&q_q, &ones, &d, &inv_gp);
    Ok(SidePrep { e, r_p, r_q, w_h, w_b, inv_gp, inv_gq, q_q, beta_p, beta_q, beta_bc, n_h: ind_h.len(), n_b: ind_b.len() })
}

/// Variance pieces of one side for a given combination vector s.
fn side_var(sp: &SidePrep, a: &EstInput, s: &[f64]) -> (Matrix, Matrix) {
    let d = sp.e.d_cols();
    let res_h = residuals(a.vce.res, &sp.e, &d, &sp.r_p, &sp.w_h, &sp.inv_gp, &sp.beta_p, a.p + 1);
    let res_b = match a.vce.res {
        ResKind::Nn(_) => res_h.clone(),
        _ => residuals(a.vce.res, &sp.e, &d, &sp.r_q, &sp.w_b, &sp.inv_gq, &sp.beta_q, a.q + 1),
    };
    let rx = scaled(&sp.r_p, &sp.w_h);
    let cl = sp.e.cl.as_deref();
    let m_cl = meat(&rx, &res_h, s, cl, 0);
    let m_rb = meat(&sp.q_q, &res_b, s, cl, a.q + 1);
    let v_cl = matmul(&matmul(&sp.inv_gp, &m_cl), &sp.inv_gp);
    let v_rb = matmul(&matmul(&sp.inv_gp, &m_rb), &sp.inv_gp);
    (v_cl, v_rb)
}

pub struct Rdrobust;

struct RdResult {
    tau_cl: f64,
    tau_bc: f64,
    se_cl: f64,
    se_rb: f64,
    tau_l: [f64; 2],
    tau_r: [f64; 2],
    bias: [f64; 2],
    first: Option<(f64, f64, f64, f64)>,
    first_lr: Option<[f64; 4]>,
    beta_y: [Vec<f64>; 2],
    beta_t: Option<[Vec<f64>; 2]>,
    gamma: Option<Matrix>,
    v_cl: [Matrix; 2],
    v_rb: [Matrix; 2],
    n_h: [usize; 2],
    n_b: [usize; 2],
    g: [usize; 2],
}

fn estimate(d: &RdData, a_l: &EstInput, a_r: &EstInput, scalepar: f64) -> Result<RdResult, String> {
    let pl = side_prep(&d.left, a_l)?;
    let pr = side_prep(&d.right, a_r)?;
    let nu = a_l.deriv;
    let nf = factorial(nu);
    let dt = d.fuzzy.is_some() as usize;
    let dz = d.covs.len();
    let diffm = |a: &Matrix, b: &Matrix| -> Matrix { b.sub(a).unwrap() };
    let beta_p = diffm(&pl.beta_p, &pr.beta_p);
    let beta_bc = diffm(&pl.beta_bc, &pr.beta_bc);
    let mut s_y: Vec<f64> = vec![1.0];
    let mut sv_t: Vec<f64> = Vec::new();
    let mut gamma = None;
    let tau_cl;
    let tau_bc;
    let tau_y_cl_l;
    let tau_y_cl_r;
    let tau_y_bc_l;
    let tau_y_bc_r;
    let bias_l;
    let bias_r;
    let beta_y_l: Vec<f64>;
    let beta_y_r: Vec<f64>;
    let mut beta_t = None;
    let mut first = None;
    let mut first_lr = None;
    let row = |m: &Matrix, s: &[f64], cols: &[usize]| -> f64 { cols.iter().zip(s).map(|(&j, sv)| sv * m[(nu, j)]).sum() };
    let full_row = |m: &Matrix, s: &[f64], cols: &[usize]| -> Vec<f64> { (0..m.rows).map(|i| cols.iter().zip(s).map(|(&j, sv)| sv * m[(i, j)]).sum::<f64>() * nf * scalepar).collect() };
    let mut tau_t_parts = None;
    if dz == 0 {
        let k = scalepar * nf;
        tau_cl = k * beta_p[(nu, 0)];
        tau_bc = k * beta_bc[(nu, 0)];
        tau_y_cl_l = k * pl.beta_p[(nu, 0)];
        tau_y_cl_r = k * pr.beta_p[(nu, 0)];
        tau_y_bc_l = k * pl.beta_bc[(nu, 0)];
        tau_y_bc_r = k * pr.beta_bc[(nu, 0)];
        beta_y_l = (0..pl.beta_p.rows).map(|i| k * pl.beta_p[(i, 0)]).collect();
        beta_y_r = (0..pr.beta_p.rows).map(|i| k * pr.beta_p[(i, 0)]).collect();
        let mut bl = tau_y_cl_l - tau_y_bc_l;
        let mut br = tau_y_cl_r - tau_y_bc_r;
        let (mut tc, mut tb) = (tau_cl, tau_bc);
        if dt > 0 {
            let tt_cl = nf * beta_p[(nu, 1)];
            let tt_bc = nf * beta_bc[(nu, 1)];
            beta_t = Some([(0..pl.beta_p.rows).map(|i| nf * pl.beta_p[(i, 1)]).collect::<Vec<_>>(), (0..pr.beta_p.rows).map(|i| nf * pr.beta_p[(i, 1)]).collect()]);
            s_y = vec![1.0 / tt_cl, -(tau_cl / (tt_cl * tt_cl))];
            tc = tau_cl / tt_cl;
            tb = tc - (s_y[0] * (tau_cl - tau_bc) + s_y[1] * (tt_cl - tt_bc));
            sv_t = vec![0.0, 1.0];
            let tl = [nf * pl.beta_p[(nu, 1)], nf * pr.beta_p[(nu, 1)], nf * pl.beta_bc[(nu, 1)], nf * pr.beta_bc[(nu, 1)]];
            bl = s_y[0] * (tau_y_cl_l - tau_y_bc_l) + s_y[1] * (tl[0] - tl[2]);
            br = s_y[0] * (tau_y_cl_r - tau_y_bc_r) + s_y[1] * (tl[1] - tl[3]);
            tau_t_parts = Some((tt_cl, tt_bc));
            first_lr = Some(tl);
        }
        bias_l = bl;
        bias_r = br;
        let (a, b) = (tc, tb);
        return finish_estimate(d, &pl, &pr, a_l, a_r, nu, scalepar, s_y, sv_t, a, b, [tau_y_cl_l, tau_y_cl_r], [tau_y_bc_l, tau_y_bc_r], [bias_l, bias_r], tau_t_parts, first_lr, [beta_y_l, beta_y_r], beta_t, gamma);
    }
    // covariates
    let g = covariate_gamma(&[(&pl.r_p, &pl.w_h, &pl.e, &pl.inv_gp), (&pr.r_p, &pr.w_h, &pr.e, &pr.inv_gp)], dt);
    let zcols: Vec<usize> = (1 + dt..1 + dt + dz).collect();
    let sy0: Vec<f64> = std::iter::once(1.0).chain((0..dz).map(|j| -g[(j, 0)])).collect();
    let ycols: Vec<usize> = std::iter::once(0).chain(zcols.iter().copied()).collect();
    let k = scalepar * nf;
    let t_cl = k * row(&beta_p, &sy0, &ycols);
    let t_bc = k * row(&beta_bc, &sy0, &ycols);
    tau_y_cl_l = k * row(&pl.beta_p, &sy0, &ycols);
    tau_y_cl_r = k * row(&pr.beta_p, &sy0, &ycols);
    tau_y_bc_l = k * row(&pl.beta_bc, &sy0, &ycols);
    tau_y_bc_r = k * row(&pr.beta_bc, &sy0, &ycols);
    beta_y_l = full_row(&pl.beta_p, &sy0, &ycols);
    beta_y_r = full_row(&pr.beta_p, &sy0, &ycols);
    if dt == 0 {
        s_y = sy0;
        tau_cl = t_cl;
        tau_bc = t_bc;
        bias_l = tau_y_cl_l - tau_y_bc_l;
        bias_r = tau_y_cl_r - tau_y_bc_r;
    } else {
        let st: Vec<f64> = std::iter::once(1.0).chain((0..dz).map(|j| -g[(j, 1)])).collect();
        let tcols: Vec<usize> = std::iter::once(1).chain(zcols.iter().copied()).collect();
        sv_t = vec![0.0, 1.0];
        sv_t.extend((0..dz).map(|j| -g[(j, 1)]));
        let tt_cl = nf * row(&beta_p, &st, &tcols);
        let tt_bc = nf * row(&beta_bc, &st, &tcols);
        let tl = [nf * row(&pl.beta_p, &st, &tcols), nf * row(&pr.beta_p, &st, &tcols), nf * row(&pl.beta_bc, &st, &tcols), nf * row(&pr.beta_bc, &st, &tcols)];
        beta_t = Some([full_row(&pl.beta_p, &st, &tcols).iter().map(|v| v / scalepar).collect::<Vec<_>>(), full_row(&pr.beta_p, &st, &tcols).iter().map(|v| v / scalepar).collect()]);
        let s2 = [1.0 / tt_cl, -(t_cl / (tt_cl * tt_cl))];
        tau_cl = t_cl / tt_cl;
        tau_bc = tau_cl - (s2[0] * (t_cl - t_bc) + s2[1] * (tt_cl - tt_bc));
        bias_l = s2[0] * (tau_y_cl_l - tau_y_bc_l) + s2[1] * (tl[0] - tl[2]);
        bias_r = s2[0] * (tau_y_cl_r - tau_y_bc_r) + s2[1] * (tl[1] - tl[3]);
        s_y = vec![s2[0], s2[1]];
        s_y.extend((0..dz).map(|j| -(1.0 / tt_cl) * g[(j, 0)] + (t_cl / (tt_cl * tt_cl)) * g[(j, 1)]));
        first = Some((tt_cl, tt_bc));
        first_lr = Some(tl);
    }
    gamma = Some(g);
    finish_estimate(d, &pl, &pr, a_l, a_r, nu, scalepar, s_y, sv_t, tau_cl, tau_bc, [tau_y_cl_l, tau_y_cl_r], [tau_y_bc_l, tau_y_bc_r], [bias_l, bias_r], first, first_lr, [beta_y_l, beta_y_r], beta_t, gamma)
}

#[allow(clippy::too_many_arguments)]
fn finish_estimate(
    d: &RdData,
    pl: &SidePrep,
    pr: &SidePrep,
    a_l: &EstInput,
    a_r: &EstInput,
    nu: usize,
    scalepar: f64,
    s_y: Vec<f64>,
    sv_t: Vec<f64>,
    tau_cl: f64,
    tau_bc: f64,
    tau_cl_lr: [f64; 2],
    tau_bc_lr: [f64; 2],
    bias: [f64; 2],
    first: Option<(f64, f64)>,
    first_lr: Option<[f64; 4]>,
    beta_y: [Vec<f64>; 2],
    beta_t: Option<[Vec<f64>; 2]>,
    gamma: Option<Matrix>,
) -> Result<RdResult, String> {
    let nf = factorial(nu);
    let (vcl_l, vrb_l) = side_var(pl, a_l, &s_y);
    let (vcl_r, vrb_r) = side_var(pr, a_r, &s_y);
    let k2 = scalepar * scalepar * nf * nf;
    let se_cl = (k2 * (vcl_l[(nu, nu)] + vcl_r[(nu, nu)])).sqrt();
    let se_rb = (k2 * (vrb_l[(nu, nu)] + vrb_r[(nu, nu)])).sqrt();
    let first = match first {
        Some((tc, tb)) => {
            let (tcl_l, trb_l) = side_var(pl, a_l, &sv_t);
            let (tcl_r, trb_r) = side_var(pr, a_r, &sv_t);
            let sc = (nf * nf * (tcl_l[(nu, nu)] + tcl_r[(nu, nu)])).sqrt();
            let sr = (nf * nf * (trb_l[(nu, nu)] + trb_r[(nu, nu)])).sqrt();
            Some((tc, tb, sc, sr))
        }
        None => None,
    };
    let _ = d;
    Ok(RdResult {
        tau_cl,
        tau_bc,
        se_cl,
        se_rb,
        tau_l: [tau_cl_lr[0], tau_bc_lr[0]],
        tau_r: [tau_cl_lr[1], tau_bc_lr[1]],
        bias,
        first,
        first_lr,
        beta_y,
        beta_t,
        gamma,
        v_cl: [vcl_l, vcl_r],
        v_rb: [vrb_l, vrb_r],
        n_h: [pl.n_h, pr.n_h],
        n_b: [pl.n_b, pr.n_b],
        g: [n_clusters(&pl.e), n_clusters(&pr.e)],
    })
}

fn col_vec(v: &[f64], names: Vec<String>) -> Matrix {
    Matrix::col_vector(v).with_names(names, vec!["y1".into()])
}

impl CommandHandler for Rdrobust {
    fn name(&self) -> &'static str {
        "rdrobust"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vce = parse_vce(cmd.opt("vce", 3))?;
        let (o, mut bwselect, all) = parse_bw_opts(cmd)?;
        let level = util::opt_num(cmd, "level", 3)?.unwrap_or(95.0);
        if level <= 0.0 || level >= 100.0 {
            return Err(rc_err(125, "level() should be a number in (0, 100)"));
        }
        let scalepar = util::opt_num(cmd, "scalepar", 6)?.unwrap_or(1.0);
        let rho = util::opt_num(cmd, "rho", 3)?.unwrap_or(0.0);
        if rho < 0.0 {
            return Err(rc_err(125, "rho() should be strictly positive"));
        }
        let detail = cmd.has_opt("detail", 6);
        let nowarn = cmd.has_opt("nowarnings", 6);
        let d = prepare(ds, st, cmd, true, &vce)?;
        check_sample(&d, &o)?;
        let two = |name: &str| -> Result<Option<(f64, f64)>, String> {
            match opt_word(cmd, name, 1) {
                None => Ok(None),
                Some(t) => {
                    let v: Vec<f64> = t.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, format!("{name}() invalid")))).collect::<Result<_, _>>()?;
                    match v.len() {
                        1 => Ok(Some((v[0], v[0]))),
                        2 => Ok(Some((v[0], v[1]))),
                        _ => Err(rc_err(125, format!("{name}() only accepts two inputs"))),
                    }
                }
            }
        };
        let mut h_given = two("h")?;
        let b_given = two("b")?;
        let range_l = (d.c - d.x_min).abs();
        let range_r = (d.x_max - d.c).abs();
        let mut notes: Vec<String> = Vec::new();
        if d.covs_dropped {
            notes.push("Multicollinearity issue detected in covs. Redundant covariates were removed.".into());
        }
        if d.n < 20 && h_given.is_none() {
            notes.push("Not enough observations to perform bandwidth calculations. Using the maximum distance from the cutoff for h.".into());
            let r = range_l.max(range_r);
            h_given = Some((r, r));
        }
        let mut masspoints_found = false;
        let (mut m_l, mut m_r) = (n_unique_sorted(&d.left.x), n_unique_sorted(&d.right.x));
        let (h_l, h_r, mut b_l, mut b_r);
        if let Some((hl, hr)) = h_given {
            bwselect = "Manual".into();
            h_l = hl;
            h_r = hr;
            let (bl, br) = b_given.unwrap_or((hl, hr));
            b_l = bl;
            b_r = br;
            if rho > 0.0 {
                if b_given.is_some() {
                    notes.push("Note: both b() and rho() were specified; rho() takes precedence and b() is ignored (b = h/rho).".into());
                }
                b_l = h_l / rho;
                b_r = h_r / rho;
            }
        } else {
            if bwselect.is_empty() {
                bwselect = "mserd".into();
            }
            let mut wanted = wanted_for(&bwselect, false);
            // combined selectors need their parts
            let parts: &[&str] = match bwselect.as_str() {
                "msecomb1" | "cercomb1" => &["mserd", "msesum", "msecomb1"],
                "msecomb2" | "cercomb2" => &["mserd", "msesum", "msetwo", "msecomb1", "msecomb2"],
                _ => &[],
            };
            wanted.extend(parts.iter().copied());
            let bw = select_bandwidths(&d, &o, &vce, &wanted, &mut notes)?;
            masspoints_found = bw.masspoints_found;
            m_l = bw.m_l;
            m_r = bw.m_r;
            let v = bw.get(&bwselect).ok_or_else(|| rc_err(498, "bandwidth selection failed"))?;
            h_l = v[0];
            h_r = v[1];
            b_l = v[2];
            b_r = v[3];
            if rho > 0.0 {
                b_l = h_l / rho;
                b_r = h_r / rho;
            }
        }
        if !(h_l > 0.0 && h_r > 0.0 && b_l > 0.0 && b_r > 0.0) {
            return Err(rc_err(198, "bandwidths must be positive"));
        }
        let a_l = EstInput { c: d.c, p: o.p, q: o.q, deriv: o.deriv, h: h_l, b: b_l, kernel: o.kernel, vce: &vce };
        let a_r = EstInput { c: d.c, p: o.p, q: o.q, deriv: o.deriv, h: h_r, b: b_r, kernel: o.kernel, vce: &vce };
        let r = estimate(&d, &a_l, &a_r, scalepar)?;
        let quant = -dist::invnormal((1.0 - level / 100.0).abs() / 2.0);
        let pv = |t: f64, s: f64| 2.0 * dist::normal(-(t / s).abs());
        // ---- output ---------------------------------------------------------
        let mut out: Vec<String> = Vec::new();
        out.extend(notes.iter().cloned());
        if r.n_h[0] < 10 || r.n_h[1] < 10 || r.n_b[0] < 10 || r.n_b[1] < 10 {
            out.push("Estimates might be unreliable due to low number of effective observations.".into());
        }
        if d.clustvar.is_some() && !nowarn {
            let gh = r.g[0].min(r.g[1]);
            if gh <= o.p + 1 {
                out.push(format!("Warning: only {} (left) and {} (right) clusters within the bandwidth. With p+1 = {} or fewer clusters on a side, cluster-robust inference may be unreliable and the estimated variance may be degenerate.", r.g[0], r.g[1], o.p + 1));
            } else if gh < 10 {
                out.push(format!("Warning: only {} (left) and {} (right) clusters within the bandwidth. Cluster-robust standard errors are unreliable with fewer than 10 clusters on a side.", r.g[0], r.g[1]));
            }
        }
        out.push(String::new());
        let kind = if o.deriv == 0 { String::new() } else if o.deriv == 1 { " Kink".into() } else { String::new() };
        let design = if d.fuzzy.is_some() { "Fuzzy" } else { "Sharp" };
        let mut title = format!("{design}{kind} RD estimates using local polynomial regression.");
        if !d.covs.is_empty() {
            title = format!("Covariate-adjusted {title}");
        }
        if o.deriv >= 2 {
            title = format!("{} Derivative of order {}.", title, o.deriv);
        }
        out.push(title);
        out.push(String::new());
        let cut = format!(" Cutoff c = {}", num_text(d.c));
        out.push(Ln::new().t(&ralign(&cut, 18)).col(19).t(" | ").col(21).t("Left of c").col(33).t("Right of c").col(55).t("Number of obs = ").t(&ff(d.n as f64, 10, 0)).s());
        out.push(Ln::new().t(&hline(19)).t("+").t(&hline(22)).col(55).t("BW type       = ").t(&ralign(&bwselect, 10)).s());
        let two_cols = |label: &str, a: String, b: String| -> Ln { Ln::new().t(&ralign(label, 18)).col(19).t(" | ").col(21).t(&a).col(34).t(&b) };
        out.push(two_cols("Number of obs", ff(d.left.x.len() as f64, 9, 0), ff(d.right.x.len() as f64, 9, 0)).col(55).t("Kernel        = ").t(&ralign(o.kernel.title(), 10)).s());
        out.push(two_cols("Eff. Number of obs", ff(r.n_h[0] as f64, 9, 0), ff(r.n_h[1] as f64, 9, 0)).col(55).t("VCE method    = ").t(&ralign(&vce.label, 10)).s());
        out.push(two_cols("Order est. (p)", ff(o.p as f64, 9, 0), ff(o.p as f64, 9, 0)).s());
        out.push(two_cols("Order bias (q)", ff(o.q as f64, 9, 0), ff(o.q as f64, 9, 0)).s());
        out.push(two_cols("BW est. (h)", ff(h_l, 9, 3), ff(h_r, 9, 3)).s());
        out.push(two_cols("BW bias (b)", ff(b_l, 9, 3), ff(b_r, 9, 3)).s());
        out.push(two_cols("rho (h/b)", ff(h_l / b_l, 9, 3), ff(h_r / b_r, 9, 3)).s());
        if o.masspoints == "check" || masspoints_found {
            out.push(two_cols("Unique obs", ff(m_l as f64, 9, 0), ff(m_r as f64, 9, 0)).s());
        }
        let (g_l, g_r) = (n_clusters(&d.left), n_clusters(&d.right));
        if d.clustvar.is_some() {
            out.push(two_cols("Number of clusters", ff(g_l as f64, 9, 0), ff(g_r as f64, 9, 0)).s());
        }
        out.push(String::new());
        let lvl = num_text(level);
        let table = |out: &mut Vec<String>, tc: f64, tb: f64, sc: f64, sr: f64| {
            let row = |name: &str, coef: String, se: String, z: f64, p: f64, lo: f64, hi: f64| -> String {
                Ln::new().t(&ralign(name, 18)).col(19).t(" | ").col(22).t(&coef).col(33).t(&se).col(43).t(&ff(z, 5, 4)).col(52).t(&ff(p, 5, 3)).col(60).t(&fg(lo, 8)).col(73).t(&fg(hi, 8)).s()
            };
            if all || detail {
                out.push(Ln::new().t(&ralign("Method", 18)).col(19).t(" | ").col(24).t("Coef.").col(33).t("Std. Err.").col(46).t("z").col(52).t("P>|z|").col(61).t(&format!("[{lvl}% Conf. Interval]")).s());
                out.push(format!("{}+{}", hline(19), hline(60)));
                out.push(row("Conventional", fg(tc, 7), fg(sc, 7), tc / sc, pv(tc, sc), tc - quant * sc, tc + quant * sc));
                if all {
                    out.push(row("Bias-corrected", fg(tb, 7), fg(sc, 7), tb / sc, pv(tb, sc), tb - quant * sc, tb + quant * sc));
                    out.push(row("Robust", fg(tb, 7), fg(sr, 7), tb / sr, pv(tb, sr), tb - quant * sr, tb + quant * sr));
                } else {
                    out.push(row("Robust", format!("{:>7}", "    -"), format!("{:>7}", "    -"), tb / sr, pv(tb, sr), tb - quant * sr, tb + quant * sr));
                }
            } else {
                out.push(Ln::new().t(&ralign("", 18)).col(19).t(" | ").col(22).t("Point").col(35).t(" | ").t("Robust Inference").s());
                out.push(Ln::new().t(&ralign("", 18)).col(19).t(" | ").col(22).t("Estimate").col(35).t(" | ").t("z-stat").col(52).t("P>|z|").col(61).t(&format!("[{lvl}% Conf. Interval]")).s());
                out.push(format!("{}+{}", hline(19), hline(60)));
                out.push(Ln::new().t(&ralign("RD Effect", 18)).col(19).t(" | ").col(22).t(&fg(tc, 7)).col(35).t(" | ").t(&ff(tb / sr, 5, 4)).col(52).t(&ff(pv(tb, sr), 5, 3)).col(61).t(&fg(tb - quant * sr, 8)).col(73).t(&fg(tb + quant * sr, 8)).s());
            }
            out.push(format!("{}+{}", hline(19), hline(60)));
        };
        if let (Some((tc, tb, sc, sr)), Some(f)) = (r.first, &d.fuzzy) {
            out.push(format!("First-stage estimates. Outcome: {f}. Running variable: {}.", d.xname));
            out.push(format!("{}+{}", hline(19), hline(60)));
            table(&mut out, tc, tb, sc, sr);
            out.push(String::new());
        }
        match &d.fuzzy {
            None => out.push(format!("Outcome: {}. Running variable: {}.", d.yname, d.xname)),
            Some(f) => out.push(format!("Treatment effect estimates. Outcome: {}. Running variable: {}. Treatment Status: {f}.", d.yname, d.xname)),
        }
        out.push(format!("{}+{}", hline(19), hline(60)));
        table(&mut out, r.tau_cl, r.tau_bc, r.se_cl, r.se_rb);
        if !d.covs.is_empty() {
            out.push(format!("Covariate-adjusted estimates. Additional covariates included: {}", d.covs.len()));
        }
        if let Some(cv) = &d.clustvar {
            out.push(format!("Std. Err. adjusted for clusters in {cv}"));
        }
        if scalepar != 1.0 {
            out.push(format!("Scale parameter: {}", num_text(scalepar)));
        }
        if o.scaleregul != 1.0 {
            out.push(format!("Scale regularization: {}", num_text(o.scaleregul)));
        }
        if o.masspoints == "check" {
            out.push("Running variable checked for mass points.".into());
        }
        if o.masspoints == "adjust" && masspoints_found {
            out.push("Estimates adjusted for mass points in the running variable.".into());
        }
        if !nowarn {
            if h_l >= range_l || h_r >= range_r {
                out.push("WARNING: bandwidth h greater than the range of the data.".into());
            }
            if b_l >= range_l || b_r >= range_r {
                out.push("WARNING: bandwidth b greater than the range of the data.".into());
            }
            if r.n_h[0] < 20 || r.n_h[1] < 20 {
                out.push("WARNING: bandwidth h too low.".into());
            }
            if r.n_b[0] < 20 || r.n_b[1] < 20 {
                out.push("WARNING: bandwidth b too low.".into());
            }
        }
        let text = out.join("\n");
        // ---- e() ------------------------------------------------------------
        let names = vec!["Conventional".to_string(), "Bias-corrected".to_string(), "Robust".to_string()];
        let (vcl, vrb) = (r.se_cl * r.se_cl, r.se_rb * r.se_rb);
        let mut mask = vec![false; ds.n_obs];
        for &i in &d.rows {
            mask[i] = true;
        }
        let ci = |t: f64, s: f64| (t - quant * s, t + quant * s);
        let e = st.begin_e();
        e.set_mat("b", Matrix::row_vector(&[r.tau_cl, r.tau_bc, r.tau_bc]).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", Matrix::from_diag(&[vcl, vcl, vrb]).with_names(names.clone(), names.clone()));
        e.sample = Some(mask);
        e.set("N", d.n as f64);
        e.set("N_l", d.left.x.len() as f64);
        e.set("N_r", d.right.x.len() as f64);
        e.set("N_h_l", r.n_h[0] as f64);
        e.set("N_h_r", r.n_h[1] as f64);
        e.set("N_b_l", r.n_b[0] as f64);
        e.set("N_b_r", r.n_b[1] as f64);
        e.set("c", d.c);
        e.set("p", o.p as f64);
        e.set("q", o.q as f64);
        e.set("h_l", h_l);
        e.set("h_r", h_r);
        e.set("b_l", b_l);
        e.set("b_r", b_r);
        let mut bws = Matrix::zeros(2, 2);
        bws[(0, 0)] = h_l;
        bws[(0, 1)] = h_r;
        bws[(1, 0)] = b_l;
        bws[(1, 1)] = b_r;
        e.set_mat("bws", bws.with_names(vec!["h".into(), "b".into()], vec!["left".into(), "right".into()]));
        e.set("tau_cl", r.tau_cl);
        e.set("tau_cl_l", r.tau_l[0]);
        e.set("tau_cl_r", r.tau_r[0]);
        e.set("tau_bc", r.tau_bc);
        e.set("tau_bc_l", r.tau_l[1]);
        e.set("tau_bc_r", r.tau_r[1]);
        e.set("bias_l", r.bias[0]);
        e.set("bias_r", r.bias[1]);
        e.set("se_tau_cl", r.se_cl);
        e.set("se_tau_rb", r.se_rb);
        e.set("level", level);
        let (a, b) = ci(r.tau_cl, r.se_cl);
        e.set("ci_l_cl", a);
        e.set("ci_r_cl", b);
        let (a2, b2) = ci(r.tau_bc, r.se_rb);
        e.set("ci_l_rb", a2);
        e.set("ci_r_rb", b2);
        e.set("pv_cl", pv(r.tau_cl, r.se_cl));
        e.set("pv_bc", pv(r.tau_bc, r.se_cl));
        e.set("pv_rb", pv(r.tau_bc, r.se_rb));
        if d.clustvar.is_some() {
            e.set("n_clust", (g_l + g_r) as f64);
        }
        if let Some((tc, tb, sc, sr)) = r.first {
            e.set("tau_T_cl", tc);
            e.set("tau_T_bc", tb);
            e.set("se_tau_T_cl", sc);
            e.set("se_tau_T_rb", sr);
            if let Some(tl) = r.first_lr {
                e.set("tau_T_cl_l", tl[0]);
                e.set("tau_T_cl_r", tl[1]);
                e.set("tau_T_bc_l", tl[2]);
                e.set("tau_T_bc_r", tl[3]);
            }
        }
        if let Some(bt) = &r.beta_t {
            let rn: Vec<String> = (0..bt[0].len()).map(|i| format!("r{}", i + 1)).collect();
            e.set_mat("beta_T_p_l", col_vec(&bt[0], rn.clone()));
            e.set_mat("beta_T_p_r", col_vec(&bt[1], rn));
        }
        let rn: Vec<String> = (0..r.beta_y[0].len()).map(|i| format!("r{}", i + 1)).collect();
        e.set_mat("beta_Y_p_l", col_vec(&r.beta_y[0], rn.clone()));
        e.set_mat("beta_Y_p_r", col_vec(&r.beta_y[1], rn));
        if let Some(g) = &r.gamma {
            e.set_mat("coef_covs", g.clone());
        }
        e.set_mat("V_cl_l", r.v_cl[0].clone());
        e.set_mat("V_cl_r", r.v_cl[1].clone());
        e.set_mat("V_rb_l", r.v_rb[0].clone());
        e.set_mat("V_rb_r", r.v_rb[1].clone());
        let mut cim = Matrix::zeros(3, 2);
        let (c1, c2) = ci(r.tau_bc, r.se_cl);
        cim[(0, 0)] = a;
        cim[(0, 1)] = b;
        cim[(1, 0)] = c1;
        cim[(1, 1)] = c2;
        cim[(2, 0)] = a2;
        cim[(2, 1)] = b2;
        e.set_mat("ci", cim.with_names(names.clone(), vec!["ll".into(), "ul".into()]));
        let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
        e.set_str("ci_rb", format!("[{} ; {}]", num_text(r3(a2)), num_text(r3(b2))));
        e.set_str("kernel", o.kernel.title());
        e.set_str("bwselect", bwselect.clone());
        e.set_str("vce_select", vce.raw.clone());
        e.set_str("vce_type", vce.label.clone());
        if !d.covs.is_empty() {
            e.set_str("covs", d.covs.join(" "));
        }
        if let Some(cv) = &d.clustvar {
            e.set_str("clustvar", cv.clone());
        }
        e.set_str("runningvar", d.xname.clone());
        e.set_str("depvar", d.yname.clone());
        let mut t = if d.fuzzy.is_some() { "Fuzzy RD estimates".to_string() } else { "Sharp RD estimates".to_string() };
        if !d.covs.is_empty() {
            t.push_str(" (covariate-adjusted)");
        }
        e.set_str("title", t);
        e.set_str("cmdline", crate::rdrobust::cmdline("rdrobust", cmd));
        e.set_str("cmd", "rdrobust");
        e.set_str("properties", "b V");
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// rdbwselect
// ---------------------------------------------------------------------------

pub struct Rdbwselect;

impl CommandHandler for Rdbwselect {
    fn name(&self) -> &'static str {
        "rdbwselect"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vce = parse_vce(cmd.opt("vce", 3))?;
        let (o, mut bwselect, all) = parse_bw_opts(cmd)?;
        if bwselect.is_empty() {
            bwselect = "mserd".into();
        }
        let d = prepare(ds, st, cmd, true, &vce)?;
        check_sample(&d, &o)?;
        if d.n < 20 {
            return Err(rc_err(2001, "Not enough observations to perform bandwidth calculations"));
        }
        let mut notes = Vec::new();
        if d.covs_dropped {
            notes.push("Multicollinearity issue detected in covs. Redundant covariates were removed.".to_string());
        }
        let mut wanted = wanted_for(&bwselect, all);
        let parts: &[&str] = match bwselect.as_str() {
            "msecomb1" | "cercomb1" => &["mserd", "msesum", "msecomb1"],
            "msecomb2" | "cercomb2" => &["mserd", "msesum", "msetwo", "msecomb1", "msecomb2"],
            "cerrd" => &["mserd"],
            "cersum" => &["msesum"],
            "certwo" => &["msetwo"],
            _ => &[],
        };
        wanted.extend(parts.iter().copied());
        let bw = select_bandwidths(&d, &o, &vce, &wanted, &mut notes)?;
        let mut out: Vec<String> = notes.clone();
        out.push(String::new());
        let design = if d.fuzzy.is_some() { "fuzzy" } else { "sharp" };
        let kink = if o.deriv == 1 { " kink" } else { "" };
        let covtxt = if d.covs.is_empty() { "" } else { "covariate-adjusted " };
        let mut title = format!("Bandwidth estimators for {covtxt}{design}{kink} RD local polynomial regression.");
        if o.deriv >= 2 {
            title = format!("{} Derivative of order {}.", title, o.deriv);
        }
        out.push(title);
        out.push(String::new());
        let cut = format!(" Cutoff c = {}", num_text(d.c));
        out.push(Ln::new().t(&ralign(&cut, 18)).col(19).t(" | ").col(21).t("Left of c").col(33).t("Right of c").col(55).t("Number of obs = ").t(&ff(d.n as f64, 10, 0)).s());
        out.push(Ln::new().t(&hline(19)).t("+").t(&hline(22)).col(55).t("Kernel        = ").t(&ralign(o.kernel.title(), 10)).s());
        let two_cols = |label: &str, a: String, b: String| -> Ln { Ln::new().t(&ralign(label, 18)).col(19).t(" | ").col(21).t(&a).col(34).t(&b) };
        out.push(two_cols("Number of obs", ff(d.left.x.len() as f64, 9, 0), ff(d.right.x.len() as f64, 9, 0)).col(55).t("VCE method    = ").t(&ralign(&vce.label, 10)).s());
        let (lmin, lmax) = (d.left.x[0], d.left.x[d.left.x.len() - 1]);
        let (rmin, rmax) = (d.right.x[0], d.right.x[d.right.x.len() - 1]);
        out.push(two_cols(&format!("Min of {}", d.xname), ff(lmin, 9, 3), ff(rmin, 9, 3)).s());
        out.push(two_cols(&format!("Max of {}", d.xname), ff(lmax, 9, 3), ff(rmax, 9, 3)).s());
        out.push(two_cols("Order est. (p)", ff(o.p as f64, 9, 0), ff(o.p as f64, 9, 0)).s());
        out.push(two_cols("Order bias (q)", ff(o.q as f64, 9, 0), ff(o.q as f64, 9, 0)).s());
        if o.masspoints == "check" || bw.masspoints_found {
            out.push(two_cols("Unique obs", ff(bw.m_l as f64, 9, 0), ff(bw.m_r as f64, 9, 0)).s());
        }
        if d.clustvar.is_some() {
            out.push(two_cols("Number of clusters", ff(bw.g_l as f64, 9, 0), ff(bw.g_r as f64, 9, 0)).s());
        }
        out.push(String::new());
        match &d.fuzzy {
            None => out.push(format!("Outcome: {}. Running variable: {}.", d.yname, d.xname)),
            Some(f) => out.push(format!("Outcome: {}. Running variable: {}. Treatment Status: {f}.", d.yname, d.xname)),
        }
        out.push(format!("{}+{}+{}", hline(19), hline(30), hline(29)));
        out.push(Ln::new().col(19).t(" | ").col(30).t("BW est. (h)").col(50).t(" | ").col(60).t("BW bias (b)").s());
        out.push(Ln::new().t(&ralign("Method", 18)).col(19).t(" | ").col(22).t("Left of c").col(40).t("Right of c").col(50).t(" | ").col(53).t("Left of c").col(70).t("Right of c").s());
        out.push(format!("{}+{}+{}", hline(19), hline(30), hline(29)));
        let show: Vec<&str> = if all { SELECTORS.to_vec() } else { vec![bwselect.as_str()] };
        for (i, name) in show.iter().enumerate() {
            if all && i == 5 {
                out.push(format!("{}+{}+{}", hline(19), hline(30), hline(29)));
            }
            if let Some(v) = bw.get(name) {
                out.push(Ln::new().t(&ralign(name, 18)).col(19).t(" | ").col(22).t(&ff(v[0], 9, 3)).col(41).t(&ff(v[1], 9, 3)).col(50).t(" | ").col(51).t(&ff(v[2], 9, 3)).col(71).t(&ff(v[3], 9, 3)).s());
            }
        }
        out.push(format!("{}+{}+{}", hline(19), hline(30), hline(29)));
        if !d.covs.is_empty() {
            out.push(format!("Covariate-adjusted estimates. Additional covariates included: {}", d.covs.len()));
        }
        if o.masspoints == "check" {
            out.push("Running variable checked for mass points.".into());
        }
        if o.masspoints == "adjust" && bw.masspoints_found {
            out.push("Estimates adjusted for mass points in the running variable.".into());
        }
        if let Some(cv) = &d.clustvar {
            out.push(format!("Std. Err. adjusted for clusters in {cv}"));
        }
        if o.scaleregul != 1.0 {
            out.push(format!("Scale regularization: {}", num_text(o.scaleregul)));
        }
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("N", d.n as f64);
        e.set("N_l", d.left.x.len() as f64);
        e.set("N_r", d.right.x.len() as f64);
        e.set("c", d.c);
        e.set("p", o.p as f64);
        e.set("q", o.q as f64);
        if d.clustvar.is_some() {
            e.set("n_clust", (bw.g_l + bw.g_r) as f64);
        }
        e.set_str("kernel", o.kernel.title());
        e.set_str("bwselect", bwselect.clone());
        e.set_str("vce_select", vce.raw.clone());
        e.set_str("vce_type", vce.label.clone());
        if !d.covs.is_empty() {
            e.set_str("covs", d.covs.join(" "));
        }
        if let Some(cv) = &d.clustvar {
            e.set_str("clustvar", cv.clone());
        }
        e.set_str("runningvar", d.xname.clone());
        e.set_str("depvar", d.yname.clone());
        let mut t = format!("RD bandwidth selection ({design})");
        if !d.covs.is_empty() {
            t.push_str(", covariate-adjusted");
        }
        e.set_str("title", t);
        e.set_str("cmdline", crate::rdrobust::cmdline("rdbwselect", cmd));
        e.set_str("cmd", "rdbwselect");
        let last = show.iter().rev().find_map(|n| bw.get(n)).unwrap_or([SYSMISS; 4]);
        e.set_mat("mat_h", Matrix::row_vector(&[last[0], last[1]]));
        e.set_mat("mat_b", Matrix::row_vector(&[last[2], last[3]]));
        for name in &show {
            if let Some(v) = bw.get(name) {
                if v[0] == v[1] && v[2] == v[3] && !name.contains("two") && !name.ends_with("comb2") {
                    e.set(&format!("h_{name}"), v[0]);
                    e.set(&format!("b_{name}"), v[2]);
                } else {
                    e.set(&format!("h_{name}_l"), v[0]);
                    e.set(&format!("h_{name}_r"), v[1]);
                    e.set(&format!("b_{name}_l"), v[2]);
                    e.set(&format!("b_{name}_r"), v[3]);
                }
            }
        }
        let _ = Results::new;
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

/// `e(cmdline)`: the command as typed.
pub(crate) fn cmdline(verb: &str, cmd: &Command) -> String {
    let raw = cmd.raw.trim();
    if raw.split_whitespace().next() == Some(cmd.verb.as_str()) {
        format!("{verb}{}", &raw[cmd.verb.len()..])
    } else {
        format!("{verb} {raw}")
    }
}
