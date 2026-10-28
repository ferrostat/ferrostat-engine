//! Manipulation testing at a cutoff with local polynomial density
//! estimators (Cattaneo, Jansson & Ma 2020, JASA):
//! `rddensity` and its bandwidth selector `rdbwdensity`.
//!
//! The empirical distribution function is regressed on polynomials of the
//! running variable on each side of the cutoff; the slope coefficients are
//! the density estimates. Standard errors come from a jackknife of the
//! empirical distribution function or from plug-in kernel constants, which
//! are computed here by exact polynomial integration.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::rdrobust::{ff, hline, ralign, Kernel, Ln};
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// Kernel constants by polynomial integration on [0, 1]
// ---------------------------------------------------------------------------

type Poly = Vec<f64>;

fn pmul(a: &Poly, b: &Poly) -> Poly {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Antiderivative vanishing at 0.
fn pint(a: &Poly) -> Poly {
    let mut out = vec![0.0];
    out.extend(a.iter().enumerate().map(|(i, c)| c / (i as f64 + 1.0)));
    out
}

fn peval(a: &Poly, x: f64) -> f64 {
    a.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

fn padd(a: &Poly, b: &Poly) -> Poly {
    let mut out = vec![0.0; a.len().max(b.len())];
    for (i, x) in a.iter().enumerate() {
        out[i] += x;
    }
    for (i, x) in b.iter().enumerate() {
        out[i] += x;
    }
    out
}

fn monomial(k: usize) -> Poly {
    let mut p = vec![0.0; k + 1];
    p[k] = 1.0;
    p
}

/// One-sided kernel on [0, 1] (normalised as a density on [-1, 1]).
fn kpoly(k: Kernel) -> Poly {
    match k {
        Kernel::Uniform => vec![0.5],
        Kernel::Triangular => vec![1.0, -1.0],
        Kernel::Epanechnikov => vec![0.75, 0.0, -0.75],
    }
}

/// int_0^1 a(u) K(u) du.
fn moment(a: &Poly, k: &Poly) -> f64 {
    peval(&pint(&pmul(a, k)), 1.0)
}

/// int int a(u) K(u) b(v) K(v) min(u, v) du dv over [0, 1]^2.
fn gmin(a: &Poly, b: &Poly, k: &Poly) -> f64 {
    let q = pmul(b, k);
    // inner(u) = int_0^u v q(v) dv + u int_u^1 q(v) dv
    let vq = pint(&pmul(&vec![0.0, 1.0], &q));
    let iq = pint(&q);
    let tail: Poly = {
        let total = peval(&iq, 1.0);
        let mut t: Poly = iq.iter().map(|c| -c).collect();
        t[0] += total;
        pmul(&vec![0.0, 1.0], &t)
    };
    let inner = padd(&vq, &tail);
    moment(&pmul(a, &inner), k)
}

fn solve(m: &Matrix, e: &[f64]) -> Vec<f64> {
    let inv = m.inverse().unwrap_or_else(|_| m.sym_inverse(1e-14).0);
    (0..e.len()).map(|i| (0..e.len()).map(|j| inv[(i, j)] * e[j]).sum()).collect()
}

/// Variance and bias constants of the one-sided estimator of coefficient
/// `s` with a polynomial of order `p`.
fn one_sided(p: usize, s: usize, k: Kernel) -> (f64, f64) {
    let kp = kpoly(k);
    let n = p + 1;
    let mut sm = Matrix::zeros(n, n);
    let mut gm = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sm[(i, j)] = moment(&monomial(i + j), &kp);
            gm[(i, j)] = gmin(&monomial(i), &monomial(j), &kp);
        }
    }
    let e: Vec<f64> = (0..n).map(|i| (i == s) as usize as f64).collect();
    let a = solve(&sm, &e);
    let v: f64 = (0..n).map(|i| (0..n).map(|j| a[i] * gm[(i, j)] * a[j]).sum::<f64>()).sum();
    let b: f64 = (0..n).map(|i| a[i] * moment(&monomial(i + p + 1), &kp)).sum();
    (v, b)
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Squared bias constant of the density estimator (unrestricted model).
fn bsq(p: usize, k: Kernel) -> f64 {
    let (_, b) = one_sided(p, 1, k);
    (b / factorial(p + 1)).powi(2)
}

fn varv0(p: usize, k: Kernel) -> f64 {
    one_sided(p, 1, k).0
}

fn c_c(p: usize) -> f64 {
    varv0(p, Kernel::Uniform) / bsq(p, Kernel::Uniform)
}

fn c_b(p: usize) -> f64 {
    let (v, b) = one_sided(p + 2, p + 1, Kernel::Uniform);
    v / (b / factorial(p + 3)).powi(2)
}

/// Basis of the restricted model on the right side: left slope (0), right
/// slope u, constant, u^2, u^3, ...; `m` entries.
fn restricted_basis(m: usize) -> Vec<Poly> {
    (0..m)
        .map(|i| match i {
            0 => vec![0.0],
            1 => monomial(1),
            2 => monomial(0),
            k => monomial(k - 1),
        })
        .collect()
}

/// (S+, G+, v+) of the restricted model with p+2 coefficients.
fn restricted_consts(p: usize, k: Kernel) -> (Matrix, Matrix, Vec<f64>) {
    let kp = kpoly(k);
    let m = p + 2;
    let basis = restricted_basis(m);
    let mut s = Matrix::zeros(m, m);
    let mut g = Matrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            s[(i, j)] = moment(&pmul(&basis[i], &basis[j]), &kp);
            g[(i, j)] = gmin(&basis[i], &basis[j], &kp);
        }
    }
    let v: Vec<f64> = (0..m).map(|i| moment(&pmul(&basis[i], &monomial(p + 1)), &kp)).collect();
    (s, g, v)
}

/// Reflection to the left side.
fn psi(m: usize) -> Matrix {
    let mut p = Matrix::zeros(m, m);
    p[(0, 1)] = -1.0;
    p[(1, 0)] = -1.0;
    for i in 2..m {
        p[(i, i)] = if i == 2 || (i - 1) % 2 == 0 { 1.0 } else { -1.0 };
    }
    p
}

fn mm(a: &Matrix, b: &Matrix) -> Matrix {
    a.mul(b).unwrap()
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Fit {
    Unrestricted,
    Restricted,
}

#[derive(Clone, Copy, PartialEq)]
enum Vce {
    Jackknife,
    Plugin,
}

/// Rows: left, right, difference, sum. Columns: estimate, variance, and the
/// coefficient used for bias estimation.
type Fv = [[f64; 3]; 4];

/// Tie groups of a sorted vector: (value, frequency, first, last).
fn unique(x: &[f64]) -> Vec<(f64, usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < x.len() {
        let mut j = i;
        while j + 1 < x.len() && x[j + 1] == x[i] {
            j += 1;
        }
        out.push((x[i], j - i + 1, i, j));
        i = j + 1;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn fv(y: &[f64], x: &[f64], n: usize, nlh: usize, hl: f64, hr: f64, p: usize, s: usize, k: Kernel, fit: Fit, vce: Vce, masspoints: bool) -> Fv {
    let nh = x.len();
    let w: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let left = i < nlh;
            let u = if left { xi / hl } else { xi / hr };
            match k {
                Kernel::Triangular => 1.0 - u.abs(),
                Kernel::Epanechnikov => 1.0 - u * u,
                Kernel::Uniform => 1.0,
            }
        })
        .collect();
    let kcols = match fit {
        Fit::Restricted => p + 2,
        Fit::Unrestricted => 2 * (p + 1),
    };
    let mut xp = vec![vec![0.0; kcols]; nh];
    for i in 0..nh {
        let xi = x[i];
        let left = i < nlh;
        match fit {
            Fit::Restricted => {
                if left {
                    xp[i][0] = xi;
                } else {
                    xp[i][1] = xi;
                }
                xp[i][2] = 1.0;
                for j in 2..=p {
                    xp[i][j + 1] = xi.powi(j as i32);
                }
            }
            Fit::Unrestricted => {
                let off = if left { 0 } else { 1 };
                xp[i][off] = xi;
                xp[i][2 + off] = 1.0;
                for j in 2..=p {
                    xp[i][2 * j + off] = xi.powi(j as i32);
                }
            }
        }
    }
    let mut sm = Matrix::zeros(kcols, kcols);
    let mut xy = vec![0.0; kcols];
    for i in 0..nh {
        for a in 0..kcols {
            xy[a] += xp[i][a] * w[i] * y[i];
            for b in 0..kcols {
                sm[(a, b)] += xp[i][a] * w[i] * xp[i][b];
            }
        }
    }
    let (sinv, _) = sm.sym_inverse(1e-14);
    let b: Vec<f64> = (0..kcols).map(|a| (0..kcols).map(|c| sinv[(a, c)] * xy[c]).sum()).collect();
    let mut out: Fv = [[0.0; 3]; 4];
    let est = [b[0], b[1], b[1] - b[0], b[1] + b[0]];
    let bias = match fit {
        Fit::Restricted => {
            let v = b.get(s + 1).copied().unwrap_or(0.0);
            [v, v, 0.0, 2.0 * v]
        }
        Fit::Unrestricted => {
            let (l, r) = (b.get(2 * s).copied().unwrap_or(0.0), b.get(2 * s + 1).copied().unwrap_or(0.0));
            [l, r, r - l, r + l]
        }
    };
    let nf = n as f64;
    let v: Matrix = match (vce, fit) {
        (Vce::Jackknife, _) => {
            let xpw: Vec<Vec<f64>> = (0..nh).map(|i| xp[i].iter().map(|v| v * w[i]).collect()).collect();
            let mut ll = Matrix::zeros(kcols, kcols);
            if masspoints {
                let groups = unique(x);
                let ng = groups.len();
                let sums: Vec<Vec<f64>> = groups.iter().map(|&(_, _, a, z)| (0..kcols).map(|c| (a..=z).map(|i| xpw[i][c]).sum()).collect()).collect();
                // L_g = sum over later groups + (freq - 1) * first row
                let mut acc = vec![0.0; kcols];
                let mut lrows = vec![vec![0.0; kcols]; ng];
                for g in (0..ng).rev() {
                    let (_, f, a, _) = groups[g];
                    for c in 0..kcols {
                        lrows[g][c] = (acc[c] + (f as f64 - 1.0) * xpw[a][c]) / (nf - 1.0);
                    }
                    for c in 0..kcols {
                        acc[c] += sums[g][c];
                    }
                }
                for (g, row) in lrows.iter().enumerate() {
                    let f = groups[g].1 as f64;
                    for a in 0..kcols {
                        for c in 0..kcols {
                            ll[(a, c)] += f * row[a] * row[c];
                        }
                    }
                }
            } else {
                let mut acc = vec![0.0; kcols];
                for i in (0..nh).rev() {
                    let row: Vec<f64> = acc.iter().map(|v| v / (nf - 1.0)).collect();
                    for a in 0..kcols {
                        for c in 0..kcols {
                            ll[(a, c)] += row[a] * row[c];
                        }
                    }
                    for c in 0..kcols {
                        acc[c] += xpw[i][c];
                    }
                }
            }
            let mut v = Matrix::zeros(2, 2);
            for a in 0..2 {
                for c in 0..2 {
                    v[(a, c)] = (0..kcols).map(|i| (0..kcols).map(|j| sinv[(a, i)] * ll[(i, j)] * sinv[(j, c)]).sum::<f64>()).sum();
                }
            }
            v
        }
        (Vce::Plugin, Fit::Unrestricted) => {
            let c = varv0(p, k);
            Matrix::from_diag(&[est[0] * c / (nf * hl), est[1] * c / (nf * hr)])
        }
        (Vce::Plugin, Fit::Restricted) => {
            let (sp, gp, _) = restricted_consts(p, k);
            let ps = psi(p + 2);
            let sminus = mm(&mm(&ps, &sp), &ps);
            let gminus = mm(&mm(&ps, &gp), &ps);
            let sinv2 = pinv(&sp.scale(est[1]).add(&sminus.scale(est[0])).unwrap());
            let mid = gp.scale(est[1].powi(3)).add(&gminus.scale(est[0].powi(3))).unwrap();
            let full = mm(&mm(&sinv2, &mid), &sinv2).scale(1.0 / (nf * hl));
            let mut v = Matrix::zeros(2, 2);
            for a in 0..2 {
                for c in 0..2 {
                    v[(a, c)] = full[(a, c)];
                }
            }
            v
        }
    };
    let var = [v[(0, 0)], v[(1, 1)], v[(0, 0)] + v[(1, 1)] - 2.0 * v[(0, 1)], v[(0, 0)] + v[(1, 1)] + 2.0 * v[(0, 1)]];
    for r in 0..4 {
        out[r] = [est[r], var[r], bias[r]];
    }
    out
}

fn pinv(m: &Matrix) -> Matrix {
    m.sym_inverse(1e-14).0
}

/// Probabilists' Hermite polynomial He_p(x).
fn hermite(x: f64, p: usize) -> f64 {
    let (mut a, mut b) = (1.0, x);
    if p == 0 {
        return 1.0;
    }
    for k in 1..p {
        let c = x * b - k as f64 * a;
        a = b;
        b = c;
    }
    b
}

/// The running variable minus the cutoff, sorted, with side counts.
struct DensData {
    x: Vec<f64>,
    nl: usize,
    nr: usize,
    name: String,
    c: f64,
    rows: Vec<usize>,
}

struct Common {
    p: usize,
    kernel: Kernel,
    fit: Fit,
    vce: Vce,
    regularize: bool,
    masspoints: bool,
    nlocalmin: i64,
    nuniquemin: i64,
}

fn parse_common(cmd: &Command) -> Result<Common, String> {
    let p = util::opt_num(cmd, "p", 1)?.unwrap_or(2.0);
    if !(1.0..=7.0).contains(&p) || p.fract() != 0.0 {
        return Err(rc_err(125, "p() should be an integer value less or equal than 7."));
    }
    let kernel = match cmd.opt("kernel", 3).map(|s| s.trim().to_lowercase()).unwrap_or_default().as_str() {
        "" | "triangular" => Kernel::Triangular,
        "uniform" => Kernel::Uniform,
        "epanechnikov" => Kernel::Epanechnikov,
        _ => return Err(rc_err(7, "kernel() incorrectly specified.")),
    };
    let fit = match cmd.opt("fitselect", 3).map(|s| s.trim().to_lowercase()).unwrap_or_default().as_str() {
        "" | "unrestricted" => Fit::Unrestricted,
        "restricted" => Fit::Restricted,
        _ => return Err(rc_err(7, "fitselect() incorrectly specified.")),
    };
    let vce = match cmd.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_default().as_str() {
        "" | "jackknife" => Vce::Jackknife,
        "plugin" => Vce::Plugin,
        _ => return Err(rc_err(7, "vce() incorrectly specified.")),
    };
    let p = p as usize;
    let nl = util::opt_num(cmd, "nlocalmin", 4)?.map(|v| v as i64).unwrap_or(-1);
    let nu = util::opt_num(cmd, "nuniquemin", 4)?.map(|v| v as i64).unwrap_or(-1);
    Ok(Common {
        p,
        kernel,
        fit,
        vce,
        regularize: !cmd.has_opt("noregularize", 5),
        masspoints: !cmd.has_opt("nomasspoints", 6),
        nlocalmin: if nl < 0 { 20 + p as i64 + 1 } else { nl },
        nuniquemin: if nu < 0 { 20 + p as i64 + 1 } else { nu },
    })
}

fn prepare(ds: &DataSet, st: &Store, cmd: &Command) -> Result<DensData, String> {
    let vars = crate::varlist::expand(ds, &cmd.main)?;
    if vars.len() != 1 {
        return Err(rc_err(if vars.is_empty() { 102 } else { 103 }, if vars.is_empty() { "too few variables specified" } else { "too many variables specified" }));
    }
    let c = util::opt_num(cmd, "c", 1)?.unwrap_or(0.0);
    let mut mask = util::touse(ds, st, cmd)?;
    util::markout(ds, &mut mask, &vars);
    let xv = util::col(ds, &vars[0])?;
    let rows = util::rows(&mask);
    let mut x: Vec<f64> = rows.iter().map(|&i| xv[i] - c).collect();
    x.sort_by(|a, b| a.total_cmp(b));
    if x.is_empty() || c <= x[0] + c || c >= x[x.len() - 1] + c {
        return Err(rc_err(125, format!("c() should be set within the range of {}.", vars[0])));
    }
    let nl = x.iter().filter(|v| **v < 0.0).count();
    let nr = x.len() - nl;
    if nl < 10 || nr < 10 {
        return Err(rc_err(2001, "Not enough observations to perform calculations."));
    }
    Ok(DensData { x, nl, nr, name: vars[0].clone(), c, rows })
}

/// Empirical CDF values used as the regression outcome.
fn cdf(x: &[f64], masspoints: bool) -> (Vec<f64>, bool) {
    let n = x.len();
    let mut y: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let repeated = x.windows(2).any(|w| w[0] == w[1]);
    let flag = repeated && masspoints;
    if flag {
        for (_, _, a, z) in unique(x) {
            let v = y[z];
            for yi in y.iter_mut().take(z + 1).skip(a) {
                *yi = v;
            }
        }
    }
    (y, flag)
}

/// Window [-hl, hr]: (y, x, number on the left).
fn window(y: &[f64], x: &[f64], hl: f64, hr: f64) -> (Vec<f64>, Vec<f64>, usize) {
    let idx: Vec<usize> = (0..x.len()).filter(|&i| x[i] >= -hl && x[i] <= hr).collect();
    let nl = idx.iter().filter(|&&i| x[i] < 0.0).count();
    (idx.iter().map(|&i| y[i]).collect(), idx.iter().map(|&i| x[i]).collect(), nl)
}

/// Bandwidths (rows: left, right, difference, sum; columns: bandwidth,
/// variance, squared bias), and the two preliminary bandwidths.
fn bandwidths(d: &DensData, o: &Common) -> ([[f64; 3]; 4], f64, f64, bool) {
    let x = &d.x;
    let n = x.len();
    let p = o.p;
    let (y, flag) = cdf(x, o.masspoints);
    let groups = unique(x);
    let xu: Vec<f64> = groups.iter().map(|g| g.0).collect();
    let nlu = xu.iter().filter(|v| **v < 0.0).count();
    let x_left_abs: Vec<f64> = x[..d.nl].iter().rev().map(|v| v.abs()).collect();
    let x_right: Vec<f64> = x[d.nl..].to_vec();
    let xu_left_abs: Vec<f64> = xu[..nlu].iter().rev().map(|v| v.abs()).collect();
    let xu_right: Vec<f64> = xu[nlu..].to_vec();
    let at = |v: &[f64], k: usize| -> f64 { v[k.min(v.len()).max(1) - 1] };
    let mu = util::mean(x);
    let sd = util::variance(x).sqrt();
    let z = -mu / sd;
    let pf = p as f64;
    let fhatb = sd.powf(2.0 * pf + 5.0) * dist::normalden(z) / (hermite(z, p + 2) * dist::normalden(z)).powi(2);
    let mut b = ((2.0 * pf + 1.0) / 4.0 * fhatb * c_b(p) / n as f64).powf(1.0 / (2.0 * pf + 5.0));
    let fhatc = sd.powf(2.0 * pf + 1.0) * dist::normalden(z) / (hermite(z, p) * dist::normalden(z)).powi(2);
    let mut c = (1.0 / (2.0 * pf) * fhatc * c_c(p) / n as f64).powf(1.0 / (2.0 * pf + 1.0));
    if o.regularize {
        let maxabs = xu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        b = b.min(maxabs);
        c = c.min(maxabs);
        if o.nlocalmin > 0 {
            b = b.max(at(&x_left_abs, 20 + p + 2 + 1)).max(at(&x_right, 20 + p + 2 + 1));
            c = c.max(at(&x_left_abs, 20 + p + 1)).max(at(&x_right, 20 + p + 1));
        }
        if o.nuniquemin > 0 {
            b = b.max(at(&xu_left_abs, 20 + p + 2 + 1)).max(at(&xu_right, 20 + p + 2 + 1));
            c = c.max(at(&xu_left_abs, 20 + p + 1)).max(at(&xu_right, 20 + p + 1));
        }
    }
    let (yb, xb, nlb) = window(&y, x, b, b);
    let (yc, xc, nlc) = window(&y, x, c, c);
    let fvb = fv(&yb, &xb, n, nlb, b, b, p + 2, p + 1, o.kernel, o.fit, o.vce, flag);
    let fvc = fv(&yc, &xc, n, nlc, c, c, p, 1, o.kernel, o.fit, o.vce, flag);
    let mut h = [[0.0; 3]; 4];
    for r in 0..4 {
        h[r][1] = n as f64 * c * fvc[r][1];
    }
    match o.fit {
        Fit::Unrestricted => {
            let bs = bsq(p, o.kernel).sqrt() * factorial(p + 1);
            h[0][2] = fvb[0][2] * bs * if p % 2 == 1 { -1.0 } else { 1.0 };
            h[1][2] = fvb[1][2] * bs;
        }
        Fit::Restricted => {
            let (sp, _, vplus) = restricted_consts(p, o.kernel);
            let ps = psi(p + 2);
            let sminus = mm(&mm(&ps, &sp), &ps);
            let vminus: Vec<f64> = (0..p + 2).map(|i| (0..p + 2).map(|j| ps[(i, j)] * vplus[j]).sum()).collect();
            let s = pinv(&sp.scale(fvc[1][0]).add(&sminus.scale(fvc[0][0])).unwrap());
            let sign = if (p + 1) % 2 == 1 { -1.0 } else { 1.0 };
            let comb: Vec<f64> = (0..p + 2).map(|j| fvc[0][0] * sign * vminus[j] + fvc[1][0] * vplus[j]).collect();
            for r in 0..2 {
                h[r][2] = fvb[0][2] * (0..p + 2).map(|j| s[(r, j)] * comb[j]).sum::<f64>();
            }
        }
    }
    h[2][2] = h[1][2] - h[0][2];
    h[3][2] = h[1][2] + h[0][2];
    for r in h.iter_mut() {
        r[2] = r[2].powi(2);
        r[0] = ((1.0 / (2.0 * pf)) * (r[1] / r[2]) / n as f64).powf(1.0 / (2.0 * pf + 1.0));
    }
    if o.regularize {
        for r in h.iter_mut() {
            if r[1] < 0.0 {
                r[0] = 0.0;
                r[1] = SYSMISS;
            }
            if !r[0].is_finite() {
                r[0] = 0.0;
            }
        }
        let (lmost, rmost) = (xu[0].abs(), xu[xu.len() - 1]);
        h[0][0] = h[0][0].min(lmost);
        h[1][0] = h[1][0].min(rmost);
        h[2][0] = h[2][0].min(lmost.max(rmost));
        h[3][0] = h[3][0].min(lmost.max(rmost));
        let mut apply = |lmin: f64, rmin: f64| {
            h[0][0] = h[0][0].max(lmin);
            h[1][0] = h[1][0].max(rmin);
            h[2][0] = h[2][0].max(lmin).max(rmin);
            h[3][0] = h[3][0].max(lmin).max(rmin);
        };
        if o.nlocalmin > 0 {
            apply(at(&x_left_abs, o.nlocalmin as usize), at(&x_right, o.nlocalmin as usize));
        }
        if o.nuniquemin > 0 {
            apply(at(&xu_left_abs, o.nuniquemin as usize), at(&xu_right, o.nuniquemin as usize));
        }
    }
    (h, b, c, flag)
}

fn kernel_name(k: Kernel) -> &'static str {
    match k {
        Kernel::Triangular => "triangular",
        Kernel::Uniform => "uniform",
        Kernel::Epanechnikov => "epanechnikov",
    }
}

fn fit_name(f: Fit) -> &'static str {
    match f {
        Fit::Unrestricted => "unrestricted",
        Fit::Restricted => "restricted",
    }
}

fn vce_name(v: Vce) -> &'static str {
    match v {
        Vce::Jackknife => "jackknife",
        Vce::Plugin => "plugin",
    }
}

fn masspoint_note(out: &mut Vec<String>) {
    out.push(String::new());
    out.push("Point estimates and standard errors have been adjusted for repeated observations.".into());
    out.push("(Use option nomasspoints to suppress this adjustment.)".into());
}

// ---------------------------------------------------------------------------
// rdbwdensity
// ---------------------------------------------------------------------------

pub struct Rdbwdensity;

impl CommandHandler for Rdbwdensity {
    fn name(&self) -> &'static str {
        "rdbwdensity"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let o = parse_common(cmd)?;
        let d = prepare(ds, st, cmd)?;
        let (h, b, c, flag) = bandwidths(&d, &o);
        let mut out = Vec::new();
        if flag {
            masspoint_note(&mut out);
        }
        out.push(String::new());
        out.push("Bandwidth selection for manipulation testing.".into());
        out.push(String::new());
        let x = &d.x;
        let (xl_min, xl_max) = (x[0] + d.c, x[d.nl - 1] + d.c);
        let (xr_min, xr_max) = (x[d.nl] + d.c, x[x.len() - 1] + d.c);
        let two = |label: &str, a: String, b: String| -> Ln { Ln::new().t(&ralign(label, 21)).col(22).t(" | ").col(23).t(&a).col(37).t(&b) };
        out.push(Ln::new().t("Cutoff c = ").t(&ff(d.c, 10, 3)).col(22).t(" | ").col(23).t("Left of c").col(36).t("Right of c").col(58).t("Number of obs = ").t(&ff(x.len() as f64, 12, 0)).s());
        out.push(Ln::new().t(&hline(22)).t("+").t(&hline(22)).col(58).t("Model         = ").t(&ralign(fit_name(o.fit), 12)).s());
        out.push(two("Number of obs", ff(d.nl as f64, 9, 0), ff(d.nr as f64, 9, 0)).col(58).t("Kernel        = ").t(&ralign(kernel_name(o.kernel), 12)).s());
        out.push(two("Min Running var.", ff(xl_min, 9, 3), ff(xr_min, 9, 3)).col(58).t("VCE method    = ").t(&ralign(vce_name(o.vce), 12)).s());
        out.push(two("Max Running var.", ff(xl_max, 9, 3), ff(xr_max, 9, 3)).s());
        out.push(two("Order loc. poly. (p)", ff(o.p as f64, 9, 0), ff(o.p as f64, 9, 0)).s());
        out.push(String::new());
        out.push(format!("Running variable: {}.", d.name));
        out.push(format!("{}+{}", hline(22), hline(34)));
        out.push(Ln::new().t(&ralign("Target", 21)).col(22).t(" | ").col(23).t("Bandwidth").col(37).t(" Variance").col(49).t("   Bias^2").s());
        out.push(format!("{}+{}", hline(22), hline(34)));
        for (r, lab) in ["left density", "right density", "difference densities", "sum densities"].iter().enumerate() {
            out.push(Ln::new().t(&ralign(lab, 21)).col(22).t(" | ").col(23).t(&ff(h[r][0], 9, 3)).col(37).t(&ff(h[r][1], 9, 3)).col(49).t(&ff(h[r][2], 9, 3)).s());
        }
        out.push(format!("{}+{}", hline(22), hline(34)));
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("c", d.c);
        e.set("p", o.p as f64);
        e.set("N_l", d.nl as f64);
        e.set("N_r", d.nr as f64);
        let mut hm = Matrix::zeros(4, 3);
        for r in 0..4 {
            for k in 0..3 {
                hm[(r, k)] = h[r][k];
            }
        }
        e.set_mat("h", hm.with_names(vec!["f_left".into(), "f_right".into(), "f_diff".into(), "f_sum".into()], vec!["bandwidth".into(), "var".into(), "bias2".into()]));
        e.set("BW_b", b);
        e.set("BW_c", c);
        e.set_str("runningvar", d.name.clone());
        e.set_str("kernel", kernel_name(o.kernel));
        e.set_str("fitmethod", fit_name(o.fit));
        e.set_str("vce", vce_name(o.vce));
        e.set_str("cmd", "rdbwdensity");
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// rddensity
// ---------------------------------------------------------------------------

pub struct Rddensity;

/// Two-sided exact binomial p-value (outcomes no more likely than k).
fn binom_two_sided(n: usize, k: usize, p: f64) -> f64 {
    let pk = dist::binomialp(n as f64, k as f64, p);
    let s: f64 = (0..=n).map(|j| dist::binomialp(n as f64, j as f64, p)).filter(|&q| q <= pk * (1.0 + 1e-7)).sum();
    s.min(1.0)
}

impl CommandHandler for Rddensity {
    fn name(&self) -> &'static str {
        "rddensity"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let o = parse_common(cmd)?;
        let q = util::opt_num(cmd, "q", 1)?.unwrap_or(0.0);
        let q = if q == 0.0 { o.p + 1 } else { q as usize };
        if o.p > q {
            return Err(rc_err(125, "p() should be an integer value no larger than q()."));
        }
        let bwselect = cmd.opt("bwselect", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "comb".into());
        if !["each", "diff", "sum", "comb"].contains(&bwselect.as_str()) {
            return Err(rc_err(7, "bwselect() incorrectly specified."));
        }
        if o.fit == Fit::Restricted && bwselect == "each" {
            return Err(rc_err(7, "bwselect(each) is not available in the restricted model."));
        }
        let all = cmd.has_opt("all", 3);
        let d = prepare(ds, st, cmd)?;
        let (mut hl, mut hr) = (0.0, 0.0);
        if let Some(t) = cmd.opt("h", 1) {
            let v: Vec<f64> = t.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "h() invalid"))).collect::<Result<_, _>>()?;
            match v.len() {
                0 => {}
                1 => {
                    hl = v[0];
                    hr = v[0];
                }
                2 => {
                    hl = v[0];
                    hr = v[1];
                }
                _ => return Err(rc_err(125, "h() only accepts two inputs.")),
            }
        }
        if hl < 0.0 || hr < 0.0 {
            return Err(rc_err(411, "hl() must be a positive real number."));
        }
        if o.fit == Fit::Restricted && hl != hr {
            return Err(rc_err(7, "hl() and hr() must be equal in the restricted model."));
        }
        let mut out: Vec<String> = Vec::new();
        let mut bwmethod = "manual".to_string();
        if hl == 0.0 || hr == 0.0 {
            bwmethod = bwselect.clone();
            out.push("Computing data-driven bandwidth selectors.".into());
            let (h, _, _, _) = bandwidths(&d, &o);
            let med = |a: f64, b: f64, c: f64| a + b + c - a.min(b).min(c) - a.max(b).max(c);
            let (l, r) = match (o.fit, bwselect.as_str()) {
                (Fit::Unrestricted, "each") => (h[0][0], h[1][0]),
                (_, "diff") => (h[2][0], h[2][0]),
                (_, "sum") => (h[3][0], h[3][0]),
                (Fit::Unrestricted, _) => (med(h[0][0], h[2][0], h[3][0]), med(h[1][0], h[2][0], h[3][0])),
                (Fit::Restricted, _) => (h[2][0].min(h[3][0]), h[2][0].min(h[3][0])),
            };
            if hl == 0.0 {
                hl = l;
            }
            if hr == 0.0 {
                hr = r;
            }
        }
        let (y, flag) = cdf(&d.x, o.masspoints);
        let (yw, xw, nlh) = window(&y, &d.x, hl, hr);
        let nrh = xw.len() - nlh;
        if nlh < 5 {
            return Err(rc_err(1, "Not enough observations on the left to perform calculations."));
        }
        if nrh < 5 {
            return Err(rc_err(1, "Not enough observations on the right to perform calculations."));
        }
        let n = d.x.len();
        let fq = fv(&yw, &xw, n, nlh, hl, hr, q, 1, o.kernel, o.fit, o.vce, flag);
        let t_q = fq[2][0] / fq[2][1].sqrt();
        let pv = |t: f64| 2.0 * dist::normal(-t.abs());
        let fp = if all { Some(fv(&yw, &xw, n, nlh, hl, hr, o.p, 1, o.kernel, o.fit, o.vce, flag)) } else { None };
        // ---- output ---------------------------------------------------------
        let (x_min, x_max) = (d.x[0], d.x[n - 1]);
        if hl > -x_min {
            out.push(String::new());
            out.push("Bandwidth hl greater than the range of the data.".into());
        }
        if hr > x_max {
            out.push(String::new());
            out.push("Bandwidth hr greater than the range of the data.".into());
        }
        if nlh < 20 || nrh < 20 {
            out.push("Bandwidth h may be too small.".into());
        }
        if flag {
            masspoint_note(&mut out);
        }
        out.push(String::new());
        out.push("RD Manipulation test using local polynomial density estimation.".into());
        out.push(String::new());
        let two = |label: &str, a: String, b: String| -> Ln { Ln::new().t(&ralign(label, 18)).col(19).t(" | ").col(21).t(&a).col(34).t(&b) };
        out.push(Ln::new().t(&ralign(" c = ", 9)).t(&ff(d.c, 9, 3)).col(19).t(" |").col(22).t("Left of c").col(33).t("Right of c").col(53).t("Number of obs = ").t(&ff(n as f64, 12, 0)).s());
        out.push(Ln::new().t(&hline(19)).t("+").t(&hline(22)).col(53).t("Model         = ").t(&ralign(fit_name(o.fit), 12)).s());
        out.push(two("Number of obs", ff(d.nl as f64, 9, 0), ff(d.nr as f64, 9, 0)).col(53).t("BW method     = ").t(&ralign(&bwmethod, 12)).s());
        out.push(two("Eff. Number of obs", ff(nlh as f64, 9, 0), ff(nrh as f64, 9, 0)).col(53).t("Kernel        = ").t(&ralign(kernel_name(o.kernel), 12)).s());
        out.push(two("Order est. (p)", ff(o.p as f64, 9, 0), ff(o.p as f64, 9, 0)).col(53).t("VCE method    = ").t(&ralign(vce_name(o.vce), 12)).s());
        out.push(two("Order bias (q)", ff(q as f64, 9, 0), ff(q as f64, 9, 0)).s());
        out.push(two("BW est. (h)", ff(hl, 9, 3), ff(hr, 9, 3)).s());
        out.push(String::new());
        out.push(format!("Running variable: {}.", d.name));
        out.push(format!("{}+{}", hline(19), hline(22)));
        out.push(Ln::new().t(&ralign("Method", 18)).col(19).t(" | ").col(23).t("    T").col(38).t("P>|T|").s());
        out.push(format!("{}+{}", hline(19), hline(22)));
        let trow = |name: &str, t: f64| Ln::new().t(&ralign(name, 18)).col(19).t(" | ").col(21).t(&ff(t, 9, 4)).col(34).t(&ff(pv(t), 9, 4)).s();
        let t_p = fp.map(|f| f[2][0] / f[2][1].sqrt());
        if q > o.p {
            if let Some(tp) = t_p {
                out.push(trow("Conventional", tp));
            }
            out.push(trow("Robust", t_q));
        } else {
            out.push(trow("Conventional", t_q));
        }
        out.push(format!("{}+{}", hline(19), hline(22)));
        // ---- binomial tests -------------------------------------------------
        let nobino = cmd.has_opt("nobinomial", 6);
        let mut bino_rows: Vec<(f64, f64, usize, usize, f64)> = Vec::new();
        if !nobino {
            let bino_n = util::opt_num(cmd, "bino_n", 6)?.unwrap_or(20.0).max(1.0) as usize;
            let nw = util::opt_num(cmd, "bino_nw", 7)?.unwrap_or(10.0).max(1.0) as usize;
            let bp = util::opt_num(cmd, "bino_p", 6)?.unwrap_or(0.5);
            let mut xl: Vec<f64> = d.x[..d.nl].iter().map(|v| v.abs()).collect();
            xl.sort_by(|a, b| a.total_cmp(b));
            let xr: Vec<f64> = d.x[d.nl..].to_vec();
            let pairv = |name: &str, min: usize| -> Result<Option<(f64, f64)>, String> {
                match cmd.opt(name, min) {
                    None => Ok(None),
                    Some(t) => {
                        let v: Vec<f64> = t.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, format!("{name}() invalid")))).collect::<Result<_, _>>()?;
                        match v.len() {
                            1 => Ok(Some((v[0], v[0]))),
                            2 => Ok(Some((v[0], v[1]))),
                            _ => Err(rc_err(125, format!("{name}() takes at most two inputs."))),
                        }
                    }
                }
            };
            let (w0l, w0r) = match pairv("bino_w", 6)? {
                Some(w) => w,
                None => {
                    let w = xl[bino_n.min(d.nl) - 1].max(xr[bino_n.min(d.nr) - 1]);
                    (w, w)
                }
            };
            let mut wl = vec![w0l];
            let mut wr = vec![w0r];
            if nw > 1 {
                if let Some((sl, sr)) = pairv("bino_wstep", 7)? {
                    for k in 1..nw {
                        wl.push(w0l + k as f64 * sl);
                        wr.push(w0r + k as f64 * sr);
                    }
                } else if !(w0l >= hl || w0r >= hr) {
                    for k in 1..nw {
                        let kf = k as f64;
                        wl.push(if w0l * nw as f64 > hl { w0l + kf * (hl - w0l) / (nw as f64 - 1.0) } else { w0l + kf * w0l });
                        wr.push(if w0r * nw as f64 > hr { w0r + kf * (hr - w0r) / (nw as f64 - 1.0) } else { w0r + kf * w0r });
                    }
                }
            }
            for (a, b) in wl.iter().zip(&wr) {
                let nlw = xl.iter().filter(|v| **v <= *a).count();
                let nrw = xr.iter().filter(|v| **v <= *b).count();
                let pval = if nlw + nrw > 0 { binom_two_sided(nlw + nrw, nlw, bp) } else { 1.0 };
                bino_rows.push((*a, *b, nlw, nrw, pval));
            }
            let equal = bino_rows.iter().all(|r| r.0 == r.1);
            out.push(String::new());
            out.push(format!("P-values of binomial tests. (H0: prob = {})", crate::format::fmt_g(bp, 9).trim()));
            out.push(format!("{}+{}+{}", hline(19), hline(22), hline(10)));
            let head = if equal { " Window Length / 2" } else { " Window Length" };
            out.push(Ln::new().t(&ralign(head, 18)).col(20).t("|").t(&ralign(" <c", 9)).col(33).t(&ralign(" >=c", 9)).col(43).t("|").col(49).t("P>|T|").s());
            out.push(format!("{}+{}+{}", hline(19), hline(22), hline(10)));
            for r in &bino_rows {
                let lead = if equal { Ln::new().col(10).t(&ff(r.0, 9, 3)) } else { Ln::new().t(&ff(r.0, 8, 3)).col(10).t("+").t(&ff(r.1, 8, 3)) };
                out.push(lead.col(20).t("|").t(&ff(r.2 as f64, 9, 0)).col(33).t(&ff(r.3 as f64, 9, 0)).col(43).t("|").col(45).t(&ff(r.4, 9, 4)).s());
            }
            out.push(format!("{}+{}+{}", hline(19), hline(22), hline(10)));
        }
        if cmd.has_opt("plot", 2) {
            out.push("(note: the density plot is not drawn; use the estimates in e())".into());
        }
        let text = out.join("\n");
        let mut mask = vec![false; ds.n_obs];
        for &i in &d.rows {
            mask[i] = true;
        }
        let e = st.begin_e();
        e.sample = Some(mask);
        e.set("c", d.c);
        e.set("p", o.p as f64);
        e.set("q", q as f64);
        e.set("N_l", d.nl as f64);
        e.set("N_r", d.nr as f64);
        e.set("N_h_l", nlh as f64);
        e.set("N_h_r", nrh as f64);
        e.set("h_l", hl);
        e.set("h_r", hr);
        e.set("f_ql", fq[0][0]);
        e.set("f_qr", fq[1][0]);
        e.set("se_ql", fq[0][1].sqrt());
        e.set("se_qr", fq[1][1].sqrt());
        e.set("se_q", fq[2][1].sqrt());
        e.set("pv_q", pv(t_q));
        e.set("T_q", t_q);
        if let (Some(f), Some(tp)) = (fp, t_p) {
            e.set("f_pl", f[0][0]);
            e.set("f_pr", f[1][0]);
            e.set("se_pl", f[0][1].sqrt());
            e.set("se_pr", f[1][1].sqrt());
            e.set("se_p", f[2][1].sqrt());
            e.set("pv_p", pv(tp));
            e.set("T_p", tp);
        }
        if !bino_rows.is_empty() {
            let mut bm = Matrix::zeros(bino_rows.len(), 5);
            for (i, r) in bino_rows.iter().enumerate() {
                bm[(i, 0)] = r.0;
                bm[(i, 1)] = r.1;
                bm[(i, 2)] = r.2 as f64;
                bm[(i, 3)] = r.3 as f64;
                bm[(i, 4)] = r.4;
            }
            e.set_mat("binomial", bm.with_names((1..=bino_rows.len()).map(|i| format!("r{i}")).collect(), vec!["w_left".into(), "w_right".into(), "N_left".into(), "N_right".into(), "p".into()]));
        }
        e.set_str("runningvar", d.name.clone());
        e.set_str("kernel", kernel_name(o.kernel));
        e.set_str("bwmethod", bwmethod);
        e.set_str("vce", vce_name(o.vce));
        e.set_str("cmd", "rddensity");
        Ok(text)
    }
}
