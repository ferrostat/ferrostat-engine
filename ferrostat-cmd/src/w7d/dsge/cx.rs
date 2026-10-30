//! Small complex linear algebra for solving linear rational-expectations
//! models: an ordered complex Schur decomposition (Householder reduction to
//! Hessenberg form, shifted QR iterations with Givens rotations, and
//! reordering by adjacent swaps) and the stable-subspace solution of the
//! pencil `A E x(t+1) = B x(t)`.
//!
//! The pencil is mapped to an ordinary eigenproblem with a Moebius
//! transform: for a scalar `mu` that is not a generalized eigenvalue,
//! `N = (B - mu A)^{-1} A` has eigenvalue `nu = 1/(lambda - mu)` for each
//! generalized eigenvalue `lambda` (infinite ones map to zero), with the
//! same eigenvectors. Ordering the Schur form of `N` so that the stable
//! `lambda` come first gives an orthonormal basis of the stable deflating
//! subspace, from which the policy and transition matrices follow.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

pub const fn c(re: f64, im: f64) -> C {
    C { re, im }
}

impl C {
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn conj(self) -> C {
        c(self.re, -self.im)
    }
    pub fn scale(self, s: f64) -> C {
        c(self.re * s, self.im * s)
    }
    pub fn sqrt(self) -> C {
        let r = self.abs();
        if r == 0.0 {
            return C::default();
        }
        let a = ((r + self.re) / 2.0).max(0.0).sqrt();
        let b = ((r - self.re) / 2.0).max(0.0).sqrt();
        c(a, if self.im < 0.0 { -b } else { b })
    }
    pub fn recip(self) -> C {
        let d = self.re * self.re + self.im * self.im;
        c(self.re / d, -self.im / d)
    }
}

impl Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        c(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        c(self.re - o.re, self.im - o.im)
    }
}
impl Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        c(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Div for C {
    type Output = C;
    fn div(self, o: C) -> C {
        self * o.recip()
    }
}
impl Neg for C {
    type Output = C;
    fn neg(self) -> C {
        c(-self.re, -self.im)
    }
}

pub type CMat = Vec<Vec<C>>;

pub fn czeros(r: usize, k: usize) -> CMat {
    vec![vec![C::default(); k]; r]
}

pub fn from_real(a: &[Vec<f64>]) -> CMat {
    a.iter().map(|r| r.iter().map(|&v| c(v, 0.0)).collect()).collect()
}

pub fn cmul(a: &CMat, b: &CMat) -> CMat {
    let (n, m, p) = (a.len(), b.len(), b.first().map_or(0, |r| r.len()));
    let mut o = czeros(n, p);
    for i in 0..n {
        for l in 0..m {
            let x = a[i][l];
            if x.re == 0.0 && x.im == 0.0 {
                continue;
            }
            for j in 0..p {
                o[i][j] = o[i][j] + x * b[l][j];
            }
        }
    }
    o
}

/// Solve `A X = B` by Gaussian elimination with partial pivoting.
pub fn csolve(a: &CMat, b: &CMat) -> Option<CMat> {
    let n = a.len();
    let p = b.first().map_or(0, |r| r.len());
    let mut m = a.clone();
    let mut x = b.clone();
    let scale = a.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    for k in 0..n {
        let piv = (k..n).max_by(|&i, &j| m[i][k].abs().total_cmp(&m[j][k].abs()))?;
        if m[piv][k].abs() <= 1e-13 * scale {
            return None;
        }
        m.swap(k, piv);
        x.swap(k, piv);
        for i in k + 1..n {
            let f = m[i][k] / m[k][k];
            if f.abs() == 0.0 {
                continue;
            }
            for j in k..n {
                let v = m[k][j];
                m[i][j] = m[i][j] - f * v;
            }
            for j in 0..p {
                let v = x[k][j];
                x[i][j] = x[i][j] - f * v;
            }
        }
    }
    for k in (0..n).rev() {
        for j in 0..p {
            let mut s = x[k][j];
            for l in k + 1..n {
                s = s - m[k][l] * x[l][j];
            }
            x[k][j] = s / m[k][k];
        }
    }
    Some(x)
}

/// Least squares `min |A X - B|` through the normal equations.
pub fn clstsq(a: &CMat, b: &CMat) -> Option<CMat> {
    let ah: CMat = (0..a.first().map_or(0, |r| r.len())).map(|j| a.iter().map(|r| r[j].conj()).collect()).collect();
    csolve(&cmul(&ah, a), &cmul(&ah, b))
}

/// Givens rotation (c real, s complex) with `G [a; b] = [r; 0]` for
/// `G = [[c, s], [-conj(s), c]]`.
fn givens(a: C, b: C) -> (f64, C) {
    let na = a.abs();
    let nb = b.abs();
    if nb == 0.0 {
        return (1.0, C::default());
    }
    if na == 0.0 {
        return (0.0, c(1.0, 0.0));
    }
    let norm = na.hypot(nb);
    let ph = a.scale(1.0 / na);
    (na / norm, (ph * b.conj()).scale(1.0 / norm))
}

/// Rows k, k+1 <- G * rows over columns `cols`.
fn rot_rows(t: &mut CMat, k: usize, cs: f64, s: C, cols: std::ops::Range<usize>) {
    for j in cols {
        let (x, y) = (t[k][j], t[k + 1][j]);
        t[k][j] = x.scale(cs) + s * y;
        t[k + 1][j] = -(s.conj() * x) + y.scale(cs);
    }
}

/// Columns k, k+1 <- columns * G^H over rows `rows`.
fn rot_cols(t: &mut CMat, k: usize, cs: f64, s: C, rows: std::ops::Range<usize>) {
    for i in rows {
        let (x, y) = (t[i][k], t[i][k + 1]);
        t[i][k] = x.scale(cs) + y * s.conj();
        t[i][k + 1] = -(x * s) + y.scale(cs);
    }
}

/// Complex Schur decomposition `A = U T U^H` (T upper triangular).
pub fn schur(a: &CMat) -> Option<(CMat, CMat)> {
    let n = a.len();
    let mut t = a.clone();
    let mut u = czeros(n, n);
    for (i, r) in u.iter_mut().enumerate() {
        r[i] = c(1.0, 0.0);
    }
    // Householder reduction to upper Hessenberg form
    for k in 0..n.saturating_sub(2) {
        let x: Vec<C> = (k + 1..n).map(|i| t[i][k]).collect();
        let nx = x.iter().map(|v| v.abs().powi(2)).sum::<f64>().sqrt();
        if nx == 0.0 {
            continue;
        }
        let ph = if x[0].abs() > 0.0 { x[0].scale(1.0 / x[0].abs()) } else { c(1.0, 0.0) };
        let mut v = x.clone();
        v[0] = v[0] + ph.scale(nx);
        let nv = v.iter().map(|z| z.abs().powi(2)).sum::<f64>().sqrt();
        if nv == 0.0 {
            continue;
        }
        for z in v.iter_mut() {
            *z = z.scale(1.0 / nv);
        }
        // T <- (I - 2vv^H) T (rows k+1..)
        for j in 0..n {
            let mut s = C::default();
            for (l, vl) in v.iter().enumerate() {
                s = s + vl.conj() * t[k + 1 + l][j];
            }
            for (l, vl) in v.iter().enumerate() {
                t[k + 1 + l][j] = t[k + 1 + l][j] - (*vl * s).scale(2.0);
            }
        }
        // T <- T (I - 2vv^H), U <- U (I - 2vv^H)
        for m in [&mut t, &mut u] {
            for row in m.iter_mut() {
                let mut s = C::default();
                for (l, vl) in v.iter().enumerate() {
                    s = s + row[k + 1 + l] * *vl;
                }
                for (l, vl) in v.iter().enumerate() {
                    row[k + 1 + l] = row[k + 1 + l] - (s * vl.conj()).scale(2.0);
                }
            }
        }
        for i in k + 2..n {
            t[i][k] = C::default();
        }
    }
    let norm = t.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    let eps = f64::EPSILON;
    let mut hi = n;
    let mut iter = 0usize;
    while hi > 1 {
        let h = hi - 1;
        // find the active window [lo, h]
        let mut lo = h;
        while lo > 0 {
            let s = t[lo - 1][lo - 1].abs() + t[lo][lo].abs();
            if t[lo][lo - 1].abs() <= eps * if s == 0.0 { norm } else { s } {
                t[lo][lo - 1] = C::default();
                break;
            }
            lo -= 1;
        }
        if lo == h {
            hi -= 1;
            iter = 0;
            continue;
        }
        iter += 1;
        if iter > 300 {
            return None;
        }
        // Wilkinson shift from the trailing 2x2 block
        let (a11, a12, a21, a22) = (t[h - 1][h - 1], t[h - 1][h], t[h][h - 1], t[h][h]);
        let mu = if iter % 11 == 10 {
            a22 + c(t[h][h - 1].abs() * 0.75, t[h][h - 1].abs() * 0.3)
        } else {
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = (tr * tr - det.scale(4.0)).sqrt();
            let l1 = (tr + disc).scale(0.5);
            let l2 = (tr - disc).scale(0.5);
            if (l1 - a22).abs() < (l2 - a22).abs() { l1 } else { l2 }
        };
        // explicit shifted QR step on the window
        for i in lo..=h {
            t[i][i] = t[i][i] - mu;
        }
        let mut rots = Vec::with_capacity(h - lo);
        for k in lo..h {
            let (cs, s) = givens(t[k][k], t[k + 1][k]);
            rot_rows(&mut t, k, cs, s, k..n);
            rots.push((k, cs, s));
        }
        for &(k, cs, s) in &rots {
            rot_cols(&mut t, k, cs, s, 0..(k + 2).min(hi));
            rot_cols(&mut u, k, cs, s, 0..n);
        }
        for i in lo..=h {
            t[i][i] = t[i][i] + mu;
        }
    }
    for i in 1..n {
        for j in 0..i {
            t[i][j] = C::default();
        }
    }
    Some((t, u))
}

/// Swap the adjacent diagonal entries k and k+1 of a triangular T.
fn swap(t: &mut CMat, u: &mut CMat, k: usize) {
    let n = t.len();
    let (a, b, x) = (t[k][k], t[k + 1][k + 1], t[k][k + 1]);
    if a == b {
        return;
    }
    let (cs, s) = givens(x, b - a);
    rot_rows(t, k, cs, s, k..n);
    rot_cols(t, k, cs, s, 0..k + 2);
    rot_cols(u, k, cs, s, 0..n);
    t[k + 1][k] = C::default();
    t[k][k] = b;
    t[k + 1][k + 1] = a;
}

/// Reorder a Schur form so that the entries flagged by `sel` come first.
pub fn reorder(t: &mut CMat, u: &mut CMat, sel: &dyn Fn(C) -> bool) {
    let n = t.len();
    let mut flags: Vec<bool> = (0..n).map(|i| sel(t[i][i])).collect();
    let mut dest = 0;
    for i in 0..n {
        if flags[i] {
            let mut k = i;
            while k > dest {
                swap(t, u, k - 1);
                flags.swap(k - 1, k);
                k -= 1;
            }
            dest += 1;
        }
    }
}

/// Solution of a linear rational-expectations model.
#[derive(Clone, Debug)]
pub struct Solution {
    /// controls = G states
    pub g: Vec<Vec<f64>>,
    /// states(t+1) = H states(t) + shocks
    pub h: Vec<Vec<f64>>,
    /// generalized eigenvalues (modulus-sorted, infinite ones as +inf)
    pub eig: Vec<C>,
    pub n_stable: usize,
}

/// Solve `A E x(t+1) = B x(t)` where the first `ns` elements of x are
/// predetermined. Errors: `Err(n_stable)` when the saddle-path condition
/// fails (also when the pencil is singular, with `usize::MAX`).
pub fn klein(a: &[Vec<f64>], b: &[Vec<f64>], ns: usize) -> Result<Solution, usize> {
    let n = a.len();
    let ca = from_real(a);
    let cb = from_real(b);
    for mu in [c(0.3711, 0.2143), c(-0.6127, 0.4413), c(1.913, -0.877), c(0.0517, -2.31)] {
        let bm: CMat = (0..n).map(|i| (0..n).map(|j| cb[i][j] - mu * ca[i][j]).collect()).collect();
        let Some(nmat) = csolve(&bm, &ca) else { continue };
        let Some((mut t, mut u)) = schur(&nmat) else { continue };
        let lam = |nu: C| -> Option<C> {
            if nu.abs() < 1e-12 {
                None
            } else {
                Some(nu.recip() + mu)
            }
        };
        let stable = |nu: C| lam(nu).is_some_and(|l| l.abs() < 1.0 - 1e-10);
        reorder(&mut t, &mut u, &stable);
        let mut eig: Vec<C> = (0..n).map(|i| lam(t[i][i]).unwrap_or(c(f64::INFINITY, 0.0))).collect();
        let n_stable = (0..n).filter(|&i| stable(t[i][i])).count();
        eig.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
        if n_stable != ns {
            return Err(n_stable);
        }
        if ns == 0 {
            return Ok(Solution { g: vec![vec![]; n], h: vec![], eig, n_stable });
        }
        let z1: CMat = u.iter().map(|r| r[..ns].to_vec()).collect();
        let z11: CMat = z1[..ns].to_vec();
        let z21: CMat = z1[ns..].to_vec();
        // G = Z21 Z11^{-1}: solve Z11^T G^T = Z21^T
        let tr = |m: &CMat| -> CMat { (0..m.first().map_or(0, |r| r.len())).map(|j| m.iter().map(|r| r[j]).collect()).collect() };
        let Some(gt) = csolve(&tr(&z11), &tr(&z21)) else { return Err(usize::MAX) };
        let g = tr(&gt);
        // dynamics in the subspace: A Z1 K = B Z1
        let Some(k) = clstsq(&cmul(&ca, &z1), &cmul(&cb, &z1)) else { return Err(usize::MAX) };
        let zk = cmul(&z11, &k);
        let Some(ht) = csolve(&tr(&z11), &tr(&zk)) else { return Err(usize::MAX) };
        let h = tr(&ht);
        let re = |m: &CMat| -> Vec<Vec<f64>> { m.iter().map(|r| r.iter().map(|v| if v.re.abs() < 1e-13 { 0.0 } else { v.re }).collect()).collect() };
        return Ok(Solution { g: re(&g), h: re(&h), eig, n_stable });
    }
    Err(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_reproduces_matrix() {
        let a = from_real(&[vec![4.0, 1.0, -2.0, 0.5], vec![1.0, 3.0, 0.0, 2.0], vec![-1.0, 0.5, 1.0, 1.0], vec![2.0, -1.0, 0.3, -2.0]]);
        let (mut t, mut u) = schur(&a).unwrap();
        reorder(&mut t, &mut u, &|z| z.abs() < 2.0);
        let uh: CMat = (0..4).map(|j| (0..4).map(|i| u[i][j].conj()).collect()).collect();
        let back = cmul(&cmul(&u, &t), &uh);
        for i in 0..4 {
            for j in 0..4 {
                assert!((back[i][j] - a[i][j]).abs() < 1e-9, "{i},{j}");
                if i > j {
                    assert!(t[i][j].abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn forward_looking_scalar() {
        // x = beta E x' + z ; z' = rho z  => x = z / (1 - beta rho)
        let (beta, rho) = (0.9, 0.5);
        // order (z, x): rows: z' = rho z ; beta x' = x - z
        let a = vec![vec![1.0, 0.0], vec![0.0, beta]];
        let b = vec![vec![rho, 0.0], vec![-1.0, 1.0]];
        let s = klein(&a, &b, 1).unwrap();
        assert!((s.g[0][0] - 1.0 / (1.0 - beta * rho)).abs() < 1e-10);
        assert!((s.h[0][0] - rho).abs() < 1e-10);
    }
}
