//! Small dense linear-algebra helpers for the multivariate time-series
//! commands (VAR, SVAR, VEC, IRF): Kronecker products, general (complex)
//! eigenvalues, symmetric eigenproblems with a relative tolerance,
//! duplication/commutation matrices and numerical Jacobians.

use crate::matrix::Matrix;

/// `a` ⊗ `b`.
pub fn kron(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = Matrix::zeros(a.rows * b.rows, a.cols * b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let x = a[(i, j)];
            if x == 0.0 {
                continue;
            }
            for k in 0..b.rows {
                for l in 0..b.cols {
                    m[(i * b.rows + k, j * b.cols + l)] = x * b[(k, l)];
                }
            }
        }
    }
    m
}

pub fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let x = a[(i, k)];
            if x == 0.0 {
                continue;
            }
            for j in 0..b.cols {
                m[(i, j)] += x * b[(k, j)];
            }
        }
    }
    m
}

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = a.clone();
    m.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x += y);
    m
}

pub fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = a.clone();
    m.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x -= y);
    m
}

pub fn tr(a: &Matrix) -> Matrix {
    let mut m = Matrix::zeros(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            m[(j, i)] = a[(i, j)];
        }
    }
    m
}

/// Inverse by Gauss-Jordan elimination with partial pivoting.
pub fn inv(a: &Matrix) -> Option<Matrix> {
    let n = a.rows;
    let mut m = a.clone();
    let mut r = Matrix::identity(n);
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[(x, c)].abs().total_cmp(&m[(y, c)].abs()))?;
        if m[(p, c)].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for j in 0..n {
                m.data.swap(c * n + j, p * n + j);
                r.data.swap(c * n + j, p * n + j);
            }
        }
        let d = m[(c, c)];
        for j in 0..n {
            m[(c, j)] /= d;
            r[(c, j)] /= d;
        }
        for i in 0..n {
            if i != c {
                let f = m[(i, c)];
                if f != 0.0 {
                    for j in 0..n {
                        m[(i, j)] -= f * m[(c, j)];
                        r[(i, j)] -= f * r[(c, j)];
                    }
                }
            }
        }
    }
    Some(r)
}

/// Inverse of a symmetric matrix; columns that are (numerically) linear
/// combinations of earlier ones get zero rows/columns (generalised inverse).
pub fn sym_ginv(a: &Matrix) -> Matrix {
    let scale: Vec<f64> = (0..a.rows).map(|i| if a[(i, i)] > 0.0 { 1.0 / a[(i, i)].sqrt() } else { 0.0 }).collect();
    let mut s = a.clone();
    for i in 0..a.rows {
        for j in 0..a.cols {
            s[(i, j)] *= scale[i] * scale[j];
        }
    }
    let (inv, _) = s.sym_inverse(1e-13);
    let mut out = inv;
    for i in 0..a.rows {
        for j in 0..a.cols {
            out[(i, j)] *= scale[i] * scale[j];
        }
    }
    out
}

/// Symmetric eigen-decomposition (cyclic Jacobi, relative tolerance);
/// eigenvalues in descending order, eigenvectors in columns.
pub fn sym_eig(a: &Matrix) -> (Vec<f64>, Matrix) {
    let n = a.rows;
    let mut m = a.clone();
    let mut v = Matrix::identity(n);
    let norm: f64 = m.data.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    for _ in 0..200 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[(i, j)] * m[(i, j)]).sum();
        if off <= norm * 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if m[(p, q)] == 0.0 {
                    continue;
                }
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * m[(p, q)]);
                let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (m[(k, p)], m[(k, q)]);
                    m[(k, p)] = c * x - s * y;
                    m[(k, q)] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (m[(p, k)], m[(q, k)]);
                    m[(p, k)] = c * x - s * y;
                    m[(q, k)] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (v[(k, p)], v[(k, q)]);
                    v[(k, p)] = c * x - s * y;
                    v[(k, q)] = s * x + c * y;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| m[(j, j)].total_cmp(&m[(i, i)]));
    let vals = idx.iter().map(|&i| m[(i, i)]).collect();
    let mut vecs = Matrix::zeros(n, n);
    for (nj, &oj) in idx.iter().enumerate() {
        for i in 0..n {
            vecs[(i, nj)] = v[(i, oj)];
        }
    }
    (vals, vecs)
}

/// Eigenvalues (re, im) of a general real matrix: balancing, reduction to
/// Hessenberg form and the shifted QR algorithm. Sorted by decreasing modulus.
pub fn eig_general(a: &Matrix) -> Vec<(f64, f64)> {
    let n = a.rows;
    if n == 0 {
        return Vec::new();
    }
    let mut h: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| a[(i, j)]).collect()).collect();
    balance(&mut h);
    hessenberg(&mut h);
    let mut ev = hqr(&mut h);
    ev.sort_by(|x, y| {
        let mx = (x.0 * x.0 + x.1 * x.1).sqrt();
        let my = (y.0 * y.0 + y.1 * y.1).sqrt();
        my.partial_cmp(&mx).unwrap_or(std::cmp::Ordering::Equal).then(y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    ev
}

fn balance(a: &mut [Vec<f64>]) {
    let n = a.len();
    let radix = 2.0f64;
    let sqrdx = radix * radix;
    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            let mut r = 0.0;
            let mut c = 0.0;
            for j in 0..n {
                if j != i {
                    c += a[j][i].abs();
                    r += a[i][j].abs();
                }
            }
            if c != 0.0 && r != 0.0 {
                let mut g = r / radix;
                let mut f = 1.0;
                let s = c + r;
                while c < g {
                    f *= radix;
                    c *= sqrdx;
                }
                g = r * radix;
                while c > g {
                    f /= radix;
                    c /= sqrdx;
                }
                if (c + r) / f < 0.95 * s {
                    done = false;
                    let g = 1.0 / f;
                    for j in 0..n {
                        a[i][j] *= g;
                    }
                    for j in 0..n {
                        a[j][i] *= f;
                    }
                }
            }
        }
    }
}

fn hessenberg(a: &mut [Vec<f64>]) {
    let n = a.len();
    for m in 1..n.saturating_sub(1) {
        let mut x: f64 = 0.0;
        let mut i = m;
        for j in m..n {
            if a[j][m - 1].abs() > x.abs() {
                x = a[j][m - 1];
                i = j;
            }
        }
        if i != m {
            for j in (m - 1)..n {
                let t = a[i][j];
                a[i][j] = a[m][j];
                a[m][j] = t;
            }
            for j in 0..n {
                let t = a[j][i];
                a[j][i] = a[j][m];
                a[j][m] = t;
            }
        }
        if x != 0.0 {
            for i in (m + 1)..n {
                let mut y = a[i][m - 1];
                if y != 0.0 {
                    y /= x;
                    a[i][m - 1] = y;
                    for j in m..n {
                        a[i][j] -= y * a[m][j];
                    }
                    for j in 0..n {
                        a[j][m] += y * a[j][i];
                    }
                }
            }
        }
    }
    for i in 2..n {
        for j in 0..i - 1 {
            a[i][j] = 0.0;
        }
    }
}

fn hqr(a: &mut [Vec<f64>]) -> Vec<(f64, f64)> {
    let n = a.len();
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let mut anorm = 0.0;
    for i in 0..n {
        for j in i.saturating_sub(1)..n {
            anorm += a[i][j].abs();
        }
    }
    let mut nn: isize = n as isize - 1;
    let mut t = 0.0;
    let (mut p, mut q, mut r, mut s, mut w, mut x, mut y, mut z);
    p = 0.0;
    q = 0.0;
    r = 0.0;
    while nn >= 0 {
        let mut its = 0;
        let mut l: isize;
        loop {
            l = nn;
            while l >= 1 {
                let lu = l as usize;
                s = a[lu - 1][lu - 1].abs() + a[lu][lu].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[lu][lu - 1].abs() + s == s {
                    a[lu][lu - 1] = 0.0;
                    break;
                }
                l -= 1;
            }
            let nu = nn as usize;
            x = a[nu][nu];
            if l == nn {
                wr[nu] = x + t;
                wi[nu] = 0.0;
                nn -= 1;
                break;
            }
            y = a[nu - 1][nu - 1];
            w = a[nu][nu - 1] * a[nu - 1][nu];
            if l == nn - 1 {
                p = 0.5 * (y - x);
                q = p * p + w;
                z = q.abs().sqrt();
                x += t;
                if q >= 0.0 {
                    z = p + if p >= 0.0 { z.abs() } else { -z.abs() };
                    wr[nu - 1] = x + z;
                    wr[nu] = wr[nu - 1];
                    if z != 0.0 {
                        wr[nu] = x - w / z;
                    }
                    wi[nu - 1] = 0.0;
                    wi[nu] = 0.0;
                } else {
                    wr[nu - 1] = x + p;
                    wr[nu] = x + p;
                    wi[nu - 1] = -z;
                    wi[nu] = z;
                }
                nn -= 2;
                break;
            }
            if its == 60 {
                // give up: report the diagonal
                for i in 0..=nu {
                    wr[i] = a[i][i] + t;
                }
                return wr.into_iter().zip(wi).collect();
            }
            if its == 10 || its == 20 {
                t += x;
                for i in 0..=nu {
                    a[i][i] -= x;
                }
                s = a[nu][nu - 1].abs() + a[nu - 1][nu - 2].abs();
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            its += 1;
            let mut m = nn - 2;
            while m >= l {
                let mu = m as usize;
                z = a[mu][mu];
                r = x - z;
                s = y - z;
                p = (r * s - w) / a[mu + 1][mu] + a[mu][mu + 1];
                q = a[mu + 1][mu + 1] - z - r - s;
                r = a[mu + 2][mu + 1];
                s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let u = a[mu][mu - 1].abs() * (q.abs() + r.abs());
                let v = p.abs() * (a[mu - 1][mu - 1].abs() + z.abs() + a[mu + 1][mu + 1].abs());
                if u + v == v {
                    break;
                }
                m -= 1;
            }
            let mu = m as usize;
            for i in (mu + 2)..=nu {
                a[i][i - 2] = 0.0;
                if i != mu + 2 {
                    a[i][i - 3] = 0.0;
                }
            }
            let mut k = mu;
            while k + 1 <= nu {
                if k != mu {
                    p = a[k][k - 1];
                    q = a[k + 1][k - 1];
                    r = 0.0;
                    if k + 1 != nu {
                        r = a[k + 2][k - 1];
                    }
                    x = p.abs() + q.abs() + r.abs();
                    if x != 0.0 {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                }
                let sv = (p * p + q * q + r * r).sqrt();
                s = if p >= 0.0 { sv } else { -sv };
                if s != 0.0 {
                    if k == mu {
                        if l != m {
                            a[k][k - 1] = -a[k][k - 1];
                        }
                    } else {
                        a[k][k - 1] = -s * x;
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q /= p;
                    r /= p;
                    for j in k..=nu {
                        p = a[k][j] + q * a[k + 1][j];
                        if k + 1 != nu {
                            p += r * a[k + 2][j];
                            a[k + 2][j] -= p * z;
                        }
                        a[k + 1][j] -= p * y;
                        a[k][j] -= p * x;
                    }
                    let mmin = if nu < k + 3 { nu } else { k + 3 };
                    for i in (l as usize)..=mmin {
                        p = x * a[i][k] + y * a[i][k + 1];
                        if k + 1 != nu {
                            p += z * a[i][k + 2];
                            a[i][k + 2] -= p * r;
                        }
                        a[i][k + 1] -= p * q;
                        a[i][k] -= p;
                    }
                }
                k += 1;
            }
            if l >= nn - 1 {
                continue;
            }
        }
    }
    wr.into_iter().zip(wi).collect()
}

/// Duplication matrix D_K: vec(F) = D vech(F) for symmetric F (K² × K(K+1)/2).
pub fn duplication(k: usize) -> Matrix {
    let m = k * (k + 1) / 2;
    let mut d = Matrix::zeros(k * k, m);
    let mut c = 0;
    for j in 0..k {
        for i in j..k {
            d[(j * k + i, c)] = 1.0;
            d[(i * k + j, c)] = 1.0;
            c += 1;
        }
    }
    d
}

/// vech of a symmetric matrix (column-wise lower triangle).
pub fn vech(a: &Matrix) -> Vec<f64> {
    let k = a.rows;
    let mut v = Vec::new();
    for j in 0..k {
        for i in j..k {
            v.push(a[(i, j)]);
        }
    }
    v
}

pub fn unvech(v: &[f64], k: usize) -> Matrix {
    let mut a = Matrix::zeros(k, k);
    let mut c = 0;
    for j in 0..k {
        for i in j..k {
            a[(i, j)] = v[c];
            a[(j, i)] = v[c];
            c += 1;
        }
    }
    a
}

/// Asymptotic VCE of vech(Σ̂): 2 D⁺ (Σ ⊗ Σ) D⁺' / T.
pub fn vech_sigma_vce(sigma: &Matrix, t: f64) -> Matrix {
    let d = duplication(sigma.rows);
    let dtd = mul(&tr(&d), &d);
    let dplus = mul(&inv(&dtd).unwrap_or_else(|| Matrix::identity(dtd.rows)), &tr(&d));
    let v = mul(&mul(&dplus, &kron(sigma, sigma)), &tr(&dplus));
    v.scale(2.0 / t)
}

/// Lower Cholesky factor (zeros for non-positive pivots).
pub fn chol(a: &Matrix) -> Matrix {
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum();
            if i == j {
                let v = a[(i, i)] - s;
                l[(i, j)] = if v > 0.0 { v.sqrt() } else { 0.0 };
            } else {
                l[(i, j)] = if l[(j, j)] != 0.0 { (a[(i, j)] - s) / l[(j, j)] } else { 0.0 };
            }
        }
    }
    l
}

pub fn logdet(a: &Matrix) -> f64 {
    a.det().ln()
}

/// Numerical Jacobian of `f` at `x` (central differences): rows are the
/// outputs, columns the inputs.
pub fn jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Matrix {
    let f0 = f(x);
    let mut j = Matrix::zeros(f0.len(), x.len());
    let mut xx = x.to_vec();
    for c in 0..x.len() {
        let h = 1e-6 * x[c].abs().max(1e-3);
        xx[c] = x[c] + h;
        let fp = f(&xx);
        xx[c] = x[c] - h;
        let fm = f(&xx);
        xx[c] = x[c];
        for r in 0..f0.len() {
            j[(r, c)] = (fp[r] - fm[r]) / (2.0 * h);
        }
    }
    j
}

/// Diagonal of J V J' (variances of the transformed quantities).
pub fn delta_var(j: &Matrix, v: &Matrix) -> Vec<f64> {
    let jv = mul(j, v);
    (0..j.rows).map(|r| (0..j.cols).map(|c| jv[(r, c)] * j[(r, c)]).sum::<f64>().max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalues_of_companion() {
        // x^2 - x + 0.5: roots 0.5 ± 0.5i
        let a = Matrix::from_rows(&[vec![1.0, -0.5], vec![1.0, 0.0]]);
        let ev = eig_general(&a);
        assert!((ev[0].0 - 0.5).abs() < 1e-12 && (ev[0].1.abs() - 0.5).abs() < 1e-12);
        let b = Matrix::from_rows(&[vec![2.0, 0.0, 0.0], vec![0.0, -3.0, 1.0], vec![0.0, 0.0, 0.5]]);
        let ev = eig_general(&b);
        assert!((ev[0].0 + 3.0).abs() < 1e-12 && (ev[1].0 - 2.0).abs() < 1e-12 && (ev[2].0 - 0.5).abs() < 1e-12);
    }
}
