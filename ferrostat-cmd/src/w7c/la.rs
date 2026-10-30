//! Small dense linear algebra on row-major `Vec<f64>` square matrices.

/// Cholesky factor (lower) of an n x n SPD matrix; None if not PD.
pub fn chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

pub fn chol_logdet(l: &[f64], n: usize) -> f64 {
    (0..n).map(|i| l[i * n + i].ln()).sum::<f64>() * 2.0
}

/// Solve L L' x = b.
pub fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

/// Inverse of an SPD matrix from its Cholesky factor.
pub fn chol_inv(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut e = vec![0.0; n];
    for j in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        let c = chol_solve(l, n, &e);
        for i in 0..n {
            inv[i * n + j] = c[i];
        }
    }
    inv
}

/// General inverse by Gauss-Jordan with partial pivoting.
pub fn inv(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut r = vec![0.0; n * n];
    for i in 0..n {
        r[i * n + i] = 1.0;
    }
    for c in 0..n {
        let mut p = c;
        for i in c + 1..n {
            if m[i * n + c].abs() > m[p * n + c].abs() {
                p = i;
            }
        }
        if m[p * n + c].abs() < 1e-13 {
            return None;
        }
        if p != c {
            for j in 0..n {
                m.swap(p * n + j, c * n + j);
                r.swap(p * n + j, c * n + j);
            }
        }
        let d = m[c * n + c];
        for j in 0..n {
            m[c * n + j] /= d;
            r[c * n + j] /= d;
        }
        for i in 0..n {
            if i != c {
                let f = m[i * n + c];
                if f != 0.0 {
                    for j in 0..n {
                        m[i * n + j] -= f * m[c * n + j];
                        r[i * n + j] -= f * r[c * n + j];
                    }
                }
            }
        }
    }
    Some(r)
}

/// C = A B (A: n x m, B: m x k)
pub fn mul(a: &[f64], b: &[f64], n: usize, m: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * k];
    for i in 0..n {
        for t in 0..m {
            let x = a[i * m + t];
            if x != 0.0 {
                for j in 0..k {
                    c[i * k + j] += x * b[t * k + j];
                }
            }
        }
    }
    c
}

pub fn transpose(a: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            t[j * n + i] = a[i * m + j];
        }
    }
    t
}

/// Submatrix rows `r`, cols `c` of an n-column matrix.
pub fn sub(a: &[f64], n: usize, r: &[usize], c: &[usize]) -> Vec<f64> {
    let mut s = Vec::with_capacity(r.len() * c.len());
    for &i in r {
        for &j in c {
            s.push(a[i * n + j]);
        }
    }
    s
}

pub fn det(a: &[f64], n: usize) -> f64 {
    let mut m = a.to_vec();
    let mut d = 1.0;
    for c in 0..n {
        let mut p = c;
        for i in c + 1..n {
            if m[i * n + c].abs() > m[p * n + c].abs() {
                p = i;
            }
        }
        if m[p * n + c] == 0.0 {
            return 0.0;
        }
        if p != c {
            for j in 0..n {
                m.swap(p * n + j, c * n + j);
            }
            d = -d;
        }
        let piv = m[c * n + c];
        d *= piv;
        for i in c + 1..n {
            let f = m[i * n + c] / piv;
            for j in c..n {
                m[i * n + j] -= f * m[c * n + j];
            }
        }
    }
    d
}
