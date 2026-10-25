//! Dense row-major matrices with row/column names.

use ferrostat_core::missing::{is_missing, SYSMISS};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub rownames: Vec<String>,
    pub colnames: Vec<String>,
}

fn default_names(prefix: &str, n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("{prefix}{i}")).collect()
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            rownames: default_names("r", rows),
            colnames: default_names("c", cols),
        }
    }

    pub fn filled(rows: usize, cols: usize, v: f64) -> Self {
        let mut m = Self::zeros(rows, cols);
        m.data.iter_mut().for_each(|x| *x = v);
        m
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let r = rows.len();
        let c = rows.first().map_or(0, |x| x.len());
        let mut m = Self::zeros(r, c);
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                m[(i, j)] = *v;
            }
        }
        m
    }

    pub fn row_vector(v: &[f64]) -> Self {
        Self::from_rows(&[v.to_vec()])
    }

    pub fn col_vector(v: &[f64]) -> Self {
        let mut m = Self::zeros(v.len(), 1);
        m.data.copy_from_slice(v);
        m
    }

    pub fn with_names(mut self, rownames: Vec<String>, colnames: Vec<String>) -> Self {
        if rownames.len() == self.rows {
            self.rownames = rownames;
        }
        if colnames.len() == self.cols {
            self.colnames = colnames;
        }
        self
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t.rownames = self.colnames.clone();
        t.colnames = self.rownames.clone();
        t
    }

    pub fn mul(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err("conformability error".into());
        }
        let mut m = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    m.data[i * other.cols + j] += a * other.data[k * other.cols + j];
                }
            }
        }
        m.rownames = self.rownames.clone();
        m.colnames = other.colnames.clone();
        Ok(m)
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("conformability error".into());
        }
        let mut m = self.clone();
        m.data.iter_mut().zip(&other.data).for_each(|(a, b)| *a += b);
        Ok(m)
    }

    pub fn sub(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("conformability error".into());
        }
        let mut m = self.clone();
        m.data.iter_mut().zip(&other.data).for_each(|(a, b)| *a -= b);
        Ok(m)
    }

    pub fn scale(&self, s: f64) -> Matrix {
        let mut m = self.clone();
        m.data.iter_mut().for_each(|a| *a *= s);
        m
    }

    pub fn diag(&self) -> Vec<f64> {
        (0..self.rows.min(self.cols)).map(|i| self[(i, i)]).collect()
    }

    pub fn from_diag(v: &[f64]) -> Matrix {
        let mut m = Matrix::zeros(v.len(), v.len());
        for (i, x) in v.iter().enumerate() {
            m[(i, i)] = *x;
        }
        m
    }

    pub fn trace(&self) -> f64 {
        self.diag().iter().sum()
    }

    /// Inverse of a symmetric matrix via the sweep operator; columns whose
    /// pivot falls below `tol` (relative) are treated as collinear and their
    /// rows/columns are set to zero. Returns (inverse, dropped indices).
    pub fn sym_inverse(&self, tol: f64) -> (Matrix, Vec<usize>) {
        let n = self.rows;
        let mut a = self.clone();
        let diag0: Vec<f64> = (0..n).map(|i| self[(i, i)].abs()).collect();
        let mut dropped = Vec::new();
        let mut swept = vec![false; n];
        for k in 0..n {
            let d = a[(k, k)];
            if d.abs() <= tol * diag0[k].max(1e-300) || d.abs() < 1e-300 {
                dropped.push(k);
                continue;
            }
            // sweep on k
            for i in 0..n {
                if i == k {
                    continue;
                }
                for j in 0..n {
                    if j == k {
                        continue;
                    }
                    a[(i, j)] -= a[(i, k)] * a[(k, j)] / d;
                }
            }
            for i in 0..n {
                if i != k {
                    a[(i, k)] /= d;
                    a[(k, i)] /= d;
                }
            }
            a[(k, k)] = -1.0 / d;
            swept[k] = true;
        }
        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if swept[i] && swept[j] {
                    inv[(i, j)] = -a[(i, j)];
                }
            }
        }
        inv.rownames = self.rownames.clone();
        inv.colnames = self.colnames.clone();
        (inv, dropped)
    }

    /// General inverse by Gauss-Jordan with partial pivoting.
    pub fn inverse(&self) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("matrix not square".into());
        }
        let n = self.rows;
        let mut a = self.clone();
        let mut inv = Matrix::identity(n);
        for c in 0..n {
            let p = (c..n).max_by(|&x, &y| a[(x, c)].abs().total_cmp(&a[(y, c)].abs())).unwrap();
            if a[(p, c)].abs() < 1e-300 {
                return Err("matrix has missing values or is singular".into());
            }
            for j in 0..n {
                a.data.swap(c * n + j, p * n + j);
                inv.data.swap(c * n + j, p * n + j);
            }
            let d = a[(c, c)];
            for j in 0..n {
                a[(c, j)] /= d;
                inv[(c, j)] /= d;
            }
            for r in 0..n {
                if r != c {
                    let f = a[(r, c)];
                    if f != 0.0 {
                        for j in 0..n {
                            a[(r, j)] -= f * a[(c, j)];
                            inv[(r, j)] -= f * inv[(c, j)];
                        }
                    }
                }
            }
        }
        inv.rownames = self.colnames.clone();
        inv.colnames = self.rownames.clone();
        Ok(inv)
    }

    /// Cholesky factor L (lower) of a symmetric positive-definite matrix.
    pub fn cholesky(&self) -> Result<Matrix, String> {
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let s: f64 = (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum();
                if i == j {
                    let v = self[(i, i)] - s;
                    if v <= 0.0 {
                        return Err("matrix not positive definite".into());
                    }
                    l[(i, j)] = v.sqrt();
                } else {
                    l[(i, j)] = (self[(i, j)] - s) / l[(j, j)];
                }
            }
        }
        Ok(l)
    }

    pub fn det(&self) -> f64 {
        let n = self.rows;
        let mut a = self.clone();
        let mut det = 1.0;
        for c in 0..n {
            let p = (c..n).max_by(|&x, &y| a[(x, c)].abs().total_cmp(&a[(y, c)].abs())).unwrap();
            if a[(p, c)] == 0.0 {
                return 0.0;
            }
            if p != c {
                for j in 0..n {
                    a.data.swap(c * n + j, p * n + j);
                }
                det = -det;
            }
            det *= a[(c, c)];
            for r in c + 1..n {
                let f = a[(r, c)] / a[(c, c)];
                for j in c..n {
                    a[(r, j)] -= f * a[(c, j)];
                }
            }
        }
        det
    }

    /// Eigenvalues/vectors of a symmetric matrix (Jacobi rotations), sorted
    /// in descending order of eigenvalue. Vectors are columns.
    pub fn sym_eigen(&self) -> (Vec<f64>, Matrix) {
        let n = self.rows;
        let mut a = self.clone();
        let mut v = Matrix::identity(n);
        for _ in 0..100 {
            let mut off = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off += a[(i, j)] * a[(i, j)];
                    }
                }
            }
            if off < 1e-22 {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    if a[(p, q)].abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * a[(p, q)]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[(k, p)];
                        let akq = a[(k, q)];
                        a[(k, p)] = c * akp - s * akq;
                        a[(k, q)] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[(p, k)];
                        let aqk = a[(q, k)];
                        a[(p, k)] = c * apk - s * aqk;
                        a[(q, k)] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let vkp = v[(k, p)];
                        let vkq = v[(k, q)];
                        v[(k, p)] = c * vkp - s * vkq;
                        v[(k, q)] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| a[(j, j)].total_cmp(&a[(i, i)]));
        let vals = idx.iter().map(|&i| a[(i, i)]).collect();
        let mut vecs = Matrix::zeros(n, n);
        for (newj, &oldj) in idx.iter().enumerate() {
            for i in 0..n {
                vecs[(i, newj)] = v[(i, oldj)];
            }
        }
        (vals, vecs)
    }

    pub fn has_missing(&self) -> bool {
        self.data.iter().any(|&x| is_missing(x))
    }

    /// Element, or missing when out of range.
    pub fn el(&self, i: usize, j: usize) -> f64 {
        if i < self.rows && j < self.cols { self[(i, j)] } else { SYSMISS }
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.colnames.iter().position(|c| c == name)
    }

    pub fn row_index(&self, name: &str) -> Option<usize> {
        self.rownames.iter().position(|c| c == name)
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip() {
        let a = Matrix::from_rows(&[vec![4.0, 2.0], vec![2.0, 3.0]]);
        let (inv, dropped) = a.sym_inverse(1e-12);
        assert!(dropped.is_empty());
        let i = a.mul(&inv).unwrap();
        assert!((i[(0, 0)] - 1.0).abs() < 1e-12 && i[(0, 1)].abs() < 1e-12);
        let g = a.inverse().unwrap();
        assert!((g[(1, 1)] - inv[(1, 1)]).abs() < 1e-12);
        assert!((a.det() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn collinear_columns_dropped() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let (_, dropped) = a.sym_inverse(1e-10);
        assert_eq!(dropped, vec![1]);
    }

    #[test]
    fn eigen() {
        let a = Matrix::from_rows(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        let (vals, _) = a.sym_eigen();
        assert!((vals[0] - 3.0).abs() < 1e-10 && (vals[1] - 1.0).abs() < 1e-10);
    }
}
