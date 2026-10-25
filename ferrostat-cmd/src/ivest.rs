//! Linear instrumental-variables estimators on plain matrices:
//! 2SLS, LIML (k-class), and efficient GMM, with their overidentification
//! tests. Textbook formulas: Davidson & MacKinnon (2004, ch. 8-9),
//! Hayashi (2000, ch. 3), Wooldridge (2010, ch. 5 and 8).
//!
//! `x` holds the regressors (endogenous and included exogenous, constant
//! included when present), `z` the instruments (excluded and included
//! exogenous), `w` observation weights (1 when unweighted).

use std::collections::BTreeMap;

use crate::matrix::Matrix;

/// How the moment covariance S is estimated (GMM weight matrix / VCE).
#[derive(Debug, Clone, PartialEq)]
pub enum Moments {
    /// σ² Z'Z (homoskedastic)
    Unadjusted,
    /// Σ u² z z' (heteroskedasticity-robust)
    Robust,
    /// Σ_g (Σ_{t∈g} z u)(…)' with cluster identifiers per observation
    Cluster(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct IvFit {
    pub b: Vec<f64>,
    pub resid: Vec<f64>,
    /// "bread" (X̃'W X)^-1 of the sandwich; V = σ² bread under homoskedasticity
    pub bread: Matrix,
    /// regressors used in the scores: X̂ for 2SLS, (I - k M_Z) X for LIML
    pub xt: Matrix,
    /// LIML κ (1 for 2SLS)
    pub kappa: f64,
    /// GMM: estimated covariance (already final)
    pub v_gmm: Option<Matrix>,
    /// GMM: Hansen J
    pub j: Option<f64>,
}

fn wcross(a: &Matrix, b: &Matrix, w: &[f64]) -> Matrix {
    let mut m = Matrix::zeros(a.cols, b.cols);
    for t in 0..a.rows {
        for i in 0..a.cols {
            let ai = a[(t, i)] * w[t];
            if ai == 0.0 {
                continue;
            }
            for j in 0..b.cols {
                m[(i, j)] += ai * b[(t, j)];
            }
        }
    }
    m
}

fn col(v: &[f64]) -> Matrix {
    Matrix::col_vector(v)
}

fn inv(m: &Matrix) -> Matrix {
    m.sym_inverse(1e-13).0
}

fn residuals(x: &Matrix, y: &[f64], b: &[f64]) -> Vec<f64> {
    (0..x.rows).map(|t| y[t] - (0..x.cols).map(|j| x[(t, j)] * b[j]).sum::<f64>()).collect()
}

/// Fitted values of every column of `a` from a weighted regression on `z`.
pub fn project(a: &Matrix, z: &Matrix, w: &[f64]) -> Matrix {
    let zzi = inv(&wcross(z, z, w));
    let coef = zzi.mul(&wcross(z, a, w)).unwrap();
    z.mul(&coef).unwrap()
}

/// k-class estimator: k = 1 is 2SLS, k = κ_LIML is LIML.
fn kclass(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64], k: f64) -> IvFit {
    let xhat = project(x, z, w);
    // X̃ = (I - k M_Z) X = X - k (X - X̂)
    let mut xt = x.clone();
    for t in 0..x.rows {
        for j in 0..x.cols {
            xt[(t, j)] = x[(t, j)] - k * (x[(t, j)] - xhat[(t, j)]);
        }
    }
    let bread = inv(&wcross(&xt, x, w));
    let b = bread.mul(&wcross(&xt, &col(y), w)).unwrap().data;
    let resid = residuals(x, y, &b);
    IvFit { b, resid, bread, xt, kappa: k, v_gmm: None, j: None }
}

pub fn tsls(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64]) -> IvFit {
    kclass(x, z, y, w, 1.0)
}

/// LIML κ: smallest eigenvalue of (Y'M_Z Y)^-1 (Y'M_X1 Y), Y = [y, endogenous].
pub fn liml_kappa(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64], exog_cols: &[usize]) -> f64 {
    let n = x.rows;
    let endog: Vec<usize> = (0..x.cols).filter(|j| !exog_cols.contains(j)).collect();
    let mut ym = Matrix::zeros(n, 1 + endog.len());
    for t in 0..n {
        ym[(t, 0)] = y[t];
        for (c, &j) in endog.iter().enumerate() {
            ym[(t, c + 1)] = x[(t, j)];
        }
    }
    let annihilate = |by: &Matrix| -> Matrix {
        if by.cols == 0 {
            return ym.clone();
        }
        ym.sub(&project(&ym, by, w)).unwrap()
    };
    let mut x1 = Matrix::zeros(n, exog_cols.len());
    for t in 0..n {
        for (c, &j) in exog_cols.iter().enumerate() {
            x1[(t, c)] = x[(t, j)];
        }
    }
    let e1 = annihilate(&x1);
    let e = annihilate(z);
    let w1 = wcross(&e1, &e1, w);
    let w0 = wcross(&e, &e, w);
    // symmetric form: W0^-1/2 W1 W0^-1/2
    let (vals, vecs) = w0.sym_eigen();
    let k = w0.rows;
    let mut h = Matrix::zeros(k, k);
    for (e, &lam) in vals.iter().enumerate() {
        let s = if lam > 0.0 { lam.powf(-0.5) } else { 0.0 };
        for i in 0..k {
            for j in 0..k {
                h[(i, j)] += s * vecs[(i, e)] * vecs[(j, e)];
            }
        }
    }
    let (ev, _) = h.mul(&w1).unwrap().mul(&h).unwrap().sym_eigen();
    ev.into_iter().fold(f64::INFINITY, f64::min)
}

pub fn liml(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64], exog_cols: &[usize]) -> IvFit {
    let k = liml_kappa(x, z, y, w, exog_cols);
    kclass(x, z, y, w, k)
}

/// Moment covariance S = Σ (z u)(z u)' (robust / cluster) or σ² Z'Z, not divided by N.
pub fn moment_cov(z: &Matrix, u: &[f64], w: &[f64], m: &Moments) -> Matrix {
    let l = z.cols;
    match m {
        Moments::Unadjusted => {
            let n: f64 = w.iter().sum();
            let s2 = (0..z.rows).map(|t| w[t] * u[t] * u[t]).sum::<f64>() / n;
            wcross(z, z, w).scale(s2)
        }
        Moments::Robust => {
            let mut s = Matrix::zeros(l, l);
            for t in 0..z.rows {
                let f = w[t] * u[t] * u[t];
                for a in 0..l {
                    for c in 0..l {
                        s[(a, c)] += f * z[(t, a)] * z[(t, c)];
                    }
                }
            }
            s
        }
        Moments::Cluster(cl) => {
            let mut sums: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
            for t in 0..z.rows {
                let g = sums.entry(cl[t].to_bits()).or_insert_with(|| vec![0.0; l]);
                for a in 0..l {
                    g[a] += w[t] * z[(t, a)] * u[t];
                }
            }
            let mut s = Matrix::zeros(l, l);
            for g in sums.values() {
                for a in 0..l {
                    for c in 0..l {
                        s[(a, c)] += g[a] * g[c];
                    }
                }
            }
            s
        }
    }
}

/// GMM with weight matrix S^-1 given S.
fn gmm_step(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64], s: &Matrix) -> Vec<f64> {
    let zx = wcross(z, x, w);
    let zy = wcross(z, &col(y), w);
    let si = inv(s);
    let a = zx.transpose().mul(&si).unwrap();
    inv(&a.mul(&zx).unwrap()).mul(&a.mul(&zy).unwrap()).unwrap().data
}

/// Two-step (or iterated) efficient GMM. The weight matrix comes from the
/// previous step's residuals; the reported covariance is the sandwich
/// evaluated with the final residuals (`vce`) and the Hansen J statistic
/// uses the final weight matrix.
pub fn gmm(x: &Matrix, z: &Matrix, y: &[f64], w: &[f64], wmat: &Moments, vce: &Moments, iterate: bool) -> IvFit {
    let first = tsls(x, z, y, w);
    let mut b = first.b.clone();
    let mut s = moment_cov(z, &first.resid, w, wmat);
    let mut steps = 0;
    loop {
        let nb = gmm_step(x, z, y, w, &s);
        let change = nb.iter().zip(&b).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max) / (1.0 + b.iter().map(|v| v.abs()).fold(0.0, f64::max));
        b = nb;
        steps += 1;
        if !iterate || change < 1e-9 || steps >= 200 {
            break;
        }
        s = moment_cov(z, &residuals(x, y, &b), w, wmat);
    }
    let resid = residuals(x, y, &b);
    let si = inv(&s);
    let zx = wcross(z, x, w);
    let bread = inv(&zx.transpose().mul(&si).unwrap().mul(&zx).unwrap());
    let s_final = moment_cov(z, &resid, w, vce);
    let meat = zx.transpose().mul(&si).unwrap().mul(&s_final).unwrap().mul(&si).unwrap().mul(&zx).unwrap();
    let v = bread.mul(&meat).unwrap().mul(&bread).unwrap();
    // Hansen J with the weight matrix re-estimated at the final residuals
    let g = wcross(z, &col(&resid), w);
    let s_j = moment_cov(z, &resid, w, wmat);
    let j = g.transpose().mul(&inv(&s_j)).unwrap().mul(&g).unwrap()[(0, 0)];
    IvFit { b, resid, bread, xt: x.clone(), kappa: 1.0, v_gmm: Some(v), j: Some(j) }
}

/// Covariance of a k-class fit: σ² bread (unadjusted, σ² = RSS/denom) or
/// the sandwich with scores x̃ u (robust / cluster, no small-sample factor).
pub fn kclass_vce(fit: &IvFit, w: &[f64], m: &Moments, denom: f64) -> Matrix {
    match m {
        Moments::Unadjusted => {
            let rss: f64 = fit.resid.iter().zip(w).map(|(u, w)| w * u * u).sum();
            fit.bread.scale(rss / denom)
        }
        _ => {
            let s = moment_cov(&fit.xt, &fit.resid, w, m);
            fit.bread.mul(&s).unwrap().mul(&fit.bread).unwrap()
        }
    }
}

/// Sargan (N R² of the 2SLS residuals on Z) and Basmann F-form statistics.
pub fn sargan_basmann(z: &Matrix, u: &[f64], w: &[f64], n: f64, overid: f64, l: f64) -> (f64, f64) {
    let fitted = project(&col(u), z, w);
    let ess: f64 = (0..z.rows).map(|t| w[t] * fitted[(t, 0)].powi(2)).sum();
    let rss: f64 = u.iter().zip(w).map(|(u, w)| w * u * u).sum();
    let sargan = n * ess / rss;
    let basmann = (ess / overid) / ((rss - ess) / (n - l));
    (sargan, basmann)
}

/// Wooldridge's (1995) heteroskedasticity- or cluster-robust score test of
/// overidentifying restrictions: residualise `q` of the excluded instruments
/// on the fitted regressors X̂ and test that they are uncorrelated with the
/// 2SLS residuals.
pub fn robust_score(xhat: &Matrix, z: &Matrix, excl: &[usize], u: &[f64], w: &[f64], m: &Moments) -> f64 {
    let mut ze = Matrix::zeros(z.rows, excl.len());
    for t in 0..z.rows {
        for (c, &j) in excl.iter().enumerate() {
            ze[(t, c)] = z[(t, j)];
        }
    }
    let r = ze.sub(&project(&ze, xhat, w)).unwrap();
    let g = wcross(&r, &col(u), w);
    let s = moment_cov(&r, u, w, m);
    g.transpose().mul(&inv(&s)).unwrap().mul(&g).unwrap()[(0, 0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = 1 + 2 x + u with x endogenous; instruments z1, z2 (overidentified).
    fn data(n: usize) -> (Matrix, Matrix, Vec<f64>, Vec<f64>) {
        let mut s = 99u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut x = Matrix::zeros(n, 2);
        let mut z = Matrix::zeros(n, 3);
        let mut y = vec![0.0; n];
        for t in 0..n {
            let (z1, z2, e, v) = (r(), r(), r(), r());
            let xv = z1 + 0.5 * z2 + 0.8 * e + 0.3 * v;
            x[(t, 0)] = xv;
            x[(t, 1)] = 1.0;
            z[(t, 0)] = z1;
            z[(t, 1)] = z2;
            z[(t, 2)] = 1.0;
            y[t] = 1.0 + 2.0 * xv + e * (1.0 + z1.abs());
        }
        (x, z, y, vec![1.0; n])
    }

    #[test]
    fn exactly_identified_estimators_coincide() {
        let (x, z, y, w) = data(80);
        let mut zj = Matrix::zeros(80, 2);
        for t in 0..80 {
            zj[(t, 0)] = z[(t, 0)];
            zj[(t, 1)] = 1.0;
        }
        let a = tsls(&x, &zj, &y, &w);
        let b = liml(&x, &zj, &y, &w, &[1]);
        let c = gmm(&x, &zj, &y, &w, &Moments::Robust, &Moments::Robust, false);
        assert!((b.kappa - 1.0).abs() < 1e-9);
        for j in 0..2 {
            assert!((a.b[j] - b.b[j]).abs() < 1e-9);
            assert!((a.b[j] - c.b[j]).abs() < 1e-9);
        }
        assert!(c.j.unwrap().abs() < 1e-12);
        // with an exactly identified model the robust GMM covariance is the robust 2SLS sandwich
        let v2 = kclass_vce(&a, &w, &Moments::Robust, 80.0);
        let vg = c.v_gmm.unwrap();
        for i in 0..2 {
            assert!((v2[(i, i)] - vg[(i, i)]).abs() < 1e-10 * v2[(i, i)]);
        }
    }

    #[test]
    fn overidentified_relations() {
        let (x, z, y, w) = data(200);
        let a = tsls(&x, &z, &y, &w);
        // 2SLS normal equations: X̂'u = 0
        for j in 0..2 {
            let s: f64 = (0..200).map(|t| a.xt[(t, j)] * a.resid[t]).sum();
            assert!(s.abs() < 1e-9);
        }
        // LIML κ >= 1, and the LIML estimate solves X'(I - κ M_Z)u = 0
        let l = liml(&x, &z, &y, &w, &[1]);
        assert!(l.kappa >= 1.0 - 1e-12);
        for j in 0..2 {
            let s: f64 = (0..200).map(|t| l.xt[(t, j)] * l.resid[t]).sum();
            assert!(s.abs() < 1e-8);
        }
        // GMM with homoskedastic weights is 2SLS; its J is the Sargan statistic
        let g = gmm(&x, &z, &y, &w, &Moments::Unadjusted, &Moments::Unadjusted, false);
        for j in 0..2 {
            assert!((g.b[j] - a.b[j]).abs() < 1e-9);
        }
        let (sargan, _) = sargan_basmann(&z, &a.resid, &w, 200.0, 1.0, 3.0);
        assert!((g.j.unwrap() - sargan).abs() < 1e-8 * sargan);
        // iterated GMM converges to a fixed point of the two-step map
        let it = gmm(&x, &z, &y, &w, &Moments::Robust, &Moments::Robust, true);
        let s = moment_cov(&z, &it.resid, &w, &Moments::Robust);
        let again = gmm_step(&x, &z, &y, &w, &s);
        for j in 0..2 {
            assert!((again[j] - it.b[j]).abs() < 1e-7);
        }
    }
}
