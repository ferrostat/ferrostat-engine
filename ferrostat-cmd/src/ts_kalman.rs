//! Linear Gaussian state-space models and the Kalman filter used by
//! `sspace`, `dfactor` and `ucm`.
//!
//!   state:        s_t = T s_{t-1} + c_t + R e_t,   Var(e) = Q
//!   observation:  y_t = Z s_t + d_t + v_t,          Var(v) = H
//!
//! Stationary states start from their unconditional distribution; the
//! states flagged as nonstationary get a diffuse prior (large variance)
//! and the likelihood contributions of the first observations that only
//! identify the diffuse part are dropped.

use crate::matrix::Matrix;
use crate::ts_ml::LN2PI;

#[derive(Clone, Debug)]
pub struct Ssm {
    pub m: usize,
    pub p: usize,
    pub t: Matrix,
    pub rqr: Matrix,
    pub z: Matrix,
    pub h: Matrix,
    /// state intercept per period (m) and observation intercept per period (p)
    pub c: Vec<Vec<f64>>,
    pub d: Vec<Vec<f64>>,
    pub diffuse: Vec<bool>,
}

pub struct KfOut {
    pub ll: Vec<f64>,
    /// one-step predictions of y (p per period) and their variances
    pub yhat: Vec<Vec<f64>>,
    pub fvar: Vec<Vec<f64>>,
    /// filtered states and one-step predicted states
    pub filt: Vec<Vec<f64>>,
    pub pred: Vec<Vec<f64>>,
}

fn mm(a: &Matrix, b: &Matrix) -> Matrix {
    a.mul(b).expect("conformable")
}

/// Stationary covariance solving P = T P T' + Qc (doubling); None if explosive.
pub fn stationary_cov(t: &Matrix, qc: &Matrix) -> Option<Matrix> {
    let mut a = t.clone();
    let mut p = qc.clone();
    for _ in 0..80 {
        let ap = mm(&a, &p);
        p = p.add(&mm(&ap, &a.transpose())).ok()?;
        a = mm(&a, &a);
        let norm = a.data.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        if !norm.is_finite() || p.data.iter().any(|v| !v.is_finite() || v.abs() > 1e14) {
            return None;
        }
        if norm < 1e-14 {
            return Some(p);
        }
    }
    None
}

/// Run the filter over y (None = missing entry). Returns None if a
/// stationary block is explosive or a variance is not positive.
pub fn kfilter(s: &Ssm, y: &[Vec<Option<f64>>]) -> Option<KfOut> {
    let m = s.m;
    let n = y.len();
    // initial state: stationary part from the unconditional distribution
    let stat: Vec<usize> = (0..m).filter(|&i| !s.diffuse[i]).collect();
    let mut p = Matrix::zeros(m, m);
    let mut a = vec![0.0; m];
    let kappa = 1e7;
    if !stat.is_empty() {
        let k = stat.len();
        let mut ts = Matrix::zeros(k, k);
        let mut qs = Matrix::zeros(k, k);
        for (i, &si) in stat.iter().enumerate() {
            for (j, &sj) in stat.iter().enumerate() {
                ts[(i, j)] = s.t[(si, sj)];
                qs[(i, j)] = s.rqr[(si, sj)];
            }
        }
        let ps = stationary_cov(&ts, &qs)?;
        // unconditional mean (I - T)^{-1} c of the first period
        let c0 = s.c.first().cloned().unwrap_or(vec![0.0; m]);
        if c0.iter().any(|v| *v != 0.0) {
            let mut imt = Matrix::identity(k);
            for i in 0..k {
                for j in 0..k {
                    imt[(i, j)] -= ts[(i, j)];
                }
            }
            let inv = imt.inverse().ok()?;
            for i in 0..k {
                a[stat[i]] = (0..k).map(|j| inv[(i, j)] * c0[stat[j]]).sum();
            }
        }
        for (i, &si) in stat.iter().enumerate() {
            for (j, &sj) in stat.iter().enumerate() {
                p[(si, sj)] = ps[(i, j)];
            }
        }
    }
    let nd = s.diffuse.iter().filter(|d| **d).count();
    for i in 0..m {
        if s.diffuse[i] {
            p[(i, i)] = kappa;
        }
    }
    let mut out = KfOut { ll: Vec::new(), yhat: Vec::with_capacity(n), fvar: Vec::with_capacity(n), filt: Vec::with_capacity(n), pred: Vec::with_capacity(n) };
    let mut seen = 0usize;
    for t in 0..n {
        if t > 0 {
            // predict: a = T a + c_t; P = T P T' + RQR'
            let na: Vec<f64> = (0..m).map(|i| (0..m).map(|j| s.t[(i, j)] * a[j]).sum::<f64>() + s.c.get(t).map_or(0.0, |c| c[i])).collect();
            a = na;
            p = mm(&mm(&s.t, &p), &s.t.transpose()).add(&s.rqr).ok()?;
        }
        out.pred.push(a.clone());
        let obs: Vec<usize> = (0..s.p).filter(|&i| y[t][i].is_some()).collect();
        let dt = s.d.get(t).cloned().unwrap_or(vec![0.0; s.p]);
        let yh: Vec<f64> = (0..s.p).map(|i| (0..m).map(|j| s.z[(i, j)] * a[j]).sum::<f64>() + dt[i]).collect();
        // full prediction variance (for reporting)
        let zp = mm(&s.z, &p);
        let fv: Vec<f64> = (0..s.p).map(|i| (0..m).map(|j| zp[(i, j)] * s.z[(i, j)]).sum::<f64>() + s.h[(i, i)]).collect();
        out.yhat.push(yh.clone());
        out.fvar.push(fv);
        if obs.is_empty() {
            out.filt.push(a.clone());
            continue;
        }
        let k = obs.len();
        let mut f = Matrix::zeros(k, k);
        for (i, &oi) in obs.iter().enumerate() {
            for (j, &oj) in obs.iter().enumerate() {
                f[(i, j)] = (0..m).map(|l| zp[(oi, l)] * s.z[(oj, l)]).sum::<f64>() + s.h[(oi, oj)];
            }
        }
        let fi = f.inverse().ok()?;
        let det = f.det();
        if !(det > 0.0) || !det.is_finite() {
            return None;
        }
        let v: Vec<f64> = obs.iter().map(|&o| y[t][o].unwrap() - yh[o]).collect();
        let fiv: Vec<f64> = (0..k).map(|i| (0..k).map(|j| fi[(i, j)] * v[j]).sum()).collect();
        let quad: f64 = v.iter().zip(&fiv).map(|(a, b)| a * b).sum();
        if seen >= nd {
            out.ll.push(-0.5 * (k as f64 * LN2PI + det.ln() + quad));
        }
        seen += k;
        // update: a += P Z' F^{-1} v ; P -= P Z' F^{-1} Z P
        let pzt: Vec<Vec<f64>> = (0..m).map(|i| obs.iter().map(|&o| zp[(o, i)]).collect()).collect();
        for i in 0..m {
            a[i] += (0..k).map(|j| pzt[i][j] * fiv[j]).sum::<f64>();
        }
        let mut kg = Matrix::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                kg[(i, j)] = (0..k).map(|l| pzt[i][l] * fi[(l, j)]).sum();
            }
        }
        let mut np = p.clone();
        for i in 0..m {
            for j in 0..m {
                np[(i, j)] -= (0..k).map(|l| kg[(i, l)] * pzt[j][l]).sum::<f64>();
            }
        }
        // symmetrise
        for i in 0..m {
            for j in 0..i {
                let v = 0.5 * (np[(i, j)] + np[(j, i)]);
                np[(i, j)] = v;
                np[(j, i)] = v;
            }
        }
        p = np;
        out.filt.push(a.clone());
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_matches_closed_form() {
        // y_t = s_t, s_t = .5 s_{t-1} + e_t, var 1
        let s = Ssm { m: 1, p: 1, t: Matrix::from_rows(&[vec![0.5]]), rqr: Matrix::from_rows(&[vec![1.0]]), z: Matrix::from_rows(&[vec![1.0]]), h: Matrix::zeros(1, 1), c: vec![], d: vec![], diffuse: vec![false] };
        let y = vec![vec![Some(1.0)], vec![Some(0.2)]];
        let o = kfilter(&s, &y).unwrap();
        let v0 = 1.0 / 0.75;
        assert!((o.ll[0] + 0.5 * (LN2PI + f64::ln(v0) + 1.0 / v0)).abs() < 1e-10);
        assert!((o.ll[1] + 0.5 * (LN2PI + (0.2f64 - 0.5).powi(2))).abs() < 1e-10);
    }
}
