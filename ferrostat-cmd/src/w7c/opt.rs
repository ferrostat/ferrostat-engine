//! Numerical maximization shared by the latent-variable estimators.
//!
//! The objective is a scalar log likelihood; derivatives are central
//! differences. Two techniques: modified Newton (numerical Hessian, with
//! the eigenvalues of a non-concave Hessian reflected) for cheap
//! objectives, and BFGS for expensive ones. Linear constraints `R th = c`
//! are handled by optimizing over the null space of `R`.

use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;

pub type Obj<'a> = dyn Fn(&[f64]) -> Option<f64> + 'a;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tech {
    Newton,
    Bfgs,
}

pub struct Opts {
    pub maxiter: usize,
    pub tech: Tech,
    pub label: &'static str,
    pub tol: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { maxiter: 300, tech: Tech::Newton, label: "log likelihood", tol: 1e-8 }
    }
}

pub struct Fit {
    pub th: Vec<f64>,
    pub ll: f64,
    pub log: Vec<String>,
    pub iters: usize,
    pub converged: bool,
}

fn val(f: &Obj, th: &[f64]) -> f64 {
    if th.iter().any(|v| !v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    match f(th) {
        Some(v) if v.is_finite() => v,
        _ => f64::NEG_INFINITY,
    }
}

fn hstep(x: f64, rel: f64) -> f64 {
    rel * x.abs().max(1.0)
}

/// Central-difference gradient.
pub fn gradient(f: &Obj, th: &[f64]) -> Option<Vec<f64>> {
    let k = th.len();
    let f0 = val(f, th);
    let mut g = vec![0.0; k];
    let mut t = th.to_vec();
    for j in 0..k {
        let h = hstep(th[j], 1e-6);
        t[j] = th[j] + h;
        let a = val(f, &t);
        t[j] = th[j] - h;
        let b = val(f, &t);
        t[j] = th[j];
        g[j] = match (a.is_finite(), b.is_finite()) {
            (true, true) => (a - b) / (2.0 * h),
            (true, false) if f0.is_finite() => (a - f0) / h,
            (false, true) if f0.is_finite() => (f0 - b) / h,
            _ => return None,
        };
    }
    Some(g)
}

/// Negative Hessian by second differences of `f`.
pub fn neg_hessian(f: &Obj, th: &[f64]) -> Option<Matrix> {
    let k = th.len();
    let f0 = val(f, th);
    if !f0.is_finite() {
        return None;
    }
    let hs: Vec<f64> = th.iter().map(|x| hstep(*x, 1e-4)).collect();
    let mut t = th.to_vec();
    let mut ev = |d: &[(usize, f64)]| -> Option<f64> {
        for &(j, v) in d {
            t[j] += v;
        }
        let r = val(f, &t);
        for &(j, v) in d {
            t[j] -= v;
        }
        // undo rounding drift
        if r.is_finite() { Some(r) } else { None }
    };
    let mut h = Matrix::zeros(k, k);
    let mut fp = vec![0.0; k];
    let mut fm = vec![0.0; k];
    for i in 0..k {
        fp[i] = ev(&[(i, hs[i])])?;
        fm[i] = ev(&[(i, -hs[i])])?;
        h[(i, i)] = -(fp[i] - 2.0 * f0 + fm[i]) / (hs[i] * hs[i]);
    }
    for i in 0..k {
        for j in 0..i {
            let a = ev(&[(i, hs[i]), (j, hs[j])])?;
            let b = ev(&[(i, -hs[i]), (j, -hs[j])])?;
            // f(x+hi+hj) + f(x-hi-hj) - f(x+hi) - f(x-hi) - f(x+hj) - f(x-hj) + 2 f0
            let v = -(a + b - fp[i] - fm[i] - fp[j] - fm[j] + 2.0 * f0) / (2.0 * hs[i] * hs[j]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    Some(h)
}

/// Negative Hessian from differences of the gradient (cheaper when the
/// gradient is the expensive part: 2k gradient calls).
pub fn neg_hessian_g(g: &dyn Fn(&[f64]) -> Option<Vec<f64>>, th: &[f64]) -> Option<Matrix> {
    let k = th.len();
    let mut h = Matrix::zeros(k, k);
    let mut t = th.to_vec();
    let mut g0: Option<Vec<f64>> = None;
    for j in 0..k {
        let s = hstep(th[j], 1e-5);
        t[j] = th[j] + s;
        let a = g(&t);
        t[j] = th[j] - s;
        let b = g(&t);
        t[j] = th[j];
        let (a, b, d) = match (a, b) {
            (Some(a), Some(b)) => (a, b, 2.0 * s),
            (Some(a), None) => {
                if g0.is_none() {
                    g0 = g(th);
                }
                (a, g0.clone()?, s)
            }
            (None, Some(b)) => {
                if g0.is_none() {
                    g0 = g(th);
                }
                (g0.clone()?, b, s)
            }
            _ => return None,
        };
        for i in 0..k {
            h[(i, j)] = -(a[i] - b[i]) / d;
        }
    }
    for i in 0..k {
        for j in 0..i {
            let v = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    Some(h)
}

fn mv(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Inverse of a positive-definite version of `h` (eigenvalues reflected
/// and floored), and whether `h` itself was positive definite.
pub fn pd_inverse(h: &Matrix) -> (Matrix, bool) {
    let k = h.rows;
    if k == 0 {
        return (Matrix::zeros(0, 0), true);
    }
    let (ev, vecs) = h.sym_eigen();
    let mx = ev.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-12);
    let concave = ev.iter().all(|&e| e > 1e-10 * mx);
    let mut inv = Matrix::zeros(k, k);
    for (c, &e) in ev.iter().enumerate() {
        let e = e.abs().max(1e-8 * mx);
        for i in 0..k {
            for j in 0..k {
                inv[(i, j)] += vecs[(i, c)] * vecs[(j, c)] / e;
            }
        }
    }
    (inv, concave)
}

pub fn iter_line(i: usize, ll: f64, label: &str, nc: bool) -> String {
    format!("Iteration {i}:{}{label} = {}{}", if i < 10 { "  " } else { " " }, fmt_g(ll, 10), if nc { "  (not concave)" } else { "" })
}

/// Maximize `f` from `th0`.
pub fn maximize(f: &Obj, th0: Vec<f64>, o: &Opts) -> Result<Fit, String> {
    let k = th0.len();
    let mut th = th0;
    let mut ll = val(f, &th);
    if !ll.is_finite() {
        return Err(rc_err(1400, "initial values not feasible"));
    }
    let mut log = Vec::new();
    if k == 0 {
        log.push(iter_line(0, ll, o.label, false));
        return Ok(Fit { th, ll, log, iters: 0, converged: true });
    }
    let mut hinv = Matrix::identity(k);
    let mut g = gradient(f, &th).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"))?;
    let mut converged = false;
    let mut it = 0;
    let mut bfgs_init = false;
    let mut stall = 0;
    loop {
        let (dir, nc) = match o.tech {
            Tech::Newton => {
                let h = neg_hessian(f, &th).ok_or_else(|| rc_err(430, "Hessian could not be computed"))?;
                let (hi, pd) = pd_inverse(&h);
                (mv(&hi, &g), !pd)
            }
            Tech::Bfgs => {
                if !bfgs_init {
                    // scale the initial metric by a diagonal curvature estimate
                    let mut m = Matrix::zeros(k, k);
                    let f0 = ll;
                    for j in 0..k {
                        let s = hstep(th[j], 1e-4);
                        let mut t = th.clone();
                        t[j] += s;
                        let a = val(f, &t);
                        t[j] -= 2.0 * s;
                        let b = val(f, &t);
                        let c = -(a - 2.0 * f0 + b) / (s * s);
                        m[(j, j)] = if c.is_finite() && c > 1e-10 { 1.0 / c } else { 1.0 / (g[j].abs() + 1.0) };
                    }
                    hinv = m;
                    bfgs_init = true;
                }
                (mv(&hinv, &g), false)
            }
        };
        log.push(iter_line(it, ll, o.label, nc));
        let gnorm = dot(&g, &dir).abs();
        if it >= o.maxiter {
            break;
        }
        let mut lambda = 1.0;
        let mut acc = None;
        for _ in 0..50 {
            let cand: Vec<f64> = th.iter().zip(&dir).map(|(t, d)| t + lambda * d).collect();
            let lc = val(f, &cand);
            if lc.is_finite() && lc >= ll - 1e-12 * ll.abs().max(1.0) {
                acc = Some((cand, lc));
                break;
            }
            lambda *= 0.5;
        }
        let Some((nth, nll)) = acc else {
            if o.tech == Tech::Bfgs && stall == 0 {
                bfgs_init = false;
                stall += 1;
                continue;
            }
            converged = gnorm < 1e-4;
            break;
        };
        stall = 0;
        let ng = gradient(f, &nth).ok_or_else(|| rc_err(430, "could not calculate numerical derivatives"))?;
        if o.tech == Tech::Bfgs {
            let s: Vec<f64> = nth.iter().zip(&th).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g.iter().zip(&ng).map(|(a, b)| a - b).collect();
            let sy = dot(&s, &y);
            if sy > 1e-14 {
                let hy = mv(&hinv, &y);
                let yhy = dot(&y, &hy);
                for i in 0..k {
                    for j in 0..k {
                        hinv[(i, j)] += (sy + yhy) * s[i] * s[j] / (sy * sy) - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                    }
                }
            }
        }
        let db = th.iter().zip(&nth).map(|(a, b)| (a - b).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        let dl = (nll - ll).abs() / (ll.abs() + 1.0);
        th = nth;
        ll = nll;
        g = ng;
        it += 1;
        if (dl < o.tol && db < 1e-6 || db < 1e-10) && (gnorm < 1e-5 || dl < 1e-13) {
            // check scaled gradient at the new point
            log.push(iter_line(it, ll, o.label, false));
            converged = true;
            break;
        }
    }
    Ok(Fit { th, ll, log, iters: it, converged })
}

/// Polish a BFGS solution with a few Newton steps (gradient-difference Hessian).
pub fn polish(f: &Obj, fit: &mut Fit, steps: usize) {
    for _ in 0..steps {
        let gf = |t: &[f64]| gradient(f, t);
        let Some(g) = gradient(f, &fit.th) else { return };
        let Some(h) = neg_hessian_g(&gf, &fit.th) else { return };
        let (hi, _) = pd_inverse(&h);
        let dir = mv(&hi, &g);
        let mut lambda = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            let c: Vec<f64> = fit.th.iter().zip(&dir).map(|(a, d)| a + lambda * d).collect();
            let l = val(f, &c);
            if l.is_finite() && l >= fit.ll {
                fit.th = c;
                fit.ll = l;
                moved = true;
                break;
            }
            lambda *= 0.5;
        }
        if !moved || dot(&g, &dir).abs() < 1e-9 {
            break;
        }
    }
}

/// Linear constraints R th = c as an affine map th = t0 + N gamma.
pub struct Affine {
    pub t0: Vec<f64>,
    pub n: Matrix,
}

impl Affine {
    pub fn new(r: &[Vec<f64>], c: &[f64], k: usize) -> Result<Affine, String> {
        if r.is_empty() {
            return Ok(Affine { t0: vec![0.0; k], n: Matrix::identity(k) });
        }
        // eigen decomposition of R'R: null space = eigenvectors with ~0 eigenvalue
        let mut rtr = Matrix::zeros(k, k);
        let mut rtc = vec![0.0; k];
        for (row, ci) in r.iter().zip(c) {
            for i in 0..k {
                rtc[i] += row[i] * ci;
                for j in 0..k {
                    rtr[(i, j)] += row[i] * row[j];
                }
            }
        }
        let (ev, vecs) = rtr.sym_eigen();
        let mx = ev.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-300);
        let null: Vec<usize> = (0..k).filter(|&i| ev[i].abs() <= 1e-10 * mx).collect();
        // particular solution: pseudo-inverse
        let mut t0 = vec![0.0; k];
        for i in 0..k {
            if ev[i].abs() > 1e-10 * mx {
                let proj: f64 = (0..k).map(|j| vecs[(j, i)] * rtc[j]).sum::<f64>() / ev[i];
                for j in 0..k {
                    t0[j] += vecs[(j, i)] * proj;
                }
            }
        }
        for (row, ci) in r.iter().zip(c) {
            if (dot(row, &t0) - ci).abs() > 1e-6 * (ci.abs() + 1.0) {
                return Err(rc_err(412, "constraints are inconsistent"));
            }
        }
        let mut n = Matrix::zeros(k, null.len());
        for (a, &i) in null.iter().enumerate() {
            for j in 0..k {
                n[(j, a)] = vecs[(j, i)];
            }
        }
        Ok(Affine { t0, n })
    }
    pub fn full(&self, g: &[f64]) -> Vec<f64> {
        let mut t = self.t0.clone();
        for i in 0..t.len() {
            for (a, gv) in g.iter().enumerate() {
                t[i] += self.n[(i, a)] * gv;
            }
        }
        t
    }
    pub fn reduce(&self, th: &[f64]) -> Vec<f64> {
        (0..self.n.cols).map(|a| (0..th.len()).map(|i| self.n[(i, a)] * (th[i] - self.t0[i])).sum()).collect()
    }
    /// V_th = N V_g N'
    pub fn expand_v(&self, vg: &Matrix) -> Matrix {
        let k = self.n.rows;
        let r = self.n.cols;
        let mut tmp = Matrix::zeros(k, r);
        for i in 0..k {
            for b in 0..r {
                tmp[(i, b)] = (0..r).map(|a| self.n[(i, a)] * vg[(a, b)]).sum();
            }
        }
        let mut v = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                v[(i, j)] = (0..r).map(|b| tmp[(i, b)] * self.n[(j, b)]).sum();
            }
        }
        v
    }
    pub fn is_identity(&self) -> bool {
        self.n.rows == self.n.cols && self.t0.iter().all(|v| *v == 0.0)
    }
}

/// Jacobian of a vector function by central differences (rows: outputs).
pub fn jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, th: &[f64]) -> Matrix {
    let k = th.len();
    let f0 = f(th);
    let mut j = Matrix::zeros(f0.len(), k);
    let mut t = th.to_vec();
    for c in 0..k {
        let h = hstep(th[c], 1e-6);
        t[c] = th[c] + h;
        let a = f(&t);
        t[c] = th[c] - h;
        let b = f(&t);
        t[c] = th[c];
        for r in 0..f0.len() {
            j[(r, c)] = (a[r] - b[r]) / (2.0 * h);
        }
    }
    j
}

/// J V J'
pub fn sandwich_jv(j: &Matrix, v: &Matrix) -> Matrix {
    let jv = j.mul(v).unwrap_or_else(|_| Matrix::zeros(j.rows, v.cols));
    jv.mul(&j.transpose()).unwrap_or_else(|_| Matrix::zeros(j.rows, j.rows))
}
