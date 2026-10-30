//! Small derivative-free maximizer (Nelder-Mead with restarts, then a
//! Newton polish with numerical derivatives) for low-dimensional
//! likelihoods such as variance components.

use crate::matrix::Matrix;

/// Maximize `f` from `x0`. Returns (argmax, max).
pub fn maximize(f: &dyn Fn(&[f64]) -> f64, x0: &[f64]) -> (Vec<f64>, f64) {
    let k = x0.len();
    if k == 0 {
        return (vec![], f(x0));
    }
    let g = |x: &[f64]| {
        let v = f(x);
        if v.is_finite() { -v } else { f64::INFINITY }
    };
    let mut best = x0.to_vec();
    let mut fbest = g(&best);
    for round in 0..4 {
        let step = if round == 0 { 0.5 } else { 0.1 };
        let mut simplex: Vec<Vec<f64>> = vec![best.clone()];
        for j in 0..k {
            let mut p = best.clone();
            p[j] += step * p[j].abs().max(1.0);
            simplex.push(p);
        }
        let mut fv: Vec<f64> = simplex.iter().map(|p| g(p)).collect();
        for _ in 0..2000 * k {
            let mut idx: Vec<usize> = (0..=k).collect();
            idx.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
            simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
            fv = idx.iter().map(|&i| fv[i]).collect();
            let spread = (fv[k] - fv[0]).abs();
            let size = simplex.iter().skip(1).map(|p| p.iter().zip(&simplex[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            if spread < 1e-12 * (1.0 + fv[0].abs()) && size < 1e-9 {
                break;
            }
            let cen: Vec<f64> = (0..k).map(|j| simplex[..k].iter().map(|p| p[j]).sum::<f64>() / k as f64).collect();
            let pt = |t: f64| -> Vec<f64> { (0..k).map(|j| cen[j] + t * (simplex[k][j] - cen[j])).collect() };
            let xr = pt(-1.0);
            let fr = g(&xr);
            if fr < fv[0] {
                let xe = pt(-2.0);
                let fe = g(&xe);
                if fe < fr {
                    simplex[k] = xe;
                    fv[k] = fe;
                } else {
                    simplex[k] = xr;
                    fv[k] = fr;
                }
            } else if fr < fv[k - 1] {
                simplex[k] = xr;
                fv[k] = fr;
            } else {
                let xc = if fr < fv[k] { pt(-0.5) } else { pt(0.5) };
                let fc = g(&xc);
                if fc < fv[k].min(fr) {
                    simplex[k] = xc;
                    fv[k] = fc;
                } else {
                    for i in 1..=k {
                        simplex[i] = (0..k).map(|j| simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])).collect();
                        fv[i] = g(&simplex[i]);
                    }
                }
            }
        }
        let i0 = (0..=k).min_by(|&a, &b| fv[a].total_cmp(&fv[b])).unwrap();
        let improved = fbest - fv[i0];
        if fv[i0] <= fbest {
            best = simplex[i0].clone();
            fbest = fv[i0];
        }
        if round > 0 && improved.abs() < 1e-10 {
            break;
        }
    }
    // Newton polish
    let ll = |x: &[f64]| -> Option<Vec<f64>> {
        let v = f(x);
        if v.is_finite() { Some(vec![v]) } else { None }
    };
    for _ in 0..20 {
        let (Some(gr), Some(h)) = (crate::ts_ml::gradient(&ll, &best), crate::ts_ml::neg_hessian(&ll, &best)) else { break };
        if h.cholesky().is_err() {
            break;
        }
        let Ok(hi) = h.inverse() else { break };
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * gr[j]).sum()).collect();
        let nx: Vec<f64> = best.iter().zip(&step).map(|(a, s)| a + s).collect();
        let nf = g(&nx);
        if nf <= fbest + 1e-12 {
            let done = (fbest - nf).abs() < 1e-12;
            best = nx;
            fbest = nf;
            if done {
                break;
            }
        } else {
            break;
        }
    }
    (best, -fbest)
}

/// Negative Hessian of `f` at `x` (None if not computable).
pub fn info(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Option<Matrix> {
    let ll = |t: &[f64]| -> Option<Vec<f64>> {
        let v = f(t);
        if v.is_finite() { Some(vec![v]) } else { None }
    };
    crate::ts_ml::neg_hessian(&ll, x)
}
