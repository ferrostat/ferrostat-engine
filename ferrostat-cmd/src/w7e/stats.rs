//! Posterior summaries and convergence diagnostics of MCMC samples.

use crate::matrix::Matrix;

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { f64::NAN } else { x.iter().sum::<f64>() / x.len() as f64 }
}

pub fn var(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0)
}

/// Autocorrelation at lag k.
pub fn acf(x: &[f64], k: usize) -> f64 {
    let n = x.len();
    if k >= n {
        return 0.0;
    }
    let m = mean(x);
    let c0: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    if c0 <= 0.0 {
        return 0.0;
    }
    let ck: f64 = (0..n - k).map(|i| (x[i] - m) * (x[i + k] - m)).sum();
    ck / c0
}

/// Effective sample size: n / (1 + 2 sum of autocorrelations), summing
/// until the autocorrelation drops below 0.01 (at most 500 lags).
pub fn ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return n as f64;
    }
    if var(x) <= 0.0 {
        return n as f64;
    }
    let maxlag = 500.min(n - 1);
    let mut s = 0.0;
    for k in 1..=maxlag {
        let r = acf(x, k);
        if r < 0.01 {
            break;
        }
        s += r;
    }
    (n as f64 / (1.0 + 2.0 * s)).min(n as f64)
}

pub fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    // equal-tailed percentile: average of neighbours at exact positions
    let h = p * n as f64;
    let lo = h.floor() as usize;
    if (h - h.floor()).abs() < 1e-12 && lo > 0 && lo < n {
        (sorted[lo - 1] + sorted[lo]) / 2.0
    } else {
        sorted[(h.ceil() as usize).clamp(1, n) - 1]
    }
}

/// Shortest interval holding fraction `level` of the draws.
pub fn hpd(sorted: &[f64], level: f64) -> (f64, f64) {
    let n = sorted.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let w = ((level * n as f64).ceil() as usize).clamp(1, n);
    let mut best = (sorted[0], sorted[n - 1]);
    let mut bw = f64::INFINITY;
    for i in 0..=n - w {
        let d = sorted[i + w - 1] - sorted[i];
        if d < bw {
            bw = d;
            best = (sorted[i], sorted[i + w - 1]);
        }
    }
    best
}

#[derive(Clone, Debug)]
pub struct Summary {
    pub mean: f64,
    pub sd: f64,
    pub mcse: f64,
    pub median: f64,
    pub lo: f64,
    pub hi: f64,
    pub ess: f64,
}

pub fn summarize(x: &[f64], level: f64, use_hpd: bool, batch: usize) -> Summary {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let m = mean(x);
    let sd = var(x).sqrt();
    let e = ess(x);
    let mcse = if batch > 1 { batch_mcse(x, batch) } else { sd / e.max(1.0).sqrt() };
    let a = (1.0 - level / 100.0) / 2.0;
    let (lo, hi) = if use_hpd { hpd(&s, level / 100.0) } else { (quantile(&s, a), quantile(&s, 1.0 - a)) };
    Summary { mean: m, sd, mcse, median: quantile(&s, 0.5), lo, hi, ess: e }
}

/// MCSE by batch means.
pub fn batch_mcse(x: &[f64], b: usize) -> f64 {
    let nb = x.len() / b;
    if nb < 2 {
        return f64::NAN;
    }
    let means: Vec<f64> = (0..nb).map(|k| mean(&x[k * b..(k + 1) * b])).collect();
    (var(&means) / nb as f64).sqrt()
}

/// Gelman-Rubin potential scale reduction from several chains.
pub fn grubin(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    if n < 2 {
        return f64::NAN;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(&c[..n])).collect();
    let w = chains.iter().map(|c| var(&c[..n])).sum::<f64>() / m as f64;
    let bn = var(&means); // B / n
    let vhat = (n as f64 - 1.0) / n as f64 * w + bn * (1.0 + 1.0 / m as f64);
    if w <= 0.0 {
        return if vhat <= 0.0 { 1.0 } else { f64::INFINITY };
    }
    (vhat / w).sqrt()
}

/// Sample covariance matrix of the columns `cols` of `draws`.
pub fn covariance(draws: &[Vec<f64>], cols: &[usize]) -> Matrix {
    let k = cols.len();
    let n = draws.len() as f64;
    let means: Vec<f64> = cols.iter().map(|&c| draws.iter().map(|d| d[c]).sum::<f64>() / n).collect();
    let mut v = Matrix::zeros(k, k);
    for d in draws {
        for a in 0..k {
            for b in 0..=a {
                v[(a, b)] += (d[cols[a]] - means[a]) * (d[cols[b]] - means[b]);
            }
        }
    }
    for a in 0..k {
        for b in 0..=a {
            let x = v[(a, b)] / (n - 1.0).max(1.0);
            v[(a, b)] = x;
            v[(b, a)] = x;
        }
    }
    v
}

/// log determinant of a symmetric positive-definite matrix.
pub fn logdet(m: &Matrix) -> Option<f64> {
    let l = m.cholesky().ok()?;
    let mut s = 0.0;
    for i in 0..m.rows {
        if !(l[(i, i)] > 0.0) {
            return None;
        }
        s += 2.0 * l[(i, i)].ln();
    }
    Some(s)
}
