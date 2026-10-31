//! Distribution helpers for power analysis: normal-approximation power,
//! noncentral chi-squared and F distributions (Poisson mixtures).

use crate::dist;

/// Critical value of a z test.
pub fn zcrit(alpha: f64, onesided: bool) -> f64 {
    dist::invnormal(1.0 - if onesided { alpha } else { alpha / 2.0 })
}

/// Power of a test whose statistic is (D)/s0 with E[D] = d under the
/// alternative and sd(D) = s1 under the alternative (s0 under the null).
pub fn pw_norm(d: f64, s0: f64, s1: f64, alpha: f64, onesided: bool) -> f64 {
    let z = zcrit(alpha, onesided);
    if s1 <= 0.0 || !s1.is_finite() {
        return f64::NAN;
    }
    if onesided {
        dist::normal((d.abs() - z * s0) / s1)
    } else {
        dist::normal((d - z * s0) / s1) + dist::normal((-d - z * s0) / s1)
    }
}

/// Power of a z test with standardized noncentrality `ncp`.
pub fn z_power(ncp: f64, alpha: f64, onesided: bool) -> f64 {
    pw_norm(ncp, 1.0, 1.0, alpha, onesided)
}

/// Power of a t test with `df` degrees of freedom and noncentrality `ncp`.
pub fn t_power(df: f64, ncp: f64, alpha: f64, onesided: bool) -> f64 {
    if !(df > 0.0) {
        return f64::NAN;
    }
    if df > 1e6 {
        return z_power(ncp, alpha, onesided);
    }
    if onesided {
        let c = dist::invttail(df, alpha);
        1.0 - crate::smallstat::nct_cdf(c, df, ncp.abs())
    } else {
        let c = dist::invttail(df, alpha / 2.0);
        1.0 - crate::smallstat::nct_cdf(c, df, ncp) + crate::smallstat::nct_cdf(-c, df, ncp)
    }
}

fn pois_w(lam: f64, j: f64) -> f64 {
    if lam <= 0.0 {
        return if j == 0.0 { 1.0 } else { 0.0 };
    }
    (-lam + j * lam.ln() - dist::lngamma(j + 1.0)).exp()
}

/// Sum of Poisson(lam)-weighted terms f(j), from the mode outwards.
fn pois_mix(lam: f64, f: impl Fn(f64) -> f64) -> f64 {
    if lam <= 0.0 {
        return f(0.0);
    }
    let mode = lam.floor();
    let mut s = 0.0;
    let mut j = mode;
    loop {
        let w = pois_w(lam, j);
        s += w * f(j);
        if (w < 1e-15 && j > lam) || j > mode + 2000.0 {
            break;
        }
        j += 1.0;
    }
    let mut j = mode - 1.0;
    while j >= 0.0 {
        let w = pois_w(lam, j);
        s += w * f(j);
        if w < 1e-15 {
            break;
        }
        j -= 1.0;
    }
    s.clamp(0.0, 1.0)
}

/// Cumulative noncentral chi-squared distribution.
pub fn ncchi2(df: f64, ncp: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    pois_mix(ncp / 2.0, |j| dist::chi2(df + 2.0 * j, x))
}

/// Cumulative noncentral F distribution.
pub fn ncf(df1: f64, df2: f64, ncp: f64, f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let x = df1 * f / (df1 * f + df2);
    pois_mix(ncp / 2.0, |j| dist::ibeta(df1 / 2.0 + j, df2 / 2.0, x))
}

/// Power of an F test (df1, df2) with noncentrality `ncp`.
pub fn f_power(df1: f64, df2: f64, ncp: f64, alpha: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0) {
        return f64::NAN;
    }
    let c = dist::invftail(df1, df2, alpha);
    1.0 - ncf(df1, df2, ncp, c)
}

/// Power of a chi-squared test (df) with noncentrality `ncp`.
pub fn chi2_power(df: f64, ncp: f64, alpha: f64) -> f64 {
    let c = dist::invchi2tail(df, alpha);
    1.0 - ncchi2(df, ncp, c)
}

/// ∫ g(x) φ(x) dx by composite Simpson on [-9, 9].
pub fn normal_expect(g: impl Fn(f64) -> f64) -> f64 {
    let n = 600;
    let (a, b) = (-9.0, 9.0);
    let h = (b - a) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let x = a + i as f64 * h;
        let w = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * g(x) * dist::normalden(x);
    }
    s * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn central_limits() {
        assert!((ncchi2(3.0, 0.0, 7.8147) - 0.95).abs() < 1e-4);
        // ncp>0: mean of noncentral chi2 is df+ncp; median check by consistency
        let p = ncchi2(1.0, 4.0, 3.841459);
        let z = 1.959964;
        // CDF of a 1-df noncentral chi2 with ncp 4 = P(|Z + 2| < z)
        let exact = dist::normal(z - 2.0) - dist::normal(-z - 2.0);
        assert!((p - exact).abs() < 1e-6);
        let f = ncf(1.0, 1e7, 4.0, 3.841459);
        assert!((f - exact).abs() < 1e-4);
    }
}
