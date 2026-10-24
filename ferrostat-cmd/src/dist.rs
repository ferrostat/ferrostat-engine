//! Probability distribution functions (cumulative, tail, density, inverse).
//!
//! Built on the regularised incomplete gamma/beta functions; inverses are
//! found by safeguarded bisection on the cumulative function. Invalid
//! arguments return system missing, as in the expression language.

use ferrostat_core::missing::SYSMISS;
use statrs::function::beta::{beta_reg, ln_beta};
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::{digamma as sr_digamma, gamma_lr, gamma_ur, ln_gamma};

const M: f64 = SYSMISS;

fn ok(v: f64) -> f64 {
    if v.is_finite() { v } else { M }
}

/// Find x in [lo, hi] with f(x) = p for increasing f (bisection, ~1e-15).
fn invert_increasing(f: impl Fn(f64) -> f64, p: f64, mut lo: f64, mut hi: f64) -> f64 {
    // expand upper bound if needed
    let mut guard = 0;
    while f(hi) < p && guard < 2000 {
        lo = hi;
        hi *= 2.0;
        guard += 1;
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if f(mid) < p { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

fn invert_decreasing(f: impl Fn(f64) -> f64, p: f64, lo: f64, hi: f64) -> f64 {
    invert_increasing(|x| -f(x), -p, lo, hi)
}

// ---- normal -------------------------------------------------------------

/// Complementary error function to near machine precision: a positive-term
/// series for small arguments and a continued fraction in the tail.
pub fn erfc_precise(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_precise(-x);
    }
    if x < 3.0 {
        // erf(x) = 2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (1*3*...*(2n+1))
        let mut term = x;
        let mut sum = x;
        let x2 = x * x;
        let mut n = 0.0;
        while term > sum * 1e-17 {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        let erf = 2.0 / std::f64::consts::PI.sqrt() * (-x2).exp() * sum;
        return 1.0 - erf;
    }
    // continued fraction (modified Lentz) for erfc(x) * sqrt(pi) * e^{x^2}
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for k in 1..300 {
        let a = k as f64 / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = x + a / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / (f * std::f64::consts::PI.sqrt())
}

pub fn normal(z: f64) -> f64 {
    0.5 * erfc_precise(-z / std::f64::consts::SQRT_2)
}

pub fn normalden(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

pub fn lnnormalden(z: f64) -> f64 {
    -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

pub fn lnnormal(z: f64) -> f64 {
    normal(z).ln()
}

pub fn invnormal(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return M;
    }
    let mut x = -std::f64::consts::SQRT_2 * erfc_inv(2.0 * p);
    // polish with Newton steps on the precise CDF
    for _ in 0..3 {
        let d = normalden(x);
        if d <= 0.0 || !x.is_finite() {
            break;
        }
        let q = if x > 0.0 { -(normal(-x) - (1.0 - p)) } else { normal(x) - p };
        x -= q / d;
    }
    ok(x)
}

// ---- Student t ------------------------------------------------------------

/// Upper-tail probability P(T > t).
pub fn ttail(df: f64, t: f64) -> f64 {
    if df <= 0.0 {
        return M;
    }
    let x = df / (df + t * t);
    let half = 0.5 * beta_reg(df / 2.0, 0.5, x);
    if t >= 0.0 { half } else { 1.0 - half }
}

pub fn t(df: f64, t: f64) -> f64 {
    if df <= 0.0 { M } else { 1.0 - ttail(df, t) }
}

pub fn tden(df: f64, t: f64) -> f64 {
    if df <= 0.0 {
        return M;
    }
    (ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0) - 0.5 * (df * std::f64::consts::PI).ln()
        - (df + 1.0) / 2.0 * (1.0 + t * t / df).ln())
    .exp()
}

pub fn invttail(df: f64, p: f64) -> f64 {
    if df <= 0.0 || !(p > 0.0 && p < 1.0) {
        return M;
    }
    if p > 0.5 {
        return -invttail(df, 1.0 - p);
    }
    invert_decreasing(|x| ttail(df, x), p, 0.0, 10.0)
}

pub fn invt(df: f64, p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) { M } else { invttail(df, 1.0 - p) }
}

// ---- chi-squared ----------------------------------------------------------

pub fn chi2(df: f64, x: f64) -> f64 {
    if df <= 0.0 {
        return M;
    }
    if x <= 0.0 { 0.0 } else { gamma_lr(df / 2.0, x / 2.0) }
}

pub fn chi2tail(df: f64, x: f64) -> f64 {
    if df <= 0.0 {
        return M;
    }
    if x <= 0.0 { 1.0 } else { gamma_ur(df / 2.0, x / 2.0) }
}

pub fn chi2den(df: f64, x: f64) -> f64 {
    gammaden(df / 2.0, 2.0, 0.0, x)
}

pub fn invchi2(df: f64, p: f64) -> f64 {
    if df <= 0.0 || !(0.0..1.0).contains(&p) {
        return M;
    }
    invert_increasing(|x| chi2(df, x), p, 0.0, df.max(1.0) * 4.0)
}

pub fn invchi2tail(df: f64, p: f64) -> f64 {
    if df <= 0.0 || !(p > 0.0 && p <= 1.0) {
        return M;
    }
    invert_decreasing(|x| chi2tail(df, x), p, 0.0, df.max(1.0) * 4.0)
}

// ---- F --------------------------------------------------------------------

pub fn f_cdf(df1: f64, df2: f64, f: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 {
        return M;
    }
    if f <= 0.0 { 0.0 } else { beta_reg(df1 / 2.0, df2 / 2.0, df1 * f / (df1 * f + df2)) }
}

pub fn ftail(df1: f64, df2: f64, f: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 {
        return M;
    }
    if f <= 0.0 { 1.0 } else { beta_reg(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f)) }
}

pub fn fden(df1: f64, df2: f64, f: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 || f < 0.0 {
        return M;
    }
    if f == 0.0 {
        return if df1 < 2.0 { M } else if df1 == 2.0 { 1.0 } else { 0.0 };
    }
    (0.5 * (df1 * (df1 * f).ln() + df2 * df2.ln() - (df1 + df2) * (df1 * f + df2).ln())
        - f.ln()
        - ln_beta(df1 / 2.0, df2 / 2.0))
    .exp()
}

pub fn invf(df1: f64, df2: f64, p: f64) -> f64 {
    if !(0.0..1.0).contains(&p) {
        return M;
    }
    invert_increasing(|x| f_cdf(df1, df2, x), p, 0.0, 10.0)
}

pub fn invftail(df1: f64, df2: f64, p: f64) -> f64 {
    if !(p > 0.0 && p <= 1.0) {
        return M;
    }
    invert_decreasing(|x| ftail(df1, df2, x), p, 0.0, 10.0)
}

// ---- gamma / beta -----------------------------------------------------------

pub fn gammap(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 { M } else if x == 0.0 { 0.0 } else { gamma_lr(a, x) }
}

pub fn gammaptail(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 { M } else if x == 0.0 { 1.0 } else { gamma_ur(a, x) }
}

pub fn invgammap(a: f64, p: f64) -> f64 {
    if a <= 0.0 || !(0.0..1.0).contains(&p) {
        return M;
    }
    invert_increasing(|x| gammap(a, x), p, 0.0, a.max(1.0) * 4.0)
}

pub fn invgammaptail(a: f64, p: f64) -> f64 {
    if a <= 0.0 || !(p > 0.0 && p <= 1.0) {
        return M;
    }
    invert_decreasing(|x| gammaptail(a, x), p, 0.0, a.max(1.0) * 4.0)
}

pub fn gammaden(a: f64, b: f64, g: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return M;
    }
    let z = x - g;
    if z < 0.0 {
        return 0.0;
    }
    ((a - 1.0) * z.ln() - z / b - ln_gamma(a) - a * b.ln()).exp()
}

pub fn ibeta(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&x) { M } else { beta_reg(a, b, x) }
}

pub fn ibetatail(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&x) { M } else { beta_reg(b, a, 1.0 - x) }
}

pub fn invibeta(a: f64, b: f64, p: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return M;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_reg(a, b, mid) < p { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

pub fn betaden(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return M;
    }
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(a, b)).exp()
}

pub fn lngamma(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 { M } else { ok(ln_gamma(x)) }
}

pub fn digamma(x: f64) -> f64 {
    ok(sr_digamma(x))
}

pub fn trigamma(x: f64) -> f64 {
    // recurrence to x >= 6, then asymptotic series
    let mut x = x;
    let mut acc = 0.0;
    while x < 6.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    acc + 1.0 / x + x2 / 2.0
        + (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 * (1.0 / 42.0 - x2 * (1.0 / 30.0)))) / (x * x * x)
}

pub fn lnfactorial(n: f64) -> f64 {
    if n < 0.0 || n.fract() != 0.0 { M } else { ln_gamma(n + 1.0) }
}

pub fn comb(n: f64, k: f64) -> f64 {
    if n.fract() != 0.0 || k.fract() != 0.0 || n < 0.0 {
        return M;
    }
    if k < 0.0 || k > n {
        return 0.0;
    }
    (lnfactorial(n) - lnfactorial(k) - lnfactorial(n - k)).exp().round()
}

// ---- discrete ---------------------------------------------------------------

pub fn binomialp(n: f64, k: f64, p: f64) -> f64 {
    if n < 0.0 || !(0.0..=1.0).contains(&p) || n.fract() != 0.0 {
        return M;
    }
    if k < 0.0 || k > n || k.fract() != 0.0 {
        return 0.0;
    }
    if p == 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    (lnfactorial(n) - lnfactorial(k) - lnfactorial(n - k) + k * p.ln() + (n - k) * (1.0 - p).ln()).exp()
}

/// P(X <= k)
pub fn binomial(n: f64, k: f64, p: f64) -> f64 {
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return M;
    }
    let k = k.floor();
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    beta_reg(n - k, k + 1.0, 1.0 - p)
}

/// P(X >= k)
pub fn binomialtail(n: f64, k: f64, p: f64) -> f64 {
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return M;
    }
    let k = k.ceil();
    if k <= 0.0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    beta_reg(k, n - k + 1.0, p)
}

pub fn poissonp(m: f64, k: f64) -> f64 {
    if m <= 0.0 {
        return M;
    }
    if k < 0.0 || k.fract() != 0.0 {
        return 0.0;
    }
    (k * m.ln() - m - lnfactorial(k)).exp()
}

/// P(X <= k)
pub fn poisson(m: f64, k: f64) -> f64 {
    if m <= 0.0 {
        return M;
    }
    let k = k.floor();
    if k < 0.0 { 0.0 } else { gamma_ur(k + 1.0, m) }
}

/// P(X >= k)
pub fn poissontail(m: f64, k: f64) -> f64 {
    if m <= 0.0 {
        return M;
    }
    let k = k.ceil();
    if k <= 0.0 { 1.0 } else { gamma_lr(k, m) }
}

/// Standard logistic CDF.
pub fn invlogit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn logit(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) { M } else { (p / (1.0 - p)).ln() }
}

/// Bivariate normal CDF (Drezner & Wesolowsky 1990 algorithm).
pub fn binormal(h: f64, k: f64, r: f64) -> f64 {
    if !(-1.0..=1.0).contains(&r) {
        return M;
    }
    let x = [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992];
    let w = [0.018854042, 0.038088059, 0.0452707394, 0.038088059, 0.018854042];
    if r.abs() < 0.7 {
        let hs = (h * h + k * k) / 2.0;
        let asr = r.asin();
        let mut s = 0.0;
        for i in 0..5 {
            let sn = (asr * x[i]).sin();
            s += w[i] * ((sn * h * k - hs) / (1.0 - sn * sn)).exp();
        }
        return s * asr + normal(h) * normal(k);
    }
    // high correlation: numeric integration of the conditional
    let n = 2000;
    let lo = -10.0f64;
    let up = h.min(10.0);
    if up <= lo {
        return 0.0;
    }
    let step = (up - lo) / n as f64;
    let sd = (1.0 - r * r).max(1e-300).sqrt();
    let mut s = 0.0;
    for i in 0..n {
        let z = lo + (i as f64 + 0.5) * step;
        s += normalden(z) * normal((k - r * z) / sd) * step;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn reference_values() {
        close(normal(1.96), 0.9750021048517795, 1e-12);
        close(invnormal(0.975), 1.959963984540054, 1e-12);
        close(ttail(10.0, 2.228138851986), 0.025, 1e-9);
        close(invttail(10.0, 0.025), 2.228138851986, 1e-8);
        close(chi2tail(1.0, 3.841458820694), 0.05, 1e-9);
        close(invchi2tail(1.0, 0.05), 3.841458820694, 1e-8);
        close(ftail(2.0, 9.0, 4.256494729), 0.05, 1e-8);
        close(binomial(10.0, 3.0, 0.5), 176.0 / 1024.0, 1e-12);
        close(binomialtail(10.0, 8.0, 0.5), 56.0 / 1024.0, 1e-12);
        close(poisson(2.0, 1.0), 3.0 * (-2.0f64).exp(), 1e-12);
        close(comb(5.0, 2.0), 10.0, 1e-12);
        close(binormal(0.0, 0.0, 0.5), 1.0 / 3.0, 1e-7);
    }
}
