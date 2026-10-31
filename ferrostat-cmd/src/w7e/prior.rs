//! Prior distributions: log densities and supports.

use crate::dist;
use crate::matrix::Matrix;

const LN2PI: f64 = 1.8378770664093453;

/// Support of a parameter, used to pick the sampler's transformation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Support {
    Real,
    /// (lo, +inf)
    Lower(f64),
    /// (lo, hi)
    Interval(f64, f64),
    /// integers in [lo, hi] (hi may be +inf)
    Discrete(f64, f64),
}

pub fn ln_normal(x: f64, m: f64, v: f64) -> f64 {
    if v <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * (LN2PI + v.ln() + (x - m) * (x - m) / v)
}

/// gamma with shape `a`, scale `b`
pub fn ln_gamma(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 || b <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (a - 1.0) * x.ln() - x / b - dist::lngamma(a) - a * b.ln()
}

/// inverse gamma with shape `a`, scale `b`
pub fn ln_igamma(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 || b <= 0.0 {
        return f64::NEG_INFINITY;
    }
    a * b.ln() - dist::lngamma(a) - (a + 1.0) * x.ln() - b / x
}

pub fn ln_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 || a <= 0.0 || b <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + dist::lngamma(a + b) - dist::lngamma(a) - dist::lngamma(b)
}

pub fn ln_t(x: f64, m: f64, s2: f64, df: f64) -> f64 {
    if s2 <= 0.0 || df <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z2 = (x - m) * (x - m) / s2;
    dist::lngamma((df + 1.0) / 2.0) - dist::lngamma(df / 2.0) - 0.5 * (df * std::f64::consts::PI * s2).ln() - (df + 1.0) / 2.0 * (1.0 + z2 / df).ln()
}

/// Univariate prior families (arguments already evaluated).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Uni {
    Normal,
    T,
    Lognormal,
    Uniform,
    Gamma,
    IGamma,
    Exponential,
    Beta,
    Chi2,
    Cauchy,
    Laplace,
    Pareto,
    Jeffreys,
    Flat,
    Bernoulli,
    Geometric,
    Poisson,
    Index,
}

impl Uni {
    pub fn from_name(s: &str) -> Option<Uni> {
        Some(match s {
            "normal" => Uni::Normal,
            "t" => Uni::T,
            "lognormal" => Uni::Lognormal,
            "uniform" => Uni::Uniform,
            "gamma" => Uni::Gamma,
            "igamma" => Uni::IGamma,
            "exponential" => Uni::Exponential,
            "beta" => Uni::Beta,
            "chi2" => Uni::Chi2,
            "cauchy" => Uni::Cauchy,
            "laplace" => Uni::Laplace,
            "pareto" => Uni::Pareto,
            "jeffreys" => Uni::Jeffreys,
            "flat" => Uni::Flat,
            "bernoulli" => Uni::Bernoulli,
            "geometric" => Uni::Geometric,
            "poisson" => Uni::Poisson,
            "index" => Uni::Index,
            _ => return None,
        })
    }

    /// Number of arguments (None = variable, for index()).
    pub fn nargs(self) -> Option<usize> {
        Some(match self {
            Uni::Normal | Uni::Lognormal | Uni::Uniform | Uni::Gamma | Uni::IGamma | Uni::Beta | Uni::Cauchy | Uni::Laplace | Uni::Pareto => 2,
            Uni::T => 3,
            Uni::Exponential | Uni::Chi2 | Uni::Bernoulli | Uni::Geometric | Uni::Poisson => 1,
            Uni::Jeffreys | Uni::Flat => 0,
            Uni::Index => return None,
        })
    }

    pub fn proper(self) -> bool {
        !matches!(self, Uni::Jeffreys | Uni::Flat)
    }

    /// Support given (constant) arguments.
    pub fn support(self, a: &[f64]) -> Support {
        match self {
            Uni::Lognormal | Uni::Gamma | Uni::IGamma | Uni::Exponential | Uni::Chi2 | Uni::Jeffreys => Support::Lower(0.0),
            Uni::Pareto => Support::Lower(a.get(1).copied().unwrap_or(0.0)),
            Uni::Uniform => Support::Interval(a.first().copied().unwrap_or(0.0), a.get(1).copied().unwrap_or(1.0)),
            Uni::Beta => Support::Interval(0.0, 1.0),
            Uni::Bernoulli => Support::Discrete(0.0, 1.0),
            Uni::Geometric | Uni::Poisson => Support::Discrete(0.0, f64::INFINITY),
            Uni::Index => Support::Discrete(1.0, a.len() as f64),
            _ => Support::Real,
        }
    }

    pub fn ln_pdf(self, x: f64, a: &[f64]) -> f64 {
        let g = |k: usize| a.get(k).copied().unwrap_or(f64::NAN);
        let r = match self {
            Uni::Normal => ln_normal(x, g(0), g(1)),
            Uni::T => ln_t(x, g(0), g(1), g(2)),
            Uni::Lognormal => {
                if x <= 0.0 { f64::NEG_INFINITY } else { ln_normal(x.ln(), g(0), g(1)) - x.ln() }
            }
            Uni::Uniform => {
                if x >= g(0) && x <= g(1) && g(1) > g(0) { -(g(1) - g(0)).ln() } else { f64::NEG_INFINITY }
            }
            Uni::Gamma => ln_gamma(x, g(0), g(1)),
            Uni::IGamma => ln_igamma(x, g(0), g(1)),
            Uni::Exponential => {
                if x < 0.0 || g(0) <= 0.0 { f64::NEG_INFINITY } else { -g(0).ln() - x / g(0) }
            }
            Uni::Beta => ln_beta(x, g(0), g(1)),
            Uni::Chi2 => ln_gamma(x, g(0) / 2.0, 2.0),
            Uni::Cauchy => {
                let s = g(1);
                if s <= 0.0 { f64::NEG_INFINITY } else { -(std::f64::consts::PI * s * (1.0 + ((x - g(0)) / s).powi(2))).ln() }
            }
            Uni::Laplace => {
                let s = g(1);
                if s <= 0.0 { f64::NEG_INFINITY } else { -(2.0 * s).ln() - (x - g(0)).abs() / s }
            }
            Uni::Pareto => {
                let (sh, sc) = (g(0), g(1));
                if x < sc || sh <= 0.0 || sc <= 0.0 { f64::NEG_INFINITY } else { sh.ln() + sh * sc.ln() - (sh + 1.0) * x.ln() }
            }
            Uni::Jeffreys => {
                if x <= 0.0 { f64::NEG_INFINITY } else { -x.ln() }
            }
            Uni::Flat => 0.0,
            Uni::Bernoulli => {
                let p = g(0);
                if x == 1.0 { p.ln() } else if x == 0.0 { (1.0 - p).ln() } else { f64::NEG_INFINITY }
            }
            Uni::Geometric => {
                let p = g(0);
                if x < 0.0 || x.fract() != 0.0 { f64::NEG_INFINITY } else { p.ln() + x * (1.0 - p).ln() }
            }
            Uni::Poisson => {
                let m = g(0);
                if x < 0.0 || x.fract() != 0.0 || m <= 0.0 { f64::NEG_INFINITY } else { x * m.ln() - m - dist::lnfactorial(x) }
            }
            Uni::Index => {
                let k = x as usize;
                if x.fract() != 0.0 || k < 1 || k > a.len() { f64::NEG_INFINITY } else { a[k - 1].ln() }
            }
        };
        if r.is_nan() { f64::NEG_INFINITY } else { r }
    }
}

/// log density of a multivariate normal with covariance `s` (via Cholesky).
pub fn ln_mvnormal(x: &[f64], m: &[f64], s: &Matrix) -> f64 {
    let d = x.len();
    let Ok(l) = s.cholesky() else { return f64::NEG_INFINITY };
    // solve L z = x - m
    let mut z = vec![0.0; d];
    let mut logdet = 0.0;
    for i in 0..d {
        let mut v = x[i] - m[i];
        for j in 0..i {
            v -= l[(i, j)] * z[j];
        }
        if l[(i, i)] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        z[i] = v / l[(i, i)];
        logdet += l[(i, i)].ln();
    }
    -0.5 * d as f64 * LN2PI - logdet - 0.5 * z.iter().map(|v| v * v).sum::<f64>()
}

fn ln_mvgamma(d: usize, a: f64) -> f64 {
    let mut s = d as f64 * (d as f64 - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for j in 0..d {
        s += dist::lngamma(a - j as f64 / 2.0);
    }
    s
}

fn logdet_pd(m: &Matrix) -> Option<f64> {
    let l = m.cholesky().ok()?;
    let mut s = 0.0;
    for i in 0..m.rows {
        if l[(i, i)] <= 0.0 {
            return None;
        }
        s += 2.0 * l[(i, i)].ln();
    }
    Some(s)
}

/// Wishart(df, V) log density of a d x d matrix.
pub fn ln_wishart(x: &Matrix, df: f64, v: &Matrix) -> f64 {
    let d = x.rows;
    let (Some(ldx), Some(ldv)) = (logdet_pd(x), logdet_pd(v)) else { return f64::NEG_INFINITY };
    let Ok(vi) = v.inverse() else { return f64::NEG_INFINITY };
    let tr = vi.mul(x).map(|m| m.trace()).unwrap_or(f64::NAN);
    (df - d as f64 - 1.0) / 2.0 * ldx - tr / 2.0 - df * d as f64 / 2.0 * 2f64.ln() - df / 2.0 * ldv - ln_mvgamma(d, df / 2.0)
}

/// Inverse-Wishart(df, V) log density.
pub fn ln_iwishart(x: &Matrix, df: f64, v: &Matrix) -> f64 {
    let d = x.rows;
    let (Some(ldx), Some(ldv)) = (logdet_pd(x), logdet_pd(v)) else { return f64::NEG_INFINITY };
    let Ok(xi) = x.inverse() else { return f64::NEG_INFINITY };
    let tr = v.mul(&xi).map(|m| m.trace()).unwrap_or(f64::NAN);
    df / 2.0 * ldv - (df + d as f64 + 1.0) / 2.0 * ldx - tr / 2.0 - df * d as f64 / 2.0 * 2f64.ln() - ln_mvgamma(d, df / 2.0)
}

pub fn ln_dirichlet(x: &[f64], a: &[f64]) -> f64 {
    let s: f64 = x.iter().sum();
    if x.iter().any(|v| *v <= 0.0) || (s - 1.0).abs() > 1e-6 {
        return f64::NEG_INFINITY;
    }
    let sa: f64 = a.iter().sum();
    dist::lngamma(sa) - a.iter().map(|v| dist::lngamma(*v)).sum::<f64>() + x.iter().zip(a).map(|(x, a)| (a - 1.0) * x.ln()).sum::<f64>()
}
