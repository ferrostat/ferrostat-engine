//! Small helpers shared by the treatment-effect commands of this family:
//! coefficient tables from (label, estimate, standard error) rows, numeric
//! gradients for the delta method and just-identified GMM sandwiches.

use crate::choice_mixed::{row_z, sep, zcrit};
use crate::format::fmt_g;
use crate::matrix::Matrix;

/// One row of a z table: optional equation header shown above it.
pub struct ZRow {
    pub eq: Option<String>,
    pub label: String,
    pub b: f64,
    pub se: f64,
}

pub fn zrow(eq: Option<&str>, label: &str, b: f64, se: f64) -> ZRow {
    ZRow { eq: eq.map(String::from), label: label.to_string(), b, se }
}

/// Coefficient table with z statistics; `title` heads the label column and
/// `setitle` the standard-error column (e.g. "Robust").
pub fn ztable(title: &str, setitle: &str, rows: &[ZRow], level: f64) -> Vec<String> {
    let lw = 12;
    let crit = zcrit(level);
    let mut out = vec!["-".repeat(78)];
    if !setitle.is_empty() {
        out.push(format!("{:>lw$} | {:>22}", "", setitle));
    }
    out.push(format!("{:>lw$} | Coefficient  Std. err.       z    P>|z|     [{}% conf. interval]", crate::summarize::abbrev(title, lw), fmt_g(level, 4)));
    out.push(sep(lw));
    let mut last: Option<String> = None;
    for (k, r) in rows.iter().enumerate() {
        if r.eq.is_some() && r.eq != last {
            if k > 0 {
                out.push(sep(lw));
            }
            out.push(format!("{:<lw$} |", crate::summarize::abbrev(r.eq.as_deref().unwrap_or(""), lw)));
            last = r.eq.clone();
        }
        out.push(row_z(lw, &r.label, r.b, r.se, crit));
    }
    out.push("-".repeat(78));
    out
}

/// r(table)-style matrix (b, se, z, p, ll, ul) for the rows.
pub fn table_matrix(names: &[String], b: &[f64], se: &[f64], level: f64) -> Matrix {
    let k = b.len();
    let crit = zcrit(level);
    let mut m = Matrix::zeros(6, k);
    for j in 0..k {
        let z = b[j] / se[j];
        m[(0, j)] = b[j];
        m[(1, j)] = se[j];
        m[(2, j)] = z;
        m[(3, j)] = 2.0 * crate::dist::normal(-z.abs());
        m[(4, j)] = b[j] - crit * se[j];
        m[(5, j)] = b[j] + crit * se[j];
    }
    m.with_names(vec!["b".into(), "se".into(), "z".into(), "pvalue".into(), "ll".into(), "ul".into()], names.to_vec())
}

/// Central-difference Jacobian of `f` (m outputs) at `x`.
pub fn jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
    let f0 = f(x);
    let m = f0.len();
    let mut j = vec![vec![0.0; x.len()]; m];
    let mut y = x.to_vec();
    for c in 0..x.len() {
        let h = 1e-6 * x[c].abs().max(1.0);
        y[c] = x[c] + h;
        let a = f(&y);
        y[c] = x[c] - h;
        let b = f(&y);
        y[c] = x[c];
        for r in 0..m {
            j[r][c] = (a[r] - b[r]) / (2.0 * h);
        }
    }
    j
}

/// Delta-method covariance J V J'.
pub fn delta(j: &[Vec<f64>], v: &Matrix) -> Matrix {
    let m = j.len();
    let k = v.rows;
    let mut out = Matrix::zeros(m, m);
    for a in 0..m {
        for b in 0..m {
            let mut s = 0.0;
            for p in 0..k {
                if j[a][p] == 0.0 {
                    continue;
                }
                for q in 0..k {
                    s += j[a][p] * v[(p, q)] * j[b][q];
                }
            }
            out[(a, b)] = s;
        }
    }
    out
}

/// Just-identified GMM sandwich: `mom(theta)` returns per-observation
/// moment vectors (n x k); V = G^-1 S G^-T / n with G the Jacobian of the
/// mean moments and S their outer-product average (clustered by `cl`).
pub fn gmm_vce(mom: &dyn Fn(&[f64]) -> Vec<Vec<f64>>, theta: &[f64], cl: Option<&[f64]>) -> Matrix {
    let k = theta.len();
    let m0 = mom(theta);
    let n = m0.len() as f64;
    let mean = |t: &[f64]| -> Vec<f64> {
        let m = mom(t);
        let mut s = vec![0.0; k];
        for r in &m {
            for c in 0..k {
                s[c] += r[c];
            }
        }
        s.iter().map(|v| v / n).collect()
    };
    let g = jacobian(&mean, theta);
    let mut gm = Matrix::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            gm[(a, b)] = g[a][b];
        }
    }
    let gi = gm.inverse().unwrap_or_else(|_| Matrix::zeros(k, k));
    // meat
    let rows: Vec<Vec<f64>> = match cl {
        None => m0,
        Some(c) => {
            let mut keys: Vec<f64> = c.to_vec();
            keys.sort_by(f64::total_cmp);
            keys.dedup();
            let mut agg = vec![vec![0.0; k]; keys.len()];
            for (i, r) in m0.iter().enumerate() {
                let g = keys.binary_search_by(|v| v.total_cmp(&c[i])).unwrap();
                for a in 0..k {
                    agg[g][a] += r[a];
                }
            }
            agg
        }
    };
    let mut s = Matrix::zeros(k, k);
    for r in &rows {
        for a in 0..k {
            for b in 0..k {
                s[(a, b)] += r[a] * r[b];
            }
        }
    }
    let ng = rows.len() as f64;
    let adj = if cl.is_some() && ng > 1.0 { ng / (ng - 1.0) } else { 1.0 };
    let mut v = Matrix::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            let mut t = 0.0;
            for p in 0..k {
                for q in 0..k {
                    t += gi[(a, p)] * s[(p, q)] * gi[(b, q)];
                }
            }
            v[(a, b)] = t * adj / (n * n);
        }
    }
    v
}

/// Standard normal density.
pub fn phi(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Inverse Mills ratio phi/Phi computed stably.
pub fn mills(z: f64) -> f64 {
    let p = crate::dist::normal(z);
    if p > 1e-300 {
        phi(z) / p
    } else {
        -z
    }
}

/// Bivariate normal CDF that accepts infinite limits.
pub fn bvn(h: f64, k: f64, r: f64) -> f64 {
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        0.0
    } else if h == f64::INFINITY {
        crate::dist::normal(k)
    } else if k == f64::INFINITY {
        crate::dist::normal(h)
    } else {
        crate::choice_mixed::bvn(h, k, r)
    }
}
