//! Identification diagnostics for linear IV models:
//!
//! * Kleibergen–Paap (2006, J. Econometrics 133) rk statistic for the null
//!   that the first-stage coefficient matrix of the excluded instruments has
//!   rank K1 - 1 (underidentification), in LM and Wald forms;
//! * Cragg–Donald (1993) minimum-eigenvalue Wald F statistic (weak
//!   identification under i.i.d. errors) and the robust KP rk Wald F.
//!
//! All inputs are the endogenous regressors and excluded instruments with the
//! included exogenous regressors already partialled out.

use crate::matrix::Matrix;

#[derive(Debug, Clone, Copy, Default)]
pub struct IdStats {
    /// KP rk LM (Anderson canonical-correlation LM when errors are i.i.d.)
    pub lm: f64,
    pub df: f64,
    /// KP rk Wald chi-squared (same degrees of freedom)
    pub wald: f64,
    /// Cragg–Donald Wald F
    pub cd_f: f64,
    /// KP rk Wald F (the robust counterpart of `cd_f`)
    pub kp_f: f64,
}

/// Small-sample settings for the F versions.
pub struct IdDof {
    /// observations net of absorbed fixed effects (N - dofminus)
    pub n: f64,
    /// number of instruments, included exogenous plus excluded (L)
    pub l: f64,
    /// number of clusters, when cluster-robust
    pub clusters: Option<f64>,
    /// heteroskedasticity-robust (or cluster) covariance
    pub robust: bool,
}

/// Matrix with the given columns.
pub fn columns(cols: &[Vec<f64>]) -> Matrix {
    let n = cols.first().map_or(0, |c| c.len());
    let mut m = Matrix::zeros(n, cols.len());
    for (j, c) in cols.iter().enumerate() {
        for t in 0..n {
            m[(t, j)] = c[t];
        }
    }
    m
}

/// Symmetric matrix power `a^p` through the eigen decomposition.
fn sym_pow(a: &Matrix, p: f64) -> Matrix {
    let (vals, vecs) = a.sym_eigen();
    let k = a.rows;
    let mut out = Matrix::zeros(k, k);
    for (e, &lam) in vals.iter().enumerate() {
        let w = if lam > 0.0 { lam.powf(p) } else { 0.0 };
        for i in 0..k {
            for j in 0..k {
                out[(i, j)] += w * vecs[(i, e)] * vecs[(j, e)];
            }
        }
    }
    out
}

/// Kronecker product a ⊗ b.
fn kron(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = Matrix::zeros(a.rows * b.rows, a.cols * b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            for r in 0..b.rows {
                for c in 0..b.cols {
                    m[(i * b.rows + r, j * b.cols + c)] = a[(i, j)] * b[(r, c)];
                }
            }
        }
    }
    m
}

/// Column-stacked vec(m).
fn vec_of(m: &Matrix) -> Vec<f64> {
    (0..m.cols).flat_map(|j| (0..m.rows).map(move |i| m[(i, j)])).collect()
}

/// Eigenvectors sorted by decreasing eigenvalue.
fn sorted_eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    let (vals, vecs) = a.sym_eigen();
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&x, &y| vals[y].total_cmp(&vals[x]));
    let mut v = Matrix::zeros(a.rows, a.cols);
    for (new, &old) in idx.iter().enumerate() {
        for i in 0..a.rows {
            v[(i, new)] = vecs[(i, old)];
        }
    }
    (idx.iter().map(|&i| vals[i]).collect(), v)
}

/// Covariance of vec(Π̂) given first-stage residuals `e` (n × K1).
fn vec_pi_cov(z: &Matrix, zzi: &Matrix, e: &Matrix, cluster: Option<&[f64]>, robust: bool) -> Matrix {
    let (n, l1, k1) = (z.rows, z.cols, e.cols);
    let dim = l1 * k1;
    let ik = Matrix::identity(k1);
    let a = kron(&ik, zzi);
    if !robust {
        let s = e.transpose().mul(e).unwrap().scale(1.0 / n as f64);
        return kron(&s, zzi);
    }
    let mut meat = Matrix::zeros(dim, dim);
    let mut add = |g: &[f64]| {
        for p in 0..dim {
            for q in 0..dim {
                meat[(p, q)] += g[p] * g[q];
            }
        }
    };
    let score = |t: usize, g: &mut [f64]| {
        for k in 0..k1 {
            for l in 0..l1 {
                g[k * l1 + l] += z[(t, l)] * e[(t, k)];
            }
        }
    };
    match cluster {
        Some(cl) => {
            let mut sums: std::collections::BTreeMap<u64, Vec<f64>> = Default::default();
            for t in 0..n {
                score(t, sums.entry(cl[t].to_bits()).or_insert_with(|| vec![0.0; dim]));
            }
            sums.values().for_each(|g| add(g));
        }
        None => {
            for t in 0..n {
                let mut g = vec![0.0; dim];
                score(t, &mut g);
                add(&g);
            }
        }
    }
    a.mul(&meat).unwrap().mul(&a).unwrap()
}

/// The rk statistic of H0: rank(Π) = K1 - 1.
/// `y`: endogenous regressors (K1 columns), `z`: excluded instruments (L1 columns).
pub fn identification(y: &[Vec<f64>], z: &[Vec<f64>], cluster: Option<&[f64]>, dof: &IdDof) -> Option<IdStats> {
    let (k1, l1) = (y.len(), z.len());
    if k1 == 0 || l1 < k1 {
        return None;
    }
    let (ym, zm) = (columns(y), columns(z));
    let n = ym.rows as f64;
    let zz = zm.transpose().mul(&zm).ok()?;
    let (zzi, _) = zz.sym_inverse(1e-14);
    let pi = zzi.mul(&zm.transpose().mul(&ym).ok()?).ok()?;
    let resid = ym.sub(&zm.mul(&pi).ok()?).ok()?;
    let svv = resid.transpose().mul(&resid).ok()?.scale(1.0 / n);
    // Θ = G Π F' with G'G = Z'Z/n and F'F = Σ_VV^-1
    let g = sym_pow(&zz.scale(1.0 / n), 0.5);
    let f = sym_pow(&svv, -0.5);
    let theta = g.mul(&pi).ok()?.mul(&f).ok()?;
    let fg = kron(&f, &g);
    // singular vectors of Θ
    let (_, v) = sorted_eigen(&theta.transpose().mul(&theta).ok()?);
    let (_, u) = sorted_eigen(&theta.mul(&theta.transpose()).ok()?);
    let q = k1 - 1;
    let block = |m: &Matrix, r0: usize, c0: usize| {
        let mut b = Matrix::zeros(m.rows - r0, m.cols - c0);
        for i in r0..m.rows {
            for j in c0..m.cols {
                b[(i - r0, j - c0)] = m[(i, j)];
            }
        }
        b
    };
    let ucols = block(&u, 0, q);
    let u22 = block(&u, q, q);
    let a_perp = ucols.mul(&u22.inverse().ok()?).ok()?.mul(&sym_pow(&u22.mul(&u22.transpose()).ok()?, 0.5)).ok()?;
    let vcols = block(&v, 0, q);
    let v22 = block(&v, q, q);
    let b_perp = sym_pow(&v22.mul(&v22.transpose()).ok()?, 0.5).mul(&v22.transpose().inverse().ok()?).ok()?.mul(&vcols.transpose()).ok()?;
    let proj = kron(&b_perp, &a_perp.transpose());
    let lambda = a_perp.transpose().mul(&theta).ok()?.mul(&b_perp.transpose()).ok()?;
    let lam = vec_of(&lambda);
    let rk = |cov_pi: &Matrix| -> Option<f64> {
        let cov_theta = fg.mul(cov_pi).ok()?.mul(&fg.transpose()).ok()?;
        let omega = proj.mul(&cov_theta).ok()?.mul(&proj.transpose()).ok()?;
        let (oi, _) = omega.sym_inverse(1e-14);
        Some((0..lam.len()).map(|a| (0..lam.len()).map(|c| lam[a] * oi[(a, c)] * lam[c]).sum::<f64>()).sum())
    };
    let robust = dof.robust || cluster.is_some();
    let wald = rk(&vec_pi_cov(&zm, &zzi, &resid, cluster, robust))?;
    // LM: covariance from residuals under the null (Π restricted to rank q)
    let pi_q = if q == 0 {
        Matrix::zeros(l1, k1)
    } else {
        let (sv, vv) = sorted_eigen(&theta.transpose().mul(&theta).ok()?);
        let mut theta_q = Matrix::zeros(l1, k1);
        for s in 0..q {
            let sigma = sv[s].max(0.0).sqrt();
            if sigma == 0.0 {
                continue;
            }
            let vs: Vec<f64> = (0..k1).map(|i| vv[(i, s)]).collect();
            let us: Vec<f64> = (0..l1).map(|i| (0..k1).map(|j| theta[(i, j)] * vs[j]).sum::<f64>() / sigma).collect();
            for i in 0..l1 {
                for j in 0..k1 {
                    theta_q[(i, j)] += sigma * us[i] * vs[j];
                }
            }
        }
        sym_pow(&zz.scale(1.0 / n), -0.5).mul(&theta_q).ok()?.mul(&sym_pow(&svv, 0.5)).ok()?
    };
    let resid0 = ym.sub(&zm.mul(&pi_q).ok()?).ok()?;
    let lm = rk(&vec_pi_cov(&zm, &zzi, &resid0, cluster, robust))?;
    // Cragg–Donald: minimum eigenvalue of Σ^-1/2 Y'P_Z Y Σ^-1/2 with Σ = V'V/(N-L)
    let ypy = pi.transpose().mul(&zz).ok()?.mul(&pi).ok()?;
    let s_cd = resid.transpose().mul(&resid).ok()?.scale(1.0 / (dof.n - dof.l));
    let h = sym_pow(&s_cd, -0.5);
    let (ev, _) = h.mul(&ypy).ok()?.mul(&h).ok()?.sym_eigen();
    let cd_f = ev.iter().cloned().fold(f64::INFINITY, f64::min) / l1 as f64;
    // KP rk Wald F: (N-L)/L1 scaling of rk/N; with clusters N/(N-1)·(G-1)/G as well
    let mut kp_f = wald / l1 as f64 * (dof.n - dof.l) / dof.n;
    if let Some(gc) = dof.clusters {
        kp_f *= dof.n / (dof.n - 1.0) * (gc - 1.0) / gc;
    }
    if !robust {
        kp_f = cd_f;
    }
    Some(IdStats { lm, df: (l1 - k1 + 1) as f64, wald, cd_f, kp_f })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        // deterministic pseudo-random design with two instruments, one endogenous regressor
        let n = 60;
        let mut s = 12345u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let z1: Vec<f64> = (0..n).map(|_| r()).collect();
        let z2: Vec<f64> = (0..n).map(|_| r()).collect();
        let y: Vec<f64> = (0..n).map(|t| 0.8 * z1[t] - 0.3 * z2[t] + r() * (1.0 + z1[t].abs())).collect();
        let cl: Vec<f64> = (0..n).map(|t| (t / 4) as f64).collect();
        let c = |v: Vec<f64>| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.into_iter().map(|x| x - m).collect::<Vec<_>>()
        };
        (vec![c(y)], vec![c(z1), c(z2)], cl)
    }

    #[test]
    fn single_endogenous_reduces_to_first_stage_tests() {
        let (y, z, cl) = data();
        let n = y[0].len() as f64;
        // i.i.d.: LM = N R² (uncentred on partialled data), CD F = classic F
        let s = identification(&y, &z, None, &IdDof { n, l: 3.0, clusters: None, robust: false }).unwrap();
        let ym = columns(&y);
        let zm = columns(&z);
        let (zzi, _) = zm.transpose().mul(&zm).unwrap().sym_inverse(1e-14);
        let pi = zzi.mul(&zm.transpose().mul(&ym).unwrap()).unwrap();
        let fitted = zm.mul(&pi).unwrap();
        let ess: f64 = (0..ym.rows).map(|t| fitted[(t, 0)].powi(2)).sum();
        let tss: f64 = (0..ym.rows).map(|t| ym[(t, 0)].powi(2)).sum();
        let rss = tss - ess;
        assert!((s.lm - n * ess / tss).abs() < 1e-8, "{} vs {}", s.lm, n * ess / tss);
        let f = (ess / 2.0) / (rss / (n - 3.0));
        assert!((s.cd_f - f).abs() < 1e-8);
        assert!((s.kp_f - f).abs() < 1e-8);
        assert_eq!(s.df, 2.0);
        // cluster-robust: KP Wald = cluster Wald test of Π, F scaled like regress
        let c = identification(&y, &z, Some(&cl), &IdDof { n, l: 3.0, clusters: Some(15.0), robust: true }).unwrap();
        let e: Vec<f64> = (0..ym.rows).map(|t| ym[(t, 0)] - fitted[(t, 0)]).collect();
        let mut meat = Matrix::zeros(2, 2);
        for g in 0..15 {
            let mut sc = [0.0; 2];
            for t in g * 4..g * 4 + 4 {
                sc[0] += zm[(t, 0)] * e[t];
                sc[1] += zm[(t, 1)] * e[t];
            }
            for a in 0..2 {
                for b in 0..2 {
                    meat[(a, b)] += sc[a] * sc[b];
                }
            }
        }
        let v = zzi.mul(&meat).unwrap().mul(&zzi).unwrap();
        let (vi, _) = v.sym_inverse(1e-14);
        let w: f64 = (0..2).map(|a| (0..2).map(|b| pi[(a, 0)] * vi[(a, b)] * pi[(b, 0)]).sum::<f64>()).sum();
        assert!((c.wald - w).abs() < 1e-8 * w.max(1.0));
        let regress_f = w / 2.0 * (14.0 / 15.0) * (n - 3.0) / (n - 1.0);
        assert!((c.kp_f - regress_f).abs() < 1e-8 * regress_f);
    }

    #[test]
    fn two_endogenous_rank_statistic_is_invariant_and_positive() {
        let (y, z, _) = data();
        let mut z3 = z.clone();
        z3.push(z[0].iter().zip(&z[1]).map(|(a, b)| a * b).collect());
        let y2 = vec![y[0].clone(), (0..y[0].len()).map(|t| z[1][t] + 0.4 * z3[2][t] + 0.3 * ((t * t) as f64 * 0.7).sin()).collect()];
        let n = y[0].len() as f64;
        let s = identification(&y2, &z3, None, &IdDof { n, l: 3.0, clusters: None, robust: false }).unwrap();
        assert_eq!(s.df, 2.0);
        assert!(s.lm > 0.0 && s.wald > 0.0 && s.cd_f > 0.0, "{s:?}");
        // i.i.d.: LM equals Anderson's N · (smallest squared canonical correlation)
        let ym = columns(&y2);
        let zm = columns(&z3);
        let (zzi, _) = zm.transpose().mul(&zm).unwrap().sym_inverse(1e-14);
        let zy = zm.transpose().mul(&ym).unwrap();
        let ypy = zy.transpose().mul(&zzi).unwrap().mul(&zy).unwrap();
        let h = sym_pow(&ym.transpose().mul(&ym).unwrap(), -0.5);
        let (ev, _) = h.mul(&ypy).unwrap().mul(&h).unwrap().sym_eigen();
        let anderson = n * ev.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((s.lm - anderson).abs() < 1e-8 * anderson, "{} vs {}", s.lm, anderson);
        // rescaling a regressor leaves the rank statistic unchanged
        let y3 = vec![y2[0].iter().map(|v| v * 7.0).collect(), y2[1].clone()];
        let s3 = identification(&y3, &z3, None, &IdDof { n, l: 3.0, clusters: None, robust: false }).unwrap();
        assert!((s.wald - s3.wald).abs() < 1e-6 * s.wald);
        assert!((s.cd_f - s3.cd_f).abs() < 1e-6 * s.cd_f);
    }
}
