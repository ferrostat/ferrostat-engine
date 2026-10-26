//! Scatterplot smoothers for twoway plots: lowess (Cleveland 1979, *JASA*
//! 74: 829-836), local polynomial regression (Fan & Gijbels, *Local
//! Polynomial Modelling and Its Applications*, 1996), median bands and
//! cubic splines through them (`mband`, `mspline`).

use crate::kdensity::kernel;
use crate::matrix::Matrix;

/// Lowess: at each x_i, a weighted linear fit (or weighted mean) over the
/// `bwidth`·n nearest points with tricube weights. Returns points sorted by x.
pub fn lowess(pts: &[(f64, f64)], bwidth: f64, mean: bool, noweight: bool) -> Vec<(f64, f64)> {
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = p.len();
    if n == 0 {
        return p;
    }
    let k = ((bwidth * n as f64).floor() as usize).clamp(1, n);
    (0..n)
        .map(|i| {
            let x0 = p[i].0;
            // the k nearest points
            let mut d: Vec<(f64, usize)> = p.iter().enumerate().map(|(j, q)| ((q.0 - x0).abs(), j)).collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            let near = &d[..k];
            let delta = near.iter().map(|q| q.0).fold(0.0, f64::max) * 1.0001;
            let w: Vec<(f64, usize)> = near
                .iter()
                .map(|&(dist, j)| (if noweight || delta == 0.0 { 1.0 } else { (1.0 - (dist / delta).powi(3)).powi(3) }, j))
                .collect();
            let sw: f64 = w.iter().map(|q| q.0).sum();
            let my = w.iter().map(|&(wi, j)| wi * p[j].1).sum::<f64>() / sw;
            if mean {
                return (x0, my);
            }
            let mx = w.iter().map(|&(wi, j)| wi * p[j].0).sum::<f64>() / sw;
            let sxx: f64 = w.iter().map(|&(wi, j)| wi * (p[j].0 - mx).powi(2)).sum();
            let sxy: f64 = w.iter().map(|&(wi, j)| wi * (p[j].0 - mx) * (p[j].1 - my)).sum();
            let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            (x0, my + b * (x0 - mx))
        })
        .collect()
}

/// Rule-of-thumb bandwidth 0.9·min(sd, IQR/1.349)·n^(-1/5).
pub fn rot_bandwidth(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let iqr = crate::util::pctile_sorted(&s, 75.0) - crate::util::pctile_sorted(&s, 25.0);
    let spread = if iqr > 0.0 { sd.min(iqr / 1.349) } else { sd };
    (0.9 * spread * n.powf(-0.2)).max(1e-12)
}

/// Local polynomial fit of the given degree at `grid`; returns (x, m̂(x),
/// standard error). Variance: σ² Σ l_i(x)², with σ² estimated from the
/// residuals at the data points.
pub fn lpoly(pts: &[(f64, f64)], grid: &[f64], kname: &str, degree: usize, h: f64) -> Vec<(f64, f64, f64)> {
    let p = degree + 1;
    // smoother weights l(x0) for one evaluation point
    let weights = |x0: f64| -> Option<Vec<f64>> {
        let mut xtwx = Matrix::zeros(p, p);
        let rows: Vec<(f64, Vec<f64>)> = pts
            .iter()
            .map(|&(x, _)| {
                let w = kernel(kname, (x - x0) / h);
                (w, (0..p).map(|k| (x - x0).powi(k as i32)).collect())
            })
            .collect();
        for (w, r) in &rows {
            if *w == 0.0 {
                continue;
            }
            for a in 0..p {
                for b in 0..p {
                    xtwx[(a, b)] += w * r[a] * r[b];
                }
            }
        }
        let (inv, dropped) = xtwx.sym_inverse(1e-12);
        if !dropped.is_empty() {
            return None;
        }
        Some(rows.iter().map(|(w, r)| w * (0..p).map(|b| inv[(0, b)] * r[b]).sum::<f64>()).collect())
    };
    let fit_at = |x0: f64| weights(x0).map(|l| (l.iter().zip(pts).map(|(li, q)| li * q.1).sum::<f64>(), l));
    let resid: Vec<f64> = pts.iter().filter_map(|&(x, y)| fit_at(x).map(|(m, _)| y - m)).collect();
    let dof = (resid.len() as f64 - p as f64).max(1.0);
    let s2 = resid.iter().map(|r| r * r).sum::<f64>() / dof;
    grid.iter().filter_map(|&x0| fit_at(x0).map(|(m, l)| (x0, m, (s2 * l.iter().map(|v| v * v).sum::<f64>()).sqrt()))).collect()
}

/// Median of a slice.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    crate::util::pctile_sorted(v, 50.0)
}

/// Median bands: the x range cut into `bands` equal-width bands; in each
/// non-empty band, the medians of x and y.
pub fn median_bands(pts: &[(f64, f64)], bands: usize) -> Vec<(f64, f64)> {
    let lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let hi = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let width = if hi > lo { (hi - lo) / bands as f64 } else { 1.0 };
    let mut groups: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bands.max(1)];
    for &p in pts {
        let k = (((p.0 - lo) / width) as usize).min(groups.len() - 1);
        groups[k].push(p);
    }
    groups
        .into_iter()
        .filter(|g| !g.is_empty())
        .map(|g| {
            let mut xs: Vec<f64> = g.iter().map(|p| p.0).collect();
            let mut ys: Vec<f64> = g.iter().map(|p| p.1).collect();
            (median(&mut xs), median(&mut ys))
        })
        .collect()
}

/// Natural cubic spline through `knots` (increasing x), evaluated at `steps`
/// points between consecutive knots.
pub fn natural_spline(knots: &[(f64, f64)], steps: usize) -> Vec<(f64, f64)> {
    let n = knots.len();
    if n < 3 {
        return knots.to_vec();
    }
    let h: Vec<f64> = (0..n - 1).map(|i| knots[i + 1].0 - knots[i].0).collect();
    // second derivatives M_i with M_0 = M_{n-1} = 0 (tridiagonal system)
    let mut a = vec![0.0; n];
    let mut b = vec![1.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * ((knots[i + 1].1 - knots[i].1) / h[i] - (knots[i].1 - knots[i - 1].1) / h[i - 1]);
    }
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    let mut mm = vec![0.0; n];
    for i in (0..n).rev() {
        mm[i] = if i == n - 1 { d[i] / b[i] } else { (d[i] - c[i] * mm[i + 1]) / b[i] };
    }
    let mut out = Vec::new();
    for i in 0..n - 1 {
        for s in 0..steps {
            let t = s as f64 / steps as f64;
            let x = knots[i].0 + t * h[i];
            let (u, v) = (knots[i + 1].0 - x, x - knots[i].0);
            let y = mm[i] * u.powi(3) / (6.0 * h[i]) + mm[i + 1] * v.powi(3) / (6.0 * h[i]) + (knots[i].1 / h[i] - mm[i] * h[i] / 6.0) * u + (knots[i + 1].1 / h[i] - mm[i + 1] * h[i] / 6.0) * v;
            out.push((x, y));
        }
    }
    out.push(knots[n - 1]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothers_reproduce_lines() {
        // on exactly linear data, lowess, local linear regression and the spline are exact
        let pts: Vec<(f64, f64)> = (0..30).map(|i| (i as f64, 2.0 + 0.5 * i as f64)).collect();
        for (x, y) in lowess(&pts, 0.5, false, false) {
            assert!((y - (2.0 + 0.5 * x)).abs() < 1e-9);
        }
        let grid: Vec<f64> = (0..10).map(|i| 1.0 + 2.5 * i as f64).collect();
        for (x, m, _) in lpoly(&pts, &grid, "epanechnikov", 1, 4.0) {
            assert!((m - (2.0 + 0.5 * x)).abs() < 1e-9);
        }
        let knots: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 3.0 * i as f64)).collect();
        for (x, y) in natural_spline(&knots, 4) {
            assert!((y - 3.0 * x).abs() < 1e-9);
        }
        // the local mean of a constant is that constant, with zero residual variance
        let flat: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 7.0)).collect();
        assert!(lpoly(&flat, &[5.0], "gaussian", 0, 2.0).iter().all(|q| (q.1 - 7.0).abs() < 1e-12 && q.2.abs() < 1e-12));
        // two bands, medians of each
        let mb = median_bands(&[(0.0, 1.0), (1.0, 3.0), (2.0, 2.0), (9.0, 10.0), (10.0, 12.0)], 2);
        assert_eq!(mb, vec![(1.0, 2.0), (9.5, 11.0)]);
    }
}
