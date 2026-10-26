//! Surfaces z = f(x, y) for contour plots and heat maps: interpolation of
//! scattered data onto a regular grid (thin-plate spline, Duchon 1977; or
//! inverse-distance weighting, Shepard 1968), contour lines by marching
//! squares, and a colour-blind-safe sequential colour ramp.

/// A regular grid of z values: z[j][i] at (xs[i], ys[j]); NaN where undefined.
#[derive(Debug, Clone)]
pub struct Grid {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub z: Vec<Vec<f64>>,
}

/// Are the points a complete regular grid (every x with every y)?
pub fn as_grid(pts: &[(f64, f64, f64)]) -> Option<Grid> {
    let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let mut ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    ys.sort_by(f64::total_cmp);
    ys.dedup();
    if xs.len() < 2 || ys.len() < 2 || xs.len() * ys.len() != pts.len() {
        return None;
    }
    let mut z = vec![vec![f64::NAN; xs.len()]; ys.len()];
    for &(x, y, v) in pts {
        let i = xs.binary_search_by(|a| a.total_cmp(&x)).ok()?;
        let j = ys.binary_search_by(|a| a.total_cmp(&y)).ok()?;
        z[j][i] = v;
    }
    if z.iter().flatten().any(|v| v.is_nan()) {
        return None;
    }
    Some(Grid { xs, ys, z })
}

/// Solve a dense linear system by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            if f != 0.0 {
                for k in c..n {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        x[r] = (b[r] - (r + 1..n).map(|k| a[r][k] * x[k]).sum::<f64>()) / a[r][r];
    }
    Some(x)
}

/// Interpolate scattered points onto an n×n grid over their bounding box.
/// Coordinates are scaled to the unit square first so both axes count equally.
pub fn interpolate(pts: &[(f64, f64, f64)], n: usize, method: &str) -> Grid {
    let lo = |f: fn(&(f64, f64, f64)) -> f64| pts.iter().map(f).fold(f64::INFINITY, f64::min);
    let hi = |f: fn(&(f64, f64, f64)) -> f64| pts.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
    let (x0, x1, y0, y1) = (lo(|p| p.0), hi(|p| p.0), lo(|p| p.1), hi(|p| p.1));
    let (sx, sy) = ((x1 - x0).max(1e-12), (y1 - y0).max(1e-12));
    let u: Vec<(f64, f64, f64)> = pts.iter().map(|&(x, y, z)| ((x - x0) / sx, (y - y0) / sy, z)).collect();
    let xs: Vec<f64> = (0..n).map(|i| x0 + sx * i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = (0..n).map(|j| y0 + sy * j as f64 / (n - 1) as f64).collect();
    let phi = |r2: f64| if r2 > 0.0 { 0.5 * r2 * r2.ln() } else { 0.0 };
    let m = u.len();
    let tps = if method != "shepard" && m <= 400 {
        // [K P; P' 0] [w; a] = [z; 0]
        let mut a = vec![vec![0.0; m + 3]; m + 3];
        let mut b = vec![0.0; m + 3];
        for i in 0..m {
            for j in 0..m {
                let (dx, dy) = (u[i].0 - u[j].0, u[i].1 - u[j].1);
                a[i][j] = phi(dx * dx + dy * dy);
            }
            for (k, v) in [1.0, u[i].0, u[i].1].into_iter().enumerate() {
                a[i][m + k] = v;
                a[m + k][i] = v;
            }
            b[i] = u[i].2;
        }
        solve(a, b)
    } else {
        None
    };
    let at = |px: f64, py: f64| -> f64 {
        match &tps {
            Some(c) => {
                let s: f64 = u.iter().enumerate().map(|(i, q)| c[i] * phi((px - q.0).powi(2) + (py - q.1).powi(2))).sum();
                s + c[m] + c[m + 1] * px + c[m + 2] * py
            }
            None => {
                let (mut num, mut den) = (0.0, 0.0);
                for q in &u {
                    let d2 = (px - q.0).powi(2) + (py - q.1).powi(2);
                    if d2 < 1e-24 {
                        return q.2;
                    }
                    num += q.2 / d2;
                    den += 1.0 / d2;
                }
                num / den
            }
        }
    };
    let z = (0..n).map(|j| (0..n).map(|i| at(i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64)).collect()).collect();
    Grid { xs, ys, z }
}

/// `levels` equal-width bands between min and max: the interior cut points.
pub fn cuts(zmin: f64, zmax: f64, levels: usize) -> Vec<f64> {
    let k = levels.max(2);
    (1..k).map(|i| zmin + (zmax - zmin) * i as f64 / k as f64).collect()
}

/// Index of the band a value falls in (0 ..= cuts.len()).
pub fn band(v: f64, cuts: &[f64]) -> usize {
    cuts.iter().take_while(|c| v >= **c).count()
}

/// Colour-blind-safe sequential ramp (viridis anchor colours, linearly interpolated).
pub fn ramp(t: f64) -> (u8, u8, u8) {
    const STOPS: [(f64, f64, f64); 5] = [(68.0, 1.0, 84.0), (59.0, 82.0, 139.0), (33.0, 145.0, 140.0), (94.0, 201.0, 98.0), (253.0, 231.0, 37.0)];
    let t = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f64;
    let i = (t.floor() as usize).min(STOPS.len() - 2);
    let f = t - i as f64;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    let mix = |p: f64, q: f64| (p + (q - p) * f).round() as u8;
    (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

/// Colours of `n` bands.
pub fn band_colors(n: usize) -> Vec<(u8, u8, u8)> {
    (0..n).map(|k| ramp(if n > 1 { k as f64 / (n - 1) as f64 } else { 0.5 })).collect()
}

/// Contour line segments at `level` by marching squares: ((x1, y1), (x2, y2)).
pub fn contour_segments(g: &Grid, level: f64) -> Vec<((f64, f64), (f64, f64))> {
    let mut out = Vec::new();
    let lerp = |a: (f64, f64, f64), b: (f64, f64, f64)| {
        let t = if b.2 != a.2 { (level - a.2) / (b.2 - a.2) } else { 0.5 };
        (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
    };
    for j in 0..g.ys.len().saturating_sub(1) {
        for i in 0..g.xs.len().saturating_sub(1) {
            // corners counter-clockwise from bottom-left
            let c = [
                (g.xs[i], g.ys[j], g.z[j][i]),
                (g.xs[i + 1], g.ys[j], g.z[j][i + 1]),
                (g.xs[i + 1], g.ys[j + 1], g.z[j + 1][i + 1]),
                (g.xs[i], g.ys[j + 1], g.z[j + 1][i]),
            ];
            if c.iter().any(|p| p.2.is_nan()) {
                continue;
            }
            // crossing points on the four edges
            let edges: Vec<(usize, (f64, f64))> = (0..4)
                .filter_map(|e| {
                    let (a, b) = (c[e], c[(e + 1) % 4]);
                    ((a.2 >= level) != (b.2 >= level)).then(|| (e, lerp(a, b)))
                })
                .collect();
            match edges.len() {
                2 => out.push((edges[0].1, edges[1].1)),
                4 => {
                    // saddle: pair the crossings according to the centre value
                    let centre = c.iter().map(|p| p.2).sum::<f64>() / 4.0;
                    let corner0_above = c[0].2 >= level;
                    if (centre >= level) == corner0_above {
                        out.push((edges[0].1, edges[1].1));
                        out.push((edges[2].1, edges[3].1));
                    } else {
                        out.push((edges[0].1, edges[3].1));
                        out.push((edges[1].1, edges[2].1));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_detection_and_interpolation() {
        // z = x + 2y on a 3x3 grid is recognised as a grid
        let pts: Vec<(f64, f64, f64)> = (0..3).flat_map(|i| (0..3).map(move |j| (i as f64, j as f64, i as f64 + 2.0 * j as f64))).collect();
        let g = as_grid(&pts).unwrap();
        assert_eq!(g.z[2][1], 1.0 + 4.0);
        // a thin-plate spline reproduces planes exactly
        let scattered = vec![(0.0, 0.0, 0.0), (1.0, 0.2, 1.4), (0.3, 1.0, 2.3), (0.9, 0.9, 2.7), (0.5, 0.4, 1.3)];
        let gi = interpolate(&scattered, 5, "thinplatespline");
        for (j, y) in gi.ys.iter().enumerate() {
            for (i, x) in gi.xs.iter().enumerate() {
                assert!((gi.z[j][i] - (x + 2.0 * y)).abs() < 1e-8, "{x} {y} {}", gi.z[j][i]);
            }
        }
        // inverse-distance weighting passes through the data
        let gs = interpolate(&[(0.0, 0.0, 1.0), (1.0, 1.0, 3.0)], 2, "shepard");
        assert_eq!((gs.z[0][0], gs.z[1][1]), (1.0, 3.0));
    }

    #[test]
    fn contours_and_bands() {
        // the level x + y = 1 on the unit square crosses at (1,0) and (0,1)
        let g = Grid { xs: vec![0.0, 1.0], ys: vec![0.0, 1.0], z: vec![vec![0.0, 1.0], vec![1.0, 2.0]] };
        let s = contour_segments(&g, 1.0);
        assert_eq!(s.len(), 1);
        assert_eq!(cuts(0.0, 10.0, 5), vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(band(5.0, &[2.0, 4.0, 6.0]), 2);
        assert_eq!(band_colors(2), vec![(68, 1, 84), (253, 231, 37)]);
    }
}
