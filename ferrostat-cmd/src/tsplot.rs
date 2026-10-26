//! Time-series diagnostic plots: correlograms (ac, pac, xcorr), the
//! periodogram (pergram), the cumulative spectral distribution (cumsp) and
//! Bartlett's cumulative-periodogram white-noise test (wntestb).
//!
//! Definitions follow Box, Jenkins & Reinsel, *Time Series Analysis*, and
//! Brockwell & Davis, *Time Series: Theory and Methods*.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

/// Values of a (possibly operator-prefixed) variable in time order, sample only.
fn series(ds: &DataSet, st: &Store, cmd: &Command, v: &str) -> Result<Vec<f64>, String> {
    let (_, t) = st.tsset.clone().ok_or_else(|| rc_err(111, "time variable not set, use tsset"))?;
    let tc = util::col(ds, &t)?;
    let mask = util::touse(ds, st, cmd)?;
    let x = graph::plot_col(ds, st, v)?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(x[i]) && !is_missing(tc[i])).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    let out: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
    if out.len() < 4 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    Ok(out)
}

fn demeaned(x: &[f64]) -> Vec<f64> {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v - m).collect()
}

/// Sample autocorrelations r_1..r_k.
pub fn autocorrelations(x: &[f64], k: usize) -> Vec<f64> {
    let d = demeaned(x);
    let c0: f64 = d.iter().map(|v| v * v).sum();
    (1..=k).map(|lag| (lag..d.len()).map(|t| d[t] * d[t - lag]).sum::<f64>() / c0).collect()
}

/// Partial autocorrelations: the coefficient on lag k when x_t is regressed
/// on a constant and its first k lags.
pub fn partial_autocorrelations(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    (1..=k)
        .map(|lag| {
            if n <= 2 * lag + 1 {
                return f64::NAN;
            }
            let p = lag + 1;
            let mut xtx = Matrix::zeros(p, p);
            let mut xty = vec![0.0; p];
            for t in lag..n {
                let row: Vec<f64> = std::iter::once(1.0).chain((1..=lag).map(|j| x[t - j])).collect();
                for a in 0..p {
                    xty[a] += row[a] * x[t];
                    for b in 0..p {
                        xtx[(a, b)] += row[a] * row[b];
                    }
                }
            }
            let (inv, _) = xtx.sym_inverse(1e-12);
            (0..p).map(|b| inv[(lag, b)] * xty[b]).sum()
        })
        .collect()
}

/// Periodogram at the Fourier frequencies j/n, j = 1..n/2.
pub fn periodogram(x: &[f64]) -> Vec<(f64, f64)> {
    let d = demeaned(x);
    let n = d.len();
    (1..=n / 2)
        .map(|j| {
            let w = std::f64::consts::TAU * j as f64 / n as f64;
            let (c, s) = d.iter().enumerate().fold((0.0, 0.0), |(c, s), (t, v)| (c + v * (w * t as f64).cos(), s + v * (w * t as f64).sin()));
            (j as f64 / n as f64, (c * c + s * s) / n as f64)
        })
        .collect()
}

fn default_lags(n: usize) -> usize {
    (n / 2).saturating_sub(2).clamp(1, 40)
}

fn spikes(points: &[(f64, f64)], label: &str) -> Layer {
    Layer { kind: LayerKind::Range(RangeStyle::Drop), ranges: points.iter().map(|&(x, y)| (x, 0.0, y)).collect(), label: label.into(), ..Default::default() }
}

fn band(points: Vec<(f64, f64, f64)>, label: &str) -> Layer {
    Layer { kind: LayerKind::Range(RangeStyle::Area), ranges: points, label: label.into(), alpha: 0.3, ..Default::default() }
}

/// recast(area|line|...) of a line layer
fn recast(cmd: &Command, mut l: Layer) -> Layer {
    if let Some(r) = cmd.opt("recast", 3) {
        l.kind = match r.trim() {
            "area" => LayerKind::Area,
            "connected" => LayerKind::Connected,
            "scatter" => LayerKind::Scatter,
            _ => LayerKind::Line,
        };
    }
    l
}

pub struct TsPlot;

impl CommandHandler for TsPlot {
    fn name(&self) -> &'static str {
        "ac"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        let need = if cmd.verb == "xcorr" { 2 } else { 1 };
        if vars.len() != need {
            return Err(rc_err(if vars.len() < need { 102 } else { 103 }, format!("{}: {} variable(s) required", cmd.verb, need)));
        }
        let x = series(ds, st, cmd, &vars[0])?;
        let n = x.len();
        let lags = util::opt_num(cmd, "lags", 1)?.map_or(default_lags(n), |l| l as usize).min(n - 2).max(1);
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut text = String::new();
        let mut graph = match cmd.verb.as_str() {
            "ac" => {
                let r = autocorrelations(&x, lags);
                // Bartlett's formula for MA(q) processes
                let mut cum = 0.0;
                let bounds: Vec<(f64, f64, f64)> = r
                    .iter()
                    .enumerate()
                    .map(|(i, ri)| {
                        let se = ((1.0 + 2.0 * cum) / n as f64).sqrt();
                        cum += ri * ri;
                        ((i + 1) as f64, -z * se, z * se)
                    })
                    .collect();
                let pts: Vec<(f64, f64)> = r.iter().enumerate().map(|(i, v)| ((i + 1) as f64, *v)).collect();
                Graph {
                    layers: vec![band(bounds, &format!("{}% CI (Bartlett's formula for MA(q))", fmt_g(level, 4))), spikes(&pts, &format!("Autocorrelations of {}", vars[0]))],
                    xtitle: "Lag".into(),
                    ytitle: format!("Autocorrelations of {}", vars[0]),
                    legend: true,
                    ..Default::default()
                }
            }
            "pac" => {
                let r = partial_autocorrelations(&x, lags);
                let se = 1.0 / (n as f64).sqrt();
                let pts: Vec<(f64, f64)> = r.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| ((i + 1) as f64, *v)).collect();
                let bounds = pts.iter().map(|p| (p.0, -z * se, z * se)).collect();
                Graph {
                    layers: vec![band(bounds, &format!("{}% CI, se = 1/sqrt(n)", fmt_g(level, 4))), spikes(&pts, &format!("Partial autocorrelations of {}", vars[0]))],
                    xtitle: "Lag".into(),
                    ytitle: format!("Partial autocorrelations of {}", vars[0]),
                    legend: true,
                    ..Default::default()
                }
            }
            "xcorr" => {
                let y = series(ds, st, cmd, &vars[1])?;
                let m = x.len().min(y.len());
                let (dx, dy) = (demeaned(&x[..m]), demeaned(&y[..m]));
                let sx = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
                let sy = dy.iter().map(|v| v * v).sum::<f64>().sqrt();
                let k = lags.min(m - 1) as i64;
                let pts: Vec<(f64, f64)> = (-k..=k)
                    .map(|lag| {
                        let s: f64 = (0..m as i64).filter(|t| t + lag >= 0 && t + lag < m as i64).map(|t| dx[t as usize] * dy[(t + lag) as usize]).sum();
                        (lag as f64, s / (sx * sy))
                    })
                    .collect();
                Graph { layers: vec![spikes(&pts, "Cross-correlogram")], xtitle: "Lag".into(), ytitle: format!("Cross-correlations of {} and {}", vars[0], vars[1]), ..Default::default() }
            }
            "pergram" => {
                let p = periodogram(&x);
                let var = demeaned(&x).iter().map(|v| v * v).sum::<f64>() / n as f64;
                // log standardized periodogram, as plotted against frequency
                let pts: Vec<(f64, f64)> = p.iter().map(|&(f, i)| (f, (i / var).max(1e-12).ln())).collect();
                let l = recast(cmd, Layer { kind: LayerKind::Line, points: pts, label: format!("{} log periodogram", vars[0]), ..Default::default() });
                Graph { layers: vec![l], xtitle: "Frequency".into(), ytitle: format!("{} log periodogram", vars[0]), ..Default::default() }
            }
            "cumsp" => {
                let p = periodogram(&x);
                let total: f64 = p.iter().map(|q| q.1).sum();
                let mut cum = 0.0;
                let mut pts = vec![(0.0, 0.0)];
                pts.extend(p.iter().map(|&(f, i)| {
                    cum += i;
                    (f, cum / total)
                }));
                let l = recast(cmd, Layer { kind: LayerKind::Line, points: pts, label: format!("{} cumulative spectral distribution", vars[0]), ..Default::default() });
                Graph { layers: vec![l], xtitle: "Frequency".into(), ytitle: format!("{} cumulative spectral distribution", vars[0]), ..Default::default() }
            }
            "wntestb" => {
                // cumulative periodogram compared with the uniform distribution
                let p = periodogram(&x);
                let q = p.len();
                let total: f64 = p.iter().map(|v| v.1).sum();
                let mut cum = 0.0;
                let mut pts = vec![(0.0, 0.0)];
                let mut dmax: f64 = 0.0;
                for (j, &(f, i)) in p.iter().enumerate() {
                    cum += i;
                    let fj = cum / total;
                    dmax = dmax.max((fj - (j + 1) as f64 / q as f64).abs());
                    pts.push((f, fj));
                }
                let b = dmax * (q as f64).sqrt();
                // Kolmogorov-Smirnov limiting distribution
                let prob = (2.0 * (1..=100).map(|k| (-1f64).powi(k - 1) * (-2.0 * (k as f64).powi(2) * b * b).exp()).sum::<f64>()).clamp(0.0, 1.0);
                text = format!("Cumulative periodogram white-noise test\n\n Bartlett's (B) statistic = {:>8.4}\n Prob > B                 = {:>8.4}", b, prob);
                st.begin_r().set("stat", b);
                st.r.set("p", prob);
                // 95% band: ±1.358/sqrt(q) around the diagonal
                let c = 1.358 / (q as f64).sqrt();
                let bounds: Vec<(f64, f64, f64)> = (0..=20).map(|k| {
                    let f = 0.5 * k as f64 / 20.0;
                    (f, (2.0 * f - c).max(0.0), (2.0 * f + c).min(1.0))
                }).collect();
                Graph {
                    layers: vec![band(bounds, "95% bands"), Layer { kind: LayerKind::Line, points: pts, label: "Cumulative periodogram".into(), ..Default::default() }],
                    xtitle: "Frequency".into(),
                    ytitle: format!("Cumulative periodogram for {}", vars[0]),
                    ..Default::default()
                }
            }
            other => return Err(rc_err(199, format!("command {other} is unrecognized"))),
        };
        let extra = graph::finish(&mut graph, cmd, st)?;
        Ok([text, extra].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlations_of_known_series() {
        // alternating series: r1 close to -1, partial r1 equal to the regression slope
        let x: Vec<f64> = (0..200).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = autocorrelations(&x, 3);
        assert!((r[0] + 0.995).abs() < 1e-3 && (r[1] - 0.99).abs() < 1e-3);
        let p = partial_autocorrelations(&x, 1);
        assert!((p[0] + 1.0).abs() < 1e-9);
        // a pure cosine at frequency 1/8 puts all periodogram mass there
        let c: Vec<f64> = (0..64).map(|t| (std::f64::consts::TAU * t as f64 / 8.0).cos()).collect();
        let pg = periodogram(&c);
        let peak = pg.iter().cloned().fold((0.0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        assert!((peak.0 - 0.125).abs() < 1e-12);
        assert!(pg.iter().filter(|q| q.0 != 0.125).all(|q| q.1 < 1e-20));
    }
}
