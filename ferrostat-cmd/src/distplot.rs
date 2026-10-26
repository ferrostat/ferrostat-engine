//! Distributional diagnostic plots: quantile, probability, symmetry and
//! ladder-of-powers plots, spike plots and sunflower plots.
//!
//! Plotting positions follow the usual textbook conventions (Chambers,
//! Cleveland, Kleiner & Tukey, *Graphical Methods for Data Analysis*, 1983):
//! the i-th of n ordered values is paired with probability i/(n+1).

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::store::Store;
use crate::util;
use crate::varlist;

/// Sorted non-missing values of a variable in the estimation sample.
fn sorted_values(ds: &DataSet, st: &Store, cmd: &Command, var: &str) -> Result<Vec<f64>, String> {
    let mask = util::touse(ds, st, cmd)?;
    let c = util::col(ds, var)?;
    let mut v: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect();
    if v.len() < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    v.sort_by(f64::total_cmp);
    Ok(v)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt())
}

fn label_of(ds: &DataSet, v: &str) -> String {
    ds.get_var(v).map(|x| if x.label.is_empty() { v.to_string() } else { x.label.clone() }).unwrap_or_else(|| v.to_string())
}

fn scatter(points: Vec<(f64, f64)>, label: &str) -> Layer {
    Layer { kind: LayerKind::Scatter, points, label: label.into(), ..Default::default() }
}

fn line(points: Vec<(f64, f64)>, label: &str) -> Layer {
    Layer { kind: LayerKind::Line, points, label: label.into(), ..Default::default() }
}

/// Quantile-type plot: data points, a reference line, axis titles.
fn qplot(points: Vec<(f64, f64)>, reference: Vec<(f64, f64)>, xtitle: &str, ytitle: &str) -> Graph {
    Graph { layers: vec![scatter(points, ytitle), line(reference, "Reference")], xtitle: xtitle.into(), ytitle: ytitle.into(), ..Default::default() }
}

fn diagonal(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let lo = points.iter().map(|p| p.0.min(p.1)).fold(f64::INFINITY, f64::min);
    let hi = points.iter().map(|p| p.0.max(p.1)).fold(f64::NEG_INFINITY, f64::max);
    vec![(lo, lo), (hi, hi)]
}

/// Ladder of powers (Tukey): name and transform.
const LADDER: [(&str, fn(f64) -> f64); 9] = [
    ("cubic", |x| x.powi(3)),
    ("square", |x| x * x),
    ("identity", |x| x),
    ("sqrt", |x| x.sqrt()),
    ("log", |x| x.ln()),
    ("1/sqrt", |x| -1.0 / x.sqrt()),
    ("inverse", |x| -1.0 / x),
    ("1/square", |x| -1.0 / (x * x)),
    ("1/cubic", |x| -1.0 / x.powi(3)),
];

/// Normal quantile plot of already-sorted values.
fn qnorm_graph(v: &[f64], title: &str) -> Graph {
    let n = v.len() as f64;
    let (m, s) = mean_sd(v);
    let pts: Vec<(f64, f64)> = v.iter().enumerate().map(|(i, &x)| (m + s * dist::invnormal((i as f64 + 1.0) / (n + 1.0)), x)).collect();
    let refl = diagonal(&pts);
    qplot(pts, refl, "Inverse Normal", title)
}

/// Histogram layer of values with `bins` equal bins, heights as fractions or densities.
fn histogram_layer(v: &[f64], fraction: bool) -> Layer {
    let n = v.len() as f64;
    let (lo, hi) = (v[0], v[v.len() - 1]);
    let bins = (n.sqrt().min(10.0 * n.ln() / 10f64.ln())).floor().max(1.0) as usize;
    let width = if hi > lo { (hi - lo) / bins as f64 } else { 1.0 };
    let mut counts = vec![0.0; bins];
    for &x in v {
        let k = (((x - lo) / width) as usize).min(bins - 1);
        counts[k] += 1.0;
    }
    let scale = if fraction { 1.0 / n } else { 1.0 / (n * width) };
    let points = counts.iter().enumerate().map(|(k, c)| (lo + (k as f64 + 0.5) * width, c * scale)).collect();
    Layer { kind: LayerKind::Bar, points, width, label: String::new(), ..Default::default() }
}

pub struct DistPlot;

impl CommandHandler for DistPlot {
    fn name(&self) -> &'static str {
        "qnorm"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let need = if matches!(cmd.verb.as_str(), "qqplot" | "sunflower") { 2 } else { 1 };
        if vars.len() != need {
            return Err(rc_err(if vars.len() < need { 102 } else { 103 }, format!("{}: {} variable(s) required", cmd.verb, need)));
        }
        let v0 = &vars[0];
        let lab = label_of(ds, v0);
        let df = || -> Result<f64, String> { util::opt_num(cmd, "df", 2)?.ok_or_else(|| rc_err(198, "option df() required")) };
        let mut graph = match cmd.verb.as_str() {
            "qnorm" => qnorm_graph(&sorted_values(ds, st, cmd, v0)?, &lab),
            "pnorm" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let n = v.len() as f64;
                let (m, s) = mean_sd(&v);
                let pts: Vec<(f64, f64)> = v.iter().enumerate().map(|(i, &x)| ((i as f64 + 1.0) / (n + 1.0), dist::normal((x - m) / s))).collect();
                qplot(pts, vec![(0.0, 0.0), (1.0, 1.0)], "Empirical P[i] = i/(N+1)", &format!("Normal F[({v0}-m)/s]"))
            }
            "qchi" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let (n, k) = (v.len() as f64, df()?);
                let pts: Vec<(f64, f64)> = v.iter().enumerate().map(|(i, &x)| (dist::invchi2(k, (i as f64 + 1.0) / (n + 1.0)), x)).collect();
                let refl = diagonal(&pts);
                qplot(pts, refl, &format!("Expected Chi-Squared d.f. = {k}"), &lab)
            }
            "pchi" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let (n, k) = (v.len() as f64, df()?);
                let pts: Vec<(f64, f64)> = v.iter().enumerate().map(|(i, &x)| ((i as f64 + 1.0) / (n + 1.0), dist::chi2(k, x))).collect();
                qplot(pts, vec![(0.0, 0.0), (1.0, 1.0)], "Empirical P[i] = i/(N+1)", &format!("Chi-squared({v0}) d.f. = {k}"))
            }
            "quantile" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let n = v.len() as f64;
                let pts: Vec<(f64, f64)> = v.iter().enumerate().map(|(i, &x)| ((i as f64 + 0.5) / n, x)).collect();
                // reference: the quantiles of a uniform distribution over the data range
                let refl = vec![(0.0, v[0]), (1.0, v[v.len() - 1])];
                qplot(pts, refl, "Fraction of the data", &format!("Quantiles of {lab}"))
            }
            "qqplot" => {
                let y = sorted_values(ds, st, cmd, v0)?;
                let x = sorted_values(ds, st, cmd, &vars[1])?;
                // quantiles of the larger sample interpolated at the smaller sample's plotting positions
                let q = |s: &[f64], p: f64| util::pctile_sorted(s, p * 100.0);
                let m = y.len().min(x.len());
                let pts: Vec<(f64, f64)> = (0..m).map(|i| {
                    let p = (i as f64 + 0.5) / m as f64;
                    let xv = if x.len() == m { x[i] } else { q(&x, p) };
                    let yv = if y.len() == m { y[i] } else { q(&y, p) };
                    (xv, yv)
                }).collect();
                let refl = diagonal(&pts);
                qplot(pts, refl, &label_of(ds, &vars[1]), &lab)
            }
            "symplot" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let n = v.len();
                let med = util::pctile_sorted(&v, 50.0);
                let pts: Vec<(f64, f64)> = (0..n / 2).map(|i| (med - v[i], v[n - 1 - i] - med)).collect();
                let refl = diagonal(&pts);
                qplot(pts, refl, "Distance below median", "Distance above median")
            }
            "gladder" | "qladder" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let children = LADDER
                    .iter()
                    .map(|(name, f)| {
                        let mut t: Vec<f64> = v.iter().map(|&x| f(x)).filter(|x| x.is_finite()).collect();
                        t.sort_by(f64::total_cmp);
                        let mut g = if t.len() < 2 {
                            Graph::default()
                        } else if cmd.verb == "gladder" {
                            Graph { layers: vec![histogram_layer(&t, cmd.has_opt("fraction", 4))], ytitle: if cmd.has_opt("fraction", 4) { "Fraction".into() } else { "Density".into() }, ..Default::default() }
                        } else {
                            qnorm_graph(&t, name)
                        };
                        g.title = name.to_string();
                        g.legend = false;
                        g
                    })
                    .collect();
                Graph { children, cols: 3, title: format!("{lab}: ladder of powers"), ..Default::default() }
            }
            "spikeplot" => {
                let v = sorted_values(ds, st, cmd, v0)?;
                let mut ranges: Vec<(f64, f64, f64)> = Vec::new();
                for x in v {
                    match ranges.last_mut() {
                        Some(r) if r.0 == x => r.2 += 1.0,
                        _ => ranges.push((x, 0.0, 1.0)),
                    }
                }
                let l = Layer { kind: LayerKind::Range(RangeStyle::Spike), ranges, label: "Frequency".into(), ..Default::default() };
                Graph { layers: vec![l], xtitle: lab.clone(), ytitle: "Frequency".into(), ..Default::default() }
            }
            "sunflower" => sunflower(ds, st, cmd, v0, &vars[1])?,
            other => return Err(rc_err(199, format!("command {other} is unrecognized"))),
        };
        if graph.children.is_empty() {
            graph.legend = false;
        }
        graph::finish(&mut graph, cmd, st)
    }
}

/// Sunflower plot (Cleveland & McGill 1984): observations are binned into a
/// grid; cells holding several observations are drawn as a sunflower with
/// one petal per observation, others as plain markers.
fn sunflower(ds: &DataSet, st: &Store, cmd: &Command, y: &str, x: &str) -> Result<Graph, String> {
    let mask = util::touse(ds, st, cmd)?;
    let (yc, xc) = (util::col(ds, y)?, util::col(ds, x)?);
    let pts: Vec<(f64, f64)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(xc[i])).map(|i| (xc[i], yc[i])).collect();
    if pts.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let range = |f: fn(&(f64, f64)) -> f64| {
        let lo = pts.iter().map(f).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
        (lo, (hi - lo).max(1e-12))
    };
    let ((x0, xr), (y0, yr)) = (range(|p| p.0), range(|p| p.1));
    let bins = util::opt_num(cmd, "binwidth", 4)?.map_or(40.0, |w| (xr / w).max(1.0));
    let (bw, bh) = (xr / bins, yr / bins);
    let mut cells: std::collections::BTreeMap<(i64, i64), Vec<(f64, f64)>> = Default::default();
    for p in &pts {
        cells.entry((((p.0 - x0) / bw).floor() as i64, ((p.1 - y0) / bh).floor() as i64)).or_default().push(*p);
    }
    let light = util::opt_num(cmd, "light", 2)?.unwrap_or(3.0) as usize;
    let mut singles = scatter(Vec::new(), &label_of(ds, y));
    let mut flowers = Layer { kind: LayerKind::Sunflower, label: "Sunflowers".into(), ..Default::default() };
    for ((cx, cy), members) in cells {
        if members.len() < light.max(2) {
            singles.points.extend(members);
        } else {
            flowers.points.push((x0 + (cx as f64 + 0.5) * bw, y0 + (cy as f64 + 0.5) * bh));
            flowers.sizes.push(members.len() as f64);
        }
    }
    Ok(Graph { layers: vec![singles, flowers], xtitle: label_of(ds, x), ytitle: label_of(ds, y), ..Default::default() })
}
