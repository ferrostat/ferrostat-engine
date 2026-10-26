//! ROC analysis: `roctab` (empirical ROC curve, area by the trapezoidal
//! rule; also the Lorenz-like curve), `roccomp` and `rocgold` (comparison of
//! correlated ROC areas) and `rocfit` (binormal model by maximum likelihood).
//!
//! Standard errors and comparisons of areas use the nonparametric method of
//! DeLong, DeLong & Clarke-Pearson (1988, *Biometrics* 44: 837-845); the
//! binormal model is that of Dorfman & Alf (1969, *J. Math. Psychology* 6:
//! 487-496).

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::varlist;

/// One rating with its weight, split by true status.
struct Ratings {
    pos: Vec<(f64, f64)>,
    neg: Vec<(f64, f64)>,
}

fn ratings(ds: &DataSet, st: &Store, cmd: &Command, refvar: &str, class: &[String]) -> Result<Vec<Ratings>, String> {
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "f")?;
    let r = util::col(ds, refvar)?;
    let cols: Vec<Vec<f64>> = class.iter().map(|c| util::col(ds, c)).collect::<Result<_, _>>()?;
    // the same observations for every rating variable
    let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(r[i]) && cols.iter().all(|c| !is_missing(c[i]))).collect();
    if !rows.iter().any(|&i| r[i] != 0.0) || !rows.iter().any(|&i| r[i] == 0.0) {
        return Err(rc_err(2000, "outcome does not vary"));
    }
    Ok(cols
        .iter()
        .map(|c| {
            let wt = |i: usize| w.as_ref().map_or(1.0, |w| w.w[i]);
            Ratings { pos: rows.iter().filter(|&&i| r[i] != 0.0).map(|&i| (c[i], wt(i))).collect(), neg: rows.iter().filter(|&&i| r[i] == 0.0).map(|&i| (c[i], wt(i))).collect() }
        })
        .collect())
}

/// Empirical ROC points (1 - specificity, sensitivity) from (1,1) to (0,0).
fn roc_points(r: &Ratings) -> Vec<(f64, f64)> {
    let mut cuts: Vec<f64> = r.pos.iter().chain(&r.neg).map(|p| p.0).collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let (np, nn): (f64, f64) = (r.pos.iter().map(|p| p.1).sum(), r.neg.iter().map(|p| p.1).sum());
    let mut pts: Vec<(f64, f64)> = cuts
        .iter()
        .map(|&c| {
            let sens = r.pos.iter().filter(|p| p.0 >= c).map(|p| p.1).sum::<f64>() / np;
            let fpr = r.neg.iter().filter(|p| p.0 >= c).map(|p| p.1).sum::<f64>() / nn;
            (fpr, sens)
        })
        .collect();
    pts.push((0.0, 0.0));
    pts
}

/// DeLong placement values: per positive, P(negative < it) + 0.5 P(tie); per negative, likewise.
fn placements(r: &Ratings) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let psi = |x: f64, y: f64| if x > y { 1.0 } else if x == y { 0.5 } else { 0.0 };
    let (np, nn): (f64, f64) = (r.pos.iter().map(|p| p.1).sum(), r.neg.iter().map(|p| p.1).sum());
    let v10 = r.pos.iter().map(|&(x, w)| (r.neg.iter().map(|&(y, wy)| wy * psi(x, y)).sum::<f64>() / nn, w)).collect();
    let v01 = r.neg.iter().map(|&(y, w)| (r.pos.iter().map(|&(x, wx)| wx * psi(x, y)).sum::<f64>() / np, w)).collect();
    (v10, v01)
}

/// Areas and their DeLong covariance matrix for ratings of the same subjects.
fn delong(rs: &[Ratings]) -> (Vec<f64>, Matrix) {
    let k = rs.len();
    let pl: Vec<(Vec<(f64, f64)>, Vec<(f64, f64)>)> = rs.iter().map(placements).collect();
    let wmean = |v: &[(f64, f64)]| v.iter().map(|(x, w)| x * w).sum::<f64>() / v.iter().map(|p| p.1).sum::<f64>();
    let auc: Vec<f64> = pl.iter().map(|(v10, _)| wmean(v10)).collect();
    let cov = |a: &[(f64, f64)], b: &[(f64, f64)], ma: f64, mb: f64| {
        let n: f64 = a.iter().map(|p| p.1).sum();
        a.iter().zip(b).map(|(x, y)| x.1 * (x.0 - ma) * (y.0 - mb)).sum::<f64>() / (n - 1.0)
    };
    let (np, nn): (f64, f64) = (rs[0].pos.iter().map(|p| p.1).sum(), rs[0].neg.iter().map(|p| p.1).sum());
    let mut s = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            s[(i, j)] = cov(&pl[i].0, &pl[j].0, auc[i], auc[j]) / np + cov(&pl[i].1, &pl[j].1, auc[i], auc[j]) / nn;
        }
    }
    (auc, s)
}

fn roc_graph(curves: Vec<(String, Vec<(f64, f64)>)>) -> Graph {
    let mut layers: Vec<Layer> = curves.into_iter().map(|(label, pts)| Layer { kind: LayerKind::Connected, points: pts, label, ..Default::default() }).collect();
    layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (1.0, 1.0)], label: "Reference".into(), ..Default::default() });
    Graph { layers, xtitle: "1 - Specificity".into(), ytitle: "Sensitivity".into(), legend: true, ..Default::default() }
}

fn area_line(name: &str, n: f64, auc: f64, se: f64, z: f64) -> String {
    format!("{:>12} {:>8} {:>10.4} {:>10.4} {:>12.5} {:>10.5}", name, n, auc, se, auc - z * se, auc + z * se)
}

pub struct Roc;

impl CommandHandler for Roc {
    fn name(&self) -> &'static str {
        "roctab"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let header = format!("{:>12} {:>8} {:>10} {:>10} {:>23}", "", "Obs", "ROC area", "Std. err.", format!("[{}% conf. interval]", fmt_g(level, 4)));
        let (text, mut graph) = match cmd.verb.as_str() {
            "roctab" => {
                if vars.len() != 2 {
                    return Err(rc_err(102, "roctab refvar classvar"));
                }
                let rs = ratings(ds, st, cmd, &vars[0], &vars[1..])?;
                let r = &rs[0];
                let n: f64 = r.pos.iter().chain(&r.neg).map(|p| p.1).sum();
                if cmd.has_opt("lorenz", 3) {
                    // order classes from most to least indicative of the outcome
                    let mut cuts: Vec<f64> = r.pos.iter().chain(&r.neg).map(|p| p.0).collect();
                    cuts.sort_by(|a, b| b.total_cmp(a));
                    cuts.dedup();
                    let np: f64 = r.pos.iter().map(|p| p.1).sum();
                    let mut pts = vec![(0.0, 0.0)];
                    let (mut cx, mut cy) = (0.0, 0.0);
                    for c in cuts {
                        cx += r.pos.iter().chain(&r.neg).filter(|p| p.0 == c).map(|p| p.1).sum::<f64>() / n;
                        cy += r.pos.iter().filter(|p| p.0 == c).map(|p| p.1).sum::<f64>() / np;
                        pts.push((cx, cy));
                    }
                    let area: f64 = pts.windows(2).map(|w| (w[1].0 - w[0].0) * (w[1].1 + w[0].1) / 2.0).sum();
                    let gini = 2.0 * area - 1.0;
                    let pietra = pts.iter().map(|p| p.1 - p.0).fold(0.0, f64::max);
                    st.begin_r().set("gini", gini);
                    st.r.set("pietra", pietra);
                    let g = Graph {
                        layers: vec![Layer { kind: LayerKind::Connected, points: pts, label: "Lorenz curve".into(), ..Default::default() }, Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (1.0, 1.0)], label: "Equality".into(), ..Default::default() }],
                        xtitle: "Cumulative fraction of the population".into(),
                        ytitle: format!("Cumulative fraction of {}", vars[0]),
                        legend: true,
                        ..Default::default()
                    };
                    (format!("Lorenz curve\n\n Pietra index = {:>8.4}\n Gini index   = {:>8.4}", pietra, gini), g)
                } else {
                    let (auc, s) = delong(&rs);
                    let se = s[(0, 0)].sqrt();
                    st.begin_r().set("area", auc[0]);
                    st.r.set("se", se);
                    st.r.set("N", n);
                    (format!("{header}\n{}", area_line("", n, auc[0], se, z)), roc_graph(vec![(format!("Area under ROC curve = {:.4}", auc[0]), roc_points(r))]))
                }
            }
            "roccomp" | "rocgold" => {
                let min = if cmd.verb == "rocgold" { 3 } else { 3 };
                if vars.len() < min {
                    return Err(rc_err(102, format!("{} refvar classvar1 classvar2 ...", cmd.verb)));
                }
                let class = &vars[1..];
                let rs = ratings(ds, st, cmd, &vars[0], class)?;
                let n: f64 = rs[0].pos.iter().chain(&rs[0].neg).map(|p| p.1).sum();
                let (auc, s) = delong(&rs);
                let mut out = vec![header.clone(), "-".repeat(78)];
                for (i, c) in class.iter().enumerate() {
                    out.push(area_line(c, n, auc[i], s[(i, i)].sqrt(), z));
                }
                out.push("-".repeat(78));
                let k = class.len();
                if cmd.verb == "roccomp" {
                    // H0: all areas equal, contrasts of each area with the last
                    let mut l = Matrix::zeros(k - 1, k);
                    for i in 0..k - 1 {
                        l[(i, i)] = 1.0;
                        l[(i, k - 1)] = -1.0;
                    }
                    let la = l.mul(&Matrix::col_vector(&auc)).unwrap();
                    let lsl = l.mul(&s).unwrap().mul(&l.transpose()).unwrap();
                    let chi2 = la.transpose().mul(&lsl.sym_inverse(1e-14).0).unwrap().mul(&la).unwrap()[(0, 0)];
                    let p = dist::chi2tail((k - 1) as f64, chi2);
                    out.push(format!("Ho: area({}) = {}", class[0], (2..=k).map(|i| format!("area({})", class[i - 1])).collect::<Vec<_>>().join(" = ")));
                    out.push(format!("    chi2({}) = {:>8.2}       Prob>chi2 = {:>8.4}", k - 1, chi2, p));
                    st.begin_r().set("chi2", chi2);
                    st.r.set("df", (k - 1) as f64);
                    st.r.set("p", p);
                } else {
                    // each rating against the gold standard (the first), Bonferroni-adjusted
                    out.push(format!("{:>12} {:>10} {:>10} {:>12}", "", "chi2(1)", "Prob>chi2", "Bonferroni"));
                    for i in 1..k {
                        let d = auc[i] - auc[0];
                        let v = s[(i, i)] + s[(0, 0)] - 2.0 * s[(0, i)];
                        let chi2 = d * d / v;
                        let p = dist::chi2tail(1.0, chi2);
                        out.push(format!("{:>12} {:>10.4} {:>10.4} {:>12.4}", class[i], chi2, p, (p * (k - 1) as f64).min(1.0)));
                    }
                }
                let curves = class.iter().zip(&rs).zip(&auc).map(|((c, r), a)| (format!("{c} ROC area: {:.4}", a), roc_points(r))).collect();
                (out.join("\n"), roc_graph(curves))
            }
            "rocfit" => {
                if vars.len() != 2 {
                    return Err(rc_err(102, "rocfit refvar classvar"));
                }
                let rs = ratings(ds, st, cmd, &vars[0], &vars[1..])?;
                rocfit(st, &rs[0], z, &header, &vars[0], &vars[1])?
            }
            other => return Err(rc_err(199, format!("command {other} is unrecognized"))),
        };
        if !cmd.has_opt("graph", 2) && cmd.verb != "rocfit" {
            return Ok(text);
        }
        let extra = graph::finish(&mut graph, cmd, st)?;
        Ok([text, extra].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"))
    }
}

/// Binormal ROC model: ratings are a monotone transform of latent normal
/// variables, N(0,1) for negatives and N(a/b, 1/b²) for positives; with
/// cutpoints z_1 < ... < z_{K-1}, P(rating > k | negative) = 1 - Φ(z_k) and
/// P(rating > k | positive) = 1 - Φ(b z_k - a). The ROC curve is
/// TPF = Φ(a + b Φ⁻¹(FPF)) and its area Φ(a / sqrt(1 + b²)).
fn rocfit(st: &mut Store, r: &Ratings, z: f64, header: &str, refvar: &str, classvar: &str) -> Result<(String, Graph), String> {
    let mut cats: Vec<f64> = r.pos.iter().chain(&r.neg).map(|p| p.0).collect();
    cats.sort_by(f64::total_cmp);
    cats.dedup();
    let k = cats.len();
    if k < 3 {
        return Err(rc_err(459, "rocfit requires at least three rating categories"));
    }
    let counts = |v: &[(f64, f64)]| -> Vec<f64> { cats.iter().map(|c| v.iter().filter(|p| p.0 == *c).map(|p| p.1).sum()).collect() };
    let (cp, cn) = (counts(&r.pos), counts(&r.neg));
    // parameters: a, ln b, z_1, ln(z_2 - z_1), ..., ln(z_{K-1} - z_{K-2})
    let unpack = |th: &[f64]| -> (f64, f64, Vec<f64>) {
        let mut zs = vec![th[2]];
        for j in 3..th.len() {
            zs.push(zs[zs.len() - 1] + th[j].exp());
        }
        (th[0], th[1].exp(), zs)
    };
    let ll = |th: &[f64]| -> f64 {
        let (a, b, zs) = unpack(th);
        let cdf = |j: usize, pos: bool| -> f64 {
            if j == 0 {
                return 0.0;
            }
            if j == k {
                return 1.0;
            }
            if pos { dist::normal(b * zs[j - 1] - a) } else { dist::normal(zs[j - 1]) }
        };
        (0..k).map(|j| cn[j] * (cdf(j + 1, false) - cdf(j, false)).max(1e-300).ln() + cp[j] * (cdf(j + 1, true) - cdf(j, true)).max(1e-300).ln()).sum()
    };
    // starting values from the empirical cumulative probits
    let (np, nn): (f64, f64) = (cp.iter().sum(), cn.iter().sum());
    let mut zs0 = Vec::new();
    let (mut acc, mut accp) = (0.0, 0.0);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for j in 0..k - 1 {
        acc += cn[j];
        accp += cp[j];
        let zn = dist::invnormal((acc / nn).clamp(0.01, 0.99));
        zs0.push(zn);
        xs.push(-zn);
        ys.push(-dist::invnormal((accp / np).clamp(0.01, 0.99)));
    }
    for j in 1..zs0.len() {
        if zs0[j] <= zs0[j - 1] {
            zs0[j] = zs0[j - 1] + 0.1;
        }
    }
    let (mx, my) = (util::mean(&xs), util::mean(&ys));
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let b0 = if sxx > 0.0 && sxy > 0.0 { sxy / sxx } else { 1.0 };
    let mut th = vec![my - b0 * mx, b0.ln(), zs0[0]];
    th.extend(zs0.windows(2).map(|w| (w[1] - w[0]).max(1e-3).ln()));
    // Newton-Raphson with numerical derivatives and step halving
    let p = th.len();
    let hess = |th: &[f64]| -> (Vec<f64>, Matrix) {
        let h = 1e-4;
        let f0 = ll(th);
        let mut g = vec![0.0; p];
        let mut hm = Matrix::zeros(p, p);
        let shifted = |i: usize, di: f64, j: usize, dj: f64| {
            let mut t = th.to_vec();
            t[i] += di;
            t[j] += dj;
            ll(&t)
        };
        for i in 0..p {
            g[i] = (shifted(i, h, i, 0.0) - shifted(i, -h, i, 0.0)) / (2.0 * h);
            for j in 0..=i {
                let v = if i == j {
                    (shifted(i, h, i, 0.0) - 2.0 * f0 + shifted(i, -h, i, 0.0)) / (h * h)
                } else {
                    (shifted(i, h, j, h) - shifted(i, h, j, -h) - shifted(i, -h, j, h) + shifted(i, -h, j, -h)) / (4.0 * h * h)
                };
                hm[(i, j)] = v;
                hm[(j, i)] = v;
            }
        }
        (g, hm)
    };
    let mut iter = 0;
    loop {
        let (g, hm) = hess(&th);
        let (hi, _) = hm.scale(-1.0).sym_inverse(1e-12);
        let step: Vec<f64> = (0..p).map(|i| (0..p).map(|j| hi[(i, j)] * g[j]).sum()).collect();
        let f0 = ll(&th);
        let mut lambda = 1.0;
        let mut moved = false;
        while lambda > 1e-6 {
            let cand: Vec<f64> = th.iter().zip(&step).map(|(t, s)| t + lambda * s).collect();
            if ll(&cand) >= f0 - 1e-12 {
                moved = (ll(&cand) - f0).abs() > 1e-10;
                th = cand;
                break;
            }
            lambda /= 2.0;
        }
        iter += 1;
        if !moved || iter > 100 {
            break;
        }
    }
    let (g_, hm) = hess(&th);
    let _ = g_;
    let (cov, _) = hm.scale(-1.0).sym_inverse(1e-12);
    let (a, b, _) = unpack(&th);
    let se_a = cov[(0, 0)].sqrt();
    let se_b = b * cov[(1, 1)].sqrt();
    // area and its delta-method standard error
    let d = (1.0 + b * b).sqrt();
    let area = dist::normal(a / d);
    let phi = (-(a / d).powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let (da, dlb) = (phi / d, phi * (-a * b * b / d.powi(3)));
    let se_area = (da * da * cov[(0, 0)] + dlb * dlb * cov[(1, 1)] + 2.0 * da * dlb * cov[(0, 1)]).max(0.0).sqrt();
    let n = np + nn;
    let llv = ll(&th);
    let out = format!(
        "Binormal model of {} on {}\nNumber of obs = {}\nLog likelihood = {:.4}\n\n{:>12} {:>10} {:>10}\n{:>12} {:>10.4} {:>10.4}\n{:>12} {:>10.4} {:>10.4}\n\n{header}\n{}",
        classvar, refvar, n, llv, "", "Coef.", "Std. err.", "intercept", a, se_a, "slope", b, se_b, area_line("", n, area, se_area, z)
    );
    let r_ = st.begin_r();
    r_.set("area", area);
    r_.set("se_area", se_area);
    r_.set("a", a);
    r_.set("b", b);
    let fitted: Vec<(f64, f64)> = (0..=100).map(|i| {
        let fpf = i as f64 / 100.0;
        let tpf = if fpf <= 0.0 { 0.0 } else if fpf >= 1.0 { 1.0 } else { dist::normal(a + b * dist::invnormal(fpf)) };
        (fpf, tpf)
    }).collect();
    let mut g = roc_graph(vec![]);
    g.layers.insert(0, Layer { kind: LayerKind::Line, points: fitted, label: format!("Fitted ROC curve, area = {:.4}", area), ..Default::default() });
    g.layers.insert(1, Layer { kind: LayerKind::Scatter, points: roc_points(r), label: "Empirical".into(), ..Default::default() });
    Ok((out, g))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delong_area_matches_mann_whitney() {
        // positives 3 4 5, negatives 1 2 3: pairs won 3+3+3 - 0.5 tie... = 8.5 of 9
        let r = Ratings { pos: vec![(3.0, 1.0), (4.0, 1.0), (5.0, 1.0)], neg: vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)] };
        let (auc, _) = delong(std::slice::from_ref(&r));
        assert!((auc[0] - 8.5 / 9.0).abs() < 1e-12);
        // trapezoidal area of the empirical curve gives the same value
        let pts = roc_points(&r);
        let trap: f64 = pts.windows(2).map(|w| (w[0].0 - w[1].0) * (w[0].1 + w[1].1) / 2.0).sum();
        assert!((trap - auc[0]).abs() < 1e-12);
    }
}
