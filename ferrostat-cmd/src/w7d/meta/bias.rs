//! Small-study effects: regression-based tests (Egger, Harbord, Peters),
//! the rank-correlation test (Begg) and the trim-and-fill method.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::pool::{self, Model};
use super::summ::opts;
use super::{settings, studies};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;

/// Weighted least squares of y on [1, x]: (b, V, residual df) with the
/// variance scaled by the residual mean square.
pub fn wls(x: &[f64], y: &[f64], w: &[f64]) -> Result<([f64; 2], [[f64; 2]; 2], f64), String> {
    let n = y.len();
    if n < 3 {
        return Err(rc_err(2001, "insufficient observations (at least 3 studies required)"));
    }
    let (mut s0, mut s1, mut s2, mut t0, mut t1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        s0 += w[i];
        s1 += w[i] * x[i];
        s2 += w[i] * x[i] * x[i];
        t0 += w[i] * y[i];
        t1 += w[i] * x[i] * y[i];
    }
    let det = s0 * s2 - s1 * s1;
    if det.abs() < 1e-14 * (s0 * s2).abs().max(1e-300) {
        return Err(rc_err(459, "regressor is constant across studies"));
    }
    let b1 = (s0 * t1 - s1 * t0) / det;
    let b0 = (t0 - b1 * s1) / s0;
    let rss: f64 = (0..n).map(|i| w[i] * (y[i] - b0 - b1 * x[i]).powi(2)).sum();
    let df = n as f64 - 2.0;
    let s2e = rss / df;
    let v = [[s2e * s2 / det, -s2e * s1 / det], [-s2e * s1 / det, s2e * s0 / det]];
    Ok(([b0, b1], v, df))
}

pub fn bias(ds: &mut DataSet, st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let s = settings(ds)?;
    let d = studies(ds, st, cmd, &s, &[])?;
    let k = d.k();
    let which = ["egger", "harbord", "peters", "begg"].into_iter().find(|t| cmd.has_opt(t, 3)).ok_or_else(|| rc_err(198, "one of egger, harbord, peters or begg is required"))?;
    if !rest.trim().is_empty() {
        return Err(rc_err(198, "moderators are not supported; use meta regress"));
    }
    let mut out = vec![String::new()];
    let r = st.begin_r();
    r.set("N", k as f64);
    r.set("K", k as f64);
    r.set_str("method", which);
    let reg = |title: &str, label: &str, b: f64, se: f64, df: f64, beta1: f64, out: &mut Vec<String>, st: &mut Store| {
        let t = b / se;
        let p = 2.0 * crate::dist::ttail(df, t.abs());
        out.push(format!("{title}"));
        out.push(String::new());
        out.push(format!("  H0: beta1 = 0; no small-study effects"));
        out.push(format!("            beta1 = {:>8.2}", beta1));
        out.push(format!("      SE of beta1 = {:>8.3}", se));
        out.push(format!("                t = {:>8.2}", t));
        out.push(format!("       Prob > |t| = {:>8.4}", p));
        let _ = label;
        st.r.set("beta1", beta1);
        st.r.set("se_bias", se);
        st.r.set("bias", b);
        st.r.set("t", t);
        st.r.set("df", df);
        st.r.set("p", p);
    };
    match which {
        "egger" => {
            if cmd.has_opt("random", 4) || cmd.has_opt("common", 4) || cmd.has_opt("fixed", 4) {
                // meta-regression of the effect sizes on their standard errors
                let o = opts(cmd, st, &s)?;
                let v = d.v();
                let (b, vb, _) = super::reg::fit_mr(&d.y, &v, &d.se.iter().map(|x| vec![1.0, *x]).collect::<Vec<_>>(), o.model, &o.method)?;
                let se = vb[(1, 1)].sqrt();
                let z = b[1] / se;
                let p = 2.0 * crate::dist::normal(-z.abs());
                out.push(format!("Regression-based Egger test for small-study effects ({})", o.model.title()));
                out.push(String::new());
                out.push("  H0: beta1 = 0; no small-study effects".into());
                out.push(format!("            beta1 = {:>8.2}", b[1]));
                out.push(format!("      SE of beta1 = {:>8.3}", se));
                out.push(format!("                z = {:>8.2}", z));
                out.push(format!("       Prob > |z| = {:>8.4}", p));
                st.r.set("beta1", b[1]);
                st.r.set("se_bias", se);
                st.r.set("z", z);
                st.r.set("p", p);
            } else {
                let x: Vec<f64> = d.se.iter().map(|s| 1.0 / s).collect();
                let y: Vec<f64> = d.y.iter().zip(&d.se).map(|(a, s)| a / s).collect();
                let (b, v, df) = wls(&x, &y, &vec![1.0; k])?;
                reg("Egger's test for small-study effects (traditional)", "bias", b[0], v[0][0].sqrt(), df, b[0], &mut out, st);
                st.r.set("slope", b[1]);
            }
        }
        "harbord" | "peters" => {
            let cells = d.cells.as_ref().filter(|_| s.es == "lnor" || s.es == "lnorpeto").ok_or_else(|| rc_err(321, format!("{which} test requires binary data with log odds ratios")))?;
            if which == "harbord" {
                let mut x = Vec::new();
                let mut y = Vec::new();
                for &[a, b, c, dd] in cells {
                    let (n1, n2) = (a + b, c + dd);
                    let n = n1 + n2;
                    let m1 = a + c;
                    let e = n1 * m1 / n;
                    let v = n1 * n2 * m1 * (n - m1) / (n * n * (n - 1.0));
                    if v > 0.0 {
                        x.push(v.sqrt());
                        y.push((a - e) / v.sqrt());
                    }
                }
                let (b, v, df) = wls(&x, &y, &vec![1.0; x.len()])?;
                reg("Harbord's test for small-study effects", "bias", b[0], v[0][0].sqrt(), df, b[0], &mut out, st);
            } else {
                let x: Vec<f64> = cells.iter().map(|c| 1.0 / c.iter().sum::<f64>()).collect();
                let w: Vec<f64> = cells.iter().map(|&[a, b, c, dd]| 1.0 / (1.0 / (a + c) + 1.0 / (b + dd))).collect();
                let (b, v, df) = wls(&x, &d.y, &w)?;
                reg("Peters's test for small-study effects", "beta1", b[1], v[1][1].sqrt(), df, b[1], &mut out, st);
            }
        }
        _ => {
            // Begg: Kendall's tau between standardized deviates and variances
            let v = d.v();
            let w: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
            let sw: f64 = w.iter().sum();
            let th = d.y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
            let zs: Vec<f64> = (0..k).map(|i| (d.y[i] - th) / (v[i] - 1.0 / sw).max(1e-300).sqrt()).collect();
            let mut score = 0.0;
            for i in 0..k {
                for j in i + 1..k {
                    let s1 = (zs[i] - zs[j]).signum() * if zs[i] == zs[j] { 0.0 } else { 1.0 };
                    let s2 = (v[i] - v[j]).signum() * if v[i] == v[j] { 0.0 } else { 1.0 };
                    score += s1 * s2;
                }
            }
            let kf = k as f64;
            let se = (kf * (kf - 1.0) * (2.0 * kf + 5.0) / 18.0).sqrt();
            let z = score / se;
            let p = 2.0 * crate::dist::normal(-z.abs());
            let tau = score / (kf * (kf - 1.0) / 2.0);
            out.push("Begg's test for small-study effects".into());
            out.push(String::new());
            out.push(format!("    Kendall's score = {:>8.2}", score));
            out.push(format!("         SE of score = {:>8.3}", se));
            out.push(format!("                   z = {:>8.2}", z));
            out.push(format!("          Prob > |z| = {:>8.4}", p));
            st.r.set("score", score);
            st.r.set("se_score", se);
            st.r.set("z", z);
            st.r.set("p", p);
            st.r.set("tau", tau);
        }
    }
    Ok(out.join("\n"))
}

/// Number of missing studies on the right of `theta` (data already flipped
/// so that missing studies are on the right... i.e. trimming the largest).
fn k0_est(y: &[f64], theta: f64, est: &str) -> f64 {
    let n = y.len();
    let scale = y.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    let d: Vec<f64> = y.iter().map(|v| v - theta).map(|v| if v.abs() < 1e-10 * scale { 0.0 } else { v }).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs()));
    // ranks of |d| with ties averaged
    let mut rank = vec![0.0; n];
    let mut i0 = 0;
    while i0 < n {
        let mut i1 = i0;
        while i1 + 1 < n && (d[idx[i1 + 1]].abs() - d[idx[i0]].abs()).abs() <= 1e-10 * scale {
            i1 += 1;
        }
        let avg = (i0 + i1) as f64 / 2.0 + 1.0;
        for &i in &idx[i0..=i1] {
            rank[i] = avg;
        }
        i0 = i1 + 1;
    }
    let nf = n as f64;
    match est {
        "run" => {
            // length of the rightmost run of positive deviations among ranks
            let mut g = 0;
            for &i in idx.iter().rev() {
                if d[i] > 0.0 {
                    g += 1;
                } else {
                    break;
                }
            }
            (g as f64 - 1.0).max(0.0)
        }
        "quadratic" => {
            let tn: f64 = (0..n).filter(|&i| d[i] > 0.0).map(|i| rank[i]).sum();
            let disc = (2.0 * nf + 1.0).powi(2) - 8.0 * (2.0 * tn - nf * (nf + 1.0) / 2.0).max(0.0) * 0.0;
            let _ = disc;
            // Q0 = n - 1/2 - sqrt(2n^2 - 4Tn + 1/4)
            (nf - 0.5 - (2.0 * nf * nf - 4.0 * tn + 0.25).max(0.0).sqrt()).max(0.0)
        }
        _ => {
            let tn: f64 = (0..n).filter(|&i| d[i] > 0.0).map(|i| rank[i]).sum();
            ((4.0 * tn - nf * (nf + 1.0)) / (2.0 * nf - 1.0)).max(0.0)
        }
    }
}

pub fn trimfill(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    let o = opts(cmd, st, &s)?;
    let d = studies(ds, st, cmd, &s, &[])?;
    let k = d.k();
    if k < 3 {
        return Err(rc_err(2001, "trim-and-fill requires at least 3 studies"));
    }
    let est = cmd.opt("estimator", 3).map(|x| x.trim().to_lowercase()).unwrap_or_else(|| "linear".into());
    let est = match est.as_str() {
        "linear" | "l" => "linear",
        "run" | "r" => "run",
        "quadratic" | "q" => "quadratic",
        o => return Err(rc_err(198, format!("estimator({o}) not allowed"))),
    };
    let v = d.v();
    // iteration model: the declared or requested one
    let pooled = |y: &[f64], v: &[f64]| -> Result<f64, String> { Ok(pool::pool(y, v, o.model, if o.model == Model::Random { &o.method } else { "iv" }, o.level, None, &s.es, None)?.theta) };
    // side with the missing studies: from the sign of the Egger slope unless given
    let left = if cmd.has_opt("left", 4) {
        true
    } else if cmd.has_opt("right", 5) {
        false
    } else {
        let x: Vec<f64> = d.se.iter().map(|s| 1.0 / s).collect();
        let yy: Vec<f64> = d.y.iter().zip(&d.se).map(|(a, s)| a / s).collect();
        wls(&x, &yy, &vec![1.0; k]).map(|(b, _, _)| b[0] > 0.0).unwrap_or(true)
    };
    // flip so that the missing studies are on the left of the trimmed ones
    let sign = if left { 1.0 } else { -1.0 };
    let y: Vec<f64> = d.y.iter().map(|a| sign * a).collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
    let mut k0 = 0usize;
    let mut theta = pooled(&y, &v)?;
    for _ in 0..100 {
        let keep: Vec<usize> = order[..k - k0].to_vec();
        let yt: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
        let vt: Vec<f64> = keep.iter().map(|&i| v[i]).collect();
        theta = pooled(&yt, &vt)?;
        let nk = (k0_est(&y, theta, est).round() as usize).min(k - 2);
        if nk == k0 {
            break;
        }
        k0 = nk;
    }
    // fill: mirror the k0 largest studies about the trimmed estimate
    let mut yf = y.clone();
    let mut vf = v.clone();
    let mut imputed = Vec::new();
    for &i in order.iter().rev().take(k0) {
        let yi = 2.0 * theta - y[i];
        yf.push(yi);
        vf.push(v[i]);
        imputed.push((sign * yi, v[i].sqrt()));
    }
    let yf: Vec<f64> = yf.iter().map(|a| sign * a).collect();
    let pf = pool::pool(&yf, &vf, o.model, if o.model == Model::Random { &o.method } else { "iv" }, o.level, None, &s.es, None)?;
    let p0 = pool::pool(&d.y, &v, o.model, if o.model == Model::Random { &o.method } else { "iv" }, o.level, None, &s.es, None)?;
    let t = |x: f64| if o.eform { x.exp() } else { x };
    let mut out = vec![String::new(), "Nonparametric trim-and-fill analysis of publication bias".to_string()];
    out.push(format!("{} estimator, imputing on the {}", match est { "run" => "Run", "quadratic" => "Quadratic", _ => "Linear" }, if left { "left" } else { "right" }));
    out.push(String::new());
    out.push(format!("Iteration{:>37}Number of studies = {:>5}", "", k + k0));
    out.push(format!("  Model: {}{:>27}observed = {:>5}", o.model.title(), "", k));
    out.push(format!("{:>46}imputed = {:>5}", "", k0));
    out.push(String::new());
    out.push("-".repeat(62));
    out.push(format!("{:>20} | {:>11}  [{}% conf. interval]", "Studies", "Effect size", crate::format::fmt_g(o.level, 4)));
    out.push(format!("{}+{}", "-".repeat(21), "-".repeat(40)));
    out.push(format!("{:>20} | {:>11.3}  {:>10.3}  {:>10.3}", "Observed", t(p0.theta), t(p0.lo), t(p0.hi)));
    out.push(format!("{:>20} | {:>11.3}  {:>10.3}  {:>10.3}", "Observed + Imputed", t(pf.theta), t(pf.lo), t(pf.hi)));
    out.push("-".repeat(62));
    let r = st.begin_r();
    r.set("K_observed", k as f64);
    r.set("K_imputed", k0 as f64);
    r.set("K_total", (k + k0) as f64);
    r.set("theta_observed", p0.theta);
    r.set("theta", pf.theta);
    r.set("se", pf.se);
    r.set("ci_lb", pf.lo);
    r.set("ci_ub", pf.hi);
    r.set_str("side", if left { "left" } else { "right" });
    r.set_str("estimator", est);
    if !imputed.is_empty() {
        r.set_mat("imputed", Matrix::from_rows(&imputed.iter().map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>()).with_names((1..=imputed.len()).map(|i| format!("r{i}")).collect(), vec!["_meta_es".into(), "_meta_se".into()]));
    }
    if cmd.has_opt("funnel", 3) {
        let mut g = crate::graph::Graph::default();
        g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Scatter, points: (0..k).map(|i| (d.y[i], -d.se[i])).collect(), label: "Observed studies".into(), ..Default::default() });
        g.layers.push(crate::graph::Layer { kind: crate::graph::LayerKind::Scatter, points: imputed.iter().map(|(a, b)| (*a, -b)).collect(), hollow: true, label: "Imputed studies".into(), ..Default::default() });
        g.ytitle = "Standard error (negated)".into();
        g.xtitle = s.eslabel.clone();
        g.title = "Funnel plot".into();
        g.legend = true;
        g.command = "meta trimfill, funnel".into();
        crate::graph::finish(&mut g, cmd, st)?;
    }
    Ok(out.join("\n"))
}
