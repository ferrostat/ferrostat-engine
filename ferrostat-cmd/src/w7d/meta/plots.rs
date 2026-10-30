//! Meta-analysis graphs: forest, funnel (with significance contours),
//! L'Abbe and Galbraith plots.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::summ::{opts, run_pool};
use super::{settings, studies};
use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind, PcStyle};
use crate::store::Store;

fn finish(mut g: Graph, cmd: &Command, st: &mut Store, verb: &str) -> Result<String, String> {
    g.command = format!("meta {verb} {}", cmd.main.split_once(char::is_whitespace).map_or("", |x| x.1)).trim().to_string();
    crate::graph::finish(&mut g, cmd, st)
}

pub fn forest(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    let o = opts(cmd, st, &s)?;
    let d = studies(ds, st, cmd, &s, &[])?;
    let all: Vec<usize> = (0..d.k()).collect();
    let p = run_pool(&d, &all, &s, &o)?;
    let crit = crate::dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
    let t = |x: f64| if o.eform { x.exp() } else { x };
    let k = d.k();
    let mut g = Graph::default();
    // studies from the top: y = k+1 .. 2, overall at 0
    let ypos: Vec<f64> = (0..k).map(|i| (k - i + 1) as f64).collect();
    let segs: Vec<[f64; 4]> = (0..k).map(|i| [ypos[i], t(d.y[i] - crit * d.se[i]), ypos[i], t(d.y[i] + crit * d.se[i])]).collect();
    g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Spike), segments: segs, label: format!("{}% CI", crate::format::fmt_g(o.level, 4)), ..Default::default() });
    let wmax = p.wpct.iter().cloned().fold(0.0, f64::max).max(1e-12);
    g.layers.push(Layer {
        kind: LayerKind::Scatter,
        points: (0..k).map(|i| (t(d.y[i]), ypos[i])).collect(),
        sizes: p.wpct.iter().map(|w| 0.5 + 1.5 * (w / wmax).sqrt()).collect(),
        symbol: "square".into(),
        label: "Study".into(),
        ..Default::default()
    });
    // overall estimate as a diamond outline
    let diamond = vec![[0.0, t(p.lo), 0.4, t(p.theta)], [0.4, t(p.theta), 0.0, t(p.hi)], [0.0, t(p.hi), -0.4, t(p.theta)], [-0.4, t(p.theta), 0.0, t(p.lo)]];
    g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Spike), segments: diamond, label: "Overall".into(), ..Default::default() });
    if let Some((a, b)) = p.pi {
        g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::CapSym), segments: vec![[-0.8, t(a), -0.8, t(b)]], label: "Prediction interval".into(), ..Default::default() });
    }
    g.ylabels = (0..k).map(|i| (ypos[i], d.labels[i].clone())).collect();
    g.ylabels.push((0.0, "Overall".into()));
    g.xlines.push(if o.eform { 1.0 } else { 0.0 });
    g.xlines.push(t(p.theta));
    g.xtitle = if o.eform { super::eform_label(&s.es).into() } else { s.eslabel.clone() };
    g.title = "Forest plot".into();
    g.legend = false;
    let r = st.begin_r();
    r.set("theta", p.theta);
    r.set("K", k as f64);
    finish(g, cmd, st, "forestplot")
}

pub fn funnel(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    let mut o = opts(cmd, st, &s)?;
    if !cmd.has_opt("random", 4) && !cmd.has_opt("fixed", 4) {
        // funnel plots are centred on the common-effect estimate by default
        o.model = super::pool::Model::Common;
        o.method = "iv".into();
    }
    let d = studies(ds, st, cmd, &s, &[])?;
    let all: Vec<usize> = (0..d.k()).collect();
    let p = run_pool(&d, &all, &s, &o)?;
    let semax = d.se.iter().cloned().fold(0.0, f64::max) * 1.05;
    let mut g = Graph::default();
    // the vertical axis shows the standard error, largest at the bottom: plot -se
    g.layers.push(Layer { kind: LayerKind::Scatter, points: (0..d.k()).map(|i| (d.y[i], -d.se[i])).collect(), label: "Studies".into(), ..Default::default() });
    let contours: Vec<f64> = match cmd.opt("contours", 4) {
        Some(c) => crate::margins::numlist(c)?,
        None => vec![],
    };
    if contours.is_empty() {
        let z = crate::dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
        g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Spike), segments: vec![[0.0, p.theta, -semax, p.theta - z * semax], [0.0, p.theta, -semax, p.theta + z * semax]], label: format!("{}% pseudo CI", crate::format::fmt_g(o.level, 4)), ..Default::default() });
        g.layers.push(Layer { kind: LayerKind::Line, points: vec![(p.theta, 0.0), (p.theta, -semax)], label: "Estimated theta_IV".into(), ..Default::default() });
    } else {
        // significance contours around zero: |es| = z_{a/2} se
        for c in &contours {
            if !(*c > 0.0 && *c < 100.0) {
                return Err(rc_err(198, "contours() values must be between 0 and 100"));
            }
            let z = crate::dist::invnormal(1.0 - c / 200.0);
            g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Spike), segments: vec![[0.0, 0.0, -semax, -z * semax], [0.0, 0.0, -semax, z * semax]], label: format!("{}%", crate::format::fmt_g(*c, 4)), ..Default::default() });
        }
        g.legend = true;
    }
    let step = semax / 4.0;
    g.ylabels = (0..=4).map(|j| (-(j as f64) * step, crate::format::fmt_g(j as f64 * step, 6))).collect();
    g.ytitle = "Standard error".into();
    g.xtitle = s.eslabel.clone();
    g.title = if contours.is_empty() { "Funnel plot".into() } else { "Contour-enhanced funnel plot".into() };
    let r = st.begin_r();
    r.set("theta", p.theta);
    r.set_str("model", o.model.name());
    finish(g, cmd, st, "funnelplot")
}

pub fn labbe(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    if s.datatype != "binary" {
        return Err(rc_err(321, "meta labbeplot requires binary data declared with meta esize"));
    }
    let o = opts(cmd, st, &s)?;
    let d = studies(ds, st, cmd, &s, &[])?;
    let cells = d.cells.clone().unwrap();
    let all: Vec<usize> = (0..d.k()).collect();
    let p = run_pool(&d, &all, &s, &o)?;
    let pts: Vec<(f64, f64)> = cells.iter().map(|&[a, b, c, dd]| (c / (c + dd), a / (a + b))).collect();
    let mut g = Graph::default();
    g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, sizes: p.wpct.iter().map(|w| 0.5 + (w / 10.0).sqrt()).collect(), hollow: true, label: "Studies".into(), ..Default::default() });
    let grid: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
    g.layers.push(Layer { kind: LayerKind::Line, points: grid.iter().map(|&x| (x, x)).collect(), label: "No effect".into(), ..Default::default() });
    // pooled effect mapped to the treated-group risk
    let f = |pc: f64| -> f64 {
        match s.es.as_str() {
            "lnrr" => (pc * p.theta.exp()).min(1.0),
            "rdiff" => (pc + p.theta).clamp(0.0, 1.0),
            _ => {
                let odds = pc / (1.0 - pc) * p.theta.exp();
                odds / (1.0 + odds)
            }
        }
    };
    g.layers.push(Layer { kind: LayerKind::Line, points: grid.iter().filter(|&&x| x > 0.0 && x < 1.0).map(|&x| (x, f(x))).collect(), label: "Overall".into(), ..Default::default() });
    g.xtitle = "Control group risk".into();
    g.ytitle = "Treatment group risk".into();
    g.title = "L'Abbe plot".into();
    g.legend = true;
    let r = st.begin_r();
    r.set("theta", p.theta);
    finish(g, cmd, st, "labbeplot")
}

pub fn galbraith(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let s = settings(ds)?;
    let mut o = opts(cmd, st, &s)?;
    if !cmd.has_opt("random", 4) {
        o.model = super::pool::Model::Common;
        o.method = "iv".into();
    }
    let d = studies(ds, st, cmd, &s, &[])?;
    let all: Vec<usize> = (0..d.k()).collect();
    let p = run_pool(&d, &all, &s, &o)?;
    let prec: Vec<f64> = d.se.iter().map(|x| 1.0 / x).collect();
    let z: Vec<f64> = d.y.iter().zip(&d.se).map(|(y, s)| y / s).collect();
    let xmax = prec.iter().cloned().fold(0.0, f64::max) * 1.05;
    let crit = crate::dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
    let mut g = Graph::default();
    g.layers.push(Layer { kind: LayerKind::Scatter, points: prec.iter().cloned().zip(z.iter().cloned()).collect(), label: "Studies".into(), ..Default::default() });
    g.layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (xmax, p.theta * xmax)], label: "Regression line".into(), ..Default::default() });
    g.layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, crit), (xmax, crit + p.theta * xmax)], label: format!("{}% CI", crate::format::fmt_g(o.level, 4)), pattern: "dash".into(), ..Default::default() });
    g.layers.push(Layer { kind: LayerKind::Line, points: vec![(0.0, -crit), (xmax, -crit + p.theta * xmax)], label: String::new(), pattern: "dash".into(), ..Default::default() });
    g.xtitle = "Precision (1/se)".into();
    g.ytitle = "Standardized effect size (es/se)".into();
    g.title = "Galbraith plot".into();
    g.legend = true;
    let r = st.begin_r();
    r.set("theta", p.theta);
    finish(g, cmd, st, "galbraithplot")
}
