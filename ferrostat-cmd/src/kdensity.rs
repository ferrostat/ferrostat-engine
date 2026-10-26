//! Kernel density estimation.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::graph::{Graph, Layer, LayerKind};
use crate::store::Store;
use crate::util;
use crate::varlist;

pub fn kernel(name: &str, u: f64) -> f64 {
    match name {
        "gaussian" | "gau" => crate::dist::normalden(u),
        "epan2" => if u.abs() < 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 },
        "biweight" | "bi" => if u.abs() < 1.0 { 15.0 / 16.0 * (1.0 - u * u).powi(2) } else { 0.0 },
        "triweight" => if u.abs() < 1.0 { 35.0 / 32.0 * (1.0 - u * u).powi(3) } else { 0.0 },
        "cosine" | "cos" => if u.abs() < 0.5 { 1.0 + (2.0 * std::f64::consts::PI * u).cos() } else { 0.0 },
        "rectangle" | "rec" => if u.abs() < 1.0 { 0.5 } else { 0.0 },
        "triangle" | "tri" => if u.abs() < 1.0 { 1.0 - u.abs() } else { 0.0 },
        "parzen" | "parz" => {
            let a = u.abs();
            if a <= 0.5 { 4.0 / 3.0 - 8.0 * a * a + 8.0 * a.powi(3) } else if a <= 1.0 { 8.0 / 3.0 * (1.0 - a).powi(3) } else { 0.0 }
        }
        // epanechnikov with support +-sqrt(5)
        _ => {
            let s5 = 5f64.sqrt();
            if u.abs() < s5 { 3.0 / (4.0 * s5) * (1.0 - u * u / 5.0) } else { 0.0 }
        }
    }
}

pub fn canonical_kernel(k: &str) -> &'static str {
    match k {
        k if k.starts_with("gau") => "gaussian",
        k if k.starts_with("epan2") => "epan2",
        k if k.starts_with("bi") => "biweight",
        k if k.starts_with("triw") => "triweight",
        k if k.starts_with("cos") => "cosine",
        k if k.starts_with("rec") => "rectangle",
        k if k.starts_with("tri") => "triangle",
        k if k.starts_with("par") => "parzen",
        _ => "epanechnikov",
    }
}

pub struct Kdensity;

impl CommandHandler for Kdensity {
    fn name(&self) -> &'static str {
        "kdensity"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fa")?;
        let x = util::col(ds, &v)?;
        let pts: Vec<(f64, f64)> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(x[i])).map(|i| (x[i], w.as_ref().map_or(1.0, |w| w.w[i]))).collect();
        if pts.len() < 2 {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let sw: f64 = pts.iter().map(|p| p.1).sum();
        let kname = canonical_kernel(&cmd.opt("kernel", 1).unwrap_or("epanechnikov").trim().to_lowercase()).to_string();
        let s = crate::summarize::summarize_values(&pts.iter().map(|p| p.0).collect::<Vec<_>>(), Some(&pts.iter().map(|p| p.1).collect::<Vec<_>>()), w.as_ref().map(|w| w.kind));
        let iqr = s.pctile(75.0) - s.pctile(25.0);
        let m = if iqr > 0.0 { s.sd().min(iqr / 1.349) } else { s.sd() };
        let h = match util::opt_num(cmd, "bwidth", 1)? {
            Some(b) => b,
            None => 0.9 * m * s.n.powf(-0.2),
        };
        let grid: Vec<f64> = if let Some(at) = cmd.opt("at", 2) {
            let av = varlist::resolve_name(ds, at.trim())?;
            let c = util::col(ds, &av)?;
            c.into_iter().filter(|v| !is_missing(*v)).collect()
        } else {
            let n = util::opt_num(cmd, "n", 1)?.map(|n| n as usize).unwrap_or_else(|| pts.len().min(50)).max(2);
            let (lo, hi) = (s.min, s.max);
            (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect()
        };
        let dens: Vec<f64> = grid.iter().map(|&g| pts.iter().map(|(xi, wi)| wi * kernel(&kname, (g - xi) / h)).sum::<f64>() / (sw * h)).collect();
        if let Some(gen_opt) = cmd.opt("generate", 3) {
            let names: Vec<&str> = gen_opt.split_whitespace().collect();
            let (xn, dn) = match names.as_slice() {
                [a, b] => (Some(*a), *b),
                [b] => (None, *b),
                _ => return Err(rc_err(198, "generate() requires one or two names")),
            };
            let n = ds.n_obs.max(grid.len());
            while ds.n_obs < n {
                ds.add_observation();
            }
            let pad = |v: &[f64]| -> Vec<f64> { (0..n).map(|i| v.get(i).copied().unwrap_or(ferrostat_core::SYSMISS)).collect() };
            if let Some(xn) = xn {
                ds.add_variable(util::new_double(xn, pad(&grid)))?;
            }
            ds.add_variable(util::new_double(dn, pad(&dens)))?;
        }
        let r = st.begin_r();
        r.set("bwidth", h);
        r.set("n", grid.len() as f64);
        r.set("scale", 1.0);
        r.set_str("kernel", kname.clone());
        if !cmd.has_opt("nograph", 3) {
            let mut g = Graph { title: "Kernel density estimate".into(), xtitle: v.clone(), ytitle: "Density".into(), ..Default::default() };
            g.layers.push(Layer { kind: LayerKind::Line, points: grid.iter().cloned().zip(dens.iter().cloned()).collect(), label: "kdensity".into(), ..Default::default() });
            st.graph = Some(g.clone());
            st.graphs.insert("Graph".into(), g);
        }
        let mut out = vec![
            "Kernel density estimate".to_string(),
            format!("Kernel = {kname}, bandwidth = {:.4}", h),
            format!("Bandwidth: {}", crate::format::fmt_g(h, 12)),
        ];
        if cmd.has_opt("list", 4) || !cmd.has_opt("nograph", 3) {
            out.push(format!("{:>14} {:>14}", "x", "density"));
            for (g, d) in grid.iter().zip(&dens) {
                out.push(format!("{:>14} {:>14}", crate::format::fmt_g(*g, 12), crate::format::fmt_g(*d, 12)));
            }
        }
        Ok(out.join("\n"))
    }
}
