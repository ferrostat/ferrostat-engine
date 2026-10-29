//! Remaining time-series helpers: `varbasic` (fit a VAR and graph its
//! impulse responses in one step) and `estat acplot` (model-implied
//! autocorrelations of an ARMA fit with delta-method confidence bands).

use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::graph::{self, Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::{Results, Store};

pub struct Varbasic;

impl CommandHandler for Varbasic {
    fn name(&self) -> &'static str {
        "varbasic"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "var");
        }
        let lags = cmd.opt("lags", 1).map(|s| s.trim().to_string()).unwrap_or_else(|| "1 2".into());
        let step = cmd.opt("step", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| "8".into());
        let stat = if cmd.has_opt("fevd", 4) {
            "fevd"
        } else if cmd.has_opt("irf", 3) {
            "irf"
        } else {
            "oirf"
        };
        let cond = cmd.if_raw.as_ref().map(|c| format!(" if {c}")).unwrap_or_default();
        let mut vcmd = parse_line(&format!("var {}{cond}, lags({lags})", cmd.main.trim())).map_err(|e| rc_err(198, e))?;
        vcmd.in_range = cmd.in_range.clone();
        let mut out = crate::tsvar::Var.execute(ds, st, &vcmd)?;
        let icmd = parse_line(&format!("irf create varbasic, set(_varbasic, replace) step({step})")).map_err(|e| rc_err(198, e))?;
        crate::tsirf::Irf.execute(ds, st, &icmd)?;
        if !cmd.has_opt("nograph", 5) {
            let gcmd = parse_line(&format!("irf graph {stat}, irf(varbasic)")).map_err(|e| rc_err(198, e))?;
            let g = crate::tsirf::Irf.execute(ds, st, &gcmd)?;
            if !g.trim().is_empty() {
                out.push('\n');
                out.push_str(&g);
            }
        }
        st.e.set_str("cmd", "var");
        st.e.set_str("varbasic", "varbasic");
        Ok(out)
    }
}

/// Autocovariances gamma(0..=lags) of an ARMA process with unit innovation
/// variance (AR coefficients `a`, MA coefficients `m`), from psi weights.
pub fn arma_autocov(a: &[f64], m: &[f64], lags: usize) -> Vec<f64> {
    let n = 3000 + lags;
    let mut psi = vec![0.0; n];
    psi[0] = 1.0;
    for j in 1..n {
        let mut v = if j <= m.len() { m[j - 1] } else { 0.0 };
        for (l, al) in a.iter().enumerate() {
            if j > l {
                v += al * psi[j - l - 1];
            }
        }
        psi[j] = if v.is_finite() { v } else { 0.0 };
    }
    (0..=lags).map(|k| (0..n - k).map(|j| psi[j] * psi[j + k]).sum()).collect()
}

fn acplot(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd != "arima" {
        return Err(rc_err(321, "estat acplot requires arima estimates"));
    }
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(b.cols, b.cols));
    let spec = crate::arima::ArmaSpec::from_text(st.e.macros.get("_spec").map(String::as_str).unwrap_or(""));
    let kx = st.e.get("k1").unwrap_or(0.0) as usize;
    let k = b.cols;
    let np = spec.n_coef();
    if np == 0 {
        return Err(rc_err(321, "no ARMA terms in the model"));
    }
    let sigma = st.e.get("sigma").unwrap_or(1.0);
    let lags = crate::util::opt_num(cmd, "lags", 3)?.map(|x| x.max(1.0) as usize).unwrap_or(50);
    let level = crate::util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
    let covariance = cmd.has_opt("covariance", 3);
    let theta: Vec<f64> = b.data[kx..kx + np].to_vec();
    let curve = |th: &[f64], s: f64| -> Vec<f64> {
        let (a, m) = spec.expand(th);
        let g = arma_autocov(&a, &m, lags);
        if covariance {
            g.iter().map(|x| x * s * s).collect()
        } else {
            g.iter().map(|x| x / g[0]).collect()
        }
    };
    let est = curve(&theta, sigma);
    // delta method over the ARMA parameters (and sigma for covariances)
    let mut idx: Vec<usize> = (kx..kx + np).collect();
    if covariance {
        idx.push(k - 1);
    }
    let mut jac = vec![vec![0.0; idx.len()]; lags + 1];
    for (c, &p) in idx.iter().enumerate() {
        let h = 1e-6 * b.data[p].abs().max(1e-3);
        let mut th1 = theta.clone();
        let mut th2 = theta.clone();
        let (mut s1, mut s2) = (sigma, sigma);
        if p == k - 1 && covariance {
            s1 += h;
            s2 -= h;
        } else {
            th1[p - kx] += h;
            th2[p - kx] -= h;
        }
        let f1 = curve(&th1, s1);
        let f2 = curve(&th2, s2);
        for l in 0..=lags {
            jac[l][c] = (f1[l] - f2[l]) / (2.0 * h);
        }
    }
    let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut se = vec![0.0; lags + 1];
    for l in 0..=lags {
        let mut s = 0.0;
        for (a, &pa) in idx.iter().enumerate() {
            for (c, &pc) in idx.iter().enumerate() {
                s += jac[l][a] * v.get(pa, pc) * jac[l][c];
            }
        }
        se[l] = s.max(0.0).sqrt();
    }
    let what = if covariance { "Autocovariance" } else { "Autocorrelation" };
    let mut r = Results::new();
    let mut m = Matrix::zeros(lags + 1, 4);
    for l in 0..=lags {
        m.data[l * 4] = l as f64;
        m.data[l * 4 + 1] = est[l];
        m.data[l * 4 + 2] = est[l] - z * se[l];
        m.data[l * 4 + 3] = est[l] + z * se[l];
    }
    let m = m.with_names((0..=lags).map(|l| l.to_string()).collect(), vec!["lag".into(), "ac".into(), "lb".into(), "ub".into()]);
    r.set_mat("ac", m);
    r.set("lags", lags as f64);
    st.r = r;
    let band = Layer {
        kind: LayerKind::Range(RangeStyle::Area),
        ranges: (0..=lags).map(|l| (l as f64, est[l] - z * se[l], est[l] + z * se[l])).collect(),
        label: format!("{}% CI", level),
        alpha: 0.3,
        ..Default::default()
    };
    let line = Layer { kind: LayerKind::Connected, points: (0..=lags).map(|l| (l as f64, est[l])).collect(), label: what.into(), ..Default::default() };
    let mut g = Graph { layers: vec![band, line], xtitle: "Lag".into(), ytitle: format!("Parametric {}", what.to_lowercase()), legend: true, ..Default::default() };
    graph::finish(&mut g, cmd, st)
}

/// `estat` subcommands provided here (None if not ours).
pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "acplot" | "acpl" => Some(acplot(st, cmd)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_autocorrelations_are_powers() {
        let g = arma_autocov(&[0.6], &[], 5);
        for k in 0..=5 {
            assert!((g[k] / g[0] - 0.6f64.powi(k as i32)).abs() < 1e-10);
        }
        assert!((g[0] - 1.0 / (1.0 - 0.36)).abs() < 1e-9);
        // MA(1): rho1 = m/(1+m^2), zero beyond
        let g = arma_autocov(&[], &[0.5], 3);
        assert!((g[1] / g[0] - 0.4).abs() < 1e-12 && g[2].abs() < 1e-12);
    }
}
