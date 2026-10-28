//! Two-period difference-in-differences estimators shared by `drdid` and
//! `csdid`: outcome regression, (stabilised) inverse probability weighting
//! and the doubly robust estimators of Sant'Anna and Zhao (2020), for panel
//! data and for repeated cross sections.
//!
//! Every estimator returns the point estimate and, for each unit (panel) or
//! observation (repeated cross sections), the recentred influence function
//! `att + IF_i`; standard errors are `sqrt(sum (rif - att)^2) / n`.

use crate::matrix::Matrix;
use crate::store::Store;

/// Covariate columns; the constant is the last column.
pub type Cols = Vec<Vec<f64>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    /// improved doubly robust: inverse probability tilting + weighted least squares
    Drimp,
    /// doubly robust: logit propensity score + least squares
    Dripw,
    /// outcome regression
    Reg,
    /// stabilised inverse probability weighting
    Stdipw,
    /// Abadie (2005) inverse probability weighting
    Ipw,
}

impl Method {
    pub fn parse(s: &str) -> Option<Method> {
        match s.trim().to_lowercase().as_str() {
            "drimp" => Some(Method::Drimp),
            "dripw" => Some(Method::Dripw),
            "reg" => Some(Method::Reg),
            "stdipw" => Some(Method::Stdipw),
            "ipw" | "aipw" => Some(Method::Ipw),
            _ => None,
        }
    }
    /// Name stored in e(method).
    pub fn name(self) -> &'static str {
        match self {
            Method::Drimp => "drimp",
            Method::Dripw => "dripw",
            Method::Reg => "reg",
            Method::Stdipw => "stdipw",
            Method::Ipw => "aipw",
        }
    }
    /// (outcome model, treatment model) as shown in the output header.
    pub fn models(self) -> (&'static str, &'static str) {
        match self {
            Method::Drimp => ("weighted least squares", "inverse probability tilting"),
            Method::Dripw => ("least squares", "inverse probability"),
            Method::Reg => ("regression adjustment", "none"),
            Method::Stdipw => ("weighted mean", "stabilized inverse probability"),
            Method::Ipw => ("weighted mean", "inverse probability"),
        }
    }
}

/// Two-period panel: one entry per unit.
pub struct Panel2 {
    pub dy: Vec<f64>,
    pub d: Vec<f64>,
    pub w: Vec<f64>,
    pub x: Cols,
}

/// Two-period repeated cross section: one entry per observation.
pub struct Rc2 {
    pub y: Vec<f64>,
    pub post: Vec<f64>,
    pub d: Vec<f64>,
    pub w: Vec<f64>,
    pub x: Cols,
}

pub struct Fit {
    pub att: f64,
    /// att + influence function, per unit / observation
    pub rif: Vec<f64>,
    /// intermediate coefficient vectors and variances for e()
    pub mats: Vec<(String, Vec<f64>, Option<Matrix>)>,
}

// ---------------------------------------------------------------------------
// small linear-algebra helpers
// ---------------------------------------------------------------------------

pub fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn xb(x: &Cols, b: &[f64], i: usize) -> f64 {
    x.iter().zip(b).map(|(c, bj)| c[i] * bj).sum()
}

fn fitted(x: &Cols, b: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| xb(x, b, i)).collect()
}

/// X' diag(w) X.
fn xtwx(x: &Cols, w: &[f64]) -> Matrix {
    let k = x.len();
    let mut m = Matrix::zeros(k, k);
    for a in 0..k {
        for b in 0..=a {
            let v: f64 = x[a].iter().zip(&x[b]).zip(w).map(|((p, q), wi)| p * q * wi).sum();
            m[(a, b)] = v;
            m[(b, a)] = v;
        }
    }
    m
}

fn inv(m: &Matrix) -> Matrix {
    m.sym_inverse(1e-11).0
}

fn matvec(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum()).collect()
}

/// Weighted least squares (weights zero exclude observations).
/// Returns the coefficients and (X'WX)^-1.
pub fn wls(x: &Cols, y: &[f64], w: &[f64]) -> Result<(Vec<f64>, Matrix), String> {
    if w.iter().all(|v| *v == 0.0) {
        return Err("no observations".into());
    }
    let a = inv(&xtwx(x, w));
    let xty: Vec<f64> = x.iter().map(|c| c.iter().zip(y).zip(w).map(|((p, q), wi)| if *wi == 0.0 { 0.0 } else { p * q * wi }).sum()).collect();
    Ok((matvec(&a, &xty), a))
}

/// Weighted logit by Newton-Raphson. Returns (b, V) with V the inverse of
/// the weighted information matrix.
pub fn logit(x: &Cols, d: &[f64], w: &[f64]) -> Result<(Vec<f64>, Matrix), String> {
    let k = x.len();
    let n = d.len();
    let mut b = vec![0.0; k];
    let ll = |b: &[f64]| -> f64 {
        (0..n)
            .map(|i| {
                let z = xb(x, b, i);
                let p = logistic(z);
                w[i] * if d[i] > 0.5 { p.max(1e-300).ln() } else { (1.0 - p).max(1e-300).ln() }
            })
            .sum()
    };
    let mut cur = ll(&b);
    for _ in 0..100 {
        let p: Vec<f64> = (0..n).map(|i| logistic(xb(x, &b, i))).collect();
        let g: Vec<f64> = x.iter().map(|c| (0..n).map(|i| w[i] * (d[i] - p[i]) * c[i]).sum()).collect();
        let hw: Vec<f64> = (0..n).map(|i| w[i] * p[i] * (1.0 - p[i])).collect();
        let hinv = inv(&xtwx(x, &hw));
        let step = matvec(&hinv, &g);
        let mut t = 1.0;
        let mut nb;
        loop {
            nb = b.iter().zip(&step).map(|(a, s)| a + t * s).collect::<Vec<f64>>();
            let v = ll(&nb);
            if v >= cur - 1e-12 || t < 1e-8 {
                cur = v;
                break;
            }
            t *= 0.5;
        }
        let change: f64 = nb.iter().zip(&b).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        b = nb;
        if change < 1e-10 {
            break;
        }
    }
    let p: Vec<f64> = (0..n).map(|i| logistic(xb(x, &b, i))).collect();
    let hw: Vec<f64> = (0..n).map(|i| w[i] * p[i] * (1.0 - p[i])).collect();
    if p.iter().zip(d).any(|(pi, di)| *di < 0.5 && *pi > 1.0 - 1e-12) {
        return Err("propensity score is one for some untreated observations (no overlap)".into());
    }
    Ok((b, inv(&xtwx(x, &hw))))
}

/// Inverse probability tilting (Graham, Pinto and Egel 2012) as used by the
/// improved doubly robust estimator: calibrates exp(x'g) on the untreated to
/// the treated covariate totals.
pub fn ipt(x: &Cols, d: &[f64], w: &[f64]) -> Result<Vec<f64>, String> {
    let k = x.len();
    let n = d.len();
    let obj = |g: &[f64]| -> f64 {
        -(0..n)
            .map(|i| {
                let z = xb(x, g, i);
                w[i] * if d[i] > 0.5 { z } else { -z.min(700.0).exp() }
            })
            .sum::<f64>()
            / n as f64
    };
    // start from the logit fit (as a good initial value)
    let mut g = match logit(x, d, w) {
        Ok((b, _)) => b,
        Err(_) => vec![0.0; k],
    };
    let mut cur = obj(&g);
    let mut converged = false;
    for _ in 0..200 {
        let e: Vec<f64> = (0..n).map(|i| if d[i] > 0.5 { 0.0 } else { xb(x, &g, i).min(700.0).exp() }).collect();
        let grad: Vec<f64> = x.iter().map(|c| -(0..n).map(|i| w[i] * (if d[i] > 0.5 { 1.0 } else { -e[i] }) * c[i]).sum::<f64>() / n as f64).collect();
        let hw: Vec<f64> = (0..n).map(|i| w[i] * e[i] / n as f64).collect();
        let hinv = inv(&xtwx(x, &hw));
        let step = matvec(&hinv, &grad);
        let mut t = 1.0;
        let mut ng;
        loop {
            ng = g.iter().zip(&step).map(|(a, s)| a - t * s).collect::<Vec<f64>>();
            let v = obj(&ng);
            if v <= cur + 1e-14 || t < 1e-10 {
                cur = v;
                break;
            }
            t *= 0.5;
        }
        let change: f64 = ng.iter().zip(&g).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        g = ng;
        let gnorm: f64 = grad.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if change < 1e-10 || gnorm < 1e-12 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("inverse probability tilting did not converge".into());
    }
    Ok(g)
}

/// s_i * x_i' (A n m): the projection of a linear representation onto m.
fn lin_dot(x: &Cols, s: &[f64], a: &Matrix, n: f64, m: &[f64]) -> Vec<f64> {
    let am: Vec<f64> = matvec(a, m).into_iter().map(|v| v * n).collect();
    (0..s.len()).map(|i| if s[i] == 0.0 { 0.0 } else { s[i] * xb(x, &am, i) }).collect()
}

/// mean over i of f_i x_i.
fn mean_x(x: &Cols, f: &[f64]) -> Vec<f64> {
    let n = f.len() as f64;
    x.iter().map(|c| c.iter().zip(f).map(|(a, b)| a * b).sum::<f64>() / n).collect()
}

fn norm(v: &[f64]) -> Vec<f64> {
    let m = mean(v);
    v.iter().map(|x| x / m).collect()
}

fn check_groups(d: &[f64], w: &[f64]) -> Result<(), String> {
    let nt: f64 = d.iter().zip(w).map(|(a, b)| a * b).sum();
    let nc: f64 = d.iter().zip(w).map(|(a, b)| (1.0 - a) * b).sum();
    if nt <= 0.0 || nc <= 0.0 {
        return Err("treated or comparison group is empty".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// panel estimators
// ---------------------------------------------------------------------------

pub fn panel(m: Method, p: &Panel2) -> Result<Fit, String> {
    let n = p.dy.len();
    let nf = n as f64;
    let (d, w, x, dy) = (&p.d, &p.w, &p.x, &p.dy);
    check_groups(d, w)?;
    let mut mats = Vec::new();
    let wt: Vec<f64> = (0..n).map(|i| w[i] * d[i]).collect();
    let wols: Vec<f64> = (0..n).map(|i| w[i] * (1.0 - d[i])).collect();
    match m {
        Method::Reg => {
            let (b, a) = wls(x, dy, &wols)?;
            mats.push(("regb".into(), b.clone(), None));
            let fx = fitted(x, &b, n);
            let mwt = mean(&wt);
            let eta_t = (0..n).map(|i| wt[i] * dy[i]).sum::<f64>() / nf / mwt;
            let eta_c = (0..n).map(|i| wt[i] * fx[i]).sum::<f64>() / nf / mwt;
            let att = eta_t - eta_c;
            let s: Vec<f64> = (0..n).map(|i| wols[i] * (dy[i] - fx[i])).collect();
            let c2 = lin_dot(x, &s, &a, nf, &mean_x(x, &wt));
            let rif = (0..n)
                .map(|i| {
                    let it = (wt[i] * dy[i] - wt[i] * eta_t) / mwt;
                    let ic = (wt[i] * fx[i] - wt[i] * eta_c + c2[i]) / mwt;
                    att + it - ic
                })
                .collect();
            Ok(Fit { att, rif, mats })
        }
        Method::Dripw | Method::Stdipw | Method::Ipw => {
            let (pb, pv) = logit(x, d, w)?;
            mats.push(("ipwb".into(), pb.clone(), Some(pv.clone())));
            let ps: Vec<f64> = (0..n).map(|i| logistic(xb(x, &pb, i))).collect();
            let wc: Vec<f64> = (0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) / (1.0 - ps[i])).collect();
            let sps: Vec<f64> = (0..n).map(|i| w[i] * (d[i] - ps[i])).collect();
            let (mwt, mwc) = (mean(&wt), mean(&wc));
            match m {
                Method::Dripw => {
                    let (b, a) = wls(x, dy, &wols)?;
                    mats.push(("regb".into(), b.clone(), None));
                    let fx = fitted(x, &b, n);
                    let r: Vec<f64> = (0..n).map(|i| dy[i] - fx[i]).collect();
                    let eta_t = (0..n).map(|i| wt[i] * r[i]).sum::<f64>() / nf / mwt;
                    let eta_c = (0..n).map(|i| wc[i] * r[i]).sum::<f64>() / nf / mwc;
                    let att = eta_t - eta_c;
                    let s: Vec<f64> = (0..n).map(|i| wols[i] * r[i]).collect();
                    let t2 = lin_dot(x, &s, &a, nf, &mean_x(x, &wt));
                    let m2f: Vec<f64> = (0..n).map(|i| wc[i] * (r[i] - eta_c)).collect();
                    let c2 = lin_dot(x, &sps, &pv, nf, &mean_x(x, &m2f));
                    let c3 = lin_dot(x, &s, &a, nf, &mean_x(x, &wc));
                    let rif = (0..n)
                        .map(|i| {
                            let it = (wt[i] * r[i] - wt[i] * eta_t - t2[i]) / mwt;
                            let ic = (wc[i] * r[i] - wc[i] * eta_c + c2[i] - c3[i]) / mwc;
                            att + it - ic
                        })
                        .collect();
                    Ok(Fit { att, rif, mats })
                }
                Method::Stdipw => {
                    let eta_t = (0..n).map(|i| wt[i] * dy[i]).sum::<f64>() / nf / mwt;
                    let eta_c = (0..n).map(|i| wc[i] * dy[i]).sum::<f64>() / nf / mwc;
                    let att = eta_t - eta_c;
                    let m2f: Vec<f64> = (0..n).map(|i| wc[i] * (dy[i] - eta_c)).collect();
                    let c2 = lin_dot(x, &sps, &pv, nf, &mean_x(x, &m2f));
                    let rif = (0..n)
                        .map(|i| {
                            let it = (wt[i] * dy[i] - wt[i] * eta_t) / mwt;
                            let ic = (wc[i] * dy[i] - wc[i] * eta_c + c2[i]) / mwc;
                            att + it - ic
                        })
                        .collect();
                    Ok(Fit { att, rif, mats })
                }
                _ => {
                    // Abadie (2005): both parts normalised by the treated share
                    let eta_t = (0..n).map(|i| wt[i] * dy[i]).sum::<f64>() / nf / mwt;
                    let eta_c = (0..n).map(|i| wc[i] * dy[i]).sum::<f64>() / nf / mwt;
                    let att = eta_t - eta_c;
                    let mf: Vec<f64> = (0..n).map(|i| wc[i] * dy[i]).collect();
                    let l2 = lin_dot(x, &sps, &pv, nf, &mean_x(x, &mf));
                    let rif = (0..n).map(|i| att + (wt[i] * dy[i] - wc[i] * dy[i] - l2[i] - wt[i] * att) / mwt).collect();
                    Ok(Fit { att, rif, mats })
                }
            }
        }
        Method::Drimp => {
            let g = ipt(x, d, w)?;
            mats.push(("iptb".into(), g.clone(), None));
            let ps: Vec<f64> = (0..n).map(|i| logistic(xb(x, &g, i))).collect();
            let wc: Vec<f64> = (0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) / (1.0 - ps[i])).collect();
            let (b, _) = wls(x, dy, &wc)?;
            mats.push(("regb".into(), b.clone(), None));
            let fx = fitted(x, &b, n);
            let (w1, w0) = (norm(&wt), norm(&wc));
            let r: Vec<f64> = (0..n).map(|i| dy[i] - fx[i]).collect();
            let eta_t = (0..n).map(|i| w1[i] * r[i]).sum::<f64>() / nf;
            let eta_c = (0..n).map(|i| w0[i] * r[i]).sum::<f64>() / nf;
            let att = eta_t - eta_c;
            let rif = (0..n).map(|i| att + (w1[i] * r[i] - w1[i] * eta_t) - (w0[i] * r[i] - w0[i] * eta_c)).collect();
            Ok(Fit { att, rif, mats })
        }
    }
}

// ---------------------------------------------------------------------------
// repeated cross sections
// ---------------------------------------------------------------------------

/// Outcome regressions within the four treatment x period cells.
struct CellRegs {
    fits: [Vec<f64>; 4], // 00, 01, 10, 11 (treat, post)
    inv: [Matrix; 4],
    wts: [Vec<f64>; 4],
}

fn cell_regs(r: &Rc2, wctrl: &[f64], need_treated: bool) -> Result<CellRegs, String> {
    let n = r.y.len();
    let mut fits: Vec<Vec<f64>> = Vec::new();
    let mut invs: Vec<Matrix> = Vec::new();
    let mut wts: Vec<Vec<f64>> = Vec::new();
    for (dt, pt) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
        let base = if dt == 0.0 { wctrl } else { &r.w[..] };
        let wcell: Vec<f64> = (0..n).map(|i| if r.d[i] == dt && r.post[i] == pt { base[i] } else { 0.0 }).collect();
        if dt == 1.0 && !need_treated {
            fits.push(vec![0.0; n]);
            invs.push(Matrix::zeros(r.x.len(), r.x.len()));
            wts.push(wcell);
            continue;
        }
        let (b, a) = wls(&r.x, &r.y, &wcell).map_err(|_| "a treatment x period cell is empty".to_string())?;
        fits.push(fitted(&r.x, &b, n));
        invs.push(a);
        wts.push(wcell);
    }
    let f: [Vec<f64>; 4] = fits.try_into().unwrap();
    let i: [Matrix; 4] = invs.try_into().unwrap();
    let w: [Vec<f64>; 4] = wts.try_into().unwrap();
    Ok(CellRegs { fits: f, inv: i, wts: w })
}

pub fn rc(m: Method, rc1: bool, r: &Rc2) -> Result<Fit, String> {
    let n = r.y.len();
    let nf = n as f64;
    let (y, t, d, w, x) = (&r.y, &r.post, &r.d, &r.w, &r.x);
    check_groups(d, w)?;
    for (dt, pt) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
        if !(0..n).any(|i| d[i] == dt && t[i] == pt && w[i] > 0.0) {
            return Err("a treatment x period cell is empty".into());
        }
    }
    let mut mats = Vec::new();
    let w10: Vec<f64> = (0..n).map(|i| w[i] * d[i] * (1.0 - t[i])).collect();
    let w11: Vec<f64> = (0..n).map(|i| w[i] * d[i] * t[i]).collect();
    let means = |a: &[f64]| a.iter().sum::<f64>() / nf;
    match m {
        Method::Reg => {
            let regs = cell_regs(r, w, false)?;
            let (y00, y01) = (&regs.fits[0], &regs.fits[1]);
            let w0: Vec<f64> = norm(&(0..n).map(|i| w[i] * d[i]).collect::<Vec<_>>());
            let (w10n, w11n) = (norm(&w10), norm(&w11));
            let e10 = means(&(0..n).map(|i| w10n[i] * y[i]).collect::<Vec<_>>());
            let e11 = means(&(0..n).map(|i| w11n[i] * y[i]).collect::<Vec<_>>());
            let ac: Vec<f64> = (0..n).map(|i| w0[i] * (y01[i] - y00[i])).collect();
            let ec = means(&ac);
            let att = (e11 - e10) - ec;
            let mx = mean_x(x, &w0);
            let s_pre: Vec<f64> = (0..n).map(|i| regs.wts[0][i] * (y[i] - y00[i])).collect();
            let s_post: Vec<f64> = (0..n).map(|i| regs.wts[1][i] * (y[i] - y01[i])).collect();
            let c_post = lin_dot(x, &s_post, &regs.inv[1], nf, &mx);
            let c_pre = lin_dot(x, &s_pre, &regs.inv[0], nf, &mx);
            let rif = (0..n)
                .map(|i| {
                    let it = (w11n[i] * y[i] - w11n[i] * e11) - (w10n[i] * y[i] - w10n[i] * e10);
                    let ic = ac[i] - w0[i] * ec + c_post[i] - c_pre[i];
                    att + it - ic
                })
                .collect();
            Ok(Fit { att, rif, mats })
        }
        Method::Stdipw | Method::Ipw => {
            let (pb, pv) = logit(x, d, w)?;
            mats.push(("ipwb".into(), pb.clone(), Some(pv.clone())));
            let ps: Vec<f64> = (0..n).map(|i| logistic(xb(x, &pb, i))).collect();
            let sps: Vec<f64> = (0..n).map(|i| w[i] * (d[i] - ps[i])).collect();
            let w00: Vec<f64> = (0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) * (1.0 - t[i]) / (1.0 - ps[i])).collect();
            let w01: Vec<f64> = (0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) * t[i] / (1.0 - ps[i])).collect();
            if m == Method::Stdipw {
                let (w00, w01, w10, w11) = (norm(&w00), norm(&w01), norm(&w10), norm(&w11));
                let e = |wv: &[f64]| means(&(0..n).map(|i| wv[i] * y[i]).collect::<Vec<_>>());
                let (e10, e11, e00, e01) = (e(&w10), e(&w11), e(&w00), e(&w01));
                let att = (e11 - e10) - (e01 - e00);
                let m2pre = mean_x(x, &(0..n).map(|i| w00[i] * (y[i] - e00)).collect::<Vec<_>>());
                let m2post = mean_x(x, &(0..n).map(|i| w01[i] * (y[i] - e01)).collect::<Vec<_>>());
                let mdiff: Vec<f64> = m2post.iter().zip(&m2pre).map(|(a, b)| a - b).collect();
                let cps = lin_dot(x, &sps, &pv, nf, &mdiff);
                let rif = (0..n)
                    .map(|i| {
                        let it = (w11[i] * y[i] - w11[i] * e11) - (w10[i] * y[i] - w10[i] * e10);
                        let ic = (w01[i] * y[i] - w01[i] * e01) - (w00[i] * y[i] - w00[i] * e00) + cps[i];
                        att + it - ic
                    })
                    .collect();
                Ok(Fit { att, rif, mats })
            } else {
                let pi = means(&(0..n).map(|i| w[i] * d[i]).collect::<Vec<_>>());
                let lam = means(&(0..n).map(|i| w[i] * t[i]).collect::<Vec<_>>());
                let olam = means(&(0..n).map(|i| w[i] * (1.0 - t[i])).collect::<Vec<_>>());
                let atp: Vec<f64> = (0..n).map(|i| w10[i] * y[i] / (pi * olam)).collect();
                let ats: Vec<f64> = (0..n).map(|i| w11[i] * y[i] / (pi * lam)).collect();
                let acp: Vec<f64> = (0..n).map(|i| w00[i] * y[i] / (pi * olam)).collect();
                let acs: Vec<f64> = (0..n).map(|i| w01[i] * y[i] / (pi * lam)).collect();
                let (etp, ets, ecp, ecs) = (means(&atp), means(&ats), means(&acp), means(&acs));
                let att = (ets - etp) - (ecs - ecp);
                let mpost = mean_x(x, &acs.iter().map(|v| -v).collect::<Vec<_>>());
                let mpre = mean_x(x, &acp.iter().map(|v| -v).collect::<Vec<_>>());
                let md: Vec<f64> = mpost.iter().zip(&mpre).map(|(a, b)| a - b).collect();
                let il = lin_dot(x, &sps, &pv, nf, &md);
                let rif = (0..n)
                    .map(|i| {
                        let wd = w[i] * d[i] - pi;
                        let wt1 = w[i] * t[i] - lam;
                        let wt0 = w[i] * (1.0 - t[i]) - olam;
                        let its = ats[i] - ets - wd * ets / pi - wt1 * ets / lam;
                        let itp = atp[i] - etp - wd * etp / pi - wt0 * etp / olam;
                        let ics = acs[i] - ecs - wd * ecs / pi - wt1 * ecs / lam;
                        let icp = acp[i] - ecp - wd * ecp / pi - wt0 * ecp / olam;
                        att + (its - itp) - (ics - icp) + il[i]
                    })
                    .collect();
                Ok(Fit { att, rif, mats })
            }
        }
        Method::Dripw | Method::Drimp => {
            let (ps, pv, wctrl) = if m == Method::Dripw {
                let (pb, pv) = logit(x, d, w)?;
                mats.push(("ipwb".into(), pb.clone(), Some(pv.clone())));
                let ps: Vec<f64> = (0..n).map(|i| logistic(xb(x, &pb, i))).collect();
                (ps, Some(pv), w.clone())
            } else {
                let g = ipt(x, d, w)?;
                mats.push(("iptb".into(), g.clone(), None));
                let ps: Vec<f64> = (0..n).map(|i| logistic(xb(x, &g, i))).collect();
                let wc: Vec<f64> = (0..n).map(|i| w[i] * (1.0 - d[i]) * ps[i] / (1.0 - ps[i])).collect();
                (ps, None, wc)
            };
            let regs = cell_regs(r, &wctrl, !rc1)?;
            let (y00, y01, y10, y11) = (&regs.fits[0], &regs.fits[1], &regs.fits[2], &regs.fits[3]);
            let y0: Vec<f64> = (0..n).map(|i| if t[i] > 0.5 { y01[i] } else { y00[i] }).collect();
            let w00: Vec<f64> = norm(&(0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) * (1.0 - t[i]) / (1.0 - ps[i])).collect::<Vec<_>>());
            let w01: Vec<f64> = norm(&(0..n).map(|i| w[i] * ps[i] * (1.0 - d[i]) * t[i] / (1.0 - ps[i])).collect::<Vec<_>>());
            let (w10n, w11n) = (norm(&w10), norm(&w11));
            let w1: Vec<f64> = norm(&(0..n).map(|i| w[i] * d[i]).collect::<Vec<_>>());
            let r0: Vec<f64> = (0..n).map(|i| y[i] - y0[i]).collect();
            let e = |wv: &[f64], v: &[f64]| means(&(0..n).map(|i| wv[i] * v[i]).collect::<Vec<_>>());
            let (e10, e11, e00, e01) = (e(&w10n, &r0), e(&w11n, &r0), e(&w00, &r0), e(&w01, &r0));
            let simple: Vec<f64> = (0..n).map(|i| ((w11n[i] * r0[i] - w11n[i] * e11) - (w10n[i] * r0[i] - w10n[i] * e10)) - ((w01[i] * r0[i] - w01[i] * e01) - (w00[i] * r0[i] - w00[i] * e00))).collect();
            let mut att = (e11 - e10) - (e01 - e00);
            let mut inf = simple;
            if m == Method::Dripw {
                // estimation effects of the propensity score and the control
                // regressions (pre-period terms enter with a minus sign, as
                // in the corrected influence function of the reference R
                // implementation)
                let pv = pv.unwrap();
                let sps: Vec<f64> = (0..n).map(|i| w[i] * (d[i] - ps[i])).collect();
                let s_pre: Vec<f64> = (0..n).map(|i| regs.wts[0][i] * (y[i] - y00[i])).collect();
                let s_post: Vec<f64> = (0..n).map(|i| regs.wts[1][i] * (y[i] - y01[i])).collect();
                let m1post = mean_x(x, &w11n.iter().zip(t).map(|(a, b)| -a * b).collect::<Vec<_>>());
                let m1pre = mean_x(x, &w10n.iter().zip(t).map(|(a, b)| -a * (1.0 - b)).collect::<Vec<_>>());
                let tor_post = lin_dot(x, &s_post, &regs.inv[1], nf, &m1post);
                let tor_pre = lin_dot(x, &s_pre, &regs.inv[0], nf, &m1pre);
                let m2pre = mean_x(x, &(0..n).map(|i| w00[i] * (r0[i] - e00)).collect::<Vec<_>>());
                let m2post = mean_x(x, &(0..n).map(|i| w01[i] * (r0[i] - e01)).collect::<Vec<_>>());
                let md: Vec<f64> = m2post.iter().zip(&m2pre).map(|(a, b)| a - b).collect();
                let cps = lin_dot(x, &sps, &pv, nf, &md);
                let m3post = mean_x(x, &w01.iter().zip(t).map(|(a, b)| -a * b).collect::<Vec<_>>());
                let m3pre = mean_x(x, &w00.iter().zip(t).map(|(a, b)| -a * (1.0 - b)).collect::<Vec<_>>());
                let cor_post = lin_dot(x, &s_post, &regs.inv[1], nf, &m3post);
                let cor_pre = lin_dot(x, &s_pre, &regs.inv[0], nf, &m3pre);
                for i in 0..n {
                    inf[i] += tor_post[i] - tor_pre[i] - (cps[i] + cor_post[i] - cor_pre[i]);
                }
                if !rc1 {
                    let s_pre_t: Vec<f64> = (0..n).map(|i| regs.wts[2][i] * (y[i] - y10[i])).collect();
                    let s_post_t: Vec<f64> = (0..n).map(|i| regs.wts[3][i] * (y[i] - y11[i])).collect();
                    let mom_post = mean_x(x, &(0..n).map(|i| w1[i] - w11n[i]).collect::<Vec<_>>());
                    let mom_pre = mean_x(x, &(0..n).map(|i| w1[i] - w10n[i]).collect::<Vec<_>>());
                    let a = lin_dot(x, &s_post_t, &regs.inv[3], nf, &mom_post);
                    let b = lin_dot(x, &s_post, &regs.inv[1], nf, &mom_post);
                    let c = lin_dot(x, &s_pre_t, &regs.inv[2], nf, &mom_pre);
                    let dd = lin_dot(x, &s_pre, &regs.inv[0], nf, &mom_pre);
                    for i in 0..n {
                        inf[i] += (a[i] - b[i]) - (c[i] - dd[i]);
                    }
                }
            }
            if !rc1 {
                // local efficiency terms
                let dpost: Vec<f64> = (0..n).map(|i| y11[i] - y01[i]).collect();
                let dpre: Vec<f64> = (0..n).map(|i| y10[i] - y00[i]).collect();
                let (etp, et1p, etpre, et0pre) = (e(&w1, &dpost), e(&w11n, &dpost), e(&w1, &dpre), e(&w10n, &dpre));
                att += (etp - et1p) - (etpre - et0pre);
                for i in 0..n {
                    inf[i] += ((w1[i] * dpost[i] - w1[i] * etp) - (w11n[i] * dpost[i] - w11n[i] * et1p)) - ((w1[i] * dpre[i] - w1[i] * etpre) - (w10n[i] * dpre[i] - w10n[i] * et0pre));
                }
            }
            let rif = inf.iter().map(|v| att + v).collect();
            Ok(Fit { att, rif, mats })
        }
    }
}

// ---------------------------------------------------------------------------
// variance and wild bootstrap
// ---------------------------------------------------------------------------

/// Variance of means of influence functions: rif is n x k (column vectors),
/// clusters (codes) optional. Returns k x k.
pub fn if_vce(rif: &[Vec<f64>], b: &[f64], cl: Option<&[usize]>) -> Matrix {
    let k = rif.len();
    let n = rif.first().map_or(0, |c| c.len());
    let nf = n as f64;
    let mut v = Matrix::zeros(k, k);
    match cl {
        None => {
            for a in 0..k {
                for c in 0..=a {
                    let s: f64 = (0..n).map(|i| (rif[a][i] - b[a]) * (rif[c][i] - b[c])).sum::<f64>() / (nf * nf);
                    v[(a, c)] = s;
                    v[(c, a)] = s;
                }
            }
        }
        Some(cl) => {
            let g = cl.iter().copied().max().map_or(0, |m| m + 1);
            let mut sums = vec![vec![0.0; g]; k];
            for a in 0..k {
                for i in 0..n {
                    sums[a][cl[i]] += rif[a][i] - b[a];
                }
            }
            for a in 0..k {
                for c in 0..=a {
                    let s: f64 = (0..g).map(|j| sums[a][j] * sums[c][j]).sum::<f64>() / (nf * nf);
                    v[(a, c)] = s;
                    v[(c, a)] = s;
                }
            }
        }
    }
    v
}

/// Multiplier (wild) bootstrap of means of influence functions.
pub struct Wboot {
    pub reps: usize,
    /// true = Rademacher, false = Mammen
    pub rademacher: bool,
    pub pointwise: bool,
}

/// Returns per-column (se, critical value) from the bootstrap; se from the
/// interquartile range of the bootstrap draws, critical value from the
/// sup-t (uniform) or pointwise distribution.
pub fn wild_bootstrap(rif: &[Vec<f64>], b: &[f64], cl: Option<&[usize]>, opt: &Wboot, level: f64, st: &Store) -> Vec<(f64, f64)> {
    let k = rif.len();
    let n = rif.first().map_or(0, |c| c.len());
    let nf = n as f64;
    let s5 = 5f64.sqrt();
    let k1 = (1.0 + s5) / (2.0 * s5);
    let k2 = 0.5 * (1.0 + s5);
    let ng = cl.map_or(n, |c| c.iter().copied().max().map_or(0, |m| m + 1));
    let mut draws = vec![vec![0.0; opt.reps]; k];
    let mut rng = st.rng.borrow_mut();
    let mut mult = vec![0.0; ng];
    for r in 0..opt.reps {
        for m in mult.iter_mut() {
            let u = rng.uniform();
            *m = if opt.rademacher {
                if u < 0.5 { -1.0 } else { 1.0 }
            } else if u < k1 {
                k2 - s5
            } else {
                k2
            };
        }
        for a in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                let g = cl.map_or(i, |c| c[i]);
                s += (rif[a][i] - b[a]) * mult[g];
            }
            draws[a][r] = s / nf;
        }
    }
    drop(rng);
    let q = |sorted: &[f64], p: f64| -> f64 {
        let idx = (((sorted.len() + 1) as f64) * p).ceil() as usize;
        sorted[idx.clamp(1, sorted.len()) - 1]
    };
    let denom = crate::dist::invnormal(0.75) - crate::dist::invnormal(0.25);
    let ses: Vec<f64> = draws
        .iter()
        .map(|d| {
            let mut s = d.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            (q(&s, 0.75) - q(&s, 0.25)) / denom
        })
        .collect();
    let p = level / 100.0;
    if opt.pointwise {
        draws
            .iter()
            .zip(&ses)
            .map(|(d, se)| {
                let mut t: Vec<f64> = d.iter().map(|v| (v / se).abs()).collect();
                t.sort_by(|a, b| a.total_cmp(b));
                (*se, q(&t, p))
            })
            .collect()
    } else {
        let mut tmax: Vec<f64> = (0..opt.reps).map(|r| (0..k).map(|a| (draws[a][r] / ses[a]).abs()).filter(|v| v.is_finite()).fold(0.0, f64::max)).collect();
        tmax.sort_by(|a, b| a.total_cmp(b));
        let c = q(&tmax, p);
        ses.iter().map(|se| (*se, c)).collect()
    }
}
