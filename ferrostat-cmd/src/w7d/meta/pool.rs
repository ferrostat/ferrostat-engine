//! Pooling of study effect sizes: inverse-variance and Mantel-Haenszel
//! common/fixed-effects estimates, the between-study variance estimators
//! (REML, ML, DerSimonian-Laird, empirical Bayes, Hedges, Sidik-Jonkman,
//! Paule-Mandel) and the heterogeneity statistics.

use crate::dist;
use crate::eval::rc_err;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Random,
    Common,
    Fixed,
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Model::Random => "random",
            Model::Common => "common",
            Model::Fixed => "fixed",
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Model::Random => "Random-effects model",
            Model::Common => "Common-effect model",
            Model::Fixed => "Fixed-effects model",
        }
    }
    pub fn parse(s: &str) -> Option<Model> {
        match s {
            "random" => Some(Model::Random),
            "common" => Some(Model::Common),
            "fixed" => Some(Model::Fixed),
            _ => None,
        }
    }
}

/// Canonical method name from a (possibly abbreviated) option argument.
pub fn method_name(model: Model, m: &str) -> Result<String, String> {
    let m = m.trim().to_lowercase();
    if model == Model::Random {
        let v = match m.as_str() {
            "" | "reml" => "reml",
            "ml" | "mle" => "ml",
            "dl" | "dlaird" | "dersimonian" => "dl",
            "eb" | "ebayes" | "pm" | "pmandel" => if m.starts_with('p') { "pm" } else { "eb" },
            "he" | "hedges" => "he",
            "sj" | "sjonkman" => "sj",
            "hs" | "hschmidt" => "hs",
            _ => return Err(rc_err(198, format!("random-effects method {m} not allowed"))),
        };
        Ok(v.into())
    } else {
        let v = match m.as_str() {
            "" => "",
            "iv" | "invvariance" => "iv",
            "mh" | "mhaenszel" => "mh",
            _ => return Err(rc_err(198, format!("method {m} not allowed with {} model", model.name()))),
        };
        Ok(v.into())
    }
}

pub fn method_label(m: &str) -> &'static str {
    match m {
        "reml" => "REML",
        "ml" => "ML",
        "dl" => "DerSimonian-Laird",
        "eb" => "Empirical Bayes",
        "pm" => "Paule-Mandel",
        "he" => "Hedges",
        "sj" => "Sidik-Jonkman",
        "hs" => "Hunter-Schmidt",
        "mh" => "Mantel-Haenszel",
        _ => "Inverse-variance",
    }
}

fn wmean(y: &[f64], w: &[f64]) -> f64 {
    y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
}

/// Cochran's Q with inverse-variance weights.
pub fn cochran_q(y: &[f64], v: &[f64]) -> f64 {
    let w: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
    let t = wmean(y, &w);
    y.iter().zip(&w).map(|(a, b)| b * (a - t) * (a - t)).sum()
}

/// Between-study variance by `method`.
pub fn tau2(y: &[f64], v: &[f64], method: &str) -> f64 {
    let k = y.len() as f64;
    if y.len() < 2 {
        return 0.0;
    }
    let w: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
    let sw: f64 = w.iter().sum();
    let sw2: f64 = w.iter().map(|x| x * x).sum();
    let q = cochran_q(y, v);
    let dl = ((q - (k - 1.0)) / (sw - sw2 / sw)).max(0.0);
    let ybar = y.iter().sum::<f64>() / k;
    let ss: f64 = y.iter().map(|a| (a - ybar) * (a - ybar)).sum();
    match method {
        "dl" => dl,
        "he" => (ss / (k - 1.0) - v.iter().sum::<f64>() / k).max(0.0),
        "hs" => {
            // Hunter-Schmidt: (Q - k) / sum w
            ((q - k) / sw).max(0.0)
        }
        "sj" => {
            let t0 = (ss / k).max(1e-8);
            let ww: Vec<f64> = v.iter().map(|x| 1.0 / (x / t0 + 1.0)).collect();
            let th = wmean(y, &ww);
            y.iter().zip(&ww).map(|(a, b)| b * (a - th) * (a - th)).sum::<f64>() / (k - 1.0)
        }
        "pm" => {
            // solve sum w(t)(y - theta(t))^2 = k - 1
            let g = |t: f64| -> f64 {
                let ww: Vec<f64> = v.iter().map(|x| 1.0 / (x + t)).collect();
                let th = wmean(y, &ww);
                y.iter().zip(&ww).map(|(a, b)| b * (a - th) * (a - th)).sum::<f64>() - (k - 1.0)
            };
            if g(0.0) <= 0.0 {
                return 0.0;
            }
            let mut hi = ss.max(1e-6);
            while g(hi) > 0.0 && hi < 1e12 {
                hi *= 2.0;
            }
            let mut lo = 0.0;
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if g(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        }
        "eb" => {
            let mut t = dl;
            for _ in 0..1000 {
                let ww: Vec<f64> = v.iter().map(|x| 1.0 / (x + t)).collect();
                let th = wmean(y, &ww);
                let nt = (ww.iter().zip(y.iter().zip(v)).map(|(wi, (a, vi))| wi * (k / (k - 1.0) * (a - th) * (a - th) - vi)).sum::<f64>() / ww.iter().sum::<f64>()).max(0.0);
                if (nt - t).abs() < 1e-12 * (1.0 + t) {
                    return nt;
                }
                t = nt;
            }
            t
        }
        "ml" | "reml" => {
            let reml = method == "reml";
            // maximize the (restricted) log likelihood over tau2 >= 0 by
            // golden-section search on a bracket, then Fisher-scoring polish
            let ll = |t: f64| -> f64 {
                let ww: Vec<f64> = v.iter().map(|x| 1.0 / (x + t)).collect();
                let th = wmean(y, &ww);
                let mut l = -0.5 * ww.iter().zip(y).map(|(b, a)| -b.ln() + b * (a - th) * (a - th)).sum::<f64>();
                if reml {
                    l -= 0.5 * ww.iter().sum::<f64>().ln();
                }
                l
            };
            let mut hi = (ss / (k - 1.0)).max(dl).max(1e-6) * 4.0 + 1e-6;
            while ll(hi * 2.0) > ll(hi) && hi < 1e12 {
                hi *= 2.0;
            }
            let (mut a, mut b) = (0.0, hi * 2.0);
            let gr = 0.618_033_988_749_895;
            let mut c = b - gr * (b - a);
            let mut d = a + gr * (b - a);
            for _ in 0..300 {
                if ll(c) > ll(d) {
                    b = d;
                } else {
                    a = c;
                }
                c = b - gr * (b - a);
                d = a + gr * (b - a);
                if (b - a).abs() < 1e-14 * (1.0 + a) {
                    break;
                }
            }
            let t = 0.5 * (a + b);
            if ll(0.0) >= ll(t) { 0.0 } else { t }
        }
        _ => dl,
    }
}

#[derive(Clone, Debug, Default)]
pub struct Pool {
    pub theta: f64,
    pub se: f64,
    pub z: f64,
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
    pub tau2: f64,
    pub q: f64,
    pub df: f64,
    pub pq: f64,
    pub i2: f64,
    pub h2: f64,
    pub k: usize,
    /// percent weights
    pub wpct: Vec<f64>,
    pub pi: Option<(f64, f64)>,
}

/// Mantel-Haenszel pooled log odds ratio, log risk ratio or risk
/// difference (with Robins-Breslow-Greenland / Greenland-Robins variances).
/// `cells` are (a, b, c, d): treated events, treated non-events, control
/// events, control non-events.
pub fn mantel_haenszel(cells: &[[f64; 4]], es: &str) -> Result<(f64, f64, Vec<f64>), String> {
    let mut w = Vec::new();
    match es {
        "lnor" => {
            let (mut r, mut s, mut pr, mut ps_qr, mut qs) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &[a, b, c, d] in cells {
                let n = a + b + c + d;
                let (ri, si) = (a * d / n, b * c / n);
                let (pi, qi) = ((a + d) / n, (b + c) / n);
                r += ri;
                s += si;
                pr += pi * ri;
                ps_qr += pi * si + qi * ri;
                qs += qi * si;
                w.push(si);
            }
            if r <= 0.0 || s <= 0.0 {
                return Err(rc_err(459, "Mantel-Haenszel odds ratio not estimable"));
            }
            let var = pr / (2.0 * r * r) + ps_qr / (2.0 * r * s) + qs / (2.0 * s * s);
            Ok(((r / s).ln(), var.sqrt(), w))
        }
        "lnrr" => {
            let (mut r, mut s, mut p) = (0.0, 0.0, 0.0);
            for &[a, b, c, d] in cells {
                let (n1, n2) = (a + b, c + d);
                let n = n1 + n2;
                r += a * n2 / n;
                s += c * n1 / n;
                p += (n1 * n2 * (a + c) - a * c * n) / (n * n);
                w.push(c * n1 / n);
            }
            if r <= 0.0 || s <= 0.0 {
                return Err(rc_err(459, "Mantel-Haenszel risk ratio not estimable"));
            }
            Ok(((r / s).ln(), (p / (r * s)).sqrt(), w))
        }
        "rdiff" => {
            let (mut num, mut den, mut var) = (0.0, 0.0, 0.0);
            for &[a, b, c, d] in cells {
                let (n1, n2) = (a + b, c + d);
                let n = n1 + n2;
                num += (a * n2 - c * n1) / n;
                den += n1 * n2 / n;
                var += (a * b * n2.powi(3) + c * d * n1.powi(3)) / (n1 * n2 * n * n);
                w.push(n1 * n2 / n);
            }
            Ok((num / den, (var / (den * den)).sqrt(), w))
        }
        _ => Err(rc_err(198, "Mantel-Haenszel method requires log odds ratios, log risk ratios or risk differences")),
    }
}

/// Pool effect sizes `y` with variances `v`.
#[allow(clippy::too_many_arguments)]
pub fn pool(y: &[f64], v: &[f64], model: Model, method: &str, level: f64, cells: Option<&[[f64; 4]]>, es: &str, predint: Option<f64>) -> Result<Pool, String> {
    let k = y.len();
    if k == 0 {
        return Err(rc_err(2000, "no studies"));
    }
    if v.iter().any(|x| !(*x > 0.0)) {
        return Err(rc_err(459, "effect-size standard errors must be positive"));
    }
    let q = cochran_q(y, v);
    let df = k as f64 - 1.0;
    let w: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
    let sw: f64 = w.iter().sum();
    let sw2: f64 = w.iter().map(|x| x * x).sum();
    let s2 = if k > 1 { df * sw / (sw * sw - sw2) } else { 0.0 };
    let mut p = Pool { q, df, k, pq: if k > 1 { dist::chi2tail(df, q) } else { 1.0 }, ..Default::default() };
    let (theta, se, wts) = if model == Model::Random {
        let t2 = tau2(y, v, method);
        p.tau2 = t2;
        let ww: Vec<f64> = v.iter().map(|x| 1.0 / (x + t2)).collect();
        let s: f64 = ww.iter().sum();
        p.i2 = if k > 1 { 100.0 * t2 / (t2 + s2) } else { 0.0 };
        p.h2 = if k > 1 { (t2 + s2) / s2 } else { 1.0 };
        (wmean(y, &ww), 1.0 / s.sqrt(), ww)
    } else {
        p.i2 = if k > 1 && q > 0.0 { (100.0 * (q - df) / q).max(0.0) } else { 0.0 };
        p.h2 = if k > 1 { (q / df).max(1.0) } else { 1.0 };
        if method == "mh" {
            let c = cells.ok_or_else(|| rc_err(198, "Mantel-Haenszel method requires binary data declared with meta esize"))?;
            mantel_haenszel(c, es)?
        } else {
            (wmean(y, &w), 1.0 / sw.sqrt(), w.clone())
        }
    };
    let tw: f64 = wts.iter().sum();
    p.wpct = wts.iter().map(|x| 100.0 * x / tw).collect();
    p.theta = theta;
    p.se = se;
    p.z = theta / se;
    p.p = 2.0 * dist::normal(-p.z.abs());
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    p.lo = theta - crit * se;
    p.hi = theta + crit * se;
    if let Some(pl) = predint {
        if model == Model::Random && k >= 3 {
            let t = dist::invttail(k as f64 - 2.0, (1.0 - pl / 100.0) / 2.0);
            let half = t * (p.tau2 + se * se).sqrt();
            p.pi = Some((theta - half, theta + half));
        }
    }
    Ok(p)
}
