//! Small statistics commands: `sktest`, `ladder`, `esize`, `kap`,
//! `kappa`, `icc`, `loneway`, `pcorr`, `cumul`, `bitest`, `bitesti`,
//! `mkspline`, `orthog`, `orthpoly`, `nptrend` and `lv`.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;
use crate::varlist;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("sktest", Box::new(Sktest)),
        ("ladder", Box::new(Ladder)),
        ("esize", Box::new(Esize)),
        ("esizei", Box::new(Esize)),
        ("kap", Box::new(Kap)),
        ("kappa", Box::new(Kappa)),
        ("icc", Box::new(Icc)),
        ("loneway", Box::new(Loneway)),
        ("pcorr", Box::new(Pcorr)),
        ("cumul", Box::new(Cumul)),
        ("bitest", Box::new(Bitest)),
        ("bitesti", Box::new(Bitest)),
        ("mkspline", Box::new(Mkspline)),
        ("orthog", Box::new(Orthog)),
        ("orthpoly", Box::new(Orthpoly)),
        ("nptrend", Box::new(Nptrend)),
        ("lv", Box::new(Lv)),
    ]
}

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

/// Nonmissing values of `var` in the if/in sample.
fn values_of(ds: &DataSet, st: &Store, cmd: &Command, var: &str) -> Result<Vec<f64>, String> {
    let mask = util::touse(ds, st, cmd)?;
    let c = util::col(ds, var)?;
    Ok((0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect())
}

fn add_var(ds: &mut DataSet, name: &str, ty: VarType, vals: &[f64], label: &str) -> Result<(), String> {
    varlist::check_name(name)?;
    if ds.var_index.contains_key(name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let mut v = Variable::new(name.to_string(), ty);
    v.label = label.to_string();
    v.data = vals.iter().map(|&x| util::typed_value(ty, x)).collect();
    ds.add_variable(v)
}

// ---------------------------------------------------------------------------
// sktest and ladder
// ---------------------------------------------------------------------------

/// D'Agostino–Pearson tests with Royston's adjustment of the joint test:
/// (Pr(skewness), Pr(kurtosis), adjusted chi2(2), Prob > chi2).
pub fn sktest_values(x: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    let s = m3 / m2.powf(1.5);
    let k = m4 / (m2 * m2);
    // skewness (D'Agostino 1970)
    let y = s * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let b2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (b2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let ya = y / alpha;
    let z1 = delta * (ya + (ya * ya + 1.0).sqrt()).ln();
    // kurtosis (Anscombe & Glynn 1983)
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let v = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
    let xk = (k - e) / v.sqrt();
    let sb1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a = 6.0 + 8.0 / sb1 * (2.0 / sb1 + (1.0 + 4.0 / (sb1 * sb1)).sqrt());
    let t = (1.0 - 2.0 / a) / (1.0 + xk * (2.0 / (a - 4.0)).sqrt());
    let z2 = ((1.0 - 2.0 / (9.0 * a)) - t.signum() * t.abs().powf(1.0 / 3.0)) / (2.0 / (9.0 * a)).sqrt();
    let ps = 2.0 * dist::normal(-z1.abs());
    let pk = 2.0 * dist::normal(-z2.abs());
    // Royston (1991) adjustment of K² = z1² + z2²
    let k2 = z1 * z1 + z2 * z2;
    let zc2 = -dist::invnormal((-0.5 * k2).exp());
    let logn = n.ln();
    let cut = 0.55 * n.powf(0.2) - 0.21;
    let a1 = (-5.0 + 3.46 * logn) * (-1.37 * logn).exp();
    let b1 = 1.0 + (0.854 - 0.148 * logn) * (-0.55 * logn).exp();
    let b2r = 2.13 / (1.0 - 2.37 * logn);
    let a2 = a1 - b2r * cut.powf(b1);
    let zz = if zc2 < cut { a1 + b1 * zc2 } else { a2 + (b1 + b2r) * zc2 };
    let p = 1.0 - dist::normal(zz);
    let chi2 = if p > 0.0 { -2.0 * p.ln() } else { SYSMISS };
    (ps, pk, chi2, p)
}

pub struct Sktest;

impl CommandHandler for Sktest {
    fn name(&self) -> &'static str {
        "sktest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let noadjust = cmd.has_opt("noadjust", 5);
        let mut out = vec![String::new(), "Skewness and kurtosis tests for normality".to_string()];
        out.push(format!("{:>66}", "----- Joint test -----"));
        out.push(format!("    Variable |       Obs   Pr(skewness)   Pr(kurtosis)  {:>9}  Prob>chi2", if noadjust { "chi2(2)" } else { "Adj chi2(2)" }));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(62)));
        let mut r = Results::new();
        for v in &vars {
            let x = values_of(ds, st, cmd, v)?;
            if x.len() < 8 {
                return Err(rc_err(2001, "insufficient observations"));
            }
            let (ps, pk, chi2, p) = sktest_values(&x);
            let (chi2, p) = if noadjust {
                let z1 = dist::invnormal(1.0 - ps / 2.0);
                let z2 = dist::invnormal(1.0 - pk / 2.0);
                let c = z1 * z1 + z2 * z2;
                (c, dist::chi2tail(2.0, c))
            } else {
                (chi2, p)
            };
            out.push(format!("{:>12} | {:>9}    {:>10.4}     {:>10.4}  {:>11}  {:>9.4}", abbrev(v, 12), util::commas(x.len()), ps, pk, if is_missing(chi2) { ".".into() } else { format!("{:.2}", chi2) }, p));
            r.set("N", x.len() as f64);
            r.set("p_skew", ps);
            r.set("p_kurt", pk);
            r.set("chi2", chi2);
            r.set("p_chi2", p);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Ladder;

const LADDER: [(&str, &str, f64); 9] = [
    ("Cubic", "{}^3", 3.0),
    ("Square", "{}^2", 2.0),
    ("Identity", "{}", 1.0),
    ("Square root", "sqrt({})", 0.5),
    ("Log", "log({})", 0.0),
    ("1/(Square root)", "1/sqrt({})", -0.5),
    ("Inverse", "1/{}", -1.0),
    ("1/Square", "1/({}^2)", -2.0),
    ("1/Cubic", "1/({}^3)", -3.0),
];

fn power_tr(x: f64, p: f64) -> f64 {
    if p == 0.0 {
        x.ln()
    } else {
        x.powf(p)
    }
}

impl CommandHandler for Ladder {
    fn name(&self) -> &'static str {
        "ladder"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 1 {
            return Err(rc_err(103, "too many variables specified"));
        }
        let v = &vars[0];
        let x = values_of(ds, st, cmd, v)?;
        let mut out = vec![String::new(), format!("{:<22} {:<21} {:>9}   {:>11}", "Transformation", "Formula", "chi2(2)", "Prob > chi2"), "-".repeat(70)];
        let mut r = Results::new();
        let keys = ["cube", "square", "raw", "sqrt", "log", "invsqrt", "inv", "invsq", "invcube"];
        for (i, (name, f, p)) in LADDER.iter().enumerate() {
            let formula = f.replace("{}", v);
            let positive_needed = *p <= 0.5 && *p != 1.0;
            let ok = !positive_needed || x.iter().all(|&v| v > 0.0);
            let (chi2, pv) = if ok {
                let t: Vec<f64> = x.iter().map(|&v| power_tr(v, *p)).collect();
                let (_, _, c, pp) = sktest_values(&t);
                (c, pp)
            } else {
                (SYSMISS, SYSMISS)
            };
            let fmtc = |c: f64| if is_missing(c) { ".".to_string() } else { format!("{c:.2}") };
            out.push(format!("{:<22} {:<21} {:>9}   {:>11}", name, abbrev(&formula, 21), fmtc(chi2), if is_missing(pv) { ".".to_string() } else { format!("{pv:.3}") }));
            r.set(&format!("{}", keys[i]), chi2);
            r.set(&format!("P_{}", keys[i]), pv);
        }
        r.set("N", x.len() as f64);
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// esize
// ---------------------------------------------------------------------------

/// Cumulative distribution of the noncentral t (Lenth 1989, AS 243).
pub fn nct_cdf(t: f64, df: f64, delta: f64) -> f64 {
    let (tt, del, neg) = if t < 0.0 { (-t, -delta, true) } else { (t, delta, false) };
    let x = tt * tt / (tt * tt + df);
    let mut tnc = 0.0;
    if x > 0.0 {
        let lambda = del * del;
        let mut p = 0.5 * (-0.5 * lambda).exp();
        let mut q = (2.0 / std::f64::consts::PI).sqrt() * p * del;
        let mut s = 0.5 - p;
        let mut a = 0.5;
        let b = 0.5 * df;
        let rxb = (1.0 - x).powf(b);
        let albeta = std::f64::consts::PI.sqrt().ln() + dist::lngamma(b) - dist::lngamma(0.5 + b);
        let mut xodd = dist::ibeta(a, b, x);
        let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
        let mut xeven = 1.0 - rxb;
        let mut geven = b * x * rxb;
        tnc = p * xodd + q * xeven;
        let mut en = 1.0;
        loop {
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * en);
            q *= lambda / (2.0 * en + 1.0);
            s -= p;
            en += 1.0;
            tnc += p * xodd + q * xeven;
            let errbd = 2.0 * s * (xodd - godd);
            if errbd.abs() <= 1e-12 || en > 2000.0 {
                break;
            }
        }
    }
    tnc += dist::normal(-del);
    let r = if neg { 1.0 - tnc } else { tnc };
    r.clamp(0.0, 1.0)
}

/// Noncentrality parameter with P(T <= t | ncp) = p.
fn nct_ncp(t: f64, df: f64, p: f64) -> f64 {
    let (mut lo, mut hi) = (t - 10.0 - t.abs(), t + 10.0 + t.abs());
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        // the cdf decreases in ncp
        if nct_cdf(t, df, mid) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub struct Esize;

impl CommandHandler for Esize {
    fn name(&self) -> &'static str {
        "esize"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = util::level(cmd, st)?;
        let (n1, m1, s1, n2, m2, s2, l1, l2) = if cmd.verb == "esizei" {
            let v: Vec<f64> = cmd.main.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "esizei #obs1 #mean1 #sd1 #obs2 #mean2 #sd2"))).collect::<Result<_, _>>()?;
            if v.len() != 6 {
                return Err(rc_err(198, "esizei #obs1 #mean1 #sd1 #obs2 #mean2 #sd2"));
            }
            (v[0], v[1], v[2], v[3], v[4], v[5], "Group 1".to_string(), "Group 2".to_string())
        } else {
            let main = cmd.main.trim();
            let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
            let mask = util::touse(ds, st, cmd)?;
            let stats = |x: &[f64]| (x.len() as f64, util::mean(x), util::variance(x).sqrt());
            match sub {
                "twosample" | "two" | "twos" => {
                    let var = varlist::expand(ds, rest)?;
                    let by = cmd.opt("by", 2).ok_or_else(|| rc_err(198, "option by() required"))?.trim().to_string();
                    let y = util::col(ds, &var[0])?;
                    let g = util::col(ds, &by)?;
                    let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(g[i]) && !is_missing(y[i])).map(|i| g[i]).collect();
                    lv.sort_by(f64::total_cmp);
                    lv.dedup();
                    if lv.len() != 2 {
                        return Err(rc_err(420, "more than 2 groups found, only 2 allowed"));
                    }
                    let x1: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && g[i] == lv[0] && !is_missing(y[i])).map(|i| y[i]).collect();
                    let x2: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && g[i] == lv[1] && !is_missing(y[i])).map(|i| y[i]).collect();
                    let (a, b, c) = stats(&x1);
                    let (d, e, f) = stats(&x2);
                    let lab = |l: f64| ds.get_var(&by).and_then(|v| ds.value_labels.get(&v.value_label_name)).and_then(|vl| vl.lookup(l).map(String::from)).unwrap_or_else(|| fmt_g(l, 12));
                    (a, b, c, d, e, f, lab(lv[0]), lab(lv[1]))
                }
                "unpaired" | "unp" => {
                    let parts: Vec<&str> = rest.split("==").collect();
                    if parts.len() != 2 {
                        return Err(rc_err(198, "esize unpaired varname1 == varname2"));
                    }
                    let v1 = varlist::resolve_name(ds, parts[0].trim())?;
                    let v2 = varlist::resolve_name(ds, parts[1].trim().split_whitespace().next().unwrap_or(""))?;
                    let c1 = util::col(ds, &v1)?;
                    let c2 = util::col(ds, &v2)?;
                    let x1: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c1[i])).map(|i| c1[i]).collect();
                    let x2: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c2[i])).map(|i| c2[i]).collect();
                    let (a, b, c) = stats(&x1);
                    let (d, e, f) = stats(&x2);
                    (a, b, c, d, e, f, v1, v2)
                }
                _ => return Err(rc_err(198, "esize twosample varname, by(groupvar) | esize unpaired var1 == var2")),
            }
        };
        let unequal = cmd.has_opt("unequal", 3);
        let df = if unequal {
            let (a, b) = (s1 * s1 / n1, s2 * s2 / n2);
            (a + b).powi(2) / (a * a / (n1 - 1.0) + b * b / (n2 - 1.0))
        } else {
            n1 + n2 - 2.0
        };
        let sp = if unequal { ((s1 * s1 + s2 * s2) / 2.0).sqrt() } else { (((n1 - 1.0) * s1 * s1 + (n2 - 1.0) * s2 * s2) / (n1 + n2 - 2.0)).sqrt() };
        let d = (m1 - m2) / sp;
        let scale = (1.0 / n1 + 1.0 / n2).sqrt();
        let t = d / scale;
        let alpha = 1.0 - level / 100.0;
        let lo_ncp = nct_ncp(t, df, 1.0 - alpha / 2.0);
        let hi_ncp = nct_ncp(t, df, alpha / 2.0);
        let (dl, du) = (lo_ncp * scale, hi_ncp * scale);
        let j = (dist::lngamma(df / 2.0) - (df / 2.0).ln() * 0.5 - dist::lngamma((df - 1.0) / 2.0)).exp();
        let (g, gl, gu) = (d * j, dl * j, du * j);
        // Glass's delta: standardised by the first group's sd
        let glass = (m1 - m2) / s1;
        let gt = glass / (1.0 / n1 + s2 * s2 / (s1 * s1 * n2)).sqrt();
        let gscale = glass / gt;
        let gdf = n1 - 1.0;
        let (gll, gul) = (nct_ncp(gt, gdf, 1.0 - alpha / 2.0) * gscale, nct_ncp(gt, gdf, alpha / 2.0) * gscale);
        let all = cmd.has_opt("all", 3);
        let want_d = all || cmd.has_opt("cohensd", 6) || !(cmd.has_opt("hedgesg", 6) || cmd.has_opt("glassdelta", 5));
        let want_g = all || cmd.has_opt("hedgesg", 6) || !(cmd.has_opt("cohensd", 6) || cmd.has_opt("glassdelta", 5));
        let want_gl = all || cmd.has_opt("glassdelta", 5);
        let mut out = vec![String::new(), "Effect size based on mean comparison".to_string(), String::new()];
        let lw = l1.chars().count().max(l2.chars().count()).max(10);
        out.push(format!("{:>44}", "Obs per group:"));
        out.push(format!("{:>w$} = {:>10}", abbrev(&l1, 16), util::commas(n1 as usize), w = 49usize.saturating_sub(13).max(lw)));
        out.push(format!("{:>w$} = {:>10}", abbrev(&l2, 16), util::commas(n2 as usize), w = 49usize.saturating_sub(13).max(lw)));
        out.push("-".repeat(57));
        out.push(format!("{:>19} |   Estimate     [{}% conf. interval]", "Effect size", fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(36)));
        let row = |l: &str, e: f64, a: f64, b: f64| format!("{:>19} |  {:>9}   {:>9}   {:>9}", l, g9(e), g9(a), g9(b));
        if want_d {
            out.push(row("Cohen's d", d, dl, du));
        }
        if want_g {
            out.push(row("Hedges's g", g, gl, gu));
        }
        if want_gl {
            out.push(row("Glass's Delta 1", glass, gll, gul));
        }
        out.push("-".repeat(57));
        let mut r = Results::new();
        r.set("N_1", n1);
        r.set("N_2", n2);
        r.set("level", level);
        r.set("d", d);
        r.set("lb_d", dl);
        r.set("ub_d", du);
        r.set("g", g);
        r.set("lb_g", gl);
        r.set("ub_g", gu);
        if want_gl {
            r.set("delta1", glass);
            r.set("lb_delta1", gll);
            r.set("ub_delta1", gul);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// kap, kappa
// ---------------------------------------------------------------------------

pub struct Kap;

impl CommandHandler for Kap {
    fn name(&self) -> &'static str {
        "kap"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "kap varname1 varname2 (two raters)"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "f")?;
        util::markout(ds, &mut mask, &vars);
        let a = util::col(ds, &vars[0])?;
        let b = util::col(ds, &vars[1])?;
        let rows = util::rows(&mask);
        let mut cats: Vec<f64> = rows.iter().flat_map(|&i| [a[i], b[i]]).collect();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let k = cats.len();
        let idx = |v: f64| cats.iter().position(|&c| c == v).unwrap();
        let mut p = Matrix::zeros(k, k);
        let mut n = 0.0;
        for &i in &rows {
            let wt = w.as_ref().map_or(1.0, |w| w.w[i]);
            p[(idx(a[i]), idx(b[i]))] += wt;
            n += wt;
        }
        for x in p.data.iter_mut() {
            *x /= n;
        }
        let wopt = cmd.opt("wgt", 1).map(|s| s.trim().to_lowercase());
        let wm = |i: usize, j: usize| -> f64 {
            let d = (i as f64 - j as f64).abs() / (k as f64 - 1.0).max(1.0);
            match wopt.as_deref() {
                Some("w") => 1.0 - d,
                Some("w2") => 1.0 - d * d,
                _ => if i == j { 1.0 } else { 0.0 },
            }
        };
        let pr: Vec<f64> = (0..k).map(|i| (0..k).map(|j| p[(i, j)]).sum()).collect();
        let pc: Vec<f64> = (0..k).map(|j| (0..k).map(|i| p[(i, j)]).sum()).collect();
        let po: f64 = (0..k).flat_map(|i| (0..k).map(move |j| (i, j))).map(|(i, j)| wm(i, j) * p[(i, j)]).sum();
        let pe: f64 = (0..k).flat_map(|i| (0..k).map(move |j| (i, j))).map(|(i, j)| wm(i, j) * pr[i] * pc[j]).sum();
        let kappa = (po - pe) / (1.0 - pe);
        let wbar_r: Vec<f64> = (0..k).map(|i| (0..k).map(|j| pc[j] * wm(i, j)).sum()).collect();
        let wbar_c: Vec<f64> = (0..k).map(|j| (0..k).map(|i| pr[i] * wm(i, j)).sum()).collect();
        let s: f64 = (0..k).flat_map(|i| (0..k).map(move |j| (i, j))).map(|(i, j)| pr[i] * pc[j] * (wm(i, j) - (wbar_r[i] + wbar_c[j])).powi(2)).sum();
        let se = (s - pe * pe).max(0.0).sqrt() / ((1.0 - pe) * n.sqrt());
        let z = kappa / se;
        let pz = 1.0 - dist::normal(z);
        let mut out = vec![String::new()];
        if let Some(wt) = &wopt {
            out.push(format!("Ratings weighted by:"));
            let wl: Vec<String> = (0..k).map(|i| (0..k).map(|j| format!("{:>7.4}", wm(i, j))).collect::<Vec<_>>().join("")).collect();
            out.extend(wl);
            let _ = wt;
            out.push(String::new());
        }
        out.push("             Expected".into());
        out.push("Agreement   agreement     Kappa   Std. err.         Z      Prob>Z".into());
        out.push("-".repeat(65));
        out.push(format!("{:>7.2}%    {:>7.2}%    {:>7.4}    {:>7.4}    {:>7.2}    {:>8.4}", 100.0 * po, 100.0 * pe, kappa, se, z, pz));
        let mut r = Results::new();
        r.set("N", n);
        r.set("prop_o", po);
        r.set("prop_e", pe);
        r.set("kappa", kappa);
        r.set("se", se);
        r.set("z", z);
        st.r = r;
        Ok(out.join("\n"))
    }
}

/// `kappa freqvars`: Fleiss' kappa for any number of raters; each variable
/// counts the raters assigning the subject to that category.
pub struct Kappa;

impl CommandHandler for Kappa {
    fn name(&self) -> &'static str {
        "kappa"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let rows = util::rows(&mask);
        let n = rows.len() as f64;
        let m = rows.first().map_or(0.0, |&i| cols.iter().map(|c| c[i]).sum::<f64>());
        if rows.iter().any(|&i| (cols.iter().map(|c| c[i]).sum::<f64>() - m).abs() > 1e-9) {
            return Err(rc_err(459, "kappa requires the same number of raters for every subject"));
        }
        let k = vars.len();
        let pj: Vec<f64> = (0..k).map(|j| rows.iter().map(|&i| cols[j][i]).sum::<f64>() / (n * m)).collect();
        let mut out = vec![String::new()];
        let mut r = Results::new();
        let kj: Vec<f64> = (0..k)
            .map(|j| {
                let num: f64 = rows.iter().map(|&i| cols[j][i] * (m - cols[j][i])).sum();
                1.0 - num / (n * m * (m - 1.0) * pj[j] * (1.0 - pj[j]))
            })
            .collect();
        let sum_pq: f64 = pj.iter().map(|p| p * (1.0 - p)).sum();
        let num: f64 = rows.iter().map(|&i| (0..k).map(|j| cols[j][i] * (m - cols[j][i])).sum::<f64>()).sum();
        let kappa = 1.0 - num / (n * m * (m - 1.0) * sum_pq);
        // standard errors under the null (Fleiss, Nee and Landis 1979)
        let se_j = (2.0 / (n * m * (m - 1.0))).sqrt();
        let sum_pqq: f64 = pj.iter().map(|p| p * (1.0 - p) * (1.0 - 2.0 * p)).sum();
        let se = (2.0f64).sqrt() / (sum_pq * (n * m * (m - 1.0)).sqrt()) * (sum_pq * sum_pq - sum_pqq).max(0.0).sqrt();
        if k > 2 {
            out.push(format!("{:>12}     Kappa         Z     Prob>Z", "Outcome"));
            out.push("-".repeat(46));
            for j in 0..k {
                let z = kj[j] / se_j;
                out.push(format!("{:>12}    {:>6.4}    {:>6.2}     {:>6.4}", abbrev(&vars[j], 12), kj[j], z, 1.0 - dist::normal(z)));
            }
            out.push("-".repeat(46));
            let z = kappa / se;
            out.push(format!("{:>12}    {:>6.4}    {:>6.2}     {:>6.4}", "combined", kappa, z, 1.0 - dist::normal(z)));
        } else {
            out.push("Two-outcomes, multiple raters:".into());
            out.push(String::new());
            out.push("         Kappa        Z        Prob>Z".into());
            out.push("-".repeat(37));
            let z = kappa / se_j;
            out.push(format!("        {:>6.4}    {:>6.2}      {:>6.4}", kappa, z, 1.0 - dist::normal(z)));
        }
        r.set("N", n);
        r.set("kappa", kappa);
        r.set("z", kappa / if k > 2 { se } else { se_j });
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// icc, loneway
// ---------------------------------------------------------------------------

pub struct Icc;

impl CommandHandler for Icc {
    fn name(&self) -> &'static str {
        "icc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 || vars.len() > 3 {
            return Err(rc_err(198, "icc depvar target [rater]"));
        }
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let y = util::col(ds, &vars[0])?;
        let t = util::col(ds, &vars[1])?;
        let rows = util::rows(&mask);
        let tl: Vec<f64> = {
            let mut v: Vec<f64> = rows.iter().map(|&i| t[i]).collect();
            v.sort_by(f64::total_cmp);
            v.dedup();
            v
        };
        let n = tl.len() as f64;
        let alpha = 1.0 - level / 100.0;
        let mut out = vec![String::new(), "Intraclass correlations".to_string()];
        let mut r = Results::new();
        let (icc1, ci1, icck, cik, f, df1, df2, pf);
        let k;
        if vars.len() == 2 {
            // one-way random effects
            let mut groups: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
            for &i in &rows {
                groups.entry(t[i].to_bits()).or_default().push(y[i]);
            }
            let kk = groups.values().next().map_or(0, |g| g.len());
            if groups.values().any(|g| g.len() != kk) {
                return Err(rc_err(459, "unbalanced data: each target must have the same number of ratings"));
            }
            k = kk as f64;
            let gm = rows.iter().map(|&i| y[i]).sum::<f64>() / (n * k);
            let msr = groups.values().map(|g| k * (util::mean(g) - gm).powi(2)).sum::<f64>() / (n - 1.0);
            let msw = groups.values().map(|g| {
                let m = util::mean(g);
                g.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            }).sum::<f64>() / (n * (k - 1.0));
            icc1 = (msr - msw) / (msr + (k - 1.0) * msw);
            icck = (msr - msw) / msr;
            f = msr / msw;
            df1 = n - 1.0;
            df2 = n * (k - 1.0);
            let fl = f / dist::invftail(df1, df2, alpha / 2.0);
            let fu = f * dist::invftail(df2, df1, alpha / 2.0);
            ci1 = ((fl - 1.0) / (fl + k - 1.0), (fu - 1.0) / (fu + k - 1.0));
            cik = (1.0 - 1.0 / fl, 1.0 - 1.0 / fu);
            pf = dist::ftail(df1, df2, f);
            out.push("One-way random-effects model".into());
            out.push("Absolute agreement".into());
        } else {
            let rv = util::col(ds, &vars[2])?;
            let rl: Vec<f64> = {
                let mut v: Vec<f64> = rows.iter().map(|&i| rv[i]).collect();
                v.sort_by(f64::total_cmp);
                v.dedup();
                v
            };
            k = rl.len() as f64;
            if (rows.len() as f64 - n * k).abs() > 0.5 {
                return Err(rc_err(459, "unbalanced data: each target must be rated by every rater once"));
            }
            let gm = rows.iter().map(|&i| y[i]).sum::<f64>() / (n * k);
            let tm: BTreeMap<u64, f64> = tl.iter().map(|&l| (l.to_bits(), rows.iter().filter(|&&i| t[i] == l).map(|&i| y[i]).sum::<f64>() / k)).collect();
            let rm: BTreeMap<u64, f64> = rl.iter().map(|&l| (l.to_bits(), rows.iter().filter(|&&i| rv[i] == l).map(|&i| y[i]).sum::<f64>() / n)).collect();
            let msr = k * tm.values().map(|m| (m - gm).powi(2)).sum::<f64>() / (n - 1.0);
            let msc = n * rm.values().map(|m| (m - gm).powi(2)).sum::<f64>() / (k - 1.0);
            let sse: f64 = rows.iter().map(|&i| (y[i] - tm[&t[i].to_bits()] - rm[&rv[i].to_bits()] + gm).powi(2)).sum();
            let mse = sse / ((n - 1.0) * (k - 1.0));
            let consistency = cmd.has_opt("consistency", 4);
            let mixed = cmd.has_opt("mixed", 3);
            f = msr / mse;
            df1 = n - 1.0;
            df2 = (n - 1.0) * (k - 1.0);
            pf = dist::ftail(df1, df2, f);
            out.push(if mixed { "Two-way mixed-effects model".into() } else { "Two-way random-effects model".into() });
            if consistency {
                out.push("Consistency of agreement".into());
                icc1 = (msr - mse) / (msr + (k - 1.0) * mse);
                icck = (msr - mse) / msr;
                let fl = f / dist::invftail(df1, df2, alpha / 2.0);
                let fu = f * dist::invftail(df2, df1, alpha / 2.0);
                ci1 = ((fl - 1.0) / (fl + k - 1.0), (fu - 1.0) / (fu + k - 1.0));
                cik = (1.0 - 1.0 / fl, 1.0 - 1.0 / fu);
            } else {
                out.push("Absolute agreement".into());
                icc1 = (msr - mse) / (msr + (k - 1.0) * mse + k / n * (msc - mse));
                icck = (msr - mse) / (msr + (msc - mse) / n);
                // Satterthwaite df (McGraw and Wong 1996)
                let a = k * icc1 / (n * (1.0 - icc1));
                let b = 1.0 + k * icc1 * (n - 1.0) / (n * (1.0 - icc1));
                let v = (a * msc + b * mse).powi(2) / ((a * msc).powi(2) / (k - 1.0) + (b * mse).powi(2) / ((n - 1.0) * (k - 1.0)));
                let fs = dist::invftail(n - 1.0, v, alpha / 2.0);
                let fs2 = dist::invftail(v, n - 1.0, alpha / 2.0);
                let lo = n * (msr - fs * mse) / (fs * (k * msc + (k * n - k - n) * mse) + n * msr);
                let hi = n * (fs2 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fs2 * msr);
                ci1 = (lo, hi);
                let lk = |x: f64| k * x / (1.0 + (k - 1.0) * x);
                cik = (lk(lo), lk(hi));
            }
        }
        out.push(String::new());
        out.push(format!("Random effects: {:<16} Number of targets = {:>9}", abbrev(&vars[1], 16), n));
        out.push(format!("{:<32} Number of raters  = {:>9}", "", k));
        out.push(String::new());
        out.push("-".repeat(62));
        out.push(format!("{:>23} |        ICC       [{}% conf. interval]", abbrev(&vars[0], 23), fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(24), "-".repeat(37)));
        out.push(format!("{:>23} |  {:>9}    {:>9}    {:>9}", "Individual", g9(icc1), g9(ci1.0), g9(ci1.1)));
        out.push(format!("{:>23} |  {:>9}    {:>9}    {:>9}", "Average", g9(icck), g9(cik.0), g9(cik.1)));
        out.push("-".repeat(62));
        out.push("F test that".into());
        out.push(format!("  ICC=0.00: F({:.1}, {:.1}) = {:.2}{:>w$}Prob > F = {:.3}", df1, df2, f, "", pf, w = 8));
        r.set("N_target", n);
        r.set("N_rater", k);
        r.set("icc_i", icc1);
        r.set("icc_i_lb", ci1.0);
        r.set("icc_i_ub", ci1.1);
        r.set("icc_avg", icck);
        r.set("icc_avg_lb", cik.0);
        r.set("icc_avg_ub", cik.1);
        r.set("icc_i_F", f);
        r.set("icc_i_df1", df1);
        r.set("icc_i_df2", df2);
        r.set("icc_i_p", pf);
        r.set("level", level);
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Loneway;

impl CommandHandler for Loneway {
    fn name(&self) -> &'static str {
        "loneway"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "loneway response_var group_var"));
        }
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let y = util::col(ds, &vars[0])?;
        let g = util::col(ds, &vars[1])?;
        let mut groups: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
        for i in util::rows(&mask) {
            groups.entry(g[i].to_bits()).or_default().push(y[i]);
        }
        let ng = groups.len() as f64;
        let nn: f64 = groups.values().map(|v| v.len() as f64).sum();
        let gm = groups.values().flatten().sum::<f64>() / nn;
        let ssb: f64 = groups.values().map(|v| v.len() as f64 * (util::mean(v) - gm).powi(2)).sum();
        let ssw: f64 = groups.values().map(|v| {
            let m = util::mean(v);
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        }).sum();
        let (dfb, dfw) = (ng - 1.0, nn - ng);
        let (msb, msw) = (ssb / dfb, ssw / dfw);
        let f = msb / msw;
        let p = dist::ftail(dfb, dfw, f);
        let sum_n2: f64 = groups.values().map(|v| (v.len() as f64).powi(2)).sum();
        let n0 = (nn - sum_n2 / nn) / dfb;
        let sb2 = ((msb - msw) / n0).max(0.0);
        let rho = sb2 / (sb2 + msw);
        // asymptotic standard error (Smith 1957)
        let nbar = nn / ng;
        let _ = nbar;
        let sum_n3: f64 = groups.values().map(|v| (v.len() as f64).powi(3)).sum();
        let r = rho;
        let v_rho = 2.0 * (1.0 - r).powi(2) / (n0 * n0)
            * ((1.0 + r * (n0 - 1.0)).powi(2) / (nn - ng) + (ng - 1.0) * (1.0 - r) * (1.0 + r * (2.0 * n0 - 1.0)) / (ng - 1.0).powi(2)
                + r * r * (sum_n2 - 2.0 * sum_n3 / nn + sum_n2 * sum_n2 / (nn * nn)) / (ng - 1.0).powi(2));
        let se = v_rho.sqrt();
        let alpha = 1.0 - level / 100.0;
        let fl = f / dist::invftail(dfb, dfw, alpha / 2.0);
        let fu = f / dist::invftail(dfb, dfw, 1.0 - alpha / 2.0);
        let lb = ((fl - 1.0) / (fl + n0 - 1.0)).max(0.0);
        let ub = ((fu - 1.0) / (fu + n0 - 1.0)).min(1.0);
        let rel = n0 * rho / (1.0 + (n0 - 1.0) * rho);
        let gv = &vars[1];
        let mut out = vec![String::new(), format!("{:>18}One-way analysis of variance for {}: {}", "", vars[0], ds.get_var(&vars[0]).map(|v| v.label.clone()).unwrap_or_default()).trim_end().to_string(), String::new()];
        out.push(format!("{:>60} = {:>9}", "Number of obs", util::commas(nn as usize)));
        out.push(format!("{:>60} = {:>9.4}", "R-squared", ssb / (ssb + ssw)));
        out.push(String::new());
        out.push("    Source                SS         df      MS            F     Prob > F".into());
        out.push("-".repeat(73));
        out.push(format!("{:<18}{:>12} {:>8}  {:>11}  {:>10.2}   {:>8.4}", format!("Between {}", abbrev(gv, 10)), g9(ssb), dfb, g9(msb), f, p));
        out.push(format!("{:<18}{:>12} {:>8}  {:>11}", format!("Within {}", abbrev(gv, 10)), g9(ssw), dfw, g9(msw)));
        out.push("-".repeat(73));
        out.push(format!("{:<18}{:>12} {:>8}  {:>11}", "Total", g9(ssb + ssw), nn - 1.0, g9((ssb + ssw) / (nn - 1.0))));
        out.push(String::new());
        out.push("         Intraclass       Asy.".into());
        out.push(format!("         correlation      S.E.       [{}% conf. interval]", fmt_g(level, 4)));
        out.push("         ------------------------------------------------".into());
        out.push(format!("            {:>8.5}     {:>8.5}       {:>8.5}     {:>8.5}", rho, se, lb, ub));
        out.push(String::new());
        out.push(format!("         Estimated SD of {} effect{:>w$}", abbrev(gv, 12), format!("{:.6}", sb2.sqrt()), w = 34usize.saturating_sub(abbrev(gv, 12).len())));
        out.push(format!("         Estimated SD within {}{:>w$}", abbrev(gv, 12), format!("{:.6}", msw.sqrt()), w = 37usize.saturating_sub(abbrev(gv, 12).len())));
        out.push(format!("         Est. reliability of a {} mean{:>w$}", abbrev(gv, 12), format!("{:.5}", rel), w = 30usize.saturating_sub(abbrev(gv, 12).len())));
        out.push(format!("              (evaluated at n={:.2})", n0));
        let mut res = Results::new();
        res.set("N", nn);
        res.set("rho", rho);
        res.set("se", se);
        res.set("lb", lb);
        res.set("ub", ub);
        res.set("sd_b", sb2.sqrt());
        res.set("sd_w", msw.sqrt());
        res.set("rho_t", rel);
        res.set("F", f);
        res.set("df_m", dfb);
        res.set("df_r", dfw);
        res.set("mss", ssb);
        res.set("rss", ssw);
        st.r = res;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// pcorr
// ---------------------------------------------------------------------------

pub struct Pcorr;

impl CommandHandler for Pcorr {
    fn name(&self) -> &'static str {
        "pcorr"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let n = rows.len();
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
        let k = vars.len() - 1;
        // OLS of y on the others (with constant)
        let p = k + 1;
        let mut xtx = Matrix::zeros(p, p);
        let mut xty = vec![0.0; p];
        for t in 0..n {
            let mut row: Vec<f64> = (1..=k).map(|j| cols[j][t]).collect();
            row.push(1.0);
            for a in 0..p {
                xty[a] += row[a] * cols[0][t];
                for c in 0..p {
                    xtx[(a, c)] += row[a] * row[c];
                }
            }
        }
        let (inv, _) = xtx.sym_inverse(1e-14);
        let b: Vec<f64> = (0..p).map(|a| (0..p).map(|c| inv[(a, c)] * xty[c]).sum()).collect();
        let ym = util::mean(&cols[0]);
        let mut rss = 0.0;
        let mut tss = 0.0;
        for t in 0..n {
            let f: f64 = (0..k).map(|j| b[j] * cols[j + 1][t]).sum::<f64>() + b[k];
            rss += (cols[0][t] - f).powi(2);
            tss += (cols[0][t] - ym).powi(2);
        }
        let df = (n - p) as f64;
        let s2 = rss / df;
        let r2 = 1.0 - rss / tss;
        let mut out = vec![String::new(), format!("Partial and semipartial correlations of {} with", vars[0]), String::new()];
        out.push("               Partial   Semipartial      Partial   Semipartial   Significance".into());
        out.push("   Variable |    corr.         corr.      corr.^2       corr.^2          value".into());
        out.push(format!("{}+{}", "-".repeat(12), "-".repeat(65)));
        let mut pc = Vec::new();
        let mut sc = Vec::new();
        for j in 0..k {
            let t = b[j] / (s2 * inv[(j, j)]).sqrt();
            let pr = t / (t * t + df).sqrt();
            let sp = t * ((1.0 - r2) / df).sqrt();
            let pv = 2.0 * dist::ttail(df, t.abs());
            out.push(format!("{:>11} | {:>8.4}      {:>8.4}     {:>8.4}      {:>8.4}         {:>6.4}", abbrev(&vars[j + 1], 11), pr, sp, pr * pr, sp * sp, pv));
            pc.push(pr);
            sc.push(sp);
        }
        out.push(format!("{}", "-".repeat(78)));
        let names: Vec<String> = vars[1..].to_vec();
        let mut r = Results::new();
        r.set("N", n as f64);
        r.set("df", df);
        r.set_mat("p_corr", Matrix::col_vector(&pc).with_names(names.clone(), vec!["c1".into()]));
        r.set_mat("sp_corr", Matrix::col_vector(&sc).with_names(names, vec!["c1".into()]));
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// cumul
// ---------------------------------------------------------------------------

pub struct Cumul;

impl CommandHandler for Cumul {
    fn name(&self) -> &'static str {
        "cumul"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 1 {
            return Err(rc_err(103, "too many variables specified"));
        }
        let newv = cmd.opt("generate", 1).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fa")?;
        let x = util::col(ds, &vars[0])?;
        let freq = cmd.has_opt("freq", 1);
        let equal = cmd.has_opt("equal", 2);
        let by: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        util::markout(ds, &mut mask, &by);
        let (gid, _) = util::group_ids(ds, &by, &mask);
        let mut out = vec![SYSMISS; ds.n_obs];
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..ds.n_obs {
            if mask[i] && !is_missing(x[i]) {
                groups.entry(gid[i]).or_default().push(i);
            }
        }
        for rows in groups.values() {
            let mut r = rows.clone();
            r.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
            let wt = |i: usize| w.as_ref().map_or(1.0, |w| w.w[i]);
            let total: f64 = r.iter().map(|&i| wt(i)).sum();
            let mut cum = 0.0;
            let mut t = 0;
            while t < r.len() {
                let mut e = t;
                while e < r.len() && x[r[e]] == x[r[t]] {
                    e += 1;
                }
                if equal {
                    for &i in &r[t..e] {
                        cum += wt(i);
                    }
                    for &i in &r[t..e] {
                        out[i] = if freq { cum } else { cum / total };
                    }
                } else {
                    for &i in &r[t..e] {
                        cum += wt(i);
                        out[i] = if freq { cum } else { cum / total };
                    }
                }
                t = e;
            }
        }
        add_var(ds, &newv, if freq { VarType::Double } else { VarType::Float }, &out, &format!("ECDF of {}", vars[0]))?;
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// bitest, bitesti
// ---------------------------------------------------------------------------

pub struct Bitest;

impl CommandHandler for Bitest {
    fn name(&self) -> &'static str {
        "bitest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (label, n, k, p) = if cmd.verb == "bitesti" {
            let v: Vec<f64> = cmd.main.split_whitespace().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "bitesti #N #succ #p"))).collect::<Result<_, _>>()?;
            if v.len() != 3 {
                return Err(rc_err(198, "bitesti #N #succ #p"));
            }
            (String::new(), v[0], v[1], v[2])
        } else {
            let (lhs, rhs) = cmd.main.split_once("==").ok_or_else(|| rc_err(198, "bitest varname == #p"))?;
            let var = varlist::resolve_name(ds, lhs.trim())?;
            let p: f64 = rhs.trim().parse().map_err(|_| rc_err(198, "bitest varname == #p"))?;
            let mut mask = util::touse(ds, st, cmd)?;
            let w = util::weights(ds, st, cmd, &mut mask, "f")?;
            let x = util::col(ds, &var)?;
            let mut n = 0.0;
            let mut k = 0.0;
            for i in 0..ds.n_obs {
                if mask[i] && !is_missing(x[i]) {
                    if x[i] != 0.0 && x[i] != 1.0 {
                        return Err(rc_err(450, format!("{var} is not a 0/1 variable")));
                    }
                    let wt = w.as_ref().map_or(1.0, |w| w.w[i]);
                    n += wt;
                    k += wt * x[i];
                }
            }
            (var, n, k, p)
        };
        if !(0.0..=1.0).contains(&p) {
            return Err(rc_err(198, "p must be between 0 and 1"));
        }
        let ge = dist::binomialtail(n, k, p);
        let le = dist::binomial(n, k, p);
        // two-sided: outcomes no more likely than the observed one
        let pk = dist::binomialp(n, k, p);
        let mut two = 0.0;
        let mut other = SYSMISS;
        let exp_k = n * p;
        for j in 0..=(n as usize) {
            let pj = dist::binomialp(n, j as f64, p);
            if pj <= pk * (1.0 + 1e-7) {
                two += pj;
                let jf = j as f64;
                if (k <= exp_k && jf > exp_k) || (k > exp_k && jf < exp_k) {
                    if k <= exp_k {
                        if is_missing(other) {
                            other = jf;
                        }
                    } else {
                        other = jf;
                    }
                }
            }
        }
        let two = two.min(1.0);
        let mut out = vec![String::new()];
        out.push(format!("{:>12} |        N   Observed k   Expected k   Assumed p   Observed p", if label.is_empty() { "" } else { "Variable" }));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(60)));
        out.push(format!("{:>12} | {:>8}   {:>10}   {:>10.1}     {:>7.5}      {:>7.5}", abbrev(&label, 12), n, k, exp_k, p, k / n));
        out.push(String::new());
        out.push(format!("  Pr(k >= {})           = {:.6}  (one-sided test)", k, ge));
        out.push(format!("  Pr(k <= {})           = {:.6}  (one-sided test)", k, le));
        if is_missing(other) {
            out.push(format!("  Pr(k <= {} or k >= .) = {:.6}  (two-sided test)", k, two));
        } else if k <= exp_k {
            out.push(format!("  Pr(k <= {} or k >= {}) = {:.6}  (two-sided test)", k, other, two));
        } else {
            out.push(format!("  Pr(k <= {} or k >= {}) = {:.6}  (two-sided test)", other, k, two));
        }
        let mut r = Results::new();
        r.set("N", n);
        r.set("P_p", p);
        r.set("k", k);
        r.set("p_l", le);
        r.set("p_u", ge);
        r.set("p", two);
        if !is_missing(other) {
            r.set("k_opp", other);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// mkspline
// ---------------------------------------------------------------------------

pub struct Mkspline;

/// Default knot percentiles of restricted cubic splines (Harrell).
fn rcs_pctiles(k: usize) -> Option<Vec<f64>> {
    Some(match k {
        3 => vec![10.0, 50.0, 90.0],
        4 => vec![5.0, 35.0, 65.0, 95.0],
        5 => vec![5.0, 27.5, 50.0, 72.5, 95.0],
        6 => vec![5.0, 23.0, 41.0, 59.0, 77.0, 95.0],
        7 => vec![2.5, 18.3333, 34.1667, 50.0, 65.8333, 81.6667, 97.5],
        _ => return None,
    })
}

impl CommandHandler for Mkspline {
    fn name(&self) -> &'static str {
        "mkspline"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (lhs, rhs) = main.split_once('=').ok_or_else(|| rc_err(198, "mkspline newvar1 # newvar2 ... = oldvar"))?;
        let var = varlist::resolve_name(ds, rhs.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fa")?;
        let x = util::col(ds, &var)?;
        let words: Vec<&str> = lhs.split_whitespace().collect();
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(x[i])).collect();
        let mut sorted: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
        sorted.sort_by(f64::total_cmp);
        let _ = &w;
        let cubic = cmd.has_opt("cubic", 2);
        let marginal = cmd.has_opt("marginal", 3);
        let (names, knots): (Vec<String>, Vec<f64>) = if words.len() == 2 && words[1].parse::<usize>().is_ok() && !words[0].parse::<f64>().is_ok() {
            // mkspline stub # = x
            let stub = words[0];
            let nk = words[1].parse::<usize>().unwrap();
            let knots = if let Some(k) = cmd.opt("knots", 1) {
                k.split_whitespace().map(|s| s.parse::<f64>().map_err(|_| rc_err(198, "knots() invalid"))).collect::<Result<Vec<_>, _>>()?
            } else if cubic {
                let pc = rcs_pctiles(nk).ok_or_else(|| rc_err(198, "number of knots must be between 3 and 7"))?;
                pc.iter().map(|p| util::pctile_sorted(&sorted, *p)).collect()
            } else if cmd.has_opt("pctile", 2) {
                (1..nk).map(|j| util::pctile_sorted(&sorted, 100.0 * j as f64 / nk as f64)).collect()
            } else {
                let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
                (1..nk).map(|j| lo + (hi - lo) * j as f64 / nk as f64).collect()
            };
            let nv = if cubic { knots.len() - 1 } else { knots.len() + 1 };
            ((1..=nv).map(|j| format!("{stub}{j}")).collect(), knots)
        } else {
            let mut names = Vec::new();
            let mut knots = Vec::new();
            for (i, w) in words.iter().enumerate() {
                if i % 2 == 0 {
                    names.push(w.to_string());
                } else {
                    knots.push(w.parse::<f64>().map_err(|_| rc_err(198, format!("{w}: knot must be a number")))?);
                }
            }
            if names.len() != knots.len() + 1 {
                return Err(rc_err(198, "mkspline newvar1 #1 newvar2 #2 ... newvark = oldvar"));
            }
            (names, knots)
        };
        if knots.windows(2).any(|k| k[1] <= k[0]) {
            return Err(rc_err(198, "knot specification out of order"));
        }
        for nm in &names {
            varlist::check_name(nm)?;
            if ds.var_index.contains_key(nm) {
                return Err(rc_err(110, format!("variable {nm} already defined")));
            }
        }
        let n = ds.n_obs;
        let mut cols: Vec<Vec<f64>> = vec![vec![SYSMISS; n]; names.len()];
        for i in 0..n {
            if is_missing(x[i]) || !mask[i] && !cmd.has_opt("all", 3) && cmd.if_expr.is_some() {
                continue;
            }
            let xv = x[i];
            if cubic {
                let k = knots.len();
                let (t1, tk, tk1) = (knots[0], knots[k - 1], knots[k - 2]);
                let p3 = |v: f64| if v > 0.0 { v * v * v } else { 0.0 };
                cols[0][i] = xv;
                for j in 0..k - 2 {
                    let tj = knots[j];
                    cols[j + 1][i] = (p3(xv - tj) - p3(xv - tk1) * (tk - tj) / (tk - tk1) + p3(xv - tk) * (tk1 - tj) / (tk - tk1)) / ((tk - t1) * (tk - t1));
                }
            } else if marginal {
                cols[0][i] = xv;
                for (j, &k) in knots.iter().enumerate() {
                    cols[j + 1][i] = (xv - k).max(0.0);
                }
            } else {
                // piecewise: each variable carries the part of x within its interval
                let mut lo = f64::NEG_INFINITY;
                for j in 0..names.len() {
                    let hi = knots.get(j).copied().unwrap_or(f64::INFINITY);
                    cols[j][i] = if j == 0 {
                        xv.min(hi)
                    } else if xv < lo {
                        0.0
                    } else {
                        xv.min(hi) - lo
                    };
                    lo = hi;
                }
            }
        }
        for (j, nm) in names.iter().enumerate() {
            add_var(ds, nm, VarType::Float, &cols[j], "")?;
        }
        st.data_changed = true;
        let mut r = Results::new();
        r.set("N_knots", knots.len() as f64);
        r.set_mat("knots", Matrix::row_vector(&knots));
        st.r = r;
        if cmd.has_opt("displayknots", 4) {
            let mut out = vec![String::new(), format!("{:>12} |{}", "", (1..=knots.len()).map(|j| format!("{:>11}", format!("knot{j}"))).collect::<String>())];
            out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11 * knots.len())));
            out.push(format!("{:>12} |{}", abbrev(&var, 12), knots.iter().map(|k| format!("{:>11}", g9(*k))).collect::<String>()));
            return Ok(out.join("\n"));
        }
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// orthog, orthpoly
// ---------------------------------------------------------------------------

/// Orthonormalise columns (with a constant first) so that Q'Q = N·I with
/// mean-zero non-constant columns; returns Q (without the constant) and R
/// such that X = Q R (rows: new variables and _cons).
fn orthonormalize(cols: &[Vec<f64>]) -> (Vec<Vec<f64>>, Matrix) {
    let n = cols.first().map_or(0, |c| c.len());
    let nf = n as f64;
    let k = cols.len();
    let mut q: Vec<Vec<f64>> = vec![vec![1.0; n]];
    let mut r = Matrix::zeros(k + 1, k + 1);
    r[(k, k)] = 1.0;
    for (j, c) in cols.iter().enumerate() {
        let mut v = c.clone();
        // project out previous columns (q_0 = constant with q'q = N)
        let mut coefs = Vec::new();
        for qq in &q {
            let a = v.iter().zip(qq).map(|(x, y)| x * y).sum::<f64>() / nf;
            coefs.push(a);
            for t in 0..n {
                v[t] -= a * qq[t];
            }
        }
        let norm = (v.iter().map(|x| x * x).sum::<f64>() / nf).sqrt();
        for t in 0..n {
            v[t] /= norm;
        }
        // x_j = sum coefs q + norm v
        r[(k, j)] = coefs[0];
        for (m, a) in coefs.iter().enumerate().skip(1) {
            r[(m - 1, j)] = *a;
        }
        r[(j, j)] = norm;
        q.push(v);
    }
    (q.into_iter().skip(1).collect(), r)
}

pub struct Orthog;

impl CommandHandler for Orthog {
    fn name(&self) -> &'static str {
        "orthog"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let gspec = cmd.opt("generate", 1).ok_or_else(|| rc_err(198, "option generate() required"))?;
        let names = new_names(gspec, vars.len())?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
        let (q, r) = orthonormalize(&cols);
        put_cols(ds, &names, &q, &rows)?;
        if let Some(m) = cmd.opt("matrix", 3) {
            let mut rn = names.clone();
            rn.push("_cons".into());
            let mut cn = vars.clone();
            cn.push("_cons".into());
            st.matrices.insert(m.trim().to_string(), r.with_names(rn, cn));
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

fn new_names(spec: &str, k: usize) -> Result<Vec<String>, String> {
    let w: Vec<&str> = spec.split_whitespace().collect();
    if w.len() == 1 && w[0].ends_with('*') {
        let s = w[0].trim_end_matches('*');
        return Ok((1..=k).map(|j| format!("{s}{j}")).collect());
    }
    if w.len() != k {
        return Err(rc_err(198, format!("{k} new variable names required")));
    }
    Ok(w.iter().map(|s| s.to_string()).collect())
}

fn put_cols(ds: &mut DataSet, names: &[String], cols: &[Vec<f64>], rows: &[usize]) -> Result<(), String> {
    for (j, nm) in names.iter().enumerate() {
        let mut vals = vec![SYSMISS; ds.n_obs];
        for (t, &r) in rows.iter().enumerate() {
            vals[r] = cols[j][t];
        }
        add_var(ds, nm, VarType::Double, &vals, "")?;
    }
    Ok(())
}

pub struct Orthpoly;

impl CommandHandler for Orthpoly {
    fn name(&self) -> &'static str {
        "orthpoly"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 1 {
            return Err(rc_err(103, "too many variables specified"));
        }
        let deg = util::opt_num(cmd, "degree", 1)?.ok_or_else(|| rc_err(198, "option degree() required"))? as usize;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let x = util::col(ds, &vars[0])?;
        let cols: Vec<Vec<f64>> = (1..=deg).map(|p| rows.iter().map(|&i| x[i].powi(p as i32)).collect()).collect();
        let (q, r) = orthonormalize(&cols);
        if let Some(gspec) = cmd.opt("generate", 1) {
            let names = new_names(gspec, deg)?;
            put_cols(ds, &names, &q, &rows)?;
            st.data_changed = true;
        }
        if let Some(m) = cmd.opt("poly", 1) {
            // coefficients of the orthogonal polynomials in powers of x: inverse of R
            let inv = r.inverse()?;
            let mut cn: Vec<String> = (1..=deg).map(|p| format!("deg{p}")).collect();
            cn.push("_cons".into());
            let rn: Vec<String> = (1..=deg).map(|p| format!("deg{p}")).chain(std::iter::once("_cons".to_string())).collect();
            st.matrices.insert(m.trim().to_string(), inv.transpose().with_names(rn, cn));
        }
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// nptrend
// ---------------------------------------------------------------------------

pub struct Nptrend;

impl CommandHandler for Nptrend {
    fn name(&self) -> &'static str {
        "nptrend"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let gvar = varlist::resolve_name(ds, cmd.opt("by", 2).or_else(|| cmd.opt("group", 2)).ok_or_else(|| rc_err(198, "option group() required"))?.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = vars.clone();
        all.push(gvar.clone());
        util::markout(ds, &mut mask, &all);
        let y = util::col(ds, &vars[0])?;
        let g = util::col(ds, &gvar)?;
        let rows = util::rows(&mask);
        let n = rows.len() as f64;
        // midranks
        let mut idx = rows.clone();
        idx.sort_by(|&a, &b| y[a].total_cmp(&y[b]));
        let mut rank = vec![0.0; ds.n_obs];
        let mut ties = 0.0;
        let mut t = 0;
        while t < idx.len() {
            let mut e = t;
            while e < idx.len() && y[idx[e]] == y[idx[t]] {
                e += 1;
            }
            let m = (t + e + 1) as f64 / 2.0;
            for &i in &idx[t..e] {
                rank[i] = m;
            }
            let c = (e - t) as f64;
            ties += c * c * c - c;
            t = e;
        }
        let score_var = cmd.opt("score", 2).map(|s| varlist::resolve_name(ds, s.trim())).transpose()?;
        let sc = match &score_var {
            Some(s) => util::col(ds, s)?,
            None => g.clone(),
        };
        let mut groups: BTreeMap<u64, (f64, f64, f64, f64)> = BTreeMap::new(); // key -> (group value, score, n, rank sum)
        for &i in &rows {
            let key = {
                let b = g[i].to_bits();
                if g[i] >= 0.0 { b ^ (1 << 63) } else { !b }
            };
            let e = groups.entry(key).or_insert((g[i], sc[i], 0.0, 0.0));
            e.2 += 1.0;
            e.3 += rank[i];
        }
        let lsum: f64 = groups.values().map(|v| v.1 * v.2).sum();
        let tstat: f64 = groups.values().map(|v| v.1 * v.3).sum();
        let et = 0.5 * (n + 1.0) * lsum;
        let l2: f64 = groups.values().map(|v| v.1 * v.1 * v.2).sum();
        let var = (n + 1.0) / 12.0 * (n * l2 - lsum * lsum);
        let var = var * (1.0 - ties / (n * n * n - n));
        let z = (tstat - et) / var.sqrt();
        let p = 2.0 * dist::normal(-z.abs());
        let mut out = vec![String::new(), "Test for trend across ordered groups".to_string(), String::new()];
        out.push(format!("{:>12} |  score     obs   sum of ranks", abbrev(&gvar, 12)));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(31)));
        for v in groups.values() {
            out.push(format!("{:>12} | {:>6} {:>7} {:>14}", g9(v.0), g9(v.1), v.2, g9(v.3)));
        }
        out.push(String::new());
        out.push(format!("{:>14}  = {:>6.2}", "z", z));
        out.push(format!("{:>14}  = {:>6.3}", "Prob > |z|", p));
        let mut r = Results::new();
        r.set("N", n);
        r.set("z", z);
        r.set("p", p);
        r.set("T", tstat);
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// lv
// ---------------------------------------------------------------------------

pub struct Lv;

impl CommandHandler for Lv {
    fn name(&self) -> &'static str {
        "lv"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut out = Vec::new();
        let mut r = Results::new();
        for v in &vars {
            let mut x = values_of(ds, st, cmd, v)?;
            x.sort_by(f64::total_cmp);
            let n = x.len();
            if n == 0 {
                continue;
            }
            let at = |d: f64| -> (f64, f64) {
                // values at depth d from below and from above
                let lo = if d.fract() == 0.0 { x[d as usize - 1] } else { (x[d.floor() as usize - 1] + x[d.ceil() as usize - 1]) / 2.0 };
                let hi = if d.fract() == 0.0 { x[n - d as usize] } else { (x[n - d.floor() as usize] + x[n - d.ceil() as usize]) / 2.0 };
                (lo, hi)
            };
            out.push(String::new());
            out.push(format!("{:>3}{:>6}{:>26}", "#", n, v));
            out.push("-".repeat(64));
            let mut depth = (n as f64 + 1.0) / 2.0;
            let (m, _) = at(depth);
            out.push(format!("M{:>7}  |{:>23}{:>13}|{:>10}   {:>12}", fmt_dep(depth), "", g9(m), "spread", "pseudosigma"));
            r.set("median", m);
            let letters = ["F", "E", "D", "C", "B", "A", "Z", "Y", "X", "W"];
            let mut li = 0;
            let mut fourth = (m, m);
            loop {
                depth = (depth.floor() + 1.0) / 2.0;
                if depth < 1.0 || li >= letters.len() {
                    break;
                }
                let (lo, hi) = at(depth);
                if li == 0 {
                    fourth = (lo, hi);
                }
                let mid = (lo + hi) / 2.0;
                let spread = hi - lo;
                let p = (depth - 1.0 / 3.0) / (n as f64 + 1.0 / 3.0);
                let ps = spread / (-2.0 * dist::invnormal(p));
                out.push(format!("{}{:>7}  | {:>10} {:>11} {:>10} | {:>9}   {:>12}", letters[li], fmt_dep(depth), g9(lo), g9(mid), g9(hi), g9(spread), g9(ps)));
                li += 1;
                if depth <= 1.0 {
                    break;
                }
            }
            if depth > 1.0 || li == 0 {
                out.push(format!("{}{:>7}  | {:>10} {:>11} {:>10} | {:>9}", " ", "1", g9(x[0]), g9((x[0] + x[n - 1]) / 2.0), g9(x[n - 1]), g9(x[n - 1] - x[0])));
            }
            out.push("".into());
            let iqr = fourth.1 - fourth.0;
            let (if_l, if_u) = (fourth.0 - 1.5 * iqr, fourth.1 + 1.5 * iqr);
            let (of_l, of_u) = (fourth.0 - 3.0 * iqr, fourth.1 + 3.0 * iqr);
            let cnt = |lo: f64, hi: f64| (x.iter().filter(|&&v| v < lo).count(), x.iter().filter(|&&v| v > hi).count());
            let (a, b) = cnt(if_l, if_u);
            let (c, d) = cnt(of_l, of_u);
            out.push(format!("{:>10}  |{:>11}{:>23}|{:>10}", "", "", "", "# below     # above"));
            out.push(format!("{:>10}  | {:>10} {:>11} {:>10} | {:>8} {:>9}", "inner fence", g9(if_l), "", g9(if_u), a, b));
            out.push(format!("{:>10}  | {:>10} {:>11} {:>10} | {:>8} {:>9}", "outer fence", g9(of_l), "", g9(of_u), c, d));
            r.set("N", n as f64);
            r.set("l_F", fourth.0);
            r.set("u_F", fourth.1);
            r.set("min", x[0]);
            r.set("max", x[n - 1]);
            r.set("l_in", if_l);
            r.set("u_in", if_u);
            r.set("l_out", of_l);
            r.set("u_out", of_u);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn fmt_dep(d: f64) -> String {
    if d.fract() == 0.0 {
        format!("{}", d as i64)
    } else {
        format!("{:.1}", d)
    }
}
