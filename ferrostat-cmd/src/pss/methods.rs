//! Power methods for means, proportions, variances, correlations and the
//! slope of a simple regression.

use ferrostat_parser::Command;

use super::ncdist::{pw_norm, t_power, z_power, zcrit};
use super::{flag, g, has, opt_list, word_opt, Def, Row};
use crate::dist;
use crate::eval::rc_err;

pub(crate) type Lists = Vec<(String, Vec<f64>)>;

/// Full method name for a (possibly abbreviated) method word.
pub(crate) fn canonical(w: &str) -> Option<&'static str> {
    let w = w.to_lowercase();
    let names: &[(&str, usize)] = &[
        ("onemean", 4),
        ("twomeans", 4),
        ("pairedmeans", 7),
        ("oneproportion", 6),
        ("twoproportions", 6),
        ("pairedproportions", 7),
        ("onevariance", 6),
        ("twovariances", 6),
        ("onecorrelation", 6),
        ("twocorrelations", 6),
        ("oneway", 6),
        ("twoway", 6),
        ("repeated", 6),
        ("oneslope", 8),
        ("rsquared", 8),
        ("pcorr", 5),
        ("cmh", 3),
        ("mcc", 3),
        ("trend", 5),
        ("cox", 3),
        ("exponential", 3),
        ("logrank", 7),
        ("logistic", 8),
    ];
    names.iter().find(|(n, m)| w.len() >= *m && n.starts_with(w.as_str())).map(|(n, _)| *n)
}

pub(crate) fn item(pos: &[Vec<Vec<f64>>], i: usize) -> Option<Vec<f64>> {
    pos.first().and_then(|r| r.get(i)).cloned()
}

pub(crate) fn push(l: &mut Lists, k: &str, v: Option<Vec<f64>>) {
    if let Some(v) = v {
        l.push((k.to_string(), v));
    }
}

pub(crate) fn push_flag(l: &mut Lists, k: &str, on: bool) {
    if on {
        l.push((k.to_string(), vec![1.0]));
    }
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Alternative value: the parameter itself, or null + diff.
fn alt(r: &Row, key: &str, null: f64) -> f64 {
    if has(r, key) {
        g(r, key)
    } else if has(r, "diff") {
        null + g(r, "diff")
    } else {
        f64::NAN
    }
}

fn os(r: &Row) -> bool {
    flag(r, "_onesided")
}

fn hyp(par: &str, null: &str) -> String {
    format!("H0: {par} = {null}  versus  Ha: {par} != {null}")
}

pub(crate) fn setup(m: &str, cmd: &Command, pos: &[Vec<Vec<f64>>]) -> Result<(Def, Lists), String> {
    let mut l: Lists = Vec::new();
    let mut d = Def { method: m.to_string(), ..Default::default() };
    let o = |n: &str, k: usize| opt_list(cmd, n, k);
    match m {
        "onemean" => {
            push(&mut l, "m0", Some(item(pos, 0).unwrap_or(vec![0.0])));
            push(&mut l, "ma", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            push(&mut l, "sd", Some(o("sd", 2)?.unwrap_or(vec![1.0])));
            let known = cmd.has_opt("knownsd", 6);
            push_flag(&mut l, "_z", known);
            d.title = "a one-sample mean test".into();
            d.test = if known { "z test".into() } else { "t test".into() };
            d.hyp = hyp("m", "m0");
            d.cluster = true;
            d.target = Some("ma");
            d.tword = "mean";
            d.show = strs(&["m0", "ma", "diff", "sd"]);
            d.power = |r, n, _| {
                let m0 = g(r, "m0");
                let ncp = (alt(r, "ma", m0) - m0) / g(r, "sd") * n.sqrt();
                if flag(r, "_z") || flag(r, "_cluster") {
                    z_power(ncp, g(r, "alpha"), os(r))
                } else {
                    t_power(n - 1.0, ncp, g(r, "alpha"), os(r))
                }
            };
            d.delta = |r| (alt(r, "ma", g(r, "m0")) - g(r, "m0")) / g(r, "sd");
            d.erange = |r, up| (g(r, "m0"), g(r, "m0") + if up { 1e3 } else { -1e3 } * g(r, "sd"));
        }
        "twomeans" => {
            push(&mut l, "m1", Some(item(pos, 0).unwrap_or(vec![0.0])));
            push(&mut l, "m2", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            push(&mut l, "sd", o("sd", 2)?);
            push(&mut l, "sd1", o("sd1", 3)?);
            push(&mut l, "sd2", o("sd2", 3)?);
            let known = cmd.has_opt("knownsds", 6);
            push_flag(&mut l, "_z", known);
            d.title = "a two-sample means test".into();
            d.test = if known { "z test".into() } else { "t test".into() };
            d.hyp = hyp("m2", "m1");
            d.two = true;
            d.cluster = true;
            d.target = Some("m2");
            d.tword = "experimental-group mean";
            d.show = strs(&["m1", "m2", "diff", "sd", "sd1", "sd2"]);
            d.power = |r, n1, n2| {
                let m1 = g(r, "m1");
                let (s1, s2) = (sdk(r, "sd1"), sdk(r, "sd2"));
                let (u, v) = (s1 * s1 / n1, s2 * s2 / n2);
                let ncp = (alt(r, "m2", m1) - m1) / (u + v).sqrt();
                if flag(r, "_z") || flag(r, "_cluster") {
                    z_power(ncp, g(r, "alpha"), os(r))
                } else {
                    let df = if (s1 - s2).abs() < 1e-12 { n1 + n2 - 2.0 } else { (u + v).powi(2) / (u * u / (n1 - 1.0) + v * v / (n2 - 1.0)) };
                    t_power(df, ncp, g(r, "alpha"), os(r))
                }
            };
            d.delta = |r| alt(r, "m2", g(r, "m1")) - g(r, "m1");
            d.erange = |r, up| (g(r, "m1"), g(r, "m1") + if up { 1e3 } else { -1e3 } * sdk(r, "sd1").max(sdk(r, "sd2")));
        }
        "pairedmeans" => {
            let have_ma1 = item(pos, 0).is_some();
            push(&mut l, "ma1", item(pos, 0));
            push(&mut l, "ma2", item(pos, 1));
            push(&mut l, "nulldiff", Some(o("nulldiff", 5)?.unwrap_or(vec![0.0])));
            push(&mut l, "altdiff", o("altdiff", 4)?);
            push(&mut l, "sd_d", o("sddiff", 4)?);
            push(&mut l, "sd", o("sd", 2)?);
            push(&mut l, "sd1", o("sd1", 3)?);
            push(&mut l, "sd2", o("sd2", 3)?);
            push(&mut l, "corr", o("corr", 4)?);
            let known = cmd.has_opt("knownsd", 6);
            push_flag(&mut l, "_z", known);
            d.title = "a two-sample paired-means test".into();
            d.test = if known { "z test".into() } else { "t test".into() };
            d.hyp = hyp("d", "d0");
            d.target = Some(if have_ma1 { "ma2" } else { "altdiff" });
            d.tword = if have_ma1 { "mean" } else { "mean difference" };
            d.show = strs(&["nulldiff", "altdiff", "sd_d", "ma1", "ma2", "sd", "sd1", "sd2", "corr"]);
            d.power = |r, n, _| {
                let da = if has(r, "ma2") && has(r, "ma1") { g(r, "ma2") - g(r, "ma1") } else { g(r, "altdiff") };
                let ncp = (da - g(r, "nulldiff")) / g(r, "sd_d") * n.sqrt();
                if flag(r, "_z") {
                    z_power(ncp, g(r, "alpha"), os(r))
                } else {
                    t_power(n - 1.0, ncp, g(r, "alpha"), os(r))
                }
            };
            d.delta = |r| {
                let da = if has(r, "ma2") && has(r, "ma1") { g(r, "ma2") - g(r, "ma1") } else { g(r, "altdiff") };
                (da - g(r, "nulldiff")) / g(r, "sd_d")
            };
            d.erange = |r, up| {
                let base = if has(r, "ma1") { g(r, "ma1") } else { 0.0 } + g(r, "nulldiff");
                (base, base + if up { 1e3 } else { -1e3 } * g(r, "sd_d"))
            };
        }
        "oneproportion" => {
            push(&mut l, "p0", Some(item(pos, 0).ok_or_else(|| rc_err(198, "null proportion p0 required"))?));
            push(&mut l, "pa", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            let test = word_opt(cmd, "test", 4).unwrap_or_else(|| "score".into());
            let code = match test.as_str() {
                "score" => 0.0,
                "wald" => 1.0,
                "binomial" => 2.0,
                t => return Err(rc_err(198, format!("test({t}) not allowed"))),
            };
            l.push(("_test".into(), vec![code]));
            d.exact = code == 2.0;
            d.title = "a one-sample proportion test".into();
            d.test = match code as i32 {
                0 => "Score z test",
                1 => "Wald z test",
                _ => "Binomial test",
            }
            .into();
            d.hyp = hyp("p", "p0");
            d.cluster = code < 2.0;
            d.target = Some("pa");
            d.tword = "proportion";
            d.nmin = 1.0;
            d.show = strs(&["p0", "pa", "diff"]);
            d.power = |r, n, _| {
                let p0 = g(r, "p0");
                let pa = alt(r, "pa", p0);
                let (a, one) = (g(r, "alpha"), os(r));
                match g(r, "_test") as i32 {
                    2 => binomial_power(n.round(), p0, pa, a, one),
                    1 => pw_norm((pa - p0) * n.sqrt(), (pa * (1.0 - pa)).sqrt(), (pa * (1.0 - pa)).sqrt(), a, one),
                    _ => pw_norm((pa - p0) * n.sqrt(), (p0 * (1.0 - p0)).sqrt(), (pa * (1.0 - pa)).sqrt(), a, one),
                }
            };
            d.delta = |r| alt(r, "pa", g(r, "p0")) - g(r, "p0");
            d.erange = |r, up| (g(r, "p0"), if up { 1.0 - 1e-9 } else { 1e-9 });
        }
        "twoproportions" => {
            push(&mut l, "p1", Some(item(pos, 0).ok_or_else(|| rc_err(198, "control-group proportion p1 required"))?));
            push(&mut l, "p2", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            push(&mut l, "ratio", o("ratio", 5)?);
            push(&mut l, "oratio", o("oratio", 2)?);
            let test = word_opt(cmd, "test", 4).unwrap_or_else(|| "chi2".into());
            let code = match test.as_str() {
                "chi2" => 0.0,
                "lrchi2" => 1.0,
                "fisher" => 2.0,
                t => return Err(rc_err(198, format!("test({t}) not allowed"))),
            };
            l.push(("_test".into(), vec![code]));
            push_flag(&mut l, "_cc", cmd.has_opt("continuity", 4));
            d.exact = code == 2.0;
            d.title = "a two-sample proportions test".into();
            d.test = match code as i32 {
                0 => "Pearson's chi-squared test",
                1 => "Likelihood-ratio test",
                _ => "Fisher's exact test",
            }
            .into();
            d.hyp = hyp("p2", "p1");
            d.two = true;
            d.cluster = code == 0.0;
            d.target = Some("p2");
            d.tword = "experimental-group proportion";
            d.nmin = 1.0;
            d.show = strs(&["p1", "p2", "diff", "ratio", "oratio"]);
            d.power = |r, n1, n2| {
                let p1 = g(r, "p1");
                let p2 = p2_of(r);
                let (a, one) = (g(r, "alpha"), os(r));
                match g(r, "_test") as i32 {
                    2 => fisher_power(n1.round() as usize, n2.round() as usize, p1, p2, a, one),
                    1 => lr_power(n1, n2, p1, p2, a, one),
                    _ => {
                        let n = n1 + n2;
                        let pbar = (n1 * p1 + n2 * p2) / n;
                        let s0 = (pbar * (1.0 - pbar) * (1.0 / n1 + 1.0 / n2)).sqrt();
                        let s1 = (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt();
                        let mut dd = p2 - p1;
                        if flag(r, "_cc") {
                            let c = 0.5 * (1.0 / n1 + 1.0 / n2);
                            dd = dd.signum() * (dd.abs() - c).max(0.0);
                        }
                        pw_norm(dd, s0, s1, a, one)
                    }
                }
            };
            d.delta = |r| p2_of(r) - g(r, "p1");
            d.erange = |r, up| (g(r, "p1"), if up { 1.0 - 1e-9 } else { 1e-9 });
        }
        "pairedproportions" => {
            let corr = o("corr", 4)?;
            if corr.is_some() {
                push(&mut l, "pmarg1", item(pos, 0));
                push(&mut l, "pmarg2", item(pos, 1));
                push(&mut l, "corr", corr);
            } else {
                push(&mut l, "p12", item(pos, 0));
                push(&mut l, "p21", item(pos, 1));
                push(&mut l, "diff", o("diff", 4)?);
                d.target = Some("p21");
            }
            d.title = "a two-sample paired-proportions test".into();
            d.test = "Large-sample McNemar's test".into();
            d.hyp = "H0: p21 = p12  versus  Ha: p21 != p12".into();
            d.tword = "discordant proportion";
            d.nmin = 1.0;
            d.show = strs(&["pmarg1", "pmarg2", "corr", "p12", "p21", "diff"]);
            d.power = |r, n, _| {
                let p12 = g(r, "p12");
                let p21 = if has(r, "p21") { g(r, "p21") } else { p12 + g(r, "diff") };
                let dd = p21 - p12;
                let pd = p12 + p21;
                pw_norm(dd * n.sqrt(), pd.sqrt(), (pd - dd * dd).max(1e-300).sqrt(), g(r, "alpha"), os(r))
            };
            d.delta = |r| if has(r, "p21") { g(r, "p21") - g(r, "p12") } else { g(r, "diff") };
            d.erange = |r, up| (g(r, "p12"), if up { 1.0 - g(r, "p12") - 1e-9 } else { 1e-9 });
        }
        "onevariance" => {
            let sd = cmd.opts.iter().any(|(n, a)| n.eq_ignore_ascii_case("sd") && a.is_none());
            if sd {
                push(&mut l, "s0", item(pos, 0));
                push(&mut l, "sa", item(pos, 1));
                l.push(("_sd".into(), vec![1.0]));
            } else {
                push(&mut l, "v0", item(pos, 0));
                push(&mut l, "va", item(pos, 1));
            }
            push(&mut l, "ratio", o("ratio", 5)?);
            d.title = if sd { "a one-sample standard-deviation test" } else { "a one-sample variance test" }.into();
            d.test = "Chi-squared test".into();
            d.hyp = if sd { hyp("sd", "sd0") } else { hyp("v", "v0") };
            d.target = Some("va");
            d.tword = "variance";
            d.show = strs(&["s0", "sa", "v0", "va", "ratio"]);
            d.power = |r, n, _| var_power(n - 1.0, va_of(r) / g(r, "v0"), g(r, "alpha"), os(r));
            d.delta = |r| va_of(r) / g(r, "v0");
            d.erange = |r, up| (g(r, "v0"), g(r, "v0") * if up { 1e8 } else { 1e-8 });
            d.extra = Some(|r, _, _| if flag(r, "_sd") { vec![("sa".into(), va_of(r).sqrt())] } else { vec![] });
        }
        "twovariances" => {
            let sd = cmd.opts.iter().any(|(n, a)| n.eq_ignore_ascii_case("sd") && a.is_none());
            if sd {
                push(&mut l, "s1", item(pos, 0));
                push(&mut l, "s2", item(pos, 1));
                l.push(("_sd".into(), vec![1.0]));
            } else {
                push(&mut l, "v1", item(pos, 0));
                push(&mut l, "v2", item(pos, 1));
            }
            push(&mut l, "ratio", o("ratio", 5)?);
            d.title = if sd { "a two-sample standard-deviations test" } else { "a two-sample variances test" }.into();
            d.test = "F test".into();
            d.hyp = if sd { hyp("sd2", "sd1") } else { hyp("v2", "v1") };
            d.two = true;
            d.target = Some("v2");
            d.tword = "experimental-group variance";
            d.show = strs(&["s1", "s2", "v1", "v2", "ratio"]);
            d.power = |r, n1, n2| {
                let rho = v2_of(r) / g(r, "v1");
                let (d1, d2) = (n2 - 1.0, n1 - 1.0);
                if !(d1 > 0.0 && d2 > 0.0) {
                    return f64::NAN;
                }
                let a = g(r, "alpha");
                if os(r) {
                    if rho >= 1.0 {
                        dist::ftail(d1, d2, dist::invftail(d1, d2, a) / rho)
                    } else {
                        dist::f_cdf(d1, d2, dist::invf(d1, d2, a) / rho)
                    }
                } else {
                    dist::ftail(d1, d2, dist::invftail(d1, d2, a / 2.0) / rho) + dist::f_cdf(d1, d2, dist::invf(d1, d2, a / 2.0) / rho)
                }
            };
            d.delta = |r| v2_of(r) / g(r, "v1");
            d.erange = |r, up| (g(r, "v1"), g(r, "v1") * if up { 1e8 } else { 1e-8 });
            d.extra = Some(|r, _, _| if flag(r, "_sd") { vec![("s2".into(), v2_of(r).sqrt())] } else { vec![] });
        }
        "onecorrelation" => {
            push(&mut l, "r0", Some(item(pos, 0).unwrap_or(vec![0.0])));
            push(&mut l, "ra", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            d.title = "a one-sample correlation test".into();
            d.test = "Fisher's z test".into();
            d.hyp = hyp("r", "r0");
            d.target = Some("ra");
            d.tword = "correlation";
            d.nmin = 4.0;
            d.show = strs(&["r0", "ra", "diff"]);
            d.power = |r, n, _| {
                let r0 = g(r, "r0");
                let ncp = (alt(r, "ra", r0).atanh() - r0.atanh()) * (n - 3.0).sqrt();
                z_power(ncp, g(r, "alpha"), os(r))
            };
            d.delta = |r| alt(r, "ra", g(r, "r0")) - g(r, "r0");
            d.erange = |r, up| (g(r, "r0"), if up { 1.0 - 1e-12 } else { -1.0 + 1e-12 });
        }
        "twocorrelations" => {
            push(&mut l, "r1", Some(item(pos, 0).ok_or_else(|| rc_err(198, "control-group correlation r1 required"))?));
            push(&mut l, "r2", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            d.title = "a two-sample correlations test".into();
            d.test = "Fisher's z test".into();
            d.hyp = hyp("r2", "r1");
            d.two = true;
            d.target = Some("r2");
            d.tword = "experimental-group correlation";
            d.nmin = 4.0;
            d.show = strs(&["r1", "r2", "diff"]);
            d.power = |r, n1, n2| {
                let r1 = g(r, "r1");
                let ncp = (alt(r, "r2", r1).atanh() - r1.atanh()) / (1.0 / (n1 - 3.0) + 1.0 / (n2 - 3.0)).sqrt();
                z_power(ncp, g(r, "alpha"), os(r))
            };
            d.delta = |r| alt(r, "r2", g(r, "r1")) - g(r, "r1");
            d.erange = |r, up| (g(r, "r1"), if up { 1.0 - 1e-12 } else { -1.0 + 1e-12 });
        }
        "oneslope" => {
            push(&mut l, "b0", Some(item(pos, 0).unwrap_or(vec![0.0])));
            push(&mut l, "ba", item(pos, 1));
            push(&mut l, "diff", o("diff", 4)?);
            push(&mut l, "sdx", Some(o("sdx", 3)?.unwrap_or(vec![1.0])));
            push(&mut l, "sderror", o("sderror", 4)?);
            push(&mut l, "sdy", o("sdy", 3)?);
            d.title = "a linear regression slope test".into();
            d.test = "t test".into();
            d.hyp = hyp("b", "b0");
            d.target = Some("ba");
            d.tword = "slope";
            d.nmin = 3.0;
            d.show = strs(&["b0", "ba", "diff", "sdx", "sdy", "sderror"]);
            d.power = |r, n, _| {
                let b0 = g(r, "b0");
                let ncp = (alt(r, "ba", b0) - b0) * g(r, "sdx") * n.sqrt() / sderr(r);
                t_power(n - 2.0, ncp, g(r, "alpha"), os(r))
            };
            d.delta = |r| (alt(r, "ba", g(r, "b0")) - g(r, "b0")) * g(r, "sdx") / sderr(r);
            d.erange = |r, up| (g(r, "b0"), g(r, "b0") + if up { 1e3 } else { -1e3 } * g(r, "sderror").max(1e-8) / g(r, "sdx"));
        }
        other => return super::methods2::setup(other, cmd, pos, l, d),
    }
    Ok((d, l))
}

fn sdk(r: &Row, k: &str) -> f64 {
    if has(r, k) {
        g(r, k)
    } else {
        g(r, "sd")
    }
}

fn sderr(r: &Row) -> f64 {
    if has(r, "sderror") {
        g(r, "sderror")
    } else {
        let b = alt(r, "ba", g(r, "b0"));
        (g(r, "sdy").powi(2) - b * b * g(r, "sdx").powi(2)).sqrt()
    }
}

fn va_of(r: &Row) -> f64 {
    if has(r, "va") {
        g(r, "va")
    } else if has(r, "ratio") {
        g(r, "v0") * g(r, "ratio")
    } else {
        f64::NAN
    }
}

fn v2_of(r: &Row) -> f64 {
    if has(r, "v2") {
        g(r, "v2")
    } else if has(r, "ratio") {
        g(r, "v1") * g(r, "ratio")
    } else {
        f64::NAN
    }
}

pub(crate) fn p2_of(r: &Row) -> f64 {
    let p1 = g(r, "p1");
    if has(r, "p2") {
        g(r, "p2")
    } else if has(r, "diff") {
        p1 + g(r, "diff")
    } else if has(r, "ratio") {
        p1 * g(r, "ratio")
    } else if has(r, "oratio") {
        let o = g(r, "oratio");
        o * p1 / (1.0 - p1 + o * p1)
    } else {
        f64::NAN
    }
}

/// Derived parameters of a design point.
pub(crate) fn prepare(def: &Def, r: &mut Row) -> Result<(), String> {
    match def.method.as_str() {
        "twomeans" => {
            if !has(r, "sd") && !(has(r, "sd1") && has(r, "sd2")) {
                r.insert("sd".into(), 1.0);
            }
        }
        "pairedmeans" => {
            if !has(r, "sd_d") {
                let sd = if has(r, "sd") { g(r, "sd") } else { 1.0 };
                let s1 = if has(r, "sd1") { g(r, "sd1") } else { sd };
                let s2 = if has(r, "sd2") { g(r, "sd2") } else { sd };
                if !has(r, "corr") {
                    return Err(rc_err(198, "power pairedmeans: sddiff() or corr() required"));
                }
                let c = g(r, "corr");
                r.insert("sd_d".into(), (s1 * s1 + s2 * s2 - 2.0 * c * s1 * s2).sqrt());
            }
            if has(r, "ma1") && has(r, "ma2") {
                r.insert("altdiff".into(), g(r, "ma2") - g(r, "ma1"));
            }
        }
        "pairedproportions" => {
            if has(r, "corr") {
                let (a, b, c) = (g(r, "pmarg1"), g(r, "pmarg2"), g(r, "corr"));
                let p11 = a * b + c * (a * (1.0 - a) * b * (1.0 - b)).sqrt();
                r.insert("p12".into(), a - p11);
                r.insert("p21".into(), b - p11);
            }
        }
        "onevariance" => {
            if flag(r, "_sd") {
                if has(r, "s0") {
                    r.insert("v0".into(), g(r, "s0").powi(2));
                }
                if has(r, "sa") {
                    r.insert("va".into(), g(r, "sa").powi(2));
                }
            }
            if !has(r, "v0") {
                return Err(rc_err(198, "null variance required"));
            }
        }
        "twovariances" => {
            if flag(r, "_sd") {
                if has(r, "s1") {
                    r.insert("v1".into(), g(r, "s1").powi(2));
                }
                if has(r, "s2") {
                    r.insert("v2".into(), g(r, "s2").powi(2));
                }
            }
            if !has(r, "v1") {
                return Err(rc_err(198, "control-group variance required"));
            }
        }
        "oneslope" => {
            if !has(r, "sderror") && !has(r, "sdy") {
                r.insert("sderror".into(), 1.0);
            }
        }
        _ => super::methods2::prepare(def, r)?,
    }
    Ok(())
}

/// Exact binomial test power (equal-tailed two-sided rejection region).
pub(crate) fn binomial_power(n: f64, p0: f64, pa: f64, alpha: f64, onesided: bool) -> f64 {
    if n < 1.0 {
        return f64::NAN;
    }
    let a = if onesided { alpha } else { alpha / 2.0 };
    let lower = !onesided || pa < p0;
    let upper = !onesided || pa > p0;
    let mut pw = 0.0;
    if lower {
        // largest k with P(X <= k | p0) <= a
        let mut k = -1.0;
        let mut lo = 0.0;
        let mut hi = n;
        if dist::binomial(n, 0.0, p0) <= a {
            while hi - lo > 0.5 {
                let mid = ((lo + hi) / 2.0).ceil();
                if dist::binomial(n, mid, p0) <= a {
                    lo = mid;
                } else {
                    hi = mid - 1.0;
                }
            }
            k = lo;
        }
        if k >= 0.0 {
            pw += dist::binomial(n, k, pa);
        }
    }
    if upper {
        // smallest k with P(X >= k | p0) <= a
        let mut k = n + 1.0;
        if dist::binomialtail(n, n, p0) <= a {
            let (mut lo, mut hi) = (0.0, n);
            while hi - lo > 0.5 {
                let mid = ((lo + hi) / 2.0).floor();
                if dist::binomialtail(n, mid, p0) <= a {
                    hi = mid;
                } else {
                    lo = mid + 1.0;
                }
            }
            k = hi;
        }
        if k <= n {
            pw += dist::binomialtail(n, k, pa);
        }
    }
    pw
}

/// Power of the likelihood-ratio test of two proportions: noncentral
/// chi-squared with the expected LR statistic as noncentrality.
fn lr_power(n1: f64, n2: f64, p1: f64, p2: f64, alpha: f64, onesided: bool) -> f64 {
    let n = n1 + n2;
    let pb = (n1 * p1 + n2 * p2) / n;
    let term = |ni: f64, p: f64| {
        let mut s = 0.0;
        if p > 0.0 {
            s += p * (p / pb).ln();
        }
        if p < 1.0 {
            s += (1.0 - p) * ((1.0 - p) / (1.0 - pb)).ln();
        }
        ni * s
    };
    let ncp = 2.0 * (term(n1, p1) + term(n2, p2));
    if onesided {
        z_power(ncp.sqrt(), alpha, true)
    } else {
        super::ncdist::chi2_power(1.0, ncp, alpha)
    }
}

/// Exact power of Fisher's exact test, summing binomial probabilities over
/// the conditional rejection regions of each margin total.
pub(crate) fn fisher_power(n1: usize, n2: usize, p1: f64, p2: f64, alpha: f64, onesided: bool) -> f64 {
    if n1 == 0 || n2 == 0 {
        return f64::NAN;
    }
    let lb1: Vec<f64> = (0..=n1).map(|k| ln_binom_pmf(n1, k, p1)).collect();
    let lb2: Vec<f64> = (0..=n2).map(|k| ln_binom_pmf(n2, k, p2)).collect();
    let lc1: Vec<f64> = (0..=n1).map(|k| ln_comb(n1, k)).collect();
    let lc2: Vec<f64> = (0..=n2).map(|k| ln_comb(n2, k)).collect();
    let up = p2 > p1;
    let mut pw = 0.0;
    let tol = 1.0 + 1e-7;
    for t in 0..=(n1 + n2) {
        let lo = t.saturating_sub(n2);
        let hi = t.min(n1);
        let lw: Vec<f64> = (lo..=hi).map(|x| lc1[x] + lc2[t - x]).collect();
        let mx = lw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = lw.iter().map(|v| (v - mx).exp()).collect();
        let tot: f64 = w.iter().sum();
        let pr: Vec<f64> = w.iter().map(|v| v / tot).collect();
        let m = pr.len();
        // p-values
        let mut reject = vec![false; m];
        if onesided {
            // experimental group (x2 = t - x1) large ⇔ x1 small
            let mut c = 0.0;
            if up {
                for i in 0..m {
                    c += pr[i];
                    reject[i] = c <= alpha * tol;
                }
            } else {
                for i in (0..m).rev() {
                    c += pr[i];
                    reject[i] = c <= alpha * tol;
                }
            }
        } else {
            let mut idx: Vec<usize> = (0..m).collect();
            idx.sort_by(|a, b| pr[*a].total_cmp(&pr[*b]));
            let mut c = 0.0;
            let mut k = 0;
            while k < m {
                // ties share their p-value
                let mut j = k;
                let mut s = 0.0;
                while j < m && pr[idx[j]] <= pr[idx[k]] * tol {
                    s += pr[idx[j]];
                    j += 1;
                }
                c += s;
                for &i in &idx[k..j] {
                    reject[i] = c <= alpha * tol;
                }
                k = j;
            }
        }
        for (i, rj) in reject.iter().enumerate() {
            if *rj {
                let x1 = lo + i;
                pw += (lb1[x1] + lb2[t - x1]).exp();
            }
        }
    }
    pw
}

fn ln_comb(n: usize, k: usize) -> f64 {
    dist::lnfactorial(n as f64) - dist::lnfactorial(k as f64) - dist::lnfactorial((n - k) as f64)
}

fn ln_binom_pmf(n: usize, k: usize, p: f64) -> f64 {
    let lp = if k == 0 { 0.0 } else { k as f64 * p.ln() };
    let lq = if k == n { 0.0 } else { (n - k) as f64 * (1.0 - p).ln() };
    ln_comb(n, k) + lp + lq
}

/// Power of the chi-squared test of one variance (variance ratio `rho`).
fn var_power(df: f64, rho: f64, a: f64, onesided: bool) -> f64 {
    if !(df > 0.0) {
        return f64::NAN;
    }
    if onesided {
        if rho >= 1.0 {
            dist::chi2tail(df, dist::invchi2tail(df, a) / rho)
        } else {
            dist::chi2(df, dist::invchi2(df, a) / rho)
        }
    } else {
        dist::chi2tail(df, dist::invchi2tail(df, a / 2.0) / rho) + dist::chi2(df, dist::invchi2(df, a / 2.0) / rho)
    }
}

#[allow(dead_code)]
pub(crate) fn zc(r: &Row) -> f64 {
    zcrit(g(r, "alpha"), os(r))
}
