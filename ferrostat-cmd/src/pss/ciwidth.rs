//! Precision analysis: `ciwidth onemean | pairedmeans | twomeans |
//! onevariance`. Solves for the sample size, the CI width or the
//! probability that the CI width does not exceed a target (`probwidth`).
//!
//! With an estimated standard deviation the width is random:
//! W = c·s with s² ~ σ²·χ²(df)/df, so Pr(W ≤ ω) = F_χ²(df)(df·(ω/(cσ))²).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::methods::{item, push, push_flag, Lists};
use super::report::{report, Report, Solve};
use super::{flag, g, grid, has, opt_list, positional, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::store::Store;

pub struct Ciwidth;

#[derive(Clone, Copy, PartialEq)]
enum M {
    OneMean,
    Paired,
    TwoMeans,
    OneVar,
}

/// Width factor c (W = c·s) and the degrees of freedom of s.
fn factor(m: M, r: &Row, n1: f64, n2: f64) -> (f64, f64) {
    let a = g(r, "alpha");
    let os = flag(r, "_onesided");
    let tail = if os { a } else { a / 2.0 };
    let mult = if os { 1.0 } else { 2.0 };
    let known = flag(r, "_z");
    match m {
        M::OneMean | M::Paired => {
            let df = n1 - 1.0;
            let q = if known { dist::invnormal(1.0 - tail) } else { dist::invttail(df, tail) };
            (mult * q / n1.sqrt(), if known { f64::INFINITY } else { df })
        }
        M::TwoMeans => {
            let (s1, s2) = (sd_of(r, "sd1"), sd_of(r, "sd2"));
            if (s1 - s2).abs() < 1e-12 {
                let df = n1 + n2 - 2.0;
                let q = if known { dist::invnormal(1.0 - tail) } else { dist::invttail(df, tail) };
                (mult * q * (1.0 / n1 + 1.0 / n2).sqrt(), if known { f64::INFINITY } else { df })
            } else {
                // Satterthwaite: express W relative to sqrt(s1²/n1 + s2²/n2) at the true sds
                let (u, v) = (s1 * s1 / n1, s2 * s2 / n2);
                let df = (u + v).powi(2) / (u * u / (n1 - 1.0) + v * v / (n2 - 1.0));
                let q = if known { dist::invnormal(1.0 - tail) } else { dist::invttail(df, tail) };
                (mult * q * (u + v).sqrt() / s1, if known { f64::INFINITY } else { df })
            }
        }
        M::OneVar => {
            let df = n1 - 1.0;
            if os {
                // upper one-sided bound minus the estimate, per unit s²
                (df / dist::invchi2(df, a) - 1.0, df)
            } else {
                (df * (1.0 / dist::invchi2(df, a / 2.0) - 1.0 / dist::invchi2tail(df, a / 2.0)), df)
            }
        }
    }
}

fn sd_of(r: &Row, k: &str) -> f64 {
    if has(r, k) {
        g(r, k)
    } else if has(r, "sd") {
        g(r, "sd")
    } else {
        1.0
    }
}

/// The population scale of the estimated quantity (σ, or σ² for variances).
fn scale(m: M, r: &Row) -> f64 {
    match m {
        M::OneMean => sd_of(r, "sd"),
        M::Paired => g(r, "sd_d"),
        M::TwoMeans => sd_of(r, "sd1"),
        M::OneVar => g(r, "variance"),
    }
}

/// Pr(W ≤ ω).
fn prob(m: M, r: &Row, n1: f64, n2: f64, w: f64) -> f64 {
    let (c, df) = factor(m, r, n1, n2);
    if !(df > 0.0) || !c.is_finite() {
        return 0.0;
    }
    let sig = scale(m, r);
    if df.is_infinite() {
        return if c * sig <= w { 1.0 } else { 0.0 };
    }
    let x = if m == M::OneVar { df * w / (c * sig) } else { df * (w / (c * sig)).powi(2) };
    dist::chi2(df, x)
}

/// Width at probability `p` (None: the width at s = σ).
fn width(m: M, r: &Row, n1: f64, n2: f64, p: Option<f64>) -> f64 {
    let (c, df) = factor(m, r, n1, n2);
    let sig = scale(m, r);
    match p {
        Some(p) if df.is_finite() => {
            let q = dist::invchi2(df, p) / df;
            if m == M::OneVar {
                c * sig * q
            } else {
                c * sig * q.sqrt()
            }
        }
        _ => c * sig,
    }
}

fn sizes(two: bool, r: &Row) -> (f64, f64) {
    if two {
        (g(r, "N1"), g(r, "N2"))
    } else {
        (g(r, "N"), 0.0)
    }
}

fn set_size(two: bool, r: &mut Row, m: f64) {
    if two {
        let ratio = if has(r, "nratio") { g(r, "nratio") } else { 1.0 };
        let frac = flag(r, "_nfrac");
        r.insert("N1".into(), m);
        r.insert("N2".into(), if frac { m * ratio } else { (m * ratio).ceil() });
        r.insert("N".into(), m + g(r, "N2"));
    } else {
        r.insert("N".into(), m);
    }
}

/// Does size `m` meet the precision requirement?
fn meets(m_: M, two: bool, r: &Row, m: f64) -> bool {
    let mut rr = r.clone();
    set_size(two, &mut rr, m);
    let (a, b) = sizes(two, &rr);
    if flag(r, "_probgiven") {
        prob(m_, &rr, a, b, g(r, "width")) >= g(r, "Pr_width")
    } else {
        width(m_, &rr, a, b, None) <= g(r, "width")
    }
}

fn solve_n(m_: M, two: bool, r: &mut Row) -> Result<(), String> {
    let frac = flag(r, "_nfrac");
    let mut lo = 2.0;
    if meets(m_, two, r, lo) && !frac {
        set_size(two, r, lo);
        return Ok(());
    }
    let mut hi = 4.0;
    while !meets(m_, two, r, hi) {
        lo = hi;
        hi *= 2.0;
        if hi > 1e10 {
            return Err(rc_err(498, "sample size could not be computed"));
        }
    }
    if frac {
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if meets(m_, two, r, mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    } else {
        while hi - lo > 1.0 {
            let mid = ((lo + hi) / 2.0).floor();
            if meets(m_, two, r, mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }
    set_size(two, r, hi);
    Ok(())
}

impl CommandHandler for Ciwidth {
    fn name(&self) -> &'static str {
        "ciwidth"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub_l = sub.to_lowercase();
        let m = match sub_l.as_str() {
            s if s.len() >= 4 && "onemean".starts_with(s) => M::OneMean,
            s if s.len() >= 7 && "pairedmeans".starts_with(s) => M::Paired,
            s if s.len() >= 4 && "twomeans".starts_with(s) => M::TwoMeans,
            s if s.len() >= 6 && "onevariance".starts_with(s) => M::OneVar,
            "" => return Err(rc_err(198, "ciwidth: method required")),
            other => return Err(rc_err(199, format!("ciwidth method {other} not found"))),
        };
        let pos = positional(rest)?;
        let o = |n: &str, k: usize| opt_list(cmd, n, k);
        let mut l: Lists = Vec::new();
        let two = m == M::TwoMeans;
        // confidence level: level() or alpha()
        let alpha = match o("level", 1)? {
            Some(lv) => lv.iter().map(|v| 1.0 - v / 100.0).collect(),
            None => o("alpha", 1)?.unwrap_or(vec![0.05]),
        };
        l.push(("alpha".into(), alpha));
        push(&mut l, "N", super::list_exact(cmd, "n")?);
        if two {
            push(&mut l, "N1", super::list_exact(cmd, "n1")?);
            push(&mut l, "N2", super::list_exact(cmd, "n2")?);
            push(&mut l, "nratio", o("nratio", 6)?);
        }
        push(&mut l, "width", o("width", 5)?);
        let pw = o("probwidth", 5)?;
        if pw.is_some() {
            l.push(("_probgiven".into(), vec![1.0]));
        }
        push(&mut l, "Pr_width", pw);
        push_flag(&mut l, "_onesided", cmd.has_opt("onesided", 3) || cmd.has_opt("lower", 5) || cmd.has_opt("upper", 5));
        push_flag(&mut l, "_nfrac", cmd.has_opt("nfractional", 5));
        let known = cmd.has_opt("knownsd", 6) || cmd.has_opt("knownsds", 6);
        push_flag(&mut l, "_z", known && m != M::OneVar);
        let (title, show): (&str, Vec<&str>) = match m {
            M::OneMean => {
                push(&mut l, "sd", Some(o("sd", 2)?.or(item(&pos, 0)).unwrap_or(vec![1.0])));
                ("one-sample mean CI", vec!["sd"])
            }
            M::Paired => {
                push(&mut l, "sd_d", o("sddiff", 4)?);
                push(&mut l, "sd", o("sd", 2)?);
                push(&mut l, "sd1", o("sd1", 3)?);
                push(&mut l, "sd2", o("sd2", 3)?);
                push(&mut l, "corr", o("corr", 4)?);
                ("paired-means-difference CI", vec!["sd_d", "corr", "sd1", "sd2"])
            }
            M::TwoMeans => {
                push(&mut l, "sd", o("sd", 2)?);
                push(&mut l, "sd1", o("sd1", 3)?);
                push(&mut l, "sd2", o("sd2", 3)?);
                ("two-sample means-difference CI", vec!["sd", "sd1", "sd2"])
            }
            M::OneVar => {
                let v = o("variance", 3)?.or(o("sd", 2)?.map(|s| s.iter().map(|x| x * x).collect())).or(item(&pos, 0));
                push(&mut l, "variance", Some(v.unwrap_or(vec![1.0])));
                ("one-sample variance CI", vec!["variance"])
            }
        };
        let mut rows = grid(&l, cmd.has_opt("parallel", 3))?;
        let mut solve = Solve::Width;
        for r in rows.iter_mut() {
            if m == M::Paired && !has(r, "sd_d") {
                let s = if has(r, "sd") { g(r, "sd") } else { 1.0 };
                let s1 = if has(r, "sd1") { g(r, "sd1") } else { s };
                let s2 = if has(r, "sd2") { g(r, "sd2") } else { s };
                if !has(r, "corr") {
                    return Err(rc_err(198, "ciwidth pairedmeans: sddiff() or corr() required"));
                }
                let c = g(r, "corr");
                r.insert("sd_d".into(), (s1 * s1 + s2 * s2 - 2.0 * c * s1 * s2).sqrt());
            }
            let have_n = if two { has(r, "N") || has(r, "N1") || has(r, "N2") } else { has(r, "N") };
            if two && have_n {
                let ratio = if has(r, "nratio") { g(r, "nratio") } else { 1.0 };
                if has(r, "N1") && !has(r, "N2") {
                    r.insert("N2".into(), (g(r, "N1") * ratio).ceil());
                } else if has(r, "N2") && !has(r, "N1") {
                    r.insert("N1".into(), (g(r, "N2") / ratio).ceil());
                } else if !has(r, "N1") {
                    let n1 = (g(r, "N") / (1.0 + ratio)).round();
                    r.insert("N1".into(), n1);
                    r.insert("N2".into(), g(r, "N") - n1);
                }
                r.insert("N".into(), g(r, "N1") + g(r, "N2"));
            }
            if !have_n {
                if !has(r, "width") {
                    return Err(rc_err(198, "width() required to compute the sample size"));
                }
                solve = Solve::N;
                solve_n(m, two, r)?;
            } else if has(r, "width") {
                solve = Solve::ProbWidth;
            } else {
                solve = Solve::Width;
                let (a, b) = sizes(two, r);
                let p = if flag(r, "_probgiven") { Some(g(r, "Pr_width")) } else { None };
                r.insert("width".into(), width(m, r, a, b, p));
            }
            let (a, b) = sizes(two, r);
            if two {
                r.insert("nratio".into(), b / a);
            }
            let p = prob(m, r, a, b, g(r, "width"));
            if !(solve == Solve::Width && flag(r, "_probgiven")) {
                r.insert("Pr_width".into(), p);
            }
        }
        let rep = Report {
            kind: "ciwidth".into(),
            method: sub_l.clone(),
            title: format!("a {title}"),
            test: if known { "Normal-based CI (known standard deviation)".into() } else { String::new() },
            hyp: String::new(),
            solve,
            target: None,
            tword: String::new(),
            two,
            cluster: false,
            show: show.iter().map(|s| s.to_string()).collect(),
            extra: vec![],
            rows,
        };
        report(st, cmd, rep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_sd_width() {
        let mut r = Row::new();
        r.insert("alpha".into(), 0.05);
        r.insert("sd".into(), 2.0);
        r.insert("_z".into(), 1.0);
        let w = width(M::OneMean, &r, 100.0, 0.0, None);
        assert!((w - 2.0 * 1.959964 * 2.0 / 10.0).abs() < 1e-5);
    }
}
