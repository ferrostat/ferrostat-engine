//! Power and sample-size analysis: `power onemean`, `power twomeans`,
//! `power oneproportion`, `power twoproportions`; solving for sample size,
//! power or effect size.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("power", Box::new(Power))]
}

pub struct Power;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    OneMean,
    TwoMeans,
    OneProp,
    TwoProps,
}

struct Spec {
    kind: Kind,
    a: f64,
    sd1: f64,
    sd2: f64,
    alpha: f64,
    onesided: bool,
    ratio: f64,
    knownsd: bool,
}

/// Two-sided (or one-sided) power of a t test with noncentrality `ncp`.
fn t_power(df: f64, ncp: f64, alpha: f64, onesided: bool) -> f64 {
    if !df.is_finite() || df > 1e6 {
        return z_power(ncp, alpha, onesided);
    }
    if onesided {
        let c = dist::invttail(df, alpha);
        let s = ncp.signum();
        1.0 - crate::smallstat::nct_cdf(c, df, ncp * s)
    } else {
        let c = dist::invttail(df, alpha / 2.0);
        1.0 - crate::smallstat::nct_cdf(c, df, ncp) + crate::smallstat::nct_cdf(-c, df, ncp)
    }
}

fn z_power(ncp: f64, alpha: f64, onesided: bool) -> f64 {
    if onesided {
        dist::normal(ncp.abs() - dist::invnormal(1.0 - alpha))
    } else {
        let z = dist::invnormal(1.0 - alpha / 2.0);
        dist::normal(ncp - z) + dist::normal(-ncp - z)
    }
}

/// Power at total sample size `n` (two-sample designs split by `ratio`).
fn power_at(s: &Spec, b: f64, n: f64) -> f64 {
    match s.kind {
        Kind::OneMean => {
            let ncp = (b - s.a) * n.sqrt() / s.sd1;
            if s.knownsd { z_power(ncp, s.alpha, s.onesided) } else { t_power(n - 1.0, ncp, s.alpha, s.onesided) }
        }
        Kind::TwoMeans => {
            let n1 = n / (1.0 + s.ratio);
            let n2 = n - n1;
            let se = (s.sd1 * s.sd1 / n1 + s.sd2 * s.sd2 / n2).sqrt();
            let ncp = (b - s.a) / se;
            if s.knownsd {
                z_power(ncp, s.alpha, s.onesided)
            } else {
                let df = if (s.sd1 - s.sd2).abs() < 1e-12 {
                    n - 2.0
                } else {
                    let (u, v) = (s.sd1 * s.sd1 / n1, s.sd2 * s.sd2 / n2);
                    (u + v).powi(2) / (u * u / (n1 - 1.0) + v * v / (n2 - 1.0))
                };
                t_power(df, ncp, s.alpha, s.onesided)
            }
        }
        Kind::OneProp => {
            let (p0, pa) = (s.a, b);
            let z = if s.onesided { dist::invnormal(1.0 - s.alpha) } else { dist::invnormal(1.0 - s.alpha / 2.0) };
            let s0 = (p0 * (1.0 - p0)).sqrt();
            let sa = (pa * (1.0 - pa)).sqrt();
            let d = (pa - p0) * n.sqrt();
            if s.onesided {
                dist::normal((d.abs() - z * s0) / sa)
            } else {
                dist::normal((d - z * s0) / sa) + dist::normal((-d - z * s0) / sa)
            }
        }
        Kind::TwoProps => {
            let (p1, p2) = (s.a, b);
            let n1 = n / (1.0 + s.ratio);
            let n2 = n - n1;
            let pbar = (n1 * p1 + n2 * p2) / n;
            let z = if s.onesided { dist::invnormal(1.0 - s.alpha) } else { dist::invnormal(1.0 - s.alpha / 2.0) };
            let s0 = (pbar * (1.0 - pbar) * (1.0 / n1 + 1.0 / n2)).sqrt();
            let sa = (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2).sqrt();
            let d = p2 - p1;
            if s.onesided {
                dist::normal((d.abs() - z * s0) / sa)
            } else {
                dist::normal((d - z * s0) / sa) + dist::normal((-d - z * s0) / sa)
            }
        }
    }
}

/// Smallest total sample size (two-sample: each group an integer) with
/// power at least `target`.
fn solve_n(s: &Spec, b: f64, target: f64) -> f64 {
    let two = matches!(s.kind, Kind::TwoMeans | Kind::TwoProps);
    let total = |k: f64| if two { k + (k * s.ratio).ceil() } else { k };
    let mut lo = 2.0;
    let mut hi = 4.0;
    while power_at(s, b, total(hi)) < target && hi < 1e9 {
        lo = hi;
        hi *= 2.0;
    }
    while hi - lo > 1.0 {
        let mid = ((lo + hi) / 2.0).floor();
        if power_at(s, b, total(mid)) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let k = if power_at(s, b, total(lo)) >= target { lo } else { hi };
    total(k)
}

/// Alternative value with the target power (searched beyond the null).
fn solve_effect(s: &Spec, n: f64, target: f64, upper: bool) -> f64 {
    let (mut lo, mut hi) = match s.kind {
        Kind::OneProp | Kind::TwoProps => {
            if upper { (s.a, 1.0 - 1e-10) } else { (1e-10, s.a) }
        }
        _ => {
            let sd = s.sd1.max(s.sd2);
            if upper { (s.a, s.a + 100.0 * sd) } else { (s.a - 100.0 * sd, s.a) }
        }
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let p = power_at(s, mid, n);
        let too_close = p < target;
        if upper == too_close {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn num_list(t: &str) -> Vec<f64> {
    t.split_whitespace().filter_map(|w| w.parse::<f64>().ok()).collect()
}

impl CommandHandler for Power {
    fn name(&self) -> &'static str {
        "power"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let kind = match sub.to_lowercase().as_str() {
            "onemean" => Kind::OneMean,
            "twomeans" => Kind::TwoMeans,
            "oneproportion" | "oneprop" => Kind::OneProp,
            "twoproportions" | "twoprop" => Kind::TwoProps,
            other => return Err(rc_err(198, format!("power {other}: method not available"))),
        };
        let nums = num_list(rest);
        let a = *nums.first().ok_or_else(|| rc_err(198, "null value required"))?;
        let opt = |n: &str, m: usize| util::opt_num(cmd, n, m);
        let sd = opt("sd", 2)?.unwrap_or(1.0);
        let mut b = nums.get(1).copied();
        if b.is_none() {
            if let Some(d) = opt("diff", 3)? {
                b = Some(a + d);
            }
        }
        let spec = Spec {
            kind,
            a,
            sd1: opt("sd1", 3)?.unwrap_or(sd),
            sd2: opt("sd2", 3)?.unwrap_or(sd),
            alpha: opt("alpha", 1)?.unwrap_or(0.05),
            onesided: cmd.has_opt("onesided", 3),
            ratio: opt("nratio", 3)?.unwrap_or(1.0),
            knownsd: cmd.has_opt("knownsds", 6) || cmd.has_opt("knownsd", 6),
        };
        let n_opt = opt("n", 1)?.or(match (opt("n1", 2)?, opt("n2", 2)?) {
            (Some(x), Some(y)) => Some(x + y),
            _ => None,
        });
        let pw_opt = opt("power", 1)?;
        let beta_opt = opt("beta", 1)?;
        let target = pw_opt.or(beta_opt.map(|x| 1.0 - x)).unwrap_or(0.8);
        let (title_kind, test) = match kind {
            Kind::OneMean => ("a one-sample mean test", if spec.knownsd { "z test" } else { "t test" }),
            Kind::TwoMeans => ("a two-sample means test", if spec.knownsd { "z test" } else { "t test assuming sd1 = sd2 = sd" }),
            Kind::OneProp => ("a one-sample proportion test", "Score z test"),
            Kind::TwoProps => ("a two-sample proportions test", "Pearson's chi-squared test"),
        };
        let (h0, ha) = match kind {
            Kind::OneMean => ("m = m0", "m != m0"),
            Kind::TwoMeans => ("m2 = m1", "m2 != m1"),
            Kind::OneProp => ("p = p0", "p != p0"),
            Kind::TwoProps => ("p2 = p1", "p2 != p1"),
        };
        let mut r = Results::new();
        let (what, n, pw, bval) = match (n_opt, b) {
            (Some(n), Some(bv)) => ("power", n, power_at(&spec, bv, n), bv),
            (None, Some(bv)) => {
                let n = solve_n(&spec, bv, target);
                ("sample size", n, power_at(&spec, bv, n), bv)
            }
            (Some(n), None) => {
                let upper = !cmd.has_opt("direction(lower)", 10) && cmd.opt("direction", 3).map_or(true, |d| !d.trim().starts_with('l'));
                let bv = solve_effect(&spec, n, target, upper);
                ("target", n, power_at(&spec, bv, n), bv)
            }
            (None, None) => return Err(rc_err(198, "alternative value or n() required")),
        };
        let delta = match kind {
            Kind::OneMean => (bval - a) / spec.sd1,
            Kind::TwoMeans => bval - a,
            _ => bval - a,
        };
        let mut out = vec![String::new()];
        out.push(match what {
            "power" => format!("Estimated power for {title_kind}"),
            "sample size" => format!("Estimated sample size{} for {title_kind}", if matches!(kind, Kind::TwoMeans | Kind::TwoProps) { "s" } else { "" }),
            _ => format!("Estimated target {} for {title_kind}", if matches!(kind, Kind::OneProp | Kind::TwoProps) { "proportion" } else { "mean" }),
        });
        out.push(test.to_string());
        out.push(format!("H0: {h0}  versus  Ha: {ha}"));
        out.push(String::new());
        out.push("Study parameters:".into());
        out.push(String::new());
        let line = |l: &str, v: f64| format!("{:>13} = {:>9.4}", l, v);
        out.push(line("alpha", spec.alpha));
        if what != "power" {
            out.push(line("power", if what == "target" { target } else { target }));
        }
        if what != "sample size" {
            match kind {
                Kind::TwoMeans | Kind::TwoProps => {
                    let n1 = (n / (1.0 + spec.ratio)).round();
                    out.push(format!("{:>13} = {:>9}", "N", n));
                    out.push(format!("{:>13} = {:>9}", "N per group", n1));
                }
                _ => out.push(format!("{:>13} = {:>9}", "N", n)),
            }
        }
        out.push(line("delta", delta));
        match kind {
            Kind::OneMean => {
                out.push(line("m0", a));
                out.push(line("ma", bval));
                out.push(line("sd", spec.sd1));
            }
            Kind::TwoMeans => {
                out.push(line("m1", a));
                out.push(line("m2", bval));
                out.push(line("sd", spec.sd1));
            }
            Kind::OneProp => {
                out.push(line("p0", a));
                out.push(line("pa", bval));
            }
            Kind::TwoProps => {
                out.push(line("p1", a));
                out.push(line("p2", bval));
            }
        }
        out.push(String::new());
        match what {
            "power" => {
                out.push("Estimated power:".into());
                out.push(String::new());
                out.push(line("power", pw));
            }
            "sample size" => {
                out.push(format!("Estimated sample size{}:", if matches!(kind, Kind::TwoMeans | Kind::TwoProps) { "s" } else { "" }));
                out.push(String::new());
                match kind {
                    Kind::TwoMeans | Kind::TwoProps => {
                        let n1 = (n / (1.0 + spec.ratio)).round();
                        out.push(format!("{:>13} = {:>9}", "N", n));
                        out.push(format!("{:>13} = {:>9}", "N per group", n1));
                        r.set("N1", n1);
                        r.set("N2", n - n1);
                    }
                    _ => out.push(format!("{:>13} = {:>9}", "N", n)),
                }
            }
            _ => {
                out.push(format!("Estimated target {}:", if matches!(kind, Kind::OneProp | Kind::TwoProps) { "proportion" } else { "mean" }));
                out.push(String::new());
                out.push(line(match kind {
                    Kind::OneMean => "ma",
                    Kind::TwoMeans => "m2",
                    Kind::OneProp => "pa",
                    Kind::TwoProps => "p2",
                }, bval));
            }
        }
        r.set("N", n);
        r.set("power", pw);
        r.set("alpha", spec.alpha);
        r.set("delta", delta);
        r.set("onesided", if spec.onesided { 1.0 } else { 0.0 });
        match kind {
            Kind::OneMean => {
                r.set("m0", a);
                r.set("ma", bval);
                r.set("sd", spec.sd1);
            }
            Kind::TwoMeans => {
                r.set("m1", a);
                r.set("m2", bval);
                r.set("sd", spec.sd1);
            }
            Kind::OneProp => {
                r.set("p0", a);
                r.set("pa", bval);
            }
            Kind::TwoProps => {
                r.set("p1", a);
                r.set("p2", bval);
            }
        }
        r.set_str("type", "test");
        r.set_str("method", sub.to_lowercase());
        st.r = r;
        Ok(out.join("\n"))
    }
}
