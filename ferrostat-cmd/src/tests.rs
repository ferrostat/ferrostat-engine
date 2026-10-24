//! Hypothesis tests: ttest, prtest, sdtest, ranksum, signrank, signtest,
//! kwallis, oneway, ksmirnov, swilk.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, parse_expression};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::correlate::ranks;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;
use crate::varlist;

fn g9(v: f64) -> String {
    fmt_g(v, 9)
}

/// `a == b` in the main text: returns (lhs, rhs text).
fn split_eq(main: &str) -> Option<(String, String)> {
    let i = find_top_level(main, '=')?;
    let lhs = main[..i].trim().to_string();
    let rhs = main[i..].trim_start_matches('=').trim().to_string();
    Some((lhs, rhs))
}

fn values(ds: &DataSet, v: &str, mask: &[bool]) -> Result<Vec<f64>, String> {
    let c = util::col(ds, v)?;
    Ok((0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect())
}

/// Two groups defined by `by(var)`: values of `v` for group 1 and 2.
fn by_groups(ds: &DataSet, v: &str, by: &str, mask: &[bool]) -> Result<(Vec<f64>, Vec<f64>, String, String), String> {
    let g = ds.get_var(by).ok_or_else(|| rc_err(111, format!("variable {by} not found")))?;
    let mut m = mask.to_vec();
    util::markout(ds, &mut m, &[v.to_string(), by.to_string()]);
    let (ids, reps) = util::group_ids(ds, &[by.to_string()], &m);
    if reps.len() != 2 {
        return Err(rc_err(420, format!("more than 2 groups found, only 2 allowed")));
    }
    let x = util::col(ds, v)?;
    let a: Vec<f64> = (0..ds.n_obs).filter(|&i| ids[i] == 0).map(|i| x[i]).collect();
    let b: Vec<f64> = (0..ds.n_obs).filter(|&i| ids[i] == 1).map(|i| x[i]).collect();
    Ok((a, b, util::cell_text(ds, g, reps[0], true), util::cell_text(ds, g, reps[1], true)))
}

struct Group {
    name: String,
    n: f64,
    mean: f64,
    sd: f64,
}

impl Group {
    fn of(name: &str, x: &[f64]) -> Group {
        let n = x.len() as f64;
        let mean = if n > 0.0 { util::mean(x) } else { SYSMISS };
        let sd = if n > 1.0 { util::variance(x).sqrt() } else { SYSMISS };
        Group { name: name.into(), n, mean, sd }
    }
    fn se(&self) -> f64 {
        self.sd / self.n.sqrt()
    }
}

fn ttable(title: &str, groups: &[Group], extra: &[(String, f64, f64, f64)], level: f64) -> Vec<String> {
    let mut out = vec![
        title.to_string(),
        "------------------------------------------------------------------------------".to_string(),
        format!("{:>8} |     Obs        Mean    Std. err.   Std. dev.   [{}% conf. interval]", if groups.len() > 1 && groups[0].name.parse::<f64>().is_ok() { "Group" } else { "Variable" }, fmt_g(level, 4)),
        "---------+--------------------------------------------------------------------".to_string(),
    ];
    let a = (1.0 - level / 100.0) / 2.0;
    for g in groups {
        let t = dist::invttail(g.n - 1.0, a);
        out.push(format!(
            "{:>8} | {:>7} {:>11} {:>11} {:>11} {:>11} {:>11}",
            abbrev(&g.name, 8),
            g9(g.n),
            g9(g.mean),
            g9(g.se()),
            g9(g.sd),
            g9(g.mean - t * g.se()),
            g9(g.mean + t * g.se())
        ));
    }
    if !extra.is_empty() {
        out.push("---------+--------------------------------------------------------------------".to_string());
        for (name, est, se, df) in extra {
            let t = dist::invttail(*df, a);
            out.push(format!("{:>8} |         {:>11} {:>11}             {:>11} {:>11}", name, g9(*est), g9(*se), g9(est - t * se), g9(est + t * se)));
        }
    }
    out.push("------------------------------------------------------------------------------".to_string());
    out
}

fn tails(t: f64, df: f64, diffname: &str, h0: &str, r: &mut Results) -> Vec<String> {
    let p_u = dist::ttail(df, t);
    let p_l = 1.0 - p_u;
    let p = 2.0 * dist::ttail(df, t.abs());
    r.set("t", t);
    r.set("df_t", df);
    r.set("p", p);
    r.set("p_l", p_l);
    r.set("p_u", p_u);
    vec![
        format!("    {:<50}t = {:>9.4}", diffname, t),
        format!("H0: {:<34}Degrees of freedom = {:>8}", h0, fmt_g((df * 10000.0).round() / 10000.0, 8)),
        String::new(),
        format!("    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0"),
        format!(" Pr(T < t) = {:.4}         Pr(|T| > |t|) = {:.4}          Pr(T > t) = {:.4}", p_l, p, p_u),
    ]
}

pub struct Ttest;

impl CommandHandler for Ttest {
    fn name(&self) -> &'static str {
        "ttest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let level = util::level(cmd, st)?;
        let mut r = Results::new();
        r.set("level", level);
        let mut out;
        if let Some(by) = cmd.opt("by", 2) {
            let v = varlist::resolve_name(ds, cmd.main.trim())?;
            let by = varlist::resolve_name(ds, by.trim())?;
            let (a, b, na, nb) = by_groups(ds, &v, &by, &mask)?;
            out = two_sample(&a, &b, &na, &nb, cmd, level, &mut r, &format!("mean({}) - mean({})", na, nb))?;
        } else {
            let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax; use ttest var == #, var1 == var2, or var, by()"))?;
            let v1 = varlist::resolve_name(ds, &lhs)?;
            if let Ok(v2) = varlist::resolve_name(ds, &rhs) {
                if cmd.has_opt("unpaired", 3) {
                    let a = values(ds, &v1, &mask)?;
                    let b = values(ds, &v2, &mask)?;
                    out = two_sample(&a, &b, &v1, &v2, cmd, level, &mut r, &format!("mean({v1}) - mean({v2})"))?;
                } else {
                    // paired
                    let mut m = mask.clone();
                    util::markout(ds, &mut m, &[v1.clone(), v2.clone()]);
                    let a = values(ds, &v1, &m)?;
                    let b = values(ds, &v2, &m)?;
                    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
                    let ga = Group::of(&v1, &a);
                    let gb = Group::of(&v2, &b);
                    let gd = Group::of("diff", &d);
                    let df = gd.n - 1.0;
                    out = ttable("Paired t test", &[ga, gb], &[], level);
                    out.pop();
                    out.push("---------+--------------------------------------------------------------------".into());
                    let t = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
                    out.push(format!("{:>8} | {:>7} {:>11} {:>11} {:>11} {:>11} {:>11}", "diff", g9(gd.n), g9(gd.mean), g9(gd.se()), g9(gd.sd), g9(gd.mean - t * gd.se()), g9(gd.mean + t * gd.se())));
                    out.push("------------------------------------------------------------------------------".into());
                    let tstat = gd.mean / gd.se();
                    out.extend(tails(tstat, df, &format!("mean(diff) = mean({v1} - {v2})"), "mean(diff) = 0", &mut r));
                    r.set("N_1", gd.n);
                    r.set("N_2", gd.n);
                    r.set("mu_1", util::mean(&a));
                    r.set("mu_2", util::mean(&b));
                    r.set("sd_1", util::variance(&a).sqrt());
                    r.set("sd_2", util::variance(&b).sqrt());
                    r.set("sd", gd.sd);
                    r.set("se", gd.se());
                    r.set("mu_diff", gd.mean);
                }
            } else {
                let e = parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
                let mu0 = Env::with_store(ds, st).eval(&e, 0)?.num()?;
                let a = values(ds, &v1, &mask)?;
                let g = Group::of(&v1, &a);
                let df = g.n - 1.0;
                out = ttable("One-sample t test", &[Group::of(&v1, &a)], &[], level);
                let t = (g.mean - mu0) / g.se();
                let mut tl = tails(t, df, &format!("mean = mean({v1})"), &format!("mean = {}", fmt_g(mu0, 9)), &mut r);
                tl[3] = format!("    Ha: mean < {0:<18}Ha: mean != {0:<18}Ha: mean > {0}", fmt_g(mu0, 9));
                out.extend(tl);
                r.set("N_1", g.n);
                r.set("mu_1", g.mean);
                r.set("sd_1", g.sd);
                r.set("sd", g.sd);
                r.set("se", g.se());
            }
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn two_sample(a: &[f64], b: &[f64], na: &str, nb: &str, cmd: &Command, level: f64, r: &mut Results, diffname: &str) -> Result<Vec<String>, String> {
    let ga = Group::of(na, a);
    let gb = Group::of(nb, b);
    let mut all = a.to_vec();
    all.extend_from_slice(b);
    let gc = Group::of("combined", &all);
    let unequal = cmd.has_opt("unequal", 3) || cmd.has_opt("welch", 1);
    let diff = ga.mean - gb.mean;
    let (se, df) = if unequal {
        let (va, vb) = (ga.sd.powi(2) / ga.n, gb.sd.powi(2) / gb.n);
        let se = (va + vb).sqrt();
        let df = if cmd.has_opt("welch", 1) {
            -2.0 + (va + vb).powi(2) / (va * va / (ga.n + 1.0) + vb * vb / (gb.n + 1.0))
        } else {
            (va + vb).powi(2) / (va * va / (ga.n - 1.0) + vb * vb / (gb.n - 1.0))
        };
        (se, df)
    } else {
        let sp = ((ga.n - 1.0) * ga.sd.powi(2) + (gb.n - 1.0) * gb.sd.powi(2)) / (ga.n + gb.n - 2.0);
        ((sp * (1.0 / ga.n + 1.0 / gb.n)).sqrt(), ga.n + gb.n - 2.0)
    };
    let title = if unequal { "Two-sample t test with unequal variances" } else { "Two-sample t test with equal variances" };
    let mut out = ttable(title, &[ga, gb, gc], &[("diff".into(), diff, se, df)], level);
    let t = diff / se;
    out.extend(tails(t, df, &format!("diff = {diffname}"), "diff = 0", r));
    r.set("N_1", a.len() as f64);
    r.set("N_2", b.len() as f64);
    r.set("mu_1", util::mean(a));
    r.set("mu_2", util::mean(b));
    r.set("sd_1", util::variance(a).sqrt());
    r.set("sd_2", util::variance(b).sqrt());
    r.set("se", se);
    Ok(out)
}

pub struct Prtest;

impl CommandHandler for Prtest {
    fn name(&self) -> &'static str {
        "prtest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let level = util::level(cmd, st)?;
        let zc = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mut r = Results::new();
        let mut out = Vec::new();
        let header = |out: &mut Vec<String>| {
            out.push(format!("{:>12} |       Mean   Std. err.                     [{}% conf. interval]", "Variable", fmt_g(level, 4)));
            out.push("-------------+----------------------------------------------------------------".into());
        };
        let line = |name: &str, p: f64, n: f64| format!("{:>12} | {:>10} {:>10}                      {:>10} {:>10}", abbrev(name, 12), g9(p), g9((p * (1.0 - p) / n).sqrt()), g9(p - zc * (p * (1.0 - p) / n).sqrt()), g9(p + zc * (p * (1.0 - p) / n).sqrt()));
        let finish = |z: f64, r: &mut Results, out: &mut Vec<String>| {
            let p = 2.0 * (1.0 - dist::normal(z.abs()));
            r.set("z", z);
            r.set("p", p);
            out.push(format!("{:>60}z = {:>8.4}", "", z));
            out.push(format!(" Pr(Z < z) = {:.4}         Pr(|Z| > |z|) = {:.4}          Pr(Z > z) = {:.4}", dist::normal(z), p, 1.0 - dist::normal(z)));
        };
        let two = |a: &[f64], b: &[f64], na: &str, nb: &str, r: &mut Results, out: &mut Vec<String>| {
            let (n1, n2) = (a.len() as f64, b.len() as f64);
            let (p1, p2) = (util::mean(a), util::mean(b));
            let pp = (p1 * n1 + p2 * n2) / (n1 + n2);
            let z = (p1 - p2) / (pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n2)).sqrt();
            out.push(format!("Two-sample test of proportions{:>30}{}: Number of obs = {}", "", na, n1));
            out.push(format!("{:>60}{}: Number of obs = {}", "", nb, n2));
            header(out);
            out.push(line(na, p1, n1));
            out.push(line(nb, p2, n2));
            r.set("P_1", p1);
            r.set("P_2", p2);
            r.set("N_1", n1);
            r.set("N_2", n2);
            finish(z, r, out);
        };
        if let Some(by) = cmd.opt("by", 2) {
            let v = varlist::resolve_name(ds, cmd.main.trim())?;
            let by = varlist::resolve_name(ds, by.trim())?;
            let (a, b, na, nb) = by_groups(ds, &v, &by, &mask)?;
            two(&a, &b, &na, &nb, &mut r, &mut out);
        } else {
            let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax"))?;
            let v1 = varlist::resolve_name(ds, &lhs)?;
            if let Ok(v2) = varlist::resolve_name(ds, &rhs) {
                let a = values(ds, &v1, &mask)?;
                let b = values(ds, &v2, &mask)?;
                two(&a, &b, &v1, &v2, &mut r, &mut out);
            } else {
                let p0: f64 = rhs.trim().parse().map_err(|_| rc_err(198, "invalid proportion"))?;
                let a = values(ds, &v1, &mask)?;
                let n = a.len() as f64;
                let p = util::mean(&a);
                let z = (p - p0) / (p0 * (1.0 - p0) / n).sqrt();
                out.push(format!("One-sample test of proportion{:>27}Number of obs = {}", "", n));
                header(&mut out);
                out.push(line(&v1, p, n));
                r.set("P_1", p);
                r.set("N_1", n);
                finish(z, &mut r, &mut out);
            }
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Sdtest;

impl CommandHandler for Sdtest {
    fn name(&self) -> &'static str {
        "sdtest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let mut r = Results::new();
        let (f, df1, df2, title) = if let Some(by) = cmd.opt("by", 2) {
            let v = varlist::resolve_name(ds, cmd.main.trim())?;
            let by = varlist::resolve_name(ds, by.trim())?;
            let (a, b, _, _) = by_groups(ds, &v, &by, &mask)?;
            (util::variance(&a) / util::variance(&b), a.len() as f64 - 1.0, b.len() as f64 - 1.0, "Variance ratio test")
        } else {
            let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax"))?;
            let v1 = varlist::resolve_name(ds, &lhs)?;
            if let Ok(v2) = varlist::resolve_name(ds, &rhs) {
                let a = values(ds, &v1, &mask)?;
                let b = values(ds, &v2, &mask)?;
                (util::variance(&a) / util::variance(&b), a.len() as f64 - 1.0, b.len() as f64 - 1.0, "Variance ratio test")
            } else {
                let sd0: f64 = rhs.trim().parse().map_err(|_| rc_err(198, "invalid sd"))?;
                let a = values(ds, &v1, &mask)?;
                let df = a.len() as f64 - 1.0;
                let chi2 = df * util::variance(&a) / (sd0 * sd0);
                let p_l = dist::chi2(df, chi2);
                let p_u = dist::chi2tail(df, chi2);
                r.set("chi2", chi2);
                r.set("df", df);
                r.set("p_l", p_l);
                r.set("p_u", p_u);
                r.set("p", 2.0 * p_l.min(p_u));
                st.r = r;
                return Ok(format!("One-sample test of variance\n    c = chi2 = {:>9.4}   degrees of freedom = {}\n  Pr(C < c) = {:.4}   2*Pr(C < c) = {:.4}   Pr(C > c) = {:.4}", chi2, df, p_l, 2.0 * p_l.min(p_u), p_u));
            }
        };
        let p_l = dist::f_cdf(df1, df2, f);
        let p_u = dist::ftail(df1, df2, f);
        let p = 2.0 * p_l.min(p_u);
        r.set("F", f);
        r.set("df_1", df1);
        r.set("df_2", df2);
        r.set("p_l", p_l);
        r.set("p_u", p_u);
        r.set("p", p);
        st.r = r;
        Ok(format!("{title}\n    f = {:>9.4}   degrees of freedom = {}, {}\n  Pr(F < f) = {:.4}   2*Pr(F < f) = {:.4}   Pr(F > f) = {:.4}", f, df1, df2, p_l, p, p_u))
    }
}

pub struct Ranksum;

impl CommandHandler for Ranksum {
    fn name(&self) -> &'static str {
        "ranksum"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let by = varlist::resolve_name(ds, cmd.opt("by", 2).ok_or_else(|| rc_err(198, "option by() required"))?.trim())?;
        let (a, b, na, nb) = by_groups(ds, &v, &by, &mask)?;
        let mut all = a.clone();
        all.extend_from_slice(&b);
        let rk = ranks(&all);
        let (n1, n2) = (a.len() as f64, b.len() as f64);
        let n = n1 + n2;
        let r1: f64 = rk[..a.len()].iter().sum();
        let e1 = n1 * (n + 1.0) / 2.0;
        // variance with tie adjustment
        let mut sorted = all.clone();
        sorted.sort_by(f64::total_cmp);
        let mut ties = 0.0;
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i;
            while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            ties += t * t * t - t;
            i = j + 1;
        }
        let var_u = n1 * n2 * (n + 1.0) / 12.0;
        let var_a = var_u - n1 * n2 * ties / (12.0 * n * (n - 1.0));
        let z = (r1 - e1) / var_a.sqrt();
        let p = 2.0 * (1.0 - dist::normal(z.abs()));
        let mut r = Results::new();
        r.set("N_1", n1);
        r.set("N_2", n2);
        r.set("sum_obs", r1);
        r.set("sum_exp", e1);
        r.set("Var_a", var_a);
        r.set("z", z);
        r.set("p", p);
        r.set("p_l", dist::normal(z));
        r.set("p_u", 1.0 - dist::normal(z));
        st.r = r;
        Ok(format!(
            "Two-sample Wilcoxon rank-sum (Mann–Whitney) test\n\n{:>12} |      Obs    Rank sum    Expected\n-------------+---------------------------------\n{:>12} | {:>8} {:>11} {:>11}\n{:>12} | {:>8} {:>11} {:>11}\n-------------+---------------------------------\n    Combined | {:>8} {:>11} {:>11}\n\nUnadjusted variance {:>12}\nAdjustment for ties {:>12}\n                    ----------\nAdjusted variance   {:>12}\n\nH0: {v}({by}=={na}) = {v}({by}=={nb})\n         z = {:>7.3}\n Prob > |z| = {:>7.4}",
            by, abbrev(&na, 12), n1, g9(r1), g9(e1), abbrev(&nb, 12), n2, g9(n * (n + 1.0) / 2.0 - r1), g9(n2 * (n + 1.0) / 2.0),
            n, g9(n * (n + 1.0) / 2.0), g9(n * (n + 1.0) / 2.0), g9(var_u), g9(var_a - var_u), g9(var_a), z, p
        ))
    }
}

pub struct Signrank;

impl CommandHandler for Signrank {
    fn name(&self) -> &'static str {
        "signrank"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let v = varlist::resolve_name(ds, &lhs)?;
        let e = parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
        let env = Env::with_store(ds, st);
        let x = util::col(ds, &v)?;
        let mut d = Vec::new();
        for i in 0..ds.n_obs {
            if mask[i] {
                let y = env.eval(&e, i)?.num()?;
                if !is_missing(x[i]) && !is_missing(y) {
                    d.push(x[i] - y);
                }
            }
        }
        let n = d.len() as f64;
        let absd: Vec<f64> = d.iter().map(|v| v.abs()).collect();
        let rk = ranks(&absd);
        let mut pos = 0.0;
        let mut zero_adj = 0.0;
        let nzero = d.iter().filter(|&&v| v == 0.0).count() as f64;
        for (k, &v) in d.iter().enumerate() {
            if v > 0.0 {
                pos += rk[k];
            } else if v == 0.0 {
                pos += rk[k] / 2.0;
                zero_adj += 1.0;
            }
        }
        let _ = zero_adj;
        let expect = n * (n + 1.0) / 4.0;
        let mut var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0;
        // ties adjustment
        let mut sorted = absd.clone();
        sorted.sort_by(f64::total_cmp);
        let mut i = 0;
        let mut ties = 0.0;
        while i < sorted.len() {
            let mut j = i;
            while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            ties += t * t * t - t;
            i = j + 1;
        }
        let zeros_adj = nzero * (nzero + 1.0) * (2.0 * nzero + 1.0) / 24.0;
        var -= zeros_adj + ties / 48.0;
        let z = (pos - expect) / var.sqrt();
        let p = 2.0 * (1.0 - dist::normal(z.abs()));
        let mut r = Results::new();
        r.set("N", n);
        r.set("sum_pos", pos);
        r.set("sum_neg", n * (n + 1.0) / 2.0 - pos);
        r.set("z", z);
        r.set("Var_a", var);
        r.set("p", p);
        st.r = r;
        Ok(format!("Wilcoxon signed-rank test\n\n        Sign |      Obs   Sum ranks    Expected\n-------------+---------------------------------\n    Positive | {:>8} {:>11} {:>11}\n    Negative | {:>8} {:>11} {:>11}\n        Zero | {:>8} {:>11} {:>11}\n-------------+---------------------------------\n         All | {:>8} {:>11} {:>11}\n\nH0: {lhs} = {rhs}\n         z = {:>7.3}\n Prob > |z| = {:>7.4}",
            d.iter().filter(|&&v| v > 0.0).count(), g9(pos), g9(expect),
            d.iter().filter(|&&v| v < 0.0).count(), g9(n * (n + 1.0) / 2.0 - pos), g9(expect),
            nzero, 0, 0, n, g9(n * (n + 1.0) / 2.0), g9(n * (n + 1.0) / 2.0), z, p))
    }
}

pub struct Signtest;

impl CommandHandler for Signtest {
    fn name(&self) -> &'static str {
        "signtest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let v = varlist::resolve_name(ds, &lhs)?;
        let e = parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
        let env = Env::with_store(ds, st);
        let x = util::col(ds, &v)?;
        let (mut pos, mut neg, mut zero) = (0.0, 0.0, 0.0);
        for i in 0..ds.n_obs {
            if mask[i] {
                let y = env.eval(&e, i)?.num()?;
                if is_missing(x[i]) || is_missing(y) {
                    continue;
                }
                let d = x[i] - y;
                if d > 0.0 {
                    pos += 1.0;
                } else if d < 0.0 {
                    neg += 1.0;
                } else {
                    zero += 1.0;
                }
            }
        }
        let n = pos + neg;
        let p_pos = dist::binomialtail(n, pos, 0.5);
        let p_neg = dist::binomialtail(n, neg, 0.5);
        let p2 = (2.0 * p_pos.min(p_neg)).min(1.0);
        let mut r = Results::new();
        r.set("N", n + zero);
        r.set("pos", pos);
        r.set("neg", neg);
        r.set("p", p2);
        r.set("p_pos", p_pos);
        r.set("p_neg", p_neg);
        st.r = r;
        Ok(format!("Sign test\n\n        Sign |    Observed    Expected\n-------------+------------------------\n    Positive | {:>11} {:>11}\n    Negative | {:>11} {:>11}\n        Zero | {:>11} {:>11}\n-------------+------------------------\n         All | {:>11} {:>11}\n\nTwo-sided test:\n  H0: median of {lhs} - {rhs} = 0 vs.\n  Ha: median of {lhs} - {rhs} != 0\n      Pr(#positive >= {} or #negative >= {}) = {:.4}", pos, n / 2.0, neg, n / 2.0, zero, zero, n + zero, n + zero, pos.max(neg), pos.max(neg), p2))
    }
}

pub struct Kwallis;

impl CommandHandler for Kwallis {
    fn name(&self) -> &'static str {
        "kwallis"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::resolve_name(ds, cmd.main.trim())?;
        let by = varlist::resolve_name(ds, cmd.opt("by", 2).ok_or_else(|| rc_err(198, "option by() required"))?.trim())?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &[v.clone(), by.clone()]);
        let (ids, reps) = util::group_ids(ds, &[by.clone()], &mask);
        let x = util::col(ds, &v)?;
        let rows = util::rows(&mask);
        let vals: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
        let rk = ranks(&vals);
        let n = vals.len() as f64;
        let k = reps.len();
        let mut rs = vec![0.0; k];
        let mut ns = vec![0.0; k];
        for (t, &i) in rows.iter().enumerate() {
            rs[ids[i]] += rk[t];
            ns[ids[i]] += 1.0;
        }
        let h = 12.0 / (n * (n + 1.0)) * (0..k).map(|g| rs[g] * rs[g] / ns[g]).sum::<f64>() - 3.0 * (n + 1.0);
        let mut sorted = vals.clone();
        sorted.sort_by(f64::total_cmp);
        let mut ties = 0.0;
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i;
            while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            ties += t * t * t - t;
            i = j + 1;
        }
        let h_ties = h / (1.0 - ties / (n * n * n - n));
        let df = (k - 1) as f64;
        let mut r = Results::new();
        r.set("df", df);
        r.set("chi2", h);
        r.set("chi2_adj", h_ties);
        st.r = r;
        let g = ds.get_var(&by).unwrap();
        let mut out = vec!["Kruskal–Wallis equality-of-populations rank test".to_string(), String::new()];
        out.push(format!("  +{}+", "-".repeat(24)));
        out.push(format!("  | {:>8} | Obs | Rank sum |", abbrev(&by, 8)));
        for gidx in 0..k {
            out.push(format!("  | {:>8} | {:>3} | {:>8} |", abbrev(&util::cell_text(ds, g, reps[gidx], true), 8), ns[gidx], g9(rs[gidx])));
        }
        out.push(format!("  +{}+", "-".repeat(24)));
        out.push(String::new());
        out.push(format!("chi2({}) = {:>8.3}\nProbability = {:>8.4}", df, h, dist::chi2tail(df, h)));
        out.push(String::new());
        out.push(format!("chi2({}) with ties = {:>8.3}\nProbability = {:>8.4}", df, h_ties, dist::chi2tail(df, h_ties)));
        Ok(out.join("\n"))
    }
}

pub struct Oneway;

impl CommandHandler for Oneway {
    fn name(&self) -> &'static str {
        "oneway"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "oneway response_var factor_var"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let (ids, reps) = util::group_ids(ds, &[vars[1].clone()], &mask);
        let y = util::col(ds, &vars[0])?;
        let k = reps.len();
        let mut groups: Vec<Vec<f64>> = vec![Vec::new(); k];
        for i in 0..ds.n_obs {
            if ids[i] != usize::MAX {
                groups[ids[i]].push(y[i]);
            }
        }
        let all: Vec<f64> = groups.iter().flatten().cloned().collect();
        let n = all.len() as f64;
        let gm = util::mean(&all);
        let ssb: f64 = groups.iter().map(|g| g.len() as f64 * (util::mean(g) - gm).powi(2)).sum();
        let ssw: f64 = groups.iter().map(|g| { let m = util::mean(g); g.iter().map(|x| (x - m).powi(2)).sum::<f64>() }).sum();
        let dfb = (k - 1) as f64;
        let dfw = n - k as f64;
        let f = (ssb / dfb) / (ssw / dfw);
        let p = dist::ftail(dfb, dfw, f);
        // Bartlett's test
        let sp2 = ssw / dfw;
        let mut num = dfw * sp2.ln();
        let mut c = 0.0;
        for g in &groups {
            let ni = g.len() as f64;
            if ni > 1.0 {
                num -= (ni - 1.0) * util::variance(g).ln();
                c += 1.0 / (ni - 1.0);
            }
        }
        let bart = num / (1.0 + (c - 1.0 / dfw) / (3.0 * (k as f64 - 1.0)));
        let mut out = Vec::new();
        if cmd.has_opt("tabulate", 3) {
            let g = ds.get_var(&vars[1]).unwrap();
            out.push(format!("{:>12} |            Summary of {}\n{:>12} |        Mean   Std. dev.       Freq.\n------------+------------------------------------", "", vars[0], abbrev(&vars[1], 12)));
            for (gi, &rep) in reps.iter().enumerate() {
                out.push(format!("{:>12} | {:>11} {:>11} {:>11}", abbrev(&util::cell_text(ds, g, rep, true), 12), g9(util::mean(&groups[gi])), g9(if groups[gi].len() > 1 { util::variance(&groups[gi]).sqrt() } else { SYSMISS }), groups[gi].len()));
            }
            out.push(format!("------------+------------------------------------\n{:>12} | {:>11} {:>11} {:>11}\n", "Total", g9(gm), g9(util::variance(&all).sqrt()), n));
        }
        out.push("                        Analysis of variance".into());
        out.push("    Source              SS         df      MS            F     Prob > F".into());
        out.push("------------------------------------------------------------------------".into());
        out.push(format!("Between groups   {:>12} {:>6} {:>12} {:>10.2}     {:.4}", g9(ssb), dfb, g9(ssb / dfb), f, p));
        out.push(format!(" Within groups   {:>12} {:>6} {:>12}", g9(ssw), dfw, g9(ssw / dfw)));
        out.push("------------------------------------------------------------------------".into());
        out.push(format!("    Total        {:>12} {:>6} {:>12}", g9(ssb + ssw), n - 1.0, g9((ssb + ssw) / (n - 1.0))));
        out.push(String::new());
        out.push(format!("Bartlett's equal-variances test: chi2({}) = {:>8.4}    Prob>chi2 = {:.3}", dfb, bart, dist::chi2tail(dfb, bart)));
        let mut r = Results::new();
        r.set("N", n);
        r.set("df_r", dfw);
        r.set("df_m", dfb);
        r.set("F", f);
        r.set("mss", ssb);
        r.set("rss", ssw);
        r.set("chi2bart", bart);
        r.set("df_bart", dfb);
        st.r = r;
        Ok(out.join("\n"))
    }
}

pub struct Ksmirnov;

impl CommandHandler for Ksmirnov {
    fn name(&self) -> &'static str {
        "ksmirnov"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let ks_p = |d: f64, ne: f64| {
            // asymptotic Kolmogorov distribution
            let lam = d * ne.sqrt();
            let mut s = 0.0;
            for j in 1..=100 {
                let jf = j as f64;
                s += 2.0 * (if j % 2 == 1 { 1.0 } else { -1.0 }) * (-2.0 * jf * jf * lam * lam).exp();
            }
            s.clamp(0.0, 1.0)
        };
        let mut r = Results::new();
        if let Some(by) = cmd.opt("by", 2) {
            let v = varlist::resolve_name(ds, cmd.main.trim())?;
            let by = varlist::resolve_name(ds, by.trim())?;
            let (mut a, mut b, na, nb) = by_groups(ds, &v, &by, &mask)?;
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            let (n1, n2) = (a.len() as f64, b.len() as f64);
            let mut pts: Vec<f64> = a.iter().chain(&b).cloned().collect();
            pts.sort_by(f64::total_cmp);
            pts.dedup();
            let (mut dpos, mut dneg) = (0.0f64, 0.0f64);
            for &x in &pts {
                let fa = a.iter().filter(|&&v| v <= x).count() as f64 / n1;
                let fb = b.iter().filter(|&&v| v <= x).count() as f64 / n2;
                dpos = dpos.max(fa - fb);
                dneg = dneg.max(fb - fa);
            }
            let d = dpos.max(dneg);
            let ne = n1 * n2 / (n1 + n2);
            let p = ks_p(d, ne);
            r.set("D", d);
            r.set("p", p);
            r.set("D_1", dpos);
            r.set("D_2", -dneg);
            st.r = r;
            return Ok(format!("Two-sample Kolmogorov–Smirnov test for equality of distribution functions\n\n Smaller group       D     p-value  \n ----------------------------------\n {:<12} {:>9.4}   {:>7.3}\n {:<12} {:>9.4}   {:>7.3}\n Combined K-S {:>9.4}   {:>7.3}", na, dpos, (-2.0 * ne * dpos * dpos).exp(), nb, -dneg, (-2.0 * ne * dneg * dneg).exp(), d, p));
        }
        let (lhs, rhs) = split_eq(&cmd.main).ok_or_else(|| rc_err(198, "invalid syntax"))?;
        let v = varlist::resolve_name(ds, &lhs)?;
        let e = parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
        let env = Env::with_store(ds, st);
        let x = util::col(ds, &v)?;
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for i in 0..ds.n_obs {
            if mask[i] && !is_missing(x[i]) {
                pairs.push((x[i], env.eval(&e, i)?.num()?));
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n = pairs.len() as f64;
        let mut d = 0.0f64;
        for (k, (_, f)) in pairs.iter().enumerate() {
            d = d.max(((k + 1) as f64 / n - f).abs()).max((f - k as f64 / n).abs());
        }
        let p = ks_p(d, n);
        r.set("D", d);
        r.set("p", p);
        st.r = r;
        Ok(format!("One-sample Kolmogorov–Smirnov test against theoretical distribution\n{rhs}\n\n Smaller group       D     p-value\n ----------------------------------\n Combined K-S {:>9.4}   {:>7.3}", d, p))
    }
}

/// Shapiro-Wilk W test (Royston 1995 approximation).
pub fn shapiro_wilk(x: &mut [f64]) -> (f64, f64, f64) {
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let nf = n as f64;
    if n < 3 {
        return (SYSMISS, SYSMISS, SYSMISS);
    }
    let m: Vec<f64> = (1..=n).map(|i| dist::invnormal((i as f64 - 0.375) / (nf + 0.25))).collect();
    let summ2: f64 = m.iter().map(|v| v * v).sum();
    let ssumm2 = summ2.sqrt();
    let u = 1.0 / nf.sqrt();
    let poly = |c: &[f64], x: f64| c.iter().rev().fold(0.0, |acc, &ci| acc * x + ci);
    let c1 = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056];
    let c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
    let mut a = vec![0.0; n];
    if n == 3 {
        a[0] = -std::f64::consts::FRAC_1_SQRT_2;
        a[2] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an = poly(&c1, u) + m[n - 1] / ssumm2;
        if n > 5 {
            let an1 = poly(&c2, u) + m[n - 2] / ssumm2;
            let fac = ((summ2 - 2.0 * m[n - 1].powi(2) - 2.0 * m[n - 2].powi(2)) / (1.0 - 2.0 * an * an - 2.0 * an1 * an1)).sqrt();
            for i in 2..n - 2 {
                a[i] = m[i] / fac;
            }
            a[n - 1] = an;
            a[0] = -an;
            a[n - 2] = an1;
            a[1] = -an1;
        } else {
            let fac = ((summ2 - 2.0 * m[n - 1].powi(2)) / (1.0 - 2.0 * an * an)).sqrt();
            for i in 1..n - 1 {
                a[i] = m[i] / fac;
            }
            a[n - 1] = an;
            a[0] = -an;
        }
    }
    let mean = x.iter().sum::<f64>() / nf;
    let ss: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    let w = (a.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum::<f64>()).powi(2) / ss;
    let w = w.min(1.0);
    let z = if n == 3 {
        let p = (6.0 / std::f64::consts::PI) * (w.sqrt().asin() - (0.75f64).sqrt().asin());
        let p = p.clamp(0.0, 1.0);
        return (w, SYSMISS, p);
    } else if n <= 11 {
        let gamma = -2.273 + 0.459 * nf;
        let mu = 0.5440 - 0.39978 * nf + 0.025054 * nf * nf - 0.0006714 * nf.powi(3);
        let s = (1.3822 - 0.77857 * nf + 0.062767 * nf * nf - 0.0020322 * nf.powi(3)).exp();
        let w1 = -(gamma - (1.0 - w).ln()).ln();
        (w1 - mu) / s
    } else {
        let xx = nf.ln();
        let mu = -1.5861 - 0.31082 * xx - 0.083751 * xx * xx + 0.0038915 * xx.powi(3);
        let s = (-0.4803 - 0.082676 * xx + 0.0030302 * xx * xx).exp();
        ((1.0 - w).ln() - mu) / s
    };
    (w, z, (1.0 - dist::normal(z)).max(0.0))
}

pub struct Swilk;

impl CommandHandler for Swilk {
    fn name(&self) -> &'static str {
        "swilk"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mask = util::touse(ds, st, cmd)?;
        let mut out = vec![
            format!("{:>50}", "Shapiro–Wilk W test for normal data"),
            String::new(),
            "    Variable |        Obs       W           V         z       Prob>z".into(),
            "-------------+------------------------------------------------------".into(),
        ];
        let mut r = Results::new();
        for v in vars {
            let mut x = values(ds, &v, &mask)?;
            let n = x.len() as f64;
            let (w, z, p) = shapiro_wilk(&mut x);
            // V: departure index relative to the median of W under normality
            let vstat = if n >= 12.0 {
                let xx = n.ln();
                let mu = -1.5861 - 0.31082 * xx - 0.083751 * xx * xx + 0.0038915 * xx.powi(3);
                (1.0 - w) / mu.exp()
            } else {
                SYSMISS
            };
            out.push(format!("{:>12} | {:>10} {:>9.5} {:>9.3} {:>9.3}    {:>8.5}", abbrev(&v, 12), n, w, vstat, z, p));
            r.set("N", n);
            r.set("W", w);
            r.set("V", vstat);
            r.set("z", z);
            r.set("p", p);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swilk_known_value() {
        // R: shapiro.test(c(148,154,158,160,161,162,166,170,182,195,236))
        // W = 0.79, p = 0.0067 (approx)
        let mut x = vec![148.0, 154.0, 158.0, 160.0, 161.0, 162.0, 166.0, 170.0, 182.0, 195.0, 236.0];
        let (w, _, p) = shapiro_wilk(&mut x);
        assert!((w - 0.79).abs() < 0.01, "{w}");
        assert!(p < 0.02 && p > 0.001, "{p}");
    }
}
