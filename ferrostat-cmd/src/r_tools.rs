//! Small base-reference tools: z tests (`ztest`, `ztesti`), arithmetic,
//! geometric and harmonic means (`ameans`), symmetry and marginal
//! homogeneity tests (`symmetry`, `symmi`), stem-and-leaf plots (`stem`),
//! robust nonlinear smoothers (`smooth`), variance-weighted least squares
//! (`vwls`), ratios (`ratio`), tetrachoric correlations (`tetrachoric`),
//! numerical derivatives and integrals (`dydx`, `integ`), factor-variable
//! settings (`fvset`), replicate weights (`rwgen`) and informational
//! commands (`query`, `about`, `copyright`, `more`, `search`, `sj`,
//! `translate`).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![
        ("ztest", Box::new(Ztest)),
        ("ztesti", Box::new(Ztesti)),
        ("ameans", Box::new(Ameans)),
        ("gmeans", Box::new(Ameans)),
        ("hmeans", Box::new(Ameans)),
        ("means", Box::new(Ameans)),
        ("symmetry", Box::new(Symmetry { immediate: false })),
        ("symmi", Box::new(Symmetry { immediate: true })),
        ("stem", Box::new(Stem)),
        ("smooth", Box::new(Smooth)),
        ("vwls", Box::new(Vwls)),
        ("ratio", Box::new(Ratio)),
        ("tetrachoric", Box::new(Tetrachoric)),
        ("dydx", Box::new(Dydx { integ: false })),
        ("integ", Box::new(Dydx { integ: true })),
        ("fvset", Box::new(Fvset)),
        ("rwgen", Box::new(Rwgen)),
        ("query", Box::new(Info("query"))),
        ("about", Box::new(Info("about"))),
        ("copyright", Box::new(Info("copyright"))),
        ("more", Box::new(Info("more"))),
        ("search", Box::new(Info("search"))),
        ("sj", Box::new(Info("sj"))),
        ("translate", Box::new(Translate)),
    ]
}

fn g8(v: f64) -> String {
    if is_missing(v) || !v.is_finite() { ".".into() } else { fmt_g(v, 8) }
}

fn col_of(ds: &DataSet, v: &str) -> Result<(String, Vec<f64>), String> {
    let n = xm::var_name(ds, v)?;
    let c = util::col(ds, &n)?;
    Ok((n, c))
}

fn var_label(ds: &DataSet, v: &str) -> String {
    ds.get_var(v).map(|x| x.label.clone()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// ztest / ztesti
// ---------------------------------------------------------------------------

struct ZGroup {
    name: String,
    n: f64,
    mean: f64,
    sd: f64,
}

fn zline(name: &str, n: Option<f64>, mean: f64, se: f64, sd: Option<f64>, crit: f64) -> String {
    format!(
        "{:>8} |{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}",
        crate::summarize::abbrev(name, 8),
        n.map_or(String::new(), |n| util::commas(n as usize)),
        g8(mean),
        g8(se),
        sd.map_or(String::new(), g8),
        g8(mean - crit * se),
        g8(mean + crit * se)
    )
}

const ZBAR: &str = "------------------------------------------------------------------------------";
const ZSEP: &str = "---------+--------------------------------------------------------------------";

/// Render a z test: rows, optional difference row, the statistic and tails.
#[allow(clippy::too_many_arguments)]
fn ztest_out(title: &str, first: &str, groups: &[ZGroup], diff: Option<(String, Option<f64>, f64, f64, Option<f64>)>, stat_lhs: &str, h0: &str, ha: (&str, &str, &str), z: f64, level: f64) -> (Vec<String>, f64, f64, f64) {
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut out = vec![title.to_string(), String::new(), ZBAR.to_string(), format!("{:>8} |     Obs        Mean    Std. err.   Std. dev.   [{}% conf. interval]", first, fmt_g(level, 4)), ZSEP.to_string()];
    for g in groups {
        out.push(zline(&g.name, Some(g.n), g.mean, g.sd / g.n.sqrt(), Some(g.sd), crit));
    }
    if let Some((name, n, est, se, sd)) = diff {
        if !groups.is_empty() {
            out.push(ZSEP.to_string());
        }
        out.push(zline(&name, n, est, se, sd, crit));
    }
    out.push(ZBAR.to_string());
    let p_l = dist::normal(z);
    let p_u = dist::normal(-z);
    let p = 2.0 * dist::normal(-z.abs());
    out.push(format!("{:<66}z = {:>8.4}", stat_lhs, z));
    out.push(h0.to_string());
    out.push(String::new());
    // the three alternatives are centred on columns 10, 39 and 69
    let mut line = String::new();
    for (txt, c) in [(ha.0.trim(), 10usize), (ha.1.trim(), 39), (ha.2.trim(), 69)] {
        let start = c.saturating_sub(txt.len() / 2).max(line.len() + 1);
        line.push_str(&" ".repeat(start - line.len()));
        line.push_str(txt);
    }
    out.push(line);
    out.push(format!(" Pr(Z < z) = {:.4}         Pr(|Z| > |z|) = {:.4}          Pr(Z > z) = {:.4}", p_l, p, p_u));
    (out, p_l, p, p_u)
}

fn set_zr(st: &mut Store, z: f64, p_l: f64, p: f64, p_u: f64, se: f64, level: f64, extra: &[(&str, f64)]) {
    let mut r = Results::new();
    r.set("z", z);
    r.set("p_l", p_l);
    r.set("p", p);
    r.set("p_u", p_u);
    r.set("se", se);
    r.set("level", level);
    for (k, v) in extra {
        r.set(k, *v);
    }
    st.r = r;
}

fn num_opt(cmd: &Command, name: &str, min: usize) -> Result<Option<f64>, String> {
    match cmd.opt(name, min) {
        None => Ok(None),
        Some(s) => s.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("{name}() invalid"))),
    }
}

pub struct Ztest;

impl CommandHandler for Ztest {
    fn name(&self) -> &'static str {
        "ztest"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = util::level(cmd, st)?;
        let mask = util::touse(ds, st, cmd)?;
        let main = cmd.main.trim();
        let sd = num_opt(cmd, "sd", 2)?;
        let sd1 = num_opt(cmd, "sd1", 3)?;
        let sd2 = num_opt(cmd, "sd2", 3)?;
        if let Some((a, b)) = main.split_once("==") {
            let (an, ac) = col_of(ds, a.trim())?;
            if let Ok(h) = b.trim().parse::<f64>() {
                // one-sample
                let x: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(ac[i])).map(|i| ac[i]).collect();
                let s = sd.unwrap_or(1.0);
                let n = x.len() as f64;
                let m = util::mean(&x);
                let se = s / n.sqrt();
                let z = (m - h) / se;
                let hs = fmt_g(h, 8);
                let (out, pl, p, pu) = ztest_out("One-sample z test", "Variable", &[ZGroup { name: an.clone(), n, mean: m, sd: s }], None, &format!("    mean = mean({an})"), &format!("H0: mean = {hs}"), (&format!("Ha: mean < {hs}"), &format!("Ha: mean != {hs}"), &format!("Ha: mean > {hs}")), z, level);
                set_zr(st, z, pl, p, pu, se, level, &[("N_1", n), ("mu_1", m), ("sd_1", s)]);
                return Ok(out.join("\n"));
            }
            let (bn, bc) = col_of(ds, b.trim())?;
            if cmd.has_opt("unpaired", 3) {
                let x: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(ac[i])).map(|i| ac[i]).collect();
                let y: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(bc[i])).map(|i| bc[i]).collect();
                return two_sample(st, &an, &bn, &x, &y, sd1.or(sd).unwrap_or(1.0), sd2.or(sd).unwrap_or(1.0), level, "Variable", &format!("mean({an}) - mean({bn})"));
            }
            // paired
            let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(ac[i]) && !is_missing(bc[i])).collect();
            let x: Vec<f64> = rows.iter().map(|&i| ac[i]).collect();
            let y: Vec<f64> = rows.iter().map(|&i| bc[i]).collect();
            let n = rows.len() as f64;
            let (mx, my) = (util::mean(&x), util::mean(&y));
            let d = mx - my;
            let mut groups = Vec::new();
            let sdd = match num_opt(cmd, "sddiff", 3)? {
                Some(v) => v,
                None => {
                    let (s1, s2) = (sd1.or(sd).unwrap_or(1.0), sd2.or(sd).unwrap_or(1.0));
                    let rho = num_opt(cmd, "corr", 2)?.unwrap_or(0.0);
                    groups.push(ZGroup { name: an.clone(), n, mean: mx, sd: s1 });
                    groups.push(ZGroup { name: bn.clone(), n, mean: my, sd: s2 });
                    (s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2).max(0.0).sqrt()
                }
            };
            let se = sdd / n.sqrt();
            let z = d / se;
            let (out, pl, p, pu) = ztest_out(
                "Paired z test",
                "Variable",
                &groups,
                Some(("diff".into(), Some(n), d, se, Some(sdd))),
                &format!("     mean(diff) = mean({an} - {bn})"),
                " H0: mean(diff) = 0",
                (" Ha: mean(diff) < 0", "Ha: mean(diff) != 0", "Ha: mean(diff) > 0"),
                z,
                level,
            );
            set_zr(st, z, pl, p, pu, se, level, &[("N_1", n), ("N_2", n), ("mu_1", mx), ("mu_2", my), ("sd_d", sdd)]);
            return Ok(out.join("\n"));
        }
        let by = cmd.opt("by", 2).ok_or_else(|| rc_err(198, "== or by() required"))?;
        let (vn, vc) = col_of(ds, main)?;
        let byn = xm::var_name(ds, by)?;
        let mut m = mask.clone();
        util::markout(ds, &mut m, &[vn.clone(), byn.clone()]);
        let (ids, reps) = util::group_ids(ds, &[byn.clone()], &m);
        if reps.len() != 2 {
            return Err(rc_err(420, format!("{} groups found, 2 required", reps.len())));
        }
        let bv = ds.get_var(&byn).unwrap();
        let names = (util::cell_text(ds, bv, reps[0], true), util::cell_text(ds, bv, reps[1], true));
        let x: Vec<f64> = (0..ds.n_obs).filter(|&i| m[i] && ids[i] == 0).map(|i| vc[i]).collect();
        let y: Vec<f64> = (0..ds.n_obs).filter(|&i| m[i] && ids[i] == 1).map(|i| vc[i]).collect();
        two_sample(st, &names.0, &names.1, &x, &y, sd1.or(sd).unwrap_or(1.0), sd2.or(sd).unwrap_or(1.0), level, "Group", &format!("mean({}) - mean({})", names.0, names.1))
    }
}

#[allow(clippy::too_many_arguments)]
fn two_sample(st: &mut Store, n1: &str, n2: &str, x: &[f64], y: &[f64], s1: f64, s2: f64, level: f64, first: &str, diffname: &str) -> Result<String, String> {
    let (a, b) = (x.len() as f64, y.len() as f64);
    let (m1, m2) = (util::mean(x), util::mean(y));
    let se = (s1 * s1 / a + s2 * s2 / b).sqrt();
    let z = (m1 - m2) / se;
    let (out, pl, p, pu) = ztest_out(
        "Two-sample z test",
        first,
        &[ZGroup { name: n1.into(), n: a, mean: m1, sd: s1 }, ZGroup { name: n2.into(), n: b, mean: m2, sd: s2 }],
        Some(("diff".into(), None, m1 - m2, se, None)),
        &format!("    diff = {diffname}"),
        "H0: diff = 0",
        ("Ha: diff < 0", "Ha: diff != 0", "Ha: diff > 0"),
        z,
        level,
    );
    set_zr(st, z, pl, p, pu, se, level, &[("N_1", a), ("N_2", b), ("mu_1", m1), ("mu_2", m2), ("sd_1", s1), ("sd_2", s2)]);
    Ok(out.join("\n"))
}

pub struct Ztesti;

impl CommandHandler for Ztesti {
    fn name(&self) -> &'static str {
        "ztesti"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = util::level(cmd, st)?;
        let v: Vec<f64> = cmd.main.split_whitespace().map(|s| s.parse::<f64>().map_err(|_| rc_err(198, format!("'{s}' found where number expected")))).collect::<Result<_, _>>()?;
        match v.len() {
            4 => {
                let (n, m, s, h) = (v[0], v[1], v[2], v[3]);
                let se = s / n.sqrt();
                let z = (m - h) / se;
                let hs = fmt_g(h, 8);
                let (out, pl, p, pu) = ztest_out("One-sample z test", "", &[ZGroup { name: "x".into(), n, mean: m, sd: s }], None, "    mean = mean(x)", &format!("H0: mean = {hs}"), (&format!("Ha: mean < {hs}"), &format!("Ha: mean != {hs}"), &format!("Ha: mean > {hs}")), z, level);
                set_zr(st, z, pl, p, pu, se, level, &[("N_1", n), ("mu_1", m), ("sd_1", s)]);
                Ok(out.join("\n"))
            }
            6 => {
                let (a, m1, s1, b, m2, s2) = (v[0], v[1], v[2], v[3], v[4], v[5]);
                let se = (s1 * s1 / a + s2 * s2 / b).sqrt();
                let z = (m1 - m2) / se;
                let (out, pl, p, pu) = ztest_out(
                    "Two-sample z test",
                    "",
                    &[ZGroup { name: "x".into(), n: a, mean: m1, sd: s1 }, ZGroup { name: "y".into(), n: b, mean: m2, sd: s2 }],
                    Some(("diff".into(), None, m1 - m2, se, None)),
                    "    diff = mean(x) - mean(y)",
                    "H0: diff = 0",
                    ("Ha: diff < 0", "Ha: diff != 0", "Ha: diff > 0"),
                    z,
                    level,
                );
                set_zr(st, z, pl, p, pu, se, level, &[("N_1", a), ("N_2", b), ("mu_1", m1), ("mu_2", m2), ("sd_1", s1), ("sd_2", s2)]);
                Ok(out.join("\n"))
            }
            _ => Err(rc_err(198, "ztesti requires 4 or 6 numbers")),
        }
    }
}

// ---------------------------------------------------------------------------
// ameans
// ---------------------------------------------------------------------------

pub struct Ameans;

impl CommandHandler for Ameans {
    fn name(&self) -> &'static str {
        "ameans"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = util::level(cmd, st)?;
        let mask = util::touse(ds, st, cmd)?;
        let vars = crate::varlist::expand(ds, if cmd.main.trim().is_empty() { "_all" } else { cmd.main.trim() })?;
        let add = num_opt(cmd, "add", 1)?.unwrap_or(0.0);
        let only = cmd.has_opt("only", 4);
        let w = util::weights(ds, st, cmd, &mut mask.clone(), "af")?;
        let star = if add != 0.0 { "*" } else { "" };
        let a = (1.0 - level / 100.0) / 2.0;
        let mut out = vec![String::new(), format!("{:>12} | {:>10}{:>13}{:>12}       [{}% conf. interval]", "Variable", "Type", "Obs", "Mean", fmt_g(level, 4)), format!("{}+{}", "-".repeat(13), "-".repeat(64))];
        let mut r = Results::new();
        let mut any_harm_missing = false;
        let nv = vars.len();
        for (vi, v) in vars.iter().enumerate() {
            let c = util::col(ds, v)?;
            if ds.get_var(v).is_some_and(|x| x.var_type.is_string()) {
                continue;
            }
            let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).collect();
            let wt = |i: usize| w.as_ref().map_or(1.0, |w| w.w[i]);
            let stats = |vals: &[(f64, f64)]| -> (f64, f64, f64) {
                // weighted mean, variance of the mean, count
                let sw: f64 = vals.iter().map(|x| x.1).sum();
                let n = vals.len() as f64;
                let m = vals.iter().map(|x| x.0 * x.1).sum::<f64>() / sw;
                let var = vals.iter().map(|x| x.1 * (x.0 - m).powi(2)).sum::<f64>() / sw * n / (n - 1.0);
                (m, var / n, n)
            };
            let ar: Vec<(f64, f64)> = rows.iter().map(|&i| (c[i] + add, wt(i))).collect();
            let pos: Vec<(f64, f64)> = ar.iter().filter(|x| x.0 > 0.0).cloned().collect();
            let (m, vm, n) = stats(&ar);
            let t = dist::invttail(n - 1.0, a);
            let (lb, ub) = (m - t * vm.sqrt(), m + t * vm.sqrt());
            let lg: Vec<(f64, f64)> = pos.iter().map(|x| (x.0.ln(), x.1)).collect();
            let (mg, vg, np) = stats(&lg);
            let tp = dist::invttail(np - 1.0, a);
            let hm: Vec<(f64, f64)> = pos.iter().map(|x| (1.0 / x.0, x.1)).collect();
            let (mh, vh, _) = stats(&hm);
            let (gm, glb, gub) = (mg.exp(), (mg - tp * vg.sqrt()).exp(), (mg + tp * vg.sqrt()).exp());
            let hmean = 1.0 / mh;
            let (hlb, hub) = if mh - tp * vh.sqrt() > 0.0 { (1.0 / (mh + tp * vh.sqrt()), 1.0 / (mh - tp * vh.sqrt())) } else { (SYSMISS, SYSMISS) };
            if is_missing(hlb) {
                any_harm_missing = true;
            }
            let num = |v: f64| if is_missing(v) || !v.is_finite() { ".".to_string() } else { fmt_g(v, 8) };
            let line = |name: &str, ty: &str, n: f64, m: f64, l: f64, u: f64| format!("{:>12} | {:>10}{:>13}{:>12}{:>16}{:>11}", name, ty, util::commas(n as usize), num(m), num(l), format!("{}{star}", num(u)));
            out.push(line(&crate::summarize::abbrev(v, 12), "Arithmetic", n, m, lb, ub));
            if !only {
                out.push(line("", "Geometric", np, gm, glb, gub));
                out.push(line("", "Harmonic", np, hmean, hlb, hub));
            }
            if vi + 1 < nv {
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
            }
            for (k, val) in [("N", n), ("N_pos", np), ("mean", m), ("lb", lb), ("ub", ub), ("Var", vm * n), ("mean_g", gm), ("lb_g", glb), ("ub_g", gub), ("Var_g", vg * np), ("mean_h", hmean), ("lb_h", hlb), ("ub_h", hub), ("Var_h", vh * np)] {
                r.set(k, val);
            }
        }
        out.push("-".repeat(78));
        if add != 0.0 {
            out.push(format!("* {} was added to the variables prior to calculating the results.", fmt_g(add, 8)));
        }
        if any_harm_missing && !only {
            out.push("Note: Missing values in confidence intervals for harmonic mean indicate that".into());
            out.push("      confidence interval is undefined for corresponding variables.".into());
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// symmetry / symmi
// ---------------------------------------------------------------------------

pub struct Symmetry {
    pub immediate: bool,
}

/// Exact probability that the Bowker statistic is at least `obs` given the
/// off-diagonal pair totals (each pair splits Binomial(m, 1/2)).
fn bowker_exact(pairs: &[f64], obs: f64) -> f64 {
    let mut dist: Vec<(f64, f64)> = vec![(0.0, 1.0)];
    for &m in pairs {
        if m <= 0.0 {
            continue;
        }
        let mi = m as usize;
        let mut terms: Vec<(f64, f64)> = Vec::new();
        for k in 0..=mi {
            let d = 2.0 * k as f64 - m;
            terms.push((d * d / m, dist::binomialp(m, k as f64, 0.5)));
        }
        let mut next: std::collections::BTreeMap<i64, (f64, f64)> = Default::default();
        for &(v, p) in &dist {
            for &(t, q) in &terms {
                let s = v + t;
                let key = (s * 1e8).round() as i64;
                let e = next.entry(key).or_insert((s, 0.0));
                e.1 += p * q;
            }
        }
        dist = next.into_values().collect();
    }
    dist.iter().filter(|(v, _)| *v >= obs - 1e-7 * obs.abs().max(1.0)).map(|x| x.1).sum::<f64>().min(1.0)
}

impl CommandHandler for Symmetry {
    fn name(&self) -> &'static str {
        if self.immediate { "symmi" } else { "symmetry" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // the table n[i][j]: rows = first variable, columns = second
        let (labels, t, rowtitle, coltitle) = if self.immediate {
            let rows: Vec<Vec<f64>> = cmd
                .main
                .split('\\')
                .map(|r| r.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| rc_err(198, format!("'{x}' found where number expected")))).collect::<Result<Vec<_>, _>>())
                .collect::<Result<_, _>>()?;
            let k = rows.len();
            if rows.iter().any(|r| r.len() != k) {
                return Err(rc_err(198, "table must be square"));
            }
            ((1..=k).map(|i| i.to_string()).collect::<Vec<_>>(), rows, String::new(), String::new())
        } else {
            let vars = crate::varlist::expand(ds, cmd.main.trim())?;
            if vars.len() != 2 {
                return Err(rc_err(102, "two variables required"));
            }
            let mut mask = util::touse(ds, st, cmd)?;
            let w = util::weights(ds, st, cmd, &mut mask, "f")?;
            util::markout(ds, &mut mask, &vars);
            let a = util::col(ds, &vars[0])?;
            let b = util::col(ds, &vars[1])?;
            let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i]).flat_map(|i| [a[i], b[i]]).collect();
            lv.sort_by(f64::total_cmp);
            lv.dedup();
            let k = lv.len();
            let mut t = vec![vec![0.0; k]; k];
            for i in 0..ds.n_obs {
                if mask[i] {
                    let r = lv.iter().position(|x| *x == a[i]).unwrap();
                    let c = lv.iter().position(|x| *x == b[i]).unwrap();
                    t[r][c] += w.as_ref().map_or(1.0, |w| w.w[i]);
                }
            }
            let va = ds.get_var(&vars[0]).unwrap();
            let labels: Vec<String> = lv.iter().map(|&l| crate::teffects::level_label(ds, &vars[0], l)).collect();
            let _ = va;
            let lab = |v: &str| {
                let l = var_label(ds, v);
                if l.is_empty() { v.to_string() } else { l }
            };
            (labels, t, lab(&vars[0]), lab(&vars[1]))
        };
        let k = t.len();
        let n: f64 = t.iter().flatten().sum();
        let rs: Vec<f64> = t.iter().map(|r| r.iter().sum()).collect();
        let cs: Vec<f64> = (0..k).map(|j| (0..k).map(|i| t[i][j]).sum()).collect();
        // Bowker
        let mut chi2 = 0.0;
        let mut df = 0.0;
        let mut contrib = Vec::new();
        let mut pairs = Vec::new();
        for i in 0..k {
            for j in i + 1..k {
                let s = t[i][j] + t[j][i];
                if s > 0.0 {
                    let c = (t[i][j] - t[j][i]).powi(2) / s;
                    chi2 += c;
                    df += 1.0;
                    contrib.push((i, j, c));
                    pairs.push(s);
                }
            }
        }
        // Stuart-Maxwell
        let m = k - 1;
        let d: Vec<f64> = (0..m).map(|i| rs[i] - cs[i]).collect();
        let mut v = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                v[(i, j)] = if i == j { rs[i] + cs[i] - 2.0 * t[i][i] } else { -(t[i][j] + t[j][i]) };
            }
        }
        let (vi, dropped) = v.sym_inverse(1e-12);
        let sm: f64 = (0..m).map(|a| (0..m).map(|b| d[a] * vi[(a, b)] * d[b]).sum::<f64>()).sum();
        let df_sm = (m - dropped.len()) as f64;
        let mut out = Vec::new();
        if !cmd.has_opt("notable", 3) {
            out.extend(sym_table(&labels, &t, &rs, &cs, n, &rowtitle, &coltitle));
            out.push(String::new());
        }
        if cmd.has_opt("contrib", 3) {
            out.push("------------------------------------".into());
            out.push("                  Contribution".into());
            out.push("                   to symmetry".into());
            out.push("   Cells           chi-squared".into());
            out.push("------------------------------------".into());
            for (i, j, c) in &contrib {
                out.push(format!(" {:<12}{:>15.4}", format!("n{}_{} & n{}_{}", i + 1, j + 1, j + 1, i + 1), c));
            }
            out.push("------------------------------------".into());
            out.push(String::new());
        }
        out.push(format!("{:<45}chi2     df      Prob>chi2", ""));
        out.push("-".repeat(70));
        let line = |l: &str, c: f64, d: f64| format!("{:<42}{:>7.2}{:>7}{:>15.4}", l, c, fmt_g(d, 4), dist::chi2tail(d, c));
        out.push(line("Symmetry (asymptotic)", chi2, df));
        out.push(line("Marginal homogeneity (Stuart--Maxwell)", sm, df_sm));
        let mut r = Results::new();
        if cmd.has_opt("mh", 2) {
            let b: f64 = (0..k).filter(|&i| rs[i] + cs[i] > 0.0).map(|i| (rs[i] - cs[i]).powi(2) / (rs[i] + cs[i])).sum();
            let nd: f64 = (0..k).filter(|&i| rs[i] + cs[i] - 2.0 * t[i][i] > 0.0).map(|i| (rs[i] - cs[i]).powi(2) / (rs[i] + cs[i] - 2.0 * t[i][i])).sum::<f64>() * (k as f64 - 1.0) / k as f64;
            out.push(line("Marginal homogeneity (Bickenboller)", b, m as f64));
            out.push(line("Marginal homogeneity (no diagonals)", nd, m as f64));
            r.set("chi2_b", b);
            r.set("p_b", dist::chi2tail(m as f64, b));
            r.set("chi2_nd", nd);
            r.set("p_nd", dist::chi2tail(m as f64, nd));
        }
        if cmd.has_opt("exact", 2) {
            let pe = bowker_exact(&pairs, chi2);
            out.push(format!("{:<42}{:>29.4}", "Symmetry (exact significance probability)", pe));
            r.set("p_exact", pe);
        }
        out.push("-".repeat(70));
        r.set("N_pair", n);
        r.set("chi2", chi2);
        r.set("df", df);
        r.set("p", dist::chi2tail(df, chi2));
        r.set("chi2_sm", sm);
        r.set("df_sm", df_sm);
        r.set("p_sm", dist::chi2tail(df_sm, sm));
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn sym_table(labels: &[String], t: &[Vec<f64>], rs: &[f64], cs: &[f64], n: f64, rowtitle: &str, coltitle: &str) -> Vec<String> {
    let k = t.len();
    let lw = labels.iter().map(|s| s.len()).max().unwrap_or(5).clamp(9, 16).max(rowtitle.len().min(16));
    let cw = labels.iter().map(|s| s.len() + 1).max().unwrap_or(10).max(10);
    let mut out = Vec::new();
    let total_w = cw * (k + 1);
    out.push(format!("{:<lw$} | {:^total_w$}", if rowtitle.is_empty() { "" } else { rowtitle }, coltitle));
    let mut h = format!("{:<lw$} |", "");
    for l in labels {
        h.push_str(&format!("{:>cw$}", l));
    }
    h.push_str(&format!("{:>cw$}", "Total"));
    out.push(h);
    out.push(format!("{}-+-{}", "-".repeat(lw), "-".repeat(total_w)));
    for i in 0..k {
        let mut s = format!("{:>lw$} |", labels[i]);
        for j in 0..k {
            s.push_str(&format!("{:>cw$}", fmt_g(t[i][j], 9)));
        }
        s.push_str(&format!("{:>cw$}", fmt_g(rs[i], 9)));
        out.push(s);
    }
    out.push(format!("{}-+-{}", "-".repeat(lw), "-".repeat(total_w)));
    let mut s = format!("{:>lw$} |", "Total");
    for c in cs {
        s.push_str(&format!("{:>cw$}", fmt_g(*c, 9)));
    }
    s.push_str(&format!("{:>cw$}", fmt_g(n, 9)));
    out.push(s);
    out
}

// ---------------------------------------------------------------------------
// stem
// ---------------------------------------------------------------------------

pub struct Stem;

impl CommandHandler for Stem {
    fn name(&self) -> &'static str {
        "stem"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (v, c) = col_of(ds, cmd.main.trim())?;
        let mask = util::touse(ds, st, cmd)?;
        let mut x: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect();
        if x.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let mut out = Vec::new();
        let lab = var_label(ds, &v);
        out.push(String::new());
        out.push(if lab.is_empty() { format!("Stem-and-leaf plot for {v}") } else { format!("Stem-and-leaf plot for {v} ({lab})") });
        let digits: usize = num_opt(cmd, "digits", 1)?.map_or(1, |d| d.max(1.0) as usize);
        // unit of the leaves: 10^p
        let mut p: i32;
        if let Some(r) = num_opt(cmd, "round", 1)? {
            for y in x.iter_mut() {
                *y = (*y / r).round() * r;
            }
            out.push(format!("{v} rounded to nearest multiple of {}", fmt_g(r, 8)));
            p = r.log10().round() as i32;
        } else {
            // smallest unit for which the data are integral
            p = 0;
            while p > -8 && x.iter().any(|y| ((y / 10f64.powi(p)) - (y / 10f64.powi(p)).round()).abs() > 1e-6) {
                p -= 1;
            }
            if num_opt(cmd, "lines", 1)?.is_none() && cmd.opt("width", 1).is_none() && digits == 1 {
                // coarsen until a one-line-per-stem plot fits the target length
                let target = (10.0 * (x.len() as f64).log10()).floor().max(1.0);
                loop {
                    let unit = 10f64.powi(p + 1);
                    let lo = (x.iter().cloned().fold(f64::INFINITY, f64::min) / unit).floor();
                    let hi = (x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / unit).floor();
                    if hi - lo + 1.0 <= target || p > 15 {
                        break;
                    }
                    p += 1;
                }
                if p > 0 {
                    let r = 10f64.powi(p);
                    for y in x.iter_mut() {
                        *y = (*y / r).round() * r;
                    }
                    out.push(format!("{v} rounded to nearest multiple of {}", fmt_g(r, 8)));
                }
            }
        }
        if p != 0 {
            out.push(format!("plot in units of {}", fmt_g(10f64.powi(p), 8)));
        }
        out.push(String::new());
        // integer values on the leaf scale
        let iv: Vec<i64> = x.iter().map(|y| (y / 10f64.powi(p)).round() as i64).collect();
        let base = 10i64.pow(digits as u32);
        let stem_of = |q: i64| -> i64 { q.div_euclid(base) };
        let (lo, hi) = (*iv.iter().min().unwrap(), *iv.iter().max().unwrap());
        let lines: usize = match (num_opt(cmd, "lines", 1)?, num_opt(cmd, "width", 1)?) {
            (Some(l), _) => l as usize,
            (None, Some(w)) => ((base as f64) / w).round().max(1.0) as usize,
            (None, None) => {
                if digits > 1 {
                    1
                } else {
                    let target = (10.0 * (x.len() as f64).log10()).floor().max(1.0) as i64;
                    let mut best = 1;
                    for l in [2usize, 5] {
                        let width = 10 / l as i64;
                        let count = hi.div_euclid(width) - lo.div_euclid(width) + 1;
                        if count <= target {
                            best = l;
                        }
                    }
                    best
                }
            }
        };
        if !matches!(lines, 1 | 2 | 5) || (lines > 1 && digits > 1) {
            return Err(rc_err(198, "lines() must be 1, 2, or 5 (1 when digits() > 1)"));
        }
        let width = base / lines as i64;
        let mark = |k: i64| -> String {
            // k: line index within the stem
            match lines {
                1 => "*".repeat(digits),
                2 => if k == 0 { "*".into() } else { ".".into() },
                _ => ["*", "t", "f", "s", "."][k as usize].to_string(),
            }
        };
        let first_line = lo.div_euclid(width);
        let last_line = hi.div_euclid(width);
        let swidth = {
            let a = stem_of(lo).to_string().len();
            let b = stem_of(hi).to_string().len();
            a.max(b) + 2
        };
        let mut sorted = iv.clone();
        sorted.sort();
        for li in first_line..=last_line {
            let s = (li * width).div_euclid(base);
            let k = (li * width).rem_euclid(base) / width;
            let leaves: Vec<i64> = sorted.iter().filter(|&&q| q.div_euclid(width) == li).map(|&q| q.rem_euclid(base)).collect();
            let txt: Vec<String> = leaves.iter().map(|l| format!("{:0digits$}", l)).collect();
            let joined = if digits > 1 { txt.join(",") } else { txt.concat() };
            let prefix = format!("{:>swidth$}{}   ", s, mark(k));
            let avail = 79usize.saturating_sub(prefix.len());
            let body = if joined.len() > avail {
                let mut cut = String::new();
                let sepw = if digits > 1 { 1 } else { 0 };
                let tail = format!(" ... ({})", leaves.len());
                for (i, t) in txt.iter().enumerate() {
                    let add = t.len() + if i > 0 { sepw } else { 0 };
                    if cut.len() + add + tail.len() + sepw > avail {
                        break;
                    }
                    if i > 0 && digits > 1 {
                        cut.push(',');
                    }
                    cut.push_str(t);
                }
                if digits > 1 {
                    cut.push(',');
                }
                format!("{cut}{tail}")
            } else {
                joined
            };
            out.push(format!("{prefix}{body}").trim_end().to_string());
        }
        let mut r = Results::new();
        r.set("width", width as f64);
        r.set("digits", digits as f64);
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// smooth
// ---------------------------------------------------------------------------

pub struct Smooth;

impl CommandHandler for Smooth {
    fn name(&self) -> &'static str {
        "smooth"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // `smooth 4253eh,twice var, gen(new)`: the comma inside the smoother
        // specification splits the command early, so reparse the raw line
        let raw = cmd.raw.trim();
        let rest = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim_start();
        let (spec, rest) = rest.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "smoother and variable required"))?;
        let (spec, twice) = match spec.split_once(',') {
            Some((a, b)) if b.eq_ignore_ascii_case("twice") => (a.to_string(), true),
            Some((a, b)) if b.is_empty() => (a.to_string(), rest.trim_start().to_lowercase().starts_with("twice")),
            _ => (spec.to_string(), false),
        };
        let rest = if twice && rest.trim_start().to_lowercase().starts_with("twice") { rest.trim_start()[5..].to_string() } else { rest.to_string() };
        let c2 = ferrostat_parser::parser::parse_line(&format!("smooth {}", rest.trim())).map_err(|e| rc_err(198, e.message))?;
        let (v, c) = col_of(ds, c2.main.trim())?;
        let genv = c2.opt("generate", 1).ok_or_else(|| rc_err(198, "option generate() required"))?.trim().to_string();
        crate::varlist::check_name(&genv)?;
        if ds.var_index.contains_key(&genv) {
            return Err(rc_err(110, format!("variable {genv} already defined")));
        }
        let mask = util::touse(ds, st, &c2)?;
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i]).collect();
        // leading and trailing missing values are ignored; inner ones are not allowed
        let first = rows.iter().position(|&i| !is_missing(c[i]));
        let last = rows.iter().rposition(|&i| !is_missing(c[i]));
        let (Some(a), Some(b)) = (first, last) else { return Err(rc_err(2000, "no observations")) };
        let use_rows = &rows[a..=b];
        if use_rows.iter().any(|&i| is_missing(c[i])) {
            return Err(rc_err(416, format!("missing values encountered in {v}")));
        }
        let x: Vec<f64> = use_rows.iter().map(|&i| c[i]).collect();
        let sm = crate::tssmooth::tukey(&x, &spec.to_uppercase())?;
        let sm = if twice {
            let r: Vec<f64> = x.iter().zip(&sm).map(|(a, b)| a - b).collect();
            let rs = crate::tssmooth::tukey(&r, &spec.to_uppercase())?;
            sm.iter().zip(&rs).map(|(a, b)| a + b).collect()
        } else {
            sm
        };
        let mut vals = vec![SYSMISS; ds.n_obs];
        for (k, &i) in use_rows.iter().enumerate() {
            vals[i] = sm[k];
        }
        ds.add_variable(util::new_double(&genv, vals)).map_err(|e| rc_err(110, e))?;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// vwls
// ---------------------------------------------------------------------------

pub struct Vwls;

impl CommandHandler for Vwls {
    fn name(&self) -> &'static str {
        "vwls"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::r_trunc::replay(st, "vwls");
        }
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let sdv = cmd.opt("sd", 2).map(|s| xm::var_name(ds, s)).transpose()?;
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "f", extra: sdv.iter().cloned().collect() })?;
        let act = d.active();
        let k = act.len();
        let fw: Vec<f64> = d.w.clone();
        let nobs: f64 = fw.iter().sum();
        // observations (or covariate-pattern groups): x, y, variance of y
        let mut units: Vec<(Vec<f64>, f64, f64)> = Vec::new();
        if let Some(s) = &sdv {
            let sc = util::col(ds, s)?;
            for t in 0..d.n() {
                let sv = sc[d.rows[t]];
                if !(sv > 0.0) {
                    return Err(rc_err(400, format!("{s} must be positive")));
                }
                for _ in 0..(fw[t].round() as usize).max(1) {
                    units.push((act.iter().map(|&j| d.cols[j][t]).collect(), d.y[t], sv * sv));
                }
            }
        } else {
            let mut groups: Vec<(Vec<f64>, Vec<(f64, f64)>)> = Vec::new();
            for t in 0..d.n() {
                let xr: Vec<f64> = act.iter().map(|&j| d.cols[j][t]).collect();
                match groups.iter_mut().find(|g| g.0 == xr) {
                    Some(g) => g.1.push((d.y[t], fw[t])),
                    None => groups.push((xr, vec![(d.y[t], fw[t])])),
                }
            }
            for (xr, ys) in groups {
                let n: f64 = ys.iter().map(|x| x.1).sum();
                if n < 2.0 {
                    return Err(rc_err(399, "at least two observations per covariate pattern are required"));
                }
                let m = ys.iter().map(|x| x.0 * x.1).sum::<f64>() / n;
                let var = ys.iter().map(|x| x.1 * (x.0 - m).powi(2)).sum::<f64>() / (n - 1.0);
                if !(var > 0.0) {
                    return Err(rc_err(399, "zero variance within a covariate pattern"));
                }
                units.push((xr, m, var / n));
            }
        }
        let mut xtx = Matrix::zeros(k, k);
        let mut xty = vec![0.0; k];
        for (x, y, v) in &units {
            for a in 0..k {
                xty[a] += x[a] * y / v;
                for b in 0..k {
                    xtx[(a, b)] += x[a] * x[b] / v;
                }
            }
        }
        let (vinv, _) = xtx.sym_inverse(1e-12);
        let ba: Vec<f64> = (0..k).map(|a| (0..k).map(|b| vinv[(a, b)] * xty[b]).sum()).collect();
        let gof: f64 = units.iter().map(|(x, y, v)| (y - x.iter().zip(&ba).map(|(a, b)| a * b).sum::<f64>()).powi(2) / v).sum();
        let df_gof = units.len() as f64 - k as f64;
        let kf = d.k();
        let mut b = vec![0.0; kf];
        for (a, &j) in act.iter().enumerate() {
            b[j] = ba[a];
        }
        let v = est::expand_v(&vinv, &act, kf);
        let (chi2, df_m) = est::wald_all(&d, &b, &v);
        let level = util::level(cmd, st)?;
        let hrow = crate::choice_mixed::hrow;
        let mut out = vec![String::new()];
        out.push(hrow("Variance-weighted least-squares regression", "Number of obs", util::commas(nobs as usize)));
        out.push(hrow(&format!("{:<27}= {:>7}", format!("Goodness-of-fit chi2({})", fmt_g(df_gof, 8)), xm::f2(gof)), &format!("Model chi2({})", fmt_g(df_m, 8)), xm::f2(chi2)));
        out.push(hrow(&format!("{:<27}= {:>7}", "Prob > chi2", xm::f4(dist::chi2tail(df_gof, gof))), "Prob > chi2", xm::f4(dist::chi2tail(df_m, chi2))));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        est::post(st, "vwls", &d, &b, &v, &|e: &mut Results| {
            e.set("N", nobs);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("chi2_gf", gof);
            e.set("df_gf", df_gof);
            e.set_str("title", "Variance-weighted least-squares regression");
            e.set_str("predict", "vwls_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// ratio
// ---------------------------------------------------------------------------

pub struct Ratio;

impl CommandHandler for Ratio {
    fn name(&self) -> &'static str {
        "ratio"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::r_trunc::replay(st, "ratio");
        }
        // specifications: [name:] y/x  or  (name: y/x)
        let mut specs: Vec<(String, String, String)> = Vec::new();
        let text = cmd.main.replace('(', " ( ").replace(')', " ) ").replace(':', " : ").replace('/', " / ");
        let toks: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;
        while i < toks.len() {
            if toks[i] == "(" || toks[i] == ")" {
                i += 1;
                continue;
            }
            let mut name = None;
            if i + 1 < toks.len() && toks[i + 1] == ":" {
                name = Some(toks[i].to_string());
                i += 2;
            }
            if i + 2 >= toks.len() || toks[i + 1] != "/" {
                return Err(rc_err(198, "invalid ratio specification"));
            }
            let y = xm::var_name(ds, toks[i])?;
            let x = xm::var_name(ds, toks[i + 2])?;
            let nm = name.unwrap_or_else(|| format!("_ratio_{}", specs.len() + 1));
            specs.push((nm, y, x));
            i += 3;
        }
        if specs.is_empty() {
            return Err(rc_err(198, "ratio specification required"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "pfi")?;
        let all: Vec<String> = specs.iter().flat_map(|s| [s.1.clone(), s.2.clone()]).collect();
        util::markout(ds, &mut mask, &all);
        let rows = util::rows(&mask);
        let n = rows.len() as f64;
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let wt = |i: usize| w.as_ref().map_or(1.0, |w| w.w[i]);
        let fw = w.as_ref().is_some_and(|w| w.kind == util::WeightKind::F);
        let nn: f64 = if fw { rows.iter().map(|&i| wt(i)).sum() } else { n };
        let q = specs.len();
        let mut est_r = Vec::new();
        let mut z: Vec<Vec<f64>> = Vec::new();
        for (_, y, x) in &specs {
            let yc = util::col(ds, y)?;
            let xc = util::col(ds, x)?;
            let sy: f64 = rows.iter().map(|&i| wt(i) * yc[i]).sum();
            let sx: f64 = rows.iter().map(|&i| wt(i) * xc[i]).sum();
            if sx == 0.0 {
                return Err(rc_err(459, "denominator sums to zero"));
            }
            let r = sy / sx;
            est_r.push(r);
            z.push(rows.iter().map(|&i| wt(i) * (yc[i] - r * xc[i]) / sx).collect());
        }
        // linearized variance with the n/(n-1) correction
        let mut v = Matrix::zeros(q, q);
        for a in 0..q {
            let ma = z[a].iter().sum::<f64>() / n;
            for b in 0..q {
                let mb = z[b].iter().sum::<f64>() / n;
                let s: f64 = if fw {
                    rows.iter().enumerate().map(|(t, &i)| (z[a][t] / wt(i) - ma / wt(i)) * (z[b][t] / wt(i) - mb / wt(i)) * wt(i)).sum()
                } else {
                    (0..rows.len()).map(|t| (z[a][t] - ma) * (z[b][t] - mb)).sum()
                };
                v[(a, b)] = s * nn / (nn - 1.0);
            }
        }
        let df_r = nn - 1.0;
        let level = util::level(cmd, st)?;
        let tcrit = dist::invttail(df_r, (1.0 - level / 100.0) / 2.0);
        let mut out = vec![String::new(), format!("{:<44}Number of obs = {}", "Ratio estimation", util::commas(nn as usize)), String::new()];
        for (nm, y, x) in &specs {
            out.push(format!("{:>13}: {y}/{x}", nm));
        }
        out.push(String::new());
        out.push(format!("{}", "-".repeat(62)));
        out.push(format!("{:>12} |             Linearized", ""));
        out.push(format!("{:>12} |      Ratio   std. err.     [{}% conf. interval]", "", fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(48)));
        for (a, (nm, _, _)) in specs.iter().enumerate() {
            let se = v[(a, a)].max(0.0).sqrt();
            out.push(format!("{:>12} | {:>10}  {:>9}     {:>9}   {:>9}", crate::summarize::abbrev(nm, 12), fmt_g(est_r[a], 8), fmt_g(se, 8), fmt_g(est_r[a] - tcrit * se, 8), fmt_g(est_r[a] + tcrit * se, 8)));
        }
        out.push("-".repeat(62));
        let text = out.join("\n");
        let names: Vec<String> = specs.iter().map(|s| s.0.clone()).collect();
        // r(table)
        let mut tm = Matrix::zeros(9, q);
        for a in 0..q {
            let se = v[(a, a)].max(0.0).sqrt();
            let t = est_r[a] / se;
            let vals = [est_r[a], se, t, 2.0 * dist::ttail(df_r, t.abs()), est_r[a] - tcrit * se, est_r[a] + tcrit * se, df_r, tcrit, 0.0];
            for (r, x) in vals.iter().enumerate() {
                tm[(r, a)] = *x;
            }
        }
        est::stash_table(tm.with_names(["b", "se", "t", "pvalue", "ll", "ul", "df", "crit", "eform"].iter().map(|s| s.to_string()).collect(), names.clone()), level);
        let e = st.begin_e();
        e.set("N", nn);
        e.set("df_r", df_r);
        e.set("k_eq", q as f64);
        e.set("rank", q as f64);
        e.set_str("cmd", "ratio");
        e.set_str("title", "Ratio estimation");
        e.set_str("varlist", all.join(" "));
        e.set_str("namelist", names.join(" "));
        e.set_str("vce", "linearized");
        e.set_str("vcetype", "Linearized");
        e.set_str("properties", "b V");
        e.set_mat("b", Matrix::row_vector(&est_r).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.set_mat("_N", Matrix::row_vector(&vec![nn; q]));
        e.sample = Some(mask);
        e.set_str("_output", text.clone());
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// tetrachoric
// ---------------------------------------------------------------------------

pub struct Tetrachoric;

/// Bivariate normal probability P(X <= h, Y <= k) with correlation r.
fn bvn(h: f64, k: f64, r: f64) -> f64 {
    dist::binormal(h, k, r)
}

/// Tetrachoric correlation of a 2x2 table [[n00, n01], [n10, n11]]
/// (0 = first category): (rho, standard error).
pub fn tetra(t: [[f64; 2]; 2], edwards: bool) -> (f64, f64) {
    let n = t[0][0] + t[0][1] + t[1][0] + t[1][1];
    let (a, b, c, d) = (t[1][1], t[1][0], t[0][1], t[0][0]);
    if edwards || [a, b, c, d].iter().any(|x| *x == 0.0) {
        if b * c == 0.0 {
            return (1.0, SYSMISS);
        }
        if a * d == 0.0 {
            return (-1.0, SYSMISS);
        }
        // Edwards and Edwards (1984) approximation
        let alpha = ((a * d) / (b * c)).powf(std::f64::consts::PI / 4.0);
        let rho = (alpha - 1.0) / (alpha + 1.0);
        let se = std::f64::consts::PI / 2.0 * alpha / (alpha + 1.0).powi(2) * (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d).sqrt();
        return (rho, se);
    }
    let p00 = t[0][0] / n;
    let h = dist::invnormal((t[0][0] + t[0][1]) / n);
    let k = dist::invnormal((t[0][0] + t[1][0]) / n);
    // solve Phi2(h, k, rho) = p00 by bisection (monotone in rho)
    let (mut lo, mut hi) = (-0.999999, 0.999999);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if bvn(h, k, mid) < p00 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let rho = 0.5 * (lo + hi);
    let phi2 = (-(h * h - 2.0 * rho * h * k + k * k) / (2.0 * (1.0 - rho * rho))).exp() / (2.0 * std::f64::consts::PI * (1.0 - rho * rho).sqrt());
    let ps = [t[0][0] / n, t[0][1] / n, t[1][0] / n, t[1][1] / n];
    let se = (1.0 / (n * phi2 * phi2) / ps.iter().map(|p| 1.0 / p).sum::<f64>()).sqrt();
    (rho, se)
}

fn fisher2(t: [[f64; 2]; 2]) -> f64 {
    let (a, b, c, d) = (t[0][0], t[0][1], t[1][0], t[1][1]);
    let (r1, c1, n) = (a + b, a + c, a + b + c + d);
    let lo = (r1 + c1 - n).max(0.0) as i64;
    let hi = r1.min(c1) as i64;
    let lf = dist::lnfactorial;
    let p = |x: f64| (lf(r1) + lf(n - r1) + lf(c1) + lf(n - c1) - lf(n) - lf(x) - lf(r1 - x) - lf(c1 - x) - lf(n - r1 - c1 + x)).exp();
    let p0 = p(a);
    (lo..=hi).map(|x| p(x as f64)).filter(|px| *px <= p0 * (1.0 + 1e-7)).sum::<f64>().min(1.0)
}

impl CommandHandler for Tetrachoric {
    fn name(&self) -> &'static str {
        "tetrachoric"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "f")?;
        let pairwise = cmd.has_opt("pw", 2);
        if !pairwise {
            util::markout(ds, &mut mask, &vars);
        }
        let edwards = cmd.has_opt("edwards", 2);
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        for (v, c) in vars.iter().zip(&cols) {
            let mut lv: Vec<f64> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(c[i])).map(|i| c[i]).collect();
            lv.sort_by(f64::total_cmp);
            lv.dedup();
            if lv.len() > 2 {
                return Err(rc_err(198, format!("{v} is not a binary variable")));
            }
        }
        let p = vars.len();
        let mut rho = Matrix::identity(p);
        let mut se_m = Matrix::zeros(p, p);
        let mut nm = Matrix::zeros(p, p);
        let mut last = (0.0, 0.0, 0.0, 0.0);
        for a in 0..p {
            for b in 0..a {
                let mut t = [[0.0; 2]; 2];
                let mut n = 0.0;
                for i in 0..ds.n_obs {
                    if mask[i] && !is_missing(cols[a][i]) && !is_missing(cols[b][i]) {
                        let wt = w.as_ref().map_or(1.0, |w| w.w[i]);
                        t[(cols[a][i] != 0.0) as usize][(cols[b][i] != 0.0) as usize] += wt;
                        n += wt;
                    }
                }
                // rows: first variable of the pair (the earlier one)
                let tt = [[t[0][0], t[1][0]], [t[0][1], t[1][1]]];
                let (r, se) = tetra(tt, edwards);
                rho[(a, b)] = r;
                rho[(b, a)] = r;
                se_m[(a, b)] = se;
                se_m[(b, a)] = se;
                nm[(a, b)] = n;
                nm[(b, a)] = n;
                last = (r, se, n, fisher2(tt));
            }
        }
        let mut out = Vec::new();
        let mut r = Results::new();
        if p == 2 {
            let (rr, se, n, pe) = last;
            out.push(String::new());
            out.push(format!("   Number of obs = {:>12}", util::commas(n as usize)));
            out.push(format!(" Tetrachoric rho = {:>12.4}", rr));
            out.push(format!("       Std error = {:>12}", if is_missing(se) { ".".to_string() } else { format!("{se:.4}") }));
            out.push(String::new());
            out.push(format!("Test of H0: {} and {} are independent", vars[0], vars[1]));
            out.push(format!(" 2-sided exact P = {:>12.4}", pe));
            r.set("rho", rr);
            r.set("se_rho", se);
            r.set("N", n);
            r.set("nstep", 0.0);
            r.set("p", pe);
        } else {
            let n = nm[(1, 0)];
            out.push(format!("(obs={})", util::commas(n as usize)));
            out.push(String::new());
            if !cmd.has_opt("notable", 3) {
                let mut h = format!("{:>12} |", "");
                for v in &vars {
                    h.push_str(&format!(" {:>8}", crate::summarize::abbrev(v, 8)));
                }
                out.push(h);
                out.push(format!("{}+{}", "-".repeat(13), "-".repeat(9 * p)));
                for a in 0..p {
                    let mut s = format!("{:>12} |", crate::summarize::abbrev(&vars[a], 12));
                    for b in 0..=a {
                        s.push_str(&format!("   {:.4}", rho[(a, b)]));
                    }
                    out.push(s);
                }
            }
            r.set("N", n);
            r.set("rho", rho[(p - 1, p - 2)]);
        }
        r.set_mat("Rho", rho.with_names(vars.clone(), vars.clone()));
        r.set_mat("Se_Rho", se_m.with_names(vars.clone(), vars.clone()));
        st.r = r;
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// dydx / integ
// ---------------------------------------------------------------------------

pub struct Dydx {
    pub integ: bool,
}

/// Derivative at x[0] of the quadratic through the first three points.
fn quad_deriv(x0: f64, x1: f64, x2: f64, y0: f64, y1: f64, y2: f64) -> f64 {
    let (h1, h2) = (x1 - x0, x2 - x0);
    // y = y0 + b t + c t^2, through (h1, y1) and (h2, y2)
    let c = ((y2 - y0) / h2 - (y1 - y0) / h1) / (h2 - h1);
    (y1 - y0) / h1 - c * h1
}

/// Cubic spline second derivatives at the knots; the end slopes are those
/// of quadratics through the three points at each end.
fn spline_m(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let d0 = quad_deriv(x[0], x[1], x[2], y[0], y[1], y[2]);
    let dn = quad_deriv(x[n - 1], x[n - 2], x[n - 3], y[n - 1], y[n - 2], y[n - 3]);
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut r = vec![0.0; n];
    let h0 = x[1] - x[0];
    b[0] = h0 / 3.0;
    c[0] = h0 / 6.0;
    r[0] = (y[1] - y[0]) / h0 - d0;
    for i in 1..n - 1 {
        let h0 = x[i] - x[i - 1];
        let h1 = x[i + 1] - x[i];
        a[i] = h0 / 6.0;
        b[i] = (h0 + h1) / 3.0;
        c[i] = h1 / 6.0;
        r[i] = (y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0;
    }
    let hn = x[n - 1] - x[n - 2];
    a[n - 1] = hn / 6.0;
    b[n - 1] = hn / 3.0;
    r[n - 1] = dn - (y[n - 1] - y[n - 2]) / hn;
    for i in 1..n {
        let f = a[i] / b[i - 1];
        b[i] -= f * c[i - 1];
        r[i] -= f * r[i - 1];
    }
    let mut m = vec![0.0; n];
    m[n - 1] = r[n - 1] / b[n - 1];
    for i in (0..n - 1).rev() {
        m[i] = (r[i] - c[i] * m[i + 1]) / b[i];
    }
    m
}

impl CommandHandler for Dydx {
    fn name(&self) -> &'static str {
        if self.integ { "integ" } else { "dydx" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        if vars.len() != 2 {
            return Err(rc_err(102, "yvar and xvar required"));
        }
        let genv = cmd.opt("generate", 1).map(|s| s.trim().to_string());
        if !self.integ && genv.is_none() {
            return Err(rc_err(198, "option generate() required"));
        }
        let replace = cmd.has_opt("replace", 3);
        if let Some(g) = &genv {
            crate::varlist::check_name(g)?;
            if ds.var_index.contains_key(g) && !replace {
                return Err(rc_err(110, format!("variable {g} already defined")));
            }
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let yc = util::col(ds, &vars[0])?;
        let xc = util::col(ds, &vars[1])?;
        let byv: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => crate::varlist::expand(ds, b)?,
            None => Vec::new(),
        };
        let (ids, _) = if byv.is_empty() { (vec![0usize; ds.n_obs], vec![0]) } else { util::group_ids(ds, &byv, &mask) };
        let ngroups = ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(g, _)| *g).max().map_or(0, |g| g + 1);
        let mut out_vals = vec![SYSMISS; ds.n_obs];
        let trap = cmd.has_opt("trapezoid", 4);
        let init = num_opt(cmd, "initial", 1)?.unwrap_or(0.0);
        let mut total = 0.0;
        let mut npts = 0usize;
        for g in 0..ngroups {
            let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && ids[i] == g).collect();
            rows.sort_by(|&a, &b| xc[a].total_cmp(&xc[b]));
            // average y over tied x values
            let mut x: Vec<f64> = Vec::new();
            let mut y: Vec<f64> = Vec::new();
            let mut cnt: Vec<f64> = Vec::new();
            for &i in &rows {
                if x.last() == Some(&xc[i]) {
                    let k = x.len() - 1;
                    y[k] += yc[i];
                    cnt[k] += 1.0;
                } else {
                    x.push(xc[i]);
                    y.push(yc[i]);
                    cnt.push(1.0);
                }
            }
            for k in 0..y.len() {
                y[k] /= cnt[k];
            }
            let n = x.len();
            npts += n;
            if n < 2 {
                continue;
            }
            let m = if trap { vec![0.0; n] } else { spline_m(&x, &y) };
            let deriv = |k: usize| -> f64 {
                // derivative of the spline at knot k
                if k + 1 < n {
                    let h = x[k + 1] - x[k];
                    (y[k + 1] - y[k]) / h - h * (2.0 * m[k] + m[k + 1]) / 6.0
                } else {
                    let h = x[k] - x[k - 1];
                    (y[k] - y[k - 1]) / h + h * (m[k - 1] + 2.0 * m[k]) / 6.0
                }
            };
            let mut cum = vec![init; n];
            for k in 1..n {
                let h = x[k] - x[k - 1];
                cum[k] = cum[k - 1] + h * (y[k - 1] + y[k]) / 2.0 - h * h * h * (m[k - 1] + m[k]) / 24.0;
            }
            total += cum[n - 1] - init;
            for &i in &rows {
                let k = x.iter().position(|v| *v == xc[i]).unwrap();
                out_vals[i] = if self.integ { cum[k] } else { deriv(k) };
            }
        }
        if let Some(g) = &genv {
            if ds.var_index.contains_key(g) {
                ds.drop_variable(g).map_err(|e| rc_err(110, e))?;
            }
            ds.add_variable(util::new_double(g, out_vals)).map_err(|e| rc_err(110, e))?;
        }
        let mut r = Results::new();
        if self.integ {
            r.set("N_points", npts as f64);
            r.set("integral", total);
            st.r = r;
            return Ok(format!("\nnumber of points = {npts}\n\nintegral         = {}", fmt_g(total, 9)));
        }
        r.set_str("y", vars[0].clone());
        st.r = r;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// fvset
// ---------------------------------------------------------------------------

pub struct Fvset;

impl CommandHandler for Fvset {
    fn name(&self) -> &'static str {
        "fvset"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let sub = sub.to_lowercase();
        let rest = rest.trim();
        let set_char = |ds: &mut DataSet, v: &str, key: &str, val: Option<String>| {
            let e = ds.characteristics.entry(v.to_string()).or_default();
            match val {
                Some(s) => {
                    e.insert(key.to_string(), s);
                }
                None => {
                    e.remove(key);
                }
            }
        };
        match sub.as_str() {
            "base" => {
                let (spec, vl) = rest.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "base specification and varlist required"))?;
                let spec = spec.to_lowercase();
                let val = match spec.as_str() {
                    "default" => None,
                    "first" | "last" | "frequent" | "none" => Some(spec.clone()),
                    s => {
                        let v: f64 = s.parse().map_err(|_| rc_err(198, format!("invalid base specification {s}")))?;
                        if v < 0.0 || v.fract() != 0.0 {
                            return Err(rc_err(198, "base level must be a nonnegative integer"));
                        }
                        Some(fmt_g(v, 12))
                    }
                };
                for v in crate::varlist::expand(ds, vl)? {
                    set_char(ds, &v, "_fv_base", val.clone());
                }
            }
            "design" => {
                let (spec, vl) = rest.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "design specification and varlist required"))?;
                let spec = spec.to_lowercase();
                let val = match spec.as_str() {
                    "default" => None,
                    "asbalanced" | "asobserved" => Some(spec.clone()),
                    s => return Err(rc_err(198, format!("invalid design specification {s}"))),
                };
                for v in crate::varlist::expand(ds, vl)? {
                    set_char(ds, &v, "_fv_design", val.clone());
                }
            }
            "clear" => {
                for v in crate::varlist::expand(ds, rest)? {
                    set_char(ds, &v, "_fv_base", None);
                    set_char(ds, &v, "_fv_design", None);
                }
            }
            "report" => {
                let vars = if rest.is_empty() { ds.varlist() } else { crate::varlist::expand(ds, rest)? };
                let fb = cmd.opt("base", 1).map(|s| s.trim().to_lowercase());
                let fd = cmd.opt("design", 1).map(|s| s.trim().to_lowercase());
                let mut out = vec![String::new(), "-------------------------------".into(), "Variable     Base   Design".into(), "-------------------------------".into()];
                for v in vars {
                    let ch = ds.characteristics.get(&v);
                    let b = ch.and_then(|c| c.get("_fv_base")).cloned();
                    let d = ch.and_then(|c| c.get("_fv_design")).cloned();
                    if b.is_none() && d.is_none() {
                        continue;
                    }
                    if fb.as_ref().is_some_and(|f| b.as_ref() != Some(f)) || fd.as_ref().is_some_and(|f| d.as_ref() != Some(f)) {
                        continue;
                    }
                    out.push(format!("{:<12} {:<6} {}", crate::summarize::abbrev(&v, 12), b.unwrap_or_default(), d.unwrap_or_default()));
                }
                out.push("-------------------------------".into());
                return Ok(out.join("\n"));
            }
            _ => return Err(rc_err(198, format!("fvset {sub} invalid"))),
        }
        Ok(String::new())
    }
}

/// Base level set by `fvset base` for a variable: Some(level) when a
/// specific value applies, given the variable's observed levels.
pub fn fvset_base(ds: &DataSet, var: &str, levels: &[f64], counts: &dyn Fn(f64) -> usize) -> Option<Option<f64>> {
    let spec = ds.characteristics.get(var)?.get("_fv_base")?;
    Some(match spec.as_str() {
        "first" => levels.first().copied(),
        "last" => levels.last().copied(),
        "none" => None,
        "frequent" => {
            let mut best: Option<(usize, f64)> = None;
            for &l in levels {
                let c = counts(l);
                if best.is_none_or(|b| c > b.0) {
                    best = Some((c, l));
                }
            }
            best.map(|b| b.1)
        }
        s => s.parse::<f64>().ok(),
    })
}

// ---------------------------------------------------------------------------
// rwgen
// ---------------------------------------------------------------------------

pub struct Rwgen;

impl CommandHandler for Rwgen {
    fn name(&self) -> &'static str {
        "rwgen"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // rwgen {bsample|jackknife|bootstrap} stub [, reps() cluster() strata() rseed()]
        let mut w = cmd.main.split_whitespace();
        let method = w.next().unwrap_or("").to_lowercase();
        let stub = w.next().ok_or_else(|| rc_err(198, "variable stub required"))?.to_string();
        if let Some(s) = cmd.opt("rseed", 2) {
            let s: u64 = s.trim().parse().map_err(|_| rc_err(198, "rseed() invalid"))?;
            st.rng.borrow_mut().reseed(s);
        }
        let mask = util::touse(ds, st, cmd)?;
        let cl = cmd.opt("cluster", 2).map(|s| crate::varlist::expand(ds, s)).transpose()?.unwrap_or_default();
        let sv = cmd.opt("strata", 3).map(|s| crate::varlist::expand(ds, s)).transpose()?.unwrap_or_default();
        let n = ds.n_obs;
        let (cids, _) = if cl.is_empty() { ((0..n).collect::<Vec<_>>(), vec![]) } else { util::group_ids(ds, &cl, &mask) };
        let (sids, _) = if sv.is_empty() { (vec![0usize; n], vec![]) } else { util::group_ids(ds, &sv, &mask) };
        // clusters per stratum
        let mut strata: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for i in 0..n {
            if mask[i] {
                let e = strata.entry(sids[i]).or_default();
                if !e.contains(&cids[i]) {
                    e.push(cids[i]);
                }
            }
        }
        for v in strata.values_mut() {
            v.sort();
        }
        let mut cols: Vec<Vec<f64>> = Vec::new();
        match method.as_str() {
            "bsample" | "bootstrap" | "bs" => {
                let reps: usize = cmd.opt("reps", 1).map(|s| s.trim().parse().map_err(|_| rc_err(198, "reps() invalid"))).transpose()?.unwrap_or(50);
                for _ in 0..reps {
                    let mut freq: std::collections::HashMap<usize, f64> = Default::default();
                    for cls in strata.values() {
                        let m = cls.len();
                        for _ in 0..m {
                            let u = st.rng.borrow_mut().uniform();
                            let pick = cls[((u * m as f64) as usize).min(m - 1)];
                            *freq.entry(pick).or_default() += 1.0;
                        }
                    }
                    cols.push((0..n).map(|i| if mask[i] { freq.get(&cids[i]).copied().unwrap_or(0.0) } else { SYSMISS }).collect());
                }
            }
            "jackknife" | "jk" => {
                for (s, cls) in &strata {
                    let m = cls.len() as f64;
                    for &c in cls {
                        cols.push((0..n).map(|i| if !mask[i] { SYSMISS } else if sids[i] != *s { 1.0 } else if cids[i] == c { 0.0 } else { m / (m - 1.0) }).collect());
                    }
                }
            }
            _ => return Err(rc_err(198, format!("rwgen {method} invalid"))),
        }
        let names: Vec<String> = (1..=cols.len()).map(|k| format!("{stub}{k}")).collect();
        for nm in &names {
            if ds.var_index.contains_key(nm) {
                return Err(rc_err(110, format!("variable {nm} already defined")));
            }
        }
        for (nm, c) in names.iter().zip(cols) {
            ds.add_variable(util::new_double(nm, c)).map_err(|e| rc_err(110, e))?;
        }
        let mut r = Results::new();
        r.set("reps", names.len() as f64);
        r.set_str("varlist", names.join(" "));
        st.r = r;
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// informational commands
// ---------------------------------------------------------------------------

pub struct Info(pub &'static str);

impl CommandHandler for Info {
    fn name(&self) -> &'static str {
        self.0
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let version = env!("CARGO_PKG_VERSION");
        Ok(match self.0 {
            "about" => format!("\nFerrostat {version}\nStatistics engine reading and writing .do and .dta files.\nLicensed under the terms in the project's LICENSE file."),
            "copyright" => "\nFerrostat is free software; see the LICENSE file distributed with it.\nThird-party components are listed in the README's Provenance section.".into(),
            "more" => String::new(),
            "search" | "sj" => {
                let what = cmd.main.trim();
                if what.is_empty() {
                    String::new()
                } else {
                    format!("(no online search available; try -help {what}-)")
                }
            }
            _ => {
                // query: current settings
                let mut out = vec![String::new(), "-------------------------------------------------------------------------------".to_string(), "    Output settings".to_string()];
                let get = |k: &str, d: &str| st.settings.get(k).cloned().unwrap_or_else(|| d.to_string());
                let rows = [
                    ("more", get("more", "off")),
                    ("rmsg", if st.rmsg { "on".into() } else { "off".into() }),
                    ("dp", get("dp", "period")),
                    ("linesize", get("linesize", "80")),
                    ("cformat", get("cformat", "")),
                    ("pformat", get("pformat", "")),
                    ("sformat", get("sformat", "")),
                    ("level", fmt_g(st.level, 4)),
                    ("showbaselevels", get("showbaselevels", "")),
                    ("showemptycells", get("showemptycells", "")),
                    ("showomitted", get("showomitted", "")),
                    ("fvlabel", get("fvlabel", "on")),
                ];
                for (k, v) in rows {
                    out.push(format!("    {:<20}{}", k, v));
                }
                out.push(String::new());
                out.push("    Interface settings".into());
                out.push(format!("    {:<20}{}", "graphics", if st.graphics_off { "off" } else { "on" }));
                out.push(format!("    {:<20}{}", "scheme", if st.scheme.is_empty() { "default".to_string() } else { st.scheme.clone() }));
                out.push(format!("    {:<20}{}", "varabbrev", if st.varabbrev { "on" } else { "off" }));
                out.push(String::new());
                out.push("    Random-number settings".into());
                out.push(format!("    {:<20}{}", "rng", get("rng", "default")));
                out.push(format!("    {:<20}{}", "rngstream", get("rngstream", "1")));
                out.push(String::new());
                out.push("    Estimation settings".into());
                out.push(format!("    {:<20}{}", "emptycells", get("emptycells", "keep")));
                out.push(format!("    {:<20}{}", "maxiter", get("maxiter", "300")));
                out.push("-------------------------------------------------------------------------------".to_string());
                out.join("\n")
            }
        })
    }
}

/// `translate in out [, replace]`: SMCL logs and output to plain text.
pub struct Translate;

/// Strip SMCL directives, keeping the text they display.
pub fn smcl_to_text(s: &str) -> String {
    let mut out = String::new();
    for line in s.lines() {
        if line.trim_start().starts_with("{smcl}") || line.starts_with("{com}{sf}{ul off}") && line.trim().len() <= 20 {
            continue;
        }
        let mut res = String::new();
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' {
                let mut tag = String::new();
                let mut depth = 1;
                for d in chars.by_ref() {
                    if d == '{' {
                        depth += 1;
                    } else if d == '}' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    tag.push(d);
                }
                // {tag:text} shows its text; {hline n}, {space n}, {col n} expand
                let (name, arg) = tag.split_once(':').map(|(a, b)| (a.to_string(), Some(b.to_string()))).unwrap_or((tag.clone(), None));
                let mut parts = name.split_whitespace();
                let nm = parts.next().unwrap_or("");
                let n: usize = parts.next().and_then(|x| x.parse().ok()).unwrap_or(1);
                match (nm, arg) {
                    (_, Some(t)) => res.push_str(&t),
                    ("hline", None) => res.push_str(&"-".repeat(if tag.trim() == "hline" { 79usize.saturating_sub(res.len()) } else { n })),
                    ("space", None) => res.push_str(&" ".repeat(n)),
                    ("col", None) => {
                        let target = n.saturating_sub(1);
                        if res.len() < target {
                            res.push_str(&" ".repeat(target - res.len()));
                        }
                    }
                    ("c", None) => res.push(match tag.trim_start_matches('c').trim() {
                        "|" => '|',
                        "+" => '+',
                        "-" => '-',
                        _ => '+',
                    }),
                    ("char", None) => res.push(match tag.trim_start_matches("char").trim() {
                        "|" => '|',
                        "+" => '+',
                        "-" => '-',
                        _ => '+',
                    }),
                    ("dup", None) => {}
                    _ => {}
                }
            } else {
                res.push(c);
            }
        }
        out.push_str(res.trim_end());
        out.push('\n');
    }
    out
}

impl CommandHandler for Translate {
    fn name(&self) -> &'static str {
        "translate"
    }
    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words: Vec<String> = ferrostat_parser::parser::split_top_level_words(cmd.main.trim()).into_iter().map(|w| w.trim_matches('"').to_string()).collect();
        if words.len() != 2 {
            return Err(rc_err(198, "translate requires an input and an output file"));
        }
        let (inp, outp) = (&words[0], &words[1]);
        if ferrostat_core::vfs::exists(outp) && !cmd.has_opt("replace", 3) {
            return Err(rc_err(602, format!("file {outp} already exists")));
        }
        let text = ferrostat_core::vfs::read_to_string(inp).map_err(|_| rc_err(601, format!("file {inp} not found")))?;
        let lower = inp.to_lowercase();
        let res = if lower.ends_with(".smcl") || text.trim_start().starts_with("{smcl}") { smcl_to_text(&text) } else { text };
        ferrostat_core::vfs::write(outp, res.as_bytes()).map_err(|e| rc_err(603, format!("file {outp} could not be opened: {e}")))?;
        Ok(format!("(file {outp} written in .txt format)"))
    }
}
