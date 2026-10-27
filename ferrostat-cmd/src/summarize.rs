//! summarize, tabstat, centile, mean/ci.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::{fmt, fmt_g};
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util::{self, WeightKind, Weights};
use crate::varlist;

/// Weighted univariate statistics as computed by `summarize`.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub n: f64,
    pub sum_w: f64,
    pub mean: f64,
    pub var: f64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub x: Vec<(f64, f64)>, // (value, weight) sorted by value
}

pub fn summarize_values(x: &[f64], w: Option<&[f64]>, kind: Option<WeightKind>) -> Summary {
    let mut pairs: Vec<(f64, f64)> = x
        .iter()
        .enumerate()
        .filter(|(_, v)| !is_missing(**v))
        .map(|(i, &v)| (v, w.map_or(1.0, |w| w[i])))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let nobs = pairs.len() as f64;
    if pairs.is_empty() {
        return Summary { n: 0.0, sum_w: 0.0, mean: SYSMISS, var: SYSMISS, min: SYSMISS, max: SYSMISS, sum: 0.0, x: pairs };
    }
    let sw: f64 = pairs.iter().map(|p| p.1).sum();
    let mean = pairs.iter().map(|p| p.0 * p.1).sum::<f64>() / sw;
    let ss: f64 = pairs.iter().map(|p| p.1 * (p.0 - mean).powi(2)).sum();
    let (n, var, sum) = match kind {
        Some(WeightKind::F) => (sw, if sw > 1.0 { ss / (sw - 1.0) } else { SYSMISS }, pairs.iter().map(|p| p.0 * p.1).sum()),
        Some(WeightKind::A) | Some(WeightKind::P) => {
            // weights normalised to sum to N
            (nobs, if nobs > 1.0 { ss * nobs / sw / (nobs - 1.0) } else { SYSMISS }, mean * nobs)
        }
        Some(WeightKind::I) => (nobs, if sw > 1.0 { ss / (sw - 1.0) } else { SYSMISS }, pairs.iter().map(|p| p.0 * p.1).sum()),
        None => (nobs, if nobs > 1.0 { ss / (nobs - 1.0) } else { SYSMISS }, pairs.iter().map(|p| p.0).sum()),
    };
    Summary { n, sum_w: sw, mean, var, min: pairs[0].0, max: pairs.last().unwrap().0, sum, x: pairs }
}

impl Summary {
    pub fn sd(&self) -> f64 {
        if is_missing(self.var) { SYSMISS } else { self.var.sqrt() }
    }

    /// Percentile (unweighted definition when all weights are 1).
    pub fn pctile(&self, p: f64) -> f64 {
        if self.x.iter().all(|x| x.1 == 1.0) {
            let s: Vec<f64> = self.x.iter().map(|x| x.0).collect();
            util::pctile_sorted(&s, p)
        } else {
            let mut v = self.x.clone();
            util::pctile_weighted(&mut v, p)
        }
    }

    pub fn moments(&self) -> (f64, f64) {
        let sw = self.sum_w;
        if self.x.len() < 2 {
            return (SYSMISS, SYSMISS);
        }
        let m2 = self.x.iter().map(|p| p.1 * (p.0 - self.mean).powi(2)).sum::<f64>() / sw;
        let m3 = self.x.iter().map(|p| p.1 * (p.0 - self.mean).powi(3)).sum::<f64>() / sw;
        let m4 = self.x.iter().map(|p| p.1 * (p.0 - self.mean).powi(4)).sum::<f64>() / sw;
        if m2 == 0.0 {
            return (SYSMISS, SYSMISS);
        }
        (m3 / m2.powf(1.5), m4 / (m2 * m2))
    }
}

fn g9(v: f64) -> String {
    fmt_g(v, 8)
}

pub struct Summarize;

impl CommandHandler for Summarize {
    fn name(&self) -> &'static str {
        "summarize"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "faiw")?;
        let detail = cmd.has_opt("detail", 1);
        let meanonly = cmd.has_opt("meanonly", 5);
        let mut out = Vec::new();
        if !detail && !meanonly {
            out.push("    Variable |        Obs        Mean    Std. dev.       Min        Max".to_string());
            out.push("-------------+---------------------------------------------------------".to_string());
        }
        let mut last: Option<Summary> = None;
        for (k, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            if var.var_type.is_string() {
                if !detail && !meanonly {
                    out.push(format!("{:>12} |{:>11}", abbrev(v, 12), 0));
                }
                last = Some(Summary { n: 0.0, ..Default::default() });
                continue;
            }
            let x: Vec<f64> = (0..ds.n_obs).map(|i| if mask[i] { var.num(i) } else { SYSMISS }).collect();
            let s = summarize_values(&x, w.as_ref().map(|w| w.w.as_slice()), w.as_ref().map(|w| w.kind));
            if detail {
                if k > 0 {
                    out.push(String::new());
                }
                out.push(detail_table(v, &var.label, &s));
            } else if !meanonly {
                if s.n == 0.0 {
                    out.push(format!("{:>12} |{:>11}", abbrev(v, 12), 0));
                } else {
                    out.push(format!(
                        "{:>12} |{:>11} {:>11} {:>11} {:>10} {:>10}",
                        abbrev(v, 12),
                        g9(s.n),
                        g9(s.mean),
                        g9(s.sd()),
                        g9(s.min),
                        g9(s.max)
                    ));
                }
            }
            last = Some(s);
        }
        let r = st.begin_r();
        if let Some(s) = last {
            r.set("N", s.n);
            r.set("sum_w", s.sum_w);
            if s.n > 0.0 {
                r.set("mean", s.mean);
                r.set("min", s.min);
                r.set("max", s.max);
                r.set("sum", s.sum);
                if !meanonly {
                    r.set("Var", s.var);
                    r.set("sd", s.sd());
                }
                if detail {
                    for p in [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0] {
                        r.set(&format!("p{}", p as i32), s.pctile(p));
                    }
                    let (sk, ku) = s.moments();
                    r.set("skewness", sk);
                    r.set("kurtosis", ku);
                }
            }
        }
        Ok(if meanonly { String::new() } else { out.join("\n") })
    }
}

pub fn abbrev(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n - 1).collect();
        format!("{head}~")
    }
}

fn detail_table(name: &str, label: &str, s: &Summary) -> String {
    let title = if label.is_empty() { name } else { label };
    let mut out = vec![format!("{:^61}", title), "-".repeat(61)];
    out.push("      Percentiles      Smallest".to_string());
    let small: Vec<f64> = s.x.iter().map(|p| p.0).take(4).collect();
    let large: Vec<f64> = s.x.iter().rev().map(|p| p.0).take(4).collect();
    let g = |v: Option<&f64>| v.map(|x| g9(*x)).unwrap_or_default();
    let (sk, ku) = s.moments();
    let p = |q: f64| g9(s.pctile(q));
    out.push(format!(" 1%    {:>9}      {:>9}", p(1.0), g(small.first())));
    out.push(format!(" 5%    {:>9}      {:>9}", p(5.0), g(small.get(1))));
    out.push(format!("10%    {:>9}      {:>9}       Obs           {:>11}", p(10.0), g(small.get(2)), g9(s.n)));
    out.push(format!("25%    {:>9}      {:>9}       Sum of wgt.   {:>11}", p(25.0), g(small.get(3)), g9(s.sum_w)));
    out.push(String::new());
    out.push(format!("50%    {:>9}                      Mean          {:>11}", p(50.0), g9(s.mean)));
    out.push(format!("                        Largest       Std. dev.     {:>11}", g9(s.sd())));
    out.push(format!("75%    {:>9}      {:>9}", p(75.0), g(large.get(3))));
    out.push(format!("90%    {:>9}      {:>9}       Variance      {:>11}", p(90.0), g(large.get(2)), g9(s.var)));
    out.push(format!("95%    {:>9}      {:>9}       Skewness      {:>11}", p(95.0), g(large.get(1)), g9(sk)));
    out.push(format!("99%    {:>9}      {:>9}       Kurtosis      {:>11}", p(99.0), g(large.first()), g9(ku)));
    out.join("\n")
}

// ---- tabstat ------------------------------------------------------------------

fn stat_of(name: &str, s: &Summary) -> Result<f64, String> {
    Ok(match name {
        "mean" => s.mean,
        "count" | "n" => s.n,
        "sum" => s.sum,
        "max" => s.max,
        "min" => s.min,
        "range" | "r" => s.max - s.min,
        "sd" => s.sd(),
        "variance" | "var" => s.var,
        "cv" => s.sd() / s.mean,
        "semean" => s.sd() / s.n.sqrt(),
        "skewness" => s.moments().0,
        "kurtosis" => s.moments().1,
        "median" | "p50" => s.pctile(50.0),
        "iqr" => s.pctile(75.0) - s.pctile(25.0),
        "q" => s.pctile(50.0),
        p if p.starts_with('p') => s.pctile(p[1..].parse().map_err(|_| rc_err(198, format!("{p} invalid statistic")))?),
        other => return Err(rc_err(198, format!("{other} invalid statistic"))),
    })
}

pub struct Tabstat;

impl CommandHandler for Tabstat {
    fn name(&self) -> &'static str {
        "tabstat"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fa")?;
        let stats: Vec<String> = cmd
            .opt("statistics", 1)
            .map(|s| s.split_whitespace().map(|x| x.to_lowercase()).collect())
            .unwrap_or_else(|| vec!["mean".into()]);
        let fmtspec = cmd.opt("format", 1).map(|f| if f.trim().is_empty() { "%9.0g".to_string() } else { f.trim().to_string() });
        let by = cmd.opt("by", 2).map(|b| varlist::resolve_name(ds, b.trim())).transpose()?;
        let cols_are_stats = cmd.opt("columns", 1).map_or(false, |c| c.trim().starts_with('s'));
        let cell = |v: f64| match &fmtspec {
            Some(f) => fmt(v, f).trim().to_string(),
            None => g9(v),
        };
        let summarize_mask = |m: &[bool], v: &str| -> Result<Summary, String> {
            let x: Vec<f64> = (0..ds.n_obs).map(|i| if m[i] { ds.get_var(v).unwrap().num(i) } else { SYSMISS }).collect();
            Ok(summarize_values(&x, w.as_ref().map(|w| w.w.as_slice()), w.as_ref().map(|w| w.kind)))
        };
        let mut out = Vec::new();
        let r = crate::store::Results::new();
        st.r = r;
        let mut groups: Vec<(String, Vec<bool>)> = Vec::new();
        if let Some(b) = &by {
            let (ids, reps) = util::group_ids(ds, &[b.clone()], &mask);
            let bv = ds.get_var(b).unwrap();
            for (g, &rep) in reps.iter().enumerate() {
                let m: Vec<bool> = (0..ds.n_obs).map(|i| ids[i] == g).collect();
                groups.push((util::cell_text(ds, bv, rep, true), m));
            }
        }
        if !cmd.has_opt("nototal", 3) || by.is_none() {
            groups.push(("Total".into(), mask.clone()));
        }
        // header
        let byname = by.clone().unwrap_or_else(|| if cols_are_stats { "Variable".into() } else { "Stats".into() });
        if by.is_none() && !cols_are_stats {
            out.push(format!("{:>9} | {}", "Stats", vars.iter().map(|v| format!("{:>9}", abbrev(v, 9))).collect::<Vec<_>>().join(" ")));
            out.push(format!("{}-+-{}", "-".repeat(9), "-".repeat(vars.len() * 10)));
            for s in &stats {
                let vals: Vec<String> = vars.iter().map(|v| summarize_mask(&mask, v).and_then(|x| stat_of(s, &x)).map(cell)).collect::<Result<_, _>>()?;
                out.push(format!("{:>9} | {}", abbrev(s, 9), vals.iter().map(|x| format!("{x:>9}")).collect::<Vec<_>>().join(" ")));
            }
            let mut m = Matrix::zeros(stats.len(), vars.len()).with_names(stats.clone(), vars.clone());
            for (i, s) in stats.iter().enumerate() {
                for (j, v) in vars.iter().enumerate() {
                    m[(i, j)] = stat_of(s, &summarize_mask(&mask, v)?)?;
                }
            }
            st.r.set_mat("StatTotal", m);
        } else if by.is_some() && !cols_are_stats {
            out.push(format!("{:>9} | {}", abbrev(&byname, 9), vars.iter().map(|v| format!("{:>9}", abbrev(v, 9))).collect::<Vec<_>>().join(" ")));
            out.push(format!("{}-+-{}", "-".repeat(9), "-".repeat(vars.len() * 10)));
            for (gname, m) in &groups {
                if gname == "Total" {
                    out.push(format!("{}-+-{}", "-".repeat(9), "-".repeat(vars.len() * 10)));
                }
                for (k, s) in stats.iter().enumerate() {
                    let vals: Vec<String> = vars.iter().map(|v| summarize_mask(m, v).and_then(|x| stat_of(s, &x)).map(cell)).collect::<Result<_, _>>()?;
                    out.push(format!("{:>9} | {}", if k == 0 { abbrev(gname, 9) } else { String::new() }, vals.iter().map(|x| format!("{x:>9}")).collect::<Vec<_>>().join(" ")));
                }
            }
        } else {
            out.push(format!("{:>12} | {}", "Variable", stats.iter().map(|s| format!("{:>9}", abbrev(s, 9))).collect::<Vec<_>>().join(" ")));
            out.push(format!("{}-+-{}", "-".repeat(12), "-".repeat(stats.len() * 10)));
            for (gname, m) in &groups {
                if by.is_some() {
                    out.push(format!("{gname}"));
                }
                for v in &vars {
                    let s = summarize_mask(m, v)?;
                    let vals: Vec<String> = stats.iter().map(|x| stat_of(x, &s).map(cell)).collect::<Result<_, _>>()?;
                    out.push(format!("{:>12} | {}", abbrev(v, 12), vals.iter().map(|x| format!("{x:>9}")).collect::<Vec<_>>().join(" ")));
                }
            }
        }
        Ok(out.join("\n"))
    }
}

// ---- centile --------------------------------------------------------------------

pub struct Centile;

impl CommandHandler for Centile {
    fn name(&self) -> &'static str {
        "centile"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mask = util::touse(ds, st, cmd)?;
        let cents: Vec<f64> = cmd
            .opt("centile", 1)
            .map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![50.0]);
        let level = util::level(cmd, st)?;
        let mut out = vec![
            format!("{:>51}", "-- Binom. interp. --"),
            format!("    Variable |       Obs  Percentile    Centile        [{}% conf. interval]", fmt_g(level, 4)),
            "-------------+-------------------------------------------------------------".to_string(),
        ];
        let mut r = crate::store::Results::new();
        for v in &vars {
            let mut x: Vec<f64> = util::rows(&mask).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
            x.sort_by(f64::total_cmp);
            let n = x.len();
            for (k, &p) in cents.iter().enumerate() {
                // centile definition: (n+1)p/100 with interpolation
                let pos = (n as f64 + 1.0) * p / 100.0;
                let c = interp(&x, pos);
                let (lo, hi) = centile_ci(&x, p / 100.0, level);
                out.push(format!(
                    "{:>12} |{:>10}{:>12}{:>12}{:>14}{:>12}",
                    if k == 0 { abbrev(v, 12) } else { String::new() },
                    if k == 0 { n.to_string() } else { String::new() },
                    fmt_g(p, 9),
                    g9(c),
                    g9(lo),
                    g9(hi)
                ));
                r.set(&format!("c_{}", k + 1), c);
                r.set(&format!("lb_{}", k + 1), lo);
                r.set(&format!("ub_{}", k + 1), hi);
            }
            r.set("N", n as f64);
        }
        r.set_str("centiles", cents.iter().map(|c| fmt_g(*c, 9)).collect::<Vec<_>>().join(" "));
        st.r = r;
        Ok(out.join("\n"))
    }
}

fn interp(x: &[f64], pos: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return SYSMISS;
    }
    if pos <= 1.0 {
        return x[0];
    }
    if pos >= n as f64 {
        return x[n - 1];
    }
    let i = pos.floor() as usize;
    let f = pos - i as f64;
    x[i - 1] + f * (x[i] - x[i - 1])
}

/// Exact binomial confidence interval for a quantile (with interpolation).
fn centile_ci(x: &[f64], q: f64, level: f64) -> (f64, f64) {
    let n = x.len() as f64;
    if x.is_empty() {
        return (SYSMISS, SYSMISS);
    }
    let a = (1.0 - level / 100.0) / 2.0;
    // find ranks r, s with binomial probabilities
    let mut lo_rank = 0.0;
    for k in 0..=x.len() {
        if dist::binomial(n, k as f64, q) >= a {
            lo_rank = k as f64;
            break;
        }
    }
    let mut hi_rank = n;
    for k in (0..=x.len()).rev() {
        if dist::binomialtail(n, k as f64, q) >= a {
            hi_rank = k as f64 + 1.0;
            break;
        }
    }
    (interp(x, lo_rank.max(1.0)), interp(x, hi_rank.min(n)))
}

// ---- mean / ci ----------------------------------------------------------------------

pub struct MeanCmd;

impl CommandHandler for MeanCmd {
    fn name(&self) -> &'static str {
        "mean"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "fap")?;
        util::markout(ds, &mut mask, &vars);
        // over(varlist [, nolabel]): one mean per variable and group
        let (over, over_labels) = match cmd.opt("over", 4) {
            Some(o) => {
                let (spec, sub) = o.split_once(',').unwrap_or((o, ""));
                (varlist::expand(ds, spec)?, !sub.trim().starts_with("nol"))
            }
            None => (Vec::new(), true),
        };
        for o in &over {
            if ds.get_var(o).is_some_and(|v| v.var_type.is_string()) {
                return Err(rc_err(109, format!("over(): {o} is a string variable")));
            }
        }
        util::markout(ds, &mut mask, &over);
        let level = util::level(cmd, st)?;
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let ww: Vec<f64> = rows.iter().map(|&i| w.as_ref().map_or(1.0, |w| w.w[i])).collect();
        let wkind = w.as_ref().map(|w| w.kind);
        let count = |idx: &[usize]| match wkind {
            Some(WeightKind::F) => idx.iter().map(|&t| ww[t]).sum::<f64>(),
            _ => idx.len() as f64,
        };
        let cols: Vec<Vec<f64>> = vars.iter().map(|v| rows.iter().map(|&i| ds.get_var(v).unwrap().num(i)).collect()).collect();
        let ovals: Vec<Vec<f64>> = rows.iter().map(|&i| over.iter().map(|v| ds.get_var(v).unwrap().num(i)).collect()).collect();
        let mut levels = ovals.clone();
        levels.sort_by(|a, b| a.iter().zip(b).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
        levels.dedup();
        let ng = levels.len();
        let gid: Vec<usize> = ovals.iter().map(|o| levels.iter().position(|l| l == o).unwrap()).collect();
        let k = vars.len();
        let kk = k * ng;
        let mut b = vec![0.0; kk];
        let mut v = Matrix::zeros(kk, kk);
        let mut n_g = vec![0.0; ng];
        for g in 0..ng {
            let idx: Vec<usize> = (0..rows.len()).filter(|&t| gid[t] == g).collect();
            let sw: f64 = idx.iter().map(|&t| ww[t]).sum();
            let n = count(&idx);
            let nobs = idx.len() as f64;
            n_g[g] = n;
            let at = |j: usize| j * ng + g;
            for j in 0..k {
                b[at(j)] = idx.iter().map(|&t| cols[j][t] * ww[t]).sum::<f64>() / sw;
            }
            for a in 0..k {
                for c in 0..k {
                    let (ma, mc) = (b[at(a)], b[at(c)]);
                    let s: f64 = idx.iter().map(|&t| ww[t] * (cols[a][t] - ma) * (cols[c][t] - mc)).sum();
                    v[(at(a), at(c))] = match wkind {
                        Some(WeightKind::P) => {
                            let s2: f64 = idx.iter().map(|&t| ww[t] * ww[t] * (cols[a][t] - ma) * (cols[c][t] - mc)).sum();
                            s2 / (sw * sw) * nobs / (nobs - 1.0)
                        }
                        Some(WeightKind::F) => s / (sw - 1.0) / sw,
                        _ => s * n / sw / (n - 1.0) / n,
                    };
                }
            }
        }
        let n = count(&(0..rows.len()).collect::<Vec<_>>());
        let df = n - 1.0;
        let tcrit = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
        let fmt_level = |l: f64| fmt_g(l, 12);
        // coefficient names: c.var@lev.over[#lev.over2]
        let names: Vec<String> = if over.is_empty() {
            vars.clone()
        } else {
            vars.iter()
                .flat_map(|x| levels.iter().map(move |l| (x, l)))
                .map(|(x, l)| format!("c.{x}@{}", l.iter().zip(&over).map(|(lv, o)| format!("{}.{o}", fmt_level(*lv))).collect::<Vec<_>>().join("#")))
                .collect()
        };
        let group_label = |l: &[f64]| -> String {
            l.iter()
                .zip(&over)
                .map(|(lv, o)| {
                    let var = ds.get_var(o).unwrap();
                    let lab = if over_labels && !var.value_label_name.is_empty() { ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(*lv)).map(String::from) } else { None };
                    lab.unwrap_or_else(|| fmt_level(*lv))
                })
                .collect::<Vec<_>>()
                .join("#")
        };
        // before version 16, groups are named by their value labels with the
        // variable as equation name
        let legacy = !over.is_empty() && st.settings.get("version").and_then(|v| v.parse::<f64>().ok()).is_some_and(|v| v < 16.0);
        let legacy_name = |g: usize| {
            let l = group_label(&levels[g]);
            if l.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') && l.chars().all(|c| c.is_alphanumeric() || c == '_') && l.len() <= 32 {
                l
            } else {
                format!("_subpop_{}", g + 1)
            }
        };
        let names: Vec<String> = if legacy { vars.iter().flat_map(|x| (0..ng).map(move |g| (x, g))).map(|(x, g)| format!("{x}:{}", legacy_name(g))).collect() } else { names };
        let headers: Vec<String> = vars.iter().map(|x| if legacy { x.clone() } else { format!("c.{x}@{}", over.join("#")) }).collect();
        let lw = if over.is_empty() {
            12
        } else {
            headers.iter().map(|h| h.chars().count()).chain(levels.iter().map(|l| group_label(l).chars().count() + 1)).max().unwrap_or(12).clamp(12, 32)
        };
        let mut out = vec![format!("Mean estimation{:>40}Number of obs = {:>7}", "", fmt_g(n, 9)), String::new()];
        let bar = "-".repeat(lw + 50);
        out.push(bar.clone());
        out.push(format!("{:>lw$} |       Mean   Std. err.     [{}% conf. interval]", if legacy { "Over" } else { "" }, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(48)));
        let row = |label: &str, j: usize| {
            let se = v[(j, j)].sqrt();
            format!("{:>lw$} | {:>10} {:>10}  {:>10}  {:>10}", abbrev(label, lw), g9(b[j]), g9(se), g9(b[j] - tcrit * se), g9(b[j] + tcrit * se))
        };
        if over.is_empty() {
            for j in 0..k {
                out.push(row(&vars[j], j));
            }
        } else {
            for (a, h) in headers.iter().enumerate() {
                out.push(format!("{:<lw$} |", abbrev(h, lw)));
                for (g, l) in levels.iter().enumerate() {
                    let label = if legacy { legacy_name(g) } else { format!("{} ", group_label(l)) };
                    out.push(row(&label, a * ng + g));
                }
            }
        }
        out.push(bar);
        let spec_omit = vec![false; kk];
        crate::est::stash_table(
            crate::est::table_matrix(&crate::est::TableSpec { depvar: "", names: &names, b: &b, v: &v, omitted: &spec_omit, is_base: &spec_omit, df: Some(df), level, eform: None, coef_title: "Mean" }),
            level,
        );
        let e = st.begin_e();
        e.set("N", n);
        e.set("df_r", df);
        e.set("k_eq", 1.0);
        e.set("rank", kk as f64);
        e.set_str("cmd", "mean");
        e.set_str("title", "Mean estimation");
        e.set_str("varlist", vars.join(" "));
        e.set_str("depvar", vars.join(" "));
        e.set_str("properties", "b V");
        e.set_str("vce", "analytic");
        let mut nm = Matrix::row_vector(&(0..kk).map(|j| if over.is_empty() { n } else { n_g[j % ng] }).collect::<Vec<_>>());
        nm.colnames = names.clone();
        if !over.is_empty() {
            e.set("N_over", ng as f64);
            e.set_str("over", over.join(" "));
            e.set_str("over_namelist", levels.iter().map(|l| l.iter().zip(&over).map(|(lv, o)| format!("{}.{o}", fmt_level(*lv))).collect::<Vec<_>>().join("#")).collect::<Vec<_>>().join(" "));
            e.set_str("over_labels", levels.iter().map(|l| format!("\"{}\"", group_label(l))).collect::<Vec<_>>().join(" "));
        }
        e.set_mat("_N", nm);
        e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.sample = Some(mask);
        Ok(out.join("\n"))
    }
}

pub struct Ci;

impl CommandHandler for Ci {
    fn name(&self) -> &'static str {
        "ci"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut words = cmd.main.split_whitespace();
        let first = words.next().unwrap_or("");
        let (kind, rest): (&str, Vec<&str>) = match first {
            "means" | "mean" | "proportions" | "prop" | "variances" => (first, words.collect()),
            _ => ("means", cmd.main.split_whitespace().collect()),
        };
        let vars = varlist::expand(ds, &rest.join(" "))?;
        let mask = util::touse(ds, st, cmd)?;
        let level = util::level(cmd, st)?;
        let a = (1.0 - level / 100.0) / 2.0;
        let mut out = vec![
            format!("    Variable |        Obs        Mean    Std. err.       [{}% conf. interval]", fmt_g(level, 4)),
            "-------------+---------------------------------------------------------------".to_string(),
        ];
        let mut r = crate::store::Results::new();
        for v in &vars {
            let x: Vec<f64> = util::rows(&mask).iter().map(|&i| ds.get_var(v).unwrap().num(i)).filter(|x| !is_missing(*x)).collect();
            let n = x.len() as f64;
            let m = util::mean(&x);
            let (se, lo, hi) = if kind.starts_with("prop") {
                let se = (m * (1.0 - m) / n).sqrt();
                // exact (Clopper-Pearson)
                let k = x.iter().filter(|&&v| v == 1.0).count() as f64;
                let lo = if k == 0.0 { 0.0 } else { dist::invibeta(k, n - k + 1.0, a) };
                let hi = if k == n { 1.0 } else { dist::invibeta(k + 1.0, n - k, 1.0 - a) };
                (se, lo, hi)
            } else {
                let se = (util::variance(&x) / n).sqrt();
                let t = dist::invttail(n - 1.0, a);
                (se, m - t * se, m + t * se)
            };
            out.push(format!("{:>12} |{:>11} {:>11} {:>11}       {:>10}   {:>10}", abbrev(v, 12), n, g9(m), g9(se), g9(lo), g9(hi)));
            r.set("N", n);
            r.set("mean", m);
            r.set("se", se);
            r.set("lb", lo);
            r.set("ub", hi);
            r.set("level", level);
        }
        st.r = r;
        Ok(out.join("\n"))
    }
}

/// Helper used by other modules: unweighted summary of a column subset.
pub fn quick_summary(ds: &DataSet, v: &str, mask: &[bool]) -> Summary {
    let x: Vec<f64> = (0..ds.n_obs).map(|i| if mask[i] { ds.get_var(v).map_or(SYSMISS, |c| c.num(i)) } else { SYSMISS }).collect();
    summarize_values(&x, None, None)
}

#[allow(dead_code)]
fn unused(_: &Weights) {}
