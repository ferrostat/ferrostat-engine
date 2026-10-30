//! Power methods for ANOVA designs (one-way, two-way, repeated measures),
//! R² and partial correlations, stratified and matched case-control
//! studies, trend tests, survival comparisons and logistic regression.

use ferrostat_parser::Command;

use super::methods::{item, push, Lists};
use super::ncdist::{f_power, normal_expect, pw_norm, z_power, zcrit};
use super::{flag, g, has, opt_list, word_opt, Def, Row};
use crate::dist;
use crate::eval::rc_err;

fn os(r: &Row) -> bool {
    flag(r, "_onesided")
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Mean of squared deviations from the mean.
fn pvar(x: &[f64]) -> f64 {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

/// Cell means of a design point (row-major, `nr` × `nc`).
fn cells(r: &Row, nr: usize, nc: usize) -> Option<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for i in 1..=nr {
        let mut row = Vec::new();
        for j in 1..=nc {
            let k = if nr == 1 { format!("m{j}") } else { format!("m{i}{j}") };
            if !has(r, &k) {
                return None;
            }
            row.push(g(r, &k));
        }
        out.push(row);
    }
    Some(out)
}

/// Variance components of an `nr` × `nc` table of cell means: rows,
/// columns and interaction (means of squared effects).
fn components(m: &[Vec<f64>]) -> (f64, f64, f64) {
    let nr = m.len();
    let nc = m[0].len();
    let rm: Vec<f64> = m.iter().map(|r| r.iter().sum::<f64>() / nc as f64).collect();
    let cm: Vec<f64> = (0..nc).map(|j| m.iter().map(|r| r[j]).sum::<f64>() / nr as f64).collect();
    let gm = rm.iter().sum::<f64>() / nr as f64;
    let mut inter = 0.0;
    for i in 0..nr {
        for j in 0..nc {
            inter += (m[i][j] - rm[i] - cm[j] + gm).powi(2);
        }
    }
    (pvar(&rm), pvar(&cm), inter / (nr * nc) as f64)
}

fn cell_keys(nr: usize, nc: usize) -> Vec<String> {
    let mut v = Vec::new();
    for i in 1..=nr {
        for j in 1..=nc {
            v.push(if nr == 1 { format!("m{j}") } else { format!("m{i}{j}") });
        }
    }
    v
}

fn push_cells(l: &mut Lists, pos: &[Vec<Vec<f64>>]) -> (usize, usize) {
    let nr = pos.len();
    let nc = pos.iter().map(|r| r.len()).max().unwrap_or(0);
    for (i, row) in pos.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let k = if nr == 1 { format!("m{}", j + 1) } else { format!("m{}{}", i + 1, j + 1) };
            l.push((k, v.clone()));
        }
    }
    (nr, nc)
}

fn first(v: Option<Vec<f64>>) -> Option<f64> {
    v.and_then(|v| v.first().copied())
}

pub(crate) fn setup(m: &str, cmd: &Command, pos: &[Vec<Vec<f64>>], mut l: Lists, mut d: Def) -> Result<(Def, Lists), String> {
    let o = |n: &str, k: usize| opt_list(cmd, n, k);
    let have_pos = pos.first().is_some_and(|r| !r.is_empty());
    match m {
        "oneway" => {
            let k = if have_pos {
                push_cells(&mut l, &pos[..1]).1
            } else {
                first(o("ngroups", 2)?).ok_or_else(|| rc_err(198, "group means or ngroups() required"))? as usize
            };
            if k < 2 {
                return Err(rc_err(198, "at least two groups required"));
            }
            l.push(("ngroups".into(), vec![k as f64]));
            push(&mut l, "varmeans", o("varmeans", 4)?);
            push(&mut l, "varerror", Some(o("varerror", 4)?.unwrap_or(vec![1.0])));
            if let Some(npg) = o("npergroup", 4)? {
                l.push(("N".into(), npg.iter().map(|v| v * k as f64).collect()));
            }
            d.title = "one-way ANOVA".into();
            d.test = "F test for group effect".into();
            d.hyp = "H0: delta = 0  versus  Ha: delta != 0".into();
            d.step = k as f64;
            d.nmin = k as f64 + 1.0;
            d.target = Some("varmeans");
            d.tword = "between-group variance";
            let mut show = strs(&["ngroups"]);
            show.extend(cell_keys(1, if have_pos { k } else { 0 }));
            show.extend(strs(&["varmeans", "varerror"]));
            d.show = show;
            d.power = |r, n, _| {
                let k = g(r, "ngroups");
                f_power(k - 1.0, n - k, n * g(r, "varmeans") / g(r, "varerror"), g(r, "alpha"))
            };
            d.delta = |r| (g(r, "varmeans") / g(r, "varerror")).sqrt();
            d.erange = |r, _| (0.0, 1e6 * g(r, "varerror"));
        }
        "twoway" => {
            let (nr, nc) = if have_pos {
                if pos.iter().any(|r| r.len() != pos[0].len()) {
                    return Err(rc_err(503, "cell means must form a full table"));
                }
                push_cells(&mut l, pos)
            } else {
                let nr = first(o("nrows", 2)?).ok_or_else(|| rc_err(198, "cell means or nrows() and ncols() required"))?;
                let nc = first(o("ncols", 2)?).ok_or_else(|| rc_err(198, "ncols() required"))?;
                (nr as usize, nc as usize)
            };
            if nr < 2 || nc < 2 {
                return Err(rc_err(198, "at least two rows and two columns required"));
            }
            l.push(("nrows".into(), vec![nr as f64]));
            l.push(("ncols".into(), vec![nc as f64]));
            push(&mut l, "varrow", o("varrow", 4)?);
            push(&mut l, "varcolumn", o("varcolumn", 4)?);
            push(&mut l, "varrowcolumn", o("varrowcolumn", 7)?);
            push(&mut l, "varerror", Some(o("varerror", 4)?.unwrap_or(vec![1.0])));
            let f = word_opt(cmd, "factor", 3).unwrap_or_else(|| "row".into());
            let (code, key, label) = match f.as_str() {
                "row" | "r" => (0.0, "varrow", "row"),
                "column" | "col" | "c" => (1.0, "varcolumn", "column"),
                "rowcol" | "rowcolumn" | "rc" => (2.0, "varrowcolumn", "row-by-column"),
                other => return Err(rc_err(198, format!("factor({other}) not allowed"))),
            };
            l.push(("_factor".into(), vec![code]));
            d.title = "two-way ANOVA".into();
            d.test = format!("F test for {label} effect");
            d.hyp = "H0: delta = 0  versus  Ha: delta != 0".into();
            d.step = (nr * nc) as f64;
            d.nmin = (nr * nc) as f64 + 1.0;
            d.target = Some(key);
            d.tword = "effect variance";
            let mut show = strs(&["nrows", "ncols"]);
            show.extend(cell_keys(nr, if have_pos { nc } else { 0 }));
            show.extend(strs(&["varrow", "varcolumn", "varrowcolumn", "varerror"]));
            d.show = show;
            d.power = |r, n, _| {
                let (nr, nc) = (g(r, "nrows"), g(r, "ncols"));
                let (df1, v) = match g(r, "_factor") as i32 {
                    0 => (nr - 1.0, g(r, "varrow")),
                    1 => (nc - 1.0, g(r, "varcolumn")),
                    _ => ((nr - 1.0) * (nc - 1.0), g(r, "varrowcolumn")),
                };
                f_power(df1, n - nr * nc, n * v / g(r, "varerror"), g(r, "alpha"))
            };
            d.delta = |r| {
                let v = match g(r, "_factor") as i32 {
                    0 => g(r, "varrow"),
                    1 => g(r, "varcolumn"),
                    _ => g(r, "varrowcolumn"),
                };
                (v / g(r, "varerror")).sqrt()
            };
            d.erange = |r, _| (0.0, 1e6 * g(r, "varerror"));
        }
        "repeated" => {
            if !have_pos {
                return Err(rc_err(198, "cell means required"));
            }
            if pos.iter().any(|r| r.len() != pos[0].len()) {
                return Err(rc_err(503, "cell means must form a full table"));
            }
            let (ng, nk) = push_cells(&mut l, pos);
            if nk < 2 {
                return Err(rc_err(198, "at least two repeated measures required"));
            }
            l.push(("ngroups".into(), vec![ng as f64]));
            l.push(("nrepeated".into(), vec![nk as f64]));
            push(&mut l, "corr", Some(o("corr", 4)?.ok_or_else(|| rc_err(198, "corr() required"))?));
            push(&mut l, "varerror", Some(o("varerror", 4)?.unwrap_or(vec![1.0])));
            let f = word_opt(cmd, "factor", 3).unwrap_or_else(|| if ng == 1 { "within".into() } else { "between".into() });
            let (code, key, label) = match f.as_str() {
                "within" | "w" => (0.0, "var_w", "within-subject"),
                "between" | "b" => (1.0, "var_b", "between-subjects"),
                "bwithin" | "bw" => (2.0, "var_bw", "between-within"),
                other => return Err(rc_err(198, format!("factor({other}) not allowed"))),
            };
            if ng == 1 && code != 0.0 {
                return Err(rc_err(198, "only factor(within) is available with one group"));
            }
            l.push(("_factor".into(), vec![code]));
            d.title = "repeated-measures ANOVA".into();
            d.test = format!("F test for {label} effect, compound symmetry");
            d.hyp = "H0: delta = 0  versus  Ha: delta != 0".into();
            d.step = ng as f64;
            d.nmin = ng as f64 + 1.0;
            d.target = Some(key);
            d.tword = "effect variance";
            let mut show = strs(&["ngroups", "nrepeated"]);
            show.extend(cell_keys(ng, nk));
            show.extend(strs(&["var_b", "var_w", "var_bw", "corr", "varerror"]));
            d.show = show;
            d.power = |r, n, _| {
                let (ng, nk, rho, s2) = (g(r, "ngroups"), g(r, "nrepeated"), g(r, "corr"), g(r, "varerror"));
                let (df1, df2, ncp) = match g(r, "_factor") as i32 {
                    0 => (nk - 1.0, (n - ng) * (nk - 1.0), n * nk * g(r, "var_w") / (s2 * (1.0 - rho))),
                    1 => (ng - 1.0, n - ng, n * nk * g(r, "var_b") / (s2 * (1.0 + (nk - 1.0) * rho))),
                    _ => ((ng - 1.0) * (nk - 1.0), (n - ng) * (nk - 1.0), n * nk * g(r, "var_bw") / (s2 * (1.0 - rho))),
                };
                f_power(df1, df2, ncp, g(r, "alpha"))
            };
            d.delta = |r| {
                let v = match g(r, "_factor") as i32 {
                    0 => g(r, "var_w"),
                    1 => g(r, "var_b"),
                    _ => g(r, "var_bw"),
                };
                (v / g(r, "varerror")).sqrt()
            };
            d.erange = |r, _| (0.0, 1e6 * g(r, "varerror"));
        }
        "rsquared" => {
            let n = pos.first().map_or(0, |r| r.len());
            match n {
                1 => push(&mut l, "r2", item(pos, 0)),
                2 => {
                    push(&mut l, "r2r", item(pos, 0));
                    push(&mut l, "r2f", item(pos, 1));
                }
                0 => {}
                _ => return Err(rc_err(198, "one or two R-squared values allowed")),
            }
            push(&mut l, "ntested", Some(o("ntested", 5)?.unwrap_or(vec![1.0])));
            push(&mut l, "ncontrol", Some(o("ncontrol", 5)?.unwrap_or(vec![if n == 2 { 1.0 } else { 0.0 }])));
            d.title = if n == 2 { "a test of R-squared change in a multiple linear regression" } else { "a test of R-squared in a multiple linear regression" }.into();
            d.test = "Wald F test".into();
            d.hyp = if n == 2 { "H0: R2_F = R2_R  versus  Ha: R2_F != R2_R".into() } else { "H0: R2 = 0  versus  Ha: R2 != 0".into() };
            d.target = Some(if n == 2 { "r2f" } else { "r2" });
            d.tword = "R-squared";
            d.show = strs(&["r2", "r2r", "r2f", "ntested", "ncontrol"]);
            d.power = |r, n, _| {
                let f2 = f2_rsq(r);
                f_power(g(r, "ntested"), n - g(r, "ntested") - g(r, "ncontrol") - 1.0, f2 * n, g(r, "alpha"))
            };
            d.delta = f2_rsq;
            d.erange = |r, _| (if has(r, "r2r") { g(r, "r2r") } else { 0.0 }, 1.0 - 1e-10);
        }
        "pcorr" => {
            push(&mut l, "pcorr2", item(pos, 0));
            push(&mut l, "ntested", Some(o("ntested", 5)?.unwrap_or(vec![1.0])));
            push(&mut l, "ncontrol", Some(o("ncontrol", 5)?.unwrap_or(vec![1.0])));
            d.title = "a test of a partial correlation in a multiple linear regression".into();
            d.test = "Wald F test".into();
            d.hyp = "H0: pcorr = 0  versus  Ha: pcorr != 0".into();
            d.target = Some("pcorr2");
            d.tword = "squared partial correlation";
            d.show = strs(&["pcorr2", "ntested", "ncontrol"]);
            d.power = |r, n, _| {
                let p = g(r, "pcorr2");
                f_power(g(r, "ntested"), n - g(r, "ntested") - g(r, "ncontrol") - 1.0, p / (1.0 - p) * n, g(r, "alpha"))
            };
            d.delta = |r| g(r, "pcorr2") / (1.0 - g(r, "pcorr2"));
            d.erange = |_, _| (0.0, 1.0 - 1e-10);
        }
        "cmh" => {
            let row = pos.first().cloned().unwrap_or_default();
            if row.is_empty() {
                return Err(rc_err(198, "control-group probabilities of each stratum required"));
            }
            let ns = row.len();
            for (j, v) in row.iter().enumerate() {
                if v.len() != 1 {
                    return Err(rc_err(198, "one probability per stratum"));
                }
                l.push((format!("p1_{}", j + 1), v.clone()));
            }
            let sf = o("sfractions", 2)?.unwrap_or(vec![1.0 / ns as f64; ns]);
            if sf.len() != ns {
                return Err(rc_err(198, "sfractions() must have one value per stratum"));
            }
            let tot: f64 = sf.iter().sum();
            for (j, v) in sf.iter().enumerate() {
                l.push((format!("_sf{}", j + 1), vec![v / tot]));
            }
            l.push(("nstrata".into(), vec![ns as f64]));
            push(&mut l, "oratio", o("oratio", 2)?);
            push(&mut l, "nratio", o("nratio", 6)?);
            d.title = "a Cochran-Mantel-Haenszel test".into();
            d.test = "CMH test of the common odds ratio".into();
            d.hyp = "H0: OR = 1  versus  Ha: OR != 1".into();
            d.target = Some("oratio");
            d.tword = "odds ratio";
            d.nmin = 2.0 * ns as f64;
            let mut show = strs(&["nstrata", "oratio", "nratio"]);
            show.extend((1..=ns).map(|j| format!("p1_{j}")));
            d.show = show;
            d.power = cmh_power;
            d.delta = |r| g(r, "oratio");
            d.erange = |_, up| (1.0, if up { 1e4 } else { 1e-4 });
        }
        "mcc" => {
            push(&mut l, "p0", Some(item(pos, 0).ok_or_else(|| rc_err(198, "exposure probability among controls required"))?));
            push(&mut l, "oratio", o("oratio", 2)?);
            push(&mut l, "M", Some(super::list_exact(cmd, "m")?.unwrap_or(vec![1.0])));
            if let Some(c) = o("corr", 4)? {
                if c.iter().any(|v| *v != 0.0) {
                    return Err(rc_err(198, "power mcc: only corr(0) is supported"));
                }
            }
            d.title = "a matched case-control study".into();
            d.test = "Score test of the conditional-logistic odds ratio".into();
            d.hyp = "H0: OR = 1  versus  Ha: OR != 1".into();
            d.target = Some("oratio");
            d.tword = "odds ratio";
            d.nmin = 1.0;
            d.show = strs(&["M", "p0", "oratio"]);
            d.power = mcc_power;
            d.delta = |r| g(r, "oratio");
            d.erange = |_, up| (1.0, if up { 1e4 } else { 1e-4 });
        }
        "trend" => {
            let row = pos.first().cloned().unwrap_or_default();
            if row.len() < 2 {
                return Err(rc_err(198, "at least two group proportions required"));
            }
            let k = row.len();
            for (j, v) in row.iter().enumerate() {
                l.push((format!("p{}", j + 1), v.clone()));
            }
            let x = o("exposure", 3)?.unwrap_or((1..=k).map(|v| v as f64).collect());
            if x.len() != k {
                return Err(rc_err(198, "exposure() must have one value per group"));
            }
            for (j, v) in x.iter().enumerate() {
                l.push((format!("_x{}", j + 1), vec![*v]));
            }
            l.push(("ngroups".into(), vec![k as f64]));
            d.title = "a trend test of proportions".into();
            d.test = "Cochran-Armitage test".into();
            d.hyp = "H0: b = 0  versus  Ha: b != 0".into();
            d.step = k as f64;
            d.nmin = k as f64;
            let mut show = strs(&["ngroups"]);
            show.extend((1..=k).map(|j| format!("p{j}")));
            d.show = show;
            d.power = |r, n, _| {
                let (t, v0, v1, _) = trend_parts(r);
                pw_norm(t * n.sqrt(), v0.sqrt(), v1.sqrt(), g(r, "alpha"), os(r))
            };
            d.delta = |r| trend_parts(r).3;
        }
        "cox" => {
            push(&mut l, "b1", item(pos, 0));
            push(&mut l, "hratio", o("hratio", 2)?);
            push(&mut l, "sd", Some(o("sd", 2)?.unwrap_or(vec![0.5])));
            push(&mut l, "r2", Some(o("r2", 2)?.unwrap_or(vec![0.0])));
            push(&mut l, "Pr_E", Some(o("eventprob", 5)?.or(o("failprob", 4)?).unwrap_or(vec![1.0])));
            d.title = "the Cox proportional hazards model".into();
            d.test = "Partial-likelihood test of one coefficient".into();
            d.hyp = "H0: b1 = 0  versus  Ha: b1 != 0".into();
            d.target = Some("b1");
            d.tword = "coefficient";
            d.nmin = 1.0;
            d.show = strs(&["b1", "hratio", "sd", "r2", "Pr_E"]);
            d.power = |r, n, _| {
                let ev = n * g(r, "Pr_E");
                z_power((ev * (1.0 - g(r, "r2"))).sqrt() * g(r, "sd") * b1_of(r), g(r, "alpha"), os(r))
            };
            d.delta = b1_of;
            d.erange = |_, up| (0.0, if up { 20.0 } else { -20.0 });
            d.lower_default = true;
            d.extra = Some(|r, n, _| vec![("E".into(), n * g(r, "Pr_E")), ("hratio".into(), b1_of(r).exp())]);
        }
        "exponential" => {
            push(&mut l, "h1", item(pos, 0));
            push(&mut l, "h2", item(pos, 1));
            push(&mut l, "hratio", o("hratio", 2)?);
            push(&mut l, "aperiod", o("aperiod", 2)?);
            push(&mut l, "fperiod", o("fperiod", 2)?);
            push(&mut l, "losshaz", o("losshaz", 5)?);
            let lh = cmd.has_opt("loghazard", 4);
            if lh {
                l.push(("_loghaz".into(), vec![1.0]));
            }
            d.title = "a two-sample comparison of survivor functions".into();
            d.test = if lh { "Exponential test, log-hazard difference" } else { "Exponential test, hazard difference" }.into();
            d.hyp = "H0: h2 = h1  versus  Ha: h2 != h1".into();
            d.two = true;
            d.target = Some("h2");
            d.tword = "experimental-group hazard";
            d.nmin = 1.0;
            d.lower_default = true;
            d.show = strs(&["h1", "h2", "hratio", "aperiod", "fperiod", "losshaz"]);
            d.power = |r, n1, n2| {
                let (h1, h2) = (g(r, "h1"), h2_of(r));
                let (d1, d2) = (pevent(r, h1), pevent(r, h2));
                let n = n1 + n2;
                let (w1, w2) = (n1 / n, n2 / n);
                if flag(r, "_loghaz") {
                    z_power((h2.ln() - h1.ln()) / (1.0 / (n1 * d1) + 1.0 / (n2 * d2)).sqrt(), g(r, "alpha"), os(r))
                } else {
                    let phi = |h: f64| h * h / pevent(r, h);
                    let hb = w1 * h1 + w2 * h2;
                    let s0 = (phi(hb) * (1.0 / w1 + 1.0 / w2)).sqrt();
                    let s1 = (phi(h1) / w1 + phi(h2) / w2).sqrt();
                    pw_norm(n.sqrt() * (h2 - h1), s0, s1, g(r, "alpha"), os(r))
                }
            };
            d.delta = |r| h2_of(r) / g(r, "h1");
            d.erange = |r, up| (g(r, "h1"), g(r, "h1") * if up { 1e4 } else { 1e-4 });
            d.extra = Some(|r, n1, n2| {
                let (e1, e2) = (n1 * pevent(r, g(r, "h1")), n2 * pevent(r, h2_of(r)));
                vec![("E".into(), e1 + e2), ("hratio".into(), h2_of(r) / g(r, "h1"))]
            });
        }
        "logrank" => {
            push(&mut l, "s1", item(pos, 0));
            push(&mut l, "s2", item(pos, 1));
            push(&mut l, "hratio", o("hratio", 2)?);
            push(&mut l, "wdprob", o("wdprob", 3)?);
            let sch = cmd.has_opt("schoenfeld", 5);
            if sch {
                l.push(("_schoenfeld".into(), vec![1.0]));
            }
            d.title = "two-sample comparison of survivor functions".into();
            d.test = if sch { "Log-rank test, Schoenfeld method" } else { "Log-rank test, Freedman method" }.into();
            d.hyp = "H0: HR = 1  versus  Ha: HR != 1".into();
            d.two = true;
            d.cluster = true;
            d.target = Some("hratio");
            d.tword = "hazard ratio";
            d.nmin = 1.0;
            d.lower_default = true;
            d.show = strs(&["hratio", "s1", "s2", "wdprob", "Pr_E"]);
            d.power = |r, n1, n2| {
                let ev = (n1 + n2) * prob_event(r, n1, n2);
                let psi = g(r, "hratio");
                let z = if flag(r, "_schoenfeld") {
                    let w1 = n1 / (n1 + n2);
                    (ev * w1 * (1.0 - w1)).sqrt() * psi.ln().abs()
                } else {
                    let k = n2 / n1;
                    (ev * k).sqrt() * (1.0 - psi).abs() / (1.0 + k * psi)
                };
                z_power(z, g(r, "alpha"), os(r))
            };
            d.delta = |r| g(r, "hratio");
            d.erange = |_, up| (1.0, if up { 1e3 } else { 1e-3 });
            d.extra = Some(|r, n1, n2| {
                let pe = prob_event(r, n1, n2);
                let mut v = vec![("E".into(), (n1 + n2) * pe)];
                if has(r, "s1") {
                    v.push(("s2".into(), g(r, "s1").powf(g(r, "hratio"))));
                    v.push(("Pr_E".into(), pe));
                }
                v
            });
        }
        "logistic" => {
            push(&mut l, "b1", item(pos, 0));
            push(&mut l, "oratio", o("oratio", 2)?);
            push(&mut l, "pr", Some(o("pr", 2)?.unwrap_or(vec![0.5])));
            push(&mut l, "covsd", o("covsd", 4)?);
            push(&mut l, "covmean", o("covmean", 4)?);
            push(&mut l, "covprob", o("covprob", 4)?);
            push(&mut l, "r2", Some(o("r2", 2)?.unwrap_or(vec![0.0])));
            let general = cmd.has_opt("general", 3) || cmd.has_opt("wald", 4);
            if general {
                l.push(("_general".into(), vec![1.0]));
            }
            d.title = "a logistic regression slope test".into();
            d.test = if general { "Wald test, information-based (general covariate distribution)".into() } else { "Large-sample test, closed-form approximation".into() };
            d.hyp = "H0: b1 = 0  versus  Ha: b1 != 0".into();
            d.target = Some("b1");
            d.tword = "coefficient";
            d.nmin = 2.0;
            d.show = strs(&["b1", "oratio", "pr", "covmean", "covsd", "covprob", "r2"]);
            d.power = logistic_power;
            d.delta = b1_of;
            d.erange = |_, up| (0.0, if up { 15.0 } else { -15.0 });
            d.extra = Some(|r, _, _| vec![("oratio".into(), b1_of(r).exp())]);
        }
        other => return Err(rc_err(199, format!("power method {other} not found"))),
    }
    Ok((d, l))
}

fn f2_rsq(r: &Row) -> f64 {
    if has(r, "r2f") {
        (g(r, "r2f") - g(r, "r2r")) / (1.0 - g(r, "r2f"))
    } else {
        g(r, "r2") / (1.0 - g(r, "r2"))
    }
}

fn b1_of(r: &Row) -> f64 {
    if has(r, "b1") {
        g(r, "b1")
    } else if has(r, "hratio") {
        g(r, "hratio").ln()
    } else if has(r, "oratio") {
        g(r, "oratio").ln()
    } else {
        f64::NAN
    }
}

fn h2_of(r: &Row) -> f64 {
    if has(r, "h2") {
        g(r, "h2")
    } else {
        g(r, "h1") * g(r, "hratio")
    }
}

/// Probability of observing an event with exponential hazard `h`, uniform
/// accrual over `aperiod`, further follow-up `fperiod` and exponential
/// losses `losshaz`.
fn pevent(r: &Row, h: f64) -> f64 {
    let eta = if has(r, "losshaz") { g(r, "losshaz") } else { 0.0 };
    let ra = if has(r, "aperiod") { g(r, "aperiod") } else { 0.0 };
    let has_f = has(r, "fperiod");
    if !has_f && ra == 0.0 {
        return h / (h + eta);
    }
    let f = if has_f { g(r, "fperiod") } else { 0.0 };
    let lam = h + eta;
    let tail = if ra > 0.0 { ((-lam * f).exp() - (-lam * (f + ra)).exp()) / (lam * ra) } else { (-lam * f).exp() };
    h / lam * (1.0 - tail)
}

/// Probability of an event in a log-rank design.
fn prob_event(r: &Row, n1: f64, n2: f64) -> f64 {
    let wd = if has(r, "wdprob") { g(r, "wdprob") } else { 0.0 };
    if !has(r, "s1") {
        return 1.0 - wd;
    }
    let s1 = g(r, "s1");
    let s2 = s1.powf(g(r, "hratio"));
    let w1 = n1 / (n1 + n2);
    (w1 * (1.0 - s1) + (1.0 - w1) * (1.0 - s2)) * (1.0 - wd)
}

fn cmh_power(r: &Row, n: f64, _: f64) -> f64 {
    let ns = g(r, "nstrata") as usize;
    let ratio = if has(r, "nratio") { g(r, "nratio") } else { 1.0 };
    let psi = g(r, "oratio");
    let (mut e, mut v0, mut v1) = (0.0, 0.0, 0.0);
    for j in 1..=ns {
        let nj = n * g(r, &format!("_sf{j}"));
        let n1 = nj / (1.0 + ratio);
        let n2 = nj - n1;
        let p1 = g(r, &format!("p1_{j}"));
        let p2 = psi * p1 / (1.0 - p1 + psi * p1);
        let c = n1 * n2 / nj;
        let pb = (n1 * p1 + n2 * p2) / nj;
        e += c * (p2 - p1);
        v0 += c * c * pb * (1.0 - pb) * (1.0 / n1 + 1.0 / n2);
        v1 += c * c * (p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2);
    }
    pw_norm(e, v0.sqrt(), v1.sqrt(), g(r, "alpha"), os(r))
}

/// Matched sets of one case and M controls: score test of the conditional
/// logistic model, with the distribution of exposure configurations under
/// the alternative.
fn mcc_power(r: &Row, n: f64, _: f64) -> f64 {
    let m = g(r, "M").round().max(1.0);
    let p0 = g(r, "p0");
    let psi = g(r, "oratio");
    let p1 = psi * p0 / (1.0 - p0 + psi * p0);
    let mi = m as usize;
    let (mut e, mut s2, mut v0) = (0.0, 0.0, 0.0);
    for case in 0..=1usize {
        let pc = if case == 1 { p1 } else { 1.0 - p1 };
        for j in 0..=mi {
            let pj = dist::binomialp(m, j as f64, p0);
            let pr = pc * pj;
            let t = (case + j) as f64;
            let u = case as f64 - t / (m + 1.0);
            e += pr * u;
            s2 += pr * u * u;
            v0 += pr * t * (m + 1.0 - t) / (m + 1.0).powi(2);
        }
    }
    let v1 = s2 - e * e;
    pw_norm(n * e, (n * v0).sqrt(), (n * v1).sqrt(), g(r, "alpha"), os(r))
}

/// (score mean, null variance, alternative variance, slope) per unit sample.
fn trend_parts(r: &Row) -> (f64, f64, f64, f64) {
    let k = g(r, "ngroups") as usize;
    let p: Vec<f64> = (1..=k).map(|j| g(r, &format!("p{j}"))).collect();
    let x: Vec<f64> = (1..=k).map(|j| g(r, &format!("_x{j}"))).collect();
    let w = 1.0 / k as f64;
    let xb = x.iter().sum::<f64>() * w;
    let pb = p.iter().sum::<f64>() * w;
    let mut t = 0.0;
    let mut sxx = 0.0;
    let mut v1 = 0.0;
    for j in 0..k {
        let dx = x[j] - xb;
        t += w * p[j] * dx;
        sxx += w * dx * dx;
        v1 += w * p[j] * (1.0 - p[j]) * dx * dx;
    }
    (t, pb * (1.0 - pb) * sxx, v1, t / sxx)
}

/// Per-observation information matrix entries (i00, i01, i11) of a
/// logistic model with intercept `b0` and slope `b`, averaged by `ex`.
fn logit_info(b0: f64, b: f64, ex: &dyn Fn(&dyn Fn(f64) -> f64) -> f64) -> (f64, f64, f64) {
    let w = |x: f64| {
        let p = dist::invlogit(b0 + b * x);
        p * (1.0 - p)
    };
    (ex(&|x| w(x)), ex(&|x| w(x) * x), ex(&|x| w(x) * x * x))
}

fn logistic_power(r: &Row, n: f64, _: f64) -> f64 {
    let b = b1_of(r);
    let a = g(r, "alpha");
    let one = os(r);
    let neff = n * (1.0 - g(r, "r2"));
    let pr = g(r, "pr");
    let binary = has(r, "covprob");
    if flag(r, "_general") {
        let (i00, i01, i11) = if binary {
            let q = g(r, "covprob");
            logit_info(dist::logit(pr), b, &|f: &dyn Fn(f64) -> f64| (1.0 - q) * f(0.0) + q * f(1.0))
        } else {
            let mu = if has(r, "covmean") { g(r, "covmean") } else { 0.0 };
            let sd = if has(r, "covsd") { g(r, "covsd") } else { 1.0 };
            // pr is the event probability at the covariate mean
            let b0 = dist::logit(pr) - b * mu;
            logit_info(b0, b, &|f: &dyn Fn(f64) -> f64| normal_expect(|z| f(mu + sd * z)))
        };
        let det = i00 * i11 - i01 * i01;
        let v = i00 / det;
        return z_power(b * (neff / v).sqrt(), a, one);
    }
    if binary {
        let bq = g(r, "covprob");
        let p1 = pr;
        let p2 = dist::invlogit(dist::logit(p1) + b);
        let p = (1.0 - bq) * p1 + bq * p2;
        let z = zcrit(a, one);
        let num = (neff * (1.0 - bq)).sqrt() * (p1 - p2).abs() - z * (p * (1.0 - p) / bq).sqrt();
        dist::normal(num / (p1 * (1.0 - p1) + p2 * (1.0 - p2) * (1.0 - bq) / bq).sqrt())
    } else {
        let sd = if has(r, "covsd") { g(r, "covsd") } else { 1.0 };
        z_power(b * sd * (neff * pr * (1.0 - pr)).sqrt(), a, one)
    }
}

pub(crate) fn prepare(def: &Def, r: &mut Row) -> Result<(), String> {
    match def.method.as_str() {
        "oneway" => {
            let k = g(r, "ngroups") as usize;
            if let Some(c) = cells(r, 1, k) {
                r.insert("varmeans".into(), pvar(&c[0]));
            }
        }
        "twoway" => {
            let (nr, nc) = (g(r, "nrows") as usize, g(r, "ncols") as usize);
            if let Some(c) = cells(r, nr, nc) {
                let (a, b, i) = components(&c);
                r.insert("varrow".into(), a);
                r.insert("varcolumn".into(), b);
                r.insert("varrowcolumn".into(), i);
            }
        }
        "repeated" => {
            let (ng, nk) = (g(r, "ngroups") as usize, g(r, "nrepeated") as usize);
            if let Some(c) = cells(r, ng, nk) {
                let (b, w, i) = components(&c);
                r.insert("var_b".into(), b);
                r.insert("var_w".into(), w);
                r.insert("var_bw".into(), i);
            }
        }
        "cox" | "logistic" => {
            if !has(r, "b1") && (has(r, "hratio") || has(r, "oratio")) {
                let v = b1_of(r);
                r.insert("b1".into(), v);
            }
        }
        "exponential" => {
            if !has(r, "h1") {
                return Err(rc_err(198, "control-group hazard h1 required"));
            }
            if has(r, "h2") {
                r.insert("hratio".into(), g(r, "h2") / g(r, "h1"));
            } else if has(r, "hratio") {
                r.insert("h2".into(), g(r, "h1") * g(r, "hratio"));
            }
        }
        "logrank" => {
            if has(r, "s1") && has(r, "s2") {
                r.insert("hratio".into(), g(r, "s2").ln() / g(r, "s1").ln());
            } else if !has(r, "hratio") && !(has(r, "N") || has(r, "N1") || has(r, "K") || has(r, "K1")) {
                r.insert("hratio".into(), 0.5);
            }
        }
        _ => {}
    }
    Ok(())
}
