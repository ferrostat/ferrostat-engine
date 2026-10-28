//! Post-estimation for binary-outcome models (logit, logistic, probit):
//! `estat classification`, `estat gof` (Pearson and Hosmer–Lemeshow),
//! `lroc`, `lsens`; and the `brier` score command.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, VarType};
use ferrostat_parser::parser::parse_expression;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::graph::{Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("lroc", Box::new(Lroc)), ("lsens", Box::new(Lsens)), ("brier", Box::new(Brier))]
}

/// Outcome, predicted probability, frequency weight and covariate values
/// of each estimation-sample observation.
struct Fit {
    rows: Vec<usize>,
    y: Vec<bool>,
    p: Vec<f64>,
    w: Vec<f64>,
    x: Vec<Vec<f64>>,
    depvar: String,
    title: String,
}

fn binary_fit(ds: &DataSet, st: &Store, all: bool) -> Result<Fit, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let title = match ecmd.as_str() {
        "logit" | "logistic" => "Logistic model",
        "probit" => "Probit model",
        "cloglog" => "Complementary log-log model",
        _ => return Err(rc_err(321, "requested action not valid after this command; only after logit, logistic, probit")),
    };
    let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "e(sample) not found"))?;
    let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let ye = parse_expression(&depvar).map_err(|e| rc_err(198, e.message))?;
    let fw = st.e.macros.get("wtype").is_some_and(|w| w.starts_with("fw"));
    let we = match st.e.macros.get("wexp").filter(|_| fw) {
        Some(w) => Some(parse_expression(w.trim().trim_start_matches('=').trim()).map_err(|e| rc_err(198, e.message))?),
        None => None,
    };
    let env = Env::with_store(ds, st);
    let names: Vec<String> = b.colnames.iter().filter(|c| c.as_str() != "_cons").cloned().collect();
    let mut f = Fit { rows: Vec::new(), y: Vec::new(), p: Vec::new(), w: Vec::new(), x: Vec::new(), depvar, title: title.to_string() };
    for i in 0..ds.n_obs {
        if !all && !sample.get(i).copied().unwrap_or(false) {
            continue;
        }
        let y = env.eval(&ye, i)?.num()?;
        let p = crate::postest::default_prediction(st, &env, &b, i)?;
        if is_missing(y) || is_missing(p) {
            continue;
        }
        let w = match &we {
            Some(e) => env.eval(e, i)?.num()?,
            None => 1.0,
        };
        let x: Vec<f64> = names.iter().map(|n| crate::postest::coef_value(&env, n, i)).collect::<Result<_, _>>()?;
        f.rows.push(i);
        f.y.push(y != 0.0);
        f.p.push(p);
        f.w.push(w);
        f.x.push(x);
    }
    if f.rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    Ok(f)
}

fn pct(x: f64) -> String {
    if x.is_nan() || is_missing(x) {
        ".".into()
    } else {
        format!("{:.2}%", x)
    }
}

/// estat subcommands handled here; None when `sub` is not one of them.
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if !matches!(ecmd.as_str(), "logit" | "logistic" | "probit" | "cloglog") {
        return None;
    }
    let s = sub.to_lowercase();
    let is = |full: &str, min: usize| s.len() >= min && full.starts_with(s.as_str());
    if is("classification", 5) {
        return Some(classification(ds, st, cmd));
    }
    if s == "gof" {
        return Some(gof(ds, st, cmd));
    }
    None
}

fn classification(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let f = binary_fit(ds, st, cmd.has_opt("all", 3))?;
    let cut = util::opt_num(cmd, "cutoff", 3)?.unwrap_or(0.5);
    let (mut a, mut b, mut c, mut d) = (0.0, 0.0, 0.0, 0.0); // +D, +~D, -D, -~D
    for i in 0..f.y.len() {
        let plus = f.p[i] >= cut;
        match (plus, f.y[i]) {
            (true, true) => a += f.w[i],
            (true, false) => b += f.w[i],
            (false, true) => c += f.w[i],
            (false, false) => d += f.w[i],
        }
    }
    let n = a + b + c + d;
    let fmtn = |x: f64| util::commas(x as usize);
    let mut out = vec![String::new(), format!("{} for {}", f.title, f.depvar), String::new()];
    out.push("              -------- True --------".into());
    out.push("Classified |         D            ~D  |      Total".into());
    out.push("-----------+--------------------------+-----------".into());
    out.push(format!("     +     | {:>9}     {:>9}  |  {:>9}", fmtn(a), fmtn(b), fmtn(a + b)));
    out.push(format!("     -     | {:>9}     {:>9}  |  {:>9}", fmtn(c), fmtn(d), fmtn(c + d)));
    out.push("-----------+--------------------------+-----------".into());
    out.push(format!("   Total   | {:>9}     {:>9}  |  {:>9}", fmtn(a + c), fmtn(b + d), fmtn(n)));
    out.push(String::new());
    out.push(format!("Classified + if predicted Pr(D) >= {}", crate::format::fmt_g(cut, 8)));
    out.push(format!("True D defined as {} != 0", f.depvar));
    let sens = 100.0 * a / (a + c);
    let spec = 100.0 * d / (b + d);
    let ppv = 100.0 * a / (a + b);
    let npv = 100.0 * d / (c + d);
    let fp_true = 100.0 * b / (b + d);
    let fn_true = 100.0 * c / (a + c);
    let fp_cl = 100.0 * b / (a + b);
    let fn_cl = 100.0 * c / (c + d);
    let corr = 100.0 * (a + d) / n;
    let line = |l: &str, p: &str, v: f64| format!("{:<32}{:<10}{:>8}", l, p, pct(v));
    let rule = "-".repeat(50);
    out.push(rule.clone());
    out.push(line("Sensitivity", "Pr( +| D)", sens));
    out.push(line("Specificity", "Pr( -|~D)", spec));
    out.push(line("Positive predictive value", "Pr( D| +)", ppv));
    out.push(line("Negative predictive value", "Pr(~D| -)", npv));
    out.push(rule.clone());
    out.push(line("False + rate for true ~D", "Pr( +|~D)", fp_true));
    out.push(line("False - rate for true D", "Pr( -| D)", fn_true));
    out.push(line("False + rate for classified +", "Pr(~D| +)", fp_cl));
    out.push(line("False - rate for classified -", "Pr( D| -)", fn_cl));
    out.push(rule.clone());
    out.push(line("Correctly classified", "", corr));
    out.push(rule);
    let mut r = Results::new();
    r.set("N", n);
    r.set("P_corr", corr);
    r.set("P_p1", sens);
    r.set("P_n0", spec);
    r.set("P_1p", ppv);
    r.set("P_0n", npv);
    r.set("P_p0", fp_true);
    r.set("P_n1", fn_true);
    r.set("P_0p", fp_cl);
    r.set("P_1n", fn_cl);
    let mut ct = Matrix::zeros(3, 3);
    ct.data = vec![a, b, a + b, c, d, c + d, a + c, b + d, n];
    r.set_mat("ctable", ct.with_names(vec!["+".into(), "-".into(), "Total".into()], vec!["D".into(), "~D".into(), "Total".into()]));
    st.r = r;
    Ok(out.join("\n"))
}

/// Stata-style percentile of sorted weighted values (`_pctile` definition).
fn pctile_w(sorted: &[(f64, f64)], p: f64) -> f64 {
    let total: f64 = sorted.iter().map(|x| x.1).sum();
    let target = total * p / 100.0;
    let mut cum = 0.0;
    for (i, &(v, w)) in sorted.iter().enumerate() {
        cum += w;
        if (cum - target).abs() < 1e-9 * total.max(1.0) {
            return match sorted.get(i + 1) {
                Some(nx) => (v + nx.0) / 2.0,
                None => v,
            };
        }
        if cum > target {
            return v;
        }
    }
    sorted.last().map_or(SYSMISS, |x| x.0)
}

fn gof(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let f = binary_fit(ds, st, cmd.has_opt("all", 3))?;
    let k = st.e.get("rank").unwrap_or(1.0);
    let model = f.title.to_lowercase();
    let mut out = vec![String::new(), format!("Goodness-of-fit test after {model}"), format!("Variable: {}", f.depvar), String::new()];
    let n: f64 = f.w.iter().sum();
    let mut r = Results::new();
    r.set("N", n);
    let groups = util::opt_num(cmd, "group", 1)?;
    let rows: Vec<(String, String)>;
    if let Some(g) = groups {
        let g = g as usize;
        if g < 2 {
            return Err(rc_err(198, "group() must be at least 2"));
        }
        let mut sorted: Vec<(f64, f64)> = f.p.iter().zip(&f.w).map(|(p, w)| (*p, *w)).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cuts: Vec<f64> = (1..g).map(|j| pctile_w(&sorted, 100.0 * j as f64 / g as f64)).collect();
        // group of each observation: first cut not below its probability
        let grp: Vec<usize> = f.p.iter().map(|&p| cuts.iter().position(|&c| p <= c).unwrap_or(g - 1)).collect();
        let mut stats = vec![(0.0, 0.0, 0.0, 0.0f64); g]; // (n, obs1, exp1, maxp)
        for i in 0..f.p.len() {
            let s = &mut stats[grp[i]];
            s.0 += f.w[i];
            if f.y[i] {
                s.1 += f.w[i];
            }
            s.2 += f.w[i] * f.p[i];
            s.3 = s.3.max(f.p[i]);
        }
        let used: Vec<usize> = (0..g).filter(|&j| stats[j].0 > 0.0).collect();
        let chi2: f64 = used.iter().map(|&j| {
            let (nj, o, e, _) = stats[j];
            (o - e).powi(2) / (e * (1.0 - e / nj))
        }).sum();
        let ng = used.len();
        let df = ng as f64 - 2.0;
        let p = dist::chi2tail(df, chi2);
        if cmd.has_opt("table", 3) {
            out.push("  (Table collapsed on quantiles of estimated probabilities)".into());
            out.push("  +--------------------------------------------------------+".into());
            out.push("  | Group |   Prob | Obs_1 | Exp_1 | Obs_0 | Exp_0 | Total |".into());
            out.push("  |-------+--------+-------+-------+-------+-------+-------|".into());
            for (gi, &j) in used.iter().enumerate() {
                let (nj, o, e, mx) = stats[j];
                if gi > 0 && gi % 5 == 0 {
                    out.push("  |-------+--------+-------+-------+-------+-------+-------|".into());
                }
                out.push(format!("  | {:>5} | {:>6.4} | {:>5} | {:>5.1} | {:>5} | {:>5.1} | {:>5} |", gi + 1, mx, o, e, nj - o, nj - e, nj));
            }
            out.push("  +--------------------------------------------------------+".into());
            out.push(String::new());
        }
        rows = vec![
            ("Number of observations".into(), util::commas(n as usize)),
            ("Number of groups".into(), ng.to_string()),
            (format!("Hosmer–Lemeshow chi2({})", df), format!("{:.2}", chi2)),
            ("Prob > chi2".into(), format!("{:.4}", p)),
        ];
        r.set("m", ng as f64);
        r.set("df", df);
        r.set("chi2", chi2);
        r.set("p", p);
    } else {
        // covariate patterns
        let mut pats: std::collections::BTreeMap<Vec<u64>, (f64, f64, f64)> = Default::default();
        for i in 0..f.p.len() {
            let key: Vec<u64> = f.x[i].iter().map(|v| v.to_bits()).collect();
            let e = pats.entry(key).or_insert((0.0, 0.0, f.p[i]));
            e.0 += f.w[i];
            if f.y[i] {
                e.1 += f.w[i];
            }
        }
        let m = pats.len();
        let chi2: f64 = pats.values().map(|&(mj, yj, pj)| (yj - mj * pj).powi(2) / (mj * pj * (1.0 - pj))).sum();
        let df = m as f64 - k;
        let p = dist::chi2tail(df, chi2);
        rows = vec![
            ("Number of observations".into(), util::commas(n as usize)),
            ("Number of covariate patterns".into(), util::commas(m)),
            (format!("Pearson chi2({})", df), format!("{:.2}", chi2)),
            ("Prob > chi2".into(), format!("{:.4}", p)),
        ];
        r.set("m", m as f64);
        r.set("df", df);
        r.set("chi2", chi2);
        r.set("p", p);
    }
    let lw = rows.iter().map(|r| r.0.chars().count()).max().unwrap_or(0);
    let vw = rows.iter().map(|r| r.1.len()).max().unwrap_or(0).max(6);
    for (l, v) in rows {
        let pad = lw - l.chars().count();
        out.push(format!("{}{l} = {v:>vw$}", " ".repeat(pad)));
    }
    st.r = r;
    Ok(out.join("\n"))
}

/// Area under the empirical ROC curve (ties count one half).
fn auc(y: &[bool], p: &[f64], w: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..y.len()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let (mut num, mut neg_below) = (0.0, 0.0);
    let (np, nn): (f64, f64) = (idx.iter().filter(|&&i| y[i]).map(|&i| w[i]).sum(), idx.iter().filter(|&&i| !y[i]).map(|&i| w[i]).sum());
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && p[idx[j]] == p[idx[i]] {
            j += 1;
        }
        let pos: f64 = idx[i..j].iter().filter(|&&t| y[t]).map(|&t| w[t]).sum();
        let neg: f64 = idx[i..j].iter().filter(|&&t| !y[t]).map(|&t| w[t]).sum();
        num += pos * (neg_below + 0.5 * neg);
        neg_below += neg;
        i = j;
    }
    num / (np * nn)
}

/// ROC points (1 - specificity, sensitivity) over all cutoffs.
fn roc_curve(y: &[bool], p: &[f64], w: &[f64]) -> Vec<(f64, f64)> {
    let mut cuts: Vec<f64> = p.to_vec();
    cuts.sort_by(|a, b| b.total_cmp(a));
    cuts.dedup();
    let np: f64 = (0..y.len()).filter(|&i| y[i]).map(|i| w[i]).sum();
    let nn: f64 = (0..y.len()).filter(|&i| !y[i]).map(|i| w[i]).sum();
    let mut pts = vec![(0.0, 0.0)];
    for c in cuts {
        let tp: f64 = (0..y.len()).filter(|&i| y[i] && p[i] >= c).map(|i| w[i]).sum();
        let fp: f64 = (0..y.len()).filter(|&i| !y[i] && p[i] >= c).map(|i| w[i]).sum();
        pts.push((fp / nn, tp / np));
    }
    pts
}

pub struct Lroc;

impl CommandHandler for Lroc {
    fn name(&self) -> &'static str {
        "lroc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (y, p, w, title, depvar) = if cmd.main.trim().is_empty() {
            let f = binary_fit(ds, st, cmd.has_opt("all", 3))?;
            (f.y, f.p, f.w, f.title, f.depvar)
        } else {
            // lroc depvar varname: outcome and probabilities from the data
            let vars = crate::varlist::expand(ds, &cmd.main)?;
            if vars.len() != 2 {
                return Err(rc_err(198, "lroc [depvar varname]"));
            }
            let mask = util::touse(ds, st, cmd)?;
            let (yc, pc) = (util::col(ds, &vars[0])?, util::col(ds, &vars[1])?);
            let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(yc[i]) && !is_missing(pc[i])).collect();
            (rows.iter().map(|&i| yc[i] != 0.0).collect(), rows.iter().map(|&i| pc[i]).collect(), vec![1.0; rows.len()], String::new(), vars[0].clone())
        };
        if !y.iter().any(|v| *v) || y.iter().all(|v| *v) {
            return Err(rc_err(2000, "outcome does not vary"));
        }
        let a = auc(&y, &p, &w);
        let n: f64 = w.iter().sum();
        let mut out = vec![String::new()];
        if !title.is_empty() {
            out.push(format!("{title} for {depvar}"));
            out.push(String::new());
        }
        out.push(format!("Number of observations = {:>8}", util::commas(n as usize)));
        out.push(format!("Area under ROC curve   = {:>8.4}", a));
        let mut r = Results::new();
        r.set("N", n);
        r.set("area", a);
        st.r = r;
        if !cmd.has_opt("nograph", 4) {
            let mut g = Graph {
                layers: vec![
                    Layer { kind: LayerKind::Connected, points: roc_curve(&y, &p, &w), label: format!("Area under ROC curve = {:.4}", a), ..Default::default() },
                    Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (1.0, 1.0)], label: String::new(), ..Default::default() },
                ],
                xtitle: "1 - specificity".into(),
                ytitle: "Sensitivity".into(),
                title: format!("Area under ROC curve = {:.4}", a),
                legend: false,
                ..Default::default()
            };
            let extra = crate::graph::finish(&mut g, cmd, st)?;
            if !extra.is_empty() {
                out.push(extra);
            }
        }
        Ok(out.join("\n"))
    }
}

pub struct Lsens;

impl CommandHandler for Lsens {
    fn name(&self) -> &'static str {
        "lsens"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let f = binary_fit(ds, st, cmd.has_opt("all", 3))?;
        let mut cuts: Vec<f64> = f.p.clone();
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let np: f64 = (0..f.y.len()).filter(|&i| f.y[i]).map(|i| f.w[i]).sum();
        let nn: f64 = (0..f.y.len()).filter(|&i| !f.y[i]).map(|i| f.w[i]).sum();
        // (cutoff, sensitivity, specificity), from 0 to 1
        let mut pts = vec![(0.0, 1.0, 0.0)];
        for &c in &cuts {
            let tp: f64 = (0..f.y.len()).filter(|&i| f.y[i] && f.p[i] >= c).map(|i| f.w[i]).sum();
            let tn: f64 = (0..f.y.len()).filter(|&i| !f.y[i] && f.p[i] < c).map(|i| f.w[i]).sum();
            pts.push((c, tp / np, tn / nn));
        }
        pts.push((1.0, 0.0, 1.0));
        let gens = [("genprob", 0usize), ("gensens", 1), ("genspec", 2)];
        let mut made = false;
        for (opt, k) in gens {
            if let Some(name) = cmd.opt(opt, 5) {
                let name = name.trim();
                crate::varlist::check_name(name)?;
                if ds.var_index.contains_key(name) {
                    return Err(rc_err(110, format!("variable {name} already defined")));
                }
                let mut v = ferrostat_core::Variable::new(name.to_string(), VarType::Float);
                v.data = (0..ds.n_obs).map(|i| util::typed_value(VarType::Float, pts.get(i).map_or(SYSMISS, |p| [p.0, p.1, p.2][k]))).collect();
                ds.add_variable(v)?;
                made = true;
            }
        }
        if made {
            st.data_changed = true;
        }
        if !cmd.has_opt("nograph", 4) {
            let mut g = Graph {
                layers: vec![
                    Layer { kind: LayerKind::Connected, points: pts.iter().map(|p| (p.0, p.1)).collect(), label: "Sensitivity".into(), ..Default::default() },
                    Layer { kind: LayerKind::Connected, points: pts.iter().map(|p| (p.0, p.2)).collect(), label: "Specificity".into(), ..Default::default() },
                ],
                xtitle: "Probability cutoff".into(),
                ytitle: "Sensitivity/Specificity".into(),
                legend: true,
                ..Default::default()
            };
            let extra = crate::graph::finish(&mut g, cmd, st)?;
            return Ok(extra);
        }
        Ok(String::new())
    }
}

// ---------------------------------------------------------------------------
// brier
// ---------------------------------------------------------------------------

pub struct Brier;

impl CommandHandler for Brier {
    fn name(&self) -> &'static str {
        "brier"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "brier outcomevar forecastvar"));
        }
        let mask = util::touse(ds, st, cmd)?;
        let (dc, fc) = (util::col(ds, &vars[0])?, util::col(ds, &vars[1])?);
        let rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask[i] && !is_missing(dc[i]) && !is_missing(fc[i])).collect();
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        if rows.iter().any(|&i| fc[i] < 0.0 || fc[i] > 1.0) {
            return Err(rc_err(459, format!("{} not between 0 and 1", vars[1])));
        }
        let d: Vec<f64> = rows.iter().map(|&i| if dc[i] != 0.0 { 1.0 } else { 0.0 }).collect();
        let f: Vec<f64> = rows.iter().map(|&i| fc[i]).collect();
        let n = d.len() as f64;
        let groups = util::opt_num(cmd, "group", 1)?.unwrap_or(10.0).max(1.0) as usize;
        let db = d.iter().sum::<f64>() / n;
        let fb = f.iter().sum::<f64>() / n;
        let var_f = f.iter().map(|x| (x - fb).powi(2)).sum::<f64>() / n;
        let cov = d.iter().zip(&f).map(|(a, b)| (a - db) * (b - fb)).sum::<f64>() / n;
        let oiv = db * (1.0 - db);
        let corr = cov / (var_f * oiv).sqrt();
        let brier = d.iter().zip(&f).map(|(a, b)| (b - a).powi(2)).sum::<f64>() / n;
        let yv: Vec<bool> = d.iter().map(|x| *x != 0.0).collect();
        let area = auc(&yv, &f, &vec![1.0; d.len()]);
        let (n1, n0) = (d.iter().sum::<f64>(), n - d.iter().sum::<f64>());
        let se0 = ((n1 + n0 + 1.0) / (12.0 * n1 * n0)).sqrt();
        let p_roc = 1.0 - dist::normal((area - 0.5) / se0);
        let znum: f64 = d.iter().zip(&f).map(|(a, b)| (a - b) * (1.0 - 2.0 * b)).sum();
        let zden: f64 = f.iter().map(|b| (1.0 - 2.0 * b).powi(2) * b * (1.0 - b)).sum::<f64>().sqrt();
        let z = znum / zden;
        let pz = 1.0 - dist::normal(z);
        // groups of forecasts: equal-width intervals of [0, 1]
        let gi: Vec<usize> = f.iter().map(|&x| (((x * groups as f64).ceil() as usize).max(1) - 1).min(groups - 1)).collect();
        let mut gs = vec![(0.0, 0.0, 0.0); groups];
        for t in 0..d.len() {
            gs[gi[t]].0 += 1.0;
            gs[gi[t]].1 += d[t];
            gs[gi[t]].2 += f[t];
        }
        let (mut sanders, mut murphy, mut relsm) = (0.0, 0.0, 0.0);
        for &(nj, dj, fj) in &gs {
            if nj == 0.0 {
                continue;
            }
            let (dm, fm) = (dj / nj, fj / nj);
            sanders += nj * dm * (1.0 - dm);
            murphy += nj * (dm - db).powi(2);
            relsm += nj * (fm - dm).powi(2);
        }
        sanders /= n;
        murphy /= n;
        relsm /= n;
        let brier_s = sanders + relsm;
        let f1 = d.iter().zip(&f).filter(|(a, _)| **a != 0.0).map(|(_, b)| *b).sum::<f64>() / n1;
        let f0 = d.iter().zip(&f).filter(|(a, _)| **a == 0.0).map(|(_, b)| *b).sum::<f64>() / n0;
        let var_min = (f1 - f0).powi(2) * oiv;
        let var_ex = var_f - var_min;
        let rel_la = (fb - db).powi(2);
        let cov2 = 2.0 * cov;
        let l = |label: &str, v: f64| format!("{:<31}{:>7.4}", label, v);
        let out = [
            String::new(),
            l("Mean probability of outcome", db),
            l("                 of forecast", fb),
            String::new(),
            l("Correlation", corr),
            format!("{}  p = {:.4}", l("ROC area", area), p_roc),
            String::new(),
            l("Brier score", brier),
            String::new(),
            format!("{}  p = {:.4}", l("Spiegelhalter's z-statistic", z), pz),
            l("Sanders-modified Brier score", brier_s),
            l("Sanders resolution", sanders),
            l("Outcome index variance", oiv),
            l("Murphy's resolution", murphy),
            l("Reliability-in-the-small", relsm),
            l("Forecast variance", var_f),
            l("Excess forecast variance", var_ex),
            l("Minimum forecast variance", var_min),
            l("Reliability-in-the-large", rel_la),
            l("2*Forecast-Outcome-Covar", cov2),
        ];
        let mut r = Results::new();
        r.set("N", n);
        r.set("p_roc", p_roc);
        r.set("roc_area", area);
        r.set("z", z);
        r.set("p", pz);
        r.set("brier", brier);
        r.set("brier_s", brier_s);
        r.set("sanders", sanders);
        r.set("oiv", oiv);
        r.set("murphy", murphy);
        r.set("relinsm", relsm);
        r.set("Var_f", var_f);
        r.set("Var_fex", var_ex);
        r.set("Var_fmin", var_min);
        r.set("relinla", rel_la);
        r.set("cov_2f", cov2);
        st.r = r;
        Ok(out.join("\n"))
    }
}
