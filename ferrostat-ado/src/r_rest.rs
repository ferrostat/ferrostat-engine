//! Base-reference commands that run other commands: `bayesboot` (Bayesian
//! bootstrap), `wildbootstrap` (wild cluster bootstrap inference after
//! linear regression), `reri` (interaction on the additive scale),
//! `serrbar` (standard-error bar charts) and `fvrevar` (factor-variable and
//! time-series operator expansion into temporary variables).

use ferrostat_cmd::dist;
use ferrostat_cmd::est::{self, TableSpec};
use ferrostat_cmd::eval::{rc_err, Env};
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_parser::parser::{find_top_level, split_options, split_top_level_words};

use crate::interpreter::{split_verb, AdoInterpreter, Flow};
use crate::resample::{self, arg, has, num_arg};

pub(crate) fn intercept(me: &mut AdoInterpreter, verb: &str, rest: &str) -> Result<Option<Flow>, String> {
    if me.engine.store.programs.contains_key(verb) {
        return Ok(None);
    }
    match verb {
        "bayesboot" => bayesboot(me, rest)?,
        "wildbootstrap" => wildbootstrap(me, rest)?,
        "reri" => reri(me, rest)?,
        "serrbar" => serrbar(me, rest)?,
        "fvrevar" => fvrevar(me, rest)?,
        _ => return Ok(None),
    }
    Ok(Some(Flow::Next))
}

fn split_opt_text(s: &str) -> (String, String) {
    match find_top_level(s, ',') {
        Some(i) => (s[..i].trim().to_string(), s[i + 1..].trim().to_string()),
        None => (s.trim().to_string(), String::new()),
    }
}

/// Insert a weight clause and options into a command line.
fn with_weight(cmd: &str, w: &str) -> String {
    let (main, opts) = split_opt_text(cmd);
    if opts.is_empty() { format!("{main} [iweight={w}]") } else { format!("{main} [iweight={w}], {opts}") }
}

// ---------------------------------------------------------------------------
// bayesboot
// ---------------------------------------------------------------------------

fn bayesboot(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let colon = find_top_level(rest, ':').ok_or_else(|| rc_err(198, "bayesboot: command required"))?;
    let head = rest[..colon].trim();
    let command = rest[colon + 1..].trim().to_string();
    let (explist, optpart) = split_opt_text(head);
    let opts = split_options(&optpart);
    let reps: usize = num_arg(&opts, "reps", 1)?.map_or(50, |r| r as usize);
    if let Some(s) = num_arg(&opts, "rseed", 2)? {
        me.engine.store.rng.borrow_mut().reseed(s as u64);
    }
    let level = resample::level_opt(me, &opts)?;
    let prior = arg(&opts, "priorpowers", 5).map(|p| varlist::resolve_name(&me.engine.dataset, p.trim())).transpose()?;
    let genv = arg(&opts, "generate", 3).map(|s| s.trim().to_string());
    // observed fit on the estimation sample
    me.emit(&format!("(running {} on estimation sample)", split_verb(&command).0));
    resample::run_cmd(me, &command, false)?;
    let e0 = me.engine.store.e.clone();
    let sample = e0.sample.clone().unwrap_or_else(|| vec![true; me.engine.dataset.n_obs]);
    let use_b = explist.trim().is_empty() || explist.trim() == "_b";
    let (names, obs): (Vec<String>, Vec<f64>) = if use_b {
        let b = e0.matrices.get("b").ok_or_else(|| rc_err(301, "command did not set e(b)"))?;
        (b.colnames.clone(), (0..b.cols).map(|j| b.get(0, j)).collect())
    } else {
        // expressions: r() / e() scalars by name, e.g. r(mean) or name=(r(mean))
        let mut nm = Vec::new();
        let mut vals = Vec::new();
        for (k, item) in split_top_level_words(&explist).iter().enumerate() {
            let (n, ex) = match item.split_once('=') {
                Some((a, b)) => (a.trim().to_string(), b.trim().trim_matches(|c| c == '(' || c == ')').to_string()),
                None => (format!("_bs_{}", k + 1), item.trim().trim_matches(|c| c == '(' || c == ')').to_string()),
            };
            nm.push(n);
            vals.push(scalar_of(me, &ex)?);
        }
        (nm, vals)
    };
    let exprs: Vec<String> = if use_b { Vec::new() } else { split_top_level_words(&explist).iter().map(|i| i.split_once('=').map_or(i.as_str(), |x| x.1).trim().trim_matches(|c| c == '(' || c == ')').to_string()).collect() };
    let n = me.engine.dataset.n_obs;
    let pw: Vec<f64> = match &prior {
        Some(p) => util::col(&me.engine.dataset, p)?,
        None => vec![0.0; n],
    };
    let wname = "__bayesboot_w";
    let mut draws: Vec<Vec<f64>> = Vec::new();
    let mut marks = Vec::new();
    let mut saved_w: Vec<Vec<f64>> = Vec::new();
    let nsamp = sample.iter().filter(|s| **s).count() as f64;
    for _ in 0..reps {
        let mut w = vec![0.0; n];
        let mut tot = 0.0;
        for i in 0..n {
            if sample[i] {
                let a = 1.0 + if is_missing(pw[i]) { 0.0 } else { pw[i] };
                let g = me.engine.store.rng.borrow_mut().gamma(a);
                w[i] = g;
                tot += g;
            }
        }
        for v in w.iter_mut() {
            *v *= nsamp / tot;
        }
        let vals: Vec<f64> = (0..n).map(|i| if sample[i] { w[i] } else { SYSMISS }).collect();
        if me.engine.dataset.var_index.contains_key(wname) {
            let _ = me.engine.dataset.drop_variable(wname);
        }
        me.engine.dataset.add_variable(util::new_double(wname, vals.clone())).map_err(|e| rc_err(110, e))?;
        let ok = resample::run_cmd(me, &with_weight(&command, wname), false);
        let row: Option<Vec<f64>> = match ok {
            Ok(()) if use_b => me.engine.store.e.matrices.get("b").map(|b| names.iter().map(|c| b.col_index(c).map_or(SYSMISS, |j| b.get(0, j))).collect()),
            Ok(()) => exprs.iter().map(|x| scalar_of(me, x).ok()).collect(),
            Err(_) => None,
        };
        marks.push(if row.is_some() { '.' } else { 'x' });
        if let Some(r) = row {
            draws.push(r);
        }
        if genv.is_some() {
            saved_w.push(vals);
        }
    }
    let _ = me.engine.dataset.drop_variable(wname);
    if let Some(g) = &genv {
        for (k, w) in saved_w.into_iter().enumerate() {
            let nm = format!("{g}{}", k + 1);
            if !me.engine.dataset.var_index.contains_key(&nm) {
                me.engine.dataset.add_variable(util::new_double(&nm, w)).map_err(|e| rc_err(110, e))?;
            }
        }
    }
    me.emit(&resample::dots_line("Bayesian bootstrap replications", reps, &marks));
    let k = names.len();
    let r = draws.len() as f64;
    let mut v = Matrix::zeros(k, k);
    let means: Vec<f64> = (0..k).map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / r).collect();
    for a in 0..k {
        for c in 0..k {
            v[(a, c)] = draws.iter().map(|d| (d[a] - means[a]) * (d[c] - means[c])).sum::<f64>() / (r - 1.0);
        }
    }
    let dep = e0.macros.get("depvar").cloned().unwrap_or_default();
    let omitted: Vec<bool> = names.iter().map(|c| resample::is_omitted_name(c) || resample::is_base_name(c)).collect();
    let is_base: Vec<bool> = names.iter().map(|c| resample::is_base_name(c)).collect();
    let mut out = vec![String::new(), "Bayesian bootstrap".to_string(), format!("Observation prior: {}", if prior.is_some() { "Informative" } else { "Improper" })];
    let title = resample::title_for(&e0);
    let mut rows: Vec<(String, String, String)> = vec![(title, "Number of obs".into(), util::commas(e0.get("N").unwrap_or(nsamp) as usize)), (String::new(), "Replications".into(), util::commas(draws.len()))];
    if use_b {
        let nm2: Vec<String> = names.clone();
        if let Some((w, df)) = resample::wald(&nm2, &obs, &v) {
            rows.push((String::new(), format!("Wald chi2({df})"), format!("{w:.2}")));
            rows.push((String::new(), "Prob > chi2".into(), format!("{:.4}", dist::chi2tail(df as f64, w))));
        }
        for (key, lab) in [("r2", "R-squared"), ("r2_a", "Adj R-squared")] {
            if let Some(x) = e0.get(key) {
                rows.push((String::new(), lab.into(), format!("{x:.4}")));
            }
        }
        if let Some(x) = e0.get("rmse") {
            rows.push((String::new(), "Root MSE".into(), fmt_g(x, 10)));
        }
    }
    out.extend(resample::header_lines("", &rows).into_iter().filter(|l| !l.trim().is_empty() || true));
    out.push(String::new());
    let mut tbl = est::coef_table(&TableSpec { depvar: &dep, names: &names, b: &obs, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Observed coefficient" });
    // header: "Bayesian bootstrap" above the standard error column and a
    // normal-based interval label
    if tbl.len() > 2 {
        let w = tbl[1].find('|').unwrap_or(13);
        tbl.insert(1, format!("{:>w$} |    Observed   Bayesian{:>26}Normal-based", "", ""));
        tbl[2] = tbl[2].replace("Observed coefficient", "coefficient").replace(" Coefficient", " coefficient");
        let _ = &tbl;
    }
    out.extend(tbl);
    me.emit(&out.join("\n"));
    // results
    let e = me.engine.store.begin_e();
    *e = e0.clone();
    e.set_mat("b", Matrix::row_vector(&obs).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", v.with_names(names.clone(), names));
    e.set("N_reps", draws.len() as f64);
    e.set_str("prefix", "bayesboot");
    e.set_str("vce", "bayesboot");
    e.set_str("vcetype", "Bayesian bootstrap");
    e.set_str("cmdline", format!("bayesboot {rest}"));
    Ok(())
}

fn scalar_of(me: &mut AdoInterpreter, ex: &str) -> Result<f64, String> {
    let s = ex.trim();
    let get = |res: &Results, k: &str| res.get(k).ok_or_else(|| rc_err(111, format!("{s} not found")));
    if let Some(k) = s.strip_prefix("r(").and_then(|x| x.strip_suffix(')')) {
        return get(&me.engine.store.r, k);
    }
    if let Some(k) = s.strip_prefix("e(").and_then(|x| x.strip_suffix(')')) {
        return get(&me.engine.store.e, k);
    }
    if let Some(k) = s.strip_prefix("_b[").and_then(|x| x.strip_suffix(']')) {
        let b = me.engine.store.e.matrices.get("b").ok_or_else(|| rc_err(301, "e(b) not found"))?;
        return b.col_index(k).map(|j| b.get(0, j)).ok_or_else(|| rc_err(111, format!("{k} not found")));
    }
    Err(rc_err(198, format!("{s}: expression not supported")))
}

// ---------------------------------------------------------------------------
// wildbootstrap
// ---------------------------------------------------------------------------

fn wildbootstrap(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (main, optpart) = split_opt_text(rest);
    let opts = split_options(&optpart);
    let (verb, _) = split_verb(&main);
    if verb != "regress" && verb != "reg" && verb != "areg" {
        return Err(rc_err(198, format!("wildbootstrap {verb} not supported; regress required")));
    }
    let cl = arg(&opts, "cluster", 2).ok_or_else(|| rc_err(198, "option cluster() required"))?.trim().to_string();
    let cl = varlist::resolve_name(&me.engine.dataset, &cl)?;
    let reps: usize = num_arg(&opts, "reps", 1)?.map_or(1000, |r| r as usize);
    let seed = num_arg(&opts, "rseed", 2)?;
    let level = resample::level_opt(me, &opts)?;
    let ew = arg(&opts, "errorweight", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "rademacher".into());
    let ewname = match ew.as_str() {
        "mammen" => "Mammen",
        "webb" => "Webb",
        "normal" | "gaussian" => "Normal",
        _ => "Rademacher",
    };
    // pass-through options for the regression (absorb() for areg etc.)
    let reg_opts: Vec<String> = opts.iter().filter(|(o, _)| !matches!(o.to_lowercase().as_str(), "cluster" | "reps" | "rseed" | "errorweight" | "coefficients" | "test" | "level" | "nodots" | "ptype")).map(|(o, a)| a.as_ref().map_or(o.clone(), |x| format!("{o}({x})"))).collect();
    let line = format!("{main}, vce(cluster {cl}) {}", reg_opts.join(" "));
    resample::run_cmd(me, &line, false)?;
    let e0 = me.engine.store.e.clone();
    let b = e0.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let coefs: Vec<String> = match arg(&opts, "coefficients", 4) {
        Some(c) => split_top_level_words(&c),
        None => b.colnames.iter().filter(|c| c.as_str() != "_cons" && !resample::is_base_name(c) && !resample::is_omitted_name(c)).cloned().collect(),
    };
    let mut hyps: Vec<(String, String)> = coefs.iter().map(|c| (format!("{c} = 0"), c.clone())).collect();
    if let Some(t) = arg(&opts, "test", 4) {
        hyps = vec![(t.trim().to_string(), t.trim().to_string())];
    }
    let mut rows = Vec::new();
    for (label, hyp) in &hyps {
        me.emit(&format!("Performing {} replications for p-value for {label} ...", util::commas(reps)));
        me.emit(&format!("Computing confidence interval for {}", label.split('=').next().unwrap_or(label).trim()));
        let seed_opt = seed.map_or(String::new(), |s| format!(" seed({})", s as u64));
        resample::run_cmd(me, &format!("boottest {hyp}, cluster({cl}) reps({reps}) weighttype({ew}) level({}){seed_opt}", fmt_g(level, 4)), false)?;
        let r = me.engine.store.r.clone();
        let p = r.get("p").unwrap_or(SYSMISS);
        let t = r.get("t").unwrap_or_else(|| r.get("F").map_or(SYSMISS, |f| f.sqrt()));
        let (lo, hi) = r.matrices.get("CI").map_or((SYSMISS, SYSMISS), |m| (m.get(0, 0), m.get(0, 1)));
        // estimate of the linear combination
        let (rv, c) = { let hs = if hyp.contains('=') { hyp.clone() } else { format!("{hyp} = 0") }; ferrostat_cmd::xmodels::parse_constraint(&hs, &b.colnames)? };
        let est_v: f64 = rv.iter().enumerate().map(|(j, a)| a * b.get(0, j)).sum::<f64>() - c;
        rows.push((label.clone(), est_v, t, p, lo, hi));
        me.engine.store.e = e0.clone();
    }
    // cluster sizes
    let sample = e0.sample.clone().unwrap_or_default();
    let cc = util::col(&me.engine.dataset, &cl)?;
    let mut sizes: std::collections::BTreeMap<i64, usize> = Default::default();
    for (i, s) in sample.iter().enumerate() {
        if *s {
            *sizes.entry((cc[i] * 1e6).round() as i64).or_default() += 1;
        }
    }
    let g = sizes.len();
    let (mn, mx) = (sizes.values().min().copied().unwrap_or(0), sizes.values().max().copied().unwrap_or(0));
    let nobs = e0.get("N").unwrap_or(0.0);
    let avg = nobs / g.max(1) as f64;
    let dep = e0.macros.get("depvar").cloned().unwrap_or_default();
    let hr = |l: &str, k: &str, v: String| format!("{:<51}{:<18} = {:>5}", l, k, v);
    let mut out = vec![String::new()];
    out.push(hr("Wild cluster bootstrap", "Number of obs", util::commas(nobs as usize)));
    out.push(hr("Linear regression", "Number of clusters", util::commas(g)));
    out.push(format!("{:<51}Cluster size:", ""));
    out.push(format!("{:<66}min = {:>5}", format!("Cluster variable: {cl}"), mn));
    out.push(format!("{:<66}avg = {:>5}", format!("Error weight: {ewname}"), fmt_g((avg * 10.0).round() / 10.0, 5)));
    out.push(format!("{:<66}max = {:>5}", "", mx));
    out.push(String::new());
    out.push("-".repeat(78));
    out.push(format!("{:>24} |   Estimate      t  p-value    [{}% conf. interval]", ferrostat_cmd::summarize::abbrev(&dep, 24), fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(25), "-".repeat(52)));
    out.push(format!("{:<24} |", "constraint"));
    for (label, est_v, t, p, lo, hi) in &rows {
        out.push(format!("{:>24} | {:>10}  {:>6.2}  {:>6.3}  {:>10}  {:>10}", label, fmt_g(*est_v, 8), t, p, fmt_g(*lo, 8), fmt_g(*hi, 8)));
    }
    out.push("-".repeat(78));
    me.emit(&out.join("\n"));
    let k = rows.len();
    let mut m = Matrix::zeros(k, 5);
    for (i, (_, ev, t, p, lo, hi)) in rows.iter().enumerate() {
        for (j, x) in [*ev, *t, *p, *lo, *hi].iter().enumerate() {
            m[(i, j)] = *x;
        }
    }
    let e = &mut me.engine.store.e;
    e.set_mat("b_wild", m.with_names(rows.iter().map(|r| r.0.clone()).collect(), ["estimate", "t", "p", "lb", "ub"].iter().map(|s| s.to_string()).collect()));
    e.set("N_clust", g as f64);
    e.set("reps", reps as f64);
    e.set_str("errorweight", ew);
    e.set_str("wildbootstrap", "wildbootstrap");
    Ok(())
}

// ---------------------------------------------------------------------------
// reri
// ---------------------------------------------------------------------------

fn reri(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (main, optpart) = split_opt_text(rest);
    let opts = split_options(&optpart);
    let words = split_top_level_words(&main);
    if words.len() < 4 {
        return Err(rc_err(198, "reri cmd depvar var1 var2 [covariates] required"));
    }
    let cmd = words[0].to_lowercase();
    let dep = words[1].clone();
    let v1 = varlist::resolve_name(&me.engine.dataset, &words[2])?;
    let v2 = varlist::resolve_name(&me.engine.dataset, &words[3])?;
    let cov = words[4..].join(" ");
    let level = resample::level_opt(me, &opts)?;
    let pass: Vec<String> = opts.iter().filter(|(o, _)| !matches!(o.to_lowercase().as_str(), "rr" | "noisily" | "level" | "nocoef" | "or")).map(|(o, a)| a.as_ref().map_or(o.clone(), |x| format!("{o}({x})"))).collect();
    let extra = match cmd.as_str() {
        "binreg" => "rr",
        "logit" | "logistic" => "or",
        "poisson" | "nbreg" => "irr",
        _ => "",
    };
    let line = format!("{cmd} {dep} i.{v1}##i.{v2} {cov}, {extra} {}", pass.join(" "));
    me.emit(&format!("Fitting {cmd} ..."));
    resample::run_cmd(me, &line, has(&opts, "noisily", 1))?;
    let e = me.engine.store.e.clone();
    let b = e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
    let v = e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
    let find = |pred: &dyn Fn(&str) -> bool| -> Result<usize, String> {
        b.colnames.iter().position(|c| {
            let n = c.rsplit(':').next().unwrap_or(c);
            pred(n)
        }).ok_or_else(|| rc_err(111, "interaction coefficients not found"))
    };
    let lev = |n: &str, var: &str| -> bool { n.split_once('.').is_some_and(|(l, vv)| vv == var && !l.ends_with('b') && !l.ends_with('o')) };
    let j1 = find(&|n| !n.contains('#') && lev(n, &v1))?;
    let j2 = find(&|n| !n.contains('#') && lev(n, &v2))?;
    let j3 = find(&|n| n.split('#').count() == 2 && n.split('#').next().is_some_and(|a| lev(a, &v1)) && n.split('#').nth(1).is_some_and(|a| lev(a, &v2)))?;
    let (b1, b2, b3) = (b.get(0, j1), b.get(0, j2), b.get(0, j3));
    let idx = [j1, j2, j3];
    let vv = |a: usize, c: usize| v[(idx[a], idx[c])];
    let quad = |g: [f64; 3]| -> f64 { (0..3).map(|a| (0..3).map(|c| g[a] * vv(a, c) * g[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt() };
    let (r10, r01, r11) = (b1.exp(), b2.exp(), (b1 + b2 + b3).exp());
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    // excess relative risks with intervals from the log scale
    let err = |lr: f64, g: [f64; 3]| -> (f64, f64, f64, f64, f64) {
        let se_l = quad(g);
        let est_v = lr.exp() - 1.0;
        let se = lr.exp() * se_l;
        let z = lr / se_l;
        (est_v, se, z, (lr - crit * se_l).exp() - 1.0, (lr + crit * se_l).exp() - 1.0)
    };
    let e01 = err(b2, [0.0, 1.0, 0.0]);
    let e10 = err(b1, [1.0, 0.0, 0.0]);
    let e11 = err(b1 + b2 + b3, [1.0, 1.0, 1.0]);
    let reri = r11 - r10 - r01 + 1.0;
    let g_reri = [r11 - r10, r11 - r01, r11];
    let se_reri = quad(g_reri);
    let ap = reri / r11;
    // AP = 1 - exp(-b1-b2-b3)(exp(b1) + exp(b2) - 1)
    let g_ap = [(-(b1 + b2 + b3)).exp() * (r10 + r01 - 1.0) - (-(b1 + b2 + b3)).exp() * r10, (-(b1 + b2 + b3)).exp() * (r10 + r01 - 1.0) - (-(b1 + b2 + b3)).exp() * r01, (-(b1 + b2 + b3)).exp() * (r10 + r01 - 1.0)];
    let se_ap = quad(g_ap);
    let si = (r11 - 1.0) / (r10 + r01 - 2.0);
    let g_lsi = [r11 / (r11 - 1.0) - r10 / (r10 + r01 - 2.0), r11 / (r11 - 1.0) - r01 / (r10 + r01 - 2.0), r11 / (r11 - 1.0)];
    let se_lsi = quad(g_lsi);
    let lbl = |var: &str, val_idx: usize| -> String {
        let c = util::col(&me.engine.dataset, var).unwrap_or_default();
        let mut lv: Vec<f64> = c.iter().cloned().filter(|x| !is_missing(*x)).collect();
        lv.sort_by(f64::total_cmp);
        lv.dedup();
        lv.get(val_idx).map(|l| { let ds = &me.engine.dataset; ds.get_var(var).and_then(|vv| ds.value_labels.get(&vv.value_label_name)).and_then(|vl| vl.lookup(*l)).map(str::to_string).unwrap_or_else(|| fmt_g(*l, 9)) }).unwrap_or_default()
    };
    let (a0, a1, c0, c1) = (lbl(&v1, 0), lbl(&v1, 1), lbl(&v2, 0), lbl(&v2, 1));
    let _ = (&a0, &c0);
    let (label_t, title) = match extra {
        "or" => ("EOR", "odds"),
        "irr" => ("EIRR", "rate"),
        _ => ("ERR", "risk"),
    };
    let _ = title;
    let mut out = vec![String::new(), format!("Interaction of {v1} and {v2} on an additive scale"), String::new()];
    out.push(format!("{:<48}Number of observations = {}", format!("Model: {}", resample::title_for(&e)), util::commas(e.get("N").unwrap_or(0.0) as usize)));
    out.push(String::new());
    out.push(format!("{v1}#{v2}"));
    out.push(format!("        - +    {a0}#{c1}"));
    out.push(format!("        + -    {a1}#{c0}"));
    out.push(format!("        + +    {a1}#{c1}"));
    out.push(String::new());
    out.push("-".repeat(78));
    out.push(format!("{:>12} |{:>11}{:>11}{:>9}{:>9}     [{}% conf. interval]", "", label_t, "Std. err.", "z", "P>|z|", fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    out.push(format!("{v1}#{v2}"));
    let row = |l: &str, est_v: f64, se: f64, z: f64, lo: f64, hi: f64| format!("{:>12} |  {:>9}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", l, fmt_g(est_v, 8), fmt_g(se, 8), z, 2.0 * dist::normal(-z.abs()), fmt_g(lo, 8), fmt_g(hi, 8));
    out.push(row("- +", e01.0, e01.1, e01.2, e01.3, e01.4));
    out.push(row("+ -", e10.0, e10.1, e10.2, e10.3, e10.4));
    out.push(row("+ +", e11.0, e11.1, e11.2, e11.3, e11.4));
    out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
    out.push(row("RERI", reri, se_reri, reri / se_reri, reri - crit * se_reri, reri + crit * se_reri));
    out.push(row("Attr. prop.", ap, se_ap, ap / se_ap, ap - crit * se_ap, ap + crit * se_ap));
    let lsi = si.ln();
    out.push(row("Synergy ind.", si, si * se_lsi, lsi / se_lsi, (lsi - crit * se_lsi).exp(), (lsi + crit * se_lsi).exp()));
    out.push("-".repeat(78));
    out.push("Note: P>|z| for synergy index (SI) is for test H0: SI = 1.".to_string());
    me.emit(&out.join("\n"));
    let r = &mut me.engine.store.r;
    *r = Results::new();
    r.set("reri", reri);
    r.set("se_reri", se_reri);
    r.set("ap", ap);
    r.set("se_ap", se_ap);
    r.set("si", si);
    r.set("se_si", si * se_lsi);
    r.set("level", level);
    Ok(())
}

// ---------------------------------------------------------------------------
// serrbar
// ---------------------------------------------------------------------------

fn serrbar(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (main, optpart) = split_opt_text(rest);
    let (vars_part, cond) = {
        let lw = main.to_lowercase();
        let cut = [" if ", " in "].iter().filter_map(|k| lw.find(k)).min().unwrap_or(main.len());
        (main[..cut].to_string(), main[cut..].to_string())
    };
    let vars = varlist::expand(&me.engine.dataset, &vars_part)?;
    if vars.len() != 3 {
        return Err(rc_err(102, "serrbar mvar svar xvar required"));
    }
    let opts = split_options(&optpart);
    let scale = num_arg(&opts, "scale", 2)?.unwrap_or(1.0);
    let mv = arg(&opts, "mvopts", 3).unwrap_or_default();
    let pass: Vec<String> = opts.iter().filter(|(o, _)| !matches!(o.to_lowercase().as_str(), "scale" | "sc" | "mvopts" | "mvo")).map(|(o, a)| a.as_ref().map_or(o.clone(), |x| format!("{o}({x})"))).collect();
    let (m, s, x) = (&vars[0], &vars[1], &vars[2]);
    let mc = util::col(&me.engine.dataset, m)?;
    let sc = util::col(&me.engine.dataset, s)?;
    let hi: Vec<f64> = mc.iter().zip(&sc).map(|(a, b)| if is_missing(*a) || is_missing(*b) { SYSMISS } else { a + scale * b }).collect();
    let lo: Vec<f64> = mc.iter().zip(&sc).map(|(a, b)| if is_missing(*a) || is_missing(*b) { SYSMISS } else { a - scale * b }).collect();
    for (n, c) in [("__serr_hi", hi), ("__serr_lo", lo)] {
        if me.engine.dataset.var_index.contains_key(n) {
            let _ = me.engine.dataset.drop_variable(n);
        }
        me.engine.dataset.add_variable(util::new_double(n, c)).map_err(|e| rc_err(110, e))?;
    }
    let mlab = me.engine.dataset.get_var(m).map(|v| if v.label.is_empty() { m.clone() } else { v.label.clone() }).unwrap_or_default();
    let line = format!("twoway (rcap __serr_hi __serr_lo {x}{cond}) (scatter {m} {x}{cond}, {mv}), legend(off) ytitle(\"{mlab}\") {}", pass.join(" "));
    let r = me.exec_text(&line, None);
    let _ = me.engine.dataset.drop_variable("__serr_hi");
    let _ = me.engine.dataset.drop_variable("__serr_lo");
    r.map(|_| ())
}

// ---------------------------------------------------------------------------
// fvrevar
// ---------------------------------------------------------------------------

/// Expand `i(2,3)bn.var`-style level lists into `2.var 3.var` tokens and
/// return the coefficient-style names of a factor-variable list.
fn fv_names(me: &AdoInterpreter, text: &str) -> Result<Vec<String>, String> {
    let ds = &me.engine.dataset;
    let mut out = Vec::new();
    for tok in split_top_level_words(text) {
        // i(numlist)[b|bn].var
        if let Some(r) = tok.strip_prefix("i(") {
            if let Some((list, after)) = r.split_once(')') {
                let var = after.split_once('.').map(|x| x.1).unwrap_or(after);
                let var = varlist::resolve_name(ds, var)?;
                for l in list.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()) {
                    out.push(format!("{l}.{var}"));
                }
                continue;
            }
        }
        if !tok.contains('.') && !tok.contains('#') {
            out.extend(varlist::expand(ds, &tok)?);
            continue;
        }
        // factor terms: names from a design over all observations
        let has_fv = tok.split('#').any(|p| p.split_once('.').is_some_and(|(o, _)| {
            let o = o.to_lowercase();
            o.starts_with('i') || o.starts_with('c') || o.trim_end_matches(['b', 'n', 'o']).parse::<f64>().is_ok()
        }));
        if !has_fv {
            out.push(tok.clone());
            continue;
        }
        let mut parts: Vec<Vec<String>> = Vec::new();
        for p in tok.split('#').filter(|s| !s.is_empty()) {
            let (op, var) = p.split_once('.').map(|(a, b)| (a.to_lowercase(), b.to_string())).unwrap_or((String::new(), p.to_string()));
            if op == "c" || op.is_empty() {
                parts.push(vec![format!("c.{}", varlist::resolve_name(ds, &var)?)]);
                continue;
            }
            if let Some(lst) = op.strip_prefix("i(") {
                let (list, _) = lst.split_once(')').unwrap_or((lst, ""));
                let var = varlist::resolve_name(ds, &var)?;
                parts.push(list.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).map(|l| format!("{l}.{var}")).collect());
                continue;
            }
            let var = varlist::resolve_name(ds, &var)?;
            if op.starts_with('i') {
                let c = util::col(ds, &var)?;
                let mut lv: Vec<f64> = c.iter().cloned().filter(|x| !is_missing(*x)).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                parts.push(lv.iter().map(|l| format!("{}.{var}", fmt_g(*l, 12))).collect());
            } else {
                parts.push(vec![format!("{}.{var}", op.trim_end_matches(['b', 'n', 'o']))]);
            }
        }
        let mut acc: Vec<String> = vec![String::new()];
        for p in parts {
            let mut next = Vec::new();
            for a in &acc {
                for q in &p {
                    next.push(if a.is_empty() { q.clone() } else { format!("{a}#{q}") });
                }
            }
            acc = next;
        }
        out.extend(acc.into_iter().map(|n| if n.split('#').count() == 1 { n.trim_start_matches("c.").to_string() } else { n }));
    }
    Ok(out)
}

fn fvrevar(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let (main, optpart) = split_opt_text(rest);
    let opts = split_options(&optpart);
    let (vtext, cond) = {
        let lw = main.to_lowercase();
        let cut = [" if ", " in "].iter().filter_map(|k| lw.find(k)).min().unwrap_or(main.len());
        (main[..cut].to_string(), main[cut..].to_string())
    };
    let _ = cond;
    if has(&opts, "list", 1) {
        // base variable names only
        let mut names: Vec<String> = Vec::new();
        for tok in split_top_level_words(&vtext) {
            for p in tok.split('#') {
                let var = p.rsplit('.').next().unwrap_or(p);
                for v in varlist::expand(&me.engine.dataset, var).unwrap_or_else(|_| vec![var.to_string()]) {
                    if !names.contains(&v) {
                        names.push(v);
                    }
                }
            }
        }
        let r = &mut me.engine.store.r;
        *r = Results::new();
        r.set_str("varlist", names.join(" "));
        return Ok(());
    }
    let tsonly = has(&opts, "tsonly", 2);
    let names = fv_names(me, &vtext)?;
    let mut out: Vec<String> = Vec::new();
    // reuse variables created for identical terms earlier in this session
    let n = me.engine.dataset.n_obs;
    for nm in names {
        let plain = me.engine.dataset.var_index.contains_key(&nm);
        let is_ts = nm.split('#').any(|p| p.split_once('.').is_some_and(|(o, _)| o.chars().next().is_some_and(|c| matches!(c.to_ascii_uppercase(), 'L' | 'F' | 'D' | 'S')) && o.trim_end_matches(char::is_numeric).len() <= 2 && !o.trim_end_matches(['b', 'n', 'o']).parse::<f64>().is_ok()));
        if plain || (tsonly && !is_ts) {
            out.push(nm);
            continue;
        }
        let key = format!("_fvrevar:{nm}");
        if let Some(v) = me.engine.store.settings.get(&key).cloned() {
            if me.engine.dataset.var_index.contains_key(&v) {
                out.push(v);
                continue;
            }
        }
        let vals: Vec<f64> = {
            let env = Env::with_store(&me.engine.dataset, &me.engine.store);
            (0..n).map(|i| ferrostat_cmd::postest::coef_value(&env, &nm, i)).collect::<Result<_, _>>()?
        };
        me.temp_counter += 1;
        let tv = format!("__{:06}", me.temp_counter);
        if let Some(f) = me.tempvars.last_mut() {
            f.push(tv.clone());
        }
        me.engine.dataset.add_variable(util::new_double(&tv, vals)).map_err(|e| rc_err(110, e))?;
        me.engine.store.settings.insert(key, tv.clone());
        out.push(tv);
    }
    let r = &mut me.engine.store.r;
    *r = Results::new();
    r.set_str("varlist", out.join(" "));
    Ok(())
}
