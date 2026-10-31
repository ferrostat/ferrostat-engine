//! Running a built model: mode search, sampling, the results table and
//! the stored results (`e()`), including the simulation sample used by the
//! postestimation commands.

use ferrostat_core::DataSet;

use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;

use super::bayesmh::{self, Built};
use super::mcmc::{self, Sim};
use super::model::{Fam, Model, PK};
use super::prior::Support;
use super::stats;
use super::text;

/// Columns of a number in the results table.
pub fn num9(v: f64) -> String {
    if !v.is_finite() {
        return ".".into();
    }
    fmt_g(v, 9)
}

fn lead(v: f64) -> String {
    // `.3512` style for rates
    let s = format!("{v:.4}");
    if let Some(r) = s.strip_prefix("0.") { format!(".{r}") } else { s }
}

pub fn hdr(left: &str, key: &str, val: &str) -> String {
    let kw = 17usize.max(key.len() + 1);
    let lw = 47 - (kw - 17).min(10);
    let left = if left.len() > lw { left.to_string() } else { format!("{left:<lw$}") };
    format!("{left}{key:<kw$}= {val:>10}")
}

/// Which parameters appear in the table.
pub fn shown(m: &Model, show_re: bool, noshow: &[usize]) -> Vec<usize> {
    (0..m.params.len()).filter(|&j| (show_re || m.params[j].re.is_none()) && !noshow.contains(&j)).collect()
}

/// Results table rows for parameters `idx` (labels grouped by equation).
pub fn table(m: &Model, sim: &Sim, idx: &[usize], level: f64, use_hpd: bool, batch: usize) -> (Vec<String>, Vec<stats::Summary>) {
    let mut out = Vec::new();
    let lw = 12usize.max(idx.iter().map(|&j| m.params[j].name.rsplit(':').next().unwrap_or("").len().min(32)).max().unwrap_or(0));
    let line = |c: char| format!("{}{}{}", "-".repeat(lw + 1), c, "-".repeat(64));
    out.push("-".repeat(lw + 66));
    out.push(format!("{:>w$} |{:>64}", "", if use_hpd { "HPD" } else { "Equal-tailed" }, w = lw));
    out.push(format!("{:>w$} |      Mean   Std. dev.     MCSE     Median  [{}% cred. interval]", "", fmt_g(level, 4), w = lw));
    out.push(line('+'));
    let mut last_eq: Option<String> = None;
    let mut sums = Vec::new();
    for (pos, &j) in idx.iter().enumerate() {
        let name = &m.params[j].name;
        let (eq, nm) = match name.split_once(':') {
            Some((e, n)) if !name.contains('[') || e.len() < name.find('[').unwrap() => (Some(e.to_string()), n.to_string()),
            _ => (None, name.clone()),
        };
        if eq != last_eq {
            if pos > 0 {
                out.push(line('+'));
            }
            if let Some(e) = &eq {
                out.push(format!("{:<w$} |", e, w = lw));
            }
            last_eq = eq.clone();
        }
        let x: Vec<f64> = sim.draws.iter().map(|d| d[j]).collect();
        let s = stats::summarize(&x, level, use_hpd, batch);
        let lab = if nm.len() > lw { format!("{}~{}", &nm[..lw - 3], &nm[nm.len() - 2..]) } else { nm.clone() };
        out.push(format!("{:>w$} | {:>9}  {:>9}  {:>8}  {:>9}  {:>9}  {:>9}", lab, num9(s.mean), num9(s.sd), num9(s.mcse), num9(s.median), num9(s.lo), num9(s.hi), w = lw));
        sums.push(s);
    }
    out.push("-".repeat(lw + 66));
    (out, sums)
}

fn sampling_label(sim: &Sim) -> &'static str {
    match (sim.any_mh, sim.any_gibbs) {
        (true, true) => "Metropolis-Hastings and Gibbs sampling",
        (false, true) => "Gibbs sampling",
        _ => "Random-walk Metropolis-Hastings sampling",
    }
}

/// Model summary block.
pub fn model_summary(m: &Model) -> Vec<String> {
    let mut out = vec!["Model summary".to_string(), "-".repeat(78), "Likelihood: ".to_string()];
    for l in &m.liks {
        let lhs = l.eqs.iter().map(|&k| m.eqs[k].depvar.clone()).filter(|s| !s.is_empty()).collect::<Vec<_>>();
        let lhs = if lhs.is_empty() { "_".to_string() } else { lhs[0].clone() };
        let arg = match &l.fam {
            Fam::Normal(a) => format!("normal(xb_{},{})", l.eqs.first().map(|&k| m.eqs[k].name.clone()).unwrap_or_default(), a.label(m)),
            Fam::Quad { .. } => "normal approximation of the likelihood at its maximum".into(),
            f => format!("{}(xb_{})", f.label(), l.eqs.first().map(|&k| m.eqs[k].name.clone()).unwrap_or_default()),
        };
        out.push(format!("  {lhs} ~ {arg}"));
    }
    out.push(String::new());
    out.push("Priors: ".to_string());
    for p in &m.priors {
        let names: Vec<&str> = p.params.iter().map(|&j| m.params[j].name.as_str()).collect();
        let lhs = if p.params.iter().all(|&j| m.params[j].re.is_some()) && !p.params.is_empty() {
            let t = m.params[p.params[0]].re.unwrap();
            format!("{{{}[{}]}}", m.re_terms[t].name, m.re_terms[t].groupvar)
        } else if names.len() > 4 {
            format!("{{{} ... {}}}", names[0], names[names.len() - 1])
        } else {
            names.iter().map(|n| format!("{{{n}}}")).collect::<Vec<_>>().join(" ")
        };
        out.push(format!("  {lhs} ~ {}", p.label));
    }
    out.push("-".repeat(78));
    out
}

/// Posterior means of the draws.
pub fn post_mean(sim: &Sim, np: usize) -> Vec<f64> {
    (0..np).map(|j| stats::mean(&sim.draws.iter().map(|d| d[j]).collect::<Vec<_>>())).collect()
}

/// DIC and log marginal likelihood (Laplace-Metropolis).
pub fn ic(m: &Model, sim: &Sim) -> (f64, f64, f64) {
    let np = m.params.len();
    let mut tbar = post_mean(sim, np);
    // discrete parameters: posterior mode
    for j in 0..np {
        if let Support::Discrete(..) = m.params[j].support {
            let mut counts = std::collections::BTreeMap::new();
            for d in &sim.draws {
                *counts.entry(d[j] as i64).or_insert(0usize) += 1;
            }
            if let Some((v, _)) = counts.iter().max_by_key(|x| x.1) {
                tbar[j] = *v as f64;
            }
        }
    }
    let llbar = m.loglik(&tbar);
    let dbar = -2.0 * stats::mean(&sim.ll);
    let dic = if llbar.is_finite() { 2.0 * dbar + 2.0 * llbar } else { f64::NAN };
    let cont: Vec<usize> = (0..np).filter(|&j| !matches!(m.params[j].support, Support::Discrete(..))).filter(|&j| stats::var(&sim.draws.iter().map(|d| d[j]).collect::<Vec<_>>()) > 0.0).collect();
    let lp = m.log_prior(&tbar);
    let improper = m.priors.iter().any(|p| matches!(&p.kind, PK::Uni(u, _) if !u.proper()));
    let lml = if improper || !lp.is_finite() || !llbar.is_finite() {
        f64::NAN
    } else {
        let cov = stats::covariance(&sim.draws, &cont);
        match stats::logdet(&cov) {
            Some(ld) => cont.len() as f64 / 2.0 * (2.0 * std::f64::consts::PI).ln() + 0.5 * ld + llbar + lp,
            None => f64::NAN,
        }
    };
    (dic, lml, dbar)
}

pub struct Shown {
    pub text: String,
}

fn miss(v: f64) -> f64 {
    if v.is_finite() { v } else { ferrostat_core::missing::SYSMISS }
}

/// Estimate: build from `rest` (text after the verb), sample, display, store.
pub fn estimate(ds: &mut DataSet, st: &mut Store, rest: &str) -> Result<String, String> {
    let Built { model: m, mut init, user_init, settings, spec, title } = bayesmh::build(ds, st, rest)?;
    if let Some(s) = spec.num("rseed", 5)? {
        st.rng.borrow_mut().reseed(s as u64);
    }
    let level = match spec.num("clevel", 2)? {
        Some(v) => v,
        None => spec.num("level", 3)?.or_else(|| st.settings.get("clevel").and_then(|v| v.parse().ok())).unwrap_or(st.level),
    };
    let use_hpd = spec.has("hpd", 3);
    let batch = spec.num("batch", 5)?.unwrap_or(0.0) as usize;
    if spec.has("dryrun", 3) {
        return Ok(model_summary(&m).join("\n"));
    }
    // mode search for the proposal covariance (and starting values)
    let free: Vec<usize> = (0..m.params.len()).filter(|&j| m.params[j].re.is_none() && m.params[j].mat.is_none() && !matches!(m.params[j].support, Support::Discrete(..))).collect();
    let user0 = init.clone();
    let hinv = if spec.has("nomode", 5) { None } else { mcmc::find_mode(&m, &mut init, &free) };
    for j in 0..init.len() {
        if user_init[j] {
            init[j] = user0[j];
        }
    }
    if !m.log_post(&init).is_finite() {
        init = user0.clone();
    }
    let sim = {
        let mut rng = st.rng.borrow_mut();
        mcmc::run(&m, &init, &settings, &mut rng, hinv.as_ref().map(|h| (free.as_slice(), h)))?
    };
    let show_re = spec.has("showreffects", 5);
    let noshow: Vec<usize> = match spec.opt("noshow", 6) {
        Some(Some(s)) => bayesmh::resolve_refs(&mut m.clone(), s, false).unwrap_or_default(),
        _ => Vec::new(),
    };
    let idx = shown(&m, show_re, &noshow);
    let (tbl, sums) = table(&m, &sim, &idx, level, use_hpd, batch);
    let (dic, lml, dbar) = ic(&m, &sim);
    let ndraws = sim.draws.len();
    let per_chain = ndraws / settings.nchains.max(1);
    let effs: Vec<f64> = sums.iter().map(|s| s.ess / ndraws.max(1) as f64).collect();
    let (emin, emax) = effs.iter().fold((f64::INFINITY, 0.0f64), |a, &e| (a.0.min(e), a.1.max(e)));
    let eavg = stats::mean(&effs);
    let arate = stats::mean(&sim.arate);
    // Gelman-Rubin
    let rc: Vec<f64> = if settings.nchains > 1 {
        idx.iter().map(|&j| {
            let chains: Vec<Vec<f64>> = (0..settings.nchains).map(|c| sim.draws.iter().zip(&sim.chain).filter(|(_, ch)| **ch == c).map(|(d, _)| d[j]).collect()).collect();
            stats::grubin(&chains)
        }).collect()
    } else {
        Vec::new()
    };
    let rcmax = rc.iter().copied().fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a });
    // output
    let prefix_cmd = spec.opt("_prefix", 7).and_then(|x| x.clone());
    let ttl = spec.opt("_title", 6).and_then(|x| x.clone()).unwrap_or(title.clone());
    let mut out = Vec::new();
    if !spec.has("nomodelsummary", 5) && !spec.has("noheader", 5) && prefix_cmd.is_none() {
        out.extend(model_summary(&m));
        out.push(String::new());
    }
    if !spec.has("noheader", 5) {
        let nobs = spec.num("_nobs", 5)?.map(|v| v as usize).unwrap_or(m.n);
        let it = settings.burnin + settings.mcmcsize * settings.thin;
        let left0 = format!("Bayesian {ttl}");
        let mut rows = vec![
            hdr(&left0, "MCMC iterations", &util::commas(it)),
            hdr(sampling_label(&sim), "Burn-in", &util::commas(settings.burnin)),
            hdr("", "MCMC sample size", &util::commas(per_chain)),
        ];
        if settings.nchains > 1 {
            rows.push(hdr("", "Number of chains", &settings.nchains.to_string()));
        }
        if nobs > 0 && (!m.eqs.is_empty() || spec.has("_nobs", 5)) {
            rows.push(hdr("", "Number of obs", &util::commas(nobs)));
        }
        let multi = settings.nchains > 1;
        rows.push(hdr("", if multi { "Avg acceptance rate" } else { "Acceptance rate" }, &lead(arate)));
        rows.push(hdr("", if multi { "Avg efficiency: min" } else { "Efficiency:  min" }, &lead(emin)));
        rows.push(hdr("", "             avg", &lead(eavg)));
        let lml_txt = if lml.is_finite() { format!("Log marginal-likelihood = {}", fmt_g(lml, 9)) } else { "Log marginal-likelihood".to_string() };
        rows.push(hdr(&lml_txt, "             max", &lead(emax)));
        if settings.nchains > 1 {
            rows.push(hdr("", "Max Gelman-Rubin Rc", &format!("{rcmax:.4}")));
        }
        out.extend(rows);
        out.push(String::new());
    }
    if !spec.has("notable", 5) {
        out.extend(tbl);
    }
    if settings.nchains == 1 && emin < 0.01 {
        out.push("Note: There is a high autocorrelation after 500 lags.".into());
    }
    if arate < 0.1 && sim.any_mh {
        out.push("Note: Adaptation tolerance is not met.".into());
    }
    // stored results
    let nobs_e = spec.num("_nobs", 5)?.unwrap_or(m.n as f64);
    store_results(st, &m, &sim, &idx, &sums, StoreArgs { rest, level, dic, lml, dbar, arate, emin, eavg, emax, rcmax, rc: &rc, settings: &settings, prefix_cmd: prefix_cmd.as_deref(), title: &ttl });
    st.e.set_str("_output", out.join("\n"));
    st.e.set("N", nobs_e);
    // simulation dataset
    if let Some(Some(sv)) = spec.opt("saving", 3) {
        let parts = text::split_top(sv, ',');
        let file = util::unquote(parts[0].trim());
        let file = if file.contains('.') { file } else { format!("{file}.dta") };
        let replace = parts.iter().skip(1).any(|p| p.trim() == "replace");
        save_sim(&m, &sim, &file, replace)?;
        st.e.set_str("filename", &file);
    }
    Ok(out.join("\n"))
}

pub struct StoreArgs<'a> {
    pub rest: &'a str,
    pub level: f64,
    pub dic: f64,
    pub lml: f64,
    pub dbar: f64,
    pub arate: f64,
    pub emin: f64,
    pub eavg: f64,
    pub emax: f64,
    pub rcmax: f64,
    pub rc: &'a [f64],
    pub settings: &'a mcmc::Settings,
    pub prefix_cmd: Option<&'a str>,
    pub title: &'a str,
}

pub(crate) fn store_results(st: &mut Store, m: &Model, sim: &Sim, idx: &[usize], sums: &[stats::Summary], a: StoreArgs) {
    let names: Vec<String> = idx.iter().map(|&j| m.params[j].name.clone()).collect();
    let k = idx.len();
    let b: Vec<f64> = sums.iter().map(|s| s.mean).collect();
    let v = stats::covariance(&sim.draws, idx);
    let mut cri = Matrix::zeros(2, k);
    for (c, s) in sums.iter().enumerate() {
        cri[(0, c)] = s.lo;
        cri[(1, c)] = s.hi;
    }
    // simulation matrix: all parameters, log likelihood, log posterior, chain
    let np = m.params.len();
    let mut sim_m = Matrix::zeros(sim.draws.len(), np + 3);
    for (r, d) in sim.draws.iter().enumerate() {
        for j in 0..np {
            sim_m[(r, j)] = d[j];
        }
        sim_m[(r, np)] = sim.ll[r];
        sim_m[(r, np + 1)] = sim.ll[r] + sim.lp[r];
        sim_m[(r, np + 2)] = (sim.chain[r] + 1) as f64;
    }
    let mut cn: Vec<String> = m.params.iter().map(|p| p.name.clone()).collect();
    cn.extend(["_loglik".to_string(), "_logpost".to_string(), "_chain".to_string()]);
    let qb = st.matrices.get("__bayes_qb").cloned();
    let qv = st.matrices.get("__bayes_qV").cloned();
    let sigma = st.matrices.get("__bayes_Sigma").cloned();
    let e = st.begin_e();
    let row = vec!["y1".to_string()];
    e.set_mat("b", Matrix::row_vector(&b).with_names(row.clone(), names.clone()));
    e.set_mat("V", v.with_names(names.clone(), names.clone()));
    e.set_mat("mean", Matrix::row_vector(&b).with_names(vec!["mean".into()], names.clone()));
    e.set_mat("sd", Matrix::row_vector(&sums.iter().map(|s| s.sd).collect::<Vec<_>>()).with_names(vec!["sd".into()], names.clone()));
    e.set_mat("mcse", Matrix::row_vector(&sums.iter().map(|s| s.mcse).collect::<Vec<_>>()).with_names(vec!["mcse".into()], names.clone()));
    e.set_mat("median", Matrix::row_vector(&sums.iter().map(|s| s.median).collect::<Vec<_>>()).with_names(vec!["median".into()], names.clone()));
    e.set_mat("cri", cri.with_names(vec!["lower".into(), "upper".into()], names.clone()));
    e.set_mat("ess", Matrix::row_vector(&sums.iter().map(|s| s.ess).collect::<Vec<_>>()).with_names(vec!["ess".into()], names.clone()));
    if !a.rc.is_empty() {
        e.set_mat("Rc", Matrix::row_vector(a.rc).with_names(vec!["Rc".into()], names.clone()));
        e.set("Rc_max", miss(a.rcmax));
    }
    e.set_mat("_mcmc", sim_m.with_names(Vec::new(), cn));
    if let (Some(qb), Some(qv)) = (qb, qv) {
        e.set_mat("_qb", qb);
        e.set_mat("_qV", qv);
    }
    if let Some(sg) = sigma {
        e.set_mat("Sigma", sg);
    }
    e.set("N", m.n as f64);
    e.set("k", np as f64);
    e.set("mcmcsize", a.settings.mcmcsize as f64);
    e.set("burnin", a.settings.burnin as f64);
    e.set("mcmciter", (a.settings.burnin + a.settings.mcmcsize * a.settings.thin) as f64);
    e.set("thinning", a.settings.thin as f64);
    e.set("nchains", a.settings.nchains as f64);
    e.set("arate", miss(a.arate));
    e.set("eff_min", miss(a.emin));
    e.set("eff_avg", miss(a.eavg));
    e.set("eff_max", miss(a.emax));
    e.set("clevel", a.level);
    e.set("dic", miss(a.dic));
    e.set("lml_lm", miss(a.lml));
    e.set("dbar", miss(a.dbar));
    e.set_str("cmd", "bayesmh");
    e.set_str("title", format!("Bayesian {}", a.title));
    e.set_str("parnames", names.join(" "));
    e.set_str("_spec", a.rest.to_string());
    e.set_str("likelihood", m.liks.iter().map(|l| l.fam.label()).collect::<Vec<_>>().join(" "));
    let deps: Vec<String> = m.eqs.iter().map(|e| e.depvar.clone()).filter(|d| !d.is_empty()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    e.set_str("depvar", deps.join(" "));
    e.set_str("properties", "b V");
    e.set_str("predict", "bayespredict");
    e.set_str("sampling", sampling_label(sim).to_string());
    if let Some(p) = a.prefix_cmd {
        e.set_str("prefix", "bayes");
        e.set_str("cmdname", p.to_string());
    }
    e.sample = Some(m.mask.clone());
}

/// Write the simulation results as a dataset: one row per draw with a
/// frequency weight `_frequency`, the chain, the log likelihood and log
/// posterior, and the parameters as `eq_1`, `eq_2`, ...
pub fn save_sim(m: &Model, sim: &Sim, file: &str, replace: bool) -> Result<(), String> {
    if ferrostat_core::vfs::exists(file) && !replace {
        return Err(rc_err(602, format!("file {file} already exists")));
    }
    let mut ds = DataSet::new();
    let n = sim.draws.len();
    ds.add_variable(util::new_double("_chain", sim.chain.iter().map(|c| (*c + 1) as f64).collect())).map_err(|e| rc_err(110, e))?;
    ds.add_variable(util::new_double("_index", (1..=n).map(|i| i as f64).collect())).map_err(|e| rc_err(110, e))?;
    ds.add_variable(util::new_double("_loglikelihood", sim.ll.clone())).map_err(|e| rc_err(110, e))?;
    ds.add_variable(util::new_double("_logposterior", sim.ll.iter().zip(&sim.lp).map(|(a, b)| a + b).collect())).map_err(|e| rc_err(110, e))?;
    ds.add_variable(util::new_double("_frequency", vec![1.0; n])).map_err(|e| rc_err(110, e))?;
    for (j, p) in m.params.iter().enumerate() {
        let mut v = util::new_double(&format!("eq0_p{}", j + 1), sim.draws.iter().map(|d| d[j]).collect());
        v.label = p.name.clone();
        ds.add_variable(v).map_err(|e| rc_err(110, e))?;
    }
    ds.n_obs = n;
    let mut buf = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut buf, &ds).map_err(|e| rc_err(693, e))?;
    ferrostat_core::vfs::write(file, &buf).map_err(|e| rc_err(603, e))
}
