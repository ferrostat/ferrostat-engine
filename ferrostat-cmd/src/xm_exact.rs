//! Exact conditional inference for small samples: `exlogistic` (binary or
//! binomial outcomes) and `expoisson` (counts). The joint distribution of
//! the sufficient statistics is enumerated by dynamic programming over the
//! observations; each parameter is estimated from the conditional
//! distribution of its sufficient statistic given the others (CMLE, or the
//! median unbiased estimate when the statistic lies on the boundary).

use std::collections::HashMap;

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::ln_gamma;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util::{self, WeightKind};
use crate::xmodels as xm;

pub fn handlers() -> xm::Handlers {
    vec![("exlogistic", Box::new(Exact { poisson: false })), ("expoisson", Box::new(Exact { poisson: true }))]
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Conditional distribution of one statistic: support values and log weights.
struct Cond {
    u: Vec<f64>,
    lw: Vec<f64>,
}

impl Cond {
    /// ln sum c(u) exp(b u)
    fn lnorm(&self, b: f64) -> f64 {
        self.u.iter().zip(&self.lw).fold(f64::NEG_INFINITY, |acc, (u, l)| lse(acc, l + b * u))
    }
    /// P_b(T >= t) (ge = true) or P_b(T <= t)
    fn tail(&self, b: f64, t: f64, ge: bool) -> f64 {
        let z = self.lnorm(b);
        let mut s = f64::NEG_INFINITY;
        for (u, l) in self.u.iter().zip(&self.lw) {
            if (ge && *u >= t - 1e-9) || (!ge && *u <= t + 1e-9) {
                s = lse(s, l + b * u);
            }
        }
        (s - z).exp()
    }
    fn mean(&self, b: f64) -> f64 {
        let z = self.lnorm(b);
        self.u.iter().zip(&self.lw).map(|(u, l)| u * (l + b * u - z).exp()).sum()
    }
    fn var(&self, b: f64) -> f64 {
        let z = self.lnorm(b);
        let m = self.mean(b);
        self.u.iter().zip(&self.lw).map(|(u, l)| (u - m).powi(2) * (l + b * u - z).exp()).sum()
    }
    fn prob(&self, b: f64, t: f64) -> f64 {
        let z = self.lnorm(b);
        self.u.iter().zip(&self.lw).filter(|(u, _)| (**u - t).abs() < 1e-9).map(|(u, l)| (l + b * u - z).exp()).sum()
    }
}

/// Solve g(b) = target for increasing g on a wide bracket.
fn bisect(g: &dyn Fn(f64) -> f64, target: f64) -> f64 {
    let (mut lo, mut hi) = (-1.0, 1.0);
    let mut k = 0;
    while g(lo) > target && k < 200 {
        lo *= 2.0;
        k += 1;
    }
    k = 0;
    while g(hi) < target && k < 200 {
        hi *= 2.0;
        k += 1;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) < target {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

pub struct Exact {
    pub poisson: bool,
}

impl CommandHandler for Exact {
    fn name(&self) -> &'static str {
        if self.poisson { "expoisson" } else { "exlogistic" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let cond_text = cmd.opt("condvars", 4).unwrap_or("").to_string();
        let terms_text = format!("{indep} {cond_text}");
        let mut extra = Vec::new();
        let binom = if self.poisson { None } else { cmd.opt("binomial", 3).map(|s| s.trim().to_string()) };
        let expo = if self.poisson { cmd.opt("exposure", 1).map(|s| s.trim().to_string()) } else { None };
        if let Some(b) = &binom {
            if b.parse::<f64>().is_err() {
                extra.push(xm::var_name(ds, b)?);
            }
        }
        if let Some(e) = &expo {
            extra.push(xm::var_name(ds, e)?);
        }
        let d = crate::est::build(ds, st, cmd, crate::est::DesignSpec { depvar: Some(&dep), indep: &terms_text, noconstant: true, weights: "f", extra: extra.clone() })?;
        let act = d.active();
        let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        let ncond = if cond_text.trim().is_empty() { 0 } else { crate::est::build(ds, st, cmd, crate::est::DesignSpec { depvar: None, indep: &cond_text, noconstant: true, weights: "f", extra: vec![] })?.active().len() };
        let nint = names.len() - ncond;
        let p = names.len();
        let n = d.n();
        // trials / exposure per observation
        let trials: Vec<f64> = match &binom {
            None => vec![1.0; n],
            Some(b) => match b.parse::<f64>() {
                Ok(v) => vec![v; n],
                Err(_) => {
                    let c = util::col(ds, &extra[0])?;
                    d.rows.iter().map(|&i| c[i]).collect()
                }
            },
        };
        let lexp: Vec<f64> = match &expo {
            None => vec![0.0; n],
            Some(_) => {
                let c = util::col(ds, extra.last().unwrap())?;
                d.rows.iter().map(|&i| c[i].ln()).collect()
            }
        };
        let fw: Vec<f64> = if d.wkind == Some(WeightKind::F) { d.w.clone() } else { vec![1.0; n] };
        // integer scaling of the covariates
        let mut scale = vec![1.0; p];
        for (a, &j) in act.iter().enumerate() {
            let mut s = 1.0;
            while d.cols[j].iter().any(|x| ((x * s) - (x * s).round()).abs() > 1e-8) && s < 1e6 {
                s *= 10.0;
            }
            scale[a] = s;
        }
        let xi: Vec<Vec<i64>> = (0..n).map(|t| (0..p).map(|a| (d.cols[act[a]][t] * scale[a]).round() as i64).collect()).collect();
        let y: Vec<f64> = d.y.iter().zip(&fw).map(|(v, w)| if self.poisson { v * w } else if binom.is_some() { v * w } else if *v != 0.0 { *w } else { 0.0 }).collect();
        let m_tot: f64 = y.iter().sum();
        let mtot = m_tot.round() as i64;
        // observed sufficient statistics
        let tobs: Vec<i64> = (0..p).map(|a| (0..n).map(|t| (y[t] as i64) * xi[t][a]).sum()).collect();
        // dynamic programming over observations: key (total, T_1..T_p)
        let mut dist: HashMap<Vec<i64>, f64> = HashMap::new();
        dist.insert(vec![0; p + 1], 0.0);
        let mut log = Vec::new();
        for t in 0..n {
            let mtrials = if self.poisson { mtot } else { (trials[t] * fw[t]).round() as i64 };
            let mut next: HashMap<Vec<i64>, f64> = HashMap::with_capacity(dist.len() * 2);
            for (key, lw) in &dist {
                let room = mtot - key[0];
                for k in 0..=mtrials.min(room) {
                    let lc = if self.poisson {
                        k as f64 * lexp[t] - ln_gamma(k as f64 + 1.0)
                    } else {
                        ln_gamma(mtrials as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((mtrials - k) as f64 + 1.0)
                    };
                    let mut nk = key.clone();
                    nk[0] += k;
                    for a in 0..p {
                        nk[a + 1] += k * xi[t][a];
                    }
                    let e = next.entry(nk).or_insert(f64::NEG_INFINITY);
                    *e = lse(*e, lw + lc);
                }
            }
            dist = next;
            log.push(format!("Observation {}:  Enumerations = {}", t + 1, dist.len()));
            if dist.len() > 5_000_000 {
                return Err(rc_err(910, "exact distribution too large"));
            }
        }
        dist.retain(|k, _| k[0] == mtot);
        let level = util::level(cmd, st)?;
        let alpha = (100.0 - level) / 200.0;
        // per-term inference
        let mut rows = Vec::new();
        let mut bvec = Vec::new();
        let mut notes = Vec::new();
        for a in 0..nint {
            let mut cm: HashMap<i64, f64> = HashMap::new();
            for (key, lw) in &dist {
                if (0..p).all(|c| c == a || key[c + 1] == tobs[c]) {
                    let e = cm.entry(key[a + 1]).or_insert(f64::NEG_INFINITY);
                    *e = lse(*e, *lw);
                }
            }
            let mut pairs: Vec<(i64, f64)> = cm.into_iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let cd = Cond { u: pairs.iter().map(|p| p.0 as f64).collect(), lw: pairs.iter().map(|p| p.1).collect() };
            let t = tobs[a] as f64;
            let umin = cd.u.first().copied().unwrap_or(t);
            let umax = cd.u.last().copied().unwrap_or(t);
            let s = scale[a];
            let (b, mue) = if cd.u.len() < 2 {
                (SYSMISS, false)
            } else if t <= umin {
                notes.push(format!("note: CMLE estimate for {} is -inf; computing MUE.", names[a]));
                (bisect(&|b| -cd.tail(b, t, false), -0.5), true)
            } else if t >= umax {
                notes.push(format!("note: CMLE estimate for {} is +inf; computing MUE.", names[a]));
                (bisect(&|b| cd.tail(b, t, true), 0.5), true)
            } else {
                (bisect(&|b| cd.mean(b), t), false)
            };
            let lo = if t <= umin { f64::NEG_INFINITY } else { bisect(&|b| cd.tail(b, t, true), alpha) };
            let hi = if t >= umax { f64::INFINITY } else { bisect(&|b| -cd.tail(b, t, false), -alpha) };
            let p_suff = (2.0 * cd.tail(0.0, t, true).min(cd.tail(0.0, t, false))).min(1.0);
            let mu0 = cd.mean(0.0);
            let v0 = cd.var(0.0);
            let score = if v0 > 0.0 { (t - mu0).powi(2) / v0 } else { 0.0 };
            let p_score: f64 = cd.u.iter().zip(&cd.lw).filter(|(u, _)| v0 > 0.0 && (**u - mu0).powi(2) / v0 >= score - 1e-9).map(|(_, l)| (l - cd.lnorm(0.0)).exp()).sum();
            let pobs = cd.prob(0.0, t);
            let p_prob: f64 = cd.u.iter().map(|u| cd.prob(0.0, *u)).filter(|q| *q <= pobs + 1e-12).sum();
            let bb = if b == SYSMISS { SYSMISS } else { b * s };
            bvec.push(if bb == SYSMISS { 0.0 } else { bb });
            rows.push((names[a].clone(), bb, mue, tobs[a] as f64 / s, p_suff, score, p_score, pobs, p_prob, lo * s, hi * s));
        }
        // model score test: joint conditional distribution of the terms of interest
        let mut joint: HashMap<Vec<i64>, f64> = HashMap::new();
        for (key, lw) in &dist {
            if (nint..p).all(|c| key[c + 1] == tobs[c]) {
                let e = joint.entry(key[1..=nint].to_vec()).or_insert(f64::NEG_INFINITY);
                *e = lse(*e, *lw);
            }
        }
        let lz = joint.values().fold(f64::NEG_INFINITY, |a, b| lse(a, *b));
        let probs: Vec<(Vec<f64>, f64)> = joint.iter().map(|(k, l)| (k.iter().map(|v| *v as f64).collect(), (l - lz).exp())).collect();
        let mu: Vec<f64> = (0..nint).map(|a| probs.iter().map(|(k, q)| k[a] * q).sum()).collect();
        let mut sig = Matrix::zeros(nint, nint);
        for a in 0..nint {
            for c in 0..nint {
                sig[(a, c)] = probs.iter().map(|(k, q)| (k[a] - mu[a]) * (k[c] - mu[c]) * q).sum();
            }
        }
        let (si, _) = sig.sym_inverse(1e-12);
        let sc = |k: &[f64]| -> f64 { (0..nint).map(|a| (0..nint).map(|c| (k[a] - mu[a]) * si[(a, c)] * (k[c] - mu[c])).sum::<f64>()).sum() };
        let tob: Vec<f64> = tobs[..nint].iter().map(|v| *v as f64).collect();
        let mscore = sc(&tob);
        let mp: f64 = probs.iter().filter(|(k, _)| sc(k) >= mscore - 1e-7 * (1.0 + mscore)).map(|(_, q)| q).sum();
        let pobs_m = probs.iter().filter(|(k, _)| k.iter().zip(&tob).all(|(a, b)| (a - b).abs() < 1e-9)).map(|(_, q)| q).sum::<f64>();
        let mprob: f64 = probs.iter().filter(|(_, q)| *q <= pobs_m + 1e-12).map(|(_, q)| q).sum();
        let test = cmd.opt("test", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "sufficient".into());
        let coef = cmd.has_opt("coef", 4);
        let title = if self.poisson { "Exact Poisson regression" } else { "Exact logistic regression" };
        let mut out = Vec::new();
        if !cmd.has_opt("nolog", 3) {
            out.push("Enumerating sample-space combinations:".into());
            out.extend(log.iter().cloned());
            out.push(String::new());
        }
        out.extend(notes.iter().cloned());
        out.push(format!("{:<49}{:<13} = {:>10}", title, "Number of obs", util::commas(d.n_eff() as usize)));
        if test.starts_with('p') {
            out.push(format!("{:<49}{:<13} = {:>10}", binom.as_ref().map(|b| format!("Binomial variable: {b}")).unwrap_or_default(), "Model prob.", format!("{:.4}", pobs_m)));
            out.push(format!("{:<49}{:<13} = {:>10}", "", "Pr <= prob.", format!("{:.4}", mprob)));
        } else {
            out.push(format!("{:<49}{:<13} = {:>10}", binom.as_ref().map(|b| format!("Binomial variable: {b}")).unwrap_or_default(), "Model score", fmt_g(mscore, 8)));
            out.push(format!("{:<49}{:<13} = {:>10}", "", "Pr >= score", format!("{:.4}", mp)));
        }
        out.push(String::new());
        let lw = 12;
        let est_title = if coef { "Coefficient" } else if self.poisson { "IRR" } else { "Odds ratio" };
        let (c3, c4) = match test.chars().next() {
            Some('s') if test.starts_with("sc") => ("Score", "Pr>=Score"),
            Some('p') => ("Prob.", "Pr<=Prob."),
            _ => ("Suff.", "2*Pr(Suff.)"),
        };
        out.push("-".repeat(lw + 66));
        out.push(format!("{:>lw$} | {:>11}  {:>10}  {:>11}     [{}% conf. interval]", crate::summarize::abbrev(&dep, lw), est_title, c3, c4, fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let fmt_b = |v: f64| -> String {
            if v == f64::INFINITY {
                "+inf".into()
            } else if v == f64::NEG_INFINITY {
                "-inf".into()
            } else {
                fmt_g(v, 8)
            }
        };
        for r in &rows {
            let tr = |v: f64| if coef || v == SYSMISS { v } else { v.exp() };
            let (e3, e4) = match c3 {
                "Score" => (fmt_g(r.5, 8), format!("{:.4}", r.6)),
                "Prob." => (format!("{:.4}", r.7), format!("{:.4}", r.8)),
                _ => (fmt_g(r.3, 8), format!("{:.4}", r.4)),
            };
            let (lo, hi) = if coef { (r.9, r.10) } else { (r.9.exp(), r.10.exp()) };
            out.push(format!(
                "{:>lw$} | {:>10}{} {:>10}  {:>11}     {:>9}  {:>9}",
                crate::summarize::abbrev(&r.0, lw),
                fmt_b(tr(r.1)),
                if r.2 { "*" } else { " " },
                e3,
                e4,
                fmt_b(lo),
                fmt_b(hi)
            ));
        }
        out.push("-".repeat(lw + 66));
        if rows.iter().any(|r| r.2) {
            out.push("(*) median unbiased estimates (MUE)".into());
        }
        let text = out.join("\n");
        let bn: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
        let k = bn.len();
        let ci = Matrix::from_rows(&[rows.iter().map(|r| r.9).collect::<Vec<_>>(), rows.iter().map(|r| r.10).collect()]);
        let mut sample = vec![false; ds.n_obs];
        for &r in &d.rows {
            sample[r] = true;
        }
        let verb = self.name();
        let e = st.begin_e();
        e.set("N", d.n_eff());
        e.set("k_groups", 1.0);
        e.set("sum_y", m_tot);
        e.set("score", mscore);
        e.set("p_score", mp);
        e.set_str("cmd", verb);
        e.set_str("depvar", dep.clone());
        e.set_str("title", title);
        e.set_str("properties", "b");
        e.set_mat("b", Matrix::row_vector(&bvec).with_names(vec!["y1".into()], bn.clone()));
        e.set_mat("ci", ci.with_names(vec!["ll".into(), "ul".into()], bn.clone()));
        e.set_mat("sufficient", Matrix::row_vector(&rows.iter().map(|r| r.3).collect::<Vec<_>>()).with_names(vec!["y1".into()], bn.clone()));
        e.set_mat("p_sufficient", Matrix::row_vector(&rows.iter().map(|r| r.4).collect::<Vec<_>>()).with_names(vec!["y1".into()], bn.clone()));
        e.set_mat("scoretest", Matrix::row_vector(&rows.iter().map(|r| r.5).collect::<Vec<_>>()).with_names(vec!["y1".into()], bn.clone()));
        e.set_mat("p_scoretest", Matrix::row_vector(&rows.iter().map(|r| r.6).collect::<Vec<_>>()).with_names(vec!["y1".into()], bn));
        e.sample = Some(sample);
        e.set_str("_output", text.clone());
        let _ = k;
        Ok(text)
    }
}
