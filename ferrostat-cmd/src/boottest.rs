//! `boottest`: wild (cluster) bootstrap tests of linear hypotheses after
//! `regress`, after Roodman, MacKinnon, Nielsen & Webb (2019, Stata
//! Journal 19:4-60), Cameron, Gelbach & Miller (2008) and MacKinnon & Webb
//! (2017). The null is imposed by default (WCR); bootstrap-t / Wald
//! statistics use the same cluster-robust variance as the estimation.
//! Draws come from the session random-number generator, so results are
//! reproducible after `set seed`. For one restriction the statistic is
//! affine/quadratic in the hypothesized value, which gives the confidence
//! set by test inversion at little cost.
//!
//!   boottest hyp [hyp ...] [, reps(#) weighttype(rademacher|mammen|webb|normal)
//!       cluster(var) nonull noci level(#) seed(#)]
//!
//! where hyp is `x`, `x = 1`, `2*x - z = 0`, or several in `{ }`.

use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::est::{self, DesignSpec};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::teffects::{dot, invert};
use crate::util;

#[derive(Clone, Copy, PartialEq)]
enum WType {
    Rademacher,
    Mammen,
    Webb,
    Normal,
}

/// Parse one linear restriction `a*x + b*z = c` over coefficient names.
fn parse_hyp(text: &str, names: &[String]) -> Result<(Vec<f64>, f64), String> {
    let (lhs, rhs) = match text.split_once('=') {
        Some((l, r)) => (l.trim(), r.trim().parse::<f64>().map_err(|_| rc_err(198, format!("invalid hypothesis {text}")))?),
        None => (text.trim(), 0.0),
    };
    let mut r = vec![0.0; names.len()];
    let mut s = lhs.replace(' ', "");
    if !s.starts_with('-') && !s.starts_with('+') {
        s.insert(0, '+');
    }
    let mut terms = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if (ch == '+' || ch == '-') && !cur.is_empty() && !cur.ends_with('*') && !cur.ends_with('e') {
            terms.push(std::mem::take(&mut cur));
        }
        cur.push(ch);
    }
    terms.push(cur);
    for t in terms {
        let sign = if t.starts_with('-') { -1.0 } else { 1.0 };
        let body = &t[1..];
        let (coef, var) = match body.split_once('*') {
            Some((a, b)) => match a.parse::<f64>() {
                Ok(c) => (c, b.to_string()),
                Err(_) => (b.parse::<f64>().map_err(|_| rc_err(198, format!("invalid term {body}")))?, a.to_string()),
            },
            None => (1.0, body.to_string()),
        };
        let var = var.trim_start_matches("_b[").trim_end_matches(']').to_string();
        let k = names.iter().position(|n| *n == var).ok_or_else(|| rc_err(111, format!("{var} not found among the coefficients")))?;
        r[k] += sign * coef;
    }
    Ok((r, rhs))
}

pub struct Boottest;

impl CommandHandler for Boottest {
    fn name(&self) -> &'static str {
        "boottest"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        if ecmd != "regress" {
            return Err(rc_err(301, if ecmd.is_empty() { "last estimates not found".to_string() } else { format!("boottest is implemented after regress, not after {ecmd}") }));
        }
        if st.e.macros.get("wtype").is_some_and(|w| !w.is_empty()) {
            return Err(rc_err(198, "boottest after weighted regressions is not implemented"));
        }
        let cmdline = st.e.macros.get("cmdline").cloned().ok_or_else(|| rc_err(301, "estimation command line not found"))?;
        let ec = parse_line(&cmdline).map_err(|e| rc_err(198, e.message))?;
        let main = ec.main.trim().to_string();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main.as_str(), ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let clustvar = match cmd.opt("cluster", 2) {
            Some(c) => Some(crate::varlist::resolve_name(ds, c.trim())?),
            None => st.e.macros.get("clustvar").cloned().filter(|c| !c.is_empty()),
        };
        let evce = st.e.macros.get("vce").cloned().unwrap_or_default();
        let robust = clustvar.is_some() || evce == "robust" || evce.starts_with("hc");
        let mut d = est::build(ds, st, &ec, DesignSpec { depvar: Some(&dep), indep, noconstant: ec.has_opt("noconstant", 3), weights: "", extra: clustvar.iter().cloned().collect() })?;
        est::drop_collinear(&mut d);
        let act = d.active();
        let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        let n = d.rows.len();
        let k = act.len();
        let x: Vec<Vec<f64>> = (0..n).map(|i| act.iter().map(|&j| d.cols[j][i]).collect()).collect();
        let y = d.y.clone();
        // hypotheses
        let text = cmd.main.trim();
        if text.is_empty() {
            return Err(rc_err(198, "hypothesis required"));
        }
        // several hypotheses in (...) or {...} groups
        let hyps: Vec<String> = if text.starts_with('(') || text.starts_with('{') {
            let mut v = Vec::new();
            let mut depth = 0;
            let mut cur = String::new();
            for ch in text.chars() {
                match ch {
                    '(' | '{' => {
                        if depth > 0 {
                            cur.push(ch);
                        }
                        depth += 1;
                    }
                    ')' | '}' => {
                        depth -= 1;
                        if depth == 0 {
                            v.push(std::mem::take(&mut cur).trim().to_string());
                        } else {
                            cur.push(ch);
                        }
                    }
                    _ if depth > 0 => cur.push(ch),
                    _ => {}
                }
            }
            v.into_iter().filter(|s| !s.is_empty()).collect()
        } else {
            vec![text.to_string()]
        };
        let mut rmat = Vec::new();
        let mut rvec = Vec::new();
        for h in &hyps {
            let (r, c) = parse_hyp(h, &names)?;
            rmat.push(r);
            rvec.push(c);
        }
        let q = rmat.len();
        let reps = util::opt_num(cmd, "reps", 3)?.unwrap_or(999.0).max(1.0) as usize;
        let wtype = match cmd.opt("weighttype", 2).map(|s| s.trim().to_lowercase()).as_deref() {
            None | Some("rademacher") => WType::Rademacher,
            Some("mammen") => WType::Mammen,
            Some("webb") => WType::Webb,
            Some("normal") => WType::Normal,
            Some(o) => return Err(rc_err(198, format!("weighttype({o}) not allowed"))),
        };
        if let Some(s) = util::opt_num(cmd, "seed", 4)? {
            st.rng.borrow_mut().reseed(s as u64);
        }
        let nonull = cmd.has_opt("nonull", 5);
        let level = util::level(cmd, st)?;
        // clusters (each observation is its own cluster when not clustered)
        let (gid, g) = match &clustvar {
            Some(c) => {
                let col = util::col(ds, c)?;
                let mut map = std::collections::BTreeMap::new();
                let ids: Vec<usize> = d.rows.iter().map(|&r| {
                    let next = map.len();
                    *map.entry(col[r].to_bits()).or_insert(next)
                }).collect();
                let g = map.len();
                (ids, g)
            }
            None => ((0..n).collect(), n),
        };
        if g < 2 {
            return Err(rc_err(459, "at least two clusters are required"));
        }
        let gf = g as f64;
        let (nf, kf) = (n as f64, k as f64);
        let scale = if clustvar_is(&clustvar) { gf / (gf - 1.0) * (nf - 1.0) / (nf - kf) } else { nf / (nf - kf) };
        // OLS pieces
        let mut xx = vec![vec![0.0; k]; k];
        for r in &x {
            for a in 0..k {
                for b in 0..k {
                    xx[a][b] += r[a] * r[b];
                }
            }
        }
        let ainv = invert(&xx).ok_or_else(|| rc_err(459, "X'X is singular"))?;
        let xy: Vec<f64> = (0..k).map(|a| (0..n).map(|i| x[i][a] * y[i]).sum()).collect();
        let bhat: Vec<f64> = (0..k).map(|a| dot(&ainv[a], &xy)).collect();
        let uhat: Vec<f64> = (0..n).map(|i| y[i] - dot(&x[i], &bhat)).collect();
        // a_{h,i} = R_h A x_i
        let ra: Vec<Vec<f64>> = rmat.iter().map(|r| (0..k).map(|b| (0..k).map(|a| r[a] * ainv[a][b]).sum()).collect()).collect();
        let a_i: Vec<Vec<f64>> = (0..q).map(|h| (0..n).map(|i| dot(&ra[h], &x[i])).collect()).collect();
        let rar: Vec<Vec<f64>> = (0..q).map(|h| (0..q).map(|l| dot(&ra[h], &rmat[l])).collect()).collect();
        let hat = |v: &[f64]| -> Vec<f64> {
            let xv: Vec<f64> = (0..k).map(|a| (0..n).map(|i| x[i][a] * v[i]).sum()).collect();
            let b: Vec<f64> = (0..k).map(|a| dot(&ainv[a], &xv)).collect();
            (0..n).map(|i| dot(&x[i], &b)).collect()
        };
        // variance of R b from residuals u (q x q)
        let vr = |u: &[f64]| -> Vec<Vec<f64>> {
            let mut m = vec![vec![0.0; q]; q];
            if robust {
                let mut s = vec![vec![0.0; q]; g];
                for i in 0..n {
                    for h in 0..q {
                        s[gid[i]][h] += a_i[h][i] * u[i];
                    }
                }
                for sg in &s {
                    for h in 0..q {
                        for l in 0..q {
                            m[h][l] += scale * sg[h] * sg[l];
                        }
                    }
                }
            } else {
                let s2 = u.iter().map(|v| v * v).sum::<f64>() / (nf - kf);
                for h in 0..q {
                    for l in 0..q {
                        m[h][l] = s2 * rar[h][l];
                    }
                }
            }
            m
        };
        let wald = |num: &[f64], v: &[Vec<f64>]| -> f64 {
            if q == 1 {
                return num[0] / v[0][0].sqrt();
            }
            let vi = invert(v).unwrap_or_else(|| vec![vec![f64::NAN; q]; q]);
            (0..q).map(|h| (0..q).map(|l| num[h] * vi[h][l] * num[l]).sum::<f64>()).sum::<f64>() / q as f64
        };
        let num_hat: Vec<f64> = (0..q).map(|h| dot(&rmat[h], &bhat) - rvec[h]).collect();
        let stat = wald(&num_hat, &vr(&uhat));
        // restricted fit for a given right-hand side
        let restricted = |rv: &[f64]| -> Option<Vec<f64>> {
            let rari = invert(&rar)?;
            let dev: Vec<f64> = (0..q).map(|h| dot(&rmat[h], &bhat) - rv[h]).collect();
            let lam: Vec<f64> = (0..q).map(|h| dot(&rari[h], &dev)).collect();
            let bt: Vec<f64> = (0..k).map(|a| bhat[a] - (0..q).map(|h| ra[h][a] * lam[h]).sum::<f64>()).collect();
            Some((0..n).map(|i| y[i] - dot(&x[i], &bt)).collect())
        };
        // weights: all 2^G sign vectors when that is fewer than reps
        let enumerate = wtype == WType::Rademacher && g < 30 && (1usize << g) <= reps;
        let b_reps = if enumerate { 1usize << g } else { reps };
        let draw = |st: &Store, b: usize| -> Vec<f64> {
            if enumerate {
                return (0..g).map(|c| if (b >> c) & 1 == 1 { 1.0 } else { -1.0 }).collect();
            }
            let mut rng = st.rng.borrow_mut();
            (0..g)
                .map(|_| match wtype {
                    WType::Rademacher => if rng.uniform() < 0.5 { -1.0 } else { 1.0 },
                    WType::Mammen => {
                        let s5 = 5f64.sqrt();
                        if rng.uniform() < (s5 + 1.0) / (2.0 * s5) { (1.0 - s5) / 2.0 } else { (1.0 + s5) / 2.0 }
                    }
                    WType::Webb => {
                        let u = rng.uniform() * 6.0;
                        let vals = [-(1.5f64).sqrt(), -1.0, -(0.5f64).sqrt(), (0.5f64).sqrt(), 1.0, (1.5f64).sqrt()];
                        vals[(u as usize).min(5)]
                    }
                    WType::Normal => rng.normal(),
                })
                .collect()
        };
        let draws: Vec<Vec<f64>> = (0..b_reps).map(|b| draw(st, b)).collect();
        let u0 = if nonull { uhat.clone() } else { restricted(&rvec).ok_or_else(|| rc_err(459, "restrictions are collinear"))? };
        // bootstrap statistics at the tested value
        let mut stars = Vec::with_capacity(b_reps);
        for v in &draws {
            let vu: Vec<f64> = (0..n).map(|i| v[gid[i]] * u0[i]).collect();
            let hv = hat(&vu);
            let ustar: Vec<f64> = (0..n).map(|i| vu[i] - hv[i]).collect();
            let num: Vec<f64> = (0..q).map(|h| (0..n).map(|i| a_i[h][i] * vu[i]).sum()).collect();
            stars.push(wald(&num, &vr(&ustar)));
        }
        let pval = |s: f64, st_: &[f64]| -> f64 {
            if q == 1 {
                st_.iter().filter(|t| t.abs() >= s.abs()).count() as f64 / st_.len() as f64
            } else {
                st_.iter().filter(|t| **t >= s).count() as f64 / st_.len() as f64
            }
        };
        let p = pval(stat, &stars);
        let mut out = Vec::new();
        let wname = match wtype {
            WType::Rademacher => "Rademacher",
            WType::Mammen => "Mammen",
            WType::Webb => "Webb",
            WType::Normal => "normal",
        };
        out.push(format!(
            "Wild bootstrap-t, null {}imposed, {} replications{}, {} weights:",
            if nonull { "not " } else { "" },
            b_reps,
            if enumerate { " (full enumeration)" } else { "" },
            wname
        ));
        for h in &hyps {
            out.push(format!("  {h}{}", if h.contains('=') { "" } else { " = 0" }));
        }
        out.push(String::new());
        let df_r = if clustvar_is(&clustvar) { gf - 1.0 } else { nf - kf };
        if q == 1 {
            out.push(format!("{:>34} = {:>10.4}", format!("t({})", fmt_g(df_r, 9).trim()), stat));
            out.push(format!("{:>34} = {:>10.4}", "Prob>|t|", p));
        } else {
            out.push(format!("{:>34} = {:>10.4}", format!("F({}, {})", q, fmt_g(df_r, 9).trim()), stat));
            out.push(format!("{:>34} = {:>10.4}", "Prob > F", p));
        }
        // confidence set by test inversion (one restriction, null imposed)
        let mut ci: Option<(f64, f64)> = None;
        if q == 1 && !nonull && !cmd.has_opt("noci", 4) {
            // pieces affine in the hypothesized value r: u~(r) = u0 + (r - r0) e
            let rari = 1.0 / rar[0][0];
            let delta: Vec<f64> = (0..k).map(|a| ra[0][a] * rari).collect();
            let xd: Vec<f64> = (0..n).map(|i| dot(&x[i], &delta)).collect();
            // u~(r) = y - X b~(r), b~(r) = bhat - delta (R bhat - r): du/dr = -X delta
            let e1: Vec<f64> = xd.iter().map(|v| -v).collect();
            let r0 = rvec[0];
            let rb = dot(&rmat[0], &bhat);
            let vhat = vr(&uhat)[0][0];
            let mut coefs = Vec::with_capacity(b_reps);
            for v in &draws {
                let va: Vec<f64> = (0..n).map(|i| v[gid[i]] * u0[i]).collect();
                let vb: Vec<f64> = (0..n).map(|i| v[gid[i]] * e1[i]).collect();
                let ha = hat(&va);
                let hb = hat(&vb);
                let n0: f64 = (0..n).map(|i| a_i[0][i] * va[i]).sum();
                let n1: f64 = (0..n).map(|i| a_i[0][i] * vb[i]).sum();
                let (mut q00, mut q01, mut q11) = (0.0, 0.0, 0.0);
                if robust {
                    let mut s0 = vec![0.0; g];
                    let mut s1 = vec![0.0; g];
                    for i in 0..n {
                        s0[gid[i]] += a_i[0][i] * (va[i] - ha[i]);
                        s1[gid[i]] += a_i[0][i] * (vb[i] - hb[i]);
                    }
                    for c in 0..g {
                        q00 += scale * s0[c] * s0[c];
                        q01 += scale * s0[c] * s1[c];
                        q11 += scale * s1[c] * s1[c];
                    }
                } else {
                    let f = rar[0][0] / (nf - kf);
                    for i in 0..n {
                        let (a0, a1) = (va[i] - ha[i], vb[i] - hb[i]);
                        q00 += f * a0 * a0;
                        q01 += f * a0 * a1;
                        q11 += f * a1 * a1;
                    }
                }
                coefs.push((n0, n1, q00, q01, q11));
            }
            let p_at = |r: f64| -> f64 {
                let dr = r - r0;
                let t = (rb - r) / vhat.sqrt();
                let cnt = coefs.iter().filter(|c| {
                    let num = c.0 + dr * c.1;
                    let den = c.2 + 2.0 * dr * c.3 + dr * dr * c.4;
                    (num / den.max(1e-300).sqrt()).abs() >= t.abs()
                }).count();
                cnt as f64 / coefs.len() as f64
            };
            let alpha = 1.0 - level / 100.0;
            let se = vhat.sqrt();
            let find = |dir: f64| -> Option<f64> {
                let mut inside = rb;
                let mut step = se;
                let mut outside = None;
                for _ in 0..60 {
                    let cand = rb + dir * step;
                    if p_at(cand) < alpha {
                        outside = Some(cand);
                        break;
                    }
                    inside = cand;
                    step *= 1.5;
                }
                let mut o = outside?;
                for _ in 0..60 {
                    let mid = 0.5 * (inside + o);
                    if p_at(mid) < alpha { o = mid } else { inside = mid }
                }
                Some(0.5 * (inside + o))
            };
            if let (Some(lo), Some(hi)) = (find(-1.0), find(1.0)) {
                ci = Some((lo, hi));
                out.push(String::new());
                out.push(format!("{}% confidence set for null hypothesis expression: [{}, {}]", fmt_g(level, 4).trim(), fmt_g(lo, 7).trim(), fmt_g(hi, 7).trim()));
            }
        }
        let r = st.begin_r();
        r.set("p", p);
        r.set("reps", b_reps as f64);
        r.set("df", q as f64);
        r.set("df_r", df_r);
        r.set("level", level);
        if q == 1 {
            r.set("t", stat);
        } else {
            r.set("F", stat);
        }
        if let Some((lo, hi)) = ci {
            let mut m = Matrix::zeros(1, 2);
            m[(0, 0)] = lo;
            m[(0, 1)] = hi;
            r.set_mat("CI", m.with_names(vec!["r1".into()], vec!["lo".into(), "hi".into()]));
        }
        let mut dm = Matrix::zeros(stars.len(), 1);
        for (i, s) in stars.iter().enumerate() {
            dm[(i, 0)] = *s;
        }
        r.set_mat("dist", dm);
        r.set_str("weighttype", wname.to_lowercase());
        Ok(out.join("\n"))
    }
}

fn clustvar_is(c: &Option<String>) -> bool {
    c.is_some()
}
