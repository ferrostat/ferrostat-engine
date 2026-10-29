//! Structural-break tests after `regress` on time-series data:
//! `estat sbknown` (Wald test at known break dates), `estat sbsingle`
//! (supremum, average and exponential Wald tests for an unknown break date,
//! p-values from the limiting Brownian-bridge distribution by simulation)
//! and `estat sbcusum` (CUSUM of recursive or OLS residuals).

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::ts_ml;
use crate::tsops;
use crate::util;

struct Reg {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    names: Vec<String>,
    times: Vec<f64>,
    tvar: String,
}

fn regdata(ds: &DataSet, st: &Store) -> Result<Reg, String> {
    let e = &st.e;
    if e.macros.get("cmd").map(String::as_str) != Some("regress") {
        return Err(rc_err(321, "requested action not valid after this estimation command; run regress first"));
    }
    let tvar = tsops::time_var(st)?;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let names: Vec<String> = b.colnames.iter().map(|c| c.rsplit(':').next().unwrap_or(c).to_string()).filter(|c| !c.starts_with("o.") && !c.contains("b.")).collect();
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let yv = tsops::values(ds, st, &dep)?;
    let xv: Vec<Vec<f64>> = names.iter().map(|n| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) }).collect::<Result<_, _>>()?;
    let sample = e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let tc = util::col(ds, &tvar)?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false) && !is_missing(yv[i]) && xv.iter().all(|c| !is_missing(c[i]))).collect();
    rows.sort_by(|&a, &b| tc[a].total_cmp(&tc[b]));
    Ok(Reg { y: rows.iter().map(|&i| yv[i]).collect(), x: rows.iter().map(|&i| xv.iter().map(|c| c[i]).collect()).collect(), names, times: rows.iter().map(|&i| tc[i]).collect(), tvar })
}

fn ols(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, Matrix, f64)> {
    let k = x.first()?.len();
    let mut m = Matrix::zeros(k, k);
    let mut xy = vec![0.0; k];
    for (r, yv) in x.iter().zip(y) {
        for a in 0..k {
            xy[a] += r[a] * yv;
            for c in 0..k {
                m[(a, c)] += r[a] * r[c];
            }
        }
    }
    let (inv, dropped) = m.sym_inverse(1e-12);
    if !dropped.is_empty() {
        return None;
    }
    let b = ts_ml::mv(&inv, &xy);
    let ssr = x.iter().zip(y).map(|(r, yv)| (yv - ts_ml::dot(r, &b)).powi(2)).sum();
    Some((b, inv, ssr))
}

/// Wald statistic for breaks at positions `brk` (first index of each new regime) in the coefficients `test`.
fn wald_break(r: &Reg, brk: &[usize], test: &[usize]) -> Option<f64> {
    let n = r.y.len();
    let k = r.x[0].len();
    let q = test.len();
    let x: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            let mut row = r.x[t].clone();
            for &bp in brk {
                for &j in test {
                    row.push(if t >= bp { r.x[t][j] } else { 0.0 });
                }
            }
            row
        })
        .collect();
    let (b, inv, ssr) = ols(&x, &r.y)?;
    let kk = k + q * brk.len();
    let s2 = ssr / (n - kk) as f64;
    let idx: Vec<usize> = (k..kk).collect();
    let v = inv.scale(s2);
    Some(ts_ml::wald(&b, &v, &idx).0)
}

fn test_cols(r: &Reg, cmd: &Command, ds: &DataSet) -> Result<Vec<usize>, String> {
    match cmd.opt("breakvars", 6) {
        None => Ok((0..r.names.len()).collect()),
        Some(s) => {
            let (vars, opts) = s.split_once(',').unwrap_or((s, ""));
            let mut out = Vec::new();
            for v in tsops::expand_text(vars).split_whitespace() {
                let v = tsops::resolve(ds, v)?;
                out.push(r.names.iter().position(|n| *n == v).ok_or_else(|| rc_err(111, format!("{v} not in the model")))?);
            }
            if opts.contains("cons") {
                if let Some(j) = r.names.iter().position(|n| n == "_cons") {
                    out.push(j);
                }
            }
            Ok(out)
        }
    }
}

fn footer(r: &Reg, test: &[usize]) -> Vec<String> {
    let exo: Vec<&str> = r.names.iter().filter(|n| *n != "_cons").map(String::as_str).collect();
    vec![format!("Exogenous variables:            {}", exo.join(" ")), format!("Coefficients included in test:  {}", test.iter().map(|&j| r.names[j].as_str()).collect::<Vec<_>>().join(" "))]
}

/// Simple deterministic generator for the simulated null distributions.
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Null distributions of the sup, average and exp Wald statistics (q restrictions, trimming pi).
fn simulate(q: usize, pi: f64) -> [Vec<f64>; 3] {
    let steps = 1000usize;
    let reps = 2000usize;
    let mut g = Xs(0x9E37_79B9_7F4A_7C15);
    let lo = ((pi * steps as f64).ceil() as usize).max(1);
    let hi = (((1.0 - pi) * steps as f64).floor() as usize).min(steps - 1);
    let mut out = [Vec::with_capacity(reps), Vec::with_capacity(reps), Vec::with_capacity(reps)];
    for _ in 0..reps {
        let mut paths = vec![vec![0.0; steps + 1]; q];
        for p in paths.iter_mut() {
            for s in 1..=steps {
                p[s] = p[s - 1] + g.normal() / (steps as f64).sqrt();
            }
        }
        let (mut sup, mut avg, mut ex) = (0.0f64, 0.0, 0.0);
        let mut cnt = 0.0;
        for s in lo..=hi {
            let r = s as f64 / steps as f64;
            let qv: f64 = paths.iter().map(|p| (p[s] - r * p[steps]).powi(2)).sum::<f64>() / (r * (1.0 - r));
            sup = sup.max(qv);
            avg += qv;
            ex += (qv / 2.0).exp();
            cnt += 1.0;
        }
        out[0].push(sup);
        out[1].push(avg / cnt);
        out[2].push((ex / cnt).ln());
    }
    out
}

fn sim_p(dist: &[f64], stat: f64) -> f64 {
    dist.iter().filter(|&&v| v >= stat).count() as f64 / dist.len() as f64
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    match sub {
        "sbknown" | "sbsingle" | "sbcusum" => {}
        _ => return None,
    }
    Some(run(ds, st, sub, cmd))
}

fn run(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let r = regdata(ds, st)?;
    let n = r.y.len();
    let k = r.names.len();
    let lab = |t: f64| tsops::time_label(ds, &r.tvar, t);
    let mut out = vec![String::new()];
    let mut res = Results::new();
    match sub {
        "sbknown" => {
            let spec = cmd.opt("break", 3).ok_or_else(|| rc_err(198, "option break() required"))?;
            let mut brk = Vec::new();
            let mut dates = Vec::new();
            for d in spec.split(|c: char| c == ',' || c.is_whitespace()).filter(|w| !w.is_empty()) {
                let tv = match d.parse::<f64>() {
                    Ok(v) => v,
                    Err(_) => {
                        let ex = ferrostat_parser::parser::parse_expression(d).map_err(|e| rc_err(198, e.message))?;
                        crate::eval::Env::with_store(ds, st).eval(&ex, 0)?.num()?
                    }
                };
                let p = r.times.iter().position(|&t| t >= tv).ok_or_else(|| rc_err(198, "break date outside the estimation sample"))?;
                if p == 0 {
                    return Err(rc_err(198, "break date outside the estimation sample"));
                }
                brk.push(p);
                dates.push(tv);
            }
            let test = test_cols(&r, cmd, ds)?;
            let w = wald_break(&r, &brk, &test).ok_or_else(|| rc_err(498, "test could not be computed"))?;
            let df = (test.len() * brk.len()) as f64;
            let p = dist::chi2tail(df, w);
            out.push("Wald test for a structural break: Known break date".into());
            out.push(String::new());
            out.push(format!("Number of obs = {:>10}", n));
            out.push(format!("Sample: {} thru {}", lab(r.times[0]), lab(r.times[n - 1])));
            out.push(format!("Break date{}: {}", if dates.len() > 1 { "s" } else { "" }, dates.iter().map(|&d| lab(d)).collect::<Vec<_>>().join(", ")));
            out.push("Ho: No structural break".into());
            out.push(String::new());
            out.push(format!("{:>14} = {:>10.4}", format!("chi2({})", df), w));
            out.push(format!("{:>14} = {:>10.4}", "Prob > chi2", p));
            out.push(String::new());
            out.extend(footer(&r, &test));
            res.set("chi2", w);
            res.set("p", p);
            res.set("df", df);
            res.set("nbreaks", brk.len() as f64);
            res.set_str("breakdate", dates.iter().map(|&d| lab(d)).collect::<Vec<_>>().join(" "));
        }
        "sbsingle" => {
            let test = test_cols(&r, cmd, ds)?;
            let trim = util::opt_num(cmd, "trim", 4)?.unwrap_or(15.0) / 100.0;
            let lo = ((trim * n as f64).floor() as usize).max(1);
            let hi = n - lo;
            let mut best = (0usize, f64::NEG_INFINITY);
            let mut sum = 0.0;
            let mut esum = 0.0;
            let mut cnt = 0.0;
            for p in lo..=hi {
                if let Some(w) = wald_break(&r, &[p], &test) {
                    if w > best.1 {
                        best = (p, w);
                    }
                    sum += w;
                    esum += (w / 2.0).exp();
                    cnt += 1.0;
                }
            }
            if cnt == 0.0 {
                return Err(rc_err(498, "test could not be computed"));
            }
            let sims = simulate(test.len(), trim);
            let mut stats = vec![("swald", best.1, sim_p(&sims[0], best.1))];
            if cmd.has_opt("all", 3) || cmd.has_opt("awald", 2) {
                stats.push(("awald", sum / cnt, sim_p(&sims[1], sum / cnt)));
            }
            if cmd.has_opt("all", 3) || cmd.has_opt("ewald", 2) {
                let ev = (esum / cnt).ln();
                stats.push(("ewald", ev, sim_p(&sims[2], ev)));
            }
            out.push("Test for a structural break: Unknown break date".into());
            out.push(String::new());
            out.push(format!("Number of obs = {:>10}", n));
            out.push(String::new());
            out.push(format!("Full sample:    {} thru {}", lab(r.times[0]), lab(r.times[n - 1])));
            out.push(format!("Trimmed sample: {} thru {}", lab(r.times[lo]), lab(r.times[hi])));
            out.push(format!("Estimated break date: {}", lab(r.times[best.0])));
            out.push("Ho: No structural break".into());
            out.push(String::new());
            out.push("       Test   Statistic     p-value".into());
            out.push("    -------------------------------".into());
            for (nm, s, p) in &stats {
                out.push(format!("    {:>7}   {:>9.4}   {:>9.4}", nm, s, p));
                res.set(nm, *s);
                res.set(&format!("p_{nm}"), *p);
            }
            out.push("    -------------------------------".into());
            out.push(String::new());
            out.extend(footer(&r, &test));
            res.set("breakdate", r.times[best.0]);
            res.set("chi2", best.1);
            res.set("p", stats[0].2);
            res.set("df", test.len() as f64);
        }
        _ => {
            // sbcusum
            let use_ols = cmd.has_opt("ols", 3);
            let (stat, crit) = if use_ols {
                let (b, _, ssr) = ols(&r.x, &r.y).ok_or_else(|| rc_err(498, "regression failed"))?;
                let s = (ssr / (n - k) as f64).sqrt();
                let mut c = 0.0f64;
                let mut mx = 0.0f64;
                for t in 0..n {
                    c += r.y[t] - ts_ml::dot(&r.x[t], &b);
                    mx = mx.max(c.abs());
                }
                (mx / (s * (n as f64).sqrt()), [1.6276, 1.3581, 1.2238])
            } else {
                // recursive residuals
                let mut w = Vec::new();
                for t in k..n {
                    let Some((b, inv, _)) = ols(&r.x[..t], &r.y[..t]) else { continue };
                    let f = 1.0 + ts_ml::dot(&r.x[t], &ts_ml::mv(&inv, &r.x[t]));
                    w.push((r.y[t] - ts_ml::dot(&r.x[t], &b)) / f.sqrt());
                }
                let m = w.len() as f64;
                let mean = w.iter().sum::<f64>() / m;
                let s = (w.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
                let mut c = 0.0;
                let mut mx = 0.0f64;
                for (j, v) in w.iter().enumerate() {
                    c += v;
                    let bound = 1.0 + 2.0 * (j as f64 + 1.0) / m;
                    mx = mx.max((c / (s * m.sqrt())).abs() / bound);
                }
                (mx, [1.1430, 0.9479, 0.8499])
            };
            out.push("Cumulative sum test for parameter stability".into());
            out.push(String::new());
            out.push(format!("{:<48}Number of obs = {:>10}", format!("Sample: {} thru {}", lab(r.times[0]), lab(r.times[n - 1])), n));
            out.push("Ho: No structural break".into());
            out.push(String::new());
            out.push("                 Test     1% critical     5% critical    10% critical".into());
            out.push("     Type     statistic         value           value           value".into());
            out.push("  ----------------------------------------------------------------------".into());
            out.push(format!("  {:>9}  {:>11.4}  {:>14.4}  {:>14.4}  {:>14.4}", if use_ols { "OLS" } else { "Recursive" }, stat, crit[0], crit[1], crit[2]));
            out.push("  ----------------------------------------------------------------------".into());
            res.set("statistic", stat);
            res.set("cvalue1", crit[0]);
            res.set("cvalue5", crit[1]);
            res.set("cvalue10", crit[2]);
            res.set_str("cusum", if use_ols { "ols" } else { "recursive" });
        }
    }
    res.set("N", n as f64);
    st.r = res;
    Ok(out.join("\n"))
}
