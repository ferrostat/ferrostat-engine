//! `mswitch dr|ar`: Markov-switching dynamic regression and autoregression
//! estimated by maximum likelihood with the Hamilton filter; `predict`
//! (yhat, pr) and `estat transition` / `estat duration`.
//!
//!   dr: y_t = mu_{s_t} + x_t a + z_t b_{s_t} + e_t,      e_t ~ N(0, sigma_{s_t}^2)
//!   ar: y_t = mu_{s_t} + x_t a + z_t b_{s_t}
//!             + sum_i phi_i (y_{t-i} - mu_{s_{t-i}} - x_{t-i} a - z_{t-i} b_{s_{t-i}}) + e_t
//!
//! Transition probabilities p_ij = P(s_t = j | s_{t-1} = i) are estimated
//! on the multinomial-logit scale (last state is the base of every row);
//! the filter starts from the ergodic distribution.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::ts_ml::{self, LN2PI};
use crate::tsops;
use crate::tsreg::hline;
use crate::util;

#[derive(Clone, Debug)]
struct Spec {
    ar: Vec<usize>,
    m: usize,
    kn: usize,       // nonswitching covariates
    ks: usize,       // switching covariates (incl. constant)
    varswitch: bool,
    arswitch: bool,
}

impl Spec {
    fn p(&self) -> usize {
        self.ar.iter().max().copied().unwrap_or(0)
    }
    fn n_ar(&self) -> usize {
        self.ar.len() * if self.arswitch { self.m } else { 1 }
    }
    fn n_sig(&self) -> usize {
        if self.varswitch { self.m } else { 1 }
    }
    fn npar(&self) -> usize {
        self.kn + self.ks * self.m + self.n_ar() + self.n_sig() + self.m * (self.m - 1)
    }
    fn i_ar(&self) -> usize {
        self.kn + self.ks * self.m
    }
    fn i_sig(&self) -> usize {
        self.i_ar() + self.n_ar()
    }
    fn i_p(&self) -> usize {
        self.i_sig() + self.n_sig()
    }
    fn trans(&self, th: &[f64]) -> Vec<Vec<f64>> {
        let m = self.m;
        let mut p = vec![vec![0.0; m]; m];
        for i in 0..m {
            let q: Vec<f64> = (0..m - 1).map(|j| th[self.i_p() + i * (m - 1) + j]).collect();
            let mx = q.iter().fold(0.0f64, |a, b| a.max(*b));
            let den = (-mx).exp() + q.iter().map(|v| (v - mx).exp()).sum::<f64>();
            for j in 0..m - 1 {
                p[i][j] = (q[j] - mx).exp() / den;
            }
            p[i][m - 1] = (-mx).exp() / den;
        }
        p
    }
}

fn ergodic(p: &[Vec<f64>]) -> Vec<f64> {
    let m = p.len();
    let mut pi = vec![1.0 / m as f64; m];
    for _ in 0..5000 {
        let n: Vec<f64> = (0..m).map(|j| (0..m).map(|i| pi[i] * p[i][j]).sum()).collect();
        let d: f64 = n.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum();
        pi = n;
        if d < 1e-14 {
            break;
        }
    }
    pi
}

struct Data {
    y: Vec<f64>,
    xn: Vec<Vec<f64>>,
    xs: Vec<Vec<f64>>,
}

struct FilterOut {
    ll: Vec<f64>,
    /// filtered and predicted marginal state probabilities per used period
    filt: Vec<Vec<f64>>,
    pred: Vec<Vec<f64>>,
    yhat: Vec<f64>,
}

/// Hamilton filter over the joint states (s_t, ..., s_{t-p}).
fn hfilter(sp: &Spec, th: &[f64], d: &Data) -> Option<FilterOut> {
    let m = sp.m;
    let p = sp.p();
    let n = d.y.len();
    let tp = sp.trans(th);
    let nj = m.pow(p as u32 + 1);
    // joint state j encodes s_t = j % m, s_{t-1} = (j / m) % m, ...
    let digit = |j: usize, l: usize| (j / m.pow(l as u32)) % m;
    let sig: Vec<f64> = (0..m).map(|s| th[sp.i_sig() + if sp.varswitch { s } else { 0 }].exp()).collect();
    if sig.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    let mean = |t: usize, s: usize| -> f64 { ts_ml::dot(&d.xn[t], &th[..sp.kn]) + ts_ml::dot(&d.xs[t], &th[sp.kn + s * sp.ks..sp.kn + (s + 1) * sp.ks]) };
    let phi = |i: usize, s: usize| -> f64 { th[sp.i_ar() + if sp.arswitch { s * sp.ar.len() } else { 0 } + i] };
    // initial joint distribution of (s_{p}, ..., s_0)
    let pi = ergodic(&tp);
    let mut xi = vec![0.0; nj];
    for j in 0..nj {
        let mut pr = pi[digit(j, p)];
        for l in (0..p).rev() {
            pr *= tp[digit(j, l + 1)][digit(j, l)];
        }
        xi[j] = pr;
    }
    // xi is the prediction for period p (first usable)
    let mut out = FilterOut { ll: Vec::with_capacity(n), filt: Vec::new(), pred: Vec::new(), yhat: Vec::new() };
    for t in p..n {
        if t > p {
            let mut nx = vec![0.0; nj];
            for j in 0..nj {
                if xi[j] == 0.0 {
                    continue;
                }
                // shift: new joint = (s_new, s_t-1 = digit0 of j, ...)
                let base = (j % m.pow(p as u32)) * m;
                for s in 0..m {
                    nx[base + s] += xi[j] * tp[j % m][s];
                }
            }
            xi = nx;
        }
        let mut dens = vec![0.0; nj];
        let mut tot = 0.0;
        let mut yh = 0.0;
        for j in 0..nj {
            let s = digit(j, 0);
            let mut mu = mean(t, s);
            for (i, &l) in sp.ar.iter().enumerate() {
                let sl = digit(j, l);
                mu += phi(i, s) * (d.y[t - l] - mean(t - l, sl));
            }
            let e = (d.y[t] - mu) / sig[s];
            let f = (-0.5 * (LN2PI + e * e)).exp() / sig[s];
            dens[j] = xi[j] * f;
            tot += dens[j];
            yh += xi[j] * mu;
        }
        if !(tot > 0.0) || !tot.is_finite() {
            return None;
        }
        let mut pm = vec![0.0; m];
        for j in 0..nj {
            pm[digit(j, 0)] += xi[j];
        }
        out.pred.push(pm);
        out.ll.push(tot.ln());
        out.yhat.push(yh);
        for j in 0..nj {
            xi[j] = dens[j] / tot;
        }
        let mut fm = vec![0.0; m];
        for j in 0..nj {
            fm[digit(j, 0)] += xi[j];
        }
        out.filt.push(fm);
    }
    Some(out)
}

/// Kim smoother on the marginal state probabilities.
fn smooth(tp: &[Vec<f64>], filt: &[Vec<f64>], pred: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = filt.len();
    let m = tp.len();
    let mut s = filt.to_vec();
    for t in (0..n.saturating_sub(1)).rev() {
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..m {
                if pred[t + 1][j] > 0.0 {
                    acc += tp[i][j] * s[t + 1][j] / pred[t + 1][j];
                }
            }
            s[t][i] = filt[t][i] * acc;
        }
    }
    s
}

pub struct Mswitch;

struct Fitted {
    sp: Spec,
    th: Vec<f64>,
    data: Data,
    rows: Vec<usize>,
}

fn parse_vars(ds: &DataSet, s: &str) -> Result<Vec<String>, String> {
    tsops::expand_text(s).split_whitespace().map(|v| tsops::resolve(ds, v)).collect()
}

fn setup(ds: &DataSet, st: &Store, cmd: &Command, model: &str, main: &str) -> Result<(Spec, Data, Vec<usize>, Vec<f64>, String, Vec<String>, Vec<String>), String> {
    let (dep, indep) = crate::tsreg::split_dep(ds, main)?;
    let nonsw = parse_vars(ds, &indep)?;
    let (sw_text, sw_nocons) = match cmd.opt("switch", 3) {
        Some(s) => {
            let (a, b) = s.split_once(',').unwrap_or((s, ""));
            (a.to_string(), b.to_lowercase().contains("nocons"))
        }
        None => (String::new(), false),
    };
    let mut sw = parse_vars(ds, &sw_text)?;
    let constant = !(cmd.has_opt("noconstant", 4) || sw_nocons);
    if constant {
        sw.push("_cons".into());
    }
    let m = util::opt_num(cmd, "states", 5)?.unwrap_or(2.0) as usize;
    if m < 2 {
        return Err(rc_err(198, "states() must be at least 2"));
    }
    let ar = if model == "ar" { ts_ml::parse_lags(cmd.opt("ar", 2).ok_or_else(|| rc_err(198, "option ar() required"))?)? } else { vec![] };
    let p = ar.iter().max().copied().unwrap_or(0);
    let yv = tsops::values(ds, st, &dep)?;
    let colv = |n: &String| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) };
    let cn: Vec<Vec<f64>> = nonsw.iter().map(colv).collect::<Result<_, _>>()?;
    let cs: Vec<Vec<f64>> = sw.iter().map(colv).collect::<Result<_, _>>()?;
    let mask = util::touse(ds, st, cmd)?;
    let mut refs: Vec<&Vec<f64>> = vec![&yv];
    refs.extend(cn.iter());
    refs.extend(cs.iter());
    let (rows, times) = tsops::ordered_rows(ds, st, &mask, &refs)?;
    if tsops::gaps(&times) > 0 {
        return Err(rc_err(498, "gaps not allowed"));
    }
    if rows.len() <= p + 3 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let data = Data { y: rows.iter().map(|&i| yv[i]).collect(), xn: rows.iter().map(|&i| cn.iter().map(|c| c[i]).collect()).collect(), xs: rows.iter().map(|&i| cs.iter().map(|c| c[i]).collect()).collect() };
    let sp = Spec { ar, m, kn: nonsw.len(), ks: sw.len(), varswitch: cmd.has_opt("varswitch", 4), arswitch: cmd.has_opt("arswitch", 4) };
    Ok((sp, data, rows, times, dep, nonsw, sw))
}

impl CommandHandler for Mswitch {
    fn name(&self) -> &'static str {
        "mswitch"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (model, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        if model.is_empty() {
            return crate::models::replay(st, "mswitch");
        }
        if model != "dr" && model != "ar" {
            return Err(rc_err(198, format!("unknown model {model}; use dr or ar")));
        }
        let (sp, data, rows, times, dep, nonsw, sw) = setup(ds, st, cmd, model, rest)?;
        let p = sp.p();
        let m = sp.m;
        let llf = |th: &[f64]| -> Option<Vec<f64>> { hfilter(&sp, th, &data).map(|o| o.ll) };
        // starting values: OLS for the nonswitching part, switching parts spread by quantiles of the residual
        let mut xall: Vec<Vec<f64>> = Vec::new();
        for t in 0..data.y.len() {
            let mut r = data.xn[t].clone();
            r.extend(data.xs[t].iter().copied());
            for &l in &sp.ar {
                r.push(if t >= l { data.y[t - l] } else { 0.0 });
            }
            xall.push(r);
        }
        let used: Vec<usize> = (p..data.y.len()).collect();
        let b = ts_ml::lsq(&used.iter().map(|&t| xall[t].clone()).collect::<Vec<_>>(), &used.iter().map(|&t| data.y[t]).collect::<Vec<_>>());
        let resid: Vec<f64> = used.iter().map(|&t| data.y[t] - ts_ml::dot(&xall[t], &b)).collect();
        let sd = (resid.iter().map(|v| v * v).sum::<f64>() / resid.len() as f64).sqrt();
        let cons_pos = sw.iter().position(|n| n == "_cons");
        let mk = |spread: f64, stay: f64, sfac: f64| -> Vec<f64> {
            let mut th = vec![0.0; sp.npar()];
            th[..sp.kn].copy_from_slice(&b[..sp.kn]);
            for s in 0..m {
                for j in 0..sp.ks {
                    let mut v = b[sp.kn + j];
                    if Some(j) == cons_pos {
                        let off = if m > 1 { -1.0 + 2.0 * s as f64 / (m - 1) as f64 } else { 0.0 };
                        let arsum: f64 = sp.ar.iter().enumerate().map(|(i, _)| b[sp.kn + sp.ks + i]).sum();
                        v = v / (1.0 - arsum).max(0.05) + off * spread * sd;
                        if sp.ar.is_empty() {
                            v = b[sp.kn + j] + off * spread * sd;
                        }
                    }
                    th[sp.kn + s * sp.ks + j] = v;
                }
            }
            for k in 0..sp.n_ar() {
                th[sp.i_ar() + k] = b[sp.kn + sp.ks + k % sp.ar.len().max(1)];
            }
            for k in 0..sp.n_sig() {
                th[sp.i_sig() + k] = (sd * sfac).ln();
            }
            let off = ((m - 1) as f64 * (1.0 - stay) / stay).max(1e-6);
            for i in 0..m {
                for j in 0..m - 1 {
                    th[sp.i_p() + i * (m - 1) + j] = if i == j { (stay / ((1.0 - stay) / (m - 1) as f64)).ln() } else if i == m - 1 { (1.0 / (m - 1) as f64 * (1.0 - stay) / stay).ln() } else { 0.0 };
                }
            }
            let _ = off;
            th
        };
        let mut th0 = mk(1.0, 0.9, 0.6);
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (a, s, f) in [(0.5, 0.9, 0.8), (1.5, 0.95, 0.5), (1.0, 0.7, 0.6), (2.0, 0.9, 0.4), (0.5, 0.8, 0.9)] {
            let c = mk(a, s, f);
            let l = ts_ml::ll_sum(&llf, &c);
            if l > best {
                best = l;
                th0 = c;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, sp.npar())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let tech = ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs);
        let opt = ts_ml::optimize(&llf, th0, maxiter, tech, "log likelihood", true)?;
        let mut th = opt.th.clone();
        // order the states by their constant (or first switching coefficient)
        let key = |th: &[f64], s: usize| th[sp.kn + s * sp.ks + cons_pos.unwrap_or(0)];
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| key(&th, a).total_cmp(&key(&th, b)));
        if order.iter().enumerate().any(|(i, &o)| i != o) {
            let tp = sp.trans(&th);
            let mut nt = th.clone();
            for (ns, &os) in order.iter().enumerate() {
                for j in 0..sp.ks {
                    nt[sp.kn + ns * sp.ks + j] = th[sp.kn + os * sp.ks + j];
                }
                if sp.arswitch {
                    for i in 0..sp.ar.len() {
                        nt[sp.i_ar() + ns * sp.ar.len() + i] = th[sp.i_ar() + os * sp.ar.len() + i];
                    }
                }
                if sp.varswitch {
                    nt[sp.i_sig() + ns] = th[sp.i_sig() + os];
                }
            }
            for (ni, &oi) in order.iter().enumerate() {
                let row: Vec<f64> = order.iter().map(|&oj| tp[oi][oj]).collect();
                let last = row[m - 1].max(1e-300);
                for j in 0..m - 1 {
                    nt[sp.i_p() + ni * (m - 1) + j] = (row[j].max(1e-300) / last).ln();
                }
            }
            th = nt;
        }
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "oim")?;
        let k = sp.npar();
        // names (estimation metric)
        let base = dep.rsplit('.').next().unwrap_or(&dep).to_string();
        let lag = |l: usize| if l == 1 { "L.ar".to_string() } else { format!("L{l}.ar") };
        let mut names: Vec<String> = nonsw.iter().map(|n| format!("{base}:{n}")).collect();
        if !sp.arswitch {
            names.extend(sp.ar.iter().map(|&l| format!("{base}:{}", lag(l))));
        }
        let mut post_names = names.clone();
        for s in 0..m {
            for n in &sw {
                post_names.push(format!("State{}:{n}", s + 1));
            }
        }
        if sp.arswitch {
            for s in 0..m {
                post_names.extend(sp.ar.iter().map(|&l| format!("State{}:{}", s + 1, lag(l))));
            }
        } else {
            // AR terms were named above but sit after the switching block in th
            post_names.retain(|n| !n.ends_with(".ar"));
            post_names.extend(sp.ar.iter().map(|&l| format!("{base}:{}", lag(l))));
        }
        if sp.varswitch {
            post_names.extend((1..=m).map(|s| format!("lnsigma{s}:_cons")));
        } else {
            post_names.push("lnsigma:_cons".into());
        }
        for i in 0..m {
            for j in 0..m - 1 {
                post_names.push(format!("p{}{}:_cons", i + 1, j + 1));
            }
        }
        // display
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let n_used = data.y.len() - p;
        let nf = n_used as f64;
        let ll = opt.ll;
        let kf = k as f64;
        let aic = (-2.0 * ll + 2.0 * kf) / nf;
        let hqic = (-2.0 * ll + 2.0 * kf * nf.ln().ln()) / nf;
        let sbic = (-2.0 * ll + kf * nf.ln()) / nf;
        let tvar = tsops::time_var(st)?;
        let s0 = tsops::time_label(ds, &tvar, times[p]);
        let s1 = tsops::time_label(ds, &tvar, *times.last().unwrap());
        let title = if model == "dr" { "Markov-switching dynamic regression" } else { "Markov-switching autoregression" };
        let mut out = vec![String::new()];
        out.extend(opt.log.iter().cloned());
        out.push(String::new());
        out.push(title.to_string());
        out.push(String::new());
        out.push(hline(&format!("Sample: {s0} thru {s1}"), "Number of obs", &fmt_g(nf, 10)));
        out.push(hline(&format!("Number of states = {m}"), "AIC", &format!("{:.4}", aic)));
        out.push(hline("Unconditional probabilities: transition", "HQIC", &format!("{:.4}", hqic)));
        out.push(hline("", "SBIC", &format!("{:.4}", sbic)));
        out.push(format!("Log likelihood = {}", fmt_g(ll, 9)));
        out.push(String::new());
        // regression coefficients in display order: nonswitching + AR (common), then states
        let idx_of = |name: &str| post_names.iter().position(|n| n == name).unwrap();
        let mut disp: Vec<usize> = Vec::new();
        for n in &post_names {
            if n.starts_with(&format!("{base}:")) {
                disp.push(idx_of(n));
            }
        }
        for s in 0..m {
            for n in &post_names {
                if n.starts_with(&format!("State{}:", s + 1)) {
                    disp.push(idx_of(n));
                }
            }
        }
        let dn: Vec<String> = disp.iter().map(|&i| post_names[i].clone()).collect();
        let db: Vec<f64> = disp.iter().map(|&i| th[i]).collect();
        let mut dv = Matrix::zeros(disp.len(), disp.len());
        for (a, &i) in disp.iter().enumerate() {
            for (c, &j) in disp.iter().enumerate() {
                dv[(a, c)] = v[(i, j)];
            }
        }
        let zeros = vec![false; disp.len()];
        let mut table = crate::est::coef_table(&crate::est::TableSpec { depvar: &dep, names: &dn, b: &db, v: &dv, omitted: &zeros, is_base: &zeros, df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = table.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        let sep = format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64));
        let anc = |label: &str, est: f64, se: f64, lo: f64, hi: f64| format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", label, fmt_g(est, 8), fmt_g(se, 8), "", fmt_g(lo, 8), fmt_g(hi, 8));
        let mut r = Results::new();
        let _ = &mut r;
        for s in 0..sp.n_sig() {
            let i = sp.i_sig() + s;
            let ls = th[i];
            let se = v[(i, i)].max(0.0).sqrt();
            table.push(sep.clone());
            let lab = if sp.varswitch { format!("sigma{}", s + 1) } else { "sigma".into() };
            table.push(anc(&lab, ls.exp(), ls.exp() * se, (ls - z * se).exp(), (ls + z * se).exp()));
        }
        // transition probabilities: delta method on the logit scale
        let tp = sp.trans(&th);
        for i in 0..m {
            for j in 0..m - 1 {
                let pij = tp[i][j];
                // dp_ij/dq_ik = p_ij (1{j=k} - p_ik)
                let g: Vec<(usize, f64)> = (0..m - 1).map(|kk| (sp.i_p() + i * (m - 1) + kk, pij * (if kk == j { 1.0 } else { 0.0 } - tp[i][kk]))).collect();
                let var: f64 = g.iter().map(|&(a, ga)| g.iter().map(|&(c, gc)| ga * gc * v[(a, c)]).sum::<f64>()).sum();
                let se = var.max(0.0).sqrt();
                let lg = (pij / (1.0 - pij)).ln();
                let sel = se / (pij * (1.0 - pij));
                let (lo, hi) = (dist::invlogit(lg - z * sel), dist::invlogit(lg + z * sel));
                table.push(sep.clone());
                table.push(anc(&format!("p{}{}", i + 1, j + 1), pij, se, lo, hi));
            }
        }
        table.push(bar);
        out.extend(crate::tsreg::with_vcetype(table, vcetype));
        if !opt.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let e = st.begin_e();
        e.set("N", nf);
        e.set("k", kf);
        e.set("ll", ll);
        e.set("aic", aic);
        e.set("hqic", hqic);
        e.set("sbic", sbic);
        e.set("states", m as f64);
        e.set("k_eq", (post_names.iter().map(|n| n.split(':').next().unwrap()).collect::<std::collections::BTreeSet<_>>().len()) as f64);
        e.set("tmin", times[p]);
        e.set("tmax", *times.last().unwrap());
        e.set("converged", if opt.converged { 1.0 } else { 0.0 });
        e.set("ic", opt.iters as f64);
        e.set("rank", kf);
        e.set_str("cmd", "mswitch");
        e.set_str("model", model);
        e.set_str("cmdline", format!("mswitch {}", cmd.raw.trim_start_matches(cmd.verb.as_str()).trim()));
        e.set_str("depvar", dep.clone());
        e.set_str("title", title);
        e.set_str("tmins", s0);
        e.set_str("tmaxs", s1);
        e.set_str("nonswitchvars", nonsw.join(" "));
        e.set_str("switchvars", sw.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
        e.set_str("ar", sp.ar.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
        e.set_str("vce", vce_name);
        if !vcetype.is_empty() {
            e.set_str("vcetype", vcetype);
        }
        e.set_str("predict", "mswitch_p");
        e.set_str("estat_cmd", "mswitch_estat");
        e.set_str("properties", "b V");
        e.set_str("_output", text.clone());
        e.set_str("_ms", format!("{}|{}|{}", u8::from(sp.varswitch), u8::from(sp.arswitch), m));
        let mut pm = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                pm[(i, j)] = tp[i][j];
            }
        }
        e.set_mat("p", pm);
        e.set_mat("b", Matrix::row_vector(&th).with_names(vec!["y1".into()], post_names.clone()));
        e.set_mat("V", v.with_names(post_names.clone(), post_names));
        let mut sample = vec![false; ds.n_obs];
        for &r in rows.iter().skip(p) {
            sample[r] = true;
        }
        e.sample = Some(sample);
        let _ = Fitted { sp: sp.clone(), th: vec![], data: Data { y: vec![], xn: vec![], xs: vec![] }, rows: vec![] };
        let _ = SYSMISS;
        Ok(text)
    }
}

fn refit(ds: &DataSet, st: &Store) -> Result<Fitted, String> {
    let e = &st.e;
    let cmdline = e.macros.get("cmdline").cloned().unwrap_or_default();
    let cmd = ferrostat_parser::parser::parse_line(&cmdline).map_err(|e| rc_err(198, e.message))?;
    let main = cmd.main.trim();
    let (model, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
    let (sp, data, rows, _, _, _, _) = setup(ds, st, &cmd, model, rest)?;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    Ok(Fitted { sp, th: b.data.clone(), data, rows })
}

/// `predict` after mswitch: yhat (default) or pr (state probabilities, smoothed unless smethod(filter)).
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let f = refit(ds, st)?;
    let o = hfilter(&f.sp, &f.th, &f.data).ok_or_else(|| rc_err(498, "filter failed"))?;
    let p = f.sp.p();
    let m = f.sp.m;
    let words: Vec<String> = cmd.main.split_whitespace().filter(|w| ferrostat_core::VarType::from_type_str(w).is_none()).map(String::from).collect();
    let mask = util::touse(ds, st, cmd)?;
    if cmd.has_opt("pr", 2) {
        let filt = cmd.opt("smethod", 2).is_some_and(|s| s.trim().starts_with("filt"));
        let probs = if filt { o.filt.clone() } else { smooth(&f.sp.trans(&f.th), &o.filt, &o.pred) };
        let names: Vec<String> = if words.len() == 1 && words[0].ends_with('*') { (1..=m).map(|s| format!("{}{s}", words[0].trim_end_matches('*'))).collect() } else { words.clone() };
        if names.is_empty() || names.len() > m {
            return Err(rc_err(198, "invalid new variable list"));
        }
        let mut msgs = Vec::new();
        for (s, n) in names.iter().enumerate() {
            let mut v = vec![SYSMISS; ds.n_obs];
            for (t, &r) in f.rows.iter().enumerate().skip(p) {
                v[r] = probs[t - p][s];
            }
            msgs.push(ts_ml::add_prediction(ds, n, v, &mask, "")?);
        }
        return Ok(msgs.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"));
    }
    if words.len() != 1 {
        return Err(rc_err(198, "one new variable name required"));
    }
    let mut v = vec![SYSMISS; ds.n_obs];
    for (t, &r) in f.rows.iter().enumerate().skip(p) {
        v[r] = o.yhat[t - p];
    }
    ts_ml::add_prediction(ds, &words[0], v, &mask, if cmd.has_opt("yhat", 2) { "" } else { "(option yhat assumed; predicted values)" })
}

/// `estat transition` and `estat duration`.
pub fn estat(st: &mut Store, sub: &str) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("mswitch") {
        return None;
    }
    let p = st.e.matrices.get("p")?.clone();
    let m = p.rows;
    match sub {
        "transition" | "trans" => {
            let mut out = vec![String::new(), "Number of obs = ".to_string() + &fmt_g(st.e.get("N").unwrap_or(0.0), 10), String::new()];
            out.push("Transition probabilities".into());
            out.push("-------------------------------".into());
            out.push("             |    Estimate".into());
            out.push("-------------+-----------------".into());
            for i in 0..m {
                for j in 0..m {
                    out.push(format!("{:>12} | {:>10}", format!("p{}{}", i + 1, j + 1), fmt_g(p[(i, j)], 8)));
                }
            }
            out.push("-------------------------------".into());
            let mut r = Results::new();
            r.set_mat("p", p);
            st.r = r;
            Some(Ok(out.join("\n")))
        }
        "duration" | "dur" => {
            let mut out = vec![String::new(), "Expected duration".into(), "-------------------------------".into(), "             |    Estimate".into(), "-------------+-----------------".into()];
            let mut d = Matrix::zeros(1, m);
            for i in 0..m {
                let v = 1.0 / (1.0 - p[(i, i)]);
                d[(0, i)] = v;
                out.push(format!("{:>12} | {:>10}", format!("State{}", i + 1), fmt_g(v, 8)));
            }
            out.push("-------------------------------".into());
            let mut r = Results::new();
            r.set_mat("duration", d);
            st.r = r;
            Some(Ok(out.join("\n")))
        }
        _ => None,
    }
}
