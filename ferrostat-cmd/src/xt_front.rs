//! `xtfrontier`: stochastic frontier models for panel data with
//! time-invariant (`ti`) or time-varying decay (`tvd`, Battese and Coelli
//! 1992) inefficiency, truncated-normal u_i.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec, Likelihood};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

type Handlers = Vec<(&'static str, Box<dyn CommandHandler>)>;

pub fn handlers() -> Handlers {
    vec![("xtfrontier", Box::new(XtFrontier))]
}

struct Front<'a> {
    x: Vec<&'a [f64]>,
    y: &'a [f64],
    time: &'a [f64],
    groups: &'a [Vec<usize>],
    tvd: bool,
    sgn: f64,
}

impl<'a> Front<'a> {
    fn np(&self) -> usize {
        self.x.len() + if self.tvd { 4 } else { 3 }
    }
    fn ll(&self, b: &[f64]) -> f64 {
        let k = self.x.len();
        let mu = b[k];
        let eta = if self.tvd { b[k + 1] } else { 0.0 };
        let o = if self.tvd { k + 2 } else { k + 1 };
        let s2 = b[o].exp();
        let gam = dist::invlogit(b[o + 1]);
        let su2 = gam * s2;
        let sv2 = (1.0 - gam) * s2;
        if !(su2 > 0.0 && sv2 > 0.0) {
            return f64::NEG_INFINITY;
        }
        let su = su2.sqrt();
        let mut ll = 0.0;
        for g in self.groups {
            let tmax = g.iter().map(|&t| self.time[t]).fold(f64::NEG_INFINITY, f64::max);
            let (mut ee, mut ne, mut nn) = (0.0, 0.0, 0.0);
            for &t in g {
                let e = self.y[t] - self.x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum::<f64>();
                let h = (-eta * (self.time[t] - tmax)).exp();
                ee += e * e;
                ne += h * e;
                nn += h * h;
            }
            let ti = g.len() as f64;
            let den = sv2 + su2 * nn;
            let mus = (mu * sv2 - self.sgn * su2 * ne) / den;
            let ss = (sv2 * su2 / den).sqrt();
            ll += -0.5 * ti * (2.0 * std::f64::consts::PI).ln() - 0.5 * (ti - 1.0) * sv2.ln() - 0.5 * den.ln() - ee / (2.0 * sv2) + 0.5 * (mus / ss).powi(2) - 0.5 * (mu / su).powi(2) + crate::xtnl::ln_phi(mus / ss) - crate::xtnl::ln_phi(mu / su);
        }
        ll
    }
    fn grad(&self, b: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; b.len()];
        let mut bb = b.to_vec();
        for j in 0..b.len() {
            let h = 1e-4 * (1.0 + b[j].abs());
            let mut d = |h: f64| {
                bb[j] = b[j] + h;
                let a = self.ll(&bb);
                bb[j] = b[j] - h;
                let c = self.ll(&bb);
                bb[j] = b[j];
                (a - c) / (2.0 * h)
            };
            let d1 = d(h);
            let d2 = d(h / 2.0);
            g[j] = (4.0 * d2 - d1) / 3.0;
        }
        g
    }
}

impl<'a> Likelihood for Front<'a> {
    fn k(&self) -> usize {
        self.np()
    }
    fn eval(&self, b: &[f64], _scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let p = self.np();
        let ll = self.ll(b);
        if !ll.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; p], Matrix::identity(p), vec![]);
        }
        let g = self.grad(b);
        let mut h = Matrix::zeros(p, p);
        let mut bb = b.to_vec();
        for j in 0..p {
            let s = 1e-4 * (1.0 + b[j].abs());
            bb[j] = b[j] + s;
            let gp = self.grad(&bb);
            bb[j] = b[j] - s;
            let gm = self.grad(&bb);
            bb[j] = b[j];
            for i in 0..p {
                h[(i, j)] = -(gp[i] - gm[i]) / (2.0 * s);
            }
        }
        for i in 0..p {
            for j in 0..i {
                let m = 0.5 * (h[(i, j)] + h[(j, i)]);
                h[(i, j)] = m;
                h[(j, i)] = m;
            }
        }
        (ll, g, h, vec![])
    }
}

pub struct XtFrontier;

impl CommandHandler for XtFrontier {
    fn name(&self) -> &'static str {
        "xtfrontier"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return st.e.macros.get("_output").cloned().filter(|_| st.e.macros.get("cmd").map(String::as_str) == Some("xtfrontier")).ok_or_else(|| rc_err(301, "last estimates not found"));
        }
        let tvd = cmd.has_opt("tvd", 3);
        if !tvd && !cmd.has_opt("ti", 2) {
            return Err(rc_err(198, "must specify either ti or tvd"));
        }
        let sgn = if cmd.has_opt("cost", 4) { -1.0 } else { 1.0 };
        let main = cmd.main.trim();
        let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let ivar = crate::xtnl::panel_var(ds, st, cmd)?;
        let tvar = crate::xtnl::time_var(st).ok();
        if tvd && tvar.is_none() {
            return Err(rc_err(459, "time variable not set; use xtset panelvar timevar"));
        }
        let mut extra = vec![ivar.clone()];
        extra.extend(tvar.iter().cloned());
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: cmd.has_opt("noconstant", 3), weights: "", extra })?;
        let tc = match &tvar {
            Some(t) => util::col(ds, t)?,
            None => vec![0.0; ds.n_obs],
        };
        let time: Vec<f64> = d.rows.iter().map(|&i| tc[i]).collect();
        let (_, groups) = crate::xtnl::panel_groups(ds, &ivar, &d.rows)?;
        let act = d.active();
        let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let kx = act.len();
        // OLS starting values
        let xs: Vec<Vec<f64>> = cols.iter().map(|c| c.to_vec()).collect();
        let (b0, _, e) = crate::xt_lin::ls(&xs, &d.y);
        let s2 = e.iter().map(|v| v * v).sum::<f64>() / d.n() as f64;
        let m = Front { x: cols.clone(), y: &d.y, time: &time, groups: &groups, tvd, sgn };
        let mut best = (f64::NEG_INFINITY, vec![]);
        for g0 in [-1.0, 0.0, 1.0, 2.0] {
            let mut s = b0.clone();
            if let Some(c) = act.iter().position(|&j| d.names[j] == "_cons") {
                s[c] += sgn * (s2 * dist::invlogit(g0)).sqrt() * 0.8;
            }
            s.push(0.0);
            if tvd {
                s.push(0.0);
            }
            s.push(s2.ln());
            s.push(g0);
            let ll = m.ll(&s);
            if ll > best.0 {
                best = (ll, s);
            }
        }
        let res = crate::zip::maximize_robust(&m, best.1, 300)?;
        let (vth, _) = res.info.sym_inverse(1e-14);
        let k = d.k();
        let aux: Vec<&str> = if tvd { vec!["mu", "eta", "lnsigma2", "ilgtgamma"] } else { vec!["mu", "lnsigma2", "ilgtgamma"] };
        let total = k + aux.len();
        let mut idx: Vec<usize> = act.clone();
        for a in 0..aux.len() {
            idx.push(k + a);
        }
        let mut b = vec![0.0; total];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vth, &idx, total);
        let mut dd = d.clone();
        dd.names = d.names.iter().map(|n| format!("{dep}:{n}")).chain(aux.iter().map(|a| format!("/:{a}"))).collect();
        dd.omitted.extend(vec![false; aux.len()]);
        dd.is_base.extend(vec![false; aux.len()]);
        let mut vk = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                vk[(i, j)] = v[(i, j)];
            }
        }
        let (chi2, df_m) = est::wald_all(&d, &b[..k], &vk);
        let level = util::level(cmd, st)?;
        let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        let mn = groups.iter().map(|g| g.len()).min().unwrap_or(0);
        let mx = groups.iter().map(|g| g.len()).max().unwrap_or(0);
        let av = d.n() as f64 / groups.len() as f64;
        let row = |l: &str, k: &str, v: String| format!("{:<52}{:<15} = {:>10}", l, k, v);
        let title = if tvd { "Time-varying decay inefficiency model" } else { "Time-invariant inefficiency model" };
        let mut out = vec![String::new(), "Iteration log:".to_string()];
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        out.push(row(title, "Number of obs", util::commas(d.n())));
        out.push(row(&format!("Group variable: {ivar}"), "Number of groups", util::commas(groups.len())));
        if let Some(t) = &tvar {
            out.push(format!("Time variable: {t}"));
        }
        out.push(format!("{:<52}Obs per group:", ""));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "min", mn));
        out.push(format!("{:<52}{:>15} = {:>10.1}", "", "avg", av));
        out.push(format!("{:<52}{:>15} = {:>10}", "", "max", mx));
        out.push(String::new());
        out.push(row("", &format!("Wald chi2({})", fmt_g(df_m, 9)), format!("{:.2}", chi2)));
        out.push(row(&format!("Log likelihood = {}", fmt_g(res.ll, 9)), "Prob > chi2", format!("{:.4}", dist::chi2tail(df_m, chi2))));
        out.push(String::new());
        let names: Vec<String> = d.names.clone();
        let mut tbl = est::coef_table(&est::TableSpec { depvar: &dep, names: &names, b: &b[..k], v: &vk, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        let c8 = |x: f64| fmt_g(x, 8);
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        for (a, nm) in aux.iter().enumerate() {
            let j = k + a;
            let (bj, se) = (b[j], v[(j, j)].max(0.0).sqrt());
            let z = bj / se;
            tbl.push(format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", format!("/{nm}"), c8(bj), c8(se), z, 2.0 * (1.0 - dist::normal(z.abs())), c8(bj - crit * se), c8(bj + crit * se)));
        }
        tbl.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
        let o = if tvd { k + 2 } else { k + 1 };
        let (ls2, lse2) = (b[o], v[(o, o)].sqrt());
        let (ig, ise) = (b[o + 1], v[(o + 1, o + 1)].sqrt());
        let sig2 = ls2.exp();
        let gam = dist::invlogit(ig);
        let cov = v[(o, o + 1)];
        let su2 = gam * sig2;
        let sv2 = (1.0 - gam) * sig2;
        let gsig = sig2;
        let ggam = gam * (1.0 - gam);
        let var_su2 = (gam * gsig).powi(2) * lse2 * lse2 + (sig2 * ggam).powi(2) * ise * ise + 2.0 * gam * gsig * sig2 * ggam * cov;
        let var_sv2 = ((1.0 - gam) * gsig).powi(2) * lse2 * lse2 + (sig2 * ggam).powi(2) * ise * ise - 2.0 * (1.0 - gam) * gsig * sig2 * ggam * cov;
        let plain = |lab: &str, val: f64, se: f64, lo: f64, hi: f64| format!("{:>lw$} | {:>10}  {:>9}{:21}{:>9}   {:>9}", lab, c8(val), c8(se), "", c8(lo), c8(hi));
        tbl.push(plain("sigma2", sig2, sig2 * lse2, (ls2 - crit * lse2).exp(), (ls2 + crit * lse2).exp()));
        tbl.push(plain("gamma", gam, ggam * ise, dist::invlogit(ig - crit * ise), dist::invlogit(ig + crit * ise)));
        tbl.push(plain("sigma_u2", su2, var_su2.max(0.0).sqrt(), su2 - crit * var_su2.max(0.0).sqrt(), su2 + crit * var_su2.max(0.0).sqrt()));
        tbl.push(plain("sigma_v2", sv2, var_sv2.max(0.0).sqrt(), sv2 - crit * var_sv2.max(0.0).sqrt(), sv2 + crit * var_sv2.max(0.0).sqrt()));
        tbl.push(bar);
        out.extend(tbl);
        let text = out.join("\n");
        let ng = groups.len();
        est::post(st, "xtfrontier", &dd, &b, &v, &|e: &mut Results| {
            e.set("N_g", ng as f64);
            e.set("g_min", mn as f64);
            e.set("g_avg", av);
            e.set("g_max", mx as f64);
            e.set("ll", res.ll);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("sigma2", sig2);
            e.set("gamma", gam);
            e.set("sigma_u", su2.sqrt());
            e.set("sigma_v", sv2.sqrt());
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            e.set_str("model", if tvd { "tvd" } else { "ti" });
            e.set_str("function", if sgn > 0.0 { "production" } else { "cost" });
            e.set_str("ivar", ivar.clone());
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
