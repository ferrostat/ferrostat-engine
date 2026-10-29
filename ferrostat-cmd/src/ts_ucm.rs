//! `ucm` (unobserved-components models: trend, stochastic seasonal and
//! stochastic cycles) and `dfactor` (dynamic-factor models), both cast in
//! state-space form and estimated by maximum likelihood with the Kalman
//! filter of `ts_kalman` (diffuse initial states for the nonstationary
//! components). `predict` after both: xb/one-step predictions, residuals,
//! filtered states.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::ts_kalman::{kfilter, Ssm};
use crate::ts_ml::{self, Post};
use crate::tsops;
use crate::util;

// ---------------------------------------------------------------------------
// ucm
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Trend {
    level: bool,      // stochastic level variance
    slope: bool,      // stochastic slope variance
    has_level: bool,  // level state present
    has_slope: bool,  // slope state present
    irregular: bool,  // observation noise
}

fn trend_of(model: &str) -> Result<Trend, String> {
    let t = |level, slope, has_level, has_slope, irregular| Trend { level, slope, has_level, has_slope, irregular };
    Ok(match model {
        "rwalk" => t(true, false, true, false, false),
        "none" | "ntrend" => t(false, false, false, false, true),
        "dconstant" => t(false, false, true, false, true),
        "llevel" => t(true, false, true, false, true),
        "dtrend" => t(false, false, true, true, true),
        "lldtrend" => t(true, false, true, true, true),
        "rwdrift" => t(true, false, true, true, false),
        "lltrend" => t(true, true, true, true, true),
        "strend" => t(false, true, true, true, true),
        "rtrend" => t(false, true, true, true, false),
        o => return Err(rc_err(198, format!("model({o}) not allowed"))),
    })
}

struct Ucm {
    tr: Trend,
    season: usize,
    ncyc: usize,
    kx: usize,
}

impl Ucm {
    /// parameter layout: cycles (freq, damping)*, x coefs, variances (level, slope, seasonal, cycles, irregular)
    fn var_names(&self, dep: &str) -> Vec<String> {
        let mut v = Vec::new();
        if self.tr.level {
            v.push("/var(level)".to_string());
        }
        if self.tr.slope {
            v.push("/var(slope)".to_string());
        }
        if self.season > 0 {
            v.push("/var(seasonal)".to_string());
        }
        for c in 0..self.ncyc {
            v.push(format!("/var(cycle{})", c + 1));
        }
        if self.tr.irregular {
            v.push(format!("/var({dep})"));
        }
        v
    }
    fn n_states(&self) -> usize {
        usize::from(self.tr.has_level) + usize::from(self.tr.has_slope) + self.season.saturating_sub(1) + 2 * self.ncyc
    }
    fn ssm(&self, th: &[f64], x: &[Vec<f64>]) -> Option<Ssm> {
        let m = self.n_states();
        let n = x.len();
        let mut t = Matrix::zeros(m, m);
        let mut q = Matrix::zeros(m, m);
        let mut z = Matrix::zeros(1, m);
        let mut diffuse = vec![false; m];
        let nv = self.var_names("y").len();
        let vb = 2 * self.ncyc + self.kx;
        let vars = &th[vb..vb + nv];
        if vars.iter().any(|v| *v < 0.0) {
            return None;
        }
        let mut vi = 0;
        let mut s = 0;
        if self.tr.has_level {
            t[(0, 0)] = 1.0;
            z[(0, 0)] = 1.0;
            diffuse[0] = true;
            if self.tr.has_slope {
                t[(0, 1)] = 1.0;
                t[(1, 1)] = 1.0;
                diffuse[1] = true;
            }
            if self.tr.level {
                q[(0, 0)] = vars[vi];
                vi += 1;
            }
            if self.tr.slope {
                q[(1, 1)] = vars[vi];
                vi += 1;
            }
            s = 1 + usize::from(self.tr.has_slope);
        }
        if self.season > 0 {
            let k = self.season - 1;
            for j in 0..k {
                t[(s, s + j)] = -1.0;
                if j > 0 {
                    t[(s + j, s + j - 1)] = 1.0;
                }
                diffuse[s + j] = true;
            }
            z[(0, s)] = 1.0;
            q[(s, s)] = vars[vi];
            vi += 1;
            s += k;
        }
        for c in 0..self.ncyc {
            let lam = th[2 * c];
            let rho = th[2 * c + 1];
            if !(lam > 0.0 && lam < std::f64::consts::PI && rho > 0.0 && rho < 1.0) {
                return None;
            }
            t[(s, s)] = rho * lam.cos();
            t[(s, s + 1)] = rho * lam.sin();
            t[(s + 1, s)] = -rho * lam.sin();
            t[(s + 1, s + 1)] = rho * lam.cos();
            q[(s, s)] = vars[vi];
            q[(s + 1, s + 1)] = vars[vi];
            vi += 1;
            z[(0, s)] = 1.0;
            s += 2;
        }
        let mut h = Matrix::zeros(1, 1);
        if self.tr.irregular {
            h[(0, 0)] = vars[vi];
        }
        let beta = &th[2 * self.ncyc..2 * self.ncyc + self.kx];
        let d: Vec<Vec<f64>> = (0..n).map(|i| vec![ts_ml::dot(&x[i], beta)]).collect();
        Some(Ssm { m, p: 1, t, rqr: q, z, h, c: vec![], d, diffuse })
    }
}

fn states_x(ds: &DataSet, st: &Store, cmd: &Command, main: &str) -> Result<(String, Vec<String>, crate::sspace::SsData, Vec<f64>, Vec<Vec<f64>>), String> {
    let (dep, indep) = crate::tsreg::split_dep(ds, main)?;
    let xn: Vec<String> = tsops::expand_text(&indep).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
    let yv = tsops::values(ds, st, &dep)?;
    let xc: Vec<Vec<f64>> = xn.iter().map(|v| tsops::values(ds, st, v)).collect::<Result<_, _>>()?;
    let mut refs: Vec<&Vec<f64>> = vec![&yv];
    refs.extend(xc.iter());
    let sd = crate::sspace::ss_rows(ds, st, cmd, &refs)?;
    let y = sd.rows.iter().map(|&r| yv[r]).collect();
    let x = sd.rows.iter().map(|&r| xc.iter().map(|c| c[r]).collect()).collect();
    Ok((dep, xn, sd, y, x))
}

pub struct UcmCmd;

impl CommandHandler for UcmCmd {
    fn name(&self) -> &'static str {
        "ucm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "ucm");
        }
        let model = cmd.opt("model", 3).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "rwalk".into());
        let tr = trend_of(&model)?;
        let season = util::opt_num(cmd, "seasonal", 3)?.unwrap_or(0.0) as usize;
        let mut cyc_freq: Vec<Option<f64>> = Vec::new();
        for (n, v) in &cmd.opts {
            if n.to_lowercase().starts_with("cyc") {
                let order = v.as_deref().and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
                if order != 1 {
                    return Err(rc_err(198, "only first-order cycles are supported"));
                }
                let freq = v.as_deref().and_then(|s| s.split_once("frequency(")).and_then(|(_, r)| r.trim_end_matches(')').trim().parse().ok());
                cyc_freq.push(freq);
            }
        }
        let (dep, xn, sd, y, x) = states_x(ds, st, cmd, &cmd.main)?;
        let n = y.len();
        let u = Ucm { tr, season, ncyc: cyc_freq.len(), kx: xn.len() };
        let yy: Vec<Vec<Option<f64>>> = y.iter().map(|v| vec![Some(*v)]).collect();
        let llf = |th: &[f64]| -> Option<Vec<f64>> { kfilter(&u.ssm(th, &x)?, &yy).map(|o| o.ll) };
        let vy = util::variance(&y.iter().zip(0..).map(|(v, _)| *v).collect::<Vec<_>>());
        let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
        let vdy = if dy.len() > 1 { util::variance(&dy) } else { vy };
        let nv = u.var_names(&dep).len();
        let mk = |share: f64, rho: f64| -> Vec<f64> {
            let mut th = Vec::new();
            for f in &cyc_freq {
                th.push(f.unwrap_or(std::f64::consts::PI / 8.0));
                th.push(rho);
            }
            th.extend(ts_ml::lsq(&x, &y));
            for _ in 0..nv {
                th.push(vdy.max(vy * 1e-3) * share);
            }
            th
        };
        let mut th0 = mk(0.3, 0.7);
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (s, r) in [(0.1, 0.7), (0.5, 0.9), (1.0, 0.5), (0.05, 0.9), (0.2, 0.3)] {
            let c = mk(s, r);
            let l = ts_ml::ll_sum(&llf, &c);
            if l > best {
                best = l;
                th0 = c;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, th0.len())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let opt = ts_ml::optimize(&llf, th0, maxiter, ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs), "log likelihood", true)?;
        let th = opt.th.clone();
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "oim").or_else(|_| Ok::<_, String>((Matrix::zeros(th.len(), th.len()), "", "oim")))?;
        let mut names = Vec::new();
        for c in 0..u.ncyc {
            names.push(format!("cycle{}:frequency", c + 1));
            names.push(format!("cycle{}:damping", c + 1));
        }
        names.extend(xn.iter().map(|nm| format!("{dep}:{nm}")));
        names.extend(u.var_names(&dep));
        let test_idx: Vec<usize> = (0..2 * u.ncyc + u.kx).collect();
        let (chi2, df) = ts_ml::wald(&th, &v, &test_idx);
        let samp = ts_ml::TsSample { tvar: sd.tvar.clone(), y: vec![], x: vec![], xnames: vec![], times: sd.times.clone(), rows: sd.rows.clone(), mask: sd.mask.clone(), depvar: String::new(), n_gaps: tsops::gaps(&sd.times) };
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let mtitle = match model.as_str() {
            "rwalk" => "random walk",
            "llevel" => "local level",
            "lltrend" => "local-linear trend",
            "dconstant" => "deterministic constant",
            "dtrend" => "deterministic trend",
            "lldtrend" => "local-level deterministic trend",
            "rwdrift" => "random walk with drift",
            "strend" => "smooth trend",
            "rtrend" => "random trend",
            _ => "no trend",
        };
        let header = ts_ml::std_header(&sline, n, chi2, df, opt.ll, "Log likelihood");
        let title = format!("Unobserved-components model\nComponents: {mtitle}{}{}", if season > 0 { format!(", seasonal({season})") } else { String::new() }, if u.ncyc > 0 { format!(", order {} cycle", 1) } else { String::new() });
        let notes = vec!["Note: Model is not stationary.".to_string(), "Note: Tests of variances against zero are one sided, and the two-sided".to_string(), "      confidence intervals are truncated at zero.".to_string()];
        let notes = if tr.has_level || season > 0 { notes } else { notes[1..].to_vec() };
        let text = ts_ml::post(
            st,
            cmd,
            Post { title: &title, cmd: "ucm", depvar: &dep, names, b: th.clone(), v, ll: opt.ll, n, header, vcetype, vce: vce_name, log: opt.log, notes, converged: opt.converged, iters: opt.iters, df: None },
            Some(&sd.mask),
        )?;
        let e = &mut st.e;
        e.set_str("title", "Unobserved-components model");
        e.set_str("trend", model.clone());
        e.set("seasonal", season as f64);
        e.set("k_cycles", u.ncyc as f64);
        e.set("chi2", chi2);
        e.set("df_m", df);
        e.set("tmin", sd.times[0]);
        e.set("tmax", *sd.times.last().unwrap());
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("covariates", xn.join(" "));
        e.set_str("predict", "ucm_p");
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// dfactor
// ---------------------------------------------------------------------------

struct Dfac {
    deps: Vec<String>,
    xobs: Vec<String>,
    fac: Vec<String>,
    xfac: Vec<String>,
    far: Vec<usize>,
    ear: Vec<usize>,
    unstr: bool,
}

impl Dfac {
    fn k(&self) -> usize {
        self.deps.len()
    }
    fn nf(&self) -> usize {
        self.fac.len()
    }
    fn pf(&self) -> usize {
        self.far.iter().max().copied().unwrap_or(1)
    }
    fn pe(&self) -> usize {
        self.ear.iter().max().copied().unwrap_or(0)
    }
    fn names(&self) -> Vec<String> {
        let lag = |l: usize, w: &str| if l == 1 { format!("L.{w}") } else { format!("L{l}.{w}") };
        let mut v = Vec::new();
        for f in &self.fac {
            for &l in &self.far {
                for g in &self.fac {
                    v.push(format!("{f}:{}", lag(l, g)));
                }
            }
            for x in &self.xfac {
                v.push(format!("{f}:{x}"));
            }
        }
        for d in &self.deps {
            let ed = format!("e.{d}");
            for &l in &self.ear {
                v.push(format!("{ed}:{}", lag(l, &ed)));
            }
        }
        for d in &self.deps {
            for f in &self.fac {
                v.push(format!("{d}:{f}"));
            }
            for x in &self.xobs {
                v.push(format!("{d}:{x}"));
            }
        }
        if self.unstr {
            for j in 0..self.k() {
                for i in j..self.k() {
                    if i == j {
                        v.push(format!("/observable:var(e.{})", self.deps[i]));
                    } else {
                        v.push(format!("/observable:cov(e.{},e.{})", self.deps[j], self.deps[i]));
                    }
                }
            }
        } else {
            v.extend(self.deps.iter().map(|d| format!("/observable:var(e.{d})")));
        }
        v
    }
    fn ssm(&self, th: &[f64], xo: &[Vec<f64>], xf: &[Vec<f64>]) -> Option<Ssm> {
        let (k, nf, pf, pe) = (self.k(), self.nf(), self.pf(), self.pe());
        let mf = nf * pf;
        let m = mf + k * pe.max(0);
        let n = xo.len();
        let mut t = Matrix::zeros(m, m);
        let mut q = Matrix::zeros(m, m);
        let mut c = vec![vec![0.0; m]; n];
        let mut i = 0;
        for a in 0..nf {
            for &l in &self.far {
                for b in 0..nf {
                    t[(a, (l - 1) * nf + b)] = th[i];
                    i += 1;
                }
            }
            for (j, _) in self.xfac.iter().enumerate() {
                for tt in 0..n {
                    c[tt][a] += th[i] * xf[tt][j];
                }
                i += 1;
            }
            q[(a, a)] = 1.0;
        }
        for l in 1..pf {
            for a in 0..nf {
                t[(l * nf + a, (l - 1) * nf + a)] = 1.0;
            }
        }
        for d in 0..k {
            let base = mf + d * pe;
            for &l in &self.ear {
                t[(base, base + l - 1)] = th[i];
                i += 1;
            }
            for l in 1..pe {
                t[(base + l, base + l - 1)] = 1.0;
            }
        }
        let mut z = Matrix::zeros(k, m);
        let mut dd = vec![vec![0.0; k]; n];
        for d in 0..k {
            for a in 0..nf {
                z[(d, a)] = th[i];
                i += 1;
            }
            for (j, _) in self.xobs.iter().enumerate() {
                for tt in 0..n {
                    dd[tt][d] += th[i] * xo[tt][j];
                }
                i += 1;
            }
            if pe > 0 {
                z[(d, mf + d * pe)] = 1.0;
            }
        }
        let mut sig = Matrix::zeros(k, k);
        if self.unstr {
            for j in 0..k {
                for r in j..k {
                    sig[(r, j)] = th[i];
                    sig[(j, r)] = th[i];
                    i += 1;
                }
            }
        } else {
            for d in 0..k {
                sig[(d, d)] = th[i];
                i += 1;
            }
        }
        let (ev, _) = sig.sym_eigen();
        if ev.iter().any(|v| *v < 0.0) {
            return None;
        }
        let h = if pe > 0 {
            for a in 0..k {
                for b in 0..k {
                    q[(mf + a * pe, mf + b * pe)] = sig[(a, b)];
                }
            }
            Matrix::zeros(k, k)
        } else {
            sig
        };
        Some(Ssm { m, p: k, t, rqr: q, z, h, c, d: dd, diffuse: vec![false; m] })
    }
}

fn parse_dfac(ds: &DataSet, groups: &[String]) -> Result<Dfac, String> {
    let mut d = Dfac { deps: vec![], xobs: vec![], fac: vec![], xfac: vec![], far: vec![], ear: vec![], unstr: false };
    for (gi, g) in groups.iter().enumerate() {
        let (body, opts) = g.split_once(',').unwrap_or((g.as_str(), ""));
        let (l, r) = body.split_once('=').unwrap_or((body, ""));
        let ol = opts.to_lowercase();
        let nocons = ol.contains("nocons");
        let ar = ol.split_once("ar(").and_then(|(_, r)| r.split_once(')')).map(|(a, _)| a.to_string());
        let mut xs: Vec<String> = tsops::expand_text(r.trim()).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
        if gi == 0 {
            d.deps = tsops::expand_text(l.trim()).split_whitespace().map(|v| tsops::resolve(ds, v)).collect::<Result<_, _>>()?;
            if !nocons {
                xs.push("_cons".into());
            }
            d.xobs = xs;
            d.ear = ar.map_or(Ok(vec![]), |a| ts_ml::parse_lags(&a))?;
            d.unstr = ol.contains("covstructure(un");
        } else {
            d.fac = l.split_whitespace().map(String::from).collect();
            d.xfac = xs;
            d.far = ar.map_or(Ok(vec![]), |a| ts_ml::parse_lags(&a))?;
        }
    }
    if d.deps.is_empty() {
        return Err(rc_err(198, "observation equations required"));
    }
    Ok(d)
}

pub struct DfactorCmd;

impl CommandHandler for DfactorCmd {
    fn name(&self) -> &'static str {
        "dfactor"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            return crate::models::replay(st, "dfactor");
        }
        let groups = crate::sspace::paren_groups(&cmd.main)?;
        let mut df = parse_dfac(ds, &groups)?;
        if df.fac.is_empty() && df.ear.is_empty() {
            return Err(rc_err(198, "the model needs factors or autoregressive errors"));
        }
        if df.fac.is_empty() {
            df.far.clear();
        }
        let ycols: Vec<Vec<f64>> = df.deps.iter().map(|v| tsops::values(ds, st, v)).collect::<Result<_, _>>()?;
        let colv = |n: &String| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) };
        let xo: Vec<Vec<f64>> = df.xobs.iter().map(colv).collect::<Result<_, _>>()?;
        let xf: Vec<Vec<f64>> = df.xfac.iter().map(colv).collect::<Result<_, _>>()?;
        let refs: Vec<&Vec<f64>> = ycols.iter().chain(xo.iter()).chain(xf.iter()).collect();
        let sd = crate::sspace::ss_rows(ds, st, cmd, &refs)?;
        let n = sd.rows.len();
        let xo_r: Vec<Vec<f64>> = sd.rows.iter().map(|&r| xo.iter().map(|c| c[r]).collect()).collect();
        let xf_r: Vec<Vec<f64>> = sd.rows.iter().map(|&r| xf.iter().map(|c| c[r]).collect()).collect();
        let y: Vec<Vec<Option<f64>>> = sd.rows.iter().map(|&r| ycols.iter().map(|c| Some(c[r])).collect()).collect();
        let llf = |th: &[f64]| -> Option<Vec<f64>> { kfilter(&df.ssm(th, &xo_r, &xf_r)?, &y).map(|o| o.ll) };
        let names = df.names();
        let k = df.k();
        let yv: Vec<f64> = (0..k).map(|i| util::variance(&y.iter().map(|r| r[i].unwrap()).collect::<Vec<_>>())).collect();
        let ym: Vec<f64> = (0..k).map(|i| util::mean(&y.iter().map(|r| r[i].unwrap()).collect::<Vec<_>>())).collect();
        let mk = |rho: f64, load: f64| -> Vec<f64> {
            let mut th = Vec::new();
            let nf = df.nf();
            for a in 0..nf {
                for (li, _) in df.far.iter().enumerate() {
                    for b in 0..nf {
                        th.push(if a == b && li == 0 { rho } else { 0.0 });
                    }
                }
                th.extend(df.xfac.iter().map(|_| 0.0));
            }
            for _ in 0..k {
                for (li, _) in df.ear.iter().enumerate() {
                    th.push(if li == 0 { rho } else { 0.0 });
                }
            }
            for d in 0..k {
                th.extend((0..nf).map(|_| load * yv[d].sqrt()));
                th.extend(df.xobs.iter().map(|x| if x == "_cons" { ym[d] } else { 0.0 }));
            }
            if df.unstr {
                for j in 0..k {
                    for i in j..k {
                        th.push(if i == j { yv[i] * 0.5 } else { 0.0 });
                    }
                }
            } else {
                th.extend(yv.iter().map(|v| v * 0.5));
            }
            th
        };
        let mut th0 = mk(0.3, 0.5);
        let mut best = ts_ml::ll_sum(&llf, &th0);
        for (r, l) in [(0.1, 0.3), (0.5, 0.5), (0.3, 0.8), (0.0, 0.5)] {
            let c = mk(r, l);
            let lv = ts_ml::ll_sum(&llf, &c);
            if lv > best {
                best = lv;
                th0 = c;
            }
        }
        if let Some(f) = ts_ml::from_opt(cmd, th0.len())? {
            th0 = f;
        }
        let maxiter = util::opt_num(cmd, "iterate", 4)?.unwrap_or(300.0) as usize;
        let opt = ts_ml::optimize(&llf, th0, maxiter, ts_ml::tech_from(cmd, ts_ml::Tech::Bfgs), "log likelihood", true)?;
        let mut th = opt.th.clone();
        // sign normalization: first loading of each factor positive
        let nf = df.nf();
        let load0 = names.iter().position(|nm| df.fac.first().is_some_and(|f| nm == &format!("{}:{f}", df.deps[0])));
        if let Some(p) = load0 {
            if th[p] < 0.0 && nf == 1 {
                for (j, nm) in names.iter().enumerate() {
                    if df.deps.iter().any(|d| nm == &format!("{d}:{}", df.fac[0])) || df.xfac.iter().any(|x| nm == &format!("{}:{x}", df.fac[0])) {
                        th[j] = -th[j];
                    }
                }
            }
        }
        let (v, vcetype, vce_name) = ts_ml::vce(&llf, &th, cmd, "oim")?;
        let test_idx: Vec<usize> = (0..names.len()).filter(|&j| !names[j].starts_with('/') && !names[j].ends_with(":_cons")).collect();
        let (chi2, dfm) = ts_ml::wald(&th, &v, &test_idx);
        let samp = ts_ml::TsSample { tvar: sd.tvar.clone(), y: vec![], x: vec![], xnames: vec![], times: sd.times.clone(), rows: sd.rows.clone(), mask: sd.mask.clone(), depvar: String::new(), n_gaps: tsops::gaps(&sd.times) };
        let (sline, tmins, tmaxs) = ts_ml::sample_text(ds, &samp);
        let header = ts_ml::std_header(&sline, n, chi2, dfm, opt.ll, "Log likelihood");
        let notes = vec!["Note: Tests of variances against zero are one sided, and the two-sided".to_string(), "      confidence intervals are truncated at zero.".to_string()];
        let text = ts_ml::post(
            st,
            cmd,
            Post { title: "Dynamic-factor model", cmd: "dfactor", depvar: "", names, b: th.clone(), v, ll: opt.ll, n, header, vcetype, vce: vce_name, log: opt.log, notes, converged: opt.converged, iters: opt.iters, df: None },
            Some(&sd.mask),
        )?;
        let e = &mut st.e;
        e.set_str("depvar", df.deps.join(" "));
        e.set_str("obser_deps", df.deps.join(" "));
        e.set_str("factors", df.fac.join(" "));
        e.set("k_factor", nf as f64);
        e.set("chi2", chi2);
        e.set("df_m", dfm);
        e.set("tmin", sd.times[0]);
        e.set("tmax", *sd.times.last().unwrap());
        e.set_str("tmins", tmins);
        e.set_str("tmaxs", tmaxs);
        e.set_str("predict", "dfactor_p");
        e.set_str("_df", groups.join("\u{1}"));
        Ok(text)
    }
}

/// `predict` after ucm or dfactor: xb (one-step prediction, default), residuals, states (filtered), equation().
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Result<String, String> {
    let e = &st.e;
    let ecmd = e.macros.get("cmd").cloned().unwrap_or_default();
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let newvar = ts_ml::predict_name(cmd)?;
    let mask = util::touse(ds, st, cmd)?;
    let tmin = e.get("tmin").unwrap_or(f64::NEG_INFINITY);
    let tvar = tsops::time_var(st)?;
    let tc = util::col(ds, &tvar)?;
    let (ssm, ycols, rows) = if ecmd == "ucm" {
        let dep = e.macros.get("depvar").cloned().unwrap_or_default();
        let xn: Vec<String> = e.macros.get("covariates").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
        let ncyc = e.get("k_cycles").unwrap_or(0.0) as usize;
        let u = Ucm { tr: trend_of(e.macros.get("trend").map(String::as_str).unwrap_or("rwalk"))?, season: e.get("seasonal").unwrap_or(0.0) as usize, ncyc, kx: xn.len() };
        let xc: Vec<Vec<f64>> = xn.iter().map(|v| tsops::values(ds, st, v)).collect::<Result<_, _>>()?;
        let refs: Vec<&Vec<f64>> = xc.iter().collect();
        let all = vec![true; ds.n_obs];
        let (r0, _) = tsops::ordered_rows(ds, st, &all, &refs)?;
        let rows: Vec<usize> = r0.into_iter().filter(|&r| tc[r] >= tmin).collect();
        let x: Vec<Vec<f64>> = rows.iter().map(|&r| xc.iter().map(|c| c[r]).collect()).collect();
        (u.ssm(&b.data, &x).ok_or_else(|| rc_err(498, "model could not be evaluated"))?, vec![tsops::values(ds, st, &dep)?], rows)
    } else {
        let groups: Vec<String> = e.macros.get("_df").map(|s| s.split('\u{1}').map(String::from).collect()).unwrap_or_default();
        let df = parse_dfac(ds, &groups)?;
        let colv = |n: &String| if n == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { tsops::values(ds, st, n) };
        let xo: Vec<Vec<f64>> = df.xobs.iter().map(colv).collect::<Result<_, _>>()?;
        let xf: Vec<Vec<f64>> = df.xfac.iter().map(colv).collect::<Result<_, _>>()?;
        let refs: Vec<&Vec<f64>> = xo.iter().chain(xf.iter()).collect();
        let all = vec![true; ds.n_obs];
        let (r0, _) = tsops::ordered_rows(ds, st, &all, &refs)?;
        let rows: Vec<usize> = r0.into_iter().filter(|&r| tc[r] >= tmin).collect();
        let xo_r: Vec<Vec<f64>> = rows.iter().map(|&r| xo.iter().map(|c| c[r]).collect()).collect();
        let xf_r: Vec<Vec<f64>> = rows.iter().map(|&r| xf.iter().map(|c| c[r]).collect()).collect();
        let yc: Vec<Vec<f64>> = df.deps.iter().map(|v| tsops::values(ds, st, v)).collect::<Result<_, _>>()?;
        (df.ssm(&b.data, &xo_r, &xf_r).ok_or_else(|| rc_err(498, "model could not be evaluated"))?, yc, rows)
    };
    let y: Vec<Vec<Option<f64>>> = rows.iter().map(|&r| ycols.iter().map(|c| if is_missing(c[r]) { None } else { Some(c[r]) }).collect()).collect();
    let kf = kfilter(&ssm, &y).ok_or_else(|| rc_err(498, "Kalman filter failed"))?;
    let eq = cmd.opt("equation", 2).and_then(|s| s.trim().trim_start_matches('#').parse::<usize>().ok()).map_or(0, |v| v.saturating_sub(1));
    let mut out = vec![SYSMISS; ds.n_obs];
    let stat = if cmd.has_opt("states", 5) { "states" } else if cmd.has_opt("residuals", 1) { "residuals" } else { "xb" };
    for (t, &r) in rows.iter().enumerate() {
        out[r] = match stat {
            "states" => kf.filt[t].get(eq).copied().unwrap_or(SYSMISS),
            "residuals" => y[t][eq.min(ycols.len() - 1)].map_or(SYSMISS, |v| v - kf.yhat[t][eq.min(ycols.len() - 1)]),
            _ => kf.yhat[t][eq.min(ycols.len() - 1)],
        };
    }
    ts_ml::add_prediction(ds, &newvar, out, &mask, "")
}
