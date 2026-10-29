//! Selection, hurdle and frontier models: `heckoprobit`, `heckpoisson`,
//! `churdle linear|exponential` and `frontier` (half-normal, exponential
//! and truncated-normal inefficiency; production or cost frontiers).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use statrs::function::gamma::ln_gamma;

use crate::choice_mixed::bvn;
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels::{self as xm, Header, Row};

pub fn handlers() -> xm::Handlers {
    vec![
        ("heckoprobit", Box::new(Heckoprobit)),
        ("heckpoisson", Box::new(Heckpoisson)),
        ("churdle", Box::new(Churdle)),
        ("frontier", Box::new(Frontier)),
    ]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    match ecmd {
        "frontier" => {
            let dist_ = st.e.macros.get("dist").cloned().unwrap_or_default();
            let s = if st.e.macros.get("function").is_some_and(|f| f == "cost") { -1.0 } else { 1.0 };
            Some(xm::generic_predict(ds, st, cmd, &["xb", "u", "m", "te", "stdp"], "(option xb assumed; fitted values)", &|stat, c| {
                if stat == "xb" {
                    return c.eta[0];
                }
                let Some(y) = c.y.first() else { return SYSMISS };
                let e = y - c.eta[0];
                let (mus, ss) = match dist_.as_str() {
                    "exponential" => {
                        let sv = ((c.aux)("lnsig2v") / 2.0).exp();
                        let su = ((c.aux)("lnsig2u") / 2.0).exp();
                        (-s * e - sv * sv / su, sv)
                    }
                    "tnormal" => {
                        let s2 = (c.aux)("lnsigma2").exp();
                        let g = dist::invlogit((c.aux)("lgtgamma"));
                        let (su2, sv2) = (g * s2, (1.0 - g) * s2);
                        let mu = (c.aux)("mu");
                        ((-s * e * su2 + mu * sv2) / s2, (su2 * sv2 / s2).sqrt())
                    }
                    _ => {
                        let sv2 = (c.aux)("lnsig2v").exp();
                        let su2 = (c.aux)("lnsig2u").exp();
                        let s2 = su2 + sv2;
                        (-s * e * su2 / s2, (su2 * sv2 / s2).sqrt())
                    }
                };
                let z = mus / ss;
                match stat {
                    "u" => mus + ss * dist::normalden(z) / dist::normal(z),
                    "m" => mus.max(0.0),
                    "te" => (-mus + 0.5 * ss * ss).exp() * dist::normal(z - ss) / dist::normal(z),
                    _ => SYSMISS,
                }
            }))
        }
        "heckpoisson" => Some(xm::generic_predict(ds, st, cmd, &["n", "ir", "xb", "xbsel", "psel", "stdp"], "(option n assumed; predicted number of events)", &|s, c| {
            let sig = (c.aux)("lnsigma").exp();
            match s {
                "n" => (c.eta[0] + 0.5 * sig * sig).exp(),
                "ir" => c.eta[0].exp(),
                "xb" => c.eta[0],
                "xbsel" => c.eta.get(1).copied().unwrap_or(SYSMISS),
                "psel" => c.eta.get(1).map_or(SYSMISS, |z| dist::normal(*z)),
                _ => SYSMISS,
            }
        })),
        "churdle" => {
            let model = st.e.macros.get("model").cloned().unwrap_or_default();
            let ll = st.e.get("ll_lim").unwrap_or(0.0);
            Some(xm::generic_predict(ds, st, cmd, &["ystar", "xb", "pr", "stdp"], "(option ystar assumed; conditional mean)", &|s, c| {
                let sig = c.eta.get(2).map_or(1.0, |z| z.exp());
                let p = c.eta.get(1).map_or(SYSMISS, |z| dist::normal(*z));
                match s {
                    "xb" => c.eta[0],
                    "pr" => p,
                    _ => {
                        if model == "exponential" {
                            p * (c.eta[0] + 0.5 * sig * sig).exp()
                        } else {
                            let z = (c.eta[0] - ll) / sig;
                            let m = c.eta[0] + sig * dist::normalden(z) / dist::normal(z);
                            ll * (1.0 - p) + p * m
                        }
                    }
                }
            }))
        }
        "heckoprobit" => Some(xm::generic_predict(ds, st, cmd, &["xb", "xbsel", "psel", "stdp"], "(option xb assumed; linear prediction)", &|s, c| match s {
            "xb" => c.eta[0],
            "xbsel" => c.eta.get(1).copied().unwrap_or(SYSMISS),
            "psel" => c.eta.get(1).map_or(SYSMISS, |z| dist::normal(*z)),
            _ => SYSMISS,
        })),
        _ => None,
    }
}

/// `select(depvar = varlist[, noconstant])` or `select(varlist)`.
fn select_opt(cmd: &Command) -> Result<(Option<String>, String, bool), String> {
    let s = cmd.opt("select", 3).ok_or_else(|| rc_err(198, "option select() required"))?;
    let (body, sub) = match s.split_once(',') {
        Some((a, b)) => (a.to_string(), b.to_lowercase()),
        None => (s.to_string(), String::new()),
    };
    let nocons = sub.contains("nocons");
    Ok(match body.split_once('=') {
        Some((a, b)) => (Some(a.trim().to_string()), b.trim().to_string(), nocons),
        None => (None, body.trim().to_string(), nocons),
    })
}

/// Selection-model data: outcome design over the full sample (outcome
/// missing for unselected rows), selection design, selection indicator.
struct SelData {
    d: Design,
    dz: Design,
    s: Vec<bool>,
    y: Vec<f64>,
    dep: String,
    sdep: String,
}

fn sel_data(ds: &DataSet, st: &Store, cmd: &Command, main: &str, force_nocons: bool) -> Result<SelData, String> {
    let (dep, xs) = xm::first_word(main);
    let dep = xm::var_name(ds, &dep)?;
    let (sdep, zs, znocons) = select_opt(cmd)?;
    let ycol = util::col(ds, &dep)?;
    // outcome regressors must be observed; the outcome itself only when selected
    let dz = est::build(ds, st, cmd, est::DesignSpec { depvar: sdep.as_deref(), indep: &zs, noconstant: znocons, weights: "fpi", extra: crate::zip::base_vars(ds, &xs) })?;
    let mut d = est::build(ds, st, cmd, est::DesignSpec { depvar: None, indep: &xs, noconstant: force_nocons || cmd.has_opt("noconstant", 3), weights: "fpi", extra: crate::zip::base_vars(ds, &zs) })?;
    let s: Vec<bool> = match &sdep {
        Some(_) => dz.y.iter().map(|v| *v != 0.0).collect(),
        None => dz.rows.iter().map(|&i| !is_missing(ycol[i])).collect(),
    };
    // align (both designs share the same sample)
    if d.rows != dz.rows {
        return Err(rc_err(459, "selection and outcome samples differ"));
    }
    let y: Vec<f64> = d.rows.iter().map(|&i| ycol[i]).collect();
    for (t, sel) in s.iter().enumerate() {
        if *sel && is_missing(y[t]) {
            return Err(rc_err(459, format!("{dep} missing for selected observations")));
        }
    }
    d.y = y.iter().map(|v| if is_missing(*v) { 0.0 } else { *v }).collect();
    d.depvar = dep.clone();
    let sdep = sdep.map(|s| crate::varlist::resolve_name(ds, &s).unwrap_or(s)).unwrap_or_else(|| "select".into());
    Ok(SelData { y: d.y.clone(), d, dz, s, dep, sdep })
}

fn sel_header(title: &str, n: usize, n1: usize, extra: Option<&str>, chi2: f64, df: f64, ll: f64, robust: bool) -> Vec<String> {
    let mut out = vec![crate::choice_mixed::hrow(title, "Number of obs", util::commas(n))];
    out.push(format!("{:<54}{:<11} = {:>10}", extra.unwrap_or(""), "Selected", util::commas(n1)));
    out.push(format!("{:<54}{:<11} = {:>10}", "", "Nonselected", util::commas(n - n1)));
    out.push(String::new());
    out.push(crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
    out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9)), "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
    out.push(String::new());
    out
}

// ---------------------------------------------------------------------------
// heckoprobit
// ---------------------------------------------------------------------------

pub struct Heckoprobit;

impl CommandHandler for Heckoprobit {
    fn name(&self) -> &'static str {
        "heckoprobit"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ds_ = &*ds;
        let sd = sel_data(ds_, st, cmd, &cmd.main, true)?;
        let d = &sd.d;
        let dz = &sd.dz;
        let n = d.n();
        let mut cats: Vec<f64> = (0..n).filter(|&t| sd.s[t]).map(|t| sd.y[t]).collect();
        cats.sort_by(f64::total_cmp);
        cats.dedup();
        let nc = cats.len();
        let kidx: Vec<usize> = (0..n).map(|t| if sd.s[t] { cats.iter().position(|c| *c == sd.y[t]).unwrap() } else { 0 }).collect();
        let s = sd.s.clone();
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let zcols: Vec<&[f64]> = dz.active().iter().map(|&j| dz.cols[j].as_slice()).collect();
        let sv: Vec<f64> = s.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
        let pr = crate::heckman::probit_fit(&zcols, &sv)?;
        // ordered probit start on the selected sample
        let n1 = s.iter().filter(|b| **b).count();
        let mut cum = 0.0;
        let mut cuts = Vec::new();
        for c in 0..nc - 1 {
            cum += (0..n).filter(|&t| s[t] && kidx[t] == c).count() as f64;
            cuts.push(dist::invnormal(cum / n1 as f64));
        }
        let kx = d.active().len();
        // analytic gradient: d Phi2(h, k, r)/dh = phi(h) Phi((k - r h)/sqrt(1 - r^2)),
        // d/dr = bivariate normal density
        let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>) {
            let np = v.len();
            let mut g = vec![0.0; np];
            let (xb, zg) = (v[0], v[1]);
            if !s[t] {
                let (l, m) = crate::choice_mixed::ln_phi_d(-zg);
                g[1] = -m;
                return (l, g);
            }
            let r = -v[2].tanh();
            let dr = -(1.0 - r * r);
            let sr = (1.0 - r * r).sqrt();
            let c = &v[3..];
            let k = kidx[t];
            let pdf2 = |h: f64, q: f64| (-(h * h - 2.0 * r * h * q + q * q) / (2.0 * (1.0 - r * r))).exp() / (2.0 * std::f64::consts::PI * sr);
            // pieces: (value, d/dh, d/dk, d/dr) of Phi2(h, zg, r)
            let piece = |h: f64| -> (f64, f64, f64, f64) {
                if h == f64::INFINITY {
                    return (dist::normal(zg), 0.0, dist::normalden(zg), 0.0);
                }
                (bvn(h, zg, r), dist::normalden(h) * dist::normal((zg - r * h) / sr), dist::normalden(zg) * dist::normal((h - r * zg) / sr), pdf2(h, zg))
            };
            let (hv, hh, hk, hr) = if k + 1 < nc { piece(c[k] - xb) } else { piece(f64::INFINITY) };
            let (lv, lh, lk, lr) = if k > 0 { piece(c[k - 1] - xb) } else { (0.0, 0.0, 0.0, 0.0) };
            let p = (hv - lv).max(1e-300);
            g[0] = -(hh - lh) / p;
            g[1] = (hk - lk) / p;
            g[2] = (hr - lr) * dr / p;
            if k + 1 < nc {
                g[3 + k] += hh / p;
            }
            if k > 0 {
                g[3 + k - 1] -= lh / p;
            }
            (p.ln(), g)
        };
        let g = xm::with_grad(&f);
        let mut b0 = vec![0.0; kx];
        b0.extend(pr.b.iter().cloned());
        b0.push(0.0);
        b0.extend(cuts);
        let mut aux = vec!["athrho".to_string()];
        aux.extend((1..nc).map(|i| format!("cut{i}")));
        let auxr: Vec<&str> = aux.iter().map(|x| x.as_str()).collect();
        let fit = xm::ml_fit(&[(sd.dep.as_str(), d), (sd.sdep.as_str(), dz)], &auxr, &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        out.extend(sel_header("Ordered probit model with sample selection", n, n1, None, chi2, df, fit.ll(), robust));
        let level = util::level(cmd, st)?;
        let ja = fit.idx("/:athrho").unwrap();
        let (a, se) = (fit.b[ja], fit.v[(ja, ja)].max(0.0).sqrt());
        let r = a.tanh();
        let crit = crate::choice_mixed::zcrit(level);
        let mut extra = Vec::new();
        for j in ja + 1..fit.names.len() {
            let (bj, sj) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
            extra.push(Row::Ci(format!("/{}", &fit.names[j][2..]), bj, sj, bj - crit * sj, bj + crit * sj));
        }
        extra.push(Row::Sep);
        extra.push(Row::Tr("rho".into(), r, se * (1.0 - r * r), a, se, f64::tanh));
        // athrho row with test then cutpoints
        let mut fit2 = clone_fit(&fit);
        fit2.names.truncate(ja + 1);
        let mut tbl = xm::table("", &fit2, level, None, &[]);
        let bar = tbl.pop().unwrap();
        let lw = bar.len().saturating_sub(66);
        tbl.extend(xm::extra_rows(lw, level, &extra, false));
        tbl.push(bar);
        out.extend(tbl);
        let wr = (a / se).powi(2);
        out.push(format!("Wald test of indep. eqns. (rho = 0): chi2(1) = {:>8}   Prob > chi2 = {}", xm::f2(wr), xm::f4(xm::chi2p(1.0, wr))));
        let text = out.join("\n");
        let catm = Matrix::row_vector(&cats);
        let (dep, sdep) = (sd.dep.clone(), sd.sdep.clone());
        xm::post_fit(st, "heckoprobit", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("N_selected", n1 as f64);
            e.set("N_nonselected", (n - n1) as f64);
            e.set("rho", r);
            e.set("chi2_c", wr);
            e.set("k_cat", nc as f64);
            e.set_mat("cat", catm.clone());
            e.set_str("depvar", format!("{dep} {sdep}"));
            e.set_str("title", "Ordered probit model with sample selection");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

fn clone_fit(f: &xm::Fit) -> xm::Fit {
    xm::Fit {
        names: f.names.clone(),
        b: f.b.clone(),
        v: f.v.clone(),
        omitted: f.omitted.clone(),
        is_base: f.is_base.clone(),
        res: est::MlResult { b: f.res.b.clone(), ll: f.res.ll, info: f.res.info.clone(), scores: vec![], iterations: f.res.iterations.clone(), converged: f.res.converged },
        bt: f.bt.clone(),
        vt: f.vt.clone(),
        pos: f.pos.clone(),
        n_clust: f.n_clust,
        neq: f.neq,
        naux: f.naux,
    }
}

// ---------------------------------------------------------------------------
// heckpoisson
// ---------------------------------------------------------------------------

thread_local! {
    static GH25: (Vec<f64>, Vec<f64>) = gh(25);
}

/// Gauss-Hermite nodes and weights against the standard normal density.
fn gh(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = Matrix::zeros(n, n);
    for i in 1..n {
        let b = (i as f64).sqrt();
        j[(i, i - 1)] = b;
        j[(i - 1, i)] = b;
    }
    let (vals, vecs) = j.sym_eigen();
    let mut p: Vec<(f64, f64)> = (0..n).map(|k| (vals[k], vecs[(0, k)] * vecs[(0, k)])).collect();
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    (p.iter().map(|x| x.0).collect(), p.iter().map(|x| x.1).collect())
}

pub struct Heckpoisson;

impl CommandHandler for Heckpoisson {
    fn name(&self) -> &'static str {
        "heckpoisson"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let ds_ = &*ds;
        let sd = sel_data(ds_, st, cmd, &cmd.main, false)?;
        let (d, dz) = (&sd.d, &sd.dz);
        let n = d.n();
        let s = sd.s.clone();
        let y = sd.y.clone();
        let n1 = s.iter().filter(|b| **b).count();
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let zcols: Vec<&[f64]> = dz.active().iter().map(|&j| dz.cols[j].as_slice()).collect();
        let sv: Vec<f64> = s.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
        let pr = crate::heckman::probit_fit(&zcols, &sv)?;
        let f = |t: usize, v: &[f64]| -> f64 {
            let (xb, zg) = (v[0], v[1]);
            if !s[t] {
                return crate::xtnl::ln_phi(-zg);
            }
            let r = v[2].tanh();
            let sig = v[3].exp();
            let sr = (1.0 - r * r).sqrt();
            GH25.with(|(x, w)| {
                let mut acc = 0.0;
                let mut mx = f64::NEG_INFINITY;
                let terms: Vec<f64> = x
                    .iter()
                    .zip(w)
                    .map(|(e, wt)| {
                        let eta = xb + sig * e;
                        let lp = y[t] * eta - eta.exp() - ln_gamma(y[t] + 1.0);
                        let l = wt.ln() + lp + crate::xtnl::ln_phi((zg + r * e) / sr);
                        mx = mx.max(l);
                        l
                    })
                    .collect();
                for l in terms {
                    acc += (l - mx).exp();
                }
                mx + acc.ln()
            })
        };
        let g = xm::numeric(&f);
        // start: Poisson on the selected observations
        let kx = d.active().len();
        let ly: Vec<f64> = y.iter().map(|v| (v + 0.5).ln()).collect();
        let (bx, _) = xm::ols_start(d, &ly);
        let mut b0 = bx;
        b0.extend(pr.b.iter().cloned());
        b0.push(0.0);
        b0.push(-0.7);
        let fit = xm::ml_fit(&[(sd.dep.as_str(), d), (sd.sdep.as_str(), dz)], &["athrho", "lnsigma"], &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        let _ = kx;
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        out.extend(sel_header("Poisson regression with endogenous selection", n, n1, Some("(25 quadrature points)"), chi2, df, fit.ll(), robust));
        let level = util::level(cmd, st)?;
        let ja = fit.idx("/:athrho").unwrap();
        let jl = fit.idx("/:lnsigma").unwrap();
        let (a, sa) = (fit.b[ja], fit.v[(ja, ja)].max(0.0).sqrt());
        let (l, sl) = (fit.b[jl], fit.v[(jl, jl)].max(0.0).sqrt());
        let extra = vec![Row::Tr("rho".into(), a.tanh(), sa * (1.0 - a.tanh().powi(2)), a, sa, f64::tanh), Row::Tr("sigma".into(), l.exp(), sl * l.exp(), l, sl, f64::exp)];
        out.extend(xm::table("", &fit, level, None, &extra));
        let wr = (a / sa).powi(2);
        out.push(format!("Wald test of indep. eqns. (rho = 0): chi2(1) = {:>8}   Prob > chi2 = {}", xm::f2(wr), xm::f4(xm::chi2p(1.0, wr))));
        let text = out.join("\n");
        let (dep, sdep) = (sd.dep.clone(), sd.sdep.clone());
        xm::post_fit(st, "heckpoisson", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("N_selected", n1 as f64);
            e.set("N_nonselected", (n - n1) as f64);
            e.set("rho", a.tanh());
            e.set("sigma", l.exp());
            e.set("chi2_c", wr);
            e.set("n_quad", 25.0);
            e.set_str("depvar", format!("{dep} {sdep}"));
            e.set_str("title", "Poisson regression with endogenous selection");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// churdle
// ---------------------------------------------------------------------------

pub struct Churdle;

impl CommandHandler for Churdle {
    fn name(&self) -> &'static str {
        "churdle"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (model, rest) = xm::first_word(&cmd.main);
        let model = model.to_lowercase();
        let expo = match model.as_str() {
            "linear" => false,
            "exponential" => true,
            _ => return Err(rc_err(198, "churdle: model must be linear or exponential")),
        };
        let (dep, xs) = xm::first_word(&rest);
        let dep = xm::var_name(ds, &dep)?;
        let s = cmd.opt("select", 3).ok_or_else(|| rc_err(198, "option select() required"))?.to_string();
        let (zs, zsub) = match s.split_once(',') {
            Some((a, b)) => (a.trim().to_string(), b.to_string()),
            None => (s.clone(), String::new()),
        };
        let zhet = zsub.find("het(").map(|i| {
            let r = &zsub[i + 4..];
            r[..r.find(')').unwrap_or(r.len())].to_string()
        });
        let het = cmd.opt("het", 3).map(|s| s.to_string());
        let ll_lim = util::opt_num(cmd, "ll", 2)?.ok_or_else(|| rc_err(198, "option ll() required"))?;
        let ds_ = &*ds;
        let mut specs: Vec<(Option<&str>, &str, bool)> = vec![(Some(dep.as_str()), xs.as_str(), cmd.has_opt("noconstant", 3)), (None, zs.as_str(), false), (None, het.as_deref().unwrap_or(""), false)];
        if let Some(h) = &zhet {
            specs.push((None, h.as_str(), true));
        }
        let des = xm::build_eqs(ds_, st, cmd, &specs, "fpi", &[])?;
        let d = &des[0];
        let y = d.y.clone();
        if y.iter().any(|v| *v < ll_lim) {
            return Err(rc_err(459, format!("{dep} has values below the lower limit")));
        }
        if expo && ll_lim != 0.0 {
            return Err(rc_err(198, "exponential model requires ll(0)"));
        }
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let hz = zhet.is_some();
        let f = |t: usize, v: &[f64]| -> f64 {
            let (xb, zg, lns) = (v[0], v[1], v[2]);
            let zs = if hz { zg / v[3].exp() } else { zg };
            if y[t] <= ll_lim {
                return crate::xtnl::ln_phi(-zs);
            }
            let sig = lns.exp();
            let p = crate::xtnl::ln_phi(zs);
            if expo {
                p + dist::lnnormalden((y[t].ln() - xb) / sig) - lns
            } else {
                p + dist::lnnormalden((y[t] - xb) / sig) - lns - crate::xtnl::ln_phi((xb - ll_lim) / sig)
            }
        };
        let g = xm::numeric(&f);
        let pos: Vec<f64> = y.iter().map(|v| if *v > ll_lim { 1.0 } else { 0.0 }).collect();
        let zcols: Vec<&[f64]> = des[1].active().iter().map(|&j| des[1].cols[j].as_slice()).collect();
        let pr = crate::heckman::probit_fit(&zcols, &pos)?;
        let posd = xm::subset(d, &pos.iter().map(|v| *v > 0.0).collect::<Vec<_>>());
        let yy: Vec<f64> = posd.y.iter().map(|v| if expo { v.ln() } else { *v }).collect();
        let (bx, sx) = xm::ols_start(&posd, &yy);
        let mut b0 = bx;
        b0.extend(pr.b.iter().cloned());
        let hact = des[2].active();
        let mut gh = vec![0.0; hact.len()];
        if let Some(c) = hact.iter().position(|&j| des[2].names[j] == "_cons") {
            gh[c] = sx.max(1e-6).ln();
        }
        b0.extend(gh.iter().cloned());
        if hz {
            b0.extend(vec![0.0; des[3].active().len()]);
        }
        let sel_name = "selection_ll";
        let mut eqs: Vec<(&str, &Design)> = vec![(dep.as_str(), d), (sel_name, &des[1]), ("lnsigma", &des[2])];
        if hz {
            eqs.push(("lnsigma_ll", &des[3]));
        }
        let fw = d.wkind == Some(WeightKind::F);
        let fit = xm::ml_fit(&eqs, &[], &g, b0, &d.w, fw, &vce, cluster.as_deref())?;
        // constant-only model (all equations with constants only)
        let mut ll0 = SYSMISS;
        if let (Some(c0), Some(c1)) = (xm::const_only(d), xm::const_only(&des[1])) {
            let c2 = xm::const_only(&des[2]).unwrap_or_else(|| des[2].clone());
            let mut e0: Vec<(&str, &Design)> = vec![(dep.as_str(), &c0), (sel_name, &c1), ("lnsigma", &c2)];
            let c3 = if hz { Some(des[3].clone()) } else { None };
            let mut b00 = vec![util::mean(&yy), dist::invnormal(util::mean(&pos).clamp(1e-6, 1.0 - 1e-6)), sx.max(1e-6).ln()];
            if let Some(c3) = &c3 {
                let mut cc = c3.clone();
                for o in cc.omitted.iter_mut() {
                    *o = true;
                }
                let _ = &cc;
                e0.push(("lnsigma_ll", c3));
                b00.extend(vec![0.0; c3.active().len()]);
            }
            if let Ok(f0) = xm::ml_fit(&e0, &[], &g, b00, &d.w, fw, &Vce::Oim, None) {
                ll0 = f0.ll();
            }
        }
        let robust = vce != Vce::Oim;
        let (wchi, df) = fit.wald_eq(0);
        let (tn, chi2) = if robust || is_missing(ll0) { ("Wald", wchi) } else { ("LR", 2.0 * (fit.ll() - ll0)) };
        let r2p = if is_missing(ll0) { SYSMISS } else { 1.0 - fit.ll() / ll0 };
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(crate::choice_mixed::hrow("Cragg hurdle regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push(crate::choice_mixed::hrow("", &format!("{tn} chi2({})", fmt_g(df, 9)), xm::f2(chi2)));
        out.push(crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))));
        out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(fit.ll(), 9)), "Pseudo R2", xm::f4(r2p)));
        out.push(String::new());
        let level = util::level(cmd, st)?;
        let mut extra = Vec::new();
        if het.is_none() {
            let j = fit.idx("lnsigma:_cons").unwrap();
            let (l, sl) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
            extra.push(Row::Tr("/sigma".into(), l.exp(), sl * l.exp(), l, sl, f64::exp));
        }
        out.extend(xm::table(&dep, &fit, level, None, &extra));
        let text = out.join("\n");
        xm::post_fit(st, "churdle", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, tn, chi2, df);
            e.set("ll_0", ll0);
            e.set("r2_p", r2p);
            e.set("ll_lim", ll_lim);
            e.set_str("model", model.clone());
            e.set_str("title", "Cragg hurdle regression");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// frontier
// ---------------------------------------------------------------------------

pub struct Frontier;

fn frontier_ll(dist_: &str, s: f64, e: f64, a: &[f64]) -> f64 {
    match dist_ {
        "exponential" => {
            let sv = (a[0] / 2.0).exp();
            let su = (a[1] / 2.0).exp();
            -su.ln() + crate::xtnl::ln_phi(-s * e / sv - sv / su) + s * e / su + sv * sv / (2.0 * su * su)
        }
        "tnormal" => {
            let mu = a[0];
            let s2 = a[1].exp();
            let g = dist::invlogit(a[2]);
            let (su2, sv2) = (g * s2, (1.0 - g) * s2);
            let sig = s2.sqrt();
            let mus = (mu * sv2 - s * e * su2) / s2;
            let ss = (su2 * sv2 / s2).sqrt();
            -sig.ln() + dist::lnnormalden((s * e + mu) / sig) + crate::xtnl::ln_phi(mus / ss) - crate::xtnl::ln_phi(mu / su2.sqrt())
        }
        _ => {
            let sv2 = a[0].exp();
            let su2 = a[1].exp();
            let sig = (su2 + sv2).sqrt();
            let lam = (su2 / sv2).sqrt();
            (2.0f64).ln() - sig.ln() + dist::lnnormalden(e / sig) + crate::xtnl::ln_phi(-s * lam * e / sig)
        }
    }
}

impl CommandHandler for Frontier {
    fn name(&self) -> &'static str {
        "frontier"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, xs) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let dist_ = match cmd.opt("distribution", 1).map(|s| s.trim().to_lowercase()) {
            None => "hnormal".to_string(),
            Some(s) if "hnormal".starts_with(&s) && !s.is_empty() => "hnormal".into(),
            Some(s) if s.len() >= 1 && "exponential".starts_with(&s) => "exponential".into(),
            Some(s) if s.len() >= 1 && "tnormal".starts_with(&s) => "tnormal".into(),
            Some(s) => return Err(rc_err(198, format!("distribution({s}) not allowed"))),
        };
        let cost = cmd.has_opt("cost", 4);
        let sgn = if cost { -1.0 } else { 1.0 };
        let d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&dep), indep: &xs, noconstant: cmd.has_opt("noconstant", 3), weights: "fpi", extra: vec![] })?;
        let vce = xm::ml_vce_opt(cmd, &d)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let ols = est::ols(&d, &Vce::Ols, None)?;
        let n = d.n() as f64;
        let y = d.y.clone();
        // moment-based starting values from OLS residual moments
        let m2 = ols.resid.iter().map(|e| e * e).sum::<f64>() / n;
        let m3 = ols.resid.iter().map(|e| e.powi(3)).sum::<f64>() / n;
        let su_mom = ((-sgn * m3) / ((2.0 / std::f64::consts::PI).sqrt() * (1.0 - 4.0 / std::f64::consts::PI))).max(1e-6).cbrt();
        let su2 = (su_mom * su_mom).min(0.9 * m2).max(0.05 * m2);
        let sv2 = (m2 - (1.0 - 2.0 / std::f64::consts::PI) * su2).max(0.05 * m2);
        let mut b0: Vec<f64> = d.active().iter().map(|&j| ols.b[j]).collect();
        if let Some(c) = d.active().iter().position(|&j| d.names[j] == "_cons") {
            b0[c] += sgn * (2.0 / std::f64::consts::PI).sqrt() * su2.sqrt();
        }
        let aux: Vec<&str> = match dist_.as_str() {
            "tnormal" => {
                b0.push(0.0);
                b0.push((su2 + sv2).ln());
                b0.push((su2 / sv2).ln());
                vec!["mu", "lnsigma2", "lgtgamma"]
            }
            "exponential" => {
                b0.push(sv2.ln());
                b0.push((su2 / 2.0).max(1e-8).ln());
                vec!["lnsig2v", "lnsig2u"]
            }
            _ => {
                b0.push(sv2.ln());
                b0.push(su2.ln());
                vec!["lnsig2v", "lnsig2u"]
            }
        };
        let dd = dist_.clone();
        let f = move |t: usize, v: &[f64]| frontier_ll(&dd, sgn, y[t] - v[0], &v[1..]);
        let g = xm::numeric(&f);
        let fit = xm::ml_fit(&[("", &d)], &aux, &g, b0, &d.w, d.wkind == Some(WeightKind::F), &vce, cluster.as_deref())?;
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        let title = match dist_.as_str() {
            "exponential" => "Stoc. frontier normal/exponential model",
            "tnormal" => "Stoc. frontier normal/truncated-normal model",
            _ => "Stoc. frontier normal/half-normal model",
        };
        let mut out = est::iteration_log(&fit.res.iterations, "Log likelihood");
        out.push(String::new());
        out.extend(Header { title, n: d.n_eff(), lines: vec![], test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: fit.n_clust.map(|g| (g, if let Vce::Cluster(c) = &vce { c.clone() } else { String::new() })) }.render());
        let level = util::level(cmd, st)?;
        let crit = crate::choice_mixed::zcrit(level);
        let k = fit.names.len();
        let a0 = k - aux.len();
        let vsub = |i: usize, j: usize| fit.v[(a0 + i, a0 + j)];
        let mut extra = Vec::new();
        let scal: Vec<(&str, f64)>;
        if dist_ == "tnormal" {
            let (ls2, lg) = (fit.b[a0 + 1], fit.b[a0 + 2]);
            let s2 = ls2.exp();
            let gm = dist::invlogit(lg);
            let se_ls2 = vsub(1, 1).max(0.0).sqrt();
            let se_lg = vsub(2, 2).max(0.0).sqrt();
            extra.push(Row::Tr("sigma2".into(), s2, s2 * se_ls2, ls2, se_ls2, f64::exp));
            extra.push(Row::Tr("gamma".into(), gm, gm * (1.0 - gm) * se_lg, lg, se_lg, dist::invlogit));
            // sigma_u2 = gamma sigma2, sigma_v2 = (1 - gamma) sigma2 (delta method)
            let gu = [gm * s2, gm * (1.0 - gm) * s2];
            let gv = [(1.0 - gm) * s2, -gm * (1.0 - gm) * s2];
            let var = |g: &[f64; 2]| g[0] * g[0] * vsub(1, 1) + 2.0 * g[0] * g[1] * vsub(1, 2) + g[1] * g[1] * vsub(2, 2);
            let (su2, sv2) = (gm * s2, (1.0 - gm) * s2);
            let (seu, sev) = (var(&gu).sqrt(), var(&gv).sqrt());
            extra.push(Row::Ci("sigma_u2".into(), su2, seu, su2 - crit * seu, su2 + crit * seu));
            extra.push(Row::Ci("sigma_v2".into(), sv2, sev, sv2 - crit * sev, sv2 + crit * sev));
            scal = vec![("sigma2", s2), ("gamma", gm), ("sigma_u", su2.sqrt()), ("sigma_v", sv2.sqrt()), ("mu", fit.b[a0])];
        } else {
            let (lv, lu) = (fit.b[a0], fit.b[a0 + 1]);
            let (sev, seu) = (vsub(0, 0).max(0.0).sqrt(), vsub(1, 1).max(0.0).sqrt());
            let sv = (lv / 2.0).exp();
            let su = (lu / 2.0).exp();
            let half = |x: f64| (x / 2.0).exp();
            extra.push(Row::Tr("sigma_v".into(), sv, sv * sev / 2.0, lv, sev, half));
            extra.push(Row::Tr("sigma_u".into(), su, su * seu / 2.0, lu, seu, half));
            let s2 = sv * sv + su * su;
            let ses2 = (sv.powi(4) * vsub(0, 0) + 2.0 * sv * sv * su * su * vsub(0, 1) + su.powi(4) * vsub(1, 1)).max(0.0).sqrt();
            extra.push(Row::Ci("sigma2".into(), s2, ses2, s2 - crit * ses2, s2 + crit * ses2));
            let lam = su / sv;
            let gl = [-lam / 2.0, lam / 2.0];
            let sel = (gl[0] * gl[0] * vsub(0, 0) + 2.0 * gl[0] * gl[1] * vsub(0, 1) + gl[1] * gl[1] * vsub(1, 1)).max(0.0).sqrt();
            extra.push(Row::Ci("lambda".into(), lam, sel, lam - crit * sel, lam + crit * sel));
            scal = vec![("sigma_u", su), ("sigma_v", sv), ("sigma2", s2), ("lambda", lam)];
        }
        out.extend(xm::table(&dep, &fit, level, None, &extra));
        // test of no inefficiency
        let mut z_ineff = SYSMISS;
        let mut chibar = SYSMISS;
        if dist_ == "tnormal" {
            // third-moment test on OLS residuals
            let z = -sgn * m3 / (6.0 * m2.powi(3) / n).sqrt();
            z_ineff = z;
            out.push(format!("H0: No inefficiency component: z = {:.3}   Prob >= z = {:.4}", z, dist::normal(-z)));
        } else {
            let ll_ols = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + (ols.rss / n).ln() + 1.0);
            chibar = (2.0 * (fit.ll() - ll_ols)).max(0.0);
            let p = 0.5 * xm::chi2p(1.0, chibar);
            out.push(format!("LR test of sigma_u=0: chibar2(01) = {}   Prob >= chibar2 = {:.3}", xm::f2(chibar), if chibar == 0.0 { 1.0 } else { p }));
        }
        let text = out.join("\n");
        xm::post_fit(st, "frontier", &d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            for (k, v) in &scal {
                e.set(k, *v);
            }
            e.set("chi2_c", chibar);
            e.set("z", z_ineff);
            e.set_str("dist", dist_.clone());
            e.set_str("function", if cost { "cost" } else { "production" });
            e.set_str("title", title);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}
