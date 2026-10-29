//! Linear-model variants: interval regression (`intreg`), constrained
//! least squares (`cnsreg`), errors-in-variables regression (`eivreg`),
//! heteroskedastic linear regression (`hetregress`) and systems of
//! equations (`sureg`, `reg3`).

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util::{self, WeightKind};
use crate::xmodels::{self as xm, Header, Row};

pub fn handlers() -> xm::Handlers {
    vec![
        ("intreg", Box::new(Intreg)),
        ("cnsreg", Box::new(Cnsreg)),
        ("eivreg", Box::new(Eivreg)),
        ("hetregress", Box::new(Hetregress)),
        ("sureg", Box::new(System { three: false })),
        ("reg3", Box::new(System { three: true })),
    ]
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command, ecmd: &str) -> Option<Result<String, String>> {
    match ecmd {
        "intreg" => {
            let nd = |z: f64| if z.is_finite() { dist::normalden(z) } else { 0.0 };
            Some(xm::generic_predict(ds, st, cmd, &["xb", "stdp", "e", "ystar", "pr"], "(option xb assumed; fitted values)", &|s, c| {
                let sigma = if c.eta.len() > 1 { c.eta[1].exp() } else { (c.aux)("lnsigma").exp() };
                match s {
                    "xb" => c.eta[0],
                    _ => {
                        let (a, b) = pr_bounds(cmd);
                        let xb = c.eta[0];
                        let za = (a - xb) / sigma;
                        let zb = (b - xb) / sigma;
                        let p = xm::ncdf(zb) - xm::ncdf(za);
                        match s {
                            "pr" => p,
                            "e" => xb - sigma * (nd(zb) - nd(za)) / p,
                            _ => {
                                let ex = xb - sigma * (nd(zb) - nd(za)) / p;
                                (if a.is_finite() { xm::ncdf(za) * a } else { 0.0 }) + p * ex + (if b.is_finite() { (1.0 - xm::ncdf(zb)) * b } else { 0.0 })
                            }
                        }
                    }
                }
            }))
        }
        "hetregress" => Some(xm::generic_predict(ds, st, cmd, &["xb", "stdp", "sigma", "residuals"], "(option xb assumed; fitted values)", &|s, c| match s {
            "xb" => c.eta[0],
            "sigma" => c.eta.get(1).map_or(SYSMISS, |z| (z / 2.0).exp()),
            "residuals" => c.y.first().map_or(SYSMISS, |y| y - c.eta[0]),
            _ => SYSMISS,
        })),
        "sureg" | "reg3" => {
            let eq = cmd.opt("equation", 2).map(|s| s.trim().trim_start_matches('#').to_string());
            let b = st.e.matrices.get("b").cloned()?;
            let eqs = xm::eq_names(&b);
            let k = match &eq {
                Some(e) => e.parse::<usize>().ok().map(|n| n.saturating_sub(1)).or_else(|| eqs.iter().position(|x| x.as_deref() == Some(e.as_str()))).unwrap_or(0),
                None => 0,
            };
            let deps: Vec<String> = st.e.macros.get("depvar").map(|d| d.split_whitespace().map(String::from).collect()).unwrap_or_default();
            let dcol = deps.get(k).and_then(|d| util::col(ds, d).ok());
            Some(xm::generic_predict(ds, st, cmd, &["xb", "stdp", "residuals"], "(option xb assumed; fitted values)", &|s, c| match s {
                "xb" => c.eta.get(k).copied().unwrap_or(SYSMISS),
                "residuals" => match &dcol {
                    Some(y) if !is_missing(y[c.row]) => y[c.row] - c.eta.get(k).copied().unwrap_or(SYSMISS),
                    _ => SYSMISS,
                },
                _ => SYSMISS,
            }))
        }
        _ => None,
    }
}

/// `pr(a, b)` / `e(a, b)` / `ystar(a, b)` bounds of a predict command.
pub fn pr_bounds(cmd: &Command) -> (f64, f64) {
    for (n, a) in &cmd.opts {
        let n = n.to_lowercase();
        if matches!(n.as_str(), "pr" | "e" | "ystar") {
            if let Some(a) = a {
                let parts: Vec<&str> = a.split(',').collect();
                let p = |s: &str| -> f64 {
                    let s = s.trim();
                    if s == "." || s.is_empty() {
                        f64::NAN
                    } else {
                        s.parse().unwrap_or(f64::NAN)
                    }
                };
                let lo = parts.first().map_or(f64::NAN, |s| p(s));
                let hi = parts.get(1).map_or(f64::NAN, |s| p(s));
                return (if lo.is_nan() { f64::NEG_INFINITY } else { lo }, if hi.is_nan() { f64::INFINITY } else { hi });
            }
        }
    }
    (f64::NEG_INFINITY, f64::INFINITY)
}

// ---------------------------------------------------------------------------
// intreg
// ---------------------------------------------------------------------------

/// Interval-regression contribution for bounds (lo, hi) (NaN = unbounded).
fn intreg_ll(lo: f64, hi: f64, xb: f64, lns: f64) -> f64 {
    let s = lns.exp();
    if !lo.is_nan() && !hi.is_nan() && lo == hi {
        dist::lnnormalden((lo - xb) / s) - lns
    } else if lo.is_nan() {
        crate::xtnl::ln_phi((hi - xb) / s)
    } else if hi.is_nan() {
        crate::xtnl::ln_phi((xb - lo) / s)
    } else {
        let a = (lo - xb) / s;
        let b = (hi - xb) / s;
        // difference of normal CDFs computed on the side with more precision
        let p = if a > 0.0 { dist::normal(-a) - dist::normal(-b) } else { dist::normal(b) - dist::normal(a) };
        p.max(1e-300).ln()
    }
}

pub struct Intreg;

impl CommandHandler for Intreg {
    fn name(&self) -> &'static str {
        "intreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (y1, rest) = xm::first_word(&cmd.main);
        let (y2, indep) = xm::first_word(&rest);
        if y2.is_empty() {
            return Err(rc_err(102, "too few variables specified"));
        }
        let y1 = xm::var_name(ds, &y1)?;
        let y2 = xm::var_name(ds, &y2)?;
        let nocons = cmd.has_opt("noconstant", 3);
        let het = xm::opt_with_sub(cmd, "het", 3);
        let mut specs: Vec<(Option<&str>, &str, bool)> = vec![(None, indep.as_str(), nocons)];
        if let Some((h, sub)) = &het {
            specs.push((None, h.as_str(), sub.contains("nocons")));
        }
        let ds_ = &*ds;
        let designs = xm::build_eqs(ds_, st, cmd, &specs, "fapi", &[])?;
        let c1 = util::col(ds_, &y1)?;
        let c2 = util::col(ds_, &y2)?;
        let keep: Vec<bool> = designs[0].rows.iter().map(|&i| !(is_missing(c1[i]) && is_missing(c2[i]))).collect();
        let designs: Vec<Design> = designs.iter().map(|d| xm::subset(d, &keep)).collect();
        let d = &designs[0];
        let n = d.n();
        let lo: Vec<f64> = d.rows.iter().map(|&i| if is_missing(c1[i]) { f64::NAN } else { c1[i] }).collect();
        let hi: Vec<f64> = d.rows.iter().map(|&i| if is_missing(c2[i]) { f64::NAN } else { c2[i] }).collect();
        let (mut nu, mut nl, mut nr, mut ni) = (0.0, 0.0, 0.0, 0.0);
        for t in 0..n {
            let w = if d.wkind == Some(WeightKind::F) { d.w[t] } else { 1.0 };
            if !lo[t].is_nan() && !hi[t].is_nan() {
                if lo[t] > hi[t] {
                    return Err(rc_err(498, format!("observations with {y1} > {y2} not allowed")));
                }
                if lo[t] == hi[t] {
                    nu += w
                } else {
                    ni += w
                }
            } else if lo[t].is_nan() {
                nl += w
            } else {
                nr += w
            }
        }
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds_, d, &vce)?;
        let mid: Vec<f64> = (0..n)
            .map(|t| match (lo[t].is_nan(), hi[t].is_nan()) {
                (false, false) => (lo[t] + hi[t]) / 2.0,
                (true, _) => hi[t],
                (_, true) => lo[t],
            })
            .collect();
        let fweight = d.wkind == Some(WeightKind::F);
        let f_hom = |t: usize, v: &[f64]| intreg_ll(lo[t], hi[t], v[0], v[1]);
        let level = util::level(cmd, st)?;
        let mut out = Vec::new();
        // constant-only model
        let mut ll0 = SYSMISS;
        if let Some(d0) = xm::const_only(d) {
            let (bs, s) = xm::ols_start(&d0, &mid);
            let mut b0 = bs;
            b0.push(s.max(1e-8).ln());
            let g = xm::numeric(&f_hom);
            let f0 = xm::ml_fit(&[("", &d0)], &["lnsigma"], &g, b0, &d.w, fweight, &Vce::Oim, None)?;
            out.push("Fitting constant-only model:".to_string());
            out.push(String::new());
            out.extend(est::iteration_log(&f0.res.iterations, "Log likelihood"));
            out.push(String::new());
            ll0 = f0.ll();
        }
        let (bs, s) = xm::ols_start(d, &mid);
        let fit = if het.is_some() {
            let dz = &designs[1];
            let mut b0 = bs;
            let kz = dz.active().len();
            let mut gz = vec![0.0; kz];
            if let Some(c) = dz.active().iter().position(|&j| dz.names[j] == "_cons") {
                gz[c] = s.max(1e-8).ln();
            }
            b0.extend(gz);
            let f = |t: usize, v: &[f64]| intreg_ll(lo[t], hi[t], v[0], v[1]);
            let g = xm::numeric(&f);
            xm::ml_fit(&[("model", d), ("lnsigma", dz)], &[], &g, b0, &d.w, fweight, &vce, cluster.as_deref())?
        } else {
            let mut b0 = bs;
            b0.push(s.max(1e-8).ln());
            let g = xm::numeric(&f_hom);
            xm::ml_fit(&[("", d)], &["lnsigma"], &g, b0, &d.w, fweight, &vce, cluster.as_deref())?
        };
        out.push("Fitting full model:".to_string());
        out.push(String::new());
        out.extend(est::iteration_log(&fit.res.iterations, "Log likelihood"));
        out.push(String::new());
        let robust = vce != Vce::Oim;
        let (wchi, wdf) = fit.wald_eq(0);
        let (tname, chi2, df) = if !robust && !is_missing(ll0) && het.is_none() { ("LR", 2.0 * (fit.ll() - ll0), wdf) } else { ("Wald", wchi, wdf) };
        let hdr = Header {
            title: "Interval regression",
            n: d.n_eff(),
            lines: vec![
                (String::new(), "Uncensored".into(), util::commas(nu as usize)),
                (String::new(), "Left-censored".into(), util::commas(nl as usize)),
                (String::new(), "Right-censored".into(), util::commas(nr as usize)),
                (String::new(), "Interval-cens.".into(), util::commas(ni as usize)),
                (String::new(), String::new(), String::new()),
            ],
            test: Some((tname.into(), chi2, df)),
            ll: Some(fit.ll()),
            robust,
            n_clust: fit.n_clust.map(|g| (g, if let Vce::Cluster(c) = &vce { c.clone() } else { String::new() })),
        };
        let mut hl = hdr.render();
        // blank right-hand rows are spacers
        for l in hl.iter_mut() {
            if l.trim_end().ends_with('=') {
                *l = String::new();
            }
        }
        out.extend(hl);
        let mut extra = Vec::new();
        let mut sigma = SYSMISS;
        if het.is_none() {
            let j = fit.idx("/:lnsigma").unwrap();
            let (ls, se) = (fit.b[j], fit.v[(j, j)].max(0.0).sqrt());
            sigma = ls.exp();
            extra.push(Row::Tr("sigma".into(), sigma, sigma * se, ls, se, f64::exp));
        }
        out.extend(xm::table("", &fit, level, None, &extra));
        let text = out.join("\n");
        let depv = format!("{y1} {y2}");
        xm::post_fit(st, "intreg", d, &fit, &vce, &|e: &mut Results| {
            e.set("N_unc", nu);
            e.set("N_lc", nl);
            e.set("N_rc", nr);
            e.set("N_int", ni);
            e.set("ll_0", ll0);
            e.set("sigma", sigma);
            e.set("k_dv", 2.0);
            xm::set_test(e, tname, chi2, df);
            e.set_str("depvar", depv.clone());
            e.set_str("title", "Interval regression");
            e.set_str("predict", "intreg_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// cnsreg
// ---------------------------------------------------------------------------

pub struct Cnsreg;

impl CommandHandler for Cnsreg {
    fn name(&self) -> &'static str {
        "cnsreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: cmd.has_opt("noconstant", 3), weights: "fai", extra: vec![] })?;
        let act = d.active();
        let names: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
        let (rm, rv, ctext) = xm::constraint_matrix(st, cmd, &names)?;
        if rm.rows == 0 {
            return Err(rc_err(198, "constraints() required"));
        }
        let (tm, b0) = xm::constraint_basis(&rm, &rv)?;
        let ka = act.len();
        let p = tm.cols;
        let n = d.n();
        // regress y - X b0 on X T
        let xb0: Vec<f64> = (0..n).map(|t| act.iter().enumerate().map(|(a, &j)| d.cols[j][t] * b0[a]).sum()).collect();
        let z: Vec<Vec<f64>> = (0..p).map(|c| (0..n).map(|t| act.iter().enumerate().map(|(a, &j)| d.cols[j][t] * tm[(a, c)]).sum()).collect()).collect();
        let mut ztz = Matrix::zeros(p, p);
        let mut zty = vec![0.0; p];
        for t in 0..n {
            let w = d.w[t];
            for a in 0..p {
                zty[a] += w * z[a][t] * (d.y[t] - xb0[t]);
                for c in 0..=a {
                    ztz[(a, c)] += w * z[a][t] * z[c][t];
                }
            }
        }
        for a in 0..p {
            for c in 0..a {
                ztz[(c, a)] = ztz[(a, c)];
            }
        }
        let (zi, _) = ztz.sym_inverse(1e-14);
        let g: Vec<f64> = (0..p).map(|a| (0..p).map(|c| zi[(a, c)] * zty[c]).sum()).collect();
        let ba: Vec<f64> = (0..ka).map(|a| b0[a] + (0..p).map(|c| tm[(a, c)] * g[c]).sum::<f64>()).collect();
        let resid: Vec<f64> = (0..n).map(|t| d.y[t] - act.iter().enumerate().map(|(a, &j)| d.cols[j][t] * ba[a]).sum::<f64>()).collect();
        let rss: f64 = (0..n).map(|t| d.w[t] * resid[t] * resid[t]).sum();
        let n_eff = d.n_eff();
        let df_r = n_eff - p as f64;
        let s2 = rss / df_r;
        let vce = est::parse_vce(cmd, Vce::Ols)?;
        // V = T (Z'Z)^-1 T' s2  (or sandwich)
        let vg = match &vce {
            Vce::Ols | Vce::Oim => zi.scale(s2),
            _ => {
                let cl = est::cluster_values(ds, &d, &vce)?;
                let scores: Vec<Vec<f64>> = (0..n).map(|t| (0..p).map(|a| z[a][t] * d.w[t] * resid[t]).collect()).collect();
                let (v, g) = est::sandwich(&zi, &scores, cl.as_deref(), 1.0);
                match g {
                    gg if cl.is_some() => {
                        let gf = gg as f64;
                        v.scale(gf / (gf - 1.0) * (n_eff - 1.0) / df_r)
                    }
                    _ => v.scale(n_eff / df_r),
                }
            }
        };
        let mut va = Matrix::zeros(ka, ka);
        for a in 0..ka {
            for c in 0..ka {
                let mut s = 0.0;
                for i in 0..p {
                    for j in 0..p {
                        s += tm[(a, i)] * vg[(i, j)] * tm[(c, j)];
                    }
                }
                va[(a, c)] = s;
            }
        }
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = ba[a];
        }
        let v = est::expand_v(&va, &act, k);
        // model F test of the slopes
        let idx: Vec<usize> = (0..ka).filter(|&a| names[a] != "_cons").collect();
        let (w, rank) = crate::heckman::wald(&ba, &va, &idx);
        let fstat = if rank > 0.0 { w / rank } else { SYSMISS };
        let level = util::level(cmd, st)?;
        let rmse = s2.sqrt();
        let mut out = vec![
            format!("{:<52}{:<15} = {:>10}", "Constrained linear regression", "Number of obs", util::commas(n_eff as usize)),
            format!("{:<52}{:<15} = {:>10}", "", format!("F({}, {})", fmt_g(rank, 9), fmt_g(df_r, 9)), xm::f2(fstat)),
            format!("{:<52}{:<15} = {:>10}", "", "Prob > F", xm::f4(dist::ftail(rank, df_r, fstat))),
            format!("{:<52}{:<15} = {:>10}", "", "Root MSE", format!("{:.4}", rmse)),
            String::new(),
        ];
        out.extend(ctext.iter().cloned());
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: Some(df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let cmat = {
            let mut m = Matrix::zeros(rm.rows, k + 1);
            for i in 0..rm.rows {
                for (a, &j) in act.iter().enumerate() {
                    m[(i, j)] = rm[(i, a)];
                }
                m[(i, k)] = rv[i];
            }
            let mut cn = d.names.clone();
            cn.push("_r".into());
            m.with_names((1..=rm.rows).map(|i| format!("r{i}")).collect(), cn)
        };
        est::post(st, "cnsreg", &d, &b, &v, &|e: &mut Results| {
            e.set("df_m", rank);
            e.set("df_r", df_r);
            e.set("F", fstat);
            e.set("rmse", rmse);
            e.set("rank", p as f64);
            e.set("ll", -0.5 * n_eff * ((2.0 * std::f64::consts::PI).ln() + (rss / n_eff).ln() + 1.0));
            e.set_mat("Cns", cmat.clone());
            e.set_str("title", "Constrained linear regression");
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("properties", "b V");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// eivreg
// ---------------------------------------------------------------------------

pub struct Eivreg;

impl CommandHandler for Eivreg {
    fn name(&self) -> &'static str {
        "eivreg"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: &indep, noconstant: false, weights: "fa", extra: vec![] })?;
        let act = d.active();
        let ka = act.len();
        let mut rel = vec![1.0; ka];
        if let Some(r) = cmd.opt("reliab", 5) {
            let toks: Vec<&str> = r.split_whitespace().collect();
            let mut i = 0;
            while i + 1 < toks.len() + 1 && i < toks.len() {
                let v = xm::var_name(ds, toks[i])?;
                let val: f64 = toks.get(i + 1).and_then(|s| s.parse().ok()).ok_or_else(|| rc_err(198, "reliab(): variable reliability pairs required"))?;
                if !(val > 0.0 && val <= 1.0) {
                    return Err(rc_err(198, "reliability must be in (0,1]"));
                }
                let a = act.iter().position(|&j| d.names[j] == v).ok_or_else(|| rc_err(198, format!("{v} not in model")))?;
                rel[a] = val;
                i += 2;
            }
        }
        let n = d.n();
        let sw: f64 = d.w.iter().sum();
        let means: Vec<f64> = act.iter().map(|&j| (0..n).map(|t| d.w[t] * d.cols[j][t]).sum::<f64>() / sw).collect();
        let mut xtx = Matrix::zeros(ka, ka);
        let mut xty = vec![0.0; ka];
        for t in 0..n {
            for a in 0..ka {
                xty[a] += d.w[t] * d.cols[act[a]][t] * d.y[t];
                for c in 0..ka {
                    xtx[(a, c)] += d.w[t] * d.cols[act[a]][t] * d.cols[act[c]][t];
                }
            }
        }
        let mut amat = xtx.clone();
        let mut ss = vec![0.0; ka];
        for a in 0..ka {
            ss[a] = (0..n).map(|t| d.w[t] * (d.cols[act[a]][t] - means[a]).powi(2)).sum::<f64>();
            amat[(a, a)] -= (1.0 - rel[a]) * ss[a];
        }
        if amat.cholesky().is_err() {
            return Err(rc_err(399, "reliability r() too small"));
        }
        let (ai, _) = amat.sym_inverse(1e-14);
        let ba: Vec<f64> = (0..ka).map(|a| (0..ka).map(|c| ai[(a, c)] * xty[c]).sum()).collect();
        let yty: f64 = (0..n).map(|t| d.w[t] * d.y[t] * d.y[t]).sum();
        let bab: f64 = (0..ka).map(|a| (0..ka).map(|c| ba[a] * amat[(a, c)] * ba[c]).sum::<f64>()).sum();
        let n_eff = d.n_eff();
        let df_r = n_eff - ka as f64;
        let mse = (yty - bab) / df_r;
        // sandwich A^-1 H'WH A^-1
        let mut meat = Matrix::zeros(ka, ka);
        for t in 0..n {
            let e = d.y[t] - (0..ka).map(|a| d.cols[act[a]][t] * ba[a]).sum::<f64>();
            let h: Vec<f64> = (0..ka).map(|a| e * d.cols[act[a]][t] + (d.cols[act[a]][t] - means[a]).powi(2) * (1.0 - rel[a]) * ba[a]).collect();
            let wt = if d.wkind == Some(WeightKind::A) { d.w[t] * d.w[t] } else { d.w[t] };
            for a in 0..ka {
                for c in 0..ka {
                    meat[(a, c)] += wt * h[a] * h[c];
                }
            }
        }
        let va = ai.mul(&meat)?.mul(&ai)?;
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = ba[a];
        }
        let v = est::expand_v(&va, &act, k);
        let idx: Vec<usize> = (0..ka).filter(|&a| d.names[act[a]] != "_cons").collect();
        let (w, rank) = crate::heckman::wald(&ba, &va, &idx);
        let fstat = w / rank;
        let ybar = (0..n).map(|t| d.w[t] * d.y[t]).sum::<f64>() / sw;
        let tss: f64 = (0..n).map(|t| d.w[t] * (d.y[t] - ybar).powi(2)).sum();
        let r2 = 1.0 - (yty - bab) / tss;
        let level = util::level(cmd, st)?;
        let mut out = vec!["Errors-in-variables regression".to_string(), String::new()];
        let mut rlines = vec![format!("{:>12}  {:>11}", "", "Assumed"), format!("{:>12}  {:>11}", "Variable", "reliability"), format!("{:-<12}--{:-<11}", "", "")];
        for a in 0..ka {
            if rel[a] < 1.0 {
                rlines.push(format!("{:>12}  {:>11.4}", d.names[act[a]], rel[a]));
            }
        }
        rlines.push(format!("{:>12}  {:>11.4}", "*", 1.0));
        let right = [
            format!("{:<13} = {:>10}", "Number of obs", util::commas(n_eff as usize)),
            format!("{:<13} = {:>10}", format!("F({}, {})", fmt_g(rank, 9), fmt_g(df_r, 9)), xm::f2(fstat)),
            format!("{:<13} = {:>10}", "Prob > F", xm::f4(dist::ftail(rank, df_r, fstat))),
            format!("{:<13} = {:>10}", "R-squared", xm::f4(r2)),
            format!("{:<13} = {:>10}", "Root MSE", fmt_g(mse.sqrt(), 6)),
        ];
        for i in 0..rlines.len().max(right.len() + 1) {
            let l = rlines.get(i).cloned().unwrap_or_default();
            let r = if i >= 1 { right.get(i - 1).cloned().unwrap_or_default() } else { String::new() };
            out.push(format!("{:<52}{}", l, r).trim_end().to_string());
        }
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: Some(df_r), level, eform: None, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let relm = Matrix::row_vector(&rel).with_names(vec!["r1".into()], act.iter().map(|&j| d.names[j].clone()).collect());
        est::post(st, "eivreg", &d, &b, &v, &|e: &mut Results| {
            e.set("df_m", rank);
            e.set("df_r", df_r);
            e.set("F", fstat);
            e.set("r2", r2);
            e.set("rmse", mse.sqrt());
            e.set_mat("rel", relm.clone());
            e.set_str("title", "Errors-in-variables regression");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// hetregress
// ---------------------------------------------------------------------------

pub struct Hetregress;

impl CommandHandler for Hetregress {
    fn name(&self) -> &'static str {
        "hetregress"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (dep, indep) = xm::first_word(&cmd.main);
        let dep = xm::var_name(ds, &dep)?;
        let het = xm::opt_with_sub(cmd, "het", 3).unwrap_or_default();
        let twostep = cmd.has_opt("twostep", 3);
        if twostep && het.0.is_empty() {
            return Err(rc_err(198, "option het() required with twostep"));
        }
        let designs = xm::build_eqs(ds, st, cmd, &[(Some(dep.as_str()), indep.as_str(), cmd.has_opt("noconstant", 3)), (None, het.0.as_str(), false)], if twostep { "" } else { "fapi" }, &[])?;
        let (d, dz) = (&designs[0], &designs[1]);
        let n = d.n();
        let level = util::level(cmd, st)?;
        let ols = est::ols(d, &Vce::Ols, None)?;
        let kz = dz.active().len();
        let zact = dz.active();
        if twostep {
            // Harvey: regress ln(e^2) on z, then weighted least squares
            let mut dl = dz.clone();
            dl.y = ols.resid.iter().map(|e| (e * e).max(1e-300).ln()).collect();
            let fz = est::ols(&dl, &Vce::Ols, None)?;
            let mut g: Vec<f64> = zact.iter().map(|&j| fz.b[j]).collect();
            if let Some(c) = zact.iter().position(|&j| dz.names[j] == "_cons") {
                g[c] += 1.2704;
            }
            let (zi, _) = est::cross(dz, &zact).sym_inverse(1e-14);
            let vg = zi.scale(4.9348);
            let s2: Vec<f64> = (0..n).map(|t| zact.iter().enumerate().map(|(a, &j)| dz.cols[j][t] * g[a]).sum::<f64>().exp()).collect();
            let mut dw = d.clone();
            dw.w = s2.iter().map(|s| 1.0 / s).collect();
            let act = d.active();
            let (xi, _) = est::cross(&dw, &act).sym_inverse(1e-14);
            let mut xty = vec![0.0; act.len()];
            for t in 0..n {
                for (a, &j) in act.iter().enumerate() {
                    xty[a] += dw.w[t] * d.cols[j][t] * d.y[t];
                }
            }
            let bx: Vec<f64> = (0..act.len()).map(|a| (0..act.len()).map(|c| xi[(a, c)] * xty[c]).sum()).collect();
            // scale by the weighted residual variance
            let wrss: f64 = (0..n).map(|t| dw.w[t] * (d.y[t] - act.iter().enumerate().map(|(a, &j)| d.cols[j][t] * bx[a]).sum::<f64>()).powi(2)).sum();
            let xi = xi.scale(wrss / (n as f64 - act.len() as f64));
            let mut bt = bx.clone();
            bt.extend(g.iter().cloned());
            let p = bt.len();
            let mut vt = Matrix::zeros(p, p);
            for a in 0..act.len() {
                for c in 0..act.len() {
                    vt[(a, c)] = xi[(a, c)];
                }
            }
            for a in 0..kz {
                for c in 0..kz {
                    vt[(act.len() + a, act.len() + c)] = vg[(a, c)];
                }
            }
            let res = est::MlResult { b: bt.clone(), ll: SYSMISS, info: Matrix::zeros(p, p), scores: vec![], iterations: vec![], converged: true };
            let mut fit = xm::finish_fit(&[(dep.as_str(), d), ("lnsigma2", dz)], &[], res, &Vce::Oim, None)?;
            fit.vt = vt.clone();
            fit.v = est::expand_v(&vt, &fit.pos, fit.names.len());
            let (chi2, df) = fit.wald_eq(0);
            let (hw, hdf) = fit.wald_eq(1);
            let mut out = vec![
                crate::choice_mixed::hrow("Heteroskedastic linear regression", "Number of obs", util::commas(n)),
                "Two-step GLS estimation".to_string(),
                crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), xm::f2(chi2)),
                crate::choice_mixed::hrow("", "Prob > chi2", xm::f4(xm::chi2p(df, chi2))),
                String::new(),
            ];
            out.extend(xm::table("", &fit, level, None, &[]));
            out.push(format!("Wald test of lnsigma2=0: chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(hdf, 9), xm::f2(hw), xm::f4(xm::chi2p(hdf, hw))));
            let text = out.join("\n");
            xm::post_fit(st, "hetregress", d, &fit, &Vce::Oim, &|e: &mut Results| {
                xm::set_test(e, "Wald", chi2, df);
                e.set("chi2_c", hw);
                e.set("df_m_c", hdf);
                e.set("p_c", xm::chi2p(hdf, hw));
                e.set_str("method", "twostep");
                e.set_str("vce", "conventional");
                e.set_str("title", "Heteroskedastic linear regression");
                e.set_str("predict", "hetregress_p");
                e.set_str("_output", text.clone());
            });
            return Ok(text);
        }
        let vce = xm::ml_vce_opt(cmd, d)?;
        let cluster = est::cluster_values(ds, d, &vce)?;
        let y = d.y.clone();
        let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>) {
            let r = y[t] - v[0];
            let s2 = v[1].exp();
            (-0.5 * ((2.0 * std::f64::consts::PI).ln() + v[1] + r * r / s2), vec![r / s2, -0.5 + 0.5 * r * r / s2])
        };
        let g = xm::with_grad(&f);
        let mut b0: Vec<f64> = d.active().iter().map(|&j| ols.b[j]).collect();
        let mut gz = vec![0.0; kz];
        if let Some(c) = zact.iter().position(|&j| dz.names[j] == "_cons") {
            gz[c] = (ols.rss / n as f64).ln();
        }
        b0.extend(gz);
        let fweight = d.wkind == Some(WeightKind::F);
        let fit = xm::ml_fit(&[(dep.as_str(), d), ("lnsigma2", dz)], &[], &g, b0, &d.w, fweight, &vce, cluster.as_deref())?;
        let robust = vce != Vce::Oim;
        let (chi2, df) = fit.wald_eq(0);
        let sw: f64 = d.w.iter().sum();
        let ll_ols = -0.5 * sw * ((2.0 * std::f64::consts::PI).ln() + (ols.rss / sw).ln() + 1.0);
        let (hw, hdf) = fit.wald_eq(1);
        let lr = 2.0 * (fit.ll() - ll_ols);
        let use_wald = robust || cmd.has_opt("waldhet", 5);
        let (ctype, cval) = if use_wald { ("Wald", hw) } else { ("LR", lr) };
        let mut out = vec!["Fitting full model:".to_string(), String::new()];
        out.extend(est::iteration_log(&fit.res.iterations, "Log likelihood"));
        out.push(String::new());
        out.push(crate::choice_mixed::hrow("Heteroskedastic linear regression", "Number of obs", util::commas(d.n_eff() as usize)));
        out.push("ML estimation".into());
        out.extend(Header { title: "", n: 0.0, lines: vec![], test: Some(("Wald".into(), chi2, df)), ll: Some(fit.ll()), robust, n_clust: fit.n_clust.map(|g| (g, if let Vce::Cluster(c) = &vce { c.clone() } else { String::new() })) }.render().into_iter().skip(1));
        out.extend(xm::table("", &fit, level, None, &[]));
        if hdf > 0.0 {
            out.push(format!("{ctype} test of lnsigma2=0: chi2({}) = {:>8}   Prob > chi2 = {}", fmt_g(hdf, 9), xm::f2(cval), xm::f4(xm::chi2p(hdf, cval))));
        }
        let text = out.join("\n");
        xm::post_fit(st, "hetregress", d, &fit, &vce, &|e: &mut Results| {
            xm::set_test(e, "Wald", chi2, df);
            e.set("chi2_c", cval);
            e.set("df_m_c", hdf);
            e.set("p_c", xm::chi2p(hdf, cval));
            e.set("ll_c", ll_ols);
            e.set_str("chi2_ct", ctype);
            e.set_str("method", "ml");
            e.set_str("title", "Heteroskedastic linear regression");
            e.set_str("predict", "hetregress_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// sureg / reg3
// ---------------------------------------------------------------------------

pub struct System {
    pub three: bool,
}

struct EqSpec {
    name: String,
    dep: String,
    rhs: String,
    nocons: bool,
}

fn parse_system(ds: &DataSet, main: &str) -> Result<Vec<EqSpec>, String> {
    let mut out = Vec::new();
    for g in xm::paren_groups(main) {
        let (g, nocons) = match g.split_once(',') {
            Some((a, o)) => (a.to_string(), o.to_lowercase().contains("nocons")),
            None => (g.clone(), false),
        };
        let (name, body) = match g.split_once(':') {
            Some((a, b)) if !a.trim().contains(' ') && !a.contains('.') => (Some(a.trim().to_string()), b.to_string()),
            _ => (None, g.clone()),
        };
        if let Some((l, r)) = body.split_once('=') {
            let deps = crate::varlist::expand(ds, l.trim())?;
            for dv in deps {
                out.push(EqSpec { name: if out.is_empty() { name.clone() } else { None }.unwrap_or_else(|| dv.clone()), dep: dv, rhs: r.trim().to_string(), nocons });
            }
        } else {
            let (dep, rhs) = xm::first_word(&body);
            let dep = xm::var_name(ds, &dep)?;
            out.push(EqSpec { name: name.unwrap_or_else(|| dep.clone()), dep, rhs, nocons });
        }
    }
    if out.is_empty() {
        return Err(rc_err(198, "equation(s) required: (depvar varlist) ..."));
    }
    Ok(out)
}

/// Generalised least squares for a system: b = (X'(S^-1 (x) I)X)^-1 X'(S^-1 (x) I) y
/// with `xs` the (projected) regressors and `xr` the regressors entering the
/// cross products with y.
fn system_gls(xs: &[Vec<Vec<f64>>], ys: &[Vec<f64>], sinv: &Matrix, w: &[f64]) -> (Vec<f64>, Matrix) {
    let m = xs.len();
    let ks: Vec<usize> = xs.iter().map(|x| x.len()).collect();
    let off: Vec<usize> = (0..m).map(|i| ks[..i].iter().sum()).collect();
    let kt: usize = ks.iter().sum();
    let n = w.len();
    let mut a = Matrix::zeros(kt, kt);
    let mut r = vec![0.0; kt];
    for i in 0..m {
        for j in 0..m {
            let s = sinv[(i, j)];
            for p in 0..ks[i] {
                let xp = &xs[i][p];
                r[off[i] + p] += s * (0..n).map(|t| w[t] * xp[t] * ys[j][t]).sum::<f64>();
                for q in 0..ks[j] {
                    let xq = &xs[j][q];
                    a[(off[i] + p, off[j] + q)] += s * (0..n).map(|t| w[t] * xp[t] * xq[t]).sum::<f64>();
                }
            }
        }
    }
    let (ai, _) = a.sym_inverse(1e-14);
    let b = (0..kt).map(|p| (0..kt).map(|q| ai[(p, q)] * r[q]).sum()).collect();
    (b, ai)
}

impl CommandHandler for System {
    fn name(&self) -> &'static str {
        if self.three { "reg3" } else { "sureg" }
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let specs = parse_system(ds, &cmd.main)?;
        let m = specs.len();
        // estimation sample common to all equations
        let mut all: Vec<String> = Vec::new();
        for s in &specs {
            all.push(s.dep.clone());
            all.extend(crate::zip::base_vars(ds, &s.rhs));
        }
        let exo_opt = cmd.opt("exog", 3).map(|s| crate::varlist::expand(ds, s)).transpose()?.unwrap_or_default();
        let endo_opt = cmd.opt("endog", 3).map(|s| crate::varlist::expand(ds, s)).transpose()?.unwrap_or_default();
        let inst_opt = cmd.opt("inst", 4).map(|s| crate::varlist::expand(ds, s)).transpose()?;
        all.extend(exo_opt.iter().cloned());
        all.extend(endo_opt.iter().cloned());
        if let Some(i) = &inst_opt {
            all.extend(i.iter().cloned());
        }
        let mut designs = Vec::new();
        for s in &specs {
            designs.push(est::build(ds, st, cmd, DesignSpec { depvar: Some(&s.dep), indep: &s.rhs, noconstant: s.nocons, weights: "fa", extra: all.clone() })?);
        }
        let n = designs[0].n();
        let w = designs[0].w.clone();
        let sw: f64 = w.iter().sum();
        let rows = designs[0].rows.clone();
        let method = if !self.three {
            "sure"
        } else if cmd.has_opt("2sls", 4) {
            "2sls"
        } else if cmd.has_opt("ols", 3) {
            "ols"
        } else if cmd.has_opt("sure", 4) || cmd.has_opt("allexog", 3) {
            "sure"
        } else if cmd.has_opt("ireg3", 5) {
            "ireg3"
        } else {
            "3sls"
        };
        let isure = cmd.has_opt("isure", 5);
        // endogenous and exogenous variables
        let mut endog: Vec<String> = Vec::new();
        let mut exog: Vec<String> = Vec::new();
        if self.three {
            for s in &specs {
                if !endog.contains(&s.dep) {
                    endog.push(s.dep.clone());
                }
            }
            for e in &endo_opt {
                if !endog.contains(e) {
                    endog.push(e.clone());
                }
            }
            for d in &designs {
                for (j, nm) in d.names.iter().enumerate() {
                    if nm == "_cons" || d.omitted[j] {
                        continue;
                    }
                    if !endog.contains(nm) && !exog.contains(nm) {
                        exog.push(nm.clone());
                    }
                }
            }
            for e in &exo_opt {
                if !exog.contains(e) {
                    exog.push(e.clone());
                }
            }
        }
        // instruments
        let zcols: Vec<Vec<f64>> = if self.three && !matches!(method, "sure" | "ols") {
            let list = inst_opt.clone().unwrap_or_else(|| exog.clone());
            let mut z: Vec<Vec<f64>> = Vec::new();
            for v in &list {
                let c = match designs.iter().find_map(|d| d.names.iter().position(|x| x == v).map(|j| d.cols[j].clone())) {
                    Some(c) => c,
                    None => {
                        let col = crate::est::build(ds, st, cmd, DesignSpec { depvar: None, indep: v, noconstant: true, weights: "fa", extra: all.clone() })?;
                        col.cols[0].clone()
                    }
                };
                z.push(c);
            }
            z.push(vec![1.0; n]);
            z
        } else {
            Vec::new()
        };
        let project = |x: &[f64]| -> Vec<f64> {
            if zcols.is_empty() {
                return x.to_vec();
            }
            let kz = zcols.len();
            let mut ztz = Matrix::zeros(kz, kz);
            let mut ztx = vec![0.0; kz];
            for a in 0..kz {
                ztx[a] = (0..n).map(|t| w[t] * zcols[a][t] * x[t]).sum();
                for c in 0..kz {
                    ztz[(a, c)] = (0..n).map(|t| w[t] * zcols[a][t] * zcols[c][t]).sum();
                }
            }
            let (zi, _) = ztz.sym_inverse(1e-14);
            let g: Vec<f64> = (0..kz).map(|a| (0..kz).map(|c| zi[(a, c)] * ztx[c]).sum()).collect();
            (0..n).map(|t| (0..kz).map(|a| zcols[a][t] * g[a]).sum()).collect()
        };
        let acts: Vec<Vec<usize>> = designs.iter().map(|d| d.active()).collect();
        let xs: Vec<Vec<Vec<f64>>> = designs.iter().zip(&acts).map(|(d, a)| a.iter().map(|&j| d.cols[j].clone()).collect()).collect();
        let xh: Vec<Vec<Vec<f64>>> = xs.iter().map(|x| x.iter().map(|c| if self.three && !matches!(method, "sure" | "ols") { project(c) } else { c.clone() }).collect()).collect();
        let ys: Vec<Vec<f64>> = designs.iter().map(|d| d.y.clone()).collect();
        let ks: Vec<usize> = xs.iter().map(|x| x.len()).collect();
        let off: Vec<usize> = (0..m).map(|i| ks[..i].iter().sum()).collect();
        let dfk = cmd.has_opt("dfk", 3);
        let resid_of = |b: &[f64]| -> Vec<Vec<f64>> { (0..m).map(|i| (0..n).map(|t| ys[i][t] - (0..ks[i]).map(|p| xs[i][p][t] * b[off[i] + p]).sum::<f64>()).collect()).collect() };
        let sigma_of = |e: &[Vec<f64>]| -> Matrix {
            let mut s = Matrix::zeros(m, m);
            for i in 0..m {
                for j in 0..m {
                    let den = if dfk { ((sw - ks[i] as f64) * (sw - ks[j] as f64)).sqrt() } else { sw };
                    s[(i, j)] = (0..n).map(|t| w[t] * e[i][t] * e[j][t]).sum::<f64>() / den;
                }
            }
            s
        };
        // first round: equation by equation (OLS or 2SLS)
        let ident = Matrix::identity(m);
        let mut b = Vec::new();
        for i in 0..m {
            let (bi, _) = system_gls(&[xh[i].clone()], &[ys[i].clone()], &Matrix::identity(1), &w);
            // 2SLS: (Xh'X)^-1 Xh'y equals (Xh'Xh)^-1 Xh'y since Xh'X = Xh'Xh
            b.extend(bi);
        }
        let mut e = resid_of(&b);
        let mut sigma = sigma_of(&e);
        let mut v;
        match method {
            "ols" | "2sls" => {
                // equation by equation with a system covariance of the coefficients
                let sd = Matrix::from_diag(&(0..m).map(|i| 1.0 / sigma[(i, i)]).collect::<Vec<_>>());
                let (_, vv) = system_gls(&xh, &ys, &sd, &w);
                v = vv;
                let _ = ident;
            }
            _ => {
                let mut iter = 0;
                loop {
                    let (sinv, _) = sigma.sym_inverse(1e-14);
                    let (nb, vv) = system_gls(&xh, &ys, &sinv, &w);
                    v = vv;
                    let change = nb.iter().zip(&b).map(|(a, c)| (a - c).abs() / (c.abs() + 1.0)).fold(0.0, f64::max);
                    b = nb;
                    e = resid_of(&b);
                    iter += 1;
                    if !(isure || method == "ireg3") || change < 1e-10 || iter > 500 {
                        break;
                    }
                    sigma = sigma_of(&e);
                }
                if isure || method == "ireg3" {
                    sigma = sigma_of(&e);
                }
            }
        }
        let small = cmd.has_opt("small", 2);
        // per-equation statistics
        let mut eqstats = Vec::new();
        let mut names = Vec::new();
        let mut omitted = Vec::new();
        let mut is_base = Vec::new();
        let mut pos = Vec::new();
        for (i, d) in designs.iter().enumerate() {
            for j in 0..d.k() {
                if !d.omitted[j] {
                    pos.push(names.len());
                }
                names.push(format!("{}:{}", specs[i].name, d.names[j]));
                omitted.push(d.omitted[j]);
                is_base.push(d.is_base[j]);
            }
        }
        let kt = names.len();
        let mut bf = vec![0.0; kt];
        for (a, &j) in pos.iter().enumerate() {
            bf[j] = b[a];
        }
        let total_k: usize = ks.iter().sum();
        let df_small = (m as f64) * sw - total_k as f64;
        let vscale = if small && !dfk { 1.0 } else { 1.0 };
        let vt = v.scale(vscale);
        let vf = est::expand_v(&vt, &pos, kt);
        for i in 0..m {
            let rss: f64 = (0..n).map(|t| w[t] * e[i][t] * e[i][t]).sum();
            let ybar = (0..n).map(|t| w[t] * ys[i][t]).sum::<f64>() / sw;
            let tss: f64 = (0..n).map(|t| w[t] * (ys[i][t] - ybar).powi(2)).sum();
            let has_cons = designs[i].has_cons;
            let r2 = if has_cons { 1.0 - rss / tss } else { 1.0 - rss / (0..n).map(|t| w[t] * ys[i][t] * ys[i][t]).sum::<f64>() };
            let idx: Vec<usize> = (0..ks[i]).filter(|&p| designs[i].names[acts[i][p]] != "_cons").map(|p| off[i] + p).collect();
            let (chi2, df) = crate::heckman::wald(&b, &vt, &idx);
            let rmse_den = if dfk { sw - ks[i] as f64 } else { sw };
            eqstats.push((specs[i].name.clone(), idx.len(), (rss / rmse_den).sqrt(), r2, chi2, df));
        }
        let level = util::level(cmd, st)?;
        let title = if !self.three || method == "sure" {
            "Seemingly unrelated regression"
        } else {
            match method {
                "2sls" => "Two-stage least-squares regression",
                "ols" => "Multivariate regression",
                "ireg3" => "Three-stage least-squares regression, iterated",
                _ => "Three-stage least-squares regression",
            }
        };
        let mut out = vec![title.to_string(), "-".repeat(78)];
        out.push(format!("{:<12} {:>8} {:>8} {:>12} {:>11} {:>11} {:>8}", "Equation", "Obs", "Params", "RMSE", "\"R-squared\"", if small { "F" } else { "chi2" }, if small { "P>F" } else { "P>chi2" }));
        out.push("-".repeat(78));
        for (nm, p, rmse, r2, chi2, df) in &eqstats {
            let (stat, pv) = if small { (chi2 / df, dist::ftail(*df, df_small, chi2 / df)) } else { (*chi2, xm::chi2p(*df, *chi2)) };
            out.push(format!("{:<12} {:>8} {:>8} {:>12} {:>11} {:>11} {:>8}", crate::summarize::abbrev(nm, 12), util::commas(n), p, fmt_g(*rmse, 7), format!("{:.4}", r2), xm::f2(stat), xm::f4(pv)));
        }
        out.push("-".repeat(78));
        out.push(String::new());
        out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &names, b: &bf, v: &vf, omitted: &omitted, is_base: &is_base, df: if small { Some(df_small) } else { None }, level, eform: None, coef_title: "Coefficient" }));
        if self.three && !matches!(method, "sure" | "ols") {
            out.push(format!("Endogenous: {}", endog.join(" ")));
            out.push(format!("Exogenous:  {}", inst_opt.clone().unwrap_or_else(|| exog.clone()).join(" ")));
        }
        if dfk && small {
            out.push("Note: Small-sample degrees-of-freedom adjustment applied when estimating".into());
            out.push("      covariance matrix of residuals.".into());
        }
        // correlation of residuals and Breusch-Pagan test
        let mut bp = 0.0;
        let mut corr = Matrix::identity(m);
        for i in 0..m {
            for j in 0..i {
                let r = sigma[(i, j)] / (sigma[(i, i)] * sigma[(j, j)]).sqrt();
                corr[(i, j)] = r;
                corr[(j, i)] = r;
                bp += sw * r * r;
            }
        }
        if cmd.has_opt("corr", 4) {
            out.push(String::new());
            out.push("Correlation matrix of residuals:".into());
            out.push(format!("{:>12}{}", "", specs.iter().map(|s| format!("{:>10}", crate::summarize::abbrev(&s.name, 9))).collect::<String>()));
            for i in 0..m {
                out.push(format!("{:>12}{}", crate::summarize::abbrev(&specs[i].name, 12), (0..=i).map(|j| format!("{:>10.4}", corr[(i, j)])).collect::<String>()));
            }
            out.push(String::new());
            let bdf = (m * (m - 1) / 2) as f64;
            out.push(format!("Breusch-Pagan test of independence: chi2({}) = {:>8.3}, Pr = {:.4}", bdf, bp, xm::chi2p(bdf, bp)));
        }
        let text = out.join("\n");
        let mut dp = designs[0].clone();
        dp.names = names.clone();
        dp.omitted = omitted;
        dp.is_base = is_base;
        let verb = if self.three { "reg3" } else { "sureg" };
        let deps = specs.iter().map(|s| s.dep.clone()).collect::<Vec<_>>().join(" ");
        let eqn = specs.iter().map(|s| s.name.clone()).collect::<Vec<_>>().join(" ");
        let snames: Vec<String> = specs.iter().map(|s| s.dep.clone()).collect();
        let _ = rows;
        est::post(st, verb, &dp, &bf, &vf, &|r: &mut Results| {
            r.set("k_eq", m as f64);
            r.set("k", total_k as f64);
            r.set("rank", total_k as f64);
            r.set("chi2_bp", bp);
            r.set("df_bp", (m * (m - 1) / 2) as f64);
            if small {
                r.set("df_r", df_small);
            }
            for (i, (_, p, rmse, r2, chi2, _)) in eqstats.iter().enumerate() {
                let k = i + 1;
                r.set(&format!("rmse_{k}"), *rmse);
                r.set(&format!("r2_{k}"), *r2);
                r.set(&format!("chi2_{k}"), *chi2);
                r.set(&format!("df_m{k}"), *p as f64);
                r.set(&format!("p_{k}"), xm::chi2p(*p as f64, *chi2));
            }
            r.set_mat("Sigma", sigma.clone().with_names(snames.clone(), snames.clone()));
            r.set_str("depvar", deps.clone());
            r.set_str("eqnames", eqn.clone());
            r.set_str("method", method);
            r.set_str("title", title);
            if verb == "reg3" {
                r.set_str("endog", endog.join(" "));
                r.set_str("exog", exog.join(" "));
            }
            r.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

#[allow(dead_code)]
fn unused(_: &Design) {}
