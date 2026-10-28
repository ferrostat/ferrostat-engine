//! Sample-selection and bivariate probit models: `heckman` (maximum
//! likelihood and the two-step estimator), `heckprobit` and `biprobit`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::choice_mixed::{f2, f4, hrow, ln_bvn, ln_phi_d, num_hess, row_ci, row_z, sep, zcrit};
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood, MlResult, Vce};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::zip::MultiIndex;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Probit fit of `y` (nonzero = success) on columns `cols`.
pub fn probit_fit(cols: &[&[f64]], y: &[f64]) -> Result<MlResult, String> {
    let w = vec![1.0; y.len()];
    let f = |t: usize, idx: &[f64]| {
        let q = if y[t] != 0.0 { 1.0 } else { -1.0 };
        let z = q * idx[0];
        let (l, d) = ln_phi_d(z);
        (l, vec![q * d], vec![vec![-d * (z + d)]])
    };
    let m = MultiIndex { x: vec![cols.to_vec()], naux: 0, w: &w, fweight: false, f: &f };
    let mut b0 = vec![0.0; cols.len()];
    // start the constant (if any) at the probit of the mean
    let n = y.len() as f64;
    let my = y.iter().filter(|v| **v != 0.0).count() as f64 / n;
    if let Some(c) = cols.iter().position(|c| c.iter().all(|v| *v == 1.0)) {
        b0[c] = dist::invnormal(my.clamp(1e-4, 1.0 - 1e-4));
    }
    crate::choice_mixed::maximize_tight(&m, b0, 300)
}

fn idx_of(cols: &[&[f64]], b: &[f64], t: usize) -> f64 {
    cols.iter().zip(b).map(|(c, bj)| c[t] * bj).sum()
}

/// Wald test that the coefficients at positions `idx` are zero.
pub fn wald(b: &[f64], v: &Matrix, idx: &[usize]) -> (f64, f64) {
    let k = idx.len();
    if k == 0 {
        return (0.0, 0.0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-14);
    let w: f64 = (0..k).map(|a| (0..k).map(|c| b[idx[a]] * vi[(a, c)] * b[idx[c]]).sum::<f64>()).sum();
    (w, (k - dropped.len()) as f64)
}

/// Split `dep = vars` or `dep vars` into (dep, vars).
fn split_eq(text: &str) -> (Option<String>, String) {
    let t = text.trim();
    if let Some((a, b)) = t.split_once('=') {
        let a = a.trim();
        let a = a.rsplit(':').next().unwrap_or(a).trim();
        return (Some(a.to_string()), b.trim().to_string());
    }
    (None, t.to_string())
}

/// Parts of an equation-list command line: `(eq1) (eq2)`.
fn paren_groups(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for ch in text.chars() {
        match ch {
            '(' => {
                if depth > 0 {
                    cur.push(ch);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            _ if depth > 0 => cur.push(ch),
            _ => {}
        }
    }
    out
}

/// Equation-prefixed names of a design's columns.
fn eq_names(eq: &str, d: &Design) -> Vec<String> {
    d.names.iter().map(|n| format!("{eq}:{n}")).collect()
}

/// Positions in `rows` of each original row (for aligning designs).
fn row_pos(rows: &[usize], n_all: usize) -> Vec<usize> {
    let mut p = vec![usize::MAX; n_all];
    for (t, &r) in rows.iter().enumerate() {
        p[r] = t;
    }
    p
}

// ---------------------------------------------------------------------------
// heckman / heckprobit
// ---------------------------------------------------------------------------

pub struct Heckman {
    pub probit: bool,
}

struct SelData {
    /// outcome design restricted to the selected observations
    dx: Design,
    /// selection design over the estimation sample
    dz: Design,
    /// selection indicator over the estimation sample
    s: Vec<bool>,
    /// position of each estimation-sample row in `dx` (selected only)
    xpos: Vec<usize>,
    dep: String,
    sel_dep: String,
}

fn selection_data(ds: &DataSet, st: &Store, cmd: &Command, binary: bool) -> Result<SelData, String> {
    let main = cmd.main.trim();
    let (dep, xtext) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
    if dep.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let dep = crate::varlist::resolve_name(ds, dep)?;
    let sel = cmd.opt("select", 3).ok_or_else(|| rc_err(198, "select() required"))?;
    let (sel, sel_nocons) = match sel.split_once(',') {
        Some((a, o)) => (a.to_string(), o.trim().to_lowercase().starts_with("nocon")),
        None => (sel.to_string(), false),
    };
    let (sdep, ztext) = split_eq(&sel);
    let sdep = match sdep {
        Some(s) => Some(crate::varlist::resolve_name(ds, &s)?),
        None => None,
    };
    let nocons = cmd.has_opt("noconstant", 3);
    let dz = est::build(ds, st, cmd, DesignSpec { depvar: sdep.as_deref(), indep: &ztext, noconstant: sel_nocons, weights: "", extra: vec![] })?;
    let ycol = util::col(ds, &dep)?;
    let s_all: Vec<bool> = match &sdep {
        Some(_) => dz.y.clone().into_iter().map(|v| v != 0.0).collect(),
        None => dz.rows.iter().map(|&i| !is_missing(ycol[i])).collect(),
    };
    let dx_full = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep: xtext, noconstant: nocons, weights: "", extra: vec![] })?;
    let in_x = row_pos(&dx_full.rows, ds.n_obs);
    // estimation sample: nonselected, or selected with complete outcome data
    let keep: Vec<bool> = dz.rows.iter().zip(&s_all).map(|(&r, &s)| !s || in_x[r] != usize::MAX).collect();
    let dz = crate::xtnl::subset_design(&dz, &keep);
    let s: Vec<bool> = s_all.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| *s).collect();
    let selrow: Vec<bool> = {
        let mut m = vec![false; ds.n_obs];
        for (t, &r) in dz.rows.iter().enumerate() {
            if s[t] {
                m[r] = true;
            }
        }
        m
    };
    let xkeep: Vec<bool> = dx_full.rows.iter().map(|&r| selrow[r]).collect();
    let mut dx = crate::xtnl::subset_design(&dx_full, &xkeep);
    if binary {
        for v in dx.y.iter_mut() {
            *v = if *v != 0.0 { 1.0 } else { 0.0 };
        }
    }
    let xp = row_pos(&dx.rows, ds.n_obs);
    let xpos: Vec<usize> = dz.rows.iter().map(|&r| xp[r]).collect();
    let n1 = s.iter().filter(|v| **v).count();
    if n1 == 0 || n1 == s.len() {
        return Err(rc_err(2000, "no observations (the selection equation needs both selected and nonselected observations)"));
    }
    let sel_dep = sdep.unwrap_or_else(|| "select".to_string());
    Ok(SelData { dx, dz, s, xpos, dep, sel_dep })
}

/// Two-step estimates: (b over [x active, lambda], V, sigma, rho, probit fit).
struct TwoStep {
    beta: Vec<f64>,
    v: Matrix,
    sigma: f64,
    rho: f64,
}

fn two_step(sd: &SelData, xact: &[usize], zcols: &[&[f64]], pr: &MlResult) -> Result<TwoStep, String> {
    let n = sd.s.len();
    let zg: Vec<f64> = (0..n).map(|t| idx_of(zcols, &pr.b, t)).collect();
    let lam: Vec<f64> = zg.iter().map(|&z| ln_phi_d(z).1).collect();
    let sel: Vec<usize> = (0..n).filter(|&t| sd.s[t]).collect();
    let k = xact.len() + 1;
    let xs = |t: usize, j: usize| -> f64 { if j < xact.len() { sd.dx.cols[xact[j]][sd.xpos[t]] } else { lam[t] } };
    let mut xtx = Matrix::zeros(k, k);
    let mut xty = vec![0.0; k];
    for &t in &sel {
        let y = sd.dx.y[sd.xpos[t]];
        for a in 0..k {
            xty[a] += xs(t, a) * y;
            for c in 0..k {
                xtx[(a, c)] += xs(t, a) * xs(t, c);
            }
        }
    }
    let (a, _) = xtx.sym_inverse(1e-14);
    let beta: Vec<f64> = (0..k).map(|i| (0..k).map(|j| a[(i, j)] * xty[j]).sum()).collect();
    let n1 = sel.len() as f64;
    let mut rss = 0.0;
    let mut dsum = 0.0;
    let delta: Vec<f64> = (0..n).map(|t| lam[t] * (lam[t] + zg[t])).collect();
    for &t in &sel {
        let e = sd.dx.y[sd.xpos[t]] - (0..k).map(|j| xs(t, j) * beta[j]).sum::<f64>();
        rss += e * e;
        dsum += delta[t];
    }
    let bl = beta[k - 1];
    let sigma = (rss / n1 + bl * bl * dsum / n1).sqrt();
    let rho = bl / sigma;
    // Heckman's variance
    let kz = zcols.len();
    let (vg, _) = pr.info.sym_inverse(1e-14);
    let mut m1 = Matrix::zeros(k, k);
    let mut xdz = Matrix::zeros(k, kz);
    for &t in &sel {
        for a in 0..k {
            for c in 0..k {
                m1[(a, c)] += (1.0 - rho * rho * delta[t]) * xs(t, a) * xs(t, c);
            }
            for c in 0..kz {
                xdz[(a, c)] += delta[t] * xs(t, a) * zcols[c][t];
            }
        }
    }
    let q = xdz.mul(&vg)?.mul(&xdz.transpose())?.scale(rho * rho);
    let mid = m1.add(&q)?;
    let v = a.mul(&mid)?.mul(&a)?.scale(sigma * sigma);
    Ok(TwoStep { beta, v, sigma, rho })
}

/// Heckman ML contribution of a selected observation: (ll, gradient) in
/// the indexes (xb, zg, athrho, lnsigma).
fn heck_sel(y: f64, v: &[f64]) -> (f64, Vec<f64>) {
    let (xb, zg, a, ls) = (v[0], v[1], v[2], v[3]);
    let rho = a.tanh();
    let sig = ls.exp();
    let e = (y - xb) / sig;
    let r = 1.0 / a.cosh();
    let q = (zg + rho * e) / r;
    let (lp, lam) = ln_phi_d(q);
    let ll = lp - 0.5 * e * e - ls - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let g = vec![-lam * rho / (r * sig) + e / sig, lam / r, lam * (e * r + q * rho), -lam * rho * e / r + e * e - 1.0];
    (ll, g)
}

/// Selected-observation contribution of the probit selection model in
/// (xb, zg, athrho).
fn hprobit_sel(y: f64, v: &[f64]) -> (f64, Vec<f64>) {
    let q = if y != 0.0 { 1.0 } else { -1.0 };
    let rho = v[2].tanh();
    let (l, dh, dk, dr) = ln_bvn(q * v[0], v[1], q * rho);
    (l, vec![q * dh, dk, q * dr * (1.0 - rho * rho)])
}

/// Bivariate probit contribution in (xb1, xb2, athrho).
fn biprobit_obs(y1: f64, y2: f64, v: &[f64]) -> (f64, Vec<f64>) {
    let q1 = if y1 != 0.0 { 1.0 } else { -1.0 };
    let q2 = if y2 != 0.0 { 1.0 } else { -1.0 };
    let rho = v[2].tanh();
    let (l, dh, dk, dr) = ln_bvn(q1 * v[0], q2 * v[1], q1 * q2 * rho);
    (l, vec![q1 * dh, q2 * dk, q1 * q2 * dr * (1.0 - rho * rho)])
}

impl CommandHandler for Heckman {
    fn name(&self) -> &'static str {
        if self.probit { "heckprobit" } else { "heckman" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some(verb) {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let sd = selection_data(ds, st, cmd, self.probit)?;
        let level = util::level(cmd, st)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let twostep = cmd.has_opt("twostep", 3);
        if twostep && self.probit {
            return Err(rc_err(198, "option twostep not allowed"));
        }
        if twostep && vce != Vce::Oim {
            return Err(rc_err(198, "vce() not allowed with twostep"));
        }
        let n = sd.s.len();
        let n1 = sd.s.iter().filter(|v| **v).count();
        let xact = sd.dx.active();
        let zact = sd.dz.active();
        let zcols: Vec<&[f64]> = zact.iter().map(|&j| sd.dz.cols[j].as_slice()).collect();
        // x columns over the estimation sample (zero for nonselected)
        let xfull: Vec<Vec<f64>> = xact.iter().map(|&j| (0..n).map(|t| if sd.s[t] { sd.dx.cols[j][sd.xpos[t]] } else { 0.0 }).collect()).collect();
        let xcols: Vec<&[f64]> = xfull.iter().map(|c| c.as_slice()).collect();
        let yfull: Vec<f64> = (0..n).map(|t| if sd.s[t] { sd.dx.y[sd.xpos[t]] } else { 0.0 }).collect();
        let sf: Vec<f64> = sd.s.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();
        let mut out = Vec::new();
        // selection probit
        let pr = probit_fit(&zcols, &sf)?;
        let kx = xact.len();
        let kz = zact.len();
        let (kxt, kzt) = (sd.dx.k(), sd.dz.k());
        let mut names = eq_names(&sd.dep, &sd.dx);
        names.extend(eq_names("select", &sd.dz));
        let mut omitted = sd.dx.omitted.clone();
        omitted.extend(sd.dz.omitted.iter().cloned());
        let mut is_base = sd.dx.is_base.clone();
        is_base.extend(sd.dz.is_base.iter().cloned());
        let mut idx: Vec<usize> = xact.clone();
        idx.extend(zact.iter().map(|&j| kxt + j));
        let cluster = est::cluster_values(ds, &sd.dz, &vce)?;
        let cons_x = xact.iter().position(|&j| sd.dx.names[j] == "_cons");
        let wald_idx: Vec<usize> = (0..kx).filter(|&a| Some(a) != cons_x).collect();
        if twostep {
            let ts = two_step(&sd, &xact, &zcols, &pr)?;
            let (vg, _) = pr.info.sym_inverse(1e-14);
            names.push("/mills:lambda".into());
            omitted.push(false);
            is_base.push(false);
            idx.push(kxt + kzt);
            let p = kx + kz + 1;
            let mut bt = vec![0.0; p];
            let mut vt = Matrix::zeros(p, p);
            for a in 0..kx {
                bt[a] = ts.beta[a];
                for c in 0..kx {
                    vt[(a, c)] = ts.v[(a, c)];
                }
                vt[(a, p - 1)] = ts.v[(a, kx)];
                vt[(p - 1, a)] = ts.v[(kx, a)];
            }
            bt[p - 1] = ts.beta[kx];
            vt[(p - 1, p - 1)] = ts.v[(kx, kx)];
            for a in 0..kz {
                bt[kx + a] = pr.b[a];
                for c in 0..kz {
                    vt[(kx + a, kx + c)] = vg[(a, c)];
                }
            }
            let total = kxt + kzt + 1;
            let mut b = vec![0.0; total];
            for (a, &j) in idx.iter().enumerate() {
                b[j] = bt[a];
            }
            let v = est::expand_v(&vt, &idx, total);
            let (chi2, df_m) = wald(&bt, &vt, &wald_idx);
            let rho = ts.rho;
            let rho_d = rho.clamp(-1.0, 1.0);
            let lambda = ts.beta[kx];
            let selam = ts.v[(kx, kx)].sqrt();
            out.push(hrow("Heckman selection model -- two-step estimates", "Number of obs", util::commas(n)));
            out.push(format!("{:<54}{:<11} = {:>10}", "(regression model with sample selection)", "Selected", util::commas(n1)));
            out.push(format!("{:<54}{:<11} = {:>10}", "", "Nonselected", util::commas(n - n1)));
            out.push(String::new());
            out.push(hrow("", &format!("Wald chi2({})", fmt_g(df_m, 9)), f2(chi2)));
            out.push(hrow("", "Prob > chi2", f4(dist::chi2tail(df_m, chi2))));
            out.push(String::new());
            let mut tbl = est::coef_table(&est::TableSpec { depvar: &sd.dep, names: &names, b: &b, v: &v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" });
            let bar = tbl.pop().unwrap_or_default();
            let lw = bar.len().saturating_sub(66);
            tbl.push(sep(lw));
            tbl.push(format!("{:>lw$} | {:>10}", "rho", format!("{:.5}", rho_d)));
            tbl.push(format!("{:>lw$} | {:>10}", "sigma", fmt_g(ts.sigma, 8)));
            tbl.push(bar);
            out.extend(tbl);
            let text = out.join("\n");
            let mut dpost = sd.dz.clone();
            dpost.names = names.clone();
            dpost.omitted = omitted;
            dpost.is_base = is_base;
            dpost.depvar = sd.dep.clone();
            let sel_dep = sd.sel_dep.clone();
            est::post(st, verb, &dpost, &b, &v, &|e: &mut Results| {
                e.set("N_selected", n1 as f64);
                e.set("N_nonselected", (n - n1) as f64);
                e.set("chi2", chi2);
                e.set("df_m", df_m);
                e.set("p", dist::chi2tail(df_m, chi2));
                e.set("rho", rho_d);
                e.set("sigma", ts.sigma);
                e.set("lambda", lambda);
                e.set("selambda", selam);
                e.set("k_eq", 3.0);
                e.set_str("method", "twostep");
                e.set_str("title", "Heckman selection model -- two-step estimates");
                e.set_str("title2", "(regression model with sample selection)");
                e.set_str("depvar", format!("{} {}", dpost.depvar, sel_dep));
                e.set_str("chi2type", "Wald");
                e.set_str("vce", "conventional");
                e.set_str("predict", "heckman_p");
                e.set_str("_output", text.clone());
            });
            mills_var(ds, st, cmd, &sd, &zcols, &pr.b)?;
            return Ok(text);
        }
        // ---- maximum likelihood ----
        let w1 = vec![1.0; n];
        let s = &sd.s;
        let (res, ll_c, b0, naux) = if self.probit {
            let yo: Vec<f64> = (0..n).filter(|&t| s[t]).map(|t| yfull[t]).collect();
            let xo: Vec<Vec<f64>> = xfull.iter().map(|c| (0..n).filter(|&t| s[t]).map(|t| c[t]).collect()).collect();
            let xor: Vec<&[f64]> = xo.iter().map(|c| c.as_slice()).collect();
            let po = probit_fit(&xor, &yo)?;
            out.push("Fitting probit model:".into());
            out.push(String::new());
            out.extend(est::iteration_log(&po.iterations, "Log likelihood"));
            out.push(String::new());
            out.push("Fitting selection model:".into());
            out.push(String::new());
            out.extend(est::iteration_log(&pr.iterations, "Log likelihood"));
            out.push(String::new());
            out.push(format!("Comparison:    Log likelihood = {}", fmt_g(po.ll + pr.ll, 10)));
            out.push(String::new());
            out.push("Fitting full model:".into());
            out.push(String::new());
            let mut b0 = po.b.clone();
            b0.extend(pr.b.iter().cloned());
            b0.push(0.0);
            let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>, Vec<Vec<f64>>) {
                if s[t] {
                    let y = yfull[t];
                    let (l, g) = hprobit_sel(y, v);
                    (l, g, num_hess(&|u: &[f64]| hprobit_sel(y, u).1, v))
                } else {
                    let (l, d) = ln_phi_d(-v[1]);
                    let z = -v[1];
                    let mut h = vec![vec![0.0; 3]; 3];
                    h[1][1] = -d * (z + d);
                    (l, vec![0.0, -d, 0.0], h)
                }
            };
            let m = MultiIndex { x: vec![xcols.clone(), zcols.clone()], naux: 1, w: &w1, fweight: false, f: &f };
            let res = crate::choice_mixed::maximize_tight(&m, b0.clone(), 300)?;
            (res, po.ll + pr.ll, b0, 1)
        } else {
            let ts = two_step(&sd, &xact, &zcols, &pr)?;
            let mut b0: Vec<f64> = ts.beta[..kx].to_vec();
            b0.extend(pr.b.iter().cloned());
            b0.push(ts.rho.clamp(-0.85, 0.85).atanh());
            b0.push(ts.sigma.max(1e-8).ln());
            // regression on the selected observations (for the LR test)
            let mut rss = 0.0;
            {
                let mut xtx = Matrix::zeros(kx, kx);
                let mut xty = vec![0.0; kx];
                for t in (0..n).filter(|&t| s[t]) {
                    for a in 0..kx {
                        xty[a] += xcols[a][t] * yfull[t];
                        for c in 0..kx {
                            xtx[(a, c)] += xcols[a][t] * xcols[c][t];
                        }
                    }
                }
                let (xi, _) = xtx.sym_inverse(1e-14);
                let bo: Vec<f64> = (0..kx).map(|i| (0..kx).map(|j| xi[(i, j)] * xty[j]).sum()).collect();
                for t in (0..n).filter(|&t| s[t]) {
                    let e = yfull[t] - idx_of(&xcols, &bo, t);
                    rss += e * e;
                }
            }
            let nn1 = n1 as f64;
            let ll_reg = -0.5 * nn1 * ((2.0 * std::f64::consts::PI).ln() + (rss / nn1).ln() + 1.0);
            let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>, Vec<Vec<f64>>) {
                if s[t] {
                    let y = yfull[t];
                    let (l, g) = heck_sel(y, v);
                    (l, g, num_hess(&|u: &[f64]| heck_sel(y, u).1, v))
                } else {
                    let (l, d) = ln_phi_d(-v[1]);
                    let z = -v[1];
                    let mut h = vec![vec![0.0; 4]; 4];
                    h[1][1] = -d * (z + d);
                    (l, vec![0.0, -d, 0.0, 0.0], h)
                }
            };
            let m = MultiIndex { x: vec![xcols.clone(), zcols.clone()], naux: 2, w: &w1, fweight: false, f: &f };
            let res = crate::choice_mixed::maximize_tight(&m, b0.clone(), 300)?;
            (res, pr.ll + ll_reg, b0, 2)
        };
        let _ = b0;
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        let p = kx + kz + naux;
        let (vt, n_clust) = crate::zip::ml_vce(&res, &vce, cluster.as_deref(), n as f64);
        let bt = res.b.clone();
        let aux_names: Vec<&str> = if naux == 2 { vec!["athrho", "lnsigma"] } else { vec!["athrho"] };
        for (r, a) in aux_names.iter().enumerate() {
            names.push(format!("/:{a}"));
            omitted.push(false);
            is_base.push(false);
            idx.push(kxt + kzt + r);
        }
        let total = kxt + kzt + naux;
        let mut b = vec![0.0; total];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = bt[a];
        }
        let v = est::expand_v(&vt, &idx, total);
        let (chi2, df_m) = wald(&bt, &vt, &wald_idx);
        let ia = kx + kz;
        let ath = bt[ia];
        let se_a = vt[(ia, ia)].max(0.0).sqrt();
        let rho = ath.tanh();
        let robust = vce != Vce::Oim;
        let (chi2_c, test_c) = if robust { ((ath / se_a).powi(2), "Wald") } else { ((2.0 * (res.ll - ll_c)).max(0.0), "LR") };
        let p_c = dist::chi2tail(1.0, chi2_c);
        let crit = zcrit(level);
        let title = if self.probit { "Probit model with sample selection" } else { "Heckman selection model" };
        out.push(hrow(title, "Number of obs", util::commas(n)));
        if !self.probit {
            out.push(format!("{:<54}{:<11} = {:>10}", "(regression model with sample selection)", "Selected", util::commas(n1)));
        } else {
            out.push(format!("{:<54}{:<11} = {:>10}", "", "Selected", util::commas(n1)));
        }
        out.push(format!("{:<54}{:<11} = {:>10}", "", "Nonselected", util::commas(n - n1)));
        out.push(String::new());
        out.push(hrow("", &format!("Wald chi2({})", fmt_g(df_m, 9)), f2(chi2)));
        out.push(hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(res.ll, 9)), "Prob > chi2", f4(dist::chi2tail(df_m, chi2))));
        if let Some(g) = n_clust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), match &vce { Vce::Cluster(c) => c.as_str(), _ => "" })));
        }
        out.push(String::new());
        let ncoef = kxt + kzt;
        let mut tbl = est::coef_table(&est::TableSpec {
            depvar: &sd.dep,
            names: &names[..ncoef],
            b: &b[..ncoef],
            v: &v,
            omitted: &omitted[..ncoef],
            is_base: &is_base[..ncoef],
            df: None,
            level,
            eform: None,
            coef_title: "Coefficient",
        });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(sep(lw));
        for (r, a) in aux_names.iter().enumerate() {
            let j = ia + r;
            tbl.push(row_z(lw, &format!("/{a}"), bt[j], vt[(j, j)].max(0.0).sqrt(), crit));
        }
        tbl.push(sep(lw));
        tbl.push(row_ci(lw, "rho", rho, (1.0 - rho * rho) * se_a, (ath - crit * se_a).tanh(), (ath + crit * se_a).tanh()));
        let mut extra: Vec<(&str, f64)> = vec![("rho", rho)];
        let mut selam = SYSMISS;
        if naux == 2 {
            let ls = bt[ia + 1];
            let se_s = vt[(ia + 1, ia + 1)].max(0.0).sqrt();
            let sigma = ls.exp();
            let lambda = rho * sigma;
            let ga = sigma * (1.0 - rho * rho);
            let gs = lambda;
            selam = (ga * ga * vt[(ia, ia)] + 2.0 * ga * gs * vt[(ia, ia + 1)] + gs * gs * vt[(ia + 1, ia + 1)]).max(0.0).sqrt();
            tbl.push(row_ci(lw, "sigma", sigma, sigma * se_s, (ls - crit * se_s).exp(), (ls + crit * se_s).exp()));
            tbl.push(row_ci(lw, "lambda", lambda, selam, lambda - crit * selam, lambda + crit * selam));
            extra.push(("sigma", sigma));
            extra.push(("lambda", lambda));
        }
        tbl.push(bar);
        out.extend(tbl);
        out.push(format!("{test_c} test of indep. eqns. (rho = 0):   chi2(1) = {:>8}   Prob > chi2 = {:.4}", f2(chi2_c), p_c));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let mut dpost = sd.dz.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        dpost.depvar = sd.dep.clone();
        let sel_dep = sd.sel_dep.clone();
        let probit = self.probit;
        est::post(st, verb, &dpost, &b, &v, &|e: &mut Results| {
            e.set("N_selected", n1 as f64);
            e.set("N_nonselected", (n - n1) as f64);
            e.set("ll", res.ll);
            e.set("ll_c", ll_c);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("chi2_c", chi2_c);
            e.set("p_c", p_c);
            e.set("k", p as f64);
            e.set("k_eq", (2 + naux) as f64);
            e.set("k_aux", naux as f64);
            e.set("rank", p as f64);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            for (k, v) in &extra {
                e.set(k, *v);
            }
            if !probit {
                e.set("selambda", selam);
            }
            e.set_str("method", "ml");
            e.set_str("title", title);
            e.set_str("depvar", format!("{} {}", dpost.depvar, sel_dep));
            e.set_str("chi2type", "Wald");
            e.set_str("chi2_ct", test_c);
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("predict", if probit { "heckprobit_p" } else { "heckman_p" });
            e.set_str("_output", text.clone());
        });
        mills_var(ds, st, cmd, &sd, &zcols, &res.b[kx..kx + kz])?;
        Ok(text)
    }
}

/// `nshazard()` / `mills()`: inverse Mills ratio of the selection equation.
fn mills_var(ds: &mut DataSet, st: &Store, cmd: &Command, sd: &SelData, zcols: &[&[f64]], g: &[f64]) -> Result<(), String> {
    let _ = st;
    let Some(name) = cmd.opt("nshazard", 3).or_else(|| cmd.opt("mills", 5)) else { return Ok(()) };
    let name = name.trim();
    crate::varlist::check_name(name)?;
    let mut vals = vec![SYSMISS; ds.n_obs];
    for (t, &r) in sd.dz.rows.iter().enumerate() {
        vals[r] = ln_phi_d(idx_of(zcols, g, t)).1;
    }
    ds.add_variable(util::new_double(name, vals)).map_err(|e| rc_err(110, e))
}

// ---------------------------------------------------------------------------
// biprobit
// ---------------------------------------------------------------------------

pub struct Biprobit;

impl CommandHandler for Biprobit {
    fn name(&self) -> &'static str {
        "biprobit"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("biprobit") {
                return Err(rc_err(301, "last estimates not found"));
            }
            est::stash_table_from_e(&st.e, st.level);
            return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
        }
        let main = cmd.main.trim();
        let sur = main.starts_with('(');
        let eqs: Vec<(String, String)> = if sur {
            let g = paren_groups(main);
            if g.len() != 2 {
                return Err(rc_err(198, "two equations required"));
            }
            g.iter()
                .map(|e| {
                    let (d, r) = split_eq(e);
                    match d {
                        Some(d) => Ok((d, r)),
                        None => {
                            let t = e.trim();
                            let (d, r) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
                            Ok((d.to_string(), r.trim().to_string()))
                        }
                    }
                })
                .collect::<Result<_, String>>()?
        } else {
            let w: Vec<&str> = main.split_whitespace().collect();
            if w.len() < 2 {
                return Err(rc_err(102, "too few variables specified"));
            }
            let rest = w[2..].join(" ");
            vec![(w[0].to_string(), rest.clone()), (w[1].to_string(), rest)]
        };
        let dep1 = crate::varlist::resolve_name(ds, &eqs[0].0)?;
        let dep2 = crate::varlist::resolve_name(ds, &eqs[1].0)?;
        let nocons = cmd.has_opt("noconstant", 3);
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let level = util::level(cmd, st)?;
        // common sample: complete on both equations
        let d1 = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep1), indep: &eqs[0].1, noconstant: nocons, weights: "", extra: vec![dep2.clone()] })?;
        let d2 = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep2), indep: &eqs[1].1, noconstant: nocons, weights: "", extra: vec![dep1.clone()] })?;
        let p2 = row_pos(&d2.rows, ds.n_obs);
        let keep1: Vec<bool> = d1.rows.iter().map(|&r| p2[r] != usize::MAX).collect();
        let mut d1 = crate::xtnl::subset_design(&d1, &keep1);
        let p1 = row_pos(&d1.rows, ds.n_obs);
        let keep2: Vec<bool> = d2.rows.iter().map(|&r| p1[r] != usize::MAX).collect();
        let mut d2 = crate::xtnl::subset_design(&d2, &keep2);
        for v in d1.y.iter_mut().chain(d2.y.iter_mut()) {
            *v = if *v != 0.0 { 1.0 } else { 0.0 };
        }
        let n = d1.n();
        if n == 0 {
            return Err(rc_err(2000, "no observations"));
        }
        let (a1, a2) = (d1.active(), d2.active());
        let c1: Vec<&[f64]> = a1.iter().map(|&j| d1.cols[j].as_slice()).collect();
        let c2: Vec<&[f64]> = a2.iter().map(|&j| d2.cols[j].as_slice()).collect();
        let mut out = Vec::new();
        let f1 = probit_fit(&c1, &d1.y)?;
        let f2_ = probit_fit(&c2, &d2.y)?;
        out.push("Fitting comparison equation 1:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&f1.iterations, "Log likelihood"));
        out.push(String::new());
        out.push("Fitting comparison equation 2:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&f2_.iterations, "Log likelihood"));
        out.push(String::new());
        let ll_c = f1.ll + f2_.ll;
        out.push(format!("Comparison:    Log likelihood = {}", fmt_g(ll_c, 10)));
        out.push(String::new());
        out.push("Fitting full model:".into());
        out.push(String::new());
        let mut b0 = f1.b.clone();
        b0.extend(f2_.b.iter().cloned());
        b0.push(0.0);
        let (y1, y2) = (&d1.y, &d2.y);
        let w1 = vec![1.0; n];
        let f = |t: usize, v: &[f64]| -> (f64, Vec<f64>, Vec<Vec<f64>>) {
            let (l, g) = biprobit_obs(y1[t], y2[t], v);
            (l, g, num_hess(&|u: &[f64]| biprobit_obs(y1[t], y2[t], u).1, v))
        };
        let m = MultiIndex { x: vec![c1.clone(), c2.clone()], naux: 1, w: &w1, fweight: false, f: &f };
        let res = crate::choice_mixed::maximize_tight(&m, b0, 300)?;
        out.extend(est::iteration_log(&res.iterations, "Log likelihood"));
        out.push(String::new());
        let cluster = est::cluster_values(ds, &d1, &vce)?;
        let (vt, n_clust) = crate::zip::ml_vce(&res, &vce, cluster.as_deref(), n as f64);
        let (k1, k2) = (a1.len(), a2.len());
        let (k1t, k2t) = (d1.k(), d2.k());
        let mut names = eq_names(&dep1, &d1);
        names.extend(eq_names(&dep2, &d2));
        names.push("/:athrho".into());
        let mut omitted = d1.omitted.clone();
        omitted.extend(d2.omitted.iter().cloned());
        omitted.push(false);
        let mut is_base = d1.is_base.clone();
        is_base.extend(d2.is_base.iter().cloned());
        is_base.push(false);
        let mut idx: Vec<usize> = a1.clone();
        idx.extend(a2.iter().map(|&j| k1t + j));
        idx.push(k1t + k2t);
        let total = k1t + k2t + 1;
        let mut b = vec![0.0; total];
        for (a, &j) in idx.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vt, &idx, total);
        let wald_idx: Vec<usize> = (0..k1).filter(|&a| d1.names[a1[a]] != "_cons").chain((0..k2).filter(|&a| d2.names[a2[a]] != "_cons").map(|a| k1 + a)).collect();
        let (chi2, df_m) = wald(&res.b, &vt, &wald_idx);
        let ia = k1 + k2;
        let ath = res.b[ia];
        let se_a = vt[(ia, ia)].max(0.0).sqrt();
        let rho = ath.tanh();
        let robust = vce != Vce::Oim;
        let (chi2_c, test_c) = if robust { ((ath / se_a).powi(2), "Wald") } else { ((2.0 * (res.ll - ll_c)).max(0.0), "LR") };
        let p_c = dist::chi2tail(1.0, chi2_c);
        let crit = zcrit(level);
        let title = if sur { "Seemingly unrelated bivariate probit" } else { "Bivariate probit regression" };
        out.push(hrow(title, "Number of obs", util::commas(n)));
        out.push(hrow("", &format!("Wald chi2({})", fmt_g(df_m, 9)), f2(chi2)));
        out.push(hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(res.ll, 9)), "Prob > chi2", f4(dist::chi2tail(df_m, chi2))));
        if let Some(g) = n_clust {
            out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {})", util::commas(g), match &vce { Vce::Cluster(c) => c.as_str(), _ => "" })));
        }
        out.push(String::new());
        let ncoef = k1t + k2t;
        let mut tbl = est::coef_table(&est::TableSpec { depvar: "", names: &names[..ncoef], b: &b[..ncoef], v: &v, omitted: &omitted[..ncoef], is_base: &is_base[..ncoef], df: None, level, eform: None, coef_title: "Coefficient" });
        let bar = tbl.pop().unwrap_or_default();
        let lw = bar.len().saturating_sub(66);
        tbl.push(sep(lw));
        tbl.push(row_z(lw, "/athrho", ath, se_a, crit));
        tbl.push(sep(lw));
        tbl.push(row_ci(lw, "rho", rho, (1.0 - rho * rho) * se_a, (ath - crit * se_a).tanh(), (ath + crit * se_a).tanh()));
        tbl.push(bar);
        out.extend(tbl);
        out.push(format!("{test_c} test of rho=0: chi2(1) = {}{:>27}", fmt_g(chi2_c, 6), format!("Prob > chi2 = {:.4}", p_c)));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let mut dpost = d1.clone();
        dpost.names = names;
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        let p = k1 + k2 + 1;
        est::post(st, "biprobit", &dpost, &b, &v, &|e: &mut Results| {
            e.set("ll", res.ll);
            e.set("ll_c", ll_c);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", dist::chi2tail(df_m, chi2));
            e.set("chi2_c", chi2_c);
            e.set("p_c", p_c);
            e.set("rho", rho);
            e.set("k", p as f64);
            e.set("rank", p as f64);
            e.set("k_eq", 3.0);
            e.set("k_aux", 1.0);
            e.set("ic", (res.iterations.len() - 1) as f64);
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            if let Some(g) = n_clust {
                e.set("N_clust", g as f64);
            }
            e.set_str("depvar", format!("{dep1} {dep2}"));
            e.set_str("title", title);
            e.set_str("chi2type", "Wald");
            e.set_str("chi2_ct", test_c);
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("predict", "biprobit_p");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// `predict` after heckman, heckprobit and biprobit.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let name = match words.as_slice() {
        [n] => n.to_string(),
        [_, n] => n.to_string(),
        _ => return Err(rc_err(198, "invalid syntax")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let known = ["xb", "xbsel", "stdp", "stdpsel", "ycond", "yexpected", "psel", "pmargin", "pcond", "p11", "p10", "p01", "p00", "pmarg1", "pmarg2", "pcond1", "pcond2", "xb1", "xb2", "mills", "nshazard"];
    let stat = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).find(|n| known.contains(&n.as_str()));
    let (stat, msg) = match stat {
        Some(s) => (s, String::new()),
        None => match ecmd.as_str() {
            "heckman" => ("xb".to_string(), "(option xb assumed; fitted values)".to_string()),
            "heckprobit" => ("pmargin".to_string(), format!("(option pmargin assumed; Pr({}))", st.e.macros.get("depvar").and_then(|d| d.split_whitespace().next().map(String::from)).unwrap_or_default())),
            _ => ("p11".to_string(), "(option p11 assumed; Pr(depvar1=1, depvar2=1))".to_string()),
        },
    };
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let eqs: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for c in &b.colnames {
            if let Some((e, _)) = c.split_once(':') {
                if !e.starts_with('/') && !v.iter().any(|x| x == e) {
                    v.push(e.to_string());
                }
            }
        }
        v
    };
    let rho = st.e.get("rho").unwrap_or(0.0);
    let sigma = st.e.get("sigma").unwrap_or(SYSMISS);
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let env = Env::with_store(ds, st);
    let mut vals = vec![SYSMISS; n];
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let x1 = crate::postest::xb_row(&env, &b, Some(&eqs[0]), i)?;
        let x2 = if eqs.len() > 1 { crate::postest::xb_row(&env, &b, Some(&eqs[1]), i)? } else { SYSMISS };
        let need2 = !matches!(stat.as_str(), "xb" | "xb1" | "pmargin" | "pmarg1" | "stdp");
        let need1 = !matches!(stat.as_str(), "xbsel" | "xb2" | "psel" | "pmarg2" | "mills" | "nshazard" | "stdpsel");
        if (need1 && is_missing(x1)) || (need2 && is_missing(x2)) {
            continue;
        }
        let pp = |a: f64, c: f64, r: f64| crate::choice_mixed::bvn(a, c, r);
        vals[i] = match (ecmd.as_str(), stat.as_str()) {
            (_, "xb") | (_, "xb1") => x1,
            (_, "xbsel") | (_, "xb2") => x2,
            (_, "stdp") | (_, "stdpsel") => {
                let eq = if stat == "stdp" { &eqs[0] } else { &eqs[1] };
                let x: Vec<f64> = b.colnames.iter().map(|c| if c.split_once(':').map(|x| x.0) == Some(eq.as_str()) { crate::postest::coef_value(&env, c, i) } else { Ok(0.0) }).collect::<Result<_, _>>()?;
                let k = x.len();
                (0..k).map(|a| (0..k).map(|c| x[a] * v[(a, c)] * x[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt()
            }
            (_, "psel") => dist::normal(x2),
            (_, "mills") | (_, "nshazard") => ln_phi_d(x2).1,
            ("heckman", "ycond") => x1 + rho * sigma * ln_phi_d(x2).1,
            ("heckman", "yexpected") => dist::normal(x2) * (x1 + rho * sigma * ln_phi_d(x2).1),
            (_, "pmargin") | (_, "pmarg1") => dist::normal(x1),
            ("biprobit", "pmarg2") => dist::normal(x2),
            ("heckprobit", "pcond") => pp(x1, x2, rho) / dist::normal(x2),
            ("biprobit", "pcond1") => pp(x1, x2, rho) / dist::normal(x2),
            ("biprobit", "pcond2") => pp(x1, x2, rho) / dist::normal(x1),
            (_, "p11") => pp(x1, x2, rho),
            (_, "p10") => pp(x1, -x2, -rho),
            (_, "p01") => pp(-x1, x2, -rho),
            (_, "p00") => pp(-x1, -x2, rho),
            _ => return Err(rc_err(198, format!("option {stat} not allowed"))),
        };
    }
    let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
    ds.add_variable(util::new_double(&name, vals)).map_err(|e| rc_err(110, e))?;
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    if nmiss > 0 {
        lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
    }
    Ok(lines.join("\n"))
}

#[allow(dead_code)]
fn unused(_: &dyn Likelihood) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(f: &dyn Fn(&[f64]) -> (f64, Vec<f64>), x: &[f64]) {
        let g = f(x).1;
        for j in 0..x.len() {
            let mut a = x.to_vec();
            a[j] += 1e-6;
            let mut c = x.to_vec();
            c[j] -= 1e-6;
            let num = (f(&a).0 - f(&c).0) / 2e-6;
            assert!((num - g[j]).abs() < 1e-6 * (1.0 + g[j].abs()), "param {j}: {} vs {num}", g[j]);
        }
    }

    #[test]
    fn analytic_gradients() {
        check(&|v| heck_sel(1.7, v), &[0.4, -0.3, 0.6, 0.2]);
        check(&|v| hprobit_sel(1.0, v), &[0.4, -0.3, 0.6]);
        check(&|v| hprobit_sel(0.0, v), &[0.4, 0.8, -1.2]);
        check(&|v| biprobit_obs(0.0, 1.0, v), &[-0.2, 0.5, 0.3]);
    }
}
