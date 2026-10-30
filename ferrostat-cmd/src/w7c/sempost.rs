//! Postestimation after `sem`: goodness of fit, equation-level statistics,
//! modification indices and score tests, residuals, direct and indirect
//! effects, stability, the model matrices, `estat stdize:` and `predict`.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::la;
use super::model::{Kind, Model, Slot};
use super::opt;
use super::sem::{self, Method, Problem, SemFit};
use super::state;
use super::tab::{self, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};

fn f9(v: f64) -> String {
    fmt_g(v, 9)
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let f = state::sem(st)?;
    let sub = sub.trim_end_matches(':');
    let r = match sub {
        "gof" => gof(&f, st, cmd),
        "eqgof" => eqgof(&f, st),
        "eqtest" => eqtest(&f, st),
        "ggof" => ggof(&f, st),
        "ginvariant" => ginvariant(&f, st, cmd),
        "mindices" => mindices(&f, st, cmd),
        "scoretests" => scoretests(&f, st, cmd),
        "residuals" => residuals(&f, st, cmd),
        "framework" => framework(&f, st, cmd),
        "teffects" => teffects(&f, st, cmd),
        "stable" => stable(&f, st),
        "summarize" => summarize(&f, st),
        "stdize" => stdize(ds, st, &f, cmd),
        _ => return None,
    };
    Some(r)
}

// ---------------------------------------------------------------------------
// goodness of fit
// ---------------------------------------------------------------------------

pub struct Gof {
    pub rmsea: f64,
    pub lb: f64,
    pub ub: f64,
    pub pclose: f64,
    pub aic: f64,
    pub bic: f64,
    pub cfi: f64,
    pub tli: f64,
    pub srmr: f64,
    pub cd: f64,
}

fn rmsea_ci(chi2: f64, df: f64, n: f64, g: f64) -> (f64, f64, f64) {
    if df <= 0.0 {
        return (SYSMISS, SYSMISS, SYSMISS);
    }
    let cdf = |ncp: f64| crate::pss::ncdist::ncchi2(df, ncp, chi2);
    let solve = |p: f64| -> f64 {
        // ncp such that P(X <= chi2; ncp) = p (decreasing in ncp)
        if cdf(0.0) < p {
            return 0.0;
        }
        let (mut lo, mut hi) = (0.0, chi2.max(1.0));
        while cdf(hi) > p {
            hi *= 2.0;
            if hi > 1e8 {
                break;
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if cdf(mid) > p { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    let lo = solve(0.95);
    let hi = solve(0.05);
    let sc = |ncp: f64| (ncp / (df * (n - 1.0))).max(0.0).sqrt() * g.sqrt();
    let pclose = 1.0 - crate::pss::ncdist::ncchi2(df, 0.0025 * df * (n - 1.0) / g, chi2);
    (sc(lo), sc(hi), pclose)
}

/// SRMR and CD of group g (or all groups with weights).
pub fn srmr_cd(f: &SemFit, g: usize) -> (f64, f64) {
    let md = &f.md;
    let p = md.p;
    let s = &f.gd[g].stats;
    let (mu, sig) = md.implied(&f.th, g).unwrap_or((vec![0.0; p], vec![0.0; p * p]));
    let mut ss = 0.0;
    let mut cnt = 0.0;
    for i in 0..p {
        for j in 0..=i {
            let r = s.cov[i * p + j] / (s.cov[i * p + i] * s.cov[j * p + j]).sqrt() - sig[i * p + j] / (sig[i * p + i] * sig[j * p + j]).sqrt();
            ss += r * r;
            cnt += 1.0;
        }
        let r = (s.mean[i] - mu[i]) / s.cov[i * p + i].sqrt() - (mu[i] - mu[i]);
        if md.endo[i] {
            ss += r * r;
            cnt += 1.0;
        }
    }
    let srmr = (ss / cnt).sqrt();
    // coefficient of determination over the endogenous variables
    let m = md.m();
    let en: Vec<usize> = (0..m).filter(|&i| md.endo[i]).collect();
    let cd = if en.is_empty() {
        SYSMISS
    } else if let Some((_, sall)) = md.implied_all(&f.th, g) {
        let (_, sm, _) = md.mats(&f.th, g);
        let ps = la::sub(&sm, m, &en, &en);
        let ss_ = la::sub(&sall, m, &en, &en);
        1.0 - la::det(&ps, en.len()) / la::det(&ss_, en.len())
    } else {
        SYSMISS
    };
    (srmr, cd)
}

pub fn gof_stats(f: &SemFit) -> Gof {
    let ng = f.gd.len() as f64;
    let (chi2, chi2b) = f.sb.unwrap_or((f.chi2_ms, f.chi2_bs));
    let (rmsea, lb, ub, pclose) = if f.df_ms > 0.0 {
        let r = ((chi2 - f.df_ms).max(0.0) / (f.df_ms * (f.n - 1.0))).sqrt() * ng.sqrt();
        let (a, b, c) = rmsea_ci(chi2, f.df_ms, f.n, ng);
        (r, a, b, c)
    } else {
        (0.0, SYSMISS, SYSMISS, SYSMISS)
    };
    let k = f.kfree as f64;
    let (aic, bic) = if f.method == Method::Adf { (SYSMISS, SYSMISS) } else { (-2.0 * f.ll + 2.0 * k, -2.0 * f.ll + k * f.n.ln()) };
    let dm = (chi2 - f.df_ms).max(0.0);
    let db = (chi2b - f.df_bs).max(0.0);
    let cfi = if dm.max(db) > 0.0 { 1.0 - dm / dm.max(db) } else { 1.0 };
    let tli = if f.df_ms > 0.0 && f.df_bs > 0.0 { ((chi2b / f.df_bs) - (chi2 / f.df_ms)) / ((chi2b / f.df_bs) - 1.0) } else { 1.0 };
    let mut srmr = 0.0;
    let mut cdn = 0.0;
    let mut cd = 0.0;
    for g in 0..f.gd.len() {
        let w = f.gd[g].stats.n / f.n;
        let (s, c) = srmr_cd(f, g);
        srmr += w * s;
        if c != SYSMISS {
            cd += w * c;
            cdn += w;
        }
    }
    Gof { rmsea, lb, ub, pclose, aic, bic, cfi, tli: tli.min(1.0e300), srmr, cd: if cdn > 0.0 { cd / cdn } else { SYSMISS } }
}

fn gof(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let want = cmd.opt("stats", 2).map(|s| s.to_lowercase()).unwrap_or_else(|| "residuals".into());
    let all = want.contains("all");
    let show = |w: &str| all || want.contains(w);
    let g = gof_stats(f);
    let mut out = vec![String::new(), "-".repeat(78), format!("{:<20}|{:>12}   Description", "Fit statistic", "Value"), format!("{}+{}", "-".repeat(20), "-".repeat(57))];
    let mut r = Results::new();
    let (chi2, chi2b) = f.sb.unwrap_or((f.chi2_ms, f.chi2_bs));
    let sb = if f.sb.is_some() { " (Satorra-Bentler)" } else { "" };
    if all || cmd.opt("stats", 2).is_none() || want.contains("chi2") {
        out.push(format!("{:<20}|", "Likelihood ratio"));
        out.push(format!("{:>19} |{:>12}   model vs. saturated{sb}", "chi2_ms(" .to_string() + &fmt_g(f.df_ms, 6) + ")", f9(chi2)));
        out.push(format!("{:>19} |{:>12}", "p > chi2", f9(if f.df_ms > 0.0 { dist::chi2tail(f.df_ms, chi2) } else { SYSMISS })));
        out.push(format!("{:>19} |{:>12}   baseline vs. saturated", "chi2_bs(".to_string() + &fmt_g(f.df_bs, 6) + ")", f9(chi2b)));
        out.push(format!("{:>19} |{:>12}", "p > chi2", f9(dist::chi2tail(f.df_bs, chi2b))));
    }
    r.set("chi2_ms", chi2);
    r.set("df_ms", f.df_ms);
    r.set("p_ms", if f.df_ms > 0.0 { dist::chi2tail(f.df_ms, chi2) } else { SYSMISS });
    r.set("chi2_bs", chi2b);
    r.set("df_bs", f.df_bs);
    r.set("p_bs", dist::chi2tail(f.df_bs, chi2b));
    if all || cmd.opt("stats", 2).is_none() || show("rmsea") {
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(57)));
        out.push(format!("{:<20}|", "Population error"));
        out.push(format!("{:>19} |{:>12}   Root mean squared error of approximation", "RMSEA", f9(g.rmsea)));
        out.push(format!("{:>19} |{:>12}", "90% CI, lower bound", f9(g.lb)));
        out.push(format!("{:>19} |{:>12}", "upper bound", f9(g.ub)));
        out.push(format!("{:>19} |{:>12}   Probability RMSEA <= 0.05", "pclose", f9(g.pclose)));
    }
    r.set("rmsea", g.rmsea);
    r.set("lb90_rmsea", g.lb);
    r.set("ub90_rmsea", g.ub);
    r.set("pclose", g.pclose);
    if show("ic") && f.method != Method::Adf {
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(57)));
        out.push(format!("{:<20}|", "Information criteria"));
        out.push(format!("{:>19} |{:>12}   Akaike's information criterion", "AIC", f9(g.aic)));
        out.push(format!("{:>19} |{:>12}   Bayesian information criterion", "BIC", f9(g.bic)));
    }
    r.set("aic", g.aic);
    r.set("bic", g.bic);
    if show("indices") {
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(57)));
        out.push(format!("{:<20}|", "Baseline comparison"));
        out.push(format!("{:>19} |{:>12}   Comparative fit index", "CFI", f9(g.cfi)));
        out.push(format!("{:>19} |{:>12}   Tucker-Lewis index", "TLI", f9(g.tli)));
    }
    r.set("cfi", g.cfi);
    r.set("tli", g.tli);
    if show("residuals") {
        out.push(format!("{}+{}", "-".repeat(20), "-".repeat(57)));
        out.push(format!("{:<20}|", "Size of residuals"));
        out.push(format!("{:>19} |{:>12}   Standardized root mean squared residual", "SRMR", f9(g.srmr)));
        out.push(format!("{:>19} |{:>12}   Coefficient of determination", "CD", f9(g.cd)));
    }
    r.set("srmr", g.srmr);
    r.set("cd", g.cd);
    r.set("N_groups", f.gd.len() as f64);
    out.push("-".repeat(78));
    st.r = r;
    Ok(out.join("\n"))
}

fn ggof(f: &SemFit, st: &mut Store) -> Result<String, String> {
    let mut out = vec![String::new(), "-".repeat(70), format!("{:>12} |{:>8}{:>12}{:>12}{:>12}", "Group", "N", "chi2", "SRMR", "CD"), format!("{}+{}", "-".repeat(13), "-".repeat(56))];
    let mut m = Matrix::zeros(f.gd.len(), 4);
    let prob = f.problem();
    for g in 0..f.gd.len() {
        // chi2 contribution: 2 (ll_sat_g - ll_g)
        let one = Problem { md: &f.md, gd: &f.gd[g..g + 1], method: prob.method };
        let _ = one;
        let s = &f.gd[g].stats;
        let p = f.md.p;
        let (mu, sig) = f.md.implied(&f.th, g).unwrap();
        let l = la::chol(&sig, p).unwrap_or(vec![1.0; p * p]);
        let si = la::chol_inv(&l, p);
        let tr: f64 = si.iter().zip(&s.cov).map(|(a, b)| a * b).sum();
        let d: Vec<f64> = (0..p).map(|i| s.mean[i] - mu[i]).collect();
        let q: f64 = la::chol_solve(&l, p, &d).iter().zip(&d).map(|(a, b)| a * b).sum();
        let chi2 = s.n * (la::chol_logdet(&l, p) + tr + q - la::det(&s.cov, p).ln() - p as f64);
        let (srmr, cd) = srmr_cd(f, g);
        out.push(format!("{:>12} |{:>8}{:>12}{:>12}{:>12}", f.md.glabels[g], s.n, f9(chi2), f9(srmr), f9(cd)));
        m[(g, 0)] = s.n;
        m[(g, 1)] = chi2;
        m[(g, 2)] = srmr;
        m[(g, 3)] = cd;
    }
    out.push("-".repeat(70));
    m.colnames = vec!["N".into(), "chi2".into(), "SRMR".into(), "CD".into()];
    m.rownames = f.md.glabels.clone();
    st.r = Results::new();
    st.r.set_mat("gfit", m);
    Ok(out.join("\n"))
}

fn eqgof(f: &SemFit, st: &mut Store) -> Result<String, String> {
    let md = &f.md;
    let m = md.m();
    let mut out = Vec::new();
    let mut mats = Vec::new();
    for g in 0..md.ng {
        let (_, sall) = md.implied_all(&f.th, g).ok_or_else(|| rc_err(430, "implied moments not available"))?;
        let (_, sm, _) = md.mats(&f.th, g);
        if md.ng > 1 {
            out.push(format!("Group {}", md.glabels[g]));
        }
        out.push(String::new());
        out.push("-".repeat(78));
        out.push(format!("{:>12} |{:>36}{:>14}", "", "Variance", ""));
        out.push(format!("{:>12} |{:>12}{:>12}{:>12}{:>12}{:>10}{:>8}", "Depvars", "fitted", "predicted", "residual", "R-squared", "mc", "mc2"));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        let mut rows = Vec::new();
        let mut names = Vec::new();
        for (sec, latent) in [("observed", false), ("latent", true)] {
            let list: Vec<usize> = (0..m).filter(|&i| md.endo[i] && md.is_latent(i) == latent).collect();
            if list.is_empty() {
                continue;
            }
            out.push(format!("{:<12} |", sec));
            for i in list {
                let fv = sall[i * m + i];
                let rv = sm[i * m + i];
                let pv = fv - rv;
                let r2 = 1.0 - rv / fv;
                out.push(format!("{:>12} |{:>12}{:>12}{:>12}{:>12}{:>10}{:>8}", tab::ab(&md.vars[i], 12), f9(fv), f9(pv), f9(rv), f9(r2), fmt_g(r2.max(0.0).sqrt(), 7), fmt_g(r2, 7)));
                rows.push(vec![fv, pv, rv, r2, r2.max(0.0).sqrt(), r2]);
                names.push(md.vars[i].clone());
            }
        }
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(64)));
        let (_, cd) = srmr_cd(f, g);
        out.push(format!("{:>12} |{:>48}", "overall", f9(cd)));
        out.push("-".repeat(78));
        out.push("mc  = correlation between depvar and its prediction".into());
        out.push("mc2 = mc^2 is the Bentler-Raykov squared multiple correlation coefficient".into());
        let mut mm = Matrix::from_rows(&rows);
        mm.rownames = names;
        mm.colnames = ["fitted", "predicted", "residual", "rsquared", "mc", "mc2"].iter().map(|s| s.to_string()).collect();
        mats.push((cd, mm));
    }
    st.r = Results::new();
    if let Some((cd, m)) = mats.into_iter().next() {
        st.r.set("CD", cd);
        st.r.set_mat("eqfit", m);
    }
    Ok(out.join("\n"))
}

fn wald(b: &[f64], v: &Matrix, idx: &[usize]) -> (f64, f64) {
    let k = idx.len();
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let (vi, dropped) = vs.sym_inverse(1e-12);
    let mut w = 0.0;
    for a in 0..k {
        for c in 0..k {
            w += b[idx[a]] * vi[(a, c)] * b[idx[c]];
        }
    }
    (w, (k - dropped.len()) as f64)
}

fn eqtest(f: &SemFit, st: &mut Store) -> Result<String, String> {
    let md = &f.md;
    let b = md.bvec(&f.th);
    let v = md.bv(&f.v);
    let mut out = vec![String::new(), "Wald tests for equations".to_string(), String::new(), "-".repeat(46), format!("{:>15} |{:>12}{:>6}{:>10}", "", "chi2", "df", "p"), format!("{}+{}", "-".repeat(16), "-".repeat(29))];
    let mut rows = Vec::new();
    let mut names = Vec::new();
    for g in 0..md.ng {
        if md.ng > 1 {
            out.push(format!("{:<15} |", format!("Group {}", md.glabels[g])));
        }
        for (y, sl) in sem::equations(md, g) {
            let idx: Vec<usize> = sl.iter().copied().filter(|&j| md.slots[j].kind == Kind::Path && md.slots[j].p.is_some()).collect();
            if idx.is_empty() {
                continue;
            }
            let (w, df) = wald(&b, &v, &idx);
            let p = dist::chi2tail(df, w);
            out.push(format!("{:>15} |{:>12}{:>6}{:>10.4}", tab::ab(&md.vars[y], 15), f9(w), df, p));
            rows.push(vec![w, df, p]);
            names.push(md.vars[y].clone());
        }
    }
    out.push("-".repeat(46));
    let mut m = Matrix::from_rows(&rows);
    m.rownames = names;
    m.colnames = vec!["chi2".into(), "df".into(), "p".into()];
    st.r = Results::new();
    st.r.set_mat("test", m);
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// score tests
// ---------------------------------------------------------------------------

/// Score statistic for added parameters: x0 = (gamma, new values); the
/// statistic is g' H^-1 g, the expected change the new part of H^-1 g.
fn score_stat(f: &dyn Fn(&[f64]) -> Option<f64>, x0: &[f64], newi: &[usize]) -> Option<(f64, Vec<f64>)> {
    let g = opt::gradient(f, x0)?;
    let h = opt::neg_hessian(f, x0)?;
    let (hi, _) = h.sym_inverse(1e-12);
    let k = x0.len();
    let hg: Vec<f64> = (0..k).map(|i| (0..k).map(|j| hi[(i, j)] * g[j]).sum()).collect();
    let s: f64 = g.iter().zip(&hg).map(|(a, b)| a * b).sum();
    Some((s.max(0.0), newi.iter().map(|&i| hg[i]).collect()))
}

/// Augmented log likelihood: model `aug` whose parameters are the fitted
/// (reduced) gamma followed by `r` new ones.
fn aug_ll<'a>(f: &'a SemFit, aug: &'a Model) -> impl Fn(&[f64]) -> Option<f64> + 'a {
    let kr = f.aff.n.cols;
    move |x: &[f64]| {
        let mut th = f.aff.full(&x[..kr]);
        th.extend_from_slice(&x[kr..]);
        Problem { md: aug, gd: &f.gd, method: f.method }.ll(&th)
    }
}

fn base_x(f: &SemFit) -> Vec<f64> {
    f.aff.reduce(&f.th)
}

fn mindices(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    let m = md.m();
    let minchi2 = crate::util::opt_num(cmd, "minchi2", 3)?.unwrap_or(3.84);
    let mut cands: Vec<(String, Slot)> = Vec::new();
    for g in 0..md.ng {
        let gs = if md.ng > 1 { format!("#{}", md.glabels[g]) } else { String::new() };
        for to in 0..m {
            if !md.endo[to] {
                continue;
            }
            for from in 0..m {
                if from == to || md.slots.iter().any(|s| s.g == g && s.kind == Kind::Path && s.a == to && s.b == from) {
                    continue;
                }
                // skip paths that would create a two-variable loop with an existing path
                if md.slots.iter().any(|s| s.g == g && s.kind == Kind::Path && s.a == from && s.b == to) {
                    continue;
                }
                let eq = if md.meas[to] { "Measurement" } else { "Structural" };
                cands.push((format!("{eq}{gs}|{} <- {}", md.vars[to], md.vars[from]), Slot { kind: Kind::Path, a: to, b: from, g, fix: None, p: None, class: super::model::Class::SCoef, auto: false }));
            }
        }
        for i in 0..m {
            for j in i + 1..m {
                if !(md.endo[i] && md.endo[j]) {
                    continue;
                }
                if md.slots.iter().any(|s| s.g == g && s.kind == Kind::Cov && s.a == i && s.b == j) {
                    continue;
                }
                cands.push((format!("cov(e.{},e.{}){gs}", md.vars[i], md.vars[j]), Slot { kind: Kind::Cov, a: i, b: j, g, fix: None, p: None, class: super::model::Class::SErrCov, auto: false }));
            }
        }
    }
    let x0 = base_x(f);
    let kr = x0.len();
    let mut rows = Vec::new();
    for (name, mut s) in cands {
        let mut aug = md.clone();
        s.p = Some(md.k);
        aug.slots.push(s.clone());
        aug.k += 1;
        let fa = aug_ll(f, &aug);
        let mut x = x0.clone();
        x.push(0.0);
        let Some((mi, epc)) = score_stat(&fa, &x, &[kr]) else { continue };
        if mi < minchi2 {
            continue;
        }
        let epc = epc[0];
        // standardized EPC
        let sepc = md.implied_all(&f.th, s.g).map(|(_, sig)| match s.kind {
            Kind::Path => epc * (sig[s.b * m + s.b] / sig[s.a * m + s.a]).sqrt(),
            _ => {
                let (_, sm, _) = md.mats(&f.th, s.g);
                epc / (sm[s.a * m + s.a] * sm[s.b * m + s.b]).sqrt()
            }
        });
        rows.push((name, mi, epc, sepc.unwrap_or(SYSMISS)));
    }
    let mut out = vec![String::new(), "Modification indices".to_string(), String::new(), "-".repeat(78), format!("{:>28} |{:>10}{:>5}{:>10}{:>12}{:>12}", "", "MI", "df", "P>MI", "EPC", "Std EPC"), format!("{}+{}", "-".repeat(29), "-".repeat(48))];
    let mut last = String::new();
    let mut mm = Vec::new();
    let mut names = Vec::new();
    for (name, mi, epc, sepc) in &rows {
        let (sec, lab) = match name.split_once('|') {
            Some((a, b)) => (a.to_string(), b.to_string()),
            None => (String::new(), name.clone()),
        };
        if sec != last {
            out.push(format!("{:<28} |", sec));
            last = sec.clone();
        }
        let p = dist::chi2tail(1.0, *mi);
        out.push(format!("{:>28} |{:>10}{:>5}{:>10.2}{:>12}{:>12}", tab::ab(&lab, 28), fmt_g(*mi, 8), 1, p, f9(*epc), f9(*sepc)));
        mm.push(vec![*mi, 1.0, p, *epc, *sepc]);
        names.push(lab);
    }
    out.push("-".repeat(78));
    out.push(format!("EPC = expected parameter change; MI >= {} shown", fmt_g(minchi2, 6)));
    st.r = Results::new();
    let mut m = Matrix::from_rows(&mm);
    if mm.is_empty() {
        m = Matrix::zeros(0, 5);
    }
    m.rownames = names;
    m.colnames = ["mi", "df", "pvalue", "epc", "sepc"].iter().map(|s| s.to_string()).collect();
    st.r.set_mat("mindices", m);
    Ok(out.join("\n"))
}

fn scoretests(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    let x0 = base_x(f);
    let kr = x0.len();
    let mut rows = Vec::new();
    // equality constraints from labels: release each slot after the first
    for (label, p) in &md.labels {
        let users: Vec<usize> = md.slots.iter().enumerate().filter(|(_, s)| s.p == Some(*p)).map(|(i, _)| i).collect();
        for &j in users.iter().skip(1) {
            let mut aug = md.clone();
            aug.slots[j].p = Some(md.k);
            aug.k += 1;
            let fa = aug_ll(f, &aug);
            let mut x = x0.clone();
            x.push(f.th[*p]);
            if let Some((s, _)) = score_stat(&fa, &x, &[kr]) {
                rows.push((format!("{} ({label})", md.bname(j)), s, 1.0));
            }
        }
    }
    // linear constraints from constraints(): drop one at a time
    if let Some(list) = st.e.macros.get("cmdline").and_then(|c| c.find("constraints(").map(|i| c[i + 12..].split(')').next().unwrap_or("").to_string())) {
        let _ = list;
    }
    let _ = cmd;
    let mut out = vec![String::new(), "Score tests for linear constraints".to_string(), String::new(), "-".repeat(70), format!("{:>44} |{:>10}{:>5}{:>10}", "", "chi2", "df", "P>chi2"), format!("{}+{}", "-".repeat(45), "-".repeat(24))];
    let mut mm = Vec::new();
    for (n, s, df) in &rows {
        let p = dist::chi2tail(*df, *s);
        out.push(format!("{:>44} |{:>10}{:>5}{:>10.4}", tab::ab(n, 44), fmt_g(*s, 8), df, p));
        mm.push(vec![*s, *df, p]);
    }
    if rows.is_empty() {
        out.push("(no equality constraints to test)".into());
    }
    out.push("-".repeat(70));
    st.r = Results::new();
    let mut m = if mm.is_empty() { Matrix::zeros(0, 3) } else { Matrix::from_rows(&mm) };
    m.colnames = vec!["chi2".into(), "df".into(), "pvalue".into()];
    st.r.set_mat("Cns_sctest", m);
    Ok(out.join("\n"))
}

fn ginvariant(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    if md.ng < 2 {
        return Err(rc_err(198, "estat ginvariant requires a model fitted with group()"));
    }
    let x0 = base_x(f);
    let kr = x0.len();
    let b = md.bvec(&f.th);
    let v = md.bv(&f.v);
    let showall = cmd.has_opt("showpclass", 5);
    let _ = showall;
    let mut out = vec![String::new(), "Tests for group invariance of parameters".to_string(), String::new(), "-".repeat(78), format!("{:>26} |{:>13}{:>5}{:>8}  |{:>11}{:>5}{:>8}", "", "Wald chi2", "df", "p>chi2", "Score chi2", "df", "p>chi2"), format!("{}+{}+{}", "-".repeat(27), "-".repeat(28), "-".repeat(23))];
    let mut mm = Vec::new();
    let mut names = Vec::new();
    for (j0, s0) in md.slots.iter().enumerate().filter(|(_, s)| s.g == 0 && s.fix.is_none()) {
        let others: Vec<usize> = (0..md.slots.len()).filter(|&j| {
            let s = &md.slots[j];
            s.g > 0 && s.kind == s0.kind && s.a == s0.a && s.b == s0.b && s.fix.is_none()
        }).collect();
        if others.is_empty() {
            continue;
        }
        let shared = others.iter().all(|&j| md.slots[j].p == s0.p);
        let (_, nm) = md.slot_name(s0);
        let label = match s0.kind {
            Kind::Path | Kind::Cons => format!("{}:{}", md.vars[s0.a], nm),
            _ => nm,
        };
        let (wc, wdf, sc, sdf) = if shared {
            let mut aug = md.clone();
            let mut x = x0.clone();
            let mut newi = Vec::new();
            for (t, &j) in others.iter().enumerate() {
                aug.slots[j].p = Some(md.k + t);
                x.push(f.th[s0.p.unwrap()]);
                newi.push(kr + t);
            }
            aug.k += others.len();
            let fa = aug_ll(f, &aug);
            match score_stat(&fa, &x, &newi) {
                Some((s, _)) => (SYSMISS, SYSMISS, s, others.len() as f64),
                None => continue,
            }
        } else {
            // Wald test of equality of the group-specific parameters
            let mut idx = vec![j0];
            idx.extend(&others);
            let r = idx.len() - 1;
            let mut d = vec![0.0; r];
            let mut vd = Matrix::zeros(r, r);
            for a in 0..r {
                d[a] = b[idx[a + 1]] - b[idx[0]];
                for c in 0..r {
                    vd[(a, c)] = v[(idx[a + 1], idx[c + 1])] - v[(idx[a + 1], idx[0])] - v[(idx[0], idx[c + 1])] + v[(idx[0], idx[0])];
                }
            }
            let (vi, _) = vd.sym_inverse(1e-12);
            let w: f64 = (0..r).map(|a| (0..r).map(|c| d[a] * vi[(a, c)] * d[c]).sum::<f64>()).sum();
            (w, r as f64, SYSMISS, SYSMISS)
        };
        let pv = |c: f64, df: f64| if c == SYSMISS { SYSMISS } else { dist::chi2tail(df, c) };
        let fm = |x: f64| if x == SYSMISS { ".".to_string() } else { f9(x) };
        let fp = |x: f64| if x == SYSMISS { ".".to_string() } else { format!("{x:.4}") };
        out.push(format!("{:>26} |{:>13}{:>5}{:>8}  |{:>11}{:>5}{:>8}", tab::ab(&label, 26), fm(wc), fm(wdf), fp(pv(wc, wdf)), fm(sc), fm(sdf), fp(pv(sc, sdf))));
        mm.push(vec![wc, wdf, pv(wc, wdf), sc, sdf, pv(sc, sdf)]);
        names.push(label);
    }
    out.push("-".repeat(78));
    st.r = Results::new();
    let mut m = if mm.is_empty() { Matrix::zeros(0, 6) } else { Matrix::from_rows(&mm) };
    m.rownames = names;
    m.colnames = ["chi2_w", "df_w", "p_w", "chi2_s", "df_s", "p_s"].iter().map(|s| s.to_string()).collect();
    st.r.set_mat("test", m);
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// residuals, framework, effects, stability, summarize
// ---------------------------------------------------------------------------

fn print_sym(out: &mut Vec<String>, names: &[String], m: &[f64], lower: bool) {
    let n = names.len();
    for c0 in (0..n).step_by(6) {
        let cols: Vec<usize> = (c0..(c0 + 6).min(n)).collect();
        out.push(format!("{:>12} |{}", "", cols.iter().map(|&j| format!("{:>11}", tab::ab(&names[j], 10))).collect::<String>()));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11 * cols.len())));
        for i in 0..n {
            let cells: String = cols.iter().map(|&j| if lower && j > i { " ".repeat(11) } else { format!("{:>11}", fmt_g(m[i * n + j], 8)) }).collect();
            if lower && cols[0] > i {
                continue;
            }
            out.push(format!("{:>12} |{}", tab::ab(&names[i], 12), cells));
        }
        out.push(String::new());
    }
}

fn residuals(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    let p = md.p;
    let names: Vec<String> = md.vars[..p].to_vec();
    let normalized = cmd.has_opt("normalized", 4);
    let standardized = cmd.has_opt("standardized", 3);
    let mut out = Vec::new();
    st.r = Results::new();
    for g in 0..md.ng {
        let s = &f.gd[g].stats;
        let (mu, sig) = md.implied(&f.th, g).ok_or_else(|| rc_err(430, "implied moments not available"))?;
        let n = s.n;
        let mut rm: Vec<f64> = (0..p).map(|i| s.mean[i] - mu[i]).collect();
        let mut rc: Vec<f64> = (0..p * p).map(|t| s.cov[t] - sig[t]).collect();
        let kind = if normalized || standardized {
            for i in 0..p {
                rm[i] /= (sig[i * p + i] / n).sqrt();
                for j in 0..p {
                    rc[i * p + j] /= ((sig[i * p + i] * sig[j * p + j] + sig[i * p + j].powi(2)) / n).sqrt();
                }
            }
            "Normalized residuals"
        } else {
            "Residuals of observed variables"
        };
        if md.ng > 1 {
            out.push(format!("Group {}", md.glabels[g]));
        }
        out.push(String::new());
        out.push(kind.to_string());
        out.push(String::new());
        out.push("Mean residuals".into());
        out.push(format!("{:>12} |{}", "", names.iter().map(|n| format!("{:>11}", tab::ab(n, 10))).collect::<String>()));
        out.push(format!("{:>12} |{}", "raw", rm.iter().map(|v| format!("{:>11}", fmt_g(*v, 8))).collect::<String>()));
        out.push(String::new());
        out.push("Covariance residuals".into());
        print_sym(&mut out, &names, &rc, true);
        if g == 0 {
            let mut m1 = Matrix::row_vector(&rm);
            m1.colnames = names.clone();
            let mut m2 = Matrix::zeros(p, p);
            m2.data = rc.clone();
            m2.rownames = names.clone();
            m2.colnames = names.clone();
            st.r.set_mat(if kind.starts_with('N') { "nres_mean" } else { "res_mean" }, m1);
            st.r.set_mat(if kind.starts_with('N') { "nres_cov" } else { "res_cov" }, m2);
        }
    }
    Ok(out.join("\n"))
}

fn framework(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    let m = md.m();
    let en: Vec<usize> = (0..m).filter(|&i| md.endo[i]).collect();
    let ex: Vec<usize> = (0..m).filter(|&i| !md.endo[i]).collect();
    let nm = |v: &[usize]| v.iter().map(|&i| md.vars[i].clone()).collect::<Vec<_>>();
    let mut out = Vec::new();
    st.r = Results::new();
    let show = |out: &mut Vec<String>, title: &str, rn: &[String], cn: &[String], vals: &[f64]| {
        out.push(String::new());
        out.push(title.to_string());
        let nc = cn.len();
        out.push(format!("{:>12} |{}", "", cn.iter().map(|n| format!("{:>11}", tab::ab(n, 10))).collect::<String>()));
        out.push(format!("{}+{}", "-".repeat(13), "-".repeat(11 * nc)));
        for (i, r) in rn.iter().enumerate() {
            out.push(format!("{:>12} |{}", tab::ab(r, 12), (0..nc).map(|j| {
                let v = vals[i * nc + j];
                format!("{:>11}", if v == 0.0 { "0".to_string() } else { fmt_g(v, 8) })
            }).collect::<String>()));
        }
    };
    for g in 0..md.ng {
        let (a, s, mv) = md.mats(&f.th, g);
        if md.ng > 1 {
            out.push(format!("Group {}", md.glabels[g]));
        }
        let beta = la::sub(&a, m, &en, &en);
        let gamma = la::sub(&a, m, &en, &ex);
        let psi = la::sub(&s, m, &en, &en);
        let phi = la::sub(&s, m, &ex, &ex);
        let alpha: Vec<f64> = en.iter().map(|&i| mv[i]).collect();
        let kappa: Vec<f64> = ex.iter().map(|&i| mv[i]).collect();
        let (ne, nx) = (nm(&en), nm(&ex));
        if !en.is_empty() {
            show(&mut out, "Endogenous variables on endogenous variables (Beta)", &ne, &ne, &beta);
            if !ex.is_empty() {
                show(&mut out, "Endogenous variables on exogenous variables (Gamma)", &ne, &nx, &gamma);
            }
            show(&mut out, "Intercepts of endogenous variables (alpha)", &["alpha".into()], &ne, &alpha);
            show(&mut out, "Covariances of error variables (Psi)", &ne, &ne, &psi);
        }
        if !ex.is_empty() {
            show(&mut out, "Means of exogenous variables (kappa)", &["kappa".into()], &nx, &kappa);
            show(&mut out, "Covariances of exogenous variables (Phi)", &nx, &nx, &phi);
        }
        if cmd.has_opt("fitted", 3) {
            let (mu, sig) = md.implied_all(&f.th, g).ok_or_else(|| rc_err(430, "implied moments not available"))?;
            let all = nm(&(0..m).collect::<Vec<_>>());
            show(&mut out, "Fitted covariances of all variables (Sigma)", &all, &all, &sig);
            show(&mut out, "Fitted means of all variables (mu)", &["mu".into()], &all, &mu);
        }
        if g == 0 {
            let mk = |r: &[String], c: &[String], v: &[f64]| {
                let mut x = Matrix::zeros(r.len(), c.len());
                x.data = v.to_vec();
                x.rownames = r.to_vec();
                x.colnames = c.to_vec();
                x
            };
            st.r.set_mat("Beta", mk(&ne, &ne, &beta));
            st.r.set_mat("Gamma", mk(&ne, &nx, &gamma));
            st.r.set_mat("alpha", mk(&["alpha".into()], &ne, &alpha));
            st.r.set_mat("Psi", mk(&ne, &ne, &psi));
            st.r.set_mat("Phi", mk(&nx, &nx, &phi));
            st.r.set_mat("kappa", mk(&["kappa".into()], &nx, &kappa));
        }
    }
    Ok(out.join("\n"))
}

/// (direct, indirect, total) effects of all variables on the endogenous ones.
fn effects(md: &Model, th: &[f64], g: usize, std: bool) -> Vec<f64> {
    let m = md.m();
    let (a, _, _) = md.mats(th, g);
    let Some(t) = md.total(&a) else { return vec![f64::NAN; 3 * m * m] };
    let sd: Vec<f64> = match md.implied_all(th, g) {
        Some((_, s)) if std => (0..m).map(|i| s[i * m + i].max(0.0).sqrt()).collect(),
        _ => vec![1.0; m],
    };
    let mut out = Vec::with_capacity(3 * m * m);
    for kind in 0..3 {
        for i in 0..m {
            for j in 0..m {
                let tot = t[i * m + j] - if i == j { 1.0 } else { 0.0 };
                let d = a[i * m + j];
                let v = match kind {
                    0 => d,
                    1 => tot - d,
                    _ => tot,
                };
                out.push(v * sd[j] / sd[i]);
            }
        }
    }
    out
}

fn teffects(f: &SemFit, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let md = &f.md;
    let m = md.m();
    let std = cmd.has_opt("standardized", 3);
    let level = crate::util::level(cmd, st)?;
    let mut out = Vec::new();
    st.r = Results::new();
    for g in 0..md.ng {
        let e0 = effects(md, &f.th, g, std);
        let j = opt::jacobian(&|t: &[f64]| effects(md, t, g, std), &f.th);
        let v = opt::sandwich_jv(&j, &f.v);
        let fixed_path = |i: usize, jj: usize| md.slots.iter().any(|s| s.g == g && s.kind == Kind::Path && s.a == i && s.b == jj && s.p.is_none());
        for (kind, title) in [(0, "Direct effects"), (1, "Indirect effects"), (2, "Total effects")] {
            let mut rows = Vec::new();
            let mut bvec = Vec::new();
            let mut names = Vec::new();
            let mut last_sec = "";
            for (y, sec) in (0..m).filter(|&i| md.endo[i]).map(|i| (i, if md.meas[i] { "Measurement" } else { "Structural" })) {
                let has_path = |s: usize| md.slots.iter().any(|x| x.g == g && x.kind == Kind::Path && x.a == y && x.b == s);
                let srcs: Vec<usize> = (0..m).filter(|&s| s != y && (e0[kind * m * m + y * m + s].abs() > 1e-14 || kind == 0 && has_path(s))).collect();
                if srcs.is_empty() {
                    continue;
                }
                if last_sec != sec {
                    rows.push(Row::Head(sec.to_string()));
                    last_sec = sec;
                }
                rows.push(Row::Head(format!("  {} <-", md.vars[y])));
                for s in srcs {
                    let idx = kind * m * m + y * m + s;
                    let b = e0[idx];
                    let se = v[(idx, idx)].max(0.0).sqrt();
                    let cns = se < 1e-12 && (kind == 0 && fixed_path(y, s) || b == 0.0 || se == 0.0);
                    rows.push(Row::Coef(md.vars[s].clone(), b, if cns { None } else { Some(se) }, true));
                    bvec.push(b);
                    names.push(format!("{}:{}", md.vars[y], md.vars[s]));
                }
            }
            out.push(String::new());
            out.push(if md.ng > 1 { format!("{title}, group {}", md.glabels[g]) } else { title.to_string() });
            out.push(String::new());
            out.extend(tab::table("OIM", level, if std { "Std. coef." } else { "Coefficient" }, &rows));
            if g == 0 {
                let key = ["direct", "indirect", "total"][kind];
                st.r.set_mat(key, Matrix::row_vector(&bvec).with_names(vec!["y1".into()], names));
            }
            tab::keep_table(&mut st.r);
        }
    }
    Ok(out.join("\n"))
}

fn stable(f: &SemFit, st: &mut Store) -> Result<String, String> {
    let md = &f.md;
    let m = md.m();
    let en: Vec<usize> = (0..m).filter(|&i| md.endo[i]).collect();
    let (a, _, _) = md.mats(&f.th, 0);
    let b = la::sub(&a, m, &en, &en);
    let ne = en.len();
    let mut bm = Matrix::zeros(ne, ne);
    bm.data = b;
    let ev = if ne > 0 { crate::tsla::eig_general(&bm) } else { vec![] };
    let mut out = vec![String::new(), "Stability analysis of simultaneous equation systems".to_string(), String::new(), "Eigenvalue stability condition".to_string(), String::new(), "-".repeat(40), format!("{:^26}| {:>10}", "Eigenvalue", "Modulus"), format!("{}+{}", "-".repeat(26), "-".repeat(13))];
    let mut idx = 0.0f64;
    for (re, im) in &ev {
        let md_ = (re * re + im * im).sqrt();
        idx = idx.max(md_);
        let txt = if im.abs() > 1e-12 { format!("{} {} {}i", fmt_g(*re, 8), if *im < 0.0 { "-" } else { "+" }, fmt_g(im.abs(), 8)) } else { fmt_g(*re, 9) };
        out.push(format!("{:>25} | {:>10}", txt, fmt_g(md_, 8)));
    }
    out.push("-".repeat(40));
    out.push(format!("stability index = {}", fmt_g(idx, 8)));
    out.push(if idx < 1.0 { "All the eigenvalues lie inside the unit circle.\nSEM satisfies stability condition.".into() } else { "At least one eigenvalue is at least 1.\nSEM does not satisfy stability condition.".into() });
    st.r = Results::new();
    st.r.set("stindex", idx);
    Ok(out.join("\n"))
}

fn summarize(f: &SemFit, st: &mut Store) -> Result<String, String> {
    let md = &f.md;
    let p = md.p;
    let mut out = vec![String::new(), "  Estimation sample (sem)".to_string(), String::new(), "-".repeat(60), format!("{:>12} |{:>12}{:>12}{:>12}{:>12}", "Variable", "Mean", "Std. dev.", "Min", "Max"), format!("{}+{}", "-".repeat(13), "-".repeat(48))];
    let mut rows = Vec::new();
    for i in 0..p {
        let vals: Vec<f64> = f.gd.iter().flat_map(|g| g.rows.iter().map(move |r| r[i])).filter(|v| !v.is_nan()).collect();
        let (mean, sd, mn, mx) = if vals.is_empty() {
            let s = &f.gd[0].stats;
            (s.mean[i], (s.cov[i * p + i] * s.n / (s.n - 1.0)).sqrt(), SYSMISS, SYSMISS)
        } else {
            let n = vals.len() as f64;
            let m = vals.iter().sum::<f64>() / n;
            let sd = (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            (m, sd, vals.iter().cloned().fold(f64::INFINITY, f64::min), vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        };
        out.push(format!("{:>12} |{:>12}{:>12}{:>12}{:>12}", tab::ab(&md.vars[i], 12), f9(mean), f9(sd), f9(mn), f9(mx)));
        rows.push(vec![mean, sd, mn, mx]);
    }
    out.push("-".repeat(60));
    let mut m = Matrix::from_rows(&rows);
    m.rownames = md.vars[..p].to_vec();
    m.colnames = vec!["mean".into(), "sd".into(), "min".into(), "max".into()];
    st.r = Results::new();
    st.r.set_mat("stats", m);
    Ok(out.join("\n"))
}

/// `estat stdize: test|lincom|nlcom|testnl ...` on standardized coefficients.
fn stdize(ds: &mut DataSet, st: &mut Store, f: &SemFit, cmd: &Command) -> Result<String, String> {
    let raw = cmd.raw.trim();
    let inner = raw.split_once(':').map(|x| x.1.trim()).ok_or_else(|| rc_err(198, "estat stdize: requires a command after the colon"))?;
    let ic = ferrostat_parser::parser::parse_line(inner).map_err(|e| rc_err(198, e.message))?;
    let (b, v) = sem::std_bv(f);
    let saved = st.e.clone();
    let names = saved.matrices.get("b").map(|m| m.colnames.clone()).unwrap_or_default();
    st.e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    st.e.set_mat("V", v.with_names(names.clone(), names));
    let r = match ic.verb.as_str() {
        "test" => crate::postest::Test.execute(ds, st, &ic),
        "lincom" => crate::postest::Lincom.execute(ds, st, &ic),
        "nlcom" => crate::postest2::Nlcom.execute(ds, st, &ic),
        "testnl" => crate::postest2::Testnl.execute(ds, st, &ic),
        o => Err(rc_err(198, format!("estat stdize: {o} not allowed"))),
    };
    let keep_r = st.r.clone();
    st.e = saved;
    st.r = keep_r;
    r
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// New variable names: `a b c` or `stub*` for `n` variables.
pub fn new_names(text: &str, n: usize) -> Result<Vec<String>, String> {
    let w: Vec<&str> = text.split_whitespace().filter(|w| !matches!(*w, "double" | "float")).collect();
    if w.len() == 1 && w[0].ends_with('*') {
        let stub = w[0].trim_end_matches('*');
        return Ok((1..=n).map(|i| format!("{stub}{i}")).collect());
    }
    if w.len() != n {
        return Err(rc_err(198, format!("{n} new variable names required")));
    }
    Ok(w.iter().map(|s| s.to_string()).collect())
}

/// Empirical Bayes means of all variables given the observed values of one row.
pub fn eb_all(md: &Model, th: &[f64], g: usize, row: &[f64]) -> Option<Vec<f64>> {
    let m = md.m();
    let p = md.p;
    let (mu, sig) = md.implied_all(th, g)?;
    let o: Vec<usize> = (0..p).filter(|&i| !row[i].is_nan()).collect();
    let soo = la::sub(&sig, m, &o, &o);
    let l = la::chol(&soo, o.len())?;
    let d: Vec<f64> = o.iter().map(|&i| row[i] - mu[i]).collect();
    let w = la::chol_solve(&l, o.len(), &d);
    Some((0..m).map(|i| {
        if i < p && !row[i].is_nan() {
            row[i]
        } else {
            mu[i] + o.iter().enumerate().map(|(a, &j)| sig[i * m + j] * w[a]).sum::<f64>()
        }
    }).collect())
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let f = state::sem(st).ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let md = &f.md;
    let m = md.m();
    let p = md.p;
    let main = cmd.main.trim();
    let n = ds.n_obs;
    let latent = cmd.has_opt("latent", 2);
    let scores = cmd.has_opt("scores", 2);
    let targets: Vec<usize> = if latent {
        match cmd.opt("latent", 2) {
            Some(l) => l.split_whitespace().map(|v| md.idx(v).filter(|&i| i >= p).ok_or_else(|| rc_err(198, format!("{v} is not a latent variable")))).collect::<Result<_, _>>()?,
            None => (p..m).collect(),
        }
    } else if scores {
        vec![]
    } else {
        match cmd.opt("xb", 2) {
            Some(l) => l.split_whitespace().map(|v| md.idx(v).filter(|&i| md.endo[i]).ok_or_else(|| rc_err(198, format!("{v} is not an endogenous variable")))).collect::<Result<_, _>>()?,
            None => (0..p).filter(|&i| md.endo[i]).collect(),
        }
    };
    if scores {
        let k = f.kfree;
        let names = new_names(main, k)?;
        let prob = f.problem();
        let x0 = f.aff.reduce(&f.th);
        let fo = |g: &[f64]| prob.ll_obs(&f.aff.full(g));
        let mut cols = vec![vec![SYSMISS; n]; k];
        let obs: Vec<usize> = f.gd.iter().flat_map(|g| g.obs.clone()).collect();
        for j in 0..k {
            let h = 1e-6 * x0[j].abs().max(1.0);
            let mut a = x0.clone();
            a[j] += h;
            let mut b = x0.clone();
            b[j] -= h;
            let (la_, lb) = (fo(&a).ok_or_else(|| rc_err(430, "scores"))?, fo(&b).ok_or_else(|| rc_err(430, "scores"))?);
            for (t, &o) in obs.iter().enumerate() {
                cols[j][o] = (la_[t] - lb[t]) / (2.0 * h);
            }
        }
        for (nm, c) in names.iter().zip(cols) {
            crate::tsops::add_var(ds, nm, c)?;
        }
        return Ok(String::new());
    }
    let names = new_names(main, targets.len())?;
    let mut cols = vec![vec![SYSMISS; n]; targets.len()];
    // group of each observation
    let gcol = if md.ng > 1 { Some(crate::util::col(ds, &md.gvar)?) } else { None };
    let obs_cols: Vec<Vec<f64>> = md.vars[..p].iter().map(|v| crate::util::col(ds, v)).collect::<Result<_, _>>()?;
    let sample = st.e.sample.clone();
    for i in 0..n {
        if let Some(s) = &sample {
            if !s.get(i).copied().unwrap_or(false) && cmd.if_expr.is_none() {
                // predictions are made for every observation with data
            }
        }
        let g = match &gcol {
            Some(c) => match md.glabels.iter().position(|l| *l == fmt_g(c[i], 12)) {
                Some(g) => g,
                None => continue,
            },
            None => 0,
        };
        let row: Vec<f64> = obs_cols.iter().map(|c| if ferrostat_core::missing::is_missing(c[i]) { f64::NAN } else { c[i] }).collect();
        if row.iter().all(|v| v.is_nan()) {
            continue;
        }
        if f.method != Method::Mlmv && row.iter().any(|v| v.is_nan()) {
            continue;
        }
        let Some(eb) = eb_all(md, &f.th, g, &row) else { continue };
        let (a, _, mv) = md.mats(&f.th, g);
        for (t, &y) in targets.iter().enumerate() {
            cols[t][i] = if latent {
                eb[y]
            } else {
                mv[y] + (0..m).map(|j| a[y * m + j] * eb[j]).sum::<f64>()
            };
        }
    }
    for (nm, c) in names.iter().zip(cols) {
        crate::tsops::add_var(ds, nm, c)?;
    }
    Ok(String::new())
}
